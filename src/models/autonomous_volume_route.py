"""Fit-only q80 routing among the frozen C1_B/C2_B Classification experts."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd

from src.models.classification_models import build_classifier, _array_digest
from src.models.classification_round2_models import stream_missing_mask


THRESHOLD = 0.50
VOLUME_QUANTILE = 0.80
MIN_PRESENT_EXPERT_ROWS = 200
MIN_PRESENT_CLASS_ROWS = 50


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                         separators=(",", ":")).encode("utf-8", errors="surrogatepass")
    return hashlib.sha256(payload).hexdigest()


def _positions_sha256(positions: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(positions, dtype="<i8").tobytes(order="C")).hexdigest()


class VolumeRouteClassifier:
    """Compose the original C4 route with a present-stream specialist above q80.

    ``fit`` learns q80 from its own input rows only, fits the unchanged C2_B on
    all input rows and present rows, and fits the unchanged C1_B on all input
    rows. Predictions route missing streams to C1_B, non-high-volume present
    streams to all-row C2_B, and present streams strictly above the fit-only
    q80 threshold to the present-only C2_B expert. Equal-to-q80 values remain
    on all-row C2_B. The decision threshold is fixed at 0.50.
    """

    def __init__(self):
        self.input_feature_names = list(build_classifier({"id": "C2_B"}).input_feature_names)
        self._experts: dict[str, Any] = {}
        self._q80: float | None = None
        self._fitted = False
        self._state: dict[str, Any] = {}

    @staticmethod
    def _validate_frame(X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if X.columns.has_duplicates:
            raise ValueError("X has duplicate feature columns")
        forbidden = {"artifact_signal", "threat_rank"}.intersection(X.columns)
        if forbidden:
            raise ValueError(f"Target columns are forbidden in X: {sorted(forbidden)}")
        missing = {"signal_stream", "text_load"}.difference(X.columns)
        if missing:
            raise ValueError(f"X is missing routing features: {sorted(missing)}")

    @staticmethod
    def _volume_values(X: pd.DataFrame, *, fit: bool) -> np.ndarray:
        values = pd.to_numeric(X["text_load"], errors="coerce").to_numpy(dtype=float)
        if fit and (not np.isfinite(values).all()):
            raise ValueError("text_load must be finite for the fit-only q80 routing threshold")
        return values

    def fit(self, X: pd.DataFrame, y: Any) -> "VolumeRouteClassifier":
        self._validate_frame(X)
        target = np.asarray(y)
        if target.ndim != 1 or len(target) != len(X) or len(X) == 0:
            raise ValueError("y must be nonempty, one-dimensional, and aligned with X")
        try:
            target_float = target.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError("artifact_signal must contain binary values 0/1") from exc
        if not np.isfinite(target_float).all() or not np.isin(target_float, (0.0, 1.0)).all():
            raise ValueError("artifact_signal must contain finite binary values 0/1")
        target = target_float.astype(np.int8)

        volume = self._volume_values(X, fit=True)
        self._q80 = float(np.quantile(volume, VOLUME_QUANTILE))
        missing = stream_missing_mask(X)
        present_positions = np.flatnonzero(~missing)
        present_counts = np.bincount(target[present_positions], minlength=2)
        if len(present_positions) < MIN_PRESENT_EXPERT_ROWS or int(present_counts.min()) < MIN_PRESENT_CLASS_ROWS:
            raise ValueError(
                f"Insufficient present-only expert support: n={len(present_positions)}, "
                f"class_counts={present_counts.astype(int).tolist()}"
            )

        models = {
            "C2_B_all": ("C2_B", np.arange(len(X), dtype=np.int64)),
            "C2_B_present": ("C2_B", present_positions),
            "C1_B_all": ("C1_B", np.arange(len(X), dtype=np.int64)),
        }
        self._experts = {}
        expert_audits: dict[str, Any] = {}
        for name, (model_id, positions) in models.items():
            expert = build_classifier({"id": model_id})
            fields = list(expert.input_feature_names)
            expert.fit(X.iloc[positions].loc[:, fields], target[positions])
            self._experts[name] = expert
            expert_audits[name] = {
                "model_id": model_id,
                "fit_rows": int(len(positions)),
                "fit_class_counts": np.bincount(target[positions], minlength=2).astype(int).tolist(),
                "fit_positions_sha256": _positions_sha256(positions),
                "expert_state": expert.audit_state(),
            }
        self._state = {
            "recipe_id": "present_q80_route",
            "routing": {
                "missing": "C1_B_all",
                "present_text_load_le_q80": "C2_B_all",
                "present_text_load_gt_q80": "C2_B_present",
            },
            "availability_rule": "null, empty, or whitespace-only signal_stream is missing",
            "volume_threshold": {
                "feature": "text_load",
                "quantile": VOLUME_QUANTILE,
                "value": self._q80,
                "fit_rows": int(len(X)),
                "fit_text_load_sha256": _array_digest(volume),
                "definition": "np.quantile over rows supplied to fit; validation inputs do not update it; route strictly above the cutoff",
            },
            "classification_threshold": THRESHOLD,
            "minimum_present_expert_support": {
                "rows": MIN_PRESENT_EXPERT_ROWS,
                "per_class": MIN_PRESENT_CLASS_ROWS,
            },
            "experts": expert_audits,
        }
        self._state["state_sha256"] = _json_sha256(self._state)
        self._fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted or self._q80 is None:
            raise RuntimeError("Classifier must be fitted before predict_proba")
        self._validate_frame(X)
        missing = stream_missing_mask(X)
        volume = self._volume_values(X, fit=False)
        high_present = (~missing) & np.isfinite(volume) & (volume > self._q80)
        low_present = (~missing) & ~high_present
        masks = {
            "C1_B_all": missing,
            "C2_B_present": high_present,
            "C2_B_all": low_present,
        }
        probabilities = np.empty(len(X), dtype=float)
        before = self.audit_state()
        for name, mask in masks.items():
            if not mask.any():
                continue
            expert = self._experts[name]
            fields = list(expert.input_feature_names)
            branch_probability = np.asarray(expert.predict_proba(X.loc[mask, fields]), dtype=float)
            if branch_probability.shape != (int(mask.sum()),):
                raise RuntimeError(f"{name} returned a misaligned probability vector")
            probabilities[mask] = branch_probability
        if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
            raise RuntimeError("Classifier returned probabilities outside [0, 1]")
        if self.audit_state() != before:
            raise RuntimeError("Prediction changed the fitted model audit state")
        return probabilities

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X) >= THRESHOLD).astype(np.int8)

    def audit_state(self) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("Classifier must be fitted before audit_state")
        return json.loads(json.dumps(self._state, sort_keys=True, allow_nan=False))


def build_volume_route_classifier() -> VolumeRouteClassifier:
    """Build the fit/predict API for the frozen present-stream q80 route."""
    return VolumeRouteClassifier()


def build_model() -> VolumeRouteClassifier:
    """Backward-friendly alias used by the frozen volume-route protocol."""
    return build_volume_route_classifier()


__all__ = ["VolumeRouteClassifier", "build_volume_route_classifier", "build_model", "THRESHOLD"]
