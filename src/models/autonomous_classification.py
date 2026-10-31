"""Availability-specialized variants of the frozen TP2 C4 classifier.

The three research candidates keep the Round 1 C1_B/C2_B expert recipes,
feature allowlists, preprocessing, hyperparameters, and the fixed 0.50 decision
threshold. They change only which fit rows are supplied to each expert.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd

from src.models.classification_models import build_classifier
from src.models.classification_round2_models import stream_missing_mask


_CANDIDATES = {
    "present_specialist": {"present": "present_only", "missing": "all_train"},
    "missing_specialist": {"present": "all_train", "missing": "missing_only"},
    "both_specialist": {"present": "present_only", "missing": "missing_only"},
}
_EXPERTS = {"present": "C2_B", "missing": "C1_B"}
_MIN_FIT_ROWS = 200
_MIN_CLASS_ROWS = 50
THRESHOLD = 0.50


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                         separators=(",", ":")).encode("utf-8", errors="surrogatepass")
    return hashlib.sha256(payload).hexdigest()


def _index_sha256(positions: np.ndarray) -> str:
    positions = np.asarray(positions, dtype="<i8")
    return hashlib.sha256(positions.tobytes(order="C")).hexdigest()


class AvailabilitySpecialistClassifier:
    """Fit and route the unchanged C1_B/C2_B experts using fixed availability.

    Parameters
    ----------
    candidate_id:
        One of ``present_specialist``, ``missing_specialist``, or
        ``both_specialist``. Availability follows the official Round 2 rule:
        null, empty, and whitespace-only ``signal_stream`` values are missing.

    ``fit`` accepts feature columns only. It rejects target columns so callers
    cannot accidentally pass ``artifact_signal`` through the feature frame.
    The experts keep their own fit-only preprocessing and expose their original
    audit states. No feature, threshold, or hyperparameter search happens here.
    """

    def __init__(self, candidate_id: str):
        if candidate_id not in _CANDIDATES:
            raise ValueError(f"Unknown availability candidate: {candidate_id!r}")
        self.candidate_id = candidate_id
        self.input_feature_names = list(build_classifier({"id": "C2_B"}).input_feature_names)
        self._fit_modes = dict(_CANDIDATES[candidate_id])
        self._experts: dict[str, Any] = {}
        self._state: dict[str, Any] = {}
        self._fitted = False

    @staticmethod
    def _validate_frame(X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if X.columns.has_duplicates:
            raise ValueError("X has duplicate feature columns")
        forbidden = {"artifact_signal", "threat_rank"}.intersection(X.columns)
        if forbidden:
            raise ValueError(f"Target columns are forbidden in X: {sorted(forbidden)}")

    @staticmethod
    def _validate_target(y: Any, rows: int) -> np.ndarray:
        target = np.asarray(y)
        if target.ndim != 1 or target.shape[0] != rows or rows == 0:
            raise ValueError("y must be nonempty, one-dimensional, and aligned with X")
        try:
            values = target.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError("artifact_signal must contain binary values 0/1") from exc
        if not np.isfinite(values).all() or not np.isin(values, (0.0, 1.0)).all():
            raise ValueError("artifact_signal must contain finite binary values 0/1")
        return values.astype(np.int8)

    def fit(self, X: pd.DataFrame, y: Any) -> "AvailabilitySpecialistClassifier":
        self._validate_frame(X)
        selected_target = self._validate_target(y, len(X))
        missing = stream_missing_mask(X)
        self._experts = {}
        branch_states: dict[str, Any] = {}

        for branch, model_id in _EXPERTS.items():
            mode = self._fit_modes[branch]
            mask = np.ones(len(X), dtype=bool)
            if mode == "present_only":
                mask = ~missing
            elif mode == "missing_only":
                mask = missing
            positions = np.flatnonzero(mask)
            branch_target = selected_target[mask]
            counts = np.bincount(branch_target, minlength=2)
            if len(positions) < _MIN_FIT_ROWS or int(counts.min()) < _MIN_CLASS_ROWS:
                raise ValueError(
                    f"Insufficient {branch} fit support: n={len(positions)}, "
                    f"class_counts={counts.tolist()}, required n>={_MIN_FIT_ROWS} "
                    f"and each class>={_MIN_CLASS_ROWS}"
                )
            expert = build_classifier({"id": model_id})
            fields = list(expert.input_feature_names)
            expert.fit(X.iloc[positions].loc[:, fields], branch_target)
            audit = expert.audit_state()
            self._experts[branch] = expert
            branch_states[branch] = {
                "model_id": model_id,
                "fit_mode": mode,
                "fit_rows": int(len(positions)),
                "class_counts": counts.astype(int).tolist(),
                "fit_positions_sha256": _index_sha256(positions),
                "expert_state": audit,
            }

        state = {
            "candidate_id": self.candidate_id,
            "routing": {"present": "C2_B", "missing": "C1_B"},
            "fit_modes": dict(self._fit_modes),
            "stream_missing_rule": "null, empty, or whitespace-only signal_stream",
            "threshold": THRESHOLD,
            "minimum_fit_support": {"rows": _MIN_FIT_ROWS, "per_class": _MIN_CLASS_ROWS},
            "branches": branch_states,
        }
        state["state_sha256"] = _json_sha256(state)
        self._state = state
        self._fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Classifier must be fitted before predict_proba")
        self._validate_frame(X)
        missing = stream_missing_mask(X)
        probabilities = np.empty(len(X), dtype=float)
        before = self.audit_state()
        for branch, mask in (("missing", missing), ("present", ~missing)):
            if not mask.any():
                continue
            expert = self._experts[branch]
            fields = list(expert.input_feature_names)
            branch_p = np.asarray(expert.predict_proba(X.loc[mask, fields]), dtype=float)
            if branch_p.shape != (int(mask.sum()),):
                raise RuntimeError(f"{branch} expert returned a misaligned probability vector")
            probabilities[mask] = branch_p
        if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
            raise RuntimeError("Classifier returned non-finite probabilities outside [0, 1]")
        if self.audit_state() != before:
            raise RuntimeError("Prediction changed the fitted model audit state")
        return probabilities

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X) >= THRESHOLD).astype(np.int8)

    def audit_state(self) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("Classifier must be fitted before audit_state")
        return json.loads(json.dumps(self._state, sort_keys=True, allow_nan=False))


def build_autonomous_classifier(candidate_id: str) -> AvailabilitySpecialistClassifier:
    """Build one fixed availability-specialized candidate without fitting."""
    return AvailabilitySpecialistClassifier(candidate_id)


__all__ = [
    "AvailabilitySpecialistClassifier",
    "build_autonomous_classifier",
    "THRESHOLD",
]
