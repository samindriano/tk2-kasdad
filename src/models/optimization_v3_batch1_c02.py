"""C02 present-only sparse expert with the frozen V3 blend policy.

The classifier and its preprocessing are inherited unchanged from the accepted
sparse-binary source.  This adapter selects nonempty-stream fit rows before
calling that implementation, so every learned transform and the estimator see
the same present-only population.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd

from src.models.classification_round2_models import stream_missing_mask
from src.models.optimization_v3_sparse_classification import SparseTextBoostClassifier


FEATURE_NAMES = tuple(SparseTextBoostClassifier.input_feature_names)
MISSING_RAW_SENTINEL = 0.5
BLEND_C2B_WEIGHT = 0.75
BLEND_EXPERT_WEIGHT = 0.25
DECISION_THRESHOLD = 0.5


def _index_sha256(values: np.ndarray) -> str:
    indices = np.asarray(values, dtype="<i8")
    return hashlib.sha256(indices.tobytes(order="C")).hexdigest()


def _mask_sha256(values: np.ndarray) -> str:
    mask = np.asarray(values, dtype=np.uint8)
    return hashlib.sha256(mask.tobytes(order="C")).hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class PresentOnlySparseClassifier:
    """Fit the existing sparse model only on fit rows with a present stream."""

    def __init__(self) -> None:
        self._model: SparseTextBoostClassifier | None = None
        self._state: dict[str, Any] | None = None
        self._fit_diagnostics: dict[str, Any] = {}

    def fit(self, features: pd.DataFrame, target: Any) -> "PresentOnlySparseClassifier":
        if features.columns.has_duplicates or set(features.columns) != set(FEATURE_NAMES):
            raise ValueError("C02 requires the exact structured + signal_stream allowlist")
        selected = features.loc[:, list(FEATURE_NAMES)].copy()
        y = np.asarray(target)
        if y.shape != (len(selected),) or not np.isin(y, [0, 1]).all():
            raise ValueError("C02 requires a row-aligned binary fit target")

        missing = stream_missing_mask(selected)
        present_positions = np.flatnonzero(~missing)
        if not len(present_positions):
            raise ValueError("C02 has no present-stream fit rows")
        y_present = y[present_positions]
        class_counts = {
            "class_0": int(np.sum(y_present == 0)),
            "class_1": int(np.sum(y_present == 1)),
        }
        if min(class_counts.values()) == 0:
            raise ValueError("C02 present-stream fit population must contain both classes")

        # SparseTextBoostClassifier is the exact frozen source implementation;
        # all preprocessing is fit only after this present-row selection.
        self._model = SparseTextBoostClassifier().fit(
            selected.iloc[present_positions].copy(), y_present.copy()
        )
        base_state = self._model.audit_state()
        if int(base_state.get("fit_rows", -1)) != len(present_positions):
            raise RuntimeError("C02 base model fit-row count differs from present support")
        self._fit_diagnostics = {
            "fit_rows_total": int(len(selected)),
            "fit_rows_present": int(len(present_positions)),
            "fit_rows_missing": int(missing.sum()),
            "present_class_counts": class_counts,
            "present_fit_positions_sha256": _index_sha256(present_positions),
            "fit_missing_mask_sha256": _mask_sha256(missing),
        }
        state = dict(base_state)
        state.update({
            "id": "C02_PRESENT_ONLY_SPARSE_LGB_BINARY",
            "source_model_id": base_state["id"],
            "fit_population_rule": "fit only rows where signal_stream is non-null and nonblank after strip",
            "fit_population": self._fit_diagnostics,
            "fit_transforms_population": "present_stream_fit_rows_only",
            "missing_stream_indicator": {
                "included_in_frozen_design": True,
                "fit_population_value": 0,
                "fit_missing_rows": 0,
                "constant_zero": True,
            },
            "fit_population_change_is_not_claimed_as_leakage": True,
            "validation_data_passed_to_fit": False,
            "test_data_used": False,
        })
        state["state_sha256"] = _json_sha256({k: v for k, v in state.items() if k != "state_sha256"})
        self._state = state
        return self

    def predict_present(self, features: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("C02 model must be fitted before prediction")
        if features.columns.has_duplicates or set(features.columns) != set(FEATURE_NAMES):
            raise ValueError("C02 requires the exact structured + signal_stream allowlist")
        selected = features.loc[:, list(FEATURE_NAMES)].copy()
        missing = stream_missing_mask(selected)
        if missing.any():
            raise ValueError("C02 expert prediction accepts present-stream rows only")
        probability = np.asarray(self._model.predict_proba(selected), dtype=np.float64)
        if probability.shape != (len(selected),) or not np.isfinite(probability).all():
            raise RuntimeError("C02 sparse expert produced invalid probabilities")
        if np.any((probability < 0.0) | (probability > 1.0)):
            raise RuntimeError("C02 sparse expert probability outside [0,1]")
        return probability

    def audit_state(self) -> dict[str, Any]:
        if self._state is None:
            raise RuntimeError("C02 model must be fitted before audit")
        return json.loads(json.dumps(self._state, sort_keys=True, allow_nan=False))


def fit_predict(
    fit_features: pd.DataFrame,
    fit_target: np.ndarray,
    score_features: pd.DataFrame,
    *,
    controls: dict[str, np.ndarray],
    card_id: str,
    cutoff: int,
) -> dict[str, Any]:
    if card_id != "C02":
        raise ValueError(f"C02 adapter cannot execute {card_id}")
    if score_features.columns.has_duplicates or set(score_features.columns) != set(FEATURE_NAMES):
        raise ValueError("C02 requires the exact structured + signal_stream allowlist")

    missing = np.asarray(controls["stream_missing"], dtype=bool)
    c2b = np.asarray(controls["c2b_probability"], dtype=np.float64)
    cb = np.asarray(controls["cb_probability"], dtype=np.float64)
    old_sparse = np.asarray(controls["baseline_sparse_raw"], dtype=np.float64)
    baseline = np.asarray(controls["baseline_prediction"], dtype=np.float64)
    if not (missing.shape == c2b.shape == cb.shape == old_sparse.shape == baseline.shape == (len(score_features),)):
        raise ValueError("C02 controls are not aligned to score rows")
    for name, values in (("C2_B", c2b), ("CB", cb), ("sparse25", old_sparse), ("baseline", baseline)):
        if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
            raise ValueError(f"C02 {name} control must be finite probabilities in [0,1]")
    expected_baseline = np.where(
        missing, cb, BLEND_C2B_WEIGHT * c2b + BLEND_EXPERT_WEIGHT * old_sparse
    )
    if not np.array_equal(baseline, expected_baseline):
        raise ValueError("C02 frozen baseline does not reconstruct from C2_B/sparse25/CB controls")

    model = PresentOnlySparseClassifier().fit(fit_features, fit_target)
    score_missing = stream_missing_mask(score_features)
    if not np.array_equal(score_missing, missing):
        raise ValueError("C02 stream route mask differs from frozen control route")
    present_positions = np.flatnonzero(~missing)
    raw = np.full(len(score_features), MISSING_RAW_SENTINEL, dtype=np.float64)
    if len(present_positions):
        raw[present_positions] = model.predict_present(score_features.iloc[present_positions])
    prediction = np.where(
        missing, cb, BLEND_C2B_WEIGHT * c2b + BLEND_EXPERT_WEIGHT * raw
    )
    if prediction.shape != (len(score_features),) or not np.isfinite(prediction).all():
        raise RuntimeError("C02 final policy prediction is invalid")
    if np.any((prediction < 0.0) | (prediction > 1.0)):
        raise RuntimeError("C02 final policy probability outside [0,1]")

    fit_metadata = {
        **model._fit_diagnostics,
        "cutoff": int(cutoff),
        "score_rows": int(len(score_features)),
        "score_rows_present": int((~missing).sum()),
        "score_rows_missing": int(missing.sum()),
        "expert_applied_rows": int((~missing).sum()),
        "missing_raw_candidate_sentinel": MISSING_RAW_SENTINEL,
        "sentinel_is_ignored_by_policy_when_missing": True,
        "blend": {"c2b_weight": BLEND_C2B_WEIGHT, "expert_weight": BLEND_EXPERT_WEIGHT},
        "missing_route": "frozen CatBoost probability",
        "threshold": DECISION_THRESHOLD,
    }
    return {
        "prediction": prediction,
        "raw_candidate": raw,
        "fit_predictor_count": 1,
        "state": model.audit_state(),
        "fit_metadata": fit_metadata,
        "audit_arrays": {
            "stream_missing": missing,
            "expert_applied": ~missing,
        },
    }


__all__ = [
    "DECISION_THRESHOLD", "FEATURE_NAMES", "MISSING_RAW_SENTINEL",
    "PresentOnlySparseClassifier", "fit_predict",
]
