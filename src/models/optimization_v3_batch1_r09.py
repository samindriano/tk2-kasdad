"""R09: frozen SVD32 LightGBM design with smooth Huber objective."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd

from src.models.optimization_v3_batch1_r21 import rank50_prediction
from src.models.regression_research_lgb_text import (
    RegressionResearchLGBText,
    _select_exact,
)
from src.models.regression_research_tabular import COMMON_PARAMETERS, PARAMETER_SETS
from src.models.regression_research_text import HYBRID_FEATURES, _array_digest, _json_digest
from src.models.round1_models import _target_array


CARD_ID = "R09"
FEATURE_NAMES = tuple(HYBRID_FEATURES)
PILOT_CUTOFFS = (55, 67)
RANK_WEIGHT = 0.50
BLEND_WEIGHT = 0.25
HUBER_ALPHA = 0.90
HUBER_PARAMETERS = {
    **dict(COMMON_PARAMETERS),
    **dict(PARAMETER_SETS["base"]),
    "objective": "huber",
    "alpha": HUBER_ALPHA,
}
_CONTROL_KEYS = (
    "baseline_prediction", "baseline_lgb_svd32", "baseline_hybrid_ridge",
    "baseline_sparse_raw", "baseline_sparse_rank50", "control_rows",
)


class HuberSVD32Regressor(RegressionResearchLGBText):
    """Reuse the inherited fit-only SVD32 pipeline; change only objective and alpha."""

    def __init__(self, cutoff: int):
        if int(cutoff) not in PILOT_CUTOFFS:
            raise ValueError(f"R09 cutoff must be one of {PILOT_CUTOFFS}")
        super().__init__(params_id="base")
        self.cutoff = int(cutoff)
        self.model_id = "R09_HUBER_SVD32_LGBM"
        self.actual_parameters = dict(HUBER_PARAMETERS)
        self._fit_cycles: np.ndarray | None = None
        self._fit_row_positions: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: Any) -> "HuberSVD32Regressor":
        selected = _select_exact(X)
        target = _target_array(y, len(selected))
        cycles_raw = pd.to_numeric(selected["cycle_code"], errors="raise").to_numpy(dtype=np.float64)
        if (cycles_raw.ndim != 1 or not np.isfinite(cycles_raw).all()
                or not np.equal(cycles_raw, np.floor(cycles_raw)).all()):
            raise ValueError("R09 fit cycle_code must be finite integer values")
        cycles = cycles_raw.astype(np.int64)
        if (np.any((cycles < 47) | (cycles > self.cutoff))
                or not selected.index.is_unique or not selected.index.is_monotonic_increasing):
            raise ValueError("R09 fit rows must be unique ordered Train rows from 47 through cutoff")
        self._fit_cycles = cycles.copy()
        self._fit_row_positions = selected.index.to_numpy(dtype="<i8", copy=True)
        super().fit(selected, target)
        if self._svd_components_actual != 32:
            raise ValueError(f"R09 requires the inherited full SVD32; actual={self._svd_components_actual}")
        return self

    def _make_state(self) -> dict[str, Any]:
        state = super()._make_state()
        if self._fit_cycles is None or self._fit_row_positions is None:
            raise RuntimeError("R09 fit chronology was not captured before preprocessing")
        state.update({
            "id": "R09_HUBER_SVD32_LGBM",
            "objective": "huber",
            "alpha": HUBER_ALPHA,
            "cutoff": self.cutoff,
            "fit_cycle_min": int(self._fit_cycles.min()),
            "fit_cycle_max": int(self._fit_cycles.max()),
            "fit_cycle_counts": {
                str(int(cycle)): int(count)
                for cycle, count in zip(*np.unique(self._fit_cycles, return_counts=True))
            },
            "fit_row_positions_sha256": hashlib.sha256(
                self._fit_row_positions.tobytes(order="C")
            ).hexdigest(),
            "actual_hyperparameters": dict(self.actual_parameters),
            "validation_labels_passed_to_fit": False,
            "sample_weight_used": False,
        })
        state.pop("state_sha256", None)
        state["state_sha256"] = _json_digest(state)
        return json.loads(json.dumps(state, sort_keys=True, allow_nan=False))


def rank50_independent(raw_prediction: Any, cycle_codes: Any) -> np.ndarray:
    """Independent pandas average-rank implementation for audit/test parity."""
    raw = np.asarray(raw_prediction, dtype=np.float64)
    cycles = np.asarray(cycle_codes, dtype=np.int64)
    if raw.ndim != 1 or raw.size == 0 or raw.shape != cycles.shape or not np.isfinite(raw).all():
        raise ValueError("R09 rank50 values must be nonempty, finite and aligned")
    output = np.empty(raw.shape, dtype=np.float64)
    for cycle in np.unique(cycles):
        mask = cycles == cycle
        rank = pd.Series(raw[mask]).rank(method="average").to_numpy(dtype=np.float64)
        output[mask] = 0.5 * raw[mask] + 0.5 * 100.0 * rank / int(mask.sum())
    return np.clip(output, 0.0, 100.0)


def reconstruct_r09(raw_prediction: Any, cycle_codes: Any,
                    baseline_prediction: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.asarray(raw_prediction, dtype=np.float64)
    cycles = np.asarray(cycle_codes, dtype=np.int64)
    baseline = np.asarray(baseline_prediction, dtype=np.float64)
    if raw.ndim != 1 or raw.size == 0 or raw.shape != cycles.shape or raw.shape != baseline.shape:
        raise ValueError("R09 route arrays must be aligned nonempty vectors")
    if (not np.isfinite(raw).all() or not np.isfinite(baseline).all()
            or np.any((baseline < 0.0) | (baseline > 100.0))):
        raise ValueError("R09 route inputs must be finite and baseline in [0,100]")
    mapped = rank50_prediction(raw, cycles)
    candidate = np.clip(0.75 * baseline + BLEND_WEIGHT * mapped, 0.0, 100.0)
    return candidate, baseline.copy(), mapped


def _validated_controls(controls: dict[str, Any], score_cycles: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    absent = [name for name in _CONTROL_KEYS if name not in controls]
    n = len(score_cycles)
    if absent or n <= 0:
        raise ValueError(f"R09 controls are missing required arrays: {absent}")
    arrays: dict[str, np.ndarray] = {}
    for name in _CONTROL_KEYS[:-1]:
        values = np.asarray(controls[name], dtype=np.float64)
        if values.shape != (n,) or not np.isfinite(values).all():
            raise ValueError(f"R09 control {name} is nonfinite or misaligned")
        if name != "baseline_sparse_raw" and np.any((values < 0.0) | (values > 100.0)):
            raise ValueError(f"R09 control {name} is outside [0,100]")
        arrays[name] = values
    rows = np.asarray(controls["control_rows"])
    if (rows.shape != (n,) or rows.dtype.kind not in "iu" or len(np.unique(rows)) != n
            or np.any(np.diff(rows.astype(np.int64)) <= 0)):
        raise ValueError("R09 control rows must be unique and increasing")
    sparse_rank = rank50_prediction(arrays["baseline_sparse_raw"], score_cycles)
    if not np.array_equal(sparse_rank, arrays["baseline_sparse_rank50"]):
        raise ValueError("R09 fixed sparse rank50 component does not reconstruct")
    rebuilt = np.clip(
        0.75 * (0.75 * arrays["baseline_lgb_svd32"]
                + 0.25 * arrays["baseline_hybrid_ridge"])
        + 0.25 * sparse_rank, 0.0, 100.0,
    )
    if not np.array_equal(rebuilt, arrays["baseline_prediction"]):
        raise ValueError("R09 control is not the exact fixed three-component champion")
    return arrays, rows.astype(np.int64, copy=False)


def fit_predict(fit_features: pd.DataFrame, fit_target: Any,
                score_features: pd.DataFrame, *, controls: dict[str, Any],
                card_id: str, cutoff: int) -> dict[str, Any]:
    if card_id != CARD_ID or int(cutoff) not in PILOT_CUTOFFS:
        raise ValueError("R09 adapter accepts only card R09 at cutoffs 55 and 67")
    fit_x = _select_exact(fit_features)
    score_x = _select_exact(score_features)
    if (not fit_x.index.is_unique or not fit_x.index.is_monotonic_increasing
            or not score_x.index.is_unique or not score_x.index.is_monotonic_increasing):
        raise ValueError("R09 fit and score rows must preserve increasing Train positions")
    fit_cycles = pd.to_numeric(fit_x["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    score_cycles = pd.to_numeric(score_x["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    if (np.any((fit_cycles < 47) | (fit_cycles > int(cutoff)))
            or np.any((score_cycles <= int(cutoff)) | (score_cycles > int(cutoff) + 12))):
        raise ValueError("R09 fit/score rows violate the frozen temporal horizon")
    control, rows = _validated_controls(controls, score_cycles)
    if not np.array_equal(score_x.index.to_numpy(dtype=np.int64), rows):
        raise ValueError("R09 row-matched controls do not align with score Train positions")
    model = HuberSVD32Regressor(cutoff).fit(fit_x, fit_target)
    raw = model.predict(score_x)
    prediction, baseline, mapped = reconstruct_r09(raw, score_cycles, control["baseline_prediction"])
    return {
        "prediction": prediction,
        "baseline_prediction": baseline,
        "raw_candidate": raw,
        "fit_predictor_count": 1,
        "state": model.audit_state(),
        "fit_metadata": {
            "cutoff": int(cutoff), "fit_rows": int(len(fit_x)),
            "fit_cycle_min": int(fit_cycles.min()), "fit_cycle_max": int(fit_cycles.max()),
            "policy": ".75 fixed three-component champion + .25 rank50(new target-scale Huber-SVD32 LGB), clipped [0,100]",
            "score_labels_passed_to_adapter": False,
            "prediction_diagnostics": model.prediction_diagnostics(),
        },
        "audit_arrays": {"score_cycles": score_cycles, "candidate_rank50": mapped,
                         "control_rows": rows},
    }


__all__ = [
    "BLEND_WEIGHT", "CARD_ID", "FEATURE_NAMES", "HUBER_ALPHA", "HUBER_PARAMETERS",
    "HUBER_SVD32Regressor", "PILOT_CUTOFFS", "fit_predict", "rank50_independent",
    "reconstruct_r09",
]
