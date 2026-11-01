"""EF1-only raw-scale sparse-L1 specialist for V3 card R14."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.models.optimization_v2_regression import build_sparse_hybrid
from src.models.regression_research_text import HYBRID_FEATURES


CARD_ID = "R14"
FEATURE_NAMES = tuple(HYBRID_FEATURES)
MIN_EF1_FIT_ROWS = 500
MIN_EF1_SCORE_ROWS = 100
INCUMBENT_WEIGHT = 0.75
EXPERT_WEIGHT = 0.25
PREDICTION_RANGE = (0.0, 100.0)


class R14IntegrityError(ValueError):
    """Raised when the frozen R14 fit or prediction contract is violated."""


def _vector(name: str, values: Any, *, dtype: Any = np.float64) -> np.ndarray:
    result = np.asarray(values, dtype=dtype)
    if result.ndim != 1:
        raise R14IntegrityError(f"{name} must be one-dimensional")
    return result


def ef1_mask(values: Any) -> np.ndarray:
    """Use the diagnostics' parsed entry_friction==1 rule; telemetry is unrelated."""
    parsed = pd.to_numeric(pd.Series(values, copy=False), errors="coerce")
    return parsed.eq(1).to_numpy(dtype=bool)


def compose_r14_prediction(
    raw_expert: Any, incumbent: Any, expert_applied: Any,
) -> np.ndarray:
    """Blend clipped raw expert predictions on EF1 only; keep other rows exact."""
    raw = _vector("raw_expert", raw_expert)
    base = _vector("incumbent", incumbent)
    mask = _vector("expert_applied", expert_applied, dtype=bool)
    if raw.shape != base.shape or mask.shape != base.shape or len(base) == 0:
        raise R14IntegrityError("R14 route inputs are empty or misaligned")
    if not np.isfinite(raw).all() or not np.isfinite(base).all():
        raise R14IntegrityError("R14 route inputs contain nonfinite values")
    if np.any((base < 0.0) | (base > 100.0)):
        raise R14IntegrityError("Frozen incumbent predictions must be in [0,100]")
    clipped_expert = np.clip(raw, *PREDICTION_RANGE)
    prediction = base.copy()
    prediction[mask] = (
        INCUMBENT_WEIGHT * base[mask] + EXPERT_WEIGHT * clipped_expert[mask]
    )
    prediction = np.clip(prediction, *PREDICTION_RANGE)
    if not np.isfinite(prediction).all():
        raise R14IntegrityError("R14 route returned nonfinite predictions")
    return prediction


def fit_predict(
    fit_x: pd.DataFrame,
    fit_y: Any,
    score_x: pd.DataFrame,
    *,
    controls: Mapping[str, Any],
    card_id: str,
    cutoff: int,
) -> dict[str, Any]:
    """Fit one original sparse L1 model on EF1 fit rows and apply fixed route."""
    if card_id != CARD_ID:
        raise R14IntegrityError(f"Unexpected card id: {card_id}")
    if set(fit_x.columns) != set(FEATURE_NAMES) or len(fit_x.columns) != len(FEATURE_NAMES):
        raise R14IntegrityError("Fit frame differs from the frozen feature allowlist")
    if set(score_x.columns) != set(FEATURE_NAMES) or len(score_x.columns) != len(FEATURE_NAMES):
        raise R14IntegrityError("Score frame differs from the frozen feature allowlist")
    target = _vector("fit_y", fit_y)
    if target.shape != (len(fit_x),) or not np.isfinite(target).all():
        raise R14IntegrityError("R14 fit target is invalid or row-misaligned")
    if np.any((target < 0.0) | (target > 100.0)):
        raise R14IntegrityError("R14 fit target must be in [0,100]")

    fit_mask = ef1_mask(fit_x["entry_friction"])
    score_mask = ef1_mask(score_x["entry_friction"])
    ef_fit_n = int(fit_mask.sum())
    if ef_fit_n < MIN_EF1_FIT_ROWS:
        raise R14IntegrityError(
            f"R14 requires at least {MIN_EF1_FIT_ROWS} EF1 fit rows; got {ef_fit_n}"
        )
    incumbent = _vector("baseline_prediction", controls.get("baseline_prediction"))
    if incumbent.shape != (len(score_x),):
        raise R14IntegrityError("R14 incumbent is not aligned with score rows")
    model = build_sparse_hybrid("lgbm_sparse_l1")
    model.fit(fit_x.loc[fit_mask, list(FEATURE_NAMES)].copy(), target[fit_mask])
    raw = np.zeros(len(score_x), dtype=np.float64)
    raw[score_mask] = model.predict(score_x.loc[score_mask, list(FEATURE_NAMES)].copy())
    prediction = compose_r14_prediction(raw, incumbent, score_mask)
    state = model.audit_state()
    if int(state.get("fit_rows", -1)) != ef_fit_n:
        raise R14IntegrityError("Sparse model state does not confirm the EF1-only fit size")
    return {
        "prediction": prediction,
        "raw_candidate": raw,
        "audit_arrays": {"expert_applied_mask": score_mask.astype(np.uint8)},
        "fit_metadata": {
            "cutoff": int(cutoff),
            "outer_fit_rows": int(len(fit_x)),
            "ef1_fit_rows": ef_fit_n,
            "ef1_score_rows": int(score_mask.sum()),
            "expert_fit_population": "parsed entry_friction == 1",
            "raw_candidate_zero_outside_ef1": True,
            "prediction_policy": "EF1: clip(0.75*incumbent + 0.25*clip(raw_expert,0,100),0,100); other rows exact incumbent",
            "training_labels_passed": "outer fit rows only; EF1 subset only",
            "score_labels_passed": False,
        },
        "state": state,
        "fit_predictor_count": 1,
    }


__all__ = [
    "CARD_ID", "EXPERT_WEIGHT", "FEATURE_NAMES", "INCUMBENT_WEIGHT",
    "MIN_EF1_FIT_ROWS", "MIN_EF1_SCORE_ROWS", "PREDICTION_RANGE",
    "R14IntegrityError", "compose_r14_prediction", "ef1_mask", "fit_predict",
]
