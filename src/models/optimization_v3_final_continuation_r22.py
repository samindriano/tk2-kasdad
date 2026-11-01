"""Frozen R22 recipe adapter for the final continuation origins only.

The estimator, feature preparation, and fit method are inherited unchanged from
the audited pilot implementation. This adapter only admits the four predeclared
continuation cutoffs; it does not alter the model recipe.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

from src.models.optimization_v3_batch1_r22 import (
    CARD_ID,
    FEATURE_NAMES,
    XGBoostPseudoHuberSVD32Regressor,
    _cycle_codes,
    _select_exact,
    _target,
)


CONTINUATION_CUTOFFS = (59, 63, 71, 75)
THREADS = 2


class FinalContinuationR22Regressor(XGBoostPseudoHuberSVD32Regressor):
    """Use the byte-frozen R22 fit implementation at continuation cutoffs."""

    def __init__(self, cutoff: int):
        cutoff = int(cutoff)
        if cutoff not in CONTINUATION_CUTOFFS:
            raise ValueError(f"R22 continuation cutoff must be one of {CONTINUATION_CUTOFFS}")
        # Parent initialization only validates the cutoff and initializes state.
        # Set the continuation cutoff immediately after that unchanged setup.
        super().__init__(cutoff=55)
        self.cutoff = cutoff


def cycle_average_percentile(raw_prediction: Any, cycle_codes: Any) -> np.ndarray:
    """Return 100*average_rank/n within each complete cycle, preserving ties."""
    raw = np.asarray(raw_prediction, dtype=np.float64)
    cycles = np.asarray(cycle_codes)
    if (raw.ndim != 1 or raw.size == 0 or cycles.shape != raw.shape
            or not np.isfinite(raw).all() or cycles.dtype.kind not in "iu"):
        raise ValueError("R22 percentile inputs must be aligned finite predictions and integer cycles")
    percentile = np.empty(raw.shape, dtype=np.float64)
    for cycle in np.unique(cycles):
        mask = cycles == cycle
        percentile[mask] = 100.0 * rankdata(raw[mask], method="average") / int(mask.sum())
    if not np.isfinite(percentile).all():
        raise ValueError("R22 percentile mapping returned nonfinite values")
    return percentile


def postprocess_policies(raw_prediction: Any, champion_prediction: Any,
                         cycle_codes: Any, stream_present: Any) -> dict[str, np.ndarray]:
    """Apply the two predeclared R22 mappings to one set of raw model scores."""
    raw = np.asarray(raw_prediction, dtype=np.float64)
    champion = np.asarray(champion_prediction, dtype=np.float64)
    cycles = np.asarray(cycle_codes)
    present = np.asarray(stream_present, dtype=bool)
    if (raw.ndim != 1 or raw.size == 0 or champion.shape != raw.shape
            or cycles.shape != raw.shape or present.shape != raw.shape):
        raise ValueError("R22 policy arrays must be aligned nonempty vectors")
    if not np.isfinite(raw).all() or not np.isfinite(champion).all():
        raise ValueError("R22 policy inputs must be finite")
    if np.any((champion < 0.0) | (champion > 100.0)):
        raise ValueError("R22 champion predictions must be in [0,100]")

    pct = cycle_average_percentile(raw, cycles)
    original_component = np.clip(0.5 * raw + 0.5 * pct, 0.0, 100.0)
    original = np.clip(0.75 * champion + 0.25 * original_component, 0.0, 100.0)
    pct_present_value = np.clip(0.75 * champion + 0.25 * pct, 0.0, 100.0)
    pct_present = np.where(present, pct_present_value, champion)
    if not all(np.isfinite(v).all() for v in (pct, original_component, original, pct_present)):
        raise ValueError("R22 postprocessing returned nonfinite predictions")
    return {
        "cycle_percentile": pct,
        "original_component": original_component,
        "R22_ORIGINAL": original,
        "R22_PCT_PRESENT": pct_present,
    }


def fit_raw_predict(fit_features: pd.DataFrame, fit_target: Any,
                    score_features: pd.DataFrame, *, cutoff: int) -> dict[str, Any]:
    """Fit the unchanged R22 recipe and return raw scores without score labels."""
    cutoff = int(cutoff)
    if cutoff not in CONTINUATION_CUTOFFS:
        raise ValueError(f"R22 continuation cutoff must be one of {CONTINUATION_CUTOFFS}")
    fit_x = _select_exact(fit_features)
    score_x = _select_exact(score_features)
    if (not fit_x.index.is_unique or not fit_x.index.is_monotonic_increasing
            or not score_x.index.is_unique or not score_x.index.is_monotonic_increasing):
        raise ValueError("R22 fit and score rows must preserve increasing source positions")
    target = _target(fit_target, len(fit_x))
    fit_cycles = _cycle_codes(fit_x, label="R22 continuation fit features")
    score_cycles = _cycle_codes(score_x, label="R22 continuation score features")
    if np.any(fit_cycles > cutoff) or np.any((score_cycles <= cutoff) | (score_cycles > cutoff + 4)):
        raise ValueError("R22 continuation rows violate the frozen four-cycle horizon")
    if not np.array_equal(np.unique(score_cycles), np.arange(cutoff + 1, cutoff + 5, dtype=np.int64)):
        raise ValueError("R22 continuation score rows must cover exactly four consecutive cycles")

    with threadpool_limits(limits=THREADS):
        model = FinalContinuationR22Regressor(cutoff=cutoff).fit(fit_x, target)
        raw = model.predict(score_x)
    if raw.shape != (len(score_x),) or not np.isfinite(raw).all():
        raise RuntimeError("R22 continuation raw predictions are misaligned or nonfinite")
    return {
        "raw_prediction": raw,
        "state": model.audit_state(),
        "fit_metadata": {
            "card_id": CARD_ID,
            "cutoff": cutoff,
            "fit_rows": int(len(fit_x)),
            "score_rows": int(len(score_x)),
            "fit_cycle_min": int(fit_cycles.min()),
            "fit_cycle_max": int(fit_cycles.max()),
            "score_cycle_min": int(score_cycles.min()),
            "score_cycle_max": int(score_cycles.max()),
            "feature_names": list(FEATURE_NAMES),
            "fit_prefix_only_transforms": True,
            "score_labels_passed_to_fit": False,
            "early_stopping_used": False,
            "predictor_fit_count": 1,
        },
    }


__all__ = [
    "CONTINUATION_CUTOFFS",
    "FinalContinuationR22Regressor",
    "cycle_average_percentile",
    "fit_raw_predict",
    "postprocess_policies",
]
