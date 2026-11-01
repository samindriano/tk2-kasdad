"""Continuation-only adapter for the frozen C16 classification recipe.

The estimator/preprocessing implementation is inherited unchanged from the
verified C16 pilot adapter.  Only the newly approved temporal origins are
accepted here; pilot origins 55/67 remain immutable cached evidence.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.models.optimization_v3_batch1_c16 import (
    BLEND_WEIGHT,
    CARD_ID,
    FEATURE_NAMES,
    THRESHOLD,
    ModerateSparseCapacityClassifier as _PilotClassifier,
    _validated_controls,
)


NEW_CUTOFFS = (59, 63, 71, 75)


class FinalContinuationC16Classifier(_PilotClassifier):
    """The unchanged C16 estimator, gated to continuation cutoffs only."""

    def __init__(self, cutoff: int):
        if isinstance(cutoff, bool) or cutoff not in NEW_CUTOFFS:
            raise ValueError(f"C16 continuation cutoff must be one of {NEW_CUTOFFS}")
        # The parent constructor rejects new cutoffs because its job is only
        # the 55/67 pilot.  Its fit/predict implementation is reused as-is.
        self.cutoff = int(cutoff)
        self._fitted = False


def fit_predict(
    fit_features: pd.DataFrame,
    fit_target: np.ndarray,
    score_features: pd.DataFrame,
    *,
    controls: dict[str, Any],
    card_id: str,
    cutoff: int,
) -> dict[str, Any]:
    """Fit one C16 predictor on prefix rows and score exactly one 4-cycle window.

    The function accepts no score labels.  Baseline probabilities are aligned
    by original Train row positions and reconstructed by the unchanged C16
    present/missing route before the candidate is combined.
    """
    if card_id != CARD_ID or isinstance(cutoff, bool) or cutoff not in NEW_CUTOFFS:
        raise ValueError("C16 continuation only accepts card C16 at cutoffs 59/63/71/75")
    model = FinalContinuationC16Classifier(int(cutoff))
    fit_x = model._select(fit_features)
    score_x = model._select(score_features)
    if (not fit_x.index.is_unique or not fit_x.index.is_monotonic_increasing
            or not score_x.index.is_unique or not score_x.index.is_monotonic_increasing):
        raise ValueError("C16 continuation rows must preserve increasing source positions")
    fit_cycles = pd.to_numeric(fit_x["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    score_cycles = pd.to_numeric(score_x["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    if (not len(fit_cycles) or int(fit_cycles.min()) < 47
            or np.any(fit_cycles > int(cutoff))
            or not np.array_equal(np.unique(score_cycles),
                                  np.arange(int(cutoff) + 1, int(cutoff) + 5))):
        raise ValueError("C16 continuation requires prefix fit rows and the exact next four score cycles")

    arrays, control_rows = _validated_controls(controls, len(score_x))
    if not np.array_equal(score_x.index.to_numpy(dtype=np.int64), control_rows):
        raise ValueError("C16 continuation controls do not match ordered score row positions")
    model.fit(fit_x, fit_target)
    raw = model.predict_probability(score_x)
    prediction = np.where(
        arrays["stream_missing"], arrays["cb_probability"],
        0.75 * arrays["c2b_probability"] + BLEND_WEIGHT * raw,
    )
    if (prediction.shape != (len(score_x),) or not np.isfinite(prediction).all()
            or np.any((prediction < 0.0) | (prediction > 1.0))):
        raise RuntimeError("C16 continuation route returned invalid probabilities")
    return {
        "prediction": prediction,
        "raw_candidate": raw,
        "fit_predictor_count": 1,
        "state": model.audit_state(),
        "fit_metadata": {
            **model._prediction_diagnostics,
            "card_id": CARD_ID,
            "cutoff": int(cutoff),
            "fit_rows": int(len(fit_x)),
            "fit_cycle_min": int(fit_cycles.min()),
            "fit_cycle_max": int(fit_cycles.max()),
            "score_cycles": [int(value) for value in np.unique(score_cycles)],
            "candidate_present_stream_rows": int((~arrays["stream_missing"]).sum()),
            "frozen_cb_missing_stream_rows": int(arrays["stream_missing"].sum()),
            "blend_weight_new_present_expert": BLEND_WEIGHT,
            "threshold": THRESHOLD,
            "policy": ".75 frozen C2_B + .25 C16 on present; frozen CB on missing",
            "score_labels_passed_to_adapter": False,
        },
        "audit_arrays": {"stream_missing": arrays["stream_missing"]},
    }


__all__ = [
    "FEATURE_NAMES", "NEW_CUTOFFS", "FinalContinuationC16Classifier", "fit_predict",
]
