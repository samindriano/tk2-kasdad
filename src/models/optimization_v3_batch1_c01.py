"""C01 cache-only recomposition of the missing-stream expert route."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import numpy as np


CARD_ID = "C01"
THRESHOLD = 0.50
C2_WEIGHT = 0.75
SPARSE_WEIGHT = 0.25
FIT_ENDPOINTS = (55, 59, 63, 67, 71, 75)
FEATURE_NAMES = ("cycle_code",)  # generic harness context only; no model is fitted.


class C01IntegrityError(ValueError):
    """Raised when a source prediction or route is not exactly aligned."""


def _vector(name: str, values: Any, *, dtype: Any) -> np.ndarray:
    result = np.asarray(values, dtype=dtype)
    if result.ndim != 1:
        raise C01IntegrityError(f"{name} must be one-dimensional")
    return result


def _probability(name: str, values: Any, length: int) -> np.ndarray:
    result = _vector(name, values, dtype=np.float64)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise C01IntegrityError(f"{name} must be finite and aligned")
    if np.any((result < 0.0) | (result > 1.0)):
        raise C01IntegrityError(f"{name} must be within [0,1]")
    return result


def _route_arrays(arrays: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    required = {
        "baseline_prediction",
        "control__c1b_probability",
        "control__c2b_probability",
        "control__cb_missing_probability",
        "control__sparse_raw_probability",
        "control__stream_missing",
        "control__source_rows",
        "control__source_record_ids",
        "score_rows",
        "score_record_ids",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise C01IntegrityError(f"recomposition receipt omits fields: {missing}")
    baseline = _vector("baseline_prediction", arrays["baseline_prediction"], dtype=np.float64)
    n = len(baseline)
    c1b = _probability("C1_B probability", arrays["control__c1b_probability"], n)
    c2b = _probability("C2_B probability", arrays["control__c2b_probability"], n)
    cb = _probability("CatBoost missing probability", arrays["control__cb_missing_probability"], n)
    sparse = _probability("sparse25 present probability", arrays["control__sparse_raw_probability"], n)
    missing_mask = _vector("stream_missing", arrays["control__stream_missing"], dtype=bool)
    score_rows = _vector("score_rows", arrays["score_rows"], dtype=np.int64)
    source_rows = _vector("source_rows", arrays["control__source_rows"], dtype=np.int64)
    score_ids = _vector("score_record_ids", arrays["score_record_ids"], dtype=str)
    source_ids = _vector("source_record_ids", arrays["control__source_record_ids"], dtype=str)
    if any(value.shape != (n,) for value in (missing_mask, score_rows, source_rows, score_ids, source_ids)):
        raise C01IntegrityError("C01 masks and row/ID receipts are not aligned")
    if not np.array_equal(score_rows, source_rows) or not np.array_equal(score_ids, source_ids):
        raise C01IntegrityError("C01 source cache rows or IDs differ from the generic Train view")
    if len(np.unique(score_rows)) != n or len(np.unique(score_ids)) != n:
        raise C01IntegrityError("C01 score rows or IDs are duplicated")
    expected_baseline = np.where(missing_mask, cb, C2_WEIGHT * c2b + SPARSE_WEIGHT * sparse)
    if not np.array_equal(baseline, expected_baseline):
        raise C01IntegrityError("C01 sparse25 incumbent does not reconstruct exactly")
    candidate = np.where(missing_mask, c1b, C2_WEIGHT * c2b + SPARSE_WEIGHT * sparse)
    if not np.isfinite(candidate).all() or np.any((candidate < 0.0) | (candidate > 1.0)):
        raise C01IntegrityError("C01 recomposed probabilities are outside [0,1]")
    return candidate, expected_baseline


def fit_predict(
    fit_x: Any,
    fit_y: Any,
    score_x: Any,
    *,
    controls: Mapping[str, Any],
    card_id: str,
    cutoff: int,
) -> dict[str, Any]:
    """Recompose verified cache vectors; perform no predictor/preprocessor fit."""
    if str(card_id) != CARD_ID or int(cutoff) not in FIT_ENDPOINTS:
        raise C01IntegrityError(f"C01 cannot serve card={card_id!r}, cutoff={cutoff!r}")
    del fit_x, fit_y, score_x  # No training or label-dependent policy is part of C01.
    arrays = {
        "baseline_prediction": controls["baseline_prediction"],
        **{f"control__{key}": controls[key] for key in (
            "c1b_probability", "c2b_probability", "cb_missing_probability",
            "sparse_raw_probability", "stream_missing", "source_rows", "source_record_ids")},
        "score_rows": controls["source_rows"],
        "score_record_ids": controls["source_record_ids"],
    }
    candidate, baseline = _route_arrays(arrays)
    state = {
        "card_id": CARD_ID,
        "cutoff": int(cutoff),
        "policy": "missing rows use historical C1_B; present rows use 0.75*C2_B + 0.25*SPARSE_BINARY",
        "threshold": THRESHOLD,
        "predictor_fits": 0,
        "preprocessing_fits": 0,
        "source_cache_recomposition_only": True,
        "missing_rows": int(np.sum(np.asarray(controls["stream_missing"], dtype=bool))),
        "present_rows": int(len(candidate) - np.sum(np.asarray(controls["stream_missing"], dtype=bool))),
        "candidate_probability_sha256": hashlib.sha256(candidate.astype("<f8").tobytes()).hexdigest(),
        "baseline_probability_sha256": hashlib.sha256(baseline.astype("<f8").tobytes()).hexdigest(),
        "outer_validation_labels_passed_to_policy": False,
        "test_data_used": False,
    }
    state["state_sha256"] = hashlib.sha256(json.dumps(
        state, sort_keys=True, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    return {
        "prediction": candidate,
        "fit_predictor_count": 0,
        "fit_metadata": {
            "predictor_fits": 0,
            "preprocessing_fits": 0,
            "cache_recomposition_count": 1,
            "threshold": THRESHOLD,
        },
        "state": state,
    }


def reconstruct_c01(arrays: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Independently rebuild C01 candidate and the frozen sparse25 baseline."""
    return _route_arrays(arrays)


__all__ = [
    "C01IntegrityError", "CARD_ID", "C2_WEIGHT", "FEATURE_NAMES", "FIT_ENDPOINTS",
    "SPARSE_WEIGHT", "THRESHOLD", "fit_predict", "reconstruct_c01",
]
