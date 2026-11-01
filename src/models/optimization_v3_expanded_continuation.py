"""Continuation-only adapters for the five remaining V3 ADVANCE candidates.

The frozen pilot modules are imported unchanged. These adapters extend only the
accepted temporal origins and reproduce each candidate's original fit and
prediction recipe; they never accept validation labels.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.models import optimization_v3_batch1_c11 as c11
from src.models import optimization_v3_batch1_c17 as c17
from src.models import optimization_v3_batch1_r06 as r06
from src.models import optimization_v3_batch1_r11 as r11
from src.models import optimization_v3_batch1_r21 as r21


NEW_CUTOFFS = (59, 63, 71, 75)
THREADS = 2


def _validate_temporal_rows(
    fit_features: pd.DataFrame,
    score_features: pd.DataFrame,
    target: Any,
    *,
    cutoff: int,
    feature_names: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(cutoff, bool) or int(cutoff) not in NEW_CUTOFFS:
        raise ValueError(f"Continuation cutoff must be in {NEW_CUTOFFS}")
    cutoff = int(cutoff)
    if not isinstance(fit_features, pd.DataFrame) or not isinstance(score_features, pd.DataFrame):
        raise TypeError("Fit and score features must be pandas DataFrames")
    for frame in (fit_features, score_features):
        if frame.columns.has_duplicates or set(frame.columns) != set(feature_names):
            raise ValueError("Candidate feature frame does not match its frozen allowlist")
        if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
            raise ValueError("Candidate rows must retain unique, increasing official Train positions")
    fit_x = fit_features.loc[:, list(feature_names)].copy()
    score_x = score_features.loc[:, list(feature_names)].copy()
    fit_cycles = pd.to_numeric(fit_x["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    score_cycles = pd.to_numeric(score_x["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    y = np.asarray(target)
    if y.ndim != 1 or y.shape != (len(fit_x),):
        raise ValueError("Fit labels are not aligned to the fit prefix")
    if (not len(fit_cycles) or int(fit_cycles.min()) != 47
            or np.any(fit_cycles > cutoff)
            or not np.array_equal(np.unique(score_cycles),
                                  np.arange(cutoff + 1, cutoff + 5, dtype=np.int64))):
        raise ValueError("Continuation must fit cycles 47..cutoff and score the next four full cycles")
    return fit_x, score_x, y, fit_cycles, score_cycles


def _check_controls(controls: Mapping[str, Any], score_rows: np.ndarray,
                    score_cycles: np.ndarray, *, task: str) -> None:
    if not isinstance(controls, Mapping):
        raise TypeError(f"{task} continuation controls must be a mapping")
    control_rows = np.asarray(controls.get("control_rows"), dtype=np.int64)
    if not np.array_equal(control_rows, score_rows):
        raise ValueError(f"{task} frozen control rows do not match the score rows")
    if "cycle_code" in controls and not np.array_equal(
            np.asarray(controls["cycle_code"], dtype=np.int64), score_cycles):
        raise ValueError(f"{task} cycle controls do not match the score cycles")


class _ContinuationC17(c17.LongerRegularizedSparseClassifier):
    def __init__(self, cutoff: int):
        if isinstance(cutoff, bool) or int(cutoff) not in NEW_CUTOFFS:
            raise ValueError("C17 continuation only accepts cutoffs 59, 63, 71, and 75")
        self.cutoff = int(cutoff)
        self._fitted = False


class _ContinuationR21(r21.XGBoostSVD32Regressor):
    def __init__(self, cutoff: int):
        if isinstance(cutoff, bool) or int(cutoff) not in NEW_CUTOFFS:
            raise ValueError("R21 continuation only accepts cutoffs 59, 63, 71, and 75")
        super().__init__(cutoff=55)
        self.cutoff = int(cutoff)


def fit_c17(fit_features: pd.DataFrame, fit_target: Any,
            score_features: pd.DataFrame, *, controls: Mapping[str, Any],
            cutoff: int) -> dict[str, Any]:
    fit_x, score_x, y, fit_cycles, score_cycles = _validate_temporal_rows(
        fit_features, score_features, fit_target, cutoff=cutoff,
        feature_names=tuple(c17.FEATURE_NAMES))
    _check_controls(controls, score_x.index.to_numpy(dtype=np.int64), score_cycles, task="C17")
    arrays, control_rows = c17._validated_controls(dict(controls), len(score_x))
    if not np.array_equal(control_rows, score_x.index.to_numpy(dtype=np.int64)):
        raise ValueError("C17 controls are not row-exact")
    with threadpool_limits(limits=THREADS):
        model = _ContinuationC17(cutoff).fit(fit_x, y)
        raw = model.predict_probability(score_x)
    prediction = np.where(
        arrays["stream_missing"], arrays["cb_probability"],
        0.75 * arrays["c2b_probability"] + c17.BLEND_WEIGHT * raw,
    )
    if not np.isfinite(prediction).all() or np.any((prediction < 0) | (prediction > 1)):
        raise RuntimeError("C17 continuation produced invalid probabilities")
    return {
        "prediction": prediction, "raw_candidate": raw,
        "baseline_prediction": arrays["baseline_prediction"],
        "fit_predictor_count": 1, "state": model.audit_state(),
        "fit_metadata": {
            "cutoff": int(cutoff), "fit_rows": len(fit_x), "score_rows": len(score_x),
            "fit_cycle_min": int(fit_cycles.min()), "fit_cycle_max": int(fit_cycles.max()),
            "score_cycle_min": int(score_cycles.min()), "score_cycle_max": int(score_cycles.max()),
            "predictor_fit_count": 1, "structured_preprocessing_fit_calls": 1,
            "word_tfidf_fit_calls": 1, "validation_labels_passed_to_fit": False,
            "test_data_used": False, "threshold": c17.THRESHOLD,
            "policy": ".75 C2_B + .25 C17 for present; frozen CB_MISSING for missing",
        },
    }


def fit_c11(fit_features: pd.DataFrame, fit_target: Any,
            score_features: pd.DataFrame, *, controls: Mapping[str, Any],
            cutoff: int) -> dict[str, Any]:
    fit_x, score_x, y, fit_cycles, score_cycles = _validate_temporal_rows(
        fit_features, score_features, fit_target, cutoff=cutoff,
        feature_names=tuple(c11.FEATURE_NAMES))
    _check_controls(controls, score_x.index.to_numpy(dtype=np.int64), score_cycles, task="C11")
    required = {"baseline_prediction", "baseline_sparse_raw", "c2b_probability",
                "cb_probability", "stream_missing", "control_rows"}
    if not required.issubset(controls):
        raise ValueError(f"C11 controls are missing {sorted(required - set(controls))}")
    missing = np.asarray(controls["stream_missing"], dtype=bool)
    c2b = np.asarray(controls["c2b_probability"], dtype=np.float64)
    cb = np.asarray(controls["cb_probability"], dtype=np.float64)
    old_sparse = np.asarray(controls["baseline_sparse_raw"], dtype=np.float64)
    baseline = np.asarray(controls["baseline_prediction"], dtype=np.float64)
    if not (missing.shape == c2b.shape == cb.shape == old_sparse.shape == baseline.shape == (len(score_x),)):
        raise ValueError("C11 controls are not aligned")
    reconstructed = np.where(missing, cb, 0.75 * c2b + 0.25 * old_sparse)
    if (not np.isfinite(reconstructed).all()
            or not np.allclose(reconstructed, baseline, rtol=0.0, atol=1e-12)):
        raise ValueError("C11 frozen comparator does not reconstruct from its row-aligned controls")
    with threadpool_limits(limits=THREADS):
        model = c11.WordCharSparseClassifier().fit(fit_x, y)
        raw = model.predict_proba(score_x)
    prediction = np.where(missing, cb, 0.75 * c2b + 0.25 * raw)
    if not np.isfinite(prediction).all() or np.any((prediction < 0) | (prediction > 1)):
        raise RuntimeError("C11 continuation produced invalid probabilities")
    return {
        "prediction": prediction, "raw_candidate": raw, "baseline_prediction": baseline,
        "fit_predictor_count": 1,
        "state": {"card_id": "C11", "cutoff": int(cutoff), "model": model.audit_state(),
                  "threshold": 0.50, "blend_weight_new": 0.25,
                  "missing_stream_route": "frozen CB_MISSING"},
        "fit_metadata": {
            "cutoff": int(cutoff), "fit_rows": len(fit_x), "score_rows": len(score_x),
            "fit_cycle_min": int(fit_cycles.min()), "fit_cycle_max": int(fit_cycles.max()),
            "score_cycle_min": int(score_cycles.min()), "score_cycle_max": int(score_cycles.max()),
            "predictor_fit_count": 1, "structured_preprocessing_fit_calls": 1,
            "word_tfidf_fit_calls": 1, "character_tfidf_fit_calls": 1,
            "validation_labels_passed_to_fit": False, "test_data_used": False,
            "threshold": 0.50,
        },
    }


def fit_r21(fit_features: pd.DataFrame, fit_target: Any,
            score_features: pd.DataFrame, *, controls: Mapping[str, Any],
            cutoff: int) -> dict[str, Any]:
    fit_x, score_x, y, fit_cycles, score_cycles = _validate_temporal_rows(
        fit_features, score_features, fit_target, cutoff=cutoff,
        feature_names=tuple(r21.FEATURE_NAMES))
    _check_controls(controls, score_x.index.to_numpy(dtype=np.int64), score_cycles, task="R21")
    control, rows = r21._validated_controls(dict(controls), len(score_x))
    if not np.array_equal(rows, score_x.index.to_numpy(dtype=np.int64)):
        raise ValueError("R21 controls are not row-exact")
    with threadpool_limits(limits=THREADS):
        model = _ContinuationR21(cutoff).fit(fit_x, y)
        raw = model.predict(score_x)
    prediction, baseline, mapped = r21.reconstruct_r21(
        raw, control["baseline_prediction"], score_cycles)
    if not np.isfinite(prediction).all():
        raise RuntimeError("R21 continuation produced nonfinite predictions")
    return {
        "prediction": prediction, "raw_candidate": raw, "baseline_prediction": baseline,
        "fit_predictor_count": 1, "state": model.audit_state(),
        "fit_metadata": {
            "cutoff": int(cutoff), "fit_rows": len(fit_x), "score_rows": len(score_x),
            "fit_cycle_min": int(fit_cycles.min()), "fit_cycle_max": int(fit_cycles.max()),
            "score_cycle_min": int(score_cycles.min()), "score_cycle_max": int(score_cycles.max()),
            "predictor_fit_count": 1, "structured_preprocessing_fit_calls": 1,
            "tfidf_fit_calls": 1, "svd_fit_calls": 1,
            "validation_labels_passed_to_fit": False, "test_data_used": False,
            "rank_mapping": "rank50 within each complete cycle using average ties",
        },
        "audit_arrays": {"candidate_rank50": mapped},
    }


def fit_r11(fit_features: pd.DataFrame, fit_target: Any,
            score_features: pd.DataFrame, *, controls: Mapping[str, Any],
            cutoff: int) -> dict[str, Any]:
    fit_x, score_x, y, fit_cycles, score_cycles = _validate_temporal_rows(
        fit_features, score_features, fit_target, cutoff=cutoff,
        feature_names=tuple(r11.FEATURE_NAMES))
    _check_controls(controls, score_x.index.to_numpy(dtype=np.int64), score_cycles, task="R11")
    model = r11.CumulativeOrdinalSVD32()
    with threadpool_limits(limits=THREADS):
        model.fit(fit_x, y)
        raw = model.predict(score_x)
    baseline = np.asarray(controls["baseline_prediction"], dtype=np.float64)
    cycles = np.asarray(controls["cycle_code"], dtype=np.int64)
    if baseline.shape != (len(score_x),) or not np.array_equal(cycles, score_cycles):
        raise ValueError("R11 champion controls are not row-aligned")
    mapped = r11.rank50_prediction(raw, score_cycles)
    prediction = r11.compose_r11_prediction(baseline, raw, score_cycles)
    return {
        "prediction": prediction, "raw_candidate": raw, "baseline_prediction": baseline,
        "fit_predictor_count": int(model.head_fit_calls), "state": model.audit_state(),
        "fit_metadata": {
            "cutoff": int(cutoff), "fit_rows": len(fit_x), "score_rows": len(score_x),
            "fit_cycle_min": int(fit_cycles.min()), "fit_cycle_max": int(fit_cycles.max()),
            "score_cycle_min": int(score_cycles.min()), "score_cycle_max": int(score_cycles.max()),
            "predictor_fit_count": int(model.head_fit_calls), "head_count": 9,
            "structured_preprocessing_fit_calls": 1, "tfidf_fit_calls": 1,
            "svd_fit_calls": 1, "validation_labels_passed_to_fit": False,
            "test_data_used": False, "pava_and_median": True,
            "rank_mapping": "rank50 within each complete cycle using average ties",
            "threshold_support": model.threshold_support,
        },
        "audit_arrays": {"candidate_rank50": mapped},
    }


def fit_r06(fit_features: pd.DataFrame, fit_target: Any,
            score_features: pd.DataFrame, *, controls: Mapping[str, Any],
            cutoff: int) -> dict[str, Any]:
    fit_x, score_x, y, fit_cycles, score_cycles = _validate_temporal_rows(
        fit_features, score_features, fit_target, cutoff=cutoff,
        feature_names=tuple(r06.FEATURE_NAMES))
    _check_controls(controls, score_x.index.to_numpy(dtype=np.int64), score_cycles, task="R06")
    baseline = np.asarray(controls["baseline_prediction"], dtype=np.float64)
    cycles = np.asarray(controls["cycle_code"], dtype=np.int64)
    if baseline.shape != (len(score_x),) or not np.array_equal(cycles, score_cycles):
        raise ValueError("R06 champion controls are not row-aligned")
    with threadpool_limits(limits=THREADS):
        model = r06.R06CharacterSparseL1Hybrid().fit(fit_x, y)
        raw = model.predict(score_x)
    mapped = r06.rank50_prediction(raw, score_cycles)
    prediction = r06.compose_r06_prediction(baseline, raw, score_cycles)
    if not np.isfinite(prediction).all():
        raise RuntimeError("R06 continuation produced nonfinite predictions")
    return {
        "prediction": prediction, "raw_candidate": raw, "baseline_prediction": baseline,
        "fit_predictor_count": 1, "state": model.audit_state(),
        "fit_metadata": {
            "cutoff": int(cutoff), "fit_rows": len(fit_x), "score_rows": len(score_x),
            "fit_cycle_min": int(fit_cycles.min()), "fit_cycle_max": int(fit_cycles.max()),
            "score_cycle_min": int(score_cycles.min()), "score_cycle_max": int(score_cycles.max()),
            "predictor_fit_count": 1, "structured_preprocessing_fit_calls": 1,
            "character_tfidf_fit_calls": 1, "validation_labels_passed_to_fit": False,
            "test_data_used": False, "rank_mapping": "rank50 within each complete cycle using average ties",
        },
        "audit_arrays": {"candidate_rank50": mapped},
    }


FIT_FUNCTIONS = {"C17": fit_c17, "C11": fit_c11, "R21": fit_r21,
                 "R11": fit_r11, "R06": fit_r06}

__all__ = ["FIT_FUNCTIONS", "NEW_CUTOFFS", "fit_c11", "fit_c17", "fit_r06", "fit_r11", "fit_r21"]
