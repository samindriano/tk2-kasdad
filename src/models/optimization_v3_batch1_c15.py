"""C15: two-score prequential logistic-regression meta-head."""
from __future__ import annotations

import warnings
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning

from src.models.classification_round2_models import stream_missing_mask


CARD_ID = "C15"
PILOT_CUTOFFS = (67, 75)
EARLY_FALLBACK_CUTOFF = 55
LOGIT_EPSILON = 1e-6
PARAMETERS = {
    "C": 1.0,
    "class_weight": None,
    "fit_intercept": True,
    "max_iter": 2000,
    "random_state": 42,
    "solver": "liblinear",
    "tol": 1e-4,
}


def clipped_logits(probability: Any, *, epsilon: float = LOGIT_EPSILON) -> np.ndarray:
    """Transform probabilities using the frozen clipping and logit rule."""
    p = np.asarray(probability, dtype=np.float64)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("C15 probabilities must be a finite one-dimensional array in [0,1]")
    if not 0.0 < float(epsilon) < 0.5:
        raise ValueError("C15 logit epsilon must be in (0,0.5)")
    q = np.clip(p, float(epsilon), 1.0 - float(epsilon))
    return np.log(q) - np.log1p(-q)


def two_score_design(c2b_probability: Any, sparse_probability: Any) -> np.ndarray:
    """Build the two-column meta design from the frozen base probabilities."""
    c2b = np.asarray(c2b_probability, dtype=np.float64)
    sparse = np.asarray(sparse_probability, dtype=np.float64)
    if c2b.ndim != 1 or sparse.shape != c2b.shape:
        raise ValueError("C15 base scores must be aligned one-dimensional arrays")
    return np.column_stack((clipped_logits(c2b), clipped_logits(sparse)))


def _check_controls(controls: Mapping[str, Any], score_rows: np.ndarray,
                    score_features: pd.DataFrame) -> dict[str, np.ndarray]:
    required = {
        "baseline_prediction", "baseline_sparse_raw", "c2b_probability",
        "cb_probability", "stream_missing", "control_rows", "text_load_bin",
    }
    absent = sorted(required - set(controls))
    if absent:
        raise ValueError(f"C15 controls omitted required fields: {absent}")
    n = len(score_rows)
    arrays: dict[str, np.ndarray] = {}
    for key in ("baseline_prediction", "baseline_sparse_raw", "c2b_probability", "cb_probability"):
        value = np.asarray(controls[key], dtype=np.float64)
        if value.shape != (n,) or not np.isfinite(value).all() or np.any((value < 0.0) | (value > 1.0)):
            raise ValueError(f"C15 control {key} is invalid or misaligned")
        arrays[key] = value
    missing = np.asarray(controls["stream_missing"], dtype=bool)
    rows = np.asarray(controls["control_rows"])
    bins = np.asarray(controls["text_load_bin"])
    if missing.shape != (n,) or not np.array_equal(missing, stream_missing_mask(score_features["signal_stream"])):
        raise ValueError("C15 stream availability does not match score features")
    if (rows.shape != (n,) or rows.dtype.kind not in "iu" or not np.array_equal(rows.astype(np.int64), score_rows)
            or len(np.unique(rows)) != n):
        raise ValueError("C15 controls do not preserve exact source row positions")
    if bins.shape != (n,) or bins.dtype.kind not in "iu" or np.any((bins < -1) | (bins > 3)):
        raise ValueError("C15 fit-defined text-load subgroup bins are invalid")
    reconstructed = np.where(missing, arrays["cb_probability"],
                             0.75 * arrays["c2b_probability"] + 0.25 * arrays["baseline_sparse_raw"])
    if not np.array_equal(arrays["baseline_prediction"], reconstructed):
        raise ValueError("C15 baseline does not reconstruct the accepted sparse25 route")
    arrays["stream_missing"] = missing
    arrays["control_rows"] = rows.astype(np.int64, copy=False)
    arrays["text_load_bin"] = bins.astype(np.int8, copy=False)
    return arrays


def fit_predict(fit_features: pd.DataFrame, fit_target: Any, score_features: pd.DataFrame,
                *, controls: Mapping[str, Any], card_id: str, cutoff: int,
                oof: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Fit one C15 meta-head from strictly past OOF rows, then route predictions."""
    if str(card_id) != CARD_ID:
        raise ValueError(f"C15 adapter cannot execute {card_id!r}")
    cutoff = int(cutoff)
    if cutoff not in (*PILOT_CUTOFFS, EARLY_FALLBACK_CUTOFF):
        raise ValueError(f"C15 cutoff {cutoff} is not authorized")
    if not isinstance(fit_features, pd.DataFrame) or not isinstance(score_features, pd.DataFrame):
        raise TypeError("C15 requires DataFrame feature inputs")
    allowed = {"cycle_code", "signal_stream"}
    if not set(fit_features.columns).issubset(allowed) or set(fit_features.columns) != allowed:
        raise ValueError("C15 fit feature allowlist must be cycle_code and signal_stream")
    if set(score_features.columns) != allowed or fit_features.columns.has_duplicates or score_features.columns.has_duplicates:
        raise ValueError("C15 score feature allowlist must be cycle_code and signal_stream")
    if (not fit_features.index.is_unique or not fit_features.index.is_monotonic_increasing
            or not score_features.index.is_unique or not score_features.index.is_monotonic_increasing):
        raise ValueError("C15 requires increasing unique official Train row positions")

    fit_rows = fit_features.index.to_numpy(dtype=np.int64)
    score_rows = score_features.index.to_numpy(dtype=np.int64)
    fit_cycles = pd.to_numeric(fit_features["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    score_cycles = pd.to_numeric(score_features["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    if (np.any((fit_cycles < 47) | (fit_cycles > cutoff))
            or np.any((score_cycles <= cutoff) | (score_cycles > cutoff + 12))):
        raise ValueError("C15 fit and scoring cycles violate the temporal cutoff")
    arrays = _check_controls(controls, score_rows, score_features)

    if cutoff == EARLY_FALLBACK_CUTOFF:
        if len(fit_target) != len(fit_rows):
            raise ValueError("C15 early fallback target is not row-aligned")
        return {
            "prediction": arrays["baseline_prediction"].copy(),
            "raw_candidate": arrays["baseline_prediction"].copy(),
            "fit_predictor_count": 0,
            "state": {
                "id": "C15_EXACT_CHAMPION_FALLBACK",
                "cutoff": cutoff,
                "reason": "No authorized sparse OOF prediction exists for cycles 56-55; early view retains current route.",
                "threshold": 0.50,
                "fit_predictor_count": 0,
            },
            "fit_metadata": {
                "cutoff": cutoff,
                "fit_rows": int(len(fit_rows)),
                "score_rows": int(len(score_rows)),
                "candidate_present_stream_rows": 0,
                "frozen_cb_missing_stream_rows": int(arrays["stream_missing"].sum()),
                "policy": "exact v3_cls_sparse25 fallback; no meta-head fit",
                "score_labels_passed_to_adapter": False,
            },
            "audit_arrays": {"stream_missing": arrays["stream_missing"], "text_load_bin": arrays["text_load_bin"]},
        }

    oof_rows = np.asarray(oof["rows"], dtype=np.int64)
    oof_c2b = np.asarray(oof["c2b_probability"], dtype=np.float64)
    oof_sparse = np.asarray(oof["sparse_probability"], dtype=np.float64)
    oof_missing = np.asarray(oof["stream_missing"], dtype=bool)
    oof_labels = np.asarray(oof["labels"], dtype=np.int8)
    oof_cycles = np.asarray(oof["cycles"], dtype=np.int64)
    oof_origins = np.asarray(oof["origin_cutoff"], dtype=np.int64)
    n = len(oof_rows)
    if not (oof_c2b.shape == oof_sparse.shape == oof_missing.shape == oof_labels.shape == oof_cycles.shape == oof_origins.shape == (n,)):
        raise ValueError("C15 OOF arrays are not row-aligned")
    if (n == 0 or len(np.unique(oof_rows)) != n or np.any(oof_rows < 0)
            or not np.isin(oof_labels, [0, 1]).all() or not np.isfinite(oof_c2b).all()
            or not np.isfinite(oof_sparse).all() or np.any((oof_c2b < 0) | (oof_c2b > 1))
            or np.any((oof_sparse < 0) | (oof_sparse > 1))):
        raise ValueError("C15 OOF values are invalid")
    oof_stream = pd.Series(np.asarray(oof["signal_stream"], dtype=object))
    if oof_stream.shape != (n,) or not np.array_equal(oof_missing, stream_missing_mask(oof_stream)):
        raise ValueError("C15 OOF availability mask does not match its signal_stream values")
    if (np.any(oof_cycles <= 55) or np.any(oof_cycles > cutoff)
            or np.any(oof_cycles <= oof_origins)):
        raise ValueError("C15 OOF score chronology is not strictly prequential")
    if np.any(~np.isin(oof_rows, fit_rows)):
        raise ValueError("C15 OOF training rows fall outside the outer fit population")
    fit_target = np.asarray(fit_target, dtype=np.int8)
    if fit_target.shape != (len(fit_rows),):
        raise ValueError("C15 outer-fit target is not row-aligned")
    positions = np.searchsorted(fit_rows, oof_rows)
    if (np.any(positions >= len(fit_rows))
            or not np.array_equal(fit_rows[positions], oof_rows)
            or not np.array_equal(fit_target[positions], oof_labels)):
        raise ValueError("C15 OOF labels do not match their official Train fit rows")

    present = ~oof_missing
    fit_classes, fit_counts = np.unique(oof_labels[present], return_counts=True)
    class_counts = {str(int(label)): int(count) for label, count in zip(fit_classes, fit_counts)}
    if len(fit_classes) != 2:
        raise ValueError("C15 meta-head requires both classes in present-stream OOF rows")
    if len(oof_rows) < 5_000 or min(class_counts.values()) < 1_000:
        raise ValueError(f"C15 OOF support no longer meets frozen minimum: n={len(oof_rows)}, classes={class_counts}")

    fit_x = two_score_design(oof_c2b[present], oof_sparse[present])
    estimator = LogisticRegression(**PARAMETERS)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        estimator.fit(fit_x, oof_labels[present])
    fit_warnings = [{"category": item.category.__name__, "message": str(item.message)}
                    for item in captured]
    score_missing = arrays["stream_missing"]
    raw = arrays["baseline_prediction"].copy()
    present_score = ~score_missing
    if present_score.any():
        score_x = two_score_design(arrays["c2b_probability"][present_score],
                                   arrays["baseline_sparse_raw"][present_score])
        raw[present_score] = estimator.predict_proba(score_x)[:, 1]
    prediction = np.where(score_missing, arrays["cb_probability"], raw)
    state = {
        "id": "C15_TWO_SCORE_PREQUENTIAL_LOGISTIC_REGRESSION",
        "cutoff": cutoff,
        "fit_predictor_count": 1,
        "input_score_order": ["clipped_logit(C2_B probability)", "clipped_logit(raw sparse-binary probability)"],
        "logit_epsilon": LOGIT_EPSILON,
        "logistic_regression_parameters": {**PARAMETERS, "penalty": "l2 (scikit-learn default)"},
        "oof_fit_rows": int(n),
        "oof_present_fit_rows": int(present.sum()),
        "oof_present_class_counts": class_counts,
        "oof_cycle_min": int(oof_cycles.min()),
        "oof_cycle_max": int(oof_cycles.max()),
        "oof_origin_cutoffs": {str(int(origin)): int(np.sum(oof_origins == origin))
                               for origin in np.unique(oof_origins)},
        "coef": estimator.coef_.astype(float).tolist(),
        "intercept": estimator.intercept_.astype(float).tolist(),
        "n_iter": estimator.n_iter_.astype(int).tolist(),
        "converged": bool(np.all(estimator.n_iter_ < int(PARAMETERS["max_iter"]))
                          and not any(item.category is ConvergenceWarning for item in captured)),
        "fit_warnings": fit_warnings,
        "threshold": 0.50,
        "missing_stream_route": "exact frozen CatBoost probability",
        "present_stream_route": "C15 logistic regression on the two clipped-logit scores",
        "score_labels_passed_to_fit": False,
        "test_data_used": False,
    }
    return {
        "prediction": prediction,
        "raw_candidate": raw,
        "fit_predictor_count": 1,
        "state": state,
        "fit_metadata": {
            "cutoff": cutoff,
            "fit_rows": int(len(fit_rows)),
            "score_rows": int(len(score_rows)),
            "oof_meta_rows": int(n),
            "oof_present_meta_rows": int(present.sum()),
            "oof_present_class_counts": class_counts,
            "n_iter": estimator.n_iter_.astype(int).tolist(),
            "fit_warnings": fit_warnings,
            "candidate_present_stream_rows": int(present_score.sum()),
            "frozen_cb_missing_stream_rows": int(score_missing.sum()),
            "threshold": 0.50,
            "policy": "C15 LR on present; exact CatBoost route on missing",
            "score_labels_passed_to_adapter": False,
        },
        "audit_arrays": {"stream_missing": score_missing, "text_load_bin": arrays["text_load_bin"]},
    }


__all__ = [
    "CARD_ID", "EARLY_FALLBACK_CUTOFF", "LOGIT_EPSILON", "PARAMETERS",
    "PILOT_CUTOFFS", "clipped_logits", "fit_predict", "two_score_design",
]
