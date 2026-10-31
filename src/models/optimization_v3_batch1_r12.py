"""R12 cycle-grouped LambdaRank adapter for the frozen V3 queue.

All feature transforms are fitted on the current temporal fit prefix. Training
rows are stably ordered by cycle and supplied to LightGBM as contiguous query
groups. Ranker scores are never blended in their arbitrary raw units.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import lightgbm
import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from scipy import sparse
from scipy.stats import rankdata

from src.models.optimization_v2_regression import SPARSE_LGB_PARAMETERS
from src.models.regression_research_text import (
    HYBRID_FEATURES,
    RegressionTextModel,
    _array_digest,
    _safe_builtin,
    _target,
    _text_documents,
)


CARD_ID = "R12"
GRADE_COUNT = 10
LABEL_GAIN = tuple(range(GRADE_COUNT))
BLEND_WEIGHT = 0.25
BASELINE_WEIGHT = 1.0 - BLEND_WEIGHT
TARGET_RANGE = (0.0, 100.0)

# Inherit the verified V2 sparse LightGBM implementation defaults, then apply
# only the R12 queue changes. The query-dependent parameters are added per fit.
RANKER_BASE_PARAMETERS: dict[str, Any] = {
    **dict(SPARSE_LGB_PARAMETERS),
    "objective": "lambdarank",
    "n_estimators": 200,
    "num_leaves": 15,
    "min_child_samples": 40,
    "reg_lambda": 3.0,
}


class R12IntegrityError(ValueError):
    """Raised when a card input violates its frozen alignment or support rule."""


def _vector(name: str, values: Any, *, dtype: Any) -> np.ndarray:
    result = np.asarray(values, dtype=dtype)
    if result.ndim != 1:
        raise R12IntegrityError(f"{name} must be a one-dimensional vector")
    return result


def _hash_int_array(values: Any) -> str:
    array = np.asarray(values, dtype="<i8")
    if array.ndim != 1:
        raise R12IntegrityError("row-order audit arrays must be one-dimensional")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def grouped_fit_order(cycles: Any, target: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a stable cycle sort, contiguous query sizes, and integer grades.

    Returns ``(order, inverse_order, group_sizes, sorted_grades)``. ``order``
    maps sorted positions to original input positions; ``inverse_order`` maps
    original positions back to sorted positions.
    """
    cycle_values = _vector("cycles", cycles, dtype=np.int64)
    target_values = _vector("target", target, dtype=np.float64)
    if len(cycle_values) == 0 or cycle_values.shape != target_values.shape:
        raise R12IntegrityError("fit cycles and target must be nonempty and aligned")
    if not np.isfinite(target_values).all() or np.any((target_values < 0.0) | (target_values > 100.0)):
        raise R12IntegrityError("fit target must be finite and within [0,100]")

    grades = np.clip(np.floor(target_values / 10.0), 0, GRADE_COUNT - 1).astype(np.int32)
    if np.any((grades < 0) | (grades >= GRADE_COUNT)):
        raise R12IntegrityError("graded labels are outside label_gain support")
    order = np.argsort(cycle_values, kind="stable")
    sorted_cycles = cycle_values[order]
    group_cycles, group_sizes = np.unique(sorted_cycles, return_counts=True)
    if len(group_cycles) == 0 or np.any(group_sizes <= 0):
        raise R12IntegrityError("each fit cycle must form a nonempty query")
    if not np.array_equal(sorted_cycles, np.repeat(group_cycles, group_sizes)):
        raise R12IntegrityError("cycle queries are not contiguous after stable sort")
    sorted_grades = grades[order]
    for cycle, size in zip(group_cycles, group_sizes):
        local = sorted_grades[np.searchsorted(sorted_cycles, cycle):
                              np.searchsorted(sorted_cycles, cycle, side="right")]
        if int(size) < 2 or len(np.unique(local)) < 2:
            raise R12IntegrityError(f"cycle {int(cycle)} has no pairwise relevance contrast")
    inverse_order = np.argsort(order, kind="stable")
    if not np.array_equal(order[inverse_order], np.arange(len(order), dtype=np.int64)):
        raise R12IntegrityError("fit row inverse mapping does not recover input order")
    return order.astype(np.int64), inverse_order.astype(np.int64), group_sizes.astype(np.int32), sorted_grades


def cycle_average_rank_percentile(
    scores: Any,
    cycles: Any,
    *,
    expected_cycles: Sequence[int] | None = None,
    expected_cycle_counts: Any | None = None,
) -> np.ndarray:
    """Map arbitrary ranker scores to ``100 * average_rank / full_cycle_n``."""
    score_values = _vector("scores", scores, dtype=np.float64)
    cycle_values = _vector("cycles", cycles, dtype=np.int64)
    if len(score_values) == 0 or score_values.shape != cycle_values.shape:
        raise R12IntegrityError("score values and cycles must be nonempty and aligned")
    if not np.isfinite(score_values).all():
        raise R12IntegrityError("ranker scores must be finite")
    unique_cycles = np.unique(cycle_values)
    if expected_cycles is not None and not np.array_equal(
        unique_cycles, np.asarray(tuple(expected_cycles), dtype=np.int64)
    ):
        raise R12IntegrityError("score rows do not cover the expected complete future cycles")
    expected_counts = None
    if expected_cycle_counts is not None:
        expected_counts = _vector("expected_cycle_counts", expected_cycle_counts, dtype=np.int64)
        if expected_counts.shape != cycle_values.shape or np.any(expected_counts <= 0):
            raise R12IntegrityError("expected full-cycle counts are not aligned and positive")

    result = np.empty(len(score_values), dtype=np.float64)
    for cycle in unique_cycles:
        mask = cycle_values == cycle
        denominator = int(mask.sum())
        if expected_counts is not None and not np.array_equal(
            np.unique(expected_counts[mask]), np.asarray([denominator], dtype=np.int64)
        ):
            raise R12IntegrityError(f"cycle {int(cycle)} score rows are not a complete cycle")
        result[mask] = 100.0 * rankdata(score_values[mask], method="average") / denominator
    if not np.isfinite(result).all() or np.any((result < 0.0) | (result > 100.0)):
        raise R12IntegrityError("cycle percentile mapping returned invalid values")
    return result


def blend_rank_percentile(baseline_prediction: Any, rank_percentile: Any) -> np.ndarray:
    """Apply the frozen 0.75 champion + 0.25 within-cycle percentile policy."""
    baseline = _vector("baseline_prediction", baseline_prediction, dtype=np.float64)
    percentile = _vector("rank_percentile", rank_percentile, dtype=np.float64)
    if baseline.shape != percentile.shape or len(baseline) == 0:
        raise R12IntegrityError("baseline and rank percentile must be nonempty and aligned")
    if (not np.isfinite(baseline).all() or not np.isfinite(percentile).all()
            or np.any((baseline < 0.0) | (baseline > 100.0))
            or np.any((percentile < 0.0) | (percentile > 100.0))):
        raise R12IntegrityError("blend inputs must be finite and within [0,100]")
    result = np.clip(BASELINE_WEIGHT * baseline + BLEND_WEIGHT * percentile, *TARGET_RANGE)
    if not np.isfinite(result).all():
        raise R12IntegrityError("blended predictions are nonfinite")
    return result


def _select_features(frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.columns.has_duplicates:
        raise R12IntegrityError(f"{label} must be a DataFrame with unique columns")
    if set(frame.columns) != set(HYBRID_FEATURES) or len(frame.columns) != len(HYBRID_FEATURES):
        raise R12IntegrityError(f"{label} does not match the frozen HYBRID_FEATURES allowlist")
    if not frame.index.is_unique:
        raise R12IntegrityError(f"{label} index positions must be unique")
    return frame.loc[:, list(HYBRID_FEATURES)].copy()


def _fit_sparse_design(frame: pd.DataFrame):
    documents, missing = _text_documents(frame["signal_stream"])
    vectorizer, text_matrix, text_fallback = RegressionTextModel._fit_vectorizer(documents)
    structured_matrix, processor, scaler, encoder, numeric_positions, structured_state = (
        RegressionTextModel._structured_fit(frame)
    )
    missing_column = sparse.csr_matrix(missing.astype(np.float64).reshape(-1, 1))
    design = sparse.hstack((structured_matrix, text_matrix, missing_column), format="csr", dtype=np.float64)
    if design.shape[0] != len(frame) or not np.isfinite(design.data).all():
        raise R12IntegrityError("fit sparse design is misaligned or nonfinite")
    return (design, vectorizer, processor, scaler, encoder, numeric_positions,
            text_fallback, structured_state, missing)


def _transform_sparse_design(
    frame: pd.DataFrame,
    *,
    vectorizer: Any,
    processor: Any,
    scaler: Any,
    encoder: Any,
    numeric_positions: list[int],
) -> tuple[sparse.csr_matrix, dict[str, int], np.ndarray]:
    documents, missing = _text_documents(frame["signal_stream"])
    if vectorizer is None:
        text_matrix = sparse.csr_matrix((len(documents), 0), dtype=np.float64)
    else:
        text_matrix = vectorizer.transform(documents).tocsr()
    structured_matrix, unknown = RegressionTextModel._structured_transform(
        frame, processor, scaler, encoder, numeric_positions
    )
    missing_column = sparse.csr_matrix(missing.astype(np.float64).reshape(-1, 1))
    design = sparse.hstack((structured_matrix, text_matrix, missing_column), format="csr", dtype=np.float64)
    if design.shape[0] != len(frame) or not np.isfinite(design.data).all():
        raise R12IntegrityError("score sparse design is misaligned or nonfinite")
    return design, {str(key): int(value) for key, value in unknown.items()}, missing


def fit_predict(
    fit_x: pd.DataFrame,
    fit_y: Any,
    score_x: pd.DataFrame,
    *,
    controls: Mapping[str, Any],
    card_id: str,
    cutoff: int,
) -> dict[str, Any]:
    """Fit one grouped ranker and return the fixed rank-percentile blend."""
    if str(card_id) != CARD_ID:
        raise R12IntegrityError(f"R12 adapter cannot serve card {card_id!r}")
    cutoff = int(cutoff)
    if cutoff not in (55, 67):
        raise R12IntegrityError(f"unsupported R12 pilot cutoff {cutoff}")
    train = _select_features(fit_x, label="fit_x")
    score = _select_features(score_x, label="score_x")
    target = _target(fit_y, len(train))
    fit_cycles = pd.to_numeric(train["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    expected_fit_cycles = np.arange(47, cutoff + 1, dtype=np.int64)
    if not np.array_equal(np.unique(fit_cycles), expected_fit_cycles):
        raise R12IntegrityError("fit prefix cycles differ from the frozen cutoff")
    score_cycles = pd.to_numeric(score["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    expected_score_cycles = np.arange(cutoff + 1, cutoff + 13, dtype=np.int64)
    if not np.array_equal(np.unique(score_cycles), expected_score_cycles):
        raise R12IntegrityError("score rows must cover all twelve future cycles")
    if not {"baseline_prediction", "expected_score_cycle_count"}.issubset(controls):
        raise R12IntegrityError("controls omit the exact champion or full-cycle support counts")
    baseline = _vector("baseline_prediction", controls["baseline_prediction"], dtype=np.float64)
    expected_counts = _vector(
        "expected_score_cycle_count", controls["expected_score_cycle_count"], dtype=np.int64
    )
    if baseline.shape != (len(score),) or expected_counts.shape != (len(score),):
        raise R12IntegrityError("R12 controls are not aligned to score rows")

    design, vectorizer, processor, scaler, encoder, numeric_positions, text_fallback, structured_state, missing = (
        _fit_sparse_design(train)
    )
    order, inverse_order, group_sizes, sorted_grades = grouped_fit_order(fit_cycles, target)
    sorted_cycles = fit_cycles[order]
    group_cycles = np.unique(sorted_cycles)
    if not np.array_equal(group_cycles, expected_fit_cycles):
        raise R12IntegrityError("grouped query order does not cover every fit cycle exactly once")

    score_design, unknown_categories, score_missing = _transform_sparse_design(
        score, vectorizer=vectorizer, processor=processor, scaler=scaler,
        encoder=encoder, numeric_positions=numeric_positions,
    )
    score_order = np.argsort(score_cycles, kind="stable").astype(np.int64)
    inverse_score_order = np.argsort(score_order, kind="stable").astype(np.int64)
    if not np.array_equal(score_order[inverse_score_order], np.arange(len(score), dtype=np.int64)):
        raise R12IntegrityError("score row inverse mapping does not recover input order")

    parameters = {
        **RANKER_BASE_PARAMETERS,
        "label_gain": list(LABEL_GAIN),
        "lambdarank_truncation_level": int(np.max(group_sizes)),
    }
    ranker = LGBMRanker(**parameters)
    fit_design_sorted = design[order].tocsr()
    ranker.fit(fit_design_sorted, sorted_grades, group=group_sizes.tolist())
    sorted_raw_scores = np.asarray(ranker.predict(score_design[score_order].tocsr()), dtype=np.float64)
    if sorted_raw_scores.shape != (len(score),) or not np.isfinite(sorted_raw_scores).all():
        raise R12IntegrityError("LightGBM Ranker returned invalid score rows")
    raw_scores = sorted_raw_scores[inverse_score_order]
    if not np.isfinite(raw_scores).all():
        raise R12IntegrityError("inverse score mapping produced nonfinite scores")
    percentile = cycle_average_rank_percentile(
        raw_scores, score_cycles, expected_cycles=expected_score_cycles,
        expected_cycle_counts=expected_counts,
    )
    prediction = blend_rank_percentile(baseline, percentile)

    booster_text = ranker.booster_.model_to_string(num_iteration=-1)
    grade_counts = np.bincount(sorted_grades, minlength=GRADE_COUNT)
    fit_index = np.asarray(train.index, dtype=np.int64)
    score_index = np.asarray(score.index, dtype=np.int64)
    state: dict[str, Any] = {
        "card_id": CARD_ID,
        "cutoff": cutoff,
        "model_family": "lightgbm.LGBMRanker",
        "objective": "lambdarank",
        "parameters_explicit": parameters,
        "parameters_effective": ranker.get_params(deep=False),
        "label_gain": list(LABEL_GAIN),
        "query_cycles": group_cycles.tolist(),
        "query_group_sizes": group_sizes.astype(int).tolist(),
        "maximum_query_size": int(np.max(group_sizes)),
        "grade_counts": grade_counts.astype(int).tolist(),
        "fit_rows": int(len(train)),
        "fit_row_positions_sha256": _hash_int_array(fit_index),
        "sorted_fit_row_positions_sha256": _hash_int_array(fit_index[order]),
        "fit_sort_order_sha256": _hash_int_array(order),
        "fit_inverse_order_sha256": _hash_int_array(inverse_order),
        "fit_training_design_shape": [int(x) for x in design.shape],
        "fit_training_design_nnz": int(design.nnz),
        "fit_missing_stream_rows": int(missing.sum()),
        "fit_missing_stream_rate": float(missing.mean()),
        "score_rows": int(len(score)),
        "score_row_positions_sha256": _hash_int_array(score_index),
        "sorted_score_row_positions_sha256": _hash_int_array(score_index[score_order]),
        "score_sort_order_sha256": _hash_int_array(score_order),
        "score_inverse_order_sha256": _hash_int_array(inverse_score_order),
        "score_design_shape": [int(x) for x in score_design.shape],
        "score_design_nnz": int(score_design.nnz),
        "score_missing_stream_rows": int(score_missing.sum()),
        "unknown_category_rows": dict(sorted(unknown_categories.items())),
        "structured_preprocessing": structured_state,
        "tfidf": RegressionTextModel._text_state(vectorizer, text_fallback),
        "sparse_design": True,
        "dense_full_word_matrix_created": False,
        "validation_data_or_labels_passed_to_fit": False,
        "test_data_used": False,
        "predictor_fits": 1,
        "preprocessing_pipeline_fits": 1,
        "lightgbm_version": lightgbm.__version__,
        "scikit_learn_version": __import__("sklearn").__version__,
        "fitted_trees": int(ranker.booster_.current_iteration()),
        "fitted_model_sha256": hashlib.sha256(booster_text.encode("utf-8")).hexdigest(),
        "raw_scores_sorted_by_cycle_then_inverse_mapped": True,
        "score_percentile_formula": "100 * average_rank(raw_score_within_complete_cycle) / full_cycle_n",
        "blend_policy": "clip(0.75 * exact_champion + 0.25 * whole_cycle_rank_percentile, 0, 100)",
    }
    state_bytes = json.dumps(_safe_builtin(state), sort_keys=True, allow_nan=False,
                             separators=(",", ":")).encode("utf-8")
    state["state_sha256"] = hashlib.sha256(state_bytes).hexdigest()
    return {
        "prediction": prediction,
        "raw_candidate": raw_scores,
        "fit_predictor_count": 1,
        "fit_metadata": {
            "predictor_fits": 1,
            "preprocessing_pipeline_fits": 1,
            "estimator_class": "lightgbm.LGBMRanker",
            "fit_target": "integer relevance grade floor(threat_rank/10), clipped 0..9",
            "fit_query_count": int(len(group_sizes)),
            "fit_max_query_size": int(np.max(group_sizes)),
            "raw_score_units_are_unbounded": True,
        },
        "state": state,
        "audit_arrays": {
            "score_cycle": score_cycles,
            "rank_percentile": percentile,
            "expected_full_cycle_count": expected_counts,
        },
    }


def reconstruct_r12(arrays: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Independent reconstruction of the saved rank transform and fixed blend."""
    raw = _vector("raw_candidate", arrays["raw_candidate"], dtype=np.float64)
    cycles = _vector("score_cycle", arrays["audit__score_cycle"], dtype=np.int64)
    expected_counts = _vector(
        "expected_full_cycle_count", arrays["audit__expected_full_cycle_count"], dtype=np.int64
    )
    baseline = _vector("baseline_prediction", arrays["baseline_prediction"], dtype=np.float64)
    stored_percentile = _vector("rank_percentile", arrays["audit__rank_percentile"], dtype=np.float64)
    rebuilt_percentile = cycle_average_rank_percentile(
        raw, cycles, expected_cycle_counts=expected_counts
    )
    if not np.array_equal(stored_percentile, rebuilt_percentile):
        raise R12IntegrityError("saved percentile does not reconstruct from raw ranker scores")
    return blend_rank_percentile(baseline, rebuilt_percentile), baseline


__all__ = [
    "BLEND_WEIGHT", "CARD_ID", "GRADE_COUNT", "HYBRID_FEATURES", "LABEL_GAIN",
    "RANKER_BASE_PARAMETERS", "R12IntegrityError", "blend_rank_percentile",
    "cycle_average_rank_percentile", "fit_predict", "grouped_fit_order", "reconstruct_r12",
]
