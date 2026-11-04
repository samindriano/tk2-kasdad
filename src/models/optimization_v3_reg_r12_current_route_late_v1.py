"""Frozen R12 late-history ranker with a current-R06 availability route.

This module is isolated from the historical R12 runner. It retains the R12
ranker and full-cycle average-rank mapping, while using the exact saved R06
prediction as the present-row blend base and as the missing-row fallback.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Mapping, Sequence

import lightgbm
import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from scipy import sparse
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.models.optimization_v3_batch1_r12 import (
    BLEND_WEIGHT,
    LABEL_GAIN,
    RANKER_BASE_PARAMETERS,
    R12IntegrityError,
    _hash_int_array,
    _select_features,
    blend_rank_percentile,
    cycle_average_rank_percentile,
    grouped_fit_order,
)
from src.models.regression_research_text import (
    CATEGORICAL_FEATURES,
    HYBRID_FEATURES,
    STRUCTURED_FEATURES,
    RegressionTextModel,
    _MISSING_MARKER,
    _TFIDF_PARAMS,
    _array_digest,
    _is_missing,
    _json_digest,
    _target,
    _text_documents,
)
from src.models.regression_round2_models import _SelectedPreprocessor


NAMESPACE = "REG_R12_CURRENT_ROUTE_LATE_V1"
ORIGINS = (71, 75)
SCORE_CYCLES = {71: tuple(range(72, 80)), 75: tuple(range(76, 80))}
RANKER_MAX_FITS = 2
PREPROCESSING_MAX_FITS = 8


def _vector(name: str, values: Any, *, dtype: Any) -> np.ndarray:
    result = np.asarray(values, dtype=dtype)
    if result.ndim != 1:
        raise R12IntegrityError(f"{name} must be a one-dimensional vector")
    return result


def _fit_text_with_unit_log(
    documents: list[str],
    *,
    unit_start: Callable[[str, Mapping[str, Any]], str],
    unit_finish: Callable[[str, Mapping[str, Any]], None],
) -> tuple[Any, sparse.csr_matrix, str | None]:
    """Mirror RegressionTextModel's fixed word TF-IDF fit with durable starts."""
    if not any(documents):
        return None, sparse.csr_matrix((len(documents), 0), dtype=np.float64), "all_training_text_empty"
    vectorizer = TfidfVectorizer(**_TFIDF_PARAMS)
    unit_id = unit_start("word_tfidf", {"fit_rows": len(documents), "parameters": RegressionTextModel._text_state(None, None)["parameters"]})
    try:
        matrix = vectorizer.fit_transform(documents).tocsr()
    except ValueError as exc:
        message = str(exc).casefold()
        if not any(text in message for text in ("empty vocabulary", "after pruning, no terms remain")):
            unit_finish(unit_id, {"status": "FAILED", "error_type": type(exc).__name__})
            raise
        unit_finish(unit_id, {"status": "EMPTY_VOCABULARY_FALLBACK"})
        return None, sparse.csr_matrix((len(documents), 0), dtype=np.float64), "empty_vocabulary_after_filters"
    unit_finish(unit_id, {"status": "COMPLETED", "matrix_shape": list(matrix.shape), "nnz": int(matrix.nnz)})
    return vectorizer, matrix, None


def _structured_fit_with_unit_log(
    frame: pd.DataFrame,
    *,
    unit_start: Callable[[str, Mapping[str, Any]], str],
    unit_finish: Callable[[str, Mapping[str, Any]], None],
):
    """Instrument the exact R12 structured fit sequence without changing it."""
    processor = _SelectedPreprocessor(tuple(STRUCTURED_FEATURES))
    processor_unit = unit_start("structured_processor", {"fit_rows": len(frame), "features": list(STRUCTURED_FEATURES)})
    try:
        processed, _ = processor.fit_transform(frame.loc[:, list(STRUCTURED_FEATURES)])
    except Exception as exc:
        unit_finish(processor_unit, {"status": "FAILED", "error_type": type(exc).__name__})
        raise
    names = processor.transformed_feature_names
    categorical_positions = list(processor.categorical_positions)
    numeric_positions = [i for i in range(len(names)) if i not in set(categorical_positions)]
    numeric_names = [names[i] for i in numeric_positions]
    numeric = np.asarray(processed[:, numeric_positions], dtype=np.float64)

    scaler = StandardScaler()
    scaler_unit = unit_start("numeric_scaler", {"fit_rows": len(frame), "numeric_width": int(numeric.shape[1])})
    try:
        numeric_scaled = scaler.fit_transform(numeric)
    except Exception as exc:
        unit_finish(scaler_unit, {"status": "FAILED", "error_type": type(exc).__name__})
        raise
    unit_finish(scaler_unit, {"status": "COMPLETED", "feature_count": int(numeric.shape[1])})

    raw = {name: np.empty(len(frame), dtype=object) for name in CATEGORICAL_FEATURES}
    for name in CATEGORICAL_FEATURES:
        for i, value in enumerate(frame[name].to_numpy(dtype=object)):
            raw[name][i] = _MISSING_MARKER if _is_missing(value) else "V:" + str(value).strip()
    tokens = np.column_stack([raw[name] for name in CATEGORICAL_FEATURES])
    encoder = OneHotEncoder(
        min_frequency=30,
        max_categories=254,
        handle_unknown="infrequent_if_exist",
        sparse_output=True,
        dtype=np.float64,
    )
    encoder_unit = unit_start(
        "categorical_ohe",
        {"fit_rows": len(frame), "min_frequency": 30, "max_categories": 254,
          "handle_unknown": "infrequent_if_exist", "dtype": "float64"},
    )
    try:
        categorical = encoder.fit_transform(tokens)
    except Exception as exc:
        unit_finish(encoder_unit, {"status": "FAILED", "error_type": type(exc).__name__})
        raise
    unit_finish(encoder_unit, {"status": "COMPLETED", "output_width": int(categorical.shape[1])})
    matrix = sparse.hstack((sparse.csr_matrix(numeric_scaled), categorical), format="csr", dtype=np.float64)
    category_counts = {}
    for name, values in zip(CATEGORICAL_FEATURES, encoder.categories_):
        raw_values = raw[name]
        category_counts[name] = {
            "vocabulary_size": int(len(values)),
            "missing_fit_rows": int(np.sum(raw_values == _MISSING_MARKER)),
            "fit_rows": int(len(raw_values)),
            "vocabulary_sha256": _json_digest([str(v) for v in values]),
        }
    transform_state = {
        "feature_names": names,
        "numeric_feature_names": numeric_names,
        "numeric_positions": numeric_positions,
        "categorical_positions": categorical_positions,
        "numeric_imputer_medians": dict(processor.medians),
        "numeric_scaler_mean": _array_digest(scaler.mean_),
        "numeric_scaler_scale": _array_digest(scaler.scale_),
        "category_counts": category_counts,
        "ohe_output_width": int(categorical.shape[1]),
    }
    unit_finish(processor_unit, {"status": "COMPLETED", "transformed_width": int(processed.shape[1])})
    return matrix, processor, scaler, encoder, numeric_positions, transform_state


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


def build_routes(
    raw_candidate: Any,
    score_cycles: Any,
    current_r06: Any,
    present_mask: Any,
    legacy_baseline: Any,
    *,
    expected_cycles: Sequence[int],
    expected_cycle_counts: Any,
) -> dict[str, np.ndarray]:
    """Create the frozen candidate, mapping-only control, and optional old route."""
    raw = _vector("raw_candidate", raw_candidate, dtype=np.float64)
    cycles = _vector("score_cycles", score_cycles, dtype=np.int64)
    r06 = _vector("current_r06", current_r06, dtype=np.float64)
    present = _vector("present_mask", present_mask, dtype=bool)
    legacy = _vector("legacy_baseline", legacy_baseline, dtype=np.float64)
    if not (raw.shape == cycles.shape == r06.shape == present.shape == legacy.shape):
        raise R12IntegrityError("route inputs are not row-aligned")
    counts = _vector("expected_cycle_counts", expected_cycle_counts, dtype=np.int64)
    r12_percentile = cycle_average_rank_percentile(
        raw, cycles, expected_cycles=expected_cycles, expected_cycle_counts=counts
    )
    r06_percentile = cycle_average_rank_percentile(
        r06, cycles, expected_cycles=expected_cycles, expected_cycle_counts=counts
    )
    candidate_present = blend_rank_percentile(r06, r12_percentile)
    control_present = blend_rank_percentile(r06, r06_percentile)
    original_present = blend_rank_percentile(legacy, r12_percentile)
    return {
        "rank_percentile": r12_percentile,
        "mapping_only_rank_percentile": r06_percentile,
        "candidate": np.where(present, candidate_present, r06),
        "mapping_only_control": np.where(present, control_present, r06),
        "original_legacy_route": np.where(present, original_present, legacy),
    }


def fit_predict_current_route(
    fit_x: pd.DataFrame,
    fit_y: Any,
    score_x: pd.DataFrame,
    *,
    current_r06: Any,
    legacy_baseline: Any,
    expected_cycle_counts: Any,
    cutoff: int,
    unit_start: Callable[[str, Mapping[str, Any]], str],
    unit_finish: Callable[[str, Mapping[str, Any]], None],
) -> dict[str, Any]:
    """Fit exactly one R12 ranker at origin 71 or 75 and map its raw scores."""
    cutoff = int(cutoff)
    if cutoff not in ORIGINS:
        raise R12IntegrityError(f"unsupported late R12 origin {cutoff}")
    train = _select_features(fit_x, label="fit_x")
    score = _select_features(score_x, label="score_x")
    target = _target(fit_y, len(train))
    fit_cycles = pd.to_numeric(train["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    expected_fit_cycles = np.arange(47, cutoff + 1, dtype=np.int64)
    if not np.array_equal(np.unique(fit_cycles), expected_fit_cycles):
        raise R12IntegrityError("fit prefix cycles differ from the frozen late-origin cutoff")
    score_cycles = pd.to_numeric(score["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    expected_score_cycles = np.asarray(SCORE_CYCLES[cutoff], dtype=np.int64)
    if not np.array_equal(np.unique(score_cycles), expected_score_cycles):
        raise R12IntegrityError("score rows do not cover the frozen complete future cycles")
    r06 = _vector("current_r06", current_r06, dtype=np.float64)
    legacy = _vector("legacy_baseline", legacy_baseline, dtype=np.float64)
    counts = _vector("expected_cycle_counts", expected_cycle_counts, dtype=np.int64)
    if r06.shape != (len(score),) or legacy.shape != (len(score),) or counts.shape != (len(score),):
        raise R12IntegrityError("saved controls are not aligned to score rows")

    documents, missing = _text_documents(train["signal_stream"])
    vectorizer, text_matrix, text_fallback = _fit_text_with_unit_log(
        documents, unit_start=unit_start, unit_finish=unit_finish
    )
    structured_matrix, processor, scaler, encoder, numeric_positions, structured_state = (
        _structured_fit_with_unit_log(train, unit_start=unit_start, unit_finish=unit_finish)
    )
    missing_column = sparse.csr_matrix(missing.astype(np.float64).reshape(-1, 1))
    design = sparse.hstack((structured_matrix, text_matrix, missing_column), format="csr", dtype=np.float64)
    if design.shape[0] != len(train) or not np.isfinite(design.data).all():
        raise R12IntegrityError("fit sparse design is misaligned or nonfinite")

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
    effective = ranker.get_params(deep=False)
    if any(effective.get(key) != value for key, value in parameters.items()):
        raise R12IntegrityError("LightGBM constructor did not retain the frozen R12 parameters")
    fit_design_sorted = design[order].tocsr()
    fit_unit = unit_start(
        "lgbm_ranker_predictor",
        {"origin": cutoff, "fit_rows": len(train), "fit_cycles": group_cycles.tolist(),
          "max_query_size": int(np.max(group_sizes)), "parameters": parameters},
    )
    try:
        ranker.fit(fit_design_sorted, sorted_grades, group=group_sizes.tolist())
    except Exception as exc:
        unit_finish(fit_unit, {"status": "FAILED", "error_type": type(exc).__name__})
        raise
    unit_finish(fit_unit, {"status": "COMPLETED", "trees_fitted": int(ranker.booster_.current_iteration())})

    sorted_raw_scores = np.asarray(ranker.predict(score_design[score_order].tocsr()), dtype=np.float64)
    if sorted_raw_scores.shape != (len(score),) or not np.isfinite(sorted_raw_scores).all():
        raise R12IntegrityError("LightGBM Ranker returned invalid score rows")
    raw_scores = sorted_raw_scores[inverse_score_order]
    if not np.isfinite(raw_scores).all():
        raise R12IntegrityError("inverse score mapping produced nonfinite scores")
    present = score["signal_stream"].notna().to_numpy(dtype=bool) & (
        score["signal_stream"].fillna("").astype(str).str.strip().ne("").to_numpy(dtype=bool)
    )
    routes = build_routes(
        raw_scores,
        score_cycles,
        r06,
        present,
        legacy,
        expected_cycles=expected_score_cycles,
        expected_cycle_counts=counts,
    )

    booster_text = ranker.booster_.model_to_string(num_iteration=-1)
    grade_counts = np.bincount(sorted_grades, minlength=len(LABEL_GAIN))
    fit_index = np.asarray(train.index, dtype=np.int64)
    score_index = np.asarray(score.index, dtype=np.int64)
    preprocessing_fit_calls = (
        ([] if text_fallback == "all_training_text_empty" else ["word_tfidf"])
        + ["structured_processor", "numeric_scaler", "categorical_ohe"]
    )
    state: dict[str, Any] = {
        "namespace": NAMESPACE,
        "cutoff": cutoff,
        "model_family": "lightgbm.LGBMRanker",
        "objective": "lambdarank",
        "parameters_explicit": parameters,
        "parameters_effective": ranker.get_params(deep=False),
        "label_gain": list(LABEL_GAIN),
        "grade_definition": "clip(floor(threat_rank/10),0,9)",
        "query_cycles": group_cycles.tolist(),
        "query_group_sizes": group_sizes.astype(int).tolist(),
        "maximum_query_size": int(np.max(group_sizes)),
        "grade_counts": grade_counts.astype(int).tolist(),
        "fit_rows": int(len(train)),
        "fit_row_positions_sha256": _hash_int_array(fit_index),
        "sorted_fit_row_positions_sha256": _hash_int_array(fit_index[order]),
        "fit_sort_order_sha256": _hash_int_array(order),
        "fit_inverse_order_sha256": _hash_int_array(inverse_order),
        "fit_target_sha256": _array_digest(target),
        "fit_training_design_shape": [int(x) for x in design.shape],
        "fit_training_design_nnz": int(design.nnz),
        "fit_missing_stream_rows": int(missing.sum()),
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
        "validation_labels_passed_to_fit": False,
        "test_data_used": False,
        "predictor_fit_count": 1,
        "preprocessing_fit_calls": preprocessing_fit_calls,
        "lightgbm_version": lightgbm.__version__,
        "scikit_learn_version": __import__("sklearn").__version__,
        "fitted_trees": int(ranker.booster_.current_iteration()),
        "fitted_model_sha256": hashlib.sha256(booster_text.encode("utf-8")).hexdigest(),
        "raw_scores_sorted_by_cycle_then_inverse_mapped": True,
        "score_percentile_formula": "100*average_rank(raw_candidate_within_complete_cycle)/full_cycle_n",
        "policy": "present=clip(.75*current_R06+.25*R12_rank_percentile,0,100); missing=exact_current_R06",
    }
    state_bytes = json.dumps(state, sort_keys=True, allow_nan=False, separators=(",", ":")).encode("utf-8")
    state["state_sha256"] = hashlib.sha256(state_bytes).hexdigest()
    return {
        "raw_candidate": raw_scores,
        "present_mask": present,
        **routes,
        "state": state,
        "audit_arrays": {
            "score_cycle": score_cycles,
            "expected_full_cycle_count": counts,
            "r06_rank_percentile": routes["mapping_only_rank_percentile"],
            "rank_percentile": routes["rank_percentile"],
        },
        "fit_rows": int(len(train)),
        "score_rows": int(len(score)),
        "predictor_fit_count": 1,
        "preprocessing_fit_count": len(preprocessing_fit_calls),
        "fitted_model": booster_text.encode("utf-8"),
    }


__all__ = [
    "NAMESPACE",
    "ORIGINS",
    "SCORE_CYCLES",
    "RANKER_MAX_FITS",
    "PREPROCESSING_MAX_FITS",
    "build_routes",
    "fit_predict_current_route",
]
