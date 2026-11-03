"""Shared, fit-only transforms for the controlled positional-text pilot.

This adapter deliberately separates the five learned preprocessing units per
origin from the four estimator fits.  It reuses the existing C2_B/C1_B
preprocessing primitives but never fits estimators implicitly.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler, normalize
import warnings

from src.models.classification_models import (
    ClassificationModel,
    _ENCODER_PARAMS,
    _HGB_PARAMS,
    _LR_PARAMS,
    _TFIDF_PARAMS,
    _text_values,
)
from src.models.positional_text_features import absolute_position_features, deterministic_order_shuffle, token_thirds
from src.models.round1_models import CATEGORICAL_FEATURES, EXPANDED_FEATURES, _HGBPreprocessor


PILOT_ID = "POSITIONAL_TEXT_PILOT_V1"
ORIGINS = (67, 75)
LR_ARMS = ("bag", "position", "shuffled_position")
ALL_ARMS = (*LR_ARMS, "missing_hgb")
THREADS = 2
THRESHOLD = 0.50


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, type):
        return value.__name__
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(_jsonable(value), sort_keys=True, ensure_ascii=False,
                         allow_nan=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_array(value: Any) -> str:
    array = np.asarray(value)
    header = canonical_sha256({"shape": array.shape, "dtype": str(array.dtype)})
    digest = hashlib.sha256(header.encode("ascii"))
    if array.dtype.kind in "OUS":
        digest.update(json.dumps(array.astype(str).tolist(), ensure_ascii=False,
                                 separators=(",", ":")).encode("utf-8"))
    else:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def sha256_csr(value: sparse.spmatrix) -> str:
    matrix = sparse.csr_matrix(value, dtype=np.float64, copy=True)
    matrix.sum_duplicates()
    matrix.sort_indices()
    digest = hashlib.sha256(canonical_sha256({"shape": matrix.shape, "dtype": str(matrix.dtype)}).encode("ascii"))
    digest.update(np.asarray(matrix.indptr, dtype="<i8").tobytes())
    digest.update(np.asarray(matrix.indices, dtype="<i8").tobytes())
    digest.update(np.asarray(matrix.data, dtype="<f8").tobytes())
    return digest.hexdigest()


def validate_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.columns.has_duplicates:
        raise ValueError("Input features must be a DataFrame with unique columns")
    required = {*EXPANDED_FEATURES, "signal_stream"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Feature frame is missing required columns: {missing}")
    if "artifact_signal" in frame.columns or "threat_rank" in frame.columns:
        raise ValueError("Targets must not be included in the feature frame")
    return frame.loc[:, [*EXPANDED_FEATURES, "signal_stream"]].copy()


def fit_c2b_structured_state(frame: pd.DataFrame) -> dict[str, Any]:
    """One aggregate structured-state fit: ordinal categories plus imputation."""
    selected = validate_feature_frame(frame)
    preprocessor = _HGBPreprocessor("expanded")
    fit_matrix, categorical_mask = preprocessor.fit_transform(selected.loc[:, list(EXPANDED_FEATURES)])
    numeric_positions = [i for i in range(fit_matrix.shape[1]) if i not in preprocessor.categorical_positions]
    raw, names, numeric_names, category_names, _, _ = preprocessor._raw_matrix(
        selected.loc[:, list(EXPANDED_FEATURES)], fit_categories=False
    )
    if names != preprocessor.transformed_feature_names:
        raise RuntimeError("C2_B structured feature schema changed during state fit")
    return {
        "preprocessor": preprocessor,
        "transformed_feature_names": list(names),
        "numeric_names": list(numeric_names),
        "category_names": list(category_names),
        "numeric_positions": numeric_positions,
        "categorical_mask": list(categorical_mask),
        "fit_matrix_sha256": sha256_array(fit_matrix),
        "fit_rows": int(len(selected)),
        "preprocessor_audit_state": preprocessor.audit_state(),
    }


def transform_c2b_numeric(state: Mapping[str, Any], frame: pd.DataFrame) -> np.ndarray:
    selected = validate_feature_frame(frame)
    processed, _ = state["preprocessor"].transform(selected.loc[:, list(EXPANDED_FEATURES)])
    return np.asarray(processed[:, state["numeric_positions"]], dtype=np.float64)


def fit_c2b_scaler(state: Mapping[str, Any], frame: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(transform_c2b_numeric(state, frame))
    return scaler


def _c2b_category_tokens(state: Mapping[str, Any], frame: pd.DataFrame) -> np.ndarray:
    selected = validate_feature_frame(frame)
    _, names, _, category_names, category_raw, _ = state["preprocessor"]._raw_matrix(
        selected.loc[:, list(EXPANDED_FEATURES)], fit_categories=False
    )
    if names != state["transformed_feature_names"] or category_names != state["category_names"]:
        raise RuntimeError("C2_B category schema changed after structured-state fit")
    return ClassificationModel._category_tokens(category_raw)


def structured_state_sha256(state: Mapping[str, Any]) -> str:
    preprocessor = state["preprocessor"]
    encoder = preprocessor.encoder
    categories = [[str(value) for value in values] for values in encoder.categories_]
    infrequent = getattr(encoder, "infrequent_categories_", None)
    infrequent_payload = [None if values is None else [str(value) for value in values]
                          for values in infrequent]
    return canonical_sha256({
        "audit_state": preprocessor.audit_state(),
        "ordinal_encoder_categories": categories,
        "ordinal_encoder_infrequent_categories": infrequent_payload,
    })


def fit_c2b_ohe(state: Mapping[str, Any], frame: pd.DataFrame) -> OneHotEncoder:
    encoder = OneHotEncoder(**_ENCODER_PARAMS)
    encoder.fit(_c2b_category_tokens(state, frame))
    return encoder


def fit_c2b_tfidf(frame: pd.DataFrame) -> TfidfVectorizer:
    documents, _ = _text_values(frame["signal_stream"])
    if not any(documents):
        raise ValueError("C2_B TF-IDF has no nonempty fit-prefix stream")
    vectorizer = TfidfVectorizer(**_TFIDF_PARAMS)
    vectorizer.fit(documents)
    if not vectorizer.vocabulary_:
        raise ValueError("C2_B TF-IDF fit yielded an empty vocabulary")
    return vectorizer


def transform_c2b_structured(
    state: Mapping[str, Any], scaler: StandardScaler, encoder: OneHotEncoder,
    frame: pd.DataFrame,
) -> sparse.csr_matrix:
    selected = validate_feature_frame(frame)
    processed, _ = state["preprocessor"].transform(selected.loc[:, list(EXPANDED_FEATURES)])
    numeric = scaler.transform(np.asarray(processed[:, state["numeric_positions"]], dtype=np.float64))
    categorical = encoder.transform(_c2b_category_tokens(state, selected))
    return sparse.hstack((sparse.csr_matrix(numeric), categorical), format="csr", dtype=np.float64)


def transform_c2b_text(vectorizer: TfidfVectorizer, frame: pd.DataFrame) -> tuple[sparse.csr_matrix, np.ndarray]:
    documents, missing = _text_values(frame["signal_stream"])
    return vectorizer.transform(documents).tocsr().astype(np.float64), missing


def fit_c1b_structured_state(frame: pd.DataFrame) -> dict[str, Any]:
    """One aggregate C1_B state fit, including native categorical schema."""
    selected = validate_feature_frame(frame)
    preprocessor = _HGBPreprocessor("expanded")
    matrix, categorical_mask = preprocessor.fit_transform(selected.loc[:, list(EXPANDED_FEATURES)])
    return {
        "preprocessor": preprocessor,
        "categorical_mask": list(categorical_mask),
        "fit_matrix_sha256": sha256_array(matrix),
        "fit_rows": int(len(selected)),
        "preprocessor_audit_state": preprocessor.audit_state(),
    }


def transform_c1b_structured(state: Mapping[str, Any], frame: pd.DataFrame) -> np.ndarray:
    selected = validate_feature_frame(frame)
    matrix, _ = state["preprocessor"].transform(selected.loc[:, list(EXPANDED_FEATURES)])
    return np.asarray(matrix, dtype=np.float64)


def build_position_block(
    documents: Sequence[str], vectorizer: TfidfVectorizer, *, shuffled: bool,
    seed: int = 42,
) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    """Build third-position TF-IDF over existing unigram entries only.

    Counts map to (1+log(c))*shared_IDF, three blocks are concatenated, then
    the entire positional block is L2-normalized exactly once per row.
    """
    if not vectorizer.vocabulary_:
        return sparse.csr_matrix((len(documents), 0), dtype=np.float64), {
            "unigram_vocabulary_size": 0, "nonzero_rows": 0,
            "shuffle_seed": seed if shuffled else None,
            "rows_with_changed_descriptor": 0, "mean_occurrence_movement": 0.0,
            "bag_preserved_rows": len(documents) if shuffled else None,
            "segment_sizes_preserved_rows": len(documents) if shuffled else None,
        }
    unigram_entries = [(term, int(index)) for term, index in
                       sorted(vectorizer.vocabulary_.items(), key=lambda pair: pair[1])
                       if " " not in term]
    unigram_vocab = {term: position for position, (term, _) in enumerate(unigram_entries)}
    unigram_idf = np.asarray([vectorizer.idf_[index] for _, index in unigram_entries], dtype=np.float64)
    unigram_vectorizer = TfidfVectorizer(**{**_TFIDF_PARAMS, "ngram_range": (1, 1)})
    analyzer = unigram_vectorizer.build_analyzer()
    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    changed = 0
    movement: list[float] = []
    bag_checks = 0
    segment_checks = 0

    for row, document in enumerate(documents):
        tokens = list(analyzer(document))
        if shuffled:
            control = deterministic_order_shuffle(tokens, seed=seed)
            original_bag = Counter(tokens)
            original_segment_sizes = absolute_position_features(tokens)["segment_sizes"]
            tokens = list(control["tokens"])
            bag_checks += int(Counter(tokens) == original_bag)
            segment_checks += int(control["segment_sizes"] == original_segment_sizes)
            changed += int(control["descriptor_changed"])
            movement.append(float(control["fraction_of_occurrences_moved_between_thirds"]))
        segments = token_thirds(len(tokens))
        counts: Counter[tuple[int, int]] = Counter()
        for token, segment in zip(tokens, segments):
            block_index = unigram_vocab.get(token)
            if block_index is not None:
                counts[(segment, block_index)] += 1
        for (segment, block_index), count in sorted(counts.items()):
            rows.append(row)
            columns.append(segment * len(unigram_vocab) + block_index)
            data.append((1.0 + math.log(count)) * float(unigram_idf[block_index]))

    block = sparse.csr_matrix((np.asarray(data, dtype=np.float64),
                               (np.asarray(rows, dtype=np.int64), np.asarray(columns, dtype=np.int64))),
                              shape=(len(documents), 3 * len(unigram_vocab)), dtype=np.float64)
    block = normalize(block, norm="l2", axis=1, copy=False).tocsr()
    norms = np.sqrt(np.asarray(block.multiply(block).sum(axis=1)).ravel())
    if not np.isfinite(block.data).all() or not np.isfinite(norms).all():
        raise ValueError("Positional block contains nonfinite values")
    if np.any((norms > 0) & (np.abs(norms - 1.0) > 1e-10)):
        raise AssertionError("Combined positional blocks were not normalized once to unit L2 norm")
    return block, {
        "unigram_vocabulary_size": len(unigram_vocab),
        "nonzero_rows": int(np.count_nonzero(norms)),
        "shuffle_seed": seed if shuffled else None,
        "rows_with_changed_descriptor": changed if shuffled else None,
        "mean_occurrence_movement": float(np.mean(movement)) if movement else None,
        "bag_preserved_rows": bag_checks if shuffled else None,
        "segment_sizes_preserved_rows": segment_checks if shuffled else None,
        "block_sha256": sha256_csr(block),
    }


def build_c2b_arm_matrices(
    state: Mapping[str, Any], scaler: StandardScaler, encoder: OneHotEncoder,
    vectorizer: TfidfVectorizer, fit_frame: pd.DataFrame, score_frame: pd.DataFrame,
) -> dict[str, sparse.csr_matrix | dict[str, Any]]:
    fit_frame = validate_feature_frame(fit_frame)
    score_frame = validate_feature_frame(score_frame)
    fit_struct = transform_c2b_structured(state, scaler, encoder, fit_frame)
    score_struct = transform_c2b_structured(state, scaler, encoder, score_frame)
    fit_text, fit_missing = transform_c2b_text(vectorizer, fit_frame)
    score_text, score_missing = transform_c2b_text(vectorizer, score_frame)
    fit_base = sparse.hstack((fit_struct, fit_text,
                              sparse.csr_matrix(fit_missing.astype(np.float64).reshape(-1, 1))),
                             format="csr", dtype=np.float64)
    score_base = sparse.hstack((score_struct, score_text,
                                sparse.csr_matrix(score_missing.astype(np.float64).reshape(-1, 1))),
                               format="csr", dtype=np.float64)
    fit_docs, _ = _text_values(fit_frame["signal_stream"])
    score_docs, _ = _text_values(score_frame["signal_stream"])
    fit_position, fit_position_audit = build_position_block(fit_docs, vectorizer, shuffled=False)
    fit_shuffled, fit_shuffle_audit = build_position_block(fit_docs, vectorizer, shuffled=True, seed=42)
    score_position, score_position_audit = build_position_block(score_docs, vectorizer, shuffled=False)
    score_shuffled, score_shuffle_audit = build_position_block(score_docs, vectorizer, shuffled=True, seed=42)
    result: dict[str, sparse.csr_matrix | dict[str, Any]] = {
        "fit_base": fit_base,
        "score_base": score_base,
        "fit_position": sparse.hstack((fit_base, fit_position), format="csr", dtype=np.float64),
        "score_position": sparse.hstack((score_base, score_position), format="csr", dtype=np.float64),
        "fit_shuffled_position": sparse.hstack((fit_base, fit_shuffled), format="csr", dtype=np.float64),
        "score_shuffled_position": sparse.hstack((score_base, score_shuffled), format="csr", dtype=np.float64),
        "position_audit": {
            "fit_original": fit_position_audit,
            "fit_shuffled": fit_shuffle_audit,
            "score_original": score_position_audit,
            "score_shuffled": score_shuffle_audit,
        },
    }
    matrix_keys = ("fit_base", "score_base", "fit_position", "score_position",
                   "fit_shuffled_position", "score_shuffled_position")
    shapes = {key: tuple(result[key].shape) for key in matrix_keys}  # type: ignore[union-attr]
    if shapes["fit_position"] != shapes["fit_shuffled_position"] or shapes["score_position"] != shapes["score_shuffled_position"]:
        raise AssertionError("Position and shuffle arms have different feature dimensions")
    if sha256_csr(result["fit_position"][:, :fit_base.shape[1]]) != sha256_csr(fit_base):  # type: ignore[index,arg-type]
        raise AssertionError("Position arm changed the unordered fit bag/structured matrix")
    if sha256_csr(result["fit_shuffled_position"][:, :fit_base.shape[1]]) != sha256_csr(fit_base):  # type: ignore[index,arg-type]
        raise AssertionError("Shuffle arm changed the unordered fit bag/structured matrix")
    result["matrix_sha256"] = {key: sha256_csr(result[key]) for key in matrix_keys}  # type: ignore[arg-type]
    result["matrix_shapes"] = {key: list(value) for key, value in shapes.items()}
    result["shared_preprocessing_sha256"] = {
        "structured_state": structured_state_sha256(state),
        "scaler_mean": sha256_array(scaler.mean_),
        "scaler_scale": sha256_array(scaler.scale_),
        "ohe_categories": canonical_sha256([[str(v) for v in categories] for categories in encoder.categories_]),
        "tfidf_vocabulary_idf": canonical_sha256({
            "vocabulary": sorted((str(k), int(v)) for k, v in vectorizer.vocabulary_.items()),
            "idf_sha256": sha256_array(vectorizer.idf_),
        }),
    }
    return result


def fit_logistic(matrix: sparse.spmatrix, target: Any) -> tuple[LogisticRegression, dict[str, Any]]:
    y = _binary_target(target, int(matrix.shape[0]))
    estimator = LogisticRegression(**_LR_PARAMS)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(matrix, y)
    convergence = [str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)]
    if convergence:
        raise RuntimeError(f"C2_B LogisticRegression did not converge: {convergence}")
    if not np.array_equal(estimator.classes_, np.array([0, 1])):
        raise RuntimeError(f"Unexpected C2_B class order: {estimator.classes_.tolist()}")
    return estimator, {
        "estimator": "sklearn.linear_model.LogisticRegression",
        "parameters": dict(_LR_PARAMS),
        "classes": estimator.classes_.tolist(),
        "coef_sha256": sha256_array(estimator.coef_),
        "intercept_sha256": sha256_array(estimator.intercept_),
        "n_iter": estimator.n_iter_.tolist(),
        "convergence_warnings": convergence,
    }


def fit_missing_hgb(
    preprocessor_state: Mapping[str, Any], fit_matrix: np.ndarray, target: Any,
) -> tuple[HistGradientBoostingClassifier, dict[str, Any]]:
    y = _binary_target(target, len(fit_matrix))
    estimator = HistGradientBoostingClassifier(
        **_HGB_PARAMS, categorical_features=preprocessor_state["categorical_mask"]
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(fit_matrix, y)
    convergence = [str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)]
    if convergence:
        raise RuntimeError(f"C1_B HistGradientBoosting did not converge: {convergence}")
    if not np.array_equal(estimator.classes_, np.array([0, 1])):
        raise RuntimeError(f"Unexpected C1_B class order: {estimator.classes_.tolist()}")
    state = {
        "estimator": "sklearn.ensemble.HistGradientBoostingClassifier",
        "parameters": {**_HGB_PARAMS, "categorical_features": list(preprocessor_state["categorical_mask"])},
        "classes": estimator.classes_.tolist(),
        "n_iter": int(estimator.n_iter_),
        "convergence_warnings": convergence,
        "native_preprocessor_sha256": structured_state_sha256(preprocessor_state),
    }
    return estimator, state


def _binary_target(target: Any, expected_rows: int) -> np.ndarray:
    values = np.asarray(target)
    if values.ndim != 1 or len(values) != expected_rows:
        raise ValueError("artifact_signal must align with the fit matrix")
    try:
        numeric = values.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("artifact_signal must contain binary 0/1") from exc
    if not np.isfinite(numeric).all() or not np.isin(numeric, (0.0, 1.0)).all():
        raise ValueError("artifact_signal must contain finite binary 0/1")
    numeric = numeric.astype(np.int8)
    if not np.array_equal(np.unique(numeric), np.array([0, 1], dtype=np.int8)):
        raise ValueError("Every fit prefix must contain both artifact_signal classes")
    return numeric


def positive_probability(estimator: Any, matrix: Any) -> np.ndarray:
    if not np.array_equal(np.asarray(estimator.classes_), np.array([0, 1])):
        raise ValueError("Probability class order must be exactly [0, 1]")
    probabilities = np.asarray(estimator.predict_proba(matrix)[:, 1], dtype=np.float64)
    if probabilities.ndim != 1 or not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Estimator produced invalid positive-class probabilities")
    return probabilities


def availability_route(
    present_mask: Any, bag: Any, position: Any, shuffled: Any, missing: Any,
) -> dict[str, np.ndarray]:
    present = np.asarray(present_mask, dtype=bool)
    arrays = {"bag": np.asarray(bag, dtype=np.float64),
              "position": np.asarray(position, dtype=np.float64),
              "shuffled_position": np.asarray(shuffled, dtype=np.float64),
              "missing_hgb": np.asarray(missing, dtype=np.float64)}
    if any(values.shape != present.shape or not np.isfinite(values).all()
           or np.any((values < 0) | (values > 1)) for values in arrays.values()):
        raise ValueError("Availability route inputs must be finite aligned probabilities")
    result = {
        "bag": np.where(present, arrays["bag"], arrays["missing_hgb"]),
        "position": np.where(present, arrays["position"], arrays["missing_hgb"]),
        "shuffled_position": np.where(present, arrays["shuffled_position"], arrays["missing_hgb"]),
    }
    for name, probability in result.items():
        if not np.array_equal(probability[~present], arrays["missing_hgb"][~present]):
            raise AssertionError(f"Missing-stream route differs for {name}")
    return result


__all__ = [
    "ALL_ARMS", "LR_ARMS", "ORIGINS", "PILOT_ID", "THRESHOLD", "THREADS",
    "_ENCODER_PARAMS", "_HGB_PARAMS", "_LR_PARAMS", "_TFIDF_PARAMS",
    "availability_route", "build_c2b_arm_matrices", "canonical_sha256",
    "fit_c1b_structured_state", "fit_c2b_ohe", "fit_c2b_scaler",
    "fit_c2b_structured_state", "fit_c2b_tfidf", "fit_logistic",
    "fit_missing_hgb", "positive_probability", "sha256_array", "sha256_csr",
    "sha256_bytes", "structured_state_sha256", "transform_c1b_structured", "transform_c2b_numeric",
    "transform_c2b_structured", "transform_c2b_text", "validate_feature_frame",
]
