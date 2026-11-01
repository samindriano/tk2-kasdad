"""C10: append a separately capped trigram TF-IDF block to the sparse classifier."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src.models.classification_models import _TFIDF_PARAMS, _text_values
from src.models.optimization_v3_sparse_classification import PARAMS as SPARSE_PARAMS
from src.models.regression_research_text import RegressionTextModel, HYBRID_FEATURES, _array_digest


CARD_ID = "C10"
THRESHOLD = 0.50
C2_WEIGHT = 0.75
CANDIDATE_WEIGHT = 0.25
TRIGRAM_PARAMS = {
    "ngram_range": (3, 3),
    "min_df": 3,
    "max_df": 0.995,
    "max_features": 10_000,
    "sublinear_tf": True,
    "lowercase": True,
    "strip_accents": "unicode",
    "dtype": np.float64,
}
FEATURE_NAMES = tuple(HYBRID_FEATURES)


class C10IntegrityError(ValueError):
    """Raised when C10's frozen design or route is malformed."""


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         allow_nan=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _vector(name: str, values: Any, *, dtype: Any) -> np.ndarray:
    result = np.asarray(values, dtype=dtype)
    if result.ndim != 1:
        raise C10IntegrityError(f"{name} must be one-dimensional")
    return result


def _probability(name: str, values: Any, length: int) -> np.ndarray:
    result = _vector(name, values, dtype=np.float64)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise C10IntegrityError(f"{name} must be finite and aligned")
    if np.any((result < 0.0) | (result > 1.0)):
        raise C10IntegrityError(f"{name} must be within [0,1]")
    return result


def _select_features(X: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")
    if X.columns.has_duplicates or set(X.columns) != set(FEATURE_NAMES) or len(X.columns) != len(FEATURE_NAMES):
        raise C10IntegrityError("C10 input must contain exactly its frozen feature allowlist")
    return X.loc[:, list(FEATURE_NAMES)].copy()


def _ordered_vocabulary_digest(vocabulary: Mapping[str, int]) -> str:
    ordered = [(str(token), int(index)) for token, index in
               sorted(vocabulary.items(), key=lambda item: item[1])]
    return _json_sha256(ordered)


def word_vectorizer() -> TfidfVectorizer:
    """Return the unchanged original word unigram/bigram vectorizer."""
    return TfidfVectorizer(**dict(_TFIDF_PARAMS))


def trigram_vectorizer() -> TfidfVectorizer:
    """Return the separate, word-three-only vectorizer frozen by C10."""
    return TfidfVectorizer(**dict(TRIGRAM_PARAMS))


class SeparateTrigramSparseClassifier:
    """Original sparse LightGBM with an additional independent trigram block."""

    input_feature_names = FEATURE_NAMES

    def fit(self, X: pd.DataFrame, y: Any) -> "SeparateTrigramSparseClassifier":
        selected = _select_features(X)
        target = np.asarray(y)
        if (target.shape != (len(selected),) or len(target) == 0
                or not np.isin(target, [0, 1]).all() or len(np.unique(target)) != 2):
            raise C10IntegrityError("C10 requires an aligned binary fit target containing both classes")

        structured, processor, scaler, encoder, numeric_positions, structured_state = (
            RegressionTextModel._structured_fit(selected)
        )
        documents, missing = _text_values(selected["signal_stream"])
        self.word = word_vectorizer()
        word_matrix = self.word.fit_transform(documents).tocsr()
        self.trigram = trigram_vectorizer()
        trigram_matrix = self.trigram.fit_transform(documents).tocsr()
        if not self.trigram.vocabulary_:
            raise C10IntegrityError("C10 has no supported trigram after min_df/max_df filtering")

        missing_matrix = sparse.csr_matrix(np.asarray(missing, dtype=np.float64).reshape(-1, 1))
        design = sparse.hstack(
            [structured, word_matrix, trigram_matrix, missing_matrix],
            format="csr", dtype=np.float64,
        )
        self.model = LGBMClassifier(**dict(SPARSE_PARAMS)).fit(design, target.astype(np.int8))
        self.processor = processor
        self.scaler = scaler
        self.encoder = encoder
        self.numeric_positions = numeric_positions
        self.state = {
            "id": CARD_ID,
            "fit_rows": int(len(target)),
            "fit_target_sha256": _array_digest(target.astype(np.int8)),
            "input_feature_names": list(FEATURE_NAMES),
            "structured_state": structured_state,
            "word_tfidf": {
                "parameters": _normalise_parameters(_TFIDF_PARAMS),
                "vocabulary_size": int(len(self.word.vocabulary_)),
                "vocabulary_sha256": _ordered_vocabulary_digest(self.word.vocabulary_),
                "idf_sha256": _array_digest(self.word.idf_),
                "shape": list(word_matrix.shape),
                "nnz": int(word_matrix.nnz),
            },
            "trigram_tfidf": {
                "parameters": _normalise_parameters(TRIGRAM_PARAMS),
                "vocabulary_size": int(len(self.trigram.vocabulary_)),
                "vocabulary_sha256": _ordered_vocabulary_digest(self.trigram.vocabulary_),
                "idf_sha256": _array_digest(self.trigram.idf_),
                "shape": list(trigram_matrix.shape),
                "nnz": int(trigram_matrix.nnz),
                "rows_with_selected_trigram": int(np.count_nonzero(np.diff(trigram_matrix.indptr))),
                "empty_trigram_rows": int(len(target) - np.count_nonzero(np.diff(trigram_matrix.indptr))),
            },
            "design": {
                "feature_block_order": ["structured", "word_1_2", "word_3", "stream_missing"],
                "shape": list(design.shape),
                "nnz": int(design.nnz),
                "data_sha256": _array_digest(design.data),
            },
            "classifier_parameters": dict(SPARSE_PARAMS),
            "model_sha256": hashlib.sha256(
                self.model.booster_.model_to_string().encode("utf-8")
            ).hexdigest(),
            "eval_set_used": False,
            "early_stopping_used": False,
        }
        self.state["state_sha256"] = _json_sha256(self.state)
        self._fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not getattr(self, "_fitted", False):
            raise C10IntegrityError("C10 model must be fitted before prediction")
        selected = _select_features(X)
        structured, _ = RegressionTextModel._structured_transform(
            selected, self.processor, self.scaler, self.encoder, self.numeric_positions
        )
        documents, missing = _text_values(selected["signal_stream"])
        word_matrix = self.word.transform(documents).tocsr()
        trigram_matrix = self.trigram.transform(documents).tocsr()
        missing_matrix = sparse.csr_matrix(np.asarray(missing, dtype=np.float64).reshape(-1, 1))
        design = sparse.hstack(
            [structured, word_matrix, trigram_matrix, missing_matrix],
            format="csr", dtype=np.float64,
        )
        probability = np.asarray(self.model.predict_proba(design)[:, 1], dtype=np.float64)
        if (probability.shape != (len(selected),) or not np.isfinite(probability).all()
                or np.any((probability < 0.0) | (probability > 1.0))):
            raise C10IntegrityError("C10 produced invalid probabilities")
        return probability

    def audit_state(self) -> dict[str, Any]:
        if not getattr(self, "_fitted", False):
            raise C10IntegrityError("C10 has no fitted state")
        return json.loads(json.dumps(self.state, allow_nan=False))


def _normalise_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in parameters.items():
        if isinstance(value, tuple):
            result[str(key)] = list(value)
        elif isinstance(value, type):
            result[str(key)] = f"{value.__module__}.{value.__qualname__}"
        elif isinstance(value, np.generic):
            result[str(key)] = value.item()
        else:
            result[str(key)] = value
    return result


def _route_candidate(
    arrays: Mapping[str, Any], raw_candidate: Any,
) -> tuple[np.ndarray, np.ndarray]:
    required = {
        "baseline_prediction", "c2b_probability", "cb_missing_probability",
        "sparse_raw_probability", "stream_missing", "source_rows",
        "source_record_ids", "score_rows", "score_record_ids",
    }
    absent = sorted(required - set(arrays))
    if absent:
        raise C10IntegrityError(f"C10 route receipt is missing fields: {absent}")
    baseline = _vector("baseline_prediction", arrays["baseline_prediction"], dtype=np.float64)
    n = len(baseline)
    raw = _probability("raw candidate", raw_candidate, n)
    c2b = _probability("C2_B", arrays["c2b_probability"], n)
    cb = _probability("CatBoost", arrays["cb_missing_probability"], n)
    sparse_raw = _probability("frozen sparse expert", arrays["sparse_raw_probability"], n)
    missing = _vector("stream_missing", arrays["stream_missing"], dtype=bool)
    source_rows = _vector("source_rows", arrays["source_rows"], dtype=np.int64)
    score_rows = _vector("score_rows", arrays["score_rows"], dtype=np.int64)
    source_ids = _vector("source_record_ids", arrays["source_record_ids"], dtype=str)
    score_ids = _vector("score_record_ids", arrays["score_record_ids"], dtype=str)
    if any(value.shape != (n,) for value in (missing, source_rows, score_rows, source_ids, score_ids)):
        raise C10IntegrityError("C10 route rows, IDs, and masks are not aligned")
    if (not np.array_equal(source_rows, score_rows) or not np.array_equal(source_ids, score_ids)
            or len(np.unique(score_rows)) != n or len(np.unique(score_ids)) != n):
        raise C10IntegrityError("C10 source cache rows or IDs differ from the ordered Train score rows")
    expected_baseline = np.where(missing, cb, C2_WEIGHT * c2b + CANDIDATE_WEIGHT * sparse_raw)
    if not np.array_equal(baseline, expected_baseline):
        raise C10IntegrityError("C10 accepted sparse25 baseline does not reconstruct exactly")
    candidate = np.where(missing, cb, C2_WEIGHT * c2b + CANDIDATE_WEIGHT * raw)
    if not np.isfinite(candidate).all() or np.any((candidate < 0.0) | (candidate > 1.0)):
        raise C10IntegrityError("C10 routed probabilities are invalid")
    if not np.array_equal(candidate[missing], cb[missing]):
        raise C10IntegrityError("C10 changed the frozen missing-stream branch")
    return candidate, expected_baseline


def fit_predict(
    fit_x: pd.DataFrame,
    fit_y: Any,
    score_x: pd.DataFrame,
    *, controls: Mapping[str, Any], card_id: str, cutoff: int,
) -> dict[str, Any]:
    """Fit exactly one C10 predictor on the temporal prefix and apply frozen routing."""
    if str(card_id) != CARD_ID or int(cutoff) not in (55, 67):
        raise C10IntegrityError(f"C10 cannot serve card={card_id!r}, cutoff={cutoff!r}")
    model = SeparateTrigramSparseClassifier()
    model.fit(fit_x, fit_y)
    raw = model.predict_proba(score_x)
    route_arrays = {
        "baseline_prediction": controls["baseline_prediction"],
        "c2b_probability": controls["c2b_probability"],
        "cb_missing_probability": controls["cb_missing_probability"],
        "sparse_raw_probability": controls["sparse_raw_probability"],
        "stream_missing": controls["stream_missing"],
        "source_rows": controls["source_rows"],
        "source_record_ids": controls["source_record_ids"],
        "score_rows": controls["source_rows"],
        "score_record_ids": controls["source_record_ids"],
    }
    candidate, baseline = _route_candidate(route_arrays, raw)
    if not np.array_equal(baseline, np.asarray(controls["baseline_prediction"], dtype=np.float64)):
        raise C10IntegrityError("C10 baseline changed inside adapter")
    return {
        "prediction": candidate,
        "raw_candidate": raw,
        "fit_predictor_count": 1,
        "fit_metadata": {
            "predictor_fits": 1,
            "preprocessing_fits": 3,
            "preprocessing_fit_breakdown": {
                "structured_preprocessor": 1,
                "word_tfidf": 1,
                "trigram_tfidf": 1,
            },
            "threshold": THRESHOLD,
        },
        "state": model.audit_state(),
        "audit_arrays": {
            "raw_candidate_probability": raw,
            "routed_probability": candidate,
        },
    }


def reconstruct_c10(arrays: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Independently rebuild the candidate route and frozen sparse25 control."""
    control_map = {
        "baseline_prediction": arrays["baseline_prediction"],
        "c2b_probability": arrays["control__c2b_probability"],
        "cb_missing_probability": arrays["control__cb_missing_probability"],
        "sparse_raw_probability": arrays["control__sparse_raw_probability"],
        "stream_missing": arrays["control__stream_missing"],
        "source_rows": arrays["control__source_rows"],
        "source_record_ids": arrays["control__source_record_ids"],
        "score_rows": arrays["score_rows"],
        "score_record_ids": arrays["score_record_ids"],
    }
    return _route_candidate(control_map, arrays["raw_candidate"])


__all__ = [
    "C10IntegrityError", "CARD_ID", "CANDIDATE_WEIGHT", "C2_WEIGHT",
    "FEATURE_NAMES", "THRESHOLD", "TRIGRAM_PARAMS", "SeparateTrigramSparseClassifier",
    "fit_predict", "reconstruct_c10", "trigram_vectorizer", "word_vectorizer",
]
