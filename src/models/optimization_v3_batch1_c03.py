"""C03: split unigram and bigram budgets with fixed sparse LightGBM recipe."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from lightgbm import LGBMClassifier

from src.models.classification_models import _text_values
from src.models.optimization_v3_sparse_classification import PARAMS as SPARSE_PARAMS
from src.models.regression_research_text import HYBRID_FEATURES, RegressionTextModel, _array_digest, _json_digest
from src.models.round1_models import _fit_fingerprint


FEATURE_NAMES = tuple(HYBRID_FEATURES)
UNIGRAM_PARAMS = {
    "ngram_range": (1, 1), "min_df": 3, "max_df": 0.995,
    "max_features": 30000, "sublinear_tf": True, "norm": "l2",
    "lowercase": True, "strip_accents": "unicode", "dtype": np.float64,
}
BIGRAM_PARAMS = {
    "ngram_range": (2, 2), "min_df": 3, "max_df": 0.995,
    "max_features": 10000, "sublinear_tf": True, "norm": "l2",
    "lowercase": True, "strip_accents": "unicode", "dtype": np.float64,
}


def _safe_params(params: dict[str, Any]) -> dict[str, Any]:
    return {key: (list(value) if isinstance(value, tuple) else value.__name__ if isinstance(value, type) else value)
            for key, value in params.items()}


def _ordered_vocabulary(vectorizer: TfidfVectorizer) -> list[tuple[str, int]]:
    return [(str(token), int(index)) for token, index in sorted(
        vectorizer.vocabulary_.items(), key=lambda item: item[1])]


class SplitUnigramBigramClassifier:
    """Structured sparse block plus fit-only, separately capped word TF-IDF."""

    def __init__(self) -> None:
        self._fitted = False
        self.input_feature_names = FEATURE_NAMES

    def _select(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.columns.has_duplicates or set(frame.columns) != set(FEATURE_NAMES):
            raise ValueError("C03 requires the exact structured + signal_stream allowlist")
        return frame.loc[:, list(FEATURE_NAMES)].copy()

    def fit(self, frame: pd.DataFrame, target: Any) -> "SplitUnigramBigramClassifier":
        self._fitted = False
        selected = self._select(frame)
        y = np.asarray(target, dtype=np.int8)
        if y.shape != (len(selected),) or not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
            raise ValueError("C03 requires a row-aligned binary fit target containing both classes")

        (structured, self._processor, self._scaler, self._encoder,
         self._numeric_positions, self._structured_state) = RegressionTextModel._structured_fit(selected)
        documents, missing = _text_values(selected["signal_stream"])
        self._unigram = TfidfVectorizer(**dict(UNIGRAM_PARAMS))
        self._bigram = TfidfVectorizer(**dict(BIGRAM_PARAMS))
        unigram = self._unigram.fit_transform(documents).tocsr()
        bigram = self._bigram.fit_transform(documents).tocsr()
        missing_column = sparse.csr_matrix(missing.astype(np.float64).reshape(-1, 1))
        design = sparse.hstack((structured, unigram, bigram, missing_column), format="csr", dtype=np.float64)
        parameters = dict(SPARSE_PARAMS)
        parameters.update({"n_jobs": 2, "deterministic": True, "force_col_wise": True})
        self._estimator = LGBMClassifier(**parameters)
        self._estimator.fit(design, y)
        self._fit_fingerprint = _fit_fingerprint(selected, FEATURE_NAMES, y)
        self._target_sha256 = _array_digest(y)
        self._design_shape = [int(x) for x in design.shape]
        self._design_nnz = int(design.nnz)
        self._design_data_sha256 = _array_digest(design.data)
        self._vocabulary = {
            "unigram": _ordered_vocabulary(self._unigram),
            "bigram": _ordered_vocabulary(self._bigram),
        }
        self._state = {
            "id": "C03_SPLIT_UNIGRAM_BIGRAM_LGBM",
            "fit_rows": int(len(y)),
            "input_feature_names": list(FEATURE_NAMES),
            "fit_data_sha256": self._fit_fingerprint,
            "target_sha256": self._target_sha256,
            "structured_transform": self._structured_state,
            "unigram": {"parameters": _safe_params(UNIGRAM_PARAMS),
                "vocabulary_size": int(len(self._vocabulary["unigram"])),
                "vocabulary_sha256": _json_digest(self._vocabulary["unigram"]),
                "idf_sha256": _array_digest(self._unigram.idf_),
                "nnz": int(unigram.nnz)},
            "bigram": {"parameters": _safe_params(BIGRAM_PARAMS),
                "vocabulary_size": int(len(self._vocabulary["bigram"])),
                "vocabulary_sha256": _json_digest(self._vocabulary["bigram"]),
                "idf_sha256": _array_digest(self._bigram.idf_),
                "nnz": int(bigram.nnz)},
            "design_shape": self._design_shape,
            "design_nnz": self._design_nnz,
            "design_data_sha256": self._design_data_sha256,
            "lightgbm_parameters": parameters,
            "model_sha256": hashlib.sha256(self._estimator.booster_.model_to_string().encode("utf-8")).hexdigest(),
            "validation_data_passed_to_fit": False,
            "early_stopping_used": False,
            "test_data_used": False,
        }
        self._state["state_sha256"] = _json_digest(self._state)
        self._fitted = True
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("C03 model must be fitted before prediction")
        selected = self._select(frame)
        structured, unknown = RegressionTextModel._structured_transform(
            selected, self._processor, self._scaler, self._encoder, self._numeric_positions)
        documents, missing = _text_values(selected["signal_stream"])
        unigram = self._unigram.transform(documents)
        bigram = self._bigram.transform(documents)
        missing_column = sparse.csr_matrix(missing.astype(np.float64).reshape(-1, 1))
        design = sparse.hstack((structured, unigram, bigram, missing_column), format="csr", dtype=np.float64)
        # Use the underlying Booster for sparse prediction: LightGBM otherwise
        # emits a feature-name warning for scipy CSR although the column order is
        # identical and is already frozen in the fitted state.
        probability = np.asarray(self._estimator.booster_.predict(design, num_threads=2), dtype=np.float64)
        if probability.shape != (len(selected),) or not np.isfinite(probability).all():
            raise RuntimeError("C03 produced invalid probabilities")
        if np.any((probability < 0) | (probability > 1)):
            raise RuntimeError("C03 probability outside [0,1]")
        self._prediction_diagnostics = {
            "rows": int(len(probability)), "unigram_nnz": int(unigram.nnz),
            "bigram_nnz": int(bigram.nnz), "unknown_structured_categories": unknown,
            "missing_stream_rows": int(missing.sum()),
        }
        return probability

    def audit_state(self) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("C03 model must be fitted before audit")
        return json.loads(json.dumps(self._state, sort_keys=True, allow_nan=False))


def fit_predict(fit_features: pd.DataFrame, fit_target: np.ndarray,
                score_features: pd.DataFrame, *, controls: dict[str, np.ndarray],
                card_id: str, cutoff: int) -> dict[str, Any]:
    if card_id != "C03":
        raise ValueError(f"C03 adapter cannot execute {card_id}")
    model = SplitUnigramBigramClassifier().fit(fit_features, fit_target)
    raw = model.predict_proba(score_features)
    missing = np.asarray(controls["stream_missing"], dtype=bool)
    c2b = np.asarray(controls["c2b_probability"], dtype=np.float64)
    cb = np.asarray(controls["cb_probability"], dtype=np.float64)
    if not (missing.shape == c2b.shape == cb.shape == raw.shape):
        raise ValueError("C03 routing controls are not row aligned")
    prediction = np.where(missing, cb, 0.75 * c2b + 0.25 * raw)
    return {
        "prediction": prediction,
        "raw_candidate": raw,
        "fit_predictor_count": 1,
        "state": model.audit_state(),
        "fit_metadata": {**model._prediction_diagnostics, "cutoff": int(cutoff)},
        "audit_arrays": {"stream_missing": missing},
    }


__all__ = ["FEATURE_NAMES", "SplitUnigramBigramClassifier", "fit_predict"]
