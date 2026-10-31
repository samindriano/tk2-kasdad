"""Two fixed unigram variants of the frozen C2_B hybrid classifier.

The original ClassificationModel owns every structured transform and the
logistic-regression estimator. This subclass changes only the instance-local
TfidfVectorizer configuration; module-level Round 1 constants are never
modified.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src.models.classification_models import (
    ClassificationModel,
    _TFIDF_PARAMS,
    _array_digest,
    _json_digest,
    _text_values,
)


VARIANT_PARAMS: dict[str, dict[str, Any]] = {
    "unigram_counts": {"binary": False, "sublinear_tf": True},
    "unigram_binary": {"binary": True, "sublinear_tf": False},
}


def _json_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, type):
        return value.__name__
    if isinstance(value, np.generic):
        return value.item()
    return value


def _empty_vocabulary_error(exc: ValueError) -> bool:
    message = str(exc).casefold()
    return any(marker in message for marker in (
        "empty vocabulary",
        "after pruning, no terms remain",
        "max_df corresponds to < documents than min_df",
        "max_df corresponds to less documents than min_df",
    ))


class UnigramClassificationModel(ClassificationModel):
    """C2_B with one prespecified unigram-only word representation."""

    def __init__(self, variant: str):
        if variant not in VARIANT_PARAMS:
            raise ValueError(f"Unknown unigram variant: {variant!r}")
        super().__init__({"id": "C2_B"})
        self.variant = variant
        self.model_id = f"OV2_{variant.upper()}"
        self._variant_tfidf_params = deepcopy(_TFIDF_PARAMS)
        self._variant_tfidf_params["ngram_range"] = (1, 1)
        self._variant_tfidf_params.update(VARIANT_PARAMS[variant])

    def _fit_text(self, series: pd.Series) -> sparse.csr_matrix:
        documents, _ = _text_values(series)
        if not any(documents):
            self._fallback_all_empty = True
            self._text_vectorizer = None
            return sparse.csr_matrix((len(documents), 0), dtype=np.float64)
        vectorizer = TfidfVectorizer(**deepcopy(self._variant_tfidf_params))
        try:
            matrix = vectorizer.fit_transform(documents).tocsr()
        except ValueError as exc:
            if not _empty_vocabulary_error(exc):
                raise
            self._fallback_vocab_empty = True
            self._text_vectorizer = None
            return sparse.csr_matrix((len(documents), 0), dtype=np.float64)
        self._text_vectorizer = vectorizer
        return matrix

    def _text_state(self) -> dict[str, Any]:
        requested = _json_value(self._variant_tfidf_params)
        if self._text_vectorizer is None:
            return {
                "variant": self.variant,
                "parameters": requested,
                "effective_vectorizer_parameters": None,
                "vocabulary_size": 0,
                "vocabulary_sha256": None,
                "idf_sha256": None,
                "empty_all_training_text": self._fallback_all_empty,
                "empty_after_tfidf_filters": self._fallback_vocab_empty,
            }
        vectorizer = self._text_vectorizer
        ordered = [(str(token), int(index)) for token, index in
                   sorted(vectorizer.vocabulary_.items(), key=lambda pair: pair[1])]
        return {
            "variant": self.variant,
            "parameters": requested,
            "effective_vectorizer_parameters": _json_value(vectorizer.get_params(deep=False)),
            "vocabulary_size": len(ordered),
            "vocabulary_sha256": _json_digest(ordered),
            "idf_sha256": _array_digest(vectorizer.idf_),
            "empty_all_training_text": False,
            "empty_after_tfidf_filters": False,
        }

    def _make_audit_state(self) -> None:
        super()._make_audit_state()
        self._state["text_representation"] = {
            "variant": self.variant,
            "base_c2b_tfidf_parameters": _json_value(_TFIDF_PARAMS),
            "actual_text_state": self._text_state(),
        }
        self._state["model_state_sha256"] = _json_digest({
            key: value for key, value in self._state.items() if key != "model_state_sha256"
        })


def build_unigram_classifier(variant: str) -> UnigramClassificationModel:
    """Build one of the two fixed C2_B unigram ablation variants."""
    return UnigramClassificationModel(variant)


__all__ = ["VARIANT_PARAMS", "UnigramClassificationModel", "build_unigram_classifier"]
