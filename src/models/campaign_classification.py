"""Fixed text-representation candidates for the dual-campaign classifier lane.

All candidates retain the audited C2_B structured transform, the same word
TF-IDF recipe, the missing-stream indicator, and the C1_B fallback supplied by
the evaluation runner. Only the present-stream expert geometry changes.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from src.models.classification_models import (
    ClassificationModel,
    _array_digest,
    _json_digest,
    _text_values,
    _to_builtin,
)


_CHAR_TFIDF_PARAMS = {
    "analyzer": "char_wb",
    "ngram_range": (3, 5),
    "min_df": 3,
    "max_features": 30000,
    "sublinear_tf": True,
    "lowercase": True,
    "strip_accents": "unicode",
    "dtype": np.float64,
}
_SVC_PARAMS = {
    "C": 1.0,
    "class_weight": None,
    "dual": "auto",
    "fit_intercept": True,
    "intercept_scaling": 1.0,
    "loss": "squared_hinge",
    "max_iter": 20000,
    "penalty": "l2",
    "random_state": 42,
    "tol": 1e-4,
}
_NB_SMOOTHING = 1.0
_RECIPE_IDS = ("DC_SVC", "DC_NBSVM", "DC_WORD_CHAR")


def _vectorizer_parameters(params: dict[str, Any]) -> dict[str, Any]:
    return {key: (list(value) if isinstance(value, tuple) else value)
            for key, value in params.items()}


def _empty_vocab_error(exc: ValueError) -> bool:
    message = str(exc).casefold()
    return any(marker in message for marker in (
        "empty vocabulary", "after pruning, no terms remain",
        "max_df corresponds to < documents than min_df",
        "max_df corresponds to less documents than min_df",
    ))


class CampaignClassificationModel(ClassificationModel):
    """A C2_B present-stream expert with one prespecified text change."""

    def __init__(self, recipe_id: str):
        if recipe_id not in _RECIPE_IDS:
            raise ValueError(f"Unknown dual-campaign recipe: {recipe_id!r}")
        super().__init__({"id": "C2_B"})
        self.model_id = recipe_id
        self.campaign_recipe = recipe_id
        self._char_vectorizer: TfidfVectorizer | None = None
        self._char_empty_reason: str | None = None
        self._nb_log_count_ratio: np.ndarray | None = None
        self._nb_counts: dict[str, int] = {}
        self._fit_labels_for_recipe: np.ndarray | None = None
        self._campaign_margin_diagnostics: dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: Any) -> "CampaignClassificationModel":
        selected = self._select(X)
        target = self._target(y, len(selected), require_both=True)
        self._char_vectorizer = None
        self._char_empty_reason = None
        self._nb_log_count_ratio = None
        self._nb_counts = {}
        self._fit_labels_for_recipe = target.copy()
        self._campaign_margin_diagnostics = {}
        try:
            super().fit(X, target)
        finally:
            self._fit_labels_for_recipe = None
        return self

    def _fit_text(self, series: pd.Series) -> sparse.csr_matrix:
        word = super()._fit_text(series)
        if self.campaign_recipe != "DC_WORD_CHAR":
            return word
        documents, _ = _text_values(series)
        if not any(documents):
            self._char_empty_reason = "all_training_text_empty"
            self._char_vectorizer = None
            char = sparse.csr_matrix((len(documents), 0), dtype=np.float64)
        else:
            vectorizer = TfidfVectorizer(**_CHAR_TFIDF_PARAMS)
            try:
                char = vectorizer.fit_transform(documents).tocsr()
            except ValueError as exc:
                if not _empty_vocab_error(exc):
                    raise
                self._char_empty_reason = "empty_after_tfidf_filters"
                self._char_vectorizer = None
                char = sparse.csr_matrix((len(documents), 0), dtype=np.float64)
            else:
                self._char_vectorizer = vectorizer
        return sparse.hstack((word, char), format="csr", dtype=np.float64)

    def _transform_text(self, series: pd.Series) -> sparse.csr_matrix:
        word = super()._transform_text(series)
        if self.campaign_recipe == "DC_NBSVM" and self._nb_log_count_ratio is not None:
            if word.shape[1] != len(self._nb_log_count_ratio):
                raise RuntimeError("NB-SVM log-count-ratio vocabulary width changed")
            return word.multiply(self._nb_log_count_ratio).tocsr()
        if self.campaign_recipe != "DC_WORD_CHAR":
            return word
        documents, _ = _text_values(series)
        if self._char_vectorizer is None:
            char = sparse.csr_matrix((len(documents), 0), dtype=np.float64)
        else:
            char = self._char_vectorizer.transform(documents).tocsr()
        return sparse.hstack((word, char), format="csr", dtype=np.float64)

    def _fit_logistic(self, matrix: Any, target: np.ndarray) -> None:
        matrix = sparse.csr_matrix(matrix, dtype=np.float64)
        if self.campaign_recipe == "DC_SVC":
            estimator = LinearSVC(**_SVC_PARAMS)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                estimator.fit(matrix, target)
            if any(issubclass(item.category, ConvergenceWarning) for item in caught):
                raise RuntimeError("DC_SVC LinearSVC did not converge")
            self._estimator = estimator
            self._campaign_design_width = int(matrix.shape[1])
            return
        if self.campaign_recipe == "DC_NBSVM":
            if self._fit_labels_for_recipe is None or len(self._fit_labels_for_recipe) != matrix.shape[0]:
                raise RuntimeError("NB-SVM class document counts are not aligned to fit rows")
            text_width = int(self._transform_state["text_columns"])
            structured_width = int(matrix.shape[1] - text_width - 1)
            if structured_width < 0:
                raise RuntimeError("C2_B structured, word, and missing blocks do not reconcile")
            raw_text = matrix[:, structured_width:structured_width + text_width].tocsr()
            present = np.asarray((raw_text != 0).sum(axis=0)).ravel().astype(float)
            labels = self._fit_labels_for_recipe
            n_positive = int((labels == 1).sum())
            n_negative = int((labels == 0).sum())
            positive = np.asarray((raw_text[labels == 1] != 0).sum(axis=0)).ravel().astype(float)
            negative = np.asarray((raw_text[labels == 0] != 0).sum(axis=0)).ravel().astype(float)
            vocab_size = int(raw_text.shape[1])
            if not (len(positive) == len(negative) == len(present) == vocab_size):
                raise RuntimeError("NB-SVM document-presence counts do not align with the word vocabulary")
            if vocab_size:
                positive_total = float(positive.sum())
                negative_total = float(negative.sum())
                positive_probability = (positive + _NB_SMOOTHING) / (
                    positive_total + _NB_SMOOTHING * vocab_size
                )
                negative_probability = (negative + _NB_SMOOTHING) / (
                    negative_total + _NB_SMOOTHING * vocab_size
                )
                ratio = np.log(positive_probability) - np.log(negative_probability)
            else:
                positive_total = 0.0
                negative_total = 0.0
                ratio = np.empty(0, dtype=float)
            self._nb_log_count_ratio = ratio
            self._nb_counts = {
                "positive_fit_documents": n_positive,
                "negative_fit_documents": n_negative,
                "word_vocabulary_size": vocab_size,
                "positive_documents_with_word_total": int(positive.sum()),
                "negative_documents_with_word_total": int(negative.sum()),
                "positive_l1_normalization_total_with_smoothing": positive_total + _NB_SMOOTHING * vocab_size,
                "negative_l1_normalization_total_with_smoothing": negative_total + _NB_SMOOTHING * vocab_size,
            }
            weighted_text = raw_text.multiply(ratio).tocsr()
            matrix = sparse.hstack((matrix[:, :structured_width], weighted_text,
                                    matrix[:, structured_width + text_width:]),
                                   format="csr", dtype=np.float64)
        super()._fit_logistic(matrix, target)

    def _make_audit_state(self) -> None:
        super()._make_audit_state()
        if self.campaign_recipe == "DC_SVC":
            parameters = self._state.get("parameters", {})
            parameters.pop("logistic_regression", None)
            parameters["linear_svc"] = dict(_SVC_PARAMS)
            self._state["decision_rule"] = {
                "score": "LinearSVC decision_function",
                "positive_class_margin": ">= 0",
                "positive_class_margin_probability_view": "expit(margin); uncalibrated monotone display only",
                "threshold_probability_view": 0.5,
            }
            self._state["coefficient_sha256"] = _array_digest(self._estimator.coef_)
            self._state["intercept"] = self._estimator.intercept_.tolist()
        elif self.campaign_recipe == "DC_NBSVM":
            self._state["nb_svm"] = {
                "method": "NB-SVM-inspired L1-normalized document-presence log-count ratio applied to fit-only word TF-IDF; unchanged LogisticRegression head",
                "smoothing_alpha": _NB_SMOOTHING,
                "counts": dict(self._nb_counts),
                "log_count_ratio_sha256": _array_digest(self._nb_log_count_ratio),
                "weighted_word_block": True,
            }
        elif self.campaign_recipe == "DC_WORD_CHAR":
            char_state = self._char_text_state()
            self._state["char_text"] = char_state
            self._state["text"]["word_plus_char_feature_count"] = int(
                self._state["text"].get("vocabulary_size", 0) + char_state["vocabulary_size"]
            )
        self._state["campaign"] = {
            "recipe_id": self.campaign_recipe,
            "base_recipe": "C2_B",
            "availability_fallback": "C1_B",
            "threshold": 0.50,
            "missing_stream_indicator_retained": True,
        }
        self._state["model_state_sha256"] = _json_digest(
            {key: value for key, value in self._state.items() if key != "model_state_sha256"}
        )

    def _char_text_state(self) -> dict[str, Any]:
        if self._char_vectorizer is None:
            return {
                "parameters": _vectorizer_parameters(_CHAR_TFIDF_PARAMS),
                "vocabulary_size": 0,
                "vocabulary_sha256": None,
                "idf_sha256": None,
                "empty_reason": self._char_empty_reason,
            }
        ordered = [(str(token), int(index)) for token, index in
                   sorted(self._char_vectorizer.vocabulary_.items(), key=lambda pair: pair[1])]
        return {
            "parameters": _vectorizer_parameters(_CHAR_TFIDF_PARAMS),
            "vocabulary_size": len(ordered),
            "vocabulary_sha256": _json_digest(ordered),
            "idf_sha256": _array_digest(self._char_vectorizer.idf_),
            "empty_reason": None,
        }

    def _linear_design(self, X: pd.DataFrame) -> sparse.csr_matrix:
        selected = self._select(X)
        structured = self._transform_structured(selected)
        text = self._transform_text(selected["signal_stream"])
        _, missing = _text_values(selected["signal_stream"])
        return sparse.hstack((structured, text,
                              sparse.csr_matrix(missing.astype(np.float64).reshape(-1, 1))),
                             format="csr", dtype=np.float64)

    def predict_margin(self, X: pd.DataFrame) -> np.ndarray:
        if self.campaign_recipe != "DC_SVC":
            raise RuntimeError("predict_margin is available only for DC_SVC")
        if not self._fitted:
            raise RuntimeError("Classifier must be fitted before predict_margin")
        selected = self._select(X)
        self._diagnostics = {"n_predictions": int(len(selected))}
        margin = np.asarray(self._estimator.decision_function(self._linear_design(selected)), dtype=float)
        if margin.shape != (len(selected),) or not np.isfinite(margin).all():
            raise RuntimeError("LinearSVC returned an invalid decision margin")
        self._campaign_margin_diagnostics = {
            "margin_min": float(margin.min()) if len(margin) else 0.0,
            "margin_max": float(margin.max()) if len(margin) else 0.0,
            "margin_positive_fraction": float(np.mean(margin >= 0)) if len(margin) else 0.0,
        }
        self._diagnostics["uncalibrated_linear_svc_margin"] = dict(self._campaign_margin_diagnostics)
        return margin

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.campaign_recipe != "DC_SVC":
            return super().predict_proba(X)
        return expit(self.predict_margin(X))

    def prediction_diagnostics(self) -> dict[str, Any]:
        state = super().prediction_diagnostics()
        if self.campaign_recipe == "DC_SVC":
            state["decision_rule"] = "expit(LinearSVC margin); .50 maps exactly to margin 0; uncalibrated monotone view"
        return state


def build_campaign_classifier(recipe_id: str) -> CampaignClassificationModel:
    """Return one fixed new text recipe; no data or files are loaded here."""
    return CampaignClassificationModel(recipe_id)
