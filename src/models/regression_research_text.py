"""Small, fit-only text regression recipes for the TP2 research track.

The module contains fixed TF-IDF/Ridge baselines and one HGB + truncated-SVD
variant. It does not load data, choose folds, or persist vocabularies/models.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.models.classification_models import _text_values
from src.models.regression_round2_models import _SelectedPreprocessor, _tree_complexity, _tree_digest
from src.models.round1_models import (
    CATEGORICAL_FEATURES,
    EXPANDED_FEATURES,
    _fit_fingerprint,
    _is_missing,
)

STRUCTURED_FEATURES = tuple(EXPANDED_FEATURES)
TEXT_ONLY_FEATURES = ("signal_stream",)
HYBRID_FEATURES = (*STRUCTURED_FEATURES, "signal_stream")
TEXT_KINDS = {"text_ridge", "hybrid_ridge", "svd_hgb"}

_TFIDF_PARAMS = {
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_features": 20000,
    "sublinear_tf": True,
    "dtype": np.float64,
}
_RIDGE_PARAMS = {"alpha": 10.0, "solver": "lsqr", "tol": 1e-4}
_HGB_PARAMS = {
    "loss": "absolute_error",
    "learning_rate": 0.05,
    "max_iter": 200,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 30,
    "l2_regularization": 1.0,
    "max_bins": 255,
    "early_stopping": False,
    "random_state": 42,
}
_SVD_PARAMS = {"algorithm": "randomized", "n_iter": 7, "random_state": 42}
_MISSING_MARKER = "<TP2_MISSING_CATEGORY>"


def _json_digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                     separators=(",", ":")).encode("utf-8", errors="surrogatepass")
    return hashlib.sha256(raw).hexdigest()


def _array_digest(values: Any, dtype: str = "<f8") -> str:
    arr = np.asarray(values, dtype=dtype)
    return hashlib.sha256(arr.tobytes(order="C")).hexdigest()


def _safe_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _safe_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_safe_builtin(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return _safe_builtin(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _text_documents(series: pd.Series) -> tuple[list[str], np.ndarray]:
    docs, missing = _text_values(series)
    return docs, np.asarray(missing, dtype=bool)


def _target(y: Any, n: int) -> np.ndarray:
    target = np.asarray(y)
    if target.ndim != 1 or len(target) != n or n == 0:
        raise ValueError("y must be a nonempty one-dimensional vector aligned with X")
    try:
        numeric = target.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("threat_rank must be numeric") from exc
    if not np.isfinite(numeric).all():
        raise ValueError("threat_rank must contain finite values")
    if np.any((numeric < 0) | (numeric > 100)):
        raise ValueError("threat_rank values must be within [0, 100]")
    return numeric


class RegressionTextModel:
    """One fixed text recipe with training-fold-only transforms and compact audits."""

    def __init__(self, kind: str, *, ridge_alpha: float = 10.0, svd_components: int = 32):
        if kind not in TEXT_KINDS:
            raise ValueError(f"kind must be one of {sorted(TEXT_KINDS)}")
        if ridge_alpha != 10.0:
            raise ValueError("Initial research Ridge alpha is frozen at 10.0")
        if not isinstance(svd_components, int) or isinstance(svd_components, bool) or svd_components != 32:
            raise ValueError("Initial HGB+SVD component cap is frozen at 32")
        self.kind = kind
        self.model_id = kind.upper()
        self.ridge_alpha = float(ridge_alpha)
        self.svd_components = svd_components
        self.input_feature_names = list(TEXT_ONLY_FEATURES if kind == "text_ridge" else HYBRID_FEATURES)
        self._fitted = False
        self._state: dict[str, Any] = {}
        self._diagnostics: dict[str, Any] = {"n_predictions": 0}

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(self.input_feature_names)

    def _select(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if X.columns.has_duplicates:
            raise ValueError("X contains duplicate columns")
        if set(X.columns) != set(self.input_feature_names) or len(X.columns) != len(self.input_feature_names):
            extra = sorted(set(X.columns) - set(self.input_feature_names))
            absent = sorted(set(self.input_feature_names) - set(X.columns))
            raise ValueError(f"X must contain exactly the declared features; extra={extra}, missing={absent}")
        return X.loc[:, self.input_feature_names].copy()

    @staticmethod
    def _structured_fit(X: pd.DataFrame):
        """Return scaled numeric + one-hot categorical design and fit audit objects."""
        processor = _SelectedPreprocessor(tuple(STRUCTURED_FEATURES))
        processed, _ = processor.fit_transform(X.loc[:, list(STRUCTURED_FEATURES)])
        names = processor.transformed_feature_names
        categorical_positions = list(processor.categorical_positions)
        numeric_positions = [i for i in range(len(names)) if i not in set(categorical_positions)]
        numeric_names = [names[i] for i in numeric_positions]
        numeric = np.asarray(processed[:, numeric_positions], dtype=np.float64)
        scaler = StandardScaler()
        numeric_scaled = scaler.fit_transform(numeric)

        # Derive categorical tokens from raw values with a distinct missing
        # token; fit the rare-category grouping on the current training fold.
        raw = {name: np.empty(len(X), dtype=object) for name in CATEGORICAL_FEATURES}
        for name in CATEGORICAL_FEATURES:
            for i, value in enumerate(X[name].to_numpy(dtype=object)):
                raw[name][i] = _MISSING_MARKER if _is_missing(value) else "V:" + str(value).strip()
        tokens = np.column_stack([raw[name] for name in CATEGORICAL_FEATURES])
        encoder = OneHotEncoder(
            min_frequency=30,
            max_categories=254,
            handle_unknown="infrequent_if_exist",
            sparse_output=True,
            dtype=np.float64,
        )
        categorical = encoder.fit_transform(tokens)
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
        return matrix, processor, scaler, encoder, numeric_positions, transform_state

    @staticmethod
    def _structured_transform(X: pd.DataFrame, processor: _SelectedPreprocessor,
                              scaler: StandardScaler, encoder: OneHotEncoder,
                              numeric_positions: list[int]):
        processed, unknown = processor.transform(X.loc[:, list(STRUCTURED_FEATURES)])
        numeric = scaler.transform(np.asarray(processed[:, numeric_positions], dtype=np.float64))
        raw = {name: np.empty(len(X), dtype=object) for name in CATEGORICAL_FEATURES}
        for name in CATEGORICAL_FEATURES:
            for i, value in enumerate(X[name].to_numpy(dtype=object)):
                raw[name][i] = _MISSING_MARKER if _is_missing(value) else "V:" + str(value).strip()
        tokens = np.column_stack([raw[name] for name in CATEGORICAL_FEATURES])
        categorical = encoder.transform(tokens)
        return sparse.hstack((sparse.csr_matrix(numeric), categorical), format="csr", dtype=np.float64), unknown

    @staticmethod
    def _fit_vectorizer(docs: list[str]):
        if not any(docs):
            return None, sparse.csr_matrix((len(docs), 0), dtype=np.float64), "all_training_text_empty"
        vectorizer = TfidfVectorizer(**_TFIDF_PARAMS)
        try:
            matrix = vectorizer.fit_transform(docs).tocsr()
        except ValueError as exc:
            message = str(exc).casefold()
            if not any(text in message for text in ("empty vocabulary", "after pruning, no terms remain")):
                raise
            return None, sparse.csr_matrix((len(docs), 0), dtype=np.float64), "empty_vocabulary_after_filters"
        return vectorizer, matrix, None

    @staticmethod
    def _text_state(vectorizer: TfidfVectorizer | None, fallback: str | None) -> dict[str, Any]:
        params = {
            key: (
                list(value) if isinstance(value, tuple)
                else value.__name__ if isinstance(value, type)
                else value
            )
            for key, value in _TFIDF_PARAMS.items()
        }
        if vectorizer is None:
            return {"parameters": params, "vocabulary_size": 0, "vocabulary_sha256": None,
                    "idf_sha256": None, "fallback": fallback}
        ordered = [(str(token), int(index)) for token, index in sorted(vectorizer.vocabulary_.items(), key=lambda pair: pair[1])]
        return {"parameters": params, "vocabulary_size": len(ordered),
                "vocabulary_sha256": _json_digest(ordered), "idf_sha256": _array_digest(vectorizer.idf_),
                "fallback": None}

    def fit(self, X: pd.DataFrame, y: Any) -> "RegressionTextModel":
        self._fitted = False
        selected = self._select(X)
        target = _target(y, len(selected))
        self._fit_rows = int(len(target))
        self._fit_fingerprint = _fit_fingerprint(selected, tuple(self.input_feature_names), target)
        self._target_sha256 = _array_digest(target)
        self._vectorizer = None
        self._processor = self._scaler = self._encoder = None
        self._svd = None
        self._estimator = None
        self._fallback = None
        self._numeric_positions: list[int] = []
        self._native_categorical_mask: list[bool] = []
        self._transform_audit: dict[str, Any] = {}

        docs, missing = _text_documents(selected["signal_stream"])
        self._missing_fit_rows = int(missing.sum())
        self._vectorizer, text_matrix, self._fallback = self._fit_vectorizer(docs)
        self._text_fit_shape = [int(text_matrix.shape[0]), int(text_matrix.shape[1])]
        self._missing_column_fit = sparse.csr_matrix(missing.astype(np.float64).reshape(-1, 1))

        if self.kind == "text_ridge":
            design = sparse.hstack((text_matrix, self._missing_column_fit), format="csr", dtype=np.float64)
            self._transform_audit = {"text_width": int(text_matrix.shape[1]), "missing_indicator": True}
            self._fit_ridge(design, target)
        else:
            if self.kind == "hybrid_ridge":
                structured_matrix, self._processor, self._scaler, self._encoder, self._numeric_positions, struct_state = self._structured_fit(selected)
                self._transform_audit = {"structured": struct_state, "text_width": int(text_matrix.shape[1]),
                                         "missing_indicator": True}
                design = sparse.hstack((structured_matrix, text_matrix, self._missing_column_fit),
                                       format="csr", dtype=np.float64)
                self._transform_audit["structured_width"] = int(structured_matrix.shape[1])
                self._fit_ridge(design, target)
            else:
                self._processor = _SelectedPreprocessor(tuple(STRUCTURED_FEATURES))
                structured_matrix, categorical_mask = self._processor.fit_transform(
                    selected.loc[:, list(STRUCTURED_FEATURES)]
                )
                self._native_categorical_mask = list(categorical_mask)
                self._transform_audit = {
                    "structured": self._processor.audit_state(),
                    "text_width": int(text_matrix.shape[1]),
                    "missing_indicator": True,
                    "structured_width": int(structured_matrix.shape[1]),
                    "structured_categorical_mask": list(self._native_categorical_mask),
                }
                self._svd_components_actual = 0
                if text_matrix.shape[1] > 1:
                    self._svd_components_actual = min(self.svd_components, text_matrix.shape[1] - 1)
                if self._svd_components_actual:
                    self._svd = TruncatedSVD(n_components=self._svd_components_actual, **_SVD_PARAMS)
                    text_dense = self._svd.fit_transform(text_matrix)
                else:
                    text_dense = np.empty((len(selected), 0), dtype=np.float64)
                design = np.column_stack((structured_matrix, text_dense, missing.astype(np.float64)))
                extended_categorical_mask = self._native_categorical_mask + [
                    False
                ] * (self._svd_components_actual + 1)
                self._transform_audit["design_categorical_mask"] = extended_categorical_mask
                self._transform_audit["svd_components_requested"] = self.svd_components
                self._transform_audit["svd_components_actual"] = int(self._svd_components_actual)
                self._transform_audit["svd_explained_variance_ratio_sum"] = (
                    float(np.sum(self._svd.explained_variance_ratio_)) if self._svd is not None else 0.0
                )
                self._transform_audit["svd_components_sha256"] = (
                    _array_digest(self._svd.components_) if self._svd is not None else None
                )
                params = {**_HGB_PARAMS, "categorical_features": extended_categorical_mask}
                self._estimator = HistGradientBoostingRegressor(**params)
                self._estimator.fit(design, target)
                self._model_complexity = _tree_complexity(self._estimator)
                self._tree_sha256 = _tree_digest(self._estimator)

        self._fitted = True
        self._make_state(design)
        self._diagnostics = {"n_predictions": 0}
        return self

    def _fit_ridge(self, design: sparse.spmatrix, target: np.ndarray) -> None:
        if design.shape[1] == 0:
            self._fallback = self._fallback or "zero_width_design_constant_mean"
            self._constant_prediction = float(np.mean(target))
            return
        self._estimator = Ridge(**_RIDGE_PARAMS)
        self._estimator.fit(design, target)

    def _design(self, selected: pd.DataFrame):
        docs, missing = _text_documents(selected["signal_stream"])
        if self._vectorizer is None:
            text_matrix = sparse.csr_matrix((len(docs), 0), dtype=np.float64)
            analyzer_unknown = sum(len(doc.split()) for doc in docs)
        else:
            text_matrix = self._vectorizer.transform(docs).tocsr()
            analyzer = self._vectorizer.build_analyzer()
            vocab = self._vectorizer.vocabulary_
            analyzer_unknown = sum(token not in vocab for doc in docs for token in analyzer(doc))
        if self.kind == "text_ridge":
            design = sparse.hstack((text_matrix, sparse.csr_matrix(missing.astype(float).reshape(-1, 1))),
                                   format="csr", dtype=np.float64)
            unknown = {}
        else:
            if self.kind == "hybrid_ridge":
                structured, unknown = self._structured_transform(selected, self._processor, self._scaler,
                                                                  self._encoder, self._numeric_positions)
                design = sparse.hstack((structured, text_matrix,
                                        sparse.csr_matrix(missing.astype(float).reshape(-1, 1))),
                                       format="csr", dtype=np.float64)
            else:
                # The SVD/HGB recipe keeps the Round 1 native-categorical
                # structured matrix; only the text block is reduced.
                structured, unknown = self._processor.transform(
                    selected.loc[:, list(STRUCTURED_FEATURES)]
                )
                text_dense = self._svd.transform(text_matrix) if self._svd is not None else np.empty((len(docs), 0))
                design = np.column_stack((structured, text_dense, missing.astype(float)))
        return design, missing, unknown, int(analyzer_unknown)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction")
        selected = self._select(X)
        design, missing, unknown, unknown_tokens = self._design(selected)
        if self._estimator is None:
            prediction = np.full(len(selected), self._constant_prediction, dtype=np.float64)
        else:
            prediction = np.asarray(self._estimator.predict(design), dtype=np.float64)
        if prediction.shape != (len(selected),) or not np.isfinite(prediction).all():
            raise RuntimeError("Text regression model returned invalid predictions")
        self._diagnostics = {"n_predictions": int(len(selected)),
                             "signal_stream_missing_rows": int(missing.sum()),
                             "unseen_tfidf_token_count": unknown_tokens,
                             "unknown_category_rows": dict(sorted(unknown.items()))}
        return prediction

    def _make_state(self, fit_design: Any) -> None:
        state: dict[str, Any] = {
            "id": self.model_id,
            "kind": self.kind,
            "input_feature_names": list(self.input_feature_names),
            "fit_rows": self._fit_rows,
            "fit_data_sha256": self._fit_fingerprint,
            "target_sha256": self._target_sha256,
            "fit_missing_stream_rows": self._missing_fit_rows,
            "fit_missing_stream_rate": float(self._missing_fit_rows / self._fit_rows),
            "training_design_shape": [int(fit_design.shape[0]), int(fit_design.shape[1])],
            "tfidf": self._text_state(self._vectorizer, self._fallback),
            "transform": _safe_builtin(self._transform_audit),
        }
        if self.kind in {"text_ridge", "hybrid_ridge"}:
            state["parameters"] = {"ridge": dict(_RIDGE_PARAMS), "missing_stream_indicator": True}
            if self._estimator is not None:
                state["coefficient_sha256"] = _array_digest(self._estimator.coef_)
                state["intercept"] = _safe_builtin(self._estimator.intercept_)
            else:
                state["constant_prediction"] = self._constant_prediction
        else:
            state["parameters"] = {"hgb": dict(_HGB_PARAMS), "svd": {**_SVD_PARAMS,
                                                                         "n_components_requested": self.svd_components,
                                                                         "n_components_actual": self._svd_components_actual},
                                    "missing_stream_indicator": True}
            state["model_complexity"] = self._model_complexity
            state["fitted_tree_sha256"] = self._tree_sha256
        self._state = state
        self._state["state_sha256"] = _json_digest(state)

    def audit_state(self) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("Model must be fitted before audit_state")
        return json.loads(json.dumps(_safe_builtin(self._state), sort_keys=True, allow_nan=False))

    def prediction_diagnostics(self) -> dict[str, Any]:
        return json.loads(json.dumps(_safe_builtin(self._diagnostics), sort_keys=True, allow_nan=False))


def build_text_model(kind: str, *, ridge_alpha: float = 10.0,
                     svd_components: int = 32) -> RegressionTextModel:
    """Build one fixed, reproducible Regression text candidate."""
    return RegressionTextModel(kind, ridge_alpha=ridge_alpha, svd_components=svd_components)


__all__ = ["RegressionTextModel", "build_text_model", "TEXT_ONLY_FEATURES", "STRUCTURED_FEATURES", "HYBRID_FEATURES"]
