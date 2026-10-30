"""Leakage-safe classifier wrappers for the TP2 Classification Round 1.

The wrappers implement only the fixed, interpretable recipe families. They do
not load data, choose folds, optimize decision thresholds, or write model
artifacts; the experiment runner owns those decisions.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import warnings

from src.models.round1_models import (
    CATEGORICAL_FEATURES,
    EXPANDED_FEATURES,
    TWO_FEATURES,
    _HGBPreprocessor,
    _fit_fingerprint,
    _is_missing,
    _number_array,
    _to_builtin,
)


_RECIPES = {
    "C0_A": ("majority", ("entry_friction",)),
    "C0_B": ("ef_lookup", ("entry_friction",)),
    "C0_C": ("lr_two", TWO_FEATURES),
    "C1_A": ("lr_structured", EXPANDED_FEATURES),
    "C1_B": ("hgb_structured", EXPANDED_FEATURES),
    "C2_A": ("lr_text", ("signal_stream",)),
    "C2_B": ("lr_hybrid", (*EXPANDED_FEATURES, "signal_stream")),
}
_LOOKUP_MIN_SUPPORT = 30
_LR_PARAMS = {
    "C": 1.0,
    "solver": "liblinear",
    "max_iter": 2000,
    "tol": 1e-4,
    "class_weight": None,
    "random_state": 42,
}
_HGB_PARAMS = {
    "loss": "log_loss",
    "learning_rate": 0.05,
    "max_iter": 200,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 30,
    "l2_regularization": 1.0,
    "max_bins": 255,
    "early_stopping": False,
    "random_state": 42,
}
_TFIDF_PARAMS = {
    "ngram_range": (1, 2),
    "min_df": 3,
    "max_df": 0.995,
    "max_features": 20000,
    "sublinear_tf": True,
    "lowercase": True,
    "strip_accents": "unicode",
    "dtype": np.float64,
}
_ENCODER_PARAMS = {
    "min_frequency": 30,
    "max_categories": 254,
    "handle_unknown": "infrequent_if_exist",
    "sparse_output": True,
    "dtype": np.float64,
}


def _json_digest(value: Any) -> str:
    payload = json.dumps(_to_builtin(value), sort_keys=True, ensure_ascii=False, allow_nan=False,
                         separators=(",", ":")).encode("utf-8", errors="surrogatepass")
    return hashlib.sha256(payload).hexdigest()


def _array_digest(value: Any, dtype: str = "<f8") -> str:
    array = np.asarray(value, dtype=dtype)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _text_values(series: pd.Series) -> tuple[list[str], np.ndarray]:
    """Convert missing/blank stream entries to empty text and flag them."""
    values: list[str] = []
    missing = np.zeros(len(series), dtype=bool)
    for index, value in enumerate(series.to_numpy(dtype=object)):
        if _is_missing(value):
            values.append("")
            missing[index] = True
        else:
            text = str(value).strip()
            values.append(text)
            missing[index] = not bool(text)
    return values, missing


class ClassificationModel:
    """One frozen recipe that exposes positive-class probabilities and audits."""

    def __init__(self, recipe: dict[str, Any]):
        if not isinstance(recipe, dict):
            raise TypeError("recipe must be a dictionary")
        if set(recipe) - {"id", "kind", "params"}:
            raise ValueError("Classification recipe accepts only id, kind, and params")
        recipe_id = recipe.get("id")
        if recipe_id not in _RECIPES:
            raise ValueError(f"Unknown classification Round 1 recipe: {recipe_id!r}")
        kind, fields = _RECIPES[recipe_id]
        if recipe.get("kind", kind) != kind:
            raise ValueError(f"{recipe_id} kind is frozen as {kind!r}")
        if recipe.get("params", {}):
            raise ValueError("Round 1 classifier recipes do not accept parameter overrides")
        self.model_id = recipe_id
        self.kind = kind
        self.input_feature_names = list(fields)
        self._fitted = False
        self._state: dict[str, Any] = {}
        self._diagnostics: dict[str, Any] = {"n_predictions": 0}

    def _select(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame for feature auditing")
        if X.columns.has_duplicates:
            raise ValueError("X has duplicate feature columns")
        missing = [name for name in self.input_feature_names if name not in X.columns]
        if missing:
            raise ValueError(f"X is missing required classification features: {missing}")
        if self.input_feature_names:
            return X.loc[:, self.input_feature_names]
        return X.iloc[:, :0]

    @staticmethod
    def _target(y: Any, rows: int, *, require_both: bool) -> np.ndarray:
        target = np.asarray(y)
        if target.ndim != 1 or len(target) != rows or rows == 0:
            raise ValueError("y must be nonempty, one-dimensional, and aligned with X")
        try:
            numeric = target.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError("artifact_signal must contain binary labels 0/1") from exc
        if not np.isfinite(numeric).all() or not np.isin(numeric, (0.0, 1.0)).all():
            raise ValueError("artifact_signal must contain finite binary labels 0/1")
        classes = np.unique(numeric)
        if require_both and not np.array_equal(classes, np.array([0.0, 1.0])):
            raise ValueError("Logistic and HGB classifiers require both training classes")
        return numeric.astype(np.int8)

    def fit(self, X: pd.DataFrame, y: Any) -> "ClassificationModel":
        self._fitted = False
        selected = self._select(X)
        target = self._target(y, len(selected), require_both=self.kind not in {"majority", "ef_lookup"})
        self._fit_rows = int(len(target))
        self._fit_fingerprint = _fit_fingerprint(selected, tuple(self.input_feature_names), target)
        self._positive_rate = float(target.mean())
        self._estimator = None
        self._structured_preprocessor = None
        self._text_vectorizer = None
        self._linear_scaler = None
        self._linear_encoder = None
        self._lookup_rates: dict[str, float] = {}
        self._lookup_counts: dict[str, int] = {}
        self._fallback_all_empty = False
        self._fallback_vocab_empty = False
        self._transform_state: dict[str, Any] = {}
        self._hgb_categorical_mask: list[bool] = []
        self._native_vocabulary_hashes: dict[str, str] = {}

        if self.kind == "ef_lookup":
            grouped: dict[str, list[int]] = defaultdict(list)
            for value, label in zip(selected["entry_friction"].to_numpy(dtype=object), target):
                if _is_missing(value):
                    continue
                parsed, missing, invalid = _number_array(pd.Series([value]))
                if missing[0] or invalid[0] or parsed[0] not in (0.0, 1.0):
                    continue
                grouped[str(int(parsed[0]))].append(int(label))
            for key, labels in grouped.items():
                self._lookup_counts[key] = len(labels)
                if len(labels) >= _LOOKUP_MIN_SUPPORT:
                    self._lookup_rates[key] = float(np.mean(labels))
            self._transform_state = {"fallback_counts": {}}

        elif self.kind == "lr_two":
            numeric = self._numeric_two(selected)
            self._linear_scaler = StandardScaler()
            matrix = self._linear_scaler.fit_transform(numeric)
            self._fit_logistic(matrix, target)
            self._transform_state = {"n_features": int(matrix.shape[1])}

        elif self.kind in {"lr_structured", "lr_hybrid", "hgb_structured"}:
            matrix = self._fit_structured(selected)
            if self.kind == "hgb_structured":
                self._estimator = HistGradientBoostingClassifier(
                    **_HGB_PARAMS,
                    categorical_features=self._hgb_categorical_mask,
                )
                self._fit_hgb(matrix, target)
            else:
                if self.kind == "lr_hybrid":
                    text_matrix = self._fit_text(selected["signal_stream"])
                    _, missing = _text_values(selected["signal_stream"])
                    missing_matrix = sparse.csr_matrix(missing.astype(float).reshape(-1, 1))
                    matrix = sparse.hstack((matrix, text_matrix, missing_matrix), format="csr", dtype=np.float64)
                    self._transform_state["structured_columns"] = int(matrix.shape[1] - text_matrix.shape[1])
                    self._transform_state["text_columns"] = int(text_matrix.shape[1])
                    self._transform_state["stream_missing_indicator"] = True
                self._fit_logistic(matrix, target)

        elif self.kind == "lr_text":
            text_matrix = self._fit_text(selected["signal_stream"])
            _, missing = _text_values(selected["signal_stream"])
            matrix = sparse.hstack((text_matrix, sparse.csr_matrix(missing.astype(float).reshape(-1, 1))),
                                   format="csr", dtype=np.float64)
            self._transform_state = {"text_columns": int(text_matrix.shape[1]),
                                     "stream_missing_indicator": True}
            self._fit_logistic(matrix, target)

        self._fitted = True
        self._diagnostics = {"n_predictions": 0}
        self._make_audit_state()
        return self

    @staticmethod
    def _numeric_two(X: pd.DataFrame) -> np.ndarray:
        values = []
        for name in TWO_FEATURES:
            numeric, missing, invalid = _number_array(X[name])
            if missing.any() or invalid.any() or not np.isfinite(numeric).all():
                raise ValueError(f"Two-feature logistic requires complete numeric {name!r}")
            values.append(numeric)
        return np.column_stack(values)

    def _fit_logistic(self, matrix: Any, target: np.ndarray) -> None:
        self._estimator = LogisticRegression(**_LR_PARAMS)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            self._estimator.fit(matrix, target)
        if any(issubclass(item.category, ConvergenceWarning) for item in caught):
            raise RuntimeError(f"{self.model_id} LogisticRegression did not converge")

    def _fit_hgb(self, matrix: np.ndarray, target: np.ndarray) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            self._estimator.fit(matrix, target)
        if any(issubclass(item.category, ConvergenceWarning) for item in caught):
            raise RuntimeError(f"{self.model_id} HistGradientBoostingClassifier did not converge")

    def _structured_raw(self, X: pd.DataFrame):
        return self._structured_preprocessor._raw_matrix(X.loc[:, list(EXPANDED_FEATURES)], fit_categories=False)

    def _fit_structured(self, X: pd.DataFrame) -> Any:
        self._structured_preprocessor = _HGBPreprocessor("expanded")
        fit_matrix, categorical_mask = self._structured_preprocessor.fit_transform(X.loc[:, list(EXPANDED_FEATURES)])
        self._hgb_categorical_mask = list(categorical_mask)
        self._native_vocabulary_hashes = {
            name: _json_digest([None if _is_missing(value) else str(value)
                                for value in self._structured_preprocessor.encoder.categories_[index]])
            for index, name in enumerate(CATEGORICAL_FEATURES)
        }
        raw, names, numeric_names, category_names, category_raw, _ = self._structured_raw(X)
        if names != self._structured_preprocessor.transformed_feature_names:
            raise RuntimeError("Expanded structured feature names changed during fit")
        numeric_positions = [index for index, name in enumerate(names)
                             if index not in self._structured_preprocessor.categorical_positions]
        if len(numeric_positions) != len(numeric_names):
            raise RuntimeError("Expanded structured numeric schema did not reconcile")
        # fit_transform performs imputation on its returned matrix. Reuse those
        # values so missing numeric fields never reach the scaler or estimator.
        numeric = fit_matrix[:, numeric_positions].copy()
        scaler = StandardScaler()
        numeric = scaler.fit_transform(numeric)
        self._linear_scaler = scaler
        self._linear_numeric_positions = numeric_positions
        self._linear_numeric_names = list(numeric_names)
        self._category_names = list(category_names)
        if self.kind in {"lr_structured", "lr_hybrid"}:
            token_columns = self._category_tokens(category_raw)
            encoder = OneHotEncoder(**_ENCODER_PARAMS)
            categorical = encoder.fit_transform(token_columns)
            self._linear_encoder = encoder
            self._ohe_fit_categories = [set(values.tolist()) for values in encoder.categories_]
            self._transform_state = {"numeric_feature_count": int(numeric.shape[1]),
                                     "categorical_feature_count": int(categorical.shape[1]),
                                     "category_names": list(category_names)}
            return sparse.hstack((sparse.csr_matrix(numeric), categorical), format="csr", dtype=np.float64)
        # HGB keeps ordinal category columns as native categorical features;
        # numeric positions and medians are learned by the same fit-only transformer.
        self._transform_state = {"transformed_feature_count": int(raw.shape[1]),
                                 "category_names": list(category_names)}
        return fit_matrix

    @staticmethod
    def _category_tokens(category_raw: dict[str, np.ndarray]) -> np.ndarray:
        if not category_raw:
            return np.empty((0, 0), dtype=object)
        result = np.empty((len(next(iter(category_raw.values()))), len(category_raw)), dtype=object)
        for column_index, name in enumerate(category_raw):
            result[:, column_index] = ["M:" if _is_missing(value) else "V:" + str(value)
                                       for value in category_raw[name]]
        return result

    def _fit_text(self, series: pd.Series) -> sparse.csr_matrix:
        documents, missing = _text_values(series)
        if not any(documents):
            self._fallback_all_empty = True
            self._text_vectorizer = None
            return sparse.csr_matrix((len(documents), 0), dtype=np.float64)
        vectorizer = TfidfVectorizer(**_TFIDF_PARAMS)
        try:
            matrix = vectorizer.fit_transform(documents).tocsr()
        except ValueError as exc:
            message = str(exc).casefold()
            empty_vocabulary_errors = (
                "empty vocabulary",
                "after pruning, no terms remain",
                "max_df corresponds to < documents than min_df",
                "max_df corresponds to less documents than min_df",
            )
            if not any(marker in message for marker in empty_vocabulary_errors):
                raise
            self._fallback_vocab_empty = True
            self._text_vectorizer = None
            matrix = sparse.csr_matrix((len(documents), 0), dtype=np.float64)
        else:
            self._text_vectorizer = vectorizer
        return matrix

    def _transform_structured(self, X: pd.DataFrame) -> Any:
        processed, unknown_counts = self._structured_preprocessor.transform(
            X.loc[:, list(EXPANDED_FEATURES)]
        )
        if self.kind == "hgb_structured":
            self._diagnostics["unknown_category_rows"] = unknown_counts
            return processed
        _, names, numeric_names, category_names, category_raw, _ = self._structured_raw(X)
        if list(category_names) != self._category_names or list(numeric_names) != self._linear_numeric_names:
            raise RuntimeError("Structured feature schema changed after fit")
        numeric = self._linear_scaler.transform(processed[:, self._linear_numeric_positions])
        token_columns = self._category_tokens(category_raw)
        categorical = self._linear_encoder.transform(token_columns)
        unknown_rows: dict[str, int] = {}
        for index, name in enumerate(category_names):
            seen = self._ohe_fit_categories[index]
            unknown_rows[name] = int(sum(token not in seen for token in token_columns[:, index]))
        self._diagnostics["unknown_category_rows"] = unknown_rows
        return sparse.hstack((sparse.csr_matrix(numeric), categorical), format="csr", dtype=np.float64)

    def _transform_text(self, series: pd.Series) -> sparse.csr_matrix:
        documents, missing = _text_values(series)
        if self._text_vectorizer is None:
            matrix = sparse.csr_matrix((len(documents), 0), dtype=np.float64)
            unknown_tokens = sum(len(text.split()) for text in documents)
        else:
            matrix = self._text_vectorizer.transform(documents).tocsr()
            analyzer = self._text_vectorizer.build_analyzer()
            vocabulary = self._text_vectorizer.vocabulary_
            unknown_tokens = sum(token not in vocabulary for doc in documents for token in analyzer(doc))
        missing_count = int(missing.sum())
        self._diagnostics.update({"signal_stream_missing_rows": missing_count,
                                  "unseen_tfidf_token_count": int(unknown_tokens)})
        return matrix

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Classifier must be fitted before predict_proba")
        selected = self._select(X)
        self._diagnostics = {"n_predictions": int(len(selected))}
        if self.kind == "majority":
            probabilities = np.full(len(selected), self._positive_rate, dtype=float)
        elif self.kind == "ef_lookup":
            probabilities = np.full(len(selected), self._positive_rate, dtype=float)
            fallback: Counter[str] = Counter()
            for position, value in enumerate(selected["entry_friction"].to_numpy(dtype=object)):
                if _is_missing(value):
                    fallback["missing_or_unknown"] += 1
                    continue
                numeric, missing, invalid = _number_array(pd.Series([value]))
                if missing[0] or invalid[0] or numeric[0] not in (0.0, 1.0):
                    fallback["missing_or_unknown"] += 1
                    continue
                key = str(int(numeric[0]))
                if key in self._lookup_rates:
                    probabilities[position] = self._lookup_rates[key]
                    fallback[f"entry_friction_{key}"] += 1
                else:
                    fallback["global"] += 1
            self._diagnostics["fallback_counts"] = dict(sorted(fallback.items()))
        elif self.kind == "lr_two":
            numeric = self._numeric_two(selected)
            probabilities = self._estimator.predict_proba(self._linear_scaler.transform(numeric))[:, 1]
        elif self.kind in {"lr_structured", "lr_hybrid", "hgb_structured"}:
            matrix = self._transform_structured(selected)
            if self.kind == "lr_hybrid":
                text = self._transform_text(selected["signal_stream"])
                _, missing = _text_values(selected["signal_stream"])
                matrix = sparse.hstack((matrix, text,
                    sparse.csr_matrix(missing.astype(float).reshape(-1, 1))), format="csr", dtype=np.float64)
            if self.kind == "hgb_structured":
                probabilities = self._estimator.predict_proba(matrix)[:, 1]
            else:
                probabilities = self._estimator.predict_proba(matrix)[:, 1]
        elif self.kind == "lr_text":
            text = self._transform_text(selected["signal_stream"])
            _, missing = _text_values(selected["signal_stream"])
            matrix = sparse.hstack((text, sparse.csr_matrix(missing.astype(float).reshape(-1, 1))),
                                   format="csr", dtype=np.float64)
            probabilities = self._estimator.predict_proba(matrix)[:, 1]
        else:
            raise RuntimeError(f"Unhandled classifier kind: {self.kind}")
        probabilities = np.asarray(probabilities, dtype=float)
        if probabilities.shape != (len(selected),) or not np.isfinite(probabilities).all():
            raise RuntimeError("Classifier returned invalid positive-class probabilities")
        if ((probabilities < 0) | (probabilities > 1)).any():
            raise RuntimeError("Classifier probabilities fall outside [0, 1]")
        return probabilities

    def _make_audit_state(self) -> None:
        state: dict[str, Any] = {
            "id": self.model_id,
            "kind": self.kind,
            "input_feature_names": list(self.input_feature_names),
            "fit_rows": self._fit_rows,
            "fit_data_sha256": self._fit_fingerprint,
            "positive_prevalence": self._positive_rate,
            "sklearn_version": __import__("sklearn").__version__,
            "parameters": {},
            "prediction_diagnostics": {},
        }
        if self.kind in {"majority", "ef_lookup"}:
            state["parameters"] = {"minimum_exact_ef_support": _LOOKUP_MIN_SUPPORT,
                                    "fallback": "fit-fold global positive rate"}
            state["supported_ef_counts"] = dict(sorted(self._lookup_counts.items()))
            state["supported_ef_positive_rates"] = dict(sorted(self._lookup_rates.items()))
            state["supported_ef_rate_sha256"] = _json_digest(self._lookup_rates)
        elif self.kind == "lr_two":
            state["parameters"] = dict(_LR_PARAMS)
            state["scaler"] = {"mean": self._linear_scaler.mean_.tolist(),
                               "scale": self._linear_scaler.scale_.tolist()}
            state["coefficient_sha256"] = _array_digest(self._estimator.coef_)
            state["intercept"] = self._estimator.intercept_.tolist()
        elif self.kind in {"lr_structured", "lr_hybrid"}:
            state["parameters"] = {
                "logistic_regression": dict(_LR_PARAMS),
                "one_hot_encoder": dict(_ENCODER_PARAMS),
                "minimum_category_support": _LOOKUP_MIN_SUPPORT,
            }
            state["transformed_features"] = {
                "numeric_names": list(self._linear_numeric_names),
                "input_category_names": list(self._category_names),
                "numeric_count": int(len(self._linear_numeric_names)),
                "encoded_category_counts": [int(len(values)) for values in self._linear_encoder.categories_],
                "encoded_category_sha256": _json_digest(
                    [[str(value) for value in values] for values in self._linear_encoder.categories_]
                ),
                "infrequent_category_sha256": _json_digest([
                    None if values is None else [str(value) for value in values]
                    for values in self._linear_encoder.infrequent_categories_
                ]),
            }
            state["scaler"] = {"mean_sha256": _array_digest(self._linear_scaler.mean_),
                               "scale_sha256": _array_digest(self._linear_scaler.scale_)}
            state["preprocessor"] = self._structured_preprocessor.audit_state()
            state["coefficient_sha256"] = _array_digest(self._estimator.coef_)
            state["intercept"] = self._estimator.intercept_.tolist()
            if self.kind == "lr_hybrid":
                state["text"] = self._text_state()
        elif self.kind == "hgb_structured":
            state["parameters"] = {key: _to_builtin(value) for key, value in self._estimator.get_params(deep=False).items()}
            state["preprocessor"] = self._structured_preprocessor.audit_state()
            state["prediction_shape"] = self._transform_state
            state["categorical_mask"] = list(self._structured_preprocessor.categorical_positions)
            state["native_vocabulary_sha256"] = dict(self._native_vocabulary_hashes)
        elif self.kind == "lr_text":
            state["parameters"] = {"logistic_regression": dict(_LR_PARAMS), "text": dict(_TFIDF_PARAMS),
                                    "stream_missing_indicator": True}
            state["coefficient_sha256"] = _array_digest(self._estimator.coef_)
            state["intercept"] = self._estimator.intercept_.tolist()
            state["text"] = self._text_state()
        state["model_state_sha256"] = _json_digest({key: value for key, value in state.items()
                                                       if key != "model_state_sha256"})
        self._state = state

    def _text_state(self) -> dict[str, Any]:
        if self._text_vectorizer is None:
            return {"parameters": {key: (list(value) if isinstance(value, tuple) else value)
                                    for key, value in _TFIDF_PARAMS.items()},
                    "vocabulary_size": 0, "vocabulary_sha256": None, "idf_sha256": None,
                    "empty_all_training_text": self._fallback_all_empty,
                    "empty_after_tfidf_filters": self._fallback_vocab_empty}
        vectorizer = self._text_vectorizer
        ordered = [(str(token), int(index)) for token, index in
                   sorted(vectorizer.vocabulary_.items(), key=lambda pair: pair[1])]
        return {"parameters": {key: (list(value) if isinstance(value, tuple) else value)
                                for key, value in _TFIDF_PARAMS.items()},
                "vocabulary_size": len(ordered),
                "vocabulary_sha256": _json_digest(ordered),
                "idf_sha256": _array_digest(vectorizer.idf_),
                "empty_all_training_text": False,
                "empty_after_tfidf_filters": False}

    def audit_state(self) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("Classifier must be fitted before audit_state")
        return json.loads(json.dumps(_to_builtin(self._state), sort_keys=True, allow_nan=False))

    def prediction_diagnostics(self) -> dict[str, Any]:
        return json.loads(json.dumps(_to_builtin(self._diagnostics), sort_keys=True, allow_nan=False))


def build_classifier(recipe: dict[str, Any]) -> ClassificationModel:
    """Build one fixed Round 1 classifier without reading data or fitting it."""
    return ClassificationModel(recipe)


__all__ = ["ClassificationModel", "build_classifier"]
