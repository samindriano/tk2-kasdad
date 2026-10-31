"""Fixed sparse hybrid regression candidates for the bounded Optimization V2 pilot.

Both recipes reuse the Round 2 hybrid preprocessing contract: structured numeric
features are imputed/scaled, categorical fields are one-hot encoded, signal text
is fit-only TF-IDF, and a missing-stream indicator is appended. No source data is
loaded and no folds are selected in this module.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import lightgbm
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy import sparse
from sklearn.linear_model import SGDRegressor

from src.models.regression_research_tabular import COMMON_PARAMETERS, PARAMETER_SETS
from src.models.regression_research_text import (
    HYBRID_FEATURES,
    RegressionTextModel,
    _array_digest,
    _safe_builtin,
    _target,
    _text_documents,
)
from src.models.round1_models import _fit_fingerprint


SGD_PARAMETERS: dict[str, Any] = {
    "loss": "epsilon_insensitive",
    "epsilon": 0.0,
    "penalty": "l2",
    "alpha": 1e-5,
    "max_iter": 2000,
    "tol": 1e-5,
    "n_iter_no_change": 30,
    "early_stopping": False,
    "learning_rate": "invscaling",
    "eta0": 0.01,
    "power_t": 0.25,
    "average": True,
    "random_state": 42,
}

SPARSE_LGB_PARAMETERS: dict[str, Any] = {
    **COMMON_PARAMETERS,
    **PARAMETER_SETS["base"],
}


def _digest_json(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                     separators=(",", ":")).encode("utf-8", errors="surrogatepass")
    return hashlib.sha256(raw).hexdigest()


class SparseHybridRegressor:
    """A fixed SGD-L1 or sparse LightGBM-L1 hybrid on train-only transforms."""

    def __init__(self, recipe: str):
        if recipe not in {"sgd_l1", "lgbm_sparse_l1"}:
            raise ValueError("recipe must be 'sgd_l1' or 'lgbm_sparse_l1'")
        self.recipe = recipe
        self.model_id = recipe.upper()
        self.input_feature_names = tuple(HYBRID_FEATURES)
        self._fitted = False
        self._diagnostics: dict[str, Any] = {"n_predictions": 0}

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.input_feature_names

    def _select(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if X.columns.has_duplicates:
            raise ValueError("X contains duplicate columns")
        if set(X.columns) != set(self.input_feature_names) or len(X.columns) != len(self.input_feature_names):
            extra = sorted(set(X.columns) - set(self.input_feature_names))
            absent = sorted(set(self.input_feature_names) - set(X.columns))
            raise ValueError(f"X must contain exactly the declared hybrid features; extra={extra}, missing={absent}")
        return X.loc[:, list(self.input_feature_names)].copy()

    def fit(self, X: pd.DataFrame, y: Any) -> "SparseHybridRegressor":
        self._fitted = False
        selected = self._select(X)
        target = _target(y, len(selected))
        self._fit_rows = int(len(target))
        self._fit_fingerprint = _fit_fingerprint(selected, self.input_feature_names, target)
        self._target_sha256 = _array_digest(target)

        docs, missing = _text_documents(selected["signal_stream"])
        self._missing_fit_rows = int(missing.sum())
        self._vectorizer, text_matrix, self._text_fallback = RegressionTextModel._fit_vectorizer(docs)
        (structured_matrix, self._processor, self._scaler, self._encoder,
         self._numeric_positions, structured_state) = RegressionTextModel._structured_fit(selected)
        missing_column = sparse.csr_matrix(missing.astype(np.float64).reshape(-1, 1))
        design = sparse.hstack((structured_matrix, text_matrix, missing_column),
                               format="csr", dtype=np.float64)
        if design.shape[0] != len(target) or not np.isfinite(design.data).all():
            raise RuntimeError("Sparse hybrid fit design is misaligned or non-finite")
        self._text_width = int(text_matrix.shape[1])
        self._structured_width = int(structured_matrix.shape[1])

        if self.recipe == "sgd_l1":
            self._estimator = SGDRegressor(**SGD_PARAMETERS)
            scaled_target = (target - 50.0) / 50.0
            self._estimator.fit(design, scaled_target)
            estimator_state: dict[str, Any] = {
                "parameters": dict(SGD_PARAMETERS),
                "target_transform": "(threat_rank - 50) / 50",
                "target_inverse": "prediction * 50 + 50",
                "n_iter": int(self._estimator.n_iter_),
                "t": float(self._estimator.t_),
                "coefficient_sha256": _array_digest(self._estimator.coef_),
                "intercept": _safe_builtin(self._estimator.intercept_),
            }
        else:
            self._estimator = LGBMRegressor(**SPARSE_LGB_PARAMETERS)
            self._estimator.fit(design, target)
            model_text = self._estimator.booster_.model_to_string(num_iteration=-1)
            estimator_state = {
                "parameters": dict(SPARSE_LGB_PARAMETERS),
                "target_transform": "identity",
                "trees_fitted": int(self._estimator.booster_.current_iteration()),
                "feature_importance_sha256": _array_digest(self._estimator.feature_importances_, dtype="<i8"),
                "fitted_model_sha256": hashlib.sha256(model_text.encode("utf-8")).hexdigest(),
            }

        self._fit_state = {
            "id": self.model_id,
            "recipe": self.recipe,
            "model_family": "sklearn.linear_model.SGDRegressor" if self.recipe == "sgd_l1"
            else "lightgbm.LGBMRegressor",
            "objective": "epsilon_insensitive_epsilon_0" if self.recipe == "sgd_l1" else "regression_l1",
            "fit_rows": self._fit_rows,
            "fit_data_sha256": self._fit_fingerprint,
            "target_sha256": self._target_sha256,
            "fit_missing_stream_rows": self._missing_fit_rows,
            "fit_missing_stream_rate": float(self._missing_fit_rows / self._fit_rows),
            "input_feature_names": list(self.input_feature_names),
            "training_design_shape": [int(design.shape[0]), int(design.shape[1])],
            "structured_width": self._structured_width,
            "tfidf_width": self._text_width,
            "tfidf": RegressionTextModel._text_state(self._vectorizer, self._text_fallback),
            "structured_transform": structured_state,
            "missing_stream_indicator": True,
            "sparse_design": True,
            "svd_used": False,
            "native_categorical_features_used": False,
            "lightgbm_version": lightgbm.__version__ if self.recipe == "lgbm_sparse_l1" else None,
            "scikit_learn_version": __import__("sklearn").__version__,
            "estimator": estimator_state,
            "validation_data_passed_to_fit": False,
            "early_stopping_used": False,
        }
        self._fit_state["state_sha256"] = _digest_json(self._fit_state)
        self._fitted = True
        self._diagnostics = {"n_predictions": 0}
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model must be fitted before predict")
        selected = self._select(X)
        docs, missing = _text_documents(selected["signal_stream"])
        if self._vectorizer is None:
            text_matrix = sparse.csr_matrix((len(docs), 0), dtype=np.float64)
            unseen_token_count = sum(len(doc.split()) for doc in docs)
        else:
            text_matrix = self._vectorizer.transform(docs).tocsr()
            analyzer = self._vectorizer.build_analyzer()
            vocabulary = self._vectorizer.vocabulary_
            unseen_token_count = sum(token not in vocabulary for doc in docs for token in analyzer(doc))
        structured, unknown = RegressionTextModel._structured_transform(
            selected, self._processor, self._scaler, self._encoder, self._numeric_positions
        )
        missing_column = sparse.csr_matrix(missing.astype(np.float64).reshape(-1, 1))
        design = sparse.hstack((structured, text_matrix, missing_column), format="csr", dtype=np.float64)
        if design.shape[1] != self._fit_state["training_design_shape"][1] or not np.isfinite(design.data).all():
            raise RuntimeError("Sparse hybrid prediction design differs from fitted design")
        prediction = np.asarray(self._estimator.predict(design), dtype=np.float64)
        if self.recipe == "sgd_l1":
            prediction = prediction * 50.0 + 50.0
        if prediction.shape != (len(selected),) or not np.isfinite(prediction).all():
            raise RuntimeError("Sparse hybrid model returned invalid predictions")
        self._diagnostics = {
            "n_predictions": int(len(prediction)),
            "signal_stream_missing_rows": int(missing.sum()),
            "unseen_tfidf_token_count": int(unseen_token_count),
            "unknown_category_rows": dict(sorted(unknown.items())),
        }
        return prediction

    def audit_state(self) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("Model must be fitted before audit_state")
        return json.loads(json.dumps(_safe_builtin(self._fit_state), sort_keys=True, allow_nan=False))

    def prediction_diagnostics(self) -> dict[str, Any]:
        return json.loads(json.dumps(_safe_builtin(self._diagnostics), sort_keys=True, allow_nan=False))


def build_sparse_hybrid(recipe: str) -> SparseHybridRegressor:
    """Build a fixed candidate; no parameters are searched or inferred."""
    return SparseHybridRegressor(recipe)


__all__ = ["SGD_PARAMETERS", "SPARSE_LGB_PARAMETERS", "SparseHybridRegressor", "build_sparse_hybrid"]
