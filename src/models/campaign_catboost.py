"""One fixed CatBoost MAE challenger with Train-fold-only structured/text prep."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from src.models.regression_research_text import (
    HYBRID_FEATURES, STRUCTURED_FEATURES, RegressionTextModel,
    _SVD_PARAMS, _array_digest, _json_digest, _target, _text_documents,
)
from src.models.regression_round2_models import _SelectedPreprocessor
from src.models.round1_models import _fit_fingerprint


CATBOOST_PARAMETERS = {
    "loss_function": "MAE",
    "iterations": 500,
    "depth": 6,
    "learning_rate": 0.05,
    "random_seed": 42,
    "thread_count": 2,
    "task_type": "CPU",
    "verbose": False,
    "allow_writing_files": False,
}


def _select_exact(X: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame) or X.columns.has_duplicates:
        raise TypeError("X must be a duplicate-free pandas DataFrame")
    if set(X.columns) != set(HYBRID_FEATURES) or len(X.columns) != len(HYBRID_FEATURES):
        raise ValueError("X must contain exactly the R1-D structured plus signal_stream feature allowlist")
    return X.loc[:, list(HYBRID_FEATURES)].copy()


def _frame(matrix: np.ndarray, names: list[str], categorical_positions: list[int]) -> pd.DataFrame:
    result = pd.DataFrame(matrix, columns=names)
    for position in categorical_positions:
        name = names[position]
        values = result[name].to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.allclose(values, np.rint(values), rtol=0, atol=0):
            raise RuntimeError(f"native categorical values are not finite integer tokens: {name}")
        result[name] = np.rint(values).astype(np.int64).astype(str)
    return result


class CampaignCatBoostRegressor:
    """CatBoost MAE over R1-D structured features and the fixed SVD32 text view.

    Every learned transform (parsers/imputers/category vocabulary, TF-IDF and
    TruncatedSVD) is fitted only on rows passed to :meth:`fit`.
    """

    def __init__(self):
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:
            raise RuntimeError("CatBoost is required for this research candidate") from exc
        self._estimator = CatBoostRegressor(**CATBOOST_PARAMETERS)
        self.input_feature_names = tuple(HYBRID_FEATURES)
        self._fitted = False

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.input_feature_names

    def fit(self, X: pd.DataFrame, y: Any) -> "CampaignCatBoostRegressor":
        self._fitted = False
        selected = _select_exact(X)
        target = _target(y, len(selected))
        self._preprocessor = _SelectedPreprocessor(tuple(STRUCTURED_FEATURES))
        structured, categorical_mask = self._preprocessor.fit_transform(
            selected.loc[:, list(STRUCTURED_FEATURES)]
        )
        self._categorical_positions = [i for i, value in enumerate(categorical_mask) if value]
        self._categorical_names = [
            self._preprocessor.transformed_feature_names[i] for i in self._categorical_positions
        ]
        documents, missing = _text_documents(selected["signal_stream"])
        self._vectorizer, text_matrix, fallback = RegressionTextModel._fit_vectorizer(documents)
        self._svd_components_actual = min(32, text_matrix.shape[1] - 1) if text_matrix.shape[1] > 1 else 0
        self._svd = None
        if self._svd_components_actual:
            self._svd = TruncatedSVD(n_components=self._svd_components_actual, **dict(_SVD_PARAMS))
            text_dense = self._svd.fit_transform(text_matrix)
        else:
            text_dense = np.empty((len(selected), 0), dtype=np.float64)
        design = np.column_stack((structured, text_dense, missing.astype(np.float64)))
        if not np.isfinite(design).all():
            raise RuntimeError("CatBoost structured/SVD32 design contains nonfinite values")
        self._feature_names = (
            list(self._preprocessor.transformed_feature_names)
            + [f"signal_stream_svd_{i + 1}" for i in range(self._svd_components_actual)]
            + ["signal_stream_missing"]
        )
        self._categorical_positions = list(self._categorical_positions)
        train_frame = _frame(design, self._feature_names, self._categorical_positions)
        self._estimator.fit(train_frame, target, cat_features=self._categorical_names)
        self._fit_rows = len(target)
        self._fit_fingerprint = _fit_fingerprint(selected, self.input_feature_names, target)
        self._target_sha256 = _array_digest(target)
        self._training_design_sha256 = _array_digest(design)
        self._training_design_shape = [int(design.shape[0]), int(design.shape[1])]
        self._text_state = RegressionTextModel._text_state(self._vectorizer, fallback)
        self._state = self._make_state()
        self._fitted = True
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        selected = _select_exact(X)
        structured, _ = self._preprocessor.transform(
            selected.loc[:, list(STRUCTURED_FEATURES)]
        )
        documents, missing = _text_documents(selected["signal_stream"])
        if self._vectorizer is None:
            text_matrix = sparse.csr_matrix((len(documents), 0), dtype=np.float64)
        else:
            text_matrix = self._vectorizer.transform(documents).tocsr()
        text_dense = self._svd.transform(text_matrix) if self._svd is not None else np.empty((len(documents), 0))
        design = np.column_stack((structured, text_dense, missing.astype(np.float64)))
        if not np.isfinite(design).all():
            raise RuntimeError("CatBoost transformed design contains nonfinite values")
        return _frame(design, self._feature_names, self._categorical_positions)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("fit must be called before predict")
        pred = np.asarray(self._estimator.predict(self._transform(X), thread_count=2), dtype=np.float64)
        if pred.shape != (len(X),) or not np.isfinite(pred).all():
            raise RuntimeError("CatBoost returned invalid predictions")
        return pred

    def _make_state(self) -> dict[str, Any]:
        import catboost
        state = {
            "id": "CATBOOST_MAE_SVD32",
            "family": "CatBoostRegressor",
            "catboost_version": catboost.__version__,
            "parameters": dict(CATBOOST_PARAMETERS),
            "input_feature_names": list(self.input_feature_names),
            "transformed_feature_names": list(self._feature_names),
            "native_categorical_feature_names": list(self._categorical_names),
            "fit_rows": int(self._fit_rows),
            "fit_data_sha256": self._fit_fingerprint,
            "target_sha256": self._target_sha256,
            "training_design_shape": self._training_design_shape,
            "training_design_sha256": self._training_design_sha256,
            "structured_preprocessing": self._preprocessor.audit_state(),
            "tfidf": self._text_state,
            "svd": {
                **dict(_SVD_PARAMS), "n_components_requested": 32,
                "n_components_actual": int(self._svd_components_actual),
                "components_sha256": _array_digest(self._svd.components_) if self._svd is not None else None,
            },
            "tree_count": int(self._estimator.tree_count_),
            "validation_data_passed_to_fit": False,
            "test_data_used_for_fit": False,
        }
        state["state_sha256"] = _json_digest(state)
        return state

    def audit_state(self) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("fit must be called before audit_state")
        return dict(self._state)


def build_catboost_campaign() -> CampaignCatBoostRegressor:
    """Build the single frozen CatBoost regression challenger."""
    return CampaignCatBoostRegressor()


__all__ = ["CATBOOST_PARAMETERS", "CampaignCatBoostRegressor", "build_catboost_campaign"]
