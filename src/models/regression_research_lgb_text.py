"""Fixed LightGBM refinements over the frozen full SVD32 text representation.

Only two preregisterable LightGBM parameter sets are exposed. All imputers,
category encoders, TF-IDF vocabulary/IDF, and SVD components are fit on the
current training rows only. This module never loads data or chooses folds.
"""
from __future__ import annotations

import json
from typing import Any

import lightgbm
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from lightgbm import LGBMRegressor

from src.models.regression_research_tabular import (
    COMMON_PARAMETERS, PARAMETER_SETS, _tree_summary,
)
from src.models.regression_research_text import (
    HYBRID_FEATURES, STRUCTURED_FEATURES, RegressionTextModel,
    _SVD_PARAMS, _array_digest, _json_digest, _target,
    _text_documents,
)
from src.models.regression_round2_models import _SelectedPreprocessor
from src.models.round1_models import _fit_fingerprint


LGB_TEXT_PARAMS = ("base", "regularized")


def _select_exact(X: pd.DataFrame) -> pd.DataFrame:
    names = tuple(HYBRID_FEATURES)
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")
    if X.columns.has_duplicates:
        raise ValueError("X contains duplicate columns")
    if set(X.columns) != set(names) or len(X.columns) != len(names):
        extra = sorted(set(X.columns) - set(names))
        missing = sorted(set(names) - set(X.columns))
        raise ValueError(f"X must contain exactly the declared features; extra={extra}, missing={missing}")
    return X.loc[:, list(names)].copy()


class RegressionResearchLGBText:
    """LGBMRegressor with regression-L1 loss over the frozen SVD32 design."""

    def __init__(self, params_id: str = "base"):
        if params_id not in LGB_TEXT_PARAMS:
            raise ValueError(f"params_id must be one of {list(LGB_TEXT_PARAMS)}")
        self.params_id = params_id
        self.model_id = f"LIGHTGBM_SVD32_{params_id.upper()}"
        self.input_feature_names = tuple(HYBRID_FEATURES)
        self.actual_parameters = {**COMMON_PARAMETERS, **PARAMETER_SETS[params_id]}
        self._fitted = False
        self._state: dict[str, Any] = {}
        self._diagnostics: dict[str, Any] = {"n_predictions": 0}
        self._vectorizer = None
        self._svd: TruncatedSVD | None = None
        self._preprocessor: _SelectedPreprocessor | None = None
        self._estimator: LGBMRegressor | None = None

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.input_feature_names

    def fit(self, X: pd.DataFrame, y: Any) -> "RegressionResearchLGBText":
        self._fitted = False
        selected = _select_exact(X)
        target = _target(y, len(selected))
        self._fit_rows = int(len(target))
        self._fit_fingerprint = _fit_fingerprint(selected, self.input_feature_names, target)
        self._target_sha256 = _array_digest(target)

        self._preprocessor = _SelectedPreprocessor(tuple(STRUCTURED_FEATURES))
        structured, categorical_mask = self._preprocessor.fit_transform(
            selected.loc[:, list(STRUCTURED_FEATURES)]
        )
        categorical_mask = list(categorical_mask)
        self._categorical_positions = [i for i, is_cat in enumerate(categorical_mask) if is_cat]

        docs, missing = _text_documents(selected["signal_stream"])
        self._vectorizer, text_matrix, fallback = RegressionTextModel._fit_vectorizer(docs)
        self._svd_components_actual = min(32, text_matrix.shape[1] - 1) if text_matrix.shape[1] > 1 else 0
        if self._svd_components_actual:
            self._svd = TruncatedSVD(n_components=self._svd_components_actual, **dict(_SVD_PARAMS))
            text_dense = self._svd.fit_transform(text_matrix)
        else:
            text_dense = np.empty((len(selected), 0), dtype=np.float64)
        design = np.column_stack((structured, text_dense, missing.astype(np.float64)))
        if not np.isfinite(design).all():
            raise RuntimeError("SVD32 LightGBM design contains nonfinite values")
        self._transformed_feature_names = (
            list(self._preprocessor.transformed_feature_names)
            + [f"signal_stream_svd_{i + 1}" for i in range(self._svd_components_actual)]
            + ["signal_stream_missing"]
        )
        if len(self._transformed_feature_names) != design.shape[1]:
            raise RuntimeError("SVD32 feature-name schema does not match the design matrix")
        design_frame = pd.DataFrame(design, columns=self._transformed_feature_names)

        self._estimator = LGBMRegressor(**dict(self.actual_parameters))
        self._estimator.fit(design_frame, target, categorical_feature=self._categorical_positions)
        self._tree_audit = _tree_summary(self._estimator.booster_)
        self._training_design_sha256 = _array_digest(design)
        self._training_design_shape = [int(design.shape[0]), int(design.shape[1])]
        self._fit_missing_stream_rows = int(missing.sum())
        self._vectorizer_state = RegressionTextModel._text_state(self._vectorizer, fallback)
        self._structured_mask = categorical_mask
        self._state = self._make_state()
        self._fitted = True
        self._diagnostics = {"n_predictions": 0}
        return self

    def _make_state(self) -> dict[str, Any]:
        assert self._preprocessor is not None
        assert self._estimator is not None
        payload = {
            "id": self.model_id,
            "model_family": "LightGBM LGBMRegressor",
            "objective": "regression_l1",
            "params_id": self.params_id,
            "input_feature_names": list(self.input_feature_names),
            "transformed_feature_names": list(self._transformed_feature_names),
            "fit_rows": self._fit_rows,
            "fit_data_sha256": self._fit_fingerprint,
            "target_sha256": self._target_sha256,
            "training_design_shape": self._training_design_shape,
            "training_design_sha256": self._training_design_sha256,
            "structured_preprocessing": self._preprocessor.audit_state(),
            "structured_categorical_mask": list(self._structured_mask),
            "categorical_feature_indices": list(self._categorical_positions),
            "categorical_unknown_value": -1,
            "categorical_negative_values_are_missing": True,
            "tfidf": self._vectorizer_state,
            "svd": {
                **dict(_SVD_PARAMS),
                "n_components_requested": 32,
                "n_components_actual": int(self._svd_components_actual),
                "components_sha256": (
                    _array_digest(self._svd.components_) if self._svd is not None else None
                ),
            },
            "fit_missing_stream_rows": self._fit_missing_stream_rows,
            "lightgbm_version": lightgbm.__version__,
            "actual_hyperparameters": dict(self.actual_parameters),
            "categorical_feature_argument": list(self._categorical_positions),
            "tree_complexity": dict(self._tree_audit),
            "validation_data_passed_to_fit": False,
            "early_stopping_used": False,
            "test_data_used_for_fit": False,
        }
        payload["state_sha256"] = _json_digest(payload)
        return json.loads(json.dumps(payload, sort_keys=True, allow_nan=False))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model must be fitted before predict")
        assert self._preprocessor is not None
        assert self._estimator is not None
        selected = _select_exact(X)
        structured, unknown = self._preprocessor.transform(
            selected.loc[:, list(STRUCTURED_FEATURES)]
        )
        docs, missing = _text_documents(selected["signal_stream"])
        if self._vectorizer is None:
            text_matrix = sparse.csr_matrix((len(docs), 0), dtype=np.float64)
            unseen_tokens = sum(len(text.split()) for text in docs)
        else:
            text_matrix = self._vectorizer.transform(docs).tocsr()
            analyzer = self._vectorizer.build_analyzer()
            vocabulary = self._vectorizer.vocabulary_
            unseen_tokens = sum(token not in vocabulary for text in docs for token in analyzer(text))
        text_dense = self._svd.transform(text_matrix) if self._svd is not None else np.empty((len(docs), 0))
        design = np.column_stack((structured, text_dense, missing.astype(np.float64)))
        design_frame = pd.DataFrame(design, columns=self._transformed_feature_names)
        prediction = np.asarray(self._estimator.predict(design_frame), dtype=np.float64)
        if prediction.shape != (len(selected),) or not np.isfinite(prediction).all():
            raise RuntimeError("LightGBM SVD32 returned invalid predictions")
        self._diagnostics = {
            "n_predictions": int(len(prediction)),
            "signal_stream_missing_rows": int(missing.sum()),
            "unseen_tfidf_token_count": int(unseen_tokens),
            "unknown_category_rows": dict(sorted(unknown.items())),
        }
        return prediction

    def audit_state(self) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("Model must be fitted before audit_state")
        return json.loads(json.dumps(self._state, sort_keys=True, allow_nan=False))

    def prediction_diagnostics(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._diagnostics, sort_keys=True, allow_nan=False))


def build_lgb_text(params_id: str = "base") -> RegressionResearchLGBText:
    """Build one of the two preregistered LightGBM SVD32 variants."""
    return RegressionResearchLGBText(params_id=params_id)


__all__ = ["LGB_TEXT_PARAMS", "RegressionResearchLGBText", "build_lgb_text"]
