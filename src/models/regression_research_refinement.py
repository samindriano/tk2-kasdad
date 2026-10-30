"""Fixed, fit-only Stage 4 regression refinements.

These adapters expose the same minimal ``feature_names``/``fit``/``predict``/
``audit_state`` interface used by the Stage 2/3 runner. They never load data,
choose folds, read Test targets, or persist fitted model objects.
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

from src.models.regression_round2_models import (
    _SelectedPreprocessor, _tree_complexity, _tree_digest,
)
from src.models.regression_research_text import (
    HYBRID_FEATURES, STRUCTURED_FEATURES, RegressionTextModel,
    _HGB_PARAMS, _SVD_PARAMS, _array_digest, _json_digest, _target,
    _text_documents, build_text_model,
)
from src.models.round1_models import (
    EXPANDED_FEATURES, _fit_fingerprint,
)


SCOPE_SPREAD_FEATURES = (
    "scope_pressure", "scope_trace", "org_spread", "module_spread",
)
NO_SCOPE_STRUCTURED_FEATURES = tuple(
    name for name in EXPANDED_FEATURES if name not in set(SCOPE_SPREAD_FEATURES)
)
NO_SCOPE_FEATURES = (*NO_SCOPE_STRUCTURED_FEATURES, "signal_stream")
REFINEMENT_RECIPES = ("svd128", "svd_noscope32", "structured_ridge")


def _sealed_state(payload: dict[str, Any]) -> dict[str, Any]:
    state = json.loads(json.dumps(payload, ensure_ascii=True, allow_nan=False, sort_keys=True))
    state["state_sha256"] = _json_digest(state)
    return state


def _select_exact(X: pd.DataFrame, names: tuple[str, ...]) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")
    if X.columns.has_duplicates:
        raise ValueError("X contains duplicate columns")
    if set(X.columns) != set(names) or len(X.columns) != len(names):
        extra = sorted(set(X.columns) - set(names))
        absent = sorted(set(names) - set(X.columns))
        raise ValueError(f"X must contain exactly the declared features; extra={extra}, missing={absent}")
    return X.loc[:, list(names)].copy()


def _empty_stream_frame(X: pd.DataFrame) -> pd.DataFrame:
    augmented = X.copy()
    augmented["signal_stream"] = ""
    return augmented.loc[:, list(HYBRID_FEATURES)]


class RegressionResearchRefinement:
    """One of three fixed refined recipes, with preprocessing learned at fit."""

    def __init__(self, recipe: str):
        if recipe not in REFINEMENT_RECIPES:
            raise ValueError(f"recipe must be one of {list(REFINEMENT_RECIPES)}")
        self.recipe = recipe
        self.model_id = f"REGRESSION_REFINEMENT_{recipe.upper()}"
        self._fitted = False
        self._state: dict[str, Any] = {}
        self._diagnostics: dict[str, Any] = {"n_predictions": 0}
        self._delegate: RegressionTextModel | None = None
        self._estimator: HistGradientBoostingRegressor | None = None
        self._preprocessor: _SelectedPreprocessor | None = None
        self._vectorizer = None
        self._svd: TruncatedSVD | None = None
        self._fit_feature_names: tuple[str, ...] = ()

        if recipe == "svd128":
            # First use the public 128-component constructor if the frozen
            # text adapter has added support. In the currently frozen API it
            # accepts 32 only; then widen only this newly created instance.
            try:
                delegate = build_text_model("svd_hgb", svd_components=128)
            except ValueError as exc:
                if "component cap is frozen at 32" not in str(exc):
                    raise
                delegate = build_text_model("svd_hgb", svd_components=32)
                delegate.svd_components = 128
            self._delegate = delegate
            self._fit_feature_names = tuple(delegate.feature_names)
        elif recipe == "svd_noscope32":
            self._fit_feature_names = tuple(NO_SCOPE_FEATURES)
        else:
            self._fit_feature_names = tuple(STRUCTURED_FEATURES)

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Source columns accepted by this recipe, in deterministic order."""
        return self._fit_feature_names

    def fit(self, X: pd.DataFrame, y: Any) -> "RegressionResearchRefinement":
        self._fitted = False
        selected = _select_exact(X, self.feature_names)
        target = _target(y, len(selected))
        if self.recipe == "svd128":
            assert self._delegate is not None
            self._delegate.fit(selected, target)
            delegate_state = self._delegate.audit_state()
            self._state = _sealed_state({
                "id": self.model_id,
                "recipe": self.recipe,
                "input_feature_names": list(self.feature_names),
                "fit_rows": int(len(target)),
                "svd_components_requested": 128,
                "svd_components_actual": int(delegate_state["parameters"]["svd"]["n_components_actual"]),
                "delegated_model": delegate_state,
                "test_data_used_for_fit": False,
                "validation_data_passed_to_fit": False,
            })
        elif self.recipe == "structured_ridge":
            assert tuple(self.feature_names) == tuple(STRUCTURED_FEATURES)
            # The original hybrid Ridge owns the frozen imputer, scaler, OHE,
            # and Ridge implementation. A constant empty stream contributes
            # only its constant missingness indicator; no real stream is read.
            augmented = _empty_stream_frame(selected)
            delegate = build_text_model("hybrid_ridge").fit(augmented, target)
            delegate_state = delegate.audit_state()
            self._delegate = delegate
            self._state = _sealed_state({
                "id": self.model_id,
                "recipe": self.recipe,
                "input_feature_names": list(self.feature_names),
                "fit_rows": int(len(target)),
                "fit_data_sha256": delegate_state["fit_data_sha256"],
                "target_sha256": delegate_state["target_sha256"],
                "structured_ridge_alpha": 10.0,
                "constructed_signal_stream": {
                    "value": "",
                    "fit_rows": int(len(target)),
                    "actual_signal_stream_consumed": False,
                    "constant_missing_indicator": 1,
                },
                "delegated_model": delegate_state,
                "test_data_used_for_fit": False,
                "validation_data_passed_to_fit": False,
            })
        else:
            self._fit_svd_noscope(selected, target)
        self._fitted = True
        self._diagnostics = {"n_predictions": 0}
        return self

    def _fit_svd_noscope(self, selected: pd.DataFrame, target: np.ndarray) -> None:
        self._preprocessor = _SelectedPreprocessor(tuple(NO_SCOPE_STRUCTURED_FEATURES))
        structured, categorical_mask = self._preprocessor.fit_transform(
            selected.loc[:, list(NO_SCOPE_STRUCTURED_FEATURES)]
        )
        docs, missing = _text_documents(selected["signal_stream"])
        self._vectorizer, text_matrix, fallback = RegressionTextModel._fit_vectorizer(docs)
        self._svd_components_actual = min(32, text_matrix.shape[1] - 1) if text_matrix.shape[1] > 1 else 0
        if self._svd_components_actual:
            self._svd = TruncatedSVD(n_components=self._svd_components_actual, **dict(_SVD_PARAMS))
            text_dense = self._svd.fit_transform(text_matrix)
        else:
            text_dense = np.empty((len(selected), 0), dtype=np.float64)
        design = np.column_stack((structured, text_dense, missing.astype(np.float64)))
        design_categorical_mask = list(categorical_mask) + [False] * (self._svd_components_actual + 1)
        params = {**_HGB_PARAMS, "categorical_features": design_categorical_mask}
        self._estimator = HistGradientBoostingRegressor(**params)
        self._estimator.fit(design, target)
        self._model_complexity = _tree_complexity(self._estimator)
        self._tree_sha256 = _tree_digest(self._estimator)
        self._fit_fingerprint = _fit_fingerprint(selected, self.feature_names, target)
        self._target_sha256 = _array_digest(target)
        self._fit_rows = int(len(target))
        self._vectorizer_state = RegressionTextModel._text_state(self._vectorizer, fallback)
        self._training_shape = [int(design.shape[0]), int(design.shape[1])]
        self._design_categorical_mask = design_categorical_mask
        self._fit_missing_stream_rows = int(missing.sum())
        self._state = _sealed_state({
            "id": self.model_id,
            "recipe": self.recipe,
            "input_feature_names": list(self.feature_names),
            "removed_scope_spread_sources_and_derived_flags": list(SCOPE_SPREAD_FEATURES),
            "fit_rows": self._fit_rows,
            "fit_data_sha256": self._fit_fingerprint,
            "target_sha256": self._target_sha256,
            "training_design_shape": self._training_shape,
            "structured_preprocessing": self._preprocessor.audit_state(),
            "tfidf": self._vectorizer_state,
            "svd": {
                **dict(_SVD_PARAMS),
                "n_components_requested": 32,
                "n_components_actual": int(self._svd_components_actual),
                "components_sha256": _array_digest(self._svd.components_) if self._svd is not None else None,
            },
            "native_categorical_mask": design_categorical_mask,
            "fit_missing_stream_rows": self._fit_missing_stream_rows,
            "hgb_parameters": {**_HGB_PARAMS, "categorical_features": design_categorical_mask},
            "model_complexity": self._model_complexity,
            "fitted_tree_sha256": self._tree_sha256,
            "test_data_used_for_fit": False,
            "validation_data_passed_to_fit": False,
        })

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model must be fitted before predict")
        selected = _select_exact(X, self.feature_names)
        if self.recipe in {"svd128", "structured_ridge"}:
            assert self._delegate is not None
            if self.recipe == "structured_ridge":
                selected = _empty_stream_frame(selected)
            predictions = np.asarray(self._delegate.predict(selected), dtype=np.float64)
            if predictions.shape != (len(selected),) or not np.isfinite(predictions).all():
                raise RuntimeError("Delegated refinement model returned invalid predictions")
            if self.recipe == "structured_ridge":
                self._diagnostics = {
                    "n_predictions": int(len(predictions)),
                    "actual_signal_stream_consumed": False,
                    "constructed_empty_stream_rows": int(len(predictions)),
                    "unknown_category_rows": self._delegate.prediction_diagnostics().get("unknown_category_rows", {}),
                }
            else:
                self._diagnostics = self._delegate.prediction_diagnostics()
            return predictions
        return self._predict_svd_noscope(selected)

    def _predict_svd_noscope(self, selected: pd.DataFrame) -> np.ndarray:
        assert self._preprocessor is not None and self._estimator is not None
        assert hasattr(self, "_vectorizer") and hasattr(self, "_svd")
        docs, missing = _text_documents(selected["signal_stream"])
        if self._vectorizer is None:
            text_matrix = sparse.csr_matrix((len(docs), 0), dtype=np.float64)
            unseen_tokens = sum(len(text.split()) for text in docs)
        else:
            text_matrix = self._vectorizer.transform(docs).tocsr()
            analyzer = self._vectorizer.build_analyzer()
            vocabulary = self._vectorizer.vocabulary_
            unseen_tokens = sum(token not in vocabulary for text in docs for token in analyzer(text))
        structured, unknown = self._preprocessor.transform(
            selected.loc[:, list(NO_SCOPE_STRUCTURED_FEATURES)]
        )
        text_dense = self._svd.transform(text_matrix) if self._svd is not None else np.empty((len(docs), 0))
        design = np.column_stack((structured, text_dense, missing.astype(np.float64)))
        predictions = np.asarray(self._estimator.predict(design), dtype=np.float64)
        if predictions.shape != (len(selected),) or not np.isfinite(predictions).all():
            raise RuntimeError("No-scope SVD model returned invalid predictions")
        self._diagnostics = {
            "n_predictions": int(len(predictions)),
            "signal_stream_missing_rows": int(missing.sum()),
            "unseen_tfidf_token_count": int(unseen_tokens),
            "unknown_category_rows": dict(sorted(unknown.items())),
        }
        return predictions

    def audit_state(self) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("Model must be fitted before audit_state")
        return json.loads(json.dumps(self._state, sort_keys=True, allow_nan=False))

    def prediction_diagnostics(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._diagnostics, sort_keys=True, allow_nan=False))


def build_refinement(recipe: str) -> RegressionResearchRefinement:
    """Build one frozen Stage 4 diagnostic candidate without fitting it."""
    return RegressionResearchRefinement(recipe)


__all__ = [
    "NO_SCOPE_FEATURES", "NO_SCOPE_STRUCTURED_FEATURES", "REFINEMENT_RECIPES",
    "RegressionResearchRefinement", "build_refinement",
]
