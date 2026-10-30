"""Small deterministic LightGBM MAE adapters for focused regression research.

The Round 2 selected-feature preprocessor and its eligibility allowlist are
reused unchanged. Encoders, numeric imputers, and category support are learned
from fit rows only. No validation data or early-stopping set is accepted here.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

import lightgbm
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.models.round1_models import (
    EXPANDED_FEATURES,
    _fit_fingerprint,
    _target_array,
    _to_builtin,
    _validate_dataframe,
)
from src.models.regression_round2_models import (
    _SelectedPreprocessor,
    build_model as build_round2_feature_model,
)


PARAMETER_SETS: dict[str, dict[str, Any]] = {
    "base": {"num_leaves": 31, "min_child_samples": 40, "reg_lambda": 1.0},
    "regularized": {"num_leaves": 15, "min_child_samples": 80, "reg_lambda": 3.0},
    "deeper": {"num_leaves": 63, "min_child_samples": 30, "reg_lambda": 2.0},
}
COMMON_PARAMETERS: dict[str, Any] = {
    "boosting_type": "gbdt",
    "objective": "regression_l1",
    "learning_rate": 0.05,
    "n_estimators": 400,
    "max_depth": -1,
    "min_child_weight": 0.001,
    "reg_alpha": 0.0,
    "colsample_bytree": 1.0,
    "subsample": 1.0,
    "subsample_freq": 0,
    "random_state": 42,
    "n_jobs": 2,
    "verbosity": -1,
    "deterministic": True,
    "force_col_wise": True,
}


def _tree_summary(booster: Any) -> dict[str, Any]:
    """Return compact structural counts and a digest, never the fitted trees."""
    model_dump = booster.dump_model()
    total_nodes = 0
    total_leaves = 0
    max_depth = 0

    def visit(node: dict[str, Any], depth: int) -> None:
        nonlocal total_nodes, total_leaves, max_depth
        total_nodes += 1
        max_depth = max(max_depth, depth)
        if "leaf_value" in node:
            total_leaves += 1
            return
        visit(node["left_child"], depth + 1)
        visit(node["right_child"], depth + 1)

    for tree in model_dump["tree_info"]:
        visit(tree["tree_structure"], 0)
    encoded = json.dumps(model_dump, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return {
        "trees_fitted": int(len(model_dump["tree_info"])),
        "total_nodes": int(total_nodes),
        "total_leaves": int(total_leaves),
        "maximum_depth": int(max_depth),
        "fitted_tree_sha256": hashlib.sha256(encoded).hexdigest(),
    }


class RegressionResearchTabular:
    """Train-only preprocessed LightGBM MAE model for TP2 research candidates."""

    def __init__(self, params_id: str, features: Iterable[str] | None):
        if params_id not in PARAMETER_SETS:
            raise ValueError(f"params_id must be one of {sorted(PARAMETER_SETS)}")
        if isinstance(features, str):
            raise TypeError("features must be a source-column iterable, not one string")
        # Delegate feature eligibility/order/required-core checks to the frozen
        # Round 2 builder. This object is used only to resolve selected names.
        requested_features = EXPANDED_FEATURES if features is None else features
        feature_spec = build_round2_feature_model(features=requested_features)
        self.selected_features = tuple(feature_spec.feature_names)
        self.params_id = params_id
        self.model_id = f"LIGHTGBM_{params_id.upper()}"
        self.actual_parameters = {**COMMON_PARAMETERS, **PARAMETER_SETS[params_id]}
        self._fitted = False

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.selected_features

    def _selected(self, X: pd.DataFrame) -> pd.DataFrame:
        _validate_dataframe(X)
        missing = [name for name in self.selected_features if name not in X.columns]
        if missing:
            raise ValueError(f"X is missing required source features: {missing}")
        return X.loc[:, list(self.selected_features)]

    def fit(self, X: pd.DataFrame, y: Any) -> "RegressionResearchTabular":
        selected = self._selected(X)
        target = _target_array(y, len(selected))
        self._fit_rows = int(len(target))
        self._fit_fingerprint = _fit_fingerprint(selected, self.selected_features, target)

        self._preprocessor = _SelectedPreprocessor(self.selected_features)
        matrix, categorical_mask = self._preprocessor.fit_transform(selected)
        self._categorical_positions = [i for i, is_category in enumerate(categorical_mask) if is_category]
        transformed = pd.DataFrame(matrix, columns=self._preprocessor.transformed_feature_names)
        self._estimator = LGBMRegressor(**self.actual_parameters)
        self._estimator.fit(transformed, target, categorical_feature=self._categorical_positions)
        self._tree_audit = _tree_summary(self._estimator.booster_)
        self._fit_state = {
            "id": self.model_id,
            "model_family": "LightGBM LGBMRegressor",
            "objective": "regression_l1",
            "params_id": self.params_id,
            "input_feature_names": list(self.selected_features),
            "transformed_feature_names": list(self._preprocessor.transformed_feature_names),
            "categorical_feature_indices": list(self._categorical_positions),
            "fit_rows": self._fit_rows,
            "fit_data_sha256": self._fit_fingerprint,
            "lightgbm_version": lightgbm.__version__,
            "actual_hyperparameters": dict(self.actual_parameters),
            "preprocessing": self._preprocessor.audit_state(),
            "tree_complexity": dict(self._tree_audit),
            "validation_data_passed_to_fit": False,
            "early_stopping_used": False,
        }
        self._diagnostics: dict[str, Any] = {"n_predictions": 0, "unknown_category_rows": {}}
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model must be fitted before predict")
        selected = self._selected(X)
        matrix, unknown = self._preprocessor.transform(selected)
        transformed = pd.DataFrame(matrix, columns=self._preprocessor.transformed_feature_names)
        predictions = np.asarray(self._estimator.predict(transformed), dtype=float)
        if predictions.shape != (len(selected),) or not np.isfinite(predictions).all():
            raise RuntimeError("LightGBM returned invalid regression predictions")
        self._diagnostics = {
            "n_predictions": int(len(predictions)),
            "unknown_category_rows": dict(sorted(unknown.items())),
        }
        return predictions

    def prediction_diagnostics(self) -> dict[str, Any]:
        return json.loads(json.dumps(_to_builtin(self._diagnostics), sort_keys=True))

    def audit_state(self) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("Model must be fitted before audit_state")
        # Deliberately excludes mutable prediction diagnostics; repeated calls
        # and calls before/after predict return the same fit-only receipt.
        return json.loads(json.dumps(_to_builtin(self._fit_state), sort_keys=True))


def build_tabular(
    params_id: str = "base",
    features: Iterable[str] | None = None,
) -> RegressionResearchTabular:
    """Build one preregistered LightGBM MAE candidate without fitting it."""
    return RegressionResearchTabular(params_id=params_id, features=features)


__all__ = ["COMMON_PARAMETERS", "PARAMETER_SETS", "RegressionResearchTabular", "build_tabular"]
