"""Controlled Regression Round 2 HGB feature-set adapters.

This module reuses the frozen Round 1 HGB objective and parser contracts. Round
2 variants may select only explicit Round 1 eligible structured source columns;
all transforms and category vocabularies are learned from fit rows only.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder
from collections import Counter

from src.models.round1_models import (
    CATEGORICAL_FEATURES,
    EXPANDED_FEATURES,
    MIXED_COUNT_FEATURES,
    NUMERIC_FEATURES,
    TELEMETRY_FEATURE,
    TWO_FEATURES,
    _FIXED_HGB_PARAMS,
    _category_array,
    _fit_fingerprint,
    _is_missing,
    _mixed_count_arrays,
    _number_array,
    _target_array,
    _telemetry_arrays,
    _to_builtin,
    _validate_dataframe,
    build_model as build_round1_model,
)


def _params() -> dict[str, Any]:
    return {**_FIXED_HGB_PARAMS, "loss": "absolute_error"}


def _tree_complexity(estimator: HistGradientBoostingRegressor) -> dict[str, Any]:
    predictors = getattr(estimator, "_predictors", [])
    nodes = [tree.nodes for iteration in predictors for tree in iteration]
    node_counts = [int(len(array)) for array in nodes]
    leaf_counts = [int(np.count_nonzero(array["is_leaf"])) for array in nodes]
    return {
        "iterations_fitted": int(len(predictors)),
        "trees_fitted": int(len(nodes)),
        "total_nodes": int(sum(node_counts)),
        "total_leaves": int(sum(leaf_counts)),
        "node_counts_per_tree": node_counts,
        "leaf_counts_per_tree": leaf_counts,
    }


def _tree_digest(estimator: HistGradientBoostingRegressor) -> str:
    digest = hashlib.sha256()
    for iteration in getattr(estimator, "_predictors", []):
        for tree in iteration:
            nodes = tree.nodes
            digest.update(repr((nodes.dtype.descr, nodes.shape)).encode("utf-8"))
            digest.update(nodes.tobytes())
    baseline = np.asarray(getattr(estimator, "_baseline_prediction", []), dtype=np.float64)
    digest.update(repr((baseline.dtype.str, baseline.shape)).encode("ascii"))
    digest.update(baseline.tobytes())
    return digest.hexdigest()


class _SelectedPreprocessor:
    """Round 1-compatible transform over an explicitly selected source subset."""

    def __init__(self, selected_features: tuple[str, ...]):
        self.selected_features = selected_features
        self.encoder: OrdinalEncoder | None = None
        self.medians: dict[str, float] = {}
        self.transformed_feature_names: list[str] = []
        self.numeric_feature_names: list[str] = []
        self.categorical_positions: list[int] = []
        self.category_counts: dict[str, dict[str, Any]] = {}
        self.seen_categories: dict[str, set[str]] = {}
        self.category_names = tuple(name for name in selected_features if name in CATEGORICAL_FEATURES)

    def _matrix(self, X: pd.DataFrame, *, fit_categories: bool):
        # X was sliced to the explicit selected-feature allowlist by the wrapper.
        numeric: dict[str, list[np.ndarray]] = {}
        numeric_names: dict[str, list[str]] = {}
        category_raw: dict[str, np.ndarray] = {}
        category_missing: dict[str, np.ndarray] = {}
        for feature in self.selected_features:
            if feature in NUMERIC_FEATURES:
                values, missing, invalid = _number_array(X[feature])
                numeric[feature] = [values, invalid, missing]
                numeric_names[feature] = [feature, f"{feature}__parsefail", f"{feature}__missing"]
            elif feature in MIXED_COUNT_FEATURES:
                values, approximate, suffix, invalid, missing = _mixed_count_arrays(X[feature])
                numeric[feature] = [values, approximate, suffix, invalid, missing]
                numeric_names[feature] = [feature, f"{feature}__approximate", f"{feature}__suffix_present",
                                          f"{feature}__parsefail", f"{feature}__missing"]
            elif feature == TELEMETRY_FEATURE:
                values, unknown, missing = _telemetry_arrays(X[feature])
                numeric[feature] = [values, unknown, missing]
                numeric_names[feature] = [feature, f"{feature}__unknown", f"{feature}__missing"]
            elif feature in CATEGORICAL_FEATURES:
                raw = _category_array(X[feature])
                category_raw[feature] = raw
                category_missing[feature] = np.array([_is_missing(value) for value in raw], dtype=float)

        if fit_categories and self.category_names:
            self.encoder = OrdinalEncoder(
                min_frequency=30, max_categories=254, handle_unknown="use_encoded_value",
                unknown_value=-1, encoded_missing_value=-1, dtype=np.float64,
            )
            category_matrix = np.column_stack([category_raw[name] for name in self.category_names])
            self.encoder.fit(category_matrix)
            for index, feature in enumerate(self.category_names):
                self.seen_categories[feature] = {
                    str(value) for value in self.encoder.categories_[index] if not _is_missing(value)
                }
        if self.category_names:
            if self.encoder is None:
                raise RuntimeError("Categorical encoder was not fitted")
            encoded = self.encoder.transform(np.column_stack([category_raw[name] for name in self.category_names]))
        else:
            encoded = np.empty((len(X), 0), dtype=float)

        matrix_columns: list[np.ndarray] = []
        names: list[str] = []
        numeric_output_names: list[str] = []
        categorical_positions: list[int] = []
        category_index = 0
        for feature in self.selected_features:
            if feature in numeric:
                matrix_columns.extend(numeric[feature])
                names.extend(numeric_names[feature])
                numeric_output_names.extend(numeric_names[feature])
            else:
                categorical_positions.append(len(names))
                names.append(f"{feature}__ordinal")
                matrix_columns.append(encoded[:, category_index])
                category_index += 1
                names.append(f"{feature}__missing")
                matrix_columns.append(category_missing[feature])
                numeric_output_names.append(f"{feature}__missing")
        matrix = np.column_stack(matrix_columns).astype(float, copy=False)

        if fit_categories:
            for index, feature in enumerate(self.category_names):
                raw = category_raw[feature]
                observed = [str(value) for value in raw if not _is_missing(value)]
                represented = np.unique(encoded[:, index][encoded[:, index] >= 0])
                infrequent = getattr(self.encoder, "infrequent_categories_", None)
                infrequent_count = 0 if infrequent is None or infrequent[index] is None else len(infrequent[index])
                self.category_counts[feature] = {
                    "observed_unique": len(Counter(observed)),
                    "encoded_categories": int(len(represented)),
                    "infrequent_categories": int(infrequent_count),
                    "training_missing_rows": int(sum(_is_missing(value) for value in raw)),
                    "training_rows": len(raw), "min_frequency": 30, "max_categories": 254,
                }
        return matrix, names, numeric_output_names, category_raw

    def fit_transform(self, X: pd.DataFrame) -> tuple[np.ndarray, list[bool]]:
        matrix, names, numeric_names, _ = self._matrix(X, fit_categories=True)
        self.transformed_feature_names = names
        self.numeric_feature_names = numeric_names
        self.categorical_positions = [
            i for i, name in enumerate(names) if name.endswith("__ordinal")
        ]
        categorical = set(self.categorical_positions)
        self.medians = {}
        for position, name in enumerate(names):
            if position in categorical:
                continue
            column = matrix[:, position]
            finite = column[np.isfinite(column)]
            median = float(np.median(finite)) if len(finite) else 0.0
            self.medians[name] = median
            column[~np.isfinite(column)] = median
        return matrix, [position in categorical for position in range(len(names))]

    def transform(self, X: pd.DataFrame) -> tuple[np.ndarray, dict[str, int]]:
        matrix, names, _, category_raw = self._matrix(X, fit_categories=False)
        if names != self.transformed_feature_names:
            raise RuntimeError("Transformed feature schema changed after fit")
        unknown = {
            feature: int(sum(not _is_missing(value) and str(value) not in self.seen_categories[feature]
                             for value in category_raw[feature]))
            for feature in self.category_names
        }
        categorical = set(self.categorical_positions)
        for position, name in enumerate(names):
            if position in categorical:
                continue
            column = matrix[:, position]
            column[~np.isfinite(column)] = self.medians[name]
        return matrix, unknown

    def audit_state(self) -> dict[str, Any]:
        return {
            "transformed_feature_names": list(self.transformed_feature_names),
            "numeric_imputer_medians": dict(self.medians),
            "categorical_feature_names": [self.transformed_feature_names[i] for i in self.categorical_positions],
            "category_counts": dict(self.category_counts),
            "ordinal_encoder": {"min_frequency": 30, "max_categories": 254,
                                "handle_unknown": "use_encoded_value", "unknown_value": -1,
                                "encoded_missing_value": -1, "dtype": "float64"},
        }


class RegressionRound2Model:
    """Round 1-compatible HGB regressor with controlled selected source groups."""

    def __init__(self, features: tuple[str, ...], control: str | None = None):
        self.control = control
        self.selected_features = features
        self.model_id = control or "R2_FEATURE_ABLATION"
        self._fitted = False
        self._prediction_diagnostics: dict[str, Any] = {}
        if control:
            recipe = {"id": control, "kind": "hgb", "features": "two" if control == "R1_C" else "expanded",
                      "params": {"loss": "absolute_error"}}
            self._delegate = build_round1_model(recipe)
        else:
            self._delegate = None

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.selected_features

    def _selected(self, X: pd.DataFrame) -> pd.DataFrame:
        _validate_dataframe(X)
        missing = [name for name in self.selected_features if name not in X.columns]
        if missing:
            raise ValueError(f"X is missing required feature columns: {missing}")
        # The estimator never reads any source outside this frozen allowlist.
        return X.loc[:, list(self.selected_features)]

    def fit(self, X: pd.DataFrame, y: Any) -> "RegressionRound2Model":
        selected = self._selected(X)
        target = _target_array(y, len(selected))
        if self._delegate is not None:
            self._delegate.fit(selected, target)
            self._fitted = True
            return self
        self._fit_rows = len(target)
        self._fit_fingerprint = _fit_fingerprint(selected, self.selected_features, target)
        self._preprocessor = _SelectedPreprocessor(self.selected_features)
        matrix, categorical_mask = self._preprocessor.fit_transform(selected)
        params = _params()
        params["categorical_features"] = categorical_mask if any(categorical_mask) else None
        self._estimator = HistGradientBoostingRegressor(**params)
        self._estimator.fit(matrix, target)
        self._hyperparameters = {key: _to_builtin(value) for key, value in self._estimator.get_params(deep=False).items()}
        self._tree_state_sha256 = _tree_digest(self._estimator)
        self._prediction_diagnostics = {"n_predictions": 0}
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model must be fitted before predict")
        selected = self._selected(X)
        if self._delegate is not None:
            return self._delegate.predict(selected)
        matrix, unknown = self._preprocessor.transform(selected)
        predictions = np.asarray(self._estimator.predict(matrix), dtype=float)
        self._prediction_diagnostics = {"n_predictions": int(len(selected)),
                                        "unknown_category_rows": dict(sorted(unknown.items()))}
        return predictions

    def prediction_diagnostics(self) -> dict[str, Any]:
        if self._delegate is not None:
            return self._delegate.prediction_diagnostics()
        return json.loads(json.dumps(_to_builtin(self._prediction_diagnostics), sort_keys=True))

    def audit_state(self) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("Model must be fitted before audit_state")
        if self._delegate is not None:
            # Preserve exact Round 1 audit-state identity for the shared controls.
            return self._delegate.audit_state()
        state = {
            "id": self.model_id, "kind": "hgb", "feature_preset": "selected_expanded",
            "input_feature_names": list(self.selected_features),
            "transformed_feature_names": list(self._preprocessor.transformed_feature_names),
            "fit_rows": int(self._fit_rows), "fit_data_sha256": self._fit_fingerprint,
            "sklearn_version": sklearn.__version__, "actual_hyperparameters": self._hyperparameters,
            "prediction_diagnostics": self._prediction_diagnostics,
            "numeric_imputer_medians": self._preprocessor.medians,
            "category_counts": self._preprocessor.category_counts,
            "ordinal_encoder": self._preprocessor.audit_state()["ordinal_encoder"],
            "selected_source_features": list(self.selected_features),
            "tree_complexity": _tree_complexity(self._estimator),
            "fitted_tree_sha256": self._tree_state_sha256,
            "validation_labels_passed_to_fit": False,
            "early_stopping_validation_used": False,
        }
        return json.loads(json.dumps(_to_builtin(state), sort_keys=True))


def build_model(features: Iterable[str] | None = None, control: str | None = None) -> RegressionRound2Model:
    """Build an exact R1-C/R1-D control or one frozen structured HGB feature subset."""
    if control not in {None, "R1_C", "R1_D"}:
        raise ValueError("control must be None, 'R1_C', or 'R1_D'")
    expected = TWO_FEATURES if control == "R1_C" else EXPANDED_FEATURES if control == "R1_D" else None
    if features is None:
        if expected is None:
            raise ValueError("Feature variants require an explicit selected feature list")
        selected = tuple(expected)
    else:
        raw = tuple(features)
        if len(set(raw)) != len(raw):
            raise ValueError("Selected feature list contains duplicates")
        unknown = set(raw) - set(EXPANDED_FEATURES)
        if unknown:
            raise ValueError(f"Ineligible or unsupported source features: {sorted(unknown)}")
        selected = tuple(name for name in EXPANDED_FEATURES if name in raw)
    if control:
        if selected != expected:
            raise ValueError(f"{control} control requires its exact frozen feature set")
    else:
        missing_core = [name for name in TWO_FEATURES if name not in selected]
        if missing_core:
            raise ValueError(f"Feature ablations must retain the Round 1 core features: {missing_core}")
        if selected == TWO_FEATURES:
            raise ValueError("Use control='R1_C' for the exact two-feature control")
    return RegressionRound2Model(selected, control)


__all__ = ["RegressionRound2Model", "build_model"]
