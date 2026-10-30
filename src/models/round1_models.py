"""Leakage-safe estimator wrappers for TP2 Regression Round 1.

The module deliberately implements only the preregistered baseline and model
families. It does not load data, split rows, train a full model, or write model
artifacts; the experiment runner owns those choices.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import re
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


TWO_FEATURES = ("impact_load", "entry_friction")
EXPANDED_FEATURES = (
    "impact_load",
    "entry_friction",
    "cycle_code",
    "scope_pressure",
    "text_load",
    "scope_trace",
    "org_spread",
    "module_spread",
    "telemetry_flag",
    "org_code",
    "module_code",
    "pattern_code",
)
NUMERIC_FEATURES = (
    "impact_load",
    "entry_friction",
    "cycle_code",
    "scope_pressure",
    "text_load",
)
MIXED_COUNT_FEATURES = ("scope_trace", "org_spread", "module_spread")
CATEGORICAL_FEATURES = ("org_code", "module_code", "pattern_code")
TELEMETRY_FEATURE = "telemetry_flag"
_KIND_NAMES = {"global_median", "impact_median", "joint_median", "linear", "hgb"}
_COUNT_TOKEN = re.compile(
    r"^\s*(?P<approx>~)?\s*(?P<number>\d+(?:\.\d+)?)\s*(?P<unit>.*?)\s*$"
)
_TELEMETRY_MAP = {
    "inactive": 0.0,
    "false": 0.0,
    "n": 0.0,
    "0": 0.0,
    "active": 1.0,
    "true": 1.0,
    "y": 1.0,
    "1": 1.0,
}
_FIXED_HGB_PARAMS: dict[str, Any] = {
    "learning_rate": 0.05,
    "max_iter": 200,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 30,
    "l2_regularization": 1.0,
    "max_bins": 255,
    "early_stopping": False,
    "random_state": 42,
}


def _is_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _number_array(series: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return numeric values, missing flags, and invalid-format flags."""
    missing = series.map(_is_missing).to_numpy(dtype=bool)
    text_blank = series.map(lambda value: isinstance(value, str) and not value.strip()).to_numpy(dtype=bool)
    missing |= text_blank
    numeric = pd.to_numeric(series.mask(missing), errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(numeric)
    invalid = (~missing) & (~finite)
    numeric[~finite] = np.nan
    return numeric, missing.astype(float), invalid.astype(float)


def _mixed_count_arrays(
    series: pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse counts without guessing; only numbers and an optional ``nodes`` unit are accepted."""
    n = len(series)
    value = np.full(n, np.nan, dtype=float)
    approximate = np.zeros(n, dtype=float)
    suffix_present = np.zeros(n, dtype=float)
    parse_fail = np.zeros(n, dtype=float)
    missing = np.zeros(n, dtype=float)
    for position, raw in enumerate(series.to_numpy(dtype=object)):
        if _is_missing(raw) or (isinstance(raw, str) and not raw.strip()):
            missing[position] = 1.0
            continue
        match = _COUNT_TOKEN.fullmatch(str(raw))
        if match is None:
            parse_fail[position] = 1.0
            continue
        unit = match.group("unit").strip().casefold()
        suffix_present[position] = float(bool(unit))
        if unit not in {"", "nodes"}:
            parse_fail[position] = 1.0
            continue
        parsed = float(match.group("number"))
        if not math.isfinite(parsed):
            parse_fail[position] = 1.0
            continue
        value[position] = parsed
        approximate[position] = float(match.group("approx") is not None)
    return value, approximate, suffix_present, parse_fail, missing


def _telemetry_arrays(series: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map the observed binary spellings and retain unknown/missing flags."""
    n = len(series)
    value = np.full(n, np.nan, dtype=float)
    unknown = np.zeros(n, dtype=float)
    missing = np.zeros(n, dtype=float)
    for position, raw in enumerate(series.to_numpy(dtype=object)):
        if _is_missing(raw) or (isinstance(raw, str) and not raw.strip()):
            missing[position] = 1.0
            continue
        normalized = str(raw).strip().casefold()
        if normalized not in _TELEMETRY_MAP:
            unknown[position] = 1.0
            continue
        value[position] = _TELEMETRY_MAP[normalized]
    return value, unknown, missing


def _category_array(series: pd.Series) -> np.ndarray:
    normalized = np.empty(len(series), dtype=object)
    for position, raw in enumerate(series.to_numpy(dtype=object)):
        if _is_missing(raw) or (isinstance(raw, str) and not raw.strip()):
            normalized[position] = np.nan
        else:
            normalized[position] = str(raw).strip()
    return normalized


def _to_builtin(value: Any) -> Any:
    """Convert audit metadata to JSON-safe Python scalars and containers."""
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_to_builtin(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _to_builtin(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _fingerprint_value(value: Any) -> str:
    if _is_missing(value):
        return "null"
    if isinstance(value, (bool, np.bool_)):
        return "bool:" + ("1" if value else "0")
    if isinstance(value, numbers.Integral):
        return f"int:{int(value)}"
    if isinstance(value, numbers.Real):
        value_float = float(value)
        if not math.isfinite(value_float):
            return "float:" + repr(value_float)
        return "float:" + value_float.hex()
    return "str:" + str(value)


def _fit_fingerprint(X: pd.DataFrame, feature_names: tuple[str, ...], y: np.ndarray) -> str:
    """Hash the exact selected training inputs and target without retaining rows."""
    digest = hashlib.sha256()

    def add_text(text: str) -> None:
        encoded = text.encode("utf-8", errors="surrogatepass")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)

    add_text(json.dumps(feature_names, ensure_ascii=False, separators=(",", ":")))
    add_text(str(len(X)))
    for column in feature_names:
        for value in X[column].to_numpy(dtype=object):
            add_text(_fingerprint_value(value))
    for value in y:
        add_text(_fingerprint_value(float(value)))
    return digest.hexdigest()


def _validate_dataframe(X: pd.DataFrame) -> None:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame so feature names can be audited")
    if X.columns.has_duplicates:
        raise ValueError("X has duplicate column names")


def _target_array(y: Any, expected_rows: int) -> np.ndarray:
    target = np.asarray(y)
    if target.ndim != 1:
        raise ValueError("y must be one-dimensional")
    if len(target) != expected_rows:
        raise ValueError("X and y row counts differ")
    try:
        target = target.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("y must contain numeric regression targets") from exc
    if not np.isfinite(target).all():
        raise ValueError("y must be finite and non-missing")
    if expected_rows == 0:
        raise ValueError("Cannot fit on an empty training set")
    return target


class _HGBPreprocessor:
    """Training-fold-fitted numeric imputation and native categorical encoding."""

    def __init__(self, feature_preset: str):
        self.feature_preset = feature_preset
        self.encoder: OrdinalEncoder | None = None
        self.medians: dict[str, float] = {}
        self.transformed_feature_names: list[str] = []
        self.numeric_feature_names: list[str] = []
        self.categorical_positions: list[int] = []
        self.category_counts: dict[str, dict[str, Any]] = {}
        self.seen_categories: dict[str, set[str]] = {}

    def _raw_matrix(
        self, X: pd.DataFrame, *, fit_categories: bool
    ) -> tuple[np.ndarray, list[str], list[str], list[str], dict[str, np.ndarray], dict[str, np.ndarray]]:
        if self.feature_preset == "two":
            columns: list[np.ndarray] = []
            names: list[str] = []
            for feature in TWO_FEATURES:
                values, missing, invalid = _number_array(X[feature])
                if np.any(missing) or np.any(invalid):
                    raise ValueError(f"two-feature HGB requires complete numeric {feature!r} values")
                columns.append(values)
                names.append(feature)
            return np.column_stack(columns), names, names.copy(), [], {}, {}

        numeric_columns: list[np.ndarray] = []
        numeric_names: list[str] = []
        categorical_columns: list[np.ndarray] = []
        categorical_names: list[str] = []
        category_raw: dict[str, np.ndarray] = {}
        category_missing: dict[str, np.ndarray] = {}
        for feature in EXPANDED_FEATURES:
            if feature in NUMERIC_FEATURES:
                values, missing, invalid = _number_array(X[feature])
                numeric_columns.extend((values, invalid, missing))
                numeric_names.extend((feature, f"{feature}__parsefail", f"{feature}__missing"))
            elif feature in MIXED_COUNT_FEATURES:
                values, approximate, suffix_present, invalid, missing = _mixed_count_arrays(X[feature])
                numeric_columns.extend((values, approximate, suffix_present, invalid, missing))
                numeric_names.extend(
                    (
                        feature,
                        f"{feature}__approximate",
                        f"{feature}__suffix_present",
                        f"{feature}__parsefail",
                        f"{feature}__missing",
                    )
                )
            elif feature == TELEMETRY_FEATURE:
                values, unknown, missing = _telemetry_arrays(X[feature])
                numeric_columns.extend((values, unknown, missing))
                numeric_names.extend(
                    (feature, f"{feature}__unknown", f"{feature}__missing")
                )
            elif feature in CATEGORICAL_FEATURES:
                raw = _category_array(X[feature])
                category_raw[feature] = raw
                category_missing[feature] = np.array(
                    [value is np.nan or _is_missing(value) for value in raw], dtype=float
                )
                categorical_columns.append(raw)
                categorical_names.append(feature)

        if fit_categories:
            self.encoder = OrdinalEncoder(
                min_frequency=30,
                max_categories=254,
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                encoded_missing_value=-1,
                dtype=np.float64,
            )
            category_matrix = np.column_stack(categorical_columns)
            self.encoder.fit(category_matrix)
            for index, feature in enumerate(categorical_names):
                categories = self.encoder.categories_[index]
                seen = {
                    str(value)
                    for value in categories
                    if not _is_missing(value)
                }
                self.seen_categories[feature] = seen

        if categorical_columns:
            if self.encoder is None:
                raise RuntimeError("Categorical encoder was not fitted")
            encoded = self.encoder.transform(np.column_stack(categorical_columns))
        else:
            encoded = np.empty((len(X), 0), dtype=float)

        matrix_columns: list[np.ndarray] = []
        all_names: list[str] = []
        numeric_output_names: list[str] = []
        categorical_positions: list[int] = []
        numeric_index = 0
        categorical_index = 0
        for feature in EXPANDED_FEATURES:
            if feature in NUMERIC_FEATURES:
                for _ in range(3):
                    matrix_columns.append(numeric_columns[numeric_index])
                    all_names.append(numeric_names[numeric_index])
                    numeric_output_names.append(numeric_names[numeric_index])
                    numeric_index += 1
            elif feature in MIXED_COUNT_FEATURES:
                for _ in range(5):
                    matrix_columns.append(numeric_columns[numeric_index])
                    all_names.append(numeric_names[numeric_index])
                    numeric_output_names.append(numeric_names[numeric_index])
                    numeric_index += 1
            elif feature == TELEMETRY_FEATURE:
                for _ in range(3):
                    matrix_columns.append(numeric_columns[numeric_index])
                    all_names.append(numeric_names[numeric_index])
                    numeric_output_names.append(numeric_names[numeric_index])
                    numeric_index += 1
            elif feature in CATEGORICAL_FEATURES:
                categorical_positions.append(len(all_names))
                matrix_columns.append(encoded[:, categorical_index])
                all_names.append(f"{feature}__ordinal")
                categorical_index += 1
                matrix_columns.append(category_missing[feature])
                all_names.append(f"{feature}__missing")
                numeric_output_names.append(f"{feature}__missing")

        matrix = np.column_stack(matrix_columns).astype(float, copy=False)
        if fit_categories:
            for index, feature in enumerate(categorical_names):
                raw = category_raw[feature]
                observed = [str(value) for value in raw if not _is_missing(value)]
                freq = Counter(observed)
                encoded_train = encoded[:, index]
                represented = np.unique(encoded_train[encoded_train >= 0])
                infrequent = getattr(self.encoder, "infrequent_categories_", None)
                infrequent_count = 0
                if infrequent is not None and infrequent[index] is not None:
                    infrequent_count = len(infrequent[index])
                self.category_counts[feature] = {
                    "observed_unique": len(freq),
                    "encoded_categories": int(len(represented)),
                    "infrequent_categories": int(infrequent_count),
                    "training_missing_rows": int(sum(_is_missing(value) for value in raw)),
                    "training_rows": len(raw),
                    "min_frequency": 30,
                    "max_categories": 254,
                }
        return matrix, all_names, numeric_output_names, categorical_names, category_raw, category_missing

    def fit_transform(self, X: pd.DataFrame) -> tuple[np.ndarray, list[bool]]:
        matrix, names, numeric_names, _, _, _ = self._raw_matrix(X, fit_categories=True)
        self.transformed_feature_names = names
        self.numeric_feature_names = numeric_names
        categorical_name_set = {f"{name}__ordinal" for name in CATEGORICAL_FEATURES}
        self.categorical_positions = [
            position for position, name in enumerate(names) if name in categorical_name_set
        ]
        numeric_positions = [
            position for position, name in enumerate(names) if name not in categorical_name_set
        ]
        self.medians = {}
        for position in numeric_positions:
            column = matrix[:, position]
            finite = column[np.isfinite(column)]
            median = float(np.median(finite)) if len(finite) else 0.0
            self.medians[names[position]] = median
            column[~np.isfinite(column)] = median
        categorical_mask = [position in self.categorical_positions for position in range(len(names))]
        return matrix, categorical_mask

    def transform(self, X: pd.DataFrame) -> tuple[np.ndarray, dict[str, int]]:
        matrix, names, _, categorical_names, category_raw, _ = self._raw_matrix(
            X, fit_categories=False
        )
        if names != self.transformed_feature_names:
            raise RuntimeError("Transformed feature schema changed after fit")
        unknown_counts: dict[str, int] = {}
        for feature in categorical_names:
            raw = category_raw[feature]
            seen = self.seen_categories[feature]
            unknown_counts[feature] = int(
                sum(not _is_missing(value) and str(value) not in seen for value in raw)
            )
        categorical_set = set(self.categorical_positions)
        for position, name in enumerate(names):
            if position in categorical_set:
                continue
            column = matrix[:, position]
            column[~np.isfinite(column)] = self.medians[name]
        return matrix, unknown_counts

    def audit_state(self) -> dict[str, Any]:
        return {
            "transformed_feature_names": list(self.transformed_feature_names),
            "numeric_imputer_medians": dict(self.medians),
            "categorical_feature_names": [
                self.transformed_feature_names[position]
                for position in self.categorical_positions
            ],
            "category_counts": dict(self.category_counts),
            "ordinal_encoder": {
                "min_frequency": 30,
                "max_categories": 254,
                "handle_unknown": "use_encoded_value",
                "unknown_value": -1,
                "encoded_missing_value": -1,
                "dtype": "float64",
            },
        }


class Round1Model:
    """Small estimator-compatible wrapper with reproducible audit metadata."""

    def __init__(self, recipe: dict[str, Any]):
        if not isinstance(recipe, dict):
            raise TypeError("recipe must be a dictionary")
        unknown_keys = set(recipe) - {"id", "kind", "features", "min_support", "params"}
        if unknown_keys:
            raise ValueError(f"Unsupported recipe fields: {sorted(unknown_keys)}")
        self.model_id = str(recipe.get("id", "round1"))
        self.kind = recipe.get("kind")
        if self.kind not in _KIND_NAMES:
            raise ValueError(f"kind must be one of {sorted(_KIND_NAMES)}")
        self.feature_preset = recipe.get("features", "two")
        if self.feature_preset not in {"two", "expanded"}:
            raise ValueError("features must be 'two' or 'expanded'")
        if self.kind == "linear" and self.feature_preset != "two":
            raise ValueError("linear is restricted to the two numeric features")
        self.min_support = recipe.get("min_support", 30)
        if not isinstance(self.min_support, numbers.Integral) or int(self.min_support) != 30:
            raise ValueError("Round 1 lookup baselines use the frozen support threshold 30")
        self.min_support = int(self.min_support)
        params = recipe.get("params", {})
        if not isinstance(params, dict):
            raise TypeError("params must be a dictionary")
        self.hgb_params: dict[str, Any] = {}
        if self.kind == "hgb":
            allowed = set(_FIXED_HGB_PARAMS) | {"loss", "interaction_cst"}
            unsupported = set(params) - allowed
            if unsupported:
                raise ValueError(f"Unsupported HGB parameters: {sorted(unsupported)}")
            loss = params.get("loss", "squared_error")
            if loss not in {"squared_error", "absolute_error"}:
                raise ValueError("HGB loss must be squared_error or absolute_error")
            for name, fixed_value in _FIXED_HGB_PARAMS.items():
                if name in params and params[name] != fixed_value:
                    raise ValueError(f"HGB {name} is frozen at {fixed_value!r}")
            interaction = params.get("interaction_cst", None)
            if interaction not in {None, "no_interactions"}:
                raise ValueError("interaction_cst may only be None or 'no_interactions'")
            self.hgb_params = {**_FIXED_HGB_PARAMS, "loss": loss}
            if interaction is not None:
                self.hgb_params["interaction_cst"] = interaction
        elif params:
            raise ValueError("Only HGB recipes accept model parameters")
        self._fitted = False
        self._state: dict[str, Any] = {}
        self._prediction_diagnostics: dict[str, Any] = {}

    def _required_feature_names(self) -> tuple[str, ...]:
        if self.kind == "global_median":
            return ()
        if self.kind == "impact_median":
            return ("impact_load",)
        if self.kind == "joint_median":
            return TWO_FEATURES
        return TWO_FEATURES if self.feature_preset == "two" else EXPANDED_FEATURES

    def _selected(self, X: pd.DataFrame) -> pd.DataFrame:
        _validate_dataframe(X)
        names = self._required_feature_names()
        missing = [name for name in names if name not in X.columns]
        if missing:
            raise ValueError(f"X is missing required feature columns: {missing}")
        return X.loc[:, list(names)]

    @staticmethod
    def _group_component(value: Any) -> float | None:
        if _is_missing(value):
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if math.isfinite(parsed) else None

    def _fit_lookup(self, X: pd.DataFrame, y: np.ndarray) -> None:
        self._global_median = float(np.median(y))
        self._group_medians: dict[tuple[float, ...], float] = {}
        self._group_counts: dict[tuple[float, ...], int] = {}
        self._impact_medians: dict[tuple[float, ...], float] = {}
        self._impact_counts: dict[tuple[float, ...], int] = {}
        self._entry_medians: dict[tuple[float, ...], float] = {}
        self._entry_counts: dict[tuple[float, ...], int] = {}

        def summarize(columns: tuple[str, ...]) -> tuple[dict[tuple[float, ...], float], dict[tuple[float, ...], int]]:
            values: dict[tuple[float, ...], list[float]] = defaultdict(list)
            for row, target in zip(X.loc[:, list(columns)].itertuples(index=False, name=None), y):
                key_values = tuple(self._group_component(value) for value in row)
                if any(value is None for value in key_values):
                    continue
                key = tuple(float(value) for value in key_values if value is not None)
                values[key].append(float(target))
            counts = {key: len(group_y) for key, group_y in values.items()}
            medians = {key: float(np.median(group_y)) for key, group_y in values.items()}
            return medians, counts

        if self.kind == "impact_median":
            all_medians, all_counts = summarize(("impact_load",))
            kept = {key for key, count in all_counts.items() if count >= self.min_support}
            self._group_medians = {key: value for key, value in all_medians.items() if key in kept}
            self._group_counts = {key: all_counts[key] for key in kept}
        elif self.kind == "joint_median":
            joint_medians, joint_counts = summarize(TWO_FEATURES)
            impact_medians, impact_counts = summarize(("impact_load",))
            entry_medians, entry_counts = summarize(("entry_friction",))
            self._group_medians = {
                key: value for key, value in joint_medians.items() if joint_counts[key] >= self.min_support
            }
            self._group_counts = {
                key: count for key, count in joint_counts.items() if count >= self.min_support
            }
            self._impact_medians = {
                key: value for key, value in impact_medians.items() if impact_counts[key] >= self.min_support
            }
            self._impact_counts = {
                key: count for key, count in impact_counts.items() if count >= self.min_support
            }
            self._entry_medians = {
                key: value for key, value in entry_medians.items() if entry_counts[key] >= self.min_support
            }
            self._entry_counts = {
                key: count for key, count in entry_counts.items() if count >= self.min_support
            }

    def fit(self, X: pd.DataFrame, y: Any) -> "Round1Model":
        selected = self._selected(X)
        target = _target_array(y, len(selected))
        feature_names = tuple(selected.columns)
        fingerprint = _fit_fingerprint(selected, feature_names, target)
        self._fit_rows = len(target)
        self._fit_fingerprint = fingerprint
        self._input_feature_names = list(feature_names)
        self._transformed_feature_names: list[str] = []
        self._imputer_medians: dict[str, float] = {}
        self._category_counts: dict[str, dict[str, Any]] = {}
        self._hyperparameters: dict[str, Any] = {}

        if self.kind in {"global_median", "impact_median", "joint_median"}:
            self._fit_lookup(selected, target)
            self._transformed_feature_names = list(feature_names)
            self._hyperparameters = {"kind": self.kind, "min_support": self.min_support}
        elif self.kind == "linear":
            numeric = np.column_stack(
                [_number_array(selected[name])[0] for name in TWO_FEATURES]
            )
            missing = ~np.isfinite(numeric)
            if np.any(missing):
                raise ValueError("linear requires complete numeric impact_load and entry_friction")
            self._scaler = StandardScaler()
            scaled = self._scaler.fit_transform(numeric)
            self._estimator = LinearRegression()
            self._estimator.fit(scaled, target)
            self._transformed_feature_names = list(TWO_FEATURES)
            self._hyperparameters = {
                "estimator": "LinearRegression",
                "fit_intercept": True,
                "scaler": "StandardScaler",
                "scaler_with_mean": True,
                "scaler_with_std": True,
            }
        elif self.kind == "hgb":
            self._preprocessor = _HGBPreprocessor(self.feature_preset)
            matrix, categorical_mask = self._preprocessor.fit_transform(selected)
            actual_params = dict(self.hgb_params)
            actual_params["categorical_features"] = categorical_mask if any(categorical_mask) else None
            self._estimator = HistGradientBoostingRegressor(**actual_params)
            self._estimator.fit(matrix, target)
            self._transformed_feature_names = list(self._preprocessor.transformed_feature_names)
            self._imputer_medians = dict(self._preprocessor.medians)
            self._category_counts = dict(self._preprocessor.category_counts)
            self._hyperparameters = {
                key: _to_builtin(value)
                for key, value in self._estimator.get_params(deep=False).items()
            }
        self._fitted = True
        self._prediction_diagnostics = {"n_predictions": 0}
        return self

    def _predict_lookup(self, X: pd.DataFrame) -> np.ndarray:
        predictions = np.full(len(X), self._global_median, dtype=float)
        fallback_counts: Counter[str] = Counter()
        if self.kind == "global_median":
            fallback_counts["global"] = len(X)
            self._prediction_diagnostics = {
                "n_predictions": int(len(X)),
                "fallback_counts": dict(sorted(fallback_counts.items())),
            }
            return predictions
        for position, row in enumerate(X.itertuples(index=False, name=None)):
            if self.kind == "impact_median":
                impact = self._group_component(row[0])
                key = None if impact is None else (impact,)
                if key in self._group_medians:
                    predictions[position] = self._group_medians[key]
                    fallback_counts["impact"] += 1
                else:
                    fallback_counts["global"] += 1
                continue

            impact = self._group_component(row[0])
            entry = self._group_component(row[1])
            joint_key = None if impact is None or entry is None else (impact, entry)
            impact_key = None if impact is None else (impact,)
            entry_key = None if entry is None else (entry,)
            if joint_key in self._group_medians:
                predictions[position] = self._group_medians[joint_key]
                fallback_counts["joint"] += 1
            elif impact_key in self._impact_medians:
                predictions[position] = self._impact_medians[impact_key]
                fallback_counts["impact"] += 1
            elif entry_key in self._entry_medians:
                predictions[position] = self._entry_medians[entry_key]
                fallback_counts["entry_friction"] += 1
            else:
                fallback_counts["global"] += 1
        self._prediction_diagnostics = {
            "n_predictions": int(len(X)),
            "fallback_counts": dict(sorted(fallback_counts.items())),
        }
        return predictions

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model must be fitted before predict")
        selected = self._selected(X)
        if self.kind in {"global_median", "impact_median", "joint_median"}:
            return self._predict_lookup(selected)
        if self.kind == "linear":
            numeric = np.column_stack(
                [_number_array(selected[name])[0] for name in TWO_FEATURES]
            )
            if not np.isfinite(numeric).all():
                raise ValueError("linear requires complete numeric impact_load and entry_friction")
            predictions = self._estimator.predict(self._scaler.transform(numeric))
            self._prediction_diagnostics = {"n_predictions": int(len(selected))}
            return np.asarray(predictions, dtype=float)
        matrix, unknown_counts = self._preprocessor.transform(selected)
        predictions = self._estimator.predict(matrix)
        self._prediction_diagnostics = {
            "n_predictions": int(len(selected)),
            "unknown_category_rows": dict(sorted(unknown_counts.items())),
        }
        return np.asarray(predictions, dtype=float)

    def prediction_diagnostics(self) -> dict[str, Any]:
        return json.loads(json.dumps(_to_builtin(self._prediction_diagnostics), sort_keys=True))

    def audit_state(self) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("Model must be fitted before audit_state")
        state: dict[str, Any] = {
            "id": self.model_id,
            "kind": self.kind,
            "feature_preset": self.feature_preset,
            "input_feature_names": list(self._input_feature_names),
            "transformed_feature_names": list(self._transformed_feature_names),
            "fit_rows": int(self._fit_rows),
            "fit_data_sha256": self._fit_fingerprint,
            "sklearn_version": sklearn.__version__,
            "actual_hyperparameters": self._hyperparameters,
            "prediction_diagnostics": self._prediction_diagnostics,
        }
        if self.kind in {"global_median", "impact_median", "joint_median"}:
            state["min_support"] = self.min_support
            state["global_target_median"] = self._global_median
            state["supported_group_counts"] = {
                "primary": len(self._group_counts),
                "impact": len(self._impact_counts),
                "entry_friction": len(self._entry_counts),
            }

            def entries(
                medians: dict[tuple[float, ...], float], counts: dict[tuple[float, ...], int]
            ) -> list[dict[str, Any]]:
                return [
                    {"key": list(key), "rows": int(counts[key]), "median": medians[key]}
                    for key in sorted(medians)
                ]

            state["learned_group_medians"] = {
                "primary": entries(self._group_medians, self._group_counts),
                "impact": entries(self._impact_medians, self._impact_counts),
                "entry_friction": entries(self._entry_medians, self._entry_counts),
            }
        elif self.kind == "linear":
            state["standard_scaler"] = {
                "mean": self._scaler.mean_.tolist(),
                "scale": self._scaler.scale_.tolist(),
                "variance": self._scaler.var_.tolist(),
            }
            state["linear_coefficients"] = self._estimator.coef_.tolist()
            state["linear_intercept"] = float(self._estimator.intercept_)
        elif self.kind == "hgb":
            state["numeric_imputer_medians"] = dict(self._imputer_medians)
            state["category_counts"] = dict(self._category_counts)
            state["ordinal_encoder"] = self._preprocessor.audit_state()["ordinal_encoder"]
        return json.loads(json.dumps(_to_builtin(state), sort_keys=True))


def build_model(recipe: dict[str, Any]) -> Round1Model:
    """Build an unfitted Round 1 estimator from the runner's explicit recipe."""
    return Round1Model(recipe)


__all__ = [
    "CATEGORICAL_FEATURES",
    "EXPANDED_FEATURES",
    "MIXED_COUNT_FEATURES",
    "TWO_FEATURES",
    "Round1Model",
    "build_model",
]
