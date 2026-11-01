"""Research Next V1 N04: fit-only domain features and weighted R22 design."""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost
from xgboost import XGBRegressor

from src.models.classification_models import _text_values
from src.models.optimization_v3_batch1_r22 import (
    FEATURE_NAMES as R22_FEATURE_NAMES,
    XGB_PARAMETERS,
)
from src.models.regression_research_text import (
    STRUCTURED_FEATURES, _SVD_PARAMS, _TFIDF_PARAMS, _array_digest,
    _json_digest, _text_documents, RegressionTextModel,
)
from src.models.round1_models import (
    CATEGORICAL_FEATURES, MIXED_COUNT_FEATURES, NUMERIC_FEATURES,
    TELEMETRY_FEATURE, _is_missing, _mixed_count_arrays, _number_array,
    _telemetry_arrays,
)


CARD_ID = "N04"
DOMAIN_STRUCTURED_FEATURES = tuple(name for name in STRUCTURED_FEATURES if name != "cycle_code")
DOMAIN_CATEGORICAL_FEATURES = tuple(name for name in CATEGORICAL_FEATURES if name in DOMAIN_STRUCTURED_FEATURES)
DOMAIN_LR_PARAMETERS: dict[str, Any] = {
    "C": 0.1, "solver": "liblinear", "max_iter": 2000, "tol": 1e-4,
    "class_weight": "balanced", "random_state": 42,
}
DOMAIN_PROBABILITY_CLIP = (1e-6, 1.0 - 1e-6)
SOURCE_WEIGHT_CLIP = (0.25, 4.0)
DOMAIN_AUC_RANGE = (0.55, 0.95)
MINIMUM_ESS_FRACTION = 0.5
MAXIMUM_SOURCE_CLIP_FRACTION = 0.2
RECENT_CYCLES = 4
SHUFFLE_SEED_OFFSET = 42


def _category_token(value: Any) -> str:
    return "<TP2_MISSING_CATEGORY>" if _is_missing(value) else "V:" + str(value).strip()


def _domain_numeric_block(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    columns: list[np.ndarray] = []
    names: list[str] = []
    for feature in DOMAIN_STRUCTURED_FEATURES:
        if feature == TELEMETRY_FEATURE:
            value, unknown, missing = _telemetry_arrays(frame[feature])
            block = (value, unknown, missing)
            block_names = (feature, f"{feature}__unknown", f"{feature}__missing")
        elif feature in MIXED_COUNT_FEATURES:
            value, approximate, suffix, invalid, missing = _mixed_count_arrays(frame[feature])
            block = (value, approximate, suffix, invalid, missing)
            block_names = (feature, f"{feature}__approximate", f"{feature}__suffix_present",
                           f"{feature}__parsefail", f"{feature}__missing")
        elif feature in NUMERIC_FEATURES:
            value, missing, invalid = _number_array(frame[feature])
            block = (value, invalid, missing)
            block_names = (feature, f"{feature}__parsefail", f"{feature}__missing")
        else:
            continue
        columns.extend(np.asarray(item, dtype=np.float64) for item in block)
        names.extend(block_names)
    if not columns:
        return np.empty((len(frame), 0), dtype=np.float64), names
    return np.column_stack(columns).astype(np.float64, copy=False), names


class DomainFeatureBuilder:
    """Structured/OHE + word TF-IDF transform, fitted only on one domain-fold train set."""

    def __init__(self) -> None:
        self._fitted = False
        self._fit_audit: dict[str, Any] = {}

    @staticmethod
    def _select(frame: pd.DataFrame) -> pd.DataFrame:
        expected = set(DOMAIN_STRUCTURED_FEATURES) | {"signal_stream"}
        if (not isinstance(frame, pd.DataFrame) or frame.columns.has_duplicates
                or set(frame.columns) != expected or "cycle_code" in frame.columns
                or "record_id" in frame.columns or "threat_rank" in frame.columns
                or "artifact_signal" in frame.columns):
            raise ValueError("N04 domain inputs must exclude ID, cycle and all task targets")
        return frame.loc[:, [*DOMAIN_STRUCTURED_FEATURES, "signal_stream"]].copy()

    def fit_transform(self, frame: pd.DataFrame) -> sparse.csr_matrix:
        selected = self._select(frame)
        numeric, self._numeric_names = _domain_numeric_block(selected)
        self._numeric_medians = np.zeros(numeric.shape[1], dtype=np.float64)
        for column in range(numeric.shape[1]):
            finite = numeric[:, column][np.isfinite(numeric[:, column])]
            self._numeric_medians[column] = float(np.median(finite)) if len(finite) else 0.0
            numeric[~np.isfinite(numeric[:, column]), column] = self._numeric_medians[column]
        self._scaler = StandardScaler()
        numeric_scaled = self._scaler.fit_transform(numeric) if numeric.shape[1] else numeric

        categorical = np.column_stack([
            np.fromiter((_category_token(value) for value in selected[name].to_numpy(dtype=object)),
                        dtype=object, count=len(selected))
            for name in DOMAIN_CATEGORICAL_FEATURES
        ]) if DOMAIN_CATEGORICAL_FEATURES else np.empty((len(selected), 0), dtype=object)
        self._encoder = OneHotEncoder(min_frequency=30, max_categories=254,
                                      handle_unknown="infrequent_if_exist", sparse_output=True,
                                      dtype=np.float64)
        category_matrix = (self._encoder.fit_transform(categorical) if categorical.shape[1]
                           else sparse.csr_matrix((len(selected), 0), dtype=np.float64))
        docs, text_missing = _text_values(selected["signal_stream"])
        self._vectorizer = None
        try:
            self._vectorizer = TfidfVectorizer(**dict(_TFIDF_PARAMS))
            text_matrix = self._vectorizer.fit_transform(docs).tocsr()
        except ValueError as exc:
            message = str(exc).casefold()
            if not any(token in message for token in ("empty vocabulary", "after pruning, no terms remain")):
                raise
            text_matrix = sparse.csr_matrix((len(docs), 0), dtype=np.float64)
            self._vectorizer = None
        missing_column = sparse.csr_matrix(np.asarray(text_missing, dtype=np.float64).reshape(-1, 1))
        matrix = sparse.hstack((sparse.csr_matrix(numeric_scaled), category_matrix, text_matrix,
                                missing_column), format="csr", dtype=np.float64)
        if matrix.shape[0] != len(selected) or not np.isfinite(matrix.data).all():
            raise RuntimeError("N04 domain training design is misaligned or nonfinite")
        self._fit_audit = {
            "domain_feature_names": list(DOMAIN_STRUCTURED_FEATURES) + ["signal_stream"],
            "cycle_code_used": False, "record_id_used": False, "task_target_used": False,
            "fit_rows": int(len(selected)), "numeric_names": list(self._numeric_names),
            "numeric_medians": self._numeric_medians.tolist(),
            "numeric_scaler_mean_sha256": _array_digest(self._scaler.mean_) if numeric.shape[1] else None,
            "numeric_scaler_scale_sha256": _array_digest(self._scaler.scale_) if numeric.shape[1] else None,
            "categorical_names": list(DOMAIN_CATEGORICAL_FEATURES),
            "category_width": int(category_matrix.shape[1]), "text_width": int(text_matrix.shape[1]),
            "text_missing_rows": int(np.asarray(text_missing, dtype=bool).sum()),
            "tfidf_parameters": {key: (list(value) if isinstance(value, tuple)
                                       else value.__name__ if isinstance(value, type) else value)
                                 for key, value in _TFIDF_PARAMS.items()},
            "design_shape": [int(x) for x in matrix.shape], "design_nnz": int(matrix.nnz),
            "design_sha256": _array_digest(matrix.data),
        }
        self._fitted = True
        return matrix

    def transform(self, frame: pd.DataFrame) -> sparse.csr_matrix:
        if not self._fitted:
            raise RuntimeError("N04 domain feature builder must be fitted before transform")
        selected = self._select(frame)
        numeric, names = _domain_numeric_block(selected)
        if names != self._numeric_names:
            raise RuntimeError("N04 domain numeric schema changed after fold fit")
        for column in range(numeric.shape[1]):
            numeric[~np.isfinite(numeric[:, column]), column] = self._numeric_medians[column]
        numeric_scaled = self._scaler.transform(numeric) if numeric.shape[1] else numeric
        categorical = np.column_stack([
            np.fromiter((_category_token(value) for value in selected[name].to_numpy(dtype=object)),
                        dtype=object, count=len(selected))
            for name in DOMAIN_CATEGORICAL_FEATURES
        ]) if DOMAIN_CATEGORICAL_FEATURES else np.empty((len(selected), 0), dtype=object)
        category_matrix = (self._encoder.transform(categorical) if categorical.shape[1]
                           else sparse.csr_matrix((len(selected), 0), dtype=np.float64))
        docs, text_missing = _text_values(selected["signal_stream"])
        text_matrix = (self._vectorizer.transform(docs).tocsr() if self._vectorizer is not None
                       else sparse.csr_matrix((len(docs), 0), dtype=np.float64))
        matrix = sparse.hstack((sparse.csr_matrix(numeric_scaled), category_matrix, text_matrix,
                                sparse.csr_matrix(np.asarray(text_missing, dtype=np.float64).reshape(-1, 1))),
                               format="csr", dtype=np.float64)
        if matrix.shape[1] != self._fit_audit["design_shape"][1] or not np.isfinite(matrix.data).all():
            raise RuntimeError("N04 domain score design differs from its fold-fitted schema")
        return matrix

    def audit_state(self) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("N04 domain feature builder is not fitted")
        return dict(self._fit_audit)


def domain_labels(cycles: Any, cutoff: int) -> np.ndarray:
    values = np.asarray(cycles, dtype=np.int64)
    if values.ndim != 1 or not len(values) or np.any(values > int(cutoff)):
        raise ValueError("N04 domain labels require nonempty fit-prefix cycles")
    unique = np.unique(values)
    if len(unique) < RECENT_CYCLES:
        raise ValueError("N04 fit prefix must contain four recent cycles")
    return np.isin(values, unique[-RECENT_CYCLES:]).astype(np.int8)


def density_ratio_weights(domain_probability: Any, labels: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply balanced-prior odds weights and normalize the complete fit vector to mean one."""
    p = np.asarray(domain_probability, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int8)
    if p.ndim != 1 or y.shape != p.shape or not np.isfinite(p).all() or not np.isin(y, (0, 1)).all():
        raise ValueError("N04 domain probabilities and labels must be finite aligned vectors")
    if set(np.unique(y)) != {0, 1} or np.any((p < 0) | (p > 1)):
        raise ValueError("N04 domain weights require both domains and probabilities in [0,1]")
    clipped_p = np.clip(p, *DOMAIN_PROBABILITY_CLIP)
    odds = clipped_p / (1.0 - clipped_p)
    source = y == 0
    source_unclipped = odds[source]
    source_weights = np.clip(source_unclipped, *SOURCE_WEIGHT_CLIP)
    before_normalization = np.ones(len(p), dtype=np.float64)
    before_normalization[source] = source_weights
    weights = before_normalization / before_normalization.mean()
    ess = float(weights.sum() ** 2 / np.square(weights).sum())
    at_bounds = (source_unclipped <= SOURCE_WEIGHT_CLIP[0]) | (source_unclipped >= SOURCE_WEIGHT_CLIP[1])
    diag = {
        "fit_rows": int(len(p)), "source_rows": int(source.sum()), "recent_rows": int((~source).sum()),
        "probability_clip": list(DOMAIN_PROBABILITY_CLIP), "source_odds_clip": list(SOURCE_WEIGHT_CLIP),
        "probability_min": float(p.min()), "probability_max": float(p.max()),
        "source_odds_raw_min": float(source_unclipped.min()), "source_odds_raw_max": float(source_unclipped.max()),
        "source_fraction_at_or_beyond_clip_bounds": float(at_bounds.mean()),
        "effective_sample_size": ess, "effective_sample_size_fraction": ess / len(p),
        "weight_mean": float(weights.mean()), "weight_min": float(weights.min()),
        "weight_max": float(weights.max()), "weight_sha256": _array_digest(weights),
        "recent_weight_before_normalization": 1.0, "all_weights_finite_positive": bool(
            np.isfinite(weights).all() and np.all(weights > 0)),
    }
    return weights, diag


def shuffle_weights_within_cycle(weights: Any, cycles: Any, seed: int) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64)
    group = np.asarray(cycles, dtype=np.int64)
    if values.ndim != 1 or group.shape != values.shape or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("N04 within-cycle shuffle inputs are invalid")
    shuffled = values.copy()
    rng = np.random.default_rng(int(seed))
    for cycle in np.unique(group):
        positions = np.flatnonzero(group == cycle)
        shuffled[positions] = rng.permutation(values[positions])
        if not np.array_equal(np.sort(shuffled[positions]), np.sort(values[positions])):
            raise RuntimeError("N04 within-cycle shuffle changed a cycle's exact weight multiset")
    return shuffled


def _complete_cycle_rank50(raw: Any, cycles: Any) -> np.ndarray:
    prediction = np.asarray(raw, dtype=np.float64)
    group = np.asarray(cycles, dtype=np.int64)
    if prediction.ndim != 1 or group.shape != prediction.shape or not np.isfinite(prediction).all():
        raise ValueError("N04 rank50 inputs must be finite aligned vectors")
    result = np.empty_like(prediction)
    for cycle in np.unique(group):
        mask = group == cycle
        pct = 100.0 * rankdata(prediction[mask], method="average") / int(mask.sum())
        result[mask] = np.clip(0.5 * prediction[mask] + 0.5 * pct, 0.0, 100.0)
    return result


def compose_r22_candidate(raw: Any, legacy: Any, cycles: Any) -> np.ndarray:
    raw_arr = np.asarray(raw, dtype=np.float64)
    baseline = np.asarray(legacy, dtype=np.float64)
    mapped = _complete_cycle_rank50(raw_arr, cycles)
    if raw_arr.shape != baseline.shape or not np.isfinite(baseline).all() or np.any((baseline < 0) | (baseline > 100)):
        raise ValueError("N04 R22 composition needs finite aligned scores and legacy [0,100]")
    return np.clip(0.75 * baseline + 0.25 * mapped, 0.0, 100.0)


class R22WeightedDesign:
    """Exact frozen R22 structured/OHE + word TF-IDF/SVD32 design, fit once per origin."""

    def __init__(self, cutoff: int):
        self.cutoff = int(cutoff)
        self._fitted = False

    @staticmethod
    def _select(frame: pd.DataFrame) -> pd.DataFrame:
        if (not isinstance(frame, pd.DataFrame) or frame.columns.has_duplicates
                or set(frame.columns) != set(R22_FEATURE_NAMES) or len(frame.columns) != len(R22_FEATURE_NAMES)):
            raise ValueError("N04 task design must receive the exact frozen R22 feature allowlist")
        if {"record_id", "artifact_signal", "threat_rank"} & set(frame.columns):
            raise ValueError("N04 task feature allowlist contains an ID or task target")
        return frame.loc[:, list(R22_FEATURE_NAMES)].copy()

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        selected = self._select(frame)
        cycles = selected["cycle_code"].to_numpy(dtype=np.int64)
        if np.any(cycles > self.cutoff) or not selected.index.is_unique or not selected.index.is_monotonic_increasing:
            raise ValueError("N04 task design fit rows violate the R22 temporal source boundary")
        docs, missing = _text_documents(selected["signal_stream"])
        self._vectorizer, text_matrix, self._text_fallback = RegressionTextModel._fit_vectorizer(docs)
        self._svd_components_actual = min(32, text_matrix.shape[1] - 1) if text_matrix.shape[1] > 1 else 0
        if self._svd_components_actual:
            from sklearn.decomposition import TruncatedSVD
            self._svd = TruncatedSVD(n_components=self._svd_components_actual, **dict(_SVD_PARAMS))
            text_dense = self._svd.fit_transform(text_matrix)
        else:
            self._svd = None
            text_dense = np.empty((len(selected), 0), dtype=np.float64)
        structured, self._processor, self._scaler, self._encoder, self._numeric_positions, structured_state = (
            RegressionTextModel._structured_fit(selected)
        )
        structured_dense = structured.toarray()
        self._fit_design = np.column_stack((structured_dense, text_dense, missing.astype(np.float64)))
        if not np.isfinite(self._fit_design).all():
            raise RuntimeError("N04 R22 fit design is nonfinite")
        self._audit = {
            "feature_names": list(R22_FEATURE_NAMES), "fit_rows": int(len(selected)),
            "fit_cycle_min": int(cycles.min()), "fit_cycle_max": int(cycles.max()),
            "transformed_width": int(self._fit_design.shape[1]),
            "fit_design_sha256": _array_digest(self._fit_design),
            "structured_preprocessing": structured_state,
            "tfidf": RegressionTextModel._text_state(self._vectorizer, self._text_fallback),
            "svd": {**dict(_SVD_PARAMS), "n_components_requested": 32,
                    "n_components_actual": int(self._svd_components_actual),
                    "components_sha256": _array_digest(self._svd.components_) if self._svd is not None else None},
            "transform_fit_population_is_train_prefix": True,
        }
        self._fitted = True
        return self._fit_design

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("N04 R22 design must be fitted before transform")
        selected = self._select(frame)
        structured, _ = RegressionTextModel._structured_transform(
            selected, self._processor, self._scaler, self._encoder, self._numeric_positions,
        )
        docs, missing = _text_documents(selected["signal_stream"])
        if self._vectorizer is None:
            text_matrix = sparse.csr_matrix((len(docs), 0), dtype=np.float64)
        else:
            text_matrix = self._vectorizer.transform(docs).tocsr()
        text_dense = self._svd.transform(text_matrix) if self._svd is not None else np.empty((len(docs), 0))
        design = np.column_stack((structured.toarray(), text_dense, missing.astype(np.float64)))
        if design.shape[1] != self._fit_design.shape[1] or not np.isfinite(design).all():
            raise RuntimeError("N04 R22 score design is nonfinite or changed schema")
        return design

    @property
    def fit_design(self) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("N04 R22 design is not fitted")
        return self._fit_design

    def audit_state(self) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("N04 R22 design is not fitted")
        return dict(self._audit)


def fit_weighted_r22_head(
    design: Any, target: Any, weights: Any, *, cutoff: int,
) -> tuple[XGBRegressor, dict[str, Any]]:
    x = np.asarray(design, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if (x.ndim != 2 or y.shape != (len(x),) or w.shape != y.shape or not np.isfinite(x).all()
            or not np.isfinite(y).all() or not np.isfinite(w).all() or np.any(w <= 0)):
        raise ValueError("N04 weighted R22 fit arrays are invalid")
    if not np.isclose(w.mean(), 1.0, rtol=0, atol=1e-12):
        raise ValueError("N04 sample weights must be normalized to mean one")
    estimator = XGBRegressor(**dict(XGB_PARAMETERS))
    estimator.fit(x, y, sample_weight=w, verbose=False)
    model_text = estimator.get_booster().save_raw(raw_format="json")
    state = {
        "estimator": "xgboost.XGBRegressor", "xgboost_version": xgboost.__version__,
        "parameters": dict(XGB_PARAMETERS), "fit_rows": int(len(y)), "cutoff": int(cutoff),
        "fit_design_sha256": _array_digest(x), "target_sha256": _array_digest(y),
        "sample_weight_sha256": _array_digest(w), "sample_weight_mean": float(w.mean()),
        "sample_weight_min": float(w.min()), "sample_weight_max": float(w.max()),
        "feature_importance_sha256": _array_digest(estimator.feature_importances_),
        "model_sha256": hashlib.sha256(model_text).hexdigest(),
        "sample_weight_passed_to_fit": True, "test_data_used": False,
    }
    state["state_sha256"] = _json_digest(state)
    return estimator, state


def predict_r22_head(estimator: XGBRegressor, design: Any) -> np.ndarray:
    x = np.asarray(design, dtype=np.float64)
    raw = np.asarray(estimator.predict(x), dtype=np.float64)
    if raw.shape != (len(x),) or not np.isfinite(raw).all():
        raise RuntimeError("N04 weighted R22 head returned invalid raw predictions")
    return raw


__all__ = [
    "CARD_ID", "DOMAIN_AUC_RANGE", "DOMAIN_LR_PARAMETERS", "DOMAIN_PROBABILITY_CLIP",
    "DOMAIN_STRUCTURED_FEATURES", "MAXIMUM_SOURCE_CLIP_FRACTION", "MINIMUM_ESS_FRACTION",
    "R22WeightedDesign", "SOURCE_WEIGHT_CLIP", "XGB_PARAMETERS", "compose_r22_candidate",
    "density_ratio_weights", "domain_labels", "fit_weighted_r22_head", "predict_r22_head",
    "shuffle_weights_within_cycle",
]
