"""C08: cycle-ordered, smoothed category target-rate features for sparse LightGBM."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src.models.classification_models import _TFIDF_PARAMS, _text_values
from src.models.optimization_v3_sparse_classification import PARAMS as SPARSE_PARAMS
from src.models.regression_research_text import (
    HYBRID_FEATURES,
    RegressionTextModel,
    _array_digest,
    _json_digest,
)
from src.models.round1_models import CATEGORICAL_FEATURES, _fit_fingerprint, _is_missing


CARD_ID = "C08"
FEATURE_NAMES = tuple(HYBRID_FEATURES)
TARGET_RATE_FIELDS = tuple(CATEGORICAL_FEATURES)
TARGET_RATE_FEATURE_NAMES = tuple(f"{field}__ordered_target_logit" for field in TARGET_RATE_FIELDS)
PILOT_CUTOFFS = (55, 67)
MINIMUM_PRIOR_SUPPORT = 30
SMOOTHING = 30.0
INITIAL_PRIOR = 0.5
LOGIT_EPSILON = 1e-6
BLEND_WEIGHT = 0.25
THRESHOLD = 0.50
FALLBACK_CODES = {"supported": 0, "below_minimum_support": 1, "missing_category": 2, "no_prior_category_history": 3}
_CONTROL_KEYS = (
    "baseline_prediction", "baseline_sparse_raw", "c2b_probability",
    "cb_probability", "stream_missing", "control_rows",
)


def _normalized_category(value: Any) -> str | None:
    """Match C07 structured category handling; blank strings remain a category."""
    if _is_missing(value):
        return None
    return str(value).strip()


def _logit(rate: float) -> float:
    clipped = float(np.clip(rate, LOGIT_EPSILON, 1.0 - LOGIT_EPSILON))
    return float(np.log(clipped / (1.0 - clipped)))


def _select_exact(frame: pd.DataFrame) -> pd.DataFrame:
    if (not isinstance(frame, pd.DataFrame) or frame.columns.has_duplicates
            or set(frame.columns) != set(FEATURE_NAMES)
            or len(frame.columns) != len(FEATURE_NAMES)):
        raise ValueError("C08 requires exactly the expanded structured + signal_stream feature allowlist")
    if set(FEATURE_NAMES) & {"record_id", "artifact_signal", "threat_rank"}:
        raise ValueError("C08 feature allowlist contains an identity or target")
    return frame.loc[:, list(FEATURE_NAMES)].copy()


def _cycles(frame: pd.DataFrame, cutoff: int, label: str) -> np.ndarray:
    values = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.float64)
    if (values.ndim != 1 or not np.isfinite(values).all()
            or not np.equal(values, np.floor(values)).all()):
        raise ValueError(f"{label} cycle_code must contain finite integers")
    result = values.astype(np.int64)
    if len(result) == 0 or np.any((result < 47) | (result > cutoff)):
        raise ValueError(f"{label} cycle_code is outside 47..{cutoff}")
    return result


class OrderedCategoryTargetRates:
    """Strictly earlier-cycle category rates, encoded prequentially per cycle."""

    def __init__(self) -> None:
        self._fitted = False
        self._totals: dict[str, dict[str, float]] = {field: {} for field in TARGET_RATE_FIELDS}
        self._counts: dict[str, dict[str, int]] = {field: {} for field in TARGET_RATE_FIELDS}
        self._audit: dict[str, Any] = {}

    @staticmethod
    def _keys(frame: pd.DataFrame) -> dict[str, list[str | None]]:
        return {
            field: [_normalized_category(value) for value in frame[field].to_numpy(dtype=object)]
            for field in TARGET_RATE_FIELDS
        }

    @staticmethod
    def _rate(total: float, count: int) -> float:
        if count < MINIMUM_PRIOR_SUPPORT:
            return INITIAL_PRIOR
        value = (total + SMOOTHING * INITIAL_PRIOR) / (count + SMOOTHING)
        return float(np.clip(value, LOGIT_EPSILON, 1.0 - LOGIT_EPSILON))

    def fit_transform(self, frame: pd.DataFrame, target: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        selected = _select_exact(frame)
        y = np.asarray(target, dtype=np.float64)
        if (y.shape != (len(selected),) or not np.isfinite(y).all()
                or not np.isin(y, (0.0, 1.0)).all()
                or not np.array_equal(np.unique(y), np.array([0.0, 1.0]))):
            raise ValueError("C08 needs an aligned binary target containing both classes")
        cycles = _cycles(selected, int(pd.to_numeric(selected.cycle_code).max()), "C08 fit")
        if not selected.index.is_unique or not selected.index.is_monotonic_increasing:
            raise ValueError("C08 fit rows must retain unique increasing official Train positions")
        keys = self._keys(selected)
        values = np.full((len(selected), len(TARGET_RATE_FIELDS)), _logit(INITIAL_PRIOR), dtype=np.float64)
        supports = np.zeros((len(selected), len(TARGET_RATE_FIELDS)), dtype=np.int32)
        fallback = np.full((len(selected), len(TARGET_RATE_FIELDS)), FALLBACK_CODES["below_minimum_support"], dtype=np.uint8)
        cycle_audit: list[dict[str, Any]] = []

        for cycle in sorted(np.unique(cycles).tolist()):
            row_positions = np.flatnonzero(cycles == cycle)
            field_supported_rows: dict[str, int] = {}
            for column_index, field in enumerate(TARGET_RATE_FIELDS):
                field_keys = keys[field]
                supported_rows = 0
                for position in row_positions:
                    key = field_keys[position]
                    if key is None:
                        fallback[position, column_index] = FALLBACK_CODES["missing_category"]
                        continue
                    count = int(self._counts[field].get(key, 0))
                    total = float(self._totals[field].get(key, 0.0))
                    supports[position, column_index] = count
                    if count == 0:
                        fallback[position, column_index] = FALLBACK_CODES["no_prior_category_history"]
                    elif count < MINIMUM_PRIOR_SUPPORT:
                        fallback[position, column_index] = FALLBACK_CODES["below_minimum_support"]
                    else:
                        fallback[position, column_index] = FALLBACK_CODES["supported"]
                        supported_rows += 1
                    values[position, column_index] = _logit(self._rate(total, count))
                field_supported_rows[field] = supported_rows

            # Update category target sums only after every row in this cycle
            # has received features from strictly earlier cycles.
            for field in TARGET_RATE_FIELDS:
                grouped: dict[str, list[int]] = defaultdict(list)
                for position in row_positions:
                    key = keys[field][position]
                    if key is not None:
                        grouped[key].append(int(position))
                for key in sorted(grouped):
                    positions = np.asarray(grouped[key], dtype=np.int64)
                    self._counts[field][key] = self._counts[field].get(key, 0) + int(len(positions))
                    self._totals[field][key] = self._totals[field].get(key, 0.0) + float(y[positions].sum(dtype=np.float64))
            cycle_audit.append({
                "cycle": int(cycle),
                "rows": int(len(row_positions)),
                "labels_seen_before_cycle": int(np.sum(cycles < cycle)),
                "supported_rate_rows_by_field": field_supported_rows,
                "all_cycle_rows_encoded_before_label_updates": True,
            })

        self._fitted = True
        self._audit = {
            "fields": list(TARGET_RATE_FIELDS),
            "minimum_prior_support": MINIMUM_PRIOR_SUPPORT,
            "smoothing": SMOOTHING,
            "initial_and_fallback_prior": INITIAL_PRIOR,
            "prior_policy": "fixed 0.5 prior, including the first cycle and all unsupported/missing/unseen categories",
            "logit_clip_epsilon": LOGIT_EPSILON,
            "category_normalization": "C07 convention: pandas missing is None; otherwise str(value).strip(); blank string remains a category",
            "training_rate_logits_sha256": _array_digest(values),
            "training_support_counts_sha256": _array_digest(supports, dtype="<i4"),
            "training_fallback_codes_sha256": _array_digest(fallback, dtype="|u1"),
            "fit_rows": int(len(y)),
            "category_dictionary_sizes": {field: int(len(self._counts[field])) for field in TARGET_RATE_FIELDS},
            "category_dictionary_sha256": {
                field: _json_digest([
                    {"key": key, "count": int(self._counts[field][key]), "target_sum": float(self._totals[field][key])}
                    for key in sorted(self._counts[field])
                ]) for field in TARGET_RATE_FIELDS
            },
            "lineage": cycle_audit,
            "target_labels_updated_after_same_cycle_features": True,
        }
        if (not np.isfinite(values).all() or values.shape != (len(selected), len(TARGET_RATE_FIELDS))
                or supports.shape != values.shape or fallback.shape != values.shape):
            raise RuntimeError("C08 training target-rate feature arrays are invalid")
        return values, supports, fallback

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        if not self._fitted:
            raise RuntimeError("C08 ordered target-rate encoder must be fit before transform")
        selected = _select_exact(frame)
        keys = self._keys(selected)
        values = np.full((len(selected), len(TARGET_RATE_FIELDS)), _logit(INITIAL_PRIOR), dtype=np.float64)
        supports = np.zeros((len(selected), len(TARGET_RATE_FIELDS)), dtype=np.int32)
        fallback = np.full((len(selected), len(TARGET_RATE_FIELDS)), FALLBACK_CODES["below_minimum_support"], dtype=np.uint8)
        for column_index, field in enumerate(TARGET_RATE_FIELDS):
            for position, key in enumerate(keys[field]):
                if key is None:
                    fallback[position, column_index] = FALLBACK_CODES["missing_category"]
                    continue
                count = int(self._counts[field].get(key, 0))
                total = float(self._totals[field].get(key, 0.0))
                supports[position, column_index] = count
                if count == 0:
                    fallback[position, column_index] = FALLBACK_CODES["no_prior_category_history"]
                elif count < MINIMUM_PRIOR_SUPPORT:
                    fallback[position, column_index] = FALLBACK_CODES["below_minimum_support"]
                else:
                    fallback[position, column_index] = FALLBACK_CODES["supported"]
                values[position, column_index] = _logit(self._rate(total, count))
        diagnostic = {
            "rows": int(len(selected)),
            "supported_rows_by_field": {
                field: int(np.sum(fallback[:, index] == FALLBACK_CODES["supported"]))
                for index, field in enumerate(TARGET_RATE_FIELDS)
            },
            "below_support_rows_by_field": {
                field: int(np.sum(fallback[:, index] == FALLBACK_CODES["below_minimum_support"]))
                for index, field in enumerate(TARGET_RATE_FIELDS)
            },
            "missing_category_rows_by_field": {
                field: int(np.sum(fallback[:, index] == FALLBACK_CODES["missing_category"]))
                for index, field in enumerate(TARGET_RATE_FIELDS)
            },
            "no_prior_category_history_rows_by_field": {
                field: int(np.sum(fallback[:, index] == FALLBACK_CODES["no_prior_category_history"]))
                for index, field in enumerate(TARGET_RATE_FIELDS)
            },
            "rate_logits_sha256": _array_digest(values),
            "support_counts_sha256": _array_digest(supports, dtype="<i4"),
            "fallback_codes_sha256": _array_digest(fallback, dtype="|u1"),
        }
        if not np.isfinite(values).all():
            raise RuntimeError("C08 transform produced nonfinite logit features")
        return values, supports, fallback, diagnostic

    def audit_state(self) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("C08 ordered target-rate encoder is not fitted")
        return json.loads(json.dumps(self._audit, sort_keys=True, allow_nan=False))


def _validated_controls(controls: dict[str, Any], n_score: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    absent = [name for name in _CONTROL_KEYS if name not in controls]
    if absent or n_score <= 0:
        raise ValueError(f"C08 controls are missing fields or empty: {absent}")
    arrays: dict[str, np.ndarray] = {}
    for name in _CONTROL_KEYS[:-1]:
        dtype = bool if name == "stream_missing" else np.float64
        values = np.asarray(controls[name], dtype=dtype)
        if values.shape != (n_score,):
            raise ValueError(f"C08 control {name} is not aligned")
        if name != "stream_missing" and not np.isfinite(values).all():
            raise ValueError(f"C08 control {name} is nonfinite")
        if name in {"baseline_prediction", "baseline_sparse_raw", "c2b_probability", "cb_probability"}:
            if np.any((values < 0.0) | (values > 1.0)):
                raise ValueError(f"C08 control {name} is outside [0,1]")
        arrays[name] = values
    rows = np.asarray(controls["control_rows"])
    if (rows.shape != (n_score,) or rows.dtype.kind not in "iu"
            or len(np.unique(rows)) != n_score or np.any(np.diff(rows.astype(np.int64)) <= 0)):
        raise ValueError("C08 control rows must be increasing unique source positions")
    expected = np.where(
        arrays["stream_missing"], arrays["cb_probability"],
        0.75 * arrays["c2b_probability"] + 0.25 * arrays["baseline_sparse_raw"],
    )
    if not np.array_equal(arrays["baseline_prediction"], expected):
        raise ValueError("C08 baseline is not the fixed row-matched sparse25 route")
    return arrays, rows.astype(np.int64, copy=False)


class OrderedRateSparseClassifier:
    """Original sparse LightGBM plus exactly three prequential category logits."""

    input_feature_names = FEATURE_NAMES

    def __init__(self, cutoff: int):
        if int(cutoff) not in PILOT_CUTOFFS:
            raise ValueError(f"C08 cutoff must be one of {PILOT_CUTOFFS}")
        self.cutoff = int(cutoff)
        self._fitted = False

    def fit(self, frame: pd.DataFrame, target: Any) -> "OrderedRateSparseClassifier":
        self._fitted = False
        selected = _select_exact(frame)
        y_numeric = np.asarray(target, dtype=np.float64)
        if (y_numeric.shape != (len(selected),) or not np.isfinite(y_numeric).all()
                or not np.isin(y_numeric, (0.0, 1.0)).all()
                or not np.array_equal(np.unique(y_numeric), np.array([0.0, 1.0]))):
            raise ValueError("C08 requires an aligned binary fit target containing both classes")
        y = y_numeric.astype(np.int8)
        cycles = _cycles(selected, self.cutoff, "C08 fit")
        if not selected.index.is_unique or not selected.index.is_monotonic_increasing:
            raise ValueError("C08 fit rows must retain unique increasing official Train positions")

        self._rate_encoder = OrderedCategoryTargetRates()
        rate_values, rate_supports, rate_fallbacks = self._rate_encoder.fit_transform(selected, y)
        self._rate_training_diagnostics = self._rate_encoder.audit_state()
        (structured, self._processor, self._scaler, self._encoder, self._numeric_positions,
         structured_state) = RegressionTextModel._structured_fit(selected)
        documents, missing = _text_values(selected["signal_stream"])
        self._vectorizer = TfidfVectorizer(**dict(_TFIDF_PARAMS))
        text = self._vectorizer.fit_transform(documents).tocsr()
        rate_sparse = sparse.csr_matrix(rate_values, dtype=np.float64)
        missing_column = sparse.csr_matrix(missing.astype(np.float64).reshape(-1, 1))
        design = sparse.hstack((structured, rate_sparse, text, missing_column), format="csr", dtype=np.float64)
        parameters = dict(SPARSE_PARAMS)
        parameters.update({"n_jobs": 2, "deterministic": True, "force_col_wise": True})
        self._estimator = LGBMClassifier(**parameters)
        self._estimator.fit(design, y)
        ordered_vocabulary = [
            (str(token), int(index)) for token, index in sorted(
                self._vectorizer.vocabulary_.items(), key=lambda item: item[1])
        ]
        self._state = {
            "id": "V3_BATCH1_C08_ORDERED_CATEGORY_TARGET_RATES_SPARSE_LGBM",
            "cutoff": self.cutoff,
            "fit_rows": int(len(y)),
            "fit_cycle_min": int(cycles.min()),
            "fit_cycle_max": int(cycles.max()),
            "fit_cycle_counts": {str(int(c)): int(n) for c, n in zip(*np.unique(cycles, return_counts=True))},
            "input_feature_names": list(FEATURE_NAMES),
            "target_rate_feature_names": list(TARGET_RATE_FEATURE_NAMES),
            "fit_data_sha256": _fit_fingerprint(selected, FEATURE_NAMES, y),
            "target_sha256": _array_digest(y, dtype="<i8"),
            "structured_preprocessing": structured_state,
            "ordered_target_rate_encoder": self._rate_training_diagnostics,
            "training_rate_support_summary": {
                field: {
                    "rows_supported_minimum_30": int(np.sum(rate_fallbacks[:, index] == FALLBACK_CODES["supported"])),
                    "rows_fallback_prior": int(np.sum(rate_fallbacks[:, index] != FALLBACK_CODES["supported"])),
                    "support_count_max": int(rate_supports[:, index].max(initial=0)),
                } for index, field in enumerate(TARGET_RATE_FIELDS)
            },
            "tfidf_parameters": {
                key: list(value) if isinstance(value, tuple)
                else value.__name__ if isinstance(value, type)
                else value.item() if isinstance(value, np.generic) else value
                for key, value in _TFIDF_PARAMS.items()
            },
            "tfidf_vocabulary_size": int(len(ordered_vocabulary)),
            "tfidf_vocabulary_sha256": _json_digest(ordered_vocabulary),
            "tfidf_idf_sha256": _array_digest(self._vectorizer.idf_),
            "training_design_shape": [int(value) for value in design.shape],
            "training_design_nnz": int(design.nnz),
            "training_design_data_sha256": _array_digest(design.data),
            "lightgbm_parameters": parameters,
            "lightgbm_version": __import__("lightgbm").__version__,
            "model_sha256": hashlib.sha256(self._estimator.booster_.model_to_string().encode("utf-8")).hexdigest(),
            "fit_missing_stream_rows": int(missing.sum()),
            "validation_data_passed_to_fit": False,
            "validation_labels_passed_to_fit": False,
            "early_stopping_used": False,
            "sample_weight_used": False,
            "test_data_used": False,
        }
        self._state["state_sha256"] = _json_digest(self._state)
        self._fitted = True
        return self

    def predict_probability(self, frame: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("C08 sparse classifier must be fit before prediction")
        selected = _select_exact(frame)
        structured, unknown = RegressionTextModel._structured_transform(
            selected, self._processor, self._scaler, self._encoder, self._numeric_positions
        )
        rate_values, rate_supports, rate_fallbacks, rate_diagnostics = self._rate_encoder.transform(selected)
        documents, missing = _text_values(selected["signal_stream"])
        text = self._vectorizer.transform(documents).tocsr()
        design = sparse.hstack((
            structured, sparse.csr_matrix(rate_values, dtype=np.float64), text,
            sparse.csr_matrix(missing.astype(np.float64).reshape(-1, 1)),
        ), format="csr", dtype=np.float64)
        probability = np.asarray(self._estimator.booster_.predict(design, num_threads=2), dtype=np.float64)
        if (probability.shape != (len(selected),) or not np.isfinite(probability).all()
                or np.any((probability < 0.0) | (probability > 1.0))):
            raise RuntimeError("C08 returned invalid positive-class probabilities")
        self._prediction_diagnostics = {
            "n_predictions": int(len(probability)),
            "missing_stream_rows": int(missing.sum()),
            "unknown_structured_categories": unknown,
            "ordered_target_rate_transform": rate_diagnostics,
        }
        return probability, rate_values, rate_supports, rate_fallbacks

    def audit_state(self) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("C08 sparse classifier must be fit before audit")
        return json.loads(json.dumps(self._state, sort_keys=True, allow_nan=False))


def fit_predict(fit_features: pd.DataFrame, fit_target: np.ndarray,
                score_features: pd.DataFrame, *, controls: dict[str, Any],
                card_id: str, cutoff: int) -> dict[str, Any]:
    if card_id != CARD_ID or int(cutoff) not in PILOT_CUTOFFS:
        raise ValueError("C08 adapter only accepts its declared card and cutoffs 55/67")
    fit_x = _select_exact(fit_features)
    score_x = _select_exact(score_features)
    if (not fit_x.index.is_unique or not fit_x.index.is_monotonic_increasing
            or not score_x.index.is_unique or not score_x.index.is_monotonic_increasing):
        raise ValueError("C08 fit and score rows must preserve increasing source positions")
    fit_cycles = _cycles(fit_x, int(cutoff), "C08 fit")
    score_cycle_values = pd.to_numeric(score_x["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    if np.any(score_cycle_values <= int(cutoff)) or np.any(score_cycle_values > int(cutoff) + 12):
        raise ValueError("C08 score rows violate the frozen temporal horizon")
    arrays, control_rows = _validated_controls(controls, len(score_x))
    if not np.array_equal(score_x.index.to_numpy(dtype=np.int64), control_rows):
        raise ValueError("C08 controls do not match score row positions")
    model = OrderedRateSparseClassifier(cutoff).fit(fit_x, fit_target)
    raw, rate_values, rate_supports, rate_fallbacks = model.predict_probability(score_x)
    candidate = np.where(
        arrays["stream_missing"], arrays["cb_probability"],
        0.75 * arrays["c2b_probability"] + BLEND_WEIGHT * raw,
    )
    if (candidate.shape != (len(score_x),) or not np.isfinite(candidate).all()
            or np.any((candidate < 0.0) | (candidate > 1.0))):
        raise RuntimeError("C08 fixed availability route returned invalid probabilities")
    return {
        "prediction": candidate,
        "raw_candidate": raw,
        "fit_predictor_count": 1,
        "state": model.audit_state(),
        "fit_metadata": {
            **model._prediction_diagnostics,
            "cutoff": int(cutoff),
            "fit_rows": int(len(fit_x)),
            "fit_cycle_min": int(fit_cycles.min()),
            "fit_cycle_max": int(fit_cycles.max()),
            "candidate_present_stream_rows": int((~arrays["stream_missing"]).sum()),
            "frozen_cb_missing_stream_rows": int(arrays["stream_missing"].sum()),
            "blend_weight_new_present_expert": BLEND_WEIGHT,
            "threshold": THRESHOLD,
            "policy": ".75 frozen C2_B + .25 C08 sparse expert on present; frozen CB on missing",
            "score_labels_passed_to_adapter": False,
        },
        "audit_arrays": {
            "stream_missing": arrays["stream_missing"],
            "ordered_rate_logits": rate_values,
            "ordered_rate_support": rate_supports,
            "ordered_rate_fallback": rate_fallbacks,
        },
    }


__all__ = [
    "BLEND_WEIGHT", "CARD_ID", "FALLBACK_CODES", "FEATURE_NAMES", "MINIMUM_PRIOR_SUPPORT",
    "PILOT_CUTOFFS", "SMOOTHING", "TARGET_RATE_FIELDS", "TARGET_RATE_FEATURE_NAMES",
    "THRESHOLD", "INITIAL_PRIOR", "LOGIT_EPSILON", "OrderedCategoryTargetRates",
    "OrderedRateSparseClassifier", "fit_predict",
]
