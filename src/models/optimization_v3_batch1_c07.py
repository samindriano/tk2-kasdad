"""C07 fit-only category-frequency extension of the frozen sparse classifier."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from lightgbm import LGBMClassifier

from src.models.classification_models import _TFIDF_PARAMS, _text_values
from src.models.classification_round2_models import stream_missing_mask
from src.models.optimization_v3_sparse_classification import PARAMS as SPARSE_PARAMS
from src.models.regression_research_text import RegressionTextModel, HYBRID_FEATURES, _array_digest
from src.models.round1_models import CATEGORICAL_FEATURES, _fit_fingerprint, _is_missing


FEATURE_NAMES = tuple(HYBRID_FEATURES)
FREQUENCY_COLUMNS = tuple(CATEGORICAL_FEATURES)
RARE_COUNT_THRESHOLD = 30
FREQUENCY_FEATURE_NAMES = tuple(
    name
    for column in FREQUENCY_COLUMNS
    for name in (f"{column}__fit_log1p_count", f"{column}__fit_count_lt_30")
)
THRESHOLD = 0.50
BLEND_WEIGHT = 0.25


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                         separators=(",", ":")).encode("utf-8", errors="surrogatepass")
    return hashlib.sha256(payload).hexdigest()


def _normalized_category(value: Any) -> str | None:
    """Match structured category normalization and reserve None for missing."""
    if _is_missing(value):
        return None
    return str(value).strip()


class FitOnlyCategoryFrequency:
    """Six numerical category-support features learned only from supplied fit rows."""

    def __init__(self) -> None:
        self._fitted = False
        self.counts: dict[str, Counter[str | None]] = {}
        self.fit_rows = 0
        self.audit: dict[str, Any] = {}

    @staticmethod
    def _select(X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("C07 frequency inputs must be a pandas DataFrame")
        if X.columns.has_duplicates:
            raise ValueError("C07 frequency inputs contain duplicate columns")
        absent = sorted(set(FREQUENCY_COLUMNS) - set(X.columns))
        if absent:
            raise ValueError(f"C07 category columns are missing: {absent}")
        return X.loc[:, list(FREQUENCY_COLUMNS)].copy()

    def _matrix(self, X: pd.DataFrame) -> np.ndarray:
        selected = self._select(X)
        columns: list[np.ndarray] = []
        for field in FREQUENCY_COLUMNS:
            counts = self.counts[field]
            normalized = [_normalized_category(value) for value in selected[field].to_numpy(dtype=object)]
            values = np.fromiter((int(counts.get(key, 0)) for key in normalized),
                                 dtype=np.int64, count=len(normalized))
            columns.append(np.log1p(values.astype(np.float64)))
            columns.append((values < RARE_COUNT_THRESHOLD).astype(np.float64))
        output = np.column_stack(columns).astype(np.float64, copy=False)
        if output.shape != (len(selected), len(FREQUENCY_FEATURE_NAMES)) or not np.isfinite(output).all():
            raise RuntimeError("C07 frequency feature matrix is invalid")
        return output

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        selected = self._select(X)
        if len(selected) == 0:
            raise ValueError("C07 requires nonempty fit rows")
        self._fitted = False
        self.fit_rows = int(len(selected))
        self.counts = {}
        field_audit: dict[str, Any] = {}
        for field in FREQUENCY_COLUMNS:
            normalized = [_normalized_category(value) for value in selected[field].to_numpy(dtype=object)]
            counts: Counter[str | None] = Counter(normalized)
            self.counts[field] = counts
            serialized = [
                {"category_kind": "missing" if key is None else "value",
                 "value": None if key is None else key, "fit_count": int(count)}
                for key, count in sorted(counts.items(), key=lambda item: (item[0] is not None, item[0] or ""))
            ]
            field_audit[field] = {
                "fit_rows": self.fit_rows,
                "missing_fit_rows": int(counts.get(None, 0)),
                "distinct_nonmissing_fit_categories": int(sum(key is not None for key in counts)),
                "distinct_fit_categories_including_missing": int(len(counts)),
                "categories_below_30": int(sum(value < RARE_COUNT_THRESHOLD for value in counts.values())),
                "category_count_map_sha256": _json_hash(serialized),
            }
        matrix = self._matrix(selected)
        nonconstant = np.ptp(matrix, axis=0) > 0
        if not bool(np.all(nonconstant)):
            constant = [FREQUENCY_FEATURE_NAMES[index] for index in np.flatnonzero(~nonconstant)]
            raise ValueError(f"C07 fit category-frequency columns must be nonconstant: {constant}")
        for index, name in enumerate(FREQUENCY_FEATURE_NAMES):
            field_audit[name] = {
                "training_nonconstant": bool(nonconstant[index]),
                "unique_fit_values": int(np.unique(matrix[:, index]).size),
                "fit_min": float(np.min(matrix[:, index])),
                "fit_max": float(np.max(matrix[:, index])),
            }
        self.audit = {
            "feature_names": list(FREQUENCY_FEATURE_NAMES),
            "fit_rows": self.fit_rows,
            "count_definition": "exact normalized category frequency within this outer fit prefix; missing is a dedicated key",
            "normalization": "the frozen structured transform uses pd.isna for missing; all other values are stringified and stripped",
            "missing_key": "explicit dedicated None category; its count is learned only from fit rows",
            "blank_string": "preserved as the stripped empty-string category, matching the frozen structured transform",
            "unseen_key": "count=0 when absent from fit category map",
            "rare_flag_definition": f"fit_count < {RARE_COUNT_THRESHOLD}",
            "target_rates_used": False,
            "validation_or_test_category_counts_used": False,
            "fields": field_audit,
        }
        self._fitted = True
        return matrix

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("C07 category-frequency map must be fitted first")
        return self._matrix(X)


class CategoryFrequencySparseClassifier:
    """Frozen sparse classifier with six appended fit-prefix category features."""

    def __init__(self) -> None:
        self.input_feature_names = FEATURE_NAMES
        self._fitted = False

    def _select(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame) or X.columns.has_duplicates:
            raise ValueError("C07 requires a DataFrame with unique feature columns")
        if set(X.columns) != set(FEATURE_NAMES) or len(X.columns) != len(FEATURE_NAMES):
            absent = sorted(set(FEATURE_NAMES) - set(X.columns))
            extra = sorted(set(X.columns) - set(FEATURE_NAMES))
            raise ValueError(f"C07 requires the exact unchanged sparse feature allowlist; absent={absent}, extra={extra}")
        return X.loc[:, list(FEATURE_NAMES)].copy()

    def fit(self, X: pd.DataFrame, y: Any) -> "CategoryFrequencySparseClassifier":
        self._fitted = False
        selected = self._select(X)
        target = np.asarray(y, dtype=np.int8)
        if target.shape != (len(selected),) or not np.isin(target, [0, 1]).all() or len(np.unique(target)) != 2:
            raise ValueError("C07 requires aligned binary fit labels containing both classes")

        (structured, self.processor, self.scaler, self.encoder,
         self.numeric_positions, structured_state) = RegressionTextModel._structured_fit(selected)
        self.category_frequency = FitOnlyCategoryFrequency()
        frequency_matrix = self.category_frequency.fit_transform(selected)
        documents, missing = _text_values(selected["signal_stream"])
        self.vectorizer = TfidfVectorizer(**dict(_TFIDF_PARAMS))
        text = self.vectorizer.fit_transform(documents).tocsr()
        missing_column = sparse.csr_matrix(missing.astype(np.float64).reshape(-1, 1))
        original_design = sparse.hstack((structured, text, missing_column), format="csr", dtype=np.float64)
        self.design = sparse.hstack(
            (original_design, sparse.csr_matrix(frequency_matrix)),
            format="csr", dtype=np.float64,
        )
        if self.design.shape[0] != len(target) or not np.isfinite(self.design.data).all():
            raise RuntimeError("C07 sparse fit design is misaligned or nonfinite")

        params = dict(SPARSE_PARAMS)
        self.estimator = LGBMClassifier(**params).fit(self.design, target)
        ordered_vocab = [(str(token), int(index)) for token, index in sorted(
            self.vectorizer.vocabulary_.items(), key=lambda item: item[1])]
        self.state = {
            "id": "V3_BATCH1_C07_CATEGORY_FREQUENCY_SPARSE_LGB",
            "fit_rows": int(len(target)),
            "input_feature_names": list(FEATURE_NAMES),
            "design_feature_names_appended": list(FREQUENCY_FEATURE_NAMES),
            "fit_data_sha256": _fit_fingerprint(selected, FEATURE_NAMES, target),
            "target_sha256": _array_digest(target, dtype="<i8"),
            "sparse_classifier_parameters": params,
            "tfidf_parameters": {
                key: list(value) if isinstance(value, tuple) else value.__name__ if isinstance(value, type) else value
                for key, value in _TFIDF_PARAMS.items()
            },
            "vocabulary_size": int(len(ordered_vocab)),
            "vocabulary_sha256": _json_hash(ordered_vocab),
            "idf_sha256": _array_digest(self.vectorizer.idf_),
            "structured_state": structured_state,
            "category_frequency_state": self.category_frequency.audit,
            "original_sparse_design_shape": [int(value) for value in original_design.shape],
            "training_design_shape": [int(value) for value in self.design.shape],
            "training_design_nnz": int(self.design.nnz),
            "training_design_data_sha256": _array_digest(self.design.data),
            "model_sha256": hashlib.sha256(self.estimator.booster_.model_to_string().encode("utf-8")).hexdigest(),
            "target_rates_used": False,
            "validation_data_passed_to_fit": False,
            "early_stopping_used": False,
            "test_data_used": False,
        }
        self.state["state_sha256"] = _json_hash(self.state)
        self._fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("C07 classifier must be fitted before prediction")
        selected = self._select(X)
        structured, _ = RegressionTextModel._structured_transform(
            selected, self.processor, self.scaler, self.encoder, self.numeric_positions
        )
        documents, missing = _text_values(selected["signal_stream"])
        text = self.vectorizer.transform(documents).tocsr()
        missing_column = sparse.csr_matrix(missing.astype(np.float64).reshape(-1, 1))
        original_design = sparse.hstack((structured, text, missing_column), format="csr", dtype=np.float64)
        frequency = self.category_frequency.transform(selected)
        design = sparse.hstack((original_design, sparse.csr_matrix(frequency)), format="csr", dtype=np.float64)
        if design.shape[1] != self.state["training_design_shape"][1] or not np.isfinite(design.data).all():
            raise RuntimeError("C07 score design differs from the fitted schema or contains nonfinite values")
        probability = np.asarray(self.estimator.predict_proba(design)[:, 1], dtype=np.float64)
        if probability.shape != (len(selected),) or not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
            raise RuntimeError("C07 classifier returned invalid probabilities")
        return probability

    def audit_state(self) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("C07 classifier has no fitted audit state")
        return json.loads(json.dumps(self.state, sort_keys=True, allow_nan=False))


def fit_predict(fit_features: pd.DataFrame, fit_target: np.ndarray, score_features: pd.DataFrame,
                *, controls: dict[str, np.ndarray], card_id: str, cutoff: int) -> dict[str, Any]:
    if card_id != "C07":
        raise ValueError(f"C07 adapter cannot execute {card_id}")
    model = CategoryFrequencySparseClassifier().fit(fit_features, fit_target)
    raw = model.predict_proba(score_features)
    missing = stream_missing_mask(score_features)
    c2b = np.asarray(controls["c2b_probability"], dtype=np.float64)
    cb = np.asarray(controls["cb_probability"], dtype=np.float64)
    old_sparse = np.asarray(controls["baseline_sparse_raw"], dtype=np.float64)
    control_missing = np.asarray(controls["stream_missing"], dtype=bool)
    baseline = np.asarray(controls["baseline_prediction"], dtype=np.float64)
    if not (raw.shape == missing.shape == c2b.shape == cb.shape == old_sparse.shape
            == control_missing.shape == baseline.shape):
        raise ValueError("C07 routing controls are not row aligned")
    if not np.array_equal(missing, control_missing):
        raise ValueError("C07 stream missing mask differs from the frozen control route")
    candidate = np.where(missing, cb, 0.75 * c2b + BLEND_WEIGHT * raw)
    reconstructed_baseline = np.where(missing, cb, 0.75 * c2b + BLEND_WEIGHT * old_sparse)
    if not np.array_equal(reconstructed_baseline, baseline):
        raise ValueError("C07 baseline is not the frozen V3 sparse25 control")
    if not np.isfinite(candidate).all() or np.any((candidate < 0) | (candidate > 1)):
        raise ValueError("C07 routed candidate probabilities are invalid")
    return {
        "prediction": candidate,
        "raw_candidate": raw,
        "fit_predictor_count": 1,
        "state": model.audit_state(),
        "fit_metadata": {
            "cutoff": int(cutoff),
            "fit_rows": int(len(fit_features)),
            "fit_predictor_count": 1,
            "candidate_present_rows": int((~missing).sum()),
            "frozen_catboost_missing_rows": int(missing.sum()),
            "threshold": THRESHOLD,
            "blend_weight": BLEND_WEIGHT,
        },
        "audit_arrays": {"stream_missing": missing},
    }


__all__ = [
    "FEATURE_NAMES", "FREQUENCY_COLUMNS", "FREQUENCY_FEATURE_NAMES",
    "RARE_COUNT_THRESHOLD", "THRESHOLD", "BLEND_WEIGHT",
    "FitOnlyCategoryFrequency", "CategoryFrequencySparseClassifier", "fit_predict",
]
