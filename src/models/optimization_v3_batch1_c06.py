"""C06 chronological category-novelty extension of the frozen sparse classifier."""
from __future__ import annotations

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


CARD_ID = "C06"
FEATURE_NAMES = tuple(HYBRID_FEATURES)
CATEGORY_COLUMNS = tuple(CATEGORICAL_FEATURES)
NOVELTY_FEATURE_NAMES = tuple(
    name for column in CATEGORY_COLUMNS
    for name in (f"{column}__novel_from_prior_cycles", f"{column}__missing")
)
MINIMUM_CLASS_SUPPORT = 30
THRESHOLD = 0.50
BLEND_WEIGHT = 0.25


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                         separators=(",", ":")).encode("utf-8", errors="surrogatepass")
    return hashlib.sha256(payload).hexdigest()


def _normalized_category(value: Any) -> str | None:
    """Match the frozen sparse structured path: NA is missing; blank stays a category."""
    if _is_missing(value):
        return None
    return str(value).strip()


def _select_categories(X: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame) or X.columns.has_duplicates:
        raise ValueError("C06 novelty inputs must be a DataFrame with unique columns")
    absent = sorted(set(CATEGORY_COLUMNS) - set(X.columns))
    if absent:
        raise ValueError(f"C06 category columns are missing: {absent}")
    return X.loc[:, list(CATEGORY_COLUMNS)].copy()


def validate_flag_support(matrix: np.ndarray, minimum: int = MINIMUM_CLASS_SUPPORT) -> dict[str, Any]:
    """Fail closed unless every binary novelty/missing feature has both classes."""
    values = np.asarray(matrix)
    if values.ndim != 2 or values.shape[1] != len(NOVELTY_FEATURE_NAMES):
        raise ValueError("C06 expected exactly six binary novelty/missing columns")
    if not np.isin(values, [0, 1]).all():
        raise ValueError("C06 novelty and missing features must be binary")
    support = {
        name: {
            "zeros": int(np.count_nonzero(values[:, index] == 0)),
            "ones": int(np.count_nonzero(values[:, index] == 1)),
            "minimum_per_class": int(minimum),
        }
        for index, name in enumerate(NOVELTY_FEATURE_NAMES)
    }
    failed = [name for name, counts in support.items()
              if min(counts["zeros"], counts["ones"]) < int(minimum)]
    if failed:
        raise ValueError(f"C06 feature support below {minimum} zeros/ones: {failed}")
    return support


class FitChronologicalCategoryNovelty:
    """Encode each fit row against strictly earlier cycles; score against all fit cycles."""

    def __init__(self, minimum_class_support: int = MINIMUM_CLASS_SUPPORT) -> None:
        self.minimum_class_support = int(minimum_class_support)
        self._fitted = False
        self.vocabularies: dict[str, set[str]] = {}
        self.fit_rows = 0
        self.fit_cycle_min: int | None = None
        self.fit_cycle_max: int | None = None
        self.audit: dict[str, Any] = {}

    @staticmethod
    def _cycles(values: Any, n_rows: int) -> np.ndarray:
        cycles = np.asarray(values)
        if cycles.ndim != 1 or len(cycles) != n_rows:
            raise ValueError("C06 cycles must be one-dimensional and row-aligned")
        try:
            numeric = cycles.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("C06 cycle values must be integers") from exc
        if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError("C06 cycle values must be finite integers")
        return numeric.astype(np.int64)

    @staticmethod
    def _feature_matrix(selected: pd.DataFrame, novelty: dict[str, np.ndarray],
                        missing: dict[str, np.ndarray]) -> np.ndarray:
        columns: list[np.ndarray] = []
        for field in CATEGORY_COLUMNS:
            columns.extend((novelty[field], missing[field]))
        matrix = np.column_stack(columns).astype(np.uint8, copy=False)
        if matrix.shape != (len(selected), len(NOVELTY_FEATURE_NAMES)):
            raise RuntimeError("C06 novelty feature schema is invalid")
        return matrix

    def fit_transform(self, X: pd.DataFrame, cycles: Any) -> np.ndarray:
        self._fitted = False
        selected = _select_categories(X)
        if len(selected) == 0:
            raise ValueError("C06 requires nonempty chronological fit rows")
        cycle_values = self._cycles(cycles, len(selected))
        normalized = {
            field: [_normalized_category(value) for value in selected[field].to_numpy(dtype=object)]
            for field in CATEGORY_COLUMNS
        }
        novelty = {field: np.zeros(len(selected), dtype=np.uint8) for field in CATEGORY_COLUMNS}
        missing = {field: np.zeros(len(selected), dtype=np.uint8) for field in CATEGORY_COLUMNS}
        vocabularies: dict[str, set[str]] = {field: set() for field in CATEGORY_COLUMNS}
        cycle_audit: dict[str, Any] = {}

        # Rows in a cycle are encoded against the same prior-cycle vocabulary.
        for cycle in np.sort(np.unique(cycle_values)):
            positions = np.flatnonzero(cycle_values == cycle)
            cycle_record: dict[str, Any] = {"rows": int(len(positions)), "fields": {}}
            for field in CATEGORY_COLUMNS:
                values = normalized[field]
                prior = vocabularies[field]
                additions: set[str] = set()
                novel_count = 0
                missing_count = 0
                for position in positions:
                    value = values[int(position)]
                    if value is None:
                        missing[field][position] = 1
                        missing_count += 1
                    elif value not in prior:
                        novelty[field][position] = 1
                        novel_count += 1
                        additions.add(value)
                # Vocabulary updates only after every row of the cycle is encoded.
                prior.update(additions)
                cycle_record["fields"][field] = {
                    "novel_rows": novel_count,
                    "missing_rows": missing_count,
                    "new_categories_added_after_cycle": len(additions),
                    "vocabulary_size_after_cycle": len(prior),
                }
            cycle_audit[str(int(cycle))] = cycle_record

        matrix = self._feature_matrix(selected, novelty, missing)
        support = validate_flag_support(matrix, self.minimum_class_support)
        self.fit_rows = int(len(selected))
        self.fit_cycle_min = int(np.min(cycle_values))
        self.fit_cycle_max = int(np.max(cycle_values))
        self.vocabularies = vocabularies
        category_audit = {}
        for field in CATEGORY_COLUMNS:
            category_audit[field] = {
                "fit_categories_seen_by_final_cycle": len(vocabularies[field]),
                "fit_vocabulary_sha256": _json_hash(sorted(vocabularies[field])),
                "novel_fit_rows": int(novelty[field].sum()),
                "missing_fit_rows": int(missing[field].sum()),
            }
        self.audit = {
            "feature_names": list(NOVELTY_FEATURE_NAMES),
            "fit_rows": self.fit_rows,
            "fit_cycle_min": self.fit_cycle_min,
            "fit_cycle_max": self.fit_cycle_max,
            "cycle_counts": {str(int(cycle)): int(np.count_nonzero(cycle_values == cycle))
                             for cycle in np.sort(np.unique(cycle_values))},
            "rule": "for each fit row, novel=1 iff normalized nonmissing category is absent from vocabulary of strictly earlier cycles; update vocabulary after encoding the entire cycle",
            "score_rule": "novel=1 iff normalized nonmissing score category is absent from the vocabulary of all fit cycles",
            "missing_rule": "pd.isna category values receive missing=1 and novel=0; blank strings remain the stripped empty-string category, matching the sparse structured model",
            "minimum_class_support": self.minimum_class_support,
            "flag_support": support,
            "cycle_audit": cycle_audit,
            "category_audit": category_audit,
            "target_rates_used": False,
            "validation_or_test_categories_used_for_fit_features": False,
        }
        self._fitted = True
        return matrix

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("C06 chronological vocabulary must be fitted first")
        selected = _select_categories(X)
        novelty: dict[str, np.ndarray] = {}
        missing: dict[str, np.ndarray] = {}
        for field in CATEGORY_COLUMNS:
            normalized = [_normalized_category(value) for value in selected[field].to_numpy(dtype=object)]
            missing[field] = np.fromiter((value is None for value in normalized), dtype=np.uint8, count=len(normalized))
            novelty[field] = np.fromiter(
                (value is not None and value not in self.vocabularies[field] for value in normalized),
                dtype=np.uint8, count=len(normalized),
            )
        return self._feature_matrix(selected, novelty, missing)


class CategoryNoveltySparseClassifier:
    """Unchanged frozen sparse classifier with six chronological category flags."""

    def __init__(self) -> None:
        self.input_feature_names = FEATURE_NAMES
        self._fitted = False

    def _select(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame) or X.columns.has_duplicates:
            raise ValueError("C06 requires a DataFrame with unique feature columns")
        if set(X.columns) != set(FEATURE_NAMES) or len(X.columns) != len(FEATURE_NAMES):
            absent = sorted(set(FEATURE_NAMES) - set(X.columns))
            extra = sorted(set(X.columns) - set(FEATURE_NAMES))
            raise ValueError(f"C06 requires the exact unchanged sparse feature allowlist; absent={absent}, extra={extra}")
        return X.loc[:, list(FEATURE_NAMES)].copy()

    def fit(self, X: pd.DataFrame, y: Any) -> "CategoryNoveltySparseClassifier":
        self._fitted = False
        selected = self._select(X)
        target = np.asarray(y, dtype=np.int8)
        if target.shape != (len(selected),) or not np.isin(target, [0, 1]).all() or len(np.unique(target)) != 2:
            raise ValueError("C06 requires aligned binary fit labels containing both classes")
        if "cycle_code" not in selected:
            raise ValueError("C06 requires cycle_code for chronological encoding")
        raw_fit_cycles = pd.to_numeric(selected["cycle_code"], errors="raise").to_numpy()
        fit_cycles = FitChronologicalCategoryNovelty._cycles(raw_fit_cycles, len(selected))

        (structured, self.processor, self.scaler, self.encoder,
         self.numeric_positions, structured_state) = RegressionTextModel._structured_fit(selected)
        self.novelty = FitChronologicalCategoryNovelty()
        novelty_matrix = self.novelty.fit_transform(selected, fit_cycles)
        documents, missing_stream = _text_values(selected["signal_stream"])
        self.vectorizer = TfidfVectorizer(**dict(_TFIDF_PARAMS))
        text = self.vectorizer.fit_transform(documents).tocsr()
        missing_column = sparse.csr_matrix(missing_stream.astype(np.float64).reshape(-1, 1))
        original_design = sparse.hstack((structured, text, missing_column), format="csr", dtype=np.float64)
        self.design = sparse.hstack(
            (original_design, sparse.csr_matrix(novelty_matrix.astype(np.float64))),
            format="csr", dtype=np.float64,
        )
        if self.design.shape[0] != len(target) or not np.isfinite(self.design.data).all():
            raise RuntimeError("C06 sparse fit design is misaligned or nonfinite")

        params = dict(SPARSE_PARAMS)
        self.estimator = LGBMClassifier(**params).fit(self.design, target)
        ordered_vocab = [(str(token), int(index)) for token, index in sorted(
            self.vectorizer.vocabulary_.items(), key=lambda item: item[1])]
        self.state = {
            "id": "V3_BATCH1_C06_ORDERED_CATEGORY_NOVELTY_SPARSE_LGB",
            "fit_rows": int(len(target)),
            "input_feature_names": list(FEATURE_NAMES),
            "design_feature_names_appended": list(NOVELTY_FEATURE_NAMES),
            "fit_cycle_min": int(np.min(fit_cycles)),
            "fit_cycle_max": int(np.max(fit_cycles)),
            "fit_cycle_counts": {
                str(int(cycle)): int(np.count_nonzero(fit_cycles == cycle))
                for cycle in np.sort(np.unique(fit_cycles))
            },
            "fit_row_positions_sha256": _array_digest(
                selected.index.to_numpy(dtype=np.int64), dtype="<i8"
            ),
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
            "chronological_category_novelty_state": self.novelty.audit,
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
            raise RuntimeError("C06 classifier must be fitted before prediction")
        selected = self._select(X)
        structured, _ = RegressionTextModel._structured_transform(
            selected, self.processor, self.scaler, self.encoder, self.numeric_positions
        )
        documents, missing_stream = _text_values(selected["signal_stream"])
        text = self.vectorizer.transform(documents).tocsr()
        missing_column = sparse.csr_matrix(missing_stream.astype(np.float64).reshape(-1, 1))
        original_design = sparse.hstack((structured, text, missing_column), format="csr", dtype=np.float64)
        novelty = self.novelty.transform(selected)
        design = sparse.hstack(
            (original_design, sparse.csr_matrix(novelty.astype(np.float64))),
            format="csr", dtype=np.float64,
        )
        if design.shape[1] != self.state["training_design_shape"][1] or not np.isfinite(design.data).all():
            raise RuntimeError("C06 score design differs from the fitted schema or contains nonfinite values")
        probability = np.asarray(self.estimator.predict_proba(design)[:, 1], dtype=np.float64)
        if (probability.shape != (len(selected),) or not np.isfinite(probability).all()
                or np.any((probability < 0) | (probability > 1))):
            raise RuntimeError("C06 classifier returned invalid probabilities")
        return probability

    def audit_state(self) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("C06 classifier has no fitted audit state")
        return json.loads(json.dumps(self.state, sort_keys=True, allow_nan=False))


def fit_predict(fit_features: pd.DataFrame, fit_target: np.ndarray, score_features: pd.DataFrame,
                *, controls: dict[str, np.ndarray], card_id: str, cutoff: int) -> dict[str, Any]:
    if card_id != CARD_ID:
        raise ValueError(f"C06 adapter cannot execute {card_id}")
    model = CategoryNoveltySparseClassifier().fit(fit_features, fit_target)
    raw = model.predict_proba(score_features)
    missing = stream_missing_mask(score_features)
    c2b = np.asarray(controls["c2b_probability"], dtype=np.float64)
    cb = np.asarray(controls["cb_probability"], dtype=np.float64)
    old_sparse = np.asarray(controls["baseline_sparse_raw"], dtype=np.float64)
    control_missing = np.asarray(controls["stream_missing"], dtype=bool)
    baseline = np.asarray(controls["baseline_prediction"], dtype=np.float64)
    if not (raw.shape == missing.shape == c2b.shape == cb.shape == old_sparse.shape
            == control_missing.shape == baseline.shape):
        raise ValueError("C06 routing controls are not row-aligned")
    if not np.array_equal(missing, control_missing):
        raise ValueError("C06 stream missing mask differs from the frozen control route")
    candidate = np.where(missing, cb, 0.75 * c2b + BLEND_WEIGHT * raw)
    reconstructed_baseline = np.where(missing, cb, 0.75 * c2b + BLEND_WEIGHT * old_sparse)
    if not np.array_equal(reconstructed_baseline, baseline):
        raise ValueError("C06 baseline is not the frozen V3 sparse25 control")
    if not np.isfinite(candidate).all() or np.any((candidate < 0) | (candidate > 1)):
        raise ValueError("C06 routed candidate probabilities are invalid")
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
    "CARD_ID", "FEATURE_NAMES", "CATEGORY_COLUMNS", "NOVELTY_FEATURE_NAMES",
    "MINIMUM_CLASS_SUPPORT", "THRESHOLD", "BLEND_WEIGHT",
    "FitChronologicalCategoryNovelty", "CategoryNoveltySparseClassifier",
    "validate_flag_support", "fit_predict",
]
