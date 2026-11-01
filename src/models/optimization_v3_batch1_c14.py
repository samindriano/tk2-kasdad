"""C14: fit-only random partitions of numeric metadata plus sparse text LR."""
from __future__ import annotations

import hashlib
import json
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.models.classification_models import _LR_PARAMS, _TFIDF_PARAMS, _text_values
from src.models.regression_research_text import HYBRID_FEATURES, RegressionTextModel, _array_digest
from src.models.round1_models import _fit_fingerprint


CARD_ID = "C14"
FEATURE_NAMES = tuple(HYBRID_FEATURES)
PILOT_CUTOFFS = (55, 67)
NUMERIC_TREE_FEATURES = (
    "impact_load", "impact_load__parsefail", "impact_load__missing",
    "entry_friction", "entry_friction__parsefail", "entry_friction__missing",
    "cycle_code", "cycle_code__parsefail", "cycle_code__missing",
    "scope_pressure", "scope_pressure__parsefail", "scope_pressure__missing",
    "text_load", "text_load__parsefail", "text_load__missing",
    "scope_trace", "scope_trace__approximate", "scope_trace__suffix_present",
    "scope_trace__parsefail", "scope_trace__missing",
    "org_spread", "org_spread__approximate", "org_spread__suffix_present",
    "org_spread__parsefail", "org_spread__missing",
    "module_spread", "module_spread__approximate", "module_spread__suffix_present",
    "module_spread__parsefail", "module_spread__missing",
    "telemetry_flag", "telemetry_flag__unknown", "telemetry_flag__missing",
    "org_code__missing", "module_code__missing", "pattern_code__missing",
)
TREE_PARAMS = {
    "n_estimators": 64,
    "max_depth": 5,
    "min_samples_leaf": 30,
    "random_state": 42,
    "n_jobs": 2,
    "sparse_output": True,
}
LR_PARAMS = dict(_LR_PARAMS)
TFIDF_PARAMS = dict(_TFIDF_PARAMS)
CANDIDATE_WEIGHT = 0.75
FROZEN_SPARSE_WEIGHT = 0.25
THRESHOLD = 0.50
_CONTROL_KEYS = (
    "baseline_prediction", "baseline_sparse_raw", "c2b_probability",
    "cb_probability", "stream_missing", "control_rows",
)


def _json_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                         separators=(",", ":")).encode("utf-8", errors="surrogatepass")
    return hashlib.sha256(payload).hexdigest()


def _sparse_digest(value: sparse.spmatrix) -> str:
    matrix = value.tocsr(copy=True)
    matrix.sum_duplicates()
    matrix.sort_indices()
    digest = hashlib.sha256()
    digest.update(np.asarray(matrix.shape, dtype="<i8").tobytes(order="C"))
    digest.update(np.asarray(matrix.indptr, dtype="<i8").tobytes(order="C"))
    digest.update(np.asarray(matrix.indices, dtype="<i8").tobytes(order="C"))
    digest.update(np.asarray(matrix.data, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def _params_json(value: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, tuple):
            result[key] = list(item)
        elif isinstance(item, type):
            result[key] = item.__name__
        elif isinstance(item, np.generic):
            result[key] = item.item()
        else:
            result[key] = item
    return result


class RandomPartitionTextClassifier:
    """C14 classifier with fit-only numeric partitions and sparse LR inputs.

    Numeric/count fields are scaled inside the fit fold and used only to learn
    the unsupervised random leaf map. The LR sees the resulting leaf indicators,
    fit-only word TF-IDF, fit-only categorical one-hot columns, and the missing
    stream indicator. Ordinal category codes never enter the tree map.
    """

    input_feature_names = FEATURE_NAMES

    def __init__(self, cutoff: int):
        if isinstance(cutoff, bool) or int(cutoff) not in PILOT_CUTOFFS:
            raise ValueError(f"C14 cutoff must be one of {PILOT_CUTOFFS}")
        self.cutoff = int(cutoff)
        self._fitted = False

    def _select(self, frame: pd.DataFrame) -> pd.DataFrame:
        if (not isinstance(frame, pd.DataFrame) or frame.columns.has_duplicates
                or set(frame.columns) != set(FEATURE_NAMES)
                or len(frame.columns) != len(FEATURE_NAMES)):
            raise ValueError("C14 requires the exact structured + signal_stream feature allowlist")
        if set(FEATURE_NAMES) & {"record_id", "artifact_signal", "threat_rank"}:
            raise ValueError("C14 feature allowlist contains an identity or target")
        return frame.loc[:, list(FEATURE_NAMES)].copy()

    @staticmethod
    def _validate_target(target: Any, rows: int) -> np.ndarray:
        values = np.asarray(target)
        if values.ndim != 1 or values.shape != (rows,):
            raise ValueError("C14 target must be a row-aligned binary vector")
        try:
            numeric = values.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("C14 target must contain binary labels 0/1") from exc
        if (not np.isfinite(numeric).all() or not np.isin(numeric, (0.0, 1.0)).all()
                or not np.array_equal(np.unique(numeric), np.array([0.0, 1.0]))):
            raise ValueError("C14 fit requires finite binary labels containing both classes")
        return numeric.astype(np.int8)

    @staticmethod
    def _cycle_codes(frame: pd.DataFrame, *, label: str) -> np.ndarray:
        try:
            values = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"C14 {label} cycle_code must be numeric") from exc
        if (values.ndim != 1 or not np.isfinite(values).all()
                or not np.equal(values, np.floor(values)).all()):
            raise ValueError(f"C14 {label} cycle_code must contain finite integer values")
        return values.astype(np.int64)

    @staticmethod
    def _numeric_and_category_blocks(structured: sparse.csr_matrix,
                                     numeric_count: int) -> tuple[np.ndarray, sparse.csr_matrix]:
        if numeric_count <= 0 or numeric_count >= structured.shape[1]:
            raise RuntimeError("C14 structured design does not contain numeric and categorical blocks")
        numeric = np.asarray(structured[:, :numeric_count].toarray(), dtype=np.float64)
        categorical = structured[:, numeric_count:].tocsr()
        if not np.isfinite(numeric).all() or not np.isfinite(categorical.data).all():
            raise RuntimeError("C14 structured preprocessing produced nonfinite values")
        return numeric, categorical

    def _feature_blocks(self, selected: pd.DataFrame, *, fitting: bool):
        if fitting:
            structured, self._processor, self._scaler, self._encoder, self._numeric_positions, state = (
                RegressionTextModel._structured_fit(selected)
            )
            self._structured_state = state
            if tuple(state["numeric_feature_names"]) != NUMERIC_TREE_FEATURES:
                raise RuntimeError("C14 numeric tree feature order differs from the frozen 36-column allowlist")
            numeric_count = len(self._numeric_positions)
            numeric, categorical = self._numeric_and_category_blocks(structured.tocsr(), numeric_count)
            self._tree_map = RandomTreesEmbedding(**dict(TREE_PARAMS))
            leaf = self._tree_map.fit_transform(numeric).tocsr().astype(np.float64)
            documents, missing = _text_values(selected["signal_stream"])
            self._vectorizer = TfidfVectorizer(**dict(TFIDF_PARAMS))
            text = self._vectorizer.fit_transform(documents).tocsr().astype(np.float64)
            return numeric, leaf, text, categorical, missing

        structured, unknown = RegressionTextModel._structured_transform(
            selected, self._processor, self._scaler, self._encoder, self._numeric_positions
        )
        numeric_count = len(self._numeric_positions)
        numeric, categorical = self._numeric_and_category_blocks(structured.tocsr(), numeric_count)
        leaf = self._tree_map.transform(numeric).tocsr().astype(np.float64)
        documents, missing = _text_values(selected["signal_stream"])
        text = self._vectorizer.transform(documents).tocsr().astype(np.float64)
        return numeric, leaf, text, categorical, missing, unknown

    def fit(self, frame: pd.DataFrame, target: Any) -> "RandomPartitionTextClassifier":
        self._fitted = False
        selected = self._select(frame)
        y = self._validate_target(target, len(selected))
        cycles = self._cycle_codes(selected, label="fit")
        if len(cycles) == 0 or np.any((cycles < 47) | (cycles > self.cutoff)):
            raise ValueError("C14 fit rows contain a cycle outside 47..cutoff")
        if not selected.index.is_unique or not selected.index.is_monotonic_increasing:
            raise ValueError("C14 fit rows must preserve unique increasing Train positions")

        numeric, leaf, text, categorical, missing = self._feature_blocks(selected, fitting=True)
        missing_column = sparse.csr_matrix(np.asarray(missing, dtype=np.float64).reshape(-1, 1))
        design = sparse.hstack((leaf, text, categorical, missing_column), format="csr", dtype=np.float64)
        design.sum_duplicates()
        design.sort_indices()
        if (design.shape[0] != len(y) or design.shape[1] == 0
                or not np.isfinite(design.data).all()):
            raise RuntimeError("C14 sparse LR design is misaligned, empty, or nonfinite")

        self._estimator = LogisticRegression(**dict(LR_PARAMS))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            self._estimator.fit(design, y)
        if any(issubclass(item.category, ConvergenceWarning) for item in caught):
            raise RuntimeError("C14 LogisticRegression did not converge")

        ordered_vocabulary = [
            (str(token), int(index))
            for token, index in sorted(self._vectorizer.vocabulary_.items(), key=lambda item: item[1])
        ]
        cycle_values, cycle_counts = np.unique(cycles, return_counts=True)
        self._state = {
            "id": "C14_RANDOM_TREES_PARTITIONS_TEXT_LR",
            "cutoff": self.cutoff,
            "fit_rows": int(len(y)),
            "fit_cycle_min": int(cycles.min()),
            "fit_cycle_max": int(cycles.max()),
            "fit_cycle_counts": {str(int(cycle)): int(count)
                                 for cycle, count in zip(cycle_values, cycle_counts)},
            "input_feature_names": list(FEATURE_NAMES),
            "fit_data_sha256": _fit_fingerprint(selected, FEATURE_NAMES, y),
            "target_sha256": _array_digest(y),
            "numeric_tree_input_names": list(NUMERIC_TREE_FEATURES),
            "tree_input_width": int(numeric.shape[1]),
            "random_trees_embedding_parameters": _params_json(TREE_PARAMS),
            "leaf_design_shape": [int(value) for value in leaf.shape],
            "leaf_design_nnz": int(leaf.nnz),
            "leaf_design_sha256": _sparse_digest(leaf),
            "tfidf_parameters": _params_json(TFIDF_PARAMS),
            "vocabulary_size": int(len(ordered_vocabulary)),
            "vocabulary_sha256": _json_digest(ordered_vocabulary),
            "idf_sha256": _array_digest(self._vectorizer.idf_),
            "categorical_ohe_width": int(categorical.shape[1]),
            "missing_stream_fit_rows": int(np.asarray(missing, dtype=bool).sum()),
            "feature_block_order": ["random_tree_leaf_indicators", "word_tfidf",
                                    "categorical_one_hot", "signal_stream_missing_indicator"],
            "structured_preprocessing": self._structured_state,
            "logistic_regression_parameters": _params_json(LR_PARAMS),
            "design_shape": [int(value) for value in design.shape],
            "design_nnz": int(design.nnz),
            "design_sha256": _sparse_digest(design),
            "model_sha256": hashlib.sha256(
                np.asarray(self._estimator.coef_, dtype="<f8").tobytes(order="C")
                + np.asarray(self._estimator.intercept_, dtype="<f8").tobytes(order="C")
            ).hexdigest(),
            "convergence_warning": False,
            "validation_data_passed_to_fit": False,
            "validation_labels_passed_to_fit": False,
            "sample_weight_used": False,
            "early_stopping_used": False,
            "test_data_used": False,
        }
        self._state["state_sha256"] = _json_digest(self._state)
        self._fitted = True
        return self

    def predict_probability(self, frame: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("C14 model must be fitted before prediction")
        selected = self._select(frame)
        numeric, leaf, text, categorical, missing, unknown = self._feature_blocks(selected, fitting=False)
        del numeric  # Numeric values are represented through the fitted leaf map only.
        missing_column = sparse.csr_matrix(np.asarray(missing, dtype=np.float64).reshape(-1, 1))
        design = sparse.hstack((leaf, text, categorical, missing_column), format="csr", dtype=np.float64)
        probability = np.asarray(self._estimator.predict_proba(design)[:, 1], dtype=np.float64)
        if (probability.shape != (len(selected),) or not np.isfinite(probability).all()
                or np.any((probability < 0.0) | (probability > 1.0))):
            raise RuntimeError("C14 returned invalid probabilities")
        self._prediction_diagnostics = {
            "n_predictions": int(len(probability)),
            "missing_stream_rows_scored": int(np.asarray(missing, dtype=bool).sum()),
            "unknown_structured_categories": unknown,
        }
        return probability

    def audit_state(self) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("C14 model must be fitted before audit")
        return json.loads(json.dumps(self._state, sort_keys=True, allow_nan=False))


def _validated_controls(controls: dict[str, Any], n_score: int,
                        actual_missing: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    absent = [name for name in _CONTROL_KEYS if name not in controls]
    if absent or n_score <= 0:
        raise ValueError(f"C14 controls are missing fields or empty: {absent}")
    arrays: dict[str, np.ndarray] = {}
    for name in _CONTROL_KEYS[:-1]:
        dtype = bool if name == "stream_missing" else np.float64
        values = np.asarray(controls[name], dtype=dtype)
        if values.shape != (n_score,):
            raise ValueError(f"C14 control {name} is not aligned")
        if name != "stream_missing" and not np.isfinite(values).all():
            raise ValueError(f"C14 control {name} is nonfinite")
        if name != "stream_missing" and np.any((values < 0.0) | (values > 1.0)):
            raise ValueError(f"C14 control {name} is outside [0,1]")
        arrays[name] = values
    if not np.array_equal(arrays["stream_missing"], np.asarray(actual_missing, dtype=bool)):
        raise ValueError("C14 control missing-stream mask differs from score features")
    rows = np.asarray(controls["control_rows"])
    if (rows.shape != (n_score,) or rows.ndim != 1 or rows.dtype.kind not in "iu"
            or len(np.unique(rows)) != n_score or np.any(np.diff(rows.astype(np.int64)) <= 0)):
        raise ValueError("C14 control rows must be increasing unique source positions")
    expected_baseline = np.where(
        arrays["stream_missing"], arrays["cb_probability"],
        0.75 * arrays["c2b_probability"] + FROZEN_SPARSE_WEIGHT * arrays["baseline_sparse_raw"],
    )
    if not np.array_equal(arrays["baseline_prediction"], expected_baseline):
        raise ValueError("C14 baseline is not the row-matched fixed sparse25 route")
    return arrays, rows.astype(np.int64, copy=False)


def fit_predict(fit_features: pd.DataFrame, fit_target: np.ndarray,
                score_features: pd.DataFrame, *, controls: dict[str, Any],
                card_id: str, cutoff: int) -> dict[str, Any]:
    if card_id != CARD_ID or isinstance(cutoff, bool) or int(cutoff) not in PILOT_CUTOFFS:
        raise ValueError("C14 adapter only accepts its declared card and cutoffs 55/67")
    selector = RandomPartitionTextClassifier(int(cutoff))
    fit_x = selector._select(fit_features)
    score_x = selector._select(score_features)
    if (not fit_x.index.is_unique or not fit_x.index.is_monotonic_increasing
            or not score_x.index.is_unique or not score_x.index.is_monotonic_increasing):
        raise ValueError("C14 fit and score rows must preserve increasing source positions")
    fit_cycles = selector._cycle_codes(fit_x, label="fit")
    score_cycles = selector._cycle_codes(score_x, label="score")
    if (len(fit_cycles) == 0 or len(score_cycles) == 0
            or np.any((fit_cycles < 47) | (fit_cycles > int(cutoff)))
            or np.any((score_cycles <= int(cutoff)) | (score_cycles > int(cutoff) + 12))):
        raise ValueError("C14 row cycles violate the frozen temporal horizon")
    _, actual_missing = _text_values(score_x["signal_stream"])
    arrays, control_rows = _validated_controls(controls, len(score_x), actual_missing)
    if not np.array_equal(score_x.index.to_numpy(dtype=np.int64), control_rows):
        raise ValueError("C14 controls do not match the score row positions")
    model = RandomPartitionTextClassifier(int(cutoff)).fit(fit_x, fit_target)
    raw = model.predict_probability(score_x)
    prediction = np.where(
        arrays["stream_missing"], arrays["cb_probability"],
        CANDIDATE_WEIGHT * raw + FROZEN_SPARSE_WEIGHT * arrays["baseline_sparse_raw"],
    )
    if (prediction.shape != (len(score_x),) or not np.isfinite(prediction).all()
            or np.any((prediction < 0.0) | (prediction > 1.0))):
        raise RuntimeError("C14 fixed availability route returned invalid probabilities")
    return {
        "prediction": prediction,
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
            "candidate_weight_new_present_expert": CANDIDATE_WEIGHT,
            "frozen_sparse_weight_present_expert": FROZEN_SPARSE_WEIGHT,
            "threshold": THRESHOLD,
            "policy": ".75 new C14 hybrid LR + .25 frozen sparse binary on present; frozen CB on missing",
            "score_labels_passed_to_adapter": False,
        },
        "audit_arrays": {"stream_missing": arrays["stream_missing"]},
    }


__all__ = [
    "CANDIDATE_WEIGHT", "CARD_ID", "FEATURE_NAMES", "FROZEN_SPARSE_WEIGHT", "LR_PARAMS", "NUMERIC_TREE_FEATURES",
    "PILOT_CUTOFFS", "RandomPartitionTextClassifier", "THRESHOLD", "TFIDF_PARAMS",
    "TREE_PARAMS", "fit_predict",
]
