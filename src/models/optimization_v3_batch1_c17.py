"""C17: fixed longer-regularized variant of the sparse LightGBM classifier."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src.models.classification_models import _TFIDF_PARAMS, _text_values
from src.models.optimization_v3_sparse_classification import PARAMS as BASE_PARAMS
from src.models.regression_research_text import HYBRID_FEATURES, RegressionTextModel, _array_digest
from src.models.round1_models import _fit_fingerprint


CARD_ID = "C17"
FEATURE_NAMES = tuple(HYBRID_FEATURES)
PILOT_CUTOFFS = (55, 67)
CHANGES = {"num_leaves": 15, "min_child_samples": 100, "reg_lambda": 5.0, "n_estimators": 600}
PARAMS = {**dict(BASE_PARAMS), **CHANGES}
BLEND_WEIGHT = 0.25
THRESHOLD = 0.50
_CONTROL_KEYS = ("baseline_prediction", "baseline_sparse_raw", "c2b_probability",
                 "cb_probability", "stream_missing", "control_rows")


def _json_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _params_json(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: (list(item) if isinstance(item, tuple)
                  else item.__name__ if isinstance(item, type) else item)
            for key, item in value.items()}

class LongerRegularizedSparseClassifier:
    """Keep the inherited sparse text design and use C17's fixed tree recipe."""

    input_feature_names = FEATURE_NAMES

    def __init__(self, cutoff: int):
        if int(cutoff) not in PILOT_CUTOFFS:
            raise ValueError(f"C17 cutoff must be one of {PILOT_CUTOFFS}")
        self.cutoff = int(cutoff)
        self._fitted = False

    def _select(self, frame: pd.DataFrame) -> pd.DataFrame:
        if (not isinstance(frame, pd.DataFrame) or frame.columns.has_duplicates
                or set(frame.columns) != set(FEATURE_NAMES)
                or len(frame.columns) != len(FEATURE_NAMES)):
            raise ValueError("C17 requires the exact structured + signal_stream feature allowlist")
        if set(FEATURE_NAMES) & {"record_id", "artifact_signal", "threat_rank"}:
            raise ValueError("C17 feature allowlist contains an identity or target")
        return frame.loc[:, list(FEATURE_NAMES)].copy()

    def fit(self, frame: pd.DataFrame, target: Any) -> "LongerRegularizedSparseClassifier":
        self._fitted = False
        selected = self._select(frame)
        y = np.asarray(target, dtype=np.int8)
        if y.shape != (len(selected),) or not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
            raise ValueError("C17 requires a row-aligned binary fit target containing both classes")
        fit_cycles = pd.to_numeric(selected["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
        if (fit_cycles.ndim != 1 or len(fit_cycles) == 0
                or np.any((fit_cycles < 47) | (fit_cycles > self.cutoff))):
            raise ValueError("C17 fit rows contain a cycle outside 47..cutoff")
        if not selected.index.is_unique or not selected.index.is_monotonic_increasing:
            raise ValueError("C17 fit rows must preserve unique increasing Train positions")

        (structured, self._processor, self._scaler, self._encoder,
         self._numeric_positions, structured_state) = RegressionTextModel._structured_fit(selected)
        documents, missing = _text_values(selected["signal_stream"])
        self._vectorizer = TfidfVectorizer(**dict(_TFIDF_PARAMS))
        text = self._vectorizer.fit_transform(documents).tocsr()
        missing_column = sparse.csr_matrix(missing.astype(np.float64).reshape(-1, 1))
        design = sparse.hstack((structured, text, missing_column), format="csr", dtype=np.float64)
        if design.shape[0] != len(y) or not np.isfinite(design.data).all():
            raise RuntimeError("C17 sparse fit design is misaligned or nonfinite")

        self._estimator = LGBMClassifier(**dict(PARAMS))
        self._estimator.fit(design, y)
        ordered_vocabulary = [
            (str(token), int(index))
            for token, index in sorted(self._vectorizer.vocabulary_.items(), key=lambda item: item[1])
        ]
        self._state = {
            "id": "C17_LONGER_REGULARIZED_SPARSE_LGBM",
            "fit_rows": int(len(y)), "cutoff": self.cutoff,
            "fit_cycle_min": int(fit_cycles.min()), "fit_cycle_max": int(fit_cycles.max()),
            "fit_cycle_counts": {str(int(cycle)): int(count)
                                 for cycle, count in zip(*np.unique(fit_cycles, return_counts=True))},
            "input_feature_names": list(FEATURE_NAMES),
            "fit_data_sha256": _fit_fingerprint(selected, FEATURE_NAMES, y),
            "target_sha256": _array_digest(y),
            "base_parameters": dict(BASE_PARAMS), "lightgbm_parameters": dict(PARAMS),
            "only_parameter_changes": dict(CHANGES),
            "tfidf_parameters": _params_json(_TFIDF_PARAMS),
            "vocabulary_size": int(len(ordered_vocabulary)),
            "vocabulary_sha256": _json_digest(ordered_vocabulary),
            "idf_sha256": _array_digest(self._vectorizer.idf_),
            "structured_preprocessing": structured_state,
            "design_shape": [int(value) for value in design.shape],
            "design_nnz": int(design.nnz),
            "design_data_sha256": _array_digest(design.data),
            "model_sha256": hashlib.sha256(
                self._estimator.booster_.model_to_string().encode("utf-8")
            ).hexdigest(),
            "validation_data_passed_to_fit": False,
            "validation_labels_passed_to_fit": False,
            "early_stopping_used": False,
            "sample_weight_used": False,
            "test_data_used": False,
        }
        self._state["state_sha256"] = _json_digest(self._state)
        self._prediction_diagnostics = {"n_predictions": 0}
        self._fitted = True
        return self

    def predict_probability(self, frame: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("C17 model must be fitted before prediction")
        selected = self._select(frame)
        structured, unknown = RegressionTextModel._structured_transform(
            selected, self._processor, self._scaler, self._encoder, self._numeric_positions
        )
        documents, missing = _text_values(selected["signal_stream"])
        text = self._vectorizer.transform(documents).tocsr()
        missing_column = sparse.csr_matrix(missing.astype(np.float64).reshape(-1, 1))
        design = sparse.hstack((structured, text, missing_column), format="csr", dtype=np.float64)
        probability = np.asarray(self._estimator.booster_.predict(design, num_threads=2), dtype=np.float64)
        if (probability.shape != (len(selected),) or not np.isfinite(probability).all()
                or np.any((probability < 0.0) | (probability > 1.0))):
            raise RuntimeError("C17 returned invalid probabilities")
        self._prediction_diagnostics = {
            "n_predictions": int(len(probability)),
            "missing_stream_rows_scored": int(missing.sum()),
            "unknown_structured_categories": unknown,
        }
        return probability

    def audit_state(self) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("C17 model must be fitted before audit")
        return json.loads(json.dumps(self._state, sort_keys=True, allow_nan=False))


def _validated_controls(controls: dict[str, Any], n_score: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    absent = [name for name in _CONTROL_KEYS if name not in controls]
    if absent or n_score <= 0:
        raise ValueError(f"C17 controls are missing fields or empty: {absent}")
    arrays: dict[str, np.ndarray] = {}
    for name in _CONTROL_KEYS[:-1]:
        dtype = bool if name == "stream_missing" else np.float64
        values = np.asarray(controls[name], dtype=dtype)
        if values.shape != (n_score,):
            raise ValueError(f"C17 control {name} is not aligned")
        if name != "stream_missing" and not np.isfinite(values).all():
            raise ValueError(f"C17 control {name} is nonfinite")
        if name in {"baseline_prediction", "baseline_sparse_raw", "c2b_probability", "cb_probability"}:
            if np.any((values < 0.0) | (values > 1.0)):
                raise ValueError(f"C17 control {name} is outside [0,1]")
        arrays[name] = values
    rows = np.asarray(controls["control_rows"])
    if (rows.shape != (n_score,) or rows.ndim != 1 or rows.dtype.kind not in "iu"
            or len(np.unique(rows)) != n_score or np.any(np.diff(rows.astype(np.int64)) <= 0)):
        raise ValueError("C17 control rows must be increasing unique source positions")
    expected_baseline = np.where(
        arrays["stream_missing"], arrays["cb_probability"],
        0.75 * arrays["c2b_probability"] + BLEND_WEIGHT * arrays["baseline_sparse_raw"],
    )
    if not np.array_equal(arrays["baseline_prediction"], expected_baseline):
        raise ValueError("C17 baseline is not the row-matched fixed sparse25 route")
    return arrays, rows.astype(np.int64, copy=False)


def fit_predict(fit_features: pd.DataFrame, fit_target: np.ndarray,
                score_features: pd.DataFrame, *, controls: dict[str, Any],
                card_id: str, cutoff: int) -> dict[str, Any]:
    if card_id != CARD_ID or int(cutoff) not in PILOT_CUTOFFS:
        raise ValueError("C17 adapter only accepts its declared card and cutoffs 55/67")
    selector = LongerRegularizedSparseClassifier(cutoff)
    fit_x = selector._select(fit_features)
    score_x = selector._select(score_features)
    if (not fit_x.index.is_unique or not fit_x.index.is_monotonic_increasing
            or not score_x.index.is_unique or not score_x.index.is_monotonic_increasing):
        raise ValueError("C17 fit and score rows must preserve increasing source positions")
    fit_cycles = pd.to_numeric(fit_x["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    score_cycles = pd.to_numeric(score_x["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    if (np.any(fit_cycles > int(cutoff))
            or np.any((score_cycles <= int(cutoff)) | (score_cycles > int(cutoff) + 12))):
        raise ValueError("C17 row cycles violate the frozen temporal horizon")
    arrays, control_rows = _validated_controls(controls, len(score_x))
    if not np.array_equal(score_x.index.to_numpy(dtype=np.int64), control_rows):
        raise ValueError("C17 controls do not match the score row positions")
    model = LongerRegularizedSparseClassifier(cutoff).fit(fit_x, fit_target)
    raw = model.predict_probability(score_x)
    prediction = np.where(
        arrays["stream_missing"], arrays["cb_probability"],
        0.75 * arrays["c2b_probability"] + BLEND_WEIGHT * raw,
    )
    if (prediction.shape != (len(score_x),) or not np.isfinite(prediction).all()
            or np.any((prediction < 0.0) | (prediction > 1.0))):
        raise RuntimeError("C17 fixed availability route returned invalid probabilities")
    return {
        "prediction": prediction, "raw_candidate": raw, "fit_predictor_count": 1,
        "state": model.audit_state(),
        "fit_metadata": {
            **model._prediction_diagnostics,
            "cutoff": int(cutoff), "fit_rows": int(len(fit_x)),
            "fit_cycle_min": int(fit_cycles.min()), "fit_cycle_max": int(fit_cycles.max()),
            "candidate_present_stream_rows": int((~arrays["stream_missing"]).sum()),
            "frozen_cb_missing_stream_rows": int(arrays["stream_missing"].sum()),
            "blend_weight_new_present_expert": BLEND_WEIGHT,
            "threshold": THRESHOLD,
            "policy": ".75 frozen C2_B + .25 C17 on present; frozen CB on missing",
            "score_labels_passed_to_adapter": False,
        },
        "audit_arrays": {"stream_missing": arrays["stream_missing"]},
    }


__all__ = [
    "BLEND_WEIGHT", "CARD_ID", "CHANGES", "FEATURE_NAMES", "LongerRegularizedSparseClassifier",
    "PARAMS", "PILOT_CUTOFFS", "THRESHOLD", "fit_predict",
]
