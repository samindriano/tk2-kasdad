"""C19: XGBoost binary classifier on structured + fit-only SVD32 text inputs."""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import xgboost
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier

from src.models.classification_models import (
    _TFIDF_PARAMS as _CLASSIFICATION_TFIDF_PARAMS,
    _array_digest as _classification_array_digest,
    _json_digest as _classification_json_digest,
    _text_values,
)
from src.models.regression_research_text import (
    HYBRID_FEATURES,
    RegressionTextModel,
    _SVD_PARAMS,
    _array_digest,
    _json_digest,
)
from src.models.round1_models import _fit_fingerprint, _to_builtin


CARD_ID = "C19"
FEATURE_NAMES = tuple(HYBRID_FEATURES)
PILOT_CUTOFFS = (55, 67)
SVD_COMPONENTS = 32
SVD_PARAMETERS = dict(_SVD_PARAMS)
TFIDF_PARAMETERS = dict(_CLASSIFICATION_TFIDF_PARAMS)
BLEND_WEIGHT = 0.25
THRESHOLD = 0.50
XGB_PARAMETERS = {
    "objective": "binary:logistic",
    "tree_method": "hist",
    "device": "cpu",
    "n_estimators": 600,
    "max_depth": 4,
    "learning_rate": 0.05,
    "min_child_weight": 20,
    "reg_lambda": 5.0,
    "colsample_bytree": 0.8,
    "subsample": 1.0,
    "random_state": 42,
    "n_jobs": 2,
    "verbosity": 0,
}
_CONTROL_KEYS = (
    "baseline_prediction", "baseline_sparse_raw", "c2b_probability",
    "cb_probability", "stream_missing", "control_rows",
)


def _select_exact(frame: pd.DataFrame) -> pd.DataFrame:
    if (not isinstance(frame, pd.DataFrame) or frame.columns.has_duplicates
            or set(frame.columns) != set(FEATURE_NAMES)
            or len(frame.columns) != len(FEATURE_NAMES)):
        raise ValueError("C19 requires exactly the expanded structured + signal_stream feature allowlist")
    if set(FEATURE_NAMES) & {"record_id", "artifact_signal", "threat_rank"}:
        raise ValueError("C19 feature allowlist contains an identity or target")
    return frame.loc[:, list(FEATURE_NAMES)].copy()


def _validated_controls(controls: dict[str, Any], n_score: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    absent = [name for name in _CONTROL_KEYS if name not in controls]
    if absent or n_score <= 0:
        raise ValueError(f"C19 controls are missing fields or empty: {absent}")
    arrays: dict[str, np.ndarray] = {}
    for name in _CONTROL_KEYS[:-1]:
        dtype = bool if name == "stream_missing" else np.float64
        values = np.asarray(controls[name], dtype=dtype)
        if values.shape != (n_score,):
            raise ValueError(f"C19 control {name} is not aligned")
        if name != "stream_missing" and not np.isfinite(values).all():
            raise ValueError(f"C19 control {name} is nonfinite")
        if name in {"baseline_prediction", "baseline_sparse_raw", "c2b_probability", "cb_probability"}:
            if np.any((values < 0.0) | (values > 1.0)):
                raise ValueError(f"C19 control {name} is outside [0,1]")
        arrays[name] = values
    rows = np.asarray(controls["control_rows"])
    if (rows.shape != (n_score,) or rows.dtype.kind not in "iu"
            or len(np.unique(rows)) != n_score or np.any(np.diff(rows.astype(np.int64)) <= 0)):
        raise ValueError("C19 control rows must be increasing unique source positions")
    expected_baseline = np.where(
        arrays["stream_missing"], arrays["cb_probability"],
        0.75 * arrays["c2b_probability"] + 0.25 * arrays["baseline_sparse_raw"],
    )
    if not np.array_equal(arrays["baseline_prediction"], expected_baseline):
        raise ValueError("C19 baseline is not the fixed row-matched sparse25 route")
    return arrays, rows.astype(np.int64, copy=False)


class XGBoostSVD32Classifier:
    """Fit-only scaled numeric/OHE + TF-IDF/SVD32, then CPU XGBoost."""

    input_feature_names = FEATURE_NAMES

    def __init__(self, cutoff: int):
        if int(cutoff) not in PILOT_CUTOFFS:
            raise ValueError(f"C19 cutoff must be one of {PILOT_CUTOFFS}")
        self.cutoff = int(cutoff)
        self._fitted = False
        self._fit_warnings: list[dict[str, str]] = []
        self._state: dict[str, Any] = {}

    def _design(self, selected: pd.DataFrame, *, fit: bool) -> tuple[np.ndarray, np.ndarray]:
        documents, missing = _text_values(selected["signal_stream"])
        if fit:
            self._vectorizer = TfidfVectorizer(**dict(TFIDF_PARAMETERS))
            text_sparse = self._vectorizer.fit_transform(documents).tocsr()
            if text_sparse.shape[1] <= SVD_COMPONENTS:
                raise ValueError(
                    f"C19 requires at least {SVD_COMPONENTS + 1} fit-only TF-IDF columns; "
                    f"got {text_sparse.shape[1]}"
                )
            self._svd = TruncatedSVD(n_components=SVD_COMPONENTS, **dict(SVD_PARAMETERS))
            text_dense = np.asarray(self._svd.fit_transform(text_sparse), dtype=np.float64)
            (structured_sparse, self._processor, self._scaler, self._encoder,
             self._numeric_positions, self._structured_state) = RegressionTextModel._structured_fit(selected)
        else:
            text_sparse = self._vectorizer.transform(documents).tocsr()
            text_dense = np.asarray(self._svd.transform(text_sparse), dtype=np.float64)
            structured_sparse, _unknown = RegressionTextModel._structured_transform(
                selected, self._processor, self._scaler, self._encoder, self._numeric_positions
            )
        structured_dense = np.asarray(structured_sparse.toarray(), dtype=np.float64)
        design = np.column_stack((structured_dense, text_dense, missing.astype(np.float64)))
        if design.shape[0] != len(selected) or not np.isfinite(design).all():
            raise RuntimeError("C19 dense SVD32 design is misaligned or nonfinite")
        return design, missing

    def fit(self, frame: pd.DataFrame, target: Any) -> "XGBoostSVD32Classifier":
        self._fitted = False
        selected = _select_exact(frame)
        y_numeric = np.asarray(target, dtype=np.float64)
        if (y_numeric.shape != (len(selected),) or not np.isfinite(y_numeric).all()
                or not np.isin(y_numeric, (0.0, 1.0)).all()
                or not np.array_equal(np.unique(y_numeric), np.array([0.0, 1.0]))):
            raise ValueError("C19 requires an aligned binary fit target containing both classes")
        y = y_numeric.astype(np.int8)
        cycles = pd.to_numeric(selected["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
        if (not len(cycles) or np.any((cycles < 47) | (cycles > self.cutoff))
                or not selected.index.is_unique or not selected.index.is_monotonic_increasing):
            raise ValueError("C19 fit rows must be increasing official Train rows through the cutoff")

        design, missing = self._design(selected, fit=True)
        self._estimator = XGBClassifier(**dict(XGB_PARAMETERS))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._estimator.fit(design, y, verbose=False)
        self._fit_warnings = [
            {"category": item.category.__name__, "message": str(item.message)} for item in caught
        ]
        effective: dict[str, Any] = {}
        for name, value in self._estimator.get_params(deep=False).items():
            if isinstance(value, (float, np.floating)) and not np.isfinite(float(value)):
                effective[name] = "NaN (estimator default)"
            elif isinstance(value, np.generic):
                effective[name] = value.item()
            elif isinstance(value, tuple):
                effective[name] = list(value)
            elif isinstance(value, np.ndarray):
                effective[name] = value.tolist()
            else:
                effective[name] = value
        ordered_vocabulary = [
            (str(token), int(index))
            for token, index in sorted(self._vectorizer.vocabulary_.items(), key=lambda item: item[1])
        ]
        names = (
            list(self._processor.transformed_feature_names)
            + [f"signal_stream_svd_{index + 1:02d}" for index in range(SVD_COMPONENTS)]
            + ["signal_stream_missing"]
        )
        self._state = {
            "card_id": CARD_ID,
            "estimator": "xgboost.XGBClassifier",
            "xgboost_version": xgboost.__version__,
            "parameters_explicit": dict(XGB_PARAMETERS),
            "parameters_effective": effective,
            "fit_rows": int(len(y)),
            "cutoff": self.cutoff,
            "fit_cycle_min": int(cycles.min()),
            "fit_cycle_max": int(cycles.max()),
            "fit_source_rows_sha256": _array_digest(selected.index.to_numpy(dtype="<i8"), dtype="<i8"),
            "fit_data_sha256": _fit_fingerprint(selected, FEATURE_NAMES, y),
            "target_sha256": _array_digest(y, dtype="<i8"),
            "feature_names": list(FEATURE_NAMES),
            "transformed_feature_names": names,
            "training_design_shape": [int(value) for value in design.shape],
            "training_design_sha256": _array_digest(design),
            "structured_preprocessing": self._structured_state,
            "structured_dense_conversion": "fit-only scaled numeric/OHE sparse matrix converted to float64 dense",
            "tfidf_parameters": {
                key: list(value) if isinstance(value, tuple)
                else value.__name__ if isinstance(value, type)
                else value.item() if isinstance(value, np.generic) else value
                for key, value in TFIDF_PARAMETERS.items()
            },
            "tfidf_vocabulary_size": int(len(ordered_vocabulary)),
            "tfidf_vocabulary_sha256": _classification_json_digest(ordered_vocabulary),
            "tfidf_idf_sha256": _classification_array_digest(self._vectorizer.idf_),
            "svd": {
                **dict(SVD_PARAMETERS),
                "n_components_requested": SVD_COMPONENTS,
                "n_components_actual": int(self._svd.components_.shape[0]),
                "components_sha256": _array_digest(self._svd.components_),
                "fit_rows": int(len(y)),
            },
            "fit_missing_stream_rows": int(missing.sum()),
            "fit_warnings": list(self._fit_warnings),
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
            raise RuntimeError("C19 classifier must be fitted before prediction")
        selected = _select_exact(frame)
        design, missing = self._design(selected, fit=False)
        probability = np.asarray(self._estimator.predict_proba(design)[:, 1], dtype=np.float64)
        if (probability.shape != (len(selected),) or not np.isfinite(probability).all()
                or np.any((probability < 0.0) | (probability > 1.0))):
            raise RuntimeError("C19 returned invalid positive-class probabilities")
        self._prediction_diagnostics = {
            "n_predictions": int(len(probability)),
            "missing_stream_rows": int(missing.sum()),
        }
        return probability

    def audit_state(self) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("C19 model must be fitted before audit")
        return _to_builtin(self._state)


def fit_predict(fit_features: pd.DataFrame, fit_target: np.ndarray,
                score_features: pd.DataFrame, *, controls: dict[str, Any],
                card_id: str, cutoff: int) -> dict[str, Any]:
    if card_id != CARD_ID or int(cutoff) not in PILOT_CUTOFFS:
        raise ValueError("C19 adapter only accepts its declared card and cutoffs 55/67")
    fit_x = _select_exact(fit_features)
    score_x = _select_exact(score_features)
    if (not fit_x.index.is_unique or not fit_x.index.is_monotonic_increasing
            or not score_x.index.is_unique or not score_x.index.is_monotonic_increasing):
        raise ValueError("C19 fit and score rows must preserve increasing source positions")
    fit_cycles = pd.to_numeric(fit_x["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    score_cycles = pd.to_numeric(score_x["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    if (np.any(fit_cycles < 47) or np.any(fit_cycles > int(cutoff))
            or np.any(score_cycles <= int(cutoff))
            or np.any(score_cycles > int(cutoff) + 12)):
        raise ValueError("C19 fit/score rows violate the frozen temporal horizon")
    arrays, control_rows = _validated_controls(controls, len(score_x))
    if not np.array_equal(score_x.index.to_numpy(dtype=np.int64), control_rows):
        raise ValueError("C19 controls do not match score row positions")
    model = XGBoostSVD32Classifier(cutoff).fit(fit_x, fit_target)
    raw = model.predict_probability(score_x)
    current_present = np.where(
        arrays["stream_missing"], arrays["cb_probability"],
        0.75 * arrays["c2b_probability"] + 0.25 * arrays["baseline_sparse_raw"],
    )
    prediction = np.where(
        arrays["stream_missing"], arrays["cb_probability"],
        (1.0 - BLEND_WEIGHT) * current_present + BLEND_WEIGHT * raw,
    )
    if (prediction.shape != (len(score_x),) or not np.isfinite(prediction).all()
            or np.any((prediction < 0.0) | (prediction > 1.0))):
        raise RuntimeError("C19 fixed availability route returned invalid probabilities")
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
            "current_route_composition": "0.75 C2_B + 0.25 frozen sparse expert on present; frozen CB on missing",
            "blend_weight_new_present_expert": BLEND_WEIGHT,
            "threshold": THRESHOLD,
            "policy": ".75 current sparse25 probability + .25 C19 on present; frozen CB on missing",
            "score_labels_passed_to_adapter": False,
        },
        "audit_arrays": {"stream_missing": arrays["stream_missing"]},
    }


__all__ = [
    "BLEND_WEIGHT", "CARD_ID", "FEATURE_NAMES", "PILOT_CUTOFFS", "SVD_COMPONENTS",
    "SVD_PARAMETERS", "TFIDF_PARAMETERS", "THRESHOLD", "XGB_PARAMETERS",
    "XGBoostSVD32Classifier", "fit_predict",
]
