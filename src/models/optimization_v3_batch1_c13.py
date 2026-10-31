"""C13 fit-only TF-IDF/SVD32 plus native structured HistGradientBoosting."""
from __future__ import annotations

import hashlib
import json
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer

from src.models.classification_models import (
    ClassificationModel,
    _HGB_PARAMS,
    _TFIDF_PARAMS,
    _json_digest,
    _text_values,
)
from src.models.regression_research_text import HYBRID_FEATURES, _SVD_PARAMS, _array_digest
from src.models.round1_models import _fit_fingerprint, _to_builtin


CARD_ID = "C13"
FEATURE_NAMES = tuple(HYBRID_FEATURES)
PILOT_CUTOFFS = (55, 67)
SVD_COMPONENTS = 32
SVD_PARAMETERS = dict(_SVD_PARAMS)
HGB_PARAMETERS = dict(_HGB_PARAMS)
BLEND_WEIGHT = 0.25
THRESHOLD = 0.50
_CONTROL_KEYS = (
    "baseline_prediction", "baseline_sparse_raw", "c2b_probability",
    "cb_probability", "stream_missing", "control_rows",
)


def _select_exact(frame: pd.DataFrame) -> pd.DataFrame:
    if (not isinstance(frame, pd.DataFrame) or frame.columns.has_duplicates
            or set(frame.columns) != set(FEATURE_NAMES)
            or len(frame.columns) != len(FEATURE_NAMES)):
        raise ValueError("C13 requires the exact structured + signal_stream feature allowlist")
    if set(FEATURE_NAMES) & {"record_id", "artifact_signal", "threat_rank"}:
        raise ValueError("C13 feature allowlist contains an identity or target")
    return frame.loc[:, list(FEATURE_NAMES)].copy()


def _params_json(values: dict[str, Any]) -> dict[str, Any]:
    return {
        key: list(value) if isinstance(value, tuple)
        else value.item() if isinstance(value, np.generic)
        else value
        for key, value in values.items()
    }


def _ordered_vocabulary(vectorizer: TfidfVectorizer) -> list[tuple[str, int]]:
    return [(str(token), int(index))
            for token, index in sorted(vectorizer.vocabulary_.items(), key=lambda item: item[1])]


class SVD32StructuredHGBClassifier:
    """Project fit-only word TF-IDF and append C1_B-native tabular features."""

    input_feature_names = FEATURE_NAMES

    def __init__(self, cutoff: int):
        if int(cutoff) not in PILOT_CUTOFFS:
            raise ValueError(f"C13 cutoff must be one of {PILOT_CUTOFFS}")
        self.cutoff = int(cutoff)
        self._fitted = False
        self._fit_warnings: list[dict[str, str]] = []
        self._state: dict[str, Any] = {}

    def fit(self, frame: pd.DataFrame, target: Any) -> "SVD32StructuredHGBClassifier":
        self._fitted = False
        selected = _select_exact(frame)
        y_numeric = np.asarray(target, dtype=np.float64)
        if (y_numeric.shape != (len(selected),) or not np.isfinite(y_numeric).all()
                or not np.isin(y_numeric, (0.0, 1.0)).all()
                or not np.array_equal(np.unique(y_numeric), np.array([0.0, 1.0]))):
            raise ValueError("C13 requires an aligned binary fit target containing both classes")
        y = y_numeric.astype(np.int8)
        cycles = pd.to_numeric(selected["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
        if (not len(cycles) or np.any((cycles < 47) | (cycles > self.cutoff))
                or not selected.index.is_unique or not selected.index.is_monotonic_increasing):
            raise ValueError("C13 fit rows must be unique ordered Train rows from 47 through cutoff")

        # Reuse C1_B's exact fit-only native structured transform without fitting
        # its classifier as an additional predictor.
        self._structured = ClassificationModel({"id": "C1_B"})
        structured = np.asarray(self._structured._fit_structured(selected), dtype=np.float64)
        structured_mask = list(self._structured._hgb_categorical_mask)
        if structured.ndim != 2 or structured.shape[0] != len(selected):
            raise RuntimeError("C13 C1_B structured fit transform is misaligned")

        documents, missing = _text_values(selected["signal_stream"])
        self._vectorizer = TfidfVectorizer(**dict(_TFIDF_PARAMS))
        text = self._vectorizer.fit_transform(documents).tocsr()
        if text.shape[1] <= SVD_COMPONENTS:
            raise ValueError(
                f"C13 requires more than {SVD_COMPONENTS} fit-only TF-IDF columns; got {text.shape[1]}"
            )
        self._svd = TruncatedSVD(n_components=SVD_COMPONENTS, **dict(SVD_PARAMETERS))
        text_dense = np.asarray(self._svd.fit_transform(text), dtype=np.float64)
        missing_column = missing.astype(np.float64).reshape(-1, 1)
        design = np.column_stack((text_dense, structured, missing_column))
        categorical_mask = [False] * SVD_COMPONENTS + structured_mask + [False]
        if (len(categorical_mask) != design.shape[1]
                or any(categorical_mask[:SVD_COMPONENTS]) or categorical_mask[-1]
                or not np.isfinite(design).all()):
            raise RuntimeError("C13 projected/structured design or categorical mask is invalid")

        self._estimator = HistGradientBoostingClassifier(
            **dict(HGB_PARAMETERS), categorical_features=categorical_mask,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._estimator.fit(design, y)
        self._fit_warnings = [
            {"category": item.category.__name__, "message": str(item.message)}
            for item in caught
        ]
        convergence_warnings = [
            item for item in caught if issubclass(item.category, ConvergenceWarning)
        ]
        if convergence_warnings:
            raise RuntimeError("C13 HistGradientBoostingClassifier did not converge")

        names = (
            [f"signal_stream_svd_{index + 1:02d}" for index in range(SVD_COMPONENTS)]
            + list(self._structured._structured_preprocessor.transformed_feature_names)
            + ["signal_stream_missing"]
        )
        params = self._estimator.get_params(deep=False)
        self._state = {
            "id": "C13_SVD32_STRUCTURED_HGB",
            "fit_rows": int(len(y)),
            "cutoff": self.cutoff,
            "fit_cycle_min": int(cycles.min()),
            "fit_cycle_max": int(cycles.max()),
            "fit_cycle_counts": {
                str(int(cycle)): int(count)
                for cycle, count in zip(*np.unique(cycles, return_counts=True))
            },
            "input_feature_names": list(FEATURE_NAMES),
            "fit_source_rows_sha256": _array_digest(
                selected.index.to_numpy(dtype="<i8"), dtype="<i8"
            ),
            "fit_data_sha256": _fit_fingerprint(selected, FEATURE_NAMES, y),
            "target_sha256": _array_digest(y, dtype="<i8"),
            "tfidf_parameters": _params_json(dict(_TFIDF_PARAMS)),
            "tfidf_vocabulary_size": int(len(self._vectorizer.vocabulary_)),
            "tfidf_vocabulary_sha256": _json_digest(_ordered_vocabulary(self._vectorizer)),
            "tfidf_idf_sha256": _array_digest(self._vectorizer.idf_),
            "svd_parameters": {
                **_params_json(dict(SVD_PARAMETERS)),
                "n_components_requested": SVD_COMPONENTS,
                "n_components_actual": int(self._svd.components_.shape[0]),
                "components_sha256": _array_digest(self._svd.components_),
                "fit_rows": int(len(y)),
            },
            "structured_preprocessing": self._structured._structured_preprocessor.audit_state(),
            "feature_names": names,
            "categorical_mask": categorical_mask,
            "categorical_positions": [
                index for index, categorical in enumerate(categorical_mask) if categorical
            ],
            "projected_text_columns_categorical": False,
            "missing_indicator_categorical": False,
            "training_design_shape": [int(value) for value in design.shape],
            "training_design_sha256": _array_digest(design),
            "hgb_parameters": _params_json(dict(params)),
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
            raise RuntimeError("C13 model must be fitted before prediction")
        selected = _select_exact(frame)
        self._structured._diagnostics = {"n_predictions": int(len(selected))}
        structured = np.asarray(self._structured._transform_structured(selected), dtype=np.float64)
        documents, missing = _text_values(selected["signal_stream"])
        text = self._vectorizer.transform(documents).tocsr()
        text_dense = np.asarray(self._svd.transform(text), dtype=np.float64)
        design = np.column_stack((text_dense, structured, missing.astype(np.float64)))
        probability = np.asarray(self._estimator.predict_proba(design)[:, 1], dtype=np.float64)
        if (probability.shape != (len(selected),) or not np.isfinite(probability).all()
                or np.any((probability < 0.0) | (probability > 1.0))):
            raise RuntimeError("C13 returned invalid positive-class probabilities")
        self._prediction_diagnostics = {
            "n_predictions": int(len(probability)),
            "missing_stream_rows": int(missing.sum()),
            "unknown_structured_categories": dict(
                self._structured._diagnostics.get("unknown_category_rows", {})
            ),
        }
        return probability

    def audit_state(self) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("C13 model must be fitted before audit")
        return json.loads(json.dumps(_to_builtin(self._state), sort_keys=True, allow_nan=False))


def _validated_controls(controls: dict[str, Any], n_score: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    absent = [name for name in _CONTROL_KEYS if name not in controls]
    if absent or n_score <= 0:
        raise ValueError(f"C13 controls are missing fields or empty: {absent}")
    arrays: dict[str, np.ndarray] = {}
    for name in _CONTROL_KEYS[:-1]:
        dtype = bool if name == "stream_missing" else np.float64
        values = np.asarray(controls[name], dtype=dtype)
        if values.shape != (n_score,):
            raise ValueError(f"C13 control {name} is not aligned")
        if name != "stream_missing" and not np.isfinite(values).all():
            raise ValueError(f"C13 control {name} is nonfinite")
        if name in {"baseline_prediction", "baseline_sparse_raw", "c2b_probability", "cb_probability"}:
            if np.any((values < 0.0) | (values > 1.0)):
                raise ValueError(f"C13 control {name} is outside [0,1]")
        arrays[name] = values
    rows = np.asarray(controls["control_rows"])
    if (rows.shape != (n_score,) or rows.dtype.kind not in "iu"
            or len(np.unique(rows)) != n_score or np.any(np.diff(rows.astype(np.int64)) <= 0)):
        raise ValueError("C13 control rows must be increasing unique source positions")
    expected_baseline = np.where(
        arrays["stream_missing"], arrays["cb_probability"],
        0.75 * arrays["c2b_probability"] + BLEND_WEIGHT * arrays["baseline_sparse_raw"],
    )
    if not np.array_equal(arrays["baseline_prediction"], expected_baseline):
        raise ValueError("C13 baseline is not the fixed row-matched sparse25 route")
    return arrays, rows.astype(np.int64, copy=False)


def fit_predict(fit_features: pd.DataFrame, fit_target: np.ndarray,
                score_features: pd.DataFrame, *, controls: dict[str, Any],
                card_id: str, cutoff: int) -> dict[str, Any]:
    if card_id != CARD_ID or int(cutoff) not in PILOT_CUTOFFS:
        raise ValueError("C13 adapter only accepts its declared card and cutoffs 55/67")
    fit_x = _select_exact(fit_features)
    score_x = _select_exact(score_features)
    if (not fit_x.index.is_unique or not fit_x.index.is_monotonic_increasing
            or not score_x.index.is_unique or not score_x.index.is_monotonic_increasing):
        raise ValueError("C13 fit and score rows must preserve increasing source positions")
    fit_cycles = pd.to_numeric(fit_x["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    score_cycles = pd.to_numeric(score_x["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    if (np.any(fit_cycles < 47) or np.any(fit_cycles > int(cutoff))
            or np.any(score_cycles <= int(cutoff))
            or np.any(score_cycles > int(cutoff) + 12)):
        raise ValueError("C13 fit/score rows violate the frozen temporal horizon")
    arrays, control_rows = _validated_controls(controls, len(score_x))
    if not np.array_equal(score_x.index.to_numpy(dtype=np.int64), control_rows):
        raise ValueError("C13 controls do not match the score row positions")
    model = SVD32StructuredHGBClassifier(cutoff).fit(fit_x, fit_target)
    raw = model.predict_probability(score_x)
    prediction = np.where(
        arrays["stream_missing"], arrays["cb_probability"],
        0.75 * arrays["c2b_probability"] + BLEND_WEIGHT * raw,
    )
    if (prediction.shape != (len(score_x),) or not np.isfinite(prediction).all()
            or np.any((prediction < 0.0) | (prediction > 1.0))):
        raise RuntimeError("C13 fixed availability route returned invalid probabilities")
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
            "policy": ".75 frozen C2_B + .25 C13 SVD32+structured HGB on present; frozen CB on missing",
            "score_labels_passed_to_adapter": False,
        },
        "audit_arrays": {"stream_missing": arrays["stream_missing"]},
    }


__all__ = [
    "BLEND_WEIGHT", "CARD_ID", "FEATURE_NAMES", "HGB_PARAMETERS", "PILOT_CUTOFFS",
    "SVD_COMPONENTS", "SVD_PARAMETERS", "SVD32StructuredHGBClassifier", "THRESHOLD",
    "fit_predict",
]
