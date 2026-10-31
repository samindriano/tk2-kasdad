"""C18 sparse LightGBM with randomized thresholds and fixed availability routing."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src.models.classification_models import _TFIDF_PARAMS, _text_values
from src.models.optimization_v3_sparse_classification import PARAMS as BASE_PARAMS
from src.models.regression_research_text import (
    HYBRID_FEATURES,
    RegressionTextModel,
    _array_digest,
)
from src.models.round1_models import _fit_fingerprint


FEATURE_NAMES = tuple(HYBRID_FEATURES)
PARAMS = {**dict(BASE_PARAMS), "extra_trees": True, "extra_seed": 42}
BLEND_WEIGHT = 0.25
THRESHOLD = 0.50


def _json_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _safe_params(params: dict[str, Any]) -> dict[str, Any]:
    return {key: (list(value) if isinstance(value, tuple) else
                  value.__name__ if isinstance(value, type) else value)
            for key, value in params.items()}


class RandomThresholdSparseTextClassifier:
    """Fit the original sparse design, changing only LightGBM threshold randomness."""

    input_feature_names = FEATURE_NAMES

    def __init__(self, cutoff: int | None = None) -> None:
        self._fitted = False
        self._cutoff = None if cutoff is None else int(cutoff)

    def _select(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.columns.has_duplicates or set(frame.columns) != set(FEATURE_NAMES):
            raise ValueError("C18 requires the exact structured + signal_stream feature allowlist")
        return frame.loc[:, list(FEATURE_NAMES)].copy()

    def fit(self, frame: pd.DataFrame, target: Any) -> "RandomThresholdSparseTextClassifier":
        self._fitted = False
        selected = self._select(frame)
        y = np.asarray(target, dtype=np.int8)
        if y.shape != (len(selected),) or not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
            raise ValueError("C18 requires a row-aligned binary fit target with both classes")
        fit_cycles = pd.to_numeric(selected["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
        if (fit_cycles.ndim != 1 or not len(fit_cycles) or np.any((fit_cycles < 47) | (fit_cycles > 79))
                or (self._cutoff is not None and np.any(fit_cycles > self._cutoff))):
            raise ValueError("C18 fit cycle_code must be within the current cutoff")

        (structured, self._processor, self._scaler, self._encoder,
         self._numeric_positions, structured_state) = RegressionTextModel._structured_fit(selected)
        documents, missing = _text_values(selected["signal_stream"])
        self._vectorizer = TfidfVectorizer(**dict(_TFIDF_PARAMS))
        text = self._vectorizer.fit_transform(documents).tocsr()
        missing_column = sparse.csr_matrix(missing.astype(np.float64).reshape(-1, 1))
        design = sparse.hstack((structured, text, missing_column), format="csr", dtype=np.float64)

        self._estimator = LGBMClassifier(**dict(PARAMS))
        self._estimator.fit(design, y)
        ordered_vocabulary = [
            (str(token), int(index))
            for token, index in sorted(self._vectorizer.vocabulary_.items(), key=lambda item: item[1])
        ]
        self._state = {
            "id": "C18_RANDOM_THRESHOLD_SPARSE_LGBM",
            "fit_rows": int(len(y)),
            "cutoff": self._cutoff,
            "fit_cycle_min": int(fit_cycles.min()),
            "fit_cycle_max": int(fit_cycles.max()),
            "fit_cycle_counts": {
                str(cycle): int(count) for cycle, count in zip(*np.unique(fit_cycles, return_counts=True))
            },
            "input_feature_names": list(FEATURE_NAMES),
            "fit_data_sha256": _fit_fingerprint(selected, FEATURE_NAMES, y),
            "target_sha256": _array_digest(y),
            "lightgbm_parameters": dict(PARAMS),
            "only_changes_vs_sparse_binary": {"extra_trees": True, "extra_seed": 42},
            "tfidf_parameters": _safe_params(dict(_TFIDF_PARAMS)),
            "vocabulary_size": int(len(ordered_vocabulary)),
            "vocabulary_sha256": _json_digest(ordered_vocabulary),
            "idf_sha256": _array_digest(self._vectorizer.idf_),
            "structured_state": structured_state,
            "design_shape": [int(value) for value in design.shape],
            "design_nnz": int(design.nnz),
            "design_data_sha256": _array_digest(design.data),
            "model_sha256": hashlib.sha256(
                self._estimator.booster_.model_to_string().encode("utf-8")
            ).hexdigest(),
            "validation_data_passed_to_fit": False,
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
            raise RuntimeError("C18 model must be fitted before prediction")
        selected = self._select(frame)
        structured, unknown = RegressionTextModel._structured_transform(
            selected, self._processor, self._scaler, self._encoder, self._numeric_positions
        )
        documents, missing = _text_values(selected["signal_stream"])
        text = self._vectorizer.transform(documents)
        missing_column = sparse.csr_matrix(missing.astype(np.float64).reshape(-1, 1))
        design = sparse.hstack((structured, text, missing_column), format="csr", dtype=np.float64)
        probability = np.asarray(
            self._estimator.booster_.predict(design, num_threads=2), dtype=np.float64
        )
        if (probability.shape != (len(selected),) or not np.isfinite(probability).all()
                or np.any((probability < 0) | (probability > 1))):
            raise RuntimeError("C18 produced invalid probabilities")
        self._prediction_diagnostics = {
            "n_predictions": int(len(probability)),
            "missing_stream_rows_scored": int(missing.sum()),
            "unknown_structured_categories": unknown,
        }
        return probability

    def audit_state(self) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("C18 model must be fitted before audit")
        return json.loads(json.dumps(self._state, sort_keys=True, allow_nan=False))


def fit_predict(
    fit_features: pd.DataFrame,
    fit_target: np.ndarray,
    score_features: pd.DataFrame,
    *,
    controls: dict[str, np.ndarray],
    card_id: str,
    cutoff: int,
) -> dict[str, Any]:
    if card_id != "C18" or cutoff not in (55, 67):
        raise ValueError("C18 adapter only accepts its declared card and cutoffs 55/67")
    model = RandomThresholdSparseTextClassifier(cutoff=cutoff).fit(fit_features, fit_target)
    raw = model.predict_probability(score_features)
    missing = np.asarray(controls["stream_missing"], dtype=bool)
    c2b = np.asarray(controls["c2b_probability"], dtype=np.float64)
    cb = np.asarray(controls["cb_probability"], dtype=np.float64)
    old_sparse = np.asarray(controls["baseline_sparse_raw"], dtype=np.float64)
    baseline = np.asarray(controls["baseline_prediction"], dtype=np.float64)
    if not (missing.shape == c2b.shape == cb.shape == old_sparse.shape == baseline.shape == raw.shape):
        raise ValueError("C18 baseline, route, and scoring arrays are not row aligned")
    if (not np.isfinite(c2b).all() or not np.isfinite(cb).all()
            or not np.isfinite(old_sparse).all() or not np.isfinite(baseline).all()
            or np.any((c2b < 0) | (c2b > 1)) or np.any((cb < 0) | (cb > 1))
            or np.any((old_sparse < 0) | (old_sparse > 1))):
        raise ValueError("C18 routing controls must be finite probabilities")

    reconstructed_baseline = np.where(
        missing, cb, 0.75 * c2b + BLEND_WEIGHT * old_sparse
    )
    if not np.array_equal(baseline, reconstructed_baseline):
        raise ValueError("C18 incumbent control is not the fixed sparse25 row-matched recipe")
    prediction = np.where(missing, cb, 0.75 * c2b + BLEND_WEIGHT * raw)
    return {
        "prediction": prediction,
        "raw_candidate": raw,
        "fit_predictor_count": 1,
        "state": model.audit_state(),
        "fit_metadata": {
            **model._prediction_diagnostics,
            "cutoff": int(cutoff),
            "candidate_present_stream_rows": int((~missing).sum()),
            "frozen_cb_missing_stream_rows": int(missing.sum()),
            "blend_weight_new_present_expert": BLEND_WEIGHT,
            "threshold": THRESHOLD,
        },
        "audit_arrays": {"stream_missing": missing},
    }


__all__ = [
    "BLEND_WEIGHT",
    "FEATURE_NAMES",
    "PARAMS",
    "RandomThresholdSparseTextClassifier",
    "THRESHOLD",
    "fit_predict",
]
