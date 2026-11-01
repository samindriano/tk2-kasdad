"""R23: ExtraTrees absolute-error model on structured + SVD32 dense inputs.

Every learned transform and the estimator are fit only on the temporal fit
rows. The adapter receives no validation labels and does not load data.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import rankdata
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor

from src.models.regression_research_text import (
    HYBRID_FEATURES,
    RegressionTextModel,
    _SVD_PARAMS,
    _TFIDF_PARAMS,
    _array_digest,
    _json_digest,
    _text_documents,
    _target,
)


CARD_ID = "R23"
FEATURE_NAMES = HYBRID_FEATURES
PILOT_CUTOFFS = (55, 67)
EXTRA_TREES_PARAMETERS = {
    "n_estimators": 200,
    "criterion": "absolute_error",
    "min_samples_leaf": 30,
    "max_features": 1.0,
    "random_state": 42,
    "n_jobs": 2,
}
_CONTROL_KEYS = (
    "baseline_prediction",
    "baseline_lgb_svd32",
    "baseline_hybrid_ridge",
    "baseline_sparse_raw",
    "baseline_sparse_rank50",
    "control_rows",
)


def _select_exact(X: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("R23 inputs must be pandas DataFrames")
    if X.columns.has_duplicates or set(X.columns) != set(FEATURE_NAMES) or len(X.columns) != len(FEATURE_NAMES):
        raise ValueError("R23 inputs must contain exactly the frozen hybrid feature allowlist")
    if set(FEATURE_NAMES) & {"record_id", "artifact_signal", "threat_rank"}:
        raise ValueError("R23 feature allowlist contains an identity or target")
    return X.loc[:, list(FEATURE_NAMES)].copy()


def rank50_prediction(raw_prediction: Any, cycle_codes: Any) -> np.ndarray:
    """Map target-scale predictions to the frozen within-cycle rank50 scale."""
    raw = np.asarray(raw_prediction, dtype=np.float64)
    cycles = np.asarray(cycle_codes)
    if raw.ndim != 1 or cycles.shape != raw.shape or raw.size == 0:
        raise ValueError("R23 rank50 inputs must be nonempty aligned vectors")
    if not np.isfinite(raw).all() or cycles.dtype.kind not in "iu":
        raise ValueError("R23 rank50 needs finite predictions and integer cycle codes")
    mapped = np.empty(raw.shape, dtype=np.float64)
    for cycle in np.unique(cycles):
        mask = cycles == cycle
        percentile = 100.0 * rankdata(raw[mask], method="average") / int(mask.sum())
        mapped[mask] = 0.5 * raw[mask] + 0.5 * percentile
    mapped = np.clip(mapped, 0.0, 100.0)
    if not np.isfinite(mapped).all():
        raise ValueError("R23 rank50 mapping returned nonfinite values")
    return mapped


def reconstruct_r23(raw_prediction: Any, baseline_prediction: Any,
                    cycle_codes: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rebuild the candidate and mapped component from raw score arrays."""
    raw = np.asarray(raw_prediction, dtype=np.float64)
    baseline = np.asarray(baseline_prediction, dtype=np.float64)
    if raw.shape != baseline.shape or raw.ndim != 1 or raw.size == 0:
        raise ValueError("R23 raw and baseline predictions must be aligned vectors")
    if not np.isfinite(baseline).all() or np.any((baseline < 0.0) | (baseline > 100.0)):
        raise ValueError("R23 baseline predictions must be finite in [0,100]")
    mapped = rank50_prediction(raw, cycle_codes)
    candidate = np.clip(0.75 * baseline + 0.25 * mapped, 0.0, 100.0)
    return candidate, baseline.copy(), mapped


def text_svd32_support(fit_X: pd.DataFrame, score_X: pd.DataFrame | None = None) -> dict[str, Any]:
    """Check fit-only SVD32 vocabulary support and sparse TF-IDF buffer sizes."""
    fit = _select_exact(fit_X)
    fit_docs, _fit_missing = _text_documents(fit["signal_stream"])
    vectorizer, fit_matrix, fallback = RegressionTextModel._fit_vectorizer(fit_docs)
    width = int(fit_matrix.shape[1])
    fit_nnz = int(fit_matrix.nnz)
    fit_bytes = int(fit_matrix.data.nbytes + fit_matrix.indices.nbytes + fit_matrix.indptr.nbytes)
    score_nnz = 0
    score_bytes = 0
    score_rows = 0
    if score_X is not None:
        score = _select_exact(score_X)
        score_docs, _score_missing = _text_documents(score["signal_stream"])
        score_rows = len(score_docs)
        if vectorizer is not None:
            score_matrix = vectorizer.transform(score_docs).tocsr()
            score_nnz = int(score_matrix.nnz)
            score_bytes = int(score_matrix.data.nbytes + score_matrix.indices.nbytes + score_matrix.indptr.nbytes)
            del score_matrix
    del fit_matrix
    return {
        "fit_rows": int(len(fit)),
        "score_rows": int(score_rows),
        "tfidf_vocabulary_width": width,
        "fit_tfidf_nonzero_count": fit_nnz,
        "fit_tfidf_csr_bytes": fit_bytes,
        "score_tfidf_nonzero_count": score_nnz,
        "score_tfidf_csr_bytes": score_bytes,
        "required_minimum_tfidf_columns": 33,
        "exact_svd32_supported": width >= 33,
        "fallback": fallback,
        "fit_only_support_check": True,
        "vectorizer_fitted_for_support_check": vectorizer is not None,
    }


def _cycle_codes(features: pd.DataFrame, *, label: str) -> np.ndarray:
    values = pd.to_numeric(features["cycle_code"], errors="raise").to_numpy(dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValueError(f"{label} cycle_code must be finite integer values")
    cycles = values.astype(np.int64)
    if np.any((cycles < 47) | (cycles > 79)):
        raise ValueError(f"{label} cycle_code is outside the frozen Train range 47-79")
    return cycles


class ExtraTreesSVD32Regressor:
    """ExtraTrees candidate with fold-fit numeric/OHE and TF-IDF/SVD transforms."""

    def __init__(self, cutoff: int):
        if int(cutoff) not in PILOT_CUTOFFS:
            raise ValueError(f"R23 cutoff must be one of {PILOT_CUTOFFS}")
        self.cutoff = int(cutoff)
        self._fitted = False
        self._fit_warnings: list[dict[str, str]] = []
        self._state: dict[str, Any] = {}

    def fit(self, X: pd.DataFrame, y: Any) -> "ExtraTreesSVD32Regressor":
        self._fitted = False
        selected = _select_exact(X)
        target = _target(y, len(selected))
        cycles = _cycle_codes(selected, label="R23 fit features")
        if np.any(cycles > self.cutoff):
            raise ValueError("R23 fit features contain a cycle after the model cutoff")
        if not selected.index.is_unique or not selected.index.is_monotonic_increasing:
            raise ValueError("R23 fit rows must retain unique increasing source positions")

        docs, missing = _text_documents(selected["signal_stream"])
        self._vectorizer, text_matrix, self._text_fallback = RegressionTextModel._fit_vectorizer(docs)
        if text_matrix.shape[1] < 33:
            raise ValueError(
                f"R23 requires at least 33 fit-only TF-IDF columns for exact SVD32; got {text_matrix.shape[1]}"
            )
        self._svd_components_actual = 32
        self._svd = TruncatedSVD(n_components=32, **dict(_SVD_PARAMS))
        text_dense = self._svd.fit_transform(text_matrix)

        structured_sparse, self._processor, self._scaler, self._encoder, self._numeric_positions, structured_state = (
            RegressionTextModel._structured_fit(selected)
        )
        structured_dense = structured_sparse.toarray()
        design = np.column_stack((structured_dense, text_dense, missing.astype(np.float64)))
        if design.shape[0] != len(target) or not np.isfinite(design).all():
            raise RuntimeError("R23 dense SVD32 design is misaligned or nonfinite")

        self._estimator = ExtraTreesRegressor(**dict(EXTRA_TREES_PARAMETERS))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._estimator.fit(design, target)
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
        self._state = {
            "card_id": CARD_ID,
            "estimator": "sklearn.ensemble.ExtraTreesRegressor",
            "scikit_learn_version": sklearn.__version__,
            "parameters_explicit": dict(EXTRA_TREES_PARAMETERS),
            "parameters_effective": effective,
            "fit_rows": int(len(target)),
            "fit_cycle_min": int(cycles.min()),
            "fit_cycle_max": int(cycles.max()),
            "fit_source_rows_sha256": _array_digest(selected.index.to_numpy(dtype="<i8"), dtype="<i8"),
            "fit_data_sha256": _json_digest({
                "features": [str(x) for x in selected.columns],
                "row_positions": selected.index.to_numpy(dtype="<i8").tolist(),
                "target_sha256": _array_digest(target),
            }),
            "target_sha256": _array_digest(target),
            "feature_names": list(FEATURE_NAMES),
            "transformed_width": int(design.shape[1]),
            "transformed_feature_names": (
                list(self._processor.transformed_feature_names)
                + [f"signal_stream_svd_{i + 1}" for i in range(self._svd_components_actual)]
                + ["signal_stream_missing"]
            ),
            "training_design_shape": [int(design.shape[0]), int(design.shape[1])],
            "training_design_sha256": _array_digest(design),
            "structured_preprocessing": structured_state,
            "structured_dense_conversion": "fold-fit numeric/OHE sparse matrix converted to float64 dense",
            "tfidf": RegressionTextModel._text_state(self._vectorizer, self._text_fallback),
            "svd": {
                **dict(_SVD_PARAMS),
                "n_components_requested": 32,
                "n_components_actual": int(self._svd_components_actual),
                "components_sha256": _array_digest(self._svd.components_) if self._svd is not None else None,
            },
            "fit_missing_stream_rows": int(missing.sum()),
            "validation_data_passed_to_fit": False,
            "validation_labels_passed_to_fit": False,
            "test_data_used_for_fit": False,
            "early_stopping_used": False,
            "fit_warnings": list(self._fit_warnings),
        }
        self._state["state_sha256"] = _json_digest(self._state)
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("R23 model must be fitted before predict")
        selected = _select_exact(X)
        structured_sparse, _ = RegressionTextModel._structured_transform(
            selected, self._processor, self._scaler, self._encoder, self._numeric_positions
        )
        structured_dense = structured_sparse.toarray()
        docs, missing = _text_documents(selected["signal_stream"])
        if self._vectorizer is None:
            from scipy import sparse
            text_matrix = sparse.csr_matrix((len(docs), 0), dtype=np.float64)
        else:
            text_matrix = self._vectorizer.transform(docs).tocsr()
        text_dense = self._svd.transform(text_matrix) if self._svd is not None else np.empty((len(docs), 0))
        design = np.column_stack((structured_dense, text_dense, missing.astype(np.float64)))
        if design.shape[1] != int(self._state["transformed_width"]) or not np.isfinite(design).all():
            raise RuntimeError("R23 score design is nonfinite or has a different schema")
        prediction = np.asarray(self._estimator.predict(design), dtype=np.float64)
        if prediction.shape != (len(selected),) or not np.isfinite(prediction).all():
            raise RuntimeError("R23 ExtraTrees predictions are misaligned or nonfinite")
        return prediction

    def audit_state(self) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("R23 model must be fitted before audit_state")
        return dict(self._state)


def _validated_controls(controls: dict[str, Any], n_score: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    missing = [name for name in _CONTROL_KEYS if name not in controls]
    if missing or n_score <= 0:
        raise ValueError(f"R23 controls are missing fields or empty: {missing}")
    arrays: dict[str, np.ndarray] = {}
    for name in _CONTROL_KEYS[:-1]:
        values = np.asarray(controls[name], dtype=np.float64)
        if values.shape != (n_score,) or not np.isfinite(values).all():
            raise ValueError(f"R23 control {name} is misaligned or nonfinite")
        if name != "baseline_sparse_raw" and np.any((values < 0.0) | (values > 100.0)):
            raise ValueError(f"R23 control {name} is outside [0,100]")
        arrays[name] = values
    rows = np.asarray(controls["control_rows"])
    if (rows.ndim != 1 or rows.shape != (n_score,) or rows.dtype.kind not in "iu"
            or len(np.unique(rows)) != n_score or np.any(np.diff(rows.astype(np.int64)) <= 0)):
        raise ValueError("R23 control rows must be unique increasing aligned positions")
    rebuilt = np.clip(
        0.75 * (0.75 * arrays["baseline_lgb_svd32"]
                + 0.25 * arrays["baseline_hybrid_ridge"])
        + 0.25 * arrays["baseline_sparse_rank50"], 0.0, 100.0
    )
    if not np.allclose(rebuilt, arrays["baseline_prediction"], rtol=0.0, atol=1e-10):
        raise ValueError("R23 frozen three-component champion does not reconstruct from exact controls")
    if np.any(~np.isfinite(arrays["baseline_sparse_raw"])):
        raise ValueError("R23 sparse raw control is nonfinite")
    return arrays, rows.astype(np.int64, copy=False)


def fit_predict(fit_features: pd.DataFrame, fit_target: np.ndarray,
                score_features: pd.DataFrame, *, controls: dict[str, Any],
                card_id: str, cutoff: int) -> dict[str, Any]:
    """Fit one temporal fold and apply the fixed R_AUX25 policy."""
    if card_id != CARD_ID or int(cutoff) not in PILOT_CUTOFFS:
        raise ValueError("R23 adapter received a different card or cutoff")
    fit_x = _select_exact(fit_features)
    score_x = _select_exact(score_features)
    if (not fit_x.index.is_unique or not fit_x.index.is_monotonic_increasing
            or not score_x.index.is_unique or not score_x.index.is_monotonic_increasing):
        raise ValueError("R23 fit and score rows must preserve increasing source row positions")
    target = _target(fit_target, len(fit_x))
    fit_cycles = _cycle_codes(fit_x, label="R23 fit features")
    score_cycles = _cycle_codes(score_x, label="R23 score features")
    if np.any(fit_cycles > int(cutoff)) or np.any((score_cycles <= int(cutoff)) | (score_cycles > int(cutoff) + 12)):
        raise ValueError("R23 row cycles violate the frozen temporal horizon")
    control, control_rows = _validated_controls(controls, len(score_x))
    if not np.array_equal(score_x.index.to_numpy(dtype=np.int64), control_rows):
        raise ValueError("R23 controls are not aligned to the score row positions")

    model = ExtraTreesSVD32Regressor(cutoff=int(cutoff)).fit(fit_x, target)
    raw = model.predict(score_x)
    prediction, baseline, mapped = reconstruct_r23(raw, control["baseline_prediction"], score_cycles)
    return {
        "prediction": prediction,
        "baseline_prediction": baseline,
        "raw_candidate": raw,
        "fit_predictor_count": 1,
        "state": model.audit_state(),
        "fit_metadata": {
            "cutoff": int(cutoff),
            "fit_rows": int(len(fit_x)),
            "fit_cycle_min": int(fit_cycles.min()),
            "fit_cycle_max": int(fit_cycles.max()),
            "feature_names": list(FEATURE_NAMES),
            "policy": ".75 current three-component champion + .25 within-cycle average-rank rank50(ExtraTrees raw prediction), clipped [0,100]",
            "control_rows_verified_unique_increasing": True,
            "score_labels_passed_to_adapter": False,
            "fit_warnings": list(model._fit_warnings),
        },
        "audit_arrays": {"score_cycles": score_cycles, "candidate_rank50": mapped,
                         "control_rows": control_rows},
    }


__all__ = [
    "CARD_ID", "FEATURE_NAMES", "PILOT_CUTOFFS", "EXTRA_TREES_PARAMETERS",
    "ExtraTreesSVD32Regressor", "fit_predict", "rank50_prediction", "reconstruct_r23",
    "text_svd32_support",
]
