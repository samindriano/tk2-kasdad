"""R24: hybrid Ridge with a reference-coded entry_friction text interaction."""
from __future__ import annotations

import hashlib
import json
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata
from sklearn.linear_model import Ridge

from src.models.regression_research_text import (
    HYBRID_FEATURES,
    RegressionTextModel,
    _RIDGE_PARAMS,
    _array_digest,
    _json_digest,
    _target,
    _text_documents,
)
from src.models.round1_models import _fit_fingerprint


CARD_ID = "R24"
FEATURE_NAMES = tuple(HYBRID_FEATURES)
PILOT_CUTOFFS = (55, 67)
RIDGE_PARAMETERS = dict(_RIDGE_PARAMS)
RANK_WEIGHT = 0.75
_CONTROL_KEYS = (
    "baseline_prediction", "baseline_lgb_svd32", "baseline_hybrid_ridge",
    "baseline_sparse_raw", "baseline_sparse_rank50", "control_rows",
)


def _select_exact(frame: pd.DataFrame) -> pd.DataFrame:
    if (not isinstance(frame, pd.DataFrame) or frame.columns.has_duplicates
            or set(frame.columns) != set(FEATURE_NAMES) or len(frame.columns) != len(FEATURE_NAMES)):
        raise ValueError("R24 requires exactly the frozen hybrid structured + signal_stream feature allowlist")
    if set(FEATURE_NAMES) & {"record_id", "artifact_signal", "threat_rank"}:
        raise ValueError("R24 feature allowlist contains an identity or target")
    return frame.loc[:, list(FEATURE_NAMES)].copy()


def _entry_friction_indicator(frame: pd.DataFrame) -> np.ndarray:
    values = pd.to_numeric(frame["entry_friction"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or not np.isin(values, [0.0, 1.0]).all():
        raise ValueError("R24 entry_friction must contain only finite parsed binary values 0/1")
    return (values == 1.0).astype(np.float64)


def rank75_prediction(raw_prediction: Any, cycle_codes: Any) -> np.ndarray:
    """Map raw target-scale predictions with 75% within-cycle average-rank weight."""
    raw = np.asarray(raw_prediction, dtype=np.float64)
    cycles = np.asarray(cycle_codes)
    if raw.ndim != 1 or raw.size == 0 or cycles.shape != raw.shape:
        raise ValueError("R24 rank75 values and cycle codes must be nonempty aligned vectors")
    if not np.isfinite(raw).all() or cycles.dtype.kind not in "iu":
        raise ValueError("R24 rank75 requires finite predictions and integer cycle codes")
    mapped = np.empty(raw.shape, dtype=np.float64)
    for cycle in np.unique(cycles):
        mask = cycles == cycle
        percentile = 100.0 * rankdata(raw[mask], method="average") / int(mask.sum())
        mapped[mask] = (1.0 - RANK_WEIGHT) * raw[mask] + RANK_WEIGHT * percentile
    mapped = np.clip(mapped, 0.0, 100.0)
    if not np.isfinite(mapped).all():
        raise ValueError("R24 rank75 mapping returned nonfinite predictions")
    return mapped


def reconstruct_r24(raw_prediction: Any, cycles: Any, lgb_rank50: Any,
                    sparse_rank50: Any) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild the incumbent and R24 Ridge-component replacement policy."""
    raw = np.asarray(raw_prediction, dtype=np.float64)
    cycle_values = np.asarray(cycles, dtype=np.int64)
    lgb = np.asarray(lgb_rank50, dtype=np.float64)
    sparse_rank = np.asarray(sparse_rank50, dtype=np.float64)
    if len({values.shape for values in (raw, cycle_values, lgb, sparse_rank)}) != 1:
        raise ValueError("R24 policy arrays must have identical shapes")
    if (raw.ndim != 1 or not len(raw) or not np.isfinite(raw).all()
            or not np.isfinite(lgb).all() or not np.isfinite(sparse_rank).all()
            or np.any((lgb < 0) | (lgb > 100)) or np.any((sparse_rank < 0) | (sparse_rank > 100))):
        raise ValueError("R24 policy arrays contain invalid values")
    mapped = rank75_prediction(raw, cycle_values)
    candidate = np.clip(0.75 * (0.75 * lgb + 0.25 * mapped) + 0.25 * sparse_rank, 0.0, 100.0)
    return candidate, mapped


def _ridge_stationarity_ratio(design: sparse.spmatrix, target: np.ndarray,
                              coefficients: np.ndarray, intercept: float, alpha: float) -> float:
    residual = np.asarray(design @ coefficients, dtype=np.float64).reshape(-1) + intercept - target
    gradient = np.asarray(design.T @ residual, dtype=np.float64).reshape(-1) + alpha * coefficients
    reference = np.asarray(design.T @ target, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(reference) + alpha * np.linalg.norm(coefficients))
    return float(np.linalg.norm(gradient) / max(denominator, np.finfo(np.float64).eps))


class EFInteractionHybridRidge:
    """Fit the inherited hybrid Ridge matrix plus TF-IDF x I(entry_friction == 1)."""

    input_feature_names = FEATURE_NAMES

    def __init__(self, cutoff: int):
        if int(cutoff) not in PILOT_CUTOFFS:
            raise ValueError(f"R24 cutoff must be one of {PILOT_CUTOFFS}")
        self.cutoff = int(cutoff)
        self._fitted = False
        self._warnings: list[dict[str, str]] = []

    def fit(self, frame: pd.DataFrame, target: Any) -> "EFInteractionHybridRidge":
        self._fitted = False
        selected = _select_exact(frame)
        y = _target(target, len(selected))
        cycles = pd.to_numeric(selected["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
        if (cycles.ndim != 1 or not len(cycles) or np.any((cycles < 47) | (cycles > self.cutoff))
                or not selected.index.is_unique or not selected.index.is_monotonic_increasing):
            raise ValueError("R24 fit rows must be unique ordered Train rows through the cutoff")

        ef = _entry_friction_indicator(selected)
        docs, missing = _text_documents(selected["signal_stream"])
        self._vectorizer, text, self._text_fallback = RegressionTextModel._fit_vectorizer(docs)
        (structured, self._processor, self._scaler, self._encoder,
         self._numeric_positions, structured_state) = RegressionTextModel._structured_fit(selected)
        missing_column = sparse.csr_matrix(missing.astype(np.float64).reshape(-1, 1))
        interaction = text.multiply(ef.reshape(-1, 1)).tocsr()
        design = sparse.hstack(
            (structured, text, missing_column, interaction), format="csr", dtype=np.float64
        )
        if design.shape[0] != len(y) or not np.isfinite(design.data).all():
            raise RuntimeError("R24 sparse fit design is misaligned or nonfinite")

        self._estimator = Ridge(**dict(RIDGE_PARAMETERS))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._estimator.fit(design, y)
        self._warnings = [{"category": item.category.__name__, "message": str(item.message)}
                          for item in caught]
        coefficients = np.asarray(self._estimator.coef_, dtype=np.float64).reshape(-1)
        if not np.isfinite(coefficients).all() or not np.isfinite(float(self._estimator.intercept_)):
            raise RuntimeError("R24 Ridge fitted nonfinite parameters")
        iterations = getattr(self._estimator, "n_iter_", None)
        if iterations is None:
            iteration_values = None
        else:
            iteration_values = np.asarray(iterations, dtype=np.float64).reshape(-1)
            if not np.isfinite(iteration_values).all():
                raise RuntimeError("R24 Ridge LSQR returned nonfinite iteration counts")
            iteration_values = [int(value) for value in iteration_values]
        stationarity = _ridge_stationarity_ratio(
            design, y, coefficients, float(self._estimator.intercept_),
            float(RIDGE_PARAMETERS["alpha"]),
        )
        if not np.isfinite(stationarity):
            raise RuntimeError("R24 Ridge LSQR stationarity diagnostic is nonfinite")

        base_width = int(structured.shape[1] + text.shape[1] + 1)
        self._state = {
            "card_id": CARD_ID,
            "estimator": "sklearn.linear_model.Ridge",
            "parameters": dict(RIDGE_PARAMETERS),
            "fit_rows": int(len(y)),
            "cutoff": self.cutoff,
            "fit_cycle_min": int(cycles.min()),
            "fit_cycle_max": int(cycles.max()),
            "fit_cycle_counts": {str(int(c)): int(n) for c, n in zip(*np.unique(cycles, return_counts=True))},
            "fit_row_positions_sha256": hashlib.sha256(
                selected.index.to_numpy(dtype="<i8").tobytes(order="C")
            ).hexdigest(),
            "fit_data_sha256": _fit_fingerprint(selected, FEATURE_NAMES, y),
            "target_sha256": _array_digest(y),
            "feature_names": list(FEATURE_NAMES),
            "structured_preprocessing": structured_state,
            "text": RegressionTextModel._text_state(self._vectorizer, self._text_fallback),
            "interaction": {
                "definition": "TF-IDF block multiplied row-wise by I(parsed entry_friction == 1)",
                "reference_group": "entry_friction == 0",
                "fit_entry_friction_1_rows": int(np.sum(ef == 1.0)),
                "fit_entry_friction_0_rows": int(np.sum(ef == 0.0)),
                "fit_interaction_nonzero": int(interaction.nnz),
                "appended_after_inherited_hybrid_design": True,
                "telemetry_substitution": False,
            },
            "design_shape": [int(value) for value in design.shape],
            "inherited_hybrid_width": base_width,
            "interaction_width": int(text.shape[1]),
            "design_nnz": int(design.nnz),
            "design_data_sha256": _array_digest(design.data),
            "coefficient_sha256": _array_digest(coefficients),
            "intercept": float(self._estimator.intercept_),
            "solver_diagnostics": {
                "solver": "lsqr", "n_iter": iteration_values,
                "stationarity_ratio": stationarity,
                "stationarity_tolerance_reference": float(RIDGE_PARAMETERS["tol"]),
                "warnings": list(self._warnings),
            },
            "validation_data_passed_to_fit": False,
            "validation_labels_passed_to_fit": False,
            "test_data_used_for_fit": False,
            "early_stopping_used": False,
            "sample_weight_used": False,
        }
        self._state["state_sha256"] = _json_digest(self._state)
        self._prediction_diagnostics = {"n_predictions": 0}
        self._fitted = True
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("R24 model must be fitted before prediction")
        selected = _select_exact(frame)
        ef = _entry_friction_indicator(selected)
        docs, missing = _text_documents(selected["signal_stream"])
        text = (sparse.csr_matrix((len(docs), 0), dtype=np.float64) if self._vectorizer is None
                else self._vectorizer.transform(docs).tocsr())
        structured, unknown = RegressionTextModel._structured_transform(
            selected, self._processor, self._scaler, self._encoder, self._numeric_positions
        )
        missing_column = sparse.csr_matrix(missing.astype(np.float64).reshape(-1, 1))
        interaction = text.multiply(ef.reshape(-1, 1)).tocsr()
        design = sparse.hstack(
            (structured, text, missing_column, interaction), format="csr", dtype=np.float64
        )
        if design.shape[1] != int(self._state["design_shape"][1]) or not np.isfinite(design.data).all():
            raise RuntimeError("R24 score design changed schema or contains nonfinite values")
        prediction = np.asarray(self._estimator.predict(design), dtype=np.float64)
        if prediction.shape != (len(selected),) or not np.isfinite(prediction).all():
            raise RuntimeError("R24 Ridge predictions are misaligned or nonfinite")
        self._prediction_diagnostics = {
            "n_predictions": int(len(prediction)),
            "score_entry_friction_1_rows": int(np.sum(ef == 1.0)),
            "unknown_structured_categories": unknown,
        }
        return prediction

    def audit_state(self) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("R24 model must be fitted before audit_state")
        return json.loads(json.dumps(self._state, sort_keys=True, allow_nan=False))


def _validated_controls(controls: dict[str, Any], n_score: int,
                        score_cycles: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    absent = [name for name in _CONTROL_KEYS if name not in controls]
    if absent or n_score <= 0:
        raise ValueError(f"R24 fixed controls are missing fields or empty: {absent}")
    arrays: dict[str, np.ndarray] = {}
    for name in _CONTROL_KEYS[:-1]:
        values = np.asarray(controls[name], dtype=np.float64)
        if values.shape != (n_score,) or not np.isfinite(values).all():
            raise ValueError(f"R24 control {name} is misaligned or nonfinite")
        if name != "baseline_sparse_raw" and np.any((values < 0.0) | (values > 100.0)):
            raise ValueError(f"R24 control {name} is outside [0,100]")
        arrays[name] = values
    rows = np.asarray(controls["control_rows"])
    if (rows.shape != (n_score,) or rows.ndim != 1 or rows.dtype.kind not in "iu"
            or len(np.unique(rows)) != n_score or np.any(np.diff(rows.astype(np.int64)) <= 0)):
        raise ValueError("R24 control rows must be unique increasing Train positions")
    sparse_rank = np.empty(n_score, dtype=np.float64)
    for cycle in np.unique(score_cycles):
        mask = score_cycles == cycle
        sparse_rank[mask] = (
            0.5 * arrays["baseline_sparse_raw"][mask]
            + 0.5 * 100.0 * rankdata(arrays["baseline_sparse_raw"][mask], method="average") / int(mask.sum())
        )
    sparse_rank = np.clip(sparse_rank, 0.0, 100.0)
    if not np.array_equal(sparse_rank, arrays["baseline_sparse_rank50"]):
        raise ValueError("R24 frozen sparse rank50 control does not reconstruct")
    rebuilt_baseline = np.clip(
        0.75 * (0.75 * arrays["baseline_lgb_svd32"]
                + 0.25 * arrays["baseline_hybrid_ridge"])
        + 0.25 * sparse_rank,
        0.0, 100.0,
    )
    if not np.array_equal(rebuilt_baseline, arrays["baseline_prediction"]):
        raise ValueError("R24 three-component baseline does not reconstruct exactly")
    return arrays, rows.astype(np.int64, copy=False)


def fit_predict(fit_features: pd.DataFrame, fit_target: np.ndarray,
                score_features: pd.DataFrame, *, controls: dict[str, Any],
                card_id: str, cutoff: int) -> dict[str, Any]:
    if card_id != CARD_ID or int(cutoff) not in PILOT_CUTOFFS:
        raise ValueError("R24 adapter accepts only its declared card and cutoffs 55/67")
    fit_x = _select_exact(fit_features)
    score_x = _select_exact(score_features)
    if (not fit_x.index.is_unique or not fit_x.index.is_monotonic_increasing
            or not score_x.index.is_unique or not score_x.index.is_monotonic_increasing):
        raise ValueError("R24 fit and score rows must preserve increasing Train source positions")
    fit_cycles = pd.to_numeric(fit_x["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    score_cycles = pd.to_numeric(score_x["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    if (np.any(fit_cycles > int(cutoff))
            or np.any((score_cycles <= int(cutoff)) | (score_cycles > int(cutoff) + 12))):
        raise ValueError("R24 row cycles violate the frozen temporal horizon")
    arrays, control_rows = _validated_controls(controls, len(score_x), score_cycles)
    if not np.array_equal(score_x.index.to_numpy(dtype=np.int64), control_rows):
        raise ValueError("R24 controls do not align with score source rows")

    model = EFInteractionHybridRidge(int(cutoff)).fit(fit_x, fit_target)
    raw = model.predict(score_x)
    candidate, mapped = reconstruct_r24(
        raw, score_cycles, arrays["baseline_lgb_svd32"], arrays["baseline_sparse_rank50"]
    )
    baseline = arrays["baseline_prediction"].copy()
    if (not np.isfinite(candidate).all() or np.any((candidate < 0.0) | (candidate > 100.0))
            or not np.array_equal(baseline, arrays["baseline_prediction"])):
        raise RuntimeError("R24 replacement policy returned invalid predictions")
    return {
        "prediction": candidate,
        "baseline_prediction": baseline,
        "raw_candidate": raw,
        "fit_predictor_count": 1,
        "state": model.audit_state(),
        "fit_metadata": {
            **model._prediction_diagnostics,
            "cutoff": int(cutoff), "fit_rows": int(len(fit_x)),
            "fit_cycle_min": int(fit_cycles.min()), "fit_cycle_max": int(fit_cycles.max()),
            "policy": ".5625 frozen mapped LGB + .1875 rank75(new Hybrid Ridge) + .25 frozen mapped sparse LGB",
            "score_labels_passed_to_adapter": False,
            "solver_diagnostics": model.audit_state()["solver_diagnostics"],
        },
        "audit_arrays": {"score_cycles": score_cycles, "candidate_rank75": mapped},
    }


__all__ = [
    "CARD_ID", "EFInteractionHybridRidge", "FEATURE_NAMES", "PILOT_CUTOFFS",
    "RIDGE_PARAMETERS", "RANK_WEIGHT", "fit_predict", "rank75_prediction", "reconstruct_r24",
]
