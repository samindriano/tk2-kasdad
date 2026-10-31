"""R16: exponentially weighted all-past SVD32 LightGBM component.

This adapter refits only the frozen SVD32 LightGBM component. Structured
preprocessing, TF-IDF, and SVD remain fit-only and use every row through the
origin; exponential weights are passed only to the LightGBM estimator.
"""
from __future__ import annotations

import warnings
import hashlib
from typing import Any

import lightgbm
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import rankdata
from sklearn.decomposition import TruncatedSVD

from src.models.regression_research_lgb_text import (
    RegressionResearchLGBText,
    _select_exact,
)
from src.models.regression_research_tabular import _tree_summary
from src.models.regression_research_text import (
    HYBRID_FEATURES,
    STRUCTURED_FEATURES,
    RegressionTextModel,
    _SVD_PARAMS,
    _array_digest,
    _json_digest,
    _target,
    _text_documents,
)
from src.models.regression_round2_models import _SelectedPreprocessor
from src.models.round1_models import _fit_fingerprint


FEATURE_NAMES = HYBRID_FEATURES
WEIGHT_FORMULA = "raw_i=2**(-(cutoff-cycle_code_i)/8); normalized_i=raw_i/mean(raw)"
_PILOT_CUTOFFS = (55, 67)
_COMPONENT_KEYS = (
    "baseline_prediction",
    "old_lgb_svd32",
    "ridge_prediction",
    "sparse_rank50",
    "control_rows",
)


def _cycle_codes(features: pd.DataFrame, *, label: str) -> np.ndarray:
    if "cycle_code" not in features.columns:
        raise ValueError(f"{label} is missing required cycle_code")
    values = pd.to_numeric(features["cycle_code"], errors="raise").to_numpy(dtype=np.float64)
    if (values.ndim != 1 or not np.isfinite(values).all()
            or not np.equal(values, np.floor(values)).all()):
        raise ValueError(f"{label} cycle_code must be finite integer values")
    cycles = values.astype(np.int64)
    if np.any((cycles < 47) | (cycles > 79)):
        raise ValueError(f"{label} cycle_code falls outside the frozen Train cycle range 47-79")
    return cycles


def normalized_recency_weights(cycle_code: Any, cutoff: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Return deterministic mean-one exponential weights and an audit summary."""
    cutoff = int(cutoff)
    if cutoff not in _PILOT_CUTOFFS:
        raise ValueError(f"R16 cutoff must be one of {_PILOT_CUTOFFS}")
    values = pd.to_numeric(pd.Series(cycle_code), errors="raise").to_numpy(dtype=np.float64)
    if (values.ndim != 1 or values.size == 0 or not np.isfinite(values).all()
            or not np.equal(values, np.floor(values)).all()):
        raise ValueError("R16 cycle_code must be a nonempty finite integer vector")
    cycles = values.astype(np.int64)
    if np.any((cycles < 47) | (cycles > cutoff)):
        raise ValueError("R16 fit cycles must be within 47..cutoff")
    ages = cutoff - cycles
    raw_weights = np.exp2(-ages.astype(np.float64) / 8.0)
    raw_mean = float(np.mean(raw_weights))
    if not np.isfinite(raw_mean) or raw_mean <= 0.0:
        raise ValueError("R16 raw sample weights have an invalid mean")
    weights = raw_weights / raw_mean
    mean = float(np.mean(weights))
    if (not np.isfinite(weights).all() or np.any(weights <= 0.0)
            or not np.isclose(mean, 1.0, rtol=0.0, atol=1e-12)):
        raise ValueError("R16 normalized sample weights are invalid")
    cycles_bytes = np.asarray(cycles, dtype="<i8").tobytes(order="C")
    summary = {
        "formula": WEIGHT_FORMULA,
        "cutoff": cutoff,
        "fit_rows": int(len(cycles)),
        "fit_cycle_min": int(cycles.min()),
        "fit_cycle_max": int(cycles.max()),
        "raw_weight_mean": raw_mean,
        "normalization": "divide each raw weight by the arithmetic mean over all fit rows",
        "mean": mean,
        "min": float(weights.min()),
        "max": float(weights.max()),
        "sha256": _array_digest(weights),
        "source_fit_cycles_sha256": hashlib.sha256(cycles_bytes).hexdigest(),
        "source_fit_cycle_counts": {
            str(int(cycle)): int(np.sum(cycles == cycle)) for cycle in np.unique(cycles)
        },
    }
    return weights, summary


def rank50_prediction(raw_prediction: Any, cycle_codes: Any) -> np.ndarray:
    """Apply the fixed within-cycle raw/percentile rank50 map."""
    raw = np.asarray(raw_prediction, dtype=np.float64)
    cycles = np.asarray(cycle_codes)
    if raw.ndim != 1 or cycles.ndim != 1 or cycles.shape != raw.shape or not len(raw):
        raise ValueError("R16 raw scores and cycle codes must be nonempty and aligned")
    if not np.isfinite(raw).all() or cycles.dtype.kind not in "iu":
        raise ValueError("R16 raw scores must be finite and cycle codes must be integer")
    mapped = np.empty(len(raw), dtype=np.float64)
    for cycle in np.unique(cycles):
        mask = cycles == cycle
        percentile = 100.0 * rankdata(raw[mask], method="average") / int(mask.sum())
        mapped[mask] = 0.5 * raw[mask] + 0.5 * percentile
    mapped = np.clip(mapped, 0.0, 100.0)
    if not np.isfinite(mapped).all():
        raise ValueError("R16 rank50 mapping produced nonfinite values")
    return mapped


def replace_svd_component(
    new_rank50_lgb: Any,
    frozen_mapped_ridge: Any,
    frozen_rank50_sparse: Any,
) -> np.ndarray:
    """Replace only the incumbent .5625 SVD-LGB component in its frozen route."""
    arrays = [
        np.asarray(new_rank50_lgb, dtype=np.float64),
        np.asarray(frozen_mapped_ridge, dtype=np.float64),
        np.asarray(frozen_rank50_sparse, dtype=np.float64),
    ]
    if (any(value.ndim != 1 for value in arrays)
            or len({value.shape for value in arrays}) != 1
            or not len(arrays[0])
            or any(not np.isfinite(value).all() for value in arrays)):
        raise ValueError("R16 replacement components must be finite, nonempty, aligned vectors")
    if any(np.any((value < 0.0) | (value > 100.0)) for value in arrays):
        raise ValueError("R16 replacement components must be in [0,100]")
    return np.clip(0.5625 * arrays[0] + 0.1875 * arrays[1] + 0.25 * arrays[2], 0.0, 100.0)


class ExponentiallyWeightedSVDLGB(RegressionResearchLGBText):
    """Inherited SVD32 LightGBM recipe with origin-relative fit weights."""

    def __init__(self, cutoff: int):
        if int(cutoff) not in _PILOT_CUTOFFS:
            raise ValueError(f"R16 cutoff must be one of {_PILOT_CUTOFFS}")
        super().__init__(params_id="base")
        self.cutoff = int(cutoff)
        self._weight_audit: dict[str, Any] = {}
        self._fit_warnings: list[dict[str, str]] = []

    def fit(self, X: pd.DataFrame, y: Any) -> "ExponentiallyWeightedSVDLGB":
        self._fitted = False
        selected = _select_exact(X)
        target = _target(y, len(selected))
        cycles = _cycle_codes(selected, label="R16 fit features")
        if np.any(cycles > self.cutoff):
            raise ValueError("R16 fit features contain a cycle after the model cutoff")

        sample_weight, self._weight_audit = normalized_recency_weights(cycles, self.cutoff)
        self._weight_audit["applied_to"] = (
            "LightGBM estimator only; TF-IDF, structured preprocessing and SVD are fit on all fit rows unweighted"
        )

        self._fit_rows = int(len(target))
        self._fit_fingerprint = _fit_fingerprint(selected, self.input_feature_names, target)
        self._target_sha256 = _array_digest(target)

        self._preprocessor = _SelectedPreprocessor(tuple(STRUCTURED_FEATURES))
        structured, categorical_mask = self._preprocessor.fit_transform(
            selected.loc[:, list(STRUCTURED_FEATURES)]
        )
        categorical_mask = list(categorical_mask)
        self._categorical_positions = [
            i for i, is_cat in enumerate(categorical_mask) if is_cat
        ]

        docs, missing = _text_documents(selected["signal_stream"])
        self._vectorizer, text_matrix, fallback = RegressionTextModel._fit_vectorizer(docs)
        self._svd_components_actual = (
            min(32, text_matrix.shape[1] - 1) if text_matrix.shape[1] > 1 else 0
        )
        if self._svd_components_actual:
            self._svd = TruncatedSVD(
                n_components=self._svd_components_actual, **dict(_SVD_PARAMS)
            )
            text_dense = self._svd.fit_transform(text_matrix)
        else:
            text_dense = np.empty((len(selected), 0), dtype=np.float64)
        design = np.column_stack((structured, text_dense, missing.astype(np.float64)))
        if design.shape[0] != len(target) or not np.isfinite(design).all():
            raise RuntimeError("R16 SVD32 design is misaligned or nonfinite")
        self._transformed_feature_names = (
            list(self._preprocessor.transformed_feature_names)
            + [f"signal_stream_svd_{i + 1}" for i in range(self._svd_components_actual)]
            + ["signal_stream_missing"]
        )
        if len(self._transformed_feature_names) != design.shape[1]:
            raise RuntimeError("R16 SVD32 feature-name schema does not match the design")
        design_frame = pd.DataFrame(design, columns=self._transformed_feature_names)

        self._estimator = LGBMRegressor(**dict(self.actual_parameters))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._estimator.fit(
                design_frame,
                target,
                sample_weight=sample_weight,
                categorical_feature=self._categorical_positions,
            )
        self._fit_warnings = [
            {"category": item.category.__name__, "message": str(item.message)}
            for item in caught
        ]

        self._tree_audit = _tree_summary(self._estimator.booster_)
        self._training_design_sha256 = _array_digest(design)
        self._training_design_shape = [int(design.shape[0]), int(design.shape[1])]
        self._fit_missing_stream_rows = int(missing.sum())
        self._vectorizer_state = RegressionTextModel._text_state(self._vectorizer, fallback)
        self._structured_mask = categorical_mask
        state = self._make_state()
        state.pop("state_sha256", None)
        state["cutoff"] = self.cutoff
        state["sample_weighting"] = dict(self._weight_audit)
        state["source_fit_cycles"] = cycles.astype(np.int64).tolist()
        state["source_fit_cycle_max"] = int(cycles.max())
        state["fit_warnings"] = list(self._fit_warnings)
        state["state_sha256"] = _json_digest(state)
        self._state = state
        self._fitted = True
        self._diagnostics = {"n_predictions": 0}
        return self


def _validated_controls(controls: dict[str, Any], n_score: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    if n_score <= 0:
        raise ValueError("R16 requires at least one scored row")
    missing = [name for name in _COMPONENT_KEYS if name not in controls]
    if missing:
        raise ValueError(f"R16 control bridge is missing required keys: {missing}")
    arrays: dict[str, np.ndarray] = {}
    for name in _COMPONENT_KEYS[:-1]:
        values = np.asarray(controls[name], dtype=np.float64)
        if values.shape != (n_score,) or not np.isfinite(values).all():
            raise ValueError(f"R16 control component {name} is misaligned or nonfinite")
        if np.any((values < 0.0) | (values > 100.0)):
            raise ValueError(f"R16 control component {name} is outside [0,100]")
        arrays[name] = values

    rows = np.asarray(controls["control_rows"])
    if (rows.ndim != 1 or rows.shape != (n_score,) or rows.dtype.kind not in "iu"
            or len(np.unique(rows)) != n_score or np.any(np.diff(rows.astype(np.int64)) <= 0)):
        raise ValueError("R16 control_rows must be unique, increasing row positions aligned to scores")
    rows = rows.astype(np.int64, copy=False)

    reconstructed = np.clip(
        0.5625 * arrays["old_lgb_svd32"]
        + 0.1875 * arrays["ridge_prediction"]
        + 0.25 * arrays["sparse_rank50"],
        0.0,
        100.0,
    )
    if not np.allclose(
        reconstructed, arrays["baseline_prediction"], rtol=0.0, atol=1e-10
    ):
        raise ValueError("R16 control components do not reconstruct the frozen champion")
    return arrays, rows


def fit_predict(
    fit_features: pd.DataFrame,
    fit_target: np.ndarray,
    score_features: pd.DataFrame,
    *,
    controls: dict[str, np.ndarray],
    card_id: str,
    cutoff: int,
) -> dict[str, Any]:
    """Fit one weighted temporal fold and replace only its mapped SVD-LGB part."""
    if card_id != "R16":
        raise ValueError(f"R16 adapter cannot execute {card_id}")
    cutoff = int(cutoff)
    if cutoff not in _PILOT_CUTOFFS:
        raise ValueError(f"R16 pilot cutoff must be one of {_PILOT_CUTOFFS}")

    fit_x = _select_exact(fit_features)
    score_x = _select_exact(score_features)
    if (not fit_x.index.is_unique or not fit_x.index.is_monotonic_increasing
            or not score_x.index.is_unique or not score_x.index.is_monotonic_increasing):
        raise ValueError("R16 feature rows must retain unique, increasing source row positions")
    target = _target(fit_target, len(fit_x))
    fit_cycles = _cycle_codes(fit_x, label="R16 fit features")
    if np.any(fit_cycles > cutoff):
        raise ValueError("R16 fit features contain future cycles")
    score_cycles = _cycle_codes(score_x, label="R16 score features")
    if np.any((score_cycles <= cutoff) | (score_cycles > cutoff + 12)):
        raise ValueError("R16 score features fall outside the frozen 12-cycle validation horizon")

    control_arrays, control_rows = _validated_controls(controls, len(score_x))
    try:
        score_positions = score_x.index.to_numpy(dtype=np.int64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("R16 score feature index must contain source row positions") from exc
    if not np.array_equal(score_positions, control_rows):
        raise ValueError("R16 control rows do not match score feature row positions")
    model = ExponentiallyWeightedSVDLGB(cutoff=cutoff)
    model.fit(fit_x, target)
    raw = np.asarray(model.predict(score_x), dtype=np.float64)
    if raw.shape != (len(score_x),) or not np.isfinite(raw).all():
        raise ValueError("R16 SVD32 LightGBM returned misaligned or nonfinite predictions")

    mapped = rank50_prediction(raw, score_cycles)
    candidate = replace_svd_component(
        mapped,
        control_arrays["ridge_prediction"],
        control_arrays["sparse_rank50"],
    )
    if candidate.shape != (len(score_x),) or not np.isfinite(candidate).all():
        raise ValueError("R16 replacement route returned misaligned or nonfinite predictions")

    state = model.audit_state()
    fit_metadata = {
        "cutoff": cutoff,
        "fit_rows": int(len(fit_x)),
        "fit_min_cycle": int(fit_cycles.min()),
        "fit_max_cycle": int(fit_cycles.max()),
        "feature_names": list(FEATURE_NAMES),
        "sample_weighting": dict(model._weight_audit),
        "fit_warnings": list(model._fit_warnings),
        "policy": "clip(.5625*rank50(new weighted SVD32 LGB)+.1875*frozen mapped Ridge+.25*frozen mapped sparse,0,100)",
        "control_rows_verified_as_unique_increasing_positions": True,
        "score_labels_passed_to_adapter": False,
    }
    return {
        "prediction": candidate,
        "raw_candidate": raw,
        "fit_predictor_count": 1,
        "state": state,
        "fit_metadata": fit_metadata,
        "audit_arrays": {
            "score_cycles": score_cycles,
            "control_rows": control_rows,
            "candidate_rank50": mapped,
        },
    }


__all__ = [
    "FEATURE_NAMES",
    "WEIGHT_FORMULA",
    "ExponentiallyWeightedSVDLGB",
    "fit_predict",
    "normalized_recency_weights",
    "rank50_prediction",
    "replace_svd_component",
]
