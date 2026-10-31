"""Fit-only composition utilities for the dual regression campaign.

The component estimators retain their own train-only preprocessing. This module
only combines their predictions and selects a convex weight from past-only
validation vectors supplied by the caller; it does not load data or choose folds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.models.regression_research_lgb_text import build_lgb_text
from src.models.regression_research_text import build_text_model


WEIGHT_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def _finite_vector(values: Any, name: str, *, n: int | None = None) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or (n is not None and len(result) != n):
        raise ValueError(f"{name} must be a one-dimensional vector aligned to rows")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def cycle_percentiles(prediction: Any, cycles: Any) -> np.ndarray:
    """Map scores to 0-100 within each supplied cycle, averaging ties."""
    pred = _finite_vector(prediction, "prediction")
    cycle = np.asarray(cycles)
    if cycle.ndim != 1 or len(cycle) != len(pred) or pd.isna(cycle).any():
        raise ValueError("cycles must be a nonmissing one-dimensional vector aligned to predictions")
    mapped = np.empty(len(pred), dtype=np.float64)
    for code in pd.unique(cycle):
        mask = cycle == code
        mapped[mask] = 100.0 * rankdata(pred[mask], method="average") / int(mask.sum())
    return mapped


def apply_cycle_rank_blend(prediction: Any, cycles: Any, rank_weight: float) -> np.ndarray:
    """Apply the frozen per-cycle average-rank blend and [0, 100] clip."""
    if not np.isfinite(rank_weight) or not 0.0 <= rank_weight <= 1.0:
        raise ValueError("rank_weight must be finite and in [0, 1]")
    pred = _finite_vector(prediction, "prediction")
    if rank_weight == 0:
        return np.clip(pred, 0.0, 100.0)
    pct = cycle_percentiles(pred, cycles)
    return np.clip((1.0 - rank_weight) * pred + rank_weight * pct, 0.0, 100.0)


def blend_predictions(lightgbm_prediction: Any, ridge_prediction: Any,
                      ridge_weight: float) -> np.ndarray:
    """Convexly blend already policy-mapped component predictions."""
    if not np.isfinite(ridge_weight) or not 0.0 <= ridge_weight <= 1.0:
        raise ValueError("ridge_weight must be finite and in [0, 1]")
    lgb = _finite_vector(lightgbm_prediction, "lightgbm_prediction")
    ridge = _finite_vector(ridge_prediction, "ridge_prediction", n=len(lgb))
    return np.clip((1.0 - ridge_weight) * lgb + ridge_weight * ridge, 0.0, 100.0)


@dataclass(frozen=True)
class WeightSelection:
    ridge_weight: float
    inner_mae: float
    candidates: tuple[tuple[float, float], ...]


def select_blend_weight(y_true: Any, lightgbm_prediction: Any,
                        ridge_prediction: Any, *,
                        weights: Iterable[float] = WEIGHT_GRID,
                        tolerance: float = 1e-10) -> WeightSelection:
    """Choose Ridge share using only aligned historical validation predictions.

    The caller is responsible for creating past-only predictions. Ties within
    ``tolerance`` favor the lower Ridge share, retaining more of LightGBM.
    """
    actual = _finite_vector(y_true, "y_true")
    lgb = _finite_vector(lightgbm_prediction, "lightgbm_prediction", n=len(actual))
    ridge = _finite_vector(ridge_prediction, "ridge_prediction", n=len(actual))
    candidate_weights = tuple(float(w) for w in weights)
    if not candidate_weights or len(candidate_weights) != len(set(candidate_weights)):
        raise ValueError("weights must be a nonempty sequence without duplicates")
    if any(not np.isfinite(w) or not 0.0 <= w <= 1.0 for w in candidate_weights):
        raise ValueError("all weights must be finite and in [0, 1]")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    scores = [(w, float(np.mean(np.abs(blend_predictions(lgb, ridge, w) - actual))))
              for w in candidate_weights]
    minimum = min(score for _, score in scores)
    chosen_weight, chosen_score = min(
        ((w, score) for w, score in scores if score <= minimum + tolerance),
        key=lambda item: item[0],
    )
    return WeightSelection(chosen_weight, chosen_score, tuple(scores))


class RegressionCampaignBlend:
    """Fit both frozen TP2 text components and blend mapped predictions.

    ``rank_weight_lgb`` and ``rank_weight_ridge`` are hyperparameters selected
    by the caller on past-only temporal validation. They are never learned from
    the prediction batch passed to :meth:`predict`.
    """

    def __init__(self, *, lgb_params: str = "base", ridge_weight: float = 0.25):
        if not 0.0 <= ridge_weight <= 1.0:
            raise ValueError("ridge_weight must be in [0, 1]")
        self.lgb_params = lgb_params
        self.ridge_weight = float(ridge_weight)
        self.lgb_model = build_lgb_text(lgb_params)
        self.ridge_model = build_text_model("hybrid_ridge")
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: Any) -> "RegressionCampaignBlend":
        target = _finite_vector(y, "y", n=len(X))
        if len(X) == 0:
            raise ValueError("X must contain at least one training row")
        lgb_features = list(self.lgb_model.feature_names)
        ridge_features = list(self.ridge_model.feature_names)
        self.lgb_model.fit(X.loc[:, lgb_features].copy(), target.copy())
        self.ridge_model.fit(X.loc[:, ridge_features].copy(), target.copy())
        self._fit_rows = len(target)
        self._fitted = True
        return self

    def predict_components(self, X: pd.DataFrame, cycles: Any, *,
                           rank_weight_lgb: float,
                           rank_weight_ridge: float) -> tuple[np.ndarray, np.ndarray]:
        if not self._fitted:
            raise RuntimeError("fit must be called before predict")
        cycle = np.asarray(cycles)
        if cycle.ndim != 1 or len(cycle) != len(X) or pd.isna(cycle).any():
            raise ValueError("cycles must be a nonmissing vector aligned to X")
        lgb_features = list(self.lgb_model.feature_names)
        ridge_features = list(self.ridge_model.feature_names)
        raw_lgb = self.lgb_model.predict(X.loc[:, lgb_features].copy())
        raw_ridge = self.ridge_model.predict(X.loc[:, ridge_features].copy())
        mapped_lgb = apply_cycle_rank_blend(raw_lgb, cycle, rank_weight_lgb)
        mapped_ridge = apply_cycle_rank_blend(raw_ridge, cycle, rank_weight_ridge)
        return mapped_lgb, mapped_ridge

    def predict(self, X: pd.DataFrame, cycles: Any, *,
                rank_weight_lgb: float,
                rank_weight_ridge: float) -> np.ndarray:
        lgb, ridge = self.predict_components(
            X, cycles,
            rank_weight_lgb=rank_weight_lgb,
            rank_weight_ridge=rank_weight_ridge,
        )
        return blend_predictions(lgb, ridge, self.ridge_weight)

    def audit_state(self) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("fit must be called before audit_state")
        return {
            "model": "RegressionCampaignBlend",
            "lgb_params": self.lgb_params,
            "ridge_weight": self.ridge_weight,
            "fit_rows": self._fit_rows,
            "lightgbm": self.lgb_model.audit_state(),
            "hybrid_ridge": self.ridge_model.audit_state(),
        }


__all__ = [
    "WEIGHT_GRID", "WeightSelection", "RegressionCampaignBlend",
    "apply_cycle_rank_blend", "blend_predictions", "cycle_percentiles",
    "select_blend_weight",
]
