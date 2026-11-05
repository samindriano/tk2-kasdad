"""Pure recipe helpers for the frozen CLS one-sided positional-ridge pilot."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from scipy import sparse
from scipy.special import expit

from src.models.pos_frequency_ridge_v1 import (
    FrequencyRidgeError,
    fit_weighted_offset,
    repeat_token_weights,
    weighted_offset_objective_gradient,
)


CAMPAIGN_ID = "CLS_POS_ONE_SIDED_RIDGE_V1"
ARMS = ("ONE_SIDED", "GLOBAL_STRENGTH")
ORIGINS = (67, 75)
FIT_IDS = tuple(f"E{origin}_{arm}" for origin in ORIGINS for arm in ARMS)


def build_penalty_vectors(aligned_token_weights: Sequence[float]) -> dict[str, Any]:
    """Build v=max(1,w) and its mean-matched scalar control, without renormalizing v."""
    w = np.asarray(aligned_token_weights, dtype=np.float64).reshape(-1)
    if w.size < 2 or not np.isfinite(w).all() or np.any(w <= 0):
        raise FrequencyRidgeError("Saved aligned weights must be finite positive and nontrivial")
    one_sided = np.maximum(1.0, w)
    scalar = float(np.mean(one_sided, dtype=np.float64))
    if not np.isfinite(scalar) or scalar < 1.0:
        raise FrequencyRidgeError("Global-strength scalar must be finite and at least one")
    global_token = np.full(w.shape, scalar, dtype=np.float64)
    one_sided_repeated = repeat_token_weights(one_sided)
    global_repeated = repeat_token_weights(global_token)
    if not np.isclose(float(np.mean(one_sided_repeated)), scalar, rtol=0.0, atol=1e-14):
        raise FrequencyRidgeError("Global control does not match the mean one-sided penalty")
    if not np.array_equal(one_sided_repeated[:w.size], one_sided_repeated[w.size:2*w.size]):
        raise FrequencyRidgeError("One-sided penalties differ across positional blocks")
    if not np.array_equal(global_repeated, np.full(3*w.size, scalar, dtype=np.float64)):
        raise FrequencyRidgeError("Global-strength control is not a repeated single scalar")
    return {
        "aligned_token_weights": w.copy(),
        "one_sided_token_weights": one_sided,
        "global_strength_scalar": scalar,
        "global_strength_token_weights": global_token,
        "one_sided_position_weights": one_sided_repeated,
        "global_strength_position_weights": global_repeated,
        "one_sided_mean": float(np.mean(one_sided)),
        "aligned_mean": float(np.mean(w)),
        "one_sided_renormalized": False,
    }


def objective_gradient(
    gamma: np.ndarray,
    x_position: sparse.spmatrix,
    offset: Sequence[float],
    y: Sequence[int],
    position_weights: Sequence[float],
) -> tuple[float, np.ndarray]:
    """Use the already frozen summed-logloss plus half weighted-L2 objective."""
    return weighted_offset_objective_gradient(gamma, x_position, offset, y, position_weights)


def fit_offset(
    x_position: sparse.spmatrix,
    offset: Sequence[float],
    y: Sequence[int],
    position_weights: Sequence[float],
) -> tuple[np.ndarray, dict[str, Any]]:
    gamma, metadata = fit_weighted_offset(x_position, offset, y, position_weights)
    if metadata.get("success") is not True:
        raise FrequencyRidgeError("Frozen L-BFGS-B optimizer did not report success")
    if not np.isfinite(gamma).all() or not np.isfinite(metadata.get("loss_final", np.nan)):
        raise FrequencyRidgeError("Optimizer returned non-finite coefficient or objective")
    if float(metadata["loss_final"]) > float(metadata["loss_gamma0"]) + 1e-8:
        raise FrequencyRidgeError("Fitted objective exceeds the gamma=0 objective tolerance")
    return gamma, metadata


def route_probabilities(
    x_position: sparse.spmatrix,
    offset: Sequence[float],
    gamma: Sequence[float],
    present: Sequence[bool],
    missing_fallback: Sequence[float],
) -> np.ndarray:
    """Score the present route and preserve every missing fallback value bit-for-bit."""
    x = sparse.csr_matrix(x_position, dtype=np.float64)
    z0 = np.asarray(offset, dtype=np.float64).reshape(-1)
    coef = np.asarray(gamma, dtype=np.float64).reshape(-1)
    route = np.asarray(present, dtype=bool).reshape(-1)
    fallback = np.asarray(missing_fallback, dtype=np.float64).reshape(-1)
    if (x.shape != (len(route), len(coef)) or len(z0) != len(route)
            or len(fallback) != len(route) or not np.isfinite(x.data).all()
            or not np.isfinite(z0).all() or not np.isfinite(coef).all()
            or not np.isfinite(fallback).all() or np.any((fallback < 0) | (fallback > 1))):
        raise FrequencyRidgeError("Prediction inputs are not finite or dimensionally aligned")
    out = fallback.copy()
    if route.any():
        out[route] = expit(z0[route] + np.asarray(x[route] @ coef, dtype=np.float64).reshape(-1))
    if not np.array_equal(out[~route], fallback[~route]):
        raise FrequencyRidgeError("Missing-stream fallback changed during availability routing")
    if not np.isfinite(out).all() or np.any((out < 0) | (out > 1)):
        raise FrequencyRidgeError("Routed probabilities are invalid")
    return out


def make_arm_penalties(aligned_token_weights: Sequence[float]) -> dict[str, np.ndarray]:
    vectors = build_penalty_vectors(aligned_token_weights)
    return {
        "ONE_SIDED": vectors["one_sided_position_weights"],
        "GLOBAL_STRENGTH": vectors["global_strength_position_weights"],
    }
