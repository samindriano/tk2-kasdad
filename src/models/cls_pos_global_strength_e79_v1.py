"""Pure helpers for the source-only E79 global-strength positional recipe.

The only estimator recipe is imported from the frozen one-sided pilot. This
module contains no file or dataset access, so synthetic QA can exercise it
without touching official labels, Test data, or cached predictions.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import sparse

from src.models.pos_one_sided_ridge_v1 import (
    build_penalty_vectors,
    fit_offset,
    route_probabilities,
)


FIT_ID = "E79_GLOBAL_STRENGTH"
GLOBAL_PRESENT_BLEND_WEIGHT = 0.5
PRESENT_THRESHOLD = 0.50
MISSING_THRESHOLD = 0.53230976129477


def global_strength_penalty(aligned_token_weights: Sequence[float]) -> tuple[np.ndarray, float]:
    """Return the frozen mean-matched scalar repeated over all 3 positional blocks."""
    arms = build_penalty_vectors(aligned_token_weights)
    scalar = float(arms["global_strength_scalar"])
    penalty = np.asarray(arms["global_strength_position_weights"], dtype=np.float64)
    expected = np.full(3 * len(np.asarray(aligned_token_weights).reshape(-1)), scalar,
                       dtype=np.float64)
    if not np.isfinite(penalty).all() or not np.array_equal(penalty, expected):
        raise ValueError("Global-strength penalty must repeat the fixed scalar across all coordinates")
    return penalty, scalar


def fit_global_strength_gamma(
    x_position: sparse.spmatrix,
    offset: Sequence[float],
    target: Sequence[int],
    aligned_token_weights: Sequence[float],
) -> tuple[np.ndarray, dict[str, object]]:
    """Use the frozen summed-logloss, weighted-L2, L-BFGS-B implementation."""
    penalty, _ = global_strength_penalty(aligned_token_weights)
    return fit_offset(x_position, offset, target, penalty)


def combine_present_and_missing_probabilities(
    positional_probability: Sequence[float],
    a02_probability: Sequence[float],
    present_stream: Sequence[bool],
) -> np.ndarray:
    """Apply the fixed 50/50 present blend and bit-exact A02 missing fallback."""
    positional = np.asarray(positional_probability, dtype=np.float64).reshape(-1)
    a02 = np.asarray(a02_probability, dtype=np.float64).reshape(-1)
    present = np.asarray(present_stream, dtype=bool).reshape(-1)
    if positional.shape != a02.shape or present.shape != a02.shape:
        raise ValueError("Probability and availability arrays must have identical lengths")
    if not np.isfinite(a02).all() or np.any((a02 < 0.0) | (a02 > 1.0)):
        raise ValueError("A02 probabilities must be finite and in [0, 1]")
    if not np.isfinite(positional[present]).all() or np.any(
        (positional[present] < 0.0) | (positional[present] > 1.0)
    ):
        raise ValueError("Present-route positional probabilities must be finite and in [0, 1]")
    result = a02.copy()
    result[present] = (
        GLOBAL_PRESENT_BLEND_WEIGHT * positional[present]
        + (1.0 - GLOBAL_PRESENT_BLEND_WEIGHT) * a02[present]
    )
    if not np.array_equal(result[~present], a02[~present]):
        raise ValueError("Missing-stream A02 probabilities changed")
    if not np.isfinite(result).all() or np.any((result < 0.0) | (result > 1.0)):
        raise ValueError("Combined probabilities are invalid")
    return result


def apply_frozen_thresholds(
    probability: Sequence[float],
    present_stream: Sequence[bool],
) -> np.ndarray:
    """Apply threshold 0.50 to present streams and the frozen A02 threshold to missing."""
    values = np.asarray(probability, dtype=np.float64).reshape(-1)
    present = np.asarray(present_stream, dtype=bool).reshape(-1)
    if values.shape != present.shape or not np.isfinite(values).all():
        raise ValueError("Probability and availability arrays must be finite and aligned")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("Probabilities must be in [0, 1]")
    thresholds = np.where(present, PRESENT_THRESHOLD, MISSING_THRESHOLD)
    return (values >= thresholds).astype(np.int8)


def route_and_combine(
    x_position: sparse.spmatrix,
    offset: Sequence[float],
    gamma: Sequence[float],
    present_stream: Sequence[bool],
    a02_probability: Sequence[float],
) -> np.ndarray:
    """Score only present rows, then apply the frozen present blend and fallback."""
    positional_route = route_probabilities(
        x_position,
        offset,
        gamma,
        present_stream,
        a02_probability,
    )
    return combine_present_and_missing_probabilities(
        positional_route,
        a02_probability,
        present_stream,
    )
