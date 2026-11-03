"""Pure math for the bounded RNBR_TARGET_MEDIAN_V1 diagnostic.

This module does not read data, access a ledger, fit an estimator, or persist
artifacts. It reuses the established stable cosine-squared weighted-median
ordering while applying it to actual donor targets rather than residuals.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.models.regression_neighbor_rnbr import (
    ALPHA,
    CORRECTION_CAP,
    MAX_NEIGHBORS,
    MIN_NEIGHBORS,
    SHUFFLE_SEED,
    RnbrProtocolError,
    fixed_prediction_decile,
    shuffle_residuals_within_cycle_decile,
    weighted_median_residual,
)


PILOT_ID = "RNBR_TARGET_MEDIAN_V1"
ORIGINS = (67, 75)
ARMS = ("observed", "shuffled")
MIN_PRESENT_SUPPORT_COVERAGE = 0.95
MIN_SHUFFLE_SOURCE_MOVEMENT = 0.80
EXPECTED_SOURCE_CUTOFFS = {
    **{cycle: 55 for cycle in range(56, 60)},
    **{cycle: 59 for cycle in range(60, 64)},
    **{cycle: 63 for cycle in range(64, 68)},
    **{cycle: 67 for cycle in range(68, 72)},
    **{cycle: 71 for cycle in range(72, 76)},
    **{cycle: 75 for cycle in range(76, 80)},
}


class TargetMedianError(RnbrProtocolError):
    """A target-median input, lineage, or prediction invariant failed."""


def weighted_median_target(targets: Any, cosines: Any, source_rows: Any) -> float:
    """Weighted median of donor ``threat_rank`` values.

    The established implementation is value-generic; this named wrapper makes
    the new functional explicit while preserving its source-row tie order and
    exact-half-mass midpoint rule.
    """
    values = np.asarray(targets, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise TargetMedianError("Donor targets must be a finite one-dimensional vector")
    if np.any((values < 0.0) | (values > 100.0)):
        raise TargetMedianError("Donor targets must be in [0,100]")
    return weighted_median_residual(values, cosines, source_rows)


def shuffle_targets_within_cycle_decile(
    targets: Any,
    cycles: Any,
    r06_predictions: Any,
    source_rows: Any,
    *,
    seed: int = SHUFFLE_SEED,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return shuffled target values, their source rows, and a shuffle audit.

    The mapping uses the frozen ascending cycle × fixed-R06-decile × source-row
    traversal and a fresh seed-42 RNG per origin. Feeding unique source positions
    through the existing pure shuffle primitive exposes the exact permutation.
    """
    values = np.asarray(targets, dtype=np.float64)
    rows = np.asarray(source_rows, dtype=np.int64)
    cycles_array = np.asarray(cycles, dtype=np.int64)
    predictions = np.asarray(r06_predictions, dtype=np.float64)
    n = len(values)
    if (values.ndim != 1 or rows.shape != (n,) or cycles_array.shape != (n,)
            or predictions.shape != (n,)):
        raise TargetMedianError("Target shuffle vectors are not aligned")
    if (not np.isfinite(values).all() or not np.isfinite(predictions).all()
            or np.any((values < 0.0) | (values > 100.0))
            or np.any((predictions < 0.0) | (predictions > 100.0))
            or len(np.unique(rows)) != n):
        raise TargetMedianError("Target shuffle inputs must be finite, bounded, and uniquely row-aligned")
    source_positions = shuffle_residuals_within_cycle_decile(
        np.arange(n, dtype=np.float64), cycles_array, predictions, rows, seed=seed,
    )
    permutation = source_positions.astype(np.int64)
    if (not np.array_equal(source_positions, permutation.astype(np.float64))
            or np.any((permutation < 0) | (permutation >= n))
            or len(np.unique(permutation)) != n):
        raise TargetMedianError("Target shuffle did not produce a donor permutation")
    shuffled = values[permutation]
    shuffled_source_rows = rows[permutation]
    deciles = fixed_prediction_decile(predictions)
    moved = int(np.sum(permutation != np.arange(n, dtype=np.int64)))
    return shuffled, shuffled_source_rows, {
        "seed": int(seed),
        "fixed_prediction_deciles": deciles,
        "source_positions": permutation,
        "moved_rows": moved,
        "donor_rows": int(n),
        "fraction_moved": moved / n if n else 0.0,
        "target_multiset_preserved": bool(np.array_equal(np.sort(shuffled), np.sort(values))),
    }


def predict_from_target_median(
    *,
    donor_rows: Any,
    donor_targets: Any,
    query_r06_predictions: Any,
    query_missing: Any,
    neighbor_rows: Any,
    neighbor_cosines: Any,
    neighbor_counts: Any,
) -> dict[str, np.ndarray]:
    """Apply the frozen shrinkage to a cosine-squared median of target levels.

    ``median_target`` is the weighted median of actual donor targets. The
    correction is ``clip(0.25 * (median_target - R06_query), -5, +5)``;
    missing and fewer-than-five-neighbor rows return the R06 prediction exactly.
    """
    donors = np.asarray(donor_rows, dtype=np.int64)
    targets = np.asarray(donor_targets, dtype=np.float64)
    baseline = np.asarray(query_r06_predictions, dtype=np.float64)
    missing = np.asarray(query_missing, dtype=bool)
    rows = np.asarray(neighbor_rows, dtype=np.int64)
    cosines = np.asarray(neighbor_cosines, dtype=np.float64)
    counts = np.asarray(neighbor_counts, dtype=np.int64)
    n_query = len(baseline)
    if (donors.ndim != 1 or targets.shape != donors.shape or len(np.unique(donors)) != len(donors)
            or baseline.ndim != 1 or missing.shape != baseline.shape or rows.ndim != 2
            or rows.shape[0] != n_query or cosines.shape != rows.shape
            or counts.shape != baseline.shape or rows.shape[1] > MAX_NEIGHBORS
            or np.any((counts < 0) | (counts > rows.shape[1]))):
        raise TargetMedianError("Target-median donor/query arrays have invalid shapes or counts")
    if (not np.isfinite(targets).all() or not np.isfinite(baseline).all()
            or not np.isfinite(cosines).all() or np.any((targets < 0.0) | (targets > 100.0))
            or np.any((baseline < 0.0) | (baseline > 100.0))):
        raise TargetMedianError("Target-median values, baseline, and cosine values must be finite and bounded")
    lookup = {int(row): i for i, row in enumerate(donors)}
    median = np.zeros(n_query, dtype=np.float64)
    median_applied = np.zeros(n_query, dtype=bool)
    delta = np.zeros(n_query, dtype=np.float64)
    used = np.zeros(n_query, dtype=np.int16)
    for i, count_value in enumerate(counts):
        k = int(count_value)
        if missing[i] or k < MIN_NEIGHBORS:
            continue
        active_rows = rows[i, :k]
        if len(np.unique(active_rows)) != k:
            raise TargetMedianError("A supported query has duplicate neighbor source rows")
        try:
            positions = np.asarray([lookup[int(row)] for row in active_rows], dtype=np.int64)
        except KeyError as exc:
            raise TargetMedianError("A neighbor row is outside the donor target memory") from exc
        active_cosines = cosines[i, :k]
        if np.any((active_cosines <= 0.0) | (active_cosines > 1.0 + 1e-12)):
            raise TargetMedianError("Supported neighbor cosine must be positive and at most one")
        value = weighted_median_target(targets[positions], active_cosines, active_rows)
        median[i] = value
        median_applied[i] = True
        correction = float(np.clip(ALPHA * (value - baseline[i]), -CORRECTION_CAP, CORRECTION_CAP))
        delta[i] = correction
        used[i] = k
    prediction = np.clip(baseline + delta, 0.0, 100.0)
    if (not np.isfinite(prediction).all() or np.any(delta[missing] != 0.0)
            or not np.array_equal(prediction[missing], baseline[missing])
            or not np.array_equal(prediction[~median_applied], baseline[~median_applied])):
        raise TargetMedianError("Prediction is invalid or exact R06 fallback was not preserved")
    return {
        "baseline": baseline.copy(),
        "weighted_median_target": median,
        "median_applied": median_applied,
        "delta": delta,
        "prediction": prediction,
        "neighbor_count_used": used,
    }


__all__ = [
    "ARMS", "EXPECTED_SOURCE_CUTOFFS", "MIN_PRESENT_SUPPORT_COVERAGE",
    "MIN_SHUFFLE_SOURCE_MOVEMENT", "ORIGINS", "PILOT_ID", "TargetMedianError",
    "predict_from_target_median", "shuffle_targets_within_cycle_decile",
    "weighted_median_target",
]
