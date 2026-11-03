from __future__ import annotations

import numpy as np

from scripts.optimization_v3_rnbr_target_median_v1_synthetic import run_synthetic_qa
from src.models.regression_neighbor_rnbr_target_median import (
    predict_from_target_median,
    shuffle_targets_within_cycle_decile,
    weighted_median_target,
)


def test_target_median_exact_half_midpoint_and_grouped_ties() -> None:
    assert weighted_median_target(np.asarray([10.0, 30.0]), np.ones(2), np.asarray([2, 1])) == 20.0
    assert weighted_median_target(np.asarray([10.0, 10.0, 40.0]), np.ones(3), np.asarray([3, 1, 2])) == 10.0


def test_direct_target_median_is_distinct_from_residual_median() -> None:
    direct = weighted_median_target(np.asarray([40.0, 80.0]), np.ones(2), np.asarray([1, 2]))
    residual = weighted_median_target(np.asarray([30.0, 10.0]), np.ones(2), np.asarray([1, 2]))
    assert direct == 60.0
    assert residual == 20.0
    assert 50.0 + 0.25 * (direct - 50.0) == 52.5
    assert 50.0 + 0.25 * residual == 55.0


def test_target_shuffle_is_deterministic_and_preserves_values() -> None:
    targets = np.arange(32, dtype=np.float64)
    cycles = np.repeat(np.asarray([56, 57]), 16)
    prediction = np.full(32, 5.0)
    rows = np.arange(100, 132, dtype=np.int64)
    first, source1, audit1 = shuffle_targets_within_cycle_decile(targets, cycles, prediction, rows, seed=42)
    second, source2, audit2 = shuffle_targets_within_cycle_decile(targets, cycles, prediction, rows, seed=42)
    assert np.array_equal(first, second)
    assert np.array_equal(source1, source2)
    assert np.array_equal(audit1["source_positions"], audit2["source_positions"])
    assert np.array_equal(np.sort(first), np.sort(targets))
    assert audit1["fraction_moved"] >= 0.8


def test_fallback_is_exact_for_missing_and_under_supported() -> None:
    result = predict_from_target_median(
        donor_rows=np.arange(10), donor_targets=np.arange(10, dtype=np.float64),
        query_r06_predictions=np.asarray([40.0, 30.0]), query_missing=np.asarray([False, True]),
        neighbor_rows=np.asarray([[0, 1, 2, 3, -1], [0, 1, 2, 3, 4]]),
        neighbor_cosines=np.asarray([[0.9, 0.8, 0.7, 0.6, 0.0], [0.9, 0.8, 0.7, 0.6, 0.5]]),
        neighbor_counts=np.asarray([4, 5]),
    )
    assert np.array_equal(result["prediction"], np.asarray([40.0, 30.0]))
    assert not result["median_applied"].any()


def test_synthetic_persistence_budget_and_resume_lifecycle(tmp_path) -> None:
    result = run_synthetic_qa(tmp_path / "synthetic_qa.json")
    assert result["status"] == "PASS_SYNTHETIC_END_TO_END"
    assert result["live_ledger_unchanged"] is True
    assert result["synthetic_fit_starts_and_saves"] == {"starts": 4, "saved": 4, "pending": 0}
    assert result["resume_added_starts"] == 0
