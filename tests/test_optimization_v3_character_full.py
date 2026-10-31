import numpy as np
import pytest

from scripts.optimization_v3_character_fullsix import (
    blend25,
    error_summary,
    rank50_within_cycle,
)


def test_rank50_uses_average_ties_and_separate_cycle_ranks():
    raw = np.array([20.0, 20.0, 40.0, 5.0, 10.0])
    cycles = np.array([58, 58, 58, 59, 59])
    got = rank50_within_cycle(raw, cycles)
    np.testing.assert_allclose(got, [35.0, 35.0, 70.0, 27.5, 55.0], atol=0, rtol=0)


def test_rank50_clips_to_target_range_and_rejects_misalignment():
    got = rank50_within_cycle(np.array([200.0, 300.0]), np.array([58, 58]))
    np.testing.assert_array_equal(got, [100.0, 100.0])
    with pytest.raises(ValueError):
        rank50_within_cycle(np.array([1.0, 2.0]), np.array([58]))


def test_fixed_blend_clips_and_keeps_exact_quarter_weight():
    got = blend25(np.array([0.0, 100.0, 20.0]), np.array([100.0, 100.0, 80.0]))
    np.testing.assert_array_equal(got, [25.0, 100.0, 35.0])


def test_error_summary_signed_bias_and_over_under_rates():
    got = error_summary(np.array([10.0, 20.0]), np.array([12.0, 18.0]))
    assert got["mae"] == 2.0
    assert got["bias_prediction_minus_target"] == 0.0
    assert got["over_rate"] == 0.5
    assert got["under_rate"] == 0.5
    assert got["tie_rate"] == 0.0
