import numpy as np
import pytest

from src.models.optimization_v3_batch1_r19 import (
    apply_ef_median_offsets,
    fit_ef_median_offsets,
    parse_entry_friction,
)


def test_parse_entry_friction_keeps_only_binary_values():
    np.testing.assert_array_equal(
        parse_entry_friction([0, 1, "0", "1", None, 2, "unknown"]),
        [0, 1, 0, 1, -1, -1, -1],
    )


def test_offsets_use_half_median_of_target_minus_prequential_prediction():
    y = np.array([12.0] * 200 + [8.0] * 200)
    p = np.array([10.0] * 200 + [10.0] * 200)
    ef = np.array([0] * 200 + [1] * 200)
    offsets, support = fit_ef_median_offsets(y, p, ef)
    assert offsets == {0: 1.0, 1: -1.0}
    assert support["0"]["supported"] is True
    assert support["1"]["supported"] is True


def test_unsupported_and_unknown_groups_receive_zero_offset():
    offsets, support = fit_ef_median_offsets(
        [10.0, 12.0, 20.0], [0.0, 0.0, 0.0], [0, 0, None], minimum_support=2,
    )
    assert offsets == {0: 2.0, 1: 0.0}
    assert support["0"]["rows"] == 2
    assert support["1"]["rows"] == 0
    assert support["unknown"]["rows"] == 1


def test_offsets_are_clipped_and_predictions_stay_in_target_range():
    offsets, _ = fit_ef_median_offsets(
        [100.0] * 200, [0.0] * 200, [1] * 200,
    )
    assert offsets[1] == 2.0
    adjusted = apply_ef_median_offsets([99.5, 50.0, 0.5], [1, 0, 1], {0: -2.0, 1: 2.0})
    np.testing.assert_allclose(adjusted, [100.0, 48.0, 2.5])


def test_misaligned_or_nonfinite_calibration_inputs_are_rejected():
    with pytest.raises(ValueError):
        fit_ef_median_offsets([1, 2], [1], [0, 1])
    with pytest.raises(ValueError):
        fit_ef_median_offsets([1, np.nan], [1, 1], [0, 1])

