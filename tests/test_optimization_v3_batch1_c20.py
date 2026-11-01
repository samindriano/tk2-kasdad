import numpy as np
import pytest

from src.models.optimization_v3_batch1_c20 import (
    compose_c20_predictions,
    fit_only_q80,
    missing_stream_mask,
)


def test_q80_uses_all_rows_available_by_cutoff_and_excludes_future_rows():
    values = np.array([0.0, 10.0, 20.0, 1000.0])
    cycles = np.array([47, 48, 49, 50])
    # Ordinary linear q80 of [0, 10, 20] is 16; the future row is excluded.
    assert fit_only_q80(values, cycles, 49) == pytest.approx(16.0)


def test_q80_population_does_not_drop_missing_stream_rows():
    values = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    cycles = np.array([47, 48, 49, 50, 51])
    # The feature-only helper receives all fit rows; q80 is 32 rather than a
    # present-only threshold that would drop the first two values.
    assert fit_only_q80(values, cycles, 51) == pytest.approx(32.0)


def test_missing_precedence_strict_threshold_and_remaining_present_route():
    probability, counts = compose_c20_predictions(
        stream_missing=[True, False, False, False],
        text_load=[10.0, 10.0, 11.0, 8.0],
        q80=10.0,
        sparse25_probability=[0.7, 0.7, 0.7, 0.7],
        c2b_probability=[0.2, 0.2, 0.2, 0.2],
        cb_probability=[0.9, 0.9, 0.9, 0.9],
    )
    np.testing.assert_allclose(probability, [0.9, 0.2, 0.7, 0.2])
    assert counts == {
        "missing_cb": 1,
        "present_high_sparse25": 1,
        "present_other_c2b": 2,
    }


def test_missing_stream_mask_treats_null_and_whitespace_as_missing():
    np.testing.assert_array_equal(
        missing_stream_mask([None, np.nan, "", "  \t", "signal"]),
        [True, True, True, True, False],
    )


def test_route_rejects_nonfinite_or_misaligned_inputs():
    with pytest.raises(ValueError):
        compose_c20_predictions(
            stream_missing=[False, False], text_load=[1.0, np.nan], q80=1.0,
            sparse25_probability=[0.5, 0.5], c2b_probability=[0.5, 0.5],
            cb_probability=[0.5, 0.5],
        )
    with pytest.raises(ValueError):
        compose_c20_predictions(
            stream_missing=[False], text_load=[1.0, 2.0], q80=1.0,
            sparse25_probability=[0.5], c2b_probability=[0.5], cb_probability=[0.5],
        )
