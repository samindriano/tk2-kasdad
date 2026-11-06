from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from src.models import cls_pos_recency_e79_v1 as e79
from src.models import cls_pos_recency_v1 as pilot


def test_e79_recency_weights_and_floor() -> None:
    cycles = np.asarray([47, 55, 67, 75, 79], dtype=np.int64)
    expected = np.maximum(0.25, 2.0 ** (-(79 - cycles) / 8.0))
    np.testing.assert_array_equal(e79.recency_base_weights_e79(cycles), expected)


def test_half_life_floor_rule_matches_frozen_pilot_at_e67_and_e75() -> None:
    # The E79 helper is pinned to origin 79. Check the shared half-life/floor
    # rule against both supported historical origins without extending them.
    for origin in (67, 75):
        cycles = np.arange(47, origin + 1, dtype=np.int64)
        expected = np.maximum(0.25, 2.0 ** (-(origin - cycles) / 8.0))
        np.testing.assert_array_equal(pilot.recency_base_weights(cycles, origin), expected)


@pytest.mark.parametrize("cycles", [[46], [80], [79.5], [float("nan")]])
def test_e79_rejects_cycles_outside_frozen_train_population(cycles: list[float]) -> None:
    with pytest.raises(RuntimeError):
        e79.recency_base_weights_e79(cycles)


def test_class_normalization_preserves_each_present_class_row_mass() -> None:
    cycles = np.asarray([47, 55, 67, 75, 79, 79], dtype=np.int64)
    target = np.asarray([0, 0, 1, 1, 1, 0], dtype=np.int8)
    weights = e79.normalized_class_weights(cycles, target)
    for label in (0, 1):
        mask = target == label
        assert np.isclose(weights[mask].sum(), mask.sum(), rtol=1e-12, atol=1e-10)
        assert np.isclose(weights[mask].mean(), 1.0, rtol=1e-12, atol=1e-12)


def test_route_keeps_missing_probability_bit_exact_and_thresholds_frozen() -> None:
    present = np.asarray([True, False, True, False])
    p_a02 = np.asarray([0.2, 0.53230976129477, 0.8, 0.1], dtype=np.float64)
    p_pos = np.asarray([0.8, 0.9, 0.2, 0.9], dtype=np.float64)
    probability = e79.route_probabilities(present, p_a02, p_pos)
    assert np.array_equal(probability[~present].view(np.uint64), p_a02[~present].view(np.uint64))
    labels = e79.apply_frozen_thresholds(probability, present)
    np.testing.assert_array_equal(labels, np.asarray([1, 1, 1, 0], dtype=np.int8))


def test_positional_probability_matches_direct_frozen_formula() -> None:
    x = sparse.csr_matrix(np.asarray([[1.0, 0.0], [0.0, 2.0], [1.0, -1.0]]))
    offset = np.asarray([-0.2, 0.3, 0.0])
    gamma = np.asarray([0.1, -0.15])
    expected = 1.0 / (1.0 + np.exp(-(offset + x @ gamma)))
    np.testing.assert_allclose(e79.positional_probability(x, offset, gamma), expected, rtol=0.0, atol=1e-15)
