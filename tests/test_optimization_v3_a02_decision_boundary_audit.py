import numpy as np
import pytest

from scripts.optimization_v3_a02_decision_boundary_audit import (
    BASELINE_BANDS,
    fixed_band_mask,
    flip_accounting,
    outside_margin_change_count,
)


def test_threshold_uses_greater_equal_and_separates_useful_from_harmful_flips():
    y = np.array([0, 1, 0, 1], dtype=np.int8)
    baseline = np.array([0.50, 0.50, 0.49, 0.51])
    arm = np.array([0.49, 0.51, 0.51, 0.50])

    result = flip_accounting(y, baseline, arm)

    assert result["baseline_exact_050_rows"] == 2
    assert result["baseline_positive_predictions"] == 3
    assert result["arm_positive_predictions"] == 3
    assert result["label_changes"] == 2
    assert result["flip_1_to_0"] == 1
    assert result["flip_0_to_1"] == 1
    assert result["useful_flips_baseline_wrong_to_correct"] == 1
    assert result["harmful_flips_baseline_correct_to_wrong"] == 1
    assert result["net_error_reduction_count"] == 0
    assert result["baseline_macro_f1"] is None  # class support is explicitly insufficient


def test_only_inclusive_fixed_margin_can_contain_changes_and_upper_edge_respects_ge_half():
    baseline = np.array([0.44, 0.45, 0.55, 0.56])
    arm = np.array([0.49, 0.50, 0.50, 0.51])

    assert outside_margin_change_count(baseline, arm) == 0
    # p=.45 can rise to exactly .50 and flip; p=.55 to .50 stays positive.
    labels_base = baseline >= 0.50
    labels_arm = arm >= 0.50
    changes = labels_base != labels_arm
    assert changes.tolist() == [False, True, False, False]


def test_fixed_probability_band_edges_are_left_closed_and_last_band_right_closed():
    edge_values = np.array([0.45, 0.475, 0.50, 0.525, 0.55])
    expected = [
        [True, False, False, False, False],
        [False, True, False, False, False],
        [False, False, True, False, False],
        [False, False, False, True, True],
    ]
    assert [fixed_band_mask(edge_values, i).tolist() for i in range(len(BASELINE_BANDS))] == expected


def test_flip_accounting_rejects_misaligned_or_nonbinary_inputs():
    with pytest.raises(ValueError, match="aligned"):
        flip_accounting(np.array([0, 1]), np.array([0.4]), np.array([0.6]))
    with pytest.raises(ValueError, match="nonbinary"):
        flip_accounting(np.array([0, 2]), np.array([0.4, 0.6]), np.array([0.5, 0.5]))
