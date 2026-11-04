import numpy as np
import pytest

from scripts import reg_position_failure_mechanism_v1 as audit


def test_rank50_uses_average_ties_within_each_cycle():
    raw = np.array([1.0, 1.0, 3.0, 9.0])
    cycles = np.array([70, 70, 70, 71])

    mapped, percentile = audit.rank50_map(raw, cycles)

    np.testing.assert_array_equal(percentile, np.array([50.0, 50.0, 100.0, 100.0]))
    np.testing.assert_array_equal(mapped, np.array([25.5, 25.5, 51.5, 54.5]))


def test_rank50_rejects_row_cycle_misalignment():
    with pytest.raises(audit.AuditError, match="misaligned"):
        audit.rank50_map(np.array([1.0, 2.0]), np.array([70]))


def test_direct_policy_keeps_missing_rows_exactly_at_r06():
    baseline = np.array([20.0, 99.0, 40.0])
    mapped = np.array([80.0, 0.0, 120.0])
    missing = np.array([False, True, False])

    result = audit.direct_policy(baseline, mapped, missing)

    np.testing.assert_array_equal(result, np.array([35.0, 99.0, 60.0]))


def test_residual_policy_clips_delta_and_restores_missing_baseline():
    baseline = np.array([98.0, 30.0, 50.0])
    head = np.array([40.0, -40.0, 8.0])
    missing = np.array([False, False, True])

    delta, prediction = audit.residual_policy(baseline, head, missing)

    np.testing.assert_array_equal(delta, np.array([5.0, -5.0, 0.0]))
    np.testing.assert_array_equal(prediction, np.array([100.0, 25.0, 50.0]))


def test_alignment_rejects_wrong_record_id_even_when_rows_and_cycles_match():
    with pytest.raises(audit.AuditError, match="alignment failed"):
        audit.verify_alignment(
            np.array([5, 6]), np.array(["id-a", "wrong"]), np.array([70, 71]),
            np.array([5, 6]), np.array(["id-a", "id-b"]), np.array([70, 71]),
            "synthetic",
        )
