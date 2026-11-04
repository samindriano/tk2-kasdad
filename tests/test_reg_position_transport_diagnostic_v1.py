import numpy as np
import pytest

from scripts.reg_position_transport_diagnostic_v1 import (
    DiagnosticError,
    distribution,
    fixed_decile,
    pair_by_id,
)


def test_fixed_r06_deciles_clip_bounds_and_cap_100_at_nine():
    np.testing.assert_array_equal(
        fixed_decile(np.array([-10.0, 0.0, 9.999, 10.0, 99.999, 100.0, 130.0])),
        np.array([0, 0, 0, 1, 9, 9, 9], dtype=np.int8),
    )


def test_residual_distribution_withholds_sparse_cell_quantiles():
    unsupported = distribution(np.arange(99.0))
    assert unsupported == {
        "n": 99,
        "status": "UNKNOWN_UNSUPPORTED",
        "median": None,
        "p10": None,
        "p90": None,
    }
    supported = distribution(np.arange(100.0))
    assert supported["n"] == 100
    assert supported["status"] == "SUPPORTED"
    assert supported["median"] == pytest.approx(49.5)
    assert supported["p10"] < supported["median"] < supported["p90"]


def test_w6_pairing_joins_by_id_not_file_order():
    left_ids = np.array(["β", "a", "長"])
    right_ids = np.array(["長", "β", "a"])
    left, right = pair_by_id(left_ids, right_ids)
    assert left_ids[left].tolist() == ["a", "β", "長"]
    assert right_ids[right].tolist() == ["a", "β", "長"]


def test_w6_pairing_rejects_missing_duplicate_ids():
    with pytest.raises(DiagnosticError, match="not identical"):
        pair_by_id(np.array(["a", "b"]), np.array(["a", "c"]))
    with pytest.raises(DiagnosticError, match="unique"):
        pair_by_id(np.array(["a", "a"]), np.array(["a", "b"]))
