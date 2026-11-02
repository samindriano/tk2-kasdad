import numpy as np

from scripts import optimization_v3_b02_scope_reconciliation as correction


def test_corrected_legacy_new_origin_is_exact_w2_w3_w5_support():
    assert correction.LEGACY_NEW_ORIGIN_ORIGINS == (59, 63, 71)
    assert correction.LEGACY_NEW_ORIGIN_CYCLES == (
        60, 61, 62, 63, 64, 65, 66, 67, 72, 73, 74, 75)
    assert len(correction.LEGACY_NEW_ORIGIN_CYCLES) == 12


def test_stored_k_reconstruction_preserves_exact_ties_and_missing_route():
    p = np.array([0.9, 0.9, 0.6, 0.2])
    missing = np.array([False, False, True, False])
    ids = np.array(["b", "a", "m", "c"])
    prediction, floor, excluded = correction._prediction_at_stored_k(p, missing, ids, 2)
    np.testing.assert_array_equal(prediction, np.array([1, 1, 1, 0], dtype=np.int8))
    assert floor == 0.9
    assert excluded == 0.2
    try:
        correction._prediction_at_stored_k(p, missing, ids, 1)
    except RuntimeError as error:
        assert "splits an exact probability tie" in str(error)
    else:
        raise AssertionError("Stored K must not split tied probability rows")
