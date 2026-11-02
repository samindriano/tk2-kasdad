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


def test_scope_row_sort_keeps_ids_cycles_and_predictions_aligned():
    rows, ids, cycles, predictions = correction._sort_scope_arrays(
        np.array([8, 2, 5]), np.array(["r8", "r2", "r5"]), np.array([75, 60, 66]),
        {"B01": {"plugin": np.array([1, 0, 1]), "fixed_half": np.array([0, 0, 1])}})
    np.testing.assert_array_equal(rows, [2, 5, 8])
    np.testing.assert_array_equal(ids, ["r2", "r5", "r8"])
    np.testing.assert_array_equal(cycles, [60, 66, 75])
    np.testing.assert_array_equal(predictions["B01"]["plugin"], [0, 1, 1])
    np.testing.assert_array_equal(predictions["B01"]["fixed_half"], [0, 1, 0])
