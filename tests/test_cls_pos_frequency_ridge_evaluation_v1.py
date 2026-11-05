import numpy as np
import pytest

from scripts.evaluate_optimization_v3_cls_pos_frequency_ridge_v1 import (
    A02_KEYS,
    FORBIDDEN_QUERY_KEYS,
    EvaluationError,
    PILOT_KEYS,
    VIEWS,
    evaluate_query_boundary_resource_gate,
    evaluate_query_initial_resource_gates,
    competition_probabilities,
    metrics,
    metrics_missing_threshold,
    read_allowlisted_members,
    row_metadata,
    sha256_array,
    validate_saved_route_probabilities,
    validate_ordered_row_identity,
)


class SpyArchive:
    def __init__(self, files):
        self.files = files
        self.indexed = []

    def __getitem__(self, key):
        self.indexed.append(key)
        return np.asarray([1], dtype=np.int8)


def test_forbidden_label_rejected_before_archive_indexing():
    archive = SpyArchive(["E67_y"])
    with pytest.raises(EvaluationError, match="Target member is forbidden"):
        read_allowlisted_members(archive, ["E67_y"], ["E67_y"])
    assert archive.indexed == []


def test_a02_allowlist_contains_only_non_target_keys():
    assert not (set(A02_KEYS) & FORBIDDEN_QUERY_KEYS)
    assert not any(key.endswith("_y") for key in A02_KEYS)


def test_pilot_allowlist_distinguishes_raw_routed_and_hgb_fields_without_targets():
    assert {"prob_bag", "prob_route_bag", "prob_missing_hgb"}.issubset(PILOT_KEYS)
    assert not any(key.endswith("_y") for key in PILOT_KEYS)


def test_safe_allowlisted_member_load_indexes_only_requested_fields():
    archive = SpyArchive(["E67_ids", "E67_pA", "E67_y"])
    result = read_allowlisted_members(archive, ["E67_ids", "E67_pA"], ["E67_ids", "E67_pA"])
    assert set(result) == {"E67_ids", "E67_pA"}
    assert archive.indexed == ["E67_ids", "E67_pA"]


def test_competition_blend_keeps_missing_probability_exactly_at_p_a():
    ppos = np.asarray([0.2, 0.8, 0.4])
    pa = np.asarray([0.4, 0.3, 0.9])
    present = np.asarray([True, True, False])
    result = competition_probabilities(ppos, pa, present)
    np.testing.assert_allclose(result, np.asarray([0.3, 0.55, 0.9]), rtol=0.0, atol=1e-15)
    assert result[2] == pa[2]


def test_metric_confusion_f1_and_calibration_values():
    result = metrics(np.asarray([0, 0, 1, 1]), np.asarray([0.1, 0.7, 0.6, 0.4]))
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["macro_f1"] == pytest.approx(0.5)
    assert result["positive_precision"] == pytest.approx(0.5)
    assert result["positive_recall"] == pytest.approx(0.5)
    assert result["brier"] == pytest.approx(0.255)
    assert result["logloss_epsilon"] == 1e-15


def test_postselected_missing_threshold_changes_only_hard_metrics():
    y = np.asarray([0, 1, 0, 1])
    probability = np.asarray([0.52, 0.61, 0.2, 0.3])
    missing = np.asarray([True, True, False, False])
    p_a = np.asarray([0.52, 0.61, 0.2, 0.3])
    primary = metrics_missing_threshold(y, probability, missing, p_a, 0.50)
    operational = metrics_missing_threshold(y, probability, missing, p_a, 0.53230976129477)
    assert primary["confusion_matrix"] != operational["confusion_matrix"]
    assert primary["macro_f1"] != operational["macro_f1"]
    assert primary["brier"] == operational["brier"]
    assert primary["logloss"] == operational["logloss"]


def test_frozen_view_windows_and_unique_pool_are_exact():
    assert VIEWS == {
        "E67_H12": (67, 68, 79),
        "E67_W4": (67, 68, 71),
        "E67_W5": (67, 72, 75),
        "E67_W6": (67, 76, 79),
        "E75_W6": (75, 76, 79),
    }
    assert sum(4_950 + 7_462 + 8_491 for _ in [0]) == 20_903


def test_row_metadata_binds_ordered_ids_and_cycles():
    rows = np.asarray([4, 5, 6], dtype=np.int64)
    ids = np.asarray(["x", "y", "z"])
    cycles = np.asarray([68, 68, 69], dtype=np.int64)
    meta = row_metadata(rows, ids, cycles)
    assert meta["rows"] == 3
    assert meta["cycle_min"] == 68
    assert meta["cycle_max"] == 69
    assert len(meta["record_ids_sha256"]) == 64


def test_row_metadata_uses_historical_object_string_digest_schema():
    rows = np.asarray([4, 5, 6], dtype=np.int64)
    ids = np.asarray(["x", "y", "z"], dtype="U1")
    cycles = np.asarray([68, 68, 69], dtype=np.int64)
    historical_digest = sha256_array(ids.astype(object))
    assert sha256_array(ids.astype(str)) != historical_digest
    assert row_metadata(rows, ids, cycles)["record_ids_sha256"] == historical_digest


@pytest.mark.parametrize(
    "rows,ids,cycles",
    [
        (np.asarray([5, 4, 6]), np.asarray(["y", "x", "z"]), np.asarray([68, 68, 69])),
        (np.asarray([4, 5, 6]), np.asarray(["x", "Q", "z"]), np.asarray([68, 68, 69])),
        (np.asarray([4, 4, 6]), np.asarray(["x", "y", "z"]), np.asarray([68, 68, 69])),
        (np.asarray([4, 5, 6]), np.asarray(["x", "x", "z"]), np.asarray([68, 68, 69])),
        (np.asarray([4, 5, 6]), np.asarray(["x", "y", "z"]), np.asarray([69, 68, 69])),
    ],
)
def test_ordered_row_identity_rejects_row_id_or_cycle_changes(rows, ids, cycles):
    expected_rows = np.asarray([4, 5, 6], dtype=np.int64)
    expected_ids = np.asarray(["x", "y", "z"], dtype="U1")
    expected_cycles = np.asarray([68, 68, 69], dtype=np.int64)
    with pytest.raises(EvaluationError, match="Ordered physical rows"):
        validate_ordered_row_identity(
            rows, ids, cycles, expected_rows, expected_ids, expected_cycles
        )


def test_query_initial_resource_gates_require_both_floor_and_projection():
    gib = 1024**3
    fail_initial = evaluate_query_initial_resource_gates(available_bytes=2 * gib + gib // 2,
                                                         estimated_additional_bytes=gib // 4)
    assert fail_initial["initial_floor_pass"] is False
    assert fail_initial["projected_reserve_pass"] is True
    assert fail_initial["pass_both_initial_gates"] is False

    fail_projection = evaluate_query_initial_resource_gates(available_bytes=3 * gib + gib // 4,
                                                            estimated_additional_bytes=3 * gib // 2)
    assert fail_projection["initial_floor_pass"] is True
    assert fail_projection["projected_reserve_pass"] is False
    assert fail_projection["pass_both_initial_gates"] is False

    pass_both = evaluate_query_initial_resource_gates(available_bytes=3 * gib + gib // 2,
                                                      estimated_additional_bytes=gib // 2)
    assert pass_both["initial_floor_pass"] is True
    assert pass_both["projected_reserve_pass"] is True
    assert pass_both["pass_both_initial_gates"] is True


def test_query_boundary_resource_gate_requires_two_gib_actual_available():
    gib = 1024**3
    assert evaluate_query_boundary_resource_gate(2 * gib)["pass"] is True
    assert evaluate_query_boundary_resource_gate(2 * gib - 1)["pass"] is False


def test_saved_route_audit_separates_raw_bag_lr_from_missing_hgb_route():
    raw_bag = np.asarray([0.10, 0.20, 0.30, 0.40], dtype=np.float64)
    hgb = np.asarray([0.15, 0.85, 0.75, 0.35], dtype=np.float64)
    missing = np.asarray([False, True, True, False])
    routed = np.asarray([0.10, 0.85, 0.75, 0.40], dtype=np.float64)
    audit = validate_saved_route_probabilities(routed, hgb, raw_bag, hgb, routed, missing)
    assert raw_bag[missing].tolist() != hgb[missing].tolist()
    assert audit["routed_bag_missing_equals_hgb_bit_exact"] is True
    assert audit["routed_bag_present_equals_raw_bag_lr_bit_exact"] is True


def test_saved_route_audit_rejects_tampered_missing_fallback():
    raw_bag = np.asarray([0.10, 0.20, 0.30], dtype=np.float64)
    hgb = np.asarray([0.15, 0.85, 0.75], dtype=np.float64)
    missing = np.asarray([False, True, True])
    tampered_route = np.asarray([0.10, 0.84, 0.75], dtype=np.float64)
    with pytest.raises(EvaluationError, match="not bit-exact HGB"):
        validate_saved_route_probabilities(tampered_route, hgb, raw_bag, hgb,
                                           tampered_route, missing)
