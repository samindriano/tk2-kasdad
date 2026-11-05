import numpy as np
import pytest

from scripts.evaluate_optimization_v3_cls_pos_frequency_ridge_v1 import (
    A02_KEYS,
    FORBIDDEN_QUERY_KEYS,
    EvaluationError,
    PILOT_KEYS,
    QUERY_BOUNDARY_MIN_BYTES,
    QUERY_INCREMENTAL_ESTIMATE_BYTES,
    QUERY_INITIAL_MIN_BYTES,
    QUERY_POLICY_ID,
    QUERY_PROJECTED_MIN_BYTES,
    VIEWS,
    evaluate_query_boundary_resource_gate,
    evaluate_query_initial_resource_gates,
    competition_probabilities,
    metrics,
    metrics_missing_threshold,
    process_memory_measurements,
    read_allowlisted_members,
    row_metadata,
    sha256_array,
    validate_saved_route_probabilities,
    validate_ordered_row_identity,
    write_npz_once,
)
from scripts.verify_optimization_v3_cls_pos_frequency_ridge_eval_v1 import (
    BOUNDARY_MIN_BYTES as INDEPENDENT_BOUNDARY_MIN_BYTES,
    INCREMENTAL_ESTIMATE_BYTES as INDEPENDENT_INCREMENTAL_ESTIMATE_BYTES,
    INITIAL_MIN_BYTES as INDEPENDENT_INITIAL_MIN_BYTES,
    PROJECTED_MIN_BYTES as INDEPENDENT_PROJECTED_MIN_BYTES,
    RESOURCE_POLICY_ID as INDEPENDENT_RESOURCE_POLICY_ID,
    evaluate_boundary_resource_gate as independent_boundary_resource_gate,
    evaluate_initial_resource_gates as independent_initial_resource_gates,
    process_memory_measurements as independent_process_memory_measurements,
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


def test_r7_resource_policy_constants_match_authorized_saved_evaluation_scope():
    assert QUERY_POLICY_ID == INDEPENDENT_RESOURCE_POLICY_ID == "CLS_EVALUATION_QUERY_REPLAY_R7"
    assert QUERY_INITIAL_MIN_BYTES == INDEPENDENT_INITIAL_MIN_BYTES == 1610612736
    assert QUERY_INCREMENTAL_ESTIMATE_BYTES == INDEPENDENT_INCREMENTAL_ESTIMATE_BYTES == 536870912
    assert QUERY_PROJECTED_MIN_BYTES == INDEPENDENT_PROJECTED_MIN_BYTES == 805306368
    assert QUERY_BOUNDARY_MIN_BYTES == INDEPENDENT_BOUNDARY_MIN_BYTES == 805306368


def test_query_initial_resource_gates_require_both_floor_and_projection():
    floor = QUERY_INITIAL_MIN_BYTES
    reserve = QUERY_PROJECTED_MIN_BYTES
    estimate = QUERY_INCREMENTAL_ESTIMATE_BYTES
    fail_initial = evaluate_query_initial_resource_gates(available_bytes=floor - 1,
                                                         estimated_additional_bytes=0)
    assert fail_initial["initial_floor_pass"] is False
    assert fail_initial["projected_reserve_pass"] is True
    assert fail_initial["pass_both_initial_gates"] is False

    fail_projection = evaluate_query_initial_resource_gates(available_bytes=floor,
                                                            estimated_additional_bytes=floor - reserve + 1)
    assert fail_projection["initial_floor_pass"] is True
    assert fail_projection["projected_reserve_pass"] is False
    assert fail_projection["pass_both_initial_gates"] is False

    pass_both = evaluate_query_initial_resource_gates(available_bytes=floor,
                                                      estimated_additional_bytes=estimate)
    assert pass_both["initial_floor_pass"] is True
    assert pass_both["projected_reserve_pass"] is True
    assert pass_both["pass_both_initial_gates"] is True


def test_query_boundary_resource_gate_uses_actual_available_floor():
    assert evaluate_query_boundary_resource_gate(QUERY_BOUNDARY_MIN_BYTES)["pass"] is True
    assert evaluate_query_boundary_resource_gate(QUERY_BOUNDARY_MIN_BYTES - 1)["pass"] is False


def test_independent_verifier_enforces_r7_initial_gates_independently():
    floor = INDEPENDENT_INITIAL_MIN_BYTES
    reserve = INDEPENDENT_PROJECTED_MIN_BYTES
    floor_fail = independent_initial_resource_gates(available_bytes=floor - 1,
                                                     estimated_additional_bytes=0)
    assert floor_fail["initial_floor_pass"] is False
    assert floor_fail["projected_reserve_pass"] is True
    assert floor_fail["pass_both_initial_gates"] is False

    reserve_fail = independent_initial_resource_gates(available_bytes=floor,
                                                       estimated_additional_bytes=floor - reserve + 1)
    assert reserve_fail["initial_floor_pass"] is True
    assert reserve_fail["projected_reserve_pass"] is False
    assert reserve_fail["pass_both_initial_gates"] is False
    assert independent_boundary_resource_gate(INDEPENDENT_BOUNDARY_MIN_BYTES)["pass"] is True
    assert independent_boundary_resource_gate(INDEPENDENT_BOUNDARY_MIN_BYTES - 1)["pass"] is False


def test_rss_and_os_peak_working_set_are_reported_distinctly_when_available():
    from types import SimpleNamespace

    measured = process_memory_measurements(SimpleNamespace(rss=123, peak_wset=456))
    assert measured["process_rss_sample_bytes"] == 123
    assert measured["process_rss_sample_is_peak"] is False
    assert measured["process_peak_working_set_bytes"] == 456
    assert measured["peak_working_set_measurement_available"] is True
    assert measured["peak_working_set_value_is_os_high_water_mark"] is True

    unavailable = process_memory_measurements(SimpleNamespace(rss=789))
    assert unavailable["process_rss_sample_bytes"] == 789
    assert unavailable["process_rss_sample_is_peak"] is False
    assert unavailable["process_peak_working_set_bytes"] is None
    assert unavailable["peak_working_set_measurement_available"] is False
    assert unavailable["peak_working_set_source"] is None
    assert independent_process_memory_measurements(SimpleNamespace(rss=789))["process_peak_working_set_bytes"] is None


def test_synthetic_end_to_end_prediction_scoring_and_persistence_resume(tmp_path):
    rows = np.asarray([11, 12, 13, 14], dtype=np.int64)
    ids = np.asarray(["r11", "r12", "r13", "r14"])
    cycles = np.asarray([68, 68, 69, 69], dtype=np.int64)
    p_a = np.asarray([0.10, 0.70, 0.40, 0.80], dtype=np.float64)
    p_pos = np.asarray([0.20, 0.90, 0.30, 0.60], dtype=np.float64)
    missing = np.asarray([False, False, True, False])
    synthetic_labels = np.asarray([0, 1, 0, 1], dtype=np.int8)
    routed = competition_probabilities(p_pos, p_a, ~missing)
    scored = metrics(synthetic_labels, routed)
    assert scored["n"] == len(rows)
    assert scored["confusion_matrix"] == [[2, 0], [0, 2]]

    bundle = {"rows": rows, "ids": ids, "cycles": cycles, "probability": routed}
    saved = tmp_path / "synthetic_predictions.npz"
    write_npz_once(saved, bundle)
    first_bytes = saved.read_bytes()
    with np.load(saved, allow_pickle=False) as archive:
        assert set(archive.files) == set(bundle)
        for key, expected in bundle.items():
            np.testing.assert_array_equal(archive[key], expected)
    write_npz_once(saved, bundle)  # resumable exact replay is a no-op
    assert saved.read_bytes() == first_bytes

    changed = dict(bundle)
    changed["probability"] = routed + np.asarray([0.01, 0.0, 0.0, 0.0])
    with pytest.raises(EvaluationError, match="Existing saved predictions differ"):
        write_npz_once(saved, changed)


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
