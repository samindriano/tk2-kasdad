from __future__ import annotations

import numpy as np
import pytest

from scripts import optimization_v3_cls_pos_a02_meta_v1 as meta


def _small_oof(ids: list[str], cycles: list[int]) -> dict[str, np.ndarray]:
    n = len(ids)
    p_pos = np.linspace(.2, .8, n)
    p_a02 = np.linspace(.75, .25, n)
    missing = np.zeros(n, dtype=bool)
    mix = .5 * (p_pos + p_a02)
    return {
        "ids": np.asarray(ids, dtype="U8"),
        "rows": np.arange(n, dtype=np.int64),
        "cycles": np.asarray(cycles, dtype=np.int16),
        "pP": p_pos,
        "pA": p_a02,
        "missing": missing,
        "MIX50": mix,
        "group_availability": np.asarray(["present"] * n, dtype="U7"),
        "group_text_load": np.asarray(["Q1"] * n, dtype="U2"),
        "group_entry_friction": np.asarray(["0"] * n, dtype="U1"),
    }


def test_clipped_logit_is_finite_at_probability_edges():
    result = meta.clipped_logit(np.asarray([0.0, .5, 1.0]))
    assert np.isfinite(result).all()
    assert result[0] < 0 < result[2]
    assert result[1] == 0


def test_pair_and_scalar_feature_definitions_are_exact():
    p_pos = np.asarray([.2, .8])
    p_a02 = np.asarray([.4, .6])
    mix = np.asarray([.3, .7])
    pair = meta.feature_matrix(p_pos, p_a02, mix, "PAIR")
    scalar = meta.feature_matrix(p_pos, p_a02, mix, "SCALAR")
    assert pair.shape == (2, 2)
    assert np.array_equal(pair[:, 0], meta.clipped_logit(p_pos))
    assert np.array_equal(pair[:, 1], meta.clipped_logit(p_a02))
    assert np.array_equal(scalar[:, 0], meta.clipped_logit(mix))


def test_support_gate_enforces_rows_and_both_classes():
    good = np.asarray([0] * 400 + [1] * 100, dtype=np.int8)
    bad = np.asarray([0] * 450 + [1] * 49, dtype=np.int8)
    assert meta.fit_support(good)["status"] == "PASS"
    assert meta.fit_support(bad)["status"] == "BLOCKED_UNSUPPORTED_SELECTION"


def test_missing_stream_route_is_exact_p_pos_threshold():
    p_pos = np.asarray([.49, .5, .8])
    p_a02 = np.asarray([.99, .01, .1])
    missing = np.asarray([True, True, False])
    mix = np.where(missing, p_pos, .5 * (p_pos + p_a02))
    state = {"coef": np.asarray([[0.2, -0.1]]), "intercept": np.asarray([.05])}
    result = meta.policy_probabilities(p_pos, p_a02, missing, mix, "PAIR", state)
    assert np.array_equal(result[missing], p_pos[missing])
    assert np.array_equal(result[missing] >= .5, np.asarray([False, True]))


def test_final_stitch_excludes_duplicate_e67_w6_rows():
    e67_cycles = list(range(68, 80))
    e75_cycles = list(range(76, 80))
    e67 = _small_oof([f"A{i}" for i in range(12)], e67_cycles)
    e75 = _small_oof([f"A{i}" for i in range(8, 12)], e75_cycles)
    stitched = meta.make_unique_stitch({"67": e67, "75": e75}, expected_rows=12)
    assert stitched["ids"].tolist() == [f"A{i}" for i in range(12)]
    assert stitched["cycles"].tolist() == e67_cycles[:8] + e75_cycles


def test_subgroup_below_support_is_unknown_not_reported_as_metric():
    y = np.asarray([0] * 90 + [1] * 10, dtype=np.int8)
    p = np.full(100, .4)
    groups = {"text_load": np.asarray(["Q1"] * 100, dtype="U2")}
    result = meta.metric_bundle(y, p, groups=groups)
    assert result["groups"]["text_load"]["Q1"]["status"] == "UNKNOWN_UNSUPPORTED"
    assert result["groups"]["text_load"]["Q1"]["metrics"] is None


def test_pair_error_summary_counts_correctness_changes():
    y = np.asarray([0, 1, 1, 0], dtype=np.int8)
    base = np.asarray([0, 0, 1, 1], dtype=np.int8)
    candidate = np.asarray([1, 1, 0, 1], dtype=np.int8)
    counts = meta.pair_error_summary(y, candidate, base)
    assert counts == {
        "changed_labels": 3,
        "changed_to_correct": 1,
        "changed_to_wrong": 2,
        "changed_but_both_same_correctness": 0,
    }


def test_root_approval_must_bind_every_prefit_artifact():
    freeze = {
        "experiment_id": meta.EXPERIMENT,
        "prefit_commit": "abc123",
        "receipt_sha256": "receipt",
        "source_hashes": {
            "protocol": "p", "script": "s", "tests": "t",
            "research_next_contract": "c", "r5_script": "r5",
        },
    }
    approval = {
        "status": "PASS", "scope": meta.EXPERIMENT, "prefit_commit": "abc123",
        "prefit_freeze_receipt_sha256": "receipt", "prefit_freeze_file_sha256": "file",
        "protocol_sha256": "p", "script_sha256": "s", "tests_sha256": "t",
        "research_next_contract_sha256": "c", "r5_script_sha256": "r5",
    }
    meta.validate_root_approval_payload(approval, freeze, "file")
    approval["script_sha256"] = "tampered"
    with pytest.raises(meta.MetaError, match="script_sha256"):
        meta.validate_root_approval_payload(approval, freeze, "file")


def test_resource_gate_keeps_four_gib_floor_and_two_gib_reserve():
    assert meta.resource_gate(available_bytes=4 * meta.GIB)["status"] == "PASS"
    with pytest.raises(meta.MetaError, match="RAM gate"):
        meta.resource_gate(available_bytes=4 * meta.GIB - 1)
    assert meta.SAFETY_RESERVE_RAM == 2 * meta.GIB


def test_synthetic_lifecycle_persists_and_reloads_without_refit():
    result = meta.synthetic_lifecycle()
    assert result["status"] == "PASS_SYNTHETIC_LIFECYCLE_ONLY"
    assert result["synthetic_official_fits"] == 0
    assert result["synthetic_lr_fits"] == 2
    assert result["persist_reload_bit_exact"] is True
    assert result["resume_reuses_saved_state_without_refit"] is True
    assert result["missing_route_bit_exact"] is True


def test_exact_fit_allowlist_contains_six_unique_fits():
    assert len(meta.FIT_IDS) == meta.MAX_OFFICIAL_FITS == 6
    assert len(set(meta.FIT_IDS)) == 6
    assert set(meta.FIT_IDS) == {
        "D1_PAIR", "D1_SCALAR", "D2_PAIR", "D2_SCALAR", "FINAL_PAIR", "FINAL_SCALAR",
    }
