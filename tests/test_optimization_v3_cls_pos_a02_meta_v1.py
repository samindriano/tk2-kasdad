from __future__ import annotations

import hashlib
import json
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
    assert result["resume_binding_check"]["changed_id_feature_source_and_complete_hash_rejections"] == 4


def test_resume_binding_rejects_changed_row_feature_source_and_complete_hash():
    result = meta.synthetic_resume_binding_check()
    assert result["exact_resume"] == "PASS"
    assert result["changed_id_feature_source_and_complete_hash_rejections"] == 4
    assert result["valid_persisted_receipt_missing_complete"] == "RECOVER_COMPLETION_EVENT"


def test_dangling_started_fit_is_terminal_and_cannot_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(meta, "FIT_ROOT", tmp_path / "fits")
    signature = {"fit_id": "D1_PAIR", "arm": "PAIR", "approval_sha256": "approval"}
    signature_sha = meta.canonical_sha256(signature)
    reservation_id = "00000000-0000-4000-8000-000000000001"
    reserve = {
        "event": "RESERVE", "experiment_id": meta.EXPERIMENT, "fit_id": "D1_PAIR",
        "reservation_id": reservation_id, "cap": 6, "fit_ids_allowlist": list(meta.FIT_IDS),
        "fit_signature": signature, "fit_signature_sha256": signature_sha,
        "arm": "PAIR", "approval_sha256": "approval",
    }
    start = {
        "event": "FIT_START", "experiment_id": meta.EXPERIMENT, "fit_id": "D1_PAIR",
        "reservation_id": reservation_id, "fit_signature_sha256": signature_sha,
        "approval_sha256": "approval",
    }
    with pytest.raises(meta.MetaError, match="no durable receipt; retry prohibited"):
        meta._validate_fit_ledger([reserve, start])


def test_saved_model_iteration_guard_rejects_max_iter_state(tmp_path):
    path = tmp_path / "state.npz"
    payload = meta._npz_bytes(
        coef=np.asarray([[0.1, 0.2]]), intercept=np.asarray([0.0]),
        classes=np.asarray([0, 1], dtype=np.int8), n_iter=np.asarray([meta.MAX_ITER], dtype=np.int32),
    )
    path.write_bytes(payload)
    receipt = {
        "model_state_file_sha256": hashlib.sha256(payload).hexdigest(),
        "feature_width": 2, "n_iter": [meta.MAX_ITER], "convergence_warnings": [],
    }
    with pytest.raises(meta.MetaError, match="iteration count"):
        meta._read_model_state(path, receipt)


def test_both_head_states_and_predictions_exist_before_eval_target_read(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    fit_root = run_dir / "fits"
    report_dir = tmp_path / "reports"
    monkeypatch.setattr(meta, "RUN_DIR", run_dir)
    monkeypatch.setattr(meta, "FIT_ROOT", fit_root)
    monkeypatch.setattr(meta, "REPORT_DIR", report_dir)
    monkeypatch.setattr(meta, "resource_gate", lambda: {"status": "PASS"})
    fitted = set()
    label_calls = 0

    def fake_fit(fit_id, arm, x, y, ids, cycles, approval_sha256):
        state = {
            "coef": np.full((1, x.shape[1]), 0.1, dtype=np.float64),
            "intercept": np.asarray([0.0], dtype=np.float64),
            "classes": np.asarray([0, 1], dtype=np.int8),
            "n_iter": np.asarray([4], dtype=np.int32),
        }
        state_path, receipt_path, _ = meta._fit_paths(fit_id)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = meta._npz_bytes(**state)
        state_path.write_bytes(payload)
        receipt_path.write_text(json.dumps({"status": "FIT_SAVED"}), encoding="utf-8")
        fitted.add(fit_id)
        return state

    def fake_labels(ids, cycles):
        nonlocal label_calls
        label_calls += 1
        if label_calls == 1:
            return np.tile(np.asarray([0, 1], dtype=np.int8), len(ids) // 2)
        assert fitted == {"D1_PAIR", "D1_SCALAR"}
        for fit_id in fitted:
            state_path, receipt_path, _ = meta._fit_paths(fit_id)
            assert state_path.is_file() and receipt_path.is_file()
        assert (run_dir / "stages" / "D1" / "predictions.npz").is_file()
        return np.tile(np.asarray([0, 1], dtype=np.int8), len(ids) // 2)

    monkeypatch.setattr(meta, "_fit_one", fake_fit)
    monkeypatch.setattr(meta, "_load_train_labels", fake_labels)

    def data(prefix, n, cycle):
        p_pos = np.linspace(.1, .9, n)
        p_a02 = np.linspace(.9, .1, n)
        missing = (np.arange(n) % 10 == 0)
        mix = np.where(missing, p_pos, .5 * (p_pos + p_a02))
        return {
            "ids": np.asarray([f"{prefix}{i:04d}" for i in range(n)], dtype="U9"),
            "rows": np.arange(n, dtype=np.int64),
            "cycles": np.full(n, cycle, dtype=np.int16),
            "pP": p_pos, "pA": p_a02, "missing": missing, "MIX50": mix,
            "group_availability": np.where(missing, "missing", "present").astype("U7"),
            "group_text_load": np.full(n, "Q1", dtype="U2"),
            "group_entry_friction": np.full(n, "0", dtype="U1"),
        }

    fit_data = data("F", 600, 68)
    eval_data = data("E", 300, 72)
    result = meta._fit_stage("D1", fit_data, eval_data, "approval")
    assert label_calls == 2
    assert result["target_reads_started_after_both_fit_states_persisted"] is True


def test_exact_fit_allowlist_contains_six_unique_fits():
    assert len(meta.FIT_IDS) == meta.MAX_OFFICIAL_FITS == 6
    assert len(set(meta.FIT_IDS)) == 6
    assert set(meta.FIT_IDS) == {
        "D1_PAIR", "D1_SCALAR", "D2_PAIR", "D2_SCALAR", "FINAL_PAIR", "FINAL_SCALAR",
    }
