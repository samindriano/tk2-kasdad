from __future__ import annotations

import json

import numpy as np
import pytest

import scripts.reg_final_wave_followup_v2 as reg
import scripts.verify_reg_final_wave_followup_v2 as independent


def _row_identity(n: int = 4) -> dict[str, np.ndarray]:
    return {
        "rows": np.arange(100, 100 + n, dtype=np.int64),
        "ids": np.asarray([f"row-{i}" for i in range(n)], dtype="U8"),
        "origins": np.full(n, 67, dtype=np.int8),
        "indices": np.arange(n, dtype=np.int64),
    }


def _fit_inputs():
    y = np.asarray([2., 4., 6., 8.])
    p = np.asarray([1., 2., 3., 4.])
    cycles = np.asarray([68, 69, 70, 71], dtype=np.int16)
    return y, p, cycles, _row_identity(len(y))


def _freeze(tmp_path):
    path = tmp_path / "prefit.json"
    path.write_text("synthetic-freeze", encoding="utf-8")
    return path, {"source_commit": "synthetic-commit", "source_hashes": {"runner": "a"},
                  "input_hashes": {"synthetic": {"sha256": "b"}}}


def test_h8_formula_and_mean_one():
    cycles = np.asarray([68, 69, 71, 75], dtype=np.int16)
    weights, info = reg.recency_weights(cycles, 75)
    expected_raw = 2.0 ** (-(75 - cycles) / 8.0)
    expected = expected_raw / expected_raw.mean()
    np.testing.assert_allclose(weights, expected, rtol=0, atol=1e-15)
    assert weights.mean() == pytest.approx(1.0)
    assert info["fit_rows"] == 4
    assert info["effective_sample_size"] <= 4


def test_diagnostic_support_and_identity_uniqueness():
    cycles = np.resize(np.arange(68, 80, dtype=np.int16), reg.EXPECTED_SUPPORT_ROWS)
    n = len(cycles)
    identity = {"oof": {"cycles": cycles, "rows": np.arange(n),
                        "ids": np.asarray([f"id-{i}" for i in range(n)], dtype="U12")}}
    support = reg.diagnostic_support(identity)
    assert support["D1"]["fit_rows"] >= reg.DIAGNOSTIC_MIN_ROWS
    assert support["D1"]["score_rows"] >= reg.DIAGNOSTIC_MIN_ROWS
    assert support["D2"]["fit_rows"] >= reg.DIAGNOSTIC_MIN_ROWS
    bad = {"oof": dict(identity["oof"])}
    bad["oof"]["ids"] = identity["oof"]["ids"].copy()
    bad["oof"]["ids"][1] = bad["oof"]["ids"][0]
    with pytest.raises(reg.GateError, match="not unique"):
        reg.diagnostic_support(bad)


def test_fit_receipt_binds_row_origin_and_bundle_index(tmp_path, monkeypatch):
    prefit, freeze = _freeze(tmp_path)
    monkeypatch.setattr(reg, "PREFIT_FREEZE", prefit)
    y, p, cycles, row_id = _fit_inputs()
    a = reg.fit_signature_payload("D1_GLOBAL", y, p, cycles, None, row_id, freeze, "pass-sha")
    changed = {key: value.copy() for key, value in row_id.items()}
    changed["indices"][0] += 50
    b = reg.fit_signature_payload("D1_GLOBAL", y, p, cycles, None, changed, freeze, "pass-sha")
    assert a["physical_rows_sha256"] == reg.array_sha256(row_id["rows"])
    assert a["record_ids_sha256"] == reg.array_sha256(row_id["ids"])
    assert a["original_bundle_indices_sha256"] != b["original_bundle_indices_sha256"]


def test_mismatched_source_pass_is_rejected_before_any_reader(monkeypatch):
    freeze = {"source_commit": "commit-a", "source_hashes": {"runner": "one"},
              "input_hashes": {"train": {"sha256": "train-hash"}}}
    approval = reg.sign_body({"verdict": "PASS", "experiment_id": reg.EXPERIMENT,
                              "source_commit": "commit-b", "prefit_freeze_sha256": "freeze-sha",
                              "source_hashes": freeze["source_hashes"],
                              "input_hashes": freeze["input_hashes"],
                              "authorized_fit_ids": list(reg.DIAGNOSTIC_FIT_IDS)})
    reads = []
    monkeypatch.setattr(reg, "read_scored_query", lambda *args, **kwargs: reads.append(True))
    with pytest.raises(reg.GateError, match="does not bind"):
        reg.validate_source_pass_body(approval, freeze, "freeze-sha")
    assert reads == []


def test_ledger_cap_and_write_once(tmp_path, monkeypatch):
    ledger = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(reg, "LEDGER", ledger)
    for fit_id in reg.FIT_IDS:
        reg.append_jsonl(ledger, {"event": "RESERVE", "fit_id": fit_id,
                                  "reservation_id": f"reservation-{fit_id}",
                                  "fit_input": {"fit_id": fit_id},
                                  "source_commit": "synthetic-commit",
                                  "root_source_pass_sha256": "synthetic-pass"})
    assert reg._next_unreserved_fit(ledger) is None
    path = tmp_path / "immutable.json"
    first = reg.write_json_once(path, {"value": 1})
    assert reg.write_json_once(path, {"value": 1}) == first
    with pytest.raises(reg.GateError, match="Refusing to overwrite"):
        reg.write_json_once(path, {"value": 2})


def test_synthetic_receipt_is_idempotent(tmp_path, monkeypatch):
    path = tmp_path / "SYNTHETIC_QA.json"
    monkeypatch.setattr(reg, "SYNTHETIC_QA", path)
    stable_body = {"experiment_id": reg.EXPERIMENT, "status": "PASS_SYNTHETIC_ONLY",
                   "official_labels_accessed": False, "official_lp_starts": 0,
                   "test_or_sample_accessed": False}
    monkeypatch.setattr(reg, "synthetic_qa_body", lambda: stable_body)
    first = reg.run_synthetic_qa()
    second = reg.run_synthetic_qa()
    assert first["receipt_sha256"] == second["receipt_sha256"]
    monkeypatch.setattr(reg, "synthetic_qa_body", lambda: {**stable_body, "different": True})
    with pytest.raises(reg.GateError, match="immutable synthetic QA differs"):
        reg.run_synthetic_qa()


def test_recovery_after_state_publish_does_not_refit(tmp_path, monkeypatch):
    ledger = tmp_path / "ledger.jsonl"
    run_dir = tmp_path / "run"
    prefit, freeze = _freeze(tmp_path)
    monkeypatch.setattr(reg, "LEDGER", ledger)
    monkeypatch.setattr(reg, "RUN_DIR", run_dir)
    monkeypatch.setattr(reg, "PREFIT_FREEZE", prefit)
    monkeypatch.setattr(reg, "resource_snapshot", lambda: {"status": "PASS", "available_physical_ram_bytes_measured": 9})
    monkeypatch.setattr(reg.v2, "one_job_gate", lambda: {"status": "PASS"})
    monkeypatch.setattr(reg, "_verify_root_source_pass", lambda _freeze: ({"verdict": "PASS"}, "pass-sha"))
    monkeypatch.setattr(reg, "git", lambda *args: "synthetic-commit" if args == ("rev-parse", "HEAD") else reg.BRANCH)
    calls = []

    def fake_solver(y, p, threads=2):
        calls.append(1)
        return {"success": True, "status": 0, "message": "synthetic", "intercept": 0.0,
                "slope": 2.0, "objective_l1_sum": 0.0}

    monkeypatch.setattr(reg, "solve_nonnegative_l1", fake_solver)
    y, p, cycles, row_id = _fit_inputs()
    append_original = reg.append_jsonl

    def fail_after_state(event_path, event):
        if event.get("event") == "LP_COMPLETE":
            raise OSError("synthetic interruption after state publication")
        append_original(event_path, event)

    monkeypatch.setattr(reg, "append_jsonl", fail_after_state)
    with pytest.raises(OSError, match="after state publication"):
        reg._record_or_reuse_fit("D1_GLOBAL", y, p, cycles, None, row_id, freeze, "pass-sha")
    state_path = reg._fit_file_path("D1_GLOBAL")
    assert state_path.is_file()
    monkeypatch.setattr(reg, "append_jsonl", append_original)
    recovered = reg._record_or_reuse_fit("D1_GLOBAL", y, p, cycles, None, row_id, freeze, "pass-sha")
    assert recovered["status"] == "LP_COMPLETE"
    assert len(calls) == 1
    events = reg.read_ledger(ledger)
    assert [event["event"] for event in events] == ["RESERVE", "LP_STARTED", "LP_COMPLETE"]
    assert events[-1]["recovered_after_state_publish"] is True


def test_reservation_without_start_resumes_same_bound_slot(tmp_path, monkeypatch):
    ledger = tmp_path / "ledger.jsonl"
    run_dir = tmp_path / "run"
    prefit, freeze = _freeze(tmp_path)
    monkeypatch.setattr(reg, "LEDGER", ledger)
    monkeypatch.setattr(reg, "RUN_DIR", run_dir)
    monkeypatch.setattr(reg, "PREFIT_FREEZE", prefit)
    monkeypatch.setattr(reg, "resource_snapshot", lambda: {"status": "PASS", "available_physical_ram_bytes_measured": 9})
    monkeypatch.setattr(reg.v2, "one_job_gate", lambda: {"status": "PASS"})
    monkeypatch.setattr(reg, "_verify_root_source_pass", lambda _freeze: ({"verdict": "PASS"}, "pass-sha"))
    monkeypatch.setattr(reg, "git", lambda *args: "synthetic-commit" if args == ("rev-parse", "HEAD") else reg.BRANCH)
    y, p, cycles, row_id = _fit_inputs()
    expected = reg.fit_signature_payload("D1_GLOBAL", y, p, cycles, None, row_id, freeze, "pass-sha")
    reg.append_jsonl(ledger, {"event": "RESERVE", "fit_id": "D1_GLOBAL",
                              "reservation_id": "same-reservation", "source_commit": freeze["source_commit"],
                              "root_source_pass_sha256": "pass-sha", "fit_input": expected})
    monkeypatch.setattr(reg, "solve_nonnegative_l1", lambda *args, **kwargs: {
        "success": True, "status": 0, "message": "synthetic", "intercept": 0.0,
        "slope": 2.0, "objective_l1_sum": 0.0})
    saved = reg._record_or_reuse_fit("D1_GLOBAL", y, p, cycles, None, row_id, freeze, "pass-sha")
    assert saved["status"] == "LP_COMPLETE"
    events = reg.read_ledger(ledger)
    assert [event["event"] for event in events] == ["RESERVE", "LP_STARTED", "LP_COMPLETE"]
    assert {event["reservation_id"] for event in events} == {"same-reservation"}


def test_prediction_and_subgroup_risk_audits():
    direct = np.asarray([0., 20., 40., 60.])
    pred = np.asarray([1., 21., 41., 61.])
    audit = reg.prediction_audit(direct, pred)
    assert audit["monotone_non_decreasing_in_direct_score"] is True
    assert audit["changed_rows"] == 4
    supported = {"status": "SUPPORTED", "mae": 3.0}
    metrics = {
        "causal_global_all_rows": {"groups": {"missing": supported, "target_le_5": supported,
                                               "target_ge_95": supported}},
        "causal_recency_h8_all_rows": {"groups": {"missing": {"status": "SUPPORTED", "mae": 3.3},
                                                   "target_le_5": {"status": "SUPPORTED", "mae": 3.2},
                                                   "target_ge_95": {"status": "SUPPORTED", "mae": 3.0}}},
    }
    flags = reg.subgroup_harm_flags(metrics)
    assert flags["causal_recency_h8_all_rows"]["missing"]["risk_flag_gt_0_25"] is True
    assert flags["causal_recency_h8_all_rows"]["target_ge_95"]["risk_flag_gt_0_25"] is False


def test_independent_verifier_recomputes_solver_objectives():
    y = np.asarray([2., 5., 8., 12.])
    p = np.asarray([1., 2., 3., 4.])
    intercept, slope = 1.0, 2.0
    residual = np.abs(y - intercept - slope * p)
    independent._verify_solver_objective(
        {"intercept": intercept, "slope": slope, "objective_l1_sum": float(residual.sum())},
        y, p, None)
    weights = np.asarray([.5, .75, 1.25, 1.5])
    weighted_sum = float(np.dot(weights, residual))
    independent._verify_solver_objective(
        {"intercept": intercept, "slope": slope, "weighted_l1_sum": weighted_sum,
         "solver_objective": weighted_sum, "objective_matches_recomputed": True},
        y, p, weights)
    with pytest.raises(independent.VerificationError, match="Weighted LP objective"):
        independent._verify_solver_objective(
            {"intercept": intercept, "slope": slope, "weighted_l1_sum": weighted_sum + 1,
             "solver_objective": weighted_sum, "objective_matches_recomputed": True},
            y, p, weights)
