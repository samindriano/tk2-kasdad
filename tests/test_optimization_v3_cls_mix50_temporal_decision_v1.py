from __future__ import annotations

import csv
import json
import subprocess

import numpy as np
import pytest
from sklearn.metrics import f1_score

import scripts.optimization_v3_cls_mix50_temporal_decision_v1 as decision
from scripts.optimization_v3_cls_mix50_temporal_decision_v1 import (
    GIB,
    _counts_for_rule,
    _evaluate_stage,
    _macro_f1_from_counts,
    _read_receipt,
    _reserve_selection,
    _resource_preflight,
    _select_stage,
    _stage_paths,
    _synthetic_end_to_end,
    _validate_prefit_commit,
    _validate_oof_boundaries,
    _write_exclusive_json,
    evaluate,
    file_sha256,
    select_threshold,
    threshold_candidates,
)


def _supported_case():
    p = np.tile(np.asarray([0.0, 0.0, 0.2, 0.5, 0.5, 0.8, 1.0]), 200)
    y = np.tile(np.asarray([0, 1, 0, 1, 0, 0, 1], dtype=np.int8), 200)
    missing = np.tile(np.asarray([False, False, True, False, False, False, True]), 200)
    return p, y, missing


def _evaluation_fixture(tmp_path):
    ids = []
    cycles = []
    labels = []
    per_cycle_index = []
    for cycle in range(68, 76):
        for index in range(300):
            ids.append(f"eval-{cycle}-{index}")
            cycles.append(cycle)
            labels.append(index % 2)
            per_cycle_index.append(index)
    ids = np.asarray(ids, dtype="U24")
    cycles = np.asarray(cycles, dtype=np.int16)
    labels = np.asarray(labels, dtype=np.int8)
    index = np.asarray(per_cycle_index)
    missing = index % 7 == 0
    p_pos = np.where(labels == 1, 0.72, 0.28)
    p_a = np.where(labels == 1, 0.68, 0.32)
    mix = np.where(missing, p_pos, 0.5 * p_pos + 0.5 * p_a)
    evaluator = {
        "ids": ids,
        "cycles": cycles,
        "MIX50": mix.astype(np.float64),
        "pP": p_pos.astype(np.float64),
        "pA": p_a.astype(np.float64),
        "missing": missing,
        "group_availability": np.where(missing, "missing", "present").astype("U7"),
        "group_text_load": np.where(index % 2 == 0, "Q1", "Q2").astype("U2"),
        "group_entry_friction": np.where(index % 3 == 0, "1", "0").astype("U1"),
    }
    train = tmp_path / "Train.csv"
    with train.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["record_id", "cycle_code", "artifact_signal"])
        writer.writerows(zip(ids.tolist(), cycles.tolist(), labels.tolist()))
    train_sha = file_sha256(train)
    source_hashes = {"train_csv_sha256": train_sha, "oof_bundle_sha256": "synthetic-bundle"}
    selection_mask = (cycles >= 68) & (cycles <= 71)
    evaluation_mask = (cycles >= 72) & (cycles <= 75)
    run_dir = tmp_path / "run"
    threshold = _select_stage(
        stage_id="D1_E67_W4_TO_W5",
        selector=evaluator,
        selection_mask=selection_mask,
        selection_cycles=(68, 71),
        train_path=train,
        train_sha=train_sha,
        run_dir=run_dir,
        source_hashes=source_hashes,
        resource_gate_fn=lambda: _resource_preflight(available_bytes=8 * GIB),
    )
    evaluation = _evaluate_stage(
        stage_id="D1_E67_W4_TO_W5",
        evaluator=evaluator,
        evaluation_mask=evaluation_mask,
        threshold_receipt=threshold,
        train_path=train,
        train_sha=train_sha,
        run_dir=run_dir,
    )
    return evaluator, evaluation_mask, threshold, evaluation, train, train_sha, run_dir


def _reseal_receipt(path, mutate):
    body = json.loads(path.read_text(encoding="utf-8"))
    body.pop("receipt_sha256")
    mutate(body)
    body["receipt_sha256"] = decision.canonical_sha256(body)
    path.write_text(json.dumps(body, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def test_threshold_candidates_preserve_ties_and_extreme_allocations():
    p = np.asarray([0.0, 0.0, 0.5, 0.5, 1.0])
    candidates = threshold_candidates(p)
    assert 0.0 in candidates
    assert 0.5 in candidates
    assert float(np.nextafter(1.0, np.inf)) in candidates
    for threshold in candidates:
        labels = (p >= threshold).astype(np.int8)
        assert np.all(labels[p == 0.0] == labels[p == 0.0][0])
        assert np.all(labels[p == 0.5] == labels[p == 0.5][0])


def test_adjacent_floats_use_upper_score_as_safe_cutpoint():
    low = np.float64(0.5)
    high = np.nextafter(low, np.inf)
    candidates = threshold_candidates(np.asarray([low, high]))
    assert high in candidates
    assert (np.asarray([low, high]) >= high).tolist() == [False, True]


def test_selector_matches_bruteforce_and_sklearn_macro_f1():
    p, y, missing = _supported_case()
    result = select_threshold(p, missing, y)
    assert result["status"] == "SELECTED"
    assert result["selection_fit_count"] == 1
    candidate_scores = []
    for threshold in threshold_candidates(p[~missing]):
        counts = _counts_for_rule(y, p, missing, float(threshold))
        own = _macro_f1_from_counts(counts)
        pred = (p >= 0.5).astype(np.int8)
        pred[~missing] = (p[~missing] >= threshold).astype(np.int8)
        sklearn_score = f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0)
        assert own == pytest.approx(sklearn_score, abs=1e-15)
        candidate_scores.append((float(threshold), own))
    maximum = max(score for _, score in candidate_scores)
    tied = [cut for cut, score in candidate_scores if maximum - score <= 1e-12]
    expected = min(tied, key=lambda cut: (abs(cut - 0.5), cut))
    assert result["threshold"] == expected
    assert result["selected_macro_f1"] == pytest.approx(maximum, abs=1e-15)


def test_missing_branch_is_fixed_at_half_independent_of_selected_threshold():
    p, y, missing = _supported_case()
    chosen = evaluate(y, p, missing, np.repeat(68, len(y)), 0.9)
    fixed = evaluate(y, p, missing, np.repeat(68, len(y)), 0.5)
    assert chosen["overall"]["rows"] == len(y)
    # Missing branch confusion is invariant to the learned present threshold.
    assert chosen["availability"]["missing"]["confusion_matrix"] == fixed["availability"]["missing"]["confusion_matrix"]
    # Present branch can change when its threshold changes.
    assert chosen["availability"]["present"]["confusion_matrix"] != fixed["availability"]["present"]["confusion_matrix"]


def test_insufficient_support_falls_back_without_selection():
    result = select_threshold(np.asarray([0.1, 0.9]), np.asarray([False, False]), np.asarray([0, 1]))
    assert result["status"] == "FIXED_0_50_INSUFFICIENT_SUPPORT"
    assert result["threshold"] == 0.5
    assert result["selection_fit_count"] == 0


def test_oof_origin_boundary_rejects_wrong_or_duplicate_w6_ids():
    e67_cycles = np.repeat(np.arange(68, 80, dtype=np.int16), 2)
    e67_ids = np.asarray([f"id-{cycle}-{i}" for cycle in range(68, 80) for i in range(2)], dtype="U16")
    e75_cycles = np.repeat(np.arange(76, 80, dtype=np.int16), 2)
    e75_ids = np.asarray([f"id-{cycle}-{i}" for cycle in range(76, 80) for i in range(2)], dtype="U16")
    def origin(ids, cycles):
        n = len(ids)
        return {"ids": ids, "cycles": cycles, "rows": np.arange(n), "pP": np.zeros(n),
                "pA": np.zeros(n), "MIX50": np.zeros(n), "missing": np.zeros(n, dtype=bool),
                "group_availability": np.full(n, "present", dtype="U7"),
                "group_text_load": np.full(n, "Q1", dtype="U2"),
                "group_entry_friction": np.full(n, "0", dtype="U1")}
    oof = {"67": origin(e67_ids, e67_cycles), "75": origin(e75_ids, e75_cycles)}
    _validate_oof_boundaries(oof)
    bad = {"67": oof["67"], "75": dict(oof["75"])}
    bad["75"]["ids"] = bad["75"]["ids"].copy()
    bad["75"]["ids"][0] = "duplicate-id"
    with pytest.raises(RuntimeError, match="exact same unique Train IDs"):
        _validate_oof_boundaries(bad)


def test_synthetic_end_to_end_checks_temporal_persistence_and_resume(tmp_path):
    result = _synthetic_end_to_end(tmp_path)
    assert result["status"] == "PASS"
    assert result["selection_stages"] == 3
    assert result["reserved_slots"] == 3
    assert result["estimator_fits"] == 0
    assert result["evaluation_targets_requested_after_receipt"] is True
    assert result["e79_probability_loader_guard"] == "PASS"
    assert result["persistence_resume"].startswith("PASS")
    assert result["budget_cap_and_no_retry"] == "PASS"
    assert result["all_cutoff_metrics_match_bruteforce"] is True


def test_resource_block_happens_before_reservation_and_consumes_no_slot(tmp_path):
    run_dir = tmp_path / "run"
    ledger = run_dir / "selection_ledger.jsonl"
    receipt, _ = _stage_paths("LOW_RAM_STAGE", run_dir)
    with pytest.raises(RuntimeError, match="blocked before reservation"):
        _reserve_selection(
            "LOW_RAM_STAGE", ledger, receipt,
            resource_gate_fn=lambda: _resource_preflight(available_bytes=4 * GIB - 1),
        )
    assert not ledger.exists()
    assert not receipt.exists()
    assert _reserve_selection(
        "LOW_RAM_STAGE", ledger, receipt,
        resource_gate_fn=lambda: _resource_preflight(available_bytes=8 * GIB),
    ) is True
    events = decision._read_jsonl(ledger)
    assert len(events) == 1
    assert events[0]["event"] == "RESERVE"
    assert events[0]["resource_gate"]["status"] == "PASS"


def test_complete_event_must_bind_exact_receipt_and_reservation_id(tmp_path):
    run_dir = tmp_path / "run"
    ledger = run_dir / "selection_ledger.jsonl"
    receipt_path, _ = _stage_paths("S1", run_dir)
    assert _reserve_selection(
        "S1", ledger, receipt_path,
        resource_gate_fn=lambda: _resource_preflight(available_bytes=8 * GIB),
    )
    reservation = decision._reservation_for_stage("S1", ledger)
    digest = _write_exclusive_json(receipt_path, {
        "selection_id": "S1",
        "reservation_id": reservation["reservation_id"],
        "threshold": 0.5,
    })
    decision._append_jsonl_fsync(ledger, {
        "event": "COMPLETE", "selection_id": "S1",
        "reservation_id": reservation["reservation_id"], "receipt_sha256": digest,
    })
    events = decision._read_jsonl(ledger)
    events[-1]["receipt_sha256"] = "0" * 64
    ledger.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Completed ledger receipt digest mismatch"):
        _reserve_selection("S1", ledger, receipt_path)


def test_selection_receipt_tampering_is_rejected_on_resume(tmp_path):
    evaluator, _, threshold, _, train, train_sha, run_dir = _evaluation_fixture(tmp_path)
    receipt_path, _ = _stage_paths("D1_E67_W4_TO_W5", run_dir)
    _reseal_receipt(receipt_path, lambda body: body.__setitem__("threshold", 0.99))
    with pytest.raises(RuntimeError, match="Completed ledger receipt digest mismatch"):
        _select_stage(
            stage_id="D1_E67_W4_TO_W5",
            selector=evaluator,
            selection_mask=(evaluator["cycles"] >= 68) & (evaluator["cycles"] <= 71),
            selection_cycles=(68, 71),
            train_path=train,
            train_sha=train_sha,
            run_dir=run_dir,
            source_hashes=threshold["source_hashes"],
            resource_gate_fn=lambda: pytest.fail("resume must not consume another resource reservation"),
        )


def test_reviewed_prefit_commit_allows_report_only_descendant_but_rejects_source_change(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    expected = {}
    for relative_path in decision.PREFIT_COMMITTED_FILES:
        path = repo / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"frozen source {relative_path}\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-m", "prefit source commit")
    prefit_commit = git("rev-parse", "HEAD")
    for relative_path in decision.PREFIT_COMMITTED_FILES:
        committed_bytes = subprocess.run(
            ["git", "-C", str(repo), "show", f"{prefit_commit}:{relative_path}"],
            check=True, capture_output=True,
        ).stdout
        expected[relative_path] = {
            "file_sha256": file_sha256(repo / relative_path),
            "git_blob_oid": git("rev-parse", f"{prefit_commit}:{relative_path}"),
            "git_blob_sha256": decision.hashlib.sha256(committed_bytes).hexdigest(),
        }
    _validate_prefit_commit(prefit_commit, expected, repo_root=repo)

    report = repo / "reports" / "review.md"
    report.parent.mkdir(parents=True)
    report.write_text("report-only descendant\n", encoding="utf-8")
    git("add", "reports/review.md")
    git("commit", "-m", "report only")
    report_head = git("rev-parse", "HEAD")
    _validate_prefit_commit(prefit_commit, expected, repo_root=repo, current_head=report_head)

    changed = repo / "scripts/optimization_v3_cls_mix50_temporal_decision_v1.py"
    changed.write_text("changed source\n", encoding="utf-8")
    git("add", "scripts/optimization_v3_cls_mix50_temporal_decision_v1.py")
    git("commit", "-m", "source changed")
    changed_head = git("rev-parse", "HEAD")
    with pytest.raises(RuntimeError, match="source blob hash mismatch"):
        _validate_prefit_commit(prefit_commit, expected, repo_root=repo, current_head=changed_head)


def test_root_approval_binds_freeze_file_and_internal_receipt_digests_separately(tmp_path, monkeypatch):
    freeze_path = tmp_path / "PREFIT_FREEZE_R5.json"
    freeze = {
        "prefit_commit": "a" * 40,
        "reviewed_source_files": {},
        "source_hashes": {},
        "input_hashes": {},
    }
    _write_exclusive_json(freeze_path, freeze)
    stored = _read_receipt(freeze_path)
    monkeypatch.setattr(decision, "FREEZE", freeze_path)
    monkeypatch.setattr(decision, "_validate_prefit_commit", lambda *args, **kwargs: None)
    monkeypatch.setattr(decision, "_verify_frozen_source_hashes", lambda _: None)
    monkeypatch.setattr(decision, "_verify_frozen_input_hashes", lambda _: None)
    expected = {
        "status": "PASS",
        "prefit_commit": stored["prefit_commit"],
        "prefit_freeze_file_sha256": file_sha256(freeze_path),
        "prefit_freeze_receipt_sha256": stored["receipt_sha256"],
        "protocol_sha256": file_sha256(decision.PROTOCOL),
        "script_sha256": file_sha256(decision.Path(decision.__file__)),
        "tests_sha256": file_sha256(decision.TESTS),
        "research_next_contract_sha256": file_sha256(decision.ROOT / "src/models/research_next_contract.py"),
        "scope": "CLS_MIX50_TEMPORAL_DECISION_V1",
    }
    approval = tmp_path / "approval.json"
    approval.write_text(json.dumps(expected), encoding="utf-8")
    assert file_sha256(freeze_path) != stored["receipt_sha256"]
    assert decision._validate_root_approval(approval, stored) == expected
    expected["prefit_freeze_file_sha256"] = stored["receipt_sha256"]
    approval.write_text(json.dumps(expected), encoding="utf-8")
    with pytest.raises(RuntimeError, match="prefit_freeze_file_sha256"):
        decision._validate_root_approval(approval, stored)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("evaluation_id", "WRONG_STAGE", "evaluation_id"),
        ("source_hashes", {"stale": "hash"}, "source_hashes"),
        ("evaluation_train_labels_sha256", "0" * 64, "target-label digest"),
    ],
)
def test_self_consistent_but_stale_evaluation_receipt_is_rejected(tmp_path, field, value, message):
    evaluator, mask, threshold, _, train, train_sha, run_dir = _evaluation_fixture(tmp_path)
    _, eval_path = _stage_paths("D1_E67_W4_TO_W5", run_dir)
    _reseal_receipt(eval_path, lambda body: body.__setitem__(field, value))
    with pytest.raises(RuntimeError, match=message):
        _evaluate_stage(
            stage_id="D1_E67_W4_TO_W5", evaluator=evaluator, evaluation_mask=mask,
            threshold_receipt=threshold, train_path=train, train_sha=train_sha, run_dir=run_dir,
        )


def test_evaluation_resume_rejects_reordered_ids_changed_probabilities_and_source(tmp_path):
    evaluator, mask, threshold, _, train, train_sha, run_dir = _evaluation_fixture(tmp_path)
    reordered = {key: value[::-1].copy() for key, value in evaluator.items()}
    with pytest.raises(RuntimeError, match="evaluation_record_ids_sha256"):
        _evaluate_stage(
            stage_id="D1_E67_W4_TO_W5", evaluator=reordered,
            evaluation_mask=mask[::-1], threshold_receipt=threshold,
            train_path=train, train_sha=train_sha, run_dir=run_dir,
        )
    changed = {key: value.copy() for key, value in evaluator.items()}
    changed_index = int(np.flatnonzero(mask)[0])
    changed["MIX50"][changed_index] = 0.1
    with pytest.raises(RuntimeError, match="evaluation_input_hashes"):
        _evaluate_stage(
            stage_id="D1_E67_W4_TO_W5", evaluator=changed, evaluation_mask=mask,
            threshold_receipt=threshold, train_path=train, train_sha=train_sha, run_dir=run_dir,
        )
    altered_threshold = dict(threshold)
    altered_threshold["source_hashes"] = {"stale": "source"}
    with pytest.raises(RuntimeError, match="not the persisted selection receipt"):
        _evaluate_stage(
            stage_id="D1_E67_W4_TO_W5", evaluator=evaluator, evaluation_mask=mask,
            threshold_receipt=altered_threshold, train_path=train, train_sha=train_sha, run_dir=run_dir,
        )


def test_official_entrypoint_mock_persists_threshold_before_evaluation_label_reads(tmp_path, monkeypatch):
    counts = {
        68: 1238, 69: 1237, 70: 1237, 71: 1238,
        72: 1865, 73: 1866, 74: 1865, 75: 1866,
        76: 2123, 77: 2123, 78: 2123, 79: 2122,
    }
    cycles = np.concatenate([np.full(counts[cycle], cycle, dtype=np.int16) for cycle in counts])
    ids = np.asarray([f"mock-{cycle}-{i}" for cycle in counts for i in range(counts[cycle])], dtype="U24")
    labels = np.concatenate([np.arange(counts[cycle], dtype=np.int8) % 2 for cycle in counts])
    local_index = np.concatenate([np.arange(counts[cycle]) for cycle in counts])
    missing = local_index % 5 == 0
    p_pos = np.where(labels == 1, 0.76, 0.24).astype(np.float64)
    p_a = np.where(labels == 1, 0.68, 0.32).astype(np.float64)
    mix = np.where(missing, p_pos, 0.5 * p_pos + 0.5 * p_a)
    groups = {
        "group_availability": np.where(missing, "missing", "present").astype("U7"),
        "group_text_load": np.where(local_index % 2 == 0, "Q1", "Q2").astype("U2"),
        "group_entry_friction": np.where(local_index % 3 == 0, "1", "0").astype("U1"),
    }
    train = tmp_path / "Train.csv"
    with train.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["record_id", "cycle_code", "artifact_signal"])
        writer.writerows(zip(ids.tolist(), cycles.tolist(), labels.tolist()))
    train_sha = file_sha256(train)
    oof_path = tmp_path / "policy_predictions.npz"
    is_w6 = cycles >= 76
    oof_fields = {
        "ids": ids, "rows": np.arange(len(ids), dtype=np.int64), "cycles": cycles,
        "pP": p_pos, "pA": p_a, "missing": missing, "MIX50": mix,
        **groups,
    }
    e75_fields = {key: value[is_w6] for key, value in oof_fields.items()}
    np.savez_compressed(
        oof_path,
        **{f"E67_{key}": value for key, value in oof_fields.items()},
        **{f"E75_{key}": value for key, value in e75_fields.items()},
    )
    freeze_path = tmp_path / "PREFIT_FREEZE_R5.json"
    _write_exclusive_json(freeze_path, {
        "prefit_commit": "a" * 40,
        "input_hashes": {
            "official_train_csv": train_sha,
            "exact_mix50_train_oof_bundle": file_sha256(oof_path),
            "expected_e79_probability_bundle": "0" * 64,
        },
    })
    report_dir = tmp_path / "reports"
    run_dir = tmp_path / "run"
    monkeypatch.setattr(decision, "FREEZE", freeze_path)
    monkeypatch.setattr(decision, "TRAIN_CSV", train)
    monkeypatch.setattr(decision, "OOF_NPZ", oof_path)
    monkeypatch.setattr(decision, "RUN_DIR", run_dir)
    monkeypatch.setattr(decision, "REPORT_DIR", report_dir)
    monkeypatch.setattr(decision, "_validate_root_approval", lambda *args: {})
    monkeypatch.setattr(decision, "_verify_frozen_source_hashes", lambda _: None)
    monkeypatch.setattr(decision, "_resource_preflight", lambda available_bytes=None: _resource_preflight(8 * GIB))

    original_loader = decision.load_train_labels_by_id
    evaluation_ids = {
        "D1_E67_W4_TO_W5": set(ids[(cycles >= 72) & (cycles <= 75)].tolist()),
        "D2_E67_TO_E75_W6": set(ids[is_w6].tolist()),
    }
    observed_evaluation_reads = []

    def tracked_loader(train_path, requested_ids, expected_cycle_by_id, *, expected_train_sha256):
        requested = set(np.asarray(requested_ids, dtype=str).tolist())
        for stage_id, stage_eval_ids in evaluation_ids.items():
            if requested == stage_eval_ids:
                selection_path, _ = _stage_paths(stage_id, run_dir)
                assert selection_path.exists(), f"{stage_id} evaluation labels accessed before durable threshold"
                _read_receipt(selection_path)
                observed_evaluation_reads.append(stage_id)
        return original_loader(
            train_path, requested_ids, expected_cycle_by_id,
            expected_train_sha256=expected_train_sha256,
        )

    monkeypatch.setattr(decision, "load_train_labels_by_id", tracked_loader)
    result = decision.run_official_selectors(tmp_path / "approval.json")
    assert result["status"] == "OFFICIAL_SELECTORS_COMPLETE"
    assert result["estimator_fits"] == 0
    assert result["decision_selection_budget_max"] == 3
    assert sorted(observed_evaluation_reads) == ["D1_E67_W4_TO_W5", "D2_E67_TO_E75_W6"]
