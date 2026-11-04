from __future__ import annotations

import numpy as np
import pytest
import subprocess
import sys
import json
from contextlib import contextmanager

import threadpoolctl

from scripts import optimization_v3_cls_a02_missing_temporal_decision_v2 as decision
from scripts import optimization_v3_cls_mix50_temporal_decision_v1 as mix

_PARENT_RESOURCE_PREFLIGHT = mix._resource_preflight


def _make_view(prefix: str, cycles: list[int]) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    ids: list[str] = []
    cycle_values: list[int] = []
    labels: dict[str, int] = {}
    missing: list[bool] = []
    p_p: list[float] = []
    p_a: list[float] = []
    p_p_old: list[float] = []
    entry: list[str] = []
    text: list[str] = []
    for cycle in cycles:
        for i, y in enumerate((0, 1, 0, 1)):
            rid = f"{prefix}-{cycle}-{i}"
            ids.append(rid)
            cycle_values.append(cycle)
            labels[rid] = y
            missing.append(i < 2)
            p_p.append((0.18, 0.82, 0.32, 0.68)[i])
            p_a.append((0.25, 0.74, 0.42, 0.58)[i])
            p_p_old.append((0.28, 0.72, 0.38, 0.62)[i])
            entry.append("low" if i % 2 == 0 else "high")
            text.append("Q1" if i < 2 else "Q2")
    ids_array = np.asarray(ids, dtype="U")
    pa = np.asarray(p_a, dtype=np.float64)
    p_p_array = np.asarray(p_p_old, dtype=np.float64)
    return ({
        "ids": ids_array,
        "rows": np.arange(len(ids), dtype=np.int64),
        "cycles": np.asarray(cycle_values, dtype=np.int16),
        "pP": np.asarray(p_p_old, dtype=np.float64),
        "pA": pa,
        "missing": np.asarray(missing, dtype=bool),
        "MIX50": 0.5 * (p_p_array + pa),
        "group_availability": np.where(missing, "missing", "present"),
        "group_text_load": np.asarray(text, dtype="U"),
        "group_entry_friction": np.asarray(entry, dtype="U"),
    }, labels)


def _redirect_artifacts(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(decision, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(decision, "RECEIPT_DIR", tmp_path / "receipts")
    monkeypatch.setattr(decision, "EVAL_DIR", tmp_path / "evaluations")
    monkeypatch.setattr(decision, "LEDGER_PATH", tmp_path / "SELECTOR_LEDGER.jsonl")
    monkeypatch.setattr(decision, "RESULTS_PATH", tmp_path / "RESULTS.json")
    monkeypatch.setattr(decision, "RUN_MANIFEST_PATH", tmp_path / "RUN_MANIFEST.json")
    monkeypatch.setattr(decision, "MIN_SELECTION_ROWS", 4)
    monkeypatch.setattr(decision, "MIN_SELECTION_PER_CLASS", 1)
    monkeypatch.setattr(decision, "MIN_VARYING_ROWS", 2)
    monkeypatch.setattr(decision, "MIN_VARYING_PER_CLASS", 1)
    monkeypatch.setattr(decision, "MIN_GROUP_ROWS", 2)
    monkeypatch.setattr(decision, "MIN_GROUP_PER_CLASS", 1)
    synthetic_protocol = decision._read_json(decision.PROTOCOL_PATH)
    synthetic_protocol["support_gate"].update({
        "full_selection_rows_min": decision.MIN_SELECTION_ROWS,
        "full_selection_each_class_min": decision.MIN_SELECTION_PER_CLASS,
        "varying_missing_rows_min": decision.MIN_VARYING_ROWS,
        "varying_missing_each_class_min": decision.MIN_VARYING_PER_CLASS,
    })
    monkeypatch.setattr(decision, "_load_protocol", lambda: synthetic_protocol)


def _synthetic_resource_gate() -> dict[str, object]:
    return _PARENT_RESOURCE_PREFLIGHT(available_bytes=mix.REQUIRED_AVAILABLE_RAM_BYTES)


def test_threshold_solver_matches_brute_force_and_ties(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(decision, "MIN_SELECTION_ROWS", 4)
    monkeypatch.setattr(decision, "MIN_SELECTION_PER_CLASS", 1)
    monkeypatch.setattr(decision, "MIN_VARYING_ROWS", 2)
    monkeypatch.setattr(decision, "MIN_VARYING_PER_CLASS", 1)
    y = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int8)
    present = np.asarray([0.8, 0.2, 0.6, 0.4, 0.9, 0.1, 0.7, 0.3])
    p_a = np.asarray([0.10, 0.90, 0.40, 0.60, 0.50, 0.50, 0.25, 0.75])
    missing = np.asarray([1, 1, 1, 1, 1, 1, 1, 1], dtype=bool)
    result = decision.choose_missing_threshold(y, present, p_a, missing)
    candidates = mix.threshold_candidates(p_a)
    scored = []
    for threshold in candidates:
        pred = p_a >= threshold
        scored.append((float(threshold), decision.macro_f1_from_confusion(decision.confusion(y, pred))))
    best = max(score for _, score in scored)
    tied = [item for item in scored if best - item[1] <= decision.TIE_TOLERANCE]
    expected = min(tied, key=lambda item: (abs(item[0] - 0.5), item[0]))
    assert result["threshold"] == expected[0]
    assert result["selection_macro_f1_at_threshold"] == pytest.approx(expected[1])
    assert result["threshold_operator"] == ">="
    assert np.array_equal(np.asarray([0.49, 0.50, 0.51]) >= 0.50, [False, True, True])


def test_support_gate_and_fixed_present_decisions() -> None:
    y = np.asarray([0, 1] * 600, dtype=np.int8)
    missing = np.zeros(len(y), dtype=bool)
    missing[:200] = True
    support = decision.selection_support(y, missing)
    assert support["status"] == "SUPPORTED"
    assert support["selector_consumed"] is True
    too_small = decision.selection_support(y[:500], missing[:500])
    assert too_small["status"] == "FALLBACK_FIXED_0_50_INSUFFICIENT_SUPPORT"
    p_present = np.asarray([0.1, 0.9, 0.1, 0.9])
    p_a = np.asarray([0.2, 0.8, 0.3, 0.7])
    miss = np.asarray([False, False, True, True])
    pred_035 = decision._fixed_predictions(p_present, p_a, miss, 0.35)
    pred_080 = decision._fixed_predictions(p_present, p_a, miss, 0.80)
    assert np.array_equal(pred_035[~miss], pred_080[~miss])
    assert np.array_equal(pred_035[~miss], p_present[~miss] >= 0.50)


def test_frozen_protocol_matches_code_constants() -> None:
    protocol = decision._load_protocol()
    assert protocol["policy_id"] == decision.POLICY_ID
    assert protocol["stage_order"] == list(decision.EXPECTED_STAGES)
    assert protocol["resource_and_source_boundaries"]["max_decision_selectors"] == 3
    assert protocol["resource_and_source_boundaries"]["project_job_limit"] == 1
    assert protocol["resource_and_source_boundaries"]["threadpool_thread_limit"] == 2
    assert protocol["resource_and_source_boundaries"]["new_estimator_fits"] == 0
    assert protocol["root_source_pass_contract"]["receipt_must_be_committed_before_official_selectors"] is True


def test_temporal_end_to_end_persistence_resume_and_three_selector_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _redirect_artifacts(monkeypatch, tmp_path)
    origin67, labels67 = _make_view("E67", [68, 69, 70, 71, 72, 73, 74, 75])
    origin75, labels75 = _make_view("E75", [76, 77, 78, 79])
    all_labels = {**labels67, **labels75}
    monkeypatch.setattr(decision, "_labels_by_id", lambda view: np.asarray([all_labels[x] for x in view["ids"]], dtype=np.int8))
    monkeypatch.setattr(decision, "_load_oof", lambda: {"67": origin67, "75": origin75})
    monkeypatch.setattr(decision, "verify_prefit_checkout", lambda: "frozen-source-commit")
    monkeypatch.setattr(decision, "EXPECTED_POOLED_W5_W6_ROWS", 32)
    monkeypatch.setattr(decision, "_validate_planned_support", lambda stage, support: None)
    monkeypatch.setattr(decision, "_validate_planned_evaluation", lambda stage, view: None)
    resource_calls: list[int] = []

    def resource_preflight():
        resource_calls.append(1)
        return _synthetic_resource_gate()

    monkeypatch.setattr(mix, "_resource_preflight", resource_preflight)

    first = decision.run_selectors()
    assert first["status"] == "THREE_SELECTOR_PILOT_COMPLETE"
    assert first["decision_selector_budget"]["reserved"] == 3
    assert first["decision_selector_budget"]["saved_selectors"] == 3
    assert first["decision_selector_budget"]["new_estimator_fits"] == 0
    assert first["test_accessed"] is False
    assert set(first["evaluations"]) == {"D1_E67_W4_TO_W5", "D2_E67_W4W5_TO_E75_W6"}
    assert first["pooled_unique_w5_w6"]["rows"] == 32
    assert len(resource_calls) == 3
    for stage in decision.EXPECTED_STAGES:
        receipt = decision._read_verified_receipt(decision._receipt_path(stage))
        mix._validate_resource_gate_record(receipt["resource_gate"])

    second = decision.run_selectors()
    assert second["selectors"] == first["selectors"]
    assert second["evaluations"] == first["evaluations"]
    assert len(resource_calls) == 3
    events = decision._read_ledger()
    assert sum(event["event"] == "SELECTOR_RESERVED" for event in events) == 3
    assert sum(event["event"] == "SELECTOR_SAVED" for event in events) == 3
    assert sum(event["event"] == "EVALUATION_SAVED" for event in events) == 2
    run_manifest = decision._read_verified_receipt(decision.RUN_MANIFEST_PATH)
    assert run_manifest["selector_count"] == 3
    assert run_manifest["new_model_inference"] == 0

    final_view = decision._view_for_spec({"67": origin67, "75": origin75}, decision._stage_spec()["FINAL_STITCHED_E79"]["selection"])
    context = decision._selector_context("FINAL_STITCHED_E79", final_view, "frozen-source-commit")
    context["resource_gate"] = _synthetic_resource_gate()
    with pytest.raises(RuntimeError, match="already reserved"):
        decision._reserve_selector("FINAL_STITCHED_E79", context, events)
    with pytest.raises(RuntimeError, match="context mismatch"):
        decision._run_selection("FINAL_STITCHED_E79", final_view, "unapproved-source-commit")

    changed = {key: np.array(value, copy=True) for key, value in final_view.items()}
    changed["pA"][0] = 0.99
    with pytest.raises(RuntimeError, match="context mismatch"):
        decision._run_selection("FINAL_STITCHED_E79", changed, "frozen-source-commit")

    ledger_events = decision._read_ledger()
    final_saved = next(event for event in ledger_events if event.get("selector_id") == "FINAL_STITCHED_E79" and event.get("event") == "SELECTOR_SAVED")
    final_saved["threshold_receipt_sha256"] = "0" * 64
    decision.LEDGER_PATH.write_text("".join(json.dumps(event, sort_keys=True) + "\n" for event in ledger_events), encoding="utf-8")
    with pytest.raises(RuntimeError, match="exact durable threshold receipt file hash"):
        decision._run_selection("FINAL_STITCHED_E79", final_view, "frozen-source-commit")


def test_low_ram_blocks_before_target_read_or_selector_reservation(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _redirect_artifacts(monkeypatch, tmp_path)
    view, _ = _make_view("E67", [68, 69, 70, 71])
    labels_read = False

    def forbidden_label_read(_view):
        nonlocal labels_read
        labels_read = True
        raise AssertionError("Train labels must not be opened below the RAM gate")

    monkeypatch.setattr(decision, "_labels_by_id", forbidden_label_read)
    monkeypatch.setattr(mix, "_resource_preflight", lambda: _PARENT_RESOURCE_PREFLIGHT(available_bytes=mix.REQUIRED_AVAILABLE_RAM_BYTES - 1))
    with pytest.raises(RuntimeError, match="Selector resource gate blocked"):
        decision._run_selection("D1_E67_W4_TO_W5", view, "synthetic-source-commit")
    assert labels_read is False
    assert not decision.LEDGER_PATH.exists()


def test_single_project_job_lock_blocks_a_concurrent_process(tmp_path) -> None:
    script = f"""
from pathlib import Path
from scripts import optimization_v3_cls_mix50_temporal_decision_v1 as m
with m._single_project_job(Path({str(tmp_path)!r})):
    pass
"""
    with mix._single_project_job(tmp_path):
        result = subprocess.run([sys.executable, "-c", script], cwd=decision.ROOT, capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "already active" in result.stderr


def test_selector_and_e79_entrypoints_apply_two_thread_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(decision, "RUN_MANIFEST_PATH", tmp_path / "run" / "RUN_MANIFEST.json")
    applied: list[int] = []

    @contextmanager
    def synthetic_thread_limit(limits: int):
        applied.append(limits)
        yield

    monkeypatch.setattr(threadpoolctl, "threadpool_limits", synthetic_thread_limit)
    monkeypatch.setattr(decision, "_run_selectors_locked", lambda: {"status": "synthetic selector body"})
    monkeypatch.setattr(decision, "_build_e79_candidate_locked", lambda: {"status": "synthetic E79 body"})
    assert decision.run_selectors()["status"] == "synthetic selector body"
    assert decision.build_e79_candidate()["status"] == "synthetic E79 body"
    assert applied == [decision.MAX_THREADPOOL_THREADS, decision.MAX_THREADPOOL_THREADS]


def test_evaluation_event_cannot_precede_a_saved_decision() -> None:
    with pytest.raises(RuntimeError, match="Invalid evaluation"):
        decision._ledger_state([{"event": "EVALUATION_SAVED", "selector_id": decision.EXPECTED_STAGES[0]}])
