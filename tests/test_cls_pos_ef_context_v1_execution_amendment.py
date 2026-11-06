from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = ROOT / "scripts/run_cls_pos_ef_context_v1_amended.py"


def load_adapter():
    spec = importlib.util.spec_from_file_location("cls_pos_ef_context_v1_amended_test", ADAPTER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def setup_synthetic(tmp_path, monkeypatch):
    adapter = load_adapter()
    pilot = adapter.load_frozen_driver()
    monkeypatch.setattr(pilot, "RUN_ROOT", tmp_path)
    (tmp_path / "fit_ledger.jsonl").write_text("", encoding="utf-8")
    fit_ids = ["CLS_POS_EF_CONTEXT_V1_E67_REAL", "CLS_POS_EF_CONTEXT_V1_E67_SHUFFLED",
               "CLS_POS_EF_CONTEXT_V1_E75_REAL", "CLS_POS_EF_CONTEXT_V1_E75_SHUFFLED"]
    freeze = {"candidate_id": "CLS_POS_EF_CONTEXT_V1", "fit_cap": 4,
              "fit_ids": fit_ids, "freeze_signature": "synthetic-freeze"}
    pilot.append_event(tmp_path / "fit_ledger.jsonl", {"event": "CARD_FITS_RESERVED",
        "candidate_id": freeze["candidate_id"], "freeze_signature": freeze["freeze_signature"],
        "fit_ids": fit_ids, "fit_cap": 4})

    def synthetic_fit(repository, fit_id):
        ev = pilot.check_live_budget(freeze, allow_current=fit_id)
        started = {event["fit_id"] for event in ev if event.get("event") == "FIT_STARTED"}
        next_fit = next((candidate for candidate in fit_ids if candidate not in started), None)
        pilot.require(fit_id == next_fit, f"wrong synthetic next fit: {next_fit}")
        # Exercise the exact defective predicate from the frozen driver.
        pilot.require(not any(event.get("event") == "FIT_STARTED" and event.get("fit_id") != fit_id
                              for event in ev if event.get("event") == "FIT_STARTED"),
                      adapter.PENDING_GUARD_MESSAGE)
        pilot.append_event(tmp_path / "fit_ledger.jsonl", {"event": "FIT_STARTED", "fit_id": fit_id})
        pilot.append_event(tmp_path / "fit_ledger.jsonl", {"event": "FIT_SAVED", "fit_id": fit_id})
        return {"fit_id": fit_id, "status": "SAVED_SYNTHETIC"}

    monkeypatch.setattr(pilot, "fit_one", synthetic_fit)
    return adapter, pilot, freeze, fit_ids


def invoke_amended(pilot, freeze, fit_id):
    original_require = load_adapter().completed_fit_guard(pilot, freeze)
    try:
        return pilot.fit_one(None, fit_id)
    finally:
        pilot.require = original_require


def test_two_sequential_fit_entries_allow_prior_saved_fit(tmp_path, monkeypatch):
    _, pilot, freeze, fit_ids = setup_synthetic(tmp_path, monkeypatch)
    first = invoke_amended(pilot, freeze, fit_ids[0])
    second = invoke_amended(pilot, freeze, fit_ids[1])
    assert first["status"] == second["status"] == "SAVED_SYNTHETIC"
    events = pilot.events(tmp_path / "fit_ledger.jsonl")
    assert [event["fit_id"] for event in events if event.get("event") == "FIT_STARTED"] == fit_ids[:2]
    assert [event["fit_id"] for event in events if event.get("event") == "FIT_SAVED"] == fit_ids[:2]


def test_unfinished_fit_still_blocks_next_fit(tmp_path, monkeypatch):
    adapter, pilot, freeze, fit_ids = setup_synthetic(tmp_path, monkeypatch)
    pilot.append_event(tmp_path / "fit_ledger.jsonl", {"event": "FIT_STARTED", "fit_id": fit_ids[0]})
    with pytest.raises(pilot.PilotError, match="pending fit differs"):
        invoke_amended(pilot, freeze, fit_ids[1])
    original_require = adapter.completed_fit_guard(pilot, freeze)
    try:
        with pytest.raises(pilot.PilotError):
            pilot.require(False, adapter.PENDING_GUARD_MESSAGE)
    finally:
        pilot.require = original_require
    events = pilot.events(tmp_path / "fit_ledger.jsonl")
    assert [event["fit_id"] for event in events if event.get("event") == "FIT_STARTED"] == [fit_ids[0]]


def test_failed_fit_is_consumed_but_not_active(tmp_path, monkeypatch):
    adapter, pilot, freeze, fit_ids = setup_synthetic(tmp_path, monkeypatch)
    pilot.append_event(tmp_path / "fit_ledger.jsonl", {"event": "FIT_STARTED", "fit_id": fit_ids[0]})
    pilot.append_event(tmp_path / "fit_ledger.jsonl", {"event": "FIT_FAILED", "fit_id": fit_ids[0]})
    result = invoke_amended(pilot, freeze, fit_ids[1])
    assert result["status"] == "SAVED_SYNTHETIC"


def test_duplicate_start_events_are_rejected(tmp_path, monkeypatch):
    adapter, pilot, freeze, fit_ids = setup_synthetic(tmp_path, monkeypatch)
    for _ in range(2):
        pilot.append_event(tmp_path / "fit_ledger.jsonl", {"event": "FIT_STARTED", "fit_id": fit_ids[0]})
    with pytest.raises(pilot.PilotError, match="duplicate a frozen ID"):
        pilot.check_live_budget(freeze)
