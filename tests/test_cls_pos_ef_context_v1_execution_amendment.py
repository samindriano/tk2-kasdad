from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pytest
from scipy import sparse

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


def test_importlib_driver_loads_frozen_core_with_parent_src_preloaded_and_runs_two_toy_fits(tmp_path, monkeypatch):
    adapter = load_adapter()
    historical_src = types.ModuleType("src")
    historical_src.__path__ = [str(tmp_path / "historical_src")]
    historical_models = types.ModuleType("src.models")
    historical_models.__path__ = [str(tmp_path / "historical_src" / "models")]
    historical_src.models = historical_models
    monkeypatch.setitem(sys.modules, "src", historical_src)
    monkeypatch.setitem(sys.modules, "src.models", historical_models)
    monkeypatch.setattr(historical_models, "cls_pos_ef_context_v1", None, raising=False)
    monkeypatch.setitem(sys.modules, "src.models.cls_pos_ef_context_v1", None)

    pilot = adapter.load_frozen_driver()
    monkeypatch.setattr(pilot, "RUN_ROOT", tmp_path)
    freeze = pilot.read_json(pilot.FREEZE_PATH)
    core = adapter.load_audit_core(pilot, freeze)
    assert Path(core.__file__).resolve() == ROOT / "src/models/cls_pos_ef_context_v1.py"
    assert sys.modules["src.models.cls_pos_ef_context_v1"] is core
    assert historical_models.cls_pos_ef_context_v1 is core

    ledger = tmp_path / "fit_ledger.jsonl"
    ledger.write_text("", encoding="utf-8")
    pilot.append_event(ledger, {"event": "CARD_FITS_RESERVED", "candidate_id": freeze["candidate_id"],
        "freeze_signature": freeze["freeze_signature"], "fit_ids": freeze["fit_ids"], "fit_cap": 4})
    calls = []
    x = sparse.csr_matrix(np.linspace(-1.0, 1.0, 40, dtype=np.float64).reshape(-1, 1))
    offset = np.zeros(40, dtype=np.float64)
    target = (np.arange(40) >= 20).astype(np.int8)

    def synthetic_fit(repository, fit_id):
        events = pilot.check_live_budget(freeze, allow_current=fit_id)
        started = {e["fit_id"] for e in events if e.get("event") == "FIT_STARTED"}
        next_fit = next((candidate for candidate in freeze["fit_ids"] if candidate not in started), None)
        pilot.require(fit_id == next_fit, f"wrong synthetic next fit: {next_fit}")
        pilot.require(not any(e.get("event") == "FIT_STARTED" and e.get("fit_id") != fit_id
                              for e in events if e.get("event") == "FIT_STARTED"),
                      adapter.PENDING_GUARD_MESSAGE)
        pilot.append_event(ledger, {"event": "FIT_STARTED", "fit_id": fit_id})
        # This is the same canonical import statement used by the frozen fit entrypoint.
        from src.models import cls_pos_ef_context_v1 as imported_core
        assert imported_core is core
        gamma, metadata = imported_core.fit_context_gamma(x, offset, target)
        assert metadata["success"] is True and np.isfinite(gamma).all()
        calls.append(fit_id)
        pilot.append_event(ledger, {"event": "FIT_SAVED", "fit_id": fit_id})
        return {"fit_id": fit_id, "status": "SAVED_SYNTHETIC", "gamma": gamma.tolist()}

    monkeypatch.setattr(pilot, "fit_one", synthetic_fit)
    for fit_id in freeze["fit_ids"][:2]:
        original_require = adapter.completed_fit_guard(pilot, freeze)
        try:
            pilot.fit_one(None, fit_id)
        finally:
            pilot.require = original_require
    assert calls == freeze["fit_ids"][:2]
    starts = [e["fit_id"] for e in pilot.events(ledger) if e.get("event") == "FIT_STARTED"]
    assert starts == freeze["fit_ids"][:2]


def test_prefit_abort_reconciliation_proves_no_fit_start_or_optimizer_retry(tmp_path, monkeypatch):
    adapter, pilot, freeze, fit_ids = setup_synthetic(tmp_path, monkeypatch)
    monkeypatch.setattr(adapter, "RUN_ROOT", tmp_path)
    monkeypatch.setattr(adapter, "LINEAGE_PATH", tmp_path / "execution_lineage.jsonl")
    adapter.append_lineage({"event": "AMENDMENT_EXECUTION_STARTED", "fit_id": fit_ids[1],
        "amendment_id": "synthetic-amendment", "freeze_signature": freeze["freeze_signature"],
        "execution_code_sha256": "old-adapter-hash", "execution_commit": "old-adapter-commit"})
    adapter.append_lineage({"event": "AMENDMENT_EXECUTION_RAISED", "fit_id": fit_ids[1],
        "amendment_id": "synthetic-amendment", "freeze_signature": freeze["freeze_signature"],
        "fit_started": False, "fit_terminal_events": [], "error_type": "ImportError"})
    rows = adapter.reconcile_prefit_aborts(pilot, freeze, fit_ids[1], adapter.read_lineage())
    assert len(rows) == 1
    assert rows[0]["fit_slot_consumed"] is False
    assert rows[0]["optimizer_invocations"] == 0
    assert rows[0]["optimizer_retry"] is False
    assert rows[0]["adapter_plumbing_resume"] is True
    assert rows[0]["official_fit_ledger_event_count"] == 1
    pilot.append_event(tmp_path / "fit_ledger.jsonl", {"event": "FIT_STARTED", "fit_id": fit_ids[1]})
    with pytest.raises(pilot.PilotError, match="already contains this ID"):
        adapter.reconcile_prefit_aborts(pilot, freeze, fit_ids[1], adapter.read_lineage())
