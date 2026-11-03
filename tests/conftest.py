from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_pre_fit_budget_fixture(path: Path) -> Path:
    caps = {"pilot": 42, "continuation": 38, "control_replay": 2, "technical_retry": 1}
    events: list[dict] = [{"event": "BATCH_ALLOWANCE_AUTHORIZED", "cap": 83, "category_caps": caps}]
    reservation_keys = {
        category: f"SYNTHETIC:{category.upper()}" for category in caps
    }
    for category, fits in caps.items():
        fit_ids = [f"SYNTH_BASE_{category}_{i:02d}" for i in range(fits)]
        events.append({"event": "CARD_FITS_RESERVED", "reservation_key": reservation_keys[category],
                       "category": category, "fits": fits, "fit_ids": fit_ids})
    all_fit_ids: list[tuple[str, str]] = []
    for category, fits in caps.items():
        all_fit_ids.extend((f"SYNTH_BASE_{category}_{i:02d}", reservation_keys[category])
                           for i in range(fits))
    for index, (fit_id, reservation_key) in enumerate(all_fit_ids):
        events.append({"event": "FIT_STARTED", "fit_id": fit_id, "reservation_key": reservation_key,
                       "component": "synthetic fixture", "fit_consumed_at_start": True})
    for index, (fit_id, reservation_key) in enumerate(all_fit_ids):
        if index < 80:
            events.append({"event": "FIT_SAVED", "fit_id": fit_id, "reservation_key": reservation_key,
                           "receipt_sha256": f"{index:064x}"})
        else:
            events.append({"event": "FIT_FAILED", "fit_id": fit_id, "reservation_key": reservation_key,
                           "error": "historical synthetic fixture failure"})
    while len(events) < 216:
        events.append({"event": "SYNTHETIC_LEDGER_NOTE", "sequence": len(events)})
    if len(events) != 216:
        raise AssertionError("Synthetic pre-fit ledger fixture must contain 216 events")
    payload = "".join(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n" for event in events)
    path.write_text(payload, encoding="utf-8", newline="\n")
    return path


@pytest.fixture
def synthetic_pre_fit_budget_ledger(tmp_path: Path) -> Path:
    return _write_pre_fit_budget_fixture(tmp_path / "pre_fit_budget.jsonl")


@pytest.fixture(autouse=True)
def isolate_frozen_target_median_lifecycle_test(request, monkeypatch):
    """Keep the frozen prefit lifecycle check independent of live ledger state."""
    if request.node.name != "test_synthetic_persistence_budget_and_resume_lifecycle":
        yield
        return
    from scripts import optimization_v3_rnbr_target_median_v1 as runner

    synthetic_pre_fit_budget_ledger = request.getfixturevalue("synthetic_pre_fit_budget_ledger")
    live_path = runner.LEDGER_PATH
    live_sha = runner._sha_file(live_path)
    snapshot = runner.ledger_snapshot(runner._ledger_events(synthetic_pre_fit_budget_ledger))
    if any(snapshot.get(key) != value for key, value in runner.PREVIOUS_LEDGER_SNAPSHOT.items()):
        raise AssertionError(f"Temporary pre-fit budget fixture differs: {snapshot}")
    monkeypatch.setattr(runner, "LEDGER_PATH", synthetic_pre_fit_budget_ledger)
    yield
    if runner._sha_file(live_path) != live_sha:
        raise AssertionError("Target-median synthetic QA changed the live shared ledger")
