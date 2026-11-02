import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import optimization_v3_a02_continuation_verify as verifier
from src.models.optimization_v3_a02_continuation_budget import (
    AMENDMENT_ID,
    EFFECTIVE_CAPS,
    apply_cap_amendment,
    reserve,
    start_fit,
    verify_accounting,
)


def _write_events(path, events):
    path.write_text(
        "".join(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n" for event in events),
        encoding="utf-8",
    )


def _historical_prefix():
    prior_caps = {"pilot": 34, "continuation": 24, "control_replay": 0, "technical_retry": 2}
    events = [
        {"event": "BATCH_ALLOWANCE_AUTHORIZED", "cap": 60},
        {"event": "BUDGET_CATEGORY_AMENDMENT", "amendment_id": "PREVIOUS_AMENDMENT"},
        {"event": "BUDGET_CATEGORY_AMENDMENT", "amendment_id": "C19_A02_PILOT_TRANSFER_2026_11_02",
         "effective_caps": prior_caps},
        {"event": "CARD_FITS_RESERVED", "reservation_key": "PILOT:OLD", "category": "pilot",
         "card_id": "PILOT", "stage": "OLD", "fits": 37},
        {"event": "CARD_FITS_RESERVED", "reservation_key": "CONT:OLD", "category": "continuation",
         "card_id": "CONT", "stage": "OLD", "fits": 25},
        {"event": "CARD_FITS_RESERVED", "reservation_key": "RETRY:OLD", "category": "technical_retry",
         "card_id": "RETRY", "stage": "OLD", "fits": 1},
        {"event": "RESERVATION_UNUSED_RELEASED", "reservation_key": "PILOT:OLD", "unused_fits": 3},
        {"event": "RESERVATION_UNUSED_RELEASED", "reservation_key": "CONT:OLD", "unused_fits": 1},
    ]
    for index in range(59):
        key = "PILOT:OLD" if index < 34 else "CONT:OLD" if index < 58 else "RETRY:OLD"
        card, stage = key.split(":")
        fit_id = f"OLD_{index:02d}"
        events.append({"event": "FIT_STARTED", "reservation_key": key, "card_id": card,
                       "stage": stage, "fit_id": fit_id})
        events.append({"event": "FIT_SAVED" if index < 57 else "FIT_FAILED", "fit_id": fit_id})
    while len(events) < 155:
        events.append({"event": "HISTORICAL_EVIDENCE", "sequence": len(events)})
    return events


def test_synthetic_end_to_end_and_persistence_resume(monkeypatch, tmp_path):
    monkeypatch.setattr(verifier, "REPORT_ROOT", tmp_path)
    receipt = verifier.run_synthetic_qa()
    assert receipt["status"] == "PASS"
    assert receipt["fit_type"] == "SYNTHETIC_ONLY"
    assert receipt["official_fit_budget_consumed"] == 0
    assert all(receipt["checks"].values())


def test_budget_amendment_is_exact_idempotent_and_enforces_cap(tmp_path):
    ledger = tmp_path / "budget.jsonl"
    events = _historical_prefix()
    _write_events(ledger, events)
    source_commit = "synthetic-pre-continuation-commit"
    freeze_signature = "synthetic-freeze-signature"
    freeze = {
        "freeze_signature": freeze_signature,
        "identity": {
            "source_commit": source_commit,
            "budget_checkpoint": {
                "event_rows": 155,
                "ledger_sha256": hashlib.sha256(ledger.read_bytes()).hexdigest(),
            "active_reservations_by_category": {
                "pilot": 34, "continuation": 24, "control_replay": 0, "technical_retry": 1,
                },
            "started": 59, "saved": 57, "failed": 2,
            },
        },
    }
    amendment = apply_cap_amendment(ledger, freeze=freeze, source_commit=source_commit)
    assert amendment["amendment_id"] == AMENDMENT_ID
    assert amendment["effective_caps"] == EFFECTIVE_CAPS
    assert apply_cap_amendment(ledger, freeze=freeze, source_commit=source_commit) == amendment

    continuation = reserve(ledger, card_id="A02", stage="SIX", fits=6,
                           freeze_signature=freeze_signature)
    assert reserve(ledger, card_id="A02", stage="SIX", fits=6,
                   freeze_signature=freeze_signature) == continuation
    for index in range(6):
        start_fit(ledger, fit_id=f"A02_{index}", card_id="A02", stage="SIX",
                  cutoff=59 + index, component="synthetic", freeze_signature=freeze_signature)
    try:
        start_fit(ledger, fit_id="A02_EXTRA", card_id="A02", stage="SIX",
                  cutoff=79, component="synthetic", freeze_signature=freeze_signature)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Continuation reservation must reject a seventh start")
    reserve(ledger, card_id="A02", stage="FINAL", fits=1, freeze_signature=freeze_signature)
    start_fit(ledger, fit_id="A02_FINAL", card_id="A02", stage="FINAL",
              cutoff=79, component="synthetic", freeze_signature=freeze_signature)
    accounting = verify_accounting(ledger, freeze_signature=freeze_signature)
    assert accounting["started"] == 66
    assert accounting["active_reserved"] == 66
    assert accounting["remaining_fit_capacity"] == 0
