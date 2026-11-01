from __future__ import annotations

import json

from scripts import optimization_v3_final_continuation_coordinator as coordinator
from src.models.optimization_v3_batch1 import reserve_card_budget


def test_resume_policy_reuses_only_complete_pairs_and_blocks_unknown_attempts() -> None:
    assert coordinator._resume_action(False, False, {"started": False, "failed": False}) == "FIT_ONCE"
    assert coordinator._resume_action(True, True, {"started": True, "failed": False}) == "VERIFY_EXISTING_BUNDLE"
    assert coordinator._resume_action(True, False, {"started": False, "failed": False}) == "BLOCK_PARTIAL_BUNDLE"
    assert coordinator._resume_action(False, True, {"started": False, "failed": False}) == "BLOCK_PARTIAL_BUNDLE"
    assert coordinator._resume_action(False, False, {"started": True, "failed": False}) == "BLOCK_PRIOR_ATTEMPT_WITHOUT_BUNDLE"
    assert coordinator._resume_action(False, False, {"started": False, "failed": True}) == "BLOCK_PRIOR_ATTEMPT_WITHOUT_BUNDLE"


def test_joint_budget_reservation_is_atomic_idempotent_and_does_not_reset_ledger(tmp_path) -> None:
    ledger = tmp_path / "batch_ledger.jsonl"
    ledger.write_text(json.dumps({
        "event": "BUDGET_RESERVED", "budget_key": "OLD:PILOT", "card_id": "OLD",
        "predictor_fits": 94, "cap": 180,
    }) + "\n", encoding="utf-8")
    first = reserve_card_budget(ledger, budget_key=coordinator.JOINT_KEY,
                               card_id=coordinator.JOINT_CARD,
                               predictor_fits=coordinator.JOINT_FITS, cap=180)
    second = reserve_card_budget(ledger, budget_key=coordinator.JOINT_KEY,
                                 card_id=coordinator.JOINT_CARD,
                                 predictor_fits=coordinator.JOINT_FITS, cap=180)
    events = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    reservations = [e for e in events if e.get("event") == "BUDGET_RESERVED"]
    assert first["status"] == "RESERVED"
    assert second["status"] == "EXISTING_RESERVATION"
    assert len(reservations) == 2
    assert sum(int(e["predictor_fits"]) for e in reservations) == 102
    assert reservations[-1]["budget_key"] == coordinator.JOINT_KEY

