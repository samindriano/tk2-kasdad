import json

import pytest

from src.models.research_next_budget import (
    DEFAULT_CAPS,
    ResearchNextBudgetError,
    authorize_new_phase,
    finish_fit,
    ledger_summary,
    read_events,
    release_unused,
    reserve_card,
    start_fit,
)


def test_new_phase_authorization_and_card_reservation_are_idempotent(tmp_path):
    ledger = tmp_path / "budget.jsonl"
    old = tmp_path / "old.jsonl"
    old.write_text('{"historic":true}\n', encoding="utf-8")
    old_before = old.read_bytes()
    inherited = {"sha256": "abc", "reserved_predictor_fits": 175,
                 "offledger_full_train_fits": 2, "effective_total": 177}
    first = authorize_new_phase(ledger, cap=60, category_caps=DEFAULT_CAPS,
                                inherited_old_budget_snapshot=inherited)
    second = authorize_new_phase(ledger, cap=60, category_caps=DEFAULT_CAPS,
                                 inherited_old_budget_snapshot=inherited)
    assert first == second
    reserved = reserve_card(ledger, card_id="N01", stage="PILOT", fits=4, category="pilot")
    assert reserve_card(ledger, card_id="N01", stage="PILOT", fits=4, category="pilot") == reserved
    started = start_fit(ledger, fit_id="N01:PILOT:55:CANDIDATE", card_id="N01",
                        stage="PILOT", cutoff=55, component="candidate")
    assert start_fit(ledger, fit_id="N01:PILOT:55:CANDIDATE", card_id="N01",
                     stage="PILOT", cutoff=55, component="candidate") == started
    finish_fit(ledger, fit_id="N01:PILOT:55:CANDIDATE", status="FIT_FAILED", error="synthetic")
    release_unused(ledger, card_id="N01", stage="PILOT")
    summary = ledger_summary(ledger)
    assert summary["authorized_cap"] == 60
    assert summary["started_and_consumed"] == 1
    assert summary["failed_or_interrupted_started"] == 1
    assert summary["reserved_total"] == 4
    assert summary["released_unused"] == 3
    assert summary["active_reserved"] == 1
    assert old.read_bytes() == old_before
    assert len(read_events(ledger)) == 5


def test_fit_requires_authorization_and_reservation(tmp_path):
    ledger = tmp_path / "budget.jsonl"
    with pytest.raises(ResearchNextBudgetError, match="not authorized"):
        reserve_card(ledger, card_id="N01", stage="PILOT", fits=4, category="pilot")
    authorize_new_phase(ledger, cap=60, category_caps=DEFAULT_CAPS,
                        inherited_old_budget_snapshot={"effective_total": 177})
    with pytest.raises(ResearchNextBudgetError, match="no unique card reservation"):
        start_fit(ledger, fit_id="N01:PILOT:55:CANDIDATE", card_id="N01",
                  stage="PILOT", cutoff=55, component="candidate")


def test_total_and_category_caps_fail_closed(tmp_path):
    ledger = tmp_path / "budget.jsonl"
    caps = {"pilot": 4, "continuation": 0, "control_replay": 0, "technical_retry": 0}
    authorize_new_phase(ledger, cap=4, category_caps=caps,
                        inherited_old_budget_snapshot={"effective_total": 177})
    reserve_card(ledger, card_id="N01", stage="PILOT", fits=4, category="pilot")
    with pytest.raises(ResearchNextBudgetError, match="would exceed cap"):
        reserve_card(ledger, card_id="N02", stage="PILOT", fits=1, category="pilot")


def test_authorization_mismatch_does_not_reset_existing_ledger(tmp_path):
    ledger = tmp_path / "budget.jsonl"
    authorize_new_phase(ledger, cap=60, category_caps=DEFAULT_CAPS,
                        inherited_old_budget_snapshot={"effective_total": 177})
    before = ledger.read_bytes()
    with pytest.raises(ResearchNextBudgetError, match="differs"):
        authorize_new_phase(ledger, cap=60, category_caps=DEFAULT_CAPS,
                            inherited_old_budget_snapshot={"effective_total": 179})
    assert ledger.read_bytes() == before
    assert all(isinstance(json.loads(line), dict) for line in ledger.read_text().splitlines())
