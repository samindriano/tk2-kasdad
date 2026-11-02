from __future__ import annotations

import json

import pytest

from src.models.optimization_v3_budget_amendment import (
    AMENDMENT_ID,
    BudgetAmendmentError,
    EFFECTIVE_CAPS,
    reserve_with_effective_caps,
)


def _append(path, value):
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def test_amended_reservations_are_idempotent_and_stay_under_global_and_category_caps(tmp_path):
    ledger = tmp_path / "budget.jsonl"
    _append(ledger, {"event": "BATCH_ALLOWANCE_AUTHORIZED", "cap": 60,
                     "category_caps": {"pilot": 26, "continuation": 24,
                                       "control_replay": 6, "technical_retry": 4}})
    _append(ledger, {"event": "BUDGET_CATEGORY_AMENDMENT", "amendment_id": AMENDMENT_ID,
                     "effective_caps": EFFECTIVE_CAPS})
    for category, fits in (("pilot", 30), ("continuation", 23), ("technical_retry", 1)):
        _append(ledger, {"event": "CARD_FITS_RESERVED", "reservation_key": f"prior-{category}",
                         "card_id": f"prior-{category}", "stage": "prior",
                         "category": category, "fits": fits})

    c19 = reserve_with_effective_caps(
        ledger, card_id="C19", stage="BENCHMARK", fits=1, category="continuation",
        freeze_signature="freeze-c19",
    )
    a02 = reserve_with_effective_caps(
        ledger, card_id="A02", stage="PILOT", fits=4, category="pilot",
        freeze_signature="freeze-a02",
    )
    assert c19["fits"] == 1 and a02["fits"] == 4
    assert reserve_with_effective_caps(
        ledger, card_id="A02", stage="PILOT", fits=4, category="pilot",
        freeze_signature="freeze-a02",
    ) == a02
    with pytest.raises(BudgetAmendmentError, match="pilot cap"):
        reserve_with_effective_caps(
            ledger, card_id="A02_EXTRA", stage="PILOT", fits=1, category="pilot",
            freeze_signature="freeze-a02-extra",
        )
