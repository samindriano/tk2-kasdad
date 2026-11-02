"""Append-only, idempotent V3 budget transfer for C19/A02.

The historical budget module is frozen.  This adapter validates the single
authorized transfer against the live ledger, then reuses its ``FIT_STARTED``
and ``FIT_SAVED`` event functions for consumed-fit accounting.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any

from src.models.research_next_budget import read_events


AMENDMENT_ID = "C19_A02_PILOT_TRANSFER_2026_11_02"
EXPECTED_PREFIX_ROWS = 142
EXPECTED_PREFIX_SHA256 = "4157abd03cbad744350793ea33109154e813bb726fe0c27c346e971ad4ec538c"
GLOBAL_CAP = 60
PRIOR_CAPS = {"pilot": 30, "continuation": 24, "control_replay": 2, "technical_retry": 4}
EFFECTIVE_CAPS = {"pilot": 34, "continuation": 24, "control_replay": 0, "technical_retry": 2}
EXPECTED_ACTIVE = {"pilot": 30, "continuation": 23, "control_replay": 0, "technical_retry": 1}


class BudgetAmendmentError(RuntimeError):
    pass


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _json_line(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def _active_by_category(events: list[dict[str, Any]]) -> tuple[dict[str, int], int]:
    reservations = [event for event in events if event.get("event") == "CARD_FITS_RESERVED"]
    released = {event.get("reservation_key"): int(event.get("unused_fits", 0))
                for event in events if event.get("event") == "RESERVATION_UNUSED_RELEASED"}
    counts = Counter()
    total = 0
    for reservation in reservations:
        active = int(reservation["fits"]) - released.get(reservation["reservation_key"], 0)
        if active < 0:
            raise BudgetAmendmentError("Budget release exceeds a reservation")
        counts[str(reservation.get("category"))] += active
        total += active
    return dict(counts), total


def _latest_effective_caps(events: list[dict[str, Any]]) -> dict[str, int]:
    amendments = [event for event in events if event.get("event") == "BUDGET_CATEGORY_AMENDMENT"]
    if len(amendments) != 1 or amendments[0].get("amendment_id") != "A01_PILOT_REALLOCATION_2026_11_02":
        raise BudgetAmendmentError("Expected exactly the preserved A01 budget amendment")
    caps = amendments[0].get("effective_caps")
    if caps != PRIOR_CAPS:
        raise BudgetAmendmentError(f"Unexpected prior effective caps: {caps}")
    return {str(k): int(v) for k, v in caps.items()}


def apply_c19_a02_transfer(ledger_path: str | Path, *, source_commit: str) -> dict[str, Any]:
    """Append one exact 4-slot transfer; never resets or rewrites prior events."""
    path = Path(ledger_path)
    if not path.is_file():
        raise BudgetAmendmentError("Research Next ledger is missing")
    events = read_events(path)
    prior = [event for event in events if event.get("event") == "BUDGET_CATEGORY_AMENDMENT"
             and event.get("amendment_id") == AMENDMENT_ID]
    if prior:
        if len(prior) != 1:
            raise BudgetAmendmentError("Duplicate C19/A02 budget amendment ID")
        event = prior[0]
        expected_fixed = {
            "event": "BUDGET_CATEGORY_AMENDMENT", "amendment_id": AMENDMENT_ID,
            "global_cap": GLOBAL_CAP, "original_caps": PRIOR_CAPS,
            "effective_caps": EFFECTIVE_CAPS, "source_category": "control_replay",
            "source_fits": 2, "second_source_category": "technical_retry",
            "second_source_fits": 2, "destination_category": "pilot", "destination_fits": 4,
            "source_commit": source_commit,
        }
        if any(event.get(key) != value for key, value in expected_fixed.items()):
            raise BudgetAmendmentError("Existing C19/A02 amendment differs; refusing duplicate or reset")
        return event

    if len(events) != EXPECTED_PREFIX_ROWS or _digest(path) != EXPECTED_PREFIX_SHA256:
        raise BudgetAmendmentError("Ledger prefix differs from the reviewed 142-row checkpoint")
    if not source_commit:
        raise BudgetAmendmentError("Preparation source commit is required")
    original = next((event for event in events if event.get("event") == "BATCH_ALLOWANCE_AUTHORIZED"), None)
    if not original or int(original.get("cap", 0)) != GLOBAL_CAP:
        raise BudgetAmendmentError("Original global 60-fit authorization is missing")
    if _latest_effective_caps(events) != PRIOR_CAPS:
        raise BudgetAmendmentError("Prior effective category caps differ")
    active, total = _active_by_category(events)
    if active != EXPECTED_ACTIVE or total != 54:
        raise BudgetAmendmentError(f"Live active fit reservations differ: {active}, total={total}")

    event: dict[str, Any] = {
        "event": "BUDGET_CATEGORY_AMENDMENT",
        "batch_id": "optimization_v3_research_next_v1",
        "amendment_id": AMENDMENT_ID,
        "global_cap": GLOBAL_CAP,
        "original_caps": PRIOR_CAPS,
        "prior_effective_caps": PRIOR_CAPS,
        "source_category": "control_replay",
        "source_fits": 2,
        "second_source_category": "technical_retry",
        "second_source_fits": 2,
        "destination_category": "pilot",
        "destination_fits": 4,
        "effective_caps": EFFECTIVE_CAPS,
        "active_reservations_before": active,
        "active_total_before": total,
        "remaining_global_before": GLOBAL_CAP - total,
        "transferred_slots_are_unreserved_and_unstarted": True,
        "pre_amendment_event_rows": len(events),
        "pre_amendment_ledger_sha256": _digest(path),
        "source_commit": source_commit,
        "reason": "User-authorized four-fit C19/A02 scope: transfer two unused control-replay and two unused technical-retry slots to pilot; reserve one continuation slot for C19; global cap remains 60.",
    }
    encoded = _json_line(event)
    fd = os.open(path, os.O_APPEND | os.O_WRONLY)
    try:
        with os.fdopen(fd, "ab", closefd=False) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)
    return json.loads(encoded)


def reserve_with_effective_caps(
    ledger_path: str | Path, *, card_id: str, stage: str, fits: int,
    category: str, freeze_signature: str,
) -> dict[str, Any]:
    """Idempotently reserve a card using the amendment-aware category caps."""
    events = read_events(ledger_path)
    amendment = [event for event in events if event.get("event") == "BUDGET_CATEGORY_AMENDMENT"
                 and event.get("amendment_id") == AMENDMENT_ID]
    if len(amendment) != 1 or amendment[0].get("effective_caps") != EFFECTIVE_CAPS:
        raise BudgetAmendmentError("C19/A02 budget transfer has not been applied")
    if category not in EFFECTIVE_CAPS or int(fits) <= 0:
        raise BudgetAmendmentError("Invalid reservation category or fit count")
    key = f"{card_id}:{stage}"
    found = [event for event in events if event.get("event") == "CARD_FITS_RESERVED"
             and event.get("reservation_key") == key]
    if len(found) > 1:
        raise BudgetAmendmentError(f"Duplicate reservation {key}")
    if found:
        expected = {"fits": int(fits), "category": category, "card_id": card_id,
                    "stage": stage, "freeze_signature": freeze_signature,
                    "amendment_id": AMENDMENT_ID}
        if any(found[0].get(name) != value for name, value in expected.items()):
            raise BudgetAmendmentError(f"Existing reservation differs for {key}")
        return found[0]

    active, total = _active_by_category(events)
    if total + int(fits) > GLOBAL_CAP:
        raise BudgetAmendmentError("Reservation would exceed global cap")
    if active.get(category, 0) + int(fits) > EFFECTIVE_CAPS[category]:
        raise BudgetAmendmentError(f"Reservation would exceed {category} cap")
    event = {
        "event": "CARD_FITS_RESERVED",
        "batch_id": "optimization_v3_research_next_v1",
        "reservation_key": key,
        "card_id": card_id,
        "stage": stage,
        "category": category,
        "fits": int(fits),
        "amendment_id": AMENDMENT_ID,
        "freeze_signature": freeze_signature,
        "effective_category_cap": EFFECTIVE_CAPS[category],
        "global_cap": GLOBAL_CAP,
    }
    fd = os.open(Path(ledger_path), os.O_APPEND | os.O_WRONLY)
    try:
        with os.fdopen(fd, "ab", closefd=False) as stream:
            stream.write(_json_line(event))
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)
    return event


__all__ = [
    "AMENDMENT_ID", "BudgetAmendmentError", "EFFECTIVE_CAPS", "GLOBAL_CAP",
    "apply_c19_a02_transfer", "reserve_with_effective_caps",
]
