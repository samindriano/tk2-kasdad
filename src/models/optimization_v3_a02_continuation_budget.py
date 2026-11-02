"""Append-only Research Next cap amendment for the A02 continuation.

The historical authorization and prior amendments remain unchanged.  This
adapter records the explicitly authorized 60 -> 66 cap change and provides
amendment-aware reserve/start operations; fit terminal events still use the
shared Research Next ledger format.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any

from src.models.research_next_budget import read_events


AMENDMENT_ID = "A02_CONTINUATION_CAP66_2026_11_02"
GLOBAL_CAP = 66
PRIOR_CAPS = {"pilot": 34, "continuation": 24, "control_replay": 0, "technical_retry": 2}
EFFECTIVE_CAPS = {"pilot": 34, "continuation": 31, "control_replay": 0, "technical_retry": 1}
PRIOR_ACTIVE_RESERVATIONS = {"pilot": 34, "continuation": 24, "control_replay": 0, "technical_retry": 1}


class A02ContinuationBudgetError(RuntimeError):
    pass


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _append(path: Path, event: dict[str, Any]) -> None:
    encoded = (json.dumps(event, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    fd = os.open(path, os.O_APPEND | os.O_WRONLY)
    try:
        with os.fdopen(fd, "ab", closefd=False) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)


def _reservation_accounting(events: list[dict[str, Any]]) -> tuple[dict[str, int], int, int]:
    reservations = [e for e in events if e.get("event") == "CARD_FITS_RESERVED"]
    releases = [e for e in events if e.get("event") == "RESERVATION_UNUSED_RELEASED"]
    if len({e.get("reservation_key") for e in releases}) != len(releases):
        raise A02ContinuationBudgetError("Duplicate reservation release event")
    released = {e.get("reservation_key"): int(e.get("unused_fits", -1)) for e in releases}
    counts: Counter[str] = Counter()
    total = 0
    gross = 0
    for reservation in reservations:
        key = reservation.get("reservation_key")
        fits = int(reservation.get("fits", -1))
        unused = released.get(key, 0)
        if fits < 0 or unused < 0 or unused > fits:
            raise A02ContinuationBudgetError(f"Invalid reservation/release accounting for {key}")
        active = fits - unused
        counts[str(reservation.get("category"))] += active
        total += active
        gross += fits
    return dict(counts), total, gross - total


def apply_cap_amendment(
    ledger_path: str | Path,
    *,
    freeze: dict[str, Any],
    source_commit: str,
) -> dict[str, Any]:
    """Append the authorized amendment once, guarded by its frozen ledger prefix."""
    path = Path(ledger_path)
    if not path.is_file():
        raise A02ContinuationBudgetError("Research Next ledger is missing")
    events = read_events(path)
    existing = [e for e in events if e.get("event") == "BUDGET_CAP_AMENDMENT"
                and e.get("amendment_id") == AMENDMENT_ID]
    identity = freeze.get("identity", {})
    if existing:
        if len(existing) != 1:
            raise A02ContinuationBudgetError("Duplicate A02 continuation cap amendment")
        event = existing[0]
        expected = {
            "freeze_signature": freeze.get("freeze_signature"),
            "source_commit": source_commit,
            "pre_amendment_ledger_sha256": identity.get("budget_checkpoint", {}).get("ledger_sha256"),
            "pre_amendment_event_rows": identity.get("budget_checkpoint", {}).get("event_rows"),
            "global_cap": GLOBAL_CAP,
            "effective_caps": EFFECTIVE_CAPS,
        }
        if any(event.get(key) != value for key, value in expected.items()):
            raise A02ContinuationBudgetError("Existing A02 amendment differs from freeze; no reset allowed")
        return event

    checkpoint = identity.get("budget_checkpoint", {})
    if (len(events) != int(checkpoint.get("event_rows", -1))
            or _digest(path) != checkpoint.get("ledger_sha256")):
        raise A02ContinuationBudgetError("Live budget ledger differs from frozen pre-amendment checkpoint")
    authorization = [e for e in events if e.get("event") == "BATCH_ALLOWANCE_AUTHORIZED"]
    amendments = [e for e in events if e.get("event") == "BUDGET_CATEGORY_AMENDMENT"]
    if (len(authorization) != 1 or int(authorization[0].get("cap", -1)) != 60
            or len(amendments) != 2
            or amendments[-1].get("amendment_id") != "C19_A02_PILOT_TRANSFER_2026_11_02"
            or amendments[-1].get("effective_caps") != PRIOR_CAPS):
        raise A02ContinuationBudgetError("Prior authorization/amendment lineage differs from reviewed state")
    categories, active_total, released = _reservation_accounting(events)
    categories = {name: categories.get(name, 0) for name in EFFECTIVE_CAPS}
    starts = [e for e in events if e.get("event") == "FIT_STARTED"]
    terminal = [e for e in events if e.get("event") in {"FIT_SAVED", "FIT_FAILED"}]
    pending = {e.get("fit_id") for e in starts} - {e.get("fit_id") for e in terminal}
    retry_reservations = [e for e in events if e.get("event") == "CARD_FITS_RESERVED"
                          and e.get("category") == "technical_retry"]
    retry_started = [e for e in starts if any(r.get("reservation_key") == e.get("reservation_key")
                                               for r in retry_reservations)]
    expected = checkpoint
    if (categories != PRIOR_ACTIVE_RESERVATIONS or active_total != 59 or len(starts) != 59
            or len(terminal) != 59 or pending or released != 4
            or len(retry_reservations) != 1 or int(retry_reservations[0].get("fits", -1)) != 1
            or len(retry_started) != 1
            or expected.get("active_reservations_by_category") != categories
            or expected.get("started") != len(starts)
            or expected.get("saved") != 57 or expected.get("failed") != 2):
        raise A02ContinuationBudgetError(
            f"Live accounting conflicts with amendment: active={categories}/{active_total}, "
            f"starts={len(starts)}, saved={sum(e.get('event') == 'FIT_SAVED' for e in terminal)}, "
            f"failed={sum(e.get('event') == 'FIT_FAILED' for e in terminal)}, pending={sorted(map(str, pending))}"
        )
    if not source_commit or identity.get("source_commit") != source_commit:
        raise A02ContinuationBudgetError("Source commit differs from frozen continuation identity")

    event = {
        "event": "BUDGET_CAP_AMENDMENT",
        "batch_id": "optimization_v3_research_next_v1",
        "amendment_id": AMENDMENT_ID,
        "authorized_by": "Samuel, explicit continuation authorization",
        "global_cap": GLOBAL_CAP,
        "prior_global_cap": 60,
        "prior_effective_caps": PRIOR_CAPS,
        "effective_caps": EFFECTIVE_CAPS,
        "continuation_capacity_added": 7,
        "unused_technical_retry_capacity_released": 1,
        "active_reservations_before": categories,
        "active_total_before": active_total,
        "historical_started_saved_failed_before": {
            "started": len(starts),
            "saved": sum(e.get("event") == "FIT_SAVED" for e in terminal),
            "failed": sum(e.get("event") == "FIT_FAILED" for e in terminal),
        },
        "retry_reservations_and_starts_before": {"reserved": len(retry_reservations), "started": len(retry_started)},
        "pre_amendment_event_rows": len(events),
        "pre_amendment_ledger_sha256": _digest(path),
        "freeze_signature": freeze["freeze_signature"],
        "source_commit": source_commit,
        "reason": "Explicit user-authorized six-origin continuation and one conditional full-Train fit; one unreserved retry slot was released to keep category caps equal to global cap 66.",
    }
    _append(path, event)
    return event


def _amendment(events: list[dict[str, Any]], freeze_signature: str) -> dict[str, Any]:
    found = [e for e in events if e.get("event") == "BUDGET_CAP_AMENDMENT"
             and e.get("amendment_id") == AMENDMENT_ID]
    if (len(found) != 1 or found[0].get("freeze_signature") != freeze_signature
            or found[0].get("global_cap") != GLOBAL_CAP
            or found[0].get("effective_caps") != EFFECTIVE_CAPS):
        raise A02ContinuationBudgetError("Exact A02 cap amendment has not been applied")
    return found[0]


def reserve(
    ledger_path: str | Path,
    *,
    card_id: str,
    stage: str,
    fits: int,
    freeze_signature: str,
) -> dict[str, Any]:
    path = Path(ledger_path)
    events = read_events(path)
    amendment = _amendment(events, freeze_signature)
    fits = int(fits)
    if fits <= 0:
        raise A02ContinuationBudgetError("Reservation must contain at least one fit")
    key = f"{card_id}:{stage}"
    found = [e for e in events if e.get("event") == "CARD_FITS_RESERVED"
             and e.get("reservation_key") == key]
    if len(found) > 1:
        raise A02ContinuationBudgetError(f"Duplicate reservation {key}")
    if found:
        expected = {"fits": fits, "category": "continuation", "card_id": card_id,
                    "stage": stage, "freeze_signature": freeze_signature,
                    "amendment_id": AMENDMENT_ID}
        if any(found[0].get(k) != v for k, v in expected.items()):
            raise A02ContinuationBudgetError(f"Existing reservation differs for {key}")
        return found[0]
    category_active, total, _ = _reservation_accounting(events)
    if (total + fits > GLOBAL_CAP
            or category_active.get("continuation", 0) + fits > EFFECTIVE_CAPS["continuation"]):
        raise A02ContinuationBudgetError(
            f"Reservation exceeds amended cap: total {total}+{fits}/{GLOBAL_CAP}; "
            f"continuation {category_active.get('continuation', 0)}+{fits}/{EFFECTIVE_CAPS['continuation']}"
        )
    event = {
        "event": "CARD_FITS_RESERVED", "batch_id": "optimization_v3_research_next_v1",
        "reservation_key": key, "card_id": card_id, "stage": stage,
        "category": "continuation", "fits": fits, "amendment_id": AMENDMENT_ID,
        "freeze_signature": freeze_signature, "effective_category_cap": EFFECTIVE_CAPS["continuation"],
        "global_cap": GLOBAL_CAP,
    }
    _append(path, event)
    return event


def start_fit(
    ledger_path: str | Path,
    *,
    fit_id: str,
    card_id: str,
    stage: str,
    cutoff: int,
    component: str,
    freeze_signature: str,
) -> dict[str, Any]:
    path = Path(ledger_path)
    events = read_events(path)
    _amendment(events, freeze_signature)
    key = f"{card_id}:{stage}"
    reservations = [e for e in events if e.get("event") == "CARD_FITS_RESERVED"
                    and e.get("reservation_key") == key]
    if len(reservations) != 1 or reservations[0].get("freeze_signature") != freeze_signature:
        raise A02ContinuationBudgetError(f"No unique frozen reservation for {key}")
    prior_fit = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("fit_id") == fit_id]
    if len(prior_fit) > 1:
        raise A02ContinuationBudgetError(f"Duplicate start for {fit_id}")
    if prior_fit:
        return prior_fit[0]
    starts = [e for e in events if e.get("event") == "FIT_STARTED"]
    key_starts = [e for e in starts if e.get("reservation_key") == key]
    if len(key_starts) >= int(reservations[0]["fits"]):
        raise A02ContinuationBudgetError(f"Reservation exhausted for {key}")
    if len(starts) >= GLOBAL_CAP:
        raise A02ContinuationBudgetError("Amended global fit cap exhausted")
    event = {
        "event": "FIT_STARTED", "batch_id": "optimization_v3_research_next_v1",
        "reservation_key": key, "card_id": card_id, "stage": stage,
        "fit_id": fit_id, "cutoff": int(cutoff), "component": component,
        "fit_consumed_at_start": True, "amendment_id": AMENDMENT_ID,
        "freeze_signature": freeze_signature,
    }
    _append(path, event)
    return event


def verify_accounting(ledger_path: str | Path, *, freeze_signature: str) -> dict[str, Any]:
    events = read_events(ledger_path)
    amendment = _amendment(events, freeze_signature)
    categories, active_total, released = _reservation_accounting(events)
    starts = [e for e in events if e.get("event") == "FIT_STARTED"]
    return {
        "amendment_id": amendment["amendment_id"], "global_cap": GLOBAL_CAP,
        "effective_caps": EFFECTIVE_CAPS, "event_rows": len(events),
        "started": len(starts), "active_reserved": active_total,
        "active_reservations_by_category": {k: categories.get(k, 0) for k in EFFECTIVE_CAPS},
        "released_unused_total": released,
        "remaining_fit_capacity": GLOBAL_CAP - len(starts),
    }


__all__ = [
    "AMENDMENT_ID", "A02ContinuationBudgetError", "EFFECTIVE_CAPS", "GLOBAL_CAP",
    "apply_cap_amendment", "reserve", "start_fit", "verify_accounting",
]
