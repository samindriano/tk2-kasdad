"""Append-only accounting for the separately authorized Research Next V1 cap."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Mapping


class ResearchNextBudgetError(RuntimeError):
    pass


DEFAULT_CAPS = {
    "pilot": 26,
    "continuation": 24,
    "control_replay": 6,
    "technical_retry": 4,
}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def read_events(path: str | Path) -> list[dict[str, Any]]:
    ledger = Path(path)
    if not ledger.exists():
        return []
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(ledger.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ResearchNextBudgetError(f"Malformed Research Next ledger line {line_number}") from exc
        if not isinstance(value, dict) or not isinstance(value.get("event"), str):
            raise ResearchNextBudgetError(f"Invalid Research Next ledger event on line {line_number}")
        events.append(value)
    return events


def _append(path: str | Path, event: Mapping[str, Any]) -> dict[str, Any]:
    ledger = Path(path)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    value = dict(event)
    value.setdefault("at_utc", datetime.now(timezone.utc).isoformat())
    encoded = (_canonical(value) + "\n").encode("utf-8")
    fd = os.open(ledger, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "ab", closefd=False) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)
    return value


def _authorization(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    authorizations = [event for event in events if event.get("event") == "BATCH_ALLOWANCE_AUTHORIZED"]
    if len(authorizations) > 1:
        raise ResearchNextBudgetError("Duplicate new-phase budget authorization in ledger")
    return authorizations[0] if authorizations else None


def authorize_new_phase(
    ledger_path: str | Path,
    *,
    cap: int,
    category_caps: Mapping[str, int],
    inherited_old_budget_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Record the user's new-phase authorization once, without touching the old ledger."""
    events = read_events(ledger_path)
    if cap <= 0 or any(int(value) < 0 for value in category_caps.values()):
        raise ResearchNextBudgetError("Invalid Research Next budget cap")
    existing = _authorization(events)
    expected = {
        "event": "BATCH_ALLOWANCE_AUTHORIZED",
        "batch_id": "optimization_v3_research_next_v1",
        "cap": int(cap),
        "category_caps": {str(key): int(value) for key, value in category_caps.items()},
        "inherited_old_budget_snapshot": dict(inherited_old_budget_snapshot),
        "old_phase_budget_reset": False,
    }
    if existing is not None:
        if any(existing.get(key) != value for key, value in expected.items()):
            raise ResearchNextBudgetError("Existing new-phase authorization differs; refusing reset")
        return existing
    if events:
        raise ResearchNextBudgetError("New-phase ledger has events but no authorization record")
    return _append(ledger_path, expected)


def reserve_card(
    ledger_path: str | Path,
    *,
    card_id: str,
    stage: str,
    fits: int,
    category: str,
) -> dict[str, Any]:
    """Reserve a complete card stage idempotently before any estimator fit."""
    events = read_events(ledger_path)
    authorization = _authorization(events)
    if authorization is None:
        raise ResearchNextBudgetError("New-phase allowance is not authorized in the ledger")
    fits = int(fits)
    if fits <= 0:
        raise ResearchNextBudgetError("A card reservation must have a positive fit count")
    caps = authorization.get("category_caps", {})
    if category not in caps:
        raise ResearchNextBudgetError(f"Unknown budget category {category!r}")
    key = f"{card_id}:{stage}"
    existing = [event for event in events if event.get("event") == "CARD_FITS_RESERVED"
                and event.get("reservation_key") == key]
    if len(existing) > 1:
        raise ResearchNextBudgetError(f"Duplicate reservation key {key}")
    if existing:
        reservation = existing[0]
        if (int(reservation.get("fits", -1)) != fits or reservation.get("category") != category
                or reservation.get("card_id") != card_id or reservation.get("stage") != stage):
            raise ResearchNextBudgetError(f"Existing reservation differs for {key}")
        return reservation

    reservations = [event for event in events if event.get("event") == "CARD_FITS_RESERVED"]
    released = {event.get("reservation_key"): int(event.get("unused_fits", 0))
                for event in events if event.get("event") == "RESERVATION_UNUSED_RELEASED"}
    active_total = sum(int(event["fits"]) - released.get(event["reservation_key"], 0)
                       for event in reservations)
    active_category = sum(int(event["fits"]) - released.get(event["reservation_key"], 0)
                          for event in reservations if event.get("category") == category)
    cap = int(authorization["cap"])
    category_cap = int(caps[category])
    if active_total + fits > cap or active_category + fits > category_cap:
        raise ResearchNextBudgetError(
            f"Reservation {key} would exceed cap: total {active_total}+{fits}/{cap}, "
            f"{category} {active_category}+{fits}/{category_cap}"
        )
    return _append(ledger_path, {
        "event": "CARD_FITS_RESERVED", "batch_id": authorization["batch_id"],
        "reservation_key": key, "card_id": card_id, "stage": stage,
        "category": category, "fits": fits,
    })


def start_fit(
    ledger_path: str | Path,
    *,
    fit_id: str,
    card_id: str,
    stage: str,
    cutoff: int,
    component: str,
) -> dict[str, Any]:
    """Durably consume one authorized fit immediately before estimator.fit."""
    events = read_events(ledger_path)
    key = f"{card_id}:{stage}"
    reservations = [event for event in events if event.get("event") == "CARD_FITS_RESERVED"
                    and event.get("reservation_key") == key]
    if len(reservations) != 1:
        raise ResearchNextBudgetError(f"Fit {fit_id} has no unique card reservation")
    existing = [event for event in events if event.get("event") == "FIT_STARTED" and event.get("fit_id") == fit_id]
    if len(existing) > 1:
        raise ResearchNextBudgetError(f"Fit {fit_id} has duplicate start events")
    if existing:
        return existing[0]
    starts = [event for event in events if event.get("event") == "FIT_STARTED"
              and event.get("reservation_key") == key]
    if len(starts) >= int(reservations[0]["fits"]):
        raise ResearchNextBudgetError(f"Fit reservation {key} is exhausted")
    authorization = _authorization(events)
    all_started = sum(event.get("event") == "FIT_STARTED" for event in events)
    if all_started >= int(authorization["cap"]):
        raise ResearchNextBudgetError("Research Next estimator fit cap is exhausted")
    return _append(ledger_path, {
        "event": "FIT_STARTED", "batch_id": authorization["batch_id"],
        "reservation_key": key, "card_id": card_id, "stage": stage,
        "fit_id": fit_id, "cutoff": int(cutoff), "component": component,
        "fit_consumed_at_start": True,
    })


def finish_fit(
    ledger_path: str | Path,
    *,
    fit_id: str,
    status: str,
    receipt_sha256: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Record a terminal fit status; started failures remain consumed."""
    if status not in {"FIT_SAVED", "FIT_FAILED"}:
        raise ResearchNextBudgetError("Fit terminal status must be FIT_SAVED or FIT_FAILED")
    events = read_events(ledger_path)
    started = [event for event in events if event.get("event") == "FIT_STARTED" and event.get("fit_id") == fit_id]
    if len(started) != 1:
        raise ResearchNextBudgetError(f"Fit {fit_id} must have one FIT_STARTED event")
    terminal = [event for event in events if event.get("fit_id") == fit_id
                and event.get("event") in {"FIT_SAVED", "FIT_FAILED"}]
    if len(terminal) > 1:
        raise ResearchNextBudgetError(f"Fit {fit_id} already has conflicting terminal events")
    if terminal:
        event = terminal[0]
        if event.get("event") != status or event.get("receipt_sha256") != receipt_sha256:
            raise ResearchNextBudgetError(f"Fit {fit_id} terminal status differs")
        return event
    return _append(ledger_path, {
        "event": status, "batch_id": started[0]["batch_id"],
        "reservation_key": started[0]["reservation_key"], "card_id": started[0]["card_id"],
        "stage": started[0]["stage"], "fit_id": fit_id, "cutoff": started[0]["cutoff"],
        "receipt_sha256": receipt_sha256, "error": error,
    })


def release_unused(ledger_path: str | Path, *, card_id: str, stage: str) -> dict[str, Any]:
    """Release only unstarted slots after a card stage is terminal."""
    events = read_events(ledger_path)
    key = f"{card_id}:{stage}"
    reservations = [event for event in events if event.get("event") == "CARD_FITS_RESERVED"
                    and event.get("reservation_key") == key]
    if len(reservations) != 1:
        raise ResearchNextBudgetError(f"Cannot release absent/duplicate reservation {key}")
    previous = [event for event in events if event.get("event") == "RESERVATION_UNUSED_RELEASED"
                and event.get("reservation_key") == key]
    if previous:
        if len(previous) != 1:
            raise ResearchNextBudgetError(f"Duplicate unused release for {key}")
        return previous[0]
    started = sum(event.get("event") == "FIT_STARTED" and event.get("reservation_key") == key for event in events)
    pending = [event for event in events if event.get("event") == "FIT_STARTED"
               and event.get("reservation_key") == key
               and not any(done.get("fit_id") == event.get("fit_id")
                           and done.get("event") in {"FIT_SAVED", "FIT_FAILED"} for done in events)]
    if pending:
        raise ResearchNextBudgetError(f"Cannot release reservation {key} with unfinished started fits")
    unused = int(reservations[0]["fits"]) - started
    if unused < 0:
        raise ResearchNextBudgetError(f"More fits started than reserved for {key}")
    return _append(ledger_path, {
        "event": "RESERVATION_UNUSED_RELEASED", "reservation_key": key,
        "card_id": card_id, "stage": stage, "unused_fits": unused,
    })


def ledger_summary(path: str | Path) -> dict[str, Any]:
    events = read_events(path)
    authorization = _authorization(events)
    reservations = [event for event in events if event.get("event") == "CARD_FITS_RESERVED"]
    starts = [event for event in events if event.get("event") == "FIT_STARTED"]
    saved = [event for event in events if event.get("event") == "FIT_SAVED"]
    failed = [event for event in events if event.get("event") == "FIT_FAILED"]
    released = sum(int(event.get("unused_fits", 0)) for event in events
                   if event.get("event") == "RESERVATION_UNUSED_RELEASED")
    reserved_total = sum(int(event["fits"]) for event in reservations)
    return {
        "event_rows": len(events),
        "authorized_cap": int(authorization["cap"]) if authorization else 0,
        "reserved_total": reserved_total,
        "released_unused": released,
        "active_reserved": reserved_total - released,
        "started_and_consumed": len(starts),
        "saved": len(saved),
        "failed_or_interrupted_started": len(failed),
        "remaining_capacity_after_reservations": (int(authorization["cap"]) - reserved_total) if authorization else 0,
        "category_reservations": {
            category: sum(int(event["fits"]) for event in reservations if event.get("category") == category)
            for category in (authorization or {}).get("category_caps", {})
        },
    }


__all__ = [
    "DEFAULT_CAPS", "ResearchNextBudgetError", "authorize_new_phase", "finish_fit",
    "ledger_summary", "read_events", "release_unused", "reserve_card", "start_fit",
]
