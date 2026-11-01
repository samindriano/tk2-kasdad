"""Accounting correction for the zero-fit N06 verification amendment.

The original N06 independent verifier reconstructs every bundle, metric,
subgroup, and per-cycle result. Its final pilot-cap check compares gross
reservations with an active-reservation cap and therefore counts four N04
slots that were already released. This wrapper preserves that verifier and
independently reconciles the append-only ledger using the recorded release
events; it does not alter N06 recipes, bundles, predictions, or fit counts.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts import optimization_v3_research_next_n06 as runner
from scripts import optimization_v3_research_next_n06_verify as original_verifier
from src.models.research_next_budget import ledger_summary, read_events
from src.models.research_next_contract import canonical_sha256, file_sha256


REPORT_ROOT = runner.REPORT_ROOT.parent / "amendment_02_verification_correction"


def reconcile() -> dict[str, Any]:
    report = original_verifier.verify()
    checks = dict(report["checks"])
    expected_failed = ["global_and_pilot_caps_respected"]
    if report.get("failed_checks") != expected_failed:
        raise RuntimeError(
            "N06 accounting amendment is scoped only to the known gross-versus-active cap calculation; "
            f"original independent verification failures differ: {report.get('failed_checks')}"
        )

    events = read_events(runner.LEDGER_PATH)
    authorization = [event for event in events if event.get("event") == "BATCH_ALLOWANCE_AUTHORIZED"]
    reservations = [event for event in events if event.get("event") == "CARD_FITS_RESERVED"]
    releases = [event for event in events if event.get("event") == "RESERVATION_UNUSED_RELEASED"]
    if len(authorization) != 1:
        raise RuntimeError("N06 active-slot reconciliation requires one unchanged phase authorization")
    released_by_key = {}
    for event in releases:
        key = event.get("reservation_key")
        if key in released_by_key:
            raise RuntimeError(f"Duplicate unused-slot release for {key}")
        released_by_key[key] = int(event.get("unused_fits", 0))
    active_by_category: dict[str, int] = {}
    active_by_key: dict[str, int] = {}
    active_total = 0
    for reservation in reservations:
        key = reservation["reservation_key"]
        gross = int(reservation["fits"])
        released = released_by_key.get(key, 0)
        if released < 0 or released > gross:
            raise RuntimeError(f"Invalid release count for {key}: {released}/{gross}")
        active = gross - released
        active_total += active
        active_by_key[key] = active
        category = str(reservation["category"])
        active_by_category[category] = active_by_category.get(category, 0) + active

    n06_reservations = [event for event in reservations
                        if event.get("reservation_key") == "N06:PILOT"]
    n06_releases = [event for event in releases
                    if event.get("reservation_key") == "N06:PILOT"]
    summary = ledger_summary(runner.LEDGER_PATH)
    n06_checks = {
        "single_exact_N06_four_fit_reservation": (
            len(n06_reservations) == 1 and int(n06_reservations[0]["fits"]) == 4
            and n06_reservations[0].get("category") == "pilot"
        ),
        "N06_unused_release_zero_once": len(n06_releases) == 1 and int(n06_releases[0]["unused_fits"]) == 0,
        "active_total_equals_summary_and_within_60": (
            active_total == int(summary["active_reserved"])
            and active_total <= int(authorization[0]["cap"]) == 60
        ),
        "active_pilot_with_releases_within_26": (
            active_by_category.get("pilot", 0) <= int(authorization[0]["category_caps"]["pilot"]) == 26
        ),
        "N04_four_released_slots_subtracted_once": (
            released_by_key.get("N04:PILOT") == 4 and active_by_key.get("N04:PILOT") == 4
        ),
        "N06_four_started_saved_no_retry_or_failure": (
            summary["started_and_consumed"] == summary["saved"] == 26
            and summary["failed_or_interrupted_started"] == 0
            and active_by_key.get("N06:PILOT") == 4
        ),
    }
    if not all(n06_checks.values()):
        raise RuntimeError(f"N06 active reservation reconciliation failed: {n06_checks}")

    checks["global_and_pilot_caps_respected"] = True
    report["checks"] = checks
    report["failed_checks"] = [key for key, passed in checks.items() if not passed]
    report["status"] = "PASS" if not report["failed_checks"] else "FAIL"
    report["accounting_amendment"] = {
        "status": "ZERO_FIT_VERIFICATION_CORRECTION",
        "reason": "Original verifier compared gross pilot reservations (30) against active pilot cap (26); it omitted the previously recorded N04:PILOT release of four unused slots.",
        "original_N06_verification_preserved": str(runner.REPORT_ROOT / "verification.json"),
        "original_failed_check": expected_failed[0],
        "recomputed_active_reservations": {
            "total": active_total, "by_category": active_by_category,
            "by_key": active_by_key, "released_unused_by_key": released_by_key,
        },
        "ledger_reconciliation_checks": n06_checks,
        "new_estimator_fits": 0,
        "recipe_or_prediction_changed": False,
    }
    report["budget"] = summary
    report["source_lineage"] = {
        "original_freeze_signature": json.loads(runner.FREEZE_PATH.read_text(encoding="utf-8"))["freeze_signature"],
        "original_result_sha256": file_sha256(runner.REPORT_ROOT / "results.json"),
        "original_failed_verification_sha256": file_sha256(runner.REPORT_ROOT / "verification.json"),
        "amendment_source_sha256": file_sha256(Path(__file__)),
        "amendment_source_commit": runner._git_head(),
    }
    return report


def _write_once(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if canonical_sha256(previous) != canonical_sha256(value):
            raise RuntimeError(f"Refusing to replace amendment report: {path}")
        return
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(data)
        stream.flush()


if __name__ == "__main__":
    result = reconcile()
    _write_once(REPORT_ROOT / "verification.json", result)
    summary = {
        "status": result["status"],
        "amendment": "zero-fit accounting reconciliation only",
        "original_N06_verdict": "COMPLETE_INFORMATION; no continuation eligibility",
        "fit_count": 4,
        "fit_ids": result["verified_fit_ids"],
        "fit_verification": result["checks"],
        "accounting": result["accounting_amendment"],
        "budget": result["budget"],
        "source_lineage": result["source_lineage"],
    }
    _write_once(REPORT_ROOT / "SUMMARY.json", summary)
    print(json.dumps(summary, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False))
    raise SystemExit(0 if result["status"] == "PASS" else 1)
