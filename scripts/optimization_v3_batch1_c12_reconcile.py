"""Validate and idempotently reconcile the completed C12 pilot budget."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.optimization_v3_batch1 import (  # noqa: E402
    BatchIntegrityError,
    append_event,
    file_sha256,
    write_json_exclusive,
)

REPORT_DIR = ROOT / "reports/optimization_v3/batch1/C12"
EXPERIMENT_DIR = ROOT / "experiments/optimization_v3/batch1/C12"
RUN_DIR = ROOT / "experiments/runs/optimization_v3_batch1/C12"
LEDGER_PATH = RUN_DIR.parent / "batch_ledger.jsonl"
RECONCILIATION_PATH = REPORT_DIR / "budget_reconciliation.json"
REQUIRED_FREEZE = EXPERIMENT_DIR / "freeze_revision_02.json"
REQUIRED_ATTEMPTS = REPORT_DIR / "attempts.jsonl"


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BatchIntegrityError(f"Cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BatchIntegrityError(f"Expected JSON object in {path}")
    return value


def _source_hashes_match(freeze: dict[str, Any]) -> bool:
    identity = freeze.get("identity", {})
    for raw_path, expected in identity.get("source_sha256", {}).items():
        path = Path(raw_path)
        if not path.is_file() or file_sha256(path) != expected:
            return False
    return True


def _attempt_events() -> list[dict[str, Any]]:
    try:
        lines = REQUIRED_ATTEMPTS.read_text(encoding="utf-8").splitlines()
        events = [json.loads(line) for line in lines if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise BatchIntegrityError(f"C12 attempt log is invalid: {exc}") from exc
    if any(not isinstance(event, dict) for event in events):
        raise BatchIntegrityError("C12 attempt log contains a non-object event")
    return events


def reconcile() -> dict[str, Any]:
    results_path = REPORT_DIR / "results.json"
    verification_path = REPORT_DIR / "verification.json"
    gate_path = REPORT_DIR / "gate.json"
    qa_path = REPORT_DIR / "independent_qa.json"
    result, verification, gate, qa, freeze = (
        _json(results_path), _json(verification_path), _json(gate_path),
        _json(qa_path), _json(REQUIRED_FREEZE),
    )
    signature = freeze.get("signature")
    if not signature or any(
        item.get("freeze_signature") != signature
        for item in (result, verification)
    ):
        raise BatchIntegrityError("C12 results/QA do not share the committed revision 02 freeze")
    if (result.get("card_id") != "C12"
            or result.get("status") != "PILOT_COMPLETE_RETIRE"
            or result.get("fit_predictor_count") != 2
            or result.get("fit_accounting", {}).get("budget_reserved_predictor_fits") != 2
            or result.get("fit_accounting", {}).get("successful_predictor_fits") != 2
            or result.get("test_data_used") is not False):
        raise BatchIntegrityError("C12 results are incomplete, miscounted, or used Test")
    if (verification.get("status") != "INDEPENDENT_QA_PASS"
            or verification.get("decision") != "RETIRE"
            or verification.get("fit_predictor_count") != 2
            or verification.get("test_data_used") is not False
            or verification.get("kaggle_access") is not False
            or gate.get("status") != "RETIRE"
            or gate.get("checks", {}).get("primary_gain_at_least_0_0005") is not True
            or gate.get("checks", {}).get("H12_harm_at_most_0_001") is not False):
        raise BatchIntegrityError("C12 independent verification/gate do not support RETIRE")
    if (qa.get("status") != "INDEPENDENT_QA_PASS"
            or qa.get("freeze_signature") != signature
            or qa.get("decision") != "RETIRE"
            or qa.get("test_data_used") is not False
            or qa.get("kaggle_access") is not False):
        raise BatchIntegrityError("C12 saved independent QA does not support reconciliation")
    if not _source_hashes_match(freeze):
        raise BatchIntegrityError("A C12 frozen source hash no longer matches")

    attempts = _attempt_events()
    counts = {name: sum(event.get("event") == name for event in attempts)
              for name in ("PREFLIGHT_PASS", "FIT_STARTED", "FIT_SAVED", "FIT_FAILED")}
    cutoffs = sorted(event.get("cutoff") for event in attempts if event.get("event") == "FIT_STARTED")
    if counts != {"PREFLIGHT_PASS": 1, "FIT_STARTED": 2, "FIT_SAVED": 2, "FIT_FAILED": 0}:
        raise BatchIntegrityError(f"Unexpected C12 attempt accounting: {counts}")
    if cutoffs != [55, 67]:
        raise BatchIntegrityError(f"Unexpected C12 fit origins: {cutoffs}")

    ledger_lines = LEDGER_PATH.read_text(encoding="utf-8").splitlines() if LEDGER_PATH.exists() else []
    ledger = [json.loads(line) for line in ledger_lines if line.strip()]
    reservations = [event for event in ledger
                    if event.get("event") == "BUDGET_RESERVED"
                    and event.get("budget_key") == "C12:PILOT"]
    if (len(reservations) != 1
            or reservations[0].get("predictor_fits") != 2
            or reservations[0].get("reserved_total_after") != 68):
        raise BatchIntegrityError("C12 reservation is missing, duplicated, or has the wrong total")

    hashes = {
        "results_sha256": file_sha256(results_path),
        "verification_sha256": file_sha256(verification_path),
        "gate_sha256": file_sha256(gate_path),
        "independent_qa_sha256": file_sha256(qa_path),
    }
    existing = [event for event in ledger
                if event.get("event") == "FIT_BUDGET_RECONCILED"
                and event.get("card_id") == "C12"]
    if len(existing) > 1:
        raise BatchIntegrityError("C12 has duplicate runtime budget reconciliation events")
    if existing:
        event = existing[0]
        if (event.get("budget_key") != "C12:PILOT"
                or event.get("freeze_signature") != signature
                or any(event.get(key) != value for key, value in hashes.items())
                or event.get("fit_started_events") != 2
                or event.get("fit_saved_events") != 2
                or event.get("fit_failed_events") != 0):
            raise BatchIntegrityError("Existing C12 reconciliation conflicts with saved evidence")
    else:
        event = {
            "event": "FIT_BUDGET_RECONCILED",
            "at_utc": datetime.now(timezone.utc).isoformat(),
            "card_id": "C12",
            "budget_key": "C12:PILOT",
            "attempt_accounting_complete": True,
            "reserved_predictor_fits": 2,
            "successful_predictor_fits": 2,
            "fit_started_events": 2,
            "fit_saved_events": 2,
            "fit_failed_events": 0,
            "technical_retries": 0,
            "resumed_verified_events": 0,
            "reserved_global_total_after": 68,
            "decision": "RETIRE",
            "freeze_signature": signature,
            "pre_reconciliation_ledger_sha256": hashlib.sha256(
                ("\n".join(ledger_lines) + ("\n" if ledger_lines else "")).encode("utf-8")
            ).hexdigest(),
            **hashes,
        }
        append_event(LEDGER_PATH, event)

    record = {
        "schema_version": 1,
        "card_id": "C12",
        "status": "FIT_BUDGET_RECONCILED",
        "decision": "RETIRE",
        "freeze_signature": signature,
        "attempt_counts": counts,
        "reserved_predictor_fits": 2,
        "successful_predictor_fits": 2,
        "reserved_global_total_after": 68,
        "source_hashes_match_freeze": True,
        **hashes,
        "runtime_ledger_event": event,
    }
    if RECONCILIATION_PATH.exists():
        if _json(RECONCILIATION_PATH) != record:
            raise BatchIntegrityError("Existing C12 reconciliation report differs")
    else:
        write_json_exclusive(RECONCILIATION_PATH, record)
    return record


if __name__ == "__main__":
    print(json.dumps(reconcile(), indent=2, sort_keys=True, allow_nan=False))
