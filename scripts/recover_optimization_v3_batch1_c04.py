"""Read-only recovery/publication for saved C04 pilot receipts.

This recovery path never calls the training engine. It revalidates the frozen
identity, existing fit bundles and run ledger, recomputes independent metrics,
and publishes reports from the two already-saved pilot fits.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts import optimization_v3_batch1_c04 as c04
from src.models.optimization_v3_batch1 import (
    BatchIntegrityError,
    file_sha256,
    json_sha256,
    load_queue,
    read_bundle,
)


ROOT = Path(__file__).resolve().parents[1]


def _events() -> list[dict[str, Any]]:
    if not c04.LEDGER_PATH.is_file():
        raise BatchIntegrityError("C04 run ledger is missing")
    return [json.loads(line) for line in c04.LEDGER_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def recover() -> dict[str, Any]:
    # The frozen pilot runner's reporting helper resolves read_bundle as a
    # module global. Supply the shared read-only bundle verifier without
    # altering the already-frozen training source.
    c04.read_bundle = read_bundle
    queue = load_queue(c04.QUEUE_PATH)
    card = next(item for item in queue["cards"] if item.get("id") == c04.CARD_ID)
    context = c04._prepare(card, queue)
    if context["public"]["status"] != "PREFLIGHT_PASS":
        raise BatchIntegrityError("C04 recovery preflight is blocked")

    report_path = c04.ENGINE_REPORT_PATH
    report = c04._read_json(report_path)
    expected_signature = json_sha256(context["identity"])
    if (report.get("card_id") != c04.CARD_ID
            or report.get("status") != "PILOT_SAVED_AWAITING_INDEPENDENT_QA"
            or report.get("freeze_signature") != expected_signature
            or report.get("fit_predictor_count") != 2
            or report.get("cutoffs") != [55, 67]
            or report.get("fit_count_matches_card") is not True):
        raise BatchIntegrityError("Saved C04 engine report is not the exact frozen two-fit pilot")

    events = [event for event in _events() if event.get("card_id") == c04.CARD_ID]
    reservations = [event for event in events if event.get("event") == "BUDGET_RESERVED"]
    if (len(reservations) != 1 or reservations[0].get("budget_key") != "C04:PILOT"
            or int(reservations[0].get("predictor_fits", -1)) != 2):
        raise BatchIntegrityError("C04 reservation is absent, duplicated, or not exactly two fits")
    starts = [event for event in events if event.get("event") == "FIT_STARTED"]
    saves = [event for event in events if event.get("event") == "FIT_SAVED"]
    failures = [event for event in events if event.get("event") == "FIT_FAILED"]
    if ([(int(event.get("cutoff", -1))) for event in starts] != [55, 67]
            or [(int(event.get("cutoff", -1))) for event in saves] != [55, 67]
            or failures):
        raise BatchIntegrityError("C04 ledger does not show exactly two successful saved fits and no failures")

    verification, gate = c04._gate_and_verify(context, report)
    fit_accounting = c04._fit_accounting(context)
    if fit_accounting["successful_predictor_fits"] != 2:
        raise BatchIntegrityError("C04 saved fit accounting does not reconcile to two successful fits")
    context["public"]["postfit_recovery"] = {
        "path": "scripts/recover_optimization_v3_batch1_c04.py",
        "sha256": file_sha256(Path(__file__)),
        "reason": "Frozen runner omitted read_bundle from the publication module namespace; both model fits and independent route verification were saved.",
        "refit_performed": False,
    }
    c04._publish(context, report, verification, gate, fit_accounting)
    return {
        "card_id": c04.CARD_ID,
        "status": "RECOVERED_AND_VERIFIED_WITHOUT_REFIT",
        "decision": gate["status"],
        "freeze_signature": expected_signature,
        "fit_predictor_count": 2,
        "fit_attempt_count": len(starts),
        "fit_failure_count": len(failures),
        "fit_accounting": fit_accounting,
        "views": verification["views"],
        "gate_checks": gate["checks"],
        "test_or_sample_access": False,
        "kaggle_access": False,
        "refit_performed": False,
    }


if __name__ == "__main__":
    print(json.dumps(recover(), indent=2, allow_nan=False))
