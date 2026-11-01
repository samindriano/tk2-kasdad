"""Publish a report-only R04 amendment for the saved bundle SHA-256 receipts."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import optimization_v3_batch1_r04 as r04
from src.models.optimization_v3_batch1 import BatchIntegrityError, json_sha256


AMENDMENT_ID = "R04_REPORT_AMENDMENT_01"
OUTPUT_DIR = r04.REPORT_DIR / "report_amendment_01"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validated_bundle_hashes(accounting: dict[str, Any]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for cutoff in (55, 67):
        stem = f"R04_fit{cutoff}"
        bundle_path = r04.RUN_DIR / f"{stem}.npz"
        sidecar_path = r04.RUN_DIR / f"{stem}.json"
        sidecar = _read_json(sidecar_path)
        expected = sidecar.get("npz_sha256")
        actual = r04.file_sha256(bundle_path)
        if not isinstance(expected, str) or expected != actual:
            raise BatchIntegrityError(f"Saved R04 bundle SHA-256 does not match its sidecar at cutoff {cutoff}")
        accounting["folds"][str(cutoff)]["bundle_npz_sha256"] = actual
        hashes[str(cutoff)] = actual
    return hashes


def _write_or_match(path: Path, value: dict[str, Any]) -> None:
    if path.is_file():
        if _read_json(path) != value:
            raise BatchIntegrityError(f"Existing report amendment differs; refusing overwrite: {path}")
        return
    r04.write_json_exclusive(path, value)


def run_amendment() -> dict[str, Any]:
    queue = r04.load_queue(r04.QUEUE_PATH)
    card = next((item for item in queue.get("cards", []) if item.get("id") == r04.CARD_ID), None)
    if card is None:
        raise BatchIntegrityError("R04 is missing from the canonical queue")

    # Rebind the saved outputs to the same current data, controls, and frozen source
    # identity. _prepare performs no model fitting and fails closed on any drift.
    context = r04._prepare(card, queue)
    report = _read_json(r04.ENGINE_REPORT_PATH)
    signature = json_sha256(context["identity"])
    if (report.get("card_id") != r04.CARD_ID
            or report.get("freeze_signature") != signature
            or report.get("status") != "PILOT_SAVED_AWAITING_INDEPENDENT_QA"):
        raise BatchIntegrityError("Saved R04 pilot report does not match the active frozen amendment")

    prior_results = _read_json(r04.RESULTS_PATH)
    prior_verification = _read_json(r04.VERIFICATION_PATH)
    prior_subgroups = _read_json(r04.SUBGROUPS_PATH)
    prior_gate = _read_json(r04.GATE_PATH)
    if (prior_results.get("freeze_signature") != signature
            or prior_verification.get("freeze_signature") != signature
            or prior_verification.get("status") != "PASS"
            or prior_gate.get("card_id") != r04.CARD_ID):
        raise BatchIntegrityError("Existing R04 verifier outputs do not match the active freeze")

    verification, subgroups, gate = r04._verify_and_gate(context, report, queue)
    if verification.get("status") != "PASS" or gate.get("decision") != prior_results.get("status"):
        raise BatchIntegrityError("Independent R04 re-verification changed the saved verdict")
    if report.get("views") != prior_results.get("views"):
        raise BatchIntegrityError("Rebuilt R04 metric views differ from the original saved result")

    accounting = r04._fit_accounting(context)
    bundle_hashes = _validated_bundle_hashes(accounting)
    amended_results = dict(prior_results)
    amended_results.update({
        "fit_accounting": accounting,
        "independent_verification": verification,
        "subgroups": subgroups,
        "gate": gate,
        "report_amendment": {
            "id": AMENDMENT_ID,
            "reason": "Correct the bundle hash field using the sidecar npz_sha256 key and independently verify each NPZ byte hash.",
            "source_freeze_signature": signature,
            "candidate_model_refit": False,
            "prediction_bundles_changed": False,
            "metrics_and_verdict_unchanged": True,
            "bundle_npz_sha256": bundle_hashes,
        },
    })
    amended_summary = dict(_read_json(r04.SUMMARY_PATH))
    amended_summary.update({
        "report_amendment_id": AMENDMENT_ID,
        "bundle_hashes_verified": True,
        "candidate_model_refit": False,
        "status": str(gate["decision"]),
    })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    amendment = {
        "id": AMENDMENT_ID,
        "card_id": r04.CARD_ID,
        "source_freeze_signature": signature,
        "prior_results_sha256": r04.file_sha256(r04.RESULTS_PATH),
        "prior_verification_sha256": r04.file_sha256(r04.VERIFICATION_PATH),
        "saved_bundle_npz_sha256": bundle_hashes,
        "independent_verification_status": verification["status"],
        "decision": gate["decision"],
        "candidate_model_refit": False,
        "test_data_used": False,
        "kaggle_access": False,
    }
    _write_or_match(OUTPUT_DIR / "results.json", amended_results)
    _write_or_match(OUTPUT_DIR / "verification.json", verification)
    _write_or_match(OUTPUT_DIR / "subgroups.json", subgroups)
    _write_or_match(OUTPUT_DIR / "gate.json", gate)
    _write_or_match(OUTPUT_DIR / "summary.json", amended_summary)
    _write_or_match(OUTPUT_DIR / "amendment.json", amendment)
    return amendment


def main() -> int:
    result = run_amendment()
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
