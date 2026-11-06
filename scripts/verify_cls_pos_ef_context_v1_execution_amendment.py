"""Independent post-fit verification for the narrow execution-guard amendment."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

AUDIT_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = AUDIT_ROOT / "reports/simplification_v1/CLS_POS_EF_CONTEXT_V1"
RUN_ROOT = AUDIT_ROOT / "experiments/CLS_POS_EF_CONTEXT_V1"
FREEZE_REL = "reports/simplification_v1/CLS_POS_EF_CONTEXT_V1/FREEZE.json"
AMENDMENT_REL = "reports/simplification_v1/CLS_POS_EF_CONTEXT_V1/EXECUTION_GUARD_AMENDMENT.json"
ADAPTER_REL = "scripts/run_cls_pos_ef_context_v1_amended.py"
VERIFIER_REL = "scripts/verify_cls_pos_ef_context_v1_execution_amendment.py"
FROZEN_DRIVER_REL = "scripts/cls_pos_ef_context_v1.py"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_normalized_text(path: Path) -> str:
    return sha256_bytes(path.read_bytes().replace(b"\r\n", b"\n"))


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(AUDIT_ROOT), *args]).decode("utf-8").strip()


def load_driver():
    if str(AUDIT_ROOT) not in sys.path:
        sys.path.insert(0, str(AUDIT_ROOT))
    spec = importlib.util.spec_from_file_location("cls_pos_ef_context_v1_independent_driver", AUDIT_ROOT / FROZEN_DRIVER_REL)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load frozen pilot verifier")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verify(official_root: Path) -> dict[str, Any]:
    pilot = load_driver()
    official_root = official_root.resolve()
    freeze = pilot.verify_freeze(official_root)
    amendment = pilot.read_json(REPORT_ROOT / "EXECUTION_GUARD_AMENDMENT.json")
    pilot.require(amendment.get("status") == "OPERATIONAL_GUARD_ONLY"
                  and amendment.get("parent_freeze_signature") == freeze["freeze_signature"]
                  and amendment.get("parent_source_commit") == freeze["source_commit"],
                  "Amendment does not bind to the unchanged source freeze")
    pilot.require(sha256_file(AUDIT_ROOT / FROZEN_DRIVER_REL) == freeze["source_hashes"][FROZEN_DRIVER_REL],
                  "Original frozen driver was modified")
    for rel, key in ((ADAPTER_REL, "adapter_sha256"), (VERIFIER_REL, "verifier_sha256")):
        actual = sha256_file(AUDIT_ROOT / rel)
        pilot.require(actual == amendment.get(key), f"Amendment code hash mismatch: {rel}")
        committed = subprocess.check_output(["git", "-C", str(AUDIT_ROOT), "show", f"HEAD:{rel}"])
        pilot.require(sha256_bytes(committed) == actual, f"Amendment code is not committed: {rel}")
    manifest_path = REPORT_ROOT / "EXECUTION_GUARD_AMENDMENT.json"
    manifest_blob = subprocess.check_output(["git", "-C", str(AUDIT_ROOT), "show", f"HEAD:{AMENDMENT_REL}"])
    pilot.require(sha256_bytes(manifest_blob) == sha256_normalized_text(manifest_path),
                  "Amendment manifest is not committed")

    ledger_path = RUN_ROOT / "fit_ledger.jsonl"
    lineage_path = RUN_ROOT / "execution_lineage.jsonl"
    events = pilot.events(ledger_path)
    lineage = []
    if lineage_path.is_file():
        lineage = [json.loads(line) for line in lineage_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    reservation_events = [e for e in events if e.get("event") == "CARD_FITS_RESERVED"]
    expected_reservation = {"event": "CARD_FITS_RESERVED", "candidate_id": freeze["candidate_id"],
        "freeze_signature": freeze["freeze_signature"], "fit_ids": freeze["fit_ids"], "fit_cap": 4}
    pilot.require(reservation_events == [expected_reservation], "Reservation/cap was changed or duplicated")
    starts = [e for e in events if e.get("event") == "FIT_STARTED"]
    terminals = [e for e in events if e.get("event") in {"FIT_SAVED", "FIT_FAILED"}]
    pilot.require([e.get("fit_id") for e in starts] == freeze["fit_ids"],
                  "Official fit starts do not exactly match the frozen four-ID order")
    pilot.require([e.get("fit_id") for e in terminals] == freeze["fit_ids"]
                  and all(e.get("event") == "FIT_SAVED" for e in terminals),
                  "Every reserved fit must have one saved terminal receipt")
    pilot.require(all(e.get("source_commit") == freeze["source_commit"]
                      and e.get("freeze_signature") == freeze["freeze_signature"] for e in starts),
                  "Fit ledger source/freeze lineage differs")
    pilot.require(not any(e.get("event") == "AMENDMENT_EXECUTION_RAISED" for e in lineage),
                  "An amendment execution raised; inspect its durable receipt before proceeding")

    original_id = freeze["fit_ids"][0]
    amended_ids = amendment["authorized_fit_ids"]
    pilot.require(amended_ids == freeze["fit_ids"][1:], "Amendment scope is not exactly the three unstarted fits")
    original_lineage = [e for e in lineage if e.get("event") == "ORIGINAL_FROZEN_FIT_RECONCILED"
                        and e.get("fit_id") == original_id]
    pilot.require(len(original_lineage) == 1
                  and original_lineage[0].get("execution_code_sha256") == freeze["source_hashes"][FROZEN_DRIVER_REL]
                  and original_lineage[0].get("source_commit") == freeze["source_commit"],
                  "First fit was not retained under its original frozen execution lineage")

    receipt_hashes: dict[str, str] = {}
    fit_receipts = {}
    for fit_id in freeze["fit_ids"]:
        fit = pilot.verify_fit_artifact(freeze, fit_id)
        receipt = fit["receipt"]
        pilot.require(receipt.get("query_labels_read") is False and receipt.get("test_or_sample_access") is False,
                      f"Forbidden data boundary recorded in {fit_id}")
        receipt_hashes[fit_id] = fit["receipt_sha256"]
        fit_receipts[fit_id] = {"receipt_sha256": fit["receipt_sha256"],
            "artifact_sha256": receipt["artifact_sha256"], "gamma_sha256": receipt["gamma_sha256"],
            "optimizer_success": receipt["optimizer"].get("success")}

    execution_commit: str | None = None
    for fit_id in amended_ids:
        starts_for = [e for e in lineage if e.get("event") == "AMENDMENT_EXECUTION_STARTED" and e.get("fit_id") == fit_id]
        returned_for = [e for e in lineage if e.get("event") == "AMENDMENT_EXECUTION_RETURNED" and e.get("fit_id") == fit_id]
        pilot.require(len(starts_for) == len(returned_for) == 1,
                      f"Execution lineage is missing or duplicated for {fit_id}")
        start, returned = starts_for[0], returned_for[0]
        for row in (start, returned):
            pilot.require(row.get("amendment_id") == amendment["amendment_id"]
                          and row.get("freeze_signature") == freeze["freeze_signature"]
                          and row.get("execution_code") == ADAPTER_REL
                          and row.get("execution_code_sha256") == amendment["adapter_sha256"],
                          f"Actual adapter lineage mismatch for {fit_id}")
        pilot.require(returned.get("status") == "SAVED"
                      and returned.get("receipt_sha256") == receipt_hashes[fit_id],
                      f"Adapter return does not match verified receipt for {fit_id}")
        if execution_commit is None:
            execution_commit = start.get("execution_commit")
        pilot.require(start.get("execution_commit") == execution_commit == returned.get("execution_commit"),
                      "Amended fits used inconsistent execution commits")
        pilot.require(git("merge-base", "--is-ancestor", execution_commit, git("rev-parse", "HEAD")) == "",
                      "Amendment execution commit is not an ancestor of current HEAD")
        adapter_blob = subprocess.check_output(["git", "-C", str(AUDIT_ROOT), "show", f"{execution_commit}:{ADAPTER_REL}"])
        pilot.require(sha256_bytes(adapter_blob) == amendment["adapter_sha256"],
                      "Amendment execution commit does not contain the authorized adapter bytes")

    return {"schema_version": 1, "status": "PASS_EXECUTION_AMENDMENT_LINEAGE",
        "candidate_id": freeze["candidate_id"], "amendment_id": amendment["amendment_id"],
        "freeze_signature": freeze["freeze_signature"], "source_commit": freeze["source_commit"],
        "original_fit_id": original_id, "amended_fit_ids": amended_ids,
        "execution_commit": execution_commit, "fit_start_count": len(starts),
        "reservation_cap": 4, "fit_receipts": fit_receipts,
        "query_labels_read": False, "test_or_sample_access": False,
        "fit_ledger_sha256": sha256_file(ledger_path),
        "execution_lineage_sha256": sha256_file(lineage_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", required=True)
    parser.add_argument("--write", action="store_true", help="Persist the verified report once")
    args = parser.parse_args()
    result = verify(Path(args.official_root))
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    if args.write:
        pilot = load_driver()
        pilot.write_once(REPORT_ROOT / "EXECUTION_AMENDMENT_VERIFICATION.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
