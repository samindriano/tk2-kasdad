"""Narrow runner for the frozen pilot's completed-fit guard defect.

All model fitting and artifact writing remain in the original frozen driver.
This adapter changes only the interpretation of the driver's stale
"FIT_STARTED means active" check: completed/failed fits are terminal, while
starts without a terminal event remain pending and block execution.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

AUDIT_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = AUDIT_ROOT / "reports/simplification_v1/CLS_POS_EF_CONTEXT_V1"
RUN_ROOT = AUDIT_ROOT / "experiments/CLS_POS_EF_CONTEXT_V1"
AMENDMENT_PATH = REPORT_ROOT / "EXECUTION_GUARD_AMENDMENT.json"
LINEAGE_PATH = RUN_ROOT / "execution_lineage.jsonl"
ADAPTER_REL = "scripts/run_cls_pos_ef_context_v1_amended.py"
VERIFIER_REL = "scripts/verify_cls_pos_ef_context_v1_execution_amendment.py"
FROZEN_DRIVER_REL = "scripts/cls_pos_ef_context_v1.py"
PENDING_GUARD_MESSAGE = "Another frozen fit is pending; one heavy fit at a time"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_normalized_text(path: Path) -> str:
    return sha256_bytes(path.read_bytes().replace(b"\r\n", b"\n"))


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(AUDIT_ROOT), *args]).decode("utf-8").strip()


def load_frozen_driver():
    if str(AUDIT_ROOT) not in sys.path:
        sys.path.insert(0, str(AUDIT_ROOT))
    path = AUDIT_ROOT / FROZEN_DRIVER_REL
    spec = importlib.util.spec_from_file_location("cls_pos_ef_context_v1_frozen_driver", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load frozen pilot driver")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def append_lineage(event: Mapping[str, Any]) -> None:
    LINEAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LINEAGE_PATH.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(dict(event), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_lineage() -> list[dict[str, Any]]:
    if not LINEAGE_PATH.exists():
        return []
    rows = []
    for line in LINEAGE_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def verify_amendment(pilot, repository: Path, fit_id: str) -> tuple[dict[str, Any], dict[str, Any], str]:
    freeze = pilot.verify_freeze(repository)
    amendment = pilot.read_json(AMENDMENT_PATH)
    pilot.require(amendment.get("status") == "OPERATIONAL_GUARD_ONLY",
                  "Execution amendment is not approved for use")
    pilot.require(amendment.get("candidate_id") == freeze["candidate_id"]
                  and amendment.get("parent_freeze_signature") == freeze["freeze_signature"]
                  and amendment.get("parent_source_commit") == freeze["source_commit"],
                  "Execution amendment does not bind to the frozen pilot")
    pilot.require(amendment.get("frozen_driver_sha256") == freeze["source_hashes"][FROZEN_DRIVER_REL],
                  "Frozen driver hash differs from the original freeze")
    pilot.require(fit_id in amendment.get("authorized_fit_ids", []),
                  f"Fit is not authorized by the guard amendment: {fit_id}")
    adapter_hash = sha256_file(AUDIT_ROOT / ADAPTER_REL)
    pilot.require(adapter_hash == amendment.get("adapter_sha256"), "Amendment adapter hash mismatch")
    verifier_hash = sha256_file(AUDIT_ROOT / VERIFIER_REL)
    pilot.require(verifier_hash == amendment.get("verifier_sha256"), "Independent verifier hash mismatch")
    head = git("rev-parse", "HEAD")
    for rel, expected in ((ADAPTER_REL, adapter_hash), (VERIFIER_REL, verifier_hash)):
        committed = subprocess.check_output(["git", "-C", str(AUDIT_ROOT), "show", f"HEAD:{rel}"])
        pilot.require(sha256_bytes(committed) == expected, f"Amendment code must be committed before execution: {rel}")
    manifest_rel = AMENDMENT_PATH.relative_to(AUDIT_ROOT).as_posix()
    committed_manifest = subprocess.check_output(["git", "-C", str(AUDIT_ROOT), "show", f"HEAD:{manifest_rel}"])
    pilot.require(sha256_bytes(committed_manifest) == sha256_normalized_text(AMENDMENT_PATH),
                  "Execution amendment manifest must be committed before execution")
    dirty = git("status", "--porcelain", "--", ADAPTER_REL, VERIFIER_REL, manifest_rel)
    pilot.require(not dirty, "Amendment code/manifest has uncommitted changes")
    return freeze, amendment, head


def completed_fit_guard(pilot, freeze: Mapping[str, Any]):
    """Temporarily replace only the defective active-fit predicate."""
    original_require = pilot.require

    def amended_require(condition: bool, message: str) -> None:
        if message == PENDING_GUARD_MESSAGE:
            # Re-run the authoritative ledger verifier with no pending ID allowed.
            # This validates reservation, uniqueness, cap, and start/terminal pairing.
            ev = pilot.check_live_budget(freeze)
            starts = {e["fit_id"] for e in ev if e.get("event") == "FIT_STARTED"}
            terminals = {e["fit_id"] for e in ev if e.get("event") in {"FIT_SAVED", "FIT_FAILED"}}
            pending = starts - terminals
            original_require(not pending, f"A fit is still pending: {sorted(pending)}")
            return
        original_require(condition, message)

    pilot.require = amended_require
    return original_require


def run_one(repository: Path, fit_id: str) -> dict[str, Any]:
    pilot = load_frozen_driver()
    repository = repository.resolve()
    freeze, amendment, head = verify_amendment(pilot, repository, fit_id)

    ev = pilot.check_live_budget(freeze, allow_current=fit_id)
    starts = {e["fit_id"] for e in ev if e.get("event") == "FIT_STARTED"}
    next_fit = next((candidate for candidate in freeze["fit_ids"] if candidate not in starts), None)
    pilot.require(fit_id == next_fit, f"Frozen order requires {next_fit}; no retry or skip is allowed")
    pilot.require(not any(e.get("fit_id") == fit_id for e in ev), f"Fit ID already appears in ledger: {fit_id}")

    prior_ids = [candidate for candidate in freeze["fit_ids"] if candidate in starts]
    lineage = read_lineage()
    known = {e.get("fit_id") for e in lineage if e.get("event") in {
        "ORIGINAL_FROZEN_FIT_RECONCILED", "AMENDMENT_EXECUTION_RETURNED"}}
    for previous_id in prior_ids:
        if previous_id not in known:
            receipt = pilot.verify_fit_artifact(freeze, previous_id)
            append_lineage({"event": "ORIGINAL_FROZEN_FIT_RECONCILED", "fit_id": previous_id,
                "freeze_signature": freeze["freeze_signature"], "source_commit": freeze["source_commit"],
                "execution_code": FROZEN_DRIVER_REL,
                "execution_code_sha256": freeze["source_hashes"][FROZEN_DRIVER_REL],
                "receipt_sha256": receipt["receipt_sha256"], "artifact_sha256": receipt["receipt"]["artifact_sha256"]})

    adapter_hash = sha256_file(AUDIT_ROOT / ADAPTER_REL)
    append_lineage({"event": "AMENDMENT_EXECUTION_STARTED", "fit_id": fit_id,
        "amendment_id": amendment["amendment_id"], "freeze_signature": freeze["freeze_signature"],
        "execution_code": ADAPTER_REL, "execution_code_sha256": adapter_hash,
        "execution_commit": head, "started_at_unix": time.time()})
    original_require = completed_fit_guard(pilot, freeze)
    try:
        result = pilot.fit_one(repository, fit_id)
    except BaseException as exc:
        ledger = pilot.events(RUN_ROOT / "fit_ledger.jsonl")
        matching_starts = [e for e in ledger if e.get("event") == "FIT_STARTED" and e.get("fit_id") == fit_id]
        matching_terminal = [e for e in ledger if e.get("event") in {"FIT_SAVED", "FIT_FAILED"} and e.get("fit_id") == fit_id]
        append_lineage({"event": "AMENDMENT_EXECUTION_RAISED", "fit_id": fit_id,
            "amendment_id": amendment["amendment_id"], "freeze_signature": freeze["freeze_signature"],
            "fit_started": bool(matching_starts), "fit_terminal_events": [e["event"] for e in matching_terminal],
            "error_type": type(exc).__name__, "error": str(exc), "retry_allowed": False})
        raise
    finally:
        pilot.require = original_require

    receipt_path = pilot.fit_paths(fit_id)[1]
    receipt_hash = pilot.sha256_file(receipt_path)
    append_lineage({"event": "AMENDMENT_EXECUTION_RETURNED", "fit_id": fit_id,
        "amendment_id": amendment["amendment_id"], "freeze_signature": freeze["freeze_signature"],
        "execution_code": ADAPTER_REL, "execution_code_sha256": adapter_hash,
        "execution_commit": head, "status": result["status"], "receipt_sha256": receipt_hash})
    return {**result, "execution_amendment": amendment["amendment_id"], "execution_commit": head}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", required=True)
    parser.add_argument("--fit-id", required=True)
    args = parser.parse_args()
    result = run_one(Path(args.official_root), args.fit_id)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
