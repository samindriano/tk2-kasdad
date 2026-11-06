from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/simplification_v1/CLS_POST_BAG2_NEXT_V1"
RUN = ROOT / "experiments/runs/simplification_v1/CLS_POST_BAG2_NEXT_V1"
FREEZE = REPORT / "FREEZE.json"
LEDGER = REPORT / "BUDGET_LEDGER.jsonl"
OLD_VERIFIER = ROOT / "scripts/verify_cls_a02_position_kernel_v1_prelabel.py"
NEW_VERIFIER = ROOT / "scripts/verify_cls_a02_position_kernel_v1_prelabel_amended_01.py"
OUTPUT = REPORT / "PRELABEL_AMENDMENT_01.json"
UNITS = ("E67_REAL", "E67_SHUFFLED", "E75_REAL", "E75_SHUFFLED")
EVENTS = ("PREDICTOR_RESERVED", "PREDICTOR_STARTED", "PREDICTOR_SAVED")
AMENDMENT_ID = "CLS_A02_POSITION_KERNEL_V1_PRELABEL_AMENDMENT_01"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_json(value) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=True)
    return result.stdout.strip()


def committed_clean(path: Path) -> bool:
    rel = path.relative_to(ROOT).as_posix()
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=ROOT,
                             text=True, capture_output=True).returncode == 0
    unchanged = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=ROOT,
                               text=True, capture_output=True).returncode == 0
    return tracked and unchanged


def main() -> int:
    if OUTPUT.exists():
        raise RuntimeError(f"Refusing to overwrite immutable amendment manifest: {OUTPUT}")
    if git("status", "--porcelain"):
        raise RuntimeError("The implementation commit must be clean before creating the manifest")
    for path in (FREEZE, LEDGER, OLD_VERIFIER, NEW_VERIFIER):
        if not path.is_file() or not committed_clean(path):
            raise RuntimeError(f"Required committed input is missing or modified: {path}")

    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    events = [json.loads(line) for line in LEDGER.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected_ids = list(UNITS)
    if (freeze.get("pilot_id") != "CLS_A02_POSITION_KERNEL_V1"
            or freeze.get("original_prelabel_boundary_compliant") is not False
            or freeze.get("scientific_qualification") is not False):
        raise RuntimeError("Frozen pilot identity or historical verdict changed")
    cap = [event for event in events if event.get("event") == "CAP_SET"]
    if len(cap) != 1 or cap[0].get("cap") != 4 or cap[0].get("units") != expected_ids:
        raise RuntimeError("Expected immutable cap4 ledger not found")
    counts = {}
    event_ids = {}
    event_digests = {}
    for event_name in EVENTS:
        selected = [event for event in events if event.get("event") == event_name]
        ids = [event.get("fit_id") for event in selected]
        if len(ids) != 4 or sorted(ids) != sorted(expected_ids):
            raise RuntimeError(f"Ledger does not contain exactly one {event_name} per unit: {ids}")
        counts[event_name] = len(ids)
        event_ids[event_name] = ids
        event_digests[event_name] = sha_json(selected)
    if any(event.get("event") in {"PREDICTOR_INTERRUPTED", "PREDICTOR_RETRY", "PREDICTOR_REFUNDED"}
           for event in events):
        raise RuntimeError("Unexpected retry/interruption/refund event in the frozen pilot ledger")

    expected_old_path = "scripts/verify_cls_a02_position_kernel_v1_prelabel.py"
    old_sha_from_freeze = next((value for path, value in freeze.get("source_hashes", {}).items()
                                if path.replace("\\", "/") == expected_old_path), None)
    if old_sha_from_freeze != sha(OLD_VERIFIER):
        raise RuntimeError("Old verifier hash differs from immutable freeze")
    bundles = {}
    for unit_id in UNITS:
        bundle = RUN / unit_id / "predictions.npz"
        receipt = RUN / unit_id / "receipt.json"
        if not bundle.is_file() or not receipt.is_file():
            raise RuntimeError(f"Saved target-free output is missing: {unit_id}")
        receipt_data = json.loads(receipt.read_text(encoding="utf-8"))
        if receipt_data.get("fit_id") != unit_id or receipt_data.get("artifact_sha256") != sha(bundle):
            raise RuntimeError(f"Unit receipt does not bind its frozen bundle: {unit_id}")
        bundles[unit_id] = {"bundle_sha256": sha(bundle), "receipt_sha256": sha(receipt)}

    implementation_commit = git("rev-parse", "HEAD")
    manifest = {
        "amendment_id": AMENDMENT_ID,
        "status": "APPROVED_VERIFIER_ONLY",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization": "Sol KASDAD chat explicitly approved a verifier-only amendment after the source mismatch was discovered and before any target labels were opened.",
        "old_freeze_sha256": sha(FREEZE),
        "old_freeze_source_commit": freeze["source_commit"],
        "official_source_commit": freeze["official_source_commit"],
        "implementation_commit": implementation_commit,
        "old_verifier_path": OLD_VERIFIER.relative_to(ROOT).as_posix(),
        "old_verifier_sha256": old_sha_from_freeze,
        "old_verifier_invoked": False,
        "new_verifier_path": NEW_VERIFIER.relative_to(ROOT).as_posix(),
        "new_verifier_sha256": sha(NEW_VERIFIER),
        "identified_incompatibilities": [
            "The old verifier binds POS p_pos to POSITIONAL_TEXT_PILOT_V1 prob_position instead of the champion POS_BAG_OFFSET_V1 prob_offset_real / verified original-only p_pos_original.",
            "The old verifier canonicalizes A02 bank IDs with str: tags instead of the producer's NumPy str_ tags.",
        ],
        "recipe_or_freeze_changed": False,
        "candidate_runner_or_outputs_changed": False,
        "query_labels_accessed": False,
        "test_or_sample_accessed": False,
        "kaggle_accessed": False,
        "budget_ledger_sha256": sha(LEDGER),
        "budget_event_counts": counts,
        "budget_event_ids": event_ids,
        "budget_event_sha256": event_digests,
        "four_units": bundles,
        "scientific_qualification": False,
        "original_prelabel_boundary_compliant": False,
    }
    with OUTPUT.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
    print(json.dumps({"manifest": OUTPUT.relative_to(ROOT).as_posix(), "implementation_commit": implementation_commit,
                      "new_verifier_sha256": manifest["new_verifier_sha256"],
                      "budget_ledger_sha256": manifest["budget_ledger_sha256"],
                      "unit_count": len(bundles)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
