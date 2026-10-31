"""Versioned, narrowly amended entry point for the frozen six-window verifier.

The frozen verifier's first run raised KeyError('end') because its control
loader did not carry through the cutoff argument needed to locate a new-fit
sidecar. This wrapper adds only that asserted argument to the verifier's local
control record. It does not alter model inputs, probabilities, metrics, route,
or evaluation configuration.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import optimization_v3_sparse_classification_fullsix as run
from scripts import optimization_v3_verify_sparse_classification_fullsix as verifier
from scripts.optimization_v3_sparse_classification import check as check_pilot

AMENDMENT = run.CONFIG / "verification_amendment.json"
VERIFY_RECEIPT = run.OUT / "verification_wrapper_run.json"


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def current_head() -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def write_new(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def build_amendment(frozen: dict) -> dict:
    original_code = frozen["identity"]["code_sha256"]
    original_inputs = frozen["identity"]["inputs_sha256"]
    tracked_outputs = [
        run.CONFIG / "protocol.json",
        run.CONFIG / "freeze.json",
        run.OUT / "fullsix.json",
        run.OUT / "FULLSIX.md",
    ]
    for end in run.NEW_ENDS:
        tracked_outputs.extend([
            run.CACHE / f"fit{end}.npz",
            run.CACHE / f"fit{end}.json",
        ])
    absent = [str(path) for path in tracked_outputs if not path.is_file()]
    if absent:
        raise FileNotFoundError(f"Amendment artifacts are missing: {absent}")
    frozen_mismatches = {}
    for rel, expected in original_code.items():
        actual = sha(ROOT / rel)
        if actual != expected:
            frozen_mismatches[rel] = {"expected": expected, "actual": actual}
    for rel, expected in original_inputs.items():
        actual = sha(ROOT / rel)
        if actual != expected:
            frozen_mismatches[rel] = {"expected": expected, "actual": actual}
    if frozen_mismatches:
        raise AssertionError(f"Prefit freeze hashes changed: {frozen_mismatches}")
    pilot = check_pilot()
    if pilot["signature"] != "59104d95490adc147c07cfa18559074dae71599530fc1d0a7a5d948ae26a2d33":
        raise AssertionError("Root sparse pilot signature mismatch")
    extra_hashes = {
        path.resolve().relative_to(ROOT).as_posix(): sha(path) for path in tracked_outputs
    }
    body = {
        "status": "FROZEN_BEFORE_METRIC_RECHECK",
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "source_head_contextual": current_head(),
        "continuation_freeze_signature": frozen["signature"],
        "original_prefit_verifier_sha256": original_code[
            "scripts/optimization_v3_verify_sparse_classification_fullsix.py"
        ],
        "wrapper_sha256": sha(Path(__file__)),
        "old_frozen_code_hashes_all_match": True,
        "old_frozen_input_hashes_all_match": True,
        "old_frozen_code_hashes": original_code,
        "old_frozen_input_hashes": original_inputs,
        "root_pilot_signature": pilot["signature"],
        "root_pilot_verification_status": "PASS",
        "full_report_and_prediction_cache_sha256": extra_hashes,
        "original_verifier_error": {
            "type": "KeyError",
            "message": "'end'",
            "location": "scripts/optimization_v3_verify_sparse_classification_fullsix.py::_newfit",
        },
        "amendment_scope": {
            "only_patch": "The wrapper adds the exact asserted end/cutoff argument to the local dictionary returned by the original verifier _controls(frame, end).",
            "assertions": ["end is an allowed frozen cutoff", "original record has no end key"],
            "prediction_edits": False,
            "metric_edits": False,
            "model_or_configuration_edits": False,
            "new_fits": 0,
            "test_read": False,
            "kaggle_access": False,
        },
    }
    return {**body, "amendment_signature": canonical_sha(body)}


def run_once() -> dict:
    if AMENDMENT.exists() or VERIFY_RECEIPT.exists() or (run.OUT / "verification.json").exists():
        raise FileExistsError("The versioned amendment verification is immutable and one-shot")
    frozen = run.check_freeze()
    amendment = build_amendment(frozen)
    write_new(AMENDMENT, amendment)

    original_controls = verifier._controls

    def controls_with_asserted_end(frame, end):
        if end not in run.ENDS:
            raise AssertionError(f"Cutoff {end} is outside the frozen continuation")
        record = original_controls(frame, end)
        if "end" in record:
            raise AssertionError("Original control record unexpectedly contains an end field")
        record["end"] = int(end)
        return record

    verifier._controls = controls_with_asserted_end
    verified = verifier.verify()
    if verified.get("status") != "PASS" or verified.get("newfits") != 0:
        raise AssertionError("Independent verifier did not complete the no-fit audit")
    receipt = {
        "status": "PASS",
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "amendment_signature": amendment["amendment_signature"],
        "verification_artifact_sha256": sha(run.OUT / "verification.json"),
        "fullsix_results_sha256": sha(run.OUT / "fullsix.json"),
        "newfits": 0,
        "test_read": False,
        "kaggle_access": False,
        "primary_gain": verified["parent_external_gates"]["primary_gain"],
        "h12_gain": verified["parent_external_gates"]["h12_gain"],
        "nonnegative_windows": verified["parent_external_gates"]["nonnegative_windows"],
        "entry_point": "scripts/optimization_v3_verify_sparse_classification_fullsix_wrapper.py",
        "note": "The original prefit verifier file and its frozen hash remain unchanged; this receipt records the one-field adapter and independently recomputed metrics.",
    }
    write_new(VERIFY_RECEIPT, receipt)
    print(json.dumps(receipt, indent=2, allow_nan=False))
    return receipt


if __name__ == "__main__":
    run_once()
