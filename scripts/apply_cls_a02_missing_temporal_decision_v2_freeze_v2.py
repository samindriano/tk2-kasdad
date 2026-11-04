"""Verify a versioned E79 freeze successor, then build the one frozen CSV.

The original V2 application freeze remains byte-for-byte unchanged. This
adapter adds only the two Train/OOF hashes omitted from that five-input freeze
so the pinned builder's seven-input guard can pass before its first E79 array
read. It never selects a threshold or fits a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import optimization_v3_cls_a02_missing_temporal_decision_v2 as decision  # noqa: E402

REPORT_DIR = ROOT / "reports/optimization_v3/cls_a02_missing_temporal_decision_v2"
ORIGINAL_FREEZE = REPORT_DIR / "E79_APPLICATION_FREEZE.json"
SUCCESSOR_FREEZE = REPORT_DIR / "E79_APPLICATION_FREEZE_V2.json"
GUARD_QA = REPORT_DIR / "E79_APPLICATION_FREEZE_V2_GUARD_QA.json"
ORIGINAL_FREEZE_SHA256 = "b833b5d078a233ae17dcb7d14fecd451cdeac3cd8f63d0b94e9db7a8838cd493"
APPLICATION_ADDED_INPUTS = {
    "data/raw/classification/Train.csv",
    "experiments/runs/optimization_v3_posoffset_a02_endgame_v1/policy_predictions.npz",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_signed_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    signature = value.pop("canonical_sha256", None)
    if signature != canonical_sha256(value):
        raise RuntimeError(f"Canonical JSON signature mismatch: {path}")
    value["canonical_sha256"] = signature
    return value


def _require_committed(path: Path) -> None:
    relative = path.relative_to(ROOT).as_posix()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--", relative], cwd=ROOT, text=True).strip()
    if dirty:
        raise RuntimeError(f"Freeze input must be committed before this stage: {relative}")


def validate_successor_payload(
    parent: Mapping[str, Any],
    successor: Mapping[str, Any],
    selection_input_hashes: Mapping[str, str],
    *,
    parent_file_sha256: str,
) -> dict[str, Any]:
    if parent_file_sha256 != ORIGINAL_FREEZE_SHA256:
        raise RuntimeError("Original application freeze bytes differ from the preserved V1 artifact")
    if successor.get("freeze_version") != 2:
        raise RuntimeError("Expected application freeze successor version 2")
    if successor.get("parent_application_freeze_path") != ORIGINAL_FREEZE.relative_to(ROOT).as_posix():
        raise RuntimeError("Successor points to a different parent freeze")
    if successor.get("parent_application_freeze_sha256") != parent_file_sha256:
        raise RuntimeError("Successor parent freeze hash mismatch")
    if successor.get("status") != "FROZEN_BEFORE_E79_PROBABILITY_BUNDLE_ACCESS":
        raise RuntimeError("Successor is not frozen before E79 probability access")

    allowed_successor_only = {
        "freeze_version",
        "parent_application_freeze_path",
        "parent_application_freeze_sha256",
        "freeze_amendment",
        "application_input_hashes",
        "canonical_sha256",
    }
    for key, value in parent.items():
        if key in {"canonical_sha256", "application_input_hashes"}:
            continue
        if successor.get(key) != value:
            raise RuntimeError(f"Successor changed frozen decision field: {key}")
    unexpected = set(successor) - set(parent) - allowed_successor_only
    if unexpected:
        raise RuntimeError(f"Unexpected successor fields: {sorted(unexpected)}")

    parent_inputs = dict(parent["application_input_hashes"])
    if set(parent_inputs) != {
        "experiments/runs/optimization_v3_a02_cgeom_benchmark_v1/test_predictions.npz",
        "data/raw/classification/Test.csv",
        "submissions/optimization_v3/cls_mix50_a02_missing_e79_v1/classification_mix50_present_a02_missing_e79.csv",
        "reports/optimization_v3/cls_mix50_a02_missing_e79_v1/MANIFEST.json",
        "reports/optimization_v3/cls_mix50_a02_missing_e79_v1/QA.json",
    }:
        raise RuntimeError("Preserved parent application freeze is not the expected five-input artifact")
    if set(selection_input_hashes) != APPLICATION_ADDED_INPUTS:
        raise RuntimeError("Prefit freeze does not contain the exact two selection-input hashes")
    expected_inputs = {**parent_inputs, **dict(selection_input_hashes)}
    if successor.get("application_input_hashes") != expected_inputs:
        raise RuntimeError("Successor input map is not exactly parent five plus Train/OOF hashes")
    if set(expected_inputs) != set(parent_inputs) | APPLICATION_ADDED_INPUTS or len(expected_inputs) != 7:
        raise RuntimeError("Successor must bind exactly seven verified builder inputs")
    amendment = successor.get("freeze_amendment", {})
    if (
        amendment.get("threshold_unchanged") is not True
        or amendment.get("final_selector_receipt_unchanged") is not True
        or amendment.get("official_selectors_rerun") is not False
        or amendment.get("E79_probability_arrays_read_before_successor_commit") is not False
        or amendment.get("Test_content_read_before_successor_commit") is not False
        or amendment.get("model_fits_or_inference") != 0
    ):
        raise RuntimeError("Successor amendment changed frozen decisions or reports early E79/Test access")
    return expected_inputs


def verify_successor_guard(*, require_committed_guard: bool = False) -> dict[str, Any]:
    _require_committed(ORIGINAL_FREEZE)
    _require_committed(SUCCESSOR_FREEZE)
    if sha256_file(ORIGINAL_FREEZE) != ORIGINAL_FREEZE_SHA256:
        raise RuntimeError("Original application freeze was modified")
    parent = read_signed_json(ORIGINAL_FREEZE)
    successor = read_signed_json(SUCCESSOR_FREEZE)

    # Source PASS and Train/OOF pins are rechecked before any Test/E79 array access.
    approved_prefit_commit = decision.verify_prefit_checkout()
    prefit = decision._read_json(decision.PREFIT_FREEZE_PATH)
    expected_inputs = validate_successor_payload(
        parent,
        successor,
        prefit["selection_input_hashes"],
        parent_file_sha256=sha256_file(ORIGINAL_FREEZE),
    )
    actual_inputs = decision.validate_inputs(include_e79=True)
    if actual_inputs != expected_inputs:
        raise RuntimeError("Successor input map differs from the actual seven frozen file hashes")

    # Direct the unchanged, pinned builder to the committed V2 successor.
    decision.E79_APPLICATION_FREEZE_PATH = SUCCESSOR_FREEZE
    application_freeze, _ = decision._verify_application_freeze_committed()
    if application_freeze["final_selector_threshold"] != parent["final_selector_threshold"]:
        raise RuntimeError("Versioned successor changed the final threshold")
    if application_freeze["final_selector_receipt_sha256"] != parent["final_selector_receipt_sha256"]:
        raise RuntimeError("Versioned successor changed the final selector receipt")

    result = {
        "schema_version": 1,
        "status": "PASS_VERSIONED_FREEZE_GUARD",
        "policy_id": decision.POLICY_ID,
        "implementation_version": decision.IMPLEMENTATION_VERSION,
        "approved_prefit_commit": approved_prefit_commit,
        "original_freeze_sha256": sha256_file(ORIGINAL_FREEZE),
        "successor_freeze_path": SUCCESSOR_FREEZE.relative_to(ROOT).as_posix(),
        "successor_freeze_sha256": sha256_file(SUCCESSOR_FREEZE),
        "successor_canonical_sha256": successor["canonical_sha256"],
        "application_input_hashes": actual_inputs,
        "final_selector_threshold": parent["final_selector_threshold"],
        "final_selector_receipt_sha256": parent["final_selector_receipt_sha256"],
        "original_freeze_byte_identical_to_parent_commit": True,
        "seven_input_map_matches_builder": True,
        "official_selectors_rerun": False,
        "test_rows_or_fields_parsed": False,
        "E79_probability_arrays_opened": False,
        "sample_submission_read": False,
        "model_fits_or_inference": 0,
        "kaggle_contacted": False,
    }
    if require_committed_guard:
        _require_committed(GUARD_QA)
        guard = read_signed_json(GUARD_QA)
        for key in (
            "status",
            "original_freeze_sha256",
            "successor_freeze_sha256",
            "successor_canonical_sha256",
            "application_input_hashes",
            "final_selector_threshold",
            "final_selector_receipt_sha256",
        ):
            if guard.get(key) != result[key]:
                raise RuntimeError(f"Committed successor guard QA differs at {key}")
    return result


def write_guard_qa() -> dict[str, Any]:
    result = verify_successor_guard()
    body = dict(result)
    body["canonical_sha256"] = canonical_sha256(body)
    payload = (json.dumps(body, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    GUARD_QA.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    fd = os.open(GUARD_QA, flags, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return {**body, "guard_qa_sha256": hashlib.sha256(payload).hexdigest()}


def build_candidate() -> dict[str, Any]:
    verify_successor_guard(require_committed_guard=True)
    # The frozen V2 builder now sees only the committed versioned successor.
    decision.E79_APPLICATION_FREEZE_PATH = SUCCESSOR_FREEZE
    return decision.build_e79_candidate()


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--verify-freeze", action="store_true")
    group.add_argument("--build-e79", action="store_true")
    args = parser.parse_args()
    result = write_guard_qa() if args.verify_freeze else build_candidate()
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
