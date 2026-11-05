"""Build and submit the two root-authorized REG simplification E79 policies.

This executor consumes only pinned, already-saved prediction arrays. It performs
no model, preprocessing, or calibration fits and never reads target labels.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping
import uuid

import numpy as np
import psutil
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
TP2_ROOT = ROOT.parent / "TP 2"
AUDIT_ROOT = ROOT.parent / "tk2-simplification-audit-v1"
FRONTIER_ROOT = ROOT.parent / "tk2-regression-frontier"
NAMESPACE = "reg_simplification_benchmark_v1"
EXPERIMENT_DIR = ROOT / "experiments" / NAMESPACE
REPORT_DIR = ROOT / "reports" / "optimization_v3" / NAMESPACE
SUBMISSION_DIR = ROOT / "submissions" / NAMESPACE
MANIFEST_PATH = EXPERIMENT_DIR / "INPUT_MANIFEST_R1.json"
BUILD_RECEIPT = REPORT_DIR / "BUILD_RECEIPT_R1.json"
RECONSTRUCTION_RECEIPT = REPORT_DIR / "RECONSTRUCTION_QA_R1.json"
LOCAL_DUPLICATE_RECEIPT = REPORT_DIR / "LOCAL_DUPLICATE_SCAN_R1.json"
EXECUTION_POLICY_PATH = REPORT_DIR / "EXECUTION_POLICY_R1.json"
RESOURCE_RECEIPT = REPORT_DIR / "RESOURCE_GATE_R1.json"
KAGGLE_HISTORY_DIR = REPORT_DIR / "KAGGLE_HISTORY_R1"
UPLOAD_DIR = REPORT_DIR / "UPLOAD_R1"

AUDIT_COMMIT = "0052f9c9aaa564caf6d9292dc9d9fa369adeef4e"
AUDIT_FREEZE_HEAD = "a39721172b6d293cb68ee0d120060082d222699e"
COMPETITION = "tp2-regression-cyber-vulnerability-kasdd-gasal-2026-2027"
EXPECTED_ROWS = 29_421
EXPECTED_MISSING_ROWS = 1_062
KNOT = 50.0
HINGE = {
    "intercept": 1.7288079036674198,
    "slope_low": 0.9933285599956824,
    "slope_high": 0.957778072288735,
}
RAM_START_FLOOR_BYTES = int(2.5 * 1024**3)
RAM_RESERVE_BYTES = int(1.0 * 1024**3)
RAM_ESTIMATE_BYTES = 512 * 1024**2
LANE_LOCK_PATH = Path(tempfile.gettempdir()) / "kasdad_reg_simplification_benchmark_v1.lock"

CSV_PATHS = {
    "NO_POS": SUBMISSION_DIR / "reg_no_pos_hinge_e79.csv",
    "NO_CHAR": SUBMISSION_DIR / "reg_no_char_hinge_e79.csv",
}
UPLOAD_MESSAGES = {
    "NO_POS": "Samuel REG Simplified No POS Hinge Benchmark",
    "NO_CHAR": "Samuel REG Simplified No Character Hinge Benchmark",
}

# These hashes are copied from the signed predecessor receipts, frozen source
# audit, or exact artifact manifests. The manifest repeats them for review.
INPUTS: dict[str, tuple[Path, str]] = {
    "audit_findings": (AUDIT_ROOT / "reports/simplification_v1/FINDINGS.md",
                       "5073d955f5d2b932bef7e18df1b8498adec3885873867d64da3f5b272e4cc000"),
    "audit_protocol": (AUDIT_ROOT / "reports/simplification_v1/PROTOCOL.md",
                       "907c2b1323ea759a057527aabb043ae2179039bc67596dcd03a385187da62b55"),
    "audit_decisions": (AUDIT_ROOT / "reports/simplification_v1/DECISIONS.json",
                        "e9ef07ff4a364951ca13598f4601aff4d4b005fa5bc3855382e6b028f279157d"),
    "audit_input_freeze": (AUDIT_ROOT / "reports/simplification_v1/INPUT_FREEZE.json",
                           "d1f8f77864c1b2c1df8dc11732ffb61b68ac08f5e0b37ded48abf33f7506ac56"),
    "audit_verification": (AUDIT_ROOT / "reports/simplification_v1/VERIFICATION.json",
                           "3c342d1e30a41fa882a4b6fed241b9212c533a3138980124d91a3ae6fbac35ac"),
    "audit_runner": (AUDIT_ROOT / "scripts/simplification_audit_v1.py",
                     "b943e16f459d78fec736a801ac917f925737a266d13d3c160f6dae4b06ef3edc"),
    "audit_tests": (AUDIT_ROOT / "tests/test_simplification_audit_v1.py",
                    "72e3348502e61e70982089c82c84c313453c3ab8b57ecd7bc21a512d5eab5e61"),
    "r06_archive": (TP2_ROOT / "experiments/runs/optimization_v3_expanded_submissions_v1/R06_ORIGINAL/predictions.npz",
                    "ce034e82a7063ec5dd873b1126d60731dd291d3852c57936c7aa1ee449a81eba"),
    "r06_manifest": (TP2_ROOT / "reports/submissions/optimization_v3_expanded_v1/R06_ORIGINAL/manifest.json",
                     "3f77e4ac148ada5fb79e5e2c16e7975e77726f4bfcc42a29f9234f801c697d67"),
    "r06_fit_state": (TP2_ROOT / "experiments/runs/optimization_v3_expanded_submissions_v1/R06_ORIGINAL/fit_state.json",
                      "6227dae63e73ee4cf282729bac0221dc558cc08111975adf54f7348cee3f9f0a"),
    "r06_csv": (TP2_ROOT / "submissions/optimization_v3_expanded_v1/r06_original.csv",
                "4a72b13e0d6ac93e18adee23010bb15ddf15fcb2a9543dc903154d5af60ee64f"),
    "current_r06_archive": (ROOT / "experiments/runs/REG_CURRENT_R06_E79_EXPLORATORY_BATCH_V1/inference_B.npz",
                            "e29b1eabb1692729749b25d88b7cf86d1bdf44c34ce74ab964b124074f5817f2"),
    "current_r06_manifest": (ROOT / "reports/optimization_v3/REG_CURRENT_R06_E79_EXPLORATORY_BATCH_V1/INFERENCE_MANIFEST_B.json",
                             "d2bd7dea17feab1740ca4630bb5cf7ace62caf8746051b3322cc38d168674484"),
    "current_r06_csv_qa": (ROOT / "reports/optimization_v3/REG_CURRENT_R06_E79_EXPLORATORY_BATCH_V1/CSV_QA_B.json",
                           "bc19efcbbf57ae0e621fe1b338f00e1726380ed9a736947f6bf760b8a9fa67c8"),
    "current_r06_prelabel": (ROOT / "reports/optimization_v3/REG_CURRENT_R06_E79_EXPLORATORY_BATCH_V1/PRELABEL_PASS_B.json",
                             "a4771eacfe808e5645977008827ebc9552c288dbed5910606649e3d4670716c3"),
    "current_r06_trained_freeze": (ROOT / "reports/optimization_v3/REG_CURRENT_R06_E79_EXPLORATORY_BATCH_V1/TRAINED_STATE_FREEZE_B.json",
                                   "3364e75a8c791cebbd9fd4e2693149fa1429e7e5ef19f68bb1cadb6828b51218"),
    "current_r06_csv": (ROOT / "submissions/reg_current_r06_e79_exploratory_v1/direct_position_real_e79.csv",
                        "49ae7f668aea99e433c2bec72ed5ff51126fb12d888c3185ec048730d767ea6f"),
    "final_hinge_report": (ROOT / "reports/optimization_v3/reg_monotone_hinge_p50_v1/FINAL_APPLICATION_REPORT_R1.json",
                           "fec7182eace04b425b7dbf1d871dbb39f14a1080e479ba95243803e4b1cdae73"),
    "final_hinge_verification": (ROOT / "reports/optimization_v3/reg_monotone_hinge_p50_v1/FINAL_APPLICATION_VERIFICATION_R1.json",
                                 "af859d6efcf90b3044ba55c37f2ae3e5e27132ae520c1cd90aad28580fadf97c"),
    "final_hinge_receipt": (ROOT / "reports/optimization_v3/reg_monotone_hinge_p50_v1/FINAL_APPLICATION_RECEIPT_R1.json",
                            "364a0cbe8258127e6ff963f9117e11a8d61c1ec05bb3bc4d2ca5962f117ad89f"),
    "final_hinge_root_pass": (ROOT / "experiments/reg_monotone_hinge_p50_v1/FINAL_APPLICATION_ROOT_SOURCE_PASS_R1.json",
                              "ca35d7f580ee2fd2c5e7173309e89482b69c3eba7a2c29a7adbd0a1c791acd2f"),
    "final_hinge_freeze": (ROOT / "experiments/reg_monotone_hinge_p50_v1/FINAL_APPLICATION_FREEZE_R1.json",
                           "e3522ab957381dc32c1e37ba41863781a736291982b7c8f1364dd4d8c7def049"),
    "final_hinge_fit_state": (ROOT / "experiments/runs/reg_monotone_hinge_p50_v1/FINAL_APPLICATION_R1/fits/HINGE_FINAL.json",
                              "a81b2e8e709a33e9a15036ab040c87c92faabd08f16b48ff7f42916fd23ad99a"),
    "final_hinge_fit_input": (ROOT / "experiments/runs/reg_monotone_hinge_p50_v1/FINAL_APPLICATION_R1/fit_inputs/HINGE_FINAL.npz",
                              "e75f70cccd1bb231499060b84155af876173214d984461e39e4c1364e4c5b2bf"),
    "final_hinge_csv": (ROOT / "submissions/reg_monotone_hinge_p50_v1/hinge_p50_final_exploratory_e79.csv",
                        "a5813fdd384da31611db1e945c7db33199d8ef9044fb71dd2941ef7af6422da9"),
    "sample_submission": (TP2_ROOT / "data/raw/regression/SampleSubmission.csv",
                          "63a51f4b7dfbb0eec9800099407abd8be14d25ec18a533cd39dc854ffc190a32"),
}


class BenchmarkError(RuntimeError):
    """Fail-closed input, resource, output, or submission condition."""


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(",", ":")).encode("utf-8")


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def array_sha(value: Any, *, dtype: str | None = None) -> str:
    array = np.asarray(value, dtype=dtype)
    if array.dtype.kind in "OUS":
        data = canonical_bytes(array.astype(str).tolist())
    else:
        data = np.ascontiguousarray(array).tobytes(order="C")
    return hashlib.sha256(data).hexdigest()


def source_array_sha(value: Any, *, dtype: str | None = None) -> str:
    """Match the shape/dtype-prefixed vector digest used by the pinned E79 source."""
    array = np.asarray(value, dtype=dtype)
    digest = hashlib.sha256(canonical_bytes({"shape": array.shape, "dtype": str(array.dtype)}))
    if array.dtype.kind in "OUS":
        digest.update(canonical_bytes(array.astype(str).tolist()))
    else:
        digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BenchmarkError(f"Cannot read required JSON: {path}") from exc
    if not isinstance(result, dict):
        raise BenchmarkError(f"Expected JSON object at {path}")
    return result


def verify_signature(document: Mapping[str, Any], *, key: str = "signature") -> None:
    body = {name: value for name, value in document.items()
            if name not in {key, "signature_method"}}
    if document.get(key) != canonical_sha(body):
        raise BenchmarkError(f"Canonical signature mismatch for {key}")


def write_once(path: Path, value: Mapping[str, Any]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(value), ensure_ascii=False, sort_keys=True,
                         allow_nan=False, indent=2).encode("utf-8") + b"\n"
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise BenchmarkError(f"Refusing to overwrite immutable artifact: {path}") from exc
    return hashlib.sha256(payload).hexdigest()


def append_upload_event(path: Path, value: Mapping[str, Any], *, first: bool) -> None:
    """Durably create the first event or append one response to its intent."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_bytes(dict(value)) + b"\n"
    try:
        if first:
            stream = path.open("xb")
        else:
            if not path.is_file():
                raise BenchmarkError("Cannot append an upload response without a durable intent")
            existing = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            new_event = value.get("event")
            if (not existing or existing[0].get("event") != "UPLOAD_INTENT"
                    or (len(existing) == 1 and new_event != "UPLOAD_RESPONSE")
                    or (len(existing) >= 2 and existing[1].get("event") != "UPLOAD_RESPONSE")
                    or (len(existing) >= 2 and new_event != "COMPETITION_STATUS_CONFIRMED")
                    or (new_event == "UPLOAD_RESPONSE" and len(existing) != 1)):
                raise BenchmarkError("Upload log event order is invalid; never retry")
            if any(event.get("event") == "COMPETITION_STATUS_CONFIRMED"
                   and event.get("status") == "COMPLETE_SCORE_CONFIRMED" for event in existing):
                raise BenchmarkError("A COMPLETE score is already confirmed")
            stream = path.open("ab")
        with stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise BenchmarkError(f"Upload event log already exists; never retry: {path}") from exc


def verify_execution_policy() -> dict[str, Any]:
    policy = _read_json(EXECUTION_POLICY_PATH)
    required = {
        "schema_version": 1,
        "policy_id": "REG_CACHED_VECTOR_PARALLEL_R1",
        "lane": "REG",
        "maximum_concurrent_light_jobs_total": 2,
        "reg_candidates_serial_within_lane": True,
        "incremental_memory_estimate_bytes": RAM_ESTIMATE_BYTES,
        "parallel_start_available_floor_bytes": RAM_START_FLOOR_BYTES,
        "minimum_post_run_available_bytes": RAM_RESERVE_BYTES,
        "heavy_lock_required": False,
        "per_lane_idempotency_lock_required": True,
        "active_polling_or_scheduler_allowed": False,
        "upload_retry_allowed": False,
        "data_guards_changed": False,
        "new_fits_allowed": 0,
        "test_labels_allowed": False,
    }
    for key, expected in required.items():
        if policy.get(key) != expected:
            raise BenchmarkError(f"Execution-policy receipt differs for {key}")
    return policy


def verify_inputs() -> dict[str, str]:
    if not MANIFEST_PATH.is_file():
        raise BenchmarkError(f"Input manifest is missing: {MANIFEST_PATH}")
    manifest = _read_json(MANIFEST_PATH)
    if manifest.get("namespace") != NAMESPACE or manifest.get("zero_new_fits") is not True:
        raise BenchmarkError("Input manifest namespace or zero-fit contract differs")
    actual_hashes: dict[str, str] = {}
    pinned = manifest.get("inputs", {})
    if set(pinned) != set(INPUTS):
        raise BenchmarkError("Input manifest file inventory differs from the reviewed allowlist")
    for name, (path, expected_sha) in INPUTS.items():
        record = pinned.get(name)
        if (not isinstance(record, dict) or Path(record.get("path", "")).resolve() != path.resolve()
                or record.get("sha256") != expected_sha):
            raise BenchmarkError(f"Manifest pin differs for {name}")
        if not path.is_file():
            raise BenchmarkError(f"Pinned input is missing: {name} ({path})")
        actual = sha_file(path)
        if actual != expected_sha:
            raise BenchmarkError(f"Pinned input bytes changed: {name}")
        actual_hashes[name] = actual

    _verify_root_audit()
    _verify_r06_and_hinge_authority()
    return actual_hashes


def _git_is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    result = subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor",
                             ancestor, descendant], capture_output=True)
    return result.returncode == 0


def _verify_root_audit() -> None:
    result = subprocess.run(["git", "-C", str(AUDIT_ROOT), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=False)
    if result.returncode:
        raise BenchmarkError("Cannot resolve read-only simplification audit HEAD")
    head = result.stdout.strip()
    if not _git_is_ancestor(AUDIT_ROOT, AUDIT_FREEZE_HEAD, AUDIT_COMMIT):
        raise BenchmarkError("Root audit input freeze is not an ancestor of the findings commit")
    if not _git_is_ancestor(AUDIT_ROOT, AUDIT_COMMIT, head):
        raise BenchmarkError("Named root audit findings commit is not an ancestor of its read-only worktree")
    freeze = _read_json(AUDIT_ROOT / "reports/simplification_v1/INPUT_FREEZE.json")
    if freeze.get("head") != AUDIT_FREEZE_HEAD or freeze.get("new_fits") != 0:
        raise BenchmarkError("Root audit freeze head or fit accounting changed")
    if freeze.get("new_raw_data_access") is not False:
        raise BenchmarkError("Root audit no-raw-data boundary differs")


def _verify_r06_and_hinge_authority() -> None:
    r06 = _read_json(INPUTS["r06_manifest"][0])
    if r06.get("candidate") != "R06_ORIGINAL" or r06.get("status") != "PREPARED_VALIDATED_NOT_SUBMITTED":
        raise BenchmarkError("R06 ORIGINAL source manifest identity/status differs")
    recipe = r06.get("recipe", {})
    if recipe.get("blend") != "clip(0.75*three-component champion + 0.25*R06 rank50, 0, 100)":
        raise BenchmarkError("R06 ORIGINAL full-champion recipe differs")

    b_manifest = _read_json(INPUTS["current_r06_manifest"][0])
    verify_signature(b_manifest)
    if (b_manifest.get("status") != "CSV_READY_PENDING_INDEPENDENT_QA"
            or b_manifest.get("candidate_id") != "DIRECT_POSITION_REAL_E79"
            or b_manifest.get("test_labels_loaded") is not False
            or b_manifest.get("upload_performed") is not False
            or b_manifest.get("baseline_member_used") != "manifest-bound final mapped R06 prediction"
            or b_manifest.get("baseline_raw_candidate_member_used") is not False):
        raise BenchmarkError("Existing E79 position archive boundary/status differs")
    if (b_manifest.get("baseline_prediction_member_digest")
            != r06.get("prediction_audit", {}).get("prediction_sha256")):
        raise BenchmarkError("E79 baseline digest does not match the pinned R06 prediction manifest")
    trained = _read_json(INPUTS["current_r06_trained_freeze"][0])
    verify_signature(trained)
    if trained.get("status") != "TRAINED_STATE_FROZEN_EXPLORATORY_ONLY" or trained.get("test_labels_loaded") is not False:
        raise BenchmarkError("Existing position model trained-state freeze differs")
    prelabel = _read_json(INPUTS["current_r06_prelabel"][0])
    verify_signature(prelabel)
    if prelabel.get("status") != "PRELABEL_PASS" or prelabel.get("test_labels_loaded") is not False:
        raise BenchmarkError("Existing position model prelabel pass differs")
    csv_qa = _read_json(INPUTS["current_r06_csv_qa"][0])
    verify_signature(csv_qa)
    if (csv_qa.get("status") != "PASS_INDEPENDENT_CSV_QA"
            or csv_qa.get("test_labels_read") is not False
            or csv_qa.get("ordered_test_ids_sha256") != b_manifest.get("ordered_official_test_ids_sha256")):
        raise BenchmarkError("Existing position E79 QA receipt differs")

    final_report = _read_json(INPUTS["final_hinge_report"][0])
    verify_signature(final_report)
    fit = final_report.get("final_slot", {})
    coefficients = fit.get("coefficients", {})
    if (fit.get("fit_id") != "HINGE_FINAL" or fit.get("fit_rows") != 20_903
            or coefficients != HINGE or final_report.get("test_labels_read") is not False
            or final_report.get("kaggle_upload") is not False):
        raise BenchmarkError("Fixed final HINGE coefficients or boundary differs from signed report")
    old_pass = _read_json(INPUTS["final_hinge_root_pass"][0])
    verify_signature(old_pass)
    if (old_pass.get("verdict") != "PASS"
            or old_pass.get("authorized_fit_ids") != ["HINGE_FINAL"]
            or old_pass.get("authorized_final_lp_starts") != 1):
        raise BenchmarkError("Prior HINGE fit authorization is not the expected one-slot source")
    verification = _read_json(INPUTS["final_hinge_verification"][0])
    receipt = _read_json(INPUTS["final_hinge_receipt"][0])
    if (verification.get("status") != "PASS_INDEPENDENT_FINAL_APPLICATION_REPLAY"
            or verification.get("final_report_sha256") != sha_file(INPUTS["final_hinge_report"][0])
            or receipt.get("status") != "PASS_INDEPENDENT_FINAL_APPLICATION_REPLAY"
            or receipt.get("final_report_sha256") != sha_file(INPUTS["final_hinge_report"][0])
            or receipt.get("verification_sha256") != sha_file(INPUTS["final_hinge_verification"][0])):
        raise BenchmarkError("Existing final HINGE report/verification/receipt chain differs")


def _acquire_resource_lock():
    """Take one REG-lane idempotency lock and apply the explicit light-job gate."""
    LANE_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    owner = {
        "lane": "REG",
        "pid": os.getpid(),
        "token": uuid.uuid4().hex,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    payload = canonical_bytes(owner)
    descriptor: int | None = None
    acquired = False
    try:
        descriptor = os.open(str(LANE_LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        acquired = True
        os.write(descriptor, payload)
        os.fsync(descriptor)
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
            descriptor = None
        if acquired:
            try:
                LANE_LOCK_PATH.unlink()
            except FileNotFoundError:
                pass
        raise BenchmarkError("REG lane lock exists or cannot be acquired; do not poll or reuse") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    available = int(psutil.virtual_memory().available)
    projected = available - RAM_ESTIMATE_BYTES
    if available < RAM_START_FLOOR_BYTES or projected < RAM_RESERVE_BYTES:
        _release_resource_lock()
        raise BenchmarkError(
            f"Light-job RAM gate failed: available={available}, projected={projected}; floors remain unchanged")
    return owner, {"start_available_bytes": available,
                   "estimated_incremental_bytes": RAM_ESTIMATE_BYTES,
                   "projected_available_bytes": projected,
                   "parallel_start_floor_bytes": RAM_START_FLOOR_BYTES,
                   "projected_reserve_floor_bytes": RAM_RESERVE_BYTES,
                   "per_lane_lock_path": str(LANE_LOCK_PATH),
                   "per_lane_lock_token": owner["token"],
                   "shared_heavy_lock_used": False}


def _release_resource_lock() -> None:
    try:
        LANE_LOCK_PATH.unlink()
    except FileNotFoundError:
        pass


def _load_npz_allowlist(path: Path, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(keys):
                raise BenchmarkError(f"Unexpected NPZ member inventory: {path}")
            return {key: np.asarray(archive[key]) for key in keys}
    except (OSError, ValueError) as exc:
        raise BenchmarkError(f"Cannot load pinned saved predictions: {path}") from exc


def _csv_sample_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise BenchmarkError("Official SampleSubmission is empty") from exc
        if header != ["record_id", "threat_rank"]:
            raise BenchmarkError(f"Official SampleSubmission columns differ: {header}")
        ids = []
        for row_number, row in enumerate(reader, start=2):
            if len(row) != 2 or not row[0]:
                raise BenchmarkError(f"Malformed SampleSubmission row {row_number}")
            ids.append(row[0])
    return ids


def _require_vector(value: Any, name: str, *, n: int = EXPECTED_ROWS,
                    finite: bool = True) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or len(array) != n:
        raise BenchmarkError(f"{name} has unexpected shape")
    if finite and not np.isfinite(array).all():
        raise BenchmarkError(f"{name} contains nonfinite values")
    return array


def _hinge_present(raw: np.ndarray, missing: np.ndarray) -> np.ndarray:
    out = np.asarray(raw, dtype="<f8").copy()
    present = ~np.asarray(missing, dtype=bool)
    x = np.asarray(raw[present], dtype=np.float64)
    mapped = (HINGE["intercept"]
              + HINGE["slope_low"] * np.minimum(x, KNOT)
              + HINGE["slope_high"] * np.maximum(x - KNOT, 0.0))
    out[present] = np.clip(mapped, 0.0, 100.0)
    return out


def _rank50_unclipped(raw: np.ndarray, cycles: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    raw = np.asarray(raw, dtype=np.float64)
    cycles = np.asarray(cycles)
    mapped = np.empty_like(raw)
    counts: dict[str, int] = {}
    for cycle in np.unique(cycles):
        mask = cycles == cycle
        percentile = 100.0 * rankdata(raw[mask], method="average") / int(mask.sum())
        mapped[mask] = 0.5 * raw[mask] + 0.5 * percentile
        counts[str(int(cycle))] = int(mask.sum())
    if not np.isfinite(mapped).all():
        raise BenchmarkError("Rank50 position mapping produced nonfinite values")
    return mapped, counts


def _reconstruct_vectors() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    r06 = _load_npz_allowlist(INPUTS["r06_archive"][0],
        ("champion", "raw_candidate", "rank_mapped_candidate", "cycle_code", "prediction"))
    current = _load_npz_allowlist(INPUTS["current_r06_archive"][0],
        ("record_id", "raw", "cycle_code", "missing", "baseline", "prediction"))
    source_ids = _require_vector(current["record_id"], "record_id")
    ids = source_ids.astype(str)
    raw_pos = _require_vector(current["raw"], "raw position", finite=True).astype(np.float64)
    cycles = _require_vector(current["cycle_code"], "cycle_code").astype(np.int64)
    missing = _require_vector(current["missing"], "missing mask").astype(bool)
    r06_cycles = _require_vector(r06["cycle_code"], "R06 cycle_code").astype(np.int64)
    legacy_three = _require_vector(r06["champion"], "R06 three-component champion").astype(np.float64)
    r06_char_mapped = _require_vector(r06["rank_mapped_candidate"], "R06 rank-mapped branch").astype(np.float64)
    r06_final = _require_vector(r06["prediction"], "R06 final prediction").astype(np.float64)
    current_baseline = _require_vector(current["baseline"], "current R06 baseline").astype(np.float64)
    current_prediction = _require_vector(current["prediction"], "current position prediction").astype(np.float64)
    r06_legacy_raw = _require_vector(r06["raw_candidate"], "R06 raw branch").astype(np.float64)
    if len(set(ids.tolist())) != EXPECTED_ROWS:
        raise BenchmarkError("E79 official IDs are not unique")
    if len(np.unique(r06_cycles)) != len(np.unique(cycles)) or not np.array_equal(r06_cycles, cycles):
        raise BenchmarkError("R06 and current E79 cycle order differs")
    if int(missing.sum()) != EXPECTED_MISSING_ROWS:
        raise BenchmarkError(f"Unexpected missing rows: {int(missing.sum())}")
    if not np.isfinite(legacy_three).all() or np.any((legacy_three < 0.0) | (legacy_three > 100.0)):
        raise BenchmarkError("Saved legacyThree is outside [0,100]")
    if not np.isfinite(r06_final).all() or np.any((r06_final < 0.0) | (r06_final > 100.0)):
        raise BenchmarkError("Saved R06 final prediction is outside [0,100]")
    if not np.array_equal(r06_final.view("<u8"), current_baseline.view("<u8")):
        raise BenchmarkError("Current E79 baseline is not bit-identical to the pinned R06 final prediction")
    if (source_array_sha(r06_final, dtype="<f8") != _read_json(INPUTS["r06_manifest"][0])
            .get("prediction_audit", {}).get("prediction_sha256")
            or source_array_sha(current_baseline, dtype="<f8")
            != _read_json(INPUTS["current_r06_manifest"][0]).get("baseline_prediction_member_digest")
            or source_array_sha(current_prediction, dtype="<f8")
            != _read_json(INPUTS["current_r06_manifest"][0]).get("prediction_sha256")
            or source_array_sha(source_ids) != _read_json(INPUTS["current_r06_manifest"][0])
            .get("ordered_official_test_ids_sha256")):
        raise BenchmarkError("Saved R06/E79 prediction or ID digests differ from source manifests")
    r06_rebuilt = np.clip(0.75 * legacy_three + 0.25 * r06_char_mapped, 0.0, 100.0)
    if not np.allclose(r06_rebuilt, r06_final, rtol=0.0, atol=1e-10):
        raise BenchmarkError("Full R06 champion reconstruction failed against the saved final prediction")

    # Keep the original character leaf and position auxiliary mapped exactly
    # once over every complete cycle before applying the missing route.
    pos_rank50, cycle_counts = _rank50_unclipped(raw_pos, cycles)
    no_pos = _hinge_present(r06_final, missing)
    no_char_raw = np.asarray(legacy_three, dtype=np.float64).copy()
    present = ~missing
    no_char_raw[present] = np.clip(
        (9.0 / 13.0) * legacy_three[present] + (4.0 / 13.0) * pos_rank50[present],
        0.0, 100.0)
    no_char = _hinge_present(no_char_raw, missing)

    if not np.array_equal(no_pos[missing].view("<u8"), r06_final[missing].view("<u8")):
        raise BenchmarkError("NO_POS missing route changed raw R06 bits")
    if not np.array_equal(no_char[missing].view("<u8"), legacy_three[missing].view("<u8")):
        raise BenchmarkError("NO_CHAR missing route changed legacyThree bits")
    if np.any((no_pos < 0.0) | (no_pos > 100.0)) or np.any((no_char < 0.0) | (no_char > 100.0)):
        raise BenchmarkError("Final candidate values are outside [0,100]")

    sample_ids = _csv_sample_ids(INPUTS["sample_submission"][0])
    if sample_ids != ids.tolist():
        raise BenchmarkError("Saved E79 IDs/order differ from official SampleSubmission")
    if len(sample_ids) != EXPECTED_ROWS:
        raise BenchmarkError("Official SampleSubmission row count differs")

    arrays = {
        "record_ids": ids,
        "cycles": cycles,
        "missing": missing,
        "legacy_three": legacy_three,
        "r06_final": r06_final,
        "raw_pos": raw_pos,
        "position_rank50_unclipped": pos_rank50,
        "no_pos": no_pos,
        "no_char_raw": no_char_raw,
        "no_char": no_char,
    }
    reconstruction = {
        "status": "PASS_EXACT_R06_AND_ABLATION_RECONSTRUCTION",
        "rows": EXPECTED_ROWS,
        "unique_ids": len(set(ids.tolist())),
        "missing_rows": int(missing.sum()),
        "present_rows": int(present.sum()),
        "cycles": cycle_counts,
        "r06_three_component_champion_sha256": array_sha(legacy_three, dtype="<f8"),
        "r06_character_rank50_sha256": array_sha(r06_char_mapped, dtype="<f8"),
        "r06_full_champion_sha256": array_sha(r06_final, dtype="<f8"),
        "current_r06_baseline_sha256": array_sha(current_baseline, dtype="<f8"),
        "current_position_prediction_sha256": array_sha(current_prediction, dtype="<f8"),
        "r06_raw_branch_sha256": array_sha(r06_legacy_raw, dtype="<f8"),
        "raw_position_auxiliary_sha256": array_sha(raw_pos, dtype="<f8"),
        "unclipped_rank50_position_sha256": array_sha(pos_rank50, dtype="<f8"),
        "r06_reconstruction_max_abs_error": float(np.max(np.abs(r06_rebuilt - r06_final))),
        "r06_baseline_bit_exact_to_current_inference": True,
        "official_sample_id_order_exact": True,
        "no_pos_missing_bit_exact_to_r06": True,
        "no_char_missing_bit_exact_to_legacyThree": True,
        "new_fits": 0,
        "test_labels_read": False,
    }
    return arrays, reconstruction


def _write_csv_once(path: Path, ids: np.ndarray, values: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise BenchmarkError(f"Refusing to overwrite a candidate CSV: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise BenchmarkError(f"Refusing to reuse a temporary output: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(["record_id", "threat_rank"])
            for record_id, value in zip(ids, values, strict=True):
                writer.writerow([str(record_id), format(float(value), ".17g")])
            stream.flush()
            os.fsync(stream.fileno())
        # Hard-link creation is exclusive on the same volume; it will fail if
        # a concurrent process created the immutable destination.
        os.link(temporary, path)
        temporary.unlink()
    except FileExistsError as exc:
        raise BenchmarkError(f"Refusing to overwrite candidate CSV: {path}") from exc
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return sha_file(path)


def _iter_submission_csvs() -> list[Path]:
    roots = [TP2_ROOT / "submissions", ROOT / "submissions", FRONTIER_ROOT / "submissions"]
    paths: set[Path] = set()
    for root in roots:
        if root.is_dir():
            for path in root.rglob("*.csv"):
                if path.is_file():
                    paths.add(path.resolve())
    return sorted(paths, key=lambda path: str(path).casefold())


def _read_submission_if_matching(path: Path, ids: list[str]) -> tuple[np.ndarray | None, list[str] | None]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            header = next(reader, None)
            if header != ["record_id", "threat_rank"]:
                return None, None
            file_ids: list[str] = []
            values: list[float] = []
            for row in reader:
                if len(row) != 2:
                    return None, None
                file_ids.append(row[0])
                values.append(float(row[1]))
    except (OSError, ValueError, StopIteration):
        return None, None
    if file_ids != ids:
        return None, None
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        return None, None
    return array, file_ids


def local_duplicate_scan(ids: np.ndarray, candidates: Mapping[str, np.ndarray]) -> dict[str, Any]:
    matches: dict[str, list[dict[str, Any]]] = {candidate: [] for candidate in candidates}
    scanned = 0
    for path in _iter_submission_csvs():
        loaded, _ = _read_submission_if_matching(path, ids.tolist())
        if loaded is None:
            continue
        scanned += 1
        for name, values in candidates.items():
            if np.array_equal(loaded, values):
                matches[name].append({"path": str(path), "kind": "exact_float_vector"})
            elif np.allclose(loaded, values, rtol=0.0, atol=1e-12):
                matches[name].append({"path": str(path), "kind": "equal_within_1e-12"})
    return {"status": "PASS_LOCAL_DUPLICATE_SCAN", "candidate_matches": matches,
            "same_schema_and_order_csv_vectors_scanned": scanned,
            "output_candidate_paths_excluded": [str(path) for path in CSV_PATHS.values()],
            "test_labels_read": False}


def _verify_source_committed() -> tuple[str, dict[str, str]]:
    source_paths = {
        "runner": Path(__file__).resolve(),
        "verifier": ROOT / "scripts/verify_reg_simplification_benchmark_v1.py",
        "tests": ROOT / "tests/test_reg_simplification_benchmark_v1.py",
        "protocol": REPORT_DIR / "PROTOCOL_R1.md",
        "input_manifest": MANIFEST_PATH,
        "execution_policy": EXECUTION_POLICY_PATH,
    }
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                          check=False, capture_output=True, text=True)
    if head.returncode:
        raise BenchmarkError("Cannot resolve implementation source commit")
    commit = head.stdout.strip()
    hashes: dict[str, str] = {}
    for name, path in source_paths.items():
        if not path.is_file():
            raise BenchmarkError(f"Committed source file is missing: {name}")
        relative = path.resolve().relative_to(ROOT).as_posix()
        result = subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:{relative}"],
                                check=False, capture_output=True)
        if result.returncode:
            raise BenchmarkError(f"Implementation source is not committed at HEAD: {relative}")
        digest = hashlib.sha256(result.stdout).hexdigest()
        if digest != sha_file(path):
            raise BenchmarkError(f"Working source bytes differ from committed bytes: {relative}")
        hashes[name] = digest
    return commit, hashes


def build() -> dict[str, Any]:
    input_hashes = verify_inputs()
    policy = verify_execution_policy()
    source_commit, source_hashes = _verify_source_committed()
    lock_owner, resources = _acquire_resource_lock()
    try:
        arrays, reconstruction = _reconstruct_vectors()
        ids = arrays["record_ids"]
        candidates = {"NO_POS": arrays["no_pos"], "NO_CHAR": arrays["no_char"]}
        duplicate_receipt = local_duplicate_scan(ids, candidates)
        write_once(LOCAL_DUPLICATE_RECEIPT, duplicate_receipt)
        output_hashes = {}
        for name, values in candidates.items():
            output_hashes[name] = {
                "path": str(CSV_PATHS[name].resolve()),
                "sha256": _write_csv_once(CSV_PATHS[name], ids, values),
                "rows": int(len(values)),
                "columns": ["record_id", "threat_rank"],
                "prediction_sha256": array_sha(values, dtype="<f8"),
                "finite_min": float(np.min(values)),
                "finite_max": float(np.max(values)),
            }
        reconstruction["r06_final_source_hash"] = input_hashes["r06_archive"]
        reconstruction["current_inference_source_hash"] = input_hashes["current_r06_archive"]
        reconstruction["source_commit"] = source_commit
        reconstruction["source_hashes"] = source_hashes
        reconstruction["input_hashes"] = input_hashes
        write_once(RECONSTRUCTION_RECEIPT, reconstruction)
        resources["final_available_bytes"] = int(psutil.virtual_memory().available)
        resources["post_run_reserve_floor_bytes"] = RAM_RESERVE_BYTES
        resources["post_run_reserve_pass"] = resources["final_available_bytes"] >= RAM_RESERVE_BYTES
        resources["lock_owner"] = lock_owner
        resource_receipt = {
            "schema_version": 1,
            "status": ("PASS_REG_LIGHT_JOB_RESOURCE_GATE" if resources["post_run_reserve_pass"]
                       else "FAIL_POST_RUN_MEMORY_RESERVE"),
            "execution_policy_sha256": sha_file(EXECUTION_POLICY_PATH),
            "execution_policy": policy,
            "resource": resources,
        }
        write_once(RESOURCE_RECEIPT, resource_receipt)
        if not resources["post_run_reserve_pass"]:
            raise BenchmarkError("Post-build RAM reserve floor breached; do not submit")
        receipt = {
            "schema_version": 1,
            "status": "PASS_REG_SIMPLIFICATION_E79_BUILD",
            "experiment_id": NAMESPACE.upper() + "_R1",
            "candidates": output_hashes,
            "source_commit": source_commit,
            "source_hashes": source_hashes,
            "input_hashes": input_hashes,
            "reconstruction_receipt_sha256": sha_file(RECONSTRUCTION_RECEIPT),
            "local_duplicate_scan_sha256": sha_file(LOCAL_DUPLICATE_RECEIPT),
            "execution_policy_sha256": sha_file(EXECUTION_POLICY_PATH),
            "resource_receipt_sha256": sha_file(RESOURCE_RECEIPT),
            "local_duplicate_matches": duplicate_receipt["candidate_matches"],
            "full_r06_reconstruction_max_abs_error": reconstruction["r06_reconstruction_max_abs_error"],
            "resource": resources,
            "new_predictor_fits": 0,
            "new_preprocessing_fits": 0,
            "new_calibrator_fits": 0,
            "test_features_read": False,
            "test_labels_read": False,
            "kaggle_accessed": False,
        }
        write_once(BUILD_RECEIPT, receipt)
        return receipt
    finally:
        _release_resource_lock()


def _capture_kaggle_history(path: Path) -> tuple[int, str, str]:
    kaggle = shutil.which("kaggle")
    if not kaggle:
        raise BenchmarkError("Kaggle CLI is unavailable")
    command = [kaggle, "competitions", "submissions", COMPETITION,
               "--format", "json", "--page-size", "200", "-q"]
    result = subprocess.run(command,
                            capture_output=True, text=True, check=False, timeout=90)
    history = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "competition": COMPETITION,
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "read_only": True,
    }
    write_once(path, history)
    return result.returncode, result.stdout, result.stderr


def _find_history_row(stdout: str, message: str) -> dict[str, Any] | None:
    try:
        document = json.loads(stdout)
    except ValueError:
        return None
    matches: list[dict[str, Any]] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if any(isinstance(item, str) and item.strip() == message for item in value.values()):
                matches.append(value)
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(document)
    # Rows may also carry the description outside their immediate object; the
    # exact raw-string guard below keeps a malformed/unknown shape fail-closed.
    if len(matches) != 1:
        return None
    return matches[0]


def _row_field(row: Mapping[str, Any], *names: str) -> Any:
    indexed = {str(key).casefold().replace("_", ""): value for key, value in row.items()}
    for name in names:
        key = name.casefold().replace("_", "")
        if key in indexed:
            return indexed[key]
    return None


def _confirm_once(candidate: str) -> dict[str, Any]:
    if candidate not in CSV_PATHS:
        raise BenchmarkError(f"Unknown candidate: {candidate}")
    event_path = UPLOAD_DIR / f"{candidate}_EVENTS.jsonl"
    if not event_path.is_file():
        raise BenchmarkError("Cannot confirm a candidate with no durable upload intent")
    events = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()]
    if (len(events) < 2 or events[0].get("event") != "UPLOAD_INTENT"
            or events[1].get("event") != "UPLOAD_RESPONSE"
            or events[1].get("returncode") != 0
            or any(event.get("event") != "COMPETITION_STATUS_CONFIRMED" for event in events[2:])):
        raise BenchmarkError("Upload event log is not a single CLI-accepted attempt")
    if any(event.get("status") == "COMPLETE_SCORE_CONFIRMED" for event in events[2:]):
        raise BenchmarkError("A COMPLETE score is already confirmed")
    confirm_number = 1 + sum(event.get("event") == "COMPETITION_STATUS_CONFIRMED" for event in events)
    history_path = KAGGLE_HISTORY_DIR / f"POST_UPLOAD_{candidate}_R{confirm_number}.json"
    rc, stdout, stderr = _capture_kaggle_history(history_path)
    if rc != 0:
        raise BenchmarkError(f"Single post-upload status read failed; do not poll: {stderr[:300]}")
    message = UPLOAD_MESSAGES[candidate]
    row = _find_history_row(stdout, message)
    if row is None and message in stdout:
        status = "UNKNOWN_HISTORY_JSON_SHAPE"
        public_score = private_score = submission_ref = None
    elif row is None:
        status = "NOT_YET_VISIBLE"
        public_score = private_score = submission_ref = None
    else:
        raw_status = _row_field(row, "status", "submissionStatus")
        normalized_status = str(raw_status or "").split(".")[-1].strip().upper()
        public_score = _row_field(row, "publicScore", "public_score")
        private_score = _row_field(row, "privateScore", "private_score")
        submission_ref = _row_field(row, "id", "submissionId", "submission_id")
        status = ("COMPLETE_SCORE_CONFIRMED" if normalized_status == "COMPLETE" and public_score is not None
                  else ("VISIBLE_" + normalized_status if normalized_status else "UNKNOWN_STATUS"))
    event = {
        "event": "COMPETITION_STATUS_CONFIRMED",
        "candidate": candidate,
        "status": status,
        "public_score": public_score,
        "private_score": private_score,
        "submission_ref": submission_ref,
        "post_upload_history_sha256": sha_file(history_path),
        "history_row": row,
        "read_count": confirm_number,
        "single_read_no_automatic_polling": True,
    }
    append_upload_event(event_path, event, first=False)
    return event


def _require_ready_for_upload(candidate: str) -> tuple[Path, str]:
    if candidate not in CSV_PATHS:
        raise BenchmarkError(f"Unknown candidate: {candidate}")
    build_receipt = _read_json(BUILD_RECEIPT)
    reconstruction = _read_json(RECONSTRUCTION_RECEIPT)
    duplicate = _read_json(LOCAL_DUPLICATE_RECEIPT)
    if (build_receipt.get("status") != "PASS_REG_SIMPLIFICATION_E79_BUILD"
            or reconstruction.get("status") != "PASS_EXACT_R06_AND_ABLATION_RECONSTRUCTION"
            or duplicate.get("status") != "PASS_LOCAL_DUPLICATE_SCAN"):
        raise BenchmarkError("Required committed build/reconstruction/local duplicate QA is missing")
    local_matches = duplicate.get("candidate_matches", {}).get(candidate, [])
    if local_matches:
        raise BenchmarkError(f"Candidate matches a local prior vector; do not submit: {candidate}")
    path = CSV_PATHS[candidate]
    if not path.is_file():
        raise BenchmarkError(f"Candidate CSV is missing: {path}")
    digest = sha_file(path)
    recorded = build_receipt.get("candidates", {}).get(candidate, {})
    if digest != recorded.get("sha256"):
        raise BenchmarkError("Candidate CSV changed after independent QA")
    return path, digest


def submit_once(candidate: str) -> dict[str, Any]:
    path, digest = _require_ready_for_upload(candidate)
    verifier_receipt = REPORT_DIR / f"INDEPENDENT_VERIFICATION_{candidate}_R1.json"
    if not verifier_receipt.is_file():
        raise BenchmarkError("Independent verification receipt is missing")
    verification = _read_json(verifier_receipt)
    if verification.get("status") != "PASS_INDEPENDENT_REG_SIMPLIFICATION_CSV_QA":
        raise BenchmarkError("Independent verification did not pass")
    intent_path = UPLOAD_DIR / f"{candidate}_EVENTS.jsonl"
    if intent_path.exists():
        raise BenchmarkError(f"A durable upload intent already exists; never retry: {intent_path}")
    history_path = KAGGLE_HISTORY_DIR / f"PRE_UPLOAD_{candidate}.json"
    kaggle = shutil.which("kaggle")
    if not kaggle:
        raise BenchmarkError("Kaggle CLI is unavailable; upload not attempted")
    rc, stdout, stderr = _capture_kaggle_history(history_path)
    if rc != 0:
        raise BenchmarkError(f"Cannot capture pre-upload Kaggle history; upload not attempted: {stderr[:300]}")
    if UPLOAD_MESSAGES[candidate] in stdout:
        raise BenchmarkError("Exact upload message already exists in competition history; do not submit")

    source_commit, source_hashes = _verify_source_committed()
    # Intent is durable before the external side effect. An interrupted command
    # leaves an intent with no response, which is an UNKNOWN consumed attempt.
    append_upload_event(intent_path, {
        "event": "UPLOAD_INTENT",
        "candidate": candidate,
        "competition": COMPETITION,
        "message": UPLOAD_MESSAGES[candidate],
        "csv_path": str(path.resolve()),
        "csv_sha256": digest,
        "independent_verification_sha256": sha_file(verifier_receipt),
        "pre_upload_history_sha256": sha_file(history_path),
        "source_commit": source_commit,
        "source_hashes": source_hashes,
        "attempt_number": 1,
        "do_not_retry": True,
        "test_labels_read": False,
    }, first=True)
    command = [kaggle, "competitions", "submit", COMPETITION,
               "-f", str(path.resolve()), "-m", UPLOAD_MESSAGES[candidate]]
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                check=False, timeout=180)
        out = result.stdout
        err = result.stderr
        code: int | None = result.returncode
        status = "ACCEPTED_BY_CLI" if code == 0 else "CLI_ERROR_ATTEMPT_CONSUMED"
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        err = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        code = None
        status = "UNKNOWN_TIMEOUT_ATTEMPT_CONSUMED"
    submission_ref_match = re.search(r"(?:Submission\s*(?:ID|Id|Ref)\s*[:#]?\s*)(\d+)", out + "\n" + err)
    response = {
        "event": "UPLOAD_RESPONSE",
        "candidate": candidate,
        "status": status,
        "returncode": code,
        "stdout": out,
        "stderr": err,
        "submission_ref": submission_ref_match.group(1) if submission_ref_match else None,
        "attempt_number": 1,
        "retry_authorized": False,
    }
    append_upload_event(intent_path, response, first=False)
    if code != 0:
        raise BenchmarkError(f"Submission attempt consumed with status {status}; do not retry")
    return response


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-pins", action="store_true")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--submit-once", choices=("NO_POS", "NO_CHAR"))
    parser.add_argument("--confirm-once", choices=("NO_POS", "NO_CHAR"))
    parser.add_argument("--kaggle-version", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.kaggle_version:
            kaggle = shutil.which("kaggle")
            if not kaggle:
                raise BenchmarkError("Kaggle CLI is unavailable")
            result = subprocess.run([kaggle, "--version"], capture_output=True, text=True,
                                    check=False, timeout=30)
            sys.stdout.write(result.stdout)
            if result.returncode:
                sys.stderr.write(result.stderr)
            return result.returncode
        if args.check_pins:
            policy = verify_execution_policy()
            print(json.dumps({"status": "PASS_INPUT_PINS_AND_EXECUTION_POLICY",
                              "input_hashes": verify_inputs(),
                              "execution_policy_sha256": sha_file(EXECUTION_POLICY_PATH),
                              "execution_policy_id": policy["policy_id"]}, indent=2))
            return 0
        if args.build:
            print(json.dumps(build(), indent=2))
            return 0
        if args.submit_once:
            print(json.dumps(submit_once(args.submit_once), indent=2))
            return 0
        if args.confirm_once:
            print(json.dumps(_confirm_once(args.confirm_once), indent=2))
            return 0
        parser.error("choose --check-pins, --build, --submit-once, or --kaggle-version")
    except BenchmarkError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
