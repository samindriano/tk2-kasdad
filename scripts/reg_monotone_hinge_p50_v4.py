"""R4 executable, source-pass-gated diagnostics for the fixed p50 hinge lane.

Metadata/freeze commands are outcome-blind. The diagnostic command verifies a
new R4 ROOT_SOURCE_PASS before selected Train targets or an LP start can occur.
The identity-only preflight uses only allowlisted direct prediction metadata and
arrays before that source pass exists.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Mapping
import uuid
import warnings

import numpy as np
import psutil
from scipy import sparse
from scipy.optimize import OptimizeWarning, linprog
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import reg_monotone_hinge_p50_v1 as r1  # noqa: E402
from scripts import reg_monotone_hinge_p50_v3 as predecessor  # noqa: E402
from scripts import reg_final_wave_v2 as v2  # noqa: E402
from scripts.reg_position_failure_mechanism_v1 import read_scored_query  # noqa: E402

EXPERIMENT_ID = "REG_MONOTONE_HINGE_P50_V1_R4"
REPORT_DIR = ROOT / "reports/optimization_v3/reg_monotone_hinge_p50_v1"
EXPERIMENT_DIR = ROOT / "experiments/reg_monotone_hinge_p50_v1"
RUN_DIR = ROOT / "experiments/runs/reg_monotone_hinge_p50_v1/R4"
R1_FREEZE = EXPERIMENT_DIR / "IDENTITY_FREEZE.json"
R1_FREEZE_SHA256 = "dec06ab227539d6f2675bc4f5e0a6994de702ceed38a8124618940aa08509a29"
R3_FREEZE = EXPERIMENT_DIR / "IDENTITY_FREEZE_R3.json"
R3_ROOT_SOURCE_PASS = EXPERIMENT_DIR / "ROOT_SOURCE_PASS_R3.json"
R3_IDENTITY_PREFLIGHT = REPORT_DIR / "IDENTITY_PREFLIGHT_R3.json"
R3_PRESTART_BLOCKER = REPORT_DIR / "PRESTART_RESOURCE_BLOCKED_R3.json"
R3_PRETARGET_FAILURE = REPORT_DIR / "PRETARGET_INTEGRATION_FAILURE_R3.json"
PREFIT_FREEZE = EXPERIMENT_DIR / "IDENTITY_FREEZE_R4.json"
ROOT_SOURCE_PASS = EXPERIMENT_DIR / "ROOT_SOURCE_PASS_R4.json"
IDENTITY_PREFLIGHT = REPORT_DIR / "IDENTITY_PREFLIGHT_R4.json"
SYNTHETIC_QA = REPORT_DIR / "SYNTHETIC_QA_R4.json"
PROTOCOL = REPORT_DIR / "PROTOCOL_R4.md"
RESOURCE_POLICY_FILE = ROOT / "reports/optimization_v3/RESOURCE_POLICY_USER_REVISED_20261105.json"
TESTS = ROOT / "tests/test_reg_monotone_hinge_p50_v4.py"
VERIFIER = ROOT / "scripts/verify_reg_monotone_hinge_p50_v4.py"
SCRIPT = Path(__file__).resolve()
LEDGER = RUN_DIR / "lp_ledger.jsonl"
DIAGNOSTICS = REPORT_DIR / "DIAGNOSTICS_R4.json"
VERIFICATION = REPORT_DIR / "DIAGNOSTICS_VERIFICATION_R4.json"
FIT_IDS = ("HINGE_D1", "HINGE_D2", "HINGE_FINAL")
DIAGNOSTIC_FIT_IDS = FIT_IDS[:2]
R3_FREEZE_SHA256 = "9ddf50d2796b9b19e4e4ca61767dea80cbc81161daa50a8536f3f5356aceedb5"
R3_SOURCE_PASS_SHA256 = "340043c85cec86413f0fe9530f5017ca40a1b0af2008aa22b003bb77d62a9c8e"
R3_PREFLIGHT_SHA256 = "0343e5e91aa934f9bd0270b1ebd110b7b2148a850179d0f8d1b7f01f2e9e8c0f"
R3_PRESTART_BLOCKER_SHA256 = "f26f827555688d9cd0d9fb220d78459d13cb181261e31bd23c6e3d6a7f6dd94b"
R3_PRETARGET_FAILURE_SHA256 = "2308b66db3a6bee5a73af40acbe23fbd55840d8d3051177a11d41e257f000eb2"
PREFLIGHT_INPUT_ALLOWLIST = (
    "r3_identity_freeze", "r3_root_source_pass", "r3_identity_preflight",
    "r3_prestart_resource_blocker", "r3_pretarget_integration_failure",
    "user_revised_resource_policy",
    "r3_input__pinned_direct_E67", "r3_input__pinned_direct_E67_receipt",
    "r3_input__pinned_direct_E75", "r3_input__pinned_direct_E75_receipt",
    "r3_input__pinned_direct_freeze", "r3_input__pinned_direct_prelabel",
    "r3_stage__D1_GLOBAL_state", "r3_stage__D2_GLOBAL_state",
)
MAX_LP_STARTS = 3
KNOT = 50.0
SLOPE_LOWER, SLOPE_UPPER = 0.0, 2.0
DIAGNOSTIC_MIN_ROWS = 1_000
FINAL_MIN_ROWS = 5_000
MIN_SIDE_ROWS = 200
THREADS = 2
TIME_LIMIT_SECONDS = 180
MIN_AVAILABLE_RAM = 2 * 1024**3
ESTIMATED_PEAK_RAM = 512 * 1024**2
PROJECTED_FREE_RAM = 1 * 1024**3
ACTUAL_BOUNDARY_FREE_RAM = 1 * 1024**3
SHARED_JOB_LOCK = Path(tempfile.gettempdir()) / "kasdad_single_heavy_job.lock"
EXPECTED_OOF_ROWS = 20_903
EXPECTED_POOLED_ROWS = 15_953
EXPECTED_TEST_ROWS = 29_421
EXPECTED_MISSING_TEST_ROWS = 1_062
DIRECT_NPZ_ALLOWLIST = ("origin", "arm", "fit_rows", "query_rows", "query_record_ids",
    "query_cycles", "baseline", "raw_auxiliary", "mapped_auxiliary", "prediction",
    "missing_mask")


class GateError(RuntimeError):
    """Fail-closed source, identity, resource, or execution error."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8", errors="surrogatepass")


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def path_identity(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _atomic_bytes_once(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=f".{path.name}.",
                                         suffix=".tmp", dir=path.parent,
                                         delete=False) as stream:
            temp_path = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temp_path, path)
        except FileExistsError as exc:
            raise GateError(f"Immutable output exists; refusing overwrite: {path}") from exc
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass
    return hashlib.sha256(payload).hexdigest()


def write_json_once(path: Path, body: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(body), sort_keys=True, ensure_ascii=False,
                         allow_nan=False, indent=2).encode("utf-8") + b"\n"
    return _atomic_bytes_once(path, payload)


def write_json_idempotent(path: Path, body: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(body), sort_keys=True, ensure_ascii=False,
                         allow_nan=False, indent=2).encode("utf-8") + b"\n"
    if path.exists():
        if path.read_bytes() != payload:
            raise GateError(f"Existing immutable report differs: {path}")
        return hashlib.sha256(payload).hexdigest()
    return _atomic_bytes_once(path, payload)


def _npz_bytes(arrays: Mapping[str, Any]) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **{name: np.asarray(value) for name, value in arrays.items()})
    return buffer.getvalue()


def write_npz_once(path: Path, arrays: Mapping[str, Any]) -> str:
    expected = {name: np.asarray(value) for name, value in arrays.items()}
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != set(expected):
                    raise GateError(f"Existing NPZ schema differs: {path}")
                for name, value in expected.items():
                    stored = np.asarray(archive[name])
                    if stored.shape != value.shape or stored.dtype != value.dtype:
                        raise GateError(f"Existing NPZ dtype/shape differs: {path}:{name}")
                    if not np.array_equal(stored.view(np.uint8), value.view(np.uint8)):
                        raise GateError(f"Existing NPZ bytes differ: {path}:{name}")
        except ValueError as exc:
            raise GateError(f"Existing NPZ cannot be verified: {path}") from exc
        return file_sha256(path)
    return _atomic_bytes_once(path, _npz_bytes(expected))


def _git(*args: str, check: bool = True) -> str:
    result = subprocess.run(["git", "-C", str(ROOT), *args], check=check,
                            capture_output=True, text=True)
    if check:
        return result.stdout.strip()
    return result.stdout.strip()


def _source_paths() -> dict[str, Path]:
    return {
        "runner": SCRIPT,
        "verifier": VERIFIER,
        "tests": TESTS,
        "protocol": PROTOCOL,
        "synthetic_qa": SYNTHETIC_QA,
        "r1_hinge_core": Path(r1.__file__).resolve(),
        "r1_hinge_verifier": Path(r1.__file__).with_name("verify_reg_monotone_hinge_p50_v1.py"),
        "r3_direct_identity_helpers": Path(v2.__file__).resolve(),
        "selected_train_row_reader": Path(read_scored_query.__code__.co_filename).resolve(),
    }


def source_hashes() -> dict[str, str]:
    result = {}
    for name, path in _source_paths().items():
        if not path.is_file():
            raise GateError(f"Required frozen source is missing: {name} ({path})")
        result[name] = file_sha256(path)
    return result


def _assert_committed_sources() -> tuple[str, str, dict[str, str]]:
    branch = _git("branch", "--show-current")
    commit = _git("rev-parse", "HEAD")
    if branch != "codex/reg-position-failure-mechanism-v1":
        raise GateError(f"Unexpected source branch: {branch}")
    hashes = {}
    for name, path in _source_paths().items():
        try:
            relative = path.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            # External source dependencies are pinned by their byte hash.
            hashes[name] = file_sha256(path)
            continue
        committed = subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:{relative}"],
                                   check=True, capture_output=True).stdout
        local = path.read_bytes()
        if local != committed:
            raise GateError(f"R4 source is not byte-identical to committed file: {relative}")
        hashes[name] = hashlib.sha256(committed).hexdigest()
    return branch, commit, hashes


def _r4_inputs_for_freeze(r3_freeze: Mapping[str, Any], r3_pass_sha: str) -> dict[str, dict[str, str]]:
    inputs = {name: dict(record) for name, record in r3_freeze["input_hashes"].items()}
    inputs.update({
        "r3_identity_freeze": {"path": str(R3_FREEZE.resolve()),
                               "sha256": R3_FREEZE_SHA256},
        "r3_root_source_pass": {"path": str(R3_ROOT_SOURCE_PASS.resolve()),
                                "sha256": r3_pass_sha},
        "r3_identity_preflight": {"path": str(R3_IDENTITY_PREFLIGHT.resolve()),
                                  "sha256": R3_PREFLIGHT_SHA256},
        "r3_prestart_resource_blocker": {"path": str(R3_PRESTART_BLOCKER.resolve()),
                                          "sha256": R3_PRESTART_BLOCKER_SHA256},
        "r3_pretarget_integration_failure": {"path": str(R3_PRETARGET_FAILURE.resolve()),
                                             "sha256": R3_PRETARGET_FAILURE_SHA256},
    })
    for name, path in predecessor._source_paths().items():
        inputs[f"r3_source__{name}"] = {"path": str(path.resolve()),
                                        "sha256": file_sha256(path)}
    return inputs


def _validate_synthetic_qa(inputs: Mapping[str, Mapping[str, str]]) -> None:
    qa = json.loads(SYNTHETIC_QA.read_text(encoding="utf-8"))
    qa_sources = qa.get("source_hashes", {})
    for path in (SCRIPT, VERIFIER, TESTS, PROTOCOL):
        relative = path.relative_to(ROOT).as_posix()
        if qa_sources.get(relative) != file_sha256(path):
            raise GateError(f"R4 synthetic QA does not bind current source: {relative}")
    checks = qa.get("checks")
    if (qa.get("status") != "PASS_SYNTHETIC_ONLY"
            or not isinstance(checks, Mapping) or not checks
            or any(not isinstance(check, Mapping) or check.get("pass") is not True
                   for check in checks.values())
            or qa.get("candidate_data_read") is not False
            or qa.get("official_train_labels_read") is not False
            or qa.get("test_or_sample_read") is not False
            or qa.get("official_lp_starts") != 0
            or qa.get("resource_policy_sha256")
                != inputs["user_revised_resource_policy"]["sha256"]):
        raise GateError("R4 synthetic QA status/source boundary is not clean")


def _validate_r3_predecessor_artifacts() -> dict[str, Any]:
    pinned = ((R3_FREEZE, R3_FREEZE_SHA256, "identity freeze"),
              (R3_ROOT_SOURCE_PASS, R3_SOURCE_PASS_SHA256, "root source pass"),
              (R3_IDENTITY_PREFLIGHT, R3_PREFLIGHT_SHA256, "identity preflight"),
              (R3_PRESTART_BLOCKER, R3_PRESTART_BLOCKER_SHA256, "prestart resource blocker"),
              (R3_PRETARGET_FAILURE, R3_PRETARGET_FAILURE_SHA256, "pretarget integration failure"))
    for path, expected, label in pinned:
        if not path.is_file() or file_sha256(path) != expected:
            raise GateError(f"Sealed R3 {label} is missing or changed")
    blocker = json.loads(R3_PRESTART_BLOCKER.read_text(encoding="utf-8"))
    blocker_boundary = blocker.get("work_boundary", {})
    blocker_gate = blocker.get("resource_gate", {})
    if (blocker.get("status") != "PRESTART_RESOURCE_BLOCKED"
            or blocker.get("classification") != "PRESTART_RESOURCE_BLOCKED_NOT_FIT_OR_SCIENTIFIC_FAILURE"
            or blocker_gate.get("gate_passed") is not False
            or blocker_boundary.get("diagnostics_command_invoked") is not False
            or blocker_boundary.get("official_lp_starts") != 0
            or blocker_boundary.get("official_train_labels_parsed") is not False
            or blocker_boundary.get("target_loader_called") is not False
            or blocker_boundary.get("ledger_exists") is not False):
        raise GateError("R3 prestart resource blocker does not preserve its zero-start boundary")
    failure = json.loads(R3_PRETARGET_FAILURE.read_text(encoding="utf-8"))
    attempt = failure.get("failure", {})
    boundary = failure.get("work_boundary", {})
    post_failure = failure.get("post_failure_state", {})
    if (failure.get("status") != "PRETARGET_TECHNICAL_BLOCKER"
            or failure.get("classification") != "INTEGRATION_SCHEMA_FAILURE_BEFORE_LABEL_READ_OR_LP_START"
            or attempt.get("exception_type") != "KeyError"
            or attempt.get("message") != "'physical_rows'"
            or attempt.get("retry_attempted") is not False
            or attempt.get("source_fix_applied") is not False
            or boundary.get("official_lp_starts") != 0
            or boundary.get("official_train_labels_parsed") is not False
            or boundary.get("selected_train_row_reader_called") is not False
            or boundary.get("target_arrays_loaded") is not False
            or boundary.get("final_fit_started") is not False
            or post_failure.get("ledger_lp_started_count") != 0):
        raise GateError("R3 failed attempt is not the sealed pretarget zero-start boundary")
    for path in (predecessor.LEDGER, predecessor.DIAGNOSTICS, predecessor.VERIFICATION,
                 predecessor.RUN_DIR / "RESOURCE_SNAPSHOTS_R3.json"):
        if path.exists():
            raise GateError(f"Unexpected R3 diagnostic output exists after preserved failure: {path}")
    return failure


def freeze_identity_r4() -> tuple[Path, str]:
    if PREFIT_FREEZE.exists():
        raise GateError(f"R4 identity freeze already exists: {PREFIT_FREEZE}")
    failure = _validate_r3_predecessor_artifacts()
    r3_freeze, r3_approval, r3_pass_sha = predecessor.verify_source_pass()
    if (r3_pass_sha != R3_SOURCE_PASS_SHA256
            or r3_freeze.get("official_lp_starts") != 0
            or r3_freeze.get("official_labels_accessed") is not False
            or r3_approval.get("authorized_fit_ids") != list(DIAGNOSTIC_FIT_IDS)):
        raise GateError("Sealed R3 predecessor is not the approved zero-start source freeze")
    inputs = _r4_inputs_for_freeze(r3_freeze, r3_pass_sha)
    if set(PREFLIGHT_INPUT_ALLOWLIST) - set(inputs):
        raise GateError("R4 identity preflight input allowlist is not fully pinned")
    for name, item in inputs.items():
        path = Path(item["path"])
        if not path.is_file() or file_sha256(path) != item["sha256"]:
            raise GateError(f"R4 frozen input changed: {name}")
    branch, commit, sources = _assert_committed_sources()
    _validate_synthetic_qa(inputs)
    support = dict(r3_freeze["support_identity"])
    stage_support = dict(r3_freeze["stage_support"])
    if (support.get("row_counts", {}).get("unique_oof") != EXPECTED_OOF_ROWS
            or stage_support != {
                "D1": {"fit_cycles": [68, 71], "fit_rows": 4_950,
                       "score_cycles": [72, 75], "score_rows": 7_462},
                "D2": {"fit_cycles": [68, 75], "fit_rows": 12_412,
                       "score_cycles": [76, 79], "score_rows": 8_491}}):
        raise GateError("Sealed R3 support identity/stage boundaries differ from R4 protocol")
    body = {
        "schema_version": 4,
        "experiment_id": EXPERIMENT_ID,
        "status": "PREFIT_IDENTITY_FROZEN_PENDING_IDENTITY_PREFLIGHT",
        "branch": branch,
        "source_commit": commit,
        "source_hashes": sources,
        "input_hashes": inputs,
        "r3_predecessor": {"freeze_path": path_identity(R3_FREEZE),
            "freeze_sha256": R3_FREEZE_SHA256,
            "source_pass_path": path_identity(R3_ROOT_SOURCE_PASS),
            "source_pass_sha256": r3_pass_sha,
            "identity_preflight_path": path_identity(R3_IDENTITY_PREFLIGHT),
            "identity_preflight_sha256": R3_PREFLIGHT_SHA256,
            "prestart_blocker_path": path_identity(R3_PRESTART_BLOCKER),
            "prestart_blocker_sha256": R3_PRESTART_BLOCKER_SHA256,
            "pretarget_failure_path": path_identity(R3_PRETARGET_FAILURE),
            "pretarget_failure_sha256": R3_PRETARGET_FAILURE_SHA256,
            "failed_attempt": dict(failure)},
        "support_identity": support,
        "stage_support": stage_support,
        "fit_ids": list(FIT_IDS),
        "diagnostic_fit_ids": list(DIAGNOSTIC_FIT_IDS),
        "official_lp_start_cap": MAX_LP_STARTS,
        "official_lp_starts": 0,
        "official_labels_accessed": False,
        "identity_preflight_completed": False,
        "test_or_sample_accessed": False,
        "csv_created": False,
        "kaggle_upload": False,
        "csv_schema": ["record_id", "threat_rank"],
        "historical_r3_source_pass_authorizes_r4": False,
        "identity_preflight_path": path_identity(IDENTITY_PREFLIGHT),
        "preflight_input_allowlist": list(PREFLIGHT_INPUT_ALLOWLIST),
        "resource_contract": {"shared_lock": str(SHARED_JOB_LOCK),
            "policy_path": path_identity(RESOURCE_POLICY_FILE),
            "policy_sha256": inputs["user_revised_resource_policy"]["sha256"],
            "minimum_available_ram_bytes": MIN_AVAILABLE_RAM,
            "incremental_estimate_bytes": ESTIMATED_PEAK_RAM,
            "minimum_projected_free_bytes": PROJECTED_FREE_RAM,
            "minimum_actual_free_at_boundary_bytes": ACTUAL_BOUNDARY_FREE_RAM,
            "estimated_peak_is_measured": False, "threads": THREADS,
            "time_limit_seconds_per_start": TIME_LIMIT_SECONDS,
            "maximum_diagnostic_starts": 2,
            "maximum_total_starts_including_conditional_final": MAX_LP_STARTS,
            "actual_official_compute_cost": "UNKNOWN_PENDING_R4_ROOT_SOURCE_PASS"},
    }
    return PREFIT_FREEZE, write_json_once(PREFIT_FREEZE, body)


def verify_identity_freeze_r4(*, preflight_only: bool = False) -> tuple[dict[str, Any], str]:
    if not PREFIT_FREEZE.is_file():
        raise GateError("R4 identity freeze is absent; no preflight/labels/LP starts allowed")
    freeze_sha = file_sha256(PREFIT_FREEZE)
    freeze = json.loads(PREFIT_FREEZE.read_text(encoding="utf-8"))
    if (freeze.get("status") != "PREFIT_IDENTITY_FROZEN_PENDING_IDENTITY_PREFLIGHT"
            or freeze.get("experiment_id") != EXPERIMENT_ID
            or freeze.get("official_lp_starts") != 0
            or freeze.get("identity_preflight_completed") is not False
            or not _source_ancestor(str(freeze.get("source_commit", "")))):
        raise GateError("R4 identity freeze status or source ancestry failed")
    if source_hashes() != freeze.get("source_hashes"):
        raise GateError("R4 committed source bytes differ from identity freeze")
    names = PREFLIGHT_INPUT_ALLOWLIST if preflight_only else tuple(freeze.get("input_hashes", {}))
    for name in names:
        item = freeze.get("input_hashes", {}).get(name)
        if not item:
            raise GateError(f"R4 freeze omits required input: {name}")
        path = Path(item["path"])
        if not path.is_file() or file_sha256(path) != item["sha256"]:
            raise GateError(f"R4 frozen input bytes changed: {name}")
    return freeze, freeze_sha


def _source_ancestor(source_commit: str) -> bool:
    result = subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor",
                             source_commit, "HEAD"], capture_output=True)
    return result.returncode == 0


def validate_source_pass_body(approval: Mapping[str, Any], freeze: Mapping[str, Any],
                              freeze_sha256: str, preflight_sha256: str) -> None:
    body = {key: value for key, value in approval.items() if key != "signature"}
    expected_signature = canonical_sha(body)
    if approval.get("signature") != expected_signature:
        raise GateError("R4 ROOT_SOURCE_PASS signature failed")
    if (approval.get("verdict") != "PASS"
            or approval.get("experiment_id") != EXPERIMENT_ID
            or approval.get("source_commit") != freeze.get("source_commit")
            or approval.get("prefit_freeze_sha256") != freeze_sha256
            or approval.get("identity_preflight_sha256") != preflight_sha256
            or approval.get("identity_preflight_status") != "PASS_IDENTITY_ONLY_PREFLIGHT"
            or approval.get("source_hashes") != freeze.get("source_hashes")
            or approval.get("input_hashes") != freeze.get("input_hashes")
            or approval.get("authorized_fit_ids") != list(DIAGNOSTIC_FIT_IDS)
            or approval.get("official_lp_start_cap") != MAX_LP_STARTS):
        raise GateError("ROOT_SOURCE_PASS does not bind exact R4 source/preflight/input/fit scope")


def _read_identity_preflight(freeze: Mapping[str, Any], freeze_sha: str) -> tuple[dict[str, Any], str]:
    if not IDENTITY_PREFLIGHT.is_file():
        raise GateError("R4 identity-only preflight is absent; no labels or LP starts allowed")
    report_sha = file_sha256(IDENTITY_PREFLIGHT)
    report = json.loads(IDENTITY_PREFLIGHT.read_text(encoding="utf-8"))
    if (report.get("status") != "PASS_IDENTITY_ONLY_PREFLIGHT"
            or report.get("experiment_id") != EXPERIMENT_ID
            or report.get("prefit_freeze_sha256") != freeze_sha
            or report.get("source_hashes") != freeze.get("source_hashes")
            or report.get("input_hashes") != freeze.get("input_hashes")
            or report.get("official_lp_starts") != 0
            or report.get("target_loader_called") is not False
            or report.get("identity_validation") != {
                "function": "verify_oof_identity",
                "passed": True,
                "row_counts_recomputed": True}
            or report.get("train_targets_parsed") is not False
            or report.get("cached_r4_target_arrays_loaded") is not False
            or report.get("loaded_target_or_label_arrays") is not False
            or report.get("scoring_started") is not False
            or report.get("ledger_created") is not False
            or report.get("test_or_sample_accessed") is not False):
        raise GateError("R4 identity preflight status/boundary failed")
    if set(report.get("direct_bundle_audits", {})) != {"E67", "E75"}:
        raise GateError("R4 identity preflight must bind exactly E67 and E75")
    for origin, name in ((67, "E67"), (75, "E75")):
        audit = report["direct_bundle_audits"][name]
        archive_pin = freeze["input_hashes"].get(f"r3_input__pinned_direct_E{origin}")
        receipt_pin = freeze["input_hashes"].get(f"r3_input__pinned_direct_E{origin}_receipt")
        if (audit.get("receipt_signature_verified") is not True
                or audit.get("allowlisted_loaded_npz_members") != list(DIRECT_NPZ_ALLOWLIST)
                or audit.get("target_member_loaded") is not False
                or archive_pin is None or audit.get("archive_sha256") != archive_pin.get("sha256")
                or receipt_pin is None or audit.get("receipt_sha256") != receipt_pin.get("sha256")
                or set(audit.get("producer_array_hashes", {})) != {
                    "query_cycles_sha256", "query_ids_sha256", "raw_auxiliary_sha256",
                    "mapped_auxiliary_sha256", "prediction_sha256"}):
            raise GateError(f"R4 direct bundle audit is incomplete: {name}")
    if set(report.get("causal_global_states", {})) != {"D1_GLOBAL", "D2_GLOBAL"}:
        raise GateError("R4 identity preflight must verify both saved GLOBAL controls")
    for stage in ("D1", "D2"):
        pin = freeze["input_hashes"].get(f"r3_stage__{stage}_GLOBAL_state")
        state = report["causal_global_states"].get(f"{stage}_GLOBAL", {})
        intercept, slope, digest = _load_causal_global(stage, freeze)
        if (pin is None or state != {"state_sha256": digest,
                "intercept": intercept, "slope": slope}
                or digest != pin.get("sha256")):
            raise GateError(f"R4 identity preflight does not bind {stage}_GLOBAL state")
    support = freeze.get("support_identity", {})
    expected_row_hashes = {key: support[key] for key in (
        "rows_sha256", "ids_sha256", "origins_sha256",
        "original_bundle_indices_sha256", "cycles_sha256",
        "direct_prediction_sha256", "missing_mask_sha256")}
    if (report.get("row_identity_hashes") != expected_row_hashes
            or report.get("row_counts") != support.get("row_counts")):
        raise GateError("R4 identity preflight row fingerprints differ from frozen support")
    snapshots = report.get("resource_snapshots", [])
    if ([item.get("boundary") for item in snapshots] != [
            "identity_preflight_locked_initial", "identity_preflight_after_direct_identity",
            "identity_preflight_after_global_controls"]):
        raise GateError("R4 identity preflight RAM boundary sequence differs")
    for index, snapshot in enumerate(snapshots):
        available = snapshot.get("available_bytes_measured")
        projected = snapshot.get("projected_free_bytes")
        initial = index == 0
        if (snapshot.get("status") != "PASS"
                or not isinstance(available, int) or not isinstance(projected, int)
                or available < ACTUAL_BOUNDARY_FREE_RAM
                or projected != available - ESTIMATED_PEAK_RAM
                or snapshot.get("actual_boundary_pass") is not True
                or snapshot.get("incremental_estimate_bytes") != ESTIMATED_PEAK_RAM
                or snapshot.get("minimum_projected_free_bytes") != PROJECTED_FREE_RAM
                or snapshot.get("actual_available_boundary_minimum_bytes")
                    != ACTUAL_BOUNDARY_FREE_RAM
                or snapshot.get("estimated_peak_is_measured") is not False
                or snapshot.get("process_rss_measured") is not True
                or not isinstance(snapshot.get("process_rss_bytes"), int)
                or snapshot.get("process_rss_bytes", 0) <= 0
                or snapshot.get("threads") != THREADS
                or snapshot.get("concurrent_heavy_jobs") != 1
                or (initial and (available < MIN_AVAILABLE_RAM
                    or projected < PROJECTED_FREE_RAM
                    or snapshot.get("minimum_initial_available_bytes") != MIN_AVAILABLE_RAM
                    or snapshot.get("initial_floor_pass") is not True
                    or snapshot.get("projected_free_pass") is not True))):
            raise GateError(f"R4 identity preflight resource gate failed at index {index}")
        peak = snapshot.get("os_peak_working_set_bytes")
        if ((peak is None and snapshot.get("os_peak_working_set_measured") is not False)
                or (peak is not None and (not isinstance(peak, int) or peak <= 0
                    or snapshot.get("os_peak_working_set_measured") is not True))
                or snapshot.get("os_peak_working_set_scope") != (
                    "unavailable" if peak is None else "process_lifetime")):
            raise GateError(f"R4 identity preflight peak-working-set flags failed at index {index}")
    return report, report_sha


def verify_source_pass() -> tuple[dict[str, Any], dict[str, Any], str]:
    freeze, freeze_sha = verify_identity_freeze_r4(preflight_only=False)
    for name, item in freeze.get("input_hashes", {}).items():
        path = Path(item["path"])
        if not path.is_file() or file_sha256(path) != item["sha256"]:
            raise GateError(f"R4 frozen input bytes changed: {name}")
    _preflight, preflight_sha = _read_identity_preflight(freeze, freeze_sha)
    if not ROOT_SOURCE_PASS.is_file():
        raise GateError("R4 ROOT_SOURCE_PASS is absent; no labels or LP starts allowed")
    approval = json.loads(ROOT_SOURCE_PASS.read_text(encoding="utf-8"))
    validate_source_pass_body(approval, freeze, freeze_sha, preflight_sha)
    return freeze, approval, file_sha256(ROOT_SOURCE_PASS)


def identity_preflight() -> dict[str, Any]:
    freeze, freeze_sha = verify_identity_freeze_r4(preflight_only=True)
    if ROOT_SOURCE_PASS.exists():
        raise GateError("R4 source pass already exists; identity preflight is a pre-pass stage")
    with SharedHeavyJobLock():
        resources = [resource_snapshot(boundary="identity_preflight_locked_initial", initial=True)]
        v2.one_job_gate()
        identity = load_oof_identity(freeze)
        verify_oof_identity(identity, freeze["support_identity"])
        resources.append(resource_snapshot(boundary="identity_preflight_after_direct_identity"))
        global_states = {}
        for stage in ("D1", "D2"):
            intercept, slope, digest = _load_causal_global(stage, freeze)
            global_states[f"{stage}_GLOBAL"] = {
                "state_sha256": digest, "intercept": intercept, "slope": slope}
        resources.append(resource_snapshot(boundary="identity_preflight_after_global_controls"))
        if source_hashes() != freeze.get("source_hashes"):
            raise GateError("R4 source bytes changed during identity preflight")
        for name in PREFLIGHT_INPUT_ALLOWLIST:
            item = freeze["input_hashes"][name]
            if file_sha256(Path(item["path"])) != item["sha256"]:
                raise GateError(f"R4 preflight input changed during identity read: {name}")
        body = _build_identity_preflight_body(freeze, freeze_sha, identity,
                                              global_states, resources)
        body["identity_validation"] = {
            "function": "verify_oof_identity",
            "passed": True,
            "row_counts_recomputed": True,
        }
        write_json_once(IDENTITY_PREFLIGHT, body)
        return {"status": body["status"],
            "identity_preflight_sha256": file_sha256(IDENTITY_PREFLIGHT),
            "row_counts": body["row_counts"],
            "loaded_target_or_label_arrays": False,
            "official_lp_starts": 0}


def _build_identity_preflight_body(freeze: Mapping[str, Any], freeze_sha: str,
                                   identity: Mapping[str, Any],
                                   global_states: Mapping[str, Any],
                                   resources: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS_IDENTITY_ONLY_PREFLIGHT",
        "source_commit": freeze["source_commit"],
        "source_hashes": freeze["source_hashes"],
        "input_hashes": freeze["input_hashes"],
        "prefit_freeze_sha256": freeze_sha,
        "r3_predecessor": freeze["r3_predecessor"],
        "direct_bundle_audits": identity["direct_bundle_audits"],
        "row_identity_hashes": identity["row_identity_hashes"],
        "row_counts": {"unique_oof": len(identity["rows"]),
            "E67_68_75": int(np.sum((identity["origins"] == 67)
                                     & (identity["cycles"] >= 68)
                                     & (identity["cycles"] <= 75))),
            "E75_76_79": int(np.sum((identity["origins"] == 75)
                                     & (identity["cycles"] >= 76)
                                     & (identity["cycles"] <= 79)))},
        "causal_global_states": dict(global_states),
        "resource_snapshots": [dict(item) for item in resources],
        "loaded_target_or_label_arrays": False,
        "target_loader_called": False,
        "train_targets_parsed": False,
        "cached_r4_target_arrays_loaded": False,
        "scoring_started": False,
        "ledger_created": False,
        "official_lp_starts": 0,
        "test_or_sample_accessed": False,
        "csv_created": False,
        "kaggle_upload": False,
    }


class SharedHeavyJobLock:
    """Nonblocking OS-level lock shared by KASDAD heavy jobs."""

    def __init__(self, path: Path = SHARED_JOB_LOCK):
        self.path = path
        self.stream = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+b")
        if self.path.stat().st_size == 0:
            self.stream.write(b"\0")
            self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.stream.close()
            self.stream = None
            raise GateError(f"Another shared KASDAD heavy job holds {self.path}") from exc
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.stream is not None:
            try:
                self.stream.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
            finally:
                self.stream.close()
                self.stream = None


def resource_decision(available_bytes: int, *, initial: bool = False) -> dict[str, Any]:
    available = int(available_bytes)
    projected = available - ESTIMATED_PEAK_RAM
    initial_ok = available >= MIN_AVAILABLE_RAM and projected >= PROJECTED_FREE_RAM
    boundary_ok = available >= ACTUAL_BOUNDARY_FREE_RAM
    return {"initial_floor_pass": initial_ok,
            "projected_free_bytes": projected,
            "projected_free_pass": projected >= PROJECTED_FREE_RAM,
            "actual_boundary_pass": boundary_ok,
            "pass": (initial_ok and boundary_ok) if initial else boundary_ok}


def resource_snapshot(*, boundary: str, initial: bool = False) -> dict[str, Any]:
    available = int(psutil.virtual_memory().available)
    decision = resource_decision(available, initial=initial)
    if not decision["pass"]:
        raise GateError(f"RAM gate failed at {boundary}: available={available}, "
                        f"estimate={ESTIMATED_PEAK_RAM}, projected={decision['projected_free_bytes']}")
    memory = psutil.Process(os.getpid()).memory_info()
    peak = getattr(memory, "peak_wset", None)
    return {"status": "PASS", "boundary": boundary,
            "available_bytes_measured": available,
            "initial_floor_pass": decision["initial_floor_pass"],
            "minimum_initial_available_bytes": MIN_AVAILABLE_RAM,
            "incremental_estimate_bytes": ESTIMATED_PEAK_RAM,
            "minimum_projected_free_bytes": PROJECTED_FREE_RAM,
            "actual_available_boundary_minimum_bytes": ACTUAL_BOUNDARY_FREE_RAM,
            "projected_free_bytes": decision["projected_free_bytes"],
            "projected_free_pass": decision["projected_free_pass"],
            "actual_boundary_pass": decision["actual_boundary_pass"],
            "estimated_peak_is_measured": False,
            "process_rss_bytes": int(memory.rss), "process_rss_measured": True,
            "os_peak_working_set_bytes": None if peak is None else int(peak),
            "os_peak_working_set_measured": peak is not None,
            "os_peak_working_set_scope": "process_lifetime" if peak is not None else "unavailable",
            "threads": THREADS, "concurrent_heavy_jobs": 1}


def append_ledger(path: Path, event: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(dict(event), sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(",", ":")).encode("utf-8") + b"\n"
    with path.open("ab") as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())


def read_ledger(path: Path | None = None) -> dict[str, dict[str, list[dict[str, Any]]]]:
    path = LEDGER if path is None else Path(path)
    by_fit: dict[str, dict[str, list[dict[str, Any]]]] = {}
    order: dict[str, list[str]] = {}
    if not path.exists():
        return by_fit
    try:
        lines = path.read_bytes().splitlines()
        for raw in lines:
            event = json.loads(raw)
            fit_id, kind = event["fit_id"], event["event"]
            if fit_id not in FIT_IDS or kind not in {"RESERVE", "LP_STARTED", "LP_COMPLETE", "LP_FAILED"}:
                raise GateError("Ledger contains an unknown fit ID/event")
            bucket = by_fit.setdefault(fit_id, {name: [] for name in
                ("RESERVE", "LP_STARTED", "LP_COMPLETE", "LP_FAILED")})
            bucket[kind].append(event)
            order.setdefault(fit_id, []).append(kind)
    except (ValueError, KeyError, TypeError) as exc:
        raise GateError("Ledger contains malformed or partial metadata; no retry") from exc
    starts = 0
    reserved_ids = []
    for fit_id in FIT_IDS:
        bucket = by_fit.get(fit_id)
        if bucket is None:
            continue
        if any(len(bucket[kind]) > 1 for kind in bucket):
            raise GateError(f"Duplicate ledger event for {fit_id}; no retry")
        sequence = order.get(fit_id, [])
        if sequence not in (["RESERVE"], ["RESERVE", "LP_STARTED"],
                            ["RESERVE", "LP_STARTED", "LP_COMPLETE"],
                            ["RESERVE", "LP_STARTED", "LP_FAILED"]):
            raise GateError(f"Ledger event sequence invalid for {fit_id}; no retry")
        if bucket["RESERVE"]:
            reserved_ids.append(fit_id)
        if bucket["LP_STARTED"]:
            starts += 1
        if bucket["LP_COMPLETE"] and bucket["LP_FAILED"]:
            raise GateError(f"Conflicting terminal ledger events for {fit_id}")
        if (bucket["LP_STARTED"] and not bucket["RESERVE"]
                or (bucket["LP_COMPLETE"] or bucket["LP_FAILED"]) and not bucket["LP_STARTED"]):
            raise GateError(f"Invalid ledger event order for {fit_id}")
        if bucket["RESERVE"] and bucket["LP_STARTED"]:
            reserve, started = bucket["RESERVE"][0], bucket["LP_STARTED"][0]
            if any(reserve.get(key) != started.get(key) for key in
                   ("reservation_id", "fit_input", "source_commit", "root_source_pass_sha256")):
                raise GateError(f"Reservation/start binding differs for {fit_id}")
        if bucket["RESERVE"]:
            reserve = bucket["RESERVE"][0]
            for kind in ("LP_COMPLETE", "LP_FAILED"):
                if bucket[kind] and any(bucket[kind][0].get(key) != reserve.get(key) for key in
                    ("reservation_id", "fit_input", "source_commit", "root_source_pass_sha256")):
                    raise GateError(f"Terminal event binding differs for {fit_id}")
    if starts > MAX_LP_STARTS or reserved_ids != list(FIT_IDS[:len(reserved_ids)]):
        raise GateError("Ledger exceeds cap3 or fit IDs are not a contiguous prefix")
    return by_fit


def _next_fit_id(ledger: Mapping[str, Any]) -> str | None:
    reserved = [fit_id for fit_id in FIT_IDS if fit_id in ledger and ledger[fit_id]["RESERVE"]]
    if reserved != list(FIT_IDS[:len(reserved)]):
        raise GateError("Reserved fit IDs are not a contiguous prefix")
    return FIT_IDS[len(reserved)] if len(reserved) < MAX_LP_STARTS else None


def _validate_oof_support(identity: Mapping[str, Any]) -> None:
    cycles = np.asarray(identity["cycles"], dtype=np.int64)
    origins = np.asarray(identity["origins"], dtype=np.int64)
    direct = np.asarray(identity["direct"], dtype=np.float64)
    if not (cycles.ndim == origins.ndim == direct.ndim == 1
            and len(cycles) == len(origins) == len(direct) == EXPECTED_OOF_ROWS):
        raise GateError("R4 stitched OOF vectors do not match 20,903 rows")
    stages = (("HINGE_D1", (origins == 67) & (cycles >= 68) & (cycles <= 71), 4_950),
              ("HINGE_D2", (origins == 67) & (cycles >= 68) & (cycles <= 75), 12_412))
    for fit_id, mask, expected_rows in stages:
        side = (int(np.sum(mask & (direct < KNOT))), int(np.sum(mask & (direct >= KNOT))))
        if int(mask.sum()) != expected_rows:
            raise GateError(f"Unsupported {fit_id}: row count differs from frozen stage")
        if min(side) < MIN_SIDE_ROWS or expected_rows < DIAGNOSTIC_MIN_ROWS:
            raise GateError(f"Unsupported {fit_id}: side support {side}; LP not reserved")


def _load_direct_bundle(origin: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    fit_id = f"REG_POSITION_DIRECT_L1_V1_E{origin}_REAL"
    archive_path, expected_archive_sha = v2.PINS[f"direct_E{origin}"]
    receipt_path, expected_receipt_sha = v2.PINS[f"direct_E{origin}_receipt"]
    if (file_sha256(archive_path) != expected_archive_sha
            or file_sha256(receipt_path) != expected_receipt_sha):
        raise GateError(f"Pinned E{origin} direct artifact bytes changed")
    receipt = v2.read_json(receipt_path)
    v2.verify_signed(receipt, "receipt_sha256", f"{fit_id} receipt")
    if (receipt.get("fit_id") != fit_id or receipt.get("query_targets_read") is not False
            or receipt.get("bundle_sha256") != expected_archive_sha):
        raise GateError(f"E{origin} direct receipt is not the frozen label-blind receipt")
    selected = DIRECT_NPZ_ALLOWLIST
    with np.load(archive_path, allow_pickle=False) as archive:
        members = set(archive.files)
        extras = members - set(selected)
        forbidden = sorted(name for name in extras
            if any(token in name.casefold() for token in ("target", "label", "outcome", "score")))
        if not set(selected).issubset(members) or forbidden:
            raise GateError(f"E{origin} direct bundle lacks allowlisted fields or exposes target-like members")
        arrays = {name: np.asarray(archive[name]).copy() for name in selected}
    if (arrays["origin"].ndim != 1 or not len(arrays["origin"])
            or int(arrays["origin"][0]) != origin or str(arrays["arm"][0]) != "REAL"):
        raise GateError(f"E{origin} direct bundle origin/arm differs")
    n = len(arrays["query_rows"])
    for name in selected[3:]:
        if arrays[name].ndim != 1 or len(arrays[name]) != n:
            raise GateError(f"E{origin} selected row field is misaligned: {name}")
    checks = {
        "query_cycles_sha256": v2.sha_pilot_array(arrays["query_cycles"]),
        "query_ids_sha256": v2.sha_pilot_array(arrays["query_record_ids"]),
        "raw_auxiliary_sha256": v2.sha_pilot_array(arrays["raw_auxiliary"], dtype="<f8"),
        "mapped_auxiliary_sha256": v2.sha_pilot_array(arrays["mapped_auxiliary"], dtype="<f8"),
        "prediction_sha256": v2.sha_pilot_array(arrays["prediction"], dtype="<f8"),
    }
    if any(receipt.get(key) != value for key, value in checks.items()):
        raise GateError(f"E{origin} selected direct bundle arrays disagree with receipt")
    prelabel = v2.read_json(v2.DIRECT_PRELABEL)
    frozen = prelabel.get("predictions_verified", {}).get(str(origin), {}).get("REAL", {})
    if (frozen.get("bundle_sha256") != expected_archive_sha
            or frozen.get("receipt_sha256") != expected_receipt_sha
            or frozen.get("missing_route_exact") is not True
            or frozen.get("n_query") != n):
        raise GateError(f"E{origin} independent prelabel receipt differs")
    recomputed = np.clip(0.75 * arrays["baseline"]
                         + 0.25 * arrays["mapped_auxiliary"], 0.0, 100.0)
    missing = np.asarray(arrays["missing_mask"], dtype=bool)
    recomputed[missing] = arrays["baseline"][missing]
    if not np.allclose(recomputed, arrays["prediction"], rtol=0.0, atol=1e-10):
        raise GateError(f"E{origin} raw DIRECT saved prediction does not replay")
    arrays["missing_mask"] = missing
    audit = {"fit_id": fit_id,
        "archive_path": path_identity(archive_path),
        "archive_sha256": expected_archive_sha,
        "receipt_path": path_identity(receipt_path),
        "receipt_sha256": expected_receipt_sha,
        "receipt_signature_verified": True,
        "producer_array_hashes": checks,
        "allowlisted_loaded_npz_members": list(selected),
        "unloaded_extra_npz_member_names": sorted(extras),
        "query_rows": int(n),
        "target_member_loaded": False}
    return arrays, audit


def load_oof_identity(freeze: Mapping[str, Any]) -> dict[str, Any]:
    """Selective direct-array reader; never opens old cached target arrays."""
    e67, audit67 = _load_direct_bundle(67)
    e75, audit75 = _load_direct_bundle(75)
    direct_freeze = v2.read_json(v2.DIRECT_FREEZE)
    v2.verify_prefixes(e67, e75, direct_freeze)
    plan = v2.stitch_oof_rows(e67, e75)
    rows = np.asarray([item[0] for item in plan], dtype=np.int64)
    origins = np.asarray([item[1] for item in plan], dtype=np.int8)
    indices = np.asarray([item[2] for item in plan], dtype=np.int64)
    ids = np.asarray([item[3] for item in plan], dtype=str)
    cycles = np.asarray([item[4] for item in plan], dtype=np.int16)
    direct = np.empty(len(plan), dtype=np.float64)
    missing = np.empty(len(plan), dtype=bool)
    for origin, bundle in ((67, e67), (75, e75)):
        selected = origins == origin
        bundle_indices = indices[selected]
        direct[selected] = np.asarray(bundle["prediction"][bundle_indices], dtype=np.float64)
        missing[selected] = bundle["missing_mask"][bundle_indices]
    actual = {
        "rows_sha256": r1.tagged_array_sha256(rows),
        "ids_sha256": r1.tagged_array_sha256(ids),
        "origins_sha256": r1.tagged_array_sha256(origins),
        "original_bundle_indices_sha256": r1.tagged_array_sha256(indices),
        "cycles_sha256": r1.tagged_array_sha256(cycles),
        "direct_prediction_sha256": r1.tagged_array_sha256(direct, dtype="<f8"),
        "missing_mask_sha256": r1.tagged_array_sha256(missing),
        "row_counts": {"unique_oof": int(len(rows)),
                       "E67_68_75": int(np.sum((origins == 67) & (cycles >= 68) & (cycles <= 75))),
                       "E75_76_79": int(np.sum((origins == 75) & (cycles >= 76) & (cycles <= 79)))},
    }
    if (len(rows) != EXPECTED_OOF_ROWS or len(np.unique(rows)) != EXPECTED_OOF_ROWS
            or len(np.unique(ids)) != EXPECTED_OOF_ROWS
            or not np.array_equal(np.unique(cycles), np.arange(68, 80, dtype=np.int16))
            or not np.all(np.isfinite(direct)) or np.any((direct < 0.0) | (direct > 100.0))):
        raise GateError("R4 stitched OOF rows/IDs/cycles/direct range failed")
    out = {"rows": rows, "origins": origins, "indices": indices,
           "ids": ids, "cycles": cycles, "direct": direct, "missing": missing,
           "row_counts": actual["row_counts"],
           "row_identity_hashes": {k: v for k, v in actual.items() if k != "row_counts"},
           "direct_bundle_audits": {"E67": audit67, "E75": audit75}}
    verify_oof_identity(out, freeze["support_identity"])
    _validate_oof_support(out)
    return out


def _adapt_identity_for_frozen_hash_check(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Map the compact returned identity schema to the immutable R3 hash schema.

    Row counts are recomputed from the arrays here so both the preflight and the
    pre-target boundary validate the same ordered rows, IDs, origins, cycles,
    predictions, missing route, and stage support.
    """
    if not isinstance(identity, Mapping):
        raise GateError("R4 OOF identity must be a mapping")
    required = {"rows", "ids", "origins", "indices", "cycles", "direct",
                "missing", "row_counts"}
    missing_keys = required - set(identity)
    if missing_keys:
        raise GateError(f"R4 OOF identity schema missing keys: {sorted(missing_keys)}")
    allowed = required | {"row_identity_hashes", "direct_bundle_audits"}
    extra_keys = set(identity) - allowed
    if extra_keys:
        raise GateError(f"R4 OOF identity schema has unknown keys: {sorted(extra_keys)}")
    compact = {name: np.asarray(identity[name]) for name in
               ("rows", "ids", "origins", "indices", "cycles", "direct", "missing")}
    if any(value.ndim != 1 for value in compact.values()):
        raise GateError("R4 OOF identity arrays must be one-dimensional")
    lengths = {len(value) for value in compact.values()}
    if len(lengths) != 1 or not lengths or not next(iter(lengths)):
        raise GateError("R4 OOF identity arrays must be nonempty and row-aligned")
    rows, ids = compact["rows"], compact["ids"]
    origins, cycles = compact["origins"], compact["cycles"]
    if len(np.unique(rows)) != len(rows) or len(np.unique(ids)) != len(ids):
        raise GateError("R4 OOF physical rows and record IDs must be unique")
    row_counts = {
        "unique_oof": int(len(rows)),
        "E67_68_75": int(np.sum((origins == 67) & (cycles >= 68) & (cycles <= 75))),
        "E75_76_79": int(np.sum((origins == 75) & (cycles >= 76) & (cycles <= 79))),
    }
    supplied_counts = identity["row_counts"]
    if not isinstance(supplied_counts, Mapping) or dict(supplied_counts) != row_counts:
        raise GateError("R4 OOF row counts differ from recomputed origin/cycle arrays")
    return {
        "physical_rows": rows,
        "record_ids": ids,
        "prediction_origins": origins,
        "original_bundle_indices": compact["indices"],
        "cycles": cycles,
        "direct_predictions": compact["direct"],
        "missing_mask": compact["missing"],
        "row_counts": row_counts,
    }


def verify_oof_identity(identity: Mapping[str, Any], expected: Mapping[str, Any]) -> dict[str, Any]:
    canonical = _adapt_identity_for_frozen_hash_check(identity)
    arrays = {
        "rows_sha256": r1.tagged_array_sha256(canonical["physical_rows"]),
        "ids_sha256": r1.tagged_array_sha256(canonical["record_ids"]),
        "origins_sha256": r1.tagged_array_sha256(canonical["prediction_origins"]),
        "original_bundle_indices_sha256": r1.tagged_array_sha256(canonical["original_bundle_indices"]),
        "cycles_sha256": r1.tagged_array_sha256(canonical["cycles"]),
        "direct_prediction_sha256": r1.tagged_array_sha256(canonical["direct_predictions"], dtype="<f8"),
        "missing_mask_sha256": r1.tagged_array_sha256(canonical["missing_mask"]),
    }
    if ("row_identity_hashes" in identity
            and (not isinstance(identity["row_identity_hashes"], Mapping)
                 or dict(identity["row_identity_hashes"]) != arrays)):
        raise GateError("R4 returned row fingerprint metadata differs from recomputed hashes")
    actual = {**arrays, "row_counts": canonical["row_counts"]}
    if actual != dict(expected):
        raise GateError("Selective R4 OOF row order/hash differs from frozen R3 identity")
    return actual


def load_targets_after_source_pass(identity: Mapping[str, Any], approval_sha: str,
                                   freeze: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    _current_freeze, _approval, current_sha = verify_source_pass()
    if current_sha != approval_sha:
        raise GateError("R4 ROOT_SOURCE_PASS changed before Train target access")
    verify_oof_identity(identity, freeze["support_identity"])
    labels, _text_load, parsed_rows, audit = read_scored_query(
        v2.SOURCE, np.asarray(identity["rows"], dtype=np.int64),
        np.asarray(identity["ids"], dtype=str), np.asarray(identity["cycles"], dtype=np.int64),
        expected_total_rows=50_608)
    y = np.asarray(labels, dtype=np.float64)
    if (not np.array_equal(parsed_rows, identity["rows"])
            or y.shape != (EXPECTED_OOF_ROWS,)
            or not np.all(np.isfinite(y)) or np.any((y < 0.0) | (y > 100.0))
            or audit.get("fit_target_rows_parsed") != 0
            or audit.get("sha256") != v2.TRAIN_SHA):
        raise GateError("Selected Train labels failed frozen row/range/hash boundary")
    return y, audit


def _load_causal_global(stage: str, freeze: Mapping[str, Any]) -> tuple[float, float, str]:
    name = f"{stage}_GLOBAL_state"
    artifact = freeze["input_hashes"].get(f"r3_stage__{name}")
    if artifact is None:
        raise GateError(f"Frozen {stage}_GLOBAL state is absent")
    path = Path(artifact["path"])
    digest = file_sha256(path)
    if digest != artifact["sha256"]:
        raise GateError(f"Frozen {stage}_GLOBAL bytes changed")
    state = json.loads(path.read_text(encoding="utf-8"))
    result = state.get("solver_result", {})
    if state.get("status") != "LP_COMPLETE" or result.get("success") is not True:
        raise GateError(f"Stored causal {stage}_GLOBAL is not a completed fit")
    intercept, slope = float(result["intercept"]), float(result["slope"])
    if not math.isfinite(intercept) or not math.isfinite(slope) or slope < 0:
        raise GateError(f"Stored causal {stage}_GLOBAL coefficients invalid")
    return intercept, slope, digest


def _np_vectors(**values: Any) -> dict[str, np.ndarray]:
    arrays = {name: np.asarray(value) for name, value in values.items()}
    if any(array.ndim != 1 for array in arrays.values()):
        raise GateError("Fit/score data arrays must be one dimensional")
    if len({len(array) for array in arrays.values()}) != 1:
        raise GateError("Fit/score data arrays are not row aligned")
    return arrays


def _fit_input_identity(fit_id: str, fit_arrays: Mapping[str, np.ndarray],
                         bundle_sha: str, freeze: Mapping[str, Any],
                         freeze_sha: str, approval_sha: str) -> dict[str, Any]:
    identity_keys = {
        "physical_rows": "physical_rows_sha256",
        "record_ids": "record_ids_sha256",
        "prediction_origins": "prediction_origins_sha256",
        "original_bundle_indices": "original_bundle_indices_sha256",
        "cycles": "cycles_sha256",
        "missing_mask": "missing_mask_sha256",
    }
    tagged = {digest_key: r1.tagged_array_sha256(fit_arrays[array_key])
              for array_key, digest_key in identity_keys.items()}
    tagged["direct_prediction_sha256"] = r1.tagged_array_sha256(
        fit_arrays["direct_predictions"], dtype="<f8")
    tagged["target_sha256"] = r1.tagged_array_sha256(fit_arrays["targets"], dtype="<f8")
    tagged["direct_raw_f64_sha256"] = r1.raw_direct_f64_sha256(fit_arrays["direct_predictions"])
    return {"experiment_id": EXPERIMENT_ID, "fit_id": fit_id,
            "fit_rows": int(len(fit_arrays["targets"])),
            "row_hashes": tagged, "fit_input_bundle_sha256": bundle_sha,
            "source_commit": freeze["source_commit"],
            "source_hashes": freeze["source_hashes"],
            "input_hashes": freeze["input_hashes"],
            "prefit_freeze_sha256": freeze_sha,
            "root_source_pass_sha256": approval_sha}


def _supports(p: np.ndarray, *, purpose: str) -> dict[str, int]:
    n = len(p)
    minimum = DIAGNOSTIC_MIN_ROWS if purpose == "diagnostic" else FINAL_MIN_ROWS
    low, high = int(np.count_nonzero(p < KNOT)), int(np.count_nonzero(p >= KNOT))
    if n < minimum or min(low, high) < MIN_SIDE_ROWS:
        raise GateError(f"Unsupported {purpose} fit: rows={n}, p<50={low}, p>=50={high}; no LP start")
    return {"rows": n, "p_lt_50": low, "p_gte_50": high}


def solve_hinge_l1(p_values: Any, y_values: Any, *, purpose: str) -> dict[str, Any]:
    p = np.asarray(p_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    if (p.ndim != 1 or y.ndim != 1 or len(p) != len(y) or not len(p)
            or not np.all(np.isfinite(p)) or not np.all(np.isfinite(y))
            or np.any((p < 0) | (p > 100)) or np.any((y < 0) | (y > 100))):
        raise GateError("LP inputs must be aligned, finite vectors in [0,100]")
    support = _supports(p, purpose=purpose)
    n = len(p)
    low, high = np.minimum(p, KNOT), np.maximum(p - KNOT, 0.0)
    ix = np.arange(n, dtype=np.int64)
    rows = np.concatenate((ix, ix, ix, n + ix, n + ix, n + ix, ix, n + ix))
    cols = np.concatenate((np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64),
                           3 + ix, np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64),
                           3 + ix, np.full(n, 2, dtype=np.int64),
                           np.full(n, 2, dtype=np.int64)))
    values = np.concatenate((-np.ones(n), -low, -np.ones(n),
                             np.ones(n), low, -np.ones(n), -high, high))
    a_ub = sparse.coo_matrix((values, (rows, cols)), shape=(2 * n, n + 3)).tocsr()
    b_ub = np.concatenate((-y, y))
    objective = np.concatenate((np.zeros(3, dtype=np.float64), np.ones(n, dtype=np.float64)))
    bounds = [(None, None), (SLOPE_LOWER, SLOPE_UPPER),
              (SLOPE_LOWER, SLOPE_UPPER)] + [(0.0, None)] * n
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        with threadpool_limits(limits=THREADS):
            result = linprog(objective, A_ub=a_ub, b_ub=b_ub, bounds=bounds,
                             method="highs", options={"threads": THREADS,
                                                       "time_limit": TIME_LIMIT_SECONDS})
    elapsed = time.perf_counter() - start
    warning_texts = [str(item.message) for item in seen]
    accepted_thread_warning = [message for item, message in zip(seen, warning_texts, strict=True)
        if issubclass(item.category, OptimizeWarning)
        and "unrecognized options detected" in message.casefold()
        and "threads" in message.casefold()
        and "passed to highs" in message.casefold()]
    unexpected = [message for message in warning_texts if message not in accepted_thread_warning]
    if unexpected:
        raise GateError("Unexpected solver warning(s): " + " | ".join(unexpected))
    if not result.success or result.x is None or not np.all(np.isfinite(result.x)):
        raise GateError(f"HiGHS failed: status={result.status}; {result.message}")
    solution = np.asarray(result.x, dtype=np.float64)
    intercept, slope_low, slope_high = map(float, solution[:3])
    if not (SLOPE_LOWER <= slope_low <= SLOPE_UPPER
            and SLOPE_LOWER <= slope_high <= SLOPE_UPPER):
        raise GateError("HiGHS returned slopes outside fixed [0,2] bounds")
    prediction = intercept + slope_low * low + slope_high * high
    slacks = solution[3:]
    loss = float(np.abs(y - prediction).sum())
    slack_objective = float(slacks.sum())
    violation = float(np.max(np.abs(y - prediction) - slacks))
    if (slacks.shape != (n,) or np.any(slacks < -1e-8) or violation > 1e-7
            or not np.isclose(loss, float(result.fun), rtol=1e-8, atol=1e-7)
            or not np.isclose(slack_objective, float(result.fun), rtol=1e-8, atol=1e-7)):
        raise GateError("HiGHS solution fails primal/slack/unweighted-objective replay")
    return {"success": True, "status": int(result.status), "message": str(result.message),
            "intercept": intercept, "slope_low": slope_low, "slope_high": slope_high,
            "knot": KNOT, "slope_bounds": [SLOPE_LOWER, SLOPE_UPPER],
            "objective_unweighted_sum_absolute": loss, "solver_objective": float(result.fun),
            "slack_objective": slack_objective, "max_primal_violation": violation,
            "minimum_slack": float(np.min(slacks)), "fit_rows": n, "support": support,
            "elapsed_seconds": elapsed,
            "accepted_thread_option_warnings": accepted_thread_warning,
            "solver": {"method": "highs", "threads": THREADS,
                       "time_limit_seconds": TIME_LIMIT_SECONDS},
            "_slacks": slacks}


def _append_json_once(path: Path, body: Mapping[str, Any]) -> str:
    line = json.dumps(dict(body), sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(",", ":")).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(line).hexdigest()


def _validate_saved_fit(path: Path, expected: Mapping[str, Any],
                        slacks_path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise GateError("Started fit has no published state; start consumed and retry is blocked")
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("fit_input") != dict(expected):
        raise GateError("Published fit input/source identity differs; retry is blocked")
    slack_sha = state.get("solver_slacks_sha256")
    if state.get("status") == "LP_COMPLETE":
        if not slacks_path.is_file() or file_sha256(slacks_path) != slack_sha:
            raise GateError("Published fit lacks matching primal slack artifact")
    return state


def run_fit_slot(fit_id: str, fit_arrays: Mapping[str, np.ndarray], *,
                 expected_identity: Mapping[str, Any], approval_sha: str,
                 freeze: Mapping[str, Any], ledger_path: Path | None = None,
                 run_dir: Path | None = None,
                 solver: Callable[[Any, Any], Mapping[str, Any]] | None = None,
                 approval_check: Callable[[], str] | None = None,
                 resource_check: Callable[[], Mapping[str, Any]] =
                    lambda: resource_snapshot(boundary="fit_start", initial=True),
                 process_check: Callable[[], Any] = v2.one_job_gate,
                 after_publish: Callable[[], Any] | None = None) -> dict[str, Any]:
    """Durable RESERVE -> LP_STARTED -> state -> terminal with recovery and cap3."""
    ledger_path = LEDGER if ledger_path is None else Path(ledger_path)
    run_dir = RUN_DIR if run_dir is None else Path(run_dir)
    if fit_id not in FIT_IDS or fit_id == "HINGE_FINAL":
        raise GateError("Diagnostic stage may start only HINGE_D1 or HINGE_D2")
    if dict(expected_identity).get("fit_id") != fit_id:
        raise GateError("Fit identity does not bind requested fit ID")
    _supports(np.asarray(fit_arrays["direct_predictions"], dtype=np.float64), purpose="diagnostic")
    fit_path = run_dir / "fits" / f"{fit_id}.json"
    fit_inputs_path = run_dir / "fit_inputs" / f"{fit_id}.npz"
    slacks_path = run_dir / "solver_slacks" / f"{fit_id}.npz"
    input_sha = write_npz_once(fit_inputs_path, fit_arrays)
    if input_sha != expected_identity.get("fit_input_bundle_sha256"):
        raise GateError("Fit-input bundle hash differs from its prepared identity")
    if approval_check is not None and approval_check() != approval_sha:
        raise GateError("ROOT_SOURCE_PASS changed before fit reservation/start handling")
    expected = dict(expected_identity)
    ledger = read_ledger(ledger_path)
    stage = ledger.get(fit_id, {name: [] for name in
        ("RESERVE", "LP_STARTED", "LP_COMPLETE", "LP_FAILED")})
    reserve = stage["RESERVE"][0] if stage["RESERVE"] else None
    started = stage["LP_STARTED"][0] if stage["LP_STARTED"] else None
    terminal = stage["LP_COMPLETE"][0] if stage["LP_COMPLETE"] else None
    failed = stage["LP_FAILED"][0] if stage["LP_FAILED"] else None
    if (terminal or failed) and not fit_path.is_file():
        raise GateError("Terminal ledger event lacks state; consumed slot cannot be retried")
    if fit_path.exists():
        if not reserve or not started:
            raise GateError("Saved fit state lacks durable reservation/start proof")
        saved = _validate_saved_fit(fit_path, expected, slacks_path)
        state_sha = file_sha256(fit_path)
        if terminal or failed:
            event = terminal or failed
            if event.get("state_sha256") != state_sha:
                raise GateError("Terminal event does not bind current immutable fit state")
            if saved["status"] != "LP_COMPLETE":
                raise GateError("Prior failed start is terminal; no retry")
            return saved
        # Recover a fit published just before interruption of terminal metadata.
        if saved["status"] == "LP_COMPLETE":
            _append_json_once(ledger_path, {"event": "LP_COMPLETE", "fit_id": fit_id,
                "reservation_id": reserve["reservation_id"], "fit_input": expected,
                "source_commit": freeze["source_commit"],
                "root_source_pass_sha256": approval_sha, "state_sha256": state_sha,
                "recovered_after_state_publish": True,
                "at_utc": datetime.now(timezone.utc).isoformat(), "retry_allowed": False})
            return saved
        if saved["status"] == "LP_FAILED_TERMINAL":
            _append_json_once(ledger_path, {"event": "LP_FAILED", "fit_id": fit_id,
                "reservation_id": reserve["reservation_id"], "fit_input": expected,
                "source_commit": freeze["source_commit"],
                "root_source_pass_sha256": approval_sha, "state_sha256": state_sha,
                "recovered_after_state_publish": True,
                "at_utc": datetime.now(timezone.utc).isoformat(), "retry_allowed": False})
            raise GateError("Published failed LP recovered as terminal; no retry")
        raise GateError("Published failed fit has no terminal marker; no retry")
    if started:
        raise GateError("LP_STARTED exists without proof of a published fit; ambiguous start consumed")
    if terminal or failed:
        raise GateError("Terminal fit state was removed; consumed slot cannot be retried")
    if reserve:
        if (reserve.get("fit_input") != expected
                or reserve.get("source_commit") != freeze["source_commit"]
                or reserve.get("root_source_pass_sha256") != approval_sha):
            raise GateError("Prior reservation differs from exact current fit input")
        reserved_ids = [name for name in FIT_IDS
                        if name in ledger and ledger[name]["RESERVE"]]
        if (not reserved_ids or reserved_ids != list(FIT_IDS[:len(reserved_ids)])
                or reserved_ids[-1] != fit_id
                or any(not ledger[name]["LP_COMPLETE"] for name in reserved_ids[:-1])):
            raise GateError("Reserved fit is not the last resumable slot")
        reservation_id = reserve["reservation_id"]
    else:
        if _next_fit_id(ledger) != fit_id:
            raise GateError(f"Next available fit slot is {_next_fit_id(ledger)}; refusing {fit_id}")
        if any(name in ledger and ledger[name]["RESERVE"]
               and not ledger[name]["LP_COMPLETE"] for name in FIT_IDS if name != fit_id):
            raise GateError("Earlier reserved fit is incomplete; next slot is blocked")
        if approval_check is not None and approval_check() != approval_sha:
            raise GateError("ROOT_SOURCE_PASS changed before LP reservation")
        reservation_id = str(uuid.uuid4())
        resource = resource_check()
        _append_json_once(ledger_path, {"event": "RESERVE", "fit_id": fit_id,
            "reservation_id": reservation_id, "fit_input": expected,
            "source_commit": freeze["source_commit"],
            "root_source_pass_sha256": approval_sha, "resource": dict(resource),
            "attempt": 1, "retry_allowed": False,
            "at_utc": datetime.now(timezone.utc).isoformat()})
    if approval_check is not None and approval_check() != approval_sha:
        raise GateError("ROOT_SOURCE_PASS changed before LP start")
    resource = resource_check()
    process_check()
    _append_json_once(ledger_path, {"event": "LP_STARTED", "fit_id": fit_id,
        "reservation_id": reservation_id, "fit_input": expected,
        "source_commit": freeze["source_commit"],
        "root_source_pass_sha256": approval_sha, "resource": dict(resource),
        "at_utc": datetime.now(timezone.utc).isoformat(), "retry_allowed": False})
    try:
        if solver is None:
            solver_result = dict(solve_hinge_l1(
                fit_arrays["direct_predictions"], fit_arrays["targets"],
                purpose="diagnostic"))
        else:
            solver_result = dict(solver(fit_arrays["direct_predictions"], fit_arrays["targets"]))
        if solver_result.get("success") is not True:
            raise GateError("LP solver returned a non-success result")
        p = np.asarray(fit_arrays["direct_predictions"], dtype=np.float64)
        y = np.asarray(fit_arrays["targets"], dtype=np.float64)
        pred = (solver_result["intercept"] + solver_result["slope_low"] * np.minimum(p, KNOT)
                + solver_result["slope_high"] * np.maximum(p - KNOT, 0.0))
        slacks = np.asarray(solver_result.pop("_slacks"), dtype=np.float64)
        if slacks.shape != y.shape:
            raise GateError("LP solver did not provide one primal slack per fit row")
        if np.max(np.abs(y - pred) - slacks) > 1e-7 or np.any(slacks < -1e-8):
            raise GateError("LP primal slack constraints failed before publication")
        slack_sha = write_npz_once(slacks_path, {"slacks": slacks})
        after_fit_resources = dict(resource_check())
        fit_status = "LP_COMPLETE"
        failure = None
        slack_meta = {"path": path_identity(slacks_path), "sha256": slack_sha}
    except Exception as exc:
        solver_result = {"success": False, "error_type": type(exc).__name__, "error": str(exc)}
        fit_status = "LP_FAILED_TERMINAL"
        failure = {"type": type(exc).__name__, "message": str(exc)}
        slack_meta = None
    state_body = {"schema_version": 2, "experiment_id": EXPERIMENT_ID,
        "fit_id": fit_id, "status": fit_status, "fit_input": expected,
        "solver_result": solver_result, "solver_slacks": slack_meta,
        "solver_slacks_sha256": None if slack_meta is None else slack_meta["sha256"],
        "resources_after_solver": after_fit_resources if fit_status == "LP_COMPLETE" else None,
        "failure": failure,
        "saved_utc": datetime.now(timezone.utc).isoformat()}
    state_sha = write_json_once(fit_path, state_body)
    if after_publish is not None:
        after_publish()
    event_kind = "LP_COMPLETE" if fit_status == "LP_COMPLETE" else "LP_FAILED"
    _append_json_once(ledger_path, {"event": event_kind, "fit_id": fit_id,
        "reservation_id": reservation_id, "fit_input": expected,
        "source_commit": freeze["source_commit"],
        "root_source_pass_sha256": approval_sha, "state_sha256": state_sha,
        "recovered_after_state_publish": False,
        "at_utc": datetime.now(timezone.utc).isoformat(), "retry_allowed": False})
    if fit_status != "LP_COMPLETE":
        raise GateError(f"{fit_id} LP failed terminally; slot consumed")
    return state_body


def _stage_masks(identity: Mapping[str, Any], fit_id: str) -> tuple[np.ndarray, np.ndarray]:
    cycles = np.asarray(identity["cycles"], dtype=np.int64)
    origins = np.asarray(identity["origins"], dtype=np.int64)
    if fit_id == "HINGE_D1":
        fit = (origins == 67) & (cycles >= 68) & (cycles <= 71)
        score = (origins == 67) & (cycles >= 72) & (cycles <= 75)
    elif fit_id == "HINGE_D2":
        fit = (origins == 67) & (cycles >= 68) & (cycles <= 75)
        score = (origins == 75) & (cycles >= 76) & (cycles <= 79)
    else:
        raise GateError(f"Not a diagnostic fit ID: {fit_id}")
    if not fit.any() or not score.any() or np.any(fit & score):
        raise GateError(f"Empty or overlapping causal stage masks for {fit_id}")
    return fit, score


def _metric(y: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    return {"n": int(len(error)), "mae": float(np.mean(np.abs(error))),
            "mean_error": float(np.mean(error)),
            "p90_absolute_error": float(np.quantile(np.abs(error), 0.90))}


def _policy_metrics(y: np.ndarray, cycles: np.ndarray, missing: np.ndarray,
                    prediction: np.ndarray) -> dict[str, Any]:
    masks = {"present": ~missing, "missing": missing,
             "target_le_5": y <= 5.0, "target_ge_95": y >= 95.0}
    groups = {}
    for name, mask in masks.items():
        n = int(mask.sum())
        groups[name] = ({"n": n, "status": "UNKNOWN_UNSUPPORTED", "mae": None}
                        if n < 200 else {"status": "SUPPORTED", **_metric(y[mask], prediction[mask])})
    groups["by_cycle"] = {str(int(cycle)): _metric(y[cycles == cycle], prediction[cycles == cycle])
                          for cycle in np.unique(cycles)}
    return {"overall": _metric(y, prediction), "groups": groups}


def _score_metrics(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    y, cycles, missing = arrays["targets"], arrays["cycles"], arrays["missing_mask"]
    return {name: _policy_metrics(y, cycles, missing, arrays[name]) for name in
            ("raw_direct", "causal_global_all_rows", "causal_global_present_only", "hinge_present_only")}


def _subgroup_harm(metrics: Mapping[str, Any]) -> dict[str, Any]:
    baseline = metrics["causal_global_present_only"]["groups"]
    result = {}
    harms = []
    for name in ("present", "missing", "target_le_5", "target_ge_95"):
        candidate, control = metrics["hinge_present_only"]["groups"][name], baseline[name]
        if candidate["status"] != "SUPPORTED" or control["status"] != "SUPPORTED":
            result[name] = {"status": "UNKNOWN_UNSUPPORTED", "n": candidate["n"], "harm": None}
        else:
            harm = float(candidate["mae"] - control["mae"])
            result[name] = {"status": "SUPPORTED", "n": candidate["n"], "harm": harm}
            harms.append(harm)
    return {"groups": result, "maximum_supported_harm": max(harms) if harms else None}


def _resource_receipt(freeze_sha: str, approval_sha: str,
                      snapshots: list[Mapping[str, Any]]) -> dict[str, Any]:
    path = RUN_DIR / "RESOURCE_SNAPSHOTS_R4.json"
    body = {"schema_version": 1, "experiment_id": EXPERIMENT_ID,
            "prefit_freeze_sha256": freeze_sha,
            "root_source_pass_sha256": approval_sha,
            "resource_policy_sha256": file_sha256(RESOURCE_POLICY_FILE),
            "boundaries": [dict(item) for item in snapshots]}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if (existing.get("prefit_freeze_sha256") != freeze_sha
                or existing.get("root_source_pass_sha256") != approval_sha
                or existing.get("resource_policy_sha256") != body["resource_policy_sha256"]):
            raise GateError("Existing resource snapshot receipt binds another source/pass")
        return existing
    write_json_once(path, body)
    return body


def _prepare_fit_arrays(identity: Mapping[str, Any], targets: np.ndarray,
                        mask: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "physical_rows": np.asarray(identity["rows"][mask], dtype=np.int64),
        "record_ids": np.asarray(identity["ids"][mask], dtype=str),
        "prediction_origins": np.asarray(identity["origins"][mask], dtype=np.int8),
        "original_bundle_indices": np.asarray(identity["indices"][mask], dtype=np.int64),
        "cycles": np.asarray(identity["cycles"][mask], dtype=np.int16),
        "direct_predictions": np.asarray(identity["direct"][mask], dtype="<f8"),
        "missing_mask": np.asarray(identity["missing"][mask], dtype=bool),
        "targets": np.asarray(targets[mask], dtype="<f8"),
    }


def _prepare_score_arrays(identity: Mapping[str, Any], targets: np.ndarray,
                          mask: np.ndarray, fit_state: Mapping[str, Any],
                          global_state: tuple[float, float, str]) -> dict[str, np.ndarray]:
    direct = np.asarray(identity["direct"][mask], dtype="<f8")
    missing = np.asarray(identity["missing"][mask], dtype=bool)
    y = np.asarray(targets[mask], dtype="<f8")
    result = fit_state["solver_result"]
    params = {"intercept": result["intercept"], "slope_low": result["slope_low"],
              "slope_high": result["slope_high"]}
    hinge_prediction = r1.apply_hinge_present_only(direct, missing, params).astype("<f8", copy=False)
    global_a, global_b, _ = global_state
    global_all = np.clip(global_a + global_b * direct, 0.0, 100.0).astype("<f8", copy=False)
    global_present = r1.apply_global_present_only(direct, missing,
        intercept=global_a, slope=global_b).astype("<f8", copy=False)
    if not np.array_equal(hinge_prediction[missing].view("<u8"), direct[missing].view("<u8")):
        raise GateError("Hinge missing fallback lost raw DIRECT float64 bits")
    return {
        "physical_rows": np.asarray(identity["rows"][mask], dtype=np.int64),
        "record_ids": np.asarray(identity["ids"][mask], dtype=str),
        "prediction_origins": np.asarray(identity["origins"][mask], dtype=np.int8),
        "original_bundle_indices": np.asarray(identity["indices"][mask], dtype=np.int64),
        "cycles": np.asarray(identity["cycles"][mask], dtype=np.int16),
        "direct_predictions": direct,
        "missing_mask": missing,
        "targets": y,
        "raw_direct": direct.copy(),
        "hinge_present_only": hinge_prediction,
        "causal_global_all_rows": global_all,
        "causal_global_present_only": global_present,
    }


def _run_one_diagnostic(fit_id: str, identity: Mapping[str, Any], targets: np.ndarray,
                        freeze: Mapping[str, Any], freeze_sha: str,
                        approval_sha: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    fit_mask, score_mask = _stage_masks(identity, fit_id)
    fit_arrays = _prepare_fit_arrays(identity, targets, fit_mask)
    support = _supports(fit_arrays["direct_predictions"], purpose="diagnostic")
    fit_input_path = RUN_DIR / "fit_inputs" / f"{fit_id}.npz"
    bundle_sha = write_npz_once(fit_input_path, fit_arrays)
    expected = _fit_input_identity(fit_id, fit_arrays, bundle_sha, freeze, freeze_sha, approval_sha)
    _current_freeze, _approval, current_sha = verify_source_pass()
    if current_sha != approval_sha:
        raise GateError("ROOT_SOURCE_PASS changed before diagnostic stage")
    resource_calls = 0
    def stage_resource_check() -> Mapping[str, Any]:
        nonlocal resource_calls
        resource_calls += 1
        return resource_snapshot(boundary=f"{fit_id}_fit_boundary_{resource_calls}",
                                 initial=resource_calls <= 2)
    fit_state = run_fit_slot(fit_id, fit_arrays, expected_identity=expected,
        approval_sha=approval_sha, freeze=freeze, ledger_path=LEDGER,
        run_dir=RUN_DIR,
        approval_check=lambda: verify_source_pass()[2],
        resource_check=stage_resource_check)
    global_state = _load_causal_global("D1" if fit_id == "HINGE_D1" else "D2", freeze)
    score_arrays = _prepare_score_arrays(identity, targets, score_mask, fit_state, global_state)
    score_path = RUN_DIR / "score_rows" / f"{fit_id}.npz"
    score_sha = write_npz_once(score_path, score_arrays)
    fit_result = fit_state["solver_result"]
    raw_unclipped = (fit_result["intercept"]
        + fit_result["slope_low"] * np.minimum(score_arrays["direct_predictions"], KNOT)
        + fit_result["slope_high"] * np.maximum(score_arrays["direct_predictions"] - KNOT, 0.0))
    metrics = _score_metrics(score_arrays)
    global_mae = metrics["causal_global_present_only"]["overall"]["mae"]
    hinge_mae = metrics["hinge_present_only"]["overall"]["mae"]
    record = {
        "fit_id": fit_id,
        "fit_rows": support["rows"],
        "fit_support": support,
        "fit_state_path": fit_state_path_rel(fit_id),
        "fit_state_sha256": file_sha256(RUN_DIR / "fits" / f"{fit_id}.json"),
        "fit_input_path": path_identity(fit_input_path),
        "fit_input_sha256": bundle_sha,
        "solver_slacks_path": fit_state["solver_slacks"]["path"],
        "solver_slacks_sha256": fit_state["solver_slacks_sha256"],
        "parameters": {name: fit_result[name] for name in ("intercept", "slope_low", "slope_high")},
        "objective_unweighted_sum_absolute": fit_result["objective_unweighted_sum_absolute"],
        "source_global_state_sha256": global_state[2],
        "score_rows": int(len(score_arrays["targets"])),
        "score_row_bundle_path": path_identity(score_path),
        "score_row_bundle_sha256": score_sha,
        "score_metrics": metrics,
        "gain_vs_causal_global_present_only": float(global_mae - hinge_mae),
        "supported_subgroup_harm": _subgroup_harm(metrics),
        "application_clipping_rows_present_only": int(np.sum(
            (~score_arrays["missing_mask"]) & ((raw_unclipped < 0.0) | (raw_unclipped > 100.0)))),
    }
    return record, score_arrays


def fit_state_path_rel(fit_id: str) -> str:
    return (RUN_DIR / "fits" / f"{fit_id}.json").relative_to(ROOT).as_posix()


def _pooled_metrics(score_arrays: list[Mapping[str, np.ndarray]]) -> tuple[dict[str, Any], int]:
    names = ("record_ids", "cycles", "missing_mask", "targets", "raw_direct",
             "causal_global_all_rows", "causal_global_present_only", "hinge_present_only")
    pooled = {name: np.concatenate([array[name] for array in score_arrays]) for name in names}
    if len(np.unique(pooled["record_ids"])) != len(pooled["record_ids"]):
        raise GateError("D1/D2 evaluation IDs overlap; unique pooling refused")
    rows = len(pooled["record_ids"])
    if rows != EXPECTED_POOLED_ROWS:
        raise GateError(f"Pooled evaluation row count {rows} differs from 15,953")
    metrics = {name: _policy_metrics(pooled["targets"], pooled["cycles"],
                                    pooled["missing_mask"], pooled[name])
               for name in ("raw_direct", "causal_global_all_rows",
                            "causal_global_present_only", "hinge_present_only")}
    return metrics, rows


def _decision(records: list[Mapping[str, Any]], pooled: Mapping[str, Any]) -> dict[str, Any]:
    control = pooled["causal_global_present_only"]["overall"]["mae"]
    candidate = pooled["hinge_present_only"]["overall"]["mae"]
    pooled_gain, pooled_harm = float(control - candidate), float(candidate - control)
    gains = [float(record["gain_vs_causal_global_present_only"]) for record in records]
    harms = [float(-gain) for gain in gains]
    supported_harms = [record["supported_subgroup_harm"]["maximum_supported_harm"]
                       for record in records]
    supported_harms = [float(value) for value in supported_harms if value is not None]
    max_subgroup_harm = max(supported_harms) if supported_harms else None
    subgroup_ok = max_subgroup_harm is None or max_subgroup_harm <= 0.25
    qualified = pooled_gain >= 0.02 and all(gain >= 0.0 for gain in gains) and subgroup_ok
    exploratory = (not qualified and (pooled_gain >= 0.0 or
        (max(gains) >= 0.02 and pooled_harm <= 0.01))
        and all(harm <= 0.03 for harm in harms) and subgroup_ok)
    return {"pooled_gain_vs_causal_global_present_only": pooled_gain,
            "window_gains_vs_causal_global_present_only": gains,
            "pooled_harm": pooled_harm,
            "window_harms": harms,
            "maximum_supported_subgroup_harm": max_subgroup_harm,
            "qualified_gate": bool(qualified),
            "exploratory_root_review_gate": bool(exploratory),
            "status": "QUALIFIED" if qualified else
                      "EXPLORATORY_ROOT_REVIEW" if exploratory else "RETIRE"}


def _save_resource_snapshots(freeze_sha: str, approval_sha: str,
                             snapshots: list[Mapping[str, Any]]) -> dict[str, Any]:
    return _resource_receipt(freeze_sha, approval_sha, snapshots)


def _current_approval_sha(expected: str) -> str:
    _freeze, _approval, digest = verify_source_pass()
    if digest != expected:
        raise GateError("R4 ROOT_SOURCE_PASS changed during diagnostics")
    return digest


def run_diagnostics() -> dict[str, Any]:
    freeze, _approval, approval_sha = verify_source_pass()
    freeze_sha = file_sha256(PREFIT_FREEZE)
    with SharedHeavyJobLock():
        resources = [resource_snapshot(boundary="initial_locked_preallocation", initial=True)]
        v2.one_job_gate()
        _current_approval_sha(approval_sha)
        identity = load_oof_identity(freeze)
        resources.append(resource_snapshot(boundary="after_selective_oof_identity"))
        _current_approval_sha(approval_sha)
        targets, target_audit = load_targets_after_source_pass(identity, approval_sha, freeze)
        resources.append(resource_snapshot(boundary="after_selected_train_targets"))
        resource_receipt = _save_resource_snapshots(freeze_sha, approval_sha, resources)
        records, score_arrays = [], []
        for fit_id in DIAGNOSTIC_FIT_IDS:
            record, score = _run_one_diagnostic(fit_id, identity, targets, freeze,
                                                freeze_sha, approval_sha)
            records.append(record)
            score_arrays.append(score)
        pooled, pooled_rows = _pooled_metrics(score_arrays)
        report = {
            "schema_version": 2,
            "experiment_id": EXPERIMENT_ID,
            "status": "TWO_DIAGNOSTICS_COMPLETE_PENDING_ROOT_REVIEW",
            "source_commit": freeze["source_commit"],
            "source_hashes": freeze["source_hashes"],
            "input_hashes": freeze["input_hashes"],
            "prefit_freeze_sha256": freeze_sha,
            "root_source_pass_sha256": approval_sha,
            "train_target_audit": target_audit,
            "resource_receipt_path": (RUN_DIR / "RESOURCE_SNAPSHOTS_R4.json").relative_to(ROOT).as_posix(),
            "resource_receipt_sha256": file_sha256(RUN_DIR / "RESOURCE_SNAPSHOTS_R4.json"),
            "resource_policy_sha256": file_sha256(RESOURCE_POLICY_FILE),
            "stages": records,
            "pooled_unique_evaluation": {"rows": pooled_rows, "metrics": pooled},
            "decision": _decision(records, pooled),
            "diagnostic_fit_ids": list(DIAGNOSTIC_FIT_IDS),
            "official_lp_starts": sum(bool(events["LP_STARTED"])
                                       for events in read_ledger().values()),
            "official_lp_start_cap": MAX_LP_STARTS,
            "conditional_final_fit_id": "HINGE_FINAL",
            "final_fit_started": False,
            "base_or_preprocessing_fits": 0,
            "test_or_sample_accessed": False,
            "csv_created": False,
            "kaggle_upload": False,
        }
        write_json_idempotent(DIAGNOSTICS, report)
        from scripts.verify_reg_monotone_hinge_p50_v4 import verify_diagnostics
        verification = verify_diagnostics(DIAGNOSTICS, freeze=freeze, approval_sha=approval_sha)
        write_json_idempotent(VERIFICATION, verification)
        return {"diagnostics_sha256": file_sha256(DIAGNOSTICS),
                "verification_sha256": file_sha256(VERIFICATION),
                "status": report["status"], "decision": report["decision"]["status"]}


def write_candidate_csv(path: Path, record_ids: Any, threat_rank: Any, *,
                        missing_mask: Any, raw_direct: Any) -> str:
    ids = np.asarray(record_ids, dtype=str)
    values = np.asarray(threat_rank, dtype="<f8")
    if ids.ndim != 1 or values.ndim != 1 or len(ids) != len(values) or not len(ids):
        raise GateError("Candidate CSV IDs and threat_rank must be nonempty aligned vectors")
    if len(np.unique(ids)) != len(ids):
        raise GateError("Candidate CSV official IDs are not unique")
    if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 100.0)):
        raise GateError("Candidate CSV threat_rank must be finite in [0,100]")
    missing = np.asarray(missing_mask, dtype=bool)
    raw = np.asarray(raw_direct, dtype="<f8")
    if missing.shape != values.shape or raw.shape != values.shape:
        raise GateError("Candidate CSV fallback vectors are misaligned")
    if not np.array_equal(values[missing].view("<u8"), raw[missing].view("<u8")):
        raise GateError("Candidate CSV missing route differs from raw DIRECT bits")
    if len(ids) != EXPECTED_TEST_ROWS or int(missing.sum()) != EXPECTED_MISSING_TEST_ROWS:
        raise GateError("Production CSV row/missing count differs from frozen contract")
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(("record_id", "threat_rank"))
    for record_id, value in zip(ids, values, strict=True):
        writer.writerow((record_id, format(float(value), ".17g")))
    return _atomic_bytes_once(path, buffer.getvalue().encode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--freeze-identity", action="store_true")
    actions.add_argument("--identity-preflight", action="store_true")
    actions.add_argument("--synthetic-qa", action="store_true")
    actions.add_argument("--run-diagnostics", action="store_true")
    actions.add_argument("--verify-diagnostics", action="store_true")
    args = parser.parse_args()
    if args.freeze_identity:
        path, digest = freeze_identity_r4()
        print(json.dumps({"status": "IDENTITY_FROZEN_ONLY", "path": str(path),
                          "sha256": digest}, sort_keys=True))
        return 0
    if args.identity_preflight:
        print(json.dumps(identity_preflight(), sort_keys=True))
        return 0
    if args.synthetic_qa:
        from scripts.verify_reg_monotone_hinge_p50_v4 import write_synthetic_qa
        print(json.dumps(write_synthetic_qa(), sort_keys=True))
        return 0
    if args.run_diagnostics:
        print(json.dumps(run_diagnostics(), sort_keys=True))
        return 0
    freeze, _approval, approval_sha = verify_source_pass()
    from scripts.verify_reg_monotone_hinge_p50_v4 import verify_diagnostics
    result = verify_diagnostics(DIAGNOSTICS, freeze=freeze, approval_sha=approval_sha)
    write_json_idempotent(VERIFICATION, result)
    print(json.dumps({"status": result["status"],
                      "verification_sha256": file_sha256(VERIFICATION)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GateError as exc:
        print(json.dumps({"status": "BLOCKED_FAIL_CLOSED", "reason": str(exc)}, sort_keys=True),
              file=sys.stderr)
        raise SystemExit(2)
