"""Bounded REG follow-up: causal global vs H8 calibration and exact fallback.

Official Train target access and LP execution require a matching Sol ROOT_SOURCE_PASS.
The default source-preparation commands are identity-only and never parse targets.
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
import subprocess
import tempfile
import time
from typing import Any, Mapping
import uuid
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import OptimizeWarning, linprog
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))
import scripts.reg_final_wave_v2 as v2  # noqa: E402
from scripts.reg_position_failure_mechanism_v1 import read_scored_query  # noqa: E402
from src.models.optimization_v3_batch1_r17 import apply_affine, solve_nonnegative_l1  # noqa: E402

EXPERIMENT = "REG_FINAL_WAVE_FOLLOWUP_V2"
BRANCH = "codex/reg-position-failure-mechanism-v1"
REPORT_DIR = ROOT / "reports/optimization_v3/REG_FINAL_WAVE_FOLLOWUP_V2"
EXPERIMENT_DIR = ROOT / "experiments/REG_FINAL_WAVE_FOLLOWUP_V2"
RUN_DIR = ROOT / "experiments/runs/REG_FINAL_WAVE_FOLLOWUP_V2"
SUBMISSION_DIR = ROOT / "submissions/reg_final_wave_followup_v2"
PROTOCOL = REPORT_DIR / "PROTOCOL.md"
PRIOR_SYNTHETIC_QA = REPORT_DIR / "SYNTHETIC_QA.json"
SYNTHETIC_QA = REPORT_DIR / "SYNTHETIC_QA_R2.json"
PRIOR_PREFIT_FREEZE = EXPERIMENT_DIR / "PREFIT_FREEZE.json"
PREFIT_FREEZE = EXPERIMENT_DIR / "PREFIT_FREEZE_R2.json"
PRIOR_BLOCKER_REPORT = REPORT_DIR / "SOURCE_PREPARATION_BLOCKED.json"
BLOCKER_REPORT = REPORT_DIR / "SOURCE_PREPARATION_BLOCKED_R2.json"
IDENTITY_PREFLIGHT = REPORT_DIR / "IDENTITY_PREFLIGHT_R2.json"
PRIOR_PREFLIGHT = REPORT_DIR / "IDENTITY_PREFLIGHT.json"
ROOT_SOURCE_PASS = EXPERIMENT_DIR / "ROOT_SOURCE_PASS.json"
ROOT_PRODUCTION_PASS = EXPERIMENT_DIR / "ROOT_PRODUCTION_PASS.json"
LEDGER = RUN_DIR / "lp_ledger.jsonl"
DIAGNOSTICS = REPORT_DIR / "DIAGNOSTICS.json"
DIAGNOSTIC_BUNDLE = RUN_DIR / "diagnostic_rows.npz"
FINAL_STATE = RUN_DIR / "fits" / "FINAL_RECENCY_H8.json"
FINAL_APPLICATION_FREEZE = REPORT_DIR / "FINAL_APPLICATION_FREEZE.json"
OUTPUT_A = SUBMISSION_DIR / "recency_affine_h8_e79.csv"
OUTPUT_B = SUBMISSION_DIR / "global_affine_present_only_e79.csv"
MANIFEST_A = REPORT_DIR / "MANIFEST_A.json"
MANIFEST_B = REPORT_DIR / "MANIFEST_B.json"

SCRIPT = Path(__file__).resolve()
VERIFIER = ROOT / "scripts/verify_reg_final_wave_followup_v2.py"
TESTS = ROOT / "tests/test_reg_final_wave_followup_v2.py"
R16_SOURCE = v2.R16_MODEL_SOURCE
V2_SOURCE = Path(v2.__file__).resolve()
R17_SOURCE = v2.R17_SOURCE
QUERY_READER = v2.READER_SOURCE
DIRECT_SOURCE = v2.DIRECT_SOURCE
TRAIN = v2.TRAIN
TRAIN_SHA256 = v2.TRAIN_SHA

HALF_LIFE = 8.0
THREADS = 2
MIN_AVAILABLE_RAM = 4 * 1024**3
RESERVED_RAM = 2 * 1024**3
ESTIMATED_PEAK_RAM = 512 * 1024**2  # planning estimate, not a measured peak
DIAGNOSTIC_MIN_ROWS = 1000
PRODUCTION_MIN_ROWS = 5000
EXPECTED_SUPPORT_ROWS = 20903
EXPECTED_TEST_ROWS = 29421
EXPECTED_MISSING_ROWS = 1062
WINDOWS = ((71, 72, 75), (75, 76, 79))
FIT_IDS = (
    "D1_GLOBAL", "D1_RECENCY_H8", "D2_GLOBAL", "D2_RECENCY_H8",
    "FINAL_RECENCY_H8",
)
MAX_LP_STARTS = len(FIT_IDS)
DIAGNOSTIC_FIT_IDS = FIT_IDS[:4]

# Shared with other authorized KASDAD compute lanes. Advisory OS locking releases
# automatically on process exit; the lock file is never deleted to infer liveness.
SHARED_JOB_LOCK = Path(tempfile.gettempdir()) / "kasdad_single_heavy_job.lock"
PRIOR_PREFIT_FREEZE_SHA256 = "564c1dcfbab9370f14dfb7d5698c7961a144da026c5b6e5eaf180f9f43c312e0"
PRIOR_BLOCKER_REPORT_SHA256 = "2d0bdae1de295b24aabc49ae4a39d9d7021556b2d810b9e4d3f915bf175ef26f"
PRIOR_BLOCKER_RECEIPT_SHA256 = "e12f319a45d94b781edfc4fad484e7a0636126a7582c7a6c3206edc4755cb033"

SOURCE_FILES = {
    "runner": SCRIPT,
    "verifier": VERIFIER,
    "tests": TESTS,
    "protocol": PROTOCOL,
    "synthetic_qa": SYNTHETIC_QA,
    "v2_identity_helpers": V2_SOURCE,
    "r17_lp_helper": R17_SOURCE,
    "selected_row_reader": QUERY_READER,
    "direct_prediction_source": DIRECT_SOURCE,
    "r16_h8_rationale_source": R16_SOURCE,
}

GLOBAL_A = 2.169196428141562
GLOBAL_B = 0.9744451489746011
GLOBAL_REF = 57994383
GLOBAL_PUBLIC_MAE = 11.04552


class GateError(RuntimeError):
    """Fail-closed source, identity, resource, or execution gate."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8", errors="surrogatepass")


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_sha256(values: Any, *, dtype: str | None = None) -> str:
    array = np.asarray(values, dtype=dtype)
    h = hashlib.sha256(canonical_bytes({"dtype": str(array.dtype), "shape": list(array.shape)}))
    if array.dtype.kind in "USO":
        h.update(canonical_bytes(array.astype(str).tolist()))
    else:
        h.update(np.ascontiguousarray(array).tobytes(order="C"))
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GateError(f"Expected JSON object: {path}")
    return value


def sign_body(body: Mapping[str, Any], field: str = "receipt_sha256") -> dict[str, Any]:
    payload = dict(body)
    payload[field] = canonical_sha(payload)
    return payload


def verify_signed(value: Mapping[str, Any], field: str = "receipt_sha256") -> dict[str, Any]:
    body = {k: v for k, v in value.items() if k != field}
    if value.get(field) != canonical_sha(body):
        raise GateError(f"Integrity digest failed for {field}")
    return body


def _atomic_write_once(path: Path, payload: bytes) -> str:
    """Same-directory temp + atomic no-replace hard link; never overwrites."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            if file_sha256(path) != hashlib.sha256(payload).hexdigest():
                raise GateError(f"Refusing to overwrite a different immutable file: {path}")
        return file_sha256(path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def write_json_once(path: Path, body: Mapping[str, Any]) -> str:
    signed = sign_body(body)
    payload = json.dumps(signed, sort_keys=True, ensure_ascii=False,
                         allow_nan=False, indent=2).encode("utf-8") + b"\n"
    return _atomic_write_once(path, payload)


def append_jsonl(path: Path, event: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(event), sort_keys=True, ensure_ascii=False,
                         allow_nan=False, separators=(",", ":")).encode("utf-8") + b"\n"
    fd = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "ab") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def read_ledger(path: Path | None = None) -> list[dict[str, Any]]:
    path = LEDGER if path is None else path
    if not path.exists():
        return []
    events = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GateError(f"Malformed ledger line {line_no}; stop without retry") from exc
        if not isinstance(event, dict):
            raise GateError(f"Invalid ledger event at line {line_no}")
        events.append(event)
    return events


def source_hashes() -> dict[str, str]:
    missing = [name for name, path in SOURCE_FILES.items() if not path.is_file()]
    if missing:
        raise GateError(f"Required source files missing: {missing}")
    return {name: file_sha256(path) for name, path in SOURCE_FILES.items()}


def input_paths() -> dict[str, Path]:
    paths = {f"pinned_{name}": value[0] for name, value in v2.PINS.items()}
    paths.update({
        "official_train": TRAIN,
        "global_calibrator_state": v2.OUT_STATE,
        "global_fit_receipt": v2.OUT_RECEIPT,
        "global_candidate_csv": v2.OUT_CSV,
        "global_candidate_manifest": v2.OUT_MANIFEST,
        "global_candidate_qa": v2.OUT_QA,
    })
    return paths


def snapshot_inputs() -> dict[str, dict[str, str]]:
    snapshot: dict[str, dict[str, str]] = {}
    for name, path in input_paths().items():
        if not path.is_file():
            raise GateError(f"Required identity input missing: {name}: {path}")
        snapshot[name] = {"path": str(path.resolve()), "sha256": file_sha256(path)}
    if snapshot["official_train"]["sha256"] != TRAIN_SHA256:
        raise GateError("Regression Train raw file hash differs from frozen identity")
    for name, (path, expected) in v2.PINS.items():
        if snapshot[f"pinned_{name}"]["sha256"] != expected:
            raise GateError(f"Pinned historical input changed: {name}")
    return snapshot


def verify_snapshot(expected: Mapping[str, Mapping[str, str]]) -> None:
    actual = snapshot_inputs()
    if dict(expected) != actual:
        changed = [name for name in sorted(set(expected) | set(actual))
                   if expected.get(name) != actual.get(name)]
        raise GateError(f"Pinned source/input identities changed: {changed}")


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()


def _repo_path(path: str) -> str:
    return path.replace("\\", "/").removeprefix("./")


def _is_report_only_path(path: str) -> bool:
    normalized = _repo_path(path)
    pinned = set()
    for value in SOURCE_FILES.values():
        try:
            pinned.add(_repo_path(str(value.resolve().relative_to(ROOT))))
        except ValueError:
            # Some pinned historical dependencies live in neighboring checkouts.
            continue
    prefix = "reports/optimization_v3/REG_FINAL_WAVE_FOLLOWUP_V2/"
    return (normalized.startswith(prefix) and normalized not in pinned
            and Path(normalized).suffix.casefold() in {".md", ".txt"})


def _is_known_generated_path(path: str) -> bool:
    normalized = _repo_path(path)
    experiment = "experiments/REG_FINAL_WAVE_FOLLOWUP_V2/"
    reports = "reports/optimization_v3/REG_FINAL_WAVE_FOLLOWUP_V2/"
    known_experiment_names = {
        "PREFIT_FREEZE.json", "PREFIT_FREEZE_R2.json",
        "ROOT_SOURCE_PASS.json", "ROOT_PRODUCTION_PASS.json",
    }
    known_report_names = {
        "SOURCE_PREPARATION_BLOCKED.json", "SOURCE_PREPARATION_BLOCKED_R2.json",
        "IDENTITY_PREFLIGHT.json", "IDENTITY_PREFLIGHT_R2.json",
        "DIAGNOSTICS.json", "PRODUCTION_RESULT.json", "FINAL_APPLICATION_FREEZE.json",
        "MANIFEST_A.json", "MANIFEST_B.json",
    }
    return ((normalized.startswith(experiment) and Path(normalized).name in known_experiment_names)
            or (normalized.startswith(reports) and Path(normalized).name in known_report_names))


def _commit_is_ancestor(ancestor: str, descendant: str) -> bool:
    try:
        result = subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor",
                                 ancestor, descendant], text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, check=False)
    except OSError as exc:
        raise GateError(f"Unable to verify frozen commit ancestry: {exc}") from exc
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    raise GateError(f"Git could not prove frozen commit ancestry: {result.stderr.strip()}")


def validate_live_source_checkpoint(freeze_body: Mapping[str, Any]) -> dict[str, Any]:
    """Accept report-only descendants while enforcing all frozen executable/input bytes."""
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    frozen_commit = str(freeze_body.get("source_commit", ""))
    if branch != BRANCH:
        raise GateError(f"Expected isolated branch {BRANCH}; observed {branch!r}")
    if not frozen_commit or not _commit_is_ancestor(frozen_commit, head):
        raise GateError("Frozen source commit is not an ancestor of current HEAD")

    descendant_paths = git("diff", "--name-only", f"{frozen_commit}..{head}", "--").splitlines()
    changed_paths = (git("diff", "--name-only", "HEAD", "--").splitlines()
                     + git("diff", "--cached", "--name-only", "--").splitlines())
    non_report_changes = [path for path in descendant_paths + changed_paths
                          if not _is_report_only_path(path)]
    if non_report_changes:
        raise GateError(f"Frozen checkpoint has non-report source changes: {sorted(set(non_report_changes))}")

    untracked = git("ls-files", "--others", "--exclude-standard").splitlines()
    unexpected_untracked = [path for path in untracked
                            if not _is_known_generated_path(path) and not _is_report_only_path(path)]
    if unexpected_untracked:
        raise GateError(f"Unexpected untracked files beside frozen source: {unexpected_untracked}")
    expected_sources = dict(freeze_body.get("source_hashes", {}))
    actual_sources = source_hashes()
    if expected_sources != actual_sources:
        changed = [name for name in sorted(set(expected_sources) | set(actual_sources))
                   if expected_sources.get(name) != actual_sources.get(name)]
        raise GateError(f"Pinned executable/dependency/protocol/test/QA bytes changed: {changed}")
    verify_snapshot(freeze_body.get("input_hashes", {}))
    return {"source_commit": frozen_commit, "live_head": head, "branch": branch,
            "report_only_descendant_paths": descendant_paths,
            "pinned_source_hashes_verified": True, "input_hashes_verified": True}


def assert_source_checkout(*, allow_new_artifacts: bool = False) -> tuple[str, str]:
    commit = git("rev-parse", "HEAD")
    branch = git("branch", "--show-current")
    if branch != BRANCH:
        raise GateError(f"Expected isolated branch {BRANCH}; observed {branch!r}")
    tracked_changes = git("diff", "--name-only", "HEAD", "--").splitlines()
    staged_changes = git("diff", "--cached", "--name-only", "--").splitlines()
    if tracked_changes or staged_changes:
        raise GateError(f"Source must be committed before freeze; tracked changes={tracked_changes + staged_changes}")
    if not allow_new_artifacts:
        untracked = git("ls-files", "--others", "--exclude-standard").splitlines()
        unexpected = [path for path in untracked if not _is_known_generated_path(path)]
        if unexpected:
            raise GateError(f"Unexpected untracked files before source freeze: {unexpected}")
    return commit, branch


def shared_lock_probe(path: Path = SHARED_JOB_LOCK) -> dict[str, Any]:
    """Probe the advisory KASDAD lock without unlinking or mutating ownership."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+b")
    try:
        if path.stat().st_size == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        except OSError:
            return {"status": "BLOCKED_HELD", "path": str(path), "owner": "UNKNOWN"}
        return {"status": "AVAILABLE", "path": str(path), "owner": None}
    finally:
        stream.close()


class SharedHeavyJobLock:
    """Process-lifetime cross-platform advisory lock shared with CLS heavy work."""
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
            raise GateError(f"Another KASDAD heavy job holds {self.path}; no wait/poll") from exc
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


def resource_snapshot(*, allow_blocked: bool = False) -> dict[str, Any]:
    # Measure once so a blocked receipt records the same reading used by the gate.
    available = int(v2.psutil.virtual_memory().available)
    required = max(MIN_AVAILABLE_RAM, ESTIMATED_PEAK_RAM + RESERVED_RAM)
    status = "PASS" if available >= required else "BLOCKED_RAM"
    result = {
        "status": status, "available_physical_ram_bytes_measured": available,
        "minimum_available_ram_bytes": MIN_AVAILABLE_RAM,
        "safety_reserve_bytes": RESERVED_RAM,
        "estimated_incremental_peak_bytes": ESTIMATED_PEAK_RAM,
        "estimated_peak_is_measured": False, "threads": THREADS,
        "single_heavy_job_required": True,
    }
    if status != "PASS" and not allow_blocked:
        raise GateError(f"RAM gate failed: available={available}, required={required}")
    return result


def record_resource_blocker(freeze: Mapping[str, Any], resource: Mapping[str, Any]) -> dict[str, Any]:
    body = {
        "schema_version": 1, "experiment_id": EXPERIMENT,
        "status": "BLOCKED_RAM_IDENTITY_PREFLIGHT",
        "source_commit": freeze["source_commit"], "branch": freeze["branch"],
        "prefit_freeze_sha256": file_sha256(PREFIT_FREEZE),
        "prefit_freeze_signature": read_json(PREFIT_FREEZE)["receipt_sha256"],
        "successor_of": freeze["successor_of"],
        "resource": dict(resource),
        "official_target_rows_parsed": 0, "official_labels_accessed": False,
        "official_lp_starts": 0, "test_csv_or_sample_opened": False,
        "csv_created": False, "kaggle_upload": False,
    }
    write_json_once(BLOCKER_REPORT, body)
    return read_json(BLOCKER_REPORT)


def require_row_support(count: int, minimum: int, label: str) -> None:
    if count < minimum:
        raise GateError(f"{label} support {count} is below the frozen minimum {minimum}")


def diagnostic_support(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Validate cycle/row support without touching official target columns."""
    oof = identity["oof"]
    cycles = np.asarray(oof["cycles"], dtype=np.int64)
    rows, ids = np.asarray(oof["rows"]), np.asarray(oof["ids"])
    if len(rows) != EXPECTED_SUPPORT_ROWS or len(ids) != EXPECTED_SUPPORT_ROWS:
        raise GateError("Diagnostic identity support differs from frozen 20,903 rows")
    if len(np.unique(rows)) != len(rows) or len(np.unique(ids)) != len(ids):
        raise GateError("Diagnostic source identity rows/IDs are not unique")
    results = {}
    for name, cutoff, score_lo, score_hi in (("D1", 71, 72, 75), ("D2", 75, 76, 79)):
        history, score = _stage_masks(cycles, cutoff, score_lo, score_hi)
        require_row_support(int(history.sum()), DIAGNOSTIC_MIN_ROWS, f"{name} fit")
        require_row_support(int(score.sum()), DIAGNOSTIC_MIN_ROWS, f"{name} score")
        results[name] = {"fit_rows": int(history.sum()), "score_rows": int(score.sum()),
                         "fit_cycles": [68, cutoff], "score_cycles": [score_lo, score_hi]}
    return results


def production_support(identity: Mapping[str, Any]) -> dict[str, Any]:
    cycles = np.asarray(identity["oof"]["cycles"], dtype=np.int64)
    selected = (cycles >= 68) & (cycles <= 79)
    require_row_support(int(selected.sum()), PRODUCTION_MIN_ROWS, "Final production fit")
    return {"fit_rows": int(selected.sum()), "fit_cycles": [68, 79]}


def verify_global_reference() -> tuple[dict[str, Any], dict[str, Any]]:
    manifest, qa = v2.read_json(v2.OUT_MANIFEST), v2.read_json(v2.OUT_QA)
    v2.verify_signed(manifest, "signature", "prior global affine manifest")
    v2.verify_signed(qa, "signature", "prior global affine QA")
    state, receipt = v2.read_json(v2.OUT_STATE), v2.read_json(v2.OUT_RECEIPT)
    if (receipt.get("status") != "FIT_AND_OUTPUT_SAVED"
            or receipt.get("candidate_fits") != 1
            or receipt.get("state_sha256") != file_sha256(v2.OUT_STATE)
            or receipt.get("test_features_read") is not False
            or receipt.get("test_labels_read") is not False
            or receipt.get("upload_performed") is not False):
        raise GateError("Prior global fit receipt does not bind the saved state and boundaries")
    if (state.get("candidate") != "CURRENT_DIRECT_AFFINE_L1_E79"
            or state.get("baseline") != "CURRENT_DIRECT_POSITION"
            or state.get("intercept") != GLOBAL_A or state.get("slope") != GLOBAL_B
            or manifest.get("candidate") != "CURRENT_DIRECT_AFFINE_L1_E79"
            or manifest.get("intercept") != GLOBAL_A or manifest.get("slope") != GLOBAL_B
            or qa.get("status") != "PASS_LOCAL_OUTPUT_QA"):
        raise GateError("Prior global affine candidate identity/coefficients changed")
    return state, receipt


def identity_inputs() -> dict[str, Any]:
    """Read hashes and prediction/identity arrays only; never parse Train targets."""
    actual_inputs = snapshot_inputs()
    v2.verify_files(check_train=True)
    base_manifest, _qa, direct_freeze = v2.verify_base()
    test_ids, direct_e79, direct_check = v2.base_csv_values(base_manifest)
    with np.load(v2.PINS["base_archive"][0], allow_pickle=False) as archive:
        archive_ids = np.asarray(archive["record_id"])
        archive_prediction = np.asarray(archive["prediction"], dtype=np.float64)
        missing = np.asarray(archive["missing"], dtype=bool)
    if (archive_ids.dtype.kind != "U" or not np.array_equal(test_ids, archive_ids)
            or not np.array_equal(direct_e79, archive_prediction)
            or int(missing.sum()) != EXPECTED_MISSING_ROWS):
        raise GateError("Saved E79 direct IDs/predictions/missing mask fail identity checks")
    global_state, global_receipt = verify_global_reference()
    global_frame = pd.read_csv(v2.OUT_CSV, dtype={"record_id": str}, float_precision="round_trip")
    if list(global_frame.columns) != ["record_id", "threat_rank"] or len(global_frame) != EXPECTED_TEST_ROWS:
        raise GateError("Saved global affine CSV schema/count changed")
    global_ids = global_frame["record_id"].astype(str).to_numpy(dtype=str)
    global_e79 = pd.to_numeric(global_frame["threat_rank"], errors="raise").to_numpy(dtype=np.float64)
    if (not np.array_equal(test_ids, global_ids) or not np.isfinite(global_e79).all()
            or not np.array_equal(global_e79, apply_affine(direct_e79, GLOBAL_A, GLOBAL_B))):
        raise GateError("Saved prior global affine CSV is not exact round-trip replay")
    e67, e75 = v2.load_bundle(67), v2.load_bundle(75)
    v2.verify_prefixes(e67, e75, direct_freeze)
    plan = v2.stitch_oof_rows(e67, e75)
    rows = np.asarray([x[0] for x in plan], dtype=np.int64)
    origins = np.asarray([x[1] for x in plan], dtype=np.int8)
    indices = np.asarray([x[2] for x in plan], dtype=np.int64)
    ids = np.asarray([x[3] for x in plan], dtype=str)
    cycles = np.asarray([x[4] for x in plan], dtype=np.int16)
    direct, missing_oof = [], []
    for _row, origin, index, _rid, _cycle in plan:
        bundle = e67 if origin == 67 else e75
        direct.append(float(bundle["prediction"][index]))
        missing_oof.append(bool(bundle["missing_mask"][index]))
    direct = np.asarray(direct, dtype=np.float64)
    missing_oof = np.asarray(missing_oof, dtype=bool)
    if (len(rows) != EXPECTED_SUPPORT_ROWS or len(set(rows.tolist())) != len(rows)
            or len(set(ids.tolist())) != len(ids)
            or not np.array_equal(np.unique(cycles), np.arange(68, 80, dtype=np.int16))
            or not np.isfinite(direct).all() or np.any((direct < 0) | (direct > 100))):
        raise GateError("Unique OOF physical row/ID/cycle/direct prediction identity failed")
    return {
        "inputs": actual_inputs,
        "oof": {"rows": rows, "origins": origins, "indices": indices,
                "ids": ids, "cycles": cycles, "direct": direct,
                "missing": missing_oof},
        "e79": {"ids": test_ids, "direct": direct_e79,
                "missing": missing, "global": global_e79},
        "global_state": global_state, "global_receipt": global_receipt,
        "direct_identity": direct_check,
        "row_counts": {"unique_oof": int(len(rows)), "E67_68_75": int(np.sum((cycles >= 68) & (cycles <= 75))),
                        "E75_76_79": int(np.sum((cycles >= 76) & (cycles <= 79))),
                        "E79": int(len(test_ids)), "E79_missing": int(missing.sum())},
    }


def recency_weights(cycles: Any, reference_cycle: int) -> tuple[np.ndarray, dict[str, Any]]:
    c = np.asarray(cycles, dtype=np.int64)
    if c.ndim != 1 or not len(c) or np.any(c > reference_cycle):
        raise GateError("Invalid cycle vector for H8 recency weights")
    raw = np.exp2(-(reference_cycle - c).astype(np.float64) / HALF_LIFE)
    w = raw / raw.mean()
    if not np.isfinite(w).all() or np.any(w <= 0) or not np.isclose(w.mean(), 1.0, atol=1e-14, rtol=0):
        raise GateError("H8 weights failed finite, positive, mean-one contract")
    info = {
        "formula": "raw=2**(-(reference_cycle-cycle)/8); normalized=raw/mean(raw)",
        "reference_cycle": int(reference_cycle), "half_life_cycles": int(HALF_LIFE),
        "normalization": "arithmetic mean over exact fit rows equals one",
        "fit_rows": int(len(w)), "sum": float(w.sum()), "mean": float(w.mean()),
        "minimum": float(w.min()), "maximum": float(w.max()),
        "effective_sample_size": float(w.sum() ** 2 / np.dot(w, w)),
        "cycle_counts": {str(int(x)): int(np.sum(c == x)) for x in np.unique(c)},
        "weights_sha256": array_sha256(w, dtype="<f8"),
    }
    return w, info


def solve_weighted_nonnegative_l1(target: Any, prediction: Any,
                                  weights: Any, *, threads: int = THREADS) -> dict[str, Any]:
    y = np.asarray(target, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if (y.ndim != 1 or p.ndim != 1 or w.ndim != 1 or not len(y)
            or not (len(y) == len(p) == len(w))):
        raise GateError("Weighted affine LP inputs must be nonempty aligned vectors")
    if (not np.isfinite(y).all() or not np.isfinite(p).all() or not np.isfinite(w).all()
            or np.any((y < 0) | (y > 100)) or np.any((p < 0) | (p > 100))
            or np.any(w <= 0)):
        raise GateError("Weighted affine LP input range/finite/positive-weight check failed")
    n = len(y)
    ix = np.arange(n, dtype=np.int64)
    row = np.concatenate((ix, ix, ix, n + ix, n + ix, n + ix))
    col = np.concatenate((np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64), 2 + ix,
                          np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64), 2 + ix))
    val = np.concatenate((-np.ones(n), -p, -np.ones(n), np.ones(n), p, -np.ones(n)))
    a_ub = sparse.coo_matrix((val, (row, col)), shape=(2 * n, n + 2), dtype=np.float64).tocsr()
    b_ub = np.concatenate((-y, y))
    objective = np.concatenate((np.zeros(2, dtype=np.float64), w))
    bounds = [(None, None), (0.0, None), *((0.0, None) for _ in range(n))]
    started = time.perf_counter()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = linprog(objective, A_ub=a_ub, b_ub=b_ub, bounds=bounds,
                             method="highs", options={"threads": int(threads)})
        optimizer_warnings = [type(item.message).__name__ for item in caught
                              if not (issubclass(item.category, OptimizeWarning)
                                      and "threads" in str(item.message).casefold())]
    except Exception as exc:
        return {"success": False, "status": None, "message": f"{type(exc).__name__}: {exc}",
                "intercept": None, "slope": None, "weighted_l1_sum": None,
                "seconds": time.perf_counter() - started, "history_rows": n,
                "constraint_shape": [2 * n, n + 2], "constraint_nnz": int(a_ub.nnz),
                "optimizer_warnings": [type(exc).__name__]}
    elapsed = time.perf_counter() - started
    if optimizer_warnings or not result.success or result.x is None:
        return {"success": False, "status": int(result.status), "message": str(result.message),
                "intercept": None, "slope": None, "weighted_l1_sum": None,
                "seconds": elapsed, "history_rows": n,
                "constraint_shape": [2 * n, n + 2], "constraint_nnz": int(a_ub.nnz),
                "optimizer_warnings": optimizer_warnings}
    solution = np.asarray(result.x, dtype=np.float64)
    if solution.shape != (n + 2,) or not np.isfinite(solution).all() or solution[1] < -1e-9:
        return {"success": False, "status": int(result.status), "message": "invalid LP solution",
                "intercept": None, "slope": None, "weighted_l1_sum": None,
                "seconds": elapsed, "history_rows": n,
                "constraint_shape": [2 * n, n + 2], "constraint_nnz": int(a_ub.nnz),
                "optimizer_warnings": optimizer_warnings}
    a, b = float(solution[0]), max(0.0, float(solution[1]))
    loss = float(np.dot(w, np.abs(y - a - b * p)))
    lp_loss = float(result.fun)
    matched = np.isclose(loss, lp_loss, atol=max(1e-5, float(w.sum()) * 1e-9), rtol=1e-8)
    return {"success": bool(matched), "status": int(result.status),
            "message": str(result.message), "intercept": a if matched else None,
            "slope": b if matched else None, "weighted_l1_sum": loss if matched else None,
            "solver_objective": lp_loss, "objective_matches_recomputed": bool(matched),
            "seconds": elapsed, "history_rows": n,
            "constraint_shape": [2 * n, n + 2], "constraint_nnz": int(a_ub.nnz),
            "optimizer_warnings": optimizer_warnings, "threads": int(threads)}


def macro_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    return {"n": int(len(error)), "mae": float(np.mean(np.abs(error))),
            "mean_error": float(np.mean(error)),
            "p90_absolute_error": float(np.quantile(np.abs(error), .90))}


def subgroup_metrics(y: np.ndarray, pred: np.ndarray, cycles: np.ndarray,
                     missing: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    masks = {
        "present": ~missing,
        "missing": missing,
        "target_le_5": y <= 5.0,
        "target_ge_95": y >= 95.0,
    }
    for group, mask in masks.items():
        n = int(mask.sum())
        result[group] = ({"n": n, "status": "UNKNOWN_UNSUPPORTED", "mae": None}
                         if n < 200 else {"status": "SUPPORTED", **macro_metrics(y[mask], pred[mask])})
    result["by_cycle"] = {str(int(c)): macro_metrics(y[cycles == c], pred[cycles == c])
                          for c in np.unique(cycles)}
    return result


def comparison_metrics(y: np.ndarray, cycles: np.ndarray, missing: np.ndarray,
                       direct: np.ndarray, causal_global: np.ndarray,
                       recency: np.ndarray, global_present_only: np.ndarray,
                       prior_global: np.ndarray) -> dict[str, Any]:
    policies = {
        "raw_direct": direct,
        "causal_global_all_rows": causal_global,
        "causal_recency_h8_all_rows": recency,
        "causal_global_present_only": global_present_only,
        "prior_global_postselected_descriptive_only": prior_global,
    }
    return {name: {"overall": macro_metrics(y, pred),
                   "groups": subgroup_metrics(y, pred, cycles, missing)}
            for name, pred in policies.items()}


def verify_prefit_freeze() -> dict[str, Any]:
    freeze = read_json(PREFIT_FREEZE)
    body = verify_signed(freeze)
    if (body.get("experiment_id") != EXPERIMENT
            or body.get("status") != "PREFIT_FROZEN_IDENTITY_ONLY"
            or body.get("branch") != BRANCH):
        raise GateError("Prefit freeze identity or branch differs")
    if body.get("successor_of") != prior_source_artifact_binding():
        raise GateError("Successor freeze no longer binds the preserved R1 freeze/blocker")
    validate_live_source_checkpoint(body)
    return freeze


def prior_source_artifact_binding() -> dict[str, Any]:
    if not PRIOR_PREFIT_FREEZE.is_file() or not PRIOR_BLOCKER_REPORT.is_file():
        raise GateError("R1 prefit freeze and source-preparation blocker must be preserved")
    prior_freeze = read_json(PRIOR_PREFIT_FREEZE)
    prior_freeze_body = verify_signed(prior_freeze)
    blocker = read_json(PRIOR_BLOCKER_REPORT)
    blocker_body = verify_signed(blocker)
    freeze_sha = file_sha256(PRIOR_PREFIT_FREEZE)
    blocker_sha = file_sha256(PRIOR_BLOCKER_REPORT)
    if (freeze_sha != PRIOR_PREFIT_FREEZE_SHA256 or blocker_sha != PRIOR_BLOCKER_REPORT_SHA256
            or blocker.get("receipt_sha256") != PRIOR_BLOCKER_RECEIPT_SHA256
            or prior_freeze_body.get("experiment_id") != EXPERIMENT
            or prior_freeze_body.get("official_labels_accessed") is not False
            or prior_freeze_body.get("official_lp_starts") != 0
            or blocker_body.get("status") != "BLOCKED_RAM_IDENTITY_PREFLIGHT"
            or blocker_body.get("source_commit") != prior_freeze_body.get("source_commit")
            or blocker_body.get("prefit_freeze_sha256") != freeze_sha
            or blocker_body.get("official_target_rows_parsed") != 0
            or blocker_body.get("official_labels_accessed") is not False
            or blocker_body.get("official_lp_starts") != 0
            or blocker_body.get("candidate_csvs_created") is not False
            or blocker_body.get("kaggle_upload") is not False):
        raise GateError("R1 freeze/blocker lineage failed exact receipt or zero-fit checks")
    return {
        "freeze_path": str(PRIOR_PREFIT_FREEZE.relative_to(ROOT).as_posix()),
        "freeze_sha256": freeze_sha,
        "blocker_report_path": str(PRIOR_BLOCKER_REPORT.relative_to(ROOT).as_posix()),
        "blocker_report_sha256": blocker_sha,
        "blocker_receipt_sha256": blocker["receipt_sha256"],
        "prior_official_lp_starts": 0,
    }


def prepare_freeze() -> dict[str, Any]:
    commit, branch = assert_source_checkout()
    if PREFIT_FREEZE.exists():
        return verify_prefit_freeze()
    successor = prior_source_artifact_binding()
    if LEDGER.exists() and read_ledger(LEDGER):
        raise GateError("R2 LP ledger is nonempty; cannot certify that no old official slots were consumed")
    if not SYNTHETIC_QA.is_file():
        raise GateError("Synthetic QA must exist before prefit freeze")
    qa = read_json(SYNTHETIC_QA)
    qa_body = verify_signed(qa)
    qa_predecessor = {**successor,
                      "synthetic_qa_path": str(PRIOR_SYNTHETIC_QA.relative_to(ROOT).as_posix()),
                      "synthetic_qa_sha256": file_sha256(PRIOR_SYNTHETIC_QA)}
    if (qa_body.get("status") != "PASS_SYNTHETIC_ONLY"
            or qa_body.get("qa_revision") != "R2"
            or qa_body.get("predecessor") != qa_predecessor
            or qa_body.get("official_labels_accessed") is not False
            or qa_body.get("official_lp_starts") != 0):
        raise GateError("Synthetic QA does not prove candidate-blind zero-official scope")
    inputs = snapshot_inputs()
    identity = identity_inputs()
    body = {
        "schema_version": 1, "experiment_id": EXPERIMENT,
        "status": "PREFIT_FROZEN_IDENTITY_ONLY",
        "source_commit": commit, "branch": branch,
        "source_hashes": source_hashes(), "input_hashes": inputs,
        "successor_of": successor,
        "prior_official_lp_starts": successor["prior_official_lp_starts"],
        "synthetic_qa_file_sha256": file_sha256(SYNTHETIC_QA),
        "support_identity": {
            "rows_sha256": array_sha256(identity["oof"]["rows"]),
            "ids_sha256": array_sha256(identity["oof"]["ids"]),
            "origins_sha256": array_sha256(identity["oof"]["origins"]),
            "original_bundle_indices_sha256": array_sha256(identity["oof"]["indices"]),
            "cycles_sha256": array_sha256(identity["oof"]["cycles"]),
            "direct_prediction_sha256": array_sha256(identity["oof"]["direct"], dtype="<f8"),
            "missing_mask_sha256": array_sha256(identity["oof"]["missing"]),
            "row_counts": identity["row_counts"],
        },
        "global_affine_reference": {"intercept": GLOBAL_A, "slope": GLOBAL_B,
            "public_ref": GLOBAL_REF, "observed_public_mae": GLOBAL_PUBLIC_MAE,
            "state_sha256": inputs["global_calibrator_state"]["sha256"],
            "fit_receipt_sha256": inputs["global_fit_receipt"]["sha256"]},
        "fit_plan": list(FIT_IDS), "official_lp_start_cap": MAX_LP_STARTS,
        "official_labels_accessed": False, "official_lp_starts": 0,
        "test_or_sample_accessed": False, "csv_created": False,
        "kaggle_upload": False,
    }
    write_json_once(PREFIT_FREEZE, body)
    return verify_prefit_freeze()


def identity_preflight() -> dict[str, Any]:
    freeze = verify_prefit_freeze()
    body = verify_signed(freeze)
    resource = resource_snapshot(allow_blocked=True)
    if resource["status"] != "PASS":
        blocker = record_resource_blocker(body, resource)
        return {"status": "BLOCKED_RAM_IDENTITY_PREFLIGHT",
                "receipt_sha256": blocker["receipt_sha256"],
                "resource": resource, "official_labels_accessed": False,
                "official_lp_starts": 0}
    process = v2.one_job_gate()
    lock = shared_lock_probe()
    if lock["status"] != "AVAILABLE":
        raise GateError("Shared KASDAD heavy-job lock is held; no polling")
    identity = identity_inputs()
    support = diagnostic_support(identity)
    production = production_support(identity)
    if file_sha256(TRAIN) != TRAIN_SHA256:
        raise GateError("Regression Train identity changed")
    result = {
        "schema_version": 1, "experiment_id": EXPERIMENT,
        "status": "PASS_IDENTITY_ONLY_PREFIT",
        "source_commit": body["source_commit"], "freeze_sha256": file_sha256(PREFIT_FREEZE),
        "freeze_signature": freeze["receipt_sha256"],
        "successor_of": body["successor_of"],
        "source_checkpoint": validate_live_source_checkpoint(body),
        "source_hashes": body["source_hashes"],
        "source_paths": {name: str(path.resolve()) for name, path in SOURCE_FILES.items()},
        "input_hashes": body["input_hashes"],
        "support_identity": body["support_identity"],
        "stage_support": support,
        "production_support": production,
        "exact_commands": [
            "python scripts/reg_final_wave_followup_v2.py --synthetic-qa",
            "python -m pytest tests/test_reg_final_wave_followup_v2.py",
            "python scripts/reg_final_wave_followup_v2.py --prepare-freeze",
            "python scripts/reg_final_wave_followup_v2.py --identity-preflight",
            "python scripts/verify_reg_final_wave_followup_v2.py --verify-prefit",
        ],
        "row_counts": identity["row_counts"],
        "global_reference": body["global_affine_reference"],
        "resource": resource, "process_gate": process,
        "shared_heavy_job_lock": lock,
        "official_train_hash_verified": True,
        "official_target_rows_parsed": 0, "official_labels_accessed": False,
        "official_lp_starts": 0, "official_scoring_performed": False,
        "test_csv_or_sample_opened": False, "saved_e79_identity_arrays_read": True,
        "e79_probabilities_used_for_scoring": False, "csv_created": False,
        "kaggle_upload": False,
        "unknowns": ["Official Train selected targets remain unparsed.",
                     "No official LP result exists.",
                     "Root SOURCE_PASS and later production authorization are absent."],
    }
    if IDENTITY_PREFLIGHT.is_file():
        saved = read_json(IDENTITY_PREFLIGHT)
        saved_body = verify_signed(saved)
        if (saved_body.get("status") != "PASS_IDENTITY_ONLY_PREFIT"
                or saved_body.get("experiment_id") != EXPERIMENT
                or saved_body.get("source_commit") != body.get("source_commit")
                or saved_body.get("freeze_sha256") != file_sha256(PREFIT_FREEZE)
                or saved_body.get("freeze_signature") != freeze.get("receipt_sha256")
                or saved_body.get("source_hashes") != body.get("source_hashes")
                or saved_body.get("input_hashes") != body.get("input_hashes")
                or saved_body.get("support_identity") != body.get("support_identity")
                or saved_body.get("successor_of") != body.get("successor_of")
                or saved_body.get("official_target_rows_parsed") != 0
                or saved_body.get("official_lp_starts") != 0
                or saved_body.get("test_csv_or_sample_opened") is not False
                or saved_body.get("csv_created") is not False):
            raise GateError("Existing identity preflight receipt does not bind this exact freeze")
        return {"status": saved_body["status"], "receipt_reused": True,
                "receipt_sha256": saved["receipt_sha256"],
                "source_commit": saved_body["source_commit"],
                "fresh_resource_gate": resource, "fresh_process_gate": process,
                "fresh_shared_lock_probe": lock,
                "fresh_source_checkpoint": result["source_checkpoint"],
                "official_labels_accessed": False, "official_lp_starts": 0}
    write_json_once(IDENTITY_PREFLIGHT, result)
    return read_json(IDENTITY_PREFLIGHT)


def r16_h8_weights(cycles: Any, reference_cycle: int) -> tuple[np.ndarray, dict[str, Any]]:
    return recency_weights(cycles, reference_cycle)


def validate_oof_labels(labels: Any) -> np.ndarray:
    y = np.asarray(labels, dtype=np.float64)
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all() or np.any((y < 0) | (y > 100)):
        raise GateError("Selected REG targets must be finite, aligned, and within [0,100]")
    return y


def fit_signature_payload(fit_id: str, y: np.ndarray, p: np.ndarray,
                          cycles: np.ndarray, weights: np.ndarray | None,
                          row_identity: Mapping[str, Any],
                          freeze: Mapping[str, Any], approval_sha: str) -> dict[str, Any]:
    if (set(row_identity) != {"rows", "ids", "origins", "indices"}
            or any(np.asarray(value).ndim != 1 or len(value) != len(y)
                   for value in row_identity.values())
            or any(np.asarray(value).ndim != 1 or len(value) != len(y)
                   for value in (p, cycles))
            or (weights is not None and (np.asarray(weights).ndim != 1 or len(weights) != len(y)))):
        raise GateError("Fit signature arrays must be one-dimensional and aligned")
    return {
        "experiment_id": EXPERIMENT, "fit_id": fit_id,
        "fit_rows": int(len(y)), "cycles_sha256": array_sha256(cycles),
        "physical_rows_sha256": array_sha256(row_identity["rows"]),
        "record_ids_sha256": array_sha256(row_identity["ids"]),
        "prediction_origins_sha256": array_sha256(row_identity["origins"]),
        "original_bundle_indices_sha256": array_sha256(row_identity["indices"]),
        "targets_sha256": array_sha256(y, dtype="<f8"),
        "predictions_sha256": array_sha256(p, dtype="<f8"),
        "weights_sha256": array_sha256(weights, dtype="<f8") if weights is not None else None,
        "source_commit": freeze["source_commit"],
        "source_hashes": freeze["source_hashes"],
        "input_hashes": freeze["input_hashes"],
        "prefit_freeze_sha256": file_sha256(PREFIT_FREEZE),
        "root_source_pass_sha256": approval_sha,
        "fit_ids_allowlist": list(FIT_IDS),
    }


def _ledger_state(path: Path | None = None) -> dict[str, dict[str, Any]]:
    events = read_ledger(path)
    by_fit: dict[str, dict[str, Any]] = {}
    for event in events:
        fit_id = event.get("fit_id")
        if fit_id not in FIT_IDS:
            raise GateError(f"Unknown fit ID in ledger: {fit_id!r}")
        state = by_fit.setdefault(str(fit_id), {"RESERVE": [], "LP_STARTED": [],
                                                "LP_COMPLETE": [], "LP_FAILED": []})
        kind = event.get("event")
        if kind not in state:
            raise GateError(f"Unknown ledger event for {fit_id}: {kind!r}")
        state[kind].append(event)
    for fit_id, stages in by_fit.items():
        for event_kind, items in stages.items():
            if len(items) > 1:
                raise GateError(f"Duplicate {event_kind} events for {fit_id}")
        if stages["LP_COMPLETE"] and stages["LP_FAILED"]:
            raise GateError(f"Conflicting terminal LP events for {fit_id}")
        if (stages["LP_STARTED"] and not stages["RESERVE"]
                or (stages["LP_COMPLETE"] or stages["LP_FAILED"]) and not stages["LP_STARTED"]):
            raise GateError(f"Invalid LP event order for {fit_id}")
        if stages["RESERVE"] and stages["LP_STARTED"]:
            reservation = stages["RESERVE"][0]
            started = stages["LP_STARTED"][0]
            if (reservation.get("reservation_id") != started.get("reservation_id")
                    or reservation.get("fit_input") != started.get("fit_input")
                    or reservation.get("source_commit") != started.get("source_commit")
                    or reservation.get("root_source_pass_sha256") != started.get("root_source_pass_sha256")):
                raise GateError(f"Reservation/start bindings differ for {fit_id}")
        if stages["RESERVE"]:
            reservation = stages["RESERVE"][0]
            if (not reservation.get("reservation_id") or not isinstance(reservation.get("fit_input"), dict)
                    or not reservation.get("source_commit")
                    or not reservation.get("root_source_pass_sha256")):
                raise GateError(f"LP reservation lacks complete source/input bindings for {fit_id}")
            for terminal_kind in ("LP_COMPLETE", "LP_FAILED"):
                if stages[terminal_kind]:
                    terminal = stages[terminal_kind][0]
                    if (terminal.get("reservation_id") != reservation.get("reservation_id")
                            or terminal.get("fit_input") != reservation.get("fit_input")
                            or terminal.get("source_commit") != reservation.get("source_commit")
                            or terminal.get("root_source_pass_sha256") != reservation.get("root_source_pass_sha256")
                            or not terminal.get("state_sha256")):
                        raise GateError(f"Terminal event lacks reservation/input/state binding for {fit_id}")
    return by_fit


def _next_unreserved_fit(path: Path | None = None) -> str | None:
    state = _ledger_state(path)
    reserved = [fit_id for fit_id in FIT_IDS if fit_id in state and state[fit_id]["RESERVE"]]
    if reserved != list(FIT_IDS[:len(reserved)]):
        raise GateError("LP reservations are not a contiguous prefix of the frozen fit order")
    return FIT_IDS[len(reserved)] if len(reserved) < MAX_LP_STARTS else None


def _verify_root_source_pass(freeze: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    if not ROOT_SOURCE_PASS.is_file():
        raise GateError("ROOT_SOURCE_PASS is absent; no official labels or LP may start")
    approval = read_json(ROOT_SOURCE_PASS)
    validate_source_pass_body(approval, freeze, file_sha256(PREFIT_FREEZE))
    validate_live_source_checkpoint(freeze)
    return approval, file_sha256(ROOT_SOURCE_PASS)


def validate_source_pass_body(approval: Mapping[str, Any], freeze: Mapping[str, Any],
                              freeze_sha256: str) -> None:
    verify_signed(approval)
    if (approval.get("verdict") != "PASS" or approval.get("experiment_id") != EXPERIMENT
            or approval.get("source_commit") != freeze.get("source_commit")
            or approval.get("prefit_freeze_sha256") != freeze_sha256
            or approval.get("source_hashes") != freeze.get("source_hashes")
            or approval.get("input_hashes") != freeze.get("input_hashes")
            or approval.get("authorized_fit_ids") != list(DIAGNOSTIC_FIT_IDS)):
        raise GateError("ROOT_SOURCE_PASS does not bind this exact diagnostic source/input scope")


def _load_official_labels(identity: Mapping[str, Any], freeze: Mapping[str, Any],
                          approval_sha: str) -> tuple[np.ndarray, dict[str, Any]]:
    resource_snapshot()
    _approval, current_sha = _verify_root_source_pass(freeze)
    if current_sha != approval_sha:
        raise GateError("Source approval changed before target access")
    oof = identity["oof"]
    labels, _text_load, parsed_rows, audit = read_scored_query(
        v2.SOURCE, oof["rows"], oof["ids"], oof["cycles"], expected_total_rows=50608)
    y = validate_oof_labels(labels)
    if (not np.array_equal(parsed_rows, oof["rows"]) or len(y) != EXPECTED_SUPPORT_ROWS
            or audit.get("fit_target_rows_parsed") != 0
            or audit.get("sha256") != TRAIN_SHA256):
        raise GateError("Selected-row label reader did not preserve the frozen OOF identity boundary")
    return y, audit


def fit_row_identity(oof: Mapping[str, Any], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {name: np.asarray(oof[key])[mask] for name, key in (
        ("rows", "rows"), ("ids", "ids"), ("origins", "origins"), ("indices", "indices"))}


def _fit_file_path(fit_id: str) -> Path:
    return RUN_DIR / "fits" / f"{fit_id}.json"


def _validate_fit_state(path: Path, expected_payload: Mapping[str, Any]) -> dict[str, Any]:
    state = read_json(path)
    body = verify_signed(state)
    for key, value in expected_payload.items():
        if body.get(key) != value:
            raise GateError(f"Persisted LP state differs from current inputs at {key}: {path}")
    if body.get("status") not in {"LP_COMPLETE", "LP_FAILED_TERMINAL"}:
        raise GateError(f"Persisted LP state has no terminal status: {path}")
    return state


def _record_or_reuse_fit(fit_id: str, y: np.ndarray, p: np.ndarray,
                         cycles: np.ndarray, weights: np.ndarray | None,
                         row_identity: Mapping[str, Any],
                         freeze: Mapping[str, Any], approval_sha: str) -> dict[str, Any]:
    if fit_id not in FIT_IDS:
        raise GateError(f"LP ID outside frozen allowlist: {fit_id}")
    validate_live_source_checkpoint(freeze)
    resource_snapshot()
    existing = _ledger_state()
    stage = existing.get(fit_id)
    if any(len(np.asarray(value)) != len(y) for value in row_identity.values()):
        raise GateError("LP physical-row identity arrays do not align with fit rows")
    expected_payload = fit_signature_payload(fit_id, y, p, cycles, weights,
                                             row_identity, freeze, approval_sha)
    path = _fit_file_path(fit_id)
    reservation = stage["RESERVE"][0] if stage and stage["RESERVE"] else None
    started_event = stage["LP_STARTED"][0] if stage and stage["LP_STARTED"] else None
    if reservation is not None:
        if (reservation.get("fit_input") != expected_payload
                or reservation.get("source_commit") != freeze["source_commit"]
                or reservation.get("root_source_pass_sha256") != approval_sha):
            raise GateError(f"Existing reservation differs from current bound inputs: {fit_id}")
    if started_event is not None:
        if (started_event.get("reservation_id") != reservation.get("reservation_id")
                or started_event.get("fit_input") != expected_payload
                or started_event.get("source_commit") != freeze["source_commit"]
                or started_event.get("root_source_pass_sha256") != approval_sha):
            raise GateError(f"LP_STARTED event differs from current bound inputs: {fit_id}")
    if stage and (stage["LP_COMPLETE"] or stage["LP_FAILED"]):
        if not path.is_file():
            raise GateError(f"Terminal ledger event lacks exact state; no retry: {fit_id}")
        saved = _validate_fit_state(path, expected_payload)
        expected_event = "LP_COMPLETE" if saved["status"] == "LP_COMPLETE" else "LP_FAILED"
        terminal = stage[expected_event]
        if (not terminal or terminal[0].get("state_sha256") != file_sha256(path)
                or terminal[0].get("reservation_id") != reservation.get("reservation_id")
                or terminal[0].get("fit_input") != expected_payload
                or terminal[0].get("source_commit") != freeze["source_commit"]
                or terminal[0].get("root_source_pass_sha256") != approval_sha):
            raise GateError(f"Ledger terminal receipt does not bind saved LP state: {fit_id}")
        if saved["status"] == "LP_FAILED_TERMINAL":
            raise GateError(f"Prior LP failed terminally; no retry: {fit_id}")
        return saved
    if path.exists():
        if not reservation or not started_event:
            raise GateError(f"Saved LP state lacks durable start/reservation: {fit_id}")
        saved = _validate_fit_state(path, expected_payload)
        terminal_kind = "LP_COMPLETE" if saved["status"] == "LP_COMPLETE" else "LP_FAILED"
        if not stage[terminal_kind]:
            append_jsonl(LEDGER, {"event": terminal_kind, "fit_id": fit_id,
                "reservation_id": reservation["reservation_id"],
                "state_sha256": file_sha256(path), "source_commit": freeze["source_commit"],
                "root_source_pass_sha256": approval_sha, "fit_input": expected_payload,
                "recovered_after_state_publish": True,
                "at_utc": datetime.now(timezone.utc).isoformat(), "retry_allowed": False})
        if saved["status"] == "LP_FAILED_TERMINAL":
            raise GateError(f"Saved LP failure is terminal: {fit_id}")
        return saved
    if started_event is not None:
        raise GateError(f"LP_STARTED exists without provable terminal state; no retry: {fit_id}")
    if reservation is not None:
        reserved_sequence = [name for name in FIT_IDS
                             if (entry := _ledger_state().get(name)) and entry["RESERVE"]]
        if (reserved_sequence != list(FIT_IDS[:len(reserved_sequence)])
                or not reserved_sequence or reserved_sequence[-1] != fit_id):
            raise GateError(f"Unstarted reservation is not the last valid bound slot: {fit_id}")
        current_ledger = _ledger_state()
        if any(not current_ledger[name]["LP_COMPLETE"] for name in reserved_sequence[:-1]):
            raise GateError(f"Earlier LP slots are not successfully complete before resuming {fit_id}")
        reservation_id = str(reservation["reservation_id"])
    else:
        next_id = _next_unreserved_fit()
        if next_id != fit_id:
            raise GateError(f"Next permitted LP is {next_id}; refusing {fit_id}")
        current_ledger = _ledger_state()
        if any(name in current_ledger and current_ledger[name]["RESERVE"]
               and not current_ledger[name]["LP_COMPLETE"]
               for name in FIT_IDS if name != fit_id):
            raise GateError(f"A prior LP slot is not successfully complete before reserving {fit_id}")
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        reservation_id = str(uuid.uuid4())
        event_body = {"fit_id": fit_id, "reservation_id": reservation_id,
                      "source_commit": freeze["source_commit"],
                      "root_source_pass_sha256": approval_sha,
                      "fit_input": expected_payload,
                      "resource": resource_snapshot(), "thread_limit": THREADS,
                      "attempt": 1, "retry_allowed": False}
        append_jsonl(LEDGER, {"event": "RESERVE", **event_body,
                               "at_utc": datetime.now(timezone.utc).isoformat()})
    # Recheck durable authorization, source revision, resource, and process status
    # immediately before each unique LP start.
    _approval, current_approval_sha = _verify_root_source_pass(freeze)
    if current_approval_sha != approval_sha:
        raise GateError("Source authorization changed before LP start")
    resource = resource_snapshot()
    v2.one_job_gate()
    append_jsonl(LEDGER, {"event": "LP_STARTED", "fit_id": fit_id,
        "reservation_id": reservation_id, "source_commit": freeze["source_commit"],
        "root_source_pass_sha256": approval_sha,
        "fit_input": expected_payload, "resource": resource,
        "at_utc": datetime.now(timezone.utc).isoformat(), "retry_allowed": False})
    start = time.perf_counter()
    try:
        with threadpool_limits(limits=THREADS):
            if weights is None:
                result = solve_nonnegative_l1(y, p, threads=THREADS)
            else:
                result = solve_weighted_nonnegative_l1(y, p, weights, threads=THREADS)
        if not result.get("success"):
            raise GateError(f"HiGHS returned unsuccessful LP: {result}")
        fit_status = "LP_COMPLETE"
        solver_result = result
        failure = None
    except Exception as exc:
        fit_status = "LP_FAILED_TERMINAL"
        solver_result = {"success": False, "error_type": type(exc).__name__, "error": str(exc)}
        failure = {"type": type(exc).__name__, "message": str(exc)}
    elapsed = time.perf_counter() - start
    body = {**expected_payload, "status": fit_status, "solver_result": solver_result,
            "elapsed_seconds": elapsed, "weights_summary": None if weights is None else {
                "sum": float(np.sum(weights)), "min": float(np.min(weights)),
                "max": float(np.max(weights)), "effective_sample_size": float(weights.sum() ** 2 / np.dot(weights, weights))},
            "failure": failure, "saved_utc": datetime.now(timezone.utc).isoformat()}
    digest = write_json_once(path, body)
    terminal_event = {"event": "LP_COMPLETE" if fit_status == "LP_COMPLETE" else "LP_FAILED",
        "fit_id": fit_id, "reservation_id": reservation_id, "state_sha256": digest,
        "source_commit": freeze["source_commit"], "root_source_pass_sha256": approval_sha,
        "fit_input": expected_payload,
        "at_utc": datetime.now(timezone.utc).isoformat(), "retry_allowed": False}
    append_jsonl(LEDGER, terminal_event)
    saved = _validate_fit_state(path, expected_payload)
    if fit_status != "LP_COMPLETE":
        raise GateError(f"LP {fit_id} failed terminally; no retry")
    return saved


def _affine_coefficients(prediction: np.ndarray, intercept: float,
                         slope: float) -> tuple[np.ndarray, int]:
    raw = float(intercept) + float(slope) * np.asarray(prediction, dtype=np.float64)
    clipped = np.clip(raw, 0.0, 100.0)
    return clipped, int(np.sum((raw < 0.0) | (raw > 100.0)))


def _affine_values(prediction: np.ndarray, state: Mapping[str, Any]) -> tuple[np.ndarray, int]:
    result = state["solver_result"]
    return _affine_coefficients(prediction, result["intercept"], result["slope"])


def prediction_audit(direct: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    direct = np.asarray(direct, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if direct.shape != prediction.shape or not np.isfinite(direct).all() or not np.isfinite(prediction).all():
        raise GateError("Prediction audit received misaligned/nonfinite arrays")
    order = np.argsort(direct, kind="stable")
    changes = np.abs(prediction - direct)
    return {
        "rows": int(len(prediction)), "minimum": float(np.min(prediction)),
        "maximum": float(np.max(prediction)),
        "range_within_0_100": bool(np.all((prediction >= 0.0) & (prediction <= 100.0))),
        "monotone_non_decreasing_in_direct_score": bool(np.all(np.diff(prediction[order]) >= -1e-12)),
        "changed_rows": int(np.count_nonzero(changes > 0.0)),
        "mean_absolute_change_vs_direct": float(changes.mean()),
        "maximum_absolute_change_vs_direct": float(changes.max()),
    }


def subgroup_harm_flags(metrics: Mapping[str, Any], baseline: str = "causal_global_all_rows") -> dict[str, Any]:
    output = {}
    baseline_groups = metrics[baseline]["groups"]
    for policy, record in metrics.items():
        if policy == baseline:
            continue
        flags = {}
        for group in ("missing", "target_le_5", "target_ge_95"):
            base = baseline_groups[group]
            candidate = record["groups"][group]
            supported = base.get("status") == "SUPPORTED" and candidate.get("status") == "SUPPORTED"
            delta = (float(candidate["mae"] - base["mae"]) if supported else None)
            flags[group] = {"status": "SUPPORTED" if supported else "UNKNOWN_UNSUPPORTED",
                            "mae_delta_vs_causal_global": delta,
                            "risk_flag_gt_0_25": bool(supported and delta > 0.25)}
        output[policy] = flags
    return output


def mae_differences(metrics: Mapping[str, Any]) -> dict[str, Any]:
    direct = float(metrics["raw_direct"]["overall"]["mae"])
    global_mae = float(metrics["causal_global_all_rows"]["overall"]["mae"])
    return {
        "vs_raw_direct": {name: float(value["overall"]["mae"] - direct)
                          for name, value in metrics.items()},
        "vs_causal_global_all_rows": {name: float(value["overall"]["mae"] - global_mae)
                                      for name, value in metrics.items()},
    }


def _stage_masks(cycles: np.ndarray, cutoff: int, score_lo: int, score_hi: int):
    history = (cycles >= 68) & (cycles <= cutoff)
    score = (cycles >= score_lo) & (cycles <= score_hi)
    if not history.any() or not score.any() or np.any(history & score):
        raise GateError("Temporal fit/evaluation support is empty or overlaps")
    return history, score


def _read_saved_state(fit_id: str, y: np.ndarray, p: np.ndarray, cycles: np.ndarray,
                      weights: np.ndarray | None, row_identity: Mapping[str, Any],
                      freeze: Mapping[str, Any], approval_sha: str) -> dict[str, Any]:
    expected = fit_signature_payload(fit_id, y, p, cycles, weights,
                                     row_identity, freeze, approval_sha)
    path = _fit_file_path(fit_id)
    if not path.is_file():
        raise GateError(f"Required completed fit state missing: {fit_id}")
    state = _validate_fit_state(path, expected)
    if state["status"] != "LP_COMPLETE":
        raise GateError(f"Required fit is not complete: {fit_id}")
    return state


def _diagnostic_stage(stage_id: str, cutoff: int, score_lo: int, score_hi: int,
                      y: np.ndarray, identity: Mapping[str, Any], freeze: Mapping[str, Any],
                      approval_sha: str) -> dict[str, Any]:
    oof = identity["oof"]
    cycles, direct, missing = oof["cycles"], oof["direct"], oof["missing"]
    history, score = _stage_masks(cycles, cutoff, score_lo, score_hi)
    fit_y, fit_p, fit_cycles = y[history], direct[history], cycles[history]
    fit_identity = fit_row_identity(oof, history)
    d1_global_id = f"{stage_id}_GLOBAL"
    d1_h8_id = f"{stage_id}_RECENCY_H8"
    if stage_id == "D1":
        reference = 71
    else:
        reference = 75
    weights, weight_info = recency_weights(fit_cycles, reference)
    global_state = _record_or_reuse_fit(d1_global_id, fit_y, fit_p, fit_cycles,
                                        None, fit_identity, freeze, approval_sha)
    h8_state = _record_or_reuse_fit(d1_h8_id, fit_y, fit_p, fit_cycles,
                                    weights, fit_identity, freeze, approval_sha)
    y_score, p_score, cyc_score = y[score], direct[score], cycles[score]
    miss_score = missing[score]
    global_all, clip_global = _affine_values(p_score, global_state)
    h8_all, clip_h8 = _affine_values(p_score, h8_state)
    global_present = global_all.copy()
    global_present[miss_score] = p_score[miss_score]
    prior_global, clip_prior = _affine_coefficients(p_score, GLOBAL_A, GLOBAL_B)
    predictions = {"raw_direct": p_score.copy(), "causal_global_all_rows": global_all,
                   "causal_recency_h8_all_rows": h8_all,
                   "causal_global_present_only": global_present,
                   "prior_global_postselected_descriptive_only": prior_global}
    record = {
        "stage_id": stage_id, "fit_cycles": [68, cutoff], "score_cycles": [score_lo, score_hi],
        "fit_rows": int(history.sum()), "score_rows": int(score.sum()),
        "fit_ids_sha256": array_sha256(oof["ids"][history]),
        "score_ids_sha256": array_sha256(oof["ids"][score]),
        "score_targets_sha256": array_sha256(y_score, dtype="<f8"),
        "weights": weight_info,
        "global_coefficients": {"intercept": global_state["solver_result"]["intercept"],
                                 "slope": global_state["solver_result"]["slope"]},
        "recency_coefficients": {"intercept": h8_state["solver_result"]["intercept"],
                                  "slope": h8_state["solver_result"]["slope"]},
        "global_fit_state_sha256": file_sha256(_fit_file_path(d1_global_id)),
        "recency_fit_state_sha256": file_sha256(_fit_file_path(d1_h8_id)),
        "fit_row_identity": {name: array_sha256(value) for name, value in fit_identity.items()},
        "score_row_identity": {
            "physical_rows_sha256": array_sha256(oof["rows"][score]),
            "record_ids_sha256": array_sha256(oof["ids"][score]),
            "prediction_origins_sha256": array_sha256(oof["origins"][score]),
            "original_bundle_indices_sha256": array_sha256(oof["indices"][score]),
            "direct_predictions_sha256": array_sha256(direct[score], dtype="<f8"),
        },
        "clip_count": {"global": clip_global, "recency_h8": clip_h8,
                       "prior_global_descriptive": clip_prior},
        "metrics": comparison_metrics(y_score, cyc_score, miss_score,
                                      predictions["raw_direct"], predictions["causal_global_all_rows"],
                                      predictions["causal_recency_h8_all_rows"],
                                      predictions["causal_global_present_only"],
                                      predictions["prior_global_postselected_descriptive_only"]),
        "evidence_limits": {"labels_reused": True, "fresh_holdout": False,
                            "prior_global_is_descriptive_only": True,
                            "origin_transition_is_not_isolated_recency": True},
    }
    record["mae_differences"] = mae_differences(record["metrics"])
    record["prediction_audit"] = {
        name: prediction_audit(predictions["raw_direct"], values)
        for name, values in predictions.items()
    }
    record["supported_missing_tail_harm_flags"] = subgroup_harm_flags(record["metrics"])
    arrays = {"ids": oof["ids"][score], "rows": oof["rows"][score],
              "origins": oof["origins"][score], "indices": oof["indices"][score],
              "cycles": cyc_score, "targets": y_score, "missing": miss_score,
              **{f"prediction__{k}": value for k, value in predictions.items()}}
    data_path = RUN_DIR / f"{stage_id}_diagnostic_rows.npz"
    payload = _npz_bytes(**arrays)
    digest = _atomic_write_once(data_path, payload)
    record["diagnostic_rows_path"] = str(data_path.relative_to(ROOT).as_posix())
    record["diagnostic_rows_sha256"] = digest
    return record


def _npz_bytes(**arrays: np.ndarray) -> bytes:
    import io
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def _validate_source_approval() -> tuple[dict[str, Any], dict[str, Any], str]:
    freeze = verify_prefit_freeze()
    approval, approval_sha = _verify_root_source_pass(freeze)
    if file_sha256(SYNTHETIC_QA) != freeze["synthetic_qa_file_sha256"]:
        raise GateError("Candidate-blind synthetic QA changed after source freeze")
    preflight = read_json(IDENTITY_PREFLIGHT)
    verify_signed(preflight)
    if (preflight.get("status") != "PASS_IDENTITY_ONLY_PREFIT"
            or preflight.get("freeze_sha256") != file_sha256(PREFIT_FREEZE)
            or preflight.get("official_target_rows_parsed") != 0
            or preflight.get("official_lp_starts") != 0):
        raise GateError("Identity-only preflight absent or not source-bound")
    return freeze, approval, approval_sha


def run_diagnostics() -> dict[str, Any]:
    freeze, _approval, approval_sha = _validate_source_approval()
    with SharedHeavyJobLock():
        resource_snapshot()
        v2.one_job_gate()
        identity = identity_inputs()
        support = diagnostic_support(identity)
        y, label_audit = _load_official_labels(identity, freeze, approval_sha)
        if len(y) != EXPECTED_SUPPORT_ROWS:
            raise GateError("Official selected label support differs from frozen 20,903 rows")
        records = []
        settings = (("D1", 71, 72, 75), ("D2", 75, 76, 79))
        for stage_id, cutoff, score_lo, score_hi in settings:
            records.append(_diagnostic_stage(stage_id, cutoff, score_lo, score_hi,
                                             y, identity, freeze, approval_sha))
        pooled = _pooled_diagnostics(records)
        body = {
            "schema_version": 1, "experiment_id": EXPERIMENT,
            "status": "FOUR_CAUSAL_DIAGNOSTICS_COMPLETE",
            "source_commit": freeze["source_commit"],
            "root_source_pass_sha256": approval_sha,
            "prefit_freeze_sha256": file_sha256(PREFIT_FREEZE),
            "input_hashes": freeze["input_hashes"], "source_hashes": freeze["source_hashes"],
            "prelabel_stage_support": support,
            "train_target_audit": label_audit,
            "unique_evaluation_rows": int(sum(r["score_rows"] for r in records)),
            "windows": records, "pooled_unique_W5_W6": pooled,
            "diagnostic_lp_ids": list(DIAGNOSTIC_FIT_IDS),
            "official_lp_starts": sum(1 for e in read_ledger() if e.get("event") == "LP_STARTED"),
            "official_lp_start_cap": MAX_LP_STARTS,
            "final_production_lp_started": False,
            "base_or_preprocessing_fits": 0,
            "test_or_sample_accessed": False, "csv_created": False, "kaggle_upload": False,
        }
        write_json_once(DIAGNOSTICS, body)
        return read_json(DIAGNOSTICS)


def _pooled_diagnostics(records: list[dict[str, Any]]) -> dict[str, Any]:
    arrays = []
    for record in records:
        path = ROOT / record["diagnostic_rows_path"]
        if file_sha256(path) != record["diagnostic_rows_sha256"]:
            raise GateError("Diagnostic row bundle digest mismatch before unique pooling")
        with np.load(path, allow_pickle=False) as z:
            arrays.append({key: np.asarray(z[key]).copy() for key in z.files})
    ids = np.concatenate([a["ids"] for a in arrays])
    if len(np.unique(ids)) != len(ids):
        raise GateError("W5/W6 diagnostic evaluation IDs overlap")
    y = np.concatenate([a["targets"] for a in arrays])
    cycles = np.concatenate([a["cycles"] for a in arrays])
    missing = np.concatenate([a["missing"] for a in arrays])
    names = ("raw_direct", "causal_global_all_rows", "causal_recency_h8_all_rows",
             "causal_global_present_only", "prior_global_postselected_descriptive_only")
    policies = {name: np.concatenate([a[f"prediction__{name}"] for a in arrays]) for name in names}
    metrics = {name: {"overall": macro_metrics(y, p), "groups": subgroup_metrics(y, p, cycles, missing)}
               for name, p in policies.items()}
    prediction_audits = {name: prediction_audit(policies["raw_direct"], pred)
                         for name, pred in policies.items()}
    harm_flags = subgroup_harm_flags(metrics)
    global_mae = metrics["causal_global_all_rows"]["overall"]["mae"]
    recency_mae = metrics["causal_recency_h8_all_rows"]["overall"]["mae"]
    present_only_mae = metrics["causal_global_present_only"]["overall"]["mae"]
    window_deltas = {}
    for record in records:
        key = record["stage_id"]
        window_deltas[key] = {
            "recency_minus_global_mae": record["metrics"]["causal_recency_h8_all_rows"]["overall"]["mae"]
                                           - record["metrics"]["causal_global_all_rows"]["overall"]["mae"],
            "global_present_only_minus_global_mae": record["metrics"]["causal_global_present_only"]["overall"]["mae"]
                                                        - record["metrics"]["causal_global_all_rows"]["overall"]["mae"],
        }
    recency_gate = (global_mae - recency_mae >= 0.02
                    and all(v["recency_minus_global_mae"] <= 0.03 for v in window_deltas.values()))
    missing_gate = (present_only_mae <= global_mae
                    and all(v["global_present_only_minus_global_mae"] <= 0.03
                            for v in window_deltas.values()))
    return {
        "rows": int(len(y)), "unique_record_ids": int(len(np.unique(ids))),
        "metrics": metrics, "window_delta_mae": window_deltas,
        "mae_differences": mae_differences(metrics),
        "prediction_audit": prediction_audits,
        "supported_missing_tail_harm_flags": harm_flags,
        "predeclared_screen": {
            "recency_h8_support": {"required_pooled_gain": 0.02,
                "max_window_harm": 0.03, "status": "PASS_SCREEN" if recency_gate else "FAIL_SCREEN"},
            "global_present_only_support": {"required_pooled_no_harm": True,
                "max_window_harm": 0.03, "status": "PASS_SCREEN" if missing_gate else "FAIL_SCREEN"},
        },
        "pooled_mae_is_row_weighted": True, "window_maes_not_averaged": True,
        "public_score_used_as_gate": False,
    }


def _verify_production_pass(freeze: Mapping[str, Any], approval_sha: str) -> dict[str, Any]:
    if not ROOT_PRODUCTION_PASS.is_file():
        raise GateError("ROOT_PRODUCTION_PASS absent; no final production LP or candidate CSV")
    approval = read_json(ROOT_PRODUCTION_PASS)
    verify_signed(approval)
    diag_sha = file_sha256(DIAGNOSTICS)
    if (approval.get("verdict") != "PASS" or approval.get("experiment_id") != EXPERIMENT
            or approval.get("source_commit") != freeze["source_commit"]
            or approval.get("prefit_freeze_sha256") != file_sha256(PREFIT_FREEZE)
            or approval.get("root_source_pass_sha256") != approval_sha
            or approval.get("source_hashes") != freeze.get("source_hashes")
            or approval.get("input_hashes") != freeze.get("input_hashes")
            or approval.get("diagnostics_sha256") != diag_sha
            or approval.get("authorized_fit_ids") != ["FINAL_RECENCY_H8"]
            or approval.get("candidate_csvs_max") != 2
            or approval.get("authorized_candidates") != ["A", "B"]):
        raise GateError("ROOT_PRODUCTION_PASS does not bind exact final LP and CSV limit")
    return approval


def run_production_and_candidates() -> dict[str, Any]:
    freeze, _source_approval, source_approval_sha = _validate_source_approval()
    if not DIAGNOSTICS.is_file():
        raise GateError("Four diagnostic fits/results are required before production stage")
    diagnostic = read_json(DIAGNOSTICS)
    diagnostic_body = verify_signed(diagnostic)
    if (diagnostic_body.get("status") != "FOUR_CAUSAL_DIAGNOSTICS_COMPLETE"
            or diagnostic_body.get("source_commit") != freeze.get("source_commit")
            or diagnostic_body.get("root_source_pass_sha256") != source_approval_sha
            or diagnostic_body.get("prefit_freeze_sha256") != file_sha256(PREFIT_FREEZE)):
        raise GateError("Diagnostic report does not bind this source/authorization freeze")
    production_approval = _verify_production_pass(freeze, source_approval_sha)
    with SharedHeavyJobLock():
        resource_snapshot()
        v2.one_job_gate()
        identity = identity_inputs()
        production_rows = production_support(identity)
        y, label_audit = _load_official_labels(identity, freeze, source_approval_sha)
        oof = identity["oof"]
        selected = (oof["cycles"] >= 68) & (oof["cycles"] <= 79)
        if int(selected.sum()) != production_rows["fit_rows"]:
            raise GateError("Final production support changed after prelabel support check")
        weights, weight_info = recency_weights(oof["cycles"][selected], 79)
        final_state = _record_or_reuse_fit("FINAL_RECENCY_H8", y[selected], oof["direct"][selected],
                                           oof["cycles"][selected], weights,
                                           fit_row_identity(oof, selected), freeze,
                                           file_sha256(ROOT_SOURCE_PASS))
        final_receipt_sha = file_sha256(FINAL_STATE)
        app_body = {
            "experiment_id": EXPERIMENT, "status": "FINAL_RECENCY_APPLICATION_FROZEN",
            "prefit_freeze_sha256": file_sha256(PREFIT_FREEZE),
            "source_pass_sha256": file_sha256(ROOT_SOURCE_PASS),
            "production_pass_sha256": file_sha256(ROOT_PRODUCTION_PASS),
            "diagnostics_sha256": file_sha256(DIAGNOSTICS),
            "final_fit_state_sha256": final_receipt_sha,
            "input_hashes": freeze["input_hashes"], "source_hashes": freeze["source_hashes"],
            "test_identity_sha256": array_sha256(identity["e79"]["ids"]),
            "test_direct_prediction_sha256": array_sha256(identity["e79"]["direct"], dtype="<f8"),
            "test_missing_mask_sha256": array_sha256(identity["e79"]["missing"]),
            "threshold": None, "application_range": [0.0, 100.0],
            "test_labels_read": False, "sample_submission_read": False,
        }
        write_json_once(FINAL_APPLICATION_FREEZE, app_body)
        prior_direct = identity["e79"]["direct"]
        recency_e79, recency_clips = _affine_values(prior_direct, final_state)
        global_present = identity["e79"]["global"].copy()
        global_present[identity["e79"]["missing"]] = prior_direct[identity["e79"]["missing"]]
        a_ready = _write_candidate("A", OUTPUT_A, identity["e79"]["ids"], recency_e79,
            {"identity": "RECENCY_AFFINE_H8", "intercept": final_state["solver_result"]["intercept"],
             "slope": final_state["solver_result"]["slope"], "weights": weight_info,
             "apply_to": "all saved E79 direct predictions, including former fallback rows",
             "clip_at_prediction_only": True, "clip_count": recency_clips},
            identity, label_audit, final_receipt_sha)
        b_ready = _write_candidate("B", OUTPUT_B, identity["e79"]["ids"], global_present,
            {"identity": "GLOBAL_AFFINE_PRESENT_ONLY", "intercept": GLOBAL_A, "slope": GLOBAL_B,
             "apply_to": "saved signed global affine CSV present rows; raw direct restored on exact missing mask",
             "new_lp_calls": 0}, identity, label_audit, file_sha256(v2.OUT_RECEIPT))
        body = {"experiment_id": EXPERIMENT, "status": "CANDIDATES_READY_LOCAL_EXPLORATORY",
                "root_source_pass_sha256": source_approval_sha,
                "root_production_pass_sha256": file_sha256(ROOT_PRODUCTION_PASS),
                "diagnostics_sha256": file_sha256(DIAGNOSTICS),
                "final_fit_state_sha256": final_receipt_sha,
                "candidate_A": a_ready, "candidate_B": b_ready,
                "official_lp_starts": sum(e.get("event") == "LP_STARTED" for e in read_ledger()),
                "official_lp_start_cap": MAX_LP_STARTS,
                "official_labels_accessed": True, "test_labels_read": False,
                "sample_submission_read": False, "kaggle_upload": False}
        write_json_once(REPORT_DIR / "PRODUCTION_RESULT.json", body)
        return read_json(REPORT_DIR / "PRODUCTION_RESULT.json")


def _write_candidate(candidate: str, path: Path, ids: np.ndarray, pred: np.ndarray,
                     recipe: Mapping[str, Any], identity: Mapping[str, Any],
                     label_audit: Mapping[str, Any], state_sha: str) -> dict[str, Any]:
    ids = np.asarray(ids)
    pred = np.asarray(pred, dtype=np.float64)
    if (ids.dtype.kind != "U" or ids.shape != (EXPECTED_TEST_ROWS,)
            or len(np.unique(ids)) != EXPECTED_TEST_ROWS or pred.shape != ids.shape
            or not np.isfinite(pred).all() or np.any((pred < 0) | (pred > 100))):
        raise GateError(f"Candidate {candidate} ID/probability shape/range invalid")
    import io
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["record_id", "threat_rank"])
    writer.writerows((rid, format(float(x), ".17g")) for rid, x in zip(ids, pred, strict=True))
    csv_bytes = buffer.getvalue().encode("utf-8")
    csv_sha = _atomic_write_once(path, csv_bytes)
    frame = pd.read_csv(path, dtype={"record_id": str}, float_precision="round_trip")
    restored = pd.to_numeric(frame["threat_rank"], errors="raise").to_numpy(dtype=np.float64)
    if (list(frame.columns) != ["record_id", "threat_rank"] or len(frame) != EXPECTED_TEST_ROWS
            or not np.array_equal(frame["record_id"].astype(str).to_numpy(dtype=str), ids)
            or not np.array_equal(restored, pred)):
        raise GateError(f"Candidate {candidate} failed round-trip CSV validation")
    manifest_body = {
        "experiment_id": EXPERIMENT, "candidate": recipe["identity"],
        "status": "READY_LOCAL_EXPLORATORY", "csv_path": str(path.relative_to(ROOT).as_posix()),
        "csv_sha256": csv_sha, "rows": int(len(ids)),
        "ordered_ids_sha256": array_sha256(ids), "prediction_sha256": array_sha256(pred, dtype="<f8"),
        "recipe": dict(recipe), "support_rows": EXPECTED_SUPPORT_ROWS,
        "train_target_sha256": label_audit.get("query_target_values_sha256"),
        "fit_state_sha256": state_sha,
        "input_hashes": identity["inputs"],
        "test_or_sample_csv_opened": False, "kaggle_upload": False,
    }
    manifest_path = MANIFEST_A if candidate == "A" else MANIFEST_B
    write_json_once(manifest_path, manifest_body)
    return {"candidate": recipe["identity"], "status": "READY_LOCAL_EXPLORATORY",
            "csv_path": str(path.relative_to(ROOT).as_posix()), "csv_sha256": csv_sha,
            "manifest_sha256": file_sha256(manifest_path), "rows": len(ids)}


def _synthetic_stitch() -> list[tuple[int, int, int, str, int]]:
    e67 = {"query_rows": np.asarray([50, 10, 40, 20, 30, 60]),
           "query_record_ids": np.asarray(["id50", "id10", "id40", "id20", "id30", "id60"]),
           "query_cycles": np.asarray([78, 68, 77, 69, 76, 79])}
    e75 = {"query_rows": np.asarray([60, 40, 30, 50]),
           "query_record_ids": np.asarray(["id60", "id40", "id30", "id50"]),
           "query_cycles": np.asarray([79, 77, 76, 78])}
    return v2.stitch_oof_rows(e67, e75)


def synthetic_qa_body() -> dict[str, Any]:
    p = np.linspace(1.0, 90.0, 19, dtype=np.float64)
    y = 4.25 + .83 * p
    cycles = np.repeat(np.arange(68, 80, dtype=np.int16), 2)[:len(p)]
    w, weight_info = recency_weights(cycles, int(cycles.max()))
    weighted = solve_weighted_nonnegative_l1(y, p, w)
    scaled = solve_weighted_nonnegative_l1(y, p, 7.0 * w)
    unit = solve_weighted_nonnegative_l1(y, p, np.ones(len(y)))
    ordinary = solve_nonnegative_l1(y, p, threads=THREADS)
    if not all(x.get("success") for x in (weighted, scaled, unit, ordinary)):
        raise GateError("Synthetic known/unit/scale LP did not solve")
    pred_weighted = apply_affine(p, weighted["intercept"], weighted["slope"])
    pred_scaled = apply_affine(p, scaled["intercept"], scaled["slope"])
    pred_unit = apply_affine(p, unit["intercept"], unit["slope"])
    pred_ordinary = apply_affine(p, ordinary["intercept"], ordinary["slope"])
    if np.max(np.abs(pred_weighted - y)) > 1e-6:
        raise GateError("Synthetic weighted LP failed known-affine recovery")
    if (np.max(np.abs(pred_scaled - pred_weighted)) > 1e-6
            or abs(weighted["intercept"] - 4.25) > 1e-6
            or abs(weighted["slope"] - .83) > 1e-6):
        raise GateError("Synthetic recency-weight scale invariance or known coefficients failed")
    if (np.max(np.abs(pred_unit - pred_ordinary)) > 1e-7
            or abs(unit["weighted_l1_sum"] - ordinary["objective_l1_sum"]) > 1e-7):
        raise GateError("Unit-weight objective/predictions differ from pinned R17 helper")
    plan = _synthetic_stitch()
    expected = [(10, 67, 1, "id10", 68), (20, 67, 3, "id20", 69),
                (30, 75, 2, "id30", 76), (40, 75, 1, "id40", 77),
                (50, 75, 3, "id50", 78), (60, 75, 0, "id60", 79)]
    if plan != expected:
        raise GateError("Synthetic shuffled-origin physical-index stitch differs")
    duplicate_rejected = missing_rejected = False
    bad = {"query_rows": np.asarray([10, 10]), "query_record_ids": np.asarray(["x", "y"]),
           "query_cycles": np.asarray([68, 69])}
    try:
        v2.stitch_oof_rows(bad, {"query_rows": np.asarray([], dtype=np.int64),
                                  "query_record_ids": np.asarray([], dtype=str),
                                  "query_cycles": np.asarray([], dtype=np.int64)})
    except Exception:
        duplicate_rejected = True
    if not duplicate_rejected:
        raise GateError("Duplicate row identity was not rejected")
    try:
        v2.stitch_oof_rows({"query_rows": np.asarray([10]), "query_record_ids": np.asarray(["x"]),
                             "query_cycles": np.asarray([68])},
                           {"query_rows": np.asarray([20]), "query_record_ids": np.asarray(["x"]),
                             "query_cycles": np.asarray([76])})
    except Exception:
        missing_rejected = True
    if not missing_rejected:
        raise GateError("Missing/unequal duplicate-origin identity was not rejected")
    support_errors = []
    for count, minimum, label in ((999, 1000, "synthetic diagnostic"),
                                  (199, 200, "synthetic subgroup")):
        try:
            require_row_support(count, minimum, label)
        except GateError:
            support_errors.append(True)
    require_row_support(1000, 1000, "synthetic diagnostic")
    require_row_support(200, 200, "synthetic subgroup")
    if len(support_errors) != 2:
        raise GateError("Diagnostic/production support boundaries failed")
    p_route = np.asarray([10., 20., 30.])
    miss = np.asarray([False, True, False])
    causal_g, _ = _apply_affine_values_for_test(p_route, 5., .5)
    present_only = causal_g.copy()
    present_only[miss] = p_route[miss]
    if present_only[1] != 20. or causal_g[1] == present_only[1]:
        raise GateError("Causal GLOBAL_PRESENT_ONLY missing route failed")
    descriptive_coefficients = {"intercept": GLOBAL_A, "slope": GLOBAL_B,
                                 "role": "POSTSELECTED_DESCRIPTIVE_ONLY"}
    if descriptive_coefficients["role"] != "POSTSELECTED_DESCRIPTIVE_ONLY":
        raise GateError("Postselected global comparator classification failed")
    test_ids = np.asarray(["x-1", "x-2"], dtype="U3")
    test_values = np.asarray([0.0, 100.0])
    csv = _synthetic_csv_round_trip(test_ids, test_values)
    if not csv:
        raise GateError("Synthetic CSV exact finite/range round-trip failed")
    return {
        "schema_version": 2, "experiment_id": EXPERIMENT, "qa_revision": "R2",
        "status": "PASS_SYNTHETIC_ONLY",
        "predecessor": {**prior_source_artifact_binding(),
            "synthetic_qa_path": str(PRIOR_SYNTHETIC_QA.relative_to(ROOT).as_posix()),
            "synthetic_qa_sha256": file_sha256(PRIOR_SYNTHETIC_QA)},
        "official_labels_accessed": False, "official_lp_starts": 0,
        "test_or_sample_accessed": False, "csv_created": False, "kaggle_upload": False,
        "synthetic_solver_calls": 4,
        "known_weighted_affine": {"intercept": weighted["intercept"], "slope": weighted["slope"],
                                   "max_prediction_abs_error": float(np.max(np.abs(pred_weighted-y)))},
        "weight_summary": weight_info,
        "scale_invariance_max_prediction_delta": float(np.max(np.abs(pred_scaled-pred_weighted))),
        "unit_weight_r17_max_prediction_delta": float(np.max(np.abs(pred_unit-pred_ordinary))),
        "unit_weight_objective_delta": float(abs(unit["weighted_l1_sum"]-ordinary["objective_l1_sum"])),
        "shuffled_index_stitch_pass": True, "duplicate_identity_rejected": duplicate_rejected,
        "missing_identity_rejected": missing_rejected,
        "support_boundary_pass": True, "causal_present_only_route_pass": True,
        "postselected_comparator_separated": True, "csv_round_trip_pass": csv,
        "official_fit_plan": list(FIT_IDS), "official_fit_cap": MAX_LP_STARTS,
    }


def _apply_affine_values_for_test(p: np.ndarray, a: float, b: float) -> tuple[np.ndarray, int]:
    raw = a + b * p
    return np.clip(raw, 0, 100), int(np.sum((raw < 0) | (raw > 100)))


def _synthetic_csv_round_trip(ids: np.ndarray, pred: np.ndarray) -> bool:
    import io
    import tempfile
    if (ids.dtype.kind != "U" or pred.shape != ids.shape or not np.isfinite(pred).all()
            or np.any((pred < 0) | (pred > 100))):
        return False
    text = io.StringIO(newline="")
    writer = csv.writer(text, lineterminator="\n")
    writer.writerow(["record_id", "threat_rank"])
    writer.writerows((rid, format(float(value), ".17g")) for rid, value in zip(ids, pred, strict=True))
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "candidate.csv"
        path.write_text(text.getvalue(), encoding="utf-8", newline="")
        frame = pd.read_csv(path, dtype={"record_id": str}, float_precision="round_trip")
        values = pd.to_numeric(frame["threat_rank"], errors="raise").to_numpy(dtype=np.float64)
        return (list(frame.columns) == ["record_id", "threat_rank"]
                and np.array_equal(frame["record_id"].to_numpy(dtype=str), ids)
                and np.array_equal(values, pred))


def run_synthetic_qa() -> dict[str, Any]:
    body = synthetic_qa_body()
    if SYNTHETIC_QA.exists():
        saved = read_json(SYNTHETIC_QA)
        saved_body = verify_signed(saved)
        if canonical_bytes(saved_body) != canonical_bytes(body):
            raise GateError("Existing immutable synthetic QA differs from current deterministic result")
        return saved
    write_json_once(SYNTHETIC_QA, body)
    return read_json(SYNTHETIC_QA)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--synthetic-qa", action="store_true")
    group.add_argument("--prepare-freeze", action="store_true")
    group.add_argument("--identity-preflight", action="store_true")
    group.add_argument("--run-diagnostics", action="store_true")
    group.add_argument("--run-production", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.synthetic_qa:
            result = run_synthetic_qa()
        elif args.prepare_freeze:
            result = prepare_freeze()
        elif args.identity_preflight:
            result = identity_preflight()
        elif args.run_diagnostics:
            result = run_diagnostics()
        else:
            result = run_production_and_candidates()
        print(json.dumps({"status": result.get("status"), "experiment_id": EXPERIMENT,
                          "fit_starts": result.get("official_lp_starts", 0),
                          "official_labels_accessed": result.get("official_labels_accessed", False),
                          "receipt_sha256": result.get("receipt_sha256"),
                          "receipt_reused": result.get("receipt_reused"),
                          "resource": result.get("resource")},
                         sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED_OR_FAILED", "error_type": type(exc).__name__,
                          "message": str(exc), "retry_allowed": False}, sort_keys=True), file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
