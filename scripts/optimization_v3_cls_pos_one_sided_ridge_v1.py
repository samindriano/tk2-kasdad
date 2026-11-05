"""Source-gated, four-fit one-sided positional-ridge research executor.

The only authorized scientific campaign has two frozen temporal origins and two
penalty arms per origin. This file contains no Test, CSV-submission, or Kaggle
path. Official fit execution is blocked until a separately reviewed and
committed ROOT_SOURCE_PASS binds the exact committed sources and preflight.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import psutil
from scipy import sparse
from threadpoolctl import threadpool_limits

from scripts import evaluate_optimization_v3_cls_pos_frequency_ridge_v1 as OLD_EVAL
from scripts import optimization_v3_cls_pos_frequency_ridge_v1 as OLD_RUNNER
from src.models.pos_frequency_ridge_v1 import (
    FrequencyRidgeError,
    enforce_fit_cap,
    fit_resume_decision,
    sha256_array,
)
from src.models.optimization_v3_positional_text_pilot_v1 import sha256_csr
from src.models.pos_one_sided_ridge_v1 import (
    CAMPAIGN_ID,
    FIT_IDS,
    ARMS,
    ORIGINS,
    build_penalty_vectors,
    fit_offset,
)


PROTOCOL_R1_REL = "experiments/optimization_v3/cls_pos_one_sided_ridge_v1/protocol_r1.json"
PROTOCOL_REL = "experiments/optimization_v3/cls_pos_one_sided_ridge_v1/protocol_r2.json"
PROTOCOL_CORRECTION_REL = "experiments/optimization_v3/cls_pos_one_sided_ridge_v1/PROTOCOL_R2_CORRECTION.md"
EXPERIMENT_ROOT = "experiments/optimization_v3/cls_pos_one_sided_ridge_v1"
REPORT_ROOT = "reports/optimization_v3/cls_pos_one_sided_ridge_v1"
RUN_ROOT = "experiments/runs/optimization_v3_cls_pos_one_sided_ridge_v1"
SOURCE_PREFLIGHT_REL = f"{REPORT_ROOT}/SOURCE_PREFLIGHT.json"
SOURCE_FREEZE_REL = f"{EXPERIMENT_ROOT}/SOURCE_FREEZE.json"
ROOT_SOURCE_PASS_REL = f"{REPORT_ROOT}/ROOT_SOURCE_PASS.json"
LEDGER_REL = f"{RUN_ROOT}/FIT_LEDGER.jsonl"
SHARED_LOCK = Path(tempfile.gettempdir()) / "kasdad_single_heavy_job.lock"
TRAIN_REL = "data/raw/classification/Train.csv"
TRAIN_SHA256 = "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c"
PRIOR_SHARED_LEDGER_REL = "experiments/runs/optimization_v3_research_next_v1/budget.jsonl"
PRIOR_SHARED_LEDGER_SHA256 = "6fd0ca3d43cb8145e2c0bc50b278a5eb4b424d71500a80c4414332e87a8aa57a"

SOURCE_FILES = (
    PROTOCOL_R1_REL,
    PROTOCOL_REL,
    PROTOCOL_CORRECTION_REL,
    "src/models/pos_one_sided_ridge_v1.py",
    "scripts/optimization_v3_cls_pos_one_sided_ridge_v1.py",
    "scripts/evaluate_optimization_v3_cls_pos_one_sided_ridge_v1.py",
    "scripts/verify_optimization_v3_cls_pos_one_sided_ridge_v1.py",
    "tests/test_pos_one_sided_ridge_v1.py",
    "requirements.txt",
    "requirements-research.txt",
    "src/models/pos_frequency_ridge_v1.py",
    "src/models/optimization_v3_positional_text_pilot_v1.py",
    "scripts/optimization_v3_cls_pos_frequency_ridge_v1.py",
    "scripts/evaluate_optimization_v3_cls_pos_frequency_ridge_v1.py",
    "scripts/verify_optimization_v3_cls_pos_frequency_ridge_v1.py",
    "scripts/verify_optimization_v3_cls_pos_frequency_ridge_eval_v1.py",
    "scripts/optimization_v3_posoffset_a02_endgame_v1.py",
)

MIB = 1024**2
GIB = 1024**3
RESOURCE_POLICIES = {
    "source_preflight": {
        "minimum_available_bytes": 1536 * MIB,
        "estimated_additional_bytes": 512 * MIB,
        "minimum_projected_free_bytes": 768 * MIB,
        "minimum_actual_free_at_boundary_bytes": 768 * MIB,
    },
    "saved_replay": {
        "minimum_available_bytes": 1536 * MIB,
        "estimated_additional_bytes": 512 * MIB,
        "minimum_projected_free_bytes": 768 * MIB,
        "minimum_actual_free_at_boundary_bytes": 768 * MIB,
    },
    "small_gamma_fit": {
        "minimum_available_bytes": 2 * GIB,
        "estimated_additional_bytes": 1 * GIB,
        "minimum_projected_free_bytes": 1 * GIB,
        "minimum_actual_free_at_boundary_bytes": 1 * GIB,
    },
}
THREADS = 2
MAX_FIT_RSS = 6 * GIB


class OneSidedRunnerError(RuntimeError):
    """Raised when a source, identity, resource, ledger, or fit boundary fails."""


def _path(relative: str) -> Path:
    return ROOT / relative


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise OneSidedRunnerError(f"Expected a JSON object: {path}")
    return value


def write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(dict(payload), ensure_ascii=False, allow_nan=False,
                         sort_keys=True, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise OneSidedRunnerError(f"Refusing to overwrite immutable evidence: {path}")
        return
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def write_bytes_exclusive(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def append_event(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        stream.write(_canonical_bytes(dict(payload)) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise OneSidedRunnerError(f"Invalid fit ledger JSON at line {number}") from exc
        if not isinstance(value, dict):
            raise OneSidedRunnerError(f"Fit ledger row {number} is not an object")
        events.append(value)
    return events


def git_text(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, text=True,
                            capture_output=True, check=False)
    if result.returncode:
        raise OneSidedRunnerError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def require_committed(relative: str) -> dict[str, str]:
    blob = git_text("rev-parse", f"HEAD:{relative}")
    if blob != git_text("hash-object", relative):
        raise OneSidedRunnerError(f"Source must be committed and unchanged: {relative}")
    return {"git_blob": blob, "commit": git_text("log", "-1", "--format=%H", "--", relative)}


def source_hashes(*, require_git: bool) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = _path(relative)
        if not path.is_file():
            raise OneSidedRunnerError(f"Required executable source is missing: {relative}")
        result[relative] = file_sha256(path)
        if require_git:
            require_committed(relative)
    return result


def historical_input_paths() -> list[str]:
    """Return the exact verified target-free dependency inventory from the R8 evaluator."""
    relatives = set(OLD_EVAL.expected_input_paths())
    relatives.update({
        "reports/optimization_v3/cls_pos_frequency_ridge_v1/EVALUATION_RESULTS_R1.json",
        "reports/optimization_v3/cls_pos_frequency_ridge_v1/MEAN_ONE_ENERGY_DIAGNOSTIC_R1_2.json",
        "reports/optimization_v3/cls_pos_frequency_ridge_v1/MEAN_ONE_ENERGY_DIAGNOSTIC_R1_2.md",
        "reports/optimization_v3/cls_pos_frequency_ridge_v1/MEAN_ONE_ENERGY_DIAGNOSTIC_R1_2_VERIFY.json",
        "experiments/optimization_v3/cls_pos_frequency_ridge_v1/MEAN_ONE_ENERGY_DIAGNOSTIC_R1_2_FREEZE.json",
        "experiments/optimization_v3/cls_pos_frequency_ridge_v1/MEAN_ONE_ENERGY_DIAGNOSTIC_R1_2_VERIFY_FREEZE.json",
    })
    for relative in relatives:
        lower = relative.replace("\\", "/").lower()
        if (lower.endswith("/test.csv") or "samplesubmission" in lower
                or lower.endswith("/submission.csv") or "/kaggle/" in lower):
            raise OneSidedRunnerError(f"Test/submission path is forbidden in source inventory: {relative}")
    return sorted(relatives)


def input_hashes() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in historical_input_paths():
        path = _path(relative)
        if not path.is_file():
            raise OneSidedRunnerError(f"Frozen historical input is missing: {relative}")
        result[relative] = file_sha256(path)
    return result


def resource_snapshot(policy_name: str) -> dict[str, Any]:
    policy = RESOURCE_POLICIES[policy_name]
    memory = psutil.virtual_memory()
    process = psutil.Process().memory_info()
    rss = int(process.rss)
    available = int(memory.available)
    estimate = int(policy["estimated_additional_bytes"])
    peak = getattr(process, "peak_wset", None)
    projected = max(0, available - estimate)
    return {
        "policy": policy_name,
        "available_bytes": available,
        "total_bytes": int(memory.total),
        "process_rss_bytes": rss,
        "os_peak_working_set_bytes": int(peak) if peak is not None else None,
        "os_peak_working_set_source": "psutil.Process().memory_info().peak_wset" if peak is not None else None,
        "estimated_additional_bytes": estimate,
        "projected_free_bytes": projected,
        "minimum_available_bytes": int(policy["minimum_available_bytes"]),
        "minimum_projected_free_bytes": int(policy["minimum_projected_free_bytes"]),
        "minimum_actual_free_at_boundary_bytes": int(policy["minimum_actual_free_at_boundary_bytes"]),
        "threads": THREADS,
        "jobs": 1,
        "initial_ram_pass": available >= int(policy["minimum_available_bytes"]),
        "projected_reserve_pass": projected >= int(policy["minimum_projected_free_bytes"]),
        "rss_pass": policy_name != "small_gamma_fit" or rss + estimate <= MAX_FIT_RSS,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "samples_are_peak_measurements": False,
    }


def require_resource(policy_name: str, stage: str) -> dict[str, Any]:
    resource = resource_snapshot(policy_name)
    if not all(resource[key] for key in ("initial_ram_pass", "projected_reserve_pass", "rss_pass")):
        raise OneSidedRunnerError(f"Resource gate failed at {stage}: {resource}")
    return resource


def require_actual_boundary(policy_name: str, stage: str) -> dict[str, Any]:
    resource = resource_snapshot(policy_name)
    floor = resource["minimum_actual_free_at_boundary_bytes"]
    if resource["available_bytes"] < floor:
        raise OneSidedRunnerError(f"Actual free RAM fell below {floor} bytes at {stage}: {resource}")
    return resource


def read_identity_frame(train_path: Path, expected_sha256: str | None = None,
                        expected_rows: int | None = None,
                        cycle_bounds: tuple[int, int] | None = None) -> pd.DataFrame:
    if expected_sha256 is not None and file_sha256(train_path) != expected_sha256:
        raise OneSidedRunnerError("Official Train byte hash differs from the frozen source")
    frame = pd.read_csv(train_path, usecols=["record_id", "cycle_code"],
                        dtype={"record_id": "string"})
    if ((expected_rows is not None and len(frame) != expected_rows) or frame.record_id.isna().any()
            or frame.record_id.duplicated().any()):
        raise OneSidedRunnerError("Official Train identity schema/count/uniqueness check failed")
    frame["cycle_code"] = pd.to_numeric(frame["cycle_code"], errors="raise").astype(np.int64)
    if cycle_bounds is not None and (int(frame.cycle_code.min()), int(frame.cycle_code.max())) != cycle_bounds:
        raise OneSidedRunnerError(f"Train cycle coverage differs from expected interval {cycle_bounds}")
    return frame


def _read_identity_frame() -> pd.DataFrame:
    return read_identity_frame(_path(TRAIN_REL), TRAIN_SHA256, 50608, (47, 79))


def require_matrix_manifest_digest(origin: int, actual_sha256: str,
                                   expected_sha256: str) -> None:
    """Require the one exact manifest digest pinned by the current protocol."""
    if actual_sha256 != expected_sha256:
        raise OneSidedRunnerError(
            f"E{origin} matrix manifest digest mismatch: actual={actual_sha256}, "
            f"expected={expected_sha256}"
        )


def read_fit_labels(train_path: Path, expected_train_sha256: str,
                    fit_rows: np.ndarray, expected_ids: np.ndarray,
                    expected_target_sha256: str) -> np.ndarray:
    """Read only labels for the preverified fit rows after the start reservation."""
    if file_sha256(train_path) != expected_train_sha256:
        raise OneSidedRunnerError("Train bytes differ before fit-label access")
    rows = np.asarray(fit_rows, dtype=np.int64)
    ids = np.asarray(expected_ids).astype(str)
    if (rows.ndim != 1 or ids.shape != rows.shape or len(np.unique(rows)) != len(rows)
            or np.any(rows < 0)):
        raise OneSidedRunnerError("Fit row/ID loader inputs are invalid")
    position = {int(row): index for index, row in enumerate(rows.tolist())}
    target = np.empty(len(rows), dtype=np.int8)
    seen = np.zeros(len(rows), dtype=bool)
    with train_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = __import__("csv").reader(stream)
        header = next(reader, None)
        if not header or "record_id" not in header or "artifact_signal" not in header:
            raise OneSidedRunnerError("Train loader schema lacks record_id/artifact_signal")
        id_col, target_col = header.index("record_id"), header.index("artifact_signal")
        for row_index, record in enumerate(reader):
            out = position.get(row_index)
            if out is None:
                continue
            if max(id_col, target_col) >= len(record) or record[id_col] != ids[out]:
                raise OneSidedRunnerError("Fit label row ID does not match its verified identity")
            try:
                target[out] = int(record[target_col])
            except (TypeError, ValueError) as exc:
                raise OneSidedRunnerError("Fit target loader encountered a non-integer label") from exc
            seen[out] = True
    if not seen.all() or not np.isin(target, (0, 1)).all() or sha256_array(target) != expected_target_sha256:
        raise OneSidedRunnerError("Fit labels are incomplete, non-binary, or differ from frozen target hash")
    return target


def check_official_identities() -> dict[str, Any]:
    """Shared pre-target check called by both --identity-preflight and fit execution."""
    train = _path(TRAIN_REL)
    if file_sha256(train) != TRAIN_SHA256:
        raise OneSidedRunnerError("Official Train hash mismatch before identity-only read")
    historical = OLD_RUNNER._verify_historical_freezes()
    frame = _read_identity_frame()
    protocol = read_json(_path(PROTOCOL_REL))
    origins: dict[str, Any] = {}
    for origin in ORIGINS:
        key = str(origin)
        result = OLD_RUNNER._verify_origin_identity_and_files(frame, origin)
        expected = protocol["origins"][key]
        require_matrix_manifest_digest(
            origin, result["matrix_manifest_sha256"], expected["matrix_manifest_sha256"]
        )
        if (result["fit_rows"] != expected["fit_rows"]
                or result["score_rows"] != expected["query_rows"]
                or result["prediction_bundle_sha256"] != expected["query_prediction_bundle_sha256"]):
            raise OneSidedRunnerError(f"E{origin} row/cache identity differs from one-sided protocol")
        map_artifact = _path(f"experiments/runs/optimization_v3_cls_pos_frequency_ridge_v1/maps/penalty_map_E{origin}.npz")
        map_receipt_path = _path(f"experiments/runs/optimization_v3_cls_pos_frequency_ridge_v1/maps/penalty_map_E{origin}.json")
        map_receipt = read_json(map_receipt_path)
        if map_receipt.get("artifact_sha256") != file_sha256(map_artifact):
            raise OneSidedRunnerError(f"E{origin} saved frequency map receipt does not match its artifact")
        origins[key] = {
            "identity_and_cache": result,
            "saved_aligned_map_artifact_sha256": file_sha256(map_artifact),
            "saved_aligned_map_receipt_sha256": file_sha256(map_receipt_path),
            "target_values_opened": False,
        }
    shared_ledger = _path(PRIOR_SHARED_LEDGER_REL)
    shared_hash = file_sha256(shared_ledger) if shared_ledger.is_file() else None
    if shared_hash != PRIOR_SHARED_LEDGER_SHA256:
        raise OneSidedRunnerError("Prior shared fit ledger changed since the approved checkpoint")
    r1 = read_json(_path("reports/optimization_v3/cls_pos_frequency_ridge_v1/EVALUATION_RESULTS_R1.json"))
    if (r1.get("mechanism_gate", {}).get("pass") is not False
            or r1.get("competition_gate", {}).get("pass") is not False):
        raise OneSidedRunnerError("Historical failed-gate evidence changed; stop for review")
    r1_views = r1["views"]
    historical_negative = {
        name: {
            "one_sided_parent_aligned_vs_uniform": float(
                r1_views[name]["component_full"]["ALIGNED"]["macro_f1"]
                - r1_views[name]["component_full"]["UNIFORM"]["macro_f1"]),
            "competition_gain_vs_primary": float(r1_views[name]["competition_gains_macro_f1"]["full"]),
        }
        for name in ("E67_H12", "E75_W6", "UNIQUE_E67_W4_W5_PLUS_E75_W6")
    }
    return {
        "train_sha256": TRAIN_SHA256,
        "train_rows": len(frame),
        "train_cycle_min": int(frame.cycle_code.min()),
        "train_cycle_max": int(frame.cycle_code.max()),
        "origins": origins,
        "historical_freeze_checks": historical,
        "prior_shared_ledger_sha256": shared_hash,
        "prior_mean_one_negative_evidence": historical_negative,
        "target_values_opened": False,
        "target_npz_members_requested": [],
    }


def run_identity_preflight() -> dict[str, Any]:
    """Real target-free integration pass under the shared lock; never reads y members."""
    with OLD_EVAL.SharedJobLock():
        initial = require_resource("source_preflight", "before identity preflight")
        identities = check_official_identities()
        boundary = require_actual_boundary("source_preflight", "identity preflight boundary")
        sources = source_hashes(require_git=True)
        inputs = input_hashes()
        result = {
            "schema_version": 1,
            "campaign_id": CAMPAIGN_ID,
            "status": "PASS_TARGET_FREE_IDENTITY_PREFLIGHT",
            "source_commit": git_text("rev-parse", "HEAD"),
            "source_hashes": sources,
            "input_hashes": inputs,
            "identity_checks": identities,
            "resource_snapshot": {
                "policy": RESOURCE_POLICIES["source_preflight"],
                "initial_under_shared_lock": initial,
                "boundary": boundary,
                "samples_are_peak_measurements": False,
            },
            "shared_job_lock_path": str(SHARED_LOCK),
            "target_values_opened": False,
            "target_npz_members_requested": [],
            "official_fit_started": False,
            "query_predictions_created": False,
            "test_or_samplesubmission_accessed": False,
            "kaggle_accessed": False,
        }
        write_json_once(_path(SOURCE_PREFLIGHT_REL), result)
        return result


def build_source_freeze() -> dict[str, Any]:
    preflight_path = _path(SOURCE_PREFLIGHT_REL)
    if not preflight_path.is_file():
        raise OneSidedRunnerError("Source preflight is required before source freeze")
    if git_text("rev-parse", f"HEAD:{SOURCE_PREFLIGHT_REL}") != git_text("hash-object", SOURCE_PREFLIGHT_REL):
        raise OneSidedRunnerError("Source preflight must be committed before freezing its source")
    preflight = read_json(preflight_path)
    sources = source_hashes(require_git=True)
    inputs = input_hashes()
    if (preflight.get("status") != "PASS_TARGET_FREE_IDENTITY_PREFLIGHT"
            or preflight.get("source_hashes") != sources
            or preflight.get("input_hashes") != inputs
            or preflight.get("identity_checks", {}).get("target_values_opened") is not False):
        raise OneSidedRunnerError("Target-free preflight no longer binds current sources/inputs")
    body: dict[str, Any] = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "status": "FROZEN_SOURCE_ONLY_NO_FIT_AUTHORIZATION",
        "source_commit": git_text("rev-parse", "HEAD"),
        "source_hashes": sources,
        "input_hashes": inputs,
        "source_preflight_sha256": file_sha256(preflight_path),
        "fit_ids": list(FIT_IDS),
        "fit_cap": len(FIT_IDS),
        "official_fits_authorized": False,
        "prediction_manifest_committed_before_query_labels": True,
        "test_or_samplesubmission_accessed": False,
        "kaggle_accessed": False,
    }
    body["freeze_signature"] = canonical_sha256(body)
    return body


def write_source_freeze() -> dict[str, Any]:
    value = build_source_freeze()
    write_json_once(_path(SOURCE_FREEZE_REL), value)
    return value


def verify_source_freeze(*, require_frozen_commit: bool) -> dict[str, Any]:
    freeze_path = _path(SOURCE_FREEZE_REL)
    if require_frozen_commit:
        if git_text("rev-parse", f"HEAD:{SOURCE_FREEZE_REL}") != git_text("hash-object", SOURCE_FREEZE_REL):
            raise OneSidedRunnerError("Source freeze must be committed and unchanged")
    freeze = read_json(freeze_path)
    body = dict(freeze)
    claimed = body.pop("freeze_signature", None)
    if claimed != canonical_sha256(body):
        raise OneSidedRunnerError("Source freeze canonical signature mismatch")
    if freeze.get("status") != "FROZEN_SOURCE_ONLY_NO_FIT_AUTHORIZATION":
        raise OneSidedRunnerError("Source freeze status is not the reviewed source-only state")
    if source_hashes(require_git=require_frozen_commit) != freeze.get("source_hashes"):
        raise OneSidedRunnerError("Source hashes differ from the committed source freeze")
    if input_hashes() != freeze.get("input_hashes"):
        raise OneSidedRunnerError("Historical input hashes differ from the source freeze")
    if file_sha256(_path(SOURCE_PREFLIGHT_REL)) != freeze.get("source_preflight_sha256"):
        raise OneSidedRunnerError("Source preflight hash differs from source freeze")
    return freeze


def require_root_source_pass() -> dict[str, Any]:
    root_pass_path = _path(ROOT_SOURCE_PASS_REL)
    if not root_pass_path.is_file():
        raise OneSidedRunnerError("Official fits are blocked: ROOT_SOURCE_PASS.json is absent")
    if git_text("rev-parse", f"HEAD:{ROOT_SOURCE_PASS_REL}") != git_text("hash-object", ROOT_SOURCE_PASS_REL):
        raise OneSidedRunnerError("ROOT_SOURCE_PASS.json must be committed before official fits")
    passed = read_json(root_pass_path)
    freeze = verify_source_freeze(require_frozen_commit=True)
    if (passed.get("status") != "ROOT_SOURCE_PASS"
            or passed.get("campaign_id") != CAMPAIGN_ID
            or passed.get("source_freeze_sha256") != file_sha256(_path(SOURCE_FREEZE_REL))
            or passed.get("source_preflight_sha256") != file_sha256(_path(SOURCE_PREFLIGHT_REL))
            or passed.get("source_hashes") != freeze.get("source_hashes")
            or passed.get("input_hashes") != freeze.get("input_hashes")):
        raise OneSidedRunnerError("Root source pass does not bind this exact source, freeze, and preflight")
    source_commit = str(passed.get("source_commit", ""))
    if not source_commit or subprocess.run(["git", "merge-base", "--is-ancestor", source_commit, "HEAD"],
                                           cwd=ROOT, check=False).returncode != 0:
        raise OneSidedRunnerError("ROOT_SOURCE_PASS source commit is not an ancestor of HEAD")
    return passed


def load_saved_frequency_map(origin: int, x_fit: sparse.csr_matrix,
                             vocabulary_size: int, fit_rows: np.ndarray,
                             fit_ids: np.ndarray) -> dict[str, Any]:
    old_root_pass = OLD_RUNNER._require_root_source_pass()
    binding = OLD_RUNNER._penalty_map_binding(origin, x_fit, vocabulary_size,
                                              fit_rows, fit_ids, old_root_pass)
    artifact = _path(f"experiments/runs/optimization_v3_cls_pos_frequency_ridge_v1/maps/penalty_map_E{origin}.npz")
    receipt = _path(f"experiments/runs/optimization_v3_cls_pos_frequency_ridge_v1/maps/penalty_map_E{origin}.json")
    saved = OLD_RUNNER._load_validate_penalty_map(artifact, receipt, binding)
    if (saved["receipt"].get("artifact_sha256") != file_sha256(artifact)
            or saved["receipt"].get("aligned_sha256") != sha256_array(saved["aligned_token_weights"])):
        raise OneSidedRunnerError(f"E{origin} frozen ALIGNED map failed independent lineage checks")
    return saved


def fit_binding(fit_id: str, origin: int, arm: str, root_pass: Mapping[str, Any],
                source_freeze: Mapping[str, Any], fit_rows: np.ndarray,
                fit_ids: np.ndarray, x_fit: sparse.csr_matrix,
                base_fit: sparse.csr_matrix, offset: np.ndarray,
                bag_entry: Mapping[str, Any], map_receipt: Mapping[str, Any],
                penalty_weights: np.ndarray, scalar: float,
                uniform_gamma_sha256: str) -> dict[str, Any]:
    map_receipt_path = _path(f"experiments/runs/optimization_v3_cls_pos_frequency_ridge_v1/maps/penalty_map_E{origin}.json")
    return {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "fit_id": fit_id,
        "origin": int(origin),
        "arm": arm,
        "fit_kind": "PREDICTOR",
        "source_commit": root_pass["source_commit"],
        "source_hashes": root_pass["source_hashes"],
        "input_hashes": root_pass["input_hashes"],
        "root_source_pass_sha256": file_sha256(_path(ROOT_SOURCE_PASS_REL)),
        "source_freeze_sha256": file_sha256(_path(SOURCE_FREEZE_REL)),
        "source_preflight_sha256": file_sha256(_path(SOURCE_PREFLIGHT_REL)),
        "train_sha256": TRAIN_SHA256,
        "fit_rows": int(len(fit_rows)),
        "fit_row_indices_sha256": sha256_array(np.asarray(fit_rows, dtype=np.int64)),
        "fit_ids_sha256": sha256_array(np.asarray(fit_ids).astype(str)),
        "fit_position_sha256": sha256_csr(x_fit),
        "fit_base_sha256": sha256_csr(base_fit),
        "fixed_bag_artifact_sha256": bag_entry["artifact_sha256"],
        "fixed_bag_receipt_sha256": bag_entry["receipt_sha256"],
        "fixed_bag_offset_sha256": sha256_array(np.asarray(offset, dtype=np.float64)),
        "penalty_map_receipt_sha256": file_sha256(map_receipt_path),
        "penalty_map_artifact_sha256": map_receipt["artifact_sha256"],
        "aligned_map_sha256": map_receipt["aligned_sha256"],
        "penalty_vector_sha256": sha256_array(np.asarray(penalty_weights, dtype=np.float64)),
        "global_strength_scalar": float(scalar),
        "one_sided_renormalized": False,
        "uniform_reference_gamma_sha256": str(uniform_gamma_sha256),
        "target_sha256_expected_from_frozen_bag_accounting": bag_entry["fit_population"]["fit_target_sha256"],
    }


def fit_paths(fit_id: str) -> tuple[Path, Path]:
    root = _path(RUN_ROOT)
    return root / "units" / f"{fit_id}.npz", root / "receipts" / f"{fit_id}.json"


def validate_saved_fit(fit_id: str, binding: Mapping[str, Any], width: int) -> dict[str, Any]:
    artifact_path, receipt_path = fit_paths(fit_id)
    events = read_events(_path(LEDGER_REL))
    starts = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("fit_id") == fit_id]
    saved = [e for e in events if e.get("event") == "FIT_SAVED" and e.get("fit_id") == fit_id]
    failed = [e for e in events if e.get("event") == "FIT_FAILED" and e.get("fit_id") == fit_id]
    if len(starts) != 1 or len(saved) > 1 or len(failed) > 1 or saved and failed:
        raise OneSidedRunnerError(f"Duplicate or contradictory lifecycle events for {fit_id}")
    if failed:
        raise OneSidedRunnerError(f"Fit {fit_id} previously failed; no retry is allowed")
    for key, expected in binding.items():
        if starts[0].get(key) != expected:
            raise OneSidedRunnerError(f"FIT_STARTED binding mismatch for {fit_id}: {key}")
    if not artifact_path.is_file() or not receipt_path.is_file():
        raise OneSidedRunnerError(f"Started fit {fit_id} lacks complete durable output; retry prohibited")
    receipt = read_json(receipt_path)
    for key, expected in binding.items():
        if receipt.get(key) != expected:
            raise OneSidedRunnerError(f"Fit receipt binding mismatch for {fit_id}: {key}")
    if receipt.get("fit_target_sha256") != binding["target_sha256_expected_from_frozen_bag_accounting"]:
        raise OneSidedRunnerError(f"Fit target hash mismatch for {fit_id}")
    if receipt.get("artifact_sha256") != file_sha256(artifact_path):
        raise OneSidedRunnerError(f"Gamma artifact hash mismatch for {fit_id}")
    with np.load(artifact_path, allow_pickle=False) as archive:
        if set(archive.files) != {"gamma"}:
            raise OneSidedRunnerError(f"Gamma archive schema mismatch for {fit_id}")
        gamma = np.asarray(archive["gamma"], dtype=np.float64)
    if gamma.shape != (width,) or not np.isfinite(gamma).all() or sha256_array(gamma) != receipt.get("gamma_sha256"):
        raise OneSidedRunnerError(f"Gamma vector shape/content mismatch for {fit_id}")
    artifact_sha, receipt_sha = file_sha256(artifact_path), file_sha256(receipt_path)
    if saved:
        if saved[0].get("artifact_sha256") != artifact_sha or saved[0].get("receipt_sha256") != receipt_sha:
            raise OneSidedRunnerError(f"FIT_SAVED digests mismatch for {fit_id}")
        status = "ALREADY_SAVED"
    else:
        append_event(_path(LEDGER_REL), {
            "event": "FIT_SAVED", "campaign_id": CAMPAIGN_ID, "fit_id": fit_id,
            "artifact_sha256": artifact_sha, "receipt_sha256": receipt_sha,
            "recovered_after_publish": True, "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        status = "RECOVERED_AFTER_PUBLISH"
    if fit_resume_decision(read_events(_path(LEDGER_REL)), fit_id, FIT_IDS) != "RESUME_SAVED":
        raise OneSidedRunnerError(f"Fit lifecycle failed durable resume validation: {fit_id}")
    return {"fit_id": fit_id, "status": status, "artifact_sha256": artifact_sha,
            "receipt_sha256": receipt_sha, "gamma": gamma, "receipt": receipt}


class _SyntheticPublishInterruption(BaseException):
    """Synthetic-only process interruption after durable files, before terminal event."""


def validate_fit_transaction(
    fit_id: str,
    binding: Mapping[str, Any],
    width: int,
    *,
    ledger_path: Path,
    artifact_path: Path,
    receipt_path: Path,
    root_dir: Path,
    allowed_fit_ids: Sequence[str],
) -> dict[str, Any]:
    events = read_events(ledger_path)
    starts = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("fit_id") == fit_id]
    saved = [e for e in events if e.get("event") == "FIT_SAVED" and e.get("fit_id") == fit_id]
    failed = [e for e in events if e.get("event") == "FIT_FAILED" and e.get("fit_id") == fit_id]
    if len(starts) != 1 or len(saved) > 1 or len(failed) > 1 or (saved and failed):
        raise OneSidedRunnerError(f"Duplicate or contradictory lifecycle events for {fit_id}")
    if failed:
        raise OneSidedRunnerError(f"Fit {fit_id} previously failed; automatic retry is prohibited")
    for key, expected in binding.items():
        if starts[0].get(key) != expected:
            raise OneSidedRunnerError(f"FIT_STARTED binding mismatch for {fit_id}: {key}")
    if not artifact_path.is_file() or not receipt_path.is_file():
        raise OneSidedRunnerError(f"Started fit {fit_id} lacks a complete durable output; retry prohibited")
    receipt = read_json(receipt_path)
    for key, expected in binding.items():
        if receipt.get(key) != expected:
            raise OneSidedRunnerError(f"Fit receipt binding mismatch for {fit_id}: {key}")
    if receipt.get("fit_target_sha256") != binding.get("target_sha256_expected_from_frozen_bag_accounting"):
        raise OneSidedRunnerError(f"Fit target hash mismatch for {fit_id}")
    if (receipt.get("artifact_path") != artifact_path.relative_to(root_dir).as_posix()
            or receipt.get("artifact_sha256") != file_sha256(artifact_path)):
        raise OneSidedRunnerError(f"Gamma artifact path/hash mismatch for {fit_id}")
    with np.load(artifact_path, allow_pickle=False) as archive:
        if set(archive.files) != {"gamma"}:
            raise OneSidedRunnerError(f"Gamma archive schema mismatch for {fit_id}")
        gamma = np.asarray(archive["gamma"], dtype=np.float64)
    if gamma.shape != (width,) or not np.isfinite(gamma).all() or sha256_array(gamma) != receipt.get("gamma_sha256"):
        raise OneSidedRunnerError(f"Gamma vector content mismatch for {fit_id}")
    artifact_sha, receipt_sha = file_sha256(artifact_path), file_sha256(receipt_path)
    if saved:
        if saved[0].get("artifact_sha256") != artifact_sha or saved[0].get("receipt_sha256") != receipt_sha:
            raise OneSidedRunnerError(f"FIT_SAVED digest mismatch for {fit_id}")
        status = "ALREADY_SAVED"
    else:
        append_event(ledger_path, {
            "event": "FIT_SAVED", "campaign_id": CAMPAIGN_ID, "fit_id": fit_id,
            "artifact_sha256": artifact_sha, "receipt_sha256": receipt_sha,
            "recovered_after_publish": True, "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        status = "RECOVERED_AFTER_PUBLISH"
    if fit_resume_decision(read_events(ledger_path), fit_id, allowed_fit_ids) != "RESUME_SAVED":
        raise OneSidedRunnerError(f"Fit lifecycle failed durable resume validation for {fit_id}")
    return {"fit_id": fit_id, "status": status, "artifact_sha256": artifact_sha,
            "receipt_sha256": receipt_sha, "gamma": gamma, "receipt": receipt}


def execute_fit_transaction(
    fit_id: str,
    binding: Mapping[str, Any],
    x_fit: sparse.spmatrix,
    offset: Sequence[float],
    penalty_weights: Sequence[float],
    label_loader: Any,
    *,
    ledger_path: Path,
    artifact_path: Path,
    receipt_path: Path,
    root_dir: Path,
    allowed_fit_ids: Sequence[str] = FIT_IDS,
    resource_gate: Any = None,
    inject_publish_interruption: bool = False,
) -> dict[str, Any]:
    """Fit/save/reconcile transaction shared by official and synthetic orchestration."""
    events = read_events(ledger_path)
    if any(event.get("event") == "FIT_FAILED" for event in events):
        raise OneSidedRunnerError("A prior predictor fit failed; stop without retry or rescue")
    index = tuple(allowed_fit_ids).index(fit_id)
    for previous in tuple(allowed_fit_ids)[:index]:
        if fit_resume_decision(events, previous, allowed_fit_ids) != "RESUME_SAVED":
            raise OneSidedRunnerError(f"Fit order is fixed; predecessor {previous} is not saved")
    enforce_fit_cap(events, allowed_fit_ids)
    matching = [event for event in events if event.get("fit_id") == fit_id]
    starts = [event for event in matching if event.get("event") == "FIT_STARTED"]
    terminals = [event for event in matching if event.get("event") in {"FIT_SAVED", "FIT_FAILED"}]
    if len(starts) > 1 or len(terminals) > 1:
        raise OneSidedRunnerError(f"Duplicate lifecycle events block resume for {fit_id}")
    if starts and not terminals:
        # Durable outputs can be verified and closed without another fit or
        # target read. An incomplete output pair remains consumed and blocked.
        if artifact_path.is_file() and receipt_path.is_file():
            return validate_fit_transaction(
                fit_id, binding, sparse.csr_matrix(x_fit).shape[1], ledger_path=ledger_path,
                artifact_path=artifact_path, receipt_path=receipt_path,
                root_dir=root_dir, allowed_fit_ids=allowed_fit_ids,
            )
        raise OneSidedRunnerError(
            f"Started fit {fit_id} lacks a complete durable output; retry is prohibited"
        )
    state = fit_resume_decision(events, fit_id, allowed_fit_ids)
    if state == "RESUME_SAVED":
        return validate_fit_transaction(
            fit_id, binding, sparse.csr_matrix(x_fit).shape[1], ledger_path=ledger_path,
            artifact_path=artifact_path, receipt_path=receipt_path,
            root_dir=root_dir, allowed_fit_ids=allowed_fit_ids,
        )
    if state != "START_NEW":
        raise OneSidedRunnerError(f"Fit lifecycle is not startable: {fit_id}: {state}")
    if artifact_path.exists() or receipt_path.exists():
        raise OneSidedRunnerError(f"Unstarted fit {fit_id} already has durable files")
    resource = resource_gate("before_fit_target_access") if resource_gate else None
    start = {
        "event": "FIT_STARTED", **dict(binding),
        "fit_target_sha256": binding.get("target_sha256_expected_from_frozen_bag_accounting"),
        "fit_consumed_at_start": True,
        "resource": resource,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    append_event(ledger_path, start)
    try:
        y = np.asarray(label_loader(), dtype=np.int8).reshape(-1)
        expected_target_sha = binding.get("target_sha256_expected_from_frozen_bag_accounting")
        if (not np.isin(y, (0, 1)).all() or sha256_array(y) != expected_target_sha):
            raise OneSidedRunnerError(f"{fit_id} label loader returned invalid or unexpected target values")
        fit_resource = resource_gate("immediately_before_optimizer") if resource_gate else None
        started = time.perf_counter()
        with threadpool_limits(limits=THREADS):
            gamma, optimizer = fit_offset(x_fit, offset, y, penalty_weights)
        buffer = io.BytesIO()
        np.savez_compressed(buffer, gamma=np.asarray(gamma, dtype=np.float64))
        write_bytes_exclusive(artifact_path, buffer.getvalue())
        receipt = {
            **dict(binding),
            "fit_target_sha256": sha256_array(y),
            "gamma_sha256": sha256_array(gamma),
            "artifact_path": artifact_path.relative_to(root_dir).as_posix(),
            "artifact_sha256": file_sha256(artifact_path),
            "optimizer": optimizer,
            "elapsed_seconds": float(time.perf_counter() - started),
            "resource_before_optimizer": fit_resource,
            "target_access": "fit-prefix only after source and identity boundary",
            "query_labels_read": False,
            "test_or_samplesubmission_accessed": False,
        }
        write_json_once(receipt_path, receipt)
        if inject_publish_interruption:
            raise _SyntheticPublishInterruption("synthetic interruption after durable gamma and receipt")
        artifact_sha, receipt_sha = file_sha256(artifact_path), file_sha256(receipt_path)
        append_event(ledger_path, {
            "event": "FIT_SAVED", "campaign_id": CAMPAIGN_ID, "fit_id": fit_id,
            "artifact_sha256": artifact_sha, "receipt_sha256": receipt_sha,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        return {"fit_id": fit_id, "status": "SAVED", "artifact_sha256": artifact_sha,
                "receipt_sha256": receipt_sha, "optimizer": optimizer, "gamma": gamma}
    except _SyntheticPublishInterruption:
        raise
    except Exception as exc:
        append_event(ledger_path, {
            "event": "FIT_FAILED", "campaign_id": CAMPAIGN_ID, "fit_id": fit_id,
            "error_type": type(exc).__name__, "error": str(exc),
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        raise


def run_one_fit(fit_id: str) -> dict[str, Any]:
    if fit_id not in FIT_IDS:
        raise OneSidedRunnerError(f"Fit ID is outside the exact four-fit cap: {fit_id}")
    root_pass = require_root_source_pass()
    source_freeze = verify_source_freeze(require_frozen_commit=True)
    preflight = read_json(_path(SOURCE_PREFLIGHT_REL))
    if preflight.get("source_hashes") != source_freeze.get("source_hashes"):
        raise OneSidedRunnerError("Committed target-free source preflight no longer binds this source")
    prior_ledger = _path(PRIOR_SHARED_LEDGER_REL)
    if file_sha256(prior_ledger) != PRIOR_SHARED_LEDGER_SHA256:
        raise OneSidedRunnerError("Prior shared ledger changed after the approved checkpoint")

    with OLD_EVAL.SharedJobLock():
        resource = require_resource("small_gamma_fit", "before fit reservation and official inputs")
        # The exact function used by the real identity-only preflight is called
        # again here immediately before the first target access.
        identity = check_official_identities()
        identity_boundary = require_actual_boundary("small_gamma_fit", "after pre-target identity checks")
        origin = 67 if fit_id.startswith("E67_") else 75
        arm = fit_id.removeprefix(f"E{origin}_")
        protocol = read_json(_path(PROTOCOL_REL))
        manifest_path = _path(f"experiments/runs/optimization_v3_research_next_v1/POSITIONAL_TEXT_PILOT_V1/MATRIX_MANIFEST_E{origin}.json")
        manifest = read_json(manifest_path)
        frame = _read_identity_frame()
        cycles = frame["cycle_code"].to_numpy(dtype=np.int64)
        fit_rows = np.flatnonzero(cycles <= origin)
        fit_ids = frame.iloc[fit_rows]["record_id"].astype(str).to_numpy()
        if len(fit_rows) != int(protocol["origins"][str(origin)]["fit_rows"]):
            raise OneSidedRunnerError(f"E{origin} fit row population changed")
        base_fit = OLD_RUNNER._load_csr_from_manifest(manifest, "fit_base")
        augmented_fit = OLD_RUNNER._load_csr_from_manifest(manifest, "fit_position")
        if (augmented_fit[:, :base_fit.shape[1]] != base_fit).nnz:
            raise OneSidedRunnerError(f"E{origin} augmented training matrix changed the base feature prefix")
        x_fit = augmented_fit[:, base_fit.shape[1]:].tocsr()
        vocab = x_fit.shape[1] // 3
        if (x_fit.shape != (len(fit_rows), 3 * vocab)
                or vocab != int(protocol["origins"][str(origin)]["vocabulary_size"])):
            raise OneSidedRunnerError(f"E{origin} positional dimensions differ from the frozen protocol")
        maps = load_saved_frequency_map(origin, x_fit, vocab, fit_rows, fit_ids)
        penalties = build_penalty_vectors(maps["aligned_token_weights"])
        pos_weights = penalties["one_sided_position_weights"] if arm == "ONE_SIDED" else penalties["global_strength_position_weights"]
        uniform_rel = f"{OLD_EVAL.POS_BAG_ROOT}/units/POS_BAG_OFFSET_V1_E{origin}_REAL.npz"
        uniform_receipt_rel = f"{OLD_EVAL.POS_BAG_ROOT}/receipts/POS_BAG_OFFSET_V1_E{origin}_REAL.json"
        uniform_gamma, uniform_lineage = OLD_EVAL.gamma_from_artifact(uniform_rel, uniform_receipt_rel)
        bag_id = f"POSITIONAL_TEXT_PILOT_V1_E{origin}_BAG_LR"
        accounting = read_json(_path("reports/optimization_v3/positional_text_pilot_v1/FIT_ACCOUNTING.json"))
        bag_entry = next((entry for entry in accounting["fit_units"] if entry.get("fit_id") == bag_id), None)
        if bag_entry is None:
            raise OneSidedRunnerError(f"Frozen same-origin C2_B fit accounting is absent: E{origin}")
        bag_rel = f"experiments/runs/optimization_v3_research_next_v1/POSITIONAL_TEXT_PILOT_V1/units/{bag_id}.joblib"
        bag_receipt_rel = f"experiments/runs/optimization_v3_research_next_v1/POSITIONAL_TEXT_PILOT_V1/receipts/{bag_id}.json"
        bag_path = _path(bag_rel)
        bag_receipt_path = _path(bag_receipt_rel)
        if (file_sha256(bag_path) != bag_entry["artifact_sha256"]
                or file_sha256(bag_receipt_path) != bag_entry["receipt_sha256"]):
            raise OneSidedRunnerError(f"Frozen same-origin C2_B artifact/receipt mismatch: E{origin}")
        bag = joblib.load(bag_path).get("estimator")
        if bag is None or list(np.asarray(bag.classes_).tolist()) != [0, 1]:
            raise OneSidedRunnerError(f"Frozen C2_B estimator class order invalid: E{origin}")
        coef = np.asarray(bag.coef_, dtype=np.float64)
        intercept = np.asarray(bag.intercept_, dtype=np.float64).reshape(-1)
        if coef.shape != (1, base_fit.shape[1]) or intercept.shape != (1,):
            raise OneSidedRunnerError(f"Frozen C2_B estimator width mismatch: E{origin}")
        offset = np.asarray(base_fit @ coef[0], dtype=np.float64).reshape(-1) + float(intercept[0])
        if not np.isfinite(offset).all():
            raise OneSidedRunnerError(f"Frozen C2_B fit offset is non-finite: E{origin}")
        binding = fit_binding(
            fit_id, origin, arm, root_pass, source_freeze, fit_rows, fit_ids, x_fit,
            base_fit, offset, bag_entry, maps["receipt"], pos_weights,
            penalties["global_strength_scalar"], uniform_lineage["gamma_sha256"],
        )
        binding["identity_pre_target_check_sha256"] = canonical_sha256(identity)
        artifact_path, receipt_path = fit_paths(fit_id)
        return execute_fit_transaction(
            fit_id, binding, x_fit, offset, pos_weights,
            lambda: read_fit_labels(
                _path(TRAIN_REL), TRAIN_SHA256, fit_rows, fit_ids,
                binding["target_sha256_expected_from_frozen_bag_accounting"]),
            ledger_path=_path(LEDGER_REL), artifact_path=artifact_path,
            receipt_path=receipt_path, root_dir=ROOT,
            resource_gate=lambda stage: require_resource("small_gamma_fit", f"{fit_id} {stage}"),
        )


def run_synthetic_qa() -> dict[str, Any]:
    """Deferred to the integrated test module, which calls this runner's orchestration."""
    import importlib.util
    test_path = ROOT / "tests/test_pos_one_sided_ridge_v1.py"
    spec = importlib.util.spec_from_file_location("_synthetic_one_sided_campaign", test_path)
    if spec is None or spec.loader is None:
        raise OneSidedRunnerError("Synthetic orchestration QA module cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with OLD_EVAL.SharedJobLock():
        resource = require_resource("small_gamma_fit", "before synthetic orchestration QA")
        result = module.run_synthetic_campaign_qa()
        boundary = require_actual_boundary("small_gamma_fit", "after synthetic orchestration QA")
    return {"status": "PASS", "resource_initial": resource,
            "resource_boundary": boundary, **result}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--synthetic-qa", action="store_true")
    modes.add_argument("--identity-preflight", action="store_true")
    modes.add_argument("--write-source-freeze", action="store_true")
    modes.add_argument("--run-fit", choices=FIT_IDS)
    args = parser.parse_args(argv)
    try:
        if args.synthetic_qa:
            result = run_synthetic_qa()
        elif args.identity_preflight:
            result = run_identity_preflight()
        elif args.write_source_freeze:
            result = write_source_freeze()
        else:
            result = run_one_fit(str(args.run_fit))
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2))
        return 0
    except (OneSidedRunnerError, FrequencyRidgeError, OLD_EVAL.EvaluationError) as exc:
        print(json.dumps({"status": "BLOCKED", "error": str(exc)}, ensure_ascii=False, sort_keys=True, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
