"""Source-frozen, four-calibrator center-local residual pilot.

The normal source-preparation flow is ``--freeze`` then ``--preflight``. Real
fits and score-target access require a separate signed ROOT_SOURCE_PASS_R1.json
and run in separate ``--fit`` and ``--score`` commands. Nothing in this module
authorizes those stages by itself.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import psutil
from threadpoolctl import threadpool_limits

from scripts import reg_monotone_hinge_p50_v1 as r1
from scripts import reg_monotone_hinge_p50_v4 as r4

ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = "reg_center_local_correction_v1"
FIT_IDS = ("D1_LOCAL", "D1_GLOBAL_CONTROL", "D2_LOCAL", "D2_GLOBAL_CONTROL")
FIT_TO_STAGE = {"D1_LOCAL": "HINGE_D1", "D1_GLOBAL_CONTROL": "HINGE_D1",
                "D2_LOCAL": "HINGE_D2", "D2_GLOBAL_CONTROL": "HINGE_D2"}
STAGES = ("HINGE_D1", "HINGE_D2")
FIT_COUNTS = {"HINGE_D1": 4_950, "HINGE_D2": 12_412}
PRESENT_FIT_COUNTS = {"HINGE_D1": 4_608, "HINGE_D2": 11_563}
SCORE_COUNTS = {"HINGE_D1": 7_462, "HINGE_D2": 8_491}
EXPECTED_ORIGIN = {("HINGE_D1", "fit"): 67, ("HINGE_D1", "score"): 67,
                   ("HINGE_D2", "fit"): 67, ("HINGE_D2", "score"): 75}
EXPECTED_CYCLES = {
    ("HINGE_D1", "fit"): tuple(range(68, 72)),
    ("HINGE_D1", "score"): tuple(range(72, 76)),
    ("HINGE_D2", "fit"): tuple(range(68, 76)),
    ("HINGE_D2", "score"): tuple(range(76, 80)),
}
BIN_EDGES = (0.0, 20.0, 40.0, 60.0, 80.0, 100.0)
BIN_LABELS = ("[0,20)", "[20,40)", "[40,60)", "[60,80)", "[80,100]")
LOCAL_CENTER = 50.0
LOCAL_HALF_WIDTH = 10.0
DELTA_BOUNDS = (-2.0, 0.0)
MIN_LOCAL_SUPPORT = 200
MIN_GROUP_SUPPORT = 200
MIN_CYCLE_SUPPORT = 50
FIT_EPSILON = 1e-12

NOVELTY_HASHES = {
    "regression_residual_findings": "48465bfc35162315da247600673ae9bd54121dfb12c7315d0a4449dc6b5e81ae",
    "regression_position_frontier_findings": "736837b510762965dd8c41089a88ed943a83b72bb7e56ae48ce41e5ce5703345",
    "reg_position_transport_diagnostic": "5b0840a80ce80cb478d6dbf888c733b7f7cd9f901ee67006b8c4206bd5be0085",
    "r4_decision": "538007fd2449b472e1fad657c916760710b1c6dd148287f051959162318ff7ba",
    "r4_protocol": "534192cfa32f181b1a168110517ae6290598a529a02911bf32f93f70753623a1",
    "r3_protocol": "65bd376d5645f47338dac844be9c0a573a951919c8fccc0e4321da65674646cf",
}


class PilotError(RuntimeError):
    """Fail-closed input, authorization, ledger, or resource condition."""


@dataclass(frozen=True)
class Layout:
    root: Path
    report_dir: Path
    experiment_dir: Path
    source_paths: Mapping[str, Path]
    reference_paths: Mapping[str, Path]
    input_paths: Mapping[str, Path]
    lock_path: Path

    @property
    def freeze(self) -> Path:
        return self.experiment_dir / "PREFIT_FREEZE_R1.json"

    @property
    def preflight(self) -> Path:
        return self.report_dir / "IDENTITY_PREFLIGHT_R1.json"

    @property
    def root_pass(self) -> Path:
        return self.experiment_dir / "ROOT_SOURCE_PASS_R1.json"

    @property
    def ledger(self) -> Path:
        return self.experiment_dir / "calibrator_ledger_R1.jsonl"

    @property
    def fit_dir(self) -> Path:
        return self.experiment_dir / "fits_R1"

    @property
    def prediction_dir(self) -> Path:
        return self.experiment_dir / "score_predictions_R1"

    @property
    def prediction_freeze(self) -> Path:
        return self.experiment_dir / "PREDICTION_FREEZE_R1.json"

    @property
    def evaluation(self) -> Path:
        return self.report_dir / "EVALUATION_R1.json"

    @property
    def verification(self) -> Path:
        return self.report_dir / "INDEPENDENT_VERIFICATION_R1.json"


def make_layout(root: Path = ROOT, *, source_root: Path | None = None,
                lock_path: Path | None = None) -> Layout:
    root = Path(root).resolve()
    source_root = root if source_root is None else Path(source_root).resolve()
    report_dir = root / "reports" / "optimization_v3" / NAMESPACE
    experiment_dir = root / "experiments" / NAMESPACE
    source_paths = {
        "runner": source_root / "scripts" / "reg_center_local_correction_v1.py",
        "verifier": source_root / "scripts" / "verify_reg_center_local_correction_v1.py",
        "tests": source_root / "tests" / "test_reg_center_local_correction_v1.py",
        "policy": source_root / "reports" / "optimization_v3" / NAMESPACE / "POLICY_R1.md",
        "handoff": source_root / "reports" / "optimization_v3" / NAMESPACE / "SOL_HANDOFF.md",
        "gitattributes": source_root / ".gitattributes",
    }
    reference_paths = {
        "hinge_core": source_root / "scripts" / "reg_monotone_hinge_p50_v1.py",
        "r4_runner": source_root / "scripts" / "reg_monotone_hinge_p50_v4.py",
        "r4_verifier": source_root / "scripts" / "verify_reg_monotone_hinge_p50_v4.py",
        "r4_protocol_r3": source_root / "reports" / "optimization_v3" / "reg_monotone_hinge_p50_v1" / "PROTOCOL_R3.md",
        "r4_protocol_r4": source_root / "reports" / "optimization_v3" / "reg_monotone_hinge_p50_v1" / "PROTOCOL_R4.md",
        "r4_decision": source_root / "reports" / "optimization_v3" / "reg_monotone_hinge_p50_v1" / "SOL_DIAGNOSTIC_DECISION_R4.md",
        "residual_findings": source_root / "reports" / "optimization_v3" / "regression_residual" / "FINDINGS.md",
        "position_frontier_findings": source_root / "reports" / "optimization_v3" / "regression_position_frontier_v1" / "FINDINGS.md",
        "transport_diagnostic": source_root / "reports" / "optimization_v3" / "reg_position_transport_diagnostic_v1" / "TRANSPORT_DIAGNOSTIC.md",
    }
    input_paths = {
        "r4_diagnostics": root / "reports" / "optimization_v3" / "reg_monotone_hinge_p50_v1" / "DIAGNOSTICS_R4.json",
        "r4_verification": root / "reports" / "optimization_v3" / "reg_monotone_hinge_p50_v1" / "DIAGNOSTICS_VERIFICATION_R4.json",
        "r4_identity_freeze": root / "experiments" / "reg_monotone_hinge_p50_v1" / "IDENTITY_FREEZE_R4.json",
        "r4_root_pass": root / "experiments" / "reg_monotone_hinge_p50_v1" / "ROOT_SOURCE_PASS_R4.json",
        "resource_policy": root / "reports" / "optimization_v3" / "RESOURCE_POLICY_USER_REVISED_20261105.json",
        "d1_hinge_state": root / "experiments" / "runs" / "reg_monotone_hinge_p50_v1" / "R4" / "fits" / "HINGE_D1.json",
        "d1_global_state": root / "experiments" / "runs" / "REG_FINAL_WAVE_FOLLOWUP_V2" / "fits" / "D1_GLOBAL.json",
        "d1_fit_input": root / "experiments" / "runs" / "reg_monotone_hinge_p50_v1" / "R4" / "fit_inputs" / "HINGE_D1.npz",
        "d1_score_rows": root / "experiments" / "runs" / "reg_monotone_hinge_p50_v1" / "R4" / "score_rows" / "HINGE_D1.npz",
        "d2_hinge_state": root / "experiments" / "runs" / "reg_monotone_hinge_p50_v1" / "R4" / "fits" / "HINGE_D2.json",
        "d2_global_state": root / "experiments" / "runs" / "REG_FINAL_WAVE_FOLLOWUP_V2" / "fits" / "D2_GLOBAL.json",
        "d2_fit_input": root / "experiments" / "runs" / "reg_monotone_hinge_p50_v1" / "R4" / "fit_inputs" / "HINGE_D2.npz",
        "d2_score_rows": root / "experiments" / "runs" / "reg_monotone_hinge_p50_v1" / "R4" / "score_rows" / "HINGE_D2.npz",
    }
    if lock_path is None:
        lock_path = r4.SHARED_JOB_LOCK
    return Layout(root, report_dir, experiment_dir, source_paths,
                  reference_paths, input_paths, Path(lock_path).resolve())


DEFAULT_LAYOUT = make_layout()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(value), sort_keys=True, ensure_ascii=False,
                         allow_nan=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json_once(path: Path, value: Mapping[str, Any]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(value), sort_keys=True, ensure_ascii=False,
                         allow_nan=False, indent=2).encode("utf-8") + b"\n"
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise PilotError(f"Refusing to overwrite immutable artifact: {path}") from exc
    return hashlib.sha256(payload).hexdigest()


def write_npz_once(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            np.savez(stream, **{key: np.asarray(value) for key, value in arrays.items()})
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise PilotError(f"Refusing to overwrite immutable prediction bundle: {path}") from exc
    return sha256_file(path)


def _git_head(layout: Layout) -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=layout.root,
                            capture_output=True, text=True, check=False)
    if result.returncode:
        raise PilotError("Cannot resolve source commit")
    return result.stdout.strip()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PilotError(f"Cannot read JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise PilotError(f"Expected JSON object: {path}")
    return value


def _validate_signed(path: Path, *, verdict: str | None = None) -> dict[str, Any]:
    value = _read_json(path)
    body = dict(value)
    signature = body.pop("signature", None)
    if not isinstance(signature, str) or canonical_sha(body) != signature:
        raise PilotError(f"Canonical signature mismatch: {path}")
    if verdict is not None and value.get("verdict") != verdict:
        raise PilotError(f"Unexpected signed verdict at {path}: {value.get('verdict')}")
    return value


def _expected_novelty_paths(layout: Layout) -> dict[str, Path]:
    return {
        "regression_residual_findings": layout.reference_paths["residual_findings"],
        "regression_position_frontier_findings": layout.reference_paths["position_frontier_findings"],
        "reg_position_transport_diagnostic": layout.reference_paths["transport_diagnostic"],
        "r4_decision": layout.reference_paths["r4_decision"],
        "r4_protocol": layout.reference_paths["r4_protocol_r4"],
        "r3_protocol": layout.reference_paths["r4_protocol_r3"],
    }


def _check_novelty_documents(layout: Layout) -> dict[str, str]:
    hashes = {}
    for name, path in _expected_novelty_paths(layout).items():
        if not path.is_file():
            raise PilotError(f"Novelty scan document missing: {path}")
        digest = sha256_file(path)
        if digest != NOVELTY_HASHES[name]:
            raise PilotError(f"Novelty scan document changed after the six-file review: {name}")
        hashes[name] = digest
    if len(hashes) != 6:
        raise PilotError("Novelty scan must bind exactly six summaries")
    return hashes


def _check_r4_predecessor(layout: Layout) -> dict[str, Any]:
    diagnostics = _read_json(layout.input_paths["r4_diagnostics"])
    verification = _read_json(layout.input_paths["r4_verification"])
    identity_freeze = _read_json(layout.input_paths["r4_identity_freeze"])
    root_pass = _validate_signed(layout.input_paths["r4_root_pass"])
    if (diagnostics.get("experiment_id") != "REG_MONOTONE_HINGE_P50_V1_R4"
            or diagnostics.get("status") != "TWO_DIAGNOSTICS_COMPLETE_PENDING_ROOT_REVIEW"
            or diagnostics.get("test_or_sample_accessed") is not False):
        raise PilotError("R4 diagnostics metadata or access boundary is unexpected")
    if (verification.get("experiment_id") != diagnostics.get("experiment_id")
            or verification.get("status") != "PASS_INDEPENDENT_DIAGNOSTICS_REPLAY"
            or verification.get("test_or_sample_accessed") is not False
            or verification.get("diagnostics_sha256") != sha256_file(layout.input_paths["r4_diagnostics"])):
        raise PilotError("R4 verification does not bind the expected diagnostics receipt")
    if (root_pass.get("authorized_fit_ids") != ["HINGE_D1", "HINGE_D2"]
            or root_pass.get("prefit_freeze_sha256") != sha256_file(layout.input_paths["r4_identity_freeze"])
            or root_pass.get("source_commit") != identity_freeze.get("source_commit")):
        raise PilotError("Signed R4 predecessor pass does not bind the fixed D1/D2 freeze")
    stages = diagnostics.get("stages", [])
    if {stage.get("fit_id") for stage in stages} != set(STAGES) or len(stages) != 2:
        raise PilotError("R4 diagnostics do not contain exactly D1 and D2")
    fit_replays = {item.get("fit_id"): item for item in verification.get("fit_replays", [])}
    score_replays = {item.get("fit_id"): item for item in verification.get("score_replays", [])}
    if set(fit_replays) != set(STAGES) or set(score_replays) != set(STAGES):
        raise PilotError("R4 verification is missing fit or score replay records")
    for record in stages:
        stage_id = record["fit_id"]
        stage = "d1" if stage_id == "HINGE_D1" else "d2"
        fit_path = layout.input_paths[f"{stage}_fit_input"]
        score_path = layout.input_paths[f"{stage}_score_rows"]
        hinge_path = layout.input_paths[f"{stage}_hinge_state"]
        global_path = layout.input_paths[f"{stage}_global_state"]
        pins = ((fit_path, record["fit_input_path"], record["fit_input_sha256"]),
                (score_path, record["score_row_bundle_path"], record["score_row_bundle_sha256"]),
                (hinge_path, record["fit_state_path"], record["fit_state_sha256"]))
        for actual_path, referenced, expected_digest in pins:
            resolved = Path(referenced)
            if not resolved.is_absolute():
                resolved = (layout.root / resolved).resolve()
            if resolved != actual_path.resolve() or sha256_file(actual_path) != expected_digest:
                raise PilotError(f"R4 {stage_id} path/hash mismatch: {actual_path.name}")
        global_pin = identity_freeze.get("input_hashes", {}).get(
            f"r3_stage__{stage_id[-2:]}_GLOBAL_state", {})
        global_pin_path = Path(global_pin.get("path", ""))
        if not global_pin_path.is_absolute():
            global_pin_path = (layout.root / global_pin_path).resolve()
        if (global_pin_path.resolve() != global_path.resolve()
                or global_pin.get("sha256") != sha256_file(global_path)
                or record.get("source_global_state_sha256") != global_pin.get("sha256")):
            raise PilotError(f"R4 {stage_id} saved GLOBAL state hash mismatch")
        if (record.get("fit_rows") != FIT_COUNTS[stage_id]
                or record.get("score_rows") != SCORE_COUNTS[stage_id]
                or fit_replays[stage_id].get("fit_input_sha256") != record["fit_input_sha256"]
                or score_replays[stage_id].get("score_bundle_sha256") != record["score_row_bundle_sha256"]):
            raise PilotError(f"R4 {stage_id} verified row counts or bundle hashes differ")
        hinge = _read_json(hinge_path).get("solver_result", {})
        if (_read_json(hinge_path).get("status") != "LP_COMPLETE"
                or hinge.get("success") is not True
                or not all(math.isfinite(float(hinge[key])) for key in
                           ("intercept", "slope_low", "slope_high"))):
            raise PilotError(f"R4 {stage_id} saved HINGE parameters are invalid")
        slope_low = float(hinge["slope_low"])
        slope_high = float(hinge["slope_high"])
        worst_local_slope = min(slope_low - abs(DELTA_BOUNDS[0]) / LOCAL_HALF_WIDTH,
                                slope_high)
        if worst_local_slope < 0.0:
            raise PilotError(f"R4 {stage_id} saved HINGE slopes cannot guarantee monotone LOCAL bounds")
    return {"diagnostics_sha256": sha256_file(layout.input_paths["r4_diagnostics"]),
            "verification_sha256": sha256_file(layout.input_paths["r4_verification"]),
            "r4_root_pass_sha256": sha256_file(layout.input_paths["r4_root_pass"]),
            "r4_identity_freeze_sha256": sha256_file(layout.input_paths["r4_identity_freeze"])}


def _pinned_file_map(layout: Layout) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for category, files in (("source", layout.source_paths),
                            ("reference_source", layout.reference_paths),
                            ("input", layout.input_paths)):
        for name, path in files.items():
            resolved = Path(path).resolve()
            if not resolved.is_file():
                raise PilotError(f"Frozen file is absent: {resolved}")
            result[f"{category}:{name}"] = {"path": str(resolved),
                                             "sha256": sha256_file(resolved)}
    novelty = _check_novelty_documents(layout)
    for name, digest in novelty.items():
        path = _expected_novelty_paths(layout)[name].resolve()
        result[f"novelty_summary:{name}"] = {"path": str(path), "sha256": digest}
    return result


def freeze_inputs(layout: Layout | None = None) -> dict[str, Any]:
    layout = DEFAULT_LAYOUT if layout is None else layout
    if layout.freeze.exists():
        raise PilotError(f"Prefit freeze already exists: {layout.freeze}")
    forbidden_outputs = (layout.preflight, layout.ledger, layout.evaluation, layout.verification,
                         layout.prediction_freeze)
    if any(path.exists() for path in forbidden_outputs):
        raise PilotError("A fit, prediction, or scoring artifact exists before prefit freeze")
    r4_meta = _check_r4_predecessor(layout)
    policy = _read_json(layout.input_paths["resource_policy"])
    small = policy.get("workloads", {}).get("small_calibration_lp", {})
    saved = policy.get("workloads", {}).get("saved_prediction_evaluation", {})
    expected_small = {"minimum_available_bytes": 2_147_483_648,
        "incremental_estimate_bytes": 536_870_912,
        "minimum_projected_free_bytes": 1_073_741_824,
        "minimum_actual_free_at_boundary_bytes": 1_073_741_824}
    expected_saved = {"minimum_available_bytes": 1_610_612_736,
        "incremental_estimate_bytes": 536_870_912,
        "minimum_projected_free_bytes": 805_306_368,
        "minimum_actual_free_at_boundary_bytes": 805_306_368}
    if small != expected_small or saved != expected_saved:
        raise PilotError("Current workload-specific resource policy differs from the handoff")
    files = _pinned_file_map(layout)
    body = {"schema_version": 1, "experiment_id": NAMESPACE.upper() + "_R1",
        "status": "FROZEN_BEFORE_CANDIDATE_TARGET_ACCESS",
        "source_commit": _git_head(layout), "source_and_input_files": files,
        "r4_predecessor_receipts": r4_meta,
        "novelty_scan": {"document_count": 6,
            "documents": {key: {"path": str(path.resolve()), "sha256": NOVELTY_HASHES[key]}
                          for key, path in _expected_novelty_paths(layout).items()},
            "exact_candidate_duplicate_found": False,
            "repository_wide_novelty": "UNKNOWN"},
        "policy": {"fit_ids": list(FIT_IDS), "fit_cap": 4,
            "base_model_fits": 0, "preprocessing_fits": 0,
            "local_center": LOCAL_CENTER, "local_half_width": LOCAL_HALF_WIDTH,
            "delta_bounds": list(DELTA_BOUNDS), "objective": "unweighted_sum_absolute_without_clipping",
            "local_optimizer": "exact_lower_weighted_median_with_weight_b_and_exact_binary_weight_ties",
            "global_control_optimizer": "lower_empirical_median_of_present_y_minus_saved_hinge",
            "fit_present_rows": PRESENT_FIT_COUNTS, "score_all_rows": SCORE_COUNTS,
            "score_unique_record_ids": 15_953,
            "bins": {"edges": list(BIN_EDGES), "labels": list(BIN_LABELS), "min_support": MIN_GROUP_SUPPORT},
            "cycle_min_support": MIN_CYCLE_SUPPORT,
            "target_tail_groups": ["y<=5", "y>=95"],
            "missing_route": "unchanged_raw_DIRECT_float64_bits",
            "prediction_before_score": True,
            "counterfactual_policies": False,
            "adaptive_post_discovery": True,
            "mechanism_gate": "LOCAL gain >1e-12 over both HINGE and GLOBAL_CONTROL in each score window; both fit deltas <0",
            "candidate_gate": "pooled all-row gain over HINGE >=0.01; both window gains >=0; max supported all-row subgroup harm <=0.25",
            "exploratory_gate": "pooled gain >=0; each window harm versus HINGE <=0.02; max supported all-row subgroup harm <=0.25"},
        "resource_policy_sha256": sha256_file(layout.input_paths["resource_policy"]),
        "shared_lock_path": str(layout.lock_path), "job_limit": 1, "thread_limit": 2,
        "candidate_targets_loaded": False, "fit_started": False, "score_targets_loaded": False,
        "test_or_sample_accessed": False, "e79_accessed": False, "csv_created": False,
        "kaggle_upload": False}
    body["signature"] = canonical_sha(body)
    digest = write_json_once(layout.freeze, body)
    return {"status": body["status"], "path": str(layout.freeze.resolve()),
            "sha256": digest, "source_commit": body["source_commit"],
            "pinned_file_count": len(files)}


def verify_freeze(layout: Layout | None = None) -> tuple[dict[str, Any], str]:
    layout = DEFAULT_LAYOUT if layout is None else layout
    freeze = _read_json(layout.freeze)
    signature = freeze.pop("signature", None)
    if signature != canonical_sha(freeze):
        raise PilotError("Prefit freeze signature mismatch")
    freeze["signature"] = signature
    if freeze.get("status") != "FROZEN_BEFORE_CANDIDATE_TARGET_ACCESS":
        raise PilotError("Prefit freeze status mismatch")
    if _git_head(layout) != freeze.get("source_commit"):
        raise PilotError("Source HEAD differs from prefit freeze")
    if sha256_file(layout.input_paths["resource_policy"]) != freeze.get("resource_policy_sha256"):
        raise PilotError("Resource policy changed after prefit freeze")
    _check_r4_predecessor(layout)
    _check_novelty_documents(layout)
    for item in freeze.get("source_and_input_files", {}).values():
        path = Path(item["path"])
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise PilotError(f"Frozen source/input bytes changed: {path}")
    return freeze, sha256_file(layout.freeze)


def _resource_snapshot(layout: Layout, workload: str, boundary: str,
                       *, initial: bool) -> dict[str, Any]:
    policy = _read_json(layout.input_paths["resource_policy"])
    limits = policy["workloads"][workload]
    available = int(psutil.virtual_memory().available)
    estimate = int(limits["incremental_estimate_bytes"])
    projected = available - estimate
    initial_ok = (available >= int(limits["minimum_available_bytes"])
                  and projected >= int(limits["minimum_projected_free_bytes"]))
    boundary_ok = available >= int(limits["minimum_actual_free_at_boundary_bytes"])
    if not (initial_ok if initial else boundary_ok):
        raise PilotError(f"RAM gate failed at {boundary}: available={available}, projected={projected}")
    memory = psutil.Process(os.getpid()).memory_info()
    peak = getattr(memory, "peak_wset", None)
    return {"status": "PASS", "workload": workload, "boundary": boundary,
        "available_bytes": available, "incremental_estimate_bytes": estimate,
        "projected_free_bytes": projected,
        "minimum_initial_available_bytes": int(limits["minimum_available_bytes"]),
        "minimum_projected_free_bytes": int(limits["minimum_projected_free_bytes"]),
        "minimum_actual_free_at_boundary_bytes": int(limits["minimum_actual_free_at_boundary_bytes"]),
        "initial_gate_pass": bool(initial_ok), "actual_boundary_pass": bool(boundary_ok),
        "process_rss_bytes": int(memory.rss), "process_rss_measured": True,
        "os_peak_working_set_bytes": int(peak) if peak is not None else None,
        "os_peak_working_set_measured": peak is not None,
        "threads": 2, "concurrent_heavy_jobs": 1}


def _read_npz_fields(path: Path, fields: Sequence[str], *,
                     allow_unloaded: Sequence[str] = (),
                     load_targets: bool = False) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        names = set(archive.files)
        required = set(fields) | set(allow_unloaded) | ({"targets"} if load_targets else set())
        if not required.issubset(names):
            raise PilotError(f"NPZ missing expected fields: {path}/{sorted(required - names)}")
        extras = names - required
        if extras:
            raise PilotError(f"NPZ schema contains unknown arrays: {path}/{sorted(extras)}")
        selected = list(fields)
        if load_targets:
            if "targets" not in selected:
                selected.append("targets")
        return {name: np.asarray(archive[name]).copy() for name in selected}


IDENTITY_FIELDS = ("physical_rows", "record_ids", "prediction_origins",
    "original_bundle_indices", "cycles", "direct_predictions", "missing_mask")
FIT_ARCHIVE_FIELDS = IDENTITY_FIELDS + ("targets",)
SCORE_NON_TARGET_FIELDS = IDENTITY_FIELDS + ("raw_direct", "hinge_present_only")


def _stage_files(layout: Layout, stage_id: str) -> tuple[Path, Path, Path]:
    slug = "d1" if stage_id == "HINGE_D1" else "d2"
    return (layout.input_paths[f"{slug}_fit_input"],
            layout.input_paths[f"{slug}_score_rows"],
            layout.input_paths[f"{slug}_hinge_state"])


def _row_hashes(arrays: Mapping[str, np.ndarray]) -> dict[str, str]:
    return r1.row_identity_hashes(
        physical_rows=arrays["physical_rows"], record_ids=arrays["record_ids"],
        prediction_origins=arrays["prediction_origins"],
        original_bundle_indices=arrays["original_bundle_indices"],
        cycles=arrays["cycles"], direct_predictions=arrays["direct_predictions"],
        missing_mask=arrays["missing_mask"])


def _validate_population(stage_id: str, role: str,
                         arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    expected_n = FIT_COUNTS[stage_id] if role == "fit" else SCORE_COUNTS[stage_id]
    vectors = [np.asarray(arrays[name]) for name in IDENTITY_FIELDS]
    if any(value.ndim != 1 or len(value) != expected_n for value in vectors):
        raise PilotError(f"Non-target identity vectors are misaligned for {stage_id}/{role}")
    p = np.asarray(arrays["direct_predictions"])
    raw_missing = np.asarray(arrays["missing_mask"])
    if raw_missing.dtype.kind != "b":
        raise PilotError(f"Missing-mask dtype is not boolean for {stage_id}/{role}")
    missing = raw_missing.astype(bool, copy=False)
    ids = np.asarray(arrays["record_ids"]).astype(str)
    physical = np.asarray(arrays["physical_rows"], dtype=np.int64)
    origins = np.asarray(arrays["prediction_origins"], dtype=np.int64)
    cycles = np.asarray(arrays["cycles"], dtype=np.int64)
    expected_present = PRESENT_FIT_COUNTS[stage_id] if role == "fit" else None
    if (p.dtype.kind != "f" or p.dtype.itemsize != 8 or not np.isfinite(p).all()
            or np.any((p < 0.0) | (p > 100.0)) or len(np.unique(ids)) != expected_n
            or len(np.unique(physical)) != expected_n or missing.sum() == expected_n):
        raise PilotError(f"DIRECT/missing/row identity validation failed for {stage_id}/{role}")
    if (not np.all(origins == EXPECTED_ORIGIN[(stage_id, role)])
            or tuple(sorted(np.unique(cycles).tolist())) != EXPECTED_CYCLES[(stage_id, role)]):
        raise PilotError(f"Origin/cycle boundary failed for {stage_id}/{role}")
    present = ~missing
    if expected_present is not None and int(present.sum()) != expected_present:
        raise PilotError(f"Unexpected present-fit count for {stage_id}: {int(present.sum())}")
    b = local_basis(p)
    active = present & (b > 0.0)
    if role == "fit" and int(active.sum()) < MIN_LOCAL_SUPPORT:
        raise PilotError(f"LOCAL support is unsupported for {stage_id}: {int(active.sum())} < {MIN_LOCAL_SUPPORT}")
    return {"fit_id": stage_id, "population": role, "n_all": expected_n,
        "n_present": int(present.sum()), "n_missing": int(missing.sum()),
        "local_positive_basis_present_rows": int(active.sum()),
        "origins": sorted(np.unique(origins).tolist()),
        "cycles": sorted(np.unique(cycles).tolist()),
        "row_identity_hashes": _row_hashes(arrays)}


def local_basis(direct: Any) -> np.ndarray:
    p = np.asarray(direct, dtype=np.float64)
    if p.ndim != 1 or not np.isfinite(p).all():
        raise PilotError("LOCAL basis requires finite one-dimensional DIRECT values")
    return np.maximum(0.0, 1.0 - np.abs(p - LOCAL_CENTER) / LOCAL_HALF_WIDTH)


def base_hinge_unclipped(direct: Any, parameters: Mapping[str, float]) -> np.ndarray:
    p = np.asarray(direct, dtype=np.float64)
    a = float(parameters["intercept"])
    sl = float(parameters["slope_low"])
    sh = float(parameters["slope_high"])
    if (p.ndim != 1 or not np.isfinite(p).all()
            or not all(math.isfinite(v) for v in (a, sl, sh))
            or sl < 0.0 or sh < 0.0):
        raise PilotError("Saved HINGE base parameters or predictions are invalid")
    return a + sl * np.minimum(p, 50.0) + sh * np.maximum(p - 50.0, 0.0)


def _clip_interval(low: float, high: float) -> tuple[float, float]:
    return (float(np.clip(low, DELTA_BOUNDS[0], DELTA_BOUNDS[1])),
            float(np.clip(high, DELTA_BOUNDS[0], DELTA_BOUNDS[1])))


def lower_weighted_median_interval(values: Any, weights: Any) -> tuple[float, float, int]:
    """Return exact-binary-weight median interval [low, high] and row count.

    The first cumulative weight reaching half is the deterministic lower
    endpoint. Binary float weights are converted to common-denominator integer
    units, so exact-half ties do not depend on a numerical tolerance.
    """
    x = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if (x.ndim != 1 or w.ndim != 1 or x.shape != w.shape or len(x) == 0
            or not np.isfinite(x).all() or not np.isfinite(w).all() or np.any(w <= 0.0)):
        raise PilotError("Weighted median requires aligned finite values and positive weights")
    ratios = [float(value).as_integer_ratio() for value in w]
    common_denominator = max(den for _, den in ratios)
    integer_weights = [num * (common_denominator // den) for num, den in ratios]
    order = np.argsort(x, kind="mergesort")
    grouped_values: list[float] = []
    grouped_weights: list[int] = []
    for index in order.tolist():
        value = float(x[index])
        units = integer_weights[index]
        if grouped_values and value == grouped_values[-1]:
            grouped_weights[-1] += units
        else:
            grouped_values.append(value)
            grouped_weights.append(units)
    total = sum(grouped_weights)
    cumulative = 0
    for index, (value, units) in enumerate(zip(grouped_values, grouped_weights, strict=True)):
        cumulative += units
        if 2 * cumulative >= total:
            lower = value
            upper = grouped_values[index + 1] if 2 * cumulative == total and index + 1 < len(grouped_values) else value
            return lower, upper, len(x)
    raise PilotError("Weighted median accumulation failed")


def lower_empirical_median_interval(values: Any) -> tuple[float, float, int]:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or len(x) == 0 or not np.isfinite(x).all():
        raise PilotError("Empirical median requires nonempty finite values")
    ordered = np.sort(x, kind="mergesort")
    return float(ordered[(len(ordered) - 1) // 2]), float(ordered[len(ordered) // 2]), len(ordered)


def fit_local_delta(direct: Any, targets: Any, present: Any,
                    hinge: Mapping[str, float]) -> dict[str, Any]:
    p = np.asarray(direct, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    mask = np.asarray(present, dtype=bool)
    if p.ndim != 1 or y.shape != p.shape or mask.shape != p.shape:
        raise PilotError("LOCAL fit vectors must be aligned")
    if not np.isfinite(y).all() or np.any((y < 0.0) | (y > 100.0)):
        raise PilotError("LOCAL fit targets are nonfinite or outside [0,100]")
    g = base_hinge_unclipped(p[mask], hinge)
    b = local_basis(p[mask])
    active = b > 0.0
    if int(active.sum()) < MIN_LOCAL_SUPPORT:
        raise PilotError(f"LOCAL support unsupported: {int(active.sum())} < {MIN_LOCAL_SUPPORT}")
    ratios = (y[mask][active] - g[active]) / b[active]
    weights = b[active]
    if not np.isfinite(ratios).all() or not np.isfinite(weights).all() or not np.any(weights > 0.0):
        raise PilotError("LOCAL residual ratios/weights are invalid")
    raw_low, raw_high, n_weighted = lower_weighted_median_interval(ratios, weights)
    bounded_low, bounded_high = _clip_interval(raw_low, raw_high)
    delta = bounded_low
    objective = math.fsum(float(v) for v in np.abs(y[mask] - (g + delta * b)))
    return {"delta": delta, "unbounded_median_interval": [raw_low, raw_high],
        "bounded_optimal_interval": [bounded_low, bounded_high],
        "solution_nonunique": bounded_low != bounded_high,
        "unbounded_solution_nonunique": raw_low != raw_high,
        "fit_present_rows": int(mask.sum()), "fit_active_rows": int(active.sum()),
        "weighted_rows": n_weighted,
        "objective_unweighted_sum_absolute_unclipped": objective}


def fit_global_control_delta(direct: Any, targets: Any, present: Any,
                             hinge: Mapping[str, float]) -> dict[str, Any]:
    p = np.asarray(direct, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    mask = np.asarray(present, dtype=bool)
    if p.ndim != 1 or y.shape != p.shape or mask.shape != p.shape or not np.any(mask):
        raise PilotError("GLOBAL_CONTROL needs aligned nonempty present fit rows")
    if not np.isfinite(y).all() or np.any((y < 0.0) | (y > 100.0)):
        raise PilotError("GLOBAL_CONTROL fit targets are nonfinite or outside [0,100]")
    residual = y[mask] - base_hinge_unclipped(p[mask], hinge)
    raw_low, raw_high, n = lower_empirical_median_interval(residual)
    bounded_low, bounded_high = _clip_interval(raw_low, raw_high)
    delta = bounded_low
    objective = math.fsum(float(v) for v in np.abs(residual - delta))
    return {"delta": delta, "unbounded_median_interval": [raw_low, raw_high],
        "bounded_optimal_interval": [bounded_low, bounded_high],
        "solution_nonunique": bounded_low != bounded_high,
        "unbounded_solution_nonunique": raw_low != raw_high,
        "fit_present_rows": int(mask.sum()), "fit_active_rows": int(mask.sum()),
        "weighted_rows": n, "objective_unweighted_sum_absolute_unclipped": objective}


def monotonicity_record(hinge: Mapping[str, float], delta: float) -> dict[str, Any]:
    if not math.isfinite(delta) or not (DELTA_BOUNDS[0] <= delta <= DELTA_BOUNDS[1]):
        raise PilotError("Correction delta is outside frozen bounds")
    left_local = float(hinge["slope_low"]) + delta / LOCAL_HALF_WIDTH
    right_local = float(hinge["slope_high"]) - delta / LOCAL_HALF_WIDTH
    slopes = {"outside_left": float(hinge["slope_low"]),
        "local_40_to_50": left_local, "local_50_to_60": right_local,
        "outside_right": float(hinge["slope_high"])}
    minimum = min(slopes.values())
    return {"piecewise_derivatives": slopes, "minimum_derivative": minimum,
            "monotone_non_decreasing": minimum >= 0.0}


def apply_correction(direct: Any, missing: Any, hinge: Mapping[str, float],
                     delta: float, *, local: bool) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(direct)
    miss = np.asarray(missing, dtype=bool)
    if (p.ndim != 1 or p.dtype.kind != "f" or p.dtype.itemsize != 8
            or miss.shape != p.shape or not np.isfinite(p).all()
            or np.any((p < 0.0) | (p > 100.0))):
        raise PilotError("Correction input must be aligned raw float64 DIRECT and missing mask")
    if not math.isfinite(delta) or not DELTA_BOUNDS[0] <= delta <= DELTA_BOUNDS[1]:
        raise PilotError("Correction delta is outside [-2,0]")
    present = ~miss
    raw_base = np.zeros(len(p), dtype=np.float64)
    raw_corrected = np.zeros(len(p), dtype=np.float64)
    raw_base[present] = base_hinge_unclipped(p[present], hinge)
    if local:
        raw_corrected[present] = raw_base[present] + delta * local_basis(p[present])
    else:
        raw_corrected[present] = raw_base[present] + delta
    out = p.copy()
    out[present] = np.clip(raw_corrected[present], 0.0, 100.0)
    if not np.array_equal(out[miss].view("<u8"), p[miss].view("<u8")):
        raise PilotError("Missing prediction route lost raw DIRECT float64 bits")
    return out.astype("<f8", copy=False), raw_corrected


def _load_state(layout: Layout, stage_id: str) -> dict[str, float]:
    state_path = layout.input_paths["d1_hinge_state" if stage_id == "HINGE_D1" else "d2_hinge_state"]
    state = _read_json(state_path)
    if state.get("status") != "LP_COMPLETE" or state.get("solver_result", {}).get("success") is not True:
        raise PilotError(f"Saved same-history HINGE state is incomplete: {stage_id}")
    result = state["solver_result"]
    params = {key: float(result[key]) for key in ("intercept", "slope_low", "slope_high")}
    if not all(math.isfinite(value) for value in params.values()):
        raise PilotError(f"Saved HINGE state contains nonfinite coefficients: {stage_id}")
    return params


def non_target_preflight(layout: Layout | None = None) -> dict[str, Any]:
    layout = DEFAULT_LAYOUT if layout is None else layout
    freeze, freeze_sha = verify_freeze(layout)
    resources: list[dict[str, Any]] = []
    population_rows = []
    all_score_ids: list[str] = []
    all_score_rows: list[int] = []
    with r4.SharedHeavyJobLock(layout.lock_path):
        verify_freeze(layout)
        resources.append(_resource_snapshot(layout, "saved_prediction_evaluation",
                                            "initial_locked_before_non_target_npz", initial=True))
        for stage_id in STAGES:
            fit_path, score_path, _ = _stage_files(layout, stage_id)
            fit = _read_npz_fields(fit_path, IDENTITY_FIELDS, allow_unloaded=("targets",))
            resources.append(_resource_snapshot(layout, "saved_prediction_evaluation",
                f"{stage_id}_fit_identity_loaded", initial=False))
            population_rows.append(_validate_population(stage_id, "fit", fit))
            score = _read_npz_fields(score_path, SCORE_NON_TARGET_FIELDS,
                allow_unloaded=("targets", "causal_global_all_rows", "causal_global_present_only"))
            resources.append(_resource_snapshot(layout, "saved_prediction_evaluation",
                f"{stage_id}_score_identity_loaded", initial=False))
            population_rows.append(_validate_population(stage_id, "score", score))
            p = np.asarray(score["direct_predictions"], dtype="<f8")
            missing = np.asarray(score["missing_mask"], dtype=bool)
            if not np.array_equal(np.asarray(score["raw_direct"], dtype="<f8").view("<u8"), p.view("<u8")):
                raise PilotError(f"Cached raw DIRECT score route differs: {stage_id}")
            if not np.array_equal(np.asarray(score["hinge_present_only"], dtype="<f8")[missing].view("<u8"),
                                  p[missing].view("<u8")):
                raise PilotError(f"Cached HINGE fallback differs from raw DIRECT: {stage_id}")
            params = _load_state(layout, stage_id)
            replay = r1.apply_hinge_present_only(p, missing, params).astype("<f8", copy=False)
            if not np.array_equal(replay.view("<u8"),
                                  np.asarray(score["hinge_present_only"], dtype="<f8").view("<u8")):
                raise PilotError(f"Saved fixed-p50 HINGE score prediction replay failed: {stage_id}")
            all_score_ids.extend(np.asarray(score["record_ids"]).astype(str).tolist())
            all_score_rows.extend(np.asarray(score["physical_rows"], dtype=np.int64).tolist())
            del fit, score
        if (len(all_score_ids) != 15_953 or len(set(all_score_ids)) != 15_953
                or len(set(all_score_rows)) != 15_953):
            raise PilotError("Pooled D1/D2 score rows do not have 15,953 unique identities")
        resources.append(_resource_snapshot(layout, "saved_prediction_evaluation",
            "after_non_target_identity_replay", initial=False))
    body = {"schema_version": 1, "experiment_id": NAMESPACE.upper() + "_R1",
        "status": "PASS_NON_TARGET_IDENTITY_PREFLIGHT",
        "input_freeze_sha256": freeze_sha, "source_commit": freeze["source_commit"],
        "populations": population_rows,
        "pooled_score_rows": 15_953, "pooled_score_record_ids_unique": True,
        "pooled_score_physical_rows_unique": True,
        "saved_hinge_present_only_replay": True,
        "local_support_counts_present_fit_rows": {
            row["fit_id"]: row["local_positive_basis_present_rows"]
            for row in population_rows if row["population"] == "fit"},
        "saved_hinge_monotonicity_at_delta_minus_2": {
            stage: monotonicity_record(_load_state(layout, stage), -2.0)
            for stage in STAGES},
        "resource_snapshots": resources,
        "target_arrays_present_in_npz_but_loaded": False,
        "candidate_target_values_loaded": False, "calibrator_fit_starts": 0,
        "ledger_created": False, "score_target_values_loaded": False,
        "test_or_sample_accessed": False, "e79_accessed": False,
        "csv_created": False, "kaggle_upload": False}
    body["signature"] = canonical_sha(body)
    digest = write_json_once(layout.preflight, body)
    return {"status": body["status"], "path": str(layout.preflight.resolve()),
            "sha256": digest, "populations": len(population_rows),
            "target_arrays_loaded": False}


def verify_preflight(layout: Layout | None = None) -> tuple[dict[str, Any], str]:
    layout = DEFAULT_LAYOUT if layout is None else layout
    freeze, freeze_sha = verify_freeze(layout)
    body = _read_json(layout.preflight)
    signature = body.pop("signature", None)
    if signature != canonical_sha(body):
        raise PilotError("Non-target preflight signature mismatch")
    body["signature"] = signature
    if (body.get("status") != "PASS_NON_TARGET_IDENTITY_PREFLIGHT"
            or body.get("input_freeze_sha256") != freeze_sha
            or body.get("source_commit") != freeze.get("source_commit")
            or body.get("candidate_target_values_loaded") is not False
            or body.get("calibrator_fit_starts") != 0
            or body.get("score_target_values_loaded") is not False):
        raise PilotError("Non-target preflight receipt boundary or freeze binding failed")
    return body, sha256_file(layout.preflight)


def _validate_root_pass(layout: Layout, freeze_sha: str,
                        preflight_sha: str) -> tuple[dict[str, Any], str]:
    receipt = _validate_signed(layout.root_pass, verdict="PASS")
    expected_fields = {"verdict", "experiment_id", "pass_kind", "source_commit",
        "prefit_freeze_sha256", "identity_preflight_sha256", "authorized_fit_ids",
        "calibrator_fit_cap", "authorized_calibrator_fits", "base_model_fits",
        "preprocessing_fits", "official_lp_starts", "prior_three_slot_budget_reset",
        "score_target_access_authorized", "test_or_sample_authorized",
        "kaggle_upload_authorized", "signature"}
    if set(receipt) != expected_fields:
        raise PilotError("ROOT_SOURCE_PASS_R1 fields differ from the exact callback schema")
    if (receipt.get("experiment_id") != NAMESPACE.upper() + "_R1"
            or receipt.get("pass_kind") != "CENTER_LOCAL_DIAGNOSTIC"
            or receipt.get("source_commit") != _git_head(layout)
            or receipt.get("prefit_freeze_sha256") != freeze_sha
            or receipt.get("identity_preflight_sha256") != preflight_sha
            or receipt.get("authorized_fit_ids") != list(FIT_IDS)
            or receipt.get("calibrator_fit_cap") != 4
            or receipt.get("authorized_calibrator_fits") != 4
            or receipt.get("base_model_fits") != 0
            or receipt.get("preprocessing_fits") != 0
            or receipt.get("official_lp_starts") != 0
            or receipt.get("prior_three_slot_budget_reset") is not False
            or receipt.get("score_target_access_authorized") is not True
            or receipt.get("test_or_sample_authorized") is not False
            or receipt.get("kaggle_upload_authorized") is not False):
        raise PilotError("ROOT_SOURCE_PASS_R1 is not the exact four-fit source authorization")
    return receipt, sha256_file(layout.root_pass)


def _row_identity_sha(arrays: Mapping[str, np.ndarray]) -> str:
    return canonical_sha(_row_hashes(arrays))


def _fit_state_path(layout: Layout, fit_id: str) -> Path:
    return layout.fit_dir / f"{fit_id}.json"


def _read_ledger(layout: Layout) -> dict[str, list[dict[str, Any]]]:
    events: dict[str, list[dict[str, Any]]] = {}
    ordered_events: list[tuple[str, str]] = []
    if not layout.ledger.exists():
        return events
    try:
        lines = layout.ledger.read_bytes().splitlines()
        for raw in lines:
            item = json.loads(raw)
            fit_id, event = item["fit_id"], item["event"]
            if fit_id not in FIT_IDS or event not in {"RESERVE", "FIT_STARTED", "FIT_COMPLETE", "FIT_FAILED"}:
                raise PilotError("Calibrator ledger contains unknown fit or event")
            events.setdefault(fit_id, []).append(item)
            ordered_events.append((fit_id, event))
    except (ValueError, KeyError, TypeError) as exc:
        raise PilotError("Calibrator ledger is malformed or partially written; no retry") from exc
    reserved = []
    starts = 0
    for fit_id in FIT_IDS:
        records = events.get(fit_id, [])
        kinds = [record["event"] for record in records]
        if kinds not in ([], ["RESERVE"], ["RESERVE", "FIT_STARTED"],
                         ["RESERVE", "FIT_STARTED", "FIT_COMPLETE"],
                         ["RESERVE", "FIT_STARTED", "FIT_FAILED"]):
            raise PilotError(f"Invalid ledger sequence for {fit_id}; no retry")
        if kinds:
            reserved.append(fit_id)
        if "FIT_STARTED" in kinds:
            starts += 1
        if any(record.get("schema_version") != 1 for record in records):
            raise PilotError(f"Ledger schema mismatch for {fit_id}")
        if kinds:
            ids = [record.get("reservation_id") for record in records]
            if len(set(ids)) != 1 or any(not record.get("source_commit")
                    or not record.get("prefit_freeze_sha256")
                    or not record.get("root_source_pass_sha256") for record in records):
                raise PilotError(f"Ledger bindings differ for {fit_id}; no retry")
            for key in ("reservation_id", "source_commit", "prefit_freeze_sha256",
                        "root_source_pass_sha256", "fit_input_sha256"):
                if len({record.get(key) for record in records}) != 1:
                    raise PilotError(f"Ledger {key} binding differs for {fit_id}; no retry")
            expected_input = _stage_files(layout, FIT_TO_STAGE[fit_id])[0]
            if records[0].get("fit_input_sha256") != sha256_file(expected_input):
                raise PilotError(f"Ledger fit input no longer matches the frozen source: {fit_id}")
            expected_reservation = hashlib.sha256(
                f"{NAMESPACE}:R1:{fit_id}:{records[0]['prefit_freeze_sha256']}:{records[0]['root_source_pass_sha256']}".encode("utf-8")).hexdigest()
            if records[0].get("reservation_id") != expected_reservation:
                raise PilotError(f"Ledger reservation ID is not deterministic: {fit_id}")
            if not all(item.get(key) for item in records for key in
                    ("source_commit", "prefit_freeze_sha256",
                     "root_source_pass_sha256", "fit_input_sha256")):
                raise PilotError(f"Ledger binding field is empty: {fit_id}")
        if "FIT_COMPLETE" in kinds:
            state_path = _fit_state_path(layout, fit_id)
            event = records[-1]
            if (Path(event.get("fit_state_path", "")).resolve() != state_path.resolve()
                    or not state_path.is_file() or sha256_file(state_path) != event.get("fit_state_sha256")):
                raise PilotError(f"Completed ledger state hash differs for {fit_id}")
            state = _read_json(state_path)
            if (state.get("fit_id") != fit_id or state.get("status") != "FIT_COMPLETE"
                    or state.get("source_commit") != records[0].get("source_commit")
                    or state.get("prefit_freeze_sha256") != records[0].get("prefit_freeze_sha256")
                    or state.get("root_source_pass_sha256") != records[0].get("root_source_pass_sha256")
                    or state.get("fit_input_sha256") != records[0].get("fit_input_sha256")):
                raise PilotError(f"Completed fit state bindings differ from ledger: {fit_id}")
        if "FIT_FAILED" in kinds:
            raise PilotError(f"Consumed calibrator fit failed for {fit_id}; no retry")
    if starts > 4 or reserved != list(FIT_IDS[:len(reserved)]):
        raise PilotError("Ledger exceeds cap four or is not a contiguous fit prefix")
    expected_order = [(fit_id, event["event"]) for fit_id in reserved
                      for event in events[fit_id]]
    if ordered_events != expected_order:
        raise PilotError("Ledger events are not in append-only fit order; no retry")
    all_started = [record for records in events.values() for record in records]
    for key in ("source_commit", "prefit_freeze_sha256", "root_source_pass_sha256"):
        if len({record.get(key) for record in all_started}) > 1:
            raise PilotError(f"Ledger cross-fit {key} binding differs; no retry")
    return events


def _append_ledger(layout: Layout, event: Mapping[str, Any]) -> None:
    layout.ledger.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(dict(event), sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(",", ":")).encode("utf-8") + b"\n"
    with layout.ledger.open("ab") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _candidate_fit(layout: Layout, fit_id: str, freeze_sha: str,
                   pass_sha: str) -> dict[str, Any]:
    stage_id = FIT_TO_STAGE[fit_id]
    fit_path, _, _ = _stage_files(layout, stage_id)
    identity_arrays = _read_npz_fields(fit_path, IDENTITY_FIELDS,
                                       allow_unloaded=("targets",))
    identity = _validate_population(stage_id, "fit", identity_arrays)
    verified_row_sha = _row_identity_sha(identity_arrays)
    arrays = _read_npz_fields(fit_path, IDENTITY_FIELDS, load_targets=True)
    if _row_identity_sha(arrays) != verified_row_sha:
        raise PilotError(f"Fit row identity changed before target load: {fit_id}")
    targets = np.asarray(arrays["targets"], dtype=np.float64)
    if not np.isfinite(targets).all() or np.any((targets < 0.0) | (targets > 100.0)):
        raise PilotError(f"Fit targets invalid for {fit_id}")
    params = _load_state(layout, stage_id)
    present = ~np.asarray(arrays["missing_mask"], dtype=bool)
    if fit_id.endswith("_LOCAL"):
        fit = fit_local_delta(arrays["direct_predictions"], targets, present, params)
        local = True
    else:
        fit = fit_global_control_delta(arrays["direct_predictions"], targets, present, params)
        local = False
    monotonicity = monotonicity_record(params, fit["delta"])
    if local and not monotonicity["monotone_non_decreasing"]:
        raise PilotError(f"LOCAL monotonicity failed for {fit_id}; bounds remain frozen")
    output = {"schema_version": 1, "experiment_id": NAMESPACE.upper() + "_R1",
        "status": "FIT_COMPLETE", "fit_id": fit_id, "stage_id": stage_id,
        "arm": "LOCAL" if local else "GLOBAL_CONTROL",
        "recipe": {"base": "saved fixed-p50 HINGE", "center": LOCAL_CENTER,
            "half_width": LOCAL_HALF_WIDTH if local else None,
            "delta_bounds": list(DELTA_BOUNDS), "clipping_in_objective": False,
            "fit_rows": "present only", "objective": "unweighted sum absolute"},
        "fit": fit, "monotonicity": monotonicity,
        "fit_population": identity,
        "target_sha256": r1.tagged_array_sha256(targets, dtype="<f8"),
        "saved_hinge_parameters": params,
        "fit_input_sha256": sha256_file(fit_path),
        "hinge_state_sha256": sha256_file(layout.input_paths["d1_hinge_state" if stage_id == "HINGE_D1" else "d2_hinge_state"]),
        "source_commit": _git_head(layout), "prefit_freeze_sha256": freeze_sha,
        "root_source_pass_sha256": pass_sha, "counted_as_one_calibrator_fit": True,
        "base_model_fit": False, "preprocessing_fit": False}
    return output


def _make_prediction_arrays(layout: Layout, stage_id: str,
                            fits: Mapping[str, Mapping[str, Any]]) -> dict[str, np.ndarray]:
    _, score_path, _ = _stage_files(layout, stage_id)
    arrays = _read_npz_fields(score_path, SCORE_NON_TARGET_FIELDS,
        allow_unloaded=("targets", "causal_global_all_rows", "causal_global_present_only"))
    _validate_population(stage_id, "score", arrays)
    direct = np.asarray(arrays["direct_predictions"], dtype="<f8")
    missing = np.asarray(arrays["missing_mask"], dtype=bool)
    params = _load_state(layout, stage_id)
    hinge = r1.apply_hinge_present_only(direct, missing, params).astype("<f8", copy=False)
    cached = np.asarray(arrays["hinge_present_only"], dtype="<f8")
    if not np.array_equal(hinge.view("<u8"), cached.view("<u8")):
        raise PilotError(f"Saved hinge baseline changed for {stage_id}")
    local_fit = fits["D1_LOCAL" if stage_id == "HINGE_D1" else "D2_LOCAL"]
    control_fit = fits["D1_GLOBAL_CONTROL" if stage_id == "HINGE_D1" else "D2_GLOBAL_CONTROL"]
    local, local_raw = apply_correction(direct, missing, params,
        float(local_fit["fit"]["delta"]), local=True)
    control, control_raw = apply_correction(direct, missing, params,
        float(control_fit["fit"]["delta"]), local=False)
    changed = np.abs(direct - LOCAL_CENTER) < LOCAL_HALF_WIDTH
    outside = (~missing) & ~changed
    local_direct_present = direct[~missing]
    if np.any(local_basis(local_direct_present)[np.abs(local_direct_present - LOCAL_CENTER) >= LOCAL_HALF_WIDTH] != 0.0):
        raise PilotError("LOCAL basis must be exactly zero outside its fixed support")
    outside_effect = local_raw[outside] - base_hinge_unclipped(direct[outside], params)
    if np.any(outside_effect != 0.0):
        raise PilotError("LOCAL pre-clipping effect is nonzero outside [40,60]")
    if not np.array_equal(local[missing].view("<u8"), direct[missing].view("<u8")):
        raise PilotError("LOCAL prediction missing route is not raw DIRECT bit-exact")
    if not np.array_equal(control[missing].view("<u8"), direct[missing].view("<u8")):
        raise PilotError("GLOBAL_CONTROL missing route is not raw DIRECT bit-exact")
    return {"physical_rows": np.asarray(arrays["physical_rows"], dtype=np.int64),
        "record_ids": np.asarray(arrays["record_ids"]).astype(str),
        "prediction_origins": np.asarray(arrays["prediction_origins"], dtype=np.int8),
        "original_bundle_indices": np.asarray(arrays["original_bundle_indices"], dtype=np.int64),
        "cycles": np.asarray(arrays["cycles"], dtype=np.int16),
        "direct_predictions": direct.copy(), "missing_mask": missing.copy(),
        "hinge_baseline": hinge, "local_prediction": local,
        "global_control_prediction": control}


def _verify_or_write_prediction_file(path: Path,
                                     arrays: Mapping[str, np.ndarray]) -> str:
    if path.exists():
        loaded = _read_npz_fields(path, tuple(arrays))
        if set(loaded) != set(arrays):
            raise PilotError(f"Existing prediction file has a different schema: {path}")
        for key, value in arrays.items():
            left = np.asarray(loaded[key])
            right = np.asarray(value)
            if left.dtype.kind == "f" and left.dtype.itemsize == 8:
                equal = np.array_equal(left.astype("<f8", copy=False).view("<u8"),
                                       right.astype("<f8", copy=False).view("<u8"))
            else:
                equal = np.array_equal(left, right)
            if not equal:
                raise PilotError(f"Existing committed prediction differs: {path}/{key}")
        return sha256_file(path)
    return write_npz_once(path, arrays)


def _load_completed_fits(layout: Layout) -> dict[str, dict[str, Any]]:
    fits = {}
    for fit_id in FIT_IDS:
        path = _fit_state_path(layout, fit_id)
        if not path.is_file():
            raise PilotError(f"Completed fit state is missing: {fit_id}")
        state = _read_json(path)
        if (state.get("fit_id") != fit_id or state.get("status") != "FIT_COMPLETE"
                or state.get("arm") != ("LOCAL" if fit_id.endswith("_LOCAL") else "GLOBAL_CONTROL")
                or state.get("counted_as_one_calibrator_fit") is not True
                or state.get("base_model_fit") is not False
                or state.get("preprocessing_fit") is not False):
            raise PilotError(f"Fit state identity/status mismatch: {fit_id}")
        fits[fit_id] = state
    return fits


def _verify_prediction_freeze(layout: Layout, freeze_sha: str,
                               pass_sha: str) -> tuple[dict[str, Any], str]:
    body = _read_json(layout.prediction_freeze)
    signature = body.pop("signature", None)
    if signature != canonical_sha(body):
        raise PilotError("Prediction freeze signature mismatch")
    body["signature"] = signature
    if (body.get("status") != "SCORE_PREDICTIONS_FROZEN_BEFORE_SCORE_TARGET_LOAD"
            or body.get("prefit_freeze_sha256") != freeze_sha
            or body.get("root_source_pass_sha256") != pass_sha
            or body.get("source_commit") != _git_head(layout)
            or body.get("score_targets_loaded") is not False
            or body.get("test_or_sample_accessed") is not False
            or body.get("e79_accessed") is not False
            or body.get("csv_created") is not False
            or body.get("kaggle_upload") is not False):
        raise PilotError("Prediction freeze does not bind current authorization")
    records = body.get("prediction_files", [])
    if {record.get("stage_id") for record in records} != set(STAGES) or len(records) != len(STAGES):
        raise PilotError("Prediction freeze must bind exactly the two frozen score windows")
    for record in records:
        stage_id = record["stage_id"]
        path = layout.prediction_dir / f"{stage_id}_PREDICTIONS.npz"
        saved_path = Path(record.get("path", ""))
        if (saved_path.resolve() != path.resolve() or not path.is_file()
                or sha256_file(path) != record.get("sha256")):
            raise PilotError(f"Frozen score prediction changed: {path}")
        arrays = _read_npz_fields(path, IDENTITY_FIELDS +
            ("hinge_baseline", "local_prediction", "global_control_prediction"))
        if (len(arrays["record_ids"]) != SCORE_COUNTS[stage_id]
                or record.get("n_rows") != SCORE_COUNTS[stage_id]
                or record.get("row_identity_sha256") != _row_identity_sha(arrays)
                or record.get("score_target_values_loaded") is not False):
            raise PilotError(f"Frozen score prediction identity/boundary differs: {stage_id}")
    fit_hashes = body.get("fit_state_hashes", {})
    if set(fit_hashes) != set(FIT_IDS):
        raise PilotError("Prediction freeze does not bind all four calibrator states")
    for fit_id, digest in fit_hashes.items():
        path = _fit_state_path(layout, fit_id)
        if not path.is_file() or sha256_file(path) != digest:
            raise PilotError(f"Frozen calibrator state changed: {fit_id}")
    return body, sha256_file(layout.prediction_freeze)


def run_fits(layout: Layout | None = None) -> dict[str, Any]:
    layout = DEFAULT_LAYOUT if layout is None else layout
    freeze, freeze_sha = verify_freeze(layout)
    preflight, preflight_sha = verify_preflight(layout)
    _, pass_sha = _validate_root_pass(layout, freeze_sha, preflight_sha)
    if layout.prediction_freeze.exists():
        _verify_prediction_freeze(layout, freeze_sha, pass_sha)
        ledger = _read_ledger(layout)
        if any([record["event"] for record in ledger.get(fit_id, [])]
               != ["RESERVE", "FIT_STARTED", "FIT_COMPLETE"] for fit_id in FIT_IDS):
            raise PilotError("Prediction freeze exists without the four completed fit ledger slots")
        _load_completed_fits(layout)
        return {"status": "PASS_FITS_AND_PREDICTIONS_ALREADY_FROZEN",
                "prediction_freeze_sha256": sha256_file(layout.prediction_freeze)}
    resources: list[dict[str, Any]] = []
    with r4.SharedHeavyJobLock(layout.lock_path):
        verify_freeze(layout)
        verify_preflight(layout)
        _validate_root_pass(layout, freeze_sha, preflight_sha)
        resources.append(_resource_snapshot(layout, "small_calibration_lp",
                                            "initial_locked_before_fit_reservation", initial=True))
        ledger = _read_ledger(layout)
        for fit_id in FIT_IDS:
            records = ledger.get(fit_id, [])
            kinds = [record["event"] for record in records]
            if kinds == ["RESERVE", "FIT_STARTED", "FIT_COMPLETE"]:
                continue
            if kinds:
                raise PilotError(f"Consumed or ambiguous fit slot {fit_id}; no retry")
            reservation_id = hashlib.sha256(
                f"{NAMESPACE}:R1:{fit_id}:{freeze_sha}:{pass_sha}".encode("utf-8")).hexdigest()
            common = {"schema_version": 1, "experiment_id": NAMESPACE.upper() + "_R1",
                "fit_id": fit_id, "reservation_id": reservation_id,
                "source_commit": freeze["source_commit"],
                "prefit_freeze_sha256": freeze_sha,
                "fit_input_sha256": sha256_file(_stage_files(layout, FIT_TO_STAGE[fit_id])[0]),
                "root_source_pass_sha256": pass_sha}
            _append_ledger(layout, {**common, "event": "RESERVE", "timestamp_utc_ns": time.time_ns()})
            _append_ledger(layout, {**common, "event": "FIT_STARTED", "timestamp_utc_ns": time.time_ns()})
            try:
                resources.append(_resource_snapshot(layout, "small_calibration_lp",
                    f"{fit_id}_before_target_load", initial=False))
                verify_freeze(layout)
                verify_preflight(layout)
                _validate_root_pass(layout, freeze_sha, preflight_sha)
                with threadpool_limits(limits=2):
                    state = _candidate_fit(layout, fit_id, freeze_sha, pass_sha)
                resources.append(_resource_snapshot(layout, "small_calibration_lp",
                    f"{fit_id}_after_fit", initial=False))
                state["resource_snapshots"] = resources[-2:]
                state_sha = write_json_once(_fit_state_path(layout, fit_id), state)
                _append_ledger(layout, {**common, "event": "FIT_COMPLETE",
                    "fit_state_path": str(_fit_state_path(layout, fit_id).resolve()),
                    "fit_state_sha256": state_sha, "timestamp_utc_ns": time.time_ns()})
            except Exception as exc:
                _append_ledger(layout, {**common, "event": "FIT_FAILED",
                    "reason": f"{type(exc).__name__}: {exc}", "timestamp_utc_ns": time.time_ns()})
                raise
            ledger = _read_ledger(layout)
        completed_ledger = _read_ledger(layout)
        if ([fit_id for fit_id in FIT_IDS if completed_ledger.get(fit_id, [])] != list(FIT_IDS)
                or any([record["event"] for record in completed_ledger[fit_id]]
                       != ["RESERVE", "FIT_STARTED", "FIT_COMPLETE"] for fit_id in FIT_IDS)):
            raise PilotError("Four calibrator fit slots did not complete in order")
        fits = _load_completed_fits(layout)
        layout.prediction_dir.mkdir(parents=True, exist_ok=True)
        prediction_records = []
        for stage_id in STAGES:
            arrays = _make_prediction_arrays(layout, stage_id, fits)
            path = layout.prediction_dir / f"{stage_id}_PREDICTIONS.npz"
            digest = _verify_or_write_prediction_file(path, arrays)
            prediction_records.append({"stage_id": stage_id, "path": str(path.resolve()),
                "sha256": digest, "n_rows": int(len(arrays["record_ids"])),
                "row_identity_sha256": _row_identity_sha(arrays),
                "score_target_values_loaded": False})
            del arrays
        resources.append(_resource_snapshot(layout, "saved_prediction_evaluation",
            "after_predictions_before_target_access", initial=False))
        body = {"schema_version": 1, "experiment_id": NAMESPACE.upper() + "_R1",
            "status": "SCORE_PREDICTIONS_FROZEN_BEFORE_SCORE_TARGET_LOAD",
            "source_commit": freeze["source_commit"], "prefit_freeze_sha256": freeze_sha,
            "identity_preflight_sha256": preflight_sha,
            "root_source_pass_sha256": pass_sha,
            "fit_state_hashes": {fit_id: sha256_file(_fit_state_path(layout, fit_id)) for fit_id in FIT_IDS},
            "prediction_files": prediction_records,
            "resource_snapshots": resources,
            "score_targets_loaded": False, "test_or_sample_accessed": False,
            "e79_accessed": False, "csv_created": False, "kaggle_upload": False}
        body["signature"] = canonical_sha(body)
        pred_sha = write_json_once(layout.prediction_freeze, body)
    return {"status": "PASS_FOUR_CALIBRATORS_PREDICTIONS_FROZEN",
            "prediction_freeze_sha256": pred_sha,
            "fit_state_hashes": body["fit_state_hashes"],
            "prediction_files": prediction_records}


def _bin_index(direct: np.ndarray) -> np.ndarray:
    p = np.asarray(direct, dtype=np.float64)
    if not np.isfinite(p).all() or np.any((p < 0.0) | (p > 100.0)):
        raise PilotError("Raw DIRECT values are outside frozen [0,100] domain")
    return np.clip(np.searchsorted(np.asarray(BIN_EDGES), p, side="right") - 1,
                   0, len(BIN_LABELS) - 1).astype(np.int8)


def _metric_pack(prediction: np.ndarray, target: np.ndarray, minimum: int) -> dict[str, Any]:
    n = int(len(target))
    if n < minimum:
        return {"status": "UNKNOWN_UNSUPPORTED", "n": n, "mae": None,
                "mean_error": None, "p90_absolute_error": None}
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return {"status": "SUPPORTED", "n": n,
        "mae": float(np.mean(np.abs(error))), "mean_error": float(np.mean(error)),
        "p90_absolute_error": float(np.quantile(np.abs(error), 0.90))}


def _group_metrics(mask: np.ndarray, present: np.ndarray, target: np.ndarray,
                   predictions: Mapping[str, np.ndarray], *, minimum: int) -> dict[str, Any]:
    all_mask = np.asarray(mask, dtype=bool)
    present_mask = all_mask & present
    result = {"n_all": int(all_mask.sum()), "n_present": int(present_mask.sum()),
        "n_missing": int((all_mask & ~present).sum()), "metrics": {"all_rows": {}, "present_only": {}}}
    for name, prediction in predictions.items():
        result["metrics"]["all_rows"][name] = _metric_pack(
            prediction[all_mask], target[all_mask], minimum)
        result["metrics"]["present_only"][name] = _metric_pack(
            prediction[present_mask], target[present_mask], minimum)
    result["gain_over_hinge"] = {"all_rows": {}, "present_only": {}}
    for scope in ("all_rows", "present_only"):
        baseline = result["metrics"][scope]["HINGE"]
        for name in ("LOCAL", "GLOBAL_CONTROL"):
            candidate = result["metrics"][scope][name]
            if baseline["status"] == "SUPPORTED" and candidate["status"] == "SUPPORTED":
                result["gain_over_hinge"][scope][name] = baseline["mae"] - candidate["mae"]
            else:
                result["gain_over_hinge"][scope][name] = None
    return result


def _population_metrics(direct: np.ndarray, cycles: np.ndarray, missing: np.ndarray,
                        target: np.ndarray, predictions: Mapping[str, np.ndarray]) -> dict[str, Any]:
    present = ~np.asarray(missing, dtype=bool)
    bins = _bin_index(direct)
    by_bin = [{"bin_index": i, "label": label,
        **_group_metrics(bins == i, present, target, predictions, minimum=MIN_GROUP_SUPPORT)}
        for i, label in enumerate(BIN_LABELS)]
    by_cycle = [{"cycle": int(cycle),
        **_group_metrics(np.asarray(cycles) == cycle, present, target, predictions,
                         minimum=MIN_CYCLE_SUPPORT)}
        for cycle in sorted(np.unique(cycles).tolist())]
    tails = {}
    for name, mask in (("target_le_5", target <= 5.0), ("target_ge_95", target >= 95.0)):
        tails[name] = _group_metrics(mask, present, target, predictions, minimum=MIN_GROUP_SUPPORT)
    overall = _group_metrics(np.ones(len(target), dtype=bool), present, target,
                             predictions, minimum=1)
    return {"n_all": int(len(target)), "n_present": int(present.sum()),
        "n_missing": int((~present).sum()), "overall": overall,
        "by_raw_direct_bin": by_bin, "by_cycle": by_cycle,
        "target_tails": tails}


def _targeted_region_gain(direct: np.ndarray, present: np.ndarray,
                          target: np.ndarray, predictions: Mapping[str, np.ndarray]) -> dict[str, Any]:
    active = present & (direct > 40.0) & (direct < 60.0)
    metrics = {name: _metric_pack(pred[active], target[active], MIN_GROUP_SUPPORT)
               for name, pred in predictions.items()}
    return {"n_present": int(active.sum()), "support": "SUPPORTED" if active.sum() >= MIN_GROUP_SUPPORT else "UNKNOWN_UNSUPPORTED",
        "metrics": metrics,
        "local_gain_over_hinge": (metrics["HINGE"]["mae"] - metrics["LOCAL"]["mae"]
            if metrics["HINGE"]["status"] == "SUPPORTED" and metrics["LOCAL"]["status"] == "SUPPORTED" else None)}


def _score_targets(layout: Layout, stage_id: str,
                   frozen_predictions: Mapping[str, np.ndarray]) -> dict[str, Any]:
    _, score_path, _ = _stage_files(layout, stage_id)
    arrays = _read_npz_fields(score_path, IDENTITY_FIELDS, load_targets=True,
        allow_unloaded=("raw_direct", "hinge_present_only", "causal_global_all_rows",
                        "causal_global_present_only"))
    identity = _validate_population(stage_id, "score", arrays)
    pred_ids = np.asarray(frozen_predictions["record_ids"]).astype(str)
    if not np.array_equal(pred_ids, np.asarray(arrays["record_ids"]).astype(str)):
        raise PilotError(f"Frozen prediction row order differs from score targets: {stage_id}")
    for name in ("physical_rows", "prediction_origins", "original_bundle_indices", "cycles",
                 "direct_predictions", "missing_mask"):
        left, right = np.asarray(frozen_predictions[name]), np.asarray(arrays[name])
        if name == "direct_predictions":
            equal = np.array_equal(left.astype("<f8", copy=False).view("<u8"),
                                   right.astype("<f8", copy=False).view("<u8"))
        else:
            equal = np.array_equal(left, right)
        if not equal:
            raise PilotError(f"Frozen prediction row identity changed: {stage_id}/{name}")
    target = np.asarray(arrays["targets"], dtype="<f8")
    if not np.isfinite(target).all() or np.any((target < 0.0) | (target > 100.0)):
        raise PilotError(f"Score targets are nonfinite or outside [0,100]: {stage_id}")
    predictions = {"HINGE": np.asarray(frozen_predictions["hinge_baseline"], dtype="<f8"),
        "LOCAL": np.asarray(frozen_predictions["local_prediction"], dtype="<f8"),
        "GLOBAL_CONTROL": np.asarray(frozen_predictions["global_control_prediction"], dtype="<f8")}
    if any(not np.isfinite(value).all() or np.any((value < 0.0) | (value > 100.0))
           for value in predictions.values()):
        raise PilotError(f"Score predictions are nonfinite or outside [0,100]: {stage_id}")
    missing = np.asarray(arrays["missing_mask"], dtype=bool)
    direct = np.asarray(arrays["direct_predictions"], dtype="<f8")
    for value in predictions.values():
        if not np.array_equal(value[missing].view("<u8"), direct[missing].view("<u8")):
            raise PilotError(f"Score missing route changed for {stage_id}")
    return {"stage_id": stage_id, "score_rows": identity,
        "target_sha256": r1.tagged_array_sha256(target, dtype="<f8"),
        "metrics": _population_metrics(direct, arrays["cycles"], missing, target, predictions),
        "targeted_region": _targeted_region_gain(direct, ~missing, target, predictions),
        "prediction_hashes": {name: hashlib.sha256(
            np.ascontiguousarray(value, dtype="<f8").tobytes()).hexdigest()
            for name, value in predictions.items()},
        "predictions": predictions, "targets": target,
        "direct": direct, "missing": missing}


def _supported_all_row_group_harm(stage_result: Mapping[str, Any]) -> float | None:
    values = []
    population = stage_result["metrics"]
    groups = [population["overall"], *population["by_raw_direct_bin"],
              *population["by_cycle"], *population["target_tails"].values()]
    for group in groups:
        metric = group["metrics"]["all_rows"]
        hinge = metric["HINGE"]
        local = metric["LOCAL"]
        if hinge["status"] == "SUPPORTED" and local["status"] == "SUPPORTED":
            values.append(local["mae"] - hinge["mae"])
    return max(values) if values else None


def _score_gates(results: Mapping[str, Mapping[str, Any]],
                 fits: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    stage_gains = {}
    group_harms = {}
    region_gains = {}
    for stage_id in STAGES:
        result = results[stage_id]
        overall = result["metrics"]["overall"]["gain_over_hinge"]["all_rows"]
        local_mae = result["metrics"]["overall"]["metrics"]["all_rows"]["LOCAL"]["mae"]
        hinge_mae = result["metrics"]["overall"]["metrics"]["all_rows"]["HINGE"]["mae"]
        control_mae = result["metrics"]["overall"]["metrics"]["all_rows"]["GLOBAL_CONTROL"]["mae"]
        stage_gains[stage_id] = {"local_over_hinge": overall["LOCAL"],
            "local_over_global_control": control_mae - local_mae,
            "global_control_over_hinge": hinge_mae - control_mae}
        group_harms[stage_id] = _supported_all_row_group_harm(result)
        region_gains[stage_id] = result["targeted_region"]["local_gain_over_hinge"]
    score_counts = {stage: SCORE_COUNTS[stage] for stage in STAGES}
    pooled_n = sum(score_counts.values())
    if pooled_n != 15_953:
        raise PilotError("Pooled score denominator differs from the frozen row count")
    pooled_mae = {}
    for policy in ("HINGE", "LOCAL", "GLOBAL_CONTROL"):
        pooled_mae[policy] = sum(results[stage]["metrics"]["overall"]["metrics"]["all_rows"][policy]["mae"]
            * SCORE_COUNTS[stage] for stage in STAGES) / pooled_n
    pooled_gain = {name: pooled_mae["HINGE"] - pooled_mae[name]
                   for name in ("LOCAL", "GLOBAL_CONTROL")}
    local_states = {stage: ("D1_LOCAL" if stage == "HINGE_D1" else "D2_LOCAL") for stage in STAGES}
    fit_deltas = {stage: float(fits[fit_id]["fit"]["delta"])
                  for stage, fit_id in local_states.items()}
    mechanism = (all(fit_deltas[stage] < 0.0 for stage in STAGES)
        and all(stage_gains[stage]["local_over_hinge"] > FIT_EPSILON
                and stage_gains[stage]["local_over_global_control"] > FIT_EPSILON for stage in STAGES))
    max_harm = max((value for value in group_harms.values() if value is not None), default=None)
    candidate = (pooled_gain["LOCAL"] >= 0.01
        and all(stage_gains[stage]["local_over_hinge"] >= 0.0 for stage in STAGES)
        and max_harm is not None and max_harm <= 0.25)
    exploratory = (pooled_gain["LOCAL"] >= 0.0
        and all(stage_gains[stage]["local_over_hinge"] >= -0.02 for stage in STAGES)
        and max_harm is not None and max_harm <= 0.25)
    return {"score_rows_pooled_unique": pooled_n, "window_gains": stage_gains,
        "pooled_mae": pooled_mae, "pooled_gain_over_hinge": pooled_gain,
        "supported_all_row_group_harm_vs_hinge": group_harms,
        "maximum_supported_all_row_group_harm_vs_hinge": max_harm,
        "targeted_region_gain_over_hinge": region_gains,
        "local_fit_deltas": fit_deltas,
        "mechanism_gate": {"pass": mechanism,
            "rule": "delta<0 in both fit populations; LOCAL gain >1e-12 over HINGE and GLOBAL_CONTROL in both score windows"},
        "candidate_gate": {"pass": candidate,
            "rule": "pooled gain>=0.01; each window gain>=0; max supported all-row group harm<=0.25"},
        "exploratory_review_gate": {"pass": exploratory,
            "rule": "pooled gain>=0; each window harm<=0.02; max supported all-row group harm<=0.25"}}


def score_predictions(layout: Layout | None = None) -> dict[str, Any]:
    layout = DEFAULT_LAYOUT if layout is None else layout
    freeze, freeze_sha = verify_freeze(layout)
    _, preflight_sha = verify_preflight(layout)
    _, pass_sha = _validate_root_pass(layout, freeze_sha, preflight_sha)
    _, pred_sha = _verify_prediction_freeze(layout, freeze_sha, pass_sha)
    ledger = _read_ledger(layout)
    if any([record["event"] for record in ledger.get(fit_id, [])]
           != ["RESERVE", "FIT_STARTED", "FIT_COMPLETE"] for fit_id in FIT_IDS):
        raise PilotError("Score stage requires four completed calibrator fit slots")
    fits = _load_completed_fits(layout)
    if layout.evaluation.exists():
        if not layout.verification.exists():
            raise PilotError("Evaluation exists without independent verification; refusing rescoring")
        evaluation = _read_json(layout.evaluation)
        signature = evaluation.pop("signature", None)
        if (signature != canonical_sha(evaluation)
                or evaluation.get("prefit_freeze_sha256") != freeze_sha
                or evaluation.get("identity_preflight_sha256") != preflight_sha
                or evaluation.get("root_source_pass_sha256") != pass_sha):
            raise PilotError("Existing evaluation is not signature-bound to current authorization")
        evaluation["signature"] = signature
        verification = _read_json(layout.verification)
        verification_signature = verification.pop("signature", None)
        if (verification_signature != canonical_sha(verification)
                or verification.get("status") != "PASS_INDEPENDENT_VERIFICATION"
                or verification.get("evaluation_sha256") != sha256_file(layout.evaluation)
                or verification.get("prefit_freeze_sha256") != freeze_sha
                or verification.get("identity_preflight_sha256") != preflight_sha
                or verification.get("root_source_pass_sha256") != pass_sha
                or verification.get("prediction_freeze_sha256") != pred_sha):
            raise PilotError("Existing independent verification does not bind the evaluation")
        return {"status": "EVALUATION_ALREADY_WRITTEN",
                "evaluation_sha256": sha256_file(layout.evaluation),
                "verification_sha256": sha256_file(layout.verification)}
    results: dict[str, dict[str, Any]] = {}
    resources = []
    with r4.SharedHeavyJobLock(layout.lock_path):
        verify_freeze(layout)
        verify_preflight(layout)
        _validate_root_pass(layout, freeze_sha, preflight_sha)
        _verify_prediction_freeze(layout, freeze_sha, pass_sha)
        resources.append(_resource_snapshot(layout, "saved_prediction_evaluation",
            "initial_locked_before_score_targets", initial=True))
        with threadpool_limits(limits=2):
            for stage_id in STAGES:
                prediction_path = layout.prediction_dir / f"{stage_id}_PREDICTIONS.npz"
                pred_arrays = _read_npz_fields(prediction_path, IDENTITY_FIELDS +
                    ("hinge_baseline", "local_prediction", "global_control_prediction"))
                results[stage_id] = _score_targets(layout, stage_id, pred_arrays)
                resources.append(_resource_snapshot(layout, "saved_prediction_evaluation",
                    f"{stage_id}_after_score", initial=False))
                del pred_arrays
        # Discard row arrays and targets from the durable receipt; only metrics and target hashes persist.
        score_summary = {}
        for stage_id, result in results.items():
            copy = {key: value for key, value in result.items()
                    if key not in {"predictions", "targets", "direct", "missing"}}
            score_summary[stage_id] = copy
        gate_record = _score_gates(score_summary, fits)
        body = {"schema_version": 1, "experiment_id": NAMESPACE.upper() + "_R1",
            "status": "PASS_FOUR_FIT_SCORE_DIAGNOSTIC_COMPLETE",
            "source_commit": freeze["source_commit"], "prefit_freeze_sha256": freeze_sha,
            "identity_preflight_sha256": preflight_sha,
            "root_source_pass_sha256": pass_sha,
            "prediction_freeze_sha256": pred_sha,
            "fit_state_hashes": {fit_id: sha256_file(_fit_state_path(layout, fit_id)) for fit_id in FIT_IDS},
            "score_populations": score_summary, "gates": gate_record,
            "resource_snapshots": resources, "calibrator_fit_starts": 4,
            "base_model_fits": 0, "preprocessing_fits": 0,
            "score_predictions_committed_before_score_target_load": True,
            "test_or_sample_accessed": False, "e79_accessed": False,
            "csv_created": False, "kaggle_upload": False}
        body["signature"] = canonical_sha(body)
        evaluation_sha = write_json_once(layout.evaluation, body)
    return {"status": body["status"], "evaluation_path": str(layout.evaluation.resolve()),
            "evaluation_sha256": evaluation_sha, "gates": gate_record}


def _paths_summary(layout: Layout) -> dict[str, str]:
    return {"root": str(layout.root), "report_dir": str(layout.report_dir),
        "experiment_dir": str(layout.experiment_dir), "prefit_freeze": str(layout.freeze),
        "identity_preflight": str(layout.preflight), "root_source_pass": str(layout.root_pass),
        "ledger": str(layout.ledger), "prediction_freeze": str(layout.prediction_freeze),
        "evaluation": str(layout.evaluation), "verification": str(layout.verification)}


def main(argv: Sequence[str] | None = None,
         layout: Layout | None = None) -> int:
    layout = DEFAULT_LAYOUT if layout is None else layout
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--paths", action="store_true",
                         help="show frozen default artifact paths; reads no inputs")
    actions.add_argument("--freeze", action="store_true",
                         help="freeze source and cached-input bytes before target access")
    actions.add_argument("--preflight", action="store_true",
                         help="run the non-target row identity preflight only")
    actions.add_argument("--fit", action="store_true",
                         help="execute the four authorized calibrator fits")
    actions.add_argument("--score", action="store_true",
                         help="score frozen predictions after the separate score boundary")
    actions.add_argument("--verify", action="store_true",
                         help="run the independent result verifier")
    args = parser.parse_args(argv)
    if args.paths:
        print(json.dumps(_paths_summary(layout), sort_keys=True))
        return 0
    try:
        if args.freeze:
            result = freeze_inputs(layout)
        elif args.preflight:
            result = non_target_preflight(layout)
        elif args.fit:
            result = run_fits(layout)
        elif args.score:
            result = score_predictions(layout)
        else:
            from scripts.verify_reg_center_local_correction_v1 import verify_results
            result = verify_results(layout)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (PilotError, r4.GateError) as exc:
        print(json.dumps({"status": "BLOCKED_OR_FAIL", "reason": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
