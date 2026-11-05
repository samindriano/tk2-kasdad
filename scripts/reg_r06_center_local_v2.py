"""Source-frozen, four-diagnostic-calibrator R06 center-local transfer.

The normal source-preparation flow is ``--freeze`` then ``--preflight``. Real
fits and score-target access require a separate signed ROOT_SOURCE_PASS_R2.json
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
from scripts import reg_center_local_correction_v1 as prior_center_local

ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = "reg_r06_center_local_v2"
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
        return self.experiment_dir / "PREFIT_FREEZE_R2.json"

    @property
    def preflight(self) -> Path:
        return self.report_dir / "IDENTITY_PREFLIGHT_R2.json"

    @property
    def root_pass(self) -> Path:
        return self.experiment_dir / "ROOT_SOURCE_PASS_R2.json"

    @property
    def ledger(self) -> Path:
        return self.experiment_dir / "calibrator_ledger_R2.jsonl"

    @property
    def fit_dir(self) -> Path:
        return self.experiment_dir / "fits_R2"

    @property
    def prediction_dir(self) -> Path:
        return self.experiment_dir / "score_predictions_R2"

    @property
    def prediction_freeze(self) -> Path:
        return self.experiment_dir / "PREDICTION_FREEZE_R2.json"

    @property
    def evaluation(self) -> Path:
        return self.report_dir / "EVALUATION_R2.json"

    @property
    def verification(self) -> Path:
        return self.report_dir / "INDEPENDENT_VERIFICATION_R2.json"


def make_layout(root: Path = ROOT, *, source_root: Path | None = None,
                lock_path: Path | None = None) -> Layout:
    root = Path(root).resolve()
    source_root = root if source_root is None else Path(source_root).resolve()
    report_dir = root / "reports" / "optimization_v3" / NAMESPACE
    experiment_dir = root / "experiments" / NAMESPACE
    source_paths = {
        "runner": source_root / "scripts" / "reg_r06_center_local_v2.py",
        "verifier": source_root / "scripts" / "verify_reg_r06_center_local_v2.py",
        "tests": source_root / "tests" / "test_reg_r06_center_local_v2.py",
        "policy": source_root / "reports" / "optimization_v3" / NAMESPACE / "POLICY_R2.md",
        "handoff": source_root / "reports" / "optimization_v3" / NAMESPACE / "SOL_HANDOFF_R2.md",
        "gitattributes": source_root / ".gitattributes",
        "prior_center_local_helpers": source_root / "scripts" / "reg_center_local_correction_v1.py",
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
        "r2_present_freeze": root / "experiments" / "reg_r06_present_native_hinge_v1" / "PREFIT_FREEZE_R2.json",
        "oof_label_identity_bundle": root / "experiments" / "runs" / "reg_monotone_hinge_p50_v1" / "FINAL_APPLICATION_R1" / "fit_inputs" / "HINGE_FINAL.npz",
        "r06_fit67_npz": Path(r"C:\Users\Sam\OneDrive\Documents\KASDAD\TP 2\experiments\runs\optimization_v3_batch1\R06\R06_fit67.npz"),
        "r06_fit67_sidecar": Path(r"C:\Users\Sam\OneDrive\Documents\KASDAD\TP 2\experiments\runs\optimization_v3_batch1\R06\R06_fit67.json"),
        "r06_fit75_npz": Path(r"C:\Users\Sam\OneDrive\Documents\KASDAD\TP 2\experiments\runs\optimization_v3_expanded_benchmark_v1_recovery_v2\R06\R06_fit75.npz"),
        "r06_fit75_sidecar": Path(r"C:\Users\Sam\OneDrive\Documents\KASDAD\TP 2\experiments\runs\optimization_v3_expanded_benchmark_v1_recovery_v2\R06\R06_fit75.json"),
        "d1_historical_score_rows": root / "experiments" / "runs" / "reg_monotone_hinge_p50_v1" / "R4" / "score_rows" / "HINGE_D1.npz",
        "d2_historical_score_rows": root / "experiments" / "runs" / "reg_monotone_hinge_p50_v1" / "R4" / "score_rows" / "HINGE_D2.npz",
        "prior_complex_evaluation": root / "reports" / "optimization_v3" / "reg_center_local_correction_v1" / "EVALUATION_R1.json",
        "prior_complex_verification": root / "reports" / "optimization_v3" / "reg_center_local_correction_v1" / "INDEPENDENT_VERIFICATION_R1.json",
        "prior_complex_results": root / "reports" / "optimization_v3" / "reg_center_local_correction_v1" / "RESULTS_R1.md",
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
    body = {"schema_version": 1, "experiment_id": NAMESPACE.upper() + "_R2",
        "status": "FROZEN_BEFORE_CANDIDATE_TARGET_ACCESS",
        "source_commit": _git_head(layout), "source_and_input_files": files,
        "r4_predecessor_receipts": r4_meta,
        "novelty_scan": {"document_count": 6,
            "documents": {key: {"path": str(path.resolve()), "sha256": NOVELTY_HASHES[key]}
                          for key, path in _expected_novelty_paths(layout).items()},
            "exact_candidate_duplicate_found": False,
            "repository_wide_novelty": "UNKNOWN"},
        "policy": {"fit_ids": list(FIT_IDS), "fit_cap": 5,
            "diagnostic_fit_cap": 4, "conditional_final_fit_cap": 1,
            "conditional_final_authorized": False,
            "base_model_fits": 0, "preprocessing_fits": 0,
            "local_center": LOCAL_CENTER, "local_half_width": LOCAL_HALF_WIDTH,
            "delta_bounds": list(DELTA_BOUNDS), "objective": "unweighted_sum_absolute_without_clipping",
            "local_optimizer": "exact_lower_weighted_median_with_weight_b_and_exact_binary_weight_ties",
            "global_control_optimizer": "lower_empirical_median_of_present_y_minus_R4_same_cutoff_hinge_on_R06",
            "base_model": "R06 SIMPLE raw prediction plus immutable same-cutoff R4 p50 hinge parameters",
            "fit_present_rows": PRESENT_FIT_COUNTS, "score_all_rows": SCORE_COUNTS,
            "conditional_final_present_rows": 19_528,
            "score_unique_record_ids": 15_953,
            "bins": {"raw_r06_edges": list(BIN_EDGES), "labels": list(BIN_LABELS), "min_support": MIN_GROUP_SUPPORT},
            "cycle_min_support": MIN_CYCLE_SUPPORT,
            "target_tail_groups": ["y<=5", "y>=95"],
            "missing_route": "unchanged_raw_R06_float64_bits_for_all_candidate_arms",
            "prediction_before_score": True,
            "original_full_direct_hinge_is_separate_diagnostic": True,
            "prior_complex_center_local_result_is_historical_context_only": True,
            "counterfactual_policies": False,
            "adaptive_post_discovery": False,
            "mechanism_gate": "LOCAL delta<0 and gain >1e-12 over transported R4-on-R06 and GLOBAL_CONTROL in both score windows",
            "candidate_gate": "pooled all-row gain vs transported R4-on-R06 >=0.01; both window gains >=0; max supported all-row subgroup harm <0.25",
            "exploratory_gate": "pooled gain >=0; each window harm vs transported R4-on-R06 <=0.02; max supported all-row subgroup harm <0.25",
            "cycle_74_harm_mandatory": True},
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
SCORE_NON_TARGET_FIELDS = IDENTITY_FIELDS + ("original_direct_predictions",)


def _stage_files(layout: Layout, stage_id: str) -> tuple[Path, Path, Path]:
    slug = "d1" if stage_id == "HINGE_D1" else "d2"
    return (layout.experiment_dir / "fit_identity_R2" / f"{stage_id}.npz",
            layout.experiment_dir / "score_identity_R2" / f"{stage_id}.npz",
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



# Solver, fixed-basis, clipping, and application primitives reuse the reviewed
# complex-route implementation verbatim. Its source hash is frozen separately;
# this transfer adds no new median, triangle, bounds, or application logic.
local_basis = prior_center_local.local_basis
base_hinge_unclipped = prior_center_local.base_hinge_unclipped
_clip_interval = prior_center_local._clip_interval
lower_weighted_median_interval = prior_center_local.lower_weighted_median_interval
lower_empirical_median_interval = prior_center_local.lower_empirical_median_interval
fit_local_delta = prior_center_local.fit_local_delta
fit_global_control_delta = prior_center_local.fit_global_control_delta
monotonicity_record = prior_center_local.monotonicity_record
apply_correction = prior_center_local.apply_correction


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


def _load_r06_identity(layout: Layout) -> dict[str, np.ndarray]:
    label_path = layout.input_paths["oof_label_identity_bundle"]
    labels = _read_npz_fields(label_path, IDENTITY_FIELDS,
                              allow_unloaded=("targets",))
    ids = np.asarray(labels["record_ids"]).astype(str)
    cycles = np.asarray(labels["cycles"], dtype=np.int64)
    physical_rows = np.asarray(labels["physical_rows"], dtype=np.int64)
    origins = np.asarray(labels["prediction_origins"], dtype=np.int64)
    direct = np.asarray(labels["direct_predictions"], dtype="<f8")
    missing = np.asarray(labels["missing_mask"], dtype=bool)
    n = len(ids)
    if (n != 20_903 or len(set(ids.tolist())) != n
            or len(set(physical_rows.tolist())) != n
            or set(np.unique(cycles).tolist()) != set(range(68, 80))):
        raise PilotError("Frozen OOF record, physical-row, or cycle identity differs")
    index = {record_id: i for i, record_id in enumerate(ids.tolist())}
    raw_r06 = np.full(n, np.nan, dtype="<f8")
    raw_origins = np.full(n, -1, dtype=np.int16)
    for origin, member, allowed in (
            (67, "audit__cycle_code", np.arange(68, 76)),
            (75, "score_cycles", np.arange(76, 80))):
        path = layout.input_paths[f"r06_fit{origin}_npz"]
        cache_unloaded = ("baseline_prediction", "raw_candidate", "control__ridge25_prediction",
            "control__baseline_sparse_raw", "control__cycle_code", "audit__baseline_prediction",
            "audit__candidate_rank50") if origin == 67 else (
            "raw_candidate", "baseline_prediction", "audit__candidate_rank50",
            "control__rows", "control__record_ids", "control__cycle_codes",
            "control__baseline_prediction", "control__baseline_lgb_svd32",
            "control__baseline_hybrid_ridge", "control__baseline_sparse_raw",
            "control__baseline_sparse_rank50", "control__control_rows", "control__cycle_code")
        arrays = _read_npz_fields(path, ("score_rows", "score_record_ids", "prediction", member),
                                  allow_unloaded=cache_unloaded)
        cache_ids = np.asarray(arrays["score_record_ids"]).astype(str)
        cache_cycles = np.asarray(arrays[member], dtype=np.int64)
        cache_rows = np.asarray(arrays["score_rows"], dtype=np.int64)
        mask = np.isin(cache_cycles, allowed)
        actual = {(str(cache_ids[i]), int(cache_cycles[i]), int(cache_rows[i]))
                  for i in np.flatnonzero(mask)}
        expected = {(str(ids[i]), int(cycles[i]), int(physical_rows[i]))
                    for i in range(n) if int(cycles[i]) in set(allowed.tolist())}
        if actual != expected:
            raise PilotError(f"R06 fit{origin} cache identity differs from OOF rows")
        values = np.asarray(arrays["prediction"], dtype="<f8")
        if not np.isfinite(values[mask]).all() or np.any((values[mask] < 0.0) | (values[mask] > 100.0)):
            raise PilotError(f"R06 fit{origin} predictions are invalid")
        for cache_i in np.flatnonzero(mask):
            row_i = index[str(cache_ids[cache_i])]
            if raw_origins[row_i] != -1:
                raise PilotError("Duplicate R06 prediction assignment")
            raw_r06[row_i] = values[cache_i]
            raw_origins[row_i] = origin
    if (not np.isfinite(raw_r06).all() or not np.array_equal(raw_origins, origins)
            or not np.array_equal(direct[missing].view("<u8"), raw_r06[missing].view("<u8"))):
        raise PilotError("R06 cache origin or missing raw fallback differs from identity")
    return {"physical_rows": physical_rows, "record_ids": ids,
            "prediction_origins": origins.astype(np.int8),
            "original_bundle_indices": np.asarray(labels["original_bundle_indices"], dtype=np.int64),
            "cycles": cycles.astype(np.int16), "direct_predictions": raw_r06,
            "original_direct_predictions": direct, "missing_mask": missing,
            "prediction_origins_r06": raw_origins}


def _population_indices(identity: Mapping[str, np.ndarray], stage_id: str,
                        role: str) -> np.ndarray:
    cycles = np.asarray(identity["cycles"], dtype=np.int64)
    origins = np.asarray(identity["prediction_origins"], dtype=np.int64)
    missing = np.asarray(identity["missing_mask"], dtype=bool)
    if (stage_id, role) == ("HINGE_D1", "fit"):
        selected = (cycles >= 68) & (cycles <= 71) & (origins == 67)
    elif (stage_id, role) == ("HINGE_D1", "score"):
        selected = (cycles >= 72) & (cycles <= 75) & (origins == 67)
    elif (stage_id, role) == ("HINGE_D2", "fit"):
        selected = (cycles >= 68) & (cycles <= 75) & (origins == 67)
    elif (stage_id, role) == ("HINGE_D2", "score"):
        selected = (cycles >= 76) & (cycles <= 79) & (origins == 75)
    else:
        raise PilotError(f"Unknown population route: {stage_id}/{role}")
    if role == "fit":
        selected &= ~missing
    return np.flatnonzero(selected)


def _r4_direct_replay(layout: Layout, identity: Mapping[str, np.ndarray],
                      stage_id: str, score_indices: np.ndarray) -> dict[str, Any]:
    slug = "d1" if stage_id == "HINGE_D1" else "d2"
    path = layout.input_paths[f"{slug}_historical_score_rows"]
    fields = ("physical_rows", "record_ids", "prediction_origins",
              "original_bundle_indices", "cycles", "direct_predictions",
              "missing_mask", "hinge_present_only")
    r4_score = _read_npz_fields(path, fields,
        allow_unloaded=("targets", "raw_direct", "causal_global_all_rows",
                        "causal_global_present_only"))
    params = _load_state(layout, stage_id)
    for name in ("physical_rows", "record_ids", "prediction_origins",
                 "original_bundle_indices", "cycles", "missing_mask"):
        expected = np.asarray(identity[name])[score_indices]
        actual = np.asarray(r4_score[name])
        if actual.shape != expected.shape or not np.array_equal(actual, expected):
            raise PilotError(f"R4 direct diagnostic row identity differs: {stage_id}/{name}")
    direct = np.asarray(identity["original_direct_predictions"])[score_indices].astype("<f8")
    if not np.array_equal(np.asarray(r4_score["direct_predictions"], dtype="<f8").view("<u8"),
                          direct.view("<u8")):
        raise PilotError(f"R4 diagnostic DIRECT values differ: {stage_id}")
    replay = r1.apply_hinge_present_only(
        direct, np.asarray(identity["missing_mask"])[score_indices], params).astype("<f8", copy=False)
    saved = np.asarray(r4_score["hinge_present_only"], dtype="<f8")
    if not np.array_equal(replay.view("<u8"), saved.view("<u8")):
        raise PilotError(f"R4 original full-DIRECT hinge replay differs: {stage_id}")
    return {"score_rows_sha256": sha256_file(path),
            "row_identity_sha256": _row_identity_sha(r4_score),
            "full_direct_hinge_replay": True,
            "hinge_state_sha256": sha256_file(layout.input_paths[f"{slug}_hinge_state"])}


def non_target_preflight(layout: Layout | None = None) -> dict[str, Any]:
    layout = DEFAULT_LAYOUT if layout is None else layout
    freeze, freeze_sha = verify_freeze(layout)
    resources: list[dict[str, Any]] = []
    population_rows: list[dict[str, Any]] = []
    population_ids: dict[str, Any] = {}
    score_ids: list[str] = []
    score_rows: list[int] = []
    r4_replays: dict[str, Any] = {}
    with r4.SharedHeavyJobLock(layout.lock_path):
        verify_freeze(layout)
        resources.append(_resource_snapshot(layout, "saved_prediction_evaluation",
            "initial_locked_before_non_target_identity", initial=True))
        identity = _load_r06_identity(layout)
        resources.append(_resource_snapshot(layout, "saved_prediction_evaluation",
            "after_oof_identity_and_r06_cache_load", initial=False))
        for stage_id in STAGES:
            fit_indices = _population_indices(identity, stage_id, "fit")
            score_indices = _population_indices(identity, stage_id, "score")
            cycles = np.asarray(identity["cycles"], dtype=np.int64)
            origins = np.asarray(identity["prediction_origins"], dtype=np.int64)
            missing = np.asarray(identity["missing_mask"], dtype=bool)
            fit_cycles = ((cycles >= 68) & (cycles <= 71) if stage_id == "HINGE_D1"
                          else (cycles >= 68) & (cycles <= 75)) & (origins == 67)
            fit_all_indices = np.flatnonzero(fit_cycles)
            excluded_indices = fit_all_indices[missing[fit_all_indices]]
            fit_identity = {name: np.asarray(values)[fit_all_indices]
                            for name, values in identity.items()
                            if name not in {"prediction_origins_r06", "original_direct_predictions"}}
            score_identity = {name: np.asarray(values)[score_indices]
                              for name, values in identity.items()
                              if name != "prediction_origins_r06"}
            score_identity["direct_predictions"] = np.asarray(identity["direct_predictions"])[score_indices].astype("<f8")
            fit_record = _validate_population(stage_id, "fit", fit_identity)
            score_record = _validate_population(stage_id, "score", score_identity)
            fit_record["population"] = "present_fit"
            score_record["population"] = "score"
            population_rows.extend((fit_record, score_record))
            fit_populations = {
                "present_rows": [
                    {"record_id": str(identity["record_ids"][i]),
                     "physical_row": int(identity["physical_rows"][i]),
                     "prediction_origin": int(identity["prediction_origins"][i]),
                     "original_bundle_index": int(identity["original_bundle_indices"][i]),
                     "cycle": int(identity["cycles"][i])}
                    for i in fit_indices],
                "excluded_missing_rows": [
                    {"record_id": str(identity["record_ids"][i]),
                     "physical_row": int(identity["physical_rows"][i]),
                     "prediction_origin": int(identity["prediction_origins"][i]),
                     "original_bundle_index": int(identity["original_bundle_indices"][i]),
                     "cycle": int(identity["cycles"][i])}
                    for i in excluded_indices],
            }
            population_ids[stage_id] = fit_populations
            population_ids[stage_id]["score_rows"] = [
                {"record_id": str(identity["record_ids"][i]),
                 "physical_row": int(identity["physical_rows"][i]),
                 "prediction_origin": int(identity["prediction_origins"][i]),
                 "original_bundle_index": int(identity["original_bundle_indices"][i]),
                 "cycle": int(identity["cycles"][i])}
                for i in score_indices]
            stage_key = "D1" if stage_id == "HINGE_D1" else "D2"
            for fit_id in (f"{stage_key}_LOCAL", f"{stage_key}_GLOBAL_CONTROL"):
                population_ids[fit_id] = {"shares_fit_population": stage_id}
            fit_path, score_path, _ = _stage_files(layout, stage_id)
            fit_digest = _verify_or_write_prediction_file(fit_path, fit_identity)
            score_digest = _verify_or_write_prediction_file(score_path, score_identity)
            fit_record["target_free_identity_artifact_sha256"] = fit_digest
            score_record["target_free_identity_artifact_sha256"] = score_digest
            params = _load_state(layout, stage_id)
            raw_r06 = np.asarray(score_identity["direct_predictions"], dtype="<f8")
            missing_score = np.asarray(score_identity["missing_mask"], dtype=bool)
            r06_baseline, _ = apply_correction(raw_r06, missing_score, params, 0.0, local=False)
            if not np.array_equal(r06_baseline[missing_score].view("<u8"), raw_r06[missing_score].view("<u8")):
                raise PilotError(f"R4-on-R06 missing fallback differs: {stage_id}")
            active = (~np.asarray(identity["missing_mask"], dtype=bool)[fit_indices]
                      & (local_basis(np.asarray(identity["direct_predictions"])[fit_indices]) > 0.0))
            if int(active.sum()) < MIN_LOCAL_SUPPORT:
                raise PilotError(f"R06 LOCAL fit support unsupported: {stage_id}/{int(active.sum())}")
            monotonicity = monotonicity_record(params, -2.0)
            if not monotonicity["monotone_non_decreasing"]:
                raise PilotError(f"R4 base cannot guarantee nondecreasing LOCAL bounds: {stage_id}")
            r4_replays[stage_id] = _r4_direct_replay(layout, identity, stage_id, score_indices)
            score_ids.extend(np.asarray(score_identity["record_ids"]).astype(str).tolist())
            score_rows.extend(np.asarray(score_identity["physical_rows"], dtype=np.int64).tolist())
            del fit_identity, score_identity
            resources.append(_resource_snapshot(layout, "saved_prediction_evaluation",
                f"{stage_id}_identity_and_comparator_replayed", initial=False))
        if (len(score_ids) != 15_953 or len(set(score_ids)) != 15_953
                or len(set(score_rows)) != 15_953):
            raise PilotError("Pooled D1/D2 score rows are not 15,953 unique IDs/physical rows")
        all_present = int(np.sum(~np.asarray(identity["missing_mask"], dtype=bool)
                                 & (np.asarray(identity["cycles"], dtype=np.int64) >= 68)
                                 & (np.asarray(identity["cycles"], dtype=np.int64) <= 79)))
        if all_present != 19_528:
            raise PilotError(f"Conditional full-history present count differs: {all_present}")
        resources.append(_resource_snapshot(layout, "saved_prediction_evaluation",
            "after_non_target_identity_replay", initial=False))
    ids_body = {"schema_version": 1, "experiment_id": NAMESPACE.upper() + "_R2",
        "source_commit": freeze["source_commit"], "input_freeze_sha256": freeze_sha,
        "populations": population_ids,
        "fit_present_record_id_hashes": {"HINGE_D1": r1.tagged_array_sha256(
            np.asarray([item["record_id"] for item in population_ids["HINGE_D1"]["present_rows"]], dtype=str)),
            "HINGE_D2": r1.tagged_array_sha256(
                np.asarray([item["record_id"] for item in population_ids["HINGE_D2"]["present_rows"]], dtype=str))},
        "score_window_record_id_hashes": {
            stage: r1.tagged_array_sha256(np.asarray(
                [str(item["record_id"]) for item in population_ids[stage]["score_rows"]], dtype=str))
            for stage in STAGES},
        "score_windows": {"HINGE_D1": SCORE_COUNTS["HINGE_D1"],
                          "HINGE_D2": SCORE_COUNTS["HINGE_D2"]},
        "fit_counts": PRESENT_FIT_COUNTS,
        "excluded_missing_counts": {"HINGE_D1": 342, "HINGE_D2": 849},
        "conditional_final_present_rows": all_present,
        "conditional_final_authorized": False,
        "calibrator_budget": {"cap": 5, "diagnostic_slots": 4,
                              "conditional_final_slots": 1},
        "prior_ledgers_reset": False}
    ids_body["signature"] = canonical_sha(ids_body)
    ids_sha = write_json_once(layout.experiment_dir / "FIT_POPULATION_IDS_R2.json", ids_body)
    body = {"schema_version": 1, "experiment_id": NAMESPACE.upper() + "_R2",
        "status": "PASS_NON_TARGET_IDENTITY_PREFLIGHT_PENDING_ROOT_SOURCE_PASS",
        "input_freeze_sha256": freeze_sha, "source_commit": freeze["source_commit"],
        "fit_population_ids_sha256": ids_sha,
        "populations": population_rows,
        "r4_original_direct_hinge_replays": r4_replays,
        "pooled_score_rows": 15_953, "pooled_score_record_ids_unique": True,
        "pooled_score_physical_rows_unique": True,
        "fit_population_ids_counts": PRESENT_FIT_COUNTS,
        "excluded_missing_counts": {"HINGE_D1": 342, "HINGE_D2": 849},
        "raw_r06_triangle_support_present_fit_rows": {
            stage: next(item["local_positive_basis_present_rows"] for item in population_rows
                        if item["fit_id"] == stage and item["population"] == "present_fit")
            for stage in STAGES},
        "saved_r4_hinge_on_r06_monotonicity_at_delta_minus_2": {
            stage: monotonicity_record(_load_state(layout, stage), -2.0) for stage in STAGES},
        "cycle_74_scored_by_d1": int(np.sum(
            (np.asarray(identity["cycles"]) == 74)
            & (np.asarray(identity["prediction_origins"]) == 67))),
        "resource_snapshots": resources,
        "target_arrays_present_in_npz_but_loaded": False,
        "candidate_target_values_loaded": False, "calibrator_fit_starts": 0,
        "new_ledger_created": False, "score_target_values_loaded": False,
        "test_or_sample_accessed": False, "e79_accessed": False,
        "csv_created": False, "kaggle_upload": False}
    body["signature"] = canonical_sha(body)
    digest = write_json_once(layout.preflight, body)
    return {"status": body["status"], "path": str(layout.preflight.resolve()),
            "sha256": digest, "fit_population_ids_sha256": ids_sha,
            "target_arrays_loaded": False}


def verify_preflight(layout: Layout | None = None) -> tuple[dict[str, Any], str]:
    layout = DEFAULT_LAYOUT if layout is None else layout
    freeze, freeze_sha = verify_freeze(layout)
    body = _read_json(layout.preflight)
    signature = body.pop("signature", None)
    if signature != canonical_sha(body):
        raise PilotError("Non-target preflight signature mismatch")
    body["signature"] = signature
    if (body.get("status") != "PASS_NON_TARGET_IDENTITY_PREFLIGHT_PENDING_ROOT_SOURCE_PASS"
            or body.get("input_freeze_sha256") != freeze_sha
            or body.get("source_commit") != freeze.get("source_commit")
            or body.get("candidate_target_values_loaded") is not False
            or body.get("calibrator_fit_starts") != 0
            or body.get("score_target_values_loaded") is not False):
        raise PilotError("Non-target preflight receipt boundary or freeze binding failed")
    id_path = layout.experiment_dir / "FIT_POPULATION_IDS_R2.json"
    id_body = _read_json(id_path)
    id_signature = id_body.pop("signature", None)
    if id_signature != canonical_sha(id_body):
        raise PilotError("Fit population ID receipt signature mismatch")
    id_body["signature"] = id_signature
    if (body.get("fit_population_ids_sha256") != sha256_file(id_path)
            or id_body.get("input_freeze_sha256") != freeze_sha
            or id_body.get("source_commit") != freeze.get("source_commit")
            or id_body.get("conditional_final_authorized") is not False
            or id_body.get("prior_ledgers_reset") is not False):
        raise PilotError("Fit population ID receipt is not bound to this source freeze")
    rows = {(row.get("fit_id"), row.get("population")): row
            for row in body.get("populations", [])}
    for stage_id in STAGES:
        fit_path, score_path, _ = _stage_files(layout, stage_id)
        fit_record = rows.get((stage_id, "present_fit"))
        score_record = rows.get((stage_id, "score"))
        if fit_record is None or score_record is None:
            raise PilotError(f"Non-target preflight population missing: {stage_id}")
        for path, record, role, fields in (
                (fit_path, fit_record, "fit", IDENTITY_FIELDS),
                (score_path, score_record, "score", SCORE_NON_TARGET_FIELDS)):
            if sha256_file(path) != record.get("target_free_identity_artifact_sha256"):
                raise PilotError(f"Target-free identity artifact changed: {path}")
            arrays = _read_npz_fields(path, fields, allow_unloaded=("targets",))
            checked = _validate_population(stage_id, role, arrays)
            if (checked["row_identity_hashes"] != record.get("row_identity_hashes")
                    or checked["n_all"] != record.get("n_all")
                    or checked["n_present"] != record.get("n_present")):
                raise PilotError(f"Target-free population identity replay differs: {stage_id}/{role}")
    return body, sha256_file(layout.preflight)


def _validate_root_pass(layout: Layout, freeze_sha: str,
                        preflight_sha: str) -> tuple[dict[str, Any], str]:
    receipt = _validate_signed(layout.root_pass, verdict="PASS")
    expected_fields = {"verdict", "experiment_id", "pass_kind", "source_commit",
        "prefit_freeze_sha256", "identity_preflight_sha256", "authorized_fit_ids",
        "calibrator_fit_cap", "authorized_calibrator_fits", "base_model_fits",
        "preprocessing_fits", "official_lp_starts", "conditional_final_fit_authorized",
        "prior_three_slot_budget_reset",
        "score_target_access_authorized", "test_or_sample_authorized",
        "kaggle_upload_authorized", "signature"}
    if set(receipt) != expected_fields:
        raise PilotError("ROOT_SOURCE_PASS_R2 fields differ from the exact callback schema")
    if (receipt.get("experiment_id") != NAMESPACE.upper() + "_R2"
            or receipt.get("pass_kind") != "CENTER_LOCAL_DIAGNOSTIC"
            or receipt.get("source_commit") != _git_head(layout)
            or receipt.get("prefit_freeze_sha256") != freeze_sha
            or receipt.get("identity_preflight_sha256") != preflight_sha
            or receipt.get("authorized_fit_ids") != list(FIT_IDS)
            or receipt.get("calibrator_fit_cap") != 5
            or receipt.get("authorized_calibrator_fits") != 4
            or receipt.get("base_model_fits") != 0
            or receipt.get("preprocessing_fits") != 0
            or receipt.get("official_lp_starts") != 0
            or receipt.get("conditional_final_fit_authorized") is not False
            or receipt.get("prior_three_slot_budget_reset") is not False
            or receipt.get("score_target_access_authorized") is not True
            or receipt.get("test_or_sample_authorized") is not False
            or receipt.get("kaggle_upload_authorized") is not False):
        raise PilotError("ROOT_SOURCE_PASS_R2 is not the exact four-diagnostic-fit authorization under the five-slot cap")
    return receipt, sha256_file(layout.root_pass)


def _row_identity_sha(arrays: Mapping[str, np.ndarray]) -> str:
    return canonical_sha(_row_hashes(arrays))


def _fit_state_path(layout: Layout, fit_id: str) -> Path:
    return layout.fit_dir / f"{fit_id}.json"


def _load_targets_for_record_ids(layout: Layout,
                                 record_ids: Sequence[str]) -> np.ndarray:
    """Load target values only after the caller has crossed its authorization gate."""
    target_path = layout.input_paths["oof_label_identity_bundle"]
    fields = ("record_ids", "targets")
    unloaded_identity = tuple(name for name in IDENTITY_FIELDS if name != "record_ids")
    arrays = _read_npz_fields(target_path, fields, allow_unloaded=unloaded_identity)
    source_ids = np.asarray(arrays["record_ids"]).astype(str)
    targets = np.asarray(arrays["targets"], dtype="<f8")
    if (targets.ndim != 1 or targets.shape != source_ids.shape
            or len(source_ids) != 20_903 or len(np.unique(source_ids)) != len(source_ids)
            or not np.isfinite(targets).all() or np.any((targets < 0.0) | (targets > 100.0))):
        raise PilotError("Authorized OOF target member is invalid or misaligned")
    source_index = {record_id: index for index, record_id in enumerate(source_ids.tolist())}
    requested = [str(record_id) for record_id in record_ids]
    if any(record_id not in source_index for record_id in requested):
        raise PilotError("Authorized target lookup contains an unknown record ID")
    return targets[np.asarray([source_index[record_id] for record_id in requested], dtype=np.int64)]


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
                f"{NAMESPACE}:R2:{fit_id}:{records[0]['prefit_freeze_sha256']}:{records[0]['root_source_pass_sha256']}".encode("utf-8")).hexdigest()
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
        raise PilotError("Diagnostic ledger exceeds its four-fit allowance or is not a contiguous fit prefix")
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
    targets_all = _load_targets_for_record_ids(layout, identity_arrays["record_ids"])
    if _row_identity_sha(identity_arrays) != verified_row_sha:
        raise PilotError(f"Fit row identity changed before target load: {fit_id}")
    present = ~np.asarray(identity_arrays["missing_mask"], dtype=bool)
    targets = targets_all[present]
    direct = np.asarray(identity_arrays["direct_predictions"], dtype="<f8")[present]
    params = _load_state(layout, stage_id)
    if fit_id.endswith("_LOCAL"):
        fit = fit_local_delta(direct, targets, np.ones(len(targets), dtype=bool), params)
        local = True
    else:
        fit = fit_global_control_delta(direct, targets, np.ones(len(targets), dtype=bool), params)
        local = False
    monotonicity = monotonicity_record(params, fit["delta"])
    if local and not monotonicity["monotone_non_decreasing"]:
        raise PilotError(f"LOCAL monotonicity failed for {fit_id}; bounds remain frozen")
    output = {"schema_version": 1, "experiment_id": NAMESPACE.upper() + "_R2",
        "status": "FIT_COMPLETE", "fit_id": fit_id, "stage_id": stage_id,
        "arm": "LOCAL" if local else "GLOBAL_CONTROL",
        "recipe": {"base": "saved fixed-p50 HINGE", "center": LOCAL_CENTER,
            "half_width": LOCAL_HALF_WIDTH if local else None,
            "delta_bounds": list(DELTA_BOUNDS), "clipping_in_objective": False,
            "fit_rows": "present only", "objective": "unweighted sum absolute"},
        "fit": fit, "monotonicity": monotonicity,
        "fit_population": {**identity, "fit_target_rows": len(targets),
                            "target_values_loaded_after_root_pass": True},
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
    original_direct = np.asarray(arrays["original_direct_predictions"], dtype="<f8")
    missing = np.asarray(arrays["missing_mask"], dtype=bool)
    params = _load_state(layout, stage_id)
    transported, _ = apply_correction(direct, missing, params, 0.0, local=False)
    original_hinge = r1.apply_hinge_present_only(original_direct, missing, params).astype("<f8", copy=False)
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
        "direct_predictions": direct.copy(),
        "original_direct_predictions": original_direct.copy(),
        "missing_mask": missing.copy(),
        "transported_r4_on_r06": transported,
        "local_prediction": local, "global_control_prediction": control,
        "original_full_direct_hinge_diagnostic": original_hinge}


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
            ("original_direct_predictions", "transported_r4_on_r06",
             "local_prediction", "global_control_prediction",
             "original_full_direct_hinge_diagnostic"))
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
                f"{NAMESPACE}:R2:{fit_id}:{freeze_sha}:{pass_sha}".encode("utf-8")).hexdigest()
            common = {"schema_version": 1, "experiment_id": NAMESPACE.upper() + "_R2",
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
        body = {"schema_version": 1, "experiment_id": NAMESPACE.upper() + "_R2",
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
    result["gain_over_transported"] = {"all_rows": {}, "present_only": {}}
    for scope in ("all_rows", "present_only"):
        baseline = result["metrics"][scope]["TRANSPORTED_R4_ON_R06"]
        for name in ("LOCAL", "GLOBAL_CONTROL"):
            candidate = result["metrics"][scope][name]
            if baseline["status"] == "SUPPORTED" and candidate["status"] == "SUPPORTED":
                result["gain_over_transported"][scope][name] = baseline["mae"] - candidate["mae"]
            else:
                result["gain_over_transported"][scope][name] = None
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
        "local_gain_over_transported": (
            metrics["TRANSPORTED_R4_ON_R06"]["mae"] - metrics["LOCAL"]["mae"]
            if metrics["TRANSPORTED_R4_ON_R06"]["status"] == "SUPPORTED"
            and metrics["LOCAL"]["status"] == "SUPPORTED" else None)}


def _score_targets(layout: Layout, stage_id: str,
                   frozen_predictions: Mapping[str, np.ndarray]) -> dict[str, Any]:
    _, score_path, _ = _stage_files(layout, stage_id)
    arrays = _read_npz_fields(score_path, SCORE_NON_TARGET_FIELDS,
        allow_unloaded=("targets",))
    identity = _validate_population(stage_id, "score", arrays)
    pred_ids = np.asarray(frozen_predictions["record_ids"]).astype(str)
    if not np.array_equal(pred_ids, np.asarray(arrays["record_ids"]).astype(str)):
        raise PilotError(f"Frozen prediction row order differs from score targets: {stage_id}")
    for name in ("physical_rows", "prediction_origins", "original_bundle_indices", "cycles",
                 "direct_predictions", "original_direct_predictions", "missing_mask"):
        left, right = np.asarray(frozen_predictions[name]), np.asarray(arrays[name])
        if name in {"direct_predictions", "original_direct_predictions"}:
            equal = np.array_equal(left.astype("<f8", copy=False).view("<u8"),
                                   right.astype("<f8", copy=False).view("<u8"))
        else:
            equal = np.array_equal(left, right)
        if not equal:
            raise PilotError(f"Frozen prediction row identity changed: {stage_id}/{name}")
    target = _load_targets_for_record_ids(layout, pred_ids)
    if not np.isfinite(target).all() or np.any((target < 0.0) | (target > 100.0)):
        raise PilotError(f"Score targets are nonfinite or outside [0,100]: {stage_id}")
    predictions = {
        "TRANSPORTED_R4_ON_R06": np.asarray(frozen_predictions["transported_r4_on_r06"], dtype="<f8"),
        "LOCAL": np.asarray(frozen_predictions["local_prediction"], dtype="<f8"),
        "GLOBAL_CONTROL": np.asarray(frozen_predictions["global_control_prediction"], dtype="<f8"),
        "ORIGINAL_FULL_DIRECT_HINGE": np.asarray(
            frozen_predictions["original_full_direct_hinge_diagnostic"], dtype="<f8"),
        "RAW_R06": np.asarray(frozen_predictions["direct_predictions"], dtype="<f8")}
    if any(not np.isfinite(value).all() or np.any((value < 0.0) | (value > 100.0))
           for value in predictions.values()):
        raise PilotError(f"Score predictions are nonfinite or outside [0,100]: {stage_id}")
    missing = np.asarray(arrays["missing_mask"], dtype=bool)
    direct = np.asarray(arrays["direct_predictions"], dtype="<f8")
    for name in ("TRANSPORTED_R4_ON_R06", "LOCAL", "GLOBAL_CONTROL", "RAW_R06"):
        value = predictions[name]
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
        hinge = metric["TRANSPORTED_R4_ON_R06"]
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
        overall = result["metrics"]["overall"]["gain_over_transported"]["all_rows"]
        local_mae = result["metrics"]["overall"]["metrics"]["all_rows"]["LOCAL"]["mae"]
        hinge_mae = result["metrics"]["overall"]["metrics"]["all_rows"]["TRANSPORTED_R4_ON_R06"]["mae"]
        control_mae = result["metrics"]["overall"]["metrics"]["all_rows"]["GLOBAL_CONTROL"]["mae"]
        stage_gains[stage_id] = {"local_over_transported": overall["LOCAL"],
            "local_over_global_control": control_mae - local_mae,
            "global_control_over_transported": hinge_mae - control_mae}
        group_harms[stage_id] = _supported_all_row_group_harm(result)
        region_gains[stage_id] = result["targeted_region"]["local_gain_over_transported"]
    score_counts = {stage: SCORE_COUNTS[stage] for stage in STAGES}
    pooled_n = sum(score_counts.values())
    if pooled_n != 15_953:
        raise PilotError("Pooled score denominator differs from the frozen row count")
    pooled_mae = {}
    for policy in ("TRANSPORTED_R4_ON_R06", "LOCAL", "GLOBAL_CONTROL"):
        pooled_mae[policy] = sum(results[stage]["metrics"]["overall"]["metrics"]["all_rows"][policy]["mae"]
            * SCORE_COUNTS[stage] for stage in STAGES) / pooled_n
    pooled_gain = {name: pooled_mae["TRANSPORTED_R4_ON_R06"] - pooled_mae[name]
                   for name in ("LOCAL", "GLOBAL_CONTROL")}
    local_states = {stage: ("D1_LOCAL" if stage == "HINGE_D1" else "D2_LOCAL") for stage in STAGES}
    fit_deltas = {stage: float(fits[fit_id]["fit"]["delta"])
                  for stage, fit_id in local_states.items()}
    mechanism = (all(fit_deltas[stage] < 0.0 for stage in STAGES)
        and all(stage_gains[stage]["local_over_transported"] > FIT_EPSILON
                and stage_gains[stage]["local_over_global_control"] > FIT_EPSILON for stage in STAGES))
    max_harm = max((value for value in group_harms.values() if value is not None), default=None)
    cycle74_rows = [row for row in results["HINGE_D1"]["metrics"]["by_cycle"]
                    if row["cycle"] == 74]
    if len(cycle74_rows) != 1:
        raise PilotError("Mandatory cycle 74 score group is missing")
    cycle74_metric = cycle74_rows[0]["metrics"]["all_rows"]
    if (cycle74_metric["TRANSPORTED_R4_ON_R06"]["status"] != "SUPPORTED"
            or cycle74_metric["LOCAL"]["status"] != "SUPPORTED"):
        raise PilotError("Mandatory cycle 74 harm is unsupported")
    cycle74_harm = (cycle74_metric["LOCAL"]["mae"]
                    - cycle74_metric["TRANSPORTED_R4_ON_R06"]["mae"])
    window_harms = {stage: -stage_gains[stage]["local_over_transported"] for stage in STAGES}
    candidate = (pooled_gain["LOCAL"] >= 0.01
        and all(stage_gains[stage]["local_over_transported"] >= 0.0 for stage in STAGES)
        and max_harm is not None and max_harm < 0.25 and cycle74_harm < 0.25)
    exploratory = (pooled_gain["LOCAL"] >= 0.0
        and all(window_harms[stage] <= 0.02 for stage in STAGES)
        and max_harm is not None and max_harm < 0.25 and cycle74_harm < 0.25)
    return {"score_rows_pooled_unique": pooled_n, "window_gains": stage_gains,
        "pooled_mae": pooled_mae, "pooled_gain_over_transported": pooled_gain,
        "window_harm_vs_transported": window_harms,
        "supported_all_row_group_harm_vs_transported": group_harms,
        "maximum_supported_all_row_group_harm_vs_transported": max_harm,
        "cycle_74_harm_vs_transported": cycle74_harm,
        "targeted_region_gain_over_transported": region_gains,
        "local_fit_deltas": fit_deltas,
        "mechanism_gate": {"pass": mechanism,
            "rule": "delta<0 in both fit populations; LOCAL gain >1e-12 over transported R4-on-R06 and GLOBAL_CONTROL in both score windows"},
        "candidate_gate": {"pass": candidate,
            "rule": "pooled gain>=0.01; each window gain>=0; max supported all-row group harm<0.25"},
        "exploratory_review_gate": {"pass": exploratory,
            "rule": "pooled gain>=0; each window harm<=0.02; max supported all-row group harm<0.25"}}


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
                    ("original_direct_predictions", "transported_r4_on_r06",
                     "local_prediction", "global_control_prediction",
                     "original_full_direct_hinge_diagnostic"))
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
        body = {"schema_version": 1, "experiment_id": NAMESPACE.upper() + "_R2",
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
                         help="run the independent source-ready verifier")
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
            from scripts.verify_reg_r06_center_local_v2 import verify_source_ready
            result = verify_source_ready(layout)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (PilotError, r4.GateError, prior_center_local.PilotError) as exc:
        print(json.dumps({"status": "BLOCKED_OR_FAIL", "reason": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
