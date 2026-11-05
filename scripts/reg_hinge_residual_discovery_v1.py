"""Frozen, zero-fit residual-shape audit for cached R4 D1/D2 bundles."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import psutil
from threadpoolctl import threadpool_limits

from scripts import reg_monotone_hinge_p50_v4 as r4

ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = "reg_hinge_residual_discovery_v1"
REPORT_DIR = ROOT / "reports" / "optimization_v3" / NAMESPACE
EXPERIMENT_DIR = ROOT / "experiments" / NAMESPACE
POLICY = REPORT_DIR / "POLICY_R1.md"
FREEZE = EXPERIMENT_DIR / "INPUT_FREEZE_R1.json"
SUMMARY = REPORT_DIR / "MACHINE_SUMMARY_R1.json"
PLOT_BINS = REPORT_DIR / "FIGURE_RAW_BIN_RESIDUALS_R1.png"
PLOT_CYCLES = REPORT_DIR / "FIGURE_CYCLE_GAINS_R1.png"
VERIFICATION = REPORT_DIR / "INDEPENDENT_VERIFICATION_R1.json"
MEMO = REPORT_DIR / "MEMO_R1.md"
HANDOFF = REPORT_DIR / "SOL_HANDOFF.md"
GITATTRIBUTES = ROOT / ".gitattributes"
DIAGNOSTIC_DIR = ROOT / "reports" / "optimization_v3" / "reg_monotone_hinge_p50_v1"
R4_EXPERIMENT = ROOT / "experiments" / "reg_monotone_hinge_p50_v1"
R4_RUN = ROOT / "experiments" / "runs" / "reg_monotone_hinge_p50_v1" / "R4"
GLOBAL_FIT_DIR = ROOT / "experiments" / "runs" / "REG_FINAL_WAVE_FOLLOWUP_V2" / "fits"
RESOURCE_POLICY = ROOT / "reports" / "optimization_v3" / "RESOURCE_POLICY_USER_REVISED_20261105.json"

BIN_EDGES = (0.0, 20.0, 40.0, 60.0, 80.0, 100.0)
BIN_LABELS = ("[0,20)", "[20,40)", "[40,60)", "[60,80)", "[80,100]")
MIN_BIN_PRESENT = 200
MIN_CYCLE_PRESENT = 50
TARGET_TAIL_LOW = 5.0
TARGET_TAIL_HIGH = 95.0
FIT_EXPECTED = {"HINGE_D1": 4_950, "HINGE_D2": 12_412}
SCORE_EXPECTED = {"HINGE_D1": 7_462, "HINGE_D2": 8_491}
EXPECTED_CYCLES = {
    ("HINGE_D1", "fit"): tuple(range(68, 72)),
    ("HINGE_D1", "score"): tuple(range(72, 76)),
    ("HINGE_D2", "fit"): tuple(range(68, 76)),
    ("HINGE_D2", "score"): tuple(range(76, 80)),
}
EXPECTED_ORIGIN = {("HINGE_D1", "fit"): 67, ("HINGE_D1", "score"): 67,
                   ("HINGE_D2", "fit"): 67, ("HINGE_D2", "score"): 75}

SOURCE_FILES: dict[str, Path] = {
    "runner": Path(__file__).resolve(),
    "verifier": (ROOT / "scripts" / "verify_reg_hinge_residual_discovery_v1.py").resolve(),
    "policy": POLICY,
    "handoff": HANDOFF,
    "gitattributes": GITATTRIBUTES,
}
REFERENCE_SOURCES: dict[str, Path] = {
    "r4_runner": ROOT / "scripts" / "reg_monotone_hinge_p50_v4.py",
    "r4_verifier": ROOT / "scripts" / "verify_reg_monotone_hinge_p50_v4.py",
    "hinge_core": ROOT / "scripts" / "reg_monotone_hinge_p50_v1.py",
    "hinge_verifier": ROOT / "scripts" / "verify_reg_monotone_hinge_p50_v1.py",
    "direct_helpers": ROOT / "scripts" / "reg_final_wave_v2.py",
    "r4_protocol": DIAGNOSTIC_DIR / "PROTOCOL_R4.md",
    "r4_decision": DIAGNOSTIC_DIR / "SOL_DIAGNOSTIC_DECISION_R4.md",
    "novelty_research_plan": ROOT / "reports" / "optimization_v3" / "research_backlog" / "RESEARCH_PLAN.md",
    "novelty_research_lanes": ROOT / "reports" / "optimization_v3" / "research_next_v1" / "RESEARCH_LANES.md",
    "novelty_residual_results": ROOT / "reports" / "optimization_v3" / "rnbr_pilot_v1" / "official_execution_v1" / "RESULTS.md",
    "novelty_campaign_summary": ROOT / "reports" / "optimization_v3" / "research_next_v1" / "CAMPAIGN_SUMMARY.md",
}
INPUT_FILES: dict[str, Path] = {
    "r4_diagnostics": DIAGNOSTIC_DIR / "DIAGNOSTICS_R4.json",
    "r4_independent_verification": DIAGNOSTIC_DIR / "DIAGNOSTICS_VERIFICATION_R4.json",
    "r4_freeze": R4_EXPERIMENT / "IDENTITY_FREEZE_R4.json",
    "r4_root_source_pass": R4_EXPERIMENT / "ROOT_SOURCE_PASS_R4.json",
    "resource_policy": RESOURCE_POLICY,
    "d1_fit_state": R4_RUN / "fits" / "HINGE_D1.json",
    "d1_fit_input": R4_RUN / "fit_inputs" / "HINGE_D1.npz",
    "d1_score_rows": R4_RUN / "score_rows" / "HINGE_D1.npz",
    "d1_global_state": GLOBAL_FIT_DIR / "D1_GLOBAL.json",
    "d2_fit_state": R4_RUN / "fits" / "HINGE_D2.json",
    "d2_fit_input": R4_RUN / "fit_inputs" / "HINGE_D2.npz",
    "d2_score_rows": R4_RUN / "score_rows" / "HINGE_D2.npz",
    "d2_global_state": GLOBAL_FIT_DIR / "D2_GLOBAL.json",
}

FIT_FIELDS = ("physical_rows", "record_ids", "prediction_origins", "original_bundle_indices",
              "cycles", "direct_predictions", "missing_mask", "targets")
SCORE_FIELDS = FIT_FIELDS + ("raw_direct", "hinge_present_only",
    "causal_global_present_only")
SCORED_POLICIES = ("raw_direct", "causal_global_present_only", "hinge_present_only")
PRESENT_POLICIES = ("raw_direct", "causal_global_present_only", "hinge_present_only")


class DiagnosticError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(value), sort_keys=True, ensure_ascii=False,
                         allow_nan=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json_once(path: Path, body: Mapping[str, Any]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(body), sort_keys=True, ensure_ascii=False,
                         allow_nan=False, indent=2).encode("utf-8") + b"\n"
    try:
        with path.open("xb") as stream:
            stream.write(payload)
    except FileExistsError as exc:
        raise DiagnosticError(f"Refusing to overwrite immutable artifact: {path}") from exc
    return hashlib.sha256(payload).hexdigest()


def _git_head() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise DiagnosticError("Cannot resolve the source commit")
    return result.stdout.strip()


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _validate_signed_json(path: Path, *, allowed_status: tuple[str, ...] = ()) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    signature = value.get("signature")
    body = dict(value)
    body.pop("signature", None)
    if not isinstance(signature, str) or canonical_sha(body) != signature:
        raise DiagnosticError(f"Canonical signature mismatch: {path}")
    if allowed_status and value.get("status") not in allowed_status:
        raise DiagnosticError(f"Unexpected status in {path}: {value.get('status')}")
    return value


def _diagnostic_metadata_check() -> None:
    diagnostics = json.loads(INPUT_FILES["r4_diagnostics"].read_text(encoding="utf-8"))
    verification = json.loads(INPUT_FILES["r4_independent_verification"].read_text(encoding="utf-8"))
    if (diagnostics.get("experiment_id") != "REG_MONOTONE_HINGE_P50_V1_R4"
            or diagnostics.get("status") != "TWO_DIAGNOSTICS_COMPLETE_PENDING_ROOT_REVIEW"
            or diagnostics.get("test_or_sample_accessed") is not False):
        raise DiagnosticError("R4 diagnostics metadata or access boundary is unexpected")
    if (verification.get("experiment_id") != diagnostics.get("experiment_id")
            or verification.get("status") != "PASS_INDEPENDENT_DIAGNOSTICS_REPLAY"
            or verification.get("test_or_sample_accessed") is not False
            or verification.get("diagnostics_sha256") != sha256_file(INPUT_FILES["r4_diagnostics"])):
        raise DiagnosticError("R4 independent verification does not bind the expected diagnostics receipt")
    r4_freeze = json.loads(INPUT_FILES["r4_freeze"].read_text(encoding="utf-8"))
    r4_pass = _validate_signed_json(INPUT_FILES["r4_root_source_pass"])
    if set(record.get("fit_id") for record in diagnostics.get("stages", [])) != {"HINGE_D1", "HINGE_D2"}:
        raise DiagnosticError("R4 diagnostics do not contain exactly HINGE_D1/HINGE_D2")
    for record in diagnostics["stages"]:
        fit_id = record["fit_id"]
        key = fit_id[-2:].lower()
        pairs = (("fit_input", record["fit_input_path"], record["fit_input_sha256"]),
                 ("score_rows", record["score_row_bundle_path"], record["score_row_bundle_sha256"]),
                 ("fit_state", record["fit_state_path"], record["fit_state_sha256"]))
        for suffix, referenced, expected_sha in pairs:
            expected_path = INPUT_FILES[f"{key}_{suffix}"]
            resolved = _repo_path(referenced)
            if resolved != expected_path.resolve() or sha256_file(resolved) != expected_sha:
                raise DiagnosticError(f"R4 {fit_id} {suffix} metadata or file hash mismatch")
        if record["fit_rows"] != FIT_EXPECTED[fit_id] or record["score_rows"] != SCORE_EXPECTED[fit_id]:
            raise DiagnosticError(f"R4 {fit_id} expected population counts differ")
        global_pin = r4_freeze["input_hashes"].get(f"r3_stage__{fit_id[-2:]}_GLOBAL_state", {})
        global_path = INPUT_FILES[f"{key}_global_state"]
        if (_repo_path(global_pin.get("path", "")) != global_path.resolve()
                or sha256_file(global_path) != global_pin.get("sha256")
                or record.get("source_global_state_sha256") != global_pin.get("sha256")):
            raise DiagnosticError(f"R4 {fit_id} GLOBAL state pin mismatch")
    if r4_pass.get("authorized_fit_ids") != ["HINGE_D1", "HINGE_D2"]:
        raise DiagnosticError("R4 predecessor authorization differs from the diagnostic pair")
    if (r4_pass.get("prefit_freeze_sha256") != sha256_file(INPUT_FILES["r4_freeze"])
            or r4_pass.get("source_commit") != r4_freeze.get("source_commit")):
        raise DiagnosticError("R4 root pass does not bind its frozen diagnostic source")


def _resource_policy_body() -> dict[str, int]:
    value = json.loads(RESOURCE_POLICY.read_text(encoding="utf-8"))
    saved = value.get("workloads", {}).get("saved_prediction_evaluation", {})
    expected = {"minimum_available_bytes": 1_610_612_736,
        "incremental_estimate_bytes": 536_870_912,
        "minimum_projected_free_bytes": 805_306_368,
        "minimum_actual_free_at_boundary_bytes": 805_306_368}
    if saved != expected:
        raise DiagnosticError("Saved-replay memory policy differs from approved workload limits")
    return expected


def _pinned_files() -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for category, files in (("source", SOURCE_FILES), ("reference_source", REFERENCE_SOURCES),
                            ("input", INPUT_FILES)):
        for name, path in files.items():
            resolved = Path(path).resolve()
            if not resolved.is_file():
                raise DiagnosticError(f"Pinned file is absent: {resolved}")
            result[f"{category}:{name}"] = {"path": str(resolved), "sha256": sha256_file(resolved)}
    return result


def freeze_inputs() -> dict[str, Any]:
    if FREEZE.exists():
        raise DiagnosticError(f"Input freeze already exists: {FREEZE}")
    if any(path.exists() for path in (SUMMARY, PLOT_BINS, PLOT_CYCLES, VERIFICATION, MEMO)):
        raise DiagnosticError("A result artifact already exists before input freeze")
    _diagnostic_metadata_check()
    policy_resource = _resource_policy_body()
    body = {
        "schema_version": 1,
        "experiment_id": NAMESPACE.upper() + "_R1",
        "status": "FROZEN_BEFORE_NUMERICAL_ARRAY_ACCESS",
        "source_commit": _git_head(),
        "policy": {"raw_direct_bin_edges": list(BIN_EDGES),
            "raw_direct_bin_labels": list(BIN_LABELS),
            "minimum_present_per_bin": MIN_BIN_PRESENT,
            "minimum_present_per_cycle": MIN_CYCLE_PRESENT,
            "target_tail_low": TARGET_TAIL_LOW,
            "target_tail_high": TARGET_TAIL_HIGH,
            "fit_rows": FIT_EXPECTED, "score_rows": SCORE_EXPECTED,
            "policies": list(PRESENT_POLICIES),
            "missing_route": "preserve raw DIRECT float64 bits",
            "fit_population": "all cached fit rows; report denominator separately",
            "metric_sign": "gain = MAE(GLOBAL-present) - MAE(HINGE-present); positive favors HINGE",
            "unsupported": "UNKNOWN_UNSUPPORTED with null metrics; never zero-filled",
            "counterfactual_mappings": "not computed",
            "data_boundary": "cached R4 D1/D2 fit-input and score-row bundles only"},
        "resource_policy": policy_resource,
        "shared_lock_path": str(r4.SHARED_JOB_LOCK.resolve()),
        "job_limit": 1,
        "thread_limit": 2,
        "numerical_arrays_loaded": False,
        "raw_train_labels_opened": False,
        "test_or_sample_opened": False,
        "e79_opened": False,
        "source_and_input_files": _pinned_files(),
        "r4_stage_rows": {
            "HINGE_D1": {"fit": 4_950, "score": 7_462, "fit_cycles": [68, 71],
                          "score_cycles": [72, 75], "origin": 67},
            "HINGE_D2": {"fit": 12_412, "score": 8_491, "fit_cycles": [68, 75],
                          "score_cycles": [76, 79], "fit_origin": 67, "score_origin": 75}},
    }
    body["signature"] = canonical_sha(body)
    digest = write_json_once(FREEZE, body)
    return {"status": body["status"], "path": str(FREEZE.resolve()), "sha256": digest,
            "source_commit": body["source_commit"], "pinned_file_count": len(body["source_and_input_files"])}


def _verify_freeze() -> tuple[dict[str, Any], str]:
    if not FREEZE.is_file():
        raise DiagnosticError("Frozen input manifest is absent; run --freeze before analysis")
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    signature = freeze.pop("signature", None)
    if signature != canonical_sha(freeze):
        raise DiagnosticError("Input-freeze canonical signature mismatch")
    freeze["signature"] = signature
    freeze_sha = sha256_file(FREEZE)
    if freeze.get("status") != "FROZEN_BEFORE_NUMERICAL_ARRAY_ACCESS":
        raise DiagnosticError("Input-freeze status is not the required precomputation state")
    if _git_head() != freeze.get("source_commit"):
        raise DiagnosticError("Source commit changed after input freeze")
    _diagnostic_metadata_check()
    if _resource_policy_body() != freeze.get("resource_policy"):
        raise DiagnosticError("Current saved-replay resource policy differs from frozen policy")
    for item in freeze["source_and_input_files"].values():
        path = Path(item["path"])
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise DiagnosticError(f"Frozen source/input hash mismatch: {path}")
    return freeze, freeze_sha


def _resource_snapshot(boundary: str, *, initial: bool,
                       limits: Mapping[str, int]) -> dict[str, Any]:
    available = int(psutil.virtual_memory().available)
    projected = available - int(limits["incremental_estimate_bytes"])
    initial_ok = (available >= int(limits["minimum_available_bytes"])
                  and projected >= int(limits["minimum_projected_free_bytes"]))
    boundary_ok = (available >= int(limits["minimum_actual_free_at_boundary_bytes"])
                   and projected >= int(limits["minimum_projected_free_bytes"]))
    passed = (initial_ok and boundary_ok) if initial else boundary_ok
    if not passed:
        raise DiagnosticError(f"Resource gate failed at {boundary}: available={available}, projected={projected}")
    process = psutil.Process(os.getpid())
    memory = process.memory_info()
    peak = getattr(memory, "peak_wset", None)
    return {"status": "PASS", "boundary": boundary,
        "available_bytes": available,
        "incremental_estimate_bytes": int(limits["incremental_estimate_bytes"]),
        "projected_free_bytes": projected,
        "minimum_initial_available_bytes": int(limits["minimum_available_bytes"]),
        "minimum_projected_free_bytes": int(limits["minimum_projected_free_bytes"]),
        "minimum_actual_free_at_boundary_bytes": int(limits["minimum_actual_free_at_boundary_bytes"]),
        "initial_gate_pass": bool(initial_ok), "boundary_gate_pass": bool(boundary_ok),
        "process_rss_bytes": int(memory.rss), "process_rss_measured": True,
        "os_peak_working_set_bytes": int(peak) if peak is not None else None,
        "os_peak_working_set_measured": peak is not None,
        "thread_limit": 2, "concurrent_heavy_jobs": 1}


def _load_arrays(path: Path, fields: tuple[str, ...], *,
                 permitted_unloaded: tuple[str, ...] = ()) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if not set(fields).issubset(archive.files):
            raise DiagnosticError(f"Cached bundle is missing expected arrays: {path}")
        extras = set(archive.files) - set(fields)
        if extras != set(permitted_unloaded):
            raise DiagnosticError(f"Cached bundle has unexpected arrays: {path}/{sorted(extras)}")
        return {name: np.asarray(archive[name]).copy() for name in fields}


def _state_parameters(fit_state: Mapping[str, Any], global_state: Mapping[str, Any]) -> tuple[dict[str, float], tuple[float, float]]:
    hinge = fit_state.get("solver_result", {})
    glob = global_state.get("solver_result", {})
    if fit_state.get("status") != "LP_COMPLETE" or hinge.get("success") is not True:
        raise DiagnosticError("Cached HINGE state is not completed")
    if global_state.get("status") != "LP_COMPLETE" or glob.get("success") is not True:
        raise DiagnosticError("Cached GLOBAL state is not completed")
    hp = {name: float(hinge[name]) for name in ("intercept", "slope_low", "slope_high")}
    gp = (float(glob["intercept"]), float(glob["slope"]))
    if not all(math.isfinite(v) for v in (*hp.values(), *gp)):
        raise DiagnosticError("Cached coefficients are non-finite")
    return hp, gp


def _predictions(direct: np.ndarray, missing: np.ndarray,
                 hinge: Mapping[str, float], global_: tuple[float, float]) -> dict[str, np.ndarray]:
    p = np.asarray(direct, dtype="<f8")
    miss = np.asarray(missing, dtype=bool)
    if p.ndim != 1 or miss.shape != p.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 100)):
        raise DiagnosticError("Cached DIRECT/missing arrays are invalid")
    a, sl, sh = hinge["intercept"], hinge["slope_low"], hinge["slope_high"]
    present = ~miss
    h = p.copy()
    h[present] = np.clip(a + sl * np.minimum(p[present], 50.0)
                         + sh * np.maximum(p[present] - 50.0, 0.0), 0.0, 100.0)
    ga, gb = global_
    g = p.copy()
    g[present] = np.clip(ga + gb * p[present], 0.0, 100.0)
    if not np.array_equal(h[miss].view("<u8"), p[miss].view("<u8")):
        raise DiagnosticError("Cached HINGE missing route is not bit-exact")
    if not np.array_equal(g[miss].view("<u8"), p[miss].view("<u8")):
        raise DiagnosticError("Cached GLOBAL missing route is not bit-exact")
    return {"raw_direct": p.copy(), "causal_global_present_only": g,
            "hinge_present_only": h}


def _float_vector_sha(values: np.ndarray) -> str:
    data = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    return hashlib.sha256(b"float64-le\0" + str(data.shape).encode("ascii") + b"\0" + data.tobytes()).hexdigest()


def _string_vector_sha(values: np.ndarray) -> str:
    h = hashlib.sha256(b"utf8-length-prefixed\0")
    for value in np.asarray(values).astype(str):
        raw = value.encode("utf-8")
        h.update(len(raw).to_bytes(8, "little"))
        h.update(raw)
    return h.hexdigest()


def _row_identity_sha(arrays: Mapping[str, np.ndarray]) -> str:
    rows = np.asarray(arrays["physical_rows"], dtype=np.int64)
    ids = np.asarray(arrays["record_ids"]).astype(str)
    origins = np.asarray(arrays["prediction_origins"], dtype=np.int64)
    indices = np.asarray(arrays["original_bundle_indices"], dtype=np.int64)
    cycles = np.asarray(arrays["cycles"], dtype=np.int64)
    direct = np.asarray(arrays["direct_predictions"], dtype="<f8")
    missing = np.asarray(arrays["missing_mask"], dtype=bool)
    targets = np.asarray(arrays["targets"], dtype="<f8")
    n = len(rows)
    if any(len(a) != n for a in (ids, origins, indices, cycles, direct, missing, targets)):
        raise DiagnosticError("Cached row identity arrays are misaligned")
    h = hashlib.sha256(b"reg-hinge-residual-r1-row-identity\0")
    for i in range(n):
        record = [int(rows[i]), ids[i], int(origins[i]), int(indices[i]), int(cycles[i]),
                  float(direct[i]).hex(), bool(missing[i]), float(targets[i]).hex()]
        h.update(json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _row_hashes(arrays: Mapping[str, np.ndarray]) -> dict[str, str]:
    return {"row_identity_sha256": _row_identity_sha(arrays),
        "record_ids_sha256": _string_vector_sha(arrays["record_ids"]),
        "direct_sha256": _float_vector_sha(arrays["direct_predictions"]),
        "targets_sha256": _float_vector_sha(arrays["targets"]),
        "cycles_sha256": hashlib.sha256(np.ascontiguousarray(np.asarray(arrays["cycles"], dtype="<i2")).tobytes()).hexdigest(),
        "missing_mask_sha256": hashlib.sha256(np.ascontiguousarray(np.asarray(arrays["missing_mask"], dtype=np.uint8)).tobytes()).hexdigest()}


def _bin_indices(direct: np.ndarray) -> np.ndarray:
    p = np.asarray(direct, dtype=np.float64)
    if not np.isfinite(p).all() or np.any((p < 0.0) | (p > 100.0)):
        raise DiagnosticError("Raw DIRECT value outside frozen bin domain")
    return np.clip(np.searchsorted(np.asarray(BIN_EDGES), p, side="right") - 1,
                   0, len(BIN_LABELS) - 1).astype(np.int8)


def _metric(values: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    error = np.asarray(values, dtype=np.float64) - np.asarray(targets, dtype=np.float64)
    return {"n": int(len(error)), "mae": float(np.mean(np.abs(error))),
            "mean_error": float(np.mean(error)),
            "p90_absolute_error": float(np.quantile(np.abs(error), 0.90))}


def _supported_metric(values: np.ndarray, targets: np.ndarray, minimum: int) -> dict[str, Any]:
    n = len(targets)
    if n < minimum:
        return {"status": "UNKNOWN_UNSUPPORTED", "n": int(n), "mae": None,
                "mean_error": None, "p90_absolute_error": None}
    return {"status": "SUPPORTED", **_metric(values, targets)}


def _cell_summary(mask: np.ndarray, direct: np.ndarray, targets: np.ndarray,
                  missing: np.ndarray, predictions: Mapping[str, np.ndarray],
                  *, minimum: int) -> dict[str, Any]:
    rows = np.asarray(mask, dtype=bool)
    present = rows & ~missing
    absent = rows & missing
    all_n, present_n, missing_n = int(rows.sum()), int(present.sum()), int(absent.sum())
    result: dict[str, Any] = {
        "n_all": all_n, "n_present": present_n, "n_missing": missing_n,
        "support_all": "SUPPORTED" if all_n >= minimum else "UNKNOWN_UNSUPPORTED",
        "support_present": "SUPPORTED" if present_n >= minimum else "UNKNOWN_UNSUPPORTED",
        "target_le_5_n": int(np.sum(rows & (targets <= TARGET_TAIL_LOW))),
        "target_ge_95_n": int(np.sum(rows & (targets >= TARGET_TAIL_HIGH))),
    }
    result["target_tail_support"] = {}
    for name, tail_mask in (("target_le_5", targets <= TARGET_TAIL_LOW),
                            ("target_ge_95", targets >= TARGET_TAIL_HIGH)):
        tail_all = rows & tail_mask
        tail_present = present & tail_mask
        tail_missing = absent & tail_mask
        all_tail_n, present_tail_n, missing_tail_n = (int(tail_all.sum()),
            int(tail_present.sum()), int(tail_missing.sum()))
        result["target_tail_support"][name] = {
            "n_all": all_tail_n, "n_present": present_tail_n, "n_missing": missing_tail_n,
            "support_all": "SUPPORTED" if all_tail_n >= minimum else "UNKNOWN_UNSUPPORTED",
            "support_present": "SUPPORTED" if present_tail_n >= minimum else "UNKNOWN_UNSUPPORTED"}
    result["median_residual_y_minus_prediction"] = {"all_rows": {}, "present_only": {}}
    result["policy_mae"] = {"all_rows": {}, "present_only": {}}
    for name in PRESENT_POLICIES:
        residual = targets - predictions[name]
        result["median_residual_y_minus_prediction"]["all_rows"][name] = (
            float(np.median(residual[rows])) if all_n >= minimum else None)
        result["median_residual_y_minus_prediction"]["present_only"][name] = (
            float(np.median(residual[present])) if present_n >= minimum else None)
        result["policy_mae"]["all_rows"][name] = _supported_metric(
            predictions[name][rows], targets[rows], minimum)
        result["policy_mae"]["present_only"][name] = _supported_metric(
            predictions[name][present], targets[present], minimum)
    result["gain_hinge_over_global"] = {"all_rows": None, "present_only": None}
    for scope, support_n in (("all_rows", all_n), ("present_only", present_n)):
        if support_n >= minimum:
            result["gain_hinge_over_global"][scope] = (
                result["policy_mae"][scope]["causal_global_present_only"]["mae"]
                - result["policy_mae"][scope]["hinge_present_only"]["mae"])
    return result


def _population_summary(fit_id: str, role: str, arrays: Mapping[str, np.ndarray],
                        predictions: Mapping[str, np.ndarray]) -> dict[str, Any]:
    n = len(arrays["targets"])
    direct = np.asarray(arrays["direct_predictions"], dtype="<f8")
    targets = np.asarray(arrays["targets"], dtype="<f8")
    missing = np.asarray(arrays["missing_mask"], dtype=bool)
    cycles = np.asarray(arrays["cycles"], dtype=np.int64)
    origins = np.asarray(arrays["prediction_origins"], dtype=np.int64)
    if (n == 0 or any(len(v) != n for v in (direct, targets, missing, cycles, origins))
            or len(np.unique(arrays["record_ids"])) != n
            or len(np.unique(arrays["physical_rows"])) != n
            or not np.isfinite(targets).all() or np.any((targets < 0) | (targets > 100))):
        raise DiagnosticError(f"Invalid aligned arrays for {fit_id} {role}")
    if (not np.all(origins == EXPECTED_ORIGIN[(fit_id, role)])
            or tuple(sorted(np.unique(cycles).tolist())) != EXPECTED_CYCLES[(fit_id, role)]):
        raise DiagnosticError(f"Unexpected prediction origin or cycle population for {fit_id} {role}")
    bins = _bin_indices(direct)
    expected_n = FIT_EXPECTED[fit_id] if role == "fit" else SCORE_EXPECTED[fit_id]
    if n != expected_n:
        raise DiagnosticError(f"Unexpected {fit_id} {role} row count: {n}")
    minimum = MIN_BIN_PRESENT if role == "fit" else MIN_BIN_PRESENT
    bins_summary = []
    for index, label in enumerate(BIN_LABELS):
        bins_summary.append({"bin_index": index, "label": label,
            **_cell_summary(bins == index, direct, targets, missing, predictions, minimum=minimum)})
    cycles_summary = []
    for cycle in sorted(np.unique(cycles).tolist()):
        cycles_summary.append({"cycle": int(cycle),
            **_cell_summary(cycles == cycle, direct, targets, missing, predictions,
                            minimum=MIN_CYCLE_PRESENT)})
    return {"fit_id": fit_id, "population": role, "n_all": n,
        "n_present": int((~missing).sum()), "n_missing": int(missing.sum()),
        "fit_coefficients_applied_to_all_rows": role == "fit",
        "mapping_application_present_only": True,
        "row_hashes": _row_hashes(arrays),
        "target_tail_support": {"target_le_5": {
                "n_all": int(np.sum(targets <= TARGET_TAIL_LOW)),
                "n_present": int(np.sum((targets <= TARGET_TAIL_LOW) & ~missing)),
                "n_missing": int(np.sum((targets <= TARGET_TAIL_LOW) & missing))},
            "target_ge_95": {
                "n_all": int(np.sum(targets >= TARGET_TAIL_HIGH)),
                "n_present": int(np.sum((targets >= TARGET_TAIL_HIGH) & ~missing)),
                "n_missing": int(np.sum((targets >= TARGET_TAIL_HIGH) & missing))},
            "diagnostic_only": True},
        "by_raw_direct_bin": bins_summary,
        "by_cycle": cycles_summary,
        "overall_policy_mae": _cell_summary(np.ones(n, dtype=bool), direct, targets,
            missing, predictions, minimum=1)["policy_mae"],
        "overall_gain_hinge_over_global": {
            "all_rows": float(np.mean(np.abs(targets - predictions["causal_global_present_only"]))
                - np.mean(np.abs(targets - predictions["hinge_present_only"]))),
            "present_only": float(np.mean(np.abs(targets[~missing] - predictions["causal_global_present_only"][~missing]))
                - np.mean(np.abs(targets[~missing] - predictions["hinge_present_only"][~missing])))
                if np.any(~missing) else None}}


def _compare_trees(expected: Any, actual: Any, path: str = "root") -> float:
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected) != set(actual):
            raise DiagnosticError(f"Score metric structure mismatch at {path}")
        return max((_compare_trees(expected[k], actual[k], f"{path}.{k}") for k in expected), default=0.0)
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            raise DiagnosticError(f"Score metric list length mismatch at {path}")
        return max((_compare_trees(a, b, f"{path}[{i}]") for i, (a, b) in enumerate(zip(expected, actual))), default=0.0)
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(actual, (int, float)) or not math.isclose(float(expected), float(actual), rel_tol=1e-12, abs_tol=1e-12):
            raise DiagnosticError(f"Score metric value mismatch at {path}: {expected} != {actual}")
        return abs(float(expected) - float(actual))
    if expected != actual:
        raise DiagnosticError(f"Score metric mismatch at {path}: {expected} != {actual}")
    return 0.0


def _existing_score_metrics(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    y = np.asarray(arrays["targets"], dtype=np.float64)
    cycles = np.asarray(arrays["cycles"], dtype=np.int64)
    missing = np.asarray(arrays["missing_mask"], dtype=bool)
    result = {}
    for name in SCORED_POLICIES:
        prediction = np.asarray(arrays[name], dtype=np.float64)
        groups = {"present": ~missing, "missing": missing,
                  "target_le_5": y <= TARGET_TAIL_LOW,
                  "target_ge_95": y >= TARGET_TAIL_HIGH}
        detail = {"overall": _metric(prediction, y), "groups": {}}
        for group, mask in groups.items():
            count = int(mask.sum())
            detail["groups"][group] = ({"n": count, "status": "UNKNOWN_UNSUPPORTED", "mae": None}
                if count < 200 else {"status": "SUPPORTED", **_metric(prediction[mask], y[mask])})
        detail["groups"]["by_cycle"] = {str(int(cycle)): _metric(prediction[cycles == cycle], y[cycles == cycle])
            for cycle in np.unique(cycles)}
        result[name] = detail
    return result


def _signed_score_tail_metrics(diag_record: Mapping[str, Any]) -> dict[str, Any]:
    result = {}
    for policy in SCORED_POLICIES:
        metrics = diag_record["score_metrics"][policy]
        result[policy] = {"overall_p90_absolute_error": metrics["overall"]["p90_absolute_error"],
            "target_le_5": metrics["groups"]["target_le_5"],
            "target_ge_95": metrics["groups"]["target_ge_95"],
            "source": "existing_signed_R4_diagnostics"}
    return result


def _records_for_fit(fit_id: str, resources: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    key = fit_id[-2:].lower()
    fit_path, score_path = INPUT_FILES[f"{key}_fit_input"], INPUT_FILES[f"{key}_score_rows"]
    state_path, global_path = INPUT_FILES[f"{key}_fit_state"], INPUT_FILES[f"{key}_global_state"]
    fit_state = json.loads(state_path.read_text(encoding="utf-8"))
    global_state = json.loads(global_path.read_text(encoding="utf-8"))
    hinge_params, global_params = _state_parameters(fit_state, global_state)
    fit_arrays = _load_arrays(fit_path, FIT_FIELDS)
    resources.append(_resource_snapshot(f"{fit_id}_after_fit_input_load", initial=False,
                                        limits=_resource_policy_body()))
    fit_preds = _predictions(fit_arrays["direct_predictions"], fit_arrays["missing_mask"],
                             hinge_params, global_params)
    fit_summary = _population_summary(fit_id, "fit", fit_arrays, fit_preds)
    del fit_arrays, fit_preds
    gc.collect()
    resources.append(_resource_snapshot(f"{fit_id}_after_fit_release", initial=False,
                                        limits=_resource_policy_body()))

    score_arrays = _load_arrays(score_path, SCORE_FIELDS,
                                permitted_unloaded=("causal_global_all_rows",))
    resources.append(_resource_snapshot(f"{fit_id}_after_score_load", initial=False,
                                        limits=_resource_policy_body()))
    score_preds = _predictions(score_arrays["direct_predictions"], score_arrays["missing_mask"],
                               hinge_params, global_params)
    for name in ("raw_direct", "causal_global_present_only", "hinge_present_only"):
        if not np.array_equal(np.asarray(score_arrays[name], dtype="<f8").view("<u8"),
                              np.asarray(score_preds[name], dtype="<f8").view("<u8")):
            raise DiagnosticError(f"Cached score prediction differs from saved-state replay: {fit_id}/{name}")
    diagnostics = json.loads(INPUT_FILES["r4_diagnostics"].read_text(encoding="utf-8"))
    diag_record = next(stage for stage in diagnostics["stages"] if stage["fit_id"] == fit_id)
    computed_metrics = _existing_score_metrics(score_arrays)
    selected_score_metrics = {name: diag_record["score_metrics"][name] for name in SCORED_POLICIES}
    score_metric_diff = _compare_trees(selected_score_metrics, computed_metrics,
                                       f"{fit_id}.score_metrics")
    score_summary = _population_summary(fit_id, "score", score_arrays, score_preds)
    score_summary["existing_r4_score_metrics_replayed"] = True
    score_summary["max_difference_from_signed_r4_score_metrics"] = score_metric_diff
    score_summary["existing_r4_score_tail_and_p90"] = _signed_score_tail_metrics(diag_record)
    del score_arrays, score_preds
    gc.collect()
    resources.append(_resource_snapshot(f"{fit_id}_after_score_release", initial=False,
                                        limits=_resource_policy_body()))
    return fit_summary, score_summary


def _make_plots(populations: list[dict[str, Any]]) -> dict[str, str]:
    if PLOT_BINS.exists() or PLOT_CYCLES.exists():
        raise DiagnosticError("A plot output already exists; refusing overwrite")
    lookup = {(item["fit_id"], item["population"]): item for item in populations}
    colors = {"raw_direct": "#59636e", "causal_global_present_only": "#1479b8",
              "hinge_present_only": "#e07a18"}
    labels = {"raw_direct": "Raw DIRECT", "causal_global_present_only": "GLOBAL present",
              "hinge_present_only": "HINGE present"}
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.3), constrained_layout=True, sharex=True)
    for row, fit_id in enumerate(("HINGE_D1", "HINGE_D2")):
        for col, role in enumerate(("fit", "score")):
            ax = axes[row, col]
            pop = lookup[(fit_id, role)]
            bins = pop["by_raw_direct_bin"]
            x = np.arange(len(BIN_LABELS))
            for method in PRESENT_POLICIES:
                y = [cell["median_residual_y_minus_prediction"]["present_only"][method]
                     for cell in bins]
                ax.plot(x, [np.nan if value is None else value for value in y],
                        marker="o", linewidth=1.8, color=colors[method], label=labels[method])
            ax.axhline(0.0, color="#333333", linewidth=0.8, alpha=0.75)
            ax.set_title(f"{fit_id} {role}: all={pop['n_all']}, present={pop['n_present']}, missing={pop['n_missing']}")
            ax.set_ylabel("median y − prediction")
            ax.set_xticks(x, BIN_LABELS)
            ax.grid(axis="y", alpha=0.2)
            if row == 1:
                ax.set_xlabel("raw DIRECT bin")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("Cached residual medians by raw DIRECT bin (present rows; unsupported n<200 omitted)", fontsize=13)
    fig.savefig(PLOT_BINS, dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.7), constrained_layout=True, sharey=True)
    for row, fit_id in enumerate(("HINGE_D1", "HINGE_D2")):
        for col, role in enumerate(("fit", "score")):
            ax = axes[row, col]
            pop = lookup[(fit_id, role)]
            cycles = pop["by_cycle"]
            x = np.arange(len(cycles), dtype=float)
            all_gain = [c["gain_hinge_over_global"]["all_rows"] if c["support_all"] == "SUPPORTED" else np.nan
                        for c in cycles]
            present_gain = [c["gain_hinge_over_global"]["present_only"] if c["support_present"] == "SUPPORTED" else np.nan
                            for c in cycles]
            ax.bar(x - 0.18, all_gain, width=0.36, color="#9aa3ab", label="all-row denominator")
            ax.bar(x + 0.18, present_gain, width=0.36, color="#1479b8", label="present-only denominator")
            ax.axhline(0.0, color="#333333", linewidth=0.8)
            ax.set_title(f"{fit_id} {role}: n={pop['n_all']}")
            ax.set_xticks(x, [f"{c['cycle']}\np={c['n_present']}" for c in cycles])
            ax.set_ylabel("GLOBAL MAE − HINGE MAE")
            ax.grid(axis="y", alpha=0.2)
            if row == 1:
                ax.set_xlabel("cycle (present-row support below threshold is omitted)")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("Cached policy gain by cycle; positive favors HINGE", fontsize=13)
    fig.savefig(PLOT_CYCLES, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return {"raw_bin_residuals_sha256": sha256_file(PLOT_BINS),
            "cycle_gains_sha256": sha256_file(PLOT_CYCLES)}


def run_diagnostic() -> dict[str, Any]:
    freeze, freeze_sha = _verify_freeze()
    outputs = (SUMMARY, PLOT_BINS, PLOT_CYCLES, VERIFICATION, MEMO)
    if any(path.exists() for path in outputs):
        raise DiagnosticError("A diagnostic output exists; refusing rerun or overwrite")
    resources: list[dict[str, Any]] = []
    start = time.perf_counter()
    limits = freeze["resource_policy"]
    with r4.SharedHeavyJobLock(r4.SHARED_JOB_LOCK):
        _verify_freeze()
        resources.append(_resource_snapshot("initial_locked_before_npz", initial=True, limits=limits))
        populations: list[dict[str, Any]] = []
        with threadpool_limits(limits=2):
            for fit_id in ("HINGE_D1", "HINGE_D2"):
                fit_summary, score_summary = _records_for_fit(fit_id, resources)
                populations.extend((fit_summary, score_summary))
        plot_hashes = _make_plots(populations)
        resources.append(_resource_snapshot("after_summary_and_plots", initial=False, limits=limits))
        diagnostics = json.loads(INPUT_FILES["r4_diagnostics"].read_text(encoding="utf-8"))
        score_gain = {stage["fit_id"]: stage["gain_vs_causal_global_present_only"]
                      for stage in diagnostics["stages"]}
        body = {
            "schema_version": 1,
            "experiment_id": NAMESPACE.upper() + "_R1",
            "status": "PASS_ZERO_FIT_CACHED_RESIDUAL_DIAGNOSTIC",
            "input_freeze_sha256": freeze_sha,
            "source_commit": freeze["source_commit"],
            "source_and_input_hashes": freeze["source_and_input_files"],
            "policy": freeze["policy"],
            "populations": populations,
            "existing_r4_score_gains": score_gain,
            "resource_snapshots": resources,
            "runtime_seconds": float(time.perf_counter() - start),
            "new_fits": 0,
            "new_calibrators": 0,
            "new_counterfactual_policies": 0,
            "raw_train_label_access": False,
            "test_or_sample_access": False,
            "e79_access": False,
            "kaggle_access": False,
            "two_plot_hashes": plot_hashes,
        }
        body["signature"] = canonical_sha(body)
        summary_sha = write_json_once(SUMMARY, body)
    return {"status": body["status"], "summary_path": str(SUMMARY.resolve()),
            "summary_sha256": summary_sha, "freeze_sha256": freeze_sha,
            "runtime_seconds": body["runtime_seconds"], "plot_hashes": plot_hashes}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--run", action="store_true")
    args = parser.parse_args()
    result = freeze_inputs() if args.freeze else run_diagnostic()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DiagnosticError as exc:
        print(json.dumps({"status": "BLOCKED_OR_FAIL", "reason": str(exc)}, sort_keys=True), file=sys.stderr)
        raise SystemExit(1)
