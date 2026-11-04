"""Frozen two-candidate REG calibration follow-up with bounded temporal diagnostics."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any

import numpy as np
import pandas as pd
import psutil
from scipy import sparse
from scipy.optimize import OptimizeWarning, linprog
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))
import scripts.reg_final_wave_v2 as v2
from scripts.reg_position_failure_mechanism_v1 import read_scored_query
from src.models.optimization_v3_batch1_r17 import apply_affine, solve_nonnegative_l1

VERSION = "REG_FINAL_WAVE_FOLLOWUP_V1"
DOC = ROOT / "reports/optimization_v3/REG_FINAL_WAVE_FOLLOWUP_V1/PROTOCOL.md"
SYNTH_QA = ROOT / "reports/optimization_v3/REG_FINAL_WAVE_FOLLOWUP_V1/SYNTHETIC_QA.json"
FREEZE = ROOT / "experiments/REG_FINAL_WAVE_FOLLOWUP_V1/PREFIT_FREEZE.json"
REVIEW = ROOT / "experiments/REG_FINAL_WAVE_FOLLOWUP_V1/INDEPENDENT_REVIEW.json"
RUN = ROOT / "experiments/runs/REG_FINAL_WAVE_FOLLOWUP_V1"
OUT_DIR = ROOT / "submissions/reg_final_wave_followup_v1"
OUT_A = OUT_DIR / "recency_affine_h8_e79.csv"
OUT_B = OUT_DIR / "global_affine_present_only_e79.csv"
DIAG_REPORT = ROOT / "reports/optimization_v3/REG_FINAL_WAVE_FOLLOWUP_V1/DIAGNOSTICS.json"
MANIFEST_A = ROOT / "reports/optimization_v3/REG_FINAL_WAVE_FOLLOWUP_V1/MANIFEST_A.json"
MANIFEST_B = ROOT / "reports/optimization_v3/REG_FINAL_WAVE_FOLLOWUP_V1/MANIFEST_B.json"
QA_A = ROOT / "reports/optimization_v3/REG_FINAL_WAVE_FOLLOWUP_V1/QA_A.json"
QA_B = ROOT / "reports/optimization_v3/REG_FINAL_WAVE_FOLLOWUP_V1/QA_B.json"
RUN_RECEIPT = RUN / "RUN_RECEIPT.json"
LEDGER = RUN / "lp_ledger.jsonl"
BLOCKER = RUN / "BLOCKER.json"

V2_SOURCE = ROOT / "scripts/reg_final_wave_v2.py"
R17_SOURCE = ROOT / "src/models/optimization_v3_batch1_r17.py"
READER_SOURCE = ROOT / "scripts/reg_position_failure_mechanism_v1.py"
R16_SOURCE = v2.R16_MODEL_SOURCE
GLOBAL_STATE = RUN.parent / "REG_FINAL_WAVE_V2/CALIBRATOR_STATE.json"
GLOBAL_FIT_RECEIPT = RUN.parent / "REG_FINAL_WAVE_V2/FIT_RECEIPT.json"
ROOT_UPLOAD_DIR = ROOT / "reports/optimization_v3/REG_FINAL_WAVE_V1/SOL_UPLOAD_V2"
GLOBAL_SCORE = ROOT_UPLOAD_DIR / "SCORE_LATEST.json"
GLOBAL_RESULT = ROOT_UPLOAD_DIR / "RESULT.md"
GLOBAL_UPLOAD_EVENTS = ROOT_UPLOAD_DIR / "UPLOAD_EVENTS.jsonl"

EXPECTED_TRAIN_SHA = v2.TRAIN_SHA
PRIOR_GLOBAL_A = 2.169196428141562
PRIOR_GLOBAL_B = 0.9744451489746011
PRIOR_GLOBAL_REF = 57994383
PRIOR_GLOBAL_PUBLIC_MAE = 11.04552
HALF_LIFE = 8
WEIGHTED_FIT_CYCLES = tuple(range(68, 80))
WINDOWS = ((71, 72, 75), (75, 76, 79))
LP_PLAN = (
    "diag_t71_global",
    "diag_t71_recency_h8",
    "diag_t75_global",
    "diag_t75_recency_h8",
    "production_recency_h8_full_oof",
)
MAX_LP_STARTS = 5
THREADS = 2
MIN_AVAILABLE = 4 * 1024**3
RESERVED_RAM = 2 * 1024**3
ESTIMATED_PEAK = 512 * 1024**2
DIAGNOSTIC_MIN_ROWS = 1000
PRODUCTION_MIN_ROWS = 5000
EXPECTED_SUPPORT_ROWS = 20903
EXPECTED_BASE_ROWS = 29421


class GateError(RuntimeError):
    """A fail-closed source, protocol, runtime, or output gate failed."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8", errors="surrogatepass")


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_f64(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def sha_array(value: Any) -> str:
    return v2.sha_pilot_array(value)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_once(path: Path, value: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(json_safe(value), sort_keys=True, ensure_ascii=False,
                         allow_nan=False, indent=2).encode("utf-8") + b"\n"
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(payload).hexdigest()


def append_event(path: Path, event: dict[str, Any]) -> None:
    payload = json.dumps(json_safe(event), sort_keys=True, ensure_ascii=False,
                         allow_nan=False, separators=(",", ":")) + "\n"
    fd = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GateError(f"Expected JSON object at {path}")
    return value


def verify_signed(value: dict[str, Any], field: str, label: str) -> None:
    body = {k: v for k, v in value.items()
            if k != field and not (field == "signature" and k == "signature_method")}
    if value.get(field) != canonical_sha(body):
        raise GateError(f"{label} integrity signature failed")


def source_hashes() -> dict[str, str]:
    return {
        "script": sha_file(Path(__file__).resolve()),
        "protocol": sha_file(DOC),
        "synthetic_qa": sha_file(SYNTH_QA),
        "v2_input_gate_and_bundle_loader": sha_file(V2_SOURCE),
        "r17_lp_helper": sha_file(R17_SOURCE),
        "selected_row_reader": sha_file(READER_SOURCE),
        "r16_half_life_rationale_source": sha_file(R16_SOURCE),
    }


def input_paths() -> dict[str, Path]:
    names = (
        "base_csv", "base_manifest", "base_qa", "base_archive",
        "base_trained_freeze", "base_prelabel", "base_upload_events",
        "direct_E67", "direct_E67_receipt", "direct_E75", "direct_E75_receipt",
        "direct_freeze", "direct_prelabel", "direct_source", "query_reader", "r17_source",
    )
    paths = {name: v2.PINS[name][0] for name in names}
    paths.update({
        "v2_global_csv": v2.OUT_CSV,
        "v2_global_manifest": v2.OUT_MANIFEST,
        "v2_global_qa": v2.OUT_QA,
        "v2_global_state": GLOBAL_STATE,
        "v2_global_fit_receipt": GLOBAL_FIT_RECEIPT,
        "v2_global_score_latest": GLOBAL_SCORE,
        "v2_global_result": GLOBAL_RESULT,
        "v2_global_upload_events": GLOBAL_UPLOAD_EVENTS,
        "r16_weight_rationale_source": R16_SOURCE,
    })
    return paths


def snapshot_input_hashes() -> dict[str, dict[str, str]]:
    result = {}
    for name, path in input_paths().items():
        if not path.is_file():
            raise GateError(f"Required pinned input is missing: {name}: {path}")
        result[name] = {"path": str(path), "sha256": sha_file(path)}
    return result


def verify_input_hashes(expected: dict[str, dict[str, str]]) -> None:
    actual = snapshot_input_hashes()
    if actual != expected:
        changed = [name for name in sorted(set(actual) | set(expected))
                   if actual.get(name) != expected.get(name)]
        raise GateError(f"Pinned input digest/path changed: {changed}")


def verify_existing_global(ids_base: np.ndarray, base_values: np.ndarray,
                           missing: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    manifest, qa = v2.read_json(v2.OUT_MANIFEST), v2.read_json(v2.OUT_QA)
    v2.verify_signed(manifest, "signature", "existing global affine manifest")
    v2.verify_signed(qa, "signature", "existing global affine QA")
    if (manifest.get("candidate") != "CURRENT_DIRECT_AFFINE_L1_E79"
            or manifest.get("csv_sha256") != sha_file(v2.OUT_CSV)
            or manifest.get("intercept") != PRIOR_GLOBAL_A
            or manifest.get("slope") != PRIOR_GLOBAL_B
            or manifest.get("upload_performed") is not False
            or qa.get("status") != "PASS_LOCAL_OUTPUT_QA"
            or qa.get("csv_sha256") != sha_file(v2.OUT_CSV)):
        raise GateError("Existing global affine artifact identity, coefficients, or QA changed")
    frame = pd.read_csv(v2.OUT_CSV, dtype={"record_id": str}, float_precision="round_trip")
    if list(frame.columns) != ["record_id", "threat_rank"] or len(frame) != EXPECTED_BASE_ROWS:
        raise GateError("Existing global affine CSV schema/count differs")
    ids = frame["record_id"].astype(str).to_numpy(dtype=str)
    values = pd.to_numeric(frame["threat_rank"], errors="raise").to_numpy(dtype=np.float64)
    if (not np.array_equal(ids, ids_base) or not np.isfinite(values).all()
            or np.any((values < 0.0) | (values > 100.0))
            or not np.array_equal(values, apply_affine(base_values, PRIOR_GLOBAL_A, PRIOR_GLOBAL_B))):
        raise GateError("Existing global affine CSV differs from exact base ID order/formula")
    score = json.loads(GLOBAL_SCORE.read_text(encoding="utf-8"))
    selected = [item for item in score if int(item.get("ref", -1)) == PRIOR_GLOBAL_REF]
    if (len(selected) != 1 or selected[0].get("status") != "SubmissionStatus.COMPLETE"
            or selected[0].get("publicScore") != "11.04552"
            or selected[0].get("fileName") != v2.OUT_CSV.name):
        raise GateError("Pinned previous public result does not identify the expected global affine upload")
    if not len(missing) == len(ids) or int(missing.sum()) != 1062:
        raise GateError("Frozen base missing-route mask count/alignment differs")
    present_only = values.copy()
    present_only[missing] = base_values[missing]
    return values, {
        "csv_sha256": sha_file(v2.OUT_CSV),
        "manifest_sha256": sha_file(v2.OUT_MANIFEST),
        "qa_sha256": sha_file(v2.OUT_QA),
        "public_ref": PRIOR_GLOBAL_REF,
        "public_status": selected[0]["status"],
        "public_score": selected[0]["publicScore"],
        "intercept": PRIOR_GLOBAL_A, "slope": PRIOR_GLOBAL_B,
        "rows": len(ids), "present_rows": int((~missing).sum()),
        "missing_rows": int(missing.sum()),
        "test_missing_mask_source": "signed existing B inference_B.npz missing array",
        "present_only_prediction_sha256": sha_f64(present_only),
        "posthoc_public_hypothesis": True,
    }


def load_fixed_support() -> dict[str, Any]:
    v2.verify_files(check_train=False)
    base_manifest, _base_qa, direct_freeze = v2.verify_base()
    ids_test, base_prediction, base_check = v2.base_csv_values(base_manifest)
    with np.load(v2.PINS["base_archive"][0], allow_pickle=False) as archive:
        archived_ids = np.asarray(archive["record_id"], dtype=str)
        missing = np.asarray(archive["missing"], dtype=bool)
    if not np.array_equal(ids_test, archived_ids):
        raise GateError("Base CSV ID order differs from signed prediction archive")
    global_values, global_check = verify_existing_global(ids_test, base_prediction, missing)
    e67, e75 = v2.load_bundle(67), v2.load_bundle(75)
    v2.verify_prefixes(e67, e75, direct_freeze)
    plan = v2.stitch_oof_rows(e67, e75)
    rows = np.asarray([item[0] for item in plan], dtype=np.int64)
    origins = np.asarray([item[1] for item in plan], dtype=np.int64)
    indices = np.asarray([item[2] for item in plan], dtype=np.int64)
    ids = np.asarray([item[3] for item in plan], dtype=str)
    cycles = np.asarray([item[4] for item in plan], dtype=np.int64)
    predictions, route_missing = [], []
    for _row, origin, index, _record_id, _cycle in plan:
        arrays = e67 if origin == 67 else e75
        current = np.asarray(arrays["prediction"], dtype=np.float64)
        predictions.append(float(current[index]))
        route_missing.append(bool(np.asarray(arrays["missing_mask"], dtype=bool)[index]))
    predictions = np.asarray(predictions, dtype=np.float64)
    route_missing = np.asarray(route_missing, dtype=bool)
    if (len(rows) != EXPECTED_SUPPORT_ROWS or len(set(rows.tolist())) != EXPECTED_SUPPORT_ROWS
            or len(set(ids.tolist())) != EXPECTED_SUPPORT_ROWS
            or set(np.unique(cycles).tolist()) != set(WEIGHTED_FIT_CYCLES)
            or not np.isfinite(predictions).all()
            or np.any((predictions < 0.0) | (predictions > 100.0))):
        raise GateError("Fixed direct OOF support failed row/ID/cycle/value integrity checks")
    windows = {}
    for t, lo, hi in WINDOWS:
        history = (cycles >= 68) & (cycles <= t)
        validation = (cycles >= lo) & (cycles <= hi)
        windows[str(t)] = {
            "fit_rows": int(history.sum()), "score_rows": int(validation.sum()),
            "fit_cycle_counts": {str(int(c)): int(np.sum(cycles[history] == c))
                                 for c in np.unique(cycles[history])},
            "score_cycle_counts": {str(int(c)): int(np.sum(cycles[validation] == c))
                                   for c in np.unique(cycles[validation])},
            "small_diagnostic_history": bool(history.sum() < 5000),
        }
        if history.sum() < DIAGNOSTIC_MIN_ROWS:
            windows[str(t)]["status"] = "SKIP_BELOW_1000"
        else:
            windows[str(t)]["status"] = "READY"
    return {
        "ids_test": ids_test, "base_prediction": base_prediction,
        "global_prediction": global_values, "missing_test": missing,
        "base_check": base_check, "global_check": global_check,
        "plan": plan, "rows": rows, "origins": origins, "indices": indices,
        "ids": ids, "cycles": cycles, "prediction": predictions,
        "route_missing": route_missing, "windows": windows,
        "bundles": {67: e67, 75: e75},
    }


def r16_h8_weights(cycles: Any, reference_cycle: int) -> tuple[np.ndarray, dict[str, Any]]:
    cycle_values = np.asarray(cycles, dtype=np.int64)
    if cycle_values.ndim != 1 or len(cycle_values) == 0:
        raise GateError("Weight cycle vector must be a nonempty 1-D array")
    ages = int(reference_cycle) - cycle_values
    if np.any(ages < 0):
        raise GateError("Recency weight cycle is after its reference cycle")
    raw = np.exp2(-ages.astype(np.float64) / float(HALF_LIFE))
    raw_mean = float(raw.mean())
    weights = np.asarray(raw / raw_mean, dtype=np.float64)
    if (not np.isfinite(weights).all() or np.any(weights <= 0.0)
            or not np.isclose(float(weights.mean()), 1.0, atol=1e-14, rtol=0.0)):
        raise GateError("Mean-normalized H8 weights failed finite/positive/mean-one checks")
    return weights, {
        "formula": "raw_i=2**(-(reference_cycle-cycle_i)/8); normalized_i=raw_i/mean(raw)",
        "reference_cycle": int(reference_cycle), "half_life_cycles": 8,
        "normalization": "divide raw weights by arithmetic mean over these exact fit rows",
        "raw_weight_mean": raw_mean, "normalized_mean": float(weights.mean()),
        "min": float(weights.min()), "max": float(weights.max()),
        "weights_sha256": sha_f64(weights), "fit_rows": len(weights),
    }


def solve_weighted_nonnegative_l1(target: Any, prediction: Any, weights: Any,
                                  *, threads: int = THREADS) -> dict[str, Any]:
    y = np.asarray(target, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if y.ndim != 1 or p.ndim != 1 or w.ndim != 1 or not (len(y) == len(p) == len(w)) or len(y) == 0:
        raise GateError("Weighted LP y/p/weight vectors must be nonempty, one-dimensional, and aligned")
    if (not np.isfinite(y).all() or not np.isfinite(p).all() or not np.isfinite(w).all()
            or np.any((y < 0.0) | (y > 100.0)) or np.any((p < 0.0) | (p > 100.0))
            or np.any(w <= 0.0)):
        raise GateError("Weighted LP inputs violate finite/range/positive-weight constraints")
    n = len(y)
    pos = np.arange(n, dtype=np.int64)
    row = np.concatenate((pos, pos, pos, n + pos, n + pos, n + pos))
    col = np.concatenate((np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64), 2 + pos,
                          np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64), 2 + pos))
    data = np.concatenate((-np.ones(n), -p, -np.ones(n), np.ones(n), p, -np.ones(n)))
    constraints = sparse.coo_matrix((data, (row, col)), shape=(2 * n, n + 2), dtype=np.float64).tocsr()
    rhs = np.concatenate((-y, y))
    objective = np.concatenate((np.zeros(2, dtype=np.float64), w))
    bounds = [(None, None), (0.0, None), *((0.0, None) for _ in range(n))]
    started = time.perf_counter()
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=OptimizeWarning,
                                    message="Unrecognized options detected:.*threads")
            result = linprog(objective, A_ub=constraints, b_ub=rhs, bounds=bounds,
                             method="highs", options={"threads": int(threads)})
    except Exception as exc:
        return {"success": False, "fallback": True, "solver_status": None,
                "solver_message": f"{type(exc).__name__}: {exc}", "intercept": None,
                "slope": None, "weighted_l1_sum": None, "solver_seconds": time.perf_counter() - started,
                "history_rows": n, "constraint_shape": [2 * n, n + 2],
                "constraint_nnz": int(constraints.nnz), "threads": int(threads)}
    elapsed = float(time.perf_counter() - started)
    if not result.success:
        return {"success": False, "fallback": True, "solver_status": int(result.status),
                "solver_message": str(result.message), "intercept": None, "slope": None,
                "weighted_l1_sum": None, "solver_seconds": elapsed, "history_rows": n,
                "constraint_shape": [2 * n, n + 2], "constraint_nnz": int(constraints.nnz),
                "threads": int(threads)}
    solution = np.asarray(result.x, dtype=np.float64)
    if solution.shape != (n + 2,) or not np.isfinite(solution).all() or solution[1] < -1e-9:
        return {"success": False, "fallback": True, "solver_status": int(result.status),
                "solver_message": "nonfinite or negative-slope solution", "intercept": None,
                "slope": None, "weighted_l1_sum": None, "solver_seconds": elapsed,
                "history_rows": n, "constraint_shape": [2 * n, n + 2],
                "constraint_nnz": int(constraints.nnz), "threads": int(threads)}
    intercept, slope = float(solution[0]), max(0.0, float(solution[1]))
    weighted_loss = float(np.dot(w, np.abs(y - intercept - slope * p)))
    lp_loss = float(result.fun)
    matches = np.isclose(weighted_loss, lp_loss,
                         atol=max(1e-5, float(w.sum()) * 1e-9), rtol=1e-8)
    return {"success": bool(matches), "fallback": not bool(matches),
            "solver_status": int(result.status), "solver_message": str(result.message),
            "intercept": intercept if matches else None, "slope": slope if matches else None,
            "weighted_l1_sum": weighted_loss if matches else lp_loss,
            "solver_objective": lp_loss, "solver_seconds": elapsed,
            "history_rows": n, "constraint_shape": [2 * n, n + 2],
            "constraint_nnz": int(constraints.nnz), "threads": int(threads),
            "objective_matches_recomputed": bool(matches)}


def append_csv_once(path: Path, ids: np.ndarray, predictions: np.ndarray) -> str:
    if len(ids) != len(predictions):
        raise GateError("Output IDs and values are misaligned")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["record_id", "threat_rank"])
        for record_id, value in zip(ids, predictions, strict=True):
            writer.writerow([str(record_id), format(float(value), ".17g")])
        stream.flush()
        os.fsync(stream.fileno())
    return sha_file(path)


def reserve_lp_start(run_dir: Path, label: str) -> int:
    events = []
    if LEDGER.is_file():
        events = [json.loads(line) for line in LEDGER.read_text(encoding="utf-8").splitlines() if line.strip()]
    started = [event for event in events if event.get("event") == "LP_STARTED"]
    if len(started) >= MAX_LP_STARTS:
        raise GateError("Frozen five-start LP budget exhausted")
    expected = LP_PLAN[len(started)]
    if label != expected:
        raise GateError(f"Next permitted LP is {expected}; refusing {label}")
    append_event(LEDGER, {"event": "LP_STARTED", "index": len(started) + 1,
                          "label": label, "time_utc": datetime.now(timezone.utc).isoformat(),
                          "candidate_lp_budget": MAX_LP_STARTS, "threads": THREADS,
                          "attempt": 1, "retry_allowed": False})
    return len(started) + 1


def runtime_gates() -> tuple[dict[str, Any], dict[str, Any]]:
    resource = v2.resource_gate()
    process = v2.one_job_gate()
    if resource.get("status") != "PASS" or process.get("status") != "PASS":
        raise GateError("Resource/process gate returned non-PASS")
    return resource, process


def write_blocker(stage: str, reason: str, *, targets_read: bool, lp_started: int) -> None:
    if BLOCKER.exists():
        return
    write_once(BLOCKER, {
        "schema_version": 1, "status": "BLOCKED_STOP_NO_RETRY",
        "stage": stage, "reason": reason,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_targets_read": bool(targets_read),
        "candidate_lp_starts": int(lp_started),
        "lp_start_budget": MAX_LP_STARTS,
        "retry_or_polling_performed": False,
    })


def run_lp(label: str, y: np.ndarray, p: np.ndarray,
           weights: np.ndarray | None, *, run_dir: Path) -> dict[str, Any]:
    resource, process = runtime_gates()
    index = reserve_lp_start(run_dir, label)
    started_at = datetime.now(timezone.utc).isoformat()
    with threadpool_limits(limits=THREADS):
        result = (solve_nonnegative_l1(y, p, threads=THREADS) if weights is None
                  else solve_weighted_nonnegative_l1(y, p, weights, threads=THREADS))
    result = json_safe(result)
    state = {
        "schema_version": 1, "label": label, "lp_index": index,
        "started_utc": started_at, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "success": bool(result.get("success")), "result": result,
        "resource_gate": resource, "one_job_gate": process,
        "candidate_lp_started": True, "retry_allowed": False,
        "fit_rows": len(y), "target_sha256": sha_array(y),
        "prediction_sha256": sha_array(p),
        "weights_sha256": sha_f64(weights) if weights is not None else None,
    }
    state_path = run_dir / f"LP_{index:02d}_{label}.json"
    state_hash = write_once(state_path, state)
    append_event(LEDGER, {"event": "LP_COMPLETE" if state["success"] else "LP_FAILED",
                          "index": index, "label": label, "state_path": str(state_path),
                          "state_sha256": state_hash,
                          "time_utc": datetime.now(timezone.utc).isoformat(),
                          "retry_allowed": False})
    state["state_path"] = str(state_path)
    state["state_sha256"] = state_hash
    return state


def fit_state_from_lp(state: dict[str, Any]) -> dict[str, Any] | None:
    if not state.get("success"):
        return None
    result = state["result"]
    return {"intercept": float(result["intercept"]), "slope": float(result["slope"]),
            "objective": result.get("objective_l1_sum", result.get("weighted_l1_sum")),
            "fit_rows": int(result["history_rows"]), "lp_state_sha256": state["state_sha256"]}


def metric_group(y: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    count = int(mask.sum())
    if count == 0:
        return {"n": 0, "mae": None, "mean_error": None, "p90_absolute_error": None,
                "target_min": None, "target_max": None,
                "prediction_at_0_count": 0, "prediction_at_100_count": 0}
    error = pred[mask] - y[mask]
    absolute = np.abs(error)
    return {"n": count, "mae": float(absolute.mean()),
            "absolute_error_sum": float(absolute.sum()),
            "mean_error": float(error.mean()),
            "p90_absolute_error": float(np.quantile(absolute, 0.90)),
            "target_min": float(y[mask].min()), "target_max": float(y[mask].max()),
            "prediction_at_0_count": int(np.sum(pred[mask] == 0.0)),
            "prediction_at_100_count": int(np.sum(pred[mask] == 100.0))}


def diagnostic_metrics(y: np.ndarray, pred: np.ndarray, route_missing: np.ndarray,
                       cycles: np.ndarray) -> dict[str, Any]:
    groups = {
        "all": np.ones(len(y), dtype=bool),
        "aux_route_present": ~route_missing,
        "aux_route_missing_fallback": route_missing,
        "target_0_to_10_inclusive": (y >= 0.0) & (y <= 10.0),
        "target_90_to_100_inclusive": (y >= 90.0) & (y <= 100.0),
    }
    metrics = {name: metric_group(y, pred, mask) for name, mask in groups.items()}
    metrics["by_cycle"] = {
        str(int(cycle)): metric_group(y, pred, cycles == cycle)
        for cycle in np.unique(cycles)
    }
    return metrics


def run_diagnostics(data: dict[str, Any], labels: np.ndarray,
                    run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cycles = data["cycles"]
    base_p = data["prediction"]
    missing = data["route_missing"]
    diagnostic_records: list[dict[str, Any]] = []
    lp_states: list[dict[str, Any]] = []
    for t, lo, hi in WINDOWS:
        history = (cycles >= 68) & (cycles <= t)
        score = (cycles >= lo) & (cycles <= hi)
        n_history, n_score = int(history.sum()), int(score.sum())
        if n_history < DIAGNOSTIC_MIN_ROWS:
            diagnostic_records.append({"origin": t, "status": "SKIPPED_BELOW_1000",
                                       "fit_rows": n_history, "score_rows": n_score})
            continue
        if not n_score or np.any(history & score):
            raise GateError(f"Diagnostic t={t} fit/score support is empty or overlapping")
        y_history, p_history = labels[history], base_p[history]
        y_score, p_score = labels[score], base_p[score]
        cycles_history, cycles_score = cycles[history], cycles[score]
        weights, weight_info = r16_h8_weights(cycles_history, t)
        global_state = run_lp(f"diag_t{t}_global", y_history, p_history, None, run_dir=run_dir)
        lp_states.append(global_state)
        recency_state = run_lp(f"diag_t{t}_recency_h8", y_history, p_history, weights, run_dir=run_dir)
        lp_states.append(recency_state)
        predictions: dict[str, np.ndarray] = {"uncalibrated_direct": p_score.copy()}
        if global_state["success"]:
            a_g, b_g = float(global_state["result"]["intercept"]), float(global_state["result"]["slope"])
            predictions["in_prefix_global_affine"] = apply_affine(p_score, a_g, b_g)
        else:
            predictions["in_prefix_global_affine"] = None
        if recency_state["success"]:
            a_w, b_w = float(recency_state["result"]["intercept"]), float(recency_state["result"]["slope"])
            predictions["in_prefix_recency_h8_affine"] = apply_affine(p_score, a_w, b_w)
        else:
            predictions["in_prefix_recency_h8_affine"] = None
        posthoc = apply_affine(p_score, PRIOR_GLOBAL_A, PRIOR_GLOBAL_B)
        posthoc[missing[score]] = p_score[missing[score]]
        predictions["existing_global_present_only_posthoc"] = posthoc
        record = {
            "origin": int(t), "score_cycles": [int(lo), int(hi)],
            "status": "COMPLETE" if global_state["success"] and recency_state["success"] else "PARTIAL_SOLVER_FAILURE",
            "fit_rows": n_history, "score_rows": n_score,
            "small_diagnostic_history": n_history < 5000,
            "diagnostic_minimum_rows": DIAGNOSTIC_MIN_ROWS,
            "fit_cycle_counts": {str(int(c)): int(np.sum(cycles_history == c)) for c in np.unique(cycles_history)},
            "score_cycle_counts": {str(int(c)): int(np.sum(cycles_score == c)) for c in np.unique(cycles_score)},
            "fit_row_sha256": sha_array(data["rows"][history]),
            "score_row_sha256": sha_array(data["rows"][score]),
            "score_id_sha256": sha_array(data["ids"][score]),
            "score_target_sha256": sha_array(y_score),
            "score_route_missing_n": int(missing[score].sum()),
            "score_route_present_n": int((~missing[score]).sum()),
            "recency_weights": weight_info,
            "lp_state_sha256": {"global": global_state["state_sha256"],
                                "recency_h8": recency_state["state_sha256"]},
            "metrics": {},
            "existing_global_present_only_caveat": (
                "Retrospective public comparator: its fixed coefficients were fitted on all 20,903 cycles 68-79 labels, "
                "including these diagnostic score windows; descriptive only, not a causal or held-out validation estimate."
            ),
        }
        for name, pred in predictions.items():
            record["metrics"][name] = (None if pred is None else
                diagnostic_metrics(y_score, pred, missing[score], cycles_score))
        diagnostic_records.append(record)
    report = {
        "schema_version": 1, "namespace": VERSION,
        "status": "TEMPORAL_DIAGNOSTICS_COMPLETE" if all(
            item.get("status") == "COMPLETE" for item in diagnostic_records
        ) else "TEMPORAL_DIAGNOSTICS_PARTIAL_OR_SKIPPED",
        "windows": diagnostic_records,
        "fit_objectives_are_training_objectives_not_validation_metrics": True,
        "all_scores_use_next_four_already_scored_cycles_only": True,
        "diagnostic_labels_reused_from_fixed_cycles_68_79": True,
        "new_model_or_preprocessing_fits": 0,
        "candidate_lp_budget_total": MAX_LP_STARTS,
        "diagnostic_lp_starts": sum(1 for state in lp_states),
        "lp_state_hashes": [state["state_sha256"] for state in lp_states],
    }
    return report, lp_states


def make_candidate_b(data: dict[str, Any]) -> np.ndarray:
    output = data["global_prediction"].copy()
    output[data["missing_test"]] = data["base_prediction"][data["missing_test"]]
    if (len(output) != EXPECTED_BASE_ROWS or not np.isfinite(output).all()
            or np.any((output < 0.0) | (output > 100.0))
            or not np.array_equal(output[data["missing_test"]],
                                  data["base_prediction"][data["missing_test"]])):
        raise GateError("Candidate B present-only/global plus exact fallback policy failed")
    return output


def csv_digest(ids: np.ndarray, predictions: np.ndarray) -> str:
    import io
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["record_id", "threat_rank"])
    for record_id, value in zip(ids, predictions, strict=True):
        writer.writerow([str(record_id), format(float(value), ".17g")])
    return hashlib.sha256(buffer.getvalue().encode("utf-8")).hexdigest()


def write_candidate(candidate: str, path: Path, ids: np.ndarray,
                    values: np.ndarray, recipe: dict[str, Any],
                    support: dict[str, Any]) -> dict[str, Any]:
    manifest_path = MANIFEST_A if candidate == "A" else MANIFEST_B
    qa_path = QA_A if candidate == "A" else QA_B
    csv_hash = append_csv_once(path, ids, values)
    frame = pd.read_csv(path, dtype={"record_id": str}, float_precision="round_trip")
    actual_ids = frame["record_id"].astype(str).to_numpy(dtype=str)
    actual_values = pd.to_numeric(frame["threat_rank"], errors="raise").to_numpy(dtype=np.float64)
    if (len(frame) != EXPECTED_BASE_ROWS or not np.array_equal(actual_ids, ids)
            or not np.array_equal(actual_values, values)
            or not np.isfinite(actual_values).all()
            or np.any((actual_values < 0.0) | (actual_values > 100.0))):
        raise GateError(f"Candidate {candidate} CSV failed bit-exact row/order/range QA")
    body = {
        "schema_version": 1, "namespace": VERSION,
        "candidate": recipe["identity"], "status": "READY_LOCAL_EXPLORATORY",
        "csv_path": str(path), "csv_sha256": csv_hash,
        "rows": len(frame), "columns": list(frame.columns),
        "ordered_ids_sha256": v2.sha_ordered_ids(actual_ids),
        "prediction_sha256": sha_array(actual_values),
        "prediction_float64_sha256": sha_f64(actual_values),
        "range": [float(actual_values.min()), float(actual_values.max())],
        "formula_replay_bit_exact": True,
        "fit_support": support,
        "recipe": recipe,
        "test_features_read": False, "test_labels_read": False,
        "upload_performed": False,
        "diagnostic_interpretation": "Already-scored temporal views, not a fresh holdout; no private/generalization claim.",
    }
    manifest = {**body, "signature": canonical_sha(body),
                "signature_method": "SHA-256 canonical JSON integrity seal"}
    manifest_hash = write_once(manifest_path, manifest)
    qa_body = {"status": "PASS_INDEPENDENT_BIT_EXACT_CSV_QA",
               "candidate": recipe["identity"], "csv_sha256": csv_hash,
               "rows": len(frame), "ordered_ids_sha256": body["ordered_ids_sha256"],
               "formula_replay_bit_exact": True, "finite_in_range": True,
               "test_features_read": False, "test_labels_read": False,
               "upload_performed": False}
    qa = {**qa_body, "signature": canonical_sha(qa_body),
          "signature_method": "SHA-256 canonical JSON integrity seal"}
    qa_hash = write_once(qa_path, qa)
    return {"candidate": recipe["identity"], "csv_sha256": csv_hash,
            "manifest_sha256": manifest_hash, "qa_sha256": qa_hash,
            "path": str(path), "rows": len(frame),
            "identical_to_existing_global_affine": csv_hash == sha_file(v2.OUT_CSV),
            "identical_to_original_direct_base": csv_hash == sha_file(v2.BASE_CSV)}


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()


def prepare_freeze() -> None:
    if FREEZE.exists() or REVIEW.exists() or RUN.exists():
        raise GateError("Follow-up freeze/review/run already exists; refusing overwrite or retry")
    if any(path.exists() for path in (OUT_A, OUT_B, DIAG_REPORT, MANIFEST_A, MANIFEST_B, QA_A, QA_B)):
        raise GateError("Follow-up output path already exists; refusing overwrite")
    if not SYNTH_QA.is_file() or read_json(SYNTH_QA).get("status") != "PASS_SYNTHETIC_ONLY":
        raise GateError("Synthetic QA PASS receipt is required before freeze")
    data = load_fixed_support()
    inputs = snapshot_input_hashes()
    source = source_hashes()
    body = {
        "schema_version": 1, "namespace": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "worktree": str(ROOT), "branch": "codex/reg-position-failure-mechanism-v1",
        "base_head": git("rev-parse", "HEAD"),
        "scientific_status": "EXPLORATORY_TEMPORAL_REUSE_AND_AFFINE_CALIBRATION",
        "candidate_A": {
            "identity": "RECENCY_AFFINE_H8",
            "support": "Exactly the fixed 20,903 causal DIRECT OOF rows: E67 cycles 68-75 and E75 cycles 76-79.",
            "weights": "raw_i=2**(-(79-cycle_i)/8); divide by arithmetic mean over those exact full fit rows once.",
            "weight_application": "LP slack objective only: sum_i w_i*abs(y_i-a-b*p_i); no weighting of features, predictions, or targets.",
            "constraints": "free intercept; slope b>=0; R17 sparse absolute-loss LP math; one full-support production LP.",
            "output": "clip(a+b*p,0,100) applied to every row of original DIRECT_B_E79 CSV, including 1,062 missing-route fallback rows.",
            "half_life_rationale": "R16 source uses the fixed age formula 2**(-(cutoff-cycle)/8); H8 is inherited as a mechanism, not tuned here.",
            "minimum_production_rows": PRODUCTION_MIN_ROWS,
            "threshold_search": False,
        },
        "candidate_B": {
            "identity": "GLOBAL_AFFINE_PRESENT_ONLY",
            "new_fits": 0, "new_lp_calls": 0,
            "coefficients": {"intercept": PRIOR_GLOBAL_A, "slope": PRIOR_GLOBAL_B,
                             "source_candidate": "CURRENT_DIRECT_AFFINE_L1_E79",
                             "source_ref": PRIOR_GLOBAL_REF, "source_public_mae": PRIOR_GLOBAL_PUBLIC_MAE},
            "present_rows": "Use exact existing global affine CSV values.",
            "missing_rows": "Restore exact original DIRECT_B_E79/R06 fallback values using signed inference_B.npz missing mask and ordered IDs.",
            "hypothesis": "Post-score/public hypothesis; not the previously tested global all-row recipe and not causal evidence.",
            "per_target_routing": False,
        },
        "diagnostics": {
            "origins": [71, 75], "score_horizons": {"71": [72, 75], "75": [76, 79]},
            "at_each_origin": ["fit unweighted global affine on OOF cycles 68..t", "fit H8 weighted affine on OOF cycles 68..t"],
            "comparators": ["uncalibrated DIRECT", "in-prefix global affine", "in-prefix H8 affine", "existing global present-only map (retrospective/posthoc comparator)"],
            "diagnostic_minimum_rows": DIAGNOSTIC_MIN_ROWS,
            "known_counts": data["windows"],
            "metrics": ["MAE and denominator overall", "MAE and support by OOF auxiliary missing/present route", "target tails 0-10 and 90-100", "cycle support", "signed error, p90 absolute error, clipping counts"],
            "interpretation": "Reused already-scored Train views, not a new holdout; report temporal reuse and horizon limits.",
            "test_or_sample_access": False,
        },
        "budgets": {
            "candidate_lp_starts_max": MAX_LP_STARTS,
            "diagnostic_lp_starts_max": 4,
            "production_lp_starts_max": 1,
            "candidate_A_production_lps_max": 1,
            "candidate_B_new_lps": 0,
            "preprocessing_fits": 0,
            "base_or_model_fits": 0,
            "retry_after_lp_started": False,
            "threads": THREADS, "concurrent_project_jobs": 1,
            "minimum_available_ram_bytes": MIN_AVAILABLE,
            "reserved_ram_bytes": RESERVED_RAM,
            "estimated_peak_bytes": ESTIMATED_PEAK,
            "synthetic_solver_calls": read_json(SYNTH_QA).get("synthetic_lp_calls"),
            "synthetic_solver_calls_are_candidate_starts": False,
        },
        "input_hashes": inputs, "source_hashes": source,
        "train": {"path": str(v2.TRAIN), "sha256": EXPECTED_TRAIN_SHA,
                  "read_rule": "After exact Sol/root PASS only; one selected-row parse of the already-scored 20,903 OOF rows.",
                  "maximum_target_rows_parsed": EXPECTED_SUPPORT_ROWS},
        "support": {
            "rows": len(data["rows"]), "unique_rows": len(set(data["rows"].tolist())),
            "unique_ids": len(set(data["ids"].tolist())),
            "rows_sha256": sha_array(data["rows"]), "ids_sha256": sha_array(data["ids"]),
            "cycles_sha256": sha_array(data["cycles"]),
            "direct_prediction_sha256": sha_array(data["prediction"]),
            "cycle_counts": {str(int(c)): int(np.sum(data["cycles"] == c)) for c in np.unique(data["cycles"])},
            "E67_rows_cycles_68_75": 12412, "E75_rows_cycles_76_79": 8491,
            "test_ids_sha256": v2.sha_ordered_ids(data["ids_test"]),
            "test_rows": len(data["ids_test"]), "test_missing_rows": int(data["missing_test"].sum()),
            "prediction_only_preflight": True,
        },
        "review_required": True, "review_path": str(REVIEW),
        "output_paths": {"candidate_A": str(OUT_A), "candidate_B": str(OUT_B),
                         "diagnostics": str(DIAG_REPORT), "manifest_A": str(MANIFEST_A),
                         "manifest_B": str(MANIFEST_B), "qa_A": str(QA_A), "qa_B": str(QA_B),
                         "run_dir": str(RUN)},
    }
    signed = {**body, "signature": canonical_sha(body),
              "signature_method": "SHA-256 canonical JSON integrity seal"}
    digest = write_once(FREEZE, signed)
    print(json.dumps({"status": "PREFIT_FREEZE_CREATED", "freeze_sha256": digest,
                      "signature": signed["signature"], "base_head": body["base_head"],
                      "diagnostic_counts": data["windows"]}, sort_keys=True))


def verify_review_gate() -> tuple[dict[str, Any], dict[str, Any]]:
    freeze = read_json(FREEZE)
    verify_signed(freeze, "signature", "follow-up prefit freeze")
    if (freeze.get("namespace") != VERSION
            or freeze.get("budgets", {}).get("candidate_lp_starts_max") != MAX_LP_STARTS
            or freeze.get("candidate_A", {}).get("identity") != "RECENCY_AFFINE_H8"
            or freeze.get("candidate_B", {}).get("identity") != "GLOBAL_AFFINE_PRESENT_ONLY"):
        raise GateError("Follow-up candidate or LP budget differs from signed freeze")
    if source_hashes() != freeze.get("source_hashes"):
        raise GateError("Script, protocol, synthetic QA, or pinned helper source changed")
    verify_input_hashes(freeze.get("input_hashes", {}))
    if not REVIEW.is_file():
        raise GateError("Independent Sol/root source PASS receipt is missing")
    review = read_json(REVIEW)
    verify_signed(review, "signature", "independent review receipt")
    if (review.get("verdict") != "PASS"
            or review.get("reviewed_commit") != freeze.get("reviewed_commit")
            or review.get("freeze_signature") != freeze["signature"]
            or review.get("reviewed_artifacts") != freeze["source_hashes"]):
        raise GateError("Independent PASS receipt does not bind this exact reviewed source set")
    head = git("rev-parse", "HEAD")
    subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor",
                    str(review["reviewed_commit"]), head], check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if (git("branch", "--show-current") != freeze.get("branch")
            or git("status", "--porcelain")):
        raise GateError("Expected branch or clean committed worktree before fixed follow-up")
    synth = read_json(SYNTH_QA)
    if synth.get("status") != "PASS_SYNTHETIC_ONLY" or synth.get("candidate_data_read") is not False:
        raise GateError("Candidate-blind synthetic QA receipt is not PASS")
    return freeze, review


def validate_frozen_paths_absent() -> None:
    paths = (RUN, OUT_A, OUT_B, DIAG_REPORT, MANIFEST_A, MANIFEST_B, QA_A, QA_B)
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise GateError(f"A prior run/output exists; no LP retry or overwrite allowed: {existing}")


def contract_synthetic_qa() -> dict[str, Any]:
    """Exercise write-once, fixed LP order/cap, no-retry, and digest pins without candidate data."""
    with tempfile.TemporaryDirectory(prefix="reg-followup-contract-") as temp:
        run_dir = Path(temp) / "run"
        run_dir.mkdir()
        ledger_path = run_dir / "lp_ledger.jsonl"
        observed = []
        for label in LP_PLAN:
            events = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()
                      if line.strip()] if ledger_path.exists() else []
            started = sum(event.get("event") == "LP_STARTED" for event in events)
            if started >= MAX_LP_STARTS or label != LP_PLAN[started]:
                raise GateError("Synthetic LP reservation contract accepted an invalid order/cap")
            append_event(ledger_path, {"event": "LP_STARTED", "label": label})
            observed.append(label)
        cap_rejected = False
        try:
            events = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
            started = sum(event.get("event") == "LP_STARTED" for event in events)
            if started >= MAX_LP_STARTS:
                raise GateError("synthetic budget exhausted")
        except GateError:
            cap_rejected = True
        if not cap_rejected or observed != list(LP_PLAN):
            raise GateError("Synthetic fixed-order/cap check failed")
        receipt = run_dir / "immutable.json"
        digest = write_once(receipt, {"value": 1})
        no_overwrite = False
        try:
            write_once(receipt, {"value": 2})
        except FileExistsError:
            no_overwrite = True
        if not no_overwrite or sha_file(receipt) != digest:
            raise GateError("Synthetic O_EXCL/no-overwrite digest contract failed")
        resume_rejected = run_dir.exists()
        pin_mutation_rejected = (hashlib.sha256(b"frozen").hexdigest()
                                 != hashlib.sha256(b"mutated").hexdigest())
        if not resume_rejected or not pin_mutation_rejected:
            raise GateError("Synthetic no-resume or changed-digest check failed")
    return {"fixed_lp_plan": list(LP_PLAN), "max_lp_starts": MAX_LP_STARTS,
            "cap_plus_one_rejected": cap_rejected, "write_once_overwrite_rejected": no_overwrite,
            "write_once_sha256_preserved": digest, "existing_run_dir_blocks_resume": resume_rejected,
            "mutated_digest_differs": pin_mutation_rejected}


def synthetic_qa() -> None:
    if SYNTH_QA.exists():
        report = read_json(SYNTH_QA)
        verify_signed(report, "signature", "synthetic QA")
        if (report.get("status") != "PASS_SYNTHETIC_ONLY"
                or report.get("candidate_data_read") is not False
                or report.get("train_labels_read") is not False
                or report.get("synthetic_solver_calls") != 3
                or report.get("production_lp_calls") != 0
                or report.get("contract_checks", {}).get("cap_plus_one_rejected") is not True
                or report.get("contract_checks", {}).get("write_once_overwrite_rejected") is not True):
            raise GateError("Existing synthetic QA receipt does not satisfy V1 follow-up contract")
        print(json.dumps({"status": report["status"], "receipt_sha256": sha_file(SYNTH_QA),
                          "solver_reexecuted": False}, sort_keys=True))
        return
    if sha_file(R17_SOURCE) != v2.PINS["r17_source"][1] or sha_file(R16_SOURCE) != v2.PINS["r16_model_source"][1]:
        raise GateError("Pinned R17 solver or R16 H8 rationale source changed")
    p = np.asarray([2, 7, 11, 19, 23, 31, 40, 44, 51, 63, 72, 85], dtype=np.float64)
    y_exact = 4.25 + 0.83 * p
    w_test = np.linspace(0.4, 1.8, len(p), dtype=np.float64)
    w_test /= w_test.mean()
    known = solve_weighted_nonnegative_l1(y_exact, p, w_test, threads=THREADS)
    if (not known.get("success") or abs(known["intercept"] - 4.25) > 1e-7
            or abs(known["slope"] - 0.83) > 1e-7):
        raise GateError("Synthetic weighted LP failed known-coefficient recovery")
    noisy_y = y_exact + np.asarray([0.1, -0.3, 0.4, -0.2, 0.0, 0.25, -0.1, 0.15, -0.35, 0.2, -0.05, 0.3])
    weighted_ones = solve_weighted_nonnegative_l1(noisy_y, p, np.ones(len(p)), threads=THREADS)
    ordinary = solve_nonnegative_l1(noisy_y, p, threads=THREADS)
    if (not weighted_ones.get("success") or not ordinary.get("success")
            or abs(weighted_ones["intercept"] - ordinary["intercept"]) > 1e-7
            or abs(weighted_ones["slope"] - ordinary["slope"]) > 1e-7
            or abs(weighted_ones["weighted_l1_sum"] - ordinary["objective_l1_sum"]) > 1e-7):
        raise GateError("Unit-weight LP does not match pinned R17 unweighted objective/coefficients")
    contract = contract_synthetic_qa()
    report_body = {
        "schema_version": 1, "namespace": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_SYNTHETIC_ONLY", "candidate_data_read": False,
        "train_labels_read": False, "test_or_sample_accessed": False,
        "production_lp_calls": 0, "candidate_fits": 0, "preprocessing_fits": 0,
        "synthetic_solver_calls": 3,
        "weighted_known_coefficients": {"expected_intercept": 4.25, "expected_slope": 0.83,
            "actual_intercept": known["intercept"], "actual_slope": known["slope"],
            "objective": known["weighted_l1_sum"]},
        "unit_weight_matches_r17": {"intercept_delta": abs(weighted_ones["intercept"] - ordinary["intercept"]),
            "slope_delta": abs(weighted_ones["slope"] - ordinary["slope"]),
            "objective_delta": abs(weighted_ones["weighted_l1_sum"] - ordinary["objective_l1_sum"])},
        "synthetic_weight_mean_one": float(w_test.mean()),
        "contract_checks": contract,
        "solver_source_sha256": sha_file(R17_SOURCE),
        "r16_h8_rationale_source_sha256": sha_file(R16_SOURCE),
    }
    report = {**report_body, "signature": canonical_sha(report_body),
              "signature_method": "SHA-256 canonical JSON integrity seal"}
    digest = write_once(SYNTH_QA, report)
    print(json.dumps({"status": report["status"], "sha256": digest,
                      "synthetic_solver_calls": 3,
                      "known_coefficients": [known["intercept"], known["slope"]]}, sort_keys=True))


def finalize_candidate_b(data: dict[str, Any], diag_report: dict[str, Any],
                         label_audit: dict[str, Any], *, commit_lp_count: int) -> dict[str, Any]:
    output = make_candidate_b(data)
    support = {
        "new_fit_rows": 0, "new_lp_calls": 0,
        "source_global_ref": PRIOR_GLOBAL_REF,
        "source_global_csv_sha256": sha_file(v2.OUT_CSV),
        "restored_missing_rows": int(data["missing_test"].sum()),
        "restored_values_match_original_direct_csv": True,
        "test_ids_sha256": v2.sha_ordered_ids(data["ids_test"]),
        "target_read_audit": label_audit,
        "diagnostics_sha256": sha_file(DIAG_REPORT),
        "lp_starts_completed_before_output": commit_lp_count,
    }
    recipe = {
        "identity": "GLOBAL_AFFINE_PRESENT_ONLY",
        "policy": "existing global affine CSV values on present-route rows; exact original DIRECT_B/R06 fallback on signed missing-mask rows",
        "intercept": PRIOR_GLOBAL_A, "slope": PRIOR_GLOBAL_B,
        "mask_source": "inference_B.npz missing array, aligned to signed exact official test IDs",
        "posthoc_public_hypothesis": True,
        "public_score_used_to_choose_recipe": False,
        "test_features_read": False,
    }
    result = write_candidate("B", OUT_B, data["ids_test"], output, recipe, support)
    return result


def run_once() -> None:
    freeze, review = verify_review_gate()
    validate_frozen_paths_absent()
    if (freeze.get("input_hashes") != snapshot_input_hashes()
            or freeze.get("source_hashes") != source_hashes()):
        raise GateError("Signed recipe inputs/sources no longer match freeze")
    data = load_fixed_support()
    if (sha_array(data["rows"]) != freeze["support"]["rows_sha256"]
            or sha_array(data["ids"]) != freeze["support"]["ids_sha256"]
            or sha_array(data["cycles"]) != freeze["support"]["cycles_sha256"]
            or sha_array(data["prediction"]) != freeze["support"]["direct_prediction_sha256"]):
        raise GateError("Prediction-only OOF support differs from frozen recipe")
    RUN.mkdir(parents=True, exist_ok=False)
    # Fresh pre-label gate, then a fresh memory/process gate immediately before every LP start.
    try:
        prelabel_resource, prelabel_job = runtime_gates()
    except Exception as exc:
        write_blocker("PRE_LABEL_RUNTIME_GATE", f"{type(exc).__name__}: {exc}",
                      targets_read=False, lp_started=0)
        print(json.dumps({"status": "BLOCKED_STOP_NO_RETRY", "stage": "PRE_LABEL_RUNTIME_GATE",
                          "reason": str(exc), "candidate_targets_read": False, "lp_starts": 0}, sort_keys=True))
        return
    rows, ids, cycles = data["rows"], data["ids"], data["cycles"]
    try:
        labels, _text_load, parsed_rows, audit = read_scored_query(
            v2.SOURCE, rows, ids, cycles, expected_total_rows=50608)
    except Exception as exc:
        write_blocker("SELECTED_LABEL_INTEGRITY", f"{type(exc).__name__}: {exc}",
                      targets_read=True, lp_started=0)
        raise
    if (not np.array_equal(parsed_rows, rows) or len(labels) != EXPECTED_SUPPORT_ROWS
            or audit.get("fit_target_rows_parsed") != 0
            or audit.get("sha256") != EXPECTED_TRAIN_SHA):
        write_blocker("SELECTED_LABEL_ALIGNMENT", "Selected reader output/hash/support mismatch",
                      targets_read=True, lp_started=0)
        raise GateError("Selected Train labels failed frozen support alignment")
    if not np.isfinite(labels).all() or np.any((labels < 0.0) | (labels > 100.0)):
        write_blocker("SELECTED_LABEL_RANGE", "Selected targets are nonfinite or outside [0,100]",
                      targets_read=True, lp_started=0)
        raise GateError("Selected Train target range failed")

    lp_states: list[dict[str, Any]] = []
    try:
        diag_report, diag_states = run_diagnostics(data, labels, RUN)
        lp_states.extend(diag_states)
    except Exception as exc:
        started = sum(1 for line in LEDGER.read_text(encoding="utf-8").splitlines()
                      if json.loads(line).get("event") == "LP_STARTED") if LEDGER.exists() else 0
        write_blocker("DIAGNOSTIC_OR_RUNTIME_GATE", f"{type(exc).__name__}: {exc}",
                      targets_read=True, lp_started=started)
        print(json.dumps({"status": "BLOCKED_STOP_NO_RETRY", "stage": "DIAGNOSTIC_OR_RUNTIME_GATE",
                          "reason": str(exc), "candidate_targets_read": True,
                          "lp_starts": started, "retry_allowed": False}, sort_keys=True))
        return
    diag_report["train_target_audit"] = audit
    diag_report["support_sha256"] = {"rows": sha_array(rows), "ids": sha_array(ids),
                                    "cycles": sha_array(cycles), "targets": sha_array(labels),
                                    "direct_prediction": sha_array(data["prediction"])}
    diag_report["prelabel_runtime_gates"] = {"resource": prelabel_resource, "one_job": prelabel_job}
    diag_report["source_hashes"] = freeze["source_hashes"]
    diag_hash = write_once(DIAG_REPORT, diag_report)

    # Candidate B uses no new fit; persist it after diagnostics and their LP states.
    try:
        b_ready = finalize_candidate_b(data, diag_report, audit,
                                       commit_lp_count=len(lp_states))
    except Exception as exc:
        write_blocker("CANDIDATE_B_INTEGRITY", f"{type(exc).__name__}: {exc}",
                      targets_read=True, lp_started=len(lp_states))
        raise

    support_mask = (cycles >= 68) & (cycles <= 79)
    if int(support_mask.sum()) < PRODUCTION_MIN_ROWS:
        write_blocker("PRODUCTION_MIN_SUPPORT", "Frozen full OOF support below 5,000 rows",
                      targets_read=True, lp_started=len(lp_states))
        raise GateError("Production weighted LP support below the frozen minimum")
    production_weights, weight_info = r16_h8_weights(cycles[support_mask], 79)
    production_state: dict[str, Any] | None = None
    try:
        production_state = run_lp("production_recency_h8_full_oof",
                                  labels[support_mask], data["prediction"][support_mask],
                                  production_weights, run_dir=RUN)
    except Exception as exc:
        started = sum(1 for line in LEDGER.read_text(encoding="utf-8").splitlines()
                      if json.loads(line).get("event") == "LP_STARTED") if LEDGER.exists() else 0
        write_blocker("PRODUCTION_RUNTIME_GATE", f"{type(exc).__name__}: {exc}",
                      targets_read=True, lp_started=started)
        print(json.dumps({"status": "BLOCKED_STOP_NO_RETRY", "stage": "PRODUCTION_RUNTIME_GATE",
                          "reason": str(exc), "candidate_targets_read": True,
                          "candidate_B": b_ready, "diagnostics_sha256": diag_hash,
                          "lp_starts": started, "retry_allowed": False}, sort_keys=True))
        return
    lp_states.append(production_state)
    if production_state["success"]:
        fit_state = fit_state_from_lp(production_state)
        weighted_pred = apply_affine(data["base_prediction"], fit_state["intercept"], fit_state["slope"])
        fit_receipt = {
            "schema_version": 1, "candidate": "RECENCY_AFFINE_H8",
            "status": "FIT_STATE_PERSISTED", "fit_state": fit_state,
            "weights": weight_info,
            "oof_support_rows": EXPECTED_SUPPORT_ROWS,
            "target_sha256": audit["query_target_values_sha256"],
            "candidate_lp_starts": len(lp_states), "preprocessing_fits": 0,
            "base_model_fits": 0, "test_features_read": False,
            "test_labels_read": False, "retry_allowed": False,
        }
        fit_receipt_hash = write_once(RUN / "PRODUCTION_FIT_RECEIPT.json", fit_receipt)
        recipe_a = {
            "identity": "RECENCY_AFFINE_H8", "intercept": fit_state["intercept"],
            "slope": fit_state["slope"], "weights": weight_info,
            "objective": "sum_i w_i*abs(y_i-a-b*p_i), free intercept, b>=0",
            "weights_applied_to": "LP objective slack costs only",
            "support_cycles": [68, 79], "support_rows": EXPECTED_SUPPORT_ROWS,
            "former_missing_fallback_rows_transformed": int(data["missing_test"].sum()),
            "fit_receipt_sha256": fit_receipt_hash,
        }
        support_a = {
            "rows": EXPECTED_SUPPORT_ROWS, "unique_rows": EXPECTED_SUPPORT_ROWS,
            "unique_ids": EXPECTED_SUPPORT_ROWS,
            "rows_sha256": sha_array(rows), "ids_sha256": sha_array(ids),
            "cycles_sha256": sha_array(cycles), "targets_sha256": sha_array(labels),
            "direct_prediction_sha256": sha_array(data["prediction"]),
            "selected_label_reader_audit": audit,
            "fit_state_sha256": fit_receipt_hash,
            "diagnostics_sha256": diag_hash,
            "lp_starts_completed": len(lp_states),
        }
        a_ready = write_candidate("A", OUT_A, data["ids_test"], weighted_pred,
                                  recipe_a, support_a)
    else:
        a_ready = {"candidate": "RECENCY_AFFINE_H8", "status": "FAILED_NO_CSV",
                   "lp_state_sha256": production_state["state_sha256"],
                   "retry_allowed": False}
    final = {
        "schema_version": 1, "namespace": VERSION,
        "status": "FOLLOWUP_COMPLETE_WITH_READY_CANDIDATES",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_signature": freeze["signature"],
        "reviewed_commit": review["reviewed_commit"],
        "diagnostics_sha256": diag_hash,
        "candidate_A": a_ready, "candidate_B": b_ready,
        "lp_starts": sum(1 for line in LEDGER.read_text(encoding="utf-8").splitlines()
                          if json.loads(line).get("event") == "LP_STARTED"),
        "lp_start_cap": MAX_LP_STARTS, "no_retry": True,
        "candidate_lp_count": sum(1 for line in LEDGER.read_text(encoding="utf-8").splitlines()
                                  if json.loads(line).get("event") == "LP_STARTED"),
        "preprocessing_fits": 0, "base_or_model_fits": 0,
        "test_or_sample_accessed": False, "upload_performed": False,
        "train_target_rows_parsed": len(labels), "train_hash": audit["sha256"],
        "source_hashes": freeze["source_hashes"],
    }
    final_hash = write_once(RUN_RECEIPT, final)
    print(json.dumps({"status": final["status"], "run_receipt_sha256": final_hash,
                      "diagnostics_sha256": diag_hash, "candidate_A": a_ready,
                      "candidate_B": b_ready, "lp_starts": final["lp_starts"],
                      "lp_cap": MAX_LP_STARTS}, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--synthetic-qa", action="store_true")
    group.add_argument("--prepare-freeze", action="store_true")
    group.add_argument("--run", action="store_true")
    args = parser.parse_args()
    try:
        if args.synthetic_qa:
            synthetic_qa()
        elif args.prepare_freeze:
            prepare_freeze()
        else:
            run_once()
        return 0
    except Exception as exc:
        print(f"REG_FOLLOWUP_NO_GO: {type(exc).__name__}: {exc}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
