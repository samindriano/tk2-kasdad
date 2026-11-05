"""Fixed-knot p50 monotone hinge calibration for the bounded REG lane.

This module contains pure calibration and identity helpers. It does not load
course arrays, labels, Test.csv, or Kaggle artifacts. The identity freeze hashes
frozen inputs by bytes without parsing them. Official LP execution is
intentionally absent until a stage-specific ROOT_SOURCE_PASS is reviewed.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping
import warnings

import numpy as np
from scipy import sparse
from scipy.optimize import OptimizeWarning, linprog

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXPERIMENT_ID = "REG_MONOTONE_HINGE_P50_V1"
KNOT = 50.0
SLOPE_BOUNDS = (0.0, 2.0)
DIAGNOSTIC_MIN_ROWS = 1_000
FINAL_MIN_ROWS = 5_000
MIN_ROWS_EACH_SIDE = 200
SOLVER_THREADS = 2
SOLVER_TIME_LIMIT_SECONDS = 180
FIT_IDS = ("HINGE_D1", "HINGE_D2", "HINGE_FINAL")
SYNTHETIC_QA = ROOT / "reports/optimization_v3/reg_monotone_hinge_p50_v1/SYNTHETIC_QA.json"
IDENTITY_FREEZE = ROOT / "experiments/reg_monotone_hinge_p50_v1/IDENTITY_FREEZE.json"


class HingeError(RuntimeError):
    """Fail-closed calibration, identity, or immutable-output error."""


@dataclass(frozen=True)
class HingeParameters:
    intercept: float
    slope_low: float
    slope_high: float


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tagged_array_sha256(values: Any, *, dtype: str | None = None) -> str:
    """R3 global/OOF schema: dtype and shape tag followed by canonical content."""
    array = np.asarray(values, dtype=dtype)
    digest = hashlib.sha256(canonical_bytes({"dtype": str(array.dtype),
                                             "shape": list(array.shape)}))
    if array.dtype.kind in "USO":
        digest.update(canonical_bytes(array.astype(str).tolist()))
    else:
        digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()


def raw_direct_f64_sha256(values: Any) -> str:
    """Direct producer schema: raw contiguous little-endian float64 bytes only."""
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _aligned_vectors(**arrays: Any) -> dict[str, np.ndarray]:
    result = {name: np.asarray(value) for name, value in arrays.items()}
    if not result:
        raise HingeError("At least one identity vector is required")
    if any(value.ndim != 1 for value in result.values()):
        raise HingeError("Identity vectors must be one-dimensional")
    lengths = {len(value) for value in result.values()}
    if len(lengths) != 1:
        raise HingeError("Identity vectors are not aligned")
    return result


def row_identity_hashes(*, physical_rows: Any, record_ids: Any,
                        prediction_origins: Any, original_bundle_indices: Any,
                        cycles: Any, direct_predictions: Any,
                        missing_mask: Any) -> dict[str, str]:
    values = _aligned_vectors(
        physical_rows=physical_rows, record_ids=record_ids,
        prediction_origins=prediction_origins,
        original_bundle_indices=original_bundle_indices, cycles=cycles,
        direct_predictions=direct_predictions, missing_mask=missing_mask,
    )
    return {
        "physical_rows_sha256": tagged_array_sha256(values["physical_rows"]),
        "record_ids_sha256": tagged_array_sha256(values["record_ids"]),
        "prediction_origins_sha256": tagged_array_sha256(values["prediction_origins"]),
        "original_bundle_indices_sha256": tagged_array_sha256(values["original_bundle_indices"]),
        "cycles_sha256": tagged_array_sha256(values["cycles"]),
        "direct_prediction_sha256": tagged_array_sha256(values["direct_predictions"], dtype="<f8"),
        "missing_mask_sha256": tagged_array_sha256(values["missing_mask"]),
    }


def verify_row_identity(identity: Mapping[str, Any], expected: Mapping[str, str]) -> dict[str, str]:
    actual = row_identity_hashes(**identity)
    if set(actual) != set(expected):
        raise HingeError("Frozen OOF row identity schema differs")
    differences = [name for name, digest in actual.items() if expected.get(name) != digest]
    if differences:
        raise HingeError("Frozen OOF row identity mismatch: " + ", ".join(sorted(differences)))
    return actual


def load_targets_after_identity(identity: Mapping[str, Any], expected: Mapping[str, str],
                                target_loader: Callable[[], Any]) -> np.ndarray:
    """Gate identity before the caller opens or materializes any target values."""
    verify_row_identity(identity, expected)
    return np.asarray(target_loader())


def stage_mask(cycles: Any, prediction_origins: Any, *, stage: str, role: str) -> np.ndarray:
    cycles_array = np.asarray(cycles)
    origins = np.asarray(prediction_origins, dtype=str)
    if cycles_array.ndim != 1 or origins.ndim != 1 or len(cycles_array) != len(origins):
        raise HingeError("Stage cycle/origin vectors must be aligned one-dimensional arrays")
    if stage == "D1" and role == "fit":
        return (origins == "E67") & (cycles_array >= 68) & (cycles_array <= 71)
    if stage == "D1" and role == "score":
        return (origins == "E67") & (cycles_array >= 72) & (cycles_array <= 75)
    if stage == "D2" and role == "fit":
        return (origins == "E67") & (cycles_array >= 68) & (cycles_array <= 75)
    if stage == "D2" and role == "score":
        return (origins == "E75") & (cycles_array >= 76) & (cycles_array <= 79)
    raise HingeError(f"Unsupported causal stage selection: {stage}/{role}")


def _prediction_vector(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array):
        raise HingeError(f"{name} must be a nonempty one-dimensional vector")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0) or np.any(array > 100.0):
        raise HingeError(f"{name} must contain only finite values in [0, 100]")
    return array


def validate_parameters(parameters: HingeParameters | Mapping[str, Any]) -> HingeParameters:
    if isinstance(parameters, HingeParameters):
        result = parameters
    else:
        try:
            result = HingeParameters(float(parameters["intercept"]),
                                     float(parameters["slope_low"]),
                                     float(parameters["slope_high"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise HingeError("Hinge parameters must define intercept, slope_low, slope_high") from exc
    values = (result.intercept, result.slope_low, result.slope_high)
    if not all(math.isfinite(value) for value in values):
        raise HingeError("Hinge parameters must be finite")
    if not (SLOPE_BOUNDS[0] <= result.slope_low <= SLOPE_BOUNDS[1]
            and SLOPE_BOUNDS[0] <= result.slope_high <= SLOPE_BOUNDS[1]):
        raise HingeError("Hinge slopes must remain within [0, 2]")
    return result


def predict_hinge(direct_predictions: Any,
                  parameters: HingeParameters | Mapping[str, Any]) -> np.ndarray:
    p = _prediction_vector(direct_predictions, "direct predictions")
    params = validate_parameters(parameters)
    return (params.intercept + params.slope_low * np.minimum(p, KNOT)
            + params.slope_high * np.maximum(p - KNOT, 0.0))


def solve_fixed_knot_p50_l1(direct_predictions: Any, targets: Any, *,
                            purpose: str) -> dict[str, Any]:
    """One unweighted sparse LP start. No retry, weights, seed, or knot search."""
    p = _prediction_vector(direct_predictions, "direct predictions")
    y = _prediction_vector(targets, "targets")
    if len(p) != len(y):
        raise HingeError("Direct predictions and targets are not aligned")
    if purpose == "diagnostic":
        minimum = DIAGNOSTIC_MIN_ROWS
    elif purpose == "final":
        minimum = FINAL_MIN_ROWS
    else:
        raise HingeError("purpose must be 'diagnostic' or 'final'")
    low_support = int(np.count_nonzero(p < KNOT))
    high_support = int(np.count_nonzero(p >= KNOT))
    if len(p) < minimum or min(low_support, high_support) < MIN_ROWS_EACH_SIDE:
        raise HingeError("Insufficient total or per-side prediction support; LP not started")

    n = len(p)
    low = np.minimum(p, KNOT)
    high = np.maximum(p - KNOT, 0.0)
    # Residual constraints are -g(p)-u <= -y and g(p)-u <= y.
    rows = np.concatenate((np.arange(n), np.arange(n), np.arange(n),
                           n + np.arange(n), n + np.arange(n), n + np.arange(n)))
    cols = np.concatenate((np.zeros(n, dtype=int), np.ones(n, dtype=int),
                           3 + np.arange(n), np.zeros(n, dtype=int),
                           np.ones(n, dtype=int), 3 + np.arange(n)))
    values = np.concatenate((-np.ones(n), -low, -np.ones(n),
                             np.ones(n), low, -np.ones(n)))
    rows = np.concatenate((rows, np.arange(n), n + np.arange(n)))
    cols = np.concatenate((cols, np.full(n, 2, dtype=int), np.full(n, 2, dtype=int)))
    values = np.concatenate((values, -high, high))
    matrix = sparse.coo_matrix((values, (rows, cols)), shape=(2 * n, n + 3)).tocsr()
    objective = np.concatenate((np.zeros(3, dtype=np.float64), np.ones(n, dtype=np.float64)))
    rhs = np.concatenate((-y, y))
    bounds = [(None, None), (0.0, 2.0), (0.0, 2.0)] + [(0.0, None)] * n
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        result = linprog(objective, A_ub=matrix, b_ub=rhs, bounds=bounds,
                         method="highs", options={"threads": SOLVER_THREADS,
                                                   "time_limit": SOLVER_TIME_LIMIT_SECONDS})
    if not result.success or result.x is None or not np.all(np.isfinite(result.x)):
        raise HingeError(f"HiGHS LP did not complete successfully: {result.message}")
    params = validate_parameters(HingeParameters(float(result.x[0]),
                                                 float(result.x[1]),
                                                 float(result.x[2])))
    residual_sum = float(np.abs(y - predict_hinge(p, params)).sum())
    if abs(residual_sum - float(result.fun)) > 1e-7 * max(1.0, residual_sum):
        raise HingeError("LP slack objective does not match unweighted absolute loss")
    return {
        "schema_version": 1,
        "status": "OPTIMAL",
        "formula": "a+sL*min(p,50)+sH*max(p-50,0)",
        "intercept": params.intercept,
        "slope_low": params.slope_low,
        "slope_high": params.slope_high,
        "slope_bounds": [0.0, 2.0],
        "knot": 50.0,
        "fit_rows": int(n),
        "support": {"p_lt_50": low_support, "p_gte_50": high_support},
        "objective_unweighted_sum_absolute": residual_sum,
        "solver": {"method": "highs", "threads": SOLVER_THREADS,
                   "time_limit_seconds": SOLVER_TIME_LIMIT_SECONDS},
    }


def apply_hinge_present_only(direct_predictions: Any, missing_mask: Any,
                             parameters: HingeParameters | Mapping[str, Any]) -> np.ndarray:
    raw = np.asarray(direct_predictions)
    missing = np.asarray(missing_mask, dtype=bool)
    if raw.ndim != 1 or missing.ndim != 1 or len(raw) != len(missing):
        raise HingeError("Prediction and missing-mask vectors must be aligned")
    if raw.dtype.kind != "f" or raw.dtype.itemsize != 8:
        raise HingeError("Raw direct fallback must be float64 for bit-exact preservation")
    raw_values = _prediction_vector(raw, "raw direct predictions")
    output = raw.copy()
    present = ~missing
    if np.any(present):
        transformed = predict_hinge(raw_values[present], parameters)
        output[present] = np.clip(transformed, 0.0, 100.0)
    # Missing entries were never assigned and retain the input float64 bit pattern.
    return output


def apply_global_present_only(direct_predictions: Any, missing_mask: Any, *,
                              intercept: float, slope: float) -> np.ndarray:
    raw = np.asarray(direct_predictions)
    missing = np.asarray(missing_mask, dtype=bool)
    if raw.ndim != 1 or missing.ndim != 1 or len(raw) != len(missing):
        raise HingeError("Prediction and missing-mask vectors must be aligned")
    if raw.dtype.kind != "f" or raw.dtype.itemsize != 8:
        raise HingeError("Raw direct fallback must be float64 for bit-exact preservation")
    p = _prediction_vector(raw, "raw direct predictions")
    if not math.isfinite(intercept) or not math.isfinite(slope) or slope < 0.0:
        raise HingeError("Stored global affine coefficients are invalid")
    output = raw.copy()
    present = ~missing
    output[present] = np.clip(intercept + slope * p[present], 0.0, 100.0)
    return output


def write_json_once(path: Path, body: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(body), sort_keys=True, ensure_ascii=False,
                         allow_nan=False, indent=2).encode("utf-8") + b"\n"
    return _atomic_create_once(path, payload)


def _atomic_create_once(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=f".{path.name}.",
                                         suffix=".tmp", dir=path.parent,
                                         delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise HingeError(f"Immutable output already exists; refusing overwrite: {path}") from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    return hashlib.sha256(payload).hexdigest()


def fit_or_recover_published_state(path: Path, *, fit_id: str,
                                   input_identity: Mapping[str, Any],
                                   producer: Callable[[], Mapping[str, Any]]) -> tuple[dict[str, Any], str]:
    """Recover a published fit before invoking a producer after a postpublish crash."""
    if fit_id not in FIT_IDS:
        raise HingeError(f"Unknown fit id: {fit_id}")
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        if (state.get("status") != "FIT_PUBLISHED"
                or state.get("fit_id") != fit_id
                or state.get("input_identity") != dict(input_identity)):
            raise HingeError("Published fit state identity mismatch; refusing another start")
        return state, "RECOVERED_PUBLISHED"
    fit = dict(producer())
    body = {"schema_version": 1, "status": "FIT_PUBLISHED", "fit_id": fit_id,
            "input_identity": dict(input_identity), "fit": fit}
    write_json_once(path, body)
    return body, "NEWLY_PUBLISHED"


def write_candidate_csv(path: Path, record_ids: Any, predictions: Any) -> str:
    ids = np.asarray(record_ids, dtype=str)
    values = np.asarray(predictions, dtype=np.float64)
    if ids.ndim != 1 or values.ndim != 1 or len(ids) != len(values) or not len(ids):
        raise HingeError("CSV IDs and predictions must be nonempty aligned vectors")
    if len(np.unique(ids)) != len(ids):
        raise HingeError("CSV record IDs must be unique")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 100.0):
        raise HingeError("CSV predictions must be finite and in [0, 100]")
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(("record_id", "prediction"))
    for record_id, prediction in zip(ids, values, strict=True):
        writer.writerow((record_id, format(float(prediction), ".17g")))
    return _atomic_create_once(path, buffer.getvalue().encode("utf-8"))


def _run_git(*args: str) -> str:
    completed = subprocess.run(["git", *args], cwd=ROOT, check=True,
                               capture_output=True, text=True)
    return completed.stdout.strip()


def _committed_source_hashes() -> tuple[str, str, dict[str, str]]:
    branch = _run_git("branch", "--show-current")
    commit = _run_git("rev-parse", "HEAD")
    if branch != "codex/reg-position-failure-mechanism-v1":
        raise HingeError(f"Unexpected branch for identity freeze: {branch}")
    source_paths = (
        "scripts/reg_monotone_hinge_p50_v1.py",
        "scripts/verify_reg_monotone_hinge_p50_v1.py",
        "tests/test_reg_monotone_hinge_p50_v1.py",
        "reports/optimization_v3/reg_monotone_hinge_p50_v1/PROTOCOL.md",
        "reports/optimization_v3/reg_monotone_hinge_p50_v1/SYNTHETIC_QA.json",
    )
    hashes: dict[str, str] = {}
    for relative in source_paths:
        local = (ROOT / relative).read_bytes()
        committed = subprocess.run(["git", "show", f"{commit}:{relative}"], cwd=ROOT,
                                   check=True, capture_output=True).stdout
        if local != committed:
            raise HingeError(f"Source is not byte-identical to committed version: {relative}")
        hashes[relative] = hashlib.sha256(local).hexdigest()
    return branch, commit, hashes


def freeze_identity() -> tuple[Path, str]:
    """Write a hash-bound identity freeze; never opens an array or target file."""
    if IDENTITY_FREEZE.exists():
        raise HingeError(f"Identity freeze already exists: {IDENTITY_FREEZE}")
    prior_freeze_path = ROOT / "experiments/REG_FINAL_WAVE_FOLLOWUP_V2/PREFIT_FREEZE_R3.json"
    preflight_path = ROOT / "reports/optimization_v3/REG_FINAL_WAVE_FOLLOWUP_V2/IDENTITY_PREFLIGHT_R3.json"
    expected_outer = {
        "prior_r3_freeze": (prior_freeze_path, "70b49ff7dea59edf53622106392c1c5d943b46dcf0f4e792f5211b83453061b8"),
        "prior_r3_identity_preflight": (preflight_path, "639c080b91121d7d1183fc0de7f6d9f8cec2e68f496f4242cddd072dd4e5f211"),
        "prior_r3_source_pass_metadata_only": (ROOT / "experiments/REG_FINAL_WAVE_FOLLOWUP_V2/ROOT_SOURCE_PASS.json",
            "d5d71092e2824596cc7416a22af9de8f074b2123dca36ebf1a7b661c6b8da10f"),
        "prior_r3_diagnostics_receipt": (ROOT / "reports/optimization_v3/REG_FINAL_WAVE_FOLLOWUP_V2/DIAGNOSTICS.json",
            "136d4a8c7a565413ec550dca1212f7cda22a2c8ca5d70dbfdd6672ceb14961d8"),
        "prior_r3_phase_closure": (ROOT / "reports/optimization_v3/REG_FINAL_WAVE_FOLLOWUP_V2/DIAGNOSTIC_PHASE_CLOSURE_R3.md",
            "02dd7c9117d3f3fb83f98603cb5e0739ef618b505311212312ef44fa04b12243"),
        "prior_r3_ledger": (ROOT / "experiments/runs/REG_FINAL_WAVE_FOLLOWUP_V2/lp_ledger.jsonl",
            "961366c4b0e8ed998aa147b624a91fa05bae9a644b04cba9c5612eeafa9fc33b"),
    }
    outer_hashes: dict[str, str] = {}
    for name, (path, expected) in expected_outer.items():
        if not path.is_file():
            raise HingeError(f"Frozen source/input artifact is missing: {path}")
        actual = file_sha256(path)
        if actual != expected:
            raise HingeError(f"Frozen source/input artifact changed: {name}")
        outer_hashes[name] = actual

    prior_freeze = json.loads(prior_freeze_path.read_text(encoding="utf-8"))
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if (prior_freeze.get("status") != "PREFIT_FROZEN_IDENTITY_ONLY"
            or preflight.get("status") != "PASS_IDENTITY_ONLY_PREFIT"
            or prior_freeze.get("support_identity") != preflight.get("support_identity")
            or prior_freeze.get("input_hashes") != preflight.get("input_hashes")
            or prior_freeze.get("source_hashes") != preflight.get("source_hashes")):
        raise HingeError("R3 frozen OOF identity metadata is inconsistent")

    verified_historical_inputs: dict[str, dict[str, str]] = {}
    for name, record in prior_freeze["input_hashes"].items():
        path = Path(record["path"])
        expected = record["sha256"]
        if not path.is_file() or file_sha256(path) != expected:
            raise HingeError(f"R3 historical input bytes differ from freeze: {name}")
        verified_historical_inputs[name] = {"path": str(path), "sha256": expected}
    verified_historical_sources: dict[str, dict[str, str]] = {}
    for name, expected in prior_freeze["source_hashes"].items():
        path = Path(preflight["source_paths"][name])
        if not path.is_file() or file_sha256(path) != expected:
            raise HingeError(f"R3 historical source bytes differ from freeze: {name}")
        verified_historical_sources[name] = {"path": str(path), "sha256": expected}

    run_dir = ROOT / "experiments/runs/REG_FINAL_WAVE_FOLLOWUP_V2"
    stage_files = {
        "D1_diagnostic_rows": run_dir / "D1_diagnostic_rows.npz",
        "D2_diagnostic_rows": run_dir / "D2_diagnostic_rows.npz",
        "D1_GLOBAL_state": run_dir / "fits/D1_GLOBAL.json",
        "D2_GLOBAL_state": run_dir / "fits/D2_GLOBAL.json",
    }
    stage_hashes = {}
    for name, path in stage_files.items():
        if not path.is_file():
            raise HingeError(f"Required causal GLOBAL/row artifact missing: {path}")
        stage_hashes[name] = {"path": path.relative_to(ROOT).as_posix(),
                              "sha256": file_sha256(path)}
    branch, commit, source_hashes = _committed_source_hashes()
    body = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "PREFIT_IDENTITY_FROZEN_PENDING_ROOT_SOURCE_PASS",
        "branch": branch,
        "source_commit": commit,
        "source_hashes": source_hashes,
        "historical_source_and_input_hashes": {
            "r3_source_hashes_verified": verified_historical_sources,
            "r3_input_hashes_verified": verified_historical_inputs,
        },
        "outer_receipt_hashes": outer_hashes,
        "causal_stage_artifact_hashes": stage_hashes,
        "support_identity": preflight["support_identity"],
        "stage_support": preflight["stage_support"],
        "population": {"fit_rows_all_oof_including_missing": 20_903,
                       "D1_fit_rows": 4_950, "D1_score_rows": 7_462,
                       "D2_fit_rows": 12_412, "D2_score_rows": 8_491,
                       "pooled_unique_score_rows": 15_953,
                       "conditional_final_fit_rows": 20_903},
        "method": {"formula": "g(p)=a+sL*min(p,50)+sH*max(p-50,0)",
                   "knot": 50.0, "slope_low_bounds": [0.0, 2.0],
                   "slope_high_bounds": [0.0, 2.0],
                   "loss": "unweighted_sum_absolute", "recency_weights": False,
                   "solver": "sparse_highs", "threads": 2,
                   "time_limit_seconds": 180, "clip_during_fit": False,
                   "clip_at_application": [0.0, 100.0]},
        "fit_plan": list(FIT_IDS),
        "official_lp_starts": 0,
        "official_labels_accessed": False,
        "test_or_sample_accessed": False,
        "csv_created": False,
        "kaggle_upload": False,
        "old_r3_source_pass_authorizes_hinge": False,
        "new_root_source_pass_path": "experiments/reg_monotone_hinge_p50_v1/ROOT_SOURCE_PASS.json",
        "resource_plan": {"concurrent_heavy_jobs": 1, "threads": 2,
                           "reserved_ram_bytes": 2 * 1024**3,
                           "solver_cost_per_start_seconds": 180,
                           "actual_official_compute_cost": "UNKNOWN_PENDING_SOURCE_PASS"},
    }
    return IDENTITY_FREEZE, write_json_once(IDENTITY_FREEZE, body)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--synthetic-qa", action="store_true")
    group.add_argument("--freeze-identity", action="store_true")
    args = parser.parse_args()
    if args.synthetic_qa:
        from scripts.verify_reg_monotone_hinge_p50_v1 import write_synthetic_qa
        print(json.dumps(write_synthetic_qa(), sort_keys=True))
        return 0
    path, digest = freeze_identity()
    print(json.dumps({"status": "IDENTITY_FROZEN_ONLY", "path": str(path),
                      "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
