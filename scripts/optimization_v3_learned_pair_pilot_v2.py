"""Four-fit learned-factor Classification pilot; query labels open after verification only."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
import psutil
import scipy
import sklearn
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             log_loss, precision_score, recall_score, roc_auc_score)
from threadpoolctl import threadpool_limits

import optimization_v3_positional_bag_offset_v1 as offset_inputs
from src.models.learned_pair_pilot_v1 import (initialize_parameters,
                                               objective_gradient_sum,
                                               score_logits, unpack_parameters)

CARD_ID = "LEARNED_PAIR_PILOT_V2"
STAGE = "LEARNED_PAIR_DIAGNOSTIC_AMENDMENT"
AMENDMENT_ID = "LEARNED_PAIR_PILOT_V2_CAP117_2026_11_03"
RESERVATION_KEY = f"{CARD_ID}:{STAGE}"
FIT_IDS = (
    "LEARNED_PAIR_PILOT_V2_E67_FM",
    "LEARNED_PAIR_PILOT_V2_E67_NOPAIR",
    "LEARNED_PAIR_PILOT_V2_E75_FM",
    "LEARNED_PAIR_PILOT_V2_E75_NOPAIR",
)
ORIGINS = (67, 75)
TRAIN = ROOT / "data/raw/classification/Train.csv"
RECIPE = ROOT / "reports/optimization_v3/positional_bag_offset_v1/RECIPE.json"
AUTH = ROOT / "reports/optimization_v3/learned_pair_pilot_v2/SOL_EXECUTION_HANDOFF.md"
PREFLIGHT_REPORT = ROOT / "reports/optimization_v3/learned_pair_preflight_v1/REPORT.md"
PILOT_ROOT = ROOT / "experiments/runs/optimization_v3_research_next_v1/POSITIONAL_TEXT_PILOT_V1"
LEDGER = ROOT / "experiments/runs/optimization_v3_research_next_v1/budget.jsonl"
RUN_ROOT = ROOT / "experiments/runs/optimization_v3_research_next_v1/LEARNED_PAIR_PILOT_V2"
REPORT_ROOT = ROOT / "reports/optimization_v3/learned_pair_pilot_v2"
ACCOUNTING = ROOT / "reports/optimization_v3/positional_text_pilot_v1/FIT_ACCOUNTING.json"
A02_REFERENCE_ROOT = ROOT / "experiments/runs/optimization_v3_a02_neighbors_v1"
A02_PROTOCOL = ROOT / "experiments/optimization_v3/c19_a02_v2/A02/protocol.json"
A02_FREEZE = ROOT / "experiments/optimization_v3/c19_a02_v2/A02/freeze.json"
VERIFY_SCRIPT = ROOT / "scripts/verify_optimization_v3_learned_pair_pilot_v2.py"
INITIAL_CAPS = {"continuation": 38, "control_replay": 2, "pilot": 72, "technical_retry": 1}
BASE_ACTIVE = {"continuation": 38, "control_replay": 2, "pilot": 69, "technical_retry": 1}
AMENDED_CAPS = {"continuation": 38, "control_replay": 2, "pilot": 76, "technical_retry": 1}
BASE_GLOBAL_CAP = 113
AMENDED_GLOBAL_CAP = 117
BASE_STARTS = 110
BASE_SAVED = 106
BASE_FAILED = 4
RSS_LIMIT = 6 * 1024**3
INITIAL_FREE_MIN = 4 * 1024**3
FREE_RESERVE = 2 * 1024**3
RANK = 16
CHUNK_ROWS = 1024
C_VALUE = 1.0
SOLVER_OPTIONS = {"maxiter": 1000, "maxfun": 2000, "maxcor": 10,
                  "maxls": 50, "ftol": 1e-9}
SOURCE_FILES = (
    "scripts/optimization_v3_learned_pair_pilot_v2.py",
    "scripts/verify_optimization_v3_learned_pair_pilot_v2.py",
    "src/models/learned_pair_pilot_v1.py",
    "tests/test_learned_pair_pilot_v1.py",
    "tests/test_learned_pair_pilot_v2.py",
    "reports/optimization_v3/learned_pair_pilot_v2/PROTOCOL.md",
    "reports/optimization_v3/learned_pair_pilot_v2/SOL_EXECUTION_HANDOFF.md",
    "reports/optimization_v3/learned_pair_pilot_v1/PROTOCOL.md",
    "reports/optimization_v3/learned_pair_pilot_v1/FAILURE_DIAGNOSTIC.md",
    "reports/optimization_v3/learned_pair_preflight_v1/REPORT.md",
    "reports/optimization_v3/learned_pair_preflight_v1/SOL_PILOT_AUTHORIZATION.md",
    "experiments/optimization_v3/c19_a02_v2/A02/protocol.json",
    "experiments/optimization_v3/c19_a02_v2/A02/freeze.json",
    "scripts/optimization_v3_a02_neighbors.py",
    "scripts/optimization_v3_a02_neighbors_verify.py",
    "src/models/classification_a02_neighbors.py",
    "src/models/research_next_contract.py",
    "scripts/optimization_v3_positional_bag_offset_v1.py",
    "scripts/optimization_v3_positional_text_pilot_v1.py",
    "src/models/optimization_v3_positional_text_pilot_v1.py",
)


class LearnedPairPilotError(RuntimeError):
    pass


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(x) for x in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(_jsonable(value), sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(",", ":")).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, json.dumps(_jsonable(value), sort_keys=True, ensure_ascii=False,
                                   indent=2, allow_nan=False).encode("utf-8") + b"\n")


def _atomic_npz(path: Path, **arrays: Any) -> None:
    import io
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    _atomic_bytes(path, stream.getvalue())


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _create_once_receipt(path: Path, payload: Mapping[str, Any], stable_fields: tuple[str, ...]) -> dict[str, Any]:
    """Write an audit receipt once, or validate its immutable identity on resume."""
    body = dict(payload)
    if path.is_file():
        existing = _read_json(path)
        mismatches = [key for key in stable_fields if existing.get(key) != body.get(key)]
        if mismatches:
            raise LearnedPairPilotError(f"Existing audit receipt identity mismatch for {mismatches}: {path}")
        return existing
    _atomic_json(path, body)
    return body


def _git(*args: str) -> str:
    completed = subprocess.run(["git", "-C", str(ROOT), *args], check=True,
                               capture_output=True, text=True)
    return completed.stdout.strip()


def _source_hashes() -> dict[str, str]:
    return {rel: _sha256_file(ROOT / rel) for rel in SOURCE_FILES}


def _recipe_body() -> dict[str, Any]:
    return {
        "schema_version": 1, "card_id": CARD_ID, "fit_ids": list(FIT_IDS),
        "origins": list(ORIGINS), "matrix_arm": "saved C2_B fit_base/score_base; X unchanged",
        "interaction": "FM on row-L2-normalized copy Q of X; rank16",
        "score": "Xw+b+0.5*sum((QV)^2-Q^2(V^2))",
        "loss": "SUM stable binary logistic loss + (||w||2^2+||V||F^2+b^2)/(2*C), C=1",
        "chunk_rows": CHUNK_ROWS, "l2_application": "once after sum data loss/gradient accumulation",
        "factor_initialization": {"seed": 42, "distribution": "Normal(0,0.01)", "reset_per_origin": True},
        "main_initialization": {"w": 0.0, "b": 0.0},
        "solver": {"method": "L-BFGS-B", **SOLVER_OPTIONS,
                   "gtol_rule": "n_fit * 1e-4 (SUM objective); require independently verified mean-scale gradient infinity norm <= 1e-4",
                   "gtol_by_origin": {"E67": 2.9705, "E75": 4.2117}},
        "control": "same X, objective, C, solver and zero w/b; no V",
        "inference": "direct present-stream expert; exact saved C1_B HGB fallback for missing/blank; threshold=0.50",
        "forbidden": ["offset", "A02 residual", "probability blend", "calibration", "Test/Sample/Kaggle", "Regression labels"],
        "views": {"E67": ["W4", "W5", "W6", "H12_E67"], "E75": ["W6"]},
        "gates": {
            "E67_H12_gain_min_vs_bag": 0.0005,
            "E67_H12_gain_min_vs_nopair": 0.0002,
            "E75_W6_gain_min_vs_both": 0.0,
            "E67_H12_gain_min_vs_a02": 0.0005,
            "E75_W6_gain_min_vs_a02": 0.0005,
            "E67_window_harm_max_vs_both": 0.005,
            "E67_window_harm_max_vs_a02": 0.005,
            "supported_subgroup_f1_harm_max": 0.01,
            "supported_subgroup_positive_recall_harm_max": 0.02,
            "subgroup_comparators": ["bag", "nopair", "a02_observed"],
            "subgroup_support": {"n": 200, "each_class": 30},
        },
        "resource": {"jobs": 1, "threads": 2, "rss_limit_bytes": RSS_LIMIT,
                     "initial_free_min_bytes": INITIAL_FREE_MIN, "free_reserve_bytes": FREE_RESERVE},
        "runtime": {"python": sys.version.split()[0], "numpy": np.__version__,
                    "scipy": scipy.__version__, "pandas": pd.__version__,
                    "scikit_learn": sklearn.__version__, "psutil": psutil.__version__},
        "query_targets": "closed until all four successful receipts, durable predictions, and independent PASS",
        "selection_scope": "adaptive diagnostic mechanism pilot; review only; no champion change, continuation, or upload",
    }


def _solver_options(n_fit: int) -> dict[str, Any]:
    if n_fit <= 0:
        raise LearnedPairPilotError("Optimizer fit population must be positive")
    return {**SOLVER_OPTIONS, "gtol": float(n_fit * 1e-4)}


def _optimizer_convergence_gate(*, solver_success: bool, finite_state: bool,
                                initial_loss: float, final_loss: float,
                                gradient_inf_sum: float, n_fit: int) -> dict[str, Any]:
    normalized = float(gradient_inf_sum / n_fit) if n_fit > 0 else float("inf")
    stationary = bool(np.isfinite(normalized) and normalized <= 1e-4)
    loss_guard = bool(np.isfinite(initial_loss) and np.isfinite(final_loss)
                      and final_loss <= initial_loss + 1e-8)
    return {
        "optimizer_converged": bool(solver_success and finite_state and stationary and loss_guard),
        "normalized_gradient_inf_norm": normalized,
        "stationarity_guard": stationary,
        "loss_nonincrease_guard": loss_guard,
        "gtol": float(n_fit * 1e-4) if n_fit > 0 else None,
    }


def _resource_snapshot(estimate_bytes: int, *, initial: bool = False) -> dict[str, Any]:
    process = psutil.Process()
    rss = int(process.memory_info().rss)
    available = int(psutil.virtual_memory().available)
    projected_rss = rss + int(estimate_bytes)
    projected_free = available - int(estimate_bytes)
    snapshot = {
        "jobs": 1, "threads": 2, "process_rss_bytes_sample": rss,
        "system_available_ram_bytes_sample": available,
        "estimated_incremental_memory_bytes": int(estimate_bytes),
        "projected_rss_bytes": projected_rss, "projected_available_ram_bytes": projected_free,
        "max_rss_bytes": RSS_LIMIT, "initial_available_ram_min_bytes": INITIAL_FREE_MIN,
        "minimum_free_ram_bytes": FREE_RESERVE,
        "measurement_note": "psutil instantaneous sample and conservative estimate; not OS peak RSS",
    }
    if initial and available < INITIAL_FREE_MIN:
        raise LearnedPairPilotError(f"Initial free RAM below 4 GiB: {available}")
    if projected_rss > RSS_LIMIT:
        raise LearnedPairPilotError(f"Projected process RSS exceeds 6 GiB: {projected_rss}")
    if projected_free < FREE_RESERVE:
        raise LearnedPairPilotError(f"Projected free RAM below 2 GiB reserve: {projected_free}")
    return snapshot


def _resource_check() -> None:
    state = _resource_snapshot(512 * 1024**2)
    if state["projected_rss_bytes"] > RSS_LIMIT or state["projected_available_ram_bytes"] < FREE_RESERVE:
        raise LearnedPairPilotError("Running resource reserve violated")


def _ledger_events(path: Path = LEDGER) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append_ledger(event: Mapping[str, Any], *, path: Path, official: bool = False) -> dict[str, Any]:
    dest = path.resolve()
    if dest == LEDGER.resolve() and not official:
        raise LearnedPairPilotError("Production ledger writer is disabled without official authorization")
    if official and dest != LEDGER.resolve():
        raise LearnedPairPilotError("Official ledger mode cannot write to a temporary ledger")
    payload = dict(event)
    payload.setdefault("at_utc", datetime.now(timezone.utc).isoformat())
    encoded = _canonical_bytes(payload) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "ab", closefd=False) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)
    return payload


def _live_snapshot(events: list[dict[str, Any]]) -> dict[str, Any]:
    return offset_inputs.ledger_snapshot(events)


def _origin_files(origin: int) -> tuple[Path, Path, Path]:
    manifest = PILOT_ROOT / f"MATRIX_MANIFEST_E{origin}.json"
    pred = PILOT_ROOT / f"E{origin}" / f"predictions_E{origin}.npz"
    return manifest, pred, PILOT_ROOT / f"E{origin}"


def _verify_a02_freeze() -> dict[str, Any]:
    if not A02_PROTOCOL.is_file() or not A02_FREEZE.is_file():
        raise LearnedPairPilotError("Frozen A02 protocol or freeze is missing")
    protocol = _read_json(A02_PROTOCOL)
    frozen = _read_json(A02_FREEZE)
    if (_canonical_sha256(protocol) != frozen.get("protocol_sha256")
            or _canonical_sha256(frozen.get("identity")) != frozen.get("freeze_signature")
            or frozen.get("identity", {}).get("task") != "A02"):
        raise LearnedPairPilotError("Frozen A02 protocol/freeze signature is invalid")
    for relative, expected in frozen.get("source_sha256", {}).items():
        path = ROOT / relative
        if not path.is_file() or _sha256_file(path) != expected:
            raise LearnedPairPilotError(f"Frozen A02 source changed or missing: {relative}")
    for relative, expected in frozen.get("input_sha256", {}).items():
        path = ROOT / relative
        if not path.is_file() or _sha256_file(path) != expected:
            raise LearnedPairPilotError(f"Frozen A02 input changed or missing: {relative}")
    return {
        "protocol_path": A02_PROTOCOL.relative_to(ROOT).as_posix(),
        "protocol_sha256": _sha256_file(A02_PROTOCOL),
        "freeze_path": A02_FREEZE.relative_to(ROOT).as_posix(),
        "freeze_sha256": _sha256_file(A02_FREEZE),
        "freeze_signature": frozen["freeze_signature"],
        "source_sha256": frozen["source_sha256"],
        "input_sha256": frozen["input_sha256"],
    }


def _verify_a02_observed(origin: int, fit_rows: np.ndarray, rows: np.ndarray, ids: np.ndarray,
                         cycles: np.ndarray) -> dict[str, Any]:
    folder = A02_REFERENCE_ROOT / f"E{origin}" / "observed"
    required = ("predictions.npz", "fit_receipt.json", "fit_state.json", "model.joblib", "predictions.json")
    if any(not (folder / name).is_file() for name in required):
        raise LearnedPairPilotError(f"E{origin} exact-origin A02 observed bundle is incomplete")
    fit = _read_json(folder / "fit_receipt.json")
    state = _read_json(folder / "fit_state.json")
    sidecar = _read_json(folder / "predictions.json")
    archive_path = folder / "predictions.npz"
    model_path = folder / "model.joblib"
    state_sha = _canonical_sha256(state)
    receipt_body = dict(fit)
    receipt_sha = receipt_body.pop("receipt_sha256", None)
    if (receipt_sha != _canonical_sha256(receipt_body)
            or fit.get("fit_id") != f"A02_E{origin}_OBSERVED"
            or fit.get("origin_cycle") != origin or fit.get("residual_arm") != "observed"
            or fit.get("freeze_signature") != _read_json(A02_FREEZE).get("freeze_signature")
            or fit.get("model_sha256") != _sha256_file(model_path)
            or fit.get("state_sha256") != state_sha
            or state.get("card_id") != "CLASSIFICATION_A02_NEIGHBOR_RESIDUAL"
            or state.get("origin_cycle") != origin or state.get("residual_arm") != "observed"
            or fit.get("query_labels_passed_to_fit") is not False
            or fit.get("test_data_used") is not False
            or fit.get("task_estimator_fit_count") != 1
            or sidecar.get("fit_id") != fit.get("fit_id")
            or sidecar.get("fit_receipt_sha256") != receipt_sha
            or sidecar.get("origin_cycle") != origin or sidecar.get("residual_arm") != "observed"
            or sidecar.get("freeze_signature") != fit.get("freeze_signature")
            or sidecar.get("npz_sha256") != _sha256_file(archive_path)
            or sidecar.get("query_labels_passed_to_prediction") is not False):
        raise LearnedPairPilotError(f"E{origin} A02 fit/state/model/output receipts do not bind")
    # A02's recorded prefix row identities must match the already-verified C2_B fit population.
    if (int(fit.get("text_prefix_rows", -1)) != int(len(fit_rows))
            or fit.get("text_prefix_row_positions_sha256")
               != hashlib.sha256(fit_rows.astype("<i8").tobytes()).hexdigest()):
        raise LearnedPairPilotError(f"E{origin} A02 prefix row count differs from C2_B fit population")
    with np.load(archive_path, allow_pickle=False) as archive:
        a02_rows = np.asarray(archive["score_rows"], dtype=np.int64)
        a02_ids = np.asarray(archive["score_record_ids"]).astype(str)
        a02_cycles = np.asarray(archive["score_cycles"], dtype=np.int64)
        observed = np.asarray(archive["prediction"], dtype=np.float64)
    if (not np.array_equal(a02_rows, rows) or not np.array_equal(a02_ids, ids.astype(str))
            or not np.array_equal(a02_cycles, cycles.astype(np.int64))
            or observed.shape != rows.shape or not np.isfinite(observed).all()
            or np.any((observed < 0.0) | (observed > 1.0))):
        raise LearnedPairPilotError(f"E{origin} A02 predictions do not exactly match score rows/IDs/cycles")
    canonical_ids = hashlib.sha256(json.dumps(a02_ids.tolist(), separators=(",", ":"),
                                               ensure_ascii=False).encode("utf-8")).hexdigest()
    if (int(sidecar.get("score_rows", -1)) != len(rows)
            or sidecar.get("score_row_positions_sha256") != hashlib.sha256(rows.astype("<i8").tobytes()).hexdigest()
            or sidecar.get("score_record_ids_sha256") != canonical_ids):
        raise LearnedPairPilotError(f"E{origin} A02 prediction sidecar row identity/hash mismatch")
    return {
        "status": "ALIGNED_VERIFIED_REFERENCE_ONLY",
        "path": archive_path.relative_to(ROOT).as_posix(),
        "bundle_sha256": _sha256_file(archive_path),
        "fit_receipt_sha256": _sha256_file(folder / "fit_receipt.json"),
        "fit_state_sha256": _sha256_file(folder / "fit_state.json"),
        "model_sha256": _sha256_file(model_path),
        "prediction_sidecar_sha256": _sha256_file(folder / "predictions.json"),
        "freeze_signature": fit["freeze_signature"],
        "fit_id": fit["fit_id"],
        "prefix_rows": int(fit["text_prefix_rows"]),
        "observed_probability_sha256": offset_inputs.sha256_array(observed),
        "score_ids_sha256": offset_inputs.sha256_array(a02_ids),
        "score_cycles_sha256": offset_inputs.sha256_array(a02_cycles),
    }


def _verify_saved_inputs() -> dict[str, Any]:
    if not TRAIN.is_file() or _sha256_file(TRAIN) != _read_json(RECIPE)["input"]["sha256"]:
        raise LearnedPairPilotError("Official Classification Train hash mismatch")
    state = offset_inputs.verify_saved_inputs(include_current_code=True)
    frame = offset_inputs._read_identity_frame()
    a02_freeze = _verify_a02_freeze()
    accounting = _read_json(ACCOUNTING)
    by_id = {row["fit_id"]: row for row in accounting["fit_units"]}
    origins: dict[str, Any] = {}
    for origin in ORIGINS:
        manifest_path, pred_path, _ = _origin_files(origin)
        manifest = _read_json(manifest_path)
        if manifest_path.name not in {f"MATRIX_MANIFEST_E{origin}.json"}:
            raise LearnedPairPilotError("Unexpected matrix manifest identity")
        with np.load(pred_path, allow_pickle=False) as data:
            pred = {key: data[key].copy() for key in data.files}
        fit_rows = np.asarray(pred["fit_rows"], dtype=np.int64)
        score_rows = np.asarray(pred["score_rows"], dtype=np.int64)
        if len(np.intersect1d(fit_rows, score_rows)):
            raise LearnedPairPilotError(f"E{origin} fit and score rows overlap")
        if (not np.array_equal(pred["score_record_ids"].astype(str),
                               frame.iloc[score_rows]["record_id"].astype(str).to_numpy())
                or not np.array_equal(pred["score_cycles"].astype(np.int64),
                                      frame.iloc[score_rows]["cycle_code"].to_numpy(dtype=np.int64))):
            raise LearnedPairPilotError(f"E{origin} cached row IDs/cycles no longer align")
        fit_id = f"POSITIONAL_TEXT_PILOT_V1_E{origin}_BAG_LR"
        population = by_id[fit_id]["fit_population"]
        if (population.get("row_indices_sha256") != offset_inputs.sha256_array(fit_rows)
                or population.get("matrix_manifest_sha256") != _sha256_file(manifest_path)):
            raise LearnedPairPilotError(f"E{origin} training row/matrix lineage mismatch")
        matrices = {}
        matrix_meta = {}
        for matrix_name in ("fit_base", "score_base"):
            matrices[matrix_name] = offset_inputs._load_matrix(manifest, matrix_name)
            matrix_meta[matrix_name] = {
                "path": manifest["matrices"][matrix_name]["path"],
                "archive_sha256": manifest["matrices"][matrix_name]["sha256"],
                "logical_sha256": manifest["matrices"][matrix_name]["logical_sha256"],
                "shape": list(matrices[matrix_name].shape),
            }
        fit = sparse.csr_matrix(matrices["fit_base"], dtype=np.float64)
        score = sparse.csr_matrix(matrices["score_base"], dtype=np.float64)
        if (fit.shape[0] != len(fit_rows) or score.shape[0] != len(score_rows)
                or fit.shape[1] != score.shape[1]):
            raise LearnedPairPilotError(f"E{origin} C2_B matrices are not row/column aligned")
        present = np.asarray(pred["present_mask"], dtype=bool)
        fallback = np.asarray(pred["prob_missing_hgb"], dtype=np.float64)
        bag_raw = np.asarray(pred["prob_bag"], dtype=np.float64)
        bag_route = np.asarray(pred["prob_route_bag"], dtype=np.float64)
        if present.shape != (len(score_rows),) or fallback.shape != present.shape:
            raise LearnedPairPilotError(f"E{origin} saved availability/fallback arrays are misaligned")
        _validate_cached_bag_route(bag_raw, bag_route, present, fallback)
        a02_meta = _verify_a02_observed(
            origin, fit_rows, score_rows, pred["score_record_ids"].astype(str),
            pred["score_cycles"].astype(np.int64))
        origins[f"E{origin}"] = {
            "origin": origin, "manifest_path": manifest_path.relative_to(ROOT).as_posix(),
            "matrix_manifest_sha256": _sha256_file(manifest_path),
            "prediction_bundle_path": pred_path.relative_to(ROOT).as_posix(),
            "prediction_bundle_sha256": _sha256_file(pred_path),
            "fit_rows": len(fit_rows), "score_rows": len(score_rows),
            "fit_rows_sha256": offset_inputs.sha256_array(fit_rows),
            "score_rows_sha256": offset_inputs.sha256_array(score_rows),
            "fit_ids_sha256": offset_inputs.sha256_array(frame.iloc[fit_rows]["record_id"].astype(str).to_numpy()),
            "score_ids_sha256": offset_inputs.sha256_array(pred["score_record_ids"].astype(str)),
            "score_cycles_sha256": offset_inputs.sha256_array(pred["score_cycles"].astype(np.int64)),
            "fit_matrix": matrix_meta["fit_base"], "score_matrix": matrix_meta["score_base"],
            "fallback_probability_sha256": offset_inputs.sha256_array(fallback),
            "baseline_probability_sha256": offset_inputs.sha256_array(bag_route),
            "bag_raw_probability_sha256": offset_inputs.sha256_array(bag_raw),
            "present_rows": int(present.sum()), "missing_rows": int((~present).sum()),
            "text_load_q75_fit_only": manifest.get("text_load_q75_fit_only"),
            "a02_observed_reference": a02_meta,
        }
    return {
        "input_sha256": _sha256_file(TRAIN), "input_rows": len(frame),
        "historical_freeze_signature": state["pilot_freeze_signature"],
        "historical_recipe_sha256": state["pilot_recipe_sha256"],
        "historical_accounting_sha256": state["pilot_accounting_sha256"],
        "historical_source_hashes": state["source_hashes"],
        "a02_frozen_reference": a02_freeze, "origins": origins,
    }


def _expected_pre_amend_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "global_cap": BASE_GLOBAL_CAP, "effective_category_caps": dict(INITIAL_CAPS),
        "active_reservations_by_category": dict(BASE_ACTIVE),
        "active_reservations": sum(BASE_ACTIVE.values()),
        "fit_starts": BASE_STARTS, "fit_saved": BASE_SAVED,
        "fit_failed": BASE_FAILED, "fit_pending": 0,
        "event_rows": snapshot["event_rows"],
    }


def preflight() -> dict[str, Any]:
    if _git("status", "--porcelain"):
        raise LearnedPairPilotError("Official preflight requires a clean committed worktree")
    if not AUTH.is_file() or not PREFLIGHT_REPORT.is_file():
        raise LearnedPairPilotError("Authorization/preflight report missing")
    synthetic_path = RUN_ROOT / "SYNTHETIC_QA.json"
    if not synthetic_path.is_file():
        raise LearnedPairPilotError("Run the temporary end-to-end synthetic lifecycle on the committed code first")
    synthetic = _read_json(synthetic_path)
    if (synthetic.get("status") != "PASS_SYNTHETIC_END_TO_END"
            or synthetic.get("official_fit_calls") != 0
            or synthetic.get("source_commit") != _git("rev-parse", "HEAD")
            or synthetic.get("source_hashes") != _source_hashes()
            or synthetic.get("live_budget_ledger_sha256_before") != _sha256_file(LEDGER)
            or synthetic.get("live_budget_ledger_sha256_after") != _sha256_file(LEDGER)):
        raise LearnedPairPilotError("Synthetic QA is stale, incomplete, or changed the live ledger")
    state = _verify_saved_inputs()
    events = _ledger_events()
    snapshot = _live_snapshot(events)
    expected = _expected_pre_amend_snapshot(snapshot)
    if snapshot != expected:
        raise LearnedPairPilotError(f"Live ledger checkpoint differs from authorized V2 base state: {snapshot}")
    if any(e.get("fit_id") in FIT_IDS for e in events if e.get("event") in {"FIT_STARTED", "FIT_SAVED", "FIT_FAILED"}):
        raise LearnedPairPilotError("Learned-pair fit IDs already appear in live ledger")
    memory = _resource_snapshot(1024**3, initial=True)
    result = {
        "card_id": CARD_ID, "status": "PASS_READY_TO_FREEZE",
        "source_commit": _git("rev-parse", "HEAD"), "branch": _git("branch", "--show-current"),
        "source_hashes": _source_hashes(), "saved_inputs": state,
        "ledger_sha256": _sha256_file(LEDGER), "ledger_snapshot": snapshot,
        "resource": memory, "fit_ids": list(FIT_IDS), "authorization_sha256": _sha256_file(AUTH),
        "synthetic_qa_sha256": _sha256_file(synthetic_path),
    }
    _atomic_json(RUN_ROOT / "PREFLIGHT.json", result)
    return result


def freeze() -> dict[str, Any]:
    existing = RUN_ROOT / "FREEZE.json"
    if existing.is_file():
        return load_freeze()
    result = preflight()
    body = {
        "schema_version": 1, "card_id": CARD_ID,
        "source_commit": result["source_commit"], "branch": result["branch"],
        "input_sha256": result["saved_inputs"]["input_sha256"],
        "source_hashes": result["source_hashes"], "saved_inputs": result["saved_inputs"],
        "recipe": _recipe_body(), "recipe_sha256": _canonical_sha256(_recipe_body()),
        "authorization_sha256": _sha256_file(AUTH),
        "synthetic_qa_sha256": result["synthetic_qa_sha256"],
        "pre_reservation_ledger_sha256": result["ledger_sha256"],
        "pre_reservation_ledger_snapshot": result["ledger_snapshot"],
        "preflight_resource": result["resource"],
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    body["freeze_signature"] = _canonical_sha256(body)
    _atomic_json(existing, body)
    return body


def load_freeze() -> dict[str, Any]:
    path = RUN_ROOT / "FREEZE.json"
    if not path.is_file():
        raise LearnedPairPilotError("Frozen protocol missing; run freeze after clean commit")
    frozen = _read_json(path)
    body = dict(frozen)
    signature = body.pop("freeze_signature", None)
    if signature != _canonical_sha256(body):
        raise LearnedPairPilotError("Freeze signature mismatch")
    if (_git("rev-parse", "HEAD") != frozen["source_commit"]
            or _git("branch", "--show-current") != frozen["branch"]
            or _git("status", "--porcelain")):
        raise LearnedPairPilotError("Source commit, branch, or clean worktree changed after freeze")
    if _source_hashes() != frozen["source_hashes"] or _sha256_file(TRAIN) != frozen["input_sha256"]:
        raise LearnedPairPilotError("Frozen source or Train hash changed")
    if _sha256_file(RUN_ROOT / "SYNTHETIC_QA.json") != frozen["synthetic_qa_sha256"]:
        raise LearnedPairPilotError("Synthetic lifecycle QA receipt changed after freeze")
    current = _verify_saved_inputs()
    if current != frozen["saved_inputs"]:
        raise LearnedPairPilotError("Frozen saved input/matrix/fallback lineage changed")
    return frozen


def _own_events(events: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {name: [e for e in events if e.get("event") == name and e.get("fit_id") in FIT_IDS]
            for name in ("FIT_STARTED", "FIT_SAVED", "FIT_FAILED")}


def _reserve_locked() -> dict[str, Any]:
    frozen = load_freeze()
    events = _ledger_events()
    snap = _live_snapshot(events)
    amendments = [e for e in events if e.get("event") == "BUDGET_CAP_AMENDMENT" and e.get("amendment_id") == AMENDMENT_ID]
    reservations = [e for e in events if e.get("event") == "CARD_FITS_RESERVED" and e.get("reservation_key") == RESERVATION_KEY]
    if len(amendments) > 1 or len(reservations) > 1:
        raise LearnedPairPilotError("Duplicate learned-pair amendment/reservation")
    if not amendments and not reservations:
        if (snap != frozen["pre_reservation_ledger_snapshot"]
                or _sha256_file(LEDGER) != frozen["pre_reservation_ledger_sha256"]):
            raise LearnedPairPilotError("Live ledger changed after freeze; refusing new reservation")
        amendment = {
            "event": "BUDGET_CAP_AMENDMENT", "batch_id": "optimization_v3_research_next_v1",
            "amendment_id": AMENDMENT_ID, "card_id": CARD_ID,
            "authorized_by": "Sam via Sol explicit learned_pair pilot authorization",
            "prior_global_cap": BASE_GLOBAL_CAP, "global_cap": AMENDED_GLOBAL_CAP,
            "prior_effective_caps": dict(INITIAL_CAPS), "effective_caps": dict(AMENDED_CAPS),
            "active_reservations_before": dict(BASE_ACTIVE),
            "active_total_before": sum(BASE_ACTIVE.values()),
            "historical_started_saved_failed_before": {
                "started": BASE_STARTS, "saved": BASE_SAVED, "failed": BASE_FAILED},
            "pre_amendment_event_rows": len(events),
            "pre_amendment_ledger_sha256": _sha256_file(LEDGER),
            "source_commit": frozen["source_commit"], "freeze_signature": frozen["freeze_signature"],
            "reason": "Four authorized diagnostic V2 E67/E75 FM/no-pair predictor fits; no preprocessing fits or retries.",
        }
        _append_ledger(amendment, path=LEDGER, official=True)
        events = _ledger_events()
        snap = _live_snapshot(events)
    if amendments:
        amendment = amendments[0]
    else:
        amendment = next(e for e in events if e.get("event") == "BUDGET_CAP_AMENDMENT" and e.get("amendment_id") == AMENDMENT_ID)
    if any(amendment.get(k) != v for k, v in {
        "prior_global_cap": BASE_GLOBAL_CAP, "global_cap": AMENDED_GLOBAL_CAP,
        "prior_effective_caps": INITIAL_CAPS, "effective_caps": AMENDED_CAPS,
        "source_commit": frozen["source_commit"], "freeze_signature": frozen["freeze_signature"],
    }.items()):
        raise LearnedPairPilotError("Existing cap amendment differs from exact authorization")
    reservation_body = {
        "event": "CARD_FITS_RESERVED", "batch_id": "optimization_v3_research_next_v1",
        "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
        "category": "pilot", "fits": 4, "fit_ids": list(FIT_IDS),
        "amendment_id": AMENDMENT_ID, "global_cap": AMENDED_GLOBAL_CAP,
        "effective_category_cap": AMENDED_CAPS["pilot"],
        "source_commit": frozen["source_commit"], "freeze_signature": frozen["freeze_signature"],
    }
    if not reservations:
        expected_pre_reserve = {
            "global_cap": AMENDED_GLOBAL_CAP, "effective_category_caps": dict(AMENDED_CAPS),
            "active_reservations_by_category": dict(BASE_ACTIVE),
            "active_reservations": sum(BASE_ACTIVE.values()),
            "fit_starts": BASE_STARTS, "fit_saved": BASE_SAVED,
            "fit_failed": BASE_FAILED, "fit_pending": 0,
            "event_rows": snap["event_rows"],
        }
        if snap != expected_pre_reserve:
            raise LearnedPairPilotError(f"Post-amendment V2 budget does not match base snapshot: {snap}")
        _append_ledger(reservation_body, path=LEDGER, official=True)
        events = _ledger_events()
        snap = _live_snapshot(events)
    else:
        reservation_body = reservations[0]
    if any(reservation_body.get(k) != v for k, v in {
        "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
        "category": "pilot", "fits": 4, "fit_ids": list(FIT_IDS),
        "amendment_id": AMENDMENT_ID, "global_cap": AMENDED_GLOBAL_CAP,
        "effective_category_cap": AMENDED_CAPS["pilot"],
        "source_commit": frozen["source_commit"], "freeze_signature": frozen["freeze_signature"],
    }.items()):
        raise LearnedPairPilotError("Existing four-fit reservation differs from exact authorization")
    own = _own_events(events)
    released = sum(int(e.get("unused_fits", 0)) for e in events
                   if e.get("event") == "RESERVATION_UNUSED_RELEASED" and e.get("reservation_key") == RESERVATION_KEY)
    expected = {
        "global_cap": AMENDED_GLOBAL_CAP, "effective_category_caps": dict(AMENDED_CAPS),
        "active_reservations_by_category": {**BASE_ACTIVE, "pilot": BASE_ACTIVE["pilot"] + 4 - released},
        "active_reservations": sum(BASE_ACTIVE.values()) + 4 - released,
        "fit_starts": BASE_STARTS + len(own["FIT_STARTED"]),
        "fit_saved": BASE_SAVED + len(own["FIT_SAVED"]),
        "fit_failed": BASE_FAILED + len(own["FIT_FAILED"]),
        "fit_pending": len(own["FIT_STARTED"]) - len(own["FIT_SAVED"]) - len(own["FIT_FAILED"]),
        "event_rows": snap["event_rows"],
    }
    if snap != expected:
        raise LearnedPairPilotError(f"Post-reservation budget accounting differs: {snap} != {expected}")
    record = {"reservation": reservation_body, "snapshot": snap,
              "ledger_sha256": _sha256_file(LEDGER), "freeze_signature": frozen["freeze_signature"]}
    _atomic_json(RUN_ROOT / "RESERVATION.json", record)
    return record


def reserve() -> dict[str, Any]:
    with offset_inputs._fit_run_lock(RUN_ROOT / "FIT_RUN.lock"):
        return _reserve_locked()


def _check_live(*, freeze_signature: str, allow_pending: str | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    events = _ledger_events()
    snapshot = _live_snapshot(events)
    amendments = [e for e in events if e.get("event") == "BUDGET_CAP_AMENDMENT" and e.get("amendment_id") == AMENDMENT_ID]
    reservations = [e for e in events if e.get("event") == "CARD_FITS_RESERVED" and e.get("reservation_key") == RESERVATION_KEY]
    if len(amendments) != 1 or len(reservations) != 1:
        raise LearnedPairPilotError("Budget amendment/reservation must each occur exactly once")
    amendment, reservation = amendments[0], reservations[0]
    if (amendment.get("freeze_signature") != freeze_signature or reservation.get("freeze_signature") != freeze_signature
            or reservation.get("fit_ids") != list(FIT_IDS)):
        raise LearnedPairPilotError("Live ledger reservation does not match the active freeze")
    own = _own_events(events)
    terminals = {e.get("fit_id") for e in own["FIT_SAVED"] + own["FIT_FAILED"]}
    pending = [e for e in own["FIT_STARTED"] if e.get("fit_id") not in terminals]
    released = sum(int(e.get("unused_fits", 0)) for e in events
                   if e.get("event") == "RESERVATION_UNUSED_RELEASED" and e.get("reservation_key") == RESERVATION_KEY)
    expected = {
        "global_cap": AMENDED_GLOBAL_CAP, "effective_category_caps": dict(AMENDED_CAPS),
        "active_reservations_by_category": {**BASE_ACTIVE, "pilot": BASE_ACTIVE["pilot"] + 4 - released},
        "active_reservations": sum(BASE_ACTIVE.values()) + 4 - released,
        "fit_starts": BASE_STARTS + len(own["FIT_STARTED"]),
        "fit_saved": BASE_SAVED + len(own["FIT_SAVED"]),
        "fit_failed": BASE_FAILED + len(own["FIT_FAILED"]),
        "fit_pending": len(pending), "event_rows": snapshot["event_rows"],
    }
    if snapshot != expected:
        raise LearnedPairPilotError(f"Shared live budget differs from exact pilot accounting: {snapshot}")
    allowed_pending_count = 1 if allow_pending is not None else 0
    if len(pending) > allowed_pending_count or (allow_pending not in (None, "*")
                                                  and any(e.get("fit_id") != allow_pending for e in pending)):
        raise LearnedPairPilotError("Unexpected pending fit in shared budget ledger")
    return events, snapshot


@contextmanager
def _rss_sampler(interval: float = 0.5):
    stop = threading.Event()
    process = psutil.Process()
    samples = {"sample_count": 0, "peak_rss_bytes_sample": int(process.memory_info().rss),
               "minimum_available_ram_bytes_sample": int(psutil.virtual_memory().available)}

    def sample() -> None:
        while not stop.wait(interval):
            try:
                samples["sample_count"] += 1
                samples["peak_rss_bytes_sample"] = max(samples["peak_rss_bytes_sample"],
                                                         int(process.memory_info().rss))
                samples["minimum_available_ram_bytes_sample"] = min(
                    samples["minimum_available_ram_bytes_sample"], int(psutil.virtual_memory().available))
            except psutil.Error:
                return

    thread = threading.Thread(target=sample, name="learned-pair-rss-sampler", daemon=True)
    thread.start()
    try:
        yield samples
    finally:
        stop.set()
        thread.join(timeout=max(1.0, 2 * interval))
        samples["peak_rss_bytes_sample"] = max(samples["peak_rss_bytes_sample"], int(process.memory_info().rss))
        samples["minimum_available_ram_bytes_sample"] = min(
            samples["minimum_available_ram_bytes_sample"], int(psutil.virtual_memory().available))
        samples["measurement_note"] = "0.5-second process/system samples; not an OS peak RSS counter"


def _load_origin_for_fit(frozen: Mapping[str, Any], origin: int) -> dict[str, Any]:
    current = _verify_saved_inputs()
    if current != frozen["saved_inputs"]:
        raise LearnedPairPilotError("Saved matrix/fallback inputs changed after freeze")
    frame = offset_inputs._read_identity_frame()
    manifest_path, pred_path, _ = _origin_files(origin)
    manifest = _read_json(manifest_path)
    with np.load(pred_path, allow_pickle=False) as archive:
        pred = {key: archive[key].copy() for key in archive.files}
    fit_rows = np.asarray(pred["fit_rows"], dtype=np.int64)
    score_rows = np.asarray(pred["score_rows"], dtype=np.int64)
    fit_ids = frame.iloc[fit_rows]["record_id"].astype(str).to_numpy()
    fit_cycles = frame.iloc[fit_rows]["cycle_code"].to_numpy(dtype=np.int64)
    fit_id = f"POSITIONAL_TEXT_PILOT_V1_E{origin}_BAG_LR"
    accounting = _read_json(ACCOUNTING)
    by_id = {item["fit_id"]: item for item in accounting["fit_units"]}
    population = by_id[fit_id]["fit_population"]
    y = offset_inputs._fit_labels(TRAIN, fit_rows, fit_ids, population["fit_target_sha256"])
    if np.unique(y).tolist() != [0, 1]:
        raise LearnedPairPilotError(f"E{origin} training population must contain class order [0, 1]")
    x_fit = sparse.csr_matrix(offset_inputs._load_matrix(manifest, "fit_base"), dtype=np.float64)
    x_score = sparse.csr_matrix(offset_inputs._load_matrix(manifest, "score_base"), dtype=np.float64)
    if x_fit.shape[0] != len(y) or x_score.shape[0] != len(score_rows) or x_fit.shape[1] != x_score.shape[1]:
        raise LearnedPairPilotError(f"E{origin} saved matrices and row populations do not align")
    return {
        "origin": origin, "manifest": manifest, "manifest_path": manifest_path,
        "prediction_path": pred_path, "prediction": pred, "frame": frame,
        "fit_rows": fit_rows, "score_rows": score_rows, "fit_ids": fit_ids,
        "fit_cycles": fit_cycles, "x_fit": x_fit, "x_score": x_score, "y": y,
        "fit_target_sha256": population["fit_target_sha256"],
        "fit_population": population,
        "matrix_manifest_sha256": _sha256_file(manifest_path),
    }


def _fit_paths(fit_id: str) -> tuple[Path, Path]:
    return RUN_ROOT / "units" / f"{fit_id}.npz", RUN_ROOT / "receipts" / f"{fit_id}.json"


def _validate_receipt(fit_id: str, frozen: Mapping[str, Any], *, artifact_path: Path,
                      receipt_path: Path) -> dict[str, Any]:
    if not artifact_path.is_file() or not receipt_path.is_file():
        raise LearnedPairPilotError(f"Durable fit artifact or receipt missing: {fit_id}")
    receipt = _read_json(receipt_path)
    if (receipt.get("fit_id") != fit_id
            or receipt.get("freeze_signature") != frozen["freeze_signature"]
            or receipt.get("artifact_sha256") != _sha256_file(artifact_path)
            or receipt.get("source_commit") != frozen["source_commit"]):
        raise LearnedPairPilotError(f"Fit receipt lineage/hash mismatch: {fit_id}")
    optimizer = receipt.get("optimizer", {})
    n_fit = int(receipt.get("fit_population", {}).get("rows", 0))
    if receipt.get("optimizer_converged"):
        if (not optimizer.get("success") or not optimizer.get("finite_state")
                or not optimizer.get("stationarity_guard")
                or not optimizer.get("loss_nonincrease_guard")
                or n_fit <= 0
                or abs(float(optimizer.get("gtol", float("nan"))) - n_fit * 1e-4) > 1e-12
                or float(optimizer.get("verified_normalized_gradient_inf_norm", float("inf"))) > 1e-4
                or float(optimizer.get("final_loss_sum", float("inf")))
                   > float(optimizer.get("initial_loss_sum", float("nan"))) + 1e-8):
            raise LearnedPairPilotError(f"Fit receipt does not satisfy the V2 optimizer gate: {fit_id}")
    return receipt


def _resume_started_fit(fit_id: str, frozen: Mapping[str, Any], events: list[dict[str, Any]], *,
                        ledger_path: Path = LEDGER, run_root: Path = RUN_ROOT,
                        official: bool = True, artifact_path: Path | None = None,
                        receipt_path: Path | None = None) -> str:
    starts = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("fit_id") == fit_id]
    terminals = [e for e in events if e.get("event") in {"FIT_SAVED", "FIT_FAILED"} and e.get("fit_id") == fit_id]
    if len(starts) != 1 or len(terminals) > 1:
        raise LearnedPairPilotError(f"Inconsistent fit lifecycle for {fit_id}")
    if terminals:
        if terminals[0]["event"] == "FIT_FAILED":
            return "failed_consumed_no_retry"
        artifact = artifact_path or (run_root / "units" / f"{fit_id}.npz")
        receipt = receipt_path or (run_root / "receipts" / f"{fit_id}.json")
        _validate_receipt(fit_id, frozen, artifact_path=artifact, receipt_path=receipt)
        return "already_saved"
    artifact = artifact_path or (run_root / "units" / f"{fit_id}.npz")
    receipt_path = receipt_path or (run_root / "receipts" / f"{fit_id}.json")
    try:
        receipt = _validate_receipt(fit_id, frozen, artifact_path=artifact, receipt_path=receipt_path)
    except Exception as exc:
        failure_path = run_root / "failures" / f"{fit_id}.json"
        failure_record = {
            "fit_id": fit_id, "freeze_signature": frozen["freeze_signature"],
            "source_commit": frozen["source_commit"], "error_type": type(exc).__name__,
            "error": str(exc), "interrupted_before_valid_durable_receipt": True,
            "started_event": starts[0], "artifact_present": artifact.is_file(),
            "receipt_present": receipt_path.is_file(),
        }
        _atomic_json(failure_path, failure_record)
        terminal_receipt = {
            "schema_version": 1, "card_id": CARD_ID, "fit_id": fit_id,
            "fit_kind": "PREDICTOR", "optimizer_converged": False,
            "freeze_signature": frozen["freeze_signature"], "source_commit": frozen["source_commit"],
            "source_hashes": frozen["source_hashes"], "input_sha256": frozen["input_sha256"],
            "fit_population": starts[0].get("fit_population", {}), "recipe": frozen["recipe"],
            "failure": failure_record,
            "artifact_sha256": _sha256_file(artifact) if artifact.is_file() else None,
            "failure_record_sha256": _sha256_file(failure_path),
        }
        _atomic_json(receipt_path, terminal_receipt)
        _append_ledger({
            "event": "FIT_FAILED", "batch_id": "optimization_v3_research_next_v1",
            "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
            "fit_id": fit_id, "fit_kind": "PREDICTOR", "amendment_id": AMENDMENT_ID,
            "failure_receipt": receipt_path.relative_to(ROOT).as_posix(),
            "receipt_sha256": _sha256_file(receipt_path),
            "error": f"interrupted_or_corrupt_no_retry: {type(exc).__name__}: {exc}",
        }, path=ledger_path, official=official)
        return "failed_interrupted_no_retry"
    if not receipt.get("optimizer_converged", False):
        event = "FIT_FAILED"
        status = "failed_optimizer_no_retry"
    else:
        event = "FIT_SAVED"
        status = "resumed_from_durable_receipt"
    _append_ledger({
        "event": event, "batch_id": "optimization_v3_research_next_v1",
        "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
        "fit_id": fit_id, "cutoff": receipt["origin"], "fit_kind": "PREDICTOR",
        "amendment_id": AMENDMENT_ID, "receipt_sha256": _sha256_file(receipt_path),
        "resumed_from_durable_receipt": True,
    }, path=ledger_path, official=official)
    return status


def _release_unstarted(frozen: Mapping[str, Any], why: str, *, ledger_path: Path = LEDGER,
                       official: bool = True) -> dict[str, Any] | None:
    events = _ledger_events(ledger_path)
    own = _own_events(events)
    started_ids = {e["fit_id"] for e in own["FIT_STARTED"]}
    unstarted = _unstarted_fit_ids(FIT_IDS, started_ids)
    prior = [e for e in events if e.get("event") == "RESERVATION_UNUSED_RELEASED"
             and e.get("reservation_key") == RESERVATION_KEY]
    if prior:
        expected = sum(int(e.get("unused_fits", 0)) for e in prior)
        if expected != len(unstarted):
            raise LearnedPairPilotError("Prior unused-fit release does not match unstarted IDs")
        return prior[-1]
    if not unstarted:
        return None
    if official:
        _, snapshot = _check_live(freeze_signature=frozen["freeze_signature"])
        if snapshot["fit_pending"]:
            raise LearnedPairPilotError("Cannot release fits while a fit is pending")
    else:
        terminals = {e.get("fit_id") for e in own["FIT_SAVED"] + own["FIT_FAILED"]}
        if any(fit_id not in terminals for fit_id in started_ids):
            raise LearnedPairPilotError("Cannot release synthetic slots while a synthetic fit is pending")
    event = {
        "event": "RESERVATION_UNUSED_RELEASED", "batch_id": "optimization_v3_research_next_v1",
        "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
        "amendment_id": AMENDMENT_ID, "unused_fits": len(unstarted),
        "fit_ids": unstarted, "freeze_signature": frozen["freeze_signature"],
        "reason": why,
    }
    return _append_ledger(event, path=ledger_path, official=official)


def _unstarted_fit_ids(fit_ids: tuple[str, ...], started_ids: set[str]) -> list[str]:
    return [fit_id for fit_id in fit_ids if fit_id not in started_ids]


def _write_failure_checkpoint(frozen: Mapping[str, Any], reason: str) -> dict[str, Any]:
    completions = [event for event in _ledger_events()
                   if event.get("event") == "CARD_COMPLETED" and event.get("card_id") == CARD_ID]
    if len(completions) > 1:
        raise LearnedPairPilotError("Duplicate terminal failure/completion ledger rows")
    if completions:
        result_path = RUN_ROOT / "RESULTS.json"
        if not result_path.is_file() or _sha256_file(result_path) != completions[0].get("result_sha256"):
            raise LearnedPairPilotError("Existing failure checkpoint does not match CARD_COMPLETED receipt")
        return _read_json(result_path)
    _release_unstarted(frozen, reason)
    events, snapshot = _check_live(freeze_signature=frozen["freeze_signature"])
    own = _own_events(events)
    result = {
        "schema_version": 1, "card_id": CARD_ID,
        "status": "FAIL_OPTIMIZER_INCONCLUSIVE" if own["FIT_FAILED"] else "FAIL_INTERRUPTION_INCONCLUSIVE",
        "query_labels_opened": False, "predictive_gates_evaluated": False,
        "qualified_candidate": False, "champion_change": False, "upload_authorized": False,
        "source_commit": frozen["source_commit"], "freeze_signature": frozen["freeze_signature"],
        "input_sha256": frozen["input_sha256"], "fit_ids": list(FIT_IDS),
        "fit_saved_ids": [e["fit_id"] for e in own["FIT_SAVED"]],
        "fit_failed_ids": [e["fit_id"] for e in own["FIT_FAILED"]],
        "ledger_snapshot": snapshot, "ledger_sha256": _sha256_file(LEDGER),
        "interpretation": "At least one official predictor did not converge or was interrupted. Its consumed ID was not retried; remaining unstarted IDs were released. No outer/query target was opened and this is inconclusive about the interaction mechanism.",
        "reason": reason,
    }
    result_path = RUN_ROOT / "RESULTS.json"
    _atomic_json(result_path, result)
    _atomic_json(REPORT_ROOT / "RESULTS.json", result)
    _atomic_bytes(REPORT_ROOT / "RESULTS.md", (
        "# Learned pair pilot — optimizer checkpoint\n\n"
        f"**Status:** `{result['status']}`. No query target opened; no predictive gate evaluated.\n\n"
        f"{result['interpretation']}\n\nReason: `{reason}`.\n"
    ).encode("utf-8"))
    _append_ledger({
        "event": "CARD_COMPLETED", "batch_id": "optimization_v3_research_next_v1",
        "card_id": CARD_ID, "stage": STAGE, "freeze_signature": frozen["freeze_signature"],
        "status": result["status"], "fit_ids": list(FIT_IDS),
        "result_sha256": _sha256_file(result_path),
    }, path=LEDGER, official=True)
    return result


def _write_fit_terminal(fit_id: str, origin: int, frozen: Mapping[str, Any],
                        data: Mapping[str, Any], theta: np.ndarray, optimizer: Mapping[str, Any],
                        *, success: bool, initial_loss: float, final_loss: float,
                        final_gradient: np.ndarray, started_at: float,
                        resource_before: Mapping[str, Any], resource_after: Mapping[str, Any],
                        rss_samples: Mapping[str, Any]) -> dict[str, Any]:
    candidate = fit_id.endswith("_FM")
    fit_logits, pair_scores = score_logits(theta, data["x_fit"], rank=RANK,
                                           with_factors=candidate, chunk_rows=CHUNK_ROWS)
    w, b, v = unpack_parameters(theta, data["x_fit"].shape[1], RANK, with_factors=candidate)
    artifact_path, receipt_path = _fit_paths(fit_id)
    arrays = {
        "parameters": np.asarray(theta, dtype=np.float64),
        "fit_logits": np.asarray(fit_logits, dtype=np.float64),
        "fit_row_ids": np.asarray(data["fit_ids"]).astype(str),
        "fit_cycles": np.asarray(data["fit_cycles"], dtype=np.int64),
        "class_order": np.asarray([0, 1], dtype=np.int8),
    }
    if pair_scores is not None:
        arrays["fit_pair_scores"] = np.asarray(pair_scores, dtype=np.float64)
    _atomic_npz(artifact_path, **arrays)
    init_theta = initialize_parameters(data["x_fit"].shape[1], RANK, seed=42,
                                       with_factors=candidate)
    init_v = init_theta[data["x_fit"].shape[1] + 1:] if candidate else np.empty(0)
    final_v = theta[data["x_fit"].shape[1] + 1:] if candidate else np.empty(0)
    opt = dict(optimizer)
    opt.update({
        "success": bool(success), "initial_loss_sum": float(initial_loss),
        "final_loss_sum": float(final_loss),
        "final_gradient_inf_norm_sum": float(np.max(np.abs(final_gradient))),
        "final_gradient_inf_norm_mean": float(np.max(np.abs(final_gradient)) / len(data["y"])),
        "factor_initial_norm": float(np.linalg.norm(init_v)),
        "factor_final_norm": float(np.linalg.norm(final_v)),
        "factor_change_norm": float(np.linalg.norm(final_v - init_v)),
        "pair_score_fit_mean": float(np.mean(pair_scores)) if pair_scores is not None else 0.0,
        "pair_score_fit_std": float(np.std(pair_scores)) if pair_scores is not None else 0.0,
        "pair_score_fit_max_abs": float(np.max(np.abs(pair_scores))) if pair_scores is not None else 0.0,
        "parameter_w_norm": float(np.linalg.norm(w)), "intercept": float(b),
    })
    receipt = {
        "schema_version": 1, "card_id": CARD_ID, "fit_id": fit_id,
        "origin": origin, "component": "FM_RANK16" if candidate else "LOGISTIC_MAIN_ONLY",
        "fit_kind": "PREDICTOR", "optimizer_converged": bool(success),
        "freeze_signature": frozen["freeze_signature"], "source_commit": frozen["source_commit"],
        "source_hashes": frozen["source_hashes"], "input_sha256": frozen["input_sha256"],
        "fit_population": {
            "rows": len(data["y"]), "record_ids_sha256": offset_inputs.sha256_array(data["fit_ids"]),
            "row_indices_sha256": offset_inputs.sha256_array(data["fit_rows"]),
            "fit_target_sha256": data["fit_target_sha256"],
            "matrix_manifest_sha256": data["matrix_manifest_sha256"],
            "target_read_purpose": "fit-prefix-only", "class_order": [0, 1],
        },
        "matrix": {
            "fit_base_archive_sha256": frozen["saved_inputs"]["origins"][f"E{origin}"]["fit_matrix"]["archive_sha256"],
            "fit_base_logical_sha256": frozen["saved_inputs"]["origins"][f"E{origin}"]["fit_matrix"]["logical_sha256"],
            "fit_base_shape": list(data["x_fit"].shape),
            "score_base_archive_sha256": frozen["saved_inputs"]["origins"][f"E{origin}"]["score_matrix"]["archive_sha256"],
        },
        "recipe": frozen["recipe"], "optimizer": opt,
        "artifact_path": artifact_path.relative_to(ROOT).as_posix(),
        "artifact_sha256": _sha256_file(artifact_path),
        "raw_fit_logits_sha256": offset_inputs.sha256_array(fit_logits),
        "raw_fit_pair_scores_sha256": offset_inputs.sha256_array(pair_scores) if pair_scores is not None else None,
        "resource_before": dict(resource_before), "resource_after": dict(resource_after),
        "resource_during_fit_sample": dict(rss_samples),
        "elapsed_seconds": float(time.perf_counter() - started_at),
    }
    _atomic_json(receipt_path, receipt)
    receipt_sha = _sha256_file(receipt_path)
    return {"artifact_path": artifact_path, "receipt_path": receipt_path,
            "artifact_sha256": _sha256_file(artifact_path), "receipt_sha256": receipt_sha,
            "receipt": receipt}


def _fit_one(fit_id: str, frozen: Mapping[str, Any]) -> dict[str, Any]:
    if fit_id not in FIT_IDS:
        raise LearnedPairPilotError(f"Unknown fit ID: {fit_id}")
    origin = 67 if "E67" in fit_id else 75
    data = _load_origin_for_fit(frozen, origin)
    if (not np.isfinite(data["x_fit"].data).all()
            or not np.isfinite(data["x_score"].data).all()):
        raise LearnedPairPilotError(f"E{origin} matrix has non-finite values")
    resource_before = _resource_snapshot(1024**3, initial=False)
    candidate = fit_id.endswith("_FM")
    solver_options = _solver_options(len(data["y"]))
    theta0 = initialize_parameters(data["x_fit"].shape[1], RANK, seed=42,
                                   factor_std=0.01, with_factors=candidate)
    events, snapshot = _check_live(freeze_signature=frozen["freeze_signature"])
    own = _own_events(events)
    terminals = {e["fit_id"] for e in own["FIT_SAVED"] + own["FIT_FAILED"]}
    if snapshot["fit_pending"] or len(own["FIT_STARTED"]) >= 4 or fit_id in {e["fit_id"] for e in own["FIT_STARTED"]}:
        raise LearnedPairPilotError("Ledger is not ready for this exact next fit")
    expected_next = next((item for item in FIT_IDS if item not in {e["fit_id"] for e in own["FIT_STARTED"]}), None)
    if expected_next != fit_id:
        raise LearnedPairPilotError(f"Next authorized fit is {expected_next}, not {fit_id}")
    fit_record = {
        "rows": len(data["y"]), "record_ids_sha256": offset_inputs.sha256_array(data["fit_ids"]),
        "row_indices_sha256": offset_inputs.sha256_array(data["fit_rows"]),
        "fit_target_sha256": data["fit_target_sha256"],
        "matrix_manifest_sha256": data["matrix_manifest_sha256"],
        "origin": origin, "component": "FM_RANK16" if candidate else "LOGISTIC_MAIN_ONLY",
        "target_read_purpose": "fit-prefix-only",
    }
    _append_ledger({
        "event": "FIT_STARTED", "batch_id": "optimization_v3_research_next_v1",
        "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
        "fit_id": fit_id, "cutoff": origin, "component": fit_record["component"],
        "fit_kind": "PREDICTOR", "fit_consumed_at_start": True,
        "amendment_id": AMENDMENT_ID, "freeze_signature": frozen["freeze_signature"],
        "source_commit": frozen["source_commit"], "resource_snapshot": resource_before,
        "fit_population": fit_record,
    }, path=LEDGER, official=True)
    started = time.perf_counter()
    result: Any = None
    initial_loss = float("nan")
    final_loss = float("nan")
    final_grad = np.full_like(theta0, np.nan)
    opt_details: dict[str, Any] = {"method": "L-BFGS-B", **solver_options,
                                   "class_order": [0, 1], "initialization_seed": 42,
                                   "factor_std": 0.01, "chunk_rows": CHUNK_ROWS}
    try:
        initial_loss = float("nan")
        with _rss_sampler() as rss_samples, threadpool_limits(limits=2):
            initial_loss, _ = objective_gradient_sum(
                theta0, data["x_fit"], data["y"], rank=RANK,
                with_factors=candidate, chunk_rows=CHUNK_ROWS, resource_check=_resource_check)
            result = minimize(
                lambda parameters: objective_gradient_sum(
                    parameters, data["x_fit"], data["y"], rank=RANK,
                    with_factors=candidate, chunk_rows=CHUNK_ROWS,
                    resource_check=_resource_check),
                theta0, method="L-BFGS-B", jac=True, options=solver_options,
            )
            final_loss, final_grad = objective_gradient_sum(
                result.x, data["x_fit"], data["y"], rank=RANK,
                with_factors=candidate, chunk_rows=CHUNK_ROWS, resource_check=_resource_check)
        opt_details.update({
            "status_code": int(result.status), "message": str(result.message),
            "nit": int(result.nit), "nfev": int(result.nfev),
            "njev": int(getattr(result, "njev", result.nfev)),
            "solver_fun_sum": float(result.fun), "solver_gradient_inf_norm_sum": float(np.max(np.abs(result.jac))),
            "solver_gradient_inf_norm_mean": float(np.max(np.abs(result.jac)) / len(data["y"])),
            "verified_final_loss_sum": float(final_loss),
            "verified_final_gradient_inf_norm_sum": float(np.max(np.abs(final_grad))),
        })
        fitted_logits, _ = score_logits(np.asarray(result.x), data["x_fit"], rank=RANK,
                                        with_factors=candidate, chunk_rows=CHUNK_ROWS)
        normalized_gradient = float(np.max(np.abs(final_grad)) / len(data["y"]))
        finite_state = (np.isfinite(result.x).all() and np.isfinite(fitted_logits).all()
                        and np.isfinite(final_loss)
                        and np.isfinite(final_grad).all()
                        and np.asarray(result.jac).shape == final_grad.shape
                        and np.isfinite(np.asarray(result.jac)).all()
                        and np.isfinite(np.asarray(result.fun)).all())
        convergence = _optimizer_convergence_gate(
            solver_success=bool(result.success), finite_state=bool(finite_state),
            initial_loss=initial_loss, final_loss=final_loss,
            gradient_inf_sum=float(np.max(np.abs(final_grad))), n_fit=len(data["y"]))
        converged = convergence["optimizer_converged"]
        opt_details["success"] = bool(result.success)
        opt_details["finite_state"] = bool(finite_state)
        opt_details["verified_normalized_gradient_inf_norm"] = normalized_gradient
        opt_details["stationarity_guard"] = convergence["stationarity_guard"]
        opt_details["loss_nonincrease_guard"] = convergence["loss_nonincrease_guard"]
        resource_after = _resource_snapshot(0)
        with threadpool_limits(limits=2):
            durable = _write_fit_terminal(
                fit_id, origin, frozen, data, np.asarray(result.x, dtype=np.float64), opt_details,
                success=converged, initial_loss=initial_loss, final_loss=final_loss,
                final_gradient=final_grad, started_at=started, resource_before=resource_before,
                resource_after=resource_after, rss_samples=rss_samples,
            )
        event_name = "FIT_SAVED" if converged else "FIT_FAILED"
        _append_ledger({
            "event": event_name, "batch_id": "optimization_v3_research_next_v1",
            "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
            "fit_id": fit_id, "cutoff": origin, "fit_kind": "PREDICTOR",
            "amendment_id": AMENDMENT_ID, "receipt_sha256": durable["receipt_sha256"],
            "failure": None if converged else "optimizer_not_converged_or_guard_failed",
        }, path=LEDGER, official=True)
        outcome = {"fit_id": fit_id, "status": "saved" if converged else "failed_optimizer_no_retry",
                   "receipt_sha256": durable["receipt_sha256"], "optimizer": durable["receipt"]["optimizer"]}
        if not converged:
            _write_failure_checkpoint(frozen, f"{fit_id}: {result.message}; nfev={result.nfev}; success={result.success}")
        return outcome
    except Exception as exc:
        # The ID is consumed once FIT_STARTED exists; persist a terminal diagnostic and never retry it.
        artifact_path, receipt_path = _fit_paths(fit_id)
        failure_path = RUN_ROOT / "failures" / f"{fit_id}.json"
        failure_record = {
            "fit_id": fit_id, "origin": origin, "freeze_signature": frozen["freeze_signature"],
            "error_type": type(exc).__name__, "error": str(exc), "source_commit": frozen["source_commit"],
            "artifact_present": artifact_path.is_file(), "receipt_present": receipt_path.is_file(),
            "optimizer": opt_details,
        }
        _atomic_json(failure_path, failure_record)
        failure_receipt = {
            "schema_version": 1, "card_id": CARD_ID, "fit_id": fit_id, "origin": origin,
            "fit_kind": "PREDICTOR", "optimizer_converged": False,
            "freeze_signature": frozen["freeze_signature"], "source_commit": frozen["source_commit"],
            "source_hashes": frozen["source_hashes"], "input_sha256": frozen["input_sha256"],
            "fit_population": fit_record, "recipe": frozen["recipe"],
            "optimizer": opt_details, "failure": failure_record,
            "artifact_path": artifact_path.relative_to(ROOT).as_posix(),
            "artifact_sha256": _sha256_file(artifact_path) if artifact_path.is_file() else None,
            "failure_record_sha256": _sha256_file(failure_path),
        }
        _atomic_json(receipt_path, failure_receipt)
        failure_receipt_sha = _sha256_file(receipt_path)
        current_events = _ledger_events()
        terminal = [e for e in current_events if e.get("event") in {"FIT_SAVED", "FIT_FAILED"} and e.get("fit_id") == fit_id]
        if not terminal:
            _append_ledger({
                "event": "FIT_FAILED", "batch_id": "optimization_v3_research_next_v1",
                "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
                "fit_id": fit_id, "cutoff": origin, "fit_kind": "PREDICTOR",
                "amendment_id": AMENDMENT_ID, "failure_receipt": receipt_path.relative_to(ROOT).as_posix(),
                "receipt_sha256": failure_receipt_sha,
                "error": f"{type(exc).__name__}: {exc}",
            }, path=LEDGER, official=True)
        elif terminal[0]["event"] == "FIT_SAVED":
            return {"fit_id": fit_id, "status": "saved", "receipt_sha256": terminal[0].get("receipt_sha256"),
                    "note": "FIT_SAVED was already durable; no additional terminal event appended"}
        _write_failure_checkpoint(frozen, f"{fit_id}: {type(exc).__name__}: {exc}")
        return {"fit_id": fit_id, "status": "failed_consumed_no_retry", "error": str(exc)}


def fit_next() -> dict[str, Any]:
    lock_path = RUN_ROOT / "FIT_RUN.lock"
    with offset_inputs._fit_run_lock(lock_path):
        frozen = load_freeze()
        events, snapshot = _check_live(freeze_signature=frozen["freeze_signature"], allow_pending="*")
        own = _own_events(events)
        pending = [e for e in own["FIT_STARTED"]
                   if e["fit_id"] not in {x["fit_id"] for x in own["FIT_SAVED"] + own["FIT_FAILED"]}]
        if len(pending) > 1:
            raise LearnedPairPilotError("More than one fit is pending; stop for reconciliation")
        if pending:
            resumed = _resume_started_fit(pending[0]["fit_id"], frozen, events)
            if resumed.startswith("failed"):
                return _write_failure_checkpoint(frozen, resumed)
            return {"fit_id": pending[0]["fit_id"], "status": resumed, "optimizer_rerun": False}
        if own["FIT_FAILED"]:
            return _write_failure_checkpoint(frozen, "prior fit has terminal failure; retries and later fits prohibited")
        started_ids = {event["fit_id"] for event in own["FIT_STARTED"]}
        fit_id = next((item for item in FIT_IDS if item not in started_ids), None)
        if fit_id is None:
            return {"status": "all_four_fits_terminal", "snapshot": snapshot}
        return _fit_one(fit_id, frozen)


def _load_origin_unlabeled(frozen: Mapping[str, Any], origin: int) -> dict[str, Any]:
    current = _verify_saved_inputs()
    if current != frozen["saved_inputs"]:
        raise LearnedPairPilotError("Saved matrix/prediction sources changed after freeze")
    frame = offset_inputs._read_identity_frame()
    manifest_path, pred_path, _ = _origin_files(origin)
    manifest = _read_json(manifest_path)
    with np.load(pred_path, allow_pickle=False) as archive:
        pred = {key: archive[key].copy() for key in archive.files}
    score_rows = np.asarray(pred["score_rows"], dtype=np.int64)
    score_ids = frame.iloc[score_rows]["record_id"].astype(str).to_numpy()
    score_cycles = frame.iloc[score_rows]["cycle_code"].to_numpy(dtype=np.int64)
    if (not np.array_equal(score_ids, pred["score_record_ids"].astype(str))
            or not np.array_equal(score_cycles, pred["score_cycles"].astype(np.int64))):
        raise LearnedPairPilotError(f"E{origin} score identity mismatch")
    matrix = sparse.csr_matrix(offset_inputs._load_matrix(manifest, "score_base"), dtype=np.float64)
    if matrix.shape[0] != len(score_rows):
        raise LearnedPairPilotError(f"E{origin} score matrix row count mismatch")
    return {"manifest": manifest, "manifest_path": manifest_path, "prediction": pred,
            "score_rows": score_rows, "score_ids": score_ids, "score_cycles": score_cycles,
            "x_score": matrix}


def build_predictions() -> dict[str, Any]:
    frozen = load_freeze()
    events, snapshot = _check_live(freeze_signature=frozen["freeze_signature"])
    own = _own_events(events)
    if (len(own["FIT_STARTED"]) != 4 or len(own["FIT_SAVED"]) != 4
            or own["FIT_FAILED"] or snapshot["fit_pending"]):
        raise LearnedPairPilotError("All four learned-pair and no-pair predictors must converge before scoring")
    outputs: dict[str, Any] = {}
    for origin in ORIGINS:
        data = _load_origin_unlabeled(frozen, origin)
        pred = data["prediction"]
        present = np.asarray(pred["present_mask"], dtype=bool)
        fallback = np.asarray(pred["prob_missing_hgb"], dtype=np.float64)
        bag_raw = np.asarray(pred["prob_bag"], dtype=np.float64)
        bag = _validate_cached_bag_route(
            bag_raw, np.asarray(pred["prob_route_bag"], dtype=np.float64), present, fallback)
        probabilities: dict[str, np.ndarray] = {"bag": bag.copy()}
        a02_ref_meta = frozen["saved_inputs"]["origins"][f"E{origin}"]["a02_observed_reference"]
        if a02_ref_meta.get("status") != "ALIGNED_VERIFIED_REFERENCE_ONLY":
            raise LearnedPairPilotError(f"E{origin} exact-origin A02 reference was not verified before freeze")
        a02_path = ROOT / a02_ref_meta["path"]
        if _sha256_file(a02_path) != a02_ref_meta["bundle_sha256"]:
            raise LearnedPairPilotError(f"E{origin} A02 cached reference changed after freeze")
        with np.load(a02_path, allow_pickle=False) as a02_archive:
            probabilities["a02_observed"] = np.asarray(a02_archive["prediction"], dtype=np.float64)
        raw_scores: dict[str, np.ndarray] = {}
        pair_scores: dict[str, np.ndarray] = {}
        fit_receipts: dict[str, Any] = {}
        for suffix in ("FM", "NOPAIR"):
            fit_id = f"{CARD_ID}_E{origin}_{suffix}"
            artifact, receipt_path = _fit_paths(fit_id)
            receipt = _validate_receipt(fit_id, frozen, artifact_path=artifact, receipt_path=receipt_path)
            if not receipt.get("optimizer_converged"):
                raise LearnedPairPilotError(f"{fit_id} did not converge")
            with np.load(artifact, allow_pickle=False) as saved:
                theta = np.asarray(saved["parameters"], dtype=np.float64).copy()
            with_factors = suffix == "FM"
            with threadpool_limits(limits=2):
                raw, pair = score_logits(theta, data["x_score"], rank=RANK,
                                         with_factors=with_factors, chunk_rows=CHUNK_ROWS)
            probability_direct = expit(raw)
            if probability_direct.shape != present.shape or not np.isfinite(probability_direct).all():
                raise LearnedPairPilotError(f"{fit_id} produced invalid score probabilities")
            routed = _route_candidate_probabilities(probability_direct, present, fallback)
            key = "fm" if with_factors else "nopair"
            probabilities[key] = routed
            raw_scores[key] = raw
            pair_scores[key] = pair if pair is not None else np.zeros_like(raw)
            fit_receipts[fit_id] = {
                "receipt_sha256": _sha256_file(receipt_path),
                "artifact_sha256": _sha256_file(artifact),
                "optimizer": receipt["optimizer"],
            }
        payload = {
            "origin": np.asarray([origin], dtype=np.int64),
            "score_rows": data["score_rows"], "score_record_ids": data["score_ids"],
            "score_cycles": data["score_cycles"], "present_mask": present,
            "prob_bag": probabilities["bag"], "prob_fm": probabilities["fm"],
            "prob_nopair": probabilities["nopair"],
            "prob_bag_raw": bag_raw,
            "prob_missing_hgb": fallback,
            "raw_score_fm": raw_scores["fm"], "raw_score_nopair": raw_scores["nopair"],
            "pair_score_fm": pair_scores["fm"],
        }
        payload["prob_a02_observed"] = probabilities["a02_observed"]
        dest = RUN_ROOT / f"E{origin}" / f"predictions_E{origin}.npz"
        _atomic_npz(dest, **payload)
        outputs[f"E{origin}"] = {
            "path": dest.relative_to(ROOT).as_posix(), "sha256": _sha256_file(dest),
            "score_rows": len(data["score_rows"]), "fit_receipts": fit_receipts,
        }
    ready = {
        "schema_version": 1, "card_id": CARD_ID,
        "source_commit": frozen["source_commit"], "freeze_signature": frozen["freeze_signature"],
        "input_sha256": frozen["input_sha256"], "query_labels_opened": False,
        "predictions": outputs, "fit_ids": list(FIT_IDS),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(RUN_ROOT / "PREDICTIONS_READY.json", ready)
    return ready


def verify_prelabel() -> dict[str, Any]:
    frozen = load_freeze()
    ready_path = RUN_ROOT / "PREDICTIONS_READY.json"
    if not ready_path.is_file():
        raise LearnedPairPilotError("Durable prediction readiness manifest missing")
    ready = _read_json(ready_path)
    if (ready.get("freeze_signature") != frozen["freeze_signature"]
            or ready.get("query_labels_opened") is not False
            or set(ready.get("predictions", {})) != {"E67", "E75"}):
        raise LearnedPairPilotError("Prediction readiness manifest lineage mismatch")
    completed, snapshot = _check_live(freeze_signature=frozen["freeze_signature"])
    own = _own_events(completed)
    if len(own["FIT_SAVED"]) != 4 or own["FIT_FAILED"] or snapshot["fit_pending"]:
        raise LearnedPairPilotError("All four successful fit receipts must be ledger-complete before independent verification")
    command = [sys.executable, str(VERIFY_SCRIPT), "--official", "--run-root", str(RUN_ROOT)]
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        raise LearnedPairPilotError(f"Independent prelabel verifier failed: {proc.stderr or proc.stdout}")
    verified = json.loads(proc.stdout)
    if verified.get("status") != "PASS":
        raise LearnedPairPilotError("Independent prelabel verifier did not return PASS")
    verified["freeze_signature"] = frozen["freeze_signature"]
    verified["predictions_ready_sha256"] = _sha256_file(ready_path)
    verified["prediction_output_sha256s"] = {
        key: _sha256_file(ROOT / value["path"]) for key, value in ready["predictions"].items()
    }
    verified["budget_ledger_sha256"] = _sha256_file(LEDGER)
    verified["verification_signature"] = _canonical_sha256(verified)
    _atomic_json(RUN_ROOT / "PRELABEL_VERIFICATION.json", verified)
    return verified


def _assert_prelabel(verified: Mapping[str, Any], frozen: Mapping[str, Any]) -> dict[str, Any]:
    ready_path = RUN_ROOT / "PREDICTIONS_READY.json"
    ready = _read_json(ready_path)
    outputs = {key: _sha256_file(ROOT / value["path"]) for key, value in ready["predictions"].items()}
    body = dict(verified)
    sig = body.pop("verification_signature", None)
    if (verified.get("status") != "PASS"
            or verified.get("freeze_signature") != frozen["freeze_signature"]
            or verified.get("predictions_ready_sha256") != _sha256_file(ready_path)
            or verified.get("prediction_output_sha256s") != outputs
            or sig != _canonical_sha256(body)):
        raise LearnedPairPilotError("Query labels remain closed: prelabel proof is missing/stale")
    return ready


def _label(value: Any) -> str:
    return "<missing>" if pd.isna(value) else str(value)


def _metric(y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(probability, dtype=np.float64)
    if p.shape != y.shape or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise LearnedPairPilotError("Invalid metric probability vector")
    if not len(y):
        return {"n": 0, "positive": 0, "negative": 0, "macro_f1": None,
                "positive_f1": None, "precision": None, "positive_recall": None,
                "roc_auc": None, "average_precision": None, "log_loss": None,
                "brier": None, "confusion_matrix": [[0, 0], [0, 0]]}
    predicted = (p >= 0.5).astype(np.int8)
    return {
        "n": int(len(y)), "positive": int(y.sum()), "negative": int((1 - y).sum()),
        "macro_f1": float(f1_score(y, predicted, average="macro", labels=[0, 1], zero_division=0)),
        "positive_f1": float(f1_score(y, predicted, pos_label=1, zero_division=0)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "positive_recall": float(recall_score(y, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else None,
        "average_precision": float(average_precision_score(y, p)) if np.unique(y).size == 2 else None,
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(np.mean((p - y) ** 2)),
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
    }


def _view_masks(origin: int, cycles: np.ndarray) -> dict[str, np.ndarray]:
    if origin == 67:
        return {"W4": (cycles >= 68) & (cycles <= 71),
                "W5": (cycles >= 72) & (cycles <= 75),
                "W6": (cycles >= 76) & (cycles <= 79),
                "H12_E67": (cycles >= 68) & (cycles <= 79)}
    return {"W6": (cycles >= 76) & (cycles <= 79)}


def _supported(metric: Mapping[str, Any]) -> bool:
    return metric["n"] >= 200 and metric["positive"] >= 30 and metric["negative"] >= 30


def _validate_cached_bag_route(raw: np.ndarray, routed: np.ndarray,
                               present: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    routed = np.asarray(routed, dtype=np.float64)
    present = np.asarray(present, dtype=bool)
    fallback = np.asarray(fallback, dtype=np.float64)
    if (raw.ndim != 1 or routed.shape != raw.shape or present.shape != raw.shape
            or fallback.shape != raw.shape or not np.isfinite(raw).all()
            or not np.isfinite(routed).all() or not np.isfinite(fallback).all()
            or (raw < 0).any() or (raw > 1).any()
            or (routed < 0).any() or (routed > 1).any()
            or (fallback < 0).any() or (fallback > 1).any()):
        raise LearnedPairPilotError("Cached bag route inputs are invalid or misaligned")
    expected = raw.copy()
    expected[~present] = fallback[~present]
    if not np.array_equal(routed, expected):
        raise LearnedPairPilotError("Cached routed C2_B bag does not match raw-present/HGB-missing recipe")
    return routed.copy()


def _route_candidate_probabilities(direct: np.ndarray, present: np.ndarray,
                                  fallback: np.ndarray) -> np.ndarray:
    direct = np.asarray(direct, dtype=np.float64)
    present = np.asarray(present, dtype=bool)
    fallback = np.asarray(fallback, dtype=np.float64)
    if (direct.ndim != 1 or present.shape != direct.shape or fallback.shape != direct.shape
            or not np.isfinite(direct).all() or not np.isfinite(fallback).all()
            or (direct < 0).any() or (direct > 1).any()
            or (fallback < 0).any() or (fallback > 1).any()):
        raise LearnedPairPilotError("Candidate routing inputs are invalid or misaligned")
    routed = direct.copy()
    routed[~present] = fallback[~present]
    if not np.array_equal(routed[~present], fallback[~present]):
        raise LearnedPairPilotError("Candidate missing-stream fallback is not exact")
    return routed


def _meets_gate(value: float, threshold: float) -> bool:
    return bool(value >= threshold - 1e-12)


def _gates_for_origin(origin: int, views: list[dict[str, Any]],
                      groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_view = {(item["view"], item["arm"]): item for item in views}
    gates: list[dict[str, Any]] = []

    def delta(view: str, arm: str, control: str, metric: str) -> float:
        a = by_view[(view, arm)][metric]
        b = by_view[(view, control)][metric]
        if a is None or b is None:
            raise LearnedPairPilotError(f"Missing {metric} for {view}/{arm}/{control}")
        return float(a - b)

    if origin == 67:
        for control, minimum in (("bag", 0.0005), ("nopair", 0.0002)):
            value = delta("H12_E67", "fm", control, "macro_f1")
            gates.append({"track": "mechanism", "gate": f"E67_H12_gain_min_{minimum}_vs_{control}",
                          "origin": origin, "view": "H12_E67", "arm": "fm", "comparator": control,
                          "metric": "macro_f1", "value": value, "threshold": minimum,
                          "pass": _meets_gate(value, minimum)})
        value = delta("H12_E67", "fm", "a02_observed", "macro_f1")
        gates.append({"track": "competition", "gate": "E67_H12_gain_min_0.0005_vs_a02_observed",
                      "origin": origin, "view": "H12_E67", "arm": "fm", "comparator": "a02_observed",
                      "metric": "macro_f1", "value": value, "threshold": 0.0005,
                      "pass": _meets_gate(value, 0.0005)})
        for view in ("W4", "W5", "W6"):
            for control in ("bag", "nopair"):
                value = delta(view, "fm", control, "macro_f1")
                gates.append({"track": "mechanism", "gate": f"E67_{view}_harm_max_0.005_vs_{control}",
                              "origin": origin, "view": view, "arm": "fm", "comparator": control,
                              "metric": "macro_f1", "value": value, "threshold": -0.005,
                              "pass": _meets_gate(value, -0.005)})
            value = delta(view, "fm", "a02_observed", "macro_f1")
            gates.append({"track": "competition", "gate": f"E67_{view}_harm_max_0.005_vs_a02_observed",
                          "origin": origin, "view": view, "arm": "fm", "comparator": "a02_observed",
                          "metric": "macro_f1", "value": value, "threshold": -0.005,
                          "pass": _meets_gate(value, -0.005)})
    else:
        for control in ("bag", "nopair"):
            value = delta("W6", "fm", control, "macro_f1")
            gates.append({"track": "mechanism", "gate": f"E75_W6_gain_nonnegative_vs_{control}",
                          "origin": origin, "view": "W6", "arm": "fm", "comparator": control,
                          "metric": "macro_f1", "value": value, "threshold": 0.0,
                          "pass": _meets_gate(value, 0.0)})
        value = delta("W6", "fm", "a02_observed", "macro_f1")
        gates.append({"track": "competition", "gate": "E75_W6_gain_min_0.0005_vs_a02_observed",
                      "origin": origin, "view": "W6", "arm": "fm", "comparator": "a02_observed",
                      "metric": "macro_f1", "value": value, "threshold": 0.0005,
                      "pass": _meets_gate(value, 0.0005)})
    for item in groups:
        if item["arm"] != "fm":
            continue
        support = _supported(item)
        gates.append({"track": "both", "gate": f"{item['view']}_{item['group']}_{item['group_value']}_support",
                      "origin": origin, "view": item["view"], "group": item["group"],
                      "group_value": item["group_value"], "arm": "fm",
                      "value": {"n": item["n"], "positive": item["positive"], "negative": item["negative"]},
                      "threshold": {"n": 200, "each_class": 30}, "pass": True if support else None,
                      "status": "SUPPORTED" if support else "UNKNOWN_UNSUPPORTED"})
        if not support:
            continue
        control_item = {(r["view"], r["group"], r["group_value"], r["arm"]): r for r in groups}
        for control in ("bag", "nopair", "a02_observed"):
            track = "competition" if control == "a02_observed" else "mechanism"
            reference = control_item[(item["view"], item["group"], item["group_value"], control)]
            for key, maximum in (("macro_f1", 0.01), ("positive_recall", 0.02)):
                value = float(item[key] - reference[key])
                gates.append({"track": track,
                              "gate": f"{item['view']}_{item['group']}_{item['group_value']}_{key}_harm_vs_{control}",
                              "origin": origin, "view": item["view"], "group": item["group"],
                              "group_value": item["group_value"], "arm": "fm", "comparator": control,
                              "metric": key, "value": value, "threshold": -maximum,
                              "pass": _meets_gate(value, -maximum)})
    return gates


def score() -> dict[str, Any]:
    with offset_inputs._fit_run_lock(RUN_ROOT / "FIT_RUN.lock"):
        return _score_locked()


def _score_locked() -> dict[str, Any]:
    prior = [e for e in _ledger_events() if e.get("event") == "CARD_COMPLETED" and e.get("card_id") == CARD_ID]
    if len(prior) > 1:
        raise LearnedPairPilotError("Duplicate card completion events")
    if prior:
        path = RUN_ROOT / "RESULTS.json"
        if not path.is_file() or _sha256_file(path) != prior[0].get("result_sha256"):
            raise LearnedPairPilotError("Completed result receipt is missing or ambiguous")
        return _read_json(path)
    result_path = RUN_ROOT / "RESULTS.json"
    if result_path.is_file():
        verification_path = RUN_ROOT / "SCORED_VERIFICATION.json"
        if not verification_path.is_file():
            verified = _run_scored_verifier()
            _atomic_json(verification_path, verified)
        else:
            verified = _read_json(verification_path)
        verification_body = dict(verified)
        verification_signature = verification_body.pop("verification_signature", None)
        if (verified.get("status") != "PASS_INDEPENDENT_METRICS_AND_GATES"
                or verified.get("result_sha256") != _sha256_file(result_path)
                or verification_signature != _canonical_sha256(verification_body)):
            raise LearnedPairPilotError("Saved scored result lacks a matching independent verification")
        saved_result = _read_json(result_path)
        _append_ledger({
            "event": "CARD_COMPLETED", "batch_id": "optimization_v3_research_next_v1",
            "card_id": CARD_ID, "stage": STAGE,
            "freeze_signature": saved_result["freeze_signature"],
            "status": saved_result["status"], "fit_ids": list(FIT_IDS),
            "result_sha256": _sha256_file(result_path),
            "scored_verification_sha256": _sha256_file(verification_path),
        }, path=LEDGER, official=True)
        return saved_result
    if (RUN_ROOT / "QUERY_LABEL_READ.json").is_file():
        raise LearnedPairPilotError(
            "Query labels were opened but no durable result exists; stop for review instead of scoring again")
    frozen = load_freeze()
    verified = _read_json(RUN_ROOT / "PRELABEL_VERIFICATION.json")
    ready = _assert_prelabel(verified, frozen)
    events, snapshot = _check_live(freeze_signature=frozen["freeze_signature"])
    own = _own_events(events)
    if (snapshot["fit_starts"] != BASE_STARTS + 4
            or snapshot["fit_saved"] != BASE_SAVED + 4
            or snapshot["fit_failed"] != BASE_FAILED or snapshot["fit_pending"] != 0
            or len(own["FIT_SAVED"]) != 4 or own["FIT_FAILED"]):
        raise LearnedPairPilotError("All four successful predictors must be saved before opening outer labels")
    scored_identity: dict[str, Any] = {}
    for key, value in ready["predictions"].items():
        with np.load(ROOT / value["path"], allow_pickle=False) as saved_prediction:
            scored_identity[key] = {
                "score_rows_sha256": offset_inputs.sha256_array(saved_prediction["score_rows"]),
                "score_ids_sha256": offset_inputs.sha256_array(saved_prediction["score_record_ids"].astype(str)),
                "score_cycles_sha256": offset_inputs.sha256_array(saved_prediction["score_cycles"].astype(np.int64)),
            }
    access_payload = {
        "authorized_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "evaluation only after durable predictions and independent prelabel PASS",
        "prediction_ready_sha256": _sha256_file(RUN_ROOT / "PREDICTIONS_READY.json"),
        "prelabel_verification_sha256": _sha256_file(RUN_ROOT / "PRELABEL_VERIFICATION.json"),
        "train_sha256": frozen["input_sha256"],
        "columns_opened": ["record_id", "cycle_code", "artifact_signal", "text_load", "entry_friction"],
        "scored_populations": scored_identity,
    }
    access = _create_once_receipt(
        RUN_ROOT / "QUERY_LABEL_ACCESS.json", access_payload,
        ("purpose", "prediction_ready_sha256", "prelabel_verification_sha256",
         "train_sha256", "columns_opened", "scored_populations"),
    )
    # This is the first outer/query target read; all fit inputs and predictions are already durable.
    train = pd.read_csv(TRAIN, usecols=["record_id", "cycle_code", "artifact_signal", "text_load", "entry_friction"],
                        dtype={"record_id": "string"})
    read_payload = {
        "read_at_utc": datetime.now(timezone.utc).isoformat(),
        "query_label_access_sha256": _sha256_file(RUN_ROOT / "QUERY_LABEL_ACCESS.json"),
        "train_sha256": frozen["input_sha256"],
        "rows_read": int(len(train)),
        "columns_read": ["record_id", "cycle_code", "artifact_signal", "text_load", "entry_friction"],
    }
    _create_once_receipt(
        RUN_ROOT / "QUERY_LABEL_READ.json", read_payload,
        ("query_label_access_sha256", "train_sha256", "rows_read", "columns_read"),
    )
    origin_results: dict[str, Any] = {}
    all_gates: list[dict[str, Any]] = []
    for origin in ORIGINS:
        key = f"E{origin}"
        pred_path = ROOT / ready["predictions"][key]["path"]
        with np.load(pred_path, allow_pickle=False) as archive:
            pred = {name: archive[name].copy() for name in archive.files}
        rows = np.asarray(pred["score_rows"], dtype=np.int64)
        sub = train.iloc[rows]
        if not np.array_equal(sub["record_id"].astype(str).to_numpy(), pred["score_record_ids"].astype(str)):
            raise LearnedPairPilotError(f"E{origin} score IDs changed before metrics")
        y = pd.to_numeric(sub["artifact_signal"], errors="raise").to_numpy(dtype=np.int8)
        if not np.isin(y, (0, 1)).all():
            raise LearnedPairPilotError(f"E{origin} query labels are non-binary")
        cycles = pd.to_numeric(sub["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
        if "prob_a02_observed" not in pred:
            raise LearnedPairPilotError(f"E{origin} exact-origin A02 comparator missing from durable predictions")
        arms = {"bag": pred["prob_bag"], "nopair": pred["prob_nopair"],
                "a02_observed": pred["prob_a02_observed"], "fm": pred["prob_fm"]}
        views_map = _view_masks(origin, cycles)
        present = np.asarray(pred["present_mask"], dtype=bool)
        text_load = pd.to_numeric(sub["text_load"], errors="coerce").to_numpy(dtype=np.float64)
        manifest = _read_json(PILOT_ROOT / f"MATRIX_MANIFEST_E{origin}.json")
        q75 = float(manifest["text_load_q75_fit_only"])
        q4 = np.isfinite(text_load) & (text_load >= q75)
        group_masks: list[tuple[str, str, np.ndarray]] = [
            ("availability", "present", present),
            ("availability", "missing_or_blank", ~present),
            ("text_load", "Q4", q4),
            ("text_load", "below_Q4", np.isfinite(text_load) & ~q4),
            ("text_load", "missing", ~np.isfinite(text_load)),
        ]
        ef = sub["entry_friction"].to_numpy(dtype=object)
        for value in sorted({_label(v) for v in ef}):
            group_masks.append(("entry_friction", value,
                                np.asarray([_label(v) == value for v in ef], dtype=bool)))
        view_metrics: list[dict[str, Any]] = []
        subgroup_metrics: list[dict[str, Any]] = []
        per_cycle: list[dict[str, Any]] = []
        for view, vm in views_map.items():
            for arm, probability in arms.items():
                view_metrics.append({"view": view, "arm": arm, **_metric(y[vm], np.asarray(probability)[vm])})
            for group, value, gm in group_masks:
                for arm, probability in arms.items():
                    subgroup_metrics.append({"view": view, "group": group, "group_value": value,
                                             "arm": arm, **_metric(y[vm & gm], np.asarray(probability)[vm & gm])})
        for cycle in sorted(np.unique(cycles).tolist()):
            cm = cycles == cycle
            each = {arm: _metric(y[cm], np.asarray(probability)[cm]) for arm, probability in arms.items()}
            per_cycle.append({"cycle": int(cycle), "n": int(cm.sum()), "metrics": each,
                              "fm_minus_bag_macro_f1": each["fm"]["macro_f1"] - each["bag"]["macro_f1"],
                              "fm_minus_nopair_macro_f1": each["fm"]["macro_f1"] - each["nopair"]["macro_f1"],
                              "fm_minus_a02_observed_macro_f1": each["fm"]["macro_f1"] - each["a02_observed"]["macro_f1"]})
        gates = _gates_for_origin(origin, view_metrics, subgroup_metrics)
        all_gates.extend({"origin": origin, **gate} for gate in gates)
        origin_results[key] = {
            "score_rows": len(rows), "present_rows": int(present.sum()), "missing_or_blank_rows": int((~present).sum()),
            "fit_defined_text_load_q75": q75, "views": view_metrics,
            "subgroups": subgroup_metrics, "per_cycle": per_cycle, "gates": gates,
            "per_cycle_positive_gain_counts": {
                "fm_vs_bag": sum(row["fm_minus_bag_macro_f1"] > 0 for row in per_cycle),
                "fm_vs_nopair": sum(row["fm_minus_nopair_macro_f1"] > 0 for row in per_cycle),
                "fm_vs_a02_observed": sum(row["fm_minus_a02_observed_macro_f1"] > 0 for row in per_cycle),
                "cycles": len(per_cycle),
            },
            "per_cycle_gain_sign_counts": {
                comparator: {
                    "positive": int(sum(row[f"fm_minus_{comparator}_macro_f1"] > 0 for row in per_cycle)),
                    "negative": int(sum(row[f"fm_minus_{comparator}_macro_f1"] < 0 for row in per_cycle)),
                    "tie": int(sum(row[f"fm_minus_{comparator}_macro_f1"] == 0 for row in per_cycle)),
                }
                for comparator in ("bag", "nopair", "a02_observed")
            },
            "a02_observed_reference": frozen["saved_inputs"]["origins"][key]["a02_observed_reference"],
        }
    fit_receipts: dict[str, Any] = {}
    optimizer_pass = True
    for fit_id in FIT_IDS:
        artifact, receipt_path = _fit_paths(fit_id)
        receipt = _validate_receipt(fit_id, frozen, artifact_path=artifact, receipt_path=receipt_path)
        fit_receipts[fit_id] = {"receipt_sha256": _sha256_file(receipt_path),
                                "artifact_sha256": _sha256_file(artifact),
                                "optimizer": receipt["optimizer"]}
        optimizer_pass &= bool(receipt["optimizer_converged"])
    unsupported_groups = [gate for gate in all_gates if gate.get("status") == "UNKNOWN_UNSUPPORTED"]
    evaluable_gates = [gate for gate in all_gates if gate.get("pass") is not None]
    mechanism_gates = [gate for gate in all_gates if gate.get("track") == "mechanism"]
    competition_gates = [gate for gate in all_gates if gate.get("track") == "competition"]
    mechanism_pass = optimizer_pass and all(gate["pass"] is True for gate in mechanism_gates)
    competition_pass = (mechanism_pass
                        and all(gate["pass"] is True for gate in competition_gates))
    mechanism_verdict = ("MECHANISM_PASS" if mechanism_pass else
                         "MECHANISM_FAIL" if optimizer_pass else "OPTIMIZER_OR_INTEGRITY_FAILURE")
    competition_verdict = ("COMPETITION_PASS" if competition_pass else
                           "COMPETITION_FAIL" if optimizer_pass else "NOT_EVALUABLE_OPTIMIZER_OR_INTEGRITY_FAILURE")
    result = {
        "schema_version": 1, "card_id": CARD_ID,
        "status": "COMPLETED_DIAGNOSTIC_ONLY",
        "verdicts": {"mechanism": mechanism_verdict, "competition": competition_verdict},
        "adaptive_selection": True, "mechanism_pilot": True,
        "qualified_candidate": False, "champion_change": False, "upload_authorized": False,
        "source_commit": frozen["source_commit"], "freeze_signature": frozen["freeze_signature"],
        "input_sha256": frozen["input_sha256"],
        "query_labels_opened_after_prelabel_verification": True,
        "query_label_access_receipt_sha256": _sha256_file(RUN_ROOT / "QUERY_LABEL_ACCESS.json"),
        "query_label_read_receipt_sha256": _sha256_file(RUN_ROOT / "QUERY_LABEL_READ.json"),
        "prelabel_verification_sha256": _sha256_file(RUN_ROOT / "PRELABEL_VERIFICATION.json"),
        "predictions_ready_sha256": _sha256_file(RUN_ROOT / "PREDICTIONS_READY.json"),
        "ledger_snapshot": snapshot, "ledger_sha256": _sha256_file(LEDGER),
        "fit_receipts": fit_receipts, "optimizer_all_success": optimizer_pass,
        "gate_checks": all_gates, "mechanism_gate_checks": mechanism_gates,
        "competition_gate_checks": competition_gates,
        "mechanism_gates_pass": mechanism_pass, "competition_gates_pass": competition_pass,
        "evaluable_gates_pass": mechanism_pass,
        "unsupported_group_checks": unsupported_groups,
        "unsupported_group_check_count": len(unsupported_groups),
        "origins": origin_results,
        "interpretation": "Diagnostic-only, adaptively selected two-origin learned-factor pilot. E75 W6 overlaps E67 W6 and is history-sensitivity evidence, not independent replication. No verdict changes a champion or authorizes continuation, full-Train refit, Test inference, or upload.",
    }
    _atomic_json(result_path, result)
    report = _make_report(result)
    _atomic_json(REPORT_ROOT / "RESULTS.json", result)
    _atomic_bytes(REPORT_ROOT / "RESULTS.md", report.encode("utf-8"))
    verified = _run_scored_verifier()
    verification_path = RUN_ROOT / "SCORED_VERIFICATION.json"
    _atomic_json(verification_path, verified)
    _atomic_json(REPORT_ROOT / "SCORED_VERIFICATION.json", verified)
    _append_ledger({
        "event": "CARD_COMPLETED", "batch_id": "optimization_v3_research_next_v1",
        "card_id": CARD_ID, "stage": STAGE, "freeze_signature": frozen["freeze_signature"],
        "status": result["status"], "fit_ids": list(FIT_IDS),
        "result_sha256": _sha256_file(result_path),
        "scored_verification_sha256": _sha256_file(verification_path),
    }, path=LEDGER, official=True)
    return result


def _run_scored_verifier() -> dict[str, Any]:
    command = [sys.executable, str(VERIFY_SCRIPT), "--scored", "--run-root", str(RUN_ROOT)]
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        raise LearnedPairPilotError(f"Independent scored-metrics verifier failed: {proc.stderr or proc.stdout}")
    verified = json.loads(proc.stdout)
    verification_body = dict(verified)
    verification_signature = verification_body.pop("verification_signature", None)
    if verified.get("status") != "PASS_INDEPENDENT_METRICS_AND_GATES":
        raise LearnedPairPilotError("Independent scored-metrics verifier did not PASS")
    if verified.get("result_sha256") != _sha256_file(RUN_ROOT / "RESULTS.json"):
        raise LearnedPairPilotError("Independent scored-metrics verifier is not bound to the saved result")
    if verification_signature != _canonical_sha256(verification_body):
        raise LearnedPairPilotError("Independent scored-metrics verifier signature is invalid")
    return verified


def _make_report(result: Mapping[str, Any]) -> str:
    lines = [
        "# Learned-pair FM V2 — hasil diagnostic", "",
        f"**Status:** `{result['status']}`  ",
        f"**MECHANISM:** `{result['verdicts']['mechanism']}`  ",
        f"**COMPETITION:** `{result['verdicts']['competition']}`", "",
        "Kedua verdict dilaporkan terpisah. Hasil ini diagnostic-only dan tidak mengubah champion, tidak mengizinkan continuation/refit/Test, dan tidak mengizinkan submission.", "",
        "## Rancangan dan batas inferensi", "",
        "FM rank-16 memakai binary logistic SUM + penalti L2 pada main weights, factors, dan intercept. Kontrol no-pair memakai baris, matriks, objective, inisialisasi w/b, dan solver yang sama. Present stream memakai expert langsung; missing/blank memakai fallback C1_B HGB tersimpan secara bit-exact. A02 observed adalah comparator terpisah dan tidak menjadi input training maupun prediksi.",
        "V2 hanya mengubah `gtol` dari V1 menjadi `n_fit × 1e-4` pada objective SUM. Fit tetap harus memenuhi SciPy success, stationarity mean-gradient <=1e-4, finite state, dan objective nonincrease. V1 tetap `FAIL_OPTIMIZER_INCONCLUSIVE`; perubahan ini tidak memperbaiki atau menulis ulang verdict V1.",
        "E67 H12 adalah view agregat utama. E67 W4/W5/W6 dan E75 W6 adalah pemeriksaan temporal/history sensitivity; W6 E75 tumpang tindih dengan W6 E67, jadi bukan replikasi independen.", "",
        "## Metrik dan denominator", "",
        "Macro F1/precision/recall memakai threshold 0,50. Denominator, jumlah kelas, confusion matrix, dan probability metrics dihitung untuk setiap arm di setiap view.", "",
        "| Origin | View | Arm | n | Pos/Neg | Macro F1 | Pos recall | Precision | ROC AUC | AP | Log loss | Brier | Confusion [[TN,FP],[FN,TP]] |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for origin, block in result["origins"].items():
        for item in block["views"]:
            roc = f"{item['roc_auc']:.6f}" if item["roc_auc"] is not None else "NA"
            ap = f"{item['average_precision']:.6f}" if item["average_precision"] is not None else "NA"
            lines.append(f"| {origin} | {item['view']} | {item['arm']} | {item['n']} | {item['positive']}/{item['negative']} | {item['macro_f1']:.6f} | {item['positive_recall']:.6f} | {item['precision']:.6f} | {roc} | {ap} | {item['log_loss']:.6f} | {item['brier']:.6f} | {item['confusion_matrix']} |")
    lines.extend(["", "## Gate checks", "", "| Track | Origin | View | Comparator | Metric | Delta | Threshold | Result |", "|---|---:|---|---|---|---:|---:|---|"])
    for gate in result["gate_checks"]:
        value = gate["value"] if isinstance(gate["value"], (float, int)) else json.dumps(gate["value"], sort_keys=True)
        result_label = "PASS" if gate["pass"] is True else gate.get("status", "FAIL")
        lines.append(f"| {gate['track']} | {gate['origin']} | {gate.get('view', '')} | {gate.get('comparator', '')} | {gate.get('metric', 'support')} | {value} | {gate['threshold']} | {result_label} |")
    lines.extend(["", "### Per-cycle direction", "", "| Origin | Cycle | n | FM Δ vs C2_B | FM Δ vs no-pair | FM Δ vs A02 observed |", "|---|---:|---:|---:|---:|---:|"])
    for origin, block in result["origins"].items():
        for row in block["per_cycle"]:
            lines.append(f"| {origin} | {row['cycle']} | {row['n']} | {row['fm_minus_bag_macro_f1']:+.6f} | {row['fm_minus_nopair_macro_f1']:+.6f} | {row['fm_minus_a02_observed_macro_f1']:+.6f} |")
        signs = block["per_cycle_gain_sign_counts"]
        lines.append(f"\n{origin} cycle signs (positive / negative / tie): C2_B {signs['bag']['positive']}/{signs['bag']['negative']}/{signs['bag']['tie']}; no-pair {signs['nopair']['positive']}/{signs['nopair']['negative']}/{signs['nopair']['tie']}; A02 {signs['a02_observed']['positive']}/{signs['a02_observed']['negative']}/{signs['a02_observed']['tie']}.")
    lines.extend(["", "### Subgroup support", "", "Full subgroup metrics and all comparator deltas remain in `RESULTS.json`. A group is supported only for n>=200 and each class>=30; unsupported groups are `UNKNOWN_UNSUPPORTED`, with no numeric harm estimate.", ""])
    for origin, block in result["origins"].items():
        lines.append(f"#### {origin}")
        for item in block["subgroups"]:
            if item["arm"] == "fm":
                support = "supported" if _supported(item) else "UNKNOWN_UNSUPPORTED"
                lines.append(f"- {item['view']} / {item['group']}={item['group_value']}: n={item['n']} (positive={item['positive']}, negative={item['negative']}), FM Macro F1={item['macro_f1']}, recall={item['positive_recall']}; {support}.")
    lines.extend(["", "## Fit diagnostics", "", "| Fit ID | nit | nfev | SciPy success | gtol (SUM) | grad infinity / row | Initial loss | Final loss | Factor movement |", "|---|---:|---:|---|---:|---:|---:|---:|---:|"])
    for fit_id, block in result["fit_receipts"].items():
        opt = block["optimizer"]
        lines.append(f"| {fit_id} | {opt.get('nit')} | {opt.get('nfev')} | {opt.get('success')} | {opt.get('gtol')} | {opt.get('verified_normalized_gradient_inf_norm')} | {opt.get('initial_loss_sum')} | {opt.get('final_loss_sum')} | {opt.get('factor_change_norm')} |")
    lines.extend(["", "A02 reference bundles were required and hash/receipt/row-identity checked at both origins. No A02 label array was read. Resource maxima are sampled process/system values, not OS peak RSS. The independent metric/gate receipt is `SCORED_VERIFICATION.json`; it confirms recomputed metrics and verdicts without fitting.", ""])
    lines.append("## Interpretasi akhir")
    lines.append("MECHANISM_PASS means the learned-pair model met its C2_B and matched no-pair gates. COMPETITION_PASS additionally means it met the exact-origin A02 gates. Either result remains diagnostic-only; A02 is an adaptive historical comparator, and E75 W6 overlaps E67 W6.")
    return "\n".join(lines) + "\n"


def run_synthetic_lifecycle(temp_root: Path) -> dict[str, Any]:
    """Exercise optimizer, persistence, resume, independent verification, and guards on temporary data only."""
    temp_root = Path(temp_root).resolve()
    if temp_root == ROOT.resolve() or temp_root.is_relative_to(ROOT.resolve()):
        raise LearnedPairPilotError("Synthetic output root must be outside the repository")
    if temp_root.exists() and any(temp_root.iterdir()):
        raise LearnedPairPilotError(f"Synthetic root must be new/empty: {temp_root}")
    temp_root.mkdir(parents=True, exist_ok=True)
    live_hash_before = _sha256_file(LEDGER)
    report_path = temp_root / "SYNTHETIC_REPORT.json"
    fit_call_count = 0
    result: dict[str, Any] = {}
    try:
        rng = np.random.default_rng(321)
        dense = rng.normal(0, 1.0, (36, 6))
        dense[np.abs(dense) < 0.55] = 0.0
        fit_x = sparse.csr_matrix(dense)
        w_truth = np.array([0.18, -0.25, 0.12, 0.0, 0.16, -0.1])
        v_truth = np.random.default_rng(17).normal(0.0, 0.18, (6, 2))
        from src.models.learned_pair_pilot_v1 import fm_pair_score, row_l2_normalize
        latent = np.asarray(fit_x @ w_truth).reshape(-1) + fm_pair_score(row_l2_normalize(fit_x), v_truth)
        y = ((latent + rng.normal(0, 0.9, len(latent))) > np.median(latent)).astype(np.int8)
        if np.unique(y).tolist() != [0, 1]:
            raise LearnedPairPilotError("Synthetic target generation did not produce both classes")
        theta0 = initialize_parameters(6, 2, seed=42, factor_std=0.01)
        initial_loss, _ = objective_gradient_sum(theta0, fit_x, y, rank=2, chunk_rows=7)
        fit_call_count += 1
        with threadpool_limits(limits=2):
            synthetic_gtol = float(len(y) * 1e-4)
            synthetic_result = minimize(
                lambda theta: objective_gradient_sum(theta, fit_x, y, rank=2, chunk_rows=7),
                theta0, method="L-BFGS-B", jac=True,
                options={"maxiter": 500, "maxfun": 2000, "maxcor": 10,
                         "maxls": 50, "ftol": 1e-9, "gtol": synthetic_gtol},
            )
        final_loss, final_grad = objective_gradient_sum(synthetic_result.x, fit_x, y, rank=2, chunk_rows=7)
        normalized_gradient = float(np.max(np.abs(final_grad)) / len(y))
        if (not synthetic_result.success or not np.isfinite(synthetic_result.x).all()
                or not np.isfinite(final_loss) or not np.isfinite(final_grad).all()
                or normalized_gradient > 1e-4 or final_loss > initial_loss + 1e-8):
            raise LearnedPairPilotError(f"Synthetic L-BFGS-B convergence gate failed: {synthetic_result.message}")
        fit_logits, _ = score_logits(synthetic_result.x, fit_x, rank=2, chunk_rows=7)
        score_dense = np.array([
            [0.0, 1.0, -0.5, 0.0, 1.5, 0.2],
            [1.0, 0.0, 0.0, -1.0, 0.5, 0.0],
            [0.2, -0.8, 1.2, 0.0, 0.0, 0.3],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [-0.4, 0.7, 0.0, 0.8, 0.0, -0.2],
            [0.9, 0.0, -0.3, 0.1, 0.0, 1.0],
            [0.0, -0.2, 0.4, 0.6, -0.8, 0.0],
        ], dtype=np.float64)
        score_x = sparse.csr_matrix(score_dense)
        raw, pair = score_logits(synthetic_result.x, score_x, rank=2, chunk_rows=3)
        direct = expit(raw)
        present = np.array([True, False, True, False, True, True, False])
        fallback = np.array([0.31, 0.62, 0.21, 0.49, 0.7, 0.12, 0.88])
        routed = direct.copy()
        routed[~present] = fallback[~present]
        bag_raw = np.array([0.22, 0.11, 0.78, 0.91, 0.65, 0.43, 0.36], dtype=np.float64)
        bag_route = bag_raw.copy()
        bag_route[~present] = fallback[~present]
        bag_route = _validate_cached_bag_route(bag_raw, bag_route, present, fallback)
        raw_routed_swap_rejected = False
        try:
            _validate_cached_bag_route(bag_route, bag_raw, present, fallback)
        except LearnedPairPilotError:
            raw_routed_swap_rejected = True
        if not raw_routed_swap_rejected or np.array_equal(bag_raw[~present], fallback[~present]):
            raise LearnedPairPilotError("Synthetic raw/routed bag swap guard did not exercise distinct missing predictions")
        record_ids = np.array(["id-z", "id-a", "id-m", "id-b", "id-y", "id-c", "id-x"])
        cycles = np.array([79, 68, 76, 71, 75, 69, 78], dtype=np.int64)
        score_matrix_path = temp_root / "matrix.npz"
        fit_matrix_path = temp_root / "fit_matrix.npz"
        fit_y_path = temp_root / "fit_y.npy"
        sparse.save_npz(score_matrix_path, score_x)
        sparse.save_npz(fit_matrix_path, fit_x)
        with fit_y_path.open("wb") as stream:
            np.save(stream, y, allow_pickle=False)
        artifact_path = temp_root / "parameters.npz"
        _atomic_npz(artifact_path, parameters=np.asarray(synthetic_result.x, dtype=np.float64),
                    fit_logits=fit_logits, fit_row_ids=np.asarray([f"fit-{i:02d}" for i in range(len(y))]),
                    fit_cycles=np.asarray([(i * 7) % 13 + 47 for i in range(len(y))], dtype=np.int64))
        prediction_path = temp_root / "prediction.npz"
        _atomic_npz(prediction_path, raw_score=raw, probability_direct=direct,
                    probability_routed=routed, present_mask=present,
                    fallback_probability=fallback, prob_bag_raw=bag_raw, prob_bag=bag_route,
                    record_ids=record_ids,
                    cycles=cycles, class_order=np.array([0, 1], dtype=np.int8))
        with_factors = True
        matrix_logical = offset_inputs.sha256_csr(score_x)
        fit_matrix_logical = offset_inputs.sha256_csr(fit_x)
        optimizer_receipt = {
            "success": bool(synthetic_result.success), "status_code": int(synthetic_result.status),
            "message": str(synthetic_result.message), "nit": int(synthetic_result.nit),
            "nfev": int(synthetic_result.nfev), "verified_final_loss_sum": float(final_loss),
            "verified_final_gradient_inf_norm_sum": float(np.max(np.abs(final_grad))),
            "verified_normalized_gradient_inf_norm": normalized_gradient,
            "gtol": synthetic_gtol, "finite_state": True,
            "stationarity_guard": normalized_gradient <= 1e-4,
            "loss_nonincrease_guard": final_loss <= initial_loss + 1e-8,
            "initial_loss_sum": float(initial_loss), "final_loss_sum": float(final_loss),
        }
        manifest = {
            "matrix_archive_sha256": _sha256_file(score_matrix_path),
            "matrix_logical_sha256": matrix_logical, "matrix_shape": list(score_x.shape),
            "fit_matrix_name": fit_matrix_path.name,
            "fit_matrix_archive_sha256": _sha256_file(fit_matrix_path),
            "fit_matrix_logical_sha256": fit_matrix_logical,
            "fit_y_name": fit_y_path.name, "fit_y_sha256": _sha256_file(fit_y_path),
            "rank": 2, "chunk_rows": 7, "with_factors": True,
            "record_ids": record_ids.tolist(), "cycles": cycles.tolist(),
            "fit_record_ids": [f"fit-{i:02d}" for i in range(len(y))],
            "fit_cycles": [(i * 7) % 13 + 47 for i in range(len(y))],
        }
        manifest_path = temp_root / "manifest.json"
        _atomic_json(manifest_path, manifest)
        receipt = {
            "fit_id": "SYNTHETIC_LEARNED_PAIR", "origin": "synthetic",
            "freeze_signature": "synthetic-freeze", "source_commit": "synthetic-source",
            "optimizer_converged": True, "optimizer": optimizer_receipt,
            "fit_population": {"rows": len(y)},
            "artifact_sha256": _sha256_file(artifact_path),
            "prediction_sha256": _sha256_file(prediction_path),
        }
        receipt_path = temp_root / "receipt.json"
        _atomic_json(receipt_path, receipt)
        ledger_path = temp_root / "ledger.jsonl"
        _append_ledger({"event": "FIT_STARTED", "fit_id": "SYNTHETIC_LEARNED_PAIR"}, path=ledger_path)
        frozen_test = {"freeze_signature": "synthetic-freeze", "source_commit": "synthetic-source"}
        events = _ledger_events(ledger_path)
        first = _resume_started_fit("SYNTHETIC_LEARNED_PAIR", frozen_test, events,
                                    ledger_path=ledger_path, run_root=temp_root, official=False,
                                    artifact_path=artifact_path, receipt_path=receipt_path)
        second = _resume_started_fit("SYNTHETIC_LEARNED_PAIR", frozen_test, _ledger_events(ledger_path),
                                     ledger_path=ledger_path, run_root=temp_root, official=False,
                                     artifact_path=artifact_path, receipt_path=receipt_path)
        if first != "resumed_from_durable_receipt" or second != "already_saved":
            raise LearnedPairPilotError(f"Synthetic durable resume failed: {first}, {second}")
        import verify_optimization_v3_learned_pair_pilot_v2 as verifier
        verified = verifier.verify_synthetic_bundle(temp_root)
        if verified.get("status") != "PASS":
            raise LearnedPairPilotError("Independent synthetic verifier did not PASS")
        tamper_root = temp_root / "tamper_check"
        if tamper_root.exists():
            raise LearnedPairPilotError(f"Tamper test path already exists: {tamper_root}")
        if temp_root not in tamper_root.resolve().parents:
            raise LearnedPairPilotError("Tamper test directory escaped the explicit synthetic root")
        import shutil
        shutil.copytree(temp_root, tamper_root)
        with (tamper_root / "matrix.npz").open("ab") as stream:
            stream.write(b"tampered")
        try:
            verifier.verify_synthetic_bundle(tamper_root)
        except Exception:
            tamper_rejected = True
        else:
            tamper_rejected = False
        shutil.rmtree(tamper_root)
        if not tamper_rejected:
            raise LearnedPairPilotError("Independent verifier accepted a tampered matrix")
        production_writer_blocked = False
        before_guard_hash = _sha256_file(LEDGER)
        try:
            _append_ledger({"event": "SHOULD_NOT_WRITE"}, path=LEDGER, official=False)
        except LearnedPairPilotError:
            production_writer_blocked = True
        if not production_writer_blocked or _sha256_file(LEDGER) != before_guard_hash:
            raise LearnedPairPilotError("Production ledger guard failed or changed live ledger")
        failure_ledger = temp_root / "failure_ledger.jsonl"
        _append_ledger({"event": "FIT_STARTED", "fit_id": FIT_IDS[0]}, path=failure_ledger)
        _append_ledger({"event": "FIT_FAILED", "fit_id": FIT_IDS[0]}, path=failure_ledger)
        synthetic_frozen = {"freeze_signature": "synthetic-freeze", "source_commit": "synthetic-source"}
        released = _release_unstarted(synthetic_frozen, "synthetic terminal failure",
                                      ledger_path=failure_ledger, official=False)
        release_event_count = len(_ledger_events(failure_ledger))
        released_again = _release_unstarted(synthetic_frozen, "synthetic resume",
                                            ledger_path=failure_ledger, official=False)
        failure_events = _ledger_events(failure_ledger)
        if (released is None or released.get("unused_fits") != 3
                or released.get("fit_ids") != list(FIT_IDS[1:])
                or released_again != released or release_event_count != 3
                or len(failure_events) != release_event_count
                or failure_events[1].get("fit_id") in released.get("fit_ids", [])):
            raise LearnedPairPilotError("Synthetic terminal failure/release accounting was not idempotent")
        live_hash_after = _sha256_file(LEDGER)
        if live_hash_after != live_hash_before:
            raise LearnedPairPilotError("Live fit ledger changed during temporary synthetic QA")
        temp_events = _ledger_events(ledger_path)
        _append_ledger({"event": "CARD_COMPLETED", "status": "PASS_SYNTHETIC"}, path=ledger_path)
        result = {
            "status": "PASS_SYNTHETIC_END_TO_END", "live_budget_ledger_sha256_before": live_hash_before,
            "live_budget_ledger_sha256_after": live_hash_after,
            "source_commit": _git("rev-parse", "HEAD"), "source_hashes": _source_hashes(),
            "official_fit_calls": 0, "synthetic_optimizer_fit_calls": fit_call_count,
            "synthetic_optimizer": optimizer_receipt,
            "factor_movement_norm": float(np.linalg.norm(synthetic_result.x[7:] - theta0[7:])),
            "durable_artifact_sha256": _sha256_file(artifact_path),
            "durable_prediction_sha256": _sha256_file(prediction_path),
            "resume_first": first, "resume_second": second,
            "optimizer_reruns_on_resume": 0,
            "independent_verification": verified,
            "tampered_matrix_rejected": tamper_rejected,
            "production_writer_blocked": production_writer_blocked,
            "cached_bag_route_formula_pass": True,
            "raw_routed_bag_swap_rejected": raw_routed_swap_rejected,
            "terminal_failure_consumed_started_id": FIT_IDS[0],
            "unstarted_ids_released_once": released.get("fit_ids"),
            "release_idempotent": released_again == released and len(failure_events) == release_event_count,
            "temporary_ledger_events_before_card_complete": [event["event"] for event in temp_events],
            "temporary_ledger_events_final": [event["event"] for event in _ledger_events(ledger_path)],
            "report_path": report_path.as_posix(),
        }
        _atomic_json(report_path, result)
        return result
    finally:
        if _sha256_file(LEDGER) != live_hash_before:
            raise LearnedPairPilotError("Live ledger hash changed during synthetic lifecycle; investigate immediately")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "freeze", "reserve", "fit-next",
                                            "predictions", "verify-prelabel", "score", "synthetic"))
    parser.add_argument("--temp-root", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "preflight":
            value = preflight()
        elif args.command == "freeze":
            value = freeze()
        elif args.command == "reserve":
            value = reserve()
        elif args.command == "fit-next":
            value = fit_next()
        elif args.command == "predictions":
            value = build_predictions()
        elif args.command == "verify-prelabel":
            value = verify_prelabel()
        elif args.command == "score":
            value = score()
        else:
            if args.temp_root is None:
                parser.error("synthetic requires explicit --temp-root outside the repository")
                return 2
            value = run_synthetic_lifecycle(args.temp_root)
            _atomic_json(RUN_ROOT / "SYNTHETIC_QA.json", value)
        print(json.dumps(_jsonable(value), sort_keys=True, allow_nan=False))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "ERROR", "error_type": type(exc).__name__, "error": str(exc)},
                         sort_keys=True, allow_nan=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
