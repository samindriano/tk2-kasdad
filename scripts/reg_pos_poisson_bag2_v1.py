"""Frozen Poisson weighted POS replicas for the TK2 REG pilot.

The only official learner fits in this card are two weighted POS replicas at
E67 and two at E75. Preflight is target-blind. Official fit and score commands
require separate, hash-bound root passes; this module never writes submission
files or accesses Test/Sample data.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping
import zipfile

import joblib
import numpy as np
import psutil
from scipy import sparse
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path(r"C:\Users\Sam\OneDrive\Documents\KASDAD\TP 2")
FRONTIER_ROOT = Path(r"C:\Users\Sam\OneDrive\Documents\KASDAD\tk2-regression-frontier")
RUNNER_PATH = FRONTIER_ROOT / "scripts" / "reg_position_direct_l1_v1.py"
POSITION_RUN = FRONTIER_ROOT / "experiments" / "runs" / "reg_position_direct_l1_v1"
POSITION_FREEZE = POSITION_RUN / "FREEZE_V4.json"
CARD_ID = "REG_POS_POISSON_BAG2_V1"
ORIGINS = (67, 75)
SEEDS = (17, 137)
FIT_IDS = tuple(f"{CARD_ID}_E{origin}_P{seed}" for origin in ORIGINS for seed in SEEDS)
EXPERIMENT = ROOT / "experiments" / "reg_pos_poisson_bag2_v1"
REPORT = ROOT / "reports" / "optimization_v3" / "reg_pos_poisson_bag2_v1"
WEIGHTS_PATH = EXPERIMENT / "POISSON_WEIGHTS_R1.npz"
FREEZE_PATH = EXPERIMENT / "FREEZE_R1.json"
SUPERSEDED_FREEZE_PATH = EXPERIMENT / "FREEZE_R1_SUPERSEDED_2c5742e5.json"
SUPERSEDED_FREEZE_ATTEMPT_PATH = EXPERIMENT / "FREEZE_R1_SUPERSEDED_INVALID_928ba2a8.json"
ROOT_SOURCE_PASS = EXPERIMENT / "ROOT_SOURCE_PASS_R1.json"
ROOT_SCORE_PASS = EXPERIMENT / "ROOT_SCORE_PASS_R1.json"
OFFICIAL_RUN = ROOT / "experiments" / "runs" / "reg_pos_poisson_bag2_v1"
PREDICTION_FREEZE = OFFICIAL_RUN / "PREDICTION_FREEZE_R1.json"
PREFIT_REPORT = REPORT / "PREFIT_VERIFICATION_R1.json"
SUPERSEDED_PREFIT_REPORT_PATH = REPORT / "PREFIT_VERIFICATION_R1_SUPERSEDED_INVALID_928ba2a8.json"
PRELABEL_REPORT = REPORT / "PRELABEL_VERIFICATION_R1.json"
SCORE_REPORT = REPORT / "SCORE_R1.json"
SYNTHETIC_REPORT = REPORT / "SYNTHETIC_E2E.json"
IMPLEMENTATION_PATHS = {
    "runner": ROOT / "scripts" / "reg_pos_poisson_bag2_v1.py",
    "verifier": ROOT / "scripts" / "verify_reg_pos_poisson_bag2_v1.py",
    "test": ROOT / "tests" / "test_reg_pos_poisson_bag2_v1.py",
}
REQUIRED_ROOT_TRACKED_PATHS = {
    "scripts/reg_pos_poisson_bag2_v1.py",
    "scripts/verify_reg_pos_poisson_bag2_v1.py",
    "tests/test_reg_pos_poisson_bag2_v1.py",
    "experiments/reg_pos_poisson_bag2_v1/FREEZE_R1.json",
    "experiments/reg_pos_poisson_bag2_v1/FREEZE_R1_SUPERSEDED_2c5742e5.json",
    "experiments/reg_pos_poisson_bag2_v1/FREEZE_R1_SUPERSEDED_INVALID_928ba2a8.json",
    "experiments/reg_pos_poisson_bag2_v1/POISSON_WEIGHTS_R1.npz",
    "reports/optimization_v3/reg_pos_poisson_bag2_v1/PREFIT_VERIFICATION_R1.json",
    "reports/optimization_v3/reg_pos_poisson_bag2_v1/PREFIT_VERIFICATION_R1_SUPERSEDED_INVALID_928ba2a8.json",
    "reports/optimization_v3/reg_pos_poisson_bag2_v1/SYNTHETIC_E2E.json",
}
HINGE_RUNNER = ROOT / "scripts" / "reg_monotone_hinge_p50_v1.py"
LOCAL_CORRECTION = ROOT / "experiments" / "reg_center_local_correction_v1"
LOCAL_PREDICTION_FREEZE = LOCAL_CORRECTION / "PREDICTION_FREEZE_R1.json"
LOCAL_ROOT_SOURCE_PASS = LOCAL_CORRECTION / "ROOT_SOURCE_PASS_R1.json"
HINGE_PARAMS_PATHS = {
    67: LOCAL_CORRECTION / "fits_R1" / "D1_LOCAL.json",
    75: LOCAL_CORRECTION / "fits_R1" / "D2_LOCAL.json",
}
LOCAL_PREDICTION_PATHS = {
    67: LOCAL_CORRECTION / "score_predictions_R1" / "HINGE_D1_PREDICTIONS.npz",
    75: LOCAL_CORRECTION / "score_predictions_R1" / "HINGE_D2_PREDICTIONS.npz",
}
SCORE_WINDOWS = {67: (72, 75), 75: (76, 79)}
R4_SCORE_PATHS = {
    67: ROOT / "experiments" / "runs" / "reg_monotone_hinge_p50_v1" / "R4" / "score_rows" / "HINGE_D1.npz",
    75: ROOT / "experiments" / "runs" / "reg_monotone_hinge_p50_v1" / "R4" / "score_rows" / "HINGE_D2.npz",
}
AUTHORIZED_SCORE_SHA256 = {
    67: "b996d09f18446eff7f8723c76b5d44d83e0a6d684672c98d38839a2ea81e2643",
    75: "8c950bd8fd27b00f89c71a4e87eed0e0270cf385b9aece9cca33caeee91040e6",
}
SIMPLIFICATION_ROOT = Path(r"C:\Users\Sam\OneDrive\Documents\KASDAD\tk2-simplification-audit-v1")
SIMPLIFICATION_SCRIPT = SIMPLIFICATION_ROOT / "scripts" / "simplification_audit_v1.py"
SIMPLIFICATION_FREEZE = SIMPLIFICATION_ROOT / "reports" / "simplification_v1" / "INPUT_FREEZE.json"
MIN_FREE_RAM_RESERVE = 512 * 1024**2
THREADS = 2
SYNTHETIC_PARAMETERS = {
    "boosting_type": "gbdt", "colsample_bytree": 1.0, "deterministic": True,
    "force_col_wise": True, "importance_type": "split", "learning_rate": 0.08,
    "max_depth": -1, "min_child_samples": 4, "min_child_weight": 0.001,
    "min_split_gain": 0.0, "n_estimators": 8, "n_jobs": THREADS,
    "num_leaves": 7, "objective": "regression_l1", "random_state": 42,
    "reg_alpha": 0.0, "reg_lambda": 3.0, "subsample": 1.0,
    "subsample_for_bin": 2000, "subsample_freq": 0, "verbosity": -1,
}


class PilotError(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(_jsonable(value), sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8", errors="surrogatepass")


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sign_freeze(body: Mapping[str, Any]) -> dict[str, Any]:
    signed = {key: value for key, value in body.items() if key != "freeze_signature"}
    signed["freeze_signature"] = _canonical_sha(signed)
    return signed


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_frozen_training_source(source_root: Path, freeze: Mapping[str, Any], runner: Any) -> str:
    resolved_root = Path(source_root).resolve()
    if resolved_root != SOURCE_ROOT.resolve():
        raise PilotError("Training target source root differs from the frozen TP 2 checkout")
    train_path = runner._contained_file(resolved_root, runner.TRAIN_REL)
    actual_sha = _sha_file(train_path)
    expected_sha = freeze.get("official_train_sha256")
    if (not expected_sha or actual_sha != expected_sha
            or actual_sha != getattr(runner, "TRAIN_SHA256", None)):
        raise PilotError("Training CSV hash differs from the frozen source and POS runner")
    return actual_sha


def _sha_array(value: Any, *, dtype: str | None = None) -> str:
    array = np.asarray(value, dtype=dtype)
    header = _canonical({"shape": array.shape, "dtype": str(array.dtype)})
    digest = hashlib.sha256(header)
    if array.dtype.kind in "OUS":
        digest.update(_canonical(array.astype(str).tolist()))
    else:
        digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()


def _sha_csr(value: sparse.spmatrix) -> str:
    matrix = sparse.csr_matrix(value, dtype=np.float64, copy=True)
    matrix.sum_duplicates()
    matrix.sort_indices()
    digest = hashlib.sha256(_canonical({"shape": matrix.shape, "dtype": str(matrix.dtype)}))
    digest.update(np.asarray(matrix.indptr, dtype="<i8").tobytes())
    digest.update(np.asarray(matrix.indices, dtype="<i8").tobytes())
    digest.update(np.asarray(matrix.data, dtype="<f8").tobytes())
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(_jsonable(dict(value)), sort_keys=True, ensure_ascii=False,
                         allow_nan=False, indent=2).encode("utf-8") + b"\n"
    _atomic_bytes(path, payload)


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_joblib(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        joblib.dump(value, temporary, compress=3)
        with open(temporary, "r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PilotError(f"Cannot read JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise PilotError(f"Expected JSON object: {path}")
    return value


def _read_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    result = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PilotError(f"Malformed append-only ledger at line {line_no}: {path}") from exc
        if not isinstance(event, dict):
            raise PilotError(f"Invalid ledger event at line {line_no}: {path}")
        result.append(event)
    return result


def _append_event(path: Path, event: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        stream.write(_canonical(dict(event)) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


class _ExclusiveLock:
    def __init__(self, path: Path):
        self.path = path
        self.fd: int | None = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise PilotError(f"A heavy fit lock already exists; inspect before resuming: {self.path}") from exc
        os.write(self.fd, f"pid={os.getpid()} utc={datetime.now(timezone.utc).isoformat()}\n".encode())
        os.fsync(self.fd)
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.fd is not None:
            os.close(self.fd)
        self.path.unlink(missing_ok=True)


def _load_module(path: Path, module_name: str):
    if not path.is_file():
        raise PilotError(f"Required frozen source file is missing: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise PilotError(f"Cannot load frozen source module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _runner():
    if str(FRONTIER_ROOT) not in sys.path:
        sys.path.insert(0, str(FRONTIER_ROOT))
    return _load_module(RUNNER_PATH, "reg_position_direct_l1_v1_frozen")


def _hinge():
    return _load_module(HINGE_RUNNER, "reg_monotone_hinge_p50_v1_frozen")


def _git(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(["git", "-C", str(root), *args], check=True,
                                capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PilotError(f"Git identity query failed in {root}: {args}") from exc
    return result.stdout.strip()


def _freeze_body(freeze: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in freeze.items() if key != "freeze_signature"}


def _validate_signed_json(path: Path, signature_key: str) -> dict[str, Any]:
    value = _read_json(path)
    body = {key: item for key, item in value.items() if key != signature_key}
    if value.get(signature_key) != _canonical_sha(body):
        raise PilotError(f"Canonical signature failed: {path}")
    return value


def _bounded(root: Path, relative: str) -> Path:
    root = root.resolve()
    path = (root / relative).resolve()
    if root not in path.parents and path != root:
        raise PilotError(f"Frozen path escapes its root: {relative}")
    if not path.is_file():
        raise PilotError(f"Frozen file is missing: {path}")
    return path


def _weights_for(rows: np.ndarray, ids: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    counts = np.random.default_rng(seed).poisson(1.0, size=len(rows)).astype(np.uint8)
    total = int(counts.sum())
    if total <= 0:
        raise PilotError(f"Poisson replica has zero total counts for seed {seed}")
    weights = counts.astype(np.float64) * (len(rows) / float(total))
    if (len(weights) != len(rows) or np.any(weights < 0) or not np.isfinite(weights).all()
            or abs(float(weights.sum()) - len(rows)) > 1e-8 * len(rows)):
        raise PilotError("Normalized Poisson sample weights are invalid")
    return counts, weights


def _weight_payload(items: Mapping[int, Any]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for origin in ORIGINS:
        item = items[origin]
        arrays[f"fit_rows_E{origin}"] = np.asarray(item.fit_rows, dtype=np.int64)
        arrays[f"fit_ids_E{origin}"] = np.asarray(item.fit_ids, dtype=str)
        for seed in SEEDS:
            counts, weights = _weights_for(item.fit_rows, item.fit_ids, seed)
            arrays[f"counts_E{origin}_P{seed}"] = counts
            arrays[f"weights_E{origin}_P{seed}"] = weights
    return arrays


def _ensure_weight_file(payload: Mapping[str, np.ndarray]) -> str:
    if WEIGHTS_PATH.exists():
        with np.load(WEIGHTS_PATH, allow_pickle=False) as existing:
            if set(existing.files) != set(payload):
                raise PilotError("Existing Poisson weight file schema differs; no overwrite allowed")
            for name, expected in payload.items():
                if not np.array_equal(existing[name], expected):
                    raise PilotError(f"Existing Poisson weights differ from fixed seed replay: {name}")
    else:
        _atomic_npz(WEIGHTS_PATH, **payload)
    return _sha_file(WEIGHTS_PATH)


def _load_weights(freeze: Mapping[str, Any], origin: int, seed: int,
                  fit_rows: np.ndarray, fit_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if _sha_file(WEIGHTS_PATH) != freeze["weights_sha256"]:
        raise PilotError("Poisson weight artifact changed after freeze")
    with np.load(WEIGHTS_PATH, allow_pickle=False) as saved:
        rows = saved[f"fit_rows_E{origin}"]
        ids = saved[f"fit_ids_E{origin}"].astype(str)
        counts = saved[f"counts_E{origin}_P{seed}"]
        weights = saved[f"weights_E{origin}_P{seed}"]
    if (not np.array_equal(rows, fit_rows) or not np.array_equal(ids, fit_ids.astype(str))
            or not np.array_equal(counts, _weights_for(fit_rows, fit_ids, seed)[0])
            or not np.isfinite(weights).all()
            or abs(float(weights.sum()) - len(fit_rows)) > 1e-8 * len(fit_rows)):
        raise PilotError(f"Frozen weights/row identity mismatch for E{origin} seed {seed}")
    expected_weights = counts.astype(np.float64) * (len(fit_rows) / float(counts.sum()))
    if not np.array_equal(weights, expected_weights):
        raise PilotError(f"Frozen normalized weights differ for E{origin} seed {seed}")
    return counts, weights


def _source_artifact_hashes(runner: Any) -> dict[str, str]:
    hashes: dict[str, str] = {
        "frontier_runner": _sha_file(RUNNER_PATH),
        "position_freeze_v4": _sha_file(POSITION_FREEZE),
        "hinge_runner": _sha_file(HINGE_RUNNER),
        "local_prediction_freeze": _sha_file(LOCAL_PREDICTION_FREEZE),
        "local_root_source_pass": _sha_file(LOCAL_ROOT_SOURCE_PASS),
        "simplification_audit_script": _sha_file(SIMPLIFICATION_SCRIPT),
        "simplification_input_freeze": _sha_file(SIMPLIFICATION_FREEZE),
    }
    for origin in ORIGINS:
        hashes[f"hinge_parameters_E{origin}"] = _sha_file(HINGE_PARAMS_PATHS[origin])
        hashes[f"local_comparator_E{origin}"] = _sha_file(LOCAL_PREDICTION_PATHS[origin])
        fit_id = f"{runner.CARD_ID}_E{origin}_REAL"
        artifact, receipt = runner._unit_paths(fit_id, runner.RUN_ROOT)
        pred, pred_receipt = runner._prediction_paths(fit_id, runner.RUN_ROOT)
        for suffix, path in (("model", artifact), ("model_receipt", receipt),
                             ("pos42_prediction", pred), ("pos42_prediction_receipt", pred_receipt)):
            if not path.is_file():
                raise PilotError(f"Required original POS42 artifact is missing: {path}")
            hashes[f"E{origin}_{suffix}"] = _sha_file(path)
        for component in ("STRUCTURED", "SCALER", "OHE"):
            unit = f"{runner.CARD_ID}_E{origin}_{component}"
            artifact, receipt = runner._unit_paths(unit, runner.RUN_ROOT)
            if not artifact.is_file() or not receipt.is_file():
                raise PilotError(f"Required cached preprocessor is absent: {unit}")
            hashes[f"E{origin}_{component}_artifact"] = _sha_file(artifact)
            hashes[f"E{origin}_{component}_receipt"] = _sha_file(receipt)
    return hashes


def _load_hinge_parameters() -> dict[int, dict[str, float]]:
    result: dict[int, dict[str, float]] = {}
    for origin, path in HINGE_PARAMS_PATHS.items():
        state = _read_json(path)
        if state.get("status") != "FIT_COMPLETE" or not isinstance(state.get("saved_hinge_parameters"), dict):
            raise PilotError(f"Saved local hinge parameters are not in a completed state: {path}")
        params = {k: float(v) for k, v in state["saved_hinge_parameters"].items()}
        if set(params) != {"intercept", "slope_low", "slope_high"} or not all(np.isfinite(v) for v in params.values()):
            raise PilotError(f"Saved local hinge parameter schema is invalid: {path}")
        result[origin] = params
    return result


def _explicit_npz(path: Path, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Read only named non-target members from a frozen NPZ."""
    with zipfile.ZipFile(path) as archive:
        names = {name[:-4] for name in archive.namelist() if name.endswith(".npy")}
    if not set(keys).issubset(names) or "targets" in keys or any("target" in key.lower() for key in keys):
        raise PilotError(f"Required non-target members absent or target access requested: {path}")
    with np.load(path, allow_pickle=False) as bundle:
        return {key: bundle[key].copy() for key in keys}


def _compare_comparator_rows(item: Any, origin: int, direct: np.ndarray,
                             hinge: np.ndarray) -> dict[str, Any]:
    path = LOCAL_PREDICTION_PATHS[origin]
    local = _explicit_npz(path, ("physical_rows", "record_ids", "prediction_origins",
                                 "original_bundle_indices", "cycles", "direct_predictions",
                                 "missing_mask", "hinge_baseline"))
    rows = np.asarray(local["physical_rows"], dtype=np.int64)
    ids = local["record_ids"].astype(str)
    cycles = np.asarray(local["cycles"], dtype=np.int64)
    row_to_position = {int(row): index for index, row in enumerate(item.query_rows)}
    try:
        indexes = np.asarray([row_to_position[int(row)] for row in rows], dtype=np.int64)
    except KeyError as exc:
        raise PilotError(f"Frozen comparator row is not in E{origin} query: {exc}") from exc
    if (len(set(rows.tolist())) != len(rows)
            or not np.all(local["prediction_origins"] == origin)
            or len(set(local["original_bundle_indices"].tolist())) != len(rows)
            or not np.array_equal(item.query_ids[indexes].astype(str), ids)
            or not np.array_equal(item.query_cycles[indexes].astype(np.int64), cycles)
            or np.any((cycles < SCORE_WINDOWS[origin][0]) | (cycles > SCORE_WINDOWS[origin][1]))):
        raise PilotError(f"Comparator row identity/cycle window differs at E{origin}")
    if (not np.array_equal(local["direct_predictions"], direct[indexes])
            or not np.array_equal(local["hinge_baseline"], hinge[indexes])
            or not np.array_equal(local["missing_mask"].astype(bool), item.query_missing[indexes].astype(bool))):
        raise PilotError(f"POS42 full DIRECT or saved hinge comparator replay differs at E{origin}")
    return {
        "origin": origin, "score_cycles": list(SCORE_WINDOWS[origin]),
        "rows": len(rows), "present_rows": int((~local["missing_mask"].astype(bool)).sum()),
        "missing_rows": int(local["missing_mask"].sum()),
        "physical_rows_sha256": _sha_array(rows), "record_ids_sha256": _sha_array(ids),
        "cycles_sha256": _sha_array(cycles), "direct_sha256": _sha_array(direct[indexes]),
        "hinge_sha256": _sha_array(hinge[indexes]),
        "row_identity_sha256": _canonical_sha({"physical_rows": _sha_array(rows),
                                                "record_ids": _sha_array(ids),
                                                "cycles": _sha_array(cycles)}),
        "local_prediction_file_sha256": _sha_file(path),
    }


def _guard_cached_preprocessors(runner: Any, freeze: Mapping[str, Any], origin: int) -> None:
    events = runner._read_events(runner.RUN_ROOT / "fit_ledger.jsonl")
    for component in ("STRUCTURED", "SCALER", "OHE"):
        fit_id = f"{runner.CARD_ID}_E{origin}_{component}"
        artifact, receipt = runner._unit_paths(fit_id, runner.RUN_ROOT)
        if not artifact.is_file() or not receipt.is_file():
            raise PilotError(f"Preflight refuses to create missing preprocessor state: {fit_id}")
        starts = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("fit_id") == fit_id]
        saves = [e for e in events if e.get("event") == "FIT_SAVED" and e.get("fit_id") == fit_id]
        fails = [e for e in events if e.get("event") == "FIT_FAILED" and e.get("fit_id") == fit_id]
        if len(starts) != 1 or len(saves) != 1 or fails:
            raise PilotError(f"Preprocessor ledger is not exactly one start/save: {fit_id}")
        runner._validate_unit(runner.RUN_ROOT, fit_id, freeze)


def _replay_origin(runner: Any, hinge_module: Any, source_root: Path, frame: Any,
                   origin: int, position_freeze: Mapping[str, Any],
                   hinge_parameters: Mapping[int, Mapping[str, float]]) -> tuple[Any, dict[str, np.ndarray], dict[str, Any]]:
    item = runner._load_origin(source_root, frame, origin)
    _guard_cached_preprocessors(runner, position_freeze, origin)
    prefix = f"{runner.CARD_ID}_E{origin}"
    fit_frame = frame.iloc[item.fit_rows].loc[:, list(runner.HYBRID_FEATURES)].copy()
    query_frame = frame.iloc[item.query_rows].loc[:, list(runner.HYBRID_FEATURES)].copy()
    structured_fit, structured_query, structured_meta = runner._fit_or_transform_structured(
        prefix, origin, position_freeze, fit_frame, query_frame, output_root=runner.RUN_ROOT)
    X_fit = runner._combine_design(structured_fit, item.fit_bag, item.fit_position, item.fit_missing)
    X_query = runner._combine_design(structured_query, item.query_bag, item.query_position, item.query_missing)
    fit_id = f"{runner.CARD_ID}_E{origin}_REAL"
    model, model_receipt = runner._validate_unit(runner.RUN_ROOT, fit_id, position_freeze)
    with threadpool_limits(limits=THREADS):
        replay_raw = np.asarray(model.predict(X_query), dtype=np.float64)
    pred_path, pred_receipt = runner._prediction_paths(fit_id, runner.RUN_ROOT)
    saved = runner._validate_prediction_bundle(
        pred_path, pred_receipt, freeze=position_freeze, fit_id=fit_id, origin=origin,
        arm="REAL", query_rows=item.query_rows, query_ids=item.query_ids,
        query_cycles=item.query_cycles, missing=item.query_missing,
        baseline=item.query_baseline)
    if not np.array_equal(replay_raw, saved["raw_auxiliary"]):
        raise PilotError(f"Original POS42 raw predictions are not bit-identical at E{origin}")
    direct = saved["prediction"]
    local_hinge_all = hinge_module.apply_hinge_present_only(
        direct, item.query_missing, hinge_parameters[origin])
    comparator = _compare_comparator_rows(item, origin, direct, local_hinge_all)
    info = {
        "origin": origin, "fit_rows": len(item.fit_rows), "query_rows": len(item.query_rows),
        "fit_row_indices_sha256": _sha_array(item.fit_rows),
        "fit_ids_sha256": _sha_array(item.fit_ids),
        "query_row_indices_sha256": _sha_array(item.query_rows),
        "query_ids_sha256": _sha_array(item.query_ids),
        "fit_cycles": [int(item.fit_cycles.min()), int(item.fit_cycles.max())],
        "query_cycles": [int(item.query_cycles.min()), int(item.query_cycles.max())],
        "fit_position_shape": list(X_fit.shape), "fit_position_nnz": int(X_fit.nnz),
        "query_position_shape": list(X_query.shape), "query_position_nnz": int(X_query.nnz),
        "fit_design_sha256": _sha_csr(X_fit), "query_design_sha256": _sha_csr(X_query),
        "fit_design_sparse_bytes": int(X_fit.data.nbytes + X_fit.indices.nbytes + X_fit.indptr.nbytes),
        "query_design_sparse_bytes": int(X_query.data.nbytes + X_query.indices.nbytes + X_query.indptr.nbytes),
        "base_raw_sha256": _sha_array(replay_raw), "base_direct_sha256": _sha_array(direct),
        "base_hinge_sha256": _sha_array(local_hinge_all),
        "baseline_sha256": _sha_array(item.query_baseline),
        "missing_rows": int(item.query_missing.sum()),
        "group_edges": item.group_edges, "cache_audit": item.audit,
        "structured_state": structured_meta, "comparator": comparator,
        "model_receipt_sha256": _sha_file(runner._unit_paths(fit_id, runner.RUN_ROOT)[1]),
        "prediction_file_sha256": _sha_file(pred_path),
        "prediction_receipt_sha256": _sha_file(pred_receipt),
    }
    return item, info


def _preflight_body(runner: Any, position_freeze: Mapping[str, Any],
                    origins: Mapping[int, Mapping[str, Any]], weight_sha: str) -> dict[str, Any]:
    return {
        "schema_version": 1, "experiment_id": CARD_ID, "status": "PREFIT_FROZEN",
        "base_commit": _git(ROOT, "rev-parse", "HEAD"),
        "local_branch": _git(ROOT, "branch", "--show-current"),
        "source_root": str(SOURCE_ROOT.resolve()),
        "source_git_head": _git(SOURCE_ROOT, "rev-parse", "HEAD"),
        "source_git_branch": _git(SOURCE_ROOT, "branch", "--show-current"),
        "source_worktree_status_sha256": _canonical_sha(
            _git(SOURCE_ROOT, "status", "--porcelain").splitlines()),
        "frontier_root": str(FRONTIER_ROOT.resolve()),
        "runner_sha256": _sha_file(RUNNER_PATH),
        "position_freeze_path": str(POSITION_FREEZE.resolve()),
        "position_freeze_sha256": _sha_file(POSITION_FREEZE),
        "position_freeze_signature": position_freeze["freeze_signature"],
        "official_train_sha256": position_freeze["official_train_sha256"],
        "implementation_hashes": {key: _sha_file(path) for key, path in IMPLEMENTATION_PATHS.items()},
        "resolved_parameters": dict(position_freeze["resolved_parameters"]),
        "fit_ids": list(FIT_IDS), "fit_cap": len(FIT_IDS),
        "replica_contract": {"base_model": "POS42", "replicas_per_origin": 2,
                             "poisson_lambda": 1.0, "seeds": list(SEEDS),
                             "weights": "counts*n_fit/sum(counts)",
                             "average_before_rank": True, "raw_formula": "mean(POS42,P17,P137)"},
        "weight_file": WEIGHTS_PATH.relative_to(ROOT).as_posix(),
        "weights_sha256": weight_sha,
        "hinge": {"runner_sha256": _sha_file(HINGE_RUNNER),
                  "prediction_freeze_sha256": _sha_file(LOCAL_PREDICTION_FREEZE),
                  "parameters": _load_hinge_parameters()},
        "local_sources": _source_artifact_hashes(runner),
        "score_windows": {str(o): list(SCORE_WINDOWS[o]) for o in ORIGINS},
        "query_targets_read": False, "official_train_target_column_loaded": False,
        "test_or_sample_accessed": False, "csv_created": False,
        "official_fits_started": 0, "official_ledger_mutated": False,
        "resource_policy": {
            "threads": THREADS, "concurrent_heavy_fits": 1,
            "available_memory_reserve_bytes": MIN_FREE_RAM_RESERVE,
            "incremental_estimate": "max(256MiB, 2*sparse_design_storage_bytes)",
            "historical_note": "Original REG_POSITION_DIRECT_L1_V1 E67/E75 fits used the same resolved learner and row/feature recipe.",
        },
        "origins": {str(origin): dict(origins[origin]) for origin in ORIGINS},
    }


def preflight(source_root: Path = SOURCE_ROOT) -> dict[str, Any]:
    source_root = source_root.resolve()
    if source_root != SOURCE_ROOT.resolve():
        raise PilotError("Source root must match the fixed TP 2 regression source")
    if not POSITION_FREEZE.is_file():
        raise PilotError("Authoritative POS FREEZE_V4 is missing")
    runner = _runner()
    position_freeze = _validate_signed_json(POSITION_FREEZE, "freeze_signature")
    if position_freeze.get("card_id") != runner.CARD_ID or position_freeze.get("targets_read") is not False:
        raise PilotError("Authoritative POS freeze identity or target-blind status failed")
    if (position_freeze.get("official_train_sha256") != getattr(runner, "TRAIN_SHA256", None)
            or _assert_frozen_training_source(source_root, position_freeze, runner)
            != position_freeze.get("official_train_sha256")):
        raise PilotError("POS runner, source freeze, and current Train.csv hash do not agree")
    local_freeze = _validate_signed_json(LOCAL_PREDICTION_FREEZE, "signature")
    if (local_freeze.get("status") != "SCORE_PREDICTIONS_FROZEN_BEFORE_SCORE_TARGET_LOAD"
            or local_freeze.get("score_targets_loaded") is not False
            or local_freeze.get("test_or_sample_accessed") is not False
            or local_freeze.get("csv_created") is not False):
        raise PilotError("Existing local comparator freeze is not score-target-blind")
    expected_fit_hashes = local_freeze.get("fit_state_hashes", {})
    for fit_id, fit_path in (("D1_LOCAL", HINGE_PARAMS_PATHS[67]),
                             ("D2_LOCAL", HINGE_PARAMS_PATHS[75])):
        if expected_fit_hashes.get(fit_id) != _sha_file(fit_path):
            raise PilotError(f"Saved local hinge state differs from its prediction freeze: {fit_id}")
    for entry in local_freeze.get("prediction_files", []):
        path = Path(entry["path"])
        if not path.is_file() or _sha_file(path) != entry.get("sha256"):
            raise PilotError(f"Existing local comparator prediction hash failed: {path}")
    initial_available = int(psutil.virtual_memory().available)
    if initial_available < 1024 * 1024**2:
        raise PilotError(f"Target-free sparse replay needs at least 1 GiB available RAM; saw {initial_available}")
    frame = runner._load_feature_frame(source_root)
    if "threat_rank" in frame.columns:
        raise PilotError("Feature loader exposed threat_rank during preflight")
    hinge_module = _hinge()
    parameters = _load_hinge_parameters()
    items: dict[int, Any] = {}
    info: dict[int, dict[str, Any]] = {}
    official_ledger = OFFICIAL_RUN / "fit_ledger.jsonl"
    ledger_before = _sha_file(official_ledger) if official_ledger.is_file() else None
    for origin in ORIGINS:
        item, origin_info = _replay_origin(
            runner, hinge_module, source_root, frame, origin, position_freeze, parameters)
        items[origin] = item
        info[origin] = origin_info
        available_now = int(psutil.virtual_memory().available)
        if available_now < MIN_FREE_RAM_RESERVE:
            raise PilotError(f"Sparse preflight reserve fell below 512 MiB at E{origin}: {available_now}")
    payload = _weight_payload(items)
    weights_sha = _ensure_weight_file(payload)
    body = _preflight_body(runner, position_freeze, info, weights_sha)
    body["weight_vectors"] = {
        f"E{origin}_P{seed}": {
            "counts_sha256": _sha_array(payload[f"counts_E{origin}_P{seed}"]),
            "weights_sha256": _sha_array(payload[f"weights_E{origin}_P{seed}"]),
            "weight_sum": float(payload[f"weights_E{origin}_P{seed}"].sum()),
            "fit_rows": len(items[origin].fit_rows),
        }
        for origin in ORIGINS for seed in SEEDS
    }
    body["source_artifacts"] = _source_artifact_hashes(runner)
    body = _sign_freeze(body)
    if FREEZE_PATH.exists():
        old = _validate_signed_json(FREEZE_PATH, "freeze_signature")
        if old.get("supersedes"):
            body["supersedes"] = old["supersedes"]
        if old.get("superseded_attempts"):
            body["superseded_attempts"] = old["superseded_attempts"]
        if old.get("superseded_preflight_report"):
            body["superseded_preflight_report"] = old["superseded_preflight_report"]
        body = _sign_freeze(body)
        if old["freeze_signature"] != body["freeze_signature"]:
            events = _read_ledger(OFFICIAL_RUN / "fit_ledger.jsonl")
            if any(e.get("event") == "FIT_STARTED" for e in events) or ROOT_SOURCE_PASS.exists():
                raise PilotError("Frozen source changed after approval or official fit start")
            if old.get("supersedes"):
                raise PilotError("A superseded source freeze differs; preserve it and request a new revision")
            archived = FREEZE_PATH.read_bytes()
            superseded = {
                "path": SUPERSEDED_FREEZE_PATH.relative_to(ROOT).as_posix(),
                "sha256": hashlib.sha256(archived).hexdigest(),
                "freeze_signature": old["freeze_signature"],
            }
            if SUPERSEDED_FREEZE_PATH.exists():
                if _sha_file(SUPERSEDED_FREEZE_PATH) != superseded["sha256"]:
                    raise PilotError("Superseded freeze archive exists with different bytes")
            else:
                _atomic_bytes(SUPERSEDED_FREEZE_PATH, archived)
            body["supersedes"] = superseded
            body = _sign_freeze(body)
            _atomic_json(FREEZE_PATH, body)
        else:
            body = old
    else:
        _atomic_json(FREEZE_PATH, body)
    body = _validate_signed_json(FREEZE_PATH, "freeze_signature")
    ledger_after = _sha_file(official_ledger) if official_ledger.is_file() else None
    if ledger_before != ledger_after:
        raise PilotError("Preflight unexpectedly changed the official fit ledger")
    report = {
        "schema_version": 1, "experiment_id": CARD_ID, "status": "PASS",
        "freeze_signature": body["freeze_signature"],
        "freeze_sha256": _sha_file(FREEZE_PATH), "weights_sha256": weights_sha,
        "query_targets_read": False, "official_target_column_loaded": False,
        "official_fit_units_started": 0, "official_ledger_before_sha256": ledger_before,
        "official_ledger_after_sha256": ledger_after, "official_ledger_mutated": False,
        "resource_snapshots": {"initial_available_bytes": initial_available,
                               "minimum_reserve_bytes": MIN_FREE_RAM_RESERVE,
                               "available_after_replay_bytes": int(psutil.virtual_memory().available)},
        "test_or_sample_accessed": False, "csv_created": False,
        "origins": {str(origin): info[origin]["comparator"] for origin in ORIGINS},
        "weight_vectors": body["weight_vectors"],
    }
    _atomic_json(PREFIT_REPORT, report)
    return report


def _feature_designs(runner: Any, source_root: Path, freeze: Mapping[str, Any],
                     origin: int) -> tuple[Any, Any, sparse.csr_matrix, sparse.csr_matrix]:
    position_freeze = _validate_signed_json(POSITION_FREEZE, "freeze_signature")
    frame = runner._load_feature_frame(source_root)
    if "threat_rank" in frame.columns:
        raise PilotError("Feature loader exposed target column")
    item = runner._load_origin(source_root, frame, origin)
    expected = freeze["origins"][str(origin)]
    if (_sha_array(item.fit_rows) != expected["fit_row_indices_sha256"]
            or _sha_array(item.fit_ids) != expected["fit_ids_sha256"]
            or _sha_array(item.query_rows) != expected["query_row_indices_sha256"]
            or _sha_array(item.query_ids) != expected["query_ids_sha256"]):
        raise PilotError(f"Frozen row identity changed at E{origin}")
    _guard_cached_preprocessors(runner, position_freeze, origin)
    prefix = f"{runner.CARD_ID}_E{origin}"
    fit_frame = frame.iloc[item.fit_rows].loc[:, list(runner.HYBRID_FEATURES)].copy()
    query_frame = frame.iloc[item.query_rows].loc[:, list(runner.HYBRID_FEATURES)].copy()
    structured_fit, structured_query, _ = runner._fit_or_transform_structured(
        prefix, origin, position_freeze, fit_frame, query_frame, output_root=runner.RUN_ROOT)
    X_fit = runner._combine_design(structured_fit, item.fit_bag, item.fit_position, item.fit_missing)
    X_query = runner._combine_design(structured_query, item.query_bag, item.query_position, item.query_missing)
    if (_sha_csr(X_fit) != expected["fit_design_sha256"]
            or _sha_csr(X_query) != expected["query_design_sha256"]):
        raise PilotError(f"Feature design changed since prefit at E{origin}")
    return item, frame, X_fit, X_query


def _resource_gate(X: sparse.csr_matrix) -> dict[str, Any]:
    storage = int(X.data.nbytes + X.indices.nbytes + X.indptr.nbytes)
    estimated = max(256 * 1024**2, 2 * storage)
    memory = psutil.virtual_memory()
    process = psutil.Process(os.getpid()).memory_info().rss
    projected = int(memory.available) - estimated
    if projected < MIN_FREE_RAM_RESERVE:
        raise PilotError(f"Resource gate blocked before fit: available={memory.available}, estimate={estimated}, reserve={MIN_FREE_RAM_RESERVE}")
    return {"available_ram_bytes": int(memory.available), "process_rss_bytes": int(process),
            "design_storage_bytes": storage, "incremental_estimate_bytes": estimated,
            "reserve_bytes": MIN_FREE_RAM_RESERVE, "projected_free_bytes": projected,
            "threads": THREADS, "concurrent_heavy_fits": 1}


def _unit_paths(output_root: Path, fit_id: str) -> tuple[Path, Path]:
    return output_root / "units" / f"{fit_id}.joblib", output_root / "receipts" / f"{fit_id}.json"


def _validate_unit(output_root: Path, fit_id: str, freeze_signature: str) -> tuple[Any, dict[str, Any]]:
    artifact, receipt_path = _unit_paths(output_root, fit_id)
    if not artifact.is_file() or not receipt_path.is_file():
        raise PilotError(f"Durable fit unit is missing: {fit_id}")
    receipt = _read_json(receipt_path)
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    events = _read_ledger(output_root / "fit_ledger.jsonl")
    starts = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("fit_id") == fit_id]
    saves = [e for e in events if e.get("event") == "FIT_SAVED" and e.get("fit_id") == fit_id]
    fails = [e for e in events if e.get("event") == "FIT_FAILED" and e.get("fit_id") == fit_id]
    if (receipt.get("freeze_signature") != freeze_signature
            or receipt.get("artifact_sha256") != _sha_file(artifact)
            or receipt.get("receipt_sha256") != _canonical_sha(body)
            or len(starts) != 1 or len(saves) != 1 or fails
            or saves[0].get("receipt_sha256") != _sha_file(receipt_path)):
        raise PilotError(f"Durable receipt/ledger validation failed: {fit_id}")
    return joblib.load(artifact), receipt


def _fit_slot(output_root: Path, fit_id: str, origin: int, seed: int,
              freeze_signature: str, params: Mapping[str, Any], X: sparse.csr_matrix,
              y: np.ndarray, counts: np.ndarray, weights: np.ndarray,
              *, synthetic: bool = False) -> tuple[Any, dict[str, Any], bool]:
    if fit_id not in FIT_IDS:
        raise PilotError(f"Fit cap/identity violation: {fit_id}")
    artifact, receipt_path = _unit_paths(output_root, fit_id)
    events = _read_ledger(output_root / "fit_ledger.jsonl")
    starts = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("fit_id") == fit_id]
    terminals = [e for e in events if e.get("event") in ("FIT_SAVED", "FIT_FAILED") and e.get("fit_id") == fit_id]
    if artifact.exists() or receipt_path.exists() or starts or terminals:
        if starts and not terminals:
            raise PilotError(f"FIT_STARTED has no durable terminal state; retry prohibited: {fit_id}")
        model, receipt = _validate_unit(output_root, fit_id, freeze_signature)
        return model, receipt, False
    prior = [e["fit_id"] for e in events if e.get("event") == "FIT_STARTED"]
    if len(prior) >= len(FIT_IDS) or FIT_IDS[len(prior)] != fit_id:
        raise PilotError(f"Fit order/cap violation: next={FIT_IDS[len(prior)] if len(prior) < len(FIT_IDS) else 'CAP'}, got={fit_id}")
    resource = {} if synthetic else _resource_gate(X)
    start = {"event": "FIT_STARTED", "fit_id": fit_id, "origin": origin, "seed": seed,
             "component": "POS_POISSON_REPLICA", "freeze_signature": freeze_signature,
             "started_utc": datetime.now(timezone.utc).isoformat(), "resource_before": resource,
             "synthetic": synthetic}
    _append_event(output_root / "fit_ledger.jsonl", start)
    started = time.perf_counter()
    try:
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(**dict(params))
        with threadpool_limits(limits=THREADS):
            model.fit(X, y, sample_weight=weights)
        memory_after = psutil.virtual_memory()
        rss_after = psutil.Process(os.getpid()).memory_info().rss
        if not synthetic and memory_after.available < MIN_FREE_RAM_RESERVE:
            raise PilotError(f"Post-fit RAM reserve violated after consumed fit: {memory_after.available}")
        _atomic_joblib(artifact, model)
        receipt = {
            "schema_version": 1, "experiment_id": CARD_ID, "fit_id": fit_id,
            "origin": origin, "seed": seed, "component": "POS_POISSON_REPLICA",
            "freeze_signature": freeze_signature, "artifact_sha256": _sha_file(artifact),
            "fit_elapsed_seconds": time.perf_counter() - started,
            "fit_rows": len(y), "feature_count": int(X.shape[1]),
            "fit_design_sha256": _sha_csr(X), "fit_target_sha256": _sha_array(y),
            "poisson_counts_sha256": _sha_array(counts),
            "zero_weight_rows": int(np.count_nonzero(counts == 0)),
            "sample_weight_sha256": _sha_array(weights), "sample_weight_sum": float(weights.sum()),
            "parameters": dict(params), "underlying_fit_calls": ["LGBMRegressor.fit x1 with sample_weight"],
            "resource_before": resource,
            "resource_after": {} if synthetic else {
                "available_ram_bytes": int(memory_after.available),
                "process_rss_bytes": int(rss_after), "reserve_bytes": MIN_FREE_RAM_RESERVE,
                "threads": THREADS},
            "synthetic": synthetic,
        }
        receipt["receipt_sha256"] = _canonical_sha(receipt)
        _atomic_json(receipt_path, receipt)
        _append_event(output_root / "fit_ledger.jsonl", {
            "event": "FIT_SAVED", "fit_id": fit_id, "origin": origin, "seed": seed,
            "freeze_signature": freeze_signature, "receipt_sha256": _sha_file(receipt_path),
            "saved_utc": datetime.now(timezone.utc).isoformat(), "synthetic": synthetic})
        return model, receipt, True
    except Exception as exc:
        _append_event(output_root / "fit_ledger.jsonl", {
            "event": "FIT_FAILED", "fit_id": fit_id, "origin": origin, "seed": seed,
            "freeze_signature": freeze_signature, "failed_utc": datetime.now(timezone.utc).isoformat(),
            "error": f"{type(exc).__name__}: {exc}", "synthetic": synthetic})
        raise


def _root_source_gate(freeze: Mapping[str, Any]) -> dict[str, Any]:
    if not ROOT_SOURCE_PASS.is_file():
        raise PilotError("Official fitting blocked: ROOT_SOURCE_PASS_R1.json is absent")
    approval = _validate_signed_json(ROOT_SOURCE_PASS, "approval_sha256")
    expected_head = _git(ROOT, "rev-parse", "HEAD")
    if (approval.get("status") != "PASS" or approval.get("freeze_signature") != freeze["freeze_signature"]
            or approval.get("local_commit") != expected_head
            or approval.get("fit_ids") != list(FIT_IDS) or approval.get("fit_cap") != len(FIT_IDS)
            or approval.get("score_access_allowed") is not False
            or approval.get("reviewer") in (None, "", "self")):
        raise PilotError("Root source approval does not match this committed, four-fit, pre-score freeze")
    if (_git(SOURCE_ROOT, "rev-parse", "HEAD") != freeze.get("source_git_head")
            or _git(SOURCE_ROOT, "branch", "--show-current") != freeze.get("source_git_branch")
            or _canonical_sha(_git(SOURCE_ROOT, "status", "--porcelain").splitlines())
            != freeze.get("source_worktree_status_sha256")):
        raise PilotError("Frozen source checkout identity or status changed after preflight")
    tracked_hashes = approval.get("tracked_source_hashes", {})
    if not REQUIRED_ROOT_TRACKED_PATHS.issubset(tracked_hashes):
        raise PilotError("Root source approval does not bind all required source, freeze, weight, and QA files")
    for key, path in IMPLEMENTATION_PATHS.items():
        rel = path.relative_to(ROOT).as_posix()
        if tracked_hashes.get(rel) != freeze.get("implementation_hashes", {}).get(key):
            raise PilotError(f"Root source approval does not bind frozen implementation hash: {rel}")
    for rel, expected_sha in tracked_hashes.items():
        path = _bounded(ROOT, rel)
        if _sha_file(path) != expected_sha:
            raise PilotError(f"Committed source hash differs from root approval: {rel}")
        try:
            committed = subprocess.run(["git", "-C", str(ROOT), "show", f"HEAD:{rel}"],
                                       check=True, capture_output=True).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            raise PilotError(f"Approved source is not present in HEAD: {rel}") from exc
        if hashlib.sha256(committed).hexdigest() != expected_sha:
            raise PilotError(f"Committed blob differs from approved source hash: {rel}")
    synthetic = _read_json(SYNTHETIC_REPORT)
    if (synthetic.get("status") != "PASS" or synthetic.get("official_fit_units_started") != 0
            or synthetic.get("runner_sha256") != _sha_file(Path(__file__))):
        raise PilotError("Required synthetic actual-entrypoint audit is absent or stale")
    if PREFIT_REPORT.is_file():
        report = _read_json(PREFIT_REPORT)
        if report.get("status") != "PASS" or report.get("freeze_signature") != freeze["freeze_signature"]:
            raise PilotError("Target-free prefit verification report does not match freeze")
    return approval


def _prediction_paths(output_root: Path, origin: int) -> tuple[Path, Path]:
    return (output_root / "predictions" / f"E{origin}.npz",
            output_root / "predictions" / f"E{origin}.json")


def _prediction_body(freeze: Mapping[str, Any], origin: int, item: Any,
                     raw_base: np.ndarray, raw17: np.ndarray, raw137: np.ndarray,
                     direct: np.ndarray, hinge: np.ndarray, params: Mapping[str, float],
                     runner: Any, hinge_module: Any) -> dict[str, np.ndarray]:
    mean_raw = (np.asarray(raw_base, dtype=np.float64)
                + np.asarray(raw17, dtype=np.float64)
                + np.asarray(raw137, dtype=np.float64)) / 3.0
    mapped = runner._rank50(mean_raw, item.query_cycles)
    expected_direct = np.clip(0.75 * item.query_baseline + 0.25 * mapped, 0.0, 100.0)
    expected_direct[item.query_missing] = item.query_baseline[item.query_missing]
    expected_hinge = hinge_module.apply_hinge_present_only(expected_direct, item.query_missing, params)
    if not np.array_equal(direct, expected_direct) or not np.array_equal(hinge, expected_hinge):
        raise PilotError(f"Candidate rank/blend/hinge formula mismatch at E{origin}")
    return {
        "origin": np.asarray([origin], dtype=np.int64),
        "fit_rows": np.asarray(item.fit_rows, dtype=np.int64),
        "fit_ids": np.asarray(item.fit_ids, dtype=str),
        "query_rows": np.asarray(item.query_rows, dtype=np.int64),
        "query_ids": np.asarray(item.query_ids, dtype=str),
        "query_cycles": np.asarray(item.query_cycles, dtype=np.int64),
        "missing_mask": np.asarray(item.query_missing, dtype=bool),
        "r06_baseline": np.asarray(item.query_baseline, dtype=np.float64),
        "pos42_raw": np.asarray(raw_base, dtype=np.float64),
        "replica17_raw": np.asarray(raw17, dtype=np.float64),
        "replica137_raw": np.asarray(raw137, dtype=np.float64),
        "mean_raw": mean_raw, "mapped_pos": mapped,
        "full_direct": expected_direct, "same_history_hinge": expected_hinge,
    }


def _write_prediction_freeze(freeze: Mapping[str, Any], outputs: Mapping[int, Path]) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    for origin, path in outputs.items():
        receipt_path = path.with_suffix(".json")
        receipt = _read_json(receipt_path)
        artifacts[str(origin)] = {
            "prediction_path": path.relative_to(ROOT).as_posix(),
            "prediction_sha256": _sha_file(path), "receipt_path": receipt_path.relative_to(ROOT).as_posix(),
            "receipt_sha256": _sha_file(receipt_path),
            "row_identity_sha256": receipt["row_identity_sha256"],
            "full_direct_sha256": receipt["full_direct_sha256"],
            "same_history_hinge_sha256": receipt["same_history_hinge_sha256"],
        }
    body = {
        "schema_version": 1, "experiment_id": CARD_ID, "status": "PREDICTIONS_FROZEN",
        "source_freeze_signature": freeze["freeze_signature"],
        "fit_ledger_sha256": _sha_file(OFFICIAL_RUN / "fit_ledger.jsonl"),
        "fit_ids": list(FIT_IDS), "fit_units_started": len(FIT_IDS), "fit_units_saved": len(FIT_IDS),
        "fit_units_failed": 0, "prediction_targets_read": False,
        "score_targets_read": False, "test_or_sample_accessed": False,
        "csv_created": False, "artifacts": artifacts,
    }
    body["prediction_signature"] = _canonical_sha(body)
    if PREDICTION_FREEZE.exists():
        existing = _validate_signed_json(PREDICTION_FREEZE, "prediction_signature")
        if existing != body:
            raise PilotError("Existing prediction freeze differs; create a reviewed new revision")
        return existing
    _atomic_json(PREDICTION_FREEZE, body)
    return body


def fit_official(source_root: Path = SOURCE_ROOT) -> dict[str, Any]:
    freeze = _validate_signed_json(FREEZE_PATH, "freeze_signature")
    if freeze.get("status") != "PREFIT_FROZEN" or source_root.resolve() != SOURCE_ROOT.resolve():
        raise PilotError("Official fit source/freeze identity failed")
    _root_source_gate(freeze)
    runner = _runner()
    _assert_frozen_training_source(source_root, freeze, runner)
    if _source_artifact_hashes(runner) != freeze.get("source_artifacts"):
        raise PilotError("Frozen POS42, preprocessing, comparator, or simplification source changed")
    if {key: _sha_file(path) for key, path in IMPLEMENTATION_PATHS.items()} != freeze.get("implementation_hashes"):
        raise PilotError("Pilot implementation or test source changed after prefit freeze")
    hinge_module = _hinge()
    hinge_params = freeze["hinge"]["parameters"]
    if OFFICIAL_RUN.exists() and not (OFFICIAL_RUN / "fit_ledger.jsonl").is_file():
        raise PilotError("Official run root exists without an append-only ledger; refusing ambiguous state")
    outputs: dict[int, Path] = {}
    with _ExclusiveLock(OFFICIAL_RUN / "training.lock"):
        all_events = _read_ledger(OFFICIAL_RUN / "fit_ledger.jsonl")
        prior = [e.get("fit_id") for e in all_events if e.get("event") == "FIT_STARTED"]
        if prior != list(FIT_IDS[:len(prior)]):
            raise PilotError("Official ledger has an unexpected fit order or identity")
        for origin in ORIGINS:
            item, frame, X_fit, X_query = _feature_designs(runner, source_root, freeze, origin)
            _assert_frozen_training_source(source_root, freeze, runner)
            y = runner._read_targets_for_rows(
                runner._contained_file(source_root, runner.TRAIN_REL), item.fit_rows,
                item.fit_ids, expected_total_rows=len(frame))
            if len(y) != len(item.fit_rows) or not np.isfinite(y).all():
                raise PilotError(f"Training-only targets failed fit row validation at E{origin}")
            raw_by_seed: dict[int, np.ndarray] = {}
            for seed in SEEDS:
                fit_id = f"{CARD_ID}_E{origin}_P{seed}"
                counts, weights = _load_weights(freeze, origin, seed, item.fit_rows, item.fit_ids)
                model, receipt, _ = _fit_slot(
                    OFFICIAL_RUN, fit_id, origin, seed, freeze["freeze_signature"],
                    freeze["resolved_parameters"], X_fit, y, counts, weights)
                with threadpool_limits(limits=THREADS):
                    raw = np.asarray(model.predict(X_query), dtype=np.float64)
                if raw.shape != item.query_baseline.shape or not np.isfinite(raw).all():
                    raise PilotError(f"Replica query output invalid at {fit_id}")
                raw_by_seed[seed] = raw
                del model
            base_id = f"{runner.CARD_ID}_E{origin}_REAL"
            base_model, _ = runner._validate_unit(runner.RUN_ROOT, base_id,
                                                  _validate_signed_json(POSITION_FREEZE, "freeze_signature"))
            with threadpool_limits(limits=THREADS):
                base_raw = np.asarray(base_model.predict(X_query), dtype=np.float64)
            saved_base = _explicit_npz(runner._prediction_paths(base_id, runner.RUN_ROOT)[0],
                                       ("raw_auxiliary", "prediction"))
            if not np.array_equal(base_raw, saved_base["raw_auxiliary"]):
                raise PilotError(f"POS42 changed after freeze at E{origin}")
            mean_raw = (base_raw + raw_by_seed[17] + raw_by_seed[137]) / 3.0
            mapped = runner._rank50(mean_raw, item.query_cycles)
            direct = np.clip(0.75 * item.query_baseline + 0.25 * mapped, 0.0, 100.0)
            direct[item.query_missing] = item.query_baseline[item.query_missing]
            hinge = hinge_module.apply_hinge_present_only(direct, item.query_missing,
                                                         hinge_params[str(origin)])
            arrays = _prediction_body(freeze, origin, item, base_raw, raw_by_seed[17],
                                      raw_by_seed[137], direct, hinge,
                                      hinge_params[str(origin)], runner, hinge_module)
            path, receipt_path = _prediction_paths(OFFICIAL_RUN, origin)
            if path.exists() or receipt_path.exists():
                if not path.is_file() or not receipt_path.is_file():
                    raise PilotError(f"Prediction output is partial; manual inspection required: E{origin}")
                saved_receipt = _read_json(receipt_path)
                saved_bundle = _explicit_npz(path, tuple(arrays))
                if (saved_receipt.get("freeze_signature") != freeze["freeze_signature"]
                        or saved_receipt.get("prediction_sha256") != _sha_file(path)
                        or any(not np.array_equal(saved_bundle[k], v) for k, v in arrays.items())):
                    raise PilotError(f"Existing prediction output does not replay exactly: E{origin}")
            else:
                _atomic_npz(path, **arrays)
            query_rows = arrays["query_rows"]
            query_ids = arrays["query_ids"]
            receipt = {
                "schema_version": 1, "experiment_id": CARD_ID, "origin": origin,
                "freeze_signature": freeze["freeze_signature"],
                "prediction_sha256": _sha_file(path), "query_rows": len(query_rows),
                "query_rows_sha256": _sha_array(query_rows), "query_ids_sha256": _sha_array(query_ids),
                "query_cycles_sha256": _sha_array(arrays["query_cycles"]),
                "row_identity_sha256": _canonical_sha({"rows": _sha_array(query_rows),
                                                        "ids": _sha_array(query_ids),
                                                        "cycles": _sha_array(arrays["query_cycles"])}),
                "pos42_raw_sha256": _sha_array(arrays["pos42_raw"]),
                "replica17_raw_sha256": _sha_array(arrays["replica17_raw"]),
                "replica137_raw_sha256": _sha_array(arrays["replica137_raw"]),
                "mean_raw_sha256": _sha_array(arrays["mean_raw"]),
                "mapped_pos_sha256": _sha_array(arrays["mapped_pos"]),
                "full_direct_sha256": _sha_array(arrays["full_direct"]),
                "same_history_hinge_sha256": _sha_array(arrays["same_history_hinge"]),
                "missing_rows": int(arrays["missing_mask"].sum()),
                "query_targets_read": False,
            }
            receipt["receipt_sha256"] = _canonical_sha(receipt)
            if receipt_path.exists():
                if _read_json(receipt_path) != receipt:
                    raise PilotError(f"Existing prediction receipt differs: E{origin}")
            else:
                _atomic_json(receipt_path, receipt)
            outputs[origin] = path
            del X_fit, X_query, frame
        events = _read_ledger(OFFICIAL_RUN / "fit_ledger.jsonl")
        starts = [e["fit_id"] for e in events if e.get("event") == "FIT_STARTED"]
        saves = [e["fit_id"] for e in events if e.get("event") == "FIT_SAVED"]
        failures = [e for e in events if e.get("event") == "FIT_FAILED"]
        if starts != list(FIT_IDS) or saves != list(FIT_IDS) or failures:
            raise PilotError("Official ledger did not finish exactly four ordered fit units")
    prediction_freeze = _write_prediction_freeze(freeze, outputs)
    return {"status": "PREDICTIONS_FROZEN", "freeze_signature": freeze["freeze_signature"],
            "prediction_signature": prediction_freeze["prediction_signature"],
            "fit_units_started": len(FIT_IDS), "fit_units_saved": len(FIT_IDS),
            "prediction_freeze_sha256": _sha_file(PREDICTION_FREEZE)}


def _metric(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = np.asarray(pred, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    return {"mae": float(np.mean(np.abs(err))),
            "p90_absolute_error": float(np.quantile(np.abs(err), 0.90)),
            "bias_prediction_minus_target": float(np.mean(err)), "rows": int(len(err))}


def _decision_gates(pooled_gain: float, d1_gain: float, d2_gain: float,
                    fixed_groups_known: bool, supported_group_max_harm: float,
                    missing_invariant_pass: bool, independent_qa_pass: bool) -> tuple[bool, bool]:
    support_and_harm_pass = bool(fixed_groups_known and supported_group_max_harm < 0.25)
    qualification = bool(pooled_gain >= 0.02 and min(d1_gain, d2_gain) >= 0
                         and support_and_harm_pass and missing_invariant_pass)
    benchmark_eligible = bool(
        (pooled_gain > 0 or (d2_gain >= 0.01 and pooled_gain >= -0.01))
        and d1_gain > -0.05 and support_and_harm_pass
        and missing_invariant_pass and independent_qa_pass)
    return qualification, benchmark_eligible


def _score_gate(freeze: Mapping[str, Any], prediction_freeze: Mapping[str, Any]) -> dict[str, Any]:
    if not ROOT_SCORE_PASS.is_file():
        raise PilotError("Score target access blocked: ROOT_SCORE_PASS_R1.json is absent")
    approval = _validate_signed_json(ROOT_SCORE_PASS, "approval_sha256")
    prelabel_sha = _sha_file(PRELABEL_REPORT) if PRELABEL_REPORT.is_file() else None
    if (approval.get("status") != "PASS"
            or approval.get("source_freeze_signature") != freeze["freeze_signature"]
            or approval.get("prediction_signature") != prediction_freeze.get("prediction_signature")
            or approval.get("score_access_allowed") is not True
            or approval.get("fit_ids") != list(FIT_IDS)
            or approval.get("local_commit") != _git(ROOT, "rev-parse", "HEAD")
            or approval.get("prelabel_report_sha256") != prelabel_sha
            or approval.get("reviewer") in (None, "", "self")
            or approval.get("score_input_sha256") != {str(k): v for k, v in AUTHORIZED_SCORE_SHA256.items()}):
        raise PilotError("Root score approval does not match frozen predictions")
    if not PRELABEL_REPORT.is_file():
        raise PilotError("Score stage requires independent prelabel verification")
    prelabel = _read_json(PRELABEL_REPORT)
    if (prelabel.get("status") != "PASS"
            or prelabel.get("source_freeze_signature") != freeze["freeze_signature"]
            or prelabel.get("prediction_signature") != prediction_freeze.get("prediction_signature")):
        raise PilotError("Independent prelabel verifier does not match frozen predictions")
    return approval


def _target_groups(y: np.ndarray, missing: np.ndarray, fit_group: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    groups = {
        "missing": np.asarray(missing, dtype=bool),
        "target_lt5": np.asarray(y < 5.0),
        "target_ge95": np.asarray(y >= 95.0),
        "EF0": np.asarray(fit_group["entry_friction"] == 0.0),
        "EF1": np.asarray(fit_group["entry_friction"] > 0.0),
    }
    for name in ("impact_load", "text_load"):
        buckets = np.asarray(fit_group[name], dtype=np.int8)
        for bucket in range(4):
            groups[f"{name}_Q{bucket + 1}"] = buckets == bucket
    return groups


def score_official(source_root: Path = SOURCE_ROOT) -> dict[str, Any]:
    freeze = _validate_signed_json(FREEZE_PATH, "freeze_signature")
    prediction_freeze = _validate_signed_json(PREDICTION_FREEZE, "prediction_signature")
    if prediction_freeze.get("status") != "PREDICTIONS_FROZEN":
        raise PilotError("Score stage requires a frozen candidate prediction manifest")
    approval = _score_gate(freeze, prediction_freeze)
    runner = _runner()
    _assert_frozen_training_source(source_root, freeze, runner)
    results: dict[int, dict[str, Any]] = {}
    score_targets: dict[int, np.ndarray] = {}
    score_candidate: dict[int, np.ndarray] = {}
    score_champion: dict[int, np.ndarray] = {}
    missing_invariant_pass = True
    finite_outputs_pass = True
    for origin in ORIGINS:
        path, receipt_path = _prediction_paths(OFFICIAL_RUN, origin)
        if not path.is_file() or not receipt_path.is_file():
            raise PilotError(f"Frozen candidate prediction is incomplete for E{origin}")
        receipt = _read_json(receipt_path)
        if receipt.get("prediction_sha256") != _sha_file(path):
            raise PilotError(f"Prediction artifact hash failed at E{origin}")
        arrays = _explicit_npz(path, ("query_rows", "query_ids", "query_cycles", "missing_mask",
                                      "full_direct", "same_history_hinge"))
        frame = runner._load_feature_frame(source_root)
        item = runner._load_origin(source_root, frame, origin)
        row_map = {int(row): i for i, row in enumerate(item.query_rows)}
        # Score windows are exactly the frozen local-hinge rows, kept target-blind until this gate.
        local = _explicit_npz(LOCAL_PREDICTION_PATHS[origin], (
            "physical_rows", "record_ids", "cycles", "missing_mask", "direct_predictions", "hinge_baseline"))
        positions = np.asarray([row_map[int(row)] for row in local["physical_rows"]], dtype=np.int64)
        rows = np.asarray(local["physical_rows"], dtype=np.int64)
        ids = local["record_ids"].astype(str)
        if (not np.array_equal(item.query_ids[positions].astype(str), ids)
                or not np.array_equal(arrays["query_ids"][positions].astype(str), ids)
                or not np.array_equal(arrays["query_cycles"][positions], local["cycles"])):
            raise PilotError(f"Score row identity differs from frozen predictions at E{origin}")
        score_file = R4_SCORE_PATHS[origin]
        expected_score_sha = approval.get("score_input_sha256", {}).get(str(origin))
        if (expected_score_sha != AUTHORIZED_SCORE_SHA256[origin]
                or _sha_file(score_file) != expected_score_sha):
            raise PilotError(f"Authorized score source hash differs at E{origin}")
        # This is the first official read of evaluation labels, after score approval and prediction freeze.
        _assert_frozen_training_source(source_root, freeze, runner)
        y = runner._read_targets_for_rows(runner._contained_file(source_root, runner.TRAIN_REL),
                                          rows, ids, expected_total_rows=len(frame))
        present = ~local["missing_mask"].astype(bool)
        cand_direct = arrays["full_direct"][positions]
        cand_hinge = arrays["same_history_hinge"][positions]
        champ_direct = local["direct_predictions"].astype(np.float64)
        champ = local["hinge_baseline"].astype(np.float64)
        missing_exact = np.array_equal(
            cand_hinge[~present].view("<u8"),
            item.query_baseline[positions][~present].astype("<f8").view("<u8"))
        missing_invariant_pass = missing_invariant_pass and bool(missing_exact)
        if not missing_exact:
            raise PilotError(f"Candidate missing-position fallback is not bit-exact at E{origin}")
        if not all(np.isfinite(value).all() for value in
                   (y, cand_direct, cand_hinge, champ_direct, champ)):
            finite_outputs_pass = False
            raise PilotError(f"Non-finite target or prediction values at E{origin}")
        group_values = {name: values[positions] for name, values in item.group_features.items()}
        groups = _target_groups(y, local["missing_mask"], group_values)
        group_metrics: dict[str, Any] = {}
        for name, mask in groups.items():
            n = int(mask.sum())
            if n < 100:
                group_metrics[name] = {"status": "UNKNOWN", "rows": n, "harm": None}
            else:
                candidate_mae = _metric(y[mask], cand_hinge[mask])["mae"]
                champ_mae = _metric(y[mask], champ[mask])["mae"]
                group_metrics[name] = {"status": "MEASURED", "rows": n,
                                       "candidate_mae": candidate_mae, "champion_mae": champ_mae,
                                       "harm": candidate_mae - champ_mae}
        results[origin] = {
            "rows": len(y), "cycles": list(SCORE_WINDOWS[origin]),
            "full_direct": _metric(y, cand_direct),
            "full_direct_comparator": _metric(y, champ_direct),
            "full_direct_mae_gain": _metric(y, champ_direct)["mae"] - _metric(y, cand_direct)["mae"],
            "candidate_hinge": _metric(y, cand_hinge),
            "champion_hinge": _metric(y, champ),
            "candidate_mae_gain": _metric(y, champ)["mae"] - _metric(y, cand_hinge)["mae"],
            "per_cycle": {str(int(c)): {
                "candidate_hinge": _metric(y[local["cycles"] == c], cand_hinge[local["cycles"] == c]),
                "champion_hinge": _metric(y[local["cycles"] == c], champ[local["cycles"] == c]),
            } for c in np.unique(local["cycles"])},
            "groups": group_metrics, "score_input_sha256": _sha_file(score_file),
            "score_target_rows_read": len(y), "row_identity_sha256": receipt["row_identity_sha256"],
        }
        score_targets[origin] = y
        score_candidate[origin] = cand_hinge
        score_champion[origin] = champ
    window_gains = [results[o]["candidate_mae_gain"] for o in ORIGINS]
    total_rows = sum(results[o]["rows"] for o in ORIGINS)
    # Pooled calculations concatenate only the two authorized scoring windows.
    pooled_y_arr = np.concatenate([score_targets[o] for o in ORIGINS])
    pooled_candidate = np.concatenate([score_candidate[o] for o in ORIGINS])
    pooled_champion = np.concatenate([score_champion[o] for o in ORIGINS])
    pooled_gain = _metric(pooled_y_arr, pooled_champion)["mae"] - _metric(pooled_y_arr, pooled_candidate)["mae"]
    supported_harms = [entry["harm"] for o in ORIGINS for entry in results[o]["groups"].values()
                       if entry["status"] == "MEASURED"]
    fixed_groups_known = all(entry["status"] == "MEASURED"
                             for o in ORIGINS for name, entry in results[o]["groups"].items()
                             if name in {"missing", "target_lt5", "target_ge95", "EF0", "EF1"})
    supported_group_max_harm = max(supported_harms, default=float("inf"))
    support_and_harm_pass = bool(fixed_groups_known and supported_group_max_harm < 0.25)
    prelabel_verifier_pass = True  # _score_gate returns only after validating the independent prelabel report.
    qa_pass = bool(prelabel_verifier_pass and finite_outputs_pass and support_and_harm_pass
                   and missing_invariant_pass)
    qualification, benchmark_eligible = _decision_gates(
        pooled_gain, results[67]["candidate_mae_gain"], results[75]["candidate_mae_gain"],
        fixed_groups_known, supported_group_max_harm, missing_invariant_pass, qa_pass)
    report = {
        "schema_version": 1, "experiment_id": CARD_ID, "status": "SCORED",
        "source_freeze_signature": freeze["freeze_signature"],
        "prediction_signature": prediction_freeze["prediction_signature"],
        "score_access_approval_sha256": _sha_file(ROOT_SCORE_PASS),
        "pooled_rows": total_rows, "pooled_candidate_hinge": _metric(pooled_y_arr, pooled_candidate),
        "pooled_champion_hinge": _metric(pooled_y_arr, pooled_champion),
        "pooled_mae_gain": pooled_gain, "windows": {str(k): v for k, v in results.items()},
        "evaluation_design_caveat": (
            "D1 is E67 predicting cycles 72-75 (5-8 cycles ahead); D2 is E75 predicting cycles 76-79 (1-4 cycles ahead). "
            "The windows are distinct, but they are not independent because E75 training includes earlier cycles from the D1 window."
        ),
        "supported_group_max_harm": (supported_group_max_harm
                                      if np.isfinite(supported_group_max_harm) else None),
        "fixed_groups_known": fixed_groups_known,
        "prelabel_verifier_pass": prelabel_verifier_pass,
        "finite_outputs_pass": finite_outputs_pass,
        "missing_invariant_pass": missing_invariant_pass,
        "independent_qa_pass": qa_pass,
        "scientific_qualification": qualification, "exploratory_benchmark_eligible": benchmark_eligible,
        "query_targets_read": True, "test_or_sample_accessed": False, "csv_created": False,
    }
    _atomic_json(SCORE_REPORT, report)
    return report


def _synthetic_predictions(origin: int, base: np.ndarray, rep17: np.ndarray, rep137: np.ndarray,
                           baseline: np.ndarray, cycles: np.ndarray, missing: np.ndarray,
                           params: Mapping[str, float]) -> dict[str, np.ndarray]:
    mean_raw = (base + rep17 + rep137) / 3.0
    mapped = np.empty(len(mean_raw), dtype=np.float64)
    for cycle in np.unique(cycles):
        mask = cycles == cycle
        mapped[mask] = 0.5 * mean_raw[mask] + 0.5 * (100.0 * rankdata(mean_raw[mask], method="average") / mask.sum())
    direct = np.clip(0.75 * baseline + 0.25 * mapped, 0.0, 100.0)
    direct[missing] = baseline[missing]
    hinge_module = _hinge()
    hinge = hinge_module.apply_hinge_present_only(direct, missing, params)
    return {"pos42_raw": base, "replica17_raw": rep17, "replica137_raw": rep137,
            "mean_raw": mean_raw, "mapped_pos": mapped, "full_direct": direct,
            "same_history_hinge": hinge, "missing_mask": missing,
            "baseline": baseline, "query_cycles": cycles, "origin": np.asarray([origin])}


def run_synthetic_e2e(report_path: Path = SYNTHETIC_REPORT) -> dict[str, Any]:
    official_ledger = OFFICIAL_RUN / "fit_ledger.jsonl"
    official_before = _sha_file(official_ledger) if official_ledger.is_file() else None
    synthetic_hinge = {"intercept": 1.25, "slope_low": 1.01, "slope_high": 0.96}
    synthetic_body = {"schema_version": 1, "experiment_id": CARD_ID,
                      "synthetic_only": True, "fit_ids": list(FIT_IDS),
                      "parameters": SYNTHETIC_PARAMETERS,
                      "hinge_parameters": synthetic_hinge,
                      "origin_sizes": {"fit": 64, "query": 24, "width": 12}}
    synthetic_signature = _canonical_sha(synthetic_body)
    with tempfile.TemporaryDirectory(prefix="reg-pos-poisson-bag2-") as temporary:
        root = Path(temporary)
        rng = np.random.default_rng(20261106)
        ledger_path = root / "fit_ledger.jsonl"
        metrics: dict[str, Any] = {}
        model_receipts: dict[str, str] = {}
        predictions_dir = root / "predictions"
        predictions_dir.mkdir()
        _atomic_json(root / "SYNTHETIC_SPEC.json", {**synthetic_body,
                                                       "freeze_signature": synthetic_signature})
        for origin_index, origin in enumerate(ORIGINS):
            n_fit, n_query, width = 64, 24, 12
            X_train = sparse.random(n_fit, width, density=0.35, format="csr",
                                    random_state=100 + origin, data_rvs=lambda n: rng.normal(size=n))
            X_query = sparse.random(n_query, width, density=0.35, format="csr",
                                    random_state=200 + origin, data_rvs=lambda n: rng.normal(size=n))
            beta = rng.normal(size=width)
            y_fit = np.clip(48 + np.asarray(X_train @ beta).ravel() * 7 + rng.normal(0, 2, n_fit), 0, 100)
            y_query = np.clip(48 + np.asarray(X_query @ beta).ravel() * 7 + rng.normal(0, 2, n_query), 0, 100)
            base = np.clip(0.7 * y_query + 12, 0, 100).astype(np.float64)
            cycles = np.repeat(np.arange(72 + 4 * origin_index, 76 + 4 * origin_index), 6).astype(np.int64)
            missing = (np.arange(n_query) % 7 == 0)
            baseline = np.clip(0.6 * y_query + 20, 0, 100).astype(np.float64)
            raw_by_seed: dict[int, np.ndarray] = {}
            weight_arrays: dict[str, np.ndarray] = {}
            for seed in SEEDS:
                counts, weights = _weights_for(np.arange(n_fit), np.asarray([f"{origin}-{i}" for i in range(n_fit)]), seed)
                weight_arrays[f"counts_P{seed}"] = counts
                weight_arrays[f"weights_P{seed}"] = weights
                fit_id = f"{CARD_ID}_E{origin}_P{seed}"
                model, receipt, _ = _fit_slot(root, fit_id, origin, seed, synthetic_signature,
                                               SYNTHETIC_PARAMETERS, X_train, y_fit, counts, weights,
                                               synthetic=True)
                model_receipts[fit_id] = receipt["receipt_sha256"]
                model_check, reread_receipt = _validate_unit(root, fit_id, synthetic_signature)
                if reread_receipt["receipt_sha256"] != receipt["receipt_sha256"]:
                    raise PilotError("Synthetic receipt did not survive artifact readback")
                with threadpool_limits(limits=THREADS):
                    raw_by_seed[seed] = np.asarray(model_check.predict(X_query), dtype=np.float64)
            _atomic_npz(root / f"weights_E{origin}.npz", **weight_arrays)
            _atomic_npz(root / f"data_E{origin}.npz",
                        X_fit_data=X_train.data, X_fit_indices=X_train.indices.astype(np.int32),
                        X_fit_indptr=X_train.indptr.astype(np.int32), X_fit_shape=np.asarray(X_train.shape, dtype=np.int64),
                        X_query_data=X_query.data, X_query_indices=X_query.indices.astype(np.int32),
                        X_query_indptr=X_query.indptr.astype(np.int32), X_query_shape=np.asarray(X_query.shape, dtype=np.int64),
                        y_fit=y_fit, y_query=y_query, pos42_raw=base, baseline=baseline,
                        cycles=cycles, missing=missing, hinge_parameters=np.asarray(list(synthetic_hinge.values())))
            candidate = _synthetic_predictions(
                origin, base, raw_by_seed[17], raw_by_seed[137], baseline, cycles, missing,
                synthetic_hinge)
            _atomic_npz(predictions_dir / f"E{origin}.npz", **candidate,
                        synthetic_targets=y_query, fit_targets=y_fit,
                        query_y_sha256=np.asarray([_sha_array(y_query)]))
            with np.load(predictions_dir / f"E{origin}.npz", allow_pickle=False) as pred_check:
                if not np.array_equal(pred_check["full_direct"], candidate["full_direct"]):
                    raise PilotError("Synthetic prediction artifact readback failed")
            metrics[str(origin)] = _metric(y_query, candidate["same_history_hinge"])
        ledger = _read_ledger(ledger_path)
        starts = [e["fit_id"] for e in ledger if e.get("event") == "FIT_STARTED"]
        saves = [e["fit_id"] for e in ledger if e.get("event") == "FIT_SAVED"]
        if starts != list(FIT_IDS) or saves != list(FIT_IDS):
            raise PilotError("Synthetic end-to-end ledger did not record exactly four fits")
        verifier = ROOT / "scripts" / "verify_reg_pos_poisson_bag2_v1.py"
        completed = subprocess.run([sys.executable, str(verifier), "--synthetic-dir", str(root)],
                                   check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            raise PilotError(f"Independent synthetic verifier failed: {completed.stdout}\n{completed.stderr}")
        independent = json.loads(completed.stdout.strip().splitlines()[-1])
        if independent.get("status") != "PASS" or independent.get("fit_units") != 4:
            raise PilotError("Independent synthetic verifier returned an invalid result")
    official_after = _sha_file(official_ledger) if official_ledger.is_file() else None
    if official_before != official_after:
        raise PilotError("Synthetic QA changed the official fit ledger")
    result = {
        "schema_version": 1, "experiment_id": CARD_ID, "status": "PASS",
        "runner_sha256": _sha_file(Path(__file__)),
        "verifier_sha256": _sha_file(ROOT / "scripts" / "verify_reg_pos_poisson_bag2_v1.py"),
        "production_fit_units_started": 0, "official_fit_ledger_before_sha256": official_before,
        "official_fit_ledger_after_sha256": official_after, "official_ledger_mutated": False,
        "synthetic_fit_units": 4, "learned_fit_call_total": 4,
        "independent_verifier": independent, "synthetic_metrics": metrics,
        "test_or_sample_accessed": False, "official_targets_read": False, "csv_created": False,
    }
    _atomic_json(report_path, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--fit", action="store_true")
    modes.add_argument("--score", action="store_true")
    modes.add_argument("--synthetic-e2e", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.preflight:
            result = preflight()
        elif args.fit:
            result = fit_official()
        elif args.score:
            result = score_official()
        else:
            result = run_synthetic_e2e()
        print(json.dumps(_jsonable(result), sort_keys=True, allow_nan=False))
        return 0
    except PilotError as exc:
        print(json.dumps({"status": "BLOCKED", "error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
