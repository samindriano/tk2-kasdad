"""Execute the four authorized frozen-bag positional-offset fits.

Commands are deliberately staged: preflight -> freeze -> reserve -> run-all ->
predictions -> verify-prelabel -> score. The score command is the first point
at which query-prefix labels are read for evaluation.
"""
from __future__ import annotations

import argparse
import csv
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
import uuid
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import joblib
import numpy as np
import pandas as pd
import psutil
from scipy import sparse
from scipy.special import expit
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             log_loss, precision_score, recall_score, roc_auc_score)
from threadpoolctl import threadpool_limits

from src.models.positional_bag_offset_v1 import (OffsetFitError, fit_offset_coefficients,
                                                 offset_probability)


CARD_ID = "POS_BAG_OFFSET_V1"
STAGE = "FOUR_FIT_FROZEN_BAG_OFFSET"
AMENDMENT_ID = "POS_BAG_OFFSET_CAP109_2026_11_03"
RESERVATION_KEY = f"{CARD_ID}:{STAGE}"
FIT_IDS = (
    "POS_BAG_OFFSET_V1_E67_REAL",
    "POS_BAG_OFFSET_V1_E67_SHUFFLED",
    "POS_BAG_OFFSET_V1_E75_REAL",
    "POS_BAG_OFFSET_V1_E75_SHUFFLED",
)
ORIGINS = (67, 75)
TRAIN = ROOT / "data/raw/classification/Train.csv"
RECIPE = ROOT / "reports/optimization_v3/positional_bag_offset_v1/RECIPE.json"
AUTH = ROOT / "reports/optimization_v3/positional_text_diagnostic_v1/SOL_OFFSET_PILOT_AUTHORIZATION.md"
PILOT_REPORTS = ROOT / "reports/optimization_v3/positional_text_pilot_v1"
PILOT_ROOT = ROOT / "experiments/runs/optimization_v3_research_next_v1/POSITIONAL_TEXT_PILOT_V1"
LEDGER = ROOT / "experiments/runs/optimization_v3_research_next_v1/budget.jsonl"
RUN_ROOT = ROOT / "experiments/runs/optimization_v3_research_next_v1/POS_BAG_OFFSET_V1"
REPORT_ROOT = ROOT / "reports/optimization_v3/positional_bag_offset_v1"
ACCOUNTING = PILOT_REPORTS / "FIT_ACCOUNTING.json"
INITIAL_CAPS = {"continuation": 38, "control_replay": 2, "pilot": 64, "technical_retry": 1}
AMENDED_CAPS = {"continuation": 38, "control_replay": 2, "pilot": 68, "technical_retry": 1}
RSS_LIMIT = 6 * 1024**3
INITIAL_FREE_MIN = 4 * 1024**3
FREE_RESERVE = 2 * 1024**3
SOURCE_FILES = (
    "scripts/optimization_v3_positional_bag_offset_v1.py",
    "scripts/verify_optimization_v3_positional_bag_offset_v1.py",
    "src/models/positional_bag_offset_v1.py",
    "tests/test_optimization_v3_positional_bag_offset_v1.py",
    "reports/optimization_v3/positional_bag_offset_v1/RECIPE.json",
    "reports/optimization_v3/positional_text_diagnostic_v1/SOL_OFFSET_PILOT_AUTHORIZATION.md",
)
PILOT_DEPENDENCIES = (
    "scripts/optimization_v3_positional_text_pilot_v1.py",
    "src/models/optimization_v3_positional_text_pilot_v1.py",
    "src/models/classification_models.py",
    "src/models/round1_models.py",
    "src/models/positional_text_features.py",
)


class OffsetPilotError(RuntimeError):
    pass


_LOCAL_FIT_LOCK = threading.Lock()


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
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(_jsonable(value), sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(",", ":")).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: Any) -> str:
    array = np.asarray(value)
    header = canonical_sha256({"shape": array.shape, "dtype": str(array.dtype)})
    digest = hashlib.sha256(header.encode("ascii"))
    if array.dtype.kind in "OUS":
        digest.update(json.dumps(array.astype(str).tolist(), ensure_ascii=False,
                                 separators=(",", ":")).encode("utf-8"))
    else:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def sha256_csr(value: sparse.spmatrix) -> str:
    matrix = sparse.csr_matrix(value, dtype=np.float64, copy=True)
    matrix.sum_duplicates()
    matrix.sort_indices()
    digest = hashlib.sha256(canonical_sha256({"shape": matrix.shape, "dtype": str(matrix.dtype)}).encode("ascii"))
    digest.update(np.asarray(matrix.indptr, dtype="<i8").tobytes())
    digest.update(np.asarray(matrix.indices, dtype="<i8").tobytes())
    digest.update(np.asarray(matrix.data, dtype="<f8").tobytes())
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, json.dumps(_jsonable(value), sort_keys=True, ensure_ascii=False,
                                   allow_nan=False, indent=2).encode("utf-8") + b"\n")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise OffsetPilotError(f"Expected JSON object: {path}")
    return value


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, check=True, text=True,
                            capture_output=True)
    return result.stdout.strip()


def _current_source_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for rel in (*SOURCE_FILES, *PILOT_DEPENDENCIES):
        path = ROOT / rel
        if not path.is_file():
            raise OffsetPilotError(f"Frozen source is missing: {rel}")
        hashes[rel] = sha256_file(path)
    pilot_freeze = _read_json(PILOT_ROOT / "FREEZE.json")
    for rel, expected in pilot_freeze["source_hashes"].items():
        path = ROOT / rel
        if not path.is_file() or sha256_file(path) != expected:
            raise OffsetPilotError(f"Historical pilot source hash changed: {rel}")
    return hashes


def _ledger_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise OffsetPilotError(f"Ledger is missing: {path}")
    events: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        event = json.loads(line)
        if not isinstance(event, dict) or not isinstance(event.get("event"), str):
            raise OffsetPilotError(f"Invalid ledger event at line {line_no}")
        events.append(event)
    return events


def ledger_snapshot(events: list[dict[str, Any]]) -> dict[str, Any]:
    authorizations = [e for e in events if e.get("event") == "BATCH_ALLOWANCE_AUTHORIZED"]
    if len(authorizations) != 1:
        raise OffsetPilotError("Expected one Research Next base authorization")
    global_cap = int(authorizations[0]["cap"])
    caps = {str(k): int(v) for k, v in authorizations[0]["category_caps"].items()}
    for event in events:
        if event.get("event") == "BUDGET_CAP_AMENDMENT":
            global_cap = int(event["global_cap"])
            caps = {str(k): int(v) for k, v in event["effective_caps"].items()}
        elif event.get("event") == "BUDGET_CATEGORY_AMENDMENT":
            if event.get("global_cap") is not None:
                global_cap = int(event["global_cap"])
            if event.get("effective_caps") is not None:
                caps = {str(k): int(v) for k, v in event["effective_caps"].items()}
    released = {str(e["reservation_key"]): int(e.get("unused_fits", 0))
                for e in events if e.get("event") == "RESERVATION_UNUSED_RELEASED"}
    active = {key: 0 for key in caps}
    for event in events:
        if event.get("event") == "CARD_FITS_RESERVED":
            category = str(event.get("category"))
            if category not in active:
                raise OffsetPilotError(f"Unknown reservation category: {category}")
            active[category] += int(event["fits"]) - released.get(str(event["reservation_key"]), 0)
    starts = [e for e in events if e.get("event") == "FIT_STARTED"]
    saved = [e for e in events if e.get("event") == "FIT_SAVED"]
    failed = [e for e in events if e.get("event") == "FIT_FAILED"]
    terminal = {str(e.get("fit_id")) for e in events
                if e.get("event") in {"FIT_SAVED", "FIT_FAILED"}}
    return {
        "event_rows": len(events), "global_cap": global_cap,
        "effective_category_caps": caps,
        "active_reservations_by_category": active,
        "active_reservations": sum(active.values()),
        "fit_starts": len(starts), "fit_saved": len(saved),
        "fit_failed": len(failed),
        "fit_pending": sum(str(e.get("fit_id")) not in terminal for e in starts),
    }


def _append_ledger(event: Mapping[str, Any], *, path: Path, official: bool = False) -> dict[str, Any]:
    target = path.resolve()
    if target == LEDGER.resolve() and not official:
        raise OffsetPilotError("Production ledger writes require the official run path")
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


def _resource_snapshot(estimate_bytes: int, *, initial: bool = False) -> dict[str, int]:
    process = psutil.Process()
    rss = int(process.memory_info().rss)
    available = int(psutil.virtual_memory().available)
    result = {
        "jobs": 1, "threads": 2, "process_rss_bytes": rss,
        "available_ram_bytes": available,
        "estimated_incremental_memory_bytes": int(estimate_bytes),
        "projected_rss_bytes": rss + int(estimate_bytes),
        "projected_available_ram_bytes": available - int(estimate_bytes),
        "max_rss_bytes": RSS_LIMIT, "initial_available_ram_min_bytes": INITIAL_FREE_MIN,
        "minimum_free_ram_bytes": FREE_RESERVE,
    }
    if initial and available < INITIAL_FREE_MIN:
        raise OffsetPilotError(f"Initial free RAM below 4 GiB: {available}")
    if result["projected_rss_bytes"] > RSS_LIMIT:
        raise OffsetPilotError("Projected RSS exceeds 6 GiB cap")
    if result["projected_available_ram_bytes"] < FREE_RESERVE:
        raise OffsetPilotError("Projected free RAM below 2 GiB reserve")
    return result


def _matrix_path(item: Mapping[str, Any]) -> Path:
    path = Path(str(item["path"]))
    return path if path.is_absolute() else ROOT / path


def _load_matrix(manifest: Mapping[str, Any], name: str) -> sparse.csr_matrix | np.ndarray:
    item = manifest["matrices"][name]
    path = _matrix_path(item)
    if not path.is_file() or sha256_file(path) != item["sha256"]:
        raise OffsetPilotError(f"Matrix file hash mismatch: {name}")
    matrix: Any = sparse.load_npz(path).tocsr() if path.suffix == ".npz" else np.load(path, allow_pickle=False)
    logical = sha256_csr(matrix) if sparse.issparse(matrix) else sha256_array(matrix)
    if list(matrix.shape) != item["shape"] or logical != item["logical_sha256"]:
        raise OffsetPilotError(f"Matrix logical shape/hash mismatch: {name}")
    return matrix


def _row_metadata(frame: pd.DataFrame, rows: np.ndarray) -> dict[str, Any]:
    rows = np.asarray(rows, dtype=np.int64)
    ids = frame.iloc[rows]["record_id"].astype(str).to_numpy()
    cycles = pd.to_numeric(frame.iloc[rows]["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    return {
        "rows": int(len(rows)), "cycle_min": int(cycles.min()), "cycle_max": int(cycles.max()),
        "row_indices_sha256": sha256_array(rows), "record_ids_sha256": sha256_array(ids),
        "cycles_sha256": sha256_array(cycles),
    }


def _fit_and_score_rows(origin: int, prediction: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    fit_rows = np.asarray(prediction["fit_rows"], dtype=np.int64)
    score_rows = np.asarray(prediction["score_rows"], dtype=np.int64)
    if len(np.intersect1d(fit_rows, score_rows)):
        raise OffsetPilotError(f"E{origin} fit and query rows overlap")
    return fit_rows, score_rows


def _read_identity_frame() -> pd.DataFrame:
    frame = pd.read_csv(TRAIN, usecols=["record_id", "cycle_code"], dtype={"record_id": "string"})
    if len(frame) != 50608 or frame["record_id"].isna().any() or frame["record_id"].duplicated().any():
        raise OffsetPilotError("Official Train identity rows fail count/uniqueness checks")
    frame["cycle_code"] = pd.to_numeric(frame["cycle_code"], errors="raise").astype(np.int64)
    return frame


def _verify_estimator(unit_id: str, accounting: Mapping[str, Any], *, width: int) -> tuple[dict[str, Any], dict[str, Any]]:
    run_unit = PILOT_ROOT / "units" / f"{unit_id}.joblib"
    receipt_path = PILOT_ROOT / "receipts" / f"{unit_id}.json"
    if not run_unit.is_file() or not receipt_path.is_file():
        raise OffsetPilotError(f"Saved pilot unit/receipt missing: {unit_id}")
    rec = _read_json(receipt_path)
    if sha256_file(run_unit) != accounting["artifact_sha256"] or sha256_file(receipt_path) != accounting["receipt_sha256"]:
        raise OffsetPilotError(f"Saved pilot unit or receipt hash differs from FIT_ACCOUNTING: {unit_id}")
    if rec.get("artifact_sha256") != sha256_file(run_unit) or rec.get("fit_id") != unit_id:
        raise OffsetPilotError(f"Saved pilot receipt lineage mismatch: {unit_id}")
    unit = joblib.load(run_unit)
    estimator = unit.get("estimator")
    if estimator is None:
        raise OffsetPilotError(f"Saved pilot estimator absent: {unit_id}")
    return {"unit": unit, "estimator": estimator, "artifact_path": run_unit}, rec


def _verify_preprocessors(origin: int, manifest: Mapping[str, Any],
                          fit_rows: np.ndarray, frame: pd.DataFrame,
                          accounting_by_id: Mapping[str, Any]) -> dict[str, Any]:
    from src.models.optimization_v3_positional_text_pilot_v1 import structured_state_sha256

    shared = manifest["shared_preprocessing_sha256"]
    ids = frame.iloc[fit_rows]["record_id"].astype(str).to_numpy()
    cycles = frame.iloc[fit_rows]["cycle_code"].to_numpy(dtype=np.int64)
    expected_rows = {
        "rows": len(fit_rows), "cycle_min": int(cycles.min()), "cycle_max": int(cycles.max()),
        "cycles_sha256": sha256_array(cycles), "record_ids_sha256": sha256_array(ids),
        "row_indices_sha256": sha256_array(fit_rows),
    }
    checked: dict[str, Any] = {}
    units: dict[str, Any] = {}
    for component in ("C2B_STRUCTURED", "C2B_SCALER", "C2B_OHE", "C2B_TFIDF", "C1B_STRUCTURED"):
        unit_id = f"POSITIONAL_TEXT_PILOT_V1_E{origin}_{component}"
        acc = accounting_by_id.get(unit_id)
        if acc is None or acc.get("fit_kind") != "PREPROCESSING":
            raise OffsetPilotError(f"Saved preprocessing fit accounting missing: {unit_id}")
        artifact = PILOT_ROOT / "units" / f"{unit_id}.joblib"
        receipt_path = PILOT_ROOT / "receipts" / f"{unit_id}.json"
        if (not artifact.is_file() or not receipt_path.is_file()
                or sha256_file(artifact) != acc["artifact_sha256"]
                or sha256_file(receipt_path) != acc["receipt_sha256"]):
            raise OffsetPilotError(f"Saved preprocessing artifact/receipt hash mismatch: {unit_id}")
        receipt = _read_json(receipt_path)
        population = acc.get("fit_population", {})
        if (receipt.get("fit_id") != unit_id or receipt.get("artifact_sha256") != sha256_file(artifact)
                or receipt.get("fit_metadata") != acc.get("fit_metadata")
                or any(population.get(k) != v for k, v in expected_rows.items())):
            raise OffsetPilotError(f"Saved preprocessing state lineage mismatch: {unit_id}")
        units[component] = joblib.load(artifact)
        checked[component] = {"artifact_sha256": acc["artifact_sha256"],
                              "receipt_sha256": acc["receipt_sha256"],
                              "fit_rows_sha256": population["row_indices_sha256"]}
    c2b_state = units["C2B_STRUCTURED"]
    c1b_state = units["C1B_STRUCTURED"]
    scaler = units["C2B_SCALER"]
    ohe = units["C2B_OHE"]
    vectorizer = units["C2B_TFIDF"]
    ohe_hash = canonical_sha256([[str(value) for value in values] for values in ohe.categories_])
    tfidf_hash = canonical_sha256({
        "vocabulary": sorted((str(key), int(value)) for key, value in vectorizer.vocabulary_.items()),
        "idf_sha256": sha256_array(vectorizer.idf_),
    })
    state_hash = structured_state_sha256(c2b_state)
    c1b_hash = structured_state_sha256(c1b_state)
    if (state_hash != shared["structured_state"] or c1b_hash != shared["c1b_structured_state"]
            or sha256_array(scaler.mean_) != shared["scaler_mean"]
            or sha256_array(scaler.scale_) != shared["scaler_scale"]
            or ohe_hash != shared["ohe_categories"]
            or tfidf_hash != shared["tfidf_vocabulary_idf"]):
        raise OffsetPilotError(f"E{origin} fitted preprocessing state differs from matrix manifest hashes")
    return {"units": checked, "shared_state_hashes": {
        "structured_state": state_hash, "c1b_structured_state": c1b_hash,
        "scaler_mean": sha256_array(scaler.mean_), "scaler_scale": sha256_array(scaler.scale_),
        "ohe_categories": ohe_hash, "tfidf_vocabulary_idf": tfidf_hash,
    }}


def verify_saved_inputs(*, include_current_code: bool = True) -> dict[str, Any]:
    if sha256_file(TRAIN) != json.loads(RECIPE.read_text(encoding="utf-8"))["input"]["sha256"]:
        raise OffsetPilotError("Official Classification Train hash differs from the frozen source")
    recipe = _read_json(RECIPE)
    pilot_freeze = _read_json(PILOT_ROOT / "FREEZE.json")
    pilot_results = _read_json(PILOT_ROOT / "RESULTS.json")
    accounting = _read_json(ACCOUNTING)
    frame = _read_identity_frame()
    if pilot_freeze.get("freeze_signature") != "f310d0340ea4402677ca9e86aa29c43b1536ded902c0f6a6f52cb53bd333c292":
        raise OffsetPilotError("Historical pilot freeze signature changed")
    if pilot_freeze.get("input_sha256") != recipe["input"]["sha256"] or len(frame) != recipe["input"]["rows"]:
        raise OffsetPilotError("Historical pilot source/input boundary mismatch")
    if accounting.get("live_ledger_sha256") != "59ef4f3c91091cf0f81d71fdb95211992ff796672227f764bc9523e7304ed6a1":
        raise OffsetPilotError("Historical pilot accounting ledger checkpoint mismatch")
    if accounting.get("train_sha256") != recipe["input"]["sha256"]:
        raise OffsetPilotError("Historical pilot accounting has a different Train input")
    current_source_hashes = _current_source_hashes() if include_current_code else {}
    origin_results: dict[str, Any] = {}
    all_accounting = {x["fit_id"]: x for x in accounting["fit_units"]}
    for origin in ORIGINS:
        suffix = f"E{origin}"
        manifest_path = PILOT_ROOT / f"MATRIX_MANIFEST_{suffix}.json"
        manifest = _read_json(manifest_path)
        manifest_file_hash = sha256_file(manifest_path)
        if (manifest.get("freeze_signature") != pilot_freeze["freeze_signature"]
                or manifest.get("input_sha256") != recipe["input"]["sha256"]
                or accounting["matrix_manifests"][suffix]["sha256"] != manifest_file_hash):
            raise OffsetPilotError(f"E{origin} matrix manifest source hash mismatch")
        prediction_path = PILOT_ROOT / f"E{origin}" / f"predictions_E{origin}.npz"
        if not prediction_path.is_file() or sha256_file(prediction_path) != accounting["prediction_bundles"][suffix]["sha256"]:
            raise OffsetPilotError(f"E{origin} saved prediction bundle hash mismatch")
        with np.load(prediction_path, allow_pickle=False) as bundle:
            prediction = {key: bundle[key].copy() for key in bundle.files}
        if (int(prediction["origin"][0]) != origin
                or str(prediction["matrix_manifest_sha256"][0]) != manifest_file_hash):
            raise OffsetPilotError(f"E{origin} prediction bundle lineage mismatch")
        fit_rows, score_rows = _fit_and_score_rows(origin, prediction)
        if _row_metadata(frame, fit_rows) != manifest["fit_row_metadata"]:
            raise OffsetPilotError(f"E{origin} fit-row IDs/cycles differ from saved matrix metadata")
        if _row_metadata(frame, score_rows) != manifest["score_row_metadata"]:
            raise OffsetPilotError(f"E{origin} query-row IDs/cycles differ from saved matrix metadata")
        expected = recipe["origins"][suffix]
        if len(fit_rows) != expected["fit_rows"] or len(score_rows) != expected["score_rows"]:
            raise OffsetPilotError(f"E{origin} row counts differ from the frozen recipe")
        if not (np.all((frame.iloc[fit_rows]["cycle_code"] >= expected["fit_cycles"][0])
                       & (frame.iloc[fit_rows]["cycle_code"] <= expected["fit_cycles"][1]))
                and np.all((frame.iloc[score_rows]["cycle_code"] >= expected["score_cycles"][0])
                           & (frame.iloc[score_rows]["cycle_code"] <= expected["score_cycles"][1]))):
            raise OffsetPilotError(f"E{origin} cycle boundaries differ from recipe")
        preprocessing = _verify_preprocessors(origin, manifest, fit_rows, frame, all_accounting)
        matrices = {name: _load_matrix(manifest, name) for name in manifest["matrices"]}
        base_fit = sparse.csr_matrix(matrices["fit_base"], dtype=np.float64)
        base_score = sparse.csr_matrix(matrices["score_base"], dtype=np.float64)
        for name in ("fit_position", "fit_shuffled_position", "score_position", "score_shuffled_position"):
            matrix = sparse.csr_matrix(matrices[name], dtype=np.float64)
            base = base_fit if name.startswith("fit_") else base_score
            if matrix.shape[0] != base.shape[0] or matrix.shape[1] <= base.shape[1]:
                raise OffsetPilotError(f"E{origin} {name} dimensions do not contain base + positional columns")
            if sha256_csr(matrix[:, :base.shape[1]]) != sha256_csr(base):
                raise OffsetPilotError(f"E{origin} {name} base feature prefix differs from frozen bag matrix")
        if base_fit.shape[0] != len(fit_rows) or base_score.shape[0] != len(score_rows):
            raise OffsetPilotError(f"E{origin} base matrices do not align with frozen row indices")
        del matrices

        bag_id = f"POSITIONAL_TEXT_PILOT_V1_E{origin}_BAG_LR"
        bag_acc = all_accounting.get(bag_id)
        if bag_acc is None or bag_acc.get("fit_kind") != "PREDICTOR":
            raise OffsetPilotError(f"E{origin} faithful C2_B bag fit accounting missing")
        bag_obj, bag_receipt = _verify_estimator(bag_id, bag_acc, width=base_fit.shape[1])
        bag = bag_obj["estimator"]
        if (list(np.asarray(bag.classes_).tolist()) != [0, 1]
                or bag.coef_.shape != (1, base_fit.shape[1])
                or np.asarray(bag.intercept_).shape != (1,)
                or bag.solver != "liblinear" or float(bag.C) != 1.0
                or int(bag.max_iter) != 2000):
            raise OffsetPilotError(f"E{origin} saved bag estimator class/order/recipe mismatch")
        if (bag_acc["fit_metadata"]["coef_sha256"] != sha256_array(bag.coef_)
                or bag_acc["fit_metadata"]["intercept_sha256"] != sha256_array(bag.intercept_)
                or bag_acc["fit_metadata"]["fit_matrix_sha256"] != sha256_csr(base_fit)
                or bag_acc["fit_metadata"]["score_matrix_sha256"] != sha256_csr(base_score)):
            raise OffsetPilotError(f"E{origin} saved bag coefficient/intercept/matrix hash mismatch")
        fit_population = bag_acc["fit_population"]
        if (fit_population.get("matrix_manifest_sha256") != manifest_file_hash
                or fit_population.get("row_indices_sha256") != sha256_array(fit_rows)
                or fit_population.get("record_ids_sha256") != sha256_array(frame.iloc[fit_rows]["record_id"].astype(str).to_numpy())):
            raise OffsetPilotError(f"E{origin} saved bag fit population differs from exact rows")
        bag_prob = np.asarray(bag_obj["unit"]["probability"], dtype=np.float64)
        if (bag_acc["fit_metadata"]["probability_sha256"] != sha256_array(bag_prob)
                or not np.allclose(bag_prob, prediction["prob_bag"], rtol=0, atol=1e-14)):
            raise OffsetPilotError(f"E{origin} faithful bag probability bundle mismatch")

        hgb_id = f"POSITIONAL_TEXT_PILOT_V1_E{origin}_MISSING_HGB"
        hgb_acc = all_accounting.get(hgb_id)
        if hgb_acc is None:
            raise OffsetPilotError(f"E{origin} saved missing-stream HGB accounting missing")
        hgb_obj, _ = _verify_estimator(hgb_id, hgb_acc, width=0)
        fallback = np.asarray(hgb_obj["unit"]["probability"], dtype=np.float64)
        if (hgb_acc["fit_metadata"]["probability_sha256"] != sha256_array(fallback)
                or not np.array_equal(fallback, prediction["prob_missing_hgb"])):
            raise OffsetPilotError(f"E{origin} exact saved HGB fallback probabilities mismatch")
        present = np.asarray(prediction["present_mask"], dtype=bool)
        if (prediction["score_record_ids"].astype(str).tolist()
                != frame.iloc[score_rows]["record_id"].astype(str).tolist()
                or not np.array_equal(prediction["score_cycles"], frame.iloc[score_rows]["cycle_code"].to_numpy(dtype=np.int64))):
            raise OffsetPilotError(f"E{origin} prediction row IDs/cycles no longer align")
        z0_query = np.asarray(bag.intercept_[0] + base_score @ bag.coef_[0], dtype=np.float64).reshape(-1)
        if not np.allclose(expit(z0_query)[present], prediction["prob_bag"][present], rtol=0, atol=1e-12):
            raise OffsetPilotError(f"E{origin} direct saved logit does not reproduce bag probabilities")
        origin_results[suffix] = {
            "matrix_manifest_sha256": manifest_file_hash,
            "prediction_bundle_sha256": sha256_file(prediction_path),
            "matrix_file_hashes": {k: v["sha256"] for k, v in _read_json(manifest_path)["matrices"].items()},
            "matrix_logical_hashes": {k: v["logical_sha256"] for k, v in _read_json(manifest_path)["matrices"].items()},
            "fit_rows": int(len(fit_rows)), "score_rows": int(len(score_rows)),
            "fit_rows_sha256": sha256_array(fit_rows), "score_rows_sha256": sha256_array(score_rows),
            "bag_artifact_sha256": bag_acc["artifact_sha256"],
            "bag_receipt_sha256": bag_acc["receipt_sha256"],
            "bag_coef_sha256": bag_acc["fit_metadata"]["coef_sha256"],
            "bag_intercept_sha256": bag_acc["fit_metadata"]["intercept_sha256"],
            "missing_hgb_artifact_sha256": hgb_acc["artifact_sha256"],
            "missing_hgb_receipt_sha256": hgb_acc["receipt_sha256"],
            "preprocessing": preprocessing,
        }
    if pilot_results.get("freeze_signature") != pilot_freeze["freeze_signature"]:
        raise OffsetPilotError("Historical pilot results are not linked to the verified freeze")
    return {
        "input_sha256": sha256_file(TRAIN), "input_rows": len(frame),
        "pilot_freeze_signature": pilot_freeze["freeze_signature"],
        "pilot_recipe_sha256": sha256_file(PILOT_REPORTS / "RECIPE.json"),
        "pilot_accounting_sha256": sha256_file(ACCOUNTING),
        "source_hashes": current_source_hashes,
        "origins": origin_results,
    }


def _expected_pre_reservation_snapshot() -> dict[str, Any]:
    return {
        "global_cap": 105,
        "effective_category_caps": dict(INITIAL_CAPS),
        "active_reservations_by_category": dict(INITIAL_CAPS),
        "active_reservations": 105,
        "fit_starts": 105, "fit_saved": 102, "fit_failed": 3, "fit_pending": 0,
    }


def preflight() -> dict[str, Any]:
    if not TRAIN.is_file() or not LEDGER.is_file():
        raise OffsetPilotError("Official Train or live Research Next ledger missing")
    if len(_git("status", "--porcelain")) != 0:
        raise OffsetPilotError("Working tree must be clean before the frozen implementation preflight")
    state = verify_saved_inputs()
    events = _ledger_events(LEDGER)
    snapshot = ledger_snapshot(events)
    if snapshot != {**_expected_pre_reservation_snapshot(), "event_rows": snapshot["event_rows"]}:
        raise OffsetPilotError(f"Live ledger is not at the authorized 105-fit checkpoint: {snapshot}")
    if sha256_file(LEDGER) != "59ef4f3c91091cf0f81d71fdb95211992ff796672227f764bc9523e7304ed6a1":
        raise OffsetPilotError("Live ledger hash moved from verified checkpoint")
    if any(e.get("event") == "FIT_STARTED" and e.get("fit_id") in FIT_IDS for e in events):
        raise OffsetPilotError("POS_BAG_OFFSET_V1 has a prior fit start; no retry path")
    resource = _resource_snapshot(1024**3, initial=True)
    result = {
        "card_id": CARD_ID, "status": "PASS_READY_TO_FREEZE",
        "source_commit": _git("rev-parse", "HEAD"),
        "branch": _git("branch", "--show-current"),
        "input_sha256": state["input_sha256"], "origins": state["origins"],
        "ledger_sha256": sha256_file(LEDGER), "ledger_snapshot": snapshot,
        "resource": resource, "fit_ids": list(FIT_IDS),
    }
    _atomic_json(RUN_ROOT / "PREFLIGHT.json", result)
    return result


def freeze() -> dict[str, Any]:
    existing_path = RUN_ROOT / "FREEZE.json"
    if existing_path.is_file():
        return load_freeze()
    if len(_git("status", "--porcelain")) != 0:
        raise OffsetPilotError("Freeze requires clean committed code")
    state = verify_saved_inputs()
    events = _ledger_events(LEDGER)
    snapshot = ledger_snapshot(events)
    expected = {**_expected_pre_reservation_snapshot(), "event_rows": snapshot["event_rows"]}
    if snapshot != expected or sha256_file(LEDGER) != "59ef4f3c91091cf0f81d71fdb95211992ff796672227f764bc9523e7304ed6a1":
        raise OffsetPilotError("Live ledger no longer matches pre-reservation checkpoint")
    freeze_body = {
        "schema_version": 1, "card_id": CARD_ID,
        "source_commit": _git("rev-parse", "HEAD"),
        "branch": _git("branch", "--show-current"),
        "input_sha256": state["input_sha256"], "input_path": TRAIN.relative_to(ROOT).as_posix(),
        "source_hashes": state["source_hashes"],
        "pilot_freeze_signature": state["pilot_freeze_signature"],
        "pilot_recipe_sha256": state["pilot_recipe_sha256"],
        "pilot_accounting_sha256": state["pilot_accounting_sha256"],
        "origins": state["origins"],
        "pre_reservation_ledger_sha256": sha256_file(LEDGER),
        "pre_reservation_ledger_snapshot": snapshot,
        "fit_ids": list(FIT_IDS), "c": 1.0,
        "optimizer": {"method": "L-BFGS-B", "maxiter": 2000, "ftol": 1e-12, "gtol": 1e-6, "maxls": 50},
        "resource_policy": {"jobs": 1, "threads": 2, "rss_limit_bytes": RSS_LIMIT,
                            "initial_free_min_bytes": INITIAL_FREE_MIN, "free_reserve_bytes": FREE_RESERVE},
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    freeze_body["freeze_signature"] = canonical_sha256(freeze_body)
    path = existing_path
    _atomic_json(path, freeze_body)
    return freeze_body


def load_freeze() -> dict[str, Any]:
    freeze_data = _read_json(RUN_ROOT / "FREEZE.json")
    body = dict(freeze_data)
    signature = body.pop("freeze_signature", None)
    if signature != canonical_sha256(body):
        raise OffsetPilotError("Amendment freeze signature mismatch")
    if (_git("rev-parse", "HEAD") != freeze_data["source_commit"]
            or _git("branch", "--show-current") != freeze_data["branch"]):
        raise OffsetPilotError("Source commit/branch changed after freeze")
    if len(_git("status", "--porcelain")) != 0:
        raise OffsetPilotError("Working tree changed after frozen implementation commit")
    current = _current_source_hashes()
    if current != freeze_data["source_hashes"]:
        raise OffsetPilotError("Frozen source hash changed after commit")
    if sha256_file(TRAIN) != freeze_data["input_sha256"]:
        raise OffsetPilotError("Official Train changed after freeze")
    current_inputs = verify_saved_inputs()
    if (current_inputs["origins"] != freeze_data["origins"]
            or current_inputs["source_hashes"] != freeze_data["source_hashes"]):
        raise OffsetPilotError("Reused pilot matrices/models/predictions differ from their frozen origin hashes")
    return freeze_data


def reserve() -> dict[str, Any]:
    freeze_data = load_freeze()
    events = _ledger_events(LEDGER)
    snapshot = ledger_snapshot(events)
    existing_amendments = [e for e in events if e.get("event") == "BUDGET_CAP_AMENDMENT" and e.get("amendment_id") == AMENDMENT_ID]
    existing_reservations = [e for e in events if e.get("event") == "CARD_FITS_RESERVED" and e.get("reservation_key") == RESERVATION_KEY]
    if len(existing_amendments) > 1 or len(existing_reservations) > 1:
        raise OffsetPilotError("Duplicate POS_BAG_OFFSET_V1 cap amendment or fit reservation")
    if not existing_amendments and not existing_reservations:
        if sha256_file(LEDGER) != freeze_data["pre_reservation_ledger_sha256"] or snapshot != freeze_data["pre_reservation_ledger_snapshot"]:
            raise OffsetPilotError("Live ledger changed after freeze; refuse reservation")
        amendment = {
            "event": "BUDGET_CAP_AMENDMENT", "batch_id": "optimization_v3_research_next_v1",
            "amendment_id": AMENDMENT_ID, "card_id": CARD_ID,
            "authorized_by": "Sam via Sol; exactly four positional bag-offset predictor fits",
            "prior_global_cap": 105, "global_cap": 109,
            "prior_effective_caps": dict(INITIAL_CAPS), "effective_caps": dict(AMENDED_CAPS),
            "active_reservations_before": dict(INITIAL_CAPS), "active_total_before": 105,
            "historical_started_saved_failed_before": {"started": 105, "saved": 102, "failed": 3},
            "pre_amendment_event_rows": len(events),
            "pre_amendment_ledger_sha256": sha256_file(LEDGER),
            "source_commit": freeze_data["source_commit"], "freeze_signature": freeze_data["freeze_signature"],
            "reason": "Authorized exact four-fit POS_BAG_OFFSET_V1 follow-up; no preprocessing fits; other category caps unchanged.",
        }
        _append_ledger(amendment, path=LEDGER, official=True)
        events = _ledger_events(LEDGER)
        snapshot = ledger_snapshot(events)
    if len(existing_amendments) == 1:
        amendment = existing_amendments[0]
    else:
        amendment = next(e for e in events if e.get("event") == "BUDGET_CAP_AMENDMENT" and e.get("amendment_id") == AMENDMENT_ID)
    if (amendment.get("global_cap") != 109 or amendment.get("effective_caps") != AMENDED_CAPS
            or amendment.get("source_commit") != freeze_data["source_commit"]
            or amendment.get("freeze_signature") != freeze_data["freeze_signature"]):
        raise OffsetPilotError("Existing cap amendment does not match frozen authorization")
    reservation = {
        "event": "CARD_FITS_RESERVED", "batch_id": "optimization_v3_research_next_v1",
        "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
        "category": "pilot", "fits": 4, "fit_ids": list(FIT_IDS),
        "amendment_id": AMENDMENT_ID, "global_cap": 109,
        "effective_category_cap": 68, "source_commit": freeze_data["source_commit"],
        "freeze_signature": freeze_data["freeze_signature"],
    }
    if not existing_reservations:
        if snapshot != {**_expected_pre_reservation_snapshot(), "global_cap": 109,
                        "effective_category_caps": dict(AMENDED_CAPS),
                        "event_rows": snapshot["event_rows"]}:
            raise OffsetPilotError(f"Post-amendment pre-reservation accounting differs: {snapshot}")
        _append_ledger(reservation, path=LEDGER, official=True)
        events = _ledger_events(LEDGER)
    else:
        reservation = existing_reservations[0]
    if any(reservation.get(k) != v for k, v in {
        "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
        "category": "pilot", "fits": 4, "fit_ids": list(FIT_IDS),
        "amendment_id": AMENDMENT_ID, "global_cap": 109, "effective_category_cap": 68,
        "source_commit": freeze_data["source_commit"], "freeze_signature": freeze_data["freeze_signature"],
    }.items()):
        raise OffsetPilotError("Existing fit reservation differs from exact four-ID plan")
    snapshot = ledger_snapshot(events)
    own_starts = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("fit_id") in FIT_IDS]
    own_saved = [e for e in events if e.get("event") == "FIT_SAVED" and e.get("fit_id") in FIT_IDS]
    own_failed = [e for e in events if e.get("event") == "FIT_FAILED" and e.get("fit_id") in FIT_IDS]
    own_terminal = {e.get("fit_id") for e in (*own_saved, *own_failed)}
    own_pending = sum(e.get("fit_id") not in own_terminal for e in own_starts)
    expected = {"global_cap": 109, "effective_category_caps": dict(AMENDED_CAPS),
                "active_reservations_by_category": dict(AMENDED_CAPS), "active_reservations": 109,
                "fit_starts": 105 + len(own_starts), "fit_saved": 102 + len(own_saved),
                "fit_failed": 3 + len(own_failed), "fit_pending": own_pending,
                "event_rows": snapshot["event_rows"]}
    if snapshot != expected:
        raise OffsetPilotError(f"Post-reservation ledger snapshot differs: {snapshot}")
    result = {"reservation": reservation, "ledger_sha256": sha256_file(LEDGER), "snapshot": snapshot}
    _atomic_json(RUN_ROOT / "RESERVATION.json", result)
    return result


def _check_live_budget(*, allow_current: str | None = None,
                       freeze_signature: str | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    events = _ledger_events(LEDGER)
    snapshot = ledger_snapshot(events)
    amendments = [e for e in events if e.get("event") == "BUDGET_CAP_AMENDMENT" and e.get("amendment_id") == AMENDMENT_ID]
    reservations = [e for e in events if e.get("event") == "CARD_FITS_RESERVED" and e.get("reservation_key") == RESERVATION_KEY]
    starts = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("fit_id") in FIT_IDS]
    saved = [e for e in events if e.get("event") == "FIT_SAVED" and e.get("fit_id") in FIT_IDS]
    failed = [e for e in events if e.get("event") == "FIT_FAILED" and e.get("fit_id") in FIT_IDS]
    terminal_ids = {e.get("fit_id") for e in (*saved, *failed)}
    pending = [e for e in starts if e.get("fit_id") not in terminal_ids]
    if len(amendments) != 1 or len(reservations) != 1:
        raise OffsetPilotError("Amendment/reservation missing")
    reservation = _validate_reservation_contract(
        events, source_commit=_git("rev-parse", "HEAD"),
        freeze_signature=freeze_signature or amendments[0].get("freeze_signature"),
    )
    if len(starts) != len({e["fit_id"] for e in starts}) or len(saved) != len({e["fit_id"] for e in saved}):
        raise OffsetPilotError("Duplicate offset fit ledger events")
    expected = {"global_cap": 109, "effective_category_caps": dict(AMENDED_CAPS),
                "active_reservations_by_category": dict(AMENDED_CAPS), "active_reservations": 109,
                "fit_starts": 105 + len(starts), "fit_saved": 102 + len(saved),
                "fit_failed": 3 + len(failed), "fit_pending": len(pending), "event_rows": snapshot["event_rows"]}
    if snapshot != expected:
        raise OffsetPilotError(f"Shared fit ledger accounting changed unexpectedly: {snapshot}")
    if len(pending) > (1 if allow_current else 0):
        raise OffsetPilotError("Unexpected pending fit in shared ledger")
    if allow_current and any(e.get("fit_id") != allow_current for e in pending):
        raise OffsetPilotError("Pending ledger fit does not match the requested resume ID")
    return events, snapshot


def _validate_reservation_contract(events: list[dict[str, Any]], *,
                                   source_commit: str,
                                   freeze_signature: str) -> dict[str, Any]:
    amendments = [e for e in events if e.get("event") == "BUDGET_CAP_AMENDMENT" and e.get("amendment_id") == AMENDMENT_ID]
    reservations = [e for e in events if e.get("event") == "CARD_FITS_RESERVED" and e.get("reservation_key") == RESERVATION_KEY]
    if len(amendments) != 1 or len(reservations) != 1:
        raise OffsetPilotError("Exact amendment and four-ID reservation must each occur once")
    amendment, reservation = amendments[0], reservations[0]
    if (amendment.get("card_id") != CARD_ID or amendment.get("prior_global_cap") != 105
            or amendment.get("global_cap") != 109 or amendment.get("prior_effective_caps") != INITIAL_CAPS
            or amendment.get("effective_caps") != AMENDED_CAPS
            or amendment.get("source_commit") != source_commit
            or amendment.get("freeze_signature") != freeze_signature
            or reservation.get("card_id") != CARD_ID or reservation.get("stage") != STAGE
            or reservation.get("category") != "pilot" or reservation.get("fits") != 4
            or reservation.get("fit_ids") != list(FIT_IDS)
            or reservation.get("reservation_key") != RESERVATION_KEY
            or reservation.get("global_cap") != 109 or reservation.get("effective_category_cap") != 68
            or reservation.get("amendment_id") != AMENDMENT_ID
            or reservation.get("source_commit") != source_commit
            or reservation.get("freeze_signature") != freeze_signature):
        raise OffsetPilotError("Cap amendment/reservation is not the frozen exact four-ID plan")
    return reservation


@contextmanager
def _fit_run_lock(lock_path: Path):
    """Kernel-managed exclusive lock; OS releases it if a process exits abruptly."""
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if not _LOCAL_FIT_LOCK.acquire(blocking=False):
        raise OffsetPilotError("Another POS_BAG_OFFSET operation owns the local fit lock")
    stream = None
    unlock = None
    try:
        stream = lock_path.open("a+b")
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
            os.fsync(stream.fileno())
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise OffsetPilotError("Another process owns the POS_BAG_OFFSET kernel lock") from exc
            unlock = lambda: msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise OffsetPilotError("Another process owns the POS_BAG_OFFSET kernel lock") from exc
            unlock = lambda: fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        owner = psutil.Process()
        payload = {"pid": owner.pid, "process_create_time": owner.create_time(),
                   "started_at_utc": datetime.now(timezone.utc).isoformat()}
        encoded = _canonical_bytes(payload)
        stream.seek(0)
        stream.write(b"\0" + encoded)
        stream.truncate()
        stream.flush()
        os.fsync(stream.fileno())
        yield
    finally:
        try:
            if unlock is not None and stream is not None:
                stream.seek(0)
                unlock()
        finally:
            if stream is not None:
                stream.close()
            _LOCAL_FIT_LOCK.release()


def _fit_artifact_paths(fit_id: str) -> tuple[Path, Path]:
    return RUN_ROOT / "units" / f"{fit_id}.npz", RUN_ROOT / "receipts" / f"{fit_id}.json"


def _fit_rows_for_id(fit_id: str, prediction: Mapping[str, Any]) -> tuple[int, str, np.ndarray, np.ndarray]:
    if fit_id not in FIT_IDS:
        raise OffsetPilotError(f"Fit ID is not authorized: {fit_id}")
    origin = 67 if "E67" in fit_id else 75
    arm = "REAL" if fit_id.endswith("_REAL") else "SHUFFLED"
    position_name = "fit_position" if arm == "REAL" else "fit_shuffled_position"
    score_name = "score_position" if arm == "REAL" else "score_shuffled_position"
    del position_name, score_name
    fit_rows = np.asarray(prediction["fit_rows"], dtype=np.int64)
    return origin, arm, fit_rows, np.asarray(prediction["score_rows"], dtype=np.int64)


def _load_origin_payload(origin: int, freeze_data: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    manifest_path = PILOT_ROOT / f"MATRIX_MANIFEST_E{origin}.json"
    manifest = _read_json(manifest_path)
    p = PILOT_ROOT / f"E{origin}" / f"predictions_E{origin}.npz"
    frozen = freeze_data["origins"][f"E{origin}"]
    if sha256_file(manifest_path) != frozen["matrix_manifest_sha256"]:
        raise OffsetPilotError(f"E{origin} pilot matrix manifest changed after freeze")
    if not p.is_file() or sha256_file(p) != frozen["prediction_bundle_sha256"]:
        raise OffsetPilotError(f"E{origin} pilot prediction bundle changed after freeze")
    for name, item in manifest["matrices"].items():
        path = _matrix_path(item)
        if (sha256_file(path) != frozen["matrix_file_hashes"].get(name)
                or item["logical_sha256"] != frozen["matrix_logical_hashes"].get(name)):
            raise OffsetPilotError(f"E{origin} pilot matrix {name} differs from its frozen hash")
    with np.load(p, allow_pickle=False) as archive:
        prediction = {k: archive[k].copy() for k in archive.files}
    bag_id = f"POSITIONAL_TEXT_PILOT_V1_E{origin}_BAG_LR"
    acc = {x["fit_id"]: x for x in _read_json(ACCOUNTING)["fit_units"]}
    bag_obj, _ = _verify_estimator(bag_id, acc[bag_id], width=manifest["matrices"]["fit_base"]["shape"][1])
    if (sha256_file(bag_obj["artifact_path"]) != frozen["bag_artifact_sha256"]
            or acc[bag_id]["receipt_sha256"] != frozen["bag_receipt_sha256"]
            or acc[bag_id]["fit_metadata"]["coef_sha256"] != frozen["bag_coef_sha256"]
            or acc[bag_id]["fit_metadata"]["intercept_sha256"] != frozen["bag_intercept_sha256"]):
        raise OffsetPilotError(f"E{origin} faithful bag model changed after freeze")
    hgb_id = f"POSITIONAL_TEXT_PILOT_V1_E{origin}_MISSING_HGB"
    hgb_obj, _ = _verify_estimator(hgb_id, acc[hgb_id], width=0)
    if (sha256_file(hgb_obj["artifact_path"]) != frozen["missing_hgb_artifact_sha256"]
            or acc[hgb_id]["receipt_sha256"] != frozen["missing_hgb_receipt_sha256"]):
        raise OffsetPilotError(f"E{origin} missing-stream HGB fallback changed after freeze")
    matrices = {
        name: _load_matrix(manifest, name)
        for name in ("fit_base", "fit_position", "fit_shuffled_position", "score_base", "score_position", "score_shuffled_position")
    }
    return manifest, prediction, bag_obj["estimator"], {"unit": bag_obj["unit"], "hgb": hgb_obj["unit"], "matrices": matrices}


def _fit_labels(train_path: Path, fit_rows: np.ndarray, expected_ids: np.ndarray,
                expected_target_sha: str) -> np.ndarray:
    # Stream CSV records, touching the target field only at frozen fit-prefix
    # row indices. Query-target fields are never selected or converted here.
    rows = np.asarray(fit_rows, dtype=np.int64)
    ids_expected = np.asarray(expected_ids).astype(str)
    if len(rows) != len(ids_expected) or len(np.unique(rows)) != len(rows):
        raise OffsetPilotError("Frozen fit row indices/IDs are invalid")
    position = {int(row): i for i, row in enumerate(rows.tolist())}
    y = np.empty(len(rows), dtype=np.int8)
    seen = np.zeros(len(rows), dtype=bool)
    with Path(train_path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if not header or "record_id" not in header or "artifact_signal" not in header:
            raise OffsetPilotError("Official Train CSV header is invalid")
        id_col, target_col = header.index("record_id"), header.index("artifact_signal")
        for row_index, record in enumerate(reader):
            output_index = position.get(row_index)
            if output_index is None:
                continue
            if max(id_col, target_col) >= len(record):
                raise OffsetPilotError(f"Official Train fit row {row_index} is malformed")
            if record[id_col] != ids_expected[output_index]:
                raise OffsetPilotError("Fit target IDs differ from frozen prefix IDs")
            y[output_index] = int(record[target_col])
            seen[output_index] = True
    if not seen.all():
        raise OffsetPilotError("One or more frozen fit labels were not found in official Train")
    if not np.isin(y, (0, 1)).all() or sha256_array(y) != expected_target_sha:
        raise OffsetPilotError("Fit prefix labels do not match the verified saved pilot population")
    return y


def _resume_started_fit(
    fit_id: str,
    freeze_data: Mapping[str, Any],
    events: list[dict[str, Any]],
    *,
    ledger_path: Path,
    artifact_path: Path,
    receipt_path: Path,
    official: bool,
) -> str:
    """Shared production/synthetic resume path; never invokes the optimizer."""
    starts = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("fit_id") == fit_id]
    terminal = [e for e in events if e.get("event") in {"FIT_SAVED", "FIT_FAILED"} and e.get("fit_id") == fit_id]
    if terminal:
        if len(starts) != 1 or len(terminal) != 1:
            raise OffsetPilotError(f"Fit {fit_id} has inconsistent terminal accounting")
        if terminal[0]["event"] == "FIT_FAILED":
            return "failed_consumed_no_retry"
        if not artifact_path.is_file() or not receipt_path.is_file():
            raise OffsetPilotError(f"Saved fit {fit_id} is missing artifact/receipt")
        rec = _read_json(receipt_path)
        if (rec.get("freeze_signature") != freeze_data["freeze_signature"]
                or rec.get("artifact_sha256") != sha256_file(artifact_path)
                or terminal[0].get("receipt_sha256") != sha256_file(receipt_path)):
            raise OffsetPilotError(f"Saved fit {fit_id} receipt/artifact hash mismatch")
        return "already_saved"
    if not starts:
        return "not_started"
    if len(starts) != 1:
        raise OffsetPilotError(f"Fit {fit_id} has duplicate starts")
    start = starts[0]
    if not artifact_path.is_file() or not receipt_path.is_file():
        _append_ledger({
            "event": "FIT_FAILED", "batch_id": "optimization_v3_research_next_v1",
            "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
            "fit_id": fit_id, "cutoff": start.get("cutoff"), "fit_kind": "PREDICTOR",
            "amendment_id": AMENDMENT_ID,
            "error": "interrupted_before_durable_receipt; consumed fit ID; optimizer retry prohibited",
        }, path=ledger_path, official=official)
        return "failed_interrupted_no_retry"
    rec = _read_json(receipt_path)
    if (rec.get("fit_id") != fit_id or rec.get("freeze_signature") != freeze_data["freeze_signature"]
            or rec.get("artifact_sha256") != sha256_file(artifact_path)):
        raise OffsetPilotError(f"Pending fit {fit_id} receipt is not resumable")
    _append_ledger({
        "event": "FIT_SAVED", "batch_id": "optimization_v3_research_next_v1",
        "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
        "fit_id": fit_id, "cutoff": rec.get("origin", start.get("cutoff")), "fit_kind": "PREDICTOR",
        "receipt_sha256": sha256_file(receipt_path), "amendment_id": AMENDMENT_ID,
        "resumed_from_durable_receipt": True,
    }, path=ledger_path, official=official)
    return "resumed_from_durable_receipt"


def _run_fit(fit_id: str, freeze_data: Mapping[str, Any]) -> dict[str, Any]:
    if fit_id not in FIT_IDS:
        raise OffsetPilotError(f"Fit ID is not one of the four authorized IDs: {fit_id}")
    verified = verify_saved_inputs()
    events, snapshot = _check_live_budget(allow_current=fit_id, freeze_signature=freeze_data["freeze_signature"])
    reservation = next(e for e in events if e.get("event") == "CARD_FITS_RESERVED" and e.get("reservation_key") == RESERVATION_KEY)
    if reservation.get("fit_ids") != list(FIT_IDS) or fit_id not in reservation["fit_ids"]:
        raise OffsetPilotError(f"Fit {fit_id} is outside reserved IDs")
    existing_status = _resume_started_fit(
        fit_id, freeze_data, events, ledger_path=LEDGER, official=True,
        artifact_path=_fit_artifact_paths(fit_id)[0], receipt_path=_fit_artifact_paths(fit_id)[1],
    )
    if existing_status != "not_started":
        return {"fit_id": fit_id, "status": existing_status,
                "ledger_snapshot": ledger_snapshot(_ledger_events(LEDGER))}
    if any(e.get("fit_id") == fit_id for e in events):
        raise OffsetPilotError(f"Fit {fit_id} already has a start; refusing retry")
    prior_starts = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("fit_id") in FIT_IDS]
    if len(prior_starts) >= 4:
        raise OffsetPilotError("All four authorized starts are already consumed")
    origin = 67 if "E67" in fit_id else 75
    arm = "REAL" if fit_id.endswith("_REAL") else "SHUFFLED"
    manifest, prediction, bag, payload = _load_origin_payload(origin, freeze_data)
    matrices = payload["matrices"]
    fit_rows, score_rows = _fit_and_score_rows(origin, prediction)
    if _row_metadata(_read_identity_frame(), fit_rows) != manifest["fit_row_metadata"]:
        raise OffsetPilotError(f"E{origin} fit rows changed after freeze")
    base_fit = sparse.csr_matrix(matrices["fit_base"], dtype=np.float64)
    base_score = sparse.csr_matrix(matrices["score_base"], dtype=np.float64)
    augmented_fit = sparse.csr_matrix(matrices["fit_position" if arm == "REAL" else "fit_shuffled_position"], dtype=np.float64)
    augmented_score = sparse.csr_matrix(matrices["score_position" if arm == "REAL" else "score_shuffled_position"], dtype=np.float64)
    x_fit = augmented_fit[:, base_fit.shape[1]:].tocsr()
    x_score = augmented_score[:, base_score.shape[1]:].tocsr()
    if (x_fit.shape[1] != x_score.shape[1] or len(fit_rows) != x_fit.shape[0]
            or len(score_rows) != x_score.shape[0] or x_fit.dtype != np.float64):
        raise OffsetPilotError("Frozen positional matrix dimensions/dtype do not align")
    fit_ids = np.asarray(prediction["score_record_ids"])  # replace below with fit IDs from official row identity
    del fit_ids
    identity = _read_identity_frame()
    fit_ids = identity.iloc[fit_rows]["record_id"].astype(str).to_numpy()
    account_map = {e["fit_id"]: e for e in _read_json(ACCOUNTING)["fit_units"]}
    bag_acc = account_map[f"POSITIONAL_TEXT_PILOT_V1_E{origin}_BAG_LR"]
    y = _fit_labels(TRAIN, fit_rows, fit_ids, bag_acc["fit_population"]["fit_target_sha256"])
    z0_fit = np.asarray(bag.intercept_[0] + base_fit @ bag.coef_[0], dtype=np.float64).reshape(-1)
    z0_score = np.asarray(bag.intercept_[0] + base_score @ bag.coef_[0], dtype=np.float64).reshape(-1)
    if not np.allclose(expit(z0_score), prediction["prob_bag"], rtol=0, atol=1e-12):
        raise OffsetPilotError("Direct bag logit reconstruction differs before offset fit")

    # Recheck cap, fit count, memory, and source immediately before consuming this fit ID.
    events, snapshot = _check_live_budget(freeze_signature=freeze_data["freeze_signature"])
    if snapshot["fit_pending"] != 0 or len([e for e in events if e.get("event") == "FIT_STARTED" and e.get("fit_id") in FIT_IDS]) >= 4:
        raise OffsetPilotError("Live ledger is not ready for exactly one next fit")
    estimate = 1024**3
    resource = _resource_snapshot(estimate, initial=snapshot["fit_starts"] == 105)
    row_meta = {
        "rows": len(fit_rows), "fit_rows_sha256": sha256_array(fit_rows),
        "fit_ids_sha256": sha256_array(fit_ids), "fit_target_sha256": sha256_array(y),
        "target_read_purpose": "fit-prefix-only", "origin": origin,
        "arm": arm, "matrix_manifest_sha256": sha256_file(PILOT_ROOT / f"MATRIX_MANIFEST_E{origin}.json"),
    }
    _append_ledger({
        "event": "FIT_STARTED", "batch_id": "optimization_v3_research_next_v1",
        "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
        "fit_id": fit_id, "cutoff": origin, "component": f"offset_{arm.lower()}",
        "fit_kind": "PREDICTOR", "fit_consumed_at_start": True,
        "amendment_id": AMENDMENT_ID, "freeze_signature": freeze_data["freeze_signature"],
        "source_commit": freeze_data["source_commit"], "resource_snapshot": resource,
        "fit_population": row_meta,
    }, path=LEDGER, official=True)
    t0 = time.perf_counter()
    try:
        with threadpool_limits(limits=2):
            gamma, optimizer = fit_offset_coefficients(x_fit, z0_fit, y, c=1.0)
        gamma_path, receipt_path = _fit_artifact_paths(fit_id)
        gamma_sha = sha256_array(gamma)
        _atomic_npz(gamma_path, gamma=gamma)
        after = _resource_snapshot(0)
        receipt = {
            "schema_version": 1, "fit_id": fit_id, "card_id": CARD_ID,
            "origin": origin, "arm": arm, "fit_kind": "PREDICTOR",
            "freeze_signature": freeze_data["freeze_signature"],
            "source_commit": freeze_data["source_commit"], "source_hashes": freeze_data["source_hashes"],
            "input_sha256": freeze_data["input_sha256"], "fit_population": row_meta,
            "matrix": {
                "base_width": int(base_fit.shape[1]), "position_width": int(x_fit.shape[1]),
                "fit_base_sha256": sha256_csr(base_fit), "score_base_sha256": sha256_csr(base_score),
                "fit_position_sha256": sha256_csr(x_fit), "score_position_sha256": sha256_csr(x_score),
            },
            "fixed_bag": {
                "artifact_sha256": bag_acc["artifact_sha256"],
                "coef_sha256": bag_acc["fit_metadata"]["coef_sha256"],
                "intercept_sha256": bag_acc["fit_metadata"]["intercept_sha256"],
                "offset_fit_sha256": sha256_array(z0_fit),
            },
            "optimizer": optimizer, "gamma_sha256": gamma_sha,
            "artifact_path": gamma_path.relative_to(ROOT).as_posix(),
            "artifact_sha256": sha256_file(gamma_path),
            "resource_before": resource, "resource_after": after,
            "elapsed_seconds": float(time.perf_counter() - t0),
        }
        _atomic_json(receipt_path, receipt)
        if sha256_file(gamma_path) != receipt["artifact_sha256"]:
            raise OffsetPilotError("Gamma artifact hash changed during persistence")
        _append_ledger({
            "event": "FIT_SAVED", "batch_id": "optimization_v3_research_next_v1",
            "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
            "fit_id": fit_id, "cutoff": origin, "fit_kind": "PREDICTOR",
            "receipt_sha256": sha256_file(receipt_path), "amendment_id": AMENDMENT_ID,
        }, path=LEDGER, official=True)
        _check_live_budget(freeze_signature=freeze_data["freeze_signature"])
        return {"fit_id": fit_id, "status": "saved", "optimizer": optimizer,
                "gamma_sha256": gamma_sha, "receipt_sha256": sha256_file(receipt_path)}
    except Exception as exc:
        now = _ledger_events(LEDGER)
        starts = [e for e in now if e.get("event") == "FIT_STARTED" and e.get("fit_id") == fit_id]
        terminals = [e for e in now if e.get("event") in {"FIT_SAVED", "FIT_FAILED"} and e.get("fit_id") == fit_id]
        if starts and not terminals:
            failure_path = RUN_ROOT / "failures" / f"{fit_id}.json"
            _atomic_json(failure_path, {
                "fit_id": fit_id, "origin": origin, "arm": arm,
                "freeze_signature": freeze_data["freeze_signature"],
                "error_type": type(exc).__name__, "error": str(exc),
                "source_commit": freeze_data["source_commit"],
            })
            _append_ledger({
                "event": "FIT_FAILED", "batch_id": "optimization_v3_research_next_v1",
                "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
                "fit_id": fit_id, "cutoff": origin, "fit_kind": "PREDICTOR",
                "amendment_id": AMENDMENT_ID,
                "error": f"{type(exc).__name__}: {exc}",
            }, path=LEDGER, official=True)
        if isinstance(exc, OffsetFitError):
            return {"fit_id": fit_id, "status": "failed_consumed_no_retry",
                    "failure_receipt": (RUN_ROOT / "failures" / f"{fit_id}.json").relative_to(ROOT).as_posix(),
                    "error": str(exc)}
        raise


def run_all_fits() -> list[dict[str, Any]]:
    freeze_data = load_freeze()
    events = _ledger_events(LEDGER)
    terminals = {e.get("fit_id") for e in events
                 if e.get("event") in {"FIT_SAVED", "FIT_FAILED"}}
    pending = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("fit_id") in FIT_IDS
               and e.get("fit_id") not in terminals]
    if len(pending) > 1:
        raise OffsetPilotError("More than one offset fit is pending; fail closed")
    events, _ = _check_live_budget(allow_current=pending[0]["fit_id"] if pending else None,
                                   freeze_signature=freeze_data["freeze_signature"])
    reservation = next(e for e in events if e.get("event") == "CARD_FITS_RESERVED" and e.get("reservation_key") == RESERVATION_KEY)
    if reservation.get("fit_ids") != list(FIT_IDS):
        raise OffsetPilotError("Live reservation is not the exact frozen four-ID order")
    results = []
    with _fit_run_lock(RUN_ROOT / "FIT_RUN.lock"):
        for fit_id in reservation["fit_ids"]:
            events = _ledger_events(LEDGER)
            existing_failure = any(e.get("event") == "FIT_FAILED" and e.get("fit_id") == fit_id for e in events)
            if existing_failure:
                results.append({"fit_id": fit_id, "status": "failed_consumed_no_retry"})
                continue
            results.append(_run_fit(fit_id, freeze_data))
        events = _ledger_events(LEDGER)
        failure_ids = [e["fit_id"] for e in events if e.get("event") == "FIT_FAILED" and e.get("fit_id") in FIT_IDS]
        terminal_ids = {e.get("fit_id") for e in events if e.get("event") in {"FIT_SAVED", "FIT_FAILED"}}
        if failure_ids and set(FIT_IDS).issubset(terminal_ids):
            _finalize_failed_fit_attempts(freeze_data, events)
    return results


def _finalize_failed_fit_attempts(freeze_data: Mapping[str, Any],
                                  events: list[dict[str, Any]]) -> dict[str, Any]:
    existing = [e for e in events if e.get("event") == "CARD_COMPLETED" and e.get("card_id") == CARD_ID]
    if existing:
        return _read_json(RUN_ROOT / "RESULTS.json")
    _, snapshot = _check_live_budget(freeze_signature=freeze_data["freeze_signature"])
    failed = [e for e in events if e.get("event") == "FIT_FAILED" and e.get("fit_id") in FIT_IDS]
    saved = [e for e in events if e.get("event") == "FIT_SAVED" and e.get("fit_id") in FIT_IDS]
    result = {
        "schema_version": 1, "card_id": CARD_ID,
        "status": "FAIL_FIT_CONVERGENCE_OR_INTERRUPTION",
        "query_labels_opened": False, "predictions_built": False,
        "qualified_candidate": False, "champion_change": False, "upload_authorized": False,
        "source_commit": freeze_data["source_commit"], "freeze_signature": freeze_data["freeze_signature"],
        "input_sha256": freeze_data["input_sha256"], "failed_fit_events": failed,
        "saved_fit_ids": [e["fit_id"] for e in saved], "consumed_fit_ids": list(FIT_IDS),
        "ledger_snapshot": snapshot, "ledger_sha256_before_completion": sha256_file(LEDGER),
        "interpretation": "A planned predictor fit failed or was interrupted. The consumed ID was not retried; all authorized IDs were resolved. Query labels were not opened and no predictive metric gate was evaluated.",
    }
    _atomic_json(RUN_ROOT / "RESULTS.json", result)
    _atomic_json(REPORT_ROOT / "RESULTS.json", result)
    report = "\n".join([
        "# POS_BAG_OFFSET_V1 — fit failure checkpoint", "",
        "**Status:** `FAIL_FIT_CONVERGENCE_OR_INTERRUPTION`. Query labels were not opened; no temporal or subgroup metrics were evaluated.",
        "", "Each of the four authorized IDs was consumed once. Failed or interrupted fits were not retried.",
        "", "| Fit ID | Status | Error / Receipt |", "|---|---|---|",
        *[f"| {fit_id} | {'FAILED' if any(x['fit_id'] == fit_id for x in failed) else 'SAVED'} | {next((x.get('error') for x in failed if x['fit_id'] == fit_id), 'durable receipt saved')} |" for fit_id in FIT_IDS],
        "", "The diagnostic remains unqualified and does not change the champion or authorize upload.", "",
    ])
    _atomic_bytes(REPORT_ROOT / "RESULTS.md", report.encode("utf-8"))
    _append_ledger({
        "event": "CARD_COMPLETED", "batch_id": "optimization_v3_research_next_v1",
        "card_id": CARD_ID, "stage": STAGE, "freeze_signature": freeze_data["freeze_signature"],
        "status": result["status"], "fit_ids": list(FIT_IDS),
        "result_sha256": sha256_file(RUN_ROOT / "RESULTS.json"),
    }, path=LEDGER, official=True)
    return result


def build_predictions() -> dict[str, Any]:
    freeze_data = load_freeze()
    events, snapshot = _check_live_budget(freeze_signature=freeze_data["freeze_signature"])
    starts = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("fit_id") in FIT_IDS]
    saved = [e for e in events if e.get("event") == "FIT_SAVED" and e.get("fit_id") in FIT_IDS]
    if len(starts) != 4 or len(saved) != 4 or snapshot["fit_pending"] != 0:
        raise OffsetPilotError("All four exact offset fits must be durably saved before prediction build")
    output: dict[str, Any] = {}
    for origin in ORIGINS:
        manifest, prediction, bag, payload = _load_origin_payload(origin, freeze_data)
        matrices = payload["matrices"]
        base_score = sparse.csr_matrix(matrices["score_base"], dtype=np.float64)
        z0 = np.asarray(bag.intercept_[0] + base_score @ bag.coef_[0], dtype=np.float64).reshape(-1)
        present = np.asarray(prediction["present_mask"], dtype=bool)
        fallback = np.asarray(payload["hgb"]["probability"], dtype=np.float64)
        arms: dict[str, np.ndarray] = {}
        for arm, matrix_name in (("REAL", "score_position"), ("SHUFFLED", "score_shuffled_position")):
            fit_id = f"{CARD_ID}_E{origin}_{arm}"
            gamma_path, receipt_path = _fit_artifact_paths(fit_id)
            rec = _read_json(receipt_path)
            if rec.get("freeze_signature") != freeze_data["freeze_signature"] or rec.get("artifact_sha256") != sha256_file(gamma_path):
                raise OffsetPilotError(f"E{origin} {arm} gamma receipt/hash mismatch")
            with np.load(gamma_path, allow_pickle=False) as archive:
                gamma = np.asarray(archive["gamma"], dtype=np.float64)
            if sha256_array(gamma) != rec["gamma_sha256"]:
                raise OffsetPilotError(f"E{origin} {arm} gamma content hash mismatch")
            augmented = sparse.csr_matrix(matrices[matrix_name], dtype=np.float64)
            x_pos = augmented[:, base_score.shape[1]:].tocsr()
            raw = offset_probability(x_pos, z0, gamma)
            routed = raw.copy()
            routed[~present] = fallback[~present]
            if not np.array_equal(routed[~present], fallback[~present]):
                raise OffsetPilotError("Missing/blank fallback is not exact")
            arms[arm] = routed
        route_bag = np.asarray(prediction["prob_route_bag"], dtype=np.float64)
        if not np.array_equal(route_bag[~present], fallback[~present]):
            raise OffsetPilotError(f"E{origin} baseline route does not preserve exact HGB fallback")
        dest = RUN_ROOT / f"E{origin}" / f"predictions_E{origin}.npz"
        _atomic_npz(
            dest, origin=np.asarray([origin], dtype=np.int64),
            fit_rows=np.asarray(prediction["fit_rows"], dtype=np.int64),
            score_rows=np.asarray(prediction["score_rows"], dtype=np.int64),
            score_record_ids=np.asarray(prediction["score_record_ids"]).astype(str),
            score_cycles=np.asarray(prediction["score_cycles"], dtype=np.int64),
            present_mask=present.astype(bool),
            prob_bag=route_bag,
            prob_missing_hgb=fallback,
            prob_offset_real=arms["REAL"],
            prob_offset_shuffled=arms["SHUFFLED"],
            prob_joint_position=np.asarray(prediction["prob_route_position"], dtype=np.float64),
            matrix_manifest_sha256=np.asarray([sha256_file(PILOT_ROOT / f"MATRIX_MANIFEST_E{origin}.json")]),
        )
        output[f"E{origin}"] = {"path": dest.relative_to(ROOT).as_posix(), "sha256": sha256_file(dest),
                                 "rows": len(arms["REAL"]), "present_rows": int(present.sum()),
                                 "fallback_rows": int((~present).sum())}
    manifest = {"schema_version": 1, "card_id": CARD_ID, "freeze_signature": freeze_data["freeze_signature"],
                "source_commit": freeze_data["source_commit"], "input_sha256": freeze_data["input_sha256"],
                "predictions": output, "all_four_gamma_receipts_durable": True,
                "query_labels_opened": False}
    _atomic_json(RUN_ROOT / "PREDICTIONS_READY.json", manifest)
    return manifest


def verify_prelabel() -> dict[str, Any]:
    freeze_data = load_freeze()
    ready = _read_json(RUN_ROOT / "PREDICTIONS_READY.json")
    if (ready.get("freeze_signature") != freeze_data["freeze_signature"]
            or ready.get("input_sha256") != freeze_data["input_sha256"]
            or ready.get("query_labels_opened") is not False):
        raise OffsetPilotError("Prelabel prediction manifest lineage mismatch")
    from verify_optimization_v3_positional_bag_offset_v1 import verify_prelabel_outputs
    result = verify_prelabel_outputs(ROOT, RUN_ROOT, freeze_data)
    ready_path = RUN_ROOT / "PREDICTIONS_READY.json"
    ready = _read_json(ready_path)
    if set(ready.get("predictions", {})) != {"E67", "E75"}:
        raise OffsetPilotError("Prediction readiness manifest must contain exactly E67 and E75")
    result["freeze_signature"] = freeze_data["freeze_signature"]
    result["predictions_ready_sha256"] = sha256_file(ready_path)
    result["prediction_output_sha256s"] = {
        key: sha256_file(ROOT / value["path"])
        for key, value in ready["predictions"].items()
    }
    result["verification_signature"] = canonical_sha256(result)
    _atomic_json(RUN_ROOT / "PRELABEL_VERIFICATION.json", result)
    return result


def _assert_prelabel_binding(verified: Mapping[str, Any], *, freeze_signature: str,
                             ready_path: Path, root: Path) -> dict[str, Any]:
    ready = _read_json(ready_path)
    if set(ready.get("predictions", {})) != {"E67", "E75"}:
        raise OffsetPilotError("Prediction readiness manifest must contain exactly E67 and E75")
    current_output_hashes = {
        key: sha256_file(Path(root) / value["path"])
        for key, value in ready["predictions"].items()
    }
    verification_body = dict(verified)
    verification_signature = verification_body.pop("verification_signature", None)
    if (verified.get("status") != "PASS" or ready.get("query_labels_opened") is not False
            or verified.get("freeze_signature") != freeze_signature
            or verified.get("predictions_ready_sha256") != sha256_file(ready_path)
            or verified.get("prediction_output_sha256s") != current_output_hashes
            or verification_signature != canonical_sha256(verification_body)):
        raise OffsetPilotError("Query labels remain closed until the current prelabel PASS is reverified")
    return {"ready": ready, "current_output_hashes": current_output_hashes}


def _metric(y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(probability, dtype=np.float64)
    if len(y) == 0:
        return {"n": 0, "positive": 0, "negative": 0, "macro_f1": None,
                "positive_f1": None, "precision": None, "positive_recall": None,
                "roc_auc": None, "average_precision": None, "log_loss": None,
                "brier": None, "confusion_matrix": [[0, 0], [0, 0]]}
    prediction = (p >= 0.5).astype(np.int8)
    return {
        "n": int(len(y)), "positive": int(y.sum()), "negative": int((1 - y).sum()),
        "macro_f1": float(f1_score(y, prediction, average="macro", labels=[0, 1], zero_division=0)),
        "positive_f1": float(f1_score(y, prediction, pos_label=1, zero_division=0)),
        "precision": float(precision_score(y, prediction, pos_label=1, zero_division=0)),
        "positive_recall": float(recall_score(y, prediction, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else None,
        "average_precision": float(average_precision_score(y, p)) if np.unique(y).size == 2 else None,
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(np.mean((p - y) ** 2)),
        "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1]).tolist(),
    }


def _label(value: Any) -> str:
    return "<missing>" if pd.isna(value) else str(value)


def _view_masks(origin: int, cycles: np.ndarray) -> dict[str, np.ndarray]:
    if origin == 67:
        return {"W4": (cycles >= 68) & (cycles <= 71), "W5": (cycles >= 72) & (cycles <= 75),
                "W6": (cycles >= 76) & (cycles <= 79), "H12_E67": (cycles >= 68) & (cycles <= 79)}
    return {"W6": (cycles >= 76) & (cycles <= 79)}


def score() -> dict[str, Any]:
    existing_completions = [e for e in _ledger_events(LEDGER)
                             if e.get("event") == "CARD_COMPLETED" and e.get("card_id") == CARD_ID]
    if len(existing_completions) > 1:
        raise OffsetPilotError("Duplicate CARD_COMPLETED events; refusing to reopen query labels")
    if existing_completions:
        result_path = RUN_ROOT / "RESULTS.json"
        completion = existing_completions[0]
        if not result_path.is_file() or completion.get("result_sha256") != sha256_file(result_path):
            raise OffsetPilotError("Existing completion is ambiguous or result hash does not match")
        completed_result = _read_json(result_path)
        if (completion.get("freeze_signature") != completed_result.get("freeze_signature")
                or completion.get("status") != completed_result.get("status")):
            raise OffsetPilotError("Existing CARD_COMPLETED status/freeze differs from result")
        return completed_result
    freeze_data = load_freeze()
    verified = _read_json(RUN_ROOT / "PRELABEL_VERIFICATION.json")
    ready_path = RUN_ROOT / "PREDICTIONS_READY.json"
    binding = _assert_prelabel_binding(verified, freeze_signature=freeze_data["freeze_signature"],
                                       ready_path=ready_path, root=ROOT)
    ready = binding["ready"]
    current_output_hashes = binding["current_output_hashes"]
    events, snapshot = _check_live_budget(freeze_signature=freeze_data["freeze_signature"])
    if (snapshot["fit_starts"] != 109 or snapshot["fit_saved"] != 106
            or snapshot["fit_failed"] != 3 or snapshot["fit_pending"] != 0):
        raise OffsetPilotError("All four amendment fits must be saved before query labels are opened")
    # This is the first query-label read in this workflow.
    train = pd.read_csv(TRAIN, usecols=["record_id", "cycle_code", "artifact_signal", "text_load", "entry_friction"],
                        dtype={"record_id": "string"})
    origin_results: dict[str, Any] = {}
    gates: list[dict[str, Any]] = []
    for origin in ORIGINS:
        suffix = f"E{origin}"
        prediction_path = RUN_ROOT / f"E{origin}" / f"predictions_E{origin}.npz"
        with np.load(prediction_path, allow_pickle=False) as archive:
            pred = {key: archive[key].copy() for key in archive.files}
        rows = np.asarray(pred["score_rows"], dtype=np.int64)
        sub = train.iloc[rows]
        if not np.array_equal(sub["record_id"].astype(str).to_numpy(), pred["score_record_ids"].astype(str)):
            raise OffsetPilotError(f"E{origin} score IDs changed before metric evaluation")
        y = pd.to_numeric(sub["artifact_signal"], errors="raise").to_numpy(dtype=np.int8)
        cycles = pd.to_numeric(sub["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
        p_arms = {"bag": pred["prob_bag"], "offset_real": pred["prob_offset_real"],
                  "offset_shuffled": pred["prob_offset_shuffled"], "joint_position_reference": pred["prob_joint_position"]}
        views = _view_masks(origin, cycles)
        view_metrics: list[dict[str, Any]] = []
        subgroup_metrics: list[dict[str, Any]] = []
        per_cycle: list[dict[str, Any]] = []
        text_load = pd.to_numeric(sub["text_load"], errors="coerce").to_numpy(dtype=np.float64)
        q75 = float(_read_json(PILOT_ROOT / f"MATRIX_MANIFEST_E{origin}.json")["text_load_q75_fit_only"])
        q4 = np.isfinite(text_load) & (text_load >= q75)
        q4_missing = ~np.isfinite(text_load)
        present = np.asarray(pred["present_mask"], dtype=bool)
        ef = sub["entry_friction"].to_numpy(dtype=object)
        subgroup_masks: list[tuple[str, str, np.ndarray]] = [
            ("stream_availability", "present", present),
            ("stream_availability", "missing_or_blank", ~present),
            ("fit_defined_text_load_q4", "Q4", q4),
            ("fit_defined_text_load_q4", "below_Q4", np.isfinite(text_load) & ~q4),
            ("fit_defined_text_load_q4", "missing_text_load", q4_missing),
        ]
        for val in sorted({_label(x) for x in ef}):
            subgroup_masks.append(("entry_friction", val, np.asarray([_label(x) == val for x in ef], dtype=bool)))
        for view, vm in views.items():
            for arm, prob in p_arms.items():
                view_metrics.append({"view": view, "arm": arm, **_metric(y[vm], np.asarray(prob)[vm])})
            for group_name, group_value, gm in subgroup_masks:
                for arm, prob in p_arms.items():
                    subgroup_metrics.append({"view": view, "group": group_name, "group_value": group_value,
                                             "arm": arm, **_metric(y[vm & gm], np.asarray(prob)[vm & gm])})
        for cycle in sorted(np.unique(cycles).tolist()):
            cm = cycles == cycle
            scores = {arm: _metric(y[cm], np.asarray(prob)[cm]) for arm, prob in p_arms.items()}
            per_cycle.append({"cycle": int(cycle), "n": int(cm.sum()), "metrics": scores,
                              "real_minus_bag_macro_f1": scores["offset_real"]["macro_f1"] - scores["bag"]["macro_f1"],
                              "real_minus_shuffled_macro_f1": scores["offset_real"]["macro_f1"] - scores["offset_shuffled"]["macro_f1"]})
        by_view = {(r["view"], r["arm"]): r for r in view_metrics}
        by_sub = {(r["view"], r["arm"]): r for r in subgroup_metrics
                  if r["group"] == "fit_defined_text_load_q4" and r["group_value"] == "Q4"}
        if origin == 67:
            for control in ("bag", "offset_shuffled"):
                delta = by_view[("H12_E67", "offset_real")]["macro_f1"] - by_view[("H12_E67", control)]["macro_f1"]
                gates.append({"origin": origin, "gate": f"H12_E67_gain_vs_{control}_min_0.003", "value": delta,
                              "threshold": 0.003, "pass": delta >= 0.003})
            for view in ("W4", "W5", "W6"):
                for control in ("bag", "offset_shuffled"):
                    delta = by_view[(view, "offset_real")]["macro_f1"] - by_view[(view, control)]["macro_f1"]
                    gates.append({"origin": origin, "gate": f"{view}_harm_vs_{control}_max_0.005", "value": delta,
                                  "threshold": -0.005, "pass": delta >= -0.005})
        else:
            for control in ("bag", "offset_shuffled"):
                delta = by_view[("W6", "offset_real")]["macro_f1"] - by_view[("W6", control)]["macro_f1"]
                gates.append({"origin": origin, "gate": f"W6_gain_vs_{control}_min_0", "value": delta,
                              "threshold": 0.0, "pass": delta >= 0.0})
        for view in views:
            candidate = by_sub[(view, "offset_real")]
            support = (candidate["n"] >= 200 and candidate["positive"] >= 30 and candidate["negative"] >= 30)
            gates.append({"origin": origin, "gate": f"{view}_Q4_support", "value": {
                "n": candidate["n"], "positive": candidate["positive"], "negative": candidate["negative"]},
                "threshold": {"n": 200, "each_class": 30}, "pass": support})
            if support:
                for control in ("bag", "offset_shuffled"):
                    control_metric = by_sub[(view, control)]
                    df1 = candidate["macro_f1"] - control_metric["macro_f1"]
                    drec = candidate["positive_recall"] - control_metric["positive_recall"]
                    gates.append({"origin": origin, "gate": f"{view}_Q4_macro_f1_harm_vs_{control}_max_0.01",
                                  "value": df1, "threshold": -0.01, "pass": df1 >= -0.01})
                    gates.append({"origin": origin, "gate": f"{view}_Q4_positive_recall_harm_vs_{control}_max_0.02",
                                  "value": drec, "threshold": -0.02, "pass": drec >= -0.02})
        origin_results[suffix] = {
            "score_rows": len(rows), "score_present_rows": int(present.sum()),
            "views": view_metrics, "subgroups": subgroup_metrics, "per_cycle": per_cycle,
            "fit_defined_text_load_q75": q75,
            "per_cycle_positive_gain_counts": {
                "real_vs_bag": sum(x["real_minus_bag_macro_f1"] > 0 for x in per_cycle),
                "real_vs_shuffled": sum(x["real_minus_shuffled_macro_f1"] > 0 for x in per_cycle),
                "cycles": len(per_cycle),
            },
        }
    fit_receipts = {}
    optimizer_pass = True
    for fit_id in FIT_IDS:
        gamma_path, receipt_path = _fit_artifact_paths(fit_id)
        rec = _read_json(receipt_path)
        fit_receipts[fit_id] = {"receipt_sha256": sha256_file(receipt_path),
                                "artifact_sha256": sha256_file(gamma_path),
                                "optimizer": rec["optimizer"]}
        optimizer_pass &= bool(rec["optimizer"]["success"])
    all_gates_pass = optimizer_pass and all(x["pass"] for x in gates)
    result = {
        "schema_version": 1, "card_id": CARD_ID,
        "status": "PASS_DIAGNOSTIC_REVIEW_ONLY" if all_gates_pass else "FAIL_DIAGNOSTIC_ONLY",
        "qualified_candidate": False, "champion_change": False, "upload_authorized": False,
        "adaptive_selection": True, "overlapping_origins": True,
        "source_commit": freeze_data["source_commit"], "freeze_signature": freeze_data["freeze_signature"],
        "prelabel_verification_sha256": sha256_file(RUN_ROOT / "PRELABEL_VERIFICATION.json"),
        "predictions_ready_sha256": sha256_file(ready_path),
        "prediction_output_sha256s": current_output_hashes,
        "input_sha256": freeze_data["input_sha256"], "query_labels_opened_after_prelabel_verification": True,
        "ledger_snapshot": snapshot, "ledger_sha256": sha256_file(LEDGER),
        "fit_receipts": fit_receipts, "optimizer_all_success": optimizer_pass,
        "origins": origin_results, "gate_checks": gates, "all_gates_pass": all_gates_pass,
        "interpretation": "Sequential additive fit on in-sample saved bag logits; not cross-fitted residual training. E67/E75 score windows overlap. Results are diagnostic-only and do not qualify a model or authorize upload.",
    }
    _atomic_json(RUN_ROOT / "RESULTS.json", result)
    report = _make_report(result)
    _atomic_bytes(REPORT_ROOT / "RESULTS.md", report.encode("utf-8"))
    _atomic_json(REPORT_ROOT / "RESULTS.json", result)
    _append_ledger({
        "event": "CARD_COMPLETED", "batch_id": "optimization_v3_research_next_v1",
        "card_id": CARD_ID, "stage": STAGE, "freeze_signature": freeze_data["freeze_signature"],
        "status": result["status"], "fit_ids": list(FIT_IDS),
        "result_sha256": sha256_file(RUN_ROOT / "RESULTS.json"),
    }, path=LEDGER, official=True)
    return result


def _make_report(result: Mapping[str, Any]) -> str:
    lines = [
        "# POS_BAG_OFFSET_V1 — hasil diagnostic",
        "",
        f"**Status:** `{result['status']}`. Kandidat tetap diagnostic-only; tidak ada perubahan champion atau izin upload.",
        "",
        "## Pertanyaan dan batas interpretasi",
        "",
        "Eksperimen ini menilai apakah kanal posisi memberi tambahan prediktif ketika koefisien dan intercept C2_B bag dibekukan sebagai offset logit. Gamma saja dipelajari memakai objective logistic SUM dengan penalti L2 dan C=1.",
        "Offset bag berasal dari prediksi in-sample pada prefix. Karena itu ini sequential additive fitting, bukan cross-fitted residual training. Amendment ini dipilih adaptif setelah melihat diagnostic sebelumnya; hasil temporal juga tumpang tindih.",
        "",
        "## Metrik utama",
        "",
        "Macro F1 memakai threshold 0,50. `bag` adalah C2_B dengan fallback HGB tersimpan; `offset_real` dan `offset_shuffled` memakai gamma masing-masing; `joint_position_reference` hanya referensi deskriptif dari pilot lama.",
        "",
        "| Origin | View | Bag | Offset real | Offset shuffled | Joint-position ref. | Real − bag | Real − shuffled |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for origin, block in result["origins"].items():
        lookup = {(x["view"], x["arm"]): x for x in block["views"]}
        for view in sorted({x["view"] for x in block["views"]}, key=lambda x: ["W4", "W5", "W6", "H12_E67"].index(x)):
            bag = lookup[(view, "bag")]["macro_f1"]
            real = lookup[(view, "offset_real")]["macro_f1"]
            shuffled = lookup[(view, "offset_shuffled")]["macro_f1"]
            joint = lookup[(view, "joint_position_reference")]["macro_f1"]
            lines.append(f"| {origin} | {view} | {bag:.6f} | {real:.6f} | {shuffled:.6f} | {joint:.6f} | {real-bag:+.6f} | {real-shuffled:+.6f} |")
    lines.extend(["", "## Gate checks", "", "| Gate | Nilai | Ambang | Hasil |", "|---|---:|---:|---|"])
    for gate in result["gate_checks"]:
        val = gate["value"]
        val_text = json.dumps(val, ensure_ascii=False, sort_keys=True) if isinstance(val, dict) else f"{val:.6f}"
        lines.append(f"| E{gate['origin']} {gate['gate']} | {val_text} | {gate['threshold']} | {'PASS' if gate['pass'] else 'FAIL'} |")
    lines.extend(["", "## Siklus dan subgroup", "", "Detail metrik tiap siklus, stream availability, entry_friction, dan fit-defined text_load Q4 tersimpan di `RESULTS.json`. Angka negatif dipertahankan; gate Q4 mensyaratkan n≥200 dan minimal 30 observasi pada tiap kelas.", ""])
    for origin, block in result["origins"].items():
        signs = block["per_cycle_positive_gain_counts"]
        lines.append(f"- E{origin}: tanda positif per siklus real−bag {signs['real_vs_bag']}/{signs['cycles']}; real−shuffled {signs['real_vs_shuffled']}/{signs['cycles']}.")
    lines.extend(["", "## Kesimpulan", "", f"Status gerbang gabungan: **{'lolos untuk review diagnostic' if result['all_gates_pass'] else 'gagal'}**. Hasil ini tidak membuktikan mekanisme umum, tidak memenuhi qualification full-six, dan tidak mengganti champion. Hasil perlu dibaca dengan seleksi adaptif dan overlap temporal yang dinyatakan di atas.", ""])
    return "\n".join(lines)


def run_synthetic_lifecycle(temp_root: Path) -> dict[str, Any]:
    """Temporary-ledger end-to-end optimizer, receipt, resume, verify and metric path."""
    live_before = sha256_file(LEDGER)
    try:
        rng = np.random.default_rng(20261103)
        x = sparse.csr_matrix(rng.normal(size=(80, 9)))
        base = sparse.csr_matrix(rng.normal(size=(80, 5)))
        base_coef = rng.normal(size=5).astype(np.float64)
        base_intercept = np.asarray([0.25], dtype=np.float64)
        y = np.tile(np.asarray([0, 1], dtype=np.int8), 40)
        z0 = np.asarray(base_intercept[0] + base @ base_coef, dtype=np.float64).reshape(-1)
        temp_root.mkdir(parents=True, exist_ok=True)
        ledger = temp_root / "synthetic_ledger.jsonl"
        base_categories = {"pilot": 64, "continuation": 38, "control_replay": 2, "technical_retry": 1}
        _append_ledger({"event": "BATCH_ALLOWANCE_AUTHORIZED", "cap": 105,
                        "category_caps": base_categories}, path=ledger)
        for category, fits in base_categories.items():
            _append_ledger({"event": "CARD_FITS_RESERVED", "reservation_key": f"SYNTH_BASE_{category}",
                            "category": category, "fits": fits}, path=ledger)
        for i in range(105):
            _append_ledger({"event": "FIT_STARTED", "fit_id": f"SYNTH_BASE_{i:03d}"}, path=ledger)
            _append_ledger({"event": "FIT_SAVED" if i < 102 else "FIT_FAILED",
                            "fit_id": f"SYNTH_BASE_{i:03d}"}, path=ledger)
        synthetic_freeze = {"freeze_signature": "synthetic-freeze-signature"}
        _append_ledger({"event": "BUDGET_CAP_AMENDMENT", "amendment_id": AMENDMENT_ID,
                        "card_id": CARD_ID, "prior_global_cap": 105,
                        "global_cap": 109, "effective_caps": dict(AMENDED_CAPS),
                        "prior_effective_caps": dict(INITIAL_CAPS),
                        "source_commit": "synthetic-source", "freeze_signature": synthetic_freeze["freeze_signature"]}, path=ledger)
        reservation = {"event": "CARD_FITS_RESERVED", "reservation_key": RESERVATION_KEY,
                       "card_id": CARD_ID, "stage": STAGE, "category": "pilot", "fits": 4,
                       "fit_ids": list(FIT_IDS), "amendment_id": AMENDMENT_ID,
                       "global_cap": 109, "effective_category_cap": 68,
                       "source_commit": "synthetic-source", "freeze_signature": synthetic_freeze["freeze_signature"]}
        _append_ledger(reservation, path=ledger)
        _validate_reservation_contract(_ledger_events(ledger), source_commit="synthetic-source",
                                       freeze_signature=synthetic_freeze["freeze_signature"])
        if ledger_snapshot(_ledger_events(ledger))["active_reservations"] != 109:
            raise OffsetPilotError("Synthetic exact four-slot reservation accounting failed")
        synthetic_fit_id = FIT_IDS[0]
        _append_ledger({"event": "FIT_STARTED", "batch_id": "optimization_v3_research_next_v1",
                        "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
                        "fit_id": synthetic_fit_id, "cutoff": 67, "fit_kind": "PREDICTOR",
                        "amendment_id": AMENDMENT_ID,
                        "freeze_signature": synthetic_freeze["freeze_signature"]}, path=ledger)
        # Fit is completed in memory, then a durable receipt is written before
        # the shared production resume helper repairs the missing FIT_SAVED.
        gamma, meta = fit_offset_coefficients(x, z0, y, c=1.0)
        art = temp_root / "SYNTHETIC_A.npz"
        rec_path = temp_root / "SYNTHETIC_A.json"
        fit_ids = np.asarray([f"fit-{i:03d}" for i in range(len(y))])
        score_ids = np.asarray([f"score-{i:03d}" for i in range(len(y))])
        present = np.ones(len(y), dtype=bool)
        present[::7] = False
        fallback = np.full(len(y), 0.37, dtype=np.float64)
        raw_probability = offset_probability(x, z0, gamma)
        routed_probability = raw_probability.copy()
        routed_probability[~present] = fallback[~present]
        _atomic_npz(art, gamma=gamma, x_data=x.data, x_indices=x.indices,
                    x_indptr=x.indptr, x_shape=np.asarray(x.shape), y=y, z0=z0,
                    base_data=base.data, base_indices=base.indices, base_indptr=base.indptr,
                    base_shape=np.asarray(base.shape),
                    probability=routed_probability, fit_ids=fit_ids, score_ids=score_ids,
                    classes=np.asarray([0, 1], dtype=np.int8), present=present,
                    fallback=fallback, base_coef=base_coef, base_intercept=base_intercept)
        receipt = {"fit_id": synthetic_fit_id, "origin": 67,
                   "freeze_signature": synthetic_freeze["freeze_signature"],
                   "artifact_sha256": sha256_file(art),
                   "optimizer": meta, "gamma_sha256": sha256_array(gamma),
                   "fit_ids_sha256": sha256_array(fit_ids), "score_ids_sha256": sha256_array(score_ids),
                   "base_coef_sha256": sha256_array(base_coef),
                   "base_intercept_sha256": sha256_array(base_intercept),
                   "base_matrix_sha256": sha256_csr(base)}
        _atomic_json(rec_path, receipt)
        # Simulated interruption after durable bytes but before FIT_SAVED.
        resumed = _resume_started_fit(
            synthetic_fit_id, synthetic_freeze, _ledger_events(ledger),
            ledger_path=ledger, artifact_path=art, receipt_path=rec_path, official=False,
        )
        if resumed != "resumed_from_durable_receipt":
            raise OffsetPilotError(f"Production resume helper did not repair durable receipt: {resumed}")
        from verify_optimization_v3_positional_bag_offset_v1 import verify_synthetic_bundle
        independent = verify_synthetic_bundle(art, rec_path)
        pred = routed_probability
        metric = _metric(y, pred)
        _atomic_json(temp_root / "METRICS.json", metric)
        _atomic_json(temp_root / "REPORT.json", {"status": "PASS", "metric": metric, "independent": independent})
        _append_ledger({"event": "CARD_COMPLETED", "card_id": f"{CARD_ID}_SYNTHETIC"}, path=ledger)
        events = _ledger_events(ledger)
        synthetic_events = [e for e in events if e.get("fit_id") == synthetic_fit_id]
        if [e["event"] for e in synthetic_events] != ["FIT_STARTED", "FIT_SAVED"]:
            raise OffsetPilotError("Synthetic production-resume ledger sequence mismatch")
        final_budget = ledger_snapshot(events)
        if (final_budget["global_cap"] != 109 or final_budget["active_reservations"] != 109
                or final_budget["fit_starts"] != 106 or final_budget["fit_saved"] != 103
                or final_budget["fit_failed"] != 3 or final_budget["fit_pending"] != 0):
            raise OffsetPilotError(f"Synthetic production-ledger accounting mismatch: {final_budget}")
        return {"status": "PASS", "optimizer": meta, "metric": metric,
                "independent": independent, "temporary_ledger_events": len(events),
                "ledger_snapshot": final_budget}
    finally:
        if sha256_file(LEDGER) != live_before:
            raise OffsetPilotError("Live shared ledger changed during synthetic-only lifecycle")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", required=True,
                        choices=("preflight", "freeze", "reserve", "run-all", "predictions",
                                 "verify-prelabel", "score", "synthetic"))
    args = parser.parse_args(argv)
    actions = {
        "preflight": preflight, "freeze": freeze, "reserve": reserve,
        "run-all": run_all_fits, "predictions": build_predictions,
        "verify-prelabel": verify_prelabel, "score": score,
        "synthetic": lambda: run_synthetic_lifecycle(RUN_ROOT / "synthetic" / uuid.uuid4().hex),
    }
    if args.command in {"reserve", "predictions", "verify-prelabel", "score"}:
        with _fit_run_lock(RUN_ROOT / "FIT_RUN.lock"):
            result = actions[args.command]()
    else:
        result = actions[args.command]()
    print(json.dumps(_jsonable(result), sort_keys=True, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
