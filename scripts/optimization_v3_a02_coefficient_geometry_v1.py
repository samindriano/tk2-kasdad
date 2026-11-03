"""Frozen, resumable CGEOM_V1 pilot for coefficient-informed A02 geometry.

The source lineage uses cached A02 and same-origin C2_B artifacts only.  The
target-free phases never select or read query labels; labels are opened only by
``score`` after the independent prelabel verifier has written PASS.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
import psutil
from scipy import sparse
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from src.models import classification_a02_neighbors as a02
from src.models.a02_coefficient_geometry_v1 import (
    CoefficientGeometryPredictor,
    PERMUTATION_SEED,
    coefficient_weights,
)
import optimization_v3_positional_text_pilot_v1 as positional


PILOT_ID = "CGEOM_V1"
CARD_ID = "A02_COEFFICIENT_GEOMETRY_V1"
STAGE = "FOUR_INDEXED_MEMORY_PREDICTORS"
AMENDMENT_ID = "CGEOM_V1_CAP121_2026_11_03"
RESERVATION_KEY = f"{CARD_ID}:{STAGE}"
ORIGINS = (67, 75)
ARMS = ("aligned", "permuted")
FIT_IDS = tuple(f"CGEOM_V1_E{origin}_{arm.upper()}" for origin in ORIGINS for arm in ARMS)
TRAIN = ROOT / "data/raw/classification/Train.csv"
TRAIN_SHA256 = "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c"
RUN_ROOT = ROOT / "experiments/runs/optimization_v3_research_next_v1/CGEOM_V1"
REPORT_ROOT = ROOT / "reports/optimization_v3/a02_coefficient_geometry_v1"
RECIPE_PATH = REPORT_ROOT / "RECIPE.json"
LEDGER = ROOT / "experiments/runs/optimization_v3_research_next_v1/budget.jsonl"
PREFLIGHT_PATH = RUN_ROOT / "PREFLIGHT.json"
SYNTHETIC_QA_PATH = RUN_ROOT / "SYNTHETIC_QA.json"
FREEZE_PATH = RUN_ROOT / "FREEZE.json"
PRELABEL_PATH = RUN_ROOT / "PRELABEL_VERIFICATION.json"
RESULT_PATH = RUN_ROOT / "RESULTS.json"
SCORED_VERIFICATION_PATH = RUN_ROOT / "SCORED_VERIFICATION.json"
COMPLETION_PATH = RUN_ROOT / "COMPLETION_RECEIPT.json"
AMENDED_CAP = 121
BASE_CAP = 117
BASE_CATEGORY_CAPS = {"pilot": 76, "continuation": 38, "control_replay": 2, "technical_retry": 1}
AMENDED_CATEGORY_CAPS = {"pilot": 80, "continuation": 38, "control_replay": 2, "technical_retry": 1}
BASE_ACTIVE = {"pilot": 73, "continuation": 38, "control_replay": 2, "technical_retry": 1}
BASE_FIT_COUNTS = {"fit_starts": 114, "fit_saved": 110, "fit_failed": 4, "fit_pending": 0}
BASE_LEDGER_ROWS = 292
BASE_LEDGER_SHA256 = "d23c8d365fdf3daae99600ab74a4c02818b0fad27f030fd051a5bdf5c79e8c9f"
RSS_LIMIT = 6 * 1024**3
INITIAL_FREE_MIN = 4 * 1024**3
FREE_RESERVE = 2 * 1024**3
THREAD_LIMIT = 2
EPSILON = 1e-15
GATE_TOLERANCE = 1e-12
MIN_GROUP_N = 200
MIN_GROUP_PER_CLASS = 30

SOURCE_PATHS = (
    "scripts/optimization_v3_a02_coefficient_geometry_v1.py",
    "scripts/verify_optimization_v3_a02_coefficient_geometry_v1.py",
    "src/models/a02_coefficient_geometry_v1.py",
    "tests/test_optimization_v3_a02_coefficient_geometry_v1.py",
    "reports/optimization_v3/a02_coefficient_geometry_v1/RECIPE.json",
    "src/models/classification_a02_neighbors.py",
    "scripts/optimization_v3_a02_neighbors.py",
    "scripts/optimization_v3_a02_neighbors_verify.py",
    "src/models/optimization_v3_positional_text_pilot_v1.py",
    "scripts/optimization_v3_positional_text_pilot_v1.py",
    "src/models/classification_models.py",
    "src/models/round1_models.py",
)

A02_RUN = ROOT / "experiments/runs/optimization_v3_a02_neighbors_v1"
C2B_RUN = ROOT / "experiments/runs/optimization_v3_research_next_v1/POSITIONAL_TEXT_PILOT_V1"


class CGeomError(RuntimeError):
    pass


class SyntheticInterruption(RuntimeError):
    """Test-only interruption after a complete synthetic bundle is durable."""


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return _sha_bytes(canonical_bytes(value))


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False,
                      default=lambda x: x.item() if isinstance(x, np.generic) else str(x)).encode("utf-8") + b"\n"


def write_create_once(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise CGeomError(f"Create-once artifact differs on resume: {path}")
        return
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json_bytes(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise CGeomError(f"Refusing to overwrite immutable JSON artifact: {path}")
        return
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8").strip()


def require_clean_tree() -> None:
    if _git("status", "--porcelain"):
        raise CGeomError("A source/preflight/frozen stage requires a clean committed working tree")


def source_hashes() -> dict[str, str]:
    missing = [rel for rel in SOURCE_PATHS if not (ROOT / rel).is_file()]
    if missing:
        raise CGeomError(f"CGEOM source file(s) missing: {missing}")
    return {rel: sha_file(ROOT / rel) for rel in SOURCE_PATHS}


def resource_snapshot(*, require_initial_free: bool) -> dict[str, int]:
    vm = psutil.virtual_memory()
    rss = psutil.Process().memory_info().rss
    projected = rss
    if rss > RSS_LIMIT or vm.available < FREE_RESERVE:
        raise CGeomError(f"Resource limit blocked CGEOM: rss={rss}, available={vm.available}")
    if require_initial_free and vm.available < INITIAL_FREE_MIN:
        raise CGeomError(f"Initial available RAM below 4 GiB: {vm.available}")
    return {
        "rss_bytes": int(rss), "available_ram_bytes": int(vm.available),
        "projected_rss_bytes": int(projected), "rss_limit_bytes": RSS_LIMIT,
        "free_reserve_bytes": FREE_RESERVE, "initial_free_min_bytes": INITIAL_FREE_MIN,
        "threads": THREAD_LIMIT, "jobs": 1,
    }


class PeakRSS:
    def __init__(self, interval: float = 0.20):
        self.interval = interval
        self.peak = psutil.Process().memory_info().rss
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "PeakRSS":
        def poll() -> None:
            process = psutil.Process()
            while not self._stop.wait(self.interval):
                try:
                    self.peak = max(self.peak, process.memory_info().rss)
                except psutil.Error:
                    return
        self._thread = threading.Thread(target=poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        try:
            self.peak = max(self.peak, psutil.Process().memory_info().rss)
        except psutil.Error:
            pass


def _ledger_events(path: Path = LEDGER) -> list[dict[str, Any]]:
    return positional.read_ledger(path=path)


def _budget_snapshot(events: list[dict[str, Any]]) -> dict[str, Any]:
    return positional.ledger_snapshot(events)


def _own_events(events: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    fit_names = {"FIT_STARTED", "FIT_SAVED", "FIT_FAILED"}
    result = {name: [] for name in fit_names}
    for event in events:
        if event.get("reservation_key") == RESERVATION_KEY:
            if event.get("event") in fit_names:
                result[event["event"]].append(event)
    return result


def _with_event_rows(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in snapshot.items() if key != "event_rows"}


def _assert_budget(events: list[dict[str, Any]], *, phase: str) -> dict[str, Any]:
    snapshot = _budget_snapshot(events)
    amendments = [e for e in events if e.get("event") == "BUDGET_CAP_AMENDMENT" and e.get("amendment_id") == AMENDMENT_ID]
    reservations = [e for e in events if e.get("event") == "CARD_FITS_RESERVED" and e.get("reservation_key") == RESERVATION_KEY]
    releases = [e for e in events if e.get("event") == "RESERVATION_UNUSED_RELEASED" and e.get("reservation_key") == RESERVATION_KEY]
    if len(amendments) > 1 or len(reservations) > 1 or len(releases) > 1:
        raise CGeomError("Duplicate CGEOM amendment, reservation, or release event")
    own = _own_events(events)
    ids = [e.get("fit_id") for e in own["FIT_STARTED"]]
    saved = [e.get("fit_id") for e in own["FIT_SAVED"]]
    failed = [e.get("fit_id") for e in own["FIT_FAILED"]]
    if len(ids) != len(set(ids)) or len(saved) != len(set(saved)) or len(failed) != len(set(failed)):
        raise CGeomError("Duplicate CGEOM fit event for a fit id")
    if set(ids) - set(FIT_IDS) or set(saved) - set(FIT_IDS) or set(failed) - set(FIT_IDS):
        raise CGeomError("Unexpected fit ID appears under CGEOM reservation key")
    if set(saved) & set(failed) or (set(saved) | set(failed)) - set(ids):
        raise CGeomError("CGEOM fit start/terminal event accounting mismatch")
    pending = len(set(ids) - set(saved) - set(failed))
    released_count = int(releases[-1].get("unused_fits", 0)) if releases else 0
    expected_active = 114 + 4 - released_count
    expected = {
        "global_cap": AMENDED_CAP,
        "effective_category_caps": AMENDED_CATEGORY_CAPS,
        "active_reservations_by_category": {
            "pilot": 73 + 4 - released_count, "continuation": 38,
            "control_replay": 2, "technical_retry": 1,
        },
        "active_reservations": expected_active,
        "fit_starts": 114 + len(ids), "fit_saved": 110 + len(saved),
        "fit_failed": 4 + len(failed), "fit_pending": pending,
    }
    if not amendments or not reservations or _with_event_rows(snapshot) != expected:
        raise CGeomError(f"CGEOM shared budget mismatch at {phase}: {snapshot}, expected={expected}")
    amendment = amendments[0]
    reservation = reservations[0]
    if (amendment.get("prior_global_cap") != BASE_CAP or amendment.get("global_cap") != AMENDED_CAP
            or amendment.get("prior_effective_caps") != BASE_CATEGORY_CAPS
            or amendment.get("effective_caps") != AMENDED_CATEGORY_CAPS
            or amendment.get("card_id") != CARD_ID or amendment.get("freeze_signature") != load_freeze_signature_if_any()):
        raise CGeomError("CGEOM cap amendment does not match the frozen authorization")
    if (reservation.get("fit_ids") != list(FIT_IDS) or reservation.get("fits") != 4
            or reservation.get("global_cap") != AMENDED_CAP or reservation.get("category") != "pilot"
            or reservation.get("freeze_signature") != amendment.get("freeze_signature")):
        raise CGeomError("CGEOM reservation differs from the four authorized fit IDs")
    if phase == "before_fit" and (pending != 0 or len(ids) >= 4):
        raise CGeomError("CGEOM cannot start a fit with pending or exhausted budget")
    return snapshot


def load_freeze_signature_if_any() -> str | None:
    if not FREEZE_PATH.is_file():
        return None
    value = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    return value.get("freeze_signature")


def append_budget_event(event: Mapping[str, Any], *, path: Path, synthetic: bool = False) -> dict[str, Any]:
    resolved = path.resolve()
    official = LEDGER.resolve()
    if synthetic and resolved == official:
        raise CGeomError("Synthetic lifecycle is prohibited from writing to the production ledger")
    if not synthetic:
        if resolved != official:
            raise CGeomError("Official budget writer only accepts the configured shared ledger")
        return positional.append_ledger(event, path=path)
    payload = dict(event)
    payload.setdefault("at_utc", datetime.now(timezone.utc).isoformat())
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return payload


def _npz_selected(path: Path, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise CGeomError(f"Required NPZ cache is missing: {path}")
    result: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as archive:
        if "artifact_signal" in archive.files or "y_true" in archive.files or "target" in archive.files:
            raise CGeomError("A cached prediction bundle unexpectedly contains ground-truth targets")
        for key in keys:
            if key not in archive.files:
                raise CGeomError(f"Required cache member is absent: {path.name}/{key}")
            value = archive[key]
            if value.dtype.kind == "O":
                raise CGeomError(f"Object dtype is prohibited: {path.name}/{key}")
            result[key] = value.copy()
    return result


def _verify_a02_cache(frame: pd.DataFrame, origin: int, fit_rows: np.ndarray,
                      score_rows: np.ndarray, score_low: int, score_high: int) -> dict[str, Any]:
    folder = A02_RUN / f"E{origin}" / "observed"
    model_path = folder / "model.joblib"
    receipt_path = folder / "fit_receipt.json"
    state_path = folder / "fit_state.json"
    pred_meta_path = folder / "predictions.json"
    pred_path = folder / "predictions.npz"
    required = (model_path, receipt_path, state_path, pred_meta_path, pred_path)
    if any(not path.is_file() for path in required):
        raise CGeomError(f"A02 origin E{origin} cache is incomplete")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt_check = dict(receipt)
    receipt_signature = receipt_check.pop("receipt_sha256", None)
    if receipt_signature != a02._json_sha256(receipt_check):
        raise CGeomError(f"A02 E{origin} fit receipt signature mismatch")
    model_sha = sha_file(model_path)
    if model_sha != receipt.get("model_sha256"):
        raise CGeomError(f"A02 E{origin} model hash differs from receipt")
    model = joblib.load(model_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    # The original A02 receipt stores a canonical digest of the complete
    # fit_state.json document (including its embedded model-state digest).
    # Compare using that producer contract, not the nested state_sha256 field.
    if model.audit_state() != state or canonical_sha256(state) != receipt.get("state_sha256"):
        raise CGeomError(f"A02 E{origin} model/state/receipt identity mismatch")
    if (receipt.get("fit_id") != f"A02_E{origin}_OBSERVED"
            or receipt.get("origin_cycle") != origin or receipt.get("residual_arm") != "observed"
            or receipt.get("input_sha256") != TRAIN_SHA256 or receipt.get("query_labels_passed_to_fit") is not False
            or receipt.get("test_data_used") is not False or receipt.get("task_estimator_fit_count") != 1):
        raise CGeomError(f"A02 E{origin} fit receipt source-boundary mismatch")
    if (model.origin_cycle != origin or model.text_space.origin_cycle != origin
            or model.text_space.fit_rows != len(fit_rows)
            or model.text_space.fit_row_positions_sha256 != a02._array_sha256(fit_rows, "<i8")
            or receipt.get("text_prefix_rows") != len(fit_rows)
            or receipt.get("text_prefix_row_positions_sha256") != a02._array_sha256(fit_rows, "<i8")):
        raise CGeomError(f"A02 E{origin} text fit prefix is not exact")
    if not np.array_equal(model.bank_cycles <= origin, np.ones(len(model.bank_cycles), dtype=bool)):
        raise CGeomError(f"A02 E{origin} residual bank contains future cycles")
    if (model.bank_row_positions.min() < 0 or model.bank_row_positions.max() >= len(frame)
            or len(np.unique(model.bank_row_positions)) != len(model.bank_row_positions)):
        raise CGeomError(f"A02 E{origin} residual bank row positions are invalid")
    expected_bank_ids = frame.iloc[model.bank_row_positions]["record_id"].astype(str).to_numpy()
    expected_bank_cycles = frame.iloc[model.bank_row_positions]["cycle_code"].to_numpy(dtype=np.int64)
    if (tuple(f"str:{value}" for value in expected_bank_ids) != tuple(model.bank_ids)
            or not np.array_equal(expected_bank_cycles, model.bank_cycles)
            or np.any(expected_bank_cycles > origin)
            or int(model.audit_state()["bank_rows"]) != len(expected_bank_ids)):
        raise CGeomError(f"A02 E{origin} cached residual bank does not join to official Train features")
    pred_meta = json.loads(pred_meta_path.read_text(encoding="utf-8"))
    if (pred_meta.get("fit_receipt_sha256") != receipt_signature
            or pred_meta.get("npz_sha256") != sha_file(pred_path)
            or pred_meta.get("query_labels_passed_to_prediction") is not False
            or pred_meta.get("score_rows") != len(score_rows)):
        raise CGeomError(f"A02 E{origin} prediction sidecar is not bound to its verified receipt")
    cache = _npz_selected(pred_path, (
        "score_rows", "score_record_ids", "score_cycles", "baseline_prediction", "prediction",
        "missing_mask", "entry_friction", "text_load_group", "neighbor_count",
        "neighbor_source_rows", "neighbor_ids", "neighbor_similarities", "correction",
        "correction_unclipped",
    ))
    expected_cycle = frame["cycle_code"].to_numpy(dtype=np.int64)[score_rows]
    expected_ids = frame.iloc[score_rows]["record_id"].astype(str).to_numpy()
    if (not np.array_equal(cache["score_rows"], score_rows)
            or not np.array_equal(cache["score_cycles"].astype(np.int64), expected_cycle)
            or not np.array_equal(cache["score_record_ids"].astype(str), expected_ids)):
        raise CGeomError(f"A02 E{origin} query row/ID/cycle arrays do not match official Train feature rows")
    if (expected_cycle.min() != score_low or expected_cycle.max() != score_high
            or np.any(expected_cycle <= origin)):
        raise CGeomError(f"A02 E{origin} query cycle boundaries differ from the frozen plan")
    query_text = frame.iloc[score_rows]["signal_stream"]
    _, actual_missing = model.text_space.transform(query_text)
    if not np.array_equal(actual_missing, cache["missing_mask"]):
        raise CGeomError(f"A02 E{origin} cached availability route differs from the frozen tokenizer")
    expected_entry = pd.to_numeric(frame.iloc[score_rows]["entry_friction"], errors="raise").to_numpy(dtype=np.int8)
    if not np.array_equal(expected_entry, cache["entry_friction"].astype(np.int8)):
        raise CGeomError(f"A02 E{origin} cached entry_friction groups do not match source features")
    prefix_text_load = pd.to_numeric(frame.loc[frame.cycle_code <= origin, "text_load"], errors="raise").to_numpy(dtype=np.float64)
    query_text_load = pd.to_numeric(frame.iloc[score_rows]["text_load"], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(prefix_text_load).all() or not np.isfinite(query_text_load).all():
        raise CGeomError("A02 fit-defined text_load groups require finite feature values")
    edges = np.quantile(prefix_text_load, [0.25, 0.50, 0.75], method="linear")
    expected_text_group = np.searchsorted(edges, query_text_load, side="right").astype(np.int8)
    if (not np.array_equal(expected_text_group, cache["text_load_group"].astype(np.int8))
            or pred_meta.get("text_load_quartile_fit_prefix_edges") != [float(x) for x in edges]):
        raise CGeomError(f"A02 E{origin} fit-defined text_load groups do not reproduce")
    n = len(score_rows)
    counts = cache["neighbor_count"].astype(np.int64)
    nb_rows = cache["neighbor_source_rows"].astype(np.int64)
    nb_ids = cache["neighbor_ids"].astype(str)
    sims = cache["neighbor_similarities"].astype(np.float64)
    if (nb_rows.shape != (n, a02.NEIGHBORS_MAX) or nb_ids.shape != nb_rows.shape
            or sims.shape != nb_rows.shape or counts.shape != (n,)
            or np.any(counts < 0) or np.any(counts > a02.NEIGHBORS_MAX)
            or not np.isfinite(sims).all() or np.any(sims < 0.0)):
        raise CGeomError(f"A02 E{origin} neighbor arrays have invalid shapes or values")
    bank_lookup = {int(row): i for i, row in enumerate(model.bank_row_positions)}
    m0 = np.sum(sims * sims, axis=1, dtype=np.float64)
    expected_raw = np.zeros(n, dtype=np.float64)
    for i in range(n):
        count = int(counts[i])
        chosen_rows = nb_rows[i, :count]
        if len(set(chosen_rows.tolist())) != count or any(int(row) not in bank_lookup for row in chosen_rows):
            raise CGeomError(f"A02 E{origin} neighbor rows are duplicate or outside the frozen donor bank")
        if count and (not np.all(nb_rows[i, count:] == -1) or not np.all(sims[i, count:] == 0.0)):
            raise CGeomError(f"A02 E{origin} neighbor padding differs from canonical output")
        if count == 0 and (m0[i] != 0.0 or not np.all(nb_rows[i] == -1)):
            raise CGeomError(f"A02 E{origin} empty neighbor row is inconsistent")
        bank_indices = np.asarray([bank_lookup[int(row)] for row in chosen_rows], dtype=np.int64)
        if count:
            if np.any(model.bank_cycles[bank_indices] > expected_cycle[i]):
                raise CGeomError(f"A02 E{origin} neighbor contains future-cycle donor")
            expected_ids_for_row = np.asarray([model.bank_ids[j] for j in bank_indices], dtype=str)
            if not np.array_equal(nb_ids[i, :count], expected_ids_for_row):
                raise CGeomError(f"A02 E{origin} neighbor IDs do not map to source bank rows")
            expected_raw[i] = np.dot(sims[i, :count] ** 2, model.bank_residuals[bank_indices]) / (a02.SHRINKAGE + m0[i])
        if cache["missing_mask"][i]:
            if count != 0 or m0[i] != 0.0:
                raise CGeomError(f"A02 E{origin} missing row must not use neighbors")
            expected_raw[i] = 0.0
    expected_correction = np.clip(expected_raw, -a02.CORRECTION_LIMIT, a02.CORRECTION_LIMIT)
    expected_correction[cache["missing_mask"]] = 0.0
    expected_prediction = np.clip(cache["baseline_prediction"] + expected_correction, 0.0, 1.0)
    expected_prediction[cache["missing_mask"]] = cache["baseline_prediction"][cache["missing_mask"]]
    if (not np.allclose(expected_raw, cache["correction_unclipped"], atol=1e-12, rtol=0)
            or not np.allclose(expected_correction, cache["correction"], atol=1e-12, rtol=0)
            or not np.allclose(expected_prediction, cache["prediction"], atol=1e-12, rtol=0)):
        raise CGeomError(f"A02 E{origin} cached prediction formula does not reconstruct from frozen residual bank")
    if (not np.isfinite(cache["prediction"]).all() or np.any(cache["prediction"] < 0.0)
            or np.any(cache["prediction"] > 1.0)):
        raise CGeomError(f"A02 E{origin} cached comparator probabilities are invalid")
    return {
        "origin": origin, "fit_rows": fit_rows, "score_rows": score_rows,
        "score_cycles": expected_cycle, "score_ids": expected_ids,
        "model": model, "cache": cache, "M0": m0,
        "a02_artifacts": {
            relpath(path): {"sha256": sha_file(path), "size_bytes": path.stat().st_size}
            for path in required
        },
        "a02_fit_receipt_sha256": receipt_signature,
        "a02_freeze_signature": receipt["freeze_signature"],
        "a02_state_sha256": state["state_sha256"],
        "a02_prediction_sha256": pred_meta["npz_sha256"],
        "fit_prefix_rows": len(fit_rows), "query_rows": n,
        "present_rows": int(np.sum(~actual_missing)), "missing_rows": int(np.sum(actual_missing)),
        "score_cycle_min_max": [int(expected_cycle.min()), int(expected_cycle.max())],
        "text_load_fit_edges": [float(x) for x in edges],
        "entry_friction_values": sorted(set(expected_entry.tolist())),
        "text_load_group_values": sorted(set(expected_text_group.tolist())),
        "original_a02_positive_mass_rows": int(np.sum(m0 > 0.0)),
    }


def relpath(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _verify_c2b_coefficients(frame: pd.DataFrame, origin: int, fit_rows: np.ndarray,
                             score_rows: np.ndarray) -> dict[str, Any]:
    unit = lambda component: C2B_RUN / "units" / f"POSITIONAL_TEXT_PILOT_V1_E{origin}_{component}.joblib"
    receipt_path = lambda component: C2B_RUN / "receipts" / f"POSITIONAL_TEXT_PILOT_V1_E{origin}_{component}.json"
    needed = [unit(name) for name in ("BAG_LR", "C2B_TFIDF", "C2B_STRUCTURED", "C2B_SCALER", "C2B_OHE")]
    needed += [receipt_path(name) for name in ("BAG_LR", "C2B_TFIDF")]
    manifest_path = C2B_RUN / f"MATRIX_MANIFEST_E{origin}.json"
    prediction_verification_path = C2B_RUN / f"PREDICTION_VERIFICATION_E{origin}.json"
    if any(not path.is_file() for path in (*needed, manifest_path, prediction_verification_path)):
        raise CGeomError(f"C2_B coefficient/token artifacts are incomplete for E{origin}")
    bag_receipt = json.loads(receipt_path("BAG_LR").read_text(encoding="utf-8"))
    tf_receipt = json.loads(receipt_path("C2B_TFIDF").read_text(encoding="utf-8"))
    bag_artifact, tfidf = joblib.load(unit("BAG_LR")), joblib.load(unit("C2B_TFIDF"))
    structured, scaler, encoder = (joblib.load(unit(name)) for name in ("C2B_STRUCTURED", "C2B_SCALER", "C2B_OHE"))
    estimator = bag_artifact.get("estimator") if isinstance(bag_artifact, dict) else None
    if estimator is None or not hasattr(estimator, "coef_") or not hasattr(estimator, "classes_"):
        raise CGeomError(f"C2_B BAG_LR cache lacks the expected fitted logistic model at E{origin}")
    for component, path, receipt in (("BAG_LR", unit("BAG_LR"), bag_receipt),
                                    ("C2B_TFIDF", unit("C2B_TFIDF"), tf_receipt)):
        if (sha_file(path) != receipt.get("artifact_sha256")
                or receipt.get("input_sha256") != TRAIN_SHA256
                or receipt.get("pilot_id") != "POSITIONAL_TEXT_PILOT_V1"
                or receipt.get("origin") != origin):
            raise CGeomError(f"C2_B {component} E{origin} artifact/receipt/input identity mismatch")
    bag_pop, tf_pop = bag_receipt["fit_population"], tf_receipt["fit_population"]
    expected_ids_hash = positional.sha256_array(frame.iloc[fit_rows]["record_id"].astype(str).to_numpy())
    expected_cycles_hash = positional.sha256_array(frame.iloc[fit_rows]["cycle_code"].to_numpy(dtype=np.int64))
    expected_rows_hash = positional.sha256_array(fit_rows)
    for label, pop in (("BAG_LR", bag_pop), ("C2B_TFIDF", tf_pop)):
        if (pop.get("rows") != len(fit_rows) or pop.get("row_indices_sha256") != expected_rows_hash
                or pop.get("record_ids_sha256") != expected_ids_hash or pop.get("cycles_sha256") != expected_cycles_hash
                or pop.get("cycle_min") != 47 or pop.get("cycle_max") != origin):
            raise CGeomError(f"C2_B {label} fit population differs from exact official prefix at E{origin}")
    if not np.array_equal(estimator.classes_, np.asarray([0, 1])) or bag_receipt["fit_metadata"].get("classes") != [0, 1]:
        raise CGeomError(f"C2_B E{origin} class order is not [0,1]")
    if positional.sha256_array(estimator.coef_) != bag_receipt["fit_metadata"].get("coef_sha256"):
        raise CGeomError(f"C2_B E{origin} LR coefficient hash differs from receipt")
    matrix_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    matrix_verification = json.loads(prediction_verification_path.read_text(encoding="utf-8"))
    manifest_sha = sha_file(manifest_path)
    if (matrix_manifest.get("origin") != origin
            or matrix_manifest.get("input_sha256") != TRAIN_SHA256
            or matrix_verification.get("matrix_manifest_sha256") != manifest_sha
            or bag_pop.get("matrix_manifest_sha256") != manifest_sha
            or matrix_verification.get("status") != "PASS_PREDICTION_INTEGRITY"):
        raise CGeomError(f"C2_B E{origin} matrix manifest/verification identity mismatch")
    fit_shape = matrix_manifest["matrices"]["fit_base"]["shape"]
    score_shape = matrix_manifest["matrices"]["score_base"]["shape"]
    if fit_shape[0] != len(fit_rows) or score_shape[0] != len(score_rows):
        raise CGeomError(f"C2_B E{origin} matrix row dimensions differ from frozen populations")
    a02_model = joblib.load(A02_RUN / f"E{origin}" / "observed" / "model.joblib")
    a02_vectorizer = a02_model.text_space.vectorizer
    if (tfidf.get_params(deep=False) != a02_vectorizer.get_params(deep=False)
            or tfidf.vocabulary_ != a02_vectorizer.vocabulary_
            or not np.array_equal(tfidf.idf_, a02_vectorizer.idf_)
            or tfidf.norm != a02_vectorizer.norm):
        raise CGeomError(f"C2_B and A02 vectorizer, complete vocabulary map, or IDF differs at E{origin}")
    numeric_width = int(len(scaler.mean_))
    ohe_width = int(len(encoder.get_feature_names_out()))
    structured_width = numeric_width + ohe_width
    text_width = len(a02_vectorizer.vocabulary_)
    total_width = int(fit_shape[1])
    if (int(score_shape[1]) != total_width or total_width != estimator.coef_.shape[1]
            or structured_width + text_width + 1 != total_width
            or tf_receipt["fit_metadata"].get("vocabulary_size") != text_width):
        raise CGeomError(f"C2_B E{origin} assembled feature dimensions do not map structured/text/missing blocks")
    # build_c2b_arm_matrices fixes the exact order: structured, text, missing flag.
    # Map coefficients by token string, not by an assumed vocabulary column order.
    beta = np.empty(text_width, dtype=np.float64)
    for token, a02_index in a02_vectorizer.vocabulary_.items():
        c2b_index = tfidf.vocabulary_.get(token)
        if c2b_index is None:
            raise CGeomError(f"C2_B lacks exact A02 token {token!r} at E{origin}")
        beta[int(a02_index)] = float(estimator.coef_[0, structured_width + int(c2b_index)])
    if not np.isfinite(beta).all():
        raise CGeomError(f"C2_B mapped token coefficients are non-finite at E{origin}")
    return {
        "origin": origin, "estimator": estimator, "vectorizer": tfidf,
        "structured_width": structured_width, "text_width": text_width,
        "feature_width": total_width, "beta_a02_order": beta,
        "fit_matrix_sha256": bag_receipt["fit_metadata"]["fit_matrix_sha256"],
        "score_matrix_sha256": bag_receipt["fit_metadata"]["score_matrix_sha256"],
        "fit_population": bag_pop, "bag_receipt": bag_receipt, "tfidf_receipt": tf_receipt,
        "c2b_artifacts": {
            relpath(unit(name)): {"sha256": sha_file(unit(name)), "size_bytes": unit(name).stat().st_size}
            for name in ("BAG_LR", "C2B_TFIDF", "C2B_STRUCTURED", "C2B_SCALER", "C2B_OHE")
        } | {
            relpath(receipt_path("BAG_LR")): {"sha256": sha_file(receipt_path("BAG_LR")), "size_bytes": receipt_path("BAG_LR").stat().st_size},
            relpath(receipt_path("C2B_TFIDF")): {"sha256": sha_file(receipt_path("C2B_TFIDF")), "size_bytes": receipt_path("C2B_TFIDF").stat().st_size},
            relpath(manifest_path): {"sha256": manifest_sha, "size_bytes": manifest_path.stat().st_size},
            relpath(prediction_verification_path): {"sha256": sha_file(prediction_verification_path), "size_bytes": prediction_verification_path.stat().st_size},
        },
    }


def load_origin_sources(frame: pd.DataFrame, origin: int) -> dict[str, Any]:
    fit_rows, score_rows, score_low, score_high = positional._fold_rows(frame, origin)
    a02_source = _verify_a02_cache(frame, origin, fit_rows, score_rows, score_low, score_high)
    c2b_source = _verify_c2b_coefficients(frame, origin, fit_rows, score_rows)
    if len(fit_rows) not in {29_705, 42_117} or len(score_rows) not in {20_903, 8_491}:
        raise CGeomError(f"Unexpected fit or score population at E{origin}")
    if c2b_source["text_width"] != len(a02_source["model"].text_space.vectorizer.vocabulary_):
        raise CGeomError(f"C2_B and A02 full vocabulary widths differ at E{origin}")
    return {**a02_source, **c2b_source, "fit_rows": fit_rows, "score_rows": score_rows,
            "score_low": score_low, "score_high": score_high}


def _orig_origin_lineage(source: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "origin": source["origin"], "fit_rows": source["fit_prefix_rows"],
        "query_rows": source["query_rows"], "fit_cycle_min_max": [47, source["origin"]],
        "query_cycle_min_max": source["score_cycle_min_max"],
        "present_rows": source["present_rows"], "missing_rows": source["missing_rows"],
        "a02_fit_receipt_sha256": source["a02_fit_receipt_sha256"],
        "a02_freeze_signature": source["a02_freeze_signature"],
        "a02_state_sha256": source["a02_state_sha256"],
        "a02_prediction_sha256": source["a02_prediction_sha256"],
        "a02_positive_mass_rows": source["original_a02_positive_mass_rows"],
        "fit_population": source["fit_population"],
        "feature_widths": {"structured": source["structured_width"], "text": source["text_width"],
                            "missing_indicator": 1, "assembled": source["feature_width"]},
        "c2b_lr_class_order": source["bag_receipt"]["fit_metadata"]["classes"],
        "c2b_lr_coefficient_sha256": source["bag_receipt"]["fit_metadata"]["coef_sha256"],
        "c2b_tfidf_vocabulary_sha256": source["tfidf_receipt"]["fit_metadata"]["vocabulary_sha256"],
        "c2b_tfidf_idf_sha256": source["tfidf_receipt"]["fit_metadata"]["idf_sha256"],
        "c2b_source_commit": source["bag_receipt"]["source_commit"],
        "c2b_freeze_signature": source["bag_receipt"]["freeze_signature"],
        "token_map_exact_match": True, "idf_bitwise_equal": True,
        "a02_artifacts": source["a02_artifacts"],
        "c2b_artifacts": source["c2b_artifacts"],
    }


def _scaled_geometry_benchmark(source: Mapping[str, Any], frame: pd.DataFrame, arm: str) -> dict[str, Any]:
    model = source["model"]
    multipliers, _ = coefficient_weights(source["beta_a02_order"], arm=arm, seed=PERMUTATION_SEED)
    bank = model.bank_matrix.multiply(multipliers).tocsr()
    bank = sparse.csr_matrix(sparse_linalg_normalize(bank), dtype=np.float64)
    score_rows = source["score_rows"]
    sample_n = min(64, len(score_rows))
    sample_idx = np.unique(np.linspace(0, len(score_rows) - 1, sample_n, dtype=np.int64))
    sample_rows = score_rows[sample_idx]
    query, _ = model.text_space.transform(frame.iloc[sample_rows]["signal_stream"])
    query = sparse.csr_matrix(sparse_linalg_normalize(query.multiply(multipliers).tocsr()), dtype=np.float64)
    started = time.perf_counter()
    max_dense_bytes = 0
    support_count = 0
    for start in range(0, len(sample_idx), a02.QUERY_BLOCK_ROWS):
        block = (query[start:start + a02.QUERY_BLOCK_ROWS] @ bank.T).toarray()
        max_dense_bytes = max(max_dense_bytes, int(block.nbytes))
        support_count += int(np.sum(np.any(block > 0.0, axis=1)))
    elapsed = time.perf_counter() - started
    return {
        "arm": arm, "sample_queries": int(len(sample_idx)),
        "full_bank_rows": int(bank.shape[0]), "full_vocabulary_columns": int(bank.shape[1]),
        "search_seconds": float(elapsed), "estimated_full_search_seconds": float(elapsed * len(score_rows) / max(1, len(sample_idx))),
        "largest_dense_block_bytes": max_dense_bytes, "queries_with_support": support_count,
        "query_row_positions_sha256": positional.sha256_array(sample_rows),
    }


def sparse_linalg_normalize(matrix: sparse.spmatrix) -> sparse.spmatrix:
    from sklearn.preprocessing import normalize
    return normalize(matrix, norm="l2", axis=1, copy=False)


def create_target_free_preflight() -> dict[str, Any]:
    require_clean_tree()
    if sha_file(TRAIN) != TRAIN_SHA256:
        raise CGeomError("Official Classification Train hash differs from the frozen source")
    ledger = _ledger_events()
    snapshot = _budget_snapshot(ledger)
    if (sha_file(LEDGER) != BASE_LEDGER_SHA256 or len(ledger) != BASE_LEDGER_ROWS
            or snapshot["global_cap"] != BASE_CAP or snapshot["effective_category_caps"] != BASE_CATEGORY_CAPS
            or snapshot["active_reservations_by_category"] != BASE_ACTIVE
            or any(snapshot.get(key) != value for key, value in BASE_FIT_COUNTS.items())):
        raise CGeomError(f"Live ledger differs from frozen prefit checkpoint: {snapshot}")
    if not SYNTHETIC_QA_PATH.is_file():
        raise CGeomError("Run source-bound --synthetic-qa before official target-free preflight")
    synthetic_qa = json.loads(SYNTHETIC_QA_PATH.read_text(encoding="utf-8"))
    if (synthetic_qa.get("status") != "PASS_SYNTHETIC_END_TO_END_AND_PERSISTENCE_RESUME"
            or synthetic_qa.get("source_commit") != _git("rev-parse", "HEAD")
            or synthetic_qa.get("source_hashes") != source_hashes()
            or synthetic_qa.get("ledger_sha256_before") != BASE_LEDGER_SHA256
            or synthetic_qa.get("ledger_sha256_after") != BASE_LEDGER_SHA256):
        raise CGeomError("Synthetic QA receipt is stale, failed, or not bound to current source/ledger")
    frame = positional.load_features()
    if "artifact_signal" in frame.columns or "threat_rank" in frame.columns:
        raise CGeomError("Target-free preflight unexpectedly loaded a target column")
    folds: dict[str, Any] = {}
    for origin in ORIGINS:
        source = load_origin_sources(frame, origin)
        with threadpool_limits(limits=THREAD_LIMIT):
            benchmarks = [_scaled_geometry_benchmark(source, frame, arm) for arm in ARMS]
        folds[str(origin)] = {
            **_orig_origin_lineage(source),
            "fit_row_positions_sha256": positional.sha256_array(source["fit_rows"]),
            "query_row_positions_sha256": positional.sha256_array(source["score_rows"]),
            "search_timing": benchmarks,
        }
        del source
        gc.collect()
    known_sources = {
        rel: {"sha256": sha_file(ROOT / rel), "size_bytes": (ROOT / rel).stat().st_size}
        for rel in SOURCE_PATHS
    }
    result = {
        "schema_version": 1, "pilot_id": PILOT_ID,
        "status": "PASS_TARGET_FREE_PREFIT_RESOURCE_AND_LINEAGE_PREFLIGHT",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": _git("rev-parse", "HEAD"), "source_hashes": source_hashes(),
        "input_path": relpath(TRAIN), "input_sha256": TRAIN_SHA256,
        "rows": len(frame), "target_columns_loaded": [],
        "query_labels_opened": False, "test_or_sample_submission_opened": False,
        "ledger_sha256": sha_file(LEDGER), "ledger_snapshot": snapshot,
        "synthetic_qa_path": relpath(SYNTHETIC_QA_PATH), "synthetic_qa_sha256": sha_file(SYNTHETIC_QA_PATH),
        "origins": folds, "source_files": known_sources,
        "fit_plan": {"predictor_fits": 4, "classifier_fits": 0, "vectorizer_fits": 0,
                     "preprocessing_fits": 0, "fit_ids": list(FIT_IDS)},
        "resource_preflight": resource_snapshot(require_initial_free=True),
        "memory_policy": {"max_rss_bytes": RSS_LIMIT, "initial_available_min_bytes": INITIAL_FREE_MIN,
                          "reserve_available_bytes": FREE_RESERVE, "threads": THREAD_LIMIT, "jobs": 1},
    }
    if PREFLIGHT_PATH.exists():
        old = json.loads(PREFLIGHT_PATH.read_text(encoding="utf-8"))
        if old.get("source_hashes") != result["source_hashes"] or old.get("ledger_sha256") != result["ledger_sha256"]:
            raise CGeomError("Existing target-free preflight differs from current source/input/ledger; stop")
        # Timings are nondeterministic; retain the first immutable preflight receipt.
        return old
    atomic_json(PREFLIGHT_PATH, result)
    return result


def create_freeze() -> dict[str, Any]:
    if not PREFLIGHT_PATH.is_file():
        raise CGeomError("Run target-free --preflight after committing code before freeze")
    require_clean_tree()
    preflight = json.loads(PREFLIGHT_PATH.read_text(encoding="utf-8"))
    hashes = source_hashes()
    if (preflight.get("status") != "PASS_TARGET_FREE_PREFIT_RESOURCE_AND_LINEAGE_PREFLIGHT"
            or preflight.get("source_commit") != _git("rev-parse", "HEAD")
            or preflight.get("source_hashes") != hashes
            or preflight.get("synthetic_qa_sha256") != sha_file(SYNTHETIC_QA_PATH)
            or sha_file(TRAIN) != TRAIN_SHA256 or sha_file(LEDGER) != BASE_LEDGER_SHA256
            or preflight.get("target_columns_loaded") != [] or preflight.get("query_labels_opened") is not False):
        raise CGeomError("CGEOM source/input/ledger changed after target-free preflight")
    payload = {
        "schema_version": 1, "pilot_id": PILOT_ID, "card_id": CARD_ID,
        "source_commit": preflight["source_commit"], "source_hashes": hashes,
        "recipe_path": relpath(RECIPE_PATH), "recipe_sha256": sha_file(RECIPE_PATH),
        "preflight_path": relpath(PREFLIGHT_PATH), "preflight_sha256": sha_file(PREFLIGHT_PATH),
        "input_path": relpath(TRAIN), "input_sha256": TRAIN_SHA256,
        "pre_reservation_ledger_sha256": BASE_LEDGER_SHA256,
        "pre_reservation_ledger_snapshot": preflight["ledger_snapshot"],
        "fit_ids": list(FIT_IDS), "fit_count": 4,
        "test_or_sample_submission_opened": False, "score_labels_opened": False,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    payload["freeze_signature"] = canonical_sha256(payload)
    write_create_once(FREEZE_PATH, json_bytes(payload))
    return payload


def load_freeze() -> dict[str, Any]:
    if not FREEZE_PATH.is_file():
        raise CGeomError("CGEOM freeze is missing")
    frozen = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    sig = frozen.pop("freeze_signature", None)
    if sig != canonical_sha256(frozen):
        raise CGeomError("CGEOM freeze signature mismatch")
    frozen["freeze_signature"] = sig
    require_clean_tree()
    if (frozen.get("source_commit") != _git("rev-parse", "HEAD")
            or frozen.get("source_hashes") != source_hashes()
            or sha_file(TRAIN) != TRAIN_SHA256):
        raise CGeomError("CGEOM frozen source commit, source files, or Train input changed")
    return frozen


def reserve_pilot(frozen: Mapping[str, Any]) -> dict[str, Any]:
    events = _ledger_events()
    snapshot = _budget_snapshot(events)
    amendments = [e for e in events if e.get("event") == "BUDGET_CAP_AMENDMENT" and e.get("amendment_id") == AMENDMENT_ID]
    reservations = [e for e in events if e.get("event") == "CARD_FITS_RESERVED" and e.get("reservation_key") == RESERVATION_KEY]
    if len(amendments) > 1 or len(reservations) > 1:
        raise CGeomError("Duplicate CGEOM cap amendment or reservation")
    if not amendments and not reservations:
        if (sha_file(LEDGER) != frozen["pre_reservation_ledger_sha256"]
                or snapshot != frozen["pre_reservation_ledger_snapshot"]
                or len(events) != BASE_LEDGER_ROWS):
            raise CGeomError("Live shared ledger changed after CGEOM freeze; do not reserve")
        append_budget_event({
            "event": "BUDGET_CAP_AMENDMENT", "batch_id": "optimization_v3_research_next_v1",
            "amendment_id": AMENDMENT_ID, "card_id": CARD_ID,
            "authorized_by": "Sam via Sol; exactly four indexed memory predictor fits",
            "prior_global_cap": BASE_CAP, "global_cap": AMENDED_CAP,
            "prior_effective_caps": BASE_CATEGORY_CAPS, "effective_caps": AMENDED_CATEGORY_CAPS,
            "active_reservations_before": BASE_ACTIVE, "active_total_before": 114,
            "historical_started_saved_failed_before": BASE_FIT_COUNTS,
            "pre_amendment_event_rows": BASE_LEDGER_ROWS,
            "pre_amendment_ledger_sha256": BASE_LEDGER_SHA256,
            "source_commit": frozen["source_commit"], "freeze_signature": frozen["freeze_signature"],
            "reason": "Authorized CGEOM_V1: two origins times aligned/permuted indexed memory predictors; no classifier/vectorizer/preprocessing fit.",
        }, path=LEDGER)
        events = _ledger_events()
        snapshot = _budget_snapshot(events)
    if not amendments:
        amendments = [e for e in events if e.get("event") == "BUDGET_CAP_AMENDMENT" and e.get("amendment_id") == AMENDMENT_ID]
    if len(amendments) != 1 or amendments[0].get("freeze_signature") != frozen["freeze_signature"]:
        raise CGeomError("CGEOM cap amendment is absent or not bound to frozen sources")
    reservation_body = {
        "event": "CARD_FITS_RESERVED", "batch_id": "optimization_v3_research_next_v1",
        "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
        "category": "pilot", "fits": 4, "fit_ids": list(FIT_IDS),
        "amendment_id": AMENDMENT_ID, "global_cap": AMENDED_CAP,
        "effective_category_cap": AMENDED_CATEGORY_CAPS["pilot"],
        "freeze_signature": frozen["freeze_signature"], "source_commit": frozen["source_commit"],
    }
    if not reservations:
        expected = {
            "global_cap": AMENDED_CAP, "effective_category_caps": AMENDED_CATEGORY_CAPS,
            "active_reservations_by_category": BASE_ACTIVE, "active_reservations": 114,
            **BASE_FIT_COUNTS,
        }
        if _with_event_rows(snapshot) != expected:
            raise CGeomError(f"CGEOM budget state changed between cap amendment and reservation: {snapshot}")
        append_budget_event(reservation_body, path=LEDGER)
        events = _ledger_events()
    else:
        if any(reservations[0].get(k) != v for k, v in reservation_body.items()):
            raise CGeomError("Existing CGEOM reservation differs from its exact four-fit plan")
    snapshot = _assert_budget(events, phase="after_reservation")
    if (snapshot["active_reservations"] != 118 or snapshot["fit_starts"] != 114
            or snapshot["fit_saved"] != 110 or snapshot["fit_failed"] != 4 or snapshot["fit_pending"] != 0):
        raise CGeomError(f"CGEOM reservation did not preserve the 118-active/114-start checkpoint: {snapshot}")
    return {"snapshot": snapshot, "ledger_sha256": sha_file(LEDGER),
            "amendment": amendments[0], "reservation": reservation_body}


def _unit_paths(fit_id: str) -> tuple[Path, Path]:
    return RUN_ROOT / "units" / f"{fit_id}.npz", RUN_ROOT / "receipts" / f"{fit_id}.json"


def _fit_events(fit_id: str, events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    starts = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("fit_id") == fit_id and e.get("reservation_key") == RESERVATION_KEY]
    terminal = [e for e in events if e.get("event") in {"FIT_SAVED", "FIT_FAILED"} and e.get("fit_id") == fit_id and e.get("reservation_key") == RESERVATION_KEY]
    return starts, terminal


def _prediction_arrays(payload: Mapping[str, Any]) -> dict[str, np.ndarray]:
    keys = (
        "candidate_probability", "baseline_probability", "original_a02_probability", "m0",
        "neighbor_count", "neighbor_source_rows", "neighbor_ids", "neighbor_similarities",
        "neighbor_weights", "new_cosine_squared_mass", "correction_raw", "correction",
        "missing_mask", "query_rows", "query_cycles", "query_record_ids",
    )
    result = {key: np.asarray(payload[key]).copy() for key in keys}
    for key, array in result.items():
        if array.dtype.kind == "O":
            raise CGeomError(f"Object-dtype array cannot be written to the CGEOM archive: {key}")
    return result


def _write_npz_atomic(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    # Full production-schema roundtrip, with pickle disabled and every member read.
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(arrays):
            raise CGeomError("CGEOM output archive member set differs after serialization")
        for key, original in arrays.items():
            actual = archive[key]
            if actual.dtype.kind == "O" or actual.dtype != original.dtype or not np.array_equal(actual, original):
                raise CGeomError(f"CGEOM NPZ roundtrip mismatch in {key}")


def synthetic_unit_lifecycle(root: Path, ledger_path: Path, fit_id: str,
                             fit_callback: Any, *, interrupt_after_receipt: bool = False) -> dict[str, Any]:
    """Exercise persistence/resume with explicit temporary paths only."""
    if ledger_path.resolve() == LEDGER.resolve():
        raise CGeomError("Synthetic lifecycle cannot use or write the production budget ledger")
    if not fit_id or Path(fit_id).name != fit_id:
        raise CGeomError("Synthetic fit ID must be a simple filename-safe identifier")
    artifact_path = root / "units" / f"{fit_id}.npz"
    receipt_path = root / "receipts" / f"{fit_id}.json"
    events = []
    if ledger_path.exists():
        events = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    starts = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("fit_id") == fit_id]
    terminals = [e for e in events if e.get("event") in {"FIT_SAVED", "FIT_FAILED"} and e.get("fit_id") == fit_id]
    if len(starts) > 1 or len(terminals) > 1 or (terminals and not starts):
        raise CGeomError("Synthetic lifecycle ledger accounting is inconsistent")
    if terminals:
        if terminals[0]["event"] != "FIT_SAVED" or not artifact_path.is_file() or not receipt_path.is_file():
            raise CGeomError("Synthetic terminal artifact is missing or failed")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("artifact_sha256") != sha_file(artifact_path):
            raise CGeomError("Synthetic durable artifact receipt mismatch")
        with np.load(artifact_path, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        return {"status": "RESUMED_SAVED", "fit_id": fit_id, "fit_callback_called": False,
                "arrays": arrays, "receipt_sha256": sha_file(receipt_path)}
    if starts:
        if not artifact_path.is_file() or not receipt_path.is_file():
            raise CGeomError("Synthetic started fit has no complete bundle; retry is prohibited")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("artifact_sha256") != sha_file(artifact_path):
            raise CGeomError("Synthetic recovery found a mismatched durable receipt")
        with np.load(artifact_path, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        append_budget_event({"event": "FIT_SAVED", "fit_id": fit_id,
                             "receipt_sha256": sha_file(receipt_path), "synthetic": True},
                            path=ledger_path, synthetic=True)
        return {"status": "RECOVERED_DURABLE_RECEIPT", "fit_id": fit_id,
                "fit_callback_called": False, "arrays": arrays, "receipt_sha256": sha_file(receipt_path)}
    if artifact_path.exists() or receipt_path.exists():
        raise CGeomError("Synthetic unaccounted output exists before FIT_STARTED")
    append_budget_event({"event": "FIT_STARTED", "fit_id": fit_id, "synthetic": True},
                        path=ledger_path, synthetic=True)
    payload = fit_callback()
    arrays = _prediction_arrays(payload)
    _write_npz_atomic(artifact_path, arrays)
    receipt = {
        "fit_id": fit_id, "artifact_sha256": sha_file(artifact_path),
        "array_sha256": {key: positional.sha256_array(value) for key, value in arrays.items()},
        "array_dtypes": {key: str(value.dtype) for key, value in arrays.items()},
        "query_labels_opened": False,
    }
    atomic_json(receipt_path, receipt)
    if interrupt_after_receipt:
        raise SyntheticInterruption("synthetic interruption after receipt and before FIT_SAVED")
    append_budget_event({"event": "FIT_SAVED", "fit_id": fit_id,
                         "receipt_sha256": sha_file(receipt_path), "synthetic": True},
                        path=ledger_path, synthetic=True)
    return {"status": "SAVED", "fit_id": fit_id, "fit_callback_called": True,
            "arrays": arrays, "receipt_sha256": sha_file(receipt_path)}


def create_synthetic_qa() -> dict[str, Any]:
    require_clean_tree()
    before_ledger_sha = sha_file(LEDGER)
    command = [sys.executable, "-m", "pytest", "-q", "tests/test_optimization_v3_a02_coefficient_geometry_v1.py"]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    output = (completed.stdout + "\n" + completed.stderr).strip()
    if completed.returncode != 0:
        raise CGeomError("Focused synthetic QA failed before freeze:\n" + output[-6000:])
    match = re.search(r"(\d+ passed(?:, [^\n]+)? in [^\n]+)", output)
    after_ledger_sha = sha_file(LEDGER)
    if before_ledger_sha != BASE_LEDGER_SHA256 or after_ledger_sha != before_ledger_sha:
        raise CGeomError("Synthetic QA changed the live ledger or found a different starting ledger")
    qa = {
        "status": "PASS_SYNTHETIC_END_TO_END_AND_PERSISTENCE_RESUME",
        "pilot_id": PILOT_ID, "source_commit": _git("rev-parse", "HEAD"),
        "source_hashes": source_hashes(), "input_sha256": TRAIN_SHA256,
        "ledger_sha256_before": before_ledger_sha, "ledger_sha256_after": after_ledger_sha,
        "official_budget_writer_blocked_in_synthetic_mode": True,
        "synthetic_ledger_path_only": True, "pytest_exit_code": completed.returncode,
        "pytest_summary": match.group(1) if match else output[-500:],
        "pytest_command": command,
        "production_payload_roundtrip": "all output NPZ members read with allow_pickle=False and explicit Unicode IDs",
        "resume_check": "durable bundle and receipt after interrupted start recovered without refit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if SYNTHETIC_QA_PATH.exists():
        old = json.loads(SYNTHETIC_QA_PATH.read_text(encoding="utf-8"))
        if (old.get("status") != qa["status"] or old.get("source_commit") != qa["source_commit"]
                or old.get("source_hashes") != qa["source_hashes"]
                or old.get("ledger_sha256_after") != after_ledger_sha):
            raise CGeomError("Existing synthetic QA receipt is stale for current source or ledger")
        return old
    atomic_json(SYNTHETIC_QA_PATH, qa)
    return qa


def _unit_source_hashes(source: Mapping[str, Any]) -> dict[str, Any]:
    return {**source["a02_artifacts"], **source["c2b_artifacts"]}


def _make_unit_receipt(fit_id: str, origin: int, arm: str, frozen: Mapping[str, Any],
                       source: Mapping[str, Any], state: Mapping[str, Any], arrays: Mapping[str, np.ndarray],
                       started_at: str, resource_before: Mapping[str, Any], peak_rss: int,
                       elapsed: float, artifact_path: Path) -> dict[str, Any]:
    return {
        "schema_version": 1, "pilot_id": PILOT_ID, "card_id": CARD_ID,
        "fit_id": fit_id, "origin": origin, "arm": arm,
        "fit_kind": "INDEXED_RESIDUAL_MEMORY_PREDICTOR",
        "fit_consumed_at_start": True, "frozen_source_commit": frozen["source_commit"],
        "freeze_signature": frozen["freeze_signature"], "source_hashes": frozen["source_hashes"],
        "input_path": relpath(TRAIN), "input_sha256": TRAIN_SHA256,
        "fit_prefix_rows": source["fit_prefix_rows"], "fit_cycle_min_max": [47, origin],
        "donor_rows": int(len(source["model"].bank_ids)),
        "query_rows": int(len(arrays["query_rows"])),
        "query_rows_sha256": positional.sha256_array(arrays["query_rows"]),
        "query_ids_sha256": positional.sha256_array(arrays["query_record_ids"]),
        "query_cycles_sha256": positional.sha256_array(arrays["query_cycles"]),
        "fit_state": dict(state), "fit_state_sha256": canonical_sha256(state),
        "source_artifacts": _unit_source_hashes(source),
        "artifact_path": relpath(artifact_path), "artifact_sha256": sha_file(artifact_path),
        "array_sha256": {key: positional.sha256_array(value) for key, value in arrays.items()},
        "array_dtypes": {key: str(value.dtype) for key, value in arrays.items()},
        "diagnostics": dict(state.get("diagnostics", {})),
        "resource_before": dict(resource_before), "peak_rss_bytes": int(peak_rss),
        "rss_limit_bytes": RSS_LIMIT, "threads": THREAD_LIMIT, "jobs": 1,
        "started_at_utc": started_at, "elapsed_seconds": float(elapsed),
        "query_labels_opened": False, "test_or_sample_submission_opened": False,
    }


def _read_unit(fit_id: str, frozen: Mapping[str, Any], source: Mapping[str, Any], origin: int,
               arm: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    artifact_path, receipt_path = _unit_paths(fit_id)
    if not artifact_path.is_file() or not receipt_path.is_file():
        raise CGeomError(f"CGEOM durable unit artifact/receipt missing for {fit_id}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (receipt.get("fit_id") != fit_id or receipt.get("origin") != origin or receipt.get("arm") != arm
            or receipt.get("freeze_signature") != frozen["freeze_signature"]
            or receipt.get("source_hashes") != frozen["source_hashes"]
            or receipt.get("artifact_sha256") != sha_file(artifact_path)
            or receipt.get("query_labels_opened") is not False):
        raise CGeomError(f"CGEOM durable receipt does not match frozen unit {fit_id}")
    arrays = _npz_selected(artifact_path, tuple(receipt["array_sha256"].keys()))
    if set(arrays) != set(receipt["array_sha256"]):
        raise CGeomError(f"CGEOM durable output schema differs for {fit_id}")
    for key, array in arrays.items():
        if positional.sha256_array(array) != receipt["array_sha256"][key]:
            raise CGeomError(f"CGEOM durable output checksum mismatch: {fit_id}/{key}")
    return arrays, receipt


def _fit_one(frozen: Mapping[str, Any], frame: pd.DataFrame, origin: int, arm: str,
             fit_index: int) -> dict[str, Any]:
    fit_id = f"CGEOM_V1_E{origin}_{arm.upper()}"
    artifact_path, receipt_path = _unit_paths(fit_id)
    events = _ledger_events()
    _assert_budget(events, phase="before_fit")
    starts, terminal = _fit_events(fit_id, events)
    if len(starts) > 1 or len(terminal) > 1:
        raise CGeomError(f"Duplicate ledger history for {fit_id}")
    source = load_origin_sources(frame, origin)
    if terminal:
        if len(starts) != 1 or terminal[0]["event"] != "FIT_SAVED":
            raise CGeomError(f"{fit_id} is terminal-failed or has inconsistent history; retries prohibited")
        arrays, receipt = _read_unit(fit_id, frozen, source, origin, arm)
        if terminal[0].get("receipt_sha256") != sha_file(receipt_path):
            raise CGeomError(f"Saved ledger receipt hash differs for {fit_id}")
        del source
        gc.collect()
        return {"fit_id": fit_id, "resumed_from_durable_receipt": True,
                "artifact_sha256": sha_file(artifact_path), "receipt_sha256": sha_file(receipt_path)}
    if starts:
        # Crash recovery is permitted only when a complete immutable bundle and
        # receipt already exist; otherwise the consumed slot is terminally failed.
        if artifact_path.is_file() and receipt_path.is_file():
            arrays, receipt = _read_unit(fit_id, frozen, source, origin, arm)
            append_budget_event({
                "event": "FIT_SAVED", "batch_id": "optimization_v3_research_next_v1",
                "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
                "fit_id": fit_id, "cutoff": origin, "fit_kind": "INDEXED_RESIDUAL_MEMORY_PREDICTOR",
                "receipt_sha256": sha_file(receipt_path), "amendment_id": AMENDMENT_ID,
                "recovered_durable_receipt": True,
            }, path=LEDGER)
            del arrays, receipt, source
            gc.collect()
            return {"fit_id": fit_id, "resumed_from_durable_receipt": True,
                    "artifact_sha256": sha_file(artifact_path), "receipt_sha256": sha_file(receipt_path)}
        append_budget_event({
            "event": "FIT_FAILED", "batch_id": "optimization_v3_research_next_v1",
            "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
            "fit_id": fit_id, "cutoff": origin, "fit_kind": "INDEXED_RESIDUAL_MEMORY_PREDICTOR",
            "reason": "terminal_failure:started_fit_has_no_complete_durable_receipt; retry_prohibited",
            "amendment_id": AMENDMENT_ID,
        }, path=LEDGER)
        _release_unstarted(frozen, f"CGEOM recovery found incomplete started unit {fit_id}; retry prohibited")
        raise CGeomError(f"{fit_id} had a started fit without a complete durable receipt; marked failed and released only unstarted IDs")
    if artifact_path.exists() or receipt_path.exists():
        raise CGeomError(f"Unaccounted output exists before FIT_STARTED for {fit_id}")
    require_initial = fit_index == 0
    resource_before = resource_snapshot(require_initial_free=require_initial)
    start_utc = datetime.now(timezone.utc).isoformat()
    append_budget_event({
        "event": "FIT_STARTED", "batch_id": "optimization_v3_research_next_v1",
        "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
        "fit_id": fit_id, "cutoff": origin, "component": arm,
        "fit_kind": "INDEXED_RESIDUAL_MEMORY_PREDICTOR", "fit_consumed_at_start": True,
        "amendment_id": AMENDMENT_ID, "freeze_signature": frozen["freeze_signature"],
        "source_commit": frozen["source_commit"], "resource_snapshot": resource_before,
        "fit_population": {"prefix_rows": source["fit_prefix_rows"], "donor_rows": len(source["model"].bank_ids),
                           "query_rows": len(source["score_rows"]), "target_used": False},
    }, path=LEDGER)
    started = time.perf_counter()
    try:
        with PeakRSS() as monitor, threadpool_limits(limits=THREAD_LIMIT):
            predictor = CoefficientGeometryPredictor(origin_cycle=origin, arm=arm).fit(
                text_space=source["model"].text_space,
                bank_matrix=source["model"].bank_matrix,
                bank_ids=source["model"].bank_ids,
                bank_row_positions=source["model"].bank_row_positions,
                bank_cycles=source["model"].bank_cycles,
                bank_oof_probabilities=source["model"].bank_oof_probabilities,
                bank_residuals=source["model"].bank_residuals,
                beta_in_a02_column_order=source["beta_a02_order"],
            )
            query_frame = frame.iloc[source["score_rows"]]
            payload = predictor.predict(
                query_texts=query_frame["signal_stream"],
                query_record_ids=query_frame["record_id"].astype(str).to_numpy(),
                query_row_positions=source["score_rows"], query_cycles=source["score_cycles"],
                baseline_probabilities=source["cache"]["baseline_prediction"],
                original_a02_probabilities=source["cache"]["prediction"],
                original_a02_neighbor_source_rows=source["cache"]["neighbor_source_rows"],
                original_a02_neighbor_similarities=source["cache"]["neighbor_similarities"],
                missing_mask=source["cache"]["missing_mask"],
            )
            arrays = _prediction_arrays(payload)
            if not np.array_equal(arrays["original_a02_probability"], source["cache"]["prediction"]):
                raise CGeomError("Candidate output lost the exact cached A02 comparator")
            if not np.array_equal(arrays["candidate_probability"][arrays["missing_mask"]],
                                  arrays["original_a02_probability"][arrays["missing_mask"]]):
                raise CGeomError("Missing-stream route changed from exact A02 baseline")
            _write_npz_atomic(artifact_path, arrays)
            state = {**predictor.audit_state(), "diagnostics": payload["diagnostics"]}
            elapsed = time.perf_counter() - started
            receipt = _make_unit_receipt(fit_id, origin, arm, frozen, source, state, arrays,
                                         start_utc, resource_before, monitor.peak, elapsed, artifact_path)
            atomic_json(receipt_path, receipt)
            _read_unit(fit_id, frozen, source, origin, arm)
            if monitor.peak > RSS_LIMIT:
                raise CGeomError(f"CGEOM fit {fit_id} exceeded 6 GiB RSS cap: {monitor.peak}")
        receipt_sha = sha_file(receipt_path)
        append_budget_event({
            "event": "FIT_SAVED", "batch_id": "optimization_v3_research_next_v1",
            "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
            "fit_id": fit_id, "cutoff": origin, "fit_kind": "INDEXED_RESIDUAL_MEMORY_PREDICTOR",
            "receipt_sha256": receipt_sha, "amendment_id": AMENDMENT_ID,
        }, path=LEDGER)
        _assert_budget(_ledger_events(), phase="after_fit")
        del predictor, source, payload, arrays
        gc.collect()
        return {"fit_id": fit_id, "resumed_from_durable_receipt": False,
                "artifact_sha256": sha_file(artifact_path), "receipt_sha256": receipt_sha,
                "elapsed_seconds": float(elapsed), "peak_rss_bytes": int(monitor.peak)}
    except BaseException as exc:
        events = _ledger_events()
        starts, terminal = _fit_events(fit_id, events)
        if starts and not terminal:
            append_budget_event({
                "event": "FIT_FAILED", "batch_id": "optimization_v3_research_next_v1",
                "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
                "fit_id": fit_id, "cutoff": origin, "fit_kind": "INDEXED_RESIDUAL_MEMORY_PREDICTOR",
                "reason": f"terminal_failure:{type(exc).__name__}:{exc}", "amendment_id": AMENDMENT_ID,
            }, path=LEDGER)
        # No retry. Release only IDs never started, and only after all consumed
        # starts are terminal in the shared ledger.
        _release_unstarted(frozen, f"CGEOM stopped at {fit_id}: {type(exc).__name__}: {exc}")
        raise


def _release_unstarted(frozen: Mapping[str, Any], reason: str) -> dict[str, Any] | None:
    events = _ledger_events()
    own = _own_events(events)
    started = {e["fit_id"] for e in own["FIT_STARTED"]}
    unstarted = [fit_id for fit_id in FIT_IDS if fit_id not in started]
    prior = [e for e in events if e.get("event") == "RESERVATION_UNUSED_RELEASED" and e.get("reservation_key") == RESERVATION_KEY]
    if prior:
        if int(prior[-1].get("unused_fits", 0)) != len(unstarted):
            raise CGeomError("Existing CGEOM unused-slot release does not match unstarted IDs")
        return prior[-1]
    if not unstarted:
        return None
    if _budget_snapshot(events)["fit_pending"]:
        raise CGeomError("Cannot release unstarted CGEOM IDs while a consumed fit is pending")
    return append_budget_event({
        "event": "RESERVATION_UNUSED_RELEASED", "batch_id": "optimization_v3_research_next_v1",
        "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
        "amendment_id": AMENDMENT_ID, "unused_fits": len(unstarted), "fit_ids": unstarted,
        "freeze_signature": frozen["freeze_signature"], "reason": reason,
    }, path=LEDGER)


def run_predictions() -> dict[str, Any]:
    frozen = load_freeze()
    events = _ledger_events()
    _assert_budget(events, phase="before_fit")
    if _budget_snapshot(events)["fit_pending"]:
        raise CGeomError("A different fit is pending in the Research Next shared ledger")
    frame = positional.load_features()
    outputs = []
    for fit_index, (origin, arm) in enumerate((o, a) for o in ORIGINS for a in ARMS):
        if sha_file(TRAIN) != TRAIN_SHA256 or source_hashes() != frozen["source_hashes"]:
            raise CGeomError("Frozen input/source changed during CGEOM fit sequence")
        outputs.append(_fit_one(frozen, frame, origin, arm, fit_index))
        del frame
        gc.collect()
        frame = positional.load_features()
    final_events = _ledger_events()
    final_snapshot = _assert_budget(final_events, phase="after_all_fits")
    if final_snapshot["fit_pending"] != 0 or final_snapshot["fit_starts"] != 118 or final_snapshot["fit_saved"] != 114 or final_snapshot["fit_failed"] != 4:
        raise CGeomError(f"CGEOM did not complete exactly four saved fit units: {final_snapshot}")
    return {"status": "FOUR_FITS_SAVED", "fits": outputs, "ledger_sha256": sha_file(LEDGER),
            "ledger_snapshot": final_snapshot}


def _read_query_labels_after_prelabel(frame: pd.DataFrame, frozen: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    if not PRELABEL_PATH.is_file():
        raise CGeomError("CGEOM query labels remain closed: no independent prelabel verification")
    prelabel = json.loads(PRELABEL_PATH.read_text(encoding="utf-8"))
    if prelabel.get("status") != "PASS_INDEPENDENT_NUMERICAL_AND_BOUNDARY_VERIFICATION" or prelabel.get("labels_opened") is not False:
        raise CGeomError("CGEOM prelabel verifier did not produce the required PASS")
    if (prelabel.get("freeze_signature") != frozen["freeze_signature"]
            or sha_file(TRAIN) != TRAIN_SHA256 or prelabel.get("source_commit") != frozen["source_commit"]):
        raise CGeomError("CGEOM prelabel verification is not bound to frozen input/source")
    for fit_id in FIT_IDS:
        artifact, receipt = _unit_paths(fit_id)
        if (not artifact.is_file() or not receipt.is_file()
                or prelabel.get("unit_artifact_sha256", {}).get(fit_id) != sha_file(artifact)
                or prelabel.get("unit_receipt_sha256", {}).get(fit_id) != sha_file(receipt)):
            raise CGeomError(f"CGEOM output changed after independent prelabel pass: {fit_id}")
    if _budget_snapshot(_ledger_events())["fit_pending"] != 0:
        raise CGeomError("CGEOM still has a pending fit before opening query labels")
    # Target is read exactly once, for E67's full query superset. E75 is sliced
    # from those labels only after row/cycle identity has been verified.
    frame_cycles = frame["cycle_code"].to_numpy(dtype=np.int64)
    rows = np.flatnonzero((frame_cycles >= 68) & (frame_cycles <= 79)).astype(np.int64)
    expected_cycles = frame_cycles[rows]
    labels = positional.target_values_for_rows(rows, expected_cycles, purpose="outer-score-after-predictions")
    receipt = {
        "status": "QUERY_LABELS_READ_AFTER_PREFIT_PASS",
        "prelabel_verification_sha256": sha_file(PRELABEL_PATH),
        "input_sha256": TRAIN_SHA256, "query_rows": int(len(rows)),
        "query_row_positions_sha256": positional.sha256_array(rows),
        "query_cycles_sha256": positional.sha256_array(expected_cycles),
        "query_labels_opened": True, "test_or_sample_submission_opened": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(RUN_ROOT / "QUERY_LABEL_ACCESS.json", receipt)
    return labels, {"rows": rows, "cycles": expected_cycles, "receipt": receipt}


def binary_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    actual = np.asarray(y, dtype=np.int8)
    p = np.asarray(probability, dtype=np.float64)
    if (actual.ndim != 1 or p.shape != actual.shape or not np.isfinite(p).all()
            or np.any((actual != 0) & (actual != 1)) or np.any((p < 0.0) | (p > 1.0))):
        raise CGeomError("Invalid arrays supplied to the binary metric implementation")
    pred = p >= 0.50
    tp = int(np.sum((actual == 1) & pred))
    fp = int(np.sum((actual == 0) & pred))
    fn = int(np.sum((actual == 1) & ~pred))
    tn = int(np.sum((actual == 0) & ~pred))
    def class_f1(label: int) -> float:
        mask = actual == label
        p_mask = pred if label == 1 else ~pred
        ctp = int(np.sum(mask & p_mask))
        cfp = int(np.sum((actual != label) & p_mask))
        cfn = int(np.sum(mask & ~p_mask))
        precision = ctp / (ctp + cfp) if ctp + cfp else 0.0
        recall = ctp / (ctp + cfn) if ctp + cfn else 0.0
        return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    clipped = np.clip(p, EPSILON, 1.0 - EPSILON)
    return {
        "n": int(len(actual)), "positive": int(np.sum(actual == 1)), "negative": int(np.sum(actual == 0)),
        "macro_f1": float((class_f1(0) + class_f1(1)) / 2.0),
        "positive_recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "roc_auc": float(roc_auc_score(actual, p)) if np.unique(actual).size == 2 else None,
        "average_precision": float(average_precision_score(actual, p)) if np.unique(actual).size == 2 else None,
        "log_loss_clip_1e-15": float(-np.mean(actual * np.log(clipped) + (1 - actual) * np.log(1.0 - clipped))),
        "brier": float(np.mean((p - actual) ** 2)),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "threshold": 0.50, "log_loss_probability_clip_epsilon": EPSILON,
        "zero_denominator_precision_recall_f1": 0.0,
    }


def _view_indices(cycles: np.ndarray, origin: int) -> dict[str, np.ndarray]:
    if origin == 67:
        return {
            "E67_H12": np.ones(len(cycles), dtype=bool),
            "E67_W4": (cycles >= 68) & (cycles <= 71),
            "E67_W5": (cycles >= 72) & (cycles <= 75),
            "E67_W6": (cycles >= 76) & (cycles <= 79),
        }
    return {"E75_W6": (cycles >= 76) & (cycles <= 79)}


def _load_all_outputs(frozen: Mapping[str, Any], frame: pd.DataFrame) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for origin in ORIGINS:
        source = load_origin_sources(frame, origin)
        arms: dict[str, dict[str, np.ndarray]] = {}
        receipts = {}
        for arm in ARMS:
            fit_id = f"CGEOM_V1_E{origin}_{arm.upper()}"
            arrays, receipt = _read_unit(fit_id, frozen, source, origin, arm)
            arms[arm] = arrays
            receipts[arm] = receipt
        result[origin] = {"source": source, "arms": arms, "receipts": receipts}
    return result


def _view_metrics(y: np.ndarray, arms: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {name: binary_metrics(y, p) for name, p in arms.items()}


def _group_values(source: Mapping[str, Any]) -> dict[str, np.ndarray]:
    cache = source["cache"]
    return {
        "availability": np.where(cache["missing_mask"], "missing", "present"),
        "text_load_quartile": np.asarray([f"Q{int(x) + 1}" for x in cache["text_load_group"]], dtype="<U2"),
        "entry_friction": np.asarray([str(int(x)) for x in cache["entry_friction"]], dtype="<U8"),
    }


def _gate_record(name: str, gain: float, lower_bound: float, *, context: Mapping[str, Any]) -> dict[str, Any]:
    passed = bool(gain + GATE_TOLERANCE >= lower_bound)
    return {"gate": name, "gain": float(gain), "required_minimum_gain": float(lower_bound),
            "tolerance": GATE_TOLERANCE, "status": "PASS" if passed else "FAIL", **dict(context)}


def _subgroup_checks(origin_data: Mapping[int, Any], labels_by_origin: Mapping[int, np.ndarray],
                     views: Mapping[str, Any]) -> list[dict[str, Any]]:
    checks = []
    for view_name, mapping in views.items():
        origin = mapping["origin"]
        origin_item = origin_data[origin]
        source = origin_item["source"]
        local_mask = mapping["mask"]
        y = labels_by_origin[origin][local_mask]
        groups = _group_values(source)
        for group_name, group_values in groups.items():
            for value in sorted(set(group_values[local_mask].tolist())):
                group_mask = local_mask & (group_values == value)
                y_group = labels_by_origin[origin][group_mask]
                n = len(y_group)
                positive = int(np.sum(y_group == 1))
                negative = int(np.sum(y_group == 0))
                context = {"view": view_name, "origin": origin, "group": group_name,
                           "group_value": str(value), "n": n, "positive": positive, "negative": negative,
                           "minimum_support": {"n": MIN_GROUP_N, "each_class": MIN_GROUP_PER_CLASS}}
                if n < MIN_GROUP_N or positive < MIN_GROUP_PER_CLASS or negative < MIN_GROUP_PER_CLASS:
                    checks.append({**context, "status": "UNKNOWN_UNSUPPORTED",
                                   "aligned_metrics": None, "a02_metrics": None, "permuted_metrics": None,
                                   "gains": None})
                    continue
                scores = {
                    "aligned": binary_metrics(y_group, origin_item["arms"]["aligned"]["candidate_probability"][group_mask]),
                    "a02": binary_metrics(y_group, source["cache"]["prediction"][group_mask]),
                    "permuted": binary_metrics(y_group, origin_item["arms"]["permuted"]["candidate_probability"][group_mask]),
                }
                gains = {}
                for comparator in ("a02", "permuted"):
                    gains[comparator] = {
                        "macro_f1": scores["aligned"]["macro_f1"] - scores[comparator]["macro_f1"],
                        "positive_recall": scores["aligned"]["positive_recall"] - scores[comparator]["positive_recall"],
                    }
                passed = all(gains[c]["macro_f1"] + GATE_TOLERANCE >= -0.01
                             and gains[c]["positive_recall"] + GATE_TOLERANCE >= -0.02
                             for c in ("a02", "permuted"))
                checks.append({**context, "status": "PASS" if passed else "FAIL",
                               "aligned_metrics": scores["aligned"], "a02_metrics": scores["a02"],
                               "permuted_metrics": scores["permuted"], "gains": gains,
                               "thresholds": {"macro_f1_min_gain": -0.01, "positive_recall_min_gain": -0.02}})
    return checks


def _pooled_subgroup_checks(origin_data: Mapping[int, Any], labels_by_origin: Mapping[int, np.ndarray],
                            masks: Mapping[int, np.ndarray]) -> list[dict[str, Any]]:
    """Evaluate pooled subgroup guards without mixing origin-specific feature maps."""
    origins = (67, 75)
    group_values = {origin: _group_values(origin_data[origin]["source"]) for origin in origins}
    pooled_groups = {}
    for name in group_values[67]:
        values = []
        for origin in origins:
            origin_values = group_values[origin][name][masks[origin]]
            # Quartile boundaries are fit-specific. Keep the origin in the group
            # key instead of pooling unlike Q1/Q2 definitions across fits.
            if name == "text_load_quartile":
                origin_values = np.asarray([f"E{origin}_{value}" for value in origin_values], dtype="<U8")
            values.append(origin_values)
        pooled_groups[name] = np.concatenate(values)
    pooled_labels = np.concatenate([labels_by_origin[origin][masks[origin]] for origin in origins])

    def pooled_probability(arm: str) -> np.ndarray:
        values = []
        for origin in origins:
            item = origin_data[origin]
            if arm in {"aligned", "permuted"}:
                probability = item["arms"][arm]["candidate_probability"]
            else:
                cache = item["source"]["cache"]
                probability = cache["prediction" if arm == "a02" else "baseline_prediction"]
            values.append(probability[masks[origin]])
        return np.concatenate(values)

    probabilities = {arm: pooled_probability(arm) for arm in ("aligned", "a02", "permuted")}
    checks = []
    for group_name, values in pooled_groups.items():
        for value in sorted(set(values.tolist())):
            group_mask = values == value
            y_group = pooled_labels[group_mask]
            n = len(y_group)
            positive = int(np.sum(y_group == 1))
            negative = int(np.sum(y_group == 0))
            context = {"view": "POOLED_E67_W4_PLUS_E75_W6", "origin": "67+75", "group": group_name,
                       "group_value": str(value), "n": n, "positive": positive, "negative": negative,
                       "minimum_support": {"n": MIN_GROUP_N, "each_class": MIN_GROUP_PER_CLASS}}
            if n < MIN_GROUP_N or positive < MIN_GROUP_PER_CLASS or negative < MIN_GROUP_PER_CLASS:
                checks.append({**context, "status": "UNKNOWN_UNSUPPORTED", "aligned_metrics": None,
                               "a02_metrics": None, "permuted_metrics": None, "gains": None})
                continue
            scores = {arm: binary_metrics(y_group, probability[group_mask])
                      for arm, probability in probabilities.items()}
            gains = {comparator: {
                "macro_f1": scores["aligned"]["macro_f1"] - scores[comparator]["macro_f1"],
                "positive_recall": scores["aligned"]["positive_recall"] - scores[comparator]["positive_recall"],
            } for comparator in ("a02", "permuted")}
            passed = all(gains[c]["macro_f1"] + GATE_TOLERANCE >= -0.01
                         and gains[c]["positive_recall"] + GATE_TOLERANCE >= -0.02
                         for c in ("a02", "permuted"))
            checks.append({**context, "status": "PASS" if passed else "FAIL",
                           "aligned_metrics": scores["aligned"], "a02_metrics": scores["a02"],
                           "permuted_metrics": scores["permuted"], "gains": gains,
                           "thresholds": {"macro_f1_min_gain": -0.01, "positive_recall_min_gain": -0.02}})
    return checks


def score_predictions() -> dict[str, Any]:
    frozen = load_freeze()
    if not PRELABEL_PATH.is_file():
        raise CGeomError("Refusing to read query labels before independent prelabel verification")
    frame = positional.load_features()
    labels_e67, access = _read_query_labels_after_prelabel(frame, frozen)
    datasets = _load_all_outputs(frozen, frame)
    e67_rows = datasets[67]["source"]["score_rows"]
    e75_rows = datasets[75]["source"]["score_rows"]
    if not np.array_equal(e75_rows, e67_rows[np.isin(e67_rows, e75_rows)]):
        raise CGeomError("E75 W6 rows are not an exact subset of E67 H12 rows")
    lookup_e67 = {int(row): index for index, row in enumerate(e67_rows)}
    idx75_in_67 = np.asarray([lookup_e67[int(row)] for row in e75_rows], dtype=np.int64)
    if not np.array_equal(e67_rows[idx75_in_67], e75_rows):
        raise CGeomError("E67/E75 row map is not exact")
    y_e67 = labels_e67
    y75 = y_e67[idx75_in_67]
    labels_by_origin = {67: y_e67, 75: y75}
    view_specs: dict[str, dict[str, Any]] = {}
    origin_metrics: dict[str, Any] = {}
    for origin in ORIGINS:
        source = datasets[origin]["source"]
        y = labels_by_origin[origin]
        row_metrics = {}
        arm_probabilities = {
            "aligned": datasets[origin]["arms"]["aligned"]["candidate_probability"],
            "permuted": datasets[origin]["arms"]["permuted"]["candidate_probability"],
            "a02": source["cache"]["prediction"],
            "sparse25_baseline": source["cache"]["baseline_prediction"],
        }
        for view_name, mask in _view_indices(source["score_cycles"], origin).items():
            if not mask.any():
                raise CGeomError(f"CGEOM required view has no rows: {view_name}")
            row_metrics[view_name] = _view_metrics(y[mask], {name: prob[mask] for name, prob in arm_probabilities.items()})
            view_specs[view_name] = {"origin": origin, "mask": mask}
        origin_metrics[str(origin)] = {
            "fit_prefix_rows": source["fit_prefix_rows"], "query_rows": source["query_rows"],
            "present_rows": source["present_rows"], "missing_rows": source["missing_rows"],
            "metrics": row_metrics,
            "per_cycle": {},
        }
        for cycle in sorted(set(source["score_cycles"].tolist())):
            mask = source["score_cycles"] == cycle
            if not mask.any():
                continue
            origin_metrics[str(origin)]["per_cycle"][str(int(cycle))] = _view_metrics(
                y[mask], {name: prob[mask] for name, prob in arm_probabilities.items()})
    # Explicit unique pooled view: E67 W4 plus E75 W6 only.
    pooled_e67_mask = (datasets[67]["source"]["score_cycles"] >= 68) & (datasets[67]["source"]["score_cycles"] <= 71)
    pooled_e75_mask = (datasets[75]["source"]["score_cycles"] >= 76) & (datasets[75]["source"]["score_cycles"] <= 79)
    pooled_y = np.concatenate((y_e67[pooled_e67_mask], y75[pooled_e75_mask]))
    pooled_metrics = {}
    for arm in ("aligned", "permuted", "a02", "sparse25_baseline"):
        if arm in {"aligned", "permuted"}:
            p67 = datasets[67]["arms"][arm]["candidate_probability"][pooled_e67_mask]
            p75 = datasets[75]["arms"][arm]["candidate_probability"][pooled_e75_mask]
        else:
            p67 = datasets[67]["source"]["cache"]["prediction" if arm == "a02" else "baseline_prediction"][pooled_e67_mask]
            p75 = datasets[75]["source"]["cache"]["prediction" if arm == "a02" else "baseline_prediction"][pooled_e75_mask]
        pooled_metrics[arm] = binary_metrics(pooled_y, np.concatenate((p67, p75)))
    view_specs["POOLED_E67_W4_PLUS_E75_W6"] = {
        "origin": 67, "mask": pooled_e67_mask,
        "note": "Concatenated with disjoint E75 W6; per-origin feature geometry retained.",
    }
    gate_checks = []
    e67m = origin_metrics["67"]["metrics"]
    e75m = origin_metrics["75"]["metrics"]["E75_W6"]
    gate_checks.extend([
        _gate_record("E67_H12_aligned_vs_A02", e67m["E67_H12"]["aligned"]["macro_f1"] - e67m["E67_H12"]["a02"]["macro_f1"], 0.0005,
                     context={"view": "E67_H12", "comparator": "a02"}),
        _gate_record("E75_W6_aligned_vs_A02", e75m["aligned"]["macro_f1"] - e75m["a02"]["macro_f1"], 0.0005,
                     context={"view": "E75_W6", "comparator": "a02"}),
        _gate_record("E67_H12_aligned_vs_permuted", e67m["E67_H12"]["aligned"]["macro_f1"] - e67m["E67_H12"]["permuted"]["macro_f1"], 0.0002,
                     context={"view": "E67_H12", "comparator": "permuted"}),
        _gate_record("E75_W6_aligned_vs_permuted", e75m["aligned"]["macro_f1"] - e75m["permuted"]["macro_f1"], 0.0,
                     context={"view": "E75_W6", "comparator": "permuted"}),
    ])
    for view_name in ("E67_W4", "E67_W5", "E67_W6"):
        for comparator in ("a02", "permuted"):
            gain = e67m[view_name]["aligned"]["macro_f1"] - e67m[view_name][comparator]["macro_f1"]
            gate_checks.append(_gate_record(f"{view_name}_harm_vs_{comparator}", gain, -0.005,
                                            context={"view": view_name, "comparator": comparator}))
    subgroup_views: dict[str, Any] = {}
    for name, spec in view_specs.items():
        if name.startswith("POOLED"):
            # The pooled view uses its explicitly de-duplicated origin-specific halves.
            continue
        subgroup_views[name] = spec
    subgroup_checks = _subgroup_checks(datasets, labels_by_origin, subgroup_views)
    subgroup_checks.extend(_pooled_subgroup_checks(datasets, labels_by_origin,
        {67: pooled_e67_mask, 75: pooled_e75_mask}))
    gate_checks.extend({"gate": f"subgroup:{x['view']}:{x['group']}:{x['group_value']}",
                        "status": x["status"], "view": x["view"], "group": x["group"],
                        "group_value": x["group_value"], "support": {"n": x["n"], "positive": x["positive"], "negative": x["negative"]},
                        "gains": x["gains"], "thresholds": x.get("thresholds")}
                       for x in subgroup_checks)
    false = sum(x["status"] == "FAIL" for x in gate_checks)
    unknown = sum(x["status"] == "UNKNOWN_UNSUPPORTED" for x in gate_checks)
    status = "CLOSED_GATE_FAIL" if false else ("NOT_ADVANCE_UNSUPPORTED" if unknown else "ADVANCE_TO_SOL_REVIEW")
    results = {
        "schema_version": 1, "pilot_id": PILOT_ID, "card_id": CARD_ID,
        "status": status, "freeze_signature": frozen["freeze_signature"],
        "source_commit": frozen["source_commit"], "input_sha256": TRAIN_SHA256,
        "recipe_sha256": frozen["recipe_sha256"], "prelabel_verification_sha256": sha_file(PRELABEL_PATH),
        "query_label_access_receipt_sha256": sha_file(RUN_ROOT / "QUERY_LABEL_ACCESS.json"),
        "score_rows": {"E67": len(y_e67), "E75_W6": len(y75), "pooled_unique": len(pooled_y)},
        "epsilon_convention": {"threshold": ">=0.50", "macro_f1_zero_division": 0.0,
                               "precision_recall_zero_denominator": 0.0, "log_loss_clip": EPSILON,
                               "brier_unclipped": True},
        "views": {
            **{name: {"origin": spec["origin"], "rows": int(np.sum(spec["mask"])),
                      "unique_denominator_note": "same origin only"}
               for name, spec in view_specs.items() if not name.startswith("POOLED")},
            "POOLED_E67_W4_PLUS_E75_W6": {"rows": int(len(pooled_y)), "E67_W4_rows": int(pooled_e67_mask.sum()),
                                          "E75_W6_rows": int(pooled_e75_mask.sum()),
                                          "overlap_rows": 0, "unique_denominator_note": "origin-specific map; disjoint cycles"},
        },
        "origins": origin_metrics,
        "pooled_unique_metrics": pooled_metrics,
        "gate_checks": gate_checks,
        "subgroup_checks": subgroup_checks,
        "gate_counts": {"total": len(gate_checks), "fail": false, "unknown": unknown,
                        "pass": sum(x["status"] == "PASS" for x in gate_checks)},
        "source_provenance": {
            str(origin): {"fit_id": {arm: f"CGEOM_V1_E{origin}_{arm.upper()}" for arm in ARMS},
                          "artifact_sha256": {arm: sha_file(_unit_paths(f"CGEOM_V1_E{origin}_{arm.upper()}")[0]) for arm in ARMS},
                          "receipt_sha256": {arm: sha_file(_unit_paths(f"CGEOM_V1_E{origin}_{arm.upper()}")[1]) for arm in ARMS}}
            for origin in ORIGINS
        },
        "budget": {"ledger_sha256": sha_file(LEDGER), "snapshot": _budget_snapshot(_ledger_events()),
                   "new_fit_ids": list(FIT_IDS), "new_fit_count": 4, "retry_count": 0},
        "limits": ["diagnostic pilot only", "development labels reused", "no Test/Kaggle/full-Train refit",
                   "no automatic continuation or champion change"],
        "scored_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(RESULT_PATH, results)
    return results


def publish_report() -> dict[str, Any]:
    results = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    scored = json.loads(SCORED_VERIFICATION_PATH.read_text(encoding="utf-8"))
    if scored.get("status") != "PASS_INDEPENDENT_METRICS_AND_GATES" or scored.get("result_sha256") != sha_file(RESULT_PATH):
        raise CGeomError("Independent scored verification is not PASS or does not bind the result")
    if scored.get("freeze_signature") != results.get("freeze_signature"):
        raise CGeomError("Independent score receipt and result freeze signatures differ")
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    paths = {
        "RESULTS.json": REPORT_ROOT / "RESULTS.json",
        "SCORED_VERIFICATION.json": REPORT_ROOT / "SCORED_VERIFICATION.json",
        "PRELABEL_VERIFICATION.json": REPORT_ROOT / "PRELABEL_VERIFICATION.json",
    }
    for key, path in paths.items():
        src = RUN_ROOT / key
        if key == "SCORED_VERIFICATION.json":
            src = SCORED_VERIFICATION_PATH
        if key == "RESULTS.json":
            src = RESULT_PATH
        if not src.is_file():
            raise CGeomError(f"Report source artifact missing: {src}")
        write_create_once(path, src.read_bytes())
    ledger = _ledger_events()
    completed = [e for e in ledger if e.get("event") == "CARD_COMPLETED" and e.get("card_id") == CARD_ID]
    completion = {
        "status": results["status"], "card_id": CARD_ID, "freeze_signature": results["freeze_signature"],
        "result_sha256": sha_file(RESULT_PATH), "scored_verification_sha256": sha_file(SCORED_VERIFICATION_PATH),
        "prelabel_verification_sha256": sha_file(PRELABEL_PATH),
        "unit_artifact_sha256": {fit_id: sha_file(_unit_paths(fit_id)[0]) for fit_id in FIT_IDS},
        "ledger_sha256_before_completion": sha_file(LEDGER), "ledger_snapshot_before_completion": _budget_snapshot(ledger),
        "new_fits": list(FIT_IDS), "retry_count": 0, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    completion_path = RUN_ROOT / "COMPLETION_RECEIPT.json"
    atomic_json(completion_path, completion)
    completion_report_path = REPORT_ROOT / "COMPLETION_RECEIPT.json"
    write_create_once(completion_report_path, completion_path.read_bytes())
    md_path = REPORT_ROOT / "RESULTS.md"
    markdown = render_markdown(results, scored, completion)
    write_create_once(md_path, markdown.encode("utf-8"))
    return {"status": results["status"], "report_paths": {k: relpath(v) for k, v in paths.items()} |
            {"COMPLETION_RECEIPT.json": relpath(completion_report_path), "RESULTS.md": relpath(md_path)}}


def render_markdown(results: Mapping[str, Any], scored: Mapping[str, Any], completion: Mapping[str, Any]) -> str:
    lines = [
        "# CGEOM_V1 — coefficient-informed A02 geometry pilot", "",
        f"**Status:** `{results['status']}`  ",
        f"**Source commit:** `{results['source_commit']}`  ",
        f"**Freeze:** `{results['freeze_signature']}`  ",
        f"**Independent scored verification:** `{scored['status']}`", "",
        "## Question and frozen method", "",
        "CGEOM_V1 tests whether same-origin C2_B LR token-coefficient magnitudes can improve A02 donor retrieval when lexical correction mass is held at the original per-query A02 mass. The permuted arm uses the same multiplier multiset permuted with seed 42. This is an alignment diagnostic, not a formal randomization null or a causal claim.", "",
        "For each A02 token j, `a_j = 1 + |beta_j| / (1 + |beta_j|)` and `d_j = sqrt(a_j)`. C2_B and A02 vocabularies and IDF arrays were checked for exact identity; coefficients were selected by token string from the C2_B logistic text block. Weighted TF-IDF rows were L2-normalized and top-32 positive cosine donors were searched over the full original bank with source-row-stable ties. The selected squared-cosine weights were rescaled to sum to original A02 `M0(q)`. The correction retained shrinkage 4, clipping ±0.05, the original Sparse25 baseline, 0.50 threshold, and missing/zero-neighbor fallback.", "",
        "## Primary and history-sensitive results", "",
        "| View | n | Arm | Macro F1 | Positive recall | Precision | ROC AUC | AP | Log loss (clip 1e-15) | Brier | Confusion [[TN,FP],[FN,TP]] |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for view_name in ("E67_H12", "E67_W4", "E67_W5", "E67_W6"):
        metrics = results["origins"]["67"]["metrics"][view_name]
        for arm in ("aligned", "permuted", "a02", "sparse25_baseline"):
            m = metrics[arm]
            lines.append(f"| {view_name} | {m['n']} | {arm} | {m['macro_f1']:.6f} | {m['positive_recall']:.6f} | {m['precision']:.6f} | {m['roc_auc']:.6f} | {m['average_precision']:.6f} | {m['log_loss_clip_1e-15']:.6f} | {m['brier']:.6f} | {m['confusion_matrix']} |")
    for arm in ("aligned", "permuted", "a02", "sparse25_baseline"):
        m = results["origins"]["75"]["metrics"]["E75_W6"][arm]
        lines.append(f"| E75_W6 | {m['n']} | {arm} | {m['macro_f1']:.6f} | {m['positive_recall']:.6f} | {m['precision']:.6f} | {m['roc_auc']:.6f} | {m['average_precision']:.6f} | {m['log_loss_clip_1e-15']:.6f} | {m['brier']:.6f} | {m['confusion_matrix']} |")
    pooled = results["pooled_unique_metrics"]
    lines.extend(["", "### Pooled unique W4 + W6", "", f"Denominator: `{results['score_rows']['pooled_unique']}` unique rows = E67 W4 ({results['views']['POOLED_E67_W4_PLUS_E75_W6']['E67_W4_rows']}) + E75 W6 ({results['views']['POOLED_E67_W4_PLUS_E75_W6']['E75_W6_rows']}); overlap = 0. Origin-specific geometry is kept separate before concatenation.", "", "| Arm | Macro F1 | Positive recall | Precision | ROC AUC | AP | Log loss | Brier |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for arm in ("aligned", "permuted", "a02", "sparse25_baseline"):
        m = pooled[arm]
        lines.append(f"| {arm} | {m['macro_f1']:.6f} | {m['positive_recall']:.6f} | {m['precision']:.6f} | {m['roc_auc']:.6f} | {m['average_precision']:.6f} | {m['log_loss_clip_1e-15']:.6f} | {m['brier']:.6f} |")
    lines.extend(["", "## Frozen gates", "", f"Overall status: `{results['status']}`; gate counts: `{results['gate_counts']}`.", "", "| Gate | View | Comparator | Gain | Minimum | Status |", "|---|---|---|---:|---:|---|"])
    for check in results["gate_checks"]:
        lines.append(f"| {check['gate']} | {check.get('view','')} | {check.get('comparator','')} | {check.get('gain',float('nan')):.6f} | {check.get('required_minimum_gain',float('nan')):.6f} | {check['status']} |")
    lines.extend(["", "## Subgroup guards", "", "Support rule: n ≥ 200 and at least 30 examples per class. Unsupported cells are reported as UNKNOWN, without an estimated harm.", "", "| View | Group | Value | n | Positive/negative | Δ Macro F1 vs A02 | Δ Recall vs A02 | Δ Macro F1 vs permuted | Δ Recall vs permuted | Status |", "|---|---|---|---:|---:|---:|---:|---:|---:|---|"])
    for check in results["subgroup_checks"]:
        gains = check.get("gains")
        if gains is None:
            vals = ["UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN"]
        else:
            vals = [f"{gains['a02']['macro_f1']:.6f}", f"{gains['a02']['positive_recall']:.6f}",
                    f"{gains['permuted']['macro_f1']:.6f}", f"{gains['permuted']['positive_recall']:.6f}"]
        lines.append(f"| {check['view']} | {check['group']} | {check['group_value']} | {check['n']} | {check['positive']}/{check['negative']} | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} | {check['status']} |")
    lines.extend(["", "## Per-cycle effects", "", "Positive means aligned exceeds the comparator on Macro F1. All cycle denominators are included.", "", "| Origin | Cycle | n | Δ vs A02 | Δ vs permuted |", "|---:|---:|---:|---:|---:|"])
    for origin in (67, 75):
        for cycle, metrics in results["origins"][str(origin)]["per_cycle"].items():
            delta_a02 = metrics["aligned"]["macro_f1"] - metrics["a02"]["macro_f1"]
            delta_perm = metrics["aligned"]["macro_f1"] - metrics["permuted"]["macro_f1"]
            lines.append(f"| {origin} | {cycle} | {metrics['aligned']['n']} | {delta_a02:+.6f} | {delta_perm:+.6f} |")
    lines.extend(["", "## Lineage, accounting, and limitations", "",
        f"- Official Classification Train SHA-256: `{TRAIN_SHA256}`; query labels were opened only after prelabel PASS (receipt `{results['query_label_access_receipt_sha256']}`).",
        f"- Four reserved/saved indexed memory predictors: `{', '.join(FIT_IDS)}`. No classifier, vectorizer, or preprocessing refits; retries: 0.",
        f"- Budget ledger SHA-256: `{results['budget']['ledger_sha256']}`; snapshot: `{results['budget']['snapshot']}`.",
        f"- Independent score verifier SHA-256: `{completion['scored_verification_sha256']}`; completion result SHA-256: `{completion['result_sha256']}`.",
        "- All internal rows are reused development labels; E75 W6 overlaps E67 late rows and is history sensitivity rather than independent evidence.",
        "- Permutation preserves the coefficient-derived multiplier multiset, not every association with token frequency; it is not a randomization p-value.",
        "- Passing the gates means only `ADVANCE_TO_SOL_REVIEW`; no continuation, champion change, Test inference, full-Train refit, Kaggle use, or submission follows automatically.",
        "- Any subgroup below support is UNKNOWN. No subgroup estimate is fabricated.", "",
        "See `RESULTS.json`, `PRELABEL_VERIFICATION.json`, `SCORED_VERIFICATION.json`, and `COMPLETION_RECEIPT.json` for machine-readable details.", ""])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--synthetic-qa", action="store_true")
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--reserve", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--score", action="store_true")
    group.add_argument("--publish-report", action="store_true")
    args = parser.parse_args(argv)
    if args.synthetic_qa:
        result = create_synthetic_qa()
    elif args.preflight:
        result = create_target_free_preflight()
    elif args.freeze:
        result = create_freeze()
    elif args.reserve:
        result = reserve_pilot(load_freeze())
    elif args.run:
        result = run_predictions()
    elif args.score:
        result = score_predictions()
    else:
        result = publish_report()
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
