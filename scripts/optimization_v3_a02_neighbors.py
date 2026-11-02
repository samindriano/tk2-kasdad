"""Four-fit A02 lexical-neighbor residual pilot using Classification Train only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.classification_a02_neighbors import (  # noqa: E402
    ResidualNeighborPredictor,
    fit_text_space,
)
from src.models.optimization_v3_budget_amendment import reserve_with_effective_caps  # noqa: E402
from src.models.research_next_budget import finish_fit, read_events, start_fit  # noqa: E402
from src.models.research_next_contract import canonical_sha256, file_sha256  # noqa: E402


TRAIN_PATH = ROOT / "data/raw/classification/Train.csv"
PROTOCOL_PATH = ROOT / "experiments/optimization_v3/c19_a02_v1/A02/protocol.json"
FREEZE_PATH = ROOT / "experiments/optimization_v3/c19_a02_v1/A02/freeze.json"
LEDGER_PATH = ROOT / "experiments/runs/optimization_v3_research_next_v1/budget.jsonl"
RUN_ROOT = ROOT / "experiments/runs/optimization_v3_a02_neighbors_v1"
REPORT_ROOT = ROOT / "reports/optimization_v3/a02_neighbors_v1"
SYNTHETIC_QA_PATH = REPORT_ROOT / "synthetic_qa.json"

CACHE_PATHS = {
    55: ROOT / "experiments/runs/optimization_v3_batch1/C16/C16_fit55.npz",
    59: ROOT / "experiments/runs/optimization_v3_final_continuation/C16/C16_fit59.npz",
    63: ROOT / "experiments/runs/optimization_v3_final_continuation/C16/C16_fit63.npz",
    67: ROOT / "experiments/runs/optimization_v3_batch1/C16/C16_fit67.npz",
    71: ROOT / "experiments/runs/optimization_v3_final_continuation/C16/C16_fit71.npz",
    75: ROOT / "experiments/runs/optimization_v3_final_continuation/C16/C16_fit75.npz",
}

EXPECTED_TRAIN_SHA256 = "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c"
EXPECTED_ROWS = 50_608
THREADS = 2
JOB_CAP_BYTES = 6 * 1024**3
MIN_FREE_RAM_BYTES = 4 * 1024**3
GLOBAL_CAP = 60
CARD_ID = "CLASSIFICATION_A02_NEIGHBOR_RESIDUAL"
STAGE = "A02_FOUR_FIT_PILOT"
FIT_IDS = {
    (67, "observed"): "A02_E67_OBSERVED",
    (67, "shuffled"): "A02_E67_SHUFFLED",
    (75, "observed"): "A02_E75_OBSERVED",
    (75, "shuffled"): "A02_E75_SHUFFLED",
}


def _sha(path: Path) -> str:
    return file_sha256(path)


def _write_once(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"A02 immutable artifact differs; refusing overwrite: {path}")
        return
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         allow_nan=False, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_frozen() -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = _read_json(PROTOCOL_PATH)
    freeze = _read_json(FREEZE_PATH)
    if canonical_sha256(protocol) != freeze.get("protocol_sha256"):
        raise RuntimeError("A02 protocol hash differs from its frozen identity")
    identity = freeze.get("identity")
    if not isinstance(identity, dict) or canonical_sha256(identity) != freeze.get("freeze_signature"):
        raise RuntimeError("A02 freeze signature is invalid")
    for relative, expected in freeze["source_sha256"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"A02 source hash mismatch: {relative}")
    for relative, expected in freeze["input_sha256"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"A02 input hash mismatch: {relative}")
    return protocol, freeze


def _load_train(*, include_target: bool = True) -> pd.DataFrame:
    columns = ["record_id", "cycle_code", "signal_stream", "entry_friction", "text_load"]
    if include_target:
        columns.append("artifact_signal")
    frame = pd.read_csv(TRAIN_PATH, usecols=columns, dtype={"record_id": str})
    if len(frame) != EXPECTED_ROWS or frame["record_id"].isna().any() or frame["record_id"].duplicated().any():
        raise RuntimeError("A02 official Classification Train identity/row count is invalid")
    return frame


def _load_cache(origin: int, train: pd.DataFrame) -> dict[str, np.ndarray]:
    path = CACHE_PATHS[origin]
    sidecar_path = path.with_suffix(".json")
    sidecar = _read_json(sidecar_path)
    if sidecar.get("npz_sha256") != _sha(path):
        raise RuntimeError(f"A02 cache sidecar checksum differs for cutoff {origin}")
    if sidecar.get("identity_sha256") != _canonical_digest(sidecar.get("identity")):
        raise RuntimeError(f"A02 cache identity signature differs for cutoff {origin}")
    if sidecar.get("integrity_sha256") != _canonical_digest(
        {key: value for key, value in sidecar.items() if key != "integrity_sha256"}
    ):
        raise RuntimeError(f"A02 cache integrity signature differs for cutoff {origin}")
    identity = sidecar.get("identity", {})
    if identity.get("card_id") != "C16" or int(identity.get("cutoff", -1)) != origin:
        raise RuntimeError(f"A02 cache identity does not match C16 cutoff {origin}")
    metadata = sidecar.get("metadata", {})
    fit_metadata = metadata.get("fit_metadata", {})
    state = metadata.get("state", {})
    if (metadata.get("card_id") != "C16" or metadata.get("fit_predictor_count") != 1
            or fit_metadata.get("score_labels_passed_to_adapter") is not False
            or state.get("test_data_used") is not False
            or state.get("validation_labels_passed_to_fit") is not False):
        raise RuntimeError(f"A02 cache {origin} does not attest the historical C16 score/source boundary")
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "score_rows", "score_record_ids", "baseline_prediction",
            "control__baseline_sparse_raw", "control__c2b_probability",
            "control__cb_probability", "control__stream_missing", "control__control_rows",
            "audit__stream_missing",
        }
        if not required.issubset(archive.files):
            raise RuntimeError(f"A02 C16 cache {origin} is missing required baseline route arrays")
        data = {key: np.asarray(archive[key]).copy() for key in required}

    rows = data["score_rows"]
    if (rows.ndim != 1 or rows.dtype.kind not in "iu" or len(np.unique(rows)) != len(rows)
            or np.any(rows < 0) or np.any(rows >= len(train)) or np.any(np.diff(rows) <= 0)):
        raise RuntimeError(f"A02 C16 rows are invalid at cutoff {origin}")
    if data["score_record_ids"].astype(str).tolist() != train["record_id"].to_numpy()[rows].tolist():
        raise RuntimeError(f"A02 C16 record IDs do not join to official Train at cutoff {origin}")
    expected_mask = train["signal_stream"].fillna("").astype(str).str.strip().eq("").to_numpy()[rows]
    for name in ("control__stream_missing", "audit__stream_missing"):
        if not np.array_equal(data[name], expected_mask):
            raise RuntimeError(f"A02 C16 stream mask does not reproduce at cutoff {origin}")
    if not np.array_equal(data["control__control_rows"], rows):
        raise RuntimeError(f"A02 C16 control row positions differ at cutoff {origin}")

    p_base = data["baseline_prediction"].astype(np.float64)
    p_cb = data["control__cb_probability"].astype(np.float64)
    p_c2b = data["control__c2b_probability"].astype(np.float64)
    p_sparse = data["control__baseline_sparse_raw"].astype(np.float64)
    for name, array in (("baseline", p_base), ("CB", p_cb), ("C2_B", p_c2b), ("sparse25", p_sparse)):
        if (array.shape != rows.shape or not np.isfinite(array).all()
                or np.any((array < 0.0) | (array > 1.0))):
            raise RuntimeError(f"A02 C16 {name} probabilities are invalid at cutoff {origin}")
    reconstruction = np.where(expected_mask, p_cb, 0.75 * p_c2b + 0.25 * p_sparse)
    if not np.array_equal(reconstruction, p_base):
        raise RuntimeError(f"A02 C16 baseline_prediction is not bit-exact sparse25 at cutoff {origin}")
    cycles = pd.to_numeric(train["cycle_code"], errors="raise").to_numpy(dtype=np.int64)[rows]
    if not np.array_equal(np.unique(cycles), np.arange(origin + 1, min(origin + 12, 79) + 1)):
        raise RuntimeError(f"A02 C16 cache {origin} does not cover its declared complete score cycles")
    for cycle in np.unique(cycles):
        if np.count_nonzero(cycles == cycle) != np.count_nonzero(train["cycle_code"].to_numpy() == cycle):
            raise RuntimeError(f"A02 C16 cache {origin} has incomplete cycle {cycle}")
    data["cycles"] = cycles
    data["baseline_prediction"] = p_base
    data["stream_missing"] = expected_mask
    return data


def _cache_subset(cache: dict[str, np.ndarray], train: pd.DataFrame,
                  start_cycle: int, stop_cycle: int) -> dict[str, np.ndarray]:
    mask = (cache["cycles"] >= start_cycle) & (cache["cycles"] <= stop_cycle)
    rows = cache["score_rows"][mask]
    cycles = cache["cycles"][mask]
    expected = np.flatnonzero(train["cycle_code"].between(start_cycle, stop_cycle).to_numpy())
    if not np.array_equal(rows, expected):
        raise RuntimeError(f"A02 OOF cache block {start_cycle}-{stop_cycle} is not complete and row ordered")
    return {
        "rows": rows.astype(np.int64, copy=False),
        "ids": cache["score_record_ids"][mask].astype(str),
        "cycles": cycles.astype(np.int64, copy=False),
        "baseline": cache["baseline_prediction"][mask].astype(np.float64, copy=False),
        "missing": cache["stream_missing"][mask].astype(bool, copy=False),
    }


def _build_bank(origin: int, caches: dict[int, dict[str, np.ndarray]], train: pd.DataFrame) -> dict[str, np.ndarray]:
    if origin == 67:
        blocks = [(55, 56, 59), (59, 60, 63), (63, 64, 67)]
    elif origin == 75:
        blocks = [(55, 56, 59), (59, 60, 63), (63, 64, 67),
                  (67, 68, 71), (71, 72, 75)]
    else:
        raise ValueError("A02 supports only frozen origins 67 and 75")
    parts = [_cache_subset(caches[source], train, first, last) for source, first, last in blocks]
    rows = np.concatenate([part["rows"] for part in parts])
    order = np.argsort(rows, kind="stable")
    result = {
        key: np.concatenate([part[key] for part in parts])[order]
        for key in ("rows", "ids", "cycles", "baseline", "missing")
    }
    if len(np.unique(result["rows"])) != len(result["rows"]):
        raise RuntimeError("A02 OOF banks contain overlapping score rows")
    expected_rows = np.flatnonzero(train["cycle_code"].between(56, origin).to_numpy())
    if not np.array_equal(result["rows"], expected_rows):
        raise RuntimeError(f"A02 origin {origin} bank does not cover exact prior four-cycle blocks")
    if origin == 67 and (len(result["rows"]) != 17_685 or int((~result["missing"]).sum()) != 16_482):
        raise RuntimeError("A02 E67 bank support differs from its frozen count")
    if origin == 75 and (len(result["rows"]) != 30_097 or int((~result["missing"]).sum()) != 28_045):
        raise RuntimeError("A02 E75 bank support differs from its frozen count")
    return result


class _PeakRSS:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self.peak = 0
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "_PeakRSS":
        process = psutil.Process()
        self.peak = process.memory_info().rss

        def monitor() -> None:
            while not self._stop.wait(0.2):
                try:
                    self.peak = max(self.peak, process.memory_info().rss)
                except psutil.Error:
                    return

        self._thread = threading.Thread(target=monitor, daemon=True)
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


def _text_load_groups(train: pd.DataFrame, origin: int, rows: np.ndarray) -> tuple[np.ndarray, list[float]]:
    cycle = train["cycle_code"].to_numpy(dtype=np.int64)
    prefix = pd.to_numeric(train.loc[cycle <= origin, "text_load"], errors="raise").to_numpy(dtype=np.float64)
    score = pd.to_numeric(train.loc[rows, "text_load"], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(prefix).all() or not np.isfinite(score).all():
        raise RuntimeError("A02 fit-defined text_load quantiles require finite official Train values")
    edges = np.quantile(prefix, [0.25, 0.50, 0.75], method="linear")
    groups = np.searchsorted(edges, score, side="right").astype(np.int8)
    return groups, [float(value) for value in edges]


def _load_audit_inputs(train: pd.DataFrame) -> tuple[dict[int, dict[str, np.ndarray]], dict[int, dict[str, np.ndarray]]]:
    caches = {cutoff: _load_cache(cutoff, train) for cutoff in sorted(CACHE_PATHS)}
    banks = {origin: _build_bank(origin, caches, train) for origin in (67, 75)}
    queries = {
        67: _cache_subset(caches[67], train, 68, 79),
        75: _cache_subset(caches[75], train, 76, 79),
    }
    expected_counts = {67: (20_903, 19_528), 75: (8_491, None)}
    for origin, query in queries.items():
        expected_rows, expected_present = expected_counts[origin]
        if (len(query["rows"]) != expected_rows
                or (expected_present is not None and int((~query["missing"]).sum()) != expected_present)):
            raise RuntimeError(f"A02 outer query support differs from frozen counts at E{origin}")
    return banks, queries


def preflight() -> dict[str, Any]:
    protocol, freeze = _load_frozen()
    if _sha(TRAIN_PATH) != EXPECTED_TRAIN_SHA256:
        raise RuntimeError("A02 official Train raw SHA-256 differs")
    train = _load_train(include_target=False)
    cycles = pd.to_numeric(train["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    if (not np.array_equal(np.unique(cycles), np.arange(47, 80, dtype=np.int64))
            or train.columns.duplicated().any()):
        raise RuntimeError("A02 Train cycle coverage/schema differs")
    banks, queries = _load_audit_inputs(train)
    if SYNTHETIC_QA_PATH.is_file():
        synthetic = _read_json(SYNTHETIC_QA_PATH)
    else:
        raise RuntimeError("A02 synthetic end-to-end and persistence/resume QA has not passed")
    if synthetic.get("status") != "PASS" or synthetic.get("fit_type") != "SYNTHETIC_ONLY":
        raise RuntimeError("A02 synthetic QA receipt is not a passing synthetic-only result")

    # Conservative task-level memory estimate: fit/prefix sparse matrices,
    # query transform, dense 128 x bank similarity block, saved neighbors and
    # a 2 GiB runtime/allocator reserve. Live RAM is rechecked before each fit.
    max_bank = max(int((~bank["missing"]).sum()) for bank in banks.values())
    max_query = max(len(query["rows"]) for query in queries.values())
    query_dense_bytes = min(128, max_query) * max_bank * 8
    from sklearn.feature_extraction.text import TfidfVectorizer
    from src.models.classification_a02_neighbors import TFIDF_PARAMETERS
    analyzer = TfidfVectorizer(**dict(TFIDF_PARAMETERS)).build_analyzer()
    all_text = train["signal_stream"].fillna("").astype(str).tolist()
    nnz_upper = sum(len(analyzer(text)) for text in all_text)
    full_sparse_upper = nnz_upper * 12 + (len(train) + 1) * 4
    estimate = 2 * full_sparse_upper + query_dense_bytes + 512 * 1024**2
    virtual = psutil.virtual_memory()
    disk = shutil.disk_usage(RUN_ROOT.parent if RUN_ROOT.parent.exists() else ROOT)
    if virtual.available < MIN_FREE_RAM_BYTES:
        raise RuntimeError(f"A02 preflight free RAM {virtual.available} is below the 4 GiB minimum")
    if estimate > JOB_CAP_BYTES:
        raise RuntimeError(f"A02 conservative task estimate {estimate} exceeds 6 GiB")
    if disk.free < 4 * 1024**3:
        raise RuntimeError("A02 preflight has less than 4 GiB free for durable models and outputs")
    if protocol.get("task") != "classification" or protocol.get("origins") != [67, 75]:
        raise RuntimeError("A02 protocol task/origin declaration differs")

    report = {
        "status": "A02_PREFLIGHT_PASS",
        "freeze_signature": freeze["freeze_signature"],
        "protocol_sha256": freeze["protocol_sha256"],
        "official_train_rows": EXPECTED_ROWS,
        "train_raw_sha256": _sha(TRAIN_PATH),
        "cycle_range": [int(cycles.min()), int(cycles.max())],
        "full_cycle_train_labels_accessed_for_scoring": False,
        "test_or_sample_paths_present_in_scope": False,
        "inputs": {
            str(cutoff): {"npz_sha256": _sha(path), "sidecar_sha256": _sha(path.with_suffix(".json"))}
            for cutoff, path in CACHE_PATHS.items()
        },
        "bank_rows": {str(origin): len(bank["rows"]) for origin, bank in banks.items()},
        "bank_present_rows": {str(origin): int((~bank["missing"]).sum()) for origin, bank in banks.items()},
        "query_rows": {str(origin): len(query["rows"]) for origin, query in queries.items()},
        "query_present_rows": {str(origin): int((~query["missing"]).sum()) for origin, query in queries.items()},
        "baseline_oof_reconstruction": "PASS_BIT_EXACT_SPARSE25_ROUTE",
        "synthetic_qa_sha256": _sha(SYNTHETIC_QA_PATH),
        "task_fit_count_authorized": 4,
        "resource_preflight": {
            "one_job_at_a_time": True,
            "threads": THREADS,
            "job_memory_cap_bytes": JOB_CAP_BYTES,
            "minimum_free_ram_bytes": MIN_FREE_RAM_BYTES,
            "available_ram_bytes": int(virtual.available),
            "max_bank_present_rows": max_bank,
            "max_query_rows": max_query,
            "tfidf_analyzer_nnz_upper_bound": int(nnz_upper),
            "full_train_tfidf_csr_bytes_upper": int(full_sparse_upper),
            "query_dense_similarity_block_bytes": int(query_dense_bytes),
            "additional_sparse_copy_and_runtime_reserve_bytes": 512 * 1024**2,
            "estimated_peak_bytes": int(estimate),
            "free_disk_bytes": int(disk.free),
            "status": "PASS",
        },
    }
    path = REPORT_ROOT / "preflight.json"
    if path.exists():
        old = _read_json(path)
        stable_keys = ("status", "freeze_signature", "protocol_sha256", "official_train_rows",
                       "train_raw_sha256", "cycle_range", "inputs", "bank_rows", "bank_present_rows",
                       "query_rows", "query_present_rows", "baseline_oof_reconstruction",
                       "synthetic_qa_sha256", "task_fit_count_authorized")
        if any(old.get(key) != report.get(key) for key in stable_keys):
            raise RuntimeError("A02 preflight stable fields changed after freeze")
        report = old
    else:
        _write_once(path, report)
    return report


def _fit_events(fit_id: str) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    events = read_events(LEDGER_PATH)
    started = [event for event in events if event.get("event") == "FIT_STARTED" and event.get("fit_id") == fit_id]
    terminal = [event for event in events if event.get("fit_id") == fit_id
                and event.get("event") in {"FIT_SAVED", "FIT_FAILED"}]
    if len(started) > 1 or len(terminal) > 1:
        raise RuntimeError(f"A02 fit ledger has duplicate events for {fit_id}")
    return started, terminal[0] if terminal else None


def _atomic_joblib(model: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    joblib.dump(model, temporary, compress=3)
    with temporary.open("rb+") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, destination)


def _fit_one(origin: int, arm: str, bank: dict[str, np.ndarray],
             text_space: Any, train: pd.DataFrame, freeze: dict[str, Any]) -> tuple[ResidualNeighborPredictor, dict[str, Any]]:
    fit_id = FIT_IDS[(origin, arm)]
    folder = RUN_ROOT / f"E{origin}" / arm
    model_path = folder / "model.joblib"
    state_path = folder / "fit_state.json"
    receipt_path = folder / "fit_receipt.json"
    started, terminal = _fit_events(fit_id)
    if terminal and terminal.get("event") == "FIT_FAILED":
        raise RuntimeError(f"A02 fit {fit_id} failed after consuming a fit; no retry is allowed")

    y = train["artifact_signal"].to_numpy(dtype=np.int8)
    bank_present = ~bank["missing"]
    bank_rows = bank["rows"][bank_present]
    bank_cycles = bank["cycles"][bank_present]
    bank_ids = bank["ids"][bank_present].astype(str)
    bank_text = train.loc[bank_rows, "signal_stream"].reset_index(drop=True)
    bank_probability = bank["baseline"][bank_present]
    bank_residual = y[bank_rows].astype(np.float64) - bank_probability

    if model_path.exists():
        model = joblib.load(model_path)
        state = model.audit_state()
        if state_path.exists() and _read_json(state_path) != state:
            raise RuntimeError(f"A02 model/state differ for {fit_id}")
        if not state_path.exists():
            if not started:
                raise RuntimeError(f"A02 model exists without a start event for {fit_id}")
            _write_once(state_path, state)
        if receipt_path.exists():
            receipt = _read_json(receipt_path)
        else:
            if not started:
                raise RuntimeError(f"A02 model exists without a start event for {fit_id}")
            receipt = {
                "fit_id": fit_id, "origin_cycle": origin, "residual_arm": arm,
                "freeze_signature": freeze["freeze_signature"], "model_sha256": _sha(model_path),
                "state_sha256": canonical_sha256(state), "bank_rows": len(bank_rows),
                "bank_present_rows": len(bank_rows), "bank_cycle_min": int(bank_cycles.min()),
                "bank_cycle_max": int(bank_cycles.max()),
                "text_prefix_rows": text_space.fit_rows,
                "text_prefix_row_positions_sha256": text_space.fit_row_positions_sha256,
                "observed_peak_rss_bytes": None,
                "persistence_recovered_from_durable_model": True, "task_estimator_fit_count": 1,
                "preprocessing_fit_population": f"official Train prefix cycles 47-{origin}; text only",
                "residual_fit_population": f"past-only OOF sparse25 residuals from four-cycle blocks through {origin}",
                "query_labels_passed_to_fit": False, "test_data_used": False,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            receipt["receipt_sha256"] = canonical_sha256(receipt)
            _write_once(receipt_path, receipt)
        receipt_check = dict(receipt)
        receipt_sig = receipt_check.pop("receipt_sha256", None)
        if (state.get("origin_cycle") != origin or state.get("residual_arm") != arm
                or state.get("bank_rows") != len(bank_rows)
                or receipt.get("model_sha256") != _sha(model_path)
                or receipt.get("state_sha256") != canonical_sha256(state)
                or receipt_sig != canonical_sha256(receipt_check)):
            raise RuntimeError(f"A02 durable model receipt failed validation for {fit_id}")
        if terminal is None:
            finish_fit(LEDGER_PATH, fit_id=fit_id, status="FIT_SAVED",
                       receipt_sha256=receipt["receipt_sha256"])
        return model, receipt

    if started or terminal is not None:
        raise RuntimeError(f"A02 fit {fit_id} has no durable model after its start; no retry is allowed")

    query_cycle = pd.to_numeric(train["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    prefix_rows = np.flatnonzero(query_cycle <= origin)
    if text_space.fit_rows != len(prefix_rows):
        raise RuntimeError("A02 text-space was not fit on the exact declared origin prefix")
    virtual = psutil.virtual_memory()
    if virtual.available < MIN_FREE_RAM_BYTES:
        raise RuntimeError(f"A02 live free RAM {virtual.available} below required 4 GiB before {fit_id}")
    if psutil.Process().memory_info().rss > JOB_CAP_BYTES:
        raise RuntimeError(f"A02 live process RSS exceeds 6 GiB before {fit_id}")

    model = ResidualNeighborPredictor(origin_cycle=origin, residual_arm=arm)
    start_fit(LEDGER_PATH, fit_id=fit_id, card_id=CARD_ID, stage=STAGE,
              cutoff=origin, component=f"TEXT_NEIGHBOR_RESIDUAL_{arm.upper()}")
    try:
        with threadpool_limits(limits=THREADS), _PeakRSS() as monitor:
            model.fit(
                text_space=text_space,
                bank_texts=bank_text,
                bank_record_ids=bank_ids,
                bank_row_positions=bank_rows,
                bank_cycles=bank_cycles,
                bank_oof_probabilities=bank_probability,
                bank_oof_residuals=bank_residual,
            )
        state = model.audit_state()
        if (state.get("origin_cycle") != origin or state.get("residual_arm") != arm
                or state.get("bank_rows") != len(bank_rows)
                or state.get("bank_row_positions_sha256") is None
                or monitor.peak > JOB_CAP_BYTES):
            raise RuntimeError(f"A02 fitted state or actual RSS cap failed for {fit_id}")
        _atomic_joblib(model, model_path)
        _write_once(state_path, state)
        receipt = {
            "fit_id": fit_id, "origin_cycle": origin, "residual_arm": arm,
            "freeze_signature": freeze["freeze_signature"], "model_sha256": _sha(model_path),
            "state_sha256": canonical_sha256(state), "bank_rows": len(bank_rows),
            "bank_present_rows": len(bank_rows), "bank_cycle_min": int(bank_cycles.min()),
            "bank_cycle_max": int(bank_cycles.max()), "text_prefix_rows": text_space.fit_rows,
            "text_prefix_row_positions_sha256": text_space.fit_row_positions_sha256,
            "observed_peak_rss_bytes": int(monitor.peak), "task_estimator_fit_count": 1,
            "preprocessing_fit_population": f"official Train prefix cycles 47-{origin}; text only",
            "residual_fit_population": f"past-only OOF sparse25 residuals from four-cycle blocks through {origin}",
            "query_labels_passed_to_fit": False, "test_data_used": False,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        _write_once(receipt_path, receipt)
        finish_fit(LEDGER_PATH, fit_id=fit_id, status="FIT_SAVED",
                   receipt_sha256=receipt["receipt_sha256"])
        return model, receipt
    except Exception as exc:
        current_started, current_terminal = _fit_events(fit_id)
        if current_started and current_terminal is None and not model_path.exists():
            finish_fit(LEDGER_PATH, fit_id=fit_id, status="FIT_FAILED", error=type(exc).__name__)
        raise


def _write_prediction_once(origin: int, arm: str, model: ResidualNeighborPredictor,
                           query: dict[str, np.ndarray], train: pd.DataFrame,
                           receipt: dict[str, Any]) -> Path:
    path = RUN_ROOT / f"E{origin}" / arm / "predictions.npz"
    sidecar_path = path.with_suffix(".json")
    if path.exists() or sidecar_path.exists():
        if not path.exists() or not sidecar_path.exists():
            raise RuntimeError(f"A02 prediction output is incomplete for E{origin}/{arm}")
        sidecar = _read_json(sidecar_path)
        if sidecar.get("npz_sha256") != _sha(path):
            raise RuntimeError(f"A02 prediction output hash differs for E{origin}/{arm}")
        return path

    rows = query["rows"]
    docs = train.loc[rows, "signal_stream"].reset_index(drop=True)
    group, edges = _text_load_groups(train, origin, rows)
    with threadpool_limits(limits=THREADS):
        output = model.predict(
            query_texts=docs,
            query_record_ids=query["ids"].astype(str),
            query_row_positions=rows,
            query_cycles=query["cycles"],
            baseline_probabilities=query["baseline"],
            missing_mask=query["missing"],
        )
    candidate = output["candidate_probability"]
    labels = (candidate >= 0.5).astype(np.int8)
    if (len(candidate) != len(rows) or not np.isfinite(candidate).all()
            or np.any((candidate < 0) | (candidate > 1))
            or not np.array_equal(candidate[query["missing"]], query["baseline"][query["missing"]])):
        raise RuntimeError("A02 prediction route failed before persistence")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            score_rows=rows.astype(np.int64),
            score_record_ids=query["ids"].astype(str).to_numpy(dtype=str),
            score_cycles=query["cycles"].astype(np.int16),
            baseline_prediction=query["baseline"].astype(np.float64),
            prediction=candidate.astype(np.float64),
            labels=labels,
            missing_mask=query["missing"].astype(bool),
            entry_friction=train.loc[rows, "entry_friction"].to_numpy(dtype=np.int8),
            text_load_group=group.astype(np.int8),
            neighbor_count=output["neighbor_count"].astype(np.int16),
            neighbor_source_rows=output["neighbor_source_rows"].astype(np.int64),
            neighbor_ids=output["neighbor_ids"].astype(str),
            neighbor_similarities=output["neighbor_similarities"].astype(np.float64),
            correction=output["correction"].astype(np.float64),
            correction_unclipped=output["correction_unclipped"].astype(np.float64),
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    sidecar = {
        "fit_id": receipt["fit_id"], "fit_receipt_sha256": receipt["receipt_sha256"],
        "origin_cycle": origin, "residual_arm": arm,
        "freeze_signature": _read_json(FREEZE_PATH)["freeze_signature"],
        "npz_sha256": _sha(path), "score_rows": len(rows),
        "score_row_positions_sha256": hashlib.sha256(rows.astype("<i8").tobytes()).hexdigest(),
        "score_record_ids_sha256": hashlib.sha256(json.dumps(query["ids"].astype(str).tolist(),
                                                               separators=(",", ":"),
                                                               ensure_ascii=False).encode("utf-8")).hexdigest(),
        "text_load_quartile_fit_prefix_edges": edges,
        "diagnostics": output["diagnostics"],
        "query_labels_passed_to_prediction": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_once(sidecar_path, sidecar)
    return path


def run() -> dict[str, Any]:
    preflight()
    _protocol, freeze = _load_frozen()
    if _sha(TRAIN_PATH) != EXPECTED_TRAIN_SHA256:
        raise RuntimeError("A02 official Train hash changed after preflight")
    train = _load_train(include_target=True)
    caches = {cutoff: _load_cache(cutoff, train) for cutoff in sorted(CACHE_PATHS)}
    banks = {origin: _build_bank(origin, caches, train) for origin in (67, 75)}
    queries = {
        67: _cache_subset(caches[67], train, 68, 79),
        75: _cache_subset(caches[75], train, 76, 79),
    }
    # TF-IDF preprocessing is shared by both arms at an origin and fitted only
    # to the declared historical feature prefix (no labels).
    cycle_all = train["cycle_code"].to_numpy(dtype=np.int64)
    text_spaces = {}
    for origin in (67, 75):
        prefix_rows = np.flatnonzero(cycle_all <= origin)
        text_space, _prefix_matrix, _prefix_missing = fit_text_space(
            train.loc[prefix_rows, "signal_stream"].reset_index(drop=True),
            prefix_rows,
            cycle_all[prefix_rows],
            origin_cycle=origin,
        )
        text_spaces[origin] = text_space

    for origin, arm in ((67, "observed"), (67, "shuffled"), (75, "observed"), (75, "shuffled")):
        model, receipt = _fit_one(origin, arm, banks[origin], text_spaces[origin], train, freeze)
        _write_prediction_once(origin, arm, model, queries[origin], train, receipt)

    verifier = __import__("scripts.optimization_v3_a02_neighbors_verify", fromlist=["verify"])
    result = verifier.verify(write_outputs=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true", help="verify frozen Train/cache inputs; no task fit")
    args = parser.parse_args()
    result = preflight() if args.preflight else run()
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
