"""C10 frozen trigram pilot using only official Classification Train.

The default command is read-only. Freeze inputs before ``--run``; never read
official Test, SampleSubmission, or Kaggle during pilot/verification.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from threadpoolctl import threadpool_limits

from scripts import optimization_v3_sparse_classification as sparse_pilot
from scripts import optimization_v3_sparse_classification_fullsix as sparse_fullsix
from scripts.optimization_v3_batch1_verify import verify_report
from src.models.classification_models import _TFIDF_PARAMS, _text_values
from src.models.classification_round2_models import stream_missing_mask
from src.models.optimization_v3_batch1 import (
    BatchIntegrityError,
    evaluation_views_for_card,
    file_sha256,
    json_sha256,
    load_queue,
    make_freeze_identity,
    run_pilot,
    write_json_exclusive,
)
from src.models.optimization_v3_batch1_c10 import (
    C10IntegrityError,
    FEATURE_NAMES,
    SPARSE_PARAMS,
    THRESHOLD,
    TRIGRAM_PARAMS,
    fit_predict,
    reconstruct_c10,
    trigram_vectorizer,
    word_vectorizer,
)
from src.models.round1_evaluation import array_digest
from src.utils.paths import task_files


CARD_ID = "C10"
QUEUE_PATH = ROOT / "experiments/optimization_v3/research_backlog/queue.json"
CARD_PATH = ROOT / "reports/optimization_v3/research_backlog/cards/C10.md"
BASELINE_MANIFEST = ROOT / "reports/optimization_v3/kaggle/v3_cls_sparse25.json"
EXPECTED_TRAIN_SHA256 = "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c"
EXPECTED_BASELINE_MANIFEST_SHA256 = "6e7c2cb50fe6d0ee4767a44c40f152b6538b933fab6cf7082659b16d791f7d1e"
EXPECTED_PRESENT_RECIPE = {
    "kind": "availability_present_blend",
    "present": [
        {"model": "C2_B", "weight": 0.75},
        {"model": "SPARSE_BINARY", "weight": 0.25},
    ],
    "missing": "CB_MISSING",
    "threshold": 0.5,
}
FIT_CUTOFFS = (55, 67)
HORIZON_BY_CUTOFF = {55: 67, 67: 79}
PRIMARY_SEGMENTS = {55: (56, 59), 67: (68, 71)}
EXPECTED_VIEWS = {
    "W1": [(55, 56, 59)],
    "W4": [(67, 68, 71)],
    "primary": [(55, 56, 59), (67, 68, 71)],
    "H12_E": [(55, 56, 67)],
    "H12_L": [(67, 68, 79)],
    "H12": [(55, 56, 67), (67, 68, 79)],
}
CONFIG_DIR = ROOT / "experiments/optimization_v3/batch1/C10"
REPORT_DIR = ROOT / "reports/optimization_v3/batch1/C10"
RUN_DIR = ROOT / "experiments/runs/optimization_v3_batch1/C10"
TRAINING_ROOT = RUN_DIR.parent
FREEZE_PATH = CONFIG_DIR / "freeze.json"
RECIPE_PATH = CONFIG_DIR / "recipe.json"
ENGINE_REPORT_PATH = REPORT_DIR / "engine_report.json"
RESULTS_PATH = REPORT_DIR / "results.json"
VERIFICATION_PATH = REPORT_DIR / "verification.json"
SUMMARY_PATH = REPORT_DIR / "summary.md"
PREFLIGHT_PATH = REPORT_DIR / "preflight.json"
LEDGER_PATH = TRAINING_ROOT / "batch_ledger.jsonl"
MAX_FITS = 180
GIB = 1024 ** 3
MIN_FREE_RAM = 4 * GIB
SYSTEM_RESERVE = 2 * GIB
MAX_PROCESS_RSS = 6 * GIB
ESTIMATED_FIXED_EXTRA = 1 * GIB
EXPECTED_PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "threadpoolctl")

SOURCE_RELATIVE = (
    "scripts/optimization_v3_batch1_c10.py",
    "src/models/optimization_v3_batch1_c10.py",
    "tests/test_optimization_v3_batch1_c10.py",
    "scripts/optimization_v3_batch1.py",
    "scripts/optimization_v3_batch1_verify.py",
    "src/models/optimization_v3_batch1.py",
    "src/models/classification_models.py",
    "src/models/classification_round2_models.py",
    "src/models/classification_metrics.py",
    "src/models/regression_research_text.py",
    "src/models/regression_round2_models.py",
    "src/models/round1_models.py",
    "src/models/round1_evaluation.py",
    "src/models/optimization_v3_sparse_classification.py",
    "scripts/optimization_v3_sparse_classification.py",
    "scripts/optimization_v3_sparse_classification_fullsix.py",
    "scripts/optimization_v3_verify_sparse_classification.py",
    "scripts/optimization_v3_verify_sparse_classification_fullsix.py",
    "scripts/research_campaign_classification_stage2.py",
    "scripts/optimization_v2_classification.py",
    "src/utils/paths.py",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BatchIntegrityError(f"Expected JSON object in {path}")
    return value


def _plain_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in parameters.items():
        if isinstance(value, tuple):
            result[str(key)] = list(value)
        elif isinstance(value, type):
            result[str(key)] = f"{value.__module__}.{value.__qualname__}"
        elif isinstance(value, np.generic):
            result[str(key)] = value.item()
        else:
            result[str(key)] = value
    return result


def _is_no_supported_terms_error(error: ValueError) -> bool:
    message = str(error).casefold()
    return "empty vocabulary" in message or "no terms remain" in message


def _attempt_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    starts = [event for event in events if event.get("event") == "FIT_STARTED"]
    saved = [event for event in events if event.get("event") == "FIT_SAVED"]
    failed = [event for event in events if event.get("event") == "FIT_FAILED"]
    resumed = [event for event in events if event.get("event") == "RESUMED_VERIFIED"]
    if len(starts) != len(saved) + len(failed):
        raise BatchIntegrityError("C10 attempt ledger has an unfinished or unaccounted candidate fit")
    elapsed = sum(float(event.get("fit_seconds", 0.0)) for event in [*saved, *failed])
    return {
        "attempts_started": len(starts),
        "successful_fit_events": len(saved),
        "failed_predictor_attempts": len(failed),
        "reused_verified_bundles": len(resumed),
        "recorded_attempt_seconds": float(elapsed),
    }


def _write_json_match_or_new(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        if _read_json(path) != value:
            raise BatchIntegrityError(f"Owned C10 artifact differs; refusing overwrite: {path}")
        return
    write_json_exclusive(path, value)


def _load_canonical_card() -> tuple[dict[str, Any], dict[str, Any]]:
    queue = load_queue(QUEUE_PATH)
    matches = [item for item in queue.get("cards", []) if item.get("id") == CARD_ID]
    if len(matches) != 1:
        raise BatchIntegrityError("Canonical V3 queue must contain exactly one C10 card")
    card = matches[0]
    expected = {
        "id": "C10",
        "task": "classification",
        "name": "Separate trigram context",
        "family": "trigram",
        "policy": "C_SPARSE_REPLACE",
        "pilot_predictor_fits": 2,
        "evaluation": "STANDARD",
        "condition": "No additional dependency",
        "status": "PLANNED_NOT_RUN",
        "implementation": "NOT_IMPLEMENTED",
        "queue_order": 21,
        "in_priority_prefix": True,
    }
    for key, value in expected.items():
        if card.get(key) != value:
            raise BatchIntegrityError(f"Canonical C10 card differs for {key}: {card.get(key)!r}")
    change = str(card.get("change", ""))
    if not all(token in change for token in ("word3-only", "cap10000", "min_df3", "max_df0.995", "sublinear")):
        raise BatchIntegrityError("Canonical C10 change no longer describes the frozen trigram recipe")
    if card.get("preflight_checks") != [
        "Retire before fit if no supported trigrams; report occupied features/nnz"
    ]:
        raise BatchIntegrityError("Canonical C10 trigram preflight requirement changed")
    if queue.get("policies", {}).get("C_SPARSE_REPLACE") != (
        ".75 frozen C2_B +.25 new sparse expert on present; frozen CB missing; threshold.50"
    ):
        raise BatchIntegrityError("C10 frozen sparse replacement route changed")
    if queue.get("evaluation", {}).get("STANDARD", {}).get("pilot_cutoffs") != [55, 67]:
        raise BatchIntegrityError("C10 STANDARD pilot cutoffs changed")
    if queue.get("execution_preset", {}).get("concurrent_fit_jobs") != 1 or queue.get("execution_preset", {}).get("threads_per_job") != 2:
        raise BatchIntegrityError("C10 requires one fit job and two threads")
    baseline = queue.get("baseline", {}).get("classification", {})
    if (baseline.get("manifest") != "reports/optimization_v3/kaggle/v3_cls_sparse25.json"
            or baseline.get("manifest_sha256") != EXPECTED_BASELINE_MANIFEST_SHA256
            or baseline.get("Train_sha256") != EXPECTED_TRAIN_SHA256):
        raise BatchIntegrityError("C10 accepted baseline manifest or Train identity changed")
    gate = queue.get("evaluation", {}).get("classification_gate", {})
    required_gates = {
        "pilot_primary_gain_min": 0.0005,
        "pilot_H12_harm_max": 0.001,
        "individual_pilot_view_harm_max": 0.005,
        "full_primary_gain_min": 0.0005,
        "full_H12_harm_max": 0.001,
        "full_nonnegative_windows_min": 4,
        "supported_group_n_min": 200,
        "supported_group_F1_harm_max": 0.01,
        "positive_class_recall_harm_max": 0.02,
        "positive_and_negative_per_group_min": 30,
    }
    if gate != required_gates:
        raise BatchIntegrityError("C10 classification gate changed from the frozen queue values")
    return queue, card


def _official_train(*, include_target: bool) -> tuple[Path, pd.DataFrame, np.ndarray | None]:
    paths = task_files("classification")
    train_path = paths.get("train")
    if train_path is None or not Path(train_path).is_file():
        raise FileNotFoundError("Official Classification Train.csv is unavailable")
    train_path = Path(train_path).resolve()
    actual_sha = file_sha256(train_path)
    if actual_sha != EXPECTED_TRAIN_SHA256:
        raise BatchIntegrityError(f"Official Classification Train hash mismatch: {actual_sha}")
    if include_target:
        frame = pd.read_csv(train_path)
        if "artifact_signal" not in frame:
            raise BatchIntegrityError("Official Train is missing artifact_signal")
        y_float = pd.to_numeric(frame.pop("artifact_signal"), errors="raise").to_numpy(dtype=np.float64)
        if not np.isfinite(y_float).all() or not np.isin(y_float, (0.0, 1.0)).all():
            raise BatchIntegrityError("Official artifact_signal is not finite binary data")
        target = y_float.astype(np.int8)
    else:
        frame = pd.read_csv(train_path, usecols=lambda name: name != "artifact_signal")
        target = None
    required = set(FEATURE_NAMES) | {"record_id", "cycle_code"}
    if missing := sorted(required - set(frame.columns)):
        raise BatchIntegrityError(f"Official Train missing C10 fields: {missing}")
    forbidden = {"record_id", "artifact_signal", "threat_rank"}
    if forbidden.intersection(FEATURE_NAMES):
        raise BatchIntegrityError(f"C10 feature list contains protected fields: {sorted(forbidden.intersection(FEATURE_NAMES))}")
    if len(frame) != 50_608:
        raise BatchIntegrityError(f"Official Classification Train row count changed: {len(frame)}")
    ids = frame["record_id"].astype(str)
    if ids.isna().any() or ids.str.strip().eq("").any() or ids.duplicated().any():
        raise BatchIntegrityError("Official Train record_id values are blank or duplicated")
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    if not np.array_equal(np.unique(cycles), np.arange(47, 80, dtype=np.int64)):
        raise BatchIntegrityError("Official Train cycle coverage differs from 47–79")
    if target is not None and target.shape != (len(frame),):
        raise BatchIntegrityError("Official Train target is not row aligned")
    return train_path, frame, target


def _validate_baseline_manifest(queue: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if file_sha256(BASELINE_MANIFEST) != EXPECTED_BASELINE_MANIFEST_SHA256:
        raise BatchIntegrityError("Accepted sparse25 baseline manifest hash differs")
    manifest = _read_json(BASELINE_MANIFEST)
    if (manifest.get("status") != "PREPARED_AND_QA_PASS"
            or manifest.get("task") != "classification"
            or manifest.get("candidate_id") != "v3_cls_sparse25"
            or manifest.get("train_rows") != 50_608
            or manifest.get("input_sha256", {}).get("train") != EXPECTED_TRAIN_SHA256
            or manifest.get("fit_preprocessing_only_train") is not True
            or manifest.get("test_labels_used") is not False
            or manifest.get("recipe") != EXPECTED_PRESENT_RECIPE):
        raise BatchIntegrityError("Accepted sparse25 manifest status/provenance/route changed")
    pilot_freeze = _read_json(sparse_pilot.CONFIG / "freeze.json")
    pilot_report = _read_json(sparse_pilot.OUT / "pilot.json")
    pilot_verification = _read_json(sparse_pilot.OUT / "verification.json")
    if (pilot_verification.get("status") != "PASS"
            or pilot_report.get("freeze_signature") != pilot_freeze.get("signature")
            or pilot_verification.get("freeze_signature") != pilot_freeze.get("signature")
            or pilot_report.get("newfits") != 2
            or pilot_report.get("Test_read") is True
            or pilot_report.get("test_read") is True
            or pilot_report.get("kaggle_access") is True):
        raise BatchIntegrityError("C10 source sparse pilot is not a passing frozen Train-only control")
    return manifest, {
        "pilot_freeze": pilot_freeze,
        "pilot_report": pilot_report,
        "pilot_verification": pilot_verification,
    }


def _load_control_records(
    frame: pd.DataFrame,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    queue, _ = _load_canonical_card()
    manifest, source_receipts = _validate_baseline_manifest(queue)
    records: dict[int, dict[str, Any]] = {}
    lineage: dict[str, Any] = {}
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    for cutoff in FIT_CUTOFFS:
        # This existing bridge reads only the C2_B audit/cache and source frame. Its y
        # argument is unused; None makes the target-free preflight boundary explicit.
        control = sparse_fullsix.control_record(frame, None, cutoff)
        record = sparse_fullsix.pilot_record(frame, control)
        rows = np.asarray(record["rows"], dtype=np.int64)
        horizon = HORIZON_BY_CUTOFF[cutoff]
        expected_rows = np.flatnonzero((cycles > cutoff) & (cycles <= horizon))
        if not np.array_equal(rows, expected_rows):
            raise BatchIntegrityError(f"C10 frozen-control score rows differ at cutoff {cutoff}")
        ids = frame.iloc[rows]["record_id"].astype(str).to_numpy(dtype=str)
        missing = stream_missing_mask(frame.iloc[rows])
        if not np.array_equal(missing, np.asarray(record["missing"], dtype=bool)):
            raise BatchIntegrityError(f"C10 control stream mask differs at cutoff {cutoff}")
        baseline = np.where(
            missing, np.asarray(record["cb_missing"], dtype=np.float64),
            0.75 * np.asarray(record["c2b"], dtype=np.float64)
            + 0.25 * np.asarray(record["raw_candidate"], dtype=np.float64),
        )
        if not np.array_equal(baseline, np.asarray(record["blend25"], dtype=np.float64)):
            raise BatchIntegrityError(f"C10 accepted sparse25 baseline does not reconstruct at {cutoff}")
        fit_rows = np.flatnonzero((cycles >= 47) & (cycles <= cutoff))
        if not np.array_equal(fit_rows, np.asarray(control["fit_mask"], dtype=bool).nonzero()[0]):
            raise BatchIntegrityError(f"C10 control fit population differs at cutoff {cutoff}")
        sparse_meta = _read_json(sparse_pilot.CACHE / f"fit{cutoff}.json")
        if sparse_meta.get("freeze_signature") != source_receipts["pilot_freeze"].get("signature"):
            raise BatchIntegrityError(f"C10 sparse control freeze differs at cutoff {cutoff}")
        if sparse_meta.get("npz_sha256") != file_sha256(sparse_pilot.CACHE / f"fit{cutoff}.npz"):
            raise BatchIntegrityError(f"C10 sparse control cache hash differs at cutoff {cutoff}")
        if (control["c2b_identity"].get("fit_rows") != len(fit_rows)
                or control["horizon"] != horizon):
            raise BatchIntegrityError(f"C10 C2_B fit/horizon lineage differs at cutoff {cutoff}")
        records[cutoff] = {
            "cutoff": cutoff,
            "horizon": horizon,
            "fit_rows": fit_rows,
            "rows": rows,
            "ids": ids,
            "baseline_prediction": baseline,
            "c2b_probability": np.asarray(record["c2b"], dtype=np.float64),
            "cb_missing_probability": np.asarray(record["cb_missing"], dtype=np.float64),
            "sparse_raw_probability": np.asarray(record["raw_candidate"], dtype=np.float64),
            "stream_missing": missing,
            "stream_present": ~missing,
        }
        lineage[str(cutoff)] = {
            "fit_cycles": [47, cutoff],
            "fit_rows": int(len(fit_rows)),
            "score_cycles": [cutoff + 1, horizon],
            "score_rows": int(len(rows)),
            "present_rows": int(np.count_nonzero(~missing)),
            "missing_rows": int(np.count_nonzero(missing)),
            "row_positions_sha256": array_digest(rows.astype(np.float64)),
            "record_ids_sha256": hashlib.sha256("\n".join(ids.tolist()).encode("utf-8")).hexdigest(),
            "C2_B_cache_identity": control["c2b_identity"],
            "CatBoost_cache_sha256": file_sha256(sparse_fullsix.CB_CACHE / f"fit{cutoff}.npz"),
            "CatBoost_sidecar_sha256": file_sha256(sparse_fullsix.CB_CACHE / f"fit{cutoff}.json"),
            "sparse25_cache_sha256": file_sha256(sparse_pilot.CACHE / f"fit{cutoff}.npz"),
            "sparse25_sidecar_sha256": file_sha256(sparse_pilot.CACHE / f"fit{cutoff}.json"),
            "sparse25_freeze_signature": sparse_meta["freeze_signature"],
            "baseline_reconstructed_exactly": True,
            "target_accessed": False,
        }
    return records, lineage, {"manifest": manifest, **source_receipts}


def _memory_snapshot() -> dict[str, Any]:
    if os.name == "nt":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]
        memory = MEMORYSTATUSEX()
        memory.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MEMORYSTATUSEX)]
        kernel32.GlobalMemoryStatusEx.restype = ctypes.c_int
        if not kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            raise OSError(f"GlobalMemoryStatusEx failed: {ctypes.get_last_error()}")
        counters = PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX), ctypes.c_ulong]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), ctypes.sizeof(counters)):
            raise OSError(f"GetProcessMemoryInfo failed: {ctypes.get_last_error()}")
        total = int(memory.ullTotalPhys)
        available = int(memory.ullAvailPhys)
        rss = int(counters.WorkingSetSize)
        peak = int(counters.PeakWorkingSetSize)
        source = "Windows GlobalMemoryStatusEx/GetProcessMemoryInfo"
    else:
        import resource
        total = int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
        available = int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
        peak_raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        peak = peak_raw if platform.system() == "Darwin" else peak_raw * 1024
        rss = peak
        source = "POSIX sysconf/resource peak RSS"
    return {
        "measurement_source": source,
        "total_ram_bytes": total,
        "available_ram_bytes": available,
        "process_rss_bytes": rss,
        "process_peak_rss_bytes": peak,
    }


def _fit_only_support(frame: pd.DataFrame, cutoff: int) -> dict[str, Any]:
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    fit_rows = np.flatnonzero((cycles >= 47) & (cycles <= cutoff))
    fit_frame = frame.iloc[fit_rows]
    documents, missing = _text_values(fit_frame["signal_stream"])
    word = word_vectorizer()
    word_matrix = word.fit_transform(documents).tocsr()
    trigram = trigram_vectorizer()
    try:
        trigram_matrix = trigram.fit_transform(documents).tocsr()
    except ValueError as exc:
        if not _is_no_supported_terms_error(exc):
            raise
        trigram_matrix = sparse.csr_matrix((len(fit_rows), 0), dtype=np.float64)
        trigram.vocabulary_ = {}
        trigram.idf_ = np.asarray([], dtype=np.float64)
    word_bytes = int(word_matrix.data.nbytes + word_matrix.indices.nbytes + word_matrix.indptr.nbytes)
    trigram_bytes = int(trigram_matrix.data.nbytes + trigram_matrix.indices.nbytes + trigram_matrix.indptr.nbytes)
    result = {
        "fit_cycles": [47, cutoff],
        "fit_rows": int(len(fit_rows)),
        "stream_present_rows": int(np.count_nonzero(~missing)),
        "stream_missing_rows": int(np.count_nonzero(missing)),
        "word12": {
            "parameters": _plain_parameters(_TFIDF_PARAMS),
            "occupied_features": int(len(word.vocabulary_)),
            "nnz": int(word_matrix.nnz),
            "rows_with_features": int(np.count_nonzero(np.diff(word_matrix.indptr))),
            "csr_bytes": word_bytes,
        },
        "trigram": {
            "parameters": _plain_parameters(TRIGRAM_PARAMS),
            "occupied_features": int(len(trigram.vocabulary_)),
            "nnz": int(trigram_matrix.nnz),
            "rows_with_features": int(np.count_nonzero(np.diff(trigram_matrix.indptr))),
            "rows_without_selected_trigram": int(len(fit_rows) - np.count_nonzero(np.diff(trigram_matrix.indptr))),
            "csr_bytes": trigram_bytes,
            "vocabulary_sha256": _vocabulary_sha256(trigram.vocabulary_),
            "idf_sha256": array_digest(np.asarray(trigram.idf_, dtype=np.float64)),
        },
        "word_vocabulary_sha256": _vocabulary_sha256(word.vocabulary_),
        "word_idf_sha256": array_digest(np.asarray(word.idf_, dtype=np.float64)),
        "fit_labels_read": False,
    }
    del word_matrix, trigram_matrix, word, trigram, fit_frame, documents
    gc.collect()
    return result


def _vocabulary_sha256(vocabulary: Mapping[str, int]) -> str:
    ordered = [(str(token), int(index)) for token, index in sorted(vocabulary.items(), key=lambda item: item[1])]
    return hashlib.sha256(json.dumps(ordered, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def _resource_gate(support: Mapping[str, Any], records: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    snapshot = _memory_snapshot()
    cutoff = max(FIT_CUTOFFS)
    fit_rows = int(support[str(cutoff)]["fit_rows"])
    source_audit = _read_json(sparse_fullsix.ROUND1_AUDITS / f"fit{cutoff}.json").get("C2_B", {})
    transformed = source_audit.get("fitted_state", {}).get("transformed_features", {})
    numeric_width = int(transformed.get("numeric_count", 0))
    category_columns = len(transformed.get("encoded_category_counts", []))
    if numeric_width <= 0 or category_columns != 3:
        raise BatchIntegrityError("C10 cannot establish a structured sparse-memory bound from C2_B audit")
    structured_nnz_upper = fit_rows * (numeric_width + category_columns)
    missing_nnz_upper = fit_rows
    text_nnz = int(support[str(cutoff)]["word12"]["nnz"] + support[str(cutoff)]["trigram"]["nnz"])
    total_nnz_upper = structured_nnz_upper + missing_nnz_upper + text_nnz
    csr_design_bytes_upper = total_nnz_upper * 16 + (fit_rows + 1) * 8
    # Keep simultaneous sparse blocks/outputs plus a 1 GiB allowance for LightGBM
    # and parser/preprocessor internals; no dense word or trigram matrix is built.
    estimated_additional = ESTIMATED_FIXED_EXTRA + 4 * csr_design_bytes_upper
    available = int(snapshot["available_ram_bytes"])
    rss = int(snapshot["process_rss_bytes"])
    predicted_peak = rss + estimated_additional
    passed = bool(
        available >= MIN_FREE_RAM
        and estimated_additional <= available - SYSTEM_RESERVE
        and predicted_peak <= MAX_PROCESS_RSS
    )
    return {
        **snapshot,
        "policy": {
            "minimum_free_ram_bytes": MIN_FREE_RAM,
            "reserved_system_ram_bytes": SYSTEM_RESERVE,
            "maximum_process_rss_bytes": MAX_PROCESS_RSS,
            "fixed_extra_allowance_bytes": ESTIMATED_FIXED_EXTRA,
            "sparse_design_multiplier": 4,
            "structured_numeric_width": numeric_width,
            "structured_category_columns": category_columns,
        },
        "largest_fit_rows": fit_rows,
        "estimated_word_plus_trigram_nnz": text_nnz,
        "estimated_structured_nnz_upper": structured_nnz_upper,
        "estimated_missing_indicator_nnz_upper": missing_nnz_upper,
        "estimated_sparse_design_bytes_upper": csr_design_bytes_upper,
        "estimated_additional_process_bytes": estimated_additional,
        "predicted_process_peak_rss_bytes": predicted_peak,
        "status": "PASS" if passed else "SKIPPED_RESOURCE",
        "predictor_fits_performed": 0,
        "support_preflight_vectorizer_fits": 4,
    }


def _input_paths(train_path: Path) -> dict[str, Path]:
    inputs: dict[str, Path] = {
        "official_classification_train": train_path,
        "queue": QUEUE_PATH,
        "card": CARD_PATH,
        "accepted_sparse25_manifest": BASELINE_MANIFEST,
        "sparse_pilot_protocol": sparse_pilot.CONFIG / "protocol.json",
        "sparse_pilot_freeze": sparse_pilot.CONFIG / "freeze.json",
        "sparse_pilot_report": sparse_pilot.OUT / "pilot.json",
        "sparse_pilot_verification": sparse_pilot.OUT / "verification.json",
    }
    for cutoff in FIT_CUTOFFS:
        inputs[f"C2_B_fit{cutoff}_cache"] = sparse_fullsix.ROUND1_CACHE / f"fit{cutoff}__C2_B.npz"
        inputs[f"C2_B_fit{cutoff}_audit"] = sparse_fullsix.ROUND1_AUDITS / f"fit{cutoff}.json"
        inputs[f"CatBoost_fit{cutoff}_cache"] = sparse_fullsix.CB_CACHE / f"fit{cutoff}.npz"
        inputs[f"CatBoost_fit{cutoff}_sidecar"] = sparse_fullsix.CB_CACHE / f"fit{cutoff}.json"
        inputs[f"sparse25_fit{cutoff}_cache"] = sparse_pilot.CACHE / f"fit{cutoff}.npz"
        inputs[f"sparse25_fit{cutoff}_sidecar"] = sparse_pilot.CACHE / f"fit{cutoff}.json"
    return inputs


def _source_paths() -> list[Path]:
    return [ROOT / relative for relative in SOURCE_RELATIVE]


def _support_view_rows(frame: pd.DataFrame, views: Mapping[str, list[tuple[int, int, int]]],
                       records: Mapping[int, Mapping[str, Any]]) -> dict[str, int]:
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    result: dict[str, int] = {}
    for name, segments in views.items():
        selected: list[np.ndarray] = []
        for cutoff, low, high in segments:
            source_rows = np.asarray(records[int(cutoff)]["rows"], dtype=np.int64)
            selected.append(source_rows[(cycles[source_rows] >= low) & (cycles[source_rows] <= high)])
        combined = np.concatenate(selected)
        if len(np.unique(combined)) != len(combined):
            raise BatchIntegrityError(f"C10 view {name} contains duplicate Train rows")
        result[name] = int(len(combined))
    if result.get("primary") != 9_064 or result.get("H12") != 38_588:
        raise BatchIntegrityError(f"C10 canonical view denominators changed: {result}")
    return result


def _make_identity(
    queue: Mapping[str, Any], card: Mapping[str, Any], train_path: Path,
    records: Mapping[int, Mapping[str, Any]], lineage: Mapping[str, Any],
    support: Mapping[str, Any], views: Mapping[str, list[tuple[int, int, int]]],
    resource: Mapping[str, Any],
) -> dict[str, Any]:
    identity = make_freeze_identity(
        card=card,
        feature_names=FEATURE_NAMES,
        input_files=_input_paths(train_path),
        source_files=_source_paths(),
        evaluation_views=views,
        packages=EXPECTED_PACKAGES,
    )
    # Dynamic RAM readings do not enter the freeze signature; deterministic support
    # and source identities do.
    identity.update({
        "canonical_queue_sha256": file_sha256(QUEUE_PATH),
        "canonical_card_sha256": file_sha256(CARD_PATH),
        "baseline_manifest_sha256": file_sha256(BASELINE_MANIFEST),
        "control_lineage": dict(lineage),
        "control_population_by_cutoff": {
            str(cutoff): {
                "fit_rows": int(len(records[cutoff]["fit_rows"])),
                "score_rows": int(len(records[cutoff]["rows"])),
                "present_rows": int(np.count_nonzero(records[cutoff]["stream_present"])),
                "missing_rows": int(np.count_nonzero(records[cutoff]["stream_missing"])),
            }
            for cutoff in FIT_CUTOFFS
        },
        "support_preflight": {str(key): dict(value) for key, value in support.items()},
        "sparse_memory_estimate": {
            key: resource[key] for key in (
                "estimated_word_plus_trigram_nnz", "estimated_structured_nnz_upper",
                "estimated_missing_indicator_nnz_upper", "estimated_sparse_design_bytes_upper",
                "estimated_additional_process_bytes",
            )
        },
        "python_version": platform.python_version(),
        "validation_labels_passed_to_adapter": False,
        "test_data_used": False,
        "sample_submission_used": False,
        "kaggle_access": False,
    })
    return identity


def preflight() -> dict[str, Any]:
    queue, card = _load_canonical_card()
    train_path, frame, target = _official_train(include_target=False)
    if target is not None or "artifact_signal" in frame.columns:
        raise BatchIntegrityError("C10 preflight opened the target column")
    views = evaluation_views_for_card(queue, CARD_ID)
    if set(views) != {"W1", "W4", "primary", "H12"}:
        raise BatchIntegrityError("C10 STANDARD view definition changed")
    views = {**views, "H12_E": list(EXPECTED_VIEWS["H12_E"]), "H12_L": list(EXPECTED_VIEWS["H12_L"])}
    if views != EXPECTED_VIEWS:
        raise BatchIntegrityError("C10 pilot views differ from the frozen STANDARD boundaries")
    records, lineage, _ = _load_control_records(frame)
    support = {str(cutoff): _fit_only_support(frame, cutoff) for cutoff in FIT_CUTOFFS}
    unsupported = [cutoff for cutoff in FIT_CUTOFFS
                   if support[str(cutoff)]["trigram"]["occupied_features"] < 1
                   or support[str(cutoff)]["trigram"]["nnz"] < 1]
    view_rows = _support_view_rows(frame, views, records)
    resource = _resource_gate(support, records)
    identity = _make_identity(queue, card, train_path, records, lineage, support, views, resource)
    status = "PREFLIGHT_PASS_READ_ONLY_NO_TARGET_NO_SCORE_NO_PREDICTOR_FIT"
    reason = None
    if unsupported:
        status = "SKIPPED_SUPPORT"
        reason = f"No supported C10 trigram at cutoff(s): {unsupported}"
    elif resource["status"] != "PASS":
        status = "SKIPPED_RESOURCE"
        reason = "C10 memory estimate does not satisfy the frozen RAM policy"
    return {
        "status": status,
        "card_id": CARD_ID,
        "task": "classification",
        "train_sha256": file_sha256(train_path),
        "train_rows": int(len(frame)),
        "train_cycle_range": [47, 79],
        "support_preflight": support,
        "supported_trigram_cutoffs": [cutoff for cutoff in FIT_CUTOFFS if cutoff not in unsupported],
        "unsupported_trigram_cutoffs": unsupported,
        "control_lineage": lineage,
        "view_rows": view_rows,
        "resource_forecast": resource,
        "preflight_vectorizer_fits": 4,
        "predictor_fits_performed": 0,
        "target_read": False,
        "candidate_metrics_computed": False,
        "test_read": False,
        "sample_read": False,
        "kaggle_access": False,
        "reason": reason,
        "identity": identity,
        "freeze_identity_sha256": json_sha256(identity),
        "owned_outputs": {
            "recipe": RECIPE_PATH.exists(), "freeze": FREEZE_PATH.exists(),
            "results": RESULTS_PATH.exists(), "verification": VERIFICATION_PATH.exists(),
            "cache": RUN_DIR.exists(),
        },
    }


def _recipe(preflight_result: Mapping[str, Any], queue: Mapping[str, Any], card: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "card_id": CARD_ID,
        "task": "classification",
        "status": "FROZEN_BEFORE_C10_PILOT_FITS",
        "canonical_card_path": CARD_PATH.relative_to(ROOT).as_posix(),
        "canonical_card_sha256": file_sha256(CARD_PATH),
        "canonical_queue_sha256": file_sha256(QUEUE_PATH),
        "change": card["change"],
        "policy": card["policy"],
        "model": {
            "id": "SeparateTrigramSparseClassifier",
            "structured_preprocessing": "unchanged RegressionTextModel structured fit-only preprocessor from incumbent sparse classifier",
            "word_vectorizer": {
                **_plain_parameters(_TFIDF_PARAMS)
            },
            "trigram_vectorizer": {
                **_plain_parameters(TRIGRAM_PARAMS)
            },
            "feature_block_order": ["structured", "word_unigram_bigram", "word_trigram_only", "stream_missing_indicator"],
            "classifier": {
                "class": "lightgbm.LGBMClassifier",
                "parameters": dict(SPARSE_PARAMS),
                "early_stopping": False,
                "outer_validation_labels_used_for_fit": False,
            },
        },
        "routing": {
            "stream_present": "0.75 * frozen C2_B + 0.25 * C10 candidate probability",
            "stream_missing": "frozen CatBoost probability, unchanged",
            "threshold": THRESHOLD,
            "threshold_search": False,
        },
        "evaluation": {
            "profile": "STANDARD",
            "fit_cutoffs": list(FIT_CUTOFFS),
            "fit_cycles": {str(cutoff): [47, cutoff] for cutoff in FIT_CUTOFFS},
            "score_horizons": {str(cutoff): [cutoff + 1, HORIZON_BY_CUTOFF[cutoff]] for cutoff in FIT_CUTOFFS},
            "views": {name: [list(segment) for segment in segments]
                      for name, segments in preflight_result["identity"]["evaluation_views"].items()},
            "primary_rows": 9_064,
            "H12_rows": 38_588,
            "gate": dict(queue["evaluation"]["classification_gate"]),
            "metrics": "binary Macro F1 at fixed threshold 0.50",
        },
        "support_preflight": preflight_result["support_preflight"],
        "resource_forecast": preflight_result["resource_forecast"],
        "source_input_identity_sha256": preflight_result["freeze_identity_sha256"],
        "source_input_identity": preflight_result["identity"],
        "training_boundary": "official Classification Train only; fit preprocessing on each temporal prefix; no Test/Sample/Kaggle",
        "preflight_vectorizer_fits": 4,
        "official_candidate_predictor_fits": int(card["pilot_predictor_fits"]),
    }


def write_freeze() -> dict[str, Any]:
    result = preflight()
    if result["status"] != "PREFLIGHT_PASS_READ_ONLY_NO_TARGET_NO_SCORE_NO_PREDICTOR_FIT":
        raise BatchIntegrityError(f"C10 cannot freeze after blocked preflight: {result['status']}")
    if RESULTS_PATH.exists() or VERIFICATION_PATH.exists() or ENGINE_REPORT_PATH.exists():
        raise FileExistsError("C10 terminal/pilot reports already exist; no new freeze will replace them")
    queue, card = _load_canonical_card()
    identity = result["identity"]
    signature = json_sha256(identity)
    if FREEZE_PATH.exists() != RECIPE_PATH.exists():
        raise BatchIntegrityError("C10 freeze/recipe pair is incomplete; preserving partial artifacts")
    if FREEZE_PATH.exists():
        old_freeze = _read_json(FREEZE_PATH)
        old_recipe = _read_json(RECIPE_PATH)
        if (old_freeze.get("identity") != identity or old_freeze.get("signature") != signature
                or old_freeze.get("recipe_file_sha256") != file_sha256(RECIPE_PATH)
                or old_freeze.get("recipe_sha256") != old_recipe.get("recipe_sha256")
                or old_recipe.get("source_input_identity_sha256") != signature):
            raise BatchIntegrityError("Existing C10 freeze is inconsistent with current identity; preserving it")
        if PREFLIGHT_PATH.exists():
            saved_preflight = _read_json(PREFLIGHT_PATH)
            if saved_preflight.get("identity") != identity:
                raise BatchIntegrityError("Existing C10 preflight receipt identity differs")
        else:
            write_json_exclusive(PREFLIGHT_PATH, result)
        return {"status": "FROZEN_BEFORE_C10_PILOT_FITS", "freeze_signature": signature,
                "recipe_sha256": old_recipe["recipe_sha256"], "preflight": result,
                "reused_existing_freeze": True}

    recipe = _recipe(result, queue, card)
    recipe["recipe_sha256"] = json_sha256(recipe)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if PREFLIGHT_PATH.exists():
        saved_preflight = _read_json(PREFLIGHT_PATH)
        if saved_preflight.get("identity") != identity:
            raise BatchIntegrityError("Existing C10 preflight receipt identity differs")
    else:
        write_json_exclusive(PREFLIGHT_PATH, result)
    write_json_exclusive(RECIPE_PATH, recipe)
    freeze = {
        "status": "FROZEN_BEFORE_C10_PILOT_FITS",
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "signature": signature,
        "identity": identity,
        "recipe_sha256": recipe["recipe_sha256"],
        "recipe_file_sha256": file_sha256(RECIPE_PATH),
        "official_candidate_predictor_fits_authorized_by_card": int(card["pilot_predictor_fits"]),
        "preflight_vectorizer_fits_are_support_probes_not_predictors": 4,
        "target_read": False,
        "test_read": False,
        "sample_read": False,
        "kaggle_access": False,
    }
    if PREFLIGHT_PATH.exists():
        saved_preflight = _read_json(PREFLIGHT_PATH)
        if saved_preflight.get("identity") != identity:
            raise BatchIntegrityError("Existing C10 preflight receipt identity differs")
    write_json_exclusive(FREEZE_PATH, freeze)
    return {"status": "FROZEN_BEFORE_C10_PILOT_FITS", "freeze_signature": freeze["signature"],
            "recipe_sha256": recipe["recipe_sha256"], "preflight": result}


def _check_freeze() -> tuple[dict[str, Any], dict[str, Any]]:
    current = preflight()
    if current["status"] != "PREFLIGHT_PASS_READ_ONLY_NO_TARGET_NO_SCORE_NO_PREDICTOR_FIT":
        raise BatchIntegrityError(f"C10 current preflight is blocked: {current['status']}")
    freeze = _read_json(FREEZE_PATH)
    if freeze.get("identity") != current["identity"] or freeze.get("signature") != current["freeze_identity_sha256"]:
        raise BatchIntegrityError("C10 freeze does not match current source/input/support identity")
    recipe = _read_json(RECIPE_PATH)
    if (freeze.get("recipe_file_sha256") != file_sha256(RECIPE_PATH)
            or freeze.get("recipe_sha256") != recipe.get("recipe_sha256")
            or recipe.get("source_input_identity_sha256") != freeze.get("signature")):
        raise BatchIntegrityError("C10 frozen recipe hash/signature mismatch")
    return freeze, current


def _control_loader(records: Mapping[int, Mapping[str, Any]]):
    def load(cutoff: int, score_rows: np.ndarray) -> dict[str, np.ndarray]:
        record = records[int(cutoff)]
        rows = np.asarray(score_rows, dtype=np.int64)
        if not np.array_equal(rows, record["rows"]):
            raise BatchIntegrityError(f"C10 harness rows differ from frozen source controls at {cutoff}")
        return {
            "baseline_prediction": np.asarray(record["baseline_prediction"], dtype=np.float64),
            "c2b_probability": np.asarray(record["c2b_probability"], dtype=np.float64),
            "cb_missing_probability": np.asarray(record["cb_missing_probability"], dtype=np.float64),
            "sparse_raw_probability": np.asarray(record["sparse_raw_probability"], dtype=np.float64),
            "stream_missing": np.asarray(record["stream_missing"], dtype=bool),
            "source_rows": rows,
            "source_record_ids": np.asarray(record["ids"], dtype=str),
        }
    return load


def _subgroup_failures(verification: Mapping[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for view_name, view in verification.get("views", {}).items():
        diagnostic = view.get("subgroup_diagnostics", {})
        for item in diagnostic.get("supported_guard_failures", []):
            failures.append({"view": view_name, **item})
    return failures


def _gate(
    result: Mapping[str, Any], verification: Mapping[str, Any], queue: Mapping[str, Any],
    resource_postfit: Mapping[str, Any],
) -> dict[str, Any]:
    views = result["views"]
    gate = queue["evaluation"]["classification_gate"]
    primary_gain = float(views["primary"]["gain_positive_is_better"])
    h12_harm = -float(views["H12"]["gain_positive_is_better"])
    window_harms = {
        name: -float(views[name]["gain_positive_is_better"])
        for name in ("W1", "W4")
    }
    subgroup_failures = _subgroup_failures(verification)
    checks = {
        "primary_gain": primary_gain,
        "primary_gain_min": float(gate["pilot_primary_gain_min"]),
        "primary_pass": primary_gain >= float(gate["pilot_primary_gain_min"]),
        "H12_harm": h12_harm,
        "H12_harm_max": float(gate["pilot_H12_harm_max"]),
        "H12_pass": h12_harm <= float(gate["pilot_H12_harm_max"]),
        "individual_window_harms": window_harms,
        "individual_window_harm_max": float(gate["individual_pilot_view_harm_max"]),
        "individual_windows_pass": all(value <= float(gate["individual_pilot_view_harm_max"])
                                        for value in window_harms.values()),
        "supported_subgroup_failures": subgroup_failures,
        "supported_subgroups_pass": len(subgroup_failures) == 0,
        "independent_verification_pass": verification.get("status") == "PASS",
        "postfit_resource_cap_pass": int(resource_postfit.get("process_peak_rss_bytes", MAX_PROCESS_RSS + 1)) <= MAX_PROCESS_RSS,
        "postfit_process_peak_rss_bytes": resource_postfit.get("process_peak_rss_bytes"),
        "maximum_process_rss_bytes": MAX_PROCESS_RSS,
    }
    checks["pass"] = bool(
        checks["primary_pass"] and checks["H12_pass"] and checks["individual_windows_pass"]
        and checks["supported_subgroups_pass"] and checks["independent_verification_pass"]
        and checks["postfit_resource_cap_pass"]
    )
    checks["verdict"] = (
        "ADVANCE" if checks["pass"] else
        "RETIRE" if checks["independent_verification_pass"] and checks["postfit_resource_cap_pass"] else
        "REWORK"
    )
    return checks


def _write_summary(result: Mapping[str, Any], verification: Mapping[str, Any]) -> None:
    lines = [
        "# C10 — Separate trigram context pilot",
        "",
        f"- Verdict: **{result['verdict']}**",
        f"- Freeze: `{result['freeze_signature']}`",
        f"- Independent verification: **{verification.get('status')}**",
        f"- Predictor fits: {result['fit_accounting']['successful_predictor_fits']} successful / "
        f"{result['fit_accounting']['attempts_started']} attempts; support-only TF-IDF probes: 4",
        f"- Threshold: {THRESHOLD:.2f}; present route 0.75 C2_B + 0.25 C10; missing route frozen CatBoost.",
        "",
        "| View | n | Baseline Macro F1 | C10 Macro F1 | Delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("W1", "W4", "primary", "H12_E", "H12_L", "H12"):
        view = result["views"][name]
        lines.append(
            f"| {name} | {view['rows']} | {view['baseline']['macro_f1']:.6f} | "
            f"{view['candidate']['macro_f1']:.6f} | {view['gain_positive_is_better']:+.6f} |"
        )
    lines.extend([
        "",
        "Supported subgroup guard details and exact denominators are in `verification.json`.",
        "Temporal windows reuse Train outcomes and are development evidence, not an untouched holdout.",
        "No Test, SampleSubmission, or Kaggle access occurred.",
        "",
    ])
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def _publish_resource_skip(freeze: Mapping[str, Any], resource: Mapping[str, Any]) -> dict[str, Any]:
    if RUN_DIR.exists() and any(RUN_DIR.glob("C10_fit*.npz")):
        raise BatchIntegrityError("C10 has partial candidate bundles; preserve and verify them before any resource skip")
    result = {
        "status": "SKIPPED_RESOURCE",
        "verdict": "SKIPPED",
        "card_id": CARD_ID,
        "freeze_signature": freeze["signature"],
        "fit_accounting": {
            "successful_predictor_fits": 0,
            "attempts_started": 0,
            "failed_predictor_attempts": 0,
            "preflight_vectorizer_fits": 4,
        },
        "resource_pre_fit": dict(resource),
        "candidate_metrics_computed": False,
        "test_read": False,
        "sample_read": False,
        "kaggle_access": False,
        "reason": "The fresh pre-fit C10 memory gate failed after the frozen recipe was created.",
    }
    verification = {
        "status": "NOT_RUN_RESOURCE_GATE",
        "card_id": CARD_ID,
        "freeze_signature": freeze["signature"],
        "resource_pre_fit": dict(resource),
        "predictor_fit_started": False,
        "candidate_metrics_computed": False,
        "test_read": False,
        "sample_read": False,
        "kaggle_access": False,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    write_json_exclusive(RESULTS_PATH, result)
    write_json_exclusive(VERIFICATION_PATH, verification)
    SUMMARY_PATH.write_text(
        "# C10 — Skipped before predictor fit\n\n"
        "- Verdict: **SKIPPED_RESOURCE**\n"
        "- No C10 predictor fit or candidate metric was run.\n"
        "- Fresh RAM measurements and the frozen memory rule are in `results.json`.\n"
        "- Test, SampleSubmission, and Kaggle were not accessed.\n",
        encoding="utf-8", newline="\n",
    )
    return result


def run_pilot_c10() -> dict[str, Any]:
    freeze, initial_preflight = _check_freeze()
    if any(path.exists() for path in (RESULTS_PATH, VERIFICATION_PATH, SUMMARY_PATH)):
        raise FileExistsError("C10 terminal report already exists; refusing overwrite")
    queue, card = _load_canonical_card()
    train_path, prefit_frame, no_target = _official_train(include_target=False)
    if no_target is not None:
        raise BatchIntegrityError("C10 fresh resource gate frame unexpectedly includes target")
    records, lineage, _ = _load_control_records(prefit_frame)
    support = initial_preflight["support_preflight"]
    prefit_resource = _resource_gate(support, records)
    if prefit_resource["status"] != "PASS":
        del prefit_frame
        gc.collect()
        return _publish_resource_skip(freeze, prefit_resource)
    del prefit_frame
    gc.collect()
    train_path, frame, target = _official_train(include_target=True)
    if target is None:
        raise BatchIntegrityError("C10 scoring requires official Train labels after freeze")
    resource = _resource_gate(support, records)
    if resource["status"] != "PASS":
        del frame, target
        gc.collect()
        return _publish_resource_skip(freeze, resource)
    views = {name: [tuple(int(value) for value in segment) for segment in segments]
             for name, segments in EXPECTED_VIEWS.items()}
    with threadpool_limits(limits=2):
        engine_report = run_pilot(
            card=card,
            artifact_dir=RUN_DIR,
            budget_ledger_path=LEDGER_PATH,
            max_successful_fits=MAX_FITS,
            frame=frame,
            target=target,
            feature_names=FEATURE_NAMES,
            adapter=fit_predict,
            control_loader=_control_loader(records),
            freeze_identity=freeze["identity"],
            freeze_path=FREEZE_PATH,
            event_path=RUN_DIR / "attempts.jsonl",
            evaluation_views=views,
            threads=2,
            report_path=ENGINE_REPORT_PATH,
        )
    attempt_events = []
    attempts_path = RUN_DIR / "attempts.jsonl"
    if attempts_path.is_file():
        attempt_events = [json.loads(line) for line in attempts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    attempts = _attempt_summary(attempt_events)
    verification = verify_report(
        report=engine_report,
        frame=frame,
        target=target,
        artifact_dir=RUN_DIR,
        freeze_signature=freeze["signature"],
        evaluation_views=views,
        prediction_reconstructors={CARD_ID: reconstruct_c10},
    )
    resource_postfit = _memory_snapshot()
    gate = _gate(engine_report, verification, queue, resource_postfit)
    results = {
        "status": f"PILOT_COMPLETE_{gate['verdict']}",
        "verdict": gate["verdict"],
        "card_id": CARD_ID,
        "freeze_signature": freeze["signature"],
        "source_train_sha256": file_sha256(train_path),
        "control_lineage": lineage,
        "support_preflight": support,
        "resource_forecast": resource,
        "resource_postfit": resource_postfit,
        "fit_accounting": {
            "successful_predictor_fits": int(engine_report["fit_predictor_count"]),
            "expected_predictor_fits": int(card["pilot_predictor_fits"]),
            **attempts,
            "preflight_vectorizer_fits": 4,
            "preprocessing_fits_per_predictor": 3,
            "threads": 2,
            "concurrent_jobs": 1,
        },
        "views": engine_report["views"],
        "gate": gate,
        "engine_report_sha256": file_sha256(ENGINE_REPORT_PATH),
        "test_read": False,
        "sample_read": False,
        "kaggle_access": False,
        "interpretation_limit": "Repeated temporal Train-cycle development results; no untouched holdout claim.",
    }
    verification_doc = {
        **verification,
        "gate_verdict": gate["verdict"],
        "gate_checks": gate,
        "results_sha256_before_publication": json_sha256(results),
        "test_read": False,
        "sample_read": False,
        "kaggle_access": False,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    write_json_exclusive(RESULTS_PATH, results)
    write_json_exclusive(VERIFICATION_PATH, verification_doc)
    _write_summary(results, verification_doc)
    return results


def run_card(card: dict[str, Any], queue: dict[str, Any], *, preflight_only: bool = False) -> dict[str, Any]:
    canonical_queue, canonical_card = _load_canonical_card()
    if queue != canonical_queue or card != canonical_card:
        raise BatchIntegrityError("C10 dispatcher received a noncanonical queue/card")
    if preflight_only:
        return preflight()
    raise BatchIntegrityError("Use the explicit C10 --write-freeze and then --run stages")


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--write-freeze", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        result = preflight()
    elif args.write_freeze:
        result = write_freeze()
    else:
        result = run_pilot_c10()
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
