"""Staged R17 preflight, recipe freeze, and exact-OOF calibration pilot."""
from __future__ import annotations

import json
import ctypes
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import optimization_v3_batch1_r18 as r18
from scripts.optimization_v3_batch1_verify import verify_report
from src.models.optimization_v3_batch1 import (
    BatchIntegrityError,
    evaluation_views_for_card,
    file_sha256,
    json_sha256,
    load_queue,
    make_freeze_identity,
    read_bundle,
    run_pilot,
    write_json_exclusive,
)
from src.models.optimization_v3_batch1_r17 import (
    _array_digest,
    CARD_ID,
    FEATURE_NAMES,
    FULL_CUTOFFS,
    MIN_HISTORY_ROWS,
    PILOT_CUTOFFS,
    SCORE_CYCLES_BY_CUTOFF,
    SOURCE_CUTOFF_BY_CYCLE,
    apply_affine,
    fit_predict,
    solve_nonnegative_l1,
)
from src.utils.paths import task_files


QUEUE_PATH = ROOT / "experiments/optimization_v3/research_backlog/queue.json"
CARD_PATH = ROOT / "reports/optimization_v3/research_backlog/cards/R17.md"
EXPERIMENT_DIR = ROOT / "experiments/optimization_v3/batch1/R17"
FREEZE_PATH = EXPERIMENT_DIR / "freeze.json"
RECIPE_PATH = EXPERIMENT_DIR / "recipe.json"
RUN_DIR = ROOT / "experiments/runs/optimization_v3_batch1/R17"
REPORT_DIR = ROOT / "reports/optimization_v3/batch1/R17"
ENGINE_REPORT_PATH = REPORT_DIR / "engine_report.json"
RESULTS_PATH = REPORT_DIR / "results.json"
VERIFICATION_PATH = REPORT_DIR / "verification.json"
SUMMARY_PATH = REPORT_DIR / "summary.md"
PUBLICATION_PATH = REPORT_DIR / "publication.json"
LEDGER_PATH = ROOT / "experiments/runs/optimization_v3_batch1/batch_ledger.jsonl"

EXPECTED_CARD_FIELDS = {
    "id": "R17",
    "task": "regression",
    "name": "OOF nonnegative affine L1 calibration",
    "family": "calibration_affine",
    "priority": "P2",
    "policy": "R_CALIBRATE",
    "pilot_predictor_fits": 2,
    "evaluation": "LATE",
    "condition": "No additional dependency",
    "status": "PLANNED_NOT_RUN",
    "implementation": "NOT_IMPLEMENTED",
    "queue_order": 20,
    "card_path": "reports/optimization_v3/research_backlog/cards/R17.md",
    "in_priority_prefix": True,
}
EXPECTED_LATE_VIEWS = {
    "W4": [(67, 68, 71)],
    "W6": [(75, 76, 79)],
    "primary": [(67, 68, 71), (75, 76, 79)],
    "H12_L": [(67, 68, 79)],
}
EXPECTED_REGRESSION_GATES = {
    "pilot_primary_MAE_gain_min": 0.03,
    "pilot_H12_MAE_harm_max": 0.05,
    "individual_pilot_view_MAE_harm_max": 0.5,
    "full_primary_MAE_gain_min": 0.03,
    "full_H12_MAE_harm_max": 0.05,
    "full_nonnegative_windows_min": 4,
    "supported_group_n_min": 100,
    "supported_tail_or_missing_MAE_harm_max": 0.25,
}
EXPECTED_FOLDS = (
    (55, 56, 59), (59, 60, 63), (63, 64, 67),
    (67, 68, 71), (71, 72, 75), (75, 76, 79),
)
GIB = 1024 ** 3
MAX_JOB_RSS_BYTES = 6 * GIB
MIN_FREE_RAM_BYTES = 4 * GIB
RESERVED_SYSTEM_RAM_BYTES = 2 * GIB
ESTIMATED_ADDITIONAL_RSS_BYTES = 1 * GIB
RESOURCE_PREFLIGHT_PATH = REPORT_DIR / "resource_preflight.json"
RESOURCE_POSTFIT_PATH = REPORT_DIR / "resource_postfit.json"


class R17ResourceLimit(BatchIntegrityError):
    """Raised when the frozen V3 compute policy does not permit the LP job."""

    def __init__(self, report: Mapping[str, Any]):
        self.report = dict(report)
        super().__init__("SKIPPED_RESOURCE: R17 LP job exceeds frozen RAM policy; "
                         + json.dumps(self.report, sort_keys=True))


def _resource_snapshot() -> dict[str, Any]:
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        class ProcessMemoryCountersEx(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        memory = MemoryStatus()
        memory.dwLength = ctypes.sizeof(MemoryStatus)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MemoryStatus)]
        kernel32.GlobalMemoryStatusEx.restype = ctypes.c_int
        if not kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            raise BatchIntegrityError(f"GlobalMemoryStatusEx failed: {ctypes.get_last_error()}")
        counters = ProcessMemoryCountersEx()
        counters.cb = ctypes.sizeof(ProcessMemoryCountersEx)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(ProcessMemoryCountersEx), ctypes.c_ulong]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        process = kernel32.GetCurrentProcess()
        if not psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), ctypes.sizeof(ProcessMemoryCountersEx)):
            raise BatchIntegrityError(f"GetProcessMemoryInfo failed: {ctypes.get_last_error()}")
        total, available = int(memory.ullTotalPhys), int(memory.ullAvailPhys)
        rss, peak_rss = int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize)
        source = "Windows GlobalMemoryStatusEx/GetProcessMemoryInfo"
    else:
        import resource
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        total = int(os.sysconf("SC_PHYS_PAGES")) * page_size
        available = int(os.sysconf("SC_AVPHYS_PAGES")) * page_size
        peak_raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        peak_rss = peak_raw if platform.system() == "Darwin" else peak_raw * 1024
        rss = peak_rss
        source = "POSIX sysconf/resource (peak RSS used as conservative current RSS)"
    return {
        "measurement_source": source,
        "total_ram_bytes": total,
        "available_ram_bytes": available,
        "process_rss_bytes": rss,
        "process_peak_rss_bytes": peak_rss,
        "total_ram_gib": total / GIB,
        "available_ram_gib": available / GIB,
        "process_rss_gib": rss / GIB,
        "process_peak_rss_gib": peak_rss / GIB,
    }


def _resource_gate() -> dict[str, Any]:
    snapshot = _resource_snapshot()
    predicted_peak = int(snapshot["process_rss_bytes"]) + ESTIMATED_ADDITIONAL_RSS_BYTES
    allowance_after_system_reserve = int(snapshot["available_ram_bytes"]) - RESERVED_SYSTEM_RAM_BYTES
    passed = bool(
        int(snapshot["available_ram_bytes"]) >= MIN_FREE_RAM_BYTES
        and ESTIMATED_ADDITIONAL_RSS_BYTES <= allowance_after_system_reserve
        and predicted_peak <= MAX_JOB_RSS_BYTES
    )
    result = {
        **snapshot,
        "policy": {
            "minimum_free_ram_bytes": MIN_FREE_RAM_BYTES,
            "reserved_system_ram_bytes": RESERVED_SYSTEM_RAM_BYTES,
            "maximum_job_rss_bytes": MAX_JOB_RSS_BYTES,
            "estimated_additional_lp_rss_bytes": ESTIMATED_ADDITIONAL_RSS_BYTES,
            "estimate_basis": "conservative 1 GiB allowance for sparse HiGHS LP, above the estimated sub-10 MiB R17 matrix storage",
        },
        "predicted_process_peak_rss_bytes": predicted_peak,
        "predicted_process_peak_rss_gib": predicted_peak / GIB,
        "additional_ram_allowance_after_reserve_bytes": allowance_after_system_reserve,
        "status": "PASS" if passed else "SKIPPED_RESOURCE",
    }
    if not passed:
        raise R17ResourceLimit(result)
    return result


def _validate_r17_card(card: Mapping[str, Any], queue: Mapping[str, Any]) -> None:
    cards = {str(item["id"]): item for item in queue["cards"]}
    if cards.get(CARD_ID) != dict(card):
        raise BatchIntegrityError("Dispatcher R17 card differs from canonical queue entry")
    for name, expected in EXPECTED_CARD_FIELDS.items():
        if card.get(name) != expected:
            raise BatchIntegrityError(f"Canonical R17 card field changed: {name}")
    if card.get("change") != (
        "On exact champion prequential OOF56..cutoff, solve min sum|y-a-b*p| with b>=0 inside LP; "
        "use returned a,b and clip0..100. Require5000 OOF rows. If solver fails fallback incumbent. "
        "No fit51 champion fabrication; early cutoffs fallback."
    ):
        raise BatchIntegrityError("Canonical R17 calibration policy changed")
    if card.get("preflight_checks") != [
        "Strict OOF score chronology; evaluate67/75 W4+W6 and H12_L, early H12 unchanged"
    ]:
        raise BatchIntegrityError("Canonical R17 preflight checks changed")
    if card.get("evidence") != [
        "reports/optimization_v3/regression_residual/verification.json",
        "reports/optimization_v3/regression_diagnostics/diagnostics.json",
    ]:
        raise BatchIntegrityError("Canonical R17 evidence list changed")
    if not CARD_PATH.is_file() or "# R17" not in CARD_PATH.read_text(encoding="utf-8"):
        raise BatchIntegrityError("R17 human-readable card is missing or malformed")
    if queue["baseline"]["regression"].get("Train_sha256") != r18.TRAIN_SHA256:
        raise BatchIntegrityError("Canonical Regression Train hash changed")
    if dict(queue["evaluation"]["regression_gate"]) != EXPECTED_REGRESSION_GATES:
        raise BatchIntegrityError("Canonical regression evaluation gates changed")
    if queue["evaluation"].get("late_gate_note") != (
        "Apply same aggregate delta thresholds to explicitly labelled W4+W6 and H12_L; report early fallback separately. "
        "No external promotion from late-only pilot. No downgrading gates after scores."
    ):
        raise BatchIntegrityError("Late pilot gate scope changed")


def _load_train_without_targets(expected_sha256: str) -> tuple[Path, pd.DataFrame, np.ndarray]:
    path = task_files("regression").get("train")
    if path is None or not Path(path).is_file():
        raise FileNotFoundError("Official Regression Train is missing")
    path = Path(path)
    if file_sha256(path) != expected_sha256:
        raise BatchIntegrityError("Official Regression Train hash differs from the frozen queue")
    header = pd.read_csv(path, nrows=0).columns.tolist()
    if "threat_rank" not in header or "cycle_code" not in header:
        raise BatchIntegrityError("Official Regression Train lacks target/cycle columns")
    frame = pd.read_csv(path, usecols=lambda name: name != "threat_rank")
    required = {
        "record_id", "cycle_code", "impact_load", "scope_pressure", "text_load",
        "scope_trace", "entry_friction", "org_code", "module_code", "pattern_code",
        "signal_stream",
    }
    absent = sorted(required - set(frame.columns))
    if absent or len(frame) != 50_608:
        raise BatchIntegrityError(f"Official Train schema/size mismatch: missing={absent}, rows={len(frame)}")
    if frame["record_id"].isna().any() or frame["record_id"].astype(str).str.strip().eq("").any():
        raise BatchIntegrityError("Official Train record_id is missing or blank")
    if frame["record_id"].astype(str).duplicated().any():
        raise BatchIntegrityError("Official Train record_id is not unique")
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    if not np.array_equal(np.unique(cycles), np.arange(47, 80, dtype=np.int64)):
        raise BatchIntegrityError("Official Train cycle coverage differs from 47–79")
    frame["threat_rank"] = np.nan
    return path, frame, cycles


def _source_files() -> list[Path]:
    return [
        *r18._source_files(), Path(__file__),
        ROOT / "src/models/optimization_v3_batch1_r17.py",
        ROOT / "tests/test_optimization_v3_batch1_r17.py",
    ]


def _input_files(train_path: Path) -> dict[str, Path]:
    return {**r18._input_files(train_path), "r17_card": CARD_PATH}


def _validate_primary_oof_without_labels(
    *, frame: pd.DataFrame, cycles: np.ndarray, diagnostics: Mapping[str, Any],
    oof: Mapping[str, np.ndarray], sparse_cache: Mapping[str, np.ndarray],
    base_cache: Mapping[str, np.ndarray],
) -> tuple[dict[int, int], list[dict[str, Any]]]:
    rows = np.asarray(oof["primary_rows"], dtype=np.int64)
    oof_cycles = np.asarray(oof["primary_cycle"], dtype=np.int64)
    prediction = np.asarray(oof["primary_prediction"], dtype=np.float64)
    expected_rows = np.flatnonzero((cycles >= 56) & (cycles <= 79))
    if (rows.ndim != 1 or len(rows) != len(np.unique(rows))
            or not np.array_equal(np.sort(rows), expected_rows)
            or not np.array_equal(oof_cycles, cycles[rows])):
        raise BatchIntegrityError("R17 champion OOF rows/cycles do not exactly map to official Train")
    if (prediction.shape != rows.shape or not np.isfinite(prediction).all()
            or np.any((prediction < 0.0) | (prediction > 100.0))):
        raise BatchIntegrityError("R17 champion OOF predictions are invalid")
    audits = diagnostics.get("fit_and_row_audits", {}).get("primary", [])
    if len(audits) != len(EXPECTED_FOLDS):
        raise BatchIntegrityError("R17 diagnostics do not contain the six expected primary OOF folds")
    position = {int(row): index for index, row in enumerate(rows)}
    source_by_cycle: dict[int, int] = {}
    fold_audit: list[dict[str, Any]] = []
    for (cutoff, low, high), audit in zip(EXPECTED_FOLDS, audits):
        expected_n = int(np.count_nonzero((cycles >= low) & (cycles <= high)))
        if (int(audit.get("cutoff", -1)) != cutoff
                or audit.get("view") != f"{low}-{high}"
                or int(audit.get("validation_rows", -1)) != expected_n
                or float(audit.get("base_ridge25_reconstruction_max_abs_error", float("inf"))) != 0.0):
            raise BatchIntegrityError(f"R17 diagnostic fold audit changed for cycles {low}–{high}")
        for cycle in range(low, high + 1):
            if cycle in source_by_cycle:
                raise BatchIntegrityError(f"R17 source cutoff duplicates cycle {cycle}")
            source_by_cycle[cycle] = cutoff
        block, _ = r18._make_view(
            frame, cutoff, low, high, sparse_cache, base_cache,
            view_name=f"R17 exact OOF source {low}-{high}",
        )
        block_rows = block["row"].to_numpy(dtype=np.int64)
        block_positions = np.asarray([position[int(row)] for row in block_rows], dtype=np.int64)
        if not np.array_equal(block_rows, np.flatnonzero((cycles >= low) & (cycles <= high))):
            raise BatchIntegrityError(f"R17 exact fold rows differ for cycles {low}–{high}")
        if not np.array_equal(prediction[block_positions], block["prediction"].to_numpy(dtype=np.float64)):
            raise BatchIntegrityError(f"R17 exact champion OOF reconstruction differs for cycles {low}–{high}")
        for key, component in (
            ("primary_lgb_svd32", "lgb_svd32"),
            ("primary_hybrid_ridge", "hybrid_ridge"),
            ("primary_sparse_lgb_rank50", "sparse_lgb_rank50"),
        ):
            cached = np.asarray(oof[key], dtype=np.float64)
            if cached.shape != rows.shape or not np.array_equal(
                cached[block_positions], block[component].to_numpy(dtype=np.float64)
            ):
                raise BatchIntegrityError(f"R17 OOF component {key} differs for cutoff {cutoff}")
        fold_audit.append({
            "source_cutoff": cutoff, "scored_cycles": [low, high],
            "rows": expected_n, "champion_and_components_exact": True,
        })
    if source_by_cycle != SOURCE_CUTOFF_BY_CYCLE:
        raise BatchIntegrityError("R17 primary OOF source mapping differs from its canonical six folds")
    return source_by_cycle, fold_audit


def _history_by_cutoff(
    *, cycles: np.ndarray, oof: Mapping[str, np.ndarray],
) -> dict[int, dict[str, np.ndarray]]:
    oof_rows = np.asarray(oof["primary_rows"], dtype=np.int64)
    oof_cycles = np.asarray(oof["primary_cycle"], dtype=np.int64)
    oof_prediction = np.asarray(oof["primary_prediction"], dtype=np.float64)
    position = {int(row): index for index, row in enumerate(oof_rows)}
    output: dict[int, dict[str, np.ndarray]] = {}
    for cutoff in FULL_CUTOFFS:
        if cutoff == 55:
            output[cutoff] = {
                "rows": np.empty(0, dtype=np.int64),
                "cycles": np.empty(0, dtype=np.int64),
                "prediction": np.empty(0, dtype=np.float64),
            }
            continue
        rows = np.flatnonzero((cycles >= 56) & (cycles <= cutoff)).astype(np.int64)
        positions = np.asarray([position.get(int(row), -1) for row in rows], dtype=np.int64)
        if np.any(positions < 0):
            raise BatchIntegrityError(f"R17 OOF history has missing exact Train rows at cutoff {cutoff}")
        history_cycles = cycles[rows]
        if not np.array_equal(oof_cycles[positions], history_cycles):
            raise BatchIntegrityError(f"R17 OOF history row-cycle mapping differs at cutoff {cutoff}")
        expected_cycle = np.arange(56, cutoff + 1, dtype=np.int64)
        if not np.array_equal(np.unique(history_cycles), expected_cycle):
            raise BatchIntegrityError(f"R17 OOF history lacks complete cycles at cutoff {cutoff}")
        output[cutoff] = {
            "rows": rows,
            "cycles": history_cycles.astype(np.int64, copy=True),
            "prediction": oof_prediction[positions].astype(np.float64, copy=True),
        }
    return output


def _exact_control_blocks(
    *, frame: pd.DataFrame, cycles: np.ndarray, sparse_cache: Mapping[str, np.ndarray],
    base_cache: Mapping[str, np.ndarray], oof: Mapping[str, np.ndarray],
) -> dict[int, dict[str, Any]]:
    blocks: dict[int, dict[str, Any]] = {}
    for cutoff, low, high in EXPECTED_FOLDS:
        control = r18._reconstruct_control(
            frame=frame, cycles=cycles, cutoff=cutoff, low=low, high=high,
            sparse_cache=sparse_cache, base_cache=base_cache, oof=oof,
            include_arrays=True,
        )
        blocks[cutoff] = control["_arrays"]
    # The LATE pilot uses a cutoff-67 forecast across cycles 68–79 and a
    # cutoff-75 forecast across cycles 76–79, while the fold controls above
    # retain the non-overlapping four-cycle source windows for full-six.
    for cutoff, low, high in ((55, 56, 67), (67, 68, 79), (75, 76, 79)):
        control = r18._reconstruct_control(
            frame=frame, cycles=cycles, cutoff=cutoff, low=low, high=high,
            sparse_cache=sparse_cache, base_cache=base_cache, oof=oof,
            include_arrays=True,
        )
        blocks[cutoff] = control["_arrays"]
    return blocks


def _prepare_preflight(
    card: Mapping[str, Any], queue: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    r18_card = next((item for item in queue["cards"] if item["id"] == "R18"), None)
    if r18_card is None:
        raise BatchIntegrityError("Canonical queue lacks the R18 lineage validator card")
    r18._validate_card_and_queue(r18_card, queue)
    _validate_r17_card(card, queue)
    views = evaluation_views_for_card(queue, CARD_ID)
    if views != EXPECTED_LATE_VIEWS:
        raise BatchIntegrityError(f"Canonical R17 LATE evaluation views changed: {views}")
    if len(CARD_PATH.read_text(encoding="utf-8")) < 500:
        raise BatchIntegrityError("R17 human-readable card is unexpectedly short")

    train_path, frame, cycles = _load_train_without_targets(r18.TRAIN_SHA256)
    input_files = _input_files(train_path)
    input_hashes, evidence = r18._validate_evidence_hashes(queue, input_files)
    base_cache = r18._load_archive(r18.PRIMARY_CACHE)
    sparse_cache = {
        **r18._load_archive(r18.PILOT_CACHE),
        **r18._load_archive(r18.FULL_CACHE),
    }
    oof = r18._load_oof_predictions_only(r18.OOF_CACHE_PATH)
    source_by_cycle, fold_audit = _validate_primary_oof_without_labels(
        frame=frame, cycles=cycles, diagnostics=evidence["diagnostics"],
        oof=oof, sparse_cache=sparse_cache, base_cache=base_cache,
    )
    histories = _history_by_cutoff(cycles=cycles, oof=oof)
    controls = _exact_control_blocks(
        frame=frame, cycles=cycles, sparse_cache=sparse_cache,
        base_cache=base_cache, oof=oof,
    )
    support = {
        str(cutoff): {
            "rows": int(len(history["rows"])),
            "cycles": [int(x) for x in np.unique(history["cycles"])],
            "complete_cycle_support": True,
            "solver_minimum_met": bool(len(history["rows"]) >= MIN_HISTORY_ROWS),
            "fallback": cutoff == 55 or len(history["rows"]) < MIN_HISTORY_ROWS,
        }
        for cutoff, history in histories.items()
    }
    observed_input_hashes = {name: file_sha256(path) for name, path in sorted(input_files.items())}
    source_hashes = {str(path.relative_to(ROOT)): file_sha256(path) for path in _source_files()}
    if observed_input_hashes != input_hashes:
        raise BatchIntegrityError("An R17 input changed during read-only preflight")
    if {str(path.relative_to(ROOT)): file_sha256(path) for path in _source_files()} != source_hashes:
        raise BatchIntegrityError("An R17 source changed during read-only preflight")
    exact_controls = {
        str(cutoff): {
            "rows": int(len(block["rows"])),
            "row_positions_sha256": r18._array_sha256(block["rows"]),
            "cycles_sha256": r18._array_sha256(block["cycle"]),
            "champion_prediction_sha256": r18._array_sha256(block["baseline_prediction"]),
            "record_id_sha256": r18._array_sha256(block["record_id"]),
            "row_exact": True,
        }
        for cutoff, block in controls.items()
    }
    preflight = {
        "card_id": CARD_ID,
        "status": "READ_ONLY_PREFLIGHT_PASS",
        "scope": "official Train schema, frozen provenance, label-free champion OOF map and exact controls; no R17 solve or score metric",
        "queue_sha256": file_sha256(QUEUE_PATH),
        "r17_card_sha256": file_sha256(CARD_PATH),
        "train_sha256": file_sha256(train_path),
        "input_sha256": input_hashes,
        "source_sha256": source_hashes,
        "accepted_manifest_sha256": input_hashes["accepted_regression_manifest"],
        "diagnostics_sha256": input_hashes["cache_only_diagnostics"],
        "oof_cache_sha256": input_hashes["current_champion_oof_cache"],
        "oof_primary_rows": int(len(oof["primary_rows"])),
        "oof_target_arrays_opened": False,
        "official_train_target_column_read": False,
        "test_or_sample_accessed": False,
        "source_cutoff_by_cycle": {str(k): int(v) for k, v in sorted(source_by_cycle.items())},
        "fold_reconstruction": fold_audit,
        "history_support": support,
        "exact_control_blocks": exact_controls,
        "evaluation_views": {name: [list(segment) for segment in segments] for name, segments in views.items()},
        "pilot_solver_calls": 2,
        "preprocessing_fits": 0,
        "candidate_metric_computed": False,
        "no_target_score_labels_read": True,
        "python_version": platform.python_version(),
    }
    state = {
        "train_path": train_path, "frame": frame, "cycles": cycles,
        "input_files": input_files, "input_sha256": input_hashes,
        "source_sha256": source_hashes, "evidence": evidence,
        "oof": oof, "histories": histories,
        "source_cutoff_by_cycle": source_by_cycle,
        "controls": controls, "views": views,
    }
    return preflight, state


def _recipe_document(card: Mapping[str, Any], preflight: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "card_id": CARD_ID,
        "card_sha256": preflight["r17_card_sha256"],
        "queue_sha256": preflight["queue_sha256"],
        "calibration": "minimize sum(abs(y - intercept - slope * champion_oof_prediction)); slope >= 0",
        "solver": {
            "implementation": "scipy.optimize.linprog(method=highs)",
            "sparse_constraints": True, "threads": 2,
            "variables": "intercept (free), slope (nonnegative), one nonnegative absolute residual per history row",
            "failure_policy": "exact champion identity fallback",
        },
        "history_policy": {
            "exact_champion_oof_cycles": "56..outer cutoff inclusive",
            "minimum_rows": MIN_HISTORY_ROWS,
            "source_cutoff_by_cycle": preflight["source_cutoff_by_cycle"],
            "support_by_cutoff": preflight["history_support"],
            "no_champion_oof_fabrication_for_cycles_52_55": True,
        },
        "prediction_policy": {
            "future_prediction": "clip(intercept + slope * exact champion prediction, 0, 100)",
            "fallback": "return exact champion predictions unchanged",
            "pilot_cutoffs": list(PILOT_CUTOFFS),
            "full_six_cutoffs": list(FULL_CUTOFFS),
        },
        "pilot": {
            "evaluation": "LATE: W4+W6 primary, H12_L and separate exact-champion H12_E",
            "views": preflight["evaluation_views"],
            "solver_calls": 2,
            "candidate_metric_freeze": True,
            "future_score_labels_passed_to_adapter": False,
        },
        "full_six_continuation_if_pilot_gate_passes": {
            "windows": {"W1": [56, 59], "W2": [60, 63], "W3": [64, 67], "W4": [68, 71], "W5": [72, 75], "W6": [76, 79]},
            "calibration_solve_cutoffs": [63, 71],
            "reuse_frozen_pilot_calibrations": [67, 75],
            "cutoff_55_fallback": True,
            "cutoff_59_fallback_below_minimum": True,
            "additional_solve_count": 2,
            "no_external_promotion_from_late_pilot": True,
        },
        "gates": EXPECTED_REGRESSION_GATES,
        "resource_policy": {
            "minimum_free_ram_gib": MIN_FREE_RAM_BYTES / GIB,
            "system_reserve_gib": RESERVED_SYSTEM_RAM_BYTES / GIB,
            "maximum_job_rss_gib": MAX_JOB_RSS_BYTES / GIB,
            "estimated_additional_solver_rss_gib": ESTIMATED_ADDITIONAL_RSS_BYTES / GIB,
            "dense_lp_storage_forbidden": True,
            "sparse_lp_matrix_bytes_upper_estimate_at_max_history_rows_30097": int(
                6 * 30097 * 8 + 6 * 30097 * 8 + (2 * 30097 + 1) * 8
            ),
        },
        "test_or_sample_used": False,
        "official_train_target_values_in_recipe": False,
        "preprocessing_fits": 0,
    }


def _freeze_identity(card: Mapping[str, Any], preflight: Mapping[str, Any], state: Mapping[str, Any]) -> dict[str, Any]:
    identity = make_freeze_identity(
        card=card, feature_names=FEATURE_NAMES,
        input_files=state["input_files"], source_files=_source_files(),
        evaluation_views=state["views"], packages=["numpy", "pandas", "scipy"],
    )
    identity["recipe"] = _recipe_document(card, preflight)
    identity["r17_preflight"] = {
        "oof_primary_rows": preflight["oof_primary_rows"],
        "source_cutoff_by_cycle": preflight["source_cutoff_by_cycle"],
        "fold_reconstruction": preflight["fold_reconstruction"],
        "history_support": preflight["history_support"],
        "exact_control_blocks": preflight["exact_control_blocks"],
        "target_column_read": False,
        "oof_target_arrays_opened": False,
        "test_or_sample_accessed": False,
    }
    return identity


def _preflight(card: Mapping[str, Any], queue: Mapping[str, Any]) -> dict[str, Any]:
    result, _ = _prepare_preflight(card, queue)
    return result


def _freeze_only(card: Mapping[str, Any], queue: Mapping[str, Any]) -> dict[str, Any]:
    preflight, state = _prepare_preflight(card, queue)
    identity = _freeze_identity(card, preflight, state)
    signature = json_sha256(identity)
    recipe = _recipe_document(card, preflight)
    recipe_path = RECIPE_PATH
    if recipe_path.exists():
        if json.loads(recipe_path.read_text(encoding="utf-8")) != recipe:
            raise BatchIntegrityError("Existing R17 recipe differs from the score-blind preflight")
    else:
        write_json_exclusive(recipe_path, recipe)
    if FREEZE_PATH.exists():
        existing = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
        if existing.get("signature") != signature or existing.get("identity") != identity:
            raise BatchIntegrityError("Existing R17 freeze differs from current sources, inputs, or recipe")
    else:
        write_json_exclusive(FREEZE_PATH, {
            "schema_version": 1, "card_id": CARD_ID,
            "identity": identity, "signature": signature,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        })
    return {
        "card_id": CARD_ID, "status": "FROZEN_BEFORE_R17_CALIBRATION",
        "freeze_path": str(FREEZE_PATH.relative_to(ROOT)),
        "recipe_path": str(recipe_path.relative_to(ROOT)),
        "signature": signature, "freeze_sha256": file_sha256(FREEZE_PATH),
        "candidate_metrics_computed": False,
        "official_train_target_column_read": False,
        "test_or_sample_accessed": False,
    }


def _read_full_train_target_after_freeze(train_path: Path, expected_sha256: str, rows: int) -> np.ndarray:
    if file_sha256(train_path) != expected_sha256:
        raise BatchIntegrityError("Official Regression Train changed after R17 freeze verification")
    target = pd.read_csv(train_path, usecols=["threat_rank"])["threat_rank"].to_numpy(dtype=np.float64)
    if target.shape != (rows,) or not np.isfinite(target).all() or np.any((target < 0.0) | (target > 100.0)):
        raise BatchIntegrityError("Post-freeze official Train target column is invalid")
    if file_sha256(train_path) != expected_sha256:
        raise BatchIntegrityError("Official Regression Train changed while R17 targets were read")
    return target


def _assert_current_sources(identity: Mapping[str, Any], state: Mapping[str, Any]) -> None:
    current_inputs = {name: file_sha256(path) for name, path in sorted(state["input_files"].items())}
    current_sources = {str(path): file_sha256(path) for path in sorted(_source_files(), key=str)}
    if current_inputs != dict(identity["input_sha256"]) or current_sources != dict(identity["source_sha256"]):
        raise BatchIntegrityError("R17 frozen source/input changed; refusing to score or publish")


def _control_loader(state: Mapping[str, Any]):
    controls = state["controls"]

    def load(cutoff: int, rows: np.ndarray) -> dict[str, np.ndarray]:
        block = controls[int(cutoff)]
        if not np.array_equal(np.asarray(block["rows"], dtype=np.int64), np.asarray(rows, dtype=np.int64)):
            raise BatchIntegrityError(f"R17 control rows differ from generic harness at cutoff {cutoff}")
        return {
            "baseline_prediction": np.asarray(block["baseline_prediction"], dtype=np.float64),
            "score_cycle": np.asarray(block["cycle"], dtype=np.int64),
        }
    return load


def _verify_engine_and_score(
    *, engine: Mapping[str, Any], state: Mapping[str, Any], target: np.ndarray,
    signature: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    frame = state["frame"]
    cycles = state["cycles"]
    histories = state["histories"]
    views = state["views"]
    reconstruction_cache: dict[int, dict[str, Any]] = {}

    def reconstruct(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        score_rows = np.asarray(arrays["score_rows"], dtype=np.int64)
        score_cycles = np.asarray(arrays["control__score_cycle"], dtype=np.int64)
        if not np.array_equal(cycles[score_rows], score_cycles):
            raise BatchIntegrityError("R17 persisted score cycles do not map to official Train rows")
        cutoff = 67 if int(np.min(score_cycles)) == 68 else 75
        control = state["controls"][cutoff]
        baseline = np.asarray(arrays["baseline_prediction"], dtype=np.float64)
        if (not np.array_equal(score_rows, np.asarray(control["rows"], dtype=np.int64))
                or not np.array_equal(baseline, np.asarray(control["baseline_prediction"], dtype=np.float64))):
            raise BatchIntegrityError(f"R17 persisted baseline differs from exact frozen controls at cutoff {cutoff}")
        audit_success = np.asarray(arrays.get("audit__solver_success"), dtype=np.int8)
        audit_intercept = np.asarray(arrays.get("audit__intercept"), dtype=np.float64)
        audit_slope = np.asarray(arrays.get("audit__slope"), dtype=np.float64)
        if (audit_success.shape != baseline.shape or audit_intercept.shape != baseline.shape
                or audit_slope.shape != baseline.shape or not np.all(audit_success == audit_success[0])
                or not np.all(audit_intercept == audit_intercept[0])
                or not np.all(audit_slope == audit_slope[0])):
            raise BatchIntegrityError(f"R17 persisted coefficient audit arrays are inconsistent at cutoff {cutoff}")
        if int(audit_success[0]) == 0:
            if (float(audit_intercept[0]) != 0.0 or float(audit_slope[0]) != 1.0
                    or not np.array_equal(np.asarray(arrays["prediction"], dtype=np.float64), baseline)):
                raise BatchIntegrityError(f"R17 solver-failure fallback is not exact champion identity at cutoff {cutoff}")
            return baseline.copy(), baseline
        if cutoff not in reconstruction_cache:
            history = histories[cutoff]
            history_target = target[np.asarray(history["rows"], dtype=np.int64)]
            reconstruction_cache[cutoff] = solve_nonnegative_l1(
                history_target, history["prediction"], threads=2,
            )
        solved = reconstruction_cache[cutoff]
        if not solved["success"]:
            raise BatchIntegrityError(f"R17 independent LP verification failed at cutoff {cutoff}")
        expected = apply_affine(baseline, solved["intercept"], solved["slope"])
        for name, value in (
            ("audit__intercept", solved["intercept"]),
            ("audit__slope", solved["slope"]),
            ("audit__solver_success", int(bool(solved["success"]))),
        ):
            observed = np.asarray(arrays.get(name))
            if observed.shape != baseline.shape or not np.all(observed == value):
                raise BatchIntegrityError(f"R17 independent calibration field differs: {name} at {cutoff}")
        return expected, baseline

    generic_verification = verify_report(
        report=dict(engine), frame=frame, target=target, artifact_dir=RUN_DIR,
        freeze_signature=signature, evaluation_views=views,
        prediction_reconstructors={"R17": reconstruct},
    )

    blocks: dict[int, dict[str, Any]] = {}
    calibration_receipts: dict[str, Any] = {}
    for cutoff in (67, 75):
        path = RUN_DIR / f"R17_fit{cutoff}.npz"
        sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        bundle_identity = sidecar.get("identity", {})
        arrays, metadata = read_bundle(path, bundle_identity)
        rows = np.asarray(arrays["score_rows"], dtype=np.int64)
        score_ids = frame.iloc[rows]["record_id"].astype(str).to_numpy(dtype=str)
        if (metadata["metadata"].get("fit_predictor_count") != 1
                or not np.array_equal(np.asarray(arrays["score_record_ids"], dtype=str), score_ids)
                or not np.array_equal(np.asarray(arrays["control__score_cycle"], dtype=np.int64), cycles[rows])):
            raise BatchIntegrityError(f"R17 persisted receipt identity/row mapping failed at cutoff {cutoff}")
        state_meta = metadata["metadata"].get("state", {})
        history = histories[cutoff]
        expected_target_hash = _array_digest(target[history["rows"]])
        expected_source_cutoff = np.asarray(
            [state["source_cutoff_by_cycle"][int(cycle)] for cycle in history["cycles"]],
            dtype=np.int64,
        )
        if (state_meta.get("history_row_positions_sha256") != _array_digest(history["rows"])
                or state_meta.get("history_cycle_sha256") != _array_digest(history["cycles"])
                or state_meta.get("history_source_cutoff_sha256") != _array_digest(expected_source_cutoff)
                or state_meta.get("history_prediction_sha256") != _array_digest(history["prediction"])
                or state_meta.get("history_target_sha256") != expected_target_hash):
            raise BatchIntegrityError(f"R17 persisted calibration history lineage failed at cutoff {cutoff}")
        expected_control = state["controls"][cutoff]
        baseline_values = np.asarray(arrays["baseline_prediction"], dtype=np.float64)
        if (not np.array_equal(rows, np.asarray(expected_control["rows"], dtype=np.int64))
                or not np.array_equal(baseline_values, np.asarray(expected_control["baseline_prediction"], dtype=np.float64))
                or state_meta.get("baseline_prediction_sha256") != _array_digest(baseline_values)):
            raise BatchIntegrityError(f"R17 receipt baseline differs from frozen exact controls at cutoff {cutoff}")
        if bool(state_meta.get("success")):
            independently_solved = reconstruction_cache.get(cutoff)
            if (independently_solved is None or not independently_solved["success"]
                    or not np.isclose(float(state_meta.get("intercept", np.nan)), float(independently_solved["intercept"]), atol=1e-10, rtol=0.0)
                    or not np.isclose(float(state_meta.get("slope", np.nan)), float(independently_solved["slope"]), atol=1e-10, rtol=0.0)):
                raise BatchIntegrityError(f"R17 persisted LP coefficients/status differ at cutoff {cutoff}")
        elif (not state_meta.get("fallback")
              or float(state_meta.get("intercept", np.nan)) != 0.0
              or float(state_meta.get("slope", np.nan)) != 1.0
              or not np.array_equal(np.asarray(arrays["prediction"], dtype=np.float64), baseline_values)):
            raise BatchIntegrityError(f"R17 failed-solver receipt is not a verified identity fallback at cutoff {cutoff}")
        calibration_receipts[str(cutoff)] = {
            "state": dict(state_meta),
            "fit_seconds": float(metadata["metadata"].get("fit_seconds", 0.0)),
            "fit_predictor_count": int(metadata["metadata"]["fit_predictor_count"]),
        }
        blocks[cutoff] = {
            "cutoff": cutoff, "rows": rows,
            "cycle": cycles[rows],
            "baseline_prediction": np.asarray(arrays["baseline_prediction"], dtype=np.float64),
            "prediction": np.asarray(arrays["prediction"], dtype=np.float64),
            "record_id": score_ids,
            "state": dict(state_meta),
            "fit_predictor_count": int(metadata["metadata"]["fit_predictor_count"]),
        }

    early = state["controls"][55]
    blocks[55] = {
        "cutoff": 55,
        "rows": np.asarray(early["rows"], dtype=np.int64),
        "cycle": np.asarray(early["cycle"], dtype=np.int64),
        "baseline_prediction": np.asarray(early["baseline_prediction"], dtype=np.float64),
        "prediction": np.asarray(early["baseline_prediction"], dtype=np.float64).copy(),
        "record_id": np.asarray(early["record_id"], dtype=str),
        "state": {"status": "EXACT_CHAMPION_FALLBACK", "fit_predictor_count": 0},
        "fit_predictor_count": 0,
    }
    score_target_rows = np.flatnonzero((cycles >= 56) & (cycles <= 79))
    target_by_row = {int(row): float(target[row]) for row in score_target_rows}
    evaluation = r18._evaluate_frozen_predictions(
        blocks=blocks, cycles=cycles, frame=frame, target_by_row=target_by_row,
    )

    independent_views: dict[str, Any] = {}
    subgroup_checks: dict[str, Any] = {}
    specs = {
        "W4": (67, 68, 71), "W6": (75, 76, 79),
        "H12_L": (67, 68, 79), "H12_E": (55, 56, 67),
    }
    for name, (cutoff, low, high) in specs.items():
        block = blocks[cutoff]
        rows_all = np.asarray(block["rows"], dtype=np.int64)
        mask = (cycles[rows_all] >= low) & (cycles[rows_all] <= high)
        rows = rows_all[mask]
        y = target[rows]
        baseline = np.asarray(block["baseline_prediction"], dtype=np.float64)[mask]
        candidate = np.asarray(block["prediction"], dtype=np.float64)[mask]
        baseline_mae = float(np.abs(y - baseline).sum(dtype=np.float64) / len(rows))
        candidate_mae = float(np.abs(y - candidate).sum(dtype=np.float64) / len(rows))
        stored = evaluation["views"][name]
        if (stored["rows"] != len(rows)
                or not np.isclose(stored["baseline_mae"], baseline_mae, atol=1e-12, rtol=0.0)
                or not np.isclose(stored["candidate_mae"], candidate_mae, atol=1e-12, rtol=0.0)):
            raise BatchIntegrityError(f"R17 independent MAE reconstruction differs for {name}")
        independent_views[name] = {"rows": len(rows), "baseline_mae": baseline_mae, "candidate_mae": candidate_mae, "match": True}
        if name != "H12_E":
            group_view = {
                "rows": rows, "cycle": cycles[rows],
                "cutoff": np.full(len(rows), cutoff, dtype=np.int64),
                "baseline": baseline, "candidate": candidate,
            }
            subgroup_checks[name] = r18._verify_subgroup_table(
                view_name=name, view=group_view, frame=frame, target=y,
                stored=evaluation["subgroups"][name], support_min=100,
            )

    primary_rows = np.concatenate((
        np.flatnonzero(np.isin(cycles, np.arange(68, 72))),
        np.flatnonzero(np.isin(cycles, np.arange(76, 80))),
    )).astype(np.int64)
    primary_baseline = np.concatenate((
        blocks[67]["baseline_prediction"][(cycles[blocks[67]["rows"]] >= 68) & (cycles[blocks[67]["rows"]] <= 71)],
        blocks[75]["baseline_prediction"][(cycles[blocks[75]["rows"]] >= 76) & (cycles[blocks[75]["rows"]] <= 79)],
    ))
    primary_candidate = np.concatenate((
        blocks[67]["prediction"][(cycles[blocks[67]["rows"]] >= 68) & (cycles[blocks[67]["rows"]] <= 71)],
        blocks[75]["prediction"][(cycles[blocks[75]["rows"]] >= 76) & (cycles[blocks[75]["rows"]] <= 79)],
    ))
    if len(np.unique(primary_rows)) != len(primary_rows):
        raise BatchIntegrityError("R17 W4+W6 primary rows overlap")
    primary_y = target[primary_rows]
    primary_baseline_mae = float(np.abs(primary_y - primary_baseline).sum(dtype=np.float64) / len(primary_rows))
    primary_candidate_mae = float(np.abs(primary_y - primary_candidate).sum(dtype=np.float64) / len(primary_rows))
    primary_stored = evaluation["views"]["primary"]
    if (primary_stored["rows"] != len(primary_rows)
            or not np.isclose(primary_stored["baseline_mae"], primary_baseline_mae, atol=1e-12, rtol=0.0)
            or not np.isclose(primary_stored["candidate_mae"], primary_candidate_mae, atol=1e-12, rtol=0.0)):
        raise BatchIntegrityError("R17 independent W4+W6 primary reconstruction differs")
    independent_views["primary"] = {
        "rows": len(primary_rows), "baseline_mae": primary_baseline_mae,
        "candidate_mae": primary_candidate_mae, "match": True,
        "reconstructed_from": ["W4", "W6"],
    }

    verification = {
        "card_id": CARD_ID, "status": "PASS",
        "freeze_signature": signature,
        "generic_independent_verifier": generic_verification,
        "calibration": calibration_receipts,
        "history_lp_reconstruction": "PASS",
        "row_id_cycle_and_component_alignment": "PASS",
        "view_metrics": independent_views,
        "subgroup_denominators_and_mae": subgroup_checks,
        "gate_reconstruction": evaluation["gates"],
        "all_pilot_gates_pass": evaluation["all_pilot_gates_pass"],
        "test_or_sample_used": False,
    }
    return evaluation, verification


def _summary(results: Mapping[str, Any]) -> str:
    lines = [
        "# R17 — OOF nonnegative affine L1 calibration",
        "",
        f"Verdict: **{results['verdict']}** (late-only pilot; no external promotion).",
        f"Freeze signature: `{results['freeze_signature']}`",
        "",
        "| View | Rows | Baseline MAE | Candidate MAE | Gain (positive is better) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("primary", "W4", "W6", "H12_L", "H12_E"):
        item = results["views"][name]
        lines.append(
            f"| {name} | {item['rows']} | {item['baseline_mae']:.6f} | "
            f"{item['candidate_mae']:.6f} | {item['gain_positive_is_better']:.6f} |"
        )
    lines.extend([
        "",
        f"Independent QA: **{results['independent_verification']}**.",
        f"Calibration LP solves: {results['fit_predictor_count']}; preprocessing fits: 0.",
        "H12_E (cycles 56–67) is reported separately and uses exact champion fallback.",
        "This LATE pilot alone does not authorize full-Train inference or Kaggle promotion.",
        "",
    ])
    return "\n".join(lines)


def _write_json_idempotent(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise BatchIntegrityError(f"Existing R17 artifact differs; refusing overwrite: {path}")
    else:
        write_json_exclusive(path, value)


def _persist_resource_skip(
    *, identity: Mapping[str, Any], signature: str, freeze_sha256: str,
    resource_report: Mapping[str, Any],
) -> dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    resource_doc = {
        "card_id": CARD_ID, "freeze_signature": signature,
        "status": "SKIPPED_RESOURCE", "resource": dict(resource_report),
        "candidate_metric_computed": False, "solver_started": False,
    }
    results = {
        "schema_version": 1, "card_id": CARD_ID, "task": "regression",
        "status": "SKIPPED_RESOURCE", "verdict": "SKIPPED",
        "freeze_signature": signature, "freeze_sha256": freeze_sha256,
        "input_sha256": dict(identity["input_sha256"]),
        "source_sha256": dict(identity["source_sha256"]),
        "fit_predictor_count": 0, "preprocessing_fit_count": 0,
        "metrics": None, "gates": None,
        "reason": "RAM gate did not permit the sparse LP pilot; no candidate calibration was started.",
        "resource": dict(resource_report), "test_or_sample_used": False,
    }
    verification = {
        "card_id": CARD_ID, "status": "NOT_RUN_RESOURCE_GATE",
        "freeze_signature": signature, "candidate_metric_computed": False,
        "test_or_sample_used": False,
    }
    _write_json_idempotent(RESOURCE_PREFLIGHT_PATH, resource_doc)
    _write_json_idempotent(RESULTS_PATH, results)
    _write_json_idempotent(VERIFICATION_PATH, verification)
    summary = (
        "# R17 — OOF nonnegative affine L1 calibration\n\n"
        "Verdict: **SKIPPED_RESOURCE**. The frozen RAM policy failed before any calibration solve or candidate metric.\n"
        f"Freeze signature: `{signature}`\n\n"
        f"Available RAM: {resource_report.get('available_ram_gib', 'unknown')} GiB; "
        f"current process RSS: {resource_report.get('process_rss_gib', 'unknown')} GiB.\n"
    )
    if SUMMARY_PATH.exists():
        if SUMMARY_PATH.read_text(encoding="utf-8") != summary:
            raise BatchIntegrityError("Existing R17 resource-skip summary differs; refusing overwrite")
    else:
        r18._write_text_exclusive(SUMMARY_PATH, summary)
    published = [FREEZE_PATH, RECIPE_PATH, RESOURCE_PREFLIGHT_PATH, RESULTS_PATH, VERIFICATION_PATH, SUMMARY_PATH]
    manifest = {
        "schema_version": 1, "card_id": CARD_ID, "status": "COMPLETE",
        "verdict": "SKIPPED_RESOURCE", "freeze_signature": signature,
        "files": {
            str(path.relative_to(ROOT)).replace("\\", "/"): file_sha256(path)
            for path in sorted(published, key=str)
        },
        "results_sha256": file_sha256(RESULTS_PATH),
        "verification_sha256": file_sha256(VERIFICATION_PATH),
        "summary_sha256": file_sha256(SUMMARY_PATH),
    }
    _write_json_idempotent(PUBLICATION_PATH, manifest)
    return {
        "card_id": CARD_ID, "status": "SKIPPED_RESOURCE", "verdict": "SKIPPED",
        "results_path": str(RESULTS_PATH.relative_to(ROOT)),
        "summary_path": str(SUMMARY_PATH.relative_to(ROOT)),
    }


def _publish_results(
    *, engine: Mapping[str, Any], evaluation: Mapping[str, Any],
    verification: Mapping[str, Any], identity: Mapping[str, Any],
    signature: str, freeze_sha256: str,
    resource_preflight: Mapping[str, Any], resource_postfit: Mapping[str, Any],
) -> dict[str, Any]:
    gates_pass = bool(evaluation["all_pilot_gates_pass"])
    resource_pass = bool(resource_postfit.get("job_peak_rss_within_cap", False))
    calibration: dict[str, Any] = {}
    for cutoff in PILOT_CUTOFFS:
        calibration[str(cutoff)] = dict(verification["calibration"][str(cutoff)])
    results = {
        "schema_version": 1, "card_id": CARD_ID, "task": "regression",
        "status": "PILOT_VERIFIED",
        "freeze_signature": signature, "freeze_sha256": freeze_sha256,
        "input_sha256": dict(identity["input_sha256"]),
        "source_sha256": dict(identity["source_sha256"]),
        "evaluation_profile": evaluation["evaluation_profile"],
        "fit_predictor_count": int(engine["fit_predictor_count"]),
        "fit_predictor_count_matches_card": int(engine["fit_predictor_count"]) == 2,
        "preprocessing_fit_count": 0,
        "solver_calls": int(engine["fit_predictor_count"]),
        "calibration": calibration,
        "views": evaluation["views"], "subgroups": evaluation["subgroups"],
        "subgroup_gate_failures": evaluation["subgroup_gate_failures"],
        "support_rule": evaluation["support_rule"],
        "gates": evaluation["gates"],
        "all_pilot_gates_pass": gates_pass,
        "verdict": "ADVANCE" if gates_pass and resource_pass else ("RETIRE" if not gates_pass else "REWORK"),
        "promotion_limit": evaluation["promotion_limit"],
        "fallback_coverage": {
            "H12_E": {
                "rows": int(evaluation["views"]["H12_E"]["rows"]),
                "exact_champion_rows": int(evaluation["views"]["H12_E"]["rows"]),
                "coverage": 1.0, "gate_applies": False,
            },
            "history_below_minimum_or_solver_failure": {
                str(cutoff): {
                    "history_rows": int(calibration[str(cutoff)]["state"]["history_rows"]),
                    "fallback": bool(calibration[str(cutoff)]["state"]["fallback"]),
                    "reason": calibration[str(cutoff)]["state"]["fallback_reason"],
                }
                for cutoff in PILOT_CUTOFFS
            },
        },
        "fit_accounting": {
            "attempted_lp_calibrations": int(engine["fit_predictor_count"]),
            "successful_lp_calibrations": int(sum(bool(calibration[str(c)]["state"]["success"]) for c in PILOT_CUTOFFS)),
            "fit_seconds_by_cutoff": {c: calibration[c]["fit_seconds"] for c in calibration},
            "peak_process_memory_measured": True,
            "resource_preflight": dict(resource_preflight),
            "resource_postfit": dict(resource_postfit),
            "resource_policy_pass": resource_pass,
            "constraint_shapes": {c: calibration[c]["state"]["constraint_shape"] for c in calibration},
            "constraint_nnz": {c: calibration[c]["state"]["constraint_nnz"] for c in calibration},
            "csr_bytes": {c: calibration[c]["state"]["constraint_csr_bytes"] for c in calibration},
        },
        "independent_verification": verification["status"],
        "test_or_sample_used": False,
        "OOF_target_arrays_opened": False,
        "future_score_labels_passed_to_adapter": False,
        "official_train_target_loaded_after_freeze": True,
        "outcome_reuse_warning": "Temporal development outcomes are reused; the late-only pilot is not an untouched holdout.",
    }
    return results


def _run_frozen(card: Mapping[str, Any], queue: Mapping[str, Any]) -> dict[str, Any]:
    preflight, state = _prepare_preflight(card, queue)
    identity = _freeze_identity(card, preflight, state)
    signature = json_sha256(identity)
    if not FREEZE_PATH.is_file():
        raise BatchIntegrityError("R17 --run-frozen requires a prior --freeze-only artifact")
    freeze_document = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if (freeze_document.get("signature") != signature
            or freeze_document.get("identity") != identity):
        raise BatchIntegrityError("R17 freeze does not match current source/input/recipe identity")
    freeze_sha256 = file_sha256(FREEZE_PATH)
    if PUBLICATION_PATH.exists():
        publication = json.loads(PUBLICATION_PATH.read_text(encoding="utf-8"))
        for relative, expected in publication.get("files", {}).items():
            path = ROOT / relative
            if not path.is_file() or file_sha256(path) != expected:
                raise BatchIntegrityError(f"Published R17 artifact hash differs: {relative}")
        return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))

    _assert_current_sources(identity, state)
    try:
        current_resource_preflight = _resource_gate()
    except R17ResourceLimit as exc:
        if RUN_DIR.exists() or ENGINE_REPORT_PATH.exists():
            raise BatchIntegrityError(
                "R17 has a prior partial/completed fit receipt; current RAM gate blocks safe verification/resume. "
                + str(exc)
            ) from exc
        return _persist_resource_skip(
            identity=identity, signature=signature, freeze_sha256=freeze_sha256,
            resource_report=exc.report,
        )
    resource_preflight_doc = {
        "card_id": CARD_ID, "freeze_signature": signature,
        "status": "PASS", "resource": current_resource_preflight,
        "before_candidate_fit": True,
    }
    if RESOURCE_PREFLIGHT_PATH.exists():
        saved_resource = json.loads(RESOURCE_PREFLIGHT_PATH.read_text(encoding="utf-8"))
        if (saved_resource.get("freeze_signature") != signature
                or saved_resource.get("status") != "PASS"):
            raise BatchIntegrityError("Existing R17 resource preflight is not bound to this freeze")
        resource_preflight_doc = saved_resource
    else:
        _write_json_idempotent(RESOURCE_PREFLIGHT_PATH, resource_preflight_doc)
    target = _read_full_train_target_after_freeze(
        state["train_path"], identity["input_sha256"]["official_regression_train"], len(state["frame"]),
    )
    history_partial = {
        cutoff: history for cutoff, history in state["histories"].items()
    }
    adapter = __import__("functools").partial(
        fit_predict, history_by_cutoff=history_partial,
        source_cutoff_by_cycle=state["source_cutoff_by_cycle"],
        cycle_by_row=state["cycles"],
    )
    if ENGINE_REPORT_PATH.is_file():
        engine = json.loads(ENGINE_REPORT_PATH.read_text(encoding="utf-8"))
        if engine.get("freeze_signature") != signature or engine.get("card_id") != CARD_ID:
            raise BatchIntegrityError("Existing R17 engine report is not bound to the current freeze")
    else:
        engine = run_pilot(
            card=card, artifact_dir=RUN_DIR, budget_ledger_path=LEDGER_PATH,
            max_successful_fits=180, frame=state["frame"], target=target,
            feature_names=FEATURE_NAMES, adapter=adapter,
            control_loader=_control_loader(state), freeze_identity=identity,
            freeze_path=FREEZE_PATH,
            event_path=REPORT_DIR / "attempts.jsonl",
            evaluation_views=state["views"], threads=2,
            report_path=ENGINE_REPORT_PATH,
        )
    if not bool(engine.get("fit_count_matches_card")) or int(engine.get("fit_predictor_count", -1)) != 2:
        raise BatchIntegrityError("R17 pilot did not execute exactly the frozen two LP calibration solves")
    evaluation, verification = _verify_engine_and_score(
        engine=engine, state=state, target=target, signature=signature,
    )
    if verification["status"] != "PASS":
        raise BatchIntegrityError("R17 independent verification did not pass")
    post_snapshot = _resource_snapshot()
    resource_postfit_doc = {
        "card_id": CARD_ID, "freeze_signature": signature,
        "status": "PASS" if int(post_snapshot["process_peak_rss_bytes"]) <= MAX_JOB_RSS_BYTES else "JOB_RSS_CAP_EXCEEDED",
        "resource": post_snapshot,
        "job_peak_rss_within_cap": int(post_snapshot["process_peak_rss_bytes"]) <= MAX_JOB_RSS_BYTES,
        "maximum_job_rss_bytes": MAX_JOB_RSS_BYTES,
        "includes_candidate_and_independent_verification": True,
    }
    if RESOURCE_POSTFIT_PATH.exists():
        saved_postfit = json.loads(RESOURCE_POSTFIT_PATH.read_text(encoding="utf-8"))
        if saved_postfit.get("freeze_signature") != signature:
            raise BatchIntegrityError("Existing R17 postfit resource report has a different freeze")
        resource_postfit_doc = saved_postfit
    else:
        _write_json_idempotent(RESOURCE_POSTFIT_PATH, resource_postfit_doc)
    verification["resource_preflight"] = resource_preflight_doc
    verification["resource_postfit"] = resource_postfit_doc
    verification["resource_gate_status"] = resource_postfit_doc["status"]
    _assert_current_sources(identity, state)
    if file_sha256(FREEZE_PATH) != freeze_sha256:
        raise BatchIntegrityError("R17 freeze changed while the pilot was running")

    results = _publish_results(
        engine=engine, evaluation=evaluation, verification=verification,
        identity=identity, signature=signature, freeze_sha256=freeze_sha256,
        resource_preflight=resource_preflight_doc["resource"],
        resource_postfit=resource_postfit_doc,
    )
    verification_doc = {
        "card_id": CARD_ID, "status": "PASS",
        "freeze_signature": signature, "freeze_sha256": freeze_sha256,
        "checks": verification,
    }
    _write_json_idempotent(RESULTS_PATH, results)
    _write_json_idempotent(VERIFICATION_PATH, verification_doc)
    summary_text = _summary(results)
    if SUMMARY_PATH.exists():
        if SUMMARY_PATH.read_text(encoding="utf-8") != summary_text:
            raise BatchIntegrityError("Existing R17 summary differs; refusing overwrite")
    else:
        r18._write_text_exclusive(SUMMARY_PATH, summary_text)

    published = [
        FREEZE_PATH, RECIPE_PATH, ENGINE_REPORT_PATH,
        RESOURCE_PREFLIGHT_PATH, RESOURCE_POSTFIT_PATH,
        RUN_DIR / "R17_fit67.npz", RUN_DIR / "R17_fit67.json",
        RUN_DIR / "R17_fit75.npz", RUN_DIR / "R17_fit75.json",
        RESULTS_PATH, VERIFICATION_PATH, SUMMARY_PATH,
    ]
    manifest = {
        "schema_version": 1, "card_id": CARD_ID,
        "status": "COMPLETE", "verdict": results["verdict"],
        "freeze_signature": signature,
        "files": {
            str(path.relative_to(ROOT)).replace("\\", "/"): file_sha256(path)
            for path in sorted(published, key=str)
        },
        "results_sha256": file_sha256(RESULTS_PATH),
        "verification_sha256": file_sha256(VERIFICATION_PATH),
        "summary_sha256": file_sha256(SUMMARY_PATH),
    }
    _write_json_idempotent(PUBLICATION_PATH, manifest)
    return {
        "card_id": CARD_ID, "status": "PILOT_VERIFIED",
        "verdict": results["verdict"],
        "all_pilot_gates_pass": results["all_pilot_gates_pass"],
        "fit_predictor_count": results["fit_predictor_count"],
        "results_path": str(RESULTS_PATH.relative_to(ROOT)),
        "verification_path": str(VERIFICATION_PATH.relative_to(ROOT)),
        "summary_path": str(SUMMARY_PATH.relative_to(ROOT)),
        "publication_path": str(PUBLICATION_PATH.relative_to(ROOT)),
    }


def run_card(card: Mapping[str, Any], queue: Mapping[str, Any], *, preflight_only: bool = False) -> dict[str, Any]:
    if preflight_only:
        return _preflight(card, queue)
    raise BatchIntegrityError("Use the R17 runner with --preflight-only, --freeze-only, or --run-frozen")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight-only", action="store_true")
    modes.add_argument("--freeze-only", action="store_true")
    modes.add_argument("--run-frozen", action="store_true")
    args = parser.parse_args()
    queue = load_queue(QUEUE_PATH)
    card = next(item for item in queue["cards"] if item["id"] == CARD_ID)
    if args.preflight_only:
        result = _preflight(card, queue)
    elif args.freeze_only:
        result = _freeze_only(card, queue)
    else:
        result = _run_frozen(card, queue)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
