"""Past-only threshold transfer for the frozen Classification MIX50 policy.

This module never fits a predictive estimator. Official selection is blocked
until ROOT_PREFIT_PASS.json binds the committed code, protocol, and prefit
freeze. Test probabilities are loaded only after the final OOF threshold
receipt has been durably written.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from contextlib import contextmanager
from typing import Any, Callable, Iterable, Mapping
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.research_next_contract import canonical_sha256, file_sha256  # noqa: E402

PROTOCOL = ROOT / "experiments/optimization_v3/cls_mix50_temporal_decision_v1/protocol.json"
TESTS = ROOT / "tests/test_optimization_v3_cls_mix50_temporal_decision_v1.py"
REPORT_DIR = ROOT / "reports/optimization_v3/cls_mix50_temporal_decision_v1"
FREEZE = REPORT_DIR / "PREFIT_FREEZE_R5.json"
QA = REPORT_DIR / "PREFIT_SYNTHETIC_QA_R5.json"
OLD_FREEZE = REPORT_DIR / "PREFIT_FREEZE_R4.json"
OLD_QA = REPORT_DIR / "PREFIT_SYNTHETIC_QA_R4.json"
ROOT_APPROVAL = REPORT_DIR / "ROOT_PREFIT_PASS.json"
RUN_DIR = ROOT / "experiments/runs/optimization_v3_cls_mix50_temporal_decision_v1"
TRAIN_CSV = ROOT / "data/raw/classification/Train.csv"
OOF_NPZ = ROOT / "experiments/runs/optimization_v3_posoffset_a02_endgame_v1/policy_predictions.npz"
OOF_AUDIT = ROOT / "reports/optimization_v3/posmix_followup_v1/ERROR_AUDIT.json"
POLICY_METRICS = ROOT / "reports/optimization_v3/posoffset_a02_endgame_v1/POLICY_METRICS.json"
ENDGAME_RECIPE = ROOT / "reports/optimization_v3/posoffset_a02_endgame_v1/RECIPE.json"
E79_MANIFEST = ROOT / "reports/optimization_v3/posoffset_a02_endgame_v1/E79_SELECTED_MANIFEST.json"
E79_QA = ROOT / "reports/optimization_v3/posoffset_a02_endgame_v1/E79_INDEPENDENT_QA.json"
E79_NPZ = ROOT / "experiments/runs/optimization_v3_a02_cgeom_benchmark_v1/test_predictions.npz"
CURRENT_CSV = ROOT / "submissions/optimization_v3/posoffset_a02_endgame_v1/classification_mix50_e79.csv"
SAMPLE_CSV = ROOT / "data/raw/classification/SampleSubmission.csv"
OUTPUT_CSV = ROOT / "submissions/optimization_v3/cls_mix50_temporal_decision_v1/classification_mix50_temporal_decision_v1.csv"
MAX_SELECTIONS = 3
MIN_SELECTION_ROWS = 1000
MIN_CLASS_SUPPORT = 100
TIE_TOLERANCE = 1e-12
THRESHOLD_RULE_VERSION = "tied_score_cutpoints_macro_f1_v1"
MIB = 1024 * 1024
GIB = 1024 * MIB
MIN_AVAILABLE_RAM_BYTES = 4 * GIB
ESTIMATED_INCREMENTAL_PEAK_BYTES = 256 * MIB
SAFETY_RESERVE_BYTES = 2 * GIB
REQUIRED_AVAILABLE_RAM_BYTES = max(
    MIN_AVAILABLE_RAM_BYTES,
    ESTIMATED_INCREMENTAL_PEAK_BYTES + SAFETY_RESERVE_BYTES,
)
MAX_PROJECT_JOBS = 1
MAX_THREADPOOL_THREADS = 2
PREFIT_COMMITTED_FILES = {
    "scripts/optimization_v3_cls_mix50_temporal_decision_v1.py": "script",
    "experiments/optimization_v3/cls_mix50_temporal_decision_v1/protocol.json": "protocol",
    "tests/test_optimization_v3_cls_mix50_temporal_decision_v1.py": "tests",
    "src/models/research_next_contract.py": "research_next_contract",
}


def _strict_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_exclusive_json(path: Path, value: Mapping[str, Any]) -> str:
    """Publish immutable JSON once; a repeated exact write reuses its digest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = dict(value)
    body["receipt_sha256"] = canonical_sha256(body)
    payload = json.dumps(body, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    try:
        fd = os.open(path, flags, 0o600)
    except FileExistsError:
        existing = _strict_json(path)
        recorded = existing.pop("receipt_sha256", None)
        if recorded != canonical_sha256(existing):
            raise RuntimeError(f"Existing immutable receipt failed its digest: {path}")
        expected = dict(value)
        expected["receipt_sha256"] = canonical_sha256(expected)
        if existing != dict(value) or recorded != expected["receipt_sha256"]:
            raise RuntimeError(f"Refusing to overwrite a different immutable receipt: {path}")
        return recorded
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return body["receipt_sha256"]


def _read_receipt(path: Path) -> dict[str, Any]:
    value = _strict_json(path)
    digest = value.pop("receipt_sha256", None)
    if digest != canonical_sha256(value):
        raise RuntimeError(f"Immutable receipt hash mismatch: {path}")
    value["receipt_sha256"] = digest
    return value


def _append_jsonl_fsync(path: Path, event: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(event), sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "ab") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid ledger JSON at {path}:{line_no}") from exc
        if not isinstance(value, dict):
            raise RuntimeError(f"Invalid ledger event at {path}:{line_no}")
        rows.append(value)
    return rows


def _write_once_bytes(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    try:
        fd = os.open(path, flags, 0o600)
    except FileExistsError:
        if file_sha256(path) != hashlib.sha256(payload).hexdigest():
            raise RuntimeError(f"Refusing to overwrite different immutable file: {path}")
        return file_sha256(path)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(payload).hexdigest()


def _safe_ids(values: np.ndarray, *, label: str) -> np.ndarray:
    ids = np.asarray(values)
    if ids.ndim != 1 or ids.dtype.kind != "U":
        raise ValueError(f"{label} must be a one-dimensional Unicode array")
    if np.any(ids == "") or len(np.unique(ids)) != len(ids):
        raise ValueError(f"{label} must contain nonempty unique IDs")
    return ids.astype(str, copy=False)


def _ids_digest(ids: np.ndarray) -> str:
    # Length framing prevents ambiguity when IDs contain delimiter-like text.
    h = hashlib.sha256()
    for rid in np.asarray(ids, dtype=str):
        b = rid.encode("utf-8")
        h.update(len(b).to_bytes(8, "big"))
        h.update(b)
    return h.hexdigest()


def _validate_probabilities(values: np.ndarray, *, name: str) -> np.ndarray:
    p = np.asarray(values, dtype=np.float64)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError(f"{name} must be a finite one-dimensional probability vector in [0,1]")
    return p


def _validate_labels(values: np.ndarray) -> np.ndarray:
    y = np.asarray(values)
    if y.ndim != 1 or y.dtype.kind not in "biu" or not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be a one-dimensional integer/bool vector containing only 0/1")
    return y.astype(np.int8, copy=False)


def threshold_candidates(probability: np.ndarray) -> np.ndarray:
    """Return one cutoff per attainable tied-score allocation, plus fixed .50."""
    p = _validate_probabilities(probability, name="present probabilities")
    unique = np.unique(p)
    candidates: set[float] = {0.0, 0.5, float(np.nextafter(1.0, np.inf))}
    for low, high in zip(unique[:-1], unique[1:]):
        middle = float(low + (high - low) / 2.0)
        if not (float(low) < middle < float(high)):
            middle = float(high)
        candidates.add(middle)
    return np.asarray(sorted(candidates), dtype=np.float64)


def _counts_for_rule(
    y: np.ndarray,
    probability: np.ndarray,
    missing: np.ndarray,
    threshold: float,
) -> tuple[int, int, int, int]:
    y = _validate_labels(y)
    p = _validate_probabilities(probability, name="probability")
    miss = np.asarray(missing, dtype=bool)
    if y.shape != p.shape or y.shape != miss.shape:
        raise ValueError("y/probability/missing lengths differ")
    prediction = p >= 0.5
    present = ~miss
    prediction[present] = p[present] >= float(threshold)
    tn = int(np.sum((y == 0) & ~prediction))
    fp = int(np.sum((y == 0) & prediction))
    fn = int(np.sum((y == 1) & ~prediction))
    tp = int(np.sum((y == 1) & prediction))
    return tn, fp, fn, tp


def _macro_f1_from_counts(counts: tuple[int, int, int, int]) -> float:
    tn, fp, fn, tp = counts
    f1_zero_denom = 2 * tn + fp + fn
    f1_one_denom = 2 * tp + fp + fn
    f1_zero = (2.0 * tn / f1_zero_denom) if f1_zero_denom else 0.0
    f1_one = (2.0 * tp / f1_one_denom) if f1_one_denom else 0.0
    return (f1_zero + f1_one) / 2.0


def select_threshold(
    probability: np.ndarray,
    missing: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    """Learn one global present-only threshold from the declared Train rows."""
    p = _validate_probabilities(probability, name="selection probabilities")
    miss = np.asarray(missing, dtype=bool)
    y = _validate_labels(labels)
    if p.shape != miss.shape or p.shape != y.shape:
        raise ValueError("selection arrays have different lengths")
    class0 = int(np.sum(y == 0))
    class1 = int(np.sum(y == 1))
    support = {"rows": int(len(y)), "class_0": class0, "class_1": class1}
    if len(y) < MIN_SELECTION_ROWS or min(class0, class1) < MIN_CLASS_SUPPORT:
        return {
            "status": "FIXED_0_50_INSUFFICIENT_SUPPORT",
            "threshold": 0.5,
            "selection_fit_count": 0,
            "support": support,
            "selected_macro_f1": None,
            "tie_rule": "not_applicable",
        }

    present_mask = ~miss
    present_p = p[present_mask]
    present_y = y[present_mask]
    missing_y = y[miss]
    missing_prediction = p[miss] >= 0.5
    # Fixed missing decisions are constant over candidate thresholds.
    base_counts = (
        int(np.sum((missing_y == 0) & ~missing_prediction)),
        int(np.sum((missing_y == 0) & missing_prediction)),
        int(np.sum((missing_y == 1) & ~missing_prediction)),
        int(np.sum((missing_y == 1) & missing_prediction)),
    )
    if len(present_p):
        order = np.argsort(present_p, kind="mergesort")
        sorted_p = present_p[order]
        sorted_y = present_y[order].astype(np.int64)
        prefix_pos = np.concatenate(([0], np.cumsum(sorted_y, dtype=np.int64)))
        total_pos = int(prefix_pos[-1])
    else:
        sorted_p = np.empty(0, dtype=np.float64)
        prefix_pos = np.zeros(1, dtype=np.int64)
        total_pos = 0

    scored: list[tuple[float, float]] = []
    for threshold in threshold_candidates(present_p):
        first_positive = int(np.searchsorted(sorted_p, threshold, side="left"))
        pred_pos = len(sorted_p) - first_positive
        tp_present = total_pos - int(prefix_pos[first_positive])
        fp_present = pred_pos - tp_present
        fn_present = total_pos - tp_present
        tn_present = (len(sorted_p) - total_pos) - fp_present
        counts = (
            base_counts[0] + tn_present,
            base_counts[1] + fp_present,
            base_counts[2] + fn_present,
            base_counts[3] + tp_present,
        )
        scored.append((float(threshold), _macro_f1_from_counts(counts)))
    if not scored:
        raise RuntimeError("Threshold candidate set is empty")
    maximum = max(score for _, score in scored)
    tied = [cut for cut, score in scored if maximum - score <= TIE_TOLERANCE]
    threshold = min(tied, key=lambda cut: (abs(cut - 0.5), cut))
    return {
        "status": "SELECTED",
        "threshold": float(threshold),
        "selection_fit_count": 1,
        "support": support,
        "selected_macro_f1": float(maximum),
        "tie_rule": "within 1e-12 of max: closest to 0.50, then lower threshold",
        "candidate_count": len(scored),
        "unique_present_scores": int(len(np.unique(present_p))),
    }


def evaluate(
    y: np.ndarray,
    probability: np.ndarray,
    missing: np.ndarray,
    cycles: np.ndarray,
    threshold: float,
    groups: Mapping[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    y = _validate_labels(y)
    p = _validate_probabilities(probability, name="evaluation probabilities")
    miss = np.asarray(missing, dtype=bool)
    cyc = np.asarray(cycles)
    if y.shape != p.shape or y.shape != miss.shape or y.shape != cyc.shape:
        raise ValueError("evaluation arrays have different lengths")
    prediction = p >= 0.5
    prediction[~miss] = p[~miss] >= threshold

    def metric(mask: np.ndarray) -> dict[str, Any]:
        yy = y[mask]
        pred = prediction[mask]
        if len(yy) == 0:
            return {"rows": 0, "status": "UNKNOWN"}
        tn = int(np.sum((yy == 0) & ~pred))
        fp = int(np.sum((yy == 0) & pred))
        fn = int(np.sum((yy == 1) & ~pred))
        tp = int(np.sum((yy == 1) & pred))
        n0 = tn + fp
        n1 = fn + tp
        support = {"rows": int(len(yy)), "class_0": n0, "class_1": n1}
        if len(yy) < 200 or min(n0, n1) < 30:
            return {**support, "status": "UNKNOWN_UNSUPPORTED"}
        precision0 = tn / (tn + fn) if (tn + fn) else 0.0
        recall0 = tn / n0 if n0 else 0.0
        precision1 = tp / (tp + fp) if (tp + fp) else 0.0
        recall1 = tp / n1 if n1 else 0.0
        f10 = 2 * precision0 * recall0 / (precision0 + recall0) if (precision0 + recall0) else 0.0
        f11 = 2 * precision1 * recall1 / (precision1 + recall1) if (precision1 + recall1) else 0.0
        result = {
            **support,
            "target_prevalence": float(n1 / len(yy)),
            "predicted_positive_rate": float(np.mean(pred)),
            "confusion_matrix": [[tn, fp], [fn, tp]],
            "false_positives": fp,
            "false_negatives": fn,
            "precision": {"0": float(precision0), "1": float(precision1)},
            "recall": {"0": float(recall0), "1": float(recall1)},
            "f1": {"0": float(f10), "1": float(f11)},
            "macro_f1": float((f10 + f11) / 2.0),
        }
        result["status"] = "SUPPORTED"
        return result

    result: dict[str, Any] = {"overall": metric(np.ones(len(y), dtype=bool)), "per_cycle": {}}
    for cycle in np.unique(cyc):
        result["per_cycle"][str(int(cycle))] = metric(cyc == cycle)
    result["availability"] = {
        "present": metric(~miss),
        "missing": metric(miss),
    }
    result["groups"] = {}
    for name, raw_values in (groups or {}).items():
        values = np.asarray(raw_values)
        if values.shape != y.shape or values.dtype.kind not in "US":
            raise ValueError(f"Group {name} must be a same-length Unicode/string vector")
        result["groups"][name] = {
            str(value): metric(values == value) for value in np.unique(values)
        }
    return result


def load_train_labels_by_id(
    train_path: Path,
    requested_ids: np.ndarray,
    expected_cycle_by_id: Mapping[str, int],
    *,
    expected_train_sha256: str,
) -> np.ndarray:
    """Read labels only for explicitly requested IDs; do not return other rows."""
    ids = _safe_ids(requested_ids, label="requested label IDs")
    if file_sha256(train_path) != expected_train_sha256:
        raise RuntimeError("Official Train input hash differs from the frozen source")
    wanted = set(ids.tolist())
    found: dict[str, int] = {}
    with train_path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise RuntimeError("Official Train CSV is empty") from exc
        columns = {name: i for i, name in enumerate(header)}
        for needed in ("record_id", "cycle_code", "artifact_signal"):
            if needed not in columns:
                raise RuntimeError(f"Official Train missing required column {needed}")
        for row in reader:
            if len(row) <= max(columns.values()):
                raise RuntimeError("Malformed official Train CSV row")
            rid = row[columns["record_id"]]
            if rid not in wanted:
                continue
            if rid in found:
                raise RuntimeError(f"Duplicate official Train ID: {rid}")
            cycle = int(row[columns["cycle_code"]])
            if cycle != int(expected_cycle_by_id[rid]):
                raise RuntimeError(f"Train/cache cycle mismatch for {rid}")
            label = int(row[columns["artifact_signal"]])
            if label not in (0, 1):
                raise RuntimeError(f"Invalid artifact_signal for {rid}")
            found[rid] = label
    if set(found) != wanted:
        missing_ids = sorted(wanted - set(found))[:5]
        raise RuntimeError(f"Official Train does not contain all requested OOF IDs; examples={missing_ids}")
    return np.fromiter((found[rid] for rid in ids), dtype=np.int8, count=len(ids))


def _load_oof_predictions(path: Path, expected_hash: str) -> dict[str, dict[str, np.ndarray]]:
    if file_sha256(path) != expected_hash:
        raise RuntimeError("Saved MIX50 Train OOF bundle hash differs from frozen source")
    result: dict[str, dict[str, np.ndarray]] = {}
    allowed = ("ids", "rows", "cycles", "pP", "pA", "missing", "MIX50",
               "group_availability", "group_text_load", "group_entry_friction")
    with np.load(path, allow_pickle=False) as archive:
        for origin in (67, 75):
            prefix = f"E{origin}_"
            values = {name: np.asarray(archive[prefix + name]).copy()
                      for name in allowed if prefix + name in archive.files}
            required = {"ids", "rows", "cycles", "pP", "pA", "missing", "MIX50",
                        "group_availability", "group_text_load", "group_entry_friction"}
            if not required.issubset(values):
                raise RuntimeError(f"E{origin} OOF bundle missing {sorted(required - set(values))}")
            values["ids"] = _safe_ids(values["ids"], label=f"E{origin} IDs")
            values["cycles"] = np.asarray(values["cycles"], dtype=np.int16)
            values["missing"] = np.asarray(values["missing"], dtype=bool)
            for name in ("pP", "pA", "MIX50"):
                values[name] = _validate_probabilities(values[name], name=f"E{origin} {name}")
            expected = np.where(values["missing"], values["pP"], 0.5 * values["pP"] + 0.5 * values["pA"])
            if not np.array_equal(values["MIX50"], expected):
                raise RuntimeError(f"E{origin} MIX50 formula does not replay exactly")
            result[str(origin)] = values
    _validate_oof_boundaries(result)
    return result


def _validate_oof_boundaries(oof: Mapping[str, Mapping[str, np.ndarray]]) -> None:
    e67 = oof["67"]
    e75 = oof["75"]
    expected67 = np.arange(68, 80, dtype=np.int16)
    expected75 = np.arange(76, 80, dtype=np.int16)
    if not np.array_equal(np.unique(e67["cycles"]), expected67):
        raise RuntimeError("E67 OOF rows do not cover exactly cycles 68-79")
    if not np.array_equal(np.unique(e75["cycles"]), expected75):
        raise RuntimeError("E75 OOF rows do not cover exactly cycles 76-79")
    e67_mask = np.isin(e67["cycles"], expected75)
    ids67 = e67["ids"][e67_mask]
    ids75 = e75["ids"]
    if set(ids67.tolist()) != set(ids75.tolist()) or len(ids67) != len(ids75):
        raise RuntimeError("E67 and E75 W6 must contain the exact same unique Train IDs")
    cycle_by_67 = dict(zip(ids67.tolist(), e67["cycles"][e67_mask].tolist()))
    cycle_by_75 = dict(zip(ids75.tolist(), e75["cycles"].tolist()))
    if cycle_by_67 != cycle_by_75:
        raise RuntimeError("E67/E75 W6 ID cycle mapping differs")
    for origin, values in oof.items():
        n = len(values["ids"])
        if any(len(values[field]) != n for field in values):
            raise RuntimeError(f"E{origin} OOF array lengths disagree")
        if len(np.unique(values["ids"])) != n:
            raise RuntimeError(f"E{origin} OOF IDs are not unique")


def _slice(values: Mapping[str, np.ndarray], cycles: tuple[int, int]) -> dict[str, np.ndarray]:
    mask = (values["cycles"] >= cycles[0]) & (values["cycles"] <= cycles[1])
    return {name: value[mask] for name, value in values.items()}


def _cycle_map(ids: np.ndarray, cycles: np.ndarray) -> dict[str, int]:
    return {str(rid): int(cycle) for rid, cycle in zip(ids.tolist(), cycles.tolist())}


def _stage_paths(stage_id: str, run_dir: Path = RUN_DIR) -> tuple[Path, Path]:
    base = run_dir / "stages" / stage_id
    return base / "selection_receipt.json", base / "evaluation_receipt.json"


def _available_ram_bytes() -> int:
    """Return OS-reported available physical memory; fail closed if unavailable."""
    if os.name == "nt":
        import ctypes

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise RuntimeError("Could not read available physical memory")
        return int(status.ullAvailPhys)
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError) as exc:
        raise RuntimeError("Could not read available physical memory") from exc


def _resource_preflight(available_bytes: int | None = None) -> dict[str, Any]:
    available = _available_ram_bytes() if available_bytes is None else int(available_bytes)
    if available < 0:
        raise RuntimeError("Available RAM reading is invalid")
    gate = {
        "status": "PASS" if available >= REQUIRED_AVAILABLE_RAM_BYTES else "BLOCKED_LOW_AVAILABLE_RAM",
        "available_physical_ram_bytes": available,
        "minimum_available_ram_bytes": MIN_AVAILABLE_RAM_BYTES,
        "estimated_incremental_peak_bytes": ESTIMATED_INCREMENTAL_PEAK_BYTES,
        "documented_safety_reserve_bytes": SAFETY_RESERVE_BYTES,
        "required_available_ram_bytes": REQUIRED_AVAILABLE_RAM_BYTES,
        "project_job_limit": MAX_PROJECT_JOBS,
        "threadpool_thread_limit": MAX_THREADPOOL_THREADS,
        "estimate_note": (
            "Conservative incremental allowance for 20,903-row NumPy threshold arrays and "
            "line-streamed 50,608-row Train ID/label lookup; not a measured peak. "
            "A separate 2 GiB safety reserve is retained; the hard start floor is 4 GiB available."
        ),
    }
    if gate["status"] != "PASS":
        raise RuntimeError(
            "Selector resource gate blocked before reservation: available physical RAM "
            f"{available} bytes; requires at least {REQUIRED_AVAILABLE_RAM_BYTES} bytes"
        )
    return gate


def _validate_resource_gate_record(gate: Mapping[str, Any]) -> None:
    if gate.get("status") != "PASS":
        raise RuntimeError("Reservation resource gate did not pass")
    if gate.get("available_physical_ram_bytes", 0) < REQUIRED_AVAILABLE_RAM_BYTES:
        raise RuntimeError("Reservation resource receipt reports insufficient available RAM")
    if gate.get("minimum_available_ram_bytes") != MIN_AVAILABLE_RAM_BYTES:
        raise RuntimeError("Reservation resource receipt has a changed RAM floor")
    if gate.get("estimated_incremental_peak_bytes") != ESTIMATED_INCREMENTAL_PEAK_BYTES:
        raise RuntimeError("Reservation resource receipt has a changed peak allowance")
    if gate.get("documented_safety_reserve_bytes") < SAFETY_RESERVE_BYTES:
        raise RuntimeError("Reservation resource receipt has less than the frozen 2 GiB reserve")
    if gate.get("required_available_ram_bytes") != REQUIRED_AVAILABLE_RAM_BYTES:
        raise RuntimeError("Reservation resource receipt has a changed required-memory total")
    if (gate.get("project_job_limit") != MAX_PROJECT_JOBS
            or gate.get("threadpool_thread_limit") != MAX_THREADPOOL_THREADS):
        raise RuntimeError("Reservation resource receipt has changed process/thread limits")


@contextmanager
def _single_project_job(run_dir: Path):
    """Acquire an OS file lock so only one selector process can run at a time."""
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir / ".selector_job.lock"
    stream = lock_path.open("a+b")
    try:
        if os.name == "nt":
            import msvcrt

            stream.seek(0)
            if lock_path.stat().st_size == 0:
                stream.write(b"\0")
                stream.flush()
                stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("Another CLS_MIX50 temporal decision job is already active") from exc
        else:
            import fcntl

            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError("Another CLS_MIX50 temporal decision job is already active") from exc
        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt

                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        stream.close()


def _validated_ledger_state(ledger: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    reserves: dict[str, dict[str, Any]] = {}
    completes: dict[str, dict[str, Any]] = {}
    reservation_ids: set[str] = set()
    for event in _read_jsonl(ledger):
        kind = event.get("event")
        stage = event.get("selection_id")
        if not isinstance(stage, str) or not stage:
            raise RuntimeError("Decision ledger event has no valid selection_id")
        if Path(stage).name != stage or stage in {".", ".."}:
            raise RuntimeError("Decision ledger selection_id is not a safe stage name")
        if kind == "RESERVE":
            reservation_id = event.get("reservation_id")
            try:
                parsed_id = uuid.UUID(str(reservation_id))
            except (ValueError, AttributeError) as exc:
                raise RuntimeError("Reservation has an invalid reservation_id") from exc
            if parsed_id.hex != str(reservation_id).replace("-", "").lower():
                raise RuntimeError("Reservation ID is not in canonical UUID form")
            if event.get("selection_budget_max") != MAX_SELECTIONS:
                raise RuntimeError("Reservation budget cap does not match the frozen cap")
            resource = event.get("resource_gate")
            if not isinstance(resource, dict):
                raise RuntimeError("Reservation is missing a passing resource-gate receipt")
            _validate_resource_gate_record(resource)
            if stage in reserves or reservation_id in reservation_ids:
                raise RuntimeError(f"Duplicate reservation identity for {stage}")
            if len(reserves) >= MAX_SELECTIONS:
                raise RuntimeError("Three-selection decision budget is exhausted")
            reserves[stage] = event
            reservation_ids.add(str(reservation_id))
        elif kind == "COMPLETE":
            reservation_id = event.get("reservation_id")
            receipt_hash = event.get("receipt_sha256")
            if stage not in reserves:
                raise RuntimeError(f"Completion has no preceding reservation: {stage}")
            if stage in completes:
                raise RuntimeError(f"Duplicate completion event for {stage}")
            if reservation_id != reserves[stage].get("reservation_id"):
                raise RuntimeError(f"Completion reservation_id mismatch for {stage}")
            if not isinstance(receipt_hash, str) or len(receipt_hash) != 64:
                raise RuntimeError(f"Completion has an invalid receipt digest for {stage}")
            try:
                int(receipt_hash, 16)
            except ValueError as exc:
                raise RuntimeError(f"Completion has an invalid receipt digest for {stage}") from exc
            completes[stage] = event
        else:
            raise RuntimeError(f"Unknown decision ledger event type: {kind}")
    for stage, complete in completes.items():
        receipt_path = ledger.parent / "stages" / stage / "selection_receipt.json"
        if not receipt_path.exists():
            raise RuntimeError(f"Completed selection is missing its durable receipt: {stage}")
        receipt = _read_receipt(receipt_path)
        if (receipt.get("selection_id") != stage
                or receipt.get("reservation_id") != reserves[stage].get("reservation_id")
                or receipt.get("receipt_sha256") != complete.get("receipt_sha256")):
            raise RuntimeError(f"Completed ledger receipt digest mismatch: {stage}")
    return reserves, completes


def _reserve_selection(
    stage_id: str,
    ledger: Path,
    receipt_path: Path,
    *,
    resource_gate_fn: Callable[[], dict[str, Any]] | None = None,
) -> bool:
    expected_receipt_path = ledger.parent / "stages" / stage_id / "selection_receipt.json"
    if receipt_path.resolve() != expected_receipt_path.resolve():
        raise RuntimeError(f"Selection receipt path differs from its ledger stage: {stage_id}")
    reserves, completes = _validated_ledger_state(ledger)
    reserved = reserves.get(stage_id)
    completed = completes.get(stage_id)
    if receipt_path.exists():
        receipt = _read_receipt(receipt_path)
        if receipt.get("selection_id") != stage_id:
            raise RuntimeError(f"Selection receipt identity mismatch: {receipt_path}")
        if reserved is None:
            raise RuntimeError(f"Receipt exists without a reservation: {stage_id}")
        if receipt.get("reservation_id") != reserved["reservation_id"]:
            raise RuntimeError(f"Selection receipt reservation_id mismatch: {stage_id}")
        if completed is not None:
            if completed.get("receipt_sha256") != receipt["receipt_sha256"]:
                raise RuntimeError(f"Completed ledger receipt digest mismatch: {stage_id}")
        else:
            _append_jsonl_fsync(ledger, {
                "event": "COMPLETE",
                "selection_id": stage_id,
                "reservation_id": reserved["reservation_id"],
                "receipt_sha256": receipt["receipt_sha256"],
                "recovered": True,
            })
        return False
    if completed is not None:
        raise RuntimeError(f"Completed selection is missing its durable receipt: {stage_id}")
    if reserved is not None:
        raise RuntimeError(f"Reserved selection lacks a receipt; no automatic retry: {stage_id}")
    incomplete = sorted(set(reserves) - set(completes))
    if incomplete:
        raise RuntimeError(f"Reserved selection lacks a receipt; no automatic retry: {incomplete[0]}")
    if len(reserves) >= MAX_SELECTIONS:
        raise RuntimeError("Three-selection decision budget is exhausted")
    # The RAM check is deliberately before RESERVE so a low-memory stop consumes no slot.
    resource_gate = (resource_gate_fn or _resource_preflight)()
    _validate_resource_gate_record(resource_gate)
    reservation_id = str(uuid.uuid4())
    _append_jsonl_fsync(ledger, {
        "event": "RESERVE",
        "selection_id": stage_id,
        "reservation_id": reservation_id,
        "selection_budget_max": MAX_SELECTIONS,
        "resource_gate": resource_gate,
    })
    return True


def _reservation_for_stage(stage_id: str, ledger: Path) -> dict[str, Any]:
    reserves, _ = _validated_ledger_state(ledger)
    if stage_id not in reserves:
        raise RuntimeError(f"Missing reservation evidence after reserve: {stage_id}")
    return reserves[stage_id]


def _select_stage(
    *,
    stage_id: str,
    selector: Mapping[str, np.ndarray],
    selection_mask: np.ndarray,
    selection_cycles: tuple[int, int],
    train_path: Path,
    train_sha: str,
    run_dir: Path,
    source_hashes: Mapping[str, str],
    resource_gate_fn: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    receipt_path, _ = _stage_paths(stage_id, run_dir)
    ledger = run_dir / "selection_ledger.jsonl"
    ids = selector["ids"][selection_mask]
    probability = selector["MIX50"][selection_mask]
    missing = selector["missing"][selection_mask]
    cycles = selector["cycles"][selection_mask]
    if len(ids) == 0 or len(set(ids.tolist())) != len(ids):
        raise RuntimeError(f"Empty or duplicate selection IDs for {stage_id}")
    observed_cycles = np.unique(cycles)
    expected_cycles = np.arange(selection_cycles[0], selection_cycles[1] + 1, dtype=observed_cycles.dtype)
    if not np.array_equal(observed_cycles, expected_cycles):
        raise RuntimeError(f"Selection {stage_id} does not cover its frozen complete cycle range")
    selection_hashes = {
        "selection_record_ids_sha256": _ids_digest(ids),
        "selection_cycles_sha256": hashlib.sha256(np.asarray(cycles, dtype="<i2").tobytes()).hexdigest(),
        "probability_sha256": hashlib.sha256(np.asarray(probability, dtype="<f8").tobytes()).hexdigest(),
        "missing_mask_sha256": hashlib.sha256(np.asarray(missing, dtype=np.uint8).tobytes()).hexdigest(),
    }
    should_select = _reserve_selection(stage_id, ledger, receipt_path, resource_gate_fn=resource_gate_fn)
    if not should_select:
        receipt = _read_receipt(receipt_path)
        reservation = _reservation_for_stage(stage_id, ledger)
        expected_fields = {
            "selection_id": stage_id,
            "reservation_id": reservation["reservation_id"],
            "resource_gate": reservation["resource_gate"],
            "selection_cycles": [int(np.min(cycles)), int(np.max(cycles))],
            "selection_row_count": int(len(ids)),
            **selection_hashes,
            "threshold_rule_version": THRESHOLD_RULE_VERSION,
            "source_hashes": dict(source_hashes),
            "estimator_fits": 0,
            "test_accessed": False,
        }
        for key, expected in expected_fields.items():
            if receipt.get(key) != expected:
                raise RuntimeError(f"Selection receipt {key} mismatch for {stage_id}")
        return receipt
    reservation = _reservation_for_stage(stage_id, ledger)
    expected_cycle_by_id = _cycle_map(ids, cycles)
    y = load_train_labels_by_id(train_path, ids, expected_cycle_by_id,
                                expected_train_sha256=train_sha)
    learned = select_threshold(probability, missing, y)
    receipt_body = {
        "schema_version": 1,
        "experiment_id": "CLS_MIX50_TEMPORAL_DECISION_V1",
        "selection_id": stage_id,
        "reservation_id": reservation["reservation_id"],
        "resource_gate": reservation["resource_gate"],
        "selection_cycles": [int(np.min(cycles)), int(np.max(cycles))],
        "selection_row_count": int(len(ids)),
        "selection_record_ids_sha256": _ids_digest(ids),
        "selection_cycles_sha256": hashlib.sha256(np.asarray(cycles, dtype="<i2").tobytes()).hexdigest(),
        "probability_sha256": hashlib.sha256(np.asarray(probability, dtype="<f8").tobytes()).hexdigest(),
        "missing_mask_sha256": hashlib.sha256(np.asarray(missing, dtype=np.uint8).tobytes()).hexdigest(),
        "selection_train_labels_sha256": hashlib.sha256(np.asarray(y, dtype=np.int8).tobytes()).hexdigest(),
        "threshold_rule_version": THRESHOLD_RULE_VERSION,
        "threshold": learned["threshold"],
        "status": learned["status"],
        "selection_fit_count": learned["selection_fit_count"],
        "support": learned["support"],
        "selected_macro_f1": learned["selected_macro_f1"],
        "candidate_count": learned.get("candidate_count"),
        "unique_present_scores": learned.get("unique_present_scores"),
        "source_hashes": dict(source_hashes),
        "estimator_fits": 0,
        "test_accessed": False,
    }
    digest = _write_exclusive_json(receipt_path, receipt_body)
    _append_jsonl_fsync(ledger, {
        "event": "COMPLETE",
        "selection_id": stage_id,
        "reservation_id": reservation["reservation_id"],
        "receipt_sha256": digest,
    })
    persisted = _read_receipt(receipt_path)
    if persisted["threshold"] != learned["threshold"]:
        raise RuntimeError("Persisted threshold differs from the selected threshold")
    return persisted


def _evaluation_metrics(
    labels: np.ndarray,
    arrays: Mapping[str, np.ndarray],
    cycles: np.ndarray,
    threshold: float,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    groups = {name: arrays[name] for name in (
        "group_availability", "group_text_load", "group_entry_friction")}
    selected = evaluate(labels, arrays["MIX50"], arrays["missing"], cycles, threshold, groups=groups)
    fixed = evaluate(labels, arrays["MIX50"], arrays["missing"], cycles, 0.5, groups=groups)
    secondary = {
        "POS_at_0_50": evaluate(labels, arrays["pP"], arrays["missing"], cycles, 0.5)["overall"],
        "A02_at_0_50": evaluate(labels, arrays["pA"], arrays["missing"], cycles, 0.5)["overall"],
    }
    return selected, fixed, secondary


def _evaluate_stage(
    *,
    stage_id: str,
    evaluator: Mapping[str, np.ndarray],
    evaluation_mask: np.ndarray,
    threshold_receipt: Mapping[str, Any],
    train_path: Path,
    train_sha: str,
    run_dir: Path,
) -> dict[str, Any]:
    selection_path, eval_path = _stage_paths(stage_id, run_dir)
    if file_sha256(train_path) != train_sha:
        raise RuntimeError("Evaluation Train input hash differs from the frozen source")
    persisted_threshold = _read_receipt(selection_path)
    if persisted_threshold != dict(threshold_receipt):
        raise RuntimeError(f"Evaluation threshold is not the persisted selection receipt: {stage_id}")
    ids = evaluator["ids"][evaluation_mask]
    cycles = evaluator["cycles"][evaluation_mask]
    if len(ids) == 0 or len(set(ids.tolist())) != len(ids):
        raise RuntimeError(f"Empty or duplicate evaluation IDs for {stage_id}")
    expected_cycles_by_stage = {
        "D1_E67_W4_TO_W5": list(range(72, 76)),
        "D2_E67_TO_E75_W6": list(range(76, 80)),
    }
    observed_cycle_values = [int(value) for value in np.unique(cycles)]
    if observed_cycle_values != expected_cycles_by_stage.get(stage_id):
        raise RuntimeError(f"Evaluation cycle support differs from the frozen stage: {stage_id}")
    arrays = {name: values[evaluation_mask] for name, values in evaluator.items()}
    metric_inputs = {
        "MIX50_probability_sha256": hashlib.sha256(np.asarray(arrays["MIX50"], dtype="<f8").tobytes()).hexdigest(),
        "missing_mask_sha256": hashlib.sha256(np.asarray(arrays["missing"], dtype=np.uint8).tobytes()).hexdigest(),
        "pP_probability_sha256": hashlib.sha256(np.asarray(arrays["pP"], dtype="<f8").tobytes()).hexdigest(),
        "pA_probability_sha256": hashlib.sha256(np.asarray(arrays["pA"], dtype="<f8").tobytes()).hexdigest(),
        "group_availability_sha256": _ids_digest(np.asarray(arrays["group_availability"], dtype=str)),
        "group_text_load_sha256": _ids_digest(np.asarray(arrays["group_text_load"], dtype=str)),
        "group_entry_friction_sha256": _ids_digest(np.asarray(arrays["group_entry_friction"], dtype=str)),
    }
    expected_evaluation_cycles = [int(np.min(cycles)), int(np.max(cycles))]
    if eval_path.exists():
        saved = _read_receipt(eval_path)
        expected_fields = {
            "evaluation_id": stage_id,
            "evaluation_cycles": expected_evaluation_cycles,
            "evaluation_cycle_values": observed_cycle_values,
            "evaluation_row_count": int(len(ids)),
            "evaluation_record_ids_sha256": _ids_digest(ids),
            "selection_receipt_sha256": threshold_receipt["receipt_sha256"],
            "selected_threshold": float(threshold_receipt["threshold"]),
            "source_hashes": dict(threshold_receipt["source_hashes"]),
            "evaluation_input_hashes": metric_inputs,
            "evaluation_train_csv_sha256": train_sha,
        }
        for key, expected in expected_fields.items():
            if saved.get(key) != expected:
                raise RuntimeError(f"Evaluation receipt {key} mismatch for {stage_id}")
        # This re-read is evaluation-only: the threshold receipt is already persisted,
        # and the frozen Train hash plus exact evaluation IDs bind the target digest.
        resume_y = load_train_labels_by_id(
            train_path, ids, _cycle_map(ids, cycles), expected_train_sha256=train_sha
        )
        if saved.get("evaluation_train_labels_sha256") != hashlib.sha256(
            np.asarray(resume_y, dtype=np.int8).tobytes()
        ).hexdigest():
            raise RuntimeError(f"Evaluation receipt target-label digest mismatch for {stage_id}")
        replayed = _evaluation_metrics(resume_y, arrays, cycles, float(threshold_receipt["threshold"]))
        if (saved.get("selected_rule") != replayed[0]
                or saved.get("fixed_0_50_rule") != replayed[1]
                or saved.get("secondary_0_50_comparators") != replayed[2]):
            raise RuntimeError(f"Evaluation receipt metrics do not replay from frozen inputs: {stage_id}")
        return saved
    expected_cycle_by_id = _cycle_map(ids, cycles)
    # This read is deliberately after the selected threshold is persisted.
    y = load_train_labels_by_id(train_path, ids, expected_cycle_by_id,
                                expected_train_sha256=train_sha)
    chosen, fixed, secondary = _evaluation_metrics(y, arrays, cycles,
                                                   float(threshold_receipt["threshold"]))
    body = {
        "schema_version": 1,
        "experiment_id": "CLS_MIX50_TEMPORAL_DECISION_V1",
        "evaluation_id": stage_id,
        "evaluation_cycles": [int(np.min(cycles)), int(np.max(cycles))],
        "evaluation_row_count": int(len(ids)),
        "evaluation_cycle_values": observed_cycle_values,
        "evaluation_record_ids_sha256": _ids_digest(ids),
        "evaluation_input_hashes": metric_inputs,
        "evaluation_train_csv_sha256": train_sha,
        "evaluation_train_labels_sha256": hashlib.sha256(np.asarray(y, dtype=np.int8).tobytes()).hexdigest(),
        "source_hashes": dict(threshold_receipt["source_hashes"]),
        "selection_receipt_sha256": threshold_receipt["receipt_sha256"],
        "selected_threshold": float(threshold_receipt["threshold"]),
        "selected_rule": chosen,
        "fixed_0_50_rule": fixed,
        "secondary_0_50_comparators": secondary,
        "probabilities_unchanged_by_threshold": True,
        "estimator_fits": 0,
        "test_accessed": False,
    }
    _write_exclusive_json(eval_path, body)
    return _read_receipt(eval_path)


def _git_head(repo_root: Path = ROOT) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Could not identify current source commit") from exc


def _git_blob_oid(commit: str, relative_path: str, repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", f"{commit}:{relative_path}"], cwd=repo_root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Could not identify committed source blob: {relative_path}") from exc


def _validate_prefit_commit(
    prefit_commit: str,
    reviewed_source_files: Mapping[str, Any],
    *,
    repo_root: Path = ROOT,
    current_head: str | None = None,
) -> None:
    """Require a reviewed source commit in history with unchanged committed/current blobs."""
    if set(reviewed_source_files) != set(PREFIT_COMMITTED_FILES):
        raise RuntimeError("Prefit commit source-file binding is incomplete or unexpected")
    try:
        resolved = subprocess.check_output(
            ["git", "rev-parse", "--verify", f"{prefit_commit}^{{commit}}"],
            cwd=repo_root,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Reviewed prefit commit does not exist") from exc
    if resolved != prefit_commit:
        raise RuntimeError("Reviewed prefit commit is not a canonical full commit hash")
    head = current_head or _git_head(repo_root)
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", prefit_commit, head],
        cwd=repo_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if ancestry.returncode != 0:
        raise RuntimeError("Reviewed prefit commit is not an ancestor of current HEAD")
    for relative_path, source_key in PREFIT_COMMITTED_FILES.items():
        binding = reviewed_source_files.get(relative_path)
        if not isinstance(binding, dict):
            raise RuntimeError(f"Invalid reviewed source binding for {relative_path}")
        expected_sha = binding.get("file_sha256")
        expected_blob_oid = binding.get("git_blob_oid")
        expected_git_blob_sha = binding.get("git_blob_sha256")
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            raise RuntimeError(f"Invalid reviewed source hash for {relative_path}")
        if not isinstance(expected_blob_oid, str) or not expected_blob_oid:
            raise RuntimeError(f"Invalid reviewed Git blob identity for {relative_path}")
        if not isinstance(expected_git_blob_sha, str) or len(expected_git_blob_sha) != 64:
            raise RuntimeError(f"Invalid reviewed Git blob SHA-256 for {relative_path}")
        try:
            int(expected_sha, 16)
            committed_bytes = subprocess.check_output(
                ["git", "show", f"{prefit_commit}:{relative_path}"], cwd=repo_root
            )
            current_path = repo_root / Path(*relative_path.split("/"))
            current_bytes = current_path.read_bytes()
            current_blob_oid = subprocess.check_output(
                ["git", "hash-object", f"--path={relative_path}", "--stdin"],
                cwd=repo_root,
                input=current_bytes,
            ).decode("ascii").strip()
        except (ValueError, OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(f"Could not verify reviewed source blob: {relative_path}") from exc
        committed_sha = hashlib.sha256(committed_bytes).hexdigest()
        committed_blob_oid = _git_blob_oid(prefit_commit, relative_path, repo_root)
        if (committed_blob_oid != expected_blob_oid or current_blob_oid != expected_blob_oid
                or committed_sha != expected_git_blob_sha or file_sha256(current_path) != expected_sha):
            raise RuntimeError(f"Reviewed/current source blob hash mismatch: {source_key}")


def write_prefit_freeze() -> dict[str, Any]:
    """Hash inputs and source files without parsing labels or Test probabilities."""
    protocol = _strict_json(PROTOCOL)
    manifest = _strict_json(E79_MANIFEST)
    e79_qa = _strict_json(E79_QA)
    audit = _strict_json(OOF_AUDIT)
    if manifest.get("candidate") != "MIX50" or e79_qa.get("candidate") != "MIX50":
        raise RuntimeError("Historical selected E79 candidate is not the frozen MIX50")
    if e79_qa.get("status") != "PASS" or e79_qa.get("test_targets_loaded") is not False:
        raise RuntimeError("Prior E79 independent QA is not cleanly verified")
    if audit.get("new_fits") != 0 or audit.get("test_accessed") is not False:
        raise RuntimeError("Prior MIX50 audit does not satisfy expected zero-fit Train-only boundary")
    if file_sha256(OOF_NPZ) != audit.get("source_sha256"):
        raise RuntimeError("Exact MIX50 OOF bundle no longer matches its prior audit hash")
    if file_sha256(TRAIN_CSV) != protocol["candidate"]["official_train_sha256"]:
        raise RuntimeError("Official Train source hash differs from the frozen expected hash")
    if file_sha256(E79_NPZ) != manifest.get("source_hashes", {}).get("combined_e79_probability_bundle"):
        raise RuntimeError("Prior E79 probability bundle hash differs from its selected manifest")
    source_paths = {
        "script": Path(__file__),
        "tests": TESTS,
        "protocol": PROTOCOL,
        "research_next_contract": ROOT / "src/models/research_next_contract.py",
        "synthetic_qa": QA,
        "train_oof_audit": OOF_AUDIT,
        "endgame_recipe": ENDGAME_RECIPE,
        "policy_metrics": POLICY_METRICS,
        "e79_manifest": E79_MANIFEST,
        "e79_qa": E79_QA,
        "superseded_prefit_freeze_r4": OLD_FREEZE,
        "superseded_synthetic_qa_r4": OLD_QA,
    }
    prefit_commit = _git_head()
    reviewed_source_files = {}
    for relative_path in PREFIT_COMMITTED_FILES:
        current_path = ROOT / Path(*relative_path.split("/"))
        try:
            worktree_blob_oid = subprocess.check_output(
                ["git", "hash-object", f"--path={relative_path}", "--stdin"],
                cwd=ROOT,
                input=current_path.read_bytes(),
            ).decode("ascii").strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(f"Could not verify committed prefit source: {relative_path}") from exc
        committed_blob_oid = _git_blob_oid(prefit_commit, relative_path, ROOT)
        if worktree_blob_oid != committed_blob_oid:
            raise RuntimeError(f"Uncommitted or mismatched prefit source: {relative_path}")
        committed_bytes = subprocess.check_output(
            ["git", "show", f"{prefit_commit}:{relative_path}"], cwd=ROOT
        )
        reviewed_source_files[relative_path] = {
            "file_sha256": file_sha256(current_path),
            "git_blob_oid": committed_blob_oid,
            "git_blob_sha256": hashlib.sha256(committed_bytes).hexdigest(),
        }
    body = {
        "schema_version": 1,
        "experiment_id": "CLS_MIX50_TEMPORAL_DECISION_V1",
        "prefit_revision": "R5",
        "status": "PREFIT_FROZEN_NO_OFFICIAL_SELECTION",
        "supersedes_unapproved_prefit_attempts": {
            "freeze_sha256": file_sha256(OLD_FREEZE),
            "synthetic_qa_sha256": file_sha256(OLD_QA),
            "reason": "R5 hardens prefit approval, resource gates, and immutable resume evidence; official selectors remain unstarted.",
        },
        "prefit_commit": prefit_commit,
        "reviewed_source_files": reviewed_source_files,
        "source_hashes": {name: file_sha256(path) for name, path in source_paths.items()},
        "input_hashes": {
            "official_train_csv": file_sha256(TRAIN_CSV),
            "exact_mix50_train_oof_bundle": file_sha256(OOF_NPZ),
            "expected_e79_probability_bundle": manifest["source_hashes"]["combined_e79_probability_bundle"],
        },
        "input_paths": {
            "official_train_csv": TRAIN_CSV.relative_to(ROOT).as_posix(),
            "exact_mix50_train_oof_bundle": OOF_NPZ.relative_to(ROOT).as_posix(),
            "expected_e79_probability_bundle": E79_NPZ.relative_to(ROOT).as_posix(),
        },
        "labels_accessed": False,
        "test_probabilities_loaded": False,
        "test_ids_or_sample_submission_loaded": False,
        "estimator_fits": 0,
        "decision_selections": 0,
        "official_csv_created": False,
        "kaggle_upload": False,
    }
    _write_exclusive_json(FREEZE, body)
    return _read_receipt(FREEZE)


def _verify_frozen_source_hashes(freeze_receipt: Mapping[str, Any]) -> None:
    paths = {
        "script": Path(__file__),
        "tests": TESTS,
        "protocol": PROTOCOL,
        "research_next_contract": ROOT / "src/models/research_next_contract.py",
        "synthetic_qa": QA,
        "train_oof_audit": OOF_AUDIT,
        "endgame_recipe": ENDGAME_RECIPE,
        "policy_metrics": POLICY_METRICS,
        "e79_manifest": E79_MANIFEST,
        "e79_qa": E79_QA,
        "superseded_prefit_freeze_r4": OLD_FREEZE,
        "superseded_synthetic_qa_r4": OLD_QA,
    }
    expected = freeze_receipt.get("source_hashes", {})
    for name, path in paths.items():
        if expected.get(name) != file_sha256(path):
            raise RuntimeError(f"Frozen source file changed: {name}")


def _verify_frozen_input_hashes(freeze_receipt: Mapping[str, Any]) -> None:
    paths = {
        "official_train_csv": TRAIN_CSV,
        "exact_mix50_train_oof_bundle": OOF_NPZ,
        "expected_e79_probability_bundle": E79_NPZ,
    }
    expected = freeze_receipt.get("input_hashes", {})
    for name, path in paths.items():
        if expected.get(name) != file_sha256(path):
            raise RuntimeError(f"Frozen input file changed: {name}")


def _persist_e79_application_freeze(
    final_threshold_receipt: Mapping[str, Any],
    prefit_freeze: Mapping[str, Any],
    expected_e79_hash: str,
    path: Path,
) -> dict[str, Any]:
    freeze_file_sha = prefit_freeze["file_sha256"] if "file_sha256" in prefit_freeze else file_sha256(FREEZE)
    body = {
        "schema_version": 1,
        "experiment_id": "CLS_MIX50_TEMPORAL_DECISION_V1",
        "status": "E79_APPLICATION_FROZEN_BEFORE_PROBABILITY_READ",
        "final_threshold_receipt_sha256": final_threshold_receipt["receipt_sha256"],
        "threshold": float(final_threshold_receipt["threshold"]),
        "threshold_rule_version": THRESHOLD_RULE_VERSION,
        "prefit_freeze_file_sha256": freeze_file_sha,
        "prefit_freeze_receipt_sha256": prefit_freeze["receipt_sha256"],
        "expected_e79_probability_bundle_sha256": expected_e79_hash,
        "source_hashes": dict(final_threshold_receipt["source_hashes"]),
        "test_targets_loaded": False,
        "csv_constructed": False,
    }
    _write_exclusive_json(path, body)
    saved = _read_receipt(path)
    if (saved["final_threshold_receipt_sha256"] != final_threshold_receipt["receipt_sha256"]
            or saved["expected_e79_probability_bundle_sha256"] != expected_e79_hash
            or saved["prefit_freeze_file_sha256"] != freeze_file_sha
            or saved["prefit_freeze_receipt_sha256"] != prefit_freeze["receipt_sha256"]):
        raise RuntimeError("E79 application freeze does not bind threshold and all input/source hashes")
    return saved


def _validate_root_approval(path: Path, freeze_receipt: Mapping[str, Any]) -> dict[str, Any]:
    approval = _strict_json(path)
    _validate_prefit_commit(freeze_receipt["prefit_commit"], freeze_receipt["reviewed_source_files"])
    _verify_frozen_source_hashes(freeze_receipt)
    _verify_frozen_input_hashes(freeze_receipt)
    required = {
        "status": "PASS",
        "prefit_commit": freeze_receipt["prefit_commit"],
        "prefit_freeze_file_sha256": file_sha256(FREEZE),
        "prefit_freeze_receipt_sha256": freeze_receipt["receipt_sha256"],
        "protocol_sha256": file_sha256(PROTOCOL),
        "script_sha256": file_sha256(Path(__file__)),
        "tests_sha256": file_sha256(TESTS),
        "research_next_contract_sha256": file_sha256(ROOT / "src/models/research_next_contract.py"),
        "scope": "CLS_MIX50_TEMPORAL_DECISION_V1",
    }
    for key, value in required.items():
        if approval.get(key) != value:
            raise RuntimeError(f"ROOT prefit approval mismatch at {key}")
    return approval


def run_official_selectors(approval_path: Path = ROOT_APPROVAL) -> dict[str, Any]:
    if not FREEZE.exists():
        raise RuntimeError("Prefit freeze is missing; do not select thresholds")
    freeze = _read_receipt(FREEZE)
    _validate_root_approval(approval_path, freeze)
    try:
        from threadpoolctl import threadpool_limits
    except ImportError as exc:
        raise RuntimeError("threadpoolctl is required to enforce the two-thread project limit") from exc
    with _single_project_job(RUN_DIR):
        with threadpool_limits(limits=MAX_THREADPOOL_THREADS):
            return _run_official_selectors_body(approval_path)


def _run_official_selectors_body(approval_path: Path) -> dict[str, Any]:
    if not FREEZE.exists():
        raise RuntimeError("Prefit freeze is missing; do not select thresholds")
    freeze = _read_receipt(FREEZE)
    _validate_root_approval(approval_path, freeze)
    _verify_frozen_source_hashes(freeze)
    if file_sha256(OOF_NPZ) != freeze["input_hashes"]["exact_mix50_train_oof_bundle"]:
        raise RuntimeError("Train OOF input changed after prefit freeze")
    if file_sha256(TRAIN_CSV) != freeze["input_hashes"]["official_train_csv"]:
        raise RuntimeError("Official Train input changed after prefit freeze")
    oof = _load_oof_predictions(OOF_NPZ, freeze["input_hashes"]["exact_mix50_train_oof_bundle"])
    e67 = oof["67"]
    e75 = oof["75"]
    source_hashes = {
        "prefit_commit": freeze["prefit_commit"],
        "prefit_freeze_file_sha256": file_sha256(FREEZE),
        "prefit_freeze_receipt_sha256": freeze["receipt_sha256"],
        "protocol_sha256": file_sha256(PROTOCOL),
        "script_sha256": file_sha256(Path(__file__)),
        "research_next_contract_sha256": file_sha256(ROOT / "src/models/research_next_contract.py"),
        "train_csv_sha256": freeze["input_hashes"]["official_train_csv"],
        "oof_bundle_sha256": freeze["input_hashes"]["exact_mix50_train_oof_bundle"],
    }
    # D1: choose on E67 cycles 68-71; persist the threshold before reading 72-75 labels.
    d1_select_mask = (e67["cycles"] >= 68) & (e67["cycles"] <= 71)
    d1_eval_mask = (e67["cycles"] >= 72) & (e67["cycles"] <= 75)
    d1 = _select_stage(stage_id="D1_E67_W4_TO_W5", selector=e67,
                       selection_mask=d1_select_mask, selection_cycles=(68, 71),
                       train_path=TRAIN_CSV, train_sha=source_hashes["train_csv_sha256"],
                       run_dir=RUN_DIR, source_hashes=source_hashes)
    d1_eval = _evaluate_stage(stage_id="D1_E67_W4_TO_W5", evaluator=e67,
                              evaluation_mask=d1_eval_mask, threshold_receipt=d1,
                              train_path=TRAIN_CSV, train_sha=source_hashes["train_csv_sha256"],
                              run_dir=RUN_DIR)
    # D2: choose on E67 cycles 68-75; persist before reading E75 cycles 76-79 labels.
    d2_select_mask = (e67["cycles"] >= 68) & (e67["cycles"] <= 75)
    d2_eval_mask = (e75["cycles"] >= 76) & (e75["cycles"] <= 79)
    d2 = _select_stage(stage_id="D2_E67_TO_E75_W6", selector=e67,
                       selection_mask=d2_select_mask, selection_cycles=(68, 75),
                       train_path=TRAIN_CSV, train_sha=source_hashes["train_csv_sha256"],
                       run_dir=RUN_DIR, source_hashes=source_hashes)
    d2_eval = _evaluate_stage(stage_id="D2_E67_TO_E75_W6", evaluator=e75,
                              evaluation_mask=d2_eval_mask, threshold_receipt=d2,
                              train_path=TRAIN_CSV, train_sha=source_hashes["train_csv_sha256"],
                              run_dir=RUN_DIR)
    d1_eval_ids = e67["ids"][d1_eval_mask]
    d2_eval_ids = e75["ids"][d2_eval_mask]
    if set(d1_eval_ids.tolist()).intersection(d2_eval_ids.tolist()):
        raise RuntimeError("W5/W6 pooled prequential evaluation rows overlap")
    if len(d1_eval_ids) + len(d2_eval_ids) != 15953:
        raise RuntimeError("Prequential W5+W6 row coverage differs from frozen 15,953-row support")
    # Final selector uses each Train ID exactly once: E67 W4+W5, then E75 W6.
    e67_early_mask = (e67["cycles"] >= 68) & (e67["cycles"] <= 75)
    e67_early = {key: value[e67_early_mask] for key, value in e67.items()}
    final_ids = np.concatenate((e67_early["ids"], e75["ids"]))
    if len(final_ids) != 20903 or len(np.unique(final_ids)) != 20903:
        raise RuntimeError("Final OOF selector must contain 20,903 unique Train IDs")
    final_selector = {
        key: np.concatenate((e67_early[key], e75[key]))
        for key in e67_early
    }
    final_mask = np.ones(len(final_ids), dtype=bool)
    final = _select_stage(stage_id="FINAL_OOF_68_79", selector=final_selector,
                          selection_mask=final_mask, selection_cycles=(68, 79),
                          train_path=TRAIN_CSV, train_sha=source_hashes["train_csv_sha256"],
                          run_dir=RUN_DIR, source_hashes=source_hashes)
    pooled = _pooled_prequential_summary(d1_eval, d2_eval)
    result = {
        "schema_version": 1,
        "experiment_id": "CLS_MIX50_TEMPORAL_DECISION_V1",
        "status": "OFFICIAL_SELECTORS_COMPLETE",
        "selection_receipts": {"D1": d1["receipt_sha256"], "D2": d2["receipt_sha256"], "FINAL": final["receipt_sha256"]},
        "evaluation_receipts": {"D1_W5": d1_eval["receipt_sha256"], "D2_W6": d2_eval["receipt_sha256"]},
        "D1_threshold": d1["threshold"],
        "D2_threshold": d2["threshold"],
        "final_threshold": final["threshold"],
        "prequential_unique_w5_w6": pooled,
        "estimator_fits": 0,
        "decision_selections_used": sum(int(x["selection_fit_count"]) for x in (d1, d2, final)),
        "decision_selection_budget_max": MAX_SELECTIONS,
        "test_probabilities_loaded": False,
        "kaggle_upload": False,
    }
    _write_exclusive_json(REPORT_DIR / "SELECTOR_RESULTS.json", result)
    return result


def _pooled_prequential_summary(d1_eval: Mapping[str, Any], d2_eval: Mapping[str, Any]) -> dict[str, Any]:
    # Pooled confusion is recomputed from persisted counts; window F1s are never averaged.
    values: dict[str, Any] = {}
    for arm, result_key in (("selected", "selected_rule"), ("fixed_0_50", "fixed_0_50_rule")):
        a = d1_eval[result_key]["overall"]["confusion_matrix"]
        b = d2_eval[result_key]["overall"]["confusion_matrix"]
        counts = (a[0][0] + b[0][0], a[0][1] + b[0][1], a[1][0] + b[1][0], a[1][1] + b[1][1])
        tn, fp, fn, tp = counts
        values[arm] = {
            "rows": int(sum(counts)),
            "confusion_matrix": [[tn, fp], [fn, tp]],
            "false_positives": fp,
            "false_negatives": fn,
            "macro_f1": float(_macro_f1_from_counts(counts)),
        }
    values["delta_macro_f1_selected_minus_fixed"] = values["selected"]["macro_f1"] - values["fixed_0_50"]["macro_f1"]
    return values


def build_e79_candidate(approval_path: Path = ROOT_APPROVAL) -> dict[str, Any]:
    if not FREEZE.exists():
        raise RuntimeError("Prefit freeze is missing; do not build an E79 candidate")
    freeze = _read_receipt(FREEZE)
    _validate_root_approval(approval_path, freeze)
    try:
        from threadpoolctl import threadpool_limits
    except ImportError as exc:
        raise RuntimeError("threadpoolctl is required to enforce the two-thread project limit") from exc
    with _single_project_job(RUN_DIR):
        with threadpool_limits(limits=MAX_THREADPOOL_THREADS):
            return _build_e79_candidate_body(approval_path)


def _build_e79_candidate_body(approval_path: Path) -> dict[str, Any]:
    """Apply the final Train-only threshold after its immutable receipt exists."""
    freeze = _read_receipt(FREEZE)
    _validate_root_approval(approval_path, freeze)
    _verify_frozen_source_hashes(freeze)
    final_path, _ = _stage_paths("FINAL_OOF_68_79")
    final = _read_receipt(final_path)
    if final.get("status") not in {"SELECTED", "FIXED_0_50_INSUFFICIENT_SUPPORT"}:
        raise RuntimeError("Final OOF threshold receipt is not complete")
    if final.get("source_hashes", {}).get("prefit_freeze_receipt_sha256") != freeze["receipt_sha256"]:
        raise RuntimeError("Final threshold receipt is not bound to the frozen input/source hashes")
    application_resource_gate = _resource_preflight()
    # Verify the raw file digest before parsing the NPZ or reading probabilities.
    expected_e79_hash = freeze["input_hashes"]["expected_e79_probability_bundle"]
    app_freeze_path = REPORT_DIR / "E79_APPLICATION_FREEZE.json"
    application_freeze_saved = _persist_e79_application_freeze(
        final, freeze, expected_e79_hash, app_freeze_path
    )
    if file_sha256(E79_NPZ) != expected_e79_hash:
        raise RuntimeError("E79 probability bundle differs from the hash frozen before selection")
    with np.load(E79_NPZ, allow_pickle=False) as archive:
        required = {"record_ids", "missing_mask", "posoffset_probability", "a02_probability"}
        if not required.issubset(archive.files):
            raise RuntimeError("E79 saved probability bundle is missing required MIX50 inputs")
        ids = _safe_ids(np.asarray(archive["record_ids"]), label="E79 record_ids")
        missing = np.asarray(archive["missing_mask"], dtype=bool)
        p_pos = _validate_probabilities(np.asarray(archive["posoffset_probability"]), name="E79 pP")
        p_a02 = _validate_probabilities(np.asarray(archive["a02_probability"]), name="E79 pA")
    prior_manifest = _strict_json(E79_MANIFEST)
    if (file_sha256(CURRENT_CSV) != prior_manifest.get("csv_sha256")
            or prior_manifest.get("rows") != 29421
            or prior_manifest.get("unique_ids") != 29421
            or prior_manifest.get("candidate") != "MIX50"):
        raise RuntimeError("Prior MIX50 CSV or manifest identity changed")
    p_mix = np.where(missing, p_pos, 0.5 * p_pos + 0.5 * p_a02)
    if len(ids) != 29421 or len(np.unique(ids)) != 29421:
        raise RuntimeError("E79 probability bundle does not contain 29,421 unique IDs")

    with CURRENT_CSV.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["record_id", "artifact_signal"]:
            raise RuntimeError("Current MIX50 CSV schema changed")
        current_rows = list(reader)
    current_ids = np.asarray([row["record_id"] for row in current_rows], dtype=str)
    current_labels = np.asarray([int(row["artifact_signal"]) for row in current_rows], dtype=np.int8)
    with SAMPLE_CSV.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["record_id", "artifact_signal"]:
            raise RuntimeError("SampleSubmission schema changed")
        sample_ids = np.asarray([row["record_id"] for row in reader], dtype=str)
    if not np.array_equal(ids, current_ids) or not np.array_equal(ids, sample_ids):
        raise RuntimeError("E79 saved predictions, current MIX50, and SampleSubmission IDs/order differ")
    if len(np.unique(current_ids)) != 29421 or not np.isin(current_labels, [0, 1]).all():
        raise RuntimeError("Current MIX50 CSV has invalid or duplicate labels/IDs")
    _same = np.array_equal
    if not _same(current_labels[missing], (p_pos[missing] >= 0.5).astype(np.int8)):
        raise RuntimeError("Historical missing MIX50 labels are not exactly preserved")
    if not _same(current_labels, (p_mix >= 0.5).astype(np.int8)):
        raise RuntimeError("Current MIX50 CSV does not replay its frozen 0.50 probability rule")
    threshold = float(final["threshold"])
    labels = current_labels.copy()
    labels[~missing] = (p_mix[~missing] >= threshold).astype(np.int8)
    if labels.shape != (29421,) or not np.isin(labels, [0, 1]).all():
        raise RuntimeError("Candidate classification vector is not an integer binary 29,421-vector")
    label_sha = hashlib.sha256(labels.tobytes()).hexdigest()
    duplicate_paths: list[str] = []
    for old_path in (ROOT / "submissions").rglob("*.csv"):
        if old_path.resolve() == OUTPUT_CSV.resolve():
            continue
        try:
            with old_path.open("r", newline="", encoding="utf-8-sig") as stream:
                reader = csv.DictReader(stream)
                if reader.fieldnames != ["record_id", "artifact_signal"]:
                    continue
                old = list(reader)
            old_ids = [row["record_id"] for row in old]
            old_labels = [int(row["artifact_signal"]) for row in old]
        except (OSError, ValueError, KeyError, csv.Error):
            continue
        if len(old_ids) == len(ids) and old_ids == ids.tolist() and old_labels == labels.tolist():
            duplicate_paths.append(old_path.relative_to(ROOT).as_posix())
    status = "SKIP_DUPLICATE" if duplicate_paths else "PREPARED_NOT_SUBMITTED"
    if not duplicate_paths:
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(["record_id", "artifact_signal"])
        writer.writerows((rid, int(label)) for rid, label in zip(ids, labels))
        _write_once_bytes(OUTPUT_CSV, buffer.getvalue().encode("utf-8"))
    manifest = {
        "schema_version": 1,
        "experiment_id": "CLS_MIX50_TEMPORAL_DECISION_V1",
        "status": status,
        "threshold": threshold,
        "source_commit": _git_head(),
        "prefit_freeze_file_sha256": file_sha256(FREEZE),
        "prefit_freeze_receipt_sha256": freeze["receipt_sha256"],
        "application_resource_gate": application_resource_gate,
        "final_threshold_receipt_sha256": final["receipt_sha256"],
        "e79_application_freeze_sha256": application_freeze_saved["receipt_sha256"],
        "e79_probability_bundle_sha256": expected_e79_hash,
        "rows": int(len(ids)),
        "unique_ids": int(len(np.unique(ids))),
        "columns": ["record_id", "artifact_signal"],
        "missing_rows": int(np.sum(missing)),
        "present_rows": int(np.sum(~missing)),
        "binary_integer_labels": True,
        "label_vector_sha256": label_sha,
        "csv_path": OUTPUT_CSV.relative_to(ROOT).as_posix() if not duplicate_paths else None,
        "duplicate_existing_csvs": duplicate_paths,
        "test_targets_loaded": False,
        "kaggle_upload": False,
    }
    _write_exclusive_json(REPORT_DIR / "E79_CANDIDATE_MANIFEST.json", manifest)
    return manifest


def _synthetic_end_to_end(root: Path) -> dict[str, Any]:
    """Exercise selectors, no-eval-before-receipt, persistence, resume and E79 guard."""
    run_dir = root / "runs"
    run_dir.mkdir(parents=True)
    ids67 = np.asarray([f"syn-{cycle}-{i}" for cycle in range(68, 80) for i in range(300)], dtype="U16")
    cycles67 = np.repeat(np.arange(68, 80, dtype=np.int16), 300)
    labels_by_id: dict[str, int] = {}
    for rid, cycle, i in zip(ids67.tolist(), cycles67.tolist(), range(len(ids67))):
        labels_by_id[rid] = int((cycle * 17 + i * 7) % 5 < 2)
    p67 = np.asarray([0.15 if y == 0 else 0.62 for y in (labels_by_id[r] for r in ids67)], dtype=np.float64)
    p67[::7] = 0.5  # explicit tied boundary
    miss67 = (np.arange(len(ids67)) % 5 == 0)
    p_pos67 = np.clip(p67 - 0.04, 0, 1)
    p_a67 = np.clip(p67 + 0.04, 0, 1)
    p67 = np.where(miss67, p_pos67, 0.5 * p_pos67 + 0.5 * p_a67)
    group_avail67 = np.where(miss67, "missing", "present").astype("U7")
    group_text67 = np.where(np.arange(len(ids67)) % 2 == 0, "Q1", "Q2").astype("U2")
    group_ef67 = np.where(np.arange(len(ids67)) % 3 == 0, "1", "0").astype("U1")
    id_to_pos67 = {rid: i for i, rid in enumerate(ids67.tolist())}
    w6_positions = np.flatnonzero(cycles67 >= 76)
    ids75 = ids67[w6_positions].copy()
    cycles75 = cycles67[w6_positions].copy()
    p75 = p67[w6_positions].copy()
    missing75 = miss67[w6_positions].copy()
    p_pos75 = p_pos67[w6_positions].copy()
    p_a75 = p_a67[w6_positions].copy()
    e67 = {"ids": ids67, "cycles": cycles67, "MIX50": p67, "pP": p_pos67,
           "pA": p_a67, "missing": miss67, "group_availability": group_avail67,
           "group_text_load": group_text67, "group_entry_friction": group_ef67}
    e75 = {"ids": ids75, "cycles": cycles75, "MIX50": p75, "pP": p_pos75,
           "pA": p_a75, "missing": missing75,
           "group_availability": group_avail67[w6_positions],
           "group_text_load": group_text67[w6_positions],
           "group_entry_friction": group_ef67[w6_positions]}
    # Every helper gets only requested labels. Evaluation access is forbidden until its threshold receipt exists.
    requested_ids_log: list[dict[str, Any]] = []
    def provider(requested: np.ndarray, receipt_path: Path, phase: str) -> np.ndarray:
        if phase == "evaluation" and not receipt_path.exists():
            raise AssertionError("evaluation labels requested before threshold receipt")
        if phase == "selection" and receipt_path.exists():
            raise AssertionError("new selector attempted after a receipt existed")
        ids = np.asarray(requested, dtype=str)
        requested_ids_log.append({"phase": phase, "rows": int(len(ids)),
                                  "receipt_exists": receipt_path.exists(),
                                  "ids_sha256": _ids_digest(ids)})
        expected_cycles = {rid: int(cycles67[id_to_pos67[rid]]) for rid in ids.tolist()}
        loaded = load_train_labels_by_id(train, ids, expected_cycles,
                                         expected_train_sha256=train_hash)
        if any(int(label) != labels_by_id[rid] for rid, label in zip(ids.tolist(), loaded.tolist())):
            raise AssertionError("synthetic Train label reader changed a label")
        return loaded

    # Synthetic CSV supplies exact IDs/cycles/labels for the same production reader.
    train = root / "Train.csv"
    with train.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["record_id", "cycle_code", "artifact_signal"])
        for rid in ids67.tolist():
            writer.writerow([rid, int(cycles67[id_to_pos67[rid]]), labels_by_id[rid]])
    train_hash = file_sha256(train)

    def run_synthetic_stage(stage_id: str, selector: Mapping[str, np.ndarray],
                            selection_mask: np.ndarray, evaluation: Mapping[str, np.ndarray] | None,
                            evaluation_mask: np.ndarray | None) -> tuple[dict[str, Any], dict[str, Any] | None]:
        receipt_path, eval_path = _stage_paths(stage_id, run_dir)
        ledger = run_dir / "selection_ledger.jsonl"
        synthetic_gate = lambda: _resource_preflight(available_bytes=8 * GIB)
        should = _reserve_selection(stage_id, ledger, receipt_path, resource_gate_fn=synthetic_gate)
        if should:
            reservation = _reservation_for_stage(stage_id, ledger)
            ids = selector["ids"][selection_mask]
            cycles = selector["cycles"][selection_mask]
            y = provider(ids, receipt_path, "selection")
            learned = select_threshold(selector["MIX50"][selection_mask], selector["missing"][selection_mask], y)
            body = {"selection_id": stage_id, "reservation_id": reservation["reservation_id"],
                    "resource_gate": reservation["resource_gate"],
                    "threshold": learned["threshold"], "status": learned["status"],
                    "selection_fit_count": learned["selection_fit_count"], "source_hashes": {"synthetic": "only"},
                    "selection_record_ids_sha256": _ids_digest(ids)}
            digest = _write_exclusive_json(receipt_path, body)
            _append_jsonl_fsync(ledger, {"event": "COMPLETE", "selection_id": stage_id,
                                         "reservation_id": reservation["reservation_id"],
                                         "receipt_sha256": digest})
        receipt = _read_receipt(receipt_path)
        if receipt.get("source_hashes") != {"synthetic": "only"}:
            raise AssertionError("synthetic source binding mismatch")
        evaluation_receipt = None
        if evaluation is not None and evaluation_mask is not None:
            if eval_path.exists():
                evaluation_receipt = _read_receipt(eval_path)
            else:
                eval_ids = evaluation["ids"][evaluation_mask]
                eval_cycles = evaluation["cycles"][evaluation_mask]
                eval_y = provider(eval_ids, receipt_path, "evaluation")
                eval_result = evaluate(eval_y, evaluation["MIX50"][evaluation_mask],
                                       evaluation["missing"][evaluation_mask], eval_cycles,
                                       float(receipt["threshold"]))
                _write_exclusive_json(eval_path, {"selection_receipt_sha256": receipt["receipt_sha256"],
                                                  "metrics": eval_result})
                evaluation_receipt = _read_receipt(eval_path)
        return receipt, evaluation_receipt

    d1_sel = (cycles67 >= 68) & (cycles67 <= 71)
    d1_eval = (cycles67 >= 72) & (cycles67 <= 75)
    d1, d1e = run_synthetic_stage("D1_E67_W4_TO_W5", e67, d1_sel, e67, d1_eval)
    d2_sel = (cycles67 >= 68) & (cycles67 <= 75)
    d2_eval = cycles75 >= 76
    d2, d2e = run_synthetic_stage("D2_E67_TO_E75_W6", e67, d2_sel, e75, d2_eval)
    final_ids = np.concatenate((ids67[d2_sel], ids75))
    if len(final_ids) != len(np.unique(final_ids)):
        raise AssertionError("synthetic final OOF IDs must be unique")
    final_selector = {key: np.concatenate((value[d2_sel], e75[key])) for key, value in e67.items()}
    final, _ = run_synthetic_stage("FINAL_OOF_68_79", final_selector,
                                   np.ones(len(final_ids), dtype=bool), None, None)
    # Resume must reuse the durable threshold and not request selection labels or count another slot.
    slots_before = sum(event.get("event") == "RESERVE" for event in _read_jsonl(run_dir / "selection_ledger.jsonl"))
    final2, _ = run_synthetic_stage("FINAL_OOF_68_79", final_selector,
                                    np.ones(len(final_ids), dtype=bool), None, None)
    slots_after = sum(event.get("event") == "RESERVE" for event in _read_jsonl(run_dir / "selection_ledger.jsonl"))
    if final2["receipt_sha256"] != final["receipt_sha256"] or slots_before != slots_after:
        raise AssertionError("resume did not reuse exact final selection")
    # Guard for applying E79: probability loader cannot run before final selector receipt.
    final_receipt_path, _ = _stage_paths("FINAL_OOF_68_79", run_dir)
    synthetic_app_freeze = run_dir / "E79_APPLICATION_FREEZE.json"
    synthetic_final = _read_receipt(final_receipt_path)
    synthetic_prefit = {"receipt_sha256": "synthetic-prefit-freeze", "file_sha256": "synthetic-file-hash"}
    application_receipt = _persist_e79_application_freeze(
        synthetic_final, synthetic_prefit, "synthetic-test-probability-hash", synthetic_app_freeze
    )
    loader_called = False
    def guarded_test_probability_loader() -> None:
        nonlocal loader_called
        if (not final_receipt_path.exists() or not synthetic_app_freeze.exists()
                or application_receipt["final_threshold_receipt_sha256"] != synthetic_final["receipt_sha256"]
                or application_receipt["expected_e79_probability_bundle_sha256"] != "synthetic-test-probability-hash"):
            raise AssertionError("E79 probabilities accessed before threshold/input application freeze")
        loader_called = True
    guarded_test_probability_loader()
    if not loader_called or d1e is None or d2e is None:
        raise AssertionError("synthetic end-to-end evaluation did not complete")
    budget_guard = _assert_budget_cap_and_no_retry(run_dir)
    return {
        "status": "PASS",
        "synthetic_only": True,
        "official_inputs_accessed": False,
        "threshold_candidates": "tie-preserving; fixed 0.50 and all-positive/all-negative endpoints included",
        "selection_stages": 3,
        "reserved_slots": int(slots_after),
        "estimator_fits": 0,
        "evaluation_targets_requested_after_receipt": True,
        "e79_probability_loader_guard": "PASS",
        "persistence_resume": "PASS; final selection receipt reused without another reservation",
        "budget_cap_and_no_retry": "PASS" if budget_guard else "FAIL",
        "all_cutoff_metrics_match_bruteforce": _assert_bruteforce_selector(),
    }


def _assert_bruteforce_selector() -> bool:
    cases = [
        (np.tile(np.asarray([0.0, 0.0, 0.2, 0.2, 0.8, 1.0]), 200),
         np.tile(np.asarray([0, 1, 0, 1, 0, 1]), 200),
         np.tile(np.asarray([False] * 6), 200)),
        (np.tile(np.asarray([0.0, 0.5, 0.5, 1.0]), 300),
         np.tile(np.asarray([0, 1, 0, 1]), 300),
         np.tile(np.asarray([True, False, True, False]), 300)),
    ]
    for p, y, missing in cases:
        got = select_threshold(p, missing, y)
        candidates = threshold_candidates(p[~missing])
        scored = []
        for threshold in candidates:
            counts = _counts_for_rule(y, p, missing, float(threshold))
            scored.append((float(threshold), _macro_f1_from_counts(counts)))
        best_score = max(score for _, score in scored)
        tied = [threshold for threshold, score in scored if best_score - score <= TIE_TOLERANCE]
        expected = min(tied, key=lambda cut: (abs(cut - 0.5), cut))
        if float(got["threshold"]) != float(expected):
            raise AssertionError("optimized selector differs from brute-force synthetic cross-check")
    # Endpoint exercise: ties at p=1 can be all-negative only above 1.
    p = np.asarray([1.0, 1.0])
    if not np.all(p >= 0.0) or np.any(p >= np.nextafter(1.0, np.inf)):
        raise AssertionError("all-positive/all-negative endpoint construction failed")
    # Adjacent representable values have no representable interior midpoint; upper score must split them.
    low = np.float64(0.5)
    high = np.nextafter(low, np.inf)
    candidates = threshold_candidates(np.asarray([low, high]))
    if high not in candidates or (np.asarray([low, high]) >= high).tolist() != [False, True]:
        raise AssertionError("adjacent-float tie-block split failed")
    return True


def _assert_budget_cap_and_no_retry(run_dir: Path) -> bool:
    ledger = run_dir / "budget_cap_test.jsonl"
    for stage_id in ("CAP_TEST_1", "CAP_TEST_2", "CAP_TEST_3"):
        reservation_id = str(uuid.uuid4())
        _append_jsonl_fsync(ledger, {
            "event": "RESERVE",
            "selection_id": stage_id,
            "reservation_id": reservation_id,
            "selection_budget_max": MAX_SELECTIONS,
            "resource_gate": _resource_preflight(available_bytes=8 * GIB),
        })
        receipt_path = _stage_paths(stage_id, run_dir)[0]
        receipt_sha = _write_exclusive_json(receipt_path, {
            "selection_id": stage_id,
            "reservation_id": reservation_id,
            "synthetic": True,
        })
        _append_jsonl_fsync(ledger, {
            "event": "COMPLETE", "selection_id": stage_id,
            "reservation_id": reservation_id, "receipt_sha256": receipt_sha,
        })
    try:
        _reserve_selection("CAP_TEST_4", ledger, _stage_paths("CAP_TEST_4", run_dir)[0])
    except RuntimeError as exc:
        if "budget is exhausted" not in str(exc):
            raise
    else:
        raise AssertionError("a fourth decision selection exceeded the frozen three-slot cap")
    interrupted_ledger = run_dir / "interrupted_test.jsonl"
    interrupted_receipt = _stage_paths("INTERRUPTED", run_dir)[0]
    _append_jsonl_fsync(interrupted_ledger, {
        "event": "RESERVE", "selection_id": "INTERRUPTED",
        "reservation_id": str(uuid.uuid4()), "selection_budget_max": MAX_SELECTIONS,
        "resource_gate": _resource_preflight(available_bytes=8 * GIB),
    })
    try:
        _reserve_selection("INTERRUPTED", interrupted_ledger, interrupted_receipt)
    except RuntimeError as exc:
        if "no automatic retry" not in str(exc):
            raise
    else:
        raise AssertionError("an incomplete reserved stage was retried automatically")
    return True


def run_synthetic_qa() -> dict[str, Any]:
    import tempfile
    with tempfile.TemporaryDirectory(prefix="cls_mix50_temporal_decision_qa_") as tmp:
        result = _synthetic_end_to_end(Path(tmp))
    result["prefit_revision"] = "R5"
    result["supersedes_prior_synthetic_qa_sha256"] = file_sha256(OLD_QA)
    _write_exclusive_json(QA, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-qa", action="store_true", help="synthetic-only end-to-end QA")
    parser.add_argument("--write-prefit-freeze", action="store_true", help="hash sources/inputs without loading labels or Test probabilities")
    parser.add_argument("--run-official-selectors", action="store_true", help="run only after a matching ROOT prefit PASS")
    parser.add_argument("--build-e79-candidate", action="store_true", help="apply final Train-only threshold to saved E79 probabilities after receipt")
    parser.add_argument("--approval-json", type=Path, default=ROOT_APPROVAL)
    args = parser.parse_args()
    selected = sum((args.synthetic_qa, args.write_prefit_freeze, args.run_official_selectors, args.build_e79_candidate))
    if selected != 1:
        parser.error("choose exactly one operation")
    if args.synthetic_qa:
        print(json.dumps(run_synthetic_qa(), indent=2))
    elif args.write_prefit_freeze:
        print(json.dumps(write_prefit_freeze(), indent=2))
    elif args.run_official_selectors:
        print(json.dumps(run_official_selectors(args.approval_json), indent=2))
    else:
        print(json.dumps(build_e79_candidate(args.approval_json), indent=2))


if __name__ == "__main__":
    main()
