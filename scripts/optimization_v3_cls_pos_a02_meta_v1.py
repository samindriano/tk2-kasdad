"""Frozen POS+A02 score-meta experiment, with label access gated by root review.

The only real-data reads before ROOT_PREFIT_PASS are hashes and the OOF/E79
bundle structure. OOF target arrays are never loaded. The six official LR fits
and E79 application are unreachable until the exact prefit receipt is approved.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid
import warnings
from contextlib import contextmanager
from typing import Any, Mapping

import numpy as np
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.research_next_contract import canonical_sha256, file_sha256  # noqa: E402
from scripts import optimization_v3_cls_mix50_temporal_decision_v1 as mix50  # noqa: E402

EXPERIMENT = "CLS_POS_A02_META_V1"
PROTOCOL = ROOT / "experiments/optimization_v3/cls_pos_a02_meta_v1/protocol.json"
SCRIPT = Path(__file__).resolve()
TESTS = ROOT / "tests/test_optimization_v3_cls_pos_a02_meta_v1.py"
REPORT_DIR = ROOT / "reports/optimization_v3/cls_pos_a02_meta_v1"
PREFIT_QA = REPORT_DIR / "PREFIT_SYNTHETIC_QA.json"
PREFIT_FREEZE = REPORT_DIR / "PREFIT_FREEZE.json"
ROOT_APPROVAL = REPORT_DIR / "ROOT_PREFIT_PASS.json"
RUN_DIR = ROOT / "experiments/runs/optimization_v3_cls_pos_a02_meta_v1"
FIT_ROOT = RUN_DIR / "fits"
TRAIN_CSV = ROOT / "data/raw/classification/Train.csv"
OOF_NPZ = ROOT / "experiments/runs/optimization_v3_posoffset_a02_endgame_v1/policy_predictions.npz"
E79_NPZ = ROOT / "experiments/runs/optimization_v3_a02_cgeom_benchmark_v1/test_predictions.npz"
E79_FREEZE = REPORT_DIR / "E79_APPLICATION_FREEZE.json"
RESULTS = REPORT_DIR / "RESULTS.json"
MANIFEST = REPORT_DIR / "CANDIDATE_MANIFEST.json"
INFERENCE_QA = REPORT_DIR / "INFERENCE_QA.json"
BASELINE_CSV = ROOT / "submissions/optimization_v3/posoffset_a02_endgame_v1/classification_mix50_e79.csv"
PAIR_CSV = ROOT / "submissions/optimization_v3/cls_pos_a02_meta_v1/classification_pair_e79.csv"
LEDGER = RUN_DIR / "official_fit_ledger.jsonl"
JOB_LOCK = RUN_DIR / ".exclusive_meta_fit.lock"

R5_SCRIPT = ROOT / "scripts/optimization_v3_cls_mix50_temporal_decision_v1.py"
R5_FREEZE = ROOT / "reports/optimization_v3/cls_mix50_temporal_decision_v1/PREFIT_FREEZE_R5.json"
R5_APPROVAL = ROOT / "reports/optimization_v3/cls_mix50_temporal_decision_v1/ROOT_PREFIT_PASS.json"
R5_SELECTOR_RESULTS = ROOT / "reports/optimization_v3/cls_mix50_temporal_decision_v1/SELECTOR_RESULTS.json"
R5_SELECTOR_REPORT = ROOT / "reports/optimization_v3/cls_mix50_temporal_decision_v1/SELECTOR_DIAGNOSTIC.md"
C15_SUMMARY = ROOT / "reports/optimization_v3/batch1/C15/revision_05/summary.md"
C15_RESULTS = ROOT / "reports/optimization_v3/batch1/C15/revision_05/results.json"
C15_GATE = ROOT / "reports/optimization_v3/batch1/C15/revision_05/gate.json"
C15_VERIFY = ROOT / "reports/optimization_v3/batch1/C15/revision_05/verification.json"
PAIR_V2_REPORT = ROOT / "reports/optimization_v3/learned_pair_pilot_v2/POSTFIT_RESULTS.md"

MIX50_OOF_SHA256 = "25d1d2b91348a57d33b3b0e5ce78da29840d115b75e24f9502cc83c1b63ef351"
E79_SHA256 = "ffc55aa2d18b75438257f3cf0aa205baa3f4c2922180e2b3688b36c352c00271"
TRAIN_SHA256 = "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c"
EPSILON = 1e-6
THRESHOLD = 0.50
MIN_FIT_ROWS = 500
MIN_FIT_CLASS = 100
MIN_GROUP_ROWS = 200
MIN_GROUP_CLASS = 30
FIT_IDS = (
    "D1_PAIR", "D1_SCALAR", "D2_PAIR", "D2_SCALAR", "FINAL_PAIR", "FINAL_SCALAR",
)
MAX_OFFICIAL_FITS = 6
MAX_ITER = 2000
MAX_THREADS = 2
GIB = 1024 ** 3
MIN_AVAILABLE_RAM = 4 * GIB
ESTIMATED_PEAK_RAM = 256 * 1024 ** 2
SAFETY_RESERVE_RAM = 2 * GIB
REQUIRED_AVAILABLE_RAM = max(MIN_AVAILABLE_RAM, ESTIMATED_PEAK_RAM + SAFETY_RESERVE_RAM)
MODEL_PARAMS = {
    "C": 1.0,
    "solver": "liblinear",
    "penalty": "l2",
    "fit_intercept": True,
    "max_iter": MAX_ITER,
    "tol": 1e-4,
    "random_state": 42,
    "class_weight": None,
}
OOF_FIELDS = ("ids", "rows", "cycles", "pP", "pA", "missing", "MIX50",
              "group_availability", "group_text_load", "group_entry_friction")
SOURCE_FILES = {
    "script": SCRIPT,
    "protocol": PROTOCOL,
    "tests": TESTS,
    "research_next_contract": ROOT / "src/models/research_next_contract.py",
    "r5_script": R5_SCRIPT,
    "r5_freeze": R5_FREEZE,
    "r5_root_approval": R5_APPROVAL,
    "r5_selector_results": R5_SELECTOR_RESULTS,
    "r5_selector_report": R5_SELECTOR_REPORT,
    "c15_summary": C15_SUMMARY,
    "c15_results": C15_RESULTS,
    "c15_gate": C15_GATE,
    "c15_verification": C15_VERIFY,
    "learned_pair_v2_report": PAIR_V2_REPORT,
}


class MetaError(RuntimeError):
    """Fail-closed error for source, scope, budget, or state violations."""


def _strict_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MetaError(f"Expected a JSON object: {path}")
    return value


def _receipt(path: Path) -> dict[str, Any]:
    body = _strict_json(path)
    digest = body.pop("receipt_sha256", None)
    if digest != canonical_sha256(body):
        raise MetaError(f"Receipt digest mismatch: {path}")
    body["receipt_sha256"] = digest
    return body


def _write_json_once(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = dict(value)
    digest = canonical_sha256(body)
    body["receipt_sha256"] = digest
    payload = json.dumps(body, sort_keys=True, indent=2, ensure_ascii=False,
                         allow_nan=False).encode("utf-8") + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    try:
        fd = os.open(path, flags, 0o600)
    except FileExistsError:
        old = _receipt(path)
        if old != {**dict(value), "receipt_sha256": digest}:
            raise MetaError(f"Refusing to overwrite an immutable receipt: {path}")
        return digest
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return digest


def _write_bytes_once(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(payload).hexdigest()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    try:
        fd = os.open(path, flags, 0o600)
    except FileExistsError:
        if file_sha256(path) != digest:
            raise MetaError(f"Refusing to overwrite a different immutable artifact: {path}")
        return digest
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return digest


def _npz_bytes(**arrays: np.ndarray) -> bytes:
    out = io.BytesIO()
    np.savez_compressed(out, **arrays)
    return out.getvalue()


def _append_ledger(event: Mapping[str, Any]) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(event), sort_keys=True, ensure_ascii=False,
                         allow_nan=False).encode("utf-8") + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    fd = os.open(LEDGER, flags, 0o600)
    with os.fdopen(fd, "ab") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _ledger() -> list[dict[str, Any]]:
    if not LEDGER.exists():
        return []
    rows = []
    for n, line in enumerate(LEDGER.read_text(encoding="utf-8").splitlines(), 1):
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MetaError(f"Malformed official-fit ledger at line {n}") from exc
        if not isinstance(item, dict):
            raise MetaError(f"Invalid official-fit ledger event at line {n}")
        rows.append(item)
    return rows


def _digest_ids(ids: np.ndarray) -> str:
    h = hashlib.sha256()
    for value in np.asarray(ids, dtype=str):
        data = value.encode("utf-8")
        h.update(len(data).to_bytes(8, "big"))
        h.update(data)
    return h.hexdigest()


def clipped_logit(probability: np.ndarray) -> np.ndarray:
    p = np.asarray(probability, dtype=np.float64)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise MetaError("Probability input must be finite and within [0,1]")
    p = np.clip(p, EPSILON, 1.0 - EPSILON)
    return np.log(p / (1.0 - p))


def feature_matrix(p_pos: np.ndarray, p_a02: np.ndarray, mix50: np.ndarray,
                   arm: str) -> np.ndarray:
    p_pos = np.asarray(p_pos, dtype=np.float64)
    p_a02 = np.asarray(p_a02, dtype=np.float64)
    mix50 = np.asarray(mix50, dtype=np.float64)
    if p_pos.shape != p_a02.shape or p_pos.shape != mix50.shape:
        raise MetaError("POS/A02/MIX50 feature lengths differ")
    if arm == "PAIR":
        return np.column_stack((clipped_logit(p_pos), clipped_logit(p_a02)))
    if arm == "SCALAR":
        return clipped_logit(mix50).reshape(-1, 1)
    raise MetaError(f"Unknown meta arm: {arm}")


def validate_labels(labels: np.ndarray) -> np.ndarray:
    y = np.asarray(labels)
    if y.ndim != 1 or y.dtype.kind not in "biu" or not np.isin(y, [0, 1]).all():
        raise MetaError("Labels must be a one-dimensional binary integer vector")
    return y.astype(np.int8, copy=False)


def fit_support(labels: np.ndarray) -> dict[str, Any]:
    y = validate_labels(labels)
    counts = np.bincount(y, minlength=2)
    supported = (len(y) >= MIN_FIT_ROWS and int(counts[0]) >= MIN_FIT_CLASS
                 and int(counts[1]) >= MIN_FIT_CLASS)
    return {"rows": int(len(y)), "class_0": int(counts[0]), "class_1": int(counts[1]),
            "minimum_rows": MIN_FIT_ROWS, "minimum_each_class": MIN_FIT_CLASS,
            "status": "PASS" if supported else "BLOCKED_UNSUPPORTED_SELECTION"}


def policy_probabilities(
    p_pos: np.ndarray, p_a02: np.ndarray, missing: np.ndarray,
    mix50: np.ndarray, arm: str, state: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Score present rows; preserve the exact POS fallback on missing rows."""
    p_pos = np.asarray(p_pos, dtype=np.float64)
    p_a02 = np.asarray(p_a02, dtype=np.float64)
    miss = np.asarray(missing, dtype=bool)
    baseline = np.asarray(mix50, dtype=np.float64)
    if not (p_pos.shape == p_a02.shape == miss.shape == baseline.shape):
        raise MetaError("Policy input shapes differ")
    out = baseline.copy()
    present = ~miss
    if present.any():
        x = feature_matrix(p_pos[present], p_a02[present], baseline[present], arm)
        out[present] = _predict_from_state(x, state)
    out[miss] = p_pos[miss]
    if not np.isfinite(out).all() or np.any((out < 0) | (out > 1)):
        raise MetaError("Meta policy produced an invalid probability")
    return out


def _predict_from_state(x: np.ndarray, state: Mapping[str, np.ndarray]) -> np.ndarray:
    coef = np.asarray(state["coef"], dtype=np.float64).reshape(1, -1)
    intercept = float(np.asarray(state["intercept"], dtype=np.float64).reshape(-1)[0])
    if x.ndim != 2 or x.shape[1] != coef.shape[1]:
        raise MetaError("Saved model feature width does not match prediction matrix")
    return np.asarray(expit(x @ coef[0] + intercept), dtype=np.float64)


def _load_oof_without_labels(path: Path = OOF_NPZ) -> dict[str, dict[str, np.ndarray]]:
    if file_sha256(path) != MIX50_OOF_SHA256:
        raise MetaError("Saved Train OOF probability bundle hash changed")
    result: dict[str, dict[str, np.ndarray]] = {}
    with np.load(path, allow_pickle=False) as archive:
        for origin, expected_n, expected_cycles in (
            ("67", 20903, np.arange(68, 80, dtype=np.int16)),
            ("75", 8491, np.arange(76, 80, dtype=np.int16)),
        ):
            prefix = f"E{origin}_"
            required_keys = {prefix + name for name in OOF_FIELDS}
            if not required_keys.issubset(set(archive.files)):
                raise MetaError(f"E{origin} OOF source is missing probability/identity fields")
            values = {name: np.asarray(archive[prefix + name]).copy() for name in OOF_FIELDS}
            # Deliberately never index E67_y or E75_y in this loader.
            n = len(values["ids"])
            if n != expected_n or any(len(v) != n for v in values.values()):
                raise MetaError(f"E{origin} OOF shape differs from frozen row count")
            ids = np.asarray(values["ids"])
            if ids.ndim != 1 or ids.dtype.kind != "U" or np.any(ids == "") or len(np.unique(ids)) != n:
                raise MetaError(f"E{origin} IDs are not unique nonempty Unicode")
            cycles = np.asarray(values["cycles"], dtype=np.int16)
            if not np.array_equal(np.unique(cycles), expected_cycles):
                raise MetaError(f"E{origin} cycle coverage changed")
            rows = np.asarray(values["rows"])
            if rows.ndim != 1 or rows.dtype.kind not in "iu" or len(np.unique(rows)) != n:
                raise MetaError(f"E{origin} Train row positions are invalid")
            miss = np.asarray(values["missing"], dtype=bool)
            values["cycles"] = cycles
            values["missing"] = miss
            for name in ("pP", "pA", "MIX50"):
                p = np.asarray(values[name], dtype=np.float64)
                if p.shape != (n,) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
                    raise MetaError(f"E{origin} {name} is not a valid probability vector")
                values[name] = p
            expected_mix = np.where(miss, values["pP"], .5 * values["pP"] + .5 * values["pA"])
            if not np.array_equal(values["MIX50"], expected_mix):
                raise MetaError(f"E{origin} MIX50 formula fails exact replay")
            for group in ("group_availability", "group_text_load", "group_entry_friction"):
                g = np.asarray(values[group])
                if g.ndim != 1 or len(g) != n or g.dtype.kind not in "US":
                    raise MetaError(f"E{origin} group field {group} has invalid type/shape")
            values["rows"] = rows.astype(np.int64, copy=False)
            result[origin] = values
    e67, e75 = result["67"], result["75"]
    m67 = (e67["cycles"] >= 76) & (e67["cycles"] <= 79)
    ids67, ids75 = e67["ids"][m67], e75["ids"]
    if set(ids67.tolist()) != set(ids75.tolist()) or len(ids67) != len(ids75):
        raise MetaError("E67/E75 W6 IDs do not form the same unique OOF set")
    cyc67 = dict(zip(ids67.tolist(), e67["cycles"][m67].tolist()))
    cyc75 = dict(zip(ids75.tolist(), e75["cycles"].tolist()))
    if cyc67 != cyc75:
        raise MetaError("E67/E75 W6 cycle mapping differs")
    return result


def _slice(data: Mapping[str, np.ndarray], cycle_lo: int, cycle_hi: int) -> dict[str, np.ndarray]:
    mask = (data["cycles"] >= cycle_lo) & (data["cycles"] <= cycle_hi)
    return {key: np.asarray(value)[mask] for key, value in data.items()}


def make_unique_stitch(oof: Mapping[str, Mapping[str, np.ndarray]],
                       *, expected_rows: int = 20903) -> dict[str, np.ndarray]:
    early = _slice(oof["67"], 68, 75)
    late = _slice(oof["75"], 76, 79)
    result = {key: np.concatenate((early[key], late[key])) for key in OOF_FIELDS}
    if len(result["ids"]) != expected_rows or len(np.unique(result["ids"])) != expected_rows:
        raise MetaError(f"Final OOF stitch must contain {expected_rows} unique IDs")
    if not np.array_equal(np.unique(result["cycles"]), np.arange(68, 80)):
        raise MetaError("Final OOF stitch cycle coverage differs from 68-79")
    return result


def metric_bundle(y: np.ndarray, probability: np.ndarray,
                  cycles: np.ndarray | None = None,
                  groups: Mapping[str, np.ndarray] | None = None) -> dict[str, Any]:
    y = validate_labels(y)
    p = np.asarray(probability, dtype=np.float64)
    if p.shape != y.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise MetaError("Metric probabilities/labels have invalid shape or range")
    pred = (p >= THRESHOLD).astype(np.int8)

    def one(mask: np.ndarray) -> dict[str, Any]:
        yy, pp, hh = y[mask], p[mask], pred[mask]
        counts = np.bincount(yy, minlength=2)
        result: dict[str, Any] = {
            "rows": int(len(yy)), "class_0": int(counts[0]), "class_1": int(counts[1]),
            "macro_f1": float(f1_score(yy, hh, labels=[0, 1], average="macro", zero_division=0)),
            "precision": float(precision_score(yy, hh, zero_division=0)),
            "recall": float(recall_score(yy, hh, zero_division=0)),
            "confusion_matrix": confusion_matrix(yy, hh, labels=[0, 1]).astype(int).tolist(),
        }
        if len(np.unique(yy)) == 2:
            result.update({
                "average_precision": float(average_precision_score(yy, pp)),
                "roc_auc": float(roc_auc_score(yy, pp)),
                "log_loss": float(log_loss(yy, pp, labels=[0, 1])),
                "brier_score": float(np.mean((pp - yy) ** 2)),
            })
        else:
            result.update({"average_precision": None, "roc_auc": None,
                           "log_loss": None, "brier_score": None})
        return result

    full = np.ones(len(y), dtype=bool)
    result: dict[str, Any] = {"overall": one(full), "per_cycle": {}, "groups": {}}
    if cycles is not None:
        cyc = np.asarray(cycles)
        if cyc.shape != y.shape:
            raise MetaError("Cycle vector differs from metric row count")
        for cycle in np.unique(cyc):
            result["per_cycle"][str(int(cycle))] = one(cyc == cycle)
    for name, raw in (groups or {}).items():
        values = np.asarray(raw)
        if values.shape != y.shape or values.dtype.kind not in "US":
            raise MetaError(f"Invalid subgroup vector: {name}")
        groups_out = {}
        for group in np.unique(values):
            mask = values == group
            n0, n1 = int(np.sum(y[mask] == 0)), int(np.sum(y[mask] == 1))
            supported = (int(mask.sum()) >= MIN_GROUP_ROWS and min(n0, n1) >= MIN_GROUP_CLASS)
            groups_out[str(group)] = {
                "support": {"rows": int(mask.sum()), "class_0": n0, "class_1": n1,
                            "min_rows": MIN_GROUP_ROWS, "min_each_class": MIN_GROUP_CLASS},
                "status": "SUPPORTED" if supported else "UNKNOWN_UNSUPPORTED",
                "metrics": one(mask) if supported else None,
            }
        result["groups"][name] = groups_out
    return result


def pair_error_summary(y: np.ndarray, candidate: np.ndarray,
                       baseline: np.ndarray) -> dict[str, int]:
    y = validate_labels(y)
    cand = np.asarray(candidate, dtype=np.int8)
    base = np.asarray(baseline, dtype=np.int8)
    if cand.shape != y.shape or base.shape != y.shape:
        raise MetaError("Pair-error arrays have different shapes")
    changed = cand != base
    cand_correct = cand == y
    base_correct = base == y
    return {
        "changed_labels": int(changed.sum()),
        "changed_to_correct": int(np.sum(changed & cand_correct & ~base_correct)),
        "changed_to_wrong": int(np.sum(changed & ~cand_correct & base_correct)),
        "changed_but_both_same_correctness": int(np.sum(changed & (cand_correct == base_correct))),
    }


def resource_gate(available_bytes: int | None = None) -> dict[str, Any]:
    if available_bytes is None:
        if os.name == "nt":
            import ctypes

            class MemoryStatusEx(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            status = MemoryStatusEx()
            status.dwLength = ctypes.sizeof(status)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                raise MetaError("Could not read available physical RAM")
            available = int(status.ullAvailPhys)
        else:
            available = int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    else:
        available = int(available_bytes)
    gate = {
        "status": "PASS" if available >= REQUIRED_AVAILABLE_RAM else "BLOCKED_LOW_AVAILABLE_RAM",
        "available_physical_ram_bytes": available,
        "required_available_ram_bytes": REQUIRED_AVAILABLE_RAM,
        "minimum_available_ram_bytes": MIN_AVAILABLE_RAM,
        "estimated_incremental_peak_bytes": ESTIMATED_PEAK_RAM,
        "documented_safety_reserve_bytes": SAFETY_RESERVE_RAM,
        "project_fit_jobs": 1,
        "threadpool_limit": MAX_THREADS,
        "estimate_note": "LR meta fits use at most two dense score columns over 20,903 rows; the 4 GiB floor and 2 GiB reserve remain frozen even though estimated incremental peak is smaller.",
    }
    if gate["status"] != "PASS":
        raise MetaError("RAM gate failed before reservation; no fit started")
    return gate


@contextmanager
def _exclusive_fit_job():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    stream = JOB_LOCK.open("a+b")
    try:
        if os.name == "nt":
            import msvcrt
            stream.seek(0)
            if JOB_LOCK.stat().st_size == 0:
                stream.write(b"\0")
                stream.flush()
                stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise MetaError("Another project training job holds the experiment lock") from exc
        else:
            import fcntl
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise MetaError("Another project training job holds the experiment lock") from exc
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


def _fit_paths(fit_id: str) -> tuple[Path, Path, Path]:
    folder = FIT_ROOT / fit_id
    return folder / "model_state.npz", folder / "fit_receipt.json", folder / "failure.json"


def _read_model_state(path: Path, receipt: Mapping[str, Any]) -> dict[str, np.ndarray]:
    if file_sha256(path) != receipt.get("model_state_file_sha256"):
        raise MetaError("Model state file hash differs from fit receipt")
    with np.load(path, allow_pickle=False) as saved:
        state = {key: np.asarray(saved[key]).copy() for key in ("coef", "intercept", "classes", "n_iter")}
    if not all(np.isfinite(np.asarray(state[k], dtype=np.float64)).all() for k in ("coef", "intercept")):
        raise MetaError("Saved model state contains nonfinite parameters")
    if state["coef"].shape != (1, int(receipt["feature_width"])):
        raise MetaError("Saved model coefficient shape is invalid")
    if not np.array_equal(state["classes"], np.asarray([0, 1], dtype=state["classes"].dtype)):
        raise MetaError("Saved model class order is not [0,1]")
    return state


def _existing_fit(fit_id: str) -> dict[str, np.ndarray] | None:
    state_path, receipt_path, failure_path = _fit_paths(fit_id)
    events = [event for event in _ledger() if event.get("fit_id") == fit_id]
    if failure_path.exists() or any(e.get("event") == "FIT_FAILED" for e in events):
        raise MetaError(f"Fit {fit_id} has a terminal failure; automatic retry is prohibited")
    if receipt_path.exists() and state_path.exists():
        receipt = _receipt(receipt_path)
        if receipt.get("fit_id") != fit_id or receipt.get("status") != "FIT_SAVED":
            raise MetaError(f"Fit receipt identity/status mismatch for {fit_id}")
        state = _read_model_state(state_path, receipt)
        if not any(e.get("event") == "RESERVE" for e in events):
            raise MetaError(f"Saved fit {fit_id} has no durable reservation")
        if not any(e.get("event") == "FIT_START" for e in events):
            raise MetaError(f"Saved fit {fit_id} has no durable FIT_START")
        if not any(e.get("event") == "FIT_COMPLETE" for e in events):
            _append_ledger({"event": "FIT_COMPLETE", "fit_id": fit_id,
                            "receipt_sha256": receipt["receipt_sha256"],
                            "recovered_from_exact_persisted_state": True})
        return state
    if state_path.exists() or receipt_path.exists() or events:
        raise MetaError(f"Fit {fit_id} is partial or interrupted; no retry is allowed")
    return None


def _fit_one(fit_id: str, arm: str, x: np.ndarray, y: np.ndarray,
             ids: np.ndarray, cycles: np.ndarray,
             approval_sha256: str) -> dict[str, np.ndarray]:
    if fit_id not in FIT_IDS or fit_id.endswith("_" + arm) is False:
        raise MetaError("Fit ID does not match the frozen arm allowlist")
    existing = _existing_fit(fit_id)
    if existing is not None:
        return existing
    support = fit_support(y)
    if support["status"] != "PASS":
        _write_json_once(_fit_paths(fit_id)[2], {
            "experiment_id": EXPERIMENT, "fit_id": fit_id,
            "status": "BLOCKED_UNSUPPORTED_SELECTION", "support": support,
            "fit_started": False, "retry_allowed": False,
        })
        raise MetaError(f"Fit {fit_id} lacks frozen selection support")
    if x.shape != (len(y), 2 if arm == "PAIR" else 1) or len(ids) != len(y) or len(cycles) != len(y):
        raise MetaError("Fit matrices, labels, IDs, and cycles have different shapes")
    if len(np.unique(ids)) != len(ids) or not np.isfinite(x).all():
        raise MetaError("Fit input IDs/features are invalid")
    events = _ledger()
    reserved = {e.get("fit_id") for e in events if e.get("event") == "RESERVE"}
    if fit_id in reserved or len(reserved) >= MAX_OFFICIAL_FITS:
        raise MetaError("Official six-fit cap is exhausted or fit ID was previously reserved")
    resource = resource_gate()
    reservation_id = str(uuid.uuid4())
    reservation = {
        "event": "RESERVE", "experiment_id": EXPERIMENT, "fit_id": fit_id,
        "reservation_id": reservation_id, "cap": MAX_OFFICIAL_FITS,
        "fit_ids_allowlist": list(FIT_IDS), "resource_gate": resource,
        "source_commit": _git_head(), "approval_sha256": approval_sha256,
        "selection_ids_sha256": _digest_ids(ids), "selection_cycles": [int(cycles.min()), int(cycles.max())],
        "support": support,
    }
    _append_ledger(reservation)
    _append_ledger({"event": "FIT_START", "experiment_id": EXPERIMENT,
                    "fit_id": fit_id, "reservation_id": reservation_id,
                    "at_utc": _utc_now(), "retry_allowed": False})
    state_path, receipt_path, failure_path = _fit_paths(fit_id)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    model = LogisticRegression(**MODEL_PARAMS)
    caught: list[str] = []
    try:
        with warnings.catch_warnings(record=True) as seen, threadpool_limits(limits=MAX_THREADS):
            warnings.simplefilter("always")
            model.fit(x, validate_labels(y))
            caught = [type(w.message).__name__ for w in seen]
        if any(issubclass(w.category, ConvergenceWarning) for w in seen):
            raise MetaError("LogisticRegression emitted ConvergenceWarning")
        if model.n_iter_.shape != (1,) or int(model.n_iter_[0]) >= MAX_ITER:
            raise MetaError("LogisticRegression did not converge before max_iter")
        if not np.array_equal(model.classes_, np.asarray([0, 1])):
            raise MetaError("LogisticRegression class order differs from [0,1]")
        state = {
            "coef": np.asarray(model.coef_, dtype=np.float64),
            "intercept": np.asarray(model.intercept_, dtype=np.float64),
            "classes": np.asarray(model.classes_, dtype=np.int8),
            "n_iter": np.asarray(model.n_iter_, dtype=np.int32),
        }
        if not np.isfinite(state["coef"]).all() or not np.isfinite(state["intercept"]).all():
            raise MetaError("LogisticRegression state is nonfinite")
        state_sha = _write_bytes_once(state_path, _npz_bytes(**state))
        receipt_body = {
            "schema_version": 1, "experiment_id": EXPERIMENT,
            "fit_id": fit_id, "arm": arm, "status": "FIT_SAVED",
            "reservation_id": reservation_id, "approval_sha256": approval_sha256,
            "source_hashes": _source_hashes(), "input_hashes": _input_hashes(),
            "fit_population": "present-stream selection rows only",
            "fit_rows": int(len(y)), "class_0": int(np.sum(y == 0)), "class_1": int(np.sum(y == 1)),
            "selection_ids_sha256": _digest_ids(ids),
            "selection_cycle_range": [int(cycles.min()), int(cycles.max())],
            "feature_width": int(x.shape[1]), "feature_names": ["logit(pP)", "logit(pA)"] if arm == "PAIR" else ["logit(MIX50)"],
            "estimator": dict(MODEL_PARAMS), "threadpool_limit": MAX_THREADS,
            "convergence_warnings": caught, "n_iter": [int(v) for v in state["n_iter"]],
            "model_state_path": state_path.relative_to(ROOT).as_posix(),
            "model_state_file_sha256": state_sha,
        }
        receipt_sha = _write_json_once(receipt_path, receipt_body)
        _append_ledger({"event": "FIT_COMPLETE", "experiment_id": EXPERIMENT,
                        "fit_id": fit_id, "reservation_id": reservation_id,
                        "receipt_sha256": receipt_sha, "at_utc": _utc_now()})
        return _read_model_state(state_path, _receipt(receipt_path))
    except Exception as exc:
        # A durable exact state+receipt is recoverable by receipt replay; do not
        # misclassify a post-persist ledger interruption as a failed fit.
        if state_path.is_file() and receipt_path.is_file():
            raise
        failure = {"schema_version": 1, "experiment_id": EXPERIMENT,
                   "fit_id": fit_id, "reservation_id": reservation_id,
                   "status": "FIT_FAILED_TERMINAL", "exception_type": type(exc).__name__,
                   "message": str(exc), "retry_allowed": False, "at_utc": _utc_now()}
        _write_json_once(failure_path, failure)
        _append_ledger({"event": "FIT_FAILED", "fit_id": fit_id,
                        "reservation_id": reservation_id, "failure_sha256": file_sha256(failure_path),
                        "retry_allowed": False, "at_utc": _utc_now()})
        raise


def _utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _git_head() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _git_ancestor(ancestor: str, descendant: str) -> bool:
    return subprocess.run(["git", "merge-base", "--is-ancestor", ancestor, descendant],
                          cwd=ROOT, capture_output=True).returncode == 0


def _git_source_provenance(path: Path, commit: str) -> dict[str, str]:
    rel = path.resolve().relative_to(ROOT).as_posix()
    oid = subprocess.run(["git", "rev-parse", f"{commit}:{rel}"], cwd=ROOT,
                         check=True, capture_output=True, text=True).stdout.strip()
    blob = subprocess.run(["git", "cat-file", "blob", f"{commit}:{rel}"], cwd=ROOT,
                          check=True, capture_output=True).stdout
    working_sha = file_sha256(path)
    if hashlib.sha256(blob).hexdigest() != working_sha:
        raise MetaError(f"Committed source differs from working copy: {rel}")
    status = subprocess.run(["git", "status", "--porcelain", "--", rel], cwd=ROOT,
                            check=True, capture_output=True, text=True).stdout
    if status.strip():
        raise MetaError(f"Prefit source is modified or untracked: {rel}")
    return {"path": rel, "file_sha256": working_sha,
            "git_blob_oid": oid, "git_blob_sha256": hashlib.sha256(blob).hexdigest()}


def _source_hashes() -> dict[str, str]:
    missing = [name for name, path in SOURCE_FILES.items() if not path.is_file()]
    if missing:
        raise MetaError(f"Missing frozen source evidence: {missing}")
    return {name: file_sha256(path) for name, path in SOURCE_FILES.items()}


def _input_hashes() -> dict[str, str]:
    found = {
        "official_train_csv": file_sha256(TRAIN_CSV),
        "exact_mix50_train_oof_bundle": file_sha256(OOF_NPZ),
        "expected_e79_probability_bundle": file_sha256(E79_NPZ),
    }
    expected = {"official_train_csv": TRAIN_SHA256,
                "exact_mix50_train_oof_bundle": MIX50_OOF_SHA256,
                "expected_e79_probability_bundle": E79_SHA256}
    if found != expected:
        raise MetaError("Frozen official input hash set differs from the approved inputs")
    return found


def _validate_prefit_freeze() -> dict[str, Any]:
    freeze = _receipt(PREFIT_FREEZE)
    if freeze.get("experiment_id") != EXPERIMENT or freeze.get("status") != "PREFIT_FROZEN_WAIT_ROOT_APPROVAL":
        raise MetaError("Prefit freeze scope/status mismatch")
    for name, path in SOURCE_FILES.items():
        if freeze.get("source_hashes", {}).get(name) != file_sha256(path):
            raise MetaError(f"Frozen source changed after prefit: {name}")
    if freeze.get("input_hashes") != _input_hashes():
        raise MetaError("Frozen input hashes changed after prefit")
    commit = freeze.get("prefit_commit")
    if not isinstance(commit, str) or not _git_ancestor(commit, _git_head()):
        raise MetaError("Prefit commit is not an ancestor of current HEAD")
    return freeze


def validate_root_approval_payload(approval: Mapping[str, Any], freeze: Mapping[str, Any],
                                   freeze_file_sha256: str) -> None:
    required = {
        "status": "PASS", "scope": EXPERIMENT,
        "prefit_commit": freeze["prefit_commit"],
        "prefit_freeze_receipt_sha256": freeze["receipt_sha256"],
        "prefit_freeze_file_sha256": freeze_file_sha256,
        "protocol_sha256": freeze["source_hashes"]["protocol"],
        "script_sha256": freeze["source_hashes"]["script"],
        "tests_sha256": freeze["source_hashes"]["tests"],
        "research_next_contract_sha256": freeze["source_hashes"]["research_next_contract"],
        "r5_script_sha256": freeze["source_hashes"]["r5_script"],
    }
    for key, value in required.items():
        if approval.get(key) != value:
            raise MetaError(f"ROOT_PREFIT_PASS binding mismatch: {key}")


def _require_root_approval() -> tuple[dict[str, Any], str]:
    freeze = _validate_prefit_freeze()
    if not ROOT_APPROVAL.is_file():
        raise MetaError("ROOT_PREFIT_PASS is absent; official labels/fits are still blocked")
    approval = _strict_json(ROOT_APPROVAL)
    freeze_file_sha = file_sha256(PREFIT_FREEZE)
    validate_root_approval_payload(approval, freeze, freeze_file_sha)
    commit = approval.get("prefit_commit")
    if not _git_ancestor(commit, _git_head()):
        raise MetaError("Approved prefit commit is not an ancestor of current HEAD")
    return approval, file_sha256(ROOT_APPROVAL)


def _resource_versions() -> dict[str, str]:
    import sklearn
    import scipy
    return {"numpy": np.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__}


def _make_synthetic_features(n: int = 1200) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42017)
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    probability = expit(0.85 * z1 + 0.55 * z2 + rng.normal(scale=0.8, size=n))
    y = (rng.random(n) < probability).astype(np.int8)
    p_pos = np.clip(expit(z1), 0.02, 0.98)
    p_a02 = np.clip(expit(z2), 0.02, 0.98)
    mix = .5 * (p_pos + p_a02)
    missing = (np.arange(n) % 7 == 0)
    cycles = np.where(np.arange(n) < n * .7, 68, 72).astype(np.int16)
    return p_pos, p_a02, mix, missing, y, cycles


def synthetic_lifecycle() -> dict[str, Any]:
    """Exercise fit, persistence, reload, scoring, routes and metric lifecycle on synthetic rows only."""
    p_pos, p_a02, mix, missing, y, cycles = _make_synthetic_features()
    ids = np.asarray([f"SYN-{i:05d}" for i in range(len(y))], dtype="U9")
    select = cycles == 68
    evaluation = cycles == 72
    present_select = select & ~missing
    support = fit_support(y[present_select])
    if support["status"] != "PASS":
        raise MetaError("Synthetic selection does not satisfy the declared support gate")
    persisted: dict[str, dict[str, np.ndarray]] = {}
    predictions: dict[str, np.ndarray] = {}
    synthetic_fit_count = 0
    with tempfile.TemporaryDirectory(prefix="kasdad-pos-a02-meta-synthetic-") as temp:
        for arm in ("PAIR", "SCALAR"):
            x = feature_matrix(p_pos[present_select], p_a02[present_select], mix[present_select], arm)
            model = LogisticRegression(**MODEL_PARAMS)
            with warnings.catch_warnings(record=True) as seen, threadpool_limits(limits=MAX_THREADS):
                warnings.simplefilter("always")
                model.fit(x, y[present_select])
            if any(issubclass(w.category, ConvergenceWarning) for w in seen):
                raise MetaError("Synthetic model emitted a convergence warning")
            state = {"coef": np.asarray(model.coef_, dtype=np.float64),
                     "intercept": np.asarray(model.intercept_, dtype=np.float64),
                     "classes": np.asarray(model.classes_, dtype=np.int8),
                     "n_iter": np.asarray(model.n_iter_, dtype=np.int32)}
            path = Path(temp) / f"{arm}.npz"
            path.write_bytes(_npz_bytes(**state))
            receipt = {"model_state_file_sha256": file_sha256(path),
                       "feature_width": x.shape[1], "fit_id": "SYNTHETIC_" + arm,
                       "status": "FIT_SAVED"}
            reloaded = _read_model_state(path, receipt)
            x_eval = feature_matrix(p_pos[evaluation & ~missing], p_a02[evaluation & ~missing],
                                    mix[evaluation & ~missing], arm)
            before = _predict_from_state(x_eval, state)
            after = _predict_from_state(x_eval, reloaded)
            if not np.array_equal(before, after):
                raise MetaError("Synthetic persistence/resume replay is not bit-exact")
            persisted[arm] = reloaded
            predictions[arm] = policy_probabilities(
                p_pos[evaluation], p_a02[evaluation], missing[evaluation], mix[evaluation], arm, reloaded,
            )
            synthetic_fit_count += 1
        # A resume is a state reload, never a second .fit call.
        if synthetic_fit_count != 2:
            raise MetaError("Synthetic fit count accounting mismatch")
        if not np.array_equal(predictions["PAIR"][missing[evaluation]],
                              p_pos[evaluation][missing[evaluation]]):
            raise MetaError("Synthetic missing-stream fallback is not exact")
        eval_metrics = {
            arm: metric_bundle(y[evaluation], predictions[arm], cycles[evaluation])
            for arm in ("PAIR", "SCALAR")
        }
        return {
            "schema_version": 1, "experiment_id": EXPERIMENT,
            "status": "PASS_SYNTHETIC_LIFECYCLE_ONLY",
            "data_scope": "synthetic arrays generated in-memory; no official Train labels, E79 probabilities, Test IDs, or SampleSubmission accessed",
            "synthetic_official_fits": 0, "synthetic_lr_fits": synthetic_fit_count,
            "selection_support": support,
            "persist_reload_bit_exact": True,
            "resume_reuses_saved_state_without_refit": True,
            "missing_route_bit_exact": True,
            "threshold": THRESHOLD,
            "metrics_smoke": {arm: value["overall"] for arm, value in eval_metrics.items()},
            "source_boundary_test": "PASS",
        }


def write_prefit_freeze() -> dict[str, Any]:
    """Write synthetic QA and frozen source/input receipt, without reading real labels/probabilities."""
    if PREFIT_FREEZE.exists():
        return _validate_prefit_freeze()
    if ROOT_APPROVAL.exists():
        raise MetaError("Unexpected root approval exists before this prefit freeze")
    commit = _git_head()
    git_sources = {name: _git_source_provenance(path, commit) for name, path in SOURCE_FILES.items()
                   if name in {"script", "protocol", "tests", "research_next_contract", "r5_script"}}
    for name, path in SOURCE_FILES.items():
        if not path.is_file():
            raise MetaError(f"Missing source evidence for prefit freeze: {name}")
    input_hashes = _input_hashes()
    # This load uses only OOF identity/probability arrays; the NPZ y arrays are not indexed.
    oof = _load_oof_without_labels()
    stitch = make_unique_stitch(oof)
    if len(stitch["ids"]) != 20903:
        raise MetaError("Synthetic-only prefit gate found an unexpected OOF stitch size")
    qa = synthetic_lifecycle()
    qa["oof_source_audit"] = {
        "bundle_sha256": input_hashes["exact_mix50_train_oof_bundle"],
        "loaded_fields": list(OOF_FIELDS), "target_arrays_loaded": False,
        "E67_rows": len(oof["67"]["ids"]), "E75_rows": len(oof["75"]["ids"]),
        "unique_final_stitch_rows": len(stitch["ids"]),
        "E67_E75_W6_identity_mapping": "PASS",
    }
    qa["real_target_label_access"] = False
    qa["e79_probability_values_loaded"] = False
    qa["sample_submission_loaded"] = False
    qa["kaggle_contacted"] = False
    _write_json_once(PREFIT_QA, qa)
    source_hashes = {name: file_sha256(path) for name, path in SOURCE_FILES.items()}
    body = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT,
        "status": "PREFIT_FROZEN_WAIT_ROOT_APPROVAL",
        "prefit_commit": commit,
        "git_source_provenance": git_sources,
        "source_hashes": source_hashes,
        "input_hashes": input_hashes,
        "protocol_sha256": source_hashes["protocol"],
        "script_sha256": source_hashes["script"],
        "tests_sha256": source_hashes["tests"],
        "research_next_contract_sha256": source_hashes["research_next_contract"],
        "r5_script_sha256": source_hashes["r5_script"],
        "synthetic_qa_sha256": file_sha256(PREFIT_QA),
        "fit_budget": {"official_estimator_fit_cap": MAX_OFFICIAL_FITS,
                       "official_fits_started": 0, "official_fits_saved": 0,
                       "synthetic_lr_fits": qa["synthetic_lr_fits"]},
        "official_labels_accessed": False,
        "test_probability_values_loaded": False,
        "test_ids_or_sample_submission_loaded": False,
        "official_csv_created": False,
        "kaggle_upload": False,
        "root_approval_required_before_official_labels_or_fit": True,
    }
    _write_json_once(PREFIT_FREEZE, body)
    return _validate_prefit_freeze()


def _load_train_labels(ids: np.ndarray, cycles: np.ndarray) -> np.ndarray:
    # Must only be called after _require_root_approval() in run_official().
    cycle_map = dict(zip(np.asarray(ids, dtype=str).tolist(), np.asarray(cycles, dtype=int).tolist()))
    return mix50.load_train_labels_by_id(TRAIN_CSV, np.asarray(ids, dtype=str), cycle_map,
                                         expected_train_sha256=TRAIN_SHA256)


def _model_input(data: Mapping[str, np.ndarray], arm: str, mask: np.ndarray) -> np.ndarray:
    return feature_matrix(data["pP"][mask], data["pA"][mask], data["MIX50"][mask], arm)


def _save_predictions(stage: str, ids: np.ndarray, cycles: np.ndarray,
                      baseline: np.ndarray, predictions: Mapping[str, np.ndarray]) -> Path:
    path = RUN_DIR / "stages" / stage / "predictions.npz"
    arrays = {"ids": np.asarray(ids), "cycles": np.asarray(cycles),
              "baseline": np.asarray(baseline, dtype=np.float64),
              "pair": np.asarray(predictions["PAIR"], dtype=np.float64),
              "scalar": np.asarray(predictions["SCALAR"], dtype=np.float64)}
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != set(arrays):
                raise MetaError(f"Persisted prediction cache has different fields: {stage}")
            if any(not np.array_equal(saved[key], expected) for key, expected in arrays.items()):
                raise MetaError(f"Persisted prediction cache differs from deterministic replay: {stage}")
    else:
        _write_bytes_once(path, _npz_bytes(**arrays))
    with np.load(path, allow_pickle=False) as saved:
        for key, expected in arrays.items():
            if not np.array_equal(saved[key], expected):
                raise MetaError(f"Persisted predictions fail exact replay: {stage}/{key}")
    return path


def _fit_stage(stage_id: str, fit_data: Mapping[str, np.ndarray],
               eval_data: Mapping[str, np.ndarray], approval_sha: str) -> dict[str, Any]:
    resource_gate()
    present = ~np.asarray(fit_data["missing"], dtype=bool)
    fit_ids = np.asarray(fit_data["ids"])[present]
    fit_cycles = np.asarray(fit_data["cycles"])[present]
    y_fit = _load_train_labels(fit_ids, fit_cycles)
    support = fit_support(y_fit)
    if support["status"] != "PASS":
        failure = {"experiment_id": EXPERIMENT, "stage_id": stage_id,
                   "status": "BLOCKED_UNSUPPORTED_SELECTION", "support": support,
                   "fit_started": False, "retry_allowed": False}
        _write_json_once(RUN_DIR / "stages" / stage_id / "failure.json", failure)
        raise MetaError(f"{stage_id} selection fails frozen support; stopping without a fit")
    states: dict[str, dict[str, np.ndarray]] = {}
    for arm in ("PAIR", "SCALAR"):
        fit_id = f"{stage_id}_{arm}"
        x = _model_input(fit_data, arm, present)
        states[arm] = _fit_one(fit_id, arm, x, y_fit, fit_ids, fit_cycles, approval_sha)

    # Persist both model states before reading this stage's evaluation labels.
    eval_predictions = {
        arm: policy_probabilities(eval_data["pP"], eval_data["pA"], eval_data["missing"],
                                  eval_data["MIX50"], arm, states[arm])
        for arm in ("PAIR", "SCALAR")
    }
    prediction_path = _save_predictions(stage_id, eval_data["ids"], eval_data["cycles"],
                                        eval_data["MIX50"], eval_predictions)
    y_eval = _load_train_labels(eval_data["ids"], eval_data["cycles"])
    groups = {
        "availability": eval_data["group_availability"],
        "text_load": eval_data["group_text_load"],
        "entry_friction": eval_data["group_entry_friction"],
    }
    baseline_labels = (np.asarray(eval_data["MIX50"]) >= THRESHOLD).astype(np.int8)
    evaluated: dict[str, Any] = {}
    for arm, probability in eval_predictions.items():
        labels = (probability >= THRESHOLD).astype(np.int8)
        metrics = metric_bundle(y_eval, probability, eval_data["cycles"], groups)
        baseline_metrics = metric_bundle(y_eval, eval_data["MIX50"], eval_data["cycles"], groups)
        evaluated[arm] = {
            "metrics": metrics,
            "delta_macro_f1_vs_mix50": metrics["overall"]["macro_f1"] - baseline_metrics["overall"]["macro_f1"],
            "pair_errors_vs_mix50": pair_error_summary(y_eval, labels, baseline_labels),
            "changed_labels_vs_mix50": int(np.sum(labels != baseline_labels)),
        }
    confusion_pair_scalar = int(np.sum((eval_predictions["PAIR"] >= THRESHOLD)
                                       != (eval_predictions["SCALAR"] >= THRESHOLD)))
    result = {
        "schema_version": 1, "experiment_id": EXPERIMENT, "stage_id": stage_id,
        "fit_rows_present": int(present.sum()), "fit_support": support,
        "fit_ids_sha256": _digest_ids(fit_ids),
        "eval_origin_rows": int(len(y_eval)), "eval_ids_sha256": _digest_ids(eval_data["ids"]),
        "eval_cycle_range": [int(eval_data["cycles"].min()), int(eval_data["cycles"].max())],
        "prediction_cache_sha256": file_sha256(prediction_path),
        "mix50_baseline": baseline_metrics if "baseline_metrics" in locals() else metric_bundle(
            y_eval, eval_data["MIX50"], eval_data["cycles"], groups),
        "arms": evaluated,
        "pair_scalar_disagreement_rows": confusion_pair_scalar,
        "target_reads_started_after_both_fit_states_persisted": True,
    }
    path = REPORT_DIR / "stages" / f"{stage_id}_RESULTS.json"
    _write_json_once(path, result)
    return result


def _read_existing_csv(path: Path, expected_ids: np.ndarray) -> np.ndarray:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["record_id", "artifact_signal"]:
            raise MetaError(f"Existing comparison CSV schema differs: {path}")
        rows = list(reader)
    ids = np.asarray([r["record_id"] for r in rows], dtype=str)
    if not np.array_equal(ids, np.asarray(expected_ids, dtype=str)):
        raise MetaError(f"Existing comparison CSV IDs/order differ: {path}")
    labels = np.asarray([int(r["artifact_signal"]) for r in rows], dtype=np.int8)
    if not np.isin(labels, [0, 1]).all():
        raise MetaError(f"Existing comparison CSV contains nonbinary labels: {path}")
    return labels


def _read_official_test_ids() -> np.ndarray:
    ids = []
    with (ROOT / "data/raw/classification/Test.csv").open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if "record_id" not in (reader.fieldnames or []):
            raise MetaError("Official Test file has no record_id column")
        for row in reader:
            ids.append(row["record_id"])
    arr = np.asarray(ids, dtype="U")
    if len(arr) != 29421 or len(np.unique(arr)) != len(arr):
        raise MetaError("Official Test IDs fail frozen count/uniqueness")
    return arr


def _application_freeze(approval_sha: str, final_receipts: Mapping[str, str]) -> dict[str, Any]:
    if not all(_fit_paths(f"FINAL_{arm}")[1].is_file() for arm in ("PAIR", "SCALAR")):
        raise MetaError("Both final OOF fit receipts must be durable before E79 application freeze")
    body = {
        "schema_version": 1, "experiment_id": EXPERIMENT,
        "status": "E79_APPLICATION_FROZEN_BEFORE_PROBABILITY_READ",
        "prefit_freeze_sha256": file_sha256(PREFIT_FREEZE),
        "root_approval_sha256": approval_sha,
        "final_fit_receipts": dict(final_receipts),
        "source_hashes": _source_hashes(),
        "input_hashes": _input_hashes(),
        "e79_probability_bundle_sha256": E79_SHA256,
        "mix50_baseline_csv_sha256": file_sha256(BASELINE_CSV),
        "threshold": THRESHOLD,
        "missing_route": "original pP >= 0.50",
        "test_probability_values_loaded": False,
        "test_ids_loaded": False,
        "sample_submission_loaded": False,
        "kaggle_upload": False,
    }
    _write_json_once(E79_FREEZE, body)
    return _receipt(E79_FREEZE)


def _load_test_probabilities_after_freeze() -> dict[str, np.ndarray]:
    app = _receipt(E79_FREEZE)
    if app.get("status") != "E79_APPLICATION_FROZEN_BEFORE_PROBABILITY_READ":
        raise MetaError("Missing E79 application freeze")
    if file_sha256(E79_NPZ) != E79_SHA256:
        raise MetaError("E79 bundle differs from prefit/approval source")
    with np.load(E79_NPZ, allow_pickle=False) as archive:
        required = ("record_ids", "missing_mask", "posoffset_probability", "a02_probability")
        if not set(required).issubset(set(archive.files)):
            raise MetaError("E79 probability bundle lacks frozen POS/A02 inputs")
        result = {key: np.asarray(archive[key]).copy() for key in required}
    n = len(result["record_ids"])
    ids = np.asarray(result["record_ids"])
    if ids.ndim != 1 or ids.dtype.kind != "U" or len(ids) != 29421 or len(np.unique(ids)) != n:
        raise MetaError("E79 record IDs fail frozen uniqueness/count")
    result["missing_mask"] = np.asarray(result["missing_mask"], dtype=bool)
    for name in ("posoffset_probability", "a02_probability"):
        p = np.asarray(result[name], dtype=np.float64)
        if p.shape != (n,) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise MetaError(f"E79 {name} is not a valid probability vector")
        result[name] = p
    return result


def run_official() -> dict[str, Any]:
    """Run exactly the frozen six-fit sequence; requires Sol's bound prefit pass."""
    approval, approval_sha = _require_root_approval()
    oof = _load_oof_without_labels()
    e67, e75 = oof["67"], oof["75"]
    d1_fit = _slice(e67, 68, 71)
    d1_eval = _slice(e67, 72, 75)
    d2_fit = _slice(e67, 68, 75)
    d2_eval = _slice(e75, 76, 79)
    with _exclusive_fit_job(), threadpool_limits(limits=MAX_THREADS):
        d1 = _fit_stage("D1", d1_fit, d1_eval, approval_sha)
        d2 = _fit_stage("D2", d2_fit, d2_eval, approval_sha)
        # Disjoint W5/W6 evaluation views are pooled by IDs, not by averaging metrics.
        pooled = _pool_stage_results(d1, d2)
        stitch = make_unique_stitch(oof)
        present_final = ~stitch["missing"]
        final_ids = stitch["ids"][present_final]
        final_cycles = stitch["cycles"][present_final]
        resource_gate()
        y_final = _load_train_labels(final_ids, final_cycles)
        support_final = fit_support(y_final)
        if support_final["status"] != "PASS":
            for fit_id in ("FINAL_PAIR", "FINAL_SCALAR"):
                _write_json_once(_fit_paths(fit_id)[2], {
                    "experiment_id": EXPERIMENT, "fit_id": fit_id,
                    "status": "BLOCKED_UNSUPPORTED_SELECTION", "support": support_final,
                    "fit_started": False, "retry_allowed": False,
                })
            raise MetaError("Final unique OOF stitch fails the frozen support gate")
        final_receipts: dict[str, str] = {}
        for arm in ("PAIR", "SCALAR"):
            fit_id = f"FINAL_{arm}"
            x = _model_input(stitch, arm, present_final)
            state = _fit_one(fit_id, arm, x, y_final, final_ids, final_cycles, approval_sha)
            del state
            final_receipts[arm] = _receipt(_fit_paths(fit_id)[1])["receipt_sha256"]
        # The final meta state is frozen before even indexing E79 probabilities.
        app = _application_freeze(approval_sha, final_receipts)
        e79 = _load_test_probabilities_after_freeze()
        ids = e79["record_ids"].astype(str, copy=False)
        official_ids = _read_official_test_ids()
        if not np.array_equal(ids, official_ids):
            raise MetaError("E79 record IDs/order differ from official Test")
        if file_sha256(BASELINE_CSV) != app.get("mix50_baseline_csv_sha256"):
            raise MetaError("Frozen local MIX50 comparison CSV changed after application freeze")
        p_pos = e79["posoffset_probability"]
        p_a02 = e79["a02_probability"]
        missing = e79["missing_mask"]
        mix = np.where(missing, p_pos, .5 * p_pos + .5 * p_a02)
        final_states = {}
        for arm in ("PAIR", "SCALAR"):
            state_path, receipt_path, _ = _fit_paths(f"FINAL_{arm}")
            final_states[arm] = _read_model_state(state_path, _receipt(receipt_path))
        p_pair = policy_probabilities(p_pos, p_a02, missing, mix, "PAIR", final_states["PAIR"])
        p_scalar = policy_probabilities(p_pos, p_a02, missing, mix, "SCALAR", final_states["SCALAR"])
        labels_pair = (p_pair >= THRESHOLD).astype(np.int8)
        old_labels = _read_existing_csv(BASELINE_CSV, ids)
        duplicate = bool(np.array_equal(labels_pair, old_labels))
        csv_created = False
        if not duplicate:
            PAIR_CSV.parent.mkdir(parents=True, exist_ok=True)
            text = io.StringIO(newline="")
            writer = csv.writer(text, lineterminator="\n")
            writer.writerow(["record_id", "artifact_signal"])
            writer.writerows(zip(ids.tolist(), labels_pair.astype(int).tolist()))
            _write_bytes_once(PAIR_CSV, text.getvalue().encode("utf-8"))
            csv_created = True
        prob_path = RUN_DIR / "e79_meta_probabilities.npz"
        _write_bytes_once(prob_path, _npz_bytes(ids=ids, pair_probability=p_pair,
                                                scalar_probability=p_scalar,
                                                pair_labels=labels_pair,
                                                mix50_labels=old_labels))
        if csv_created:
            with PAIR_CSV.open("r", newline="", encoding="utf-8-sig") as stream:
                reader = csv.DictReader(stream)
                rows = list(reader)
            if reader.fieldnames != ["record_id", "artifact_signal"] or len(rows) != 29421:
                raise MetaError("New PAIR candidate CSV failed schema/row QA")
            if [r["record_id"] for r in rows] != ids.tolist():
                raise MetaError("New PAIR candidate ID/order QA failed")
            if any(r["artifact_signal"] not in {"0", "1"} for r in rows):
                raise MetaError("New PAIR candidate labels are not integer 0/1")
            qa = {
                "schema_version": 1, "experiment_id": EXPERIMENT,
                "status": "PASS_PAIR_CSV_INDEPENDENT_STRUCTURE_QA",
                "csv_path": PAIR_CSV.relative_to(ROOT).as_posix(),
                "csv_sha256": file_sha256(PAIR_CSV),
                "rows": len(rows), "id_order_matches_official_test": True,
                "ids_unique": len({r["record_id"] for r in rows}) == len(rows),
                "columns": reader.fieldnames,
                "integer_labels_0_1": True,
                "test_labels_loaded": False, "sample_submission_loaded": False,
                "kaggle_upload": False,
            }
        else:
            qa = {
                "schema_version": 1, "experiment_id": EXPERIMENT,
                "status": "SKIPPED_IDENTICAL_TO_MIX50",
                "csv_created": False, "identical_label_vector": True,
                "test_labels_loaded": False, "sample_submission_loaded": False,
                "kaggle_upload": False,
            }
        _write_json_once(INFERENCE_QA, qa)
        result = {
            "schema_version": 1, "experiment_id": EXPERIMENT,
            "status": "COMPLETED_DIAGNOSTIC_ONLY",
            "d1": d1, "d2": d2, "unique_pooled_w5_w6": pooled,
            "final_fit_support": support_final,
            "final_fit_receipts": final_receipts,
            "official_fit_count": len({e["fit_id"] for e in _ledger() if e.get("event") == "RESERVE"}),
            "official_fit_ids": [e["fit_id"] for e in _ledger() if e.get("event") == "RESERVE"],
            "test_rows": int(len(ids)), "test_positive_predictions": int(labels_pair.sum()),
            "candidate_csv_status": "SKIPPED_IDENTICAL_TO_MIX50" if duplicate else "CREATED_PAIR_ONLY",
            "candidate_csv_path": None if duplicate else PAIR_CSV.relative_to(ROOT).as_posix(),
            "candidate_csv_sha256": None if duplicate else file_sha256(PAIR_CSV),
            "scalar_csv_created": False, "kaggle_upload": False,
            "root_approval_sha256": approval_sha,
            "e79_application_freeze_sha256": app["receipt_sha256"],
            "contextual_counterevidence": _strict_json(PROTOCOL)["contextual_counterevidence"],
            "test_labels_loaded": False, "sample_submission_loaded": False,
        }
        _write_json_once(RESULTS, result)
        _write_json_once(MANIFEST, {
            "schema_version": 1, "experiment_id": EXPERIMENT,
            "status": "DIAGNOSTIC_ONLY_NOT_SUBMITTED",
            "source_commit": _git_head(), "source_hashes": _source_hashes(),
            "input_hashes": _input_hashes(), "root_approval_sha256": approval_sha,
            "results_sha256": file_sha256(RESULTS),
            "inference_qa_sha256": file_sha256(INFERENCE_QA),
            "e79_probabilities_sha256": file_sha256(prob_path),
            "csv_path": result["candidate_csv_path"],
            "csv_sha256": result["candidate_csv_sha256"],
            "fit_ids": list(FIT_IDS), "fit_count": 6,
            "test_labels_loaded": False, "sample_submission_loaded": False,
            "kaggle_upload": False,
        })
        return result


def _pool_stage_results(d1: Mapping[str, Any], d2: Mapping[str, Any]) -> dict[str, Any]:
    """Build pooled W5+W6 metrics from persisted raw probabilities and exact labels."""
    pred_data = []
    labels_data = []
    for stage, origin in (("D1", "67"), ("D2", "75")):
        pred_path = RUN_DIR / "stages" / stage / "predictions.npz"
        with np.load(pred_path, allow_pickle=False) as z:
            pred = {k: np.asarray(z[k]).copy() for k in z.files}
        y = _load_train_labels(pred["ids"], pred["cycles"])
        source = _load_oof_without_labels()[origin]
        by_id = {rid: ix for ix, rid in enumerate(source["ids"].tolist())}
        ix = np.asarray([by_id[rid] for rid in pred["ids"].tolist()], dtype=np.int64)
        groups = {"availability": source["group_availability"][ix],
                  "text_load": source["group_text_load"][ix],
                  "entry_friction": source["group_entry_friction"][ix]}
        pred_data.append((pred, y, groups))
        labels_data.append(y)
    pooled_y = np.concatenate([v[1] for v in pred_data])
    pooled_cycles = np.concatenate([v[0]["cycles"] for v in pred_data])
    result = {"rows": int(len(pooled_y)), "unique_ids": True, "arms": {}}
    pooled_ids = np.concatenate([v[0]["ids"] for v in pred_data])
    if len(np.unique(pooled_ids)) != len(pooled_ids):
        raise MetaError("W5/W6 pooled evaluation rows are not unique")
    for arm, key in (("MIX50", "baseline"), ("PAIR", "pair"), ("SCALAR", "scalar")):
        p = np.concatenate([v[0][key] for v in pred_data])
        groups = {name: np.concatenate([v[2][name] for v in pred_data])
                  for name in ("availability", "text_load", "entry_friction")}
        result["arms"][arm] = metric_bundle(pooled_y, p, pooled_cycles, groups)
    base_f1 = result["arms"]["MIX50"]["overall"]["macro_f1"]
    result["delta_macro_f1_vs_mix50"] = {
        arm: result["arms"][arm]["overall"]["macro_f1"] - base_f1
        for arm in ("PAIR", "SCALAR")
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-qa", action="store_true")
    parser.add_argument("--freeze-prefit", action="store_true")
    parser.add_argument("--run-official", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.synthetic_qa:
            print(json.dumps(synthetic_lifecycle(), indent=2))
        elif args.freeze_prefit:
            print(json.dumps(write_prefit_freeze(), indent=2))
        elif args.run_official:
            print(json.dumps(run_official(), indent=2))
        elif args.status:
            status = {"experiment_id": EXPERIMENT,
                      "prefit_freeze_exists": PREFIT_FREEZE.exists(),
                      "root_approval_exists": ROOT_APPROVAL.exists(),
                      "official_fit_starts": sum(e.get("event") == "FIT_START" for e in _ledger()),
                      "official_fit_completions": sum(e.get("event") == "FIT_COMPLETE" for e in _ledger()),
                      "official_fit_failures": sum(e.get("event") == "FIT_FAILED" for e in _ledger()),
                      "test_probabilities_loaded": E79_FREEZE.exists(),
                      "kaggle_upload": False}
            print(json.dumps(status, indent=2))
        else:
            parser.error("Choose --synthetic-qa, --freeze-prefit, --run-official, or --status")
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED_OR_FAILED", "type": type(exc).__name__,
                          "message": str(exc)}, indent=2), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
