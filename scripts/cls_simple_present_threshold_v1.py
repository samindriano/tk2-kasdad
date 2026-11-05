"""Bounded, past-only calibration of the canonical A02 present probability.

The source-ready/preflight commands are target-free. Official fit/score actions
require a separately issued root release and use selected-ID-only CSV reads.
This pilot fits a decision threshold only; it never fits a predictor.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
AUDIT = ROOT.parent / "tk2-simplification-audit-v1"
OUT = ROOT / "reports/optimization_v3/cls_simple_present_threshold_v1"
RUN = ROOT / "experiments/runs/cls_simple_present_threshold_v1"
FREEZE = OUT / "SOURCE_FREEZE.json"
PREFLIGHT = OUT / "SOURCE_PREFLIGHT.json"
READY = OUT / "SOURCE_READY.json"
LEDGER = RUN / "LEDGER.jsonl"
FIT_DIR = RUN / "fits"
SCORE_DIR = RUN / "scores"
DECISION = AUDIT / "reports/simplification_v1/CLS_SIMPLE_THRESHOLD_DECISION.md"
CACHE = ROOT / "experiments/runs/optimization_v3_posoffset_a02_endgame_v1/policy_predictions.npz"
SOURCE_PATHS = (
    "scripts/cls_simple_present_threshold_v1.py",
    "scripts/verify_cls_simple_present_threshold_v1.py",
    "tests/test_cls_simple_present_threshold_v1.py",
    "reports/optimization_v3/cls_simple_present_threshold_v1/PROTOCOL.md",
)
DECISION_SHA256 = "e0e0ac69ce7ce0e148e8edfe2345e21e37addf55807cdcff6b2f2995a08cd0c5"
CACHE_SHA256 = "25d1d2b91348a57d33b3b0e5ce78da29840d115b75e24f9502cc83c1b63ef351"
ROOT_SOURCE_COMMIT = "a39721172b6d293cb68ee0d120060082d222699e"
MISSING_THRESHOLD = 0.53230976129477
LOW, HIGH, CENTRE = 0.45, 0.55, 0.50
MIN_PRESENT_PER_CLASS = 200
STAGES = {
    "D1": {"origin": 67, "fit_cycles": (68, 71), "score_origin": 67, "score_cycles": (72, 75)},
    "D2": {"origin": 67, "fit_cycles": (68, 75), "score_origin": 75, "score_cycles": (76, 79)},
}
EXPECTED_ROWS = {67: 20_903, 75: 8_491}
EXPECTED_SCORE_ROWS = 15_953
OFFICIAL_TRAIN_ROWS = 50_608
OFFICIAL_CYCLE_RANGE = (47, 79)
RESOURCE_ESTIMATE_BYTES = 512 * 1024**2
RESOURCE_START_BYTES = int(1.5 * 1024**3)
RESOURCE_RESERVE_BYTES = int(0.75 * 1024**3)
SHARED_LOCK = Path(tempfile.gettempdir()) / "kasdad_single_heavy_job.lock"
LANE_LOCK = Path(tempfile.gettempdir()) / "kasdad_cls_simple_present_threshold_v1.lock"
MEMBERS = {
    origin: {
        "ids": f"E{origin}_ids", "cycles": f"E{origin}_cycles",
        "missing": f"E{origin}_missing", "pA": f"E{origin}_pA",
        "pP": f"E{origin}_pP", "availability": f"E{origin}_group_availability",
        "entry_friction": f"E{origin}_group_entry_friction",
        "text_load": f"E{origin}_group_text_load",
    }
    for origin in (67, 75)
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def ordered_id_hash(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode("utf-8")).hexdigest()


def write_json_exclusive(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(data).hexdigest()


def append_ledger(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(event, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    with path.open("ab") as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())


def read_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    output = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        event = json.loads(line)
        if not isinstance(event, dict) or "event_id" not in event:
            raise ValueError(f"Malformed ledger event at line {number}")
        output.append(event)
    ids = [item["event_id"] for item in output]
    if len(ids) != len(set(ids)):
        raise ValueError("Ledger has duplicate event IDs")
    return output


@contextmanager
def single_job_resource_gate():
    streams = []
    locked = []

    def acquire(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        stream = path.open("a+b")
        streams.append(stream)
        if path.stat().st_size == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"Required shared/lane lock is busy: {path}") from exc
        locked.append(stream)

    try:
        acquire(SHARED_LOCK)
        acquire(LANE_LOCK)
        try:
            import psutil
        except ImportError as exc:
            raise RuntimeError("psutil is required for the fit RAM gate") from exc
        available = int(psutil.virtual_memory().available)
        projected = available - RESOURCE_ESTIMATE_BYTES
        if available < RESOURCE_START_BYTES or projected < RESOURCE_RESERVE_BYTES:
            raise RuntimeError(
                f"RAM gate failed: available={available}, projected={projected}, "
                f"start_floor={RESOURCE_START_BYTES}, reserve_floor={RESOURCE_RESERVE_BYTES}"
            )
        receipt = {
            "shared_lock_path": str(SHARED_LOCK), "lane_lock_path": str(LANE_LOCK),
            "available_start_bytes": available, "estimate_bytes": RESOURCE_ESTIMATE_BYTES,
            "projected_bytes": projected, "start_floor_bytes": RESOURCE_START_BYTES,
            "reserve_floor_bytes": RESOURCE_RESERVE_BYTES,
        }
        yield receipt
        end = int(psutil.virtual_memory().available)
        if end < RESOURCE_RESERVE_BYTES:
            raise RuntimeError(f"RAM reserve failed at fit boundary: available={end}")
        receipt.update({"available_end_bytes": end, "reserve_pass": True})
    finally:
        for stream in reversed(locked):
            try:
                stream.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
        for stream in streams:
            stream.close()


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def _require_root_ancestry() -> dict[str, str]:
    head = _git(AUDIT, "rev-parse", "HEAD")
    status = _git(AUDIT, "status", "--porcelain")
    if status:
        raise RuntimeError("Root decision repository is dirty; refusing to freeze")
    result = subprocess.run(
        ["git", "-C", str(AUDIT), "merge-base", "--is-ancestor", ROOT_SOURCE_COMMIT, "HEAD"],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError("Required root source commit is not an ancestor of the decision checkout")
    if sha256_file(DECISION) != DECISION_SHA256:
        raise RuntimeError("Root decision content hash differs from the declared decision")
    return {"required_ancestor": ROOT_SOURCE_COMMIT, "artifact_head": head, "working_tree_clean": True}


def _source_hashes() -> dict[str, str]:
    return {relative: sha256_file(ROOT / relative) for relative in SOURCE_PATHS}


def _verify_committed_source_snapshot(commit: str, hashes: dict[str, str]) -> None:
    for relative in SOURCE_PATHS:
        payload = subprocess.run(
            ["git", "-C", str(ROOT), "show", f"{commit}:{relative}"],
            check=True, capture_output=True,
        ).stdout
        if hashlib.sha256(payload).hexdigest() != hashes.get(relative):
            raise RuntimeError(f"Committed source snapshot differs from working file: {relative}")


def freeze_sources(train_csv: Path, cache: Path = CACHE, decision: Path = DECISION) -> dict[str, Any]:
    if FREEZE.exists():
        raise FileExistsError(f"Refusing to overwrite source freeze: {FREEZE}")
    if decision.resolve() != DECISION.resolve() or sha256_file(decision) != DECISION_SHA256:
        raise RuntimeError("Root decision path/content is not the frozen decision")
    root_lineage = _require_root_ancestry()
    if sha256_file(cache) != CACHE_SHA256:
        raise RuntimeError("A02 OOF cache hash differs from the verified input")
    source_hashes = _source_hashes()
    source_commit = _git(ROOT, "rev-parse", "HEAD")
    if _git(ROOT, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Tracked source tree must be committed before source freeze")
    _verify_committed_source_snapshot(source_commit, source_hashes)
    try:
        import pandas
        pandas_version = pandas.__version__
    except ImportError:
        pandas_version = None
    freeze = {
        "schema_version": 1,
        "status": "SOURCE_FROZEN_BEFORE_TARGET_FREE_PREFLIGHT",
        "campaign": "CLS_SIMPLE_PRESENT_THRESHOLD_V1",
        "source_commit": source_commit,
        "source_hashes": source_hashes,
        "inputs": {
            "official_train_csv": {"path_at_freeze": str(train_csv.resolve()), "sha256": sha256_file(train_csv)},
            "a02_oof_cache": {"sha256": sha256_file(cache), "required_sha256": CACHE_SHA256},
            "root_decision": {"sha256": sha256_file(decision), "required_sha256": DECISION_SHA256},
        },
        "root_lineage": root_lineage,
        "recipe": {
            "score": "canonical OOF pA",
            "present_threshold_interval_inclusive": [LOW, HIGH],
            "threshold_operator": ">=",
            "objective": "full-row binary Macro F1 including fixed missing decisions",
            "missing_decision": f"pA >= {MISSING_THRESHOLD}",
            "minimum_present_fit_support_each_class": MIN_PRESENT_PER_CLASS,
            "stages": STAGES,
            "comparators": ["pA threshold 0.50", "0.5*pP + 0.5*pA present threshold 0.50"],
            "tie_break": ["maximum exact Macro F1", "minimum distance to 0.50", "higher threshold"],
        },
        "python": sys.version.split()[0],
        "pandas": pandas_version,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    digest = write_json_exclusive(FREEZE, freeze)
    freeze["source_freeze_sha256"] = digest
    # Freeze is immutable and hashes its pre-self-hash bytes; receipt field makes that convention explicit.
    return freeze


def _read_metadata(train_csv: Path):
    import pandas as pd
    needed = ["record_id", "cycle_code", "signal_stream", "entry_friction", "text_load"]
    columns = pd.read_csv(train_csv, nrows=0).columns.tolist()
    if not set(needed + ["artifact_signal"]).issubset(columns):
        raise ValueError("Official Train schema is missing required columns")
    frame = pd.read_csv(
        train_csv, usecols=needed,
        dtype={"record_id": "string", "signal_stream": "string"},
        low_memory=False,
    )
    if frame["record_id"].isna().any():
        raise ValueError("Official Train record_id is missing")
    frame["record_id"] = frame["record_id"].astype(str)
    if frame["record_id"].duplicated().any():
        raise ValueError("Official Train record_id is missing or duplicated")
    frame["cycle_code"] = pd.to_numeric(frame["cycle_code"], errors="raise").astype(int)
    frame["entry_friction"] = pd.to_numeric(frame["entry_friction"], errors="raise").astype(int)
    frame["text_load"] = pd.to_numeric(frame["text_load"], errors="raise").astype(float)
    if frame["text_load"].isna().any() or not frame["entry_friction"].isin([0, 1]).all():
        raise ValueError("Metadata values are missing or outside the frozen schema")
    return frame


def _load_oof_metadata(cache: Path) -> dict[int, dict[str, np.ndarray]]:
    data: dict[int, dict[str, np.ndarray]] = {}
    with np.load(cache, allow_pickle=False) as archive:
        for origin, names in MEMBERS.items():
            row = {}
            for key, member in names.items():
                row[key] = np.asarray(archive[member])
            ids = row["ids"].astype(str)
            cycles = row["cycles"].astype(np.int64)
            missing = row["missing"].astype(bool)
            p_a = row["pA"].astype(np.float64)
            p_p = row["pP"].astype(np.float64)
            groups = {k: row[k].astype(str) for k in ("availability", "entry_friction", "text_load")}
            n = len(ids)
            if len(set(ids.tolist())) != n or any(len(x) != n for x in [cycles, missing, p_a, p_p, *groups.values()]):
                raise ValueError(f"E{origin} cached non-target vectors have mismatched/duplicate IDs")
            if not np.isfinite(p_a).all() or ((p_a < 0) | (p_a > 1)).any():
                raise ValueError(f"E{origin} pA is not a finite probability")
            if not np.isfinite(p_p[~missing]).all() or ((p_p[~missing] < 0) | (p_p[~missing] > 1)).any():
                raise ValueError(f"E{origin} present pP is invalid")
            allowed_cycles = np.arange(68, 80) if origin == 67 else np.arange(76, 80)
            if not np.isin(cycles, allowed_cycles).all():
                raise ValueError(f"E{origin} cached cycle range is invalid")
            if set(groups["availability"]) != {"present", "missing"} or not np.array_equal(groups["availability"] == "missing", missing):
                raise ValueError(f"E{origin} availability groups disagree with missing mask")
            if not np.isin(groups["entry_friction"], ["0", "1"]).all():
                raise ValueError(f"E{origin} entry_friction groups invalid")
            if not np.isin(groups["text_load"], ["Q1", "Q2", "Q3", "Q4"]).all():
                raise ValueError(f"E{origin} text_load groups invalid")
            data[origin] = {"ids": ids, "cycles": cycles, "missing": missing, "pA": p_a, "pP": p_p, **groups}
    return data


def _derive_text_groups(train, origin: int, row_indices: np.ndarray) -> np.ndarray:
    values = train["text_load"].to_numpy(dtype=np.float64)
    prefix = values[train["cycle_code"].to_numpy(dtype=int) <= origin]
    if len(prefix) == 0:
        raise ValueError(f"No prefix text_load values for origin E{origin}")
    edges = np.quantile(prefix, [0.25, 0.50, 0.75], method="linear")
    return np.asarray(["Q" + str(int(x) + 1) for x in np.searchsorted(edges, values[row_indices], side="right")])


def source_preflight(train_csv: Path, cache: Path = CACHE) -> dict[str, Any]:
    if not FREEZE.exists():
        raise RuntimeError("Source freeze is required before opening cached vectors")
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    if freeze["inputs"]["official_train_csv"]["sha256"] != sha256_file(train_csv):
        raise RuntimeError("Official Train hash differs from source freeze")
    if freeze["inputs"]["a02_oof_cache"]["sha256"] != sha256_file(cache):
        raise RuntimeError("OOF cache hash differs from source freeze")
    if freeze["source_hashes"] != _source_hashes():
        raise RuntimeError("Source hashes changed after freeze")
    _verify_committed_source_snapshot(freeze["source_commit"], freeze["source_hashes"])
    current_root_lineage = _require_root_ancestry()
    if (
        current_root_lineage["required_ancestor"] != freeze["root_lineage"]["required_ancestor"]
        or current_root_lineage["working_tree_clean"] is not True
    ):
        raise RuntimeError("Root decision lineage changed after freeze")
    # Confirm physical CSV rows map one-to-one to pandas record rows before any later selected-ID label read.
    raw = train_csv.read_bytes()
    physical_lines = raw.count(b"\n") + (0 if raw.endswith(b"\n") else 1)
    train = _read_metadata(train_csv)
    if physical_lines != len(train) + 1:
        raise RuntimeError("CSV has quoted/multiline records; selected-row label parser cannot prove physical row mapping")
    if len(train) != OFFICIAL_TRAIN_ROWS:
        raise RuntimeError(f"Official Train row count changed: {len(train)}")
    if train["cycle_code"].min() != OFFICIAL_CYCLE_RANGE[0] or train["cycle_code"].max() != OFFICIAL_CYCLE_RANGE[1]:
        raise RuntimeError("Official Train cycle bounds differ from the frozen source")
    by_id = {value: index for index, value in enumerate(train["record_id"].tolist())}
    cached = _load_oof_metadata(cache)
    receipts: dict[str, Any] = {}
    all_fit_ids: set[str] = set()
    all_score_ids: set[str] = set()
    for origin, arrays in cached.items():
        ids = arrays["ids"]
        if not set(ids.tolist()).issubset(by_id):
            raise ValueError(f"E{origin} OOF contains IDs absent from official Train")
        positions = np.asarray([by_id[item] for item in ids], dtype=np.int64)
        official_cycles = train["cycle_code"].to_numpy(dtype=int)[positions]
        if not np.array_equal(official_cycles, arrays["cycles"]):
            raise ValueError(f"E{origin} cycle/ID mapping differs from official Train")
        official_missing = train["signal_stream"].isna().to_numpy()[positions] | (
            train["signal_stream"].fillna("").astype(str).to_numpy()[positions] == ""
        )
        if not np.array_equal(official_missing, arrays["missing"]):
            raise ValueError(f"E{origin} stream-availability mapping differs from official Train")
        official_entry = train["entry_friction"].astype(int).astype(str).to_numpy()[positions]
        if not np.array_equal(official_entry, arrays["entry_friction"]):
            raise ValueError(f"E{origin} entry_friction groups differ from official Train")
        derived_text = _derive_text_groups(train, origin, positions)
        if not np.array_equal(derived_text, arrays["text_load"]):
            raise ValueError(f"E{origin} prefix-defined text_load groups differ from official Train")
        expected_n = EXPECTED_ROWS[origin]
        expected_cycles = list(range(68, 80)) if origin == 67 else list(range(76, 80))
        if len(ids) != expected_n or sorted(np.unique(arrays["cycles"]).tolist()) != expected_cycles:
            raise ValueError(f"E{origin} source row/cycle support differs from frozen plan")
        receipts[str(origin)] = {
            "rows": len(ids),
            "cycles": {str(c): int(np.sum(arrays["cycles"] == c)) for c in expected_cycles},
            "ordered_id_sha256": ordered_id_hash(ids),
            "official_train_row_positions_sha256": hashlib.sha256(positions.astype("<i8").tobytes()).hexdigest(),
            "present_rows": int((~arrays["missing"]).sum()),
            "missing_rows": int(arrays["missing"].sum()),
            "pA_range": [float(arrays["pA"].min()), float(arrays["pA"].max())],
            "pP_present_range": [float(arrays["pP"][~arrays["missing"]].min()), float(arrays["pP"][~arrays["missing"]].max())],
            "groups_recomputed_from_official_non_target_columns": True,
            "target_values_accessed": False,
        }
    for stage, spec in STAGES.items():
        f_origin, s_origin = spec["origin"], spec["score_origin"]
        fit_mask = np.isin(cached[f_origin]["cycles"], np.arange(spec["fit_cycles"][0], spec["fit_cycles"][1] + 1))
        score_mask = np.isin(cached[s_origin]["cycles"], np.arange(spec["score_cycles"][0], spec["score_cycles"][1] + 1))
        fit_ids = set(cached[f_origin]["ids"][fit_mask].tolist())
        score_ids = set(cached[s_origin]["ids"][score_mask].tolist())
        if not fit_ids or not score_ids or fit_ids & score_ids:
            raise ValueError(f"{stage} fit/score ID sets are empty or overlapping")
        all_fit_ids.update(fit_ids)
        all_score_ids.update(score_ids)
        receipts[stage] = {
            "fit_rows": len(fit_ids), "fit_cycles": list(spec["fit_cycles"]),
            "fit_ids_sha256": ordered_id_hash(cached[f_origin]["ids"][fit_mask]),
            "score_rows": len(score_ids), "score_cycles": list(spec["score_cycles"]),
            "score_ids_sha256": ordered_id_hash(cached[s_origin]["ids"][score_mask]),
            "fit_score_ids_disjoint": True,
        }
    score_windows = [
        set(cached[67]["ids"][np.isin(cached[67]["cycles"], [72, 73, 74, 75])].tolist()),
        set(cached[75]["ids"][np.isin(cached[75]["cycles"], [76, 77, 78, 79])].tolist()),
    ]
    if len(score_windows[0] | score_windows[1]) != EXPECTED_SCORE_ROWS or score_windows[0] & score_windows[1]:
        raise ValueError("D1/D2 score windows are not the declared 15,953 disjoint rows")
    result = {
        "schema_version": 1,
        "status": "PASS_TARGET_FREE_PREFLIGHT",
        "campaign": "CLS_SIMPLE_PRESENT_THRESHOLD_V1",
        "source_commit": freeze["source_commit"],
        "source_freeze_sha256": sha256_file(FREEZE),
        "official_train_sha256": sha256_file(train_csv),
        "oof_cache_sha256": sha256_file(cache),
        "rows": len(train),
        "official_target_column_values_read": False,
        "test_or_sample_submission_accessed": False,
        "origins": receipts,
        "combined_score_rows": len(score_windows[0] | score_windows[1]),
        "all_folds_train_id_membership_and_cycle_mapping_verified": True,
        "availability_and_prefix_groups_recomputed": True,
        "oof_target_members_opened": False,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if PREFLIGHT.exists():
        raise FileExistsError(f"Refusing to overwrite target-free preflight: {PREFLIGHT}")
    write_json_exclusive(PREFLIGHT, result)
    return result


def macro_f1_from_counts(tp: int, fp: int, tn: int, fn: int) -> Fraction:
    positive = Fraction(2 * tp, 2 * tp + fp + fn) if 2 * tp + fp + fn else Fraction(0, 1)
    negative = Fraction(2 * tn, 2 * tn + fp + fn) if 2 * tn + fp + fn else Fraction(0, 1)
    return (positive + negative) / 2


def _counts(y: np.ndarray, pred: np.ndarray) -> tuple[int, int, int, int]:
    y = np.asarray(y, dtype=np.int8)
    pred = np.asarray(pred, dtype=np.int8)
    return (
        int(np.sum((y == 1) & (pred == 1))),
        int(np.sum((y == 0) & (pred == 1))),
        int(np.sum((y == 0) & (pred == 0))),
        int(np.sum((y == 1) & (pred == 0))),
    )


def exact_threshold_fit(p_a: np.ndarray, missing: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    p_a = np.asarray(p_a, dtype=np.float64)
    missing = np.asarray(missing, dtype=bool)
    y = np.asarray(y, dtype=np.int8)
    if not (len(p_a) == len(missing) == len(y)) or len(y) == 0:
        raise ValueError("Calibration inputs are empty or have different lengths")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p_a).all() or ((p_a < 0) | (p_a > 1)).any():
        raise ValueError("Calibration labels/probabilities are invalid")
    present = ~missing
    support = {str(c): int(np.sum(present & (y == c))) for c in (0, 1)}
    if min(support.values()) < MIN_PRESENT_PER_CLASS:
        raise RuntimeError(f"Unsupported calibration population: present class counts={support}")
    base_pred = np.zeros(len(y), dtype=np.int8)
    base_pred[missing] = (p_a[missing] >= MISSING_THRESHOLD).astype(np.int8)
    base_pred[present] = (p_a[present] >= LOW).astype(np.int8)
    tp, fp, tn, fn = _counts(y, base_pred)
    present_scores = p_a[present]
    inside_scores = np.unique(present_scores[(present_scores >= LOW) & (present_scores <= HIGH)])
    best: tuple[Fraction, float, tuple[int, int, int, int], tuple[float, float]] | None = None
    state_start = LOW

    def consider(interval_low: float, interval_high: float, counts: tuple[int, int, int, int]) -> None:
        nonlocal best
        if interval_low > interval_high:
            return
        threshold = min(max(CENTRE, interval_low), interval_high)
        objective = macro_f1_from_counts(*counts)
        interval = (interval_low, interval_high)
        candidate = (objective, threshold, counts, interval)
        if best is None or objective > best[0] or (
            objective == best[0] and (abs(threshold - CENTRE), -threshold) < (abs(best[1] - CENTRE), -best[1])
        ):
            best = candidate

    for score in inside_scores:
        s = float(score)
        if state_start <= s:
            consider(state_start, s, (tp, fp, tn, fn))
        dropped = present & (p_a == s)
        group_positive = int(np.sum(dropped & (y == 1)))
        group_negative = int(np.sum(dropped & (y == 0)))
        tp -= group_positive
        fp -= group_negative
        fn += group_positive
        tn += group_negative
        state_start = float(np.nextafter(s, math.inf))
    consider(state_start, HIGH, (tp, fp, tn, fn))
    if best is None:
        raise RuntimeError("Exact threshold optimizer found no representable state")
    objective, threshold, chosen_counts, interval = best
    decisions = np.zeros(len(y), dtype=np.int8)
    decisions[missing] = (p_a[missing] >= MISSING_THRESHOLD).astype(np.int8)
    decisions[present] = (p_a[present] >= threshold).astype(np.int8)
    actual_counts = _counts(y, decisions)
    if actual_counts != chosen_counts:
        raise AssertionError(f"Selected threshold does not realize optimizer counts: {actual_counts} != {chosen_counts}")
    return {
        "threshold": threshold,
        "threshold_hex": threshold.hex(),
        "decision_state_interval": [interval[0], interval[1]],
        "fit_macro_f1": float(objective),
        "fit_macro_f1_exact": f"{objective.numerator}/{objective.denominator}",
        "fit_confusion": {"tp": chosen_counts[0], "fp": chosen_counts[1], "tn": chosen_counts[2], "fn": chosen_counts[3]},
        "present_class_support": support,
        "present_score_breakpoints_considered": len(inside_scores),
        "predictions": decisions,
    }


def _read_selected_labels(train_csv: Path, all_ids: np.ndarray, selected_ids: np.ndarray) -> np.ndarray:
    """Read target values only for IDs in this frozen stage, using physical row positions."""
    import pandas as pd
    all_ids = np.asarray(all_ids).astype(str)
    selected_ids = np.asarray(selected_ids).astype(str)
    index = {record_id: position for position, record_id in enumerate(all_ids.tolist())}
    if len(index) != len(all_ids) or not set(selected_ids.tolist()).issubset(index):
        raise ValueError("Selected label IDs are duplicated or absent from the verified Train roster")
    selected_positions = {index[item] for item in selected_ids.tolist()}
    raw = train_csv.read_bytes()
    physical_lines = raw.count(b"\n") + (0 if raw.endswith(b"\n") else 1)
    if physical_lines != len(all_ids) + 1:
        raise ValueError("Physical CSV row mapping failed before selected target read")
    frame = pd.read_csv(
        train_csv,
        usecols=["record_id", "artifact_signal"],
        dtype={"record_id": "string", "artifact_signal": "Int8"},
        skiprows=lambda line: line > 0 and (line - 1) not in selected_positions,
        low_memory=False,
    )
    frame["record_id"] = frame["record_id"].astype(str)
    if frame["record_id"].duplicated().any() or set(frame["record_id"]) != set(selected_ids.tolist()):
        raise ValueError("Selected-only label reader returned unexpected IDs")
    y_map = dict(zip(frame["record_id"], frame["artifact_signal"].astype(int)))
    y = np.asarray([y_map[item] for item in selected_ids.tolist()], dtype=np.int8)
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Selected target labels are not binary")
    return y


def _stage_vectors(stage: str, train_csv: Path, cache: Path) -> dict[str, np.ndarray]:
    spec = STAGES[stage]
    arrays = _load_oof_metadata(cache)
    fit = arrays[spec["origin"]]
    score = arrays[spec["score_origin"]]
    fit_mask = (fit["cycles"] >= spec["fit_cycles"][0]) & (fit["cycles"] <= spec["fit_cycles"][1])
    score_mask = (score["cycles"] >= spec["score_cycles"][0]) & (score["cycles"] <= spec["score_cycles"][1])
    train_ids = _read_metadata(train_csv)["record_id"].to_numpy(dtype=str)
    fit_ids, score_ids = fit["ids"][fit_mask], score["ids"][score_mask]
    if set(fit_ids.tolist()) & set(score_ids.tolist()):
        raise ValueError(f"{stage} fit and score IDs overlap")
    return {
        "fit_ids": fit_ids, "fit_cycles": fit["cycles"][fit_mask], "fit_missing": fit["missing"][fit_mask],
        "fit_pA": fit["pA"][fit_mask], "fit_pP": fit["pP"][fit_mask],
        "score_ids": score_ids, "score_cycles": score["cycles"][score_mask],
        "score_missing": score["missing"][score_mask], "score_pA": score["pA"][score_mask],
        "score_pP": score["pP"][score_mask],
        "train_ids": train_ids,
    }


def _validate_release(release_path: Path) -> dict[str, Any]:
    if not READY.exists():
        raise RuntimeError("Independent source-ready receipt is required before official work")
    release = json.loads(release_path.read_text(encoding="utf-8"))
    ready = json.loads(READY.read_text(encoding="utf-8"))
    if release.get("status") != "ROOT_RELEASED_FOR_D1_D2" or release.get("source_ready_sha256") != sha256_file(READY):
        raise RuntimeError("Root release missing, wrong status, or bound to another source-ready receipt")
    if release.get("allowed_stages") != ["D1", "D2"] or ready.get("status") != "PASS":
        raise RuntimeError("Root release stage scope or ready receipt is invalid")
    return release


def _verify_runtime_lineage(train_csv: Path, cache: Path) -> None:
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    if train_csv.resolve() != Path(freeze["inputs"]["official_train_csv"]["path_at_freeze"]).resolve():
        raise RuntimeError("Runtime Train path differs from the frozen official source")
    if sha256_file(train_csv) != freeze["inputs"]["official_train_csv"]["sha256"]:
        raise RuntimeError("Runtime Train hash differs from source freeze")
    if sha256_file(cache) != freeze["inputs"]["a02_oof_cache"]["sha256"] or sha256_file(cache) != CACHE_SHA256:
        raise RuntimeError("Runtime OOF cache hash differs from source freeze")
    if freeze["source_hashes"] != _source_hashes():
        raise RuntimeError("Runtime source files differ from source freeze")
    _require_root_ancestry()


def _fit_stage_under_gate(stage: str, train_csv: Path, cache: Path, release_path: Path) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError("Only predeclared diagnostic stages D1 and D2 are allowed")
    _validate_release(release_path)
    _verify_runtime_lineage(train_csv, cache)
    receipt_path = FIT_DIR / f"{stage}.json"
    events = read_ledger(LEDGER)
    starts = [e for e in events if e.get("stage") == stage and e.get("event") == "FIT_STARTED"]
    done = [e for e in events if e.get("stage") == stage and e.get("event") in {"FIT_FROZEN", "FIT_RECOVERED", "FIT_UNSUPPORTED"}]
    if done:
        if not receipt_path.exists() or sha256_file(receipt_path) != done[-1].get("receipt_sha256"):
            raise RuntimeError(f"{stage} ledger/artifact mismatch; refusing resume")
        return json.loads(receipt_path.read_text(encoding="utf-8"))
    if starts:
        if receipt_path.exists():
            recovered = json.loads(receipt_path.read_text(encoding="utf-8"))
            if (
                recovered.get("status") == "FROZEN_PREDICTION_THRESHOLD"
                and recovered.get("stage") == stage
                and recovered.get("fit_ids_sha256") == starts[-1].get("fit_ids_sha256")
                and LOW <= float(recovered.get("threshold", math.nan)) <= HIGH
                and recovered.get("threshold_hex") == float(recovered["threshold"]).hex()
                and recovered.get("oof_cache_sha256") == sha256_file(cache)
                and recovered.get("official_train_sha256") == sha256_file(train_csv)
                and recovered.get("source_commit") == json.loads(FREEZE.read_text(encoding="utf-8"))["source_commit"]
                and recovered.get("source_freeze_sha256") == sha256_file(FREEZE)
                and starts[-1].get("oof_cache_sha256") == sha256_file(cache)
                and starts[-1].get("official_train_sha256") == sha256_file(train_csv)
            ):
                digest = sha256_file(receipt_path)
                append_ledger(LEDGER, {"event_id": f"{stage}:FIT_RECOVERED", "event": "FIT_RECOVERED",
                                       "stage": stage, "receipt_sha256": digest,
                                       "recovered_after_frozen_artifact": True,
                                       "at_utc": datetime.now(timezone.utc).isoformat()})
                return recovered
        raise RuntimeError(f"{stage} has ambiguous FIT_STARTED without a valid durable completion; refusing retry")
    if receipt_path.exists():
        raise RuntimeError(f"{stage} has an orphan fit artifact without a ledger start; refusing overwrite")
    vectors = _stage_vectors(stage, train_csv, cache)
    append_ledger(LEDGER, {"event_id": f"{stage}:FIT_STARTED", "event": "FIT_STARTED", "stage": stage,
                           "fit_ids_sha256": ordered_id_hash(vectors["fit_ids"]),
                           "official_train_sha256": sha256_file(train_csv),
                           "oof_cache_sha256": sha256_file(cache),
                           "source_freeze_sha256": sha256_file(FREEZE),
                           "at_utc": datetime.now(timezone.utc).isoformat()})
    fit_y = _read_selected_labels(train_csv, vectors["train_ids"], vectors["fit_ids"])
    try:
        optimized = exact_threshold_fit(vectors["fit_pA"], vectors["fit_missing"], fit_y)
    except RuntimeError as exc:
        if not str(exc).startswith("Unsupported calibration population:"):
            raise
        unsupported = {
            "status": "UNSUPPORTED_CALIBRATION_POPULATION", "stage": stage,
            "fit_rows": len(vectors["fit_ids"]), "fit_ids_sha256": ordered_id_hash(vectors["fit_ids"]),
            "present_class_support": {str(c): int(np.sum((~vectors["fit_missing"]) & (fit_y == c))) for c in (0, 1)},
        }
        digest = write_json_exclusive(receipt_path, unsupported)
        append_ledger(LEDGER, {"event_id": f"{stage}:FIT_UNSUPPORTED", "event": "FIT_UNSUPPORTED", "stage": stage,
                               "receipt_sha256": digest, "at_utc": datetime.now(timezone.utc).isoformat()})
        return unsupported
    artifact = {
        "schema_version": 1, "status": "FROZEN_PREDICTION_THRESHOLD",
        "stage": stage, "origin": STAGES[stage]["origin"],
        "fit_cycles": list(STAGES[stage]["fit_cycles"]),
        "fit_rows": len(vectors["fit_ids"]),
        "fit_ids_sha256": ordered_id_hash(vectors["fit_ids"]),
        "fit_cycle_counts": {str(c): int(np.sum(vectors["fit_cycles"] == c)) for c in np.unique(vectors["fit_cycles"])},
        "fit_source": "selected official Train artifact_signal values only",
        "cache_target_member_opened": False,
        "source_commit": json.loads(FREEZE.read_text(encoding="utf-8"))["source_commit"],
        "source_freeze_sha256": sha256_file(FREEZE),
        "official_train_sha256": sha256_file(train_csv),
        "oof_cache_sha256": sha256_file(cache),
        "threshold": optimized["threshold"], "threshold_hex": optimized["threshold_hex"],
        "decision_state_interval": optimized["decision_state_interval"],
        "fit_macro_f1": optimized["fit_macro_f1"],
        "fit_macro_f1_exact": optimized["fit_macro_f1_exact"],
        "fit_confusion": optimized["fit_confusion"],
        "present_class_support": optimized["present_class_support"],
        "breakpoints_considered": optimized["present_score_breakpoints_considered"],
        "recipe": "CLS_SIMPLE_PRESENT_THRESHOLD_V1",
    }
    digest = write_json_exclusive(receipt_path, artifact)
    append_ledger(LEDGER, {"event_id": f"{stage}:FIT_FROZEN", "event": "FIT_FROZEN", "stage": stage,
                           "receipt_sha256": digest, "at_utc": datetime.now(timezone.utc).isoformat()})
    return artifact


def fit_stage(stage: str, train_csv: Path, cache: Path, release_path: Path) -> dict[str, Any]:
    with single_job_resource_gate():
        return _fit_stage_under_gate(stage, train_csv, cache, release_path)


def _metric(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    tp, fp, tn, fn = _counts(y, pred)
    return _metric_from_confusion(len(y), int(np.sum(y == 0)), int(np.sum(y == 1)), tp, fp, tn, fn)


def _metric_from_confusion(n: int, class_0: int, class_1: int, tp: int, fp: int, tn: int, fn: int) -> dict[str, Any]:
    exact = macro_f1_from_counts(tp, fp, tn, fn)
    return {
        "n": int(n), "class_0": int(class_0), "class_1": int(class_1),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "macro_f1": float(exact), "macro_f1_exact": f"{exact.numerator}/{exact.denominator}",
        "positive_precision": tp / (tp + fp) if tp + fp else 0.0,
        "positive_recall": tp / (tp + fn) if tp + fn else 0.0,
    }


def _group_metrics(y: np.ndarray, pred: np.ndarray, groups: dict[str, np.ndarray]) -> dict[str, Any]:
    result = {}
    for dimension, labels in groups.items():
        detail = {}
        for label in sorted(set(labels.tolist())):
            mask = labels == label
            support = {"n": int(mask.sum()), "class_0": int(np.sum(y[mask] == 0)), "class_1": int(np.sum(y[mask] == 1))}
            detail[label] = {"support": support, "status": "SUPPORTED" if support["n"] >= 200 and min(support["class_0"], support["class_1"]) >= 30 else "UNKNOWN",
                             "metrics": _metric(y[mask], pred[mask]) if mask.any() else None}
        result[dimension] = detail
    return result


def summarize_scores() -> dict[str, Any]:
    paths = {stage: SCORE_DIR / f"{stage}_RESULTS.json" for stage in STAGES}
    if not all(path.exists() for path in paths.values()):
        raise RuntimeError("Both predeclared D1 and D2 score receipts are required for pooling")
    reports = {stage: json.loads(path.read_text(encoding="utf-8")) for stage, path in paths.items()}
    for stage, report in reports.items():
        if report.get("status") != "SCORED" or report.get("stage") != stage:
            raise ValueError(f"{stage} score receipt identity/status mismatch")

    pooled_metrics: dict[str, Any] = {}
    for view in ("full", "present"):
        pooled_metrics[view] = {}
        for model in ("candidate", "simple_fixed_050", "complex_fixed_050"):
            source_metrics = [reports[s]["metrics"][view][model] for s in ("D1", "D2")]
            total = {key: sum(int(item[key]) for item in source_metrics)
                     for key in ("n", "class_0", "class_1", "tp", "fp", "tn", "fn")}
            pooled_metrics[view][model] = _metric_from_confusion(
                total["n"], total["class_0"], total["class_1"],
                total["tp"], total["fp"], total["tn"], total["fn"],
            )
    deltas = {
        stage: reports[stage]["metrics"]["full"]["candidate"]["macro_f1"]
        - reports[stage]["metrics"]["full"]["simple_fixed_050"]["macro_f1"]
        for stage in ("D1", "D2")
    }
    pooled_delta = (
        pooled_metrics["full"]["candidate"]["macro_f1"]
        - pooled_metrics["full"]["simple_fixed_050"]["macro_f1"]
    )
    subgroup_guard_rows = []
    for stage in ("D1", "D2"):
        for dimension, by_label in reports[stage]["subgroup_deltas_vs_simple_fixed_050"].items():
            for label, detail in by_label.items():
                subgroup_guard_rows.append({
                    "stage": stage, "dimension": dimension, "group": label,
                    "status": detail["status"],
                    "macro_f1_delta": detail.get("macro_f1_delta"),
                    "positive_recall_delta": detail.get("positive_recall_delta"),
                })
    supported = [row for row in subgroup_guard_rows if row["status"] == "SUPPORTED"]
    subgroup_ok = all(
        row["macro_f1_delta"] >= -0.01 and row["positive_recall_delta"] >= -0.02
        for row in supported
    )
    qualification = pooled_delta >= 0.0003 and all(value >= 0 for value in deltas.values()) and subgroup_ok
    exploratory = pooled_delta >= 0 and all(value >= -0.001 for value in deltas.values()) and subgroup_ok
    thresholds = {stage: float(reports[stage]["fit"]["threshold"]) for stage in ("D1", "D2")}
    result = {
        "schema_version": 1, "status": "POOLED_REVIEW_READY",
        "campaign": "CLS_SIMPLE_PRESENT_THRESHOLD_V1",
        "score_rows": sum(int(reports[s]["score_rows"]) for s in ("D1", "D2")),
        "score_ids_disjoint": True,
        "thresholds": thresholds,
        "threshold_absolute_difference_D1_D2": abs(thresholds["D1"] - thresholds["D2"]),
        "thresholds_at_bounds": {stage: reports[stage]["fit"]["threshold_at_interval_bound"] for stage in reports},
        "per_window_full_macro_f1_gain_vs_simple_fixed_050": deltas,
        "pooled_metrics": pooled_metrics,
        "pooled_full_macro_f1_gain_vs_simple_fixed_050": pooled_delta,
        "subgroup_guards": {
            "support_rule": "n>=200 and >=30 rows per class in the score window",
            "supported_groups_checked": len(supported),
            "unknown_groups": len(subgroup_guard_rows) - len(supported),
            "all_supported_groups_within_harm_limits": subgroup_ok,
            "limits": {"macro_f1_harm_max": 0.01, "positive_recall_harm_max": 0.02},
            "details": subgroup_guard_rows,
        },
        "qualification": "PASS" if qualification else "FAIL",
        "exploratory_review": "PASS" if exploratory else "FAIL",
        "qualification_rule": "pooled gain>=0.0003; D1 and D2 gain>=0; all supported subgroup guards pass",
        "exploratory_rule": "pooled gain>=0; D1 and D2 gain>=-0.001; all supported subgroup guards pass",
        "complex_comparator_pooled_macro_f1": pooled_metrics["full"]["complex_fixed_050"]["macro_f1"],
        "champion_changed": False,
        "test_or_sample_submission_accessed": False,
        "kaggle_contacted": False,
        "summarized_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = RUN / "SUMMARY.json"
    if summary_path.exists():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        identity = {key: existing.get(key) for key in ("score_rows", "thresholds", "pooled_full_macro_f1_gain_vs_simple_fixed_050")}
        proposed = {key: result.get(key) for key in identity}
        if identity != proposed:
            raise RuntimeError("Summary already exists with conflicting results")
        events = [e for e in read_ledger(LEDGER) if e.get("event_id") == "SUMMARY:POOLED_REVIEW_READY"]
        digest = sha256_file(summary_path)
        if events and events[-1].get("summary_sha256") != digest:
            raise RuntimeError("Existing summary receipt conflicts with its durable ledger event")
        if not events:
            append_ledger(LEDGER, {"event_id": "SUMMARY:POOLED_REVIEW_READY", "event": "SUMMARY_RECOVERED",
                                   "summary_sha256": digest, "at_utc": datetime.now(timezone.utc).isoformat()})
        return existing
    write_json_exclusive(summary_path, result)
    append_ledger(LEDGER, {"event_id": "SUMMARY:POOLED_REVIEW_READY", "event": "SUMMARY_CREATED",
                           "summary_sha256": sha256_file(summary_path),
                           "at_utc": datetime.now(timezone.utc).isoformat()})
    return result


def score_stage(stage: str, train_csv: Path, cache: Path, release_path: Path) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError("Only predeclared diagnostic stages D1 and D2 are allowed")
    _validate_release(release_path)
    _verify_runtime_lineage(train_csv, cache)
    fit_receipt_path = FIT_DIR / f"{stage}.json"
    if not fit_receipt_path.exists():
        raise RuntimeError(f"{stage} threshold artifact is not frozen")
    fit_receipt = json.loads(fit_receipt_path.read_text(encoding="utf-8"))
    if fit_receipt.get("status") != "FROZEN_PREDICTION_THRESHOLD":
        raise RuntimeError(f"{stage} has no supported fitted threshold to score")
    score_result_path = SCORE_DIR / f"{stage}_RESULTS.json"
    score_events = [e for e in read_ledger(LEDGER) if e.get("stage") == stage and e.get("event") == "SCORED"]
    if score_result_path.exists():
        if not score_events or sha256_file(score_result_path) != score_events[-1].get("result_sha256"):
            raise RuntimeError("Existing score receipt is not reconciled with the durable ledger")
        return json.loads(score_result_path.read_text(encoding="utf-8"))
    vectors = _stage_vectors(stage, train_csv, cache)
    if fit_receipt.get("fit_ids_sha256") != ordered_id_hash(vectors["fit_ids"]):
        raise RuntimeError("Fit receipt and current stage fit IDs differ")
    pred_path = SCORE_DIR / f"{stage}_FROZEN_PREDICTIONS.json"
    if not pred_path.exists():
        threshold = float.fromhex(fit_receipt["threshold_hex"])
        pred_a = np.where(vectors["score_missing"], vectors["score_pA"] >= MISSING_THRESHOLD,
                          vectors["score_pA"] >= threshold).astype(np.int8)
        pred_base = np.where(vectors["score_missing"], vectors["score_pA"] >= MISSING_THRESHOLD,
                             vectors["score_pA"] >= CENTRE).astype(np.int8)
        pred_complex = np.where(vectors["score_missing"], vectors["score_pA"] >= MISSING_THRESHOLD,
                                (0.5 * vectors["score_pP"] + 0.5 * vectors["score_pA"]) >= CENTRE).astype(np.int8)
        frozen = {
            "schema_version": 1, "status": "FROZEN_BEFORE_SCORE_LABEL_READ", "stage": stage,
            "ids": vectors["score_ids"].tolist(), "cycles": vectors["score_cycles"].astype(int).tolist(),
            "missing": vectors["score_missing"].astype(int).tolist(),
            "pA": vectors["score_pA"].tolist(), "pP": vectors["score_pP"].tolist(),
            "candidate": pred_a.tolist(), "simple_fixed_050": pred_base.tolist(),
            "complex_fixed_050": pred_complex.tolist(),
            "threshold": fit_receipt["threshold"], "threshold_hex": fit_receipt["threshold_hex"],
            "score_ids_sha256": ordered_id_hash(vectors["score_ids"]),
            "cache_sha256": sha256_file(cache), "fit_receipt_sha256": sha256_file(fit_receipt_path),
        }
        write_json_exclusive(pred_path, frozen)
    frozen = json.loads(pred_path.read_text(encoding="utf-8"))
    if (
        frozen.get("status") != "FROZEN_BEFORE_SCORE_LABEL_READ"
        or frozen.get("stage") != stage
        or frozen.get("score_ids_sha256") != ordered_id_hash(vectors["score_ids"])
        or frozen.get("cache_sha256") != sha256_file(cache)
        or frozen.get("fit_receipt_sha256") != sha256_file(fit_receipt_path)
    ):
        raise RuntimeError("Frozen score predictions do not match current score rows")
    expected_threshold = float.fromhex(fit_receipt["threshold_hex"])
    expected_a = np.where(vectors["score_missing"], vectors["score_pA"] >= MISSING_THRESHOLD,
                          vectors["score_pA"] >= expected_threshold).astype(np.int8)
    expected_base = np.where(vectors["score_missing"], vectors["score_pA"] >= MISSING_THRESHOLD,
                             vectors["score_pA"] >= CENTRE).astype(np.int8)
    expected_complex = np.where(vectors["score_missing"], vectors["score_pA"] >= MISSING_THRESHOLD,
                                (0.5 * vectors["score_pP"] + 0.5 * vectors["score_pA"]) >= CENTRE).astype(np.int8)
    if (
        not np.array_equal(np.asarray(frozen.get("ids"), dtype=str), vectors["score_ids"])
        or not np.array_equal(np.asarray(frozen.get("cycles"), dtype=np.int64), vectors["score_cycles"])
        or not np.array_equal(np.asarray(frozen.get("missing"), dtype=bool), vectors["score_missing"])
        or not np.array_equal(np.asarray(frozen.get("candidate"), dtype=np.int8), expected_a)
        or not np.array_equal(np.asarray(frozen.get("simple_fixed_050"), dtype=np.int8), expected_base)
        or not np.array_equal(np.asarray(frozen.get("complex_fixed_050"), dtype=np.int8), expected_complex)
    ):
        raise RuntimeError("Frozen predictions fail recomputation from the current non-target OOF vectors")
    started = [e for e in read_ledger(LEDGER) if e.get("stage") == stage and e.get("event") == "SCORE_STARTED"]
    if started:
        if started[-1].get("frozen_predictions_sha256") != sha256_file(pred_path):
            raise RuntimeError("Existing SCORE_STARTED event is bound to different frozen predictions")
    else:
        append_ledger(LEDGER, {"event_id": f"{stage}:SCORE_STARTED", "event": "SCORE_STARTED", "stage": stage,
                               "frozen_predictions_sha256": sha256_file(pred_path),
                               "at_utc": datetime.now(timezone.utc).isoformat()})
    # Target read happens only after predictions are durably frozen and verified.
    y = _read_selected_labels(train_csv, vectors["train_ids"], vectors["score_ids"])
    candidate = np.asarray(frozen["candidate"], dtype=np.int8)
    base = np.asarray(frozen["simple_fixed_050"], dtype=np.int8)
    complex_pred = np.asarray(frozen["complex_fixed_050"], dtype=np.int8)
    if not (len(y) == len(candidate) == len(base) == len(complex_pred)):
        raise ValueError("Frozen prediction/label lengths differ")
    score_arrays = _load_oof_metadata(cache)[STAGES[stage]["score_origin"]]
    masks = np.asarray(frozen["missing"], dtype=bool)
    rows = {record_id: idx for idx, record_id in enumerate(score_arrays["ids"].tolist())}
    positions = np.asarray([rows[r] for r in vectors["score_ids"].tolist()], dtype=np.int64)
    groups = {key: score_arrays[key][positions] for key in ("availability", "entry_friction", "text_load")}
    cycles = np.asarray(frozen["cycles"], dtype=np.int64)
    result = {
        "schema_version": 1, "status": "SCORED", "stage": stage,
        "score_rows": len(y), "score_ids_sha256": ordered_id_hash(vectors["score_ids"]),
        "fit": {
            "threshold": fit_receipt["threshold"],
            "threshold_hex": fit_receipt["threshold_hex"],
            "fit_macro_f1": fit_receipt["fit_macro_f1"],
            "fit_macro_f1_exact": fit_receipt["fit_macro_f1_exact"],
            "threshold_at_interval_bound": float(fit_receipt["threshold"]) in {LOW, HIGH},
        },
        "metrics": {
            "full": {"candidate": _metric(y, candidate), "simple_fixed_050": _metric(y, base),
                     "complex_fixed_050": _metric(y, complex_pred)},
            "present": {"candidate": _metric(y[~masks], candidate[~masks]),
                        "simple_fixed_050": _metric(y[~masks], base[~masks]),
                        "complex_fixed_050": _metric(y[~masks], complex_pred[~masks])},
        },
        "deltas_vs_simple_fixed_050": {
            "macro_f1": _metric(y, candidate)["macro_f1"] - _metric(y, base)["macro_f1"],
        },
        "changed_decisions_vs_simple_fixed_050": int(np.sum(candidate != base)),
        "missing_decisions_invariant": bool(np.array_equal(candidate[masks], base[masks])),
        "by_cycle": {str(c): {"candidate": _metric(y[cycles == c], candidate[cycles == c]),
                              "simple_fixed_050": _metric(y[cycles == c], base[cycles == c])}
                     for c in sorted(np.unique(cycles).tolist())},
        "subgroups": {"candidate": _group_metrics(y, candidate, groups),
                      "simple_fixed_050": _group_metrics(y, base, groups)},
        "test_or_sample_submission_accessed": False,
        "kaggle_contacted": False,
        "scored_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    subgroup_deltas: dict[str, Any] = {}
    for dimension, by_label in result["subgroups"]["candidate"].items():
        subgroup_deltas[dimension] = {}
        baseline_dimension = result["subgroups"]["simple_fixed_050"][dimension]
        for label, detail in by_label.items():
            if detail["status"] != "SUPPORTED":
                subgroup_deltas[dimension][label] = {"status": "UNKNOWN", "reason": "support gate not met"}
                continue
            cand = detail["metrics"]
            reference = baseline_dimension[label]["metrics"]
            subgroup_deltas[dimension][label] = {
                "status": "SUPPORTED",
                "macro_f1_delta": cand["macro_f1"] - reference["macro_f1"],
                "positive_recall_delta": cand["positive_recall"] - reference["positive_recall"],
            }
    result["subgroup_deltas_vs_simple_fixed_050"] = subgroup_deltas
    if score_result_path.exists():
        existing = json.loads(score_result_path.read_text(encoding="utf-8"))
        if canonical_hash(existing) != canonical_hash(result):
            raise RuntimeError("Score result already exists with different contents")
        return existing
    digest = write_json_exclusive(score_result_path, result)
    append_ledger(LEDGER, {"event_id": f"{stage}:SCORED", "event": "SCORED", "stage": stage,
                           "result_sha256": digest, "frozen_predictions_sha256": sha256_file(pred_path),
                           "at_utc": datetime.now(timezone.utc).isoformat()})
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, default=Path("data/raw/classification/Train.csv"))
    parser.add_argument("--cache", type=Path, default=CACHE)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze")
    sub.add_parser("preflight")
    sub.add_parser("self-check")
    fit = sub.add_parser("fit")
    fit.add_argument("--stage", choices=STAGES, required=True)
    fit.add_argument("--release", type=Path, required=True)
    score = sub.add_parser("score")
    score.add_argument("--stage", choices=STAGES, required=True)
    score.add_argument("--release", type=Path, required=True)
    sub.add_parser("summarize")
    args = parser.parse_args(argv)
    if args.command == "freeze":
        value = freeze_sources(args.train_csv, args.cache)
    elif args.command == "preflight":
        value = source_preflight(args.train_csv, args.cache)
        import verify_cls_simple_present_threshold_v1 as verifier
        value["independent_verifier"] = verifier.verify_source_preflight()
        value["status"] = "PASS_SOURCE_READY"
        if READY.exists():
            raise FileExistsError(f"Refusing to overwrite source-ready receipt: {READY}")
        ready = {"status": "PASS", "campaign": value["campaign"], "source_commit": value["source_commit"],
                 "freeze_sha256": value["source_freeze_sha256"], "preflight_sha256": sha256_file(PREFLIGHT),
                 "independent_verifier": value["independent_verifier"], "ready_at_utc": datetime.now(timezone.utc).isoformat()}
        write_json_exclusive(READY, ready)
        value = ready
    elif args.command == "self-check":
        import verify_cls_simple_present_threshold_v1 as verifier
        value = verifier.run_synthetic_self_check()
    elif args.command == "fit":
        value = fit_stage(args.stage, args.train_csv, args.cache, args.release)
    elif args.command == "score":
        value = score_stage(args.stage, args.train_csv, args.cache, args.release)
    else:
        value = summarize_scores()
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
