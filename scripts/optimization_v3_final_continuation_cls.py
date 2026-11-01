"""Frozen C16-only continuation lane: preflight, fit cache, and verification.

The approved C16 pilot fits at 55/67 are reused as immutable caches.  This
runner can fit only the four newly authorized origins (59/63/71/75).  The
preflight/freeze path never calls an estimator; ``--execute`` is reserved for
the later MAIN-owned training job.
"""
from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits

from scripts.optimization_v3_batch1_c16 import (
    BASELINE_MANIFEST_PATH,
    CARD_PATH,
    CARD_SHA256,
    EXPECTED_FIT_ROWS as PILOT_FIT_ROWS,
    FULLSIX_REPORT_PATH,
    FULLSIX_VERIFICATION_PATH,
    FULLSIX_WRAPPER_PATH,
    QUEUE_PATH,
    QUEUE_SHA256,
    ROUND1_AUDITS,
    ROUND1_CACHE,
    TRAIN_SHA256,
    _validate_baseline_artifacts,
    _control_input_files,
)
from scripts.optimization_v3_sparse_classification import (
    CACHE as SPARSE_PILOT_CACHE,
    CONFIG as SPARSE_PILOT_CONFIG,
    OUT as SPARSE_PILOT_REPORT,
    check as check_sparse_pilot,
)
from scripts.optimization_v3_sparse_classification_fullsix import (
    CB_CACHE,
    CACHE as BASELINE_FULLSIX_CACHE,
    check_freeze as check_baseline_fullsix_freeze,
    control_record,
    file_sha as _baseline_file_sha,
    load_existing_prediction,
    pilot_record,
)
from src.models.optimization_v3_batch1 import (
    BatchIntegrityError,
    append_event,
    canonical_json,
    ensure_frozen,
    exclusive_training_slot,
    file_sha256,
    json_sha256,
    read_bundle,
    write_bundle_exclusive,
    write_json_exclusive,
)
from src.models.optimization_v3_batch1_c16 import (
    BLEND_WEIGHT,
    CHANGES,
    FEATURE_NAMES,
    PARAMS,
    THRESHOLD,
    _validated_controls,
)
from src.models.optimization_v3_final_continuation_c16 import (
    NEW_CUTOFFS,
    fit_predict,
)
from src.models.optimization_v3_sparse_classification import PARAMS as BASE_PARAMS
from src.models.classification_models import _TFIDF_PARAMS
from src.models.regression_research_text import _text_values
from src.utils.paths import task_files


CARD_ID = "C16"
PILOT_CUTOFFS = (55, 67)
ALL_ORIGINS = (55, 59, 63, 67, 71, 75)
PRIMARY_SEGMENTS: dict[str, tuple[int, int, int]] = {
    "W1": (55, 56, 59),
    "W2": (59, 60, 63),
    "W3": (63, 64, 67),
    "W4": (67, 68, 71),
    "W5": (71, 72, 75),
    "W6": (75, 76, 79),
}
EVALUATION_VIEWS: dict[str, list[tuple[int, int, int]]] = {
    **{name: [segment] for name, segment in PRIMARY_SEGMENTS.items()},
    "primary": list(PRIMARY_SEGMENTS.values()),
    "H12_E": [(55, 56, 67)],
    "H12_L": [(67, 68, 79)],
    "H12": [(55, 56, 67), (67, 68, 79)],
}
GATE_VIEW_NAMES = ("primary", "H12", "W1", "W2", "W3", "W4", "W5", "W6")
EXPECTED_PRIMARY_ROWS = 38_588
EXPECTED_H12_ROWS = 38_588
EXPECTED_LATE_ROWS = 15_953
EXPECTED_PILOT_ROWS = {55: 17_685, 67: 20_903}
EXPECTED_NEW_WINDOW_ROWS = {59: 6_493, 63: 7_078, 71: 7_462, 75: 8_491}
EXPECTED_FIT_ROWS = {55: 12_020, 59: 16_134, 63: 22_627, 67: 29_705,
                     71: 34_655, 75: 42_117}
JOB_MEMORY_CAP_BYTES = 6 * 1024**3
FREE_RAM_RESERVE_BYTES = 2 * 1024**3
THREADS = 2
MIN_GROUP_N = 200
MIN_GROUP_CLASS_N = 30
MAX_GROUP_F1_HARM = 0.01
MAX_GROUP_POS_RECALL_HARM = 0.02
PRIMARY_GAIN_MIN = 0.0005
H12_HARM_MAX = 0.001
NONNEGATIVE_WINDOWS_MIN = 4
LATE_GAIN_MIN = 0.0

PILOT_EXPERIMENT_DIR = ROOT / "experiments/optimization_v3/batch1/C16"
PILOT_RUN_DIR = ROOT / "experiments/runs/optimization_v3_batch1/C16"
PILOT_REPORT_DIR = ROOT / "reports/optimization_v3/batch1/C16"
EXPERIMENT_DIR = ROOT / "experiments/optimization_v3/final_continuation/C16"
RUN_DIR = ROOT / "experiments/runs/optimization_v3_final_continuation/C16"
REPORT_DIR = ROOT / "reports/optimization_v3/final_continuation/C16"
DECISION_DIR = ROOT / "reports/optimization_v3/final_continuation_plan"
RECIPE_PATH = EXPERIMENT_DIR / "recipe.json"
FREEZE_PATH = EXPERIMENT_DIR / "freeze.json"
PREFLIGHT_PATH = REPORT_DIR / "preflight.json"
RESULT_PATH = REPORT_DIR / "results.json"
VERIFICATION_PATH = REPORT_DIR / "verification.json"
EVENT_PATH = RUN_DIR / "events.jsonl"


class FinalContinuationIntegrityError(BatchIntegrityError):
    """A frozen C16 continuation input, cache, or evaluation invariant failed."""


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha(path: str | Path) -> str:
    return file_sha256(path)


def _resolve(root_relative: str) -> Path:
    return ROOT / Path(root_relative)


def _macro_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(y_pred, dtype=np.int64)
    if y.ndim != 1 or p.shape != y.shape or not len(y):
        raise FinalContinuationIntegrityError("Macro F1 requires nonempty aligned vectors")
    values = []
    for cls in (0, 1):
        tp = int(np.count_nonzero((y == cls) & (p == cls)))
        fn = int(np.count_nonzero((y == cls) & (p != cls)))
        fp = int(np.count_nonzero((y != cls) & (p == cls)))
        denom = (2 * tp) + fp + fn
        values.append(0.0 if denom == 0 else (2.0 * tp / denom))
    return float((values[0] + values[1]) / 2.0)


def _metric(y_true: Sequence[int], probability: Sequence[float]) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(probability, dtype=np.float64)
    if (y.ndim != 1 or p.shape != y.shape or not len(y)
            or not np.isin(y, [0, 1]).all() or not np.isfinite(p).all()
            or np.any((p < 0.0) | (p > 1.0))):
        raise FinalContinuationIntegrityError("Invalid classification metric input")
    pred = p >= THRESHOLD
    return {"n": int(len(y)), "macro_f1": _macro_f1(y, pred.astype(np.int8)),
            "positive_recall": (float(np.sum(pred & (y == 1)) / np.sum(y == 1))
                                if np.any(y == 1) else None),
            "threshold": THRESHOLD}


def _independent_macro_f1(y_true: Sequence[int], predicted: Sequence[int]) -> float:
    """Verifier-side macro-F1 using a separately formed 2x2 confusion matrix."""
    actual = np.asarray(y_true, dtype=np.int8)
    guess = np.asarray(predicted, dtype=np.int8)
    if actual.ndim != 1 or guess.shape != actual.shape or not len(actual):
        raise FinalContinuationIntegrityError("Independent Macro F1 vectors are invalid")
    matrix = np.zeros((2, 2), dtype=np.int64)
    for truth in (0, 1):
        for prediction in (0, 1):
            matrix[truth, prediction] = int(np.count_nonzero((actual == truth) & (guess == prediction)))
    per_class = []
    for cls in (0, 1):
        denominator = 2 * int(matrix[cls, cls]) + int(matrix[1 - cls, cls]) + int(matrix[cls, 1 - cls])
        per_class.append(0.0 if denominator == 0 else 2.0 * float(matrix[cls, cls]) / denominator)
    return float((per_class[0] + per_class[1]) / 2.0)


def _expected_rows(cycles: np.ndarray, low: int, high: int) -> np.ndarray:
    return np.flatnonzero((cycles >= int(low)) & (cycles <= int(high)))


def _entry_friction_label(value: Any) -> str:
    if value is None:
        return "missing"
    try:
        if bool(pd.isna(value)):
            return "missing"
    except (TypeError, ValueError):
        pass
    rendered = str(value).strip()
    if not rendered:
        return "missing"
    try:
        number = float(value)
        if np.isfinite(number) and number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass
    return rendered


def _stream_missing(frame: pd.DataFrame, rows: np.ndarray) -> np.ndarray:
    values = frame.iloc[rows]["signal_stream"].tolist()
    missing = []
    for value in values:
        if value is None:
            missing.append(True)
            continue
        try:
            if bool(pd.isna(value)):
                missing.append(True)
                continue
        except (TypeError, ValueError):
            pass
        missing.append(not str(value).strip())
    return np.asarray(missing, dtype=bool)


def _text_load_labels_for_origin(frame: pd.DataFrame, cutoff: int,
                                 rows: np.ndarray) -> tuple[list[str], list[float]]:
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    prefix = pd.to_numeric(frame.loc[cycles <= int(cutoff), "text_load"],
                           errors="coerce").to_numpy(dtype=np.float64)
    score = pd.to_numeric(frame.iloc[rows]["text_load"], errors="coerce").to_numpy(dtype=np.float64)
    finite = prefix[np.isfinite(prefix)]
    if not len(finite):
        return ["missing"] * len(rows), []
    edges = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, 5)))
    labels = np.full(len(score), -1, dtype=np.int16)
    valid = np.isfinite(score)
    if len(edges) == 1:
        labels[valid] = 0
    else:
        labels[valid] = np.searchsorted(edges[1:-1], score[valid], side="left").astype(np.int16)
    names = ["missing" if label < 0 else f"Q{int(label) + 1}" for label in labels]
    return names, [float(value) for value in edges]


def _group_metrics(frame: pd.DataFrame, y: np.ndarray, rows: np.ndarray,
                   origins: np.ndarray, candidate_probability: np.ndarray,
                   baseline_probability: np.ndarray) -> dict[str, Any]:
    rows = np.asarray(rows, dtype=np.int64)
    origins = np.asarray(origins, dtype=np.int64)
    candidate = np.asarray(candidate_probability, dtype=np.float64) >= THRESHOLD
    baseline = np.asarray(baseline_probability, dtype=np.float64) >= THRESHOLD
    actual = np.asarray(y, dtype=np.int64)
    if (origins.shape != rows.shape or candidate.shape != rows.shape
            or baseline.shape != rows.shape or actual.shape != rows.shape):
        raise FinalContinuationIntegrityError("Group guard vectors are not row-aligned")

    groups: dict[str, list[str]] = {}
    if "entry_friction" in frame:
        groups["entry_friction"] = [
            _entry_friction_label(value)
            for value in frame.iloc[rows]["entry_friction"].tolist()
        ]
    if "signal_stream" in frame:
        missing = _stream_missing(frame, rows)
        groups["stream_availability"] = ["missing" if value else "present" for value in missing]
    if "text_load" in frame:
        labels = [""] * len(rows)
        per_origin_edges: dict[str, list[float]] = {}
        for origin in sorted(np.unique(origins)):
            mask = origins == origin
            current, edges = _text_load_labels_for_origin(frame, int(origin), rows[mask])
            for index, label in zip(np.flatnonzero(mask), current):
                labels[int(index)] = label
            per_origin_edges[str(int(origin))] = edges
        groups["text_load_quantile"] = labels
    else:
        per_origin_edges = {}

    fields: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    for field, labels in groups.items():
        observed = set(labels)
        if field == "entry_friction":
            observed.update({"0", "1", "missing"})
        elif field == "stream_availability":
            observed.update({"missing", "present"})
        elif field == "text_load_quantile":
            observed.update({"Q1", "Q2", "Q3", "Q4", "missing"})
        detail: dict[str, Any] = {}
        for label in sorted(observed):
            mask = np.asarray([item == label for item in labels], dtype=bool)
            n = int(mask.sum())
            n0 = int(np.count_nonzero(actual[mask] == 0))
            n1 = int(np.count_nonzero(actual[mask] == 1))
            supported = n >= MIN_GROUP_N and n0 >= MIN_GROUP_CLASS_N and n1 >= MIN_GROUP_CLASS_N
            if n:
                cand_f1 = _macro_f1(actual[mask], candidate[mask].astype(np.int8))
                base_f1 = _macro_f1(actual[mask], baseline[mask].astype(np.int8))
                cand_recall = (float(np.count_nonzero(candidate[mask] & (actual[mask] == 1)) / n1)
                               if n1 else None)
                base_recall = (float(np.count_nonzero(baseline[mask] & (actual[mask] == 1)) / n1)
                               if n1 else None)
                f1_delta = cand_f1 - base_f1
                recall_delta = cand_recall - base_recall if n1 else None
            else:
                cand_f1 = base_f1 = cand_recall = base_recall = f1_delta = recall_delta = None
            guard_applies = bool(supported)
            guard_pass = (None if not guard_applies else bool(
                f1_delta >= -MAX_GROUP_F1_HARM
                and recall_delta >= -MAX_GROUP_POS_RECALL_HARM
            ))
            row = {
                "n": n, "class_0_n": n0, "class_1_n": n1,
                "supported": supported,
                "candidate_macro_f1": cand_f1, "baseline_macro_f1": base_f1,
                "candidate_minus_baseline_macro_f1": f1_delta,
                "candidate_positive_recall": cand_recall,
                "baseline_positive_recall": base_recall,
                "candidate_minus_baseline_positive_recall": recall_delta,
                "guard_applies": guard_applies, "guard_pass": guard_pass,
            }
            detail[label] = row
            if guard_applies and not guard_pass:
                failures.append({"field": field, "group": label, **row})
        fields[field] = detail
    return {
        "support_rule": "guard if n>=200 and both classes have at least 30 rows; otherwise descriptive",
        "text_load_fit_scope": "each origin's cycle_code <= cutoff prefix only",
        "text_load_quantile_edges_by_origin": per_origin_edges,
        "groups": fields,
        "supported_guard_failures": failures,
    }


def _independent_group_metrics(frame: pd.DataFrame, y: np.ndarray, rows: np.ndarray,
                               origins: np.ndarray, candidate_probability: np.ndarray,
                               baseline_probability: np.ndarray) -> dict[str, Any]:
    """Verifier-only subgroup recomputation with separate metric arithmetic."""
    rows = np.asarray(rows, dtype=np.int64)
    origins = np.asarray(origins, dtype=np.int64)
    actual = np.asarray(y, dtype=np.int8)
    cand = np.asarray(candidate_probability, dtype=np.float64) >= THRESHOLD
    base = np.asarray(baseline_probability, dtype=np.float64) >= THRESHOLD
    if any(value.shape != rows.shape for value in (origins, actual, cand, base)):
        raise FinalContinuationIntegrityError("Independent subgroup arrays are misaligned")
    labels_by_field: dict[str, list[str]] = {}
    if "entry_friction" in frame:
        labels_by_field["entry_friction"] = [
            _entry_friction_label(value) for value in frame.iloc[rows]["entry_friction"].tolist()
        ]
    if "signal_stream" in frame:
        missing = _stream_missing(frame, rows)
        labels_by_field["stream_availability"] = [
            "missing" if item else "present" for item in missing
        ]
    edges_by_origin: dict[str, list[float]] = {}
    if "text_load" in frame:
        text_labels = [""] * len(rows)
        for origin in sorted(np.unique(origins)):
            origin_mask = origins == origin
            names, edges = _text_load_labels_for_origin(frame, int(origin), rows[origin_mask])
            for position, name in zip(np.flatnonzero(origin_mask), names):
                text_labels[int(position)] = name
            edges_by_origin[str(int(origin))] = edges
        labels_by_field["text_load_quantile"] = text_labels

    group_tables: dict[str, Any] = {}
    failures = []
    for field, labels in labels_by_field.items():
        expected_names = set(labels)
        if field == "entry_friction":
            expected_names.update(("0", "1", "missing"))
        elif field == "stream_availability":
            expected_names.update(("missing", "present"))
        else:
            expected_names.update(("Q1", "Q2", "Q3", "Q4", "missing"))
        table = {}
        for name in sorted(expected_names):
            mask = np.asarray([item == name for item in labels], dtype=bool)
            sub_y = actual[mask]
            n = int(len(sub_y))
            n0, n1 = int(np.count_nonzero(sub_y == 0)), int(np.count_nonzero(sub_y == 1))
            supported = n >= MIN_GROUP_N and n0 >= MIN_GROUP_CLASS_N and n1 >= MIN_GROUP_CLASS_N
            if n:
                cand_f1 = _independent_macro_f1(sub_y, cand[mask].astype(np.int8))
                base_f1 = _independent_macro_f1(sub_y, base[mask].astype(np.int8))
                cand_recall = (float(np.count_nonzero(cand[mask] & (sub_y == 1)) / n1) if n1 else None)
                base_recall = (float(np.count_nonzero(base[mask] & (sub_y == 1)) / n1) if n1 else None)
                f1_delta = cand_f1 - base_f1
                recall_delta = cand_recall - base_recall if n1 else None
            else:
                cand_f1 = base_f1 = cand_recall = base_recall = f1_delta = recall_delta = None
            applies = bool(supported)
            passed = None if not applies else bool(
                f1_delta >= -MAX_GROUP_F1_HARM and recall_delta >= -MAX_GROUP_POS_RECALL_HARM
            )
            item = {
                "n": n, "class_0_n": n0, "class_1_n": n1, "supported": supported,
                "candidate_macro_f1": cand_f1, "baseline_macro_f1": base_f1,
                "candidate_minus_baseline_macro_f1": f1_delta,
                "candidate_positive_recall": cand_recall,
                "baseline_positive_recall": base_recall,
                "candidate_minus_baseline_positive_recall": recall_delta,
                "guard_applies": applies, "guard_pass": passed,
            }
            table[name] = item
            if applies and not passed:
                failures.append({"field": field, "group": name, **item})
        group_tables[field] = table
    return {
        "support_rule": "guard if n>=200 and both classes have at least 30 rows; otherwise descriptive",
        "text_load_fit_scope": "each origin's cycle_code <= cutoff prefix only",
        "text_load_quantile_edges_by_origin": edges_by_origin,
        "groups": group_tables,
        "supported_guard_failures": failures,
    }


def _validate_route_record(record: Mapping[str, Any], cutoff: int) -> None:
    controls = record.get("control_arrays")
    if controls is None:
        raise FinalContinuationIntegrityError(f"Origin {cutoff} lacks independent route controls")
    raw = np.asarray(record.get("raw_candidate"), dtype=np.float64)
    prediction = np.asarray(record["prediction"], dtype=np.float64)
    baseline = np.asarray(record["baseline_prediction"], dtype=np.float64)
    missing = np.asarray(controls["stream_missing"], dtype=bool)
    c2b = np.asarray(controls["c2b_probability"], dtype=np.float64)
    cb = np.asarray(controls["cb_probability"], dtype=np.float64)
    old_sparse = np.asarray(controls["baseline_sparse_raw"], dtype=np.float64)
    arrays = [raw, prediction, baseline, missing, c2b, cb, old_sparse]
    if len({len(value) for value in arrays}) != 1:
        raise FinalContinuationIntegrityError(f"Origin {cutoff} route arrays have unequal lengths")
    rebuilt_candidate = np.where(missing, cb, 0.75 * c2b + BLEND_WEIGHT * raw)
    rebuilt_baseline = np.where(missing, cb, 0.75 * c2b + BLEND_WEIGHT * old_sparse)
    if not np.array_equal(prediction, rebuilt_candidate) or not np.array_equal(baseline, rebuilt_baseline):
        raise FinalContinuationIntegrityError(f"Origin {cutoff} route reconstruction differs")


def _view_rows(frame: pd.DataFrame, records: Mapping[int, Mapping[str, Any]],
               segments: Sequence[tuple[int, int, int]]) -> dict[str, np.ndarray]:
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    row_parts: list[np.ndarray] = []
    cand_parts: list[np.ndarray] = []
    base_parts: list[np.ndarray] = []
    origin_parts: list[np.ndarray] = []
    for cutoff, low, high in segments:
        if int(cutoff) not in records:
            raise FinalContinuationIntegrityError(f"Missing prediction origin {cutoff}")
        record = records[int(cutoff)]
        saved_rows = np.asarray(record.get("score_rows"), dtype=np.int64)
        expected_window = _expected_rows(cycles, low, high)
        mask = (cycles[saved_rows] >= int(low)) & (cycles[saved_rows] <= int(high))
        rows = saved_rows[mask]
        if not np.array_equal(rows, expected_window):
            raise FinalContinuationIntegrityError(
                f"Origin {cutoff} does not cover exact declared cycles {low}..{high}"
            )
        row_parts.append(rows)
        cand_parts.append(np.asarray(record["prediction"], dtype=np.float64)[mask])
        base_parts.append(np.asarray(record["baseline_prediction"], dtype=np.float64)[mask])
        origin_parts.append(np.full(len(rows), int(cutoff), dtype=np.int64))
    rows = np.concatenate(row_parts)
    if len(np.unique(rows)) != len(rows):
        raise FinalContinuationIntegrityError("Evaluation view contains overlapping Train rows")
    return {
        "rows": rows,
        "candidate": np.concatenate(cand_parts),
        "baseline": np.concatenate(base_parts),
        "origins": np.concatenate(origin_parts),
    }


def build_report(frame: pd.DataFrame, target: np.ndarray,
                 records: Mapping[int, Mapping[str, Any]],
                 freeze_signature: str) -> dict[str, Any]:
    """Build six-window and H12 report from aligned per-origin bundles."""
    y = np.asarray(target, dtype=np.int64)
    if y.shape != (len(frame),) or not np.isin(y, [0, 1]).all():
        raise FinalContinuationIntegrityError("Train target is not aligned binary data")
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    output_views: dict[str, Any] = {}
    view_data: dict[str, dict[str, np.ndarray]] = {}
    for name, segments in EVALUATION_VIEWS.items():
        data = _view_rows(frame, records, segments)
        expected = np.unique(np.concatenate([
            _expected_rows(cycles, lo, hi) for _, lo, hi in segments
        ]))
        if not np.array_equal(np.sort(data["rows"]), expected):
            raise FinalContinuationIntegrityError(f"View {name} row denominator changed")
        candidate = _metric(y[data["rows"]], data["candidate"])
        baseline = _metric(y[data["rows"]], data["baseline"])
        output_views[name] = {
            "rows": int(len(data["rows"])),
            "candidate": candidate,
            "baseline": baseline,
            "gain_positive_is_better": float(candidate["macro_f1"] - baseline["macro_f1"]),
        }
        view_data[name] = data

    late_rows = np.concatenate([view_data["W5"]["rows"], view_data["W6"]["rows"]])
    if len(np.unique(late_rows)) != len(late_rows):
        raise FinalContinuationIntegrityError("W5/W6 late selection rows overlap")
    late_candidate = np.concatenate([view_data["W5"]["candidate"], view_data["W6"]["candidate"]])
    late_baseline = np.concatenate([view_data["W5"]["baseline"], view_data["W6"]["baseline"]])
    late_gain = (_metric(y[late_rows], late_candidate)["macro_f1"]
                 - _metric(y[late_rows], late_baseline)["macro_f1"])
    window_nonnegative = sum(
        output_views[f"W{index}"]["gain_positive_is_better"] >= 0.0 for index in range(1, 7)
    )
    group_outputs = {
        name: _group_metrics(
            frame, y[data["rows"]], data["rows"], data["origins"],
            data["candidate"], data["baseline"],
        )
        for name, data in view_data.items() if name in GATE_VIEW_NAMES
    }
    group_failures = [
        {"view": view_name, **failure}
        for view_name, groups in group_outputs.items()
        for failure in groups["supported_guard_failures"]
    ]
    primary_gain = float(output_views["primary"]["gain_positive_is_better"])
    h12_gain = float(output_views["H12"]["gain_positive_is_better"])
    gates = {
        "primary_gain_at_least_0_0005": primary_gain >= PRIMARY_GAIN_MIN,
        "H12_harm_at_most_0_001": h12_gain >= -H12_HARM_MAX,
        "at_least_4_of_6_windows_nonnegative": window_nonnegative >= NONNEGATIVE_WINDOWS_MIN,
        "late_W5_W6_gain_positive": late_gain > LATE_GAIN_MIN,
        "all_supported_group_guards_pass": not group_failures,
    }
    return {
        "schema_version": 1,
        "card_id": CARD_ID,
        "task": "classification",
        "status": "CANDIDATE_METRICS_READY_FOR_INDEPENDENT_QA",
        "freeze_signature": freeze_signature,
        "cutoffs": list(ALL_ORIGINS),
        "new_fit_cutoffs": list(NEW_CUTOFFS),
        "reused_pilot_cutoffs": list(PILOT_CUTOFFS),
        "new_fit_predictor_count": 4,
        "threshold": THRESHOLD,
        "views": output_views,
        "group_guards": group_outputs,
        "qualification": {
            "gates": gates,
            "late_W5_W6": {"rows": int(len(late_rows)), "gain_positive_is_better": late_gain},
            "nonnegative_primary_windows": int(window_nonnegative),
            "supported_group_failures": group_failures,
            "qualified": all(gates.values()),
            "gate_source": "reports/optimization_v3/final_continuation_plan/DECISION.md",
        },
        "outcome_reuse_warning": "All evaluation labels are previously used temporal development data; this is not an untouched holdout.",
        "test_data_used": False,
        "sample_submission_read": False,
        "kaggle_access": False,
    }


def verify_report_independently(frame: pd.DataFrame, target: np.ndarray,
                                records: Mapping[int, Mapping[str, Any]],
                                declared: Mapping[str, Any],
                                freeze_signature: str) -> dict[str, Any]:
    """Recompute metrics and subgroup guards without using build_report metrics."""
    y = np.asarray(target, dtype=np.int64)
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    if declared.get("freeze_signature") != freeze_signature:
        raise FinalContinuationIntegrityError("Report freeze signature mismatch")
    if set(int(key) for key in records) != set(ALL_ORIGINS):
        raise FinalContinuationIntegrityError("Verifier requires all six declared origins")
    for cutoff in ALL_ORIGINS:
        record = records[cutoff]
        expected_low, expected_high = (cutoff + 1, cutoff + 12) if cutoff in PILOT_CUTOFFS else (cutoff + 1, cutoff + 4)
        rows = np.asarray(record.get("score_rows"), dtype=np.int64)
        expected_rows = _expected_rows(cycles, expected_low, expected_high)
        ids = np.asarray(record.get("score_record_ids"), dtype=str)
        expected_ids = frame.iloc[expected_rows]["record_id"].astype(str).to_numpy(dtype=str)
        if (not np.array_equal(rows, expected_rows) or not np.array_equal(ids, expected_ids)
                or len(np.unique(ids)) != len(ids)):
            raise FinalContinuationIntegrityError(f"Verifier origin coverage/record_id mismatch at cutoff {cutoff}")
        _validate_route_record(record, cutoff)
    verified: dict[str, Any] = {}
    group_outputs: dict[str, Any] = {}
    for name, segments in EVALUATION_VIEWS.items():
        rows_parts: list[np.ndarray] = []
        cand_parts: list[np.ndarray] = []
        base_parts: list[np.ndarray] = []
        origin_parts: list[np.ndarray] = []
        for cutoff, low, high in segments:
            bundle = records[int(cutoff)]
            rows = np.asarray(bundle["score_rows"], dtype=np.int64)
            ids = np.asarray(bundle["score_record_ids"], dtype=str)
            expected_ids = frame.iloc[rows]["record_id"].astype(str).to_numpy(dtype=str)
            if (rows.ndim != 1 or len(np.unique(rows)) != len(rows)
                    or not np.array_equal(ids, expected_ids)
                    or len(np.unique(ids)) != len(ids)):
                raise FinalContinuationIntegrityError(f"Origin {cutoff} row/record_id alignment failed")
            mask = (cycles[rows] >= int(low)) & (cycles[rows] <= int(high))
            selected = rows[mask]
            expected = _expected_rows(cycles, low, high)
            if not np.array_equal(selected, expected):
                raise FinalContinuationIntegrityError(f"Origin {cutoff} omitted/added rows in {name}")
            rows_parts.append(selected)
            cand_parts.append(np.asarray(bundle["prediction"], dtype=np.float64)[mask])
            base_parts.append(np.asarray(bundle["baseline_prediction"], dtype=np.float64)[mask])
            origin_parts.append(np.full(len(selected), int(cutoff), dtype=np.int64))
        rows = np.concatenate(rows_parts)
        if len(np.unique(rows)) != len(rows):
            raise FinalContinuationIntegrityError(f"Verifier found duplicate rows in {name}")
        cand_prob = np.concatenate(cand_parts)
        base_prob = np.concatenate(base_parts)
        cand_class = cand_prob >= THRESHOLD
        base_class = base_prob >= THRESHOLD
        candidate_f1 = _independent_macro_f1(y[rows], cand_class.astype(np.int8))
        baseline_f1 = _independent_macro_f1(y[rows], base_class.astype(np.int8))
        gain = candidate_f1 - baseline_f1
        row = declared["views"][name]
        if (int(row["rows"]) != len(rows)
                or not np.isclose(float(row["candidate"]["macro_f1"]), candidate_f1, atol=1e-12, rtol=0)
                or not np.isclose(float(row["baseline"]["macro_f1"]), baseline_f1, atol=1e-12, rtol=0)
                or not np.isclose(float(row["gain_positive_is_better"]), gain, atol=1e-12, rtol=0)):
            raise FinalContinuationIntegrityError(f"Independent metric mismatch for {name}")
        verified[name] = {"n": int(len(rows)), "candidate_macro_f1": candidate_f1,
                          "baseline_macro_f1": baseline_f1,
                          "gain_positive_is_better": float(gain)}
        if name in GATE_VIEW_NAMES:
            independently_computed = _independent_group_metrics(
                frame, y[rows], rows, np.concatenate(origin_parts), cand_prob, base_prob,
            )
            declared_groups = declared.get("group_guards", {}).get(name)
            if not isinstance(declared_groups, Mapping):
                raise FinalContinuationIntegrityError(f"Declared subgroup guard table absent for {name}")
            if set(independently_computed["groups"]) != set(declared_groups.get("groups", {})):
                raise FinalContinuationIntegrityError(f"Subgroup fields differ in {name}")
            for field, table in independently_computed["groups"].items():
                if set(table) != set(declared_groups["groups"].get(field, {})):
                    raise FinalContinuationIntegrityError(f"Subgroup labels differ in {name}/{field}")
                for group, values in table.items():
                    saved = declared_groups["groups"][field][group]
                    for key in ("n", "class_0_n", "class_1_n", "supported", "guard_applies", "guard_pass"):
                        if values[key] != saved.get(key):
                            raise FinalContinuationIntegrityError(f"Subgroup support/gate differs in {name}/{field}/{group}")
                    for key in ("candidate_macro_f1", "baseline_macro_f1", "candidate_minus_baseline_macro_f1",
                                "candidate_positive_recall", "baseline_positive_recall",
                                "candidate_minus_baseline_positive_recall"):
                        left, right = values[key], saved.get(key)
                        if left is None or right is None:
                            if left is not right:
                                raise FinalContinuationIntegrityError(f"Subgroup metric null mismatch in {name}/{field}/{group}/{key}")
                        elif not np.isclose(float(left), float(right), atol=1e-12, rtol=0):
                            raise FinalContinuationIntegrityError(f"Subgroup metric differs in {name}/{field}/{group}/{key}")
            group_outputs[name] = independently_computed
    predicted_failure_set = {
        (item["view"], item["field"], item["group"])
        for item in declared["qualification"].get("supported_group_failures", [])
    }
    actual_failure_set = {
        (view, failure["field"], failure["group"])
        for view, groups in group_outputs.items()
        for failure in groups["supported_guard_failures"]
    }
    if predicted_failure_set != actual_failure_set:
        raise FinalContinuationIntegrityError("Independent supported-group failure set differs")
    primary_gain = verified["primary"]["gain_positive_is_better"]
    h12_gain = verified["H12"]["gain_positive_is_better"]
    late_rows = np.concatenate([verified_view_rows(frame, records, PRIMARY_SEGMENTS[x])
                                for x in ("W5", "W6")])
    if len(np.unique(late_rows)) != EXPECTED_LATE_ROWS:
        raise FinalContinuationIntegrityError("W5/W6 late denominator mismatch")
    late_candidate = np.concatenate([
        _view_rows(frame, records, [PRIMARY_SEGMENTS[x]])["candidate"] for x in ("W5", "W6")
    ])
    late_baseline = np.concatenate([
        _view_rows(frame, records, [PRIMARY_SEGMENTS[x]])["baseline"] for x in ("W5", "W6")
    ])
    late_gain = (_independent_macro_f1(y[late_rows], (late_candidate >= THRESHOLD).astype(np.int8))
                 - _independent_macro_f1(y[late_rows], (late_baseline >= THRESHOLD).astype(np.int8)))
    nonnegative = sum(verified[f"W{i}"]["gain_positive_is_better"] >= 0.0 for i in range(1, 7))
    checks = {
        "primary_gain_at_least_0_0005": primary_gain >= PRIMARY_GAIN_MIN,
        "H12_harm_at_most_0_001": h12_gain >= -H12_HARM_MAX,
        "at_least_4_of_6_windows_nonnegative": nonnegative >= NONNEGATIVE_WINDOWS_MIN,
        "late_W5_W6_gain_positive": late_gain > LATE_GAIN_MIN,
        "all_supported_group_guards_pass": not actual_failure_set,
    }
    if checks != declared["qualification"]["gates"]:
        raise FinalContinuationIntegrityError("Independent gate arithmetic differs")
    return {
        "status": "PASS",
        "card_id": CARD_ID,
        "freeze_signature": freeze_signature,
        "views": verified,
        "group_guards": group_outputs,
        "qualification": {
            "gates": checks,
            "late_W5_W6": {"rows": int(len(late_rows)), "gain_positive_is_better": float(late_gain)},
            "nonnegative_primary_windows": int(nonnegative),
            "supported_group_failures": [
                {"view": view, **failure}
                for view, groups in group_outputs.items()
                for failure in groups["supported_guard_failures"]
            ],
            "qualified": all(checks.values()),
        },
        "metric_implementation": "independent integer-count macro-F1 reconstruction in this verifier path",
        "test_data_used": False,
        "kaggle_access": False,
    }


def verified_view_rows(frame: pd.DataFrame, records: Mapping[int, Mapping[str, Any]],
                       segment: tuple[int, int, int]) -> np.ndarray:
    cutoff, low, high = segment
    rows = np.asarray(records[int(cutoff)]["score_rows"], dtype=np.int64)
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    selected = rows[(cycles[rows] >= int(low)) & (cycles[rows] <= int(high))]
    if not np.array_equal(selected, _expected_rows(cycles, low, high)):
        raise FinalContinuationIntegrityError("Independent view row coverage mismatch")
    return selected


def _official_train() -> tuple[Path, pd.DataFrame, np.ndarray, np.ndarray]:
    train_path = task_files("classification").get("train")
    if train_path is None or not Path(train_path).is_file():
        raise FileNotFoundError("Official Classification Train is missing")
    path = Path(train_path).resolve()
    frame = pd.read_csv(path)
    required = set(FEATURE_NAMES) | {"record_id", "artifact_signal"}
    if absent := sorted(required - set(frame.columns)):
        raise FinalContinuationIntegrityError(f"Train is missing required fields: {absent}")
    target = pd.to_numeric(frame["artifact_signal"], errors="raise").to_numpy(dtype=np.int8)
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    if (_sha(path) != TRAIN_SHA256 or len(frame) != 50_608
            or not np.array_equal(np.unique(cycles), np.arange(47, 80))
            or frame["record_id"].isna().any()
            or frame["record_id"].astype(str).str.strip().eq("").any()
            or frame["record_id"].astype(str).duplicated().any()
            or target.shape != (len(frame),) or not np.isin(target, [0, 1]).all()):
        raise FinalContinuationIntegrityError("Official Train hash/schema/IDs/cycles/target failed frozen validation")
    return path, frame, target, cycles


def _current_shared_engine_audit(old_sha256: str) -> dict[str, Any]:
    """Prove the post-pilot shared harness edit is only budget metadata logic."""
    rel = "src/models/optimization_v3_batch1.py"
    commit = "5840f4c2a75b94ea5fd5f2cd04dcb2ba758745e8"
    before = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{commit}^:{rel}"])
    after = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{commit}:{rel}"])
    current = (ROOT / rel).read_bytes()
    if hashlib.sha256(before).hexdigest() != old_sha256:
        raise FinalContinuationIntegrityError("C16 pilot freeze source hash does not match shared-engine pre-change blob")
    if hashlib.sha256(after).hexdigest() != hashlib.sha256(current).hexdigest():
        raise FinalContinuationIntegrityError("Shared-engine source changed after the audited 5840f4c metadata patch")
    import ast
    before_ast = ast.parse(before.decode("utf-8"))
    after_ast = ast.parse(after.decode("utf-8"))
    def definitions(tree: ast.Module) -> dict[str, str]:
        return {
            node.name: ast.dump(node, annotate_fields=True, include_attributes=False)
            for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
    old_defs, new_defs = definitions(before_ast), definitions(after_ast)
    if set(old_defs) != set(new_defs):
        raise FinalContinuationIntegrityError("Shared-engine public definitions changed in audited patch")
    changed_defs = sorted(name for name in old_defs if old_defs[name] != new_defs[name])
    if changed_defs != ["reserve_card_budget"]:
        raise FinalContinuationIntegrityError(f"Unexpected shared-engine code drift: {changed_defs}")
    old_rest = [ast.dump(node, annotate_fields=True, include_attributes=False)
                for node in before_ast.body
                if not (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and node.name == "reserve_card_budget")]
    new_rest = [ast.dump(node, annotate_fields=True, include_attributes=False)
                for node in after_ast.body
                if not (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and node.name == "reserve_card_budget")]
    if old_rest != new_rest:
        raise FinalContinuationIntegrityError("Shared-engine patch changed code outside reservation metadata function")
    return {
        "status": "PASS_METADATA_ONLY",
        "commit": commit,
        "changed_definitions": changed_defs,
        "pilot_source_sha256": old_sha256,
        "current_source_sha256": hashlib.sha256(current).hexdigest(),
        "fit_adapter_or_metric_changes": False,
    }


def _source_files(pilot_freeze: Mapping[str, Any]) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    old_sources = pilot_freeze["identity"].get("source_sha256", {})
    for path_text in old_sources:
        path = Path(path_text)
        if not path.is_absolute():
            path = ROOT / path
        if path.is_file():
            sources[path.resolve().relative_to(ROOT).as_posix()] = path.resolve()
    explicit = [
        Path(__file__).resolve(),
        ROOT / "src/models/optimization_v3_final_continuation_c16.py",
        ROOT / "src/models/optimization_v3_batch1.py",
        ROOT / "scripts/optimization_v3_sparse_classification_fullsix.py",
        ROOT / "scripts/optimization_v3_batch1_verify.py",
        ROOT / "reports/optimization_v3/final_continuation_plan/DECISION.md",
        ROOT / "reports/optimization_v3/final_continuation_plan/guard_audit.json",
    ]
    for path in explicit:
        if path.is_file():
            sources[path.resolve().relative_to(ROOT).as_posix()] = path.resolve()
    return sources


def _audit_queue_drift(pilot_freeze: Mapping[str, Any], current_queue: Mapping[str, Any]) -> dict[str, Any]:
    """Verify queue updates since the C16 pilot touched only R10/R13 status."""
    commit = "c2a2608"
    rel = "experiments/optimization_v3/research_backlog/queue.json"
    historical = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{commit}:{rel}"])
    historical_sha = hashlib.sha256(historical).hexdigest()
    frozen_queue_sha = pilot_freeze["identity"].get("input_sha256", {}).get("canonical_queue")
    current_sha = _sha(QUEUE_PATH)
    head_queue = subprocess.check_output(["git", "-C", str(ROOT), "show", f"HEAD:{rel}"])
    if hashlib.sha256(head_queue).hexdigest() != current_sha:
        raise FinalContinuationIntegrityError("Canonical queue has uncommitted edits; refuse freeze")
    if historical_sha != frozen_queue_sha or current_sha != _sha(QUEUE_PATH):
        raise FinalContinuationIntegrityError("Queue history does not match C16 pilot freeze/current file")
    historical_queue = json.loads(historical.decode("utf-8"))

    def leaves(left: Any, right: Any, prefix: str = "") -> list[str]:
        if isinstance(left, dict) and isinstance(right, dict):
            keys = set(left) | set(right)
            return [path for key in sorted(keys)
                    for path in leaves(left.get(key), right.get(key), f"{prefix}.{key}" if prefix else str(key))]
        if isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                return [f"{prefix}.length"]
            return [path for index, (a, b) in enumerate(zip(left, right))
                    for path in leaves(a, b, f"{prefix}[{index}]")]
        return [] if left == right else [prefix]

    current = dict(current_queue)
    changes = leaves(historical_queue, current)
    expected = {
        "cards[42].status", "cards[42].new_results",
        "cards[43].status", "cards[43].new_results",
    }
    # Queue order is canonical: R10 is index 42 and R13 is index 43.
    if set(changes) != expected:
        raise FinalContinuationIntegrityError(f"Unexpected queue drift since C16 pilot: {changes}")
    current_card = next((row for row in current.get("cards", []) if row.get("id") == CARD_ID), None)
    frozen_card = pilot_freeze.get("identity", {}).get("card")
    if current_card is None or dict(current_card) != frozen_card:
        raise FinalContinuationIntegrityError("Canonical C16 queue card changed since pilot freeze")
    return {
        "status": "PASS_UNRELATED_QUEUE_CARD_UPDATES_ONLY",
        "historical_queue_commit": commit,
        "pilot_queue_sha256": frozen_queue_sha,
        "current_queue_sha256": current_sha,
        "changed_leaf_paths": changes,
        "C16_card_unchanged": True,
    }


def _input_files(train_path: Path) -> dict[str, Path]:
    files = {
        "official_classification_train": train_path,
        "canonical_queue": QUEUE_PATH,
        "canonical_card": CARD_PATH,
        "accepted_sparse25_manifest": BASELINE_MANIFEST_PATH,
        "accepted_sparse25_decision": ROOT / "experiments/optimization_v3/decisions/v3_cls_sparse25.json",
        "accepted_sparse25_fullsix_report": FULLSIX_REPORT_PATH,
        "accepted_sparse25_fullsix_verification": FULLSIX_VERIFICATION_PATH,
        "accepted_sparse25_fullsix_wrapper": FULLSIX_WRAPPER_PATH,
        "baseline_fullsix_freeze": ROOT / "experiments/optimization_v3/classification_sparse_full/freeze.json",
        "sparse_pilot_protocol": SPARSE_PILOT_CONFIG / "protocol.json",
        "sparse_pilot_freeze": SPARSE_PILOT_CONFIG / "freeze.json",
        "sparse_pilot_report": SPARSE_PILOT_REPORT / "pilot.json",
        "sparse_pilot_verification": SPARSE_PILOT_REPORT / "verification.json",
        "pilot_c16_freeze": PILOT_EXPERIMENT_DIR / "freeze.json",
        "pilot_c16_recipe": PILOT_EXPERIMENT_DIR / "recipe.json",
        "pilot_c16_results": PILOT_REPORT_DIR / "results.json",
        "pilot_c16_verification": PILOT_REPORT_DIR / "verification.json",
        "pilot_c16_gate": PILOT_REPORT_DIR / "gate.json",
        "review_decision": DECISION_DIR / "DECISION.md",
        "review_guard_audit": DECISION_DIR / "guard_audit.json",
        "review_qa": DECISION_DIR / "QA.json",
    }
    for cutoff in ALL_ORIGINS:
        files[f"c2b_{cutoff}_npz"] = ROUND1_CACHE / f"fit{cutoff}__C2_B.npz"
        files[f"c2b_{cutoff}_audit"] = ROUND1_AUDITS / f"fit{cutoff}.json"
        files[f"catboost_{cutoff}_npz"] = CB_CACHE / f"fit{cutoff}.npz"
        files[f"catboost_{cutoff}_sidecar"] = CB_CACHE / f"fit{cutoff}.json"
    for cutoff in PILOT_CUTOFFS:
        files[f"sparse_pilot_{cutoff}_npz"] = SPARSE_PILOT_CACHE / f"fit{cutoff}.npz"
        files[f"sparse_pilot_{cutoff}_sidecar"] = SPARSE_PILOT_CACHE / f"fit{cutoff}.json"
        files[f"c16_pilot_{cutoff}_npz"] = PILOT_RUN_DIR / f"C16_fit{cutoff}.npz"
        files[f"c16_pilot_{cutoff}_sidecar"] = PILOT_RUN_DIR / f"C16_fit{cutoff}.json"
    for cutoff in NEW_CUTOFFS:
        files[f"baseline_fullsix_{cutoff}_npz"] = BASELINE_FULLSIX_CACHE / f"fit{cutoff}.npz"
        files[f"baseline_fullsix_{cutoff}_sidecar"] = BASELINE_FULLSIX_CACHE / f"fit{cutoff}.json"
    return files


def _load_c16_pilot_bundle(cutoff: int, frame: pd.DataFrame,
                           baseline_record: Mapping[str, Any],
                           freeze_signature: str) -> dict[str, Any]:
    path = PILOT_RUN_DIR / f"C16_fit{cutoff}.npz"
    sidecar_path = path.with_suffix(".json")
    sidecar = _read_json(sidecar_path)
    identity = sidecar.get("identity", {})
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    expected_rows = _expected_rows(cycles, cutoff + 1, cutoff + 12)
    if (identity.get("card_id") != CARD_ID or int(identity.get("cutoff", -1)) != cutoff
            or identity.get("freeze_signature") != freeze_signature
            or not np.array_equal(np.asarray(identity.get("score_rows", []), dtype=np.int64), expected_rows)):
        raise FinalContinuationIntegrityError(f"Frozen C16 pilot identity invalid at {cutoff}")
    arrays, receipt = read_bundle(path, identity)
    rows = np.asarray(arrays.get("score_rows"), dtype=np.int64)
    ids = np.asarray(arrays.get("score_record_ids"), dtype=str)
    expected_ids = frame.iloc[expected_rows]["record_id"].astype(str).to_numpy(dtype=str)
    if (not np.array_equal(rows, expected_rows) or not np.array_equal(ids, expected_ids)
            or len(np.unique(ids)) != len(ids)
            or len(rows) != EXPECTED_PILOT_ROWS[cutoff]
            or int(receipt["metadata"].get("fit_rows", -1)) != PILOT_FIT_ROWS[cutoff]
            or int(receipt["metadata"].get("fit_predictor_count", -1)) != 1):
        raise FinalContinuationIntegrityError(f"C16 pilot row/record_id alignment failed at {cutoff}")
    controls = {
        "baseline_prediction": np.asarray(baseline_record["baseline_prediction"], dtype=np.float64),
        "baseline_sparse_raw": np.asarray(baseline_record["baseline_sparse_raw"], dtype=np.float64),
        "c2b_probability": np.asarray(baseline_record["c2b_probability"], dtype=np.float64),
        "cb_probability": np.asarray(baseline_record["cb_probability"], dtype=np.float64),
        "stream_missing": np.asarray(baseline_record["stream_missing"], dtype=bool),
        "control_rows": np.asarray(baseline_record["control_rows"], dtype=np.int64),
    }
    _validated_controls(controls, len(rows))
    comparisons = {
        "baseline_prediction": "baseline_prediction",
        "baseline_sparse_raw": "control__baseline_sparse_raw",
        "c2b_probability": "control__c2b_probability",
        "cb_probability": "control__cb_probability",
        "stream_missing": "control__stream_missing",
        "control_rows": "control__control_rows",
    }
    for control_name, array_name in comparisons.items():
        if not np.array_equal(np.asarray(arrays[array_name]), controls[control_name]):
            raise FinalContinuationIntegrityError(
                f"C16 pilot control bridge changed at cutoff {cutoff}: {control_name}"
            )
    raw = np.asarray(arrays.get("raw_candidate"), dtype=np.float64)
    prediction = np.asarray(arrays.get("prediction"), dtype=np.float64)
    missing = controls["stream_missing"]
    reconstructed = np.where(missing, controls["cb_probability"],
                             0.75 * controls["c2b_probability"] + BLEND_WEIGHT * raw)
    if (raw.shape != rows.shape or prediction.shape != rows.shape
            or not np.isfinite(raw).all() or not np.array_equal(prediction, reconstructed)):
        raise FinalContinuationIntegrityError(f"C16 pilot route reconstruction failed at cutoff {cutoff}")
    return {
        "score_rows": rows, "score_record_ids": ids,
        "prediction": prediction,
        "baseline_prediction": np.asarray(arrays["baseline_prediction"], dtype=np.float64),
        "raw_candidate": raw,
        "stream_missing": missing,
        "control_arrays": controls,
        "metadata": receipt["metadata"],
        "source_bundle_sha256": _sha(path),
        "bundle_identity": identity,
    }


def _control_arrays(record: Mapping[str, Any], base: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {
        "baseline_prediction": np.asarray(base["blend25"], dtype=np.float64),
        "baseline_sparse_raw": np.asarray(base["raw_candidate"], dtype=np.float64),
        "c2b_probability": np.asarray(record["c2b"], dtype=np.float64),
        "cb_probability": np.asarray(record["cb_missing"], dtype=np.float64),
        "stream_missing": np.asarray(record["missing"], dtype=bool),
        "control_rows": np.asarray(record["rows"], dtype=np.int64),
    }


def _load_controls_for_all(frame: pd.DataFrame, target: np.ndarray,
                           baseline_freeze: Mapping[str, Any]) -> tuple[dict[int, dict[str, np.ndarray]], dict[int, dict[str, Any]]]:
    check_sparse_pilot()
    baseline_signature = str(baseline_freeze["signature"])
    records: dict[int, dict[str, Any]] = {}
    controls: dict[int, dict[str, np.ndarray]] = {}
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    for cutoff in ALL_ORIGINS:
        control = control_record(frame, target, int(cutoff))
        if cutoff in PILOT_CUTOFFS:
            base = pilot_record(frame, control)
            controls[cutoff] = _control_arrays(control, base)
            c16 = _load_c16_pilot_bundle(
                cutoff, frame, controls[cutoff],
                _read_json(PILOT_EXPERIMENT_DIR / "freeze.json")["signature"],
            )
            records[cutoff] = c16
        else:
            base = load_existing_prediction(frame, control, baseline_signature)
            if base is None:
                raise FinalContinuationIntegrityError(
                    f"Accepted sparse25 full-six control cache missing at cutoff {cutoff}"
                )
            controls[cutoff] = _control_arrays(control, base)
            rows = np.asarray(control["rows"], dtype=np.int64)
            expected = _expected_rows(cycles, cutoff + 1, cutoff + 4)
            if not np.array_equal(rows, expected):
                raise FinalContinuationIntegrityError(f"Control bridge rows changed at cutoff {cutoff}")
            records[cutoff] = {
                "baseline_record": base,
                "score_rows": rows,
                "score_record_ids": frame.iloc[rows]["record_id"].astype(str).to_numpy(dtype=str),
                "baseline_prediction": np.asarray(base["blend25"], dtype=np.float64),
                "control_arrays": controls[cutoff],
            }
    return controls, records


def _materialize_new_control_records(frame: pd.DataFrame,
                                     controls: Mapping[int, Mapping[str, np.ndarray]],
                                     baseline_records: Mapping[int, Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {
        cutoff: dict(value) for cutoff, value in baseline_records.items() if cutoff in PILOT_CUTOFFS
    }
    for cutoff in NEW_CUTOFFS:
        control = controls[cutoff]
        baseline = baseline_records[cutoff]
        rows = np.asarray(control["control_rows"], dtype=np.int64)
        records[cutoff] = {
            "score_rows": rows,
            "score_record_ids": frame.iloc[rows]["record_id"].astype(str).to_numpy(dtype=str),
            "prediction": np.full(len(rows), np.nan, dtype=np.float64),
            "baseline_prediction": np.asarray(baseline["baseline_prediction"], dtype=np.float64),
            "control_arrays": {k: np.asarray(v).copy() for k, v in control.items()},
        }
    return records


def _memory_estimate(fit_features: pd.DataFrame,
                     available_bytes: int | None = None) -> dict[str, Any]:
    documents, _missing = _text_values(fit_features["signal_stream"])
    token_counts = np.fromiter(
        (len(re.findall(r"(?u)\b\w\w+\b", value.lower())) for value in documents),
        dtype=np.int64, count=len(fit_features),
    )
    nnz_upper = int(np.sum(token_counts + np.maximum(token_counts - 1, 0), dtype=np.int64))
    sparse_bytes = int(nnz_upper * 16 + (len(fit_features) + 1) * 8)
    estimated_peak = int((1.5 * 1024**3) + 3 * sparse_bytes)
    available = int(psutil.virtual_memory().available if available_bytes is None else available_bytes)
    allowed = min(JOB_MEMORY_CAP_BYTES, available - FREE_RAM_RESERVE_BYTES)
    return {
        "status": "PASS" if allowed > 0 and estimated_peak <= allowed else "BLOCKED_RESOURCE",
        "fit_rows": int(len(fit_features)),
        "estimated_nnz_upper_bound": nnz_upper,
        "sparse_design_bytes_upper_bound": sparse_bytes,
        "estimated_peak_bytes": estimated_peak,
        "estimated_peak_is_observed": False,
        "available_physical_bytes_at_preflight": available,
        "free_memory_reserve_bytes": FREE_RAM_RESERVE_BYTES,
        "job_memory_cap_bytes": JOB_MEMORY_CAP_BYTES,
        "allowed_bytes": int(max(0, allowed)),
        "threads": THREADS,
        "concurrent_training_jobs": 1,
    }


def _base_metrics(frame: pd.DataFrame, target: np.ndarray,
                  records: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    output: dict[str, Any] = {}
    for name, segments in EVALUATION_VIEWS.items():
        rows_parts: list[np.ndarray] = []
        pred_parts: list[np.ndarray] = []
        for cutoff, low, high in segments:
            record = records[int(cutoff)]
            saved = np.asarray(record["score_rows"], dtype=np.int64)
            mask = (cycles[saved] >= int(low)) & (cycles[saved] <= int(high))
            rows = saved[mask]
            if not np.array_equal(rows, _expected_rows(cycles, low, high)):
                raise FinalContinuationIntegrityError(f"Baseline control row coverage failed for {name}")
            rows_parts.append(rows)
            pred_parts.append(np.asarray(record["baseline_prediction"], dtype=np.float64)[mask])
        rows = np.concatenate(rows_parts)
        if len(np.unique(rows)) != len(rows):
            raise FinalContinuationIntegrityError(f"Baseline control rows overlap in {name}")
        prediction = np.concatenate(pred_parts)
        output[name] = {
            "rows": int(len(rows)),
            "baseline_macro_f1": _metric(target[rows], prediction)["macro_f1"],
        }
    return output


def _make_recipe(pilot_recipe: Mapping[str, Any], pilot_freeze: Mapping[str, Any],
                 pilot_results: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "card_id": CARD_ID,
        "status": "FROZEN_C16_RECIPE_WITH_NEW_ORIGINS_ONLY",
        "pilot_recipe_sha256": _sha(PILOT_EXPERIMENT_DIR / "recipe.json"),
        "pilot_freeze_signature": pilot_freeze["signature"],
        "pilot_results_sha256": _sha(PILOT_REPORT_DIR / "results.json"),
        "pilot_cutoffs_reused_unchanged": list(PILOT_CUTOFFS),
        "new_fit_cutoffs_only": list(NEW_CUTOFFS),
        "new_predictor_fits_reserved_by_request": 4,
        "model_recipe_exact_pilot_copy": dict(pilot_recipe),
        "model_recipe_sha256": json_sha256(pilot_recipe),
        "temporal_protocol": {
            "primary_windows": {name: list(segment) for name, segment in PRIMARY_SEGMENTS.items()},
            "pooled_primary_rows": EXPECTED_PRIMARY_ROWS,
            "H12_E": [55, 56, 67], "H12_L": [67, 68, 79],
            "pooled_H12_rows": EXPECTED_H12_ROWS,
            "late_selection_W5_W6_rows": EXPECTED_LATE_ROWS,
            "new_cutoff_scoring": "each new origin scores only its next four cycles",
            "pilot_reuse_scoring": "existing cutoff55/67 pilot bundle; never refit or overwrite",
            "no_untouched_holdout_claim": True,
        },
        "continuation_qualification": {
            "primary_gain_min": PRIMARY_GAIN_MIN,
            "H12_harm_max": H12_HARM_MAX,
            "nonnegative_windows_min": NONNEGATIVE_WINDOWS_MIN,
            "late_W5_W6_gain_must_be_positive": True,
            "guard_views": list(GATE_VIEW_NAMES),
            "supported_group_minimum_n": MIN_GROUP_N,
            "supported_group_minimum_each_class": MIN_GROUP_CLASS_N,
            "macro_f1_harm_max": MAX_GROUP_F1_HARM,
            "positive_recall_harm_max": MAX_GROUP_POS_RECALL_HARM,
            "groups": ["entry_friction", "stream_availability", "text_load_quantile"],
            "text_load_quantiles_fit_only_on_each_origin_prefix": True,
        },
        "execution_policy": {
            "candidate_fit_count": 4,
            "threads": THREADS,
            "concurrent_training_jobs": 1,
            "job_memory_cap_bytes": JOB_MEMORY_CAP_BYTES,
            "minimum_free_ram_reserve_bytes": FREE_RAM_RESERVE_BYTES,
            "outer_score_labels_passed_to_adapter": False,
            "test_data_used": False,
            "sample_submission_read": False,
            "kaggle_access": False,
        },
        "pilot_gate_history": {
            "pilot_result_status": pilot_results.get("status"),
            "pilot_verification_decision": _read_json(PILOT_REPORT_DIR / "verification.json").get("decision"),
            "historical_only_not_a_continuation_verdict": True,
        },
    }


def _validate_pilot_artifacts() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    queue = _read_json(QUEUE_PATH)
    card = next((row for row in queue.get("cards", []) if row.get("id") == CARD_ID), None)
    if card is None:
        raise FinalContinuationIntegrityError("C16 is missing from canonical queue")
    if _sha(CARD_PATH) != CARD_SHA256:
        raise FinalContinuationIntegrityError("Canonical C16 card file hash differs from frozen pilot")
    pilot_freeze = _read_json(PILOT_EXPERIMENT_DIR / "freeze.json")
    pilot_recipe = _read_json(PILOT_EXPERIMENT_DIR / "recipe.json")
    pilot_results = _read_json(PILOT_REPORT_DIR / "results.json")
    pilot_verification = _read_json(PILOT_REPORT_DIR / "verification.json")
    if (pilot_freeze.get("signature") != json_sha256(pilot_freeze.get("identity", {}))
            or pilot_freeze.get("signature") != pilot_results.get("freeze_signature")
            or pilot_results.get("cutoffs") != [55, 67]
            or pilot_results.get("fit_predictor_count") != 2
            or pilot_results.get("status") != "PILOT_COMPLETE_ADVANCE"
            or pilot_verification.get("status") != "INDEPENDENT_QA_PASS"
            or pilot_verification.get("decision") != "ADVANCE"
            or pilot_recipe.get("card_id") != CARD_ID
            or pilot_recipe.get("lightgbm_parameters") != PARAMS
            or pilot_recipe.get("only_parameter_changes") != CHANGES
            or pilot_recipe.get("feature_allowlist") != list(FEATURE_NAMES)
            or pilot_freeze.get("identity", {}).get("recipe_artifact_sha256") != json_sha256(pilot_recipe)):
        raise FinalContinuationIntegrityError("Historical C16 pilot recipe/results/freeze are inconsistent")
    queue_audit = _audit_queue_drift(pilot_freeze, queue)
    expected_card_fields = {
        "id": CARD_ID, "task": "classification", "family": "sparse_capacity",
        "policy": "C_SPARSE_REPLACE", "pilot_predictor_fits": 2,
        "evaluation": "STANDARD", "condition": "No additional dependency",
    }
    for name, value in expected_card_fields.items():
        if card.get(name) != value:
            raise FinalContinuationIntegrityError(f"C16 canonical card field changed: {name}")
    if queue.get("baseline", {}).get("classification", {}).get("Train_sha256") != TRAIN_SHA256:
        raise FinalContinuationIntegrityError("Canonical classification baseline Train hash changed")
    pilot_freeze["_queue_drift_audit"] = queue_audit
    return queue, pilot_freeze, pilot_recipe


def _context() -> dict[str, Any]:
    queue, pilot_freeze, pilot_recipe = _validate_pilot_artifacts()
    baseline_evidence = _validate_baseline_artifacts(queue)
    baseline_freeze = check_baseline_fullsix_freeze()
    train_path, frame, target, cycles = _official_train()
    if any(len(_expected_rows(cycles, lo, hi)) != expected for (_, lo, hi), expected in zip(
            PRIMARY_SEGMENTS.values(), [4_114, 6_493, 7_078, 4_950, 7_462, 8_491])):
        raise FinalContinuationIntegrityError("Official Train primary window denominator differs from frozen plan")
    if sum(len(_expected_rows(cycles, low, high)) for _, low, high in PRIMARY_SEGMENTS.values()) != EXPECTED_PRIMARY_ROWS:
        raise FinalContinuationIntegrityError("Primary pooled denominator differs from DECISION")
    if len(_expected_rows(cycles, 56, 67)) + len(_expected_rows(cycles, 68, 79)) != EXPECTED_H12_ROWS:
        raise FinalContinuationIntegrityError("H12 pooled denominator differs from DECISION")
    if len(_expected_rows(cycles, 72, 79)) != EXPECTED_LATE_ROWS:
        raise FinalContinuationIntegrityError("W5+W6 denominator differs from DECISION")

    controls, source_records = _load_controls_for_all(frame, target, baseline_freeze)
    records = _materialize_new_control_records(frame, controls, source_records)
    baseline_views = _base_metrics(frame, target, records)
    accepted = baseline_evidence["declared_fullsix_control"]
    if (not np.isclose(baseline_views["primary"]["baseline_macro_f1"],
                       float(accepted["primary_macro_f1"]), atol=1e-15, rtol=0)
            or not np.isclose(baseline_views["H12"]["baseline_macro_f1"],
                              float(accepted["H12_macro_f1"]), atol=1e-15, rtol=0)):
        raise FinalContinuationIntegrityError("Six-window/H12 row-matched champion controls differ from accepted full-six evidence")

    input_files = _input_files(train_path)
    missing_inputs = [str(path) for path in input_files.values() if not path.is_file()]
    if missing_inputs:
        raise FileNotFoundError(f"Continuation inputs are missing: {missing_inputs}")
    source_files = _source_files(pilot_freeze)
    missing_sources = [str(path) for path in source_files.values() if not path.is_file()]
    if missing_sources:
        raise FileNotFoundError(f"Continuation sources are missing: {missing_sources}")
    old_engine_sha = pilot_freeze["identity"]["source_sha256"].get(
        str((ROOT / "src/models/optimization_v3_batch1.py").resolve())
    )
    if not old_engine_sha:
        raise FinalContinuationIntegrityError("Pilot freeze lacks shared batch engine source hash")
    engine_audit = _current_shared_engine_audit(old_engine_sha)
    source_hashes = {name: _sha(path) for name, path in sorted(source_files.items())}
    input_hashes = {
        path.resolve().relative_to(ROOT).as_posix(): _sha(path)
        for _name, path in sorted(input_files.items())
    }
    source_drift: dict[str, Any] = {}
    for old_path, expected in pilot_freeze["identity"].get("source_sha256", {}).items():
        path = Path(old_path)
        if not path.is_absolute():
            path = ROOT / path
        rel = path.resolve().relative_to(ROOT).as_posix()
        current_hash = source_hashes.get(rel)
        if current_hash != expected:
            source_drift[rel] = {"pilot_sha256": expected, "current_sha256": current_hash}
    if set(source_drift) != {"src/models/optimization_v3_batch1.py"}:
        raise FinalContinuationIntegrityError(
            f"Unexpected source drift from C16 pilot freeze: {sorted(source_drift)}"
        )
    old_pilot_views = {
        "W1": [(55, 56, 59)], "W4": [(67, 68, 71)],
        "primary": [(55, 56, 59), (67, 68, 71)],
        "H12": [(55, 56, 67), (67, 68, 79)],
    }
    pilot_inputs = _control_input_files(train_path, CARD_PATH, old_pilot_views)
    pilot_input_hashes = pilot_freeze["identity"].get("input_sha256", {})
    if set(pilot_inputs) != set(pilot_input_hashes):
        raise FinalContinuationIntegrityError("Historical C16 pilot input inventory changed")
    changed_pilot_inputs = {
        name: {"pilot_sha256": pilot_input_hashes[name], "current_sha256": _sha(path)}
        for name, path in pilot_inputs.items()
        if name != "canonical_queue" and _sha(path) != pilot_input_hashes[name]
    }
    if changed_pilot_inputs:
        raise FinalContinuationIntegrityError(
            f"Historical C16 pilot inputs changed: {sorted(changed_pilot_inputs)}"
        )
    resources: dict[str, Any] = {}
    for cutoff in NEW_CUTOFFS:
        fit_rows = np.flatnonzero(cycles <= cutoff)
        if len(fit_rows) != EXPECTED_FIT_ROWS[cutoff]:
            raise FinalContinuationIntegrityError(f"C16 continuation fit row count differs at {cutoff}")
        resources[str(cutoff)] = _memory_estimate(
            frame.iloc[fit_rows].loc[:, list(FEATURE_NAMES)]
        )
    available_inputs = {
        "official_train_rows": int(len(frame)),
        "official_train_sha256": _sha(train_path),
        "cycles": [47, 79],
        "fit_rows_by_new_cutoff": {str(c): EXPECTED_FIT_ROWS[c] for c in NEW_CUTOFFS},
        "score_rows_by_new_cutoff": {str(c): EXPECTED_NEW_WINDOW_ROWS[c] for c in NEW_CUTOFFS},
        "primary_rows": EXPECTED_PRIMARY_ROWS,
        "H12_rows": EXPECTED_H12_ROWS,
        "late_W5_W6_rows": EXPECTED_LATE_ROWS,
    }
    recipe = _make_recipe(pilot_recipe, pilot_freeze,
                          _read_json(PILOT_REPORT_DIR / "results.json"))
    identity = {
        "schema_version": 1,
        "card_id": CARD_ID,
        "pilot_freeze_signature": pilot_freeze["signature"],
        "pilot_recipe_sha256": _sha(PILOT_EXPERIMENT_DIR / "recipe.json"),
        "recipe_sha256": json_sha256(recipe),
        "review_decision_sha256": _sha(DECISION_DIR / "DECISION.md"),
        "review_guard_audit_sha256": _sha(DECISION_DIR / "guard_audit.json"),
        "pilot_queue_drift_audit": pilot_freeze["_queue_drift_audit"],
        "new_cutoffs_only": list(NEW_CUTOFFS),
        "reused_pilot_cutoffs": list(PILOT_CUTOFFS),
        "evaluation_views": {name: [list(segment) for segment in segments]
                             for name, segments in EVALUATION_VIEWS.items()},
        "expected_fit_rows": {str(c): EXPECTED_FIT_ROWS[c] for c in NEW_CUTOFFS},
        "expected_score_rows": {str(c): EXPECTED_NEW_WINDOW_ROWS[c] for c in NEW_CUTOFFS},
        "expected_primary_rows": EXPECTED_PRIMARY_ROWS,
        "expected_H12_rows": EXPECTED_H12_ROWS,
        "expected_late_rows": EXPECTED_LATE_ROWS,
        "source_sha256": source_hashes,
        "input_sha256": input_hashes,
        "shared_engine_drift_audit": engine_audit,
        "pilot_source_drift": source_drift,
        "pilot_input_hashes_revalidated": True,
        "package_versions": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "threadpoolctl", "psutil"]
        },
        "feature_allowlist": list(FEATURE_NAMES),
        "target": "artifact_signal",
        "test_data_used": False,
        "sample_submission_read": False,
        "kaggle_access": False,
        "validation_labels_passed_to_adapter": False,
    }
    resources_pass = all(item["status"] == "PASS" for item in resources.values())
    return {
        "queue": queue,
        "pilot_freeze": pilot_freeze,
        "pilot_recipe": pilot_recipe,
        "baseline_evidence": baseline_evidence,
        "baseline_freeze": baseline_freeze,
        "train_path": train_path,
        "frame": frame,
        "target": target,
        "cycles": cycles,
        "controls": controls,
        "records": records,
        "baseline_views": baseline_views,
        "recipe": recipe,
        "identity": identity,
        "freeze_signature": json_sha256(identity),
        "resources": resources,
        "resources_pass": resources_pass,
        "inputs": available_inputs,
    }


def _write_or_match(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        if _read_json(path) != dict(value):
            raise FinalContinuationIntegrityError(f"Refusing to overwrite nonmatching frozen artifact: {path}")
        return
    write_json_exclusive(path, dict(value))


def preflight_and_freeze() -> dict[str, Any]:
    """Read official Train/caches, estimate resources, and freeze C16-only lane."""
    context = _context()
    preflight_status = "READY_TO_FIT" if context["resources_pass"] else "BLOCKED_RESOURCE"
    identity = dict(context["identity"])
    signature = context["freeze_signature"]
    freeze = {
        "status": "FROZEN_BEFORE_FOUR_C16_CONTINUATION_FITS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_head_contextual": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "identity": identity,
        "signature": signature,
    }
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    _write_or_match(RECIPE_PATH, context["recipe"])
    if FREEZE_PATH.exists():
        existing = _read_json(FREEZE_PATH)
        if existing.get("identity") != identity or existing.get("signature") != signature:
            raise FinalContinuationIntegrityError("Existing continuation freeze differs from current source/input identity")
        freeze = existing
    else:
        write_json_exclusive(FREEZE_PATH, freeze)
    preflight = {
        "card_id": CARD_ID,
        "status": preflight_status,
        "scope": "C16 only; official Classification Train and existing frozen controls; no estimator fit",
        "freeze_signature": signature,
        "official_train": context["inputs"],
        "baseline_fullsix_metrics": context["baseline_views"],
        "accepted_baseline_evidence": context["baseline_evidence"],
        "resource_estimates_by_new_cutoff": context["resources"],
        "proposed_budget_reservation": {
            "budget_key": "C16:FINAL_CONTINUATION",
            "card_id": CARD_ID,
            "predictor_fits": 4,
            "reserved": False,
            "global_ledger_mutated": False,
        },
        "source_sha256": identity["source_sha256"],
        "input_sha256": identity["input_sha256"],
        "pilot_source_drift": identity["pilot_source_drift"],
        "shared_engine_drift_audit": identity["shared_engine_drift_audit"],
        "current_execution_gate": {
            "resources_pass": context["resources_pass"],
            "lock_check_required_before_execute": True,
            "fresh_ram_check_required_before_execute": True,
            "no_fit_executed": True,
            "test_or_sample_access": False,
            "kaggle_access": False,
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
    }
    if PREFLIGHT_PATH.exists():
        existing_preflight = _read_json(PREFLIGHT_PATH)
        if existing_preflight.get("freeze_signature") != signature:
            raise FinalContinuationIntegrityError("Existing preflight belongs to a different freeze")
        preflight = existing_preflight
    else:
        write_json_exclusive(PREFLIGHT_PATH, preflight)
    return {"recipe": context["recipe"], "freeze": freeze, "preflight": preflight}


def _load_new_bundle(cutoff: int, freeze_signature: str,
                     frame: pd.DataFrame, score_rows: np.ndarray) -> dict[str, Any] | None:
    path = RUN_DIR / f"C16_fit{cutoff}.npz"
    sidecar_path = path.with_suffix(".json")
    if not path.exists() and not sidecar_path.exists():
        return None
    if not (path.is_file() and sidecar_path.is_file()):
        raise FinalContinuationIntegrityError(f"Partial continuation bundle at cutoff {cutoff}; refuse overwrite")
    identity = {"freeze_signature": freeze_signature, "card_id": CARD_ID,
                "cutoff": int(cutoff), "score_rows": score_rows.tolist()}
    arrays, sidecar = read_bundle(path, identity)
    expected_ids = frame.iloc[score_rows]["record_id"].astype(str).to_numpy(dtype=str)
    if (not np.array_equal(arrays.get("score_rows"), score_rows)
            or not np.array_equal(arrays.get("score_record_ids"), expected_ids)):
        raise FinalContinuationIntegrityError(f"Continuation cache row/record_id mismatch at cutoff {cutoff}")
    prediction = np.asarray(arrays.get("prediction"), dtype=np.float64)
    baseline = np.asarray(arrays.get("baseline_prediction"), dtype=np.float64)
    raw = np.asarray(arrays.get("raw_candidate"), dtype=np.float64)
    if (prediction.shape != score_rows.shape or baseline.shape != score_rows.shape
            or raw.shape != score_rows.shape or not np.isfinite(prediction).all()
            or not np.isfinite(baseline).all() or not np.isfinite(raw).all()
            or np.any((prediction < 0) | (prediction > 1))
            or int(sidecar["metadata"].get("fit_predictor_count", -1)) != 1):
        raise FinalContinuationIntegrityError(f"Continuation cache values/fit count invalid at cutoff {cutoff}")
    return {
        "score_rows": score_rows.copy(), "score_record_ids": expected_ids,
        "prediction": prediction, "baseline_prediction": baseline,
        "raw_candidate": raw,
        "control_arrays": {
            "baseline_prediction": baseline,
            "baseline_sparse_raw": np.asarray(arrays["control__baseline_sparse_raw"], dtype=np.float64),
            "c2b_probability": np.asarray(arrays["control__c2b_probability"], dtype=np.float64),
            "cb_probability": np.asarray(arrays["control__cb_probability"], dtype=np.float64),
            "stream_missing": np.asarray(arrays["control__stream_missing"], dtype=bool),
            "control_rows": np.asarray(arrays["control__control_rows"], dtype=np.int64),
        },
        "metadata": sidecar["metadata"],
        "source_bundle_sha256": _sha(path),
    }


def execute_cutoff(context: Mapping[str, Any], freeze_signature: str,
                   cutoff: int, *, require_frozen: bool = True) -> tuple[dict[str, Any], bool]:
    """Fit or resume one approved C16 origin and save its isolated bundle.

    This single-origin API deliberately does not acquire a lock: the campaign
    coordinator must hold its one global training slot across C16 and R22.
    It never reads or writes either pilot cutoff bundle.
    """
    cutoff = int(cutoff)
    if cutoff not in NEW_CUTOFFS:
        raise FinalContinuationIntegrityError(f"C16 continuation cutoff is not authorized: {cutoff}")
    if require_frozen:
        if context.get("freeze_signature") != freeze_signature:
            raise FinalContinuationIntegrityError("Caller context does not match supplied freeze signature")
        _verify_current_freeze(context)
        if not context.get("resources_pass", False):
            raise FinalContinuationIntegrityError("Current resource estimate blocks this C16 fit")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    frame = context["frame"]
    target = np.asarray(context["target"], dtype=np.int64)
    cycles = np.asarray(context["cycles"], dtype=np.int64)
    records = _materialize_new_control_records(frame, context["controls"], context["records"])
    fit_rows = np.flatnonzero((cycles >= 47) & (cycles <= cutoff))
    score_rows = np.asarray(records[cutoff]["score_rows"], dtype=np.int64)
    expected_score = _expected_rows(cycles, cutoff + 1, cutoff + 4)
    if not np.array_equal(score_rows, expected_score):
        raise FinalContinuationIntegrityError(f"Wrong continuation score rows at cutoff {cutoff}")
    cached = _load_new_bundle(cutoff, freeze_signature, frame, score_rows)
    if cached is not None:
        if not np.array_equal(cached["baseline_prediction"], records[cutoff]["baseline_prediction"]):
            raise FinalContinuationIntegrityError(f"Continuation baseline changed at cutoff {cutoff}")
        return cached, False

    fit_features = frame.iloc[fit_rows].loc[:, list(FEATURE_NAMES)].copy()
    score_features = frame.iloc[score_rows].loc[:, list(FEATURE_NAMES)].copy()
    controls = {name: np.asarray(value).copy()
                for name, value in context["controls"][cutoff].items()}
    started = time.perf_counter()
    append_event(EVENT_PATH, {
        "event": "FIT_STARTED", "card_id": CARD_ID, "cutoff": cutoff,
        "freeze_signature": freeze_signature,
        "fit_rows": int(len(fit_rows)), "score_rows": int(len(score_rows)),
    })
    try:
        with threadpool_limits(limits=THREADS):
            result = dict(fit_predict(
                fit_features, target[fit_rows].copy(), score_features,
                controls=controls, card_id=CARD_ID, cutoff=cutoff,
            ))
        prediction = np.asarray(result["prediction"], dtype=np.float64)
        raw = np.asarray(result["raw_candidate"], dtype=np.float64)
        if (prediction.shape != score_rows.shape or raw.shape != score_rows.shape
                or not np.isfinite(prediction).all() or not np.isfinite(raw).all()
                or np.any((prediction < 0) | (prediction > 1))
                or int(result.get("fit_predictor_count", -1)) != 1):
            raise FinalContinuationIntegrityError(f"C16 model returned invalid outputs at cutoff {cutoff}")
        ids = frame.iloc[score_rows]["record_id"].astype(str).to_numpy(dtype=str)
        arrays = {
            "score_rows": score_rows,
            "score_record_ids": ids,
            "baseline_prediction": np.asarray(controls["baseline_prediction"], dtype=np.float64),
            "prediction": prediction,
            "raw_candidate": raw,
            **{f"control__{key}": np.asarray(value)
               for key, value in controls.items() if key != "baseline_prediction"},
            **{f"audit__{key}": np.asarray(value)
               for key, value in result.get("audit_arrays", {}).items()},
        }
        identity = {"freeze_signature": freeze_signature, "card_id": CARD_ID,
                    "cutoff": cutoff, "score_rows": score_rows.tolist()}
        metadata = {
            "card_id": CARD_ID, "cutoff": cutoff,
            "fit_predictor_count": 1,
            "fit_rows": int(len(fit_rows)), "score_rows": int(len(score_rows)),
            "fit_seconds": float(time.perf_counter() - started),
            "fit_metadata": result.get("fit_metadata", {}),
            "state": result.get("state", {}),
            "adapter_independent_reconstruction_required": True,
            "test_data_used": False,
            "outer_score_labels_used_for_fit": False,
        }
        path = RUN_DIR / f"C16_fit{cutoff}.npz"
        write_bundle_exclusive(path, arrays, identity, metadata)
        append_event(EVENT_PATH, {
            "event": "FIT_SAVED", "card_id": CARD_ID, "cutoff": cutoff,
            "fit_predictor_count": 1, "fit_seconds": metadata["fit_seconds"],
        })
        return {
            "score_rows": score_rows, "score_record_ids": ids,
            "prediction": prediction,
            "baseline_prediction": arrays["baseline_prediction"],
            "raw_candidate": raw,
            "control_arrays": {name: np.asarray(value).copy() for name, value in controls.items()},
            "metadata": metadata,
        }, True
    except Exception as exc:
        append_event(EVENT_PATH, {
            "event": "FIT_FAILED", "card_id": CARD_ID, "cutoff": cutoff,
            "error_type": type(exc).__name__, "error": str(exc)[:1000],
        })
        raise


def execute_new_fits(context: Mapping[str, Any], freeze_signature: str,
                     *, lock: bool = True,
                     require_frozen: bool = True) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """Fit/resume all four new origins under one local training slot."""
    records: dict[int, dict[str, Any]] = {
        cutoff: dict(context["records"][cutoff]) for cutoff in PILOT_CUTOFFS
    }
    attempts = {"fit_calls_this_invocation": 0, "resumed_bundle_count": 0, "new_bundle_count": 0}
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    slot = exclusive_training_slot(RUN_DIR / ".training.lock") if lock else nullcontext()
    with slot:
        for cutoff in NEW_CUTOFFS:
            record, fitted = execute_cutoff(
                context, freeze_signature, cutoff, require_frozen=require_frozen,
            )
            records[cutoff] = record
            if fitted:
                attempts["fit_calls_this_invocation"] += 1
                attempts["new_bundle_count"] += 1
            else:
                attempts["resumed_bundle_count"] += 1
    return records, attempts


def _verify_current_freeze(context: Mapping[str, Any]) -> str:
    if not FREEZE_PATH.is_file() or not RECIPE_PATH.is_file() or not PREFLIGHT_PATH.is_file():
        raise FinalContinuationIntegrityError("Continuation recipe/freeze/preflight must be created before fit")
    frozen = _read_json(FREEZE_PATH)
    signature = str(frozen.get("signature", ""))
    preflight = _read_json(PREFLIGHT_PATH)
    if (preflight.get("status") != "READY_TO_FIT"
            or not preflight.get("current_execution_gate", {}).get("resources_pass", False)
            or preflight.get("current_execution_gate", {}).get("no_fit_executed") is not True
            or frozen.get("identity") != context["identity"]
            or signature != json_sha256(context["identity"])
            or _read_json(RECIPE_PATH) != context["recipe"]
            or preflight.get("freeze_signature") != signature):
        raise FinalContinuationIntegrityError("Current inputs/source differ from frozen C16 continuation receipt")
    return signature


def run_execution() -> dict[str, Any]:
    """MAIN-only entry point; intentionally not invoked during worker preflight."""
    context = _context()
    if not context["resources_pass"]:
        raise FinalContinuationIntegrityError("Resource preflight blocked; do not start training")
    signature = _verify_current_freeze(context)
    records, _attempts_this_invocation = execute_new_fits(context, signature, lock=True)
    report = build_report(context["frame"], context["target"], records, signature)
    verification = verify_report_independently(
        context["frame"], context["target"], records, report, signature,
    )
    report["fit_accounting"] = {
        "candidate_predictor_fits_represented": 4,
        "reused_pilot_fit_count": 2,
        "total_new_cutoff_fit_predictors": sum(
            int(records[cutoff]["metadata"].get("fit_predictor_count", 1))
            for cutoff in NEW_CUTOFFS
        ),
        "fit_seconds_by_origin": {
            str(cutoff): float(records[cutoff]["metadata"].get(
                "fit_seconds", records[cutoff]["metadata"].get("fit_seconds", 0.0)
            )) for cutoff in NEW_CUTOFFS
        },
        "attempt_history_path": str(EVENT_PATH.relative_to(ROOT)),
        "attempt_history_note": "Per-invocation retries/resume events are recorded in the isolated append-only event log.",
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    _write_or_match(RESULT_PATH, report)
    _write_or_match(VERIFICATION_PATH, verification)
    return {"results": report, "verification": verification}


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight-freeze", action="store_true",
                       help="read Train/caches, estimate resources, and write exclusive C16 freeze")
    group.add_argument("--execute", action="store_true",
                       help="MAIN-only execution after separate approval; runs four C16 fits")
    args = parser.parse_args()
    try:
        result = preflight_and_freeze() if args.preflight_freeze else run_execution()
        if args.preflight_freeze:
            print(json.dumps(result["preflight"], indent=2, allow_nan=False))
        else:
            print(json.dumps({"status": result["verification"]["status"],
                              "qualified": result["verification"]["qualification"]["qualified"],
                              "views": result["verification"]["views"]},
                             indent=2, allow_nan=False))
        return 0
    except (BatchIntegrityError, FileNotFoundError, AssertionError, ValueError) as exc:
        print(f"BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

