"""Independent verifier for persisted V3 batch1 pilot predictions.

This intentionally uses its own metric implementations rather than importing
the training runner's ``metric`` function.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.models.optimization_v3_batch1 import BatchIntegrityError, read_bundle


def _macro_f1_independent(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.int64)
    scores: list[float] = []
    for cls in (0, 1):
        actual_positive = int(np.count_nonzero(y == cls))
        predicted_positive = int(np.count_nonzero(p == cls))
        true_positive = int(np.count_nonzero((y == cls) & (p == cls)))
        denom = actual_positive + predicted_positive
        scores.append(0.0 if denom == 0 else (2.0 * true_positive) / denom)
    return float(np.mean(scores))


def _mae_independent(y: np.ndarray, p: np.ndarray) -> float:
    error = np.asarray(p, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    return float(np.abs(error).sum(dtype=np.float64) / error.size)


def _missing_category(value: Any) -> bool:
    if value is None:
        return True
    try:
        if bool(pd.isna(value)):
            return True
    except (TypeError, ValueError):
        pass
    return not str(value).strip()


def _entry_friction_label(value: Any) -> str:
    if _missing_category(value):
        return "missing"
    try:
        number = float(value)
        if np.isfinite(number) and number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _text_load_labels(frame: pd.DataFrame, cutoff: int, rows: np.ndarray) -> tuple[list[str], list[str]]:
    cycles = frame.cycle_code.to_numpy(dtype=int)
    train = pd.to_numeric(frame.loc[cycles <= cutoff, "text_load"], errors="coerce").to_numpy(dtype=float)
    score = pd.to_numeric(frame.iloc[rows]["text_load"], errors="coerce").to_numpy(dtype=float)
    finite = train[np.isfinite(train)]
    if not len(finite):
        return ["missing"] * len(rows), []
    edges = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, 5)))
    valid = np.isfinite(score)
    indices = np.full(len(score), -1, dtype=int)
    if len(edges) == 1:
        indices[valid] = 0
    else:
        indices[valid] = np.searchsorted(edges[1:-1], score[valid], side="left")
    labels = ["missing" if value < 0 else f"Q{value + 1}" for value in indices]
    return labels, [f"Q{i + 1}" for i in range(max(1, len(edges) - 1))]


def _novelty_labels(frame: pd.DataFrame, cutoff: int, rows: np.ndarray, column: str) -> list[str]:
    cycles = frame.cycle_code.to_numpy(dtype=int)
    fit_values = frame.loc[cycles <= cutoff, column]
    known = {str(x).strip() for x in fit_values.tolist() if not _missing_category(x)}
    labels = []
    for value in frame.iloc[rows][column].tolist():
        if _missing_category(value):
            labels.append("missing")
        elif str(value).strip() in known:
            labels.append("seen")
        else:
            labels.append("unseen")
    return labels


def _classification_group_metrics(
    frame: pd.DataFrame, y: np.ndarray, candidate_probability: np.ndarray,
    baseline_probability: np.ndarray, rows: np.ndarray, row_cutoffs: np.ndarray,
) -> dict[str, Any]:
    candidate = np.asarray(candidate_probability, dtype=float) >= 0.5
    baseline = np.asarray(baseline_probability, dtype=float) >= 0.5
    y = np.asarray(y, dtype=np.int64)
    groups: dict[str, list[str]] = {}
    gate_fields: set[str] = set()

    if "signal_stream" in frame:
        values = frame.iloc[rows]["signal_stream"].tolist()
        labels = ["missing" if _missing_category(x) else "present" for x in values]
        groups["signal_stream"] = labels
        gate_fields.add("signal_stream")
    if "entry_friction" in frame:
        groups["entry_friction"] = [
            _entry_friction_label(x)
            for x in frame.iloc[rows]["entry_friction"].tolist()
        ]
        gate_fields.add("entry_friction")
    if "text_load" in frame:
        text_labels: list[str] = [""] * len(rows)
        for cutoff in sorted(np.unique(row_cutoffs)):
            mask = row_cutoffs == cutoff
            current, _ = _text_load_labels(frame, int(cutoff), rows[mask])
            for position, label in zip(np.flatnonzero(mask), current):
                text_labels[position] = label
        groups["text_load_q"] = text_labels
        gate_fields.add("text_load_q")

    for column in ("org_code", "module_code", "pattern_code"):
        if column in frame:
            groups[f"{column}_novelty"] = [
                "" for _ in range(len(rows))
            ]
            labels = groups[f"{column}_novelty"]
            for cutoff in sorted(np.unique(row_cutoffs)):
                mask = row_cutoffs == cutoff
                subset = _novelty_labels(frame, int(cutoff), rows[mask], column)
                for position, label in zip(np.flatnonzero(mask), subset):
                    labels[position] = label

    details: dict[str, Any] = {}
    gate_failures: list[dict[str, Any]] = []
    for field, labels in groups.items():
        field_result: dict[str, Any] = {}
        observed = sorted(set(labels))
        expected = observed
        if field == "signal_stream":
            expected = sorted(set(observed) | {"present", "missing"})
        elif field == "entry_friction":
            expected = sorted(set(observed) | {"0", "1", "missing"})
        elif field == "text_load_q":
            expected = sorted(set(observed) | {f"Q{i}" for i in range(1, 5)} | {"missing"})
        elif field.endswith("_novelty"):
            expected = sorted(set(observed) | {"seen", "unseen", "missing"})
        for group in expected:
            mask = np.asarray([label == group for label in labels], dtype=bool)
            n = int(mask.sum())
            n0 = int(np.sum(y[mask] == 0))
            n1 = int(np.sum(y[mask] == 1))
            supported = n >= 200 and n0 >= 30 and n1 >= 30
            if n:
                cand_f1 = _macro_f1_independent(y[mask], candidate[mask])
                base_f1 = _macro_f1_independent(y[mask], baseline[mask])
                cand_recall = float(np.sum(candidate[mask] & (y[mask] == 1)) / n1) if n1 else None
                base_recall = float(np.sum(baseline[mask] & (y[mask] == 1)) / n1) if n1 else None
                f1_delta = cand_f1 - base_f1
                recall_delta = (cand_recall - base_recall) if n1 else None
            else:
                cand_f1 = base_f1 = f1_delta = cand_recall = base_recall = recall_delta = None
            gate_applies = field in gate_fields and supported
            gate_pass = None if not gate_applies else bool(f1_delta >= -0.01 and recall_delta >= -0.02)
            record = {
                "n": n, "class_0_n": n0, "class_1_n": n1, "supported": supported,
                "candidate_macro_f1": cand_f1, "baseline_macro_f1": base_f1,
                "candidate_minus_baseline_macro_f1": f1_delta,
                "candidate_positive_recall": cand_recall, "baseline_positive_recall": base_recall,
                "candidate_minus_baseline_positive_recall": recall_delta,
                "guard_applies": gate_applies, "guard_pass": gate_pass,
            }
            field_result[group] = record
            if gate_applies and not gate_pass:
                gate_failures.append({"field": field, "group": group, **record})
        details[field] = field_result
    return {
        "support_rule": "guard only when n>=200 and each class count>=30; smaller groups are descriptive",
        "groups": details, "supported_guard_failures": gate_failures,
    }


def reconstruct_c03(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild C03 route plus challenger and frozen sparse25 baseline."""
    raw = np.asarray(arrays["raw_candidate"], dtype=np.float64)
    missing = np.asarray(arrays["control__stream_missing"], dtype=bool)
    c2b = np.asarray(arrays["control__c2b_probability"], dtype=np.float64)
    cb = np.asarray(arrays["control__cb_probability"], dtype=np.float64)
    old_sparse = np.asarray(arrays["control__baseline_sparse_raw"], dtype=np.float64)
    if len({len(x) for x in (raw, missing, c2b, cb, old_sparse)}) != 1:
        raise BatchIntegrityError("C03 route inputs have inconsistent lengths")
    challenger = np.where(missing, cb, 0.75 * c2b + 0.25 * raw)
    baseline = np.where(missing, cb, 0.75 * c2b + 0.25 * old_sparse)
    return challenger, baseline


def verify_report(
    *, report: dict[str, Any], frame: pd.DataFrame, target: np.ndarray,
    artifact_dir: str | Path, freeze_signature: str,
    evaluation_views: dict[str, list[tuple[int, int, int]]],
    prediction_reconstructors: dict[str, Callable[[dict[str, np.ndarray]], tuple[np.ndarray, np.ndarray]]] | None = None,
) -> dict[str, Any]:
    task = str(report["task"])
    y = np.asarray(target)
    cycles = frame.cycle_code.to_numpy(dtype=int)
    verified: dict[str, Any] = {}
    for name, segments in evaluation_views.items():
        ys: list[np.ndarray] = []
        ps: list[np.ndarray] = []
        bs: list[np.ndarray] = []
        rows_seen: list[np.ndarray] = []
        cutoffs_seen: list[np.ndarray] = []
        for cutoff, low, high in segments:
            path = Path(artifact_dir) / f"{report['card_id']}_fit{cutoff}.npz"
            identity = {"freeze_signature": freeze_signature, "card_id": report["card_id"], "cutoff": int(cutoff)}
            # The runner identity includes ordered score row positions; recover it from
            # the sidecar first, then validate all immutable fields and the NPZ digest.
            sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            stored_identity = sidecar.get("identity", {})
            if any(stored_identity.get(k) != v for k, v in identity.items()):
                raise BatchIntegrityError(f"Freeze/card/cutoff identity mismatch for {path}")
            arrays, _ = read_bundle(path, stored_identity)
            rows = np.asarray(arrays["score_rows"], dtype=int)
            if rows.ndim != 1 or len(np.unique(rows)) != len(rows) or np.any((rows < 0) | (rows >= len(frame))):
                raise BatchIntegrityError(f"Invalid score row positions in {path}")
            expected_cycles = {
                int(cycle)
                for view_segments in evaluation_views.values()
                for origin, view_low, view_high in view_segments
                if int(origin) == int(cutoff)
                for cycle in range(int(view_low), int(view_high) + 1)
            }
            expected_rows = np.flatnonzero(np.isin(cycles, np.asarray(sorted(expected_cycles), dtype=int)))
            if not np.array_equal(rows, expected_rows):
                raise BatchIntegrityError(f"Saved score rows do not exactly match declared views for cutoff {cutoff}")
            stored_ids = np.asarray(arrays.get("score_record_ids", []), dtype=str)
            expected_ids = frame.iloc[rows]["record_id"].astype(str).to_numpy(dtype=str)
            if stored_ids.shape != expected_ids.shape or not np.array_equal(stored_ids, expected_ids):
                raise BatchIntegrityError(f"Ordered record_id alignment differs in {path}")
            if len(np.unique(stored_ids)) != len(stored_ids):
                raise BatchIntegrityError(f"Duplicate record_id in saved predictions: {path}")
            pred = np.asarray(arrays.get("prediction"), dtype=np.float64)
            baseline_pred = np.asarray(arrays.get("baseline_prediction"), dtype=np.float64)
            if pred.shape != rows.shape or baseline_pred.shape != rows.shape:
                raise BatchIntegrityError(f"Prediction arrays are not aligned in {path}")
            if not np.isfinite(pred).all() or not np.isfinite(baseline_pred).all():
                raise BatchIntegrityError(f"Nonfinite prediction in {path}")
            if task == "classification" and (np.any((pred < 0) | (pred > 1)) or np.any((baseline_pred < 0) | (baseline_pred > 1))):
                raise BatchIntegrityError(f"Classification probability outside [0,1] in {path}")
            if task == "regression" and (np.any((pred < 0) | (pred > 100)) or np.any((baseline_pred < 0) | (baseline_pred > 100))):
                raise BatchIntegrityError(f"Regression prediction outside [0,100] in {path}")
            if task == "classification" and not np.isin(y[rows], [0, 1]).all():
                raise BatchIntegrityError("Classification Train labels are not binary")
            if prediction_reconstructors and report["card_id"] in prediction_reconstructors:
                rebuilt_prediction, rebuilt_baseline = prediction_reconstructors[report["card_id"]](arrays)
                if (not np.array_equal(pred, np.asarray(rebuilt_prediction, dtype=np.float64))
                        or not np.array_equal(baseline_pred, np.asarray(rebuilt_baseline, dtype=np.float64))):
                    raise BatchIntegrityError(f"Independent route/policy reconstruction differs at cutoff {cutoff}")
            mask = (cycles[rows] >= low) & (cycles[rows] <= high)
            selected = rows[mask]
            rows_seen.append(selected)
            cutoffs_seen.append(np.full(len(selected), int(cutoff), dtype=np.int64))
            ys.append(y[selected])
            ps.append(pred[mask])
            bs.append(baseline_pred[mask])
        row = np.concatenate(rows_seen)
        if len(np.unique(row)) != len(row):
            raise BatchIntegrityError(f"Duplicate rows in verifier view {name}")
        yy, pp, bb = np.concatenate(ys), np.concatenate(ps), np.concatenate(bs)
        if task == "classification":
            candidate = _macro_f1_independent(yy, pp >= 0.5)
            baseline = _macro_f1_independent(yy, bb >= 0.5)
            gain = candidate - baseline
            metric_name = "macro_f1"
        elif task == "regression":
            candidate = _mae_independent(yy, pp)
            baseline = _mae_independent(yy, bb)
            gain = baseline - candidate
            metric_name = "mae"
        else:
            raise BatchIntegrityError(f"Unsupported verifier task: {task}")
        declared = report["views"][name]
        if declared["rows"] != len(row):
            raise BatchIntegrityError(f"Report denominator mismatch in {name}")
        declared_candidate = float(declared["candidate"][metric_name])
        declared_baseline = float(declared["baseline"][metric_name])
        stored_gain = float(declared["gain_positive_is_better"])
        if (not np.isclose(candidate, declared_candidate, atol=1e-12, rtol=0)
                or not np.isclose(baseline, declared_baseline, atol=1e-12, rtol=0)
                or not np.isclose(gain, stored_gain, atol=1e-12, rtol=0)):
            raise BatchIntegrityError(f"Independent metric differs for {name}: candidate={candidate}, baseline={baseline}, gain={gain}")
        verified[name] = {"n": int(len(row)), "metric": metric_name, "candidate": candidate, "baseline": baseline, "gain_positive_is_better": float(gain)}
        if task == "classification":
            verified[name]["subgroup_diagnostics"] = _classification_group_metrics(
                frame, yy, pp, bb, row, np.concatenate(cutoffs_seen))
    return {
        "status": "PASS", "card_id": report["card_id"],
        "freeze_signature": freeze_signature, "views": verified,
        "independent_metric_implementation": "scripts/optimization_v3_batch1_verify.py",
    }


__all__ = ["reconstruct_c03", "verify_report"]

\n