"""Independent, zero-fit verification for the measured Classification benchmark."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import optimization_v3_research_next_n00 as n00  # noqa: E402
from scripts import optimization_v3_research_next_n01 as n01  # noqa: E402
from scripts import optimization_v3_research_next_n02 as n02  # noqa: E402
from scripts import optimization_v3_measured_classification as measured  # noqa: E402
from src.models.research_next_budget import ledger_summary, read_events  # noqa: E402
from src.models.research_next_contract import canonical_sha256, file_sha256, read_bundle  # noqa: E402
from src.models.round1_models import _number_array  # noqa: E402


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _metric(y: np.ndarray, probability: np.ndarray) -> dict[str, float | int]:
    labels = (np.asarray(probability, dtype=np.float64) >= 0.50).astype(np.int8)
    target = np.asarray(y, dtype=np.int8)
    _require(target.shape == labels.shape and np.isin(target, (0, 1)).all(), "metric inputs misalign")
    tn = int(np.sum((target == 0) & (labels == 0)))
    fp = int(np.sum((target == 0) & (labels == 1)))
    fn = int(np.sum((target == 1) & (labels == 0)))
    tp = int(np.sum((target == 1) & (labels == 1)))
    denom0, denom1 = 2 * tn + fp + fn, 2 * tp + fp + fn
    f10 = 2 * tn / denom0 if denom0 else 0.0
    f11 = 2 * tp / denom1 if denom1 else 0.0
    return {"n": len(target), "macro_f1": 0.5 * (f10 + f11),
            "positive_recall": tp / (tp + fn) if tp + fn else 0.0,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def _primary_records(context: dict[str, Any], cutoff: int) -> dict[str, np.ndarray]:
    frame = context["frame"]
    rows = np.asarray(context["cycles"] > cutoff, dtype=bool)
    rows &= np.asarray(context["cycles"] <= min(cutoff + 4, 79), dtype=bool)
    score_rows = np.flatnonzero(rows).astype(np.int64)
    if cutoff in (55, 67):
        original = context["n02_parent"]
        sk = n02._load_existing_fit(original, cutoff, "sketch")
        ns = n02._load_existing_fit(original, cutoff, "no_sketch")
        _require(sk is not None and ns is not None, f"missing immutable N02 pilot bundle at {cutoff}")
        sketch, sketch_receipt = sk
        no_sketch, no_sketch_receipt = ns
        _require(sketch_receipt["receipt_sha256"] and no_sketch_receipt["receipt_sha256"], "pilot receipts missing")
        mask = np.isin(sketch["score_rows"], score_rows)
        _require(np.array_equal(sketch["score_rows"][mask], score_rows), f"pilot {cutoff} primary window incomplete")
        current = sketch["baseline_prediction"][mask].astype(np.float64)
        missing = sketch["stream_missing"][mask].astype(bool)
        cb = sketch["control__cb_probability"][mask].astype(np.float64)
        raw_candidate = sketch["raw_prediction"][mask].astype(np.float64)
        raw_matched = no_sketch["raw_prediction"][mask].astype(np.float64)
        candidate = np.where(missing, cb, 0.75 * current + 0.25 * raw_candidate)
        matched = np.where(missing, cb, 0.75 * current + 0.25 * raw_matched)
        _require(np.allclose(candidate, sketch["prediction"][mask], rtol=0, atol=1e-12), "pilot candidate route mismatch")
        _require(np.allclose(matched, no_sketch["prediction"][mask], rtol=0, atol=1e-12), "pilot matched route mismatch")
        rec_rows = sketch["score_rows"][mask].astype(np.int64)
        cycles = sketch["cycle_code"][mask].astype(np.int64)
        ids = sketch["score_record_ids"][mask].astype(str)
    else:
        identity = measured._bundle_identity(
            json.loads(measured.FREEZE_PATH.read_text(encoding="utf-8"))["freeze_signature"], cutoff, "sketch")
        sketch_bundle = read_bundle(measured._bundle_path(cutoff, "sketch"), identity, task="classification")
        identity_no = measured._bundle_identity(
            json.loads(measured.FREEZE_PATH.read_text(encoding="utf-8"))["freeze_signature"], cutoff, "no_sketch")
        no_bundle = read_bundle(measured._bundle_path(cutoff, "no_sketch"), identity_no, task="classification")
        sketch, no_sketch = sketch_bundle.arrays, no_bundle.arrays
        rec_rows = sketch["score_rows"].astype(np.int64)
        cycles = sketch["cycle_code"].astype(np.int64)
        ids = sketch["score_record_ids"].astype(str)
        _require(np.array_equal(rec_rows, score_rows), f"measured {cutoff} Train rows differ from exact future window")
        _require(np.array_equal(rec_rows, no_sketch["score_rows"]), "matched fit rows differ")
        _require(np.array_equal(ids, no_sketch["score_record_ids"].astype(str)), "matched IDs differ")
        current = sketch["control__current_route"].astype(np.float64)
        cb = sketch["control__cb_probability"].astype(np.float64)
        missing = sketch["stream_missing"].astype(bool)
        raw_candidate = sketch["raw_prediction"].astype(np.float64)
        raw_matched = no_sketch["raw_prediction"].astype(np.float64)
        candidate = np.where(missing, cb, 0.75 * current + 0.25 * raw_candidate)
        matched = np.where(missing, cb, 0.75 * current + 0.25 * raw_matched)
        _require(np.allclose(candidate, sketch["prediction"], rtol=0, atol=1e-12), "measured candidate route mismatch")
        _require(np.allclose(matched, no_sketch["prediction"], rtol=0, atol=1e-12), "measured matched route mismatch")
        _require(np.allclose(current, sketch["baseline_prediction"], rtol=0, atol=1e-12), "measured baseline differs")
    _require(np.array_equal(ids, frame.record_id.astype(str).to_numpy()[rec_rows]), f"{cutoff} source IDs misaligned")
    _require(np.array_equal(cycles, context["cycles"][rec_rows]), f"{cutoff} cycle values misaligned")
    return {"rows": rec_rows, "record_ids": ids, "cycles": cycles,
            "origin": np.full(len(rec_rows), cutoff, dtype=np.int64), "candidate": candidate,
            "current": current, "matched": matched, "missing": missing}


def _h12_pilot_record(context: dict[str, Any], cutoff: int) -> dict[str, np.ndarray]:
    original = context["n02_parent"]
    sketch_pair = n02._load_existing_fit(original, cutoff, "sketch")
    matched_pair = n02._load_existing_fit(original, cutoff, "no_sketch")
    _require(sketch_pair is not None and matched_pair is not None, "N02 pilot H12 bundle missing")
    sketch, _ = sketch_pair
    no_sketch, _ = matched_pair
    for key in ("score_rows", "score_record_ids", "cycle_code", "stream_missing",
                "control__current_route", "control__cb_probability"):
        _require(np.array_equal(sketch[key], no_sketch[key]), f"pilot H12 component alignment mismatch: {key}")
    missing = sketch["stream_missing"].astype(bool)
    current = sketch["baseline_prediction"].astype(np.float64)
    cb = sketch["control__cb_probability"].astype(np.float64)
    candidate = np.where(missing, cb, 0.75 * current + 0.25 * sketch["raw_prediction"])
    matched = np.where(missing, cb, 0.75 * current + 0.25 * no_sketch["raw_prediction"])
    _require(np.allclose(candidate, sketch["prediction"], rtol=0, atol=1e-12), "pilot H12 candidate formula mismatch")
    _require(np.allclose(matched, no_sketch["prediction"], rtol=0, atol=1e-12), "pilot H12 matched formula mismatch")
    rows = sketch["score_rows"].astype(np.int64)
    return {"rows": rows, "record_ids": sketch["score_record_ids"].astype(str),
            "cycles": sketch["cycle_code"].astype(np.int64),
            "origin": np.full(len(rows), cutoff, dtype=np.int64), "candidate": candidate,
            "current": current, "matched": matched, "missing": missing}


def _collect(records: dict[int, dict[str, np.ndarray]], origins: tuple[int, ...],
             cycle_ranges: dict[int, tuple[int, int]]) -> dict[str, np.ndarray]:
    parts = []
    for cutoff in origins:
        item = records[cutoff]
        lo, hi = cycle_ranges[cutoff]
        mask = (item["cycles"] >= lo) & (item["cycles"] <= hi)
        parts.append({key: np.asarray(value)[mask] for key, value in item.items()})
    out = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    _require(len(np.unique(out["rows"])) == len(out["rows"]), "verification view contains overlapping Train rows")
    order = np.argsort(out["rows"], kind="mergesort")
    return {key: value[order] for key, value in out.items()}


def _group_rows(context: dict[str, Any], view_records: dict[str, dict[str, np.ndarray]]) -> list[dict[str, Any]]:
    frame, target = context["frame"], context["target"]
    report: list[dict[str, Any]] = []
    for view_name, data in view_records.items():
        for cutoff in np.unique(data["origin"]):
            select_origin = data["origin"] == cutoff
            row_ids = data["rows"][select_origin].astype(np.int64)
            y = target[row_ids]
            local = frame.iloc[row_ids]
            parsed, _is_missing, _parse_fail = _number_array(frame["text_load"])
            train_values = parsed[context["cycles"] <= cutoff]
            train_values = train_values[np.isfinite(train_values)]
            edges = np.unique(np.quantile(train_values, [0.25, 0.50, 0.75])) if len(train_values) else np.array([])
            bins = np.full(len(frame), -1, dtype=np.int64)
            valid = np.isfinite(parsed)
            if len(edges):
                bins[valid] = np.searchsorted(edges, parsed[valid], side="right").astype(np.int64)
            groups = {
                "entry_friction": local.entry_friction.astype(str).to_numpy(),
                "stream_availability": np.where(data["missing"][select_origin], "missing", "present"),
                "fit_defined_text_load_quartile": bins[row_ids].astype(str),
            }
            for field, values in groups.items():
                for label in sorted(np.unique(values)):
                    mask = values == label
                    gy = y[mask]
                    cand, base, matched = (_metric(gy, data[key][select_origin][mask])
                                           for key in ("candidate", "current", "matched"))
                    n = len(gy)
                    support = n >= measured.MIN_GROUP_N and min(int(np.sum(gy == 0)), int(np.sum(gy == 1))) >= measured.MIN_EACH_CLASS_N
                    f1_harm = base["macro_f1"] - cand["macro_f1"]
                    recall_harm = base["positive_recall"] - cand["positive_recall"]
                    matched_f1_harm = base["macro_f1"] - matched["macro_f1"]
                    matched_recall_harm = base["positive_recall"] - matched["positive_recall"]
                    report.append({
                        "view": view_name, "origin": int(cutoff), "field": field, "group": str(label),
                        "n": n, "class_0_n": int(np.sum(gy == 0)), "class_1_n": int(np.sum(gy == 1)),
                        "supported": bool(support), "candidate_macro_f1": cand["macro_f1"],
                        "current_macro_f1": base["macro_f1"], "candidate_f1_harm_vs_current": f1_harm,
                        "candidate_positive_recall_harm_vs_current": recall_harm,
                        "no_sketch_macro_f1": matched["macro_f1"],
                        "no_sketch_positive_recall": matched["positive_recall"],
                        "no_sketch_f1_harm_vs_current": matched_f1_harm,
                        "no_sketch_positive_recall_harm_vs_current": matched_recall_harm,
                        "original_0_02_recall_guard_fail": bool(support and recall_harm > 0.02),
                        "benchmark_0_03_recall_guard_fail": bool(support and recall_harm > 0.03),
                        "f1_guard_fail": bool(support and f1_harm > 0.01),
                        "fit_only_text_load_edges": edges.tolist() if field == "fit_defined_text_load_quartile" else None,
                    })
    return report


def run() -> dict[str, Any]:
    frozen = json.loads(measured.FREEZE_PATH.read_text(encoding="utf-8"))
    identity = frozen["identity"]
    _require(frozen.get("freeze_signature") == canonical_sha256(identity), "measured freeze signature invalid")
    for category in ("source_sha256", "input_sha256"):
        for rel, sha in identity[category].items():
            _require(file_sha256(ROOT / rel) == sha, f"frozen {category} hash mismatch: {rel}")
    context = measured._context()
    measured._verify_freeze(context)
    report_path = measured.REPORT_ROOT / "N02_CONTINUATION_RESULTS.json"
    stored = json.loads(report_path.read_text(encoding="utf-8"))
    _require(stored.get("freeze_signature") == frozen["freeze_signature"], "result/freeze identity differs")

    primary = {cutoff: _primary_records(context, cutoff) for cutoff in measured.ORIGINS}
    h12 = {55: _h12_pilot_record(context, 55), 67: _h12_pilot_record(context, 67)}
    primary_ranges = {55: (56, 59), 59: (60, 63), 63: (64, 67), 67: (68, 71), 71: (72, 75), 75: (76, 79)}
    view_records: dict[str, dict[str, np.ndarray]] = {}
    for i, cutoff in enumerate(measured.ORIGINS, start=1):
        view_records[f"W{i}"] = _collect({cutoff: primary[cutoff]}, (cutoff,), {cutoff: primary_ranges[cutoff]})
    view_records["PRIMARY_FULL_SIX"] = _collect(primary, measured.ORIGINS, primary_ranges)
    view_records["LATE_W5_W6"] = _collect(primary, (71, 75), primary_ranges)
    for cutoff in (55, 67):
        view_records[f"H12_CUTOFF_{cutoff}"] = _collect(
            {cutoff: h12[cutoff]}, (cutoff,), {cutoff: (cutoff + 1, cutoff + 12)})
    view_records["H12_POOLED"] = _collect(h12, (55, 67), {55: (56, 67), 67: (68, 79)})

    metrics: dict[str, Any] = {}
    for name, data in view_records.items():
        y = context["target"][data["rows"]]
        candidate, base, matched = (_metric(y, data[key]) for key in ("candidate", "current", "matched"))
        metrics[name] = {"n": int(len(y)), "candidate": candidate, "sparse25_current": base,
                         "no_sketch_matched_control": matched,
                         "gain_vs_sparse25": candidate["macro_f1"] - base["macro_f1"],
                         "gain_vs_no_sketch": candidate["macro_f1"] - matched["macro_f1"]}
    group_rows = _group_rows(context, view_records)
    _require(len(group_rows) == len(stored["groups"]), "all-view group row count differs from stored report")
    stored_by_key = {(g["view"], g["origin"], g["field"], g["group"]): g for g in stored["groups"]}
    for group in group_rows:
        key = (group["view"], group["origin"], group["field"], group["group"])
        prior = stored_by_key.get(key)
        _require(prior is not None, f"stored group row missing {key}")
        for field in ("n", "class_0_n", "class_1_n", "supported", "original_0_02_recall_guard_fail",
                      "benchmark_0_03_recall_guard_fail", "f1_guard_fail"):
            _require(prior[field] == group[field], f"group {key} differs at {field}")
        for field in ("candidate_macro_f1", "current_macro_f1", "candidate_f1_harm_vs_current",
                      "candidate_positive_recall_harm_vs_current", "no_sketch_macro_f1",
                      "no_sketch_positive_recall", "no_sketch_f1_harm_vs_current",
                      "no_sketch_positive_recall_harm_vs_current"):
            _require(np.isclose(float(prior[field]), float(group[field]), rtol=0, atol=1e-12),
                     f"group {key} metric differs at {field}")

    for view, values in metrics.items():
        prior = stored["metrics"].get(view)
        _require(prior is not None, f"stored metric view missing: {view}")
        _require(prior["n"] == values["n"], f"view denominator differs: {view}")
        for arm in ("candidate", "sparse25_current", "no_sketch_matched_control"):
            _require(prior[arm] == values[arm], f"confusion-count metric differs: {view}/{arm}")
        for field in ("gain_vs_sparse25", "gain_vs_no_sketch"):
            _require(np.isclose(float(prior[field]), float(values[field]), rtol=0, atol=1e-12),
                     f"view gain differs: {view}/{field}")

    supported = [g for g in group_rows if g["supported"]]
    gates = {
        "primary_gain_vs_sparse25_ge_0_0002": metrics["PRIMARY_FULL_SIX"]["gain_vs_sparse25"] >= 0.0002,
        "pooled_h12_harm_vs_sparse25_le_0_001": metrics["H12_POOLED"]["gain_vs_sparse25"] >= -0.001,
        "late_w5_w6_harm_vs_sparse25_le_0_001": metrics["LATE_W5_W6"]["gain_vs_sparse25"] >= -0.001,
        "w6_harm_vs_sparse25_le_0_002": metrics["W6"]["gain_vs_sparse25"] >= -0.002,
        "at_least_3_of_6_windows_nonnegative": sum(metrics[f"W{i}"]["gain_vs_sparse25"] >= 0 for i in range(1, 7)) >= 3,
        "primary_gain_vs_no_sketch_nonnegative": metrics["PRIMARY_FULL_SIX"]["gain_vs_no_sketch"] >= 0,
        "all_supported_group_f1_harm_le_0_01_and_recall_harm_le_0_03": all(
            not (g["f1_guard_fail"] or g["benchmark_0_03_recall_guard_fail"]) for g in supported),
    }
    _require(gates == stored["gates"], "independent benchmark gate decisions differ")
    events = read_events(measured.LEDGER)
    fit_events = [e for e in events if e.get("card_id") == "MEASURED_N02" and e.get("event") == "FIT_STARTED"]
    saved_events = [e for e in events if e.get("card_id") == "MEASURED_N02" and e.get("event") == "FIT_SAVED"]
    _require(len(fit_events) == len(saved_events) == 8, "measured N02 fit ledger does not contain exactly 8 saved fits")
    _require(not [e for e in events if e.get("card_id") == "MEASURED_N02" and e.get("event") == "FIT_FAILED"],
             "measured N02 ledger contains failed/interrupted fits")
    old_guard_violations = [g for g in supported if g["original_0_02_recall_guard_fail"]]
    verification = {
        "schema_version": 1, "status": "PASS_INDEPENDENT_ZERO_FIT_VERIFICATION",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_signature": frozen["freeze_signature"], "independent_reconstruction": True,
        "official_fits": 0, "verified_official_fit_bundles": 8,
        "verified_groups": len(group_rows), "supported_groups": len(supported),
        "original_0_02_recall_violations": len(old_guard_violations),
        "original_guard_violations_preserved": True,
        "metrics": metrics, "gates": gates,
        "decision": "BENCHMARK_GATES_PASS" if all(gates.values()) else "BENCHMARK_GATES_FAIL",
        "ledger": ledger_summary(measured.LEDGER),
        "test_data_used": False, "sample_submission_used": False, "kaggle_called": False,
    }
    measured._write_once(measured.REPORT_ROOT / "N02_CONTINUATION_VERIFICATION.json", verification)
    return verification


if __name__ == "__main__":
    result = run()
    print(json.dumps({"status": result["status"], "decision": result["decision"],
                      "gates": result["gates"], "supported_groups": result["supported_groups"],
                      "original_guard_violations": result["original_0_02_recall_violations"]}, sort_keys=True))
