"""Train-only, zero-fit audit of an A02 missing-stream fallback for MIX50.

The candidate is fixed in advance: use the frozen MIX50 probability on
present-stream rows and the already saved original A02 probability on
missing-stream rows, with the existing >= 0.50 decision threshold.
No estimator is fit and no Test data is opened.
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import optimization_v3_cls_mix50_temporal_decision_v1 as mix  # noqa: E402

TRAIN = ROOT / "data/raw/classification/Train.csv"
OOF = ROOT / "experiments/runs/optimization_v3_posoffset_a02_endgame_v1/policy_predictions.npz"
RECIPE = ROOT / "reports/optimization_v3/posoffset_a02_endgame_v1/RECIPE.json"
POLICY_METRICS = ROOT / "reports/optimization_v3/posoffset_a02_endgame_v1/POLICY_METRICS.json"
INDEPENDENT_QA = ROOT / "reports/optimization_v3/posoffset_a02_endgame_v1/INDEPENDENT_VERIFICATION.json"
ENDGAME_FREEZE = ROOT / "reports/optimization_v3/posoffset_a02_endgame_v1/FREEZE_SUCCESSOR_01.json"
REPORT_DIR = ROOT / "reports/optimization_v3/cls_missing_current_mix50_audit_v1"

EXPECTED_TRAIN_SHA256 = "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c"
EXPECTED_OOF_SHA256 = "25d1d2b91348a57d33b3b0e5ce78da29840d115b75e24f9502cc83c1b63ef351"
THRESHOLD = 0.50
MIN_GROUP_ROWS = 200
MIN_GROUP_PER_CLASS = 30
VIEW_SPECS = {
    "E67_H12": ("67", 68, 79),
    "E67_W4": ("67", 68, 71),
    "E67_W5": ("67", 72, 75),
    "E67_W6": ("67", 76, 79),
    "E75_W6": ("75", 76, 79),
}
GROUP_NAMES = ("group_availability", "group_text_load", "group_entry_friction")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _arrays_same_length(values: Mapping[str, np.ndarray]) -> int:
    lengths = {len(v) for v in values.values()}
    if len(lengths) != 1:
        raise ValueError("Input view arrays have mismatched lengths")
    return lengths.pop()


def candidate_probability(mix50: np.ndarray, p_a02: np.ndarray, missing: np.ndarray) -> np.ndarray:
    """Fixed diagnostic policy; present probability remains the exact MIX50 value."""
    base = np.asarray(mix50, dtype=np.float64)
    p_a = np.asarray(p_a02, dtype=np.float64)
    miss = np.asarray(missing, dtype=bool)
    if base.ndim != 1 or base.shape != p_a.shape or base.shape != miss.shape:
        raise ValueError("MIX50, A02 probability, and missing mask must be aligned vectors")
    if not np.isfinite(base).all() or not np.isfinite(p_a).all():
        raise ValueError("Probabilities must be finite")
    if np.any((base < 0) | (base > 1)) or np.any((p_a < 0) | (p_a > 1)):
        raise ValueError("Probabilities must be in [0, 1]")
    return np.where(miss, p_a, base)


def _counts(y: np.ndarray, probability: np.ndarray) -> tuple[int, int, int, int]:
    target = np.asarray(y, dtype=np.int8)
    pred = np.asarray(probability, dtype=np.float64) >= THRESHOLD
    if target.ndim != 1 or target.shape != pred.shape or not np.isin(target, [0, 1]).all():
        raise ValueError("Labels and predictions are not aligned binary vectors")
    tn = int(np.sum((target == 0) & ~pred))
    fp = int(np.sum((target == 0) & pred))
    fn = int(np.sum((target == 1) & ~pred))
    tp = int(np.sum((target == 1) & pred))
    return tn, fp, fn, tp


def _metric(y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    target = np.asarray(y, dtype=np.int8)
    p = np.asarray(probability, dtype=np.float64)
    if target.shape != p.shape or target.ndim != 1:
        raise ValueError("Labels and probability arrays are not aligned")
    n = len(target)
    if n == 0:
        return {"rows": 0, "status": "EMPTY"}
    tn, fp, fn, tp = _counts(target, p)
    n0, n1 = tn + fp, fn + tp
    precision0 = tn / (tn + fn) if tn + fn else 0.0
    precision1 = tp / (tp + fp) if tp + fp else 0.0
    recall0 = tn / n0 if n0 else 0.0
    recall1 = tp / n1 if n1 else 0.0
    f10 = 2 * precision0 * recall0 / (precision0 + recall0) if precision0 + recall0 else 0.0
    f11 = 2 * precision1 * recall1 / (precision1 + recall1) if precision1 + recall1 else 0.0
    clipped = np.clip(p, 1e-15, 1 - 1e-15)
    return {
        "rows": n,
        "class_0": n0,
        "class_1": n1,
        "target_prevalence": float(n1 / n),
        "predicted_positive_rate": float(np.mean(p >= THRESHOLD)),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "false_positives": fp,
        "false_negatives": fn,
        "precision_0": float(precision0),
        "precision_1": float(precision1),
        "recall_0": float(recall0),
        "recall_1": float(recall1),
        "f1_0": float(f10),
        "f1_1": float(f11),
        "macro_f1": float((f10 + f11) / 2),
        "brier": float(np.mean((p - target) ** 2)),
        "log_loss": float(-np.mean(target * np.log(clipped) + (1 - target) * np.log(1 - clipped))),
        "status": "SUPPORTED" if n >= MIN_GROUP_ROWS and min(n0, n1) >= MIN_GROUP_PER_CLASS else "DESCRIPTIVE_UNSUPPORTED",
    }


def _read_train_labels_and_missing(
    ids: np.ndarray, expected_cycle_by_id: Mapping[str, int], *, train_sha256: str
) -> tuple[np.ndarray, np.ndarray]:
    if sha256_file(TRAIN) != train_sha256:
        raise RuntimeError("Official Classification Train hash does not match the frozen source")
    wanted = set(np.asarray(ids, dtype=str).tolist())
    found: dict[str, tuple[int, int, bool]] = {}
    with TRAIN.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        columns = {name: idx for idx, name in enumerate(header)}
        required = ("record_id", "cycle_code", "artifact_signal", "signal_stream")
        if any(name not in columns for name in required):
            raise RuntimeError("Official Train is missing required target, ID, cycle, or stream field")
        max_col = max(columns[name] for name in required)
        for row in reader:
            if len(row) <= max_col:
                raise RuntimeError("Malformed Classification Train CSV row")
            rid = row[columns["record_id"]]
            if rid not in wanted:
                continue
            if rid in found:
                raise RuntimeError(f"Duplicate Train ID in requested OOF rows: {rid}")
            cycle = int(row[columns["cycle_code"]])
            target = int(row[columns["artifact_signal"]])
            if cycle != int(expected_cycle_by_id[rid]) or target not in (0, 1):
                raise RuntimeError(f"Train target/cycle mismatch for OOF row {rid}")
            missing = row[columns["signal_stream"]].strip() == ""
            found[rid] = (cycle, target, missing)
    if set(found) != wanted:
        raise RuntimeError(f"Train did not contain all {len(wanted)} requested OOF IDs")
    return (
        np.fromiter((found[str(rid)][1] for rid in ids), dtype=np.int8, count=len(ids)),
        np.fromiter((found[str(rid)][2] for rid in ids), dtype=bool, count=len(ids)),
    )


def _view(origin: Mapping[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    out = {name: np.asarray(value)[mask] for name, value in origin.items()}
    _arrays_same_length(out)
    return out


def _join(*parts: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: np.concatenate([np.asarray(part[name]) for part in parts]) for name in parts[0]}


def _metric_record(name: str, arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    _arrays_same_length(arrays)
    y = arrays["y"]
    baseline = arrays["MIX50"]
    candidate = arrays["candidate"]
    base_metric = _metric(y, baseline)
    cand_metric = _metric(y, candidate)
    return {
        "view": name,
        "rows": len(y),
        "missing_rows": int(arrays["missing"].sum()),
        "present_rows": int((~arrays["missing"]).sum()),
        "baseline": base_metric,
        "candidate": cand_metric,
        "delta_macro_f1": float(cand_metric["macro_f1"] - base_metric["macro_f1"]),
        "delta_brier": float(cand_metric["brier"] - base_metric["brier"]),
        "delta_log_loss": float(cand_metric["log_loss"] - base_metric["log_loss"]),
    }


def _missing_pair_record(name: str, arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    miss = np.asarray(arrays["missing"], dtype=bool)
    y = np.asarray(arrays["y"])[miss]
    p_base = np.asarray(arrays["MIX50"])[miss]
    p_candidate = np.asarray(arrays["candidate"])[miss]
    pred_base = p_base >= THRESHOLD
    pred_candidate = p_candidate >= THRESHOLD
    fp_removed = int(np.sum((y == 0) & pred_base & ~pred_candidate))
    fp_added = int(np.sum((y == 0) & ~pred_base & pred_candidate))
    fn_removed = int(np.sum((y == 1) & ~pred_base & pred_candidate))
    fn_added = int(np.sum((y == 1) & pred_base & ~pred_candidate))
    return {
        "view": name,
        "missing_rows": len(y),
        "class_0": int(np.sum(y == 0)),
        "class_1": int(np.sum(y == 1)),
        "baseline_pP_exact_metric": _metric(y, p_base),
        "candidate_pA02_metric": _metric(y, p_candidate),
        "delta_macro_f1_missing_only": float(_metric(y, p_candidate)["macro_f1"] - _metric(y, p_base)["macro_f1"]) if len(y) else None,
        "baseline_minus_candidate_paired_errors": {
            "false_positives_removed": fp_removed,
            "false_positives_added": fp_added,
            "false_negatives_removed": fn_removed,
            "false_negatives_added": fn_added,
            "net_fp_change_candidate_minus_baseline": fp_added - fp_removed,
            "net_fn_change_candidate_minus_baseline": fn_added - fn_removed,
            "prediction_labels_changed": int(np.sum(pred_base != pred_candidate)),
        },
    }


def _crosscheck_baseline(name: str, record: Mapping[str, Any], frozen_metrics: Mapping[str, Any]) -> None:
    old = frozen_metrics["views"][name]["policies"]["MIX50"]
    if (record["baseline"]["rows"] != old["n"]
            or record["baseline"]["confusion_matrix"] != old["confusion_matrix"]
            or not np.isclose(record["baseline"]["macro_f1"], old["macro_f1"], atol=1e-12, rtol=0)):
        raise RuntimeError(f"Recomputed baseline MIX50 does not match frozen metrics for {name}")


def run_audit() -> dict[str, Any]:
    if not (TRAIN.exists() and OOF.exists() and RECIPE.exists() and POLICY_METRICS.exists()):
        raise FileNotFoundError("A required frozen Train-only input is missing")
    if sha256_file(TRAIN) != EXPECTED_TRAIN_SHA256:
        raise RuntimeError("Official Train differs from the endgame frozen source hash")
    oof = mix._load_oof_predictions(OOF, EXPECTED_OOF_SHA256)
    frozen_metrics = json.loads(POLICY_METRICS.read_text(encoding="utf-8"))
    recipe = json.loads(RECIPE.read_text(encoding="utf-8"))
    if frozen_metrics.get("status") != "SCORED_FROZEN_POLICIES":
        raise RuntimeError("Frozen endgame MIX50 metrics are not in the expected completed state")
    if recipe["policies"][0]["id"] != "MIX50" or recipe["policies"][0]["missing"] != "pP exactly":
        raise RuntimeError("The frozen MIX50 recipe does not match this audit's baseline")

    # Bind labels and missing availability from official Train by ID and cycle.
    expected_cycle_by_id: dict[str, int] = {}
    for origin in oof.values():
        for rid, cycle in zip(origin["ids"].tolist(), origin["cycles"].tolist()):
            if rid in expected_cycle_by_id and expected_cycle_by_id[rid] != int(cycle):
                raise RuntimeError(f"Overlapping OOF origins disagree on cycle for {rid}")
            expected_cycle_by_id[rid] = int(cycle)
    all_ids = np.asarray(list(expected_cycle_by_id), dtype=str)
    labels, train_missing = _read_train_labels_and_missing(
        all_ids, expected_cycle_by_id, train_sha256=EXPECTED_TRAIN_SHA256
    )
    label_by_id = dict(zip(all_ids.tolist(), labels.tolist()))
    missing_by_id = dict(zip(all_ids.tolist(), train_missing.tolist()))

    prepared: dict[str, dict[str, np.ndarray]] = {}
    alignment: dict[str, Any] = {}
    for origin_name, values in oof.items():
        ids = values["ids"].astype(str)
        y = np.fromiter((label_by_id[rid] for rid in ids), dtype=np.int8, count=len(ids))
        actual_missing = np.fromiter((missing_by_id[rid] for rid in ids), dtype=bool, count=len(ids))
        if not np.array_equal(actual_missing, values["missing"]):
            raise RuntimeError(f"Train signal_stream strip-empty availability differs from OOF E{origin_name}")
        expected_availability = np.where(actual_missing, "missing", "present")
        if not np.array_equal(values["group_availability"].astype(str), expected_availability):
            raise RuntimeError(f"Availability subgroup array does not align for E{origin_name}")
        candidate = candidate_probability(values["MIX50"], values["pA"], actual_missing)
        if not np.array_equal(candidate[~actual_missing], values["MIX50"][~actual_missing]):
            raise RuntimeError(f"Present-stream probabilities changed for E{origin_name}")
        if not np.array_equal(candidate[~actual_missing] >= THRESHOLD,
                              values["MIX50"][~actual_missing] >= THRESHOLD):
            raise RuntimeError(f"Present-stream predicted labels changed for E{origin_name}")
        if not np.array_equal(candidate[actual_missing], values["pA"][actual_missing]):
            raise RuntimeError(f"Missing-stream candidate is not exact original A02 for E{origin_name}")
        if not np.array_equal(values["MIX50"][actual_missing], values["pP"][actual_missing]):
            raise RuntimeError(f"Frozen MIX50 missing branch is not exact pP for E{origin_name}")
        prepared[origin_name] = {**values, "y": y, "missing": actual_missing, "candidate": candidate}
        alignment[origin_name] = {
            "rows": int(len(ids)),
            "unique_ids": int(len(np.unique(ids))),
            "train_labels_matched_by_id_cycle": True,
            "train_signal_stream_strip_empty_matches_frozen_mask": True,
            "availability_group_matches_missing_mask": True,
            "present_candidate_probability_bit_exact_to_mix50": True,
            "present_candidate_labels_bit_exact_to_mix50": True,
            "missing_candidate_probability_bit_exact_to_original_a02": True,
        }

    views: dict[str, dict[str, np.ndarray]] = {}
    for name, (origin, low, high) in VIEW_SPECS.items():
        values = prepared[origin]
        views[name] = _view(values, (values["cycles"] >= low) & (values["cycles"] <= high))
    e67 = prepared["67"]
    e75 = prepared["75"]
    views["POOLED_UNIQUE_E67_W4_W5_PLUS_E75_W6"] = _join(
        _view(e67, (e67["cycles"] >= 68) & (e67["cycles"] <= 75)),
        _view(e75, (e75["cycles"] >= 76) & (e75["cycles"] <= 79)),
    )

    view_results = {name: _metric_record(name, arrays) for name, arrays in views.items()}
    for name in VIEW_SPECS:
        _crosscheck_baseline(name, view_results[name], frozen_metrics)
    missing_results = {name: _missing_pair_record(name, arrays) for name, arrays in views.items()}

    per_cycle: list[dict[str, Any]] = []
    cycle_delta_signs: dict[str, dict[str, int]] = {}
    for origin_name, values in prepared.items():
        for cycle in sorted(np.unique(values["cycles"]).tolist()):
            arrays = _view(values, values["cycles"] == cycle)
            base = _metric(arrays["y"], arrays["MIX50"])
            cand = _metric(arrays["y"], arrays["candidate"])
            delta = float(cand["macro_f1"] - base["macro_f1"])
            sign = "positive" if delta > 1e-15 else "negative" if delta < -1e-15 else "zero"
            per_cycle.append({
                "origin": f"E{origin_name}",
                "cycle": int(cycle),
                "rows": int(len(arrays["y"])),
                "missing_rows": int(arrays["missing"].sum()),
                "baseline_macro_f1": base["macro_f1"],
                "candidate_macro_f1": cand["macro_f1"],
                "delta_macro_f1": delta,
                "direction": sign,
            })
            counter = cycle_delta_signs.setdefault(f"E{origin_name}", {"positive": 0, "negative": 0, "zero": 0})
            counter[sign] += 1

    supported_groups: list[dict[str, Any]] = []
    for view_name, arrays in views.items():
        for group_name in GROUP_NAMES:
            values = arrays[group_name].astype(str)
            for value in sorted(np.unique(values).tolist()):
                mask = values == value
                base = _metric(arrays["y"][mask], arrays["MIX50"][mask])
                cand = _metric(arrays["y"][mask], arrays["candidate"][mask])
                supported = (base["rows"] >= MIN_GROUP_ROWS
                             and min(base["class_0"], base["class_1"]) >= MIN_GROUP_PER_CLASS)
                supported_groups.append({
                    "view": view_name,
                    "group": group_name,
                    "value": value,
                    "rows": base["rows"],
                    "class_0": base["class_0"],
                    "class_1": base["class_1"],
                    "status": "SUPPORTED" if supported else "UNKNOWN_UNSUPPORTED",
                    "baseline_macro_f1": base["macro_f1"] if supported else None,
                    "candidate_macro_f1": cand["macro_f1"] if supported else None,
                    "delta_macro_f1": cand["macro_f1"] - base["macro_f1"] if supported else None,
                    "baseline_positive_recall": base["recall_1"] if supported else None,
                    "candidate_positive_recall": cand["recall_1"] if supported else None,
                    "delta_positive_recall": cand["recall_1"] - base["recall_1"] if supported else None,
                })

    input_paths = {
        "official_train": TRAIN,
        "train_oof_probability_bundle": OOF,
        "frozen_recipe": RECIPE,
        "frozen_policy_metrics": POLICY_METRICS,
        "independent_verification": INDEPENDENT_QA,
        "endgame_freeze": ENDGAME_FREEZE,
        "mix50_loader_source": ROOT / "scripts/optimization_v3_cls_mix50_temporal_decision_v1.py",
        "audit_script": Path(__file__).resolve(),
        "focused_synthetic_test": ROOT / "tests/test_optimization_v3_cls_missing_current_mix50_audit_v1.py",
    }
    hashes = {name: sha256_file(path) for name, path in input_paths.items()}
    results = {
        "schema_version": 1,
        "status": "TRAIN_ONLY_ZERO_FIT_DIAGNOSTIC",
        "audit_id": "CLS_MISSING_CURRENT_MIX50_AUDIT_V1",
        "candidate_policy": "MIX50_PRESENT_A02_MISSING",
        "candidate_formula": "present: exact MIX50; missing: exact original A02 pA",
        "baseline_formula": "MIX50: present 0.5*pP + 0.5*pA; missing exact pP",
        "threshold": ">= 0.50 fixed",
        "new_fits": 0,
        "test_accessed": False,
        "kaggle_contacted": False,
        "adaptive_development_views_not_fresh_holdouts": True,
        "source_commit_before_audit": _head(),
        "input_hashes": hashes,
        "alignment_qa": alignment,
        "views": view_results,
        "missing_only_paired_comparison": missing_results,
        "per_cycle": per_cycle,
        "per_cycle_direction_counts": cycle_delta_signs,
        "supported_group_thresholds": {"rows_min": MIN_GROUP_ROWS, "each_class_min": MIN_GROUP_PER_CLASS},
        "supported_groups": supported_groups,
        "interpretation_limit": "E67/E75 are overlapping adaptive development views; descriptive differences do not estimate future performance or establish generality.",
    }
    return results


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(results: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "RESULTS.json").write_text(
        json.dumps(results, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    view_rows = []
    for name, item in results["views"].items():
        b, c = item["baseline"], item["candidate"]
        view_rows.append({
            "view": name, "rows": item["rows"], "missing_rows": item["missing_rows"],
            "baseline_macro_f1": b["macro_f1"], "candidate_macro_f1": c["macro_f1"],
            "delta_macro_f1": item["delta_macro_f1"],
            "baseline_fp": b["false_positives"], "candidate_fp": c["false_positives"],
            "baseline_fn": b["false_negatives"], "candidate_fn": c["false_negatives"],
            "baseline_brier": b["brier"], "candidate_brier": c["brier"],
            "delta_brier": item["delta_brier"],
            "baseline_log_loss": b["log_loss"], "candidate_log_loss": c["log_loss"],
            "delta_log_loss": item["delta_log_loss"],
        })
    _write_csv(REPORT_DIR / "views.csv", view_rows)
    missing_rows = []
    for name, item in results["missing_only_paired_comparison"].items():
        b, c, paired = item["baseline_pP_exact_metric"], item["candidate_pA02_metric"], item["baseline_minus_candidate_paired_errors"]
        missing_rows.append({
            "view": name, "missing_rows": item["missing_rows"], "class_0": item["class_0"], "class_1": item["class_1"],
            "baseline_macro_f1": b["macro_f1"], "candidate_macro_f1": c["macro_f1"],
            "delta_macro_f1": item["delta_macro_f1_missing_only"],
            "baseline_brier": b["brier"], "candidate_brier": c["brier"],
            "delta_brier": c["brier"] - b["brier"],
            "baseline_log_loss": b["log_loss"], "candidate_log_loss": c["log_loss"],
            "delta_log_loss": c["log_loss"] - b["log_loss"],
            **paired,
        })
    _write_csv(REPORT_DIR / "missing_branch.csv", missing_rows)
    _write_csv(REPORT_DIR / "per_cycle.csv", results["per_cycle"])
    _write_csv(REPORT_DIR / "supported_groups.csv", results["supported_groups"])
    (REPORT_DIR / "QA.json").write_text(json.dumps({
        "status": "PASS",
        "train_sha256_matches_frozen": results["input_hashes"]["official_train"] == EXPECTED_TRAIN_SHA256,
        "oof_bundle_sha256_matches_frozen": results["input_hashes"]["train_oof_probability_bundle"] == EXPECTED_OOF_SHA256,
        "baseline_metrics_crosschecked_to_frozen_endgame": True,
        "candidate_present_probability_exact": True,
        "candidate_present_labels_exact": True,
        "candidate_missing_probability_exact_original_a02": True,
        "train_stream_missing_mask_reconciled_by_id_cycle": True,
        "test_accessed": False,
        "new_fits": 0,
    }, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    results = run_audit()
    write_outputs(results)
    print(json.dumps({
        "status": results["status"],
        "views": {name: {"rows": v["rows"], "baseline_macro_f1": v["baseline"]["macro_f1"],
                         "candidate_macro_f1": v["candidate"]["macro_f1"], "delta_macro_f1": v["delta_macro_f1"]}
                  for name, v in results["views"].items()},
        "new_fits": 0,
        "test_accessed": False,
        "output": str(REPORT_DIR),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
