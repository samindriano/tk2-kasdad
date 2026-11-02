"""Outcome-aware reporting correction for B02's historical new-origin cohort.

Does not select K, change predictions, fit a model, or expand the evaluated
rows. It reuses B02's frozen inputs and stored per-batch K values to align one
aggregate view to the already-established W2/W3/W5 definition.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from scripts import optimization_v3_b02_plugin_allocation as b02
ROOT = Path(__file__).resolve().parents[1]
SOURCE_RESULTS = ROOT / "reports/optimization_v3/b02_plugin_allocation_v1/B02_OUTPUTS/RESULTS.json"
SOURCE_FREEZE = ROOT / "reports/optimization_v3/b02_plugin_allocation_v1/FREEZE.json"
REPORT = ROOT / "reports/optimization_v3/b02_plugin_allocation_v1/VIEW_RECONCILIATION"
STAGING = REPORT.with_name(".VIEW_RECONCILIATION.staging")
LEGACY_NEW_ORIGIN_ORIGINS = (59, 63, 71)
LEGACY_NEW_ORIGIN_CYCLES = tuple(range(60, 68)) + tuple(range(72, 76))
EXPECTED_NEW_ORIGIN_ROWS = 21_033
SOURCES = b02.SOURCES


def _metric(y: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.int8)
    prediction = np.asarray(prediction, dtype=np.int8)
    cm = confusion_matrix(y, prediction, labels=[0, 1])
    class_f1 = f1_score(y, prediction, labels=[0, 1], average=None, zero_division=0)
    return {
        "n": int(len(y)),
        "class_0": int(np.sum(y == 0)),
        "class_1": int(np.sum(y == 1)),
        "confusion": {
            "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
        },
        "f1_class_0": float(class_f1[0]),
        "f1_class_1": float(class_f1[1]),
        "macro_f1": float(f1_score(y, prediction, labels=[0, 1], average="macro", zero_division=0)),
        "positive_precision": float(precision_score(y, prediction, pos_label=1, zero_division=0)),
        "positive_recall": float(recall_score(y, prediction, pos_label=1, zero_division=0)),
    }


def _prediction_at_stored_k(probability: np.ndarray, missing: np.ndarray,
                            ids: np.ndarray, k: int) -> tuple[np.ndarray, float | None, float | None]:
    p = np.asarray(probability, dtype=np.float64)
    missing = np.asarray(missing, dtype=bool)
    ids = np.asarray(ids).astype(str)
    present = np.flatnonzero(~missing)
    if not 0 <= k <= len(present):
        raise ValueError("Stored B02 K falls outside the present-stream support")
    order = present[np.lexsort((ids[present], -p[present]))]
    floor = None if k == 0 else float(p[order[k - 1]])
    excluded = None if k == len(order) else float(p[order[k]])
    if k and k < len(order) and not floor > excluded:
        raise RuntimeError("Stored B02 K splits an exact probability tie")
    prediction = (p >= b02.THRESHOLD).astype(np.int8)
    prediction[present] = 0
    if k:
        prediction[order[:k]] = 1
    return prediction, floor, excluded


def _sort_scope_arrays(rows: np.ndarray, ids: np.ndarray, cycles: np.ndarray,
                       predictions: dict[str, dict[str, np.ndarray]]) -> tuple[
                           np.ndarray, np.ndarray, np.ndarray, dict[str, dict[str, np.ndarray]]]:
    order = np.argsort(rows, kind="stable")
    return (np.asarray(rows)[order], np.asarray(ids)[order], np.asarray(cycles)[order],
            {source: {policy: np.asarray(values)[order] for policy, values in arms.items()}
             for source, arms in predictions.items()})


def _verify_old_view_from_origin_confusions(source_results: dict[str, Any]) -> dict[str, Any]:
    old_view = source_results["scope_metrics"]["new_origin_aggregate"]
    reconstructed: dict[str, Any] = {"rows": 0, "arms": {}}
    origins = (67, 71, 75)
    for source in SOURCES:
        reconstructed["arms"][source] = {}
        for policy in ("fixed_half", "plugin"):
            cm = {key: 0 for key in ("tn", "fp", "fn", "tp")}
            n = 0
            class0 = class1 = 0
            for origin in origins:
                metric = source_results["scope_metrics"][f"E{origin}_first4"]["arms"][source][policy]
                n += metric["n"]
                class0 += metric["class_0"]
                class1 += metric["class_1"]
                for key in cm:
                    cm[key] += metric["confusion"][key]
            f0 = 0.0 if 2 * cm["tn"] + cm["fp"] + cm["fn"] == 0 else (
                2 * cm["tn"] / (2 * cm["tn"] + cm["fp"] + cm["fn"]))
            f1 = 0.0 if 2 * cm["tp"] + cm["fp"] + cm["fn"] == 0 else (
                2 * cm["tp"] / (2 * cm["tp"] + cm["fp"] + cm["fn"]))
            record = {"n": n, "class_0": class0, "class_1": class1,
                      "confusion": cm, "macro_f1": 0.5 * (f0 + f1)}
            reconstructed["arms"][source][policy] = record
            claimed = old_view["arms"][source][policy]
            if (claimed["n"] != n or claimed["confusion"] != cm
                    or not np.isclose(claimed["macro_f1"], record["macro_f1"], rtol=0, atol=1e-15)):
                raise RuntimeError("Original B02 cycle68-79 aggregate does not reconstruct from E67/E71/E75")
        reconstructed["rows"] = reconstructed["arms"][source]["plugin"]["n"]
    return reconstructed


def _build_corrected_metrics(data: dict[str, Any], source_results: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    selection = {
        (row["batch"], row["source"]): row
        for row in source_results["selection"]
    }
    rows_parts: list[np.ndarray] = []
    ids_parts: list[np.ndarray] = []
    cycles_parts: list[np.ndarray] = []
    plugin_parts: dict[str, list[np.ndarray]] = {source: [] for source in SOURCES}
    fixed_parts: dict[str, list[np.ndarray]] = {source: [] for source in SOURCES}
    k_checks = []
    for origin in LEGACY_NEW_ORIGIN_ORIGINS:
        batch_name = f"E{origin}_first4"
        record = data["origins"][origin]
        low, high = b02.FIRST_FOUR[origin]
        mask = (record["cycles"] >= low) & (record["cycles"] <= high)
        rows = record["rows"][mask].astype(np.int64)
        ids = record["ids"][mask].astype(str)
        cycles = record["cycles"][mask].astype(np.int16)
        rows_parts.append(rows)
        ids_parts.append(ids)
        cycles_parts.append(cycles)
        for source in SOURCES:
            p = record["probabilities"][source][mask].astype(np.float64)
            missing = record["missing"][mask].astype(bool)
            prior = selection[(batch_name, source)]
            plugin, floor, excluded = _prediction_at_stored_k(p, missing, ids, int(prior["k_present"]))
            if (b02._array_hash(p, "<f8") != prior["probability_sha256"]
                    or b02._array_hash(missing, np.bool_) != prior["missing_mask_sha256"]
                    or b02._array_hash(ids) != prior["record_ids_sha256"]
                    or b02._array_hash(plugin, np.int8) != prior["prediction_sha256"]
                    or floor != prior["selected_score_floor"]
                    or excluded != prior["highest_excluded_score"]):
                raise RuntimeError(f"Stored B02 K/decision identity differs at {batch_name}/{source}")
            fixed = (p >= b02.THRESHOLD).astype(np.int8)
            if not np.array_equal(plugin[missing], fixed[missing]):
                raise RuntimeError("Reconstructed K changed missing-stream labels")
            plugin_parts[source].append(plugin)
            fixed_parts[source].append(fixed)
            k_checks.append({"batch": batch_name, "source": source,
                             "k_present": int(prior["k_present"]),
                             "prediction_sha256_match": True,
                             "tie_boundary_match": True})

    rows = np.concatenate(rows_parts).astype(np.int64)
    ids = np.concatenate(ids_parts).astype(str)
    cycles = np.concatenate(cycles_parts).astype(np.int16)
    predictions = {
        source: {
            "fixed_half": np.concatenate(fixed_parts[source]).astype(np.int8),
            "plugin": np.concatenate(plugin_parts[source]).astype(np.int8),
        }
        for source in SOURCES
    }
    rows, ids, cycles, predictions = _sort_scope_arrays(rows, ids, cycles, predictions)
    if (len(np.unique(rows)) != len(rows) or len(rows) != EXPECTED_NEW_ORIGIN_ROWS
            or tuple(sorted(np.unique(cycles).astype(int))) != LEGACY_NEW_ORIGIN_CYCLES
            or not np.array_equal(rows, np.sort(rows, kind="stable"))
            or not np.array_equal(ids, data["train"].record_id.to_numpy(dtype=str)[rows])
            or not np.array_equal(cycles, data["train"].cycle_code.to_numpy(dtype=np.int16)[rows])):
        raise RuntimeError("Corrected legacy new-origin row membership/coverage/ID/cycle checks failed")

    # Prediction arrays and all K values are fixed before this selected Train-label read.
    y = b02._selected_train_labels(rows, ids, data["train"])
    groups = b02._query_group_values(data, rows)
    metrics = {
        source: {policy: _metric(y, pred) for policy, pred in arms.items()}
        for source, arms in predictions.items()
    }
    corrected_confusions_match_origin_rows = True
    for source in SOURCES:
        for policy in ("fixed_half", "plugin"):
            expected_cm = {key: 0 for key in ("tn", "fp", "fn", "tp")}
            for origin in LEGACY_NEW_ORIGIN_ORIGINS:
                prior_metric = source_results["scope_metrics"][f"E{origin}_first4"]["arms"][source][policy]
                for key in expected_cm:
                    expected_cm[key] += prior_metric["confusion"][key]
            if metrics[source][policy]["confusion"] != expected_cm:
                corrected_confusions_match_origin_rows = False
    subgroup_metrics = []
    dimensions = {
        "stream_availability": np.where(groups["missing"], "missing", "present"),
        "entry_friction": groups["entry_friction"].astype(str),
        "fit_prefix_text_load_quartile": np.asarray([
            f"Q{int(value)+1}" for value in groups["text_load_group_primary"]]),
    }
    categories = {
        "stream_availability": ("missing", "present"),
        "entry_friction": ("0", "1"),
        "fit_prefix_text_load_quartile": ("Q1", "Q2", "Q3", "Q4"),
    }
    for dimension, values in dimensions.items():
        for category in categories[dimension]:
            mask = values == category
            subgroup_y = y[mask]
            class0, class1 = int(np.sum(subgroup_y == 0)), int(np.sum(subgroup_y == 1))
            fixed = _metric(subgroup_y, predictions["B01"]["fixed_half"][mask])
            plugin = _metric(subgroup_y, predictions["B01"]["plugin"][mask])
            subgroup_metrics.append({
                "dimension": dimension, "category": category, "n": int(len(subgroup_y)),
                "class_0": class0, "class_1": class1,
                "supported": len(subgroup_y) >= 200 and class0 >= 30 and class1 >= 30,
                "b01_fixed_half": fixed, "b01_plugin": plugin,
                "b01_macro_f1_delta_vs_fixed": plugin["macro_f1"] - fixed["macro_f1"],
                "b01_positive_recall_delta_vs_fixed": plugin["positive_recall"] - fixed["positive_recall"],
            })
    return {"rows": len(rows), "origins": list(LEGACY_NEW_ORIGIN_ORIGINS),
            "cycles": [int(c) for c in sorted(np.unique(cycles))],
            "row_positions_sha256": b02._array_hash(rows, "<i8"),
            "record_ids_sha256": b02._array_hash(ids),
            "query_target_sha256": b02._array_hash(y, np.int8),
            "source_metrics": metrics, "b01_subgroups": subgroup_metrics,
            "confusions_match_sum_of_selected_origin_rows": corrected_confusions_match_origin_rows}, {
                "stored_selection_checks": k_checks,
                "target_rows_loaded_after_predictions": True,
            }


def _write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"Immutable reconciliation output differs: {path}")
        return
    staging = path.with_name(f".{path.name}.staging")
    if staging.exists():
        raise RuntimeError(f"Interrupted reconciliation staging exists: {staging}")
    with staging.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    staging.rename(path)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True,
                       allow_nan=False) + "\n").encode("utf-8")


def run() -> dict[str, Any]:
    if REPORT.exists() or STAGING.exists():
        raise RuntimeError("Reconciliation namespace already exists; inspect before rerun")
    prior = json.loads(SOURCE_RESULTS.read_text(encoding="utf-8"))
    freeze, data = b02._verify_freeze()
    corrected, audit = _build_corrected_metrics(data, prior)
    old_view = _verify_old_view_from_origin_confusions(prior)
    corrected_b01 = corrected["source_metrics"]["B01"]
    new_origin_gain = (corrected_b01["plugin"]["macro_f1"]
                       - corrected_b01["fixed_half"]["macro_f1"])
    original_gates = prior["eligibility_for_further_review_only"]["gates"]
    gates = dict(original_gates)
    gates["new_origin_gain_vs_unchanged_B01_ge_0_0005"] = new_origin_gain >= 0.0005
    overall = all(gates.values())
    ledger_after = b02.b01._verify_ledger()
    if ledger_after != freeze["fit_ledger_before"]:
        raise RuntimeError("B02 correction unexpectedly changed the fit ledger")
    failed_subgroups = [row for row in corrected["b01_subgroups"] if row["supported"] and (
        row["b01_macro_f1_delta_vs_fixed"] < -0.01
        or row["b01_positive_recall_delta_vs_fixed"] < -0.03)]
    result = {
        "status": "COMPLETED_OUTCOME_AWARE_REPORTING_CORRECTION",
        "qualification": "DIAGNOSTIC_ONLY_NOT_QUALIFIED",
        "correction_reason": "The original B02 report named cycles 68-79 (E67/E71/E75 primary windows) as new_origin_aggregate. The already-established historical B01 view is W2/W3/W5: first-four windows from E59/E63/E71, cycles 60-67 and 72-75, exactly 21,033 rows.",
        "historical_view_provenance": {
            "source": "scripts/optimization_v3_a02_reliability_audit.py:build_views",
            "source_sha256": freeze["input_sha256"]["scripts/optimization_v3_a02_reliability_audit.py"],
            "legacy_definition": "make_view([first4[59], first4[63], first4[71]])",
            "legacy_report_expected_rows": 21_033,
            "not_outcome_blind": True,
        },
        "not_outcome_blind": True,
        "no_new_or_changed_K_or_prediction": True,
        "corrected_new_origin": corrected,
        "original_b02_mislabeled_view": {
            "row_count": old_view["rows"], "origins": [67, 71, 75],
            "cycles": list(range(68, 80)),
            "same_row_count_as_h12_but_different_primary_vs_h12_decisions": True,
            "b01_fixed_half": prior["scope_metrics"]["new_origin_aggregate"]["arms"]["B01"]["fixed_half"],
            "b01_plugin": prior["scope_metrics"]["new_origin_aggregate"]["arms"]["B01"]["plugin"],
            "reconstructed_from_per_origin_confusions": old_view["arms"],
        },
        "corrected_new_origin_gain_vs_unchanged_B01": new_origin_gain,
        "corrected_new_origin_gate_pass": gates["new_origin_gain_vs_unchanged_B01_ge_0_0005"],
        "full_primary_late_h12_metrics_preserved_from_original_result": {
            scope: prior["scope_metrics"][scope]
            for scope in ("full_primary", "late", "E67_H12_full12")
        },
        "selector_efficiency_note": "The original selector recomputes cumulative sums inside candidate-K loops, giving quadratic work. This correction reuses stored K values and does not rerun or optimize selection.",
        "other_original_gates_unchanged": {
            key: value for key, value in original_gates.items()
            if key != "new_origin_gain_vs_unchanged_B01_ge_0_0005"
        },
        "corrected_overall_gate_result": "ELIGIBLE_FOR_FURTHER_REVIEW_ONLY" if overall else "INELIGIBLE_NEGATIVE_DIAGNOSTIC",
        "corrected_new_origin_failing_supported_subgroups": failed_subgroups,
        "old_result_sha256": b02._sha256_file(SOURCE_RESULTS),
        "old_freeze_sha256": b02._sha256_file(SOURCE_FREEZE),
        "source_freeze_commit": freeze["source_commit"],
        "recipe_sha256": freeze["recipe_sha256"],
        "input_sha256": freeze["input_sha256"],
        "fit_ledger_before": freeze["fit_ledger_before"],
        "fit_ledger_after": ledger_after,
        "fits_added": 0,
        "test_sample_kaggle_access": False,
    }
    current = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout.strip()
    result["correction_code_commit"] = current
    qa = {
        "status": "PASS",
        "checks": {
            "original_b02_freeze_verified": True,
            "historical_view_origins_exactly_59_63_71": corrected["origins"] == [59, 63, 71],
            "historical_cycle_set_exactly_60_67_and_72_75": corrected["cycles"] == list(LEGACY_NEW_ORIGIN_CYCLES),
            "historical_view_exact_row_count_21033": corrected["rows"] == EXPECTED_NEW_ORIGIN_ROWS,
            "stored_K_tie_boundary_probability_id_and_prediction_hashes_unchanged": (
                len(audit["stored_selection_checks"]) == len(LEGACY_NEW_ORIGIN_ORIGINS) * len(SOURCES)
                and all(row["prediction_sha256_match"] and row["tie_boundary_match"]
                        for row in audit["stored_selection_checks"])),
            "actual_confusions_and_macro_f1_replayed_with_sklearn": True,
            "corrected_confusions_equal_sum_of_E59_E63_E71_confusions": corrected["confusions_match_sum_of_selected_origin_rows"],
            "original_wrong_view_reconstructed_from_E67_E71_E75_confusions": True,
            "full_primary_late_and_h12_metrics_copied_bitwise_from_original": True,
            "no_query_labels_loaded_before_all_stored_K_predictions_reconstructed": audit["target_rows_loaded_after_predictions"],
            "fit_ledger_unchanged": ledger_after == freeze["fit_ledger_before"],
            "no_fit_or_test_sample_kaggle_access": result["fits_added"] == 0 and result["test_sample_kaggle_access"] is False,
            "failed_gate_and_diagnostic_only_status_preserved": result["corrected_overall_gate_result"] == "INELIGIBLE_NEGATIVE_DIAGNOSTIC" and result["qualification"] == "DIAGNOSTIC_ONLY_NOT_QUALIFIED",
        },
    }
    if not all(qa["checks"].values()):
        raise RuntimeError(f"B02 view reconciliation QA failed: {qa['checks']}")
    summary = (
        "# B02 new-origin scope reconciliation\n\n"
        "**Status:** outcome-aware reporting correction; no new selection or experiment.\n\n"
        "The original B02 `new_origin_aggregate` incorrectly used cycles 68-79 (E67/E71/E75), n=20,903. The frozen historical B01 definition is W2/W3/W5: E59/E63/E71 first-four windows, cycles 60-67 and 72-75, n=21,033. The original B02 artifacts remain unchanged.\n\n"
        f"For B01 on the corrected cohort, fixed 0.50 Macro F1={corrected_b01['fixed_half']['macro_f1']:.6f}; plug-in Macro F1={corrected_b01['plugin']['macro_f1']:.6f}; change={new_origin_gain:+.6f}. The corrected >=0.0005 new-origin gate is {'PASS' if gates['new_origin_gain_vs_unchanged_B01_ge_0_0005'] else 'FAIL'}.\n\n"
        "The overall diagnostic remains ineligible: the corrected new-origin gain is below its gate, H12 is negative, and only 3/5 origin windows are nonnegative. No K, probability, or prediction changed. Full-primary, late, and H12 values are copied from the original B02 result. See `CORRECTION.json` for all source arms, confusion matrices, subgroups, hashes, K identity checks, and QA.\n\n"
        f"Fits added: 0. Ledger SHA256 remains `{ledger_after['sha256']}`. No Test/Sample/Kaggle access.\n"
    )
    if REPORT.exists() or STAGING.exists():
        raise RuntimeError("Reconciliation output path appeared during evaluation")
    STAGING.mkdir(parents=True)
    for name, payload in {
        "CORRECTION.json": _json_bytes(result),
        "QA.json": _json_bytes(qa),
        "SUMMARY.md": summary.encode("utf-8"),
    }.items():
        _write_once(STAGING / name, payload)
    STAGING.rename(REPORT)
    return {"status": result["status"], "overall": result["corrected_overall_gate_result"],
            "corrected_new_origin_gain": new_origin_gain, "QA": qa["status"],
            "report": REPORT.relative_to(ROOT).as_posix()}


def main() -> None:
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
