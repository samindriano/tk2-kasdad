"""Independent, zero-fit verification for the N01 class-only diagnostic amendment."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from scripts import optimization_v3_research_next_n01_class_only_diagnostic as runner
from src.models.research_next_budget import ledger_summary, read_events
from src.models.research_next_contract import file_sha256, read_bundle
from src.models.round1_models import _number_array
from src.utils.paths import PROJECT_ROOT


ROOT = PROJECT_ROOT


def _hash_rows(values: Any) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def _metric(y: Any, probability: Any) -> dict[str, Any]:
    labels = np.asarray(y, dtype=np.int64)
    predicted = (np.asarray(probability, dtype=np.float64) >= 0.50).astype(np.int64)
    if labels.shape != predicted.shape or not np.isin(labels, (0, 1)).all():
        raise RuntimeError("Verifier received misaligned/nonbinary metric inputs")
    tn = int(np.sum((labels == 0) & (predicted == 0)))
    fp = int(np.sum((labels == 0) & (predicted == 1)))
    fn = int(np.sum((labels == 1) & (predicted == 0)))
    tp = int(np.sum((labels == 1) & (predicted == 1)))
    f0 = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"n": len(labels), "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "macro_f1": float((f0 + f1) / 2), "positive_recall": float(recall)}


def _assert_metric(actual: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    for key in ("n", "tn", "fp", "fn", "tp"):
        if int(actual[key]) != int(expected[key]):
            raise RuntimeError(f"{label} {key} differs: {actual[key]} != {expected[key]}")
    for key in ("macro_f1", "positive_recall"):
        if not np.isclose(float(actual[key]), float(expected[key]), rtol=0, atol=1e-14):
            raise RuntimeError(f"{label} {key} differs: {actual[key]} != {expected[key]}")


def _records(context: Mapping[str, Any], frozen: Mapping[str, Any]) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, Any]]:
    records: dict[int, dict[str, np.ndarray]] = {}
    bundles: dict[str, Any] = {}
    ledger_events = read_events(runner.LEDGER_PATH)
    for cutoff in runner.PILOT_ORIGINS:
        arrays = context["pilot_bundles"][cutoff]
        missing = arrays["stream_missing"].astype(bool)
        raw = arrays["raw_prediction"].astype(np.float64)
        c2b = arrays["control__c2b_probability"].astype(np.float64)
        cb = arrays["control__cb_probability"].astype(np.float64)
        candidate = np.where(missing, cb, 0.75 * c2b + 0.25 * raw)
        if not np.allclose(candidate, arrays["prediction"], rtol=0, atol=1e-12):
            raise RuntimeError(f"Independent pilot route reconstruction failed at {cutoff}")
        records[cutoff] = {
            "rows": arrays["score_rows"].astype(np.int64),
            "record_ids": arrays["score_record_ids"].astype(str),
            "cycles": arrays["cycle_code"].astype(np.int64),
            "origin": arrays["origin_cutoff"].astype(np.int64),
            "candidate": candidate,
            "current": arrays["baseline_prediction"].astype(np.float64),
            "missing": missing,
        }

    for cutoff in runner.CONTINUATION_CUTOFFS:
        path = runner._bundle_path(cutoff)
        expected_identity = runner._bundle_identity(frozen["freeze_signature"], cutoff)
        bundle = read_bundle(path, expected_identity, task="classification")
        arrays = bundle.arrays
        missing = arrays["stream_missing"].astype(bool)
        raw = arrays["raw_prediction"].astype(np.float64)
        c2b = arrays["control__c2b_probability"].astype(np.float64)
        cb = arrays["control__cb_probability"].astype(np.float64)
        sparse_raw = arrays["control__sparse_baseline_raw"].astype(np.float64)
        candidate = np.where(missing, cb, 0.75 * c2b + 0.25 * raw)
        baseline = np.where(missing, cb, 0.75 * c2b + 0.25 * sparse_raw)
        if (not np.allclose(candidate, arrays["prediction"], rtol=0, atol=1e-12)
                or not np.allclose(baseline, arrays["baseline_prediction"], rtol=0, atol=1e-12)
                or not np.array_equal(candidate[missing], cb[missing])):
            raise RuntimeError(f"Independent routed prediction reconstruction failed at {cutoff}")
        expected_fit = np.flatnonzero(context["cycles"] <= cutoff).astype(np.int64)
        expected_score = context["controls"][cutoff]["rows"]
        if (not np.array_equal(arrays["fit_rows"], expected_fit)
                or not np.array_equal(arrays["score_rows"], expected_score)
                or not np.array_equal(arrays["score_record_ids"].astype(str),
                                      context["frame"].iloc[expected_score]["record_id"].astype(str).to_numpy())
                or not np.array_equal(arrays["cycle_code"], context["cycles"][expected_score])):
            raise RuntimeError(f"Fitted or scored row lineage differs at cutoff {cutoff}")
        fit_id = f"{runner.CARD_ID}:{runner.STAGE}:{cutoff}:class_only"
        starts = [event for event in ledger_events if event.get("fit_id") == fit_id and event.get("event") == "FIT_STARTED"]
        saved = [event for event in ledger_events if event.get("fit_id") == fit_id and event.get("event") == "FIT_SAVED"]
        failed = [event for event in ledger_events if event.get("fit_id") == fit_id and event.get("event") == "FIT_FAILED"]
        if len(starts) != 1 or len(saved) != 1 or failed:
            raise RuntimeError(f"Ledger fit lineage is not exactly one saved fit for {fit_id}")
        if (saved[0].get("receipt_sha256") != bundle.receipt["receipt_sha256"]
                or int(arrays["origin_cutoff"][0]) != cutoff
                or not np.all(arrays["origin_cutoff"] == cutoff)):
            raise RuntimeError(f"Ledger receipt or origin metadata mismatch for {fit_id}")
        metadata = bundle.receipt.get("metadata", {}).get("metadata", {})
        fit_state = metadata.get("fit_state", {})
        weight_hash = hashlib.sha256(
            np.asarray(context["plans"][cutoff]["class_only_weights"], dtype="<f8").tobytes()
        ).hexdigest()
        if (metadata.get("weight_kind") != "class_only"
                or metadata.get("score_labels_passed_to_fit") is not False
                or metadata.get("test_data_used") is not False
                or fit_state.get("sample_weight_used") is not True
                or fit_state.get("sample_weight_sha256") != weight_hash
                or fit_state.get("fit_rows") != len(expected_fit)):
            raise RuntimeError(f"Fitted-state/preprocessing boundary audit failed at cutoff {cutoff}")
        records[cutoff] = {
            "rows": arrays["score_rows"].astype(np.int64),
            "record_ids": arrays["score_record_ids"].astype(str),
            "cycles": arrays["cycle_code"].astype(np.int64),
            "origin": arrays["origin_cutoff"].astype(np.int64),
            "candidate": candidate,
            "current": baseline,
            "missing": missing,
        }
        bundles[str(cutoff)] = {
            "fit_id": fit_id, "receipt_sha256": bundle.receipt["receipt_sha256"],
            "npz_sha256": file_sha256(path / "arrays.npz"),
            "fit_rows": len(expected_fit), "score_rows": len(expected_score),
        }
    return records, bundles


def _collect(records: Mapping[int, Mapping[str, np.ndarray]], blocks: Sequence[Sequence[int]]) -> dict[str, np.ndarray]:
    parts = []
    for cutoff, low, high in blocks:
        record = records[int(cutoff)]
        mask = (record["cycles"] >= int(low)) & (record["cycles"] <= int(high))
        parts.append({key: np.asarray(value)[mask] for key, value in record.items()})
    merged = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    order = np.argsort(merged["rows"], kind="mergesort")
    merged = {key: value[order] for key, value in merged.items()}
    if len(np.unique(merged["rows"])) != len(merged["rows"]):
        raise RuntimeError("Verifier found overlapping source rows in a reported view")
    return merged


def _verify_views(context: Mapping[str, Any], records: Mapping[int, Mapping[str, np.ndarray]], report: Mapping[str, Any]) -> dict[str, Any]:
    views = context["queue"]["views"]
    blocks_by_name = {"primary_full_six": views["full_six"], "late_primary_W5_W6": views["late_primary"]}
    for index, block in enumerate(views["full_six"], start=1):
        blocks_by_name[f"W{index}"] = [block]
    checked_views = {}
    y_all = context["target"]
    cycles_all = context["cycles"]
    for name, blocks in blocks_by_name.items():
        data = _collect(records, blocks)
        rows = data["rows"].astype(np.int64)
        if (not np.array_equal(context["frame"].iloc[rows]["record_id"].astype(str).to_numpy(), data["record_ids"])
                or not np.array_equal(cycles_all[rows], data["cycles"])):
            raise RuntimeError(f"Independent source row alignment failed for {name}")
        candidate = _metric(y_all[rows], data["candidate"])
        baseline = _metric(y_all[rows], data["current"])
        actual = report["views"][name]
        _assert_metric(actual["candidate"], candidate, f"{name} candidate")
        _assert_metric(actual["sparse25"], baseline, f"{name} sparse25")
        if (actual["n_unique_rows"] != len(rows)
                or actual["source_rows_sha256"] != _hash_rows(rows)
                or not np.isclose(actual["gain_macro_f1_vs_sparse25"],
                                  candidate["macro_f1"] - baseline["macro_f1"], rtol=0, atol=1e-14)):
            raise RuntimeError(f"View denominator or gain differs for {name}")
        checked_views[name] = {"rows": len(rows), "candidate_macro_f1": candidate["macro_f1"],
                               "sparse25_macro_f1": baseline["macro_f1"]}

    inherited = report["inherited_pilot_H12"]
    for cutoff, low, high in ((55, 56, 67), (67, 68, 79)):
        data = _collect(records, [[cutoff, low, high]])
        candidate = _metric(y_all[data["rows"]], data["candidate"])
        baseline = _metric(y_all[data["rows"]], data["current"])
        value = inherited[str(cutoff)]
        _assert_metric(value["candidate"], candidate, f"inherited H12 {cutoff} candidate")
        _assert_metric(value["sparse25"], baseline, f"inherited H12 {cutoff} sparse25")
        if value["n"] != len(data["rows"]):
            raise RuntimeError(f"Inherited H12 denominator differs at origin {cutoff}")
    return checked_views


def _verify_group_csv(context: Mapping[str, Any], records: Mapping[int, Mapping[str, np.ndarray]]) -> dict[str, int]:
    path = runner.REPORT_ROOT / "groups.csv"
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    observed = {(row["view"], row["field"], row["group"]): row for row in rows}
    expected_rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    bins_by_origin: dict[int, np.ndarray] = {}
    for cutoff in runner.ALL_ORIGINS:
        parsed, _missing, _bad = _number_array(context["frame"]["text_load"])
        fit_mask = context["cycles"] <= cutoff
        finite = parsed[fit_mask & np.isfinite(parsed)]
        edges = np.unique(np.quantile(finite, [0.25, 0.50, 0.75]))
        bins = np.full(len(parsed), -1, dtype=np.int64)
        valid = np.isfinite(parsed)
        bins[valid] = np.searchsorted(edges, parsed[valid], side="right")
        bins_by_origin[cutoff] = bins

    views = context["queue"]["views"]
    blocks_by_name = {"primary_full_six": views["full_six"], "late_primary_W5_W6": views["late_primary"]}
    for index, block in enumerate(views["full_six"], start=1):
        blocks_by_name[f"W{index}"] = [block]
    for view, blocks in blocks_by_name.items():
        data = _collect(records, blocks)
        local = context["frame"].iloc[data["rows"]]
        labels = {
            "entry_friction": local["entry_friction"].astype(str).to_numpy(),
            "stream_availability": np.where(data["missing"], "missing", "present"),
            "fit_defined_text_load_quartile": np.asarray([
                str(int(bins_by_origin[int(origin)][int(row)]))
                for row, origin in zip(data["rows"], data["origin"])
            ]),
        }
        y = context["target"][data["rows"]]
        for field, group_values in labels.items():
            for label in sorted(set(group_values.tolist())):
                mask = group_values == label
                group_rows = data["rows"][mask]
                group_y = y[mask]
                candidate = _metric(group_y, data["candidate"][mask])
                baseline = _metric(group_y, data["current"][mask])
                c0 = int(np.sum(group_y == 0)); c1 = int(np.sum(group_y == 1))
                supported = len(group_rows) >= runner.MIN_GROUP_N and min(c0, c1) >= runner.MIN_EACH_CLASS_N
                expected_rows[(view, field, str(label))] = {
                    "n": len(group_rows), "class_0": c0, "class_1": c1,
                    "supported": supported, "source_rows_sha256": _hash_rows(group_rows),
                    "candidate_macro_f1": candidate["macro_f1"], "sparse25_macro_f1": baseline["macro_f1"],
                    "f1_harm_vs_sparse25": baseline["macro_f1"] - candidate["macro_f1"],
                    "candidate_positive_recall": candidate["positive_recall"],
                    "sparse25_positive_recall": baseline["positive_recall"],
                    "positive_recall_harm_vs_sparse25": baseline["positive_recall"] - candidate["positive_recall"],
                    "guard_failure_descriptive": bool(supported and (
                        baseline["macro_f1"] - candidate["macro_f1"] > runner.MAX_F1_HARM
                        or baseline["positive_recall"] - candidate["positive_recall"] > runner.MAX_RECALL_HARM)),
                }
    if set(observed) != set(expected_rows):
        raise RuntimeError("Group CSV keys differ from independent all-view reconstruction")
    float_fields = ("candidate_macro_f1", "sparse25_macro_f1", "f1_harm_vs_sparse25",
                    "candidate_positive_recall", "sparse25_positive_recall", "positive_recall_harm_vs_sparse25")
    for key, expected in expected_rows.items():
        actual = observed[key]
        for field in ("n", "class_0", "class_1"):
            if int(actual[field]) != expected[field]:
                raise RuntimeError(f"Group {key} {field} differs")
        for field in float_fields:
            if not np.isclose(float(actual[field]), expected[field], rtol=0, atol=1e-14):
                raise RuntimeError(f"Group {key} {field} differs")
        for field in ("supported", "guard_failure_descriptive"):
            if str(actual[field]).casefold() not in {str(expected[field]).casefold(), "true" if expected[field] else "false"}:
                raise RuntimeError(f"Group {key} {field} differs")
        if actual["source_rows_sha256"] != expected["source_rows_sha256"]:
            raise RuntimeError(f"Group {key} row hash differs")
    return {"rows": len(rows), "supported": sum(row["supported"] for row in expected_rows.values()),
            "failures": sum(row["guard_failure_descriptive"] for row in expected_rows.values())}


def verify() -> dict[str, Any]:
    frozen, context = runner.verify_freeze()
    result_path = runner.REPORT_ROOT / "results.json"
    report = json.loads(result_path.read_text(encoding="utf-8"))
    if (report.get("status") != "COMPLETE_DIAGNOSTIC_ONLY"
            or report.get("decision_scope") != "diagnostic only; no model qualification or automatic continuation"
            or report.get("champion_or_model_selection_changed") is not False
            or report.get("no_full_six_h12_claim") is None):
        raise RuntimeError("Diagnostic report scope or status is incorrect")
    if report.get("input_and_source_sha256") != frozen["identity"]["source_and_input_sha256"]:
        raise RuntimeError("Report source hashes differ from frozen identity")
    records, bundle_info = _records(context, frozen)
    views = _verify_views(context, records, report)
    groups = _verify_group_csv(context, records)
    per_cycle_path = runner.REPORT_ROOT / "per_cycle.csv"
    per_cycle_frame = pd.read_csv(per_cycle_path)
    if len(per_cycle_frame) != len(report["per_cycle"]):
        raise RuntimeError("Per-cycle CSV row count differs from report")
    for stored in report["per_cycle"]:
        selector = per_cycle_frame[(per_cycle_frame["view"] == stored["view"])
                                   & (per_cycle_frame["cycle_code"] == stored["cycle_code"])]
        if len(selector) != 1:
            raise RuntimeError("Per-cycle CSV keys do not uniquely match report")
        for field in ("n",):
            if int(selector.iloc[0][field]) != int(stored[field]):
                raise RuntimeError(f"Per-cycle {field} differs for {stored['view']}/{stored['cycle_code']}")
        for field in ("candidate_macro_f1", "sparse25_macro_f1", "gain_macro_f1_vs_sparse25",
                      "candidate_positive_recall", "sparse25_positive_recall"):
            if not np.isclose(float(selector.iloc[0][field]), float(stored[field]), rtol=0, atol=1e-14):
                raise RuntimeError(f"Per-cycle {field} differs for {stored['view']}/{stored['cycle_code']}")

    events = read_events(runner.LEDGER_PATH)
    summary = ledger_summary(runner.LEDGER_PATH)
    reservation = [event for event in events if event.get("event") == "CARD_FITS_RESERVED"
                   and event.get("reservation_key") == f"{runner.CARD_ID}:{runner.STAGE}"]
    releases = [event for event in events if event.get("event") == "RESERVATION_UNUSED_RELEASED"
                and event.get("reservation_key") == f"{runner.CARD_ID}:{runner.STAGE}"]
    if (len(reservation) != 1 or int(reservation[0]["fits"]) != runner.N_NEW_FITS
            or reservation[0].get("category") != "continuation"
            or len(releases) != 1 or int(releases[0]["unused_fits"]) != 0
            or summary["started_and_consumed"] != 30 or summary["saved"] != 30
            or summary["failed_or_interrupted_started"] != 0 or summary["authorized_cap"] != 60
            or summary["active_reserved"] != 30 or summary["remaining_capacity_after_reservations"] != 26):
        raise RuntimeError(f"Four-fit continuation accounting/cap check failed: {summary}")
    expected_ids = {f"{runner.CARD_ID}:{runner.STAGE}:{cutoff}:class_only"
                    for cutoff in runner.CONTINUATION_CUTOFFS}
    actual_ids = {event.get("fit_id") for event in events if event.get("event") == "FIT_STARTED"
                  and event.get("card_id") == runner.CARD_ID}
    if actual_ids != expected_ids:
        raise RuntimeError("N01 diagnostic ledger contains missing or extra fit IDs")
    audit = context["audit_report"]
    if (audit["historical_verdicts_changed"] is not False
            or audit["continuation_gates"]["N02_sketch_candidate"]["eligible_for_authorized_continuation"] is not False):
        raise RuntimeError("The diagnostic amendment changed a historical gate verdict")
    output = {
        "status": "PASS_INDEPENDENT_VERIFICATION",
        "independently_reconstructed_views": views,
        "independently_reconstructed_groups": groups,
        "verified_bundles": bundle_info,
        "verified_fits": 4, "official_estimator_fits": 4,
        "ledger": summary, "test_or_sample_submission_used": False,
        "kaggle_used": False, "historical_N01_N02_verdicts_changed": False,
    }
    runner._write_once(runner.REPORT_ROOT / "verification.json", output)
    return output


def main() -> int:
    result = verify()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
