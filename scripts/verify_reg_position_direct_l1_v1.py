"""Independent prelabel and scoring verifier for REG_POSITION_DIRECT_L1_V1."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.reg_position_direct_l1_v1 as pilot  # noqa: E402


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise pilot.PilotError(f"Invalid JSON in verifier: {path}") from exc
    if not isinstance(value, dict):
        raise pilot.PilotError(f"Expected a JSON object in verifier: {path}")
    return value


def _signature(freeze: Mapping[str, Any]) -> str:
    body = {key: value for key, value in freeze.items() if key != "freeze_signature"}
    return pilot._canonical_sha(body)


def _verified_rank50(raw: np.ndarray, cycles: np.ndarray) -> np.ndarray:
    """Independent implementation of the frozen complete-cycle mapping."""
    raw = np.asarray(raw, dtype=np.float64)
    cycles = np.asarray(cycles, dtype=np.int64)
    if raw.ndim != 1 or raw.shape != cycles.shape or not len(raw) or not np.isfinite(raw).all():
        raise pilot.PilotError("Verifier received invalid raw scores/cycles")
    mapped = np.empty(raw.shape, dtype=np.float64)
    for cycle in np.unique(cycles):
        selected = np.flatnonzero(cycles == cycle)
        average_rank = rankdata(raw[selected], method="average")
        mapped[selected] = 0.5 * raw[selected] + 50.0 * average_rank / len(selected)
    return mapped


def _check_one_fit_unit(freeze: Mapping[str, Any], fit_id: str) -> dict[str, Any]:
    artifact, receipt_path = pilot._unit_paths(fit_id)
    if not artifact.is_file() or not receipt_path.is_file():
        raise pilot.PilotError(f"Missing model/preprocessor artifact or receipt: {fit_id}")
    receipt = _json(receipt_path)
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if (receipt.get("schema_version") != 1 or receipt.get("card_id") != pilot.CARD_ID
            or receipt.get("fit_id") != fit_id
            or receipt.get("freeze_signature") != freeze.get("freeze_signature")
            or receipt.get("artifact_sha256") != pilot._sha_file(artifact)
            or receipt.get("receipt_sha256") != pilot._canonical_sha(body)):
        raise pilot.PilotError(f"Model/preprocessor receipt integrity failed: {fit_id}")
    component = fit_id.rsplit("_", 1)[-1]
    expected_calls = {
        "STRUCTURED": ["_SelectedPreprocessor.fit_transform x1 (one nested OrdinalEncoder.fit)"],
        "SCALER": ["StandardScaler.fit x1"],
        "OHE": ["OneHotEncoder.fit x1"],
        "BAG": ["LGBMRegressor.fit x1"],
        "REAL": ["LGBMRegressor.fit x1"],
        "SHUFFLED": ["LGBMRegressor.fit x1"],
    }
    metadata = receipt.get("fit_metadata", {})
    origin = int(fit_id.split("_E", 1)[1].split("_", 1)[0])
    if metadata.get("underlying_fit_calls") != expected_calls.get(component):
        raise pilot.PilotError(f"Fit receipt call count differs from frozen unit: {fit_id}")
    if component in {"BAG", "REAL", "SHUFFLED"}:
        if (metadata.get("direct_target") != "threat_rank"
                or metadata.get("parameters") != freeze.get("resolved_parameters")
                or metadata.get("fit_rows") != freeze.get("origins", {}).get(str(origin), {}).get("fit_rows")):
            raise pilot.PilotError(f"Estimator receipt target/parameters/fit rows differ: {fit_id}")
    elif metadata.get("fit_rows") != freeze.get("origins", {}).get(str(origin), {}).get("fit_rows"):
        raise pilot.PilotError(f"Preprocessor receipt fit population differs: {fit_id}")
    return {"fit_id": fit_id, "artifact_sha256": receipt["artifact_sha256"],
            "receipt_sha256": pilot._sha_file(receipt_path),
            "fit_metadata": metadata}


def _load_prediction_arrays(freeze: Mapping[str, Any], origin: int, arm: str,
                            item: pilot.OriginInputs) -> dict[str, np.ndarray]:
    fit_id = f"{pilot.CARD_ID}_E{origin}_{arm}"
    bundle_path, receipt_path = pilot._prediction_paths(fit_id)
    if not bundle_path.is_file() or not receipt_path.is_file():
        raise pilot.PilotError(f"Missing prediction artifact/receipt: {fit_id}")
    receipt = _json(receipt_path)
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    model_receipt_path = pilot._unit_paths(fit_id)[1]
    model_receipt = _json(model_receipt_path)
    if (receipt.get("schema_version") != 1 or receipt.get("fit_id") != fit_id
            or receipt.get("freeze_signature") != freeze.get("freeze_signature")
            or receipt.get("bundle_sha256") != pilot._sha_file(bundle_path)
            or receipt.get("model_receipt_sha256") != model_receipt.get("receipt_sha256")
            or receipt.get("receipt_sha256") != pilot._canonical_sha(body)):
        raise pilot.PilotError(f"Prediction receipt/hash/schema failed: {fit_id}")
    with np.load(bundle_path, allow_pickle=False) as archive:
        # Explicit target-free members only; the archive contains no labels.
        origin_value = int(np.asarray(archive["origin"]).reshape(-1)[0])
        arm_value = str(np.asarray(archive["arm"]).reshape(-1)[0])
        fit_rows = np.asarray(archive["fit_rows"], dtype=np.int64)
        query_rows = np.asarray(archive["query_rows"], dtype=np.int64)
        query_ids = np.asarray(archive["query_record_ids"], dtype=str)
        query_cycles = np.asarray(archive["query_cycles"], dtype=np.int64)
        missing = np.asarray(archive["missing_mask"], dtype=bool)
        baseline = np.asarray(archive["baseline"], dtype=np.float64)
        raw = np.asarray(archive["raw_auxiliary"], dtype=np.float64)
        mapped = np.asarray(archive["mapped_auxiliary"], dtype=np.float64)
        prediction = np.asarray(archive["prediction"], dtype=np.float64)
    if (origin_value != origin or arm_value != arm
            or not np.array_equal(fit_rows, item.fit_rows)
            or not np.array_equal(query_rows, item.query_rows)
            or not np.array_equal(query_ids, item.query_ids)
            or not np.array_equal(query_cycles, item.query_cycles)
            or not np.array_equal(missing, item.query_missing)
            or not np.array_equal(baseline, item.query_baseline)
            or raw.shape != baseline.shape or mapped.shape != baseline.shape or prediction.shape != baseline.shape
            or not np.isfinite(raw).all() or not np.isfinite(mapped).all() or not np.isfinite(prediction).all()):
        raise pilot.PilotError(f"Prediction ID/value contract failed: {fit_id}")
    if (receipt.get("query_rows") != len(query_rows)
            or receipt.get("query_ids_sha256") != pilot._sha_array(query_ids)
            or receipt.get("query_cycles_sha256") != pilot._sha_array(query_cycles)
            or receipt.get("raw_auxiliary_sha256") != pilot._sha_array(raw)
            or receipt.get("mapped_auxiliary_sha256") != pilot._sha_array(mapped)
            or receipt.get("prediction_sha256") != pilot._sha_array(prediction)
            or receipt.get("rank_before_missing_mask") is not True
            or receipt.get("query_targets_read") is not False):
        raise pilot.PilotError(f"Prediction internal digest/target-boundary receipt failed: {fit_id}")
    expected_mapped = _verified_rank50(raw, query_cycles)
    # Apply rank mapping and blend to every query row, then substitute R06 for missing text.
    expected = np.clip(0.75 * baseline + 0.25 * expected_mapped, 0.0, 100.0)
    expected[missing] = baseline[missing]
    if (not np.array_equal(mapped, expected_mapped) or not np.array_equal(prediction, expected)
            or not np.array_equal(prediction[missing], baseline[missing])
            or np.any(prediction < 0.0) or np.any(prediction > 100.0)):
        raise pilot.PilotError(f"Independent rank/blend/missing-route reconstruction failed: {fit_id}")
    return {"raw": raw, "mapped": mapped, "prediction": prediction,
            "baseline": baseline, "fit_id": fit_id,
            "bundle_sha256": pilot._sha_file(bundle_path),
            "receipt_sha256": pilot._sha_file(receipt_path)}


def _validate_ledger() -> dict[str, Any]:
    events = pilot._read_events(pilot.LEDGER_PATH)
    starts = [event for event in events if event.get("event") == "FIT_STARTED"]
    saved = [event for event in events if event.get("event") == "FIT_SAVED"]
    failed = [event for event in events if event.get("event") == "FIT_FAILED"]
    if ([event.get("fit_id") for event in starts] != list(pilot.FIT_IDS)
            or [event.get("fit_id") for event in saved] != list(pilot.FIT_IDS)
            or len(starts) != 12 or len(saved) != 12 or failed):
        raise pilot.PilotError("Independent ledger check requires exactly 12 ordered starts/saves and no failures")
    return {"starts": len(starts), "saves": len(saved), "failed": 0,
            "ledger_sha256": pilot._sha_file(pilot.LEDGER_PATH)}


def prelabel_verify(source_root: Path = pilot.SOURCE_ROOT) -> dict[str, Any]:
    """Recheck all six models/predictions without opening Regression targets."""
    source_root = source_root.resolve()
    freeze = _json(pilot.FREEZE_PATH)
    if freeze.get("freeze_signature") != _signature(freeze):
        raise pilot.PilotError("Independent FREEZE.json signature failed")
    for relative, expected in freeze.get("source_hashes", {}).items():
        path = pilot._contained_file(source_root if relative.startswith(("data/", "experiments/")) else ROOT,
                                     relative)
        if pilot._sha_file(path) != expected:
            raise pilot.PilotError(f"Independent source hash changed: {relative}")
    frame = pilot._load_feature_frame(source_root)  # usecols excludes threat_rank
    origins = {origin: pilot._load_origin(source_root, frame, origin) for origin in pilot.ORIGINS}
    ledger = _validate_ledger()
    model_units = {fit_id: _check_one_fit_unit(freeze, fit_id) for fit_id in pilot.FIT_IDS}
    predictions: dict[str, Any] = {}
    for origin, item in origins.items():
        predictions[str(origin)] = {
            arm: _load_prediction_arrays(freeze, origin, arm, item)
            for arm in pilot.ARMS
        }
    ready_path = pilot.RUN_ROOT / "PREDICTIONS_READY.json"
    ready = _json(ready_path)
    if (ready.get("freeze_signature") != freeze.get("freeze_signature")
            or ready.get("fit_units_started") != 12 or ready.get("fit_units_saved") != 12
            or ready.get("query_targets_read") is not False
            or ready.get("ledger_sha256") != ledger["ledger_sha256"]):
        raise pilot.PilotError("Runner readiness record differs from independently verified artifacts")
    for origin in pilot.ORIGINS:
        for arm in pilot.ARMS:
            fit_id = f"{pilot.CARD_ID}_E{origin}_{arm}"
            expected = ready.get("predictions", {}).get(fit_id, {})
            actual = predictions[str(origin)][arm]
            if (expected.get("bundle_sha256") != actual["bundle_sha256"]
                    or expected.get("receipt_sha256") != actual["receipt_sha256"]):
                raise pilot.PilotError(f"Runner readiness index differs from durable prediction: {fit_id}")
    verified = {
        "schema_version": 1, "card_id": pilot.CARD_ID,
        "status": "PASS", "verified_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_signature": freeze["freeze_signature"],
        "fit_units_started": 12, "fit_units_saved": 12, "fit_units_failed": 0,
        "models_verified": model_units,
        "predictions_verified": {
            str(origin): {arm: {"fit_id": arrays[arm]["fit_id"],
                                "bundle_sha256": arrays[arm]["bundle_sha256"],
                                "receipt_sha256": arrays[arm]["receipt_sha256"],
                                "n_query": int(len(arrays[arm]["prediction"])),
                                "missing_route_exact": bool(np.array_equal(
                                    arrays[arm]["prediction"][origins[origin].query_missing],
                                    origins[origin].query_baseline[origins[origin].query_missing]))}
                           for arm in pilot.ARMS}
            for origin, arrays in ((int(o), v) for o, v in predictions.items())
        },
        "ledger": ledger,
        "query_targets_read": False,
        "rank_blend_formula_independently_reconstructed": True,
        "full_cycle_rank_before_missing_mask": True,
        "tamper_negative_cases": "PASS in the synthetic entrypoint audit",
    }
    pilot._atomic_json(pilot.RUN_ROOT / "PRELABEL_VERIFICATION.json", verified)
    pilot._atomic_json(pilot.PRELABEL_PATH, verified)
    return verified


def _mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y, dtype=np.float64) - np.asarray(pred, dtype=np.float64))))


def _metric_view(y: np.ndarray, baseline: np.ndarray,
                 predictions: Mapping[str, np.ndarray], mask: np.ndarray,
                 groups: Mapping[str, np.ndarray] | None = None) -> dict[str, Any]:
    selected = np.asarray(mask, dtype=bool)
    if selected.shape != y.shape or not selected.any():
        raise pilot.PilotError("Metric view has no aligned rows")
    base_mae = _mae(y[selected], baseline[selected])
    metrics: dict[str, Any] = {"R06": {"mae": base_mae}}
    for arm in pilot.ARMS:
        arm_mae = _mae(y[selected], predictions[arm][selected])
        metrics[arm] = {"mae": arm_mae,
                        "gain_vs_R06": base_mae - arm_mae,
                        "mae_harm_vs_R06": arm_mae - base_mae}
    metrics["REAL"]["gain_vs_BAG"] = metrics["BAG"]["mae"] - metrics["REAL"]["mae"]
    metrics["REAL"]["gain_vs_SHUFFLED"] = metrics["SHUFFLED"]["mae"] - metrics["REAL"]["mae"]
    result: dict[str, Any] = {"n": int(selected.sum()), "metrics": metrics}
    if groups is not None:
        by_group: dict[str, Any] = {}
        for name, group_mask in groups.items():
            chosen = selected & np.asarray(group_mask, dtype=bool)
            n = int(chosen.sum())
            if n < 100:
                by_group[name] = {"n": n, "status": "UNKNOWN_UNSUPPORTED"}
                continue
            reference = _mae(y[chosen], baseline[chosen])
            by_group[name] = {
                "n": n, "status": "SUPPORTED", "R06_mae": reference,
                "arms": {arm: {"mae": _mae(y[chosen], predictions[arm][chosen]),
                               "harm_vs_R06": _mae(y[chosen], predictions[arm][chosen]) - reference}
                         for arm in pilot.ARMS},
            }
        result["groups"] = by_group
    return result


def _group_masks(item: pilot.OriginInputs, y: np.ndarray) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {
        "missing": item.query_missing.copy(),
        "target_lt5": y < 5.0,
        "target_ge95": y >= 95.0,
    }
    friction = item.group_features["entry_friction"]
    masks["EF0"] = np.isfinite(friction) & (friction == 0.0)
    masks["EF1"] = np.isfinite(friction) & (friction == 1.0)
    for feature in ("impact_load", "text_load"):
        bins = item.group_features[feature]
        for level in range(int(bins.max()) + 1):
            masks[f"{feature}_Q{level + 1}"] = bins == level
    return masks


def _gates(views: Mapping[str, Any]) -> dict[str, Any]:
    tol = 1e-10

    def threshold(name: str, value: float, minimum: float) -> dict[str, Any]:
        return {"status": "PASS" if value + tol >= minimum else "FAIL",
                "value": value, "minimum": minimum, "tolerance": tol}

    def maximum(name: str, value: float, limit: float) -> dict[str, Any]:
        return {"status": "PASS" if value <= limit + tol else "FAIL",
                "value": value, "maximum": limit, "tolerance": tol}

    output = {
        "REAL_gain_vs_R06_E67_H12": threshold("E67", views["E67_H12"]["metrics"]["REAL"]["gain_vs_R06"], 0.03),
        "REAL_gain_vs_R06_E75_W6": threshold("E75", views["E75_W6"]["metrics"]["REAL"]["gain_vs_R06"], 0.03),
    }
    for view_name, prefix in (("E67_H12", "E67_H12"), ("E75_W6", "E75_W6")):
        m = views[view_name]["metrics"]["REAL"]
        output[f"REAL_gain_vs_BAG_{prefix}"] = threshold(prefix, m["gain_vs_BAG"], 0.01)
        output[f"REAL_gain_vs_SHUFFLED_{prefix}"] = threshold(prefix, m["gain_vs_SHUFFLED"], 0.01)
    for window in ("W4", "W5", "W6"):
        harm = views[f"E67_{window}"]["metrics"]["REAL"]["mae_harm_vs_R06"]
        output[f"REAL_E67_{window}_harm"] = maximum(window, harm, 0.10)
    group_failures = []
    group_unknowns = []
    for view_name in ("E67_H12", "E67_W4", "E67_W5", "E67_W6", "E75_W6"):
        for group, result in views[view_name].get("groups", {}).items():
            if result["status"] == "UNKNOWN_UNSUPPORTED":
                group_unknowns.append({"view": view_name, "group": group, "n": result["n"]})
                continue
            for arm, value in result["arms"].items():
                if value["harm_vs_R06"] > 0.25 + tol:
                    group_failures.append({"view": view_name, "group": group, "arm": arm,
                                           "harm": value["harm_vs_R06"], "maximum": 0.25})
    group_status = "FAIL" if group_failures else ("UNKNOWN" if group_unknowns else "PASS")
    output["protected_groups"] = {"status": group_status,
                                  "maximum_supported_harm": 0.25,
                                  "unknown_count": len(group_unknowns),
                                  "failure_count": len(group_failures),
                                  "unknowns": group_unknowns,
                                  "failures": group_failures}
    states = [value["status"] for value in output.values() if isinstance(value, dict) and "status" in value]
    overall = "FAIL" if "FAIL" in states else ("UNKNOWN" if "UNKNOWN" in states else "PASS")
    output["overall_status"] = overall
    output["promotion_eligible"] = overall == "PASS"
    output["submission_decision"] = "NO_AUTO_PROMOTION"
    return output


def score_after_prelabel(source_root: Path = pilot.SOURCE_ROOT) -> dict[str, Any]:
    """Open query labels only after independent prelabel verification passes."""
    source_root = source_root.resolve()
    prelabel = _json(pilot.PRELABEL_PATH)
    freeze = _json(pilot.FREEZE_PATH)
    if (prelabel.get("status") != "PASS" or prelabel.get("freeze_signature") != freeze.get("freeze_signature")
            or prelabel.get("query_targets_read") is not False or freeze.get("freeze_signature") != _signature(freeze)):
        raise pilot.PilotError("Scoring blocked until independent prelabel verification passes")
    for relative, expected in freeze.get("source_hashes", {}).items():
        path = pilot._contained_file(source_root if relative.startswith(("data/", "experiments/")) else ROOT,
                                     relative)
        if pilot._sha_file(path) != expected:
            raise pilot.PilotError(f"Scoring blocked because a frozen source changed: {relative}")
    if prelabel.get("ledger", {}).get("starts") != 12 or prelabel.get("ledger", {}).get("saves") != 12:
        raise pilot.PilotError("Scoring blocked until all 12 frozen units are durable")
    frame = pilot._load_feature_frame(source_root)
    origins = {origin: pilot._load_origin(source_root, frame, origin) for origin in pilot.ORIGINS}
    e67 = origins[67]
    y67 = pilot._read_targets_for_rows(pilot._contained_file(source_root, pilot.TRAIN_REL),
                                       e67.query_rows, e67.query_ids,
                                       expected_total_rows=len(frame))
    truth_by_row = {int(row): float(value) for row, value in zip(e67.query_rows, y67)}
    view_data: dict[str, dict[str, Any]] = {}
    predictions_by_origin: dict[int, dict[str, np.ndarray]] = {}
    truths_by_origin: dict[int, np.ndarray] = {}
    for origin in pilot.ORIGINS:
        item = origins[origin]
        truths = np.asarray([truth_by_row[int(row)] for row in item.query_rows], dtype=np.float64)
        truths_by_origin[origin] = truths
        arrays = {arm: _load_prediction_arrays(freeze, origin, arm, item) for arm in pilot.ARMS}
        predictions_by_origin[origin] = {arm: arrays[arm]["prediction"] for arm in pilot.ARMS}
        labels = np.asarray(item.query_cycles, dtype=np.int64)
        if origin == 67:
            masks = {"E67_H12": np.ones(len(labels), dtype=bool),
                     "E67_W4": (labels >= 68) & (labels <= 71),
                     "E67_W5": (labels >= 72) & (labels <= 75),
                     "E67_W6": (labels >= 76) & (labels <= 79)}
        else:
            masks = {"E75_W6": (labels >= 76) & (labels <= 79)}
        groups = _group_masks(item, truths)
        for name, mask in masks.items():
            view_data[name] = _metric_view(truths, item.query_baseline,
                                           predictions_by_origin[origin], mask, groups)

    # Unique late view: E67 W4 plus E75 W6. E75 W6 overlaps E67 W6 fully;
    # include only E75's predictions for cycles 76-79 and never append E67 W6 again.
    e75 = origins[75]
    mask67 = (e67.query_cycles >= 68) & (e67.query_cycles <= 71)
    mask75 = (e75.query_cycles >= 76) & (e75.query_cycles <= 79)
    ids67 = set(e67.query_ids[mask67].tolist())
    ids75 = set(e75.query_ids[mask75].tolist())
    if ids67 & ids75 or len(ids75) != int(mask75.sum()):
        raise pilot.PilotError("Unique late view contains duplicate or misaligned record IDs")
    late_y = np.concatenate((truths_by_origin[67][mask67], truths_by_origin[75][mask75]))
    late_baseline = np.concatenate((e67.query_baseline[mask67], e75.query_baseline[mask75]))
    late_predictions = {arm: np.concatenate((predictions_by_origin[67][arm][mask67],
                                             predictions_by_origin[75][arm][mask75]))
                        for arm in pilot.ARMS}
    late_mask = np.ones(len(late_y), dtype=bool)
    view_data["LATE_UNIQUE_E67_W4_PLUS_E75_W6"] = _metric_view(
        late_y, late_baseline, late_predictions, late_mask)

    # E75_W6 query IDs must be the exact row subset also present in E67_W6.
    ids_e67_w6 = set(e67.query_ids[(e67.query_cycles >= 76) & (e67.query_cycles <= 79)].tolist())
    if ids_e67_w6 != ids75:
        raise pilot.PilotError("Expected E75_W6/E67_W6 full record overlap was not exact")
    gate_result = _gates(view_data)
    result = {
        "schema_version": 1, "card_id": pilot.CARD_ID,
        "status": gate_result["overall_status"],
        "scored_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_signature": freeze["freeze_signature"],
        "query_targets_accessed": True,
        "query_target_rows": int(len(e67.query_rows)),
        "query_targets_sha256": pilot._sha_array(y67),
        "origin_query_rows": {"E67": len(e67.query_rows), "E75": len(e75.query_rows)},
        "views": view_data,
        "gates": gate_result,
        "overlap": {
            "E75_W6_is_exact_subset_of_E67_W6": True,
            "late_unique_components": {"E67_W4_rows": int(mask67.sum()), "E75_W6_rows": int(mask75.sum())},
            "late_unique_rows": int(len(late_y)), "late_components_have_disjoint_ids": True,
            "definition": "E67 cycles 68-71 plus E75 cycles 76-79; E67 cycles 76-79 is omitted because it duplicates the latter records."
        },
        "csv_exported": False,
        "test_sample_kaggle_accessed": False,
        "submission_decision": "No automatic promotion or upload.",
    }
    pilot._atomic_json(pilot.RUN_ROOT / "RESULTS.json", result)
    pilot._atomic_json(pilot.REPORT_ROOT / "RESULTS.json", result)
    (pilot.REPORT_ROOT / "REPORT.md").write_text(_render_report(result), encoding="utf-8", newline="\n")
    return result


def _render_report(result: Mapping[str, Any]) -> str:
    gates = result["gates"]
    lines = [
        "# REG_POSITION_DIRECT_L1_V1",
        "",
        f"Status: **{result['status']}**",
        "",
        "This is an adaptive development pilot. E67 and E75 query views overlap; scores are engineering gates, not significance tests.",
        "",
        "## Primary views",
        "",
        "| View | n | R06 MAE | BAG MAE | REAL MAE | SHUFFLED MAE | REAL gain vs R06 | REAL gain vs BAG | REAL gain vs SHUFFLED |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("E67_H12", "E67_W4", "E67_W5", "E67_W6", "E75_W6", "LATE_UNIQUE_E67_W4_PLUS_E75_W6"):
        view = result["views"][name]
        metric = view["metrics"]
        lines.append(f"| {name} | {view['n']} | {metric['R06']['mae']:.6f} | {metric['BAG']['mae']:.6f} | "
                     f"{metric['REAL']['mae']:.6f} | {metric['SHUFFLED']['mae']:.6f} | "
                     f"{metric['REAL']['gain_vs_R06']:.6f} | {metric['REAL']['gain_vs_BAG']:.6f} | "
                     f"{metric['REAL']['gain_vs_SHUFFLED']:.6f} |")
    lines.extend(["", "## Gate checks", ""])
    for name, value in gates.items():
        if isinstance(value, dict) and "status" in value:
            lines.append(f"- **{name}: {value['status']}**")
            if name == "protected_groups":
                lines.append(f"  - Unsupported group/view cells: {value['unknown_count']} (reported UNKNOWN)")
                lines.append(f"  - Supported harm failures: {value['failure_count']}")
            elif "value" in value:
                lines.append(f"  - Observed {value['value']:.6f}; threshold {value.get('minimum', value.get('maximum')):.6f}")
    lines.extend([
        "", "## Interpretation and overlap", "",
        "The late unique view uses E67 W4 rows for cycles 68-71 plus E75 W6 rows for cycles 76-79. E75 W6 overlaps E67 W6 exactly, so the E67 W6 copy is omitted from the unique union.",
        "", "Unsupported protected groups remain UNKNOWN. No arm is promoted or uploaded automatically.",
        "", "The verifier read Regression Train query targets only after all 12 fit units and six raw prediction bundles passed prelabel verification. Test, Sample, and Kaggle data were not accessed.",
        ""
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prelabel", "score"))
    parser.add_argument("--source-root", type=Path, default=pilot.SOURCE_ROOT)
    args = parser.parse_args(argv)
    try:
        if args.command == "prelabel":
            value = prelabel_verify(args.source_root)
        else:
            value = score_after_prelabel(args.source_root)
        print(json.dumps({"status": value["status"],
                          "freeze_signature": value["freeze_signature"],
                          "query_targets_read": value.get("query_targets_read", value.get("query_targets_accessed"))},
                         ensure_ascii=False, indent=2))
        return 0
    except pilot.PilotError as exc:
        print(f"NO-GO: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
