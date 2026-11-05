"""Independent replay of R2 hinge diagnostic artifacts and CSV schema."""
from __future__ import annotations

import csv
import hashlib
import inspect
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

from scripts import reg_monotone_hinge_p50_v1 as r1
from scripts import reg_monotone_hinge_p50_v2 as r2


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else r2.ROOT / candidate


def _load_selected(path: Path, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if not set(names).issubset(archive.files):
            raise r2.GateError(f"Independent verifier NPZ fields missing: {path}")
        return {name: np.asarray(archive[name]).copy() for name in names}


def _metric(y: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    return {"n": int(len(error)), "mae": float(np.mean(np.abs(error))),
            "mean_error": float(np.mean(error)),
            "p90_absolute_error": float(np.quantile(np.abs(error), 0.90))}


def _policy_metrics(y: np.ndarray, cycles: np.ndarray, missing: np.ndarray,
                    prediction: np.ndarray) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    masks = {"present": ~missing, "missing": missing,
             "target_le_5": y <= 5.0, "target_ge_95": y >= 95.0}
    for name, mask in masks.items():
        n = int(mask.sum())
        if n < 200:
            groups[name] = {"n": n, "status": "UNKNOWN_UNSUPPORTED", "mae": None}
        else:
            groups[name] = {"status": "SUPPORTED", **_metric(y[mask], prediction[mask])}
    groups["by_cycle"] = {str(int(cycle)): _metric(y[cycles == cycle], prediction[cycles == cycle])
                          for cycle in np.unique(cycles)}
    return {"overall": _metric(y, prediction), "groups": groups}


def _metrics_from_rows(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    y, cycles, missing = arrays["targets"], arrays["cycles"], arrays["missing_mask"]
    return {name: _policy_metrics(y, cycles, missing, arrays[name]) for name in
        ("raw_direct", "causal_global_all_rows", "causal_global_present_only", "hinge_present_only")}


def _assert_close_tree(expected: Any, actual: Any, path: str = "metrics") -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(expected) != set(actual):
            raise r2.GateError(f"Independent replay structure differs at {path}")
        for key in expected:
            _assert_close_tree(expected[key], actual[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise r2.GateError(f"Independent replay list differs at {path}")
        for i, (left, right) in enumerate(zip(expected, actual, strict=True)):
            _assert_close_tree(left, right, f"{path}[{i}]")
        return
    if isinstance(expected, (int, bool, str)) or expected is None:
        if type(expected) is not type(actual) or expected != actual:
            raise r2.GateError(f"Independent replay value differs at {path}")
        return
    if not math.isclose(float(expected), float(actual), rel_tol=1e-10, abs_tol=1e-12):
        raise r2.GateError(f"Independent replay number differs at {path}: {expected} != {actual}")


def _params(state: Mapping[str, Any]) -> tuple[float, float, float]:
    result = state["solver_result"]
    a, sl, sh = (float(result[key]) for key in ("intercept", "slope_low", "slope_high"))
    if not all(np.isfinite([a, sl, sh])) or not (0.0 <= sl <= 2.0 and 0.0 <= sh <= 2.0):
        raise r2.GateError("Independent verifier found invalid hinge coefficients")
    return a, sl, sh


def _expected_stage_rows(identity: Mapping[str, Any], targets: np.ndarray,
                         fit_id: str, *, score: bool) -> dict[str, np.ndarray]:
    origins = np.asarray(identity["origins"], dtype=np.int64)
    cycles = np.asarray(identity["cycles"], dtype=np.int64)
    if fit_id == "HINGE_D1" and score:
        mask = (origins == 67) & (cycles >= 72) & (cycles <= 75)
    elif fit_id == "HINGE_D2" and score:
        mask = (origins == 75) & (cycles >= 76) & (cycles <= 79)
    elif fit_id == "HINGE_D1":
        mask = (origins == 67) & (cycles >= 68) & (cycles <= 71)
    elif fit_id == "HINGE_D2":
        mask = (origins == 67) & (cycles >= 68) & (cycles <= 75)
    else:
        raise r2.GateError(f"Independent verifier received an unknown fit ID: {fit_id}")
    if not mask.any() or len(targets) != len(origins):
        raise r2.GateError(f"Independent verifier reconstructed an empty/misaligned window: {fit_id}")
    return {
        "physical_rows": np.asarray(identity["rows"][mask], dtype=np.int64),
        "record_ids": np.asarray(identity["ids"][mask], dtype=str),
        "prediction_origins": np.asarray(identity["origins"][mask], dtype=np.int8),
        "original_bundle_indices": np.asarray(identity["indices"][mask], dtype=np.int64),
        "cycles": np.asarray(identity["cycles"][mask], dtype=np.int16),
        "direct_predictions": np.asarray(identity["direct"][mask], dtype="<f8"),
        "missing_mask": np.asarray(identity["missing"][mask], dtype=bool),
        "targets": np.asarray(targets[mask], dtype="<f8"),
    }


def _assert_frozen_stage_rows(actual: Mapping[str, np.ndarray],
                              identity: Mapping[str, Any], targets: np.ndarray,
                              fit_id: str, *, score: bool) -> int:
    expected = _expected_stage_rows(identity, targets, fit_id, score=score)
    for name, expected_values in expected.items():
        if name not in actual:
            raise r2.GateError(f"Frozen stage artifact lacks {name}: {fit_id}")
        actual_values = np.asarray(actual[name])
        if name in {"direct_predictions", "targets"}:
            matches = np.array_equal(actual_values.astype("<f8").view("<u8"),
                                     expected_values.view("<u8"))
        else:
            matches = np.array_equal(actual_values, expected_values)
        if not matches:
            raise r2.GateError(f"Frozen OOF/Train row replay differs: {fit_id}/{name}")
    return len(expected["targets"])


def _independent_subgroup_harm(metrics: Mapping[str, Any]) -> dict[str, Any]:
    candidate = metrics["hinge_present_only"]["groups"]
    control = metrics["causal_global_present_only"]["groups"]
    groups: dict[str, Any] = {}
    harms = []
    for name in ("present", "missing", "target_le_5", "target_ge_95"):
        left, right = candidate[name], control[name]
        if left["status"] != "SUPPORTED" or right["status"] != "SUPPORTED":
            groups[name] = {"status": "UNKNOWN_UNSUPPORTED", "n": left["n"], "harm": None}
        else:
            harm = float(left["mae"] - right["mae"])
            groups[name] = {"status": "SUPPORTED", "n": left["n"], "harm": harm}
            harms.append(harm)
    return {"groups": groups,
            "maximum_supported_harm": max(harms) if harms else None}


def _resource_contract() -> dict[str, int]:
    try:
        policy = json.loads(r2.RESOURCE_POLICY_FILE.read_text(encoding="utf-8"))
        values = policy["workloads"]["small_calibration_lp"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise r2.GateError("Independent verifier cannot read the frozen RAM policy") from exc
    expected = {"minimum_available_bytes": 2 * 1024**3,
        "incremental_estimate_bytes": 512 * 1024**2,
        "minimum_projected_free_bytes": 1024**3,
        "minimum_actual_free_at_boundary_bytes": 1024**3}
    if values != expected:
        raise r2.GateError("Independent verifier RAM policy differs from approved workload floors")
    return expected


def _verify_resource_snapshot(name: str, snapshot: Mapping[str, Any], *,
                              initial: bool,
                              expected_boundary: str | None = None) -> dict[str, Any]:
    contract = _resource_contract()
    available = snapshot.get("available_bytes_measured")
    projected = snapshot.get("projected_free_bytes")
    if (snapshot.get("status") != "PASS"
            or (expected_boundary is not None
                and snapshot.get("boundary") != expected_boundary)
            or not isinstance(available, int)
            or not isinstance(projected, int)
            or available < contract["minimum_actual_free_at_boundary_bytes"]
            or projected != available - contract["incremental_estimate_bytes"]
            or snapshot.get("actual_boundary_pass") is not True
            or snapshot.get("actual_available_boundary_minimum_bytes")
                != contract["minimum_actual_free_at_boundary_bytes"]
            or snapshot.get("incremental_estimate_bytes")
                != contract["incremental_estimate_bytes"]
            or snapshot.get("minimum_projected_free_bytes")
                != contract["minimum_projected_free_bytes"]
            or snapshot.get("estimated_peak_is_measured") is not False
            or snapshot.get("process_rss_measured") is not True
            or not isinstance(snapshot.get("process_rss_bytes"), int)
            or snapshot.get("process_rss_bytes", 0) <= 0
            or snapshot.get("threads") != 2
            or snapshot.get("concurrent_heavy_jobs") != 1):
        raise r2.GateError(f"Independent verifier RAM receipt failed: {name}")
    peak = snapshot.get("os_peak_working_set_bytes")
    peak_measured = snapshot.get("os_peak_working_set_measured")
    if ((peak is None and peak_measured is not False)
            or (peak is not None and (not isinstance(peak, int) or peak <= 0
                                      or peak_measured is not True))):
        raise r2.GateError(f"Independent verifier OS peak measurement flags failed: {name}")
    if initial:
        if (available < contract["minimum_available_bytes"]
                or projected < contract["minimum_projected_free_bytes"]
                or snapshot.get("minimum_initial_available_bytes")
                    != contract["minimum_available_bytes"]
                or snapshot.get("initial_floor_pass") is not True
                or snapshot.get("projected_free_pass") is not True):
            raise r2.GateError(f"Independent verifier initial RAM floor failed: {name}")
    return {"boundary": snapshot.get("boundary"),
            "available_bytes_measured": available,
            "process_rss_measured": True,
            "os_peak_working_set_measured": peak_measured,
            "initial_floor_checked": initial}


def _verify_resource_receipt(report: Mapping[str, Any], approval_sha: str) -> dict[str, Any]:
    expected_rel = (r2.RUN_DIR / "RESOURCE_SNAPSHOTS_R2.json").relative_to(r2.ROOT).as_posix()
    if report.get("resource_receipt_path") != expected_rel:
        raise r2.GateError("Diagnostic report resource receipt path differs from the frozen R2 path")
    path = r2.ROOT / expected_rel
    if not path.is_file() or r2.file_sha256(path) != report.get("resource_receipt_sha256"):
        raise r2.GateError("Diagnostic report resource receipt hash mismatch")
    if report.get("resource_policy_sha256") != r2.file_sha256(r2.RESOURCE_POLICY_FILE):
        raise r2.GateError("Diagnostic report resource policy hash mismatch")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if (receipt.get("schema_version") != 1
            or receipt.get("experiment_id") != r2.EXPERIMENT_ID
            or receipt.get("prefit_freeze_sha256") != report.get("prefit_freeze_sha256")
            or receipt.get("root_source_pass_sha256") != approval_sha
            or receipt.get("resource_policy_sha256") != report.get("resource_policy_sha256")):
        raise r2.GateError("Diagnostic resource receipt binding failed")
    boundaries = receipt.get("boundaries", [])
    names = [item.get("boundary") for item in boundaries]
    expected_names = ["initial_locked_preallocation", "after_selective_oof_identity",
                      "after_selected_train_targets"]
    if names != expected_names:
        raise r2.GateError("Diagnostic resource receipt boundary sequence differs")
    checks = [_verify_resource_snapshot(name, snapshot, initial=(index == 0),
                                        expected_boundary=name)
              for index, (name, snapshot) in enumerate(zip(names, boundaries, strict=True))]
    return {"resource_receipt_sha256": r2.file_sha256(path), "boundaries": checks}


def _verify_fit(record: Mapping[str, Any], freeze: Mapping[str, Any],
                approval_sha: str, identity: Mapping[str, Any],
                targets: np.ndarray) -> dict[str, Any]:
    fit_id = str(record["fit_id"])
    state_path = _resolve(record["fit_state_path"])
    input_path = _resolve(record["fit_input_path"])
    slack_path = _resolve(record["solver_slacks_path"])
    for path, expected in ((state_path, record["fit_state_sha256"]),
                           (input_path, record["fit_input_sha256"]),
                           (slack_path, record["solver_slacks_sha256"])):
        if not path.is_file() or r2.file_sha256(path) != expected:
            raise r2.GateError(f"Independent verifier artifact hash mismatch: {path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if (state.get("status") != "LP_COMPLETE" or state.get("fit_id") != fit_id
            or state.get("fit_input", {}).get("source_commit") != freeze["source_commit"]
            or state.get("fit_input", {}).get("root_source_pass_sha256") != approval_sha
            or state.get("fit_input", {}).get("prefit_freeze_sha256") != r2.file_sha256(r2.PREFIT_FREEZE)):
        raise r2.GateError(f"Independent verifier fit-state binding failed: {fit_id}")
    fit = _load_selected(input_path, ("physical_rows", "record_ids", "prediction_origins",
        "original_bundle_indices", "cycles", "direct_predictions", "missing_mask", "targets"))
    frozen_rows = _assert_frozen_stage_rows(fit, identity, targets, fit_id, score=False)
    p, y = fit["direct_predictions"].astype(np.float64), fit["targets"].astype(np.float64)
    if (p.ndim != 1 or y.shape != p.shape or not np.all(np.isfinite(p))
            or not np.all(np.isfinite(y)) or np.any((p < 0) | (p > 100))
            or np.any((y < 0) | (y > 100))):
        raise r2.GateError(f"Independent verifier fit arrays invalid: {fit_id}")
    row_hashes = state["fit_input"]["row_hashes"]
    actual_hashes = {
        "physical_rows_sha256": r1.tagged_array_sha256(fit["physical_rows"]),
        "record_ids_sha256": r1.tagged_array_sha256(fit["record_ids"]),
        "prediction_origins_sha256": r1.tagged_array_sha256(fit["prediction_origins"]),
        "original_bundle_indices_sha256": r1.tagged_array_sha256(fit["original_bundle_indices"]),
        "cycles_sha256": r1.tagged_array_sha256(fit["cycles"]),
        "missing_mask_sha256": r1.tagged_array_sha256(fit["missing_mask"]),
        "direct_prediction_sha256": r1.tagged_array_sha256(fit["direct_predictions"], dtype="<f8"),
        "target_sha256": r1.tagged_array_sha256(fit["targets"], dtype="<f8"),
        "direct_raw_f64_sha256": r1.raw_direct_f64_sha256(fit["direct_predictions"]),
    }
    if actual_hashes != row_hashes:
        raise r2.GateError(f"Independent verifier fit row hashes differ: {fit_id}")
    if (state["fit_input"].get("fit_input_bundle_sha256") != record["fit_input_sha256"]
            or state["fit_input"].get("fit_rows") != len(y)):
        raise r2.GateError(f"Independent verifier fit input binding differs: {fit_id}")
    a, sl, sh = _params(state)
    prediction = a + sl * np.minimum(p, 50.0) + sh * np.maximum(p - 50.0, 0.0)
    loss = float(np.abs(y - prediction).sum())
    solver_result = state["solver_result"]
    slacks = _load_selected(slack_path, ("slacks",))["slacks"].astype(np.float64)
    violation = float(np.max(np.abs(y - prediction) - slacks))
    if (solver_result.get("success") is not True or solver_result.get("status") != 0
            or slacks.shape != y.shape or not np.all(np.isfinite(slacks))
            or np.any(slacks < -1e-8) or violation > 1e-7
            or not math.isclose(loss, float(solver_result["objective_unweighted_sum_absolute"]),
                                rel_tol=1e-8, abs_tol=1e-7)
            or not math.isclose(float(slacks.sum()), float(solver_result["solver_objective"]),
                                rel_tol=1e-8, abs_tol=1e-7)
            or not math.isclose(loss, float(solver_result["solver_objective"]),
                                rel_tol=1e-8, abs_tol=1e-7)):
        raise r2.GateError(f"Independent verifier LP objective/constraints failed: {fit_id}")
    cycles = fit["cycles"].astype(np.int64)
    origins = fit["prediction_origins"].astype(np.int64)
    expected_mask = ((origins == 67) & (cycles >= 68) & (cycles <= 71)
                     if fit_id == "HINGE_D1" else
                     (origins == 67) & (cycles >= 68) & (cycles <= 75))
    if not np.all(expected_mask) or len(y) != record["fit_rows"] or len(y) != frozen_rows:
        raise r2.GateError(f"Independent verifier causal fit boundary failed: {fit_id}")
    resources = _verify_resource_snapshot(f"{fit_id}.resources_after_solver",
        state.get("resources_after_solver", {}), initial=False,
        expected_boundary=f"{fit_id}_fit_boundary_3")
    return {"fit_id": fit_id, "fit_rows": len(y),
            "objective_replayed": loss, "max_primal_violation": violation,
            "slopes_in_bounds": True, "fit_input_sha256": record["fit_input_sha256"],
            "slack_artifact_sha256": record["solver_slacks_sha256"],
            "post_solver_resource_snapshot": resources}


def _verify_score(record: Mapping[str, Any], fit_state: Mapping[str, Any],
                  freeze: Mapping[str, Any], identity: Mapping[str, Any],
                  targets: np.ndarray) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    path = _resolve(record["score_row_bundle_path"])
    if not path.is_file() or r2.file_sha256(path) != record["score_row_bundle_sha256"]:
        raise r2.GateError(f"Independent verifier score row hash mismatch: {record['fit_id']}")
    arrays = _load_selected(path, ("physical_rows", "record_ids", "prediction_origins",
        "original_bundle_indices", "cycles", "direct_predictions", "missing_mask", "targets",
        "raw_direct", "hinge_present_only", "causal_global_all_rows",
        "causal_global_present_only"))
    frozen_rows = _assert_frozen_stage_rows(arrays, identity, targets,
                                             str(record["fit_id"]), score=True)
    n = len(arrays["targets"])
    if any(value.ndim != 1 or len(value) != n for value in arrays.values()):
        raise r2.GateError("Independent verifier score arrays are misaligned")
    fit_id = record["fit_id"]
    cycles = arrays["cycles"].astype(np.int64)
    origins = arrays["prediction_origins"].astype(np.int64)
    score_mask = ((origins == 67) & (cycles >= 72) & (cycles <= 75)
                  if fit_id == "HINGE_D1" else
                  (origins == 75) & (cycles >= 76) & (cycles <= 79))
    expected_rows = 7_462 if fit_id == "HINGE_D1" else 8_491
    if (not np.all(score_mask) or n != expected_rows or n != frozen_rows
            or len(np.unique(arrays["record_ids"])) != n):
        raise r2.GateError(f"Independent verifier causal score boundary failed: {fit_id}")
    p = arrays["direct_predictions"].astype(np.float64)
    y = arrays["targets"].astype(np.float64)
    missing = arrays["missing_mask"].astype(bool)
    if not np.array_equal(arrays["raw_direct"].view("<u8"), p.astype("<f8").view("<u8")):
        raise r2.GateError("Score raw direct artifact differs from stitched DIRECT predictions")
    a, sl, sh = _params(fit_state)
    mapped = a + sl * np.minimum(p, 50.0) + sh * np.maximum(p - 50.0, 0.0)
    hinge_route = np.clip(mapped, 0.0, 100.0)
    hinge_route[missing] = p[missing]
    global_a, global_b, global_sha = r2._load_causal_global(
        "D1" if fit_id == "HINGE_D1" else "D2", freeze)
    global_all = np.clip(global_a + global_b * p, 0.0, 100.0)
    global_present = global_all.copy()
    global_present[missing] = p[missing]
    for name, actual in (("hinge_present_only", hinge_route),
                         ("causal_global_all_rows", global_all),
                         ("causal_global_present_only", global_present)):
        if not np.array_equal(arrays[name].astype("<f8").view("<u8"), actual.astype("<f8").view("<u8")):
            raise r2.GateError(f"Independent verifier prediction replay differs: {fit_id}/{name}")
    if not np.array_equal(arrays["hinge_present_only"][missing].view("<u8"), p[missing].astype("<f8").view("<u8")):
        raise r2.GateError("Independent verifier missing DIRECT fallback is not bit exact")
    metrics = _metrics_from_rows(arrays)
    _assert_close_tree(record["score_metrics"], metrics, f"{fit_id}.score_metrics")
    gain = (metrics["causal_global_present_only"]["overall"]["mae"]
            - metrics["hinge_present_only"]["overall"]["mae"])
    if not math.isclose(gain, float(record["gain_vs_causal_global_present_only"]),
                        rel_tol=1e-10, abs_tol=1e-12):
        raise r2.GateError(f"Independent verifier window gain differs: {fit_id}")
    subgroup_harm = _independent_subgroup_harm(metrics)
    _assert_close_tree(record["supported_subgroup_harm"], subgroup_harm,
                       f"{fit_id}.supported_subgroup_harm")
    return {"fit_id": fit_id, "score_rows": n, "causal_global_state_sha256": global_sha,
            "score_bundle_sha256": record["score_row_bundle_sha256"],
            "metrics_replayed": True, "missing_fallback_bit_exact": True,
            "subgroup_harm_replayed": True}, arrays, subgroup_harm


def _independent_decision(records: list[Mapping[str, Any]], pooled: Mapping[str, Any]) -> dict[str, Any]:
    gains = [float(record["gain_vs_causal_global_present_only"]) for record in records]
    control = float(pooled["causal_global_present_only"]["overall"]["mae"])
    candidate = float(pooled["hinge_present_only"]["overall"]["mae"])
    pooled_gain, pooled_harm = control - candidate, candidate - control
    supported_harms = []
    for record in records:
        for group in record["supported_subgroup_harm"]["groups"].values():
            if group["status"] == "SUPPORTED":
                supported_harms.append(float(group["harm"]))
    max_subgroup = max(supported_harms) if supported_harms else None
    subgroup_ok = max_subgroup is None or max_subgroup <= 0.25
    qualified = pooled_gain >= 0.02 and all(value >= 0 for value in gains) and subgroup_ok
    exploratory = (not qualified and (pooled_gain >= 0 or
        (max(gains) >= 0.02 and pooled_harm <= 0.01))
        and all(-value <= 0.03 for value in gains) and subgroup_ok)
    return {"pooled_gain_vs_causal_global_present_only": pooled_gain,
        "window_gains_vs_causal_global_present_only": gains,
        "pooled_harm": pooled_harm, "window_harms": [-value for value in gains],
        "maximum_supported_subgroup_harm": max_subgroup,
        "qualified_gate": bool(qualified),
        "exploratory_root_review_gate": bool(exploratory),
        "status": "QUALIFIED" if qualified else
                  "EXPLORATORY_ROOT_REVIEW" if exploratory else "RETIRE"}


def verify_diagnostics(report_path: Path, *, freeze: Mapping[str, Any],
                       approval_sha: str) -> dict[str, Any]:
    _live_freeze, _approval, live_approval_sha = r2.verify_source_pass()
    if live_approval_sha != approval_sha:
        raise r2.GateError("Independent verifier ROOT_SOURCE_PASS changed")
    if dict(_live_freeze) != dict(freeze):
        raise r2.GateError("Independent verifier was passed a stale identity freeze")
    report_path = Path(report_path)
    if not report_path.is_file():
        raise r2.GateError("R2 diagnostic report is missing")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (report.get("status") != "TWO_DIAGNOSTICS_COMPLETE_PENDING_ROOT_REVIEW"
            or report.get("experiment_id") != r2.EXPERIMENT_ID
            or report.get("prefit_freeze_sha256") != r2.file_sha256(r2.PREFIT_FREEZE)
            or report.get("root_source_pass_sha256") != approval_sha
            or report.get("source_hashes") != freeze["source_hashes"]
            or report.get("input_hashes") != freeze["input_hashes"]
            or report.get("test_or_sample_accessed") is not False
            or report.get("csv_created") is not False
            or report.get("kaggle_upload") is not False
            or report.get("final_fit_started") is not False):
        raise r2.GateError("Independent verifier diagnostic report identity/boundary failed")
    resource_check = _verify_resource_receipt(report, approval_sha)
    identity = r2.load_oof_identity(_live_freeze)
    targets, target_audit = r2.load_targets_after_source_pass(identity, approval_sha, _live_freeze)
    if report.get("train_target_audit") != target_audit:
        raise r2.GateError("Independent verifier Train target audit differs from selected frozen rows")
    records = report.get("stages", [])
    if [record.get("fit_id") for record in records] != list(r2.DIAGNOSTIC_FIT_IDS):
        raise r2.GateError("Independent verifier expected exactly HINGE_D1 and HINGE_D2")
    fit_checks, score_checks, row_bundles, verified_records = [], [], [], []
    for record in records:
        fit_path = _resolve(record["fit_state_path"])
        fit_state = json.loads(fit_path.read_text(encoding="utf-8"))
        fit_check = _verify_fit(record, _live_freeze, approval_sha, identity, targets)
        fit_checks.append(fit_check)
        score_check, arrays, subgroup_harm = _verify_score(
            record, fit_state, _live_freeze, identity, targets)
        score_checks.append(score_check)
        row_bundles.append(arrays)
        verified_records.append({**record, "supported_subgroup_harm": subgroup_harm})
    pooled_names = ("record_ids", "cycles", "missing_mask", "targets", "raw_direct",
        "causal_global_all_rows", "causal_global_present_only", "hinge_present_only")
    pooled_arrays = {name: np.concatenate([bundle[name] for bundle in row_bundles])
                     for name in pooled_names}
    if (len(pooled_arrays["record_ids"]) != r2.EXPECTED_POOLED_ROWS
            or len(np.unique(pooled_arrays["record_ids"])) != r2.EXPECTED_POOLED_ROWS):
        raise r2.GateError("Independent verifier pooled IDs/denominator are not 15,953 unique rows")
    pooled_metrics = {name: _policy_metrics(pooled_arrays["targets"], pooled_arrays["cycles"],
        pooled_arrays["missing_mask"], pooled_arrays[name]) for name in
        ("raw_direct", "causal_global_all_rows", "causal_global_present_only", "hinge_present_only")}
    if report["pooled_unique_evaluation"].get("rows") != EXPECTED_POOLED_ROWS:
        raise r2.GateError("Report pooled denominator differs from 15,953")
    _assert_close_tree(report["pooled_unique_evaluation"]["metrics"], pooled_metrics, "pooled.metrics")
    decision = _independent_decision(verified_records, pooled_metrics)
    _assert_close_tree(report["decision"], decision, "decision")
    ledger = r2.read_ledger()
    if (sum(bool(entry["LP_STARTED"]) for entry in ledger.values()) != 2
            or any(not ledger.get(fit_id, {}).get("LP_COMPLETE") for fit_id in r2.DIAGNOSTIC_FIT_IDS)
            or ledger.get("HINGE_FINAL", {}).get("LP_STARTED")
            or set(ledger) != set(r2.DIAGNOSTIC_FIT_IDS)):
        raise r2.GateError("Independent verifier cap3 ledger does not prove two diagnostic completions")
    for record in records:
        fit_id = record["fit_id"]
        events = ledger[fit_id]
        if (len(events["RESERVE"]) != 1 or len(events["LP_STARTED"]) != 1
                or len(events["LP_COMPLETE"]) != 1 or events["LP_FAILED"]):
            raise r2.GateError(f"Independent verifier ledger terminal sequence failed: {fit_id}")
        _verify_resource_snapshot(f"{fit_id}.RESERVE",
            events["RESERVE"][0].get("resource", {}), initial=True,
            expected_boundary=f"{fit_id}_fit_boundary_1")
        _verify_resource_snapshot(f"{fit_id}.LP_STARTED",
            events["LP_STARTED"][0].get("resource", {}), initial=True,
            expected_boundary=f"{fit_id}_fit_boundary_2")
        if events["LP_COMPLETE"][0].get("state_sha256") != record["fit_state_sha256"]:
            raise r2.GateError(f"Independent verifier ledger completion state hash differs: {fit_id}")
    return {"schema_version": 1, "experiment_id": r2.EXPERIMENT_ID,
        "status": "PASS_INDEPENDENT_DIAGNOSTICS_REPLAY",
        "diagnostics_sha256": r2.file_sha256(report_path),
        "prefit_freeze_sha256": r2.file_sha256(r2.PREFIT_FREEZE),
        "root_source_pass_sha256": approval_sha,
        "fit_replays": fit_checks, "score_replays": score_checks,
        "resource_receipt_replayed": resource_check,
        "pooled_unique_rows": EXPECTED_POOLED_ROWS,
        "decision_replayed": True, "official_lp_starts": 2,
        "official_lp_start_cap": r2.MAX_LP_STARTS,
        "test_or_sample_accessed": False, "csv_created": False,
        "kaggle_upload": False}


def verify_candidate_csv(path: Path, *, expected_ids: Any,
                         expected_values: Any, missing_mask: Any,
                         raw_direct: Any, production: bool = True) -> dict[str, Any]:
    ids = np.asarray(expected_ids, dtype=str)
    values = np.asarray(expected_values, dtype="<f8")
    missing = np.asarray(missing_mask, dtype=bool)
    raw = np.asarray(raw_direct, dtype="<f8")
    if not (ids.ndim == values.ndim == missing.ndim == raw.ndim == 1
            and len(ids) == len(values) == len(missing) == len(raw)):
        raise r2.GateError("CSV verifier vectors are not aligned")
    if not len(ids) or len(np.unique(ids)) != len(ids):
        raise r2.GateError("CSV official IDs must be nonempty and unique")
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["record_id", "threat_rank"]:
            raise r2.GateError("CSV schema must be exactly record_id,threat_rank")
        rows = list(reader)
    got_ids = np.asarray([row["record_id"] for row in rows], dtype=str)
    try:
        got_values = np.asarray([float(row["threat_rank"]) for row in rows], dtype="<f8")
    except (KeyError, TypeError, ValueError) as exc:
        raise r2.GateError("CSV threat_rank values are not numeric") from exc
    if len(got_ids) != len(ids) or not np.array_equal(got_ids, ids):
        raise r2.GateError("CSV ordered official IDs or row count differ")
    if not np.all(np.isfinite(got_values)) or np.any((got_values < 0) | (got_values > 100)):
        raise r2.GateError("CSV threat_rank must be finite and in [0,100]")
    if not np.array_equal(got_values.view("<u8"), values.view("<u8")):
        raise r2.GateError("CSV route values differ from saved output array")
    if not np.array_equal(got_values[missing].view("<u8"), raw[missing].view("<u8")):
        raise r2.GateError("CSV missing values are not bit-exact raw DIRECT")
    if production and (len(ids) != r2.EXPECTED_TEST_ROWS
                       or int(missing.sum()) != r2.EXPECTED_MISSING_TEST_ROWS):
        raise r2.GateError("Production CSV row or missing count differs from freeze")
    return {"rows": int(len(ids)), "missing_rows": int(missing.sum()),
            "csv_sha256": r2.file_sha256(Path(path))}


def run_synthetic_checks() -> dict[str, Any]:
    checks: dict[str, Any] = {}
    p = np.linspace(0.0, 100.0, 1_200, dtype=np.float64)
    truth = {"intercept": 7.0, "slope_low": 0.4, "slope_high": 1.15}
    y = truth["intercept"] + truth["slope_low"] * np.minimum(p, 50.0) \
        + truth["slope_high"] * np.maximum(p - 50.0, 0.0)
    fitted = r2.solve_hinge_l1(p, y, purpose="diagnostic")
    error = max(abs(fitted[name] - value) for name, value in truth.items())
    checks["known_hinge_recovery"] = {"pass": error < 1e-7, "max_parameter_error": error}

    grid = np.linspace(0.0, 100.0, 501)
    equal = r1.predict_hinge(grid, r1.HingeParameters(2.0, 0.7, 0.7))
    affine = 2.0 + 0.7 * grid
    seam = r1.predict_hinge(np.array([50.0]), r1.HingeParameters(2.0, 0.7, 1.4))[0]
    bound_cases = 0
    for invalid in (r1.HingeParameters(0.0, -0.01, 1.0),
                    r1.HingeParameters(0.0, 1.0, 2.01)):
        try:
            r1.validate_parameters(invalid)
        except r1.HingeError:
            bound_cases += 1
    monotone = r1.predict_hinge(grid, r1.HingeParameters(-2.0, 0.0, 2.0))
    checks["hinge_shape_and_bounds"] = {
        "pass": bool(np.allclose(equal, affine, rtol=0.0, atol=1e-12)
                      and seam == 37.0 and np.all(np.diff(monotone) >= -1e-12)
                      and bound_cases == 2),
        "bound_violations_rejected": bound_cases}

    has_weights_argument = "weights" in inspect.signature(r2.solve_hinge_l1).parameters
    expected_loss = float(np.abs(y - (fitted["intercept"]
        + fitted["slope_low"] * np.minimum(p, 50.0)
        + fitted["slope_high"] * np.maximum(p - 50.0, 0.0))).sum())
    checks["unweighted_l1_no_weights"] = {
        "pass": (not has_weights_argument and math.isclose(expected_loss,
            fitted["objective_unweighted_sum_absolute"], rel_tol=1e-9, abs_tol=1e-7)),
        "weights_argument_present": has_weights_argument}

    resource = {
        "initial_below_2g_rejected": not r2.resource_decision(2 * 1024**3 - 1, initial=True)["pass"],
        "initial_at_2g_accepted": r2.resource_decision(2 * 1024**3, initial=True)["pass"],
        "actual_below_1g_rejected": not r2.resource_decision(1024**3 - 1)["pass"],
        "actual_at_1g_accepted": r2.resource_decision(1024**3)["pass"],
    }
    checks["user_revised_ram_boundaries"] = {"pass": all(resource.values()), **resource}

    target_loader_calls: list[str] = []
    old_gate, old_reader = r2.verify_source_pass, r2.read_scored_query
    try:
        r2.verify_source_pass = lambda: (_ for _ in ()).throw(r2.GateError("no synthetic ROOT_SOURCE_PASS"))
        r2.read_scored_query = lambda *a, **k: target_loader_calls.append("read")
        blocked_before_labels = False
        try:
            r2.load_targets_after_source_pass({"rows": np.array([1]), "ids": np.array(["x"]),
                "cycles": np.array([68])}, "0" * 64, {})
        except r2.GateError:
            blocked_before_labels = True
    finally:
        r2.verify_source_pass, r2.read_scored_query = old_gate, old_reader
    checks["source_pass_precedes_target_loader"] = {
        "pass": blocked_before_labels and not target_loader_calls,
        "target_loader_calls": len(target_loader_calls)}

    # Exercise the actual stage ledger path with a fake deterministic solver.
    n = 1_000
    fit_p = np.linspace(0.0, 100.0, n, dtype="<f8")
    fit_y = 4.0 + 0.6 * np.minimum(fit_p, 50.0) + 0.9 * np.maximum(fit_p - 50.0, 0.0)
    fit_arrays = {"physical_rows": np.arange(n, dtype=np.int64),
        "record_ids": np.asarray([f"syn-{i}" for i in range(n)]),
        "prediction_origins": np.full(n, 67, dtype=np.int8),
        "original_bundle_indices": np.arange(n, dtype=np.int64),
        "cycles": np.resize(np.array([68, 69, 70, 71], dtype=np.int16), n),
        "direct_predictions": fit_p, "missing_mask": np.zeros(n, dtype=bool),
        "targets": np.asarray(fit_y, dtype="<f8")}
    solver_calls = []
    def fake_solver(prediction: Any, targets: Any) -> dict[str, Any]:
        solver_calls.append("called")
        ledger = r2.read_ledger(active_ledger)
        if not ledger.get("HINGE_D1", {}).get("LP_STARTED"):
            raise AssertionError("solver invoked before durable LP_STARTED")
        pred = 4.0 + 0.6 * np.minimum(prediction, 50.0) + 0.9 * np.maximum(prediction - 50.0, 0.0)
        slack = np.abs(np.asarray(targets) - pred)
        return {"success": True, "status": 0, "intercept": 4.0,
            "slope_low": 0.6, "slope_high": 0.9,
            "objective_unweighted_sum_absolute": float(slack.sum()),
            "solver_objective": float(slack.sum()), "_slacks": slack}

    fit_temp = tempfile.TemporaryDirectory(prefix="hinge-r2-stage-")
    active_ledger = Path(fit_temp.name) / "lp_ledger.jsonl"
    run_dir = Path(fit_temp.name) / "run"
    input_path = run_dir / "fit_inputs/HINGE_D1.npz"
    input_sha = r2.write_npz_once(input_path, fit_arrays)
    identity = {"fit_id": "HINGE_D1", "fit_rows": n,
        "fit_input_bundle_sha256": input_sha,
        "source_commit": "synthetic-commit", "source_hashes": {"synthetic": "1" * 64},
        "input_hashes": {"synthetic": {"sha256": "2" * 64}}}
    freeze = {"source_commit": "synthetic-commit"}
    approval = "3" * 64
    crash = False
    def crash_after_publish() -> None:
        nonlocal crash
        crash = True
        raise RuntimeError("synthetic interruption after fit-state publication")
    try:
        try:
            r2.run_fit_slot("HINGE_D1", fit_arrays, expected_identity=identity,
                approval_sha=approval, freeze=freeze, ledger_path=active_ledger,
                run_dir=run_dir, solver=fake_solver,
                approval_check=lambda: approval,
                resource_check=lambda: {"status": "PASS", "process_rss_bytes": 1},
                process_check=lambda: None, after_publish=crash_after_publish)
        except RuntimeError as exc:
            if "synthetic interruption" not in str(exc):
                raise
        recovered = r2.run_fit_slot("HINGE_D1", fit_arrays,
            expected_identity=identity, approval_sha=approval, freeze=freeze,
            ledger_path=active_ledger, run_dir=run_dir,
            solver=lambda *_: (_ for _ in ()).throw(AssertionError("duplicate solver call")),
            approval_check=lambda: approval,
            resource_check=lambda: {"status": "PASS"}, process_check=lambda: None)
        completed = r2.read_ledger(active_ledger)["HINGE_D1"]["LP_COMPLETE"]
        checks["actual_stage_postpublish_recovery"] = {
            "pass": crash and len(solver_calls) == 1 and recovered["status"] == "LP_COMPLETE"
                    and bool(completed) and completed[0]["recovered_after_state_publish"],
            "solver_calls": len(solver_calls), "recovered": bool(completed)}
    finally:
        fit_temp.cleanup()

    # An LP_STARTED with no state is ambiguous and blocks the real stage wrapper.
    ambiguous_temp = tempfile.TemporaryDirectory(prefix="hinge-r2-ambiguous-")
    ambiguous_ledger = Path(ambiguous_temp.name) / "lp_ledger.jsonl"
    ambiguous_run = Path(ambiguous_temp.name) / "run"
    amb_input = r2.write_npz_once(ambiguous_run / "fit_inputs/HINGE_D1.npz", fit_arrays)
    amb_identity = {**identity, "fit_input_bundle_sha256": amb_input}
    ambiguous_calls = []
    def killed_solver(*_args: Any) -> Any:
        ambiguous_calls.append("called")
        raise SystemExit("synthetic abrupt interruption after LP_STARTED")
    try:
        try:
            r2.run_fit_slot("HINGE_D1", fit_arrays, expected_identity=amb_identity,
                approval_sha=approval, freeze=freeze, ledger_path=ambiguous_ledger,
                run_dir=ambiguous_run, solver=killed_solver,
                approval_check=lambda: approval,
                resource_check=lambda: {"status": "PASS"}, process_check=lambda: None)
        except SystemExit:
            pass
        blocked = False
        try:
            r2.run_fit_slot("HINGE_D1", fit_arrays, expected_identity=amb_identity,
                approval_sha=approval, freeze=freeze, ledger_path=ambiguous_ledger,
                run_dir=ambiguous_run,
                solver=lambda *_: (_ for _ in ()).throw(AssertionError("retry attempted")),
                approval_check=lambda: approval,
                resource_check=lambda: {"status": "PASS"}, process_check=lambda: None)
        except r2.GateError:
            blocked = True
        checks["ambiguous_started_no_state_blocks_retry"] = {
            "pass": blocked and len(ambiguous_calls) == 1,
            "solver_calls": len(ambiguous_calls), "blocked": blocked}
    finally:
        ambiguous_temp.cleanup()

    values = np.array([2.0, 20.5, 100.0], dtype="<f8")
    checks["producer_hash_schemas"] = {"pass": r1.raw_direct_f64_sha256(values)
                                       != r1.tagged_array_sha256(values)}

    with tempfile.TemporaryDirectory(prefix="hinge-r2-csv-") as temporary:
        # Exercise the production writer's frozen row-count/fallback contract
        # with synthetic values and IDs, without touching Test/Sample.
        ids = np.array([f"SYN{i:05d}" for i in range(r2.EXPECTED_TEST_ROWS)])
        raw = np.resize(np.array([10.0, 50.0, -0.0, 20.0], dtype="<f8"), len(ids))
        missing = np.zeros(len(ids), dtype=bool)
        missing[:r2.EXPECTED_MISSING_TEST_ROWS] = True
        candidate = raw + 2.0
        candidate[missing] = raw[missing]
        valid_path = Path(temporary) / "valid.csv"
        r2.write_candidate_csv(valid_path, ids, candidate, missing_mask=missing, raw_direct=raw)
        valid = verify_candidate_csv(valid_path, expected_ids=ids,
            expected_values=candidate, missing_mask=missing, raw_direct=raw)
        rejected_headers = 0
        for header in (("record_id", "prediction"), ("record_id", "artifact_signal")):
            bad_path = Path(temporary) / f"bad_{header[1]}.csv"
            with bad_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream, lineterminator="\n")
                writer.writerow(header)
                writer.writerows(zip(ids, candidate, strict=True))
            try:
                verify_candidate_csv(bad_path, expected_ids=ids,
                    expected_values=candidate, missing_mask=missing, raw_direct=raw)
            except r2.GateError:
                rejected_headers += 1
        checks["official_threat_rank_csv_schema"] = {
            "pass": (valid["rows"] == r2.EXPECTED_TEST_ROWS
                     and valid["missing_rows"] == r2.EXPECTED_MISSING_TEST_ROWS
                     and rejected_headers == 2),
            "right_schema_rows": valid["rows"], "wrong_headers_rejected": rejected_headers}

    live_resource = r2.resource_snapshot(boundary="synthetic_test")
    peak_is_consistent = ((live_resource["os_peak_working_set_bytes"] is None
                           and live_resource["os_peak_working_set_measured"] is False)
                          or (isinstance(live_resource["os_peak_working_set_bytes"], int)
                              and live_resource["os_peak_working_set_measured"] is True))
    checks["rss_and_os_peak_measurement_flags"] = {
        "pass": live_resource["process_rss_measured"] is True and peak_is_consistent,
        "peak_metric_supported": live_resource["os_peak_working_set_measured"]}

    contract = _resource_contract()
    synthetic_resource = {
        "status": "PASS", "boundary": "synthetic_resource_boundary",
        "available_bytes_measured": 3 * 1024**3,
        "initial_floor_pass": True,
        "minimum_initial_available_bytes": contract["minimum_available_bytes"],
        "incremental_estimate_bytes": contract["incremental_estimate_bytes"],
        "minimum_projected_free_bytes": contract["minimum_projected_free_bytes"],
        "actual_available_boundary_minimum_bytes": contract["minimum_actual_free_at_boundary_bytes"],
        "projected_free_bytes": 3 * 1024**3 - contract["incremental_estimate_bytes"],
        "projected_free_pass": True, "actual_boundary_pass": True,
        "estimated_peak_is_measured": False,
        "process_rss_bytes": 128 * 1024**2, "process_rss_measured": True,
        "os_peak_working_set_bytes": 256 * 1024**2,
        "os_peak_working_set_measured": True,
        "threads": 2, "concurrent_heavy_jobs": 1,
    }
    _verify_resource_snapshot("synthetic_resource_boundary", synthetic_resource, initial=True)
    tampered_resource = {**synthetic_resource, "os_peak_working_set_bytes": None}
    rejected_measurement = False
    try:
        _verify_resource_snapshot("tampered_peak", tampered_resource, initial=True)
    except r2.GateError:
        rejected_measurement = True
    checks["independent_resource_replay"] = {
        "pass": rejected_measurement,
        "thresholds_checked": contract,
        "inconsistent_peak_rejected": rejected_measurement}

    synthetic_identity = {
        "rows": np.array([10, 11, 12, 13], dtype=np.int64),
        "ids": np.array(["a", "b", "c", "d"]),
        "origins": np.array([67, 67, 67, 75], dtype=np.int8),
        "indices": np.array([0, 1, 2, 3], dtype=np.int64),
        "cycles": np.array([68, 71, 72, 76], dtype=np.int16),
        "direct": np.array([10.0, 20.0, 30.0, 40.0], dtype="<f8"),
        "missing": np.array([False, True, False, False]),
    }
    synthetic_targets = np.array([1.0, 2.0, 3.0, 4.0], dtype="<f8")
    expected_fit_rows = _expected_stage_rows(
        synthetic_identity, synthetic_targets, "HINGE_D1", score=False)
    frozen_rows_ok = _assert_frozen_stage_rows(
        expected_fit_rows, synthetic_identity, synthetic_targets, "HINGE_D1", score=False) == 2
    changed_targets = {name: value.copy() for name, value in expected_fit_rows.items()}
    changed_targets["targets"][0] += 0.1
    changed_rows_rejected = False
    try:
        _assert_frozen_stage_rows(
            changed_targets, synthetic_identity, synthetic_targets, "HINGE_D1", score=False)
    except r2.GateError:
        changed_rows_rejected = True
    checks["independent_frozen_row_target_replay"] = {
        "pass": frozen_rows_ok and changed_rows_rejected,
        "fit_rows": 2, "tampered_target_rejected": changed_rows_rejected}

    def synthetic_group(mae: float) -> dict[str, Any]:
        return {"status": "SUPPORTED", "n": 250, "mae": mae}
    subgroup = _independent_subgroup_harm({
        "hinge_present_only": {"groups": {name: synthetic_group(value) for name, value in
            (("present", 10.2), ("missing", 12.0), ("target_le_5", 8.1), ("target_ge_95", 14.0))}},
        "causal_global_present_only": {"groups": {name: synthetic_group(value) for name, value in
            (("present", 10.0), ("missing", 12.1), ("target_le_5", 8.0), ("target_ge_95", 13.8))}},
    })
    checks["independent_subgroup_harm_replay"] = {
        "pass": math.isclose(subgroup["maximum_supported_harm"], 0.2, abs_tol=1e-12),
        "maximum_supported_harm": subgroup["maximum_supported_harm"]}

    failures = [name for name, value in checks.items() if not value["pass"]]
    if failures:
        raise AssertionError("Synthetic R2 QA failures: " + ", ".join(failures))
    return checks


def write_synthetic_qa() -> dict[str, Any]:
    checks = run_synthetic_checks()
    sources = {}
    for path in (r2.SCRIPT, Path(__file__).resolve(), r2.TESTS, r2.PROTOCOL):
        sources[path.relative_to(r2.ROOT).as_posix()] = r2.file_sha256(path)
    report = {"schema_version": 2, "experiment_id": r2.EXPERIMENT_ID,
        "status": "PASS_SYNTHETIC_ONLY", "checks": checks,
        "source_hashes": sources,
        "resource_policy_sha256": r2.file_sha256(r2.RESOURCE_POLICY_FILE),
        "candidate_data_read": False, "official_train_labels_read": False,
        "test_or_sample_read": False, "official_lp_starts": 0,
        "synthetic_lp_starts": 1, "base_or_preprocessing_fits": 0}
    path = r2.SYNTHETIC_QA
    payload = json.dumps(report, sort_keys=True, ensure_ascii=False,
                         allow_nan=False, indent=2).encode("utf-8") + b"\n"
    if path.exists():
        if path.read_bytes() != payload:
            raise r2.GateError("Existing R2 synthetic QA differs; refusing overwrite")
        digest = r2.file_sha256(path)
    else:
        digest = r2.write_json_once(path, report)
    return {"status": report["status"], "path": str(path),
            "sha256": digest, "checks": len(checks), "source_hashes": sources}
