"""Independent post-fit replay for the approved present-only R06 hinge diagnostics."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from scripts import reg_r06_present_native_hinge_v1_r2 as pilot
from scripts import verify_reg_r06_present_native_hinge_v1_r2 as source_verify

ROOT = Path(__file__).resolve().parents[1]
REPORT = pilot.REPORT_DIR / "DIAGNOSTICS_R2.json"
OUTPUT = pilot.REPORT_DIR / "DIAGNOSTICS_VERIFICATION_R2.json"
LEDGER = pilot.LEDGER
RUN_DIR = pilot.RUN_DIR


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8", errors="surrogatepass")).hexdigest()


def _signed_json(path: Path) -> tuple[dict[str, Any], str]:
    body = json.loads(path.read_text(encoding="utf-8"))
    signature = body.pop("signature", None)
    _need(signature == _canonical_sha(body), f"Invalid signed JSON: {path}")
    body["signature"] = signature
    return body, _sha(path)


def _npz(path: Path, members: tuple[str, ...] | None = None) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if members is not None:
            _need(set(members).issubset(archive.files), f"Missing NPZ members in {path}")
            return {name: np.asarray(archive[name]).copy() for name in members}
        return {name: np.asarray(archive[name]).copy() for name in archive.files}


def _same_array(actual: np.ndarray, expected: np.ndarray, name: str) -> None:
    left, right = np.asarray(actual), np.asarray(expected)
    _need(left.shape == right.shape and left.dtype == right.dtype,
          f"Fit input shape or dtype differs for {name}: {left.shape}/{left.dtype} "
          f"!= {right.shape}/{right.dtype}")
    if left.dtype.kind == "f":
        equal = np.array_equal(left.view(np.uint8), right.view(np.uint8))
    else:
        equal = np.array_equal(left, right)
    _need(equal, f"Exact fit input differs for {name}")


def _metric(y: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    absolute = np.abs(error)
    return {
        "n": int(len(error)), "status": "MEASURED",
        "mae": float(np.mean(absolute)),
        "p90_absolute_error": float(np.quantile(absolute, 0.90, method="linear")),
        "mean_error_prediction_minus_target": float(np.mean(error)),
    }


def _summary_block(y: np.ndarray, predictions: Mapping[str, np.ndarray],
                   mask: np.ndarray, minimum: int, name: str) -> dict[str, Any]:
    selected = np.asarray(mask, dtype=bool)
    n = int(selected.sum())
    if n < minimum:
        return {"n": n, "supported": False, "status": "UNKNOWN",
                "support_rule": f"n >= {minimum} for {name}"}
    return {"n": n, "supported": True, "status": "MEASURED",
            "support_rule": f"n >= {minimum} for {name}",
            "metrics": {key: _metric(y[selected], value[selected])
                        for key, value in predictions.items()}}


def _diagnostic_summary(y: np.ndarray, predictions: Mapping[str, np.ndarray],
                        missing: np.ndarray, cycles: np.ndarray) -> dict[str, Any]:
    groups = {
        "availability_present": _summary_block(y, predictions, ~missing, 200,
                                                "availability_present"),
        "availability_missing": _summary_block(y, predictions, missing, 200,
                                                "availability_missing"),
        "target_le_5": _summary_block(y, predictions, y <= 5.0, 200, "target_le_5"),
        "target_ge_95": _summary_block(y, predictions, y >= 95.0, 200, "target_ge_95"),
    }
    for value in groups.values():
        if value["supported"]:
            candidate = value["metrics"]["present_native_hinge"]["mae"]
            primary = value["metrics"]["transported_r4_hinge_on_r06"]["mae"]
            value["candidate_harm_vs_primary"] = float(candidate - primary)
            value["candidate_gain_vs_primary"] = float(primary - candidate)
    per_cycle = {
        str(int(cycle)): _summary_block(y, predictions, cycles == cycle, 50,
                                         f"cycle_{int(cycle)}")
        for cycle in sorted(set(map(int, cycles)))
    }
    full = _summary_block(y, predictions, np.ones(len(y), dtype=bool), 1, "full rows")
    present = _summary_block(y, predictions, ~missing, 1, "present rows")
    return {
        "n": int(len(y)), "present_rows": int(np.sum(~missing)),
        "missing_rows": int(np.sum(missing)),
        "full_rows": full, "present_only": present,
        "availability_and_tail_groups": groups, "per_cycle": per_cycle,
        "all_required_groups_supported": all(item["supported"] for item in groups.values()),
        "all_scored_cycles_supported": all(item["supported"] for item in per_cycle.values()),
    }


def _coefficients(state: Mapping[str, Any], label: str) -> dict[str, float]:
    result = state["solver_result"]
    values = {key: float(result[key])
              for key in ("intercept", "slope_low", "slope_high", "knot")}
    _need(all(math.isfinite(value) for value in values.values()),
          f"Non-finite coefficients in {label}")
    _need(values["knot"] == 50.0 and 0.0 <= values["slope_low"] <= 2.0
          and 0.0 <= values["slope_high"] <= 2.0,
          f"Knot or slope bounds differ in {label}")
    return values


def _apply(raw: np.ndarray, missing: np.ndarray,
           coefficients: Mapping[str, float]) -> np.ndarray:
    p = np.asarray(raw, dtype=np.float64)
    missing = np.asarray(missing, dtype=bool)
    result = (coefficients["intercept"]
              + coefficients["slope_low"] * np.minimum(p, 50.0)
              + coefficients["slope_high"] * np.maximum(p - 50.0, 0.0))
    result = np.clip(result, 0.0, 100.0)
    result[missing] = p[missing]
    return result.astype("<f8", copy=False)


def _close_tree(expected: Any, actual: Any, path: str = "diagnostics") -> None:
    if isinstance(expected, Mapping):
        _need(isinstance(actual, Mapping) and set(expected) == set(actual),
              f"Independent replay structure differs at {path}")
        for key in expected:
            _close_tree(expected[key], actual[key], f"{path}.{key}")
    elif isinstance(expected, list):
        _need(isinstance(actual, list) and len(expected) == len(actual),
              f"Independent replay list differs at {path}")
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            _close_tree(left, right, f"{path}[{index}]")
    elif isinstance(expected, (int, bool, str)) or expected is None:
        _need(type(expected) is type(actual) and expected == actual,
              f"Independent replay value differs at {path}")
    else:
        _need(math.isclose(float(expected), float(actual), rel_tol=1e-10, abs_tol=1e-12),
              f"Independent replay number differs at {path}: {expected} != {actual}")


def _load_frozen_comparator_state(freeze: Mapping[str, Any], name: str) -> dict[str, Any]:
    item = freeze["input_hashes"][name]
    path = Path(item["path"])
    _need(path.is_file() and _sha(path) == item["sha256"],
          f"Frozen comparator state hash differs: {name}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_identity_and_targets(freeze: Mapping[str, Any], preflight: Mapping[str, Any],
                               fit_population: Mapping[str, Any],
                               root_pass_sha: str) -> dict[str, Any]:
    # Reconstruct cache alignment independently from the fit runner; the source verifier
    # also confirms the frozen population ID artifact before target values are read.
    cache_audit = source_verify._check_row_cache_alignment(
        dict(freeze), dict(preflight), dict(fit_population))
    label_item = freeze["input_hashes"]["oof_label_and_identity_bundle"]
    label_path = Path(label_item["path"])
    _need(_sha(label_path) == label_item["sha256"], "OOF label/identity bundle hash differs")
    label_arrays = _npz(label_path, ("physical_rows", "record_ids", "prediction_origins",
                                    "original_bundle_indices", "cycles", "missing_mask",
                                    "direct_predictions", "targets"))
    ids = label_arrays["record_ids"].astype(str)
    physical_rows = label_arrays["physical_rows"].astype(np.int64)
    origins = label_arrays["prediction_origins"].astype(np.int16)
    bundle_indices = label_arrays["original_bundle_indices"].astype(np.int64)
    cycles = label_arrays["cycles"].astype(np.int16)
    missing = label_arrays["missing_mask"].astype(bool)
    direct = label_arrays["direct_predictions"].astype("<f8")
    targets = label_arrays["targets"].astype("<f8")
    _need(len(ids) == pilot.EXPECTED_ROWS and len(set(ids.tolist())) == len(ids),
          "OOF record ID population is not unique and complete")
    _need(len(set(physical_rows.tolist())) == len(physical_rows),
          "OOF physical row population is not unique")
    _need(targets.shape == ids.shape and np.all(np.isfinite(targets))
          and np.all((targets >= 0.0) & (targets <= 100.0)),
          "Authorized OOF targets are misaligned or outside [0,100]")
    raw = np.full(len(ids), np.nan, dtype="<f8")
    raw_origin = np.full(len(ids), -1, dtype=np.int16)
    id_index = {record_id: index for index, record_id in enumerate(ids.tolist())}
    for origin, cycle_member, allowed in (
            (67, "audit__cycle_code", np.arange(68, 76)),
            (75, "score_cycles", np.arange(76, 80))):
        item = freeze["input_hashes"][f"r06_fit{origin}_npz"]
        path = Path(item["path"])
        _need(_sha(path) == item["sha256"], f"R06 fit{origin} cache hash differs")
        archive = _npz(path, ("score_rows", "score_record_ids", "prediction", cycle_member))
        cache_ids = archive["score_record_ids"].astype(str)
        cache_rows = archive["score_rows"].astype(np.int64)
        cache_cycles = archive[cycle_member].astype(np.int64)
        keep = np.isin(cache_cycles, allowed)
        actual = {(str(cache_ids[i]), int(cache_cycles[i]), int(cache_rows[i]))
                  for i in np.flatnonzero(keep)}
        expected = {(str(ids[i]), int(cycles[i]), int(physical_rows[i]))
                    for i in range(len(ids)) if int(cycles[i]) in set(allowed.tolist())}
        _need(actual == expected, f"R06 fit{origin} cache row identity differs")
        for cache_i in np.flatnonzero(keep):
            destination = id_index[str(cache_ids[cache_i])]
            _need(raw_origin[destination] == -1, "Duplicate R06 cache assignment")
            raw[destination] = archive["prediction"][cache_i]
            raw_origin[destination] = origin
    _need(np.all(np.isfinite(raw)) and np.array_equal(raw_origin, origins),
          "R06 cached prediction lineage differs from OOF row identity")
    _need(np.array_equal(direct[missing].view("<u8"), raw[missing].view("<u8")),
          "Missing DIRECT fallback is not raw R06 bit-exact")
    return {
        "ids": ids, "physical_rows": physical_rows, "origins": origins,
        "bundle_indices": bundle_indices, "cycles": cycles, "missing": missing,
        "direct": direct, "targets": targets, "raw": raw,
        "cache_audit": cache_audit, "root_pass_sha256": root_pass_sha,
    }


def _verify_fits(identity: Mapping[str, Any], freeze: Mapping[str, Any],
                 freeze_sha: str, root_pass_sha: str,
                 preflight_sha: str) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    ledger_events = [json.loads(line) for line in LEDGER.read_text(encoding="utf-8").splitlines()]
    expected_sequence = [
        (fit_id, event)
        for fit_id in pilot.DIAGNOSTIC_IDS
        for event in ("RESERVE", "LP_STARTED", "LP_COMPLETE")
    ]
    _need([(item.get("fit_id"), item.get("event")) for item in ledger_events]
          == expected_sequence, "R2 ledger is not exactly the two ordered D1/D2 completions")
    fit_audits: dict[str, Any] = {}
    coefficients: dict[str, dict[str, float]] = {}
    for fit_id, expected_count in zip(
            pilot.DIAGNOSTIC_IDS,
            (pilot.EXPECTED_PRESENT_FIT_D1, pilot.EXPECTED_PRESENT_FIT_D2), strict=True):
        state_path = RUN_DIR / "fits" / f"{fit_id}.json"
        input_path = RUN_DIR / "fit_inputs" / f"{fit_id}.npz"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state_sha = _sha(state_path)
        fit_arrays = _npz(input_path)
        fit_hash = _sha(input_path)
        _need(state.get("status") == "LP_COMPLETE" and state.get("fit_id") == fit_id,
              f"Saved fit state is not a complete matching fit: {fit_id}")
        fit_input = state["fit_input"]
        _need(fit_input.get("fit_id") == fit_id
              and fit_input.get("fit_rows") == expected_count
              and fit_input.get("prefit_freeze_sha256") == freeze_sha
              and fit_input.get("root_source_pass_sha256") == root_pass_sha
              and fit_input.get("fit_input_bundle_sha256") == fit_hash
              and state.get("root_source_pass_sha256") == root_pass_sha
              and state.get("root_final_approval_sha256") is None
              and state.get("diagnostics_sha256") is None,
              f"Fit state binding differs: {fit_id}")
        cycles = np.asarray(identity["cycles"], dtype=np.int64)
        missing = np.asarray(identity["missing"], dtype=bool)
        fit_cycle_mask = ((cycles >= 68) & (cycles <= 71) if fit_id == pilot.FIT_IDS[0]
                          else (cycles >= 68) & (cycles <= 75))
        fit_indices = np.flatnonzero(fit_cycle_mask & ~missing)
        _need(len(fit_indices) == expected_count, f"Fit population count differs: {fit_id}")
        expected_arrays = {
            "physical_rows": identity["physical_rows"][fit_indices].astype(np.int64),
            "record_ids": identity["ids"][fit_indices].astype(str),
            "prediction_origins": identity["origins"][fit_indices].astype(np.int8),
            "cycles": identity["cycles"][fit_indices].astype(np.int16),
            "r06_prediction_origins": identity["origins"][fit_indices].astype(np.int16),
            "r06_predictions": identity["raw"][fit_indices].astype("<f8"),
            "missing_mask": identity["missing"][fit_indices].astype(bool),
            "targets": identity["targets"][fit_indices].astype("<f8"),
        }
        _need(set(fit_arrays) == set(expected_arrays), f"Fit NPZ schema differs: {fit_id}")
        for name, expected in expected_arrays.items():
            _same_array(fit_arrays[name], expected, f"{fit_id}/{name}")
        _need(not np.any(fit_arrays["missing_mask"]),
              f"Missing rows entered present-only fit: {fit_id}")
        solver = state["solver_result"]
        coeff = _coefficients(state, fit_id)
        train_raw = fit_arrays["r06_predictions"].astype(np.float64)
        train_y = fit_arrays["targets"].astype(np.float64)
        train_pred = (coeff["intercept"] + coeff["slope_low"] * np.minimum(train_raw, 50.0)
                      + coeff["slope_high"] * np.maximum(train_raw - 50.0, 0.0))
        residual = np.abs(train_y - train_pred)
        slack_path = Path(state["solver_slacks_path"])
        _need(_sha(slack_path) == state["solver_slacks_sha256"],
              f"Solver slack artifact hash differs: {fit_id}")
        slacks_archive = _npz(slack_path, ("slacks",))
        slacks = slacks_archive["slacks"].astype(np.float64)
        max_violation = float(np.max(residual - slacks))
        loss = float(np.sum(residual))
        slack_sum = float(np.sum(slacks))
        _need(solver.get("success") is True and solver.get("status") == 0
              and solver.get("fit_rows") == expected_count
              and solver.get("solver", {}).get("threads") == 2
              and solver.get("solver", {}).get("method") == "highs"
              and slacks.shape == train_y.shape and np.all(np.isfinite(slacks))
              and np.all(slacks >= -1e-8) and max_violation <= 1e-7
              and math.isclose(loss, float(solver["objective_unweighted_sum_absolute"]),
                               rel_tol=1e-8, abs_tol=1e-7)
              and math.isclose(loss, float(solver["solver_objective"]),
                               rel_tol=1e-8, abs_tol=1e-7)
              and math.isclose(slack_sum, float(solver["slack_objective"]),
                               rel_tol=1e-8, abs_tol=1e-7),
              f"Independent LP/slack feasibility replay failed: {fit_id}")
        for field in ("RESERVE", "LP_STARTED", "LP_COMPLETE"):
            event = next(item for item in ledger_events
                         if item["fit_id"] == fit_id and item["event"] == field)
            _need(event.get("root_source_pass_sha256") == root_pass_sha
                  and event.get("root_final_approval_sha256") is None
                  and event.get("diagnostics_sha256") is None
                  and event.get("fit_input_sha256") == fit_hash
                  and event.get("state_sha256", state_sha) == state_sha,
                  f"Ledger does not bind the fit state/input: {fit_id}/{field}")
            resource = event.get("resources", event.get("resource", {}))
            if field in {"RESERVE", "LP_STARTED"}:
                _need(resource.get("status") == "PASS"
                      and resource.get("initial_floor_pass") is True
                      and resource.get("projected_free_pass") is True
                      and resource.get("threads") == 2,
                      f"Initial resource gate did not pass: {fit_id}/{field}")
            else:
                resource = state.get("resources_after_solver", {})
                _need(resource.get("status") == "PASS"
                      and resource.get("actual_boundary_pass") is True
                      and resource.get("threads") == 2,
                      f"Post-fit resource gate did not pass: {fit_id}")
        coefficients[fit_id] = coeff
        fit_audits[fit_id] = {
            "fit_rows": expected_count,
            "fit_input_sha256": fit_hash,
            "fit_state_sha256": state_sha,
            "solver_slacks_sha256": state["solver_slacks_sha256"],
            "solver_status": solver["status"],
            "solver_success": solver["success"],
            "solver_method": solver["solver"]["method"],
            "threads": solver["solver"]["threads"],
            "coefficients": coeff,
            "objective_replayed": loss,
            "slack_sum_replayed": slack_sum,
            "max_primal_violation_replayed": max_violation,
            "fit_record_id_sha256": source_verify._array_sha(fit_arrays["record_ids"]),
        }
    _need(not (RUN_DIR / "fits" / f"{pilot.FIT_IDS[2]}.json").exists(),
          "Conditional FINAL fit state exists")
    return fit_audits, coefficients


def _replay_metrics(identity: Mapping[str, Any], coefficients: Mapping[str, Mapping[str, float]],
                    freeze: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    ids = np.asarray(identity["ids"], dtype=str)
    origins = np.asarray(identity["origins"], dtype=np.int64)
    cycles = np.asarray(identity["cycles"], dtype=np.int64)
    missing = np.asarray(identity["missing"], dtype=bool)
    raw = np.asarray(identity["raw"], dtype="<f8")
    direct = np.asarray(identity["direct"], dtype="<f8")
    targets = np.asarray(identity["targets"], dtype="<f8")
    row_index = {fit_id: np.flatnonzero(
        ((cycles >= 72) & (cycles <= 75) & (origins == 67)) if fit_id == pilot.FIT_IDS[0]
        else ((cycles >= 76) & (cycles <= 79) & (origins == 75)))
        for fit_id in pilot.DIAGNOSTIC_IDS}
    _need(len(set(row_index[pilot.FIT_IDS[0]].tolist())
                & set(row_index[pilot.FIT_IDS[1]].tolist())) == 0,
          "D1/D2 score rows overlap")
    _need(len(row_index[pilot.FIT_IDS[0]]) == pilot.EXPECTED_SCORE_D1
          and len(row_index[pilot.FIT_IDS[1]]) == pilot.EXPECTED_SCORE_D2,
          "Score population sizes differ from the frozen windows")

    comparator_states: dict[str, dict[str, Any]] = {}
    for window, fit_id, key in (
            ("D1", "R06_HINGE_D1", "r2c_d1_state"),
            ("D2", "R06_HINGE_D2", "r2c_d2_state"),
            ("D1", "HINGE_D1", "r4_hinge_d1_state"),
            ("D2", "HINGE_D2", "r4_hinge_d2_state")):
        comparator_states[f"{window}:{fit_id}"] = _load_frozen_comparator_state(freeze, key)

    windows: dict[str, Any] = {}
    window_arrays: dict[str, dict[str, np.ndarray]] = {}
    fit_id_to_window = {pilot.FIT_IDS[0]: "D1", pilot.FIT_IDS[1]: "D2"}
    for fit_id in pilot.DIAGNOSTIC_IDS:
        window = fit_id_to_window[fit_id]
        idx = row_index[fit_id]
        p, m, y = raw[idx], missing[idx], targets[idx]
        all_state = comparator_states[f"{window}:R06_HINGE_{window}"]
        r4_state = comparator_states[f"{window}:HINGE_{window}"]
        predictions = {
            "present_native_hinge": _apply(p, m, coefficients[fit_id]),
            "all_row_native_hinge": _apply(p, m, _coefficients(all_state, "all-row-native-" + window)),
            "transported_r4_hinge_on_r06": _apply(p, m, _coefficients(r4_state, "transported-R4-" + window)),
            "matching_r4_hinge_on_direct": _apply(
                direct[idx], m, _coefficients(r4_state, "direct-R4-" + window)),
            "raw_r06": p.copy(),
        }
        for name in ("present_native_hinge", "all_row_native_hinge",
                     "transported_r4_hinge_on_r06", "matching_r4_hinge_on_direct"):
            _need(np.array_equal(predictions[name][m].view("<u8"), p[m].view("<u8")),
                  f"Missing-row raw fallback is not bit-exact: {window}/{name}")
        summary = _diagnostic_summary(y, predictions, m, cycles[idx])
        primary = summary["full_rows"]["metrics"]["transported_r4_hinge_on_r06"]["mae"]
        candidate = summary["full_rows"]["metrics"]["present_native_hinge"]["mae"]
        all_row = summary["full_rows"]["metrics"]["all_row_native_hinge"]["mae"]
        summary["gain_vs_transported_r4_primary"] = float(primary - candidate)
        summary["harm_vs_transported_r4_primary"] = float(candidate - primary)
        summary["gain_vs_all_row_native_mechanism"] = float(all_row - candidate)
        summary["raw_r06_gain_vs_candidate"] = float(
            summary["full_rows"]["metrics"]["raw_r06"]["mae"] - candidate)
        windows[window] = summary
        window_arrays[window] = {"indices": idx, **predictions}

    pool_indices = np.concatenate([row_index[pilot.FIT_IDS[0]], row_index[pilot.FIT_IDS[1]]])
    pooled_predictions = {
        name: np.concatenate([window_arrays["D1"][name], window_arrays["D2"][name]])
        for name in ("present_native_hinge", "all_row_native_hinge",
                     "transported_r4_hinge_on_r06", "matching_r4_hinge_on_direct", "raw_r06")
    }
    _need(len(pool_indices) == pilot.EXPECTED_POOLED
          and len(set(ids[pool_indices].tolist())) == pilot.EXPECTED_POOLED,
          "Pooled score population is not the exact unique 15,953 rows")
    pooled = _diagnostic_summary(targets[pool_indices], pooled_predictions,
                                 missing[pool_indices], cycles[pool_indices])
    primary = pooled["full_rows"]["metrics"]["transported_r4_hinge_on_r06"]["mae"]
    candidate = pooled["full_rows"]["metrics"]["present_native_hinge"]["mae"]
    all_row = pooled["full_rows"]["metrics"]["all_row_native_hinge"]["mae"]
    pooled_gain, pooled_harm = float(primary - candidate), float(candidate - primary)
    harms = [float(group["candidate_harm_vs_primary"])
             for summary in (pooled, *windows.values())
             for group in summary["availability_and_tail_groups"].values()
             if group.get("supported")]
    max_group_harm = max(harms) if harms else None
    pooled.update({
        "rows": len(pool_indices), "unique_rows": len(set(ids[pool_indices].tolist())),
        "gain_vs_transported_r4_primary": pooled_gain,
        "harm_vs_transported_r4_primary": pooled_harm,
        "gain_vs_all_row_native_mechanism": float(all_row - candidate),
        "max_supported_group_harm_vs_primary": max_group_harm,
    })
    gains_all_row = {name: float(windows[name]["gain_vs_all_row_native_mechanism"])
                     for name in ("D1", "D2")}
    mechanism_pass = gains_all_row["D1"] > 0.0 and gains_all_row["D2"] > 0.0
    support_pass = all(summary["all_required_groups_supported"]
                       and summary["all_scored_cycles_supported"]
                       for summary in (pooled, *windows.values()))
    qualification = bool(
        support_pass and pooled_gain >= 0.005
        and windows["D1"]["gain_vs_transported_r4_primary"] >= 0.0
        and windows["D2"]["gain_vs_transported_r4_primary"] >= 0.0
        and max_group_harm is not None and max_group_harm <= 0.25)
    exploratory = bool(
        support_pass and pooled_gain >= 0.0
        and windows["D1"]["harm_vs_transported_r4_primary"] <= 0.02
        and windows["D2"]["harm_vs_transported_r4_primary"] <= 0.02
        and max_group_harm is not None and max_group_harm <= 0.25)
    eligible = bool(mechanism_pass and (qualification or exploratory))
    replay = {
        "windows": windows,
        "pooled_unique": pooled,
        "mechanism_gate": {
            "comparator": "saved all-row native R2C hinge on the same score rows",
            "each_window_gain_strictly_greater_than": 0.0,
            "window_gains": gains_all_row,
            "pass": mechanism_pass,
        },
        "qualification_gate": {
            "pooled_gain_vs_primary_min": 0.005,
            "each_window_gain_vs_transported_primary_min": 0.0,
            "max_supported_availability_or_tail_harm_vs_primary_le": 0.25,
            "pass": qualification,
        },
        "exploratory_gate": {
            "pooled_gain_vs_primary_min": 0.0,
            "each_window_harm_vs_transported_primary_le": 0.02,
            "supported_group_harm_vs_primary_le": 0.25,
            "pass": exploratory,
        },
        "root_review_eligible": eligible,
    }
    return replay, {
        "score_record_id_sha256": {
            window: source_verify._array_sha(ids[row_index[fit_id]])
            for fit_id, window in fit_id_to_window.items()},
        "pooled_score_record_id_sha256": source_verify._array_sha(ids[pool_indices]),
        "candidate_prediction_sha256": {
            window: source_verify._array_sha(window_arrays[window]["present_native_hinge"])
            for window in ("D1", "D2")},
    }


def verify_and_write() -> dict[str, Any]:
    freeze, freeze_sha = _signed_json(pilot.FREEZE)
    preflight, preflight_sha = _signed_json(pilot.PREFLIGHT)
    fit_population, fit_population_sha = _signed_json(pilot.FIT_POPULATION_IDS)
    root_pass, root_pass_sha = _signed_json(pilot.ROOT_SOURCE_PASS)
    diagnostics, diagnostics_sha = _signed_json(REPORT)
    source_verification, source_verification_sha = _signed_json(pilot.VERIFICATION)

    _need(root_pass.get("experiment_id") == pilot.EXPERIMENT_ID
          and root_pass.get("verdict") == "PASS_SOURCE_AND_INPUTS"
          and root_pass.get("authorized_fit_ids") == list(pilot.DIAGNOSTIC_IDS)
          and root_pass.get("final_fit_authorized") is False
          and root_pass.get("prefit_freeze_sha256") == freeze_sha
          and root_pass.get("identity_preflight_sha256") == preflight_sha
          and root_pass.get("source_commit") == freeze.get("source_commit")
          and root_pass.get("source_hashes") == freeze.get("source_hashes")
          and root_pass.get("input_hashes") == freeze.get("input_hashes"),
          "Root source pass does not bind this exact R2 source/input package")
    _need(preflight.get("fit_population_ids_sha256") == fit_population_sha
          and source_verification.get("status") == "PASS_SOURCE_READY_PENDING_ROOT_RELEASE_R2"
          and source_verification.get("official_lp_starts") == 0,
          "Source-ready artifacts do not bind the committed fit populations")
    _need(freeze.get("source_hashes") == root_pass.get("source_hashes"),
          "Freeze and root source pass source hashes differ")
    for name, path in pilot.SOURCE_FILES.items():
        _need(path.is_file() and _sha(path) == freeze["source_hashes"][name],
              f"Frozen source changed after root review: {name}")
    for name, item in freeze["input_hashes"].items():
        if name == "official_train":
            continue  # Preserve the reviewed rule: do not reopen the CSV.
        path = Path(item["path"])
        _need(path.is_file() and _sha(path) == item["sha256"],
              f"Frozen input changed after root review: {name}")
    _need(diagnostics.get("status") == "TWO_DIAGNOSTICS_COMPLETE_PENDING_ROOT_REVIEW"
          and diagnostics.get("experiment_id") == pilot.EXPERIMENT_ID
          and diagnostics.get("prefit_freeze_sha256") == freeze_sha
          and diagnostics.get("identity_preflight_sha256") == preflight_sha
          and diagnostics.get("root_source_pass_sha256") == root_pass_sha
          and diagnostics.get("final_fit_started") is False
          and diagnostics.get("test_labels_read") is False
          and diagnostics.get("test_or_sample_accessed") is False
          and diagnostics.get("csv_created") is False
          and diagnostics.get("kaggle_accessed") is False,
          "Diagnostic report identity or boundary flags differ")

    # Independent control lineage checks bind the transported R4 and the prior
    # all-row-native R2C states to their already reviewed immutable artifacts.
    r4_lineage = source_verify._check_r4_lineage(dict(freeze))
    all_row_lineage = source_verify._check_all_row_native_lineage(dict(freeze))
    identity = _load_identity_and_targets(freeze, preflight, fit_population, root_pass_sha)
    fit_audits, coefficients = _verify_fits(
        identity, freeze, freeze_sha, root_pass_sha, preflight_sha)
    replay, score_hashes = _replay_metrics(identity, coefficients, freeze)
    for key in ("windows", "pooled_unique", "mechanism_gate", "qualification_gate",
                "exploratory_gate", "root_review_eligible"):
        _close_tree(replay[key], diagnostics[key], key)

    ledger_sha = _sha(LEDGER)
    body: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": pilot.EXPERIMENT_ID,
        "status": "PASS_INDEPENDENT_POSTFIT_REPLAY",
        "diagnostics_sha256": diagnostics_sha,
        "prefit_freeze_sha256": freeze_sha,
        "identity_preflight_sha256": preflight_sha,
        "fit_population_ids_sha256": fit_population_sha,
        "root_source_pass_sha256": root_pass_sha,
        "source_verification_sha256": source_verification_sha,
        "ledger_sha256": ledger_sha,
        "ledger_sequence": [
            {"fit_id": event["fit_id"], "event": event["event"]}
            for event in [json.loads(line) for line in LEDGER.read_text(
                encoding="utf-8").splitlines()]
        ],
        "fit_replays": fit_audits,
        "score_identity_hashes": score_hashes,
        "row_cache_audit": identity["cache_audit"],
        "r4_comparator_lineage": r4_lineage,
        "all_row_native_control_lineage": all_row_lineage,
        "independent_metrics_sha256": _canonical_sha({
            "windows": replay["windows"], "pooled_unique": replay["pooled_unique"]}),
        "mechanism_gate": replay["mechanism_gate"],
        "qualification_gate": replay["qualification_gate"],
        "exploratory_gate": replay["exploratory_gate"],
        "root_review_eligible": replay["root_review_eligible"],
        "official_lp_starts": 2,
        "final_fit_started": False,
        "test_labels_read": False,
        "test_or_sample_accessed": False,
        "csv_created": False,
        "kaggle_accessed": False,
        "train_csv_reopened": False,
    }
    body["signature"] = _canonical_sha(body)
    payload = json.dumps(body, sort_keys=True, ensure_ascii=False,
                         allow_nan=False, indent=2).encode("utf-8") + b"\n"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        _need(OUTPUT.read_bytes() == payload,
              f"Independent post-fit receipt already exists with different bytes: {OUTPUT}")
    else:
        with OUTPUT.open("xb") as stream:
            stream.write(payload)
            stream.flush()
    return {"status": body["status"], "path": str(OUTPUT.resolve()),
            "sha256": _sha(OUTPUT), "official_lp_starts": body["official_lp_starts"],
            "root_review_eligible": body["root_review_eligible"]}


def main() -> int:
    try:
        result = verify_and_write()
        print(json.dumps(result, sort_keys=True, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "FAIL_INDEPENDENT_POSTFIT_REPLAY",
                          "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
