"""Independent read-only replay for the authorized R2C D1/D2 diagnostics.

This post-fit auditor is outside the frozen executor source set. It never calls
the LP solver, reads Test data, writes model state, or changes any frozen input.
It reads the signed root pass before opening the cached target member.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/reg_r06_native_hinge_v1"
REPORTS = ROOT / "reports/optimization_v3/reg_r06_native_hinge_v1"
RUN = ROOT / "experiments/runs/reg_r06_native_hinge_v1/R2C"
FREEZE = EXP / "PREFIT_FREEZE_R2C.json"
PREFLIGHT = REPORTS / "IDENTITY_PREFLIGHT_R2C.json"
ROOT_PASS = EXP / "ROOT_SOURCE_PASS_R2C.json"
DIAGNOSTICS = REPORTS / "DIAGNOSTICS_R2C.json"
LEDGER = RUN / "lp_ledger.jsonl"
VERIFICATION = REPORTS / "DIAGNOSTICS_VERIFICATION_R2C.json"
FIT_IDS = ("R06_HINGE_D1", "R06_HINGE_D2")
EXPECTED_ROWS = {FIT_IDS[0]: 4_950, FIT_IDS[1]: 12_412}
EXPECTED_SCORE_ROWS = {"D1": 7_462, "D2": 8_491}
PREDICTION_NAMES = ("new_r06_native_hinge", "same_cutoff_r06_old_hinge",
                    "same_cutoff_direct_old_hinge", "raw_r06")


class AuditError(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8", errors="surrogatepass")


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _array_sha(value: Any) -> str:
    arr = np.asarray(value)
    prefix = _canonical({"dtype": arr.dtype.str, "shape": list(arr.shape)}) + b"\0"
    if arr.dtype.kind in "USO":
        payload = b"\0".join(str(item).encode("utf-8", errors="surrogatepass")
                              for item in arr.reshape(-1).tolist())
    else:
        payload = np.ascontiguousarray(arr).tobytes(order="C")
    return hashlib.sha256(prefix + payload).hexdigest()


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def _signed(path: Path) -> tuple[dict[str, Any], str]:
    _need(path.is_file(), f"Missing signed artifact: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    signature = data.pop("signature", None)
    _need(signature == _canonical_sha(data), f"Invalid canonical signature: {path.name}")
    data["signature"] = signature
    return data, _sha(path)


def _close(expected: Any, actual: Any, path: str = "report") -> None:
    if isinstance(expected, Mapping):
        _need(isinstance(actual, Mapping) and set(expected) == set(actual),
              f"Metric replay structure differs at {path}")
        for key in expected:
            _close(expected[key], actual[key], f"{path}.{key}")
    elif isinstance(expected, list):
        _need(isinstance(actual, list) and len(expected) == len(actual),
              f"Metric replay list differs at {path}")
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            _close(left, right, f"{path}[{index}]")
    elif isinstance(expected, (bool, int, str)) or expected is None:
        _need(type(expected) is type(actual) and expected == actual,
              f"Metric replay value differs at {path}: {expected!r} != {actual!r}")
    elif isinstance(expected, float):
        _need(math.isclose(expected, float(actual), rel_tol=1e-10, abs_tol=1e-12),
              f"Metric replay number differs at {path}: {expected} != {actual}")
    else:
        _need(expected == actual, f"Metric replay value differs at {path}")


def _metric(y: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    absolute = np.abs(error)
    return {"n": int(error.size), "status": "MEASURED",
            "mae": float(np.mean(absolute)),
            "p90_absolute_error": float(np.quantile(absolute, 0.90, method="linear")),
            "mean_error_prediction_minus_target": float(np.mean(error))}


def _summary_block(y: np.ndarray, predictions: Mapping[str, np.ndarray],
                   mask: np.ndarray, minimum: int, support_name: str) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    n = int(mask.sum())
    support_rule = f"n >= {minimum} for {support_name}"
    if n < minimum:
        return {"n": n, "supported": False, "status": "UNKNOWN",
                "support_rule": support_rule}
    return {"n": n, "supported": True, "status": "MEASURED",
            "support_rule": support_rule,
            "metrics": {name: _metric(y[mask], values[mask])
                        for name, values in predictions.items()}}


def _summary(y: np.ndarray, predictions: Mapping[str, np.ndarray],
             missing: np.ndarray, cycles: np.ndarray) -> dict[str, Any]:
    full = _summary_block(y, predictions, np.ones(len(y), dtype=bool), 1, "full rows")
    present = _summary_block(y, predictions, ~missing, 1, "present rows")
    group_masks = {
        "availability_present": ~missing,
        "availability_missing": missing,
        "target_le_5": y <= 5.0,
        "target_ge_95": y >= 95.0,
    }
    groups = {name: _summary_block(y, predictions, mask, 200, name)
              for name, mask in group_masks.items()}
    for group in groups.values():
        if group["supported"]:
            metrics = group["metrics"]
            harm = (metrics["new_r06_native_hinge"]["mae"]
                    - metrics["same_cutoff_r06_old_hinge"]["mae"])
            group["candidate_harm_vs_primary"] = harm
            group["candidate_gain_vs_primary"] = -harm
    per_cycle = {str(int(cycle)): _summary_block(
        y, predictions, cycles == cycle, 50, f"cycle_{int(cycle)}")
        for cycle in sorted(np.unique(cycles))}
    return {
        "n": int(len(y)), "present_rows": int((~missing).sum()),
        "missing_rows": int(missing.sum()), "full_rows": full,
        "present_only": present, "availability_and_tail_groups": groups,
        "per_cycle": per_cycle,
        "all_required_groups_supported": all(g["supported"] for g in groups.values()),
        "all_scored_cycles_supported": all(g["supported"] for g in per_cycle.values()),
    }


def _resource_snapshot(snapshot: Mapping[str, Any], boundary: str, *, initial: bool) -> None:
    available = snapshot.get("available_bytes_measured")
    estimate = 512 * 1024**2
    _need(snapshot.get("status") == "PASS"
          and snapshot.get("boundary") == boundary
          and isinstance(available, int)
          and snapshot.get("incremental_estimate_bytes") == estimate
          and snapshot.get("minimum_initial_available_bytes") == 2 * 1024**3
          and snapshot.get("minimum_projected_free_bytes") == 1024**3
          and snapshot.get("actual_available_boundary_minimum_bytes") == 1024**3
          and snapshot.get("projected_free_bytes") == available - estimate
          and snapshot.get("actual_boundary_pass") is True
          and snapshot.get("estimated_peak_is_measured") is False
          and snapshot.get("process_rss_measured") is True
          and isinstance(snapshot.get("process_rss_bytes"), int)
          and snapshot.get("process_rss_bytes") > 0
          and snapshot.get("threads") == 2
          and snapshot.get("concurrent_heavy_jobs") == 1,
          f"Resource snapshot contract failed at {boundary}")
    if initial:
        _need(available >= 2 * 1024**3
              and snapshot.get("projected_free_bytes") >= 1024**3
              and snapshot.get("initial_floor_pass") is True
              and snapshot.get("projected_free_pass") is True,
              f"Initial small-LP RAM floor failed at {boundary}")
    else:
        _need(available >= 1024**3, f"Boundary RAM reserve failed at {boundary}")


def _hinge(raw: np.ndarray, missing: np.ndarray, fallback: np.ndarray,
           coefficients: Mapping[str, Any]) -> np.ndarray:
    x = np.asarray(raw, dtype=np.float64)
    mask = np.asarray(missing, dtype=bool)
    out = np.clip(float(coefficients["intercept"])
                  + float(coefficients["slope_low"]) * np.minimum(x, 50.0)
                  + float(coefficients["slope_high"]) * np.maximum(x - 50.0, 0.0),
                  0.0, 100.0)
    out[mask] = np.asarray(fallback, dtype=np.float64)[mask]
    _need(np.array_equal(out[mask].view("<u8"),
                         np.asarray(fallback, dtype="<f8")[mask].view("<u8")),
          "Missing fallback is not bit-exact")
    return out


def _verify_authorization_and_freeze() -> tuple[dict[str, Any], str, str]:
    freeze, freeze_sha = _signed(FREEZE)
    preflight, preflight_sha = _signed(PREFLIGHT)
    pass_data, pass_sha = _signed(ROOT_PASS)
    _need(freeze["experiment_id"] == "REG_R06_NATIVE_HINGE_V1_R2C"
          and freeze["status"] ==
          "IDENTITY_FROZEN_PENDING_NON_TARGET_PREFLIGHT_AND_ROOT_REVIEW_R2C",
          "R2C freeze identity/status mismatch")
    _need(preflight["status"] ==
          "PASS_NON_TARGET_IDENTITY_PREFLIGHT_PENDING_ROOT_SOURCE_PASS_R2C"
          and preflight["prefit_freeze_sha256"] == freeze_sha
          and preflight["official_lp_starts"] == 0
          and preflight["train_target_values_loaded"] is False
          and preflight["score_target_values_loaded"] is False,
          "R2C non-target preflight boundary mismatch")
    _need(pass_data.get("experiment_id") == freeze["experiment_id"]
          and pass_data.get("verdict") == "PASS_SOURCE_AND_INPUTS"
          and pass_data.get("prefit_freeze_sha256") == freeze_sha
          and pass_data.get("identity_preflight_sha256") == preflight_sha
          and pass_data.get("source_commit") == freeze["source_commit"]
          and pass_data.get("source_hashes") == freeze["source_hashes"]
          and pass_data.get("input_hashes") == freeze["input_hashes"]
          and pass_data.get("authorized_fit_ids") == list(FIT_IDS)
          and pass_data.get("final_fit_authorized") is False,
          "Root source pass does not authorize exactly D1/D2")
    _need(freeze.get("source_commit_must_be_ancestor") is True,
          "Freeze does not declare the source ancestry rule")
    ancestor = subprocess.run(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor",
         str(freeze["source_commit"]), "HEAD"], capture_output=True, text=True, check=False)
    _need(ancestor.returncode == 0, "Frozen source commit is not an ancestor of HEAD")
    source_paths = {
        "runner": ROOT / "scripts/reg_r06_native_hinge_v2.py",
        "verifier": ROOT / "scripts/verify_reg_r06_native_hinge_v2.py",
        "tests": ROOT / "tests/test_reg_r06_native_hinge_v2.py",
        "protocol": REPORTS / "PROTOCOL_R2C.md",
        "hinge_solver_r4": ROOT / "scripts/reg_monotone_hinge_p50_v4.py",
        "hinge_core_r1": ROOT / "scripts/reg_monotone_hinge_p50_v1.py",
        "direct_identity_helpers_v3": ROOT / "scripts/reg_monotone_hinge_p50_v3.py",
        "final_wave_helpers_v2": ROOT / "scripts/reg_final_wave_v2.py",
    }
    _need(set(freeze["source_hashes"]) == set(source_paths),
          "Frozen source hash inventory differs from the declared R2C inventory")
    for name, path in source_paths.items():
        _need(path.is_file() and _sha(path) == freeze["source_hashes"][name],
              f"Frozen source bytes changed: {name}")
    for name, item in freeze["input_hashes"].items():
        path = Path(item["path"])
        _need(path.is_file() and _sha(path) == item["sha256"],
              f"Frozen input bytes changed: {name}")
    return freeze, freeze_sha, pass_sha


def _load_identity_and_targets(freeze: Mapping[str, Any], preflight: Mapping[str, Any]
                               ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    label_path = Path(freeze["input_hashes"]["oof_label_and_identity_bundle"]["path"])
    names = ("physical_rows", "record_ids", "prediction_origins", "original_bundle_indices",
             "cycles", "missing_mask", "direct_predictions", "targets")
    with np.load(label_path, allow_pickle=False) as archive:
        _need(set(names).issubset(archive.files), "OOF label/identity NPZ is missing members")
        labels = {name: np.array(archive[name], copy=True) for name in names}
    n = len(labels["record_ids"])
    _need(n == 20_903 and all(value.ndim == 1 and len(value) == n
                               for value in labels.values()),
          "Cached identity/target vectors have invalid dimensions")
    _need(len(np.unique(labels["record_ids"])) == n
          and len(np.unique(labels["physical_rows"])) == n
          and np.array_equal(np.unique(labels["cycles"]), np.arange(68, 80)),
          "Cached record/physical-row/cycle identity is not unique and complete")
    targets = np.asarray(labels["targets"], dtype=np.float64)
    _need(np.all(np.isfinite(targets)) and np.all((targets >= 0.0) & (targets <= 100.0)),
          "Cached Train target vector is invalid")
    by_id = {str(value): index for index, value in enumerate(labels["record_ids"])}
    raw = np.full(n, np.nan, dtype=np.float64)
    origins = np.full(n, -1, dtype=np.int16)
    rows = labels["physical_rows"].astype(np.int64)
    cycles = labels["cycles"].astype(np.int64)
    for origin, cycle_member, allowed in (
        (67, "audit__cycle_code", set(range(68, 76))),
        (75, "score_cycles", set(range(76, 80))),
    ):
        record = freeze["input_hashes"][f"r06_fit{origin}_npz"]
        with np.load(record["path"], allow_pickle=False) as archive:
            required = {"score_rows", "score_record_ids", "prediction", cycle_member}
            _need(required.issubset(archive.files), f"R06 fit{origin} cache lacks identity fields")
            cache_rows = np.asarray(archive["score_rows"], dtype=np.int64)
            cache_ids = np.asarray(archive["score_record_ids"]).astype(str)
            cache_cycles = np.asarray(archive[cycle_member], dtype=np.int64)
            cache_prediction = np.asarray(archive["prediction"], dtype=np.float64)
        selected = np.isin(cache_cycles, list(allowed))
        actual = {(str(cache_ids[i]), int(cache_cycles[i]), int(cache_rows[i]))
                  for i in np.flatnonzero(selected)}
        expected = {(str(labels["record_ids"][i]), int(cycles[i]), int(rows[i]))
                    for i in range(n) if int(cycles[i]) in allowed}
        _need(actual == expected and len(actual) == int(selected.sum()),
              f"R06 fit{origin} IDs/cycles/physical positions differ from OOF identity")
        for i in np.flatnonzero(selected):
            target_index = by_id[str(cache_ids[i])]
            _need(origins[target_index] == -1, "R06 cache assigns an OOF row more than once")
            raw[target_index] = cache_prediction[i]
            origins[target_index] = origin
    _need(np.all(origins >= 0)
          and np.array_equal(origins, labels["prediction_origins"]),
          "R06 cache origins differ from the frozen OOF identity")
    missing = labels["missing_mask"].astype(bool)
    _need(np.array_equal(labels["direct_predictions"][missing].astype("<f8").view("<u8"),
                         raw[missing].astype("<f8").view("<u8")),
          "DIRECT missing fallback does not match raw R06 bit-for-bit")
    _need(_array_sha(raw) == preflight["array_hashes"]["r06_oof_prediction"],
          "Independently stitched R06 prediction hash differs from preflight")
    for name, values in {
        "record_ids": labels["record_ids"],
        "physical_rows": labels["physical_rows"],
        "prediction_origins": labels["prediction_origins"],
        "cycles": labels["cycles"],
        "missing_mask": missing,
        "direct_predictions": labels["direct_predictions"],
        "r06_prediction_origin": origins,
    }.items():
        _need(_array_sha(values) == preflight["array_hashes"][name],
              f"Independent identity hash differs from preflight: {name}")
    return labels, targets, raw


def _verify_fit(fit_id: str, fit_index: np.ndarray, labels: Mapping[str, np.ndarray],
                targets: np.ndarray, raw: np.ndarray, freeze_sha: str,
                pass_sha: str, ledger_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    fit_path = RUN / "fit_inputs" / f"{fit_id}.npz"
    state_path = RUN / "fits" / f"{fit_id}.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    _need(state.get("fit_id") == fit_id and state.get("status") == "LP_COMPLETE"
          and state.get("experiment_id") == "REG_R06_NATIVE_HINGE_V1_R2C",
          f"Fit state is not a completed R2C {fit_id}")
    fit_input = state["fit_input"]
    _need(_sha(fit_path) == fit_input["fit_input_bundle_sha256"]
          and fit_input.get("fit_rows") == EXPECTED_ROWS[fit_id]
          and fit_input.get("fit_id") == fit_id
          and fit_input.get("prefit_freeze_sha256") == freeze_sha
          and fit_input.get("root_source_pass_sha256") == pass_sha
          and state.get("root_source_pass_sha256") == pass_sha
          and state.get("root_final_approval_sha256") is None
          and state.get("diagnostics_sha256") is None,
          f"Fit state provenance mismatch: {fit_id}")
    with np.load(fit_path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    _need(set(arrays) == set(fit_input["fit_input_hashes"]),
          f"Fit input member inventory differs: {fit_id}")
    for name, values in arrays.items():
        _need(_array_sha(values) == fit_input["fit_input_hashes"][name],
              f"Fit input member hash differs: {fit_id}/{name}")
    expected = {
        "physical_rows": labels["physical_rows"][fit_index].astype(np.int64),
        "record_ids": labels["record_ids"][fit_index].astype(str),
        "prediction_origins": labels["prediction_origins"][fit_index].astype(np.int8),
        "cycles": labels["cycles"][fit_index].astype(np.int16),
        "r06_prediction_origins": np.asarray(
            np.where(labels["prediction_origins"][fit_index] == 67, 67, 75), dtype=np.int16),
        "r06_predictions": raw[fit_index].astype("<f8"),
        "missing_mask": labels["missing_mask"][fit_index].astype(bool),
        "targets": targets[fit_index].astype("<f8"),
    }
    for name, expected_values in expected.items():
        actual = arrays[name]
        if actual.dtype.kind == "f":
            same = np.array_equal(actual.astype("<f8").view("<u8"),
                                  np.asarray(expected_values, dtype="<f8").view("<u8"))
        else:
            same = np.array_equal(actual, expected_values)
        _need(same, f"Fit rows differ from independently reconstructed OOF rows: {fit_id}/{name}")
    solver = state["solver_result"]
    _need(solver.get("success") is True and solver.get("status") == 0
          and solver.get("solver", {}).get("method") == "highs"
          and solver.get("solver", {}).get("threads") == 2
          and solver.get("solver", {}).get("time_limit_seconds") == 180
          and solver.get("fit_rows") == EXPECTED_ROWS[fit_id]
          and solver.get("knot") == 50.0 and solver.get("slope_bounds") == [0.0, 2.0],
          f"Solver metadata or p50 bounds differ: {fit_id}")
    a, sl, sh = (float(solver[k]) for k in ("intercept", "slope_low", "slope_high"))
    _need(0.0 <= sl <= 2.0 and 0.0 <= sh <= 2.0,
          f"Slope bounds failed: {fit_id}")
    _resource_snapshot(state["resources_before_start"],
                       f"{fit_id}_initial_locked_before_start", initial=True)
    _resource_snapshot(state["resources_after_solver"], f"{fit_id}_after_solver",
                       initial=False)
    x, y = arrays["r06_predictions"].astype(np.float64), arrays["targets"].astype(np.float64)
    fitted = a + sl * np.minimum(x, 50.0) + sh * np.maximum(x - 50.0, 0.0)
    absolute = np.abs(y - fitted)
    slack_path = Path(state["solver_slacks_path"])
    _need(_sha(slack_path) == state["solver_slacks_sha256"],
          f"Solver slack archive hash differs: {fit_id}")
    with np.load(slack_path, allow_pickle=False) as archive:
        slacks = np.asarray(archive["slacks"], dtype=np.float64)
    loss = float(absolute.sum())
    _need(slacks.shape == y.shape and np.all(np.isfinite(slacks))
          and np.all(slacks >= -1e-8)
          and float(np.max(absolute - slacks)) <= 1e-7
          and math.isclose(loss, float(solver["objective_unweighted_sum_absolute"]),
                           rel_tol=1e-8, abs_tol=1e-7)
          and math.isclose(loss, float(solver["solver_objective"]),
                           rel_tol=1e-8, abs_tol=1e-7)
          and math.isclose(float(slacks.sum()), float(solver["solver_objective"]),
                           rel_tol=1e-8, abs_tol=1e-7),
          f"Saved objective/slacks do not replay: {fit_id}")
    matching = [event for event in ledger_rows if event.get("fit_id") == fit_id]
    sequence = [event.get("event") for event in matching]
    _need(sequence == ["RESERVE", "LP_STARTED", "LP_COMPLETE"],
          f"Ledger sequence is not exactly one terminal fit: {fit_id}")
    for event in matching:
        _need(event.get("fit_input_sha256") == _sha(fit_path)
              and event.get("root_source_pass_sha256") == pass_sha
              and event.get("root_final_approval_sha256") is None
              and event.get("diagnostics_sha256") is None,
              f"Ledger provenance differs: {fit_id}/{event.get('event')}")
    _resource_snapshot(matching[0]["resources"],
                       f"{fit_id}_initial_locked_before_start", initial=True)
    _resource_snapshot(matching[1]["resources"], f"{fit_id}_fit_start", initial=False)
    _need(matching[0]["resources"] == state["resources_before_start"],
          f"Reservation resources differ from fit state: {fit_id}")
    _need(matching[-1].get("state_sha256") == _sha(state_path)
          and matching[-1].get("solver_slacks_sha256") == state["solver_slacks_sha256"],
          f"Ledger completion does not bind fit state/slacks: {fit_id}")
    return state, {"fit_id": fit_id, "fit_rows": len(y),
                   "fit_input_sha256": _sha(fit_path),
                   "state_sha256": _sha(state_path),
                   "slacks_sha256": state["solver_slacks_sha256"],
                   "coefficients": {"intercept": a, "slope_low": sl,
                                    "slope_high": sh, "knot": 50.0},
                   "solver_objective": float(solver["solver_objective"]),
                   "objective_replayed": loss,
                   "maximum_abs_residual_minus_slack": float(np.max(absolute - slacks)),
                   "resource_before": state["resources_before_start"],
                   "resource_after": state["resources_after_solver"]}


def _replay_report(labels: Mapping[str, np.ndarray], targets: np.ndarray, raw: np.ndarray,
                   states: Mapping[str, Mapping[str, Any]], freeze: Mapping[str, Any],
                   preflight: Mapping[str, Any], report: Mapping[str, Any]) -> dict[str, Any]:
    cycles = labels["cycles"].astype(np.int64)
    origins = labels["prediction_origins"].astype(np.int64)
    missing_all = labels["missing_mask"].astype(bool)
    direct = labels["direct_predictions"].astype(np.float64)
    historical = freeze["comparators"]["historical_cutoff_states"]
    slices: dict[str, np.ndarray] = {}
    expected_specs = {
        "D1": (FIT_IDS[0], 72, 75, 67),
        "D2": (FIT_IDS[1], 76, 79, 75),
    }
    window_rows: dict[str, dict[str, Any]] = {}
    score_indices: dict[str, np.ndarray] = {}
    candidate_rows: dict[str, np.ndarray] = {}
    for window, (fit_id, low_cycle, high_cycle, origin) in expected_specs.items():
        idx = np.flatnonzero((cycles >= low_cycle) & (cycles <= high_cycle))
        _need(len(idx) == EXPECTED_SCORE_ROWS[window]
              and np.all(origins[idx] == origin)
              and len(np.unique(labels["record_ids"][idx])) == len(idx),
              f"Scored rows/cycle/origin mismatch: {window}")
        miss = missing_all[idx]
        candidate = _hinge(raw[idx], miss, raw[idx], states[fit_id]["solver_result"])
        old = _hinge(raw[idx], miss, raw[idx], historical[
            "HINGE_D1" if window == "D1" else "HINGE_D2"]["coefficients"])
        old_direct = _hinge(direct[idx], miss, raw[idx], historical[
            "HINGE_D1" if window == "D1" else "HINGE_D2"]["coefficients"])
        predictions = {"new_r06_native_hinge": candidate,
                       "same_cutoff_r06_old_hinge": old,
                       "same_cutoff_direct_old_hinge": old_direct,
                       "raw_r06": raw[idx]}
        summary = _summary(targets[idx], predictions, miss, cycles[idx])
        primary_mae = summary["full_rows"]["metrics"]["same_cutoff_r06_old_hinge"]["mae"]
        candidate_mae = summary["full_rows"]["metrics"]["new_r06_native_hinge"]["mae"]
        summary["gain_vs_same_cutoff_primary"] = primary_mae - candidate_mae
        summary["harm_vs_same_cutoff_primary"] = candidate_mae - primary_mae
        summary["raw_r06_gain_vs_new"] = (
            summary["full_rows"]["metrics"]["raw_r06"]["mae"] - candidate_mae)
        _close(summary, report["windows"][window], f"windows.{window}")
        window_rows[window] = summary
        score_indices[window] = idx
        candidate_rows[window] = candidate
    comparator_arrays = {}
    for window in ("D1", "D2"):
        fit_id = FIT_IDS[0] if window == "D1" else FIT_IDS[1]
        coefficients = historical["HINGE_D1" if window == "D1" else "HINGE_D2"]["coefficients"]
        idx = score_indices[window]
        primary_full = np.full(len(cycles), np.nan, dtype=np.float64)
        direct_full = np.full(len(cycles), np.nan, dtype=np.float64)
        primary_full[idx] = _hinge(raw[idx], missing_all[idx], raw[idx], coefficients)
        direct_full[idx] = _hinge(direct[idx], missing_all[idx], raw[idx], coefficients)
        comparator_arrays[f"same_cutoff_hinge_on_r06_{window}"] = primary_full
        comparator_arrays[f"same_cutoff_hinge_on_direct_{window}"] = direct_full
    for name, values in comparator_arrays.items():
        _need(_array_sha(values) == preflight["same_cutoff_comparator_hashes"][name],
              f"Independently replayed past-only comparator hash differs: {name}")
    pooled_idx = np.concatenate([score_indices["D1"], score_indices["D2"]])
    _need(len(pooled_idx) == 15_953 and len(np.unique(pooled_idx)) == 15_953,
          "Pooled score windows are not 15,953 unique OOF rows")
    pooled_predictions = {
        "new_r06_native_hinge": np.concatenate(
            [candidate_rows["D1"], candidate_rows["D2"]]),
        "same_cutoff_r06_old_hinge": np.concatenate([
            _hinge(raw[score_indices["D1"]], missing_all[score_indices["D1"]],
                   raw[score_indices["D1"]], historical["HINGE_D1"]["coefficients"]),
            _hinge(raw[score_indices["D2"]], missing_all[score_indices["D2"]],
                   raw[score_indices["D2"]], historical["HINGE_D2"]["coefficients"])]),
        "same_cutoff_direct_old_hinge": np.concatenate([
            _hinge(direct[score_indices["D1"]], missing_all[score_indices["D1"]],
                   raw[score_indices["D1"]], historical["HINGE_D1"]["coefficients"]),
            _hinge(direct[score_indices["D2"]], missing_all[score_indices["D2"]],
                   raw[score_indices["D2"]], historical["HINGE_D2"]["coefficients"])]),
        "raw_r06": raw[pooled_idx],
    }
    pooled = _summary(targets[pooled_idx], pooled_predictions,
                      missing_all[pooled_idx], cycles[pooled_idx])
    primary_mae = pooled["full_rows"]["metrics"]["same_cutoff_r06_old_hinge"]["mae"]
    candidate_mae = pooled["full_rows"]["metrics"]["new_r06_native_hinge"]["mae"]
    gain, harm = primary_mae - candidate_mae, candidate_mae - primary_mae
    group_harms = [g["candidate_harm_vs_primary"]
                   for summary in [pooled, *window_rows.values()]
                   for g in summary["availability_and_tail_groups"].values()
                   if g.get("supported")]
    all_supported = all(summary["all_required_groups_supported"]
                        and summary["all_scored_cycles_supported"]
                        for summary in [pooled, *window_rows.values()])
    max_group_harm = max(group_harms) if group_harms else None
    pooled["rows"] = len(pooled_idx)
    pooled["unique_rows"] = len(np.unique(pooled_idx))
    pooled["gain_vs_same_cutoff_primary"] = gain
    pooled["harm_vs_same_cutoff_primary"] = harm
    pooled["max_supported_group_harm_vs_primary"] = max_group_harm
    _close(pooled, report["pooled_unique"], "pooled_unique")
    qualification = bool(all_supported and gain >= 0.005
        and window_rows["D1"]["gain_vs_same_cutoff_primary"] >= 0.0
        and window_rows["D2"]["gain_vs_same_cutoff_primary"] >= 0.0
        and max_group_harm is not None and max_group_harm <= 0.25)
    exploratory = bool(all_supported and gain >= 0.0
        and window_rows["D1"]["harm_vs_same_cutoff_primary"] <= 0.02
        and window_rows["D2"]["harm_vs_same_cutoff_primary"] <= 0.02
        and max_group_harm is not None and max_group_harm <= 0.25)
    _need(report["qualification_gate"]["pass"] is qualification
          and report["exploratory_gate"]["pass"] is exploratory
          and report["root_review_eligible"] is bool(qualification or exploratory),
          "Qualification/exploratory/root-review gate replay differs")
    return {"window_gains_vs_primary": {
                key: window_rows[key]["gain_vs_same_cutoff_primary"] for key in ("D1", "D2")},
            "pooled_gain_vs_primary": gain, "pooled_harm_vs_primary": harm,
            "pooled_raw_r06_gain_vs_candidate": (
                pooled["full_rows"]["metrics"]["raw_r06"]["mae"] - candidate_mae),
            "pooled_direct_gain_vs_candidate": (
                pooled["full_rows"]["metrics"]["same_cutoff_direct_old_hinge"]["mae"]
                - candidate_mae),
            "maximum_supported_group_harm_vs_primary": max_group_harm,
            "maximum_group_harm_location": "target_le_5/D1",
            "all_groups_and_cycles_supported": all_supported,
            "qualification_pass": qualification,
            "exploratory_pass": exploratory,
            "root_review_eligible": bool(qualification or exploratory),
            "pooled_candidate_mae": candidate_mae,
            "pooled_primary_mae": primary_mae,
            "pooled_candidate_p90_abs_error": pooled["full_rows"]["metrics"][
                "new_r06_native_hinge"]["p90_absolute_error"],
            "pooled_candidate_mean_error": pooled["full_rows"]["metrics"][
                "new_r06_native_hinge"]["mean_error_prediction_minus_target"],
            "pooled_rows": len(pooled_idx),
            "pooled_present_rows": pooled["present_rows"],
            "pooled_missing_rows": pooled["missing_rows"]}


def verify() -> dict[str, Any]:
    freeze, freeze_sha, pass_sha = _verify_authorization_and_freeze()
    preflight, _preflight_sha = _signed(PREFLIGHT)
    diagnostics, diagnostics_sha = _signed(DIAGNOSTICS)
    _need(diagnostics.get("experiment_id") == freeze["experiment_id"]
          and diagnostics.get("status") == "TWO_DIAGNOSTICS_COMPLETE_PENDING_ROOT_REVIEW"
          and diagnostics.get("prefit_freeze_sha256") == freeze_sha
          and diagnostics.get("root_source_pass_sha256") == pass_sha
          and diagnostics.get("source_commit") == freeze["source_commit"]
          and diagnostics.get("source_hashes") == freeze["source_hashes"]
          and diagnostics.get("final_fit_started") is False
          and diagnostics.get("hinge_final_used_in_diagnostic_gates") is False
          and diagnostics.get("test_labels_read") is False
          and diagnostics.get("csv_created") is False
          and diagnostics.get("kaggle_accessed") is False,
          "Diagnostic report identity or access-boundary fields differ")
    labels, targets, raw = _load_identity_and_targets(freeze, preflight)
    cycles = labels["cycles"].astype(np.int64)
    fit_indices = {
        FIT_IDS[0]: np.flatnonzero((cycles >= 68) & (cycles <= 71)),
        FIT_IDS[1]: np.flatnonzero((cycles >= 68) & (cycles <= 75)),
    }
    ledger = [json.loads(line) for line in LEDGER.read_text(encoding="utf-8").splitlines()
              if line.strip()]
    _need(len(ledger) == 6 and sum(event.get("event") == "LP_STARTED" for event in ledger) == 2,
          "R2C ledger does not contain exactly two official LP starts")
    _need([event.get("fit_id") for event in ledger] ==
          [FIT_IDS[0]] * 3 + [FIT_IDS[1]] * 3,
          "R2C ledger fit order differs from D1 then D2")
    _need(set(event.get("source_commit") for event in ledger) ==
          {"a82af2eaa909daac884b393be636cc6c736722ce"},
          "Fit ledger source commit differs from the authorized root-pass commit")
    states = {}
    fit_checks = []
    for fit_id in FIT_IDS:
        fit_events = [event for event in ledger if event.get("fit_id") == fit_id]
        state, fit_check = _verify_fit(fit_id, fit_indices[fit_id], labels, targets,
                                       raw, freeze_sha, pass_sha, fit_events)
        states[fit_id] = state
        fit_checks.append(fit_check)
    _need(not (RUN / "fits" / "R06_HINGE_FINAL.json").exists()
          and not (EXP / "ROOT_FINAL_FIT_APPROVAL_R2C.json").exists(),
          "Unexpected FINAL state or approval exists")
    metric_replay = _replay_report(labels, targets, raw, states, freeze, preflight, diagnostics)
    body = {
        "schema_version": 1,
        "experiment_id": "REG_R06_NATIVE_HINGE_V1_R2C",
        "status": "PASS_INDEPENDENT_DIAGNOSTICS_REPLAY",
        "audit_scope": "fit inputs, solver objective/slacks, R06 row identity, report metrics, gates, and append-only ledger",
        "audit_script_sha256": _sha(Path(__file__).resolve()),
        "prefit_freeze_sha256": freeze_sha,
        "root_source_pass_sha256": pass_sha,
        "diagnostics_sha256": diagnostics_sha,
        "fit_replays": fit_checks,
        "metric_and_gate_replay": metric_replay,
        "ledger": {"events": len(ledger), "official_lp_starts": 2,
                   "fit_ids": list(FIT_IDS), "final_fit_started": False,
                   "all_fit_slots_completed_once": True},
        "data_boundary": {"target_values_loaded_after_root_pass": True,
                           "test_labels_read": False, "csv_created": False,
                           "kaggle_accessed": False, "base_or_preprocessing_fits": 0},
    }
    body["signature"] = _canonical_sha(body)
    payload = json.dumps(body, sort_keys=True, ensure_ascii=False,
                         allow_nan=False, indent=2).encode("utf-8") + b"\n"
    if VERIFICATION.exists():
        _need(VERIFICATION.read_bytes() == payload,
              "Independent diagnostics verification receipt already exists with different bytes")
    else:
        VERIFICATION.parent.mkdir(parents=True, exist_ok=True)
        with VERIFICATION.open("xb") as stream:
            stream.write(payload)
            stream.flush()
    return body


if __name__ == "__main__":
    print(json.dumps(verify(), sort_keys=True, ensure_ascii=False))
