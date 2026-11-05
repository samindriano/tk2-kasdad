"""Independent target-free source and synthetic-metric verifier for the present-only R06 hinge."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

from scripts import reg_r06_present_native_hinge_v1_r2 as pilot


class VerificationError(RuntimeError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8", errors="surrogatepass")).hexdigest()


def _signed_json(path: Path) -> tuple[dict[str, Any], str]:
    _need(path.is_file(), f"Missing signed JSON: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    signature = data.pop("signature", None)
    _need(signature == _canonical_sha(data), f"Invalid canonical signature: {path}")
    data["signature"] = signature
    return data, _sha(path)


def _array_sha(value: Any) -> str:
    arr = np.asarray(value)
    prefix = json.dumps({"dtype": arr.dtype.str, "shape": list(arr.shape)},
                        sort_keys=True, ensure_ascii=False, allow_nan=False,
                        separators=(",", ":")).encode("utf-8", errors="surrogatepass") + b"\0"
    if arr.dtype.kind in "USO":
        payload = b"\0".join(str(item).encode("utf-8", errors="surrogatepass")
                              for item in arr.reshape(-1).tolist())
    else:
        payload = np.ascontiguousarray(arr).tobytes(order="C")
    return hashlib.sha256(prefix + payload).hexdigest()


def _read_npz(path: Path, members: tuple[str, ...]) -> tuple[dict[str, np.ndarray], list[str]]:
    with np.load(path, allow_pickle=False) as archive:
        all_members = list(archive.files)
        _need(set(members).issubset(all_members), f"Missing NPZ member in {path}")
        selected = {name: np.array(archive[name], copy=True) for name in members}
    return selected, all_members


def _records(ids: np.ndarray, rows: np.ndarray, cycles: np.ndarray,
             origins: np.ndarray, indices: np.ndarray) -> list[dict[str, Any]]:
    return [{"record_id": str(ids[i]), "physical_row": int(rows[i]),
             "cycle": int(cycles[i]), "prediction_origin": int(origins[i])}
            for i in indices]


def _check_row_cache_alignment(freeze: dict[str, Any], preflight: dict[str, Any],
                               fit_population: dict[str, Any]) -> dict[str, Any]:
    paths = {name: Path(item["path"]) for name, item in freeze["input_hashes"].items()}
    labels, label_members = _read_npz(paths["oof_label_and_identity_bundle"], (
        "physical_rows", "record_ids", "prediction_origins", "original_bundle_indices",
        "cycles", "missing_mask", "direct_predictions"))
    ids = labels["record_ids"].astype(str)
    cycles = labels["cycles"].astype(int)
    rows = labels["physical_rows"].astype(int)
    origins_expected = labels["prediction_origins"].astype(int)
    missing = labels["missing_mask"].astype(bool)
    direct = labels["direct_predictions"].astype(np.float64)
    _need(len(ids) == pilot.EXPECTED_ROWS and len(set(ids.tolist())) == len(ids),
          "Cached label/ID identity is not unique")
    _need(len(set(rows.tolist())) == len(rows), "Cached physical rows are not unique")
    _need("targets" in label_members, "Cached label bundle lost its target member")
    _need(np.array_equal(np.unique(cycles), np.arange(68, 80)), "Cached label cycles changed")
    _need(missing.dtype.kind == "b", "Cached missing mask is not boolean")
    index = {record_id: i for i, record_id in enumerate(ids.tolist())}
    raw = np.full(len(ids), np.nan, dtype=np.float64)
    origins = np.full(len(ids), -1, dtype=np.int16)
    observed_members: dict[str, list[str]] = {}
    for origin, cycle_member, allowed in (
            (67, "audit__cycle_code", set(range(68, 76))),
            (75, "score_cycles", set(range(76, 80)))):
        arrays, names = _read_npz(paths[f"r06_fit{origin}_npz"], (
            "score_rows", "score_record_ids", "prediction", cycle_member))
        cache_ids = arrays["score_record_ids"].astype(str)
        cache_cycles = np.asarray(arrays[cycle_member], dtype=int)
        cache_rows = np.asarray(arrays["score_rows"], dtype=int)
        predictions = np.asarray(arrays["prediction"], dtype=np.float64)
        mask = np.isin(cache_cycles, list(allowed))
        actual = {(str(cache_ids[j]), int(cache_cycles[j]), int(cache_rows[j]))
                  for j in np.flatnonzero(mask)}
        expected = {(str(ids[j]), int(cycles[j]), int(rows[j]))
                    for j in range(len(ids)) if int(cycles[j]) in allowed}
        _need(actual == expected, f"Fit{origin} row IDs, cycles, or positions changed")
        for j in np.flatnonzero(mask):
            target_i = index[str(cache_ids[j])]
            _need(origins[target_i] == -1, "Duplicate R06 source assignment")
            raw[target_i] = predictions[j]
            origins[target_i] = origin
        observed_members[str(origin)] = names
    _need(not np.isnan(raw).any(), "R06 OOF vector has rows without a source")
    _need(np.array_equal(origins, origins_expected),
          "R06 E67/E75 source assignments differ from cached identity")
    _need(np.array_equal(direct[missing], raw[missing]),
          "Missing DIRECT fallback differs from raw R06 bit-for-bit")
    _need(label_members == preflight["npz_member_audit"]["label_bundle_members_seen"],
          "Label NPZ member inventory changed")
    _need(observed_members == preflight["npz_member_audit"]["r06_cache_members_seen"],
          "R06 NPZ member inventory changed")

    populations: dict[str, Any] = {}
    score_hashes: dict[str, str] = {}
    for fit_id, fit_cycle_mask, score_cycle_mask in (
            (pilot.FIT_IDS[0], (cycles >= 68) & (cycles <= 71),
             (cycles >= 72) & (cycles <= 75)),
            (pilot.FIT_IDS[1], (cycles >= 68) & (cycles <= 75),
             (cycles >= 76) & (cycles <= 79))):
        fit_all = np.flatnonzero(fit_cycle_mask)
        present = fit_all[~missing[fit_all]]
        excluded = fit_all[missing[fit_all]]
        score = np.flatnonzero(score_cycle_mask)
        populations[fit_id] = {
            "fit_rows_before_missing_exclusion": len(fit_all),
            "present_fit_rows": len(present),
            "missing_fit_rows_excluded": len(excluded),
            "present_rows": _records(ids, rows, cycles, origins_expected, present),
            "excluded_missing_rows": _records(ids, rows, cycles, origins_expected, excluded),
        }
        score_hashes[fit_id] = _array_sha(ids[score])
    _need(fit_population.get("fit_populations") == populations,
          "Committed fit-population row IDs differ from non-target identity")
    _need(fit_population.get("score_record_id_hashes") == score_hashes,
          "Score-window record identity changed")
    _need(preflight.get("fit_population_ids_sha256") == _sha(pilot.FIT_POPULATION_IDS),
          "Preflight does not bind the exact fit-population ID artifact")
    _need(fit_population.get("fit_record_id_hashes") == {
        pilot.FIT_IDS[0]: _array_sha(np.asarray(
            [item["record_id"] for item in populations[pilot.FIT_IDS[0]]["present_rows"]])),
        pilot.FIT_IDS[1]: _array_sha(np.asarray(
            [item["record_id"] for item in populations[pilot.FIT_IDS[1]]["present_rows"]])),
    }, "Present fit record ID hashes differ")
    _need(preflight.get("row_counts") == {
        "fit_D1_cycle_rows": 4_950,
        "fit_D1_missing_excluded": pilot.EXPECTED_MISSING_FIT_D1,
        "fit_D1_present_rows": pilot.EXPECTED_PRESENT_FIT_D1,
        "score_D1": pilot.EXPECTED_SCORE_D1,
        "fit_D2_cycle_rows": 12_412,
        "fit_D2_missing_excluded": pilot.EXPECTED_MISSING_FIT_D2,
        "fit_D2_present_rows": pilot.EXPECTED_PRESENT_FIT_D2,
        "score_D2": pilot.EXPECTED_SCORE_D2,
        "conditional_final_present_rows": 19_528,
    }, "Preflight row counts differ from the exact present-only contract")
    return {
        "rows": len(ids),
        "unique_record_ids": len(set(ids.tolist())),
        "unique_physical_rows": len(set(rows.tolist())),
        "cycle_counts": {str(c): int(np.sum(cycles == c)) for c in range(68, 80)},
        "r06_source_counts": {str(origin): int(np.sum(origins == origin))
                              for origin in (67, 75)},
        "r06_prediction_sha256": _array_sha(raw),
        "missing_rows": int(missing.sum()),
        "present_fit_rows": {key: value["present_fit_rows"]
                             for key, value in populations.items()},
        "excluded_missing_fit_rows": {key: value["missing_fit_rows_excluded"]
                                      for key, value in populations.items()},
        "targets_member_seen_but_not_loaded": True,
        "target_values_loaded": False,
        "test_labels_read": False,
    }


def _check_r4_lineage(freeze: dict[str, Any]) -> dict[str, Any]:
    records = freeze["input_hashes"]
    names = ("r4_root_source_pass", "r4_diagnostics", "r4_diagnostics_verification",
             "r4_hinge_d1_state", "r4_hinge_d2_state", "r4_hinge_d1_input",
             "r4_hinge_d2_input", "r4_lp_ledger")
    paths = {name: Path(records[name]["path"]) for name in names}
    root_pass = json.loads(paths["r4_root_source_pass"].read_text(encoding="utf-8"))
    diagnostics = json.loads(paths["r4_diagnostics"].read_text(encoding="utf-8"))
    verification = json.loads(paths["r4_diagnostics_verification"].read_text(encoding="utf-8"))
    root_pass_sha = _sha(paths["r4_root_source_pass"])
    _need(root_pass.get("verdict") == "PASS"
          and root_pass.get("authorized_fit_ids") == ["HINGE_D1", "HINGE_D2"],
          "R4 same-cutoff comparators were not authorized")
    _need(diagnostics.get("root_source_pass_sha256") == root_pass_sha
          and diagnostics.get("diagnostic_fit_ids") == ["HINGE_D1", "HINGE_D2"],
          "R4 diagnostics do not bind the expected source pass")
    _need(verification.get("status") == "PASS_INDEPENDENT_DIAGNOSTICS_REPLAY"
          and verification.get("root_source_pass_sha256") == root_pass_sha,
          "R4 independent replay did not pass")
    ledger = [json.loads(line) for line in paths["r4_lp_ledger"].read_text(
        encoding="utf-8").splitlines() if line.strip()]
    expected = {"HINGE_D1": ("r4_hinge_d1_state", "r4_hinge_d1_input", 4_950),
                "HINGE_D2": ("r4_hinge_d2_state", "r4_hinge_d2_input", 12_412)}
    frozen = freeze["comparators"]["historical_cutoff_states"]
    for fit_id, (state_key, input_key, count) in expected.items():
        state_path, input_path = paths[state_key], paths[input_key]
        state = json.loads(state_path.read_text(encoding="utf-8"))
        solver = state["solver_result"]
        _need(state.get("status") == "LP_COMPLETE" and state.get("fit_id") == fit_id
              and solver.get("success") is True and solver.get("fit_rows") == count
              and state.get("fit_input", {}).get("fit_input_bundle_sha256") == _sha(input_path),
              f"R4 {fit_id} state identity differs")
        _need(frozen[fit_id].get("state_sha256") == _sha(state_path)
              and frozen[fit_id].get("fit_input_sha256") == _sha(input_path),
              f"R4 {fit_id} is not bound by the freeze")
        events = [event for event in ledger if event.get("fit_id") == fit_id]
        reserve = [event for event in events if event.get("event") == "RESERVE"]
        complete = [event for event in events if event.get("event") == "LP_COMPLETE"]
        _need(len(reserve) == 1 and len(complete) == 1
              and complete[0].get("state_sha256") == _sha(state_path)
              and reserve[0].get("root_source_pass_sha256") == root_pass_sha,
              f"R4 {fit_id} ledger link differs")
    return {"root_source_pass_sha256": root_pass_sha, "diagnostic_fit_ids": list(expected)}


def _check_all_row_native_lineage(freeze: dict[str, Any]) -> dict[str, Any]:
    records = freeze["input_hashes"]
    names = ("r2c_root_source_pass", "r2c_diagnostics", "r2c_independent_verification",
             "r2c_lp_ledger", "r2c_d1_state", "r2c_d2_state", "r2c_d1_input", "r2c_d2_input")
    paths = {name: Path(records[name]["path"]) for name in names}
    root_pass, root_sha = _signed_json(paths["r2c_root_source_pass"])
    diagnostics, _ = _signed_json(paths["r2c_diagnostics"])
    verification, _ = _signed_json(paths["r2c_independent_verification"])
    _need(root_pass.get("experiment_id") == "REG_R06_NATIVE_HINGE_V1_R2C"
          and root_pass.get("verdict") == "PASS_SOURCE_AND_INPUTS"
          and root_pass.get("authorized_fit_ids") == ["R06_HINGE_D1", "R06_HINGE_D2"]
          and root_pass.get("final_fit_authorized") is False,
          "R2C source-pass scope differs")
    _need(diagnostics.get("root_source_pass_sha256") == root_sha
          and diagnostics.get("qualification_gate", {}).get("pass") is False
          and diagnostics.get("exploratory_gate", {}).get("pass") is False
          and diagnostics.get("root_review_eligible") is False,
          "R2C all-row native negative diagnostic result changed")
    _need(verification.get("status") == "PASS_INDEPENDENT_DIAGNOSTICS_REPLAY"
          and verification.get("root_source_pass_sha256") == root_sha,
          "R2C independent diagnostic replay did not pass")
    ledger = [json.loads(line) for line in paths["r2c_lp_ledger"].read_text(
        encoding="utf-8").splitlines() if line.strip()]
    _need(len(ledger) == 6
          and sum(event.get("event") == "LP_STARTED" for event in ledger) == 2
          and {event.get("fit_id") for event in ledger} == {"R06_HINGE_D1", "R06_HINGE_D2"}
          and {event.get("source_commit") for event in ledger} ==
          {"a82af2eaa909daac884b393be636cc6c736722ce"}
          and not any(event.get("fit_id") == "R06_HINGE_FINAL" for event in ledger),
          "R2C prior two-slot ledger differs or contains a FINAL")
    states = {}
    frozen = freeze["comparators"]["all_row_native_states"]
    for fit_id, state_key, input_key, expected_rows in (
            ("R06_HINGE_D1", "r2c_d1_state", "r2c_d1_input", 4_950),
            ("R06_HINGE_D2", "r2c_d2_state", "r2c_d2_input", 12_412)):
        state_path, input_path = paths[state_key], paths[input_key]
        state = json.loads(state_path.read_text(encoding="utf-8"))
        solver = state.get("solver_result", {})
        history = frozen[fit_id]
        events = [event for event in ledger if event.get("fit_id") == fit_id]
        reserve = [event for event in events if event.get("event") == "RESERVE"]
        complete = [event for event in events if event.get("event") == "LP_COMPLETE"]
        _need(state.get("status") == "LP_COMPLETE" and state.get("fit_id") == fit_id
              and solver.get("success") is True and solver.get("fit_rows") == expected_rows
              and state.get("fit_input", {}).get("fit_input_bundle_sha256") == _sha(input_path),
              f"R2C {fit_id} state identity differs")
        _need(history.get("state_sha256") == _sha(state_path)
              and history.get("fit_input_sha256") == _sha(input_path)
              and history.get("fit_rows") == expected_rows,
              f"R2C {fit_id} is not bound by the new freeze")
        _need(len(reserve) == 1 and len(complete) == 1
              and complete[0].get("state_sha256") == _sha(state_path)
              and reserve[0].get("root_source_pass_sha256") == root_sha,
              f"R2C {fit_id} ledger link differs")
        states[fit_id] = {"state_sha256": _sha(state_path), "fit_input_sha256": _sha(input_path)}
    _need(freeze.get("prior_slot_accounting", {}).get(
        "prior_native_lp_starts_already_consumed") == 2
        and freeze.get("prior_slot_accounting", {}).get("old_native_final_slot_repurposed") is False,
        "Prior native budget accounting differs")
    return {"root_source_pass_sha256": root_sha, "diagnostic_states": states,
            "prior_native_lp_starts_already_consumed": 2, "final_started": False}


def _apply(raw: np.ndarray, missing: np.ndarray, coefficients: dict[str, Any]) -> np.ndarray:
    low = np.minimum(raw, pilot.KNOT)
    high = np.maximum(raw - pilot.KNOT, 0.0)
    candidate = np.clip(float(coefficients["intercept"])
                        + float(coefficients["slope_low"]) * low
                        + float(coefficients["slope_high"]) * high, 0.0, 100.0)
    return np.where(missing, raw, candidate)


def _metric(y: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    err = np.asarray(prediction, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    absolute = np.abs(err)
    return {"n": len(err), "mae": float(np.mean(absolute)),
            "p90_absolute_error": float(np.quantile(absolute, 0.90, method="linear")),
            "mean_error_prediction_minus_target": float(np.mean(err))}


def _check_synthetic_metric_e2e(receipt: dict[str, Any]) -> dict[str, Any]:
    synthetic = receipt.get("metrics_e2e", {})
    _need(synthetic.get("status") == "PASS_SYNTHETIC_METRIC_E2E",
          "Synthetic CLI did not record a metric end-to-end pass")
    per_cycle = 400
    cycles = np.repeat(np.arange(68, 80, dtype=np.int16), per_cycle)
    pattern = np.tile(np.asarray([0.0, 2.0, 4.0, 50.0, 70.0, 90.0, 96.0, 100.0]), 50)
    targets = np.tile(pattern, 12).astype(np.float64)
    row_index = np.arange(len(cycles))
    raw = np.clip(targets + 4.0 + 6.0 * np.sin(row_index * 0.071), 0.0, 100.0)
    missing = (row_index % 4) == 0
    direct = raw.copy()
    primary_coeffs = {
        "D1": {"intercept": 1.5, "slope_low": 0.98, "slope_high": 0.99, "knot": 50.0},
        "D2": {"intercept": 1.2, "slope_low": 0.99, "slope_high": 0.98, "knot": 50.0},
    }
    allrow_coeffs = {
        "D1": {"intercept": 2.4, "slope_low": 0.96, "slope_high": 0.96, "knot": 50.0},
        "D2": {"intercept": 2.1, "slope_low": 0.97, "slope_high": 0.95, "knot": 50.0},
    }
    candidate_coeffs = synthetic.get("candidate_coefficients", {})
    _need(set(candidate_coeffs) == set(pilot.DIAGNOSTIC_IDS),
          "Synthetic receipt is missing one of the two fitted control slots")
    window_arrays: dict[str, dict[str, np.ndarray]] = {}
    for window, fit_id, score_mask in (
            ("D1", pilot.FIT_IDS[0], (cycles >= 72) & (cycles <= 75)),
            ("D2", pilot.FIT_IDS[1], (cycles >= 76) & (cycles <= 79))):
        indices = np.flatnonzero(score_mask)
        coefficients = candidate_coeffs[fit_id]
        _need(coefficients.get("knot") == 50.0
              and 0.0 <= coefficients.get("slope_low", -1.0) <= 2.0
              and 0.0 <= coefficients.get("slope_high", -1.0) <= 2.0,
              f"Synthetic {window} fitted coefficients violate fixed source bounds")
        window_arrays[window] = {
            "candidate": _apply(raw[indices], missing[indices], coefficients),
            "all_row_native": _apply(raw[indices], missing[indices], allrow_coeffs[window]),
            "transported_r4_primary": _apply(raw[indices], missing[indices], primary_coeffs[window]),
            "matching_r4_direct": _apply(direct[indices], missing[indices], primary_coeffs[window]),
            "raw_r06": raw[indices],
            "targets": targets[indices],
            "missing": missing[indices],
            "cycles": cycles[indices],
        }
    report = synthetic["fit_and_score_metrics"]
    pool = {key: np.concatenate([window_arrays[name][key] for name in ("D1", "D2")])
            for key in ("candidate", "all_row_native", "transported_r4_primary",
                        "matching_r4_direct", "raw_r06", "targets", "missing", "cycles")}
    name_map = {
        "candidate": "present_native_hinge",
        "all_row_native": "all_row_native_hinge",
        "transported_r4_primary": "transported_r4_hinge_on_r06",
        "matching_r4_direct": "matching_r4_hinge_on_direct",
        "raw_r06": "raw_r06",
    }
    audited_maes = {}
    summaries = {"D1": report["windows"]["D1"], "D2": report["windows"]["D2"],
                 "pooled": report["pooled_unique"]}
    arrays = {**window_arrays, "pooled": pool}
    for window in ("D1", "D2", "pooled"):
        actual = arrays[window]
        full = summaries[window]["full_rows"]["metrics"]
        audited_maes[window] = {}
        for local, metric_name in name_map.items():
            replay = _metric(actual["targets"], actual[local])
            _need(full[metric_name]["n"] == replay["n"]
                  and abs(full[metric_name]["mae"] - replay["mae"]) <= 1e-12,
                  f"Synthetic {window} {metric_name} MAE differs on independent replay")
            audited_maes[window][metric_name] = replay["mae"]
    _need(synthetic.get("score_rows") == len(pool["targets"]) == 3_200,
          "Synthetic metric E2E did not use two disjoint windows")
    _need(all(item["all_required_groups_supported"] and item["all_scored_cycles_supported"]
              for item in (report["pooled_unique"], *report["windows"].values())),
          "Synthetic metric E2E has unsupported groups or cycles")
    gains = {window: audited_maes[window]["all_row_native_hinge"]
             - audited_maes[window]["present_native_hinge"] for window in ("D1", "D2")}
    _need(report["mechanism_gate"]["window_gains"] == gains
          and report["mechanism_gate"]["pass"] is (gains["D1"] > 0.0 and gains["D2"] > 0.0),
          "Synthetic mechanism gate differs from independent replay")
    return {"status": "PASS", "score_rows": len(pool["targets"]),
            "independently_replayed_mae": audited_maes,
            "mechanism_window_gains": gains}


def verify_source_ready(*, write_receipt: bool = False) -> dict[str, Any]:
    freeze, freeze_sha = _signed_json(pilot.FREEZE)
    preflight, preflight_sha = _signed_json(pilot.PREFLIGHT)
    fit_population, fit_population_sha = _signed_json(pilot.FIT_POPULATION_IDS)
    synthetic, synthetic_sha = _signed_json(pilot.SYNTHETIC_QA)
    _need(freeze.get("experiment_id") == pilot.EXPERIMENT_ID
          and freeze.get("status") ==
          "IDENTITY_FROZEN_PENDING_NON_TARGET_PREFLIGHT_AND_ROOT_REVIEW_R2",
          "Prefit freeze is not pending source review")
    _need(preflight.get("status") ==
          "PASS_NON_TARGET_IDENTITY_PREFLIGHT_PENDING_ROOT_SOURCE_PASS_R2"
          and preflight.get("prefit_freeze_sha256") == freeze_sha,
          "Non-target preflight is missing or not bound to the freeze")
    _need(fit_population.get("prefit_freeze_sha256") == freeze_sha
          and preflight.get("fit_population_ids_sha256") == fit_population_sha,
          "Present-only fit row list is not bound to the source freeze and preflight")
    _need(synthetic.get("status") == "PASS_SYNTHETIC_SOURCE_AND_RECEIPT_E2E"
          and synthetic.get("checks", {}).get("official_lp_starts") == 0
          and synthetic.get("checks", {}).get("target_values_loaded") is False
          and synthetic.get("checks", {}).get("test_labels_read") is False
          and synthetic.get("checks", {}).get("csv_created") is False
          and synthetic.get("checks", {}).get("kaggle_accessed") is False,
          "Synthetic CLI receipt violates the source-only boundary")
    _need(all(value is True for key, value in synthetic.get("checks", {}).items()
              if key not in {"official_lp_starts", "target_values_loaded", "test_labels_read",
                             "csv_created", "kaggle_accessed"}),
          "Synthetic source checks contain a failure")
    _need(not pilot.ROOT_SOURCE_PASS.exists() and not pilot.LEDGER.exists(),
          "Source-ready verification must finish before root approval or official ledger")
    _need(freeze.get("official_lp_starts") == 0 and preflight.get("official_lp_starts") == 0
          and preflight.get("target_reader_called") is False
          and preflight.get("train_target_values_loaded") is False
          and preflight.get("score_target_values_loaded") is False,
          "Target values or official LPs crossed the preflight boundary")
    _need(preflight.get("test_labels_read") is False
          and preflight.get("test_or_sample_accessed") is False
          and preflight.get("kaggle_accessed") is False,
          "Test/sample or Kaggle access is recorded")
    _need(freeze.get("prior_slot_accounting", {}).get("old_r4_lp_starts_already_consumed") == 3
          and freeze.get("prior_slot_accounting", {}).get(
              "prior_native_lp_starts_already_consumed") == 2
          and freeze.get("prior_slot_accounting", {}).get("new_pilot_diagnostic_slot_cap") == 2
          and freeze.get("prior_slot_accounting", {}).get(
              "new_pilot_conditional_final_slot_cap") == 1
          and freeze.get("prior_slot_accounting", {}).get("old_native_final_slot_repurposed") is False
          and freeze.get("prior_slot_accounting", {}).get("old_ledger_reset_or_rewritten") is False,
          "Prior and new budget accounting is incomplete")
    for name, expected in freeze["source_hashes"].items():
        _need(_sha(Path(pilot.SOURCE_FILES[name])) == expected,
              f"Source changed after freeze: {name}")
    for name, record in freeze["input_hashes"].items():
        path = Path(record["path"])
        _need(path.is_file() and _sha(path) == record["sha256"],
              f"Frozen input changed: {name}")
    ancestor = subprocess.run(
        ["git", "-C", str(pilot.ROOT), "merge-base", "--is-ancestor",
         str(freeze["source_commit"]), "HEAD"], capture_output=True, text=True, check=False)
    _need(ancestor.returncode == 0, "Frozen source commit is not an ancestor")
    r4_audit = _check_r4_lineage(freeze)
    r2c_audit = _check_all_row_native_lineage(freeze)
    row_audit = _check_row_cache_alignment(freeze, preflight, fit_population)
    _need(row_audit["r06_prediction_sha256"] == preflight["array_hashes"]["r06_oof_prediction"],
          "Independent R06 prediction hash differs from preflight")
    _need(row_audit["present_fit_rows"] == {
        pilot.FIT_IDS[0]: pilot.EXPECTED_PRESENT_FIT_D1,
        pilot.FIT_IDS[1]: pilot.EXPECTED_PRESENT_FIT_D2,
    }, "Independent present-only fit counts differ")
    resources = preflight.get("resource_snapshots", [])
    _need(len(resources) >= 2 and all(item.get("status") == "PASS" for item in resources),
          "Resource preflight did not pass at every recorded boundary")
    metrics_audit = _check_synthetic_metric_e2e(synthetic)
    body = {
        "schema_version": 1,
        "experiment_id": pilot.EXPERIMENT_ID,
        "status": "PASS_SOURCE_READY_PENDING_ROOT_RELEASE_R2",
        "prefit_freeze_sha256": freeze_sha,
        "identity_preflight_sha256": preflight_sha,
        "fit_population_ids_sha256": fit_population_sha,
        "synthetic_qa_sha256": synthetic_sha,
        "historical_r4_comparator_audit": r4_audit,
        "prior_all_row_native_mechanism_control_audit": r2c_audit,
        "row_identity_audit": row_audit,
        "synthetic_metric_replay": metrics_audit,
        "official_lp_starts": 0,
        "root_source_pass_exists": False,
        "test_labels_read": False,
        "csv_created": False,
        "kaggle_accessed": False,
        "review_scope": "present-only source and input preparation; no official fit or target values loaded",
    }
    if write_receipt:
        body["signature"] = _canonical_sha(body)
        pilot.write_once(pilot.VERIFICATION, body)
        return {"status": body["status"], "path": str(pilot.VERIFICATION.resolve()),
                "sha256": _sha(pilot.VERIFICATION), "fit_starts": 0}
    return body


def main() -> int:
    try:
        result = verify_source_ready(write_receipt=True)
        print(json.dumps(result, sort_keys=True, ensure_ascii=False))
        return 0
    except (VerificationError, pilot.GateError) as exc:
        print(json.dumps({"status": "FAIL_SOURCE_READY", "error": str(exc)},
                         ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
