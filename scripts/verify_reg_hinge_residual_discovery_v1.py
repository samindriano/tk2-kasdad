"""Independent row/hash/numeric reconstruction for the frozen R1 residual audit."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np
import psutil
from threadpoolctl import threadpool_limits

from scripts import reg_hinge_residual_discovery_v1 as r
from scripts import reg_monotone_hinge_p50_v4 as r4


class VerificationError(RuntimeError):
    pass


EXPECTED_ORIGIN = {("HINGE_D1", "fit"): 67, ("HINGE_D1", "score"): 67,
                   ("HINGE_D2", "fit"): 67, ("HINGE_D2", "score"): 75}
EXPECTED_CYCLES = {
    ("HINGE_D1", "fit"): tuple(range(68, 72)),
    ("HINGE_D1", "score"): tuple(range(72, 76)),
    ("HINGE_D2", "fit"): tuple(range(68, 76)),
    ("HINGE_D2", "score"): tuple(range(76, 80)),
}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _canonical_sha(value: Mapping[str, Any]) -> str:
    data = json.dumps(dict(value), sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _write_once(path: Path, value: Mapping[str, Any]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(value), sort_keys=True, ensure_ascii=False,
                         allow_nan=False, indent=2).encode("utf-8") + b"\n"
    try:
        with path.open("xb") as stream:
            stream.write(payload)
    except FileExistsError as exc:
        raise VerificationError(f"Refusing to overwrite independent verification: {path}") from exc
    return hashlib.sha256(payload).hexdigest()


def _head() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=r.ROOT,
                            capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise VerificationError("Cannot resolve source commit during independent replay")
    return result.stdout.strip()


def _verify_freeze() -> tuple[dict[str, Any], str]:
    if not r.FREEZE.is_file():
        raise VerificationError("Frozen precomputation manifest is absent")
    freeze = json.loads(r.FREEZE.read_text(encoding="utf-8"))
    signature = freeze.pop("signature", None)
    if signature != _canonical_sha(freeze):
        raise VerificationError("Independent freeze signature replay failed")
    freeze["signature"] = signature
    if freeze.get("status") != "FROZEN_BEFORE_NUMERICAL_ARRAY_ACCESS":
        raise VerificationError("Freeze status does not assert a precomputation boundary")
    if _head() != freeze.get("source_commit"):
        raise VerificationError("Source HEAD differs from the frozen analysis commit")
    if r._resource_policy_body() != freeze.get("resource_policy"):
        raise VerificationError("Saved-replay resource policy differs from frozen values")
    for item in freeze.get("source_and_input_files", {}).values():
        path = Path(item["path"])
        if not path.is_file() or _sha(path) != item["sha256"]:
            raise VerificationError(f"Frozen source/input bytes changed: {path}")
    return freeze, _sha(r.FREEZE)


def _resource_gate(boundary: str, *, initial: bool,
                   limits: Mapping[str, int]) -> dict[str, Any]:
    available = int(psutil.virtual_memory().available)
    estimate = int(limits["incremental_estimate_bytes"])
    projected = available - estimate
    initial_ok = (available >= int(limits["minimum_available_bytes"])
                  and projected >= int(limits["minimum_projected_free_bytes"]))
    actual_ok = (available >= int(limits["minimum_actual_free_at_boundary_bytes"])
                 and projected >= int(limits["minimum_projected_free_bytes"]))
    if not ((initial_ok and actual_ok) if initial else actual_ok):
        raise VerificationError(f"Independent verifier resource gate failed at {boundary}: available={available}, projected={projected}")
    memory = psutil.Process(os.getpid()).memory_info()
    peak = getattr(memory, "peak_wset", None)
    return {"status": "PASS", "boundary": boundary, "available_bytes": available,
        "incremental_estimate_bytes": estimate, "projected_free_bytes": projected,
        "minimum_initial_available_bytes": int(limits["minimum_available_bytes"]),
        "minimum_projected_free_bytes": int(limits["minimum_projected_free_bytes"]),
        "minimum_actual_free_at_boundary_bytes": int(limits["minimum_actual_free_at_boundary_bytes"]),
        "initial_gate_pass": bool(initial_ok), "actual_boundary_pass": bool(actual_ok),
        "process_rss_bytes": int(memory.rss), "process_rss_measured": True,
        "os_peak_working_set_bytes": int(peak) if peak is not None else None,
        "os_peak_working_set_measured": peak is not None,
        "thread_limit": 2, "concurrent_heavy_jobs": 1}


def _load(path: Path, names: tuple[str, ...], *,
          permitted_unloaded: tuple[str, ...] = ()) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if not set(names).issubset(archive.files) or (set(archive.files) - set(names)) != set(permitted_unloaded):
            raise VerificationError(f"Independent NPZ schema mismatch: {path}")
        return {name: np.asarray(archive[name]).copy() for name in names}


def _params(fit_state: Mapping[str, Any], global_state: Mapping[str, Any]) -> tuple[dict[str, float], tuple[float, float]]:
    h = fit_state.get("solver_result", {})
    g = global_state.get("solver_result", {})
    if fit_state.get("status") != "LP_COMPLETE" or h.get("success") is not True:
        raise VerificationError("Saved HINGE fit is not complete")
    if global_state.get("status") != "LP_COMPLETE" or g.get("success") is not True:
        raise VerificationError("Saved GLOBAL fit is not complete")
    hp = {key: float(h[key]) for key in ("intercept", "slope_low", "slope_high")}
    gp = (float(g["intercept"]), float(g["slope"]))
    if not all(math.isfinite(x) for x in (*hp.values(), *gp)):
        raise VerificationError("Saved coefficients are invalid")
    return hp, gp


def _rebuild_predictions(p: np.ndarray, missing: np.ndarray,
                         hp: Mapping[str, float], gp: tuple[float, float]) -> dict[str, np.ndarray]:
    raw = np.asarray(p, dtype="<f8")
    miss = np.asarray(missing, dtype=bool)
    a, sl, sh = hp["intercept"], hp["slope_low"], hp["slope_high"]
    present = ~miss
    hinge = raw.copy()
    hinge[present] = np.clip(a + sl * np.minimum(raw[present], 50.0)
                             + sh * np.maximum(raw[present] - 50.0, 0.0), 0.0, 100.0)
    ga, gb = gp
    global_present = raw.copy()
    global_present[present] = np.clip(ga + gb * raw[present], 0.0, 100.0)
    if not np.array_equal(hinge[miss].view("<u8"), raw[miss].view("<u8")):
        raise VerificationError("HINGE missing route failed bit replay")
    if not np.array_equal(global_present[miss].view("<u8"), raw[miss].view("<u8")):
        raise VerificationError("GLOBAL missing route failed bit replay")
    return {"raw_direct": raw.copy(), "causal_global_present_only": global_present,
        "hinge_present_only": hinge}


def _hash_float(values: np.ndarray) -> str:
    a = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    return hashlib.sha256(b"float64-le\0" + str(a.shape).encode("ascii") + b"\0" + a.tobytes()).hexdigest()


def _hash_text(values: np.ndarray) -> str:
    h = hashlib.sha256(b"utf8-length-prefixed\0")
    for value in np.asarray(values).astype(str):
        raw = value.encode("utf-8")
        h.update(len(raw).to_bytes(8, "little"))
        h.update(raw)
    return h.hexdigest()


def _hash_rows(a: Mapping[str, np.ndarray]) -> str:
    h = hashlib.sha256(b"reg-hinge-residual-r1-row-identity\0")
    rows = np.asarray(a["physical_rows"], dtype=np.int64)
    ids = np.asarray(a["record_ids"]).astype(str)
    origins = np.asarray(a["prediction_origins"], dtype=np.int64)
    indices = np.asarray(a["original_bundle_indices"], dtype=np.int64)
    cycles = np.asarray(a["cycles"], dtype=np.int64)
    p = np.asarray(a["direct_predictions"], dtype="<f8")
    miss = np.asarray(a["missing_mask"], dtype=bool)
    y = np.asarray(a["targets"], dtype="<f8")
    n = len(rows)
    if any(len(v) != n for v in (ids, origins, indices, cycles, p, miss, y)):
        raise VerificationError("Independent row reconstruction found misaligned vectors")
    for i in range(n):
        row = [int(rows[i]), ids[i], int(origins[i]), int(indices[i]), int(cycles[i]),
               float(p[i]).hex(), bool(miss[i]), float(y[i]).hex()]
        h.update(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _mae(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(prediction, dtype=np.float64)
                                - np.asarray(target, dtype=np.float64))))


def _cell(mask: np.ndarray, y: np.ndarray, miss: np.ndarray,
          pred: Mapping[str, np.ndarray], minimum: int) -> dict[str, Any]:
    rows = np.asarray(mask, dtype=bool)
    present = rows & ~miss
    absent = rows & miss
    n, pn, mn = int(rows.sum()), int(present.sum()), int(absent.sum())
    out: dict[str, Any] = {"n_all": n, "n_present": pn, "n_missing": mn,
        "support_all": "SUPPORTED" if n >= minimum else "UNKNOWN_UNSUPPORTED",
        "support_present": "SUPPORTED" if pn >= minimum else "UNKNOWN_UNSUPPORTED",
        "target_le_5_n": int(np.sum(rows & (y <= r.TARGET_TAIL_LOW))),
        "target_ge_95_n": int(np.sum(rows & (y >= r.TARGET_TAIL_HIGH)))}
    out["target_tail_support"] = {}
    for name, tail_mask in (("target_le_5", y <= r.TARGET_TAIL_LOW),
                            ("target_ge_95", y >= r.TARGET_TAIL_HIGH)):
        all_tail = rows & tail_mask
        present_tail = present & tail_mask
        missing_tail = absent & tail_mask
        all_n, present_n = int(all_tail.sum()), int(present_tail.sum())
        out["target_tail_support"][name] = {"n_all": all_n,
            "n_present": present_n, "n_missing": int(missing_tail.sum()),
            "support_all": "SUPPORTED" if all_n >= minimum else "UNKNOWN_UNSUPPORTED",
            "support_present": "SUPPORTED" if present_n >= minimum else "UNKNOWN_UNSUPPORTED"}
    out["median_residual_y_minus_prediction"] = {"all_rows": {}, "present_only": {}}
    out["policy_mae"] = {"all_rows": {}, "present_only": {}}
    for name in r.PRESENT_POLICIES:
        residual = y - pred[name]
        out["median_residual_y_minus_prediction"]["all_rows"][name] = float(np.median(residual[rows])) if n >= minimum else None
        out["median_residual_y_minus_prediction"]["present_only"][name] = float(np.median(residual[present])) if pn >= minimum else None
        for scope, active in (("all_rows", rows), ("present_only", present)):
            count = int(active.sum())
            if count < minimum:
                out["policy_mae"][scope][name] = {"status": "UNKNOWN_UNSUPPORTED", "n": count,
                                                  "mae": None, "mean_error": None, "p90_absolute_error": None}
            else:
                error = pred[name][active] - y[active]
                out["policy_mae"][scope][name] = {"status": "SUPPORTED", "n": count,
                    "mae": float(np.mean(np.abs(error))), "mean_error": float(np.mean(error)),
                    "p90_absolute_error": float(np.quantile(np.abs(error), 0.90))}
    out["gain_hinge_over_global"] = {"all_rows": None, "present_only": None}
    if n >= minimum:
        out["gain_hinge_over_global"]["all_rows"] = (
            out["policy_mae"]["all_rows"]["causal_global_present_only"]["mae"]
            - out["policy_mae"]["all_rows"]["hinge_present_only"]["mae"])
    if pn >= minimum:
        out["gain_hinge_over_global"]["present_only"] = (
            out["policy_mae"]["present_only"]["causal_global_present_only"]["mae"]
            - out["policy_mae"]["present_only"]["hinge_present_only"]["mae"])
    return out


def _population(a: Mapping[str, np.ndarray], fit_id: str, role: str,
                pred: Mapping[str, np.ndarray]) -> dict[str, Any]:
    y = np.asarray(a["targets"], dtype="<f8")
    p = np.asarray(a["direct_predictions"], dtype="<f8")
    miss = np.asarray(a["missing_mask"], dtype=bool)
    cycles = np.asarray(a["cycles"], dtype=np.int64)
    origins = np.asarray(a["prediction_origins"], dtype=np.int64)
    n = len(y)
    expected = r.FIT_EXPECTED[fit_id] if role == "fit" else r.SCORE_EXPECTED[fit_id]
    if (n != expected or len(np.unique(a["record_ids"])) != n
            or len(np.unique(a["physical_rows"])) != n or not np.isfinite(y).all()):
        raise VerificationError(f"Independent population boundary failed for {fit_id}/{role}")
    if (not np.all(origins == EXPECTED_ORIGIN[(fit_id, role)])
            or tuple(sorted(np.unique(cycles).tolist())) != EXPECTED_CYCLES[(fit_id, role)]):
        raise VerificationError(f"Independent origin/cycle boundary failed for {fit_id}/{role}")
    bin_index = np.clip(np.searchsorted(np.asarray(r.BIN_EDGES), p, side="right") - 1,
                        0, len(r.BIN_LABELS) - 1)
    bins = [{"bin_index": i, "label": label,
             **_cell(bin_index == i, y, miss, pred, r.MIN_BIN_PRESENT)}
            for i, label in enumerate(r.BIN_LABELS)]
    cycle_rows = [{"cycle": int(c), **_cell(cycles == c, y, miss, pred, r.MIN_CYCLE_PRESENT)}
                  for c in sorted(np.unique(cycles).tolist())]
    allmask = np.ones(n, dtype=bool)
    overall = _cell(allmask, y, miss, pred, 1)
    gain_all = _mae(pred["causal_global_present_only"], y) - _mae(pred["hinge_present_only"], y)
    gain_present = (_mae(pred["causal_global_present_only"][~miss], y[~miss])
                    - _mae(pred["hinge_present_only"][~miss], y[~miss])) if np.any(~miss) else None
    return {"fit_id": fit_id, "population": role, "n_all": n,
        "n_present": int((~miss).sum()), "n_missing": int(miss.sum()),
        "fit_coefficients_applied_to_all_rows": role == "fit",
        "mapping_application_present_only": True,
        "row_hashes": {"row_identity_sha256": _hash_rows(a),
            "record_ids_sha256": _hash_text(a["record_ids"]),
            "direct_sha256": _hash_float(p), "targets_sha256": _hash_float(y),
            "cycles_sha256": hashlib.sha256(np.ascontiguousarray(np.asarray(a["cycles"], dtype="<i2")).tobytes()).hexdigest(),
            "missing_mask_sha256": hashlib.sha256(np.ascontiguousarray(miss.astype(np.uint8)).tobytes()).hexdigest()},
        "target_tail_support": {"target_le_5": {
                "n_all": int(np.sum(y <= r.TARGET_TAIL_LOW)),
                "n_present": int(np.sum((y <= r.TARGET_TAIL_LOW) & ~miss)),
                "n_missing": int(np.sum((y <= r.TARGET_TAIL_LOW) & miss))},
            "target_ge_95": {
                "n_all": int(np.sum(y >= r.TARGET_TAIL_HIGH)),
                "n_present": int(np.sum((y >= r.TARGET_TAIL_HIGH) & ~miss)),
                "n_missing": int(np.sum((y >= r.TARGET_TAIL_HIGH) & miss))},
            "diagnostic_only": True},
        "by_raw_direct_bin": bins, "by_cycle": cycle_rows,
        "overall_policy_mae": overall["policy_mae"],
        "overall_gain_hinge_over_global": {"all_rows": gain_all, "present_only": gain_present}}


def _score_metrics(a: Mapping[str, np.ndarray]) -> dict[str, Any]:
    y = np.asarray(a["targets"], dtype=np.float64)
    cycles = np.asarray(a["cycles"], dtype=np.int64)
    missing = np.asarray(a["missing_mask"], dtype=bool)
    result = {}
    for name in r.SCORED_POLICIES:
        prediction = np.asarray(a[name], dtype=np.float64)
        error = prediction - y
        out = {"overall": {"n": int(len(y)), "mae": float(np.mean(np.abs(error))),
                            "mean_error": float(np.mean(error)),
                            "p90_absolute_error": float(np.quantile(np.abs(error), 0.90))},
               "groups": {}}
        masks = {"present": ~missing, "missing": missing,
                 "target_le_5": y <= r.TARGET_TAIL_LOW,
                 "target_ge_95": y >= r.TARGET_TAIL_HIGH}
        for key, mask in masks.items():
            count = int(mask.sum())
            if count < 200:
                out["groups"][key] = {"n": count, "status": "UNKNOWN_UNSUPPORTED", "mae": None}
            else:
                e = error[mask]
                out["groups"][key] = {"status": "SUPPORTED", "n": count,
                    "mae": float(np.mean(np.abs(e))), "mean_error": float(np.mean(e)),
                    "p90_absolute_error": float(np.quantile(np.abs(e), 0.90))}
        out["groups"]["by_cycle"] = {}
        for cycle in np.unique(cycles):
            e = error[cycles == cycle]
            out["groups"]["by_cycle"][str(int(cycle))] = {"n": int(len(e)),
                "mae": float(np.mean(np.abs(e))), "mean_error": float(np.mean(e)),
                "p90_absolute_error": float(np.quantile(np.abs(e), 0.90))}
        result[name] = out
    return result


def _signed_tail_metrics(diag_record: Mapping[str, Any]) -> dict[str, Any]:
    result = {}
    for policy in r.SCORED_POLICIES:
        metrics = diag_record["score_metrics"][policy]
        result[policy] = {"overall_p90_absolute_error": metrics["overall"]["p90_absolute_error"],
            "target_le_5": metrics["groups"]["target_le_5"],
            "target_ge_95": metrics["groups"]["target_ge_95"],
            "source": "existing_signed_R4_diagnostics"}
    return result


def _compare(expected: Any, actual: Any, path: str = "summary") -> tuple[int, float]:
    checks, max_diff = 0, 0.0
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(expected) != set(actual):
            raise VerificationError(f"Independent result structure differs at {path}")
        for key in expected:
            c, d = _compare(expected[key], actual[key], f"{path}.{key}")
            checks += c
            max_diff = max(max_diff, d)
        return checks, max_diff
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise VerificationError(f"Independent list structure differs at {path}")
        for i, (left, right) in enumerate(zip(expected, actual)):
            c, d = _compare(left, right, f"{path}[{i}]")
            checks += c
            max_diff = max(max_diff, d)
        return checks, max_diff
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(actual, (int, float)) or not math.isclose(float(expected), float(actual), rel_tol=1e-12, abs_tol=1e-12):
            raise VerificationError(f"Independent numeric replay differs at {path}: {expected} != {actual}")
        return 1, abs(float(expected) - float(actual))
    if expected != actual:
        raise VerificationError(f"Independent replay differs at {path}: {expected} != {actual}")
    return 1, 0.0


def verify() -> dict[str, Any]:
    freeze, freeze_sha = _verify_freeze()
    if not r.SUMMARY.is_file() or not r.PLOT_BINS.is_file() or not r.PLOT_CYCLES.is_file():
        raise VerificationError("Machine summary or required plots are absent")
    if r.VERIFICATION.exists():
        raise VerificationError("Independent verification is already present; refusing overwrite")
    summary = json.loads(r.SUMMARY.read_text(encoding="utf-8"))
    summary_signature = summary.pop("signature", None)
    if summary_signature != _canonical_sha(summary):
        raise VerificationError("Independent machine-summary signature replay failed")
    summary["signature"] = summary_signature
    if summary.get("input_freeze_sha256") != freeze_sha or summary.get("source_commit") != freeze["source_commit"]:
        raise VerificationError("Machine summary is bound to another freeze or commit")
    if set(summary.get("source_and_input_hashes", {})) != set(freeze["source_and_input_files"]):
        raise VerificationError("Machine summary file-hash inventory differs from the freeze")
    if summary.get("source_and_input_hashes") != freeze["source_and_input_files"]:
        raise VerificationError("Machine summary file hashes differ from the freeze")
    forbidden_fragments = ("Train.csv", "Test.csv", "SampleSubmission", "inference_B.npz")
    if any(any(fragment.casefold() in item["path"].casefold() for fragment in forbidden_fragments)
           for item in freeze["source_and_input_files"].values()):
        raise VerificationError("Frozen input list crosses the authorized cached-bundle boundary")

    diagnostics = json.loads(r.INPUT_FILES["r4_diagnostics"].read_text(encoding="utf-8"))
    resources: list[dict[str, Any]] = []
    max_diff = 0.0
    numeric_checks = 0
    input_hash_checks = len(freeze["source_and_input_files"])
    population_replays = []
    limits = freeze["resource_policy"]
    with r4.SharedHeavyJobLock(r4.SHARED_JOB_LOCK):
        _verify_freeze()
        resources.append(_resource_gate("independent_initial_locked_before_npz", initial=True, limits=limits))
        with threadpool_limits(limits=2):
            for fit_id in ("HINGE_D1", "HINGE_D2"):
                suffix = fit_id[-2:].lower()
                fit_state = json.loads(r.INPUT_FILES[f"{suffix}_fit_state"].read_text(encoding="utf-8"))
                global_state = json.loads(r.INPUT_FILES[f"{suffix}_global_state"].read_text(encoding="utf-8"))
                hp, gp = _params(fit_state, global_state)
                for role, name_tuple in (("fit", r.FIT_FIELDS), ("score", r.SCORE_FIELDS)):
                    path = r.INPUT_FILES[f"{suffix}_{'fit_input' if role == 'fit' else 'score_rows'}"]
                    arrays = _load(path, name_tuple,
                        permitted_unloaded=("causal_global_all_rows",) if role == "score" else ())
                    resources.append(_resource_gate(f"{fit_id}_{role}_loaded", initial=False, limits=limits))
                    predictions = _rebuild_predictions(arrays["direct_predictions"], arrays["missing_mask"], hp, gp)
                    if role == "score":
                        for key in ("raw_direct", "causal_global_present_only", "hinge_present_only"):
                            if not np.array_equal(np.asarray(arrays[key], dtype="<f8").view("<u8"),
                                                  np.asarray(predictions[key], dtype="<f8").view("<u8")):
                                raise VerificationError(f"Saved score predictions differ from independent replay: {fit_id}/{key}")
                    replay = _population(arrays, fit_id, role, predictions)
                    score_numeric_fields = 0
                    if role == "score":
                        diag_record = next(x for x in diagnostics["stages"] if x["fit_id"] == fit_id)
                        scored = _score_metrics(arrays)
                        score_expected = {name: diag_record["score_metrics"][name]
                                          for name in r.SCORED_POLICIES}
                        score_numeric_fields, score_diff = _compare(score_expected, scored,
                            f"{fit_id}.signed_R4_score_metrics")
                        numeric_checks += score_numeric_fields
                        max_diff = max(max_diff, score_diff)
                        replay["existing_r4_score_metrics_replayed"] = True
                        replay["max_difference_from_signed_r4_score_metrics"] = score_diff
                        replay["existing_r4_score_tail_and_p90"] = _signed_tail_metrics(diag_record)
                    expected = next(x for x in summary["populations"]
                                    if x["fit_id"] == fit_id and x["population"] == role)
                    count, diff = _compare(expected, replay, f"{fit_id}.{role}")
                    numeric_checks += count
                    max_diff = max(max_diff, diff)
                    population_replays.append({"fit_id": fit_id, "population": role,
                        "rows": replay["n_all"], "row_identity_sha256": replay["row_hashes"]["row_identity_sha256"],
                        "numeric_fields_replayed": count + score_numeric_fields})
                    del arrays, predictions
                    import gc
                    gc.collect()
                    resources.append(_resource_gate(f"{fit_id}_{role}_released", initial=False, limits=limits))

        plot_hashes = {"raw_bin_residuals_sha256": _sha(r.PLOT_BINS),
                       "cycle_gains_sha256": _sha(r.PLOT_CYCLES)}
        if summary.get("two_plot_hashes") != plot_hashes:
            raise VerificationError("Plot hashes differ from signed machine summary")
        resources.append(_resource_gate("independent_after_replay", initial=False, limits=limits))

    body = {"schema_version": 1,
        "experiment_id": r.NAMESPACE.upper() + "_R1",
        "status": "PASS_INDEPENDENT_NUMERIC_ROW_HASH_RECONSTRUCTION",
        "input_freeze_sha256": freeze_sha,
        "machine_summary_sha256": _sha(r.SUMMARY),
        "population_replays": population_replays,
        "input_hashes_verified": input_hash_checks,
        "numeric_fields_replayed": numeric_checks,
        "maximum_numeric_difference": max_diff,
        "plot_hashes": plot_hashes,
        "resource_snapshots": resources,
        "new_fits": 0,
        "new_calibrators": 0,
        "raw_train_label_access": False,
        "test_or_sample_access": False,
        "e79_access": False,
        "kaggle_access": False}
    body["signature"] = _canonical_sha(body)
    verification_sha = _write_once(r.VERIFICATION, body)
    return {"status": body["status"], "path": str(r.VERIFICATION.resolve()),
            "sha256": verification_sha, "input_hashes_verified": input_hash_checks,
            "numeric_fields_replayed": numeric_checks,
            "maximum_numeric_difference": max_diff}


if __name__ == "__main__":
    try:
        print(json.dumps(verify(), sort_keys=True))
    except VerificationError as exc:
        print(json.dumps({"status": "FAIL_INDEPENDENT_RECONSTRUCTION", "reason": str(exc)}, sort_keys=True), file=sys.stderr)
        raise SystemExit(1)
