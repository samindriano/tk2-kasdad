"""Independent lineage, fit-ledger, and score verification for N05."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scripts import optimization_v3_research_next_n00 as n00
from src.models.regression_research_text import HYBRID_FEATURES
from src.models.research_next_budget import read_events
from src.models.research_next_contract import canonical_sha256, file_sha256, read_bundle
from src.models.research_next_n05 import array_sha256
from src.utils.paths import PROJECT_ROOT, task_files


ROOT = PROJECT_ROOT
BATCH = "optimization_v3_research_next_v1"
QUEUE_PATH = ROOT / "experiments/optimization_v3/research_next_v1/queue.json"
FREEZE_PATH = ROOT / "experiments/optimization_v3/research_next_v1/frozen/N05/amendment_01/freeze.json"
OLD_LEDGER_PATH = ROOT / "experiments/runs/optimization_v3_batch1/batch_ledger.jsonl"
LEDGER_PATH = ROOT / "experiments/runs/optimization_v3_research_next_v1/budget.jsonl"
RESULT_PATH = ROOT / "reports/optimization_v3/research_next_v1/results/N05/amendment_01/results.json"
VERIFY_PATH = ROOT / "reports/optimization_v3/research_next_v1/results/N05/amendment_01/verification.json"
BUNDLE_ROOT = ROOT / "experiments/runs/optimization_v3_research_next_v1/N05/amendment_01"
ORIGINS = (67, 75)
ALL_ORIGINS = (55, 59, 63, 67, 71, 75)
ALPHAS = (0.1, 0.9)


def _sha256_array(values: Any, dtype: str = "<f8") -> str:
    array = np.asarray(values, dtype=dtype)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _same(left: Any, right: Any, *, atol: float = 1e-12) -> bool:
    try:
        return bool(np.isclose(float(left), float(right), rtol=0.0, atol=atol, equal_nan=False))
    except (TypeError, ValueError):
        return False


def _paths_from_identity(identity: dict[str, Any]) -> dict[str, Path]:
    result = {}
    for relative in identity["input_and_source_sha256"]:
        result[relative] = ROOT / Path(relative)
    return result


def _load_data() -> tuple[Path, pd.DataFrame, np.ndarray]:
    path = Path(task_files("regression")["train"])
    frame = pd.read_csv(path, usecols=["record_id", "threat_rank", *HYBRID_FEATURES])
    target = pd.to_numeric(frame.threat_rank, errors="raise").to_numpy(dtype=np.float64)
    cycle = pd.to_numeric(frame.cycle_code, errors="raise").to_numpy(dtype=np.int64)
    if (len(frame) != 50_608 or not frame.record_id.is_unique
            or not np.array_equal(np.unique(cycle), np.arange(47, 80))
            or not np.isfinite(target).all() or np.any((target < 0) | (target > 100))):
        raise RuntimeError("Independent N05 verifier: official Train schema/range mismatch")
    return path, frame, target


def _r22_sources(frame: pd.DataFrame) -> dict[int, dict[str, np.ndarray]]:
    result = {}
    cycle_all = frame.cycle_code.to_numpy(dtype=np.int64)
    for cutoff in ALL_ORIGINS:
        bundle = n00._load_artifact("R22", cutoff, frame)
        rebuilt = n00.reconstruct_regression(bundle)
        rows = rebuilt["rows"].astype(np.int64)
        cycles = rebuilt["cycles"].astype(np.int64)
        horizon = 12 if cutoff in (55, 67) else 4
        expected = np.flatnonzero((cycle_all > cutoff) & (cycle_all <= min(cutoff + horizon, 79)))
        ids = bundle["arrays"]["score_record_ids"].astype(str)
        if (not np.array_equal(rows, expected)
                or not np.array_equal(ids, frame.record_id.to_numpy(dtype=str)[rows])):
            raise RuntimeError(f"Independent N05 verifier: R22/{cutoff} cache lineage differs")
        prediction = rebuilt["prediction"].astype(np.float64)
        if not np.isfinite(prediction).all() or np.any((prediction < 0) | (prediction > 100)):
            raise RuntimeError(f"Independent N05 verifier: R22/{cutoff} prediction range differs")
        result[cutoff] = {"rows": rows, "cycles": cycles, "prediction": prediction,
                          "ids": ids}
    return result


def _past_residual(frame: pd.DataFrame, target: np.ndarray,
                   sources: dict[int, dict[str, np.ndarray]], cutoff: int) -> dict[str, Any]:
    queue = json.loads(QUEUE_PATH.read_text(encoding="utf-8"))
    blocks = [tuple(map(int, block)) for block in queue["views"]["full_six"]]
    rows_all, predictions_all = [], []
    for origin, first_cycle, last_cycle in blocks:
        if origin > cutoff or last_cycle > cutoff:
            continue
        source = sources[origin]
        mask = (source["cycles"] >= first_cycle) & (source["cycles"] <= last_cycle)
        rows_all.append(source["rows"][mask])
        predictions_all.append(source["prediction"][mask])
    rows = np.concatenate(rows_all).astype(np.int64)
    prediction = np.concatenate(predictions_all).astype(np.float64)
    expected = np.flatnonzero((frame.cycle_code.to_numpy(dtype=np.int64) >= 56)
                              & (frame.cycle_code.to_numpy(dtype=np.int64) <= cutoff))
    if len(np.unique(rows)) != len(rows) or not np.array_equal(np.sort(rows), expected):
        raise RuntimeError(f"Independent N05 verifier: calibration primary rows incomplete at {cutoff}")
    order = np.argsort(rows, kind="mergesort")
    rows, prediction = rows[order], prediction[order]
    residual = target[rows] - prediction
    quantiles = np.quantile(residual, [0.1, 0.9], method="linear")
    return {"rows": rows, "prediction": prediction, "target": target[rows],
            "residual": residual, "q10": float(quantiles[0]), "q90": float(quantiles[1])}


def _metric(target: np.ndarray, lower: np.ndarray, upper: np.ndarray,
            point: np.ndarray) -> dict[str, float | int | None]:
    target = np.asarray(target, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    point = np.asarray(point, dtype=np.float64)
    if not len(target):
        return {"n": 0, "coverage_80": None, "mean_width": None}
    width = upper - lower
    rho = spearmanr(width, np.abs(target - point)).statistic
    return {"n": int(len(target)), "coverage_80": float(np.mean((target >= lower) & (target <= upper))),
            "mean_width": float(width.mean()),
            "width_absolute_error_spearman": float(rho) if np.isfinite(rho) else None}


def _raw_interval(lower_raw: np.ndarray, upper_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    crossed = lower_raw > upper_raw
    lower_clip = np.clip(lower_raw, 0.0, 100.0)
    upper_clip = np.clip(upper_raw, 0.0, 100.0)
    return np.minimum(lower_clip, upper_clip), np.maximum(lower_clip, upper_clip), float(crossed.mean())


def _verify() -> dict[str, Any]:
    frozen = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    frozen_identity = frozen.get("identity")
    signature = frozen.get("freeze_signature")
    if signature != canonical_sha256(frozen_identity):
        raise RuntimeError("Independent N05 verifier: freeze signature mismatch")
    commit = frozen.get("execution_context", {}).get("source_commit")
    if (not commit or subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"],
                                     cwd=ROOT, check=False).returncode != 0):
        raise RuntimeError("Independent N05 verifier: frozen source commit is not in current history")
    verified_hashes = {}
    for relative, expected_hash in frozen_identity["input_and_source_sha256"].items():
        path = ROOT / Path(relative)
        actual = file_sha256(path)
        if actual != expected_hash:
            raise RuntimeError(f"Independent N05 verifier: frozen hash changed: {relative}")
        verified_hashes[relative] = actual
    path, frame, target = _load_data()
    sources = _r22_sources(frame)
    queue = json.loads(QUEUE_PATH.read_text(encoding="utf-8"))
    ledger = read_events(LEDGER_PATH)
    reservations = [event for event in ledger if event.get("event") == "CARD_FITS_RESERVED"
                    and event.get("reservation_key") == "N05:PILOT"]
    if len(reservations) != 1 or int(reservations[0]["fits"]) != 4:
        raise RuntimeError("Independent N05 verifier: exact four-fit reservation absent")

    bundle_data: dict[int, dict[float, dict[str, Any]]] = {}
    fit_receipts = []
    controls = {}
    for cutoff in ORIGINS:
        controls[cutoff] = _past_residual(frame, target, sources, cutoff)
        bundle_data[cutoff] = {}
        for alpha in ALPHAS:
            fit_id = f"N05:PILOT:{cutoff}:q{int(alpha * 100):02d}"
            component = f"quantile_alpha_{alpha:.1f}"
            identity = {"batch_id": BATCH, "card_id": "N05", "freeze_signature": signature,
                        "stage": "PILOT", "cutoff": cutoff, "component": component}
            bundle_path = BUNDLE_ROOT / f"fit{cutoff}" / f"q{int(alpha * 100):02d}"
            bundle = read_bundle(bundle_path, identity, task="regression")
            arrays = bundle.arrays
            diagnostics = bundle.receipt["metadata"].get("diagnostics", {})
            source = sources[cutoff]
            prefix_rows = np.flatnonzero(frame.cycle_code.to_numpy(dtype=np.int64) <= cutoff)
            prefix_target = target[prefix_rows]
            expected_control_lower, expected_control_upper, _ = _raw_interval(
                source["prediction"] + controls[cutoff]["q10"],
                source["prediction"] + controls[cutoff]["q90"],
            )
            checks = (
                np.array_equal(arrays["score_rows"], source["rows"]),
                np.array_equal(arrays["score_record_ids"].astype(str), source["ids"]),
                np.array_equal(arrays["cycle_code"], source["cycles"]),
                np.array_equal(arrays["baseline_prediction"], source["prediction"]),
                np.array_equal(arrays["control_lower"], expected_control_lower),
                np.array_equal(arrays["control_upper"], expected_control_upper),
                np.array_equal(arrays["prediction"], np.clip(arrays["raw_quantile_prediction"], 0.0, 100.0)),
                diagnostics.get("alpha") == alpha,
                diagnostics.get("cutoff") == cutoff,
                diagnostics.get("fit_rows") == len(prefix_rows),
                diagnostics.get("fit_row_indices_sha256") == _sha256_array(prefix_rows, "<i8"),
                diagnostics.get("fit_target_sha256") == _sha256_array(prefix_target),
                diagnostics.get("score_target_passed_to_fit") is False,
                diagnostics.get("test_data_used") is False,
                diagnostics.get("quantile_parameters", {}).get("objective") == "quantile",
                diagnostics.get("quantile_parameters", {}).get("alpha") == alpha,
            )
            if not all(checks):
                raise RuntimeError(f"Independent N05 verifier: bundle audit failed for {fit_id}")
            events = [event for event in ledger if event.get("fit_id") == fit_id]
            if (sum(event.get("event") == "FIT_STARTED" for event in events) != 1
                    or sum(event.get("event") == "FIT_SAVED" for event in events) != 1
                    or any(event.get("event") == "FIT_FAILED" for event in events)):
                raise RuntimeError(f"Independent N05 verifier: ledger terminal differs for {fit_id}")
            bundle_data[cutoff][alpha] = arrays
            fit_receipts.append({"fit_id": fit_id, "receipt_sha256": bundle.receipt["receipt_sha256"],
                                 "npz_sha256": bundle.receipt["npz_sha256"],
                                 "fit_rows": int(diagnostics["fit_rows"]),
                                 "score_rows": int(len(arrays["score_rows"]))})
    n05_starts = [event for event in ledger if event.get("event") == "FIT_STARTED"
                  and event.get("card_id") == "N05" and event.get("stage") == "PILOT"]
    if len(n05_starts) != 4:
        raise RuntimeError("Independent N05 verifier: official N05 fit count is not exactly four")

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    if result.get("freeze_signature") != signature or result.get("status") != "COMPLETE_INFORMATION":
        raise RuntimeError("Independent N05 verifier: result status or freeze identity differs")
    if result.get("point_model_selection", {}).get("eligible") is not False:
        raise RuntimeError("Independent N05 verifier: information lane is not explicitly ineligible")
    parts_by_origin = {}
    verified_views = {}
    for cutoff in ORIGINS:
        source = sources[cutoff]
        q10 = bundle_data[cutoff][0.1]["raw_quantile_prediction"].astype(np.float64)
        q90 = bundle_data[cutoff][0.9]["raw_quantile_prediction"].astype(np.float64)
        lo, hi, cross_rate = _raw_interval(q10, q90)
        control_lo, control_hi, _ = _raw_interval(
            source["prediction"] + controls[cutoff]["q10"],
            source["prediction"] + controls[cutoff]["q90"],
        )
        parts = {"target": target[source["rows"]], "point": source["prediction"],
                 "lower": lo, "upper": hi, "control_lower": control_lo,
                 "control_upper": control_hi, "cycles": source["cycles"]}
        parts_by_origin[cutoff] = parts
        block = next(tuple(map(int, block)) for block in queue["views"]["LATE"]["primary"]
                     if int(block[0]) == cutoff)
        primary = (source["cycles"] >= block[1]) & (source["cycles"] <= block[2])
        candidate_metrics = _metric(parts["target"][primary], lo[primary], hi[primary], source["prediction"][primary])
        control_metrics = _metric(parts["target"][primary], control_lo[primary],
                                  control_hi[primary], source["prediction"][primary])
        stored = result["origins"][str(cutoff)]["primary"]
        for label, expected_metric, stored_metric in (
            ("candidate", candidate_metrics, stored["candidate"]),
            ("control", control_metrics, stored["past_residual_control"]),
        ):
            if (expected_metric["n"] != stored_metric["n"]
                    or not _same(expected_metric["coverage_80"], stored_metric["coverage_80"])
                    or not _same(expected_metric["mean_width"], stored_metric["mean_width"])):
                raise RuntimeError(f"Independent N05 verifier: {label} primary metric differs at {cutoff}")
        if not _same(cross_rate, stored["candidate"]["raw_crossing_rate"]):
            raise RuntimeError(f"Independent N05 verifier: crossing rate differs at {cutoff}")
        for cycle_text, cycle_row in stored["per_cycle"].items():
            cycle_mask = primary & (source["cycles"] == int(cycle_text))
            cm = _metric(parts["target"][cycle_mask], lo[cycle_mask], hi[cycle_mask],
                         source["prediction"][cycle_mask])
            bm = _metric(parts["target"][cycle_mask], control_lo[cycle_mask],
                         control_hi[cycle_mask], source["prediction"][cycle_mask])
            for name, expected_metric, stored_metric in (
                ("candidate", cm, cycle_row["candidate"]),
                ("past_residual_control", bm, cycle_row["past_residual_control"]),
            ):
                if (expected_metric["n"] != stored_metric["n"]
                        or not _same(expected_metric["coverage_80"], stored_metric["coverage_80"])
                        or not _same(expected_metric["mean_width"], stored_metric["mean_width"])):
                    raise RuntimeError(f"Independent N05 verifier: per-cycle metric differs at {cutoff}/{cycle_text}")
        verified_views[f"primary_{cutoff}"] = {"n": candidate_metrics["n"],
                                               "coverage_80": candidate_metrics["coverage_80"],
                                               "mean_width": candidate_metrics["mean_width"],
                                               "control_coverage_80": control_metrics["coverage_80"],
                                               "control_mean_width": control_metrics["mean_width"],
                                               "raw_crossing_rate": cross_rate}

    late_parts = []
    for cutoff in ORIGINS:
        block = next(tuple(map(int, item)) for item in queue["views"]["LATE"]["primary"]
                     if int(item[0]) == cutoff)
        source = sources[cutoff]
        mask = (source["cycles"] >= block[1]) & (source["cycles"] <= block[2])
        late_parts.append({key: value[mask] for key, value in parts_by_origin[cutoff].items()})
    pooled = {key: np.concatenate([part[key] for part in late_parts]) for key in late_parts[0]}
    pooled_rows = np.concatenate([
        sources[cutoff]["rows"][(sources[cutoff]["cycles"] >= next(
            tuple(map(int, item))[1] for item in queue["views"]["LATE"]["primary"]
            if int(item[0]) == cutoff))
                                & (sources[cutoff]["cycles"] <= next(
            tuple(map(int, item))[2] for item in queue["views"]["LATE"]["primary"]
            if int(item[0]) == cutoff))]
        for cutoff in ORIGINS
    ])
    if len(pooled_rows) != len(np.unique(pooled_rows)):
        raise RuntimeError("Independent N05 verifier: LATE primary rows overlap")
    pooled_candidate = _metric(pooled["target"], pooled["lower"], pooled["upper"], pooled["point"])
    pooled_control = _metric(pooled["target"], pooled["control_lower"], pooled["control_upper"], pooled["point"])
    saved_pooled = result["late_primary_pooled"]
    if (len(pooled_rows) != result.get("late_primary_unique_rows")
            or not _same(pooled_candidate["coverage_80"], saved_pooled["candidate"]["coverage_80"])
            or not _same(pooled_candidate["mean_width"], saved_pooled["candidate"]["mean_width"])
            or not _same(pooled_control["coverage_80"], saved_pooled["past_residual_control"]["coverage_80"])
            or not _same(pooled_control["mean_width"], saved_pooled["past_residual_control"]["mean_width"])):
        raise RuntimeError("Independent N05 verifier: pooled primary metric differs")
    h12 = parts_by_origin[67]
    h12_candidate = _metric(h12["target"], h12["lower"], h12["upper"], h12["point"])
    h12_control = _metric(h12["target"], h12["control_lower"], h12["control_upper"], h12["point"])
    saved_h12 = result["late_h12_from_67"]
    if (not _same(h12_candidate["coverage_80"], saved_h12["candidate"]["coverage_80"])
            or not _same(h12_candidate["mean_width"], saved_h12["candidate"]["mean_width"])
            or not _same(h12_control["coverage_80"], saved_h12["past_residual_control"]["coverage_80"])):
        raise RuntimeError("Independent N05 verifier: H12 metric differs")
    verification = {
        "status": "PASS", "verification_method": "independent cache reconstruction, bundle/ledger audit and metric recomputation",
        "freeze_signature": signature, "official_train_sha256": file_sha256(path),
        "verified_source_hash_count": len(verified_hashes), "verified_fit_count": len(fit_receipts),
        "verified_fits": fit_receipts, "verified_primary_views": verified_views,
        "verified_late_primary_pooled": {
            "n": int(len(pooled_rows)), "coverage_80": pooled_candidate["coverage_80"],
            "mean_width": pooled_candidate["mean_width"],
            "control_coverage_80": pooled_control["coverage_80"],
            "control_mean_width": pooled_control["mean_width"],
        },
        "verified_late_h12_from_67": {
            "n": int(len(h12["target"])), "coverage_80": h12_candidate["coverage_80"],
            "mean_width": h12_candidate["mean_width"],
            "control_coverage_80": h12_control["coverage_80"],
        },
        "test_files_opened": False, "test_labels_read": False, "kaggle_calls": 0,
        "information_only_no_point_promotion": True,
    }
    if VERIFY_PATH.exists():
        existing = json.loads(VERIFY_PATH.read_text(encoding="utf-8"))
        if canonical_sha256(existing) != canonical_sha256(verification):
            raise RuntimeError("Independent N05 verifier refuses to overwrite changed verification")
    else:
        VERIFY_PATH.parent.mkdir(parents=True, exist_ok=True)
        VERIFY_PATH.write_text(json.dumps(verification, sort_keys=True, indent=2, allow_nan=False) + "\n",
                               encoding="utf-8")
    return verification


def main() -> int:
    result = _verify()
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
