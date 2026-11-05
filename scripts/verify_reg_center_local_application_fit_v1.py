"""Independent exact-weight numerical QA for the authorized R3 scalar fit.

This verifier performs no fit or prediction application. It reads only the
authorized OOF training bundle and durable fit receipts; it never opens E79,
Test, SampleSubmission, or any CSV.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "REG_CENTER_LOCAL_APPLICATION_V1_R3"
FIT_ID = "LOCAL_CENTER_FINAL"
DELTA_BOUNDS = (-2.0, 0.0)
COEFFICIENTS = {
    "intercept": 1.7288079036674198,
    "slope_low": 0.9933285599956824,
    "slope_high": 0.957778072288735,
}
PATHS = {
    "freeze": ROOT / "experiments/reg_center_local_application_v1/PREFIT_FREEZE_R3.json",
    "preflight": ROOT / "reports/optimization_v3/reg_center_local_application_v1/IDENTITY_PREFLIGHT_R3.json",
    "root_pass": ROOT / "experiments/reg_center_local_application_v1/ROOT_SOURCE_PASS_R3.json",
    "fit_state": ROOT / "experiments/reg_center_local_application_v1/fits_R3/LOCAL_CENTER_FINAL.json",
    "fit_receipt": ROOT / "experiments/reg_center_local_application_v1/fits_R3/LOCAL_CENTER_FINAL_RECEIPT.json",
    "ledger": ROOT / "experiments/reg_center_local_application_v1/calibrator_ledger_R3.jsonl",
    "oof_bundle": ROOT / "experiments/runs/reg_monotone_hinge_p50_v1/FINAL_APPLICATION_R1/fit_inputs/HINGE_FINAL.npz",
}
OUTPUT = ROOT / "reports/optimization_v3/reg_center_local_application_v1/INDEPENDENT_FIT_NUMERICAL_QA_R3.json"


class QAFailure(RuntimeError):
    """Independent numerical or durable-state mismatch."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(value), sort_keys=True, ensure_ascii=False,
                         allow_nan=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_signed(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise QAFailure(f"Cannot read signed artifact: {path}") from exc
    if not isinstance(value, dict):
        raise QAFailure(f"Expected object at {path}")
    signature = value.get("signature")
    body = dict(value)
    body.pop("signature", None)
    if not isinstance(signature, str) or signature != canonical_sha(body):
        raise QAFailure(f"Canonical signature mismatch: {path}")
    return value


def exact_rational_weighted_median(values: np.ndarray,
                                   weights: np.ndarray) -> tuple[float, float]:
    """Independent lower-median interval using exact Fraction weights."""
    x = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if (x.ndim != 1 or w.shape != x.shape or len(x) == 0
            or not np.isfinite(x).all() or not np.isfinite(w).all()
            or np.any(w <= 0.0)):
        raise QAFailure("Median inputs are empty, nonfinite, or misaligned")
    grouped: dict[float, Fraction] = {}
    for value, weight in zip(x.tolist(), w.tolist()):
        key = float(value)
        grouped[key] = grouped.get(key, Fraction(0, 1)) + Fraction.from_float(float(weight))
    ordered = sorted(grouped.items(), key=lambda item: item[0])
    total = sum((weight for _value, weight in ordered), Fraction(0, 1))
    cumulative = Fraction(0, 1)
    for index, (value, weight) in enumerate(ordered):
        cumulative += weight
        twice = 2 * cumulative
        if twice > total:
            return value, value
        if twice == total:
            if index + 1 >= len(ordered):
                return value, value
            return value, ordered[index + 1][0]
    raise QAFailure("Exact rational weighted median did not cross half total weight")


def _ledger(path: Path) -> list[dict[str, Any]]:
    try:
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, ValueError) as exc:
        raise QAFailure("Fit ledger is missing or malformed") from exc
    if any(not isinstance(event, dict) for event in events):
        raise QAFailure("Fit ledger contains a non-object event")
    return events


def run() -> dict[str, Any]:
    missing_paths = [str(path) for path in PATHS.values() if not path.is_file()]
    if missing_paths:
        raise QAFailure(f"Required fit/source artifact missing: {missing_paths}")

    freeze = read_signed(PATHS["freeze"])
    preflight = read_signed(PATHS["preflight"])
    root_pass = read_signed(PATHS["root_pass"])
    state = read_signed(PATHS["fit_state"])
    fit_receipt = read_signed(PATHS["fit_receipt"])
    freeze_sha = sha256_file(PATHS["freeze"])
    preflight_sha = sha256_file(PATHS["preflight"])
    pass_sha = sha256_file(PATHS["root_pass"])
    state_sha = sha256_file(PATHS["fit_state"])
    fit_receipt_sha = sha256_file(PATHS["fit_receipt"])
    bundle_sha = sha256_file(PATHS["oof_bundle"])

    if (root_pass.get("verdict") != "PASS"
            or root_pass.get("experiment_id") != EXPERIMENT_ID
            or root_pass.get("pass_kind") != "ONE_LOCAL_SCALAR_FIT"
            or root_pass.get("authorized_fit_ids") != [FIT_ID]
            or root_pass.get("authorized_calibrator_fits") != 1
            or root_pass.get("target_array_access_authorized") is not True
            or root_pass.get("e79_access_authorized") is not False
            or root_pass.get("test_or_sample_access_authorized") is not False
            or root_pass.get("csv_creation_authorized") is not False
            or root_pass.get("kaggle_upload_authorized") is not False):
        raise QAFailure("Root pass does not match the one-fit/no-application boundary")
    if (state.get("status") != "PASS_LOCAL_SCALAR_FIT_COMPLETE"
            or state.get("experiment_id") != EXPERIMENT_ID
            or state.get("fit_id") != FIT_ID
            or state.get("root_source_pass_sha256") != pass_sha
            or state.get("preparation_freeze_sha256") != freeze_sha
            or state.get("identity_preflight_sha256") != preflight_sha
            or state.get("oof_bundle_sha256") != bundle_sha
            or state.get("fixed_hinge_coefficients") != COEFFICIENTS
            or state.get("delta_bounds") != list(DELTA_BOUNDS)
            or fit_receipt.get("status") != "PASS_DURABLE_LOCAL_FIT_RECEIPT"
            or fit_receipt.get("fit_state_sha256") != state_sha
            or fit_receipt.get("root_source_pass_sha256") != pass_sha):
        raise QAFailure("Saved fit state/receipt does not bind the approved source pass")

    ledger = _ledger(PATHS["ledger"])
    if (len(ledger) != 2 or ledger[0].get("event") != "FIT_RESERVED"
            or ledger[1].get("event") != "FIT_COMPLETE"
            or any(event.get("fit_id") != FIT_ID for event in ledger)
            or any(event.get("root_source_pass_sha256") != pass_sha for event in ledger)
            or ledger[1].get("fit_state_sha256") != state_sha
            or ledger[1].get("fit_receipt_sha256") != fit_receipt_sha):
        raise QAFailure("Ledger must show exactly one reserved and completed fit")

    # The root pass permits this independent read of the frozen OOF training targets.
    with np.load(PATHS["oof_bundle"], allow_pickle=False) as archive:
        direct = np.asarray(archive["direct_predictions"], dtype=np.float64).copy()
        missing = np.asarray(archive["missing_mask"], dtype=bool).copy()
        ids = np.asarray(archive["record_ids"]).astype(str)
        origins = np.asarray(archive["prediction_origins"], dtype=np.int64)
        indices = np.asarray(archive["original_bundle_indices"], dtype=np.int64)
        cycles = np.asarray(archive["cycles"], dtype=np.int64)
        y = np.asarray(archive["targets"], dtype=np.float64).copy()

    n = len(direct)
    if (n != 20_903 or y.shape != (n,) or missing.shape != (n,)
            or len(np.unique(ids)) != n
            or len(np.unique(np.column_stack((origins, indices)), axis=0)) != n
            or not np.all(cycles > origins)):
        raise QAFailure("Independent OOF identity/population/causal validation failed")
    present = ~missing
    basis = np.maximum(0.0, 1.0 - np.abs(direct - 50.0) / 10.0)
    active = present & (basis > 0.0)
    bzero = present & (basis == 0.0)
    if (int(present.sum()) != 19_528 or int(missing.sum()) != 1_375
            or int(active.sum()) != 4_125 or int(bzero.sum()) != 15_403):
        raise QAFailure("Independent fit population/support counts do not match R3")

    baseline = (COEFFICIENTS["intercept"]
                + COEFFICIENTS["slope_low"] * np.minimum(direct, 50.0)
                + COEFFICIENTS["slope_high"] * np.maximum(direct - 50.0, 0.0))
    ratios = (y[active] - baseline[active]) / basis[active]
    raw_low, raw_high = exact_rational_weighted_median(ratios, basis[active])
    bounded_low = float(np.clip(raw_low, *DELTA_BOUNDS))
    bounded_high = float(np.clip(raw_high, *DELTA_BOUNDS))
    expected_delta = bounded_low
    saved_delta = float(state["delta"])
    if (not math.isclose(saved_delta, expected_delta, rel_tol=0.0, abs_tol=1e-12)
            or not np.allclose(state["delta_unbounded_minimizer_interval"],
                               [raw_low, raw_high], rtol=0.0, atol=1e-12)
            or not np.allclose(state["delta_bounded_minimizer_interval"],
                               [bounded_low, bounded_high], rtol=0.0, atol=1e-12)):
        raise QAFailure("Saved delta or minimizer interval differs from independent exact-weight replay")

    residual = (y[active] - baseline[active]) - saved_delta * basis[active]
    independent_objective = math.fsum(abs(float(value)) for value in residual)
    saved_objective = float(state["choice_dependent_objective"]["unweighted_absolute_residual_sum"])
    if not math.isclose(independent_objective, saved_objective,
                        rel_tol=1e-12, abs_tol=1e-8):
        raise QAFailure("Saved choice-dependent objective differs from independent replay")

    expected_ids_hash = preflight["identity"]["identity_hashes"]["record_ids"]
    ids_hash_payload = (ids.astype(ids.dtype).tobytes(order="C"))
    # Bind IDs to the already-preflighted hash separately from the median math.
    id_hash_check = hashlib.sha256()
    id_hash_check.update(str(ids.dtype).encode("utf-8"))
    id_hash_check.update(json.dumps(list(ids.shape), separators=(",", ":")).encode("ascii"))
    id_hash_check.update(ids_hash_payload)
    if id_hash_check.hexdigest() != expected_ids_hash:
        raise QAFailure("Independent record ID hash differs from target-blind preflight")

    body: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS_INDEPENDENT_NUMERICAL_FIT_AUDIT",
        "experiment_id": EXPERIMENT_ID,
        "fit_id": FIT_ID,
        "root_source_pass_sha256": pass_sha,
        "preparation_freeze_sha256": freeze_sha,
        "identity_preflight_sha256": preflight_sha,
        "fit_state_sha256": state_sha,
        "fit_receipt_sha256": fit_receipt_sha,
        "fit_ledger_sha256": sha256_file(PATHS["ledger"]),
        "oof_bundle_sha256": bundle_sha,
        "independent_checker_sha256": sha256_file(Path(__file__).resolve()),
        "population": {
            "universe_rows": n,
            "present_rows": int(present.sum()),
            "excluded_missing_rows": int(missing.sum()),
            "present_b_positive_rows": int(active.sum()),
            "present_b_zero_rows": int(bzero.sum()),
            "record_ids_unique": int(len(np.unique(ids))),
            "origin_bundle_pairs_unique": int(len(np.unique(np.column_stack((origins, indices)), axis=0))),
            "causal_origin_before_cycle": True,
        },
        "independent_numerical_replay": {
            "method": "exact Fraction.from_float cumulative weights; grouped ratios; lower weighted median interval",
            "unbounded_minimizer_interval": [raw_low, raw_high],
            "bounded_minimizer_interval": [bounded_low, bounded_high],
            "saved_delta": saved_delta,
            "independent_delta": expected_delta,
            "delta_abs_difference": abs(saved_delta - expected_delta),
            "active_choice_objective_saved": saved_objective,
            "active_choice_objective_independent_fsum": independent_objective,
            "objective_abs_difference": abs(saved_objective - independent_objective),
            "b_zero_rows_choice_independent": True,
        },
        "fit_accounting": {
            "ledger_events": [event["event"] for event in ledger],
            "calibrator_fits": 1,
            "base_model_fits": 0,
            "preprocessing_fits": 0,
            "official_lp_starts": 0,
        },
        "access_boundary": {
            "authorized_oof_target_array_loaded_for_independent_qa": True,
            "e79_read": False,
            "test_or_sample_read": False,
            "csv_created": False,
            "kaggle_accessed": False,
        },
        "out_of_sample_metrics_computed": False,
    }
    body["signature"] = canonical_sha(body)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        existing = read_signed(OUTPUT)
        old = dict(existing)
        old.pop("signature", None)
        if old != body:
            raise QAFailure("Existing independent QA receipt differs from replay")
        return existing
    payload = json.dumps(body, sort_keys=True, ensure_ascii=False,
                         allow_nan=False, indent=2).encode("utf-8") + b"\n"
    with OUTPUT.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        import os
        os.fsync(stream.fileno())
    return body


def main() -> int:
    global ROOT, PATHS, OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    ROOT = args.root.resolve()
    PATHS = {name: ROOT / path.relative_to(Path(__file__).resolve().parents[1])
             for name, path in PATHS.items()}
    OUTPUT = ROOT / "reports/optimization_v3/reg_center_local_application_v1/INDEPENDENT_FIT_NUMERICAL_QA_R3.json"
    result = run()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except QAFailure as exc:
        print(json.dumps({"status": "FAIL_INDEPENDENT_NUMERICAL_FIT_AUDIT", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        raise SystemExit(2)
