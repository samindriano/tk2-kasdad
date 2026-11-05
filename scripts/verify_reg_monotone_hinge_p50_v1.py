"""Independent replay checks for the fixed-knot p50 hinge lane."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from scripts import reg_monotone_hinge_p50_v1 as hinge

ROOT = hinge.ROOT
PROTOCOL = ROOT / "reports/optimization_v3/reg_monotone_hinge_p50_v1/PROTOCOL.md"
TESTS = ROOT / "tests/test_reg_monotone_hinge_p50_v1.py"


def verify_fit_replay(state: dict[str, Any], direct_predictions: Any,
                      targets: Any) -> dict[str, float | int]:
    """Reconstruct the fitted curve and L1 objective without calling the builder."""
    p = np.asarray(direct_predictions, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if p.ndim != 1 or y.ndim != 1 or len(p) != len(y) or not len(p):
        raise hinge.HingeError("Independent fit replay requires aligned nonempty vectors")
    intercept = float(state["intercept"])
    slope_low = float(state["slope_low"])
    slope_high = float(state["slope_high"])
    if not (0.0 <= slope_low <= 2.0 and 0.0 <= slope_high <= 2.0):
        raise hinge.HingeError("Independent replay found slope bounds violated")
    replay = intercept + slope_low * np.minimum(p, 50.0) + slope_high * np.maximum(p - 50.0, 0.0)
    objective = float(np.abs(y - replay).sum())
    recorded = float(state["objective_unweighted_sum_absolute"])
    if (not np.all(np.isfinite(replay)) or not np.isfinite(objective)
            or abs(objective - recorded) > 1e-7 * max(1.0, objective)):
        raise hinge.HingeError("Independent replay objective mismatch")
    return {"rows": len(p), "replayed_objective": objective,
            "objective_error": abs(objective - recorded)}


def read_stored_global_affine(path: Path, expected_sha256: str) -> tuple[float, float]:
    if hinge.file_sha256(path) != expected_sha256:
        raise hinge.HingeError("Stored GLOBAL state hash mismatch")
    state = json.loads(path.read_text(encoding="utf-8"))
    result = state.get("solver_result")
    if not isinstance(result, dict):
        raise hinge.HingeError("Stored GLOBAL state has no solver_result")
    intercept, slope = float(result["intercept"]), float(result["slope"])
    if not np.isfinite(intercept) or not np.isfinite(slope) or slope < 0.0:
        raise hinge.HingeError("Stored GLOBAL affine coefficients are invalid")
    return intercept, slope


def verify_candidate_csv(path: Path, *, expected_ids: Any,
                         expected_predictions: Any, missing_mask: Any,
                         raw_direct: Any) -> dict[str, int | str]:
    ids = np.asarray(expected_ids, dtype=str)
    expected = np.asarray(expected_predictions, dtype="<f8")
    missing = np.asarray(missing_mask, dtype=bool)
    raw = np.asarray(raw_direct, dtype="<f8")
    if not (ids.ndim == expected.ndim == missing.ndim == raw.ndim == 1
            and len(ids) == len(expected) == len(missing) == len(raw)):
        raise hinge.HingeError("CSV verification vectors are not aligned")
    if not len(ids) or len(np.unique(ids)) != len(ids):
        raise hinge.HingeError("CSV IDs must be nonempty and unique")
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["record_id", "prediction"]:
            raise hinge.HingeError("CSV header differs from the frozen route schema")
        rows = list(reader)
    got_ids = np.asarray([row["record_id"] for row in rows], dtype=str)
    try:
        got_values = np.asarray([float(row["prediction"]) for row in rows], dtype="<f8")
    except (TypeError, ValueError) as exc:
        raise hinge.HingeError("CSV contains a nonnumeric prediction") from exc
    if len(got_ids) != len(ids) or not np.array_equal(got_ids, ids):
        raise hinge.HingeError("CSV row count or ordered official IDs differ")
    if not np.all(np.isfinite(got_values)) or np.any(got_values < 0.0) or np.any(got_values > 100.0):
        raise hinge.HingeError("CSV values must be finite and in [0, 100]")
    if not np.array_equal(got_values.view("<u8"), expected.view("<u8")):
        raise hinge.HingeError("CSV predictions differ from the saved candidate array")
    if not np.array_equal(got_values[missing].view("<u8"), raw[missing].view("<u8")):
        raise hinge.HingeError("CSV missing route is not bit-exact raw DIRECT")
    return {"rows": len(ids), "missing_rows": int(missing.sum()),
            "csv_sha256": hinge.file_sha256(path)}


def _identity_fixture() -> dict[str, np.ndarray]:
    return {
        "physical_rows": np.array([101, 102, 103, 104], dtype=np.int64),
        "record_ids": np.array(["a", "b", "c", "d"]),
        "prediction_origins": np.array(["E67", "E67", "E75", "E75"]),
        "original_bundle_indices": np.array([4, 5, 1, 2], dtype=np.int64),
        "cycles": np.array([68, 69, 76, 77], dtype=np.int64),
        "direct_predictions": np.array([10.0, 30.0, 60.0, 90.0], dtype="<f8"),
        "missing_mask": np.array([False, True, False, False]),
    }


def run_synthetic_checks() -> dict[str, Any]:
    checks: dict[str, Any] = {}

    # Recover a known continuous two-segment mapping with balanced side support.
    p = np.linspace(0.0, 100.0, 1_200, dtype=np.float64)
    truth = {"intercept": 8.0, "slope_low": 0.45, "slope_high": 1.2}
    y = truth["intercept"] + truth["slope_low"] * np.minimum(p, 50.0) \
        + truth["slope_high"] * np.maximum(p - 50.0, 0.0)
    fitted = hinge.solve_fixed_knot_p50_l1(p, y, purpose="diagnostic")
    replay = verify_fit_replay(fitted, p, y)
    recovery_error = max(abs(fitted["intercept"] - truth["intercept"]),
                         abs(fitted["slope_low"] - truth["slope_low"]),
                         abs(fitted["slope_high"] - truth["slope_high"]))
    checks["known_hinge_recovery"] = {"pass": recovery_error < 1e-7,
                                       "max_parameter_error": recovery_error,
                                       "objective": fitted["objective_unweighted_sum_absolute"],
                                       "independent_replay": replay}

    # Equal slopes reduce algebraically to an affine mapping; check the knot seam.
    grid = np.linspace(0.0, 100.0, 501)
    affine_params = hinge.HingeParameters(2.0, 0.7, 0.7)
    hinge_affine = hinge.predict_hinge(grid, affine_params)
    affine = 2.0 + 0.7 * grid
    seam = hinge.predict_hinge(np.array([50.0]), affine_params)[0]
    checks["equal_slope_affine_and_continuity"] = {
        "pass": bool(np.allclose(hinge_affine, affine, rtol=0.0, atol=1e-12)
                      and seam == 2.0 + 0.7 * 50.0),
        "max_affine_error": float(np.max(np.abs(hinge_affine - affine))),
        "seam_value": float(seam),
    }

    monotone = hinge.predict_hinge(grid, hinge.HingeParameters(-1.0, 0.0, 2.0))
    rejected = 0
    for bad in (hinge.HingeParameters(0.0, -0.01, 1.0),
                hinge.HingeParameters(0.0, 1.0, 2.01)):
        try:
            hinge.validate_parameters(bad)
        except hinge.HingeError:
            rejected += 1
    checks["monotonicity_and_slope_bounds"] = {
        "pass": bool(np.all(np.diff(monotone) >= -1e-12) and rejected == 2),
        "minimum_step": float(np.min(np.diff(monotone))),
        "invalid_bound_cases_rejected": rejected,
    }

    import inspect
    no_weights_argument = "weights" not in inspect.signature(hinge.solve_fixed_knot_p50_l1).parameters
    exact_unweighted_loss = abs(float(np.abs(y - (fitted["intercept"]
        + fitted["slope_low"] * np.minimum(p, 50.0)
        + fitted["slope_high"] * np.maximum(p - 50.0, 0.0))).sum())
        - fitted["objective_unweighted_sum_absolute"]) < 1e-7
    checks["unweighted_objective_only"] = {"pass": no_weights_argument and exact_unweighted_loss,
                                            "weights_parameter_present": not no_weights_argument}

    identity = _identity_fixture()
    expected_identity = hinge.row_identity_hashes(**identity)
    callback_calls = []
    hinge.load_targets_after_identity(identity, expected_identity,
                                      lambda: callback_calls.append("loaded") or np.array([1.0]))
    reordered = dict(identity)
    reordered["record_ids"] = identity["record_ids"][[1, 0, 2, 3]]
    row_order_rejected = False
    try:
        hinge.load_targets_after_identity(reordered, expected_identity,
                                          lambda: callback_calls.append("bad-order"))
    except hinge.HingeError:
        row_order_rejected = True
    wrong_hash_rejected = False
    wrong = dict(expected_identity)
    wrong["cycles_sha256"] = "0" * 64
    try:
        hinge.load_targets_after_identity(identity, wrong,
                                          lambda: callback_calls.append("bad-hash"))
    except hinge.HingeError:
        wrong_hash_rejected = True
    checks["identity_before_target_access"] = {
        "pass": row_order_rejected and wrong_hash_rejected and callback_calls == ["loaded"],
        "row_order_rejected": row_order_rejected,
        "wrong_hash_rejected": wrong_hash_rejected,
        "target_loader_calls": len(callback_calls),
    }

    # Persist and replay a synthetic stored GLOBAL control by its pinned file hash.
    with tempfile.TemporaryDirectory(prefix="hinge-global-") as temporary:
        path = Path(temporary) / "D1_GLOBAL.json"
        path.write_text(json.dumps({"solver_result": {"intercept": 2.25, "slope": 0.9}},
                                   sort_keys=True), encoding="utf-8")
        digest = hinge.file_sha256(path)
        a, b = read_stored_global_affine(path, digest)
        raw = np.array([0.0, 10.0, 70.0, 100.0], dtype="<f8")
        missing = np.array([False, True, False, False])
        first = hinge.apply_global_present_only(raw, missing, intercept=a, slope=b)
        second = hinge.apply_global_present_only(raw, missing, intercept=a, slope=b)
        checks["stored_global_control_replay"] = {
            "pass": bool(np.array_equal(first.view("<u8"), second.view("<u8"))),
            "state_sha256": digest,
        }

    cycles = np.concatenate((np.arange(68, 80), np.arange(68, 80)))
    origins = np.array(["E67"] * 12 + ["E75"] * 12)
    stage_counts = {
        "D1_fit": int(hinge.stage_mask(cycles, origins, stage="D1", role="fit").sum()),
        "D1_score": int(hinge.stage_mask(cycles, origins, stage="D1", role="score").sum()),
        "D2_fit": int(hinge.stage_mask(cycles, origins, stage="D2", role="fit").sum()),
        "D2_score": int(hinge.stage_mask(cycles, origins, stage="D2", role="score").sum()),
    }
    checks["causal_stage_boundaries"] = {
        "pass": stage_counts == {"D1_fit": 4, "D1_score": 4, "D2_fit": 8, "D2_score": 4},
        "synthetic_cycle_counts": stage_counts,
        "authorized_row_counts": {"D1_fit": 4_950, "D1_score": 7_462,
                                  "D2_fit": 12_412, "D2_score": 8_491},
    }

    with tempfile.TemporaryDirectory(prefix="hinge-recovery-") as temporary:
        path = Path(temporary) / "HINGE_D1.json"
        identity_key = {"row_identity_sha256": "a" * 64, "source_freeze_sha256": "b" * 64}
        first_state, first_status = hinge.fit_or_recover_published_state(
            path, fit_id="HINGE_D1", input_identity=identity_key,
            producer=lambda: {"intercept": 1.0, "slope_low": 0.5, "slope_high": 0.8})
        second_state, second_status = hinge.fit_or_recover_published_state(
            path, fit_id="HINGE_D1", input_identity=identity_key,
            producer=lambda: (_ for _ in ()).throw(AssertionError("duplicate fit start")))
        checks["postpublish_crash_recovery"] = {
            "pass": first_status == "NEWLY_PUBLISHED" and second_status == "RECOVERED_PUBLISHED"
                    and first_state == second_state,
            "first_status": first_status, "recovery_status": second_status,
        }

    schema_values = np.array([1.25, 2.5, 50.0], dtype="<f8")
    checks["producer_hash_schemas"] = {
        "pass": (hinge.raw_direct_f64_sha256(schema_values)
                 != hinge.tagged_array_sha256(schema_values)),
        "direct_raw_sha256": hinge.raw_direct_f64_sha256(schema_values),
        "tagged_vector_sha256": hinge.tagged_array_sha256(schema_values),
    }

    with tempfile.TemporaryDirectory(prefix="hinge-csv-") as temporary:
        raw = np.array([0.0, 100.0, -0.0, 40.0], dtype="<f8")
        missing = np.array([False, False, True, True])
        params = hinge.HingeParameters(3.0, 0.5, 1.8)
        candidate = hinge.apply_hinge_present_only(raw, missing, params)
        ids = np.array(["r1", "r2", "r3", "r4"])
        path = Path(temporary) / "candidate.csv"
        hinge.write_candidate_csv(path, ids, candidate)
        csv_result = verify_candidate_csv(path, expected_ids=ids,
                                          expected_predictions=candidate,
                                          missing_mask=missing, raw_direct=raw)
        checks["csv_ids_range_and_missing_route"] = {
            "pass": csv_result["rows"] == 4 and csv_result["missing_rows"] == 2,
            **csv_result,
        }

    failures = [name for name, result in checks.items() if not result["pass"]]
    if failures:
        raise AssertionError("Synthetic QA failures: " + ", ".join(failures))
    return checks


def write_synthetic_qa() -> dict[str, Any]:
    checks = run_synthetic_checks()
    source_files = (
        "scripts/reg_monotone_hinge_p50_v1.py",
        "scripts/verify_reg_monotone_hinge_p50_v1.py",
        "tests/test_reg_monotone_hinge_p50_v1.py",
        "reports/optimization_v3/reg_monotone_hinge_p50_v1/PROTOCOL.md",
    )
    source_hashes = {relative: hinge.file_sha256(ROOT / relative) for relative in source_files}
    report = {
        "schema_version": 1,
        "status": "PASS_SYNTHETIC_ONLY",
        "checks": checks,
        "source_hashes": source_hashes,
        "candidate_data_read": False,
        "official_train_labels_read": False,
        "test_or_sample_read": False,
        "official_lp_starts": 0,
        "synthetic_lp_starts": 1,
        "base_or_preprocessing_fits": 0,
    }
    path = hinge.SYNTHETIC_QA
    payload = json.dumps(report, sort_keys=True, ensure_ascii=False,
                         allow_nan=False, indent=2).encode("utf-8") + b"\n"
    if path.exists():
        if path.read_bytes() != payload:
            raise hinge.HingeError("Existing synthetic QA receipt differs; refusing overwrite")
        digest = hinge.file_sha256(path)
    else:
        digest = hinge.write_json_once(path, report)
    return {"status": report["status"], "path": str(path),
            "sha256": digest, "checks": len(checks), "source_hashes": source_hashes}
