"""Corrected, saved-output-only verifier for frozen R14 pilot bundles.

This version amends two defects in the first post-fit report verifier without
changing the frozen recipe, source, inputs, or model fits.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from scripts import optimization_v3_batch1_r14 as r14
from src.models.optimization_v3_batch1 import file_sha256

# Expose the exact frozen runner namespace required by the copied independent
# reconstruction. The frozen module/file is never modified by this verifier.
globals().update({name: getattr(r14, name) for name in dir(r14) if not name.startswith("__")})

FROZEN_RUNNER_SHA256 = "18122ed85ab67354816dff78913d0a106537fc680cf829af838adb5116fe3518"
PRIMARY_ROW_MATCHED_BASELINE_MAE = 13.464395831218011
PRIMARY_ROW_MATCHED_ROWS = 9064
FULLSIX_PRIMARY_REFERENCE_MAE = 11.559390078665364
FULLSIX_PRIMARY_REFERENCE_ROWS = 38588
CONTROL_SOURCE = "reports/optimization_v2/regression_repr/fullsix.json"


def _verify_r14_control_populations(view_metrics: Mapping[str, Mapping[str, Any]]) -> None:
    primary = view_metrics["primary"]
    h12 = view_metrics["H12"]
    if int(primary["rows"]) != PRIMARY_ROW_MATCHED_ROWS:
        raise R14RunnerError("R14 primary pilot denominator must be the exact W1+W4 row set")
    if abs(float(primary["baseline_mae"]) - PRIMARY_ROW_MATCHED_BASELINE_MAE) > 1e-9:
        raise R14RunnerError("R14 primary baseline differs from the pinned row-matched W1+W4 control")
    if int(h12["rows"]) != EXPECTED_VIEW_ROWS["H12"]:
        raise R14RunnerError("R14 H12 denominator differs from the frozen 12-cycle control")
    if abs(float(h12["baseline_mae"]) - EXPECTED_CONTROL_METRICS["H12_mae"]) > 1e-9:
        raise R14RunnerError("R14 H12 incumbent metric differs from the accepted control")


def _control_metric_population_audit(view_metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    _verify_r14_control_populations(view_metrics)
    return {
        "R14_primary_W1_W4": {
            "rows": PRIMARY_ROW_MATCHED_ROWS,
            "baseline_mae": PRIMARY_ROW_MATCHED_BASELINE_MAE,
            "meaning": "Row-matched incumbent on the R14 STANDARD pilot primary population (W1+W4).",
        },
        "V2_fullsix_primary_reference": {
            "rows": FULLSIX_PRIMARY_REFERENCE_ROWS,
            "mae": FULLSIX_PRIMARY_REFERENCE_MAE,
            "meaning": "Full-six-window V2 candidate metric; provenance only, not the R14 pilot primary comparator.",
            "asserted_equal_to_R14_primary": False,
        },
        "R14_H12": {
            "rows": int(view_metrics["H12"]["rows"]),
            "baseline_mae": float(view_metrics["H12"]["baseline_mae"]),
            "meaning": "Full H12 temporal control; equals the accepted V2 H12 incumbent reference.",
        },
        "source": CONTROL_SOURCE,
    }


def _max_guarded_harm(group_metrics: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> float | None:
    guarded_groups = {"tail_low_le5", "tail_high_ge95", "stream_missing"}
    harms = [
        float(row["harm_candidate_minus_baseline"])
        for view in ("primary", "H12_E", "H12_L")
        for group_name, row in group_metrics[view].items()
        if group_name in guarded_groups
        and bool(row["supported"])
        and row["harm_candidate_minus_baseline"] is not None
        and math.isfinite(float(row["harm_candidate_minus_baseline"]))
    ]
    return max(harms) if harms else None


def verify_pilot_v2(context: Mapping[str, Any], report: Mapping[str, Any],
                  target: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
    if (report.get("card_id") != CARD_ID
            or report.get("status") != "PILOT_SAVED_AWAITING_INDEPENDENT_QA"
            or report.get("fit_predictor_count") != PILOT_FITS
            or report.get("cutoffs") != [55, 67]
            or report.get("test_data_used") is not False):
        raise R14RunnerError("R14 generic pilot report identity, fit count, or boundary is invalid")
    signature = json_sha256(context["identity"])
    if report.get("freeze_signature") != signature:
        raise R14RunnerError("R14 pilot report does not match the frozen identity")
    frame = context["frame"]
    cycles = context["cycles"]
    ids = frame["record_id"].astype(str).to_numpy(dtype=str)
    if target.shape != (len(frame),) or not np.isfinite(target).all() or np.any((target < 0) | (target > 100)):
        raise R14RunnerError("Official regression target is invalid or misaligned")
    per_cutoff: dict[int, dict[str, np.ndarray]] = {}

    for cutoff in (55, 67):
        controls = context["controls"][cutoff]
        score_rows = np.asarray(controls["score_rows"], dtype=np.int64)
        bundle_path = Path(context.get("artifact_dir", RUN_DIR)) / f"R14_fit{cutoff}.npz"
        bundle_identity = {"freeze_signature": signature, "card_id": CARD_ID,
                           "cutoff": cutoff, "score_rows": score_rows.tolist()}
        arrays, sidecar = read_bundle(bundle_path, bundle_identity)
        if (not np.array_equal(arrays.get("score_rows"), score_rows)
                or not np.array_equal(arrays.get("score_record_ids"), ids[score_rows])
                or len(np.unique(score_rows)) != len(score_rows)):
            raise R14RunnerError(f"R14 saved score row positions/IDs differ at cutoff {cutoff}")
        metadata = sidecar["metadata"]
        fit_rows = np.flatnonzero(cycles <= cutoff)
        fit_mask = ef1_mask(frame.iloc[fit_rows]["entry_friction"])
        ef_fit_rows = fit_rows[fit_mask]
        score_mask = ef1_mask(frame.iloc[score_rows]["entry_friction"])
        raw = np.asarray(arrays.get("raw_candidate"), dtype=np.float64)
        prediction = np.asarray(arrays.get("prediction"), dtype=np.float64)
        baseline = np.asarray(arrays.get("baseline_prediction"), dtype=np.float64)
        saved_mask = np.asarray(arrays.get("audit__expert_applied_mask"), dtype=np.uint8).astype(bool)
        if any(values.shape != score_rows.shape for values in (raw, prediction, baseline, saved_mask)):
            raise R14RunnerError(f"R14 bundle arrays are misaligned at cutoff {cutoff}")
        if not np.array_equal(saved_mask, score_mask) or np.any(raw[~score_mask] != 0.0):
            raise R14RunnerError(f"R14 expert application mask/sentinel differs at cutoff {cutoff}")
        if not np.isfinite(raw).all() or not np.isfinite(prediction).all() or not np.isfinite(baseline).all():
            raise R14RunnerError(f"R14 saved predictions are nonfinite at cutoff {cutoff}")
        if np.any((prediction < 0.0) | (prediction > 100.0)) or np.any((baseline < 0.0) | (baseline > 100.0)):
            raise R14RunnerError(f"R14 saved predictions fall outside [0,100] at cutoff {cutoff}")

        state = metadata.get("state", {})
        fit_meta = metadata.get("fit_metadata", {})
        if (int(metadata.get("fit_predictor_count", -1)) != 1
                or int(metadata.get("fit_rows", -1)) != len(fit_rows)
                or int(metadata.get("score_rows", -1)) != len(score_rows)
                or int(fit_meta.get("ef1_fit_rows", -1)) != len(ef_fit_rows)
                or int(state.get("fit_rows", -1)) != len(ef_fit_rows)
                or fit_meta.get("expert_fit_population") != "parsed entry_friction == 1"
                or fit_meta.get("score_labels_passed") is not False):
            raise R14RunnerError(f"R14 fit receipt does not prove EF1-only fit at cutoff {cutoff}")
        expected_fingerprint = _fit_fingerprint_independent(
            frame.iloc[ef_fit_rows].loc[:, list(FEATURE_NAMES)], target[ef_fit_rows]
        )
        target_sha = hashlib.sha256(np.asarray(target[ef_fit_rows], dtype="<f8").tobytes(order="C")).hexdigest()
        if state.get("fit_data_sha256") != expected_fingerprint or state.get("target_sha256") != target_sha:
            raise R14RunnerError(f"R14 saved model was not fit on the exact EF1 rows at cutoff {cutoff}")

        ridge = np.asarray(arrays.get("control__ridge25_prediction"), dtype=np.float64)
        sparse_raw = np.asarray(arrays.get("control__old_sparse_raw"), dtype=np.float64)
        saved_rank = np.asarray(arrays.get("control__old_sparse_rank50"), dtype=np.float64)
        control_rows = np.asarray(arrays.get("control__control_rows"), dtype=np.int64)
        score_cycles = cycles[score_rows]
        if (not np.array_equal(control_rows, score_rows)
                or ridge.shape != score_rows.shape or sparse_raw.shape != score_rows.shape
                or saved_rank.shape != score_rows.shape):
            raise R14RunnerError(f"R14 incumbent cache alignment differs at cutoff {cutoff}")
        if not np.array_equal(saved_rank, _independent_rank50(sparse_raw, score_cycles)):
            raise R14RunnerError(f"R14 old sparse rank50 reconstruction differs at cutoff {cutoff}")
        expected_baseline = np.clip(0.75 * ridge + 0.25 * saved_rank, 0.0, 100.0)
        if not np.allclose(baseline, expected_baseline, rtol=0.0, atol=1e-10):
            raise R14RunnerError(f"R14 incumbent route changed at cutoff {cutoff}")
        expected_prediction = baseline.copy()
        expected_prediction[score_mask] = (
            0.75 * baseline[score_mask] + 0.25 * np.clip(raw[score_mask], 0.0, 100.0)
        )
        expected_prediction = np.clip(expected_prediction, 0.0, 100.0)
        if not np.allclose(prediction, expected_prediction, rtol=0.0, atol=1e-10):
            raise R14RunnerError(f"R14 fixed EF1 blend does not reconstruct at cutoff {cutoff}")
        if not np.array_equal(prediction[~score_mask], baseline[~score_mask]):
            raise R14RunnerError(f"R14 changed non-EF1 incumbent rows at cutoff {cutoff}")
        if len(np.unique(score_cycles)) != 12 or not np.array_equal(
            np.unique(score_cycles), np.arange(cutoff + 1, cutoff + 13)
        ):
            raise R14RunnerError(f"R14 score horizon is incomplete at cutoff {cutoff}")
        per_cutoff[cutoff] = {"score_rows": score_rows, "prediction": prediction,
                              "baseline_prediction": baseline}

    view_segments = dict(context["views"])
    view_segments["H12_E"] = [(55, 56, 67)]
    view_segments["H12_L"] = [(67, 68, 79)]
    view_metrics: dict[str, Any] = {}
    view_data: dict[str, dict[str, np.ndarray]] = {}
    for name, segments in view_segments.items():
        rows, pred, base = _view_rows(frame, cycles, per_cutoff, segments)
        observed = {"rows": int(len(rows)), "candidate_mae": _mae(target[rows], pred),
                    "baseline_mae": _mae(target[rows], base)}
        observed["gain_positive_is_better"] = observed["baseline_mae"] - observed["candidate_mae"]
        if name in report.get("views", {}):
            reported = report["views"][name]
            if (int(reported.get("rows", -1)) != len(rows)
                    or abs(float(reported["candidate"]["mae"]) - observed["candidate_mae"]) > 1e-10
                    or abs(float(reported["baseline"]["mae"]) - observed["baseline_mae"]) > 1e-10):
                raise R14RunnerError(f"R14 generic aggregation differs from independent {name} metrics")
        view_metrics[name] = observed
        view_data[name] = {"rows": rows, "prediction": pred, "baseline": base}
    _verify_r14_control_populations(view_metrics)

    group_metrics: dict[str, Any] = {}
    ef = pd.to_numeric(frame["entry_friction"], errors="coerce").to_numpy()
    missing_stream = stream_missing_mask(frame["signal_stream"])
    for view_name in ("primary", "H12_E", "H12_L"):
        data = view_data[view_name]
        rows = data["rows"]
        masks = {
            "EF0": ef[rows] == 0,
            "EF1": ef[rows] == 1,
            "tail_low_le5": target[rows] <= 5,
            "tail_high_ge95": target[rows] >= 95,
            "stream_missing": missing_stream[rows],
        }
        group_metrics[view_name] = {}
        for group_name, mask in masks.items():
            n = int(mask.sum())
            supported = n >= int(EXPECTED_GATE["supported_group_n_min"])
            row = {"n": n, "supported": supported}
            if supported:
                row["candidate_mae"] = _mae(target[rows][mask], data["prediction"][mask])
                row["baseline_mae"] = _mae(target[rows][mask], data["baseline"][mask])
                row["harm_candidate_minus_baseline"] = row["candidate_mae"] - row["baseline_mae"]
            else:
                row.update({"candidate_mae": None, "baseline_mae": None,
                            "harm_candidate_minus_baseline": None})
            group_metrics[view_name][group_name] = row

    individual_harms = {
        name: view_metrics[name]["candidate_mae"] - view_metrics[name]["baseline_mae"]
        for name in ("W1", "W4")
    }
    max_guarded_harm = _max_guarded_harm(group_metrics)
    primary_gain = view_metrics["primary"]["gain_positive_is_better"]
    h12_harm = view_metrics["H12"]["candidate_mae"] - view_metrics["H12"]["baseline_mae"]
    gate_checks = {
        "primary_gain_at_least_0_03": primary_gain >= EXPECTED_GATE["pilot_primary_MAE_gain_min"],
        "H12_harm_at_most_0_05": h12_harm <= EXPECTED_GATE["pilot_H12_MAE_harm_max"],
        "W1_W4_harm_at_most_0_5": all(
            value <= EXPECTED_GATE["individual_pilot_view_MAE_harm_max"]
            for value in individual_harms.values()
        ),
        "supported_tail_or_missing_harm_at_most_0_25": (
            max_guarded_harm is not None
            and max_guarded_harm <= EXPECTED_GATE["supported_tail_or_missing_MAE_harm_max"]
        ),
    }
    decision = "ADVANCE" if all(gate_checks.values()) else "RETIRE"
    verification = {
        "control_metric_population_audit": _control_metric_population_audit(view_metrics),
        "card_id": CARD_ID, "status": "INDEPENDENT_QA_PASS", "decision": decision,
        "freeze_signature": signature, "fit_predictor_count": PILOT_FITS,
        "cutoffs": [55, 67], "views": view_metrics,
        "individual_pilot_view_harms": individual_harms,
        "supported_subgroup_metrics": group_metrics,
        "maximum_supported_tail_or_missing_harm": max_guarded_harm,
        "gate_checks": gate_checks, "gate_definition": EXPECTED_GATE,
        "control_reconstruction": "0.75*cached Ridge25 + 0.25*independently recomputed rank50(cached sparse raw), clipped [0,100]",
        "route_reconstruction": "EF1 only: 0.75*incumbent + 0.25*clipped raw expert; all others array-equal incumbent",
        "expert_clipping_order": "Raw expert clipped to [0,100] before the 25% blend; final blend clipped again.",
        "fit_population_independently_verified": True,
        "target_rows_used_for_fit": "outer temporal fit rows with parsed entry_friction == 1 only",
        "test_data_used": False, "sample_submission_used": False, "kaggle_used": False,
    }
    gate = {
        "card_id": CARD_ID, "status": decision,
        "freeze_signature": signature, "thresholds": EXPECTED_GATE,
        "observed": {"primary_gain": primary_gain, "H12_harm": h12_harm,
                     "W1_harm": individual_harms["W1"], "W4_harm": individual_harms["W4"],
                     "max_supported_tail_or_missing_harm": max_guarded_harm},
        "checks": gate_checks,
        "decision_basis": "Frozen row-exact temporal pilot and independent verification only; no Test/Kaggle evidence.",
    }
    return verification, gate



def run_saved_output_verification() -> dict[str, Any]:
    if file_sha256(Path(r14.__file__)) != FROZEN_RUNNER_SHA256:
        raise R14RunnerError("Frozen R14 runner source hash changed; this amendment is not applicable")
    queue = load_queue(QUEUE_PATH)
    card = next(item for item in queue["cards"] if item.get("id") == CARD_ID)
    context = r14._prepare(card, queue)
    if context["public"].get("status") != "PREFLIGHT_PASS":
        raise R14RunnerError("Current read-only R14 preflight does not pass")
    frozen = r14._read_json(FREEZE_PATH)
    if frozen.get("identity") != r14._safe_json(context["identity"]):
        raise R14RunnerError("Current R14 inputs differ from the original frozen identity")
    preflight = r14._read_json(PREFLIGHT_PATH)
    if preflight.get("freeze_identity_sha256") != json_sha256(context["identity"]):
        raise R14RunnerError("Saved R14 preflight does not match the frozen identity")
    if file_sha256(context["train_path"]) != TRAIN_SHA256:
        raise R14RunnerError("Official regression Train hash changed before verification")
    report = r14._read_json(ENGINE_REPORT_PATH)
    target_frame = pd.read_csv(context["train_path"], usecols=["threat_rank"])
    if file_sha256(context["train_path"]) != TRAIN_SHA256:
        raise R14RunnerError("Official regression Train changed while target was loaded")
    target = pd.to_numeric(target_frame["threat_rank"], errors="raise").to_numpy(dtype=np.float64)

    verification, gate = verify_pilot_v2(context, report, target)
    verification["verification_version"] = "R14 saved-output verifier v2"
    verification["verifier_script_sha256"] = file_sha256(Path(__file__))
    verification["frozen_runner_sha256"] = FROZEN_RUNNER_SHA256
    accounting = r14._fit_accounting()
    r14._publish(context, report, verification, gate, accounting)

    fullsix_path = ROOT / CONTROL_SOURCE
    fullsix_sha = file_sha256(fullsix_path)
    fullsix = r14._read_json(fullsix_path)
    candidate = fullsix["candidate_gates"]["lgbm_sparse_l1"]["primary_candidate"]
    if (int(candidate["n"]) != FULLSIX_PRIMARY_REFERENCE_ROWS
            or abs(float(candidate["mae"]) - FULLSIX_PRIMARY_REFERENCE_MAE) > 1e-12):
        raise R14RunnerError("V2 full-six provenance does not match the pinned value")
    amendment = {
        "card_id": CARD_ID,
        "amendment": 1,
        "verification_version": "R14 saved-output verifier v2",
        "frozen_protocol_unchanged": True,
        "freeze_signature": json_sha256(context["identity"]),
        "frozen_runner_sha256": FROZEN_RUNNER_SHA256,
        "verifier_script_sha256": file_sha256(Path(__file__)),
        "fit_bundles_reused_without_refit": True,
        "fit_bundles": {
            "R14_fit55.npz": file_sha256(RUN_DIR / "R14_fit55.npz"),
            "R14_fit55.json": file_sha256(RUN_DIR / "R14_fit55.json"),
            "R14_fit67.npz": file_sha256(RUN_DIR / "R14_fit67.npz"),
            "R14_fit67.json": file_sha256(RUN_DIR / "R14_fit67.json"),
        },
        "v1_report_defects": [
            {
                "issue": "The primary baseline assertion compared the 9,064-row W1+W4 pilot denominator to a 38,588-row V2 full-six metric.",
                "evidence": {
                    "R14_row_matched_W1_W4": {"rows": PRIMARY_ROW_MATCHED_ROWS, "baseline_mae": PRIMARY_ROW_MATCHED_BASELINE_MAE},
                    "V2_fullsix": {"rows": int(candidate["n"]), "mae": float(candidate["mae"]), "path": CONTROL_SOURCE, "sha256": fullsix_sha},
                },
                "correction": "Verify the R14 primary against its exact row-matched W1+W4 incumbent, and retain the full-six V2 number as provenance only.",
            },
            {
                "issue": "The subgroup gate aggregation indexed each group row as if it were a nested mapping, causing KeyError after valid pilot bundles had been saved.",
                "correction": "Aggregate supported tail/missing harm values directly across primary, H12_E, and H12_L.",
            },
        ],
        "corrected_metrics": {
            "primary": verification["views"]["primary"],
            "H12": verification["views"]["H12"],
            "W1": verification["views"]["W1"],
            "W4": verification["views"]["W4"],
            "maximum_supported_tail_or_missing_harm": verification["maximum_supported_tail_or_missing_harm"],
            "gate_checks": gate["checks"],
            "decision": gate["status"],
        },
        "no_test_sample_or_kaggle_used": True,
        "no_model_or_frozen_protocol_changed": True,
    }
    r14._write_or_require_same(REPORT_DIR / "report_amendment_01.json", amendment)
    r14._reconcile_budget(verification)
    return {
        "decision": gate["status"],
        "primary_gain": verification["views"]["primary"]["gain_positive_is_better"],
        "H12_harm": verification["views"]["H12"]["candidate_mae"] - verification["views"]["H12"]["baseline_mae"],
        "maximum_supported_tail_or_missing_harm": verification["maximum_supported_tail_or_missing_harm"],
        "gate_checks": gate["checks"],
        "fit_predictors": accounting["verified_successful_predictor_fits"],
        "amendment": str((REPORT_DIR / "report_amendment_01.json").relative_to(ROOT)),
    }


def main() -> int:
    result = run_saved_output_verification()
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
