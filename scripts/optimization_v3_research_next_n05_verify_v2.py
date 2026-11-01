"""Post-fit independent N05 audit; preserves the original frozen verifier."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from scripts import optimization_v3_research_next_n00 as n00
from scripts import optimization_v3_research_next_n05_verify as v1
from src.models.research_next_budget import read_events, ledger_summary
from src.models.research_next_contract import canonical_sha256, file_sha256, read_bundle


ROOT = v1.ROOT
FREEZE_PATH = v1.FREEZE_PATH
LEDGER_PATH = v1.LEDGER_PATH
RESULT_PATH = v1.RESULT_PATH
VERIFY_PATH = v1.VERIFY_PATH
SUMMARY_PATH = ROOT / "reports/optimization_v3/research_next_v1/results/N05/amendment_01/SUMMARY.json"
QUEUE_PATH = v1.QUEUE_PATH
BUNDLE_ROOT = v1.BUNDLE_ROOT
ORIGINS = v1.ORIGINS
ALPHAS = v1.ALPHAS


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify() -> dict[str, Any]:
    frozen = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    identity = frozen["identity"]
    signature = frozen["freeze_signature"]
    if signature != canonical_sha256(identity):
        raise RuntimeError("N05 v2 verifier: frozen identity signature mismatch")
    source_commit = frozen.get("execution_context", {}).get("source_commit")
    if (not source_commit or subprocess.run(["git", "merge-base", "--is-ancestor", source_commit, "HEAD"],
                                            cwd=ROOT, check=False).returncode != 0):
        raise RuntimeError("N05 v2 verifier: original frozen commit is not an ancestor")
    verified_hashes = {}
    for relative, expected in identity["input_and_source_sha256"].items():
        path = ROOT / Path(relative)
        actual = file_sha256(path)
        if actual != expected:
            raise RuntimeError(f"N05 v2 verifier: frozen input/source changed: {relative}")
        verified_hashes[relative] = actual

    train_path, frame, target = v1._load_data()
    sources = v1._r22_sources(frame)
    queue = json.loads(QUEUE_PATH.read_text(encoding="utf-8"))
    ledger = read_events(LEDGER_PATH)
    reservations = [event for event in ledger if event.get("event") == "CARD_FITS_RESERVED"
                    and event.get("reservation_key") == "N05:PILOT"]
    if len(reservations) != 1 or reservations[0].get("fits") != 4:
        raise RuntimeError("N05 v2 verifier: expected one exact four-fit reservation")

    stored_result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    if (stored_result.get("freeze_signature") != signature
            or stored_result.get("status") != "COMPLETE_INFORMATION"
            or stored_result.get("point_model_selection", {}).get("eligible") is not False):
        raise RuntimeError("N05 v2 verifier: report identity/status/eligibility is invalid")

    arrays_by_origin: dict[int, dict[str, np.ndarray]] = {}
    past_by_origin: dict[int, dict[str, Any]] = {}
    fits = []
    for cutoff in ORIGINS:
        source = sources[cutoff]
        past = v1._past_residual(frame, target, sources, cutoff)
        past_by_origin[cutoff] = past
        arrays_by_origin[cutoff] = {}
        for alpha in ALPHAS:
            fit_id = f"N05:PILOT:{cutoff}:q{int(alpha * 100):02d}"
            component = f"quantile_alpha_{alpha:.1f}"
            bundle_identity = {"batch_id": "optimization_v3_research_next_v1", "card_id": "N05",
                               "freeze_signature": signature, "stage": "PILOT",
                               "cutoff": cutoff, "component": component}
            bundle = read_bundle(BUNDLE_ROOT / f"fit{cutoff}" / f"q{int(alpha * 100):02d}",
                                 bundle_identity, task="regression")
            arrays = bundle.arrays
            diag = bundle.receipt["metadata"].get("diagnostics", {})
            fit_rows = np.flatnonzero(frame.cycle_code.to_numpy(dtype=np.int64) <= cutoff)
            fit_target = target[fit_rows]
            expected_lower, expected_upper, _ = v1._raw_interval(
                source["prediction"] + past["q10"], source["prediction"] + past["q90"])
            checks = {
                "score_rows": np.array_equal(arrays["score_rows"], source["rows"]),
                "record_ids": np.array_equal(arrays["score_record_ids"].astype(str), source["ids"]),
                "cycle_code": np.array_equal(arrays["cycle_code"], source["cycles"]),
                "R22_ORIGINAL": np.array_equal(arrays["baseline_prediction"], source["prediction"]),
                "control_lower": np.array_equal(arrays["control_lower"], expected_lower),
                "control_upper": np.array_equal(arrays["control_upper"], expected_upper),
                "prediction_clipping": np.array_equal(arrays["prediction"],
                                                       np.clip(arrays["raw_quantile_prediction"], 0, 100)),
                "alpha": diag.get("alpha") == alpha,
                "fit_prefix_rows": diag.get("fit_rows") == len(fit_rows),
                "fit_row_hash": diag.get("fit_row_indices_sha256") == v1._sha256_array(fit_rows, "<i8"),
                "fit_target_hash": diag.get("fit_target_sha256") == v1._sha256_array(fit_target),
                "score_labels_excluded": diag.get("score_target_passed_to_fit") is False,
                "test_excluded": diag.get("test_data_used") is False,
                "quantile_objective": diag.get("quantile_parameters", {}).get("objective") == "quantile",
                "quantile_alpha_parameter": diag.get("quantile_parameters", {}).get("alpha") == alpha,
            }
            if not all(checks.values()):
                failed = [name for name, passed in checks.items() if not passed]
                raise RuntimeError(f"N05 v2 verifier: {fit_id} failed {failed}")
            events = [event for event in ledger if event.get("fit_id") == fit_id]
            if (sum(event.get("event") == "FIT_STARTED" for event in events) != 1
                    or sum(event.get("event") == "FIT_SAVED" for event in events) != 1
                    or any(event.get("event") == "FIT_FAILED" for event in events)):
                raise RuntimeError(f"N05 v2 verifier: ledger fit state invalid for {fit_id}")
            arrays_by_origin[cutoff][alpha] = arrays
            fits.append({"fit_id": fit_id, "receipt_sha256": bundle.receipt["receipt_sha256"],
                         "npz_sha256": bundle.receipt["npz_sha256"], "alpha": alpha,
                         "fit_rows": int(diag["fit_rows"]), "score_rows": len(arrays["score_rows"]),
                         "checks": checks})
    n05_started = [event for event in ledger if event.get("event") == "FIT_STARTED"
                   and event.get("card_id") == "N05" and event.get("stage") == "PILOT"]
    if len(n05_started) != 4:
        raise RuntimeError(f"N05 v2 verifier: found {len(n05_started)} starts, expected four")

    verified_views: dict[str, Any] = {}
    primary_parts = []
    for cutoff in ORIGINS:
        source = sources[cutoff]
        q10 = arrays_by_origin[cutoff][0.1]["raw_quantile_prediction"].astype(np.float64)
        q90 = arrays_by_origin[cutoff][0.9]["raw_quantile_prediction"].astype(np.float64)
        low, high, _ = v1._raw_interval(q10, q90)
        control_low, control_high, _ = v1._raw_interval(
            source["prediction"] + past_by_origin[cutoff]["q10"],
            source["prediction"] + past_by_origin[cutoff]["q90"])
        cycles = source["cycles"]
        y = target[source["rows"]]
        primary_block = next(tuple(map(int, block)) for block in queue["views"]["LATE"]["primary"]
                             if int(block[0]) == cutoff)
        primary_mask = (cycles >= primary_block[1]) & (cycles <= primary_block[2])
        crossing_mask = q10[primary_mask] > q90[primary_mask]
        candidate = v1._metric(y[primary_mask], low[primary_mask], high[primary_mask],
                               source["prediction"][primary_mask])
        control = v1._metric(y[primary_mask], control_low[primary_mask], control_high[primary_mask],
                             source["prediction"][primary_mask])
        stored = stored_result["origins"][str(cutoff)]["primary"]
        checks = [
            candidate["n"] == stored["candidate"]["n"],
            v1._same(candidate["coverage_80"], stored["candidate"]["coverage_80"]),
            v1._same(candidate["mean_width"], stored["candidate"]["mean_width"]),
            control["n"] == stored["past_residual_control"]["n"],
            v1._same(control["coverage_80"], stored["past_residual_control"]["coverage_80"]),
            v1._same(control["mean_width"], stored["past_residual_control"]["mean_width"]),
            v1._same(float(crossing_mask.mean()), stored["candidate"]["raw_crossing_rate"]),
        ]
        if not all(checks):
            raise RuntimeError(f"N05 v2 verifier: primary candidate/control/crossing differs at {cutoff}")
        for cycle_text, record in stored["per_cycle"].items():
            mask = primary_mask & (cycles == int(cycle_text))
            cm = v1._metric(y[mask], low[mask], high[mask], source["prediction"][mask])
            bm = v1._metric(y[mask], control_low[mask], control_high[mask], source["prediction"][mask])
            if (cm["n"] != record["candidate"]["n"]
                    or not v1._same(cm["coverage_80"], record["candidate"]["coverage_80"])
                    or not v1._same(cm["mean_width"], record["candidate"]["mean_width"])
                    or bm["n"] != record["past_residual_control"]["n"]
                    or not v1._same(bm["coverage_80"], record["past_residual_control"]["coverage_80"])):
                raise RuntimeError(f"N05 v2 verifier: cycle metric differs at {cutoff}/{cycle_text}")
        verified_views[f"primary_{cutoff}"] = {
            "n": candidate["n"], "candidate_coverage_80": candidate["coverage_80"],
            "candidate_mean_width": candidate["mean_width"],
            "control_coverage_80": control["coverage_80"],
            "control_mean_width": control["mean_width"],
            "raw_crossing_rate": float(crossing_mask.mean()),
        }
        primary_parts.append({"y": y[primary_mask], "point": source["prediction"][primary_mask],
                              "low": low[primary_mask], "high": high[primary_mask],
                              "control_low": control_low[primary_mask],
                              "control_high": control_high[primary_mask],
                              "rows": source["rows"][primary_mask]})

    pooled_rows = np.concatenate([part["rows"] for part in primary_parts])
    if len(pooled_rows) != len(np.unique(pooled_rows)):
        raise RuntimeError("N05 v2 verifier: pooled LATE primary rows overlap")
    pooled = {key: np.concatenate([part[key] for part in primary_parts])
              for key in primary_parts[0] if key != "rows"}
    pooled_candidate = v1._metric(pooled["y"], pooled["low"], pooled["high"], pooled["point"])
    pooled_control = v1._metric(pooled["y"], pooled["control_low"], pooled["control_high"], pooled["point"])
    saved = stored_result["late_primary_pooled"]
    if (len(pooled_rows) != stored_result["late_primary_unique_rows"]
            or not v1._same(pooled_candidate["coverage_80"], saved["candidate"]["coverage_80"])
            or not v1._same(pooled_candidate["mean_width"], saved["candidate"]["mean_width"])
            or not v1._same(pooled_control["coverage_80"], saved["past_residual_control"]["coverage_80"])):
        raise RuntimeError("N05 v2 verifier: pooled LATE primary metrics differ")

    # Verify H12_L from the 67 origin separately; it is descriptive and overlaps primary.
    source67 = sources[67]
    arrays67 = arrays_by_origin[67]
    low67, high67, _ = v1._raw_interval(arrays67[0.1]["raw_quantile_prediction"],
                                         arrays67[0.9]["raw_quantile_prediction"])
    past67 = past_by_origin[67]
    control_low67, control_high67, _ = v1._raw_interval(
        source67["prediction"] + past67["q10"], source67["prediction"] + past67["q90"])
    h12_y = target[source67["rows"]]
    h12_candidate = v1._metric(h12_y, low67, high67, source67["prediction"])
    h12_control = v1._metric(h12_y, control_low67, control_high67, source67["prediction"])
    stored_h12 = stored_result["late_h12_from_67"]
    if (not v1._same(h12_candidate["coverage_80"], stored_h12["candidate"]["coverage_80"])
            or not v1._same(h12_candidate["mean_width"], stored_h12["candidate"]["mean_width"])
            or not v1._same(h12_control["coverage_80"], stored_h12["past_residual_control"]["coverage_80"])):
        raise RuntimeError("N05 v2 verifier: H12_L metrics differ")

    verification = {
        "status": "PASS", "verification_method": "independent cache reconstruction, bundle/ledger and metric recomputation",
        "original_freeze_signature": signature, "official_train_sha256": file_sha256(train_path),
        "original_frozen_source_hashes_verified": len(verified_hashes),
        "postfit_verifier_source": "scripts/optimization_v3_research_next_n05_verify_v2.py",
        "postfit_verifier_sha256": _sha256(Path(__file__)),
        "fit_count": 4, "fits": fits, "verified_primary_views": verified_views,
        "verified_pooled_late_primary": {
            "n": len(pooled_rows), "candidate_coverage_80": pooled_candidate["coverage_80"],
            "candidate_mean_width": pooled_candidate["mean_width"],
            "control_coverage_80": pooled_control["coverage_80"],
            "control_mean_width": pooled_control["mean_width"],
        },
        "verified_h12_late_from_67": {
            "n": int(len(h12_y)), "candidate_coverage_80": h12_candidate["coverage_80"],
            "candidate_mean_width": h12_candidate["mean_width"],
            "control_coverage_80": h12_control["coverage_80"],
        },
        "test_files_opened": False, "test_labels_read": False, "kaggle_calls": 0,
        "information_only_no_point_promotion": True,
        "budget": ledger_summary(LEDGER_PATH),
    }
    if VERIFY_PATH.exists():
        existing = json.loads(VERIFY_PATH.read_text(encoding="utf-8"))
        if canonical_sha256(existing) != canonical_sha256(verification):
            raise RuntimeError("N05 v2 verifier refuses to overwrite a different verification result")
    else:
        VERIFY_PATH.parent.mkdir(parents=True, exist_ok=True)
        VERIFY_PATH.write_text(json.dumps(verification, sort_keys=True, indent=2, allow_nan=False) + "\n",
                               encoding="utf-8")
    summary = {
        "status": stored_result["status"], "question": stored_result["question"],
        "origins": {key: {
            "primary_candidate_coverage": value["primary"]["candidate"]["coverage_80"],
            "primary_control_coverage": value["primary"]["past_residual_control"]["coverage_80"],
            "primary_candidate_width": value["primary"]["candidate"]["mean_width"],
            "primary_control_width": value["primary"]["past_residual_control"]["mean_width"],
            "width_error_spearman": value["primary"]["candidate"]["width_absolute_error_spearman"],
            "raw_crossing_rate": value["primary"]["candidate"]["raw_crossing_rate"],
        } for key, value in stored_result["origins"].items()},
        "pooled_late_primary": stored_result["late_primary_pooled"],
        "late_h12_from_67": stored_result["late_h12_from_67"],
        "fit_count": 4, "verification_status": verification["status"],
        "point_model_selection": stored_result["point_model_selection"],
        "budget": ledger_summary(LEDGER_PATH),
    }
    if SUMMARY_PATH.exists():
        existing_summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        if canonical_sha256(existing_summary) != canonical_sha256(summary):
            raise RuntimeError("N05 v2 verifier refuses to overwrite a different summary")
    else:
        SUMMARY_PATH.write_text(json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n",
                                encoding="utf-8")
    return verification


def main() -> int:
    result = _verify()
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
