"""Recover a completed C09 pilot from its frozen fit caches without refitting.

The original runner compared a 9,064-row W1+W4 pilot baseline to the queue's
38,588-row full-six primary metric. This script preserves the freeze and fits,
verifies both cached predictions, uses row-exact pilot controls, and records the
scope mismatch alongside the unchanged card gates.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts import optimization_v3_batch1_c09 as c09
from src.models.optimization_v3_batch1 import (
    BatchIntegrityError,
    exclusive_training_slot,
    file_sha256,
    write_json_exclusive,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = c09.REPORT / "recovery_manifest.json"


def _read_budget_reservation() -> dict[str, Any]:
    events = [json.loads(line) for line in c09.BATCH_LEDGER_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    matches = [event for event in events
               if event.get("event") == "BUDGET_RESERVED" and event.get("budget_key") == "C09:PILOT"]
    if len(matches) != 1 or matches[0].get("predictor_fits") != 2:
        raise BatchIntegrityError("C09 recovery requires one exact two-fit budget reservation")
    return matches[0]


def recover() -> dict[str, Any]:
    output_paths = [c09.RESULTS_PATH, c09.VERIFICATION_PATH, c09.SUMMARY_PATH, MANIFEST_PATH]
    if any(path.exists() for path in output_paths):
        raise FileExistsError("C09 recovery refuses to overwrite existing result artifacts")

    freeze, preflight = c09._check_freeze()
    _, frame, y = c09._load_train()
    _, control_refs = c09._baseline_files_identity()
    controls = {end: c09._load_control_fold(frame, y, end, control_refs) for end in c09.FIT_ENDS}
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=int)

    candidate_bundles: dict[str, dict[str, Any]] = {}
    fit_states: dict[str, Any] = {}
    fit_attempts: list[dict[str, Any]] = []
    cache_receipts: dict[str, Any] = {}
    for end in c09.FIT_ENDS:
        record = controls[end]
        reused = c09._reuse_verified_fit(record, freeze["signature"])
        if reused is None:
            raise BatchIntegrityError(f"C09 recovery found no completed frozen fit cache at cutoff {end}")
        _raw, routed, metadata = reused
        attempt_path = c09.RUN_DIR / f"fit{end}.attempt.json"
        attempt = c09._read_json(attempt_path)
        fit_attempts.append(attempt)
        record["candidate"] = routed
        state = metadata["model_state"]
        fit_states[str(end)] = {
            "fit_rows": int(record["fit_rows"].size),
            "fit_cycles": [47, end],
            "score_cycles": [end + 1, c09.HORIZON_BY_END[end]],
            "cache_status": "NEW_FIT_RELOADED_AND_HASH_VERIFIED",
            "cache_sha256": metadata["npz_sha256"],
            "seconds": float(metadata["seconds"]),
            "state": state,
        }
        full_bundle = c09._bundle(
            record["rows"], record["baseline"], routed, record["labels"], record["edges"],
            cycles[record["rows"]],
        )
        key = "H12_E" if end == 55 else "H12_L"
        candidate_bundles[key] = full_bundle
        low, high, window_name = c09.PRIMARY_BY_END[end]
        keep = (cycles[record["rows"]] >= low) & (cycles[record["rows"]] <= high)
        candidate_bundles[window_name] = c09._subset_bundle(full_bundle, keep)
        for suffix in ("npz", "json", "attempt.json"):
            path = c09.RUN_DIR / (f"fit{end}.{suffix}" if suffix != "attempt.json" else f"fit{end}.attempt.json")
            if not path.is_file():
                raise FileNotFoundError(path)
            cache_receipts[path.relative_to(ROOT).as_posix()] = file_sha256(path)

    primary = c09._concat_bundles([candidate_bundles["W1"], candidate_bundles["W4"]])
    h12 = c09._concat_bundles([candidate_bundles["H12_E"], candidate_bundles["H12_L"]])
    bundles = {**candidate_bundles, "primary_pilot": primary, "H12": h12}
    views = {name: c09._evaluate_bundle(y, bundle) for name, bundle in bundles.items()}

    queue, _card = c09._canonical_card()
    queue_baseline = queue["baseline"]["classification"]
    full_six_path = c09.PILOT_FULL_REPORT / "fullsix.json"
    full_six = c09._read_json(full_six_path)
    full_six_primary = full_six["results"]["primary_six_window_pooled"]
    full_six_h12 = full_six["results"]["H12_fixed_origin_pooled"]
    if (int(full_six_primary["n"]) != 38_588
            or not np.isclose(full_six_primary["models"]["blend25"]["macro_f1"],
                              float(queue_baseline["primary"]), atol=1e-12, rtol=0.0)
            or int(full_six_h12["n"]) != 38_588
            or not np.isclose(full_six_h12["models"]["blend25"]["macro_f1"],
                              float(queue_baseline["H12"]), atol=1e-12, rtol=0.0)):
        raise BatchIntegrityError("C09 queue controls do not reconstruct from the frozen full-six report")

    gate = queue["evaluation"]["classification_gate"]
    group_gate_pass = all(
        record.get("gate_pass") is not False
        for view in views.values()
        for groups in view["group_metrics"].values()
        for record in groups.values()
        if record.get("supported")
    )
    primary_gain = float(views["primary_pilot"]["delta_macro_f1"])
    h12_harm = -float(views["H12"]["delta_macro_f1"])
    individual_harms = {name: -float(views[name]["delta_macro_f1"]) for name in ("W1", "W4")}
    pilot_gate = {
        "primary_gain_min": float(gate["pilot_primary_gain_min"]),
        "primary_gain": primary_gain,
        "primary_pass": bool(primary_gain >= float(gate["pilot_primary_gain_min"])),
        "H12_harm_max": float(gate["pilot_H12_harm_max"]),
        "H12_harm": h12_harm,
        "H12_pass": bool(h12_harm <= float(gate["pilot_H12_harm_max"])),
        "individual_view_harm_max": float(gate["individual_pilot_view_harm_max"]),
        "individual_view_harms": individual_harms,
        "individual_views_pass": bool(all(
            value <= float(gate["individual_pilot_view_harm_max"])
            for value in individual_harms.values()
        )),
        "supported_group_gates_pass": bool(group_gate_pass),
    }
    pilot_gate["pass"] = bool(
        pilot_gate["primary_pass"] and pilot_gate["H12_pass"]
        and pilot_gate["individual_views_pass"] and pilot_gate["supported_group_gates_pass"]
    )
    verdict = "ADVANCE" if pilot_gate["pass"] else "RETIRE"
    reservation = _read_budget_reservation()
    result = {
        "status": "PILOT_COMPLETE_ADVANCE" if pilot_gate["pass"] else "PILOT_COMPLETE_RETIRE",
        "verdict": verdict,
        "experiment_id": "optimization_v3_batch1_C09",
        "freeze_signature": freeze["signature"],
        "campaign_budget_reservation": reservation,
        "new_predictor_fits": 2,
        "reused_verified_fits": 0,
        "recovery_cache_reuses": 2,
        "fit_attempt_accounting": {
            "attempts_started": len(fit_attempts),
            "successful_predictor_fits": int(sum(item.get("status") == "COMPLETED" for item in fit_attempts)),
            "failed_predictor_attempts": int(sum(item.get("status") == "FAILED" for item in fit_attempts)),
            "convergence_warning_count": int(sum(int(item.get("convergence_warning_count", 0)) for item in fit_attempts)),
            "fit_attempts": fit_attempts,
            "fit_seconds_sum": float(sum(float(item.get("seconds", 0.0)) for item in fit_attempts)),
        },
        "score_threshold": c09.THRESHOLD,
        "primary_denominator": int(views["primary_pilot"]["n"]),
        "H12_denominator": int(views["H12"]["n"]),
        "fallback_coverage": {
            name: {"stream_missing_n": view["group_metrics"]["stream"]["missing"]["n"],
                   "stream_present_n": view["group_metrics"]["stream"]["present"]["n"]}
            for name, view in views.items()
        },
        "memory_forecast": preflight["memory_forecast"],
        "test_read": False, "sample_read": False, "kaggle_access": False,
        "fit_states": fit_states,
        "views": views,
        "pilot_gate": pilot_gate,
        "control": {
            "present": ".75 C2_B + .25 frozen sparse binary; candidate substitutes C09 for C2_B",
            "missing": "frozen CatBoost, unchanged",
            "baseline_manifest_sha256": c09.EXPECTED_MANIFEST_SHA256,
            "baseline_macro_f1_from_queue": {
                "primary": float(queue_baseline["primary"]),
                "H12": float(queue_baseline["H12"]),
            },
            "baseline_macro_f1_row_reconstructed": {
                "primary": float(views["primary_pilot"]["baseline"]["macro_f1"]),
                "H12": float(views["H12"]["baseline"]["macro_f1"]),
            },
            "scope_reconciliation": {
                "queue_primary": "fullsix primary_six_window_pooled, 38,588 rows",
                "pilot_primary": "W1 cycles56-59 + W4 cycles68-71, 9,064 rows",
                "same_metric_population": False,
                "H12_population": "cycles56-79, 38,588 rows; exact match",
            },
        },
        "known_runner_scope_mismatch": {
            "initial_exception": "Row-reconstructed C09 primary control differs from the frozen baseline metric",
            "queue_primary": float(queue_baseline["primary"]),
            "queue_primary_rows": int(full_six_primary["n"]),
            "pilot_primary_baseline": float(views["primary_pilot"]["baseline"]["macro_f1"]),
            "pilot_primary_rows": int(views["primary_pilot"]["n"]),
            "cause": "The queue stores full-six primary, while the pilot primary is W1+W4. Comparing them as one denominator is invalid.",
            "candidate_metrics_recovered_from_frozen_fit_caches_without_refit": True,
        },
        "interpretation_limit": "Repeated Train-cycle temporal development evidence; not an untouched holdout.",
    }

    raw_verification = c09._verify_pilot_result(
        result, bundles, controls, y, frame, freeze["signature"],
    )
    raw_checks = dict(raw_verification["checks"])
    known_bad = raw_checks.pop("baseline queue metrics reconstruct", None)
    if known_bad is not False:
        raise BatchIntegrityError("Expected C09 queue/pilot primary-scope mismatch was not reproduced")
    manual_baseline_primary = c09._manual_macro_f1(
        y[primary["rows"]], primary["baseline"],
    )
    manual_candidate_primary = c09._manual_macro_f1(
        y[primary["rows"]], primary["candidate"],
    )
    manual_baseline_h12 = c09._manual_macro_f1(y[h12["rows"]], h12["baseline"])
    manual_candidate_h12 = c09._manual_macro_f1(y[h12["rows"]], h12["candidate"])
    corrected_scope_checks = {
        "queue_primary_reconstructs_from_fullsix_38588_rows": True,
        "pilot_primary_reconstructed_on_exact_9064_rows": (
            len(primary["rows"]) == 9_064
            and np.isclose(manual_baseline_primary, views["primary_pilot"]["baseline"]["macro_f1"], atol=1e-12, rtol=0.0)
            and np.isclose(manual_candidate_primary, views["primary_pilot"]["candidate"]["macro_f1"], atol=1e-12, rtol=0.0)
        ),
        "H12_reconstructed_on_exact_38588_rows": (
            len(h12["rows"]) == 38_588
            and np.isclose(manual_baseline_h12, float(queue_baseline["H12"]), atol=1e-12, rtol=0.0)
            and np.isclose(manual_candidate_h12, views["H12"]["candidate"]["macro_f1"], atol=1e-12, rtol=0.0)
        ),
        "all_other_frozen_route_fit_row_and_group_checks_pass": all(raw_checks.values()),
    }
    verification = {
        "status": "PASS_WITH_CONTROL_SCOPE_RECONCILIATION"
        if all(corrected_scope_checks.values()) else "FAIL",
        "freeze_signature": freeze["signature"],
        "original_runner_verification_status": raw_verification["status"],
        "original_runner_checks": {**raw_checks, "baseline queue metrics reconstruct": known_bad},
        "known_invalid_comparison": {
            "check": "baseline queue metrics reconstruct",
            "source": "runner assertion compares 9,064-row W1+W4 pilot primary to 38,588-row full-six queue primary",
            "preserved_failure": True,
        },
        "corrected_scope_checks": corrected_scope_checks,
        "manual_metric_reconstruction": {
            "primary_pilot_rows": int(len(primary["rows"])),
            "baseline_macro_f1": manual_baseline_primary,
            "candidate_macro_f1": manual_candidate_primary,
            "H12_rows": int(len(h12["rows"])),
            "H12_baseline_macro_f1": manual_baseline_h12,
            "H12_candidate_macro_f1": manual_candidate_h12,
        },
        "independent_group_denominators_and_guards": raw_verification["independent_group_denominators_and_guards"],
        "prediction_cache_and_receipt_hashes": {
            "runner_verifier": raw_verification["prediction_cache_and_receipt_hashes"],
            "recovery_receipts": cache_receipts,
        },
        "test_read": False, "sample_read": False, "kaggle_access": False,
        "pilot_decision": verdict,
    }
    if verification["status"] != "PASS_WITH_CONTROL_SCOPE_RECONCILIATION":
        raise BatchIntegrityError("C09 recovery did not independently pass all non-scope checks")

    # Protect the verified result and refuse partial overwrites before first write.
    if any(path.exists() for path in output_paths):
        raise FileExistsError("C09 output appeared during recovery; refusing overwrite")
    result["verification_status"] = verification["status"]
    write_json_exclusive(c09.RESULTS_PATH, result)
    verification["recovery_script_sha256"] = file_sha256(Path(__file__))
    write_json_exclusive(c09.VERIFICATION_PATH, verification)
    c09._write_summary(result, verification)
    manifest = {
        "schema_version": 1,
        "card_id": "C09",
        "status": "COMPLETE",
        "verdict": verdict,
        "freeze_signature": freeze["signature"],
        "recovery_script_sha256": file_sha256(Path(__file__)),
        "files": {
            str(path.relative_to(ROOT)).replace("\\", "/"): file_sha256(path)
            for path in [c09.RECIPE_PATH, c09.FREEZE_PATH, full_six_path,
                         *[c09.RUN_DIR / f"fit{end}.{suffix}" for end in c09.FIT_ENDS for suffix in ("npz", "json")],
                         *[c09.RUN_DIR / f"fit{end}.attempt.json" for end in c09.FIT_ENDS],
                         c09.RESULTS_PATH, c09.VERIFICATION_PATH, c09.SUMMARY_PATH]
        },
        "results_sha256": file_sha256(c09.RESULTS_PATH),
        "verification_sha256": file_sha256(c09.VERIFICATION_PATH),
        "summary_sha256": file_sha256(c09.SUMMARY_PATH),
        "candidate_fit_count": 2,
        "candidate_fit_replays": 0,
        "test_sample_kaggle_used": False,
    }
    write_json_exclusive(MANIFEST_PATH, manifest)
    return {
        "status": verification["status"],
        "verdict": verdict,
        "primary_pilot_gain": primary_gain,
        "primary_required_gain": pilot_gate["primary_gain_min"],
        "H12_harm": h12_harm,
        "failed_gates": [key for key, passed in pilot_gate.items() if key.endswith("_pass") and not passed],
        "report": str(c09.RESULTS_PATH.relative_to(ROOT)),
        "verification": str(c09.VERIFICATION_PATH.relative_to(ROOT)),
        "manifest": str(MANIFEST_PATH.relative_to(ROOT)),
    }


def main() -> int:
    with exclusive_training_slot(c09.TRAINING_LOCK_PATH):
        result = recover()
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
