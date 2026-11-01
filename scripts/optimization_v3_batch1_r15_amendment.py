"""Post-fit R15 verification amendment; reuses saved fits and never trains."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import optimization_v3_batch1_r15 as runner
from scripts.optimization_v3_batch1_verify import verify_report
from src.models.optimization_v3_batch1 import (
    BatchIntegrityError,
    file_sha256,
    json_sha256,
    load_queue,
    read_bundle,
    stream_missing_mask,
    write_json_exclusive,
)
from src.models.optimization_v3_batch1_r15 import (
    CARD_ID,
    STRUCTURED_FEATURE_NAMES,
    reconstruct_r15_arrays,
)
from src.models.round1_models import _fit_fingerprint


AMENDMENT_PATH = runner.REPORT_DIR / "verification_amendment.json"


def _expected_fit_population(context: Mapping[str, Any], cutoff: int) -> dict[str, Any]:
    all_rows = np.asarray(context["fit_rows_by_cutoff"][int(cutoff)], dtype=np.int64)
    frame = context["frame"]
    missing_mask = stream_missing_mask(
        frame.iloc[all_rows]["signal_stream"].to_numpy(dtype=object)
    )
    positions = all_rows[missing_mask]
    cycles = np.asarray(context["cycles"], dtype=np.int64)[positions]
    if not len(positions):
        raise BatchIntegrityError(f"R15 cutoff {cutoff} has no missing-stream fit rows")
    X_fit = frame.iloc[positions].loc[:, list(STRUCTURED_FEATURE_NAMES)]
    y_fit = np.asarray(context["target"], dtype=np.float64)[positions]
    return {
        "all_fit_rows": int(len(all_rows)),
        "missing_fit_rows": int(len(positions)),
        "fit_cycle_min": int(cycles.min()),
        "fit_cycle_max": int(cycles.max()),
        "fit_source_row_positions_sha256": hashlib.sha256(
            np.asarray(positions, dtype="<i8").tobytes(order="C")
        ).hexdigest(),
        "fit_missing_mask_sha256": hashlib.sha256(missing_mask.tobytes(order="C")).hexdigest(),
        "fit_data_sha256": _fit_fingerprint(X_fit, tuple(STRUCTURED_FEATURE_NAMES), y_fit),
    }


def _load_and_check_saved_cutoff(context: Mapping[str, Any], report: Mapping[str, Any],
                                 cutoff: int) -> dict[str, Any]:
    rows = np.asarray(context["score_rows_by_cutoff"][cutoff], dtype=np.int64)
    identity = {
        "freeze_signature": report["freeze_signature"],
        "card_id": CARD_ID,
        "cutoff": cutoff,
        "score_rows": rows.tolist(),
    }
    arrays, sidecar = read_bundle(runner.RUN_DIR / f"R15_fit{cutoff}.npz", identity)
    candidate, baseline = reconstruct_r15_arrays(arrays)
    official_missing = stream_missing_mask(
        context["frame"].iloc[rows]["signal_stream"].to_numpy(dtype=object)
    )
    expected_fit = _expected_fit_population(context, cutoff)
    state = sidecar["metadata"].get("state", {})
    model_state = state.get("model", {})
    checks = {
        "score_rows_exact": np.array_equal(np.asarray(arrays["score_rows"], dtype=np.int64), rows),
        "record_ids_exact": np.array_equal(
            np.asarray(arrays["score_record_ids"]).astype(str),
            context["frame"].iloc[rows]["record_id"].astype(str).to_numpy(dtype=str),
        ),
        "incumbent_exact": np.array_equal(
            baseline, np.asarray(context["controls_by_cutoff"][cutoff]["baseline_prediction"], dtype=np.float64)
        ),
        "stream_mask_exact": (
            np.array_equal(np.asarray(arrays["control__stream_missing"], dtype=bool), official_missing)
            and np.array_equal(np.asarray(arrays["audit__stream_missing"], dtype=bool), official_missing)
            and np.array_equal(np.asarray(arrays["audit__expert_missing_rows"], dtype=bool), official_missing)
        ),
        "present_rows_exact_incumbent": np.array_equal(candidate[~official_missing], baseline[~official_missing]),
        "candidate_finite_and_in_range": bool(
            np.isfinite(candidate).all() and np.all((candidate >= 0.0) & (candidate <= 100.0))
        ),
        "fit_population_missing_only_exact": (
            int(state.get("fit_rows", -1)) == expected_fit["missing_fit_rows"]
            and state.get("fit_source_row_positions_sha256") == expected_fit["fit_source_row_positions_sha256"]
            and state.get("fit_missing_mask_sha256") == expected_fit["fit_missing_mask_sha256"]
            and int(sidecar["metadata"].get("fit_rows", -1)) == expected_fit["all_fit_rows"]
            and int(sidecar["metadata"].get("fit_metadata", {}).get("fit_missing_stream_rows", -1)) == expected_fit["missing_fit_rows"]
            and int(model_state.get("fit_rows", -1)) == expected_fit["missing_fit_rows"]
            and model_state.get("fit_data_sha256") == expected_fit["fit_data_sha256"]
        ),
        "fit_cycle_range_exact": (
            int(state.get("fit_cycle_min", -1)) == expected_fit["fit_cycle_min"]
            and int(state.get("fit_cycle_max", -1)) == expected_fit["fit_cycle_max"]
        ),
        "structured_only_allowlist": state.get("feature_names") == list(STRUCTURED_FEATURE_NAMES),
        "text_excluded_and_rank_unused": (
            state.get("signal_stream_used_as_feature") is False
            and state.get("percentile_or_rank_mapping") is False
        ),
    }
    return {
        "rows": rows,
        "prediction": candidate,
        "baseline_prediction": baseline,
        "missing": official_missing,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "sidecar": sidecar,
    }


def _ensure_unchanged_freeze(context: Mapping[str, Any], frozen: Mapping[str, Any],
                             report: Mapping[str, Any]) -> None:
    identity = frozen.get("identity", {})
    signature = frozen.get("signature")
    if signature != json_sha256(identity) or report.get("freeze_signature") != signature:
        raise BatchIntegrityError("R15 saved report does not match the original frozen identity")
    current = context["identity"]
    if (identity.get("source_sha256") != current.get("source_sha256")
            or identity.get("input_sha256") != current.get("input_sha256")
            or identity.get("source_commit_context") != current.get("source_commit_context")
            or identity.get("feature_names") != current.get("feature_names")
            or identity.get("evaluation_views") != current.get("evaluation_views")
            or identity.get("package_versions") != current.get("package_versions")
            or identity.get("card") != current.get("card")):
        raise BatchIntegrityError("R15 current source/input/card identity differs from the fit freeze")
    if runner._read_json(runner.RECIPE_PATH) != identity.get("recipe"):
        raise BatchIntegrityError("R15 saved recipe artifact differs from the original frozen recipe")
    if report.get("status") != "PILOT_SAVED_AWAITING_INDEPENDENT_QA" or report.get("fit_count_matches_card") is not True:
        raise BatchIntegrityError("R15 engine report is incomplete or does not match the declared fit budget")


def verify_saved_pilot() -> dict[str, Any]:
    queue = load_queue(runner.QUEUE_PATH)
    card = next(item for item in queue["cards"] if item.get("id") == CARD_ID)
    context = runner._prepare(card, queue)
    frozen = runner._read_json(runner.FREEZE_PATH)
    report = runner._read_json(runner.ENGINE_REPORT_PATH)
    _ensure_unchanged_freeze(context, frozen, report)

    per_cutoff = {
        cutoff: _load_and_check_saved_cutoff(context, report, cutoff)
        for cutoff in runner.PILOT_CUTOFFS
    }
    if not all(item["status"] == "PASS" for item in per_cutoff.values()):
        raise BatchIntegrityError("R15 saved fit verification failed; preserve outputs and do not refit")

    verification = verify_report(
        report=report,
        frame=context["frame"],
        target=context["target"],
        artifact_dir=runner.RUN_DIR,
        freeze_signature=frozen["signature"],
        evaluation_views=context["views"],
        prediction_reconstructors={CARD_ID: reconstruct_r15_arrays},
    )
    verification["r15_saved_fit_checks"] = {
        str(cutoff): {"status": item["status"], "checks": item["checks"], "rows": len(item["rows"])}
        for cutoff, item in per_cutoff.items()
    }
    verification["verification_amendment"] = "scripts/optimization_v3_batch1_r15_amendment.py"
    verification["status"] = "PASS" if verification.get("status") == "PASS" else "FAIL"

    views = runner._view_arrays(context, per_cutoff)
    groups = runner._group_metrics(context, views)
    early_fallback = runner._early_fallback(context)
    group_payload = {
        "card_id": CARD_ID,
        "status": "PASS",
        **groups,
        "H12_E_exact_champion_fallback": early_fallback,
    }
    gate = runner._gate(report, groups, verification)
    preflight = dict(context["public"])
    preflight.update({
        "freeze_identity_sha256": frozen["signature"],
        "input_sha256": frozen["identity"]["input_sha256"],
        "source_sha256": frozen["identity"]["source_sha256"],
        "resource_preflight": frozen["identity"]["memory_preflight"],
        "support_preflight": frozen["identity"]["support_preflight"],
        "budget_preflight": frozen["identity"]["budget_preflight"],
        "row_matched_control_metrics": frozen["identity"]["recipe"]["row_matched_control_metrics"],
    })
    results = {
        "card_id": CARD_ID,
        "status": "PILOT_VERIFIED",
        "verdict": gate["status"],
        "source_commit_context": frozen["identity"]["source_commit_context"],
        "freeze_signature": frozen["signature"],
        "input_sha256": frozen["identity"]["input_sha256"],
        "source_sha256": frozen["identity"]["source_sha256"],
        "recipe_sha256": json_sha256(frozen["identity"]["recipe"]),
        "fit_accounting": runner._fit_accounting(),
        "population": {
            "fit_rows_by_cutoff": {
                str(cutoff): len(context["fit_rows_by_cutoff"][cutoff])
                for cutoff in runner.PILOT_CUTOFFS
            },
            "missing_fit_rows_by_cutoff": preflight["support_preflight"]["cutoffs"],
            "score_rows_by_cutoff": {
                str(cutoff): len(context["score_rows_by_cutoff"][cutoff])
                for cutoff in runner.PILOT_CUTOFFS
            },
            "view_rows": runner.EXPECTED_VIEW_ROWS,
            "view_missing_stream_rows": {
                name: groups["views"][name]["signal_stream_missing"]["n"]
                for name in context["views"]
            },
            "fit_subset_only_missing_stream": True,
        },
        "views": report["views"],
        "H12_E_exact_champion_fallback": early_fallback,
        "group_diagnostics": groups,
        "support_preflight": preflight["support_preflight"],
        "fallback_coverage": {
            name: {
                "rows": len(item["rows"]),
                "missing_rows_expert_route": int(np.asarray(item["missing"], dtype=bool).sum()),
                "present_rows_exact_champion": int((~np.asarray(item["missing"], dtype=bool)).sum()),
                "present_prediction_changed": False,
            }
            for name, item in views.items()
        },
        "test_or_sample_access": False,
        "kaggle_access": False,
        "candidate_test_prediction_made": False,
        "validation_labels_passed_to_adapter": False,
        "percentile_or_rank_mapping_used": False,
        "verification_amendment": "R15 post-fit report reader fix; saved fits reused without training",
    }
    runner.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    runner._write_or_match(runner.PREFLIGHT_PATH, preflight)
    runner._write_or_match(runner.VERIFICATION_PATH, verification)
    runner._write_or_match(runner.SUBGROUP_PATH, group_payload)
    runner._write_or_match(runner.GATE_PATH, gate)
    runner._write_or_match(runner.RESULTS_PATH, results)

    fit_metadata = [
        runner._read_json(runner.RUN_DIR / f"R15_fit{cutoff}.json")["metadata"]
        for cutoff in runner.PILOT_CUTOFFS
    ]
    runtime_notes = {
        "card_id": CARD_ID,
        "freeze_signature": frozen["signature"],
        "source_commit": frozen["identity"]["source_commit_context"],
        "predictor_fits": {
            "started": results["fit_accounting"]["fit_started_events"],
            "saved": results["fit_accounting"]["successful_saved_predictor_fits"],
            "failed": results["fit_accounting"]["fit_failed_events"],
            "technical_retries": results["fit_accounting"]["technical_retries"],
            "cutoff_67_seconds": float(fit_metadata[0]["fit_seconds"]),
            "cutoff_75_seconds": float(fit_metadata[1]["fit_seconds"]),
            "total_seconds": float(results["fit_accounting"]["fit_seconds_total"]),
        },
        "warnings_observed": [
            warning
            for item in fit_metadata
            for warning in item.get("fit_metadata", {}).get("fit_warnings", [])
        ],
        "memory": {
            "preflight_estimated_peak_gib_by_cutoff": {
                key: float(value["estimated_additional_peak_bytes"] / 1024**3)
                for key, value in preflight["resource_preflight"].items()
            },
            "preflight_available_ram_gib_by_cutoff": {
                key: float(value["available_ram_bytes"] / 1024**3)
                for key, value in preflight["resource_preflight"].items()
            },
            "job_cap_gib": 6.0,
            "peak_process_rss_measured": False,
            "peak_process_rss_gib": None,
        },
        "official_test_read": False,
        "kaggle_accessed": False,
        "verification_amendment": "saved pilot reused; no fit rerun",
    }
    runner._write_or_match(runner.RUNTIME_NOTES_PATH, runtime_notes)
    runner._reconcile_budget(results, verification, gate)
    summary = {
        "card_id": CARD_ID,
        "status": gate["status"],
        "verdict": gate["status"],
        "primary_gain": gate["metrics"]["primary"]["gain_positive_is_better"],
        "H12_L_gain": gate["metrics"]["H12_L"]["gain_positive_is_better"],
        "W4_gain": gate["metrics"]["W4"]["gain_positive_is_better"],
        "W6_gain": gate["metrics"]["W6"]["gain_positive_is_better"],
        "H12_E_policy": "exact champion fallback",
        "successful_predictor_fits": results["fit_accounting"]["successful_saved_predictor_fits"],
        "independent_verification": verification["status"],
        "test_or_sample_access": False,
        "kaggle_access": False,
    }
    runner._write_or_match(runner.SUMMARY_PATH, summary)
    amendment = {
        "card_id": CARD_ID,
        "amendment_type": "POST_FIT_REPORTING_BUGFIX_NO_REFIT",
        "cause": "The initial report reader looked for stream_missing inside reconstructed controls; the canonical missing mask is derived directly from official Train signal_stream aligned to saved rows.",
        "original_freeze_signature": frozen["signature"],
        "original_source_commit": frozen["identity"]["source_commit_context"],
        "amendment_script_sha256": file_sha256(Path(__file__)),
        "saved_predictor_fits_reused": results["fit_accounting"]["successful_saved_predictor_fits"],
        "additional_predictor_fits": 0,
        "verification_status": verification["status"],
        "gate_verdict": gate["status"],
        "result_artifacts": {
            "results_sha256": file_sha256(runner.RESULTS_PATH),
            "verification_sha256": file_sha256(runner.VERIFICATION_PATH),
            "gate_sha256": file_sha256(runner.GATE_PATH),
        },
        "test_or_sample_access": False,
        "kaggle_access": False,
    }
    if AMENDMENT_PATH.exists():
        existing = json.loads(AMENDMENT_PATH.read_text(encoding="utf-8"))
        if existing != amendment:
            raise BatchIntegrityError("Existing R15 amendment differs; refusing overwrite")
    else:
        write_json_exclusive(AMENDMENT_PATH, amendment)
    return {"results": results, "verification": verification, "gate": gate, "summary": summary,
            "amendment": amendment}


if __name__ == "__main__":
    print(json.dumps(verify_saved_pilot(), indent=2, allow_nan=False))
