"""Recover R21/R06 continuation bundles after the frozen-v1 audit serialization defect.

The original v1 freeze and attempts remain immutable. This recovery freezes only
the two regression candidates that can be retried inside the remaining budget;
it never changes their recipes, cutoffs, gates, data, or prediction policies.
R11 is explicitly excluded because recovering its four nine-head cutoffs would
require 36 predictor fits, above the remaining retry allowance.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import optimization_v3_expanded_benchmark as base
from src.models.optimization_v3_batch1 import (
    append_event,
    exclusive_training_slot,
    file_sha256,
    json_sha256,
    reserve_card_budget,
    write_json_exclusive,
)


PARENT_FREEZE_PATH = base.FREEZE_PATH
PARENT_PREFLIGHT_PATH = base.PREFLIGHT_PATH
CAMPAIGN_ID = "optimization_v3_expanded_benchmark_v1_serialization_recovery_v2"
RECOVERY_EXPERIMENT_DIR = ROOT / "experiments/optimization_v3/expanded_benchmark_v1_recovery_v2"
RECOVERY_REPORT_DIR = ROOT / "reports/optimization_v3/expanded_benchmark_v1_recovery_v2"
RECOVERY_RUN_DIR = ROOT / "experiments/runs/optimization_v3_expanded_benchmark_v1_recovery_v2"
FREEZE_PATH = RECOVERY_EXPERIMENT_DIR / "freeze.json"
PREFLIGHT_PATH = RECOVERY_REPORT_DIR / "preflight.json"
RUN_REPORT_PATH = RECOVERY_REPORT_DIR / "run_receipt.json"
PREVIOUS_RECOVERY_FREEZE = ROOT / "experiments/optimization_v3/expanded_benchmark_v1_recovery/freeze.json"
PREVIOUS_RECOVERY_RUN_REPORT = ROOT / "reports/optimization_v3/expanded_benchmark_v1_recovery/run_receipt.json"
RESERVATION_KEY = "EXPANDED_BENCHMARK_SERIALIZATION_RECOVERY:R21_R06"
RESERVATION_CARD = "R21+R06_SERIALIZATION_RECOVERY"
RECOVERY_CANDIDATES = ("R21", "R06")
RETRY_PREDICTOR_FITS = 8
PREVIOUS_OFFLEDGER_FULL_TRAIN_FITS = 2
GLOBAL_CAP = 180


class RecoveryError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_commit() -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "log", "-1", "--format=%H", "--",
         "scripts/optimization_v3_expanded_recovery.py"], text=True,
    ).strip()


def _current_inventory() -> tuple[dict[str, Any], dict[str, Any], dict[str, str], dict[str, str]]:
    cls_context, reg_context = base._load_contexts()
    artifacts = base._active_card_artifacts()
    sources, inputs, _ = base._source_and_input_inventory(artifacts, cls_context, reg_context)
    return cls_context, reg_context, sources, inputs


def _parent_state() -> tuple[dict[str, Any], dict[str, Any], str]:
    frozen = _read_json(PARENT_FREEZE_PATH)
    preflight = _read_json(PARENT_PREFLIGHT_PATH)
    identity = frozen.get("identity", {})
    signature = json_sha256(identity)
    if (frozen.get("signature") != signature
            or preflight.get("freeze_signature") != signature
            or frozen.get("status") != "FROZEN_BEFORE_NEW_CONTINUATION"
            or preflight.get("status") != "READY_TO_FREEZE"):
        raise RecoveryError("Parent v1 continuation freeze/preflight is not intact")
    return frozen, preflight, signature


def _class_bundle_inventory(cls_context: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    signature = _parent_state()[2]
    base.CAMPAIGN_ID = "optimization_v3_expanded_benchmark_v1"
    base.RUN_DIR = ROOT / "experiments/runs/optimization_v3_expanded_benchmark_v1"
    artifacts = base._active_card_artifacts()
    for card in ("C17", "C11"):
        records = {}
        for cutoff in base.NEW_CUTOFFS:
            frame = cls_context["frame"]
            cycles = np.asarray(cls_context["cycles"], dtype=np.int64)
            controls = base._controls_for(card, cutoff, cls_context, {})
            path = base._new_bundle_path(card, cutoff)
            if not path.is_file() or not path.with_suffix(".json").is_file():
                raise RecoveryError(f"Existing {card}/{cutoff} bundle is missing")
            record = base._load_new_bundle(card, cutoff, frame, cycles, controls, signature)
            records[str(cutoff)] = {
                "bundle_sha256": record["bundle_sha256"],
                "sidecar_sha256": record["sidecar_sha256"],
            }
        output[card] = records
    return output


def prepare() -> dict[str, Any]:
    parent_freeze, parent_preflight, parent_signature = _parent_state()
    cls_context, reg_context, sources, inputs = _current_inventory()
    parent_identity = parent_freeze["identity"]
    parent_sources = parent_identity.get("source_sha256", {})
    changed = sorted(path for path in set(parent_sources) | set(sources)
                     if parent_sources.get(path) != sources.get(path))
    allowed_change = "scripts/optimization_v3_expanded_benchmark.py"
    if changed != [allowed_change]:
        raise RecoveryError(f"Unexpected source drift versus parent freeze: {changed}")
    if inputs != parent_identity.get("input_sha256"):
        raise RecoveryError("Official Train/control/cache inputs changed versus parent freeze")
    if FREEZE_PATH.exists() or PREFLIGHT_PATH.exists():
        if not FREEZE_PATH.is_file() or not PREFLIGHT_PATH.is_file():
            raise RecoveryError("Recovery freeze/preflight pair is incomplete; refusing repair or overwrite")
        prior_freeze, prior_preflight = _read_json(FREEZE_PATH), _read_json(PREFLIGHT_PATH)
        prior_identity = prior_freeze.get("identity", {})
        prior_signature = json_sha256(prior_identity)
        if (prior_freeze.get("signature") != prior_signature
                or prior_preflight.get("freeze_signature") != prior_signature
                or prior_identity.get("source_sha256") != sources
                or prior_identity.get("input_sha256") != inputs
                or prior_identity.get("recovery_runner_sha256") != file_sha256(Path(__file__))
                or prior_identity.get("parent_freeze_sha256") != file_sha256(PARENT_FREEZE_PATH)
                or prior_identity.get("previous_recovery_freeze_sha256") != file_sha256(PREVIOUS_RECOVERY_FREEZE)
                or prior_identity.get("previous_recovery_run_report_sha256") != file_sha256(PREVIOUS_RECOVERY_RUN_REPORT)):
            raise RecoveryError("Existing recovery freeze no longer matches source/input state")
        return {"status": prior_preflight.get("status"),
                "freeze_signature": prior_signature,
                "source_head_before_freeze": prior_identity.get("source_head_before_freeze"),
                "changed_paths": changed,
                "resources_pass": prior_preflight.get("status") == "READY_TO_RUN",
                "projected_total": prior_identity.get("recovery_budget", {}).get("projected_total_with_reservation"),
                "retry_headroom_after_reservation": prior_identity.get("recovery_budget", {}).get("remaining_after_recovery_reservation")}

    estimates = parent_preflight["resource_preflight"]["by_candidate_cutoff"]
    current_resources: dict[str, dict[str, Any]] = {}
    for card in RECOVERY_CANDIDATES:
        current_resources[card] = {}
        for cutoff in base.NEW_CUTOFFS:
            check = base._runtime_resource_check(card, cutoff, estimates[card][str(cutoff)])
            current_resources[card][str(cutoff)] = check
    resources_pass = all(item["status"] == "PASS"
                         for card in current_resources.values() for item in card.values())

    ledger = base._ledger_budget()
    expected_reservation = [RETRY_PREDICTOR_FITS, RESERVATION_CARD]
    existing_reservation = ledger["reservation_keys"].get(RESERVATION_KEY)
    if ledger["reserved_predictor_fits"] == 167 and existing_reservation is None:
        reservation_mode = "NEW_RESERVATION_REQUIRED"
    elif ledger["reserved_predictor_fits"] == 175 and existing_reservation == expected_reservation:
        reservation_mode = "REUSE_EXISTING_NO_FIT_STARTED"
    else:
        raise RecoveryError(f"Retry reservation or ledger is ambiguous: {ledger}")
    previous_receipt = _read_json(PREVIOUS_RECOVERY_RUN_REPORT)
    previous_freeze = _read_json(PREVIOUS_RECOVERY_FREEZE)
    if (previous_receipt.get("status") != "RECOVERY_PARTIAL"
            or previous_receipt.get("receipts")
            or len(previous_receipt.get("errors", [])) != RETRY_PREDICTOR_FITS
            or any(error.get("error_type") != "KeyError"
                   or error.get("error") != "'resource_preflight'"
                   for error in previous_receipt.get("errors", []))
            or previous_freeze.get("signature") != json_sha256(previous_freeze.get("identity", {}))):
        raise RecoveryError("Previous recovery attempt is not the expected pre-fit path-field failure")
    prior_events = [json.loads(line) for line in base.LEDGER_PATH.read_text(encoding="utf-8").splitlines()
                    if line.strip() and json.loads(line).get("campaign_id") == previous_freeze["identity"]["campaign_id"]]
    if any(event.get("event") == "EXPANDED_CONTINUATION_FIT_STARTED" for event in prior_events):
        raise RecoveryError("Previous recovery recorded a fit start; retry reservation cannot be reused")
    projected = (ledger["reserved_predictor_fits"] + PREVIOUS_OFFLEDGER_FULL_TRAIN_FITS
                 + (RETRY_PREDICTOR_FITS if reservation_mode == "NEW_RESERVATION_REQUIRED" else 0))
    if projected > GLOBAL_CAP:
        raise RecoveryError(f"Retry reservation would exceed cap: projected {projected}/{GLOBAL_CAP}")

    class_bundles = _class_bundle_inventory(cls_context)
    identity = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "parent_campaign_id": parent_identity["campaign_id"],
        "parent_freeze_signature": parent_signature,
        "parent_freeze_sha256": file_sha256(PARENT_FREEZE_PATH),
        "parent_preflight_sha256": file_sha256(PARENT_PREFLIGHT_PATH),
        "previous_recovery_freeze_sha256": file_sha256(PREVIOUS_RECOVERY_FREEZE),
        "previous_recovery_run_report_sha256": file_sha256(PREVIOUS_RECOVERY_RUN_REPORT),
        "source_head_before_freeze": _source_commit(),
        "recovery_runner_sha256": file_sha256(Path(__file__)),
        "source_sha256": sources,
        "input_sha256": inputs,
        "source_drift_from_parent": {
            "changed_paths": changed,
            "interpretation": "NPZ bundle storage now writes mapping-valued control audits to hash-bound JSON sidecars and verifies them on read; estimator, features, preprocessing, fit rows, prediction, and gates are unchanged.",
        },
        "recovery_scope": {
            "retry_candidates": list(RECOVERY_CANDIDATES),
            "cutoffs": list(base.NEW_CUTOFFS),
            "predictor_fits": RETRY_PREDICTOR_FITS,
            "reservation_mode": reservation_mode,
            "r11_status": "BLOCKED_RETRY_BUDGET: 36 predictor fits would exceed remaining retry allowance of 11",
            "reused_valid_classification_bundles": class_bundles,
            "no_new_candidate_or_recipe": True,
        },
        "recipes": {
            card: parent_identity["active_candidate_recipes"][card]
            for card in RECOVERY_CANDIDATES
        },
        "recovery_budget": {
            "prior_reserved": ledger["reserved_predictor_fits"],
            "prior_completed_offledger_full_train": PREVIOUS_OFFLEDGER_FULL_TRAIN_FITS,
            "retry_reservation": RETRY_PREDICTOR_FITS,
            "reservation_mode": reservation_mode,
            "projected_total_with_reservation": projected,
            "global_cap": GLOBAL_CAP,
            "remaining_after_recovery_reservation": GLOBAL_CAP - projected,
            "reservation_key": RESERVATION_KEY,
        },
        "resources": {
            "single_training_job": True,
            "threads": base.THREADS,
            "job_memory_cap_bytes": base.JOB_CAP_BYTES,
            "minimum_free_memory_reserve_bytes": base.FREE_RAM_RESERVE_BYTES,
            "fresh_preflight": current_resources,
        },
        "official_train": {
            "rows": 50_608,
            "classification_sha256": inputs["classification_controls:data/raw/classification/Train.csv"],
            "regression_input_inventory_sha256": json_sha256({
                key: value for key, value in inputs.items() if key.startswith("regression_controls:")
            }),
            "test_or_sample_submission_opened": False,
        },
        "test_data_used": False,
        "sample_submission_used": False,
        "public_score_used": False,
    }
    signature = json_sha256(identity)
    preflight = {
        "schema_version": 1,
        "status": "READY_TO_RUN" if resources_pass else "RESOURCE_BLOCKED",
        "freeze_signature": signature,
        "created_at_utc": base._utc_now(),
        "parent_freeze_signature": parent_signature,
        "resources": current_resources,
        "fit_budget_snapshot": ledger,
        "reservation": {"budget_key": RESERVATION_KEY,
                        "card_id": RESERVATION_CARD,
                        "predictor_fits": RETRY_PREDICTOR_FITS,
                        "reserved": reservation_mode == "REUSE_EXISTING_NO_FIT_STARTED",
                        "mode": reservation_mode},
        "resource_preflight": {
            "by_candidate_cutoff": {
                card: estimates[card] for card in RECOVERY_CANDIDATES
            }
        },
        "class_bundles_verified_and_reused": class_bundles,
        "no_scores_or_test_access": True,
    }
    if FREEZE_PATH.exists() or PREFLIGHT_PATH.exists():
        existing = _read_json(FREEZE_PATH) if FREEZE_PATH.exists() else {}
        if existing.get("identity") != identity or existing.get("signature") != signature:
            raise RecoveryError("Recovery freeze already exists with a different identity; refusing overwrite")
        prior_preflight = _read_json(PREFLIGHT_PATH) if PREFLIGHT_PATH.exists() else {}
        if prior_preflight.get("freeze_signature") != signature:
            raise RecoveryError("Recovery preflight already exists with a different identity")
    else:
        RECOVERY_EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
        RECOVERY_REPORT_DIR.mkdir(parents=True, exist_ok=True)
        write_json_exclusive(FREEZE_PATH, {
            "status": "FROZEN_BEFORE_RECOVERY_FITS",
            "identity": identity,
            "signature": signature,
        })
        write_json_exclusive(PREFLIGHT_PATH, preflight)
    return {"status": preflight["status"], "freeze_signature": signature,
            "source_head_before_freeze": identity["source_head_before_freeze"],
            "changed_paths": changed, "resources_pass": resources_pass,
            "projected_total": projected, "retry_headroom_after_reservation": GLOBAL_CAP - projected}


def _verify_recovery_freeze() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    frozen, preflight = _read_json(FREEZE_PATH), _read_json(PREFLIGHT_PATH)
    identity = frozen.get("identity", {})
    signature = json_sha256(identity)
    if (frozen.get("signature") != signature or preflight.get("freeze_signature") != signature
            or frozen.get("status") != "FROZEN_BEFORE_RECOVERY_FITS"
            or preflight.get("status") != "READY_TO_RUN"):
        raise RecoveryError("Committed recovery freeze/preflight is invalid or resource-blocked")
    cls_context, reg_context, sources, inputs = _current_inventory()
    if sources != identity.get("source_sha256") or inputs != identity.get("input_sha256"):
        raise RecoveryError("Recovery source/input hashes differ from frozen identity")
    if file_sha256(Path(__file__)) != identity.get("recovery_runner_sha256"):
        raise RecoveryError("Recovery runner source differs from the committed freeze")
    if _source_commit() != identity.get("source_head_before_freeze"):
        raise RecoveryError("Committed recovery source revision differs from the frozen code commit")
    if file_sha256(PARENT_FREEZE_PATH) != identity.get("parent_freeze_sha256"):
        raise RecoveryError("Parent v1 freeze changed after recovery freeze")
    if (file_sha256(PREVIOUS_RECOVERY_FREEZE) != identity.get("previous_recovery_freeze_sha256")
            or file_sha256(PREVIOUS_RECOVERY_RUN_REPORT) != identity.get("previous_recovery_run_report_sha256")):
        raise RecoveryError("Previous pre-fit recovery evidence changed after recovery freeze")
    return identity, cls_context, reg_context, preflight


def run() -> dict[str, Any]:
    identity, cls_context, reg_context, preflight = _verify_recovery_freeze()
    with exclusive_training_slot(base.TRAINING_LOCK):
        return _run_locked(identity, cls_context, reg_context, preflight)


def _run_locked(identity: Mapping[str, Any], cls_context: Mapping[str, Any],
                reg_context: Mapping[str, Any], preflight: Mapping[str, Any]) -> dict[str, Any]:
    signature = json_sha256(identity)
    current = base._ledger_budget()
    existing = current["reservation_keys"].get(RESERVATION_KEY)
    expected = [RETRY_PREDICTOR_FITS, RESERVATION_CARD]
    if current["reserved_predictor_fits"] == 167 and existing is None:
        reserve = reserve_card_budget(
            base.LEDGER_PATH, budget_key=RESERVATION_KEY,
            card_id=RESERVATION_CARD, predictor_fits=RETRY_PREDICTOR_FITS,
            cap=GLOBAL_CAP,
        )
    elif current["reserved_predictor_fits"] == 175 and existing == expected:
        reserve = {"status": "IDEMPOTENT_EXISTING_RESERVATION",
                   "budget_key": RESERVATION_KEY, "predictor_fits": RETRY_PREDICTOR_FITS}
    else:
        raise RecoveryError(f"Recovery budget ledger is ambiguous: {current}")
    budget = base._ledger_budget(expected_reserved=175)
    if budget["reserved_predictor_fits"] + PREVIOUS_OFFLEDGER_FULL_TRAIN_FITS > GLOBAL_CAP:
        raise RecoveryError("Recovery plus prior full-Train fits exceeds the global cap")

    original = {
        "CAMPAIGN_ID": base.CAMPAIGN_ID, "RUN_DIR": base.RUN_DIR,
        "REPORT_DIR": base.REPORT_DIR, "PREFLIGHT_PATH": base.PREFLIGHT_PATH,
    }
    base.CAMPAIGN_ID = CAMPAIGN_ID
    base.RUN_DIR = RECOVERY_RUN_DIR
    base.REPORT_DIR = RECOVERY_REPORT_DIR
    base.PREFLIGHT_PATH = PREFLIGHT_PATH
    receipts: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    start = {"event": "EXPANDED_RECOVERY_EXECUTION_STARTED",
             "at_utc": base._utc_now(), "campaign_id": CAMPAIGN_ID,
             "freeze_signature": signature, "reservation": reserve,
             "budget_after_reservation": budget}
    append_event(base.LEDGER_PATH, start)
    RECOVERY_RUN_DIR.mkdir(parents=True, exist_ok=True)
    try:
        for card in RECOVERY_CANDIDATES:
            for cutoff in base.NEW_CUTOFFS:
                frame_context = reg_context
                frame = frame_context["frame"]
                cycles = np.asarray(frame_context["cycles"], dtype=np.int64)
                controls = base._controls_for(card, cutoff, cls_context, reg_context)
                path = base._new_bundle_path(card, cutoff)
                try:
                    if path.exists() or path.with_suffix(".json").exists():
                        record = base._load_new_bundle(card, cutoff, frame, cycles, controls, signature)
                        receipt = {"status": "VERIFIED_REUSE", "card_id": card,
                                   "cutoff": cutoff, "predictor_fits_saved": 0,
                                   "bundle_sha256": record["bundle_sha256"],
                                   "sidecar_sha256": record["sidecar_sha256"]}
                    else:
                        receipt = base._run_one_fit(card, cutoff, cls_context, reg_context,
                                                    signature, attempt_number=1)
                        receipt["status"] = "FIT_SAVED"
                    receipts.append(receipt)
                except Exception as exc:
                    error = {"status": "FIT_OR_INTEGRITY_BLOCKED", "card_id": card,
                             "cutoff": cutoff, "error_type": type(exc).__name__,
                             "error": str(exc)[:1600], "at_utc": base._utc_now()}
                    errors.append(error)
                    append_event(base.LEDGER_PATH, {
                        "event": "EXPANDED_CONTINUATION_FIT_FAILED", "campaign_id": CAMPAIGN_ID,
                        "freeze_signature": signature, **error,
                    })
                    attempt_path = RECOVERY_REPORT_DIR / card / "attempts.jsonl"
                    attempt_path.parent.mkdir(parents=True, exist_ok=True)
                    attempt_path.open("a", encoding="utf-8", newline="\n").write(
                        json.dumps(error, sort_keys=True, allow_nan=False) + "\n")
        finished = {"event": "EXPANDED_RECOVERY_EXECUTION_FINISHED",
                    "at_utc": base._utc_now(), "campaign_id": CAMPAIGN_ID,
                    "freeze_signature": signature, "receipts": receipts,
                    "errors": errors, "budget_after_fits": base._ledger_budget(expected_reserved=175)}
        append_event(base.LEDGER_PATH, finished)
    finally:
        base.CAMPAIGN_ID = original["CAMPAIGN_ID"]
        base.RUN_DIR = original["RUN_DIR"]
        base.REPORT_DIR = original["REPORT_DIR"]
        base.PREFLIGHT_PATH = original["PREFLIGHT_PATH"]
    output = {"status": "RECOVERY_FITS_COMPLETE" if not errors else "RECOVERY_PARTIAL",
              "freeze_signature": signature, "receipts": receipts,
              "errors": errors, "budget_after_fits": finished["budget_after_fits"]}
    write_json_exclusive(RUN_REPORT_PATH, output)
    return output


def evaluate() -> dict[str, Any]:
    identity, cls_context, reg_context, _preflight = _verify_recovery_freeze()
    run_report = _read_json(RUN_REPORT_PATH)
    if run_report.get("status") != "RECOVERY_FITS_COMPLETE":
        raise RecoveryError("Do not open scores until all R21/R06 recovery bundles are complete")
    recovery_signature = json_sha256(identity)
    parent_freeze, _parent_preflight, parent_signature = _parent_state()
    parent_run = ROOT / "experiments/runs/optimization_v3_expanded_benchmark_v1"
    artifacts = base._active_card_artifacts()
    results: dict[str, Any] = {}
    verifications: dict[str, Any] = {}
    candidate_freezes = {
        "C17": ("optimization_v3_expanded_benchmark_v1", parent_run, parent_signature),
        "C11": ("optimization_v3_expanded_benchmark_v1", parent_run, parent_signature),
        "R21": (CAMPAIGN_ID, RECOVERY_RUN_DIR, recovery_signature),
        "R06": (CAMPAIGN_ID, RECOVERY_RUN_DIR, recovery_signature),
    }
    for card, (campaign_id, run_dir, freeze_signature) in candidate_freezes.items():
        base.CAMPAIGN_ID, base.RUN_DIR = campaign_id, run_dir
        is_classification = card in {"C17", "C11"}
        context = cls_context if is_classification else reg_context
        records = base._records_for_candidate(card, cls_context, reg_context,
                                              artifacts, freeze_signature)
        origins = []
        for cutoff, record in records.items():
            metadata = record["metadata"]
            state = metadata.get("state", {})
            origins.append({
                "cutoff": int(cutoff),
                "fit_rows": int(metadata.get("fit_rows", state.get("fit_rows", 0))),
                "score_rows": int(len(record["score_rows"])),
                "predictor_fits": int(metadata.get("fit_predictor_count", 0)),
                "bundle_sha256": record.get("bundle_sha256"),
                "sidecar_sha256": record.get("sidecar_sha256"),
                "reused_pilot": cutoff in base.PILOT_CUTOFFS,
                "test_data_used": False,
            })
        if is_classification:
            candidate_result, verified = base._classification_candidate_report(
                card, context["frame"], np.asarray(context["target"], dtype=np.int64),
                records, freeze_signature,
            )
            candidate_result["late_W5_W6"] = candidate_result["qualification"]["late_W5_W6"]
        else:
            from scripts import optimization_v3_final_continuation_r22 as reg_report
            cycles = np.asarray(context["cycles"], dtype=np.int64)
            views = base._regression_views(context["frame"], cycles, records)
            metrics = reg_report._policy_metrics(
                context["frame"], np.asarray(context["target"], dtype=np.float64),
                views, "candidate",
            )
            qualification = reg_report._qualification(metrics)
            candidate_result = {
                "card_id": card, "candidate": card, "task": "regression",
                "status": "CANDIDATE_METRICS_READY_FOR_INDEPENDENT_QA",
                "freeze_signature": freeze_signature, "views": metrics["views"],
                "late_W5_W6": metrics["late_W5_W6"],
                "diagnostics": metrics["diagnostics"], "groups": metrics["groups"],
                "qualification": {**qualification,
                                  "status": "QUALIFIED" if qualification["qualified"]
                                  else "EXPERIMENTAL_GATE_EXCEPTION"},
                "metric_formula": "mean(abs(threat_rank - prediction)); gain = champion MAE - candidate MAE",
                "outcome_reuse_warning": "Train temporal-development labels were previously viewed; not an untouched holdout.",
                "test_data_used": False, "sample_submission_read": False,
            }
            verified = base._verify_regression_metrics_independently(
                context["frame"], np.asarray(context["target"], dtype=np.float64),
                views, records, candidate_result,
            )
        candidate_result["fit_origin_receipts"] = origins
        candidate_result["continuation_fit_accounting"] = {
            "new_fit_cutoffs": list(base.NEW_CUTOFFS),
            "reused_pilot_cutoffs": list(base.PILOT_CUTOFFS),
            "new_predictor_fits": 4, "test_data_used": False,
        }
        verified["candidate"] = card
        results[card], verifications[card] = candidate_result, verified
        target_dir = RECOVERY_REPORT_DIR / card
        target_dir.mkdir(parents=True, exist_ok=True)
        base._write_or_match(target_dir / "results.json", candidate_result)
        base._write_or_match(target_dir / "verification.json", verified)

    r11_blocker = {
        "candidate": "R11", "status": "INTEGRITY_BLOCKED_NOT_EVALUATED",
        "reason": "All four original continuation fits executed, but each bundle was rejected because a mapping-valued diagnostic control was sent to NPZ as object dtype. Repair would require 36 predictor-fit retries, exceeding the frozen retry headroom of 11.",
        "scores_opened": False, "recipe_or_gate_changed": False,
    }
    base._write_or_match(RECOVERY_REPORT_DIR / "R11_BLOCKER.json", r11_blocker)
    campaign = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "parent_campaign_freeze_signature": parent_signature,
        "recovery_freeze_signature": recovery_signature,
        "status": "FOUR_CANDIDATES_EVALUATED_R11_INTEGRITY_BLOCKED",
        "evaluated_candidates": list(candidate_freezes),
        "blocked_candidates": {"R11": r11_blocker},
        "candidates": {
            card: {
                "qualification": results[card]["qualification"]["status"],
                "primary": results[card]["views"]["primary"],
                "H12": results[card]["views"]["H12"],
                "late_W5_W6": results[card].get("late_W5_W6", {}),
                "qualification_gates": results[card]["qualification"]["gates"],
                "verification_status": verifications[card].get("status", "PASS"),
                "result_sha256": json_sha256(results[card]),
                "verification_sha256": json_sha256(verifications[card]),
            }
            for card in candidate_freezes
        },
        "test_data_used": False,
        "public_score_used": False,
    }
    base._write_or_match(RECOVERY_REPORT_DIR / "CAMPAIGN_REPORT.json", campaign)
    return campaign


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "evaluate"))
    args = parser.parse_args()
    result = prepare() if args.mode == "prepare" else run() if args.mode == "run" else evaluate()
    print(json.dumps(base._json_clean(result), sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
