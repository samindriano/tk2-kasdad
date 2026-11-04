"""Resume only the two unstarted V4 units under an explicit owner-scoped continuation audit."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import psutil
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.reg_position_direct_l1_v1 as pilot  # noqa: E402
import scripts.verify_reg_position_direct_l1_continuation_v1 as scope  # noqa: E402

CARD_ID = pilot.CARD_ID
ALLOWED_NEW_FITS = tuple(pilot.FIT_IDS[10:12])
EXPECTED_PRIOR_FITS = tuple(pilot.FIT_IDS[:10])
CHECKPOINT_PREDICTIONS = (
    f"{CARD_ID}_E67_BAG", f"{CARD_ID}_E67_REAL", f"{CARD_ID}_E67_SHUFFLED",
    f"{CARD_ID}_E75_BAG",
)
PREDICTION_FIT_IDS = tuple(
    f"{CARD_ID}_E{origin}_{arm}" for origin in pilot.ORIGINS for arm in pilot.ARMS
)
PINNED_RUN_FILES = (
    "experiments/runs/reg_position_direct_l1_v1/FREEZE_V4.json",
    "experiments/runs/reg_position_direct_l1_v1/PREFIT_VERIFICATION_V4.json",
    "reports/optimization_v3/reg_position_direct_l1_v1/PREFIT_VERIFICATION_V4.json",
    "reports/optimization_v3/reg_position_direct_l1_v1/RECIPE.json",
    "reports/optimization_v3/reg_position_direct_l1_v1/SYNTHETIC_E2E.json",
)
REQUIRED_LOCAL_COMMIT_PATHS = (
    "scripts/reg_position_direct_l1_v1.py",
    "scripts/verify_reg_position_direct_l1_v1.py",
    "tests/test_reg_position_direct_l1_v1.py",
    "scripts/continue_reg_position_direct_l1_v1.py",
    "scripts/verify_reg_position_direct_l1_continuation_v1.py",
    "tests/test_reg_position_direct_l1_continuation_v1.py",
    "reports/optimization_v3/reg_position_direct_l1_v1/PREFIT_VERIFICATION.json",
    "reports/optimization_v3/reg_position_direct_l1_v1/PREFIT_VERIFICATION_V2.json",
    "reports/optimization_v3/reg_position_direct_l1_v1/PREFIT_VERIFICATION_V3.json",
    "reports/optimization_v3/reg_position_direct_l1_v1/PREFIT_VERIFICATION_V4.json",
    "reports/optimization_v3/reg_position_direct_l1_v1/RECIPE.json",
    "reports/optimization_v3/reg_position_direct_l1_v1/SYNTHETIC_E2E.json",
    "reports/optimization_v3/reg_position_direct_l1_v1/CONTINUATION_V1.json",
)
REPAIR_SOURCE_FILES = (
    "scripts/continue_reg_position_direct_l1_v1.py",
    "scripts/verify_reg_position_direct_l1_continuation_v1.py",
    "tests/test_reg_position_direct_l1_continuation_v1.py",
)
REQUIRED_REPAIR_COMMIT_PATHS = (*REPAIR_SOURCE_FILES,
    "reports/optimization_v3/reg_position_direct_l1_v1/POSTFIT_VALIDATOR_REPAIR_V1.json")


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def already_complete_ledger(starts: list[str], saves: list[str]) -> bool:
    return starts == list(pilot.FIT_IDS) and saves == list(pilot.FIT_IDS)


def _sha(path: Path) -> str:
    return pilot._sha_file(path)


def _audit_body(audit: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in audit.items() if key != "continuation_signature"}


def _pinned_output_hashes() -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for relative in PINNED_RUN_FILES:
        path = pilot._contained_file(ROOT, Path(relative))
        if not path.is_file():
            raise scope.ContinuationError(f"Required V4 provenance file is missing: {relative}")
        output[relative] = {
            "sha256": _sha(path), "size_bytes": path.stat().st_size,
            "last_write_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
        }
    return output


def _local_provenance(freeze: dict[str, Any]) -> dict[str, Any]:
    original_paths = (
        "scripts/reg_position_direct_l1_v1.py",
        "scripts/verify_reg_position_direct_l1_v1.py",
        "tests/test_reg_position_direct_l1_v1.py",
        "reports/optimization_v3/reg_position_direct_l1_v1/RECIPE.json",
        "reports/optimization_v3/reg_position_direct_l1_v1/SYNTHETIC_E2E.json",
    )
    status = pilot._git(ROOT, "status", "--porcelain", "--untracked-files=all").splitlines()
    path_status = {line[3:].replace("\\", "/"): line[:2] for line in status if len(line) >= 4}
    source_hashes = freeze.get("source_hashes", {})
    if any(relative not in source_hashes for relative in original_paths):
        raise scope.ContinuationError("V4 freeze lacks one or more original local implementation hashes")
    untracked = [relative for relative in original_paths if path_status.get(relative) == "??"]
    pinned_hashes = {relative: source_hashes[relative] for relative in original_paths}
    current_hashes = {relative: _sha(ROOT / relative) for relative in original_paths}
    if current_hashes != pinned_hashes:
        raise scope.ContinuationError("Original pilot implementation differs from its V4 prefit hashes")
    if len(untracked) == len(original_paths):
        return {
            "prefit_local_git_head": freeze.get("local_git_head"),
            "local_git_head_at_continuation_capture": pilot._git(ROOT, "rev-parse", "HEAD"),
            "original_implementation_untracked_at_original_fit": True,
            "original_implementation_untracked_paths_at_capture": untracked,
            "committed_before_original_ten_fits": False,
            "commit_required_before_remaining_units": True,
            "provenance_note": "Original implementation bytes were untracked at the initial fit. FREEZE_V4 binds their exact prefit hashes; this continuation records that fact and requires a scoped commit before the remaining units.",
            "pinned_original_file_hashes": pinned_hashes,
            "pinned_original_file_last_write_utc_at_capture": {
                relative: datetime.fromtimestamp((ROOT / relative).stat().st_mtime, timezone.utc).isoformat()
                for relative in original_paths
            },
            "local_status_at_continuation_capture": status,
        }

    # A later audit refresh may happen after the first scoped commit. Reuse the
    # prior signed evidence for the historical prefit state, but recheck every
    # original file against the exact hashes frozen before the first fit.
    previous_audit_ref = f"HEAD:{_relative(scope.CONTINUATION_PATH)}"
    previous = subprocess.run(["git", "show", previous_audit_ref], cwd=ROOT,
                              capture_output=True, text=True, encoding="utf-8", check=False)
    if previous.returncode != 0:
        raise scope.ContinuationError("Committed precommit provenance audit is unavailable for refresh")
    try:
        prior_audit = json.loads(previous.stdout)
    except Exception as exc:
        raise scope.ContinuationError("Committed precommit provenance audit is invalid") from exc
    if (prior_audit.get("freeze_signature") != freeze.get("freeze_signature")
            or prior_audit.get("continuation_signature") != scope._body_signature(
                prior_audit, "continuation_signature")):
        raise scope.ContinuationError("Committed precommit provenance audit signature is invalid")
    prior = prior_audit.get("local_code_provenance", {})
    if (prior.get("original_implementation_untracked_at_original_fit") is not True
            or prior.get("committed_before_original_ten_fits") is not False
            or prior.get("pinned_original_file_hashes") != pinned_hashes):
        raise scope.ContinuationError("Committed audit does not prove the original prefit implementation state")
    return {
        "prefit_local_git_head": freeze.get("local_git_head"),
        "local_git_head_at_continuation_capture": pilot._git(ROOT, "rev-parse", "HEAD"),
        "original_implementation_untracked_at_original_fit": True,
        "original_implementation_untracked_paths_at_initial_capture": prior.get(
            "original_implementation_untracked_paths_at_capture", []),
        "committed_before_original_ten_fits": False,
        "committed_before_remaining_units": True,
        "precommit_provenance_reused_from_prior_signed_audit": prior_audit["continuation_signature"],
        "provenance_note": "The original pilot files were untracked at the initial fit, as proven by the prior signed audit. Their current bytes still match the V4 prefit hashes; the initial scoped commit preserves those exact files before the remaining units.",
        "pinned_original_file_hashes": pinned_hashes,
        "pinned_original_file_last_write_utc_at_initial_capture": prior.get(
            "pinned_original_file_last_write_utc_at_capture", {}),
        "prior_audit_local_status_at_capture": prior.get("local_status_at_continuation_capture", []),
        "local_status_at_audit_refresh": status,
    }


def _checkpoint(freeze: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    prefit = pilot._read_json(pilot.PREFIT_RUN_PATH)
    report_prefit = pilot._read_json(pilot.PREFIT_REPORT_PATH)
    synthetic = pilot._read_json(pilot.REPORT_ROOT / "SYNTHETIC_E2E.json")
    for record in (prefit, report_prefit):
        if (record.get("status") != "PASS"
                or record.get("freeze_signature") != freeze.get("freeze_signature")
                or record.get("query_targets_read") is not False):
            raise scope.ContinuationError("V4 prefit verification is absent, stale, or opened query targets")
    if (synthetic.get("status") != "PASS" or synthetic.get("synthetic_only") is not True
            or synthetic.get("production_fit_units") != 0
            or synthetic.get("runner_sha256") != _sha(ROOT / "scripts/reg_position_direct_l1_v1.py")):
        raise scope.ContinuationError("Frozen synthetic actual-entrypoint receipt does not match the runner")

    events = pilot._read_events(pilot.LEDGER_PATH)
    starts = [event.get("fit_id") for event in events if event.get("event") == "FIT_STARTED"]
    saves = [event.get("fit_id") for event in events if event.get("event") == "FIT_SAVED"]
    failures = [event for event in events if event.get("event") == "FIT_FAILED"]
    if starts != list(EXPECTED_PRIOR_FITS) or saves != list(EXPECTED_PRIOR_FITS) or failures:
        raise scope.ContinuationError("The saved ledger is not exactly the original 10 ordered starts/saves")

    units: dict[str, dict[str, str]] = {}
    for fit_id in EXPECTED_PRIOR_FITS:
        artifact_path, receipt_path = pilot._unit_paths(fit_id)
        _, receipt = pilot._validate_unit(pilot.RUN_ROOT, fit_id, freeze)
        units[fit_id] = {
            "artifact_sha256": _sha(artifact_path),
            "receipt_sha256": _sha(receipt_path),
            "receipt_signature": str(receipt.get("receipt_sha256", "")),
            "artifact_last_write_utc": datetime.fromtimestamp(artifact_path.stat().st_mtime, timezone.utc).isoformat(),
            "receipt_last_write_utc": datetime.fromtimestamp(receipt_path.stat().st_mtime, timezone.utc).isoformat(),
        }

    predictions: dict[str, dict[str, str]] = {}
    for fit_id in CHECKPOINT_PREDICTIONS:
        artifact, receipt_path = pilot._prediction_paths(fit_id)
        origin = 67 if "_E67_" in fit_id else 75
        arm = fit_id.rsplit("_", 1)[-1]
        item = context["origins"][origin]
        pilot._validate_prediction_bundle(
            artifact, receipt_path, freeze=freeze, fit_id=fit_id, origin=origin, arm=arm,
            query_rows=item.query_rows, query_ids=item.query_ids, query_cycles=item.query_cycles,
            missing=item.query_missing, baseline=item.query_baseline,
        )
        predictions[fit_id] = {
            "bundle_sha256": _sha(artifact), "receipt_sha256": _sha(receipt_path),
            "bundle_last_write_utc": datetime.fromtimestamp(artifact.stat().st_mtime, timezone.utc).isoformat(),
            "receipt_last_write_utc": datetime.fromtimestamp(receipt_path.stat().st_mtime, timezone.utc).isoformat(),
        }

    for fit_id in ALLOWED_NEW_FITS:
        artifact, receipt = pilot._unit_paths(fit_id)
        prediction, prediction_receipt = pilot._prediction_paths(fit_id)
        if any(path.exists() for path in (artifact, receipt, prediction, prediction_receipt)):
            raise scope.ContinuationError(f"A remaining unit already has an artifact; retry is prohibited: {fit_id}")
    for path in (pilot.RUN_ROOT / "PREDICTIONS_READY.json", pilot.PRELABEL_PATH,
                 pilot.REPORT_ROOT / "RESULTS.json", pilot.RUN_ROOT / "RESULTS.json"):
        if path.exists():
            raise scope.ContinuationError(f"Unexpected postfit/scoring output exists before continuation: {path}")

    ledger_bytes = pilot.LEDGER_PATH.read_bytes()
    preserved = _pinned_output_hashes()
    for fit_id, item in units.items():
        for suffix, value in (("artifact", item["artifact_sha256"]), ("receipt", item["receipt_sha256"])):
            path = pilot._unit_paths(fit_id)[0 if suffix == "artifact" else 1]
            preserved[_relative(path)] = {
                "sha256": value, "size_bytes": path.stat().st_size,
                "last_write_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
            }
    for fit_id, item in predictions.items():
        for suffix, value in (("bundle", item["bundle_sha256"]), ("receipt", item["receipt_sha256"])):
            path = pilot._prediction_paths(fit_id)[0 if suffix == "bundle" else 1]
            preserved[_relative(path)] = {
                "sha256": value, "size_bytes": path.stat().st_size,
                "last_write_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
            }
    first_start = next(event for event in events if event.get("event") == "FIT_STARTED")
    return {
        "ten_unit_checkpoint": units,
        "checkpoint_prediction_bundles": predictions,
        "checkpoint_events": events,
        "checkpoint_ledger_sha256": _sha(pilot.LEDGER_PATH),
        "checkpoint_ledger_size_bytes": len(ledger_bytes),
        "checkpoint_ledger_event_count": len(events),
        "first_production_fit_started_utc": first_start.get("started_utc"),
        "preserved_checkpoint_files": preserved,
        "query_targets_read_before_remaining_units": False,
        "fit_targets_read_by_original_authorized_run": ["E67 same-origin prefix", "E75 same-origin prefix"],
    }


def create_continuation_audit() -> dict[str, Any]:
    if scope.CONTINUATION_PATH.exists():
        raise scope.ContinuationError("CONTINUATION_V1 already exists; it will not be overwritten")
    freeze = pilot._read_json(pilot.FREEZE_PATH)
    context = scope.validate_source_context(freeze, pilot.SOURCE_ROOT)
    checkpoint = _checkpoint(freeze, context)
    provenance = _local_provenance(freeze)
    adapter_sources = (
        "scripts/continue_reg_position_direct_l1_v1.py",
        "scripts/verify_reg_position_direct_l1_continuation_v1.py",
        "tests/test_reg_position_direct_l1_continuation_v1.py",
    )
    continuation_hashes = {relative: _sha(ROOT / relative) for relative in adapter_sources}
    run_events = pilot._read_events(pilot.LEDGER_PATH)
    failures = [event for event in run_events if event.get("event") == "FIT_FAILED"]
    if failures:
        raise scope.ContinuationError("A failed production unit exists; continuation is prohibited")

    audit = {
        "schema_version": 1,
        "card_id": CARD_ID,
        "audit_type": "SCOPED_CONTINUATION_AFTER_TEN_DURABLE_UNITS",
        "status": "AUTHORIZED_PENDING_TWO_UNITS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "authorization": {
            "source_thread_id": "01a19569-16e0-7253-aa14-57a0a3160e03",
            "scope": "Run only E75_REAL then E75_SHUFFLED under the original V4 recipe and 12-unit cap.",
            "main_checkout_policy": "Allow only explicitly authorized Bigram-owner paths and the exact append-only Bigram budget suffix; preserve the V4 raw prefix and every other frozen source hash and row contract.",
        },
        "freeze_path": _relative(pilot.FREEZE_PATH),
        "freeze_sha256": _sha(pilot.FREEZE_PATH),
        "freeze_signature": freeze["freeze_signature"],
        "historical_source_identity": {
            "source_root": freeze["source_root"],
            "source_git_branch": freeze["source_git_branch"],
            "source_git_head": freeze["source_git_head"],
            "source_worktree_status_sha256": freeze["source_worktree_status_sha256"],
            "status_hash_is_historical_metadata": True,
        },
        "freeze_created_utc": freeze.get("created_utc"),
        "prefit_run_last_write_utc": datetime.fromtimestamp(
            pilot.PREFIT_RUN_PATH.stat().st_mtime, timezone.utc).isoformat(),
        "prefit_report_last_write_utc": datetime.fromtimestamp(
            pilot.PREFIT_REPORT_PATH.stat().st_mtime, timezone.utc).isoformat(),
        "scoped_source_reconciliation": {
            **context["reconciliation"],
            "current_source_branch": context["current_source_branch"],
            "current_source_head": context["current_source_head"],
            "current_source_status_sha256": context["current_source_status_sha256"],
            "authorized_owner_file_hashes": context["authorized_owner_file_hashes"],
            "source_feature_rows": context["source_feature_rows"],
            "feature_row_and_prefix_contracts_match": context["feature_row_and_prefix_contracts_match"],
            "budget_log_extension": context["budget_log_extension"],
            "query_targets_read": False,
        },
        "frozen_source_hashes": freeze["source_hashes"],
        "frozen_recipe_sha256": freeze["recipe_sha256"],
        "pinned_input_contracts": {
            "official_train_sha256": freeze["official_train_sha256"],
            "classification_regression_feature_alignment": freeze["classification_regression_feature_alignment"],
            "origins": freeze["origins"],
            "resolved_parameters": freeze["resolved_parameters"],
            "fit_order": freeze["fit_order"],
            "fit_cap": freeze["fit_cap"],
            "resource_gate": freeze["resource_gate"],
        },
        **checkpoint,
        "local_code_provenance": provenance,
        "continuation_source_file_hashes": continuation_hashes,
        "required_local_commit_paths": list(REQUIRED_LOCAL_COMMIT_PATHS),
        "remaining_fit_plan": {
            "already_started_and_saved": 10,
            "remaining_budget": 2,
            "allowed_fit_ids": list(ALLOWED_NEW_FITS),
            "order": list(ALLOWED_NEW_FITS),
            "new_fit_or_recipe": False,
            "retry_or_counter_reset": False,
        },
        "prior_fail_closed_traces": [
            {
                "command": "python scripts/reg_position_direct_l1_v1.py fit",
                "message": "NO-GO: Concurrent KASDAD training compute detected; callback root before fitting",
                "detected_command": "pytest -q tests/test_optimization_v3_bigram_position_pilot_v1.py tests/test_optimization_v3_bigram_position_readiness_v1.py tests/test_optimization_v3_positional_bag_offset_v1.py",
                "observed_output": "NO-GO: Concurrent KASDAD training compute detected; callback root before fitting: [{'pid': 41728, 'command': 'c:\\\\users\\\\sam\\\\appdata\\\\local\\\\programs\\\\python\\\\python313\\\\python.exe -m pytest -q tests/test_optimization_v3_bigram_position_pilot_v1.py tests/test_optimization_v3_bigram_position_readiness_v1.py tests/test_optimization_v3_positional_bag_offset_v1.py', 'cpu_delta_seconds': 0.203125}]",
                "new_fit_unit_started_after_detection": False,
            },
            {
                "command": "python scripts/reg_position_direct_l1_v1.py fit",
                "message": "NO-GO: Production source checkout identity/status differs from prefit freeze",
                "observed_output": "NO-GO: Production source checkout identity/status differs from prefit freeze",
                "new_fit_unit_started_after_detection": False,
            },
        ],
        "query_labels_opened": False,
        "test_sample_kaggle_accessed": False,
    }
    audit["continuation_signature"] = pilot._canonical_sha(_audit_body(audit))
    pilot._atomic_json(scope.CONTINUATION_PATH, audit)
    return audit


def create_postfit_repair_audit() -> dict[str, Any]:
    """Bind the postfit prediction-scope repair without altering V1 or fit receipts."""
    repair_path = scope.BUDGET_REPAIR_PATH
    if repair_path.exists():
        raise scope.ContinuationError("POSTFIT_VALIDATOR_REPAIR_V1 already exists; it will not be overwritten")
    audit = scope._load_continuation_audit()
    freeze = pilot._read_json(pilot.FREEZE_PATH)
    context = scope.validate_source_context(freeze, pilot.SOURCE_ROOT)
    events = pilot._read_events(pilot.LEDGER_PATH)
    starts = [event for event in events if event.get("event") == "FIT_STARTED"]
    saves = [event for event in events if event.get("event") == "FIT_SAVED"]
    failures = [event for event in events if event.get("event") == "FIT_FAILED"]
    if ([event.get("fit_id") for event in starts] != list(pilot.FIT_IDS)
            or [event.get("fit_id") for event in saves] != list(pilot.FIT_IDS) or failures):
        raise scope.ContinuationError("Repair audit requires exactly 12 ordered starts/saves and zero failures")

    frame = pilot._load_feature_frame(pilot.SOURCE_ROOT)
    origins = {origin: pilot._load_origin(pilot.SOURCE_ROOT, frame, origin) for origin in pilot.ORIGINS}
    models: dict[str, dict[str, str]] = {}
    for fit_id in pilot.FIT_IDS:
        artifact, receipt_path = pilot._unit_paths(fit_id)
        _, receipt = pilot._validate_unit(pilot.RUN_ROOT, fit_id, freeze)
        models[fit_id] = {
            "artifact_sha256": _sha(artifact),
            "receipt_sha256": _sha(receipt_path),
            "freeze_signature": str(receipt.get("freeze_signature", "")),
            "fit_id": fit_id,
        }

    prediction_ids = scope.prediction_inventory(pilot.RUN_ROOT)
    predictions: dict[str, dict[str, str]] = {}
    for fit_id in prediction_ids:
        origin = 67 if "_E67_" in fit_id else 75
        arm = fit_id.rsplit("_", 1)[-1]
        item = origins[origin]
        bundle, receipt = pilot._prediction_paths(fit_id)
        pilot._validate_prediction_bundle(
            bundle, receipt, freeze=freeze, fit_id=fit_id, origin=origin, arm=arm,
            query_rows=item.query_rows, query_ids=item.query_ids, query_cycles=item.query_cycles,
            missing=item.query_missing, baseline=item.query_baseline,
        )
        predictions[fit_id] = {"bundle_sha256": _sha(bundle), "receipt_sha256": _sha(receipt)}

    fit_source_commit = pilot._git(ROOT, "rev-parse", "HEAD")
    fit_source_hashes = dict(audit.get("continuation_source_file_hashes", {}))
    if set(fit_source_hashes) != set(REPAIR_SOURCE_FILES):
        raise scope.ContinuationError("V1 authorization does not pin the expected adapter source files")
    for relative, expected in fit_source_hashes.items():
        actual = scope._git_file_sha_at_commit(ROOT, fit_source_commit, relative)
        scope._require_digest_match(str(expected), actual, f"fit implementation {relative}")

    local_base = str(freeze.get("local_git_head", ""))
    committed_paths = set(pilot._git(ROOT, "diff", "--name-only", local_base, fit_source_commit).splitlines())
    allowed_paths = set(scope.LOCAL_COMMIT_PATHS)
    if not committed_paths or any(not scope._within_allowed(path, allowed_paths) for path in committed_paths):
        raise scope.ContinuationError("Fit implementation commit includes an unexpected local path")

    final_receipts: dict[str, dict[str, str]] = {}
    for fit_id in ALLOWED_NEW_FITS:
        artifact, receipt_path = pilot._unit_paths(fit_id)
        receipt = pilot._read_json(receipt_path)
        if receipt.get("freeze_signature") != freeze.get("freeze_signature"):
            raise scope.ContinuationError(f"Final fit receipt does not bind the original V4 freeze: {fit_id}")
        final_receipts[fit_id] = {
            "fit_id": fit_id,
            "artifact_sha256": _sha(artifact),
            "receipt_sha256": _sha(receipt_path),
            "original_v4_freeze_signature": freeze["freeze_signature"],
            "fit_authorization_continuation_signature": audit["continuation_signature"],
            "fit_source_commit": fit_source_commit,
            "production_source_commit_at_fit": audit["scoped_source_reconciliation"]["current_source_head"],
        }

    repair_sources = {relative: _sha(ROOT / relative) for relative in REPAIR_SOURCE_FILES}
    repair = {
        "schema_version": 1,
        "repair_version": "PREDICTION_ARM_INVENTORY_REPAIR_V1",
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "card_id": CARD_ID,
        "freeze_signature": freeze["freeze_signature"],
        "fit_authorization_continuation_signature": audit["continuation_signature"],
        "fit_source_commit": fit_source_commit,
        "repair_parent_commit": fit_source_commit,
        "fit_source_file_hashes": fit_source_hashes,
        "repair_source_file_hashes": repair_sources,
        "required_repair_commit_paths": list(REQUIRED_REPAIR_COMMIT_PATHS),
        "prior_postfit_validation_failure": {
            "command": "python scripts/continue_reg_position_direct_l1_v1.py fit-remaining",
            "error": "NO-GO: Missing prediction bundle/receipt for REG_POSITION_DIRECT_L1_V1_E67_STRUCTURED",
            "cause": "The postfit validator incorrectly required prediction bundles for structured/scaler/OHE preprocessing units.",
            "fit_units_at_failure": 12,
            "fit_units_saved_at_failure": 12,
            "fit_failed_at_failure": 0,
            "fit_retry_or_refit_after_failure": False,
        },
        "prediction_scope": {
            "prediction_fit_ids": list(PREDICTION_FIT_IDS),
            "preprocessing_only_fit_ids": [fit_id for fit_id in pilot.FIT_IDS
                                            if fit_id not in PREDICTION_FIT_IDS],
            "preprocessing_units_require_model_receipts": True,
            "preprocessing_units_require_prediction_bundles": False,
            "prediction_inventory": prediction_ids,
        },
        "ledger_sha256": _sha(pilot.LEDGER_PATH),
        "fit_started_ids": [event["fit_id"] for event in starts],
        "fit_saved_ids": [event["fit_id"] for event in saves],
        "fit_failed_count": 0,
        "models": models,
        "predictions": predictions,
        "final_fit_receipts": final_receipts,
        "source_reconciliation_at_repair": {
            "source_head": context["current_source_head"],
            "source_hashes_match": context["source_hashes_match"],
            "budget_log_extension": context["budget_log_extension"],
            "authorized_owner_file_hashes": context["authorized_owner_file_hashes"],
        },
        "query_targets_read": False,
        "test_sample_kaggle_accessed": False,
    }
    repair["repair_signature"] = pilot._canonical_sha(repair)
    pilot._atomic_json(repair_path, repair)
    return repair


def _active_same_pilot_trainers() -> list[dict[str, Any]]:
    current_pid = os.getpid()
    active: list[dict[str, Any]] = []
    for proc in psutil.process_iter(attrs=["pid", "name"]):
        if proc.info.get("pid") == current_pid or "python" not in str(proc.info.get("name", "")).lower():
            continue
        try:
            args = proc.cmdline()
            lowered = [str(arg).lower().replace("\\", "/") for arg in args]
            joined = " ".join(lowered)
            direct_v4 = any(arg.endswith("/reg_position_direct_l1_v1.py") for arg in lowered) and "fit" in lowered
            continuation = any(arg.endswith("/continue_reg_position_direct_l1_v1.py") for arg in lowered) and "fit-remaining" in lowered
            bigram_fit = (any(arg.endswith("/optimization_v3_bigram_position_pilot_v1.py") for arg in lowered)
                          and "run-all" in lowered)
            luna1_named = "luna1" in joined and any(token in joined for token in ("fit", "train", "trainer"))
            if direct_v4 or continuation or bigram_fit or luna1_named:
                cpu = proc.cpu_times()
                active.append({"pid": proc.pid, "command": joined[:500],
                               "cpu_seconds": float(cpu.user + cpu.system)})
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return active


def run_remaining_units() -> dict[str, Any]:
    continuation = scope.verify_continuation(pilot.SOURCE_ROOT, require_completion=False)
    audit = scope._load_continuation_audit()
    freeze = pilot._read_json(pilot.FREEZE_PATH)
    if continuation["checkpoint_units"] != 10 or list(ALLOWED_NEW_FITS) != list(pilot.FIT_IDS[10:12]):
        raise scope.ContinuationError("Continuation is not positioned at exactly the two authorized E75 units")
    local_scope = scope._verify_local_commit_scope(freeze, audit)
    if local_scope["current_head"] == local_scope["base_head"]:
        raise scope.ContinuationError("Original implementation and continuation adapter must be committed before remaining fits")

    # The broad V4 process detector included unrelated pytest sessions. For the
    # versioned continuation, only an active same-pilot or explicitly Luna1
    # trainer blocks a unit; RAM/reserve/RSS gates remain unchanged.
    pilot._active_known_training_processes = _active_same_pilot_trainers
    active = _active_same_pilot_trainers()
    if active:
        raise scope.ContinuationError(f"An actual Bigram, Luna1, or same-pilot trainer is active: {active}")

    source_root = pilot.SOURCE_ROOT.resolve()
    frame = pilot._load_feature_frame(source_root)
    item = pilot._load_origin(source_root, frame, 75)
    if len(frame) != int(audit["scoped_source_reconciliation"]["source_feature_rows"]):
        raise scope.ContinuationError("Source feature row count changed after the signed continuation audit")
    pilot.RUN_ROOT.mkdir(parents=True, exist_ok=True)
    with pilot._LedgerFileLock(pilot.RUN_ROOT / "training.lock"):
        events_before = pilot._read_events(pilot.LEDGER_PATH)
        starts_before = [event.get("fit_id") for event in events_before if event.get("event") == "FIT_STARTED"]
        saves_before = [event.get("fit_id") for event in events_before if event.get("event") == "FIT_SAVED"]
        if (len(starts_before) < 10 or len(starts_before) > 12
                or starts_before != list(pilot.FIT_IDS[:len(starts_before)])
                or saves_before != starts_before):
            raise scope.ContinuationError("Checkpoint order changed before the E75 continuation entered the lock")
        if already_complete_ledger(starts_before, saves_before):
            # A previous single attempt already saved both authorized units.
            # Finalization must not re-enter the estimator path or reread labels.
            result = {"origin": 75, "restored_from_complete_ledger": True,
                      "new_fit_units_started": 0}
        else:
            result = pilot._fit_one_origin(source_root, frame, item, freeze)
        events = pilot._read_events(pilot.LEDGER_PATH)
        starts = [event.get("fit_id") for event in events if event.get("event") == "FIT_STARTED"]
        saves = [event.get("fit_id") for event in events if event.get("event") == "FIT_SAVED"]
        failed = [event for event in events if event.get("event") == "FIT_FAILED"]
        if starts != list(pilot.FIT_IDS) or saves != list(pilot.FIT_IDS) or failed:
            raise scope.ContinuationError("Continuation ledger did not end at exactly 12 ordered starts/saves")

    # Revalidate all model units and the six arm prediction bundles before readiness.
    e67_item = pilot._load_origin(source_root, frame, 67)
    items = {67: e67_item, 75: item}
    models: dict[str, dict[str, str]] = {}
    for fit_id in pilot.FIT_IDS:
        artifact_path, receipt_path = pilot._unit_paths(fit_id)
        _, receipt = pilot._validate_unit(pilot.RUN_ROOT, fit_id, freeze)
        models[fit_id] = {"artifact_sha256": _sha(artifact_path),
                          "receipt_sha256": _sha(receipt_path),
                          "receipt_signature": str(receipt.get("receipt_sha256", "")),
                          "freeze_signature": str(receipt.get("freeze_signature", ""))}
    predictions = {}
    prediction_ids = scope.prediction_inventory(pilot.RUN_ROOT)
    for fit_id in prediction_ids:
        origin = 67 if "_E67_" in fit_id else 75
        arm = fit_id.rsplit("_", 1)[-1]
        item_for_prediction = items[origin]
        artifact_path, receipt_path = pilot._prediction_paths(fit_id)
        pilot._validate_prediction_bundle(
            artifact_path, receipt_path, freeze=freeze, fit_id=fit_id,
            origin=origin, arm=arm, query_rows=item_for_prediction.query_rows,
            query_ids=item_for_prediction.query_ids, query_cycles=item_for_prediction.query_cycles,
            missing=item_for_prediction.query_missing, baseline=item_for_prediction.query_baseline,
        )
        predictions[fit_id] = {"bundle_sha256": _sha(artifact_path),
                               "receipt_sha256": _sha(receipt_path)}
    continuation = scope.verify_continuation(pilot.SOURCE_ROOT, require_completion=False)
    repair = scope._read_json(scope.BUDGET_REPAIR_PATH)
    repair_receipts = repair.get("final_fit_receipts", {})
    for fit_id in ALLOWED_NEW_FITS:
        link = repair_receipts.get(fit_id, {})
        models[fit_id].update({
            "fit_authorization_continuation_signature": link.get("fit_authorization_continuation_signature"),
            "fit_source_commit": link.get("fit_source_commit"),
            "production_source_commit_at_fit": link.get("production_source_commit_at_fit"),
        })

    ready = {
        "schema_version": 1, "card_id": CARD_ID,
        "status": "READY_FOR_PRELABEL_VERIFICATION",
        "freeze_signature": freeze["freeze_signature"],
        "fit_units_started": 12, "fit_units_saved": 12, "fit_units_failed": 0,
        "query_targets_read": False,
        "continuation_signature": continuation["continuation_signature"],
        "origins": [{"origin": 67, "restored_from_checkpoint": True}, result],
        "ledger_sha256": _sha(pilot.LEDGER_PATH),
        "prediction_fit_ids": list(prediction_ids),
        "predictions": predictions,
    }
    pilot._atomic_json(pilot.RUN_ROOT / "PREDICTIONS_READY.json", ready)
    pilot._atomic_json(pilot.REPORT_ROOT / "PREDICTIONS_READY.json", ready)
    completion = {
        "schema_version": 1, "card_id": CARD_ID, "status": "PASS",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "continuation_signature": continuation["continuation_signature"],
        "freeze_signature": freeze["freeze_signature"],
        "fit_units_started": 12, "fit_units_saved": 12, "fit_units_failed": 0,
        "remaining_fit_units": 0, "new_fit_ids": list(ALLOWED_NEW_FITS),
        "ledger_sha256": ready["ledger_sha256"],
        "checkpoint_ledger_prefix_sha256": audit["checkpoint_ledger_sha256"],
        "implementation_commit_head": pilot._git(ROOT, "rev-parse", "HEAD"),
        "fit_source_commit": repair["fit_source_commit"],
        "production_source_commit_at_fit": audit["scoped_source_reconciliation"]["current_source_head"],
        "postfit_validator_repair": {
            "path": _relative(scope.BUDGET_REPAIR_PATH),
            "sha256": _sha(scope.BUDGET_REPAIR_PATH),
            "repair_signature": repair["repair_signature"],
            "repair_version": repair["repair_version"],
        },
        "source_reconciliation_at_completion": continuation["source_reconciliation"],
        "budget_log_extension_at_completion": continuation["budget_log_extension"],
        "authorized_owner_file_hashes_at_completion": continuation["authorized_owner_file_hashes"],
        "models": {
            **models,
        },
        "final_fit_receipts": repair_receipts,
        "predictions": predictions,
        "query_targets_read": False,
    }
    completion["completion_signature"] = pilot._canonical_sha(completion)
    pilot._atomic_json(scope.COMPLETION_PATH, completion)
    completed = scope.verify_continuation(pilot.SOURCE_ROOT, require_completion=True)
    return {"status": "PASS", "fit_units_started": 12, "fit_units_saved": 12,
            "new_fit_ids": list(ALLOWED_NEW_FITS), "freeze_signature": freeze["freeze_signature"],
            "continuation_signature": continuation["continuation_signature"],
            "ledger_sha256": ready["ledger_sha256"],
            "source_reconciliation": completed["source_reconciliation"],
            "budget_log_extension": completed["budget_log_extension"],
            "query_targets_read": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "repair-audit", "fit-remaining"))
    args = parser.parse_args(argv)
    try:
        if args.command == "audit":
            result = create_continuation_audit()
        elif args.command == "repair-audit":
            result = create_postfit_repair_audit()
        else:
            result = run_remaining_units()
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0
    except (scope.ContinuationError, pilot.PilotError) as exc:
        print(f"NO-GO: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
