"""Versioned progress checkpoint and resume for the approved POS+A02 run.

Use this after an approved run makes valid progress but stops at a pre-fit RAM
gate. Every checkpoint binds the complete current ledger plus hashes of all
saved states and receipts. Resume is refused if any byte changes after the
checkpoint. Existing fits are only validated/reused; the frozen runner enforces
the original six-ID cap and never retries a started fit without durable state.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterator

import psutil

from scripts import optimization_v3_cls_pos_a02_meta_v1 as frozen
from scripts import optimization_v3_cls_pos_a02_meta_v1_warning_recovery as warning_recovery
from src.models.research_next_contract import file_sha256

ROOT = frozen.ROOT
SCRIPT = Path(__file__).resolve()
TESTS = ROOT / "tests/test_optimization_v3_cls_pos_a02_meta_v1_progress_recovery.py"
REPORT_PREFIX = "RUNTIME_RECOVERY_PROGRESS_R"
REPORT_PATTERN = re.compile(r"^RUNTIME_RECOVERY_PROGRESS_R(\d+)\.json$")
EXPECTED_FROZEN_SCRIPT_SHA256 = warning_recovery.EXPECTED_FROZEN_SCRIPT_SHA256
EXPECTED_RECOVERY_ADAPTER_SHA256 = "ff1f74b612decd661a0d01ae33ccae122e829ed88897821880cb4cd83987d7fe"


class ProgressRecoveryError(RuntimeError):
    """Fail-closed error for a stale or inconsistent progress checkpoint."""


def _report_files() -> list[tuple[int, Path]]:
    found = []
    for path in frozen.REPORT_DIR.glob(f"{REPORT_PREFIX}*.json"):
        match = REPORT_PATTERN.match(path.name)
        if match:
            found.append((int(match.group(1)), path))
    return sorted(found)


def _event_line_hashes() -> list[str]:
    if not frozen.LEDGER.is_file():
        raise ProgressRecoveryError("Official fit ledger is absent")
    lines = frozen.LEDGER.read_text(encoding="utf-8").splitlines()
    return [hashlib.sha256(line.encode("utf-8")).hexdigest() for line in lines]


def ledger_is_extension(previous_hashes: list[str], current_hashes: list[str]) -> bool:
    return len(current_hashes) >= len(previous_hashes) and current_hashes[:len(previous_hashes)] == previous_hashes


def _verify_root_and_original_recovery() -> tuple[dict[str, Any], dict[str, Any], str, dict[str, Any]]:
    freeze = frozen._validate_prefit_freeze()
    if file_sha256(frozen.SCRIPT) != EXPECTED_FROZEN_SCRIPT_SHA256:
        raise ProgressRecoveryError("The R2 training script changed after root approval")
    approval, approval_sha = frozen._require_root_approval()
    if approval.get("status") != "PASS":
        raise ProgressRecoveryError("Root approval is not PASS")
    initial = frozen._receipt(warning_recovery.RECOVERY_PREFLIGHT)
    if initial.get("status") != "RUNTIME_RECOVERY_PREFLIGHT_PASS":
        raise ProgressRecoveryError("Reader recovery preflight is missing or invalid")
    if initial.get("recovery_adapter_sha256") != EXPECTED_RECOVERY_ADAPTER_SHA256:
        raise ProgressRecoveryError("Reader recovery adapter differs from its committed preflight")
    if initial.get("recovery_test_sha256") != file_sha256(warning_recovery.TESTS):
        raise ProgressRecoveryError("Reader recovery tests changed after preflight")
    if initial.get("prefit_freeze_receipt_sha256") != freeze["receipt_sha256"]:
        raise ProgressRecoveryError("Reader recovery preflight is bound to another freeze")
    if initial.get("root_approval_sha256") != approval_sha:
        raise ProgressRecoveryError("Reader recovery preflight is bound to another approval")
    if not frozen._git_ancestor(initial.get("source_commit", ""), frozen._git_head()):
        raise ProgressRecoveryError("Reader recovery preflight commit is not an ancestor")
    return freeze, approval, approval_sha, initial


def _snapshot() -> dict[str, Any]:
    freeze, _approval, approval_sha, initial = _verify_root_and_original_recovery()
    source_hashes = frozen._source_hashes()
    input_hashes = frozen._input_hashes()
    events = frozen._ledger()
    grouped = frozen._validate_fit_ledger(events)
    starts = [event for event in events if event.get("event") == "FIT_START"]
    reserves = [event for event in events if event.get("event") == "RESERVE"]
    completes = [event for event in events if event.get("event") == "FIT_COMPLETE"]
    failures = [event for event in events if event.get("event") == "FIT_FAILED"]
    if len(starts) > frozen.MAX_OFFICIAL_FITS or len(reserves) > frozen.MAX_OFFICIAL_FITS:
        raise ProgressRecoveryError("The frozen six-fit cap has been exceeded")
    if failures:
        raise ProgressRecoveryError("A terminal FIT_FAILED event prohibits automatic continuation")
    allowed = set(frozen.FIT_IDS)
    if not set(grouped).issubset(allowed):
        raise ProgressRecoveryError("The ledger contains a fit ID outside the frozen allowlist")

    fits: list[dict[str, Any]] = []
    for fit_id in frozen.FIT_IDS:
        state_path, receipt_path, failure_path = frozen._fit_paths(fit_id)
        fit_events = grouped.get(fit_id)
        if fit_events is None:
            if state_path.exists() or receipt_path.exists() or failure_path.exists():
                raise ProgressRecoveryError(f"Unreserved fit artifacts exist for {fit_id}")
            continue
        if failure_path.exists():
            raise ProgressRecoveryError(f"Terminal failure artifact exists for {fit_id}")
        if not state_path.is_file() or not receipt_path.is_file():
            raise ProgressRecoveryError(f"Started fit {fit_id} lacks durable state/receipt; retry is prohibited")
        receipt = frozen._receipt(receipt_path)
        signature = receipt.get("fit_signature")
        if not isinstance(signature, dict):
            raise ProgressRecoveryError(f"Fit receipt lacks a signature for {fit_id}")
        if (signature.get("fit_id") != fit_id
                or signature.get("approval_sha256") != approval_sha
                or signature.get("source_hashes") != source_hashes
                or signature.get("input_hashes") != input_hashes
                or signature.get("estimator") != frozen.MODEL_PARAMS
                or signature.get("threadpool_limit") != frozen.MAX_THREADS):
            raise ProgressRecoveryError(f"Fit {fit_id} does not bind to current frozen sources/recipe")
        if receipt.get("source_hashes") != source_hashes or receipt.get("input_hashes") != input_hashes:
            raise ProgressRecoveryError(f"Fit receipt source/input hashes differ for {fit_id}")
        if receipt.get("model_state_path") != state_path.relative_to(ROOT).as_posix():
            raise ProgressRecoveryError(f"Fit model state path differs for {fit_id}")
        binding_status = frozen._validate_fit_binding(fit_id, signature, receipt, fit_events)
        state = warning_recovery.read_model_state(state_path, receipt)
        reserve = fit_events["RESERVE"]
        if reserve.get("fit_signature") != signature:
            raise ProgressRecoveryError(f"Reservation signature differs from receipt for {fit_id}")
        if signature.get("feature_names") != receipt.get("feature_names"):
            raise ProgressRecoveryError(f"Fit feature names differ from its receipt for {fit_id}")
        if signature.get("feature_width") != receipt.get("feature_width"):
            raise ProgressRecoveryError(f"Fit feature width differs from its receipt for {fit_id}")
        if signature.get("rows") != receipt.get("fit_rows"):
            raise ProgressRecoveryError(f"Fit row count differs from its receipt for {fit_id}")
        fits.append({
            "fit_id": fit_id,
            "arm": receipt["arm"],
            "status": binding_status,
            "fit_receipt_sha256": receipt["receipt_sha256"],
            "model_state_file_sha256": receipt["model_state_file_sha256"],
            "fit_signature_sha256": receipt["fit_signature_sha256"],
            "n_iter": receipt["n_iter"],
            "warning_categories": receipt["convergence_warnings"],
            "fit_rows": receipt["fit_rows"],
            "selection_cycle_range": receipt["selection_cycle_range"],
            "parameter_state_finite": bool(np_isfinite(state["coef"]) and np_isfinite(state["intercept"])),
            "fit_started": True,
            "fit_complete_event_present": "FIT_COMPLETE" in fit_events,
            "fit_retry": False,
        })

    if sum(fit["fit_started"] for fit in fits) != len(starts):
        raise ProgressRecoveryError("Raw start-event count does not match validated unique fit units")
    if len(reserves) != len(starts):
        raise ProgressRecoveryError("Reservation and start-event counts differ")
    if len(completes) + sum(not fit["fit_complete_event_present"] for fit in fits) != len(starts):
        raise ProgressRecoveryError("Completion-event count does not match validated fit units")

    d1_receipt = frozen._receipt(frozen._fit_paths("D1_PAIR")[1])
    if (d1_receipt["receipt_sha256"] != initial["d1_pair"]["fit_receipt_sha256"]
            or d1_receipt["model_state_file_sha256"] != initial["d1_pair"]["model_state_file_sha256"]):
        raise ProgressRecoveryError("Original D1_PAIR state or receipt changed after recovery")
    if frozen.E79_FREEZE.exists() or frozen.PAIR_CSV.exists():
        raise ProgressRecoveryError("Test application or candidate CSV exists before final model states")
    return {
        "schema_version": 1,
        "experiment_id": frozen.EXPERIMENT,
        "recovery_id": "CLS_POS_A02_META_V1_PROGRESS_RECOVERY",
        "status": "PROGRESS_CHECKPOINT_PASS",
        "authorization": "Sol root approved the six-fit frozen run and the separate warning-aware reader recovery.",
        "source_commit": frozen._git_head(),
        "frozen_prefit_commit": freeze["prefit_commit"],
        "frozen_prefit_script_sha256": EXPECTED_FROZEN_SCRIPT_SHA256,
        "prefit_freeze_receipt_sha256": freeze["receipt_sha256"],
        "root_approval_sha256": approval_sha,
        "reader_recovery_receipt_sha256": initial["receipt_sha256"],
        "progress_adapter_sha256": file_sha256(SCRIPT),
        "progress_test_sha256": file_sha256(TESTS),
        "ledger_sha256": file_sha256(frozen.LEDGER),
        "ledger_event_count": len(events),
        "ledger_line_sha256": _event_line_hashes(),
        "official_fit_accounting": {
            "starts": len(starts), "completions": len(completes), "failures": len(failures),
            "unique_fit_ids": [fit["fit_id"] for fit in fits],
            "remaining_predeclared_fits": frozen.MAX_OFFICIAL_FITS - len(starts),
            "cap": frozen.MAX_OFFICIAL_FITS,
        },
        "validated_fit_units": fits,
        "ram_snapshot_bytes_available": int(psutil.virtual_memory().available),
        "test_probabilities_loaded": False,
        "test_ids_loaded": False,
        "sample_submission_loaded": False,
        "candidate_csv_created": False,
        "kaggle_upload": False,
    }


def np_isfinite(values: Any) -> bool:
    import numpy as np
    return bool(np.isfinite(np.asarray(values, dtype=np.float64)).all())


def _latest_report() -> tuple[int, Path, dict[str, Any]]:
    reports = _report_files()
    if not reports:
        raise ProgressRecoveryError("No committed progress checkpoint exists")
    version, path = reports[-1]
    return version, path, frozen._receipt(path)


def checkpoint() -> dict[str, Any]:
    current = _snapshot()
    reports = _report_files()
    if reports:
        prior_version, prior_path = reports[-1]
        prior = frozen._receipt(prior_path)
        if prior.get("recovery_id") != current["recovery_id"]:
            raise ProgressRecoveryError("Prior recovery checkpoint has a different identity")
        old_lines = prior.get("ledger_line_sha256")
        new_lines = current["ledger_line_sha256"]
        if not isinstance(old_lines, list) or not ledger_is_extension(old_lines, new_lines):
            raise ProgressRecoveryError("Current ledger is not a valid extension of the prior checkpoint")
        prior_states = {item["fit_id"]: item for item in prior.get("validated_fit_units", [])}
        current_states = {item["fit_id"]: item for item in current["validated_fit_units"]}
        for fit_id, prior_fit in prior_states.items():
            if current_states.get(fit_id) != prior_fit:
                raise ProgressRecoveryError(f"Previously checkpointed fit changed: {fit_id}")
        version = prior_version + 1
    else:
        version = 2
    path = frozen.REPORT_DIR / f"{REPORT_PREFIX}{version}.json"
    receipt_sha = frozen._write_json_once(path, current)
    return {**current, "checkpoint_path": path.relative_to(ROOT).as_posix(),
            "receipt_sha256": receipt_sha}


def _validate_latest_checkpoint() -> dict[str, Any]:
    _version, _path, report = _latest_report()
    if report.get("status") != "PROGRESS_CHECKPOINT_PASS":
        raise ProgressRecoveryError("Latest progress checkpoint is not PASS")
    if report.get("progress_adapter_sha256") != file_sha256(SCRIPT):
        raise ProgressRecoveryError("Progress adapter changed after checkpoint")
    if report.get("progress_test_sha256") != file_sha256(TESTS):
        raise ProgressRecoveryError("Progress recovery tests changed after checkpoint")
    if not frozen._git_ancestor(report.get("source_commit", ""), frozen._git_head()):
        raise ProgressRecoveryError("Progress checkpoint commit is not an ancestor of current HEAD")
    live = _snapshot()
    if report.get("ledger_sha256") != live.get("ledger_sha256"):
        raise ProgressRecoveryError("Ledger changed after checkpoint; create a new progress checkpoint before resume")
    if report.get("ledger_line_sha256") != live.get("ledger_line_sha256"):
        raise ProgressRecoveryError("Ledger event lines changed after checkpoint")
    for key in ("reader_recovery_receipt_sha256", "root_approval_sha256", "prefit_freeze_receipt_sha256",
                "validated_fit_units", "official_fit_accounting"):
        if report.get(key) != live.get(key):
            raise ProgressRecoveryError(f"Progress checkpoint binding changed: {key}")
    return report


@contextmanager
def _patched_reader() -> Iterator[None]:
    old_reader = frozen._read_model_state
    frozen._read_model_state = warning_recovery.read_model_state
    try:
        yield
    finally:
        frozen._read_model_state = old_reader


def resume_approved_run() -> dict[str, Any]:
    report = _validate_latest_checkpoint()
    resource = frozen.resource_gate()
    if resource.get("status") != "PASS":
        raise ProgressRecoveryError("RAM gate did not pass; no new fit may start")
    with _patched_reader():
        result = frozen.run_official()
    return {"status": "RUN_COMPLETED", "progress_checkpoint_sha256": report["receipt_sha256"],
            "result": result}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", action="store_true")
    group.add_argument("--resume-approved-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = checkpoint() if args.checkpoint else resume_approved_run()
        print(json.dumps(result, indent=2, sort_keys=True))
    except Exception as exc:
        print(json.dumps({"status": "PROGRESS_RECOVERY_BLOCKED_OR_FAILED",
                          "type": type(exc).__name__, "message": str(exc)}, indent=2),
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
