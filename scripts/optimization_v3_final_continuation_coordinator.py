"""MAIN-owned coordinator for the frozen C16 + R22 continuation only."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import psutil

from scripts import optimization_v3_final_continuation_cls as c16
from scripts import optimization_v3_final_continuation_r22 as r22
from src.models.optimization_v3_batch1 import (
    BatchIntegrityError,
    append_event,
    exclusive_training_slot,
    file_sha256,
    json_sha256,
    reserve_card_budget,
    write_json_exclusive,
)


DECISION_PATH = ROOT / "reports/optimization_v3/final_continuation_plan/DECISION.md"
AMENDMENT_PATH = ROOT / "experiments/optimization_v3/final_continuation/scope_amendment.json"
PREFLIGHT_PATH = ROOT / "reports/optimization_v3/final_continuation/combined_preflight.json"
CAMPAIGN_RESULT_PATH = ROOT / "reports/optimization_v3/final_continuation/campaign_result.json"
CAMPAIGN_VERIFY_PATH = ROOT / "reports/optimization_v3/final_continuation/campaign_verification.json"
CAMPAIGN_SUMMARY_PATH = ROOT / "reports/optimization_v3/final_continuation/CAMPAIGN_SUMMARY.md"
LEDGER_PATH = ROOT / "experiments/runs/optimization_v3_batch1/batch_ledger.jsonl"
GLOBAL_LOCK = ROOT / "experiments/runs/optimization_v3_batch1/.training.lock"
JOINT_KEY = "FINAL_CONTINUATION:C16_R22"
JOINT_CARD = "C16+R22"
JOINT_FITS = 8
FIT_CUTOFFS = (59, 63, 71, 75)
CAMPAIGN_ID = "optimization_v3_final_continuation"
TEST_PATHS = [
    ROOT / "tests/test_optimization_v3_final_continuation_cls.py",
    ROOT / "tests/test_optimization_v3_final_continuation_r22.py",
    ROOT / "tests/test_optimization_v3_final_continuation_coordinator.py",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha_identity() -> dict[str, Any]:
    c16_freeze = _read_json(c16.FREEZE_PATH)
    r22_freeze = _read_json(r22.FREEZE_PATH)
    c16_preflight = _read_json(c16.PREFLIGHT_PATH)
    r22_preflight = _read_json(r22.PREFLIGHT_PATH)
    return {
        "schema_version": 1,
        "experiment_id": CAMPAIGN_ID,
        "decision_sha256": file_sha256(DECISION_PATH),
        "scope": {
            "C16": {"new_cutoffs": list(FIT_CUTOFFS), "new_predictor_fits": 4,
                    "reused_pilot_cutoffs": [55, 67], "freeze_signature": c16_freeze["signature"],
                    "recipe_sha256": file_sha256(c16.RECIPE_PATH)},
            "R22": {"new_cutoffs": list(FIT_CUTOFFS), "new_predictor_fits": 4,
                    "reused_pilot_cutoffs": [55, 67], "freeze_signature": r22_freeze["signature"],
                    "recipe_sha256": file_sha256(r22.RECIPE_PATH),
                    "one_fixed_postprocessing_comparator": "R22_PCT_PRESENT", "additional_fits": 0},
            "total_new_predictor_fits": JOINT_FITS,
            "not_in_scope": ["C11", "R06", "C17", "R21", "R11", "R12 rescue", "R23 rescue", "R16",
                             "full-Train refit", "official Test inference", "Kaggle submission", "notebook finalization"],
        },
        "source_sha256": {
            "coordinator": file_sha256(Path(__file__)),
            "tests": {path.relative_to(ROOT).as_posix(): file_sha256(path) for path in TEST_PATHS},
            "C16": c16_freeze["identity"]["source_sha256"],
            "R22": r22_freeze["identity"]["source_sha256"],
        },
        "input_sha256": {
            "C16": c16_freeze["identity"]["input_sha256"],
            "R22": r22_freeze["identity"]["input_sha256"],
            "decision": file_sha256(DECISION_PATH),
        },
        "frozen_gate_source": "reports/optimization_v3/final_continuation_plan/DECISION.md",
        "fit_policy": {
            "global_reservation": {"budget_key": JOINT_KEY, "card_id": JOINT_CARD,
                                   "predictor_fits": JOINT_FITS, "cap": 180, "atomic": True},
            "concurrent_training_jobs": 1,
            "threads_per_job": 2,
            "job_memory_cap_bytes": 6 * 1024**3,
            "free_memory_reserve_bytes": 2 * 1024**3,
            "resume": "reuse only complete hash-verified per-origin bundles; block partial bundles or a started attempt without a saved bundle",
        },
        "package_versions": {
            "C16": c16_freeze["identity"]["package_versions"],
            "R22": r22_freeze["identity"]["package_versions"],
        },
        "test_files_frozen": {path.relative_to(ROOT).as_posix(): file_sha256(path) for path in TEST_PATHS},
        "preflight_receipts": {
            "C16_freeze_signature": c16_preflight["freeze_signature"],
            "R22_freeze_signature": r22_preflight["freeze_signature"],
        },
    }


def _validate_freezes_and_contexts() -> tuple[dict[str, Any], dict[str, Any]]:
    c16_context = c16._context()
    c16_signature = c16._verify_current_freeze(c16_context)
    if c16_signature != c16_context["freeze_signature"] or not c16_context["resources_pass"]:
        raise BatchIntegrityError("C16 continuation freeze/resource preflight does not pass")

    r22_context = r22.prepare_context()
    if not all(item["status"] == "PASS" for item in r22_context["resources"].values()):
        raise BatchIntegrityError("R22 continuation resource preflight does not pass")
    frozen = _read_json(r22.FREEZE_PATH)
    if (frozen.get("identity") != r22_context["identity"]
            or frozen.get("signature") != r22_context["signature"]
            or json_sha256(r22_context["identity"]) != r22_context["signature"]):
        raise BatchIntegrityError("R22 continuation freeze differs from current source/input identity")
    if not r22.RECIPE_PATH.is_file() or file_sha256(r22.RECIPE_PATH) != r22_context["identity"].get("recipe_sha256"):
        raise BatchIntegrityError("R22 recipe is missing or differs from the frozen hash")
    return c16_context, r22_context


def _run_tests() -> dict[str, Any]:
    missing = [str(path) for path in TEST_PATHS if not path.is_file()]
    if missing:
        raise BatchIntegrityError(f"Continuation QA test files are missing: {missing}")
    command = [sys.executable, "-m", "pytest", "-q", *[str(path.relative_to(ROOT)) for path in TEST_PATHS]]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    report = {"command": command, "exit_code": int(completed.returncode),
              "stdout": completed.stdout.strip(), "stderr": completed.stderr.strip(),
              "test_sources_sha256": {p.relative_to(ROOT).as_posix(): file_sha256(p) for p in TEST_PATHS}}
    if completed.returncode != 0:
        raise BatchIntegrityError(f"Continuation tests failed before fit: {report}")
    return report


def _static_resource_pass(context: Mapping[str, Any], card: str) -> dict[int, dict[str, Any]]:
    if card == "C16":
        source = context["resources"]
        items = {int(k): value for k, value in source.items()}
    else:
        items = {int(k): value for k, value in context["resources"].items()}
    if set(items) != set(FIT_CUTOFFS) or any(item.get("status") != "PASS" for item in items.values()):
        raise BatchIntegrityError(f"{card} static resource estimate is incomplete or blocked")
    return items


def _live_resource_check(context: Mapping[str, Any], card: str, cutoff: int) -> dict[str, Any]:
    estimate = _static_resource_pass(context, card)[int(cutoff)]
    available = int(psutil.virtual_memory().available)
    cap = 6 * 1024**3
    reserve = 2 * 1024**3
    allowed = min(cap, max(0, available - reserve))
    peak = int(estimate.get("estimated_peak_bytes", 0))
    record = {
        "card_id": card,
        "cutoff": int(cutoff),
        "estimated_peak_bytes": peak,
        "estimate_is_observed": False,
        "available_physical_bytes_before_fit": available,
        "required_free_reserve_bytes": reserve,
        "job_cap_bytes": cap,
        "allowed_bytes": allowed,
        "threads": 2,
        "concurrent_training_jobs": 1,
        "status": "PASS" if peak <= allowed else "BLOCKED_RESOURCE",
    }
    if record["status"] != "PASS":
        raise BatchIntegrityError(f"Live resource check blocks {card} cutoff {cutoff}: {record}")
    return record


def _budget_state() -> dict[str, Any]:
    return r22._budget_snapshot()


def _reservation_check(state: Mapping[str, Any]) -> None:
    if state["joint_reservation_present"]:
        if state["joint_reservation"] != (JOINT_FITS, JOINT_CARD):
            raise BatchIntegrityError(f"Existing joint reservation has conflicting identity: {state}")
    elif state["reserved_total"] + JOINT_FITS > 180:
        raise BatchIntegrityError(f"Joint reservation exceeds budget cap: {state}")
    for key in ("C16:FINAL_CONTINUATION", "R22:FINAL_CONTINUATION"):
        if state["joint_reservation_present"] is False:
            lines = LEDGER_PATH.read_text(encoding="utf-8").splitlines() if LEDGER_PATH.is_file() else []
            for line in lines:
                if line.strip() and json.loads(line).get("event") == "BUDGET_RESERVED" and json.loads(line).get("budget_key") == key:
                    raise BatchIntegrityError(f"Separate partial continuation reservation exists: {key}")


def _ensure_amendment(c16_context: Mapping[str, Any], r22_context: Mapping[str, Any]) -> dict[str, Any]:
    AMENDMENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    identity = _sha_identity()
    amendment = {"identity": identity, "signature": json_sha256(identity),
                 "created_before_official_fits": True, "status": "FROZEN_SCOPE_READY_FOR_EIGHT_FITS"}
    if AMENDMENT_PATH.exists():
        existing = _read_json(AMENDMENT_PATH)
        if existing != amendment:
            raise BatchIntegrityError("Existing combined scope amendment differs; refusing rewrite")
        return existing
    write_json_exclusive(AMENDMENT_PATH, amendment)
    return amendment


def _initial_preflight(c16_context: Mapping[str, Any], r22_context: Mapping[str, Any],
                       tests: Mapping[str, Any], amendment: Mapping[str, Any]) -> dict[str, Any]:
    budget = _budget_state()
    _reservation_check(budget)
    current_free = int(psutil.virtual_memory().available)
    lock = r22._check_lock_available()
    if not lock["available"]:
        raise BatchIntegrityError("Global V3 lock is held during combined preflight")
    resources = {
        card: {str(c): _live_resource_check(context, card, c)
               for c in FIT_CUTOFFS}
        for card, context in (("C16", c16_context), ("R22", r22_context))
    }
    return {
        "status": "READY_TO_RESERVE_AND_FIT",
        "scope_amendment_signature": amendment["signature"],
        "c16_freeze_signature": c16_context["freeze_signature"],
        "r22_freeze_signature": r22_context["signature"],
        "tests": dict(tests),
        "current_available_ram_bytes": current_free,
        "global_training_lock": lock,
        "global_budget_before_reservation": budget,
        "proposed_joint_reservation": {"budget_key": JOINT_KEY, "card_id": JOINT_CARD,
                                       "predictor_fits": JOINT_FITS, "reserved": False},
        "live_resource_checks": resources,
        "test_access": False,
        "kaggle_access": False,
        "official_fits_started": False,
        "created_at_utc": _now(),
    }


def preflight_only() -> dict[str, Any]:
    # R22 freeze is created here only after the no-fit path and synthetic tests
    # have passed. C16's freeze is already in place and independently verified.
    c16_context, r22_context = _validate_freezes_and_contexts()
    tests = _run_tests()
    amendment = _ensure_amendment(c16_context, r22_context)
    payload = _initial_preflight(c16_context, r22_context, tests, amendment)
    if PREFLIGHT_PATH.exists():
        existing = _read_json(PREFLIGHT_PATH)
        if (existing.get("scope_amendment_signature") != amendment["signature"]
                or existing.get("c16_freeze_signature") != c16_context["freeze_signature"]
                or existing.get("r22_freeze_signature") != r22_context["signature"]):
            raise BatchIntegrityError("Existing combined preflight belongs to another scope/freeze")
        return existing
    write_json_exclusive(PREFLIGHT_PATH, payload)
    return payload


def _read_global_fit_state(card: str, cutoff: int) -> dict[str, bool]:
    experiment_id = f"{CAMPAIGN_ID}_{card}"
    state = {"started": False, "saved": False, "failed": False}
    if not LEDGER_PATH.is_file():
        return state
    for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("experiment_id") != experiment_id or int(event.get("cutoff", -1)) != int(cutoff):
            continue
        name = event.get("event")
        if name in ("CONTINUATION_FIT_STARTED", "CONTINUATION_FIT_SAVED", "CONTINUATION_FIT_FAILED"):
            state[{"CONTINUATION_FIT_STARTED": "started", "CONTINUATION_FIT_SAVED": "saved",
                   "CONTINUATION_FIT_FAILED": "failed"}[name]] = True
    return state


def _bundle_state(card: str, cutoff: int) -> tuple[bool, bool]:
    if card == "C16":
        path = c16.RUN_DIR / f"C16_fit{cutoff}.npz"
    else:
        path = r22.RUN_DIR / f"R22_fit{cutoff}.npz"
    sidecar = path.with_suffix(".json")
    return path.is_file(), sidecar.is_file()


def _resume_action(has_npz: bool, has_sidecar: bool,
                   state: Mapping[str, bool]) -> str:
    if has_npz != has_sidecar:
        return "BLOCK_PARTIAL_BUNDLE"
    if has_npz and has_sidecar:
        return "VERIFY_EXISTING_BUNDLE"
    if state.get("started") or state.get("failed"):
        return "BLOCK_PRIOR_ATTEMPT_WITHOUT_BUNDLE"
    return "FIT_ONCE"


def _run_c16_cutoff(context: dict[str, Any], cutoff: int,
                    amendment_signature: str) -> dict[str, Any]:
    state = _read_global_fit_state("C16", cutoff)
    has_npz, has_sidecar = _bundle_state("C16", cutoff)
    action = _resume_action(has_npz, has_sidecar, state)
    if action == "BLOCK_PARTIAL_BUNDLE":
        raise BatchIntegrityError(f"C16 cutoff {cutoff} has a partial bundle; refusing fit/resume")
    if action == "BLOCK_PRIOR_ATTEMPT_WITHOUT_BUNDLE":
        raise BatchIntegrityError(f"C16 cutoff {cutoff} has a prior unfinished attempt without a bundle")
    resource = _live_resource_check(context, "C16", cutoff)
    if not has_npz:
        append_event(LEDGER_PATH, {
            "event": "CONTINUATION_FIT_STARTED", "at_utc": _now(),
            "experiment_id": f"{CAMPAIGN_ID}_C16", "card_id": "C16", "cutoff": int(cutoff),
            "predictor_fit_count": 1, "scope_amendment_signature": amendment_signature,
            "resource_preflight": resource,
        })
    try:
        record, fitted = c16.execute_cutoff(context, context["freeze_signature"], cutoff)
    except Exception as exc:
        if not has_npz:
            append_event(LEDGER_PATH, {
                "event": "CONTINUATION_FIT_FAILED", "at_utc": _now(),
                "experiment_id": f"{CAMPAIGN_ID}_C16", "card_id": "C16", "cutoff": int(cutoff),
                "predictor_fit_count": 1, "scope_amendment_signature": amendment_signature,
                "error_type": type(exc).__name__, "error": str(exc)[:1200],
            })
        raise
    npz_path = c16.RUN_DIR / f"C16_fit{cutoff}.npz"
    sidecar_path = npz_path.with_suffix(".json")
    if not fitted and not state["saved"]:
        append_event(LEDGER_PATH, {
            "event": "CONTINUATION_FIT_SAVED", "at_utc": _now(),
            "experiment_id": f"{CAMPAIGN_ID}_C16", "card_id": "C16", "cutoff": int(cutoff),
            "predictor_fit_count": 1, "reconciled_verified_bundle": True,
            "npz_sha256": file_sha256(npz_path), "sidecar_sha256": file_sha256(sidecar_path),
        })
    elif fitted:
        metadata = record.get("metadata", {})
        append_event(LEDGER_PATH, {
            "event": "CONTINUATION_FIT_SAVED", "at_utc": _now(),
            "experiment_id": f"{CAMPAIGN_ID}_C16", "card_id": "C16", "cutoff": int(cutoff),
            "predictor_fit_count": 1, "fit_seconds": metadata.get("fit_seconds"),
            "scope_amendment_signature": amendment_signature,
            "npz_sha256": file_sha256(npz_path), "sidecar_sha256": file_sha256(sidecar_path),
        })
    return {"card_id": "C16", "cutoff": int(cutoff), "fit_status": "FIT" if fitted else "RESUMED_VERIFIED",
            "record": record}


def _run_r22_cutoff(context: dict[str, Any], cutoff: int,
                    amendment_signature: str) -> dict[str, Any]:
    state = _read_global_fit_state("R22", cutoff)
    has_npz, has_sidecar = _bundle_state("R22", cutoff)
    action = _resume_action(has_npz, has_sidecar, state)
    if action == "BLOCK_PARTIAL_BUNDLE":
        raise BatchIntegrityError(f"R22 cutoff {cutoff} has a partial bundle; refusing fit/resume")
    if action == "BLOCK_PRIOR_ATTEMPT_WITHOUT_BUNDLE":
        raise BatchIntegrityError(f"R22 cutoff {cutoff} has a prior unfinished attempt without a bundle")
    resource = _live_resource_check(context, "R22", cutoff)
    before_events = _read_global_fit_state("R22", cutoff)
    try:
        record = r22.execute_cutoff(context, context["signature"], cutoff, lock_held=True)
    except Exception:
        raise
    npz_path = r22.RUN_DIR / f"R22_fit{cutoff}.npz"
    sidecar_path = npz_path.with_suffix(".json")
    after_events = _read_global_fit_state("R22", cutoff)
    if not after_events["saved"]:
        # A verified existing bundle was adopted without a new fit; make that
        # fact explicit in the joint ledger for later accounting. This also
        # reconciles a complete bundle written immediately before interruption.
        append_event(LEDGER_PATH, {
            "event": "CONTINUATION_FIT_SAVED", "at_utc": _now(),
            "experiment_id": f"{CAMPAIGN_ID}_R22", "card_id": "R22", "cutoff": int(cutoff),
            "predictor_fit_count": 1, "reconciled_verified_bundle": True,
            "had_prior_started_event": before_events["started"],
            "resource_check": resource,
            "npz_sha256": file_sha256(npz_path), "sidecar_sha256": file_sha256(sidecar_path),
        })
    return {"card_id": "R22", "cutoff": int(cutoff), "fit_status": record["status"], "record": record}


def _write_or_match(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        if _read_json(path) != dict(value):
            raise BatchIntegrityError(f"Existing campaign artifact differs; refusing overwrite: {path}")
    else:
        write_json_exclusive(path, dict(value))


def _campaign_summary(result: Mapping[str, Any]) -> str:
    c16_r = result.get("classification", {})
    r22_r = result.get("regression", {})
    lines = [
        "# Final continuation campaign checkpoint",
        "",
        f"- Status: `{result['status']}`",
        f"- Scope: C16 and R22 only; eight planned new predictor fits; R22_PCT_PRESENT adds no fit.",
        f"- Global budget: {result['budget']['reserved_total_after']} / 180 reserved; {result['budget']['remaining_after']} remain.",
        "- Test data and Kaggle were not accessed.",
        "",
        "## Classification C16",
        "",
        f"- Qualification: `{c16_r.get('qualification', {}).get('qualified', 'INCOMPLETE')}`",
        f"- Primary Macro F1 gain: {c16_r.get('views', {}).get('primary', {}).get('gain_positive_is_better', 'INCOMPLETE')}",
        f"- H12 Macro F1 gain: {c16_r.get('views', {}).get('H12', {}).get('gain_positive_is_better', 'INCOMPLETE')}",
        "",
        "## Regression R22",
        "",
        f"- Selected policy: `{r22_r.get('selected_policy', 'INCOMPLETE')}`",
        f"- Selection reason: {r22_r.get('selection_reason', 'Continuation results incomplete.')}",
        "",
        "Detailed per-window metrics, group guards, fit receipts, and independent verification are in the task-specific `C16/` and `R22/` result reports.",
        "",
    ]
    return "\n".join(lines)


def run_campaign() -> dict[str, Any]:
    c16_context, r22_context = _validate_freezes_and_contexts()
    tests = _run_tests()
    amendment = _ensure_amendment(c16_context, r22_context)
    amendment_signature = amendment["signature"]
    r22_context["scope_amendment_signature"] = amendment_signature
    existing_preflight = _read_json(PREFLIGHT_PATH) if PREFLIGHT_PATH.is_file() else None
    if (existing_preflight is None
            or existing_preflight.get("scope_amendment_signature") != amendment_signature
            or existing_preflight.get("c16_freeze_signature") != c16_context["freeze_signature"]
            or existing_preflight.get("r22_freeze_signature") != r22_context["signature"]):
        raise BatchIntegrityError("Combined preflight receipt is missing or differs from frozen continuation")
    _reservation_check(_budget_state())
    new_records: dict[int, dict[str, Any]] = {}
    run_receipts: list[dict[str, Any]] = []
    fit_errors: list[dict[str, Any]] = []
    # All candidate work is serialized under the shared lock; each cutoff gets
    # a fresh RAM check immediately before execution.
    with exclusive_training_slot(GLOBAL_LOCK):
        before = _budget_state()
        _reservation_check(before)
        reservation = reserve_card_budget(
            LEDGER_PATH, budget_key=JOINT_KEY, card_id=JOINT_CARD,
            predictor_fits=JOINT_FITS, cap=180,
        )
        after_reserve = _budget_state()
        if (not after_reserve["joint_reservation_present"]
                or after_reserve["joint_reservation"] != (JOINT_FITS, JOINT_CARD)):
            raise BatchIntegrityError("Atomic combined fit reservation did not verify after append")
        append_event(LEDGER_PATH, {
            "event": "FINAL_CONTINUATION_EXECUTION_STARTED", "at_utc": _now(),
            "experiment_id": CAMPAIGN_ID, "scope_amendment_signature": amendment_signature,
            "reservation": reservation, "budget_before": before, "budget_after_reservation": after_reserve,
            "tests_passed": True, "c16_freeze_signature": c16_context["freeze_signature"],
            "r22_freeze_signature": r22_context["signature"],
        })
        # Sequential, fixed order. A failure is recorded and does not authorize
        # opening any other experiment; all remaining approved cutoffs continue.
        for cutoff in FIT_CUTOFFS:
            try:
                c16_out = _run_c16_cutoff(c16_context, cutoff, amendment_signature)
                new_records[cutoff] = c16_out["record"]
                run_receipts.append({k: v for k, v in c16_out.items() if k != "record"})
            except Exception as exc:
                fit_errors.append({"card_id": "C16", "cutoff": cutoff,
                                   "error_type": type(exc).__name__, "error": str(exc)[:1200]})
        for cutoff in FIT_CUTOFFS:
            try:
                r22_out = _run_r22_cutoff(r22_context, cutoff, amendment_signature)
                run_receipts.append({k: v for k, v in r22_out.items() if k != "record"})
            except Exception as exc:
                fit_errors.append({"card_id": "R22", "cutoff": cutoff,
                                   "error_type": type(exc).__name__, "error": str(exc)[:1200]})
        append_event(LEDGER_PATH, {
            "event": "FINAL_CONTINUATION_EXECUTION_FINISHED", "at_utc": _now(),
            "experiment_id": CAMPAIGN_ID, "scope_amendment_signature": amendment_signature,
            "fit_errors": fit_errors, "run_receipts": run_receipts,
            "budget_after_fits": _budget_state(),
        })

    c16_result: dict[str, Any] | None = None
    c16_verification: dict[str, Any] | None = None
    try:
        c16_records = {cutoff: dict(c16_context["records"][cutoff]) for cutoff in c16.PILOT_CUTOFFS}
        c16_records.update(new_records)
        if set(c16_records) == set(c16.ALL_ORIGINS):
            c16_result = c16.build_report(c16_context["frame"], c16_context["target"],
                                          c16_records, c16_context["freeze_signature"])
            c16_verification = c16.verify_report_independently(
                c16_context["frame"], c16_context["target"], c16_records,
                c16_result, c16_context["freeze_signature"],
            )
            c16_result["fit_accounting"] = {
                "candidate_predictor_fits_represented": 4,
                "reused_pilot_fit_count": 2,
                "fit_seconds_by_origin": {
                    str(c): float(c16_records[c].get("metadata", {}).get("fit_seconds", 0.0))
                    for c in FIT_CUTOFFS
                },
                "attempt_history_path": str(c16.EVENT_PATH.relative_to(ROOT)).replace("\\", "/"),
            }
            _write_or_match(c16.RESULT_PATH, c16_result)
            _write_or_match(c16.VERIFICATION_PATH, c16_verification)
    except Exception as exc:
        fit_errors.append({"card_id": "C16", "stage": "independent_report", "error_type": type(exc).__name__, "error": str(exc)[:1200]})

    r22_result: dict[str, Any] | None = None
    r22_verification: dict[str, Any] | None = None
    try:
        r22_result, r22_verification = r22.publish_result(r22_context)
    except Exception as exc:
        fit_errors.append({"card_id": "R22", "stage": "independent_report", "error_type": type(exc).__name__, "error": str(exc)[:1200]})

    all_new_bundles = all(_bundle_state(card, cutoff) == (True, True)
                          for card in ("C16", "R22") for cutoff in FIT_CUTOFFS)
    status = "CONTINUATION_RESULTS_READY_FOR_REVIEW" if all_new_bundles and not fit_errors else "CONTINUATION_PARTIAL_REVIEW_BLOCKED"
    final_budget = _budget_state()
    result = {
        "schema_version": 1,
        "experiment_id": CAMPAIGN_ID,
        "status": status,
        "source_head_before_fit": _read_json(c16.FREEZE_PATH).get("source_head_contextual"),
        "scope_amendment_signature": amendment_signature,
        "c16_freeze_signature": c16_context["freeze_signature"],
        "r22_freeze_signature": r22_context["signature"],
        "new_fit_scope": {"C16": list(FIT_CUTOFFS), "R22": list(FIT_CUTOFFS), "total_planned": JOINT_FITS,
                          "R22_PCT_PRESENT_extra_fits": 0},
        "run_receipts": run_receipts,
        "fit_errors": fit_errors,
        "classification": c16_result,
        "regression": r22_result,
        "tests": tests,
        "budget": {
            "reserved_total_after": int(final_budget["reserved_total"]),
            "remaining_after": int(180 - final_budget["reserved_total"]),
            "reservation_count_after": int(final_budget["reservation_count"]),
            "joint_reservation_key": JOINT_KEY,
            "joint_reservation": final_budget.get("joint_reservation"),
            "ledger_sha256_after": final_budget.get("ledger_sha256"),
        },
        "official_test_access": False,
        "sample_submission_access": False,
        "kaggle_calls": 0,
    }
    verification = {
        "status": "PASS" if status == "CONTINUATION_RESULTS_READY_FOR_REVIEW" else "PARTIAL",
        "all_eight_new_bundles_present_and_hash_verified": all_new_bundles,
        "C16_independent_verification": c16_verification,
        "R22_independent_verification": r22_verification,
        "official_test_access": False,
        "sample_submission_access": False,
        "kaggle_calls": 0,
        "fit_errors": fit_errors,
    }
    if status == "CONTINUATION_RESULTS_READY_FOR_REVIEW":
        _write_or_match(CAMPAIGN_RESULT_PATH, result)
        _write_or_match(CAMPAIGN_VERIFY_PATH, verification)
        summary = _campaign_summary(result)
        if CAMPAIGN_SUMMARY_PATH.exists():
            if CAMPAIGN_SUMMARY_PATH.read_text(encoding="utf-8") != summary:
                raise BatchIntegrityError("Existing final continuation summary differs; refusing rewrite")
        else:
            with CAMPAIGN_SUMMARY_PATH.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(summary)
                stream.flush()
                os.fsync(stream.fileno())
    else:
        attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        partial_result_path = ROOT / f"reports/optimization_v3/final_continuation/partial_checkpoint_{attempt}.json"
        partial_verify_path = ROOT / f"reports/optimization_v3/final_continuation/partial_verification_{attempt}.json"
        _write_or_match(partial_result_path, result)
        _write_or_match(partial_verify_path, verification)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight-only", action="store_true")
    group.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    result = preflight_only() if args.preflight_only else run_campaign()
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
