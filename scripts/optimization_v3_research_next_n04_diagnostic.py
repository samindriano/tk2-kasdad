"""Freeze and run the separately authorized N04_DIAGNOSTIC amendment."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits

from scripts import optimization_v3_research_next_n00 as n00
from scripts import optimization_v3_research_next_n04 as n04
from src.models.research_next_budget import (
    finish_fit, ledger_summary, read_events, release_unused, reserve_card, start_fit,
)
from src.models.research_next_contract import (
    canonical_sha256, file_sha256, publish_bundle, read_bundle,
)
from src.models.research_next_n04 import (
    DOMAIN_AUC_RANGE, R22WeightedDesign, fit_weighted_r22_head,
    predict_r22_head, shuffle_weights_within_cycle,
)
from src.utils.paths import PROJECT_ROOT


ROOT = PROJECT_ROOT
BATCH = "optimization_v3_research_next_v1"
CARD_ID = "N04_DIAGNOSTIC"
STAGE = "DIAGNOSTIC"
PILOT_FITS = 4
THREADS = 2
JOB_CAP_BYTES = 6 * 1024**3
FREE_RESERVE_BYTES = 2 * 1024**3
MINIMUM_FREE_BYTES = 4 * 1024**3
ORIGINS = (55, 67)
PARITIES = (0, 1)
FREEZE_ROOT = ROOT / "experiments/optimization_v3/research_next_v1/frozen/N04_DIAGNOSTIC/amendment_01"
PREFLIGHT_PATH = FREEZE_ROOT / "preflight.json"
FREEZE_PATH = FREEZE_ROOT / "freeze.json"
REPORT_ROOT = ROOT / "reports/optimization_v3/research_next_v1/results/N04_DIAGNOSTIC/amendment_01"
BUNDLE_ROOT = ROOT / "experiments/runs/optimization_v3_research_next_v1/N04_DIAGNOSTIC/amendment_01"
RECOVERY_ROOT = BUNDLE_ROOT / "recovery"
LEDGER_PATH = ROOT / "experiments/runs/optimization_v3_research_next_v1/budget.jsonl"
OLD_LEDGER_PATH = ROOT / "experiments/runs/optimization_v3_batch1/batch_ledger.jsonl"
QUEUE_PATH = ROOT / "experiments/optimization_v3/research_next_v1/queue.json"
DOMAIN_GROUPS = ((55, 0), (55, 1), (67, 0), (67, 1))


class ResourceDeferred(RuntimeError):
    """The current resource state fails the frozen per-fit check."""


def _relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _is_ancestor(commit: str) -> bool:
    return subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"],
                          cwd=ROOT, check=False).returncode == 0


def _dependencies() -> dict[str, str]:
    names = ("numpy", "pandas", "scipy", "scikit-learn", "xgboost", "psutil",
             "matplotlib", "threadpoolctl")
    return {"python": platform.python_version(), "platform": platform.platform(),
            **{name: importlib.metadata.version(name) for name in names}}


def _source_paths() -> list[Path]:
    return [ROOT / relative for relative in (
        "scripts/optimization_v3_research_next_n00.py",
        "scripts/optimization_v3_research_next_n04.py",
        "scripts/optimization_v3_research_next_n04_diagnostic.py",
        "scripts/optimization_v3_research_next_n04_diagnostic_verify.py",
        "src/models/research_next_contract.py",
        "src/models/research_next_budget.py",
        "src/models/research_next_n04.py",
        "src/models/round1_models.py",
        "src/models/classification_models.py",
        "src/models/regression_research_text.py",
        "src/models/optimization_v3_batch1_r22.py",
        "src/utils/paths.py",
        "tests/test_research_next_n04_diagnostic.py",
    )]


def _n04_domain_paths(cutoff: int, parity: int) -> tuple[Path, Path, Path, Path, dict[str, Any]]:
    if cutoff == 55 and parity == 0:
        old_freeze = json.loads(n04.PREVIOUS_FREEZE_PATH.read_text(encoding="utf-8"))
        bundle_path = n04.PREVIOUS_BUNDLE_ROOT / "fit55" / "domain_even"
        recovery_npz = n04.PREVIOUS_RECOVERY_ROOT / "N04_PILOT_55_domain_even.npz"
        recovery_json = n04.PREVIOUS_RECOVERY_ROOT / "N04_PILOT_55_domain_even.json"
        signature = old_freeze["freeze_signature"]
    else:
        bundle_path = n04._bundle_path(cutoff, "domain", parity)
        recovery_npz, recovery_json = n04._recovery_paths(n04._fit_id("domain", cutoff, parity))
        signature = json.loads(n04.FREEZE_PATH.read_text(encoding="utf-8"))["freeze_signature"]
    identity = n04._bundle_identity(signature, cutoff, "domain", parity)
    verification = json.loads(n04.REPORT_ROOT.joinpath("verification.json").read_text(encoding="utf-8"))
    reported = next(item for item in verification["cutoffs"][str(cutoff)]["folds"]
                    if item["parity"] == ("odd" if parity else "even"))
    return bundle_path, recovery_npz, recovery_json, Path(n04.REPORT_ROOT / "verification.json"), {
        "identity": identity, "reported_verification": reported,
    }


def _load_verified_domains(context: Mapping[str, Any], frozen_n04: Mapping[str, Any]) -> tuple[
    dict[int, list[Any]], dict[int, dict[str, Any]], dict[str, Any]
]:
    original_results_path = n04.REPORT_ROOT / "results.json"
    original_verification_path = n04.REPORT_ROOT / "verification.json"
    original_results = json.loads(original_results_path.read_text(encoding="utf-8"))
    original_verification = json.loads(original_verification_path.read_text(encoding="utf-8"))
    if (original_results.get("status") != "SKIPPED_SCIENTIFIC_PREFLIGHT"
            or original_results.get("views") or original_verification.get("verdict") != "SKIPPED_SCIENTIFIC_PREFLIGHT"):
        raise RuntimeError("N04 original skipped verdict or no-task-fit checkpoint has changed")
    if original_verification.get("freeze_signature") != frozen_n04["freeze_signature"]:
        raise RuntimeError("N04 original verification does not match its frozen identity")
    by_cutoff: dict[int, list[Any]] = {cutoff: [] for cutoff in ORIGINS}
    audit: dict[str, Any] = {}
    original_events = read_events(LEDGER_PATH)
    for cutoff, parity in DOMAIN_GROUPS:
        bundle_path, recovery_npz, recovery_json, _, extra = _n04_domain_paths(cutoff, parity)
        bundle = read_bundle(bundle_path, extra["identity"], task="classification")
        checked = n04._independent_domain(bundle, context["frame"], cutoff, parity)
        record = extra["reported_verification"]
        if (file_sha256(bundle_path / "arrays.npz") != record["bundle_npz_sha256"]
                or bundle.receipt["receipt_sha256"] != record["bundle_receipt_sha256"]
                or not np.isclose(checked["auc"], record["auc_pairwise"], rtol=0, atol=1e-14)):
            raise RuntimeError(f"N04 domain {cutoff}/{parity} differs from its accepted evidence")
        if not recovery_npz.is_file() or not recovery_json.is_file():
            raise RuntimeError(f"N04 domain {cutoff}/{parity} recovery payload is absent")
        recovery_receipt = json.loads(recovery_json.read_text(encoding="utf-8"))
        if (recovery_receipt.get("identity") != extra["identity"]
                or recovery_receipt.get("npz_sha256") != file_sha256(recovery_npz)):
            raise RuntimeError(f"N04 domain {cutoff}/{parity} recovery identity/hash differs")
        with np.load(recovery_npz, allow_pickle=False) as archive:
            recovery_arrays = {key: archive[key].copy() for key in archive.files}
        if (set(recovery_arrays) != set(bundle.arrays)
                or any(not np.array_equal(recovery_arrays[key], bundle.arrays[key]) for key in bundle.arrays)):
            raise RuntimeError(f"N04 domain {cutoff}/{parity} recovery arrays differ from accepted bundle")
        fit_id = n04._fit_id("domain", cutoff, parity)
        events = [event for event in original_events if event.get("fit_id") == fit_id]
        if (sum(event.get("event") == "FIT_STARTED" for event in events) != 1
                or sum(event.get("event") == "FIT_SAVED" for event in events) != 1
                or any(event.get("event") == "FIT_FAILED" for event in events)):
            raise RuntimeError(f"N04 original ledger is missing the saved domain fit {fit_id}")
        by_cutoff[cutoff].append(bundle)
        audit[f"{cutoff}:{'odd' if parity else 'even'}"] = {
            "bundle_path": _relative(bundle_path), "bundle_receipt_sha256": bundle.receipt["receipt_sha256"],
            "bundle_npz_sha256": file_sha256(bundle_path / "arrays.npz"),
            "recovery_npz_sha256": file_sha256(recovery_npz),
            "recovery_receipt_sha256": file_sha256(recovery_json),
            "fit_id": fit_id, "heldout_auc": checked["auc"],
            "fit_rows": int(bundle.receipt["metadata"]["diagnostics"]["fit_rows"]),
            "score_rows": int(len(bundle.arrays["score_rows"])),
            "recovery_equal_to_bundle": True,
        }
    oof = {cutoff: n04._domain_oof(context, cutoff, by_cutoff[cutoff]) for cutoff in ORIGINS}
    pre_task = n04._pre_task_gates(oof)
    original_pre_task = original_results.get("pre_task_gates")
    if canonical_sha256(pre_task) != canonical_sha256(original_pre_task):
        raise RuntimeError("N04 original pre-task gates differ from frozen domain fit replay")
    for cutoff in ORIGINS:
        saved = original_results["domain_oof"][str(cutoff)]
        current = oof[cutoff]
        if (saved["weight_diagnostics"].get("weight_sha256")
                != current["weight_diagnostics"].get("weight_sha256")
                or not np.array_equal(np.asarray(saved["heldout_fold_aucs"]),
                                      np.asarray(current["fold_aucs"]))):
            raise RuntimeError(f"N04 reconstructed OOF probabilities/weights changed at {cutoff}")
    if (pre_task["status"] != "SKIPPED_SCIENTIFIC_PREFLIGHT"
            or pre_task["by_cutoff"]["55"]["auc_gate"] is not False
            or pre_task["by_cutoff"]["55"]["ess_gate"] is not True
            or pre_task["by_cutoff"]["55"]["source_clip_fraction_gate"] is not True
            or pre_task["by_cutoff"]["67"]["auc_gate"] is not True
            or pre_task["by_cutoff"]["67"]["ess_gate"] is not True
            or pre_task["by_cutoff"]["67"]["source_clip_fraction_gate"] is not True):
        raise RuntimeError("N04 amendment preconditions no longer match the authorized exception")
    return by_cutoff, oof, {"pre_task_gates": pre_task, "domain_fit_audit": audit,
                            "original_results_sha256": file_sha256(original_results_path),
                            "original_verification_sha256": file_sha256(original_verification_path)}


def _input_paths(train_path: Path, domain_audit: Mapping[str, Any]) -> list[Path]:
    paths = [QUEUE_PATH, n04.CARD_PATH, n04.QUEUE_PATH, n04.READINESS_PATH,
             n04.N00_FREEZE_PATH, n04.N00_SUMMARY_PATH, OLD_LEDGER_PATH, train_path,
             n04.PREVIOUS_FREEZE_PATH, n04.PREVIOUS_PREFLIGHT_PATH,
             n04.FREEZE_PATH, n04.PREFLIGHT_PATH,
             n04.REPORT_ROOT / "results.json", n04.REPORT_ROOT / "verification.json"]
    for cutoff in ORIGINS:
        for card in ("R22", "R06"):
            path = n00.cache_path(card, cutoff)
            paths.extend((path, path.with_suffix(".json")))
    for audit in domain_audit.values():
        bundle = ROOT / audit["bundle_path"]
        paths.extend((bundle / "arrays.npz", bundle / "receipt.json"))
        # Recovery paths are verified in the current N04 evidence and hashed in the same audit.
        for candidate in (n04.PREVIOUS_RECOVERY_ROOT, n04.RECOVERY_ROOT):
            for suffix in ("npz", "json"):
                p = candidate / f"{audit['fit_id'].replace(':', '_')}.{suffix}"
                if p.is_file():
                    paths.append(p)
    # The inherited first fold uses named recovery files in amendment_01.
    paths.extend((n04.PREVIOUS_RECOVERY_ROOT / "N04_PILOT_55_domain_even.npz",
                  n04.PREVIOUS_RECOVERY_ROOT / "N04_PILOT_55_domain_even.json"))
    return list(dict.fromkeys(paths))


def _resource_bounds(context: Mapping[str, Any]) -> dict[str, Any]:
    return {str(cutoff): {
        key: value for key, value in context["resources"][str(cutoff)].items()
        if key not in {"available_physical_bytes", "allowed_bytes", "status"}
    } for cutoff in ORIGINS}


def _build_identity(context: Mapping[str, Any], train_path: Path,
                    domain_audit: Mapping[str, Any], pre_task: Mapping[str, Any]) -> dict[str, Any]:
    queue = json.loads(QUEUE_PATH.read_text(encoding="utf-8"))
    n04_frozen = json.loads(n04.FREEZE_PATH.read_text(encoding="utf-8"))
    n04_results = json.loads((n04.REPORT_ROOT / "results.json").read_text(encoding="utf-8"))
    n04_verification = json.loads((n04.REPORT_ROOT / "verification.json").read_text(encoding="utf-8"))
    if (queue.get("batch_id") != BATCH or n04_results.get("status") != "SKIPPED_SCIENTIFIC_PREFLIGHT"
            or n04_verification.get("verdict") != "SKIPPED_SCIENTIFIC_PREFLIGHT"
            or n04_frozen.get("freeze_signature") != canonical_sha256(n04_frozen.get("identity"))):
        raise RuntimeError("N04_DIAGNOSTIC original checkpoint or queue is not intact")
    n04_commit = n04_frozen.get("execution_context", {}).get("source_commit")
    if not n04_commit or not _is_ancestor(n04_commit):
        raise RuntimeError("Original N04 frozen source commit is not an ancestor")
    paths = _input_paths(train_path, domain_audit)
    sources = _source_paths()
    missing = [str(path) for path in (*paths, *sources) if not path.is_file()]
    if missing:
        raise RuntimeError(f"N04_DIAGNOSTIC freeze inputs/sources missing: {missing}")
    if file_sha256(train_path) != queue["raw_train_sha256"]["regression"]:
        raise RuntimeError("N04_DIAGNOSTIC Regression Train hash differs from canonical queue")
    input_hashes = {_relative(path): file_sha256(path) for path in paths}
    source_hashes = {_relative(path): file_sha256(path) for path in sources}
    return {
        "batch_id": BATCH, "card_id": CARD_ID, "amendment": "amendment_01",
        "status_class": "diagnostic_only_not_eligible_for_continuation",
        "question": "Apakah density weighting tetap memberi predictive benefit ketika domain AUC sedikit di bawah gate 0,55, tetapi ESS dan clipping memenuhi syarat?",
        "post_auc_pre_task_outcome_amendment": True,
        "task_mae_seen_before_amendment": False,
        "original_N04_status_preserved": "SKIPPED_SCIENTIFIC_PREFLIGHT",
        "original_N04_freeze_signature": n04_frozen["freeze_signature"],
        "original_N04_results_sha256": file_sha256(n04.REPORT_ROOT / "results.json"),
        "original_N04_verification_sha256": file_sha256(n04.REPORT_ROOT / "verification.json"),
        "recipe": {
            "reused_domain_fit_count": 4,
            "reused_domain_bundle_lineage": dict(domain_audit),
            "pre_task_domain_result": dict(pre_task),
            "domain_auc_gate_inherited_unchanged": list(DOMAIN_AUC_RANGE),
            "recipe_source": "N04 original frozen amendment_02 exactly; no domain/head recipe or gate changes",
            "task_model": {"estimator": "xgboost.XGBRegressor", "parameters": dict(n04.XGB_PARAMETERS),
                           "features": list(n04.HYBRID_FEATURES),
                           "preprocessing": "exact R22 structured/OHE + TF-IDF/SVD32, fit once per origin",
                           "objective": "reg:pseudohubererror", "sample_weight": "original N04 density ratios"},
            "weighted_arm": "fit N04 original weighted task estimator at cutoff 55 and 67",
            "matched_control": "fit same learner/design with N04 original within-cycle shuffled weights; seed 42+cutoff",
            "rank_and_policy": ".75 protected legacy champion + .25 within-complete-score-cycle raw rank50; average ties; clip [0,100]",
            "fit_population": "all official Regression Train rows with cycle_code <= outer cutoff",
            "evaluation": "primary W1/W4 pooled, H12 W1-W6 pooled, every 4-cycle window, missing and evaluation-only actual target tails; compare R22, shuffled control, protected legacy and R06 secondary",
            "gate_changes": False,
            "automatic_continuation_eligibility": False,
        },
        "pilot_origins": list(ORIGINS), "pilot_fit_count": PILOT_FITS,
        "input_and_source_sha256": {**input_hashes, **source_hashes},
        "resource_bounds": _resource_bounds(context),
        "dependencies": _dependencies(),
    }


def prepare_context(*, available_bytes: int | None = None,
                    ledger_override: Mapping[str, Any] | None = None) -> dict[str, Any]:
    frozen_n04, context = n04._verify_freeze()
    domains, oof, audit = _load_verified_domains(context, frozen_n04)
    train_path = context["path"]
    identity = _build_identity(context, train_path, audit["domain_fit_audit"], audit["pre_task_gates"])
    resources = {
        str(cutoff): n04._resource_estimate(context["frame"], cutoff, available_bytes)
        for cutoff in ORIGINS
    }
    if ledger_override is None:
        budget_context = {"ledger_sha256": file_sha256(LEDGER_PATH), "ledger": ledger_summary(LEDGER_PATH)}
    else:
        budget_context = dict(ledger_override)
    execution = {
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(), "source_commit": _git_head(),
        "available_bytes": int(psutil.virtual_memory().available if available_bytes is None else available_bytes),
        **budget_context, "original_N04_task_mae_seen_before_fit": False,
        "test_files_opened": False, "test_labels_read": False, "kaggle_calls": 0,
        "reused_domain_bundles_verified": 4,
        "resource_checks": resources,
    }
    return {
        "identity": identity, "freeze_signature": canonical_sha256(identity),
        "execution_context": execution, "n04_frozen": frozen_n04,
        "context": context, "domains": domains, "oof": oof, "audit": audit,
        "resources": resources,
    }


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        if canonical_sha256(current) != canonical_sha256(value):
            raise RuntimeError(f"N04_DIAGNOSTIC refuses to overwrite immutable output: {_relative(path)}")
        return
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _amendment_slot_audit() -> dict[str, Any]:
    events = read_events(LEDGER_PATH)
    reservations = [event for event in events if event.get("event") == "CARD_FITS_RESERVED"]
    releases = [event for event in events if event.get("event") == "RESERVATION_UNUSED_RELEASED"]
    released_by_key = {event.get("reservation_key"): int(event.get("unused_fits", 0))
                       for event in releases}
    n04_reservation = [event for event in reservations if event.get("reservation_key") == "N04:PILOT"]
    n04_release = [event for event in releases if event.get("reservation_key") == "N04:PILOT"]
    n04_starts = [event for event in events if event.get("event") == "FIT_STARTED"
                  and event.get("reservation_key") == "N04:PILOT"]
    n04_saves = [event for event in events if event.get("event") == "FIT_SAVED"
                 and event.get("reservation_key") == "N04:PILOT"]
    n04_failures = [event for event in events if event.get("event") == "FIT_FAILED"
                    and event.get("reservation_key") == "N04:PILOT"]
    expected_n04_ids = {n04._fit_id("domain", cutoff, parity)
                        for cutoff in ORIGINS for parity in PARITIES}
    n05_reservation = [event for event in reservations if event.get("reservation_key") == "N05:PILOT"]
    n05_release = [event for event in releases if event.get("reservation_key") == "N05:PILOT"]
    n05_events = [event for event in events if event.get("reservation_key") == "N05:PILOT"]
    n05_starts = [event for event in n05_events if event.get("event") == "FIT_STARTED"]
    n05_saves = [event for event in n05_events if event.get("event") == "FIT_SAVED"]
    n05_failures = [event for event in n05_events if event.get("event") == "FIT_FAILED"]
    expected_n05_ids = {f"N05:PILOT:{cutoff}:{component}"
                        for cutoff in (67, 75) for component in ("q10", "q90")}
    n06_starts = [event for event in events if event.get("card_id") == "N06"
                  and event.get("event") == "FIT_STARTED"]
    amendment_reservation = [event for event in reservations
                             if event.get("reservation_key") == f"{CARD_ID}:{STAGE}"]
    amendment_starts = [event for event in events if event.get("reservation_key") == f"{CARD_ID}:{STAGE}"
                        and event.get("event") == "FIT_STARTED"]
    amendment_failures = [event for event in events if event.get("reservation_key") == f"{CARD_ID}:{STAGE}"
                          and event.get("event") == "FIT_FAILED"]
    allowed_ids = {f"{CARD_ID}:{STAGE}:{cutoff}:{component}"
                   for cutoff in ORIGINS for component in ("weighted", "within_cycle_shuffle")}
    summary = ledger_summary(LEDGER_PATH)
    auth = next(event for event in events if event.get("event") == "BATCH_ALLOWANCE_AUTHORIZED")
    active_by_category = {}
    active_total = 0
    for event in reservations:
        active = int(event["fits"]) - released_by_key.get(event["reservation_key"], 0)
        active_total += active
        active_by_category[event["category"]] = active_by_category.get(event["category"], 0) + active
    checks = {
        "original_N04_reserved_eight": len(n04_reservation) == 1 and int(n04_reservation[0]["fits"]) == 8,
        "original_N04_four_unused_slots_released_once": (
            len(n04_release) == 1 and int(n04_release[0]["unused_fits"]) == 4
        ),
        "original_N04_four_domain_fits_saved_without_failure": (
            {event["fit_id"] for event in n04_starts} == expected_n04_ids
            and {event["fit_id"] for event in n04_saves} == expected_n04_ids
            and not n04_failures
        ),
        "N05_finished_before_amendment": (
            len(n05_reservation) == 1 and int(n05_reservation[0]["fits"]) == 4
            and len(n05_release) == 1 and int(n05_release[0]["unused_fits"]) == 0
            and {event["fit_id"] for event in n05_starts} == expected_n05_ids
            and {event["fit_id"] for event in n05_saves} == expected_n05_ids
            and not n05_failures
        ),
        "N06_not_started": not n06_starts,
        "amendment_reservation_unique_and_exact": (
            len(amendment_reservation) <= 1
            and (not amendment_reservation or (
                int(amendment_reservation[0]["fits"]) == PILOT_FITS
                and amendment_reservation[0]["category"] == "pilot"
            ))
        ),
        "amendment_fit_ids_unique_and_within_allowance": (
            len({event["fit_id"] for event in amendment_starts}) == len(amendment_starts)
            and {event["fit_id"] for event in amendment_starts} <= allowed_ids
            and len(amendment_starts) <= PILOT_FITS
            and not amendment_failures
        ),
        "phase_total_cap_preserved": active_total <= int(auth["cap"]) == 60,
        "pilot_category_cap_preserved": active_by_category.get("pilot", 0) <= int(auth["category_caps"]["pilot"]) == 26,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"N04_DIAGNOSTIC budget/order audit failed: {failed}")
    new_reservation_needed = not amendment_reservation
    return {
        "checks": checks,
        "released_slot_source": "N04:PILOT RESERVATION_UNUSED_RELEASED unused_fits=4",
        "new_reservation_key": f"{CARD_ID}:{STAGE}",
        "new_reservation_fits": PILOT_FITS,
        "new_reservation_needed": new_reservation_needed,
        "amendment_started_fit_count": len(amendment_starts),
        "amendment_started_fit_ids": [event["fit_id"] for event in amendment_starts],
        "previous_release_event_preserved": True,
        "total_and_pilot_active_reserved_before_new_reservation": {
            "total": active_total, "pilot": active_by_category.get("pilot", 0),
        },
        "ledger": summary,
    }


def _payload_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {**payload["arrays"], **payload["metadata"]}


def _fit_id(cutoff: int, component: str) -> str:
    return f"{CARD_ID}:{STAGE}:{cutoff}:{component}"


def _bundle_identity(signature: str, cutoff: int, component: str) -> dict[str, Any]:
    return {"batch_id": BATCH, "card_id": CARD_ID, "freeze_signature": signature,
            "stage": STAGE, "cutoff": int(cutoff), "component": component}


def _bundle_path(cutoff: int, component: str, *, recovery: bool = False) -> Path:
    root = RECOVERY_ROOT if recovery else BUNDLE_ROOT
    return root / f"fit{cutoff}" / component


def _resource_check(estimate: int, label: str) -> dict[str, Any]:
    available = int(psutil.virtual_memory().available)
    allowed = min(JOB_CAP_BYTES, available - FREE_RESERVE_BYTES)
    result = {"label": label, "available_bytes": available, "estimated_bytes": int(estimate),
              "allowed_bytes": max(0, allowed), "minimum_free_bytes": MINIMUM_FREE_BYTES,
              "reserve_bytes": FREE_RESERVE_BYTES, "job_cap_bytes": JOB_CAP_BYTES,
              "threads": THREADS, "concurrent_jobs": 1}
    if available < MINIMUM_FREE_BYTES or estimate > allowed:
        raise ResourceDeferred(f"N04_DIAGNOSTIC {label} resource gate failed: {result}")
    return result


def _synthetic_smoke() -> dict[str, Any]:
    rng = np.random.default_rng(20261102)
    x = rng.normal(size=(128, 8)).astype(np.float64)
    y = np.clip(50 + 7 * x[:, 0] - 3 * x[:, 1] + rng.normal(scale=4, size=128), 0, 100)
    cycles = np.repeat(np.arange(4, dtype=np.int64), 32)
    weights = np.linspace(.6, 1.4, len(y), dtype=np.float64)
    weights /= weights.mean()
    shuffled = shuffle_weights_within_cycle(weights, cycles, seed=97)
    if not np.isclose(shuffled.mean(), 1.0) or not np.array_equal(np.sort(weights), np.sort(shuffled)):
        raise RuntimeError("N04_DIAGNOSTIC synthetic within-cycle shuffle failed")
    with threadpool_limits(limits=THREADS):
        model, state = fit_weighted_r22_head(x, y, weights, cutoff=55)
        raw = predict_r22_head(model, x[:20])
    cycles_score = np.repeat(np.arange(2, dtype=np.int64), 10)
    legacy = np.full(20, 50.0, dtype=np.float64)
    prediction = n04._rank50_independent(raw, cycles_score)
    prediction = np.clip(.75 * legacy + .25 * prediction, 0, 100)
    if (state["sample_weight_passed_to_fit"] is not True or len(prediction) != 20
            or not np.isfinite(prediction).all() or np.any((prediction < 0) | (prediction > 100))):
        raise RuntimeError("N04_DIAGNOSTIC synthetic weighted fit/composition failed")
    with tempfile.TemporaryDirectory(prefix="kasdad_n04d_contract_") as temporary:
        destination = Path(temporary) / "synthetic_bundle"
        identity = {"batch_id": BATCH, "card_id": CARD_ID, "synthetic": True}
        payload = {
            "score_rows": np.arange(20, 25, dtype=np.int64),
            "score_record_ids": np.asarray([f"S{i}" for i in range(5)], dtype="U2"),
            "prediction": np.asarray([45., 48., 50., 52., 55.]),
            "baseline_prediction": np.asarray([46., 49., 51., 53., 56.]),
            "raw_component": np.asarray([44., 47., 49., 51., 54.]),
            "diagnostics": {"synthetic_only": True},
        }
        publish_bundle(destination, payload, identity, task="regression")
        first = read_bundle(destination, identity, task="regression")
        resumed = read_bundle(destination, identity, task="regression")
        if (first.receipt["receipt_sha256"] != resumed.receipt["receipt_sha256"]
                or not np.array_equal(first.arrays["prediction"], resumed.arrays["prediction"])):
            raise RuntimeError("N04_DIAGNOSTIC synthetic persistence/resume failed")
    return {
        "status": "PASS", "official_task_fits": 0, "synthetic_predictor_fits": 1,
        "synthetic_weighted_estimator": state["estimator"],
        "synthetic_weighted_state": state["state_sha256"],
        "within_cycle_multiset_preserved": True,
        "rank50_and_blend_finite_bounded": True,
        "persistence_readback_resume": "PASS",
    }


def preflight() -> dict[str, Any]:
    smoke = _synthetic_smoke()
    first = prepare_context()
    second = prepare_context(
        available_bytes=max(MINIMUM_FREE_BYTES, first["execution_context"]["available_bytes"] - 1_234_567),
        ledger_override={"ledger_sha256": "0" * 64, "ledger": {"simulated": True}},
    )
    if first["freeze_signature"] != second["freeze_signature"]:
        raise RuntimeError("N04_DIAGNOSTIC stable identity depends on live RAM or ledger state")
    resources_pass = all(item["status"] == "PASS" for item in first["resources"].values())
    slot_audit = _amendment_slot_audit()
    budget = slot_audit["ledger"]
    new_reservation_fits = PILOT_FITS if slot_audit["new_reservation_needed"] else 0
    remaining_amendment_fits = PILOT_FITS - slot_audit["amendment_started_fit_count"]
    if (budget["authorized_cap"] != 60
            or budget["started_and_consumed"] + remaining_amendment_fits > 60
            or budget["active_reserved"] + new_reservation_fits > 60
            or budget["category_reservations"]["pilot"] + new_reservation_fits > 26):
        raise RuntimeError("N04_DIAGNOSTIC four-fit reservation exceeds the authorized phase cap")
    result = {
        "status": "READY_TO_FREEZE" if resources_pass else "DEFERRED_RESOURCE",
        "freeze_signature": first["freeze_signature"],
        "original_N04_status_preserved": "SKIPPED_SCIENTIFIC_PREFLIGHT",
        "original_N04_pre_task_gates": first["audit"]["pre_task_gates"],
        "reused_domain_fits": first["audit"]["domain_fit_audit"],
        "resources": first["resources"], "budget_before_amendment": budget,
        "budget_release_reuse_audit": slot_audit,
        "new_task_estimator_fits_started": 0,
        "stable_identity_repeat_check": "PASS", "synthetic_end_to_end": smoke,
        "task_mae_seen_before_amendment": False,
        "test_files_opened": False, "kaggle_calls": 0,
    }
    _write_once(PREFLIGHT_PATH, result)
    _write_once(REPORT_ROOT / "synthetic_smoke.json", smoke)
    return result


def freeze() -> dict[str, Any]:
    prepared = prepare_context()
    if any(item["status"] != "PASS" for item in prepared["resources"].values()):
        raise ResourceDeferred("N04_DIAGNOSTIC pre-fit resource estimate failed")
    if not PREFLIGHT_PATH.is_file():
        raise RuntimeError("N04_DIAGNOSTIC persisted preflight required before freeze")
    report = json.loads(PREFLIGHT_PATH.read_text(encoding="utf-8"))
    if report.get("status") != "READY_TO_FREEZE" or report.get("freeze_signature") != prepared["freeze_signature"]:
        raise RuntimeError("N04_DIAGNOSTIC preflight identity/readiness differs")
    execution = dict(prepared["execution_context"])
    execution["preflight_sha256"] = file_sha256(PREFLIGHT_PATH)
    payload = {"schema_version": 1, "identity": prepared["identity"],
               "freeze_signature": prepared["freeze_signature"], "execution_context": execution}
    _write_once(FREEZE_PATH, payload)
    return {"status": "FROZEN", "freeze_signature": prepared["freeze_signature"],
            "path": _relative(FREEZE_PATH), "source_commit": execution["source_commit"]}


def _verify_freeze() -> tuple[dict[str, Any], dict[str, Any]]:
    if not FREEZE_PATH.is_file():
        raise RuntimeError("N04_DIAGNOSTIC freeze is required before task fits")
    frozen = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if frozen.get("freeze_signature") != canonical_sha256(frozen.get("identity")):
        raise RuntimeError("N04_DIAGNOSTIC freeze signature is invalid")
    prepared = prepare_context()
    if prepared["identity"] != frozen.get("identity"):
        raise RuntimeError("N04_DIAGNOSTIC frozen source, recipe, or inputs changed")
    source_commit = frozen.get("execution_context", {}).get("source_commit")
    if not source_commit or not _is_ancestor(source_commit):
        raise RuntimeError("N04_DIAGNOSTIC frozen source commit is not an ancestor")
    if (not PREFLIGHT_PATH.is_file()
            or frozen["execution_context"].get("preflight_sha256") != file_sha256(PREFLIGHT_PATH)):
        raise RuntimeError("N04_DIAGNOSTIC preflight receipt differs from freeze")
    if any(item["status"] != "PASS" for item in prepared["resources"].values()):
        raise ResourceDeferred("N04_DIAGNOSTIC live resource precondition no longer passes")
    return frozen, prepared


def _reconstruct_weight(context: Mapping[str, Any], oof: Mapping[int, Mapping[str, Any]],
                        cutoff: int, component: str) -> np.ndarray:
    if component == "weighted":
        return oof[cutoff]["weights"]
    if component == "within_cycle_shuffle":
        return shuffle_weights_within_cycle(oof[cutoff]["weights"], oof[cutoff]["cycles"],
                                            seed=42 + cutoff)
    raise ValueError(f"Unknown N04_DIAGNOSTIC component {component}")


def _independent_bundle(bundle: Any, context: Mapping[str, Any],
                        frozen: Mapping[str, Any], cutoff: int, component: str,
                        weights: np.ndarray, design_hash: str | None = None) -> None:
    arrays = bundle.arrays
    diagnostics = bundle.receipt["metadata"].get("diagnostics", {})
    source = context["sources"][cutoff]
    rows = source["rows"].astype(np.int64)
    cycles = source["cycles"].astype(np.int64)
    if (bundle.receipt.get("identity") != _bundle_identity(frozen["freeze_signature"], cutoff, component)
            or not np.array_equal(arrays["score_rows"], rows)
            or not np.array_equal(arrays["score_record_ids"].astype(str), source["ids"].astype(str))
            or not np.array_equal(arrays["cycle_code"], cycles)
            or not np.array_equal(arrays["baseline_prediction"], source["R22_ORIGINAL"])
            or not np.array_equal(arrays["legacy_champion"], source["legacy"])
            or not np.array_equal(arrays["r06_original"], source["R06_ORIGINAL"])):
        raise RuntimeError(f"N04_DIAGNOSTIC {cutoff}/{component} score/cache alignment failed")
    raw = arrays["raw_component"].astype(np.float64)
    mapped = n04._rank50_independent(raw, cycles)
    prediction = np.clip(.75 * source["legacy"] + .25 * mapped, 0, 100)
    if (not np.allclose(mapped, arrays["rank50_component"], rtol=0, atol=1e-12)
            or not np.allclose(prediction, arrays["prediction"], rtol=0, atol=1e-12)):
        raise RuntimeError(f"N04_DIAGNOSTIC {cutoff}/{component} rank50/blend differs")
    fit_rows, _, fit_target = n04._fit_prefix(context, cutoff)
    if (diagnostics.get("component") != component or diagnostics.get("cutoff") != cutoff
            or diagnostics.get("fit_rows") != len(fit_rows)
            or diagnostics.get("fit_row_indices_sha256") != n04._array_hash(fit_rows, "<i8")
            or diagnostics.get("fit_target_sha256") != n04._array_hash(fit_target, "<f8")
            or diagnostics.get("fit_sample_weight_sha256") != n04._array_hash(weights, "<f8")
            or not diagnostics.get("score_target_passed_to_fit") is False
            or diagnostics.get("test_data_used") is not False):
        raise RuntimeError(f"N04_DIAGNOSTIC {cutoff}/{component} fit lineage/audit differs")
    if design_hash is not None and diagnostics.get("fit_design_sha256") != design_hash:
        raise RuntimeError(f"N04_DIAGNOSTIC {cutoff}/{component} transformed design differs")


def _fit_one(prepared: Mapping[str, Any], frozen: Mapping[str, Any], cutoff: int,
             component: str, design: R22WeightedDesign, x_fit: np.ndarray,
             x_score: np.ndarray, weights: np.ndarray) -> Any:
    context, oof = prepared["context"], prepared["oof"]
    fit_rows, _, fit_target = n04._fit_prefix(context, cutoff)
    fit_id = _fit_id(cutoff, component)
    identity = _bundle_identity(frozen["freeze_signature"], cutoff, component)
    path, recovery_path = _bundle_path(cutoff, component), _bundle_path(cutoff, component, recovery=True)
    source = context["sources"][cutoff]
    missing = n04._text_values(context["frame"].iloc[source["rows"]]["signal_stream"])[1].astype(bool)

    if path.is_dir():
        saved = read_bundle(path, identity, task="regression")
        _independent_bundle(saved, context, frozen, cutoff, component, weights,
                            design.audit_state()["fit_design_sha256"])
        finish_fit(LEDGER_PATH, fit_id=fit_id, status="FIT_SAVED",
                   receipt_sha256=saved.receipt["receipt_sha256"])
        return saved
    if recovery_path.is_dir():
        recovered = read_bundle(recovery_path, identity, task="regression")
        _independent_bundle(recovered, context, frozen, cutoff, component, weights,
                            design.audit_state()["fit_design_sha256"])
        payload = {**recovered.arrays, **recovered.receipt["metadata"]}
        publish_bundle(path, payload, identity, task="regression")
        saved = read_bundle(path, identity, task="regression")
        _independent_bundle(saved, context, frozen, cutoff, component, weights,
                            design.audit_state()["fit_design_sha256"])
        finish_fit(LEDGER_PATH, fit_id=fit_id, status="FIT_SAVED",
                   receipt_sha256=saved.receipt["receipt_sha256"])
        return saved
    if any(event.get("event") == "FIT_STARTED" and event.get("fit_id") == fit_id
           for event in read_events(LEDGER_PATH)):
        raise RuntimeError(f"N04_DIAGNOSTIC {fit_id} already started without recovery; no duplicate fit")
    resource = _resource_check(1024 * 1024**2 + 4 * x_fit.nbytes + 2 * x_score.nbytes,
                               f"cutoff={cutoff}, component={component}")
    start_fit(LEDGER_PATH, card_id=CARD_ID, stage=STAGE, fit_id=fit_id,
              cutoff=cutoff, component=component)
    recovery_saved = False
    try:
        with threadpool_limits(limits=THREADS):
            model, model_state = fit_weighted_r22_head(x_fit, fit_target, weights, cutoff=cutoff)
            raw = predict_r22_head(model, x_score)
        mapped = n04._rank50_independent(raw, source["cycles"])
        prediction = np.clip(.75 * source["legacy"] + .25 * mapped, 0, 100)
        arrays = {
            "score_rows": source["rows"].astype(np.int64),
            "score_record_ids": source["ids"].astype(str),
            "prediction": prediction.astype(np.float64),
            "baseline_prediction": source["R22_ORIGINAL"].astype(np.float64),
            "raw_component": raw.astype(np.float64),
            "rank50_component": mapped.astype(np.float64),
            "legacy_champion": source["legacy"].astype(np.float64),
            "r06_original": source["R06_ORIGINAL"].astype(np.float64),
            "cycle_code": source["cycles"].astype(np.int64),
            "stream_missing": missing,
        }
        diagnostics = {
            "cutoff": int(cutoff), "component": component,
            "fit_rows": int(len(fit_rows)),
            "fit_row_indices_sha256": n04._array_hash(fit_rows, "<i8"),
            "fit_record_id_sha256": n04._array_hash(
                context["frame"].record_id.to_numpy(dtype=str)[fit_rows].astype("U")),
            "fit_target_sha256": n04._array_hash(fit_target, "<f8"),
            "fit_sample_weight_sha256": n04._array_hash(weights, "<f8"),
            "fit_sample_weight_mean": float(weights.mean()),
            "fit_sample_weight_min": float(weights.min()),
            "fit_sample_weight_max": float(weights.max()),
            "fit_design_sha256": design.audit_state()["fit_design_sha256"],
            "score_design_sha256": n04._array_hash(x_score, "<f8"),
            "transform_state": design.audit_state(), "model_state": model_state,
            "domain_oof_weight_sha256": oof[cutoff]["weight_diagnostics"]["weight_sha256"],
            "reused_domain_fit_ids": [n04._fit_id("domain", cutoff, parity) for parity in PARITIES],
            "score_target_passed_to_fit": False, "test_data_used": False,
            "live_resource_check": resource,
        }
        payload = {**arrays, "diagnostics": diagnostics}
        publish_bundle(recovery_path, payload, identity, task="regression")
        recovery_saved = True
        recovery = read_bundle(recovery_path, identity, task="regression")
        _independent_bundle(recovery, context, frozen, cutoff, component, weights,
                            design.audit_state()["fit_design_sha256"])
        publish_bundle(path, payload, identity, task="regression")
        saved = read_bundle(path, identity, task="regression")
        _independent_bundle(saved, context, frozen, cutoff, component, weights,
                            design.audit_state()["fit_design_sha256"])
        finish_fit(LEDGER_PATH, fit_id=fit_id, status="FIT_SAVED",
                   receipt_sha256=saved.receipt["receipt_sha256"])
        return saved
    except Exception as exc:
        events = read_events(LEDGER_PATH)
        terminal = [event for event in events if event.get("fit_id") == fit_id
                    and event.get("event") in {"FIT_SAVED", "FIT_FAILED"}]
        if not terminal and not recovery_saved:
            finish_fit(LEDGER_PATH, fit_id=fit_id, status="FIT_FAILED",
                       error=f"{type(exc).__name__}: {exc}")
        raise


def _task_outputs(prepared: Mapping[str, Any], frozen: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    context, oof = prepared["context"], prepared["oof"]
    outputs: dict[int, dict[str, Any]] = {}
    for cutoff in ORIGINS:
        fit_rows, fit_features, fit_target = n04._fit_prefix(context, cutoff)
        if not np.array_equal(oof[cutoff]["rows"], fit_rows):
            raise RuntimeError("N04_DIAGNOSTIC verified OOF weights are not aligned to fit prefix")
        source = context["sources"][cutoff]
        score_features = context["frame"].iloc[source["rows"]].loc[:, list(n04.HYBRID_FEATURES)]
        design = R22WeightedDesign(cutoff)
        x_fit = design.fit_transform(fit_features)
        x_score = design.transform(score_features)
        weighted = _fit_one(prepared, frozen, cutoff, "weighted", design, x_fit, x_score,
                            oof[cutoff]["weights"])
        shuffled_weights = shuffle_weights_within_cycle(
            oof[cutoff]["weights"], oof[cutoff]["cycles"], seed=42 + cutoff)
        shuffled = _fit_one(prepared, frozen, cutoff, "within_cycle_shuffle", design,
                            x_fit, x_score, shuffled_weights)
        design_hash = design.audit_state()["fit_design_sha256"]
        if any(bundle.receipt["metadata"]["diagnostics"]["fit_design_sha256"] != design_hash
               for bundle in (weighted, shuffled)):
            raise RuntimeError(f"N04_DIAGNOSTIC cutoff {cutoff} weighted/shuffle designs differ")
        # N04's shared view helper expects the control under the key "shuffle".
        outputs[cutoff] = {"weighted": weighted, "shuffle": shuffled}
    return outputs


def _per_cycle(context: Mapping[str, Any], outputs: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for cutoff in ORIGINS:
        source = context["sources"][cutoff]
        weighted = outputs[cutoff]["weighted"].arrays["prediction"]
        shuffled = outputs[cutoff]["shuffle"].arrays["prediction"]
        y = context["target"][source["rows"]]
        for cycle in np.unique(source["cycles"]):
            mask = source["cycles"] == cycle
            mae = lambda pred: float(np.mean(np.abs(y[mask] - pred[mask])))
            result[str(int(cycle))] = {
                "origin": int(cutoff), "n": int(mask.sum()),
                "weighted_mae": mae(weighted), "within_cycle_shuffle_mae": mae(shuffled),
                "R22_ORIGINAL_mae": mae(source["R22_ORIGINAL"]),
                "protected_legacy_mae": mae(source["legacy"]),
                "R06_ORIGINAL_secondary_mae": mae(source["R06_ORIGINAL"]),
            }
    return result


def _collect_metrics(prepared: Mapping[str, Any], outputs: Mapping[int, Mapping[str, Any]],
                     frozen: Mapping[str, Any]) -> dict[str, Any]:
    context = prepared["context"]

    def add_subgroup_control(view: dict[str, Any], ranges: Mapping[int, tuple[int, int]]) -> None:
        shuffled = n04._view_metrics(
            context, [outputs[cutoff]["shuffle"] for cutoff in ranges], ranges,
            f"{view['name']}_shuffle_subgroups",
        )
        for group_name, group in view["groups"].items():
            control = shuffled["groups"][group_name]
            group["within_cycle_shuffle_mae"] = control.get("candidate_mae")
            if group.get("n", 0):
                group["gain_vs_shuffle"] = (
                    group["within_cycle_shuffle_mae"] - group["candidate_mae"]
                )

    windows = {}
    for index, (cutoff, first_cycle, last_cycle) in enumerate(
        ((55, 56, 59), (55, 60, 63), (55, 64, 67),
         (67, 68, 71), (67, 72, 75), (67, 76, 79)), start=1
    ):
        name = f"W{index}_{cutoff}_to_{first_cycle}_{last_cycle}"
        ranges = {cutoff: (first_cycle, last_cycle)}
        windows[name] = n04._view_with_control(context, outputs, ranges, name)
        add_subgroup_control(windows[name], ranges)
    primary_ranges = {55: (56, 59), 67: (68, 71)}
    primary = n04._view_with_control(
        context, outputs, primary_ranges, "primary_W1_W4")
    add_subgroup_control(primary, primary_ranges)
    h12_ranges = {55: (56, 67), 67: (68, 79)}
    h12 = n04._view_with_control(
        context, outputs, h12_ranges, "H12_W1_W6")
    add_subgroup_control(h12, h12_ranges)
    if primary["n"] != 9_064 or h12["n"] != 38_588:
        raise RuntimeError("N04_DIAGNOSTIC W1/W4 or H12 denominator differs from frozen protocol")

    per_cutoff = {}
    for cutoff in ORIGINS:
        primary_name = next(name for name, definition in zip(
            windows, ((55, 56, 59), (55, 60, 63), (55, 64, 67),
                      (67, 68, 71), (67, 72, 75), (67, 76, 79))) if definition[0] == cutoff
                            and definition[1] == cutoff + 1)
        diag = prepared["oof"][cutoff]["weight_diagnostics"]
        cutoff_h12 = n04._view_with_control(
            context, outputs,
            {cutoff: (cutoff + 1, cutoff + 12 if cutoff == 55 else 79)},
            f"H12_origin_{cutoff}",
        )
        cutoff_h12_ranges = {
            cutoff: (cutoff + 1, cutoff + 12 if cutoff == 55 else 79)
        }
        add_subgroup_control(cutoff_h12, cutoff_h12_ranges)
        per_cutoff[str(cutoff)] = {
            "domain_gate_context": prepared["audit"]["pre_task_gates"]["by_cutoff"][str(cutoff)],
            "fit_rows": int(np.sum(context["frame"].cycle_code.to_numpy(dtype=np.int64) <= cutoff)),
            "primary_window": primary_name,
            "primary_gain_vs_R22": windows[primary_name]["gain_vs_R22"],
            "primary_gain_vs_shuffle": windows[primary_name]["gain_vs_shuffle"],
            "H12_n": cutoff_h12["n"],
            "H12_gain_vs_R22": cutoff_h12["gain_vs_R22"],
            "H12_gain_vs_shuffle": cutoff_h12["gain_vs_shuffle"],
            "H12_view": cutoff_h12,
            "weight_diagnostics": dict(diag),
        }
    return {
        "status": "DIAGNOSTIC_ONLY", "card_id": CARD_ID,
        "question": prepared["identity"]["question"],
        "amendment_not_outcome_blind": True,
        "amendment_after_auc_before_task_mae": True,
        "original_N04_status_preserved": "SKIPPED_SCIENTIFIC_PREFLIGHT",
        "freeze_signature": frozen["freeze_signature"],
        "original_N04_freeze_signature": prepared["identity"]["original_N04_freeze_signature"],
        "pre_task_gates": prepared["audit"]["pre_task_gates"],
        "domain_fit_audit": prepared["audit"]["domain_fit_audit"],
        "cutoffs": per_cutoff, "windows_W1_W6": windows,
        "primary_W1_W4_pooled": primary, "H12_W1_W6_pooled": h12,
        "per_cycle": _per_cycle(context, outputs),
        "selection_eligibility": {"eligible_for_continuation": False,
                                   "reason": "diagnostic-only amendment; no automatic promotion"},
        "safety": {"official_test_used": False, "test_labels_read": False,
                   "sample_submission_used": False, "kaggle_calls": 0,
                   "full_train_refit": False, "notebook_changed": False},
        "budget": ledger_summary(LEDGER_PATH),
    }


def _plot(result: Mapping[str, Any], path: Path) -> None:
    if path.exists():
        return
    names = list(result["windows_W1_W6"])
    labels = [name.split("_")[0] for name in names]
    weighted = [result["windows_W1_W6"][name]["gain_vs_R22"] for name in names]
    shuffled = [result["windows_W1_W6"][name]["gain_vs_shuffle"] for name in names]
    positions = np.arange(len(names), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    width = .36
    ax.bar(positions - width / 2, weighted, width, label="Weighted − R22 gain", color="#4c78a8")
    ax.bar(positions + width / 2, shuffled, width, label="Weighted − shuffled gain", color="#f58518")
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_xticks(positions, labels)
    ax.set_ylabel("MAE gain (positive = lower MAE)")
    ax.set_title("N04_DIAGNOSTIC — gains by temporal four-cycle window")
    ax.grid(axis="y", alpha=.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _independent_verify(context: Mapping[str, Any], frozen: Mapping[str, Any],
                        result: Mapping[str, Any]) -> dict[str, Any]:
    del context, frozen, result
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.optimization_v3_research_next_n04_diagnostic_verify"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    return json.loads(completed.stdout)


def run_pilot() -> dict[str, Any]:
    frozen, prepared = _verify_freeze()
    slot_audit = _amendment_slot_audit()
    budget = slot_audit["ledger"]
    new_reservation_fits = PILOT_FITS if slot_audit["new_reservation_needed"] else 0
    remaining_amendment_fits = PILOT_FITS - slot_audit["amendment_started_fit_count"]
    if (budget["started_and_consumed"] + remaining_amendment_fits > budget["authorized_cap"]
            or budget["active_reserved"] + new_reservation_fits > budget["authorized_cap"]
            or budget["category_reservations"]["pilot"] + new_reservation_fits > 26):
        raise RuntimeError("N04_DIAGNOSTIC cap60 budget insufficient; no task fit started")
    reserve_card(LEDGER_PATH, card_id=CARD_ID, stage=STAGE, fits=PILOT_FITS, category="pilot")
    try:
        outputs = _task_outputs(prepared, frozen)
        release_unused(LEDGER_PATH, card_id=CARD_ID, stage=STAGE)
        result = _collect_metrics(prepared, outputs, frozen)
        result["budget_slot_audit"] = _amendment_slot_audit()
        _write_once(REPORT_ROOT / "results.json", result)
        _plot(result, REPORT_ROOT / "window_mae_gains.png")
        verification = _independent_verify(prepared["context"], frozen, result)
        result_summary = {
            "status": result["status"], "amendment_not_outcome_blind": True,
            "primary_n": result["primary_W1_W4_pooled"]["n"],
            "primary_gain_vs_R22": result["primary_W1_W4_pooled"]["gain_vs_R22"],
            "primary_gain_vs_shuffle": result["primary_W1_W4_pooled"]["gain_vs_shuffle"],
            "H12_n": result["H12_W1_W6_pooled"]["n"],
            "H12_gain_vs_R22": result["H12_W1_W6_pooled"]["gain_vs_R22"],
            "H12_gain_vs_shuffle": result["H12_W1_W6_pooled"]["gain_vs_shuffle"],
            "window_gains": {name: {"gain_vs_R22": item["gain_vs_R22"],
                                     "gain_vs_shuffle": item["gain_vs_shuffle"], "n": item["n"]}
                             for name, item in result["windows_W1_W6"].items()},
            "verification": verification["status"],
            "selection_eligibility": result["selection_eligibility"],
            "budget_slot_audit": result["budget_slot_audit"],
            "budget": ledger_summary(LEDGER_PATH),
        }
        _write_once(REPORT_ROOT / "SUMMARY.json", result_summary)
        _write_text_once(REPORT_ROOT / "SUMMARY.md", _summary_text(result, verification))
        return result_summary
    except ResourceDeferred as exc:
        _write_once(REPORT_ROOT / "resource_deferred.json", {
            "status": "DEFERRED_RESOURCE", "message": str(exc),
            "budget": ledger_summary(LEDGER_PATH), "resume_without_refit_if_bundle_exists": True,
        })
        return {"status": "DEFERRED_RESOURCE", "message": str(exc),
                "budget": ledger_summary(LEDGER_PATH)}
    except Exception:
        events = read_events(LEDGER_PATH)
        starts = [event for event in events if event.get("event") == "FIT_STARTED"
                  and event.get("reservation_key") == f"{CARD_ID}:{STAGE}"]
        terminal_count = sum(any(done.get("fit_id") == event.get("fit_id")
                                 and done.get("event") in {"FIT_SAVED", "FIT_FAILED"}
                                 for done in events) for event in starts)
        pending = [event for event in starts if not any(done.get("fit_id") == event.get("fit_id")
                                                       and done.get("event") in {"FIT_SAVED", "FIT_FAILED"}
                                                       for done in events)]
        if not pending and terminal_count == PILOT_FITS:
            release_unused(LEDGER_PATH, card_id=CARD_ID, stage=STAGE)
        raise


def _write_text_once(path: Path, text: str) -> None:
    data = text.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"N04_DIAGNOSTIC refuses to overwrite report: {_relative(path)}")
        return
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _summary_text(result: Mapping[str, Any], verification: Mapping[str, Any]) -> str:
    lines = [
        "# N04_DIAGNOSTIC — Task fits setelah domain screening", "",
        "Status: `DIAGNOSTIC_ONLY` — tidak eligible untuk continuation.", "",
        "Amendment dibuat setelah domain AUC diketahui, tetapi sebelum task MAE ditampilkan. Ini bukan analisis outcome-blind. N04 asli tetap `SKIPPED_SCIENTIFIC_PREFLIGHT`; threshold, gate, recipe, dan ledger N04 asli tidak diubah.", "",
        "Empat domain fits asli dipakai ulang setelah verifikasi bundle, recovery, OOF probabilities, dan hash bobot. Tidak ada domain classifier yang dilatih ulang. Tepat empat task fits dijalankan: weighted dan within-cycle-shuffle pada cutoff 55 dan 67.", "",
        "## Hasil per window", "",
        "Gain positif berarti MAE lebih rendah daripada pembanding.", "",
        "| Window | n | Weighted MAE | R22 MAE | Shuffle MAE | Protected legacy MAE | R06 sekunder MAE | Gain vs R22 | Gain vs shuffle |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, view in result["windows_W1_W6"].items():
        metrics = view["metrics"]
        lines.append(
            f"| {name.split('_')[0]} | {view['n']:,} | {metrics['candidate']:.6f} "
            f"| {metrics['R22_ORIGINAL']:.6f} | {metrics['within_cycle_shuffle']:.6f} "
            f"| {metrics['protected_legacy']:.6f} | {metrics['R06_ORIGINAL_secondary']:.6f} "
            f"| {view['gain_vs_R22']:+.6f} | {view['gain_vs_shuffle']:+.6f} |"
        )
    lines.extend(["", "## Primary, H12, dan domain context", "",
                  "| View | n | Weighted MAE | R22 MAE | Shuffle MAE | Legacy MAE | R06 MAE | Gain vs R22 | Gain vs shuffle |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for label, view in (("Primary W1 + W4", result["primary_W1_W4_pooled"]),
                        ("H12 W1–W6", result["H12_W1_W6_pooled"])):
        metrics = view["metrics"]
        lines.append(
            f"| {label} | {view['n']:,} | {metrics['candidate']:.6f} "
            f"| {metrics['R22_ORIGINAL']:.6f} | {metrics['within_cycle_shuffle']:.6f} "
            f"| {metrics['protected_legacy']:.6f} | {metrics['R06_ORIGINAL_secondary']:.6f} "
            f"| {view['gain_vs_R22']:+.6f} | {view['gain_vs_shuffle']:+.6f} |"
        )
    lines.extend(["", "| Cutoff | Domain fold AUCs | ESS fraction | Source clip fraction | Window utama | Gain vs R22 | Gain vs shuffle |",
                  "|---:|---|---:|---:|---|---:|---:|"])
    for cutoff, row in result["cutoffs"].items():
        gate = row["domain_gate_context"]
        aucs = ", ".join(f"{value:.6f}" for value in gate["heldout_fold_aucs"])
        lines.append(f"| {cutoff} | {aucs} | {gate['weight_diagnostics']['effective_sample_size_fraction']:.3f} "
                     f"| {gate['weight_diagnostics']['source_fraction_at_or_beyond_clip_bounds']:.3f} "
                     f"| {row['primary_window'].split('_')[0]} | {row['primary_gain_vs_R22']:+.6f} "
                     f"| {row['primary_gain_vs_shuffle']:+.6f} |")
    lines.extend(["", "| Cutoff | Fit rows | Primary window | Primary gain vs R22 | Primary gain vs shuffle | H12 n | H12 gain vs R22 | H12 gain vs shuffle |",
                  "|---:|---:|---|---:|---:|---:|---:|---:|"])
    for cutoff, row in result["cutoffs"].items():
        lines.append(
            f"| {cutoff} | {row['fit_rows']:,} | {row['primary_window'].split('_')[0]} "
            f"| {row['primary_gain_vs_R22']:+.6f} | {row['primary_gain_vs_shuffle']:+.6f} "
            f"| {row['H12_n']:,} | {row['H12_gain_vs_R22']:+.6f} "
            f"| {row['H12_gain_vs_shuffle']:+.6f} |"
        )
    lines.extend(["", "Cutoff 55 berada tipis di bawah batas AUC pada satu fold (0,549576) dan lebih rendah pada fold lain (0,542069), sementara ESS serta clipping memenuhi batas. Cutoff 67 memenuhi AUC, ESS, dan clipping. Dengan dua cutoff saja, hasil ini tidak mengidentifikasi hubungan umum antara AUC domain dan manfaat task.", "",
                  "## Guard subgroup", "",
                  "Setiap window, primary, dan H12 menyimpan denominator, MAE kandidat, R22, shuffled-weight, protected legacy, dan R06 untuk `stream_missing`, target aktual `<=5`, dan target aktual `>=95`. Gain positif berarti kandidat mengurangi MAE terhadap kontrol. Tail target hanya dipakai sebagai mask evaluasi. `supported` mengikuti minimum n=100; hasil ini diagnostik, bukan penerapan gate baru.", "",
                  "| View | Subgroup | n | Gain vs R22 | Gain vs shuffle | Harm vs legacy | Candidate MAE | R06 MAE |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"])
    subgroup_views = {**result["windows_W1_W6"],
                      "Primary W1+W4": result["primary_W1_W4_pooled"],
                      "H12 W1-W6": result["H12_W1_W6_pooled"],
                      **{f"H12 origin {cutoff}": row["H12_view"]
                         for cutoff, row in result["cutoffs"].items()}}
    for view_label, view in subgroup_views.items():
        for group_name, group in view["groups"].items():
            if not group.get("n"):
                lines.append(f"| {view_label} | {group_name} | 0 | n/a | n/a | n/a | n/a | n/a |")
                continue
            lines.append(
                f"| {view_label} | {group_name} | {group['n']:,} | "
                f"{group['harm_vs_R22']:+.6f} | {group['gain_vs_shuffle']:+.6f} | "
                f"{group['harm_vs_legacy']:+.6f} | {group['candidate_mae']:.6f} | "
                f"{group['R06_ORIGINAL_secondary_mae']:.6f} |"
            )
    lines.extend(["", "Tail subgroup masks use observed evaluation targets for reporting only; the task fits never receive evaluation targets.", "",
                  "## Verifikasi dan batas interpretasi", "",
                  f"Independent verification: `{verification['status']}`; domain bundles: {verification['reused_domain_fit_count']}; task fits: {verification['verified_task_fit_count']}; official Train hash, component weights, score IDs, rank50 average ties, .75/.25 policy, ledgers, and report metrics checked.", "",
                  "Amendment ini menguji empat task fits yang sudah diotorisasi. Hasilnya tidak mengubah verdict N04 asli, tidak melonggarkan gate 0,55, tidak memilih threshold baru, dan tidak layak otomatis masuk continuation. Test, Kaggle, notebook, dan full-Train refit tidak digunakan.", ""])
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "freeze", "run-pilot"))
    args = parser.parse_args(argv)
    if args.command == "preflight":
        result = preflight()
    elif args.command == "freeze":
        result = freeze()
    else:
        result = run_pilot()
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
