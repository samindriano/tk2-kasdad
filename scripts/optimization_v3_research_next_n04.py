"""Prepare, freeze, fit and verify the Research Next V1 N04 pilot."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
from io import BytesIO
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from scripts import optimization_v3_research_next_n00 as n00
from src.models.classification_models import _text_values
from src.models.research_next_budget import (
    finish_fit, ledger_summary, read_events, release_unused, reserve_card, start_fit,
)
from src.models.research_next_contract import canonical_sha256, file_sha256, publish_bundle, read_bundle
from src.models.research_next_n04 import (
    CARD_ID, DOMAIN_AUC_RANGE, DOMAIN_CATEGORICAL_FEATURES, DOMAIN_LR_PARAMETERS,
    DOMAIN_STRUCTURED_FEATURES, MAXIMUM_SOURCE_CLIP_FRACTION, MINIMUM_ESS_FRACTION,
    DomainFeatureBuilder, R22WeightedDesign, XGB_PARAMETERS, compose_r22_candidate,
    density_ratio_weights, domain_labels, fit_weighted_r22_head, predict_r22_head,
    shuffle_weights_within_cycle,
)
from src.models.regression_research_text import HYBRID_FEATURES, STRUCTURED_FEATURES
from src.utils.paths import PROJECT_ROOT, task_files

ROOT = PROJECT_ROOT
BATCH = "optimization_v3_research_next_v1"
QUEUE_PATH = ROOT / "experiments/optimization_v3/research_next_v1/queue.json"
READINESS_PATH = ROOT / "reports/optimization_v3/research_next_v1/queue_readiness.json"
CARD_PATH = ROOT / "reports/optimization_v3/research_next_v1/cards/N04.md"
N00_FREEZE_PATH = ROOT / "experiments/optimization_v3/research_next_v1/frozen/N00/amendment_05/freeze.json"
N00_SUMMARY_PATH = ROOT / "reports/optimization_v3/research_next_v1/results/N00/amendment_05/summary.json"
OLD_LEDGER_PATH = ROOT / "experiments/runs/optimization_v3_batch1/batch_ledger.jsonl"
LEDGER_PATH = ROOT / "experiments/runs/optimization_v3_research_next_v1/budget.jsonl"
PREVIOUS_FREEZE_PATH = ROOT / "experiments/optimization_v3/research_next_v1/frozen/N04/amendment_01/freeze.json"
PREVIOUS_PREFLIGHT_PATH = ROOT / "experiments/optimization_v3/research_next_v1/frozen/N04/amendment_01/preflight.json"
PREVIOUS_BUNDLE_ROOT = ROOT / "experiments/runs/optimization_v3_research_next_v1/N04/amendment_01"
PREVIOUS_RECOVERY_ROOT = PREVIOUS_BUNDLE_ROOT / "recovery"
FREEZE_PATH = ROOT / "experiments/optimization_v3/research_next_v1/frozen/N04/amendment_02/freeze.json"
PREFLIGHT_PATH = ROOT / "experiments/optimization_v3/research_next_v1/frozen/N04/amendment_02/preflight.json"
REPORT_ROOT = ROOT / "reports/optimization_v3/research_next_v1/results/N04/amendment_02"
BUNDLE_ROOT = ROOT / "experiments/runs/optimization_v3_research_next_v1/N04/amendment_02"
RECOVERY_ROOT = BUNDLE_ROOT / "recovery"
PILOT_ORIGINS = (55, 67)
PILOT_FITS = 8
THREADS = 2
JOB_CAP_BYTES = 6 * 1024**3
FREE_RESERVE_BYTES = 2 * 1024**3
MINIMUM_FREE_BYTES = 4 * 1024**3
DOMAIN_FOLDS = (0, 1)  # held-out even cycles, then held-out odd cycles
TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")


class ResourceDeferred(RuntimeError):
    """The live memory bound is no longer satisfied."""


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _is_ancestor(commit: str) -> bool:
    return subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"],
                          cwd=ROOT, check=False).returncode == 0


def _dependencies() -> dict[str, str]:
    names = ("numpy", "pandas", "scipy", "scikit-learn", "xgboost", "psutil", "matplotlib",
             "threadpoolctl")
    return {"python": platform.python_version(), "platform": platform.platform(),
            **{name: importlib.metadata.version(name) for name in names}}


def _source_paths() -> list[Path]:
    return [ROOT / rel for rel in (
        "scripts/optimization_v3_research_next_n00.py",
        "scripts/optimization_v3_research_next_n04.py",
        "src/models/research_next_contract.py",
        "src/models/research_next_budget.py",
        "src/models/research_next_n04.py",
        "src/models/round1_models.py",
        "src/models/classification_models.py",
        "src/models/regression_research_text.py",
        "src/models/optimization_v3_batch1_r22.py",
        "src/utils/paths.py",
        "tests/test_research_next_n04.py",
    )]


def _load_train() -> tuple[Path, pd.DataFrame, np.ndarray]:
    path = Path(task_files("regression")["train"])
    columns = ["record_id", "cycle_code", "signal_stream", "impact_load", "entry_friction",
               "scope_pressure", "text_load", "scope_trace", "org_spread", "module_spread",
               "telemetry_flag", "org_code", "module_code", "pattern_code", "threat_rank"]
    frame = pd.read_csv(path, usecols=columns)
    cycles = frame.cycle_code.to_numpy(dtype=np.int64)
    if (len(frame) != 50_608 or not frame.record_id.is_unique
            or not np.array_equal(np.unique(cycles), np.arange(47, 80))):
        raise RuntimeError("N04 official Regression Train schema, IDs or cycle range differs")
    target = frame.threat_rank.to_numpy(dtype=np.float64)
    if not np.isfinite(target).all() or np.any((target < 0) | (target > 100)):
        raise RuntimeError("N04 Train target must be finite in [0,100]")
    return path, frame, target


def _input_paths(train_path: Path) -> list[Path]:
    paths = [QUEUE_PATH, READINESS_PATH, CARD_PATH, N00_FREEZE_PATH, N00_SUMMARY_PATH,
             OLD_LEDGER_PATH, train_path, PREVIOUS_FREEZE_PATH, PREVIOUS_PREFLIGHT_PATH]
    previous_bundle = PREVIOUS_BUNDLE_ROOT / "fit55" / "domain_even"
    paths.extend((previous_bundle / "arrays.npz", previous_bundle / "receipt.json",
                  PREVIOUS_RECOVERY_ROOT / "N04_PILOT_55_domain_even.npz",
                  PREVIOUS_RECOVERY_ROOT / "N04_PILOT_55_domain_even.json"))
    for cutoff in PILOT_ORIGINS:
        for card in ("R22", "R06"):
            artifact = n00.cache_path(card, cutoff)
            paths.extend((artifact, artifact.with_suffix(".json")))
    return paths


def _load_verified_sources(frame: pd.DataFrame) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    sources: dict[int, dict[str, Any]] = {}
    audit: dict[str, Any] = {}
    all_rows: list[np.ndarray] = []
    for cutoff in PILOT_ORIGINS:
        r22_bundle = n00._load_artifact("R22", cutoff, frame)
        r06_bundle = n00._load_artifact("R06", cutoff, frame)
        if (not np.array_equal(r22_bundle["rows"], r06_bundle["rows"])
                or not np.array_equal(r22_bundle["arrays"]["score_record_ids"],
                                      r06_bundle["arrays"]["score_record_ids"])):
            raise RuntimeError(f"N04 R22/R06 cached rows or IDs differ at cutoff {cutoff}")
        r22 = n00.reconstruct_regression(r22_bundle)
        r06 = n00.reconstruct_regression(r06_bundle)
        if not np.array_equal(r22["prediction"], r22["control"]):
            raise RuntimeError(f"N04 R22_ORIGINAL is not the frozen unweighted control at {cutoff}")
        rows = r22["rows"].astype(np.int64)
        ids = r22_bundle["arrays"]["score_record_ids"].astype(str)
        cycles = r22["cycles"].astype(np.int64)
        expected = np.flatnonzero((frame.cycle_code.to_numpy(dtype=np.int64) > cutoff)
                                  & (frame.cycle_code.to_numpy(dtype=np.int64) <= cutoff + 12))
        if not np.array_equal(rows, expected) or not np.array_equal(ids, frame.record_id.to_numpy(dtype=str)[rows]):
            raise RuntimeError(f"N04 cutoff {cutoff} score rows are not exact complete cycles")
        if not np.array_equal(cycles, frame.cycle_code.to_numpy(dtype=np.int64)[rows]):
            raise RuntimeError(f"N04 cutoff {cutoff} cycle codes differ from official Train")
        sources[cutoff] = {
            "rows": rows, "ids": ids, "cycles": cycles,
            "R22_ORIGINAL": r22["prediction"].astype(np.float64),
            "legacy": r22["legacy"].astype(np.float64),
            "R06_ORIGINAL": r06["prediction"].astype(np.float64),
            "raw_R22": r22["raw"].astype(np.float64),
            "rank50_R22": r22["rank50"].astype(np.float64),
        }
        all_rows.append(rows)
        audit[str(cutoff)] = {
            "fit_rows": int(r22_bundle["fit_rows"]), "score_rows": int(len(rows)),
            "score_cycle_min": int(cycles.min()), "score_cycle_max": int(cycles.max()),
            "R22_npz_sha256": r22_bundle["npz_sha256"],
            "R22_sidecar_sha256": r22_bundle["sidecar_sha256"],
            "R06_npz_sha256": r06_bundle["npz_sha256"],
            "R06_sidecar_sha256": r06_bundle["sidecar_sha256"],
            "exact_R22_ORIGINAL_reconstruction": True,
            "exact_R06_ORIGINAL_reconstruction": True,
        }
    combined = np.concatenate(all_rows)
    if len(combined) != len(np.unique(combined)):
        raise RuntimeError("N04 STANDARD cache rows overlap across pilot origins")
    return sources, audit


def _text_nnz_upper_bound(streams: pd.Series) -> int:
    docs, _ = _text_values(streams)
    return int(sum(2 * len(TOKEN_RE.findall(doc)) for doc in docs))


def _resource_estimate(frame: pd.DataFrame, cutoff: int, available_bytes: int | None = None) -> dict[str, Any]:
    cycles = frame.cycle_code.to_numpy(dtype=np.int64)
    fit_rows = np.flatnonzero(cycles <= cutoff)
    domain = domain_labels(cycles[fit_rows], cutoff)
    fold_peak_bytes = 0
    fold_details = {}
    for parity in DOMAIN_FOLDS:
        held = cycles[fit_rows] % 2 == parity
        train_n, score_n = int((~held).sum()), int(held.sum())
        train_rows, score_rows = fit_rows[~held], fit_rows[held]
        train_nnz = _text_nnz_upper_bound(frame.iloc[train_rows].signal_stream)
        score_nnz = _text_nnz_upper_bound(frame.iloc[score_rows].signal_stream)
        domain_peak = (768 * 1024**2 + 5 * (train_nnz + train_n * 64) * 16
                       + 3 * (score_nnz + score_n * 64) * 16)
        fold_peak_bytes = max(fold_peak_bytes, int(domain_peak))
        fold_details[str(parity)] = {
            "fit_rows": train_n, "heldout_rows": score_n,
            "text_nnz_upper_fit": train_nnz, "text_nnz_upper_heldout": score_nnz,
            "estimated_peak_bytes": int(domain_peak),
        }
    score_n = int(np.sum((cycles > cutoff) & (cycles <= cutoff + 12)))
    width_upper = len(STRUCTURED_FEATURES) + 4 * 3 + 3 * 254 + 2 * 3 + 32 + 1
    task_nnz = _text_nnz_upper_bound(frame.iloc[fit_rows].signal_stream)
    task_peak = (1024 * 1024**2 + 4 * len(fit_rows) * width_upper * 8
                 + 2 * score_n * width_upper * 8 + 5 * task_nnz * 16)
    peak = max(fold_peak_bytes, int(task_peak))
    available = int(psutil.virtual_memory().available if available_bytes is None else available_bytes)
    allowed = min(JOB_CAP_BYTES, available - FREE_RESERVE_BYTES)
    return {
        "status": "PASS" if available >= MINIMUM_FREE_BYTES and allowed > 0 and peak <= allowed
        else "BLOCKED_RESOURCE",
        "cutoff": int(cutoff), "fit_rows": int(len(fit_rows)), "score_rows": score_n,
        "domain_folds": fold_details, "task_structured_width_upper": int(width_upper),
        "task_text_nnz_upper": task_nnz, "task_estimated_peak_bytes": int(task_peak),
        "estimated_peak_bytes": int(peak), "estimated_peak_is_observed": False,
        "formula": "max(domain 768MiB + 5*fit_sparse_upper + 3*score_sparse_upper; "
                   "task 1GiB + 4*fit_dense_upper + 2*score_dense_upper + 5*TFIDF_sparse_upper)",
        "available_physical_bytes": available, "minimum_free_bytes": MINIMUM_FREE_BYTES,
        "free_reserve_bytes": FREE_RESERVE_BYTES, "job_cap_bytes": JOB_CAP_BYTES,
        "allowed_bytes": max(0, allowed), "threads": THREADS, "concurrent_jobs": 1,
        "domain_fit_rows": int(len(fit_rows)), "recent_rows": int(domain.sum()),
    }


def _build_identity(frame: pd.DataFrame, train_path: Path, cache_audit: Mapping[str, Any],
                    resources: Mapping[str, Any], inherited_fit_audit: Mapping[str, Any]) -> dict[str, Any]:
    queue = json.loads(QUEUE_PATH.read_text(encoding="utf-8"))
    card = next((item for item in queue["cards"] if item.get("id") == CARD_ID), None)
    if (queue.get("batch_id") != BATCH or not card or card.get("profile") != "STANDARD"
            or card.get("pilot_fits") != PILOT_FITS or card.get("continuation_fits") != 16):
        raise RuntimeError("N04 canonical queue profile or fit allocation changed")
    if file_sha256(train_path) != queue["raw_train_sha256"]["regression"]:
        raise RuntimeError("N04 official Regression Train hash differs from queue")
    n00_frozen = json.loads(N00_FREEZE_PATH.read_text(encoding="utf-8"))
    if n00_frozen.get("freeze_signature") != canonical_sha256(n00_frozen.get("identity")):
        raise RuntimeError("N04 N00 freeze signature is invalid")
    n00_commit = n00_frozen.get("identity", {}).get("source_commit")
    if not n00_commit or not _is_ancestor(n00_commit):
        raise RuntimeError("N04 N00 source commit is not an ancestor of HEAD")
    inherited = n00_frozen.get("execution_context", {}).get("inherited_old_budget_snapshot", {})
    old_hash = inherited.get("sha256")
    if not old_hash or file_sha256(OLD_LEDGER_PATH) != old_hash:
        raise RuntimeError("N04 historical ledger differs from inherited N00 snapshot")
    paths, source_paths = _input_paths(train_path), _source_paths()
    missing = [str(path) for path in paths + source_paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"N04 freeze inputs are absent: {missing}")
    input_hashes = {_relative(path): file_sha256(path) for path in paths}
    source_hashes = {_relative(path): file_sha256(path) for path in source_paths}
    fit_rows = {str(c): int(np.sum(frame.cycle_code.to_numpy(dtype=np.int64) <= c))
                for c in PILOT_ORIGINS}
    return {
        "batch_id": BATCH, "card_id": CARD_ID, "profile": "STANDARD",
        "amendment": "amendment_02",
        "amendment_scope": "correct independent verification of the cutoff-55 even-fold domain labels; frozen features, split, estimator, thresholds and task recipe are unchanged",
        "recipe": {
            "pseudo_domain": "last four distinct FIT cycles are recent=1; older fit rows source=0",
            "domain_features": "R22 structured allowlist excluding cycle_code, plus signal_stream; exclude record_id and task targets",
            "domain_split": "two cross-fit folds by fit-cycle parity; preprocess each fold using only its fit rows",
            "domain_model": {"class": "sklearn.linear_model.LogisticRegression", **DOMAIN_LR_PARAMETERS},
            "density_ratio": "balanced-prior odds p/(1-p); p clip [1e-6,1-1e-6]; source odds clip [.25,4]; recent raw weight 1; normalize mean 1",
            "pre_task_gates": {
                "each_heldout_fold_auc_inclusive": list(DOMAIN_AUC_RANGE),
                "effective_sample_size_fraction_min": MINIMUM_ESS_FRACTION,
                "source_clip_fraction_max": MAXIMUM_SOURCE_CLIP_FRACTION,
                "all_weights_finite_positive": True, "failure_skips_all_task_fits": True,
            },
            "task_model": {"class": "xgboost.XGBRegressor", "parameters": dict(XGB_PARAMETERS),
                           "features": list(HYBRID_FEATURES),
                           "text": "fit-only word TF-IDF and SVD32 using frozen R22 transform parameters",
                           "only_estimator_change": "sample_weight"},
            "matched_control": "same design/learner; shuffle fitted weights within exact fit cycle, seed 42+cutoff",
            "policy": ".75 exact legacy champion + .25 complete-cycle rank50(new raw), average ties, final clip [0,100]",
            "fit_population": "official Regression Train rows with cycle_code<=outer cutoff",
        },
        "pilot_origins": list(PILOT_ORIGINS), "pilot_fit_count": PILOT_FITS,
        "fit_rows_by_cutoff": fit_rows,
        "views": {"primary": [[55, 56, 59], [67, 68, 71]],
                  "h12": [[55, 56, 67], [67, 68, 79]],
                  "expected_primary_rows": 9064, "expected_h12_rows": 38588},
        "task_gates": {
            "primary_gain_vs_R22_min": .03, "h12_harm_vs_R22_max": .05,
            "individual_window_harm_vs_R22_max": .5, "minimum_group_n": 100,
            "missing_or_target_tail_harm_vs_R22_and_legacy_max": .25,
            "primary_gain_vs_shuffle_min": .01,
        },
        "lineage": dict(cache_audit), "inherited_fit_audit": dict(inherited_fit_audit),
        "queue_sha256": file_sha256(QUEUE_PATH),
        "official_train_sha256": file_sha256(train_path),
        "input_and_source_sha256": {**input_hashes, **source_hashes},
        "resource_bounds": {str(k): {name: value for name, value in item.items()
                                     if name not in {"available_physical_bytes", "allowed_bytes", "status"}}
                            for k, item in resources.items()},
        "dependencies": _dependencies(),
    }


def prepare_context(*, available_bytes: int | None = None,
                    ledger_context_override: Mapping[str, Any] | None = None) -> dict[str, Any]:
    path, frame, target = _load_train()
    sources, cache_audit = _load_verified_sources(frame)
    inherited_bundle, inherited_fit_audit = _audit_previous_fit(frame)
    resources = {str(c): _resource_estimate(frame, c, available_bytes) for c in PILOT_ORIGINS}
    identity = _build_identity(frame, path, cache_audit, resources, inherited_fit_audit)
    live_ledger = {"new_ledger_sha256": file_sha256(LEDGER_PATH),
                   "ledger": ledger_summary(LEDGER_PATH)}
    if ledger_context_override is not None:
        live_ledger = dict(ledger_context_override)
    execution = {
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": _git_head(),
        "available_bytes": int(psutil.virtual_memory().available if available_bytes is None else available_bytes),
        **live_ledger, "old_ledger_sha256": file_sha256(OLD_LEDGER_PATH),
        "resources": resources, "test_files_opened": False, "test_labels_read": False, "kaggle_calls": 0,
        "inherited_fit_audit": inherited_fit_audit,
    }
    return {"identity": identity, "freeze_signature": canonical_sha256(identity),
            "execution_context": execution, "path": path, "frame": frame, "target": target,
            "sources": sources, "cache_audit": cache_audit, "resources": resources,
            "inherited_domain_bundle": inherited_bundle,
            "inherited_fit_audit": inherited_fit_audit}


def preflight() -> dict[str, Any]:
    first = prepare_context()
    second = prepare_context(
        available_bytes=max(MINIMUM_FREE_BYTES, int(first["execution_context"]["available_bytes"]) - 1234567),
        ledger_context_override={"new_ledger_sha256": "0" * 64, "ledger": {"simulated": True}},
    )
    if first["freeze_signature"] != second["freeze_signature"]:
        raise RuntimeError("N04 stable identity changed with dynamic RAM/ledger context")
    resources_pass = all(item["status"] == "PASS" for item in first["resources"].values())
    result = {
        "status": "READY_TO_FREEZE" if resources_pass else "DEFERRED_RESOURCE",
        "freeze_signature": first["freeze_signature"],
        "fit_rows_by_cutoff": first["identity"]["fit_rows_by_cutoff"],
        "cache_lineage": first["cache_audit"], "resources": first["resources"],
        "inherited_fit_audit": first["inherited_fit_audit"],
        "stable_identity_repeat_check": "PASS",
        "dynamic_context_sensitivity": {
            "ram_changed": first["execution_context"]["available_bytes"] != second["execution_context"]["available_bytes"],
            "ledger_changed": first["execution_context"]["new_ledger_sha256"] != second["execution_context"]["new_ledger_sha256"],
            "identity_unchanged": True,
        },
        "official_fits_started": 0, "test_files_opened": False, "kaggle_calls": 0,
    }
    _write_once(PREFLIGHT_PATH, result)
    return result


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        if canonical_sha256(current) != canonical_sha256(value):
            raise RuntimeError(f"Refusing to overwrite immutable N04 artifact: {path}")
        return
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def freeze() -> dict[str, Any]:
    context = prepare_context()
    if any(item["status"] != "PASS" for item in context["resources"].values()):
        raise RuntimeError("N04 frozen pre-fit resource estimate does not pass")
    if not PREFLIGHT_PATH.is_file():
        raise RuntimeError("N04 persisted preflight report is required before freeze")
    preflight_report = json.loads(PREFLIGHT_PATH.read_text(encoding="utf-8"))
    if preflight_report.get("freeze_signature") != context["freeze_signature"]:
        raise RuntimeError("N04 preflight report does not match the current stable identity")
    execution = dict(context["execution_context"])
    execution["preflight_sha256"] = file_sha256(PREFLIGHT_PATH)
    execution["inherited_fit_audit"] = context["inherited_fit_audit"]
    payload = {"schema_version": 1, "identity": context["identity"],
               "freeze_signature": context["freeze_signature"],
               "execution_context": execution}
    _write_once(FREEZE_PATH, payload)
    return {"status": "FROZEN", "freeze_signature": context["freeze_signature"],
            "path": _relative(FREEZE_PATH), "source_commit": execution["source_commit"]}


def _verify_freeze() -> tuple[dict[str, Any], dict[str, Any]]:
    if not FREEZE_PATH.is_file():
        raise RuntimeError("N04 freeze is required before official fits")
    frozen = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if frozen.get("freeze_signature") != canonical_sha256(frozen.get("identity")):
        raise RuntimeError("N04 freeze signature is invalid")
    current = prepare_context()
    if current["identity"] != frozen.get("identity"):
        raise RuntimeError("N04 frozen source, recipe or input identity changed")
    execution = frozen.get("execution_context", {})
    source_commit = execution.get("source_commit")
    if not source_commit or not _is_ancestor(source_commit):
        raise RuntimeError("N04 source commit for freeze is not an ancestor of HEAD")
    if (not PREFLIGHT_PATH.is_file()
            or execution.get("preflight_sha256") != file_sha256(PREFLIGHT_PATH)):
        raise RuntimeError("N04 persisted preflight report hash differs from freeze")
    if any(item["status"] != "PASS" for item in current["resources"].values()):
        raise ResourceDeferred("N04 live resource check no longer passes")
    return frozen, current


def _fit_id(component: str, cutoff: int, parity: int | None = None) -> str:
    suffix = f":{'odd' if parity else 'even'}" if parity is not None else ""
    return f"N04:PILOT:{cutoff}:{component}{suffix}"


def _bundle_identity(signature: str, cutoff: int, component: str,
                     parity: int | None = None) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "batch_id": BATCH, "card_id": CARD_ID, "freeze_signature": signature,
        "stage": "PILOT", "cutoff": int(cutoff), "component": component,
    }
    if parity is not None:
        identity["heldout_cycle_parity"] = "odd" if parity else "even"
    return identity


def _bundle_path(cutoff: int, component: str, parity: int | None = None) -> Path:
    name = component if parity is None else f"{component}_{'odd' if parity else 'even'}"
    return BUNDLE_ROOT / f"fit{cutoff}" / name


def _resource_check(estimate: int, *, label: str) -> dict[str, Any]:
    available = int(psutil.virtual_memory().available)
    allowed = min(JOB_CAP_BYTES, available - FREE_RESERVE_BYTES)
    result = {
        "status": "PASS" if available >= MINIMUM_FREE_BYTES and allowed > 0 and estimate <= allowed
        else "BLOCKED_RESOURCE",
        "label": label, "available_physical_bytes": available,
        "estimated_peak_bytes": int(estimate), "allowed_bytes": max(0, allowed),
        "minimum_free_bytes": MINIMUM_FREE_BYTES, "free_reserve_bytes": FREE_RESERVE_BYTES,
        "job_cap_bytes": JOB_CAP_BYTES, "threads": THREADS, "concurrent_jobs": 1,
    }
    if result["status"] != "PASS":
        raise ResourceDeferred(f"N04 live resource gate failed: {result}")
    return result


def _array_hash(values: Any, dtype: str | None = None) -> str:
    array = np.asarray(values, dtype=dtype) if dtype else np.asarray(values)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _independent_auc(labels: Any, probabilities: Any) -> float:
    y = np.asarray(labels, dtype=np.int8)
    p = np.asarray(probabilities, dtype=np.float64)
    if (y.ndim != 1 or p.shape != y.shape or not len(y) or not np.isfinite(p).all()
            or set(np.unique(y)) != {0, 1}):
        raise ValueError("N04 independent AUC inputs are invalid")
    ranks = rankdata(p, method="average")
    positive_n = int(y.sum())
    negative_n = len(y) - positive_n
    return float((ranks[y == 1].sum() - positive_n * (positive_n + 1) / 2)
                 / (positive_n * negative_n))


def _independent_domain(bundle: Any, frame: pd.DataFrame, cutoff: int, parity: int) -> dict[str, Any]:
    arrays = bundle.arrays
    rows = arrays["score_rows"].astype(np.int64)
    cycles = frame.cycle_code.to_numpy(dtype=np.int64)[rows]
    labels = arrays["domain_label"].astype(np.int8)
    probabilities = arrays["prediction"].astype(np.float64)
    fit_rows = np.flatnonzero(frame.cycle_code.to_numpy(dtype=np.int64) <= cutoff)
    full_prefix_labels = domain_labels(frame.cycle_code.to_numpy(dtype=np.int64)[fit_rows], cutoff)
    positions = np.searchsorted(fit_rows, rows)
    if (np.any(positions >= len(fit_rows))
            or not np.array_equal(fit_rows[positions], rows)):
        raise RuntimeError("N04 heldout domain rows do not belong to the declared fit prefix")
    expected_labels = full_prefix_labels[positions]
    if (not np.array_equal(arrays["score_record_ids"].astype(str),
                           frame.record_id.to_numpy(dtype=str)[rows])
            or not np.array_equal(cycles, arrays["cycle_code"].astype(np.int64))
            or not np.array_equal(expected_labels, labels)
            or np.any(cycles % 2 != parity) or np.any((probabilities < 0) | (probabilities > 1))):
        raise RuntimeError(f"N04 domain bundle row/label audit failed at {cutoff}/{parity}")
    auc = _independent_auc(labels, probabilities)
    saved_auc = bundle.receipt["metadata"]["diagnostics"]["independent_heldout_auc"]
    if not np.isclose(auc, saved_auc, rtol=0, atol=1e-14):
        raise RuntimeError("N04 heldout AUC failed independent reconstruction")
    return {"rows": rows, "ids": arrays["score_record_ids"].astype(str),
            "labels": labels, "probability": probabilities, "cycles": cycles, "auc": auc}


def _audit_previous_fit(frame: pd.DataFrame) -> tuple[Any, dict[str, Any]]:
    old_freeze = json.loads(PREVIOUS_FREEZE_PATH.read_text(encoding="utf-8"))
    if old_freeze.get("freeze_signature") != canonical_sha256(old_freeze.get("identity")):
        raise RuntimeError("N04 amendment_01 freeze signature is invalid")
    old_execution = old_freeze.get("execution_context", {})
    if (not PREVIOUS_PREFLIGHT_PATH.is_file()
            or old_execution.get("preflight_sha256") != file_sha256(PREVIOUS_PREFLIGHT_PATH)):
        raise RuntimeError("N04 amendment_01 preflight report does not match its freeze")
    old_identity = _bundle_identity(old_freeze["freeze_signature"], 55, "domain", 0)
    old_bundle_path = PREVIOUS_BUNDLE_ROOT / "fit55" / "domain_even"
    bundle = read_bundle(old_bundle_path, old_identity, task="classification")
    checked = _independent_domain(bundle, frame, 55, 0)
    fit_id = _fit_id("domain", 55, 0)
    recovery_path = PREVIOUS_RECOVERY_ROOT / "N04_PILOT_55_domain_even.npz"
    recovery_receipt_path = PREVIOUS_RECOVERY_ROOT / "N04_PILOT_55_domain_even.json"
    if not recovery_path.is_file() or not recovery_receipt_path.is_file():
        raise RuntimeError("N04 amendment_01 domain recovery files are absent")
    recovery_receipt = json.loads(recovery_receipt_path.read_text(encoding="utf-8"))
    if (recovery_receipt.get("identity") != old_identity
            or recovery_receipt.get("npz_sha256") != file_sha256(recovery_path)):
        raise RuntimeError("N04 amendment_01 recovery payload identity/hash failed")
    with np.load(recovery_path, allow_pickle=False) as archive:
        recovery_arrays = {key: archive[key].copy() for key in archive.files}
    if set(recovery_arrays) != set(bundle.arrays) or any(
            not np.array_equal(recovery_arrays[key], bundle.arrays[key]) for key in bundle.arrays):
        raise RuntimeError("N04 amendment_01 recovery arrays differ from the published domain bundle")
    audit = {
        "status": "PASS", "fit_id": fit_id,
        "previous_freeze_signature": old_freeze["freeze_signature"],
        "previous_bundle_receipt_sha256": bundle.receipt["receipt_sha256"],
        "previous_bundle_npz_sha256": file_sha256(old_bundle_path / "arrays.npz"),
        "previous_bundle_receipt_file_sha256": file_sha256(old_bundle_path / "receipt.json"),
        "previous_recovery_npz_sha256": file_sha256(recovery_path),
        "previous_recovery_receipt_sha256": file_sha256(recovery_receipt_path),
        "row_ids_cycles_parity_and_full_prefix_labels": "PASS",
        "recovery_equals_published_arrays": True,
        "independent_heldout_auc": float(checked["auc"]),
        "recipe_changed": False,
        "estimator_refit": False,
    }
    return bundle, audit


def _recovery_paths(fit_id: str) -> tuple[Path, Path]:
    safe = fit_id.replace(":", "_")
    return RECOVERY_ROOT / f"{safe}.npz", RECOVERY_ROOT / f"{safe}.json"


def _save_recovery(fit_id: str, identity: Mapping[str, Any],
                   payload: Mapping[str, Any]) -> None:
    RECOVERY_ROOT.mkdir(parents=True, exist_ok=True)
    array_path, receipt_path = _recovery_paths(fit_id)
    if array_path.exists() or receipt_path.exists():
        raise RuntimeError(f"N04 recovery already exists for {fit_id}; inspect before overwrite")
    descriptor, temporary = tempfile.mkstemp(prefix=fit_id.replace(":", "_") + ".",
                                              suffix=".npz", dir=RECOVERY_ROOT)
    os.close(descriptor)
    try:
        with open(temporary, "wb") as handle:
            np.savez_compressed(handle, **payload["arrays"])
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, array_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    _write_once(receipt_path, {
        "identity": dict(identity), "metadata": dict(payload["metadata"]),
        "npz_sha256": file_sha256(array_path),
    })


def _load_recovery(fit_id: str, identity: Mapping[str, Any]) -> dict[str, Any] | None:
    array_path, receipt_path = _recovery_paths(fit_id)
    if not array_path.exists() and not receipt_path.exists():
        return None
    if not array_path.exists() or not receipt_path.exists():
        raise RuntimeError(f"N04 recovery incomplete for {fit_id}; preserve and inspect")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("identity") != dict(identity) or receipt.get("npz_sha256") != file_sha256(array_path):
        raise RuntimeError(f"N04 recovery identity/hash mismatch for {fit_id}")
    with np.load(array_path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    return {"arrays": arrays, "metadata": receipt["metadata"]}


def _payload_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {**payload["arrays"], "diagnostics": payload["metadata"]["diagnostics"]}


def _mark_saved(fit_id: str, receipt_sha256: str) -> None:
    events = read_events(LEDGER_PATH)
    starts = [event for event in events if event.get("event") == "FIT_STARTED"
              and event.get("fit_id") == fit_id]
    if len(starts) != 1:
        raise RuntimeError(f"N04 {fit_id} cannot be marked saved without exactly one start")
    terminals = [event for event in events if event.get("fit_id") == fit_id
                 and event.get("event") in {"FIT_SAVED", "FIT_FAILED"}]
    if terminals:
        if (len(terminals) != 1 or terminals[0].get("event") != "FIT_SAVED"
                or terminals[0].get("receipt_sha256") != receipt_sha256):
            raise RuntimeError(f"N04 {fit_id} ledger receipt conflicts with saved bundle")
        return
    finish_fit(LEDGER_PATH, fit_id=fit_id, status="FIT_SAVED", receipt_sha256=receipt_sha256)


def _publish_recovery(fit_id: str, identity: Mapping[str, Any], path: Path, task: str,
                      verifier: Any) -> Any | None:
    payload = _load_recovery(fit_id, identity)
    if payload is None:
        return None
    if not path.exists():
        publish_bundle(path, _payload_contract(payload), identity, task=task)
    saved = read_bundle(path, identity, task=task)
    verifier(saved)
    _mark_saved(fit_id, saved.receipt["receipt_sha256"])
    return saved


def _existing_bundle(fit_id: str, identity: Mapping[str, Any], path: Path,
                     task: str, verifier: Any) -> Any | None:
    if not path.exists():
        return None
    saved = read_bundle(path, identity, task=task)
    verifier(saved)
    _mark_saved(fit_id, saved.receipt["receipt_sha256"])
    return saved


def _domain_fold(context: Mapping[str, Any], frozen: Mapping[str, Any],
                 cutoff: int, parity: int) -> Any:
    fit_id = _fit_id("domain", cutoff, parity)
    identity = _bundle_identity(frozen["freeze_signature"], cutoff, "domain", parity)
    path = _bundle_path(cutoff, "domain", parity)
    frame = context["frame"]
    verifier = lambda bundle: _independent_domain(bundle, frame, cutoff, parity)
    existing = _existing_bundle(fit_id, identity, path, "classification", verifier)
    if existing is not None:
        return existing
    recovered = _publish_recovery(fit_id, identity, path, "classification", verifier)
    if recovered is not None:
        return recovered
    if cutoff == 55 and parity == 0:
        inherited = context.get("inherited_domain_bundle")
        audit = context.get("inherited_fit_audit", {})
        if inherited is None or audit.get("status") != "PASS":
            raise RuntimeError("N04 amendment_02 lacks its independently verified inherited fold")
        _mark_saved(fit_id, inherited.receipt["receipt_sha256"])
        return inherited
    events = read_events(LEDGER_PATH)
    if any(event.get("event") == "FIT_STARTED" and event.get("fit_id") == fit_id for event in events):
        raise RuntimeError(f"N04 {fit_id} started without recoverable output; no duplicate fit")

    cycles = frame.cycle_code.to_numpy(dtype=np.int64)
    fit_rows = np.flatnonzero(cycles <= cutoff)
    labels = domain_labels(cycles[fit_rows], cutoff)
    held = cycles[fit_rows] % 2 == parity
    train_rows, score_rows = fit_rows[~held], fit_rows[held]
    train_labels, score_labels = labels[~held], labels[held]
    if set(np.unique(train_labels)) != {0, 1} or set(np.unique(score_labels)) != {0, 1}:
        raise RuntimeError(f"N04 cutoff {cutoff} parity {parity} lacks both domain labels")
    columns = [*DOMAIN_STRUCTURED_FEATURES, "signal_stream"]
    transformer = DomainFeatureBuilder()
    x_train = transformer.fit_transform(frame.iloc[train_rows].loc[:, columns])
    x_score = transformer.transform(frame.iloc[score_rows].loc[:, columns])
    if x_train.shape[0] != len(train_rows) or x_score.shape[0] != len(score_rows):
        raise RuntimeError("N04 domain feature matrices are misaligned")
    train_bytes = x_train.data.nbytes + x_train.indices.nbytes + x_train.indptr.nbytes
    score_bytes = x_score.data.nbytes + x_score.indices.nbytes + x_score.indptr.nbytes
    resource = _resource_check(768 * 1024**2 + 5 * train_bytes + 3 * score_bytes,
                               label=f"domain cutoff {cutoff} parity {parity}")
    start_fit(LEDGER_PATH, card_id=CARD_ID, stage="PILOT", fit_id=fit_id,
              cutoff=cutoff, component=f"domain_lr_{'odd' if parity else 'even'}")
    recovery_saved = False
    try:
        model = LogisticRegression(**dict(DOMAIN_LR_PARAMETERS))
        with threadpool_limits(limits=THREADS):
            model.fit(x_train, train_labels)
        positive = int(np.flatnonzero(model.classes_ == 1)[0])
        probability = model.predict_proba(x_score)[:, positive].astype(np.float64)
        auc = _independent_auc(score_labels, probability)
        rows = score_rows.astype(np.int64)
        ids = frame.record_id.to_numpy(dtype=str)[rows]
        classifier_state = {
            "coef_sha256": _array_hash(model.coef_), "intercept_sha256": _array_hash(model.intercept_),
            "classes": model.classes_.astype(int).tolist(), "n_iter": model.n_iter_.astype(int).tolist(),
            "parameters": dict(DOMAIN_LR_PARAMETERS), "fit_rows": int(len(train_rows)),
            "fit_domain_label_sha256": _array_hash(train_labels, "<i8"),
            "fit_record_id_sha256": _array_hash(frame.record_id.to_numpy(dtype=str)[train_rows].astype("U")),
        }
        classifier_state["model_state_sha256"] = canonical_sha256(classifier_state)
        diagnostics = {
            "cutoff": int(cutoff), "heldout_cycle_parity": "odd" if parity else "even",
            "fit_rows": int(len(train_rows)), "heldout_rows": int(len(score_rows)),
            "fit_source_rows": int(np.sum(train_labels == 0)),
            "fit_recent_rows": int(np.sum(train_labels == 1)),
            "heldout_source_rows": int(np.sum(score_labels == 0)),
            "heldout_recent_rows": int(np.sum(score_labels == 1)),
            "fit_row_indices_sha256": _array_hash(train_rows, "<i8"),
            "heldout_row_indices_sha256": _array_hash(rows, "<i8"),
            "heldout_label_sha256": _array_hash(score_labels, "<i8"),
            "heldout_probability_sha256": _array_hash(probability, "<f8"),
            "independent_heldout_auc": float(auc), "classifier_state": classifier_state,
            "preprocessor_state": transformer.audit_state(), "actual_sparse_fit_bytes": int(train_bytes),
            "actual_sparse_score_bytes": int(score_bytes), "live_resource_check": resource,
            "cycle_code_used_as_feature": False, "record_id_used_as_feature": False,
            "task_target_used": False, "outer_score_labels_passed_to_fit": False,
            "test_data_used": False,
        }
        payload = {
            "arrays": {
                "score_rows": rows, "score_record_ids": ids,
                "prediction": probability, "baseline_prediction": score_labels.astype(np.float64),
                "domain_label": score_labels.astype(np.int8), "cycle_code": cycles[rows].astype(np.int64),
            },
            "metadata": {"diagnostics": diagnostics},
        }
        _save_recovery(fit_id, identity, payload)
        recovery_saved = True
        publish_bundle(path, _payload_contract(payload), identity, task="classification")
        saved = read_bundle(path, identity, task="classification")
        verifier(saved)
        _mark_saved(fit_id, saved.receipt["receipt_sha256"])
        return saved
    except Exception as exc:
        events = read_events(LEDGER_PATH)
        started = [event for event in events if event.get("event") == "FIT_STARTED"
                   and event.get("fit_id") == fit_id]
        terminal = [event for event in events if event.get("fit_id") == fit_id
                    and event.get("event") in {"FIT_SAVED", "FIT_FAILED"}]
        if started and not terminal and not recovery_saved:
            finish_fit(LEDGER_PATH, fit_id=fit_id, status="FIT_FAILED",
                       error=f"{type(exc).__name__}: {exc}")
        raise


def _domain_oof(context: Mapping[str, Any], cutoff: int, bundles: Sequence[Any]) -> dict[str, Any]:
    parts = []
    for bundle in bundles:
        parity_name = bundle.receipt["identity"]["heldout_cycle_parity"]
        parity = 1 if parity_name == "odd" else 0
        parts.append(_independent_domain(bundle, context["frame"], cutoff, parity))
    rows = np.concatenate([part["rows"] for part in parts])
    order = np.argsort(rows, kind="mergesort")
    rows = rows[order]
    expected = np.flatnonzero(context["frame"].cycle_code.to_numpy(dtype=np.int64) <= cutoff)
    if len(rows) != len(np.unique(rows)) or not np.array_equal(rows, expected):
        raise RuntimeError(f"N04 domain OOF must cover all fit rows once at cutoff {cutoff}")
    ids = np.concatenate([part["ids"] for part in parts])[order]
    labels = np.concatenate([part["labels"] for part in parts])[order]
    probability = np.concatenate([part["probability"] for part in parts])[order]
    cycles = context["frame"].cycle_code.to_numpy(dtype=np.int64)[rows]
    if (not np.array_equal(ids, context["frame"].record_id.to_numpy(dtype=str)[rows])
            or not np.array_equal(labels, domain_labels(cycles, cutoff))):
        raise RuntimeError("N04 combined domain OOF rows/labels differ from official Train")
    weights, weight_diagnostics = density_ratio_weights(probability, labels)
    return {"rows": rows, "ids": ids, "cycles": cycles, "labels": labels,
            "probability": probability, "weights": weights,
            "fold_aucs": [part["auc"] for part in parts],
            "overall_oof_auc": _independent_auc(labels, probability),
            "weight_diagnostics": weight_diagnostics}


def _pre_task_gates(oof_by_cutoff: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    by_cutoff: dict[str, Any] = {}
    passes = True
    for cutoff, oof in oof_by_cutoff.items():
        diag = oof["weight_diagnostics"]
        auc_pass = all(DOMAIN_AUC_RANGE[0] <= value <= DOMAIN_AUC_RANGE[1]
                       for value in oof["fold_aucs"])
        ess_pass = diag["effective_sample_size_fraction"] >= MINIMUM_ESS_FRACTION
        clip_pass = diag["source_fraction_at_or_beyond_clip_bounds"] <= MAXIMUM_SOURCE_CLIP_FRACTION
        weights_pass = diag["all_weights_finite_positive"] and np.isclose(
            diag["weight_mean"], 1.0, rtol=0, atol=1e-12)
        ok = bool(auc_pass and ess_pass and clip_pass and weights_pass)
        passes &= ok
        by_cutoff[str(cutoff)] = {
            "heldout_fold_aucs": [float(value) for value in oof["fold_aucs"]],
            "overall_oof_auc_descriptive": float(oof["overall_oof_auc"]),
            "auc_range": list(DOMAIN_AUC_RANGE), "auc_gate": bool(auc_pass),
            "weight_diagnostics": dict(diag), "ess_gate": bool(ess_pass),
            "source_clip_fraction_gate": bool(clip_pass),
            "positive_finite_mean_one_weights_gate": bool(weights_pass), "pass": ok,
        }
    return {"status": "PASS" if passes else "SKIPPED_SCIENTIFIC_PREFLIGHT",
            "all_cutoffs_pass": bool(passes), "by_cutoff": by_cutoff}


def _rank50_independent(raw: Any, cycles: Any) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float64)
    groups = np.asarray(cycles, dtype=np.int64)
    result = np.empty(values.shape, dtype=np.float64)
    for cycle in np.unique(groups):
        mask = groups == cycle
        order = np.argsort(values[mask], kind="mergesort")
        sorted_values = values[mask][order]
        ranks = np.empty(len(order), dtype=np.float64)
        i = 0
        while i < len(order):
            j = i + 1
            while j < len(order) and sorted_values[j] == sorted_values[i]:
                j += 1
            ranks[i:j] = ((i + 1) + j) / 2.0
            i = j
        unsorted = np.empty(len(order), dtype=np.float64)
        unsorted[order] = ranks
        result[mask] = np.clip(.5 * values[mask] + 50.0 * unsorted / len(order), 0, 100)
    return result


def _independent_task(bundle: Any, frame: pd.DataFrame, source: Mapping[str, Any],
                      cutoff: int, component: str) -> dict[str, Any]:
    arrays = bundle.arrays
    rows = arrays["score_rows"].astype(np.int64)
    cycles = frame.cycle_code.to_numpy(dtype=np.int64)[rows]
    if (not np.array_equal(rows, source["rows"])
            or not np.array_equal(arrays["score_record_ids"].astype(str),
                                  frame.record_id.to_numpy(dtype=str)[rows])
            or not np.array_equal(cycles, arrays["cycle_code"].astype(np.int64))
            or not np.array_equal(arrays["baseline_prediction"], source["R22_ORIGINAL"])):
        raise RuntimeError(f"N04 {component}/{cutoff} score or R22 baseline alignment failed")
    raw = arrays["raw_component"].astype(np.float64)
    mapped = _rank50_independent(raw, cycles)
    if not np.allclose(mapped, arrays["rank50_component"], rtol=0, atol=1e-12):
        raise RuntimeError("N04 rank50 output failed independent average-tie reconstruction")
    prediction = np.clip(.75 * arrays["legacy_champion"] + .25 * mapped, 0, 100)
    if not np.allclose(prediction, arrays["prediction"], rtol=0, atol=1e-12):
        raise RuntimeError("N04 candidate failed independent .75/.25 policy reconstruction")
    if (not np.array_equal(arrays["legacy_champion"], source["legacy"])
            or not np.array_equal(arrays["r06_original"], source["R06_ORIGINAL"])):
        raise RuntimeError("N04 frozen legacy/R06 comparator differs from verified cache")
    missing = _text_values(frame.iloc[rows]["signal_stream"])[1].astype(bool)
    if not np.array_equal(missing, arrays["stream_missing"].astype(bool)):
        raise RuntimeError("N04 score availability mask differs from official Train")
    return {
        "rows": rows, "prediction": prediction,
        "baseline": arrays["baseline_prediction"].astype(np.float64),
        "legacy": arrays["legacy_champion"].astype(np.float64),
        "r06": arrays["r06_original"].astype(np.float64),
        "cycles": cycles, "missing": missing,
    }


def _task_fold(context: Mapping[str, Any], frozen: Mapping[str, Any], cutoff: int,
               component: str, design: R22WeightedDesign, x_fit: np.ndarray,
               x_score: np.ndarray, fit_rows: np.ndarray, fit_target: np.ndarray,
               weights: np.ndarray) -> Any:
    fit_id = _fit_id(component, cutoff)
    identity = _bundle_identity(frozen["freeze_signature"], cutoff, component)
    path = _bundle_path(cutoff, component)
    source = context["sources"][cutoff]
    verifier = lambda bundle: _independent_task(bundle, context["frame"], source, cutoff, component)
    existing = _existing_bundle(fit_id, identity, path, "regression", verifier)
    if existing is not None:
        if existing.receipt["metadata"]["diagnostics"]["fit_design_sha256"] != design.audit_state()["fit_design_sha256"]:
            raise RuntimeError("N04 resumed output design differs from deterministic fit-only transform")
        return existing
    recovered = _publish_recovery(fit_id, identity, path, "regression", verifier)
    if recovered is not None:
        if recovered.receipt["metadata"]["diagnostics"]["fit_design_sha256"] != design.audit_state()["fit_design_sha256"]:
            raise RuntimeError("N04 recovered output design differs from deterministic fit-only transform")
        return recovered
    if any(event.get("event") == "FIT_STARTED" and event.get("fit_id") == fit_id
           for event in read_events(LEDGER_PATH)):
        raise RuntimeError(f"N04 {fit_id} started without recovery; no duplicate fit")
    resource = _resource_check(1024 * 1024**2 + 4 * x_fit.nbytes + 2 * x_score.nbytes,
                               label=f"task {component} cutoff {cutoff}")
    start_fit(LEDGER_PATH, card_id=CARD_ID, stage="PILOT", fit_id=fit_id,
              cutoff=cutoff, component=component)
    recovery_saved = False
    try:
        with threadpool_limits(limits=THREADS):
            model, model_state = fit_weighted_r22_head(x_fit, fit_target, weights, cutoff=cutoff)
            raw = predict_r22_head(model, x_score)
        mapped = _rank50_independent(raw, source["cycles"])
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
            "stream_missing": _text_values(
                context["frame"].iloc[source["rows"]]["signal_stream"]
            )[1].astype(bool),
        }
        diagnostics = {
            "cutoff": int(cutoff), "component": component, "fit_rows": int(len(fit_rows)),
            "fit_row_indices_sha256": _array_hash(fit_rows, "<i8"),
            "fit_record_id_sha256": _array_hash(
                context["frame"].record_id.to_numpy(dtype=str)[fit_rows].astype("U")),
            "fit_target_sha256": _array_hash(fit_target, "<f8"),
            "fit_sample_weight_sha256": _array_hash(weights, "<f8"),
            "fit_sample_weight_mean": float(weights.mean()),
            "fit_sample_weight_min": float(weights.min()),
            "fit_sample_weight_max": float(weights.max()),
            "fit_design_sha256": design.audit_state()["fit_design_sha256"],
            "score_design_sha256": _array_hash(x_score, "<f8"),
            "transform_state": design.audit_state(), "model_state": model_state,
            "verified_R22_cache_baseline": True, "score_target_passed_to_fit": False,
            "test_data_used": False, "live_resource_check": resource,
        }
        payload = {"arrays": arrays, "metadata": {"diagnostics": diagnostics}}
        _save_recovery(fit_id, identity, payload)
        recovery_saved = True
        publish_bundle(path, _payload_contract(payload), identity, task="regression")
        saved = read_bundle(path, identity, task="regression")
        verifier(saved)
        _mark_saved(fit_id, saved.receipt["receipt_sha256"])
        return saved
    except Exception as exc:
        events = read_events(LEDGER_PATH)
        started = [event for event in events if event.get("event") == "FIT_STARTED"
                   and event.get("fit_id") == fit_id]
        terminal = [event for event in events if event.get("fit_id") == fit_id
                    and event.get("event") in {"FIT_SAVED", "FIT_FAILED"}]
        if started and not terminal and not recovery_saved:
            finish_fit(LEDGER_PATH, fit_id=fit_id, status="FIT_FAILED",
                       error=f"{type(exc).__name__}: {exc}")
        raise


def _fit_prefix(context: Mapping[str, Any], cutoff: int) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    rows = np.flatnonzero(context["frame"].cycle_code.to_numpy(dtype=np.int64) <= cutoff)
    features = context["frame"].iloc[rows].loc[:, list(HYBRID_FEATURES)]
    return rows, features, context["target"][rows].astype(np.float64)


def _task_outputs(context: Mapping[str, Any], frozen: Mapping[str, Any],
                  oof: Mapping[int, Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    outputs = {}
    for cutoff in PILOT_ORIGINS:
        fit_rows, fit_features, fit_target = _fit_prefix(context, cutoff)
        if not np.array_equal(oof[cutoff]["rows"], fit_rows):
            raise RuntimeError("N04 OOF weights are misaligned with exact R22 fit prefix")
        score = context["sources"][cutoff]
        score_features = context["frame"].iloc[score["rows"]].loc[:, list(HYBRID_FEATURES)]
        design = R22WeightedDesign(cutoff)
        x_fit = design.fit_transform(fit_features)
        x_score = design.transform(score_features)
        weighted = _task_fold(context, frozen, cutoff, "weighted", design, x_fit, x_score,
                              fit_rows, fit_target, oof[cutoff]["weights"])
        shuffled_weights = shuffle_weights_within_cycle(
            oof[cutoff]["weights"], oof[cutoff]["cycles"], seed=42 + cutoff)
        shuffled = _task_fold(context, frozen, cutoff, "within_cycle_shuffle", design, x_fit,
                              x_score, fit_rows, fit_target, shuffled_weights)
        design_hash = design.audit_state()["fit_design_sha256"]
        if any(bundle.receipt["metadata"]["diagnostics"]["fit_design_sha256"] != design_hash
               for bundle in (weighted, shuffled)):
            raise RuntimeError("N04 task candidate and shuffle transforms are not identical")
        outputs[cutoff] = {"weighted": weighted, "shuffle": shuffled}
    return outputs


def _mae(target: Any, prediction: Any) -> float:
    y = np.asarray(target, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    if y.ndim != 1 or p.shape != y.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("N04 MAE inputs must be finite aligned vectors")
    return float(np.mean(np.abs(y - p)))


def _view_metrics(context: Mapping[str, Any], bundles: Sequence[Any],
                  ranges: Mapping[int, tuple[int, int]], name: str) -> dict[str, Any]:
    parts = []
    for bundle in bundles:
        cutoff = int(bundle.receipt["identity"]["cutoff"])
        if cutoff not in ranges:
            continue
        item = _independent_task(bundle, context["frame"], context["sources"][cutoff],
                                 cutoff, str(bundle.receipt["identity"]["component"]))
        low, high = ranges[cutoff]
        mask = (item["cycles"] >= low) & (item["cycles"] <= high)
        if mask.any():
            parts.append({key: value[mask] for key, value in item.items()
                          if isinstance(value, np.ndarray) and value.ndim == 1})
    if not parts:
        raise RuntimeError(f"N04 view {name} has no rows")
    rows = np.concatenate([part["rows"] for part in parts])
    if len(rows) != len(np.unique(rows)):
        raise RuntimeError(f"N04 view {name} duplicates rows")
    order = np.argsort(rows, kind="mergesort")
    rows = rows[order]
    pick = lambda key: np.concatenate([part[key] for part in parts])[order]
    y = context["target"][rows]
    candidate, r22, legacy, r06 = (pick(key) for key in ("prediction", "baseline", "legacy", "r06"))
    missing = pick("missing").astype(bool)
    result: dict[str, Any] = {
        "name": name, "n": int(len(rows)), "rows_sha256": _array_hash(rows, "<i8"),
        "metrics": {
            "candidate": _mae(y, candidate), "R22_ORIGINAL": _mae(y, r22),
            "protected_legacy": _mae(y, legacy), "R06_ORIGINAL_secondary": _mae(y, r06),
        },
        "gain_vs_R22": _mae(y, r22) - _mae(y, candidate),
    }
    group_masks = {
        "stream_missing": missing,
        "observed_target_le5_evaluation_only": y <= 5,
        "observed_target_ge95_evaluation_only": y >= 95,
    }
    groups = {}
    for group_name, mask in group_masks.items():
        count = int(mask.sum())
        if count == 0:
            groups[group_name] = {"n": 0, "supported": False}
            continue
        candidate_mae = _mae(y[mask], candidate[mask])
        r22_mae = _mae(y[mask], r22[mask])
        legacy_mae = _mae(y[mask], legacy[mask])
        groups[group_name] = {
            "n": count, "supported": count >= 100,
            "candidate_mae": candidate_mae, "R22_ORIGINAL_mae": r22_mae,
            "legacy_mae": legacy_mae,
            "R06_ORIGINAL_secondary_mae": _mae(y[mask], r06[mask]),
            "harm_vs_R22": candidate_mae - r22_mae,
            "harm_vs_legacy": candidate_mae - legacy_mae,
        }
    result["groups"] = groups
    return result


def _view_with_control(context: Mapping[str, Any], outputs: Mapping[int, Mapping[str, Any]],
                       ranges: Mapping[int, tuple[int, int]], name: str) -> dict[str, Any]:
    candidate = _view_metrics(context, [outputs[c]["weighted"] for c in ranges], ranges, name)
    shuffled = _view_metrics(context, [outputs[c]["shuffle"] for c in ranges], ranges, name + "_shuffle")
    candidate["metrics"]["within_cycle_shuffle"] = shuffled["metrics"]["candidate"]
    candidate["gain_vs_shuffle"] = shuffled["metrics"]["candidate"] - candidate["metrics"]["candidate"]
    candidate["shuffle_rows_sha256"] = shuffled["rows_sha256"]
    return candidate


def _write_binary_once(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if file_sha256(path) != hashlib.sha256(data).hexdigest():
            raise RuntimeError(f"Refusing to overwrite immutable N04 visual: {path}")
        return
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _plot_domain(oof: Mapping[int, Mapping[str, Any]], gates: Mapping[str, Any]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    bins = np.linspace(0, 1, 21)
    for ax, cutoff in zip(axes, PILOT_ORIGINS):
        item = oof[cutoff]
        labels = np.asarray(item["labels"], dtype=np.int8)
        probability = np.asarray(item["probability"], dtype=np.float64)
        ax.hist(probability[labels == 0], bins=bins, density=True, alpha=.62,
                label="Source (lebih lama)", color="#4c78a8")
        ax.hist(probability[labels == 1], bins=bins, density=True, alpha=.55,
                label="Pseudo-recent (4 cycle akhir)", color="#f58518")
        aucs = ", ".join(f"{value:.3f}" for value in gates["by_cutoff"][str(cutoff)]["heldout_fold_aucs"])
        ax.set(title=f"Cutoff {cutoff}; fold AUC {aucs}",
               xlabel="Probabilitas pseudo-recent OOF", ylabel="Densitas")
        ax.set_xlim(0, 1)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    figure.suptitle("N04 — OOF domain probability overlap (fit prefixes only)")
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=160)
    plt.close(figure)
    _write_binary_once(REPORT_ROOT / "domain_probability_overlap.png", buffer.getvalue())


def _plot_primary(context: Mapping[str, Any], outputs: Mapping[int, Mapping[str, Any]]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    for ax, cutoff, (low, high), label in zip(
        axes, PILOT_ORIGINS, ((56, 59), (68, 71)), ("W1: 55 → 56–59", "W4: 67 → 68–71")
    ):
        ranges = {cutoff: (low, high)}
        candidate = _view_metrics(context, [outputs[cutoff]["weighted"]], ranges, label)
        shuffled = _view_metrics(context, [outputs[cutoff]["shuffle"]], ranges, label + "_shuffle")
        names = ("R22_ORIGINAL", "Weighted", "Within-cycle shuffle", "Legacy")
        values = (candidate["metrics"]["R22_ORIGINAL"], candidate["metrics"]["candidate"],
                  shuffled["metrics"]["candidate"], candidate["metrics"]["protected_legacy"])
        bars = ax.bar(names, values, color=("#4c78a8", "#54a24b", "#f58518", "#9d755d"))
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
        ax.set_title(label)
        ax.set_ylabel("MAE")
        ax.tick_params(axis="x", labelrotation=15, labelsize=8)
        ax.grid(axis="y", alpha=.2)
    figure.suptitle("N04 primary windows — lower MAE is better")
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=160)
    plt.close(figure)
    _write_binary_once(REPORT_ROOT / "primary_window_mae.png", buffer.getvalue())


def _summary_text(result: Mapping[str, Any]) -> str:
    lines = ["# N04 — Covariate weighting beyond time allocation", "",
             f"Status: {result['status']}", ""]
    if result.get("reason"):
        lines.extend([str(result["reason"]), ""])
    if result.get("pre_task_gates"):
        lines.extend(["## Domain overlap and pre-task gates", "",
                      "| Cutoff | OOF heldout AUCs | ESS fraction | Source clipped | Gate |",
                      "|---:|---:|---:|---:|:---:|"])
        for cutoff, item in result["pre_task_gates"]["by_cutoff"].items():
            diag = item["weight_diagnostics"]
            aucs = ", ".join(f"{value:.3f}" for value in item["heldout_fold_aucs"])
            lines.append(f"| {cutoff} | {aucs} | {diag['effective_sample_size_fraction']:.3f} "
                         f"| {diag['source_fraction_at_or_beyond_clip_bounds']:.3f} "
                         f"| {'PASS' if item['pass'] else 'FAIL'} |")
    if result.get("views"):
        lines.extend(["", "## Task metrics", "",
                      "| View | n | Candidate MAE | R22 MAE | Shuffled MAE | Gain vs R22 | Gain vs shuffle |",
                      "|---|---:|---:|---:|---:|---:|---:|"])
        for name, view in result["views"].items():
            lines.append(f"| {name} | {view['n']:,} | {view['metrics']['candidate']:.6f} "
                         f"| {view['metrics']['R22_ORIGINAL']:.6f} "
                         f"| {view['metrics']['within_cycle_shuffle']:.6f} "
                         f"| {view['gain_vs_R22']:+.6f} | {view['gain_vs_shuffle']:+.6f} |")
    if result.get("gates"):
        lines.extend(["", "## Frozen task gates", ""])
        for name, passed in result["gates"].items():
            lines.append(f"- {name}: {'PASS' if passed else 'FAIL'}")
    lines.extend(["", "OOF domain probabilities were fit by cycle-parity folds. The recent/source labels "
                  "were derived only inside each fit prefix; the domain learner excluded cycle_code, record_id "
                  "and task targets. Score labels were used only for temporal evaluation. Test and Kaggle were not used.",
                  ""])
    return "\n".join(lines)


def _write_result(result: Mapping[str, Any]) -> None:
    _write_once(REPORT_ROOT / "results.json", result)
    _write_text_once(REPORT_ROOT / "SUMMARY.md", _summary_text(result))


def _write_text_once(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = value.encode("utf-8")
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"Refusing to overwrite immutable N04 report: {path}")
        return
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _finish_reservation() -> None:
    events = read_events(LEDGER_PATH)
    pending = [event for event in events if event.get("event") == "FIT_STARTED"
               and event.get("reservation_key") == "N04:PILOT"
               and not any(done.get("fit_id") == event.get("fit_id")
                           and done.get("event") in {"FIT_SAVED", "FIT_FAILED"} for done in events)]
    if pending:
        raise RuntimeError(f"N04 cannot release unstarted slots with pending fit(s): {pending}")
    release_unused(LEDGER_PATH, card_id=CARD_ID, stage="PILOT")


def _final_metrics(context: Mapping[str, Any], outputs: Mapping[int, Mapping[str, Any]],
                   oof: Mapping[int, Mapping[str, Any]], pre_task: Mapping[str, Any],
                   signature: str) -> dict[str, Any]:
    primary_ranges = {55: (56, 59), 67: (68, 71)}
    h12_ranges = {55: (56, 67), 67: (68, 79)}
    views = {
        "W1_55_to_56_59": _view_with_control(context, outputs, {55: (56, 59)}, "W1_55_to_56_59"),
        "W4_67_to_68_71": _view_with_control(context, outputs, {67: (68, 71)}, "W4_67_to_68_71"),
        "primary_pooled": _view_with_control(context, outputs, primary_ranges, "primary_pooled"),
        "H12_pooled": _view_with_control(context, outputs, h12_ranges, "H12_pooled"),
    }
    if views["primary_pooled"]["n"] != 9_064 or views["H12_pooled"]["n"] != 38_588:
        raise RuntimeError("N04 primary/H12 denominators differ from frozen STANDARD queue")
    primary, h12 = views["primary_pooled"], views["H12_pooled"]
    gates = {
        "primary_gain_vs_R22_ge_0_03": primary["gain_vs_R22"] >= .03,
        "H12_harm_vs_R22_le_0_05": h12["metrics"]["candidate"] - h12["metrics"]["R22_ORIGINAL"] <= .05,
        "each_primary_window_harm_vs_R22_le_0_5": all(
            views[name]["metrics"]["candidate"] - views[name]["metrics"]["R22_ORIGINAL"] <= .5
            for name in ("W1_55_to_56_59", "W4_67_to_68_71")
        ),
        "primary_gain_vs_within_cycle_shuffle_ge_0_01": primary["gain_vs_shuffle"] >= .01,
    }
    group_checks = {}
    for view_name, view in views.items():
        for group_name, group in view["groups"].items():
            if group["supported"]:
                group_checks[f"{view_name}:{group_name}"] = (
                    group["harm_vs_R22"] <= .25 and group["harm_vs_legacy"] <= .25
                )
    gates["all_supported_missing_tail_groups_harm_le_0_25"] = all(group_checks.values())
    status = "PILOT_PASS_PENDING_QUEUE_SELECTION" if all(gates.values()) else "RETIRED_PERFORMANCE"
    _plot_domain(oof, pre_task)
    _plot_primary(context, outputs)
    result = {
        "schema_version": 1, "card_id": CARD_ID, "status": status,
        "freeze_signature": signature, "pre_task_gates": dict(pre_task),
        "domain_oof": {
            str(cutoff): {
                "fit_rows": int(len(item["rows"])),
                "source_rows": int(np.sum(item["labels"] == 0)),
                "recent_rows": int(np.sum(item["labels"] == 1)),
                "fit_cycle_min": int(item["cycles"].min()), "fit_cycle_max": int(item["cycles"].max()),
                "heldout_fold_aucs": [float(value) for value in item["fold_aucs"]],
                "overall_oof_auc_descriptive": float(item["overall_oof_auc"]),
                "weight_diagnostics": dict(item["weight_diagnostics"]),
            } for cutoff, item in oof.items()
        },
        "views": views, "gates": gates, "supported_group_gate_checks": group_checks,
        "budget": ledger_summary(LEDGER_PATH),
        "bundle_receipts": {
            str(cutoff): {key: outputs[cutoff][key].receipt["receipt_sha256"]
                          for key in ("weighted", "shuffle")} for cutoff in PILOT_ORIGINS
        },
        "official_test_used": False, "test_labels_read": False, "kaggle_calls": 0,
    }
    _write_result(result)
    return result


def _skip_report(context: Mapping[str, Any], oof: Mapping[int, Mapping[str, Any]],
                 pre_task: Mapping[str, Any], signature: str) -> dict[str, Any]:
    _plot_domain(oof, pre_task)
    result = {
        "schema_version": 1, "card_id": CARD_ID, "status": "SKIPPED_SCIENTIFIC_PREFLIGHT",
        "reason": "At least one cutoff failed the frozen domain-overlap or weight-stability gates; task estimators were not fit.",
        "freeze_signature": signature, "pre_task_gates": dict(pre_task),
        "domain_oof": {
            str(cutoff): {
                "fit_rows": int(len(item["rows"])),
                "source_rows": int(np.sum(item["labels"] == 0)),
                "recent_rows": int(np.sum(item["labels"] == 1)),
                "heldout_fold_aucs": [float(value) for value in item["fold_aucs"]],
                "overall_oof_auc_descriptive": float(item["overall_oof_auc"]),
                "weight_diagnostics": dict(item["weight_diagnostics"]),
            } for cutoff, item in oof.items()
        },
        "budget": ledger_summary(LEDGER_PATH),
        "official_test_used": False, "test_labels_read": False, "kaggle_calls": 0,
    }
    _write_result(result)
    return result


def run_pilot() -> dict[str, Any]:
    frozen, context = _verify_freeze()
    reserve_card(LEDGER_PATH, card_id=CARD_ID, stage="PILOT", fits=PILOT_FITS, category="pilot")
    domain_bundles: dict[int, list[Any]] = {}
    try:
        # Complete and save the four auxiliary fits before evaluating the pre-task gates.
        for cutoff in PILOT_ORIGINS:
            domain_bundles[cutoff] = [
                _domain_fold(context, frozen, cutoff, parity) for parity in DOMAIN_FOLDS
            ]
        oof = {cutoff: _domain_oof(context, cutoff, domain_bundles[cutoff])
               for cutoff in PILOT_ORIGINS}
        pre_task = _pre_task_gates(oof)
        if not pre_task["all_cutoffs_pass"]:
            _finish_reservation()
            result = _skip_report(context, oof, pre_task, frozen["freeze_signature"])
            result["budget"] = ledger_summary(LEDGER_PATH)
            return {"status": result["status"], "pre_task_gates": pre_task,
                    "budget": result["budget"]}
        outputs = _task_outputs(context, frozen, oof)
        # All eight slots are now terminal. Release/report accounting before writing final results.
        _finish_reservation()
        result = _final_metrics(context, outputs, oof, pre_task, frozen["freeze_signature"])
        result["budget"] = ledger_summary(LEDGER_PATH)
        return {"status": result["status"], "views": result["views"], "gates": result["gates"],
                "budget": result["budget"]}
    except ResourceDeferred as exc:
        # Keep the idempotent full-card reservation open so the frozen lane can be
        # revisited after other pilot work if RAM later becomes available.
        _write_once(REPORT_ROOT / "resource_deferred.json", {
            "status": "DEFERRED_RESOURCE", "message": str(exc),
            "budget": ledger_summary(LEDGER_PATH), "resume_without_refit_if_bundle_exists": True,
        })
        return {"status": "DEFERRED_RESOURCE", "message": str(exc),
                "budget": ledger_summary(LEDGER_PATH)}
    except Exception:
        events = read_events(LEDGER_PATH)
        pending = [event for event in events if event.get("event") == "FIT_STARTED"
                   and event.get("reservation_key") == "N04:PILOT"
                   and not any(done.get("fit_id") == event.get("fit_id")
                               and done.get("event") in {"FIT_SAVED", "FIT_FAILED"} for done in events)]
        if not pending:
            _finish_reservation()
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "freeze", "run-pilot"))
    arguments = parser.parse_args(argv)
    if arguments.command == "preflight":
        result = preflight()
    elif arguments.command == "freeze":
        result = freeze()
    else:
        result = run_pilot()
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
