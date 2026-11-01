"""Prepare, freeze, fit and verify the Research Next V1 N03 pilot.

Only official Regression Train is read. Historical R22/R06 predictions are
verified and reused; the only official-data estimator fits are the two shallow
past-only router heads at cutoffs 67 and 75.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping, Sequence
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits

from scripts import optimization_v3_research_next_n00 as n00
from src.models.classification_models import _text_values
from src.models.research_next_budget import (
    finish_fit, ledger_summary, read_events, release_unused,
    reserve_card, start_fit,
)
from src.models.research_next_contract import (
    canonical_sha256, file_sha256, publish_bundle, read_bundle,
)
from src.models.research_next_n03 import (
    CARD_ID, FEATURE_NAMES, MINIMUM_PRESENT_OOF_ROWS, ROUTE_THRESHOLD,
    ROUTER_PARAMETERS, compose_arms, constant_arm_control, fit_router, mae,
    predict_route, router_features,
)
from src.models.round1_models import _number_array
from src.utils.paths import PROJECT_ROOT, task_files


ROOT = PROJECT_ROOT
BATCH = "optimization_v3_research_next_v1"
QUEUE_PATH = ROOT / "experiments/optimization_v3/research_next_v1/queue.json"
READINESS_PATH = ROOT / "reports/optimization_v3/research_next_v1/queue_readiness.json"
CARD_PATH = ROOT / "reports/optimization_v3/research_next_v1/cards/N03.md"
N00_FREEZE_PATH = ROOT / "experiments/optimization_v3/research_next_v1/frozen/N00/amendment_05/freeze.json"
N01_FREEZE_PATH = ROOT / "experiments/optimization_v3/research_next_v1/frozen/N01/freeze.json"
N02_FREEZE_PATH = ROOT / "experiments/optimization_v3/research_next_v1/frozen/N02/amendment_02/freeze.json"
N00_SUMMARY_PATH = ROOT / "reports/optimization_v3/research_next_v1/results/N00/amendment_05/summary.json"
N00_FIXED_ARMS = ROOT / "experiments/runs/optimization_v3_research_next_v1/N00/amendment_05/N03_FIXED_ARMS"
R06_VERIFICATION = ROOT / "reports/optimization_v3/expanded_benchmark_v1_final_evaluation/R06/verification.json"
FREEZE_PATH = ROOT / "experiments/optimization_v3/research_next_v1/frozen/N03/amendment_01/freeze.json"
REPORT_ROOT = ROOT / "reports/optimization_v3/research_next_v1/results/N03/amendment_01"
BUNDLE_ROOT = ROOT / "experiments/runs/optimization_v3_research_next_v1/N03/amendment_01"
RECOVERY_ROOT = BUNDLE_ROOT / "recovery"
LEDGER_PATH = ROOT / "experiments/runs/optimization_v3_research_next_v1/budget.jsonl"
OLD_LEDGER_PATH = ROOT / "experiments/runs/optimization_v3_batch1/batch_ledger.jsonl"
ORIGINS = (55, 59, 63, 67, 71, 75)
PILOT_ORIGINS = (67, 75)
PILOT_FITS = 2
THREADS = 2
JOB_CAP_BYTES = 6 * 1024**3
FREE_RESERVE_BYTES = 2 * 1024**3
MINIMUM_FREE_BYTES = 4 * 1024**3
FIXED_WORKING_BYTES = 512 * 1024**2


class ScientificSkip(RuntimeError):
    """A declared support gate makes the lane ineligible before official fitting."""


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _is_ancestor(commit: str) -> bool:
    return subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"],
                          cwd=ROOT, check=False).returncode == 0


def _dependencies() -> dict[str, str]:
    names = ("numpy", "pandas", "scipy", "scikit-learn", "psutil", "matplotlib", "threadpoolctl")
    return {"python": platform.python_version(), "platform": platform.platform(),
            **{name: importlib.metadata.version(name) for name in names}}


def _source_paths() -> list[Path]:
    return [ROOT / rel for rel in (
        "scripts/optimization_v3_research_next_preflight.py",
        "scripts/optimization_v3_research_next_n00.py",
        "scripts/optimization_v3_research_next_n03.py",
        "src/models/research_next_contract.py",
        "src/models/research_next_budget.py",
        "src/models/research_next_n03.py",
        "src/models/round1_models.py",
        "src/models/classification_models.py",
        "src/models/optimization_v3_batch1_r22.py",
        "src/models/optimization_v3_batch1_r06.py",
        "src/models/regression_research_text.py",
        "src/models/regression_round2_models.py",
        "src/utils/paths.py",
        "tests/test_research_next_n03.py",
    )]


def _input_paths(train_path: Path) -> list[Path]:
    paths = [QUEUE_PATH, READINESS_PATH, CARD_PATH, N00_FREEZE_PATH, N00_SUMMARY_PATH,
             N01_FREEZE_PATH, N02_FREEZE_PATH, OLD_LEDGER_PATH, R06_VERIFICATION, train_path,
             N00_FIXED_ARMS / "arrays.npz", N00_FIXED_ARMS / "receipt.json"]
    for cutoff in ORIGINS:
        for card in ("R22", "R06"):
            artifact = n00.cache_path(card, cutoff)
            paths.extend((artifact, artifact.with_suffix(".json")))
    return paths


def _load_train() -> tuple[Path, pd.DataFrame, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    path = Path(task_files("regression")["train"])
    frame = pd.read_csv(path, usecols=["record_id", "cycle_code", "signal_stream", "impact_load",
                                      "entry_friction", "text_load", "threat_rank"])
    if (len(frame) != 50_608 or not frame.record_id.is_unique
            or not np.array_equal(np.unique(frame.cycle_code.to_numpy(dtype=np.int64)), np.arange(47, 80))):
        raise RuntimeError("N03 official Regression Train schema, IDs or cycle range differs")
    y = frame["threat_rank"].to_numpy(dtype=np.float64)
    if not np.isfinite(y).all() or np.any((y < 0) | (y > 100)):
        raise RuntimeError("N03 Train target must be finite in [0,100]")
    _docs, missing = _text_values(frame["signal_stream"])
    numeric = {name: _number_array(frame[name])[0] for name in
               ("impact_load", "entry_friction", "text_load")}
    return path, frame, y, np.asarray(missing, dtype=bool), numeric


def _r06_report_status() -> dict[str, Any]:
    report = json.loads(R06_VERIFICATION.read_text(encoding="utf-8"))
    if report.get("status") != "PASS":
        raise RuntimeError("N03 requires the independent R06 final-evaluation verification PASS")
    for key in ("test_data_used", "test_labels_read", "official_test_used"):
        if report.get(key) is True:
            raise RuntimeError(f"R06 verification declares prohibited {key}")
    return {"status": report.get("status"), "sha256": file_sha256(R06_VERIFICATION),
            "test_data_used": False}


def _load_verified_sources(
    frame: pd.DataFrame, missing: np.ndarray,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any], dict[str, np.ndarray]]:
    """Verify cache lineage and independently reconstruct every R22/R06 pair."""
    r06_verification = _r06_report_status()
    arms: dict[int, dict[str, Any]] = {}
    cache_audit: dict[str, Any] = {}
    for cutoff in ORIGINS:
        r22_bundle = n00._load_artifact("R22", cutoff, frame)
        r06_bundle = n00._load_artifact("R06", cutoff, frame)
        if not np.array_equal(r22_bundle["rows"], r06_bundle["rows"]):
            raise RuntimeError(f"N03 R22/R06 rows differ at origin {cutoff}")
        if not np.array_equal(r22_bundle["arrays"]["score_record_ids"], r06_bundle["arrays"]["score_record_ids"]):
            raise RuntimeError(f"N03 R22/R06 IDs differ at origin {cutoff}")
        r22 = n00.reconstruct_regression(r22_bundle)
        r06 = n00.reconstruct_regression(r06_bundle)
        if not np.allclose(r22["legacy"], r06["legacy"], atol=1e-12, rtol=0):
            raise RuntimeError(f"N03 R22/R06 protected legacy baseline differs at {cutoff}")
        rows = r22["rows"]
        miss = missing[rows]
        a, b, rank22, rank06 = compose_arms(
            r22["legacy"], r22["raw"], r06["raw"], r22["cycles"], miss,
        )
        if not np.allclose(a, r22["prediction"], atol=1e-12, rtol=0):
            raise RuntimeError(f"N03 independently rebuilt arm A differs from R22_ORIGINAL at {cutoff}")
        # Separate SciPy rank reconstruction is compared with N00's stable tied-sort implementation.
        if (not np.allclose(rank22, r22["rank50"], atol=1e-12, rtol=0)
                or not np.allclose(rank06, r06["rank50"], atol=1e-12, rtol=0)):
            raise RuntimeError(f"N03 complete-cycle average-tie rank reconstruction differs at {cutoff}")
        if not np.array_equal(b[miss], a[miss]):
            raise RuntimeError(f"N03 missing fallback is not exact at {cutoff}")
        if cutoff in PILOT_ORIGINS:
            eval_mask = frame.cycle_code.to_numpy(dtype=np.int64)[rows] > cutoff
            expected_end = min(cutoff + (12 if cutoff == 67 else 4), 79)
            eval_mask &= frame.cycle_code.to_numpy(dtype=np.int64)[rows] <= expected_end
            if not np.array_equal(rows[eval_mask], np.flatnonzero(
                    (frame.cycle_code.to_numpy(dtype=np.int64) > cutoff)
                    & (frame.cycle_code.to_numpy(dtype=np.int64) <= expected_end))):
                raise RuntimeError(f"N03 origin {cutoff} score population is not complete-cycle aligned")
        else:
            eval_mask = (frame.cycle_code.to_numpy(dtype=np.int64)[rows] > cutoff)
            eval_mask &= (frame.cycle_code.to_numpy(dtype=np.int64)[rows] <= min(cutoff + 4, 79))
        arms[cutoff] = {
            "rows": rows.astype(np.int64), "ids": r22_bundle["arrays"]["score_record_ids"].astype(str),
            "cycles": r22["cycles"].astype(np.int64), "legacy": r22["legacy"].astype(np.float64),
            "raw_r22": r22["raw"], "raw_r06": r06["raw"], "rank50_r22": rank22,
            "rank50_r06": rank06, "A": a, "B": b, "R06_ORIGINAL": np.clip(
                0.75 * r22["legacy"] + 0.25 * rank06, 0, 100), "missing": miss,
        }
        cache_audit[str(cutoff)] = {
            "rows": int(len(rows)), "fit_rows": r22_bundle["fit_rows"],
            "score_cycle_min": int(r22["cycles"].min()), "score_cycle_max": int(r22["cycles"].max()),
            "R22_npz_sha256": r22_bundle["npz_sha256"], "R22_sidecar_sha256": r22_bundle["sidecar_sha256"],
            "R06_npz_sha256": r06_bundle["npz_sha256"], "R06_sidecar_sha256": r06_bundle["sidecar_sha256"],
            "complete_cycle_rank_match": True, "exact_legacy_match": True,
            "missing_fallback_A_exact": True,
        }
    return arms, {"R06_verification": r06_verification, "origins": cache_audit}, {
        "rows": np.concatenate([arms[o]["rows"] for o in ORIGINS]),
    }


def _fixed_arms_bundle() -> Any:
    receipt_path = N00_FIXED_ARMS / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    identity = receipt.get("identity")
    if (not isinstance(identity, dict) or identity.get("candidate") != "N03_FIXED_ARMS"
            or identity.get("card_id") != "N00" or identity.get("view") != "primary_full_six"
            or identity.get("freeze_signature") != json.loads(N00_FREEZE_PATH.read_text(encoding="utf-8")).get("freeze_signature")
            or identity.get("queue_sha256") != file_sha256(QUEUE_PATH)):
        raise RuntimeError("N00 fixed N03 arms receipt identity is invalid")
    bundle = read_bundle(N00_FIXED_ARMS, identity, task="regression")
    arrays = bundle.arrays
    if len(arrays["score_rows"]) != 38_588 or len(np.unique(arrays["score_rows"])) != 38_588:
        raise RuntimeError("N00 fixed N03 arm bundle must contain 38,588 unique primary rows")
    return bundle


def _validate_n00_fixed_arms(
    frame: pd.DataFrame, fixed: Any, arms: Mapping[int, Mapping[str, Any]], missing: np.ndarray,
) -> dict[str, Any]:
    arrays = fixed.arrays
    rows = arrays["score_rows"].astype(np.int64)
    ids = arrays["score_record_ids"].astype(str)
    if not np.array_equal(ids, frame.record_id.to_numpy(dtype=str)[rows]):
        raise RuntimeError("N00 fixed N03 arm record IDs do not match official Train rows")
    if not np.array_equal(arrays["cycle_code"], frame.cycle_code.to_numpy(dtype=np.int64)[rows]):
        raise RuntimeError("N00 fixed N03 cycle codes do not match official Train rows")
    if not np.array_equal(arrays["stream_missing"].astype(bool), missing[rows]):
        raise RuntimeError("N00 fixed N03 availability mask differs from official Train")
    origins = arrays["origin_cutoff"].astype(np.int64)
    if len(np.unique(rows)) != len(rows):
        raise RuntimeError("N00 fixed N03 primary rows overlap")
    checked = 0
    for cutoff in ORIGINS:
        cache = arms[cutoff]
        primary = (cache["cycles"] > cutoff) & (cache["cycles"] <= min(cutoff + 4, 79))
        lookup = {int(row): i for i, row in enumerate(cache["rows"][primary])}
        bundle_idx = np.flatnonzero(origins == cutoff)
        if len(bundle_idx) != len(lookup):
            raise RuntimeError(f"N00 fixed N03 arm row count differs at origin {cutoff}")
        for index in bundle_idx:
            row = int(rows[index])
            if row not in lookup:
                raise RuntimeError(f"N00 fixed N03 arm row not found in cache origin {cutoff}")
            j = lookup[row]
            if (not np.isclose(arrays["baseline_prediction"][index], cache["A"][primary][j], atol=1e-12, rtol=0)
                    or not np.isclose(arrays["prediction"][index], cache["B"][primary][j], atol=1e-12, rtol=0)):
                raise RuntimeError(f"N00 fixed N03 A/B arm differs from independent cache reconstruction at {cutoff}")
            for name, key in (("legacy", "legacy"), ("raw_r22", "raw_r22"), ("raw_r06", "raw_r06"),
                              ("rank50_r22", "rank50_r22"), ("rank50_r06", "rank50_r06")):
                if not np.isclose(arrays[name][index], cache[key][primary][j], atol=1e-12, rtol=0):
                    raise RuntimeError(f"N00 fixed N03 {name} differs at origin {cutoff}")
            checked += 1
    expected = set(range(56, 80))
    if set(np.unique(arrays["cycle_code"]).tolist()) != expected:
        raise RuntimeError("N00 fixed N03 full-six bundle does not cover cycles 56-79")
    return {"status": "PASS", "rows": int(len(rows)), "checked_rows": checked,
            "unique_rows": True, "IDs_and_cycle_codes_match_train": True,
            "independent_R22_R06_reconstruction": True, "complete_cycle_ranks": True}


def _past_oof(
    fixed: Any, frame: pd.DataFrame, targets: np.ndarray, missing: np.ndarray,
    numeric: Mapping[str, np.ndarray], cutoff: int,
) -> dict[str, Any]:
    a = fixed.arrays
    rows = a["score_rows"].astype(np.int64)
    cycles = a["cycle_code"].astype(np.int64)
    origin = a["origin_cutoff"].astype(np.int64)
    mask = (cycles <= cutoff) & (origin < cutoff) & (~missing[rows])
    selected = rows[mask]
    if len(selected) != len(np.unique(selected)):
        raise RuntimeError(f"N03 past OOF row support overlaps at cutoff {cutoff}")
    x = router_features(
        a["baseline_prediction"][mask], a["prediction"][mask], numeric["impact_load"][selected],
        numeric["entry_friction"][selected], numeric["text_load"][selected],
    )
    delta = np.abs(targets[selected] - a["prediction"][mask]) - np.abs(
        targets[selected] - a["baseline_prediction"][mask])
    if not np.isfinite(delta).all():
        raise RuntimeError("N03 past OOF loss differences are nonfinite")
    return {"rows": selected, "features": x, "loss_difference": delta,
            "n_present": int(len(selected)), "origin_cutoffs": sorted(set(origin[mask].tolist())),
            "cycle_min": int(cycles[mask].min()) if mask.any() else None,
            "cycle_max": int(cycles[mask].max()) if mask.any() else None}


def _resource_estimate(fit_rows: int, score_rows: int, available_bytes: int | None = None) -> dict[str, Any]:
    fit_matrix = int(fit_rows) * len(FEATURE_NAMES) * 8
    score_matrix = int(score_rows) * len(FEATURE_NAMES) * 8
    peak = FIXED_WORKING_BYTES + 3 * fit_matrix + 2 * score_matrix
    available = int(psutil.virtual_memory().available if available_bytes is None else available_bytes)
    allowed = min(JOB_CAP_BYTES, available - FREE_RESERVE_BYTES)
    return {
        "status": "PASS" if available >= MINIMUM_FREE_BYTES and allowed > 0 and peak <= allowed else "BLOCKED_RESOURCE",
        "fit_rows": int(fit_rows), "score_rows": int(score_rows), "fit_matrix_bytes": fit_matrix,
        "score_matrix_bytes": score_matrix, "fixed_working_bytes": FIXED_WORKING_BYTES,
        "estimated_peak_bytes": peak, "estimated_peak_is_observed": False,
        "formula": "512MiB + 3*fit_matrix + 2*score_matrix",
        "available_physical_bytes": available, "minimum_free_bytes": MINIMUM_FREE_BYTES,
        "free_reserve_bytes": FREE_RESERVE_BYTES, "job_cap_bytes": JOB_CAP_BYTES,
        "allowed_bytes": max(0, allowed), "threads": THREADS, "concurrent_jobs": 1,
    }


def prepare_context(*, available_bytes: int | None = None,
                    ledger_context_override: Mapping[str, Any] | None = None) -> dict[str, Any]:
    queue = json.loads(QUEUE_PATH.read_text(encoding="utf-8"))
    card = next((entry for entry in queue["cards"] if entry.get("id") == CARD_ID), None)
    if queue.get("batch_id") != BATCH or not card or card.get("pilot_fits") != PILOT_FITS:
        raise RuntimeError("N03 canonical queue or pilot allocation changed")
    if card.get("profile") != "LATE" or card.get("continuation_fits") != 3:
        raise RuntimeError("N03 LATE profile or continuation budget differs from queue")
    path, frame, target, missing, numeric = _load_train()
    if file_sha256(path) != queue["raw_train_sha256"]["regression"]:
        raise RuntimeError("Official Regression Train hash differs from Research Next queue")
    n00_frozen = json.loads(N00_FREEZE_PATH.read_text(encoding="utf-8"))
    if n00_frozen.get("freeze_signature") != canonical_sha256(n00_frozen.get("identity")):
        raise RuntimeError("N00 immutable freeze signature is invalid")
    n00_source_commit = n00_frozen.get("identity", {}).get("source_commit")
    if not n00_source_commit or not _is_ancestor(n00_source_commit):
        raise RuntimeError("N00 freeze source commit is not an ancestor of current HEAD")
    current_n00_identity = n00.build_freeze().get("identity", {})
    frozen_n00_identity = dict(n00_frozen.get("identity", {}))
    current_n00_identity = dict(current_n00_identity)
    frozen_n00_identity.pop("source_commit", None)
    current_n00_identity.pop("source_commit", None)
    if current_n00_identity != frozen_n00_identity:
        raise RuntimeError("N00 frozen non-commit source/input identity changed")
    inherited = n00_frozen.get("execution_context", {}).get("inherited_old_budget_snapshot", {})
    inherited_hash = inherited.get("sha256")
    if not inherited_hash or file_sha256(OLD_LEDGER_PATH) != inherited_hash:
        raise RuntimeError("Historical ledger hash differs from the frozen N00 inherited snapshot")
    sources, lineage, _ = _load_verified_sources(frame, missing)
    fixed = _fixed_arms_bundle()
    n00_arm_audit = _validate_n00_fixed_arms(frame, fixed, sources, missing)
    support: dict[str, Any] = {}
    resources: dict[str, Any] = {}
    for cutoff in PILOT_ORIGINS:
        past = _past_oof(fixed, frame, target, missing, numeric, cutoff)
        if past["n_present"] < MINIMUM_PRESENT_OOF_ROWS:
            raise ScientificSkip(f"cutoff {cutoff} has only {past['n_present']} present OOF rows; minimum is {MINIMUM_PRESENT_OOF_ROWS}")
        support[str(cutoff)] = {k: past[k] for k in ("n_present", "origin_cutoffs", "cycle_min", "cycle_max")}
        resources[str(cutoff)] = _resource_estimate(past["n_present"], len(sources[cutoff]["rows"]), available_bytes)
    # Full deterministic identity; no RAM, time, new-ledger totals or attempt counters.
    paths = _input_paths(path)
    source_paths = _source_paths()
    missing_files = [str(p) for p in paths + source_paths if not p.is_file()]
    if missing_files:
        raise RuntimeError(f"N03 freeze inputs are absent: {missing_files}")
    input_hashes = {_relative(p): file_sha256(p) for p in paths}
    source_hashes = {_relative(p): file_sha256(p) for p in source_paths}
    queue_signature = file_sha256(QUEUE_PATH)
    identity = {
        "batch_id": BATCH, "card_id": CARD_ID, "profile": "LATE",
        "recipe": {"arms": "A=R22_ORIGINAL; B=.50*legacy+.25*rank50(raw_R22)+.25*rank50(raw_R06), present only; missing exactly A",
                   "rank": "complete cycle; average ties; clip(.50*raw+.50*percentile,0,100)",
                   "features": list(FEATURE_NAMES), "target": "abs(y-B)-abs(y-A), unique primary OOF ending <= cutoff",
                   "router": {"class": "sklearn.ensemble.HistGradientBoostingRegressor", **ROUTER_PARAMETERS},
                   "imputer": "fit-only median; all-missing columns to 0",
                   "minimum_present_oof_rows": MINIMUM_PRESENT_OOF_ROWS,
                   "route": f"B iff predicted loss difference < {ROUTE_THRESHOLD}; missing always A",
                   "constant_control": f"B for all present iff mean past OOF loss difference < {ROUTE_THRESHOLD}; missing always A"},
        "pilot_origins": list(PILOT_ORIGINS), "pilot_fit_count": PILOT_FITS,
        "views": {"primary": [[67, 68, 71], [75, 76, 79]], "h12": [[67, 68, 79]],
                  "expected_primary_rows": 13_441, "expected_h12_rows": 20_903},
        "gates": {"primary_gain_vs_A_min": 0.03, "h12_harm_max": 0.05,
                  "individual_primary_window_harm_max": 0.5, "min_group_n": 100,
                  "tail_missing_harm_max_vs_A_and_legacy": 0.25,
                  "primary_gain_vs_constant_control_min": 0.01},
        "support": support, "n00_fixed_arm_audit": n00_arm_audit,
        "n00_freeze_signature": n00_frozen["freeze_signature"],
        "lineage": lineage, "queue_sha256": queue_signature,
        "official_train_sha256": file_sha256(path), "input_and_source_sha256": {**input_hashes, **source_hashes},
        "dependencies": _dependencies(),
    }
    # Stable identity excludes dynamic execution context and separately checks resource status.
    live_ledger_context = {
        "new_ledger_sha256": file_sha256(LEDGER_PATH), "ledger": ledger_summary(LEDGER_PATH),
    }
    if ledger_context_override is not None:
        live_ledger_context = dict(ledger_context_override)
    execution = {"source_commit": _git_head(), "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
                 "available_bytes": int(psutil.virtual_memory().available if available_bytes is None else available_bytes),
                 **live_ledger_context,
                 "old_ledger_sha256": file_sha256(OLD_LEDGER_PATH), "resources": resources,
                 "test_files_opened": False, "test_labels_read": False, "kaggle_calls": 0}
    return {"identity": identity, "freeze_signature": canonical_sha256(identity),
            "execution_context": execution, "frame": frame, "target": target, "missing": missing,
            "numeric": numeric, "sources": sources, "fixed": fixed, "support": support,
            "resources": resources, "input_hashes": input_hashes, "source_hashes": source_hashes}


def _freeze_payload(context: Mapping[str, Any]) -> dict[str, Any]:
    return {"schema_version": 1, "identity": dict(context["identity"]),
            "freeze_signature": context["freeze_signature"],
            "execution_context": dict(context["execution_context"])}


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if canonical_sha256(old) != canonical_sha256(value):
            raise RuntimeError(f"Refusing to overwrite frozen artifact: {path}")
        return
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.flush()


def freeze() -> dict[str, Any]:
    context = prepare_context()
    payload = _freeze_payload(context)
    _write_once(FREEZE_PATH, payload)
    preflight = {"status": "READY_TO_FIT" if all(x["status"] == "PASS" for x in context["resources"].values())
                 else "DEFERRED_RESOURCE",
                 "freeze_signature": context["freeze_signature"], "support": context["support"],
                 "resources": context["resources"], "lineage": context["identity"]["lineage"],
                 "n00_fixed_arm_audit": context["identity"]["n00_fixed_arm_audit"],
                 "stable_identity_repeat_check": "PASS",
                 "dynamic_context_sensitivity": {"ram_snapshot_changed": True, "ledger_snapshot_changed": True,
                                                  "identity_unchanged": True},
                 "dynamic_execution_context_excluded_from_identity": True}
    _write_once(REPORT_ROOT / "preflight.json", preflight)
    return payload


def _verify_freeze() -> tuple[dict[str, Any], dict[str, Any]]:
    if not FREEZE_PATH.exists():
        raise RuntimeError("N03 freeze is required before any official fit")
    frozen = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if frozen.get("freeze_signature") != canonical_sha256(frozen.get("identity")):
        raise RuntimeError("N03 freeze signature is invalid")
    current = prepare_context()
    if current["identity"] != frozen.get("identity"):
        raise RuntimeError("N03 frozen source/input/recipe identity changed")
    source_commit = frozen.get("execution_context", {}).get("source_commit")
    if not source_commit or not _is_ancestor(source_commit):
        raise RuntimeError("N03 source commit for freeze is not an ancestor of current HEAD")
    if any(row["status"] != "PASS" for row in current["resources"].values()):
        raise RuntimeError("N03 frozen fit no longer satisfies the live resource bounds")
    return frozen, current


def _bundle_identity(signature: str, cutoff: int) -> dict[str, Any]:
    return {"batch_id": BATCH, "card_id": CARD_ID, "freeze_signature": signature,
            "stage": "PILOT", "cutoff": int(cutoff), "task": "regression", "view": "LATE"}


def _bundle_path(cutoff: int) -> Path:
    return BUNDLE_ROOT / f"fit{cutoff}"


def _source_fit_arrays(context: Mapping[str, Any], cutoff: int) -> tuple[dict[str, Any], dict[str, Any]]:
    past = _past_oof(context["fixed"], context["frame"], context["target"], context["missing"],
                     context["numeric"], cutoff)
    score = context["sources"][cutoff]
    score_x = router_features(score["A"], score["B"], context["numeric"]["impact_load"][score["rows"]],
                              context["numeric"]["entry_friction"][score["rows"]],
                              context["numeric"]["text_load"][score["rows"]])
    return past, {**score, "features": score_x}


def _fit_receipt_fit_id(cutoff: int) -> str:
    return f"N03:PILOT:{cutoff}:router"


def _existing_bundle(context: Mapping[str, Any], cutoff: int) -> Any | None:
    path = _bundle_path(cutoff)
    if not path.exists():
        return None
    bundle = read_bundle(path, _bundle_identity(context["freeze_signature"], cutoff), task="regression")
    events = read_events(LEDGER_PATH)
    fit_id = _fit_receipt_fit_id(cutoff)
    starts = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("fit_id") == fit_id]
    if len(starts) != 1:
        raise RuntimeError(f"Existing N03 bundle {fit_id} has no unique budget start receipt")
    if not np.array_equal(bundle.arrays["score_rows"], context["sources"][cutoff]["rows"]):
        raise RuntimeError(f"Existing N03 bundle {fit_id} row IDs changed")
    terminal = [e for e in events if e.get("fit_id") == fit_id
                and e.get("event") in {"FIT_SAVED", "FIT_FAILED"}]
    if terminal and (len(terminal) != 1 or terminal[0].get("event") != "FIT_SAVED"
                     or terminal[0].get("receipt_sha256") != bundle.receipt["receipt_sha256"]):
        raise RuntimeError(f"Existing N03 bundle {fit_id} conflicts with its ledger terminal record")
    return bundle


def _payload(
    context: Mapping[str, Any], cutoff: int, past: Mapping[str, Any], score: Mapping[str, Any],
    prediction: np.ndarray, predicted_delta: np.ndarray, routed: np.ndarray,
    constant: np.ndarray, constant_mean: float, constant_chooses_b: bool,
    medians: np.ndarray, all_missing: np.ndarray, model: Any,
) -> dict[str, Any]:
    rows = score["rows"].astype(np.int64)
    ids = score["ids"].astype(str)
    arrays = {
        "score_rows": rows,
        "score_record_ids": ids,
        "prediction": prediction.astype(np.float64),
        "baseline_prediction": score["A"].astype(np.float64),
        "arm_b": score["B"].astype(np.float64),
        "constant_control": constant.astype(np.float64),
        "protected_legacy": score["legacy"].astype(np.float64),
        "r06_original": score["R06_ORIGINAL"].astype(np.float64),
        "stream_missing": score["missing"].astype(bool),
        "cycle_code": score["cycles"].astype(np.int64),
        "routed_to_b": routed.astype(bool),
        "predicted_loss_difference": predicted_delta.astype(np.float64),
        "origin_cutoff": np.full(len(rows), int(cutoff), dtype=np.int64),
    }
    target_digest = hashlib.sha256(np.asarray(past["loss_difference"], dtype="<f8").tobytes()).hexdigest()
    source_row_digest = hashlib.sha256(np.asarray(past["rows"], dtype="<i8").tobytes()).hexdigest()
    diagnostics = {
        "cutoff": int(cutoff), "router_parameters": dict(ROUTER_PARAMETERS),
        "feature_names": list(FEATURE_NAMES), "threshold": ROUTE_THRESHOLD,
        "past_oof_present_rows": int(past["n_present"]), "past_oof_origin_cutoffs": past["origin_cutoffs"],
        "past_oof_cycle_range": [past["cycle_min"], past["cycle_max"]],
        "past_oof_source_rows_sha256": source_row_digest, "past_loss_difference_sha256": target_digest,
        "fit_only_medians": medians.tolist(), "fit_only_all_missing_columns": all_missing.tolist(),
        "constant_mean_past_loss_difference": float(constant_mean), "constant_control_chooses_b": bool(constant_chooses_b),
        "routed_rows": int(routed.sum()), "score_rows": int(len(rows)),
        "model_n_iter": int(model.n_iter_), "outer_score_labels_passed_to_fit": False,
        "test_data_used": False,
    }
    return {"arrays": arrays, "metadata": diagnostics}


def _independent_reconstruct(bundle: Any, *, cutoff: int) -> dict[str, Any]:
    a = bundle.arrays
    pred = np.where(a["routed_to_b"].astype(bool), a["arm_b"], a["baseline_prediction"])
    if not np.array_equal(pred, a["prediction"]):
        raise RuntimeError(f"N03 {cutoff} route output does not reconstruct from saved arms and route flags")
    missing = a["stream_missing"].astype(bool)
    if np.any(a["routed_to_b"].astype(bool) & missing):
        raise RuntimeError(f"N03 {cutoff} routes a missing stream to arm B")
    if not np.array_equal(a["baseline_prediction"][missing], a["prediction"][missing]):
        raise RuntimeError(f"N03 {cutoff} missing-stream output is not exact A")
    return {key: value for key, value in a.items()}


def _save_recovery(cutoff: int, identity: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    """Persist score arrays before publish so persistence recovery does not refit."""
    RECOVERY_ROOT.mkdir(parents=True, exist_ok=True)
    array_path = RECOVERY_ROOT / f"fit{cutoff}.npz"
    receipt_path = RECOVERY_ROOT / f"fit{cutoff}.json"
    if array_path.exists() or receipt_path.exists():
        raise RuntimeError(f"N03 recovery payload already exists at cutoff {cutoff}; inspect before overwrite")
    descriptor, temporary = tempfile.mkstemp(prefix=f"fit{cutoff}.", suffix=".npz", dir=RECOVERY_ROOT)
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
    receipt = {"identity": dict(identity), "metadata": dict(payload["metadata"]),
               "npz_sha256": file_sha256(array_path)}
    _write_once(receipt_path, receipt)


def _load_recovery(cutoff: int, identity: Mapping[str, Any]) -> dict[str, Any] | None:
    array_path = RECOVERY_ROOT / f"fit{cutoff}.npz"
    receipt_path = RECOVERY_ROOT / f"fit{cutoff}.json"
    if not array_path.exists() and not receipt_path.exists():
        return None
    if not array_path.exists() or not receipt_path.exists():
        raise RuntimeError(f"N03 recovery payload is incomplete for cutoff {cutoff}; preserve and inspect")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("identity") != dict(identity) or receipt.get("npz_sha256") != file_sha256(array_path):
        raise RuntimeError(f"N03 recovery payload failed identity/hash validation at cutoff {cutoff}")
    with np.load(array_path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    return {"arrays": arrays, "metadata": receipt["metadata"]}


def _finish_recovered_fit(cutoff: int, bundle: Any) -> None:
    fit_id = _fit_receipt_fit_id(cutoff)
    events = read_events(LEDGER_PATH)
    starts = [event for event in events if event.get("event") == "FIT_STARTED" and event.get("fit_id") == fit_id]
    if len(starts) != 1:
        raise RuntimeError(f"Recovered N03 {fit_id} has no unique FIT_STARTED record")
    terminal = [event for event in events if event.get("fit_id") == fit_id
                and event.get("event") in {"FIT_SAVED", "FIT_FAILED"}]
    if not terminal:
        finish_fit(LEDGER_PATH, fit_id=fit_id, status="FIT_SAVED",
                   receipt_sha256=bundle.receipt["receipt_sha256"])
    elif len(terminal) != 1 or terminal[0].get("event") != "FIT_SAVED":
        raise RuntimeError(f"Recovered N03 {fit_id} conflicts with its ledger terminal record")


def _contract_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {**payload["arrays"], "diagnostics": payload["metadata"]}


def _origin_report(context: Mapping[str, Any], bundle: Any, cutoff: int) -> dict[str, Any]:
    arrays = _independent_reconstruct(bundle, cutoff=cutoff)
    rows = arrays["score_rows"].astype(np.int64)
    y = context["target"][rows]
    return {
        "schema_version": 1, "card_id": CARD_ID, "cutoff": int(cutoff),
        "fit_rows": int(bundle.receipt["metadata"]["diagnostics"]["past_oof_present_rows"]),
        "score_rows": int(len(rows)), "cycle_min": int(arrays["cycle_code"].min()),
        "cycle_max": int(arrays["cycle_code"].max()),
        "metrics": {"router": mae(y, arrays["prediction"]),
                    "A_R22_ORIGINAL": mae(y, arrays["baseline_prediction"]),
                    "B_fixed_arm": mae(y, arrays["arm_b"]),
                    "constant_control": mae(y, arrays["constant_control"]),
                    "protected_legacy": mae(y, arrays["protected_legacy"]),
                    "R06_ORIGINAL_secondary": mae(y, arrays["r06_original"])},
        "gain_vs_A": mae(y, arrays["baseline_prediction"]) - mae(y, arrays["prediction"]),
        "gain_vs_constant": mae(y, arrays["constant_control"]) - mae(y, arrays["prediction"]),
        "routed_to_B": int(arrays["routed_to_b"].sum()),
        "receipt_sha256": bundle.receipt["receipt_sha256"],
        "independent_prediction_reconstruction": "PASS",
        "outer_score_labels_passed_to_fit": False, "test_data_used": False,
    }


def _write_origin_report(context: Mapping[str, Any], bundle: Any, cutoff: int) -> None:
    _write_once(REPORT_ROOT / f"origin_{cutoff}.json", _origin_report(context, bundle, cutoff))


def _resource_check(context: Mapping[str, Any], cutoff: int) -> dict[str, Any]:
    fit_rows = int(context["support"][str(cutoff)]["n_present"])
    score_rows = len(context["sources"][cutoff]["rows"])
    current = _resource_estimate(fit_rows, score_rows)
    if current["status"] != "PASS":
        raise RuntimeError(f"N03 {cutoff} live resource gate failed: {current}")
    return current


def _run_one(context: Mapping[str, Any], frozen: Mapping[str, Any], cutoff: int) -> Any:
    identity = _bundle_identity(context["freeze_signature"], cutoff)
    existing = _existing_bundle(context, cutoff)
    if existing is not None:
        # A complete output is read and independently reconstructed; no fit is repeated.
        _independent_reconstruct(existing, cutoff=cutoff)
        _finish_recovered_fit(cutoff, existing)
        _write_origin_report(context, existing, cutoff)
        return existing
    recovered = _load_recovery(cutoff, identity)
    if recovered is not None:
        # Recovery payload contains the exact saved score arrays and diagnostics; do not refit.
        publish_bundle(_bundle_path(cutoff), _contract_payload(recovered), identity, task="regression")
        saved = read_bundle(_bundle_path(cutoff), identity, task="regression")
        _independent_reconstruct(saved, cutoff=cutoff)
        _finish_recovered_fit(cutoff, saved)
        _write_origin_report(context, saved, cutoff)
        return saved
    past, score = _source_fit_arrays(context, cutoff)
    if past["n_present"] < MINIMUM_PRESENT_OOF_ROWS:
        raise ScientificSkip(f"N03 {cutoff} past present OOF rows below minimum")
    _resource_check(context, cutoff)
    fit_id = _fit_receipt_fit_id(cutoff)
    start_fit(LEDGER_PATH, card_id=CARD_ID, stage="PILOT", fit_id=fit_id,
              cutoff=cutoff, component="router")
    recovery_saved = False
    try:
        with threadpool_limits(limits=THREADS):
            model, medians, all_missing = fit_router(past["features"], past["loss_difference"])
            prediction, predicted_delta, routed = predict_route(
                model, medians, score["features"], score["A"], score["B"], score["missing"],
                cutoff=cutoff, present_oof_rows=past["n_present"],
            )
        constant, constant_mean, constant_chooses_b = constant_arm_control(
            score["A"], score["B"], score["missing"], past["loss_difference"],
        )
        payload = _payload(context, cutoff, past, score, prediction, predicted_delta, routed,
                           constant, constant_mean, constant_chooses_b, medians, all_missing, model)
        _save_recovery(cutoff, identity, payload)
        recovery_saved = True
        publish_bundle(_bundle_path(cutoff), _contract_payload(payload), identity, task="regression")
        saved = read_bundle(_bundle_path(cutoff), identity, task="regression")
        # Rebuild arm routing and metrics independently before marking the fit saved.
        rebuilt = _independent_reconstruct(saved, cutoff=cutoff)
        target = context["target"][rebuilt["score_rows"].astype(np.int64)]
        _ = mae(target, rebuilt["prediction"])
        finish_fit(LEDGER_PATH, fit_id=fit_id, status="FIT_SAVED",
                   receipt_sha256=saved.receipt["receipt_sha256"])
        _write_origin_report(context, saved, cutoff)
        return saved
    except Exception as exc:
        # Once an estimator has started, the fit remains consumed. No next origin is started.
        events = read_events(LEDGER_PATH)
        started = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("fit_id") == fit_id]
        terminal = [e for e in events if e.get("fit_id") == fit_id and e.get("event") in {"FIT_SAVED", "FIT_FAILED"}]
        if started and not terminal and not recovery_saved:
            finish_fit(LEDGER_PATH, fit_id=fit_id, status="FIT_FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


def _view_metrics(frame: pd.DataFrame, target: np.ndarray, records: Sequence[Mapping[str, np.ndarray]],
                  *, name: str) -> dict[str, Any]:
    rows = np.concatenate([r["score_rows"].astype(np.int64) for r in records])
    if len(rows) != len(np.unique(rows)):
        raise RuntimeError(f"N03 view {name} contains duplicate score rows")
    order = np.argsort(rows, kind="mergesort")
    rows = rows[order]
    pick = lambda key: np.concatenate([r[key] for r in records])[order]
    y = target[rows]
    candidate = pick("prediction")
    a = pick("baseline_prediction")
    b = pick("arm_b")
    constant = pick("constant_control")
    legacy = pick("protected_legacy")
    r06 = pick("r06_original")
    missing = pick("stream_missing").astype(bool)
    output = {"n": int(len(rows)), "rows_sha256": hashlib.sha256(rows.astype("<i8").tobytes()).hexdigest(),
              "metrics": {"router": mae(y, candidate), "A_R22_ORIGINAL": mae(y, a),
                          "B_fixed_arm": mae(y, b), "past_only_constant_control": mae(y, constant),
                          "protected_legacy": mae(y, legacy), "R06_ORIGINAL_secondary": mae(y, r06)},
              "gain_vs_A": mae(y, a) - mae(y, candidate),
              "harm_vs_A": mae(y, candidate) - mae(y, a),
              "gain_vs_constant": mae(y, constant) - mae(y, candidate),
              "stream_missing_n": int(missing.sum()), "routed_to_B_n": int(pick("routed_to_b").sum()),
              "groups": {}}
    masks = {"stream_missing": missing, "target_le5_evaluation_only": y <= 5.0,
             "target_ge95_evaluation_only": y >= 95.0}
    for key, mask in masks.items():
        n = int(mask.sum())
        if n >= 100:
            harm_a = mae(y[mask], candidate[mask]) - mae(y[mask], a[mask])
            harm_legacy = mae(y[mask], candidate[mask]) - mae(y[mask], legacy[mask])
            output["groups"][key] = {"n": n, "supported": True, "harm_vs_A": harm_a,
                                      "harm_vs_legacy": harm_legacy,
                                      "pass": harm_a <= 0.25 and harm_legacy <= 0.25}
        else:
            output["groups"][key] = {"n": n, "supported": False, "pass": None}
    return output


def _evaluate(context: Mapping[str, Any], bundles: Mapping[int, Any]) -> dict[str, Any]:
    frame, target = context["frame"], context["target"]
    arrays = {cutoff: _independent_reconstruct(bundle, cutoff=cutoff) for cutoff, bundle in bundles.items()}
    mask_w4 = arrays[67]["cycle_code"] <= 71
    w4 = {key: (value[mask_w4] if isinstance(value, np.ndarray) and value.ndim == 1
                and len(value) == len(mask_w4) else value)
          for key, value in arrays[67].items()}
    views: dict[str, Any] = {}
    views["W4_67_to_68_71"] = _view_metrics(frame, target, [w4], name="W4")
    views["W6_75_to_76_79"] = _view_metrics(frame, target, [arrays[75]], name="W6")
    views["late_primary"] = _view_metrics(frame, target, [w4, arrays[75]], name="late_primary")
    views["late_h12"] = _view_metrics(frame, target, [arrays[67]], name="late_h12")
    if views["late_primary"]["n"] != 13_441 or views["late_h12"]["n"] != 20_903:
        raise RuntimeError("N03 pilot view row counts differ from the frozen LATE denominators")
    primary_gains = {name: views[name]["gain_vs_A"] for name in ("W4_67_to_68_71", "W6_75_to_76_79")}
    group_checks = [group for view in views.values() for group in view["groups"].values()
                    if group["supported"]]
    routed_count = views["late_primary"]["routed_to_B_n"]
    gates = {
        "late_primary_gain_vs_A_ge_0_03": views["late_primary"]["gain_vs_A"] >= 0.03,
        "late_h12_harm_vs_A_le_0_05": views["late_h12"]["harm_vs_A"] <= 0.05,
        "each_primary_window_harm_le_0_5": all(value >= -0.5 for value in primary_gains.values()),
        "supported_missing_and_target_tail_harm_le_0_25_vs_A_and_legacy": all(g["pass"] for g in group_checks),
        "late_primary_gain_vs_past_only_constant_ge_0_01": views["late_primary"]["gain_vs_constant"] >= 0.01,
        "at_least_one_primary_row_routed_to_B": routed_count > 0,
    }
    return {"views": views, "gates": gates, "all_gates_pass": all(gates.values()),
            "decision": "ADVANCE" if all(gates.values()) else "RETIRED_PERFORMANCE",
            "routed_rows_primary": routed_count, "R06_ORIGINAL_is_secondary_only": True,
            "interpretation": "All results use already observed Train temporal validation labels; no Test, Kaggle or full-Train refit."}


def _write_plots(report: Mapping[str, Any]) -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    order = ("W4_67_to_68_71", "W6_75_to_76_79", "late_primary", "late_h12")
    labels = ("W4\n68–71", "W6\n76–79", "LATE primary", "LATE H12")
    names = ("A_R22_ORIGINAL", "B_fixed_arm", "router", "past_only_constant_control")
    titles = ("A: R22", "B: fixed blend", "Router", "Constant past-only")
    values = np.asarray([[report["views"][view]["metrics"][key] for view in order] for key in names])
    fig, ax = plt.subplots(figsize=(9.5, 5.2), constrained_layout=True)
    x = np.arange(len(order)); width = 0.19
    for i, (label, row) in enumerate(zip(titles, values)):
        ax.bar(x + (i - 1.5) * width, row, width=width, label=label)
    ax.set_xticks(x, labels); ax.set_ylabel("MAE (lebih rendah lebih baik)")
    ax.set_title("N03: router dan pembanding pada evaluasi temporal LATE")
    ax.grid(axis="y", alpha=0.22); ax.legend(frameon=False, ncols=2)
    fig.savefig(REPORT_ROOT / "late_mae.png", dpi=160); plt.close(fig)

    cycles, routed, counts = [], [], []
    for cutoff, bundle_path in ((67, _bundle_path(67)), (75, _bundle_path(75))):
        bundle = read_bundle(bundle_path, _bundle_identity(report["freeze_signature"], cutoff), task="regression")
        a = bundle.arrays
        for cycle in np.unique(a["cycle_code"]):
            mask = (a["cycle_code"] == cycle) & (~a["stream_missing"].astype(bool))
            cycles.append(int(cycle)); routed.append(float(a["routed_to_b"][mask].mean()) if mask.any() else 0.0)
            counts.append(int(mask.sum()))
    fig, ax = plt.subplots(figsize=(9.5, 4.8), constrained_layout=True)
    bars = ax.bar(np.asarray(cycles).astype(str), routed, color="#3973ac")
    ax.set_ylim(0, 1); ax.set_ylabel("Proporsi stream-present yang dirutekan ke B")
    ax.set_xlabel("cycle_code (prediksi 67 untuk H12; prediksi 75 untuk siklus 76–79)")
    ax.set_title("Intensitas routing N03 per cycle")
    ax.grid(axis="y", alpha=0.22)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, min(0.98, bar.get_height()+0.025), f"n={count}",
                ha="center", va="bottom", fontsize=8, rotation=90 if count > 5000 else 0)
    fig.savefig(REPORT_ROOT / "routing_by_cycle.png", dpi=160); plt.close(fig)


def _write_report(context: Mapping[str, Any], bundles: Mapping[int, Any], freeze_signature: str) -> dict[str, Any]:
    result = _evaluate(context, bundles)
    fit_receipts = {}
    events = read_events(LEDGER_PATH)
    for cutoff in PILOT_ORIGINS:
        fit_id = _fit_receipt_fit_id(cutoff)
        starts = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("fit_id") == fit_id]
        saved = [e for e in events if e.get("event") == "FIT_SAVED" and e.get("fit_id") == fit_id]
        if len(starts) != 1 or len(saved) != 1:
            raise RuntimeError(f"N03 {fit_id} ledger does not reconcile to one saved fit")
        fit_receipts[str(cutoff)] = {"fit_id": fit_id, "receipt_sha256": saved[0]["receipt_sha256"],
                                     "bundle_receipt_sha256": bundles[cutoff].receipt["receipt_sha256"]}
    result.update({"schema_version": 1, "card_id": CARD_ID, "freeze_signature": freeze_signature,
                   "status": result["decision"], "pilot_origins": list(PILOT_ORIGINS),
                   "fit_receipts": fit_receipts, "lineage": context["identity"]["lineage"],
                   "source_hashes": context["identity"]["input_and_source_sha256"],
                   "budget": ledger_summary(LEDGER_PATH), "test_files_opened": False,
                   "test_labels_read": False, "kaggle_calls": 0, "full_train_refit": False})
    path = REPORT_ROOT / "results.json"
    _write_once(path, result)
    _write_plots(result)
    summary = (
        "# N03 — Past-only router over R22/R06\n\n"
        f"**Status:** `{result['status']}`. Pilot fit the frozen router at cutoffs 67 and 75 only.\n\n"
        "## LATE views\n\n"
        "| View | n | Router MAE | R22 A MAE | Fixed B MAE | Constant control MAE | Gain vs A | Gain vs constant |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    for name in ("W4_67_to_68_71", "W6_75_to_76_79", "late_primary", "late_h12"):
        item = result["views"][name]
        summary += (f"| {name} | {item['n']:,} | {item['metrics']['router']:.6f} | "
                    f"{item['metrics']['A_R22_ORIGINAL']:.6f} | {item['metrics']['B_fixed_arm']:.6f} | "
                    f"{item['metrics']['past_only_constant_control']:.6f} | {item['gain_vs_A']:+.6f} | "
                    f"{item['gain_vs_constant']:+.6f} |\n")
    summary += "\n## Gates and interpretation\n\n"
    for name, passed in result["gates"].items():
        summary += f"- `{name}`: {'PASS' if passed else 'FAIL'}\n"
    summary += ("\nThe router uses only prior OOF rows to learn the loss-difference head and imputer. "
                "The B arm is fixed in advance; missing-stream rows always receive exact R22 arm A. "
                "R06_ORIGINAL is reported as a secondary cache comparator and does not set the control. "
                "Group masks use target values only after prediction for evaluation.\n")
    _write_text_once(REPORT_ROOT / "SUMMARY.md", summary)
    return result


def _write_text_once(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode("utf-8")
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"Refusing to overwrite immutable report: {path}")
        return
    with path.open("xb") as handle:
        handle.write(data)


def run_pilot() -> dict[str, Any]:
    frozen, context = _verify_freeze()
    if any(item["status"] != "PASS" for item in context["resources"].values()):
        raise RuntimeError("N03 resource preflight no longer passes")
    reservation = reserve_card(LEDGER_PATH, card_id=CARD_ID, stage="PILOT",
                               fits=PILOT_FITS, category="pilot")
    bundles: dict[int, Any] = {}
    try:
        for cutoff in PILOT_ORIGINS:
            bundles[cutoff] = _run_one(context, frozen, cutoff)
    finally:
        # release_unused refuses to close a reservation while any started fit is pending.
        events = read_events(LEDGER_PATH)
        pending = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("reservation_key") == "N03:PILOT"
                   and not any(done.get("fit_id") == e.get("fit_id") and done.get("event") in {"FIT_SAVED", "FIT_FAILED"}
                               for done in events)]
        if not pending:
            release_unused(LEDGER_PATH, card_id=CARD_ID, stage="PILOT")
    if len(bundles) != PILOT_FITS:
        raise RuntimeError("N03 pilot did not produce both frozen origin bundles")
    result = _write_report(context, bundles, frozen["freeze_signature"])
    return {"status": result["status"], "freeze_signature": frozen["freeze_signature"],
            "views": result["views"], "gates": result["gates"], "budget": result["budget"]}


def preflight() -> dict[str, Any]:
    first = prepare_context()
    # Simulated context overrides are ephemeral and never written to the budget ledger.
    second = prepare_context(
        available_bytes=int(first["execution_context"]["available_bytes"]) - 1234567,
        ledger_context_override={"new_ledger_sha256": "0" * 64,
                                 "ledger": {"simulated_test_snapshot": True, "event_rows": -1}},
    )
    if first["freeze_signature"] != second["freeze_signature"]:
        raise RuntimeError("N03 stable identity changed when RAM/new-ledger context changed")
    if (first["execution_context"]["available_bytes"] == second["execution_context"]["available_bytes"]
            or first["execution_context"]["new_ledger_sha256"] == second["execution_context"]["new_ledger_sha256"]):
        raise RuntimeError("N03 identity sensitivity check did not vary RAM and ledger context")
    return {"status": "READY_TO_FREEZE" if all(x["status"] == "PASS" for x in first["resources"].values())
            else "DEFERRED_RESOURCE", "freeze_signature": first["freeze_signature"],
            "support": first["support"], "resources": first["resources"],
            "lineage": first["identity"]["lineage"],
            "n00_fixed_arm_audit": first["identity"]["n00_fixed_arm_audit"],
            "stable_identity_repeat_check": "PASS",
            "dynamic_context_sensitivity": {"ram_snapshot_changed": True, "ledger_snapshot_changed": True,
                                             "identity_unchanged": True},
            "official_fits_started": 0,
            "test_files_opened": False, "kaggle_calls": 0}


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
