"""Freeze, run, and report the Research Next V1 N05 interval diagnostic."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
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
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

from scripts import optimization_v3_research_next_n00 as n00
from src.models.classification_models import _text_values
from src.models.regression_research_text import HYBRID_FEATURES, STRUCTURED_FEATURES
from src.models.research_next_budget import (
    finish_fit, ledger_summary, read_events, release_unused, reserve_card, start_fit,
)
from src.models.research_next_contract import (
    canonical_sha256, file_sha256, publish_bundle, read_bundle,
)
from src.models.research_next_n04 import R22WeightedDesign
from src.models.research_next_n05 import (
    ALPHAS, CARD_ID, QUANTILE_PARAMETERS, array_sha256, fit_quantile_head,
    interval_metrics, make_interval, past_residual_interval, pinball_loss,
    point_model_eligibility, predict_quantile_head, safe_interval_metrics,
    width_rank_groups,
)
from src.utils.paths import PROJECT_ROOT, task_files


ROOT = PROJECT_ROOT
BATCH = "optimization_v3_research_next_v1"
QUEUE_PATH = ROOT / "experiments/optimization_v3/research_next_v1/queue.json"
READINESS_PATH = ROOT / "reports/optimization_v3/research_next_v1/queue_readiness.json"
CARD_PATH = ROOT / "reports/optimization_v3/research_next_v1/cards/N05.md"
N00_FREEZE_PATH = ROOT / "experiments/optimization_v3/research_next_v1/frozen/N00/amendment_05/freeze.json"
N00_SUMMARY_PATH = ROOT / "reports/optimization_v3/research_next_v1/results/N00/amendment_05/summary.json"
OLD_LEDGER_PATH = ROOT / "experiments/runs/optimization_v3_batch1/batch_ledger.jsonl"
LEDGER_PATH = ROOT / "experiments/runs/optimization_v3_research_next_v1/budget.jsonl"
FREEZE_ROOT = ROOT / "experiments/optimization_v3/research_next_v1/frozen/N05/amendment_01"
PREFLIGHT_PATH = FREEZE_ROOT / "preflight.json"
FREEZE_PATH = FREEZE_ROOT / "freeze.json"
REPORT_ROOT = ROOT / "reports/optimization_v3/research_next_v1/results/N05/amendment_01"
BUNDLE_ROOT = ROOT / "experiments/runs/optimization_v3_research_next_v1/N05/amendment_01"
RECOVERY_ROOT = BUNDLE_ROOT / "recovery"
ORIGINS = (67, 75)
PAST_PRIMARY_ORIGINS = (55, 59, 63, 67, 71, 75)
PILOT_FITS = 4
THREADS = 2
JOB_CAP_BYTES = 6 * 1024**3
FREE_RESERVE_BYTES = 2 * 1024**3
MINIMUM_FREE_BYTES = 4 * 1024**3
TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")


class ResourceDeferred(RuntimeError):
    """The current machine state no longer satisfies the frozen resource bound."""


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _is_ancestor(commit: str) -> bool:
    return subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"],
                          cwd=ROOT, check=False).returncode == 0


def _dependencies() -> dict[str, str]:
    packages = ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "psutil",
                "matplotlib", "threadpoolctl")
    return {"python": platform.python_version(), "platform": platform.platform(),
            **{package: importlib.metadata.version(package) for package in packages}}


def _source_paths() -> list[Path]:
    return [ROOT / relative for relative in (
        "scripts/optimization_v3_research_next_n00.py",
        "scripts/optimization_v3_research_next_n05.py",
        "scripts/optimization_v3_research_next_n05_verify.py",
        "src/models/research_next_contract.py",
        "src/models/research_next_budget.py",
        "src/models/research_next_n04.py",
        "src/models/research_next_n05.py",
        "src/models/round1_models.py",
        "src/models/classification_models.py",
        "src/models/regression_research_text.py",
        "src/models/optimization_v3_batch1_r22.py",
        "src/utils/paths.py",
        "tests/test_research_next_n05.py",
    )]


def _load_train() -> tuple[Path, pd.DataFrame, np.ndarray]:
    path = Path(task_files("regression")["train"])
    columns = ["record_id", "threat_rank", *HYBRID_FEATURES]
    frame = pd.read_csv(path, usecols=list(dict.fromkeys(columns)))
    cycles = pd.to_numeric(frame.cycle_code, errors="raise").to_numpy(dtype=np.float64)
    if (len(frame) != 50_608 or frame.record_id.isna().any() or not frame.record_id.is_unique
            or not np.isfinite(cycles).all() or not np.equal(cycles, np.floor(cycles)).all()
            or not np.array_equal(np.unique(cycles.astype(np.int64)), np.arange(47, 80))):
        raise RuntimeError("N05 official Regression Train schema, IDs or cycle range differs")
    target = pd.to_numeric(frame.threat_rank, errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(target).all() or np.any((target < 0) | (target > 100)):
        raise RuntimeError("N05 Train target must be finite in [0,100]")
    return path, frame, target


def _input_paths(train_path: Path) -> list[Path]:
    paths = [QUEUE_PATH, READINESS_PATH, CARD_PATH, N00_FREEZE_PATH, N00_SUMMARY_PATH,
             OLD_LEDGER_PATH, train_path]
    for cutoff in PAST_PRIMARY_ORIGINS:
        artifact = n00.cache_path("R22", cutoff)
        paths.extend((artifact, artifact.with_suffix(".json")))
    return paths


def _load_r22_sources(frame: pd.DataFrame) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    queue = json.loads(QUEUE_PATH.read_text(encoding="utf-8"))
    blocks = [tuple(map(int, block)) for block in queue["views"]["full_six"]]
    sources: dict[int, dict[str, Any]] = {}
    audit: dict[str, Any] = {}
    cycles_all = frame.cycle_code.to_numpy(dtype=np.int64)
    ids_all = frame.record_id.to_numpy(dtype=str)
    for cutoff in PAST_PRIMARY_ORIGINS:
        bundle = n00._load_artifact("R22", cutoff, frame)
        reconstructed = n00.reconstruct_regression(bundle)
        rows = reconstructed["rows"].astype(np.int64)
        ids = bundle["arrays"]["score_record_ids"].astype(str)
        cycles = reconstructed["cycles"].astype(np.int64)
        horizon = 12 if cutoff in (55, 67) else 4
        expected = np.flatnonzero((cycles_all > cutoff)
                                  & (cycles_all <= min(cutoff + horizon, 79)))
        if (not np.array_equal(rows, expected) or not np.array_equal(ids, ids_all[rows])
                or not np.array_equal(cycles, cycles_all[rows])):
            raise RuntimeError(f"N05 R22/{cutoff} cached forecast is not aligned to official Train")
        point = reconstructed["prediction"].astype(np.float64)
        if not np.isfinite(point).all() or np.any((point < 0) | (point > 100)):
            raise RuntimeError(f"N05 R22/{cutoff} point forecast violates [0,100]")
        sources[cutoff] = {"rows": rows, "ids": ids, "cycles": cycles, "R22_ORIGINAL": point}
        primary = [block for block in blocks if block[0] == cutoff]
        audit[str(cutoff)] = {
            "fit_rows": int(bundle["fit_rows"]), "score_rows": int(len(rows)),
            "score_cycle_min": int(cycles.min()), "score_cycle_max": int(cycles.max()),
            "primary_block": list(primary[0]) if primary else None,
            "npz_sha256": bundle["npz_sha256"],
            "sidecar_sha256": bundle["sidecar_sha256"],
            "R22_ORIGINAL_reconstructed": True,
        }
    return sources, audit


def _past_primary_rows(frame: pd.DataFrame, sources: Mapping[int, Mapping[str, Any]],
                       cutoff: int) -> tuple[np.ndarray, np.ndarray]:
    queue = json.loads(QUEUE_PATH.read_text(encoding="utf-8"))
    blocks = [tuple(map(int, block)) for block in queue["views"]["full_six"]]
    row_parts: list[np.ndarray] = []
    prediction_parts: list[np.ndarray] = []
    for origin, cycle_start, cycle_end in blocks:
        if origin > cutoff or cycle_end > cutoff:
            continue
        source = sources[origin]
        mask = (source["cycles"] >= cycle_start) & (source["cycles"] <= cycle_end)
        rows = source["rows"][mask]
        if not len(rows):
            raise RuntimeError(f"N05 primary OOF block {origin}/{cycle_start}-{cycle_end} is empty")
        row_parts.append(rows.astype(np.int64))
        prediction_parts.append(source["R22_ORIGINAL"][mask].astype(np.float64))
    if not row_parts:
        raise RuntimeError(f"N05 has no past primary R22 OOF rows for cutoff {cutoff}")
    rows = np.concatenate(row_parts)
    predictions = np.concatenate(prediction_parts)
    cycles_all = frame.cycle_code.to_numpy(dtype=np.int64)
    expected = np.flatnonzero((cycles_all >= 56) & (cycles_all <= cutoff))
    if (len(rows) != len(np.unique(rows)) or not np.array_equal(np.sort(rows), expected)
            or len(predictions) != len(expected) or np.any(cycles_all[rows] > cutoff)):
        raise RuntimeError(f"N05 primary calibration rows are overlapping or incomplete at {cutoff}")
    order = np.argsort(rows, kind="mergesort")
    return rows[order], predictions[order]


def _resource_estimate(frame: pd.DataFrame, cutoff: int, available: int | None = None) -> dict[str, Any]:
    cycles = frame.cycle_code.to_numpy(dtype=np.int64)
    fit_rows = np.flatnonzero(cycles <= cutoff)
    score_rows = np.flatnonzero((cycles > cutoff) & (cycles <= min(cutoff + 12, 79)))
    docs, _ = _text_values(frame.iloc[fit_rows].signal_stream)
    fit_nnz_upper = int(sum(2 * len(TOKEN_RE.findall(doc)) for doc in docs))
    docs_score, _ = _text_values(frame.iloc[score_rows].signal_stream)
    score_nnz_upper = int(sum(2 * len(TOKEN_RE.findall(doc)) for doc in docs_score))
    width_upper = (len(STRUCTURED_FEATURES) + 4 * 3 + 3 * 254 + 2 * 3 + 32 + 1)
    peak = (1024 * 1024**2 + 4 * len(fit_rows) * width_upper * 8
            + 2 * len(score_rows) * width_upper * 8 + 5 * (fit_nnz_upper + score_nnz_upper) * 16)
    physical_available = int(psutil.virtual_memory().available if available is None else available)
    allowed = min(JOB_CAP_BYTES, physical_available - FREE_RESERVE_BYTES)
    return {
        "cutoff": int(cutoff), "fit_rows": int(len(fit_rows)), "score_rows": int(len(score_rows)),
        "transformed_width_upper": int(width_upper), "fit_tfidf_nnz_upper": fit_nnz_upper,
        "score_tfidf_nnz_upper": score_nnz_upper, "estimated_peak_bytes": int(peak),
        "available_physical_bytes": physical_available, "minimum_free_bytes": MINIMUM_FREE_BYTES,
        "free_reserve_bytes": FREE_RESERVE_BYTES, "job_cap_bytes": JOB_CAP_BYTES,
        "allowed_bytes": max(0, allowed), "threads": THREADS, "concurrent_jobs": 1,
        "status": "PASS" if (physical_available >= MINIMUM_FREE_BYTES and peak <= allowed) else "DEFERRED_RESOURCE",
    }


def _build_identity(frame: pd.DataFrame, train_path: Path,
                    cache_audit: Mapping[str, Any], resources: Mapping[str, Any]) -> dict[str, Any]:
    queue = json.loads(QUEUE_PATH.read_text(encoding="utf-8"))
    card = next((item for item in queue["cards"] if item.get("id") == CARD_ID), None)
    if (queue.get("batch_id") != BATCH or not card or card.get("task") != "regression"
            or card.get("kind") != "information" or card.get("profile") != "LATE"
            or card.get("pilot_fits") != PILOT_FITS or card.get("continuation_fits") != 0):
        raise RuntimeError("N05 canonical queue profile or fit allocation changed")
    if file_sha256(train_path) != queue["raw_train_sha256"]["regression"]:
        raise RuntimeError("N05 official Regression Train hash differs from queue")
    n00_frozen = json.loads(N00_FREEZE_PATH.read_text(encoding="utf-8"))
    if n00_frozen.get("freeze_signature") != canonical_sha256(n00_frozen.get("identity")):
        raise RuntimeError("N05 N00 freeze signature is invalid")
    n00_commit = n00_frozen.get("execution_context", {}).get("source_commit")
    if not n00_commit or not _is_ancestor(n00_commit):
        raise RuntimeError("N05 N00 source commit is not an ancestor of HEAD")
    inherited = n00_frozen.get("execution_context", {}).get("inherited_old_budget_snapshot", {})
    if not inherited.get("sha256") or file_sha256(OLD_LEDGER_PATH) != inherited["sha256"]:
        raise RuntimeError("N05 historical ledger differs from the inherited N00 snapshot")
    inputs = _input_paths(train_path)
    sources = _source_paths()
    missing = [str(path) for path in [*inputs, *sources] if not path.is_file()]
    if missing:
        raise RuntimeError(f"N05 freeze inputs/sources are absent: {missing}")
    input_hashes = {_relative(path): file_sha256(path) for path in inputs}
    source_hashes = {_relative(path): file_sha256(path) for path in sources}
    fit_rows = {str(cutoff): int(np.sum(frame.cycle_code.to_numpy(dtype=np.int64) <= cutoff))
                for cutoff in ORIGINS}
    calibration_rows = {str(cutoff): int(np.sum((frame.cycle_code.to_numpy(dtype=np.int64) >= 56)
                                                & (frame.cycle_code.to_numpy(dtype=np.int64) <= cutoff)))
                        for cutoff in ORIGINS}
    return {
        "batch_id": BATCH, "card_id": CARD_ID, "profile": "LATE",
        "lane_kind": "information_only_no_point_model_promotion",
        "recipe": {
            "model": "two LightGBM quantile regressors on exact R22 structured/OHE+TFIDF/SVD32 design",
            "parameters": {**QUANTILE_PARAMETERS, "alphas": list(ALPHAS)},
            "origins": list(ORIGINS), "preprocessing": "one exact R22 design fit per origin, shared by both heads",
            "fit_population": "all official Regression Train rows with cycle_code <= outer origin",
            "score_population": "exact verified R22 score rows; 67 scores 68-79 and 75 scores 76-79",
            "target_boundary": "outer score labels are used only after prediction for evaluation; never passed to fit",
            "interval": "raw q10/q90 crossing recorded; each endpoint clipped to [0,100]; endpoint pair sorted",
            "control": "R22_ORIGINAL point plus q10/q90 of signed residuals from unique past primary OOF rows <= cutoff; numpy linear quantiles; clip and sort",
            "calibration_rows_by_cutoff": calibration_rows,
            "primary_views": [[67, 68, 71], [75, 76, 79]],
            "h12_late_view": [67, 68, 79],
            "subgroups": ["signal_stream availability", "evaluation-only threat_rank <=5 / 5<y<95 / >=95"],
            "width_error": "Spearman and fixed five rank groups versus absolute R22_ORIGINAL error, descriptive only",
            "no_point_prediction_selection": True,
        },
        "pilot_origins": list(ORIGINS), "pilot_fit_count": PILOT_FITS,
        "fit_rows_by_cutoff": fit_rows, "cache_lineage": dict(cache_audit),
        "queue_sha256": file_sha256(QUEUE_PATH), "official_train_sha256": file_sha256(train_path),
        "input_and_source_sha256": {**input_hashes, **source_hashes},
        "resource_bounds": {str(k): {key: value for key, value in item.items()
                                     if key not in {"available_physical_bytes", "allowed_bytes", "status"}}
                            for k, item in resources.items()},
        "dependencies": _dependencies(),
    }


def prepare_context(*, available_bytes: int | None = None,
                    ledger_context_override: Mapping[str, Any] | None = None) -> dict[str, Any]:
    path, frame, target = _load_train()
    sources, cache_audit = _load_r22_sources(frame)
    calibration: dict[int, dict[str, Any]] = {}
    for cutoff in ORIGINS:
        rows, prediction = _past_primary_rows(frame, sources, cutoff)
        residual_target = target[rows]
        if len(rows) < 2_000:
            raise RuntimeError(f"N05 past primary R22 OOF support is below 2000 at {cutoff}")
        calibration[cutoff] = {
            "rows": rows, "prediction": prediction, "target": residual_target,
            "row_indices_sha256": array_sha256(rows, "<i8"),
            "record_ids_sha256": array_sha256(frame.record_id.to_numpy(dtype=str)[rows].astype("U"), "U"),
            "target_sha256": array_sha256(residual_target),
        }
    resources = {str(cutoff): _resource_estimate(frame, cutoff, available_bytes) for cutoff in ORIGINS}
    identity = _build_identity(frame, path, cache_audit, resources)
    current_ledger = {"ledger_sha256": file_sha256(LEDGER_PATH), "ledger": ledger_summary(LEDGER_PATH)}
    if ledger_context_override is not None:
        current_ledger = dict(ledger_context_override)
    return {
        "identity": identity, "freeze_signature": canonical_sha256(identity),
        "execution_context": {
            "prepared_at_utc": datetime.now(timezone.utc).isoformat(), "source_commit": _git_head(),
            "available_bytes": int(psutil.virtual_memory().available if available_bytes is None else available_bytes),
            **current_ledger, "old_ledger_sha256": file_sha256(OLD_LEDGER_PATH),
            "resources": resources, "test_files_opened": False, "test_labels_read": False,
            "kaggle_calls": 0,
        },
        "path": path, "frame": frame, "target": target, "sources": sources,
        "cache_audit": cache_audit, "calibration": calibration, "resources": resources,
    }


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        if canonical_sha256(current) != canonical_sha256(value):
            raise RuntimeError(f"Refusing to overwrite immutable N05 artifact: {_relative(path)}")
        return
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _synthetic_smoke() -> dict[str, Any]:
    from src.models.research_next_contract import publish_bundle as publish, read_bundle as read

    rng = np.random.default_rng(20261101)
    x = rng.normal(size=(120, 4))
    y = np.clip(50 + 8 * x[:, 0] - 4 * x[:, 1] + rng.normal(size=120), 0, 100)
    parameters = {**QUANTILE_PARAMETERS, "n_estimators": 6, "min_child_samples": 4}
    alpha_checks = {}
    for alpha in ALPHAS:
        model, state = fit_quantile_head(x, y, alpha=alpha, cutoff=47, parameters=parameters)
        prediction = predict_quantile_head(model, x[:12])
        if state["alpha"] != alpha or state["parameters"]["alpha"] != alpha:
            raise RuntimeError("N05 synthetic quantile head alpha mismatch")
        alpha_checks[str(alpha)] = {"fit_rows": len(y), "score_rows": len(prediction),
                                   "alpha": alpha, "finite_predictions": bool(np.isfinite(prediction).all())}
    crossing = make_interval([80, 20], [60, 90])
    if crossing["raw_crossing_rate"] != 0.5 or not np.array_equal(crossing["lower"], [60, 20]):
        raise RuntimeError("N05 synthetic crossing handling failed")
    empty = safe_interval_metrics([], [], [])
    if empty["n"] != 0 or empty["coverage_80"] is not None:
        raise RuntimeError("N05 synthetic empty-group handling failed")
    if point_model_eligibility()["eligible"]:
        raise RuntimeError("N05 information-only lane was incorrectly marked point-model eligible")
    with tempfile.TemporaryDirectory(prefix="kasdad_n05_contract_") as temporary:
        destination = Path(temporary) / "synthetic_bundle"
        identity = {"batch_id": BATCH, "card_id": CARD_ID, "synthetic": True}
        payload = {
            "arrays": {
                "score_rows": np.arange(1, 5, dtype=np.int64),
                "score_record_ids": np.asarray(["S1", "S2", "S3", "S4"], dtype="U2"),
                "prediction": np.asarray([40.0, 45.0, 50.0, 55.0]),
                "baseline_prediction": np.asarray([42.0, 47.0, 52.0, 57.0]),
                "raw_quantile_prediction": np.asarray([39.0, 44.0, 49.0, 54.0]),
            },
            "metadata": {"diagnostics": {"synthetic_only": True}},
        }
        publish(destination, {**payload["arrays"], "diagnostics": payload["metadata"]["diagnostics"]},
                identity, task="regression")
        recovered = read(destination, identity, task="regression")
        resumed = read(destination, identity, task="regression")
        if (recovered.receipt["receipt_sha256"] != resumed.receipt["receipt_sha256"]
                or not np.array_equal(recovered.arrays["prediction"], resumed.arrays["prediction"])):
            raise RuntimeError("N05 synthetic persistence/resume check failed")
    return {
        "status": "PASS", "official_predictor_fits": 0,
        "synthetic_qa_fits": len(ALPHAS), "alpha_checks": alpha_checks,
        "crossing_handling": {"raw_crossing_rate": crossing["raw_crossing_rate"],
                               "lower": crossing["lower"].tolist(), "upper": crossing["upper"].tolist()},
        "empty_group": empty, "point_model_eligibility": point_model_eligibility(),
        "contract_persistence_readback_resume": "PASS",
    }


def preflight() -> dict[str, Any]:
    smoke = _synthetic_smoke()
    first = prepare_context()
    second = prepare_context(
        available_bytes=max(MINIMUM_FREE_BYTES, first["execution_context"]["available_bytes"] - 1234567),
        ledger_context_override={"ledger_sha256": "0" * 64, "ledger": {"simulated": True}},
    )
    if first["freeze_signature"] != second["freeze_signature"]:
        raise RuntimeError("N05 stable identity depends on live RAM or budget ledger context")
    ready = all(item["status"] == "PASS" for item in first["resources"].values())
    result = {
        "status": "READY_TO_FREEZE" if ready else "DEFERRED_RESOURCE",
        "freeze_signature": first["freeze_signature"],
        "fit_rows_by_cutoff": first["identity"]["fit_rows_by_cutoff"],
        "calibration_rows": {str(c): int(len(first["calibration"][c]["rows"])) for c in ORIGINS},
        "cache_lineage": first["cache_audit"], "resources": first["resources"],
        "stable_identity_repeat_check": "PASS", "synthetic_end_to_end": smoke,
        "official_predictor_fits_started": 0, "test_files_opened": False, "kaggle_calls": 0,
    }
    _write_once(PREFLIGHT_PATH, result)
    _write_once(REPORT_ROOT / "synthetic_smoke.json", smoke)
    return result


def freeze() -> dict[str, Any]:
    context = prepare_context()
    if any(item["status"] != "PASS" for item in context["resources"].values()):
        raise ResourceDeferred("N05 frozen pre-fit resource estimate does not pass")
    if not PREFLIGHT_PATH.is_file():
        raise RuntimeError("N05 persisted preflight is required before freeze")
    preflight_report = json.loads(PREFLIGHT_PATH.read_text(encoding="utf-8"))
    if (preflight_report.get("status") != "READY_TO_FREEZE"
            or preflight_report.get("freeze_signature") != context["freeze_signature"]):
        raise RuntimeError("N05 persisted preflight is not ready or has a different identity")
    execution = dict(context["execution_context"])
    execution["preflight_sha256"] = file_sha256(PREFLIGHT_PATH)
    payload = {"schema_version": 1, "identity": context["identity"],
               "freeze_signature": context["freeze_signature"], "execution_context": execution}
    _write_once(FREEZE_PATH, payload)
    return {"status": "FROZEN", "freeze_signature": context["freeze_signature"],
            "path": _relative(FREEZE_PATH), "source_commit": execution["source_commit"]}


def _verify_freeze() -> tuple[dict[str, Any], dict[str, Any]]:
    if not FREEZE_PATH.is_file():
        raise RuntimeError("N05 freeze is required before official fits")
    frozen = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if frozen.get("freeze_signature") != canonical_sha256(frozen.get("identity")):
        raise RuntimeError("N05 freeze signature is invalid")
    context = prepare_context()
    if context["identity"] != frozen.get("identity"):
        raise RuntimeError("N05 frozen source, recipe or input identity changed")
    commit = frozen.get("execution_context", {}).get("source_commit")
    if not commit or not _is_ancestor(commit):
        raise RuntimeError("N05 source commit for freeze is not an ancestor of HEAD")
    if (not PREFLIGHT_PATH.is_file()
            or frozen.get("execution_context", {}).get("preflight_sha256") != file_sha256(PREFLIGHT_PATH)):
        raise RuntimeError("N05 preflight hash differs from freeze")
    if any(item["status"] != "PASS" for item in context["resources"].values()):
        raise ResourceDeferred("N05 live resource check no longer passes")
    return frozen, context


def _fit_id(cutoff: int, alpha: float) -> str:
    return f"N05:PILOT:{cutoff}:q{int(round(alpha * 100)):02d}"


def _bundle_identity(signature: str, cutoff: int, alpha: float) -> dict[str, Any]:
    return {"batch_id": BATCH, "card_id": CARD_ID, "freeze_signature": signature,
            "stage": "PILOT", "cutoff": int(cutoff), "component": f"quantile_alpha_{alpha:.1f}"}


def _bundle_path(cutoff: int, alpha: float, *, recovery: bool = False) -> Path:
    root = RECOVERY_ROOT if recovery else BUNDLE_ROOT
    return root / f"fit{cutoff}" / f"q{int(round(alpha * 100)):02d}"


def _resource_check(estimate: int, label: str) -> dict[str, Any]:
    available = int(psutil.virtual_memory().available)
    allowed = min(JOB_CAP_BYTES, available - FREE_RESERVE_BYTES)
    receipt = {"label": label, "available_bytes": available, "estimated_bytes": int(estimate),
               "allowed_bytes": max(0, allowed), "minimum_free_bytes": MINIMUM_FREE_BYTES,
               "reserve_bytes": FREE_RESERVE_BYTES, "job_cap_bytes": JOB_CAP_BYTES,
               "threads": THREADS, "concurrent_jobs": 1}
    if available < MINIMUM_FREE_BYTES or estimate > allowed:
        raise ResourceDeferred(f"N05 {label} needs {estimate} bytes; {available} available")
    return receipt


def _control_arrays(context: Mapping[str, Any], cutoff: int) -> dict[str, Any]:
    source = context["sources"][cutoff]
    calibration = context["calibration"][cutoff]
    return past_residual_interval(source["R22_ORIGINAL"], calibration["target"],
                                  calibration["prediction"])


def _independent_bundle(bundle: Any, context: Mapping[str, Any], signature: str,
                        cutoff: int, alpha: float) -> None:
    source = context["sources"][cutoff]
    arrays = bundle.arrays
    metadata = bundle.receipt["metadata"].get("diagnostics", {})
    expected_control = _control_arrays(context, cutoff)
    expected_rows = source["rows"].astype(np.int64)
    expected_ids = source["ids"].astype(str)
    expected_cycles = source["cycles"].astype(np.int64)
    if (bundle.receipt.get("task") != "regression"
            or bundle.receipt.get("identity") != _bundle_identity(signature, cutoff, alpha)
            or not np.array_equal(arrays["score_rows"], expected_rows)
            or not np.array_equal(arrays["score_record_ids"].astype(str), expected_ids)
            or not np.array_equal(arrays["cycle_code"], expected_cycles)
            or not np.array_equal(arrays["baseline_prediction"], source["R22_ORIGINAL"])
            or not np.array_equal(arrays["control_lower"], expected_control["lower"])
            or not np.array_equal(arrays["control_upper"], expected_control["upper"])):
        raise RuntimeError(f"N05 {cutoff}/q{alpha} bundle rows, IDs, baseline, or control differ")
    raw = arrays["raw_quantile_prediction"].astype(np.float64)
    if (not np.array_equal(arrays["prediction"], np.clip(raw, 0.0, 100.0))
            or metadata.get("alpha") != alpha or metadata.get("cutoff") != cutoff
            or metadata.get("fit_rows") != int(np.sum(context["frame"].cycle_code.to_numpy() <= cutoff))
            or metadata.get("score_target_passed_to_fit") is not False
            or metadata.get("test_data_used") is not False
            or metadata.get("quantile_parameters", {}).get("alpha") != alpha):
        raise RuntimeError(f"N05 {cutoff}/q{alpha} bundle raw prediction or fit audit differs")
    fit_rows = np.flatnonzero(context["frame"].cycle_code.to_numpy(dtype=np.int64) <= cutoff)
    if metadata.get("fit_row_indices_sha256") != array_sha256(fit_rows, "<i8"):
        raise RuntimeError(f"N05 {cutoff}/q{alpha} fit row lineage differs")


def _payload_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {**payload["arrays"], **payload["metadata"]}


def _fit_one(context: Mapping[str, Any], frozen: Mapping[str, Any], cutoff: int,
             alpha: float, design: R22WeightedDesign, x_fit: np.ndarray,
             x_score: np.ndarray, fit_rows: np.ndarray, fit_target: np.ndarray,
             score_target_never_passed: bool = True) -> Any:
    if not score_target_never_passed:
        raise RuntimeError("N05 score labels must not be supplied to the estimator path")
    fit_id = _fit_id(cutoff, alpha)
    signature = frozen["freeze_signature"]
    identity = _bundle_identity(signature, cutoff, alpha)
    path = _bundle_path(cutoff, alpha)
    recovery_path = _bundle_path(cutoff, alpha, recovery=True)

    if path.is_dir():
        saved = read_bundle(path, identity, task="regression")
        _independent_bundle(saved, context, signature, cutoff, alpha)
        finish_fit(LEDGER_PATH, fit_id=fit_id, status="FIT_SAVED",
                   receipt_sha256=saved.receipt["receipt_sha256"])
        return saved
    if recovery_path.is_dir():
        recovered = read_bundle(recovery_path, identity, task="regression")
        _independent_bundle(recovered, context, signature, cutoff, alpha)
        payload = {"arrays": recovered.arrays, "metadata": recovered.receipt["metadata"]}
        publish_bundle(path, _payload_contract(payload), identity, task="regression")
        saved = read_bundle(path, identity, task="regression")
        _independent_bundle(saved, context, signature, cutoff, alpha)
        finish_fit(LEDGER_PATH, fit_id=fit_id, status="FIT_SAVED",
                   receipt_sha256=saved.receipt["receipt_sha256"])
        return saved
    events = read_events(LEDGER_PATH)
    started = [event for event in events if event.get("event") == "FIT_STARTED" and event.get("fit_id") == fit_id]
    if started:
        raise RuntimeError(f"N05 {fit_id} already started without a recoverable output; no duplicate fit")

    diagnostics = design.audit_state()
    estimate = 1024 * 1024**2 + 4 * x_fit.nbytes + 2 * x_score.nbytes
    resource = _resource_check(estimate, f"quantile alpha={alpha:.1f} cutoff={cutoff}")
    start_fit(LEDGER_PATH, card_id=CARD_ID, stage="PILOT", fit_id=fit_id,
              cutoff=cutoff, component=f"quantile_alpha_{alpha:.1f}")
    recovery_saved = False
    try:
        with threadpool_limits(limits=THREADS):
            model, model_state = fit_quantile_head(x_fit, fit_target, alpha=alpha, cutoff=cutoff)
            raw_prediction = predict_quantile_head(model, x_score)
        source = context["sources"][cutoff]
        control = _control_arrays(context, cutoff)
        fit_ids = context["frame"].record_id.to_numpy(dtype=str)[fit_rows].astype("U")
        score_frame = context["frame"].iloc[source["rows"]]
        arrays = {
            "score_rows": source["rows"].astype(np.int64),
            "score_record_ids": source["ids"].astype(str),
            "prediction": np.clip(raw_prediction, 0.0, 100.0).astype(np.float64),
            "baseline_prediction": source["R22_ORIGINAL"].astype(np.float64),
            "raw_quantile_prediction": raw_prediction.astype(np.float64),
            "control_lower": control["lower"].astype(np.float64),
            "control_upper": control["upper"].astype(np.float64),
            "cycle_code": source["cycles"].astype(np.int64),
            "stream_missing": _text_values(score_frame["signal_stream"])[1].astype(bool),
        }
        state = {
            **model_state, "alpha": float(alpha), "cutoff": int(cutoff),
            "fit_row_indices_sha256": array_sha256(fit_rows, "<i8"),
            "fit_record_ids_sha256": array_sha256(fit_ids, "U"),
            "fit_target_sha256": array_sha256(fit_target),
            "transform_state": diagnostics,
            "score_design_sha256": array_sha256(x_score),
            "score_rows": int(len(source["rows"])),
            "score_target_passed_to_fit": False, "test_data_used": False,
            "control_residual_rows": int(context["calibration"][cutoff]["rows"].size),
            "control_residual_rows_sha256": context["calibration"][cutoff]["row_indices_sha256"],
            "control_residual_q10": control["past_residual_q10"],
            "control_residual_q90": control["past_residual_q90"],
            "live_resource_check": resource,
            "training_design_fit_population_is_train_prefix": True,
        }
        state["quantile_parameters"] = {**QUANTILE_PARAMETERS, "alpha": float(alpha)}
        payload = {"arrays": arrays, "metadata": {"diagnostics": state}}
        publish_bundle(recovery_path, _payload_contract(payload), identity, task="regression")
        recovery_saved = True
        recovery_bundle = read_bundle(recovery_path, identity, task="regression")
        _independent_bundle(recovery_bundle, context, signature, cutoff, alpha)
        publish_bundle(path, _payload_contract(payload), identity, task="regression")
        saved = read_bundle(path, identity, task="regression")
        _independent_bundle(saved, context, signature, cutoff, alpha)
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


def _score_view(context: Mapping[str, Any], model_parts: Mapping[str, np.ndarray],
                label: str) -> dict[str, Any]:
    target = np.asarray(model_parts["target"], dtype=np.float64)
    low_raw = np.asarray(model_parts["q10_raw"], dtype=np.float64)
    high_raw = np.asarray(model_parts["q90_raw"], dtype=np.float64)
    candidate = make_interval(low_raw, high_raw)
    control = make_interval(model_parts["control_lower_raw"], model_parts["control_upper_raw"])
    point = np.asarray(model_parts["point"], dtype=np.float64)
    if not (target.shape == point.shape == candidate["lower"].shape == control["lower"].shape):
        raise RuntimeError(f"N05 {label} metric arrays are misaligned")
    candidate_metrics = interval_metrics(target, candidate["lower"], candidate["upper"], point)
    control_metrics = interval_metrics(target, control["lower"], control["upper"], point)
    candidate_metrics["raw_quantile_pinball_q10"] = pinball_loss(target, low_raw, 0.1)
    candidate_metrics["raw_quantile_pinball_q90"] = pinball_loss(target, high_raw, 0.9)
    candidate_metrics["raw_crossing_rate"] = candidate["raw_crossing_rate"]
    candidate_metrics["clipped_endpoint_rows"] = candidate["clipped_endpoint_rows"]
    result: dict[str, Any] = {
        "label": label, "n": int(len(target)), "candidate": candidate_metrics,
        "past_residual_control": control_metrics,
        "candidate_minus_control": {
            "coverage_80": candidate_metrics["coverage_80"] - control_metrics["coverage_80"],
            "mean_width": candidate_metrics["mean_width"] - control_metrics["mean_width"],
            "pinball_lower_alpha_0_1": (candidate_metrics["pinball_lower_alpha_0_1"]
                                         - control_metrics["pinball_lower_alpha_0_1"]),
            "pinball_upper_alpha_0_9": (candidate_metrics["pinball_upper_alpha_0_9"]
                                         - control_metrics["pinball_upper_alpha_0_9"]),
        },
    }
    cycle = np.asarray(model_parts["cycle"], dtype=np.int64)
    result["per_cycle"] = {}
    for code in np.unique(cycle):
        mask = cycle == code
        result["per_cycle"][str(int(code))] = {
            "candidate": safe_interval_metrics(target[mask], candidate["lower"][mask],
                                               candidate["upper"][mask], point[mask]),
            "past_residual_control": safe_interval_metrics(target[mask], control["lower"][mask],
                                                           control["upper"][mask], point[mask]),
        }
    result["subgroups"] = {}
    missing = np.asarray(model_parts["stream_missing"], dtype=bool)
    subgroup_masks = {
        "signal_stream_missing": missing,
        "signal_stream_present": ~missing,
        "evaluation_target_le_5": target <= 5,
        "evaluation_target_5_to_95_exclusive": (target > 5) & (target < 95),
        "evaluation_target_ge_95": target >= 95,
    }
    for name, mask in subgroup_masks.items():
        result["subgroups"][name] = {
            "candidate": safe_interval_metrics(target[mask], candidate["lower"][mask],
                                               candidate["upper"][mask], point[mask]),
            "past_residual_control": safe_interval_metrics(target[mask], control["lower"][mask],
                                                           control["upper"][mask], point[mask]),
        }
    width_bin = width_rank_groups(candidate["upper"] - candidate["lower"], groups=5)
    result["candidate_width_rank_groups"] = {}
    for index in range(1, 6):
        mask = width_bin == index
        result["candidate_width_rank_groups"][str(index)] = {
            "candidate": safe_interval_metrics(target[mask], candidate["lower"][mask],
                                               candidate["upper"][mask], point[mask]),
            "n": int(mask.sum()),
        }
    return result


def _metrics(context: Mapping[str, Any], frozen: Mapping[str, Any],
             bundles: Mapping[int, Mapping[float, Any]]) -> dict[str, Any]:
    queue = json.loads(QUEUE_PATH.read_text(encoding="utf-8"))
    full: dict[int, dict[str, np.ndarray]] = {}
    by_origin: dict[str, Any] = {}
    for cutoff in ORIGINS:
        q10 = bundles[cutoff][0.1].arrays
        q90 = bundles[cutoff][0.9].arrays
        if not np.array_equal(q10["score_rows"], q90["score_rows"]):
            raise RuntimeError(f"N05 q10/q90 cutoff {cutoff} score rows differ")
        source = context["sources"][cutoff]
        control = _control_arrays(context, cutoff)
        parts = {
            "rows": source["rows"], "target": context["target"][source["rows"]],
            "point": source["R22_ORIGINAL"], "q10_raw": q10["raw_quantile_prediction"],
            "q90_raw": q90["raw_quantile_prediction"],
            "control_lower_raw": control["lower"], "control_upper_raw": control["upper"],
            "cycle": source["cycles"],
            "stream_missing": _text_values(context["frame"].iloc[source["rows"]]["signal_stream"])[1].astype(bool),
        }
        full[cutoff] = parts
        primary_block = next(tuple(map(int, block)) for block in queue["views"]["LATE"]["primary"]
                             if int(block[0]) == cutoff)
        primary_mask = (source["cycles"] >= primary_block[1]) & (source["cycles"] <= primary_block[2])
        primary_parts = {key: value[primary_mask] for key, value in parts.items()}
        by_origin[str(cutoff)] = {
            "primary": _score_view(context, primary_parts, f"LATE primary {primary_block[1]}-{primary_block[2]}"),
            "scored_population": _score_view(context, parts, f"origin {cutoff} score rows"),
            "primary_cycle_block": list(primary_block),
            "fit_rows": int(np.sum(context["frame"].cycle_code.to_numpy(dtype=np.int64) <= cutoff)),
            "calibration": {
                "rows": int(context["calibration"][cutoff]["rows"].size),
                "cycles": [56, int(cutoff)],
                "q10_residual": control["past_residual_q10"],
                "q90_residual": control["past_residual_q90"],
                "row_indices_sha256": context["calibration"][cutoff]["row_indices_sha256"],
                "target_sha256": context["calibration"][cutoff]["target_sha256"],
            },
            "candidate_raw_crossing_rate_full_score_rows": make_interval(parts["q10_raw"], parts["q90_raw"])["raw_crossing_rate"],
        }

    late_parts = []
    for cutoff in ORIGINS:
        block = next(tuple(map(int, item)) for item in queue["views"]["LATE"]["primary"]
                     if int(item[0]) == cutoff)
        source = context["sources"][cutoff]
        mask = (source["cycles"] >= block[1]) & (source["cycles"] <= block[2])
        parts = full[cutoff]
        late_parts.append({key: value[mask] for key, value in parts.items()})
    pooled_parts = {key: np.concatenate([item[key] for item in late_parts])
                    for key in late_parts[0]}
    pooled_rows = pooled_parts["rows"].astype(np.int64)
    if len(pooled_rows) != len(np.unique(pooled_rows)):
        raise RuntimeError("N05 LATE primary pooled score rows overlap")
    late_primary = _score_view(context, pooled_parts, "pooled unique LATE primary")

    late_h12_cutoff = 67
    h12_mask = ((full[late_h12_cutoff]["cycle"] > late_h12_cutoff)
                & (full[late_h12_cutoff]["cycle"] <= late_h12_cutoff + 12))
    h12_parts = {key: value[h12_mask] for key, value in full[late_h12_cutoff].items()}
    h12_view = _score_view(context, h12_parts, "LATE H12 from origin 67")
    return {
        "status": "COMPLETE_INFORMATION", "card_id": CARD_ID,
        "question": "Does conditional interval width characterize uncertainty beyond constant past-residual intervals?",
        "freeze_signature": frozen["freeze_signature"],
        "recipe": frozen["identity"]["recipe"], "origins": by_origin,
        "late_primary_pooled": late_primary, "late_h12_from_67": h12_view,
        "late_primary_unique_rows": int(len(pooled_rows)),
        "point_model_selection": point_model_eligibility(),
        "lineage": frozen["identity"]["cache_lineage"],
        "safety": {"official_test_used": False, "test_labels_read": False,
                   "sample_submission_used": False, "kaggle_calls": 0,
                   "outer_score_labels_passed_to_fit": False},
        "budget": ledger_summary(LEDGER_PATH),
    }


def _make_plot(result: Mapping[str, Any], path: Path) -> None:
    if path.exists():
        return
    origins = result["origins"]
    cycle_values: list[int] = []
    candidate_coverage: list[float] = []
    control_coverage: list[float] = []
    width_group_error: list[float] = []
    width_group_width: list[float] = []
    for cutoff in ORIGINS:
        record = origins[str(cutoff)]["primary"]
        for cycle, row in record["per_cycle"].items():
            cycle_values.append(int(cycle))
            candidate_coverage.append(float(row["candidate"]["coverage_80"]))
            control_coverage.append(float(row["past_residual_control"]["coverage_80"]))
        for group in record["candidate_width_rank_groups"].values():
            if group["n"]:
                width_group_error.append(float(group["candidate"]["mean_absolute_reference_error"]))
                width_group_width.append(float(group["candidate"]["mean_width"]))
    order = np.argsort(cycle_values)
    cycle_values = [cycle_values[i] for i in order]
    candidate_coverage = [candidate_coverage[i] for i in order]
    control_coverage = [control_coverage[i] for i in order]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].plot(cycle_values, candidate_coverage, marker="o", label="Conditional q10/q90")
    axes[0].plot(cycle_values, control_coverage, marker="s", label="Past residual interval")
    axes[0].axhline(0.8, color="black", linestyle="--", linewidth=1, label="Nominal 80%")
    axes[0].set(title="Coverage per LATE primary cycle", xlabel="cycle_code", ylabel="Observed coverage")
    axes[0].set_ylim(0, 1.02)
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].plot(width_group_width, width_group_error, marker="o", color="#d17c00")
    axes[1].set(title="Observed R22 error by predicted width quintile",
                xlabel="Mean conditional interval width", ylabel="Mean absolute R22 error")
    axes[1].grid(alpha=0.25)
    fig.suptitle("N05 — conditional interval information (LATE primary)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _independent_verify_report(context: Mapping[str, Any], frozen: Mapping[str, Any],
                               result: Mapping[str, Any]) -> dict[str, Any]:
    del context, frozen, result
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.optimization_v3_research_next_n05_verify"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    return json.loads(completed.stdout)


def run_pilot() -> dict[str, Any]:
    frozen, context = _verify_freeze()
    reserve_card(LEDGER_PATH, card_id=CARD_ID, stage="PILOT", fits=PILOT_FITS, category="pilot")
    saved_by_origin: dict[int, dict[float, Any]] = {}
    try:
        for cutoff in ORIGINS:
            frame = context["frame"]
            cycles = frame.cycle_code.to_numpy(dtype=np.int64)
            fit_rows = np.flatnonzero(cycles <= cutoff)
            score_rows = context["sources"][cutoff]["rows"].astype(np.int64)
            if not np.array_equal(fit_rows, np.flatnonzero(cycles <= cutoff)):
                raise RuntimeError("N05 fit rows differ from frozen temporal prefix")
            x_fit_frame = frame.iloc[fit_rows].loc[:, list(HYBRID_FEATURES)]
            x_score_frame = frame.iloc[score_rows].loc[:, list(HYBRID_FEATURES)]
            fit_target = context["target"][fit_rows].astype(np.float64)
            design = R22WeightedDesign(cutoff)
            x_fit = design.fit_transform(x_fit_frame)
            x_score = design.transform(x_score_frame)
            if int(design.audit_state()["fit_rows"]) != len(fit_rows):
                raise RuntimeError("N05 R22 design preprocessing did not use the full fit prefix")
            saved_by_origin[cutoff] = {}
            for alpha in ALPHAS:
                saved_by_origin[cutoff][alpha] = _fit_one(
                    context, frozen, cutoff, alpha, design, x_fit, x_score,
                    fit_rows, fit_target,
                )
        release_unused(LEDGER_PATH, card_id=CARD_ID, stage="PILOT")
        result = _metrics(context, frozen, saved_by_origin)
        _write_once(REPORT_ROOT / "results.json", result)
        _make_plot(result, REPORT_ROOT / "interval_diagnostic.png")
        verification = _independent_verify_report(context, frozen, result)
        _write_once(REPORT_ROOT / "verification.json", verification)
        summary = {
            "status": result["status"], "question": result["question"],
            "origins": {key: {
                "primary_coverage_q": value["primary"]["candidate"]["coverage_80"],
                "primary_coverage_control": value["primary"]["past_residual_control"]["coverage_80"],
                "primary_width_q": value["primary"]["candidate"]["mean_width"],
                "primary_width_control": value["primary"]["past_residual_control"]["mean_width"],
                "primary_width_error_spearman": value["primary"]["candidate"]["width_absolute_error_spearman"],
                "raw_crossing_rate": value["primary"]["candidate"]["raw_crossing_rate"],
            } for key, value in result["origins"].items()},
            "late_primary_coverage_q": result["late_primary_pooled"]["candidate"]["coverage_80"],
            "late_primary_coverage_control": result["late_primary_pooled"]["past_residual_control"]["coverage_80"],
            "late_primary_width_q": result["late_primary_pooled"]["candidate"]["mean_width"],
            "late_primary_width_control": result["late_primary_pooled"]["past_residual_control"]["mean_width"],
            "fit_count": 4, "verification": verification["status"],
            "budget": ledger_summary(LEDGER_PATH), "point_model_selection": point_model_eligibility(),
        }
        _write_once(REPORT_ROOT / "SUMMARY.json", summary)
        return summary
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
                  and event.get("reservation_key") == f"{CARD_ID}:PILOT"]
        pending = [event for event in starts if not any(done.get("fit_id") == event.get("fit_id")
                                                        and done.get("event") in {"FIT_SAVED", "FIT_FAILED"}
                                                        for done in events)]
        terminal_count = sum(any(done.get("fit_id") == event.get("fit_id")
                                 and done.get("event") in {"FIT_SAVED", "FIT_FAILED"}
                                 for done in events) for event in starts)
        if not pending and terminal_count == PILOT_FITS:
            release_unused(LEDGER_PATH, card_id=CARD_ID, stage="PILOT")
        raise


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
