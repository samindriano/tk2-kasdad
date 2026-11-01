"""Research Next V1 N01: class-by-volume sample-weight ablation.

This runner has three explicit stages: preflight, freeze, and run-pilot. It
reads official Classification Train only and writes immutable cache bundles.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import psutil

from scripts.optimization_v3_research_next_n00 import (
    _load_artifact,
    cache_path,
    macro_f1_from_confusion,
    reconstruct_classification,
)
from src.models.classification_models import _text_values
from src.models.optimization_v3_sparse_classification import PARAMS as BASE_PARAMS
from src.models.regression_research_text import HYBRID_FEATURES
from src.models.research_next_budget import (
    DEFAULT_CAPS,
    ResearchNextBudgetError,
    authorize_new_phase,
    finish_fit,
    ledger_summary,
    read_events,
    release_unused,
    reserve_card,
    start_fit,
)
from src.models.research_next_contract import (
    canonical_sha256,
    file_sha256,
    read_bundle,
    resume_or_publish_bundle,
)
from src.models.research_next_n01 import (
    CARD_ID,
    FEATURE_NAMES,
    MINIMUM_CLASS_BIN_N,
    PARAMS,
    THRESHOLD,
    ClassByVolumeSparseClassifier,
    WeightSupportError,
    class_by_volume_weight_plan,
    route_expert_probabilities,
)
from src.models.round1_models import _number_array
from src.utils.paths import PROJECT_ROOT, task_files


ROOT = PROJECT_ROOT
BATCH = "optimization_v3_research_next_v1"
QUEUE_PATH = ROOT / "experiments/optimization_v3/research_next_v1/queue.json"
READINESS_PATH = ROOT / "reports/optimization_v3/research_next_v1/queue_readiness.json"
N00_FREEZE_PATH = ROOT / "experiments/optimization_v3/research_next_v1/frozen/N00/amendment_05/freeze.json"
N00_SUMMARY_PATH = ROOT / "reports/optimization_v3/research_next_v1/results/N00/amendment_05/summary.json"
FREEZE_PATH = ROOT / "experiments/optimization_v3/research_next_v1/frozen/N01/freeze.json"
REPORT_ROOT = ROOT / "reports/optimization_v3/research_next_v1/results/N01"
BUNDLE_ROOT = ROOT / "experiments/runs/optimization_v3_research_next_v1/N01"
LEDGER_PATH = ROOT / "experiments/runs/optimization_v3_research_next_v1/budget.jsonl"
OLD_LEDGER_PATH = ROOT / "experiments/runs/optimization_v3_batch1/batch_ledger.jsonl"
ORIGINS = (55, 67)
JOB_CAP_BYTES = 6 * 1024**3
FREE_RESERVE_BYTES = 2 * 1024**3
MINIMUM_FREE_BYTES = 4 * 1024**3
THREADS = 2
PILOT_FITS = 4


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _write_once(path: Path, value: Any) -> None:
    data = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if canonical_sha256(previous) != canonical_sha256(value):
            raise RuntimeError(f"Refusing to replace immutable N01 output: {path}")
        return
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(data)
        stream.flush()
        import os
        os.fsync(stream.fileno())


def _write_csv_once(path: Path, rows: list[dict[str, Any]]) -> None:
    data = pd.DataFrame(rows).to_csv(index=False, lineterminator="\n", float_format="%.12g").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"Refusing to replace immutable N01 CSV: {path}")
        return
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        import os
        os.fsync(stream.fileno())


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _is_ancestor(commit: str) -> bool:
    return subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"],
                          cwd=ROOT, check=False).returncode == 0


def _dependency_versions() -> dict[str, str]:
    names = ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "psutil", "matplotlib")
    return {"python": platform.python_version(), "platform": platform.platform(),
            **{name: importlib.metadata.version(name) for name in names}}


def _source_paths() -> list[Path]:
    names = (
        "scripts/optimization_v3_research_next_preflight.py",
        "scripts/optimization_v3_research_next_n00.py",
        "scripts/optimization_v3_research_next_n01.py",
        "src/models/research_next_contract.py",
        "src/models/research_next_budget.py",
        "src/models/research_next_n01.py",
        "src/models/classification_models.py",
        "src/models/optimization_v3_sparse_classification.py",
        "src/models/optimization_v3_batch1_c16.py",
        "src/models/regression_research_text.py",
        "src/models/regression_round2_models.py",
        "src/models/round1_models.py",
        "scripts/optimization_v3_batch1_c16.py",
        "src/utils/paths.py",
        "tests/test_research_next_n01.py",
        "tests/test_research_next_budget.py",
    )
    return [ROOT / name for name in names]


def _input_paths(train_path: Path) -> list[Path]:
    paths = [QUEUE_PATH, READINESS_PATH, N00_FREEZE_PATH, N00_SUMMARY_PATH, OLD_LEDGER_PATH, train_path]
    for cutoff in ORIGINS:
        artifact = cache_path("C16", cutoff)
        paths += [artifact, artifact.with_suffix(".json")]
    paths += [ROOT / "reports/optimization_v3/final_continuation/C16/verification.json"]
    paths += [ROOT / "reports/submissions/optimization_v3_expanded_v1/EXECUTION_FREEZE.json"]
    return paths


def _read_n00_snapshot() -> tuple[dict[str, Any], dict[str, Any]]:
    freeze = json.loads(N00_FREEZE_PATH.read_text(encoding="utf-8"))
    identity = freeze.get("identity")
    if not isinstance(identity, dict) or freeze.get("freeze_signature") != canonical_sha256(identity):
        raise RuntimeError("N00 frozen identity signature is invalid")
    if identity.get("batch_id") != BATCH or identity.get("card_id") != "N00":
        raise RuntimeError("N00 freeze identifies the wrong campaign/card")
    if not _is_ancestor(str(identity.get("source_commit", ""))):
        raise RuntimeError("N00 source commit is not an ancestor of current HEAD")
    for rel, expected in identity.get("source_and_artifact_sha256", {}).items():
        path = ROOT / Path(rel)
        if not path.is_file() or file_sha256(path) != expected:
            raise RuntimeError(f"N00 frozen input changed: {rel}")
    summary = json.loads(N00_SUMMARY_PATH.read_text(encoding="utf-8"))
    if summary.get("freeze_signature") != freeze["freeze_signature"]:
        raise RuntimeError("N00 summary is not tied to the frozen amendment")
    if summary.get("status") != "CACHE_AUDIT_AND_N03_FIXED_ARM_RECONSTRUCTION_PASS":
        raise RuntimeError("N00 cache/control audit did not pass")
    return freeze, summary


def _train_path_and_frame() -> tuple[Path, pd.DataFrame, np.ndarray, np.ndarray]:
    path = Path(task_files("classification")["train"])
    columns = ["record_id", "cycle_code", "signal_stream", "entry_friction", "text_load", "artifact_signal"]
    for name in FEATURE_NAMES:
        if name not in columns:
            columns.append(name)
    frame = pd.read_csv(path, usecols=list(dict.fromkeys(columns)))
    if len(frame) != 50_608 or frame["record_id"].isna().any() or not frame["record_id"].astype(str).is_unique:
        raise RuntimeError("Official Classification Train row count/IDs differ from the frozen data boundary")
    cycle = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    if not np.array_equal(np.unique(cycle), np.arange(47, 80)):
        raise RuntimeError("Official Classification Train must contain cycles 47..79")
    raw_y = pd.to_numeric(frame["artifact_signal"], errors="raise").to_numpy()
    if not np.isin(raw_y, (0, 1)).all():
        raise RuntimeError("Official Classification target is not binary")
    return path, frame, raw_y.astype(np.int8), cycle


def _resource_estimate(fit_features: pd.DataFrame, *, available_bytes: int | None = None) -> dict[str, Any]:
    docs, _missing = _text_values(fit_features["signal_stream"])
    tokens = np.fromiter((len(re.findall(r"(?u)\b\w\w+\b", doc.lower())) for doc in docs),
                         dtype=np.int64, count=len(docs))
    nnz_upper = int(np.sum(tokens + np.maximum(tokens - 1, 0), dtype=np.int64))
    sparse_bytes = int(nnz_upper * 16 + (len(docs) + 1) * 8)
    peak = int(1.5 * 1024**3 + 3 * sparse_bytes)
    available = int(psutil.virtual_memory().available if available_bytes is None else available_bytes)
    allowed = min(JOB_CAP_BYTES, available - FREE_RESERVE_BYTES)
    free_ok = available >= MINIMUM_FREE_BYTES
    return {
        "status": "PASS" if free_ok and allowed > 0 and peak <= allowed else "BLOCKED_RESOURCE",
        "fit_rows": int(len(fit_features)), "estimated_nnz_upper_bound": nnz_upper,
        "sparse_design_bytes_upper_bound": sparse_bytes, "estimated_peak_bytes": peak,
        "estimated_peak_is_observed": False, "available_physical_bytes": available,
        "minimum_free_bytes": MINIMUM_FREE_BYTES, "free_memory_reserve_bytes": FREE_RESERVE_BYTES,
        "job_memory_cap_bytes": JOB_CAP_BYTES, "allowed_bytes": max(0, int(allowed)),
        "threads": THREADS, "concurrent_training_jobs": 1,
    }


def _stable_bin_summary(plan: dict[str, Any]) -> dict[str, Any]:
    diagnostics = dict(plan["diagnostics"])
    diagnostics.pop("candidate_weight_sha256", None)
    diagnostics.pop("class_only_weight_sha256", None)
    return diagnostics


def prepare_context() -> dict[str, Any]:
    queue = json.loads(QUEUE_PATH.read_text(encoding="utf-8"))
    if queue.get("batch_id") != BATCH or queue.get("order") != [f"N{i:02}" for i in range(7)]:
        raise RuntimeError("Research Next canonical queue changed")
    card = next((item for item in queue.get("cards", []) if item.get("id") == CARD_ID), None)
    if card is None or card.get("pilot_fits") != PILOT_FITS or card.get("profile") != "STANDARD":
        raise RuntimeError("N01 card or frozen fit allocation differs from queue")
    n00_freeze, n00_summary = _read_n00_snapshot()
    train_path, frame, target, cycle = _train_path_and_frame()
    if file_sha256(train_path) != queue["raw_train_sha256"]["classification"]:
        raise RuntimeError("Official Classification Train hash differs from queue")
    plans: dict[int, dict[str, Any]] = {}
    fit_rows: dict[int, np.ndarray] = {}
    score_rows: dict[int, np.ndarray] = {}
    controls: dict[int, dict[str, np.ndarray]] = {}
    resources: dict[int, dict[str, Any]] = {}
    for cutoff in ORIGINS:
        fit = np.flatnonzero(cycle <= cutoff).astype(np.int64)
        if not len(fit) or int(cycle[fit].max()) != cutoff:
            raise RuntimeError(f"N01/{cutoff} fit rows do not end at the declared cutoff")
        fit_frame = frame.iloc[fit].loc[:, list(FEATURE_NAMES)]
        plan = class_by_volume_weight_plan(fit_frame, target[fit], minimum_cell_n=MINIMUM_CLASS_BIN_N)
        bundle = _load_artifact("C16", cutoff, frame)
        reconstructed = reconstruct_classification(bundle, _text_values(frame["signal_stream"])[1])
        arrays = bundle["arrays"]
        if not np.array_equal(bundle["rows"], reconstructed["rows"]):
            raise RuntimeError(f"N01/{cutoff} C16 controls are misaligned")
        if not np.array_equal(arrays["control__stream_missing"].astype(bool), reconstructed["stream_missing"]):
            raise RuntimeError(f"N01/{cutoff} C16 missing-stream mask changed")
        score_cycles = bundle["cycles"].astype(np.int64)
        expected_lo, expected_hi = cutoff + 1, min(cutoff + 12, 79)
        if not np.array_equal(np.unique(score_cycles), np.arange(expected_lo, expected_hi + 1)):
            raise RuntimeError(f"N01/{cutoff} C16 pilot score horizon is incomplete")
        plans[cutoff] = plan
        fit_rows[cutoff] = fit
        score_rows[cutoff] = bundle["rows"].astype(np.int64)
        controls[cutoff] = {
            "rows": bundle["rows"].astype(np.int64),
            "cycles": bundle["cycles"].astype(np.int64),
            "record_ids": frame["record_id"].astype(str).to_numpy()[bundle["rows"]],
            "c2b_probability": arrays["control__c2b_probability"].astype(np.float64),
            "cb_probability": arrays["control__cb_probability"].astype(np.float64),
            "sparse_baseline_raw": arrays["control__baseline_sparse_raw"].astype(np.float64),
            "current_prediction": reconstructed["control"].astype(np.float64),
            "stream_missing": reconstructed["stream_missing"].astype(bool),
            "artifact_hashes": {"npz": bundle["npz_sha256"], "sidecar": bundle["sidecar_sha256"]},
        }
        resources[cutoff] = _resource_estimate(fit_frame)
    source_paths = _source_paths()
    input_paths = _input_paths(train_path)
    missing_inputs = [str(path) for path in source_paths + input_paths if not path.is_file()]
    if missing_inputs:
        raise FileNotFoundError(f"N01 frozen inputs are missing: {missing_inputs}")
    hashes = { _relative(path): file_sha256(path) for path in sorted(set(source_paths + input_paths), key=_relative) }
    old_snapshot = n00_freeze["execution_context"]["inherited_old_budget_snapshot"]
    old_bytes = OLD_LEDGER_PATH.read_bytes()
    if hashlib.sha256(old_bytes).hexdigest() != old_snapshot["sha256"]:
        raise RuntimeError("Historical V3 ledger changed since N00; keep old and new budgets separate")
    views = queue["views"]["STANDARD"]
    gates = queue["gates"]["classification"]
    recipe = {
        "model_source": "src/models/optimization_v3_sparse_classification.py::SparseTextBoostClassifier and PARAMS",
        "new_adapter": "src/models/research_next_n01.py::ClassByVolumeSparseClassifier",
        "feature_names": list(FEATURE_NAMES),
        "base_parameters": dict(PARAMS),
        "sample_weight_fit_argument": "LGBMClassifier.fit(design, y, sample_weight=weights)",
        "bins": {"feature": "text_load", "parsed_by": "round1_models._number_array",
                 "quantiles": [0.25, 0.50, 0.75], "collapse_duplicate_edges": True,
                 "side": "right", "missing_bin_only_if_observed_in_fit": True},
        "minimum_class_bin_n": MINIMUM_CLASS_BIN_N,
        "candidate_weights": "clip(sqrt(n_fit/(2*G*n_class_bin)),0.5,2.0), then normalize overall mean to 1",
        "matched_control_weights": "per-class mean(candidate_weight); exact class-total mass match",
        "fit_plan_by_cutoff": {str(cutoff): _stable_bin_summary(plans[cutoff]) for cutoff in ORIGINS},
        "threshold": THRESHOLD,
        "policy": ".75 frozen C2_B + .25 weighted sparse expert on present; frozen CB on missing",
        "current_control": "v3_cls_sparse25 from verified C16 cache bridge",
        "score_rows_per_origin": {str(cutoff): int(len(score_rows[cutoff])) for cutoff in ORIGINS},
        "fit_rows_per_origin": {str(cutoff): int(len(fit_rows[cutoff])) for cutoff in ORIGINS},
        "fit_row_sha256": {str(cutoff): hashlib.sha256(fit_rows[cutoff].astype("<i8").tobytes()).hexdigest()
                            for cutoff in ORIGINS},
        "score_row_sha256": {str(cutoff): hashlib.sha256(score_rows[cutoff].astype("<i8").tobytes()).hexdigest()
                             for cutoff in ORIGINS},
        "pilot_views": {key: views[key] for key in ("primary", "h12", "pilot_origins")},
        "pilot_gates": {key: gates[key] for key in (
            "primary_gain_min", "h12_harm_max", "individual_primary_window_harm_max",
            "subgroup_f1_harm_max", "positive_recall_harm_max", "minimum_group_n",
            "minimum_each_class_n", "matched_control_primary_gain_min", "guard_fields")},
        "fit_count": PILOT_FITS,
        "resource_rule": {"threads": THREADS, "concurrent_jobs": 1,
                          "job_cap_bytes": JOB_CAP_BYTES, "reserve_bytes": FREE_RESERVE_BYTES,
                          "minimum_free_bytes": MINIMUM_FREE_BYTES,
                          "peak_estimate": "1.5 GiB + 3 * sparse_CSR_upper_bound"},
        "controls_hashes": {str(cutoff): controls[cutoff]["artifact_hashes"] for cutoff in ORIGINS},
        "n00_freeze_signature": n00_freeze["freeze_signature"],
        "n00_summary_sha256": file_sha256(N00_SUMMARY_PATH),
    }
    identity = {
        "batch_id": BATCH,
        "card_id": CARD_ID,
        "stage": "PILOT",
        "source_commit": _git_head(),
        "queue_sha256": file_sha256(QUEUE_PATH),
        "queue_readiness_sha256": file_sha256(READINESS_PATH),
        "official_train_sha256": file_sha256(train_path),
        "input_and_source_sha256": hashes,
        "dependency_versions": _dependency_versions(),
        "recipe": recipe,
        "test_data_used": False,
        "sample_submission_used": False,
        "kaggle_access": False,
    }
    execution_context = {
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "free_ram_bytes_at_preflight": int(psutil.virtual_memory().available),
        "resource_estimates_by_cutoff": resources,
        "inherited_old_budget_snapshot": old_snapshot,
        "official_estimator_fits": 0,
        "synthetic_estimator_fits": 0,
        "test_files_opened": 0,
        "kaggle_calls": 0,
    }
    return {
        "queue": queue, "card": card, "frame": frame, "target": target, "cycles": cycle,
        "train_path": train_path, "plans": plans, "fit_rows": fit_rows,
        "score_rows": score_rows, "controls": controls, "resource_estimates": resources,
        "identity": identity, "freeze_signature": canonical_sha256(identity),
        "execution_context": execution_context,
    }


def _freeze_payload(context: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "identity": context["identity"],
        "freeze_signature": context["freeze_signature"],
        "execution_context": context["execution_context"],
    }


def _verify_freeze() -> tuple[dict[str, Any], dict[str, Any]]:
    frozen = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if frozen.get("freeze_signature") != canonical_sha256(frozen.get("identity")):
        raise RuntimeError("N01 freeze signature is invalid")
    context = prepare_context()
    expected = frozen["identity"]
    if not _is_ancestor(str(expected.get("source_commit", ""))):
        raise RuntimeError("N01 source commit is not an ancestor of current HEAD")
    actual = dict(context["identity"])
    actual.pop("source_commit", None)
    expected_without_commit = dict(expected)
    expected_without_commit.pop("source_commit", None)
    if actual != expected_without_commit:
        raise RuntimeError("N01 source/input/recipe identity changed after freeze")
    context["freeze_signature"] = frozen["freeze_signature"]
    context["execution_context"] = frozen["execution_context"]
    return frozen, context


def _independent_reconstruct(arrays: Mapping[str, np.ndarray], *, fit_kind: str) -> np.ndarray:
    raw = np.asarray(arrays["raw_prediction"], dtype=np.float64)
    c2b = np.asarray(arrays["control__c2b_probability"], dtype=np.float64)
    cb = np.asarray(arrays["control__cb_probability"], dtype=np.float64)
    missing = np.asarray(arrays["stream_missing"], dtype=bool)
    if fit_kind == "class_by_volume":
        route = np.where(missing, cb, 0.75 * c2b + 0.25 * raw)
    elif fit_kind == "class_only":
        route = np.where(missing, cb, 0.75 * c2b + 0.25 * raw)
    else:
        raise ValueError(f"Unknown N01 fit kind: {fit_kind}")
    current = np.where(missing, cb, 0.75 * c2b + 0.25 * np.asarray(arrays["control__sparse_baseline_raw"], dtype=np.float64))
    if not np.allclose(arrays["baseline_prediction"], current, rtol=0, atol=1e-12):
        raise RuntimeError("N01 stored current route does not reconstruct from C16 controls")
    if not np.allclose(arrays["prediction"], route, rtol=0, atol=1e-12):
        raise RuntimeError(f"N01 stored {fit_kind} route does not reconstruct")
    return route


def _bundle_identity(freeze_signature: str, cutoff: int, fit_kind: str) -> dict[str, Any]:
    return {"batch_id": BATCH, "card_id": CARD_ID, "stage": "PILOT",
            "freeze_signature": freeze_signature, "cutoff": int(cutoff), "fit_kind": fit_kind}


def _bundle_path(cutoff: int, fit_kind: str) -> Path:
    return BUNDLE_ROOT / "pilot" / f"cutoff_{cutoff}" / fit_kind


def _load_existing_fit(context: Mapping[str, Any], cutoff: int, fit_kind: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
    path = _bundle_path(cutoff, fit_kind)
    events = read_events(LEDGER_PATH)
    fit_id = f"N01:PILOT:{cutoff}:{fit_kind}"
    starts = [event for event in events if event.get("event") == "FIT_STARTED" and event.get("fit_id") == fit_id]
    terminals = [event for event in events if event.get("fit_id") == fit_id
                 and event.get("event") in {"FIT_SAVED", "FIT_FAILED"}]
    if not path.exists():
        if terminals and terminals[0].get("event") == "FIT_FAILED":
            raise RuntimeError(f"N01 {fit_id} previously failed; automatic refit is prohibited")
        if starts:
            raise RuntimeError(f"N01 {fit_id} is started-unsaved; recovery is required before another fit")
        return None
    if len(starts) != 1 or (terminals and terminals[0].get("event") == "FIT_FAILED"):
        raise RuntimeError(f"N01 existing bundle {fit_id} lacks a valid unique started/saved lineage")
    identity = _bundle_identity(context["freeze_signature"], cutoff, fit_kind)
    bundle = read_bundle(path, identity, task="classification")
    _independent_reconstruct(bundle.arrays, fit_kind=fit_kind)
    metadata = bundle.receipt.get("metadata", {}).get("metadata", {})
    expected_weight_key = "candidate_weight_sha256" if fit_kind == "class_by_volume" else "class_only_weight_sha256"
    if (metadata.get("cutoff") != cutoff or metadata.get("fit_kind") != fit_kind
            or metadata.get("fit_rows") != len(context["fit_rows"][cutoff])
            or metadata.get("score_rows") != len(context["score_rows"][cutoff])
            or metadata.get("test_data_used") is not False
            or metadata.get("sample_submission_used") is not False
            or metadata.get("kaggle_calls") != 0):
        raise RuntimeError(f"N01 saved receipt metadata is inconsistent for {fit_id}")
    fit_state = metadata.get("fit_state", {})
    expected_weight_hash = context["plans"][cutoff]["diagnostics"][expected_weight_key]
    if (fit_state.get("sample_weight_used") is not True
            or fit_state.get("sample_weight_sha256") != expected_weight_hash
            or fit_state.get("fit_rows") != len(context["fit_rows"][cutoff])):
        raise RuntimeError(f"N01 saved fit-state audit is inconsistent for {fit_id}")
    expected_control = context["controls"][cutoff]
    for key, expected in (("score_rows", context["score_rows"][cutoff]),
                          ("score_record_ids", expected_control["record_ids"]),
                          ("cycle_code", expected_control["cycles"])):
        if not np.array_equal(bundle.arrays[key], expected):
            raise RuntimeError(f"N01 saved bundle alignment changed for {fit_id}: {key}")
    if terminals:
        if terminals[0].get("event") != "FIT_SAVED" or terminals[0].get("receipt_sha256") != bundle.receipt["receipt_sha256"]:
            raise RuntimeError(f"N01 saved ledger receipt differs for {fit_id}")
    else:
        finish_fit(LEDGER_PATH, fit_id=fit_id, status="FIT_SAVED",
                   receipt_sha256=bundle.receipt["receipt_sha256"])
    return {key: bundle.arrays[key].copy() for key in bundle.arrays}, bundle.receipt


def _fit_one(context: Mapping[str, Any], cutoff: int, fit_kind: str) -> tuple[dict[str, Any], dict[str, Any]]:
    existing = _load_existing_fit(context, cutoff, fit_kind)
    if existing is not None:
        return existing
    fit_rows = context["fit_rows"][cutoff]
    score_rows = context["score_rows"][cutoff]
    frame = context["frame"]
    target = context["target"]
    plan = context["plans"][cutoff]
    weight_key = "candidate_weights" if fit_kind == "class_by_volume" else "class_only_weights"
    weights = plan[weight_key]
    fit_features = frame.iloc[fit_rows].loc[:, list(FEATURE_NAMES)]
    score_features = frame.iloc[score_rows].loc[:, list(FEATURE_NAMES)]
    resource = _resource_estimate(fit_features)
    if resource["status"] != "PASS":
        raise ResourceError(json.dumps(resource, sort_keys=True))
    fit_id = f"N01:PILOT:{cutoff}:{fit_kind}"
    start_fit(LEDGER_PATH, fit_id=fit_id, card_id=CARD_ID, stage="PILOT",
              cutoff=cutoff, component=fit_kind)
    try:
        model = ClassByVolumeSparseClassifier(cutoff=cutoff, weight_kind=fit_kind)
        model.fit(fit_features, target[fit_rows], weights)
        raw = model.predict_probability(score_features)
        control = context["controls"][cutoff]
        routes = route_expert_probabilities(
            raw_candidate=raw if fit_kind == "class_by_volume" else np.full_like(raw, 0.5),
            raw_class_only=raw if fit_kind == "class_only" else np.full_like(raw, 0.5),
            c2b_probability=control["c2b_probability"],
            cb_probability=control["cb_probability"],
            sparse_baseline_raw=control["sparse_baseline_raw"],
            stream_missing=control["stream_missing"],
        )
        routed = routes["candidate"] if fit_kind == "class_by_volume" else routes["matched_control"]
        if len(raw) != len(score_rows) or not np.isfinite(raw).all():
            raise RuntimeError(f"N01/{cutoff}/{fit_kind} prediction output is invalid")
        payload = {
            "score_rows": score_rows.astype(np.int64),
            "score_record_ids": control["record_ids"].astype(str),
            "cycle_code": control["cycles"].astype(np.int64),
            "origin_cutoff": np.full(len(score_rows), cutoff, dtype=np.int64),
            "prediction": routed.astype(np.float64),
            "baseline_prediction": control["current_prediction"].astype(np.float64),
            "raw_prediction": raw.astype(np.float64),
            "control__c2b_probability": control["c2b_probability"].astype(np.float64),
            "control__cb_probability": control["cb_probability"].astype(np.float64),
            "control__sparse_baseline_raw": control["sparse_baseline_raw"].astype(np.float64),
            "stream_missing": control["stream_missing"].astype(bool),
        }
        identity = _bundle_identity(context["freeze_signature"], cutoff, fit_kind)
        metadata = {
            "candidate": "class_by_volume_sparse" if fit_kind == "class_by_volume" else "class_only_weighted_sparse",
            "task": "classification", "cutoff": cutoff, "fit_kind": fit_kind,
            "fit_rows": int(len(fit_rows)), "score_rows": int(len(score_rows)),
            "fit_predictor_count": 1, "fit_state": model.audit_state(),
            "weight_diagnostics": _stable_bin_summary(plan),
            "resource_preflight": resource,
            "test_data_used": False, "score_labels_passed_to_fit": False,
            "sample_submission_used": False, "kaggle_calls": 0,
        }
        payload["metadata"] = metadata
        bundle = resume_or_publish_bundle(
            _bundle_path(cutoff, fit_kind), payload, identity,
            task="classification",
        )
        arrays = {key: bundle.arrays[key].copy() for key in bundle.arrays}
        _independent_reconstruct(arrays, fit_kind=fit_kind)
        if not np.array_equal(arrays["score_rows"], score_rows):
            raise RuntimeError(f"N01/{cutoff}/{fit_kind} read-back rows changed")
        finish_fit(LEDGER_PATH, fit_id=fit_id, status="FIT_SAVED",
                   receipt_sha256=bundle.receipt["receipt_sha256"])
        return arrays, bundle.receipt
    except Exception as exc:
        events = read_events(LEDGER_PATH)
        terminal = [event for event in events if event.get("fit_id") == fit_id
                    and event.get("event") in {"FIT_SAVED", "FIT_FAILED"}]
        # Once a bundle directory exists, preserve the started event so that a
        # later run can verify and resume its receipt without another fit.
        if not terminal and not _bundle_path(cutoff, fit_kind).exists():
            finish_fit(LEDGER_PATH, fit_id=fit_id, status="FIT_FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


class ResourceError(RuntimeError):
    pass


def _merge_origin(context: Mapping[str, Any], cutoff: int,
                  fit_bundles: Mapping[str, Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    candidate = fit_bundles["class_by_volume"]
    matched = fit_bundles["class_only"]
    common_keys = ("score_rows", "score_record_ids", "cycle_code", "baseline_prediction",
                   "control__c2b_probability", "control__cb_probability",
                   "control__sparse_baseline_raw", "stream_missing")
    for key in common_keys:
        if not np.array_equal(candidate[key], matched[key]):
            raise RuntimeError(f"N01/{cutoff} candidate and matched-control bundles differ in {key}")
    if len(np.unique(candidate["score_rows"])) != len(candidate["score_rows"]):
        raise RuntimeError(f"N01/{cutoff} score rows are duplicated")
    missing = candidate["stream_missing"].astype(bool)
    c2b = candidate["control__c2b_probability"].astype(np.float64)
    cb = candidate["control__cb_probability"].astype(np.float64)
    raw_candidate = candidate["raw_prediction"].astype(np.float64)
    raw_class_only = matched["raw_prediction"].astype(np.float64)
    sparse_raw = candidate["control__sparse_baseline_raw"].astype(np.float64)
    routed_candidate = np.where(missing, cb, 0.75 * c2b + 0.25 * raw_candidate)
    routed_match = np.where(missing, cb, 0.75 * c2b + 0.25 * raw_class_only)
    current = np.where(missing, cb, 0.75 * c2b + 0.25 * sparse_raw)
    if (not np.allclose(routed_candidate, candidate["prediction"], rtol=0, atol=1e-12)
            or not np.allclose(routed_match, matched["prediction"], rtol=0, atol=1e-12)
            or not np.allclose(current, candidate["baseline_prediction"], rtol=0, atol=1e-12)):
        raise RuntimeError(f"N01/{cutoff} independent three-route composition failed")
    return {
        "rows": candidate["score_rows"].astype(np.int64),
        "record_ids": candidate["score_record_ids"].astype(str),
        "cycles": candidate["cycle_code"].astype(np.int64),
        "origin": np.full(len(candidate["score_rows"]), cutoff, dtype=np.int64),
        "candidate": routed_candidate, "matched": routed_match, "current": current,
        "missing": missing,
    }


def _collect(outputs: Mapping[int, Mapping[str, np.ndarray]], blocks: Sequence[Sequence[int]]) -> dict[str, np.ndarray]:
    parts: list[dict[str, np.ndarray]] = []
    for cutoff, lo, hi in blocks:
        record = outputs[int(cutoff)]
        mask = (record["cycles"] >= int(lo)) & (record["cycles"] <= int(hi))
        parts.append({key: np.asarray(value)[mask] for key, value in record.items()})
    if not parts:
        raise ValueError("N01 cannot evaluate an empty view")
    result = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    order = np.argsort(result["rows"], kind="mergesort")
    result = {key: value[order] for key, value in result.items()}
    if len(result["rows"]) != len(np.unique(result["rows"])):
        raise RuntimeError("N01 evaluation view has overlapping source rows")
    return result


def _independent_metric(target: Any, probability: Any) -> dict[str, float | int]:
    y = np.asarray(target, dtype=np.int64)
    p = (np.asarray(probability, dtype=np.float64) >= THRESHOLD).astype(np.int64)
    if y.shape != p.shape or y.ndim != 1 or not np.isin(y, (0, 1)).all():
        raise ValueError("N01 metric requires aligned binary labels and predictions")
    tn = int(np.sum((y == 0) & (p == 0)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    tp = int(np.sum((y == 1) & (p == 1)))
    f1_negative = 2 * tn / (2 * tn + fp + fn) if (2 * tn + fp + fn) else 0.0
    f1_positive = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    recall_positive = tp / (tp + fn) if (tp + fn) else 0.0
    return {"macro_f1": 0.5 * (f1_negative + f1_positive), "positive_recall": recall_positive,
            "n": int(len(y)), "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def _text_quartiles(frame: pd.DataFrame, cutoff: int) -> tuple[np.ndarray, list[float]]:
    parsed, _missing, _invalid = _number_array(frame["text_load"])
    fit_mask = frame["cycle_code"].to_numpy(dtype=np.int64) <= cutoff
    finite = parsed[fit_mask & np.isfinite(parsed)]
    if not len(finite):
        return np.full(len(frame), -1, dtype=np.int64), []
    edges = np.unique(np.quantile(finite, [0.25, 0.50, 0.75]))
    bins = np.full(len(frame), -1, dtype=np.int64)
    valid = np.isfinite(parsed)
    bins[valid] = np.searchsorted(edges, parsed[valid], side="right").astype(np.int64)
    return bins, edges.astype(float).tolist()


def _primary_group_diagnostics(frame: pd.DataFrame, target: np.ndarray,
                               outputs: Mapping[int, Mapping[str, np.ndarray]],
                               primary_blocks: Sequence[Sequence[int]], gate: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cutoff, lo, hi in primary_blocks:
        record = outputs[int(cutoff)]
        mask = (record["cycles"] >= int(lo)) & (record["cycles"] <= int(hi))
        source_rows = record["rows"][mask]
        local = frame.iloc[source_rows]
        y = target[source_rows]
        candidate = record["candidate"][mask]
        baseline = record["current"][mask]
        text_bins, edges = _text_quartiles(frame, int(cutoff))
        labels_by_field = {
            "entry_friction": local["entry_friction"].astype(str).to_numpy(),
            "stream_availability": np.where(record["missing"][mask], "missing", "present"),
            "fit_defined_text_load_quartile": text_bins[source_rows].astype(str),
        }
        for field, labels in labels_by_field.items():
            for label in np.unique(labels):
                group = labels == label
                y_group = y[group]
                n = int(group.sum())
                counts = {"class_0": int(np.sum(y_group == 0)), "class_1": int(np.sum(y_group == 1))}
                supported = n >= int(gate["minimum_group_n"]) and min(counts.values()) >= int(gate["minimum_each_class_n"])
                candidate_metric = _independent_metric(y_group, candidate[group])
                baseline_metric = _independent_metric(y_group, baseline[group])
                f1_harm = baseline_metric["macro_f1"] - candidate_metric["macro_f1"]
                recall_harm = baseline_metric["positive_recall"] - candidate_metric["positive_recall"]
                failed = bool(supported and (
                    f1_harm > float(gate["subgroup_f1_harm_max"])
                    or recall_harm > float(gate["positive_recall_harm_max"])
                ))
                rows.append({
                    "origin": int(cutoff), "window_start": int(lo), "window_end": int(hi),
                    "field": field, "group": str(label), "n": n,
                    "class_counts": counts, "supported": bool(supported),
                    "candidate": candidate_metric, "current_control": baseline_metric,
                    "f1_harm_candidate_vs_control": float(f1_harm),
                    "positive_recall_harm_candidate_vs_control": float(recall_harm),
                    "gate_failure": failed, "text_load_edges_fit_only": edges if field == "fit_defined_text_load_quartile" else None,
                })
    return rows


def evaluate_pilot(
    frame: pd.DataFrame,
    target: np.ndarray,
    outputs: Mapping[int, Mapping[str, np.ndarray]],
    *,
    views: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    definitions = {
        "primary": views["primary"],
        "H12": views["h12"],
        "W1": [views["primary"][0]],
        "W4": [views["primary"][1]],
    }
    report_views: dict[str, Any] = {}
    per_cycle: list[dict[str, Any]] = []
    for view_name, blocks in definitions.items():
        data = _collect(outputs, blocks)
        rows = data["rows"].astype(np.int64)
        y = target[rows]
        candidate = _independent_metric(y, data["candidate"])
        current = _independent_metric(y, data["current"])
        matched = _independent_metric(y, data["matched"])
        report_views[view_name] = {
            "n_unique_rows": int(len(rows)), "candidate": candidate,
            "current_control": current, "matched_class_only_control": matched,
            "gain_candidate_vs_current": float(candidate["macro_f1"] - current["macro_f1"]),
            "gain_candidate_vs_matched": float(candidate["macro_f1"] - matched["macro_f1"]),
            "prediction_disagreement_vs_current": float(np.mean(np.abs(data["candidate"] - data["current"]))),
        }
        for cycle_code in np.unique(data["cycles"]):
            cycle_mask = data["cycles"] == cycle_code
            cy = y[cycle_mask]
            cm = _independent_metric(cy, data["candidate"][cycle_mask])
            bm = _independent_metric(cy, data["current"][cycle_mask])
            mm = _independent_metric(cy, data["matched"][cycle_mask])
            per_cycle.append({
                "view": view_name, "cycle_code": int(cycle_code), "n": int(cycle_mask.sum()),
                "candidate_macro_f1": cm["macro_f1"], "current_macro_f1": bm["macro_f1"],
                "matched_macro_f1": mm["macro_f1"],
                "gain_candidate_vs_current": cm["macro_f1"] - bm["macro_f1"],
                "gain_candidate_vs_matched": cm["macro_f1"] - mm["macro_f1"],
            })
    groups = _primary_group_diagnostics(frame, target, outputs, views["primary"], gate)
    subgroup_failures = [item for item in groups if item["gate_failure"]]
    p_gain = report_views["primary"]["gain_candidate_vs_current"]
    h12_gain = report_views["H12"]["gain_candidate_vs_current"]
    w1_gain = report_views["W1"]["gain_candidate_vs_current"]
    w4_gain = report_views["W4"]["gain_candidate_vs_current"]
    matched_gain = report_views["primary"]["gain_candidate_vs_matched"]
    checks = {
        "primary_gain_at_least_0_0005": bool(p_gain >= float(gate["primary_gain_min"])),
        "H12_harm_at_most_0_001": bool(h12_gain >= -float(gate["h12_harm_max"])),
        "W1_harm_at_most_0_005": bool(w1_gain >= -float(gate["individual_primary_window_harm_max"])),
        "W4_harm_at_most_0_005": bool(w4_gain >= -float(gate["individual_primary_window_harm_max"])),
        "primary_gain_over_class_only_at_least_0_0002": bool(matched_gain >= float(gate["matched_control_primary_gain_min"])),
        "supported_primary_subgroup_guards": not subgroup_failures,
    }
    decision = "ADVANCE" if all(checks.values()) else "RETIRE"
    group_summary = {
        "supported_group_count": int(sum(item["supported"] for item in groups)),
        "unsupported_group_count": int(sum(not item["supported"] for item in groups)),
        "supported_gate_failure_count": int(len(subgroup_failures)),
        "supported_gate_failures": subgroup_failures,
    }
    result = {
        "card_id": CARD_ID, "task": "classification", "profile": "STANDARD",
        "status": f"PILOT_COMPLETE_{decision}", "decision": decision,
        "threshold": THRESHOLD, "views": report_views,
        "primary_group_summary": group_summary,
        "no_test_or_kaggle_used": True,
    }
    gate_report = {
        "card_id": CARD_ID, "decision": decision,
        "status": "PASS" if all(checks.values()) else "FAIL_VALID_EVIDENCE",
        "checks": checks, "thresholds": dict(gate),
        "supported_primary_subgroup_failures": subgroup_failures,
        "selection_rule": "N01 can be considered for continuation only if every pilot gate and matched-control gain pass; N02 pilot remains required before comparison.",
    }
    return result, gate_report, groups, per_cycle


def _origin_diagnostic(context: Mapping[str, Any], cutoff: int,
                       output: Mapping[str, np.ndarray]) -> dict[str, Any]:
    rows = output["rows"]
    y = context["target"][rows]
    candidate = _independent_metric(y, output["candidate"])
    current = _independent_metric(y, output["current"])
    matched = _independent_metric(y, output["matched"])
    return {
        "card_id": CARD_ID, "cutoff": int(cutoff), "score_rows": int(len(rows)),
        "score_cycle_min": int(output["cycles"].min()), "score_cycle_max": int(output["cycles"].max()),
        "candidate": candidate, "current_control": current, "matched_class_only_control": matched,
        "gain_candidate_vs_current": float(candidate["macro_f1"] - current["macro_f1"]),
        "gain_candidate_vs_matched": float(candidate["macro_f1"] - matched["macro_f1"]),
        "used_for_continuation_selection": False,
        "test_data_used": False, "kaggle_access": False,
    }


def _execute_fit_pair(context: Mapping[str, Any], cutoff: int) -> dict[str, np.ndarray]:
    results: dict[str, dict[str, np.ndarray]] = {}
    for fit_kind in ("class_by_volume", "class_only"):
        arrays, receipt = _fit_one(context, cutoff, fit_kind)
        if receipt.get("metadata", {}).get("metadata", {}).get("test_data_used") is True:
            raise RuntimeError(f"N01/{cutoff}/{fit_kind} receipt reports Test use")
        results[fit_kind] = arrays
    return _merge_origin(context, cutoff, results)


def _write_plots(report: Mapping[str, Any]) -> None:
    import matplotlib.pyplot as plt

    destination = REPORT_ROOT / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    names = ["primary", "H12", "W1", "W4"]
    current = [report["views"][name]["gain_candidate_vs_current"] for name in names]
    matched = [report["views"][name]["gain_candidate_vs_matched"] for name in names]
    positions = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(positions - 0.18, current, width=0.36, label="vs sparse25 current control", color="#4c78a8")
    ax.bar(positions + 0.18, matched, width=0.36, label="vs class-only weighted control", color="#f58518")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(positions, names)
    ax.set_ylabel("Macro F1 gain; positive is better")
    ax.set_title("N01 pilot: fit-origin-matched temporal views")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(destination / "pilot_view_gains.png", dpi=160)
    plt.close(fig)


def _fit_receipts(cutoffs: Sequence[int]) -> dict[str, Any]:
    events = read_events(LEDGER_PATH)
    fits: dict[str, Any] = {}
    for cutoff in cutoffs:
        for fit_kind in ("class_by_volume", "class_only"):
            fit_id = f"N01:PILOT:{cutoff}:{fit_kind}"
            event_group = [event for event in events if event.get("fit_id") == fit_id]
            saved = [event for event in event_group if event.get("event") == "FIT_SAVED"]
            failed = [event for event in event_group if event.get("event") == "FIT_FAILED"]
            if len(saved) + len(failed) != 1:
                raise RuntimeError(f"N01 fit accounting is nonterminal for {fit_id}")
            fits[fit_id] = {
                "status": "FIT_SAVED" if saved else "FIT_FAILED",
                "cutoff": int(cutoff), "fit_kind": fit_kind,
                "receipt_sha256": saved[0].get("receipt_sha256") if saved else None,
            }
    return fits


def run_pilot() -> dict[str, Any]:
    frozen, context = _verify_freeze()
    old_snapshot = frozen["execution_context"]["inherited_old_budget_snapshot"]
    caps = context["queue"]["budget"]
    authorize_new_phase(
        LEDGER_PATH, cap=int(caps["new_phase_cap"]), category_caps=DEFAULT_CAPS,
        inherited_old_budget_snapshot=old_snapshot,
    )
    reservation = reserve_card(LEDGER_PATH, card_id=CARD_ID, stage="PILOT",
                               fits=PILOT_FITS, category="pilot")
    outputs: dict[int, dict[str, np.ndarray]] = {}
    completed: list[int] = []
    try:
        for cutoff in ORIGINS:
            # A cutoff pair and its origin-local report complete before the next cutoff starts.
            outputs[cutoff] = _execute_fit_pair(context, cutoff)
            origin_report = _origin_diagnostic(context, cutoff, outputs[cutoff])
            _write_once(REPORT_ROOT / f"origin_{cutoff}.json", origin_report)
            completed.append(cutoff)
        release_unused(LEDGER_PATH, card_id=CARD_ID, stage="PILOT")
    except ResourceError as exc:
        attempt_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        _write_once(REPORT_ROOT / "resource_attempts" / f"{attempt_id}.json", {
            "card_id": CARD_ID, "status": "DEFERRED_RESOURCE", "message": str(exc),
            "completed_cutoffs": completed, "test_data_used": False, "kaggle_access": False,
            "reservation_retained_for_one_queue-defined_revisit": True,
        })
        return {"card_id": CARD_ID, "status": "DEFERRED_RESOURCE", "completed_cutoffs": completed,
                "ledger": ledger_summary(LEDGER_PATH)}
    except Exception:
        # Do not start another fit or cutoff through this path after a technical failure.
        raise

    result, gate_report, groups, per_cycle = evaluate_pilot(
        context["frame"], context["target"], outputs,
        views=context["queue"]["views"]["STANDARD"],
        gate=context["queue"]["gates"]["classification"],
    )
    expected_primary = int(context["queue"]["views"]["STANDARD"]["primary_rows"])
    expected_h12 = int(context["queue"]["views"]["STANDARD"]["h12_rows"])
    if (result["views"]["primary"]["n_unique_rows"] != expected_primary
            or result["views"]["H12"]["n_unique_rows"] != expected_h12):
        raise RuntimeError("N01 view row denominators differ from the frozen queue")
    accounting = {
        "reserved_fit_count": int(reservation["fits"]),
        "fit_receipts": _fit_receipts(ORIGINS),
        "new_phase_budget": ledger_summary(LEDGER_PATH),
        "official_estimator_fits": 4,
        "synthetic_fits_excluded": True,
    }
    verification = {
        "status": "INDEPENDENT_QA_PASS",
        "route_reconstruction": "PASS",
        "metrics": "independent binary confusion-count Macro F1 at threshold 0.50",
        "fit_population_by_origin": {str(cutoff): int(len(context["fit_rows"][cutoff])) for cutoff in ORIGINS},
        "score_population_by_origin": {str(cutoff): int(len(context["score_rows"][cutoff])) for cutoff in ORIGINS},
        "outer_score_labels_passed_to_fit": False,
        "test_data_used": False, "sample_submission_used": False, "kaggle_access": False,
        "bundle_receipts": {fit_id: value["receipt_sha256"] for fit_id, value in accounting["fit_receipts"].items()},
    }
    result.update({
        "freeze_signature": frozen["freeze_signature"],
        "source_sha256": context["identity"]["input_and_source_sha256"],
        "fit_accounting": accounting,
        "gate": gate_report,
        "independent_verification": verification,
        "test_data_used": False, "sample_submission_used": False, "kaggle_access": False,
    })
    summary = {
        "card_id": CARD_ID, "decision": gate_report["decision"], "status": result["status"],
        "primary_rows": expected_primary, "primary_candidate_macro_f1": result["views"]["primary"]["candidate"]["macro_f1"],
        "primary_current_macro_f1": result["views"]["primary"]["current_control"]["macro_f1"],
        "primary_gain_vs_current": result["views"]["primary"]["gain_candidate_vs_current"],
        "primary_gain_vs_class_only": result["views"]["primary"]["gain_candidate_vs_matched"],
        "H12_rows": expected_h12, "H12_gain_vs_current": result["views"]["H12"]["gain_candidate_vs_current"],
        "W1_gain_vs_current": result["views"]["W1"]["gain_candidate_vs_current"],
        "W4_gain_vs_current": result["views"]["W4"]["gain_candidate_vs_current"],
        "supported_subgroup_failures": gate_report["supported_primary_subgroup_failures"],
        "official_estimator_fits": 4, "test_or_kaggle_used": False,
        "next_action": "wait for N02 pilot before the queue's deterministic continuation comparison",
    }
    _write_once(REPORT_ROOT / "results.json", result)
    _write_once(REPORT_ROOT / "verification.json", verification)
    _write_once(REPORT_ROOT / "gate.json", gate_report)
    _write_once(REPORT_ROOT / "summary.json", summary)
    _write_csv_once(REPORT_ROOT / "groups.csv", groups)
    _write_csv_once(REPORT_ROOT / "per_cycle.csv", per_cycle)
    _write_plots(result)
    return {"card_id": CARD_ID, "status": result["status"], "decision": gate_report["decision"],
            "views": result["views"], "gate_checks": gate_report["checks"], "fit_accounting": accounting}


def preflight() -> dict[str, Any]:
    context = prepare_context()
    return {
        "card_id": CARD_ID, "status": "PREFLIGHT_PASS",
        "freeze_signature": context["freeze_signature"],
        "official_train_sha256": file_sha256(context["train_path"]),
        "fit_rows": {str(cutoff): int(len(context["fit_rows"][cutoff])) for cutoff in ORIGINS},
        "score_rows": {str(cutoff): int(len(context["score_rows"][cutoff])) for cutoff in ORIGINS},
        "weight_support": {str(cutoff): _stable_bin_summary(context["plans"][cutoff]) for cutoff in ORIGINS},
        "resource_estimates": context["resource_estimates"],
        "queue_sha256": context["identity"]["queue_sha256"],
        "source_commit": context["identity"]["source_commit"],
        "test_data_used": False, "sample_submission_used": False, "kaggle_access": False,
        "candidate_fit_or_metric_computed": False,
    }


def freeze() -> dict[str, Any]:
    context = prepare_context()
    payload = _freeze_payload(context)
    _write_once(FREEZE_PATH, payload)
    preflight_report = preflight()
    _write_once(REPORT_ROOT / "preflight.json", preflight_report)
    return {"status": "FROZEN_READY", "path": _relative(FREEZE_PATH),
            "freeze_signature": payload["freeze_signature"],
            "source_commit": context["identity"]["source_commit"],
            "official_estimator_fits": 0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "freeze", "run-pilot"))
    args = parser.parse_args()
    if args.action == "preflight":
        result = preflight()
    elif args.action == "freeze":
        result = freeze()
    else:
        result = run_pilot()
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
