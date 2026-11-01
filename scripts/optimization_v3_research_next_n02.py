"""Research Next V1 N02: fixed degree-two interactions on C2_B."""
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
from scipy import sparse
from sklearn.kernel_approximation import PolynomialCountSketch
from sklearn.preprocessing import normalize

from scripts import optimization_v3_research_next_n01 as n01
from src.models.classification_models import ClassificationModel, _text_values
from src.models.research_next_budget import (
    DEFAULT_CAPS,
    authorize_new_phase,
    finish_fit,
    ledger_summary,
    read_events,
    release_unused,
    reserve_card,
    start_fit,
)
from src.models.research_next_contract import canonical_sha256, file_sha256, read_bundle, resume_or_publish_bundle
from src.models.research_next_n02 import (
    CARD_ID,
    SKETCH_PARAMS,
    THRESHOLD,
    C2BInteractionClassifier,
    route_expert_probabilities,
)
from src.utils.paths import PROJECT_ROOT


ROOT = PROJECT_ROOT
BATCH = "optimization_v3_research_next_v1"
QUEUE_PATH = ROOT / "experiments/optimization_v3/research_next_v1/queue.json"
READINESS_PATH = ROOT / "reports/optimization_v3/research_next_v1/queue_readiness.json"
N00_FREEZE_PATH = ROOT / "experiments/optimization_v3/research_next_v1/frozen/N00/amendment_05/freeze.json"
N00_SUMMARY_PATH = ROOT / "reports/optimization_v3/research_next_v1/results/N00/amendment_05/summary.json"
N01_FREEZE_PATH = ROOT / "experiments/optimization_v3/research_next_v1/frozen/N01/freeze.json"
N01_PREFLIGHT_PATH = ROOT / "reports/optimization_v3/research_next_v1/results/N01/preflight.json"
PREVIOUS_FREEZE_PATH = ROOT / "experiments/optimization_v3/research_next_v1/frozen/N02/freeze.json"
FREEZE_PATH = ROOT / "experiments/optimization_v3/research_next_v1/frozen/N02/amendment_02/freeze.json"
REPORT_ROOT = ROOT / "reports/optimization_v3/research_next_v1/results/N02/amendment_02"
BUNDLE_ROOT = ROOT / "experiments/runs/optimization_v3_research_next_v1/N02"
LEDGER_PATH = ROOT / "experiments/runs/optimization_v3_research_next_v1/budget.jsonl"
OLD_LEDGER_PATH = ROOT / "experiments/runs/optimization_v3_batch1/batch_ledger.jsonl"
ORIGINS = (55, 67)
PILOT_FITS = 4
THREADS = 2
JOB_CAP_BYTES = 6 * 1024**3
FREE_RESERVE_BYTES = 2 * 1024**3
MINIMUM_FREE_BYTES = 4 * 1024**3
FIXED_WORKING_BYTES = int(1.5 * 1024**3)


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _is_ancestor(commit: str) -> bool:
    return subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"],
                          cwd=ROOT, check=False).returncode == 0


def _dependencies() -> dict[str, str]:
    packages = ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "psutil", "matplotlib")
    return {"python": platform.python_version(), "platform": platform.platform(),
            **{name: importlib.metadata.version(name) for name in packages}}


def _source_paths() -> list[Path]:
    names = (
        "scripts/optimization_v3_research_next_preflight.py",
        "scripts/optimization_v3_research_next_n00.py",
        "scripts/optimization_v3_research_next_n01.py",
        "scripts/optimization_v3_research_next_n02.py",
        "src/models/research_next_contract.py",
        "src/models/research_next_budget.py",
        "src/models/research_next_n01.py",
        "src/models/research_next_n02.py",
        "src/models/classification_models.py",
        "src/models/round1_models.py",
        "src/models/regression_round2_models.py",
        "src/models/regression_research_text.py",
        "src/models/optimization_v3_sparse_classification.py",
        "src/models/optimization_v3_batch1_c16.py",
        "scripts/optimization_v3_batch1_c16.py",
        "src/utils/paths.py",
        "tests/test_research_next_n01.py",
        "tests/test_research_next_n02.py",
    )
    return [ROOT / item for item in names]


def _input_paths(train_path: Path) -> list[Path]:
    paths = [QUEUE_PATH, READINESS_PATH, N00_FREEZE_PATH, N00_SUMMARY_PATH,
             N01_FREEZE_PATH, N01_PREFLIGHT_PATH, PREVIOUS_FREEZE_PATH, OLD_LEDGER_PATH, train_path,
             ROOT / "reports/optimization_v3/final_continuation/C16/verification.json"]
    for cutoff in ORIGINS:
        artifact = n01.cache_path("C16", cutoff)
        paths.extend((artifact, artifact.with_suffix(".json")))
    return paths


def _sketch_api_smoke() -> dict[str, Any]:
    values = np.asarray(
        [[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    csr = sparse.csr_matrix(values)
    untouched = csr.copy()
    sketch = PolynomialCountSketch(**dict(SKETCH_PARAMS))
    first = np.asarray(sketch.fit_transform(normalize(csr.copy(), norm="l2", axis=1, copy=False)), dtype=np.float64)
    csc_values = np.asarray(sketch.transform(normalize(sparse.csc_matrix(values), norm="l2", axis=1, copy=True)),
                            dtype=np.float64)
    again = PolynomialCountSketch(**dict(SKETCH_PARAMS))
    second = np.asarray(again.fit_transform(normalize(csr.copy(), norm="l2", axis=1, copy=False)), dtype=np.float64)
    if first.shape != (4, int(SKETCH_PARAMS["n_components"])):
        raise RuntimeError("PolynomialCountSketch smoke returned an unexpected shape")
    if not np.array_equal(first, csc_values) or not np.array_equal(first, second):
        raise RuntimeError("PolynomialCountSketch smoke is not deterministic across CSR/CSC")
    if not np.array_equal(csr.data, untouched.data) or not np.isfinite(first).all():
        raise RuntimeError("N02 sketch smoke mutated source data or produced nonfinite values")
    return {"status": "PASS", "shape": list(first.shape), "dtype": str(first.dtype),
            "csr_csc_equal": True, "repeat_fit_equal": True, "input_unchanged": True}


def _sparse_upper_bound(features: pd.DataFrame) -> tuple[int, int]:
    docs, _missing = _text_values(features["signal_stream"])
    tokens = np.fromiter((len(re.findall(r"(?u)\b\w\w+\b", doc.lower())) for doc in docs),
                         dtype=np.int64, count=len(docs))
    # C2_B retains up to two word n-grams plus structured/OHE and one missing flag.
    nnz = int(np.sum(tokens + np.maximum(tokens - 1, 0), dtype=np.int64) + 128 * len(docs))
    return nnz, int(nnz * 16 + (len(docs) + 1) * 8)


def _resource_estimate(fit_features: pd.DataFrame, score_features: pd.DataFrame,
                       *, available_bytes: int | None = None) -> dict[str, Any]:
    fit_nnz, fit_base_bytes = _sparse_upper_bound(fit_features)
    score_nnz, score_base_bytes = _sparse_upper_bound(score_features)
    fit_dense_bytes = int(len(fit_features) * int(SKETCH_PARAMS["n_components"]) * 8)
    score_dense_bytes = int(len(score_features) * int(SKETCH_PARAMS["n_components"]) * 8)
    # Three full fit-sized dense sketch buffers, two score buffers, and extra sparse copies.
    peak = int(FIXED_WORKING_BYTES + 3 * fit_dense_bytes + 2 * score_dense_bytes
               + 3 * fit_base_bytes + 2 * score_base_bytes)
    available = int(psutil.virtual_memory().available if available_bytes is None else available_bytes)
    allowed = min(JOB_CAP_BYTES, available - FREE_RESERVE_BYTES)
    free_ok = available >= MINIMUM_FREE_BYTES
    return {
        "status": "PASS" if free_ok and allowed > 0 and peak <= allowed else "BLOCKED_RESOURCE",
        "fit_rows": int(len(fit_features)), "score_rows": int(len(score_features)),
        "fit_main_effect_nnz_upper_bound": fit_nnz, "score_main_effect_nnz_upper_bound": score_nnz,
        "fit_main_effect_sparse_bytes_upper_bound": fit_base_bytes,
        "score_main_effect_sparse_bytes_upper_bound": score_base_bytes,
        "fit_sketch_dense_bytes": fit_dense_bytes, "score_sketch_dense_bytes": score_dense_bytes,
        "estimated_peak_bytes": peak, "estimated_peak_is_observed": False,
        "estimate_formula": "1.5GiB + 3*fit_dense_sketch + 2*score_dense_sketch + 3*fit_sparse + 2*score_sparse",
        "available_physical_bytes": available, "minimum_free_bytes": MINIMUM_FREE_BYTES,
        "free_memory_reserve_bytes": FREE_RESERVE_BYTES, "job_memory_cap_bytes": JOB_CAP_BYTES,
        "allowed_bytes": max(0, int(allowed)), "threads": THREADS, "concurrent_training_jobs": 1,
    }


def prepare_context() -> dict[str, Any]:
    queue = json.loads(QUEUE_PATH.read_text(encoding="utf-8"))
    if queue.get("batch_id") != BATCH or queue.get("order") != [f"N{i:02}" for i in range(7)]:
        raise RuntimeError("Research Next canonical queue changed")
    card = next((item for item in queue.get("cards", []) if item.get("id") == CARD_ID), None)
    if card is None or card.get("pilot_fits") != PILOT_FITS or card.get("profile") != "STANDARD":
        raise RuntimeError("N02 card or frozen fit allocation differs from queue")
    queued_sketch = card.get("recipe", {}).get("sketch", {})
    if ({key: value for key, value in queued_sketch.items() if key != "class"} != SKETCH_PARAMS
            or queued_sketch.get("class") != "sklearn.kernel_approximation.PolynomialCountSketch"):
        raise RuntimeError("N02 fixed sketch parameters differ from the canonical queue")
    n01_frozen, inherited = n01._verify_freeze()
    train_path = inherited["train_path"]
    if file_sha256(train_path) != queue["raw_train_sha256"]["classification"]:
        raise RuntimeError("Official Classification Train hash differs from the queue")
    feature_names = tuple(ClassificationModel({"id": "C2_B"}).input_feature_names)
    fit_rows: dict[int, np.ndarray] = {}
    score_rows: dict[int, np.ndarray] = {}
    resources: dict[int, dict[str, Any]] = {}
    controls: dict[int, dict[str, Any]] = {}
    for cutoff in ORIGINS:
        fit = np.flatnonzero(inherited["cycles"] <= cutoff).astype(np.int64)
        if not len(fit) or int(inherited["cycles"][fit].max()) != cutoff:
            raise RuntimeError(f"N02/{cutoff} fit rows do not end at cutoff")
        score = inherited["score_rows"][cutoff]
        fit_rows[cutoff] = fit
        score_rows[cutoff] = score.astype(np.int64)
        controls[cutoff] = inherited["controls"][cutoff]
        resources[cutoff] = _resource_estimate(
            inherited["frame"].iloc[fit].loc[:, list(feature_names)],
            inherited["frame"].iloc[score].loc[:, list(feature_names)],
        )
    source_paths = _source_paths()
    input_paths = _input_paths(train_path)
    missing = [str(path) for path in source_paths + input_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"N02 frozen inputs are missing: {missing}")
    hashes = {_relative(path): file_sha256(path) for path in sorted(set(source_paths + input_paths), key=_relative)}
    old_snapshot = inherited["execution_context"]["inherited_old_budget_snapshot"]
    if hashlib.sha256(OLD_LEDGER_PATH.read_bytes()).hexdigest() != old_snapshot["sha256"]:
        raise RuntimeError("Historical V3 ledger changed since N00; keep old and new budgets separate")
    views = queue["views"]["STANDARD"]
    gate = queue["gates"]["classification"]
    recipe = {
        "model_source": "src/models/classification_models.py::ClassificationModel C2_B exact main-effect design",
        "design_builder": "C2BInteractionClassifier calls C2_B fit-only structured/text transform methods; original main matrix retained unchanged",
        "sketch": dict(SKETCH_PARAMS),
        "sketch_input": "L2-normalized copy of exact C2_B main-effect matrix; rowwise norm; original block unchanged",
        "head_parameters": {"C": 1.0, "solver": "liblinear", "max_iter": 2000, "tol": 0.0001,
                            "class_weight": None, "random_state": 42},
        "design": "CSR hstack original main effects + 1024 dense float64 sketch columns safely converted to CSR",
        "control": "same new C2_B logistic head and preprocessing without sketch; separately fitted at both origins",
        "policy": ".75 frozen current sparse25 class route + .25 new expert when stream present; exact frozen CB on missing; threshold .50",
        "fit_rows_by_cutoff": {str(cutoff): int(len(fit_rows[cutoff])) for cutoff in ORIGINS},
        "score_rows_by_cutoff": {str(cutoff): int(len(score_rows[cutoff])) for cutoff in ORIGINS},
        "fit_row_sha256": {str(cutoff): hashlib.sha256(fit_rows[cutoff].astype("<i8").tobytes()).hexdigest()
                            for cutoff in ORIGINS},
        "score_row_sha256": {str(cutoff): hashlib.sha256(score_rows[cutoff].astype("<i8").tobytes()).hexdigest()
                              for cutoff in ORIGINS},
        "views": {key: views[key] for key in ("primary", "h12", "pilot_origins")},
        "gates": {key: gate[key] for key in (
            "primary_gain_min", "h12_harm_max", "individual_primary_window_harm_max",
            "subgroup_f1_harm_max", "positive_recall_harm_max", "minimum_group_n",
            "minimum_each_class_n", "matched_control_primary_gain_min", "guard_fields")},
        "fit_count": PILOT_FITS,
        "resource_rule": {"threads": THREADS, "concurrent_jobs": 1,
                          "job_cap_bytes": JOB_CAP_BYTES, "reserve_bytes": FREE_RESERVE_BYTES,
                          "minimum_free_bytes": MINIMUM_FREE_BYTES, "sketch_components": 1024},
        "control_hashes": {str(cutoff): controls[cutoff]["artifact_hashes"] for cutoff in ORIGINS},
        "freeze_amendment": 2,
        "n00_freeze_signature": json.loads(N00_FREEZE_PATH.read_text(encoding="utf-8"))["freeze_signature"],
        "n00_summary_sha256": file_sha256(N00_SUMMARY_PATH),
        "n01_freeze_signature": n01_frozen["freeze_signature"],
    }
    identity = {
        "batch_id": BATCH, "card_id": CARD_ID, "stage": "PILOT",
        "source_commit": _git_head(),
        "official_train_sha256": file_sha256(train_path),
        "queue_sha256": file_sha256(QUEUE_PATH),
        "queue_readiness_sha256": file_sha256(READINESS_PATH),
        "input_and_source_sha256": hashes,
        "dependency_versions": _dependencies(), "recipe": recipe,
        "test_data_used": False, "sample_submission_used": False, "kaggle_access": False,
    }
    return {
        "queue": queue, "card": card, "frame": inherited["frame"],
        "target": inherited["target"], "cycles": inherited["cycles"],
        "train_path": train_path, "feature_names": feature_names,
        "fit_rows": fit_rows, "score_rows": score_rows, "controls": controls,
        "resource_estimates": resources, "identity": identity,
        "freeze_signature": canonical_sha256(identity),
        "old_budget_snapshot": old_snapshot,
        "execution_context": {
            "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
            "free_ram_bytes_at_preflight": int(psutil.virtual_memory().available),
            "resource_estimates_by_cutoff": resources,
            "sketch_api_smoke": _sketch_api_smoke(),
            "supersedes_pre_fit_freeze": {
                "path": _relative(PREVIOUS_FREEZE_PATH),
                "sha256": file_sha256(PREVIOUS_FREEZE_PATH),
                "reason": "The first N02 pre-fit freeze used a live N01 preparation signature in the N00 field. This amendment records the exact frozen N00 and N01 signatures; no official fit or N02 reservation had started.",
                "recipe_or_gate_changed": False,
            },
            "official_estimator_fits": 0, "synthetic_estimator_fits": 0,
            "test_files_opened": 0, "kaggle_calls": 0,
        },
    }


def _freeze_payload(context: Mapping[str, Any]) -> dict[str, Any]:
    return {"schema_version": 1, "identity": context["identity"],
            "freeze_signature": context["freeze_signature"],
            "execution_context": context["execution_context"]}


def _verify_freeze() -> tuple[dict[str, Any], dict[str, Any]]:
    frozen = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if frozen.get("freeze_signature") != canonical_sha256(frozen.get("identity")):
        raise RuntimeError("N02 freeze signature is invalid")
    context = prepare_context()
    expected = frozen["identity"]
    if not _is_ancestor(str(expected.get("source_commit", ""))):
        raise RuntimeError("N02 frozen source commit is not an ancestor of current HEAD")
    actual = dict(context["identity"])
    actual.pop("source_commit", None)
    expected_stable = dict(expected)
    expected_stable.pop("source_commit", None)
    if actual != expected_stable:
        raise RuntimeError("N02 source/input/recipe identity changed after freeze")
    context["freeze_signature"] = frozen["freeze_signature"]
    context["execution_context"] = frozen["execution_context"]
    return frozen, context


def _bundle_identity(signature: str, cutoff: int, fit_kind: str) -> dict[str, Any]:
    return {"batch_id": BATCH, "card_id": CARD_ID, "stage": "PILOT",
            "freeze_signature": signature, "cutoff": int(cutoff), "fit_kind": fit_kind}


def _bundle_path(cutoff: int, fit_kind: str) -> Path:
    return BUNDLE_ROOT / "pilot" / f"cutoff_{cutoff}" / fit_kind


def _independent_reconstruct(arrays: Mapping[str, np.ndarray], *, fit_kind: str) -> np.ndarray:
    raw = np.asarray(arrays["raw_prediction"], dtype=np.float64)
    current = np.asarray(arrays["control__current_route"], dtype=np.float64)
    cb = np.asarray(arrays["control__cb_probability"], dtype=np.float64)
    missing = np.asarray(arrays["stream_missing"], dtype=bool)
    if fit_kind not in {"sketch", "no_sketch"}:
        raise ValueError(f"Unknown N02 fit kind {fit_kind}")
    expected = np.where(missing, cb, 0.75 * current + 0.25 * raw)
    baseline = np.where(missing, cb, current)
    if not np.allclose(arrays["prediction"], expected, rtol=0, atol=1e-12):
        raise RuntimeError(f"N02 {fit_kind} policy reconstruction failed")
    if not np.allclose(arrays["baseline_prediction"], baseline, rtol=0, atol=1e-12):
        raise RuntimeError("N02 frozen current route reconstruction failed")
    return expected


def _load_existing_fit(context: Mapping[str, Any], cutoff: int,
                       fit_kind: str) -> tuple[dict[str, np.ndarray], dict[str, Any]] | None:
    path = _bundle_path(cutoff, fit_kind)
    fit_id = f"N02:PILOT:{cutoff}:{fit_kind}"
    events = read_events(LEDGER_PATH)
    starts = [event for event in events if event.get("event") == "FIT_STARTED" and event.get("fit_id") == fit_id]
    terminals = [event for event in events if event.get("fit_id") == fit_id
                 and event.get("event") in {"FIT_SAVED", "FIT_FAILED"}]
    if not path.exists():
        if starts:
            raise RuntimeError(f"N02 {fit_id} is started-unsaved; preserve and recover before another fit")
        return None
    if len(starts) != 1 or any(item.get("event") == "FIT_FAILED" for item in terminals):
        raise RuntimeError(f"N02 existing bundle {fit_id} lacks valid unique started lineage")
    bundle = read_bundle(path, _bundle_identity(context["freeze_signature"], cutoff, fit_kind),
                         task="classification")
    _independent_reconstruct(bundle.arrays, fit_kind=fit_kind)
    metadata = bundle.receipt.get("metadata", {}).get("metadata", {})
    if (metadata.get("fit_kind") != fit_kind or metadata.get("cutoff") != cutoff
            or metadata.get("fit_rows") != len(context["fit_rows"][cutoff])
            or metadata.get("score_rows") != len(context["score_rows"][cutoff])
            or metadata.get("test_data_used") is not False
            or metadata.get("sample_submission_used") is not False
            or metadata.get("kaggle_calls") != 0):
        raise RuntimeError(f"N02 bundle receipt metadata failed verification for {fit_id}")
    for key, expected in (("score_rows", context["score_rows"][cutoff]),
                          ("score_record_ids", context["controls"][cutoff]["record_ids"]),
                          ("cycle_code", context["controls"][cutoff]["cycles"])):
        if not np.array_equal(bundle.arrays[key], expected):
            raise RuntimeError(f"N02 bundle alignment changed for {fit_id}: {key}")
    if terminals:
        if (len(terminals) != 1 or terminals[0].get("event") != "FIT_SAVED"
                or terminals[0].get("receipt_sha256") != bundle.receipt["receipt_sha256"]):
            raise RuntimeError(f"N02 saved fit receipt differs for {fit_id}")
    else:
        finish_fit(LEDGER_PATH, fit_id=fit_id, status="FIT_SAVED",
                   receipt_sha256=bundle.receipt["receipt_sha256"])
    return {key: value.copy() for key, value in bundle.arrays.items()}, bundle.receipt


class ResourceError(RuntimeError):
    pass


def _fit_one(context: Mapping[str, Any], cutoff: int,
             fit_kind: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    existing = _load_existing_fit(context, cutoff, fit_kind)
    if existing is not None:
        return existing
    fit_rows = context["fit_rows"][cutoff]
    score_rows = context["score_rows"][cutoff]
    frame = context["frame"]
    target = context["target"]
    fit_features = frame.iloc[fit_rows].loc[:, list(context["feature_names"])]
    score_features = frame.iloc[score_rows].loc[:, list(context["feature_names"])]
    resource = _resource_estimate(fit_features, score_features)
    if resource["status"] != "PASS":
        raise ResourceError(json.dumps(resource, sort_keys=True))
    fit_id = f"N02:PILOT:{cutoff}:{fit_kind}"
    start_fit(LEDGER_PATH, fit_id=fit_id, card_id=CARD_ID, stage="PILOT",
              cutoff=cutoff, component=fit_kind)
    try:
        model = C2BInteractionClassifier(include_sketch=fit_kind == "sketch")
        model.fit(fit_features, target[fit_rows])
        raw = model.predict_probability(score_features)
        control = context["controls"][cutoff]
        routes = route_expert_probabilities(
            raw_candidate=raw if fit_kind == "sketch" else np.full_like(raw, 0.5),
            raw_no_sketch=raw if fit_kind == "no_sketch" else np.full_like(raw, 0.5),
            current_route=control["current_prediction"],
            cb_probability=control["cb_probability"],
            stream_missing=control["stream_missing"],
        )
        routed = routes["candidate"] if fit_kind == "sketch" else routes["no_sketch_control"]
        payload = {
            "score_rows": score_rows.astype(np.int64),
            "score_record_ids": control["record_ids"].astype(str),
            "cycle_code": control["cycles"].astype(np.int64),
            "origin_cutoff": np.full(len(score_rows), cutoff, dtype=np.int64),
            "prediction": routed.astype(np.float64),
            "baseline_prediction": routes["current_route"].astype(np.float64),
            "raw_prediction": raw.astype(np.float64),
            "control__current_route": control["current_prediction"].astype(np.float64),
            "control__cb_probability": control["cb_probability"].astype(np.float64),
            "stream_missing": control["stream_missing"].astype(bool),
            "metadata": {
                "candidate": "c2b_degree_two_polynomial_count_sketch" if fit_kind == "sketch"
                else "c2b_same_learner_no_sketch_control",
                "task": "classification", "cutoff": cutoff, "fit_kind": fit_kind,
                "fit_rows": int(len(fit_rows)), "score_rows": int(len(score_rows)),
                "fit_predictor_count": 1, "fit_state": model.audit_state(),
                "resource_preflight": resource,
                "test_data_used": False, "score_labels_passed_to_fit": False,
                "sample_submission_used": False, "kaggle_calls": 0,
            },
        }
        identity = _bundle_identity(context["freeze_signature"], cutoff, fit_kind)
        bundle = resume_or_publish_bundle(_bundle_path(cutoff, fit_kind), payload, identity,
                                          task="classification")
        arrays = {key: value.copy() for key, value in bundle.arrays.items()}
        _independent_reconstruct(arrays, fit_kind=fit_kind)
        if not np.array_equal(arrays["score_rows"], score_rows):
            raise RuntimeError(f"N02/{cutoff}/{fit_kind} read-back row order changed")
        finish_fit(LEDGER_PATH, fit_id=fit_id, status="FIT_SAVED",
                   receipt_sha256=bundle.receipt["receipt_sha256"])
        return arrays, bundle.receipt
    except Exception as exc:
        events = read_events(LEDGER_PATH)
        terminal = [event for event in events if event.get("fit_id") == fit_id
                    and event.get("event") in {"FIT_SAVED", "FIT_FAILED"}]
        if not terminal and not _bundle_path(cutoff, fit_kind).exists():
            finish_fit(LEDGER_PATH, fit_id=fit_id, status="FIT_FAILED",
                       error=f"{type(exc).__name__}: {exc}")
        raise


def _merge_origin(context: Mapping[str, Any], cutoff: int,
                  bundles: Mapping[str, Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    candidate = bundles["sketch"]
    no_sketch = bundles["no_sketch"]
    for key in ("score_rows", "score_record_ids", "cycle_code", "baseline_prediction",
                "control__current_route", "control__cb_probability", "stream_missing"):
        if not np.array_equal(candidate[key], no_sketch[key]):
            raise RuntimeError(f"N02/{cutoff} candidate/control bundles differ in {key}")
    missing = candidate["stream_missing"].astype(bool)
    current = candidate["control__current_route"].astype(np.float64)
    cb = candidate["control__cb_probability"].astype(np.float64)
    raw_candidate = candidate["raw_prediction"].astype(np.float64)
    raw_control = no_sketch["raw_prediction"].astype(np.float64)
    candidate_route = np.where(missing, cb, 0.75 * current + 0.25 * raw_candidate)
    control_route = np.where(missing, cb, 0.75 * current + 0.25 * raw_control)
    baseline = np.where(missing, cb, current)
    if (not np.allclose(candidate_route, candidate["prediction"], rtol=0, atol=1e-12)
            or not np.allclose(control_route, no_sketch["prediction"], rtol=0, atol=1e-12)
            or not np.allclose(baseline, candidate["baseline_prediction"], rtol=0, atol=1e-12)):
        raise RuntimeError(f"N02/{cutoff} independent three-route reconstruction failed")
    return {"rows": candidate["score_rows"].astype(np.int64),
            "record_ids": candidate["score_record_ids"].astype(str),
            "cycles": candidate["cycle_code"].astype(np.int64),
            "origin": np.full(len(candidate["score_rows"]), cutoff, dtype=np.int64),
            "candidate": candidate_route, "matched": control_route, "current": baseline,
            "missing": missing}


def _origin_report(context: Mapping[str, Any], cutoff: int,
                   output: Mapping[str, np.ndarray]) -> dict[str, Any]:
    rows = output["rows"]
    y = context["target"][rows]
    candidate = n01._independent_metric(y, output["candidate"])
    current = n01._independent_metric(y, output["current"])
    no_sketch = n01._independent_metric(y, output["matched"])
    return {"card_id": CARD_ID, "cutoff": int(cutoff), "score_rows": int(len(rows)),
            "candidate": candidate, "current_route": current, "no_sketch_control": no_sketch,
            "gain_candidate_vs_current": candidate["macro_f1"] - current["macro_f1"],
            "gain_candidate_vs_no_sketch": candidate["macro_f1"] - no_sketch["macro_f1"],
            "used_for_continuation_selection": False,
            "test_data_used": False, "kaggle_access": False}


def _collect(outputs: Mapping[int, Mapping[str, np.ndarray]], blocks: Sequence[Sequence[int]]) -> dict[str, np.ndarray]:
    parts = []
    for cutoff, lo, hi in blocks:
        record = outputs[int(cutoff)]
        mask = (record["cycles"] >= int(lo)) & (record["cycles"] <= int(hi))
        parts.append({key: np.asarray(value)[mask] for key, value in record.items()})
    result = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    if len(np.unique(result["rows"])) != len(result["rows"]):
        raise RuntimeError("N02 evaluation view has overlapping source rows")
    return result


def evaluate_pilot(frame: pd.DataFrame, target: np.ndarray,
                   outputs: Mapping[int, Mapping[str, np.ndarray]], *,
                   views: Mapping[str, Any], gate: Mapping[str, Any]
                   ) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    definitions = {"primary": views["primary"], "H12": views["h12"],
                   "W1": [views["primary"][0]], "W4": [views["primary"][1]]}
    report_views: dict[str, Any] = {}
    per_cycle: list[dict[str, Any]] = []
    for name, blocks in definitions.items():
        data = _collect(outputs, blocks)
        rows = data["rows"].astype(np.int64)
        y = target[rows]
        candidate = n01._independent_metric(y, data["candidate"])
        current = n01._independent_metric(y, data["current"])
        no_sketch = n01._independent_metric(y, data["matched"])
        report_views[name] = {
            "n_unique_rows": int(len(rows)), "candidate": candidate,
            "current_route": current, "no_sketch_control": no_sketch,
            "gain_candidate_vs_current": float(candidate["macro_f1"] - current["macro_f1"]),
            "gain_candidate_vs_no_sketch": float(candidate["macro_f1"] - no_sketch["macro_f1"]),
            "prediction_disagreement_vs_current": float(np.mean(np.abs(data["candidate"] - data["current"]))),
        }
        for cycle in np.unique(data["cycles"]):
            mask = data["cycles"] == cycle
            cy = y[mask]
            cm = n01._independent_metric(cy, data["candidate"][mask])
            bm = n01._independent_metric(cy, data["current"][mask])
            nm = n01._independent_metric(cy, data["matched"][mask])
            per_cycle.append({"view": name, "cycle_code": int(cycle), "n": int(mask.sum()),
                              "candidate_macro_f1": cm["macro_f1"], "current_macro_f1": bm["macro_f1"],
                              "no_sketch_macro_f1": nm["macro_f1"],
                              "gain_candidate_vs_current": cm["macro_f1"] - bm["macro_f1"],
                              "gain_candidate_vs_no_sketch": cm["macro_f1"] - nm["macro_f1"]})
    groups = n01._primary_group_diagnostics(frame, target, outputs, views["primary"], gate)
    subgroup_failures = [item for item in groups if item["gate_failure"]]
    primary_gain = report_views["primary"]["gain_candidate_vs_current"]
    h12_gain = report_views["H12"]["gain_candidate_vs_current"]
    w1_gain = report_views["W1"]["gain_candidate_vs_current"]
    w4_gain = report_views["W4"]["gain_candidate_vs_current"]
    control_gain = report_views["primary"]["gain_candidate_vs_no_sketch"]
    checks = {
        "primary_gain_at_least_0_0005": bool(primary_gain >= float(gate["primary_gain_min"])),
        "H12_harm_at_most_0_001": bool(h12_gain >= -float(gate["h12_harm_max"])),
        "W1_harm_at_most_0_005": bool(w1_gain >= -float(gate["individual_primary_window_harm_max"])),
        "W4_harm_at_most_0_005": bool(w4_gain >= -float(gate["individual_primary_window_harm_max"])),
        "primary_gain_over_no_sketch_at_least_0_0002": bool(
            control_gain >= float(gate["matched_control_primary_gain_min"])),
        "supported_primary_subgroup_guards": not subgroup_failures,
    }
    decision = "ADVANCE" if all(checks.values()) else "RETIRE"
    result = {"card_id": CARD_ID, "task": "classification", "profile": "STANDARD",
              "status": f"PILOT_COMPLETE_{decision}", "decision": decision,
              "threshold": THRESHOLD, "views": report_views,
              "primary_group_summary": {
                  "supported_group_count": int(sum(item["supported"] for item in groups)),
                  "unsupported_group_count": int(sum(not item["supported"] for item in groups)),
                  "supported_gate_failures": subgroup_failures,
              }, "no_test_or_kaggle_used": True}
    gate_result = {"card_id": CARD_ID, "decision": decision,
                   "status": "PASS" if all(checks.values()) else "FAIL_VALID_EVIDENCE",
                   "checks": checks, "thresholds": dict(gate),
                   "supported_primary_subgroup_failures": subgroup_failures,
                   "selection_rule": "N02 continuation eligibility requires every frozen pilot gate and at least 0.0002 primary Macro F1 gain over the separately fitted no-sketch control; compare with N01 only after both pilots are complete."}
    return result, gate_result, groups, per_cycle


def _fit_receipts() -> dict[str, Any]:
    events = read_events(LEDGER_PATH)
    result = {}
    for cutoff in ORIGINS:
        for fit_kind in ("sketch", "no_sketch"):
            fit_id = f"N02:PILOT:{cutoff}:{fit_kind}"
            group = [event for event in events if event.get("fit_id") == fit_id]
            saved = [event for event in group if event.get("event") == "FIT_SAVED"]
            failed = [event for event in group if event.get("event") == "FIT_FAILED"]
            if len(saved) + len(failed) != 1:
                raise RuntimeError(f"N02 fit accounting is nonterminal for {fit_id}")
            result[fit_id] = {"status": "FIT_SAVED" if saved else "FIT_FAILED",
                              "receipt_sha256": saved[0].get("receipt_sha256") if saved else None}
    return result


def _write_plot(report: Mapping[str, Any]) -> None:
    import matplotlib.pyplot as plt

    names = ["primary", "H12", "W1", "W4"]
    current = [report["views"][name]["gain_candidate_vs_current"] for name in names]
    no_sketch = [report["views"][name]["gain_candidate_vs_no_sketch"] for name in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(x - 0.18, current, width=0.36, label="vs frozen sparse25 class route", color="#4c78a8")
    ax.bar(x + 0.18, no_sketch, width=0.36, label="vs same-learner no-sketch", color="#f58518")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, names)
    ax.set_ylabel("Macro F1 gain; positive is better")
    ax.set_title("N02 pilot: fit-origin-matched temporal views")
    ax.legend(frameon=False)
    fig.tight_layout()
    target = REPORT_ROOT / "figures" / "pilot_view_gains.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=160)
    plt.close(fig)


def run_pilot() -> dict[str, Any]:
    frozen, context = _verify_freeze()
    caps = context["queue"]["budget"]
    authorize_new_phase(LEDGER_PATH, cap=int(caps["new_phase_cap"]), category_caps=DEFAULT_CAPS,
                        inherited_old_budget_snapshot=context["old_budget_snapshot"])
    reservation = reserve_card(LEDGER_PATH, card_id=CARD_ID, stage="PILOT",
                               fits=PILOT_FITS, category="pilot")
    outputs: dict[int, dict[str, np.ndarray]] = {}
    completed: list[int] = []
    try:
        for cutoff in ORIGINS:
            bundles = {}
            for fit_kind in ("sketch", "no_sketch"):
                arrays, receipt = _fit_one(context, cutoff, fit_kind)
                if receipt.get("metadata", {}).get("metadata", {}).get("test_data_used") is True:
                    raise RuntimeError(f"N02/{cutoff}/{fit_kind} receipt reports Test use")
                bundles[fit_kind] = arrays
            outputs[cutoff] = _merge_origin(context, cutoff, bundles)
            origin = _origin_report(context, cutoff, outputs[cutoff])
            n01._write_once(REPORT_ROOT / f"origin_{cutoff}.json", origin)
            completed.append(cutoff)
        release_unused(LEDGER_PATH, card_id=CARD_ID, stage="PILOT")
    except ResourceError as exc:
        attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        n01._write_once(REPORT_ROOT / "resource_attempts" / f"{attempt}.json", {
            "card_id": CARD_ID, "status": "DEFERRED_RESOURCE", "message": str(exc),
            "completed_cutoffs": completed, "reservation_retained_for_one_queue_defined_revisit": True,
            "test_data_used": False, "kaggle_access": False,
        })
        return {"card_id": CARD_ID, "status": "DEFERRED_RESOURCE", "completed_cutoffs": completed,
                "ledger": ledger_summary(LEDGER_PATH)}

    views = context["queue"]["views"]["STANDARD"]
    gate = context["queue"]["gates"]["classification"]
    result, gate_result, groups, per_cycle = evaluate_pilot(
        context["frame"], context["target"], outputs, views=views, gate=gate)
    if (result["views"]["primary"]["n_unique_rows"] != int(views["primary_rows"])
            or result["views"]["H12"]["n_unique_rows"] != int(views["h12_rows"])):
        raise RuntimeError("N02 evaluation row counts differ from frozen queue")
    accounting = {"reserved_fit_count": int(reservation["fits"]), "fit_receipts": _fit_receipts(),
                  "new_phase_budget": ledger_summary(LEDGER_PATH), "official_estimator_fits": 4,
                  "synthetic_fits_excluded": True}
    verification = {
        "status": "INDEPENDENT_QA_PASS", "route_reconstruction": "PASS",
        "metrics": "independent binary confusion-count Macro F1 at threshold 0.50",
        "fit_population_by_origin": {str(cutoff): int(len(context["fit_rows"][cutoff])) for cutoff in ORIGINS},
        "score_population_by_origin": {str(cutoff): int(len(context["score_rows"][cutoff])) for cutoff in ORIGINS},
        "score_labels_passed_to_fit": False, "test_data_used": False,
        "sample_submission_used": False, "kaggle_access": False,
        "bundle_receipts": {key: value["receipt_sha256"] for key, value in accounting["fit_receipts"].items()},
    }
    result.update({"freeze_signature": frozen["freeze_signature"],
                   "source_sha256": context["identity"]["input_and_source_sha256"],
                   "fit_accounting": accounting, "gate": gate_result,
                   "independent_verification": verification,
                   "test_data_used": False, "sample_submission_used": False, "kaggle_access": False})
    primary = result["views"]["primary"]
    summary = {"card_id": CARD_ID, "status": result["status"], "decision": result["decision"],
               "primary_rows": primary["n_unique_rows"],
               "primary_candidate_macro_f1": primary["candidate"]["macro_f1"],
               "primary_current_macro_f1": primary["current_route"]["macro_f1"],
               "primary_no_sketch_macro_f1": primary["no_sketch_control"]["macro_f1"],
               "primary_gain_vs_current": primary["gain_candidate_vs_current"],
               "primary_gain_vs_no_sketch": primary["gain_candidate_vs_no_sketch"],
               "H12_rows": result["views"]["H12"]["n_unique_rows"],
               "H12_gain_vs_current": result["views"]["H12"]["gain_candidate_vs_current"],
               "W1_gain_vs_current": result["views"]["W1"]["gain_candidate_vs_current"],
               "W4_gain_vs_current": result["views"]["W4"]["gain_candidate_vs_current"],
               "supported_subgroup_failures": gate_result["supported_primary_subgroup_failures"],
               "official_estimator_fits": 4, "test_or_kaggle_used": False,
               "next_action": "compare N01 and N02 only after every pilot lane completes"}
    n01._write_once(REPORT_ROOT / "results.json", result)
    n01._write_once(REPORT_ROOT / "verification.json", verification)
    n01._write_once(REPORT_ROOT / "gate.json", gate_result)
    n01._write_once(REPORT_ROOT / "summary.json", summary)
    n01._write_csv_once(REPORT_ROOT / "groups.csv", groups)
    n01._write_csv_once(REPORT_ROOT / "per_cycle.csv", per_cycle)
    _write_plot(result)
    return {"card_id": CARD_ID, "status": result["status"], "decision": result["decision"],
            "views": result["views"], "gate_checks": gate_result["checks"], "fit_accounting": accounting}


def preflight() -> dict[str, Any]:
    context = prepare_context()
    api = _sketch_api_smoke()
    overall = "PREFLIGHT_PASS" if all(item["status"] == "PASS" for item in context["resource_estimates"].values()) else "BLOCKED_RESOURCE"
    return {"card_id": CARD_ID, "status": overall,
            "freeze_signature": context["freeze_signature"],
            "official_train_sha256": file_sha256(context["train_path"]),
            "fit_rows": {str(cutoff): int(len(context["fit_rows"][cutoff])) for cutoff in ORIGINS},
            "score_rows": {str(cutoff): int(len(context["score_rows"][cutoff])) for cutoff in ORIGINS},
            "resource_estimates": context["resource_estimates"], "sketch_api_smoke": api,
            "queue_sha256": context["identity"]["queue_sha256"],
            "source_commit": context["identity"]["source_commit"],
            "candidate_fit_or_metric_computed": False,
            "test_data_used": False, "sample_submission_used": False, "kaggle_access": False}


def freeze() -> dict[str, Any]:
    context = prepare_context()
    report = preflight()
    if report["status"] != "PREFLIGHT_PASS":
        raise ResourceError("N02 preflight did not pass; refusing to freeze a runnable pilot")
    payload = _freeze_payload(context)
    n01._write_once(FREEZE_PATH, payload)
    n01._write_once(REPORT_ROOT / "preflight.json", report)
    n01._write_once(REPORT_ROOT / "amendment_note.json",
                    context["execution_context"]["supersedes_pre_fit_freeze"])
    return {"status": "FROZEN_READY", "path": _relative(FREEZE_PATH),
            "freeze_signature": payload["freeze_signature"],
            "source_commit": context["identity"]["source_commit"], "official_estimator_fits": 0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "freeze", "run-pilot"))
    action = parser.parse_args().action
    result = preflight() if action == "preflight" else freeze() if action == "freeze" else run_pilot()
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
