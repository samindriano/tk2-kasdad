"""C19 preflight, frozen pilot execution, independent verification and gate."""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import psutil
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.optimization_v3_batch1_c13 import (  # verified sparse25 control bridge
    _control_input_files as _c13_control_input_files,
    _load_controls as _c13_load_controls,
    _official_train as _c13_official_train,
    _source_files as _c13_source_files,
    _validate_baseline_artifacts as _c13_validate_baseline_artifacts,
)
from scripts.optimization_v3_batch1_verify import verify_report
from src.models.classification_models import _text_values
from src.models.optimization_v3_batch1 import (
    BatchIntegrityError,
    evaluation_views_for_card,
    file_sha256,
    json_sha256,
    load_queue,
    make_freeze_identity,
    metric,
    run_pilot,
    temporal_rows,
    write_json_exclusive,
)
from src.models.optimization_v3_batch1_c19 import (
    BLEND_WEIGHT,
    CARD_ID,
    FEATURE_NAMES,
    PILOT_CUTOFFS,
    SVD_COMPONENTS,
    SVD_PARAMETERS,
    TFIDF_PARAMETERS,
    THRESHOLD,
    XGB_PARAMETERS,
    fit_predict,
)
from src.models.regression_research_text import RegressionTextModel


QUEUE_PATH = ROOT / "experiments/optimization_v3/research_backlog/queue.json"
CARD_PATH = ROOT / "reports/optimization_v3/research_backlog/cards/C19.md"
EXPERIMENT_DIR = ROOT / "experiments/optimization_v3/batch1/C19"
RUN_DIR = ROOT / "experiments/runs/optimization_v3_batch1/C19"
REPORT_DIR = ROOT / "reports/optimization_v3/batch1/C19"
RECIPE_PATH = EXPERIMENT_DIR / "recipe.json"
FREEZE_PATH = EXPERIMENT_DIR / "freeze.json"
ENGINE_REPORT_PATH = RUN_DIR / "pilot_engine_report.json"
RESULTS_PATH = REPORT_DIR / "results.json"
VERIFICATION_PATH = REPORT_DIR / "verification.json"
GATE_PATH = REPORT_DIR / "gate.json"
SUMMARY_PATH = REPORT_DIR / "summary.json"
PREFLIGHT_PATH = REPORT_DIR / "preflight.json"
EVENT_PATH = RUN_DIR / "attempts.jsonl"
LEDGER_PATH = RUN_DIR.parent / "batch_ledger.jsonl"
QUEUE_SHA256 = "822dcd20fa28fd0e9ddc841a8c3709b372dacaca305b94b39b11a9203418fec7"
CARD_SHA256 = "5ef23a464cff05e6e503068a6a0061ed7522cf86ce9d8b2f898b447092994694"
TRAIN_SHA256 = "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c"
MANIFEST_SHA256 = "6e7c2cb50fe6d0ee4767a44c40f152b6538b933fab6cf7082659b16d791f7d1e"
EXPECTED_FIT_ROWS = {55: 12_020, 67: 29_705}
EXPECTED_VIEW_ROWS = {"W1": 4_114, "W4": 4_950, "primary": 9_064, "H12": 38_588}
PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "xgboost", "threadpoolctl", "psutil"]
JOB_MEMORY_CAP_BYTES = 6 * 1024**3
FREE_RAM_RESERVE_BYTES = 2 * 1024**3
MINIMUM_FREE_BYTES = 4 * 1024**3
THREADS = 2


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_or_match(path: Path, value: Mapping[str, Any]) -> None:
    if path.is_file():
        if _read_json(path) != dict(value):
            raise BatchIntegrityError(f"Existing C19 artifact differs: {path}")
    else:
        write_json_exclusive(path, dict(value))


def _validate_card(queue: Mapping[str, Any], card: Mapping[str, Any]) -> None:
    if file_sha256(QUEUE_PATH) != QUEUE_SHA256 or file_sha256(CARD_PATH) != CARD_SHA256:
        raise BatchIntegrityError("Canonical C19 queue/card hash changed")
    canonical = next((row for row in queue.get("cards", []) if row.get("id") == CARD_ID), None)
    if canonical is None or dict(canonical) != dict(card):
        raise BatchIntegrityError("C19 card object differs from canonical queue")
    expected = {
        "id": "C19", "task": "classification", "family": "xgb_dense",
        "policy": "C_AUX25", "pilot_predictor_fits": 2,
        "evaluation": "STANDARD", "condition": "No additional dependency",
    }
    for name, value in expected.items():
        if card.get(name) != value:
            raise BatchIntegrityError(f"Canonical C19 field changed: {name}")
    baseline = queue.get("baseline", {}).get("classification", {})
    if (baseline.get("Train_sha256") != TRAIN_SHA256
            or baseline.get("manifest_sha256") != MANIFEST_SHA256
            or baseline.get("manifest") != "reports/optimization_v3/kaggle/v3_cls_sparse25.json"
            or not np.isclose(float(baseline.get("primary", np.nan)), 0.8299435476765076, rtol=0, atol=1e-15)
            or not np.isclose(float(baseline.get("H12", np.nan)), 0.8029655654017788, rtol=0, atol=1e-15)):
        raise BatchIntegrityError("C19 fixed sparse25 baseline differs from frozen queue")
    preset = queue.get("execution_preset", {})
    if (int(preset.get("concurrent_fit_jobs", -1)) != 1
            or int(preset.get("threads_per_job", -1)) != THREADS
            or float(preset.get("max_job_RSS_GiB", -1)) != 6.0
            or float(preset.get("minimum_free_RAM_GiB_before_start", -1)) != 4.0):
        raise BatchIntegrityError("C19 training/resource policy differs from queue")
    gate = queue.get("evaluation", {}).get("classification_gate", {})
    expected_gates = {
        "pilot_primary_gain_min": 0.0005,
        "pilot_H12_harm_max": 0.001,
        "individual_pilot_view_harm_max": 0.005,
        "supported_group_n_min": 200,
        "supported_group_F1_harm_max": 0.01,
        "positive_class_recall_harm_max": 0.02,
        "positive_and_negative_per_group_min": 30,
    }
    if any(float(gate.get(name, np.nan)) != value for name, value in expected_gates.items()):
        raise BatchIntegrityError("C19 classification gates differ from queue")
    if (SVD_COMPONENTS != 32 or XGB_PARAMETERS != {
            "objective": "binary:logistic", "tree_method": "hist", "device": "cpu",
            "n_estimators": 600, "max_depth": 4, "learning_rate": 0.05,
            "min_child_weight": 20, "reg_lambda": 5.0, "colsample_bytree": 0.8,
            "subsample": 1.0, "random_state": 42, "n_jobs": 2, "verbosity": 0,
    }):
        raise BatchIntegrityError("C19 exact XGBoost/SVD32 parameters changed")
    if (int(preset.get("max_successful_new_predictor_fits", 0)) != 180
            or importlib.metadata.version("xgboost") == ""):
        raise BatchIntegrityError("C19 batch budget or XGBoost dependency is invalid")


def _control_input_files(train_path: Path, views: Mapping[str, Any]) -> dict[str, Path]:
    return _c13_control_input_files(train_path, CARD_PATH, views)


def _source_files() -> list[Path]:
    sources = set(_c13_source_files())
    sources.update({
        Path(__file__),
        ROOT / "src/models/optimization_v3_batch1_c19.py",
        ROOT / "tests/test_optimization_v3_batch1_c19.py",
    })
    return sorted(sources, key=str)


def _source_commit_context() -> str:
    paths = [
        "scripts/optimization_v3_batch1_c19.py",
        "src/models/optimization_v3_batch1_c19.py",
        "tests/test_optimization_v3_batch1_c19.py",
    ]
    result = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", *paths],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise BatchIntegrityError("Cannot identify the committed C19 source revision")
    return result.stdout.strip()


def _baseline_reconstruction(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    missing = np.asarray(arrays["stream_missing"], dtype=bool)
    cb = np.asarray(arrays["cb_probability"], dtype=np.float64)
    c2b = np.asarray(arrays["c2b_probability"], dtype=np.float64)
    old_sparse = np.asarray(arrays["baseline_sparse_raw"], dtype=np.float64)
    if len({len(missing), len(cb), len(c2b), len(old_sparse)}) != 1:
        raise BatchIntegrityError("C19 stored route inputs have inconsistent lengths")
    return np.where(missing, cb, 0.75 * c2b + 0.25 * old_sparse)


def reconstruct_c19(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Independent reconstruction from persisted control arrays and raw XGB probabilities."""
    raw = np.asarray(arrays["raw_candidate"], dtype=np.float64)
    missing = np.asarray(arrays["control__stream_missing"], dtype=bool)
    cb = np.asarray(arrays["control__cb_probability"], dtype=np.float64)
    c2b = np.asarray(arrays["control__c2b_probability"], dtype=np.float64)
    old_sparse = np.asarray(arrays["control__baseline_sparse_raw"], dtype=np.float64)
    if len({len(raw), len(missing), len(cb), len(c2b), len(old_sparse)}) != 1:
        raise BatchIntegrityError("C19 candidate/control arrays have inconsistent lengths")
    baseline = np.where(missing, cb, 0.75 * c2b + 0.25 * old_sparse)
    candidate = np.where(missing, cb, 0.75 * baseline + BLEND_WEIGHT * raw)
    return candidate, baseline


def _memory_estimate(fit_features: pd.DataFrame, score_features: pd.DataFrame,
                     available_bytes: int | None = None) -> dict[str, Any]:
    docs, _missing = _text_values(fit_features["signal_stream"])
    vectorizer = TfidfVectorizer(**dict(TFIDF_PARAMETERS))
    text_matrix = vectorizer.fit_transform(docs).tocsr()
    vocabulary_width = int(text_matrix.shape[1])
    nnz = int(text_matrix.nnz)
    if vocabulary_width <= SVD_COMPONENTS:
        raise ValueError(f"SKIPPED_SUPPORT: exact SVD32 needs >32 TF-IDF columns, got {vocabulary_width}")
    (structured, _processor, _scaler, _encoder, _numeric_positions,
     structured_state) = RegressionTextModel._structured_fit(fit_features)
    structured_width = int(structured.shape[1])
    design_width = structured_width + SVD_COMPONENTS + 1
    sparse_bytes = int(text_matrix.data.nbytes + text_matrix.indices.nbytes + text_matrix.indptr.nbytes)
    fit_dense_bytes = int(len(fit_features) * design_width * 8)
    score_dense_bytes = int(len(score_features) * design_width * 8)
    basis_bytes = int(vocabulary_width * (SVD_COMPONENTS + 10) * 8)
    # Bound includes allocator/XGBoost histogram overhead plus concurrent dense
    # train/score matrices, sparse TF-IDF copies, and randomized-SVD workspace.
    estimate = int(2 * 1024**3 + 3 * sparse_bytes + 4 * basis_bytes
                   + 8 * fit_dense_bytes + 3 * score_dense_bytes)
    available = int(psutil.virtual_memory().available if available_bytes is None else available_bytes)
    allowed = min(JOB_MEMORY_CAP_BYTES, available - FREE_RAM_RESERVE_BYTES)
    result = {
        "status": "PASS" if available >= MINIMUM_FREE_BYTES and allowed > 0 and estimate <= allowed else "BLOCKED_RESOURCE",
        "fit_rows": int(len(fit_features)),
        "score_rows": int(len(score_features)),
        "fit_tfidf_vocabulary_width": vocabulary_width,
        "fit_tfidf_nnz": nnz,
        "fit_tfidf_csr_bytes": sparse_bytes,
        "structured_ohe_width": structured_width,
        "dense_design_width": design_width,
        "fit_dense_matrix_bytes": fit_dense_bytes,
        "score_dense_matrix_bytes": score_dense_bytes,
        "svd_workspace_bytes_upper_bound": basis_bytes,
        "estimated_additional_peak_bytes": estimate,
        "estimate_is_observed_rss": False,
        "available_ram_bytes": available,
        "minimum_free_ram_bytes": MINIMUM_FREE_BYTES,
        "reserve_bytes": FREE_RAM_RESERVE_BYTES,
        "job_cap_bytes": JOB_MEMORY_CAP_BYTES,
        "allowed_bytes": int(max(0, allowed)),
        "threads": THREADS,
        "concurrent_training_jobs": 1,
        "structured_fit_audit": structured_state,
    }
    del text_matrix, structured, vectorizer
    gc.collect()
    return result


def _control_metrics(target: np.ndarray, cycles: np.ndarray,
                     views: Mapping[str, Any], controls: Mapping[int, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, segments in views.items():
        rows_parts: list[np.ndarray] = []
        prediction_parts: list[np.ndarray] = []
        for cutoff, low, high in segments:
            saved = controls[int(cutoff)]
            rows = np.asarray(saved["rows"], dtype=np.int64)
            mask = (cycles[rows] >= int(low)) & (cycles[rows] <= int(high))
            rows_parts.append(rows[mask])
            prediction_parts.append(np.asarray(saved["baseline_prediction"])[mask])
        rows = np.concatenate(rows_parts)
        if len(np.unique(rows)) != len(rows) or len(rows) != EXPECTED_VIEW_ROWS[name]:
            raise BatchIntegrityError(f"C19 row-matched baseline denominator differs for {name}")
        output[name] = {
            "rows": int(len(rows)),
            "macro_f1": float(metric("classification", target[rows], np.concatenate(prediction_parts))["macro_f1"]),
        }
    return output


def _recipe_payload(card: Mapping[str, Any], views: Mapping[str, Any],
                    control_metrics: Mapping[str, Any], queue: Mapping[str, Any]) -> dict[str, Any]:
    gate = dict(queue["evaluation"]["classification_gate"])
    return {
        "card_id": CARD_ID,
        "queue_recipe": {key: card[key] for key in (
            "id", "task", "name", "family", "priority", "change", "policy",
            "pilot_predictor_fits", "evaluation", "condition",
        )},
        "model": "XGBClassifier(binary:logistic) on scaled structured + categorical OHE + exact 32-component TF-IDF/SVD design",
        "xgboost_parameters": dict(XGB_PARAMETERS),
        "xgboost_version": importlib.metadata.version("xgboost"),
        "text": {
            "vectorizer": "sklearn.feature_extraction.text.TfidfVectorizer",
            "parameters": {key: list(value) if isinstance(value, tuple)
                           else value.__name__ if isinstance(value, type)
                           else value.item() if isinstance(value, np.generic) else value
                           for key, value in TFIDF_PARAMETERS.items()},
            "missing_stream": "empty document for fit-only text transform plus explicit missing indicator",
            "svd": {"n_components": SVD_COMPONENTS, **dict(SVD_PARAMETERS)},
        },
        "structured": "RegressionTextModel expanded structured recipe: fold-fit imputation, scaled numeric values, categorical one-hot with min_frequency=30 and max_categories=254",
        "feature_allowlist": list(FEATURE_NAMES),
        "fit_population": "all official Classification Train rows with cycle_code 47..cutoff, including rows with missing streams; no sample weights/downsampling",
        "scoring_population": "STANDARD cutoff 55/67 pilot views with row positions and record_id alignment",
        "prediction_policy": {
            "current_present": "0.75 * C2_B probability + 0.25 * frozen sparse binary expert probability",
            "candidate_present": "0.75 * current sparse25 probability + 0.25 * C19 XGBoost probability",
            "missing_stream": "frozen CatBoost missing-stream probability, unchanged",
            "threshold": THRESHOLD,
            "baseline": "current frozen sparse25 route, including same CatBoost missing-stream path",
        },
        "views": views,
        "row_matched_baseline_metrics": control_metrics,
        "expected_rows": {"fit_by_cutoff": {str(k): v for k, v in EXPECTED_FIT_ROWS.items()},
                          "views": EXPECTED_VIEW_ROWS},
        "gate": gate,
        "fit_protocol": "one XGBClassifier predictor per origin; no early stopping, threshold search, or validation labels passed to fit",
        "resource_policy": {"threads": THREADS, "concurrent_fit_jobs": 1,
                            "job_memory_cap_bytes": JOB_MEMORY_CAP_BYTES,
                            "free_ram_reserve_bytes": FREE_RAM_RESERVE_BYTES,
                            "minimum_free_ram_bytes": MINIMUM_FREE_BYTES},
    }


def _budget_preflight() -> dict[str, Any]:
    used = 0
    if LEDGER_PATH.is_file():
        for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("event") == "BUDGET_RESERVED":
                used += int(event.get("predictor_fits", 0))
    remaining = 180 - used
    return {"status": "PASS" if remaining >= 2 else "SKIP_BUDGET",
            "successful_fit_cap": 180, "already_reserved": used,
            "remaining_before_C19": remaining, "C19_reserved_fits": 2,
            "ledger_path": str(LEDGER_PATH.relative_to(ROOT))}


def _prepare(card: Mapping[str, Any], queue: Mapping[str, Any]) -> dict[str, Any]:
    _validate_card(queue, card)
    evidence = _c13_validate_baseline_artifacts(queue)
    train_path, frame, target, cycles = _c13_official_train()
    if file_sha256(train_path) != TRAIN_SHA256:
        raise BatchIntegrityError("Official Classification Train SHA256 differs from C19 pin")
    if len(frame) != 50_608 or frame["record_id"].isna().any() or frame["record_id"].astype(str).duplicated().any():
        raise BatchIntegrityError("Official Classification Train row/record_id integrity check failed")
    if (not np.isin(target, (0, 1)).all() or not np.array_equal(np.unique(target), np.array([0, 1]))
            or cycles.min() != 47 or cycles.max() != 79):
        raise BatchIntegrityError("Official Classification target or cycle support differs")
    views = evaluation_views_for_card(queue, CARD_ID)
    if views != {
        "W1": [(55, 56, 59)], "W4": [(67, 68, 71)],
        "primary": [(55, 56, 59), (67, 68, 71)],
        "H12": [(55, 56, 67), (67, 68, 79)],
    }:
        raise BatchIntegrityError("C19 evaluation views differ from frozen STANDARD card")
    controls_loader = _c13_load_controls(frame, target)
    score_rows_by_cutoff: dict[int, np.ndarray] = {}
    control_by_cutoff: dict[int, Any] = {}
    memory: dict[str, Any] = {}
    fit_class_counts: dict[str, dict[str, int]] = {}
    support_status = "PASS"
    for cutoff in PILOT_CUTOFFS:
        fit_rows, score_rows = temporal_rows(frame, cutoff, horizon=12)
        if len(fit_rows) != EXPECTED_FIT_ROWS[cutoff]:
            raise BatchIntegrityError(f"C19 fit denominator differs at cutoff {cutoff}")
        feature_fit = frame.iloc[fit_rows].loc[:, list(FEATURE_NAMES)].copy()
        feature_score = frame.iloc[score_rows].loc[:, list(FEATURE_NAMES)].copy()
        y_fit = target[fit_rows]
        fit_class_counts[str(cutoff)] = {str(int(k)): int(v) for k, v in zip(*np.unique(y_fit, return_counts=True))}
        if set(np.unique(y_fit)) != {0, 1}:
            support_status = "SKIPPED_SUPPORT"
        try:
            memory[str(cutoff)] = _memory_estimate(feature_fit, feature_score)
        except ValueError as exc:
            if "SKIPPED_SUPPORT" in str(exc):
                support_status = "SKIPPED_SUPPORT"
                memory[str(cutoff)] = {"status": "SKIPPED_SUPPORT", "reason": str(exc)}
            else:
                raise
        control = controls_loader(cutoff, score_rows)
        control_by_cutoff[cutoff] = {
            "rows": score_rows,
            "baseline_prediction": np.asarray(control["baseline_prediction"], dtype=np.float64),
        }
        score_rows_by_cutoff[cutoff] = score_rows
    control_metrics = _control_metrics(target, cycles, views, control_by_cutoff)
    if {name: row["rows"] for name, row in control_metrics.items()} != EXPECTED_VIEW_ROWS:
        raise BatchIntegrityError("C19 view denominators differ from frozen card")
    recipe = _recipe_payload(card, views, control_metrics, queue)
    input_files = _control_input_files(train_path, views)
    source_files = _source_files()
    identity = make_freeze_identity(
        card=card, feature_names=FEATURE_NAMES, input_files=input_files,
        source_files=source_files, evaluation_views=views, packages=PACKAGES,
    )
    identity.update({
        "recipe": recipe,
        "recipe_artifact_sha256": json_sha256(recipe),
        "source_commit_context": _source_commit_context(),
        "accepted_baseline_evidence": evidence,
        "expected_fit_rows": {str(k): v for k, v in EXPECTED_FIT_ROWS.items()},
        "expected_view_rows": EXPECTED_VIEW_ROWS,
        "fold_support_preflight": {
            "fit_class_counts": fit_class_counts,
            "exact_tfidf_and_ohe_widths": memory,
            "svd_components_required": SVD_COMPONENTS,
            "dimension_reduction_fallback": "prohibited; skip if fewer than 33 fit-only TF-IDF columns",
        },
    })
    budget = _budget_preflight()
    memory_pass = all(item.get("status") == "PASS" for item in memory.values())
    status = ("SKIPPED_SUPPORT" if support_status == "SKIPPED_SUPPORT" else
              "PREFLIGHT_PASS" if memory_pass and budget["status"] == "PASS" else
              "BLOCKED_RESOURCE_OR_BUDGET")
    source_hashes = {str(Path(path).relative_to(ROOT)): file_sha256(path) for path in source_files}
    input_hashes = {name: file_sha256(path) for name, path in sorted(input_files.items())}
    public = {
        "card_id": CARD_ID,
        "status": status,
        "scope": "official Classification Train and frozen sparse25 controls; no candidate score, Test/Sample access or Kaggle access",
        "official_train_sha256": TRAIN_SHA256,
        "input_sha256": input_hashes,
        "source_sha256": source_hashes,
        "freeze_identity_sha256": json_sha256(identity),
        "baseline_evidence": evidence,
        "row_matched_control_metrics": control_metrics,
        "fit_rows": {str(k): v for k, v in EXPECTED_FIT_ROWS.items()},
        "score_rows": EXPECTED_VIEW_ROWS,
        "fit_class_counts": fit_class_counts,
        "memory_preflight": memory,
        "budget_preflight": budget,
        "xgboost_version": importlib.metadata.version("xgboost"),
        "python_version": platform.python_version(),
        "test_or_sample_access": False,
        "kaggle_access": False,
        "candidate_fit_or_metric_computed": False,
        "artifacts_written": False,
    }
    return {
        "public": public, "identity": identity, "recipe": recipe,
        "frame": frame, "target": target, "cycles": cycles, "views": views,
        "control_loader": controls_loader, "score_rows_by_cutoff": score_rows_by_cutoff,
    }


def _resource_guarded_adapter(fit_features: pd.DataFrame, fit_target: np.ndarray,
                              score_features: pd.DataFrame, *, controls: dict[str, Any],
                              card_id: str, cutoff: int) -> dict[str, Any]:
    estimate = _memory_estimate(fit_features, score_features)
    if estimate["status"] != "PASS":
        raise BatchIntegrityError(f"C19 fit-time RAM gate failed at cutoff {cutoff}: {estimate}")
    result = fit_predict(fit_features, fit_target, score_features,
                         controls=controls, card_id=card_id, cutoff=cutoff)
    result["fit_metadata"]["resource_preflight_at_fit"] = {
        key: estimate[key] for key in (
            "estimated_additional_peak_bytes", "available_ram_bytes", "allowed_bytes",
            "fit_tfidf_vocabulary_width", "fit_tfidf_nnz", "dense_design_width",
            "fit_dense_matrix_bytes", "score_dense_matrix_bytes",
        )
    }
    return result


def _fit_accounting() -> dict[str, Any]:
    events = []
    if EVENT_PATH.is_file():
        events = [json.loads(line) for line in EVENT_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    saved = []
    for cutoff in PILOT_CUTOFFS:
        sidecar = RUN_DIR / f"C19_fit{cutoff}.json"
        if sidecar.is_file():
            saved.append(_read_json(sidecar)["metadata"])
    return {
        "expected_predictor_fits": 2,
        "successful_saved_predictor_fits": sum(int(item["fit_predictor_count"]) for item in saved),
        "fit_seconds_total": float(sum(float(item["fit_seconds"]) for item in saved)),
        "fit_origins_saved": [int(item["cutoff"]) for item in saved],
        "fit_started_events": sum(item.get("event") == "FIT_STARTED" for item in events),
        "fit_saved_events": sum(item.get("event") == "FIT_SAVED" for item in events),
        "fit_failed_events": sum(item.get("event") == "FIT_FAILED" for item in events),
        "resumed_verified_events": sum(item.get("event") == "RESUMED_VERIFIED" for item in events),
        "technical_retries": 0,
        "control_refits": 0,
        "rss_peak_sampled": False,
        "rss_note": "Peak RSS was not sampled; report the conservative preflight estimate and process availability only.",
    }


def _gate(context: Mapping[str, Any], report: Mapping[str, Any],
          verification: Mapping[str, Any]) -> dict[str, Any]:
    views = report["views"]
    primary_gain = float(views["primary"]["gain_positive_is_better"])
    h12_gain = float(views["H12"]["gain_positive_is_better"])
    w1_gain = float(views["W1"]["gain_positive_is_better"])
    w4_gain = float(views["W4"]["gain_positive_is_better"])
    group_failures = verification["views"]["primary"]["subgroup_diagnostics"]["supported_guard_failures"]
    checks = {
        "independent_verification_pass": verification.get("status") == "PASS",
        "primary_gain_min_0_0005": primary_gain >= 0.0005,
        "H12_harm_max_0_001": h12_gain >= -0.001,
        "W1_harm_max_0_005": w1_gain >= -0.005,
        "W4_harm_max_0_005": w4_gain >= -0.005,
        "supported_subgroup_macro_f1_and_positive_recall_guards": len(group_failures) == 0,
    }
    return {
        "card_id": CARD_ID,
        "status": "ADVANCE" if all(checks.values()) else "RETIRE",
        "checks": checks,
        "thresholds": {
            "primary_gain_min": 0.0005, "H12_harm_max": 0.001,
            "individual_pilot_view_harm_max": 0.005,
            "supported_group_n_min": 200,
            "supported_group_macro_f1_harm_max": 0.01,
            "positive_class_recall_harm_max": 0.02,
            "per_class_group_min": 30,
        },
        "metrics": {name: {
            "rows": int(record["rows"]),
            "baseline_macro_f1": float(record["baseline"]["macro_f1"]),
            "candidate_macro_f1": float(record["candidate"]["macro_f1"]),
            "gain": float(record["gain_positive_is_better"]),
        } for name, record in views.items()},
        "supported_subgroup_guard_failures": group_failures,
        "interpretation": "ADVANCE only if every pilot aggregate, window, and supported-group gate passes; public score is not used.",
    }


def run_card(card: Mapping[str, Any], queue: Mapping[str, Any], *,
             preflight_only: bool = False) -> dict[str, Any]:
    context = _prepare(card, queue)
    if preflight_only:
        return context["public"]
    status = context["public"]["status"]
    if status != "PREFLIGHT_PASS":
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        _write_or_match(PREFLIGHT_PATH, context["public"])
        skip = {
            "card_id": CARD_ID,
            "status": status,
            "verdict": status,
            "preflight": context["public"],
            "candidate_fit_or_metric_computed": False,
            "test_or_sample_access": False,
            "kaggle_access": False,
        }
        _write_or_match(RESULTS_PATH, skip)
        return skip
    recipe = context["recipe"]
    _write_or_match(RECIPE_PATH, recipe)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    expected_signature = json_sha256(context["identity"])
    if ENGINE_REPORT_PATH.is_file():
        if not FREEZE_PATH.is_file():
            raise BatchIntegrityError("C19 engine report exists without its immutable freeze")
        frozen = _read_json(FREEZE_PATH)
        report = _read_json(ENGINE_REPORT_PATH)
        if (frozen.get("identity") != context["identity"]
                or frozen.get("signature") != expected_signature
                or report.get("freeze_signature") != expected_signature):
            raise BatchIntegrityError("C19 saved engine report/freeze no longer matches source or inputs")
    else:
        report = run_pilot(
            frame=context["frame"], target=context["target"], card=card,
            feature_names=FEATURE_NAMES, adapter=_resource_guarded_adapter,
            control_loader=context["control_loader"], freeze_identity=context["identity"],
            freeze_path=FREEZE_PATH, artifact_dir=RUN_DIR, event_path=EVENT_PATH,
            evaluation_views=context["views"], threads=THREADS,
            report_path=ENGINE_REPORT_PATH, budget_ledger_path=LEDGER_PATH,
            max_successful_fits=180,
        )
    verification = verify_report(
        report=report, frame=context["frame"], target=context["target"],
        artifact_dir=RUN_DIR, freeze_signature=report["freeze_signature"],
        evaluation_views=context["views"], prediction_reconstructors={CARD_ID: reconstruct_c19},
    )
    gate = _gate(context, report, verification)
    identity = context["identity"]
    results = {
        "card_id": CARD_ID,
        "status": "PILOT_VERIFIED",
        "verdict": gate["status"],
        "source_commit_context": identity["source_commit_context"],
        "freeze_signature": report["freeze_signature"],
        "input_sha256": identity["input_sha256"],
        "source_sha256": identity["source_sha256"],
        "recipe_sha256": json_sha256(recipe),
        "fit_accounting": _fit_accounting(),
        "views": report["views"],
        "test_or_sample_access": False,
        "kaggle_access": False,
        "candidate_test_prediction_made": False,
        "threshold_search_or_outer_early_stopping": False,
        "frozen_controls_changed": False,
    }
    _write_or_match(PREFLIGHT_PATH, context["public"])
    _write_or_match(VERIFICATION_PATH, verification)
    _write_or_match(GATE_PATH, gate)
    _write_or_match(RESULTS_PATH, results)
    summary = {
        "card_id": CARD_ID,
        "status": gate["status"],
        "verdict": gate["status"],
        "primary_gain": gate["metrics"]["primary"]["gain"],
        "H12_gain": gate["metrics"]["H12"]["gain"],
        "W1_gain": gate["metrics"]["W1"]["gain"],
        "W4_gain": gate["metrics"]["W4"]["gain"],
        "successful_predictor_fits": results["fit_accounting"]["successful_saved_predictor_fits"],
        "independent_verification": verification["status"],
        "test_or_sample_access": False,
        "kaggle_access": False,
    }
    _write_or_match(SUMMARY_PATH, summary)
    return {"results": results, "verification": verification, "gate": gate, "summary": summary}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    queue = load_queue(QUEUE_PATH)
    card = next(row for row in queue["cards"] if row.get("id") == CARD_ID)
    print(json.dumps(run_card(card, queue, preflight_only=args.preflight_only), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
