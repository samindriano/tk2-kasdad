"""Prepare the two explicitly authorized C16 and R22_ORIGINAL submissions.

This wrapper reuses only attested full-Train predictions, performs one new
full-Train fit at a time, writes ignored CSV/model-prediction artifacts, and
never contacts Kaggle. Upload is a separate, authenticated, live-gated step.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
from typing import Any, Iterator

import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import prepare_initial_submissions as initial  # noqa: E402
from src.data.audit import sha256 as file_sha256  # noqa: E402
from src.models.optimization_v3_batch1 import (  # noqa: E402
    BatchIntegrityError,
    exclusive_training_slot,
)
from src.models.optimization_v3_batch1_c16 import (  # noqa: E402
    BLEND_WEIGHT as C16_WEIGHT,
    CHANGES as C16_CHANGES,
    FEATURE_NAMES as C16_FEATURES,
    ModerateSparseCapacityClassifier,
    PARAMS as C16_PARAMS,
    THRESHOLD as C16_THRESHOLD,
)
from src.models.optimization_v3_batch1_r22 import (  # noqa: E402
    FEATURE_NAMES as R22_FEATURES,
    XGB_PARAMETERS,
    XGBoostPseudoHuberSVD32Regressor,
    rank50_prediction,
)
from src.models.classification_round2_models import stream_missing_mask  # noqa: E402
from src.models.regression_research_text import _array_digest as regression_array_digest  # noqa: E402
from src.models.round1_models import _fit_fingerprint  # noqa: E402


CONTRACT_PATH = ROOT / "reports/submissions/c16_r22/CONTRACT.json"
RUN_ROOT = ROOT / "experiments/runs/c16_r22_submission"
FREEZE_PATH = RUN_ROOT / "execution_freeze.json"
OUTPUT_ROOT = ROOT / "submissions/c16_r22"
REPORT_ROOT = ROOT / "reports/submissions/c16_r22"
TRAINING_LOCK = ROOT / "experiments/runs/optimization_v3_batch1/.training.lock"
CLASS_MANIFEST = ROOT / "reports/optimization_v3/kaggle/v3_cls_sparse25.json"
CLASS_CACHE = ROOT / "experiments/runs/optimization_v3_inference/v3_cls_sparse25.npz"
REG_MANIFEST = ROOT / "reports/optimization_v2/kaggle/v2_reg_sparse_lgb25.json"
CHAMPION_CHECKPOINT = ROOT / "experiments/optimization_v3/champions_checkpoint1.json"
EXPECTED_CLASS_MANIFEST_SHA = "6e7c2cb50fe6d0ee4767a44c40f152b6538b933fab6cf7082659b16d791f7d1e"
EXPECTED_CLASS_CACHE_SHA = "31ef6945ae74a14b1e4ff46220667ac3a08de3401120257e7932745c927d6103"
EXPECTED_REG_MANIFEST_SHA = "dd166e92089f24978244496a9a0afabc0d06551dbdc326579a7c898a8844e19a"
EXPECTED_CHAMPION_REF = "57835122"
THREADS = 2
JOB_CAP_BYTES = 6 * 1024**3
FREE_RAM_RESERVE_BYTES = 2 * 1024**3
CLASS_PEAK_ESTIMATE_BYTES = 2 * 1024**3


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _array_sha(values: Any, dtype: str = "<f8") -> str:
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes(order="C")).hexdigest()


def _ids_sha(values: Any) -> str:
    ids = [str(value) for value in values]
    return hashlib.sha256(("\n".join(ids) + "\n").encode("utf-8")).hexdigest()


def route_c16_probabilities(c2b: Any, cb_missing: Any, c16: Any,
                            missing_mask: Any) -> np.ndarray:
    base = np.asarray(c2b, dtype=np.float64)
    fallback = np.asarray(cb_missing, dtype=np.float64)
    candidate = np.asarray(c16, dtype=np.float64)
    missing = np.asarray(missing_mask, dtype=bool)
    if (base.ndim != 1 or base.size == 0 or fallback.shape != base.shape
            or candidate.shape != base.shape or missing.shape != base.shape):
        raise ValueError("C16 routing inputs must be aligned nonempty vectors")
    if any(not np.isfinite(x).all() or np.any((x < 0.0) | (x > 1.0))
           for x in (base, fallback, candidate)):
        raise ValueError("C16 routing inputs must be finite probabilities")
    return np.where(missing, fallback, 0.75 * base + C16_WEIGHT * candidate)


def r22_original_prediction(raw: Any, champion: Any, cycles: Any) -> tuple[np.ndarray, np.ndarray]:
    raw_values = np.asarray(raw, dtype=np.float64)
    incumbent = np.asarray(champion, dtype=np.float64)
    if raw_values.ndim != 1 or raw_values.size == 0 or incumbent.shape != raw_values.shape:
        raise ValueError("R22_ORIGINAL inputs must be aligned nonempty vectors")
    if not np.isfinite(incumbent).all() or np.any((incumbent < 0.0) | (incumbent > 100.0)):
        raise ValueError("R22_ORIGINAL champion must be finite and within [0,100]")
    component = rank50_prediction(raw_values, cycles)
    prediction = np.clip(0.75 * incumbent + 0.25 * component, 0.0, 100.0)
    if not np.isfinite(prediction).all():
        raise ValueError("R22_ORIGINAL returned nonfinite predictions")
    return prediction, component


def _git(*args: str, binary: bool = False) -> str | bytes:
    result = subprocess.check_output(["git", "-C", str(ROOT), *args])
    return result if binary else result.decode("utf-8").strip()


def _head() -> str:
    return str(_git("rev-parse", "HEAD"))


def _contract() -> dict[str, Any]:
    return _read_json(CONTRACT_PATH)


def _source_hashes() -> dict[str, str]:
    paths = [
        "scripts/prepare_c16_r22_fulltrain_submissions.py",
        "scripts/prepare_initial_submissions.py",
        "src/models/optimization_v3_batch1.py",
        "src/models/optimization_v3_batch1_c16.py",
        "src/models/optimization_v3_sparse_classification.py",
        "src/models/optimization_v3_final_continuation_c16.py",
        "src/models/optimization_v3_batch1_r22.py",
        "src/models/optimization_v3_final_continuation_r22.py",
        "src/models/classification_round2_models.py",
        "src/models/classification_models.py",
        "src/models/round1_models.py",
        "src/models/regression_research_text.py",
        "src/data/audit.py",
        "src/utils/paths.py",
    ]
    return {path: file_sha256(ROOT / path) for path in paths}


def _verify_historical_source_manifest(manifest: dict[str, Any], label: str) -> dict[str, str]:
    commit = manifest.get("source_commit")
    source_map = manifest.get("source_code_sha256")
    if not isinstance(commit, str) or not isinstance(source_map, dict) or not source_map:
        raise RuntimeError(f"{label} manifest lacks source commit/code hashes")
    verified: dict[str, str] = {}
    for rel, expected in sorted(source_map.items()):
        blob = _git("show", f"{commit}:{rel}", binary=True)
        actual = hashlib.sha256(blob).hexdigest()
        if actual != expected:
            raise RuntimeError(f"{label} historical source hash mismatch: {rel}")
        verified[rel] = actual
    return verified


def _component_audit(manifest: dict[str, Any], key: str) -> dict[str, Any]:
    found = [item for item in manifest.get("component_audits", [])
             if item.get("fit_component", {}).get("model") == key]
    if len(found) != 1:
        raise RuntimeError(f"Expected exactly one full-Train component audit for {key}")
    return found[0]


def _load_inputs(task: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    train, test, sample, hashes = initial._load_inputs(task)
    expected = _contract()["data"][task]
    if hashes != expected:
        raise RuntimeError(f"Official {task} input hashes differ from the frozen contract")
    if len(train) != 50_608 or len(test) != 29_421 or len(sample) != 29_421:
        raise RuntimeError(f"Official {task} row counts differ from the frozen contract")
    if _ids_sha(test.record_id) != _contract()["data"]["ordered_test_ids_sha256"]:
        raise RuntimeError("Official Test ID sequence differs from the frozen contract")
    if not test.record_id.reset_index(drop=True).equals(sample.record_id.reset_index(drop=True)):
        raise RuntimeError("SampleSubmission IDs/order do not equal official Test IDs/order")
    return train, test, sample, hashes


def _check_champion_checkpoint() -> dict[str, Any]:
    checkpoint = _read_json(CHAMPION_CHECKPOINT)
    reg = checkpoint.get("regression", {})
    cfg = _contract()["regression"]["champion"]
    if (str(reg.get("ref")) != cfg["reference"]
            or str(reg.get("manifest")) != cfg["manifest"]
            or float(reg.get("public_MAE")) != float(cfg["public_mae"])):
        raise RuntimeError("Regression incumbent champion identity differs from the freeze")
    cls = checkpoint.get("classification", {})
    expected_cls = {
        "ref": _contract()["classification"]["comparison_champion"]["reference"],
        "manifest": _contract()["classification"]["comparison_champion"]["manifest"],
        "public_macro_f1": _contract()["classification"]["comparison_champion"]["public_macro_f1"],
    }
    if (str(cls.get("ref")) != expected_cls["ref"]
            or str(cls.get("manifest")) != expected_cls["manifest"]
            or float(cls.get("public_macro_f1")) != expected_cls["public_macro_f1"]):
        raise RuntimeError("Classification incumbent identity differs from the frozen checkpoint")
    return {
        "checkpoint_path": str(CHAMPION_CHECKPOINT.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": file_sha256(CHAMPION_CHECKPOINT),
        "classification": expected_cls,
        "regression": {"ref": str(reg["ref"]), "manifest": str(reg["manifest"]),
                       "public_MAE": float(reg["public_MAE"])},
    }


def _verify_continuation_verdicts() -> dict[str, Any]:
    evidence = _contract()["continuation_evidence"]
    observed: dict[str, str] = {}
    for name in ("campaign_summary", "decision", "c16_results", "r22_continuation", "r22_result_json"):
        item = evidence[name]
        path = ROOT / item["path"]
        actual = file_sha256(path)
        if actual != item["sha256"]:
            raise RuntimeError(f"Frozen continuation evidence changed: {item['path']}")
        observed[name] = actual
    c16 = _read_json(ROOT / evidence["c16_results"]["path"])
    failures = c16.get("qualification", {}).get("supported_group_failures", [])
    q4_failure_retained = any(
        item.get("view") == "W2" and item.get("field") == "text_load_quantile"
        and item.get("group") == "Q4" and item.get("guard_pass") is False
        for item in failures
    )
    if (c16.get("freeze_signature") != evidence["c16_results"]["freeze_signature"]
            or c16.get("qualification", {}).get("qualified") is not False
            or c16.get("qualification", {}).get("gates", {}).get("primary_gain_at_least_0_0005") is not False
            or c16.get("qualification", {}).get("gates", {}).get("late_W5_W6_gain_positive") is not False
            or c16.get("qualification", {}).get("gates", {}).get("all_supported_group_guards_pass") is not False
            or not q4_failure_retained):
        raise RuntimeError("C16 historical gate-failed verdict or signature changed")
    r22 = (ROOT / evidence["r22_continuation"]["path"]).read_text(encoding="utf-8")
    if (evidence["r22_continuation"]["freeze_signature"] not in r22
            or "Selected policy: **R22_ORIGINAL**" not in r22
            or "### R22_ORIGINAL: QUALIFIED" not in r22):
        raise RuntimeError("R22_ORIGINAL selection/qualification evidence changed")
    r22_result = _read_json(ROOT / evidence["r22_result_json"]["path"])
    if (r22_result.get("freeze_signature") != evidence["r22_result_json"]["freeze_signature"]
            or r22_result.get("selected_policy") != "R22_ORIGINAL"
            or r22_result.get("qualification", {}).get("R22_ORIGINAL", {}).get("qualified") is not True):
        raise RuntimeError("R22_ORIGINAL structured qualification evidence changed")
    return {
        "sha256": observed,
        "C16": {"qualified": False, "preserved_as": "experimental gate-failed benchmark"},
        "R22": {"selected_policy": "R22_ORIGINAL", "qualified": True,
                "internal_primary": r22_result["policies"]["R22_ORIGINAL"]["views"]["primary"],
                "internal_H12": r22_result["policies"]["R22_ORIGINAL"]["views"]["H12"],
                "internal_W5_W6": r22_result["policies"]["R22_ORIGINAL"]["late_W5_W6"]},
        "C16_metrics": {
            "primary": c16["views"]["primary"],
            "H12": c16["views"]["H12"],
            "late_W5_W6_gain": c16["qualification"]["late_W5_W6"]["gain_positive_is_better"],
            "W2_text_load_Q4_guard": c16["qualification"]["supported_group_failures"],
        },
    }


def _verify_classification_reuse(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    contract = _contract()["classification"]
    if file_sha256(CLASS_MANIFEST) != contract["reuse_manifest_sha256"]:
        raise RuntimeError("Classification full-Train reuse manifest hash changed")
    if file_sha256(CLASS_CACHE) != contract["reuse_cache_sha256"]:
        raise RuntimeError("Classification full-Train prediction cache hash changed")
    manifest = _read_json(CLASS_MANIFEST)
    expected_input = _contract()["data"]["classification"]
    if (manifest.get("status") != "PREPARED_AND_QA_PASS"
            or manifest.get("train_rows") != 50_608
            or manifest.get("test_rows") != 29_421
            or manifest.get("input_sha256") != expected_input
            or manifest.get("ordered_test_ids_sha256") != _contract()["data"]["ordered_test_ids_sha256"]
            or manifest.get("test_labels_used") is not False
            or manifest.get("fit_preprocessing_only_train") is not True):
        raise RuntimeError("Classification reuse manifest fails full-Train/Test-boundary audit")
    historical_sources = _verify_historical_source_manifest(manifest, "C2_B/CB_MISSING")
    expected_models = contract["reused_models"]
    train_target = train.artifact_signal.to_numpy(dtype=np.int8)
    for model_id in ("C2_B", "CB_MISSING"):
        audit = _component_audit(manifest, model_id)
        state = audit.get("state", {})
        if (audit.get("fit_rows") != 50_608
                or audit.get("learned_only_official_Train") is not True
                or audit.get("prediction_sha256") != expected_models[model_id]
                or state.get("id") != model_id
                or state.get("fit_rows") != 50_608
                or not state.get("fit_data_sha256")):
            raise RuntimeError(f"Classification {model_id} fitted-state lineage is invalid")
        features = list(C16_FEATURES) if model_id == "C2_B" else [
            name for name in C16_FEATURES if name != "signal_stream"
        ]
        if state.get("input_feature_names") != features or state.get("fit_data_sha256") != _fit_fingerprint(
                train.loc[:, features], tuple(features), train_target):
            raise RuntimeError(f"Classification {model_id} exact Train feature/target fingerprint changed")
    c2 = _component_audit(manifest, "C2_B")["state"]
    expected_lr = {"C": 1.0, "class_weight": None, "max_iter": 2000,
                   "random_state": 42, "solver": "liblinear", "tol": 0.0001}
    expected_tfidf = {"dtype": "<class 'numpy.float64'>", "lowercase": True,
                      "max_df": 0.995, "max_features": 20_000, "min_df": 3,
                      "ngram_range": [1, 2], "strip_accents": "unicode",
                      "sublinear_tf": True}
    c2_params = c2.get("parameters", {})
    c2_ohe = {"dtype": "<class 'numpy.float64'>", "handle_unknown": "infrequent_if_exist",
              "max_categories": 254, "min_frequency": 30, "sparse_output": True}
    if (c2.get("kind") != "lr_hybrid"
            or c2.get("input_feature_names") != list(C16_FEATURES)
            or c2_params.get("logistic_regression") != expected_lr
            or c2_params.get("minimum_category_support") != 30
            or c2_params.get("one_hot_encoder") != c2_ohe
            or c2.get("text", {}).get("parameters") != expected_tfidf):
        raise RuntimeError("Reused C2_B fitted state differs from its frozen hybrid-LR recipe")
    cb = _component_audit(manifest, "CB_MISSING")["state"]
    cb_preprocessor = cb.get("fit_only_preprocessor")
    cb_counts = (cb_preprocessor or {}).get("category_counts", {}) if isinstance(cb_preprocessor, dict) else {}
    cat_params = cb.get("parameters", {})
    if (cb.get("input_feature_names") != [name for name in C16_FEATURES if name != "signal_stream"]
            or not isinstance(cb_preprocessor, dict)
            or not cb_counts
            or any(item.get("training_rows") != 50_608 for item in cb_counts.values())
            or cb.get("eval_set_used") is not False
            or cb.get("early_stopping_used") is not False
            or cat_params.get("iterations") != 500
            or cat_params.get("depth") != 6
            or not np.isclose(float(cat_params.get("learning_rate", np.nan)), 0.05)
            or cat_params.get("random_seed") != 42
            or cat_params.get("loss_function") != "Logloss"
            or cat_params.get("task_type") != "CPU"
            or cat_params.get("use_best_model") is not False):
        raise RuntimeError("Reused CB_MISSING fitted state differs from the frozen structured CatBoost recipe")
    with np.load(CLASS_CACHE, allow_pickle=False) as cache:
        required = {"base", "missing", "missing_mask", "prediction"}
        if not required.issubset(cache.files):
            raise RuntimeError(f"Classification reuse cache lacks arrays: {sorted(required - set(cache.files))}")
        base = np.asarray(cache["base"], dtype=np.float64)
        missing = np.asarray(cache["missing"], dtype=np.float64)
        missing_mask = np.asarray(cache["missing_mask"], dtype=bool)
        cache_prediction = np.asarray(cache["prediction"], dtype=np.float64)
    if any(a.shape != (len(test),) for a in (base, missing, missing_mask, cache_prediction)):
        raise RuntimeError("Classification reused full-Train vectors are not Test-aligned")
    if _array_sha(base) != expected_models["C2_B"] or _array_sha(missing) != expected_models["CB_MISSING"]:
        raise RuntimeError("Classification reused prediction arrays differ from their fitted-state receipts")
    derived_missing = stream_missing_mask(test)
    if not np.array_equal(missing_mask, derived_missing):
        raise RuntimeError("Classification cache stream-availability mask differs from official Test")
    if not np.isfinite(base).all() or not np.isfinite(missing).all() or not np.isfinite(cache_prediction).all():
        raise RuntimeError("Classification reuse cache contains nonfinite probabilities")
    if any(np.any((x < 0.0) | (x > 1.0)) for x in (base, missing, cache_prediction)):
        raise RuntimeError("Classification reuse cache probabilities fall outside [0,1]")
    evidence = {
        "manifest_sha256": file_sha256(CLASS_MANIFEST),
        "cache_sha256": file_sha256(CLASS_CACHE),
        "source_commit": manifest["source_commit"],
        "source_code_sha256_verified_count": len(historical_sources),
        "reused_models": {model: {"fit_rows": 50_608,
                                   "prediction_sha256": expected_models[model]}
                          for model in expected_models},
        "missing_mask_sha256": hashlib.sha256(missing_mask.astype(np.uint8).tobytes()).hexdigest(),
    }
    return base, missing, missing_mask, evidence


def _verify_regression_reuse(test: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    contract = _contract()["regression"]["champion"]
    if file_sha256(REG_MANIFEST) != contract["manifest_sha256"]:
        raise RuntimeError("Regression champion manifest hash changed")
    manifest = _read_json(REG_MANIFEST)
    expected_input = _contract()["data"]["regression"]
    expected_recipe = {"kind": "regression_component_blend", "components": contract["components"]}
    if (manifest.get("status") != "PREPARED_AND_QA_PASS"
            or manifest.get("train_rows") != 50_608
            or manifest.get("test_rows") != 29_421
            or manifest.get("input_sha256") != expected_input
            or manifest.get("test_labels_used") is not False
            or manifest.get("fit_preprocessing_only_train") is not True
            or manifest.get("recipe") != expected_recipe):
        raise RuntimeError("Regression accepted manifest is not the frozen three-component champion")
    historical_sources = _verify_historical_source_manifest(manifest, "three-component champion")
    components = manifest.get("component_audits", [])
    expected_models = ["lgb", "ridge", "lgbm_sparse_l1"]
    if [item.get("fit_component", {}).get("model") for item in components] != expected_models:
        raise RuntimeError("Regression champion does not contain exactly its three frozen components")
    if any(item.get("fit_rows") != 50_608 for item in components):
        raise RuntimeError("Regression champion has a component without a 50,608-row fit")
    if not all(item.get("learned_only_official_Train") is True for item in components):
        raise RuntimeError("Regression champion component preprocessing provenance is incomplete")
    csv_path = ROOT / contract["csv"]
    if file_sha256(csv_path) != contract["csv_sha256"] or manifest.get("csv_sha256") != contract["csv_sha256"]:
        raise RuntimeError("Regression accepted champion CSV hash changed")
    csv = pd.read_csv(csv_path, float_precision="round_trip")
    if list(csv.columns) != ["record_id", "threat_rank"] or len(csv) != len(test):
        raise RuntimeError("Regression champion CSV schema/row count is invalid")
    if not csv.record_id.reset_index(drop=True).equals(test.record_id.reset_index(drop=True)):
        raise RuntimeError("Regression champion CSV does not match official Test IDs/order")
    values = csv.threat_rank.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 100.0)):
        raise RuntimeError("Regression champion CSV values are invalid")
    if regression_array_digest(values) != manifest.get("prediction_sha256"):
        raise RuntimeError("Regression champion prediction vector differs from its accepted manifest")
    return values, {
        "champion_reference": str(contract["reference"]),
        "public_MAE": float(contract["public_mae"]),
        "manifest_sha256": file_sha256(REG_MANIFEST),
        "csv_sha256": file_sha256(csv_path),
        "prediction_sha256": manifest["prediction_sha256"],
        "source_commit": manifest["source_commit"],
        "source_code_sha256_verified_count": len(historical_sources),
        "components": [{"model": item["fit_component"]["model"],
                        "fit_rows": int(item["fit_rows"]),
                        "prediction_sha256": item["prediction_sha256"]}
                       for item in components],
    }


def _r22_resource_preflight(available_bytes: int) -> dict[str, Any]:
    fit_rows, width = 50_608, 840
    design_bytes = fit_rows * width * 8
    histogram_bytes = width * 256 * 16 * THREADS * 3
    estimated_peak = 5 * design_bytes + histogram_bytes + 2 * 1024**3
    allowed = min(JOB_CAP_BYTES, max(0, available_bytes - FREE_RAM_RESERVE_BYTES))
    return {
        "fit_rows": fit_rows,
        "feature_width_upper_bound": width,
        "estimated_peak_bytes": estimated_peak,
        "estimated_peak_is_observed": False,
        "available_physical_bytes": int(available_bytes),
        "free_ram_reserve_bytes": FREE_RAM_RESERVE_BYTES,
        "job_memory_cap_bytes": JOB_CAP_BYTES,
        "allowed_bytes": allowed,
        "threads": THREADS,
        "concurrent_training_jobs": 1,
        "pass": estimated_peak <= allowed,
    }


def _resource_preflight(task: str, available_bytes: int | None = None) -> dict[str, Any]:
    available = int(psutil.virtual_memory().available) if available_bytes is None else int(available_bytes)
    estimate = CLASS_PEAK_ESTIMATE_BYTES if task == "classification" else _r22_resource_preflight(available)["estimated_peak_bytes"]
    allowed = min(JOB_CAP_BYTES, max(0, available - FREE_RAM_RESERVE_BYTES))
    return {
        "available_physical_bytes": available,
        "estimated_peak_bytes": int(estimate),
        "estimated_peak_is_observed": False,
        "free_ram_reserve_bytes": FREE_RAM_RESERVE_BYTES,
        "job_memory_cap_bytes": JOB_CAP_BYTES,
        "allowed_bytes": allowed,
        "threads": THREADS,
        "concurrent_training_jobs": 1,
        "pass": int(estimate) <= allowed,
        "detail": "R22 estimate includes five dense design copies and histogram allowance" if task == "regression" else "conservative sparse LightGBM full-Train cap",
    }


@contextmanager
def _training_slot(task: str) -> Iterator[dict[str, Any]]:
    """Acquire the shared one-job lock, then recheck live RAM immediately before fit."""
    with exclusive_training_slot(TRAINING_LOCK):
        resources = _resource_preflight(task)
        if not resources["pass"]:
            raise RuntimeError(f"{task} full-Train fit blocked by fresh in-lock RAM check: {resources}")
        yield resources


def _verify_source_freeze(freeze: dict[str, Any]) -> None:
    if _head() != freeze.get("execution_head"):
        raise RuntimeError("Repository HEAD changed after execution configuration was frozen")
    if freeze.get("contract_sha256") != file_sha256(CONTRACT_PATH):
        raise RuntimeError("Committed submission contract changed after freeze")
    if freeze.get("source_sha256") != _source_hashes():
        raise RuntimeError("Training source code changed after execution configuration was frozen")


def _load_freeze() -> dict[str, Any]:
    if not FREEZE_PATH.is_file():
        raise RuntimeError("Execution freeze is absent; run the freeze action after committing preparation code")
    freeze = _read_json(FREEZE_PATH)
    _verify_source_freeze(freeze)
    return freeze


def _verify_all_inputs_and_recipes() -> dict[str, Any]:
    contract = _contract()
    class_train, class_test, class_sample, class_hashes = _load_inputs("classification")
    reg_train, reg_test, reg_sample, reg_hashes = _load_inputs("regression")
    if class_train.artifact_signal.isna().any() or not set(class_train.artifact_signal.unique()).issubset({0, 1}):
        raise RuntimeError("Classification Train target is not complete binary data")
    if reg_train.threat_rank.isna().any() or not np.isfinite(reg_train.threat_rank.to_numpy(dtype=float)).all():
        raise RuntimeError("Regression Train target is incomplete or nonfinite")
    if np.any((reg_train.threat_rank.to_numpy(dtype=float) < 0.0)
              | (reg_train.threat_rank.to_numpy(dtype=float) > 100.0)):
        raise RuntimeError("Regression Train target is outside [0,100]")
    if set(C16_FEATURES) & {"record_id", "artifact_signal", "threat_rank"}:
        raise RuntimeError("C16 feature allowlist contains identity or target leakage")
    if set(R22_FEATURES) & {"record_id", "artifact_signal", "threat_rank"}:
        raise RuntimeError("R22 feature allowlist contains identity or target leakage")
    if C16_WEIGHT != 0.25 or C16_THRESHOLD != 0.50:
        raise RuntimeError("C16 frozen blend weight or threshold changed")
    if C16_CHANGES != contract["classification"]["tree_changes"]:
        raise RuntimeError("C16 frozen estimator changes differ from the contract")
    xgb_expected = contract["regression"]["xgboost"]
    for name, expected in xgb_expected.items():
        observed = XGB_PARAMETERS.get(name)
        if observed != expected:
            raise RuntimeError(f"R22 frozen XGBoost parameter changed: {name}")
    if len(class_sample) != len(class_test) or len(reg_sample) != len(reg_test):
        raise RuntimeError("SampleSubmission row count does not match official Test")
    cls_base, cls_missing, cls_mask, class_reuse = _verify_classification_reuse(class_train, class_test)
    reg_champion, reg_reuse = _verify_regression_reuse(reg_test)
    champ_ids = _check_champion_checkpoint()
    if champ_ids["classification"]["ref"] != str(contract["classification"]["comparison_champion"]["reference"]):
        raise RuntimeError("Classification comparison champion checkpoint changed")
    return {
        "head": _head(),
        "official_inputs": {
            "classification": {"sha256": class_hashes, "train_rows": len(class_train), "test_rows": len(class_test), "sample_rows": len(class_sample)},
            "regression": {"sha256": reg_hashes, "train_rows": len(reg_train), "test_rows": len(reg_test), "sample_rows": len(reg_sample)},
        },
        "champion_checkpoint": champ_ids,
        "continuation_verdicts": _verify_continuation_verdicts(),
        "classification_reuse": class_reuse,
        "regression_champion_reuse": reg_reuse,
        "recipe_hashes": {
            "classification_params": _canonical_hash(C16_PARAMS),
            "regression_params": _canonical_hash(XGB_PARAMETERS),
        },
        "test_labels_used": False,
        "model_selection_on_test": False,
    }


def preflight() -> dict[str, Any]:
    freeze = _load_freeze()
    context = _verify_all_inputs_and_recipes()
    lock_path = Path(TRAINING_LOCK)
    try:
        with exclusive_training_slot(lock_path):
            lock = {"available": True, "path": str(lock_path.relative_to(ROOT)).replace("\\", "/"), "file_deleted": False}
    except BatchIntegrityError:
        lock = {"available": False, "path": str(lock_path.relative_to(ROOT)).replace("\\", "/"), "file_deleted": False}
    memory = {task: _resource_preflight(task) for task in ("classification", "regression")}
    status = {
        "status": "READY_FOR_SERIAL_FIT" if lock["available"] and all(x["pass"] for x in memory.values()) else "BLOCKED",
        "freeze_sha256": file_sha256(FREEZE_PATH),
        "execution_head": freeze["execution_head"],
        "verification": context,
        "global_training_lock": lock,
        "resource_preflight": memory,
        "fit_jobs_authorized": ["classification C16 full-Train", "regression R22_ORIGINAL full-Train"],
        "fit_jobs_at_once": 1,
        "kaggle_contacted": False,
    }
    return status


def freeze_execution() -> dict[str, Any]:
    contract = _contract()
    base = contract["authorized_base_head"]
    head = _head()
    if head == base:
        raise RuntimeError("Preparation code must be committed before freezing fits")
    parents = str(_git("rev-list", "--parents", "-n", "1", "HEAD")).split()[1:]
    if base not in parents:
        raise RuntimeError("Current preparation commit is not a direct child of the authorized starting HEAD")
    status = str(_git("status", "--porcelain", "--untracked-files=normal"))
    if status.strip():
        raise RuntimeError("Working tree must be clean before execution freeze")
    evidence = _verify_all_inputs_and_recipes()
    output = {
        "status": "FROZEN_BEFORE_FIT",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_head": head,
        "authorized_base_head": base,
        "contract_sha256": file_sha256(CONTRACT_PATH),
        "source_sha256": _source_hashes(),
        "pre_fit_input_and_recipe_audit": evidence,
        "runtime": {
            "python": platform.python_version(),
            "packages": {name: importlib.metadata.version(name) for name in [
                "numpy", "pandas", "scipy", "scikit-learn", "lightgbm",
                "xgboost", "catboost", "threadpoolctl", "psutil"
            ]},
        },
        "fitted_output_created": False,
        "test_inference_created": False,
        "kaggle_contacted": False,
    }
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    if FREEZE_PATH.exists():
        previous = _read_json(FREEZE_PATH)
        if previous != output:
            raise RuntimeError("A different execution freeze already exists; refusing to replace it")
        return previous
    FREEZE_PATH.write_text(json.dumps(output, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return output


class FullTrainC16Classifier(ModerateSparseCapacityClassifier):
    """Permit cutoff 79 while inheriting the frozen C16 model implementation."""

    def __init__(self) -> None:
        self.cutoff = 79
        self._fitted = False


class FullTrainR22Regressor(XGBoostPseudoHuberSVD32Regressor):
    """Permit cutoff 79 while inheriting the frozen R22 model implementation."""

    def __init__(self) -> None:
        super().__init__(cutoff=55)
        self.cutoff = 79


def _write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    temp = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with temp.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _save_npz_once(path: Path, **arrays: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing fitted output: {path}")
    temp = path.with_name(f".{path.name}.{os.getpid()}.partial.npz")
    try:
        np.savez_compressed(temp, **arrays)
        os.link(temp, path)
    finally:
        if temp.exists():
            temp.unlink()
    return file_sha256(path)


def _submission_frame(test: pd.DataFrame, sample: pd.DataFrame, predictions: np.ndarray,
                      target: str, task: str) -> pd.DataFrame:
    return initial._submission_frame(test, sample, predictions, target, task)


def _write_csv_once(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing submission CSV: {path}")
    payload = frame.to_csv(index=False, float_format="%.17g").encode("utf-8")
    _write_once(path, payload)
    reread = pd.read_csv(path, float_precision="round_trip")
    if list(reread.columns) != list(frame.columns) or len(reread) != 29_421:
        raise RuntimeError("Submission CSV read-back schema/row count failed")
    if not reread.record_id.equals(frame.record_id):
        raise RuntimeError("Submission CSV read-back changed official Test IDs/order")
    target = str(frame.columns[1])
    same = (reread[target].equals(frame[target]) if target == "artifact_signal"
            else np.array_equal(reread[target].to_numpy(dtype=np.float64), frame[target].to_numpy(dtype=np.float64)))
    if not same:
        raise RuntimeError("Submission CSV read-back changed prediction values")
    return {
        "path_gitignored": str(path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": file_sha256(path),
        "rows": len(frame),
        "columns": list(frame.columns),
        "ordered_ids_sha256": _ids_sha(frame.record_id),
    }


def _audit_state(state: dict[str, Any], task: str, target: np.ndarray) -> dict[str, Any]:
    if task == "classification":
        if (state.get("fit_rows") != 50_608 or state.get("cutoff") != 79
                or state.get("fit_cycle_min") != 47 or state.get("fit_cycle_max") != 79
                or state.get("test_data_used") is not False
                or state.get("validation_labels_passed_to_fit") is not False
                or state.get("target_sha256") != regression_array_digest(target)):
            raise RuntimeError("C16 full-Train fit-state audit failed")
    else:
        if (state.get("fit_rows") != 50_608 or state.get("fit_cycle_min") != 47
                or state.get("fit_cycle_max") != 79
                or state.get("test_data_used_for_fit") is not False
                or state.get("validation_labels_passed_to_fit") is not False
                or state.get("target_sha256") != regression_array_digest(target)):
            raise RuntimeError("R22 full-Train fit-state audit failed")
    return state


def fit_classification() -> dict[str, Any]:
    freeze = _load_freeze()
    context = preflight()
    if context["status"] != "READY_FOR_SERIAL_FIT":
        raise RuntimeError(f"Classification fit blocked by preflight: {context}")
    train, test, sample, hashes = _load_inputs("classification")
    base, missing, missing_mask, reuse = _verify_classification_reuse(train, test)
    target = train.artifact_signal.to_numpy(dtype=np.int8)
    fit_features = train.loc[:, list(C16_FEATURES)]
    test_features = test.loc[:, list(C16_FEATURES)]
    started = perf_counter()
    with _training_slot("classification") as resources:
        memory_before = resources["available_physical_bytes"]
        with threadpool_limits(limits=THREADS):
            model = FullTrainC16Classifier().fit(fit_features, target)
            raw = model.predict_probability(test_features)
    elapsed = perf_counter() - started
    if len(raw) != len(test) or not np.isfinite(raw).all() or np.any((raw < 0.0) | (raw > 1.0)):
        raise RuntimeError("C16 full-Train Test probabilities are invalid")
    test_cycles = pd.to_numeric(test.cycle_code, errors="raise").to_numpy(dtype=np.int64)
    if not np.array_equal(np.unique(test_cycles), np.arange(80, 92, dtype=np.int64)):
        raise RuntimeError("C16 official Test must contain all complete cycles 80-91")
    probability = route_c16_probabilities(base, missing, raw, missing_mask)
    labels = (probability >= C16_THRESHOLD).astype(np.int64)
    state = _audit_state(model.audit_state(), "classification", target)
    frame = _submission_frame(test, sample, labels, "artifact_signal", "classification")
    out = _write_csv_once(OUTPUT_ROOT / "classification_c16.csv", frame)
    arrays_hash = _save_npz_once(RUN_ROOT / "classification_c16_outputs.npz",
                                 base=base, missing=missing, c16_raw=raw,
                                 probability=probability, labels=labels,
                                 stream_missing=missing_mask)
    state_payload = {
        "task": "classification", "candidate": "C16",
        "fit_state": state,
        "reused_full_train_components": reuse,
        "prediction_audit": {
            "threshold": C16_THRESHOLD,
            "present_rows": int((~missing_mask).sum()),
            "missing_rows": int(missing_mask.sum()),
            "label_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
            "probability_sha256": _array_sha(probability),
            "label_sha256": hashlib.sha256(labels.astype(np.uint8).tobytes()).hexdigest(),
            "class1_stream_missing_equal_cb_missing": bool(np.array_equal(probability[missing_mask], missing[missing_mask])),
        },
        "official_inputs": hashes,
        "execution_head": freeze["execution_head"],
        "fit_seconds": elapsed,
        "memory_available_before_fit_bytes": memory_before,
        "resource_preflight_at_fit": resources,
        "rss_after_fit_bytes": int(psutil.Process().memory_info().rss),
        "threads": THREADS,
        "validation_or_test_labels_passed_to_fit": False,
        "csv": out,
        "prediction_artifact_sha256": arrays_hash,
        "status": "PREPARED_VALIDATED_NOT_SUBMITTED",
    }
    _write_once(RUN_ROOT / "classification_c16_fit_audit.json",
                (json.dumps(state_payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())
    return state_payload


def fit_regression() -> dict[str, Any]:
    freeze = _load_freeze()
    context = preflight()
    if context["status"] != "READY_FOR_SERIAL_FIT":
        raise RuntimeError(f"Regression fit blocked by preflight: {context}")
    train, test, sample, hashes = _load_inputs("regression")
    champion, champion_audit = _verify_regression_reuse(test)
    target = train.threat_rank.to_numpy(dtype=np.float64)
    fit_features = train.loc[:, list(R22_FEATURES)]
    test_features = test.loc[:, list(R22_FEATURES)]
    cycles_float = pd.to_numeric(test.cycle_code, errors="raise").to_numpy(dtype=np.float64)
    if (not np.isfinite(cycles_float).all()
            or not np.equal(cycles_float, np.floor(cycles_float)).all()):
        raise RuntimeError("Official Test cycle_code values must be finite integers")
    cycles = cycles_float.astype(np.int64)
    if not np.array_equal(np.unique(cycles), np.arange(80, 92, dtype=np.int64)):
        raise RuntimeError("R22 rank mapping requires all complete official Test cycles 80-91")
    started = perf_counter()
    with _training_slot("regression") as resources:
        memory_before = resources["available_physical_bytes"]
        with threadpool_limits(limits=THREADS):
            model = FullTrainR22Regressor().fit(fit_features, target)
            raw = model.predict(test_features)
    elapsed = perf_counter() - started
    if len(raw) != len(test) or not np.isfinite(raw).all():
        raise RuntimeError("R22 full-Train Test raw predictions are invalid")
    final, component = r22_original_prediction(raw, champion, cycles)
    if not np.isfinite(final).all() or np.any((final < 0.0) | (final > 100.0)):
        raise RuntimeError("R22_ORIGINAL final predictions are invalid")
    state = _audit_state(model.audit_state(), "regression", target)
    frame = _submission_frame(test, sample, final, "threat_rank", "regression")
    out = _write_csv_once(OUTPUT_ROOT / "regression_r22_original.csv", frame)
    arrays_hash = _save_npz_once(RUN_ROOT / "regression_r22_original_outputs.npz",
                                 champion=champion, raw_xgboost=raw,
                                 rank50_component=component, prediction=final,
                                 test_cycle_code=cycles)
    cycle_sizes = {str(int(c)): int(np.sum(cycles == c)) for c in np.unique(cycles)}
    state_payload = {
        "task": "regression", "candidate": "R22_ORIGINAL",
        "fit_state": state,
        "champion": champion_audit,
        "prediction_audit": {
            "policy": "clip(0.75 * three-component champion + 0.25 * rank50(XGBoost raw), 0, 100)",
            "rank50": "0.5 * raw + 0.5 * (100 * average_rank(raw within complete cycle) / cycle_size)",
            "tie_method": "average",
            "rank_population": "all official Test rows in each complete cycle",
            "cycle_counts": cycle_sizes,
            "raw_range": [float(raw.min()), float(raw.max())],
            "rank50_component_range": [float(component.min()), float(component.max())],
            "prediction_range": [float(final.min()), float(final.max())],
            "clipped_rows_at_final_blend": int(np.sum((0.75 * champion + 0.25 * component < 0.0) | (0.75 * champion + 0.25 * component > 100.0))),
            "prediction_sha256": _array_sha(final),
        },
        "official_inputs": hashes,
        "execution_head": freeze["execution_head"],
        "fit_seconds": elapsed,
        "memory_available_before_fit_bytes": memory_before,
        "resource_preflight_at_fit": resources,
        "rss_after_fit_bytes": int(psutil.Process().memory_info().rss),
        "threads": THREADS,
        "validation_or_test_labels_passed_to_fit": False,
        "csv": out,
        "prediction_artifact_sha256": arrays_hash,
        "status": "PREPARED_VALIDATED_NOT_SUBMITTED",
    }
    _write_once(RUN_ROOT / "regression_r22_original_fit_audit.json",
                (json.dumps(state_payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())
    return state_payload


def write_status_report() -> dict[str, Any]:
    """Record local-ledger review and the live-auth blocker; do not upload."""
    class_audit = RUN_ROOT / "classification_c16_fit_audit.json"
    reg_audit = RUN_ROOT / "regression_r22_original_fit_audit.json"
    if not class_audit.is_file() or not reg_audit.is_file():
        raise RuntimeError("Both locally prepared CSVs and fit audits must exist first")
    files: list[dict[str, Any]] = []
    exact_messages = {
        _contract()["authorization"]["classification"]["message"],
        _contract()["authorization"]["regression"]["message"],
    }
    roots = [ROOT / "reports", ROOT / "experiments/optimization_v2",
             ROOT / "experiments/optimization_v3", ROOT / "experiments/dual_campaign",
             ROOT / "submissions"]
    def extract_records(node: Any, output: list[dict[str, Any]]) -> None:
        if isinstance(node, dict):
            if any(key in node for key in ("reference", "ref", "submission_id")) and any(
                    key in node for key in ("status", "description", "file_name", "message")):
                output.append({key: node.get(key) for key in [
                    "reference", "ref", "submission_id", "status", "description",
                    "file_name", "message", "public_score", "date", "submitted_at"
                ] if key in node})
            for value in node.values():
                extract_records(value, output)
        elif isinstance(node, list):
            for value in node:
                extract_records(value, output)

    for base in roots:
        for path in base.rglob("*ledger*"):
            if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl"}:
                continue
            content = path.read_text(encoding="utf-8", errors="replace")
            found_messages = [message for message in exact_messages if message in content]
            records: list[dict[str, Any]] = []
            try:
                if path.suffix.lower() == ".json":
                    extract_records(_read_json(path), records)
                else:
                    for line in content.splitlines():
                        if line.strip():
                            extract_records(json.loads(line), records)
            except (json.JSONDecodeError, OSError):
                records = [{"parse_status": "UNPARSEABLE"}]
            files.append({"path": str(path.relative_to(ROOT)).replace("\\", "/"),
                          "sha256": file_sha256(path), "record_count": len(records),
                          "records": records,
                          "authorized_message_match": found_messages})
    duplicate_authorizations = [item for item in files if item["authorized_message_match"]]
    class_preflight = _read_json(ROOT / "reports/optimization_v3/kaggle/classification_preflight.json")
    reg_preflight = _read_json(ROOT / "reports/optimization_v3/kaggle/regression_preflight.json")
    report = {
        "status": "PREPARED_BLOCKED_AUTH_AND_LIVE_PREFLIGHT",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_head": _head(),
        "submission_authorization": _contract()["authorization"],
        "local_ledger_review": {
            "ledger_files_scanned": len(files),
            "ledger_inventory": files,
            "authorized_messages_already_present_locally": duplicate_authorizations,
            "local_idempotency_check": "No matching authorized messages found in local ledger files" if not duplicate_authorizations else "Matching message exists locally; upload must remain blocked",
        },
        "historical_live_preflight_snapshots": {
            "classification": {"captured_at_utc": class_preflight.get("at_utc"),
                               "team": class_preflight.get("team_name"),
                               "deadline_api": class_preflight.get("deadline_api"),
                               "rules_sha256": class_preflight.get("rules_sha256"),
                               "quota_snapshot": {key: class_preflight.get(key) for key in ["max_daily_submissions", "num_today", "num_total", "num_allowed_now"]}},
            "regression": {"captured_at_utc": reg_preflight.get("at_utc"),
                           "team": reg_preflight.get("team_name"),
                           "deadline_api": reg_preflight.get("deadline_api"),
                           "rules_sha256": reg_preflight.get("rules_sha256"),
                           "quota_snapshot": {key: reg_preflight.get(key) for key in ["max_daily_submissions", "num_today", "num_total", "num_allowed_now"]}},
        },
        "live_kaggle_preflight": {
            "status": "NOT_AVAILABLE",
            "attempted": False,
            "reason": "Authenticated Kaggle CLI/API runtime is unavailable; no safe authentication was present in configured CLI/runtime. Cached preflight data is historical and cannot establish current shared quota, rules, deadline, team, or ledger state.",
            "upload_attempted": False,
            "chrome_used": False,
        },
        "internal_validation": _verify_continuation_verdicts(),
        "public_scores": {
            "classification_C16": {"status": "NOT_SCORED", "public_score": None, "submission_ref": None},
            "regression_R22_ORIGINAL": {"status": "NOT_SCORED", "public_score": None, "submission_ref": None},
            "champion_reference_scores_are_not_candidate_scores": True,
        },
        "classification": _read_json(class_audit),
        "regression": _read_json(reg_audit),
        "next_action": "Do not upload or retry until authenticated Kaggle API/CLI works and live team/rules/deadline/quota/full-ledger/idempotency checks pass.",
    }
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    destination = REPORT_ROOT / "SUBMISSION_STATUS.json"
    if destination.exists():
        raise FileExistsError("Refusing to replace the prepared submission status report")
    _write_once(destination, (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["freeze", "preflight", "fit-classification", "fit-regression", "write-status"])
    args = parser.parse_args()
    if args.action == "freeze":
        result = freeze_execution()
    elif args.action == "preflight":
        result = preflight()
    elif args.action == "fit-classification":
        result = fit_classification()
    elif args.action == "fit-regression":
        result = fit_regression()
    else:
        result = write_status_report()
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
