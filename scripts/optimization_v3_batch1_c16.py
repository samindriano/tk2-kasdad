"""C16 frozen sparse-classification pilot on official Train only."""
from __future__ import annotations

import json
import platform
import re
import warnings
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import psutil

from scripts.optimization_v3_batch1_verify import verify_report
from scripts.optimization_v3_sparse_classification import (
    CACHE as SPARSE_PILOT_CACHE,
    CONFIG as SPARSE_PILOT_CONFIG,
    check as check_sparse_pilot,
)
from scripts.optimization_v3_sparse_classification_fullsix import (
    CB_CACHE,
    control_record,
    pilot_record,
)
from src.models.optimization_v3_sparse_classification import PARAMS as BASE_PARAMS
from src.models.classification_models import _TFIDF_PARAMS
from src.models.optimization_v3_batch1 import (
    BatchIntegrityError,
    append_event,
    evaluation_views_for_card,
    file_sha256,
    json_sha256,
    load_queue,
    make_freeze_identity,
    metric,
    read_bundle,
    run_pilot,
    write_json_exclusive,
)
from src.models.optimization_v3_batch1_c16 import (
    BLEND_WEIGHT, CHANGES,
    FEATURE_NAMES,
    PARAMS,
    THRESHOLD,
    fit_predict,
)
from src.models.regression_research_text import _text_values
from src.utils.paths import task_files


ROOT = Path(__file__).resolve().parents[1]
CARD_ID = "C16"
QUEUE_PATH = ROOT / "experiments/optimization_v3/research_backlog/queue.json"
CARD_PATH = ROOT / "reports/optimization_v3/research_backlog/cards/C16.md"
EXPERIMENT_DIR = ROOT / "experiments/optimization_v3/batch1/C16"
RUN_DIR = ROOT / "experiments/runs/optimization_v3_batch1/C16"
REPORT_DIR = ROOT / "reports/optimization_v3/batch1/C16"
RECIPE_PATH = EXPERIMENT_DIR / "recipe.json"
FREEZE_PATH = EXPERIMENT_DIR / "freeze.json"
ENGINE_REPORT_PATH = RUN_DIR / "pilot_engine_report.json"
RESULTS_PATH = REPORT_DIR / "results.json"
VERIFICATION_PATH = REPORT_DIR / "verification.json"
GATE_PATH = REPORT_DIR / "gate.json"
SUMMARY_PATH = REPORT_DIR / "summary.json"
LEDGER_PATH = RUN_DIR.parent / "batch_ledger.jsonl"
BASELINE_MANIFEST_PATH = ROOT / "reports/optimization_v3/kaggle/v3_cls_sparse25.json"
FULLSIX_REPORT_PATH = ROOT / "reports/optimization_v3/classification_sparse_full/fullsix.json"
FULLSIX_VERIFICATION_PATH = ROOT / "reports/optimization_v3/classification_sparse_full/verification.json"
FULLSIX_WRAPPER_PATH = ROOT / "reports/optimization_v3/classification_sparse_full/verification_wrapper_run.json"
ROUND1_CACHE = ROOT / "experiments/runs/classification_round1_v1"
ROUND1_AUDITS = ROOT / "reports/classification_round1/fit_audits"

TRAIN_SHA256 = "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c"
QUEUE_SHA256 = "822dcd20fa28fd0e9ddc841a8c3709b372dacaca305b94b39b11a9203418fec7"
CARD_SHA256 = "29fd6319dfbf0e9f64083d5a91d98612e7834cd7fd35f97a122012359d50c73a"
MANIFEST_SHA256 = "6e7c2cb50fe6d0ee4767a44c40f152b6538b933fab6cf7082659b16d791f7d1e"
EXPECTED_FIT_ROWS = {55: 12_020, 67: 29_705}
EXPECTED_SCORE_ROWS_BY_CUTOFF = {55: 17_685, 67: 20_903}
EXPECTED_VIEW_ROWS = {
    "W1": 4_114,
    "W4": 4_950,
    "primary": 9_064,
    "H12": 38_588,
}
EXPECTED_VIEWS = {
    "W1": [(55, 56, 59)],
    "W4": [(67, 68, 71)],
    "primary": [(55, 56, 59), (67, 68, 71)],
    "H12": [(55, 56, 67), (67, 68, 79)],
}
PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "threadpoolctl", "psutil"]
JOB_MEMORY_CAP_BYTES = 6 * 1024**3
FREE_RAM_RESERVE_BYTES = 2 * 1024**3
THREADS = 2


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _require_hash(path: str | Path, expected: str, label: str) -> None:
    actual = file_sha256(path)
    if actual != expected:
        raise BatchIntegrityError(f"{label} SHA256 mismatch: {actual}")


def _write_or_match(path: Path, value: Mapping[str, Any]) -> None:
    if path.is_file():
        if _read_json(path) != dict(value):
            raise BatchIntegrityError(f"Existing C16 report differs: {path}")
    else:
        write_json_exclusive(path, dict(value))


def _freeze_matches(frozen: Mapping[str, Any], identity: Mapping[str, Any]) -> bool:
    expected = json_sha256(identity)
    try:
        return (frozen.get("signature") == expected
                and json_sha256(frozen.get("identity")) == expected)
    except (TypeError, ValueError):
        return False


def _validate_card(queue: Mapping[str, Any], card: Mapping[str, Any]) -> None:
    _require_hash(QUEUE_PATH, QUEUE_SHA256, "canonical queue")
    _require_hash(CARD_PATH, CARD_SHA256, "C16 card")
    canonical = next((row for row in queue.get("cards", []) if row.get("id") == CARD_ID), None)
    if canonical is None or dict(canonical) != dict(card):
        raise BatchIntegrityError("C16 card object differs from the canonical queue")
    expected = {
        "id": "C16",
        "task": "classification",
        "family": "sparse_capacity",
        "policy": "C_SPARSE_REPLACE",
        "pilot_predictor_fits": 2,
        "evaluation": "STANDARD",
        "condition": "No additional dependency",
    }
    for name, value in expected.items():
        if card.get(name) != value:
            raise BatchIntegrityError(f"Canonical C16 field changed: {name}")
    expected_changes = {"num_leaves": 31, "min_child_samples": 100, "reg_lambda": 5.0}
    if CHANGES != expected_changes or PARAMS != {**dict(BASE_PARAMS), **expected_changes}:
        raise BatchIntegrityError("C16 parameters differ from its fixed simultaneous capacity recipe")
    baseline = queue.get("baseline", {}).get("classification", {})
    if (baseline.get("Train_sha256") != TRAIN_SHA256
            or baseline.get("manifest_sha256") != MANIFEST_SHA256
            or baseline.get("manifest") != "reports/optimization_v3/kaggle/v3_cls_sparse25.json"
            or not np.isclose(float(baseline.get("primary", np.nan)), 0.8299435476765076, rtol=0, atol=1e-15)
            or not np.isclose(float(baseline.get("H12", np.nan)), 0.8029655654017788, rtol=0, atol=1e-15)):
        raise BatchIntegrityError("C16 fixed sparse25 baseline differs from the canonical queue")
    preset = queue.get("execution_preset", {})
    if (int(preset.get("concurrent_fit_jobs", -1)) != 1
            or int(preset.get("threads_per_job", -1)) != THREADS
            or float(preset.get("max_job_RSS_GiB", -1)) != 6.0
            or float(preset.get("minimum_free_RAM_GiB_before_start", -1)) != 4.0):
        raise BatchIntegrityError("C16 execution/resource policy differs from the queue")
    gate = queue.get("evaluation", {}).get("classification_gate", {})
    if (float(gate.get("pilot_primary_gain_min", np.nan)) != 0.0005
            or float(gate.get("pilot_H12_harm_max", np.nan)) != 0.001
            or float(gate.get("individual_pilot_view_harm_max", np.nan)) != 0.005
            or int(gate.get("supported_group_n_min", -1)) != 200
            or float(gate.get("supported_group_F1_harm_max", np.nan)) != 0.01
            or float(gate.get("positive_class_recall_harm_max", np.nan)) != 0.02
            or int(gate.get("positive_and_negative_per_group_min", -1)) != 30):
        raise BatchIntegrityError("C16 classification pilot gates differ from the canonical queue")


def _validate_baseline_artifacts(queue: Mapping[str, Any]) -> dict[str, Any]:
    _require_hash(BASELINE_MANIFEST_PATH, MANIFEST_SHA256, "accepted sparse25 manifest")
    manifest = _read_json(BASELINE_MANIFEST_PATH)
    if (manifest.get("status") != "PREPARED_AND_QA_PASS"
            or manifest.get("task") != "classification"
            or manifest.get("candidate_id") != "v3_cls_sparse25"
            or manifest.get("input_sha256", {}).get("train") != TRAIN_SHA256
            or manifest.get("fit_preprocessing_only_train") is not True
            or manifest.get("test_labels_used") is not False
            or manifest.get("recipe") != {
                "kind": "availability_present_blend",
                "present": [{"model": "C2_B", "weight": 0.75},
                            {"model": "SPARSE_BINARY", "weight": 0.25}],
                "missing": "CB_MISSING", "threshold": 0.5,
            }):
        raise BatchIntegrityError("Accepted sparse25 manifest is not the frozen present-blend recipe")
    decision_path = ROOT / str(manifest.get("decision_path", ""))
    _require_hash(decision_path, str(manifest.get("decision_sha256", "")), "accepted sparse25 decision")
    fullsix = _read_json(FULLSIX_REPORT_PATH)
    independent = _read_json(FULLSIX_VERIFICATION_PATH)
    wrapper = _read_json(FULLSIX_WRAPPER_PATH)
    primary = fullsix.get("results", {}).get("primary_six_window_pooled", {}).get("models", {}).get("blend25", {})
    h12 = fullsix.get("results", {}).get("H12_fixed_origin_pooled", {}).get("models", {}).get("blend25", {})
    baseline = queue.get("baseline", {}).get("classification", {})
    if (fullsix.get("test_read") is not False or fullsix.get("kaggle_access") is not False
            or independent.get("status") != "PASS" or wrapper.get("status") != "PASS"
            or not np.isclose(float(primary.get("macro_f1", np.nan)), float(baseline.get("primary")), rtol=0, atol=1e-15)
            or not np.isclose(float(h12.get("macro_f1", np.nan)), float(baseline.get("H12")), rtol=0, atol=1e-15)):
        raise BatchIntegrityError("C16 full-six evidence or independent verification changed")
    return {
        "manifest_path": str(BASELINE_MANIFEST_PATH.relative_to(ROOT)),
        "manifest_sha256": MANIFEST_SHA256,
        "decision_path": str(decision_path.relative_to(ROOT)),
        "decision_sha256": str(manifest["decision_sha256"]),
        "fullsix_path": str(FULLSIX_REPORT_PATH.relative_to(ROOT)),
        "fullsix_sha256": file_sha256(FULLSIX_REPORT_PATH),
        "fullsix_verification_sha256": file_sha256(FULLSIX_VERIFICATION_PATH),
        "fullsix_wrapper_sha256": file_sha256(FULLSIX_WRAPPER_PATH),
        "declared_fullsix_control": {
            "primary_macro_f1": float(baseline["primary"]),
            "H12_macro_f1": float(baseline["H12"]),
        },
    }


def _official_train() -> tuple[Path, pd.DataFrame, np.ndarray, np.ndarray]:
    train_path = task_files("classification").get("train")
    if train_path is None or not Path(train_path).is_file():
        raise FileNotFoundError("Official Classification Train is missing")
    path = Path(train_path)
    frame = pd.read_csv(path)
    if "artifact_signal" not in frame:
        raise BatchIntegrityError("Official Classification Train has no artifact_signal target")
    target = pd.to_numeric(frame["artifact_signal"], errors="raise").to_numpy(dtype=np.int8)
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    return path, frame, target, cycles


def _memory_estimate(fit_features: pd.DataFrame, available_bytes: int | None = None) -> dict[str, Any]:
    fit_rows = len(fit_features)
    docs, _missing = _text_values(fit_features["signal_stream"])
    token_counts = np.fromiter(
        (len(re.findall(r"(?u)\b\w\w+\b", value.lower())) for value in docs),
        dtype=np.int64,
        count=fit_rows,
    )
    nnz_upper = int(np.sum(token_counts + np.maximum(token_counts - 1, 0), dtype=np.int64))
    sparse_bytes = int(nnz_upper * 16 + (fit_rows + 1) * 8)
    estimated_peak = int((1.5 * 1024**3) + 3 * sparse_bytes)
    available = int(psutil.virtual_memory().available if available_bytes is None else available_bytes)
    allowed = min(JOB_MEMORY_CAP_BYTES, available - FREE_RAM_RESERVE_BYTES)
    return {
        "status": "PASS" if allowed > 0 and estimated_peak <= allowed else "BLOCKED_RESOURCE",
        "fit_rows": int(fit_rows),
        "estimated_nnz_upper_bound": nnz_upper,
        "sparse_design_bytes_upper_bound": sparse_bytes,
        "estimated_peak_bytes": estimated_peak,
        "estimated_peak_is_observed": False,
        "available_physical_bytes": available,
        "free_memory_reserve_bytes": FREE_RAM_RESERVE_BYTES,
        "job_memory_cap_bytes": JOB_MEMORY_CAP_BYTES,
        "allowed_bytes": int(max(0, allowed)),
        "threads": THREADS,
        "concurrent_training_jobs": 1,
    }


def _control_input_files(train_path: Path, card_path: Path, views: Mapping[str, Any]) -> dict[str, Path]:
    cutoffs = sorted({int(cutoff) for segments in views.values() for cutoff, _, _ in segments})
    files: dict[str, Path] = {
        "official_classification_train": train_path,
        "canonical_queue": QUEUE_PATH,
        "canonical_card": card_path,
        "accepted_sparse25_manifest": BASELINE_MANIFEST_PATH,
        "accepted_sparse25_decision": ROOT / "experiments/optimization_v3/decisions/v3_cls_sparse25.json",
        "accepted_sparse25_fullsix_report": FULLSIX_REPORT_PATH,
        "accepted_sparse25_fullsix_verification": FULLSIX_VERIFICATION_PATH,
        "accepted_sparse25_fullsix_wrapper": FULLSIX_WRAPPER_PATH,
        "sparse_pilot_protocol": SPARSE_PILOT_CONFIG / "protocol.json",
        "sparse_pilot_freeze": SPARSE_PILOT_CONFIG / "freeze.json",
    }
    for cutoff in cutoffs:
        files[f"sparse_pilot_{cutoff}_npz"] = SPARSE_PILOT_CACHE / f"fit{cutoff}.npz"
        files[f"sparse_pilot_{cutoff}_sidecar"] = SPARSE_PILOT_CACHE / f"fit{cutoff}.json"
        files[f"catboost_{cutoff}_npz"] = CB_CACHE / f"fit{cutoff}.npz"
        files[f"catboost_{cutoff}_sidecar"] = CB_CACHE / f"fit{cutoff}.json"
        files[f"round1_c2b_{cutoff}"] = ROUND1_CACHE / f"fit{cutoff}__C2_B.npz"
        files[f"round1_c2b_audit_{cutoff}"] = ROUND1_AUDITS / f"fit{cutoff}.json"
    return files


def _source_files() -> list[Path]:
    return [
        Path(__file__),
        ROOT / "src/models/optimization_v3_batch1_c16.py",
        ROOT / "src/models/optimization_v3_batch1.py",
        ROOT / "scripts/optimization_v3_batch1_verify.py",
        ROOT / "src/models/optimization_v3_sparse_classification.py",
        ROOT / "scripts/optimization_v3_sparse_classification.py",
        ROOT / "scripts/optimization_v3_sparse_classification_fullsix.py",
        ROOT / "scripts/optimization_v3_verify_sparse_classification.py",
        ROOT / "scripts/optimization_v3_verify_sparse_classification_fullsix.py",
        ROOT / "scripts/research_campaign_classification_stage2.py",
        ROOT / "scripts/research_campaign_classification.py",
        ROOT / "scripts/optimization_v2_classification.py",
        ROOT / "src/data/audit.py",
        ROOT / "src/models/classification_models.py",
        ROOT / "src/models/classification_round2_models.py",
        ROOT / "src/models/classification_metrics.py",
        ROOT / "src/models/round1_evaluation.py",
        ROOT / "src/models/round1_models.py",
        ROOT / "src/models/regression_research_text.py",
        ROOT / "src/models/regression_round2_models.py",
        ROOT / "src/utils/paths.py",
        ROOT / "tests/test_optimization_v3_batch1_c16.py",
    ]


def _load_controls(frame: pd.DataFrame, target: np.ndarray):
    check_sparse_pilot()

    def load(cutoff: int, score_rows: np.ndarray) -> dict[str, np.ndarray]:
        control = control_record(frame, target, int(cutoff))
        incumbent = pilot_record(frame, control)
        if not np.array_equal(control["rows"], score_rows) or not np.array_equal(incumbent["rows"], score_rows):
            raise BatchIntegrityError(f"C16 fixed controls are not row-aligned at cutoff {cutoff}")
        return {
            "baseline_prediction": np.asarray(incumbent["blend25"], dtype=np.float64),
            "baseline_sparse_raw": np.asarray(incumbent["raw_candidate"], dtype=np.float64),
            "c2b_probability": np.asarray(control["c2b"], dtype=np.float64),
            "cb_probability": np.asarray(control["cb_missing"], dtype=np.float64),
            "stream_missing": np.asarray(control["missing"], dtype=bool),
            "control_rows": np.asarray(score_rows, dtype=np.int64),
        }

    return load


def _control_metrics(target: np.ndarray, cycles: np.ndarray, views: Mapping[str, Any], controls: Mapping[int, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, segments in views.items():
        row_parts: list[np.ndarray] = []
        prediction_parts: list[np.ndarray] = []
        for cutoff, low, high in segments:
            saved = controls[int(cutoff)]
            mask = (cycles[saved["rows"]] >= low) & (cycles[saved["rows"]] <= high)
            row_parts.append(saved["rows"][mask])
            prediction_parts.append(saved["baseline_prediction"][mask])
        rows = np.concatenate(row_parts)
        if len(np.unique(rows)) != len(rows) or len(rows) != EXPECTED_VIEW_ROWS[name]:
            raise BatchIntegrityError(f"C16 fixed control view denominator differs for {name}")
        metric_value = metric("classification", target[rows], np.concatenate(prediction_parts))
        output[name] = {"rows": int(len(rows)), "macro_f1": float(metric_value["macro_f1"])}
    return output


def _recipe_payload(card: Mapping[str, Any], views: Mapping[str, Any], control_metrics: Mapping[str, Any], queue: Mapping[str, Any]) -> dict[str, Any]:
    gate = dict(queue["evaluation"]["classification_gate"])
    return {
        "card_id": CARD_ID,
        "queue_recipe": {key: card[key] for key in (
            "id", "task", "name", "family", "priority", "change", "policy",
            "pilot_predictor_fits", "evaluation", "condition",
        )},
        "source_model": "SparseTextBoostClassifier design with C16 fixed sparse-capacity changes",
        "base_lightgbm_parameters": dict(BASE_PARAMS),
        "lightgbm_parameters": dict(PARAMS),
        "only_parameter_changes": dict(CHANGES),
        "text_recipe": {
            "vectorizer": "sklearn.feature_extraction.text.TfidfVectorizer",
            "parameters": {
                key: (list(value) if isinstance(value, tuple) else value.__name__ if isinstance(value, type) else value)
                for key, value in _TFIDF_PARAMS.items()
            },
            "missing_stream": "empty document plus explicit binary missing column",
        },
        "structured_recipe": "RegressionTextModel._structured_fit/_structured_transform; fit only on current cutoff's past rows",
        "feature_allowlist": list(FEATURE_NAMES),
        "fit_population": "all official Classification Train rows with 47 <= cycle_code <= cutoff; no row weights or downsampling",
        "scoring_population": "STANDARD cutoff-55/67 views, row-aligned by saved Train row positions and record_id",
        "prediction_policy": {
            "present_stream": "0.75 * frozen C2_B probability + 0.25 * new C16 moderate-capacity sparse expert probability",
            "missing_stream": "frozen CatBoost missing-stream probability, unchanged",
            "baseline": "fixed v3_cls_sparse25: 0.75 frozen C2_B + 0.25 original sparse expert on present; frozen CatBoost on missing",
            "classification_threshold": THRESHOLD,
        },
        "evaluation_views": {name: [list(part) for part in segments] for name, segments in views.items()},
        "row_matched_control_metrics": dict(control_metrics),
        "accepted_fullsix_control_metrics": {
            "primary_macro_f1": 0.8299435476765076,
            "H12_macro_f1": 0.8029655654017788,
        },
        "pilot_gate": gate,
        "subgroup_support_rule": "guard n>=200 and each class count>=30; smaller or single-class groups descriptive only",
        "resource_policy": {
            "threads": THREADS,
            "concurrent_training_jobs": 1,
            "job_memory_cap_bytes": JOB_MEMORY_CAP_BYTES,
            "free_memory_reserve_bytes": FREE_RAM_RESERVE_BYTES,
        },
        "test_data_used": False,
        "kaggle_access": False,
    }


def _budget_preflight() -> dict[str, Any]:
    events = []
    if LEDGER_PATH.exists():
        events = [json.loads(line) for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    reserved = sum(int(event.get("predictor_fits", 0)) for event in events if event.get("event") == "BUDGET_RESERVED")
    own = [event for event in events if event.get("event") == "BUDGET_RESERVED"
           and event.get("budget_key") == f"{CARD_ID}:PILOT"]
    cap = int(load_queue(QUEUE_PATH)["execution_preset"]["max_successful_new_predictor_fits"])
    if len(own) > 1 or (own and int(own[0].get("predictor_fits", -1)) != 2):
        raise BatchIntegrityError("C16 existing budget reservation is duplicated or has a different size")
    next_total = reserved if own else reserved + 2
    return {
        "status": "PASS" if next_total <= cap else "BLOCKED_BUDGET",
        "reserved_total": reserved,
        "next_total": next_total,
        "existing_c16_reservation": own[0] if own else None,
        "cap": cap,
    }


def _prepare(card: Mapping[str, Any], queue: Mapping[str, Any]) -> dict[str, Any]:
    _validate_card(queue, card)
    evidence = _validate_baseline_artifacts(queue)
    train_path, frame, target, cycles = _official_train()
    if file_sha256(train_path) != TRAIN_SHA256:
        raise BatchIntegrityError("Official Classification Train hash differs from the frozen queue")
    if (len(frame) != 50_608 or not np.array_equal(np.unique(cycles), np.arange(47, 80))
            or frame["record_id"].isna().any() or not frame["record_id"].astype(str).is_unique
            or target.shape != (len(frame),) or not np.isin(target, [0, 1]).all()):
        raise BatchIntegrityError("Official Classification Train schema, IDs, cycles, or target is invalid")
    views = evaluation_views_for_card(queue, CARD_ID)
    if views != EXPECTED_VIEWS:
        raise BatchIntegrityError(f"C16 STANDARD views changed: {views}")
    controls_loader = _load_controls(frame, target)
    score_rows_by_cutoff: dict[int, np.ndarray] = {}
    controls: dict[int, dict[str, np.ndarray]] = {}
    memory: dict[str, Any] = {}
    for cutoff in (55, 67):
        fit_rows = np.flatnonzero(cycles <= cutoff)
        score_cycles = sorted({cycle for segments in views.values() for origin, low, high in segments
                               if origin == cutoff for cycle in range(low, high + 1)})
        score_rows = np.flatnonzero(np.isin(cycles, np.asarray(score_cycles, dtype=np.int64)))
        if len(fit_rows) != EXPECTED_FIT_ROWS[cutoff]:
            raise BatchIntegrityError(f"C16 cutoff {cutoff} fit population differs")
        score_rows_by_cutoff[cutoff] = score_rows
        controls[cutoff] = controls_loader(cutoff, score_rows)
        for key, values in controls[cutoff].items():
            if np.asarray(values).shape != (len(score_rows),):
                raise BatchIntegrityError(f"C16 control {key} is not row-aligned at cutoff {cutoff}")
        memory[str(cutoff)] = _memory_estimate(frame.iloc[fit_rows].loc[:, list(FEATURE_NAMES)])
    row_matched_metrics = _control_metrics(target, cycles, views, {
        cutoff: {"rows": score_rows_by_cutoff[cutoff],
                 "baseline_prediction": controls[cutoff]["baseline_prediction"]}
        for cutoff in score_rows_by_cutoff
    })
    recipe = _recipe_payload(card, views, row_matched_metrics, queue)
    input_files = _control_input_files(train_path, CARD_PATH, views)
    identity = make_freeze_identity(
        card=card, feature_names=FEATURE_NAMES, input_files=input_files,
        source_files=_source_files(), evaluation_views=views, packages=PACKAGES,
    )
    identity.update({
        "recipe": recipe,
        "accepted_baseline_evidence": evidence,
        "expected_fit_rows": {str(cutoff): EXPECTED_FIT_ROWS[cutoff] for cutoff in (55, 67)},
        "expected_view_rows": EXPECTED_VIEW_ROWS,
        "resource_preflight_recipe": {
            "method": "fit-only token and n-gram nonzero upper bound; sparse CSR bytes; fixed 1.5 GiB overhead plus 3x CSR bound",
            "job_memory_cap_bytes": JOB_MEMORY_CAP_BYTES,
            "free_memory_reserve_bytes": FREE_RAM_RESERVE_BYTES,
            "threads": THREADS,
            "concurrent_training_jobs": 1,
        },
    })
    identity["recipe_artifact_sha256"] = json_sha256(recipe)
    signature = json_sha256(identity)
    if RECIPE_PATH.exists() and _read_json(RECIPE_PATH) != recipe:
        raise BatchIntegrityError("Existing C16 recipe differs from canonical preflight")
    if FREEZE_PATH.exists() and not _freeze_matches(_read_json(FREEZE_PATH), identity):
        raise BatchIntegrityError("Existing C16 freeze differs from current source/input/recipe identity")
    budget = _budget_preflight()
    memory_pass = all(item["status"] == "PASS" for item in memory.values())
    status = "PREFLIGHT_PASS" if memory_pass and budget["status"] == "PASS" else "BLOCKED_RESOURCE_OR_BUDGET"
    input_hashes = {name: file_sha256(path) for name, path in sorted(input_files.items())}
    source_hashes = {str(Path(path).relative_to(ROOT)): file_sha256(path) for path in _source_files()}
    public = {
        "card_id": CARD_ID,
        "status": status,
        "scope": "official Classification Train and frozen v3_cls_sparse25 controls; no candidate fit or Test/Sample/Kaggle access",
        "official_train_sha256": TRAIN_SHA256,
        "input_sha256": input_hashes,
        "source_sha256": source_hashes,
        "freeze_identity_sha256": signature,
        "baseline_evidence": evidence,
        "accepted_fullsix_control_metrics": recipe["accepted_fullsix_control_metrics"],
        "row_matched_control_metrics": row_matched_metrics,
        "fit_rows": {str(key): value for key, value in EXPECTED_FIT_ROWS.items()},
        "score_rows": EXPECTED_VIEW_ROWS,
        "memory_estimate": memory,
        "budget_preflight": budget,
        "test_or_sample_access": False,
        "kaggle_access": False,
        "candidate_fit_or_metric_computed": False,
        "artifacts_written": False,
        "python_version": platform.python_version(),
    }
    return {
        "public": public, "identity": identity, "recipe": recipe,
        "frame": frame, "target": target, "cycles": cycles, "views": views,
        "controls": controls, "score_rows_by_cutoff": score_rows_by_cutoff,
        "train_path": train_path,
    }


def _resource_guarded_adapter(fit_features: pd.DataFrame, fit_target: np.ndarray,
                              score_features: pd.DataFrame, **kwargs: Any) -> dict[str, Any]:
    resource = _memory_estimate(fit_features)
    if resource["status"] != "PASS":
        raise BatchIntegrityError(f"C16 per-fit resource guard failed: {resource}")
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = fit_predict(fit_features, fit_target, score_features, **kwargs)
    result["fit_metadata"]["resource_preflight"] = resource
    result["fit_metadata"]["python_warnings"] = [
        {"category": item.category.__name__, "message": str(item.message)} for item in captured
    ]
    return result


def _reconstruct_c16(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(arrays["raw_candidate"], dtype=np.float64)
    missing = np.asarray(arrays["control__stream_missing"], dtype=bool)
    c2b = np.asarray(arrays["control__c2b_probability"], dtype=np.float64)
    cb = np.asarray(arrays["control__cb_probability"], dtype=np.float64)
    old_sparse = np.asarray(arrays["control__baseline_sparse_raw"], dtype=np.float64)
    candidate = np.where(missing, cb, 0.75 * c2b + BLEND_WEIGHT * raw)
    baseline = np.where(missing, cb, 0.75 * c2b + BLEND_WEIGHT * old_sparse)
    return candidate, baseline


def _verify_and_gate(context: Mapping[str, Any], report: Mapping[str, Any],
                     queue: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    verification = verify_report(
        report=dict(report), frame=context["frame"], target=context["target"],
        artifact_dir=RUN_DIR, freeze_signature=json_sha256(context["identity"]),
        evaluation_views=context["views"], prediction_reconstructors={"C16": _reconstruct_c16},
    )
    gate_config = queue["evaluation"]["classification_gate"]
    primary = report["views"]["primary"]["gain_positive_is_better"]
    h12 = report["views"]["H12"]["gain_positive_is_better"]
    w1 = report["views"]["W1"]["gain_positive_is_better"]
    w4 = report["views"]["W4"]["gain_positive_is_better"]
    subgroup_failures = verification["views"]["primary"]["subgroup_diagnostics"]["supported_guard_failures"]
    checks = {
        "primary_gain_at_least_0_0005": bool(primary >= float(gate_config["pilot_primary_gain_min"])),
        "H12_harm_at_most_0_001": bool(h12 >= -float(gate_config["pilot_H12_harm_max"])),
        "W1_harm_at_most_0_005": bool(w1 >= -float(gate_config["individual_pilot_view_harm_max"])),
        "W4_harm_at_most_0_005": bool(w4 >= -float(gate_config["individual_pilot_view_harm_max"])),
        "supported_subgroup_guards": not subgroup_failures,
    }
    decision = "ADVANCE" if all(checks.values()) else "RETIRE"
    verification.update({
        "status": "INDEPENDENT_QA_PASS",
        "decision": decision,
        "gate_checks": checks,
        "threshold": THRESHOLD,
        "test_data_used": False,
        "kaggle_access": False,
    })
    gate = {
        "card_id": CARD_ID,
        "decision": decision,
        "status": "PASS" if all(checks.values()) else "FAIL_VALID_EVIDENCE",
        "checks": checks,
        "thresholds": gate_config,
        "supported_subgroup_failures": subgroup_failures,
        "decision_basis": "Frozen STANDARD temporal pilot gates, exact row-matched v3_cls_sparse25 controls, and independent route/metric reconstruction; no Test/Kaggle evidence.",
    }
    return verification, gate


def _fit_accounting(context: Mapping[str, Any]) -> dict[str, Any]:
    events = [json.loads(line) for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    card_events = [event for event in events if event.get("card_id") == CARD_ID]
    reservations = [event for event in card_events if event.get("event") == "BUDGET_RESERVED"]
    if (len(reservations) != 1 or reservations[0].get("budget_key") != "C16:PILOT"
            or int(reservations[0].get("predictor_fits", -1)) != 2):
        raise BatchIntegrityError("C16 ledger does not have its exact two-fit pilot reservation")
    cutoffs: dict[str, Any] = {}
    for cutoff in (55, 67):
        expected_rows = context["score_rows_by_cutoff"][cutoff]
        identity = {"freeze_signature": json_sha256(context["identity"]), "card_id": CARD_ID,
                    "cutoff": cutoff, "score_rows": expected_rows.tolist()}
        arrays, sidecar = read_bundle(RUN_DIR / f"C16_fit{cutoff}.npz", identity)
        metadata = sidecar.get("metadata", {})
        state = metadata.get("state", {})
        if (int(metadata.get("fit_predictor_count", -1)) != 1
                or int(metadata.get("fit_rows", -1)) != EXPECTED_FIT_ROWS[cutoff]
                or int(metadata.get("score_rows", -1)) != EXPECTED_SCORE_ROWS_BY_CUTOFF[cutoff]
                or state.get("id") != "C16_MODERATE_SPARSE_CAPACITY_LGBM"
                or int(state.get("cutoff", -1)) != cutoff
                or int(state.get("fit_cycle_min", -1)) != 47
                or int(state.get("fit_cycle_max", -1)) != cutoff
                or sum(state.get("fit_cycle_counts", {}).values()) != EXPECTED_FIT_ROWS[cutoff]
                or state.get("lightgbm_parameters") != PARAMS
                or state.get("sample_weight_used") is not False
                or not np.isfinite(arrays["prediction"]).all()
                or np.any((arrays["prediction"] < 0) | (arrays["prediction"] > 1))):
            raise BatchIntegrityError(f"C16 saved fit receipt invalid at cutoff {cutoff}")
        cutoff_events = [event for event in card_events if event.get("cutoff") == cutoff]
        started = sum(event.get("event") == "FIT_STARTED" for event in cutoff_events)
        fit_saved = sum(event.get("event") == "FIT_SAVED" for event in cutoff_events)
        resumed = sum(event.get("event") == "RESUMED_VERIFIED" for event in cutoff_events)
        failed = sum(event.get("event") == "FIT_FAILED" for event in cutoff_events)
        if fit_saved > 1 or (fit_saved + resumed) < 1 or started < 1 or failed > started:
            raise BatchIntegrityError(f"C16 saved fit events invalid at cutoff {cutoff}")
        cutoffs[str(cutoff)] = {
            "fit_predictor_count": 1,
            "fit_rows": EXPECTED_FIT_ROWS[cutoff],
            "score_rows": int(len(expected_rows)),
            "fit_seconds": float(metadata["fit_seconds"]),
            "fit_attempt_count": int(started),
            "failed_attempt_count": int(failed),
            "attempt_events": [event.get("event") for event in cutoff_events
                               if event.get("event") in {"FIT_STARTED", "FIT_SAVED", "FIT_FAILED", "RESUMED_VERIFIED"}],
            "successful_cache_status": "FIT_SAVED" if fit_saved else "VERIFIED_REUSE",
            "training_warnings": metadata.get("fit_metadata", {}).get("python_warnings", []),
            "resource_preflight": metadata.get("fit_metadata", {}).get("resource_preflight", {}),
            "model_state_sha256": state.get("state_sha256"),
        }
    return {
        "budget_reserved_predictor_fits": 2,
        "successful_predictor_fits": 2,
        "cutoff_fits": cutoffs,
        "total_fit_seconds": float(sum(item["fit_seconds"] for item in cutoffs.values())),
        "peak_memory_observed": False,
    }


def _publish(context: Mapping[str, Any], report: Mapping[str, Any], verification: Mapping[str, Any],
             gate: Mapping[str, Any], accounting: Mapping[str, Any]) -> None:
    result = dict(report)
    decision = str(gate["decision"])
    result.update({
        "status": f"PILOT_COMPLETE_{decision}",
        "input_sha256": context["public"]["input_sha256"],
        "source_sha256": context["public"]["source_sha256"],
        "preflight": context["public"],
        "fit_accounting": dict(accounting),
        "resource_note": "Resource memory figures are pre-fit upper-bound estimates; peak process RAM was not observed.",
        "independent_verification": dict(verification),
        "gate": dict(gate),
        "test_data_used": False,
        "sample_submission_used": False,
        "kaggle_access": False,
    })
    summary = {
        "card_id": CARD_ID,
        "decision": decision,
        "primary_rows": EXPECTED_VIEW_ROWS["primary"],
        "primary_baseline_macro_f1": float(report["views"]["primary"]["baseline"]["macro_f1"]),
        "primary_candidate_macro_f1": float(report["views"]["primary"]["candidate"]["macro_f1"]),
        "primary_gain": float(report["views"]["primary"]["gain_positive_is_better"]),
        "H12_rows": EXPECTED_VIEW_ROWS["H12"],
        "H12_baseline_macro_f1": float(report["views"]["H12"]["baseline"]["macro_f1"]),
        "H12_candidate_macro_f1": float(report["views"]["H12"]["candidate"]["macro_f1"]),
        "H12_gain": float(report["views"]["H12"]["gain_positive_is_better"]),
        "W1_gain": float(report["views"]["W1"]["gain_positive_is_better"]),
        "W4_gain": float(report["views"]["W4"]["gain_positive_is_better"]),
        "supported_subgroup_failures": len(gate["supported_subgroup_failures"]),
        "fit_predictor_count": 2,
        "total_fit_seconds": accounting["total_fit_seconds"],
        "test_sample_kaggle_used": False,
        "details": ["results.json", "verification.json", "gate.json"],
    }
    _write_or_match(RESULTS_PATH, result)
    _write_or_match(VERIFICATION_PATH, verification)
    _write_or_match(GATE_PATH, gate)
    _write_or_match(SUMMARY_PATH, summary)


def run_card(card: Mapping[str, Any], queue: Mapping[str, Any], *,
             preflight_only: bool = False, freeze_only: bool = False,
             run_frozen_pilot: bool = False) -> dict[str, Any]:
    if sum((preflight_only, freeze_only, run_frozen_pilot)) != 1:
        raise BatchIntegrityError("Choose exactly one C16 mode")
    context = _prepare(card, queue)
    if preflight_only:
        return context["public"]
    if context["public"]["status"] != "PREFLIGHT_PASS":
        raise BatchIntegrityError("C16 is blocked by frozen input, resource, or budget preflight")
    if freeze_only:
        EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
        if RECIPE_PATH.exists():
            if _read_json(RECIPE_PATH) != context["recipe"]:
                raise BatchIntegrityError("Existing C16 recipe differs; refusing overwrite")
        else:
            write_json_exclusive(RECIPE_PATH, context["recipe"])
        if FREEZE_PATH.exists():
            if not _freeze_matches(_read_json(FREEZE_PATH), context["identity"]):
                raise BatchIntegrityError("Existing C16 freeze differs; refusing overwrite")
        else:
            from src.models.optimization_v3_batch1 import ensure_frozen
            ensure_frozen(FREEZE_PATH, context["identity"])
        result = dict(context["public"])
        result.update({
            "status": "FROZEN_READY",
            "recipe_path": str(RECIPE_PATH.relative_to(ROOT)),
            "freeze_path": str(FREEZE_PATH.relative_to(ROOT)),
            "freeze_sha256": file_sha256(FREEZE_PATH),
            "freeze_written": True,
            "candidate_fit_or_metric_computed": False,
            "artifacts_written": True,
        })
        return result

    if not FREEZE_PATH.is_file() or not RECIPE_PATH.is_file():
        raise BatchIntegrityError("C16 run requires a committed recipe and prefit freeze")
    if not _freeze_matches(_read_json(FREEZE_PATH), context["identity"]):
        raise BatchIntegrityError("C16 freeze changed after protocol was committed")
    target = np.asarray(context["target"], dtype=np.int8)
    frame = context["frame"]
    fit_adapter = _resource_guarded_adapter
    report = run_pilot(
        card=card,
        artifact_dir=RUN_DIR,
        budget_ledger_path=LEDGER_PATH,
        max_successful_fits=int(queue["execution_preset"]["max_successful_new_predictor_fits"]),
        frame=frame,
        target=target,
        feature_names=FEATURE_NAMES,
        adapter=fit_adapter,
        control_loader=lambda cutoff, rows: context["controls"][int(cutoff)],
        freeze_identity=context["identity"],
        freeze_path=FREEZE_PATH,
        event_path=LEDGER_PATH,
        evaluation_views=context["views"],
        threads=THREADS,
        report_path=ENGINE_REPORT_PATH,
    )
    verification, gate = _verify_and_gate(context, report, queue)
    accounting = _fit_accounting(context)
    _publish(context, report, verification, gate, accounting)
    return {
        "card_id": CARD_ID,
        "status": f"PILOT_COMPLETE_{gate['decision']}",
        "decision": gate["decision"],
        "views": verification["views"],
        "gate_checks": gate["checks"],
        "fit_accounting": accounting,
        "freeze_signature": json_sha256(context["identity"]),
        "test_data_used": False,
        "sample_submission_used": False,
        "kaggle_access": False,
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight-only", action="store_true")
    modes.add_argument("--freeze-only", action="store_true")
    modes.add_argument("--run-frozen-pilot", action="store_true")
    args = parser.parse_args()
    queue = load_queue(QUEUE_PATH)
    card = next(item for item in queue["cards"] if item.get("id") == CARD_ID)
    result = run_card(card, queue, preflight_only=args.preflight_only,
                      freeze_only=args.freeze_only, run_frozen_pilot=args.run_frozen_pilot)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["_reconstruct_c16", "_verify_and_gate", "run_card"]
