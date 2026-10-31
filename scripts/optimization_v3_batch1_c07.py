"""C07 bridge for fit-only category-frequency features and frozen controls."""
from __future__ import annotations

import json
from pathlib import Path
import platform
from typing import Any

import numpy as np
import pandas as pd
import psutil
from sklearn.feature_extraction.text import TfidfVectorizer

from scripts.optimization_v3_batch1_verify import verify_report
from scripts.optimization_v3_sparse_classification import CACHE as SPARSE_PILOT_CACHE
from scripts.optimization_v3_sparse_classification import CONFIG as SPARSE_PILOT_CONFIG
from scripts.optimization_v3_sparse_classification import check as check_sparse_pilot
from scripts.optimization_v3_sparse_classification import load_train
from scripts.optimization_v3_sparse_classification_fullsix import (
    CB_CACHE, PILOT_REPORT, control_record, pilot_record,
)
from src.models.optimization_v3_batch1 import (
    BatchIntegrityError,
    append_event,
    evaluation_views_for_card,
    file_sha256,
    json_sha256,
    load_queue,
    make_freeze_identity,
    run_pilot,
    write_json_exclusive,
)
from src.models.optimization_v3_batch1_c07 import (
    BLEND_WEIGHT,
    FEATURE_NAMES,
    FitOnlyCategoryFrequency,
    FREQUENCY_COLUMNS,
    FREQUENCY_FEATURE_NAMES,
    RARE_COUNT_THRESHOLD,
    THRESHOLD,
    CategoryFrequencySparseClassifier,
    fit_predict,
)
from src.models.classification_models import _TFIDF_PARAMS, _text_values
from src.models.classification_round2_models import stream_missing_mask
from src.utils.paths import task_files


ROOT = Path(__file__).resolve().parents[1]
QUEUE_PATH = ROOT / "experiments/optimization_v3/research_backlog/queue.json"
CARD_PATH = ROOT / "reports/optimization_v3/research_backlog/cards/C07.md"
FREEZE_PATH = ROOT / "experiments/optimization_v3/batch1/C07/freeze.json"
ARTIFACT_DIR = ROOT / "experiments/runs/optimization_v3_batch1/C07"
REPORT_DIR = ROOT / "reports/optimization_v3/batch1/C07"

EXPECTED_QUEUE_SHA256 = "822dcd20fa28fd0e9ddc841a8c3709b372dacaca305b94b39b11a9203418fec7"
EXPECTED_CARD_SHA256 = "4cc9f22d9e2412fa351467362d15fb4f25580a32f3a7fb3e2bbe906344d7e530"
EXPECTED_TRAIN_SHA256 = "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c"
EXPECTED_MANIFEST_SHA256 = "6e7c2cb50fe6d0ee4767a44c40f152b6538b933fab6cf7082659b16d791f7d1e"
EXPECTED_DIAGNOSTIC_EVIDENCE_SHA256 = "c26218e0e90715a30b7b92584ddbf6c6d0eac2b3c71ecb5d4c2473c42fe4dc67"
EXPECTED_VIEWS = {
    "W1": [(55, 56, 59)],
    "W4": [(67, 68, 71)],
    "primary": [(55, 56, 59), (67, 68, 71)],
    "H12": [(55, 56, 67), (67, 68, 79)],
}
ACCEPTED_MANIFEST = ROOT / "reports/optimization_v3/kaggle/v3_cls_sparse25.json"
DIAGNOSTIC_EVIDENCE = ROOT / "reports/optimization_v3/classification_diagnostics/audit.json"
ROUND1_CACHE = ROOT / "experiments/runs/classification_round1_v1"
ROUND1_AUDITS = ROOT / "reports/classification_round1/fit_audits"
PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "threadpoolctl", "psutil"]

EXPECTED_CARD_CHANGE = (
    "Append fit-only log1p category count and count<30 flag for each of org/module/pattern "
    "(six columns); missing explicit, unseen count0; unchanged sparse classifier. No target rates."
)
EXPECTED_POLICY = ".75 frozen C2_B +.25 new sparse expert on present; frozen CB missing; threshold.50"


def _require_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Required C07 input is missing: {label} ({path})")
    actual = file_sha256(path)
    if actual != expected:
        raise BatchIntegrityError(f"Frozen C07 {label} hash differs: {actual}")
    return actual


def _assert_card_identity(card: dict[str, Any], queue: dict[str, Any]) -> None:
    expected = {
        "id": "C07",
        "task": "classification",
        "name": "Category frequency/rarity features",
        "family": "category_frequency",
        "change": EXPECTED_CARD_CHANGE,
        "policy": "C_SPARSE_REPLACE",
        "pilot_predictor_fits": 2,
        "evaluation": "STANDARD",
        "condition": "No additional dependency",
        "status": "PLANNED_NOT_RUN",
        "implementation": "NOT_IMPLEMENTED",
    }
    for key, value in expected.items():
        if card.get(key) != value:
            raise BatchIntegrityError(f"C07 queue identity mismatch for {key}: {card.get(key)!r}")
    if card.get("card_path") != "reports/optimization_v3/research_backlog/cards/C07.md":
        raise BatchIntegrityError("C07 queue card path changed")
    if queue.get("policies", {}).get("C_SPARSE_REPLACE") != EXPECTED_POLICY:
        raise BatchIntegrityError("C07 C_SPARSE_REPLACE policy differs from the frozen card")
    if queue["execution_preset"]["concurrent_fit_jobs"] != 1 or queue["execution_preset"]["threads_per_job"] != 2:
        raise BatchIntegrityError("C07 requires one fit job and two threads")
    if queue["evaluation"]["STANDARD"]["pilot_cutoffs"] != [55, 67]:
        raise BatchIntegrityError("C07 STANDARD pilot cutoffs changed")
    gate = queue["evaluation"]["classification_gate"]
    exact_gate = {
        "pilot_primary_gain_min": 0.0005,
        "pilot_H12_harm_max": 0.001,
        "individual_pilot_view_harm_max": 0.005,
        "full_primary_gain_min": 0.0005,
        "full_H12_harm_max": 0.001,
        "full_nonnegative_windows_min": 4,
        "supported_group_n_min": 200,
        "supported_group_F1_harm_max": 0.01,
        "positive_class_recall_harm_max": 0.02,
        "positive_and_negative_per_group_min": 30,
    }
    if gate != exact_gate:
        raise BatchIntegrityError("C07 classification gates changed from their frozen values")
    baseline = queue["baseline"]["classification"]
    if (baseline["manifest"] != "reports/optimization_v3/kaggle/v3_cls_sparse25.json"
            or baseline["manifest_sha256"] != EXPECTED_MANIFEST_SHA256
            or baseline["Train_sha256"] != EXPECTED_TRAIN_SHA256):
        raise BatchIntegrityError("C07 accepted control manifest or Train identity changed")


def _validate_manifest(queue: dict[str, Any]) -> dict[str, Any]:
    if ROOT / queue["baseline"]["classification"]["manifest"] != ACCEPTED_MANIFEST:
        raise BatchIntegrityError("C07 queue points to a different accepted control manifest")
    _require_hash(ACCEPTED_MANIFEST, EXPECTED_MANIFEST_SHA256, "accepted V3 classification manifest")
    manifest = json.loads(ACCEPTED_MANIFEST.read_text(encoding="utf-8"))
    if (manifest.get("status") != "PREPARED_AND_QA_PASS"
            or manifest.get("task") != "classification"
            or manifest.get("candidate_id") != "v3_cls_sparse25"
            or manifest.get("train_rows") != 50_608
            or manifest.get("input_sha256", {}).get("train") != EXPECTED_TRAIN_SHA256
            or manifest.get("fit_preprocessing_only_train") is not True
            or manifest.get("test_labels_used") is not False):
        raise BatchIntegrityError("Accepted V3 sparse25 manifest identity or provenance changed")
    expected_recipe = {
        "kind": "availability_present_blend",
        "present": [
            {"model": "C2_B", "weight": 0.75},
            {"model": "SPARSE_BINARY", "weight": 0.25},
        ],
        "missing": "CB_MISSING",
        "threshold": 0.5,
    }
    if manifest.get("recipe") != expected_recipe:
        raise BatchIntegrityError("Accepted V3 control recipe differs from C07 policy")
    return manifest


def _official_train() -> tuple[Path, pd.DataFrame, np.ndarray, np.ndarray]:
    # load_train resolves only the official Train path through task_files; no Test/Sample is opened.
    frame, target, train_path = load_train(None)
    if not train_path.is_file():
        raise FileNotFoundError("Official Classification Train is missing")
    _require_hash(train_path, EXPECTED_TRAIN_SHA256, "official Classification Train")
    if len(frame) != 50_608:
        raise BatchIntegrityError(f"Unexpected official Classification Train row count: {len(frame)}")
    if frame.record_id.isna().any() or frame.record_id.astype(str).str.strip().eq("").any() or not frame.record_id.astype(str).is_unique:
        raise BatchIntegrityError("Classification Train record_id is blank or duplicated")
    cycles = pd.to_numeric(frame.cycle_code, errors="raise").to_numpy(dtype=np.int64)
    if not np.array_equal(np.unique(cycles), np.arange(47, 80, dtype=np.int64)):
        raise BatchIntegrityError("Classification Train cycle coverage differs from 47–79")
    target = np.asarray(target, dtype=np.int8)
    if target.shape != (len(frame),) or not np.isin(target, [0, 1]).all():
        raise BatchIntegrityError("Classification Train artifact_signal is invalid")
    if set(FEATURE_NAMES) & {"record_id", "artifact_signal", "threat_rank"}:
        raise BatchIntegrityError("C07 feature allowlist contains protected fields")
    return train_path.resolve(), frame, target, cycles


def _load_controls(frame: pd.DataFrame, target: np.ndarray):
    pilot_freeze = check_sparse_pilot()
    pilot_report = json.loads((PILOT_REPORT / "pilot.json").read_text(encoding="utf-8"))
    pilot_verification = json.loads((PILOT_REPORT / "verification.json").read_text(encoding="utf-8"))
    if pilot_verification.get("status") != "PASS":
        raise BatchIntegrityError("Frozen C2_B/CatBoost/sparse25 control pilot did not independently verify")
    if pilot_report.get("freeze_signature") != pilot_freeze.get("signature"):
        raise BatchIntegrityError("Sparse pilot report does not match its frozen identity")
    records: dict[int, dict[str, Any]] = {}

    def load(cutoff: int, score_rows: np.ndarray) -> dict[str, np.ndarray]:
        control = control_record(frame, target, int(cutoff))
        pilot = pilot_record(frame, control)
        if not np.array_equal(control["rows"], score_rows) or not np.array_equal(pilot["rows"], score_rows):
            raise BatchIntegrityError(f"C07 control row positions differ at cutoff {cutoff}")
        missing = np.asarray(control["missing"], dtype=bool)
        c2b = np.asarray(control["c2b"], dtype=np.float64)
        cb = np.asarray(control["cb_missing"], dtype=np.float64)
        old_sparse = np.asarray(pilot["raw_candidate"], dtype=np.float64)
        baseline = np.where(missing, cb, 0.75 * c2b + BLEND_WEIGHT * old_sparse)
        if not np.array_equal(baseline, np.asarray(pilot["blend25"], dtype=np.float64)):
            raise BatchIntegrityError(f"C07 accepted sparse25 route does not reconstruct at {cutoff}")
        if not np.array_equal(np.asarray(control["champion"], dtype=np.float64),
                              np.where(missing, cb, c2b)):
            raise BatchIntegrityError(f"C07 C2_B/CatBoost route does not reconstruct at {cutoff}")
        records[int(cutoff)] = {
            "score_rows": int(len(score_rows)),
            "stream_present_rows": int((~missing).sum()),
            "stream_missing_rows": int(missing.sum()),
            "c2b_control": control["c2b_identity"],
            "catboost_cache_sha256": control["cb_meta"]["npz_sha256"],
            "catboost_prediction_sha256": control["cb_meta"]["prediction_sha256"],
            "sparse_pilot_cache_sha256": pilot["pilot_cache_sha256"],
            "sparse_pilot_freeze_signature": pilot_freeze["signature"],
        }
        return {
            "baseline_prediction": baseline,
            "baseline_sparse_raw": old_sparse,
            "c2b_probability": c2b,
            "cb_probability": cb,
            "stream_missing": missing,
        }

    return load, records, pilot_freeze


def _fit_frequency_preflight(frame: pd.DataFrame, cycles: np.ndarray) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for cutoff in (55, 67):
        fit_rows = np.flatnonzero(cycles <= cutoff)
        fit_x = frame.iloc[fit_rows].loc[:, list(FEATURE_NAMES)]
        # Only fit-prefix category values enter this transform; no validation values/counts are inspected.
        category_features = FitOnlyCategoryFrequency()
        matrix = category_features.fit_transform(fit_x)
        if matrix.shape != (len(fit_rows), 6):
            raise BatchIntegrityError(f"C07 category count columns are not six at cutoff {cutoff}")
        summary[str(cutoff)] = {
            "fit_rows": int(len(fit_rows)),
            "training_columns_nonconstant": {
                name: bool(category_features.audit["fields"][name]["training_nonconstant"])
                for name in FREQUENCY_FEATURE_NAMES
            },
            "category_support": category_features.audit["fields"],
            "target_rates_used": False,
            "validation_category_counts_used": False,
        }
        if not all(summary[str(cutoff)]["training_columns_nonconstant"].values()):
            raise BatchIntegrityError(f"C07 training category-frequency columns are constant at cutoff {cutoff}")
    return summary


def _resource_preflight(frame: pd.DataFrame, cycles: np.ndarray, card: dict[str, Any],
                        queue: dict[str, Any]) -> dict[str, Any]:
    preset = queue["execution_preset"]
    fit_rows = np.flatnonzero(cycles <= 67)
    documents, _ = _text_values(frame.iloc[fit_rows]["signal_stream"])
    analyzer = TfidfVectorizer(**dict(_TFIDF_PARAMS)).build_analyzer()
    text_nnz_upper = int(sum(len(analyzer(document)) for document in documents))
    sparse_bytes = int(text_nnz_upper * 16 + (len(fit_rows) + 1) * 8 + len(fit_rows) * 6 * 8)
    estimated_peak_gib = float(1.5 + 3.0 * sparse_bytes / (1024 ** 3))
    available_gib = float(psutil.virtual_memory().available / (1024 ** 3))
    cap_gib = float(preset["max_job_RSS_GiB"])
    reserve_gib = 2.0
    min_free_gib = float(preset["minimum_free_RAM_GiB_before_start"])
    allowed_gib = min(cap_gib, available_gib - reserve_gib)
    if available_gib < min_free_gib or estimated_peak_gib > allowed_gib:
        raise BatchIntegrityError(
            f"C07 resource gate blocked: available={available_gib:.2f}GiB, "
            f"estimate={estimated_peak_gib:.2f}GiB, allowed={allowed_gib:.2f}GiB")
    if int(preset["concurrent_fit_jobs"]) != 1 or int(preset["threads_per_job"]) != 2:
        raise BatchIntegrityError("C07 requires one serialized fit job with two threads")
    if int(card["pilot_predictor_fits"]) != 2:
        raise BatchIntegrityError("C07 pilot fit count changed")
    if "no dense full-word" not in str(preset.get("RAM_policy", "")).casefold():
        raise BatchIntegrityError("Queue no longer prohibits dense full-word matrices")

    ledger = REPORT_DIR.parent / "batch_ledger.jsonl"
    reservations: dict[str, int] = {}
    if ledger.is_file():
        for line_number, line in enumerate(ledger.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BatchIntegrityError(f"Invalid batch budget ledger line {line_number}") from exc
            if event.get("event") == "BUDGET_RESERVED":
                key, count = str(event["budget_key"]), int(event["predictor_fits"])
                if key in reservations and reservations[key] != count:
                    raise BatchIntegrityError(f"Conflicting existing fit reservation for {key}")
                reservations[key] = count
    key = "C07:PILOT"
    if key in reservations and reservations[key] != int(card["pilot_predictor_fits"]):
        raise BatchIntegrityError("Existing C07 fit reservation differs from the canonical card")
    reserved_after = sum(reservations.values()) + (0 if key in reservations else int(card["pilot_predictor_fits"]))
    fit_cap = int(preset["max_successful_new_predictor_fits"])
    if reserved_after > fit_cap:
        raise BatchIntegrityError(f"C07 fit budget blocked: reserved_after={reserved_after}, cap={fit_cap}")
    return {
        "largest_fit_rows": int(len(fit_rows)),
        "word_ngram_nnz_upper_bound": text_nnz_upper,
        "sparse_design_upper_gib": float(sparse_bytes / (1024 ** 3)),
        "estimated_peak_job_gib": estimated_peak_gib,
        "available_ram_gib_observed": available_gib,
        "allowed_job_ram_gib": allowed_gib,
        "max_job_ram_gib": cap_gib,
        "minimum_free_ram_gib": min_free_gib,
        "reserved_fit_total_after_C07": int(reserved_after),
        "fit_budget_cap": fit_cap,
        "new_predictor_fits": int(card["pilot_predictor_fits"]),
        "concurrent_fit_jobs": 1,
        "threads": 2,
        "no_dense_full_word_matrix": True,
        "estimator_fits_performed": 0,
    }


def _control_input_files(train_path: Path, card_path: Path,
                         views: dict[str, list[tuple[int, int, int]]]) -> dict[str, Path]:
    cutoffs = sorted({int(cutoff) for segments in views.values() for cutoff, _, _ in segments})
    inputs: dict[str, Path] = {
        "official_classification_train": train_path,
        "queue": QUEUE_PATH,
        "card": card_path,
        "accepted_v3_sparse25_manifest": ACCEPTED_MANIFEST,
        "classification_diagnostic_evidence": DIAGNOSTIC_EVIDENCE,
        "frozen_sparse_pilot_protocol": SPARSE_PILOT_CONFIG / "protocol.json",
        "frozen_sparse_pilot_freeze": SPARSE_PILOT_CONFIG / "freeze.json",
        "sparse_pilot_report": PILOT_REPORT / "pilot.json",
        "sparse_pilot_verification": PILOT_REPORT / "verification.json",
    }
    for cutoff in cutoffs:
        inputs[f"sparse_pilot_{cutoff}_npz"] = SPARSE_PILOT_CACHE / f"fit{cutoff}.npz"
        inputs[f"sparse_pilot_{cutoff}_sidecar"] = SPARSE_PILOT_CACHE / f"fit{cutoff}.json"
        inputs[f"c2b_round1_{cutoff}_npz"] = ROUND1_CACHE / f"fit{cutoff}__C2_B.npz"
        inputs[f"c2b_round1_{cutoff}_audit"] = ROUND1_AUDITS / f"fit{cutoff}.json"
        inputs[f"catboost_{cutoff}_npz"] = CB_CACHE / f"fit{cutoff}.npz"
        inputs[f"catboost_{cutoff}_sidecar"] = CB_CACHE / f"fit{cutoff}.json"
    return inputs


def _source_files() -> list[Path]:
    return [
        Path(__file__),
        ROOT / "scripts/optimization_v3_batch1.py",
        ROOT / "scripts/optimization_v3_batch1_verify.py",
        ROOT / "scripts/optimization_v3_sparse_classification.py",
        ROOT / "scripts/optimization_v3_sparse_classification_fullsix.py",
        ROOT / "scripts/optimization_v3_verify_sparse_classification.py",
        ROOT / "scripts/research_campaign_classification.py",
        ROOT / "scripts/research_campaign_classification_stage2.py",
        ROOT / "src/models/optimization_v3_batch1.py",
        ROOT / "src/models/optimization_v3_batch1_c07.py",
        ROOT / "src/models/optimization_v3_sparse_classification.py",
        ROOT / "src/models/campaign_classification_catboost.py",
        ROOT / "src/models/classification_models.py",
        ROOT / "src/models/classification_round2_models.py",
        ROOT / "src/models/classification_metrics.py",
        ROOT / "src/models/regression_research_text.py",
        ROOT / "src/models/regression_round2_models.py",
        ROOT / "src/models/round1_models.py",
        ROOT / "src/models/round1_evaluation.py",
        ROOT / "src/utils/paths.py",
    ]


def reconstruct_c07(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild present-stream blend and unchanged CatBoost missing branch."""
    raw = np.asarray(arrays["raw_candidate"], dtype=np.float64)
    missing = np.asarray(arrays["control__stream_missing"], dtype=bool)
    c2b = np.asarray(arrays["control__c2b_probability"], dtype=np.float64)
    cb = np.asarray(arrays["control__cb_probability"], dtype=np.float64)
    old_sparse = np.asarray(arrays["control__baseline_sparse_raw"], dtype=np.float64)
    if len({len(x) for x in (raw, missing, c2b, cb, old_sparse)}) != 1:
        raise BatchIntegrityError("C07 saved routing arrays have inconsistent row counts")
    candidate = np.where(missing, cb, 0.75 * c2b + BLEND_WEIGHT * raw)
    baseline = np.where(missing, cb, 0.75 * c2b + BLEND_WEIGHT * old_sparse)
    if not np.isfinite(candidate).all() or not np.isfinite(baseline).all():
        raise BatchIntegrityError("C07 route reconstruction contains nonfinite probabilities")
    return candidate, baseline


def _validate_primary_subgroup_gates(verification: dict[str, Any],
                                     gate_config: dict[str, Any]) -> list[dict[str, Any]]:
    """Recheck the verifier's declared support/F1/positive-recall subgroup gates."""
    diagnostics = verification["views"]["primary"].get("subgroup_diagnostics", {})
    groups = diagnostics.get("groups", {})
    failures: list[dict[str, Any]] = []
    gate_fields = {"signal_stream", "entry_friction", "text_load_q"}
    minimum_group = int(gate_config["supported_group_n_min"])
    minimum_per_class = int(gate_config["positive_and_negative_per_group_min"])
    f1_harm = float(gate_config["supported_group_F1_harm_max"])
    recall_harm = float(gate_config["positive_class_recall_harm_max"])
    for field, field_groups in groups.items():
        for label, row in field_groups.items():
            supported = (int(row["n"]) >= minimum_group
                         and int(row["class_0_n"]) >= minimum_per_class
                         and int(row["class_1_n"]) >= minimum_per_class)
            if bool(row["supported"]) != supported:
                raise BatchIntegrityError(f"Independent subgroup support rule differs for {field}/{label}")
            applies = field in gate_fields and supported
            if bool(row["guard_applies"]) != applies:
                raise BatchIntegrityError(f"Independent subgroup gate coverage differs for {field}/{label}")
            if not applies:
                if row["guard_pass"] is not None:
                    raise BatchIntegrityError(f"Unsupported/descriptive subgroup unexpectedly has a gate result: {field}/{label}")
                continue
            f1_delta = float(row["candidate_minus_baseline_macro_f1"])
            recall_delta = float(row["candidate_minus_baseline_positive_recall"])
            passed = f1_delta >= -f1_harm and recall_delta >= -recall_harm
            if row["guard_pass"] is not passed:
                raise BatchIntegrityError(f"Independent subgroup gate result differs for {field}/{label}")
            if not passed:
                failures.append({"field": field, "group": label, **row})
    recorded_failures = diagnostics.get("supported_guard_failures", [])
    recorded_keys = {(item.get("field"), item.get("group")) for item in recorded_failures}
    computed_keys = {(item.get("field"), item.get("group")) for item in failures}
    if recorded_keys != computed_keys:
        raise BatchIntegrityError("Verifier subgroup failure list differs from the canonical C07 gate")
    return failures


def _write_or_match(path: Path, payload: dict[str, Any]) -> None:
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise BatchIntegrityError(f"Existing C07 artifact differs; refusing overwrite: {path}")
    else:
        write_json_exclusive(path, payload)


def _recipe(card: dict[str, Any], views: dict[str, list[tuple[int, int, int]]]) -> dict[str, Any]:
    from src.models.optimization_v3_sparse_classification import PARAMS as SPARSE_PARAMS

    return {
        "card_id": "C07",
        "status": "FROZEN_RECIPE_BEFORE_PILOT",
        "card_sha256": EXPECTED_CARD_SHA256,
        "accepted_control": {
            "manifest": str(ACCEPTED_MANIFEST.relative_to(ROOT)),
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "present_model": "C2_B probability 0.75 + frozen sparse-binary probability 0.25",
            "missing_model": "frozen CatBoost probability unchanged",
            "threshold": THRESHOLD,
        },
        "input_feature_names": list(FEATURE_NAMES),
        "appended_features": list(FREQUENCY_FEATURE_NAMES),
        "category_frequency": {
            "fields": list(FREQUENCY_COLUMNS),
            "fit_scope": "outer cutoff fit rows only",
            "normalization": "frozen structured-category rules: pd.isna -> dedicated missing key; otherwise stringified and stripped, including blank strings",
            "count_transform": "log1p(fit-prefix exact category count)",
            "rarity_flag": f"fit-prefix exact category count < {RARE_COUNT_THRESHOLD}",
            "unseen_count": 0,
            "missing_category_count": "explicit dedicated key; use its fit-prefix count or zero if absent in fit",
            "target_rates_used": False,
            "frequency_columns_required_nonconstant_on_training": True,
        },
        "unchanged_sparse_classifier": {
            "model": "LightGBM LGBMClassifier",
            "parameters": dict(SPARSE_PARAMS),
            "tfidf_parameters": {
                key: list(value) if isinstance(value, tuple) else value.__name__ if isinstance(value, type) else value
                for key, value in _TFIDF_PARAMS.items()
            },
            "structured_preprocessing": "RegressionTextModel._structured_fit/_structured_transform",
            "design_order": "original sparse structured+TFIDF+missing design, then six category-frequency columns",
            "threads": 2,
            "early_stopping": False,
        },
        "policy": EXPECTED_POLICY,
        "evaluation_views": views,
        "fit_predictor_count": int(card["pilot_predictor_fits"]),
        "metric": "Macro F1 at the frozen threshold 0.50",
        "gates": {
            "pilot_primary_gain_min": 0.0005,
            "pilot_H12_harm_max": 0.001,
            "individual_pilot_view_harm_max": 0.005,
            "supported_group_n_min": 200,
            "supported_group_F1_harm_max": 0.01,
            "positive_class_recall_harm_max": 0.02,
            "positive_and_negative_per_group_min": 30,
        },
        "test_data_used": False,
    }


def run_card(card: dict[str, Any], queue: dict[str, Any], *, preflight_only: bool = False) -> dict[str, Any]:
    _assert_card_identity(card, queue)
    _require_hash(QUEUE_PATH, EXPECTED_QUEUE_SHA256, "canonical queue")
    _require_hash(CARD_PATH, EXPECTED_CARD_SHA256, "canonical C07 card")
    if load_queue(QUEUE_PATH) != queue:
        raise BatchIntegrityError("Dispatcher queue differs from the frozen C07 queue")
    if ROOT / card["card_path"] != CARD_PATH:
        raise BatchIntegrityError("C07 queue points to an unexpected card path")
    _require_hash(DIAGNOSTIC_EVIDENCE, EXPECTED_DIAGNOSTIC_EVIDENCE_SHA256, "classification diagnostic evidence")
    _validate_manifest(queue)
    train_path, frame, target, cycles = _official_train()
    views = evaluation_views_for_card(queue, "C07")
    if views != EXPECTED_VIEWS:
        raise BatchIntegrityError(f"C07 evaluation views changed: {views}")

    control_loader, control_metadata, pilot_freeze = _load_controls(frame, target)
    for cutoff in (55, 67):
        rows = np.flatnonzero((cycles > cutoff) & (cycles <= cutoff + 12))
        record = control_loader(cutoff, rows)
        for name, values in record.items():
            if np.asarray(values).shape != (len(rows),):
                raise BatchIntegrityError(f"C07 control {name} is not row-aligned at cutoff {cutoff}")
        missing = np.asarray(record["stream_missing"], dtype=bool)
        if int((~missing).sum()) != control_metadata[cutoff]["stream_present_rows"]:
            raise BatchIntegrityError(f"C07 present/missing coverage changed at cutoff {cutoff}")

    frequency_preflight = _fit_frequency_preflight(frame, cycles)
    resources = _resource_preflight(frame, cycles, card, queue)
    inputs = _control_input_files(train_path, CARD_PATH, views)
    identity = make_freeze_identity(
        card=card, feature_names=FEATURE_NAMES, input_files=inputs,
        source_files=_source_files(), evaluation_views=views, packages=PACKAGES,
    )
    identity.update({
        "accepted_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "sparse_pilot_freeze_signature": pilot_freeze["signature"],
        "control_records": {str(key): value for key, value in sorted(control_metadata.items())},
        "category_frequency_preflight": frequency_preflight,
        "resource_preflight_recipe": {
            "estimated_peak_job_gib": resources["estimated_peak_job_gib"],
            "cap_gib": resources["max_job_ram_gib"],
            "minimum_free_ram_gib": resources["minimum_free_ram_gib"],
            "reserve_gib": 2.0,
            "threads": 2,
            "concurrent_fit_jobs": 1,
            "estimate_method": "fit-only word n-gram upper bound, sparse storage x3 plus six compact float columns and 1.5GiB fixed",
        },
        "python_version": platform.python_version(),
        "validation_labels_passed_to_adapter": False,
        "test_data_used": False,
    })

    if preflight_only:
        return {
            "status": "PREFLIGHT_PASS_NO_FREEZE_NO_FIT",
            "card_id": "C07",
            "train_sha256": EXPECTED_TRAIN_SHA256,
            "accepted_manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "control_records": {str(key): value for key, value in sorted(control_metadata.items())},
            "category_frequency_preflight": frequency_preflight,
            "resource_preflight": resources,
            "evaluation_views": views,
            "input_file_count": len(inputs),
            "source_file_count": len(_source_files()),
            "freeze_identity_sha256": json_sha256(identity),
            "official_test_read": False,
            "sample_submission_read": False,
            "model_fit_performed": False,
            "freeze_or_output_written": False,
        }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    _write_or_match(REPORT_DIR / "recipe.json", _recipe(card, views))
    append_event(REPORT_DIR / "attempts.jsonl", {
        "event": "PREFLIGHT_PASS",
        "train_sha256": EXPECTED_TRAIN_SHA256,
        "resource_preflight": resources,
        "category_frequency_nonconstant_columns": True,
    })
    report = run_pilot(
        frame=frame, target=target, card=card, feature_names=FEATURE_NAMES,
        adapter=fit_predict, control_loader=control_loader,
        freeze_identity=identity, freeze_path=FREEZE_PATH,
        artifact_dir=ARTIFACT_DIR, event_path=REPORT_DIR / "attempts.jsonl",
        evaluation_views=views, threads=2, report_path=REPORT_DIR / "results.json",
        budget_ledger_path=REPORT_DIR.parent / "batch_ledger.jsonl",
        max_successful_fits=int(queue["execution_preset"]["max_successful_new_predictor_fits"]),
    )
    verification = verify_report(
        report=report, frame=frame, target=target, artifact_dir=ARTIFACT_DIR,
        freeze_signature=report["freeze_signature"], evaluation_views=views,
        prediction_reconstructors={"C07": reconstruct_c07},
    )
    gate_config = queue["evaluation"]["classification_gate"]
    subgroup_failures = _validate_primary_subgroup_gates(verification, gate_config)
    checks = {
        "primary_gain_at_least_0_0005": report["views"]["primary"]["gain_positive_is_better"] >= float(gate_config["pilot_primary_gain_min"]),
        "H12_harm_at_most_0_001": report["views"]["H12"]["gain_positive_is_better"] >= -float(gate_config["pilot_H12_harm_max"]),
        "W1_harm_at_most_0_005": report["views"]["W1"]["gain_positive_is_better"] >= -float(gate_config["individual_pilot_view_harm_max"]),
        "W4_harm_at_most_0_005": report["views"]["W4"]["gain_positive_is_better"] >= -float(gate_config["individual_pilot_view_harm_max"]),
        "supported_primary_subgroup_guards": len(subgroup_failures) == 0,
        "independent_metric_and_route_reconstruction": verification.get("status") == "PASS",
    }
    gate = {
        "card_id": "C07",
        "profile": card["evaluation"],
        "status": "ADVANCE" if all(checks.values()) else "RETIRE",
        "checks": {name: bool(value) for name, value in checks.items()},
        "thresholds": gate_config,
        "supported_subgroup_failures": subgroup_failures,
        "decision_basis": "Predeclared temporal pilot metrics and supported subgroup guards; no Test, SampleSubmission, or Kaggle result used.",
    }
    _write_or_match(REPORT_DIR / "verification.json", verification)
    _write_or_match(REPORT_DIR / "gate.json", gate)
    summary = (
        "# C07 Pilot Summary\n\n"
        f"- Status: **{gate['status']}**\n"
        f"- Primary Macro F1 gain: {report['views']['primary']['gain_positive_is_better']:+.6f} "
        f"({report['views']['primary']['rows']:,} rows).\n"
        f"- H12 Macro F1 gain: {report['views']['H12']['gain_positive_is_better']:+.6f} "
        f"({report['views']['H12']['rows']:,} rows).\n"
        f"- W1/W4 gains: {report['views']['W1']['gain_positive_is_better']:+.6f} / "
        f"{report['views']['W4']['gain_positive_is_better']:+.6f}.\n"
        "- Categories and six frequency columns were learned from each fit prefix only; no target rates were used.\n"
        "- Temporal Train windows are development evidence, not an untouched holdout. Test data was not read.\n"
    )
    summary_path = REPORT_DIR / "summary.md"
    if summary_path.exists():
        if summary_path.read_text(encoding="utf-8") != summary:
            raise BatchIntegrityError("Existing C07 summary differs; refusing overwrite")
    else:
        with summary_path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(summary)
    return {"pilot": report, "verification": verification, "gate": gate,
            "category_frequency_preflight": frequency_preflight, "resource_preflight": resources}


__all__ = ["reconstruct_c07", "run_card"]
