"""C08 preflight, frozen pilot, independent rate/route verification and gate."""
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
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.optimization_v3_batch1_c13 import (
    _control_input_files as _c13_control_input_files,
    _load_controls as _c13_load_controls,
    _official_train as _c13_official_train,
    _source_files as _c13_source_files,
    _validate_baseline_artifacts as _c13_validate_baseline_artifacts,
)
from scripts.optimization_v3_batch1_verify import verify_report
from src.models.classification_models import _TFIDF_PARAMS, _text_values
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
    temporal_rows,
    write_json_exclusive,
)
from src.models.optimization_v3_batch1_c08 import (
    BLEND_WEIGHT,
    CARD_ID,
    FALLBACK_CODES,
    FEATURE_NAMES,
    INITIAL_PRIOR,
    LOGIT_EPSILON,
    MINIMUM_PRIOR_SUPPORT,
    PILOT_CUTOFFS,
    SMOOTHING,
    TARGET_RATE_FIELDS,
    TARGET_RATE_FEATURE_NAMES,
    THRESHOLD,
    OrderedCategoryTargetRates,
    fit_predict,
)
from src.models.optimization_v3_sparse_classification import PARAMS as SPARSE_PARAMS
from src.models.regression_research_text import RegressionTextModel


QUEUE_PATH = ROOT / "experiments/optimization_v3/research_backlog/queue.json"
CARD_PATH = ROOT / "reports/optimization_v3/research_backlog/cards/C08.md"
EXPERIMENT_DIR = ROOT / "experiments/optimization_v3/batch1/C08"
RUN_DIR = ROOT / "experiments/runs/optimization_v3_batch1/C08"
REPORT_DIR = ROOT / "reports/optimization_v3/batch1/C08"
RECIPE_PATH = EXPERIMENT_DIR / "recipe.json"
FREEZE_PATH = EXPERIMENT_DIR / "freeze.json"
ENGINE_REPORT_PATH = RUN_DIR / "pilot_engine_report.json"
EVENT_PATH = RUN_DIR / "attempts.jsonl"
LEDGER_PATH = RUN_DIR.parent / "batch_ledger.jsonl"
PREFLIGHT_PATH = REPORT_DIR / "preflight.json"
RESULTS_PATH = REPORT_DIR / "results.json"
VERIFICATION_PATH = REPORT_DIR / "verification.json"
RATE_VERIFICATION_PATH = REPORT_DIR / "ordered_rate_verification.json"
GATE_PATH = REPORT_DIR / "gate.json"
SUMMARY_PATH = REPORT_DIR / "summary.json"
RUNTIME_NOTES_PATH = REPORT_DIR / "runtime_notes.json"
QUEUE_SHA256 = "822dcd20fa28fd0e9ddc841a8c3709b372dacaca305b94b39b11a9203418fec7"
CARD_SHA256 = "a92fd5a3e5e3d7cfe83473b2eeb0e61444dab93b348d2b9b751d8b3844cc2435"
TRAIN_SHA256 = "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c"
MANIFEST_SHA256 = "6e7c2cb50fe6d0ee4767a44c40f152b6538b933fab6cf7082659b16d791f7d1e"
EXPECTED_FIT_ROWS = {55: 12_020, 67: 29_705}
EXPECTED_SCORE_ROWS = {55: 17_685, 67: 20_903}
EXPECTED_VIEW_ROWS = {"W1": 4_114, "W4": 4_950, "primary": 9_064, "H12": 38_588}
EVIDENCE_PATHS = {
    "classification_diagnostics_audit": ROOT / "reports/optimization_v3/classification_diagnostics/audit.json",
    "supervised_pilot_verification": ROOT / "reports/optimization_v3/supervised_pilot/verification.json",
}
PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "threadpoolctl", "psutil"]
JOB_MEMORY_CAP_BYTES = 6 * 1024**3
FREE_RAM_RESERVE_BYTES = 2 * 1024**3
MINIMUM_FREE_BYTES = 4 * 1024**3
THREADS = 2
TOKENIZATION_ALLOWANCE_BYTES = 512 * 1024**2
MODEL_RUNTIME_ALLOWANCE_BYTES = 1024**3


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_or_match(path: Path, value: Mapping[str, Any]) -> None:
    if path.is_file():
        if _read_json(path) != dict(value):
            raise BatchIntegrityError(f"Existing C08 artifact differs: {path}")
    else:
        write_json_exclusive(path, dict(value))


def _source_commit_context() -> str:
    paths = [
        "scripts/optimization_v3_batch1_c08.py",
        "src/models/optimization_v3_batch1_c08.py",
        "tests/test_optimization_v3_batch1_c08.py",
    ]
    result = subprocess.run(["git", "log", "-1", "--format=%H", "--", *paths],
                            cwd=ROOT, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not result.stdout.strip():
        raise BatchIntegrityError("Cannot identify the committed C08 source revision")
    return result.stdout.strip()


def _validate_card(queue: Mapping[str, Any], card: Mapping[str, Any]) -> None:
    if file_sha256(QUEUE_PATH) != QUEUE_SHA256 or file_sha256(CARD_PATH) != CARD_SHA256:
        raise BatchIntegrityError("Canonical C08 queue/card hash changed")
    canonical = next((row for row in queue.get("cards", []) if row.get("id") == CARD_ID), None)
    if canonical is None or dict(canonical) != dict(card):
        raise BatchIntegrityError("C08 card differs from canonical queue")
    expected = {
        "id": "C08", "task": "classification", "family": "category_target_rates",
        "policy": "C_SPARSE_REPLACE", "pilot_predictor_fits": 2,
        "evaluation": "STANDARD", "condition": "No additional dependency",
    }
    for name, value in expected.items():
        if card.get(name) != value:
            raise BatchIntegrityError(f"Canonical C08 field changed: {name}")
    baseline = queue.get("baseline", {}).get("classification", {})
    if (baseline.get("Train_sha256") != TRAIN_SHA256
            or baseline.get("manifest_sha256") != MANIFEST_SHA256
            or baseline.get("manifest") != "reports/optimization_v3/kaggle/v3_cls_sparse25.json"
            or not np.isclose(float(baseline.get("primary", np.nan)), 0.8299435476765076, rtol=0, atol=1e-15)
            or not np.isclose(float(baseline.get("H12", np.nan)), 0.8029655654017788, rtol=0, atol=1e-15)):
        raise BatchIntegrityError("C08 frozen sparse25 baseline differs from queue")
    preset = queue.get("execution_preset", {})
    if (int(preset.get("concurrent_fit_jobs", -1)) != 1
            or int(preset.get("threads_per_job", -1)) != THREADS
            or float(preset.get("max_job_RSS_GiB", -1)) != 6.0
            or float(preset.get("minimum_free_RAM_GiB_before_start", -1)) != 4.0
            or int(preset.get("max_successful_new_predictor_fits", -1)) != 180):
        raise BatchIntegrityError("C08 training/resource/budget policy differs from queue")
    gate = queue.get("evaluation", {}).get("classification_gate", {})
    fixed = {
        "pilot_primary_gain_min": 0.0005,
        "pilot_H12_harm_max": 0.001,
        "individual_pilot_view_harm_max": 0.005,
        "supported_group_n_min": 200,
        "supported_group_F1_harm_max": 0.01,
        "positive_class_recall_harm_max": 0.02,
        "positive_and_negative_per_group_min": 30,
    }
    if any(float(gate.get(name, np.nan)) != value for name, value in fixed.items()):
        raise BatchIntegrityError("C08 classification gate differs from queue")
    if importlib.metadata.version("lightgbm") == "":
        raise BatchIntegrityError("C08 declared sparse LightGBM dependency is not installed")


def _source_files() -> list[Path]:
    sources = set(_c13_source_files())
    sources.update({
        Path(__file__),
        ROOT / "src/models/optimization_v3_batch1_c08.py",
        ROOT / "src/models/optimization_v3_sparse_classification.py",
        ROOT / "tests/test_optimization_v3_batch1_c08.py",
    })
    return sorted(sources, key=str)


def _input_files(train_path: Path, views: Mapping[str, Any]) -> dict[str, Path]:
    files = _c13_control_input_files(train_path, CARD_PATH, views)
    files.update(EVIDENCE_PATHS)
    return files


def _category_key(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    # Preserve the frozen C07 normalization rule: blank strings are valid keys.
    return str(value).strip()


def _manual_ordered_rate_arrays(
    fit_features: pd.DataFrame, fit_target: np.ndarray,
    score_features: pd.DataFrame | None = None,
) -> dict[str, np.ndarray]:
    """Independent Counter-style reconstruction; does not call the model encoder."""
    fit = fit_features.loc[:, list(FEATURE_NAMES)].copy()
    y = np.asarray(fit_target, dtype=np.float64)
    cycles = pd.to_numeric(fit["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    if y.shape != (len(fit),) or not np.isfinite(y).all() or not np.isin(y, (0.0, 1.0)).all():
        raise BatchIntegrityError("Manual C08 target-rate verifier received invalid fit labels")
    train_logits = np.full((len(fit), len(TARGET_RATE_FIELDS)), 0.0, dtype=np.float64)
    train_support = np.zeros((len(fit), len(TARGET_RATE_FIELDS)), dtype=np.int32)
    train_fallback = np.full((len(fit), len(TARGET_RATE_FIELDS)), FALLBACK_CODES["below_minimum_support"], dtype=np.uint8)
    counts: dict[str, dict[str, int]] = {field: {} for field in TARGET_RATE_FIELDS}
    sums: dict[str, dict[str, float]] = {field: {} for field in TARGET_RATE_FIELDS}
    fit_keys = {field: [_category_key(value) for value in fit[field].to_numpy(dtype=object)]
                for field in TARGET_RATE_FIELDS}

    def encode(field: str, key: str | None) -> tuple[float, int, int]:
        if key is None:
            code = FALLBACK_CODES["missing_category"]
            count = 0
            total = 0.0
        else:
            count = int(counts[field].get(key, 0))
            total = float(sums[field].get(key, 0.0))
            code = (FALLBACK_CODES["no_prior_category_history"] if count == 0 else
                    FALLBACK_CODES["below_minimum_support"] if count < MINIMUM_PRIOR_SUPPORT else
                    FALLBACK_CODES["supported"])
        rate = INITIAL_PRIOR if count < MINIMUM_PRIOR_SUPPORT else (
            total + SMOOTHING * INITIAL_PRIOR
        ) / (count + SMOOTHING)
        clipped = float(np.clip(rate, LOGIT_EPSILON, 1.0 - LOGIT_EPSILON))
        return float(np.log(clipped / (1.0 - clipped))), count, code

    for cycle in sorted(np.unique(cycles).tolist()):
        rows = np.flatnonzero(cycles == cycle)
        for j, field in enumerate(TARGET_RATE_FIELDS):
            for row in rows:
                value, count, code = encode(field, fit_keys[field][row])
                train_logits[row, j] = value
                train_support[row, j] = count
                train_fallback[row, j] = code
        # All rate values for a cycle are complete; only now expose its labels.
        for field in TARGET_RATE_FIELDS:
            groups: dict[str, list[int]] = {}
            for row in rows:
                key = fit_keys[field][row]
                if key is not None:
                    groups.setdefault(key, []).append(int(row))
            for key, group_rows in groups.items():
                counts[field][key] = counts[field].get(key, 0) + len(group_rows)
                sums[field][key] = sums[field].get(key, 0.0) + float(y[group_rows].sum(dtype=np.float64))

    output = {
        "train_logits": train_logits,
        "train_support": train_support,
        "train_fallback": train_fallback,
    }
    if score_features is not None:
        score = score_features.loc[:, list(FEATURE_NAMES)]
        n_score = len(score)
        score_logits = np.zeros((n_score, len(TARGET_RATE_FIELDS)), dtype=np.float64)
        score_support = np.zeros((n_score, len(TARGET_RATE_FIELDS)), dtype=np.int32)
        score_fallback = np.full((n_score, len(TARGET_RATE_FIELDS)), FALLBACK_CODES["below_minimum_support"], dtype=np.uint8)
        for j, field in enumerate(TARGET_RATE_FIELDS):
            for row, value_key in enumerate(_category_key(value) for value in score[field].to_numpy(dtype=object)):
                value, count, code = encode(field, value_key)
                score_logits[row, j] = value
                score_support[row, j] = count
                score_fallback[row, j] = code
        output.update({
            "score_logits": score_logits,
            "score_support": score_support,
            "score_fallback": score_fallback,
        })
    return output


def reconstruct_c08(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild the sparse-replace route independently from raw candidate outputs."""
    raw = np.asarray(arrays["raw_candidate"], dtype=np.float64)
    missing = np.asarray(arrays["control__stream_missing"], dtype=bool)
    c2b = np.asarray(arrays["control__c2b_probability"], dtype=np.float64)
    cb = np.asarray(arrays["control__cb_probability"], dtype=np.float64)
    old_sparse = np.asarray(arrays["control__baseline_sparse_raw"], dtype=np.float64)
    if len({len(raw), len(missing), len(c2b), len(cb), len(old_sparse)}) != 1:
        raise BatchIntegrityError("C08 saved route inputs have inconsistent lengths")
    candidate = np.where(missing, cb, 0.75 * c2b + 0.25 * raw)
    baseline = np.where(missing, cb, 0.75 * c2b + 0.25 * old_sparse)
    return candidate, baseline


def _memory_estimate(fit_features: pd.DataFrame, score_features: pd.DataFrame,
                     available_bytes: int | None = None) -> dict[str, Any]:
    docs, fit_missing = _text_values(fit_features["signal_stream"])
    score_docs, score_missing = _text_values(score_features["signal_stream"])
    vectorizer = TfidfVectorizer(**dict(_TFIDF_PARAMS))
    text = vectorizer.fit_transform(docs).tocsr()
    if text.shape[1] < 1:
        raise ValueError("SKIPPED_SUPPORT: fit-only TF-IDF has no columns")
    score_text = vectorizer.transform(score_docs).tocsr()
    (structured, processor, scaler, encoder, numeric_positions,
     structured_state) = RegressionTextModel._structured_fit(fit_features)
    score_structured, _score_unknown = RegressionTextModel._structured_transform(
        score_features, processor, scaler, encoder, numeric_positions
    )
    structured_design = sparse.hstack((structured, sparse.csr_matrix((len(fit_features), 3))), format="csr")
    fit_design = sparse.hstack((
        structured_design, text, sparse.csr_matrix(fit_missing.astype(np.float64).reshape(-1, 1))
    ), format="csr")
    score_design = sparse.hstack((
        score_structured, sparse.csr_matrix((len(score_features), 3)), score_text,
        sparse.csr_matrix(score_missing.astype(np.float64).reshape(-1, 1)),
    ), format="csr")
    score_rows = len(score_features)
    csr_bytes = int(fit_design.data.nbytes + fit_design.indices.nbytes + fit_design.indptr.nbytes)
    score_csr_bytes = int(score_design.data.nbytes + score_design.indices.nbytes + score_design.indptr.nbytes)
    # Both fit and predict paths hstack CSR blocks; bound temporary copies from
    # sparse assembly instead of treating the vocabulary-wide score matrix as dense.
    score_scratch_bytes = int(5 * score_csr_bytes)
    rate_arrays_bytes = int((len(fit_features) + score_rows) * 3 * (8 + 4 + 1))
    estimated_peak = int(
        TOKENIZATION_ALLOWANCE_BYTES + MODEL_RUNTIME_ALLOWANCE_BYTES
        + 5 * csr_bytes + score_scratch_bytes + 2 * rate_arrays_bytes
    )
    available = int(psutil.virtual_memory().available if available_bytes is None else available_bytes)
    allowed = min(JOB_MEMORY_CAP_BYTES, available - FREE_RAM_RESERVE_BYTES)
    result = {
        "status": "PASS" if available >= MINIMUM_FREE_BYTES and allowed > 0 and estimated_peak <= allowed else "BLOCKED_RESOURCE",
        "fit_rows": int(len(fit_features)),
        "score_rows": int(score_rows),
        "tfidf_vocabulary_width": int(text.shape[1]),
        "tfidf_nnz": int(text.nnz),
        "score_tfidf_nnz": int(score_text.nnz),
        "sparse_fit_design_width": int(fit_design.shape[1]),
        "fit_design_nnz": int(fit_design.nnz),
        "fit_sparse_design_bytes": csr_bytes,
        "score_design_nnz": int(score_design.nnz),
        "score_sparse_design_bytes": score_csr_bytes,
        "score_scratch_bytes_upper_bound": score_scratch_bytes,
        "ordered_rate_arrays_bytes_upper_bound": rate_arrays_bytes,
        "tokenization_allowance_bytes": TOKENIZATION_ALLOWANCE_BYTES,
        "model_runtime_allowance_bytes": MODEL_RUNTIME_ALLOWANCE_BYTES,
        "estimated_additional_peak_bytes": estimated_peak,
        "estimate_is_observed_rss": False,
        "available_ram_bytes": available,
        "minimum_free_ram_bytes": MINIMUM_FREE_BYTES,
        "reserve_bytes": FREE_RAM_RESERVE_BYTES,
        "job_cap_bytes": JOB_MEMORY_CAP_BYTES,
        "allowed_bytes": int(max(0, allowed)),
        "threads": THREADS,
        "concurrent_training_jobs": 1,
        "structured_preprocessing": structured_state,
    }
    del text, score_text, structured_design, fit_design, score_design, structured, score_structured, vectorizer
    gc.collect()
    return result


def _rate_preflight(fit_features: pd.DataFrame, fit_target: np.ndarray,
                    score_features: pd.DataFrame, cutoff: int) -> dict[str, Any]:
    encoder = OrderedCategoryTargetRates()
    train_logits, train_support, train_fallback = encoder.fit_transform(fit_features, fit_target)
    score_logits, score_support, score_fallback, score_diagnostics = encoder.transform(score_features)
    manual = _manual_ordered_rate_arrays(fit_features, fit_target, score_features)
    manual_training_match = (
        np.array_equal(train_logits, manual["train_logits"])
        and np.array_equal(train_support, manual["train_support"])
        and np.array_equal(train_fallback, manual["train_fallback"])
    )
    manual_score_match = (
        np.array_equal(score_logits, manual["score_logits"])
        and np.array_equal(score_support, manual["score_support"])
        and np.array_equal(score_fallback, manual["score_fallback"])
    )
    cycles = pd.to_numeric(fit_features["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    perturb_cycle = int(sorted(np.unique(cycles).tolist())[len(np.unique(cycles)) // 2])
    perturbed_target = np.asarray(fit_target, dtype=np.int8).copy()
    perturb_mask = cycles == perturb_cycle
    perturbed_target[perturb_mask] = 1 - perturbed_target[perturb_mask]
    altered, _altered_support, _altered_fallback = OrderedCategoryTargetRates().fit_transform(
        fit_features, perturbed_target
    )
    own_or_past_mask = cycles <= perturb_cycle
    perturbation_pass = bool(np.array_equal(train_logits[own_or_past_mask], altered[own_or_past_mask]))
    feature_stats: dict[str, Any] = {}
    varied_all = True
    for index, field in enumerate(TARGET_RATE_FIELDS):
        varied = bool(np.ptp(train_logits[:, index]) > 0.0)
        varied_all &= varied
        feature_stats[field] = {
            "nonconstant_on_fit_rows": varied,
            "fit_rows_with_support_ge_30": int(np.sum(train_fallback[:, index] == FALLBACK_CODES["supported"])),
            "fit_rows_below_support_or_no_history": int(np.sum(np.isin(train_fallback[:, index], [FALLBACK_CODES["below_minimum_support"], FALLBACK_CODES["no_prior_category_history"]]))),
            "fit_rows_missing_category": int(np.sum(train_fallback[:, index] == FALLBACK_CODES["missing_category"])),
            "score_rows_with_support_ge_30": int(np.sum(score_fallback[:, index] == FALLBACK_CODES["supported"])),
            "score_rows_below_support_or_no_history": int(np.sum(np.isin(score_fallback[:, index], [FALLBACK_CODES["below_minimum_support"], FALLBACK_CODES["no_prior_category_history"]]))),
            "score_rows_missing_category": int(np.sum(score_fallback[:, index] == FALLBACK_CODES["missing_category"])),
            "fit_logit_min": float(np.min(train_logits[:, index])),
            "fit_logit_max": float(np.max(train_logits[:, index])),
            "fit_logit_std": float(np.std(train_logits[:, index])),
        }
    return {
        "cutoff": int(cutoff),
        "status": (
            "FAIL_PREFLIGHT" if not (manual_training_match and manual_score_match and perturbation_pass)
            else "PASS" if varied_all
            else "SKIPPED_SUPPORT"
        ),
        "manual_training_reconstruction_pass": bool(manual_training_match),
        "manual_score_reconstruction_pass": bool(manual_score_match),
        "same_cycle_label_perturbation_pass": perturbation_pass,
        "perturbed_cycle": perturb_cycle,
        "perturbed_rows": int(perturb_mask.sum()),
        "own_and_past_feature_rows_unchanged": int(own_or_past_mask.sum()),
        "variation_required_for_each_feature": True,
        "feature_support": feature_stats,
        "training_feature_matrix_sha256": __import__("hashlib").sha256(train_logits.tobytes(order="C")).hexdigest(),
        "score_feature_matrix_sha256": __import__("hashlib").sha256(score_logits.tobytes(order="C")).hexdigest(),
        "encoder_audit": encoder.audit_state(),
        "score_diagnostics": score_diagnostics,
    }


def _control_metrics(target: np.ndarray, cycles: np.ndarray,
                     views: Mapping[str, Any], controls: Mapping[int, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, segments in views.items():
        row_parts: list[np.ndarray] = []
        prediction_parts: list[np.ndarray] = []
        for cutoff, low, high in segments:
            saved = controls[int(cutoff)]
            rows = np.asarray(saved["rows"], dtype=np.int64)
            mask = (cycles[rows] >= int(low)) & (cycles[rows] <= int(high))
            row_parts.append(rows[mask])
            prediction_parts.append(np.asarray(saved["baseline_prediction"])[mask])
        rows = np.concatenate(row_parts)
        if len(np.unique(rows)) != len(rows) or len(rows) != EXPECTED_VIEW_ROWS[name]:
            raise BatchIntegrityError(f"C08 row-matched control denominator differs for {name}")
        output[name] = {
            "rows": int(len(rows)),
            "macro_f1": float(metric("classification", target[rows], np.concatenate(prediction_parts))["macro_f1"]),
        }
    return output


def _recipe_payload(card: Mapping[str, Any], views: Mapping[str, Any],
                    controls: Mapping[str, Any], queue: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "card_id": CARD_ID,
        "queue_recipe": {key: card[key] for key in (
            "id", "task", "name", "family", "priority", "change", "policy",
            "pilot_predictor_fits", "evaluation", "condition",
        )},
        "model": "original sparse LightGBM classifier with exactly three ordered category target-rate logit columns appended to expanded structured + TF-IDF design",
        "lightgbm_parameters": dict(SPARSE_PARAMS),
        "lightgbm_version": importlib.metadata.version("lightgbm"),
        "target_rate": {
            "fields": list(TARGET_RATE_FIELDS),
            "support_minimum": MINIMUM_PRIOR_SUPPORT,
            "smoothing": SMOOTHING,
            "prior": INITIAL_PRIOR,
            "prior_policy": "fixed 0.5; used in the first cycle and for every category with fewer than 30 earlier-cycle rows, missing categories, or unseen categories",
            "rate_formula_when_supported": "(earlier_category_positive_sum + 30 * 0.5) / (earlier_category_count + 30)",
            "logit_clip_epsilon": LOGIT_EPSILON,
            "logit": "log(p/(1-p)) after clipping p to [1e-6, 1-1e-6]",
            "cycle_rule": "encode every row of a cycle before updating category counts/sums with any labels from that cycle",
            "outer_scoring": "all labels through cutoff are eligible for score rows after cutoff; no score-cycle labels are used",
            "missing_category_normalization": "C07 convention: pd.isna maps to None; otherwise str(value).strip(); blank strings remain categories",
            "fallback_code_mapping": {str(value): key for key, value in FALLBACK_CODES.items()},
        },
        "text": {
            "vectorizer": "sklearn.feature_extraction.text.TfidfVectorizer",
            "parameters": {key: list(value) if isinstance(value, tuple)
                           else value.__name__ if isinstance(value, type)
                           else value.item() if isinstance(value, np.generic) else value
                           for key, value in _TFIDF_PARAMS.items()},
            "missing_stream": "empty document plus explicit binary missing indicator",
        },
        "structured": "RegressionTextModel expanded structured fit-only preprocessing; no target or ID inputs",
        "feature_allowlist": list(FEATURE_NAMES),
        "fit_population": "all official Classification Train rows cycle 47..cutoff, including missing streams, as in the frozen sparse expert",
        "scoring_population": "STANDARD cutoffs 55/67; rows joined by Train position and record_id",
        "prediction_policy": {
            "present_stream": ".75 frozen C2_B probability + .25 new C08 sparse expert probability",
            "missing_stream": "frozen CatBoost missing-stream probability, unchanged",
            "baseline": "frozen sparse25 route: .75 C2_B + .25 old sparse binary on present; frozen CB on missing",
            "threshold": THRESHOLD,
        },
        "row_matched_baseline_metrics": controls,
        "views": views,
        "expected_rows": {"fit_by_cutoff": {str(k): v for k, v in EXPECTED_FIT_ROWS.items()},
                          "score_by_cutoff": {str(k): v for k, v in EXPECTED_SCORE_ROWS.items()},
                          "views": EXPECTED_VIEW_ROWS},
        "gate": dict(queue["evaluation"]["classification_gate"]),
        "fit_protocol": "one sparse LightGBM predictor per cutoff; no validation labels, early stopping, threshold search, or extra fit",
        "resource_policy": {"threads": THREADS, "concurrent_fit_jobs": 1,
                            "job_memory_cap_bytes": JOB_MEMORY_CAP_BYTES,
                            "free_ram_reserve_bytes": FREE_RAM_RESERVE_BYTES,
                            "minimum_free_ram_bytes": MINIMUM_FREE_BYTES},
    }


def _budget_preflight() -> dict[str, Any]:
    reserved: dict[str, int] = {}
    if LEDGER_PATH.is_file():
        for line_number, line in enumerate(LEDGER_PATH.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BatchIntegrityError(f"Invalid fit ledger line {line_number}") from exc
            if event.get("event") == "BUDGET_RESERVED":
                key = str(event["budget_key"])
                count = int(event["predictor_fits"])
                if key in reserved and reserved[key] != count:
                    raise BatchIntegrityError(f"Conflicting reservation for {key}")
                reserved[key] = count
    current = reserved.get("C08:PILOT")
    total_before = sum(reserved.values())
    remaining = 180 - total_before
    enough = (current == 2) or (current is None and remaining >= 2)
    return {
        "status": "PASS" if enough else "SKIP_BUDGET",
        "successful_fit_cap": 180,
        "already_reserved": total_before,
        "remaining_before_C08": remaining,
        "C08_reserved_fits": current if current is not None else 2,
        "ledger_path": str(LEDGER_PATH.relative_to(ROOT)),
    }


def _prepare(card: Mapping[str, Any], queue: Mapping[str, Any]) -> dict[str, Any]:
    _validate_card(queue, card)
    baseline_evidence = _c13_validate_baseline_artifacts(queue)
    train_path, frame, target, cycles = _c13_official_train()
    if file_sha256(train_path) != TRAIN_SHA256:
        raise BatchIntegrityError("Official Classification Train SHA256 differs from C08 pin")
    if len(frame) != 50_608 or frame.record_id.isna().any() or frame.record_id.astype(str).duplicated().any():
        raise BatchIntegrityError("C08 official Train row/record_id integrity failed")
    if (not np.isin(target, [0, 1]).all() or not np.array_equal(np.unique(target), np.array([0, 1]))
            or int(cycles.min()) != 47 or int(cycles.max()) != 79):
        raise BatchIntegrityError("C08 official target/cycle support differs from frozen source")
    views = evaluation_views_for_card(queue, CARD_ID)
    expected_views = {
        "W1": [(55, 56, 59)], "W4": [(67, 68, 71)],
        "primary": [(55, 56, 59), (67, 68, 71)],
        "H12": [(55, 56, 67), (67, 68, 79)],
    }
    if views != expected_views:
        raise BatchIntegrityError("C08 STANDARD view mapping differs from canonical card")
    control_loader = _c13_load_controls(frame, target)
    controls_by_cutoff: dict[int, Any] = {}
    score_rows_by_cutoff: dict[int, np.ndarray] = {}
    rate_preflight: dict[str, Any] = {}
    memory: dict[str, Any] = {}
    fit_class_counts: dict[str, dict[str, int]] = {}
    support_skip = False
    integrity_failure = False
    for cutoff in PILOT_CUTOFFS:
        fit_rows, score_rows = temporal_rows(frame, cutoff, horizon=12)
        if len(fit_rows) != EXPECTED_FIT_ROWS[cutoff] or len(score_rows) != EXPECTED_SCORE_ROWS[cutoff]:
            raise BatchIntegrityError(f"C08 fit or score denominator differs at {cutoff}")
        fit_features = frame.iloc[fit_rows].loc[:, list(FEATURE_NAMES)].copy()
        score_features = frame.iloc[score_rows].loc[:, list(FEATURE_NAMES)].copy()
        y_fit = target[fit_rows]
        fit_class_counts[str(cutoff)] = {str(int(k)): int(v) for k, v in zip(*np.unique(y_fit, return_counts=True))}
        rate_preflight[str(cutoff)] = _rate_preflight(fit_features, y_fit, score_features, cutoff)
        if rate_preflight[str(cutoff)]["status"] == "FAIL_PREFLIGHT":
            integrity_failure = True
        elif rate_preflight[str(cutoff)]["status"] == "SKIPPED_SUPPORT":
            support_skip = True
        memory[str(cutoff)] = _memory_estimate(fit_features, score_features)
        control = control_loader(cutoff, score_rows)
        controls_by_cutoff[cutoff] = {"rows": score_rows,
                                     "baseline_prediction": np.asarray(control["baseline_prediction"], dtype=np.float64)}
        score_rows_by_cutoff[cutoff] = score_rows
    row_matched_metrics = _control_metrics(target, cycles, views, controls_by_cutoff)
    budget = _budget_preflight()
    recipe = _recipe_payload(card, views, row_matched_metrics, queue)
    input_files = _input_files(train_path, views)
    source_files = _source_files()
    identity = make_freeze_identity(
        card=card, feature_names=FEATURE_NAMES, input_files=input_files,
        source_files=source_files, evaluation_views=views, packages=PACKAGES,
    )
    identity.update({
        "recipe": recipe,
        "recipe_artifact_sha256": json_sha256(recipe),
        "accepted_baseline_evidence": baseline_evidence,
        "source_commit_context": _source_commit_context(),
        "expected_fit_rows": {str(k): v for k, v in EXPECTED_FIT_ROWS.items()},
        "expected_score_rows": {str(k): v for k, v in EXPECTED_SCORE_ROWS.items()},
        "expected_view_rows": EXPECTED_VIEW_ROWS,
        "ordered_target_rate_preflight": rate_preflight,
        "resource_preflight": memory,
    })
    memory_pass = all(record["status"] == "PASS" for record in memory.values())
    if integrity_failure:
        status = "BLOCKED_PREFLIGHT_INTEGRITY"
    elif support_skip:
        status = "SKIPPED_SUPPORT"
    elif not memory_pass or budget["status"] != "PASS":
        status = "BLOCKED_RESOURCE_OR_BUDGET"
    else:
        status = "PREFLIGHT_PASS"
    public = {
        "card_id": CARD_ID,
        "status": status,
        "scope": "official Classification Train and frozen sparse25 controls; no candidate scoring, Test/Sample or Kaggle access",
        "official_train_sha256": TRAIN_SHA256,
        "input_sha256": {name: file_sha256(path) for name, path in sorted(input_files.items())},
        "source_sha256": {str(Path(path).relative_to(ROOT)): file_sha256(path) for path in source_files},
        "freeze_identity_sha256": json_sha256(identity),
        "baseline_evidence": baseline_evidence,
        "row_matched_control_metrics": row_matched_metrics,
        "fit_rows": {str(k): v for k, v in EXPECTED_FIT_ROWS.items()},
        "score_rows": {str(k): v for k, v in EXPECTED_SCORE_ROWS.items()},
        "view_rows": EXPECTED_VIEW_ROWS,
        "fit_class_counts": fit_class_counts,
        "ordered_target_rate_preflight": rate_preflight,
        "memory_preflight": memory,
        "budget_preflight": budget,
        "lightgbm_version": importlib.metadata.version("lightgbm"),
        "python_version": platform.python_version(),
        "test_or_sample_access": False,
        "kaggle_access": False,
        "candidate_fit_or_metric_computed": False,
        "artifacts_written": False,
    }
    return {
        "public": public, "identity": identity, "recipe": recipe,
        "frame": frame, "target": target, "cycles": cycles, "views": views,
        "control_loader": control_loader,
        "score_rows_by_cutoff": score_rows_by_cutoff,
    }


def _resource_guarded_adapter(fit_features: pd.DataFrame, fit_target: np.ndarray,
                              score_features: pd.DataFrame, *, controls: dict[str, Any],
                              card_id: str, cutoff: int) -> dict[str, Any]:
    estimate = _memory_estimate(fit_features, score_features)
    if estimate["status"] != "PASS":
        raise BatchIntegrityError(f"C08 fit-time resource gate failed at cutoff {cutoff}")
    result = fit_predict(fit_features, fit_target, score_features,
                         controls=controls, card_id=card_id, cutoff=cutoff)
    result["fit_metadata"]["resource_preflight_at_fit"] = {
        key: estimate[key] for key in (
            "estimated_additional_peak_bytes", "available_ram_bytes", "allowed_bytes",
            "tfidf_vocabulary_width", "tfidf_nnz", "sparse_fit_design_width",
            "fit_design_nnz", "fit_sparse_design_bytes", "score_tfidf_nnz",
            "score_design_nnz", "score_sparse_design_bytes", "score_scratch_bytes_upper_bound",
            "tokenization_allowance_bytes", "model_runtime_allowance_bytes",
        )
    }
    return result


def _verify_saved_rates(context: Mapping[str, Any], report: Mapping[str, Any]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for cutoff in PILOT_CUTOFFS:
        score_rows = context["score_rows_by_cutoff"][cutoff]
        fit_rows = np.flatnonzero(context["cycles"] <= cutoff)
        identity = {
            "freeze_signature": report["freeze_signature"],
            "card_id": CARD_ID,
            "cutoff": cutoff,
            "score_rows": score_rows.tolist(),
        }
        arrays, sidecar = read_bundle(RUN_DIR / f"C08_fit{cutoff}.npz", identity)
        manual = _manual_ordered_rate_arrays(
            context["frame"].iloc[fit_rows], context["target"][fit_rows],
            context["frame"].iloc[score_rows],
        )
        actual_logits = np.asarray(arrays["audit__ordered_rate_logits"], dtype=np.float64)
        actual_support = np.asarray(arrays["audit__ordered_rate_support"], dtype=np.int32)
        actual_fallback = np.asarray(arrays["audit__ordered_rate_fallback"], dtype=np.uint8)
        checks = {
            "score_logits_match_manual": np.array_equal(actual_logits, manual["score_logits"]),
            "score_support_match_manual": np.array_equal(actual_support, manual["score_support"]),
            "score_fallback_match_manual": np.array_equal(actual_fallback, manual["score_fallback"]),
            "fit_cycle_min_47": int(sidecar["metadata"]["state"].get("fit_cycle_min", -1)) == 47,
            "fit_cycle_max_cutoff": int(sidecar["metadata"]["state"].get("fit_cycle_max", -1)) == cutoff,
            "exact_three_target_rate_columns": actual_logits.shape == (len(score_rows), 3),
            "finite_logits": bool(np.isfinite(actual_logits).all()),
            "support_counts_nonnegative": bool(np.all(actual_support >= 0)),
            "known_fallback_codes_only": bool(np.isin(actual_fallback, list(FALLBACK_CODES.values())).all()),
        }
        results[str(cutoff)] = {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "score_rows": int(len(score_rows)),
            "score_rate_logits_sha256": __import__("hashlib").sha256(actual_logits.tobytes(order="C")).hexdigest(),
            "score_support_sha256": __import__("hashlib").sha256(actual_support.tobytes(order="C")).hexdigest(),
            "score_fallback_sha256": __import__("hashlib").sha256(actual_fallback.tobytes(order="C")).hexdigest(),
            "supported_rows_by_field": {
                field: int(np.sum(actual_fallback[:, index] == FALLBACK_CODES["supported"]))
                for index, field in enumerate(TARGET_RATE_FIELDS)
            },
            "below_support_or_unseen_rows_by_field": {
                field: int(np.sum(np.isin(actual_fallback[:, index], [
                    FALLBACK_CODES["below_minimum_support"], FALLBACK_CODES["no_prior_category_history"]
                ]))) for index, field in enumerate(TARGET_RATE_FIELDS)
            },
            "missing_category_rows_by_field": {
                field: int(np.sum(actual_fallback[:, index] == FALLBACK_CODES["missing_category"]))
                for index, field in enumerate(TARGET_RATE_FIELDS)
            },
        }
    return {
        "status": "PASS" if all(item["status"] == "PASS" for item in results.values()) else "FAIL",
        "method": "independent Counter-style chronological reconstruction from official Train fit labels; no model encoder calls",
        "cutoffs": results,
    }


def _fit_accounting() -> dict[str, Any]:
    events = [json.loads(line) for line in EVENT_PATH.read_text(encoding="utf-8").splitlines() if line.strip()] if EVENT_PATH.is_file() else []
    saved = []
    for cutoff in PILOT_CUTOFFS:
        path = RUN_DIR / f"C08_fit{cutoff}.json"
        if path.is_file():
            saved.append(_read_json(path)["metadata"])
    event_names = ("FIT_STARTED", "FIT_SAVED", "FIT_FAILED", "RESUMED_VERIFIED")
    attempts_by_cutoff: dict[str, dict[str, int]] = {
        str(cutoff): {name: 0 for name in event_names} for cutoff in PILOT_CUTOFFS
    }
    attempt_events_valid = True
    for event in events:
        cutoff = str(event.get("cutoff", ""))
        name = str(event.get("event", ""))
        if cutoff not in attempts_by_cutoff or name not in attempts_by_cutoff[cutoff]:
            attempt_events_valid = False
            continue
        attempts_by_cutoff[cutoff][name] += 1
    for counts in attempts_by_cutoff.values():
        started = counts["FIT_STARTED"]
        saved_or_failed = counts["FIT_SAVED"] + counts["FIT_FAILED"]
        resumed = counts["RESUMED_VERIFIED"]
        # A verified resumed bundle can account for an interrupted attempt whose
        # atomic artifact completed before its final event was appended.
        if saved_or_failed > started or started > saved_or_failed + resumed:
            attempt_events_valid = False
    return {
        "expected_predictor_fits": 2,
        "successful_saved_predictor_fits": sum(int(item["fit_predictor_count"]) for item in saved),
        "fit_seconds_total": float(sum(float(item["fit_seconds"]) for item in saved)),
        "fit_origins_saved": [int(item["cutoff"]) for item in saved],
        "fit_started_events": sum(item.get("event") == "FIT_STARTED" for item in events),
        "fit_saved_events": sum(item.get("event") == "FIT_SAVED" for item in events),
        "fit_failed_events": sum(item.get("event") == "FIT_FAILED" for item in events),
        "resumed_verified_events": sum(item.get("event") == "RESUMED_VERIFIED" for item in events),
        "attempts_by_cutoff": attempts_by_cutoff,
        "attempt_accounting_complete": attempt_events_valid,
        "technical_retries": sum(item.get("event") == "FIT_FAILED" for item in events),
        "control_refits": 0,
        "peak_rss_sampled": False,
    }


def _gate(report: Mapping[str, Any], verification: Mapping[str, Any]) -> dict[str, Any]:
    views = report["views"]
    primary = float(views["primary"]["gain_positive_is_better"])
    h12 = float(views["H12"]["gain_positive_is_better"])
    w1 = float(views["W1"]["gain_positive_is_better"])
    w4 = float(views["W4"]["gain_positive_is_better"])
    subgroup_failures = verification["views"]["primary"]["subgroup_diagnostics"]["supported_guard_failures"]
    checks = {
        "independent_route_and_metric_verification": verification.get("status") == "PASS",
        "ordered_target_rate_independent_reconstruction": verification.get("ordered_target_rate_status") == "PASS",
        "primary_gain_min_0_0005": primary >= 0.0005,
        "H12_harm_max_0_001": h12 >= -0.001,
        "W1_harm_max_0_005": w1 >= -0.005,
        "W4_harm_max_0_005": w4 >= -0.005,
        "supported_subgroup_guards": len(subgroup_failures) == 0,
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
        "supported_subgroup_guard_failures": subgroup_failures,
        "interpretation": "Advance only if all frozen temporal pilot, subgroup, and independent encoding checks pass.",
    }


def _reconcile_budget(result: Mapping[str, Any], rate_verification: Mapping[str, Any],
                      gate: Mapping[str, Any]) -> None:
    accounting = result["fit_accounting"]
    if (accounting["successful_saved_predictor_fits"] != 2
            or sorted(accounting["fit_origins_saved"]) != list(PILOT_CUTOFFS)
            or not accounting["attempt_accounting_complete"]
            or rate_verification["status"] != "PASS"):
        raise BatchIntegrityError("C08 fit/verification accounting is incomplete")
    reconciliation = {
        "event": "FIT_BUDGET_RECONCILED",
        "card_id": CARD_ID,
        "budget_key": "C08:PILOT",
        "attempt_accounting_complete": accounting["attempt_accounting_complete"],
        "fit_started_events": accounting["fit_started_events"],
        "fit_saved_events": accounting["fit_saved_events"],
        "fit_failed_events": accounting["fit_failed_events"],
        "resumed_verified_events": accounting["resumed_verified_events"],
        "technical_retries": accounting["technical_retries"],
        "reserved_predictor_fits": 2,
        "successful_predictor_fits": result["fit_accounting"]["successful_saved_predictor_fits"],
        "decision": gate["status"],
        "independent_verification_status": "PASS",
        "freeze_signature": result["freeze_signature"],
        "attempts_sha256": file_sha256(EVENT_PATH),
        "results_sha256": file_sha256(RESULTS_PATH),
        "verification_sha256": file_sha256(VERIFICATION_PATH),
        "rate_verification_sha256": file_sha256(RATE_VERIFICATION_PATH),
        "gate_sha256": file_sha256(GATE_PATH),
    }
    existing: list[dict[str, Any]] = []
    if LEDGER_PATH.is_file():
        existing = [json.loads(line) for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    matches = [event for event in existing if event.get("event") == reconciliation["event"] and event.get("card_id") == CARD_ID]
    if matches:
        if len(matches) != 1 or any(matches[0].get(key) != value for key, value in reconciliation.items() if key != "event"):
            raise BatchIntegrityError("Existing C08 budget reconciliation differs")
        return
    append_event(LEDGER_PATH, {**reconciliation, "at_utc": pd.Timestamp.now(tz="UTC").isoformat()})


def run_card(card: Mapping[str, Any], queue: Mapping[str, Any], *,
             preflight_only: bool = False) -> dict[str, Any]:
    context = _prepare(card, queue)
    if preflight_only:
        return context["public"]
    if context["public"]["status"] != "PREFLIGHT_PASS":
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        _write_or_match(PREFLIGHT_PATH, context["public"])
        skipped = {
            "card_id": CARD_ID, "status": context["public"]["status"],
            "verdict": context["public"]["status"], "preflight": context["public"],
            "candidate_fit_or_metric_computed": False, "test_or_sample_access": False,
            "kaggle_access": False,
        }
        _write_or_match(RESULTS_PATH, skipped)
        return skipped
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    _write_or_match(RECIPE_PATH, context["recipe"])
    signature = json_sha256(context["identity"])
    if ENGINE_REPORT_PATH.is_file():
        if not FREEZE_PATH.is_file():
            raise BatchIntegrityError("C08 engine report exists without immutable freeze")
        frozen = _read_json(FREEZE_PATH)
        report = _read_json(ENGINE_REPORT_PATH)
        if (frozen.get("identity") != context["identity"] or frozen.get("signature") != signature
                or report.get("freeze_signature") != signature):
            raise BatchIntegrityError("C08 report/freeze no longer matches source or inputs")
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
    rate_verification = _verify_saved_rates(context, report)
    verification = verify_report(
        report=report, frame=context["frame"], target=context["target"],
        artifact_dir=RUN_DIR, freeze_signature=report["freeze_signature"],
        evaluation_views=context["views"], prediction_reconstructors={CARD_ID: reconstruct_c08},
    )
    verification["ordered_target_rate_status"] = rate_verification["status"]
    verification["ordered_target_rate_verification"] = rate_verification
    gate = _gate(report, verification)
    identity = context["identity"]
    results = {
        "card_id": CARD_ID,
        "status": "PILOT_VERIFIED",
        "verdict": gate["status"],
        "source_commit_context": identity["source_commit_context"],
        "freeze_signature": report["freeze_signature"],
        "input_sha256": identity["input_sha256"],
        "source_sha256": identity["source_sha256"],
        "recipe_sha256": json_sha256(context["recipe"]),
        "fit_accounting": _fit_accounting(),
        "population": {
            "fit_rows_by_cutoff": EXPECTED_FIT_ROWS,
            "score_rows_by_cutoff": EXPECTED_SCORE_ROWS,
            "view_rows": EXPECTED_VIEW_ROWS,
            "fit_all_rows_including_missing_stream": True,
        },
        "views": report["views"],
        "test_or_sample_access": False,
        "kaggle_access": False,
        "candidate_test_prediction_made": False,
        "threshold_search_or_outer_early_stopping": False,
        "frozen_controls_changed": False,
    }
    preflight = context["public"]
    _write_or_match(PREFLIGHT_PATH, preflight)
    _write_or_match(RATE_VERIFICATION_PATH, rate_verification)
    _write_or_match(VERIFICATION_PATH, verification)
    _write_or_match(GATE_PATH, gate)
    _write_or_match(RESULTS_PATH, results)
    fit55 = _read_json(RUN_DIR / "C08_fit55.json")["metadata"]
    fit67 = _read_json(RUN_DIR / "C08_fit67.json")["metadata"]
    runtime_notes = {
        "card_id": CARD_ID,
        "freeze_signature": report["freeze_signature"],
        "source_commit": identity["source_commit_context"],
        "predictor_fits": {
            "started": results["fit_accounting"]["fit_started_events"],
            "saved": results["fit_accounting"]["successful_saved_predictor_fits"],
            "failed": results["fit_accounting"]["fit_failed_events"],
            "cutoff_55_seconds": float(fit55["fit_seconds"]),
            "cutoff_67_seconds": float(fit67["fit_seconds"]),
            "total_seconds": float(results["fit_accounting"]["fit_seconds_total"]),
        },
        "warnings_observed": fit55.get("fit_metadata", {}).get("python_warnings", []) + fit67.get("fit_metadata", {}).get("python_warnings", []),
        "convergence_or_solver_warnings": "None observed in captured fit warnings." if not (fit55.get("fit_metadata", {}).get("python_warnings") or fit67.get("fit_metadata", {}).get("python_warnings")) else "See warnings_observed.",
        "memory": {
            "preflight_estimated_peak_gib_by_cutoff": {
                "55": preflight["memory_preflight"]["55"]["estimated_additional_peak_bytes"] / 1024**3,
                "67": preflight["memory_preflight"]["67"]["estimated_additional_peak_bytes"] / 1024**3,
            },
            "available_ram_gib_at_preflight_by_cutoff": {
                "55": preflight["memory_preflight"]["55"]["available_ram_bytes"] / 1024**3,
                "67": preflight["memory_preflight"]["67"]["available_ram_bytes"] / 1024**3,
            },
            "job_cap_gib": 6.0,
            "peak_process_rss_measured": False,
            "peak_process_rss_gib": None,
        },
        "official_test_read": False,
        "kaggle_accessed": False,
    }
    _write_or_match(RUNTIME_NOTES_PATH, runtime_notes)
    _reconcile_budget(results, rate_verification, gate)
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
        "ordered_target_rate_verification": rate_verification["status"],
        "test_or_sample_access": False,
        "kaggle_access": False,
    }
    _write_or_match(SUMMARY_PATH, summary)
    return {"results": results, "verification": verification, "rate_verification": rate_verification,
            "gate": gate, "summary": summary}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    queue = load_queue(QUEUE_PATH)
    card = next(row for row in queue["cards"] if row.get("id") == CARD_ID)
    result = run_card(card, queue, preflight_only=args.preflight_only)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
