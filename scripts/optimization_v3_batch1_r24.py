"""R24 Train-only pilot runner, frozen preflight, and independent verification."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import psutil
from scipy.stats import rankdata

from scripts import optimization_v3_batch1_r01 as r01
from scripts import optimization_v3_batch1_r21 as r21
from scripts.optimization_v3_batch1_verify import verify_report
from src.models.optimization_v3_batch1 import (
    BatchIntegrityError,
    append_event,
    ensure_frozen,
    evaluation_views_for_card,
    file_sha256,
    json_sha256,
    load_queue,
    make_freeze_identity,
    read_bundle,
    run_pilot,
    write_json_exclusive,
)
from src.models.optimization_v3_batch1_r24 import (
    CARD_ID,
    FEATURE_NAMES,
    PILOT_CUTOFFS,
    RIDGE_PARAMETERS,
    fit_predict as r24_fit_predict,
)
from src.models.regression_research_text import (
    _RIDGE_PARAMS,
    _TFIDF_PARAMS,
    _json_digest as text_json_digest,
    _text_documents,
)


ROOT = Path(__file__).resolve().parents[1]
QUEUE_PATH = ROOT / "experiments/optimization_v3/research_backlog/queue.json"
CARD_PATH = ROOT / "reports/optimization_v3/research_backlog/cards/R24.md"
EXPERIMENT_DIR = ROOT / "experiments/optimization_v3/batch1/R24"
RUN_DIR = ROOT / "experiments/runs/optimization_v3_batch1/R24"
REPORT_DIR = ROOT / "reports/optimization_v3/batch1/R24"
RECIPE_PATH = EXPERIMENT_DIR / "recipe.json"
# Preserve the original freeze snapshot for audit history. Its volatile live-RAM
# reading was incorrectly included in the identity, so the corrected identity is
# written to a versioned path after normalizing that observation out.
FREEZE_PATH = EXPERIMENT_DIR / "freeze.v2.json"
PILOT_PATH = RUN_DIR / "pilot.json"
RESULTS_PATH = REPORT_DIR / "results.json"
VERIFICATION_PATH = REPORT_DIR / "verification.json"
GATE_PATH = REPORT_DIR / "gate.json"
SUMMARY_PATH = REPORT_DIR / "summary.md"
LEDGER_PATH = RUN_DIR.parent / "batch_ledger.jsonl"

TRAIN_SHA256 = "756523ff8156a9cd776e52249f7e2a8f23dcec432b8d10570e55ff6db0e0f082"
QUEUE_SHA256 = "822dcd20fa28fd0e9ddc841a8c3709b372dacaca305b94b39b11a9203418fec7"
CARD_SHA256 = "0348b18c635bba0f1c858bb75fb9d47bead6452bc7a1e3231e513b38cfbb0cab"
MANIFEST_SHA256 = "dd166e92089f24978244496a9a0afabc0d06551dbdc326579a7c898a8844e19a"
EXPECTED_CARD_FIELDS = {
    "id": "R24", "task": "regression", "name": "EF interaction Hybrid Ridge",
    "family": "text_EF_interaction", "priority": "P1", "policy": "R_RIDGE_REPLACE",
    "pilot_predictor_fits": 2, "evaluation": "STANDARD", "condition": "No additional dependency",
    "change": "Original hybrid Ridge design/alpha10/lsqr/tol1e-4; append only TF-IDF*I(entry_friction==1). Replace only original Ridge component with its inherited rank75 map; LGB and sparse components exact.",
    "preflight_checks": ["Reference-coded interaction; convergence; no telemetry substitution"],
}
EXPECTED_FIT_ROWS = {55: 12_020, 67: 29_705}
EXPECTED_SCORE_ROWS = {55: 17_685, 67: 20_903}
EXPECTED_VIEW_ROWS = {
    "W1": 4_114, "W4": 4_950, "primary": 9_064,
    "H12_E": 17_685, "H12_L": 20_903, "H12": 38_588,
}
EXPECTED_VIEWS = {
    "W1": [(55, 56, 59)], "W4": [(67, 68, 71)],
    "primary": [(55, 56, 59), (67, 68, 71)],
    "H12": [(55, 56, 67), (67, 68, 79)],
}
THREADS = 2
MAX_SUCCESSFUL_FITS = 180
JOB_CAP_BYTES = 6 * 1024**3
FREE_RAM_RESERVE_BYTES = 2 * 1024**3
MIN_GROUP_SUPPORT = 100
GATE_KEYS = (
    "pilot_primary_MAE_gain_min", "pilot_H12_MAE_harm_max",
    "individual_pilot_view_MAE_harm_max", "supported_group_n_min",
    "supported_tail_or_missing_MAE_harm_max",
)
PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl", "psutil"]
RESOURCE_IDENTITY_FIELDS = (
    "status", "fit_rows", "tfidf_ngram_nnz_upper_bound_one_block",
    "structured_nnz_upper_bound", "total_nnz_upper_bound",
    "csr_bytes_upper_bound", "fixed_overhead_bytes", "estimated_peak_bytes",
    "estimated_peak_is_observed", "free_memory_reserve_bytes", "job_cap_bytes",
    "threads", "concurrent_training_jobs",
    "interaction_block_duplicates_at_most_one_tfidf_block",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_or_match(path: Path, value: Any) -> None:
    if path.exists():
        if _read_json(path) != value:
            raise BatchIntegrityError(f"Existing R24 artifact differs; refusing overwrite: {path}")
    else:
        write_json_exclusive(path, value)


def _require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"R24 required input missing: {label} ({path})")
    actual = file_sha256(path)
    if actual != expected:
        raise BatchIntegrityError(f"R24 {label} SHA256 differs: {actual}")


def _validate_card_queue(queue: Mapping[str, Any], card: Mapping[str, Any]) -> dict[str, Any]:
    _require_hash(QUEUE_PATH, QUEUE_SHA256, "canonical queue")
    _require_hash(CARD_PATH, CARD_SHA256, "canonical card")
    if load_queue(QUEUE_PATH) != dict(queue):
        raise BatchIntegrityError("R24 dispatcher queue differs from frozen canonical queue")
    canonical = next((item for item in queue.get("cards", []) if item.get("id") == CARD_ID), None)
    if canonical is None or dict(canonical) != dict(card):
        raise BatchIntegrityError("R24 card does not match its canonical queue entry")
    for key, value in EXPECTED_CARD_FIELDS.items():
        if card.get(key) != value:
            raise BatchIntegrityError(f"R24 canonical card changed at {key}")
    if (dict(RIDGE_PARAMETERS) != {"alpha": 10.0, "solver": "lsqr", "tol": 1e-4}
            or dict(_RIDGE_PARAMS) != dict(RIDGE_PARAMETERS)):
        raise BatchIntegrityError("R24 inherited Ridge parameters differ from alpha10/lsqr/tol1e-4")
    if tuple(FEATURE_NAMES) != tuple(r21.FEATURE_NAMES):
        raise BatchIntegrityError("R24 hybrid feature allowlist differs from the current incumbent")
    if set(FEATURE_NAMES) & {"record_id", "artifact_signal", "threat_rank"}:
        raise BatchIntegrityError("R24 feature allowlist contains a protected field")
    baseline = queue["baseline"]["regression"]
    if (baseline.get("Train_sha256") != TRAIN_SHA256
            or baseline.get("manifest_sha256") != MANIFEST_SHA256
            or baseline.get("manifest") != "reports/optimization_v2/kaggle/v2_reg_sparse_lgb25.json"
            or not np.isclose(float(baseline.get("primary", np.nan)), 11.559390078665364, rtol=0, atol=1e-12)
            or not np.isclose(float(baseline.get("H12", np.nan)), 14.04753258299653, rtol=0, atol=1e-12)):
        raise BatchIntegrityError("R24 fixed three-component regression baseline changed")
    gates = queue["evaluation"]["regression_gate"]
    expected_gates = {
        "pilot_primary_MAE_gain_min": 0.03,
        "pilot_H12_MAE_harm_max": 0.05,
        "individual_pilot_view_MAE_harm_max": 0.5,
        "supported_group_n_min": 100,
        "supported_tail_or_missing_MAE_harm_max": 0.25,
    }
    if any(float(gates.get(key, np.nan)) != value for key, value in expected_gates.items()):
        raise BatchIntegrityError("R24 canonical regression gates are missing or invalid")
    preset = queue["execution_preset"]
    if (int(preset.get("concurrent_fit_jobs", -1)) != 1
            or int(preset.get("threads_per_job", -1)) != THREADS
            or int(preset.get("max_successful_new_predictor_fits", -1)) != MAX_SUCCESSFUL_FITS
            or float(preset.get("max_job_RSS_GiB", -1)) != 6.0
            or float(preset.get("minimum_free_RAM_GiB_before_start", -1)) != 4.0):
        raise BatchIntegrityError("R24 execution concurrency/thread policy changed")
    return r01._validate_manifest(dict(queue))


def _input_files(train_path: Path) -> dict[str, Path]:
    files = r21._input_files(train_path)
    files["canonical_card"] = CARD_PATH
    return files


def _source_files() -> list[Path]:
    return [
        Path(__file__), ROOT / "src/models/optimization_v3_batch1_r24.py",
        ROOT / "tests/test_optimization_v3_batch1_r24.py",
        ROOT / "src/models/optimization_v3_batch1.py",
        ROOT / "scripts/optimization_v3_batch1.py",
        ROOT / "scripts/optimization_v3_batch1_verify.py",
        ROOT / "scripts/optimization_v3_batch1_r01.py",
        ROOT / "src/models/optimization_v3_batch1_r01.py",
        ROOT / "scripts/optimization_v3_batch1_r21.py",
        ROOT / "src/models/optimization_v3_batch1_r21.py",
        ROOT / "src/models/regression_research_text.py",
        ROOT / "src/models/round1_models.py",
        ROOT / "src/models/regression_round2_models.py",
        ROOT / "src/utils/paths.py",
    ]


def _resource_preflight(fit_features: pd.DataFrame, available_bytes: int | None = None) -> dict[str, Any]:
    docs, _missing = _text_documents(fit_features["signal_stream"])
    token_counts = np.fromiter(
        (len(re.findall(r"(?u)\b\w\w+\b", text.lower())) for text in docs),
        dtype=np.int64, count=len(docs),
    )
    per_row_ngrams = token_counts + np.maximum(token_counts - 1, 0)
    tfidf_nnz_upper = int(per_row_ngrams.sum(dtype=np.int64))
    structured_nnz_upper = int(len(fit_features) * 807)
    total_nnz_upper = 2 * tfidf_nnz_upper + structured_nnz_upper + len(fit_features)
    csr_bytes_upper = int(total_nnz_upper * 16 + (len(fit_features) + 1) * 8)
    fixed_overhead = int(1.5 * 1024**3)
    estimate = fixed_overhead + 4 * csr_bytes_upper
    available = int(psutil.virtual_memory().available if available_bytes is None else available_bytes)
    allowed = min(JOB_CAP_BYTES, available - FREE_RAM_RESERVE_BYTES)
    return {
        "status": "PASS" if allowed > 0 and estimate <= allowed else "BLOCKED_RESOURCE",
        "fit_rows": int(len(fit_features)),
        "tfidf_ngram_nnz_upper_bound_one_block": tfidf_nnz_upper,
        "structured_nnz_upper_bound": structured_nnz_upper,
        "total_nnz_upper_bound": total_nnz_upper,
        "csr_bytes_upper_bound": csr_bytes_upper,
        "fixed_overhead_bytes": fixed_overhead,
        "estimated_peak_bytes": estimate,
        "estimated_peak_is_observed": False,
        "available_physical_bytes": available,
        "allowed_bytes": int(max(0, allowed)),
        "free_memory_reserve_bytes": FREE_RAM_RESERVE_BYTES,
        "job_cap_bytes": JOB_CAP_BYTES,
        "threads": THREADS,
        "concurrent_training_jobs": 1,
        "interaction_block_duplicates_at_most_one_tfidf_block": True,
    }


def _resource_identity(memory: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Keep the frozen resource estimate stable while rechecking live RAM per run.

    Available/allowed RAM are volatile host observations. Resource status and
    the deterministic workload estimate remain frozen; _prepare still requires
    a fresh live preflight to PASS before a fit can begin.
    """
    output: dict[str, dict[str, Any]] = {}
    for cutoff, snapshot in sorted(memory.items(), key=lambda item: int(item[0])):
        missing = [field for field in RESOURCE_IDENTITY_FIELDS if field not in snapshot]
        if missing:
            raise BatchIntegrityError(f"R24 resource snapshot is missing frozen fields: {missing}")
        output[str(cutoff)] = {field: snapshot[field] for field in RESOURCE_IDENTITY_FIELDS}
    return output


def _budget_preflight() -> dict[str, Any]:
    events = []
    if LEDGER_PATH.exists():
        events = [json.loads(line) for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    reservations: dict[str, int] = {}
    for event in events:
        if event.get("event") != "BUDGET_RESERVED":
            continue
        key, count = str(event.get("budget_key")), int(event.get("predictor_fits", -1))
        if count < 0 or (key in reservations and reservations[key] != count):
            raise BatchIntegrityError("R24 shared fit ledger has invalid or conflicting reservations")
        reservations[key] = count
    own = reservations.get("R24:PILOT")
    total = sum(reservations.values())
    next_total = total if own == 2 else total + 2
    return {
        "status": "PASS" if own in (None, 2) and next_total <= MAX_SUCCESSFUL_FITS else "BLOCKED_BUDGET",
        "reserved_total": total, "next_total_if_started": next_total,
        "existing_r24_reservation": own, "cap": MAX_SUCCESSFUL_FITS,
        "reservation_count_by_key": reservations,
    }


def _control_views(cycles: np.ndarray, views: Mapping[str, Any],
                   controls: Mapping[int, Mapping[str, Any]]) -> dict[str, dict[str, np.ndarray]]:
    per_cutoff = {
        cutoff: {"rows": values["score_rows"], "candidate": values["baseline_prediction"],
                 "baseline": values["baseline_prediction"]}
        for cutoff, values in controls.items()
    }
    context = {"cycles": cycles, "views": views}
    return r21._view_arrays(context, per_cutoff)


def _pre_freeze_support(frame: pd.DataFrame, target: np.ndarray,
                        controls: Mapping[int, Mapping[str, Any]], cycles: np.ndarray,
                        views: Mapping[str, Any]) -> dict[str, Any]:
    view_arrays = _control_views(cycles, views, controls)
    groups = r21._group_diagnostics(frame, target, view_arrays)
    return {
        "group_support_minimum": MIN_GROUP_SUPPORT,
        "views": {
            view: {group: {"n": int(item["n"]), "supported": bool(item["supported"])}
                   for group, item in rows.items()}
            for view, rows in groups["views"].items()
        },
        "fixed_diagnostic_groups": ["signal_stream_missing", "target_le_5", "target_ge_95"],
    }


def _recipe(card: Mapping[str, Any], views_for_identity: Mapping[str, Any],
            controls_metrics: Mapping[str, Any], support: Mapping[str, Any],
            queue: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    from src.models.regression_research_text import CATEGORICAL_FEATURES, STRUCTURED_FEATURES

    return {
        "card_id": CARD_ID,
        "card_sha256": CARD_SHA256,
        "canonical_queue_sha256": QUEUE_SHA256,
        "queue_card": {key: card[key] for key in (
            "id", "task", "name", "family", "priority", "change", "policy",
            "pilot_predictor_fits", "evaluation", "condition", "preflight_checks",
        )},
        "accepted_control": {
            "manifest": str(r01.ACCEPTED_MANIFEST.relative_to(ROOT)),
            "manifest_sha256": MANIFEST_SHA256,
            "candidate_id": manifest["candidate_id"],
            "route": ".5625 rank50 LGB-SVD32 + .1875 rank75 hybrid Ridge + .25 rank50 sparse LGB",
            "pilot_control_is_exact_three_component_incumbent": True,
        },
        "feature_allowlist": list(FEATURE_NAMES),
        "feature_roles": {"record_id": "identity_only", "threat_rank": "target_only"},
        "structured_recipe": {
            "implementation": "RegressionTextModel._structured_fit/_structured_transform",
            "numeric_features": list(STRUCTURED_FEATURES),
            "categorical_features": list(CATEGORICAL_FEATURES),
            "fit_scope": "cutoff past rows only",
            "rare_category": {"min_frequency": 30, "max_categories": 254,
                              "unknown": "infrequent_if_exist"},
        },
        "text_recipe": {
            "vectorizer": "sklearn.feature_extraction.text.TfidfVectorizer",
            "parameters": {key: (list(value) if isinstance(value, tuple) else
                                  value.__name__ if isinstance(value, type) else value)
                           for key, value in _TFIDF_PARAMS.items()},
            "fit_scope": "cutoff past rows only",
            "missing_stream": "empty document plus explicit binary missing column",
        },
        "interaction": {
            "definition": "Append TF-IDF * I(parsed entry_friction == 1) to original hybrid Ridge design",
            "reference_group": "parsed entry_friction == 0",
            "parsed_values_in_official_train": [0, 1],
            "telemetry_substitution": False,
            "fit_scope": "same cutoff past rows as base TF-IDF; no score vocabulary fit",
        },
        "estimator": {"class": "sklearn.linear_model.Ridge", "parameters": dict(RIDGE_PARAMETERS),
                      "target": "threat_rank", "sample_weight": False},
        "prediction_policy": {
            "ridge_mapping": "rank75(raw): within each full score cycle 0.25*raw + 0.75*100*average_rank/n, then clip[0,100]",
            "candidate": "clip(0.75*(0.75*frozen_mapped_LGB + 0.25*rank75(new_Ridge)) + 0.25*frozen_sparse_rank50, 0, 100)",
            "baseline": "clip(0.75*(0.75*frozen_mapped_LGB + 0.25*frozen_rank75_Ridge) + 0.25*frozen_sparse_rank50, 0, 100)",
            "lgb_component_changes": False, "sparse_component_changes": False,
        },
        "fit_population": "all official Regression Train rows with 47 <= cycle_code <= cutoff",
        "score_population": "complete declared 12-cycle horizons at cutoffs 55 and 67",
        "evaluation_views": {name: [list(segment) for segment in segments]
                             for name, segments in views_for_identity.items()},
        "fit_rows": {str(key): value for key, value in EXPECTED_FIT_ROWS.items()},
        "score_rows": {str(key): value for key, value in EXPECTED_SCORE_ROWS.items()},
        "expected_view_rows": dict(EXPECTED_VIEW_ROWS),
        "row_matched_control_metrics": dict(controls_metrics),
        "pre_freeze_group_support": dict(support),
        "gates": {key: float(queue["evaluation"]["regression_gate"][key]) for key in GATE_KEYS},
        "convergence_diagnostics": {
            "record": "Ridge.n_iter_, fit warnings, and post-fit normalized stationarity ratio",
            "no_extra_predictor_fit": True,
            "stationarity_is_diagnostic_not_a_tuned_gate": True,
        },
        "resource_policy": {"threads": THREADS, "concurrent_jobs": 1,
                             "job_cap_bytes": JOB_CAP_BYTES,
                             "free_ram_reserve_bytes": FREE_RAM_RESERVE_BYTES},
        "official_test_read": False, "kaggle_access": False,
    }


def _prepare(card: Mapping[str, Any], queue: Mapping[str, Any]) -> dict[str, Any]:
    manifest = _validate_card_queue(queue, card)
    train_path, frame, target, cycles = r21._read_train()
    if file_sha256(train_path) != TRAIN_SHA256:
        raise BatchIntegrityError("Official Regression Train changed after R24 source read")
    ef = pd.to_numeric(frame["entry_friction"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(ef).all() or not np.isin(ef, [0.0, 1.0]).all():
        raise BatchIntegrityError("R24 parsed entry_friction support differs from fixed binary interaction")
    views = evaluation_views_for_card(queue, CARD_ID)
    if views != EXPECTED_VIEWS:
        raise BatchIntegrityError(f"R24 STANDARD temporal views changed: {views}")

    base = r01._read_npz(r01.BASE_CACHE)
    sparse_cache = r01._read_npz(r01.SPARSE_CACHE)
    diagnostic = r01._read_npz(r01.DIAGNOSTIC_CACHE)
    load_controls = r01._load_controls(frame, base, sparse_cache, diagnostic)
    controls: dict[int, dict[str, Any]] = {}
    memory: dict[str, Any] = {}
    interaction_support: dict[str, Any] = {}
    for cutoff in PILOT_CUTOFFS:
        fit_rows = np.flatnonzero(cycles <= cutoff)
        score_rows = np.flatnonzero((cycles > cutoff) & (cycles <= cutoff + 12))
        if len(fit_rows) != EXPECTED_FIT_ROWS[cutoff] or len(score_rows) != EXPECTED_SCORE_ROWS[cutoff]:
            raise BatchIntegrityError(f"R24 fit/score population differs at cutoff {cutoff}")
        values = load_controls(cutoff, score_rows)
        controls[cutoff] = {
            **values,
            "score_rows": score_rows.astype(np.int64),
            "score_record_ids": frame.iloc[score_rows]["record_id"].astype(str).to_numpy(dtype=str),
            "control_rows": score_rows.astype(np.int64),
        }
        memory[str(cutoff)] = _resource_preflight(frame.iloc[fit_rows].loc[:, list(FEATURE_NAMES)])
        fit_ef = ef[fit_rows]
        fit_docs, _ = _text_documents(frame.iloc[fit_rows]["signal_stream"])
        ngrams = np.fromiter(
            (len(re.findall(r"(?u)\b\w\w+\b", text.lower()))
             + max(len(re.findall(r"(?u)\b\w\w+\b", text.lower())) - 1, 0)
             for text in fit_docs), dtype=np.int64, count=len(fit_docs),
        )
        interaction_support[str(cutoff)] = {
            "fit_rows": int(len(fit_rows)), "entry_friction_1_rows": int(np.sum(fit_ef == 1)),
            "entry_friction_0_rows": int(np.sum(fit_ef == 0)),
            "ef1_rows_with_token_ngrams": int(np.sum((fit_ef == 1) & (ngrams > 0))),
            "text_interaction_nnz_upper_bound": int(ngrams[fit_ef == 1].sum(dtype=np.int64)),
        }
    if any(item["status"] != "PASS" for item in memory.values()):
        raise BatchIntegrityError(f"R24 sparse/LSQR resource preflight failed: {memory}")

    views_for_identity = dict(views)
    views_for_identity["H12_E"] = [(55, 56, 67)]
    views_for_identity["H12_L"] = [(67, 68, 79)]
    view_arrays = _control_views(cycles, views, controls)
    control_metrics = {
        name: {"rows": int(len(values["rows"])),
               "mae": float(r21._mae(target[values["rows"]], values["baseline"]))}
        for name, values in view_arrays.items()
    }
    if any(control_metrics[name]["rows"] != expected for name, expected in EXPECTED_VIEW_ROWS.items()):
        raise BatchIntegrityError("R24 row-matched incumbent view denominators changed")
    support = _pre_freeze_support(frame, target, controls, cycles, views)
    budget = _budget_preflight()
    if budget["status"] != "PASS":
        raise BatchIntegrityError(f"R24 shared fit budget is blocked: {budget}")

    input_files = _input_files(train_path)
    expected_hashes = {
        "official_regression_train": TRAIN_SHA256,
        "canonical_queue": QUEUE_SHA256,
        "canonical_card": CARD_SHA256,
        "accepted_regression_manifest": r01.EXPECTED_MANIFEST_SHA256,
        "v2_paired_lgb_ridge_cache": r01.EXPECTED_BASE_CACHE_SHA256,
        "v2_sparse_pilot_cache": r01.EXPECTED_SPARSE_CACHE_SHA256,
        "v3_champion_diagnostic_cache": r01.EXPECTED_DIAGNOSTIC_CACHE_SHA256,
        "v3_regression_diagnostics": r01.EXPECTED_DIAGNOSTICS_SHA256,
    }
    input_hashes = {name: file_sha256(path) for name, path in sorted(input_files.items())}
    for name, expected in expected_hashes.items():
        if input_hashes.get(name) != expected:
            raise BatchIntegrityError(f"R24 frozen input hash differs for {name}")
    source_files = _source_files()
    missing = [str(path) for path in source_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"R24 frozen source files are missing: {missing}")
    recipe = _recipe(card, views_for_identity, control_metrics, support, queue, manifest)
    identity = make_freeze_identity(
        card=card, feature_names=FEATURE_NAMES, input_files=input_files,
        source_files=source_files, evaluation_views=views_for_identity, packages=PACKAGES,
    )
    identity.update({
        "recipe": recipe,
        "recipe_sha256": json_sha256(recipe),
        "resource_preflight": _resource_identity(memory),
        "interaction_support": interaction_support,
        "control_cache_hashes": {
            "paired_lgb_ridge": r01.EXPECTED_BASE_CACHE_SHA256,
            "sparse_pilot": r01.EXPECTED_SPARSE_CACHE_SHA256,
            "independent_champion_diagnostic": r01.EXPECTED_DIAGNOSTIC_CACHE_SHA256,
        },
        "budget_preflight": budget,
        "validation_labels_passed_to_adapter": False,
        "official_test_read": False,
    })
    if RECIPE_PATH.exists() and _read_json(RECIPE_PATH) != recipe:
        raise BatchIntegrityError("Existing R24 recipe differs from the live preflight")
    if FREEZE_PATH.exists():
        frozen = _read_json(FREEZE_PATH)
        if frozen.get("identity") != identity or frozen.get("signature") != json_sha256(identity):
            raise BatchIntegrityError("Existing R24 freeze differs from live source/input identity")
    public = {
        "card_id": CARD_ID, "status": "PREFLIGHT_PASS",
        "official_train_sha256": TRAIN_SHA256,
        "input_sha256": input_hashes,
        "source_sha256": {str(path.relative_to(ROOT)): file_sha256(path) for path in source_files},
        "fit_rows": EXPECTED_FIT_ROWS, "score_rows": EXPECTED_SCORE_ROWS,
        "view_denominators": {name: value["rows"] for name, value in control_metrics.items()},
        "control_metrics": control_metrics,
        "interaction_support": interaction_support,
        "pre_freeze_group_support": support,
        "resource_preflight": memory, "fit_budget_preflight": budget,
        "freeze_identity_sha256": json_sha256(identity),
        "official_test_read": False, "candidate_fit_or_metric_computed": False,
        "freeze_or_output_written": False, "python_version": platform.python_version(),
    }
    return {
        "train_path": train_path, "frame": frame, "target": target, "cycles": cycles,
        "views": views, "views_for_identity": views_for_identity,
        "controls": controls, "identity": identity, "recipe": recipe,
        "public": public, "control_views": control_metrics,
        "budget": budget, "resource_preflight": memory, "support": support,
    }


def _control_loader(context: Mapping[str, Any]):
    def load(cutoff: int, score_rows: np.ndarray) -> dict[str, np.ndarray]:
        saved = context["controls"][int(cutoff)]
        rows = np.asarray(score_rows, dtype=np.int64)
        if not np.array_equal(rows, saved["score_rows"]):
            raise BatchIntegrityError(f"R24 frozen controls do not align at cutoff {cutoff}")
        return {
            **{key: np.asarray(saved[key]) for key in (
                "baseline_prediction", "baseline_lgb_svd32", "baseline_hybrid_ridge",
                "baseline_sparse_raw", "baseline_sparse_rank50",
            )},
            "control_rows": rows,
        }
    return load


def _rank50_independent(raw: np.ndarray, cycles: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    cycles = np.asarray(cycles, dtype=np.int64)
    if raw.ndim != 1 or raw.shape != cycles.shape or not len(raw) or not np.isfinite(raw).all():
        raise BatchIntegrityError("R24 independent sparse rank inputs are invalid")
    output = np.empty(len(raw), dtype=np.float64)
    for cycle in np.unique(cycles):
        mask = cycles == cycle
        output[mask] = 0.5 * raw[mask] + 0.5 * 100.0 * rankdata(raw[mask], method="average") / int(mask.sum())
    return np.clip(output, 0.0, 100.0)


def _rank75_independent(raw: np.ndarray, cycles: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    cycles = np.asarray(cycles, dtype=np.int64)
    if raw.ndim != 1 or raw.shape != cycles.shape or not len(raw) or not np.isfinite(raw).all():
        raise BatchIntegrityError("R24 independent rank75 inputs are invalid")
    output = np.empty(len(raw), dtype=np.float64)
    for cycle in np.unique(cycles):
        mask = cycles == cycle
        ranks = pd.Series(raw[mask]).rank(method="average").to_numpy(dtype=np.float64)
        percentile = 100.0 * ranks / int(mask.sum())
        output[mask] = 0.25 * raw[mask] + 0.75 * percentile
    return np.clip(output, 0.0, 100.0)


def reconstruct_r24_arrays(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(arrays["raw_candidate"], dtype=np.float64)
    cycles = np.asarray(arrays["audit__score_cycles"], dtype=np.int64)
    lgb = np.asarray(arrays["control__baseline_lgb_svd32"], dtype=np.float64)
    ridge = np.asarray(arrays["control__baseline_hybrid_ridge"], dtype=np.float64)
    sparse_raw = np.asarray(arrays["control__baseline_sparse_raw"], dtype=np.float64)
    sparse_saved = np.asarray(arrays["control__baseline_sparse_rank50"], dtype=np.float64)
    saved_baseline = np.asarray(arrays["baseline_prediction"], dtype=np.float64)
    if len({value.shape for value in (raw, cycles, lgb, ridge, sparse_raw, sparse_saved, saved_baseline)}) != 1:
        raise BatchIntegrityError("R24 saved reconstruction arrays are misaligned")
    sparse_rank = _rank50_independent(sparse_raw, cycles)
    if not np.array_equal(sparse_rank, sparse_saved):
        raise BatchIntegrityError("R24 frozen sparse rank50 component does not reconstruct")
    baseline = np.clip(0.75 * (0.75 * lgb + 0.25 * ridge) + 0.25 * sparse_rank, 0.0, 100.0)
    if not np.array_equal(baseline, saved_baseline):
        raise BatchIntegrityError("R24 frozen three-component baseline does not reconstruct")
    mapped = _rank75_independent(raw, cycles)
    candidate = np.clip(0.75 * (0.75 * lgb + 0.25 * mapped) + 0.25 * sparse_rank, 0.0, 100.0)
    saved_mapped = np.asarray(arrays["audit__candidate_rank75"], dtype=np.float64)
    if not np.array_equal(mapped, saved_mapped):
        raise BatchIntegrityError("R24 candidate Ridge rank75 mapping does not reconstruct")
    return candidate, baseline


def _verify_pilot(context: Mapping[str, Any], report: Mapping[str, Any],
                  queue: Mapping[str, Any]) -> dict[str, Any]:
    signature = json_sha256(context["identity"])
    if (report.get("card_id") != CARD_ID or report.get("task") != "regression"
            or report.get("status") != "PILOT_SAVED_AWAITING_INDEPENDENT_QA"
            or report.get("freeze_signature") != signature
            or report.get("cutoffs") != list(PILOT_CUTOFFS)
            or report.get("fit_predictor_count") != 2
            or report.get("fit_count_matches_card") is not True):
        raise BatchIntegrityError("R24 generic pilot report identity/fit count is invalid")
    frame = context["frame"]
    target = np.asarray(context["target"], dtype=np.float64)
    cycles = np.asarray(context["cycles"], dtype=np.int64)
    ids = frame["record_id"].astype(str).to_numpy(dtype=str)
    per_cutoff: dict[int, dict[str, np.ndarray]] = {}
    fit_records: dict[str, Any] = {}
    for cutoff in PILOT_CUTOFFS:
        control = context["controls"][cutoff]
        rows = np.asarray(control["score_rows"], dtype=np.int64)
        identity = {"freeze_signature": signature, "card_id": CARD_ID,
                    "cutoff": cutoff, "score_rows": rows.tolist()}
        arrays, sidecar = read_bundle(RUN_DIR / f"R24_fit{cutoff}.npz", identity)
        if (not np.array_equal(arrays["score_rows"], rows)
                or not np.array_equal(arrays["score_record_ids"], ids[rows])
                or len(np.unique(arrays["score_record_ids"])) != len(rows)):
            raise BatchIntegrityError(f"R24 prediction row or record_id alignment differs at {cutoff}")
        candidate, baseline = reconstruct_r24_arrays(arrays)
        if (not np.array_equal(candidate, arrays["prediction"])
                or not np.array_equal(baseline, arrays["baseline_prediction"])
                or not np.array_equal(baseline, control["baseline_prediction"])):
            raise BatchIntegrityError(f"R24 independent replacement policy differs at cutoff {cutoff}")
        if (not np.isfinite(candidate).all() or np.any((candidate < 0) | (candidate > 100))
                or not np.isfinite(baseline).all() or np.any((baseline < 0) | (baseline > 100))):
            raise BatchIntegrityError(f"R24 candidate or baseline is outside [0,100] at {cutoff}")
        metadata = sidecar.get("metadata", {})
        state = metadata.get("state", {})
        fit_rows = np.flatnonzero(cycles <= cutoff)
        expected_fit_hash = hashlib.sha256(np.asarray(fit_rows, dtype="<i8").tobytes(order="C")).hexdigest()
        expected_target_hash = hashlib.sha256(np.asarray(target[fit_rows], dtype="<f8").tobytes(order="C")).hexdigest()
        state_payload = dict(state)
        state_hash = state_payload.pop("state_sha256", None)
        state_hash_matches = state_hash == text_json_digest(state_payload)
        solver = state.get("solver_diagnostics", {})
        iterations = solver.get("n_iter")
        if (int(metadata.get("fit_rows", -1)) != EXPECTED_FIT_ROWS[cutoff]
                or int(metadata.get("fit_predictor_count", -1)) != 1
                or state.get("card_id") != CARD_ID
                or int(state.get("cutoff", -1)) != cutoff
                or int(state.get("fit_rows", -1)) != EXPECTED_FIT_ROWS[cutoff]
                or int(state.get("fit_cycle_min", -1)) != 47
                or int(state.get("fit_cycle_max", -1)) != cutoff
                or state.get("fit_row_positions_sha256") != expected_fit_hash
                or state.get("target_sha256") != expected_target_hash
                or not state_hash_matches
                or state.get("feature_names") != list(FEATURE_NAMES)
                or state.get("parameters") != RIDGE_PARAMETERS
                or state.get("validation_data_passed_to_fit") is not False
                or state.get("validation_labels_passed_to_fit") is not False
                or state.get("test_data_used_for_fit") is not False
                or state.get("early_stopping_used") is not False
                or state.get("sample_weight_used") is not False
                or solver.get("solver") != "lsqr"
                or (iterations is not None and any(int(value) < 1 for value in iterations))
                or not np.isfinite(float(solver.get("stationarity_ratio", np.nan)))):
            raise BatchIntegrityError(f"R24 fit-only/source-bound Ridge audit failed at {cutoff}")
        if not np.array_equal(arrays["audit__score_cycles"], cycles[rows]):
            raise BatchIntegrityError(f"R24 score cycle mapping differs at cutoff {cutoff}")
        per_cutoff[cutoff] = {"rows": rows, "candidate": candidate, "baseline": baseline}
        fit_records[str(cutoff)] = {
            "fit_rows": int(len(fit_rows)), "score_rows": int(len(rows)),
            "state_sha256": state.get("state_sha256"),
            "solver_diagnostics": solver,
            "fit_metadata": metadata.get("fit_metadata", {}),
        }

    views = r21._view_arrays(
        {"cycles": cycles, "views": context["views"]}, per_cutoff
    )
    metrics: dict[str, Any] = {}
    for name, values in views.items():
        rows = values["rows"]
        candidate_mae = float(np.abs(target[rows] - values["candidate"]).sum(dtype=np.float64) / len(rows))
        baseline_mae = float(np.abs(target[rows] - values["baseline"]).sum(dtype=np.float64) / len(rows))
        gain = baseline_mae - candidate_mae
        if name in context["views"]:
            declared = report["views"][name]
            if (int(declared["rows"]) != len(rows)
                    or not np.isclose(float(declared["candidate"]["mae"]), candidate_mae, atol=1e-10, rtol=0)
                    or not np.isclose(float(declared["baseline"]["mae"]), baseline_mae, atol=1e-10, rtol=0)
                    or not np.isclose(float(declared["gain_positive_is_better"]), gain, atol=1e-10, rtol=0)):
                raise BatchIntegrityError(f"R24 independently recomputed metric differs for {name}")
        metrics[name] = {"rows": int(len(rows)), "candidate_mae": candidate_mae,
                         "baseline_mae": baseline_mae, "gain_positive_is_better": float(gain)}
    groups = r21._group_diagnostics(frame, target, views)
    gate = queue["evaluation"]["regression_gate"]
    group_failures = []
    for view_name, group_values in groups["views"].items():
        for group_name, item in group_values.items():
            if (item["supported"]
                    and float(item["candidate_minus_baseline_mae"])
                    > float(gate["supported_tail_or_missing_MAE_harm_max"])):
                group_failures.append({"view": view_name, "group": group_name, **item})
    checks = {
        "pilot_primary_MAE_gain_min": metrics["primary"]["gain_positive_is_better"] >= float(gate["pilot_primary_MAE_gain_min"]),
        "pilot_H12_MAE_harm_max": metrics["H12"]["gain_positive_is_better"] >= -float(gate["pilot_H12_MAE_harm_max"]),
        "W1_MAE_harm_max": metrics["W1"]["gain_positive_is_better"] >= -float(gate["individual_pilot_view_MAE_harm_max"]),
        "W4_MAE_harm_max": metrics["W4"]["gain_positive_is_better"] >= -float(gate["individual_pilot_view_MAE_harm_max"]),
        "supported_tail_or_missing_harm_max": len(group_failures) == 0,
    }
    execution = _fit_accounting()
    if execution["successful_predictor_fits"] != 2 or not execution["attempt_event_pairs_complete"]:
        raise BatchIntegrityError("R24 successful fit and attempt ledger accounting is incomplete")
    decision = "ADVANCE" if all(checks.values()) else "RETIRE"
    return {
        "card_id": CARD_ID, "status": "INDEPENDENT_QA_PASS", "decision": decision,
        "freeze_signature": signature, "fit_predictor_count": 2,
        "cutoffs": list(PILOT_CUTOFFS), "fit_records": fit_records,
        "views": metrics, "group_diagnostics": groups,
        "supported_group_failures": group_failures,
        "fit_execution_accounting": execution,
        "gate_checks": checks,
        "gate_thresholds": {key: float(gate[key]) for key in GATE_KEYS},
        "generic_independent_verifier": "scripts/optimization_v3_batch1_verify.py",
        "metric_recomputed_from_train_labels_after_prediction": True,
        "validation_labels_passed_to_adapter": False,
        "official_test_read": False, "kaggle_access": False,
    }


def _fit_accounting() -> dict[str, Any]:
    events = [json.loads(line) for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    card = [event for event in events if event.get("card_id") == CARD_ID]
    reservations = [event for event in card if event.get("event") == "BUDGET_RESERVED"]
    if (len(reservations) != 1 or reservations[0].get("budget_key") != "R24:PILOT"
            or int(reservations[0].get("predictor_fits", -1)) != 2):
        raise BatchIntegrityError("R24 fit ledger lacks its exact two-predictor reservation")
    attempts = [event for event in card if event.get("event") in {
        "FIT_STARTED", "FIT_SAVED", "FIT_FAILED", "RESUMED_VERIFIED",
    }]
    starts = sum(event.get("event") == "FIT_STARTED" for event in attempts)
    saved = sum(event.get("event") == "FIT_SAVED" for event in attempts)
    failed = sum(event.get("event") == "FIT_FAILED" for event in attempts)
    resumed = sum(event.get("event") == "RESUMED_VERIFIED" for event in attempts)
    successful_cutoffs = {
        int(event["cutoff"]) for event in attempts
        if event.get("event") in {"FIT_SAVED", "RESUMED_VERIFIED"} and "cutoff" in event
    }
    if successful_cutoffs != set(PILOT_CUTOFFS) or starts != saved + failed:
        raise BatchIntegrityError("R24 attempt events do not reconcile to exactly two successful fits")
    return {
        "reserved_predictor_fits": 2, "successful_predictor_fits": 2,
        "fit_started_events": starts, "fit_saved_events": saved,
        "resumed_verified_events": resumed, "fit_failed_events": failed,
        "attempt_event_pairs_complete": True,
        "attempts": [{key: event.get(key) for key in (
            "event", "cutoff", "freeze_signature", "fit_rows", "score_rows",
            "fit_predictor_count", "fit_seconds", "error_type", "error",
        ) if key in event} for event in attempts],
    }


def _publish(context: Mapping[str, Any], report: Mapping[str, Any],
             verification: Mapping[str, Any], generic: Mapping[str, Any]) -> None:
    gate = {
        "card_id": CARD_ID, "profile": "STANDARD", "decision": verification["decision"],
        "status": "PASS" if verification["decision"] == "ADVANCE" else "FAIL_VALID_EVIDENCE",
        "checks": verification["gate_checks"], "thresholds": verification["gate_thresholds"],
        "supported_group_failures": verification["supported_group_failures"],
        "decision_basis": "Frozen STANDARD views, exact three-component row-matched control, independent rank/route/metric reconstruction, and supported tail/missing guards; no Test/Kaggle evidence.",
    }
    result = dict(report)
    result.update({
        "status": f"PILOT_COMPLETE_{verification['decision']}",
        "input_sha256": context["public"]["input_sha256"],
        "source_sha256": context["public"]["source_sha256"],
        "preflight": context["public"],
        "independent_verification": dict(verification),
        "generic_independent_verification": dict(generic),
        "gate": gate,
        "resource_note": "Sparse design memory was conservatively estimated before fitting; peak process RAM was not observed.",
        "test_data_used": False, "sample_submission_used": False, "kaggle_access": False,
    })
    primary, h12 = verification["views"]["primary"], verification["views"]["H12"]
    summary = (
        "# R24 Pilot Summary\n\n"
        f"- Decision: **{verification['decision']}**\n"
        f"- Primary MAE: {primary['baseline_mae']:.6f} baseline → {primary['candidate_mae']:.6f} candidate; "
        f"gain {primary['gain_positive_is_better']:+.6f} on {primary['rows']:,} rows.\n"
        f"- H12 MAE: {h12['baseline_mae']:.6f} baseline → {h12['candidate_mae']:.6f} candidate; "
        f"gain {h12['gain_positive_is_better']:+.6f} on {h12['rows']:,} rows.\n"
        f"- W1 / W4 gains: {verification['views']['W1']['gain_positive_is_better']:+.6f} / "
        f"{verification['views']['W4']['gain_positive_is_better']:+.6f}.\n"
        f"- LSQR stationarity ratios: {verification['fit_records']['55']['solver_diagnostics']['stationarity_ratio']:.3g} "
        f"(cutoff 55), {verification['fit_records']['67']['solver_diagnostics']['stationarity_ratio']:.3g} (cutoff 67).\n"
        "- Temporal development outcomes are reused, not an untouched holdout. Test/Sample/Kaggle were not read.\n"
    )
    _write_or_match(RESULTS_PATH, result)
    _write_or_match(VERIFICATION_PATH, dict(verification))
    _write_or_match(GATE_PATH, gate)
    if SUMMARY_PATH.exists():
        if SUMMARY_PATH.read_text(encoding="utf-8") != summary:
            raise BatchIntegrityError("Existing R24 summary differs; refusing overwrite")
    else:
        SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with SUMMARY_PATH.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(summary)


def run_card(card: Mapping[str, Any], queue: Mapping[str, Any], *,
             preflight_only: bool = False, freeze_only: bool = False,
             run_frozen_pilot: bool = False) -> dict[str, Any]:
    if sum((preflight_only, freeze_only, run_frozen_pilot)) != 1:
        raise BatchIntegrityError("Choose exactly one R24 mode")
    context = _prepare(card, queue)
    if preflight_only:
        return context["public"]
    if freeze_only:
        EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
        if RECIPE_PATH.exists():
            if _read_json(RECIPE_PATH) != context["recipe"]:
                raise BatchIntegrityError("Existing R24 recipe differs; refusing overwrite")
        else:
            write_json_exclusive(RECIPE_PATH, context["recipe"])
        if FREEZE_PATH.exists():
            frozen = _read_json(FREEZE_PATH)
            if frozen.get("identity") != context["identity"] or frozen.get("signature") != json_sha256(context["identity"]):
                raise BatchIntegrityError("Existing R24 freeze differs; refusing overwrite")
        else:
            ensure_frozen(FREEZE_PATH, context["identity"])
        result = dict(context["public"])
        result.update({"status": "FROZEN_READY", "recipe_path": str(RECIPE_PATH.relative_to(ROOT)),
                       "freeze_path": str(FREEZE_PATH.relative_to(ROOT)),
                       "freeze_sha256": file_sha256(FREEZE_PATH),
                       "candidate_fit_performed": False, "artifacts_written": True})
        return result
    if not RECIPE_PATH.is_file() or not FREEZE_PATH.is_file():
        raise BatchIntegrityError("R24 pilot requires a committed recipe and prefit freeze")
    if _read_json(RECIPE_PATH) != context["recipe"]:
        raise BatchIntegrityError("R24 recipe differs from live preflight; no fit started")
    frozen = _read_json(FREEZE_PATH)
    if frozen.get("identity") != context["identity"] or frozen.get("signature") != json_sha256(context["identity"]):
        raise BatchIntegrityError("R24 frozen source/input identity differs; no fit started")
    if file_sha256(context["train_path"]) != TRAIN_SHA256:
        raise BatchIntegrityError("Official Regression Train changed after R24 freeze")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = run_pilot(
        card=card, artifact_dir=RUN_DIR, budget_ledger_path=LEDGER_PATH,
        max_successful_fits=MAX_SUCCESSFUL_FITS, frame=context["frame"],
        target=np.asarray(context["target"], dtype=np.float64), feature_names=FEATURE_NAMES,
        adapter=r24_fit_predict, control_loader=_control_loader(context),
        freeze_identity=context["identity"], freeze_path=FREEZE_PATH,
        event_path=LEDGER_PATH, evaluation_views=context["views"],
        threads=THREADS, report_path=PILOT_PATH,
    )
    verification = _verify_pilot(context, report, queue)
    generic = verify_report(
        report=dict(report), frame=context["frame"], target=context["target"],
        artifact_dir=RUN_DIR, freeze_signature=report["freeze_signature"],
        evaluation_views=context["views"], prediction_reconstructors={CARD_ID: reconstruct_r24_arrays},
    )
    if generic.get("status") != "PASS":
        raise BatchIntegrityError("Generic independent verifier rejected R24")
    verification["generic_independent_verifier_status"] = generic["status"]
    _publish(context, report, verification, generic)
    return {
        "card_id": CARD_ID, "status": f"PILOT_COMPLETE_{verification['decision']}",
        "decision": verification["decision"], "views": verification["views"],
        "gate_checks": verification["gate_checks"],
        "fit_execution_accounting": verification["fit_execution_accounting"],
        "test_data_used": False, "kaggle_access": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--freeze-only", action="store_true")
    mode.add_argument("--run-frozen-pilot", action="store_true")
    args = parser.parse_args()
    queue = load_queue(QUEUE_PATH)
    card = next(item for item in queue["cards"] if item["id"] == CARD_ID)
    result = run_card(card, queue, preflight_only=args.preflight_only,
                      freeze_only=args.freeze_only, run_frozen_pilot=args.run_frozen_pilot)
    print(json.dumps(result, indent=2, allow_nan=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["reconstruct_r24_arrays", "run_card"]
