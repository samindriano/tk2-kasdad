"""R23 read-only preflight, frozen pilot runner, and independent QA.

Only official Regression Train and previously verified train-only controls are
used. This module never resolves or opens Regression Test or SampleSubmission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import optimization_v3_batch1_r01 as r01
from scripts.optimization_v3_batch1_verify import verify_report
from src.models.optimization_v3_batch1 import (
    BatchIntegrityError,
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
from src.models.optimization_v3_batch1_r23 import (
    CARD_ID,
    FEATURE_NAMES,
    EXTRA_TREES_PARAMETERS,
    fit_predict as r23_fit_predict,
    text_svd32_support,
)
from src.models.regression_research_text import (
    HYBRID_FEATURES,
    _SVD_PARAMS,
    _TFIDF_PARAMS,
    _array_digest,
    _json_digest,
)
from src.utils.paths import task_files


QUEUE_PATH = ROOT / "experiments/optimization_v3/research_backlog/queue.json"
CARD_PATH = ROOT / "reports/optimization_v3/research_backlog/cards/R23.md"
EXPERIMENT_DIR = ROOT / "experiments/optimization_v3/batch1/R23"
RUN_DIR = ROOT / "experiments/runs/optimization_v3_batch1/R23"
REPORT_DIR = ROOT / "reports/optimization_v3/batch1/R23"
RECIPE_PATH = EXPERIMENT_DIR / "recipe.json"
FREEZE_PATH = EXPERIMENT_DIR / "freeze.json"
PILOT_PATH = RUN_DIR / "pilot.json"
RESULTS_PATH = REPORT_DIR / "results.json"
VERIFICATION_PATH = REPORT_DIR / "verification.json"
GATE_PATH = REPORT_DIR / "gate.json"
SUMMARY_PATH = REPORT_DIR / "summary.md"
LEDGER_PATH = RUN_DIR.parent / "batch_ledger.jsonl"
TRAIN_SHA256 = "756523ff8156a9cd776e52249f7e2a8f23dcec432b8d10570e55ff6db0e0f082"
QUEUE_SHA256 = "822dcd20fa28fd0e9ddc841a8c3709b372dacaca305b94b39b11a9203418fec7"
CARD_SHA256 = "5ca6354483f34bbbb9edb3b54a934e223b299f04467d28512166175cb4ac0fbc"
MANIFEST_SHA256 = "dd166e92089f24978244496a9a0afabc0d06551dbdc326579a7c898a8844e19a"
EXPECTED_CONTROL_METRICS = {"primary_mae": 13.464395831218011, "H12_mae": 14.04753258299653}
EXPECTED_FIT_ROWS = {55: 12_020, 67: 29_705}
EXPECTED_VIEW_ROWS = {
    "W1": 4_114, "W4": 4_950, "primary": 9_064,
    "H12_E": 17_685, "H12_L": 20_903, "H12": 38_588,
}
THREADS = 2
MAX_SUCCESSFUL_FITS = 180
MIN_GROUP_SUPPORT = 100
JOB_CAP_BYTES = 6 * 1024**3
FREE_RAM_RESERVE_BYTES = 2 * 1024**3
EXPECTED_GATE_KEYS = (
    "pilot_primary_MAE_gain_min", "pilot_H12_MAE_harm_max",
    "individual_pilot_view_MAE_harm_max", "supported_group_n_min",
    "supported_tail_or_missing_MAE_harm_max",
)
EXPECTED_CARD = {
    "id": "R23", "task": "regression", "name": "ExtraTrees median-leaf ensemble",
    "family": "random_forest", "priority": "P2",
    "change": "Structured scaled numeric/OHE+SVD32; ExtraTreesRegressor200trees/criterion absolute_error/min_samples_leaf30/max_features1.0/random_state42/n_jobs2. OHE categories, no arbitrary ordinal distances. This averages leaf medians, not a quantile-regression forest claim.",
    "policy": "R_AUX25", "pilot_predictor_fits": 2, "evaluation": "STANDARD",
    "condition": "No additional dependency",
    "preflight_checks": ["Absolute-error trees can be slow; first-fit time budget and matrix RAM guard"],
    "evidence": ["reports/optimization_v3/regression_diagnostics/diagnostics.json"],
    "status": "PLANNED_NOT_RUN", "implementation": "NOT_IMPLEMENTED", "new_results": None,
    "queue_order": 28, "rationale": "Pohon acak memberi smoothing berbeda dari boosting; criterion median tidak otomatis menjadikannya quantile forest.",
    "card_path": "reports/optimization_v3/research_backlog/cards/R23.md",
    "in_priority_prefix": False,
}


def _clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return "NaN (estimator default)" if np.isnan(value) else str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _write_or_match(path: Path, payload: Any) -> None:
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise BatchIntegrityError(f"Existing R23 artifact differs; refusing overwrite: {path}")
    else:
        write_json_exclusive(path, payload)


def _validate_card_queue(queue: Mapping[str, Any], card: Mapping[str, Any]) -> None:
    if file_sha256(QUEUE_PATH) != QUEUE_SHA256 or file_sha256(CARD_PATH) != CARD_SHA256:
        raise BatchIntegrityError("R23 canonical queue/card bytes differ from reviewed recipe")
    if load_queue(QUEUE_PATH) != dict(queue):
        raise BatchIntegrityError("R23 dispatcher queue differs from canonical queue")
    canonical = next((item for item in queue["cards"] if item.get("id") == CARD_ID), None)
    if canonical is None or canonical != dict(card):
        raise BatchIntegrityError("R23 queue card differs from canonical entry")
    for key, expected in EXPECTED_CARD.items():
        if card.get(key) != expected:
            raise BatchIntegrityError(f"R23 canonical card differs at {key}")
    if tuple(FEATURE_NAMES) != tuple(HYBRID_FEATURES):
        raise BatchIntegrityError("R23 input feature allowlist changed")
    if set(FEATURE_NAMES) & {"record_id", "artifact_signal", "threat_rank"}:
        raise BatchIntegrityError("R23 feature allowlist contains identity or target fields")
    baseline = queue["baseline"]["regression"]
    if (baseline.get("Train_sha256") != TRAIN_SHA256
            or baseline.get("manifest_sha256") != MANIFEST_SHA256
            or float(baseline.get("primary", float("nan"))) != 11.559390078665364
            or float(baseline.get("H12", float("nan"))) != 14.04753258299653):
        raise BatchIntegrityError("R23 frozen champion identity or baseline metrics changed")
    gates = queue["evaluation"]["regression_gate"]
    for key in EXPECTED_GATE_KEYS:
        if key not in gates or not np.isfinite(float(gates[key])):
            raise BatchIntegrityError(f"R23 canonical regression gate missing/invalid at {key}")
    if int(card["pilot_predictor_fits"]) != 2:
        raise BatchIntegrityError("R23 must reserve exactly two pilot predictor fits")


def _read_train() -> tuple[Path, pd.DataFrame, np.ndarray, np.ndarray]:
    """Resolve and read only the official Regression Train file."""
    train_path = task_files("regression").get("train")
    if train_path is None or not Path(train_path).is_file():
        raise FileNotFoundError("Official Regression Train is unavailable")
    train_path = Path(train_path)
    if file_sha256(train_path) != TRAIN_SHA256:
        raise BatchIntegrityError("Official Regression Train hash differs from R23 card")
    frame = pd.read_csv(train_path)
    required = {"record_id", "threat_rank", *FEATURE_NAMES}
    if required - set(frame.columns) or len(frame) != 50_608:
        raise BatchIntegrityError("Official Regression Train schema/row count differs from R23 contract")
    ids = frame["record_id"].astype(str)
    if ids.isna().any() or ids.str.strip().eq("").any() or ids.duplicated().any():
        raise BatchIntegrityError("Official Regression Train record_id is missing or duplicated")
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    if not np.array_equal(np.unique(cycles), np.arange(47, 80, dtype=np.int64)):
        raise BatchIntegrityError("Official Regression Train cycle coverage differs from 47-79")
    target = pd.to_numeric(frame["threat_rank"], errors="raise").to_numpy(dtype=np.float64)
    if (target.shape != (len(frame),) or not np.isfinite(target).all()
            or np.any((target < 0.0) | (target > 100.0))):
        raise BatchIntegrityError("Official Regression Train threat_rank is invalid")
    if file_sha256(train_path) != TRAIN_SHA256:
        raise BatchIntegrityError("Official Regression Train changed during read")
    if not set(FEATURE_NAMES).issubset(set(frame.columns)):
        raise BatchIntegrityError("Official Train does not contain the complete R23 feature allowlist")
    return train_path, frame, target, cycles


def _input_files(train_path: Path) -> dict[str, Path]:
    return {
        "official_regression_train": train_path,
        "canonical_queue": QUEUE_PATH,
        "canonical_card": CARD_PATH,
        "accepted_regression_manifest": r01.ACCEPTED_MANIFEST,
        "v2_paired_lgb_ridge_cache": r01.BASE_CACHE,
        "v2_sparse_pilot_cache": r01.SPARSE_CACHE,
        "v3_champion_diagnostic_cache": r01.DIAGNOSTIC_CACHE,
        "v3_regression_diagnostics": r01.DIAGNOSTICS_REPORT,
        "v2_regression_fullsix_report": r01.FULLSIX_REPORT,
        "v2_regression_findings": r01.FINDINGS_REPORT,
        "v3_read_only_feasibility": ROOT / "reports/optimization_v3/research_backlog/feasibility.json",
    }


def _source_files() -> list[Path]:
    return [
        Path(__file__), ROOT / "src/models/optimization_v3_batch1_r23.py",
        ROOT / "tests/test_optimization_v3_batch1_r23.py",
        ROOT / "src/models/optimization_v3_batch1.py",
        ROOT / "scripts/optimization_v3_batch1.py",
        ROOT / "scripts/optimization_v3_batch1_verify.py",
        ROOT / "scripts/optimization_v3_batch1_r01.py",
        ROOT / "src/models/optimization_v3_batch1_r01.py",
        ROOT / "src/models/regression_research_text.py",
        ROOT / "src/models/regression_research_lgb_text.py",
        ROOT / "src/models/regression_round2_models.py",
        ROOT / "src/models/round1_models.py",
        ROOT / "scripts/optimization_v3_reg_diagnostics.py",
        ROOT / "src/utils/paths.py",
    ]


def _resource_preflight(fit_rows: int, text_support: Mapping[str, Any],
                        available_bytes: int | None = None) -> dict[str, Any]:
    # 807 structured columns is a conservative cap from the inherited rare-OHE
    # contract; add 32 SVD values and the missing-stream flag. Bound transformed
    # buffers using the larger fit/score population, observed fit/score sparse
    # TF-IDF buffers, and tree nodes bounded by min_samples_leaf=30 with a
    # conservative 256 bytes per node. This remains an estimate rather than RSS.
    width = 807 + 32 + 1
    score_rows = int(text_support["score_rows"])
    matrix_rows = max(int(fit_rows), int(score_rows))
    design_bytes = matrix_rows * width * 8
    max_nodes_per_tree = max(1, 2 * (int(fit_rows) // EXTRA_TREES_PARAMETERS["min_samples_leaf"]) - 1)
    tree_node_allowance = int(EXTRA_TREES_PARAMETERS["n_estimators"]) * max_nodes_per_tree * 256
    median_split_workspace = int(fit_rows) * 16 * THREADS
    sparse_text_bytes = int(text_support["fit_tfidf_csr_bytes"]) + int(text_support["score_tfidf_csr_bytes"])
    overhead_bytes = 2 * 1024**3
    estimate = 5 * design_bytes + sparse_text_bytes + tree_node_allowance + median_split_workspace + overhead_bytes
    if available_bytes is None:
        try:
            import psutil
            available_bytes = int(psutil.virtual_memory().available)
        except Exception:
            available_bytes = None
    allowed = None if available_bytes is None else min(
        JOB_CAP_BYTES, max(0, int(available_bytes) - FREE_RAM_RESERVE_BYTES)
    )
    status = "UNKNOWN_RESOURCE" if allowed is None else (
        "PASS" if estimate <= allowed else "BLOCKED_RESOURCE")
    return {
        "status": status, "fit_rows": int(fit_rows), "score_rows": int(score_rows),
        "matrix_rows_upper_bound": int(matrix_rows),
        "feature_width_upper_bound": width, "design_bytes_upper_bound": design_bytes,
        "estimated_peak_bytes": estimate, "estimated_peak_is_observed": False,
        "fit_score_tfidf_sparse_bytes_observed": sparse_text_bytes,
        "extra_trees_node_allowance_bytes": tree_node_allowance,
        "median_split_workspace_allowance_bytes": median_split_workspace,
        "transform_fixed_overhead_bytes": overhead_bytes,
        "available_physical_bytes": available_bytes,
        "free_memory_reserve_bytes": FREE_RAM_RESERVE_BYTES,
        "job_cap_bytes": JOB_CAP_BYTES, "allowed_bytes": allowed,
        "threads": THREADS, "concurrent_training_jobs": 1,
        "dense_structured_svd_design": True,
        "sparse_tfidf_is_never_densified": True,
        "first_fit_watch_threshold_seconds": 900,
        "hard_fit_timeout_seconds": None,
    }


def _rank50_independent(raw: np.ndarray, cycles: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    cycles = np.asarray(cycles, dtype=np.int64)
    if raw.shape != cycles.shape or raw.ndim != 1 or raw.size == 0 or not np.isfinite(raw).all():
        raise BatchIntegrityError("R23 verifier rank inputs are invalid")
    out = np.empty(raw.shape, dtype=np.float64)
    for cycle in np.unique(cycles):
        mask = cycles == cycle
        percentile = 100.0 * rankdata(raw[mask], method="average") / int(mask.sum())
        out[mask] = 0.5 * raw[mask] + 0.5 * percentile
    return np.clip(out, 0.0, 100.0)


def reconstruct_r23_arrays(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Independently rebuild incumbent and fixed challenger route from saved arrays."""
    raw = np.asarray(arrays["raw_candidate"], dtype=np.float64)
    cycles = np.asarray(arrays["audit__score_cycles"], dtype=np.int64)
    lgb = np.asarray(arrays["control__baseline_lgb_svd32"], dtype=np.float64)
    ridge = np.asarray(arrays["control__baseline_hybrid_ridge"], dtype=np.float64)
    sparse_raw = np.asarray(arrays["control__baseline_sparse_raw"], dtype=np.float64)
    sparse_rank_saved = np.asarray(arrays["control__baseline_sparse_rank50"], dtype=np.float64)
    saved_baseline = np.asarray(arrays["baseline_prediction"], dtype=np.float64)
    if len({x.shape for x in (raw, cycles, lgb, ridge, sparse_raw, sparse_rank_saved, saved_baseline)}) != 1:
        raise BatchIntegrityError("R23 saved route arrays do not align")
    sparse_rank = _rank50_independent(sparse_raw, cycles)
    if not np.array_equal(sparse_rank, sparse_rank_saved):
        raise BatchIntegrityError("R23 incumbent sparse rank50 does not reconstruct")
    baseline = np.clip(0.75 * (0.75 * lgb + 0.25 * ridge) + 0.25 * sparse_rank, 0.0, 100.0)
    if not np.array_equal(baseline, saved_baseline):
        raise BatchIntegrityError("R23 three-component champion baseline does not reconstruct")
    candidate = np.clip(0.75 * baseline + 0.25 * _rank50_independent(raw, cycles), 0.0, 100.0)
    return candidate, baseline


def _mae(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if y.shape != pred.shape or y.ndim != 1 or not len(y):
        raise BatchIntegrityError("R23 metric inputs must be aligned nonempty vectors")
    if not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise BatchIntegrityError("R23 metric inputs contain nonfinite values")
    return float(np.abs(y - pred).sum(dtype=np.float64) / len(y))


def _view_arrays(context: Mapping[str, Any], per_cutoff: Mapping[int, Mapping[str, np.ndarray]]) -> dict[str, dict[str, np.ndarray]]:
    cycles = np.asarray(context["cycles"], dtype=np.int64)
    result: dict[str, dict[str, np.ndarray]] = {}
    for name, segments in context["views"].items():
        rows_parts: list[np.ndarray] = []
        candidate_parts: list[np.ndarray] = []
        baseline_parts: list[np.ndarray] = []
        for cutoff, low, high in segments:
            fold = per_cutoff[int(cutoff)]
            mask = (cycles[fold["rows"]] >= int(low)) & (cycles[fold["rows"]] <= int(high))
            rows_parts.append(fold["rows"][mask])
            candidate_parts.append(fold["candidate"][mask])
            baseline_parts.append(fold["baseline"][mask])
        rows = np.concatenate(rows_parts)
        if len(np.unique(rows)) != len(rows) or len(rows) != EXPECTED_VIEW_ROWS[name]:
            raise BatchIntegrityError(f"R23 view denominator/uniqueness differs for {name}")
        result[name] = {"rows": rows, "candidate": np.concatenate(candidate_parts),
                        "baseline": np.concatenate(baseline_parts)}
    for name, cutoff, low, high in (("H12_E", 55, 56, 67), ("H12_L", 67, 68, 79)):
        fold = per_cutoff[cutoff]
        rows = np.asarray(fold["rows"], dtype=np.int64)
        selected = (cycles[rows] >= low) & (cycles[rows] <= high)
        group = {"rows": rows[selected],
                 "candidate": np.asarray(fold["candidate"])[selected],
                 "baseline": np.asarray(fold["baseline"])[selected]}
        if len(np.unique(group["rows"])) != len(group["rows"]) or len(group["rows"]) != EXPECTED_VIEW_ROWS[name]:
            raise BatchIntegrityError(f"R23 early/late H12 denominator differs for {name}")
        result[name] = group
    return result


def _group_diagnostics(frame: pd.DataFrame, target: np.ndarray,
                       views: Mapping[str, Mapping[str, np.ndarray]]) -> dict[str, Any]:
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    missing_all = frame["signal_stream"].isna().to_numpy() | frame["signal_stream"].fillna("").astype(str).str.strip().eq("").to_numpy()
    result: dict[str, Any] = {"support_minimum": MIN_GROUP_SUPPORT, "views": {}}
    for name, source in views.items():
        rows = np.asarray(source["rows"], dtype=np.int64)
        y = np.asarray(target[rows], dtype=np.float64)
        candidate = np.asarray(source["candidate"], dtype=np.float64)
        baseline = np.asarray(source["baseline"], dtype=np.float64)
        masks = {
            "signal_stream_missing": missing_all[rows],
            "target_le_5": y <= 5.0,
            "target_ge_95": y >= 95.0,
        }
        result["views"][name] = {}
        for label, mask in masks.items():
            n = int(mask.sum())
            supported = n >= MIN_GROUP_SUPPORT
            cand_mae = _mae(y[mask], candidate[mask]) if n else None
            base_mae = _mae(y[mask], baseline[mask]) if n else None
            result["views"][name][label] = {
                "n": n, "supported": supported,
                "candidate_mae": cand_mae, "baseline_mae": base_mae,
                "candidate_minus_baseline_mae": (cand_mae - base_mae) if n else None,
                "guard_applies": supported,
            }
    return result


def _fit_execution_accounting(validated_fit_count: int) -> dict[str, Any]:
    """Report every R23 attempt, including failed calls from an earlier freeze."""
    selected: list[dict[str, Any]] = []
    if LEDGER_PATH.is_file():
        for line_number, line in enumerate(LEDGER_PATH.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BatchIntegrityError(f"Invalid fit-ledger JSON on line {line_number}") from exc
            if event.get("card_id") == CARD_ID and event.get("event") in {
                "FIT_STARTED", "FIT_SAVED", "FIT_FAILED", "RESUMED_VERIFIED",
            }:
                selected.append(event)
    starts = [item for item in selected if item.get("event") == "FIT_STARTED"]
    saved = [item for item in selected if item.get("event") == "FIT_SAVED"]
    failed = [item for item in selected if item.get("event") == "FIT_FAILED"]
    return {
        "reserved_predictor_fits": 2,
        "successful_predictor_fits": int(validated_fit_count),
        "adapter_attempts_started": len(starts),
        "fit_saved_events": len(saved),
        "failed_adapter_attempts": len(failed),
        "attempt_event_pairs_complete": len(starts) == len(saved) + len(failed),
        "total_recorded_attempt_seconds": float(sum(float(item.get("fit_seconds", 0.0))
                                                     for item in saved + failed)),
        "attempts": [
            {key: item.get(key) for key in (
                "event", "cutoff", "freeze_signature", "fit_rows", "score_rows",
                "fit_predictor_count", "fit_seconds", "error_type", "error",
            ) if key in item}
            for item in selected
        ],
        "has_preserved_failed_prefit_attempt": bool(failed),
    }


def _recipe(card: Mapping[str, Any], views: Mapping[str, Any], card_sha: str) -> dict[str, Any]:
    import sklearn
    from sklearn.ensemble import ExtraTreesRegressor
    from src.models.regression_research_text import CATEGORICAL_FEATURES, STRUCTURED_FEATURES

    effective = _clean(ExtraTreesRegressor(**dict(EXTRA_TREES_PARAMETERS)).get_params(deep=False))
    return _clean({
        "card_id": CARD_ID, "status": "FROZEN_RECIPE_BEFORE_PILOT",
        "card_sha256": card_sha, "canonical_queue_sha256": QUEUE_SHA256,
        "accepted_control": {
            "manifest": str(r01.ACCEPTED_MANIFEST.relative_to(ROOT)),
            "manifest_sha256": MANIFEST_SHA256,
            "route": "Current three-component champion: .5625 rank50 LightGBM SVD32 + .1875 rank75 Hybrid Ridge + .25 rank50 sparse LightGBM",
            "pilot_control_is_exact_three_component_incumbent": True,
        },
        "feature_allowlist": list(FEATURE_NAMES),
        "feature_roles": {"record_id": "identity_only", "threat_rank": "target_only"},
        "preprocessing": {
            "structured_numeric": list(STRUCTURED_FEATURES),
            "categorical_features": list(CATEGORICAL_FEATURES),
            "structured_recipe": "RegressionTextModel._structured_fit/_structured_transform; numeric median imputation, StandardScaler, OHE min_frequency=30/max_categories=254/handle_unknown=infrequent_if_exist",
            "text_tfidf_parameters": _clean(_TFIDF_PARAMS),
            "text_fit_scope": "all rows with cycle<=cutoff only",
            "svd_parameters": {**_clean(_SVD_PARAMS), "n_components_requested": 32,
                               "required_fit_tfidf_columns": 33, "component_count_may_not_shrink": True},
            "svd_fit_scope": "all rows with cycle<=cutoff only",
            "missing_signal_indicator": True,
            "dense_matrix_dtype": "float64 at ExtraTreesRegressor input",
            "text_tfidf_sparse_matrix_densified": False,
            "score_rows_or_labels_used_to_fit_transform": False,
        },
        "estimator": {
            "class": "sklearn.ensemble.ExtraTreesRegressor",
            "parameters_explicit": dict(EXTRA_TREES_PARAMETERS),
            "parameters_effective_including_library_defaults": effective,
            "scikit_learn_version": sklearn.__version__,
            "criterion_interpretation": "absolute_error selects split quality by L1 reduction; leaf outputs are medians, which does not make this a quantile-regression forest",
            "early_stopping": False,
            "evaluation_set": None,
        },
        "prediction_policy": ".75 exact current three-component champion + .25 rank50(new raw target-scale prediction), clip[0,100]; rank50 uses 50% raw and 50% average-rank percentile computed separately within each complete 12-cycle scoring cycle",
        "evaluation_views": views,
        "expected_denominators": EXPECTED_VIEW_ROWS,
        "supported_group_minimum": MIN_GROUP_SUPPORT,
        "pilot_fit_rows": EXPECTED_FIT_ROWS,
        "pilot_predictor_fit_count": int(card["pilot_predictor_fits"]),
        "metric": "MAE; positive gain = baseline MAE - candidate MAE",
        "gates": {key: float(load_queue(QUEUE_PATH)["evaluation"]["regression_gate"][key])
                  for key in EXPECTED_GATE_KEYS},
        "official_test_read": False,
    })


def _budget_status() -> dict[str, Any]:
    reserved: dict[str, int] = {}
    if LEDGER_PATH.exists():
        for line_number, line in enumerate(LEDGER_PATH.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                if event.get("event") == "BUDGET_RESERVED":
                    key = str(event["budget_key"])
                    count = int(event["predictor_fits"])
                    if count < 0 or key in reserved and reserved[key] != count:
                        raise ValueError("negative or conflicting reservation")
                    reserved[key] = count
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise BatchIntegrityError(f"Invalid shared fit-budget ledger line {line_number}: {exc}") from exc
    total = int(sum(reserved.values()))
    own = reserved.get("R23:PILOT")
    next_total = total if own == 2 else total + 2
    status = "PASS" if own in (None, 2) and next_total <= MAX_SUCCESSFUL_FITS else "BLOCKED_BUDGET"
    return {"status": status, "reserved_total": total, "next_total_if_started": next_total,
            "existing_r23_reservation": own, "cap": MAX_SUCCESSFUL_FITS,
            "reservation_count_by_key": reserved}


def _prepare(card: Mapping[str, Any], queue: Mapping[str, Any]) -> dict[str, Any]:
    _validate_card_queue(queue, card)
    r01._validate_manifest(dict(queue))
    train_path, frame, target, cycles = _read_train()
    views = evaluation_views_for_card(queue, CARD_ID)
    base = r01._read_npz(r01.BASE_CACHE)
    sparse_cache = r01._read_npz(r01.SPARSE_CACHE)
    diagnostic = r01._read_npz(r01.DIAGNOSTIC_CACHE)
    controls_loader = r01._load_controls(frame, base, sparse_cache, diagnostic)
    controls: dict[int, dict[str, np.ndarray]] = {}
    for cutoff in (55, 67):
        rows = np.flatnonzero((cycles > cutoff) & (cycles <= cutoff + 12))
        if len(np.flatnonzero(cycles <= cutoff)) != EXPECTED_FIT_ROWS[cutoff]:
            raise BatchIntegrityError(f"R23 fit-row denominator differs at cutoff {cutoff}")
        values = controls_loader(cutoff, rows)
        baseline = np.asarray(values["baseline_prediction"], dtype=np.float64)
        rebuilt = np.clip(
            0.75 * (0.75 * np.asarray(values["baseline_lgb_svd32"], dtype=np.float64)
                    + 0.25 * np.asarray(values["baseline_hybrid_ridge"], dtype=np.float64))
            + 0.25 * np.asarray(values["baseline_sparse_rank50"], dtype=np.float64), 0.0, 100.0
        )
        if not np.array_equal(rebuilt, baseline):
            raise BatchIntegrityError(f"R23 current champion does not reconstruct exactly at cutoff {cutoff}")
        controls[cutoff] = {
            "score_rows": rows.astype(np.int64),
            "score_record_ids": frame.iloc[rows]["record_id"].astype(str).to_numpy(dtype=str),
            "baseline_prediction": baseline,
            "baseline_lgb_svd32": np.asarray(values["baseline_lgb_svd32"], dtype=np.float64),
            "baseline_hybrid_ridge": np.asarray(values["baseline_hybrid_ridge"], dtype=np.float64),
            "baseline_sparse_raw": np.asarray(values["baseline_sparse_raw"], dtype=np.float64),
            "baseline_sparse_rank50": np.asarray(values["baseline_sparse_rank50"], dtype=np.float64),
        }
        if len(np.unique(controls[cutoff]["score_record_ids"])) != len(rows):
            raise BatchIntegrityError(f"R23 control IDs are duplicated at cutoff {cutoff}")

    view_denominators: dict[str, int] = {}
    for name, segments in views.items():
        rows = np.concatenate([
            np.flatnonzero((cycles >= low) & (cycles <= high)) for _cutoff, low, high in segments
        ])
        if len(np.unique(rows)) != len(rows) or len(rows) != EXPECTED_VIEW_ROWS[name]:
            raise BatchIntegrityError(f"R23 fixed view denominator differs for {name}")
        view_denominators[name] = int(len(rows))

    control_views: dict[str, dict[str, Any]] = {}
    per_cutoff = {cutoff: {"rows": item["score_rows"], "candidate": item["baseline_prediction"],
                           "baseline": item["baseline_prediction"]}
                  for cutoff, item in controls.items()}
    cycles_frame = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    for name, segments in views.items():
        rows_parts: list[np.ndarray] = []
        pred_parts: list[np.ndarray] = []
        for cutoff, low, high in segments:
            fold = controls[int(cutoff)]
            mask = (cycles_frame[fold["score_rows"]] >= int(low)) & (cycles_frame[fold["score_rows"]] <= int(high))
            rows_parts.append(fold["score_rows"][mask])
            pred_parts.append(fold["baseline_prediction"][mask])
        rows = np.concatenate(rows_parts)
        prediction = np.concatenate(pred_parts)
        control_views[name] = {"rows": rows, "mae": _mae(target[rows], prediction), "rows_count": len(rows)}
    if (abs(control_views["primary"]["mae"] - EXPECTED_CONTROL_METRICS["primary_mae"]) > 1e-10
            or abs(control_views["H12"]["mae"] - EXPECTED_CONTROL_METRICS["H12_mae"]) > 1e-10):
        raise BatchIntegrityError("R23 current three-component champion metrics differ from row-matched control")

    # Explicitly record the support denominators before any candidate score exists.
    group_support: dict[str, Any] = {}
    missing = frame["signal_stream"].isna().to_numpy() | frame["signal_stream"].fillna("").astype(str).str.strip().eq("").to_numpy()
    for name, source in control_views.items():
        rows = np.asarray(source["rows"], dtype=np.int64)
        y = target[rows]
        group_support[name] = {
            "signal_stream_missing": int(missing[rows].sum()),
            "target_le_5": int(np.sum(y <= 5.0)),
            "target_ge_95": int(np.sum(y >= 95.0)),
        }
    for name, cutoff, low, high in (("H12_E", 55, 56, 67), ("H12_L", 67, 68, 79)):
        rows = np.flatnonzero((cycles_frame >= low) & (cycles_frame <= high))
        y = target[rows]
        group_support[name] = {"signal_stream_missing": int(missing[rows].sum()),
                               "target_le_5": int(np.sum(y <= 5.0)),
                               "target_ge_95": int(np.sum(y >= 95.0))}

    text_support: dict[int, dict[str, Any]] = {}
    resources: dict[str, dict[str, Any]] = {}
    for cutoff in (55, 67):
        fit_rows = np.flatnonzero(cycles <= cutoff)
        score_rows = np.flatnonzero((cycles > cutoff) & (cycles <= cutoff + 12))
        support = text_svd32_support(
            frame.iloc[fit_rows].loc[:, list(FEATURE_NAMES)],
            frame.iloc[score_rows].loc[:, list(FEATURE_NAMES)],
        )
        if (int(support["fit_rows"]) != EXPECTED_FIT_ROWS[cutoff]
                or int(support["score_rows"]) != len(score_rows)):
            raise BatchIntegrityError(f"R23 fit-only text support rows differ at cutoff {cutoff}")
        text_support[cutoff] = support
        resources[str(cutoff)] = _resource_preflight(EXPECTED_FIT_ROWS[cutoff], support)
    budget = _budget_status()
    if any(not item["exact_svd32_supported"] for item in text_support.values()):
        preflight_status = "SKIPPED_SUPPORT"
    elif budget["status"] != "PASS":
        preflight_status = "BLOCKED_BUDGET"
    elif any(item["status"] == "UNKNOWN_RESOURCE" for item in resources.values()):
        preflight_status = "BLOCKED_RESOURCE_UNKNOWN"
    elif any(item["status"] != "PASS" for item in resources.values()):
        preflight_status = "BLOCKED_RESOURCE"
    else:
        preflight_status = "PREFLIGHT_PASS"

    input_files = _input_files(train_path)
    source_files = _source_files()
    missing_sources = [str(path) for path in [*input_files.values(), *source_files] if not Path(path).is_file()]
    if missing_sources:
        raise FileNotFoundError(f"R23 frozen source/input files are missing: {missing_sources}")
    input_hashes = {name: file_sha256(path) for name, path in sorted(input_files.items())}
    expected_hashes = {
        "official_regression_train": TRAIN_SHA256,
        "canonical_queue": QUEUE_SHA256,
        "canonical_card": CARD_SHA256,
        "accepted_regression_manifest": MANIFEST_SHA256,
        "v2_paired_lgb_ridge_cache": r01.EXPECTED_BASE_CACHE_SHA256,
        "v2_sparse_pilot_cache": r01.EXPECTED_SPARSE_CACHE_SHA256,
        "v3_champion_diagnostic_cache": r01.EXPECTED_DIAGNOSTIC_CACHE_SHA256,
        "v3_regression_diagnostics": r01.EXPECTED_DIAGNOSTICS_SHA256,
    }
    for name, expected in expected_hashes.items():
        if input_hashes.get(name) != expected:
            raise BatchIntegrityError(f"R23 frozen input hash differs for {name}: {input_hashes.get(name)}")
    views_for_identity: dict[str, Any] = dict(views)
    views_for_identity["H12_E"] = [(55, 56, 67)]
    views_for_identity["H12_L"] = [(67, 68, 79)]
    recipe = _recipe(card, views_for_identity, CARD_SHA256)
    resource_policy_identity = {
        str(cutoff): {
            "fit_rows": EXPECTED_FIT_ROWS[cutoff],
            "feature_width_upper_bound": item["feature_width_upper_bound"],
            "estimated_peak_bytes": item["estimated_peak_bytes"],
            "estimated_peak_is_observed": False,
            "fit_score_tfidf_sparse_bytes_observed": item["fit_score_tfidf_sparse_bytes_observed"],
            "extra_trees_node_allowance_bytes": item["extra_trees_node_allowance_bytes"],
            "median_split_workspace_allowance_bytes": item["median_split_workspace_allowance_bytes"],
            "fixed_overhead_bytes": item["transform_fixed_overhead_bytes"],
            "matrix_rows_upper_bound": item["matrix_rows_upper_bound"],
            "free_memory_reserve_bytes": FREE_RAM_RESERVE_BYTES,
            "job_cap_bytes": JOB_CAP_BYTES,
            "threads": THREADS,
            "concurrent_training_jobs": 1,
        }
        for cutoff, item in ((55, resources["55"]), (67, resources["67"]))
    }
    identity = make_freeze_identity(
        card=card, feature_names=FEATURE_NAMES, input_files=input_files,
        source_files=source_files, evaluation_views=views_for_identity,
        packages=["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl", "psutil"],
    )
    identity.update({
        "control_cache_hashes": {
            "paired_lgb_ridge": r01.EXPECTED_BASE_CACHE_SHA256,
            "sparse_pilot": r01.EXPECTED_SPARSE_CACHE_SHA256,
            "independent_champion_diagnostic": r01.EXPECTED_DIAGNOSTIC_CACHE_SHA256,
        },
        "exact_champion_reconstruction": {
            "formula": "clip(.75*(.75*rank50(SVD32_LGB)+.25*rank75(Hybrid_Ridge))+.25*rank50(sparse_LGB),0,100)",
            "rows_reconstructed": {str(cutoff): int(len(controls[cutoff]["score_rows"])) for cutoff in (55, 67)},
            "metrics": {name: {"rows": int(item["rows_count"]), "mae": item["mae"]}
                        for name, item in control_views.items()},
        },
        "pre_freeze_support_denominators": group_support,
        "fit_only_svd32_vocabulary_support": {str(k): v for k, v in text_support.items()},
        "resource_preflight_policy": resource_policy_identity,
        "candidate_fit_budget_policy": {
            "pilot_predictor_fits": int(card["pilot_predictor_fits"]),
            "global_successful_fit_cap": MAX_SUCCESSFUL_FITS,
            "reservation_before_first_fit": True,
        },
        "python_version": platform.python_version(),
        "official_test_read": False,
        "validation_labels_passed_to_adapter": False,
    })
    public = {
        "status": preflight_status, "card_id": CARD_ID,
        "train_sha256": TRAIN_SHA256, "input_sha256": input_hashes,
        "source_sha256": {str(path.relative_to(ROOT)): file_sha256(path) for path in source_files},
        "fit_rows": EXPECTED_FIT_ROWS, "view_denominators": view_denominators,
        "control_metrics": {name: {"rows": item["rows_count"], "mae": item["mae"]}
                            for name, item in control_views.items()},
        "pre_freeze_group_support_denominators": group_support,
        "fit_only_svd32_vocabulary_support": {str(k): v for k, v in text_support.items()},
        "resource_preflight": resources, "fit_budget_preflight": budget,
        "freeze_identity_sha256": json_sha256(identity),
        "official_test_read": False, "candidate_fit_performed": False,
        "freeze_or_output_written": False,
    }
    return {
        "train_path": train_path, "frame": frame, "target": target, "cycles": cycles,
        "views": views, "views_for_identity": views_for_identity,
        "controls": controls, "identity": identity, "recipe": recipe,
        "public": public, "resource_preflight": resources,
        "budget_preflight": budget, "group_support": group_support,
        "input_hashes": input_hashes,
        "text_support": text_support,
        "preflight_status": preflight_status,
    }


def _control_loader(context: Mapping[str, Any]):
    def load(cutoff: int, score_rows: np.ndarray) -> dict[str, np.ndarray]:
        control = context["controls"][int(cutoff)]
        rows = np.asarray(score_rows, dtype=np.int64)
        if not np.array_equal(rows, control["score_rows"]):
            raise BatchIntegrityError(f"R23 control cache rows differ at cutoff {cutoff}")
        return {
            "baseline_prediction": control["baseline_prediction"],
            "baseline_lgb_svd32": control["baseline_lgb_svd32"],
            "baseline_hybrid_ridge": control["baseline_hybrid_ridge"],
            "baseline_sparse_raw": control["baseline_sparse_raw"],
            "baseline_sparse_rank50": control["baseline_sparse_rank50"],
            "control_rows": rows,
        }
    return load


def _verify_pilot(context: Mapping[str, Any], report: Mapping[str, Any]) -> dict[str, Any]:
    signature = json_sha256(context["identity"])
    if (report.get("card_id") != CARD_ID or report.get("task") != "regression"
            or report.get("status") != "PILOT_SAVED_AWAITING_INDEPENDENT_QA"
            or report.get("freeze_signature") != signature
            or report.get("cutoffs") != [55, 67]
            or report.get("fit_predictor_count") != 2
            or report.get("fit_count_matches_card") is not True):
        raise BatchIntegrityError("R23 generic pilot report identity/count is invalid")
    frame = context["frame"]
    target = np.asarray(context["target"], dtype=np.float64)
    cycles = np.asarray(context["cycles"], dtype=np.int64)
    ids = frame["record_id"].astype(str).to_numpy(dtype=str)
    per_cutoff: dict[int, dict[str, np.ndarray]] = {}
    for cutoff in (55, 67):
        controls = context["controls"][cutoff]
        rows = controls["score_rows"]
        identity = {"freeze_signature": signature, "card_id": CARD_ID,
                    "cutoff": cutoff, "score_rows": rows.tolist()}
        arrays, sidecar = read_bundle(RUN_DIR / f"R23_fit{cutoff}.npz", identity)
        if (not np.array_equal(arrays["score_rows"], rows)
                or not np.array_equal(arrays["score_record_ids"], ids[rows])
                or len(np.unique(arrays["score_record_ids"])) != len(rows)):
            raise BatchIntegrityError(f"R23 prediction row/record_id alignment differs at {cutoff}")
        candidate, baseline = reconstruct_r23_arrays(arrays)
        if (not np.array_equal(candidate, arrays["prediction"])
                or not np.array_equal(baseline, arrays["baseline_prediction"])
                or not np.array_equal(arrays["audit__candidate_rank50"], _rank50_independent(arrays["raw_candidate"], cycles[rows]))):
            raise BatchIntegrityError(f"R23 fixed challenger policy does not reconstruct at {cutoff}")
        if (not np.isfinite(candidate).all() or np.any((candidate < 0) | (candidate > 100))
                or not np.isfinite(baseline).all() or np.any((baseline < 0) | (baseline > 100))):
            raise BatchIntegrityError(f"R23 candidate/baseline predictions outside [0,100] at {cutoff}")
        metadata = sidecar.get("metadata", {})
        state = metadata.get("state", {})
        if (int(metadata.get("fit_rows", -1)) != EXPECTED_FIT_ROWS[cutoff]
                or int(metadata.get("fit_predictor_count", -1)) != 1
                or state.get("card_id") != CARD_ID
                or state.get("fit_rows") != EXPECTED_FIT_ROWS[cutoff]
                or state.get("fit_cycle_max") != cutoff
                or state.get("feature_names") != list(FEATURE_NAMES)
                or state.get("estimator") != "sklearn.ensemble.ExtraTreesRegressor"
                or state.get("parameters_explicit") != dict(EXTRA_TREES_PARAMETERS)
                or int(state.get("svd", {}).get("n_components_requested", -1)) != 32
                or int(state.get("svd", {}).get("n_components_actual", -1)) != 32
                or int(state.get("tfidf", {}).get("vocabulary_size", 0)) < 33
                or state.get("validation_data_passed_to_fit") is not False
                or state.get("validation_labels_passed_to_fit") is not False
                or state.get("test_data_used_for_fit") is not False
                or state.get("early_stopping_used") is not False):
            raise BatchIntegrityError(f"R23 fit-only/source-bound state audit failed at {cutoff}")
        fit_rows = np.flatnonzero(cycles <= cutoff)
        if (len(fit_rows) != EXPECTED_FIT_ROWS[cutoff]
                or not np.array_equal(cycles[fit_rows].max(), cutoff)
                or state.get("fit_source_rows_sha256") != hashlib.sha256(
                    np.asarray(fit_rows, dtype="<i8").tobytes(order="C")).hexdigest()):
            raise BatchIntegrityError(f"R23 model fit-row identity differs at {cutoff}")
        expected_target_sha = _array_digest(target[fit_rows])
        expected_fit_data_sha = _json_digest({
            "features": list(FEATURE_NAMES),
            "row_positions": fit_rows.astype("<i8", copy=False).tolist(),
            "target_sha256": expected_target_sha,
        })
        state_without_hash = {key: value for key, value in state.items() if key != "state_sha256"}
        missing_all = frame["signal_stream"].isna().to_numpy() | frame["signal_stream"].fillna("").astype(str).str.strip().eq("").to_numpy()
        if (state.get("target_sha256") != expected_target_sha
                or state.get("fit_data_sha256") != expected_fit_data_sha
                or state.get("state_sha256") != _json_digest(state_without_hash)
                or int(state.get("fit_missing_stream_rows", -1)) != int(missing_all[fit_rows].sum())
                or int(state.get("transformed_width", -1)) != len(state.get("transformed_feature_names", []))):
            raise BatchIntegrityError(f"R23 serialized fitted-state fingerprints differ at {cutoff}")
        per_cutoff[cutoff] = {"rows": rows, "candidate": candidate,
                              "baseline": baseline, "raw": np.asarray(arrays["raw_candidate"], dtype=np.float64)}

    view_values = _view_arrays(context, per_cutoff)
    metrics: dict[str, Any] = {}
    for name, source in view_values.items():
        rows = source["rows"]
        candidate_mae = _mae(target[rows], source["candidate"])
        baseline_mae = _mae(target[rows], source["baseline"])
        gain = baseline_mae - candidate_mae
        if name in context["views"]:
            published = report["views"][name]
            if (int(published["rows"]) != len(rows)
                    or abs(float(published["candidate"]["mae"]) - candidate_mae) > 1e-10
                    or abs(float(published["baseline"]["mae"]) - baseline_mae) > 1e-10
                    or abs(float(published["gain_positive_is_better"]) - gain) > 1e-10):
                raise BatchIntegrityError(f"R23 independent metric aggregation differs for {name}")
        metrics[name] = {"rows": int(len(rows)), "candidate_mae": candidate_mae,
                         "baseline_mae": baseline_mae, "gain_positive_is_better": gain}
    groups = _group_diagnostics(frame, target, view_values)
    thresholds = context["queue"]["evaluation"]["regression_gate"] if "queue" in context else load_queue(QUEUE_PATH)["evaluation"]["regression_gate"]
    group_failures = []
    for view, fields in groups["views"].items():
        for field, record in fields.items():
            if record["supported"] and float(record["candidate_minus_baseline_mae"]) > float(thresholds["supported_tail_or_missing_MAE_harm_max"]):
                group_failures.append({"view": view, "field": field, **record})
    checks = {
        "pilot_primary_MAE_gain_min": metrics["primary"]["gain_positive_is_better"] >= float(thresholds["pilot_primary_MAE_gain_min"]),
        "pilot_H12_MAE_harm_max": metrics["H12"]["gain_positive_is_better"] >= -float(thresholds["pilot_H12_MAE_harm_max"]),
        "W1_MAE_harm_max": metrics["W1"]["gain_positive_is_better"] >= -float(thresholds["individual_pilot_view_MAE_harm_max"]),
        "W4_MAE_harm_max": metrics["W4"]["gain_positive_is_better"] >= -float(thresholds["individual_pilot_view_MAE_harm_max"]),
        "supported_tail_or_missing_harm_max": len(group_failures) == 0,
    }
    execution_accounting = _fit_execution_accounting(validated_fit_count=len(per_cutoff))
    if (execution_accounting["successful_predictor_fits"] != 2
            or not execution_accounting["attempt_event_pairs_complete"]):
        raise BatchIntegrityError("R23 successful fit/attempt ledger accounting is incomplete")
    decision = "ADVANCE" if all(checks.values()) else "RETIRE"
    return {
        "card_id": CARD_ID, "status": "PASS", "decision": decision,
        "freeze_signature": signature, "fit_predictor_count": 2,
        "cutoffs": [55, 67], "views": metrics,
        "group_support_minimum": MIN_GROUP_SUPPORT, "groups": groups,
        "supported_group_failures": group_failures,
        "fit_execution_accounting": execution_accounting,
        "gate_checks": checks, "gate_thresholds": {key: float(thresholds[key]) for key in EXPECTED_GATE_KEYS},
        "metric_recomputed_from_train_labels_after_prediction": True,
        "validation_labels_passed_to_adapter": False,
        "official_test_read": False, "kaggle_access": False,
    }


def _publish(context: Mapping[str, Any], report: Mapping[str, Any],
             verification: Mapping[str, Any]) -> None:
    gate = {
        "card_id": CARD_ID, "profile": "STANDARD", "status": verification["decision"],
        "checks": verification["gate_checks"], "thresholds": verification["gate_thresholds"],
        "supported_group_failures": verification["supported_group_failures"],
        "decision_basis": "Predeclared temporal metrics and support-passing group guards; no Test or Kaggle evidence.",
    }
    result = dict(report)
    result.update({
        "status": f"PILOT_COMPLETE_{verification['decision']}",
        "input_sha256": context["public"]["input_sha256"],
        "source_sha256": context["public"]["source_sha256"],
        "preflight": context["public"],
        "independent_verification": verification,
        "resource_note": "Dense structured+SVD design, observed sparse TF-IDF matrices, and bounded ExtraTrees node storage were preflight-estimated; peak RSS was not directly observed.",
        "fit_only_svd32_vocabulary_support": context["public"]["fit_only_svd32_vocabulary_support"],
        "first_fit_timing_note": "The 900-second threshold triggers progress/resource inspection only; it is not a hard timeout.",
        "test_data_used": False, "kaggle_access": False,
    })
    primary, h12 = verification["views"]["primary"], verification["views"]["H12"]
    summary = (
        "# R23 Pilot Summary\n\n"
        f"- Decision: **{verification['decision']}**\n"
        f"- Primary MAE: {primary['baseline_mae']:.6f} baseline → {primary['candidate_mae']:.6f} candidate; "
        f"gain {primary['gain_positive_is_better']:+.6f} on {primary['rows']:,} rows.\n"
        f"- H12 MAE: {h12['baseline_mae']:.6f} baseline → {h12['candidate_mae']:.6f} candidate; "
        f"gain {h12['gain_positive_is_better']:+.6f} on {h12['rows']:,} rows.\n"
        f"- W1 / W4 gains: {verification['views']['W1']['gain_positive_is_better']:+.6f} / "
        f"{verification['views']['W4']['gain_positive_is_better']:+.6f}.\n"
        "- Temporal development outcomes are reused, not an untouched holdout. Test/Sample/Kaggle were not read.\n"
    )
    _write_or_match(RESULTS_PATH, result)
    _write_or_match(VERIFICATION_PATH, verification)
    _write_or_match(GATE_PATH, gate)
    if SUMMARY_PATH.exists():
        if SUMMARY_PATH.read_text(encoding="utf-8") != summary:
            raise BatchIntegrityError("Existing R23 summary differs; refusing overwrite")
    else:
        SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with SUMMARY_PATH.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(summary)


def run_card(card: Mapping[str, Any], queue: Mapping[str, Any], *,
             preflight_only: bool = False, freeze_only: bool = False,
             run_frozen_pilot: bool = False) -> dict[str, Any]:
    if sum((preflight_only, freeze_only, run_frozen_pilot)) != 1:
        raise BatchIntegrityError("Choose exactly one R23 mode")
    context = _prepare(card, queue)
    if preflight_only:
        return context["public"]
    if context["preflight_status"] != "PREFLIGHT_PASS":
        raise BatchIntegrityError(
            f"R23 cannot freeze or fit because preflight status is {context['preflight_status']}; "
            "no recipe/freeze/predictor output was written"
        )
    if freeze_only:
        EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
        if RECIPE_PATH.exists():
            if _read_json(RECIPE_PATH) != context["recipe"]:
                raise BatchIntegrityError("Existing R23 recipe differs; refusing rewrite")
        else:
            write_json_exclusive(RECIPE_PATH, context["recipe"])
        if FREEZE_PATH.exists():
            frozen = _read_json(FREEZE_PATH)
            if frozen.get("identity") != context["identity"] or frozen.get("signature") != json_sha256(context["identity"]):
                raise BatchIntegrityError("Existing R23 freeze differs; refusing overwrite")
        else:
            ensure_frozen(FREEZE_PATH, context["identity"])
        result = dict(context["public"])
        result.update({
            "status": "FROZEN_READY", "recipe_path": str(RECIPE_PATH.relative_to(ROOT)),
            "freeze_path": str(FREEZE_PATH.relative_to(ROOT)),
            "freeze_sha256": file_sha256(FREEZE_PATH),
            "candidate_prediction_or_metric_computed": False,
            "artifacts_written": True,
        })
        return result

    if not RECIPE_PATH.is_file() or not FREEZE_PATH.is_file():
        raise BatchIntegrityError("R23 frozen pilot requires a committed recipe and freeze")
    if _read_json(RECIPE_PATH) != context["recipe"]:
        raise BatchIntegrityError("R23 recipe differs from live preflight; no fit started")
    frozen = _read_json(FREEZE_PATH)
    if frozen.get("identity") != context["identity"] or frozen.get("signature") != json_sha256(context["identity"]):
        raise BatchIntegrityError("R23 source/input identity differs from committed freeze; no fit started")
    if file_sha256(context["train_path"]) != TRAIN_SHA256:
        raise BatchIntegrityError("Official Train changed after R23 freeze")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    frame = context["frame"].copy()
    target = np.asarray(context["target"], dtype=np.float64)
    frame["threat_rank"] = target
    report = run_pilot(
        card=card, artifact_dir=RUN_DIR,
        budget_ledger_path=LEDGER_PATH,
        max_successful_fits=MAX_SUCCESSFUL_FITS,
        frame=frame, target=target, feature_names=FEATURE_NAMES,
        adapter=r23_fit_predict,
        control_loader=_control_loader(context),
        freeze_identity=context["identity"], freeze_path=FREEZE_PATH,
        event_path=LEDGER_PATH, evaluation_views=context["views"],
        threads=THREADS, report_path=PILOT_PATH,
    )
    # Preserve current queue in verifier context; it is frozen and already hash-checked.
    context["queue"] = queue
    verification = _verify_pilot(context, report)
    generic_verification = verify_report(
        report=report, frame=frame, target=target, artifact_dir=RUN_DIR,
        freeze_signature=report["freeze_signature"], evaluation_views=context["views"],
        prediction_reconstructors={CARD_ID: reconstruct_r23_arrays},
    )
    if generic_verification.get("status") != "PASS":
        raise BatchIntegrityError("Generic independent verifier rejected the R23 report")
    verification["generic_independent_verifier"] = generic_verification
    _publish(context, report, verification)
    return {
        "card_id": CARD_ID, "status": f"PILOT_COMPLETE_{verification['decision']}",
        "decision": verification["decision"], "views": verification["views"],
        "gate_checks": verification["gate_checks"],
        "freeze_signature": verification["freeze_signature"],
        "test_data_used": False, "kaggle_access": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight-only", action="store_true")
    modes.add_argument("--freeze-only", action="store_true")
    modes.add_argument("--run-frozen-pilot", action="store_true")
    args = parser.parse_args()
    queue = load_queue(QUEUE_PATH)
    card = next(item for item in queue["cards"] if item["id"] == CARD_ID)
    result = run_card(card, queue, preflight_only=args.preflight_only,
                      freeze_only=args.freeze_only, run_frozen_pilot=args.run_frozen_pilot)
    print(json.dumps(_clean(result), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
