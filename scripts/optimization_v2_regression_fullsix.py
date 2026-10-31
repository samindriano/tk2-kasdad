"""Frozen four-fit continuation that completes the six-window regression pilot."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.audit import sha256 as file_sha256
from src.models.optimization_v2_regression import build_sparse_hybrid
from src.models.regression_research_text import HYBRID_FEATURES
from src.utils.paths import task_files

EXPERIMENT_DIR = ROOT / "experiments" / "optimization_v2" / "regression_repr"
REPORT_DIR = ROOT / "reports" / "optimization_v2" / "regression_repr"
RUN_DIR = ROOT / "experiments" / "runs" / "optimization_v2_regression_repr"
FREEZE = EXPERIMENT_DIR / "fullsix_freeze.json"
START = EXPERIMENT_DIR / "fullsix_started.json"
REPORT = REPORT_DIR / "fullsix.json"
VERIFY_REPORT = REPORT_DIR / "fullsix_verification.json"
PREDICTIONS = RUN_DIR / "fullsix_predictions.npz"

PILOT_FREEZE = EXPERIMENT_DIR / "freeze.json"
PILOT_REPORT = REPORT_DIR / "pilot.json"
PILOT_VERIFY = REPORT_DIR / "verification.json"
PILOT_PREDICTIONS = RUN_DIR / "pilot_predictions.npz"
BASELINE_REPORT = ROOT / "reports" / "dual_campaign" / "regression" / "stage1_paired_blend.json"
BASELINE_CACHE = ROOT / "experiments" / "runs" / "dual_reg_v1" / "stage1_paired_blends.npz"
ROUND2_PROTOCOL = ROOT / "experiments" / "regression_round2" / "protocol.json"

EXPECTED_BASELINE_CACHE_SHA256 = "1227404ffa7d9f5c636c109238bfc6ee99dcbfbc128c3a07f16a2eec7ddd0516"
SOURCE_FILES = (
    "scripts/optimization_v2_regression_fullsix.py",
    "scripts/optimization_v2_verify_regression_fullsix.py",
    "src/models/optimization_v2_regression.py",
    "src/models/regression_research_text.py",
    "src/models/regression_research_tabular.py",
    "src/models/regression_round2_models.py",
    "src/models/round1_models.py",
    "src/models/classification_models.py",
    "experiments/regression_round2/protocol.json",
    "scripts/optimization_v2_regression.py",
    "scripts/optimization_v2_verify_regression.py",
)
RECIPES = ("sgd_l1", "lgbm_sparse_l1")
NEW_CUTOFFS = (59, 63, 71, 75)
WINDOWS = {
    "W1": (55, 56, 59),
    "W2": (59, 60, 63),
    "W3": (63, 64, 67),
    "W4": (67, 68, 71),
    "W5": (71, 72, 75),
    "W6": (75, 76, 79),
}
HORIZONS = {"H12_E": (55, 56, 67), "H12_L": (67, 68, 79)}


def _hash(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _array_hash(value: Any, dtype: str = "<i8") -> str:
    return _hash(np.asarray(value, dtype=dtype).tobytes(order="C"))


def _write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                         allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)


def _read_train() -> tuple[pd.DataFrame, Path]:
    path = task_files("regression")["train"]
    if path is None or not path.is_file():
        raise FileNotFoundError("Official Regression Train.csv is unavailable")
    frame = pd.read_csv(path, low_memory=False)
    required = set(HYBRID_FEATURES) | {"threat_rank", "cycle_code", "entry_friction", "signal_stream"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Official Train is missing fields: {missing}")
    if len(frame) != 50_608:
        raise ValueError("Official Train row count differs from the frozen contract")
    return frame, path


def _baseline_rows(frame: pd.DataFrame, baseline: Any) -> dict[str, Any]:
    cycle = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=int)
    expected_intervals = {
        55: (56, 67), 59: (60, 63), 63: (64, 67),
        67: (68, 79), 71: (72, 75), 75: (76, 79),
    }
    receipt = {}
    for cutoff, (lo, hi) in expected_intervals.items():
        rows = np.flatnonzero((cycle >= lo) & (cycle <= hi)).astype(np.int64)
        cached = np.asarray(baseline[f"rows_{cutoff}"], dtype=np.int64)
        pred = np.asarray(baseline[f"blend_ridge25_{cutoff}"], dtype=np.float64)
        if not np.array_equal(rows, cached) or pred.shape != rows.shape or not np.isfinite(pred).all():
            raise RuntimeError(f"Ridge25 cache row/value mismatch at cutoff {cutoff}")
        receipt[str(cutoff)] = {
            "fit_rows": int(np.count_nonzero(cycle <= cutoff)),
            "fit_positions_sha256": _array_hash(np.flatnonzero(cycle <= cutoff)),
            "validation_rows": int(len(rows)),
            "validation_positions_sha256": _array_hash(rows),
            "ridge25_sha256": _array_hash(pred, "<f8"),
        }
    return receipt


def freeze_continuation() -> None:
    if any(p.exists() for p in (FREEZE, START, REPORT, VERIFY_REPORT, PREDICTIONS)):
        raise FileExistsError("Full-six continuation artifacts already exist")
    frame, train_path = _read_train()
    if file_sha256(BASELINE_CACHE) != EXPECTED_BASELINE_CACHE_SHA256:
        raise RuntimeError("Frozen Ridge25 cache hash mismatch")
    baseline_source = json.loads(BASELINE_REPORT.read_text(encoding="utf-8"))
    if "ridge_25" not in baseline_source.get("results", {}):
        raise RuntimeError("Verified Ridge25 metric section is missing from the baseline report")
    if baseline_source["derived_prediction_cache"]["sha256"] != file_sha256(BASELINE_CACHE):
        raise RuntimeError("Ridge25 report/cache receipt mismatch")
    if not PILOT_VERIFY.is_file() or json.loads(PILOT_VERIFY.read_text(encoding="utf-8"))["status"] != "PASS":
        raise RuntimeError("Prior pilot must have a verified PASS before continuation")
    if file_sha256(PILOT_PREDICTIONS) != json.loads(PILOT_REPORT.read_text(encoding="utf-8"))["prediction_cache_sha256"]:
        raise RuntimeError("Pilot prediction cache does not match its report")
    with np.load(BASELINE_CACHE, allow_pickle=False) as baseline:
        exact_rows = _baseline_rows(frame, baseline)

    try:
        head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception as exc:
        raise RuntimeError("Could not capture repository source HEAD") from exc
    cycle = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=int)
    source_hashes = {rel: file_sha256(ROOT / rel) for rel in SOURCE_FILES}
    receipt = {
        "experiment_id": "optimization_v2_sparse_hybrid_fullsix_continuation",
        "stage": "four_new_cutoffs_only_plus_row_exact_reuse_of_55_67",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_head": head,
        "source_sha256": source_hashes,
        "official_train": {
            "path": str(train_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": file_sha256(train_path), "rows": len(frame),
            "cycle_range": [int(cycle.min()), int(cycle.max())],
        },
        "existing_ridge25": {
            "report_sha256": file_sha256(BASELINE_REPORT),
            "cache_sha256": file_sha256(BASELINE_CACHE),
            "metric_provenance_correction": {
                "authoritative_report_key": "results.ridge_25",
                "cache_prediction_key": "blend_ridge25_{55,59,63,67,71,75}",
                "pilot_freeze_key_note": "Prior pilot freeze.json named results.hybrid_ridge_inner_v1 in baseline_metrics_key; the cached control arrays and prior independently verified row-exact metrics are Ridge25. This continuation uses results.ridge_25 as the authoritative report section and preserves the prior pilot artifact unchanged.",
            },
            "cutoff_rows": exact_rows,
            "new_control_fits": 0,
        },
        "existing_pilot": {
            "freeze_sha256": file_sha256(PILOT_FREEZE),
            "report_sha256": file_sha256(PILOT_REPORT),
            "verification_sha256": file_sha256(PILOT_VERIFY),
            "prediction_cache_sha256": file_sha256(PILOT_PREDICTIONS),
            "verified_status": "PASS",
            "reused_fits": ["sgd_l1@55", "lgbm_sparse_l1@55", "sgd_l1@67", "lgbm_sparse_l1@67"],
        },
        "protocol": {
            "new_fit_cutoffs": list(NEW_CUTOFFS),
            "new_fit_count": len(NEW_CUTOFFS) * len(RECIPES),
            "recipes": list(RECIPES),
            "feature_allowlist": list(HYBRID_FEATURES),
            "objective_and_preprocessing": "exactly as prefit receipt experiments/optimization_v2/regression_repr/freeze.json",
            "policies": ["fixed candidate rank50 blended 0.25/0.75 with current row-exact Ridge25"],
            "rank50_alone": "diagnostic only; retired as candidate by pilot",
            "new_candidate_blending": "prohibited",
            "hyperparameter_or_weight_search": "prohibited",
            "SGD_interpretation": "fixed-budget approximate L1; max_iter=2000; convergence warnings and n_iter recorded; no claim optimizer converged",
            "views": {name: {"fit_cutoff": values[0], "eval_cycles": [values[1], values[2]]}
                      for name, values in WINDOWS.items()},
            "long_horizons": {name: {"fit_cutoff": values[0], "eval_cycles": [values[1], values[2]]}
                              for name, values in HORIZONS.items()},
            "test_access": False,
            "scope_stop": "stop after independent verification of six-window results",
        },
        "fit_rows_by_cutoff": {str(c): int(np.count_nonzero(cycle <= c)) for c in NEW_CUTOFFS},
        "packages": {name: importlib.metadata.version(name)
                     for name in ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm")},
        "guardrails": {
            "primary_gain_minimum_mae": 0.03,
            "h12_maximum_harm_mae": 0.05,
            "minimum_nonworse_primary_windows": 4,
            "maximum_tail_or_stream_missing_harm_mae": 0.25,
        },
        "no_fit_completed_before_receipt": True,
    }
    _write_new(FREEZE, receipt)
    print(json.dumps({"freeze": str(FREEZE), "freeze_sha256": file_sha256(FREEZE),
                      "new_fit_count": receipt["protocol"]["new_fit_count"],
                      "fit_rows_by_cutoff": receipt["fit_rows_by_cutoff"]}, indent=2))


def _verify_frozen_inputs() -> tuple[pd.DataFrame, np.lib.npyio.NpzFile, np.lib.npyio.NpzFile, dict[str, Any]]:
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    if frozen["source_head"] != subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip():
        raise RuntimeError("Git HEAD changed after continuation freeze")
    for rel, digest in frozen["source_sha256"].items():
        if file_sha256(ROOT / rel) != digest:
            raise RuntimeError(f"Continuation source changed after freeze: {rel}")
    frame, train_path = _read_train()
    if file_sha256(train_path) != frozen["official_train"]["sha256"]:
        raise RuntimeError("Official Train changed after continuation freeze")
    for path, key in ((BASELINE_REPORT, "report_sha256"), (BASELINE_CACHE, "cache_sha256"),
                      (PILOT_FREEZE, "freeze_sha256"), (PILOT_REPORT, "report_sha256"),
                      (PILOT_VERIFY, "verification_sha256"), (PILOT_PREDICTIONS, "prediction_cache_sha256")):
        group = "existing_ridge25" if path in (BASELINE_REPORT, BASELINE_CACHE) else "existing_pilot"
        if file_sha256(path) != frozen[group][key]:
            raise RuntimeError(f"Frozen input artifact changed: {path.relative_to(ROOT)}")
    if file_sha256(ROUND2_PROTOCOL) != frozen["source_sha256"]["experiments/regression_round2/protocol.json"]:
        raise RuntimeError("Temporal protocol unavailable")
    with np.load(BASELINE_CACHE, allow_pickle=False) as baseline_check:
        if _baseline_rows(frame, baseline_check) != frozen["existing_ridge25"]["cutoff_rows"]:
            raise RuntimeError("Ridge25 row identities changed after freeze")
    return frame, np.load(BASELINE_CACHE, allow_pickle=False), np.load(PILOT_PREDICTIONS, allow_pickle=False), frozen


def _rank_average(raw: np.ndarray, cycle: np.ndarray) -> np.ndarray:
    result = np.empty(len(raw), dtype=float)
    for value in np.unique(cycle):
        mask = cycle == value
        result[mask] = 100.0 * rankdata(raw[mask], method="average") / int(mask.sum())
    return result


def _metric(target: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    error = np.asarray(pred, dtype=float) - np.asarray(target, dtype=float)
    if len(error) == 0:
        return {"n": 0, "mae": None, "bias_prediction_minus_target": None, "over_rate": None,
                "under_rate": None, "absolute_error_p95": None}
    return {
        "n": int(len(error)),
        "mae": float(np.mean(np.abs(error))),
        "bias_prediction_minus_target": float(np.mean(error)),
        "over_rate": float(np.mean(error > 1e-10)),
        "under_rate": float(np.mean(error < -1e-10)),
        "absolute_error_p95": float(np.quantile(np.abs(error), .95)),
    }


def _groups(frame: pd.DataFrame, pred: np.ndarray) -> dict[str, Any]:
    target = pd.to_numeric(frame["threat_rank"], errors="raise").to_numpy(dtype=float)
    text = frame["signal_stream"]
    missing = text.isna().to_numpy() | text.map(lambda v: isinstance(v, str) and not v.strip()).to_numpy(bool)
    friction = pd.to_numeric(frame["entry_friction"], errors="coerce").to_numpy(dtype=float)
    masks = {
        "tail_le5": target <= 5,
        "tail_ge95": target >= 95,
        "EF0": friction == 0,
        "EF1": friction == 1,
        "stream_missing": missing,
        "stream_present": ~missing,
    }
    return {key: _metric(target[m], pred[m]) for key, m in masks.items()}


def _prediction_rows(prediction_arrays: dict[str, np.ndarray], cutoff: int,
                     recipe: str, selected_rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(prediction_arrays[f"rows_{recipe}_{cutoff}"], dtype=np.int64)
    pred = np.asarray(prediction_arrays[f"raw_{recipe}_{cutoff}"], dtype=np.float64)
    mapping = {int(row): index for index, row in enumerate(rows)}
    if any(int(row) not in mapping for row in selected_rows):
        raise RuntimeError(f"No saved prediction at cutoff {cutoff} for requested rows")
    return pred[np.asarray([mapping[int(row)] for row in selected_rows], dtype=np.int64)]


def _baseline_rows_for(baseline: Any, cutoff: int, rows: np.ndarray) -> np.ndarray:
    source_rows = np.asarray(baseline[f"rows_{cutoff}"], dtype=np.int64)
    values = np.asarray(baseline[f"blend_ridge25_{cutoff}"], dtype=np.float64)
    mapping = {int(row): index for index, row in enumerate(source_rows)}
    if any(int(row) not in mapping for row in rows):
        raise RuntimeError(f"No cached Ridge25 control for cutoff {cutoff}")
    return values[np.asarray([mapping[int(row)] for row in rows], dtype=np.int64)]


def _evaluate_view(frame: pd.DataFrame, rows: np.ndarray, raw: np.ndarray,
                   ridge: np.ndarray) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    cycle = pd.to_numeric(frame.iloc[rows]["cycle_code"], errors="raise").to_numpy(dtype=int)
    target = pd.to_numeric(frame.iloc[rows]["threat_rank"], errors="raise").to_numpy(dtype=float)
    clipped = np.clip(raw, 0, 100)
    rank50 = np.clip(0.5 * raw + 0.5 * _rank_average(raw, cycle), 0, 100)
    blend25 = np.clip(0.25 * rank50 + 0.75 * ridge, 0, 100)
    outputs = {"raw": raw, "clipped": clipped, "rank50_diagnostic": rank50, "blend25": blend25}
    stats = {name: {**_metric(target, pred), "groups": _groups(frame.iloc[rows], pred)}
             for name, pred in outputs.items()}
    stats["ridge25"] = _metric(target, ridge)
    return stats, outputs


def run_fullsix() -> None:
    if any(p.exists() for p in (START, REPORT, VERIFY_REPORT, PREDICTIONS)):
        raise FileExistsError("Full-six continuation already started or produced artifacts; no refits permitted")
    frame, baseline, pilot, frozen = _verify_frozen_inputs()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    _write_new(START, {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_sha256": file_sha256(FREEZE),
        "new_fit_count": 8,
        "fit_sequence": [f"{recipe}@{cutoff}" for cutoff in NEW_CUTOFFS for recipe in RECIPES],
        "existing_55_67_reused": True,
    })
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=int)
    target = pd.to_numeric(frame["threat_rank"], errors="raise").to_numpy(dtype=float)
    X = frame.loc[:, list(HYBRID_FEATURES)]
    new_cache: dict[str, np.ndarray] = {}
    fit_receipts: dict[str, Any] = {}

    for cutoff in NEW_CUTOFFS:
        rows_fit = np.flatnonzero(cycles <= cutoff).astype(np.int64)
        rows_val = np.flatnonzero((cycles >= cutoff + 1) & (cycles <= cutoff + 4))
        if not len(rows_val) or np.any(cycles[rows_val] <= cutoff):
            raise RuntimeError(f"Temporal boundary failure for cutoff {cutoff}")
        for recipe in RECIPES:
            model = build_sparse_hybrid(recipe)
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                started = time.perf_counter()
                model.fit(X.iloc[rows_fit], target[rows_fit])
                fit_seconds = time.perf_counter() - started
                prediction_started = time.perf_counter()
                raw = model.predict(X.iloc[rows_val])
                prediction_seconds = time.perf_counter() - prediction_started
            state = model.audit_state()
            warning_receipts = [{"category": item.category.__name__, "message": str(item.message)}
                                for item in captured]
            optimizer = state["estimator"]
            sgd_audit = None
            if recipe == "sgd_l1":
                sgd_audit = {
                    "interpretation": "fixed-budget approximate L1; convergence not asserted",
                    "n_iter": int(optimizer["n_iter"]),
                    "max_iter": int(optimizer["parameters"]["max_iter"]),
                    "max_iter_reached": bool(optimizer["n_iter"] >= optimizer["parameters"]["max_iter"]),
                    "convergence_warnings": [item for item in warning_receipts
                                              if item["category"] == "ConvergenceWarning"],
                }
            if state["fit_rows"] != len(rows_fit) or int(cycles[rows_fit].max()) != cutoff:
                raise RuntimeError(f"Fit rows/fold boundary mismatch for {recipe}@{cutoff}")
            if state["validation_data_passed_to_fit"] or state["early_stopping_used"] or not state["sparse_design"]:
                raise RuntimeError(f"Candidate representation/fit boundary changed for {recipe}@{cutoff}")
            key = f"{recipe}_{cutoff}"
            new_cache[f"rows_{key}"] = rows_val.astype(np.int64)
            new_cache[f"raw_{key}"] = raw
            fit_receipts[key] = {
                "fit_rows": int(len(rows_fit)),
                "fit_positions_sha256": _array_hash(rows_fit),
                "fit_cycle_max": int(cycles[rows_fit].max()),
                "validation_rows": int(len(rows_val)),
                "validation_positions_sha256": _array_hash(rows_val),
                "validation_cycles": [int(cycles[rows_val].min()), int(cycles[rows_val].max())],
                "fit_seconds": float(fit_seconds),
                "prediction_seconds": float(prediction_seconds),
                "warnings": warning_receipts,
                "model_state": state,
                "sgd_optimizer_audit": sgd_audit,
                "prediction_diagnostics": model.prediction_diagnostics(),
                "raw_prediction_sha256": _array_hash(raw, "<f8"),
            }
            print(json.dumps({"fit": key, "fit_rows": len(rows_fit), "validation_rows": len(rows_val),
                              "fit_seconds": round(fit_seconds, 2), "prediction_seconds": round(prediction_seconds, 2),
                              "sgd_n_iter": None if sgd_audit is None else sgd_audit["n_iter"],
                              "convergence_warnings": 0 if sgd_audit is None else len(sgd_audit["convergence_warnings"])}, sort_keys=True), flush=True)

    # Assemble new four-cycle predictions together with the previously verified
    # cutoff 55/67 arrays. The latter are reused by exact source row position.
    prediction_arrays: dict[str, np.ndarray] = {}
    for recipe in RECIPES:
        for cutoff in (55, 67):
            prediction_arrays[f"rows_{recipe}_{cutoff}"] = np.asarray(pilot[f"rows_{recipe}_{cutoff}"], dtype=np.int64)
            prediction_arrays[f"raw_{recipe}_{cutoff}"] = np.asarray(pilot[f"raw_{recipe}_{cutoff}"], dtype=float)
        for cutoff in NEW_CUTOFFS:
            prediction_arrays[f"rows_{recipe}_{cutoff}"] = new_cache[f"rows_{recipe}_{cutoff}"]
            prediction_arrays[f"raw_{recipe}_{cutoff}"] = new_cache[f"raw_{recipe}_{cutoff}"]

    views: dict[str, Any] = {}
    view_component_predictions: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for view, (cutoff, lo, hi) in WINDOWS.items():
        rows = np.flatnonzero((cycles >= lo) & (cycles <= hi)).astype(np.int64)
        ridge = _baseline_rows_for(baseline, cutoff, rows)
        views[view] = {"cutoff": cutoff, "cycles": [lo, hi], "rows": int(len(rows)),
                       "positions_sha256": _array_hash(rows), "ridge25": _metric(target[rows], ridge)}
        view_component_predictions[view] = {}
        for recipe in RECIPES:
            raw = _prediction_rows(prediction_arrays, cutoff, recipe, rows)
            metrics, outputs = _evaluate_view(frame, rows, raw, ridge)
            views[view][recipe] = metrics
            view_component_predictions[view][recipe] = outputs

    horizons: dict[str, Any] = {}
    horizon_component_predictions: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for horizon, (cutoff, lo, hi) in HORIZONS.items():
        rows = np.flatnonzero((cycles >= lo) & (cycles <= hi)).astype(np.int64)
        ridge = _baseline_rows_for(baseline, cutoff, rows)
        horizons[horizon] = {"cutoff": cutoff, "cycles": [lo, hi], "rows": int(len(rows)),
                             "positions_sha256": _array_hash(rows), "ridge25": _metric(target[rows], ridge)}
        horizon_component_predictions[horizon] = {}
        for recipe in RECIPES:
            raw = _prediction_rows(prediction_arrays, cutoff, recipe, rows)
            metrics, outputs = _evaluate_view(frame, rows, raw, ridge)
            horizons[horizon][recipe] = metrics
            horizon_component_predictions[horizon][recipe] = outputs

    gates: dict[str, Any] = {}
    for recipe in RECIPES:
        rows_all = np.concatenate([np.flatnonzero((cycles >= lo) & (cycles <= hi))
                                   for _, lo, hi in WINDOWS.values()]).astype(np.int64)
        candidate_all = np.concatenate([view_component_predictions[w][recipe]["blend25"] for w in WINDOWS])
        ridge_all = np.concatenate([_baseline_rows_for(baseline, cutoff,
                                                      np.flatnonzero((cycles >= lo) & (cycles <= hi)))
                                    for cutoff, lo, hi in WINDOWS.values()])
        primary_candidate = _metric(target[rows_all], candidate_all)
        primary_ridge = _metric(target[rows_all], ridge_all)
        primary_gain = primary_ridge["mae"] - primary_candidate["mae"]
        window_deltas = {w: views[w][recipe]["blend25"]["mae"] - views[w]["ridge25"]["mae"] for w in WINDOWS}
        window_wins = sum(delta <= 0 for delta in window_deltas.values())
        h12e_delta = horizons["H12_E"][recipe]["blend25"]["mae"] - horizons["H12_E"]["ridge25"]["mae"]
        h12l_delta = horizons["H12_L"][recipe]["blend25"]["mae"] - horizons["H12_L"]["ridge25"]["mae"]
        h12_rows = np.concatenate([
            np.flatnonzero((cycles >= HORIZONS[h][1]) & (cycles <= HORIZONS[h][2])) for h in HORIZONS
        ]).astype(np.int64)
        h12_pred = np.concatenate([horizon_component_predictions[h][recipe]["blend25"] for h in HORIZONS])
        h12_base = np.concatenate([_baseline_rows_for(baseline, HORIZONS[h][0],
                                                     np.flatnonzero((cycles >= HORIZONS[h][1]) &
                                                                    (cycles <= HORIZONS[h][2]))) for h in HORIZONS])
        h12_pooled_delta = _metric(target[h12_rows], h12_pred)["mae"] - _metric(target[h12_rows], h12_base)["mae"]

        subgroup_deltas: dict[str, Any] = {}
        max_tail_missing_harm = -float("inf")
        for segment_name, metrics in [("primary_pooled", _groups(frame.iloc[rows_all], candidate_all)),
                                      ("H12_E", horizons["H12_E"][recipe]["blend25"]["groups"]),
                                      ("H12_L", horizons["H12_L"][recipe]["blend25"]["groups"])]:
            if segment_name == "primary_pooled":
                base_groups = _groups(frame.iloc[rows_all], ridge_all)
            else:
                base_groups = horizons[segment_name]["ridge25"]
                # Add target/stream subgroup summaries for the incumbent through
                # the same metric path without refitting or touching labels in fit.
                rows_h = np.flatnonzero((cycles >= HORIZONS[segment_name][1]) &
                                        (cycles <= HORIZONS[segment_name][2]))
                base_groups = _groups(frame.iloc[rows_h], _baseline_rows_for(baseline, HORIZONS[segment_name][0], rows_h))
            subgroup_deltas[segment_name] = {}
            for subgroup in ("tail_le5", "tail_ge95", "stream_missing"):
                delta = metrics[subgroup]["mae"] - base_groups[subgroup]["mae"]
                subgroup_deltas[segment_name][subgroup] = float(delta)
                max_tail_missing_harm = max(max_tail_missing_harm, float(delta))
        gate = {
            "primary_ridge25": primary_ridge,
            "primary_candidate": primary_candidate,
            "primary_gain_mae": float(primary_gain),
            "window_delta_mae_candidate_minus_ridge25": {k: float(v) for k, v in window_deltas.items()},
            "nonworse_windows_count": int(window_wins),
            "H12_E_delta_mae": float(h12e_delta),
            "H12_L_delta_mae": float(h12l_delta),
            "pooled_H12_delta_mae": float(h12_pooled_delta),
            "tail_and_missing_delta_mae": subgroup_deltas,
            "maximum_tail_or_missing_harm_mae": float(max_tail_missing_harm),
            "gates": {
                "primary_gain_at_least_0_03": bool(primary_gain >= .03),
                "H12_E_and_H12_L_harm_at_most_0_05": bool(max(h12e_delta, h12l_delta) <= .05),
                "at_least_4_of_6_nonworse_windows": bool(window_wins >= 4),
                "all_reported_tail_and_missing_harm_at_most_0_25": bool(max_tail_missing_harm <= .25),
            },
        }
        gate["all_gates_pass"] = all(gate["gates"].values())
        gates[recipe] = gate

    # Cross-candidate complementarity is descriptive only; no third blend is formed.
    complementarity: dict[str, Any] = {}
    for segment, row_blocks in {
        "primary_pooled": [(w, np.flatnonzero((cycles >= lo) & (cycles <= hi)).astype(np.int64))
                           for w, (_, lo, hi) in WINDOWS.items()],
        "H12_E": [("H12_E", np.flatnonzero((cycles >= 56) & (cycles <= 67)).astype(np.int64))],
        "H12_L": [("H12_L", np.flatnonzero((cycles >= 68) & (cycles <= 79)).astype(np.int64))],
    }.items():
        left = np.concatenate([view_component_predictions[w]["sgd_l1"]["blend25"] if w in WINDOWS
                               else horizon_component_predictions[w]["sgd_l1"]["blend25"] for w, _ in row_blocks])
        right = np.concatenate([view_component_predictions[w]["lgbm_sparse_l1"]["blend25"] if w in WINDOWS
                                else horizon_component_predictions[w]["lgbm_sparse_l1"]["blend25"] for w, _ in row_blocks])
        rows = np.concatenate([block for _, block in row_blocks])
        err_l, err_r = left - target[rows], right - target[rows]
        abs_l, abs_r = np.abs(err_l), np.abs(err_r)
        mae_l, mae_r = float(np.mean(abs_l)), float(np.mean(abs_r))
        complementarity[segment] = {
            "n": int(len(rows)),
            "prediction_correlation": float(np.corrcoef(left, right)[0, 1]),
            "signed_error_correlation": float(np.corrcoef(err_l, err_r)[0, 1]),
            "absolute_error_correlation": float(np.corrcoef(abs_l, abs_r)[0, 1]),
            "SGD_lower_absolute_error_rate": float(np.mean(abs_l < abs_r)),
            "LightGBM_lower_absolute_error_rate": float(np.mean(abs_r < abs_l)),
            "tie_rate": float(np.mean(abs_l == abs_r)),
            "MAE_SGD": mae_l,
            "MAE_LightGBM": mae_r,
            "candidate_blend_created": False,
        }

    per_cycle: dict[str, Any] = {}
    for cycle_value in range(56, 80):
        rows = np.flatnonzero(cycles == cycle_value)
        win = next(name for name, (cutoff, lo, hi) in WINDOWS.items() if lo <= cycle_value <= hi)
        base = _baseline_rows_for(baseline, WINDOWS[win][0], rows)
        per_cycle[str(cycle_value)] = {"n": int(len(rows)), "ridge25_mae": _metric(target[rows], base)["mae"]}
        for recipe in RECIPES:
            pred = view_component_predictions[win][recipe]["blend25"]
            # Each cycle is complete in this source. Select by global row positions.
            window_rows = np.flatnonzero((cycles >= WINDOWS[win][1]) & (cycles <= WINDOWS[win][2]))
            index_map = {int(row): i for i, row in enumerate(window_rows)}
            per_cycle[str(cycle_value)][recipe] = _metric(target[rows], pred[np.asarray([index_map[int(r)] for r in rows])])
        per_cycle[str(cycle_value)]["window"] = win

    total_fit_seconds = float(sum(value["fit_seconds"] for value in fit_receipts.values()))
    result = {
        "experiment_id": "optimization_v2_sparse_hybrid_fullsix_continuation",
        "stage": "six_windows_plus_H12_complete",
        "continuation_freeze_sha256": file_sha256(FREEZE),
        "started_receipt_sha256": file_sha256(START),
        "new_fit_count": len(fit_receipts),
        "reused_fit_count": 4,
        "control_fit_count": 0,
        "fits": fit_receipts,
        "fit_total_seconds": total_fit_seconds,
        "fit_total_minutes": total_fit_seconds / 60.0,
        "views": views,
        "long_horizons": horizons,
        "per_cycle_blend25": per_cycle,
        "candidate_gates": gates,
        "candidate_complementarity": complementarity,
        "baseline_metric_provenance_correction": frozen["existing_ridge25"]["metric_provenance_correction"],
        "diagnostic_policy_note": "rank50-alone is shown for context but retired at the pilot gate; fixed blend25 is the only candidate policy here",
        "model_combination_note": "SGD and sparse LightGBM candidates are compared only; no blend between candidates was fit or scored",
        "SGD_note": "epsilon-insensitive epsilon=0 SGD is a fixed-budget approximate L1 optimization; maximum-iteration convergence warnings/n_iter are retained; optimizer convergence is not claimed",
        "all_six_primary_windows_and_both_H12_views_completed": True,
        "test_access": False,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with PREDICTIONS.open("xb") as handle:
        np.savez_compressed(handle, **new_cache)
    result["new_prediction_cache_path"] = str(PREDICTIONS.relative_to(ROOT)).replace("\\", "/")
    result["new_prediction_cache_sha256"] = file_sha256(PREDICTIONS)
    _write_new(REPORT, result)
    baseline.close()
    pilot.close()
    print(json.dumps({"report": str(REPORT), "cache_sha256": result["new_prediction_cache_sha256"],
                      "fit_total_minutes": result["fit_total_minutes"], "gates": gates}, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "run"))
    args = parser.parse_args()
    freeze_continuation() if args.action == "freeze" else run_fullsix()
