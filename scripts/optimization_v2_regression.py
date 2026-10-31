"""Freeze and run the four-fit sparse-representation pilot (cutoffs 55 and 67)."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.audit import sha256 as file_sha256
from src.models.optimization_v2_regression import (
    SGD_PARAMETERS,
    SPARSE_LGB_PARAMETERS,
    build_sparse_hybrid,
)
from src.models.regression_research_text import HYBRID_FEATURES
from src.utils.paths import task_files

EXPERIMENT_DIR = ROOT / "experiments" / "optimization_v2" / "regression_repr"
REPORT_DIR = ROOT / "reports" / "optimization_v2" / "regression_repr"
RUN_DIR = ROOT / "experiments" / "runs" / "optimization_v2_regression_repr"
FREEZE_PATH = EXPERIMENT_DIR / "freeze.json"
START_PATH = EXPERIMENT_DIR / "pilot_started.json"
PILOT_PATH = REPORT_DIR / "pilot.json"
PREDICTION_PATH = RUN_DIR / "pilot_predictions.npz"
BASELINE_REPORT = ROOT / "reports" / "dual_campaign" / "regression" / "stage1_paired_blend.json"
BASELINE_CACHE = ROOT / "experiments" / "runs" / "dual_reg_v1" / "stage1_paired_blends.npz"
ROUND2_PROTOCOL = ROOT / "experiments" / "regression_round2" / "protocol.json"

EXPECTED_BASELINE_CACHE_SHA256 = "1227404ffa7d9f5c636c109238bfc6ee99dcbfbc128c3a07f16a2eec7ddd0516"
SOURCE_FILES = (
    "src/models/optimization_v2_regression.py",
    "scripts/optimization_v2_regression.py",
    "scripts/optimization_v2_verify_regression.py",
    "tests/test_optimization_v2_regression.py",
    "src/models/regression_research_text.py",
    "src/models/regression_research_tabular.py",
    "src/models/regression_round2_models.py",
    "src/models/round1_models.py",
    "src/models/classification_models.py",
    "experiments/regression_round2/protocol.json",
)
VIEWS = {
    "W1": (55, 56, 59),
    "W4": (67, 68, 71),
    "H12_E": (55, 56, 67),
    "H12_L": (67, 68, 79),
}
RECIPES = ("sgd_l1", "lgbm_sparse_l1")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return _sha_bytes(encoded)


def _array_hash(value: Any, dtype: str = "<i8") -> str:
    arr = np.asarray(value, dtype=dtype)
    return _sha_bytes(arr.tobytes(order="C"))


def _atomic_json_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                         allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)


def _read_train() -> tuple[pd.DataFrame, Path]:
    train_path = task_files("regression")["train"]
    if train_path is None or not train_path.is_file():
        raise FileNotFoundError("Official Regression Train.csv is unavailable")
    frame = pd.read_csv(train_path, low_memory=False)
    required = set(HYBRID_FEATURES) | {"threat_rank", "cycle_code", "signal_stream"}
    absent = sorted(required.difference(frame.columns))
    if absent:
        raise ValueError(f"Official Regression Train is missing required columns: {absent}")
    if len(frame) != 50_608 or set(pd.to_numeric(frame["cycle_code"], errors="raise").astype(int)) != set(range(47, 80)):
        raise ValueError("Official Regression Train row count/cycle range differs from frozen contract")
    return frame, train_path


def _baseline_identity(frame: pd.DataFrame, cache: Any) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=int)
    for cutoff in (55, 67):
        lower, upper = (56, 67) if cutoff == 55 else (68, 79)
        expected_rows = np.flatnonzero((cycles >= lower) & (cycles <= upper)).astype(np.int64)
        cached_rows = np.asarray(cache[f"rows_{cutoff}"], dtype=np.int64)
        baseline = np.asarray(cache[f"blend_ridge25_{cutoff}"], dtype=np.float64)
        if not np.array_equal(cached_rows, expected_rows):
            raise ValueError(f"Ridge25 cached row order does not match official Train at cutoff {cutoff}")
        if baseline.shape != cached_rows.shape or not np.isfinite(baseline).all():
            raise ValueError(f"Ridge25 baseline cache is invalid at cutoff {cutoff}")
        checks[str(cutoff)] = {
            "fit_rows": int(np.sum(cycles <= cutoff)),
            "validation_rows": int(len(cached_rows)),
            "fit_positions_sha256": _array_hash(np.flatnonzero(cycles <= cutoff)),
            "validation_positions_sha256": _array_hash(cached_rows),
            "baseline_prediction_sha256": _array_hash(baseline, dtype="<f8"),
        }
    return checks


def create_freeze() -> None:
    """Create the pre-fit immutable protocol and source/data/cache receipt."""
    if FREEZE_PATH.exists() or START_PATH.exists() or PILOT_PATH.exists() or PREDICTION_PATH.exists():
        raise FileExistsError("Regression representation freeze or pilot artifacts already exist")
    train, train_path = _read_train()
    raw_sha = file_sha256(train_path)
    baseline_sha = file_sha256(BASELINE_CACHE)
    if baseline_sha != EXPECTED_BASELINE_CACHE_SHA256:
        raise ValueError("Existing Ridge25 prediction cache hash differs from verified source receipt")
    baseline_report = json.loads(BASELINE_REPORT.read_text(encoding="utf-8"))
    cache_receipt = baseline_report["derived_prediction_cache"]
    if cache_receipt.get("sha256") != baseline_sha or cache_receipt.get("path", "").replace("/", "\\") != str(BASELINE_CACHE.relative_to(ROOT)).replace("/", "\\"):
        raise ValueError("Ridge25 report/cache identity does not match the frozen cache")
    with np.load(BASELINE_CACHE, allow_pickle=False) as cache:
        baseline_rows = _baseline_identity(train, cache)

    source_hashes = {rel: file_sha256(ROOT / rel) for rel in SOURCE_FILES}
    try:
        head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception as exc:
        raise RuntimeError("Could not capture the repository source commit") from exc
    cycles = pd.to_numeric(train["cycle_code"], errors="raise").to_numpy(dtype=int)
    protocol = {
        "experiment_id": "optimization_v2_sparse_hybrid_regression_pilot",
        "stage": "four_fit_pilot_only_no_full_six_continuation",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_head": head,
        "source_sha256": source_hashes,
        "official_train": {
            "path": str(train_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": raw_sha,
            "rows": int(len(train)),
            "cycle_min": int(cycles.min()),
            "cycle_max": int(cycles.max()),
            "target": "threat_rank",
            "target_min": float(train["threat_rank"].min()),
            "target_max": float(train["threat_rank"].max()),
        },
        "round2_protocol_sha256": file_sha256(ROUND2_PROTOCOL),
        "current_ridge25": {
            "report_path": str(BASELINE_REPORT.relative_to(ROOT)).replace("\\", "/"),
            "report_sha256": file_sha256(BASELINE_REPORT),
            "cache_path": str(BASELINE_CACHE.relative_to(ROOT)).replace("\\", "/"),
            "cache_sha256": baseline_sha,
            "cutoff_identity": baseline_rows,
            "prediction_key": "blend_ridge25_{55,67}",
            "control_fits_in_this_pilot": 0,
        },
        "fit_protocol": {
            "feature_allowlist": list(HYBRID_FEATURES),
            "preprocessing": "RegressionTextModel._structured_fit/_structured_transform + _fit_vectorizer; scaled numeric, train-only OHE, train-only raw TF-IDF, missing indicator",
            "no_svd": True,
            "no_native_categorical": True,
            "cutoffs": [55, 67],
            "fit_rows": {
                str(c): int(np.sum(cycles <= c)) for c in (55, 67)
            },
            "views": {name: {"fit_cutoff": val[0], "eval_cycle": [val[1], val[2]]}
                      for name, val in VIEWS.items()},
            "estimators": {
                "sgd_l1": dict(SGD_PARAMETERS),
                "lgbm_sparse_l1": dict(SPARSE_LGB_PARAMETERS),
            },
            "policies": {
                "raw": "unclipped direct prediction diagnostic",
                "clipped": "clip raw candidate prediction to [0, 100] diagnostic",
                "rank50": "clip(0.5 * raw + 0.5 * within-cycle average percentile rank, 0, 100)",
                "blend25": "clip(0.25 * candidate_rank50 + 0.75 * cached current Ridge25, 0, 100)",
            },
            "no_search": True,
            "no_validation_labels_in_fit": True,
            "official_test_loaded": False,
        },
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm")
        },
        "baseline_metrics_key": "results.hybrid_ridge_inner_v1",
        "notes": [
            "Ridge25 predictions are reused row-exactly; no control model is refit.",
            "All temporal validation cycles were seen in prior development; no inferential or untouched-holdout claim.",
            "Only these two fixed estimators at cutoffs 55 and 67 are run in this milestone.",
        ],
    }
    _atomic_json_new(FREEZE_PATH, protocol)
    print(json.dumps({"freeze_path": str(FREEZE_PATH), "sha256": file_sha256(FREEZE_PATH),
                      "official_train_sha256": raw_sha, "ridge25_cache_sha256": baseline_sha,
                      "fit_rows": protocol["fit_protocol"]["fit_rows"]}, indent=2))


def _verify_frozen_inputs() -> tuple[pd.DataFrame, Any, dict[str, Any]]:
    if not FREEZE_PATH.is_file():
        raise FileNotFoundError("Create the pre-fit receipt first: freeze")
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if freeze["source_head"] != subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip():
        raise RuntimeError("Git HEAD changed after the source/data freeze")
    for rel, expected in freeze["source_sha256"].items():
        if file_sha256(ROOT / rel) != expected:
            raise RuntimeError(f"Frozen source changed after the pre-fit freeze: {rel}")
    train, train_path = _read_train()
    if str(train_path.relative_to(ROOT)).replace("\\", "/") != freeze["official_train"]["path"]:
        raise RuntimeError("Official Train path changed after freeze")
    if file_sha256(train_path) != freeze["official_train"]["sha256"]:
        raise RuntimeError("Official Train bytes changed after freeze")
    if file_sha256(ROUND2_PROTOCOL) != freeze["round2_protocol_sha256"]:
        raise RuntimeError("Round 2 temporal protocol changed after freeze")
    if file_sha256(BASELINE_REPORT) != freeze["current_ridge25"]["report_sha256"]:
        raise RuntimeError("Current Ridge25 report changed after freeze")
    if file_sha256(BASELINE_CACHE) != freeze["current_ridge25"]["cache_sha256"]:
        raise RuntimeError("Current Ridge25 predictions changed after freeze")
    cache = np.load(BASELINE_CACHE, allow_pickle=False)
    current = _baseline_identity(train, cache)
    if current != freeze["current_ridge25"]["cutoff_identity"]:
        cache.close()
        raise RuntimeError("Current Ridge25 row/prediction identity changed after freeze")
    return train, cache, freeze


def _percentile_scores(prediction: np.ndarray, cycles: np.ndarray) -> np.ndarray:
    result = np.empty(len(prediction), dtype=np.float64)
    for cycle in np.unique(cycles):
        mask = cycles == cycle
        result[mask] = 100.0 * rankdata(prediction[mask], method="average") / int(mask.sum())
    return result


def _policy_values(raw: np.ndarray, cycles: np.ndarray, ridge25: np.ndarray) -> dict[str, np.ndarray]:
    clipped = np.clip(raw, 0.0, 100.0)
    rank50 = np.clip(0.5 * raw + 0.5 * _percentile_scores(raw, cycles), 0.0, 100.0)
    blend25 = np.clip(0.25 * rank50 + 0.75 * ridge25, 0.0, 100.0)
    return {"raw": raw, "clipped": clipped, "rank50": rank50, "blend25": blend25}


def _metric(y: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    error = prediction - y
    return {
        "n": int(len(y)),
        "mae": float(np.mean(np.abs(error))) if len(y) else None,
        "signed_bias_prediction_minus_target": float(np.mean(error)) if len(y) else None,
        "over_rate": float(np.mean(error > 1e-10)) if len(y) else None,
        "under_rate": float(np.mean(error < -1e-10)) if len(y) else None,
        "absolute_error_p95": float(np.quantile(np.abs(error), 0.95)) if len(y) else None,
    }


def _view_rows(frame: pd.DataFrame, cutoff: int, low: int, high: int) -> np.ndarray:
    cycle = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=int)
    rows = np.flatnonzero((cycle >= low) & (cycle <= high))
    if not len(rows) or np.any(cycle[rows] <= cutoff):
        raise ValueError(f"Invalid temporal view fit/eval boundary at cutoff {cutoff}")
    return rows.astype(np.int64)


def _group_metrics(frame: pd.DataFrame, prediction: np.ndarray) -> dict[str, Any]:
    y = pd.to_numeric(frame["threat_rank"], errors="raise").to_numpy(dtype=float)
    stream = frame["signal_stream"]
    stream_missing = stream.isna().to_numpy() | stream.map(
        lambda value: isinstance(value, str) and not value.strip()
    ).to_numpy(dtype=bool)
    ef = pd.to_numeric(frame["entry_friction"], errors="coerce").to_numpy(dtype=float)
    masks = {
        "tail_le5": y <= 5,
        "tail_ge95": y >= 95,
        "EF0": ef == 0,
        "EF1": ef == 1,
        "stream_missing": stream_missing,
        "stream_present": ~stream_missing,
    }
    return {name: _metric(y[mask], prediction[mask]) for name, mask in masks.items()}


def _score_result(frame: pd.DataFrame, raw: np.ndarray, ridge25: np.ndarray) -> dict[str, Any]:
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=int)
    target = pd.to_numeric(frame["threat_rank"], errors="raise").to_numpy(dtype=float)
    policies = _policy_values(raw, cycles, ridge25)
    output = {}
    for name, estimate in policies.items():
        output[name] = {**_metric(target, estimate), "groups": _group_metrics(frame, estimate)}
    return output


def run_pilot() -> None:
    if START_PATH.exists() or PILOT_PATH.exists() or PREDICTION_PATH.exists():
        raise FileExistsError("Pilot has already started or produced outputs; never refit this four-fit pilot")
    train, baseline_cache, freeze = _verify_frozen_inputs()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    # This immutable marker makes a partial failed run visible and prevents an
    # accidental second set of fits from being mistaken for the same pilot.
    _atomic_json_new(START_PATH, {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_sha256": file_sha256(FREEZE_PATH),
        "planned_fit_count": 4,
        "planned_fits": [f"{recipe}@{cutoff}" for cutoff in (55, 67) for recipe in RECIPES],
    })

    cycles = pd.to_numeric(train["cycle_code"], errors="raise").to_numpy(dtype=int)
    X = train.loc[:, list(HYBRID_FEATURES)]
    y = pd.to_numeric(train["threat_rank"], errors="raise").to_numpy(dtype=float)
    prediction_cache: dict[str, np.ndarray] = {}
    receipts: dict[str, Any] = {}

    for cutoff in (55, 67):
        fit_rows = np.flatnonzero(cycles <= cutoff).astype(np.int64)
        val_low, val_high = (56, 67) if cutoff == 55 else (68, 79)
        val_rows = _view_rows(train, cutoff, val_low, val_high)
        if len(fit_rows) != freeze["fit_protocol"]["fit_rows"][str(cutoff)]:
            raise RuntimeError(f"Fit rows differ from frozen cutoff {cutoff}")
        for recipe in RECIPES:
            model = build_sparse_hybrid(recipe)
            model.fit(X.iloc[fit_rows], y[fit_rows])
            raw = model.predict(X.iloc[val_rows])
            audit = model.audit_state()
            if audit["fit_rows"] != len(fit_rows) or audit["fit_data_sha256"] != model._fit_fingerprint:
                raise RuntimeError("Fit receipt does not match the declared train-only fold")
            if audit["validation_data_passed_to_fit"] or audit["early_stopping_used"]:
                raise RuntimeError("Unexpected validation/early-stopping path in candidate")
            key = f"{recipe}_{cutoff}"
            prediction_cache[f"rows_{key}"] = val_rows
            prediction_cache[f"raw_{key}"] = raw
            receipts[key] = {
                "fit_rows": int(len(fit_rows)),
                "fit_positions_sha256": _array_hash(fit_rows),
                "fit_cycle_max": int(cycles[fit_rows].max()),
                "validation_rows": int(len(val_rows)),
                "validation_positions_sha256": _array_hash(val_rows),
                "validation_cycle_min": int(cycles[val_rows].min()),
                "validation_cycle_max": int(cycles[val_rows].max()),
                "fit_state": audit,
                "prediction_diagnostics": model.prediction_diagnostics(),
                "raw_prediction_sha256": _array_hash(raw, dtype="<f8"),
            }
            print(json.dumps({"fit": key, "fit_rows": len(fit_rows), "validation_rows": len(val_rows),
                              "design_shape": audit["training_design_shape"],
                              "raw_prediction_sha256": receipts[key]["raw_prediction_sha256"]}, sort_keys=True), flush=True)

    if set(prediction_cache) != {f"{kind}_{part}" for kind in ("rows", "raw") for part in
                                 ("sgd_l1_55", "lgbm_sparse_l1_55", "sgd_l1_67", "lgbm_sparse_l1_67")}:
        raise RuntimeError("Pilot did not produce the exact planned prediction keys")

    views: dict[str, Any] = {}
    view_predictions: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for view_name, (cutoff, low, high) in VIEWS.items():
        base_rows = _view_rows(train, cutoff, low, high)
        baseline_rows = np.asarray(baseline_cache[f"rows_{cutoff}"], dtype=np.int64)
        row_to_base = {int(row): index for index, row in enumerate(baseline_rows)}
        if any(int(row) not in row_to_base for row in base_rows):
            raise RuntimeError(f"Ridge25 control is not available for exact view {view_name}")
        base_index = np.asarray([row_to_base[int(row)] for row in base_rows], dtype=np.int64)
        ridge25 = np.asarray(baseline_cache[f"blend_ridge25_{cutoff}"][base_index], dtype=np.float64)
        y_view = y[base_rows]
        cycles_view = cycles[base_rows]
        view_frame = train.iloc[base_rows]
        views[view_name] = {
            "fit_cutoff": cutoff,
            "eval_cycle": [low, high],
            "rows": int(len(base_rows)),
            "source_positions_sha256": _array_hash(base_rows),
            "ridge25": _metric(y_view, ridge25),
        }
        view_predictions[view_name] = {}
        for recipe in RECIPES:
            cache_key = f"{recipe}_{cutoff}"
            val_rows = prediction_cache[f"rows_{cache_key}"]
            row_map = {int(row): i for i, row in enumerate(val_rows)}
            candidate_raw = prediction_cache[f"raw_{cache_key}"][
                np.asarray([row_map[int(row)] for row in base_rows], dtype=np.int64)
            ]
            result = _score_result(view_frame, candidate_raw, ridge25)
            views[view_name][recipe] = result
            view_predictions[view_name][recipe] = _policy_values(candidate_raw, cycles_view, ridge25)

    policy_summaries: dict[str, Any] = {}
    for recipe in RECIPES:
        policy_summaries[recipe] = {}
        for policy in ("rank50", "blend25"):
            primary_y, primary_p, primary_base = [], [], []
            h12_y, h12_p, h12_base = [], [], []
            for view_name, labels in (("W1", (primary_y, primary_p, primary_base)),
                                      ("W4", (primary_y, primary_p, primary_base))):
                frame = train.iloc[_view_rows(train, *VIEWS[view_name])]
                cutoff = VIEWS[view_name][0]
                cycle = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=int)
                target = pd.to_numeric(frame["threat_rank"], errors="raise").to_numpy(dtype=float)
                rows = _view_rows(train, *VIEWS[view_name])
                base_rows = np.asarray(baseline_cache[f"rows_{cutoff}"], dtype=np.int64)
                base_map = {int(row): i for i, row in enumerate(base_rows)}
                base = np.asarray(baseline_cache[f"blend_ridge25_{cutoff}"])[
                    np.asarray([base_map[int(row)] for row in rows], dtype=np.int64)
                ]
                pred = view_predictions[view_name][recipe][policy]
                labels[0].append(target)
                labels[1].append(pred)
                labels[2].append(base)
            for view_name in ("H12_E", "H12_L"):
                frame = train.iloc[_view_rows(train, *VIEWS[view_name])]
                cutoff = VIEWS[view_name][0]
                rows = _view_rows(train, *VIEWS[view_name])
                base_rows = np.asarray(baseline_cache[f"rows_{cutoff}"], dtype=np.int64)
                base_map = {int(row): i for i, row in enumerate(base_rows)}
                base = np.asarray(baseline_cache[f"blend_ridge25_{cutoff}"])[
                    np.asarray([base_map[int(row)] for row in rows], dtype=np.int64)
                ]
                target = pd.to_numeric(frame["threat_rank"], errors="raise").to_numpy(dtype=float)
                h12_y.append(target)
                h12_p.append(view_predictions[view_name][recipe][policy])
                h12_base.append(base)
            pv = np.concatenate(primary_y)
            pp = np.concatenate(primary_p)
            pb = np.concatenate(primary_base)
            hv = np.concatenate(h12_y)
            hp = np.concatenate(h12_p)
            hb = np.concatenate(h12_base)
            primary_delta = _metric(pv, pp)["mae"] - _metric(pv, pb)["mae"]
            h12_delta = _metric(hv, hp)["mae"] - _metric(hv, hb)["mae"]
            view_deltas = {
                view_name: views[view_name][recipe][policy]["mae"] - views[view_name]["ridge25"]["mae"]
                for view_name in VIEWS
            }
            eligible = (primary_delta < 0 or h12_delta < 0) and all(delta <= 0.50 for delta in view_deltas.values())
            policy_summaries[recipe][policy] = {
                "primary_pooled": _metric(pv, pp),
                "primary_ridge25": _metric(pv, pb),
                "primary_delta_mae_candidate_minus_ridge25": float(primary_delta),
                "h12_pooled": _metric(hv, hp),
                "h12_ridge25": _metric(hv, hb),
                "h12_delta_mae_candidate_minus_ridge25": float(h12_delta),
                "individual_view_delta_mae": {k: float(v) for k, v in view_deltas.items()},
                "pilot_continuation_gate_pass": bool(eligible),
                "gate_rule": "primary pooled OR H12 pooled delta < 0, and every W1/W4/H12_E/H12_L delta <= +0.50 MAE",
            }

    result = {
        "experiment_id": "optimization_v2_sparse_hybrid_regression_pilot",
        "stage": "four_fit_pilot_only_no_full_six_continuation",
        "freeze_sha256": file_sha256(FREEZE_PATH),
        "pilot_started_sha256": file_sha256(START_PATH),
        "prediction_cache_path": str(PREDICTION_PATH.relative_to(ROOT)).replace("\\", "/"),
        "prediction_cache_sha256": "written-after-serialization",
        "fit_count": len(receipts),
        "fits": receipts,
        "views": views,
        "candidate_policy_pilot_gate": policy_summaries,
        "full_six_fit_continuation_performed": False,
        "new_control_fits": 0,
        "test_rows_or_labels_used": False,
        "run_completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with PREDICTION_PATH.open("xb") as handle:
        np.savez_compressed(handle, **prediction_cache)
    result["prediction_cache_sha256"] = file_sha256(PREDICTION_PATH)
    _atomic_json_new(PILOT_PATH, result)
    baseline_cache.close()
    print(json.dumps({"pilot_path": str(PILOT_PATH), "prediction_cache": str(PREDICTION_PATH),
                      "prediction_cache_sha256": result["prediction_cache_sha256"],
                      "gate": policy_summaries}, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "pilot"))
    args = parser.parse_args()
    if args.action == "freeze":
        create_freeze()
    else:
        run_pilot()
