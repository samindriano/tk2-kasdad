"""Unscored counterfactual sensitivity for frozen regression representations.

This script refits exactly two frozen models on cycles 47..67 and predicts only
the feature-only validation rows in cycles 68..79. It never reads validation
targets, evaluates error, predicts Kaggle Test, or persists model/prediction
arrays. The only report contains hashes and aggregate prediction deltas.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.regression_research_common import (  # noqa: E402
    CONFIG,
    REPORT,
    digest_json,
    freeze_stage,
    verify_stage,
)
from src.models.regression_research_refinement import (  # noqa: E402
    SCOPE_SPREAD_FEATURES,
    build_refinement,
)
from src.models.regression_research_text import (  # noqa: E402
    _HGB_PARAMS,
    _SVD_PARAMS,
    build_text_model,
)
from src.models.regression_round2_evaluation import (  # noqa: E402
    load_protocol,
    positions_digest,
)
from src.models.round1_evaluation import array_digest  # noqa: E402
from src.utils.paths import task_files  # noqa: E402


FIT_CYCLES = (47, 67)
VALIDATION_CYCLES = (68, 79)
PERTURBED_CATEGORY_FIELDS = ("scope_trace", "org_spread", "module_spread")
PERTURBED_NUMERIC_FIELD = "scope_pressure"
MODEL_IDS = ("svd_hgb", "svd_noscope32")
CACHE_ROOT = PROJECT_ROOT / "experiments" / "runs" / "regression_research_v1"
REPORT_PATH = REPORT / "scope_sensitivity.json"
CACHE_SPECS = {
    "svd_hgb": {
        "stem": "model__svd_hgb__base__fit67__to79",
        "freeze_name": "stage23_models",
        "kind": "svd_hgb",
        "param": "base",
        "report": REPORT / "stage23_models.json",
        "report_path": ("audits", "67", "svd_hgb", "outer"),
    },
    "svd_noscope32": {
        "stem": "refine__svd_noscope32__fit67__to79",
        "freeze_name": "stage4_refinement",
        "kind": "svd_noscope32",
        "param": None,
        "report": REPORT / "stage4_refinement.json",
        "report_path": ("audits", "67", "svd_noscope32", "outer"),
    },
}

PROTOCOL = {
    "purpose": "Unscored counterfactual representation stress; not a cohort-error estimate or model-selection result.",
    "fit_cycles": list(FIT_CYCLES),
    "validation_cycles": list(VALIDATION_CYCLES),
    "models": {
        "svd_hgb": "build_text_model('svd_hgb'), full selected structured allowlist plus signal_stream, 32 SVD dimensions.",
        "svd_noscope32": "build_refinement('svd_noscope32'), same text route and HGB recipe after excluding four scope/spread fields.",
    },
    "hgb_parameters": dict(_HGB_PARAMS),
    "svd_parameters": {**_SVD_PARAMS, "n_components": 32},
    "seed": 42,
    "thread_limit": 2,
    "perturbation": {
        "categorical_fields": {name: "0" for name in PERTURBED_CATEGORY_FIELDS},
        "numeric_field": PERTURBED_NUMERIC_FIELD,
        "numeric_value": "minimum over fit cycles 47..67 minus 1.0",
        "apply_to": "deep copy of validation features only; training data unchanged",
    },
    "validation_targets": "Not loaded into feature frames, not passed to fit, and not used for any metric.",
    "test_data": "No Test rows are read, transformed, predicted, or scored.",
    "evaluation_families": {
        "primary_W4": [68, 71],
        "secondary_H12_L": [68, 79],
        "additional_breakdown": "per cycle 68..79",
    },
    "outputs": "Aggregate deltas, hashes, provenance, and state digests only; no model weights or prediction vectors.",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nested(value: dict[str, Any], path: tuple[str, ...]) -> Any:
    for part in path:
        if not isinstance(value, dict) or part not in value:
            raise RuntimeError(f"Missing expected cache report field: {'.'.join(path)}")
        value = value[part]
    return value


def _validation_positions(cycles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fit = np.flatnonzero((cycles >= FIT_CYCLES[0]) & (cycles <= FIT_CYCLES[1]))
    validation = np.flatnonzero((cycles >= VALIDATION_CYCLES[0]) & (cycles <= VALIDATION_CYCLES[1]))
    if not len(fit) or not len(validation):
        raise RuntimeError("Frozen fit or validation window is empty.")
    if int(cycles[fit].min()) != FIT_CYCLES[0] or int(cycles[fit].max()) != FIT_CYCLES[1]:
        raise RuntimeError("Training cycles differ from the frozen sensitivity window.")
    if int(cycles[validation].min()) != VALIDATION_CYCLES[0] or int(cycles[validation].max()) != VALIDATION_CYCLES[1]:
        raise RuntimeError("Validation cycles differ from the frozen sensitivity window.")
    if int(cycles[fit].max()) >= int(cycles[validation].min()):
        raise RuntimeError("Temporal fit/validation boundary is not strictly increasing.")
    return fit.astype(np.int64), validation.astype(np.int64)


def _read_fit_targets(train_path: Path, fit_positions: np.ndarray) -> np.ndarray:
    """Convert labels only for fit row positions; validation labels stay unused."""
    wanted = set(int(index) for index in fit_positions)
    targets: list[float] = []
    with train_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "threat_rank" not in reader.fieldnames:
            raise RuntimeError("Official Train is missing threat_rank.")
        for index, row in enumerate(reader):
            if index in wanted:
                targets.append(float(row["threat_rank"]))
    result = np.asarray(targets, dtype=np.float64)
    if result.shape != (len(fit_positions),) or not np.isfinite(result).all():
        raise RuntimeError("Fit targets did not align to the declared training positions.")
    return result


def _model_state_digest(model: Any) -> tuple[dict[str, Any], str]:
    state = model.audit_state()
    return state, digest_json(state)


def _load_cache(
    model_id: str,
    *,
    fit_positions: np.ndarray,
    validation_positions: np.ndarray,
    freezes: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """Validate and load a pre-existing validation cache pair, if complete."""
    spec = CACHE_SPECS[model_id]
    stem = spec["stem"]
    meta_path = CACHE_ROOT / f"{stem}.json"
    array_path = CACHE_ROOT / f"{stem}.npz"
    if not meta_path.exists() and not array_path.exists():
        return None
    if meta_path.exists() != array_path.exists():
        raise RuntimeError(f"Existing baseline cache pair is incomplete: {stem}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    with np.load(array_path, allow_pickle=False) as cache:
        if not np.array_equal(cache["fit_rows"], fit_positions):
            raise RuntimeError(f"Baseline cache fit rows changed: {stem}")
        if not np.array_equal(cache["validation_rows"], validation_positions):
            raise RuntimeError(f"Baseline cache validation rows changed: {stem}")
        prediction = np.asarray(cache["prediction"], dtype=np.float64).copy()
    prediction_hash = array_digest(prediction)
    if prediction.shape != (len(validation_positions),) or not np.isfinite(prediction).all():
        raise RuntimeError(f"Baseline cache predictions are invalid: {stem}")
    if meta.get("prediction_sha256") != prediction_hash:
        raise RuntimeError(f"Baseline cache prediction hash changed: {stem}")
    if meta.get("fit_positions_sha256") != positions_digest(fit_positions):
        raise RuntimeError(f"Baseline cache fit-position receipt changed: {stem}")
    if meta.get("validation_positions_sha256") != positions_digest(validation_positions):
        raise RuntimeError(f"Baseline cache validation-position receipt changed: {stem}")
    if meta.get("state_sha256") != digest_json(meta.get("state")):
        raise RuntimeError(f"Baseline cache model-state receipt changed: {stem}")

    expected_key = digest_json({
        "freeze": freezes[spec["freeze_name"]]["signature"],
        "kind": spec["kind"],
        "fit": positions_digest(fit_positions),
        "validation": positions_digest(validation_positions),
        **({"param": spec["param"]} if spec["param"] is not None else {}),
    })
    if meta.get("key") != expected_key:
        raise RuntimeError(f"Baseline cache key does not match its frozen stage: {stem}")

    report_meta = None
    report_path = spec["report"]
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report_meta = _nested(report, spec["report_path"])
        for field in ("key", "prediction_sha256", "state_sha256"):
            if report_meta.get(field) != meta.get(field):
                raise RuntimeError(f"Baseline cache and stage report disagree ({field}): {stem}")

    return {
        "cache_stem": stem,
        "cache_json_sha256": _sha256_file(meta_path),
        "cache_npz_sha256": _sha256_file(array_path),
        "prediction_sha256": prediction_hash,
        "state_sha256": meta["state_sha256"],
        "prediction": prediction,
        "report_present": report_path.exists(),
        "report_meta_checked": report_meta is not None,
    }


def _prediction_quantiles(values: np.ndarray) -> dict[str, float]:
    quantiles = np.quantile(values, [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    names = ("min", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "max")
    return {name: float(value) for name, value in zip(names, quantiles)}


def _delta_summary(delta: np.ndarray) -> dict[str, Any]:
    absolute = np.abs(delta)
    q = np.quantile(absolute, [0.5, 0.9, 0.95, 0.99, 1.0])
    return {
        "n": int(len(delta)),
        "mean_delta": float(np.mean(delta)),
        "median_delta": float(np.median(delta)),
        "mean_absolute_delta": float(np.mean(absolute)),
        "median_absolute_delta": float(np.median(absolute)),
        "absolute_delta_quantiles": {
            "p50": float(q[0]),
            "p90": float(q[1]),
            "p95": float(q[2]),
            "p99": float(q[3]),
            "max": float(q[4]),
        },
        "min_delta": float(np.min(delta)),
        "max_delta": float(np.max(delta)),
        "fraction_abs_delta_gt_1": float(np.mean(absolute > 1.0)),
    }


def _family_summaries(delta: np.ndarray, cycles: np.ndarray, protocol: dict[str, Any]) -> dict[str, Any]:
    windows: dict[str, list[int]] = {}
    for entry in protocol["primary_windows"]:
        if entry["fit"][1] == FIT_CYCLES[1]:
            windows[f"primary_{entry['id']}"] = list(entry["validation"])
    for entry in protocol["secondary_windows"]:
        if entry["fit"][1] == FIT_CYCLES[1]:
            windows[f"secondary_{entry['id']}"] = list(entry["validation"])
    expected = {"primary_W4": [68, 71], "secondary_H12_L": [68, 79]}
    if windows != expected:
        raise RuntimeError("Evaluation-family windows do not match the frozen 68..79 protocol.")
    result: dict[str, Any] = {
        name: _delta_summary(delta[(cycles >= bounds[0]) & (cycles <= bounds[1])])
        for name, bounds in windows.items()
    }
    result["per_cycle"] = {
        str(int(cycle)): _delta_summary(delta[cycles == cycle])
        for cycle in np.unique(cycles)
    }
    return result


def _fit_one(
    model_id: str,
    X_fit: pd.DataFrame,
    y_fit: np.ndarray,
    X_validation: pd.DataFrame,
    X_perturbed: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if model_id == "svd_hgb":
        model = build_text_model("svd_hgb")
    elif model_id == "svd_noscope32":
        model = build_refinement("svd_noscope32")
    else:
        raise ValueError(f"Unknown sensitivity model: {model_id}")
    features = list(model.feature_names)
    start = perf_counter()
    fit_started = datetime.now(timezone.utc).isoformat()
    with threadpool_limits(limits=2):
        model.fit(X_fit.loc[:, features].copy(), y_fit.copy())
        fit_seconds = perf_counter() - start
        state, state_hash = _model_state_digest(model)
        predict_start = perf_counter()
        baseline = np.asarray(model.predict(X_validation.loc[:, features].copy()), dtype=np.float64)
        perturbed = np.asarray(model.predict(X_perturbed.loc[:, features].copy()), dtype=np.float64)
        predict_seconds = perf_counter() - predict_start
        state_after, state_hash_after = _model_state_digest(model)
    if state_hash != state_hash_after or state != state_after:
        raise RuntimeError(f"Prediction mutated fitted state for {model_id}.")
    if baseline.shape != perturbed.shape or baseline.ndim != 1 or not np.isfinite(baseline).all() or not np.isfinite(perturbed).all():
        raise RuntimeError(f"Invalid sensitivity predictions for {model_id}.")
    if model_id == "svd_noscope32" and not np.array_equal(baseline, perturbed):
        raise RuntimeError("No-scope model changed predictions after perturbing excluded fields; exact invariance failed.")
    state_receipt = {
        "model_id": model_id,
        "feature_names_sha256": hashlib.sha256("\n".join(features).encode("utf-8")).hexdigest(),
        "fit_started_at_utc": fit_started,
        "fit_seconds": float(fit_seconds),
        "prediction_seconds": float(predict_seconds),
        "fit_rows": int(len(X_fit)),
        "fit_feature_columns": len(features),
        "fit_target_sha256": state.get("target_sha256", state.get("delegated_model", {}).get("target_sha256")),
        "fit_data_sha256": state.get("fit_data_sha256", state.get("delegated_model", {}).get("fit_data_sha256")),
        "model_state_sha256": state_hash,
        "fitted_tree_sha256": state.get("fitted_tree_sha256", state.get("delegated_model", {}).get("fitted_tree_sha256")),
        "svd_components_actual": state.get("parameters", {}).get("svd", {}).get("n_components_actual"),
        "validation_labels_passed_to_fit": False,
        "test_data_used": False,
    }
    return baseline, perturbed, state_receipt


def _runtime_code_hashes() -> dict[str, str]:
    paths = (
        "scripts/research_regression_sensitivity.py",
        "src/models/regression_research_common.py",
        "src/models/regression_research_text.py",
        "src/models/regression_research_refinement.py",
        "src/models/regression_round2_models.py",
        "src/models/regression_round2_evaluation.py",
        "src/models/round1_models.py",
        "src/models/round1_evaluation.py",
        "src/utils/paths.py",
    )
    return {path: _sha256_file(PROJECT_ROOT / path) for path in paths}


def _load_feature_data(train_path: Path) -> pd.DataFrame:
    full = build_text_model("svd_hgb")
    no_scope = build_refinement("svd_noscope32")
    columns = sorted(set(full.feature_names) | set(no_scope.feature_names))
    features = pd.read_csv(train_path, usecols=columns)
    if len(features) != 50_608:
        raise RuntimeError("Official Train row count changed from the frozen 50,608 rows.")
    return features


def run(*, freeze_only: bool = False) -> dict[str, Any]:
    code = ["scripts/research_regression_sensitivity.py"]
    frozen = freeze_stage("scope_sensitivity", PROTOCOL, code)
    if freeze_only:
        return {
            "status": "PLAN_FROZEN_NO_FIT",
            "stage": "scope_sensitivity",
            "freeze_signature": frozen["signature"],
            "source_head": frozen["source_head"],
            "fit_performed": False,
        }
    if REPORT_PATH.exists():
        raise RuntimeError("Scope-sensitivity report already exists; refusing to overwrite it.")

    train_path = Path(task_files("regression")["train"]).resolve(strict=True)
    train_hash = _sha256_file(train_path)
    frozen_train_hash = frozen["identity"]["regression_input_sha256"]["train"]
    if train_hash != frozen_train_hash:
        raise RuntimeError("Official Train bytes do not match the frozen Round1 input receipt.")

    features = _load_feature_data(train_path)
    cycles = features["cycle_code"].to_numpy()
    fit_positions, validation_positions = _validation_positions(cycles)
    y_fit = _read_fit_targets(train_path, fit_positions)
    fit_cycle = cycles[fit_positions]
    validation_cycle = cycles[validation_positions]
    fit_features = features.iloc[fit_positions].copy()
    validation_features = features.iloc[validation_positions].copy()

    if PERTURBED_NUMERIC_FIELD not in fit_features.columns or any(name not in validation_features.columns for name in PERTURBED_CATEGORY_FIELDS):
        raise RuntimeError("Frozen models are missing one or more declared perturbation fields.")
    training_scope_pressure_min = float(fit_features[PERTURBED_NUMERIC_FIELD].min())
    if not np.isfinite(training_scope_pressure_min):
        raise RuntimeError("Training scope_pressure minimum is not finite.")
    perturbed_features = validation_features.copy(deep=True)
    for feature in PERTURBED_CATEGORY_FIELDS:
        perturbed_features.loc[:, feature] = "0"
    perturbed_scope_pressure = training_scope_pressure_min - 1.0
    perturbed_features.loc[:, PERTURBED_NUMERIC_FIELD] = perturbed_scope_pressure

    protocol = load_protocol()
    fit_digest = positions_digest(fit_positions)
    validation_digest = positions_digest(validation_positions)
    freezes = {
        "stage23_models": json.loads((CONFIG / "stage23_models_freeze.json").read_text(encoding="utf-8")),
        "stage4_refinement": json.loads((CONFIG / "stage4_refinement_freeze.json").read_text(encoding="utf-8")),
    }
    baseline_caches = {
        model_id: _load_cache(
            model_id,
            fit_positions=fit_positions,
            validation_positions=validation_positions,
            freezes=freezes,
        )
        for model_id in MODEL_IDS
    }

    model_results: dict[str, Any] = {}
    for model_id in MODEL_IDS:
        baseline, perturbed, state = _fit_one(
            model_id,
            fit_features,
            y_fit,
            validation_features,
            perturbed_features,
        )
        delta = perturbed - baseline
        cache = baseline_caches[model_id]
        cache_comparison = {
            "status": "CACHE_READY" if cache is not None else "CACHE_NOT_READY_PREDICTION_ONLY_RECEIPT",
            "cache_prediction_sha256": cache["prediction_sha256"] if cache is not None else None,
            "cache_json_sha256": cache["cache_json_sha256"] if cache is not None else None,
            "cache_npz_sha256": cache["cache_npz_sha256"] if cache is not None else None,
            "cache_report_present": cache["report_present"] if cache is not None else False,
            "cache_report_meta_checked": cache["report_meta_checked"] if cache is not None else False,
            "refit_baseline_prediction_exact_equal": bool(np.array_equal(baseline, cache["prediction"])) if cache is not None else None,
            "refit_baseline_max_abs_delta": float(np.max(np.abs(baseline - cache["prediction"]))) if cache is not None else None,
        }
        model_results[model_id] = {
            "fit_receipt": state,
            "baseline_prediction_sha256": array_digest(baseline),
            "perturbed_prediction_sha256": array_digest(perturbed),
            "delta_sha256": array_digest(delta),
            "baseline_prediction_quantiles": _prediction_quantiles(baseline),
            "perturbation": _delta_summary(delta),
            "by_evaluation_family": _family_summaries(delta, validation_cycle, protocol),
            "cache_comparison": cache_comparison,
            "exact_prediction_invariance": bool(np.array_equal(baseline, perturbed)),
        }

    result = {
        "status": "VALID_UNSCORED_SENSITIVITY_ONLY",
        "stage": "scope_sensitivity",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_signature": frozen["signature"],
        "source_head": frozen["source_head"],
        "protocol": PROTOCOL,
        "input_sha256": frozen["identity"]["regression_input_sha256"],
        "train_file_sha256_verified": train_hash,
        "source_code_sha256": _runtime_code_hashes(),
        "fit_rows": int(len(fit_positions)),
        "fit_cycle_range": [int(fit_cycle.min()), int(fit_cycle.max())],
        "fit_positions_sha256": fit_digest,
        "validation_rows": int(len(validation_positions)),
        "validation_cycle_range": [int(validation_cycle.min()), int(validation_cycle.max())],
        "validation_positions_sha256": validation_digest,
        "validation_labels_used_for_metrics_or_fit": False,
        "test_data_used": False,
        "perturbed_values": {
            "categorical_fields": {name: "0" for name in PERTURBED_CATEGORY_FIELDS},
            "scope_pressure_train_min": training_scope_pressure_min,
            "scope_pressure_perturbed": perturbed_scope_pressure,
            "scope_pressure_rule": "minimum among fitted training rows (cycles 47..67) minus 1.0",
        },
        "models": model_results,
        "interpretation_boundary": "Validation counterfactual representation stress only. These perturbations are artificial and are not an estimate of Test-cohort error or a model-selection score.",
    }
    verify_stage("scope_sensitivity")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=True, allow_nan=False, indent=2)
        handle.write("\n")
    return {
        "status": result["status"],
        "report": str(REPORT_PATH.relative_to(PROJECT_ROOT)),
        "freeze_signature": frozen["signature"],
        "fit_rows": result["fit_rows"],
        "validation_rows": result["validation_rows"],
        "models": {
            model_id: {
                "baseline_prediction_sha256": model_result["baseline_prediction_sha256"],
                "perturbed_prediction_sha256": model_result["perturbed_prediction_sha256"],
                "fraction_abs_delta_gt_1": model_result["perturbation"]["fraction_abs_delta_gt_1"],
                "exact_prediction_invariance": model_result["exact_prediction_invariance"],
                "cache_status": model_result["cache_comparison"]["status"],
                "refit_baseline_prediction_exact_equal": model_result["cache_comparison"]["refit_baseline_prediction_exact_equal"],
            }
            for model_id, model_result in model_results.items()
        },
        "validation_labels_used_for_metrics_or_fit": False,
        "test_data_used": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-only", action="store_true", help="Freeze and verify the plan without fitting.")
    args = parser.parse_args(argv)
    result = run(freeze_only=args.freeze_only)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
