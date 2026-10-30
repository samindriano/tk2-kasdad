"""Independently reconcile frozen Regression Round2 reports and ignored vectors."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.audit import sha256, write_json
from src.models.round1_evaluation import array_digest, digest_json, verify_freeze as verify_r1
from src.models.regression_round2_evaluation import (
    DEFAULT_RUN,
    REPORT,
    history_summary,
    load_protocol,
    positions_digest,
    safe_artifact,
    select_history,
    verify_freeze,
)
from src.models.regression_round2_metrics import metric_values
from src.models.regression_round2_models import _SelectedPreprocessor
from src.models.round1_models import CATEGORICAL_FEATURES, EXPANDED_FEATURES, NUMERIC_FEATURES, MIXED_COUNT_FEATURES, TELEMETRY_FEATURE
from src.utils.paths import PROJECT_ROOT, task_files

TOL = 2e-10


def _close(actual: Any, expected: Any, path: str, tol: float = TOL) -> None:
    if actual is None or expected is None:
        if actual is not expected:
            raise AssertionError(f"{path}: {actual!r} != {expected!r}")
        return
    if isinstance(actual, (int, np.integer)) and isinstance(expected, (int, np.integer)):
        if int(actual) != int(expected):
            raise AssertionError(f"{path}: {actual!r} != {expected!r}")
        return
    if not math.isclose(float(actual), float(expected), rel_tol=tol, abs_tol=tol):
        raise AssertionError(f"{path}: {actual!r} != {expected!r}")


def _oracle_metrics(y: Any, pred: Any) -> dict[str, Any]:
    """Recompute core moments independently using math.fsum and sklearn."""
    actual, estimate = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    if actual.ndim != 1 or estimate.ndim != 1 or len(actual) != len(estimate):
        raise ValueError("oracle inputs must be aligned vectors")
    if not len(actual):
        return metric_values(actual, estimate)
    error = estimate - actual
    n = len(actual)
    mae = math.fsum(abs(float(x)) for x in error) / n
    if not math.isclose(mae, float(mean_absolute_error(actual, estimate)), rel_tol=TOL, abs_tol=TOL):
        raise AssertionError("math.fsum MAE disagrees with scikit-learn")
    quantiles = (0, .01, .05, .25, .5, .75, .95, .99, 1)
    pred_quantiles = np.quantile(estimate, quantiles)
    target_quantiles = np.quantile(actual, quantiles)
    return {
        "n": n,
        "mae": mae,
        "rmse": math.sqrt(mean_squared_error(actual, estimate)),
        "signed_error": math.fsum(float(x) for x in error) / n,
        "over_rate": float(np.mean(error > 1e-10)),
        "under_rate": float(np.mean(error < -1e-10)),
        "equal_rate": float(np.mean(np.abs(error) <= 1e-10)),
        "absolute_error_p95": float(np.quantile(np.abs(error), .95)),
        "prediction_quantiles": {str(q): float(v) for q, v in zip(quantiles, pred_quantiles)},
        "target_quantiles": {str(q): float(v) for q, v in zip(quantiles, target_quantiles)},
        "below_zero": int(np.sum(estimate < 0)), "above_100": int(np.sum(estimate > 100)),
    }


def _verify_metric(actual: dict[str, Any], expected: dict[str, Any], path: str) -> None:
    if set(actual) != set(expected):
        raise AssertionError(f"{path}: metric keys differ")
    for key in expected:
        if isinstance(expected[key], dict):
            if actual[key] is None or set(actual[key]) != set(expected[key]):
                raise AssertionError(f"{path}.{key}: quantile keys differ")
            for q, value in expected[key].items():
                _close(actual[key][q], value, f"{path}.{key}.{q}")
        else:
            _close(actual[key], expected[key], f"{path}.{key}")


def _oracle_summary(frame: pd.DataFrame, pred: np.ndarray, *, impact: bool) -> dict[str, Any]:
    actual = np.asarray(frame["threat_rank"], dtype=float)
    estimate = np.asarray(pred, dtype=float)
    cycles = frame["cycle_code"].to_numpy()
    per_cycle = []
    for cycle in sorted(pd.unique(cycles)):
        mask = cycles == cycle
        per_cycle.append({"cycle": int(cycle), **_oracle_metrics(actual[mask], estimate[mask])})
    region_masks = {
        "region_low": actual < (100 / 3),
        "region_middle": (actual >= (100 / 3)) & (actual < (200 / 3)),
        "region_high": actual >= (200 / 3),
    }
    stream_missing = frame["signal_stream"].isna().to_numpy(dtype=bool)
    stream_missing |= frame["signal_stream"].map(lambda v: isinstance(v, str) and not v.strip()).to_numpy(dtype=bool)
    masks = {
        "EF0": frame["entry_friction"].eq(0).to_numpy(), "EF1": frame["entry_friction"].eq(1).to_numpy(),
        **region_masks, "tail_low_le5": actual <= 5, "tail_high_ge95": actual >= 95,
        "stream_missing": stream_missing, "stream_present": ~stream_missing,
    }
    subgroups = {name: _oracle_metrics(actual[mask], estimate[mask]) for name, mask in masks.items()}
    nonempty = [record["mae"] for record in per_cycle if record["mae"] is not None]
    impact_groups = {}
    if impact:
        impacts = frame["impact_load"].to_numpy()
        for value in sorted(pd.unique(impacts)):
            mask = impacts == value
            key = repr(float(value))
            by_region = {name: _oracle_metrics(actual[mask & region], estimate[mask & region])
                         for name, region in region_masks.items()}
            impact_groups[key] = {"impact_load": float(value),
                                  "metrics": _oracle_metrics(actual[mask], estimate[mask]),
                                  "by_target_region": by_region}
    return {"raw": _oracle_metrics(actual, estimate),
            "clipped": _oracle_metrics(actual, np.clip(estimate, 0, 100)),
            "equal_cycle_mae": float(np.mean(nonempty)) if nonempty else None,
            "per_cycle": per_cycle, "subgroups": subgroups, "impact_groups": impact_groups}


def _oracle_paired(frame: pd.DataFrame, candidate: np.ndarray, control: np.ndarray) -> dict[str, Any]:
    actual = frame["threat_rank"].to_numpy(dtype=float)
    candidate_error, control_error = np.asarray(candidate, dtype=float)-actual, np.asarray(control, dtype=float)-actual
    delta = np.abs(candidate_error)-np.abs(control_error)
    win, loss = delta < -1e-10, delta > 1e-10
    n = len(actual)
    per_cycle = []
    cycles = frame["cycle_code"].to_numpy()
    for cycle in sorted(pd.unique(cycles)):
        mask = cycles == cycle
        per_cycle.append({"cycle": int(cycle), "delta_mae": math.fsum(float(x) for x in delta[mask])/int(mask.sum())})
    candidate_signed = math.fsum(float(x) for x in candidate_error)/n
    control_signed = math.fsum(float(x) for x in control_error)/n
    return {"n": n, "delta_mae": math.fsum(float(x) for x in delta)/n,
            "candidate_signed_error": candidate_signed, "control_signed_error": control_signed,
            "signed_error_delta": candidate_signed-control_signed,
            "candidate_win_rate": float(np.mean(win)), "control_win_rate": float(np.mean(loss)),
            "tie_rate": float(np.mean(~win & ~loss)), "per_cycle": per_cycle}


def _verify_summary(record: dict[str, Any], frame: pd.DataFrame, pred: np.ndarray, *, impact: bool, path: str) -> None:
    expected = _oracle_summary(frame, pred, impact=impact)
    if impact and set(record["impact_groups"]) != set(expected["impact_groups"]):
        raise AssertionError(f"{path}: exact impact-group membership differs")
    if not impact and record.get("impact_groups") != {}:
        raise AssertionError(f"{path}: detailed impact groups must be omitted on per-window job records")
    _verify_metric(record["raw"], _oracle_metrics(frame.threat_rank.to_numpy(), pred), path+".raw")
    _verify_metric(record["clipped"], _oracle_metrics(frame.threat_rank.to_numpy(), np.clip(pred, 0, 100)), path+".clipped")
    if record["raw"]["n"] != len(frame):
        raise AssertionError(f"{path}: denominator does not match row count")
    _close(record["equal_cycle_mae"], expected["equal_cycle_mae"], path+".equal_cycle_mae")
    if len(record["per_cycle"]) != len(expected["per_cycle"]):
        raise AssertionError(f"{path}: cycle set differs")
    for got, want in zip(record["per_cycle"], expected["per_cycle"]):
        if got["cycle"] != want["cycle"]:
            raise AssertionError(f"{path}: cycle order differs")
        _verify_metric({k: got[k] for k in want if k != "cycle"},
                       {k: want[k] for k in want if k != "cycle"}, f"{path}.cycle{got['cycle']}")
    for subgroup, want in expected["subgroups"].items():
        if subgroup not in record["subgroups"]:
            raise AssertionError(f"{path}: missing subgroup {subgroup}")
        _verify_metric(record["subgroups"][subgroup], want, f"{path}.{subgroup}")
    masks = ["EF0", "EF1"]
    if sum(record["subgroups"][k]["n"] for k in masks) != len(frame):
        raise AssertionError(f"{path}: EF subgroup denominators do not partition rows")
    if sum(record["subgroups"][k]["n"] for k in ("region_low", "region_middle", "region_high")) != len(frame):
        raise AssertionError(f"{path}: target region denominators do not partition rows")
    if sum(record["subgroups"][k]["n"] for k in ("stream_missing", "stream_present")) != len(frame):
        raise AssertionError(f"{path}: stream subgroup denominators do not partition rows")
    if impact:
        counts = [v["metrics"]["n"] for v in record["impact_groups"].values()]
        if sum(counts) != len(frame):
            raise AssertionError(f"{path}: impact-group denominators do not partition rows")
        for key, want in expected["impact_groups"].items():
            got = record["impact_groups"][key]
            if got["impact_load"] != want["impact_load"]:
                raise AssertionError(f"{path}: impact value mismatch for {key}")
            _verify_metric(got["metrics"], want["metrics"], f"{path}.impact.{key}")
            for region in want["by_target_region"]:
                _verify_metric(got["by_target_region"][region], want["by_target_region"][region],
                               f"{path}.impact.{key}.{region}")


def _verify_model_preprocessing(state: dict[str, Any], features: list[str], fit: np.ndarray,
                                frame: pd.DataFrame, protocol: dict[str, Any], path: str) -> None:
    if state.get("input_feature_names") != features or state.get("selected_source_features", features) != features:
        raise AssertionError(f"{path}: fitted input features differ from frozen allowlist")
    if "record_id" in features or "threat_rank" in features or set(features) - set(EXPANDED_FEATURES):
        raise AssertionError(f"{path}: forbidden/non-eligible feature reached fitted state")
    if state.get("fit_rows") != len(fit):
        raise AssertionError(f"{path}: fitted row count mismatch")
    expected_fp_source = frame.iloc[fit].loc[:, features]
    from src.models.round1_models import _fit_fingerprint
    expected_fp = _fit_fingerprint(expected_fp_source, tuple(features), frame.iloc[fit].threat_rank.to_numpy())
    if state.get("fit_data_sha256") != expected_fp:
        raise AssertionError(f"{path}: fit-only input fingerprint mismatch")
    parameters = state.get("actual_hyperparameters", {})
    for name, expected in protocol["hgb_fixed_parameters"].items():
        if name not in parameters:
            raise AssertionError(f"{path}: frozen HGB parameter {name} absent from fitted audit")
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            _close(parameters[name], expected, f"{path}.parameters.{name}")
        elif parameters[name] != expected:
            raise AssertionError(f"{path}: frozen HGB parameter {name} differs from protocol")
    transformed_names = state.get("transformed_feature_names", [])
    expected_mask = [name.endswith("__ordinal") for name in transformed_names]
    if any(expected_mask):
        expected_categorical = expected_mask
    else:
        expected_categorical = None
    if parameters.get("categorical_features") != expected_categorical:
        raise AssertionError(f"{path}: native categorical mask differs from transformed feature schema")
    if features in [list(("impact_load", "entry_friction"))]:
        return
    processor = _SelectedPreprocessor(tuple(features))
    _, _ = processor.fit_transform(frame.iloc[fit][features].copy())
    pstate = processor.audit_state()
    if state.get("transformed_feature_names") != pstate["transformed_feature_names"]:
        raise AssertionError(f"{path}: transformed feature schema mismatch")
    if state.get("numeric_imputer_medians") != pstate["numeric_imputer_medians"]:
        raise AssertionError(f"{path}: numeric imputation state not reproducible from fit rows")
    if state.get("category_counts") != pstate["category_counts"]:
        raise AssertionError(f"{path}: category counts not reproducible from fit rows")
    if state.get("ordinal_encoder") != pstate["ordinal_encoder"]:
        raise AssertionError(f"{path}: category encoder settings mismatch")


def _result_digest_binding(receipt: dict[str, Any], freeze_signature: str, results_sha256: str,
                           vector_manifest_sha256: str, audit_manifest_sha256: str) -> bool:
    return (receipt.get("status") == "PASS" and receipt.get("freeze_signature") == freeze_signature
            and receipt.get("results_sha256") == results_sha256
            and receipt.get("vector_manifest_sha256") == vector_manifest_sha256
            and receipt.get("fit_audit_manifest_sha256") == audit_manifest_sha256)


def _audit_manifest(root: Path) -> str:
    manifest = {str(path.relative_to(root)): sha256(path) for path in sorted(root.glob("fit*.json"))}
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _repeat_receipt_binding_valid(repeat_receipt: dict[str, Any]) -> bool:
    """Recheck prior repeat report/cache hashes before carrying PASS forward."""
    try:
        report_path = Path(repeat_receipt["report_path"]).resolve()
        if not report_path.exists() or sha256(report_path) != repeat_receipt["report_sha256"]:
            return False
        report = json.loads(report_path.read_text(encoding="utf-8"))
        artifact = safe_artifact(report["run_id"])
        if report.get("freeze_signature") != repeat_receipt.get("freeze_signature"):
            return False
        vector_hashes = {}
        for end, model in sorted({(int(job["fit_end"]), job["model"]) for job in report["jobs"]}):
            path = artifact/f"fit{end}__{model}.npz"
            with np.load(path, allow_pickle=False) as arrays:
                fit_rows, val_rows, prediction = arrays["fit_rows"].copy(), arrays["validation_rows"].copy(), arrays["prediction"].copy()
            vector_hashes[f"{end}/{model}"] = {
                "fit_rows_sha256": positions_digest(fit_rows),
                "validation_rows_sha256": positions_digest(val_rows),
                "prediction_sha256": array_digest(prediction),
            }
        manifest = hashlib.sha256(json.dumps(vector_hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        audit_manifest = _audit_manifest(report_path.parent/"fit_audits")
        return (manifest == repeat_receipt.get("vector_manifest_sha256")
                and audit_manifest == repeat_receipt.get("fit_audit_manifest_sha256"))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _assert_reference_vector(rows, pred, expected_rows, audit, reference_path):
    """Use immutable committed hashes when ignored historical arrays are absent."""
    if not np.array_equal(rows, expected_rows) or array_digest(pred) != audit['prediction_sha256']:
        raise AssertionError('Control differs from committed Round1 positions/prediction hash')
    if reference_path.exists():
        with np.load(reference_path, allow_pickle=False) as old:
            if not np.array_equal(old['validation_rows'], rows) or not np.array_equal(old['prediction'], pred):
                raise AssertionError('Historical reference vector differs from committed hash/current control')


def _verify_control_r1(frame: pd.DataFrame, protocol: dict[str, Any], audits: dict[str, Any],
                       artifact: Path, path: Path, reference_root: Path | None = None) -> int:
    count = 0
    r1_protocol = json.loads((PROJECT_ROOT/"experiments/round1/protocol.json").read_text(encoding="utf-8"))
    r1_windows = {w["id"]: w for w in r1_protocol["primary_windows"] + r1_protocol["secondary_windows"]}
    r1_root = reference_root if reference_root is not None else PROJECT_ROOT/"experiments/runs"/"round1_v1"
    for end in [55, 59, 63, 67, 71, 75]:
        for model_id in ("R1_C", "R1_D"):
            recipe = next((r for r in protocol["recipes"] if r["id"] == model_id and r["history"] == "expanding"), None)
            if recipe is None or end not in {w["fit"][1] for w in protocol["primary_windows"]}:
                continue
            r2_audit = audits[str(end)][model_id]
            r1_window = next(w for w in protocol["primary_windows"] if w["fit"][1] == end)
            audit1_path = PROJECT_ROOT/"reports/round1/fit_audits"/(r1_window["id"]+".json")
            audit1 = json.loads(audit1_path.read_text(encoding="utf-8"))[model_id]
            if r2_audit["fitted_state"] != audit1["fitted_state"]:
                raise AssertionError(f"R1 exact control fitted state mismatch: {end}/{model_id}")
            artifact2 = artifact/f"fit{end}__{model_id}.npz"
            r2_vector = np.load(artifact2, allow_pickle=False)
            rows2, pred2 = r2_vector["validation_rows"], r2_vector["prediction"]
            r1_path = r1_root/(r1_window["id"]+"__"+model_id+".npz")
            primary_mask = frame.iloc[rows2].cycle_code.between(*r1_window["validation"]).to_numpy()
            expected_rows = np.flatnonzero(frame.cycle_code.between(*r1_window['validation']).to_numpy())
            _assert_reference_vector(rows2[primary_mask], pred2[primary_mask], expected_rows, audit1, r1_path)
            count += 1
    for h12_id, end in (("H12_E", 55), ("H12_L", 67)):
        audit1 = json.loads((PROJECT_ROOT/"reports/round1/fit_audits"/(h12_id+".json")).read_text(encoding="utf-8"))
        for model_id in ("R1_C", "R1_D"):
            if audits[str(end)][model_id]["fitted_state"] != audit1[model_id]["fitted_state"]:
                raise AssertionError(f"R1 exact control H12 fitted state mismatch: {h12_id}/{model_id}")
            with np.load(artifact/f"fit{end}__{model_id}.npz", allow_pickle=False) as current:
                rows2, pred2 = current["validation_rows"].copy(), current["prediction"].copy()
            if audits[str(end)][model_id]["prediction_sha256"] != array_digest(pred2):
                raise AssertionError(f"Round2 H12 control vector hash mismatch: {h12_id}/{model_id}")
            expected_rows = np.flatnonzero(frame.cycle_code.between(*r1_windows[h12_id]['validation']).to_numpy())
            _assert_reference_vector(rows2, pred2, expected_rows, audit1[model_id], r1_root/(h12_id+'__'+model_id+'.npz'))
            count += 1
    return count


def verify_results(results_path: Path | str, *, data_dir: Path | str | None = None,
                   repeat_report: Path | str | None = None, write_receipt: bool = True) -> dict[str, Any]:
    results_path = Path(results_path).resolve()
    results = json.loads(results_path.read_text(encoding="utf-8"))
    if results.get("status") != "REGRESSION_ROUND2_DEVELOPMENT_REVIEW_NO_FINAL_MODEL":
        raise AssertionError("Unexpected Round2 result status")
    if results.get("partial"):
        raise AssertionError("Verifier requires a complete results file")
    frozen = verify_freeze(data_dir)
    r1 = verify_r1(data_dir)
    if results["freeze_signature"] != frozen["signature"]:
        raise AssertionError("Result is not bound to current frozen protocol")
    protocol = load_protocol()
    recipes = results["recipes"]
    expected_recipes = list(protocol["recipes"])
    if results.get("stage") == "all":
        from src.models.regression_round2_evaluation import load_interaction_selection
        selection = load_interaction_selection(data_dir)
        if results.get("interaction_signature") != selection["signature"]:
            raise AssertionError("Interaction recipe signature mismatch")
        expected_recipes += selection["payload"]["recipes"]
    elif results.get("stage") != "base" or results.get("interaction_signature") is not None:
        raise AssertionError("Unexpected stage or interaction signature")
    if recipes != expected_recipes:
        raise AssertionError("Reported recipe list differs from frozen protocol/interaction selection")

    files = task_files("regression", data_dir)
    if files["train"] is None:
        raise FileNotFoundError("Frozen official Train CSV is unavailable")
    frame = pd.read_csv(files["train"])
    if not frame.record_id.is_unique or frame.record_id.isna().any() or not np.isfinite(frame.threat_rank).all():
        raise AssertionError("Official training table has invalid row identity/target")
    run_id = results["run_id"]
    artifact = safe_artifact(run_id)
    if results_path.name == "results_base.json" and results_path.parent == REPORT.resolve():
        audit_root = results_path.parent/"fit_audits"
    elif results_path.name == "results.json" and results_path.parent == REPORT.resolve():
        audit_root = results_path.parent/"fit_audits"
    else:
        audit_root = results_path.parent/"fit_audits"
    audit_files = sorted(audit_root.glob("fit*.json"))
    audits = {str(p.stem.removeprefix("fit")): json.loads(p.read_text(encoding="utf-8")) for p in audit_files}

    windows = [("primary", w) for w in protocol["primary_windows"]] + [("secondary", w) for w in protocol["secondary_windows"]]
    end_windows: dict[int, list[dict[str, Any]]] = {}
    for family, w in windows:
        end_windows.setdefault(w["fit"][1], []).append(w)
    if set(audits) != {str(end) for end in end_windows}:
        raise AssertionError("Fit audit history endpoints do not match complete frozen window set")

    row_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    vector_hashes: dict[str, str] = {}
    validated_cache_files = 0
    for end, ewindows in sorted(end_windows.items()):
        end = int(end)
        last = max(w["validation"][1] for w in ewindows)
        val_rows = np.flatnonzero(frame.cycle_code.between(end+1, last).to_numpy())
        broad_key = f"fit{end}"
        for recipe in recipes:
            fit_rows = select_history(frame, end, protocol["history_policies"][recipe["history"]])
            if not len(fit_rows) or not len(val_rows) or frame.iloc[fit_rows].cycle_code.max() >= frame.iloc[val_rows].cycle_code.min():
                raise AssertionError(f"Noncausal fold: fit{end}/{recipe['id']}")
            if np.intersect1d(fit_rows, val_rows).size:
                raise AssertionError(f"Overlapping fit/validation positions: {end}/{recipe['id']}")
            if str(end) not in audits or recipe["id"] not in audits[str(end)]:
                raise AssertionError(f"Missing fitted audit: {end}/{recipe['id']}")
            meta = audits[str(end)][recipe["id"]]
            features = protocol["feature_sets"][recipe["features"]]
            cache_path = artifact/f"{broad_key}__{recipe['id']}.npz"
            meta_path = artifact/f"{broad_key}__{recipe['id']}.json"
            with np.load(cache_path, allow_pickle=False) as stored:
                if set(stored.files) != {"fit_rows", "validation_rows", "prediction"}:
                    raise AssertionError(f"Unexpected cache arrays: {cache_path.name}")
                saved_fit, saved_val, pred = stored["fit_rows"].copy(), stored["validation_rows"].copy(), stored["prediction"].copy()
            if not np.array_equal(saved_fit, fit_rows) or not np.array_equal(saved_val, val_rows):
                raise AssertionError(f"Saved source positions differ from frozen policy: {end}/{recipe['id']}")
            if pred.shape != (len(val_rows),) or not np.isfinite(pred).all():
                raise AssertionError(f"Invalid prediction vector: {end}/{recipe['id']}")
            if not meta_path.exists():
                raise AssertionError(f"Missing cache metadata: {meta_path.name}")
            cache_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            expected_key = digest_json({"freeze": frozen["signature"], "features": features,
                                        "control": recipe["control"], "fit_rows": positions_digest(fit_rows),
                                        "validation_rows": positions_digest(val_rows)})
            if cache_meta != meta:
                raise AssertionError(f"Fit audit and cache metadata differ: {end}/{recipe['id']}")
            if meta["job_key"] != expected_key:
                raise AssertionError(f"Cache key mismatch: {end}/{recipe['id']}")
            if meta["prediction_sha256"] != array_digest(pred):
                raise AssertionError(f"Prediction hash mismatch: {end}/{recipe['id']}")
            if meta["fit_positions_sha256"] != positions_digest(fit_rows) or meta["validation_positions_sha256"] != positions_digest(val_rows):
                raise AssertionError(f"Position hash mismatch: {end}/{recipe['id']}")
            if meta["fitted_state_sha256"] != digest_json(meta["fitted_state"]):
                raise AssertionError(f"Fitted state hash mismatch: {end}/{recipe['id']}")
            if meta.get("validation_labels_passed_to_fit") is not False or meta.get("early_stopping_validation_used") is not False:
                raise AssertionError(f"Fit metadata breaks label boundary: {end}/{recipe['id']}")
            if meta["fit_rows"] != len(fit_rows) or meta["validation_rows"] != len(val_rows):
                raise AssertionError(f"Fit or validation denominator mismatch: {end}/{recipe['id']}")
            _verify_model_preprocessing(meta["fitted_state"], features, fit_rows, frame, protocol,
                                        f"{end}/{recipe['id']}")
            complexity = meta["complexity"]
            trees = complexity["trees_fitted"]
            if (complexity["iterations_fitted"] != protocol["hgb_fixed_parameters"]["max_iter"]
                    or trees != complexity["iterations_fitted"]):
                raise AssertionError(f"Unexpected estimator complexity: {end}/{recipe['id']}")
            if len(complexity["node_counts_per_tree"]) != trees or len(complexity["leaf_counts_per_tree"]) != trees:
                raise AssertionError(f"Tree complexity counts are inconsistent: {end}/{recipe['id']}")
            if sum(complexity["node_counts_per_tree"]) != complexity["total_nodes"] or sum(complexity["leaf_counts_per_tree"]) != complexity["total_leaves"]:
                raise AssertionError(f"Tree complexity totals are inconsistent: {end}/{recipe['id']}")
            if any(l > n for l, n in zip(complexity["leaf_counts_per_tree"], complexity["node_counts_per_tree"])):
                raise AssertionError(f"Invalid tree leaves/nodes: {end}/{recipe['id']}")
            vector_key = f"{end}/{recipe['id']}"
            vector_hashes[vector_key] = {"fit_rows_sha256": positions_digest(fit_rows),
                                         "validation_rows_sha256": positions_digest(val_rows),
                                         "prediction_sha256": array_digest(pred)}
            row_cache[end, recipe["id"]] = (val_rows, pred)
            validated_cache_files += 1

    actual_fit_evaluations = len(end_windows)*len(recipes)
    if results.get("fresh_fits", 0) + results.get("reused_fits_or_aliases", 0) != actual_fit_evaluations:
        raise AssertionError("Fresh/reused fit execution receipt does not reconcile to recipe histories")
    unique_keys = {audits[str(end)][recipe["id"]]["job_key"]
                   for end in end_windows for recipe in recipes}
    if results.get("unique_fit_keys") != len(unique_keys):
        raise AssertionError("Unique fit-key count does not match fitted audit entries")

    # History-support descriptions are source-feature-only and must be bound to
    # the same strictly past row positions declared by each frozen history rule.
    expected_history_keys = set()
    for end, ewindows in sorted(end_windows.items()):
        last = max(w["validation"][1] for w in ewindows)
        val = np.flatnonzero(frame.cycle_code.between(end+1, last).to_numpy())
        for history in sorted({r["history"] for r in recipes}):
            key = f"{end}__{history}"
            expected_history_keys.add(key)
            fit = select_history(frame, end, protocol["history_policies"][history])
            expected_support = history_summary(frame, fit, val)
            if results.get("history_support", {}).get(key) != expected_support:
                raise AssertionError(f"History support or feature-only row fingerprint mismatch: {key}")
    if set(results.get("history_support", {})) != expected_history_keys:
        raise AssertionError("History support keys differ from frozen history/window grid")

    # Reconcile validation metrics and counts from cache vectors, never report JSON alone.
    job_lookup = {(job["family"], job["window"], job["model"]): job for job in results["jobs"]}
    expected_job_count = len(windows)*len(recipes)
    if len(job_lookup) != expected_job_count or len(results["jobs"]) != expected_job_count:
        raise AssertionError("Result job count or uniqueness differs from complete frozen design")
    for family, window in windows:
        end = window["fit"][1]
        broad_rows, broad_pred = row_cache[end, recipes[0]["id"]]
        mask = frame.iloc[broad_rows].cycle_code.between(*window["validation"]).to_numpy()
        rows, pred = broad_rows[mask], broad_pred[mask]
        view = frame.iloc[rows]
        for recipe in recipes:
            pred_recipe = row_cache[end, recipe["id"]][1][mask]
            job = job_lookup[family, window["id"], recipe["id"]]
            if job["fit_end"] != end or job["features"] != recipe["features"] or job["history"] != recipe["history"]:
                raise AssertionError(f"Job recipe/window binding mismatch: {family}/{window['id']}/{recipe['id']}")
            if job["validation_positions_sha256"] != positions_digest(rows):
                raise AssertionError(f"Job validation position digest mismatch: {family}/{window['id']}/{recipe['id']}")
            _verify_summary(job, view, pred_recipe, impact=False, path=f"job.{family}.{window['id']}.{recipe['id']}")

    expected_aggregate: dict[str, Any] = {}
    for family in ("primary", "secondary"):
        family_windows = [window for f, window in windows if f == family]
        expected_aggregate[family] = {}
        for recipe in recipes:
            row_parts, pred_parts = [], []
            for window in family_windows:
                row, prediction = row_cache[window["fit"][1], recipe["id"]]
                mask = frame.iloc[row].cycle_code.between(*window["validation"]).to_numpy()
                row_parts.append(row[mask]); pred_parts.append(prediction[mask])
            rows = np.concatenate(row_parts); predictions = np.concatenate(pred_parts)
            order = np.argsort(rows); rows, predictions = rows[order], predictions[order]
            if len(np.unique(rows)) != len(rows):
                raise AssertionError(f"Duplicate validation rows in {family}/{recipe['id']} aggregate")
            record = results["aggregate"][family][recipe["id"]]
            _verify_summary(record, frame.iloc[rows], predictions, impact=True, path=f"aggregate.{family}.{recipe['id']}")
            expected_aggregate[family][recipe["id"]] = (rows, predictions)

    # Paired candidate/control effects and window deltas use identical validation rows.
    comparison_lookup = {(x["family"], x["candidate"], x["control"]): x for x in results["comparisons"]}
    if len(comparison_lookup) != len(results["comparisons"]):
        raise AssertionError("Duplicate paired comparison key")
    for family in ("primary", "secondary"):
        for recipe in recipes:
            candidate, _ = expected_aggregate[family][recipe["id"]]
            for control in ("R1_C", "R1_D"):
                if recipe["id"] == control and family == "primary":
                    # Self-comparison is valid and has exactly zero paired error.
                    pass
                expected_rows, candidate_pred = expected_aggregate[family][recipe["id"]]
                control_rows, control_pred = expected_aggregate[family][control]
                if not np.array_equal(expected_rows, control_rows):
                    raise AssertionError("Candidate/control validation alignment mismatch")
                actual_frame = frame.iloc[expected_rows]
                expected = _oracle_paired(actual_frame, candidate_pred, control_pred)
                got = comparison_lookup[family, recipe["id"], control]
                for key in ("n", "delta_mae", "candidate_signed_error", "control_signed_error", "signed_error_delta",
                            "candidate_win_rate", "control_win_rate", "tie_rate"):
                    _close(got[key], expected[key], f"paired.{family}.{recipe['id']}.{control}.{key}")
                if len(got["per_cycle"]) != len(expected["per_cycle"]):
                    raise AssertionError("Paired per-cycle record count mismatch")
                for a, b in zip(got["per_cycle"], expected["per_cycle"]):
                    if a["cycle"] != b["cycle"]:
                        raise AssertionError("Paired cycle mismatch")
                    _close(a["delta_mae"], b["delta_mae"], "paired.per_cycle.delta")
                if family == "primary":
                    window_delta = []
                    for window in protocol["primary_windows"]:
                        ra, pa = row_cache[window["fit"][1], recipe["id"]]
                        rb, pb = row_cache[window["fit"][1], control]
                        if not np.array_equal(ra, rb):
                            raise AssertionError("Window paired rows differ")
                        select = frame.iloc[ra].cycle_code.between(*window["validation"]).to_numpy()
                        y = frame.iloc[ra[select]].threat_rank.to_numpy()
                        da = _oracle_metrics(y, pa[select])["mae"]
                        db = _oracle_metrics(y, pb[select])["mae"]
                        window_delta.append({"window": window["id"], "delta_mae": da-db})
                    if [x["window"] for x in got["window_deltas"]] != [x["window"] for x in window_delta]:
                        raise AssertionError("Paired window ordering mismatch")
                    for a, b in zip(got["window_deltas"], window_delta):
                        _close(a["delta_mae"], b["delta_mae"], "paired.window.delta")
                    improved = sum(x["delta_mae"] < -1e-10 for x in window_delta)
                    if got["windows_improved"] != improved:
                        raise AssertionError("Paired improved-window count mismatch")

    primary_rows = expected_aggregate["primary"][recipes[0]["id"]][0]
    secondary_rows = expected_aggregate["secondary"][recipes[0]["id"]][0]
    if len(primary_rows) != int(results["aggregate"]["primary"][recipes[0]["id"]]["raw"]["n"]):
        raise AssertionError("Primary row denominator mismatch")
    if len(secondary_rows) != int(results["aggregate"]["secondary"][recipes[0]["id"]]["raw"]["n"]):
        raise AssertionError("Secondary row denominator mismatch")
    overlap = int(np.intersect1d(primary_rows, secondary_rows).size)
    if overlap != len(secondary_rows):
        raise AssertionError("H12 windows are expected to reuse primary validation rows")

    # Each expanding control must reproduce the original Round1 predictions and states.
    exact_r1_controls = _verify_control_r1(frame, protocol, audits, artifact, results_path)

    repeat_verification = {"status": "UNKNOWN", "checks": 0}
    if repeat_report is not None:
        repeat_path = Path(repeat_report).resolve()
        repeat = json.loads(repeat_path.read_text(encoding="utf-8"))
        if repeat.get("partial") is not True:
            raise AssertionError("Representative repeat is expected to cover one designated fit history")
        if (repeat.get("stage") != results.get("stage") or repeat.get("recipes") != results["recipes"]
                or repeat.get("interaction_signature") != results.get("interaction_signature")):
            raise AssertionError("Representative repeat stage, recipe set, or interaction signature differs from final results")
        if repeat.get("freeze_signature") != frozen["signature"]:
            raise AssertionError("Repeat freeze signature mismatch")
        if {int(job["fit_end"]) for job in repeat["jobs"]} != {67}:
            raise AssertionError("Representative repeat must contain only the preregistered fit_end=67 history")
        if (repeat.get("fresh_fits") != len(recipes) or repeat.get("unique_fit_keys") != len(recipes)
                or repeat.get("reused_fits_or_aliases") != 0):
            raise AssertionError("Representative repeat must freshly fit exactly one model per final recipe at fit_end=67")
        repeat_artifact = safe_artifact(repeat["run_id"])
        repeat_audits = {str(p.stem.removeprefix("fit")): json.loads(p.read_text(encoding="utf-8"))
                         for p in (repeat_path.parent/"fit_audits").glob("fit*.json")}
        repeat_lookup = {(job["family"], job["window"], job["model"]): job for job in repeat["jobs"]}
        repeat_windows = [(family, window) for family, window in windows if window["fit"][1] == 67]
        expected_repeat_jobs = {(family, window["id"], recipe["id"])
                                for family, window in repeat_windows for recipe in recipes}
        if set(repeat_lookup) != expected_repeat_jobs or len(repeat_lookup) != len(repeat["jobs"]):
            raise AssertionError("Representative repeat job set differs from the preregistered fit-end subset")
        if set(repeat_audits) != {"67"} or set(repeat_audits["67"]) != {r["id"] for r in recipes}:
            raise AssertionError("Representative repeat fit audits differ from frozen fit-end/recipe set")
        checks = 0
        repeat_vector_hashes = {}
        for job in repeat["jobs"]:
            end, model = int(job["fit_end"]), job["model"]
            path_main = artifact/f"fit{end}__{model}.npz"
            path_repeat = repeat_artifact/f"fit{end}__{model}.npz"
            with np.load(path_main, allow_pickle=False) as a, np.load(path_repeat, allow_pickle=False) as b:
                main_fit_rows, main_val_rows = a["fit_rows"].copy(), a["validation_rows"].copy()
                repeat_fit_rows, repeat_val_rows, repeat_pred = b["fit_rows"].copy(), b["validation_rows"].copy(), b["prediction"].copy()
                if not np.array_equal(main_fit_rows, repeat_fit_rows) or not np.array_equal(main_val_rows, repeat_val_rows):
                    raise AssertionError(f"Repeat positions differ: {end}/{model}")
                if not np.array_equal(a["prediction"], repeat_pred):
                    raise AssertionError(f"Repeat prediction differs: {end}/{model}")
            main_audit, repeat_audit = audits[str(end)][model], repeat_audits[str(end)][model]
            if main_audit["fitted_state_sha256"] != repeat_audit["fitted_state_sha256"]:
                raise AssertionError(f"Repeat state hash differs: {end}/{model}")
            if repeat_audit["fitted_state_sha256"] != digest_json(repeat_audit["fitted_state"]):
                raise AssertionError(f"Repeat fitted state receipt hash mismatch: {end}/{model}")
            if main_audit["complexity"] != repeat_audit["complexity"]:
                raise AssertionError(f"Repeat model complexity differs: {end}/{model}")
            repeat_cache_meta = json.loads((repeat_artifact/f"fit{end}__{model}.json").read_text(encoding="utf-8"))
            if repeat_cache_meta != repeat_audit:
                raise AssertionError(f"Repeat report and cache fit audit differ: {end}/{model}")
            if repeat_audit["fit_positions_sha256"] != positions_digest(repeat_fit_rows):
                raise AssertionError(f"Repeat fit position hash mismatch: {end}/{model}")
            if repeat_audit["validation_positions_sha256"] != positions_digest(repeat_val_rows):
                raise AssertionError(f"Repeat validation position hash mismatch: {end}/{model}")
            if repeat_audit["prediction_sha256"] != array_digest(repeat_pred):
                raise AssertionError(f"Repeat metadata prediction hash mismatch: {end}/{model}")
            repeat_vector_hashes[f"{end}/{model}"] = {
                "fit_rows_sha256": positions_digest(repeat_fit_rows),
                "validation_rows_sha256": positions_digest(repeat_val_rows),
                "prediction_sha256": array_digest(repeat_pred),
            }
            family, window = next((family, window) for family, window in repeat_windows if window["id"] == job["window"])
            source_rows = repeat_val_rows
            mask = frame.iloc[source_rows].cycle_code.between(*window["validation"]).to_numpy()
            view = frame.iloc[source_rows[mask]]
            _verify_summary(job, view, repeat_pred[mask], impact=False,
                            path=f"repeat.{family}.{window['id']}.{model}")
            checks += 1
        if checks != len(expected_repeat_jobs):
            raise AssertionError("Repeat verification did not cover every expected evaluation job")
        repeat_manifest_hash = hashlib.sha256(
            json.dumps(repeat_vector_hashes, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        repeat_verification = {"status": "PASS", "checks": checks,
                               "freeze_signature": frozen["signature"],
                               "report_path": str(repeat_path), "report_sha256": sha256(repeat_path),
                               "vector_manifest_sha256": repeat_manifest_hash,
                               "fit_audit_manifest_sha256": _audit_manifest(repeat_path.parent/"fit_audits")}

    result_hash = sha256(results_path)
    manifest_hash = hashlib.sha256(json.dumps(vector_hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    audit_manifest_hash = _audit_manifest(audit_root)
    verification_path = results_path.parent/"verification.json"
    prior = json.loads(verification_path.read_text(encoding="utf-8")) if verification_path.exists() else None
    previous_repeat = {"status": "UNKNOWN", "checks": 0}
    if prior and _result_digest_binding(prior, frozen["signature"], result_hash, manifest_hash, audit_manifest_hash):
        old_repeat = prior.get("repeat_verification", {})
        if old_repeat.get("status") == "PASS" and _repeat_receipt_binding_valid(old_repeat):
            previous_repeat = old_repeat
    if repeat_verification["status"] != "PASS":
        repeat_verification = previous_repeat
    receipt = {
        "status": "PASS", "freeze_signature": frozen["signature"],
        "round1_freeze_signature": r1["signature"], "results_path": str(results_path),
        "results_sha256": result_hash, "vector_manifest_sha256": manifest_hash,
        "fit_audit_manifest_sha256": audit_manifest_hash,
        "model_fit_cache_checks": validated_cache_files,
        "job_metric_checks": len(results["jobs"]),
        "aggregate_checks": sum(len(v) for v in results["aggregate"].values()),
        "paired_comparison_checks": len(results["comparisons"]),
        "exact_round1_control_checks": exact_r1_controls,
        "unique_validation_primary_rows": int(len(primary_rows)),
        "unique_validation_h12_rows": int(len(secondary_rows)),
        "h12_rows_reused_from_primary": overlap,
        "protected_checkpoint_files": len(frozen["protected_checkpoint_files"]),
        "repeat_verification": repeat_verification,
        "independent_recomputation": "MAE and signed error via math.fsum; RMSE via scikit-learn; arrays and subgroup partitions reconciled from ignored vectors.",
        "test_features_read": False, "test_labels_read": False, "models_refit_by_verifier": False,
    }
    if write_receipt:
        write_json(verification_path, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    final_path = REPORT/"results.json"
    default_path = final_path if final_path.exists() else REPORT/"results_base.json"
    parser.add_argument("results", nargs="?", default=str(default_path))
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--repeat-report", type=Path)
    args = parser.parse_args()
    report = verify_results(args.results, data_dir=args.data_dir, repeat_report=args.repeat_report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
