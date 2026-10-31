"""Independent, fit-free verification for the Optimization V2 full-six handoff."""
from __future__ import annotations

import hashlib
import json
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
from src.models.regression_research_text import HYBRID_FEATURES
from src.models.round1_models import _fit_fingerprint
from src.utils.paths import task_files

EXP = ROOT / "experiments" / "optimization_v2" / "regression_repr"
REPORTS = ROOT / "reports" / "optimization_v2" / "regression_repr"
RUNS = ROOT / "experiments" / "runs" / "optimization_v2_regression_repr"
FREEZE = EXP / "fullsix_freeze.json"
START = EXP / "fullsix_started.json"
REPORT = REPORTS / "fullsix.json"
VERIFICATION = REPORTS / "fullsix_verification.json"
NEW_CACHE = RUNS / "fullsix_predictions.npz"
BASELINE = ROOT / "experiments" / "runs" / "dual_reg_v1" / "stage1_paired_blends.npz"
PILOT_CACHE = RUNS / "pilot_predictions.npz"
PILOT_FREEZE = EXP / "freeze.json"
PILOT_REPORT = REPORTS / "pilot.json"
PILOT_VERIFICATION = REPORTS / "verification.json"
BASELINE_REPORT = ROOT / "reports" / "dual_campaign" / "regression" / "stage1_paired_blend.json"
RECIPES = ("sgd_l1", "lgbm_sparse_l1")
WINDOWS = {"W1": (55, 56, 59), "W2": (59, 60, 63), "W3": (63, 64, 67),
           "W4": (67, 68, 71), "W5": (71, 72, 75), "W6": (75, 76, 79)}
HORIZONS = {"H12_E": (55, 56, 67), "H12_L": (67, 68, 79)}
NEW_CUTOFFS = (59, 63, 71, 75)


def _hash_json(value: Any) -> str:
    content = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                         separators=(",", ":")).encode("utf-8", errors="surrogatepass")
    return hashlib.sha256(content).hexdigest()


def _array_hash(value: Any, dtype: str = "<i8") -> str:
    return hashlib.sha256(np.asarray(value, dtype=dtype).tobytes(order="C")).hexdigest()


def _write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def _metric(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    error = np.asarray(pred, dtype=float) - np.asarray(y, dtype=float)
    return {
        "n": int(error.size),
        "mae": float(np.abs(error).sum() / error.size) if error.size else None,
        "bias_prediction_minus_target": float(error.sum() / error.size) if error.size else None,
        "over_rate": float(np.count_nonzero(error > 1e-10) / error.size) if error.size else None,
        "under_rate": float(np.count_nonzero(error < -1e-10) / error.size) if error.size else None,
        "absolute_error_p95": float(np.quantile(np.abs(error), .95)) if error.size else None,
    }


def _average_percentile(raw: np.ndarray, cycles: np.ndarray) -> np.ndarray:
    # Independent route through scipy rankdata, retaining average ranks for ties.
    answer = np.zeros(raw.size, dtype=float)
    for cycle in pd.unique(cycles):
        take = np.flatnonzero(cycles == cycle)
        answer[take] = 100.0 * rankdata(raw[take], method="average") / len(take)
    return answer


def _groups(frame: pd.DataFrame, pred: np.ndarray) -> dict[str, Any]:
    y = pd.to_numeric(frame["threat_rank"], errors="raise").to_numpy(dtype=float)
    streams = frame["signal_stream"]
    absent = streams.isna().to_numpy() | streams.map(lambda v: isinstance(v, str) and not v.strip()).to_numpy(bool)
    friction = pd.to_numeric(frame["entry_friction"], errors="coerce").to_numpy(dtype=float)
    masks = {
        "tail_le5": y <= 5,
        "tail_ge95": y >= 95,
        "EF0": friction == 0,
        "EF1": friction == 1,
        "stream_missing": absent,
        "stream_present": ~absent,
    }
    return {name: _metric(y[mask], pred[mask]) for name, mask in masks.items()}


def _same(a: float | None, b: float | None, tol: float = 2e-9) -> bool:
    if a is None or b is None:
        return a is b
    return abs(float(a) - float(b)) <= tol


def verify_fullsix() -> None:
    for path in (FREEZE, START, REPORT, NEW_CACHE, PILOT_FREEZE, PILOT_REPORT,
                 PILOT_VERIFICATION, PILOT_CACHE, BASELINE, BASELINE_REPORT):
        if not path.is_file():
            raise FileNotFoundError(path)
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    started = json.loads(START.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    pilot = json.loads(PILOT_REPORT.read_text(encoding="utf-8"))
    pilot_ver = json.loads(PILOT_VERIFICATION.read_text(encoding="utf-8"))
    baseline_source = json.loads(BASELINE_REPORT.read_text(encoding="utf-8"))
    if pilot_ver["status"] != "PASS":
        raise AssertionError("Source pilot verification is no longer passing")
    if started["freeze_sha256"] != file_sha256(FREEZE) or report["continuation_freeze_sha256"] != file_sha256(FREEZE):
        raise AssertionError("Continuation freeze identity mismatch")
    if frozen["source_head"] != subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip():
        raise AssertionError("Current HEAD differs from continuation source HEAD")
    for rel, digest in frozen["source_sha256"].items():
        if file_sha256(ROOT / rel) != digest:
            raise AssertionError(f"Frozen code/protocol changed: {rel}")
    if file_sha256(NEW_CACHE) != report["new_prediction_cache_sha256"]:
        raise AssertionError("New prediction cache hash mismatch")
    for path, group, key in (
        (BASELINE_REPORT, "existing_ridge25", "report_sha256"),
        (BASELINE, "existing_ridge25", "cache_sha256"),
        (PILOT_FREEZE, "existing_pilot", "freeze_sha256"),
        (PILOT_REPORT, "existing_pilot", "report_sha256"),
        (PILOT_VERIFICATION, "existing_pilot", "verification_sha256"),
        (PILOT_CACHE, "existing_pilot", "prediction_cache_sha256"),
    ):
        if file_sha256(path) != frozen[group][key]:
            raise AssertionError(f"Frozen prior artifact changed: {path.name}")
    if report["new_fit_count"] != 8 or report["reused_fit_count"] != 4 or report["control_fit_count"] != 0:
        raise AssertionError("Full-six fit accounting differs from eight-new/four-reused/no-controls")
    if not report["all_six_primary_windows_and_both_H12_views_completed"] or report["test_access"]:
        raise AssertionError("Full-six completion/Test boundary receipt invalid")
    if report["model_combination_note"].find("no blend between candidates") < 0:
        raise AssertionError("Candidate candidates were improperly combined")
    correction = frozen["existing_ridge25"]["metric_provenance_correction"]
    if "ridge_25" not in baseline_source.get("results", {}):
        raise AssertionError("Authoritative results.ridge_25 section missing from baseline report")
    if correction["authoritative_report_key"] != "results.ridge_25":
        raise AssertionError("Continuation provenance correction points to the wrong report section")
    if report["baseline_metric_provenance_correction"] != correction:
        raise AssertionError("Continuation report did not carry the Ridge25 provenance correction")

    train_path = task_files("regression")["train"]
    if train_path is None or file_sha256(train_path) != frozen["official_train"]["sha256"]:
        raise AssertionError("Official Train changed after continuation freeze")
    frame = pd.read_csv(train_path, low_memory=False)
    cycle = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=int)
    y = pd.to_numeric(frame["threat_rank"], errors="raise").to_numpy(dtype=float)
    new_data = np.load(NEW_CACHE, allow_pickle=False)
    pilot_data = np.load(PILOT_CACHE, allow_pickle=False)
    baseline_data = np.load(BASELINE, allow_pickle=False)
    expected_new_keys = {f"{prefix}_{recipe}_{cutoff}"
                         for prefix in ("rows", "raw") for recipe in RECIPES for cutoff in NEW_CUTOFFS}
    if set(new_data.files) != expected_new_keys:
        raise AssertionError("Continuation cache is not exactly the eight new fits")

    verified_fits: dict[str, Any] = {}
    for cutoff in NEW_CUTOFFS:
        fit_positions = np.flatnonzero(cycle <= cutoff).astype(np.int64)
        eval_positions = np.flatnonzero((cycle >= cutoff + 1) & (cycle <= cutoff + 4)).astype(np.int64)
        if len(eval_positions) == 0 or np.any(cycle[eval_positions] <= cutoff):
            raise AssertionError(f"Evaluation boundary error at cutoff {cutoff}")
        for recipe in RECIPES:
            key = f"{recipe}_{cutoff}"
            receipt = report["fits"][key]
            saved_rows = np.asarray(new_data[f"rows_{key}"], dtype=np.int64)
            saved_pred = np.asarray(new_data[f"raw_{key}"], dtype=np.float64)
            if not np.array_equal(saved_rows, eval_positions) or saved_pred.shape != saved_rows.shape:
                raise AssertionError(f"Prediction rows/shape mismatch at {key}")
            if not np.isfinite(saved_pred).all() or _array_hash(saved_rows) != receipt["validation_positions_sha256"]:
                raise AssertionError(f"Prediction row digest/finite check failed at {key}")
            if _array_hash(saved_pred, "<f8") != receipt["raw_prediction_sha256"]:
                raise AssertionError(f"Prediction content hash failed at {key}")
            if len(fit_positions) != receipt["fit_rows"] or _array_hash(fit_positions) != receipt["fit_positions_sha256"]:
                raise AssertionError(f"Fit-row identity mismatch at {key}")
            model_state = receipt["model_state"]
            if model_state["fit_rows"] != len(fit_positions) or model_state["fit_data_sha256"] != _fit_fingerprint(
                frame.iloc[fit_positions].loc[:, list(HYBRID_FEATURES)], HYBRID_FEATURES, y[fit_positions]
            ):
                raise AssertionError(f"Fit feature/target fingerprint does not recompute at {key}")
            target_hash = hashlib.sha256(np.asarray(y[fit_positions], dtype="<f8").tobytes(order="C")).hexdigest()
            if target_hash != model_state["target_sha256"]:
                raise AssertionError(f"Fit target digest differs at {key}")
            if receipt["fit_cycle_max"] != cutoff or receipt["validation_cycles"] != [cutoff + 1, cutoff + 4]:
                raise AssertionError(f"Declared temporal cutoff differs at {key}")
            if model_state["validation_data_passed_to_fit"] or model_state["early_stopping_used"]:
                raise AssertionError(f"Validation/early stopping path used at {key}")
            if not model_state["sparse_design"] or model_state["svd_used"] or model_state["native_categorical_features_used"]:
                raise AssertionError(f"Representation differs from preregistered sparse design at {key}")
            state_without_hash = dict(model_state)
            expected_state_hash = state_without_hash.pop("state_sha256")
            if _hash_json(state_without_hash) != expected_state_hash:
                raise AssertionError(f"Fit-state checksum is not internally valid at {key}")
            if model_state["training_design_shape"][0] != len(fit_positions):
                raise AssertionError(f"Preprocessing/model fit row count is not train-only at {key}")
            if receipt["prediction_diagnostics"]["n_predictions"] != len(eval_positions):
                raise AssertionError(f"Prediction row diagnostic differs at {key}")
            if recipe == "sgd_l1":
                sgd = receipt["sgd_optimizer_audit"]
                if sgd["interpretation"] != "fixed-budget approximate L1; convergence not asserted":
                    raise AssertionError(f"SGD convergence claim risk at {key}")
                if sgd["n_iter"] != model_state["estimator"]["n_iter"]:
                    raise AssertionError(f"SGD iteration receipt mismatch at {key}")
                if sgd["max_iter_reached"] and not sgd["convergence_warnings"]:
                    raise AssertionError(f"SGD cap reached but warning was not recorded at {key}")
            if receipt["fit_seconds"] <= 0 or receipt["prediction_seconds"] < 0:
                raise AssertionError(f"Training cost receipt invalid at {key}")
            verified_fits[key] = {
                "fit_rows": int(len(fit_positions)),
                "validation_rows": int(len(eval_positions)),
                "fit_seconds": receipt["fit_seconds"],
                "fit_positions_sha256": _array_hash(fit_positions),
                "validation_positions_sha256": _array_hash(eval_positions),
                "fit_data_sha256_recomputed": model_state["fit_data_sha256"],
                "prediction_sha256": _array_hash(saved_pred, "<f8"),
                "fit_only_and_hashes_pass": True,
            }

    def raw_at(recipe: str, cutoff: int, rows: np.ndarray) -> np.ndarray:
        data = pilot_data if cutoff in (55, 67) else new_data
        key = f"{recipe}_{cutoff}"
        source = np.asarray(data[f"rows_{key}"], dtype=np.int64)
        prediction = np.asarray(data[f"raw_{key}"], dtype=float)
        positions = {int(row): pos for pos, row in enumerate(source)}
        return prediction[np.asarray([positions[int(row)] for row in rows], dtype=np.int64)]

    def ridge_at(cutoff: int, rows: np.ndarray) -> np.ndarray:
        source = np.asarray(baseline_data[f"rows_{cutoff}"], dtype=np.int64)
        prediction = np.asarray(baseline_data[f"blend_ridge25_{cutoff}"], dtype=float)
        positions = {int(row): pos for pos, row in enumerate(source)}
        return prediction[np.asarray([positions[int(row)] for row in rows], dtype=np.int64)]

    def policies_at(recipe: str, cutoff: int, rows: np.ndarray) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        raw = raw_at(recipe, cutoff, rows)
        ridge = ridge_at(cutoff, rows)
        local_cycle = cycle[rows]
        mapped = np.clip(.5 * raw + .5 * _average_percentile(raw, local_cycle), 0, 100)
        outputs = {
            "raw": raw,
            "clipped": np.clip(raw, 0, 100),
            "rank50_diagnostic": mapped,
            "blend25": np.clip(.25 * mapped + .75 * ridge, 0, 100),
        }
        f = frame.iloc[rows]
        summaries = {name: {**_metric(y[rows], value), "groups": _groups(f, value)}
                     for name, value in outputs.items()}
        summaries["ridge25"] = _metric(y[rows], ridge)
        return summaries, outputs

    recalc_views: dict[str, Any] = {}
    view_vectors: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for name, (cutoff, lo, hi) in WINDOWS.items():
        rows = np.flatnonzero((cycle >= lo) & (cycle <= hi)).astype(np.int64)
        expected_positions = report["views"][name]["positions_sha256"]
        if _array_hash(rows) != expected_positions:
            raise AssertionError(f"Temporal view source rows changed at {name}")
        recalc_views[name] = {"ridge25": _metric(y[rows], ridge_at(cutoff, rows))}
        view_vectors[name] = {}
        for recipe in RECIPES:
            summaries, outputs = policies_at(recipe, cutoff, rows)
            recalc_views[name][recipe] = summaries
            view_vectors[name][recipe] = outputs

    recalc_horizons: dict[str, Any] = {}
    horizon_vectors: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for name, (cutoff, lo, hi) in HORIZONS.items():
        rows = np.flatnonzero((cycle >= lo) & (cycle <= hi)).astype(np.int64)
        if _array_hash(rows) != report["long_horizons"][name]["positions_sha256"]:
            raise AssertionError(f"Temporal long-horizon row identity changed at {name}")
        recalc_horizons[name] = {"ridge25": _metric(y[rows], ridge_at(cutoff, rows))}
        horizon_vectors[name] = {}
        for recipe in RECIPES:
            summaries, outputs = policies_at(recipe, cutoff, rows)
            recalc_horizons[name][recipe] = summaries
            horizon_vectors[name][recipe] = outputs

    for name, actual in recalc_views.items():
        for recipe in RECIPES:
            for policy, metrics in actual[recipe].items():
                saved = report["views"][name][recipe][policy]
                if metrics["n"] != saved["n"] or not _same(metrics["mae"], saved["mae"]):
                    raise AssertionError(f"Window metric mismatch at {name}/{recipe}/{policy}")
                for group_name, group_metric in metrics["groups"].items():
                    saved_group = saved["groups"][group_name]
                    if group_metric["n"] != saved_group["n"] or not _same(group_metric["mae"], saved_group["mae"]):
                        raise AssertionError(f"Window subgroup mismatch at {name}/{recipe}/{policy}/{group_name}")
    for name, actual in recalc_horizons.items():
        for recipe in RECIPES:
            for policy, metrics in actual[recipe].items():
                saved = report["long_horizons"][name][recipe][policy]
                if metrics["n"] != saved["n"] or not _same(metrics["mae"], saved["mae"]):
                    raise AssertionError(f"H12 metric mismatch at {name}/{recipe}/{policy}")
                for group_name, group_metric in metrics["groups"].items():
                    saved_group = saved["groups"][group_name]
                    if group_metric["n"] != saved_group["n"] or not _same(group_metric["mae"], saved_group["mae"]):
                        raise AssertionError(f"H12 subgroup mismatch at {name}/{recipe}/{policy}/{group_name}")

    recalc_gates: dict[str, Any] = {}
    for recipe in RECIPES:
        primary_rows = np.concatenate([np.flatnonzero((cycle >= lo) & (cycle <= hi))
                                       for _, (_, lo, hi) in WINDOWS.items()]).astype(np.int64)
        primary_pred = np.concatenate([view_vectors[v][recipe]["blend25"] for v in WINDOWS])
        primary_base = np.concatenate([ridge_at(WINDOWS[v][0], np.flatnonzero(
            (cycle >= WINDOWS[v][1]) & (cycle <= WINDOWS[v][2]))) for v in WINDOWS])
        primary_gain = _metric(y[primary_rows], primary_base)["mae"] - _metric(y[primary_rows], primary_pred)["mae"]
        deltas = {v: recalc_views[v][recipe]["blend25"]["mae"] - recalc_views[v]["ridge25"]["mae"] for v in WINDOWS}
        wins = sum(delta <= 0 for delta in deltas.values())
        e_delta = recalc_horizons["H12_E"][recipe]["blend25"]["mae"] - recalc_horizons["H12_E"]["ridge25"]["mae"]
        l_delta = recalc_horizons["H12_L"][recipe]["blend25"]["mae"] - recalc_horizons["H12_L"]["ridge25"]["mae"]
        groups: dict[str, Any] = {}
        maximum = -np.inf
        primary_group_cand = _groups(frame.iloc[primary_rows], primary_pred)
        primary_group_base = _groups(frame.iloc[primary_rows], primary_base)
        for segment, cand_groups, base_groups in (
            ("primary_pooled", primary_group_cand, primary_group_base),
            ("H12_E", recalc_horizons["H12_E"][recipe]["blend25"]["groups"],
             _groups(frame.iloc[np.flatnonzero((cycle >= 56) & (cycle <= 67))], ridge_at(55, np.flatnonzero((cycle >= 56) & (cycle <= 67))))),
            ("H12_L", recalc_horizons["H12_L"][recipe]["blend25"]["groups"],
             _groups(frame.iloc[np.flatnonzero((cycle >= 68) & (cycle <= 79))], ridge_at(67, np.flatnonzero((cycle >= 68) & (cycle <= 79))))),
        ):
            groups[segment] = {}
            for subgroup in ("tail_le5", "tail_ge95", "stream_missing"):
                delta = cand_groups[subgroup]["mae"] - base_groups[subgroup]["mae"]
                groups[segment][subgroup] = float(delta)
                maximum = max(maximum, float(delta))
        checks = {
            "primary_gain_at_least_0_03": primary_gain >= .03,
            "H12_E_and_H12_L_harm_at_most_0_05": max(e_delta, l_delta) <= .05,
            "at_least_4_of_6_nonworse_windows": wins >= 4,
            "all_reported_tail_and_missing_harm_at_most_0_25": maximum <= .25,
        }
        recalc_gates[recipe] = {
            "primary_gain_mae": float(primary_gain),
            "window_delta_mae_candidate_minus_ridge25": {k: float(v) for k, v in deltas.items()},
            "nonworse_windows_count": int(wins),
            "H12_E_delta_mae": float(e_delta),
            "H12_L_delta_mae": float(l_delta),
            "tail_and_missing_delta_mae": groups,
            "maximum_tail_or_missing_harm_mae": float(maximum),
            "gates": checks,
            "all_gates_pass": all(checks.values()),
        }
        saved_gate = report["candidate_gates"][recipe]
        if saved_gate["all_gates_pass"] != recalc_gates[recipe]["all_gates_pass"]:
            raise AssertionError(f"Candidate acceptance gate mismatch for {recipe}")
        if not _same(saved_gate["primary_gain_mae"], primary_gain):
            raise AssertionError(f"Pooled primary gain mismatch for {recipe}")
        for key in ("H12_E_delta_mae", "H12_L_delta_mae", "maximum_tail_or_missing_harm_mae"):
            if not _same(saved_gate[key], recalc_gates[recipe][key]):
                raise AssertionError(f"Candidate subgroup/H12 gate mismatch for {recipe}/{key}")

    complementarity: dict[str, Any] = {}
    for segment, rows in (
        ("primary_pooled", np.concatenate([np.flatnonzero((cycle >= lo) & (cycle <= hi))
                                            for _, (_, lo, hi) in WINDOWS.items()])),
        ("H12_E", np.flatnonzero((cycle >= 56) & (cycle <= 67))),
        ("H12_L", np.flatnonzero((cycle >= 68) & (cycle <= 79)),),
    ):
        if segment == "primary_pooled":
            pred_sgd = np.concatenate([view_vectors[v]["sgd_l1"]["blend25"] for v in WINDOWS])
            pred_lgb = np.concatenate([view_vectors[v]["lgbm_sparse_l1"]["blend25"] for v in WINDOWS])
        else:
            horizon = segment
            pred_sgd = horizon_vectors[horizon]["sgd_l1"]["blend25"]
            pred_lgb = horizon_vectors[horizon]["lgbm_sparse_l1"]["blend25"]
        es, el = pred_sgd - y[rows], pred_lgb - y[rows]
        aes, ael = np.abs(es), np.abs(el)
        comp = {
            "n": int(len(rows)),
            "prediction_correlation": float(np.corrcoef(pred_sgd, pred_lgb)[0, 1]),
            "signed_error_correlation": float(np.corrcoef(es, el)[0, 1]),
            "absolute_error_correlation": float(np.corrcoef(aes, ael)[0, 1]),
            "SGD_lower_absolute_error_rate": float(np.mean(aes < ael)),
            "LightGBM_lower_absolute_error_rate": float(np.mean(ael < aes)),
            "tie_rate": float(np.mean(aes == ael)),
            "MAE_SGD": float(np.mean(aes)),
            "MAE_LightGBM": float(np.mean(ael)),
            "candidate_blend_created": False,
        }
        complementarity[segment] = comp
        saved_comp = report["candidate_complementarity"][segment]
        for key in ("n", "candidate_blend_created"):
            if saved_comp[key] != comp[key]:
                raise AssertionError(f"Cross-candidate comparison scope mismatch at {segment}/{key}")
        for key in ("prediction_correlation", "signed_error_correlation", "absolute_error_correlation",
                    "MAE_SGD", "MAE_LightGBM"):
            if not _same(saved_comp[key], comp[key], tol=1e-8):
                raise AssertionError(f"Cross-candidate diagnostic mismatch at {segment}/{key}")

    fit_total = sum(item["fit_seconds"] for item in report["fits"].values())
    if not _same(fit_total, report["fit_total_seconds"]):
        raise AssertionError("Training cost does not sum to fit receipts")
    verification = {
        "experiment_id": "optimization_v2_sparse_hybrid_fullsix_continuation",
        "status": "PASS",
        "verified_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "freeze_sha256": file_sha256(FREEZE),
        "report_sha256": file_sha256(REPORT),
        "prediction_cache_sha256": file_sha256(NEW_CACHE),
        "official_train_sha256": file_sha256(train_path),
        "independent_fit_receipts": verified_fits,
        "recomputed_views": recalc_views,
        "recomputed_horizons": recalc_horizons,
        "recomputed_gates": recalc_gates,
        "recomputed_complementarity": complementarity,
        "baseline_metric_provenance_correction": correction,
        "checks": {
            "eight_new_fits_only": True,
            "four_earlier_fits_reused_row_exactly": True,
            "Ridge25_control_reused_without_fits": True,
            "train_only_preprocessing_and_target_fingerprints": True,
            "validation_and_test_excluded_from_fit": True,
            "all_six_windows_and_both_H12_views_recomputed": True,
            "raw_clipped_rank50_and_fixed_blend25_recomputed": True,
            "tails_EF_stream_bias_cycle_diagnostics_recomputed": True,
            "fixed_SGD_iteration_budget_and_warning_risk_reported": True,
            "no_candidate_to_candidate_blend": True,
            "source_and_input_hashes_match_prefit_continuation_freeze": True,
        },
        "recommendation": "Main may select at most one external candidate based on these frozen full-six results. No additional fitting or blending was performed.",
    }
    _write_new(VERIFICATION, verification)
    new_data.close()
    pilot_data.close()
    baseline_data.close()
    print(json.dumps({"verification": str(VERIFICATION), "status": "PASS", "gates": recalc_gates,
                      "fit_total_minutes": fit_total / 60.0}, indent=2))


if __name__ == "__main__":
    verify_fullsix()
