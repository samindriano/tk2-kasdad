"""Independent verifier for the frozen, cache-only R19 calibration pilot."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.models.optimization_v3_batch1 import json_sha256


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SHA256 = "756523ff8156a9cd776e52249f7e2a8f23dcec432b8d10570e55ff6db0e0f082"
BASE = ROOT / "experiments/runs/dual_reg_v1/stage1_paired_blends.npz"
PILOT = ROOT / "experiments/runs/optimization_v2_regression_repr/pilot_predictions.npz"
FULL = ROOT / "experiments/runs/optimization_v2_regression_repr/fullsix_predictions.npz"
TRAIN = ROOT / "data/raw/regression/Train.csv"
FREEZE = ROOT / "experiments/optimization_v3/batch1/R19/revision_02/freeze.json"
RECIPE = ROOT / "experiments/optimization_v3/batch1/R19/revision_02/recipe.json"
RESULTS = ROOT / "reports/optimization_v3/batch1/R19/results.json"
EXPECTED_BASELINE = {"primary": 10.477525111059206, "H12_L": 11.611685946326363}
LONG_CUTOFFS = {55, 67}
WINDOWS = {
    "67": [(59, 60, 63), (63, 64, 67)],
    "75": [(67, 68, 71), (71, 72, 75)],
    "W4": [(67, 68, 71)],
    "W6": [(75, 76, 79)],
    "primary": [(67, 68, 71), (75, 76, 79)],
    "H12_L": [(67, 68, 79)],
    "H12_E": [(55, 56, 67)],
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"Expected JSON object: {path}")
    return value


def _assert_frozen() -> dict[str, Any]:
    frozen = _load_json(FREEZE)
    recipe = _load_json(RECIPE)
    if frozen.get("status") != "FROZEN_BEFORE_R19_CALIBRATION_AND_SCORING":
        raise AssertionError("R19 was not frozen before scoring")
    if any(frozen.get(name) is not False for name in (
        "target_read", "prediction_arrays_read", "candidate_metric_computed", "test_read"
    )):
        raise AssertionError("R19 freeze claims an outcome read before freeze")
    identity = frozen.get("identity")
    if not isinstance(identity, dict) or json_sha256(identity) != frozen.get("signature"):
        raise AssertionError("R19 frozen identity signature is invalid")
    if identity.get("train_sha256") != TRAIN_SHA256 or _sha256(TRAIN) != TRAIN_SHA256:
        raise AssertionError("Official Regression Train hash differs from the frozen R19 source")
    if identity.get("source_commit") != frozen.get("identity", {}).get("source_commit"):
        raise AssertionError("R19 source commit is absent")
    for relative, digest in identity.get("input_sha256", {}).items():
        path = ROOT / relative
        if not path.is_file() or _sha256(path) != digest:
            raise AssertionError(f"R19 frozen input changed: {relative}")
    for relative, digest in identity.get("source_sha256", {}).items():
        path = ROOT / relative
        if not path.is_file() or _sha256(path) != digest:
            raise AssertionError(f"R19 frozen source changed: {relative}")
    if "experiments/runs/optimization_v3_reg_diagnostics/oof_residuals.npz" in identity.get("input_sha256", {}):
        raise AssertionError("Unpinned derived OOF cache entered the frozen R19 inputs")
    if recipe != frozen.get("identity", {}).get("recipe"):
        raise AssertionError("R19 recipe differs from the frozen identity")
    if recipe.get("revision") != 2:
        raise AssertionError("R19 result does not use the row-matched revision 2 recipe")
    provenance = recipe.get("baseline_metric_provenance", {})
    if provenance.get("current_champion") != "v2_reg_sparse_lgb25":
        raise AssertionError("R19 revision 2 does not identify the correct current champion")
    for name, expected in EXPECTED_BASELINE.items():
        if not np.isclose(recipe.get("control_primary_mae_rounded" if name == "primary" else "control_H12_L_mae_rounded"), expected, rtol=0.0, atol=1e-12):
            raise AssertionError(f"R19 frozen {name} comparator differs from its late row-matched metric")
    if recipe.get("test_or_sample_access") is not False or recipe.get("kaggle_access") is not False:
        raise AssertionError("R19 recipe permits Test or Kaggle access")
    return frozen


def _predict_window(frame: pd.DataFrame, caches: tuple[Any, Any, Any],
                    cutoff: int, low: int, high: int) -> dict[str, np.ndarray]:
    base, pilot, full = caches
    sparse = pilot if cutoff in LONG_CUTOFFS else full
    rows = np.asarray(base[f"rows_{cutoff}"], dtype=np.int64)
    sparse_rows = np.asarray(sparse[f"rows_lgbm_sparse_l1_{cutoff}"], dtype=np.int64)
    if not np.array_equal(rows, sparse_rows):
        raise AssertionError(f"R19 cache row mismatch at cutoff {cutoff}")
    selected = (frame.iloc[rows]["cycle_code"].to_numpy() >= low) & (
        frame.iloc[rows]["cycle_code"].to_numpy() <= high
    )
    rows = rows[selected]
    cycles = frame.iloc[rows]["cycle_code"].to_numpy(dtype=np.int64)
    raw = np.asarray(sparse[f"raw_lgbm_sparse_l1_{cutoff}"], dtype=np.float64)[selected]
    lgb = np.asarray(base[f"lgb_{cutoff}"], dtype=np.float64)[selected]
    ridge = np.asarray(base[f"ridge_{cutoff}"], dtype=np.float64)[selected]
    ridge25 = 0.75 * lgb + 0.25 * ridge
    rank50 = np.empty(len(rows), dtype=np.float64)
    for cycle in np.unique(cycles):
        mask = cycles == cycle
        rank50[mask] = 0.5 * raw[mask] + 0.5 * 100.0 * rankdata(
            raw[mask], method="average"
        ) / int(mask.sum())
    prediction = np.clip(0.75 * ridge25 + 0.25 * np.clip(rank50, 0.0, 100.0), 0.0, 100.0)
    order = np.argsort(rows, kind="stable")
    return {"rows": rows[order], "cycle": cycles[order], "prediction": prediction[order]}


def _combine(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    result = {key: np.concatenate([part[key] for part in parts])
              for key in ("rows", "cycle", "prediction")}
    order = np.argsort(result["rows"], kind="stable")
    result = {key: value[order] for key, value in result.items()}
    if len(result["rows"]) != len(np.unique(result["rows"])):
        raise AssertionError("R19 verifier found overlapping row windows")
    return result


def _mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(y.astype(np.float64) - p.astype(np.float64))))


def _verify_results() -> dict[str, Any]:
    frozen = _assert_frozen()
    report = _load_json(RESULTS)
    if report.get("freeze_signature") != frozen["signature"]:
        raise AssertionError("R19 results do not refer to the current frozen identity")
    if report.get("derived_unpinned_oof_cache_used") is not False:
        raise AssertionError("R19 result used or failed to exclude the unpinned derived OOF cache")
    if report.get("test_data_used") is not False or report.get("sample_submission_used") is not False:
        raise AssertionError("R19 result reports forbidden Test/Sample access")
    if report.get("kaggle_access") is not False or report.get("fit_accounting") != {
        "predictor_fits": 0, "failed_fits": 0, "cache_only": True
    }:
        raise AssertionError("R19 was not a cache-only, zero-fit evaluation")

    frame = pd.read_csv(TRAIN, usecols=["cycle_code", "entry_friction", "signal_stream", "threat_rank"])
    target = pd.to_numeric(frame["threat_rank"], errors="raise").to_numpy(dtype=np.float64)
    caches = (np.load(BASE, allow_pickle=False), np.load(PILOT, allow_pickle=False),
              np.load(FULL, allow_pickle=False))
    try:
        baseline = {
            name: _combine([_predict_window(frame, caches, *window) for window in windows])
            for name, windows in WINDOWS.items()
        }
        for name in ("primary", "H12_L"):
            actual = _mae(target[baseline[name]["rows"]], baseline[name]["prediction"])
            expected = EXPECTED_BASELINE[name]
            if not np.isclose(actual, expected, rtol=0.0, atol=1e-6):
                raise AssertionError(f"R19 independent baseline {name}={actual:.9f}; expected {expected:.6f}")
            if not np.isclose(report["views"][name]["baseline_mae"], actual, rtol=0.0, atol=1e-10):
                raise AssertionError(f"R19 reported baseline {name} does not reproduce")

        offsets: dict[int, dict[int, float]] = {}
        supports: dict[str, Any] = {}
        frozen_supports = report.get("calibration_support_and_offsets", {})
        for cutoff in (67, 75):
            data = baseline[str(cutoff)]
            ef = pd.to_numeric(frame.iloc[data["rows"]]["entry_friction"], errors="coerce").to_numpy()
            residual = target[data["rows"]] - data["prediction"]
            offsets[cutoff] = {}
            supports[str(cutoff)] = {}
            for group in (0, 1):
                mask = ef == group
                n = int(mask.sum())
                median = float(np.median(residual[mask])) if n else None
                supported = n >= 200
                raw = 0.5 * median if supported else 0.0
                value = float(np.clip(raw, -2.0, 2.0))
                offsets[cutoff][group] = value
                supports[str(cutoff)][str(group)] = {
                    "rows": n, "supported": supported,
                    "median_y_minus_prediction": median,
                    "raw_half_median_offset": raw, "offset": value,
                }
            unknown_n = int(np.sum(~np.isin(ef, [0, 1])))
            supports[str(cutoff)]["unknown"] = {
                "rows": unknown_n, "supported": False,
                "median_y_minus_prediction": None,
                "raw_half_median_offset": 0.0, "offset": 0.0,
            }
            for group in (0, 1, "unknown"):
                actual = frozen_supports.get(str(cutoff), {}).get(str(group))
                expected_support = supports[str(cutoff)][str(group)]
                if actual is None or actual.get("rows") != expected_support["rows"] or actual.get("supported") != expected_support["supported"]:
                    raise AssertionError(f"R19 calibration support for cutoff {cutoff}, group {group} does not reproduce")
                for key in ("median_y_minus_prediction", "raw_half_median_offset", "offset"):
                    actual_value = actual.get(key)
                    expected_value = expected_support[key]
                    if expected_value is None:
                        if actual_value is not None:
                            raise AssertionError(f"R19 calibration {key} for unknown EF should be null")
                    elif not np.isclose(actual_value, expected_value, rtol=0.0, atol=1e-12):
                        raise AssertionError(f"R19 calibration {key} for cutoff {cutoff}, group {group} does not reproduce")
            if unknown_n:
                raise AssertionError("R19 verifier found unknown entry_friction values absent from frozen Train profile")
        candidates: dict[str, np.ndarray] = {}
        for name in ("W4", "W6", "primary", "H12_L", "H12_E"):
            data = baseline[name]
            pred = data["prediction"].copy()
            ef_all = pd.to_numeric(frame.iloc[data["rows"]]["entry_friction"], errors="coerce").to_numpy()
            if name != "H12_E":
                target_cutoffs = (67, 75) if name == "primary" else (67,) if name in ("W4", "H12_L") else (75,)
                for cutoff in target_cutoffs:
                    lower, upper = ((68, 71) if cutoff == 67 and name in ("W4", "primary")
                                    else (76, 79) if cutoff == 75 and name in ("W6", "primary")
                                    else (68, 79))
                    mask = (data["cycle"] >= lower) & (data["cycle"] <= upper)
                    for group in (0, 1):
                        group_mask = mask & (ef_all == group)
                        pred[group_mask] = np.clip(
                            pred[group_mask] + offsets[cutoff][group], 0.0, 100.0
                        )
            metric = _mae(target[data["rows"]], pred)
            if not np.isclose(report["views"][name]["candidate_mae"], metric, rtol=0.0, atol=1e-10):
                raise AssertionError(f"R19 candidate metric {name} does not independently reproduce")
            baseline_metric = _mae(target[data["rows"]], data["prediction"])
            view = report["views"][name]
            if (view["rows"] != len(data["rows"])
                    or not np.isclose(view["baseline_mae"], baseline_metric, rtol=0.0, atol=1e-10)
                    or not np.isclose(view["gain_positive_is_better"], baseline_metric - metric, rtol=0.0, atol=1e-10)
                    or not np.isclose(view["harm_candidate_minus_baseline"], metric - baseline_metric, rtol=0.0, atol=1e-10)):
                raise AssertionError(f"R19 view accounting for {name} does not reproduce")
            candidates[name] = pred

        group_harms: list[float] = []
        for view_name in ("primary", "H12_L"):
            data = baseline[view_name]
            pred = candidates[view_name]
            ef = pd.to_numeric(frame.iloc[data["rows"]]["entry_friction"], errors="coerce").to_numpy()
            stream = frame.iloc[data["rows"]]["signal_stream"]
            missing_stream = stream.isna().to_numpy(dtype=bool)
            present_indices = np.flatnonzero(~missing_stream)
            if len(present_indices):
                missing_stream[present_indices] = stream.iloc[present_indices].map(
                    lambda value: not str(value).strip()
                ).to_numpy(dtype=bool)
            masks = {
                "EF0": ef == 0, "EF1": ef == 1, "EF_unknown": ~np.isin(ef, [0, 1]),
                "tail_low_le5": target[data["rows"]] <= 5.0,
                "tail_high_ge95": target[data["rows"]] >= 95.0,
                "stream_missing": missing_stream,
            }
            actual_groups = report.get("supported_group_diagnostics", {}).get(view_name, {})
            for group_name, mask in masks.items():
                n = int(mask.sum())
                supported = n >= 100
                base_group_mae = _mae(target[data["rows"]][mask], data["prediction"][mask]) if n else None
                cand_group_mae = _mae(target[data["rows"]][mask], pred[mask]) if n else None
                harm = cand_group_mae - base_group_mae if n else None
                actual = actual_groups.get(group_name)
                if actual is None or actual.get("rows") != n or actual.get("supported") != supported:
                    raise AssertionError(f"R19 {view_name} subgroup support {group_name} does not reproduce")
                for key, value in (("baseline_mae", base_group_mae), ("candidate_mae", cand_group_mae),
                                   ("harm_candidate_minus_baseline", harm)):
                    got = actual.get(key)
                    if value is None:
                        if got is not None:
                            raise AssertionError(f"R19 {view_name}/{group_name} {key} should be null")
                    elif not np.isclose(got, value, rtol=0.0, atol=1e-10):
                        raise AssertionError(f"R19 {view_name}/{group_name} {key} does not reproduce")
                if group_name in {"tail_low_le5", "tail_high_ge95", "stream_missing"} and supported:
                    group_harms.append(float(harm))
        maximum_guarded_harm = max(group_harms) if group_harms else None
        primary_gain = report["views"]["primary"]["gain_positive_is_better"]
        h12_harm = report["views"]["H12_L"]["harm_candidate_minus_baseline"]
        expected_checks = {
            "late_primary_gain_at_least_0_03": primary_gain >= 0.03,
            "H12_L_harm_at_most_0_5": h12_harm <= 0.5,
            "W4_and_W6_harm_at_most_0_5": all(
                report["views"][name]["harm_candidate_minus_baseline"] <= 0.5
                for name in ("W4", "W6")
            ),
            "supported_tail_or_missing_harm_at_most_0_25": (
                maximum_guarded_harm is not None and maximum_guarded_harm <= 0.25
            ),
        }
        if report.get("gate_checks") != expected_checks:
            raise AssertionError("R19 frozen gate checks do not follow reproduced metrics")
        if report["decision"] != ("ADVANCE" if all(report["gate_checks"].values()) else "RETIRE"):
            raise AssertionError("R19 decision does not follow its frozen gate checks")
    finally:
        for cache in caches:
            cache.close()
    return {"card_id": "R19", "status": "INDEPENDENT_QA_PASS",
            "decision": report["decision"], "freeze_signature": frozen["signature"],
            "baseline_metrics_reproduced": ["primary", "H12_L"],
            "candidate_metrics_reproduced": ["W4", "W6", "primary", "H12_L", "H12_E"],
            "test_or_sample_access": False, "new_model_fits": 0}


def main() -> int:
    result = _verify_results()
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
