"""Freeze and evaluate R19 from pinned source caches, without new model fits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.optimization_v3_batch1 import BatchIntegrityError, file_sha256, json_sha256
from src.models.optimization_v3_batch1_r19 import (
    apply_ef_median_offsets,
    fit_ef_median_offsets,
    parse_entry_friction,
)
from src.utils.paths import task_files


ROOT = Path(__file__).resolve().parents[1]
CARD_ID = "R19"
TRAIN_SHA256 = "756523ff8156a9cd776e52249f7e2a8f23dcec432b8d10570e55ff6db0e0f082"
CUT_OFFS = (55, 59, 63, 67, 71, 75)
LONG_CUTOFFS = frozenset({55, 67})
EXPECTED_SCORE_ROWS = {55: 17685, 59: 6493, 63: 7078, 67: 20903, 71: 7462, 75: 8491}
EXPECTED_BASELINE_MAE = {"primary": 11.559390, "H12_L": 14.047533}
QUEUE_PATH = ROOT / "experiments/optimization_v3/research_backlog/queue.json"
CARD_PATH = ROOT / "reports/optimization_v3/research_backlog/cards/R19.md"
CONTROL_PATH = ROOT / "reports/optimization_v2/kaggle/v2_reg_sparse_lgb25.json"
DIAGNOSTICS_PROTOCOL_PATH = ROOT / "experiments/optimization_v3/regression_diagnostics/protocol.json"
TRAIN_FEATURE_COLUMNS = ("record_id", "cycle_code", "entry_friction", "signal_stream")
BASE_CACHE = ROOT / "experiments/runs/dual_reg_v1/stage1_paired_blends.npz"
PILOT_CACHE = ROOT / "experiments/runs/optimization_v2_regression_repr/pilot_predictions.npz"
FULL_CACHE = ROOT / "experiments/runs/optimization_v2_regression_repr/fullsix_predictions.npz"
SOURCE_DIAGNOSTICS = ROOT / "scripts/optimization_v3_reg_diagnostics.py"
SCRIPT_PATH = ROOT / "scripts/optimization_v3_batch1_r19.py"
VERIFY_PATH = ROOT / "scripts/optimization_v3_batch1_r19_verify.py"
HELPER_PATH = ROOT / "src/models/optimization_v3_batch1_r19.py"
GENERIC_PATH = ROOT / "src/models/optimization_v3_batch1.py"
EXPERIMENT_DIR = ROOT / "experiments/optimization_v3/batch1/R19"
REPORT_DIR = ROOT / "reports/optimization_v3/batch1/R19"
RECIPE_PATH = EXPERIMENT_DIR / "recipe.json"
FREEZE_PATH = EXPERIMENT_DIR / "freeze.json"
RESULTS_PATH = REPORT_DIR / "results.json"

CALIBRATION_WINDOWS = {
    "67": [(59, 60, 63), (63, 64, 67)],
    "75": [(67, 68, 71), (71, 72, 75)],
}
EVALUATION_WINDOWS = {
    "W4": [(67, 68, 71)],
    "W6": [(75, 76, 79)],
    "primary": [(67, 68, 71), (75, 76, 79)],
    "H12_L": [(67, 68, 79)],
    "H12_E": [(55, 56, 67)],
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BatchIntegrityError(f"Cannot read JSON input {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BatchIntegrityError(f"Expected a JSON object: {path}")
    return value


def _write_once(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise BatchIntegrityError(f"Refusing to overwrite different R19 artifact: {path}")
        return
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)


def _source_paths() -> list[Path]:
    return [SOURCE_DIAGNOSTICS, SCRIPT_PATH, VERIFY_PATH, HELPER_PATH, GENERIC_PATH]


def _row_cache_path(cutoff: int) -> Path:
    return PILOT_CACHE if cutoff in LONG_CUTOFFS else FULL_CACHE


def _expected_rows(cycles: np.ndarray, cutoff: int) -> np.ndarray:
    horizon = 12 if cutoff in LONG_CUTOFFS else 4
    return np.flatnonzero((cycles > cutoff) & (cycles <= cutoff + horizon)).astype(np.int64)


def _read_train_features() -> tuple[Path, pd.DataFrame, np.ndarray]:
    train_path = task_files("regression").get("train")
    if train_path is None or not Path(train_path).is_file():
        raise FileNotFoundError("Official Regression Train is unavailable")
    train_path = Path(train_path)
    if file_sha256(train_path) != TRAIN_SHA256:
        raise BatchIntegrityError("Official Regression Train hash differs from the R19 card")
    frame = pd.read_csv(train_path, usecols=list(TRAIN_FEATURE_COLUMNS))
    if len(frame) != 50_608:
        raise BatchIntegrityError(f"Expected 50,608 Regression Train rows, got {len(frame)}")
    ids = frame["record_id"].astype(str)
    if frame["record_id"].isna().any() or ids.str.strip().eq("").any() or ids.duplicated().any():
        raise BatchIntegrityError("Official Train record_id is missing, blank, or duplicated")
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    if not np.array_equal(np.unique(cycles), np.arange(47, 80, dtype=np.int64)):
        raise BatchIntegrityError("Official Regression Train cycles differ from 47–79")
    ef = parse_entry_friction(frame["entry_friction"])
    if np.any((ef < 0) & frame["entry_friction"].notna().to_numpy(dtype=bool)):
        raise BatchIntegrityError("R19 entry_friction contains nonmissing, nonbinary values")
    return train_path, frame, cycles


def _row_key_array(path: Path, key: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Pinned R19 prediction cache is missing: {path}")
    with np.load(path, allow_pickle=False) as saved:
        if key not in saved.files:
            raise BatchIntegrityError(f"R19 row cache {path.name} lacks {key}")
        rows = np.asarray(saved[key], dtype=np.int64)
    if rows.ndim != 1 or len(np.unique(rows)) != len(rows):
        raise BatchIntegrityError(f"R19 cache rows are invalid: {path.name}:{key}")
    return rows


def _validate_row_caches(cycles: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for cutoff in CUT_OFFS:
        expected = _expected_rows(cycles, cutoff)
        if len(expected) != EXPECTED_SCORE_ROWS[cutoff]:
            raise BatchIntegrityError(f"R19 score row count changed at cutoff {cutoff}")
        base_rows = _row_key_array(BASE_CACHE, f"rows_{cutoff}")
        sparse_rows = _row_key_array(_row_cache_path(cutoff), f"rows_lgbm_sparse_l1_{cutoff}")
        if not (np.array_equal(base_rows, expected) and np.array_equal(sparse_rows, expected)):
            raise BatchIntegrityError(f"R19 source cache rows are not row-exact at cutoff {cutoff}")
        result[str(cutoff)] = {
            "score_rows": int(len(expected)),
            "score_cycle_min": int(cycles[expected].min()),
            "score_cycle_max": int(cycles[expected].max()),
            "row_keys": {"base": f"rows_{cutoff}", "sparse": f"rows_lgbm_sparse_l1_{cutoff}"},
        }
    for name, windows in {**CALIBRATION_WINDOWS, **EVALUATION_WINDOWS}.items():
        all_rows: list[np.ndarray] = []
        for cutoff, low, high in windows:
            rows = _row_key_array(BASE_CACHE, f"rows_{cutoff}")
            selected = (cycles[rows] >= low) & (cycles[rows] <= high)
            use = rows[selected]
            expected = np.flatnonzero((cycles >= low) & (cycles <= high)).astype(np.int64)
            if not np.array_equal(use, expected):
                raise BatchIntegrityError(f"R19 {name} window rows differ from official Train cycles")
            all_rows.append(use)
        combined = np.concatenate(all_rows)
        if len(combined) != len(np.unique(combined)):
            raise BatchIntegrityError(f"R19 {name} windows overlap")
        result[name] = {"rows": int(len(combined)), "unique_rows": int(len(np.unique(combined)))}
    return result


def _feature_support(frame: pd.DataFrame, cycles: np.ndarray) -> dict[str, Any]:
    ef = parse_entry_friction(frame["entry_friction"])
    stream = frame["signal_stream"]
    missing_stream = stream.isna().to_numpy(dtype=bool)
    nonnull = np.flatnonzero(~missing_stream)
    if len(nonnull):
        missing_stream[nonnull] = stream.iloc[nonnull].map(lambda value: not str(value).strip()).to_numpy(dtype=bool)
    groups = {
        "calibration_at_67_cycles_60_67": (cycles >= 60) & (cycles <= 67),
        "calibration_at_75_cycles_68_75": (cycles >= 68) & (cycles <= 75),
        "W4_cycles_68_71": (cycles >= 68) & (cycles <= 71),
        "W6_cycles_76_79": (cycles >= 76) & (cycles <= 79),
        "primary_W4_plus_W6": ((cycles >= 68) & (cycles <= 71)) | ((cycles >= 76) & (cycles <= 79)),
        "H12_L_cycles_68_79": (cycles >= 68) & (cycles <= 79),
        "H12_E_fallback_cycles_56_67": (cycles >= 56) & (cycles <= 67),
    }
    result: dict[str, Any] = {}
    for name, mask in groups.items():
        result[name] = {
            "rows": int(mask.sum()),
            "EF0": int(np.sum(mask & (ef == 0))),
            "EF1": int(np.sum(mask & (ef == 1))),
            "unknown_EF": int(np.sum(mask & (ef < 0))),
            "stream_missing": int(np.sum(mask & missing_stream)),
        }
    for cutoff, mask in ((67, groups["calibration_at_67_cycles_60_67"]),
                         (75, groups["calibration_at_75_cycles_68_75"])):
        if result[f"calibration_at_{cutoff}_cycles_{60 if cutoff == 67 else 68}_{cutoff}"]["EF0"] < 200:
            raise BatchIntegrityError(f"R19 calibration EF0 support below 200 at cutoff {cutoff}")
        if result[f"calibration_at_{cutoff}_cycles_{60 if cutoff == 67 else 68}_{cutoff}"]["EF1"] < 200:
            raise BatchIntegrityError(f"R19 calibration EF1 support below 200 at cutoff {cutoff}")
    return result


def _current_head() -> str:
    import subprocess

    completed = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                               check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def _prepare() -> dict[str, Any]:
    queue = _read_json(QUEUE_PATH)
    cards = {str(card["id"]): card for card in queue.get("cards", [])}
    card = cards.get(CARD_ID)
    if card is None or card.get("task") != "regression" or card.get("evaluation") != "LATE":
        raise BatchIntegrityError("Canonical R19 card is missing or not LATE")
    if int(card.get("pilot_predictor_fits", -1)) != 0:
        raise BatchIntegrityError("R19 must remain zero-fit")
    train_path, frame, cycles = _read_train_features()
    protocol = _read_json(DIAGNOSTICS_PROTOCOL_PATH)
    protocol_inputs = protocol.get("input_sha256", {})
    expected_inputs = {
        "data/raw/regression/Train.csv": TRAIN_SHA256,
        "experiments/runs/dual_reg_v1/stage1_paired_blends.npz": file_sha256(BASE_CACHE),
        "experiments/runs/optimization_v2_regression_repr/pilot_predictions.npz": file_sha256(PILOT_CACHE),
        "experiments/runs/optimization_v2_regression_repr/fullsix_predictions.npz": file_sha256(FULL_CACHE),
    }
    if any(protocol_inputs.get(path) != digest for path, digest in expected_inputs.items()):
        raise BatchIntegrityError("R19 source-cache hash is not pinned by diagnostics protocol")
    row_metadata = _validate_row_caches(cycles)
    support = _feature_support(frame, cycles)
    input_paths = [train_path, QUEUE_PATH, CARD_PATH, CONTROL_PATH, DIAGNOSTICS_PROTOCOL_PATH,
                   BASE_CACHE, PILOT_CACHE, FULL_CACHE]
    missing_inputs = [str(path) for path in input_paths if not path.is_file()]
    if missing_inputs:
        raise FileNotFoundError(f"R19 frozen input is missing: {missing_inputs}")
    input_hashes = {str(path.relative_to(ROOT)): file_sha256(path) for path in sorted(set(input_paths))}
    source_paths = _source_paths()
    missing_sources = [str(path) for path in source_paths if not path.is_file()]
    if missing_sources:
        raise FileNotFoundError(f"R19 source is missing: {missing_sources}")
    source_hashes = {str(path.relative_to(ROOT)): file_sha256(path) for path in sorted(set(source_paths))}
    recipe = {
        "card_id": CARD_ID,
        "task": "regression",
        "new_model_fits": 0,
        "baseline": "current three-component champion reconstructed from pinned caches",
        "champion_formula": "clip(0.75*blend_ridge25 + 0.25*cyclewise_rank50(raw_sparse_lgb_l1), 0, 100)",
        "rank_rule": "within each score cycle, 0.5*raw + 0.5*100*average_rank(raw)/n; clip [0,100]",
        "calibration_windows": CALIBRATION_WINDOWS,
        "calibration_rule": "For EF0/EF1 with >=200 prequential OOF rows: clip(0.5*median(threat_rank - champion_prediction), -2, 2); unknown/unsupported EF offset=0",
        "calibration_cutoffs": [67, 75],
        "score_windows": EVALUATION_WINDOWS,
        "score_boundary": "each cutoff's calibration cycles end before that cutoff's score cycles start",
        "post_adjustment_clip": [0.0, 100.0],
        "control_primary_mae_rounded": EXPECTED_BASELINE_MAE["primary"],
        "control_H12_L_mae_rounded": EXPECTED_BASELINE_MAE["H12_L"],
        "feature_only_support": support,
        "cache_row_metadata": row_metadata,
        "unverified_derived_cache_excluded": "experiments/runs/optimization_v3_reg_diagnostics/oof_residuals.npz",
        "test_or_sample_access": False,
        "kaggle_access": False,
    }
    identity = {
        "recipe": recipe,
        "train_sha256": file_sha256(train_path),
        "input_sha256": input_hashes,
        "source_sha256": source_hashes,
        "diagnostics_protocol_input_pins": expected_inputs,
        "source_commit": _current_head(),
    }
    return {
        "card": card, "frame_features": frame, "train_path": train_path,
        "cycles": cycles, "recipe": recipe, "identity": identity,
        "signature": json_sha256(identity), "row_metadata": row_metadata,
        "support": support,
    }


def _freeze() -> dict[str, Any]:
    context = _prepare()
    _write_once(RECIPE_PATH, context["recipe"])
    _write_once(FREEZE_PATH, {
        "status": "FROZEN_BEFORE_R19_CALIBRATION_AND_SCORING",
        "identity": context["identity"],
        "signature": context["signature"],
        "target_read": False,
        "prediction_arrays_read": False,
        "candidate_metric_computed": False,
        "test_read": False,
    })
    return {
        "card_id": CARD_ID, "status": "FROZEN_READY",
        "freeze_signature": context["signature"],
        "feature_only_support": context["support"],
        "target_read": False, "prediction_arrays_read": False,
        "candidate_metric_computed": False, "test_read": False,
    }


def _verify_frozen(context: dict[str, Any]) -> None:
    frozen = _read_json(FREEZE_PATH)
    if (frozen.get("status") != "FROZEN_BEFORE_R19_CALIBRATION_AND_SCORING"
            or frozen.get("signature") != context["signature"]
            or json_sha256(frozen.get("identity", {})) != context["signature"]):
        raise BatchIntegrityError("R19 freeze is missing or no longer matches current input/source hashes")
    normalized_recipe = json.loads(json.dumps(context["recipe"], sort_keys=True, allow_nan=False))
    if _read_json(RECIPE_PATH) != normalized_recipe:
        raise BatchIntegrityError("R19 recipe differs from its frozen identity")


def _read_prediction_caches() -> tuple[Any, Any, Any]:
    base = np.load(BASE_CACHE, allow_pickle=False)
    pilot = np.load(PILOT_CACHE, allow_pickle=False)
    full = np.load(FULL_CACHE, allow_pickle=False)
    return base, pilot, full


def _champion_window(frame: pd.DataFrame, caches: tuple[Any, Any, Any],
                     cutoff: int, low_cycle: int, high_cycle: int) -> dict[str, np.ndarray]:
    from scipy.stats import rankdata

    base, pilot, full = caches
    sparse = pilot if cutoff in LONG_CUTOFFS else full
    sparse_rows = np.asarray(sparse[f"rows_lgbm_sparse_l1_{cutoff}"], dtype=np.int64)
    sparse_raw = np.asarray(sparse[f"raw_lgbm_sparse_l1_{cutoff}"], dtype=np.float64)
    base_rows = np.asarray(base[f"rows_{cutoff}"], dtype=np.int64)
    if not np.array_equal(sparse_rows, base_rows):
        raise BatchIntegrityError(f"R19 baseline cache row mismatch at cutoff {cutoff}")
    cycles_all = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    expected_all = _expected_rows(cycles_all, cutoff)
    if not np.array_equal(sparse_rows, expected_all) or sparse_raw.shape != sparse_rows.shape:
        raise BatchIntegrityError(f"R19 sparse cache row shape differs at cutoff {cutoff}")
    selected = (cycles_all[sparse_rows] >= low_cycle) & (cycles_all[sparse_rows] <= high_cycle)
    rows = sparse_rows[selected]
    raw = sparse_raw[selected]
    cycles = cycles_all[rows]
    lgb = np.asarray(base[f"lgb_{cutoff}"], dtype=np.float64)[selected]
    ridge = np.asarray(base[f"ridge_{cutoff}"], dtype=np.float64)[selected]
    ridge25 = np.asarray(base[f"blend_ridge25_{cutoff}"], dtype=np.float64)[selected]
    if not np.isfinite(raw).all() or not np.isfinite(lgb).all() or not np.isfinite(ridge).all():
        raise BatchIntegrityError("R19 frozen cache contains nonfinite predictions")
    if not np.allclose(ridge25, 0.75 * lgb + 0.25 * ridge, rtol=0.0, atol=1e-10):
        raise BatchIntegrityError(f"R19 ridge25 blend identity mismatch at cutoff {cutoff}")
    rank50 = np.empty(len(raw), dtype=np.float64)
    for cycle in np.unique(cycles):
        mask = cycles == cycle
        rank50[mask] = 0.5 * raw[mask] + 0.5 * 100.0 * rankdata(raw[mask], method="average") / int(mask.sum())
    rank50 = np.clip(rank50, 0.0, 100.0)
    prediction = np.clip(0.75 * ridge25 + 0.25 * rank50, 0.0, 100.0)
    if not np.isfinite(prediction).all():
        raise BatchIntegrityError("R19 champion reconstruction is nonfinite")
    return {"rows": rows, "cycle": cycles, "prediction": prediction,
            "raw_sparse": raw, "ridge25": ridge25, "rank50": rank50}


def _concat_windows(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    result = {name: np.concatenate([part[name] for part in parts])
              for name in ("rows", "cycle", "prediction")}
    order = np.argsort(result["rows"], kind="stable")
    result = {key: value[order] for key, value in result.items()}
    if len(result["rows"]) != len(np.unique(result["rows"])):
        raise BatchIntegrityError("R19 windows contain duplicate Train rows")
    return result


def _mae(target: np.ndarray, prediction: np.ndarray) -> float:
    if target.ndim != 1 or prediction.shape != target.shape or not len(target):
        raise BatchIntegrityError("R19 metric arrays are empty or misaligned")
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise BatchIntegrityError("R19 metric arrays contain nonfinite values")
    return float(np.mean(np.abs(target.astype(np.float64) - prediction.astype(np.float64))))


def _group_diagnostics(frame: pd.DataFrame, rows: np.ndarray, target: np.ndarray,
                       baseline: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    ef = parse_entry_friction(frame.iloc[rows]["entry_friction"])
    y = target.astype(np.float64)
    missing_stream = frame.iloc[rows]["signal_stream"].isna().to_numpy(dtype=bool)
    stream = frame.iloc[rows]["signal_stream"]
    nonnull = np.flatnonzero(~missing_stream)
    if len(nonnull):
        missing_stream[nonnull] = stream.iloc[nonnull].map(lambda value: not str(value).strip()).to_numpy(dtype=bool)
    masks = {
        "EF0": ef == 0,
        "EF1": ef == 1,
        "EF_unknown": ef < 0,
        "tail_low_le5": y <= 5.0,
        "tail_high_ge95": y >= 95.0,
        "stream_missing": missing_stream,
    }
    output: dict[str, Any] = {}
    for name, mask in masks.items():
        n = int(mask.sum())
        supported = n >= 100
        output[name] = {
            "rows": n, "supported": supported,
            "baseline_mae": _mae(y[mask], baseline[mask]) if n else None,
            "candidate_mae": _mae(y[mask], candidate[mask]) if n else None,
            "harm_candidate_minus_baseline": (
                _mae(y[mask], candidate[mask]) - _mae(y[mask], baseline[mask])
                if n else None
            ),
            "gate_applies": name in {"tail_low_le5", "tail_high_ge95", "stream_missing"} and supported,
        }
    return output


def _score(context: dict[str, Any]) -> dict[str, Any]:
    _verify_frozen(context)
    train_path = task_files("regression").get("train")
    if train_path is None or file_sha256(Path(train_path)) != context["identity"]["train_sha256"]:
        raise BatchIntegrityError("R19 official Regression Train changed after freeze")
    frame = pd.read_csv(train_path)
    if len(frame) != 50_608 or "threat_rank" not in frame:
        raise BatchIntegrityError("R19 requires the exact official Regression Train target")
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    target = pd.to_numeric(frame["threat_rank"], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(target).all() or np.any((target < 0.0) | (target > 100.0)):
        raise BatchIntegrityError("R19 Train target is outside the frozen [0,100] range")
    caches = _read_prediction_caches()

    calibration: dict[int, dict[str, Any]] = {}
    for cutoff, windows in ((67, CALIBRATION_WINDOWS["67"]), (75, CALIBRATION_WINDOWS["75"])):
        pieces = [_champion_window(frame, caches, origin, low, high)
                  for origin, low, high in windows]
        combined = _concat_windows(pieces)
        if np.any(combined["cycle"] > cutoff):
            raise BatchIntegrityError(f"R19 calibration at cutoff {cutoff} uses a future cycle")
        calibration[cutoff] = combined

    baseline_views = {
        name: _concat_windows([_champion_window(frame, caches, cutoff, low, high)
                               for cutoff, low, high in segments])
        for name, segments in EVALUATION_WINDOWS.items()
    }
    baseline_metrics = {
        name: _mae(target[data["rows"]], data["prediction"])
        for name, data in baseline_views.items()
    }
    for name, expected in EXPECTED_BASELINE_MAE.items():
        if not np.isclose(baseline_metrics[name], expected, rtol=0.0, atol=1e-6):
            raise BatchIntegrityError(
                f"R19 row-matched baseline {name}={baseline_metrics[name]:.9f} differs from frozen card {expected:.6f}; candidate scoring stopped"
            )

    offsets: dict[int, dict[int, float]] = {}
    calibration_support: dict[str, Any] = {}
    for cutoff in (67, 75):
        data = calibration[cutoff]
        ef = frame.iloc[data["rows"]]["entry_friction"].to_numpy()
        offsets[cutoff], calibration_support[str(cutoff)] = fit_ef_median_offsets(
            target[data["rows"]], data["prediction"], ef, minimum_support=200,
        )

    candidate_views: dict[str, dict[str, np.ndarray]] = {}
    view_offsets = {"W4": 67, "W6": 75, "primary": None, "H12_L": 67, "H12_E": None}
    for name, baseline_data in baseline_views.items():
        if name in {"H12_E"}:
            prediction = baseline_data["prediction"].copy()
        else:
            applicable_cutoffs = (67, 75) if name == "primary" else (view_offsets[name],)
            prediction = baseline_data["prediction"].copy()
            for cutoff in applicable_cutoffs:
                for origin, low, high in EVALUATION_WINDOWS[name]:
                    if origin != cutoff:
                        continue
                    rows_mask = ((baseline_data["cycle"] >= low) & (baseline_data["cycle"] <= high))
                    ef = frame.iloc[baseline_data["rows"][rows_mask]]["entry_friction"].to_numpy()
                    prediction[rows_mask] = apply_ef_median_offsets(
                        baseline_data["prediction"][rows_mask], ef, offsets[cutoff],
                    )
        candidate_views[name] = {**baseline_data, "prediction": prediction}

    views: dict[str, Any] = {}
    group_diagnostics: dict[str, Any] = {}
    for name, base_data in baseline_views.items():
        candidate_data = candidate_views[name]
        rows = base_data["rows"]
        y = target[rows]
        base_mae = baseline_metrics[name]
        cand_mae = _mae(y, candidate_data["prediction"])
        views[name] = {
            "rows": int(len(rows)), "baseline_mae": base_mae, "candidate_mae": cand_mae,
            "gain_positive_is_better": float(base_mae - cand_mae),
            "harm_candidate_minus_baseline": float(cand_mae - base_mae),
            "offset_cutoffs": ("none_exact_fallback" if name == "H12_E"
                               else [67, 75] if name == "primary"
                               else [view_offsets[name]]),
        }
        if name in {"primary", "H12_L", "H12_E"}:
            group_diagnostics[name] = _group_diagnostics(
                frame, rows, y, base_data["prediction"], candidate_data["prediction"],
            )

    primary_gain = views["primary"]["gain_positive_is_better"]
    h12_harm = views["H12_L"]["harm_candidate_minus_baseline"]
    individual_harms = {name: views[name]["harm_candidate_minus_baseline"] for name in ("W4", "W6")}
    guarded_harms = [
        item["harm_candidate_minus_baseline"]
        for view_name in ("primary", "H12_L")
        for group_name, item in group_diagnostics[view_name].items()
        if group_name in {"tail_low_le5", "tail_high_ge95", "stream_missing"} and item["supported"]
    ]
    maximum_guarded_harm = max(guarded_harms) if guarded_harms else None
    checks = {
        "late_primary_gain_at_least_0_03": primary_gain >= 0.03,
        "H12_L_harm_at_most_0_5": h12_harm <= 0.5,
        "W4_and_W6_harm_at_most_0_5": all(value <= 0.5 for value in individual_harms.values()),
        "supported_tail_or_missing_harm_at_most_0_25": (
            maximum_guarded_harm is not None and maximum_guarded_harm <= 0.25
        ),
    }
    decision = "ADVANCE" if all(checks.values()) else "RETIRE"
    return {
        "card_id": CARD_ID, "status": f"PILOT_COMPLETE_{decision}", "decision": decision,
        "evaluation_profile": "LATE", "cutoffs": list(CUT_OFFS),
        "calibration_windows": CALIBRATION_WINDOWS,
        "calibration_support_and_offsets": calibration_support,
        "route_counts": context["support"], "views": views,
        "supported_group_diagnostics": group_diagnostics,
        "maximum_supported_tail_or_missing_harm": maximum_guarded_harm,
        "gate_checks": checks,
        "fit_accounting": {"predictor_fits": 0, "failed_fits": 0, "cache_only": True},
        "freeze_signature": context["signature"],
        "train_sha256": context["identity"]["train_sha256"],
        "input_sha256": context["identity"]["input_sha256"],
        "source_sha256": context["identity"]["source_sha256"],
        "source_commit": context["identity"]["source_commit"],
        "target_read_from_official_train_only": True,
        "derived_unpinned_oof_cache_used": False,
        "test_data_used": False, "sample_submission_used": False, "kaggle_access": False,
        "process_note": (
            "A preparatory read-only worker transiently loaded target/prediction cache arrays while inspecting schema; "
            "it did not print or use their values. R19 outputs were reconstructed from pinned source caches after freeze. "
            "The audit is not strictly target/prediction-blind."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight-only", action="store_true")
    modes.add_argument("--freeze-only", action="store_true")
    modes.add_argument("--run-frozen-pilot", action="store_true")
    args = parser.parse_args()
    context = _prepare()
    if args.freeze_only:
        result = _freeze()
    elif args.preflight_only:
        result = {
            "card_id": CARD_ID, "status": "PREFLIGHT_PASS",
            "freeze_signature": context["signature"],
            "feature_only_support": context["support"],
            "row_metadata": context["row_metadata"],
            "target_read": False, "prediction_arrays_read": False,
            "candidate_metric_computed": False, "test_read": False,
        }
    else:
        report = _score(context)
        _write_once(RESULTS_PATH, report)
        result = {"card_id": CARD_ID, "status": report["status"],
                  "decision": report["decision"], "results_path": str(RESULTS_PATH.relative_to(ROOT)),
                  "fit_count": 0, "test_data_used": False, "kaggle_access": False}
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

