"""Fixed character-Ridge blend continuation over all six Regression windows."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

from scripts import optimization_v3_character as pilot
from scripts import optimization_v3_distributional as distributional
from scripts import optimization_v3_reg_diagnostics as diagnostics
from src.models.optimization_v3_character import CharacterRidge
from src.models.round1_evaluation import array_digest, code_digest, digest_json
from src.models.round1_models import _fit_fingerprint

CONFIG = ROOT / "experiments/optimization_v3/regression_character_full"
OUT = ROOT / "reports/optimization_v3/regression_character_full"
CACHE = ROOT / "experiments/runs/optimization_v3_regression_character_full"
ENDS = [55, 59, 63, 67, 71, 75]
NEW_ENDS = [59, 63, 71, 75]
H12_ENDS = [55, 67]
PRIMARY_WINDOWS = {
    "W1": (55, 56, 59),
    "W2": (59, 60, 63),
    "W3": (63, 64, 67),
    "W4": (67, 68, 71),
    "W5": (71, 72, 75),
    "W6": (75, 76, 79),
}
MIN_GROUP_SUPPORT = 100
MAX_GROUP_MAE_HARM = 0.25
CODE_FILES = [
    "src/models/optimization_v3_character.py",
    "scripts/optimization_v3_character.py",
    "scripts/optimization_v3_verify_character.py",
    "tests/test_optimization_v3_character.py",
    "scripts/optimization_v3_character_fullsix.py",
    "scripts/optimization_v3_verify_character_fullsix.py",
    "tests/test_optimization_v3_character_full.py",
    "scripts/optimization_v3_distributional.py",
    "scripts/optimization_v3_reg_diagnostics.py",
    "src/models/optimization_v3_distributional.py",
    "src/models/regression_research_common.py",
    "src/models/regression_research_text.py",
    "src/models/regression_round2_models.py",
    "src/models/regression_round2_metrics.py",
    "src/models/round1_models.py",
    "src/models/round1_evaluation.py",
    "src/utils/paths.py",
]
GROUP_COLUMNS = [
    "impact_load_q", "scope_pressure_q", "text_load_q", "scope_trace_count_q",
    "ef_group", "stream_missing", "tail", "category_any", "org_code_group",
    "module_code_group", "pattern_code_group",
]


def raw_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=True, allow_nan=False)
        handle.write("\n")


def source_head_contextual() -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()


def source_and_input_paths(train_path: Path) -> list[Path]:
    paths = [
        train_path,
        diagnostics.PRIMARY_CACHE,
        diagnostics.PILOT_CACHE,
        diagnostics.FULL_CACHE,
        diagnostics.OUT_DIR / "diagnostics.json",
        pilot.CONFIG / "protocol.json",
        pilot.CONFIG / "freeze.json",
        pilot.OUT / "pilot.json",
        pilot.OUT / "verification.json",
    ]
    for end in H12_ENDS:
        paths.extend([pilot.CACHE / f"fit{end}.npz", pilot.CACHE / f"fit{end}.json"])
    return paths


def make_identity() -> dict:
    frame, train_path = distributional.load()
    if len(frame) != 50608:
        raise AssertionError("Expected the official 50,608-row Regression Train")
    frozen_pilot = pilot.check()
    pilot_verification = read_json(pilot.OUT / "verification.json")
    pilot_result = read_json(pilot.OUT / "pilot.json")
    if (pilot_verification.get("status") != "PASS" or
            pilot_verification.get("freeze_signature") != frozen_pilot["signature"] or
            pilot_result.get("fits") != 2 or pilot_result.get("Test_read") is not False):
        raise AssertionError("Character pilot must be independently verified, Train-only, and exactly two fits")
    paths = source_and_input_paths(train_path)
    absent = [str(path) for path in paths if not path.is_file()]
    if absent:
        raise FileNotFoundError(f"Full-six frozen input is missing: {absent}")
    return {
        "protocol_sha256": raw_sha(CONFIG / "protocol.json"),
        "source_code_sha256_lf_utf8": {path: code_digest(ROOT / path) for path in CODE_FILES},
        "input_sha256_exact_bytes": {
            path.resolve().relative_to(ROOT).as_posix(): raw_sha(path) for path in paths
        },
        "pilot_freeze_signature": frozen_pilot["signature"],
        "pilot_verification_sha256": raw_sha(pilot.OUT / "verification.json"),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "threadpoolctl"]
        },
        "official_train_rows": len(frame),
        "official_train_sha256": raw_sha(train_path),
        "test_read": False,
        "kaggle_access": False,
    }


def freeze() -> dict:
    path = CONFIG / "freeze.json"
    current = make_identity()
    if path.exists():
        old = read_json(path)
        if old.get("identity") != current or old.get("signature") != digest_json(current):
            raise AssertionError("Existing full-six character freeze has changed")
        return old
    if CACHE.exists():
        raise AssertionError("Run cache exists before full-six freeze")
    frozen = {
        "status": "FROZEN_BEFORE_FOUR_CHARACTER_FITS",
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "source_head_contextual": source_head_contextual(),
        "identity": current,
        "signature": digest_json(current),
    }
    write_new(path, frozen)
    return frozen


def check_freeze() -> dict:
    frozen = read_json(CONFIG / "freeze.json")
    current = make_identity()
    if frozen.get("identity") != current or frozen.get("signature") != digest_json(current):
        raise AssertionError("Full-six character code/input identity differs from prefit freeze")
    return frozen


def rank50_within_cycle(raw: np.ndarray, cycles: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=float)
    cycles = np.asarray(cycles)
    if raw.ndim != 1 or cycles.shape != raw.shape or not np.isfinite(raw).all():
        raise ValueError("Character score and cycle arrays must be finite and aligned")
    rank = np.empty(len(raw), dtype=float)
    for cycle in np.unique(cycles):
        mask = cycles == cycle
        rank[mask] = 100.0 * rankdata(raw[mask], method="average") / int(mask.sum())
    return np.clip(0.5 * raw + 0.5 * rank, 0.0, 100.0)


def blend25(champion: np.ndarray, candidate_rank50: np.ndarray) -> np.ndarray:
    champion = np.asarray(champion, dtype=float)
    rank = np.asarray(candidate_rank50, dtype=float)
    if champion.ndim != 1 or rank.shape != champion.shape:
        raise ValueError("Champion and rank50 vectors must be aligned")
    if not np.isfinite(champion).all() or not np.isfinite(rank).all():
        raise ValueError("Champion and candidate must be finite")
    return np.clip(0.75 * champion + 0.25 * rank, 0.0, 100.0)


def load_base_views(frame: pd.DataFrame):
    with np.load(diagnostics.PRIMARY_CACHE, allow_pickle=False) as base, \
         np.load(diagnostics.PILOT_CACHE, allow_pickle=False) as pilot_cache, \
         np.load(diagnostics.FULL_CACHE, allow_pickle=False) as full_cache:
        primary, primary_audit = diagnostics._build_view(
            frame, base, pilot_cache, full_cache,
            [PRIMARY_WINDOWS[name] for name in PRIMARY_WINDOWS],
        )
        h12, h12_audit = diagnostics._build_view(
            frame, base, pilot_cache, full_cache, [(55, 56, 67), (67, 68, 79)]
        )
    if primary.row.duplicated().any() or h12.row.duplicated().any():
        raise AssertionError("Champion controls must have unique Train row positions")
    expected_primary = np.flatnonzero(frame.cycle_code.between(56, 79).to_numpy())
    expected_h12 = expected_primary.copy()
    np.testing.assert_array_equal(np.sort(primary.row.to_numpy(dtype=np.int64)), expected_primary)
    np.testing.assert_array_equal(np.sort(h12.row.to_numpy(dtype=np.int64)), expected_h12)
    return primary, h12, {"primary": primary_audit, "H12": h12_audit}


def load_pilot_predictions(frame: pd.DataFrame, h12_view: pd.DataFrame) -> dict[int, dict]:
    frozen = pilot.check()
    verified = read_json(pilot.OUT / "verification.json")
    if verified.get("status") != "PASS" or verified.get("freeze_signature") != frozen["signature"]:
        raise AssertionError("Root character pilot verification receipt mismatch")
    outputs = {}
    for end in H12_ENDS:
        path = pilot.CACHE / f"fit{end}.npz"
        meta = read_json(path.with_suffix(".json"))
        if meta.get("npz_sha256") != raw_sha(path) or meta.get("freeze_signature") != frozen["signature"]:
            raise AssertionError(f"Pilot cache identity mismatch for cutoff {end}")
        with np.load(path, allow_pickle=False) as archive:
            record = {key: archive[key].copy() for key in archive.files}
        expected_view = h12_view.loc[h12_view.cutoff.eq(end)]
        rows = expected_view.row.to_numpy(dtype=np.int64)
        np.testing.assert_array_equal(record["rows"], rows)
        np.testing.assert_array_equal(record["champion"], expected_view.prediction.to_numpy(dtype=float))
        if not np.isfinite(record["raw"]).all() or not np.isfinite(record["rank50"]).all():
            raise AssertionError(f"Nonfinite root pilot candidate at cutoff {end}")
        expected_rank = rank50_within_cycle(record["raw"], frame.iloc[rows].cycle_code.to_numpy(dtype=int))
        np.testing.assert_array_equal(expected_rank, record["rank50"])
        np.testing.assert_array_equal(blend25(record["champion"], expected_rank), record["blend25"])
        outputs[end] = {
            "end": end,
            "rows": rows,
            "raw": record["raw"].astype(float),
            "rank50": record["rank50"].astype(float),
            "blend25": record["blend25"].astype(float),
            "champion": record["champion"].astype(float),
            "state": meta["state"],
            "seconds": meta["seconds"],
            "source": "root_pilot_reused",
            "prediction_cache_sha256": raw_sha(path),
        }
    return outputs


def load_existing_newfit(end: int, frame: pd.DataFrame, target: np.ndarray,
                         rows: np.ndarray, champion: np.ndarray, signature: str) -> dict | None:
    path = CACHE / f"fit{end}.npz"
    meta_path = path.with_suffix(".json")
    if not path.exists() and not meta_path.exists():
        return None
    if not (path.is_file() and meta_path.is_file()):
        raise AssertionError(f"Partial fit cache at cutoff {end}")
    meta = read_json(meta_path)
    if meta.get("freeze_signature") != signature or meta.get("npz_sha256") != raw_sha(path):
        raise AssertionError(f"Fit-cache receipt mismatch at cutoff {end}")
    with np.load(path, allow_pickle=False) as archive:
        stored = {key: archive[key].copy() for key in archive.files}
    np.testing.assert_array_equal(stored["rows"], rows)
    np.testing.assert_array_equal(stored["champion"], champion)
    state = meta["state"]
    fit = frame.cycle_code.between(47, end).to_numpy()
    fields = state["input_feature_names"]
    if int(fit.sum()) != state["fit_rows"] or state["fit_data_sha256"] != _fit_fingerprint(
        frame.loc[fit, fields], tuple(fields), target[fit]
    ):
        raise AssertionError(f"Fit fingerprint mismatch at cutoff {end}")
    rank_expected = rank50_within_cycle(stored["raw"], frame.iloc[rows].cycle_code.to_numpy(dtype=int))
    np.testing.assert_array_equal(stored["rank50"], rank_expected)
    np.testing.assert_array_equal(stored["blend25"], blend25(champion, rank_expected))
    return {
        "end": end, "rows": rows, "raw": stored["raw"].astype(float),
        "rank50": stored["rank50"].astype(float), "blend25": stored["blend25"].astype(float),
        "champion": champion, "state": state, "seconds": meta["seconds"],
        "source": "new_fit_reused_from_frozen_continuation_cache",
        "prediction_cache_sha256": raw_sha(path),
    }


def fit_cutoff(end: int, frame: pd.DataFrame, target: np.ndarray, base_window: pd.DataFrame,
               signature: str) -> dict:
    rows = base_window.row.to_numpy(dtype=np.int64)
    champion = base_window.prediction.to_numpy(dtype=float)
    cached = load_existing_newfit(end, frame, target, rows, champion, signature)
    if cached is not None:
        return cached
    fit = frame.cycle_code.between(47, end).to_numpy()
    model = CharacterRidge()
    features = list(model.input_feature_names)
    started = time.perf_counter()
    with threadpool_limits(limits=2):
        model.fit(frame.loc[fit, features], target[fit])
        state = model.audit_state()
        raw = model.predict(frame.iloc[rows][features])
    if state != model.audit_state():
        raise AssertionError(f"CharacterRidge state changed during prediction at cutoff {end}")
    if state["fit_rows"] != int(fit.sum()) or state["fit_data_sha256"] != _fit_fingerprint(
        frame.loc[fit, features], tuple(features), target[fit]
    ):
        raise AssertionError(f"CharacterRidge fit did not bind only cycles 47-{end}")
    val_cycles = frame.iloc[rows].cycle_code.to_numpy(dtype=int)
    rank = rank50_within_cycle(raw, val_cycles)
    candidate = blend25(champion, rank)
    if len(np.unique(val_cycles)) != 4 or not np.array_equal(
        rows, np.flatnonzero(frame.cycle_code.between(end + 1, end + 4).to_numpy())
    ):
        raise AssertionError(f"Primary outer horizon misaligned at cutoff {end}")
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"fit{end}.npz"
    np.savez_compressed(path, rows=rows, raw=raw, rank50=rank, blend25=candidate,
                        champion=champion)
    seconds = time.perf_counter() - started
    metadata = {
        "freeze_signature": signature,
        "cutoff": end,
        "fit_cycle_range": [47, end],
        "validation_cycle_range": [end + 1, end + 4],
        "fit_rows": int(fit.sum()),
        "fit_data_sha256": state["fit_data_sha256"],
        "target_sha256": state["target_sha256"],
        "rank_rule": "within-cycle average rank, ties average; exact frozen rank50 mapping",
        "blend_rule": "clip(0.75*champion + 0.25*rank50, 0, 100)",
        "state": state,
        "raw_sha256": array_digest(raw),
        "rank50_sha256": array_digest(rank),
        "blend25_sha256": array_digest(candidate),
        "npz_sha256": raw_sha(path),
        "seconds": seconds,
        "test_read": False,
        "outer_labels_used_for_fit": False,
    }
    write_new(path.with_suffix(".json"), metadata)
    print(f"FITTED character cutoff={end} rows={len(rows)} vocab={state['tfidf']['vocabulary_size']} seconds={seconds:.1f}", flush=True)
    return {
        "end": end, "rows": rows, "raw": np.asarray(raw, dtype=float),
        "rank50": rank, "blend25": candidate, "champion": champion,
        "state": state, "seconds": seconds, "source": "new_fit",
        "prediction_cache_sha256": raw_sha(path),
    }


def error_summary(target: np.ndarray, prediction: np.ndarray) -> dict:
    actual = np.asarray(target, dtype=float)
    pred = np.asarray(prediction, dtype=float)
    if actual.ndim != 1 or pred.shape != actual.shape or len(actual) == 0:
        raise ValueError("Error metrics need aligned nonempty targets and predictions")
    if not np.isfinite(actual).all() or not np.isfinite(pred).all():
        raise ValueError("Error metric arrays must be finite")
    residual = pred - actual
    absolute = np.abs(residual)
    return {
        "n": int(len(actual)),
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "bias_prediction_minus_target": float(residual.mean()),
        "median_signed_residual": float(np.median(residual)),
        "p90_absolute_error": float(np.quantile(absolute, 0.90)),
        "p95_absolute_error": float(np.quantile(absolute, 0.95)),
        "over_rate": float(np.mean(residual > 0)),
        "under_rate": float(np.mean(residual < 0)),
        "tie_rate": float(np.mean(residual == 0)),
        "target_p10": float(np.quantile(actual, 0.10)),
        "target_median": float(np.median(actual)),
        "target_p90": float(np.quantile(actual, 0.90)),
    }


def evaluate_view(view: pd.DataFrame, candidate: np.ndarray) -> dict:
    if len(view) != len(candidate):
        raise ValueError("Subgroup view and candidate score lengths differ")
    actual = view.target.to_numpy(dtype=float)
    champion = view.prediction.to_numpy(dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    cycles = view.cycle.to_numpy(dtype=int)
    outputs = {
        "n": int(len(view)),
        "champion": error_summary(actual, champion),
        "character_blend25": error_summary(actual, candidate),
    }
    outputs["delta"] = {
        "mae_harm_candidate_minus_champion": outputs["character_blend25"]["mae"] - outputs["champion"]["mae"],
        "mae_gain_champion_minus_candidate": outputs["champion"]["mae"] - outputs["character_blend25"]["mae"],
        "rmse_harm_candidate_minus_champion": outputs["character_blend25"]["rmse"] - outputs["champion"]["rmse"],
        "bias_shift_candidate_minus_champion": (outputs["character_blend25"]["bias_prediction_minus_target"] -
                                                outputs["champion"]["bias_prediction_minus_target"]),
        "p95_absolute_error_change": outputs["character_blend25"]["p95_absolute_error"] - outputs["champion"]["p95_absolute_error"],
    }
    unique_cycles = np.unique(cycles)
    per_cycle = {}
    champion_cycle_mae, candidate_cycle_mae = [], []
    for cycle in unique_cycles:
        mask = cycles == cycle
        baseline_stats = error_summary(actual[mask], champion[mask])
        candidate_stats = error_summary(actual[mask], candidate[mask])
        per_cycle[str(int(cycle))] = {
            "champion": baseline_stats,
            "character_blend25": candidate_stats,
            "mae_gain": baseline_stats["mae"] - candidate_stats["mae"],
            "bias_shift": candidate_stats["bias_prediction_minus_target"] - baseline_stats["bias_prediction_minus_target"],
        }
        champion_cycle_mae.append(baseline_stats["mae"])
        candidate_cycle_mae.append(candidate_stats["mae"])
    outputs["per_cycle"] = per_cycle
    outputs["equal_cycle_mae"] = {
        "champion": float(np.mean(champion_cycle_mae)),
        "character_blend25": float(np.mean(candidate_cycle_mae)),
        "gain_champion_minus_candidate": float(np.mean(champion_cycle_mae) - np.mean(candidate_cycle_mae)),
    }
    outputs["cycle_bias_direction_counts"] = {
        "champion_overpredicting_cycles": int(sum(v["champion"]["bias_prediction_minus_target"] > 0 for v in per_cycle.values())),
        "champion_underpredicting_cycles": int(sum(v["champion"]["bias_prediction_minus_target"] < 0 for v in per_cycle.values())),
        "candidate_overpredicting_cycles": int(sum(v["character_blend25"]["bias_prediction_minus_target"] > 0 for v in per_cycle.values())),
        "candidate_underpredicting_cycles": int(sum(v["character_blend25"]["bias_prediction_minus_target"] < 0 for v in per_cycle.values())),
    }
    groups = {}
    for column in GROUP_COLUMNS:
        column_result = {}
        for label, part in view.groupby(column, dropna=False, sort=True, observed=True):
            mask = view.index.isin(part.index)
            base_stats = error_summary(actual[mask], champion[mask])
            candidate_stats = error_summary(actual[mask], candidate[mask])
            n = int(mask.sum())
            column_result[str(label)] = {
                "n": n,
                "support_ge_100": bool(n >= MIN_GROUP_SUPPORT),
                "champion": base_stats,
                "character_blend25": candidate_stats,
                "mae_harm_candidate_minus_champion": candidate_stats["mae"] - base_stats["mae"],
                "bias_shift_candidate_minus_champion": (candidate_stats["bias_prediction_minus_target"] -
                                                        base_stats["bias_prediction_minus_target"]),
            }
        groups[column] = column_result
    outputs["subgroups"] = groups
    return outputs


def primary_candidate_series(records: dict[int, dict], frame: pd.DataFrame) -> pd.Series:
    parts = []
    cycles = frame.cycle_code.to_numpy(dtype=int)
    for _, (end, low, high) in PRIMARY_WINDOWS.items():
        record = records[end]
        keep = (cycles[record["rows"]] >= low) & (cycles[record["rows"]] <= high)
        parts.append(pd.Series(record["blend25"][keep], index=record["rows"][keep]))
    joined = pd.concat(parts)
    if joined.index.duplicated().any():
        raise AssertionError("Primary candidate prediction rows overlap after window-specific slicing")
    return joined


def h12_candidate_series(records: dict[int, dict]) -> pd.Series:
    parts = [pd.Series(records[end]["blend25"], index=records[end]["rows"]) for end in H12_ENDS]
    joined = pd.concat(parts)
    if joined.index.duplicated().any():
        raise AssertionError("Fixed-origin H12 candidate rows overlap")
    return joined


def summarize_all(frame: pd.DataFrame, records: dict[int, dict], primary: pd.DataFrame,
                  h12: pd.DataFrame, primary_audit: list[dict], h12_audit: list[dict]) -> dict:
    primary_pred = primary_candidate_series(records, frame)
    h12_pred = h12_candidate_series(records)
    windows = {}
    primary_parts = []
    for name, (end, lo, hi) in PRIMARY_WINDOWS.items():
        view = primary.loc[(primary.cutoff == end) & primary.cycle.between(lo, hi)].copy()
        if not np.array_equal(view.row.to_numpy(dtype=np.int64),
                              np.flatnonzero(frame.cycle_code.between(lo, hi).to_numpy())):
            raise AssertionError(f"Primary rows do not align exactly for {name}")
        candidate = primary_pred.reindex(view.row.to_numpy(dtype=np.int64)).to_numpy(dtype=float)
        if not np.isfinite(candidate).all():
            raise AssertionError(f"Missing candidate probability/prediction rows in {name}")
        windows[name] = {
            "fit_cutoff": end,
            "cycles": [lo, hi],
            "fit_rows": int(frame.cycle_code.between(47, end).sum()),
            "validation_rows": int(len(view)),
            **evaluate_view(view, candidate),
        }
        primary_parts.append(view)
    primary_pooled = pd.concat(primary_parts, ignore_index=True)
    if primary_pooled.row.duplicated().any():
        raise AssertionError("Full-six primary validation rows overlap")
    primary_candidate = primary_pred.reindex(primary_pooled.row.to_numpy(dtype=np.int64)).to_numpy(dtype=float)
    primary_metrics = evaluate_view(primary_pooled, primary_candidate)
    h12_early = h12.loc[(h12.cutoff == 55) & h12.cycle.between(56, 67)].copy()
    h12_late = h12.loc[(h12.cutoff == 67) & h12.cycle.between(68, 79)].copy()
    h12_pooled = pd.concat([h12_early, h12_late], ignore_index=True)
    if h12_pooled.row.duplicated().any():
        raise AssertionError("Fixed-origin H12 rows overlap")
    h12_metrics = {}
    for name, view in [("H12_E", h12_early), ("H12_L", h12_late), ("H12_pooled", h12_pooled)]:
        candidate = h12_pred.reindex(view.row.to_numpy(dtype=np.int64)).to_numpy(dtype=float)
        if not np.isfinite(candidate).all():
            raise AssertionError(f"Missing reused pilot scores in {name}")
        h12_metrics[name] = evaluate_view(view, candidate)
    views = {**windows, "primary_six_window_pooled": primary_metrics, **h12_metrics}
    subgroup_harms = []
    for name, view in views.items():
        for column, groups in view["subgroups"].items():
            for label, result in groups.items():
                if result["support_ge_100"]:
                    subgroup_harms.append({
                        "view": name,
                        "group_column": column,
                        "group": label,
                        "n": result["n"],
                        "mae_harm": result["mae_harm_candidate_minus_champion"],
                    })
    violations = [g for g in subgroup_harms if g["mae_harm"] > MAX_GROUP_MAE_HARM]
    max_harm = max((group["mae_harm"] for group in subgroup_harms), default=None)
    primary_gain = primary_metrics["delta"]["mae_gain_champion_minus_candidate"]
    h12_gain = h12_metrics["H12_pooled"]["delta"]["mae_gain_champion_minus_candidate"]
    return {
        "views": views,
        "fit_volume": {
            str(end): {
                "fit_rows": int(records[end]["state"]["fit_rows"]),
                "validation_rows": int(len(records[end]["rows"])),
                "primary_window_rows": int(frame.cycle_code.between(end + 1, end + 4).sum()),
                "character_vocab_size": int(records[end]["state"]["tfidf"]["vocabulary_size"]),
                "idf_sha256": records[end]["state"]["tfidf"]["idf_sha256"],
                "vocabulary_sha256": records[end]["state"]["tfidf"]["vocabulary_sha256"],
                "fit_data_sha256": records[end]["state"]["fit_data_sha256"],
                "source": records[end]["source"],
            } for end in ENDS
        },
        "fit_defined_group_audits": {"primary": primary_audit, "H12": h12_audit},
        "parent_guardrails": {
            "primary_mae_gain": float(primary_gain),
            "h12_mae_gain": float(h12_gain),
            "primary_or_h12_improved": bool(primary_gain > 0 or h12_gain > 0),
            "max_supported_group_mae_harm": float(max_harm) if max_harm is not None else None,
            "max_supported_group_harm_limit": MAX_GROUP_MAE_HARM,
            "supported_group_comparisons": len(subgroup_harms),
            "supported_group_harm_violations": len(violations),
            "supported_group_harm_gate_pass": not violations,
            "worst_supported_group_harms": sorted(subgroup_harms, key=lambda g: g["mae_harm"], reverse=True)[:20],
            "violating_supported_groups": violations,
        },
        "group_support_minimum": MIN_GROUP_SUPPORT,
    }


def write_markdown(result: dict) -> None:
    views = result["results"]["views"]
    lines = [
        "# V3 Regression Character Ridge — Full Six Windows", "",
        "This continues only the frozen 25% character-Ridge rank50 component blended with 75% of the current three-component champion. The current champion is reconstructed from the official V2 row-aligned caches by the V3 diagnostic `_build_view` helper. Four new fits use cutoffs 59, 63, 71, and 75; cutoff 55 and 67 predictions are reused from the independently verified pilot.", "",
        "No parameters, rank rule, blend weight, or clipping rule changed. No official Test data or Kaggle service was read. These temporal windows have already been used during adaptive development, so results are comparative development evidence, not an untouched holdout estimate.", "",
        "## Six-window primary results", "",
        "| Window | n | Champion MAE | Character blend MAE | MAE gain | Equal-cycle gain | Bias shift | Candidate p95 AE |", "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ["W1", "W2", "W3", "W4", "W5", "W6"]:
        v = views[name]
        lines.append(
            f"| {name} | {v['n']:,} | {v['champion']['mae']:.4f} | {v['character_blend25']['mae']:.4f} | {v['delta']['mae_gain_champion_minus_candidate']:+.4f} | {v['equal_cycle_mae']['gain_champion_minus_candidate']:+.4f} | {v['delta']['bias_shift_candidate_minus_champion']:+.3f} | {v['character_blend25']['p95_absolute_error']:.3f} |"
        )
    lines += ["", "## Pooled and H12 views", "",
              "| View | n | Champion MAE | Character blend MAE | MAE gain | Equal-cycle gain | Bias shift |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, label in [("primary_six_window_pooled", "Primary six-window pooled"),
                        ("H12_pooled", "H12 fixed-origin pooled"),
                        ("H12_E", "H12 early (56–67)"), ("H12_L", "H12 late (68–79)")]:
        v = views[name]
        lines.append(
            f"| {label} | {v['n']:,} | {v['champion']['mae']:.4f} | {v['character_blend25']['mae']:.4f} | {v['delta']['mae_gain_champion_minus_candidate']:+.4f} | {v['equal_cycle_mae']['gain_champion_minus_candidate']:+.4f} | {v['delta']['bias_shift_candidate_minus_champion']:+.3f} |"
        )
    gate = result["results"]["parent_guardrails"]
    lines += ["", "## Guardrails and subgroup coverage", "",
              f"- Primary MAE gain: **{gate['primary_mae_gain']:+.4f}**.",
              f"- H12 pooled MAE gain: **{gate['h12_mae_gain']:+.4f}**.",
              f"- Supported subgroup harm limit: **{gate['max_supported_group_harm_limit']:.2f} MAE** across **{gate['supported_group_comparisons']}** comparisons with n≥{MIN_GROUP_SUPPORT}; max harm was **{gate['max_supported_group_mae_harm']:+.4f}**; violations: **{gate['supported_group_harm_violations']}**.",
              "- Subgroups include fit-defined impact_load, scope_pressure, text_load, and scope_trace quantiles; EF0/EF1; stream present/missing; target tails; and raw category novelty. Target tails are descriptive after scoring only.",
              "- Full per-cycle, group, bias, and support tables are in `fullsix.json`.", "",
              "## Fit-defined character volume", "",
              "| Cutoff | Fit rows | Validation rows | Char vocabulary | Fit source |", "|---:|---:|---:|---:|---|"]
    for end in ENDS:
        v = result["results"]["fit_volume"][str(end)]
        lines.append(f"| {end} | {v['fit_rows']:,} | {v['validation_rows']:,} | {v['character_vocab_size']:,} | {v['source']} |")
    lines += ["", "Reproduce the four new fits with `python scripts/optimization_v3_character_fullsix.py --run` after reviewing the frozen identity. Independently verify saved predictions with `python scripts/optimization_v3_verify_character_fullsix.py`.", ""]
    path = OUT / "FULLSIX.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))


def run() -> dict:
    frozen = check_freeze()
    frame, target_path = distributional.load()
    target = frame.threat_rank.to_numpy(dtype=float)
    if len(frame) != 50608 or not np.isfinite(target).all() or ((target < 0) | (target > 100)).any():
        raise AssertionError("Official Regression Train target validation failed")
    primary, h12, audits = load_base_views(frame)
    records = load_pilot_predictions(frame, h12)
    for end in NEW_ENDS:
        window = primary.loc[primary.cutoff.eq(end)].copy()
        records[end] = fit_cutoff(end, frame, target, window, frozen["signature"])
    results = summarize_all(frame, records, primary, h12, audits["primary"], audits["H12"])
    result = {
        "status": "READY_FOR_INDEPENDENT_VERIFICATION",
        "freeze_signature": frozen["signature"],
        "source_head_contextual": frozen["source_head_contextual"],
        "official_train_rows": len(frame),
        "official_train_sha256": raw_sha(target_path),
        "test_read": False,
        "kaggle_access": False,
        "new_fits": 4,
        "new_fit_cutoffs": NEW_ENDS,
        "reused_pilot_cutoffs": H12_ENDS,
        "rank_rule": "rank50=clip(0.5*raw + 0.5*(100*average_rank_within_cycle/n), 0, 100)",
        "blend_rule": "clip(0.75*current_three_component_champion + 0.25*rank50, 0, 100)",
        "results": results,
        "limits": [
            "Primary windows and H12 overlap the same official Train cycles and prior adaptive work; neither is an untouched holdout.",
            "Target tails are diagnostic groups formed after predictions and are not deployment routes.",
            "Numeric quantile groups are fold-relative and fit-defined; labels such as Q1 do not imply identical global thresholds.",
            "Groups below n=100 are shown as descriptive and excluded from the all-supported-groups 0.25 MAE harm gate.",
        ],
    }
    write_new(OUT / "fullsix.json", result)
    write_markdown(result)
    print(json.dumps({
        "status": result["status"],
        "primary_gain": results["parent_guardrails"]["primary_mae_gain"],
        "h12_gain": results["parent_guardrails"]["h12_mae_gain"],
        "max_supported_group_harm": results["parent_guardrails"]["max_supported_group_mae_harm"],
        "supported_group_harm_violations": results["parent_guardrails"]["supported_group_harm_violations"],
    }, indent=2, allow_nan=False), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        frozen = freeze()
        print(frozen["signature"])
    else:
        run()
