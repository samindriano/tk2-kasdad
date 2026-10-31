"""Independent rank, mixture, cache, fit-boundary and metric audit."""
from __future__ import annotations

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from scripts import optimization_v3_character as pilot
from scripts import optimization_v3_character_fullsix as run
from scripts import optimization_v3_distributional as distributional
from scripts import optimization_v3_reg_diagnostics as diagnostics
from src.models.optimization_v3_character import CharacterRidge
from src.models.round1_evaluation import array_digest
from src.models.round1_models import _fit_fingerprint
from src.models.regression_research_text import _array_digest


def _rank50(raw: np.ndarray, cycles: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=float)
    cycles = np.asarray(cycles)
    if raw.ndim != 1 or cycles.shape != raw.shape or not np.isfinite(raw).all():
        raise ValueError("Independent rank check requires aligned finite arrays")
    percentiles = np.empty(len(raw), dtype=float)
    for cycle in np.unique(cycles):
        use = cycles == cycle
        percentiles[use] = 100.0 * rankdata(raw[use], method="average") / int(use.sum())
    return np.clip(0.5 * raw + 0.5 * percentiles, 0.0, 100.0)


def _control_formula(frame: pd.DataFrame, end: int, low: int, high: int,
                     base_cache, sparse_cache) -> tuple[np.ndarray, np.ndarray]:
    key = f"rows_lgbm_sparse_l1_{end}"
    all_rows = np.asarray(sparse_cache[key], dtype=np.int64)
    all_sparse = np.asarray(sparse_cache[f"raw_lgbm_sparse_l1_{end}"], dtype=np.float64)
    cycles = frame.iloc[all_rows].cycle_code.to_numpy(dtype=int)
    mask = (cycles >= low) & (cycles <= high)
    rows = all_rows[mask]
    expected = np.flatnonzero(frame.cycle_code.between(low, high).to_numpy())
    np.testing.assert_array_equal(rows, expected)
    base_rows = np.asarray(base_cache[f"rows_{end}"], dtype=np.int64)
    np.testing.assert_array_equal(all_rows, base_rows)
    base_cycles = frame.iloc[base_rows].cycle_code.to_numpy(dtype=int)
    base_mask = (base_cycles >= low) & (base_cycles <= high)
    np.testing.assert_array_equal(rows, base_rows[base_mask])
    lgb = np.asarray(base_cache[f"lgb_{end}"], dtype=float)[base_mask]
    ridge = np.asarray(base_cache[f"ridge_{end}"], dtype=float)[base_mask]
    ridge25 = np.asarray(base_cache[f"blend_ridge25_{end}"], dtype=float)[base_mask]
    np.testing.assert_allclose(ridge25, 0.75 * lgb + 0.25 * ridge, atol=1e-10, rtol=0)
    raw = all_sparse[mask]
    mapped = _rank50(raw, frame.iloc[rows].cycle_code.to_numpy(dtype=int))
    champion = np.clip(0.75 * ridge25 + 0.25 * mapped, 0.0, 100.0)
    return rows, champion


def _control_views(frame: pd.DataFrame):
    specs = [run.PRIMARY_WINDOWS[name] for name in run.PRIMARY_WINDOWS]
    with np.load(diagnostics.PRIMARY_CACHE, allow_pickle=False) as base, \
         np.load(diagnostics.PILOT_CACHE, allow_pickle=False) as pilot_cache, \
         np.load(diagnostics.FULL_CACHE, allow_pickle=False) as full_cache:
        primary, primary_audit = diagnostics._build_view(frame, base, pilot_cache, full_cache, specs)
        h12, h12_audit = diagnostics._build_view(
            frame, base, pilot_cache, full_cache, [(55, 56, 67), (67, 68, 79)]
        )
        direct_specs = [(*run.PRIMARY_WINDOWS[name],) for name in run.PRIMARY_WINDOWS]
        direct_specs += [(55, 56, 67), (67, 68, 79)]
        for end, low, high in direct_specs:
            rows, champion = _control_formula(frame, end, low, high, base, full_cache if end not in (55, 67) else pilot_cache)
            if high - low == 3:
                view = primary.loc[(primary.cutoff == end) & primary.cycle.between(low, high)]
            else:
                view = h12.loc[(h12.cutoff == end) & h12.cycle.between(low, high)]
            np.testing.assert_array_equal(view.row.to_numpy(dtype=np.int64), rows)
            np.testing.assert_allclose(view.prediction.to_numpy(dtype=float), champion, atol=1e-10, rtol=0)
    return primary, h12, {"primary": primary_audit, "H12": h12_audit}


def _pilot_records(frame: pd.DataFrame, h12: pd.DataFrame) -> dict:
    frozen = pilot.check()
    receipt = run.read_json(pilot.OUT / "verification.json")
    assert receipt["status"] == "PASS" and receipt["freeze_signature"] == frozen["signature"]
    outputs = {}
    for end in (55, 67):
        path = pilot.CACHE / f"fit{end}.npz"
        meta = pilot.read(path.with_suffix(".json"))
        assert meta["npz_sha256"] == pilot.sha(path)
        assert meta["freeze_signature"] == frozen["signature"]
        with np.load(path, allow_pickle=False) as archive:
            d = {key: archive[key].copy() for key in archive.files}
        view = h12.loc[h12.cutoff.eq(end)]
        rows = view.row.to_numpy(dtype=np.int64)
        np.testing.assert_array_equal(d["rows"], rows)
        np.testing.assert_allclose(d["champion"], view.prediction.to_numpy(float), atol=1e-10, rtol=0)
        rank = _rank50(d["raw"], frame.iloc[rows].cycle_code.to_numpy(dtype=int))
        np.testing.assert_array_equal(rank, d["rank50"])
        mix = np.clip(0.75 * d["champion"] + 0.25 * rank, 0, 100)
        np.testing.assert_array_equal(mix, d["blend25"])
        outputs[end] = {
            "rows": rows,
            "raw": d["raw"].astype(float),
            "rank50": rank,
            "blend25": mix,
            "champion": d["champion"].astype(float),
            "state": meta["state"],
            "seconds": meta["seconds"],
            "source": "root_pilot_reused",
            "prediction_cache_sha256": pilot.sha(path),
        }
    return outputs


def _new_fit(frame: pd.DataFrame, target: np.ndarray, end: int,
             base_window: pd.DataFrame, signature: str) -> dict:
    path = run.CACHE / f"fit{end}.npz"
    meta = run.read_json(path.with_suffix(".json"))
    assert meta["freeze_signature"] == signature and meta["npz_sha256"] == run.raw_sha(path)
    with np.load(path, allow_pickle=False) as archive:
        d = {key: archive[key].copy() for key in archive.files}
    rows = base_window.row.to_numpy(dtype=np.int64)
    champion = base_window.prediction.to_numpy(dtype=float)
    np.testing.assert_array_equal(d["rows"], rows)
    np.testing.assert_allclose(d["champion"], champion, atol=1e-10, rtol=0)
    cycles = frame.iloc[rows].cycle_code.to_numpy(dtype=int)
    np.testing.assert_array_equal(d["rank50"], _rank50(d["raw"], cycles))
    expected_blend = np.clip(0.75 * champion + 0.25 * d["rank50"], 0, 100)
    np.testing.assert_array_equal(d["blend25"], expected_blend)
    fit = frame.cycle_code.between(47, end).to_numpy()
    state = meta["state"]
    fields = state["input_feature_names"]
    assert state["id"] == "V3_CHAR_HYBRID_RIDGE"
    assert state["fit_rows"] == int(fit.sum()) == meta["fit_rows"]
    assert state["fit_data_sha256"] == _fit_fingerprint(frame.loc[fit, fields], tuple(fields), target[fit])
    assert state["target_sha256"] == _array_digest(target[fit])
    assert state["tfidf"]["parameters"] == {
        "analyzer": "char_wb", "ngram_range": [3, 5], "min_df": 3,
        "max_features": 30000, "sublinear_tf": True, "lowercase": True,
        "strip_accents": "unicode", "dtype": "float64",
    }
    assert state["tfidf"]["word_tokens_used"] is False
    assert meta["outer_labels_used_for_fit"] is False and meta["test_read"] is False
    assert meta["raw_sha256"] == array_digest(d["raw"])
    assert meta["rank50_sha256"] == array_digest(d["rank50"])
    assert meta["blend25_sha256"] == array_digest(d["blend25"])
    assert len(np.unique(cycles)) == 4
    return {
        "rows": rows, "raw": d["raw"].astype(float), "rank50": d["rank50"].astype(float),
        "blend25": d["blend25"].astype(float), "champion": champion,
        "state": state, "seconds": meta["seconds"], "source": "new_fit",
        "prediction_cache_sha256": run.raw_sha(path),
    }


def _error(target, pred):
    target = np.asarray(target, dtype=float)
    pred = np.asarray(pred, dtype=float)
    residual = pred - target
    absolute = np.abs(residual)
    return {
        "n": int(len(target)),
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "bias_prediction_minus_target": float(residual.mean()),
        "median_signed_residual": float(np.median(residual)),
        "p90_absolute_error": float(np.quantile(absolute, 0.90)),
        "p95_absolute_error": float(np.quantile(absolute, 0.95)),
        "over_rate": float((residual > 0).mean()),
        "under_rate": float((residual < 0).mean()),
        "tie_rate": float((residual == 0).mean()),
        "target_p10": float(np.quantile(target, 0.10)),
        "target_median": float(np.median(target)),
        "target_p90": float(np.quantile(target, 0.90)),
    }


def _eval(view: pd.DataFrame, candidate: np.ndarray) -> dict:
    y = view.target.to_numpy(float)
    baseline = view.prediction.to_numpy(float)
    candidate = np.asarray(candidate, dtype=float)
    cy = view.cycle.to_numpy(dtype=int)
    out = {"n": len(view), "champion": _error(y, baseline), "character_blend25": _error(y, candidate)}
    out["delta"] = {
        "mae_harm_candidate_minus_champion": out["character_blend25"]["mae"] - out["champion"]["mae"],
        "mae_gain_champion_minus_candidate": out["champion"]["mae"] - out["character_blend25"]["mae"],
        "rmse_harm_candidate_minus_champion": out["character_blend25"]["rmse"] - out["champion"]["rmse"],
        "bias_shift_candidate_minus_champion": out["character_blend25"]["bias_prediction_minus_target"] - out["champion"]["bias_prediction_minus_target"],
        "p95_absolute_error_change": out["character_blend25"]["p95_absolute_error"] - out["champion"]["p95_absolute_error"],
    }
    per_cycle, base_mae, char_mae = {}, [], []
    for cycle in np.unique(cy):
        use = cy == cycle
        a, b = _error(y[use], baseline[use]), _error(y[use], candidate[use])
        per_cycle[str(int(cycle))] = {
            "champion": a,
            "character_blend25": b,
            "mae_gain": a["mae"] - b["mae"],
            "bias_shift": b["bias_prediction_minus_target"] - a["bias_prediction_minus_target"],
        }
        base_mae.append(a["mae"])
        char_mae.append(b["mae"])
    out["per_cycle"] = per_cycle
    out["equal_cycle_mae"] = {
        "champion": float(np.mean(base_mae)),
        "character_blend25": float(np.mean(char_mae)),
        "gain_champion_minus_candidate": float(np.mean(base_mae) - np.mean(char_mae)),
    }
    out["cycle_bias_direction_counts"] = {
        "champion_overpredicting_cycles": int(sum(v["champion"]["bias_prediction_minus_target"] > 0 for v in per_cycle.values())),
        "champion_underpredicting_cycles": int(sum(v["champion"]["bias_prediction_minus_target"] < 0 for v in per_cycle.values())),
        "candidate_overpredicting_cycles": int(sum(v["character_blend25"]["bias_prediction_minus_target"] > 0 for v in per_cycle.values())),
        "candidate_underpredicting_cycles": int(sum(v["character_blend25"]["bias_prediction_minus_target"] < 0 for v in per_cycle.values())),
    }
    groups = {}
    for column in run.GROUP_COLUMNS:
        by_label = {}
        for label, part in view.groupby(column, dropna=False, sort=True, observed=True):
            row_idx = view.index.isin(part.index)
            a, b = _error(y[row_idx], baseline[row_idx]), _error(y[row_idx], candidate[row_idx])
            n = int(row_idx.sum())
            by_label[str(label)] = {
                "n": n,
                "support_ge_100": bool(n >= 100),
                "champion": a,
                "character_blend25": b,
                "mae_harm_candidate_minus_champion": b["mae"] - a["mae"],
                "bias_shift_candidate_minus_champion": b["bias_prediction_minus_target"] - a["bias_prediction_minus_target"],
            }
        groups[column] = by_label
    out["subgroups"] = groups
    return out


def _primary_candidate_series(records: dict[int, dict], frame: pd.DataFrame) -> pd.Series:
    cycles = frame.cycle_code.to_numpy(dtype=int)
    pieces = []
    for _, (end, low, high) in run.PRIMARY_WINDOWS.items():
        record = records[end]
        use = (cycles[record["rows"]] >= low) & (cycles[record["rows"]] <= high)
        pieces.append(pd.Series(record["blend25"][use], index=record["rows"][use]))
    values = pd.concat(pieces)
    assert not values.index.duplicated().any()
    return values


def _h12_candidate_series(records: dict[int, dict]) -> pd.Series:
    values = pd.concat([
        pd.Series(records[end]["blend25"], index=records[end]["rows"])
        for end in [55, 67]
    ])
    assert not values.index.duplicated().any()
    return values


def _rebuild_results(frame, records, primary, h12):
    primary_scores = _primary_candidate_series(records, frame)
    h12_scores = _h12_candidate_series(records)
    views, parts = {}, []
    for name, (end, low, high) in run.PRIMARY_WINDOWS.items():
        view = primary.loc[(primary.cutoff == end) & primary.cycle.between(low, high)].copy()
        pred = primary_scores.reindex(view.row.to_numpy(dtype=np.int64)).to_numpy(dtype=float)
        assert np.isfinite(pred).all()
        views[name] = {
            "fit_cutoff": end, "cycles": [low, high],
            "fit_rows": int(frame.cycle_code.between(47, end).sum()),
            "validation_rows": len(view), **_eval(view, pred),
        }
        parts.append(view)
    pooled = pd.concat(parts, ignore_index=True)
    assert not pooled.row.duplicated().any()
    pooled_pred = primary_scores.reindex(pooled.row.to_numpy(dtype=np.int64)).to_numpy(dtype=float)
    views["primary_six_window_pooled"] = _eval(pooled, pooled_pred)
    hviews = {}
    for name, end, low, high in [("H12_E",55,56,67), ("H12_L",67,68,79)]:
        part = h12.loc[(h12.cutoff == end) & h12.cycle.between(low, high)].copy()
        pred = h12_scores.reindex(part.row.to_numpy(dtype=np.int64)).to_numpy(dtype=float)
        assert np.isfinite(pred).all()
        hviews[name] = part
        views[name] = _eval(part, pred)
    hp = pd.concat([hviews["H12_E"], hviews["H12_L"]], ignore_index=True)
    assert not hp.row.duplicated().any()
    hpred = h12_scores.reindex(hp.row.to_numpy(dtype=np.int64)).to_numpy(dtype=float)
    views["H12_pooled"] = _eval(hp, hpred)
    return views


def _compare_tree(actual, expected, where="metrics") -> int:
    n = 0
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), where
        for key in expected:
            n += _compare_tree(actual[key], expected[key], f"{where}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), where
        for i, (a, e) in enumerate(zip(actual, expected)):
            n += _compare_tree(a, e, f"{where}[{i}]")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        assert isinstance(actual, (int, float)) and np.isclose(actual, expected, rtol=0, atol=1e-12), where
        n += 1
    else:
        assert actual == expected, where
        n += 1
    return n


def verify() -> dict:
    frozen = run.check_freeze()
    frame, train_path = distributional.load()
    target = frame.threat_rank.to_numpy(dtype=float)
    assert len(frame) == 50608 and np.isfinite(target).all()
    primary, h12, _ = _control_views(frame)
    pilot_records = _pilot_records(frame, h12)
    records = dict(pilot_records)
    for end in run.NEW_ENDS:
        window = primary.loc[primary.cutoff.eq(end)]
        records[end] = _new_fit(frame, target, end, window, frozen["signature"])
    metrics = _rebuild_results(frame, records, primary, h12)
    report = run.read_json(run.OUT / "fullsix.json")
    assert report["freeze_signature"] == frozen["signature"]
    assert report["status"] == "READY_FOR_INDEPENDENT_VERIFICATION"
    assert report["test_read"] is False and report["kaggle_access"] is False
    assert report["new_fits"] == 4 and report["new_fit_cutoffs"] == [59, 63, 71, 75]
    assert report["reused_pilot_cutoffs"] == [55, 67]
    checks = _compare_tree(report["results"]["views"], metrics)
    gate = report["results"]["parent_guardrails"]
    supported = []
    for view_name, view in metrics.items():
        for col, groups in view["subgroups"].items():
            for group, values in groups.items():
                if values["support_ge_100"]:
                    supported.append({"view": view_name, "column": col, "group": group,
                                      "n": values["n"],
                                      "harm": values["mae_harm_candidate_minus_champion"]})
    violations = [row for row in supported if row["harm"] > run.MAX_GROUP_MAE_HARM]
    primary_gain = metrics["primary_six_window_pooled"]["delta"]["mae_gain_champion_minus_candidate"]
    h12_gain = metrics["H12_pooled"]["delta"]["mae_gain_champion_minus_candidate"]
    assert gate["primary_mae_gain"] == primary_gain
    assert gate["h12_mae_gain"] == h12_gain
    assert gate["supported_group_comparisons"] == len(supported)
    assert gate["supported_group_harm_violations"] == len(violations)
    assert gate["supported_group_harm_gate_pass"] == (not violations)
    result = {
        "status": "PASS",
        "freeze_signature": frozen["signature"],
        "source_and_input_hashes_verified": True,
        "official_train_sha256": run.raw_sha(train_path),
        "root_pilot_signature": pilot.check()["signature"],
        "root_pilot_verification_status": "PASS",
        "newfits": 0,
        "test_read": False,
        "kaggle_access": False,
        "four_fit_fingerprints_verified": True,
        "rank_mapping_recomputed_independently": True,
        "three_component_champion_reconstructed_from_v2_caches": True,
        "six_windows_and_fixed_origin_h12_verified": True,
        "independent_numeric_field_checks": checks,
        "supported_group_comparisons": len(supported),
        "supported_group_harm_violations": len(violations),
        "parent_guardrails": gate,
    }
    run.write_new(run.OUT / "verification.json", result)
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    verify()
