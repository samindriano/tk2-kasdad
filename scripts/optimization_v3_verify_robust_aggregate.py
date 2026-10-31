"""Independent zero-fit verification for the frozen robust aggregation check."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/optimization_v3/regression_robust_aggregate"
OUT = ROOT / "reports/optimization_v3/regression_robust_aggregate"
FREEZE_PATH = CONFIG / "freeze.json"
PROTOCOL_PATH = CONFIG / "protocol.json"
RESULTS_PATH = OUT / "results.json"
VERIFY_PATH = OUT / "verification.json"
TRAIN_PATH = ROOT / "data/raw/regression/Train.csv"
BASE_PATH = ROOT / "experiments/runs/dual_reg_v1/stage1_paired_blends.npz"
PILOT_PATH = ROOT / "experiments/runs/optimization_v2_regression_repr/pilot_predictions.npz"
FULL_PATH = ROOT / "experiments/runs/optimization_v2_regression_repr/fullsix_predictions.npz"
SOURCES = (
    "scripts/optimization_v3_robust_aggregate.py",
    "scripts/optimization_v3_distributional.py",
    "scripts/optimization_v3_reg_diagnostics.py",
)
INPUTS = {
    "data/raw/regression/Train.csv": TRAIN_PATH,
    "experiments/runs/dual_reg_v1/stage1_paired_blends.npz": BASE_PATH,
    "experiments/runs/optimization_v2_regression_repr/pilot_predictions.npz": PILOT_PATH,
    "experiments/runs/optimization_v2_regression_repr/fullsix_predictions.npz": FULL_PATH,
}
ENDS = (55, 59, 63, 67, 71, 75)
VIEWS = ("W1", "W2", "W3", "W4", "W5", "W6",
         "H12_E", "H12_L", "primary", "H12")
TOL = 2e-10


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_digest(path: Path) -> str:
    # Match the frozen convention: UTF-8 text with CRLF normalized to LF.
    text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_digest(value) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=True,
                     allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _assert_close(actual, expected, label: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise AssertionError(f"{label}: mapping keys differ")
        for key in expected:
            _assert_close(actual[key], expected[key], f"{label}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AssertionError(f"{label}: sequence lengths differ")
        for index, (a, e) in enumerate(zip(actual, expected)):
            _assert_close(a, e, f"{label}[{index}]")
        return
    if isinstance(expected, bool):
        if actual is not expected:
            raise AssertionError(f"{label}: boolean mismatch")
        return
    if isinstance(expected, int):
        if not isinstance(actual, (int, np.integer)) or int(actual) != expected:
            raise AssertionError(f"{label}: integer mismatch")
        return
    if isinstance(expected, float):
        if not isinstance(actual, (int, float, np.integer, np.floating)) or not math.isclose(
                float(actual), expected, rel_tol=0.0, abs_tol=TOL):
            raise AssertionError(f"{label}: numeric mismatch ({actual!r} != {expected!r})")
        return
    if actual != expected:
        raise AssertionError(f"{label}: value mismatch")


def _rank50_independent(raw: np.ndarray, cycles: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    cycles = np.asarray(cycles)
    if raw.ndim != 1 or raw.shape != cycles.shape or not np.isfinite(raw).all():
        raise AssertionError("Sparse component and cycle vectors are invalid")
    # pandas' percentile rank is average-rank / group-size, independently of
    # the aggregator's scipy.stats.rankdata implementation.
    ranks = pd.Series(raw).groupby(pd.Series(cycles), sort=False).rank(
        method="average", pct=True).to_numpy(dtype=np.float64)
    return np.clip(0.5 * raw + 50.0 * ranks, 0.0, 100.0)


def _aggregate_independent(components: np.ndarray) -> dict[str, np.ndarray]:
    matrix = np.asarray(components, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != 3 or not np.isfinite(matrix).all():
        raise AssertionError("Expected three finite mapped component vectors")
    if np.any((matrix < 0.0) | (matrix > 100.0)):
        raise AssertionError("Mapped component is outside target range [0,100]")
    # Independent explicit algebra for the incumbent component recipe.
    champion = (9.0 * matrix[0] + 3.0 * matrix[1] + 4.0 * matrix[2]) / 16.0
    median3 = np.sort(matrix, axis=0)[1]
    median25 = 0.75 * champion + 0.25 * median3
    return {"champion": champion, "median3": median3, "median25": median25}


def _metric(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if y.ndim != 1 or y.shape != p.shape or not len(y) or not np.isfinite(p).all():
        raise AssertionError("Metric target/prediction vectors are invalid")
    residual = p - y
    return {
        "n": int(len(y)),
        "mae": float(np.mean(np.abs(residual))),
        "RMSE": float(np.sqrt(np.mean(np.square(residual)))),
        "bias": float(np.mean(residual)),
    }


def _view_metrics(frame: pd.DataFrame, rows: np.ndarray,
                  predictions: dict[str, np.ndarray]) -> dict:
    rows = np.asarray(rows, dtype=np.int64)
    if rows.ndim != 1 or len(np.unique(rows)) != len(rows):
        raise AssertionError("View has invalid or duplicate source row positions")
    selected = frame.iloc[rows]
    y = pd.to_numeric(selected.threat_rank, errors="raise").to_numpy(dtype=np.float64)
    cycles = pd.to_numeric(selected.cycle_code, errors="raise").to_numpy(dtype=np.int64)
    stream = selected.signal_stream
    missing = (stream.isna() | stream.fillna("").astype(str).str.strip().eq("")).to_numpy()
    ef = selected.entry_friction.to_numpy()
    masks = {
        "tail_le5": y <= 5,
        "tail_ge95": y >= 95,
        "EF0": ef == 0,
        "EF1": ef == 1,
        "stream_missing": missing,
        "stream_present": ~missing,
    }
    output = {}
    for name, prediction in predictions.items():
        prediction = np.asarray(prediction, dtype=np.float64)
        if prediction.shape != y.shape:
            raise AssertionError(f"{name} predictions misalign to rows")
        overall = _metric(y, prediction)
        per_cycle = {}
        cycle_maes = []
        for cycle in np.unique(cycles):
            use = cycles == cycle
            measure = _metric(y[use], prediction[use])
            per_cycle[str(int(cycle))] = {
                "n": measure["n"], "mae": measure["mae"], "bias": measure["bias"]}
            cycle_maes.append(measure["mae"])
        subgroups = {}
        for group, use in masks.items():
            if not use.any():
                raise AssertionError(f"Empty required subgroup {group} in view")
            measure = _metric(y[use], prediction[use])
            subgroups[group] = {
                "n": measure["n"], "mae": measure["mae"], "bias": measure["bias"]}
        output[name] = {
            "n": overall["n"], "mae": overall["mae"], "RMSE": overall["RMSE"],
            "bias": overall["bias"], "equal_cycle_mae": float(np.mean(cycle_maes)),
            "per_cycle": per_cycle, "subgroups": subgroups,
        }
    return output


def _load_frozen_inputs() -> tuple[dict, dict, pd.DataFrame]:
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    identity = freeze.get("identity")
    if not isinstance(identity, dict):
        raise AssertionError("Freeze identity missing")
    if freeze.get("signature") != _json_digest(identity):
        raise AssertionError("Freeze signature does not match its identity")
    if identity.get("protocol") != protocol:
        raise AssertionError("On-disk protocol differs from frozen protocol")
    source_hashes = {rel: _source_digest(ROOT / rel) for rel in SOURCES}
    if identity.get("source_sha256") != source_hashes:
        raise AssertionError("One or more frozen source hashes changed")
    input_hashes = {rel: _sha(path) for rel, path in INPUTS.items()}
    if identity.get("input_sha256") != input_hashes:
        raise AssertionError("One or more frozen input hashes changed")
    required_policy = {
        "new_model_fits": 0,
        "Test_read": False,
        "Kaggle_access": False,
    }
    for name, expected in required_policy.items():
        if protocol.get(name) is not expected:
            raise AssertionError(f"Protocol boundary changed: {name}")
    gate = protocol.get("gate", {})
    expected_gate = {
        "minimum_primary_MAE_gain": 0.03,
        "maximum_H12_MAE_harm": 0.05,
        "minimum_nonnegative_windows": 4,
        "maximum_supported_tail_or_missing_harm": 0.25,
        "minimum_subgroup_support": 100,
    }
    if gate != expected_gate:
        raise AssertionError("Frozen acceptance gate differs from the predeclared gate")
    frame = pd.read_csv(TRAIN_PATH)
    if len(frame) != 50608 or set(pd.to_numeric(frame.cycle_code, errors="raise").astype(int)) != set(range(47, 80)):
        raise AssertionError("Official Regression Train rows or cycles differ")
    return freeze, protocol, frame


def _reconstruct_views(frame: pd.DataFrame) -> tuple[dict, dict]:
    cycle_codes = pd.to_numeric(frame.cycle_code, errors="raise").to_numpy(dtype=np.int64)
    with np.load(BASE_PATH, allow_pickle=False) as base, \
            np.load(PILOT_PATH, allow_pickle=False) as pilot, \
            np.load(FULL_PATH, allow_pickle=False) as full:
        all_predictions: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
        audit = {}
        for end in ENDS:
            source = pilot if end in (55, 67) else full
            rows_all = np.asarray(source[f"rows_lgbm_sparse_l1_{end}"], dtype=np.int64)
            raw_all = np.asarray(source[f"raw_lgbm_sparse_l1_{end}"], dtype=np.float64)
            base_rows = np.asarray(base[f"rows_{end}"], dtype=np.int64)
            if not np.array_equal(rows_all, base_rows):
                raise AssertionError(f"cutoff {end}: component cache row ids differ")
            if rows_all.ndim != 1 or raw_all.shape != rows_all.shape or len(np.unique(rows_all)) != len(rows_all):
                raise AssertionError(f"cutoff {end}: invalid sparse cache rows or predictions")
            available_high = 67 if end == 55 else 79 if end == 67 else end + 4
            if not np.array_equal(rows_all, np.flatnonzero((cycle_codes > end) & (cycle_codes <= available_high))):
                raise AssertionError(f"cutoff {end}: cache rows are not exact cached future Train rows")

            lgb_all = np.asarray(base[f"lgb_{end}"], dtype=np.float64)
            ridge_all = np.asarray(base[f"ridge_{end}"], dtype=np.float64)
            ridge25_all = np.asarray(base[f"blend_ridge25_{end}"], dtype=np.float64)
            if any(v.shape != rows_all.shape for v in (lgb_all, ridge_all, ridge25_all)):
                raise AssertionError(f"cutoff {end}: mapped component vectors are not row-aligned")
            if not np.allclose(ridge25_all, 0.75 * lgb_all + 0.25 * ridge_all,
                               atol=1e-10, rtol=0.0):
                raise AssertionError(f"cutoff {end}: cached mapped Ridge25 decomposition differs")

            # Reconstruct every full cycle window and H12 segment from raw cached
            # components, while independently checking source-position coverage.
            highs = {end + 4}
            if end == 55:
                highs.add(67)
            if end == 67:
                highs.add(79)
            for high in sorted(highs):
                use = (cycle_codes[rows_all] > end) & (cycle_codes[rows_all] <= high)
                rows = rows_all[use]
                expected = np.flatnonzero((cycle_codes > end) & (cycle_codes <= high))
                if not np.array_equal(rows, expected):
                    raise AssertionError(f"cutoff {end} -> {high}: row positions do not match exact cycles")
                cycles = cycle_codes[rows]
                if set(np.unique(cycles)) != set(range(end + 1, high + 1)):
                    raise AssertionError(f"cutoff {end} -> {high}: missing cycle in coverage")
                lgb = lgb_all[use]
                ridge = ridge_all[use]
                raw_sparse = raw_all[use]
                sparse_rank50 = _rank50_independent(raw_sparse, cycles)
                predictions = _aggregate_independent(np.vstack((lgb, ridge, sparse_rank50)))
                label = f"{end}_{high}"
                if label in all_predictions:
                    raise AssertionError(f"duplicate aggregate view key: {label}")
                all_predictions[label] = (rows, predictions)
                audit[label] = {
                    "rows": int(len(rows)),
                    "row_positions_sha256": hashlib.sha256(np.asarray(rows, dtype="<i8").tobytes()).hexdigest(),
                    "cycles": [int(cycles.min()), int(cycles.max())],
                    "component_bounds": {
                        "lgb_svd32": [float(lgb.min()), float(lgb.max())],
                        "hybrid_ridge": [float(ridge.min()), float(ridge.max())],
                        "sparse_lgb_rank50": [float(sparse_rank50.min()), float(sparse_rank50.max())],
                    },
                }

    # Map exact cached fold blocks to the ten report views.
    views: dict[str, dict] = {}
    primary_parts: list[tuple[np.ndarray, dict[str, np.ndarray]]] = []
    for index, end in enumerate(ENDS, start=1):
        key = f"{end}_{end + 4}"
        rows, pred = all_predictions[key]
        views[f"W{index}"] = _view_metrics(frame, rows, pred)
        primary_parts.append((rows, pred))
    h12_parts: list[tuple[np.ndarray, dict[str, np.ndarray]]] = []
    for name, end, high in (("H12_E", 55, 67), ("H12_L", 67, 79)):
        rows, pred = all_predictions[f"{end}_{high}"]
        views[name] = _view_metrics(frame, rows, pred)
        h12_parts.append((rows, pred))
    pooled: dict[str, np.ndarray] = {}
    for name, parts in (("primary", primary_parts), ("H12", h12_parts)):
        rows = np.concatenate([part[0] for part in parts])
        if len(np.unique(rows)) != len(rows) or len(rows) != 38588:
            raise AssertionError(f"{name}: pooled OOF windows overlap or have incorrect support")
        if not np.array_equal(np.sort(rows), np.flatnonzero((cycle_codes >= 56) & (cycle_codes <= 79))):
            raise AssertionError(f"{name}: pooled rows do not cover cycles 56-79 exactly")
        predictions = {key: np.concatenate([part[1][key] for part in parts])
                       for key in ("champion", "median3", "median25")}
        views[name] = _view_metrics(frame, rows, predictions)
        pooled[name] = rows
    if not np.array_equal(np.sort(pooled["primary"]), np.sort(pooled["H12"])):
        raise AssertionError("Primary and H12 aggregates do not cover the same target rows")
    return views, audit


def _recompute_gates(views: dict, protocol: dict) -> dict:
    gate_cfg = protocol["gate"]
    output = {}
    for candidate in ("median3", "median25"):
        primary_gain = views["primary"]["champion"]["mae"] - views["primary"][candidate]["mae"]
        h12_harm = views["H12"][candidate]["mae"] - views["H12"]["champion"]["mae"]
        wins = sum(views[f"W{i}"][candidate]["mae"] <= views[f"W{i}"]["champion"]["mae"]
                   for i in range(1, 7))
        harms = []
        for view_name in VIEWS:
            view = views[view_name]
            for subgroup in ("tail_le5", "tail_ge95", "stream_missing"):
                group = view[candidate]["subgroups"][subgroup]
                base = view["champion"]["subgroups"][subgroup]
                if group["n"] != base["n"]:
                    raise AssertionError(f"{view_name}/{subgroup}: baseline/candidate supports differ")
                if group["n"] >= gate_cfg["minimum_subgroup_support"]:
                    harms.append((group["mae"] - base["mae"], view_name, subgroup, group["n"]))
        if not harms:
            raise AssertionError("No tail/missing subgroup reached the frozen support threshold")
        maximum_harm, harm_view, harm_group, harm_n = max(harms, key=lambda x: x[0])
        passes = bool(
            primary_gain >= gate_cfg["minimum_primary_MAE_gain"] and
            h12_harm <= gate_cfg["maximum_H12_MAE_harm"] and
            wins >= gate_cfg["minimum_nonnegative_windows"] and
            maximum_harm <= gate_cfg["maximum_supported_tail_or_missing_harm"]
        )
        output[candidate] = {
            "primary_gain": float(primary_gain),
            "H12_harm": float(h12_harm),
            "windows_nonnegative": int(wins),
            "maximum_supported_tail_or_missing_harm": float(maximum_harm),
            "maximum_harm_location": {"view": harm_view, "group": harm_group, "n": int(harm_n)},
            "pass": passes,
        }
    return output


def verify() -> dict:
    if VERIFY_PATH.exists():
        raise FileExistsError(f"Preserve existing independent receipt: {VERIFY_PATH}")
    freeze, protocol, frame = _load_frozen_inputs()
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    if (results.get("status") != "CHECKED" or results.get("new_fits") != 0 or
            results.get("Test_read") is not False or results.get("Kaggle_access") is not False or
            results.get("freeze_signature") != freeze["signature"]):
        raise AssertionError("Results report status/boundary/freeze mismatch")
    views, row_audit = _reconstruct_views(frame)
    if set(results.get("views", {})) != set(VIEWS):
        raise AssertionError("Results view set differs from the frozen ten-view scope")
    for view_name in VIEWS:
        _assert_close(results["views"][view_name], views[view_name], f"views.{view_name}")
    gates = _recompute_gates(views, protocol)
    for candidate in ("median3", "median25"):
        _assert_close(results.get("gates", {}).get(candidate),
                      {k: v for k, v in gates[candidate].items() if k != "maximum_harm_location"},
                      f"gates.{candidate}")
    receipt = {
        "status": "PASS",
        "verification_mode": "independent cache reconstruction; zero fits",
        "freeze_signature": freeze["signature"],
        "freeze_sha256": _sha(FREEZE_PATH),
        "protocol_sha256": _sha(PROTOCOL_PATH),
        "results_sha256": _sha(RESULTS_PATH),
        "source_hashes_verified": freeze["identity"]["source_sha256"],
        "input_hashes_verified": freeze["identity"]["input_sha256"],
        "official_train_rows": int(len(frame)),
        "new_fits": 0,
        "test_read": False,
        "kaggle_access": False,
        "row_component_audit": row_audit,
        "reconstructed_views": list(VIEWS),
        "all_view_metrics_match_results": True,
        "frozen_gate": protocol["gate"],
        "independent_gates": gates,
        "decision": {
            "median3": "FAIL_GATE",
            "median25": "FAIL_GATE",
            "median25_primary_gain_below_required": bool(
                gates["median25"]["primary_gain"] < protocol["gate"]["minimum_primary_MAE_gain"]),
            "gate_lowered": False,
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with VERIFY_PATH.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(receipt, handle, indent=2, ensure_ascii=True, allow_nan=False)
        handle.write("\n")
    return receipt


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, allow_nan=False))
