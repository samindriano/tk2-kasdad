"""Independent zero-fit verifier for the frozen POS/A02 endgame outputs."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/optimization_v3/posoffset_a02_endgame_v1"
RUN = ROOT / "experiments/runs/optimization_v3_posoffset_a02_endgame_v1"
TRAIN = ROOT / "data/raw/classification/Train.csv"
FREEZE = REPORT / "FREEZE.json"
METRICS = REPORT / "POLICY_METRICS.json"
PREDICTIONS = RUN / "policy_predictions.npz"
TOL = 1e-12


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canon(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _metric(y: np.ndarray, pred: np.ndarray) -> dict:
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    tp = int(np.sum((y == 1) & (pred == 1)))
    f0den = 2 * tn + fp + fn
    f1den = 2 * tp + fp + fn
    f0 = 0.0 if f0den == 0 else (2 * tn / f0den)
    f1 = 0.0 if f1den == 0 else (2 * tp / f1den)
    recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
    return {"n": len(y), "class_0": int(np.sum(y == 0)), "class_1": int(np.sum(y == 1)),
            "confusion_matrix": [[tn, fp], [fn, tp]], "macro_f1": (f0 + f1) / 2,
            "positive_recall": recall}


def _assert_metric(actual: dict, expected: dict, label: str) -> None:
    for key in ("n", "class_0", "class_1", "confusion_matrix"):
        if actual[key] != expected[key]:
            raise AssertionError(f"{label} {key} mismatch")
    for key in ("macro_f1", "positive_recall"):
        if not math.isclose(float(actual[key]), float(expected[key]), rel_tol=0, abs_tol=TOL):
            raise AssertionError(f"{label} {key} mismatch: {actual[key]} != {expected[key]}")


def verify() -> dict:
    freeze = _load(FREEZE)
    signature = freeze.get("freeze_signature")
    freeze_body = dict(freeze)
    freeze_body.pop("freeze_signature", None)
    if signature != _canon(freeze_body):
        raise AssertionError("Freeze signature mismatch")
    for rel, digest in freeze["source_hashes"].items():
        if _sha(ROOT / rel) != digest:
            raise AssertionError(f"Frozen source hash mismatch: {rel}")
    for rel, digest in freeze["input_hashes"].items():
        if _sha(ROOT / rel) != digest:
            raise AssertionError(f"Frozen input hash mismatch: {rel}")
    train = pd.read_csv(TRAIN, usecols=["record_id", "cycle_code", "signal_stream", "entry_friction", "text_load", "artifact_signal"], dtype={"record_id": str})
    if _sha(TRAIN) != freeze["input_hashes"]["data/raw/classification/Train.csv"]:
        raise AssertionError("Train hash mismatch")
    metrics = _load(METRICS)
    if metrics.get("freeze_signature") != signature:
        raise AssertionError("Metric output points to a different freeze")
    with np.load(PREDICTIONS, allow_pickle=False) as z:
        p = {key: np.asarray(z[key]).copy() for key in z.files}
    if metrics.get("prediction_npz_sha256") != _sha(PREDICTIONS):
        raise AssertionError("Prediction bundle hash mismatch")

    y_train = pd.to_numeric(train.artifact_signal, errors="raise").to_numpy(dtype=np.int8)
    cycle_train = pd.to_numeric(train.cycle_code, errors="raise").to_numpy(dtype=np.int64)
    missing_train = train.signal_stream.fillna("").astype(str).str.strip().eq("").to_numpy()
    checks = 0
    per_origin = {}
    for origin in (67, 75):
        prefix = f"E{origin}_"
        rows = p[prefix + "rows"].astype(np.int64)
        ids = p[prefix + "ids"].astype(str)
        cycles = p[prefix + "cycles"].astype(np.int64)
        y = p[prefix + "y"].astype(np.int8)
        if not np.array_equal(ids, train.record_id.to_numpy(dtype=str)[rows]):
            raise AssertionError(f"E{origin} ID join mismatch")
        if not np.array_equal(cycles, cycle_train[rows]) or not np.array_equal(y, y_train[rows]):
            raise AssertionError(f"E{origin} cycle/target join mismatch")
        missing = p[prefix + "missing"].astype(bool)
        if not np.array_equal(missing, missing_train[rows]):
            raise AssertionError(f"E{origin} availability mismatch")
        p_pos, p_a, b = (p[prefix + key].astype(np.float64) for key in ("pP", "pA", "b"))
        for arr in (p_pos, p_a, b):
            if not np.isfinite(arr).all() or np.any((arr < 0) | (arr > 1)):
                raise AssertionError(f"E{origin} invalid comparator probabilities")
        reconstructed = {
            "MIX50": np.where(missing, p_pos, 0.5 * p_pos + 0.5 * p_a),
            "ADD_RESIDUAL": np.where(missing, p_pos, np.clip(p_pos + (p_a - b), 0, 1)),
        }
        for policy, expected in reconstructed.items():
            if not np.array_equal(expected, p[prefix + policy].astype(np.float64)):
                raise AssertionError(f"E{origin}/{policy} formula mismatch")
        groups = {
            "availability": np.where(missing, "missing", "present"),
            "entry_friction": np.asarray([str(int(x)) for x in pd.to_numeric(train.loc[rows, "entry_friction"], errors="raise").to_numpy()]),
        }
        edges = np.quantile(pd.to_numeric(train.loc[cycle_train <= origin, "text_load"], errors="raise").to_numpy(dtype=np.float64),
                            [0.25, 0.50, 0.75], method="linear")
        text_group = np.searchsorted(edges,
                                     pd.to_numeric(train.loc[rows, "text_load"], errors="raise").to_numpy(dtype=np.float64),
                                     side="right")
        groups["text_load"] = np.asarray([f"Q{int(x)+1}" for x in text_group])
        for family, vals in groups.items():
            if not np.array_equal(vals.astype(str), p[prefix + "group_" + family].astype(str)):
                raise AssertionError(f"E{origin} {family} grouping mismatch")
        per_origin[str(origin)] = {"rows": len(rows), "formula_checks": 2, "id_cycle_mask_target_join": True}
        checks += 5

    expected_views = {
        "E67_H12": (67, 68, 79), "E67_W4": (67, 68, 71), "E67_W5": (67, 72, 75),
        "E67_W6": (67, 76, 79), "E75_W6": (75, 76, 79),
    }
    gate_results = {}
    for view, (origin, first, last) in expected_views.items():
        prefix = f"E{origin}_"
        rows = p[prefix + "rows"].astype(np.int64)
        cycles = p[prefix + "cycles"].astype(np.int64)
        y = p[prefix + "y"].astype(np.int8)
        mask = (cycles >= first) & (cycles <= last)
        out = metrics["views"][view]
        for comp in ("pP", "pA", "b"):
            prob = p[prefix + comp].astype(np.float64)
            _assert_metric(out["comparators"][comp], _metric(y[mask], (prob[mask] >= 0.5).astype(np.int8)), f"{view}/{comp}")
            checks += 1
        for policy in ("MIX50", "ADD_RESIDUAL"):
            prob = p[prefix + policy].astype(np.float64)
            _assert_metric(out["policies"][policy], _metric(y[mask], (prob[mask] >= 0.5).astype(np.int8)), f"{view}/{policy}")
            checks += 1
            for family, vals in ((name, p[prefix + "group_" + name].astype(str)) for name in groups):
                report_groups = out["subgroups"][family]
                for level, entry in report_groups.items():
                    gmask = mask & (vals == level)
                    gy = y[gmask]
                    supported = len(gy) >= 200 and int(np.sum(gy == 0)) >= 30 and int(np.sum(gy == 1)) >= 30
                    if not supported:
                        if entry["status"] != "UNKNOWN_SUPPORT":
                            raise AssertionError(f"{view}/{family}/{level} unsupported group not marked UNKNOWN")
                        continue
                    if entry["status"] != "SUPPORTED":
                        raise AssertionError(f"{view}/{family}/{level} supported group not marked SUPPORTED")
                    pp = p[prefix + "pP"].astype(np.float64)
                    _assert_metric(entry["pP"], _metric(gy, (pp[gmask] >= 0.5).astype(np.int8)), "subgroup/pP")
                    _assert_metric(entry[policy], _metric(gy, (prob[gmask] >= 0.5).astype(np.int8)), "subgroup/policy")
                    harm = {"macro_f1": entry["pP"]["macro_f1"] - entry[policy]["macro_f1"],
                            "positive_recall": entry["pP"]["positive_recall"] - entry[policy]["positive_recall"]}
                    for metric in harm:
                        if not math.isclose(float(entry[f"{policy}_harm_vs_pP"][metric]), harm[metric], rel_tol=0, abs_tol=TOL):
                            raise AssertionError(f"{view}/{family}/{level}/{policy} harm mismatch")
                    checks += 2

    for policy in ("MIX50", "ADD_RESIDUAL"):
        checks_json = metrics["policies"][policy]["checks"]
        gains = {}
        harms = {}
        for view in ("E67_H12", "E75_W6"):
            gains[view] = metrics["views"][view]["policies"][policy]["macro_f1"] - metrics["views"][view]["comparators"]["pP"]["macro_f1"]
            if checks_json[f"gain_{view}_at_least_0.0005"] != (gains[view] + TOL >= 0.0005):
                raise AssertionError(f"{policy} gain gate mismatch at {view}")
        for view in ("E67_W4", "E67_W5", "E67_W6"):
            harms[view] = metrics["views"][view]["comparators"]["pP"]["macro_f1"] - metrics["views"][view]["policies"][policy]["macro_f1"]
            if checks_json[f"harm_{view}_at_most_0.003"] != (harms[view] <= 0.003 + TOL):
                raise AssertionError(f"{policy} window gate mismatch at {view}")
        all_ok = True
        for view, view_data in metrics["views"].items():
            for levels in view_data["subgroups"].values():
                for entry in levels.values():
                    if entry["status"] != "SUPPORTED":
                        continue
                    harm = entry[f"{policy}_harm_vs_pP"]
                    if harm["macro_f1"] > 0.01 + TOL or harm["positive_recall"] > 0.02 + TOL:
                        all_ok = False
        if checks_json["all_supported_subgroups_within_limits"] != all_ok:
            raise AssertionError(f"{policy} subgroup gate mismatch")
        qualifies = all(checks_json.values())
        if metrics["policies"][policy]["qualifies"] != qualifies:
            raise AssertionError(f"{policy} qualification mismatch")
        gate_results[policy] = {"qualifies": qualifies, "required_gains": gains, "window_harms": harms}
    selected = next((name for name in ("MIX50", "ADD_RESIDUAL") if gate_results[name]["qualifies"]), None)
    if metrics["selection"]["selected_policy_by_frozen_priority"] != selected:
        raise AssertionError("Frozen priority selection mismatch")
    if metrics.get("adaptive_post_public_selection") is not True or metrics.get("E67_E75_overlap_not_replication") is not True:
        raise AssertionError("Adaptive-selection or overlap disclosure missing")
    return {"status": "PASS", "independent_checks": checks, "origins": per_origin,
            "policies": gate_results, "selected_policy": selected,
            "test_accessed": False, "fits": 0}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, ensure_ascii=False, allow_nan=False))
