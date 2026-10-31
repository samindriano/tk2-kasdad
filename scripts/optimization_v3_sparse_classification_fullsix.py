"""Frozen four-fit continuation of the V3 sparse-LGB classification pilot."""
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
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from threadpoolctl import threadpool_limits

from scripts.optimization_v3_sparse_classification import load_train
from scripts.research_campaign_classification_stage2 import (
    PRIMARY_WINDOWS, _load_round1_control,
)
from src.models.classification_round2_models import stream_missing_mask
from src.models.optimization_v3_sparse_classification import SparseTextBoostClassifier
from src.models.round1_evaluation import array_digest

CONFIG = ROOT / "experiments/optimization_v3/classification_sparse_full"
PILOT_CONFIG = ROOT / "experiments/optimization_v3/classification_sparse"
PILOT_REPORT = ROOT / "reports/optimization_v3/classification_sparse"
PILOT_CACHE = ROOT / "experiments/runs/optimization_v3_classification_sparse"
OUT = ROOT / "reports/optimization_v3/classification_sparse_full"
CACHE = ROOT / "experiments/runs/optimization_v3_classification_sparse_full"
ROUND1_CACHE = ROOT / "experiments/runs/classification_round1_v1"
ROUND1_AUDITS = ROOT / "reports/classification_round1/fit_audits"
CB_CACHE = ROOT / "experiments/runs/dual_cls_catboost_v1"
ENDS = [55, 59, 63, 67, 71, 75]
NEW_ENDS = [59, 63, 71, 75]
THRESHOLD = 0.50
BLEND_WEIGHT = 0.25
CODE_FILES = [
    "src/models/optimization_v3_sparse_classification.py",
    "scripts/optimization_v3_sparse_classification.py",
    "scripts/optimization_v3_verify_sparse_classification.py",
    "scripts/optimization_v3_sparse_classification_fullsix.py",
    "scripts/optimization_v3_verify_sparse_classification_fullsix.py",
    "tests/test_optimization_v3_sparse_classification.py",
    "tests/test_optimization_v3_sparse_full.py",
    "scripts/research_campaign_classification_stage2.py",
    "scripts/optimization_v2_classification.py",
    "src/models/round1_evaluation.py",
    "src/models/round1_models.py",
    "src/models/classification_models.py",
    "src/models/classification_round2_models.py",
    "src/models/regression_research_text.py",
]


def file_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json_new(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def source_head() -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def input_paths(train_path: Path) -> list[Path]:
    paths = [
        train_path,
        PILOT_CONFIG / "protocol.json",
        PILOT_CONFIG / "freeze.json",
        PILOT_REPORT / "pilot.json",
        PILOT_REPORT / "verification.json",
    ]
    for end in (55, 67):
        paths.extend([
            PILOT_CACHE / f"fit{end}.npz",
            PILOT_CACHE / f"fit{end}.json",
        ])
    for end in ENDS:
        paths.extend([
            ROUND1_CACHE / f"fit{end}__C2_B.npz",
            ROUND1_AUDITS / f"fit{end}.json",
            CB_CACHE / f"fit{end}.npz",
            CB_CACHE / f"fit{end}.json",
        ])
    return paths


def make_identity() -> dict:
    _, _, train_path = load_train(None)
    if not train_path.is_absolute():
        train_path = ROOT / train_path
    paths = input_paths(train_path)
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Frozen continuation inputs missing: {missing}")
    pilot_verification = read_json(PILOT_REPORT / "verification.json")
    if pilot_verification.get("status") != "PASS":
        raise AssertionError("Root sparse pilot must independently verify PASS")
    return {
        "protocol_sha256": file_sha(CONFIG / "protocol.json"),
        "code_sha256": {p: file_sha(ROOT / p) for p in CODE_FILES},
        "inputs_sha256": {
            p.resolve().relative_to(ROOT).as_posix(): file_sha(p) for p in paths
        },
        "packages": {
            package: importlib.metadata.version(package)
            for package in ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "threadpoolctl"]
        },
        "test_read": False,
        "kaggle_access": False,
    }


def freeze() -> dict:
    path = CONFIG / "freeze.json"
    current = make_identity()
    if path.exists():
        frozen = read_json(path)
        if frozen.get("identity") != current:
            raise AssertionError("Continuation freeze identity changed")
        return frozen
    if CACHE.exists():
        raise AssertionError("Continuation prediction cache already exists before freeze")
    identity = current
    signature = hashlib.sha256(json.dumps(
        identity, sort_keys=True, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    frozen = {
        "status": "FROZEN_BEFORE_FOUR_NEW_FITS",
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "source_head_contextual": source_head(),
        "identity": identity,
        "signature": signature,
    }
    save_json_new(path, frozen)
    return frozen


def check_freeze() -> dict:
    frozen = read_json(CONFIG / "freeze.json")
    current = make_identity()
    if frozen.get("identity") != current:
        raise AssertionError("Continuation code/data identity differs from frozen receipt")
    expected = hashlib.sha256(json.dumps(
        current, sort_keys=True, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    if frozen.get("signature") != expected:
        raise AssertionError("Continuation freeze signature mismatch")
    return frozen


def control_record(frame: pd.DataFrame, y: np.ndarray, end: int) -> dict:
    horizon = 67 if end == 55 else 79 if end == 67 else end + 4
    rows, c2b, c2_identity = _load_round1_control(frame, end, horizon, "C2_B")
    cb_path = CB_CACHE / f"fit{end}.npz"
    cb_meta_path = cb_path.with_suffix(".json")
    cb_meta = read_json(cb_meta_path)
    if cb_meta.get("npz_sha256") != file_sha(cb_path):
        raise AssertionError(f"CatBoost cache checksum mismatch at cutoff {end}")
    with np.load(cb_path, allow_pickle=False) as archive:
        cb_rows = archive["rows"].astype(np.int64, copy=True)
        cb_probability = archive["candidate"].astype(float, copy=True)
    if not np.array_equal(cb_rows, rows):
        raise AssertionError(f"Champion branch row alignment mismatch at cutoff {end}")
    if cb_probability.shape != (len(rows),) or not np.isfinite(cb_probability).all():
        raise AssertionError(f"Invalid CatBoost probability at cutoff {end}")
    if np.any((cb_probability < 0) | (cb_probability > 1)):
        raise AssertionError(f"CatBoost probability out of range at cutoff {end}")
    if array_digest(cb_probability) != cb_meta.get("prediction_sha256"):
        raise AssertionError(f"CatBoost prediction checksum mismatch at cutoff {end}")
    missing = stream_missing_mask(frame.iloc[rows])
    champion = np.where(missing, cb_probability, c2b)
    fit_mask = pd.to_numeric(frame.cycle_code, errors="raise").between(47, end).to_numpy()
    expected_rows = np.flatnonzero(
        (pd.to_numeric(frame.cycle_code, errors="raise").to_numpy(dtype=int) > end)
        & (pd.to_numeric(frame.cycle_code, errors="raise").to_numpy(dtype=int) <= horizon)
    )
    if not np.array_equal(rows, expected_rows):
        raise AssertionError(f"Validation horizon mismatch at cutoff {end}")
    return {
        "end": end,
        "rows": rows,
        "c2b": c2b,
        "cb_missing": cb_probability,
        "champion": champion,
        "missing": missing,
        "fit_mask": fit_mask,
        "horizon": horizon,
        "c2b_identity": c2_identity,
        "cb_meta": cb_meta,
    }


def pilot_record(frame: pd.DataFrame, record: dict) -> dict:
    end = record["end"]
    path = PILOT_CACHE / f"fit{end}.npz"
    meta = read_json(path.with_suffix(".json"))
    if meta.get("npz_sha256") != file_sha(path):
        raise AssertionError(f"Sparse pilot prediction checksum mismatch at {end}")
    with np.load(path, allow_pickle=False) as archive:
        pilot_rows = archive["rows"].astype(np.int64, copy=True)
        raw = archive["raw_candidate"].astype(float, copy=True)
        blend = archive["blend25"].astype(float, copy=True)
        pilot_champion = archive["champion"].astype(float, copy=True)
    if not np.array_equal(pilot_rows, record["rows"]):
        raise AssertionError(f"Pilot rows differ from champion rows at {end}")
    np.testing.assert_array_equal(pilot_champion, record["champion"])
    expected = np.where(record["missing"], record["cb_missing"],
                        0.75 * record["c2b"] + BLEND_WEIGHT * raw)
    np.testing.assert_array_equal(blend, expected)
    return {**record, "raw_candidate": raw, "blend25": blend, "pilot_cache_sha256": file_sha(path)}


def load_existing_prediction(frame: pd.DataFrame, record: dict, signature: str) -> dict | None:
    path = CACHE / f"fit{record['end']}.npz"
    meta_path = path.with_suffix(".json")
    if not path.exists() and not meta_path.exists():
        return None
    if not (path.is_file() and meta_path.is_file()):
        raise AssertionError(f"Partial prediction cache at cutoff {record['end']}")
    meta = read_json(meta_path)
    if meta.get("freeze_signature") != signature or meta.get("npz_sha256") != file_sha(path):
        raise AssertionError(f"Cached prediction receipt mismatch at {record['end']}")
    with np.load(path, allow_pickle=False) as archive:
        rows = archive["rows"].astype(np.int64, copy=True)
        raw = archive["raw_candidate"].astype(float, copy=True)
        blend = archive["blend25"].astype(float, copy=True)
        champion = archive["champion"].astype(float, copy=True)
    if not np.array_equal(rows, record["rows"]):
        raise AssertionError(f"Cached row order mismatch at {record['end']}")
    np.testing.assert_array_equal(champion, record["champion"])
    np.testing.assert_array_equal(blend, np.where(
        record["missing"], record["cb_missing"], 0.75 * record["c2b"] + BLEND_WEIGHT * raw
    ))
    return {**record, "raw_candidate": raw, "blend25": blend,
            "newfit_state": meta["state"], "newfit_seconds": meta["seconds"]}


def fit_new(frame: pd.DataFrame, y: np.ndarray, record: dict, signature: str) -> dict:
    end = record["end"]
    path = CACHE / f"fit{end}.npz"
    meta_path = path.with_suffix(".json")
    if path.exists() or meta_path.exists():
        cached = load_existing_prediction(frame, record, signature)
        if cached is None:
            raise AssertionError("Internal cache lookup failure")
        return cached
    fit = record["fit_mask"]
    model = SparseTextBoostClassifier()
    fields = list(model.input_feature_names)
    start = time.perf_counter()
    with threadpool_limits(limits=2):
        model.fit(frame.loc[fit, fields], y[fit])
        state = model.audit_state()
        raw = model.predict_proba(frame.iloc[record["rows"]][fields])
    if state != model.audit_state():
        raise AssertionError(f"Sparse classifier state changed during prediction at {end}")
    np.testing.assert_array_equal(record["missing"], stream_missing_mask(frame.iloc[record["rows"]]))
    blend = np.where(record["missing"], record["cb_missing"],
                     0.75 * record["c2b"] + BLEND_WEIGHT * raw)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, rows=record["rows"], raw_candidate=raw,
                        champion=record["champion"], blend25=blend)
    seconds = time.perf_counter() - start
    meta = {
        "freeze_signature": signature,
        "cutoff": end,
        "horizon": record["horizon"],
        "fit_rows": int(fit.sum()),
        "fit_rows_sha256": state["fit_data_sha256"],
        "state": state,
        "npz_sha256": file_sha(path),
        "raw_prediction_sha256": array_digest(raw),
        "blend_prediction_sha256": array_digest(blend),
        "seconds": seconds,
        "eval_set_used": False,
        "early_stopping_used": False,
        "outer_labels_used_for_fit": False,
    }
    save_json_new(meta_path, meta)
    print(f"FITTED cutoff={end} rows={len(record['rows'])} seconds={seconds:.1f}", flush=True)
    return {**record, "raw_candidate": raw, "blend25": blend,
            "newfit_state": state, "newfit_seconds": seconds}


def q_edges(frame: pd.DataFrame) -> list[float]:
    cycles = pd.to_numeric(frame.cycle_code, errors="raise").to_numpy(dtype=int)
    calibration = frame.loc[(cycles >= 47) & (cycles <= 55), "text_load"]
    values = pd.to_numeric(calibration, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    edges = np.quantile(values, [0.25, 0.50, 0.75]).astype(float).tolist()
    if not (edges[0] < edges[1] < edges[2]):
        raise AssertionError("Training-period text_load quartiles are not strictly ordered")
    return edges


def metric(y: np.ndarray, probability: np.ndarray) -> dict:
    actual = np.asarray(y, dtype=np.int8)
    p = np.asarray(probability, dtype=float)
    if actual.ndim != 1 or p.shape != actual.shape or len(actual) == 0:
        raise ValueError("Metric vectors must be aligned and nonempty")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Probabilities must be finite and in [0,1]")
    pred = p >= THRESHOLD
    cm = confusion_matrix(actual, pred, labels=[0, 1])
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, pred, labels=[0, 1], zero_division=0
    )
    return {
        "n": int(len(actual)),
        "macro_f1": float(f1_score(actual, pred, average="macro", zero_division=0)),
        "confusion_matrix": cm.astype(int).tolist(),
        "per_class": [
            {"label": c, "support": int(support[c]), "precision": float(precision[c]),
             "recall": float(recall[c]), "f1": float(f1[c])} for c in (0, 1)
        ],
        "positive_rate": float(actual.mean()),
        "positive_prediction_rate": float(pred.mean()),
        "average_precision": float(average_precision_score(actual, p)) if len(np.unique(actual)) == 2 else None,
        "brier_score": float(brier_score_loss(actual, p)),
        "false_positive": int(cm[0, 1]),
        "false_negative": int(cm[1, 0]),
        "threshold": THRESHOLD,
    }


def confidence_groups(y: np.ndarray, probability: np.ndarray) -> dict:
    actual = np.asarray(y, dtype=np.int8)
    p = np.asarray(probability, dtype=float)
    confidence = np.maximum(p, 1.0 - p)
    masks = {
        "low_below_0_60": confidence < 0.60,
        "medium_0_60_to_0_80": (confidence >= 0.60) & (confidence < 0.80),
        "high_at_least_0_80": confidence >= 0.80,
    }
    result = {}
    for name, mask in masks.items():
        if not mask.any():
            result[name] = {"n": 0, "accuracy": None, "mean_confidence": None, "brier_score": None}
            continue
        prediction = p[mask] >= THRESHOLD
        result[name] = {
            "n": int(mask.sum()),
            "accuracy": float(np.mean(prediction == actual[mask])),
            "mean_confidence": float(np.mean(confidence[mask])),
            "brier_score": float(brier_score_loss(actual[mask], p[mask])),
        }
    return result


def evaluate_rows(frame: pd.DataFrame, y: np.ndarray, rows: np.ndarray,
                  champion: np.ndarray, blend: np.ndarray,
                  text_edges: list[float]) -> dict:
    rows = np.asarray(rows, dtype=np.int64)
    missing = stream_missing_mask(frame.iloc[rows])
    text = pd.to_numeric(frame.iloc[rows].text_load, errors="coerce").to_numpy(dtype=float)
    e1, e2, e3 = text_edges
    group_masks = {
        "stream_missing": missing,
        "stream_present": ~missing,
        "text_load_q1_low_tail": text <= e1,
        "text_load_q2": (text > e1) & (text <= e2),
        "text_load_q3": (text > e2) & (text <= e3),
        "text_load_q4_high_tail": text > e3,
    }
    result = {"n": int(len(rows)), "text_load_training_edges": text_edges,
              "models": {}, "subgroups": {}}
    for name, probability in [("champion", champion), ("blend25", blend)]:
        result["models"][name] = metric(y[rows], probability)
        result["models"][name]["confidence_groups"] = confidence_groups(y[rows], probability)
        cycle = pd.to_numeric(frame.iloc[rows].cycle_code, errors="raise").to_numpy(dtype=int)
        cycle_scores = [metric(y[rows][cycle == c], probability[cycle == c])["macro_f1"]
                        for c in sorted(np.unique(cycle))]
        result["models"][name]["equal_cycle_macro_f1"] = float(np.mean(cycle_scores))
        result["models"][name]["cycle_count"] = int(len(cycle_scores))
    result["delta"] = {
        key: result["models"]["blend25"][key] - result["models"]["champion"][key]
        for key in ["macro_f1", "equal_cycle_macro_f1", "average_precision", "brier_score"]
    }
    for group, mask in group_masks.items():
        if not mask.any():
            result["subgroups"][group] = {"n": 0, "champion": None, "blend25": None, "delta_macro_f1": None}
            continue
        c = metric(y[rows][mask], champion[mask])
        b = metric(y[rows][mask], blend[mask])
        result["subgroups"][group] = {
            "n": int(mask.sum()), "champion": c, "blend25": b,
            "delta_macro_f1": b["macro_f1"] - c["macro_f1"],
        }
    return result


def aggregate(frame: pd.DataFrame, y: np.ndarray, records: dict[int, dict]) -> dict:
    cycles = pd.to_numeric(frame.cycle_code, errors="raise").to_numpy(dtype=int)
    edges = q_edges(frame)
    windows = {}
    primary_rows, primary_champion, primary_blend = [], [], []
    for name, (end, start_cycle, stop_cycle) in PRIMARY_WINDOWS.items():
        record = records[end]
        cycle = cycles[record["rows"]]
        mask = (cycle >= start_cycle) & (cycle <= stop_cycle)
        rows = record["rows"][mask]
        champion = record["champion"][mask]
        blend = record["blend25"][mask]
        windows[name] = {"fit_cutoff": end, "cycles": [start_cycle, stop_cycle],
                         **evaluate_rows(frame, y, rows, champion, blend, edges)}
        primary_rows.append(rows)
        primary_champion.append(champion)
        primary_blend.append(blend)
    primary_rows = np.concatenate(primary_rows)
    if len(np.unique(primary_rows)) != len(primary_rows):
        raise AssertionError("Primary windows overlap")
    primary = evaluate_rows(frame, y, primary_rows, np.concatenate(primary_champion),
                            np.concatenate(primary_blend), edges)
    early = records[55]
    late = records[67]
    h12_parts = []
    for record, lo, hi in [(early, 56, 67), (late, 68, 79)]:
        mask = (cycles[record["rows"]] >= lo) & (cycles[record["rows"]] <= hi)
        h12_parts.append((record["rows"][mask], record["champion"][mask], record["blend25"][mask]))
    h12_rows = np.concatenate([part[0] for part in h12_parts])
    if len(np.unique(h12_rows)) != len(h12_rows):
        raise AssertionError("H12 fixed-origin parts overlap")
    h12 = evaluate_rows(frame, y, h12_rows,
                        np.concatenate([part[1] for part in h12_parts]),
                        np.concatenate([part[2] for part in h12_parts]), edges)
    h12_early = evaluate_rows(frame, y, *h12_parts[0], edges)
    h12_late = evaluate_rows(frame, y, *h12_parts[1], edges)
    return {"windows": windows, "primary_six_window_pooled": primary,
            "H12_fixed_origin_pooled": h12, "H12_early": h12_early, "H12_late": h12_late}


def run() -> dict:
    frozen = check_freeze()
    frame, y, _ = load_train(None)
    if len(y) != len(frame) or len(frame) != 50608:
        raise AssertionError("Expected official 50,608-row Classification Train")
    cycles = pd.to_numeric(frame.cycle_code, errors="raise").to_numpy(dtype=int)
    records = {end: control_record(frame, y, end) for end in ENDS}
    for end in (55, 67):
        records[end] = pilot_record(frame, records[end])
    for end in NEW_ENDS:
        records[end] = fit_new(frame, y, records[end], frozen["signature"])
    metrics = aggregate(frame, y, records)
    deltas = [metrics["windows"][name]["delta"]["macro_f1"] for name in PRIMARY_WINDOWS]
    primary_gain = metrics["primary_six_window_pooled"]["delta"]["macro_f1"]
    h12_gain = metrics["H12_fixed_origin_pooled"]["delta"]["macro_f1"]
    gates = {
        "primary_gain_at_least_0_0005": bool(primary_gain >= 0.0005),
        "h12_harm_at_most_0_001": bool(h12_gain >= -0.001),
        "at_least_four_of_six_windows_nonnegative": bool(sum(d >= 0 for d in deltas) >= 4),
        "primary_gain": float(primary_gain),
        "h12_gain": float(h12_gain),
        "nonnegative_windows": int(sum(d >= 0 for d in deltas)),
    }
    result = {
        "status": "READY_FOR_INDEPENDENT_VERIFICATION",
        "freeze_signature": frozen["signature"],
        "source_head_contextual": frozen["source_head_contextual"],
        "test_read": False,
        "kaggle_access": False,
        "new_fits": len(NEW_ENDS),
        "new_fit_cutoffs": NEW_ENDS,
        "reused_cutoffs": [55, 67],
        "threshold": THRESHOLD,
        "blend_weight_new_present": BLEND_WEIGHT,
        "missing_branch_unchanged": True,
        "window_fit_states": {
            str(e): (records[e]["newfit_state"] if e in NEW_ENDS
                     else read_json(PILOT_CACHE / f"fit{e}.json")["state"])
            for e in ENDS
        },
        "window_fit_seconds": {str(e): records[e].get("newfit_seconds") for e in NEW_ENDS},
        "results": metrics,
        "parent_external_gates": gates,
        "interpretation_limits": [
            "The six windows and H12 use overlapping temporal validation data and are development evidence, not a fresh untouched holdout.",
            "No threshold or blend weight was selected using these six-window results.",
            "text_load quartile edges are computed from feature values in the earliest fit period (cycles 47-55), without labels.",
            "All missing-stream predictions retain the frozen CatBoost missing expert exactly.",
        ],
    }
    save_json_new(OUT / "fullsix.json", result)
    write_report(result)
    print("FULLSIX", json.dumps({
        "primary_delta": primary_gain, "H12_delta": h12_gain,
        "window_deltas": deltas, "gates": gates,
    }, allow_nan=False), flush=True)
    return result


def write_report(result: dict) -> None:
    p = result["results"]
    lines = [
        "# V3 sparse-LGB Classification continuation: six temporal windows", "",
        "## Frozen comparison", "",
        "The continuation applies the previously fixed 25% sparse-LGB / 75% C2_B probability blend only when `signal_stream` is present. The existing CatBoost missing-stream expert remains unchanged. Four new fits (cutoffs 59, 63, 71, 75) complete the six four-cycle windows; predictions from the already verified 55/67 pilot are reused.", "",
        "The model, threshold (0.50), feature construction, training horizon, blend weight and evaluation windows were frozen before the four fits. The full Train file and cached controls/pilot inputs are hash-bound in `experiments/optimization_v3/classification_sparse_full/freeze.json`. No Test or Kaggle access occurred.", "",
        "## Primary window results", "",
        "| Window | Rows | Champion Macro F1 | Fixed blend Macro F1 | Delta | Equal-cycle delta | AP delta | Brier delta |", "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, view in p["windows"].items():
        m = view["models"]
        d = view["delta"]
        lines.append(f"| {name} | {view['n']} | {m['champion']['macro_f1']:.6f} | {m['blend25']['macro_f1']:.6f} | {d['macro_f1']:+.6f} | {d['equal_cycle_macro_f1']:+.6f} | {d['average_precision']:+.6f} | {d['brier_score']:+.6f} |")
    lines += ["", "## Pooled validation and H12", "",
              "| View | Rows | Champion Macro F1 | Fixed blend Macro F1 | Delta | Equal-cycle delta | AP delta | Brier delta |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for label, view in [("Six-window primary", p["primary_six_window_pooled"]),
                        ("H12 fixed-origin pooled", p["H12_fixed_origin_pooled"]),
                        ("H12 early (56-67)", p["H12_early"]), ("H12 late (68-79)", p["H12_late"])]:
        m, d = view["models"], view["delta"]
        lines.append(f"| {label} | {view['n']} | {m['champion']['macro_f1']:.6f} | {m['blend25']['macro_f1']:.6f} | {d['macro_f1']:+.6f} | {d['equal_cycle_macro_f1']:+.6f} | {d['average_precision']:+.6f} | {d['brier_score']:+.6f} |")
    gate = result["parent_external_gates"]
    lines += ["", "## Routing and subgroup checks", "",
              "The independent verifier checks that present-stream rows receive the fixed blend and missing-stream rows preserve the champion probability exactly. Results include class-wise precision/recall/F1, confusion matrices, stream-present/missing splits, and text_load quartiles. Quartile edges are computed from cycles 47-55 feature values only.", "",
              f"- Missing branch invariant: **{result['missing_branch_unchanged']}**.",
              f"- Parent primary-gain gate (>= 0.0005): **{gate['primary_gain_at_least_0_0005']}** ({gate['primary_gain']:+.6f}).",
              f"- Parent H12-harm gate (>= -0.001): **{gate['h12_harm_at_most_0_001']}** ({gate['h12_gain']:+.6f}).",
              f"- Parent window-support gate (>= 4/6 nonnegative): **{gate['at_least_four_of_six_windows_nonnegative']}** ({gate['nonnegative_windows']}/6).",
              "", "The full subgroup and per-class details are in `fullsix.json`; the independently recomputed receipt is `verification.json`.", "",
              "## Limits", "",
              "These Train-cycle windows have already informed model development, and H12 overlaps the pooled primary cycle range while using fixed-origin predictions. The results are comparative evidence for parent review, not a fresh holdout estimate and not authorization to change the official model or submit to Kaggle.", ""]
    path = OUT / "FULLSIX.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.freeze == args.run:
        parser.error("choose exactly one of --freeze or --run")
    if args.freeze:
        frozen = freeze()
        print(frozen["signature"])
    else:
        run()
