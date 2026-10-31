"""Independent audit of the frozen V3 sparse Classification full-six result."""
from __future__ import annotations

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score,
                             precision_recall_fscore_support)

from scripts import optimization_v3_sparse_classification_fullsix as run
from scripts.optimization_v3_sparse_classification import (
    CACHE as PILOT_CACHE,
    OUT as PILOT_OUT,
    check as check_pilot,
    load_train,
    read as read_json,
    sha as file_sha,
)
from scripts.research_campaign_classification_stage2 import _load_round1_control
from src.models.classification_round2_models import stream_missing_mask
from src.models.optimization_v3_sparse_classification import PARAMS
from src.models.round1_evaluation import array_digest
from src.models.round1_models import _fit_fingerprint
from src.models.regression_research_text import _array_digest

WINDOWS = {
    "W1": (55, 56, 59),
    "W2": (59, 60, 63),
    "W3": (63, 64, 67),
    "W4": (67, 68, 71),
    "W5": (71, 72, 75),
    "W6": (75, 76, 79),
}
ENDS = [55, 59, 63, 67, 71, 75]


def _controls(frame: pd.DataFrame, end: int) -> dict:
    horizon = 67 if end == 55 else 79 if end == 67 else end + 4
    c2_rows, c2b, _ = _load_round1_control(frame, end, horizon, "C2_B")
    cb_path = run.CB_CACHE / f"fit{end}.npz"
    cb_meta = read_json(cb_path.with_suffix(".json"))
    assert cb_meta["npz_sha256"] == file_sha(cb_path)
    with np.load(cb_path, allow_pickle=False) as archive:
        cb_rows = archive["rows"].astype(np.int64, copy=True)
        cb = archive["candidate"].astype(float, copy=True)
    np.testing.assert_array_equal(cb_rows, c2_rows)
    assert array_digest(cb) == cb_meta["prediction_sha256"]
    cycles = pd.to_numeric(frame.cycle_code, errors="raise").to_numpy(dtype=int)
    expected = np.flatnonzero((cycles > end) & (cycles <= horizon))
    np.testing.assert_array_equal(c2_rows, expected)
    missing = stream_missing_mask(frame.iloc[c2_rows])
    champion = np.where(missing, cb, c2b)
    return {"rows": c2_rows, "c2b": c2b, "cb": cb,
            "missing": missing, "champion": champion, "horizon": horizon}


def _pilot(record: dict, end: int) -> dict:
    path = PILOT_CACHE / f"fit{end}.npz"
    meta = read_json(path.with_suffix(".json"))
    assert meta["npz_sha256"] == file_sha(path)
    assert meta["freeze_signature"] == check_pilot()["signature"]
    with np.load(path, allow_pickle=False) as archive:
        rows = archive["rows"].astype(np.int64, copy=True)
        raw = archive["raw_candidate"].astype(float, copy=True)
        blend = archive["blend25"].astype(float, copy=True)
        champion = archive["champion"].astype(float, copy=True)
    np.testing.assert_array_equal(rows, record["rows"])
    np.testing.assert_array_equal(champion, record["champion"])
    expected = np.where(record["missing"], record["cb"], 0.75 * record["c2b"] + 0.25 * raw)
    np.testing.assert_array_equal(blend, expected)
    assert np.isfinite(raw).all() and np.all((raw >= 0) & (raw <= 1))
    return {**record, "raw": raw, "blend": blend}


def _newfit(record: dict, signature: str, train_frame: pd.DataFrame,
            target: np.ndarray) -> dict:
    end = next(end for end in run.NEW_ENDS if end == record["end"])
    path = run.CACHE / f"fit{end}.npz"
    meta = read_json(path.with_suffix(".json"))
    assert meta["freeze_signature"] == signature
    assert meta["npz_sha256"] == file_sha(path)
    with np.load(path, allow_pickle=False) as archive:
        rows = archive["rows"].astype(np.int64, copy=True)
        raw = archive["raw_candidate"].astype(float, copy=True)
        blend = archive["blend25"].astype(float, copy=True)
        champion = archive["champion"].astype(float, copy=True)
    np.testing.assert_array_equal(rows, record["rows"])
    np.testing.assert_array_equal(champion, record["champion"])
    assert len(raw) == len(rows) and np.isfinite(raw).all()
    assert np.all((raw >= 0) & (raw <= 1))
    expected_blend = np.where(record["missing"], record["cb"],
                              0.75 * record["c2b"] + 0.25 * raw)
    np.testing.assert_array_equal(blend, expected_blend)
    fit = pd.to_numeric(train_frame.cycle_code, errors="raise").between(47, end).to_numpy()
    state = meta["state"]
    assert meta["fit_rows"] == int(fit.sum()) == state["fit_rows"]
    fields = state["input_feature_names"]
    assert state["fit_data_sha256"] == _fit_fingerprint(
        train_frame.loc[fit, fields], tuple(fields), target[fit]
    )
    assert state["target_sha256"] == _array_digest(target[fit])
    assert state["parameters"] == PARAMS
    assert state["eval_set_used"] is False and state["early_stopping_used"] is False
    assert meta["outer_labels_used_for_fit"] is False
    assert meta["raw_prediction_sha256"] == array_digest(raw)
    assert meta["blend_prediction_sha256"] == array_digest(blend)
    assert meta["state"] == run.read_json(path.with_suffix(".json"))["state"]
    return {**record, "raw": raw, "blend": blend, "state": state}


def _metric(actual: np.ndarray, p: np.ndarray) -> dict:
    actual = np.asarray(actual, dtype=np.int8)
    p = np.asarray(p, dtype=float)
    pred = p >= 0.50
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
             "recall": float(recall[c]), "f1": float(f1[c])} for c in [0, 1]
        ],
        "positive_rate": float(actual.mean()),
        "positive_prediction_rate": float(pred.mean()),
        "average_precision": float(average_precision_score(actual, p)) if len(np.unique(actual)) == 2 else None,
        "brier_score": float(brier_score_loss(actual, p)),
        "false_positive": int(cm[0, 1]),
        "false_negative": int(cm[1, 0]),
        "threshold": 0.50,
    }


def _confidence(actual: np.ndarray, p: np.ndarray) -> dict:
    confidence = np.maximum(p, 1.0 - p)
    groups = {
        "low_below_0_60": confidence < 0.60,
        "medium_0_60_to_0_80": (confidence >= 0.60) & (confidence < 0.80),
        "high_at_least_0_80": confidence >= 0.80,
    }
    result = {}
    for name, mask in groups.items():
        if not mask.any():
            result[name] = {"n": 0, "accuracy": None, "mean_confidence": None, "brier_score": None}
        else:
            result[name] = {
                "n": int(mask.sum()),
                "accuracy": float(np.mean((p[mask] >= 0.5) == actual[mask])),
                "mean_confidence": float(np.mean(confidence[mask])),
                "brier_score": float(brier_score_loss(actual[mask], p[mask])),
            }
    return result


def _view(frame: pd.DataFrame, target: np.ndarray, rows: np.ndarray,
          champion: np.ndarray, blend: np.ndarray, edges: list[float]) -> dict:
    rows = np.asarray(rows, dtype=np.int64)
    actual = target[rows]
    cycles = pd.to_numeric(frame.iloc[rows].cycle_code, errors="raise").to_numpy(dtype=int)
    missing = stream_missing_mask(frame.iloc[rows])
    text_load = pd.to_numeric(frame.iloc[rows].text_load, errors="coerce").to_numpy(dtype=float)
    a, b, c = edges
    masks = {
        "stream_missing": missing,
        "stream_present": ~missing,
        "text_load_q1_low_tail": text_load <= a,
        "text_load_q2": (text_load > a) & (text_load <= b),
        "text_load_q3": (text_load > b) & (text_load <= c),
        "text_load_q4_high_tail": text_load > c,
    }
    output = {"n": len(rows), "text_load_training_edges": edges, "models": {}, "subgroups": {}}
    for name, probability in [("champion", champion), ("blend25", blend)]:
        output["models"][name] = _metric(actual, probability)
        output["models"][name]["confidence_groups"] = _confidence(actual, probability)
        by_cycle = [
            _metric(actual[cycles == cycle], probability[cycles == cycle])["macro_f1"]
            for cycle in np.unique(cycles)
        ]
        output["models"][name]["equal_cycle_macro_f1"] = float(np.mean(by_cycle))
        output["models"][name]["cycle_count"] = len(by_cycle)
    output["delta"] = {
        field: output["models"]["blend25"][field] - output["models"]["champion"][field]
        for field in ["macro_f1", "equal_cycle_macro_f1", "average_precision", "brier_score"]
    }
    for name, mask in masks.items():
        if not mask.any():
            output["subgroups"][name] = {
                "n": 0, "champion": None, "blend25": None, "delta_macro_f1": None
            }
            continue
        baseline = _metric(actual[mask], champion[mask])
        challenger = _metric(actual[mask], blend[mask])
        output["subgroups"][name] = {
            "n": int(mask.sum()), "champion": baseline, "blend25": challenger,
            "delta_macro_f1": challenger["macro_f1"] - baseline["macro_f1"],
        }
    return output


def _assert_nested(actual, expected, path="result") -> int:
    checks = 0
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), path
        for key in expected:
            checks += _assert_nested(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), path
        for index, (got, want) in enumerate(zip(actual, expected)):
            checks += _assert_nested(got, want, f"{path}[{index}]")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        assert isinstance(actual, (int, float)) and np.isclose(actual, expected, atol=1e-12, rtol=0), path
        checks += 1
    else:
        assert actual == expected, path
        checks += 1
    return checks


def verify() -> dict:
    frozen = run.check_freeze()
    pilot_freeze = check_pilot()
    pilot_receipt = read_json(PILOT_OUT / "verification.json")
    assert pilot_receipt["status"] == "PASS" and pilot_receipt["newfits"] == 0
    frame, target, train_path = load_train(None)
    assert len(frame) == len(target) == 50608
    cycles = pd.to_numeric(frame.cycle_code, errors="raise").to_numpy(dtype=int)
    records = {}
    for end in ENDS:
        horizon = 67 if end == 55 else 79 if end == 67 else end + 4
        rows, c2b, _ = _load_round1_control(frame, end, horizon, "C2_B")
        c2_info = _controls(frame, end)
        np.testing.assert_array_equal(rows, c2_info["rows"])
        records[end] = c2_info
        if end in (55, 67):
            records[end] = _pilot(records[end], end)
        else:
            records[end] = _newfit(records[end], frozen["signature"], frame, target)
        missing = stream_missing_mask(frame.iloc[rows])
        np.testing.assert_array_equal(records[end]["missing"], missing)
        np.testing.assert_array_equal(records[end]["blend"][missing], records[end]["champion"][missing])
        assert np.all((records[end]["blend"] >= 0) & (records[end]["blend"] <= 1))
    edges = np.quantile(
        pd.to_numeric(frame.loc[(cycles >= 47) & (cycles <= 55), "text_load"], errors="coerce")
        .dropna().to_numpy(dtype=float), [0.25, 0.5, 0.75]
    ).astype(float).tolist()
    views = {}
    window_parts = []
    for name, (end, lo, hi) in WINDOWS.items():
        record = records[end]
        mask = (cycles[record["rows"]] >= lo) & (cycles[record["rows"]] <= hi)
        rows = record["rows"][mask]
        champion = record["champion"][mask]
        blend = record["blend"][mask]
        views[name] = {"fit_cutoff": end, "cycles": [lo, hi],
                       **_view(frame, target, rows, champion, blend, edges)}
        window_parts.append((rows, champion, blend))
    pooled_rows = np.concatenate([item[0] for item in window_parts])
    assert len(np.unique(pooled_rows)) == len(pooled_rows)
    pooled = _view(frame, target, pooled_rows,
                   np.concatenate([item[1] for item in window_parts]),
                   np.concatenate([item[2] for item in window_parts]), edges)
    hparts = []
    for end, lo, hi in [(55, 56, 67), (67, 68, 79)]:
        record = records[end]
        mask = (cycles[record["rows"]] >= lo) & (cycles[record["rows"]] <= hi)
        hparts.append((record["rows"][mask], record["champion"][mask], record["blend"][mask]))
    hrows = np.concatenate([item[0] for item in hparts])
    assert len(np.unique(hrows)) == len(hrows)
    h12 = _view(frame, target, hrows, np.concatenate([item[1] for item in hparts]),
                np.concatenate([item[2] for item in hparts]), edges)
    computed = {
        "windows": views,
        "primary_six_window_pooled": pooled,
        "H12_fixed_origin_pooled": h12,
        "H12_early": _view(frame, target, *hparts[0], edges),
        "H12_late": _view(frame, target, *hparts[1], edges),
    }
    results = read_json(run.OUT / "fullsix.json")
    assert results["freeze_signature"] == frozen["signature"]
    assert results["status"] == "READY_FOR_INDEPENDENT_VERIFICATION"
    assert results["new_fits"] == 4 and results["new_fit_cutoffs"] == [59, 63, 71, 75]
    assert results["reused_cutoffs"] == [55, 67]
    assert results["test_read"] is False and results["kaggle_access"] is False
    assert results["threshold"] == 0.50 and results["blend_weight_new_present"] == 0.25
    metric_checks = _assert_nested(results["results"], computed)
    deltas = [views[name]["delta"]["macro_f1"] for name in WINDOWS]
    primary_gain = pooled["delta"]["macro_f1"]
    h12_gain = h12["delta"]["macro_f1"]
    gates = {
        "primary_gain_at_least_0_0005": primary_gain >= 0.0005,
        "h12_harm_at_most_0_001": h12_gain >= -0.001,
        "at_least_four_of_six_windows_nonnegative": sum(delta >= 0 for delta in deltas) >= 4,
        "primary_gain": float(primary_gain),
        "h12_gain": float(h12_gain),
        "nonnegative_windows": int(sum(delta >= 0 for delta in deltas)),
    }
    assert results["parent_external_gates"] == gates
    verification = {
        "status": "PASS",
        "freeze_signature": frozen["signature"],
        "source_and_input_hashes_verified": True,
        "root_pilot_status": pilot_receipt["status"],
        "root_pilot_signature": pilot_freeze["signature"],
        "newfits": 0,
        "test_read": False,
        "kaggle_access": False,
        "four_new_fit_audits_verified": True,
        "six_window_route_and_label_alignment_verified": True,
        "missing_branch_unchanged": True,
        "independent_metric_field_checks": metric_checks,
        "parent_external_gates": gates,
        "primary_window_deltas": {name: float(views[name]["delta"]["macro_f1"]) for name in WINDOWS},
        "primary_n": int(len(pooled_rows)),
        "h12_n": int(len(hrows)),
    }
    run.save_json_new(run.OUT / "verification.json", verification)
    print(json.dumps(verification, allow_nan=False, indent=2))
    return verification


if __name__ == "__main__":
    verify()
