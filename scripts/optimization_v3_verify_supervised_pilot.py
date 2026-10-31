"""Independent verification for the fixed four-fit supervised-token pilot."""
from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, brier_score_loss

from scripts import optimization_v3_supervised_pilot as pilot
from scripts.optimization_v3_supervised_pilot import (
    CONFIG, OUT, CACHE, ENDPOINTS, VIEW_SPECS, HYBRID_FEATURES,
    classification_fit_mask, load_official_data, get_class_controls,
    get_regression_views,
)
from src.models.classification_round2_models import stream_missing_mask
from src.models.round1_models import _fit_fingerprint


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _close(a, b, *, atol: float = 1e-10) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) is bool(b)
    if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
        return int(a) == int(b)
    if isinstance(a, (float, np.floating)) and isinstance(b, (float, np.floating)):
        return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=atol)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_close(x, y, atol=atol) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_close(a[k], b[k], atol=atol) for k in a)
    return a == b


def assert_tree(actual, expected, label: str, atol: float = 1e-10) -> None:
    if not _close(actual, expected, atol=atol):
        raise AssertionError(f"Independent verification mismatch at {label}")


def _text_tokens(text, analyzer) -> set[str]:
    if text is None or pd.isna(text):
        return set()
    return set(analyzer(str(text)))


def manual_ordered_token_audit(frame: pd.DataFrame, target: np.ndarray,
                               fit: np.ndarray, task: str, recorded: dict) -> dict:
    """Rebuild per-cycle encodings and token posterior state without OrderedTokenStats."""
    fit_frame = frame.loc[fit, list(HYBRID_FEATURES)]
    fit_target = np.asarray(target[fit], dtype=float)
    unit_target = fit_target if task == "classification" else fit_target / 100.0
    cycles = pd.to_numeric(fit_frame.cycle_code, errors="raise").to_numpy(dtype=int)
    analyzer = TfidfVectorizer(ngram_range=(1, 1), lowercase=True,
                               strip_accents="unicode").build_analyzer()
    docs = [_text_tokens(value, analyzer) for value in fit_frame.signal_stream.tolist()]
    counts: dict[str, int] = {}
    sums: dict[str, float] = {}
    total = 0
    target_sum = 0.0
    encoded = np.empty((len(fit_frame), 8), dtype=np.float64)
    lineage = []
    for cycle in np.sort(np.unique(cycles)):
        rows = np.flatnonzero(cycles == cycle)
        prior = target_sum / total if total else 0.5
        for row in rows:
            tokens = docs[row]
            eligible = [token for token in sorted(tokens) if counts.get(token, 0) >= 3]
            support = np.asarray([counts[token] for token in eligible], dtype=np.float64)
            values = np.asarray([(sums[token] + 30.0 * prior) / (counts[token] + 30.0)
                                 for token in eligible], dtype=np.float64)
            if values.size:
                encoded[row] = [
                    values.mean(), values.min(), values.max(), values.std(),
                    np.quantile(values, 0.75), len(eligible) / max(len(tokens), 1),
                    np.log1p(len(tokens)), np.log1p(support.mean()),
                ]
            else:
                encoded[row] = [prior, prior, prior, 0.0, prior, 0.0,
                                np.log1p(len(tokens)), 0.0]
        lineage.append({"cycle": float(cycle), "feature_rows": int(len(rows)),
                        "labels_previously_seen": int(total)})
        for row in rows:
            for token in docs[row]:
                counts[token] = counts.get(token, 0) + 1
                sums[token] = sums.get(token, 0.0) + float(unit_target[row])
        total += int(len(rows))
        target_sum += float(unit_target[rows].sum())
    token_state = recorded["ordered_tokens"]
    expected_feature_hash = hashlib.sha256(np.asarray(encoded, dtype="<f8").tobytes()).hexdigest()
    expected_dictionary_hash = hashlib.sha256(json.dumps(
        [(token, counts[token], sums[token]) for token in sorted(counts)],
        separators=(",", ":")).encode()).hexdigest()
    assert_tree(token_state["lineage"], lineage, f"{task} token lineage")
    if token_state["training_features_sha256"] != expected_feature_hash:
        raise AssertionError(f"{task} training token feature digest mismatch")
    if token_state["dictionary_sha256"] != expected_dictionary_hash:
        raise AssertionError(f"{task} token dictionary digest mismatch")
    assert_tree(token_state["fit_rows"], len(fit_frame), f"{task} token fit rows")
    assert_tree(token_state["fit_target_sum"], target_sum, f"{task} normalized token target sum")
    return {"manual_training_feature_sha256": expected_feature_hash,
            "manual_dictionary_sha256": expected_dictionary_hash,
            "lineage_cycles": len(lineage), "strict_earlier_cycle_labels_verified": True}


def class_metric_independent(y: np.ndarray, probability: np.ndarray) -> dict:
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(probability, dtype=float)
    pred = p >= 0.50
    cm = [[int(np.sum((y == actual) & (pred == chosen))) for chosen in [False, True]]
          for actual in [0, 1]]
    per_class = []
    for label in [0, 1]:
        tp = int(cm[label][label])
        fp = int(cm[1 - label][label])
        fn = int(cm[label][1 - label])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({"label": label, "support": int(np.sum(y == label)),
                          "precision": precision, "recall": recall, "f1": f1})
    return {
        "n": int(len(y)),
        "macro_f1": float(np.mean([x["f1"] for x in per_class])),
        "confusion_matrix": cm,
        "per_class": per_class,
        "positive_rate": float(y.mean()),
        "positive_prediction_rate": float(pred.mean()),
        "average_precision": float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else None,
        "brier_score": float(brier_score_loss(y, p)),
        "threshold": 0.50,
    }


def _class_view(frame, target, rows, policies) -> dict:
    missing = stream_missing_mask(frame.iloc[rows])
    group_masks = {"all": np.ones(len(rows), bool), "stream_present": ~missing,
                   "stream_missing": missing}
    metrics = {name: class_metric_independent(target[rows], pred)
               for name, pred in policies.items()}
    comparisons = {}
    for candidate in ["direct", "blend25"]:
        for control in ["initial_control", "latest_sparse25"]:
            left, right = metrics[control], metrics[candidate]
            comparisons[f"{candidate}_vs_{control}"] = {
                "macro_f1_delta": right["macro_f1"] - left["macro_f1"],
                "average_precision_delta": right["average_precision"] - left["average_precision"],
                "brier_delta_candidate_minus_control": right["brier_score"] - left["brier_score"],
            }
    return {
        "n": int(len(rows)),
        "metrics": metrics,
        "subgroups": {
            group: {"n": int(mask.sum()), "metrics": {
                name: class_metric_independent(target[rows][mask], pred[mask])
                for name, pred in policies.items()
            }} for group, mask in group_masks.items()
        },
        "comparisons": comparisons,
    }


def regression_metric_independent(target: np.ndarray, prediction: np.ndarray) -> dict:
    err = np.asarray(prediction, float) - np.asarray(target, float)
    ae = np.abs(err)
    return {
        "n": int(len(err)),
        "mae": float(ae.mean()),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "bias_prediction_minus_target": float(err.mean()),
        "median_signed_residual": float(np.median(err)),
        "p90_absolute_error": float(np.quantile(ae, 0.90)),
        "p95_absolute_error": float(np.quantile(ae, 0.95)),
        "over_rate": float(np.mean(err > 0)),
        "under_rate": float(np.mean(err < 0)),
        "tie_rate": float(np.mean(err == 0)),
    }


def _reg_view(frame, target, rows, predictions) -> dict:
    y = target[rows]
    cycles = frame.iloc[rows].cycle_code.to_numpy(dtype=int)
    entry = pd.to_numeric(frame.iloc[rows].entry_friction, errors="coerce").to_numpy(float)
    text = frame.iloc[rows].signal_stream
    missing = text.isna().to_numpy() | text.fillna("").astype(str).str.strip().eq("").to_numpy()
    groups = {"EF0": entry == 0, "EF1": entry == 1, "tail_le5": y <= 5,
              "tail_ge95": y >= 95, "stream_missing": missing, "stream_present": ~missing}
    metrics = {}
    per_cycle = {}
    for name, pred in predictions.items():
        metrics[name] = regression_metric_independent(y, pred)
        cycle_metrics = {str(int(c)): regression_metric_independent(y[cycles == c], pred[cycles == c])
                         for c in np.unique(cycles)}
        metrics[name]["equal_cycle_mae"] = float(np.mean([m["mae"] for m in cycle_metrics.values()]))
        per_cycle[name] = cycle_metrics
    champion = metrics["champion"]
    delta = {name: {
        "mae_gain_champion_minus_candidate": champion["mae"] - metrics[name]["mae"],
        "rmse_gain_champion_minus_candidate": champion["rmse"] - metrics[name]["rmse"],
        "bias_shift_candidate_minus_champion": metrics[name]["bias_prediction_minus_target"] - champion["bias_prediction_minus_target"],
        "equal_cycle_mae_gain": champion["equal_cycle_mae"] - metrics[name]["equal_cycle_mae"],
    } for name in ["raw", "rank50", "blend25"]}
    subgroups = {}
    for group, mask in groups.items():
        group_metrics = {name: regression_metric_independent(y[mask], pred[mask])
                         for name, pred in predictions.items()}
        subgroups[group] = {
            "n": int(mask.sum()), "metrics": group_metrics,
            "mae_gain_vs_champion": {name: group_metrics["champion"]["mae"] - group_metrics[name]["mae"]
                                      for name in predictions if name != "champion"},
        }
    return {"n": int(len(rows)), "metrics": metrics, "per_cycle": per_cycle,
            "delta_vs_champion": delta, "subgroups": subgroups}


def _classification_views(frame, target, caches: dict[int, dict]) -> dict:
    policies = {}
    for end, data in caches.items():
        policies[end] = {"initial_control": data["initial_control"],
                         "latest_sparse25": data["latest_control"],
                         "direct": data["direct"], "blend25": data["blend25"]}
    definitions = {
        "W1": (55, 56, 59), "W4": (67, 68, 71),
        "H12_E": (55, 56, 67), "H12_L": (67, 68, 79),
    }
    views = {}
    pieces = {"primary_pilot": [], "H12": []}
    for name, (end, low, high) in definitions.items():
        data = caches[end]
        use = (data["cycles"] >= low) & (data["cycles"] <= high)
        rows = data["rows"][use]
        current = {key: value[use] for key, value in policies[end].items()}
        views[name] = _class_view(frame, target, rows, current)
        pieces["primary_pilot" if name in {"W1", "W4"} else "H12"].append((rows, current))
    for name, parts in pieces.items():
        rows = np.concatenate([part[0] for part in parts])
        combined = {key: np.concatenate([part[1][key] for part in parts]) for key in parts[0][1]}
        views[name] = _class_view(frame, target, rows, combined)
    return views


def _regression_views(frame, target, caches: dict[int, dict]) -> dict:
    primary, h12, _ = get_regression_views(frame)
    views = {}
    parts = {"primary_pilot": [], "H12": []}
    definitions = {"W1": (55, 56, 59, primary), "W4": (67, 68, 71, primary),
                   "H12_E": (55, 56, 67, h12), "H12_L": (67, 68, 79, h12)}
    for name, (end, low, high, control) in definitions.items():
        data = caches[end]
        mask = (data["cycles"] >= low) & (data["cycles"] <= high)
        rows = data["rows"][mask]
        expected_rows = control.loc[(control.cutoff == end) & control.cycle.between(low, high), "row"].to_numpy(dtype=np.int64)
        np.testing.assert_array_equal(rows, expected_rows)
        pred = {key: data[key][mask] for key in ["champion", "raw", "rank50", "blend25"]}
        views[name] = _reg_view(frame, target, rows, pred)
        parts["primary_pilot" if name in {"W1", "W4"} else "H12"].append((rows, pred))
    for name, group_parts in parts.items():
        rows = np.concatenate([part[0] for part in group_parts])
        preds = {key: np.concatenate([part[1][key] for part in group_parts]) for key in group_parts[0][1]}
        views[name] = _reg_view(frame, target, rows, preds)
    return views


def _verify_schema(result: dict) -> None:
    required_views = {"W1", "W4", "H12_E", "H12_L", "primary_pilot", "H12"}
    if result.get("status") != "READY_FOR_INDEPENDENT_VERIFICATION":
        raise AssertionError("Unexpected pilot status")
    if result.get("new_official_fits") != 4 or result.get("test_read") is not False or result.get("kaggle_access") is not False:
        raise AssertionError("Pilot scope/fit-count schema failed")
    if result.get("outer_labels_used_for_fit") is not False:
        raise AssertionError("Outer validation labels were marked used for fit")
    if set(result.get("classification", {}).get("views", {})) != required_views:
        raise AssertionError("Classification view schema incomplete")
    if set(result.get("regression", {}).get("views", {})) != required_views:
        raise AssertionError("Regression view schema incomplete")
    if set(result.get("fit_audits", {})) != {
        f"{task}_{end}" for task in ["classification", "regression"] for end in ENDPOINTS
    }:
        raise AssertionError("Four-fit audit schema incomplete")
    for name in required_views:
        cv = result["classification"]["views"][name]
        if set(cv["metrics"]) != {"initial_control", "latest_sparse25", "direct", "blend25"}:
            raise AssertionError(f"Classification metric schema incomplete for {name}")
        rv = result["regression"]["views"][name]
        if set(rv["metrics"]) != {"champion", "raw", "rank50", "blend25"}:
            raise AssertionError(f"Regression metric schema incomplete for {name}")
    json.dumps(result, allow_nan=False)


def dry_run_schema() -> dict:
    frame, class_y, reg_y = pilot._synthetic_frame(120)
    rows = np.flatnonzero(frame.cycle_code.between(48, 52).to_numpy())
    class_probability = np.linspace(0.05, 0.95, len(rows))
    class_policies = {"initial_control": class_probability,
                      "latest_sparse25": class_probability,
                      "direct": class_probability,
                      "blend25": class_probability}
    class_view = _class_view(frame, class_y, rows, class_policies)
    raw = np.linspace(-3, 103, len(rows))
    cycles = frame.iloc[rows].cycle_code.to_numpy(dtype=int)
    champion = np.linspace(0, 100, len(rows))
    rank = np.empty(len(rows), dtype=float)
    for cycle in np.unique(cycles):
        use = cycles == cycle
        rank[use] = np.clip(0.5 * raw[use] + 0.5 * 100 * rankdata(raw[use], method="average") / use.sum(), 0, 100)
    reg_predictions = {"champion": champion, "raw": raw, "rank50": rank,
                      "blend25": np.clip(0.75 * champion + 0.25 * rank, 0, 100)}
    reg_view = _reg_view(frame, reg_y, rows, reg_predictions)
    required = {"W1", "W4", "H12_E", "H12_L", "primary_pilot", "H12"}
    payload = {
        "status": "READY_FOR_INDEPENDENT_VERIFICATION", "new_official_fits": 4,
        "test_read": False, "kaggle_access": False, "outer_labels_used_for_fit": False,
        "classification": {"views": {name: class_view for name in required}},
        "regression": {"views": {name: reg_view for name in required}},
        "fit_audits": {f"{task}_{end}": {} for task in ["classification", "regression"] for end in ENDPOINTS},
    }
    _verify_schema(payload)
    return {"status": "PASS", "official_data_used": False, "fit_count": 0,
            "all_metric_and_view_schemas_exercised": True,
            "classification_metric_fields": sorted(class_view["metrics"]["direct"]),
            "regression_metric_fields": sorted(reg_view["metrics"]["raw"])}


def verify() -> dict:
    frozen = pilot.check_freeze()
    result = json.loads((OUT / "pilot.json").read_text(encoding="utf-8"))
    _verify_schema(result)
    if result.get("freeze_signature") != frozen["signature"]:
        raise AssertionError("Pilot report freeze signature mismatch")
    class_frame, class_target, _, reg_frame, _ = load_official_data()
    reg_target = pd.to_numeric(reg_frame.threat_rank, errors="raise").to_numpy(dtype=float)
    class_controls = get_class_controls(class_frame, class_target)
    reg_primary, reg_h12, _ = get_regression_views(reg_frame)
    class_caches, reg_caches = {}, {}
    token_checks = {}
    fit_checks = {}
    for task, frame, target in [("classification", class_frame, class_target),
                                ("regression", reg_frame, reg_target)]:
        for end in ENDPOINTS:
            path = CACHE / f"{task}_fit{end}.npz"
            sidecar = path.with_suffix(".json")
            if not path.is_file() or not sidecar.is_file():
                raise FileNotFoundError(f"Missing frozen pilot fit output: {path}")
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            if meta.get("freeze_signature") != frozen["signature"] or meta.get("npz_sha256") != sha(path):
                raise AssertionError(f"{task}@{end} cache sidecar integrity mismatch")
            with np.load(path, allow_pickle=False) as archive:
                arrays = {name: archive[name].copy() for name in archive.files}
            if set(meta.get("array_sha256", {})) != (set(arrays) - {"rows", "missing"}):
                raise AssertionError(f"{task}@{end} array digest inventory mismatch")
            for name, digest in meta["array_sha256"].items():
                actual = hashlib.sha256(np.asarray(arrays[name], dtype="<f8").tobytes()).hexdigest()
                if actual != digest:
                    raise AssertionError(f"{task}@{end} {name} array digest mismatch")
            expected_fit = (classification_fit_mask(frame, end) if task == "classification"
                            else frame.cycle_code.between(47, end).to_numpy())
            if task == "classification" and stream_missing_mask(frame.loc[expected_fit]).any():
                raise AssertionError("Present-only Classification fit unexpectedly includes missing streams")
            fields = list(HYBRID_FEATURES)
            expected_fp = _fit_fingerprint(frame.loc[expected_fit, fields], tuple(fields), target[expected_fit])
            state = meta.get("full_model_state", {})
            if state.get("fit_data_sha256") != expected_fp or meta.get("fit_data_sha256") != expected_fp:
                raise AssertionError(f"{task}@{end} fit fingerprint mismatch")
            if state.get("fit_rows") != int(expected_fit.sum()) or meta.get("fit_rows") != int(expected_fit.sum()):
                raise AssertionError(f"{task}@{end} fit population count mismatch")
            if state.get("all_supervised_training_token_features_strictly_earlier_cycles") is not True:
                raise AssertionError(f"{task}@{end} state does not assert strict earlier-cycle encoding")
            manual = manual_ordered_token_audit(frame, target, expected_fit, task, state)
            token_checks[f"{task}_{end}"] = manual
            if not isinstance(state.get("model_sha256"), str) or len(state["model_sha256"]) != 64:
                raise AssertionError(f"{task}@{end} model state fingerprint absent")
            for required in ["input_feature_names", "parameters", "structured", "ordered_tokens", "design_sha256"]:
                if required not in state:
                    raise AssertionError(f"{task}@{end} model audit missing {required}")
            fit_checks[f"{task}_{end}"] = {
                "fit_fingerprint": expected_fp,
                "fit_rows": int(expected_fit.sum()),
                "model_fingerprint_present": True,
                "cache_sha256": sha(path),
                "fit_population": "present_stream_only" if task == "classification" else "all_train_rows",
            }
            if task == "classification":
                rows = arrays["rows"].astype(np.int64)
                horizon = 67 if end == 55 else 79
                expected_rows = np.flatnonzero(frame.cycle_code.between(end + 1, horizon).to_numpy())
                np.testing.assert_array_equal(rows, expected_rows)
                missing = stream_missing_mask(frame.iloc[rows])
                if not np.array_equal(arrays["missing"].astype(bool), missing):
                    raise AssertionError(f"Classification @{end} missing-route mask mismatch")
                if np.any(arrays["raw_probability"][missing] != 0.0):
                    raise AssertionError("Unrouted missing rows should not receive present-expert probabilities")
                control = class_controls[end]
                np.testing.assert_array_equal(arrays["c2b_present"], control["c2b"])
                np.testing.assert_array_equal(arrays["cb_missing"], control["cb"])
                initial = np.where(missing, control["cb"], control["c2b"])
                latest = np.where(missing, control["cb"], control["latest_control"])
                direct = np.where(missing, control["cb"], arrays["raw_probability"])
                blend = np.where(missing, control["cb"], 0.75 * control["latest_control"] + 0.25 * arrays["raw_probability"])
                for key, expected in [("initial_control", initial), ("latest_control", latest),
                                      ("direct", direct), ("blend25", blend)]:
                    np.testing.assert_allclose(arrays[key], expected, rtol=0.0, atol=1e-14)
                if np.any((arrays["direct"] < 0) | (arrays["direct"] > 1)) or not np.isfinite(arrays["direct"]).all():
                    raise AssertionError(f"Classification @{end} probabilities invalid")
                class_caches[end] = arrays
            else:
                rows = arrays["rows"].astype(np.int64)
                horizon = 67 if end == 55 else 79
                expected_rows = np.flatnonzero(frame.cycle_code.between(end + 1, horizon).to_numpy())
                np.testing.assert_array_equal(rows, expected_rows)
                control = reg_h12.loc[reg_h12.cutoff.eq(end)].copy()
                np.testing.assert_array_equal(rows, control.row.to_numpy(dtype=np.int64))
                champion = control.prediction.to_numpy(dtype=float)
                np.testing.assert_allclose(arrays["champion"], champion, rtol=0.0, atol=1e-12)
                cycles = frame.iloc[rows].cycle_code.to_numpy(dtype=int)
                raw = np.asarray(arrays["raw"], float)
                rank = np.empty(len(raw), dtype=float)
                for cycle in np.unique(cycles):
                    mask = cycles == cycle
                    rank[mask] = 0.5 * raw[mask] + 0.5 * 100.0 * rankdata(raw[mask], method="average") / mask.sum()
                rank = np.clip(rank, 0, 100)
                blend = np.clip(0.75 * champion + 0.25 * rank, 0, 100)
                np.testing.assert_allclose(arrays["rank50"], rank, rtol=0.0, atol=1e-12)
                np.testing.assert_allclose(arrays["blend25"], blend, rtol=0.0, atol=1e-12)
                if not np.isfinite(raw).all() or not np.isfinite(blend).all():
                    raise AssertionError(f"Regression @{end} predictions are nonfinite")
                reg_caches[end] = arrays

    class_views = _classification_views(class_frame, class_target, class_caches)
    reg_views = _regression_views(reg_frame, reg_target, reg_caches)
    for name, expected in class_views.items():
        assert_tree(result["classification"]["views"][name], expected, f"classification/{name}")
    for name, expected in reg_views.items():
        assert_tree(result["regression"]["views"][name], expected, f"regression/{name}", atol=1e-9)

    class_deltas = [class_views[name]["metrics"]["blend25"]["macro_f1"] -
                    class_views[name]["metrics"]["latest_sparse25"]["macro_f1"]
                    for name in ["W1", "W4", "H12_E", "H12_L"]]
    class_primary = class_views["primary_pilot"]["metrics"]["blend25"]["macro_f1"] - class_views["primary_pilot"]["metrics"]["latest_sparse25"]["macro_f1"]
    class_h12 = class_views["H12"]["metrics"]["blend25"]["macro_f1"] - class_views["H12"]["metrics"]["latest_sparse25"]["macro_f1"]
    class_gate = result["classification"]["advance_gate_vs_latest_sparse25"]
    assert_tree(class_gate["primary_or_h12_improves"], bool(class_primary > 0 or class_h12 > 0), "classification gate improve")
    assert_tree(class_gate["no_view_harm_over_0_01"], bool(min(class_deltas) >= -0.01), "classification gate harm")
    reg_harms = [reg_views[name]["metrics"]["blend25"]["mae"] - reg_views[name]["metrics"]["champion"]["mae"]
                 for name in ["W1", "W4", "H12_E", "H12_L"]]
    reg_primary = reg_views["primary_pilot"]["metrics"]["champion"]["mae"] - reg_views["primary_pilot"]["metrics"]["blend25"]["mae"]
    reg_h12_gain = reg_views["H12"]["metrics"]["champion"]["mae"] - reg_views["H12"]["metrics"]["blend25"]["mae"]
    reg_gate = result["regression"]["advance_gate_vs_current_champion"]
    assert_tree(reg_gate["primary_or_h12_improves"], bool(reg_primary > 0 or reg_h12_gain > 0), "regression gate improve")
    assert_tree(reg_gate["max_view_harm_at_most_0_50"], bool(max(reg_harms) <= 0.50), "regression gate harm")
    receipt = {
        "status": "PASS",
        "freeze_signature": frozen["signature"],
        "pilot_json_sha256": sha(OUT / "pilot.json"),
        "newfits": 0,
        "test_read": False,
        "kaggle_access": False,
        "official_train_rows_per_task": 50608,
        "fit_checks": fit_checks,
        "ordered_token_independent_checks": token_checks,
        "classification_metrics_independently_recomputed": True,
        "regression_metrics_independently_recomputed": True,
        "classification_control_and_missing_route_identity_verified": True,
        "regression_rank50_and_fixed_blend_recomputed": True,
        "all_six_views_per_task_verified": True,
        "classification_advance_gate": {"primary_gain": float(class_primary), "h12_gain": float(class_h12),
                                         "minimum_window_delta": float(min(class_deltas)),
                                         "pass": bool((class_primary > 0 or class_h12 > 0) and min(class_deltas) >= -0.01)},
        "regression_advance_gate": {"primary_gain": float(reg_primary), "h12_gain": float(reg_h12_gain),
                                    "maximum_window_harm": float(max(reg_harms)),
                                    "pass": bool((reg_primary > 0 or reg_h12_gain > 0) and max(reg_harms) <= 0.50)},
        "known_limit": "Estimator binaries are not cached; fit fingerprints, model-state hashes, output cache hashes, source hashes, and independent prediction/metric reconstruction were checked.",
    }
    write_new(OUT / "verification.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    print(json.dumps(dry_run_schema() if args.dry_run else verify(), indent=2, allow_nan=False))
