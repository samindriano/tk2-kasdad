"""Independent, fit-free verification of frozen Round 2 classification results."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import numbers
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.audit import sha256, write_json
from src.models.classification_round2_evaluation import verify_freeze as r2verify_freeze
from src.utils.paths import PROJECT_ROOT, task_files


REPORT_DIR = PROJECT_ROOT / "reports/classification_round2"
RESULT_PATH = REPORT_DIR / "results.json"
DIAGNOSTIC_PATH = REPORT_DIR / "routing_diagnostic.json"
VERIFICATION_PATH = REPORT_DIR / "verification.json"
R2_CONFIG = PROJECT_ROOT / "experiments/classification_round2/protocol.json"
R1_CONFIG = PROJECT_ROOT / "experiments/classification_round1/protocol.json"
R1_FREEZE = PROJECT_ROOT / "experiments/classification_round1/freeze.json"

CONTROL_IDS = ("C1_A", "C1_B", "C2_A", "C2_B")
ROUTE_IDS = ("C4_LR", "C4_HGB")
POLICY_IDS = ("fixed_0.5", "inner_pooled", "inner_shrink50", "inner_equal_cycle")
COMPONENT_NAMES = ("intercept", "numeric", "categorical", "text", "missing_indicator")
CODE_COLUMNS = ("org_code", "module_code", "pattern_code")
EXPECTED_SUBGROUP_NAMES = (
    "EF0", "EF1", "stream_missing", "stream_present", "impact_low", "impact_middle", "impact_high",
    "EF0_stream_missing", "EF0_stream_present", "EF1_stream_missing", "EF1_stream_present",
    "org_code_missing", "org_code_present", "module_code_missing", "module_code_present",
    "pattern_code_missing", "pattern_code_present", "all_codes_known_nonmissing", "any_unseen_code",
    "any_missing_code",
)


def _json_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _array_digest(value: Any) -> str:
    return hashlib.sha256(np.asarray(value, dtype="<f8").tobytes(order="C")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object at {path}")
    return value


def _assert_close(actual: Any, expected: Any, label: str, *, atol: float = 1e-12) -> None:
    if actual is None or expected is None:
        if actual is not expected:
            raise AssertionError(f"{label}: {actual!r} != {expected!r}")
        return
    if isinstance(actual, (bool, np.bool_)) or isinstance(expected, (bool, np.bool_)):
        if bool(actual) != bool(expected):
            raise AssertionError(f"{label}: {actual!r} != {expected!r}")
        return
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=atol):
        raise AssertionError(f"{label}: {actual!r} differs from {expected!r}")


def _assert_metric(saved: dict[str, Any], actual: dict[str, Any], label: str) -> None:
    for key in (
        "n", "threshold", "threshold_range", "label_order", "confusion_matrix", "macro_f1",
        "balanced_accuracy", "average_precision", "positive_rate", "positive_prediction_rate",
        "mean_probability", "brier_score", "false_positive", "false_negative", "single_class_actual",
        "metric_convention",
    ):
        if key not in saved or key not in actual:
            raise AssertionError(f"{label}: missing metric key {key!r}")
        if key in {"macro_f1", "balanced_accuracy", "average_precision", "positive_rate",
                   "positive_prediction_rate", "mean_probability", "brier_score", "threshold"}:
            _assert_close(saved[key], actual[key], f"{label}.{key}")
        elif saved[key] != actual[key]:
            raise AssertionError(f"{label}.{key}: {saved[key]!r} != {actual[key]!r}")
    if len(saved["per_class"]) != 2 or len(actual["per_class"]) != 2:
        raise AssertionError(f"{label}: expected exactly two per-class metric rows")
    for label_index in (0, 1):
        saved_class = saved["per_class"][label_index]
        actual_class = actual["per_class"][label_index]
        if saved_class.get("label") != label_index or actual_class.get("label") != label_index:
            raise AssertionError(f"{label}: class label order changed")
        for key in ("support", "precision", "recall", "f1"):
            if key == "support":
                if saved_class.get(key) != actual_class.get(key):
                    raise AssertionError(f"{label}.per_class[{label_index}].{key} mismatch")
            else:
                _assert_close(saved_class.get(key), actual_class.get(key),
                              f"{label}.per_class[{label_index}].{key}")


def _binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float,
                    *, cross_check_sklearn: bool = True) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(p, dtype=float)
    if y.ndim != 1 or p.shape != y.shape:
        raise ValueError("Independent metric calculation requires aligned vectors")
    prediction = p >= float(threshold)
    cm = np.zeros((2, 2), dtype=np.int64)
    np.add.at(cm, (y, prediction.astype(np.int8)), 1)
    per_class = []
    for label in (0, 1):
        tp = int(cm[label, label])
        fp = int(cm[:, label].sum() - tp)
        fn = int(cm[label, :].sum() - tp)
        support = int(cm[label, :].sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        per_class.append({"label": label, "support": support, "precision": precision, "recall": recall, "f1": f1})
    both = bool((cm.sum(axis=1) > 0).all())
    result = {
        "n": int(len(y)), "threshold": float(threshold), "threshold_range": [float(threshold), float(threshold)],
        "label_order": [0, 1], "confusion_matrix": cm.tolist(), "per_class": per_class,
        "macro_f1": float(np.mean([row["f1"] for row in per_class])) if len(y) else None,
        "balanced_accuracy": float(np.mean([row["recall"] for row in per_class])) if both else None,
        "average_precision": float(average_precision_score(y, p)) if both else None,
        "positive_rate": float(y.mean()) if len(y) else None,
        "positive_prediction_rate": float(prediction.mean()) if len(y) else None,
        "mean_probability": float(p.mean()) if len(y) else None,
        "brier_score": float(np.mean((p - y) ** 2)) if len(y) else None,
        "false_positive": int(cm[0, 1]), "false_negative": int(cm[1, 0]),
        "single_class_actual": bool(len(y) and not both),
        "metric_convention": (
            "Fixed labels [0,1], undefined per-class precision/recall/F1=0; AP and balanced accuracy "
            "null without both actual classes."
        ),
    }
    # Cross-check the independent confusion-count calculation against sklearn.
    if len(y) and cross_check_sklearn:
        if confusion_matrix(y, prediction, labels=[0, 1]).tolist() != result["confusion_matrix"]:
            raise AssertionError("Independent confusion matrix disagrees with sklearn")
        _assert_close(f1_score(y, prediction, labels=[0, 1], average="macro", zero_division=0),
                      result["macro_f1"], "sklearn macro-F1 reconciliation")
        precision, recall, f1, support = precision_recall_fscore_support(
            y, prediction, labels=[0, 1], zero_division=0
        )
        for i in (0, 1):
            _assert_close(precision[i], per_class[i]["precision"], f"sklearn precision class {i}")
            _assert_close(recall[i], per_class[i]["recall"], f"sklearn recall class {i}")
            _assert_close(f1[i], per_class[i]["f1"], f"sklearn F1 class {i}")
            if int(support[i]) != per_class[i]["support"]:
                raise AssertionError(f"sklearn support class {i} mismatch")
    return result


def _f1_counts(y: np.ndarray, prediction: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int8)
    prediction = np.asarray(prediction, dtype=bool)
    values = []
    for label in (0, 1):
        actual = y == label
        predicted = prediction if label == 1 else ~prediction
        tp = int(np.count_nonzero(actual & predicted))
        fp = int(np.count_nonzero(~actual & predicted))
        fn = int(np.count_nonzero(actual & ~predicted))
        denom = 2 * tp + fp + fn
        values.append(2 * tp / denom if denom else 0.0)
    return float((values[0] + values[1]) / 2)


def _fit_fingerprint(frame: pd.DataFrame, rows: np.ndarray, features: list[str], y: np.ndarray) -> str:
    """Recreate the frozen Round 1 fit fingerprint without invoking a model."""
    digest = hashlib.sha256()

    def add_text(value: str) -> None:
        encoded = value.encode("utf-8", errors="surrogatepass")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)

    def fingerprint_value(value: Any) -> str:
        try:
            missing = bool(pd.isna(value))
        except (TypeError, ValueError):
            missing = False
        if missing:
            return "null"
        if isinstance(value, (bool, np.bool_)):
            return "bool:" + ("1" if value else "0")
        if isinstance(value, numbers.Integral):
            return f"int:{int(value)}"
        if isinstance(value, numbers.Real):
            numeric = float(value)
            if not math.isfinite(numeric):
                return "float:" + repr(numeric)
            return "float:" + numeric.hex()
        return "str:" + str(value)

    selected = frame.iloc[rows].loc[:, features]
    add_text(json.dumps(tuple(features), ensure_ascii=False, separators=(",", ":")))
    add_text(str(len(selected)))
    for column in features:
        for value in selected[column].to_numpy(dtype=object):
            add_text(fingerprint_value(value))
    for value in np.asarray(y).astype(np.int8):
        add_text(fingerprint_value(float(value)))
    return digest.hexdigest()


def _window_rows(frame: pd.DataFrame, window: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    fit_start, fit_end = [int(value) for value in window["fit"]]
    val_start, val_end = [int(value) for value in window["validation"]]
    fit_rows = np.flatnonzero(frame.cycle_code.between(fit_start, fit_end).to_numpy())
    validation_rows = np.flatnonzero(frame.cycle_code.between(val_start, val_end).to_numpy())
    return fit_rows, validation_rows


def _expected_outer_rows(frame: pd.DataFrame, end: int, protocol: dict[str, Any]) -> np.ndarray:
    matching = [window for window in protocol["primary_windows"] + protocol["secondary_windows"]
                if int(window["fit"][1]) == end]
    if not matching:
        raise ValueError(f"No protocol validation window has outer history {end}")
    last = max(int(window["validation"][1]) for window in matching)
    return np.flatnonzero(frame.cycle_code.between(end + 1, last).to_numpy())


def _expected_inner_rows(frame: pd.DataFrame, end: int) -> tuple[np.ndarray, np.ndarray]:
    inner_fit = np.flatnonzero(frame.cycle_code.between(47, end - 4).to_numpy())
    inner_validation = np.flatnonzero(frame.cycle_code.between(end - 3, end).to_numpy())
    return inner_fit, inner_validation


def _stream_missing(frame: pd.DataFrame) -> np.ndarray:
    values = frame["signal_stream"].to_numpy(dtype=object)
    mask = np.zeros(len(values), dtype=bool)
    for index, value in enumerate(values):
        try:
            missing = bool(pd.isna(value))
        except (TypeError, ValueError):
            missing = False
        mask[index] = missing or not str(value).strip()
    return mask


def _route_probability(frame: pd.DataFrame, rows: np.ndarray, present: np.ndarray,
                       missing_branch: np.ndarray) -> np.ndarray:
    if not np.array_equal(present.shape, missing_branch.shape) or present.shape != rows.shape:
        raise ValueError("Route branches must be row-aligned")
    # Availability alone selects a branch; no target or subgroup label is read.
    return np.where(_stream_missing(frame.iloc[rows]), missing_branch, present)


def _raw_subgroup_masks(validation: pd.DataFrame, fit: pd.DataFrame) -> dict[str, np.ndarray]:
    """Rebuild scientific subgroup memberships directly from raw validation/fit fields."""
    stream_missing = _stream_missing(validation)
    impact = pd.to_numeric(validation["impact_load"], errors="raise").to_numpy(dtype=float)
    lower, upper = pd.to_numeric(fit["impact_load"], errors="raise").quantile([1 / 3, 2 / 3]).to_numpy()
    ef0 = validation["entry_friction"].eq(0).to_numpy()
    ef1 = validation["entry_friction"].eq(1).to_numpy()
    masks: dict[str, np.ndarray] = {
        "EF0": ef0, "EF1": ef1,
        "stream_missing": stream_missing, "stream_present": ~stream_missing,
        "impact_low": impact < lower,
        "impact_middle": (impact >= lower) & (impact < upper),
        "impact_high": impact >= upper,
        "EF0_stream_missing": ef0 & stream_missing,
        "EF0_stream_present": ef0 & ~stream_missing,
        "EF1_stream_missing": ef1 & stream_missing,
        "EF1_stream_present": ef1 & ~stream_missing,
    }
    unseen_any = np.zeros(len(validation), dtype=bool)
    missing_any = np.zeros(len(validation), dtype=bool)
    for column in CODE_COLUMNS:
        values = validation[column].to_numpy(dtype=object)
        missing = np.zeros(len(values), dtype=bool)
        normalized = np.empty(len(values), dtype=object)
        for position, value in enumerate(values):
            try:
                is_na = bool(pd.isna(value))
            except (TypeError, ValueError):
                is_na = False
            normalized[position] = str(value).strip()
            missing[position] = is_na or not normalized[position]
        known = {
            str(value).strip()
            for value in fit[column].dropna().astype(str)
        }
        unseen = (~missing) & (~pd.Series(normalized).isin(known).to_numpy())
        masks[column + "_missing"] = missing
        masks[column + "_present"] = ~missing
        unseen_any |= unseen
        missing_any |= missing
    masks["all_codes_known_nonmissing"] = ~unseen_any & ~missing_any
    masks["any_unseen_code"] = unseen_any
    masks["any_missing_code"] = missing_any
    if tuple(masks) != EXPECTED_SUBGROUP_NAMES:
        raise RuntimeError("Independent raw subgroup names/order changed")
    return masks


def _compare_subgroups(saved: dict[str, Any], masks: dict[str, np.ndarray], y: np.ndarray,
                       p: np.ndarray, threshold: float, label: str) -> int:
    if set(saved) != set(masks):
        raise AssertionError(f"{label}: subgroup schema differs")
    checked = 0
    for name, mask in masks.items():
        expected = _subgroup_counts(y[mask], p[mask] >= threshold, threshold)
        _assert_subgroup(saved[name], expected, f"{label}.{name}")
        checked += 1
    return checked


def _subgroup_counts(y: np.ndarray, prediction: np.ndarray, threshold: float | np.ndarray) -> dict[str, Any]:
    """Recompute subgroup support, confusion counts, and fixed-label F1 only."""
    y = np.asarray(y, dtype=np.int8)
    prediction = np.asarray(prediction, dtype=bool)
    cm = np.zeros((2, 2), dtype=np.int64)
    if len(y):
        np.add.at(cm, (y, prediction.astype(np.int8)), 1)
    per_class = []
    for label in (0, 1):
        tp = int(cm[label, label])
        fp = int(cm[:, label].sum() - tp)
        fn = int(cm[label, :].sum() - tp)
        support = int(cm[label, :].sum())
        per_class.append({
            "label": label, "support": support,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / support if support else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        })
    cuts = np.asarray(threshold, dtype=float)
    return {
        "n": int(len(y)), "threshold": float(cuts) if cuts.ndim == 0 else None,
        "threshold_range": [float(cuts.min()), float(cuts.max())] if cuts.size else None,
        "label_order": [0, 1],
        "confusion_matrix": cm.tolist(), "per_class": per_class,
        "macro_f1": float(np.mean([row["f1"] for row in per_class])) if len(y) else None,
        "false_positive": int(cm[0, 1]), "false_negative": int(cm[1, 0]),
    }


def _assert_subgroup(saved: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    for field in ("n", "threshold", "threshold_range", "label_order", "confusion_matrix",
                  "macro_f1", "false_positive", "false_negative"):
        if field in {"threshold", "macro_f1"}:
            _assert_close(saved.get(field), expected[field], f"{label}.{field}")
        elif saved.get(field) != expected[field]:
            raise AssertionError(f"{label}.{field} mismatch")
    for index in (0, 1):
        for field in ("label", "support"):
            if saved["per_class"][index].get(field) != expected["per_class"][index][field]:
                raise AssertionError(f"{label}.per_class[{index}].{field} mismatch")
        for field in ("precision", "recall", "f1"):
            _assert_close(saved["per_class"][index].get(field), expected["per_class"][index][field],
                          f"{label}.per_class[{index}].{field}")


def _threshold_records(frame: pd.DataFrame, end: int, inner_rows: np.ndarray,
                       inner_probability: np.ndarray, protocol: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Recompute fixed, pooled, equal-cycle, and exact half-shrink cut records."""
    inner_frame = frame.iloc[inner_rows]
    y = inner_frame.artifact_signal.to_numpy(dtype=np.int8)
    cycles = inner_frame.cycle_code.to_numpy(dtype=int)
    cfg = protocol["threshold_selection"]
    grid = np.asarray(cfg["grid"], dtype=float)
    inner_fit_end = end - 4
    fit = frame[frame.cycle_code.between(47, inner_fit_end)]
    min_fit_cycles = int(cfg["minimum_inner_fit_cycles"])
    min_fit_rows = int(cfg["minimum_inner_fit_rows"])
    min_val_rows = int(cfg["minimum_inner_validation_rows"])
    eligible_fit = (
        fit.cycle_code.nunique() >= min_fit_cycles
        and len(fit) >= min_fit_rows
        and fit.artifact_signal.nunique() == 2
    )
    cycle_values = np.unique(cycles)
    fallback_reason = None
    if not eligible_fit:
        fallback_reason = "Inner threshold selection is not eligible for this recipe."
    elif len(y) < min_val_rows:
        fallback_reason = "Insufficient inner validation rows."
    elif len(np.unique(y)) < 2:
        fallback_reason = "Inner validation does not contain both actual classes."
    elif len(cycle_values) < 4:
        fallback_reason = "Fewer than four inner validation cycles."

    pooled_threshold = equal_threshold = 0.5
    pooled_objective = equal_objective = None
    if fallback_reason is None:
        pooled_scores = [(float(t), _f1_counts(y, inner_probability >= t)) for t in grid]
        equal_scores = []
        for threshold in grid:
            prediction = inner_probability >= threshold
            scores = [_f1_counts(y[cycles == cycle], prediction[cycles == cycle])
                      for cycle in cycle_values]
            equal_scores.append((float(threshold), float(np.mean(scores))))

        def choose(values: list[tuple[float, float]]) -> tuple[float, float]:
            best = max(score for _, score in values)
            tied = [(threshold, score) for threshold, score in values if abs(score - best) <= 1e-12]
            return min(tied, key=lambda item: (round(abs(item[0] - 0.5), 12), item[0]))

        pooled_threshold, pooled_objective = choose(pooled_scores)
        equal_threshold, equal_objective = choose(equal_scores)
        for threshold in {pooled_threshold, equal_threshold}:
            prediction = inner_probability >= threshold
            _assert_close(
                _f1_counts(y, prediction),
                f1_score(y, prediction, labels=[0, 1], average="macro", zero_division=0),
                f"selected pooled-threshold sklearn cross-check at {threshold:.2f}",
            )
            counts_by_cycle = [_f1_counts(y[cycles == cycle], prediction[cycles == cycle])
                               for cycle in cycle_values]
            sklearn_by_cycle = [
                f1_score(y[cycles == cycle], prediction[cycles == cycle], labels=[0, 1],
                         average="macro", zero_division=0)
                for cycle in cycle_values
            ]
            for cycle, count_score, sklearn_score in zip(cycle_values, counts_by_cycle, sklearn_by_cycle):
                _assert_close(count_score, sklearn_score,
                              f"selected equal-cycle sklearn cross-check {cycle}/{threshold:.2f}")

    selected = {
        "fixed_0.5": 0.5,
        "inner_pooled": pooled_threshold,
        "inner_shrink50": 0.5 + 0.5 * (pooled_threshold - 0.5),
        "inner_equal_cycle": equal_threshold,
    }
    output: dict[str, dict[str, Any]] = {}
    for policy, threshold in selected.items():
        pred = inner_probability >= threshold
        pooled_f1 = _f1_counts(y, pred)
        equal_f1 = float(np.mean([_f1_counts(y[cycles == c], pred[cycles == c]) for c in cycle_values])) if len(y) else None
        is_selected = policy != "fixed_0.5"
        reason = (
            "Prespecified fixed threshold." if not is_selected else fallback_reason
            if fallback_reason is not None else
            "Half shrink of inner pooled cutoff toward 0.5; no additional optimization or rounding."
            if policy == "inner_shrink50" else
            "Max inner pooled Macro F1; ties within 1e-12 closest to 0.5 then lower."
            if policy == "inner_pooled" else
            "Max mean inner-cycle fixed-label [0,1] Macro F1; ties within 1e-12 closest to 0.5 then lower."
        )
        record: dict[str, Any] = {
            "threshold": float(threshold), "fallback": bool(is_selected and fallback_reason is not None),
            "reason": reason, "inner_pooled_f1": pooled_f1, "inner_equal_cycle_f1": equal_f1,
            "selection_n": int(len(y)), "selection_cycles": int(len(cycle_values)),
        }
        if policy == "inner_pooled":
            record["selection_objective_f1"] = pooled_objective
        elif policy == "inner_equal_cycle":
            record["selection_objective_f1"] = equal_objective
        output[policy] = record
    return output


def _compare_thresholds(saved: dict[str, Any], expected: dict[str, Any], label: str) -> int:
    if set(saved) != set(POLICY_IDS) or set(expected) != set(POLICY_IDS):
        raise AssertionError(f"{label}: all four policy threshold records are required")
    fields = ("threshold", "fallback", "reason", "inner_pooled_f1", "inner_equal_cycle_f1",
              "selection_n", "selection_cycles")
    count = 0
    for policy in POLICY_IDS:
        current, actual = saved[policy], expected[policy]
        for field in fields:
            if field in {"threshold", "inner_pooled_f1", "inner_equal_cycle_f1"}:
                _assert_close(current.get(field), actual.get(field), f"{label}.{policy}.{field}")
            elif current.get(field) != actual.get(field):
                raise AssertionError(f"{label}.{policy}.{field}: {current.get(field)!r} != {actual.get(field)!r}")
        if "selection_objective_f1" in actual:
            _assert_close(current.get("selection_objective_f1"), actual["selection_objective_f1"],
                          f"{label}.{policy}.selection_objective_f1")
        elif "selection_objective_f1" in current:
            raise AssertionError(f"{label}.{policy}: unexpected selection objective")
        count += 1
    return count


def _expected_features(recipe_id: str, protocol: dict[str, Any]) -> list[str]:
    family = {
        "C1_A": "structured", "C1_B": "structured", "C3_B": "hybrid", "C3_C": "hybrid",
        "C5_LOW": "hybrid", "C5_HIGH": "hybrid", "C2_A": "text", "C2_B": "hybrid",
    }[recipe_id]
    return list(protocol["feature_sets"][family])


def _recursive_keys(value: Any, path: str = "") -> list[str]:
    keys = []
    if isinstance(value, dict):
        for key, item in value.items():
            name = f"{path}.{key}" if path else str(key)
            keys.append(name)
            keys.extend(_recursive_keys(item, name))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            keys.extend(_recursive_keys(item, f"{path}[{index}]"))
    return keys


def _assert_no_raw_vocab_or_model_weights(meta: dict[str, Any], npz_keys: set[str], label: str) -> None:
    forbidden_leaf_keys = {
        "raw_vocabulary", "text_vocabulary", "vocabulary", "vocab", "raw_vocab", "text_vocab",
        "raw_weights", "text_weights", "model_weights", "full_weights", "weights",
        "full_coefficients", "text_coefficients", "categorical_coefficients", "coef", "coef_",
        "full_coef", "raw_coef",
    }
    bad = [key for key in _recursive_keys(meta)
           if key.rsplit(".", 1)[-1].split("[", 1)[0].casefold() in forbidden_leaf_keys]
    if bad:
        raise AssertionError(f"{label}: metadata exposes prohibited vocabulary/weight fields: {bad[:3]}")
    if any(re.search(r"coef|weight|vocab", key, re.I) for key in npz_keys if not key.startswith("logit__")):
        raise AssertionError(f"{label}: NPZ unexpectedly contains estimator coefficients or vocabulary")


def _load_r2_model_groups(frame: pd.DataFrame, protocol: dict[str, Any], result: dict[str, Any],
                          freeze: dict[str, Any]) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    run_id = result.get("run_id")
    if not isinstance(run_id, str) or not re.fullmatch(r"classification_round2_[A-Za-z0-9_-]+", run_id):
        raise ValueError("Round2 result run_id is missing or outside its namespace")
    run_root = (PROJECT_ROOT / "experiments/runs" / run_id).resolve()
    if run_root.parent != (PROJECT_ROOT / "experiments/runs").resolve():
        raise ValueError("Round2 run path escaped the experiments/runs root")
    ends = sorted({int(window["fit"][1]) for window in protocol["primary_windows"]})
    recipes = {recipe["id"]: recipe for recipe in protocol["recipes"] if recipe["kind"] != "route"}
    expected_recipes = set(protocol["fit_recipes"])
    if set(recipes) != expected_recipes or len(ends) != 6 or len(recipes) != 8:
        raise ValueError("Protocol must describe six histories and eight fitted model recipes")
    groups: dict[int, dict[str, Any]] = {}
    set_hashes = {}
    for end in ends:
        group: dict[str, Any] = {}
        inner_fit_rows, expected_inner = _expected_inner_rows(frame, end)
        expected_outer = _expected_outer_rows(frame, end, protocol)
        for recipe_id, recipe in recipes.items():
            tag = f"fit{end}__{recipe_id}"
            meta_path = run_root / f"{tag}.json"
            npz_path = run_root / f"{tag}.npz"
            if not meta_path.is_file() or not npz_path.is_file():
                raise FileNotFoundError(f"Incomplete Round2 artifact pair: {tag}")
            meta = _read_json(meta_path)
            with np.load(npz_path, allow_pickle=False) as archive:
                archive_keys = set(archive.files)
                arrays = {key: archive[key].copy() for key in archive.files}
            required = {"outer_rows", "probability", "inner_rows", "inner_probability"}
            if not required <= archive_keys:
                raise AssertionError(f"{tag}: missing required NPZ arrays")
            if meta.get("model") != recipe_id or meta.get("key") is None:
                raise AssertionError(f"{tag}: model metadata is not aligned to recipe")
            outer_rows = arrays["outer_rows"].astype(np.int64, copy=False)
            inner_rows = arrays["inner_rows"].astype(np.int64, copy=False)
            probability = np.asarray(arrays["probability"], dtype=float)
            inner_probability = np.asarray(arrays["inner_probability"], dtype=float)
            if not np.array_equal(outer_rows, expected_outer) or not np.array_equal(inner_rows, expected_inner):
                raise AssertionError(f"{tag}: saved row vector differs from protocol row positions")
            for label, vector, rows in (("outer", probability, outer_rows),
                                         ("inner", inner_probability, inner_rows)):
                if vector.shape != rows.shape or not np.isfinite(vector).all() or not ((vector >= 0) & (vector <= 1)).all():
                    raise AssertionError(f"{tag}: invalid {label} probabilities")
            if _array_digest(probability) != meta.get("probability_sha256"):
                raise AssertionError(f"{tag}: outer probability hash mismatch")
            if _array_digest(inner_probability) != meta.get("inner_probability_sha256"):
                raise AssertionError(f"{tag}: inner probability hash mismatch")
            if _json_digest(meta.get("fitted_state")) != meta.get("fitted_state_sha256"):
                raise AssertionError(f"{tag}: fitted-state digest mismatch")
            if meta.get("fit_started_at_utc", "") <= freeze.get("frozen_at_utc", ""):
                raise AssertionError(f"{tag}: fit predates the frozen protocol")
            if meta.get("outer_validation_labels_passed_to_fit_or_selection") is not False:
                raise AssertionError(f"{tag}: outer validation label gate is not explicitly false")
            fit_rows = np.flatnonzero(frame.cycle_code.between(47, end).to_numpy())
            if int(meta.get("fit_rows", -1)) != len(fit_rows) or int(meta.get("validation_rows", -1)) != len(expected_outer):
                raise AssertionError(f"{tag}: fit/validation row counts differ from the frozen split")
            state = meta["fitted_state"]
            features = _expected_features(recipe_id, protocol)
            if state.get("input_feature_names") != features:
                raise AssertionError(f"{tag}: fitted feature allowlist differs from frozen recipe")
            if set(features) & {"artifact_signal", "record_id", "threat_rank", "audit_lane", "checksum_hint"}:
                raise AssertionError(f"{tag}: forbidden target/identifier feature reached fitted model")
            actual_fit_hash = _fit_fingerprint(frame, fit_rows, features, frame.iloc[fit_rows].artifact_signal.to_numpy())
            if state.get("fit_data_sha256") != actual_fit_hash:
                raise AssertionError(f"{tag}: fitted-state fingerprint does not match outer fit rows/target")
            if int(state.get("fit_rows", -1)) != len(fit_rows):
                raise AssertionError(f"{tag}: fitted-state row count mismatch")

            expected_component_keys = (set(COMPONENT_NAMES) if recipe_id != "C1_B" else set())
            actual_component_keys = {key.removeprefix("logit__") for key in archive_keys if key.startswith("logit__")}
            if actual_component_keys != expected_component_keys:
                raise AssertionError(f"{tag}: component-array names differ: {actual_component_keys}")
            components = {}
            if expected_component_keys:
                components = {key: np.asarray(arrays["logit__" + key], dtype=float) for key in COMPONENT_NAMES}
                if any(value.shape != probability.shape or not np.isfinite(value).all() for value in components.values()):
                    raise AssertionError(f"{tag}: logit component shape/non-finite error")
                summed_logit = sum(components.values())
                reconstructed = 1.0 / (1.0 + np.exp(-summed_logit))
                probability_error = float(np.max(np.abs(reconstructed - probability)))
                if probability_error > 1e-12:
                    raise AssertionError(f"{tag}: expit(component sum) misses probability by {probability_error:.3g}")
                missing_mask = _stream_missing(frame.iloc[outer_rows])
                if np.any(components["text"][missing_mask] != 0.0):
                    raise AssertionError(f"{tag}: text logit component is nonzero for a missing stream")
                recorded_hashes = meta.get("component_sha256", {})
                if {key: _array_digest(value) for key, value in components.items()} != recorded_hashes:
                    raise AssertionError(f"{tag}: component-array hashes disagree with metadata")
                component_stat = meta.get("component_statistics", {})
                if component_stat.get("linear") is not True or component_stat.get("missing_text_component_exactly_zero") is not True:
                    raise AssertionError(f"{tag}: component summary does not attest the fitted linear components")
            elif meta.get("component_sha256") != {}:
                raise AssertionError(f"{tag}: non-linear HGB receipt unexpectedly lists logit arrays")
            _assert_no_raw_vocab_or_model_weights(meta, archive_keys, tag)
            meta["_fit_end"] = end
            group[recipe_id] = {
                "rows": outer_rows, "probability": probability,
                "inner_rows": inner_rows, "inner_probability": inner_probability,
                "components": components, "meta": meta, "recipe": recipe,
            }
            set_hashes[f"{end}__{recipe_id}"] = {
                "outer": meta["probability_sha256"], "inner": meta["inner_probability_sha256"],
                "state": meta["fitted_state_sha256"], "inner_state": meta.get("inner_state_sha256"),
                "components": meta.get("component_sha256", {}),
            }
        groups[end] = group
    if len(set_hashes) != 48:
        raise AssertionError("Expected exactly 48 fresh model vector/state groups")
    return groups, {"run_root": run_root, "ends": ends, "recipes": recipes, "hashes": set_hashes}


def _load_r1_references(frame: pd.DataFrame, groups: dict[int, dict[str, Any]],
                        r2_protocol: dict[str, Any]) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    r1_protocol = _read_json(R1_CONFIG)
    r1_freeze = _read_json(R1_FREEZE)
    r1_result = _read_json(REPORT_DIR.parent / "classification_round1/results.json")
    if r1_result.get("freeze_signature") != r1_freeze.get("signature"):
        raise AssertionError("Round1 vectors/report do not belong to the frozen Round1 checkpoint")
    r1_run_id = r1_result.get("run_id")
    r1_root = (PROJECT_ROOT / "experiments/runs" / r1_run_id).resolve()
    if r1_root.parent != (PROJECT_ROOT / "experiments/runs").resolve():
        raise ValueError("Round1 reference run path escaped experiments/runs")
    r1_ends = sorted(int(value) for value in r1_result.get("unique_outer_histories", []))
    if len(r1_ends) != 6 or r1_result.get("status") != "CLASSIFICATION_ROUND1_COMPLETE_REVIEW_NO_FINAL_MODEL":
        raise AssertionError("A complete original Round1 classification result is required")
    references: dict[int, dict[str, Any]] = {}
    reference_hashes = {}
    r1_recipes = {recipe["id"]: recipe for recipe in r1_protocol["recipes"]}
    if not set(CONTROL_IDS) <= set(r1_recipes):
        raise AssertionError("Round1 control recipes are missing from their original protocol")
    for end in groups:
        references[end] = {}
        outer_rows = groups[end]["C1_A"]["rows"]
        inner_rows = groups[end]["C1_A"]["inner_rows"]
        for model in CONTROL_IDS:
            tag = f"fit{end}__{model}"
            audit_path = PROJECT_ROOT / f"reports/classification_round1/fit_audits/fit{end}.json"
            audit_all = _read_json(audit_path)
            if model not in audit_all:
                raise AssertionError(f"Round1 fit audit lacks {tag}")
            audit = audit_all[model]
            npz_path = r1_root / f"{tag}.npz"
            if not npz_path.is_file():
                raise FileNotFoundError(f"Original Round1 reference vector missing: {tag}")
            with np.load(npz_path, allow_pickle=False) as archive:
                saved = {key: archive[key].copy() for key in archive.files}
            needed = {"outer_rows", "probability", "inner_rows", "inner_probability"}
            if not needed <= set(saved):
                raise AssertionError(f"{tag}: original Round1 archive is incomplete")
            if not np.array_equal(saved["outer_rows"], outer_rows) or not np.array_equal(saved["inner_rows"], inner_rows):
                raise AssertionError(f"{tag}: Round1 reference row positions differ from Round2 windows")
            if _array_digest(saved["probability"]) != audit.get("probability_sha256"):
                raise AssertionError(f"{tag}: Round1 outer vector hash does not match fit audit")
            if _array_digest(saved["inner_probability"]) != audit.get("inner_probability_sha256"):
                raise AssertionError(f"{tag}: Round1 inner vector hash does not match fit audit")
            r2 = groups[end][model]["meta"]
            if not np.array_equal(saved["probability"], groups[end][model]["probability"]):
                raise AssertionError(f"{tag}: fresh Round2 outer control differs from original Round1 probabilities")
            if r2.get("fitted_state_sha256") != audit.get("fitted_state_sha256"):
                raise AssertionError(f"{tag}: fresh Round2 fitted control state differs from Round1")
            if r2.get("control_matches_round1_bit_identically") is not True:
                raise AssertionError(f"{tag}: control fit metadata lacks exact Round1 match")
            if r2.get("inner_probabilities_reused_from_round1") is not True:
                raise AssertionError(f"{tag}: control inner probabilities were not marked reused")
            if not np.array_equal(saved["inner_probability"], groups[end][model]["inner_probability"]):
                raise AssertionError(f"{tag}: Round2 control inner vector differs from Round1 saved vector")
            if r2.get("inner_state_sha256") != audit.get("inner_state_sha256"):
                raise AssertionError(f"{tag}: reused Round1 inner audit-state reference mismatch")
            if r2.get("round1_reference", {}).get("key") != audit.get("key"):
                raise AssertionError(f"{tag}: original Round1 fit key mismatch")
            references[end][model] = {**saved, "audit": audit}
            reference_hashes[f"{end}__{model}"] = {
                "outer": audit["probability_sha256"], "inner": audit["inner_probability_sha256"],
                "state": audit["fitted_state_sha256"], "inner_state": audit["inner_state_sha256"],
            }
            if model in CONTROL_IDS and groups[end][model]["meta"].get("thresholds"):
                r2_cut = groups[end][model]["meta"]["thresholds"]["inner_pooled"]["threshold"]
                r1_cut = audit.get("selection", {}).get("threshold")
                if r1_cut is not None and r2_cut != r1_cut:
                    raise AssertionError(f"{tag}: Round2 inner pooled threshold changed the saved Round1 selection")
    if len(reference_hashes) != 24:
        raise AssertionError("Expected 24 original Round1 vector/state references")
    return references, {"hashes": reference_hashes, "r1_run_id": r1_run_id, "r1_freeze_signature": r1_freeze["signature"]}


def _derive_route_groups(frame: pd.DataFrame, groups: dict[int, dict[str, Any]],
                        protocol: dict[str, Any]) -> tuple[dict[int, dict[str, Any]], int]:
    candidates: dict[int, dict[str, Any]] = {}
    route_checks = 0
    for end, fit_group in groups.items():
        candidates[end] = {model: {key: value for key, value in fit_group[model].items() if key != "recipe"}
                           for model in fit_group}
        for route in (recipe for recipe in protocol["recipes"] if recipe["kind"] == "route"):
            present_id, missing_id = route["present"], route["missing"]
            present, missing = fit_group[present_id], fit_group[missing_id]
            if not np.array_equal(present["rows"], missing["rows"]) or not np.array_equal(present["inner_rows"], missing["inner_rows"]):
                raise AssertionError(f"{end}/{route['id']}: availability route branches use different rows")
            outer_mask = _stream_missing(frame.iloc[present["rows"]])
            inner_mask = _stream_missing(frame.iloc[present["inner_rows"]])
            probability = np.where(outer_mask, missing["probability"], present["probability"])
            inner_probability = np.where(inner_mask, missing["inner_probability"], present["inner_probability"])
            if not np.array_equal(probability[~outer_mask], present["probability"][~outer_mask]):
                raise AssertionError(f"{end}/{route['id']}: present-stream route branch changed C2_B predictions")
            if not np.array_equal(probability[outer_mask], missing["probability"][outer_mask]):
                raise AssertionError(f"{end}/{route['id']}: missing-stream route branch changed fallback predictions")
            if not np.array_equal(inner_probability[~inner_mask], present["inner_probability"][~inner_mask]):
                raise AssertionError(f"{end}/{route['id']}: present-stream inner route branch changed")
            if not np.array_equal(inner_probability[inner_mask], missing["inner_probability"][inner_mask]):
                raise AssertionError(f"{end}/{route['id']}: missing-stream inner route branch changed")
            candidates[end][route["id"]] = {
                "rows": present["rows"], "probability": probability,
                "inner_rows": present["inner_rows"], "inner_probability": inner_probability,
                "route": route, "present_mask": ~outer_mask,
            }
            route_checks += 1
    if route_checks != 12:
        raise AssertionError(f"Expected 12 derived routing groups, got {route_checks}")
    return candidates, route_checks


def _verify_routing_diagnostic(frame: pd.DataFrame, result: dict[str, Any],
                               diagnostic: dict[str, Any], candidates: dict[int, dict[str, Any]],
                               protocol: dict[str, Any]) -> int:
    freeze_signature = result.get("freeze_signature")
    if diagnostic.get("freeze_signature") != freeze_signature or diagnostic.get("new_round2_model_fits") != 0:
        raise AssertionError("Saved routing diagnostic does not match the frozen no-fit diagnostic contract")
    diagnostic_jobs = diagnostic.get("jobs", [])
    if len(diagnostic_jobs) != 24:
        raise AssertionError("Saved Round2 routing diagnostic must contain 24 window hashes")
    windows = {window["id"]: window for window in protocol["primary_windows"] + protocol["secondary_windows"]}
    expected_keys = set()
    for job in diagnostic_jobs:
        key = (job.get("family"), job.get("window"), job.get("model"))
        if key in expected_keys:
            raise AssertionError(f"Duplicate saved routing diagnostic window: {key}")
        expected_keys.add(key)
        family, window_id, recipe_id = key
        if recipe_id not in ("C2_B", *ROUTE_IDS) or window_id not in windows:
            raise AssertionError(f"Unexpected diagnostic job {key}")
        window = windows[window_id]
        end = int(window["fit"][1])
        fit_rows, rows = _window_rows(frame, window)
        bundle = candidates[end][recipe_id]
        where = {int(row): index for index, row in enumerate(bundle["rows"])}
        p = np.asarray([bundle["probability"][where[int(row)]] for row in rows])
        if _array_digest(p) != job.get("probability_sha256"):
            raise AssertionError(f"Diagnostic window hash mismatch for {key}")
    expected = {
        (family, window["id"], model)
        for family, windows_in_family in (("primary", protocol["primary_windows"]),
                                          ("secondary", protocol["secondary_windows"]))
        for window in windows_in_family
        for model in ("C2_B", *ROUTE_IDS)
    }
    if expected_keys != expected:
        raise AssertionError("Diagnostic window keys do not cover 8 windows × 3 control/route vectors")
    comparison = result.get("routing_diagnostic_comparison", {})
    if comparison.get("status") != "PASS" or comparison.get("window_vectors_identical") != 24:
        raise AssertionError("Full result does not attest all 24 saved diagnostic window matches")
    if comparison.get("route_window_vectors") != 16 or comparison.get("hybrid_control_window_vectors") != 8:
        raise AssertionError("Saved-versus-fresh routing comparison counts are not 16+8")
    return len(expected_keys)


def _verify_transform_fingerprints(groups: dict[int, dict[str, Any]]) -> dict[str, int]:
    structured_ids = ("C1_A", "C2_B", "C3_B", "C3_C", "C5_LOW", "C5_HIGH")
    text_ids = ("C2_A", "C2_B", "C3_C", "C5_LOW", "C5_HIGH")
    structured_comparisons = text_comparisons = 0
    for end, group in groups.items():
        structured = [group[model]["meta"]["coefficients"].get("structured_transform_sha256")
                      for model in structured_ids]
        text = [group[model]["meta"]["coefficients"].get("text_transform_sha256") for model in text_ids]
        if any(not value for value in structured) or len(set(structured)) != 1:
            raise AssertionError(f"{end}: controlled structured variants do not share one learned transform")
        if any(not value for value in text) or len(set(text)) != 1:
            raise AssertionError(f"{end}: controlled text variants do not share one learned vocabulary/transform")
        structured_comparisons += len(structured) - 1
        text_comparisons += len(text) - 1
    return {"structured_transform_comparisons": structured_comparisons,
            "text_transform_comparisons": text_comparisons}


def _verify_thresholds(frame: pd.DataFrame, groups: dict[int, dict[str, Any]],
                       candidates: dict[int, dict[str, Any]], protocol: dict[str, Any],
                       result: dict[str, Any], r1_references: dict[int, dict[str, Any]]) -> tuple[int, int, dict[int, dict[str, dict[str, Any]]]]:
    expected_cuts: dict[int, dict[str, dict[str, Any]]] = {}
    groups_checked = policy_records_checked = 0
    for end, candidates_at_end in candidates.items():
        expected_cuts[end] = {}
        inner_fit_rows, inner_rows = _expected_inner_rows(frame, end)
        for recipe in protocol["recipes"]:
            model = recipe["id"]
            bundle = candidates_at_end[model]
            if not np.array_equal(bundle["inner_rows"], inner_rows):
                raise AssertionError(f"{end}/{model}: threshold selection rows are not the frozen inner validation")
            # Recreate thresholds solely from labels/probabilities on inner rows.
            cuts = _threshold_records(frame, end, inner_rows,
                                      np.asarray(bundle["inner_probability"], dtype=float), protocol)
            expected_cuts[end][model] = cuts
            groups_checked += 1
            if recipe["kind"] != "route":
                saved_meta = groups[end][model]["meta"]
                meta_cuts = saved_meta.get("thresholds", {})
                _compare_thresholds(meta_cuts, cuts, f"fit{end}/{model}/audit")
                if recipe["kind"] == "variant":
                    inner_state = saved_meta.get("inner_state")
                    if not isinstance(inner_state, dict) or _json_digest(inner_state) != saved_meta.get("inner_state_sha256"):
                        raise AssertionError(f"fit{end}/{model}: fresh inner fitted state digest missing/mismatched")
                    if int(inner_state.get("fit_rows", -1)) != len(inner_fit_rows):
                        raise AssertionError(f"fit{end}/{model}: inner fitted-state row count mismatch")
                    features = _expected_features(model, protocol)
                    inner_hash = _fit_fingerprint(frame, inner_fit_rows, features,
                                                  frame.iloc[inner_fit_rows].artifact_signal.to_numpy())
                    if inner_state.get("fit_data_sha256") != inner_hash:
                        raise AssertionError(f"fit{end}/{model}: inner state fingerprint escapes fit-only rows")
                elif recipe["kind"] == "control":
                    if saved_meta.get("inner_state") is not None:
                        raise AssertionError(f"fit{end}/{model}: control unexpectedly refit an inner model")
                    if model in CONTROL_IDS:
                        old_selection = r1_references[end][model]["audit"].get("selection", {}).get("threshold")
                        if old_selection is not None and cuts["inner_pooled"]["threshold"] != old_selection:
                            raise AssertionError(f"fit{end}/{model}: inner pooled cutoff differs from original Round1 selection")

            # The full results table has one or two windows for each history; all
            # occurrences must carry the same four inner-derived thresholds.
            occurrences = [job for job in result["jobs"]
                           if job["fit_group"] == end and job["model"] == model]
            if not occurrences:
                raise AssertionError(f"fit{end}/{model}: no result job contains threshold records")
            for job in occurrences:
                job_cuts = job.get("thresholds", {})
                if set(job_cuts) != set(POLICY_IDS):
                    raise AssertionError(f"fit{end}/{model}/{job['window']}: not all four cuts were recorded")
                for policy in POLICY_IDS:
                    job_value = job_cuts[policy]
                    if isinstance(job_value, dict):
                        job_value = job_value.get("threshold")
                    _assert_close(job_value, cuts[policy]["threshold"],
                                  f"fit{end}/{model}/{job['window']}/{policy} cutoff")
                    policy_records_checked += 1
    if groups_checked != 60 or policy_records_checked != 320:
        raise AssertionError(f"Threshold audit incomplete: groups={groups_checked}, records={policy_records_checked}")
    return groups_checked, groups_checked * len(POLICY_IDS), expected_cuts


def _verify_metrics(frame: pd.DataFrame, protocol: dict[str, Any], result: dict[str, Any],
                    candidates: dict[int, dict[str, Any]], cuts: dict[int, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    jobs = result.get("jobs", [])
    if len(jobs) != 80 or result.get("total_evaluation_jobs") != 80:
        raise AssertionError("Expected exactly 80 complete Round2 window/model jobs")
    windows = {window["id"]: window for window in protocol["primary_windows"] + protocol["secondary_windows"]}
    recipe_ids = {recipe["id"] for recipe in protocol["recipes"]}
    seen = set()
    metric_records = class_records = cycle_records = subgroup_records = 0
    pools: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    per_job_data = {}
    for job in jobs:
        key = (job.get("family"), job.get("window"), job.get("model"))
        if key in seen:
            raise AssertionError(f"Duplicate result job {key}")
        seen.add(key)
        family, window_id, model = key
        if window_id not in windows or model not in recipe_ids:
            raise AssertionError(f"Unexpected result job {key}")
        window = windows[window_id]
        end = int(window["fit"][1])
        fit_rows, rows = _window_rows(frame, window)
        bundle = candidates[end][model]
        row_positions = {int(row): index for index, row in enumerate(bundle["rows"])}
        if any(int(row) not in row_positions for row in rows):
            raise AssertionError(f"{key}: model artifact lacks protocol validation rows")
        positions = np.asarray([row_positions[int(row)] for row in rows], dtype=int)
        p = np.asarray(bundle["probability"][positions], dtype=float)
        y = frame.iloc[rows].artifact_signal.to_numpy(dtype=np.int8)
        if job.get("probability_sha256") != _array_digest(p):
            raise AssertionError(f"{key}: job-level probability hash mismatch")
        expected_cut_map = cuts[end][model]
        actual_policy_map = job.get("policies", {})
        if set(actual_policy_map) != set(POLICY_IDS):
            raise AssertionError(f"{key}: expected four policies")
        validation = frame.iloc[rows]
        fit_frame = frame.iloc[fit_rows]
        masks = _raw_subgroup_masks(validation, fit_frame)
        per_job_data[key] = {"family": family, "model": model, "rows": rows, "p": p, "y": y,
                             "masks": masks, "end": end}
        for policy in POLICY_IDS:
            record = actual_policy_map[policy]
            threshold = float(expected_cut_map[policy]["threshold"])
            saved_threshold = job.get("thresholds", {}).get(policy)
            if isinstance(saved_threshold, dict):
                saved_threshold = saved_threshold.get("threshold")
            _assert_close(saved_threshold, threshold, f"{key}/{policy} job threshold")
            metrics = _binary_metrics(y, p, threshold)
            _assert_metric(record.get("metrics", {}), metrics, f"{key}/{policy}/metrics")
            metric_records += 1
            class_records += 2
            cycles = record.get("per_cycle", [])
            cycle_values = sorted(np.unique(validation.cycle_code.to_numpy(dtype=int)).tolist())
            if len(cycles) != len(cycle_values) or [item.get("cycle") for item in cycles] != cycle_values:
                raise AssertionError(f"{key}/{policy}: per-cycle metric coverage/order mismatch")
            cycle_f1s = []
            for cycle_record, cycle in zip(cycles, cycle_values):
                mask = validation.cycle_code.to_numpy(dtype=int) == cycle
                cycle_metric = _binary_metrics(y[mask], p[mask], threshold)
                _assert_metric(cycle_record, cycle_metric, f"{key}/{policy}/cycle{cycle}")
                cycle_f1s.append(cycle_metric["macro_f1"])
                cycle_records += 1
            equal_f1 = float(np.mean(cycle_f1s))
            _assert_close(record.get("equal_cycle_macro_f1"), equal_f1, f"{key}/{policy} equal-cycle macro-F1")
            subgroup_count = _compare_subgroups(record.get("subgroups", {}), masks, y, p, threshold,
                                                f"{key}/{policy}/subgroups")
            subgroup_records += subgroup_count
            pools.setdefault((family, policy, model), []).append({
                "rows": rows, "y": y, "p": p, "threshold": threshold,
                "masks": masks, "cycles": validation.cycle_code.to_numpy(dtype=int),
                "window": window_id,
            })
    expected_keys = {(family, window["id"], model)
                     for family, collection in (("primary", protocol["primary_windows"]),
                                                ("secondary", protocol["secondary_windows"]))
                     for window in collection for model in recipe_ids}
    if seen != expected_keys:
        raise AssertionError(f"Window/job set mismatch: actual={len(seen)} expected={len(expected_keys)}")
    if metric_records != 320 or class_records != 640 or cycle_records != 1920:
        raise AssertionError(f"Metric record counts mismatch: {metric_records}/{class_records}/{cycle_records}")
    if subgroup_records != 320 * len(EXPECTED_SUBGROUP_NAMES):
        raise AssertionError(f"Subgroup metric count mismatch: {subgroup_records}")

    aggregates = result.get("aggregate", {})
    aggregate_records = aggregate_subgroups = 0
    expected_cycles = np.arange(56, 80, dtype=int)
    expected_rows_n = int(frame.cycle_code.between(56, 79).sum())
    for family in ("primary", "secondary"):
        for policy in POLICY_IDS:
            for model in sorted(recipe_ids):
                key = (family, policy, model)
                if key not in pools:
                    raise AssertionError(f"Missing aggregate chunks for {key}")
                chunks = pools[key]
                rows = np.concatenate([chunk["rows"] for chunk in chunks])
                y = np.concatenate([chunk["y"] for chunk in chunks])
                p = np.concatenate([chunk["p"] for chunk in chunks])
                thresholds = np.concatenate([np.full(len(chunk["rows"]), chunk["threshold"]) for chunk in chunks])
                cycles = np.concatenate([chunk["cycles"] for chunk in chunks])
                if len(rows) != expected_rows_n or len(np.unique(rows)) != expected_rows_n:
                    raise AssertionError(f"{key}: aggregate pool rows are not a disjoint full cycle 56–79 set")
                if set(np.unique(cycles)) != set(expected_cycles):
                    raise AssertionError(f"{key}: aggregate pool misses one or more target cycles")
                pred = p >= thresholds
                cm = np.zeros((2, 2), dtype=np.int64)
                np.add.at(cm, (y, pred.astype(np.int8)), 1)
                # For pooled aggregate, policies may have different selected
                # thresholds by outer group; reconcile the vector-cut metrics
                # separately while retaining probability-only fields.
                class_metrics = _metrics_from_prediction(y, p, pred, thresholds)
                saved = aggregates[family][policy][model]
                _assert_metric(saved.get("metrics", {}), class_metrics, f"aggregate/{key}")
                cycle_f1 = float(np.mean([_f1_counts(y[cycles == c], pred[cycles == c]) for c in expected_cycles]))
                _assert_close(saved.get("equal_cycle_macro_f1"), cycle_f1, f"aggregate/{key}/equal-cycle")
                aggregate_records += 1
                subgroup_names = list(chunks[0]["masks"])
                if list(saved.get("subgroups", {})) != subgroup_names:
                    raise AssertionError(f"aggregate/{key}: subgroup order/schema mismatch")
                for name in subgroup_names:
                    mask = np.concatenate([chunk["masks"][name] for chunk in chunks])
                    subthresholds = thresholds[mask]
                    expected_sub = _subgroup_counts(
                        y[mask], pred[mask], subthresholds
                    )
                    _assert_subgroup(saved["subgroups"][name], expected_sub, f"aggregate/{key}/{name}")
                    aggregate_subgroups += 1
    if aggregate_records != 80 or aggregate_subgroups != 80 * len(EXPECTED_SUBGROUP_NAMES):
        raise AssertionError(f"Aggregate count mismatch: {aggregate_records}/{aggregate_subgroups}")
    return {
        "window_jobs_verified": len(jobs), "job_policy_metric_records_verified": metric_records,
        "per_class_records_verified": class_records, "per_cycle_metric_records_verified": cycle_records,
        "job_policy_subgroup_metrics_verified": subgroup_records,
        "pooled_and_equal_cycle_aggregates_verified": aggregate_records,
        "aggregate_subgroup_metrics_verified": aggregate_subgroups,
        "scientific_subgroups_per_job_policy": len(EXPECTED_SUBGROUP_NAMES),
        "pooled_cycle_rows_per_family_model": expected_rows_n,
    }


def _metrics_from_prediction(y: np.ndarray, p: np.ndarray, prediction: np.ndarray,
                             threshold: float | np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(p, dtype=float)
    prediction = np.asarray(prediction, dtype=bool)
    if not len(y) or y.shape != p.shape or y.shape != prediction.shape:
        raise ValueError("Aligned nonempty vectors required for aggregate metric verification")
    cm = np.zeros((2, 2), dtype=np.int64)
    np.add.at(cm, (y, prediction.astype(np.int8)), 1)
    per_class = []
    for label in (0, 1):
        tp, fp, fn = int(cm[label, label]), int(cm[:, label].sum() - cm[label, label]), int(cm[label, :].sum() - cm[label, label])
        support = int(cm[label, :].sum())
        per_class.append({"label": label, "support": support,
                          "precision": tp / (tp + fp) if tp + fp else 0.0,
                          "recall": tp / support if support else 0.0,
                          "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0})
    both = bool((cm.sum(axis=1) > 0).all())
    thresholds = np.asarray(threshold, dtype=float)
    return {
        "n": int(len(y)), "threshold": float(thresholds) if thresholds.ndim == 0 else None,
        "threshold_range": [float(thresholds.min()), float(thresholds.max())], "label_order": [0, 1],
        "confusion_matrix": cm.tolist(), "per_class": per_class,
        "macro_f1": float(np.mean([row["f1"] for row in per_class])),
        "balanced_accuracy": float(np.mean([row["recall"] for row in per_class])) if both else None,
        "average_precision": float(average_precision_score(y, p)) if both else None,
        "positive_rate": float(y.mean()), "positive_prediction_rate": float(prediction.mean()),
        "mean_probability": float(p.mean()), "brier_score": float(np.mean((p - y) ** 2)),
        "false_positive": int(cm[0, 1]), "false_negative": int(cm[1, 0]),
        "single_class_actual": not both,
        "metric_convention": (
            "Fixed labels [0,1], undefined per-class precision/recall/F1=0; AP and balanced accuracy "
            "null without both actual classes."
        ),
    }


def _verify_vector_accounting(result: dict[str, Any], groups: dict[int, dict[str, Any]],
                              info: dict[str, Any], route_groups: int, component_groups: int,
                              route_window_hashes: int, threshold_groups: int,
                              unique_threshold_records: int, window_threshold_records: int, ref_count: int,
                              r1_reuse_count: int) -> dict[str, int]:
    expected_fresh_fits = 72
    if int(result.get("fresh_model_fits_this_run", -1)) != expected_fresh_fits:
        raise AssertionError("Original Round2 result does not attest the frozen 72-fit run")
    if int(result.get("dependency_reconstruction_fits", -1)) != 0:
        raise AssertionError("Unexpected Round1 dependency reconstruction fit count")
    if (route_groups != 12 or route_window_hashes != 24 or threshold_groups != 60
            or unique_threshold_records != 240 or window_threshold_records != 320):
        raise AssertionError("Route, threshold, or diagnostic match counts are incomplete")
    if ref_count != 24 or r1_reuse_count != 24:
        raise AssertionError("Round1 fresh-control/reused-inner match counts are incomplete")
    return {
        "fresh_round2_fits_attested": expected_fresh_fits,
        "model_vector_groups": 48, "derived_route_groups": route_groups,
        "candidate_probability_groups": 48 + route_groups,
        "linear_component_groups": component_groups,
        "round1_outer_control_probability_and_state_matches": ref_count,
        "round1_control_inner_probability_and_state_reuses": r1_reuse_count,
        "routing_diagnostic_window_hash_matches": route_window_hashes,
        "threshold_groups_recomputed": threshold_groups,
        "policy_cut_records_recomputed": unique_threshold_records,
        "window_policy_cut_records_verified": window_threshold_records,
    }


def _verify_repeat_report(repeat_report: Path, original: dict[str, Any], groups: dict[int, dict[str, Any]],
                          candidates: dict[int, dict[str, Any]], protocol: dict[str, Any],
                          expected_cuts: dict[int, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    repeat = _read_json(repeat_report)
    if repeat.get("freeze_signature") != original.get("freeze_signature"):
        raise AssertionError("Representative refit uses a different Round2 freeze")
    if repeat.get("status") != "PARTIAL_HISTORY_REPRODUCTION" or repeat.get("unique_outer_histories") != [67]:
        raise AssertionError("Representative refit must be the separately scoped end-67 reproduction")
    if int(repeat.get("fresh_model_fits_this_run", -1)) != 12:
        raise AssertionError("Representative end-67 rerun must contain exactly 12 fresh fits")
    run_id = repeat.get("run_id")
    if not isinstance(run_id, str) or not re.fullmatch(r"classification_round2_[A-Za-z0-9_-]+", run_id):
        raise ValueError("Repeat run id is outside Round2 namespace")
    repeat_root = (PROJECT_ROOT / "experiments/runs" / run_id).resolve()
    if repeat_root.parent != (PROJECT_ROOT / "experiments/runs").resolve():
        raise ValueError("Repeat artifact root escaped experiments/runs")
    outer_vectors = inner_vectors = outer_states = inner_states = component_arrays = 0
    fitted_threshold_records = 0
    control_inner_reuses = control_inner_states = 0
    max_component_delta = 0.0
    end = 67
    fit_recipes = [recipe for recipe in protocol["recipes"] if recipe["kind"] != "route"]
    for recipe in fit_recipes:
        model = recipe["id"]
        tag = f"fit{end}__{model}"
        meta = _read_json(repeat_root / f"{tag}.json")
        with np.load(repeat_root / f"{tag}.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        baseline = groups[end][model]
        if not np.array_equal(arrays["outer_rows"], baseline["rows"]):
            raise AssertionError(f"repeat/{model}: outer row identities differ")
        if not np.array_equal(arrays["probability"], baseline["probability"]):
            raise AssertionError(f"repeat/{model}: outer probability vector is not bit-identical")
        outer_vectors += 1
        if meta.get("fitted_state_sha256") != baseline["meta"]["fitted_state_sha256"]:
            raise AssertionError(f"repeat/{model}: fitted-state hash differs")
        outer_states += 1
        if recipe["kind"] == "variant":
            if not np.array_equal(arrays["inner_rows"], baseline["inner_rows"]):
                raise AssertionError(f"repeat/{model}: variant inner row identities differ")
            if not np.array_equal(arrays["inner_probability"], baseline["inner_probability"]):
                raise AssertionError(f"repeat/{model}: variant inner vector is not bit-identical")
            if meta.get("inner_state_sha256") != baseline["meta"].get("inner_state_sha256"):
                raise AssertionError(f"repeat/{model}: variant inner-state hash differs")
            inner_vectors += 1
            inner_states += 1
        else:
            if not np.array_equal(arrays["inner_rows"], baseline["inner_rows"]):
                raise AssertionError(f"repeat/{model}: reused control inner row identities differ")
            if not np.array_equal(arrays["inner_probability"], baseline["inner_probability"]):
                raise AssertionError(f"repeat/{model}: reused control inner probabilities differ")
            if meta.get("inner_state_sha256") != baseline["meta"].get("inner_state_sha256"):
                raise AssertionError(f"repeat/{model}: reused control inner-state hash differs")
            if meta.get("inner_probabilities_reused_from_round1") is not True:
                raise AssertionError(f"repeat/{model}: control inner vector is not marked as the Round1 reuse")
            control_inner_reuses += 1
            control_inner_states += 1
        keys = {key.removeprefix("logit__") for key in arrays if key.startswith("logit__")}
        baseline_components = baseline["components"]
        if keys != set(baseline_components):
            raise AssertionError(f"repeat/{model}: component array schema differs")
        for component in keys:
            old = baseline_components[component]
            new = arrays["logit__" + component]
            max_component_delta = max(max_component_delta,
                                      float(np.max(np.abs(old - new))) if len(old) else 0.0)
            if not np.array_equal(old, new):
                raise AssertionError(f"repeat/{model}/{component}: component arrays differ (max {max_component_delta:.3g})")
            component_arrays += 1
        _compare_thresholds(meta.get("thresholds", {}), expected_cuts[end][model], f"repeat/{model}/audit thresholds")
        fitted_threshold_records += 4
    repeat_jobs = [job for job in repeat.get("jobs", []) if int(job.get("fit_group", -1)) == end]
    if len(repeat_jobs) != 20:
        raise AssertionError("Representative refit report must contain 20 end-67 result jobs")
    job_threshold_records = 0
    seen_job_keys = set()
    for job in repeat_jobs:
        model = job.get("model")
        if model not in expected_cuts[end]:
            raise AssertionError(f"Representative repeat has an unexpected model job: {model}")
        job_key = (job.get("family"), job.get("window"), model)
        if job_key in seen_job_keys:
            raise AssertionError(f"Representative repeat contains a duplicate evaluation job: {job_key}")
        seen_job_keys.add(job_key)
        recorded = job.get("thresholds", {})
        if set(recorded) != set(POLICY_IDS):
            raise AssertionError(f"Representative repeat job {job_key} lacks four policy cut records")
        for policy in POLICY_IDS:
            value = recorded[policy]
            if isinstance(value, dict):
                value = value.get("threshold")
            _assert_close(value, expected_cuts[end][model][policy]["threshold"],
                          f"representative repeat job {job_key}/{policy}")
            job_threshold_records += 1
    if len({key[2] for key in seen_job_keys}) != 10 or job_threshold_records != 80:
        raise AssertionError("Representative repeat job threshold coverage is not 10 recipes × 4 policies × 2 windows")
    if (outer_vectors, inner_vectors, outer_states, inner_states, control_inner_reuses,
            control_inner_states, fitted_threshold_records, job_threshold_records) != (8, 4, 8, 4, 4, 4, 32, 80):
        raise AssertionError("Representative refit accounting is incomplete")
    return {
        "status": "PASS", "fit_end": end, "fresh_model_fits": 12,
        "outer_probability_vectors_bit_identical": outer_vectors,
        "variant_inner_probability_vectors_bit_identical": inner_vectors,
        "outer_audit_states_bit_identical": outer_states,
        "variant_inner_audit_states_bit_identical": inner_states,
        "control_inner_vector_reuses_bit_identical": control_inner_reuses,
        "control_inner_state_reuses_identical": control_inner_states,
        "logit_component_arrays_bit_identical": component_arrays,
        "component_array_max_absolute_delta": max_component_delta,
        "fitted_model_four_policy_records_identical": fitted_threshold_records,
        "all_recipe_window_policy_cut_records_identical": job_threshold_records,
    }


def _can_preserve_representative_refit(old_receipt: Any, freeze_signature: str,
                                       original_vector_set_sha256: str,
                                       round2_vector_set_sha256: str) -> bool:
    """Only carry a PASS receipt forward when both compared vector sets still match."""
    if not isinstance(old_receipt, dict):
        return False
    representative = old_receipt.get("representative_refit")
    return (
        old_receipt.get("freeze_signature") == freeze_signature
        and old_receipt.get("original_vector_set_sha256") == original_vector_set_sha256
        and old_receipt.get("round2_vector_set_sha256") == round2_vector_set_sha256
        and isinstance(representative, dict)
        and representative.get("status") == "PASS"
    )


def verify_results(repeat_report: Path | str | None = None) -> dict[str, Any]:
    """Verify saved Round2 vectors and metrics; no model fitting is performed."""
    # This Round2 verifier reads the frozen input/code/checkpoint manifest only.
    frozen = r2verify_freeze()
    if frozen.get("status") != "FROZEN_BEFORE_ROUND2_DIAGNOSTIC_AND_MODEL_FITS":
        raise AssertionError("Round2 freeze manifest is not in the pre-fit frozen state")
    protocol = _read_json(R2_CONFIG)
    result = _read_json(RESULT_PATH)
    diagnostic = _read_json(DIAGNOSTIC_PATH)
    train_path = task_files("classification")["train"]
    frame = pd.read_csv(train_path)
    if not frame.record_id.is_unique or frame.record_id.isna().any():
        raise AssertionError("Training data record_id must be unique and nonmissing")
    if not frame.artifact_signal.isin([0, 1]).all():
        raise AssertionError("Training target must contain only binary labels")
    if result.get("status") != "CLASSIFICATION_ROUND2_COMPLETE_REVIEW_NO_FINAL_MODEL":
        raise AssertionError("Complete original Round2 result required; partial results are not eligible")
    if result.get("freeze_signature") != frozen["signature"] or result.get("identity") != frozen["identity"]:
        raise AssertionError("Round2 result identity differs from read-only freeze verification")
    if result.get("protocol_id") != protocol.get("protocol_id"):
        raise AssertionError("Round2 protocol ID differs")

    groups, info = _load_r2_model_groups(frame, protocol, result, frozen)
    references, reference_info = _load_r1_references(frame, groups, protocol)
    candidates, route_group_count = _derive_route_groups(frame, groups, protocol)
    route_window_hashes = _verify_routing_diagnostic(frame, result, diagnostic, candidates, protocol)
    transform_counts = _verify_transform_fingerprints(groups)
    threshold_group_count, unique_threshold_records, expected_cuts = _verify_thresholds(
        frame, groups, candidates, protocol, result, references
    )
    metrics_counts = _verify_metrics(frame, protocol, result, candidates, expected_cuts)
    component_groups = sum(bool(bundle["components"]) for group in groups.values() for bundle in group.values())
    if component_groups != 42:
        raise AssertionError(f"Expected 42 linear component array groups, found {component_groups}")
    reference_count = len(reference_info["hashes"])
    reuse_count = sum(
        1 for end in groups for model in CONTROL_IDS
        if groups[end][model]["meta"].get("inner_probabilities_reused_from_round1") is True
        and np.array_equal(groups[end][model]["inner_probability"], references[end][model]["inner_probability"])
    )
    vector_counts = _verify_vector_accounting(
        result, groups, info, route_group_count, component_groups, route_window_hashes,
        threshold_group_count, unique_threshold_records, len(result["jobs"]) * len(POLICY_IDS),
        reference_count, reuse_count,
    )

    # The original vector-set digest is the preservation key for any existing
    # representative-refit receipt. A receipt from another vector/freeze set is
    # dropped rather than carried forward.
    original_vector_set_sha256 = _json_digest(reference_info["hashes"])
    r2_vector_set_sha256 = _json_digest(info["hashes"])
    verification: dict[str, Any] = {
        "status": "PASS",
        "freeze_signature": frozen["signature"],
        "frozen_base_git_head": frozen.get("base_git_head"),
        "run_id": result["run_id"],
        "identity": frozen["identity"],
        "input_train_sha256": frozen["identity"]["round1_identity"]["input_sha256"]["train"],
        "round1_freeze_signature": reference_info["r1_freeze_signature"],
        "round1_run_id": reference_info["r1_run_id"],
        "freeze_manifest_files_verified": len(frozen.get("protected_checkpoint_files", {})),
        "original_vector_set_sha256": original_vector_set_sha256,
        "round2_vector_set_sha256": r2_vector_set_sha256,
        "routing_diagnostic_sha256": _json_digest(diagnostic),
        "counts": {**vector_counts, **metrics_counts, **transform_counts},
        "maximum_logit_probability_reconstruction_error": max(
            float(bundle["meta"]["component_statistics"]["max_probability_reconstruction_error"])
            for group in groups.values() for bundle in group.values() if bundle["components"]
        ),
        "subgroup_masks_source": "independent recomputation from raw train fields; fit-only impact quantiles and fit-known code vocabularies",
        "threshold_method": "independent confusion-count grid search; selected pooled and equal-cycle cutoffs cross-checked with sklearn",
        "old_round1_protection": "read-only: only original saved references consumed; no Round1 fit/report/builder invocation",
        "modeling_or_refits_performed_by_verifier": 0,
    }
    if verification["maximum_logit_probability_reconstruction_error"] > 1e-12:
        raise AssertionError("At least one linear logit decomposition exceeds the 1e-12 probability tolerance")

    old_verification = _read_json(VERIFICATION_PATH) if VERIFICATION_PATH.exists() else None
    if repeat_report is not None:
        verification["representative_refit"] = _verify_repeat_report(
            Path(repeat_report), result, groups, candidates, protocol, expected_cuts
        )
    elif _can_preserve_representative_refit(
        old_verification, frozen["signature"], original_vector_set_sha256, r2_vector_set_sha256
    ):
        verification["representative_refit"] = old_verification["representative_refit"]

    # Only this canonical Round2 verification receipt is written.
    write_json(VERIFICATION_PATH, verification)
    return verification


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat-report", type=Path, help="Optional separate end-67 12-fit reproduction report")
    args = parser.parse_args()
    receipt = verify_results(args.repeat_report)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
