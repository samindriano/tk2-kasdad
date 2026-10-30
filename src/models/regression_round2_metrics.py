"""Leakage-blind diagnostics for TP2 Regression Round 2.

This module evaluates saved predictions or summarizes feature/history support.
It does not fit a model and never uses validation targets in history/test-risk
summaries.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np
import pandas as pd

from src.models.round1_models import (
    CATEGORICAL_FEATURES,
    EXPANDED_FEATURES,
    MIXED_COUNT_FEATURES,
    NUMERIC_FEATURES,
    TELEMETRY_FEATURE,
    _category_array,
    _is_missing,
    _mixed_count_arrays,
    _number_array,
    _telemetry_arrays,
)

TOLERANCE = 1e-10
TARGET_REGIONS = ("region_low", "region_middle", "region_high")
QUANTILES = (0, .01, .05, .25, .5, .75, .95, .99, 1)


def _aligned(y: Any, pred: Any) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(y, dtype=float)
    estimate = np.asarray(pred, dtype=float)
    if actual.ndim != 1 or estimate.ndim != 1 or len(actual) != len(estimate):
        raise ValueError("actual and prediction must be aligned one-dimensional arrays")
    if not np.isfinite(actual).all() or not np.isfinite(estimate).all():
        raise ValueError("metrics require finite actual and predicted values")
    return actual, estimate


def metric_values(y: Any, pred: Any, tolerance: float = TOLERANCE) -> dict[str, Any]:
    """Return auditable regression metrics; signed error is prediction minus target."""
    actual, estimate = _aligned(y, pred)
    n = len(actual)
    if not n:
        return {"n": 0, "mae": None, "rmse": None, "signed_error": None,
                "over_rate": None, "under_rate": None, "equal_rate": None,
                "absolute_error_p95": None, "prediction_quantiles": None,
                "target_quantiles": None, "below_zero": 0, "above_100": 0}
    error = estimate - actual
    return {
        "n": n,
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "signed_error": float(np.mean(error)),
        "over_rate": float(np.mean(error > tolerance)),
        "under_rate": float(np.mean(error < -tolerance)),
        "equal_rate": float(np.mean(np.abs(error) <= tolerance)),
        "absolute_error_p95": float(np.quantile(np.abs(error), .95)),
        "prediction_quantiles": {str(q): float(np.quantile(estimate, q)) for q in QUANTILES},
        "target_quantiles": {str(q): float(np.quantile(actual, q)) for q in QUANTILES},
        "below_zero": int(np.sum(estimate < 0)),
        "above_100": int(np.sum(estimate > 100)),
    }


def target_region_masks(y: Any) -> dict[str, np.ndarray]:
    actual = np.asarray(y, dtype=float)
    return {
        "region_low": actual < (100 / 3),
        "region_middle": (actual >= (100 / 3)) & (actual < (200 / 3)),
        "region_high": actual >= (200 / 3),
    }


def summarize(frame: pd.DataFrame, pred: Any, *, include_impact: bool = True) -> dict[str, Any]:
    """Summarize raw/clipped predictions, cycles, EF, target regions, tails and impact."""
    required = {"threat_rank", "cycle_code", "entry_friction", "signal_stream", "impact_load"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"frame is missing diagnostic columns: {missing}")
    actual, estimate = _aligned(frame["threat_rank"], pred)
    if len(actual) != len(frame):
        raise ValueError("prediction length differs from frame")

    cycles = frame["cycle_code"].to_numpy()
    per_cycle = []
    for cycle in sorted(pd.unique(cycles)):
        mask = cycles == cycle
        per_cycle.append({"cycle": int(cycle), **metric_values(actual[mask], estimate[mask])})
    nonempty_cycle_maes = [item["mae"] for item in per_cycle if item["mae"] is not None]

    region_masks = target_region_masks(actual)
    masks = {
        "EF0": frame["entry_friction"].eq(0).to_numpy(),
        "EF1": frame["entry_friction"].eq(1).to_numpy(),
        **region_masks,
        "tail_low_le5": actual <= 5,
        "tail_high_ge95": actual >= 95,
        "stream_missing": frame["signal_stream"].map(_is_missing).to_numpy(dtype=bool),
    }
    masks["stream_missing"] |= frame["signal_stream"].map(
        lambda value: isinstance(value, str) and not value.strip()
    ).to_numpy(dtype=bool)
    masks["stream_present"] = ~masks["stream_missing"]
    subgroups = {name: metric_values(actual[mask], estimate[mask]) for name, mask in masks.items()}

    impact_groups: dict[str, Any] = {}
    if include_impact:
        impacts = frame["impact_load"].to_numpy()
        for value in sorted(pd.unique(impacts)):
            mask = impacts == value
            key = repr(float(value))
            by_region = {
                name: metric_values(actual[mask & region_mask], estimate[mask & region_mask])
                for name, region_mask in region_masks.items()
            }
            impact_groups[key] = {"impact_load": float(value), "metrics": metric_values(actual[mask], estimate[mask]),
                                  "by_target_region": by_region}

    return {
        "raw": metric_values(actual, estimate),
        "clipped": metric_values(actual, np.clip(estimate, 0, 100)),
        "equal_cycle_mae": float(np.mean(nonempty_cycle_maes)) if nonempty_cycle_maes else None,
        "per_cycle": per_cycle,
        "subgroups": subgroups,
        "impact_groups": impact_groups,
    }


def paired(frame: pd.DataFrame, candidate: Any, control: Any) -> dict[str, Any]:
    """Paired errors on identical rows; negative delta_mae favors candidate."""
    if "threat_rank" not in frame or "cycle_code" not in frame:
        raise ValueError("frame must contain threat_rank and cycle_code")
    actual = np.asarray(frame["threat_rank"], dtype=float)
    cand = np.asarray(candidate, dtype=float)
    ctrl = np.asarray(control, dtype=float)
    actual, cand = _aligned(actual, cand)
    _, ctrl = _aligned(actual, ctrl)
    cand_abs, ctrl_abs = np.abs(cand - actual), np.abs(ctrl - actual)
    delta = cand_abs - ctrl_abs
    win, loss = delta < -TOLERANCE, delta > TOLERANCE
    result: dict[str, Any] = {
        "n": len(actual),
        "delta_mae": float(np.mean(delta)) if len(delta) else None,
        "candidate_signed_error": metric_values(actual, cand)["signed_error"],
        "control_signed_error": metric_values(actual, ctrl)["signed_error"],
        "signed_error_delta": (None if not len(actual) else
                               metric_values(actual, cand)["signed_error"] -
                               metric_values(actual, ctrl)["signed_error"]),
        "candidate_win_rate": float(np.mean(win)) if len(delta) else None,
        "control_win_rate": float(np.mean(loss)) if len(delta) else None,
        "tie_rate": float(np.mean(~win & ~loss)) if len(delta) else None,
        "per_cycle": [],
    }
    cycles = frame["cycle_code"].to_numpy()
    for cycle in sorted(pd.unique(cycles)):
        mask = cycles == cycle
        result["per_cycle"].append({
            "cycle": int(cycle),
            "candidate": metric_values(actual[mask], cand[mask]),
            "control": metric_values(actual[mask], ctrl[mask]),
            "delta_mae": float(np.mean(delta[mask])),
        })
    return result


def _canonical_value(value: Any) -> str:
    if _is_missing(value):
        return "null"
    if isinstance(value, (bool, np.bool_)):
        return "bool:1" if value else "bool:0"
    if isinstance(value, (int, np.integer)):
        return f"int:{int(value)}"
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return "float:" + (number.hex() if math.isfinite(number) else repr(number))
    return "str:" + str(value)


def _feature_rows_digest(frame: pd.DataFrame, rows: np.ndarray) -> str:
    """Hash ordered row positions and predictors only, never target or raw IDs."""
    digest = hashlib.sha256()
    columns = [name for name in EXPANDED_FEATURES if name in frame.columns]
    digest.update(json.dumps(columns, ensure_ascii=False, separators=(",", ":")).encode())
    for position in rows:
        digest.update(int(position).to_bytes(8, "big", signed=False))
        for name in columns:
            value = _canonical_value(frame.iloc[int(position)][name])
            encoded = value.encode("utf-8", errors="surrogatepass")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


def _row_positions(value: Any, label: str, row_count: int) -> np.ndarray:
    """Accept integer position arrays only; never truncate or coerce selectors."""
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.dtype.kind not in "iu":
        raise ValueError(f"{label} must be a one-dimensional integer-position array")
    if raw.dtype.kind == "u" and len(raw) and np.any(raw > np.iinfo(np.int64).max):
        raise ValueError(f"{label} contains a position outside the supported integer range")
    positions = raw.astype(np.int64, copy=False)
    if np.any(positions < 0) or np.any(positions >= row_count):
        raise ValueError(f"{label} contains a position outside the frame")
    return positions


def _support_stats(train: pd.Series, validation: pd.Series, parser: str) -> dict[str, Any]:
    if parser == "numeric":
        train_values = _number_array(train)[0]
        val_values = _number_array(validation)[0]
    elif parser == "mixed_count":
        train_values = _mixed_count_arrays(train)[0]
        val_values = _mixed_count_arrays(validation)[0]
    else:
        raise ValueError(f"unsupported numeric parser: {parser}")
    train_values = train_values[np.isfinite(train_values)]
    val_values = val_values[np.isfinite(val_values)]
    if not len(train_values):
        lower = upper = None
        outside = 0
    else:
        lower, upper = float(train_values.min()), float(train_values.max())
        outside = int(np.sum((val_values < lower) | (val_values > upper)))
    return {"fit_nonmissing_n": int(len(train_values)), "validation_nonmissing_n": int(len(val_values)),
            "fit_min": lower, "fit_max": upper, "validation_outside_fit_range_n": outside,
            "validation_outside_fit_range_rate": float(outside / len(val_values)) if len(val_values) else None}


def history_summary(frame: pd.DataFrame, fit_rows: Any, val_rows: Any) -> dict[str, Any]:
    """Describe past fit versus validation support without accessing target labels."""
    n = len(frame)
    fit = _row_positions(fit_rows, "fit_rows", n)
    val = _row_positions(val_rows, "val_rows", n)
    if fit.ndim != 1 or val.ndim != 1 or not len(fit) or not len(val):
        raise ValueError("fit_rows and val_rows must be nonempty one-dimensional positions")
    if len(np.unique(fit)) != len(fit) or len(np.unique(val)) != len(val) or np.intersect1d(fit, val).size:
        raise ValueError("fit and validation row positions must be unique and disjoint")
    if "cycle_code" not in frame or "entry_friction" not in frame:
        raise ValueError("frame must contain cycle_code and entry_friction")
    fit_cycles = frame.iloc[fit]["cycle_code"]
    val_cycles = frame.iloc[val]["cycle_code"]
    if fit_cycles.max() >= val_cycles.min():
        raise ValueError("fit cycles must be strictly earlier than validation cycles")

    cycle_counts = lambda rows: {str(int(c)): int(np.sum(frame.iloc[rows]["cycle_code"].to_numpy() == c))
                                 for c in sorted(pd.unique(frame.iloc[rows]["cycle_code"]))}
    ef_counts = lambda rows: {str(value): int((frame.iloc[rows]["entry_friction"] == value).sum())
                              for value in sorted(pd.unique(frame.iloc[rows]["entry_friction"]))}

    categories: dict[str, Any] = {}
    for name in CATEGORICAL_FEATURES:
        fit_raw, val_raw = frame.iloc[fit][name], frame.iloc[val][name]
        fit_cat, val_cat = _category_array(fit_raw), _category_array(val_raw)
        fit_known = {str(value) for value in fit_cat if not _is_missing(value)}
        val_present = [str(value) for value in val_cat if not _is_missing(value)]
        categories[name] = {
            "fit_rows": len(fit), "fit_missing_n": int(sum(_is_missing(v) for v in fit_cat)),
            "fit_nonmissing_n": int(sum(not _is_missing(v) for v in fit_cat)),
            "fit_unique_nonmissing": len(fit_known), "validation_rows": len(val),
            "validation_missing_n": int(sum(_is_missing(v) for v in val_cat)),
            "validation_nonmissing_n": len(val_present),
            "validation_unknown_n": int(sum(value not in fit_known for value in val_present)),
        }

    numeric: dict[str, Any] = {}
    for name in NUMERIC_FEATURES:
        numeric[name] = _support_stats(frame.iloc[fit][name], frame.iloc[val][name], "numeric")
    for name in MIXED_COUNT_FEATURES:
        numeric[name] = _support_stats(frame.iloc[fit][name], frame.iloc[val][name], "mixed_count")
    value, unknown, missing = _telemetry_arrays(frame.iloc[fit][TELEMETRY_FEATURE])
    val_value, val_unknown, val_missing = _telemetry_arrays(frame.iloc[val][TELEMETRY_FEATURE])
    numeric[TELEMETRY_FEATURE] = {
        "fit_nonmissing_n": int(np.isfinite(value).sum()), "fit_unknown_n": int(unknown.sum()),
        "fit_missing_n": int(missing.sum()), "validation_nonmissing_n": int(np.isfinite(val_value).sum()),
        "validation_unknown_n": int(val_unknown.sum()), "validation_missing_n": int(val_missing.sum()),
    }
    return {
        "fit_rows": len(fit), "validation_rows": len(val),
        "fit_cycle_range": [int(fit_cycles.min()), int(fit_cycles.max())],
        "validation_cycle_range": [int(val_cycles.min()), int(val_cycles.max())],
        "fit_cycle_counts": cycle_counts(fit), "validation_cycle_counts": cycle_counts(val),
        "fit_entry_friction_counts": ef_counts(fit), "validation_entry_friction_counts": ef_counts(val),
        "category_support": categories, "numeric_support": numeric,
        "fit_source_rows_sha256": _feature_rows_digest(frame, fit),
        "validation_source_rows_sha256": _feature_rows_digest(frame, val),
        "target_labels_read": False, "raw_record_ids_written": False,
    }


def descriptive_test_risk(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    """Compare supplied predictor features only; never fits, scores, or predicts."""
    required = set(EXPANDED_FEATURES)
    for label, frame in (("train", train), ("test", test)):
        absent = sorted(required.difference(frame.columns))
        if absent:
            raise ValueError(f"{label} is missing eligible predictor columns: {absent}")
    features: dict[str, Any] = {}
    for name in EXPANDED_FEATURES:
        left, right = train[name], test[name]
        left_missing = left.map(_is_missing).to_numpy(dtype=bool)
        right_missing = right.map(_is_missing).to_numpy(dtype=bool)
        left_blank = left.map(lambda value: isinstance(value, str) and not value.strip()).to_numpy(dtype=bool)
        right_blank = right.map(lambda value: isinstance(value, str) and not value.strip()).to_numpy(dtype=bool)
        left_missing |= left_blank
        right_missing |= right_blank
        entry: dict[str, Any] = {
            "train_rows": len(left), "test_rows": len(right),
            "train_missing_n": int(left_missing.sum()), "test_missing_n": int(right_missing.sum()),
            "train_missing_rate": float(left_missing.mean()) if len(left) else None,
            "test_missing_rate": float(right_missing.mean()) if len(right) else None,
        }
        if name in NUMERIC_FEATURES:
            a, b = _number_array(left), _number_array(right)
            av, bv = a[0][np.isfinite(a[0])], b[0][np.isfinite(b[0])]
            entry.update({"train_invalid_format_n": int(a[2].sum()), "test_invalid_format_n": int(b[2].sum()),
                          "train_quantiles": {str(q): float(np.quantile(av, q)) for q in QUANTILES} if len(av) else None,
                          "test_quantiles": {str(q): float(np.quantile(bv, q)) for q in QUANTILES} if len(bv) else None,
                          "train_min": float(av.min()) if len(av) else None, "train_max": float(av.max()) if len(av) else None,
                          "test_outside_train_range_n": int(np.sum((bv < av.min()) | (bv > av.max()))) if len(av) and len(bv) else 0})
        elif name in MIXED_COUNT_FEATURES:
            a, b = _mixed_count_arrays(left), _mixed_count_arrays(right)
            av, bv = a[0][np.isfinite(a[0])], b[0][np.isfinite(b[0])]
            entry.update({"train_parse_fail_n": int(a[3].sum()), "test_parse_fail_n": int(b[3].sum()),
                          "train_approximate_n": int(a[1].sum()), "test_approximate_n": int(b[1].sum()),
                          "train_suffix_present_n": int(a[2].sum()), "test_suffix_present_n": int(b[2].sum()),
                          "train_quantiles": {str(q): float(np.quantile(av, q)) for q in QUANTILES} if len(av) else None,
                          "test_quantiles": {str(q): float(np.quantile(bv, q)) for q in QUANTILES} if len(bv) else None,
                          "train_min": float(av.min()) if len(av) else None, "train_max": float(av.max()) if len(av) else None,
                          "test_outside_train_range_n": int(np.sum((bv < av.min()) | (bv > av.max()))) if len(av) and len(bv) else 0})
        elif name in CATEGORICAL_FEATURES:
            a, b = _category_array(left), _category_array(right)
            known = {str(value) for value in a if not _is_missing(value)}
            present = [str(value) for value in b if not _is_missing(value)]
            entry.update({"train_unique_nonmissing": len(known), "test_unique_nonmissing": len(set(present)),
                          "test_nonmissing_n": len(present),
                          "test_unknown_n": int(sum(value not in known for value in present)),
                          "test_unknown_rate_nonmissing": float(sum(value not in known for value in present) / len(present)) if present else None})
        elif name == TELEMETRY_FEATURE:
            a, b = _telemetry_arrays(left), _telemetry_arrays(right)
            entry.update({"train_unrecognized_n": int(a[1].sum()), "test_unrecognized_n": int(b[1].sum()),
                          "train_missing_parser_n": int(a[2].sum()), "test_missing_parser_n": int(b[2].sum())})
        features[name] = entry
    train_cycles = pd.to_numeric(train["cycle_code"], errors="coerce").dropna()
    test_cycles = pd.to_numeric(test["cycle_code"], errors="coerce").dropna()
    if len(train_cycles) and len(test_cycles):
        lower, upper = float(train_cycles.min()), float(train_cycles.max())
        outside = ((test_cycles < lower) | (test_cycles > upper)).to_numpy()
        cycle_gap = {"train_min": lower, "train_max": upper,
                     "test_min": float(test_cycles.min()), "test_max": float(test_cycles.max()),
                     "test_outside_train_range_n": int(outside.sum()),
                     "test_outside_train_range_rate": float(outside.mean())}
    else:
        cycle_gap = None
    return {"features": features, "cycle_gap": cycle_gap,
            "operation": "descriptive supplied-feature comparison only",
            "test_labels_read": False, "fit_performed": False, "predictions_made": False,
            "raw_values_categories_and_ids_persisted": False}
