"""Pure helper functions for the R19 prequential EF median-offset route."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def parse_entry_friction(values: Any) -> np.ndarray:
    """Map official values to 0/1 and everything else to the unknown group."""
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=np.float64)
    parsed = np.full(len(numeric), -1, dtype=np.int8)
    parsed[np.isfinite(numeric) & (numeric == 0.0)] = 0
    parsed[np.isfinite(numeric) & (numeric == 1.0)] = 1
    return parsed


def fit_ef_median_offsets(
    target: Any,
    prediction: Any,
    entry_friction: Any,
    *,
    minimum_support: int = 200,
) -> tuple[dict[int, float], dict[str, dict[str, Any]]]:
    """Fit the frozen half-median residual offset for EF0/EF1, else zero."""
    y = np.asarray(target, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    ef = parse_entry_friction(entry_friction)
    if y.ndim != 1 or p.shape != y.shape or ef.shape != y.shape or not len(y):
        raise ValueError("R19 target, prediction, and entry_friction must align")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("R19 calibration target and prediction must be finite")
    if np.any((y < 0.0) | (y > 100.0)) or np.any((p < 0.0) | (p > 100.0)):
        raise ValueError("R19 calibration values must lie in [0, 100]")
    if int(minimum_support) < 1:
        raise ValueError("R19 minimum support must be positive")

    offsets: dict[int, float] = {}
    support: dict[str, dict[str, Any]] = {}
    residual = y - p
    for group in (0, 1):
        mask = ef == group
        n = int(mask.sum())
        supported = n >= int(minimum_support)
        raw = 0.5 * float(np.median(residual[mask])) if supported else 0.0
        offset = float(np.clip(raw, -2.0, 2.0))
        offsets[group] = offset
        support[str(group)] = {
            "rows": n,
            "supported": supported,
            "median_y_minus_prediction": float(np.median(residual[mask])) if n else None,
            "raw_half_median_offset": raw,
            "offset": offset,
        }
    support["unknown"] = {
        "rows": int(np.sum(ef < 0)),
        "supported": False,
        "median_y_minus_prediction": None,
        "raw_half_median_offset": 0.0,
        "offset": 0.0,
    }
    return offsets, support


def apply_ef_median_offsets(
    prediction: Any,
    entry_friction: Any,
    offsets: dict[int, float],
) -> np.ndarray:
    """Apply the fitted group offset and clamp the final prediction to [0,100]."""
    p = np.asarray(prediction, dtype=np.float64)
    ef = parse_entry_friction(entry_friction)
    if p.ndim != 1 or ef.shape != p.shape or not len(p) or not np.isfinite(p).all():
        raise ValueError("R19 prediction and entry_friction must be finite, aligned vectors")
    if np.any((p < 0.0) | (p > 100.0)):
        raise ValueError("R19 baseline prediction must lie in [0, 100]")
    if set(offsets) != {0, 1} or any(not np.isfinite(value) for value in offsets.values()):
        raise ValueError("R19 requires finite offsets for EF0 and EF1")
    if any(abs(float(value)) > 2.0 for value in offsets.values()):
        raise ValueError("R19 EF offsets must already be clipped to [-2, 2]")
    adjustment = np.zeros(len(p), dtype=np.float64)
    for group in (0, 1):
        adjustment[ef == group] = float(offsets[group])
    return np.clip(p + adjustment, 0.0, 100.0)

