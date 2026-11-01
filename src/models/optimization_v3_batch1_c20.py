"""Pure route helpers for V3 C20 cache-only classification recomposition."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def fit_only_q80(text_load: Any, cycle_code: Any, cutoff: int) -> float:
    """Return the inherited all-fit-row q80 rule, before evaluation labels."""
    values = pd.to_numeric(pd.Series(text_load), errors="coerce").to_numpy(dtype=np.float64)
    cycles = pd.to_numeric(pd.Series(cycle_code), errors="raise").to_numpy(dtype=np.int64)
    if values.shape != cycles.shape or values.ndim != 1 or not len(values):
        raise ValueError("text_load and cycle_code must be aligned nonempty vectors")
    fit = cycles <= int(cutoff)
    if not fit.any() or not np.isfinite(values[fit]).all():
        raise ValueError("All fit-row text_load values must be finite")
    # Match the inherited source rule: np.quantile(text_load[cycle <= fit_end], .80).
    return float(np.quantile(values[fit], 0.80, method="linear"))


def compose_c20_predictions(
    *,
    stream_missing: Any,
    text_load: Any,
    q80: float,
    sparse25_probability: Any,
    c2b_probability: Any,
    cb_probability: Any,
) -> tuple[np.ndarray, dict[str, int]]:
    """Use CB for missing streams, sparse25 strictly above q80, C2_B otherwise."""
    missing = np.asarray(stream_missing, dtype=bool)
    volume = pd.to_numeric(pd.Series(text_load), errors="coerce").to_numpy(dtype=np.float64)
    sparse25 = np.asarray(sparse25_probability, dtype=np.float64)
    c2b = np.asarray(c2b_probability, dtype=np.float64)
    cb = np.asarray(cb_probability, dtype=np.float64)
    arrays = (missing, volume, sparse25, c2b, cb)
    if any(array.ndim != 1 or array.shape != missing.shape for array in arrays):
        raise ValueError("C20 routing arrays must be one-dimensional and row-aligned")
    if not np.isfinite(float(q80)) or not np.isfinite(volume).all():
        raise ValueError("C20 q80 and text_load values must be finite")
    if any(not np.isfinite(probability).all() for probability in (sparse25, c2b, cb)):
        raise ValueError("C20 prediction caches must be finite")
    if any(np.any((probability < 0.0) | (probability > 1.0))
           for probability in (sparse25, c2b, cb)):
        raise ValueError("C20 prediction caches must lie in [0, 1]")
    high = (~missing) & (volume > float(q80))
    low = (~missing) & ~high
    result = np.empty(len(missing), dtype=np.float64)
    result[missing] = cb[missing]
    result[high] = sparse25[high]
    result[low] = c2b[low]
    return result, {
        "missing_cb": int(missing.sum()),
        "present_high_sparse25": int(high.sum()),
        "present_other_c2b": int(low.sum()),
    }


def missing_stream_mask(values: Any) -> np.ndarray:
    """Match project availability routing: null or whitespace-only is missing."""
    series = pd.Series(values, dtype="object")
    missing = series.isna().to_numpy(dtype=bool)
    nonnull = np.flatnonzero(~missing)
    if len(nonnull):
        missing[nonnull] = series.iloc[nonnull].map(lambda value: not str(value).strip()).to_numpy(dtype=bool)
    return missing

