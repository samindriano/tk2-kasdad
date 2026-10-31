"""Descriptive analysis helpers for Classification stream availability.

All contrasts are descriptive. Quantile boundaries are computed from
``text_load`` only; neither availability nor the outcome determines bins.
"""
from __future__ import annotations

from itertools import product
from typing import Iterable

import numpy as np
import pandas as pd

AVAILABILITY_ORDER = ("present", "missing")
PERIODS = ((47, 57, "47–57"), (58, 68, "58–68"), (69, 79, "69–79"))


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return a Wilson 95% interval for one binomial proportion."""
    if n <= 0 or successes < 0 or successes > n:
        raise ValueError("Wilson interval requires 0 <= successes <= n and n > 0")
    p = successes / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half_width = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return float(max(0.0, center - half_width)), float(min(1.0, center + half_width))


def prepare_availability_frame(
    train: pd.DataFrame,
    stream_missing: Iterable[bool],
) -> pd.DataFrame:
    """Validate official raw columns and create a small analysis-only frame."""
    required = {"record_id", "cycle_code", "signal_stream", "text_load", "artifact_signal"}
    missing_columns = sorted(required - set(train.columns))
    if missing_columns:
        raise ValueError(f"Official Classification Train lacks columns: {missing_columns}")
    mask = np.asarray(list(stream_missing), dtype=bool)
    if len(mask) != len(train):
        raise ValueError("stream_missing must have one value per training row")
    if train["record_id"].isna().any() or train["record_id"].duplicated().any():
        raise ValueError("record_id must be complete and unique")
    target = pd.to_numeric(train["artifact_signal"], errors="raise")
    if target.isna().any() or not target.isin([0, 1]).all():
        raise ValueError("artifact_signal must be complete binary 0/1")
    cycle = pd.to_numeric(train["cycle_code"], errors="raise")
    if cycle.isna().any() or not np.equal(cycle, np.floor(cycle)).all():
        raise ValueError("cycle_code must be complete integer values")
    text_load = pd.to_numeric(train["text_load"], errors="raise")
    if text_load.isna().any() or not np.isfinite(text_load).all():
        raise ValueError("text_load must be complete and finite")

    period_labels = np.select(
        [cycle.between(start, end) for start, end, _ in PERIODS],
        [label for _, _, label in PERIODS],
        default="outside expected cycles",
    )
    if np.any(period_labels == "outside expected cycles"):
        raise ValueError("Train contains cycle_code outside the documented periods 47–79")
    return pd.DataFrame(
        {
            "record_id": train["record_id"].to_numpy(),
            "cycle_code": cycle.astype(int).to_numpy(),
            "period": period_labels,
            "text_load": text_load.astype(float).to_numpy(),
            "artifact_signal": target.astype(int).to_numpy(),
            "availability": np.where(mask, "missing", "present"),
        },
        index=train.index,
    )


def prevalence_table(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Return denominators, positive counts, and proportions by requested groups."""
    if not group_columns:
        raise ValueError("At least one grouping column is required")
    grouped = (
        frame.groupby(group_columns, observed=True, dropna=False)
        .agg(n=("artifact_signal", "size"), positive=("artifact_signal", "sum"))
        .reset_index()
    )
    grouped["prevalence"] = grouped["positive"] / grouped["n"]
    return grouped


def add_global_quantile_bins(frame: pd.DataFrame, q: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Add global, outcome-blind quantile bins of text_load and return edges."""
    if q < 2:
        raise ValueError("q must be at least 2")
    if frame["text_load"].isna().any():
        raise ValueError("text_load cannot be missing when creating bins")
    labels, edges = pd.qcut(frame["text_load"], q=q, labels=False, retbins=True, duplicates="raise")
    result = frame.copy()
    result[f"text_load_q{q}"] = labels.astype(int)
    return result, np.asarray(edges, dtype=float)


def standardized_contrast(
    frame: pd.DataFrame,
    strata: list[str],
    *,
    min_support: int = 20,
    availability_column: str = "availability",
    outcome_column: str = "artifact_signal",
) -> tuple[dict, pd.DataFrame]:
    """Standardize both availability rates to the same pooled supported-cell mix.

    Cells qualify only if both availability groups have at least ``min_support``
    observations. A common cell weight is its combined denominator divided by
    all rows in supported cells. The returned summary keeps group-specific
    support and coverage denominators explicit.
    """
    if not strata or min_support < 1:
        raise ValueError("strata must be non-empty and min_support positive")
    needed = set(strata) | {availability_column, outcome_column}
    if needed - set(frame.columns):
        raise ValueError(f"Missing columns for standardization: {sorted(needed - set(frame.columns))}")
    if frame[list(strata) + [availability_column, outcome_column]].isna().any().any():
        raise ValueError("strata, availability, and outcome must be complete")
    if not frame[availability_column].isin(AVAILABILITY_ORDER).all():
        raise ValueError("availability values must be 'present' or 'missing'")
    if not frame[outcome_column].isin([0, 1]).all():
        raise ValueError("outcome must be binary 0/1")

    aggregation = (
        frame.groupby(strata + [availability_column], observed=True, dropna=False)[outcome_column]
        .agg(n="size", positive="sum")
        .reset_index()
    )
    domains = []
    for column in strata:
        values = list(pd.unique(frame[column]))
        try:
            values = sorted(values)
        except TypeError:
            pass
        domains.append(values)
    # Build lookup without relying on pandas' namedtuple field behavior for
    # arbitrary column names such as a caller-provided stratum label.
    indexed = {}
    for _, row in aggregation.iterrows():
        key = tuple(row[column] for column in strata)
        indexed[(key, row[availability_column])] = (int(row["n"]), int(row["positive"]))

    rows = []
    for cell_key in product(*domains):
        cell = dict(zip(strata, cell_key))
        state = {}
        for availability in AVAILABILITY_ORDER:
            n, positive = indexed.get((cell_key, availability), (0, 0))
            state[availability] = {
                "n": n,
                "positive": positive,
                "prevalence": positive / n if n else np.nan,
            }
        combined_n = state["present"]["n"] + state["missing"]["n"]
        supported = all(state[name]["n"] >= min_support for name in AVAILABILITY_ORDER)
        rows.append(
            {
                **cell,
                "n_present": state["present"]["n"],
                "positive_present": state["present"]["positive"],
                "prevalence_present": state["present"]["prevalence"],
                "n_missing": state["missing"]["n"],
                "positive_missing": state["missing"]["positive"],
                "prevalence_missing": state["missing"]["prevalence"],
                "combined_n": combined_n,
                "supported": supported,
            }
        )
    cells = pd.DataFrame(rows)
    supported = cells.loc[cells["supported"]].copy()
    if supported.empty:
        raise ValueError("No strata meet minimum support for both availability groups")
    supported_rows = int(supported["combined_n"].sum())
    weighted = {}
    for availability in AVAILABILITY_ORDER:
        weighted[availability] = float(
            np.sum(
                (supported["combined_n"] / supported_rows)
                * supported[f"prevalence_{availability}"].to_numpy(dtype=float)
            )
        )
    total_by_group = {
        name: int(frame[availability_column].eq(name).sum()) for name in AVAILABILITY_ORDER
    }
    covered_by_group = {
        "present": int(supported["n_present"].sum()),
        "missing": int(supported["n_missing"].sum()),
    }
    summary = {
        "strata": strata,
        "minimum_n_per_availability_per_cell": int(min_support),
        "cells_total": int(len(cells)),
        "cells_supported": int(supported.shape[0]),
        "supported_rows_total": supported_rows,
        "coverage_all_rows": float(supported_rows / len(frame)),
        "covered_rows_by_availability": covered_by_group,
        "total_rows_by_availability": total_by_group,
        "coverage_by_availability": {
            name: float(covered_by_group[name] / total_by_group[name])
            for name in AVAILABILITY_ORDER
        },
        "standardized_prevalence": weighted,
        "standardized_missing_minus_present_pp": float(100 * (weighted["missing"] - weighted["present"])),
        "standardization_weight": "combined cell n divided by all rows in supported cells; same weights for both groups",
        "support_rule": "each supported cell has at least min_support rows in both availability groups",
    }
    return summary, cells
