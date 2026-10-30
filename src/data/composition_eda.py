"""Conditional and composition diagnostics for the TP2 regression target.

This module describes how the observed association between ``impact_load`` and
``threat_rank`` varies across ordered cycle blocks. It does not fit models,
produce predictions, or make causal claims. Exact impact and impact-by-friction
support are selected using counts only; target values do not choose weights or
support.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from src.utils.paths import data_root, task_files


REQUIRED = ("cycle_code", "impact_load", "entry_friction", "threat_rank")
PRIMARY_THRESHOLD = 30
SUPPORT_THRESHOLDS = (20, 30, 50, 100)
EARLY = "period_1"
LATE = "period_3"


def _finite(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _target_stats(values: pd.Series) -> dict[str, Any]:
    vals = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return {"n": 0, "mean": None, "median": None, "p10": None, "p90": None,
                "iqr": None, "tail_le_5_count": 0, "tail_ge_95_count": 0}
    p10, p25, median, p75, p90 = vals.quantile([.10, .25, .50, .75, .90]).tolist()
    return {
        "n": int(len(vals)), "mean": _finite(vals.mean()), "median": _finite(median),
        "p10": _finite(p10), "p90": _finite(p90), "iqr": _finite(p75 - p25),
        "tail_le_5_count": int(vals.le(5).sum()),
        "tail_ge_95_count": int(vals.ge(95).sum()),
    }


def _corr(x: pd.Series, y: pd.Series) -> dict[str, Any]:
    pair = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    result: dict[str, Any] = {"n": int(len(pair)), "pearson_r": None}
    if len(pair) >= 2 and pair.x.nunique() > 1 and pair.y.nunique() > 1:
        result["pearson_r"] = _finite(pearsonr(pair.x, pair.y).statistic)
    return result


def _weighted_corr(frame: pd.DataFrame, weights: np.ndarray) -> dict[str, Any]:
    x = frame["impact_load"].to_numpy(dtype=float)
    y = frame["threat_rank"].to_numpy(dtype=float)
    w = np.asarray(weights, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x, y, w = x[keep], y[keep], w[keep]
    if not len(w):
        return {"n_rows": 0, "effective_n": None, "pearson_r": None}
    w = w / w.sum()
    x_mean, y_mean = np.sum(w * x), np.sum(w * y)
    x_centered, y_centered = x - x_mean, y - y_mean
    covariance = np.sum(w * x_centered * y_centered)
    x_sd = np.sqrt(np.sum(w * x_centered ** 2))
    y_sd = np.sqrt(np.sum(w * y_centered ** 2))
    r = covariance / (x_sd * y_sd) if x_sd > 0 and y_sd > 0 else None
    return {"n_rows": int(len(w)), "effective_n": _finite(1.0 / np.sum(w ** 2)),
            "pearson_r": _finite(r)}


def _period_blocks(cycles: pd.Series) -> tuple[pd.Series, list[dict[str, Any]]]:
    unique = np.sort(pd.to_numeric(cycles, errors="coerce").dropna().unique())
    if len(unique) < 3:
        raise ValueError("At least three unique cycle_code values are required to form three blocks.")
    chunks = np.array_split(unique, 3)
    mapping: dict[float, str] = {}
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        label = f"period_{i}"
        mapping.update({value: label for value in chunk})
        blocks.append({"label": label, "cycle_min": int(chunk.min()), "cycle_max": int(chunk.max()),
                       "unique_cycles": int(len(chunk))})
    return pd.to_numeric(cycles, errors="coerce").map(mapping), blocks


def _support(counts: pd.DataFrame, levels: list[float], periods: list[str], threshold: int) -> list[float]:
    supported = []
    for level in levels:
        if all(int(counts.get((level, period), 0)) >= threshold for period in periods):
            supported.append(float(level))
    return supported


def _impact_period_count_table(frame: pd.DataFrame) -> dict[tuple[float, str], int]:
    grouped = frame.groupby(["impact_load", "period"], observed=True).size()
    return {(float(level), str(period)): int(n) for (level, period), n in grouped.items()}


def _joint_period_count_table(frame: pd.DataFrame) -> dict[tuple[float, int, str], int]:
    grouped = frame.groupby(["impact_load", "entry_friction", "period"], observed=True).size()
    return {(float(level), int(friction), str(period)): int(n)
            for (level, friction, period), n in grouped.items()}


def _safe_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def _make_figures(frame: pd.DataFrame, blocks: list[dict[str, Any]], impact_rows: list[dict[str, Any]],
                  joint_rows: list[dict[str, Any]], decomposition: dict[str, Any],
                  output_dir: Path, common_impact: list[float], common_joint: list[tuple[float, int]]) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [item["label"] for item in blocks]
    block_ticks = [f"{item['cycle_min']}–{item['cycle_max']}\n(n={int((frame['period'] == item['label']).sum()):,})"
                   for item in blocks]
    written = []

    # Same-impact distributions. The eight levels are chosen by minimum period
    # count among the target-blind common-support set.
    totals = {row["impact_load"]: min(row["periods"][p]["n"] for p in labels)
              for row in impact_rows}
    selected = sorted(totals, key=lambda x: (-totals[x], x))[:8]
    if selected:
        ncols = 2
        nrows = int(np.ceil(len(selected) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.5 * nrows), squeeze=False,
                                 sharex=True, sharey=True)
        colors = ("#2563eb", "#d97706", "#059669")
        for ax, level in zip(axes.flat, selected):
            row = next(item for item in impact_rows if item["impact_load"] == level)
            for i, period in enumerate(labels):
                stats = row["periods"][period]
                median = stats["median"]
                if median is None:
                    continue
                lo, hi = stats["p10"], stats["p90"]
                ax.errorbar(i, median, yerr=[[max(0, median - lo)], [max(0, hi - median)]],
                            color=colors[i], marker="o", capsize=3, linewidth=1.3,
                            label=period if level == selected[0] else None)
                ax.annotate(f"n={stats['n']:,}", (i, hi), xytext=(0, 4),
                            textcoords="offset points", ha="center", fontsize=8, color=colors[i])
            ax.set_title(f"impact_load = {level:.4f}")
            ax.set_ylim(0, 104)
            ax.grid(axis="y", alpha=.25)
            ax.set_xticks(range(len(labels)), block_ticks)
            ax.set_ylabel("threat_rank")
        for ax in axes.flat[len(selected):]:
            ax.remove()
        fig.suptitle("Threat-rank distributions at the same exact impact level\n"
                     "Median with P10–P90; selected by minimum period count, not target", y=1.01)
        handles, legend_labels = axes.flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, [f"{blocks[i]['cycle_min']}–{blocks[i]['cycle_max']}"
                                 for i in range(len(labels))], loc="lower center", ncol=3,
                       bbox_to_anchor=(.5, -.005), frameon=False)
        fig.tight_layout(rect=(0, .035, 1, .97))
        path = output_dir / "composition_same_impact_distributions.png"
        fig.savefig(path, dpi=170, bbox_inches="tight")
        plt.close(fig)
        written.append(path.name)

    # Entry-friction composition within common impact values.
    mix_levels = sorted(common_impact, key=lambda x: (-totals.get(x, 0), x))[:12]
    if mix_levels:
        fig, ax = plt.subplots(figsize=(12, 5.5))
        width = .24
        colors = ("#2563eb", "#d97706", "#059669")
        for i, period in enumerate(labels):
            shares, ns = [], []
            for level in mix_levels:
                cell = next(item for item in impact_rows if item["impact_load"] == level)
                result = cell["periods"][period]
                shares.append(result["entry_friction_share"].get("1", 0.0))
                ns.append(result["n"])
            positions = np.arange(len(mix_levels)) + (i - 1) * width
            bars = ax.bar(positions, shares, width=width, color=colors[i],
                          label=f"{blocks[i]['cycle_min']}–{blocks[i]['cycle_max']}")
            for bar, n in zip(bars, ns):
                ax.annotate(f"n={n:,}", (bar.get_x() + bar.get_width()/2, bar.get_height()),
                            xytext=(0, 2), textcoords="offset points", ha="center",
                            va="bottom", fontsize=7, rotation=90)
        ax.set_xticks(np.arange(len(mix_levels)), [f"{x:.4f}" for x in mix_levels], rotation=35, ha="right")
        ax.set_xlabel("Exact impact_load (12 levels with largest minimum period count)")
        ax.set_ylabel("Share with entry_friction = 1")
        ax.set_ylim(0, 1.13)
        ax.set_title("Entry-friction composition at matched impact levels")
        ax.grid(axis="y", alpha=.22)
        ax.legend(frameon=False, ncol=3)
        fig.tight_layout()
        path = output_dir / "composition_entry_friction_by_impact.png"
        fig.savefig(path, dpi=170, bbox_inches="tight")
        plt.close(fig)
        written.append(path.name)

    # Mean decomposition on fixed common impact x friction cells.
    effects = decomposition.get("symmetric_decomposition", {})
    if effects and effects.get("composition_effect") is not None:
        comp = effects["composition_effect"]
        within = effects["within_stratum_effect"]
        total = effects["late_minus_early"]
        fig, ax = plt.subplots(figsize=(9, 5))
        names = ["Composition", "Within-stratum", "Total common-support mean change"]
        vals = [comp, within, total]
        colors = ["#7c3aed", "#0891b2", "#374151"]
        bars = ax.bar(names, vals, color=colors)
        ax.axhline(0, color="#333333", linewidth=.8)
        for bar, val in zip(bars, vals):
            ax.annotate(f"{val:+.3f}", (bar.get_x() + bar.get_width()/2, val),
                        xytext=(0, 4 if val >= 0 else -14), textcoords="offset points",
                        ha="center", fontsize=10)
        retained = effects.get("retained_rows", {})
        ax.set_title("Early-to-late mean change on exact impact × friction common support\n"
                     f"retained rows: {retained.get(EARLY, 0):,} → {retained.get(LATE, 0):,}")
        ax.set_ylabel("threat_rank points")
        ax.grid(axis="y", alpha=.22)
        fig.tight_layout()
        path = output_dir / "composition_mean_decomposition.png"
        fig.savefig(path, dpi=170, bbox_inches="tight")
        plt.close(fig)
        written.append(path.name)
    return written


def run_composition_eda(data_dir: str | Path | None = None,
                        output_dir: str | Path | None = None,
                        make_plots: bool = True) -> dict[str, Any]:
    """Run target-blind support selection and descriptive TP2 composition checks."""
    data_root_path = Path(data_dir).expanduser().resolve() if data_dir is not None else data_root()
    files = task_files("regression", root=data_root_path)
    if files["train"] is None:
        raise FileNotFoundError("Regression Train.csv not found under the configured data root.")
    train_path = files["train"]
    raw = pd.read_csv(train_path)
    missing_columns = [col for col in REQUIRED if col not in raw.columns]
    if missing_columns:
        raise ValueError(f"Regression Train.csv is missing required columns: {missing_columns}")

    numeric = raw.loc[:, list(REQUIRED)].apply(pd.to_numeric, errors="coerce")
    finite_mask = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    valid_friction = numeric["entry_friction"].isin([0.0, 1.0]).fillna(False).to_numpy()
    keep = finite_mask & valid_friction
    frame = numeric.loc[keep].copy()
    frame["entry_friction"] = frame["entry_friction"].astype(int)
    if frame.empty:
        raise ValueError("No finite rows with entry_friction coded 0 or 1 were found.")
    periods, blocks = _period_blocks(frame["cycle_code"])
    frame["period"] = periods.to_numpy()
    if frame["period"].isna().any():
        raise ValueError("Cycle block assignment failed for one or more analyzable rows.")
    labels = [b["label"] for b in blocks]
    full_period_n = {period: int((frame.period == period).sum()) for period in labels}
    impact_levels = sorted(float(x) for x in frame.impact_load.unique())

    impact_counts = _impact_period_count_table(frame)
    impact_count_frame = pd.Series(impact_counts, name="n")
    primary_impact = _support(impact_count_frame, impact_levels, labels, PRIMARY_THRESHOLD)
    support_sensitivity = []
    for threshold in SUPPORT_THRESHOLDS:
        supported = _support(impact_count_frame, impact_levels, labels, threshold)
        per_period = {}
        for period in labels:
            covered = sum(impact_counts.get((level, period), 0) for level in supported)
            per_period[period] = {"covered_rows": int(covered), "total_rows": full_period_n[period],
                                  "coverage": _finite(covered / full_period_n[period]) if full_period_n[period] else None,
                                  "excluded_rows": int(full_period_n[period] - covered)}
        total_covered = sum(value["covered_rows"] for value in per_period.values())
        total_rows = sum(full_period_n.values())
        support_sensitivity.append({
            "minimum_rows_per_impact_period": threshold,
            "supported_impact_level_count": len(supported),
            "supported_impact_levels": supported,
            "excluded_impact_level_count": len(impact_levels) - len(supported),
            "excluded_impact_levels": [value for value in impact_levels if value not in supported],
            "covered_rows_all_periods": int(total_covered), "total_rows_all_periods": int(total_rows),
            "coverage_all_periods": _finite(total_covered / total_rows) if total_rows else None,
            "excluded_rows_all_periods": int(total_rows - total_covered),
            "by_period": per_period,
        })

    impact_rows = []
    common_frame = frame[frame.impact_load.isin(primary_impact)]
    for level in primary_impact:
        per_period = {}
        for period in labels:
            cell = common_frame[(common_frame.impact_load == level) & (common_frame.period == period)]
            stats = _target_stats(cell.threat_rank)
            ef_counts = cell.entry_friction.value_counts().to_dict()
            total = len(cell)
            stats["entry_friction_counts"] = {str(v): int(ef_counts.get(v, 0)) for v in (0, 1)}
            stats["entry_friction_share"] = {
                str(v): _finite(ef_counts.get(v, 0) / total) if total else None for v in (0, 1)
            }
            per_period[period] = stats
        impact_rows.append({"impact_load": level, "periods": per_period})

    joint_counts = _joint_period_count_table(frame)
    joint_cells = sorted({(level, friction) for level, friction, _period in joint_counts
                          if level in primary_impact})
    common_joint = [(level, friction) for level, friction in joint_cells
                    if all(joint_counts.get((level, friction, period), 0) >= PRIMARY_THRESHOLD
                           for period in labels)]
    joint_rows = []
    for level, friction in joint_cells:
        per_period = {}
        for period in labels:
            cell = frame[(frame.impact_load == level) & (frame.entry_friction == friction) &
                         (frame.period == period)]
            total_impact_n = impact_counts.get((level, period), 0)
            stats = _target_stats(cell.threat_rank)
            stats["entry_friction"] = friction
            stats["share_within_impact_period"] = _finite(len(cell) / total_impact_n) if total_impact_n else None
            stats["meets_primary_support_threshold"] = bool(len(cell) >= PRIMARY_THRESHOLD)
            per_period[period] = stats
        joint_rows.append({"impact_load": level, "entry_friction": friction,
                           "common_support_all_periods": (level, friction) in common_joint,
                           "periods": per_period})

    # Descriptive standardized means at impact-only common support. Weights are
    # pooled row shares over the three periods; target is not used to choose them.
    impact_pooled_counts = {level: sum(impact_counts.get((level, period), 0) for period in labels)
                            for level in primary_impact}
    impact_denominator = sum(impact_pooled_counts.values())
    impact_weights = {level: impact_pooled_counts[level] / impact_denominator for level in primary_impact}
    standardized_by_period = {}
    for period in labels:
        level_means = {}
        for row in impact_rows:
            stat = row["periods"][period]
            level_means[row["impact_load"]] = stat["mean"]
        standardized_by_period[period] = _finite(sum(impact_weights[level] * level_means[level]
                                                      for level in primary_impact
                                                      if level_means[level] is not None))

    # Full, EF-stratified, and common-distribution-weighted Pearson association.
    correlations: dict[str, Any] = {"by_period": {}, "common_impact_ef_distribution_weighted": {}}
    for period in labels:
        period_frame = frame[frame.period == period]
        correlations["by_period"][period] = {
            "pooled": _corr(period_frame.impact_load, period_frame.threat_rank),
            "entry_friction_strata": {
                str(friction): _corr(cell.impact_load, cell.threat_rank)
                for friction, cell in period_frame.groupby("entry_friction", observed=True)
            },
        }
    common_keys = [f"{level:g}|{friction}" for level, friction in common_joint]
    common_joint_set = set(common_joint)
    row_cells = list(zip(frame.impact_load.astype(float), frame.entry_friction.astype(int)))
    common_frame = frame.loc[[cell in common_joint_set for cell in row_cells]].copy()
    cell_counts_by_period = {
        (level, friction, period): joint_counts.get((level, friction, period), 0)
        for level, friction in common_joint for period in labels
    }
    pooled_common_cell_counts = {
        (level, friction): sum(cell_counts_by_period[(level, friction, period)] for period in labels)
        for level, friction in common_joint
    }
    common_rows_total = sum(pooled_common_cell_counts.values())
    fixed_cell_weights = {cell: pooled_common_cell_counts[cell] / common_rows_total
                          for cell in common_joint} if common_rows_total else {}
    high_weight_cells = sorted(fixed_cell_weights.items(), key=lambda pair: (-pair[1], pair[0]))[:10]
    for period in labels:
        cell_weights = np.array([
            fixed_cell_weights[(float(row.impact_load), int(row.entry_friction))] /
            cell_counts_by_period[(float(row.impact_load), int(row.entry_friction), period)]
            for row in common_frame[common_frame.period == period].itertuples()
        ], dtype=float)
        sub = common_frame[common_frame.period == period]
        weighted = _weighted_corr(sub, cell_weights)
        weighted["common_stratum_count"] = len(common_joint)
        weighted["support_rows"] = int(len(sub))
        weighted["support_coverage_full_period"] = _finite(len(sub) / full_period_n[period]) if full_period_n[period] else None
        correlations["common_impact_ef_distribution_weighted"][period] = weighted
    correlations["common_impact_ef_distribution_weighted"]["weight_definition"] = (
        "Each common exact impact_load x entry_friction cell receives its pooled share across all three periods; "
        "within each period, that cell share is divided equally over its rows. Weights use counts only, not threat_rank."
    )
    correlations["common_impact_ef_distribution_weighted"]["fixed_cell_weight_sum"] = _finite(sum(fixed_cell_weights.values()))
    correlations["common_impact_ef_distribution_weighted"]["effective_n_by_period"] = {
        period: correlations["common_impact_ef_distribution_weighted"][period]["effective_n"]
        for period in labels
    }
    correlations["common_impact_ef_distribution_weighted"]["largest_fixed_cell_weights"] = [
        {"impact_load": float(cell[0]), "entry_friction": int(cell[1]), "weight": _finite(weight)}
        for cell, weight in high_weight_cells
    ]

    # Symmetric mean decomposition for the first and third blocks, on joint
    # common-support rows only. Cell shares are period-specific; their sum is 1.
    cell_mu: dict[tuple[float, int], dict[str, float]] = {}
    cell_n: dict[tuple[float, int], dict[str, int]] = {}
    for level, friction in common_joint:
        cell_mu[(level, friction)] = {}
        cell_n[(level, friction)] = {}
        for period in labels:
            cell = frame[(frame.impact_load == level) & (frame.entry_friction == friction) &
                         (frame.period == period)]
            cell_mu[(level, friction)][period] = float(cell.threat_rank.mean())
            cell_n[(level, friction)][period] = len(cell)
    retained_rows = {period: sum(cell_n[cell][period] for cell in common_joint) for period in labels}
    shares = {period: {cell: (cell_n[cell][period] / retained_rows[period] if retained_rows[period] else 0.0)
                       for cell in common_joint} for period in labels}
    common_means = {period: _finite(sum(shares[period][cell] * cell_mu[cell][period]
                                        for cell in common_joint)) for period in labels}
    within_contributions = []
    for cell in common_joint:
        contribution = (cell_mu[cell][LATE] - cell_mu[cell][EARLY]) * (
            shares[EARLY][cell] + shares[LATE][cell]) / 2
        within_contributions.append({
            "impact_load": float(cell[0]), "entry_friction": int(cell[1]),
            "early_mean": _finite(cell_mu[cell][EARLY]), "late_mean": _finite(cell_mu[cell][LATE]),
            "late_minus_early_mean": _finite(cell_mu[cell][LATE] - cell_mu[cell][EARLY]),
            "symmetric_within_stratum_contribution": _finite(contribution),
        })
    within_contributions.sort(key=lambda item: item["symmetric_within_stratum_contribution"])
    decomp = {"retained_rows": retained_rows, "retained_coverage_full_period": {
        period: _finite(retained_rows[period] / full_period_n[period]) if full_period_n[period] else None
        for period in labels
    }, "common_support_mean_by_period": common_means,
        "symmetric_decomposition": None}
    if common_joint:
        composition_effect = sum((shares[LATE][cell] - shares[EARLY][cell]) *
                                 (cell_mu[cell][EARLY] + cell_mu[cell][LATE]) / 2
                                 for cell in common_joint)
        within_effect = sum((cell_mu[cell][LATE] - cell_mu[cell][EARLY]) *
                            (shares[EARLY][cell] + shares[LATE][cell]) / 2
                            for cell in common_joint)
        decomp["symmetric_decomposition"] = {
            "early_period": EARLY, "late_period": LATE,
            "late_minus_early": _finite(common_means[LATE] - common_means[EARLY]),
            "composition_effect": _finite(composition_effect),
            "within_stratum_effect": _finite(within_effect),
            "identity_residual": _finite((common_means[LATE] - common_means[EARLY]) -
                                         composition_effect - within_effect),
            "definition": "Delta mean = sum((p_late-p_early)*(mu_early+mu_late)/2) + "
                          "sum((mu_late-mu_early)*(p_early+p_late)/2), over fixed exact impact_load x "
                          "entry_friction cells retained by >=30 rows in every block.",
            "weight_denominator": "For each period, p(cell|period) is cell n divided by all rows in retained common-support cells in that period.",
            "support_exclusion": "Rows outside the fixed common cells are excluded; support is selected only from covariate cell counts.",
            "retained_rows": retained_rows,
            "largest_positive_within_stratum_contributions": sorted(
                within_contributions, key=lambda item: item["symmetric_within_stratum_contribution"], reverse=True
            )[:5],
            "largest_negative_within_stratum_contributions": within_contributions[:5],
        }

    # Positivity/support diagnostic: for each impact value on common impact
    # support, are both friction levels sufficiently represented in every block?
    positivity = []
    both_supported = 0
    for level in primary_impact:
        by_period = {}
        both_for_level = True
        for period in labels:
            counts = {str(v): joint_counts.get((level, v, period), 0) for v in (0, 1)}
            by_period[period] = counts
            if any(counts[str(v)] < PRIMARY_THRESHOLD for v in (0, 1)):
                both_for_level = False
        both_supported += int(both_for_level)
        positivity.append({"impact_load": level, "friction_counts_by_period": by_period,
                           "both_friction_levels_meet_threshold_each_period": both_for_level})

    reversals = []
    reversal_pair = (110.5078, 134.6432)
    for period in labels:
        result = {"period": period, "impact_pair": list(reversal_pair), "by_friction": {}}
        for friction in (0, 1):
            cells = {}
            for level in reversal_pair:
                subset = frame[(frame.period == period) & (frame.entry_friction == friction) &
                               (frame.impact_load == level)]
                cells[f"{level:g}"] = _target_stats(subset.threat_rank)
            a = cells[f"{reversal_pair[0]:g}"]["mean"]
            b = cells[f"{reversal_pair[1]:g}"]["mean"]
            median_a = cells[f"{reversal_pair[0]:g}"]["median"]
            median_b = cells[f"{reversal_pair[1]:g}"]["median"]
            result["by_friction"][str(friction)] = {
                "cells": cells,
                "higher_impact_mean_minus_lower_impact_mean": _finite(b - a) if a is not None and b is not None else None,
                "higher_impact_median_minus_lower_impact_median": _finite(median_b - median_a)
                if median_a is not None and median_b is not None else None,
                "both_cells_meet_primary_threshold": bool(cells[f"{reversal_pair[0]:g}"]["n"] >= PRIMARY_THRESHOLD and
                                                           cells[f"{reversal_pair[1]:g}"]["n"] >= PRIMARY_THRESHOLD),
            }
        reversals.append(result)

    missingness = {}
    for col in ("signal_stream", "org_code", "module_code"):
        if col in raw.columns:
            period_result = {}
            raw_period = frame["period"]
            source_col = raw.loc[frame.index, col]
            for period in labels:
                mask = raw_period.eq(period)
                n = int(mask.sum())
                missing_n = int(source_col.loc[mask].isna().sum())
                period_result[period] = {"missing_rows": missing_n, "rows": n,
                                         "missing_share": _finite(missing_n / n) if n else None}
            missingness[col] = period_result

    headline_impact_profiles = []
    for level in (154.0178, 85.973):
        if level not in primary_impact:
            continue
        period_profile = {}
        for period in labels:
            subset = frame[(frame.impact_load == level) & (frame.period == period)]
            friction_counts = {str(v): int((subset.entry_friction == v).sum()) for v in (0, 1)}
            period_profile[period] = {
                **_target_stats(subset.threat_rank),
                "entry_friction_counts": friction_counts,
                "entry_friction_share": {str(v): _finite(friction_counts[str(v)] / len(subset))
                                          if len(subset) else None for v in (0, 1)},
            }
        first, last = period_profile[EARLY], period_profile[LATE]
        headline_impact_profiles.append({
            "impact_load": level,
            "periods": period_profile,
            "early_to_late_mean_change": _finite(last["mean"] - first["mean"])
            if first["mean"] is not None and last["mean"] is not None else None,
            "early_to_late_median_change": _finite(last["median"] - first["median"])
            if first["median"] is not None and last["median"] is not None else None,
            "early_to_late_p10_change": _finite(last["p10"] - first["p10"])
            if first["p10"] is not None and last["p10"] is not None else None,
            "early_to_late_p90_change": _finite(last["p90"] - first["p90"])
            if first["p90"] is not None and last["p90"] is not None else None,
            "early_to_late_iqr_change": _finite(last["iqr"] - first["iqr"])
            if first["iqr"] is not None and last["iqr"] is not None else None,
        })
    overall_target_by_period = {period: _target_stats(frame.loc[frame.period.eq(period), "threat_rank"])
                                for period in labels}

    result = {
        "status": "OK",
        "purpose": "Descriptive conditional/composition audit of impact_load and threat_rank across cycle blocks.",
        "source": {"task": "regression", "train_basename": train_path.name,
                   "train_sha256": _sha256(train_path), "input_rows": int(len(raw)),
                   "analyzable_rows": int(len(frame)),
                   "excluded_rows_required_fields_nonfinite_or_invalid_friction": int(len(raw) - len(frame)),
                   "test_used": False},
        "variables": {"impact": "impact_load", "target": "threat_rank",
                      "period": "cycle_code sorted into three contiguous blocks of unique observed values",
                      "composition_stratifier": "entry_friction"},
        "periods": blocks,
        "period_sample_sizes": full_period_n,
        "support_selection": {
            "target_blind": True, "selection_rule": "exact impact level or exact impact x entry_friction cell must have the stated count in every primary period.",
            "primary_minimum_rows_per_period": PRIMARY_THRESHOLD,
            "impact_level_count": len(impact_levels),
            "common_impact_level_count": len(primary_impact),
            "common_impact_levels": primary_impact,
            "impact_sensitivity": support_sensitivity,
            "common_impact_x_friction_cell_count": len(common_joint),
            "common_impact_x_friction_cells": [{"impact_load": level, "entry_friction": friction}
                                               for level, friction in common_joint],
            "common_impact_level_coverage_rows_by_period": {
                period: int(sum(impact_counts.get((level, period), 0) for level in primary_impact))
                for period in labels
            },
            "common_impact_x_friction_coverage_rows_by_period": retained_rows,
        },
        "same_impact_distributions": impact_rows,
        "same_impact_x_friction_distributions": joint_rows,
        "headline_exact_impact_profiles": headline_impact_profiles,
        "overall_target_by_period": overall_target_by_period,
        "impact_only_fixed_distribution_standardized_mean": {
            "weights": "pooled row shares across all three periods within impact-only common support; target-blind",
            "common_support_mean_by_period": standardized_by_period,
            "fixed_impact_weights": [{"impact_load": level, "weight": _finite(impact_weights[level])}
                                     for level in primary_impact],
            "quantile_note": "No cell medians are averaged to create pooled quantiles; distribution quantiles are reported per exact stratum.",
        },
        "correlations": correlations,
        "mean_decomposition": decomp,
        "entry_friction_common_support_positivity": {
            "primary_threshold_per_cell_per_period": PRIMARY_THRESHOLD,
            "common_impact_levels_with_both_friction_levels_supported_each_period": both_supported,
            "common_impact_level_count": len(primary_impact),
            "impact_level_diagnostics": positivity,
        },
        "reversal_pair_110_5078_vs_134_6432": reversals,
        "metadata_missingness_by_period": missingness,
        "interpretation_limits": [
            "These are observed associations and descriptive composition summaries; they do not identify causal effects.",
            "The three blocks are chronological bins of sorted unique cycle_code values, not tested change points.",
            "Support exclusions can alter the population represented; coverage is reported alongside all common-support summaries.",
            "The target distribution is summarized within exact strata; cell medians are not averaged as pooled quantiles.",
            "No TP2 model errors, drift, or break claims are assessed by this analysis.",
        ],
    }
    out = Path(output_dir).expanduser().resolve() if output_dir is not None else Path(__file__).resolve().parents[2] / "reports" / "eda"
    if make_plots:
        result["figures"] = _make_figures(frame, blocks, impact_rows, joint_rows, decomp,
                                         out, primary_impact, common_joint)
    else:
        result["figures"] = []
    _safe_json(out / "composition_robustness.json", result)
    return result
