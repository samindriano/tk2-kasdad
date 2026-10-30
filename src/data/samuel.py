"""Leakage-aware, descriptive EDA for Samuel's TP2 regression question.

No model fitting, prediction, or submission generation occurs here. The target
is accepted only when the train/test/sample schemas identify it unambiguously.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from src.utils.paths import PROJECT_ROOT, data_root, task_files

REQUIRED = ("impact_load", "threat_rank", "cycle_code")
EXPECTED = ("Train.csv", "Test.csv", "SampleSubmission.csv")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number_stats(series: pd.Series) -> dict:
    numeric = pd.to_numeric(series, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    result = {
        "rows": int(len(series)),
        "missing_or_unparseable": int(numeric.isna().sum()),
        "non_finite": int((numeric.notna() & ~np.isfinite(numeric)).sum()),
        "distinct_finite": int(finite.nunique()),
        "integer_like_count": int(np.isclose(finite, np.round(finite), rtol=0, atol=1e-8).sum()),
    }
    result["integer_like_fraction"] = (
        float(result["integer_like_count"] / len(finite)) if len(finite) else None
    )
    if finite.empty:
        result.update({"min": None, "q01": None, "q25": None, "median": None,
                       "q75": None, "q99": None, "max": None,
                       "iqr_outlier_count": None})
        return result
    q1, q3 = finite.quantile([0.25, 0.75])
    iqr = q3 - q1
    result.update({
        "min": float(finite.min()), "q01": float(finite.quantile(.01)),
        "q25": float(q1), "median": float(finite.median()),
        "q75": float(q3), "q99": float(finite.quantile(.99)),
        "max": float(finite.max()),
        "iqr_outlier_count": int(((finite < q1 - 1.5 * iqr) |
                                   (finite > q3 + 1.5 * iqr)).sum()),
    })
    return result


def _correlation(x: pd.Series, y: pd.Series) -> dict:
    pair = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    n = len(pair)
    result = {"n_complete": int(n), "pearson_r": None, "spearman_rho": None}
    if n >= 2 and pair.x.nunique() > 1 and pair.y.nunique() > 1:
        result["pearson_r"] = float(pearsonr(pair.x, pair.y).statistic)
        result["spearman_rho"] = float(spearmanr(pair.x, pair.y).statistic)
    return result


def _infer_target(train: pd.DataFrame, test: pd.DataFrame,
                  sample: pd.DataFrame | None) -> dict:
    train_only = sorted(set(train.columns) - set(test.columns))
    sample_cols = sorted(set(sample.columns) - {"record_id"}) if sample is not None else []
    confirmed = "threat_rank" if (train_only == ["threat_rank"] and sample_cols == ["threat_rank"]
                                 and sample is not None and set(sample.columns) == {"record_id", "threat_rank"}) else None
    return {
        "train_only_columns": train_only,
        "sample_label_columns": sample_cols,
        "confirmed_target": confirmed,
        "status": "CONFIRMED" if confirmed else "UNKNOWN",
        "semantics": "UNKNOWN: schema confirms the target column name only; numeric rank scale and domain meaning need an official definition.",
        "rule": "Confirm only when threat_rank is the sole train-only column and the sample submission has exactly record_id plus threat_rank.",
    }


def _periods(cycles: pd.Series) -> tuple[pd.Series, list[dict]]:
    unique = np.sort(cycles.dropna().unique())
    # Split ordered observed cycle values into three near-equal contiguous groups.
    # This is a descriptive chronology grouping; it assumes cycle order is meaningful.
    chunks = np.array_split(unique, min(3, len(unique))) if len(unique) else []
    labels = {}
    groups = []
    for index, chunk in enumerate(chunks, start=1):
        label = f"period_{index:02d}"
        labels.update({value: label for value in chunk})
        groups.append({"label": label, "cycle_min": _scalar(chunk.min()),
                       "cycle_max": _scalar(chunk.max()), "unique_cycles": int(len(chunk))})
    return cycles.map(labels), groups


def _scalar(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value if value is None or isinstance(value, (str, int, float, bool)) else str(value)


def _plot_relationship(frame: pd.DataFrame, report: dict, out_dir: Path) -> list[str]:
    figures = []
    complete_pairs = frame[["impact_load", "threat_rank"]].dropna()
    temporal_pairs = frame[["impact_load", "threat_rank", "cycle_code", "period"]].dropna()
    if complete_pairs.empty:
        return figures
    out_dir.mkdir(parents=True, exist_ok=True)
    x = complete_pairs.impact_load.to_numpy(dtype=float)
    y = complete_pairs.threat_rank.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
    artist = ax.hexbin(x, y, gridsize=45, mincnt=1, bins="log", cmap="Blues")
    fig.colorbar(artist, ax=ax, label="Observations per hexagon (log scale)")
    corr = report["relationship"]["overall"]
    title = f"impact_load vs threat_rank (n={corr['n_complete']:,})"
    ax.set(title=title, xlabel="impact_load", ylabel="threat_rank")
    path = out_dir / "samuel_impact_threat_hexbin.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figures.append(path.name)

    # Facet-by-period, sharing the same impact bins and target scale.
    bins = report["relationship"]["impact_bins"]
    periods = sorted(temporal_pairs.period.dropna().unique())
    bounds = {item['label']: item for item in report['relationship']['period_cycle_ranges']}
    if periods:
        fig, axes = plt.subplots(1, len(periods), figsize=(5 * len(periods), 4.8),
                                 sharey=True, squeeze=False, layout="constrained")
        for ax, period in zip(axes[0], periods):
            subset = temporal_pairs[temporal_pairs.period == period].copy()
            subset["impact_bin"] = pd.cut(subset.impact_load, bins=bins, include_lowest=True)
            summaries = subset.groupby("impact_bin", observed=True).threat_rank.agg(
                n="size", median="median", q25=lambda s: s.quantile(.25),
                q75=lambda s: s.quantile(.75)).reset_index()
            centers = [interval.mid for interval in summaries.impact_bin]
            ax.errorbar(centers, summaries["median"],
                        yerr=[summaries["median"] - summaries["q25"],
                              summaries["q75"] - summaries["median"]],
                        marker="o", color="#2864a5", ecolor="#7a9fca", capsize=2, linewidth=1.5)
            label = f"cycles {bounds[period]['cycle_min']}-{bounds[period]['cycle_max']}"
            ax.set(title=f"{label} (n={len(subset):,})", xlabel="impact_load bin center")
            ax.grid(axis="y", alpha=.22)
        axes[0][0].set_ylabel("threat_rank median with middle 50% range")
        path = out_dir / "samuel_impact_bins_by_period.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures.append(path.name)

    periods_summary = report["temporal"]["period_target_distribution"]
    if periods_summary:
        plot_df = pd.DataFrame(periods_summary)
        fig, ax = plt.subplots(figsize=(8.5, 5), layout="constrained")
        ax.errorbar(np.arange(len(plot_df)), plot_df["median"],
                    yerr=[plot_df["median"] - plot_df["q25"], plot_df["q75"] - plot_df["median"]],
                    marker="o", color="#2864a5", capsize=3)
        ax.set_xticks(np.arange(len(plot_df)), [f"{bounds[label]['cycle_min']}-{bounds[label]['cycle_max']}" for label in plot_df['label']])
        ax.set(title="threat_rank distribution by ordered cycle period",
               xlabel="Contiguous period of ordered cycle_code", ylabel="Median and middle 50% range")
        ax.grid(axis="y", alpha=.22)
        path = out_dir / "samuel_threat_by_period.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures.append(path.name)
    return figures


def run_eda(root: Path | str | None = None, output_dir: Path | str | None = None,
            make_plots: bool = True) -> dict:
    """Run the regression EDA or return an explicit blocked result if unavailable."""
    files = task_files("regression", root)
    target_audit = {"status": "UNKNOWN", "reason": "Train/Test/Sample schemas unavailable."}
    report = {
        "status": None,
        "question_status": "PROVISIONAL_DIAGNOSTIC_NOT_FINAL",
        "question": "How strongly are impact_load and threat_rank associated, and is the descriptive relationship consistent across ordered cycle_code periods?",
        "input_root": ("configured_data_root/regression" if root is not None else
                       "TP2_DATA_DIR/regression" if "TP2_DATA_DIR" in os.environ else
                       "data/raw/regression"),
        "expected_files": EXPECTED,
        "files": {key: (value.name if value else None) for key, value in files.items()},
        "target_verification": target_audit,
        "limitations": [
            "cycle_code ordering is treated as chronological for grouping; its domain meaning is not independently established.",
            "Associations are descriptive and do not establish causation, drift, or statistical significance.",
            "No predictive model is fitted and no Kaggle test target is inferred.",
        ],
        "versions": {"pandas": pd.__version__, "numpy": np.__version__},
    }
    out_dir = Path(output_dir) if output_dir is not None else PROJECT_ROOT / "reports" / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(files[key] is None for key in ("train", "test", "sample")):
        report["status"] = "BLOCKED_MISSING_REGRESSION_DATA"
        report["missing_files"] = [EXPECTED[0] if files["train"] is None else None,
                                   EXPECTED[1] if files["test"] is None else None,
                                   EXPECTED[2] if files["sample"] is None else None]
        report["missing_files"] = [name for name in report["missing_files"] if name]
        report["next_action"] = "Place the official regression Train.csv, Test.csv, and SampleSubmission.csv in data/raw/regression/ (or TP2_DATA_DIR/regression/), then rerun."
        report_path = out_dir / "samuel_regression_eda.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return report

    train = pd.read_csv(files["train"])
    test = pd.read_csv(files["test"])
    sample = pd.read_csv(files["sample"]) if files["sample"] else None
    target_audit = _infer_target(train, test, sample)
    report["target_verification"] = target_audit
    for key, path in files.items():
        if path:
            report.setdefault("input_sha256", {})[key] = _sha256(path)
    report["schema"] = {
        "train_shape": list(train.shape), "test_shape": list(test.shape),
        "sample_shape": list(sample.shape) if sample is not None else None,
        "train_dtypes": {name: str(dtype) for name, dtype in train.dtypes.items()},
        "test_dtypes": {name: str(dtype) for name, dtype in test.dtypes.items()},
        "train_test_feature_schema_matches": sorted(set(train.columns) - {"threat_rank"}) == sorted(test.columns),
        "train_columns_missing_test": sorted(set(train.columns) - set(test.columns)),
        "test_columns_missing_train": sorted(set(test.columns) - set(train.columns)),
    }
    missing_required = {
        "train": [col for col in REQUIRED if col not in train.columns],
        "test": [col for col in ("impact_load", "cycle_code") if col not in test.columns],
        "sample": ["threat_rank"] if sample is None or "threat_rank" not in sample.columns else [],
    }
    report["required_columns_missing"] = missing_required
    report["target_schema_confirmed"] = target_audit["confirmed_target"] is not None
    report["numeric_profiles"] = {
        "train": {col: _number_stats(train[col]) for col in REQUIRED if col in train},
        "test": {col: _number_stats(test[col]) for col in ("impact_load", "cycle_code") if col in test},
    }
    report["impact_load_availability"] = {
        split: {"column_present": "impact_load" in frame,
                "rows": int(len(frame)),
                "finite_parsed_count": int(np.isfinite(pd.to_numeric(frame["impact_load"], errors="coerce")).sum())
                if "impact_load" in frame else 0}
        for split, frame in (("train", train), ("test", test))
    }
    if any(missing_required.values()) or not report["target_schema_confirmed"]:
        report["status"] = "BLOCKED_TARGET_OR_REQUIRED_SCHEMA_UNVERIFIED"
        report["next_action"] = "Review the verified train/test/sample schemas and obtain the official regression target definition before interpreting this relationship."
        (out_dir / "samuel_regression_eda.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return report

    numeric = {}
    for col in REQUIRED:
        numeric[col] = pd.to_numeric(train[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    frame = pd.DataFrame(numeric)
    frame["period"], period_bounds = _periods(frame["cycle_code"])
    pair = frame[["impact_load", "threat_rank"]].dropna()
    if pair.empty:
        report["status"] = "BLOCKED_NO_COMPLETE_TARGET_FEATURE_PAIRS"
        report["next_action"] = "Inspect parsing and missingness in the official regression train file."
        (out_dir / "samuel_regression_eda.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return report

    x_min, x_max = float(pair.impact_load.min()), float(pair.impact_load.max())
    impact_edges = np.linspace(x_min, x_max, 9) if x_min < x_max else np.array([x_min - .5, x_max + .5])
    frame["impact_bin"] = pd.cut(frame.impact_load, bins=impact_edges, include_lowest=True, duplicates="drop")
    exact_x = pair.groupby("impact_load", observed=True).threat_rank.agg(
        row_count="size", distinct_targets="nunique")
    repeated_x = exact_x[exact_x.row_count > 1]
    report["status"] = "COMPLETE"
    report["relationship"] = {
        "overall": _correlation(frame.impact_load, frame.threat_rank),
        "exact_impact_repeat_audit": {
            "unique_impact_values_with_complete_pairs": int(len(exact_x)),
            "impact_values_repeated": int(len(repeated_x)),
            "repeated_impact_values_with_multiple_targets": int((repeated_x.distinct_targets > 1).sum()),
            "exactly_determined_on_repeated_impact_values": (
                bool((repeated_x.distinct_targets == 1).all()) if len(repeated_x) else None),
        },
        "by_cycle": [{"cycle_code": _scalar(cycle), **_correlation(group.impact_load, group.threat_rank),
                      "target_nonmissing_n": int(group.threat_rank.notna().sum()),
                      "threat_rank_median": _scalar(group.threat_rank.median()),
                      "threat_rank_q25": _scalar(group.threat_rank.quantile(.25)),
                      "threat_rank_q75": _scalar(group.threat_rank.quantile(.75))}
                     for cycle, group in frame.groupby("cycle_code", dropna=True, observed=True)],
        "by_period": [{"period": str(label), **_correlation(group.impact_load, group.threat_rank),
                       "target_nonmissing_n": int(group.threat_rank.notna().sum()),
                       "impact_load_median": _scalar(group.impact_load.median()),
                       "threat_rank_median": _scalar(group.threat_rank.median()),
                       "threat_rank_q25": _scalar(group.threat_rank.quantile(.25)),
                       "threat_rank_q75": _scalar(group.threat_rank.quantile(.75))}
                      for label, group in frame.groupby("period", dropna=True, observed=True)],
        "period_cycle_ranges": period_bounds,
        "impact_bins": [_scalar(v) for v in impact_edges],
        "target_by_impact_bin": [
            {"impact_bin": str(label), "n_complete_pairs": int(group.threat_rank.notna().sum()),
             "threat_rank_median": _scalar(group.threat_rank.median()),
             "threat_rank_q25": _scalar(group.threat_rank.quantile(.25)),
             "threat_rank_q75": _scalar(group.threat_rank.quantile(.75)),
             "threat_rank_std": _scalar(group.threat_rank.std())}
            for label, group in frame.dropna(subset=["impact_bin"]).groupby("impact_bin", observed=True)
        ],
        "by_period_and_impact_bin": [
            {"period": str(period), "impact_bin": str(bin_label),
             "n_complete_pairs": int(group[["impact_load", "threat_rank"]].dropna().shape[0]),
             "threat_rank_median": _scalar(group.threat_rank.median()),
             "threat_rank_q25": _scalar(group.threat_rank.quantile(.25)),
             "threat_rank_q75": _scalar(group.threat_rank.quantile(.75))}
            for (period, bin_label), group in frame.dropna(subset=["period", "impact_bin", "threat_rank"]).groupby(
                ["period", "impact_bin"], observed=True)
        ],
    }
    report["temporal"] = {
        "status": "COMPLETE" if len(period_bounds) > 1 else "BLOCKED_INSUFFICIENT_UNIQUE_CYCLES",
        "unique_cycle_count": int(frame.cycle_code.nunique(dropna=True)),
        "period_count": len(period_bounds),
        "period_target_distribution": [
            {"label": str(label), "n": int(group.threat_rank.notna().sum()),
             "median": _scalar(group.threat_rank.median()),
             "q25": _scalar(group.threat_rank.quantile(.25)),
             "q75": _scalar(group.threat_rank.quantile(.75)),
             "impact_load_median": _scalar(group.impact_load.median())}
            for label, group in frame.groupby("period", dropna=True, observed=True)
        ],
    }
    train_x = pd.to_numeric(train.impact_load, errors="coerce").replace([np.inf, -np.inf], np.nan)
    train_y = pd.to_numeric(train.threat_rank, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x_q1, x_q3 = train_x.quantile([.25, .75])
    y_q1, y_q3 = train_y.quantile([.25, .75])
    x_fences = (x_q1 - 1.5 * (x_q3 - x_q1), x_q3 + 1.5 * (x_q3 - x_q1))
    y_fences = (y_q1 - 1.5 * (y_q3 - y_q1), y_q3 + 1.5 * (y_q3 - y_q1))
    report["outlier_diagnostics"] = {
        "method": "Univariate 1.5 x IQR fences, descriptive flag only.",
        "impact_outlier_rows": _number_stats(train.impact_load)["iqr_outlier_count"],
        "target_outlier_rows": _number_stats(train.threat_rank)["iqr_outlier_count"],
        "complete_pair_rows": int(len(pair)),
        "outlier_counts_by_period": [
            {"period": str(label),
             "impact_outlier_count": int(((group.impact_load < x_fences[0]) |
                                           (group.impact_load > x_fences[1])).sum()),
             "target_outlier_count": int(((group.threat_rank < y_fences[0]) |
                                           (group.threat_rank > y_fences[1])).sum()),
             "rows": int(len(group))}
            for label, group in frame.groupby("period", dropna=True, observed=True)
        ],
    }
    if make_plots:
        report["figures"] = _plot_relationship(frame, report, out_dir)
    report["interpretation_guardrail"] = (
        "Report direction, strength, conditional spread and period contrasts as observed descriptions only. "
        "Pearson is calculated only for finite, nonconstant numeric pairs; Spearman is a rank association. "
        "The name threat_rank is schema evidence only; its scale and semantics remain UNKNOWN until an official definition is available. "
        "A changing association or distribution is not, by itself, evidence of a mechanism or statistically established drift."
    )
    (out_dir / "samuel_regression_eda.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
