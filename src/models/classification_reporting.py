"""Reporting and static visual summaries for completed Classification Round 1."""
from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models.classification_evaluation import load_protocol, verify_freeze
from src.models.classification_metrics import binary_metrics
from src.models.round1_evaluation import array_digest
from src.utils.paths import PROJECT_ROOT, task_files


MODELS = ["C0_A", "C0_B", "C0_C", "C1_A", "C1_B", "C2_A", "C2_B"]
POLICIES = ["fixed_0.5", "inner_selected"]
PRIMARY = ["W1", "W2", "W3", "W4", "W5", "W6"]
SECONDARY = ["H12_E", "H12_L"]
COLORS = {
    "C0_A": "#64748b", "C0_B": "#94a3b8", "C0_C": "#2563eb",
    "C1_A": "#1d4ed8", "C1_B": "#7c3aed", "C2_A": "#e87924", "C2_B": "#15966a",
}
REPRESENTATIVE = ["C0_C", "C1_A", "C1_B", "C2_A", "C2_B"]
SUBGROUP_MODELS = ["C1_A", "C1_B", "C2_A", "C2_B"]
SUBGROUPS = ["EF0", "EF1", "stream_missing", "stream_present", "all_codes_known_nonmissing", "any_unseen_code", "any_missing_code"]
FIGURES = [
    "classification_target_prevalence_by_cycle.png",
    "classification_primary_macro_f1_heatmaps.png",
    "classification_primary_per_cycle_f1.png",
    "classification_primary_confusion_matrices.png",
    "classification_primary_reliability.png",
    "classification_primary_subgroup_f1.png",
]


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_result(result: dict, protocol: dict, freeze: dict) -> None:
    if result.get("status") != "CLASSIFICATION_ROUND1_COMPLETE_REVIEW_NO_FINAL_MODEL":
        raise ValueError(f"Classification Round 1 is not complete: {result.get('status')!r}")
    if result.get("freeze_signature") != freeze["signature"] or result.get("identity") != freeze["identity"]:
        raise ValueError("Results do not match the current frozen data/protocol/code/runtime identity.")
    if len(result.get("jobs", [])) != 56 or result.get("total_evaluation_jobs") != 56:
        raise ValueError("Expected all 56 recipe-window evaluation jobs before reporting.")
    recipes = [recipe["id"] for recipe in protocol["recipes"]]
    if recipes != MODELS:
        raise ValueError("Unexpected recipe order/IDs; update the report only after protocol review.")
    expected = {(family, window, model) for family, windows in [("primary", PRIMARY), ("secondary", SECONDARY)]
                for window in windows for model in MODELS}
    jobs = {(job["family"], job["window"], job["model"]) for job in result["jobs"]}
    if expected != jobs:
        raise ValueError("Missing/duplicate result job; report requires all seven models on all eight windows.")
    for family in ["primary", "secondary"]:
        for policy in POLICIES:
            if set(result["aggregate"][family][policy]) != set(MODELS):
                raise ValueError(f"Incomplete aggregate: {family}/{policy}")


def _job_lookup(results: dict) -> dict[tuple[str, str, str], dict]:
    return {(job["family"], job["window"], job["model"]): job for job in results["jobs"]}


def _recipe_info(protocol: dict) -> dict[str, dict]:
    return {recipe["id"]: recipe for recipe in protocol["recipes"]}


def _method(recipe: dict) -> str:
    return {
        "majority": "Majority class",
        "ef_lookup": "EF positive-rate lookup",
        "lr_two": "Logistic regression, impact + EF",
        "lr_structured": "Logistic regression, structured",
        "hgb_structured": "Histogram gradient boosting, structured",
        "lr_text": "TF-IDF + logistic regression",
        "lr_hybrid": "Structured + TF-IDF logistic regression",
    }.get(recipe["kind"], recipe["kind"])


def _validate_metrics(results: dict, frame: pd.DataFrame, artifact: Path) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    """Reconcile cached probability vectors to all recorded outer-fold summaries."""
    jobs = _job_lookup(results)
    windows = {w["id"]: w for w in load_protocol()["primary_windows"] + load_protocol()["secondary_windows"]}
    arrays: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for window_id, window in windows.items():
        fit_end = str(window["fit"][1])
        fit_audit_path = Path(results["_report_dir"]) / "fit_audits" / f"fit{fit_end}.json"
        fit_audits = _json(fit_audit_path)
        for model in MODELS:
            job = jobs[("primary" if window_id in PRIMARY else "secondary", window_id, model)]
            if model not in fit_audits or fit_audits[model].get("selection") != job["threshold_selection"]:
                raise ValueError(f"Missing/mismatched fit audit for {window_id}/{model}.")
            fit_end = window["fit"][1]
            cache_path = artifact / f"fit{fit_end}__{model}.npz"
            with np.load(cache_path, allow_pickle=False) as saved:
                all_rows = saved["outer_rows"].copy()
                all_probability = saved["probability"].copy()
            fit_meta_path = artifact / f"fit{fit_end}__{model}.json"
            fit_meta = _json(fit_meta_path)
            if array_digest(all_probability) != fit_meta["probability_sha256"]:
                raise ValueError(f"Cached probability hash mismatch for {window_id}/{model}.")
            val_mask = frame.cycle_code.between(*window["validation"])
            expected_rows = np.flatnonzero(val_mask.to_numpy())
            positions = np.searchsorted(all_rows, expected_rows)
            if positions.size != expected_rows.size or (positions.size and positions.max() >= len(all_rows)) or not np.array_equal(all_rows[positions], expected_rows):
                raise ValueError(f"Cached row alignment mismatch for {window_id}/{model}.")
            probability = all_probability[positions]
            if array_digest(probability) != job["probability_sha256"]:
                raise ValueError(f"Window probability hash mismatch for {window_id}/{model}.")
            actual = frame.iloc[expected_rows].artifact_signal.to_numpy()
            for policy, threshold in [("fixed_0.5", .5), ("inner_selected", job["threshold_selection"]["threshold"] )]:
                recomputed = binary_metrics(actual, probability, threshold)
                recorded = job["policies"][policy]["metrics"]
                for metric in ["macro_f1", "average_precision", "balanced_accuracy", "positive_rate", "positive_prediction_rate"]:
                    left, right = recomputed.get(metric), recorded.get(metric)
                    if left is None or right is None:
                        if left is not right:
                            raise ValueError(f"Null mismatch in {window_id}/{model}/{policy}/{metric}.")
                    elif not np.isclose(left, right, rtol=1e-11, atol=1e-12):
                        raise ValueError(f"Metric mismatch in {window_id}/{model}/{policy}/{metric}: {left} != {right}.")
            arrays[(window_id, model)] = (expected_rows, probability)

    # Independently rebuild both pooled primary confusion-derived metrics.
    for model in MODELS:
        chunks = [arrays[(window, model)] for window in PRIMARY]
        rows = np.concatenate([x[0] for x in chunks])
        probs = np.concatenate([x[1] for x in chunks])
        if len(rows) != len(np.unique(rows)):
            raise ValueError(f"Primary validation rows overlap for {model}.")
        actual = frame.iloc[rows].artifact_signal.to_numpy()
        thresholds = np.concatenate([np.full(len(jobs[("primary", window, model)]["probability_sha256"] if False else arrays[(window, model)][0]), .5) for window in PRIMARY])
        # Threshold vectors preserve separate inner-selected cutoff by outer fit.
        selected_thresholds = np.concatenate([
            np.full(len(arrays[(window, model)][0]), jobs[("primary", window, model)]["threshold_selection"]["threshold"])
            for window in PRIMARY])
        for policy, cutoff in [("fixed_0.5", thresholds), ("inner_selected", selected_thresholds)]:
            check = binary_metrics(actual, probs, cutoff)
            recorded = results["aggregate"]["primary"][policy][model]["metrics"]
            for metric in ["macro_f1", "average_precision", "balanced_accuracy", "positive_rate", "positive_prediction_rate"]:
                left, right = check.get(metric), recorded.get(metric)
                if left is None or right is None:
                    if left is not right:
                        raise ValueError(f"Pooled metric null mismatch for {model}/{policy}/{metric}.")
                elif not np.isclose(left, right, rtol=1e-11, atol=1e-12):
                    raise ValueError(f"Pooled metric mismatch for {model}/{policy}/{metric}.")
    return arrays


def _fmt(value, digits=4) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)


def _primary_tables(results: dict, protocol: dict) -> str:
    agg = results["aggregate"]["primary"]
    recipes = _recipe_info(protocol)
    rows = []
    for model in MODELS:
        fixed, selected = (agg[policy][model]["metrics"] for policy in POLICIES)
        jobs = [j for j in results["jobs"] if j["family"] == "primary" and j["model"] == model]
        thresholds = [float(j["threshold_selection"]["threshold"]) for j in jobs]
        rows.append([model, _method(recipes[model]), f"{np.median(thresholds):.2f} [{min(thresholds):.2f}, {max(thresholds):.2f}]",
                     _fmt(fixed["macro_f1"]), _fmt(selected["macro_f1"]), _fmt(selected["macro_f1"] - fixed["macro_f1"]),
                     _fmt(fixed["positive_rate"]), _fmt(fixed["positive_prediction_rate"]),
                     _fmt(selected["positive_prediction_rate"]), _fmt(selected["balanced_accuracy"]),
                     _fmt(fixed["average_precision"]), _fmt(selected["brier_score"])])
    main = _md_table(["Model", "Method", "Inner threshold median [range]", "Pooled F1 @0.50", "Pooled F1 inner", "Δ F1", "Actual class 1 rate", "Predicted class 1 @0.50", "Predicted class 1 inner", "Balanced accuracy inner", "AP", "Brier"], rows)

    class_rows, confusion_rows = [], []
    for model in MODELS:
        for policy in POLICIES:
            metric = agg[policy][model]["metrics"]
            policy_name = "0.50" if policy == "fixed_0.5" else "inner-selected"
            for label in [0, 1]:
                class_metric = metric["per_class"][label]
                class_rows.append([model, policy_name, label, class_metric["support"], _fmt(class_metric["precision"]),
                                  _fmt(class_metric["recall"]), _fmt(class_metric["f1"])])
            cm = metric["confusion_matrix"]
            confusion_rows.append([model, policy_name, f"{cm[0][0]:,}", f"{cm[0][1]:,}", f"{cm[1][0]:,}", f"{cm[1][1]:,}",
                                   f"{metric['false_positive']:,}", f"{metric['false_negative']:,}"])
    classwise = _md_table(["Model", "Policy", "Actual class", "Support", "Precision", "Recall", "F1"], class_rows)
    confusion = _md_table(["Model", "Policy", "TN", "FP", "FN", "TP", "False positives", "False negatives"], confusion_rows)

    rows = []
    for model in MODELS:
        rows.append([model, _fmt(agg["fixed_0.5"][model]["equal_cycle_macro_f1"]),
                     _fmt(agg["inner_selected"][model]["equal_cycle_macro_f1"])])
    equal = _md_table(["Model", "Equal-cycle F1 @0.50", "Equal-cycle F1 inner"], rows)

    rows = []
    jobs = _job_lookup(results)
    for model in MODELS:
        rows.append([model] + [_fmt(jobs[("primary", window, model)]["policies"]["fixed_0.5"]["metrics"]["macro_f1"]) for window in PRIMARY])
    fixed_windows = _md_table(["Model", *PRIMARY], rows)
    rows = []
    for model in MODELS:
        rows.append([model] + [_fmt(jobs[("primary", window, model)]["policies"]["inner_selected"]["metrics"]["macro_f1"]) for window in PRIMARY])
    selected_windows = _md_table(["Model", *PRIMARY], rows)

    cycle_rows = []
    for model in REPRESENTATIVE:
        for window in PRIMARY:
            job = jobs[("primary", window, model)]
            fixed = {row["cycle"]: row for row in job["policies"]["fixed_0.5"]["per_cycle"]}
            selected = {row["cycle"]: row for row in job["policies"]["inner_selected"]["per_cycle"]}
            for cycle in sorted(selected):
                a, b = fixed[cycle], selected[cycle]
                cycle_rows.append([model, cycle, a["n"], _fmt(a["positive_rate"]),
                                   _fmt(a["positive_prediction_rate"]), _fmt(b["positive_prediction_rate"]),
                                   _fmt(b["macro_f1"])])
    cycle_behavior = _md_table(["Model", "Cycle", "n", "Actual class 1 rate", "Predicted class 1 rate @0.50",
                                "Predicted class 1 rate inner", "Macro F1 inner"], cycle_rows)

    rows = []
    for window in SECONDARY:
        for model in MODELS:
            window_job = jobs[("secondary", window, model)]
            fixed = window_job["policies"]["fixed_0.5"]["metrics"]["macro_f1"]
            selected = window_job["policies"]["inner_selected"]["metrics"]["macro_f1"]
            rows.append([window, model, _fmt(fixed), _fmt(selected), _fmt(selected-fixed)])
    secondary = _md_table(["H12 window", "Model", "Window Macro F1 @0.50", "Window Macro F1 inner", "Δ F1"], rows)

    rows = []
    for end in sorted({j["fit_cycles"][1] for j in results["jobs"] if j["family"] == "primary"}):
        matching = [j for j in results["jobs"] if j["family"] == "primary" and j["fit_cycles"][1] == end]
        for job in matching:
            selection = job["threshold_selection"]
            rule = "fixed control" if job["model"] == "C0_A" else "fallback" if selection["fallback"] else "selected"
            rows.append([f"47–{end}", job["model"], _fmt(selection["threshold"], 2),
                         _fmt(selection.get("inner_macro_f1")), selection["selection_n"], rule, selection["reason"]])
    thresholds_table = _md_table(["Outer fit history", "Model", "Threshold", "Inner F1", "Inner n", "Rule", "Selection record"], rows)

    subgroup_rows = []
    for model in SUBGROUP_MODELS:
        metrics = agg["inner_selected"][model]["subgroups"]
        for subgroup in SUBGROUPS:
            metric = metrics.get(subgroup)
            if metric is None:
                subgroup_rows.append([model, subgroup, "NOT REPORTED", "—", "—", "—", "—", "—"])
            else:
                subgroup_rows.append([model, subgroup, metric["n"], _fmt(metric["macro_f1"]),
                                      _fmt(metric["per_class"][0]["f1"]), _fmt(metric["per_class"][1]["f1"]),
                                      metric["false_positive"], metric["false_negative"]])
    subgroup_table = _md_table(["Model", "Group", "n", "Macro F1", "F1 class 0", "F1 class 1", "FP", "FN"], subgroup_rows)

    calibration_rows = []
    for model in SUBGROUP_MODELS:
        for index, bin_row in enumerate(agg["inner_selected"][model]["reliability"]):
            calibration_rows.append([model, f"[{bin_row['lower']:.1f}, {bin_row['upper']:.1f}{']' if bin_row['upper'] == 1 else ')'}",
                                     bin_row["n"], _fmt(bin_row["mean_probability"]), _fmt(bin_row["observed_positive_rate"])])
    calibration_table = _md_table(["Model", "Fixed probability bin", "n", "Mean probability", "Actual class 1 share"], calibration_rows)

    return "\n".join([
        "# Classification Round 1 — comparison tables", "",
        f"Protocol `{results['protocol_id']}`; run `{results['run_id']}`; {results['total_evaluation_jobs']} evaluated recipe-window jobs.",
        "Validation only. Macro F1 is computed with fixed labels `[0,1]`; AP is sklearn average precision. Selected thresholds are fit-history inner-only; no outer validation label chooses its threshold.", "",
        "## Primary pooled metrics", "", main, "",
        "Pooled F1 uses all primary validation rows once. Delta is selected minus fixed. AP uses the probability vector and is independent of threshold.", "",
        "## Per-class precision, recall and F1", "", classwise, "",
        "Class-specific metrics use fixed labels 0 and 1. The recorded zero-division convention sets undefined precision/recall/F1 to zero.", "",
        "## Primary pooled confusion counts", "", confusion, "",
        "Rows are actual classes 0/1; columns are predicted classes 0/1. FP means actual 0/predicted 1; FN means actual 1/predicted 0.", "",
        "## Equal-cycle F1", "", equal, "",
        "This is the arithmetic mean over 24 per-cycle Macro F1 scores; it is distinct from pooled F1.", "",
        "## Primary-window F1 — fixed threshold 0.50", "", fixed_windows, "",
        "## Primary-window F1 — inner-selected threshold", "", selected_windows, "",
        "## Per-cycle prevalence and positive-prediction rates", "", cycle_behavior, "",
        "Cycle-level output rates expose changes due to both observed prevalence and policy threshold; they are descriptive validation behavior, not prevalence estimates for test.", "",
        "## H12 window Macro F1", "", secondary, "",
        "H12 views overlap primary validation rows; they are longer-horizon sensitivity checks, not independent replications.", "",
        "## Threshold decisions by fit history", "", thresholds_table, "",
        "Majority class has no selection and uses 0.50 as a fixed control. Other recipes use inner histories described in the frozen protocol; `fallback` rows explicitly retain 0.50.", "",
        "## Selected-threshold subgroups — primary pooled", "", subgroup_table, "",
        "Subgroup metrics use fit-derived category vocabulary and impact cutpoints recorded per job. Missing/unseen groups may overlap. Single-class subsets retain fixed labels; AP/balanced accuracy are null where only one actual class occurs.", "",
        "## Calibration-bin coverage", "",
        "Ten equal-width probability bins; probability vectors are identical for both threshold policies.", "", calibration_table, "",
        "No model is selected for final use in this artifact.", ""
    ])


def _plot_prevalence(review: dict, figure_dir: Path) -> None:
    rows = review["by_cycle"]["cycles"]
    cycles = [r["cycle_code"] for r in rows]
    rate = [r["positive_rate"] for r in rows]
    n = [r["n"] for r in rows]
    fig, ax = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
    ax.plot(cycles, rate, color="#7c3aed", marker="o", linewidth=2, markersize=4)
    ax.axhline(review["target"]["positive_rate"], color="#475569", linestyle="--", linewidth=1.4,
               label=f"All training rows ({review['target']['positive_rate']:.1%})")
    ax.set(xlabel="Disclosure cycle_code (ordered index)", ylabel="Class 1 share",
           title="Observed class 1 prevalence by complete training cycle", ylim=(0, 1), xlim=(min(cycles)-.5, max(cycles)+.5))
    ax.set_xticks(cycles)
    ax.tick_params(axis="x", labelrotation=45)
    ax.grid(axis="y", alpha=.22)
    ax.legend(frameon=False, loc="best")
    for cycle, value, count in zip(cycles, rate, n):
        ax.annotate(f"n={count:,}", (cycle, value), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=6, rotation=65)
    path = figure_dir / "classification_target_prevalence_by_cycle.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_heatmaps(results: dict, figure_dir: Path) -> None:
    jobs = _job_lookup(results)
    arrays = []
    for policy in POLICIES:
        arrays.append(np.array([[jobs[("primary", window, model)]["policies"][policy]["metrics"]["macro_f1"]
                                 for window in PRIMARY] for model in MODELS], dtype=float))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), constrained_layout=True, sharey=True)
    for ax, values, policy, title in zip(axes, arrays, POLICIES, ["Fixed threshold 0.50", "Inner-selected threshold"]):
        masked = np.ma.masked_invalid(values)
        image = ax.imshow(masked, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(PRIMARY)), PRIMARY)
        ax.set_yticks(range(len(MODELS)), MODELS)
        ax.set_xlabel("Primary validation window")
        ax.set_title(f"Macro F1 — {title}")
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                value = values[i, j]
                ax.text(j, i, "n/a" if not np.isfinite(value) else f"{value:.3f}", ha="center", va="center",
                        color="white" if np.isfinite(value) and value >= .65 else "#172033", fontsize=8)
    axes[0].set_ylabel("Model")
    bar = fig.colorbar(image, ax=axes, shrink=.84)
    bar.set_label("Macro F1 (common scale 0–1)")
    fig.suptitle("All seven models across the six disjoint primary windows")
    path = figure_dir / "classification_primary_macro_f1_heatmaps.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_per_cycle(results: dict, review: dict, figure_dir: Path) -> None:
    jobs = _job_lookup(results)
    cycle_prevalence = {row["cycle_code"]: row for row in review["by_cycle"]["cycles"]}
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.4), constrained_layout=True, sharex=True)
    cycle_union = sorted({int(c["cycle"]) for model in REPRESENTATIVE for window in PRIMARY
                          for c in jobs[("primary", window, model)]["policies"]["inner_selected"]["per_cycle"]})
    for model in REPRESENTATIVE:
        lookup = {}
        for window in PRIMARY:
            for row in jobs[("primary", window, model)]["policies"]["inner_selected"]["per_cycle"]:
                lookup[int(row["cycle"])] = row
        values = [lookup.get(cycle, {}).get("macro_f1") for cycle in cycle_union]
        axes[0].plot(cycle_union, [np.nan if x is None else x for x in values], marker="o", markersize=3,
                     linewidth=1.7, color=COLORS[model], label=model)
    prevalence = [cycle_prevalence[cycle]["positive_rate"] for cycle in cycle_union]
    axes[1].plot(cycle_union, prevalence, color="#111827", marker="o", markersize=3, linewidth=1.7)
    axes[1].set_ylabel("Actual class 1 rate")
    axes[0].set_ylabel("Macro F1")
    axes[0].set_ylim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("Validation cycle_code (ordered index)")
    axes[0].set_title("Primary per-cycle Macro F1 — inner-selected thresholds")
    axes[1].set_title("Actual positive prevalence on the same validation cycles")
    axes[0].legend(title="Recipe", ncol=5, frameon=False, loc="lower center")
    for ax in axes:
        ax.set_xticks(cycle_union)
        ax.tick_params(axis="x", labelrotation=45)
        ax.grid(axis="y", alpha=.2)
    path = figure_dir / "classification_primary_per_cycle_f1.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_confusions(results: dict, figure_dir: Path) -> None:
    agg = results["aggregate"]["primary"]
    fig, axes = plt.subplots(4, 2, figsize=(10, 14), constrained_layout=True)
    image = None
    for row, model in enumerate(SUBGROUP_MODELS):
        for col, policy in enumerate(POLICIES):
            ax = axes[row, col]
            metric = agg[policy][model]["metrics"]
            cm = np.asarray(metric["confusion_matrix"], dtype=int)
            support = cm.sum(axis=1)
            row_pct = np.divide(cm, support[:, None], out=np.full((2, 2), np.nan), where=support[:, None] > 0)
            image = ax.imshow(row_pct, cmap="Blues", vmin=0, vmax=1)
            for i in range(2):
                for j in range(2):
                    count = int(cm[i, j])
                    pct = "—" if np.isnan(row_pct[i, j]) else f"{row_pct[i,j]:.1%}"
                    ax.text(j, i, f"{count:,}\n{pct}", ha="center", va="center",
                            color="white" if np.isfinite(row_pct[i, j]) and row_pct[i, j] >= .6 else "#111827", fontsize=10)
            ax.set_xticks([0, 1], ["0", "1"])
            ax.set_yticks([0, 1], ["0", "1"])
            ax.set_xlabel("Predicted class")
            ax.set_ylabel("Actual class")
            ax.set_title(f"{model} · {'0.50' if policy == 'fixed_0.5' else 'inner-selected'} · n={metric['n']:,}")
    bar = fig.colorbar(image, ax=axes, shrink=.78)
    bar.set_label("Share within actual class row (common 0–100% scale)")
    fig.suptitle("Primary pooled confusion matrices — count and actual-row percentage", fontsize=14)
    path = figure_dir / "classification_primary_confusion_matrices.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_calibration(results: dict, figure_dir: Path) -> None:
    agg = results["aggregate"]["primary"]["inner_selected"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.3), constrained_layout=True, sharex=True, sharey=True)
    for ax, model in zip(axes.flat, SUBGROUP_MODELS):
        bins = agg[model]["reliability"]
        xs = [b["mean_probability"] for b in bins if b["mean_probability"] is not None and b["observed_positive_rate"] is not None]
        ys = [b["observed_positive_rate"] for b in bins if b["mean_probability"] is not None and b["observed_positive_rate"] is not None]
        ns = [b["n"] for b in bins if b["mean_probability"] is not None and b["observed_positive_rate"] is not None]
        ax.plot([0, 1], [0, 1], color="#64748b", linestyle="--", linewidth=1.2, label="Ideal")
        if xs:
            size = [28 + 210 * n / max(ns) for n in ns]
            ax.plot(xs, ys, color=COLORS[model], linewidth=1.5, alpha=.8)
            ax.scatter(xs, ys, color=COLORS[model], s=size, edgecolor="white", linewidth=.6,
                       label="Observed (marker size ∝ bin n)", zorder=3)
        ax.set_title(model)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=.2)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    for ax in axes[-1, :]:
        ax.set_xlabel("Mean predicted probability")
    for ax in axes[:, 0]:
        ax.set_ylabel("Observed class 1 share")
    fig.suptitle("Primary validation probability calibration diagnostic (10 fixed bins)")
    path = figure_dir / "classification_primary_reliability.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_subgroups(results: dict, figure_dir: Path) -> None:
    values = results["aggregate"]["primary"]["inner_selected"]
    labels = ["EF0", "EF1", "stream_missing", "stream_present", "all_codes_known_nonmissing", "any_unseen_code", "any_missing_code"]
    short = ["EF 0", "EF 1", "Stream missing", "Stream present", "All codes known", "Any unseen", "Any missing"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), constrained_layout=True, sharex=True, sharey=True)
    for ax, model in zip(axes.flat, SUBGROUP_MODELS):
        metrics = values[model]["subgroups"]
        y = [metrics.get(label, {}).get("macro_f1") if metrics.get(label) else np.nan for label in labels]
        ns = [metrics.get(label, {}).get("n", 0) if metrics.get(label) else 0 for label in labels]
        positions = np.arange(len(labels))
        ax.barh(positions, [0 if not np.isfinite(score) else score for score in y],
                color=[COLORS[model] if np.isfinite(score) else "#cbd5e1" for score in y], alpha=.88, height=.62)
        for pos, score, n in zip(positions, y, ns):
            if np.isfinite(score):
                inside = score >= .89
                ax.text(score - .015 if inside else min(.965, score + .015), pos, f"{score:.2f}", ha="right" if inside else "left",
                        va="center", fontsize=9, color="white" if inside else "#111827")
        labels_with_n = [f"{name} (n={n:,})" for name, n in zip(short, ns)]
        ax.set_yticks(positions, labels_with_n, fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_title(f"{model} · inner-selected threshold")
        ax.grid(axis="x", alpha=.2)
    for ax in axes[-1, :]:
        ax.set_xlabel("Macro F1 (common 0–1 scale)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Validation subgroup and denominator")
    fig.suptitle("Primary pooled subgroup Macro F1 — missing/unseen groups are explicit")
    path = figure_dir / "classification_primary_subgroup_f1.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_classification_report(results_path=None, output_dir=None, data_dir=None) -> dict:
    """Build comparison tables/figures from a complete frozen result and vectors."""
    frozen = verify_freeze(data_dir)
    path = Path(results_path) if results_path is not None else PROJECT_ROOT / "reports/classification_round1/results.json"
    if not path.is_file():
        raise FileNotFoundError(f"Classification results are not ready: {path}")
    results = _json(path)
    protocol = load_protocol()
    _validate_result(results, protocol, frozen)
    results["_report_dir"] = str(path.parent)
    frame_path = task_files("classification", data_dir)["train"]
    frame = pd.read_csv(frame_path)
    expected_rows = int(frame.cycle_code.between(56, 79).sum())
    if expected_rows != 38588:
        raise ValueError(f"Unexpected primary validation coverage from protocol: {expected_rows} rows.")
    run_id = results["run_id"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", str(run_id)):
        raise ValueError("Unsafe run_id in results.")
    artifact = PROJECT_ROOT / "experiments/runs" / run_id
    vectors = _validate_metrics(results, frame, artifact)

    destination = Path(output_dir) if output_dir is not None else PROJECT_ROOT / "reports/classification_round1"
    figure_dir = destination / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    tables = _primary_tables(results, protocol)
    (destination / "comparison_tables.md").write_text(tables, encoding="utf-8")
    review_path = path.parent / "data_review.json"
    if not review_path.is_file():
        review_path = PROJECT_ROOT / "reports/classification_round1/data_review.json"
    review = _json(review_path)
    expected_train_hash = results["identity"]["input_sha256"]["train"]
    if review.get("inputs", {}).get("train", {}).get("sha256") != expected_train_hash:
        raise ValueError("Data-review profile does not match the frozen classification Train identity.")
    _plot_prevalence(review, figure_dir)
    _plot_heatmaps(results, figure_dir)
    _plot_per_cycle(results, review, figure_dir)
    _plot_confusions(results, figure_dir)
    _plot_calibration(results, figure_dir)
    _plot_subgroups(results, figure_dir)
    return {"status": "CLASSIFICATION_REPORT_BUILT_FROM_FROZEN_VALIDATION_RESULTS",
            "tables": str(destination / "comparison_tables.md"),
            "figures": [str(figure_dir / name) for name in FIGURES],
            "models": MODELS, "primary_windows": PRIMARY, "secondary_windows": SECONDARY,
            "primary_rows_per_model": expected_rows, "validated_window_vectors": len(vectors),
            "run_id": run_id, "freeze_signature": frozen["signature"]}
