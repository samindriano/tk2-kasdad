"""Aggregate reporting for the frozen Regression Round 1 experiment.

This module reads completed experiment results, fit-audit summaries, and saved
validation predictions. It never fits or predicts with a model and never reads
the official test file.
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

from src.utils.paths import PROJECT_ROOT, task_files


RECIPE_ORDER = ["R0_A", "R0_B", "R0_C", "R1_A", "R1_B", "R1_B_ADD", "R1_C", "R1_D"]
PRIMARY_WINDOWS = ["W1", "W2", "W3", "W4", "W5", "W6"]
SECONDARY_WINDOWS = ["H12_E", "H12_L"]
TAIL_MODELS = ["R0_C", "R1_B", "R1_C", "R1_D"]
HIST_MODELS = ["R1_B", "R1_C", "R1_D"]
SUBGROUPS = [
    ("EF0", "Entry friction = 0"),
    ("EF1", "Entry friction = 1"),
    ("region_low", "Low region: threat_rank < 33.333"),
    ("region_middle", "Middle region: 33.333 <= threat_rank < 66.667"),
    ("region_high", "High region: threat_rank >= 66.667"),
    ("tail_low_le5", "Low tail: threat_rank <= 5"),
    ("tail_high_ge95", "High tail: threat_rank >= 95"),
]


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _f(value: Any, digits: int = 4) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{float(value):.{digits}f}"


def _rate(value: Any) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{100 * float(value):.1f}%"


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(cell(x) for x in row) + " |" for row in rows)
    return "\n".join(lines)


def _audit_state(audit_root: Path, window: str, model: str) -> dict[str, Any]:
    path = audit_root / f"{window}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing fit audit: {path}")
    audit = _json(path)
    if model not in audit:
        raise KeyError(f"{model} missing from {path}")
    return audit[model]


def _hyperparameter_summary(model: str, states: list[dict[str, Any]]) -> str:
    actual = [state["fitted_state"].get("actual_hyperparameters", {}) for state in states]
    if model in {"R0_A", "R0_B", "R0_C"}:
        first = actual[0]
        kind = {"global_median": "global target median",
                "impact_median": "impact_load median; fallback global median",
                "joint_median": "impact_load x entry_friction median; hierarchical fallback"}.get(
                    first.get("kind"), first.get("kind", "median lookup"))
        return f"{kind}; minimum group support {first.get('min_support', 'n/a')}"
    if model == "R1_A":
        first = actual[0]
        return (f"{first.get('estimator', 'LinearRegression')}; intercept={first.get('fit_intercept')}; "
                f"StandardScaler(mean={first.get('scaler_with_mean')}, std={first.get('scaler_with_std')})")
    first = actual[0]
    params = [
        f"loss={first.get('loss')}", f"lr={first.get('learning_rate')}",
        f"iterations={first.get('max_iter')}", f"leaves={first.get('max_leaf_nodes')}",
        f"min_leaf_n={first.get('min_samples_leaf')}", f"L2={first.get('l2_regularization')}",
        f"max_bins={first.get('max_bins')}", f"early_stopping={first.get('early_stopping')}",
        f"interaction_cst={first.get('interaction_cst')}", f"seed={first.get('random_state')}",
    ]
    if model == "R1_D":
        masks = [x.get("categorical_features") or [] for x in actual]
        params.append(f"native categorical inputs={sum(bool(x) for x in masks[0]) if masks else 0}")
    return "; ".join(params)


def _state_summary(model: str, states: list[dict[str, Any]]) -> str:
    fitted = [state["fitted_state"] for state in states]
    feature_counts = [len(x.get("transformed_feature_names", [])) for x in fitted]
    fit_rows = [int(x.get("fit_rows", 0)) for x in fitted]
    hash_count = len({state.get("fitted_state_sha256") for state in states})
    base = (f"fit rows {min(fit_rows):,}–{max(fit_rows):,}; transformed features "
            f"{min(feature_counts)}–{max(feature_counts)}; {hash_count} distinct fold state hashes")
    if model in {"R0_A", "R0_B", "R0_C"}:
        medians = [x.get("global_target_median") for x in fitted if x.get("global_target_median") is not None]
        suffix = f"; global fit median {_f(min(medians), 3)}–{_f(max(medians), 3)}" if medians else ""
        counts = [x.get("supported_group_counts", {}) for x in fitted]
        count_parts = []
        for key, label in [("primary", "joint"), ("impact", "impact"), ("entry_friction", "EF")]:
            vals = [int(x.get(key, 0)) for x in counts if key in x]
            if vals and max(vals) > 0:
                count_parts.append(f"{label} supported groups {min(vals)}–{max(vals)}")
        if count_parts:
            suffix += "; " + ", ".join(count_parts)
        return base + suffix
    if model == "R1_A":
        intercepts = [x.get("linear_intercept") for x in fitted if x.get("linear_intercept") is not None]
        if intercepts:
            base += f"; fitted intercept {_f(min(intercepts), 3)}–{_f(max(intercepts), 3)}"
    if model.startswith("R1_B") or model in {"R1_C", "R1_D"}:
        iterations = [state["fitted_state"].get("actual_hyperparameters", {}).get("max_iter") for state in states]
        if iterations:
            base += f"; configured max_iter={iterations[0]} (no early stopping)"
    return base


def _validate_result(results: dict[str, Any]) -> None:
    if results.get("status") != "ROUND1_COMPLETE_REVIEW_NO_FINAL_MODEL":
        raise ValueError(f"Results are not a completed Round 1 artifact: {results.get('status')!r}")
    jobs = results.get("jobs", [])
    if len(jobs) != 64:
        raise ValueError(f"Expected 64 completed fits/jobs, got {len(jobs)}")
    for family, windows in [("primary", PRIMARY_WINDOWS), ("secondary", SECONDARY_WINDOWS)]:
        for window in windows:
            for model in RECIPE_ORDER:
                matches = [job for job in jobs if job["family"] == family and job["window"] == window and job["model"] == model]
                if len(matches) != 1:
                    raise ValueError(f"Expected one result for {family}/{window}/{model}, got {len(matches)}")


def _job(results: dict[str, Any], family: str, window: str, model: str) -> dict[str, Any]:
    return next(job for job in results["jobs"]
                if job["family"] == family and job["window"] == window and job["model"] == model)


def _build_tables(results: dict[str, Any], audit_root: Path) -> str:
    aggregate = results["aggregate"]
    chunks = [
        "# Regression Round 1 — Comparison Tables",
        "",
        f"Protocol: `{results['protocol_id']}`",
        f"Run: `{results['run_id']}`; {results['total_jobs']} completed fits; status `{results['status']}`.",
        "The reported primary metric is raw, unclipped MAE. These are validation comparisons for review; they do not select a final model.",
        "",
        "## Primary pooled metrics",
        "",
        "Pooled statistics use all primary validation rows (cycles 56–79, once each). Equal-cycle MAE weights each of those 24 cycles equally; mean-window MAE weights each of the six windows equally.",
        "",
    ]
    rows = []
    for model in RECIPE_ORDER:
        item = aggregate["primary"][model]
        raw = item["raw"]
        rows.append([model, f"{raw['n']:,}", _f(raw["mae"]), _f(raw["rmse"]), _f(raw["signed_error"]),
                     _f(item["equal_cycle_mae"]), _f(item["mean_window_mae"])])
    chunks += [_md_table(["Recipe", "n", "Pooled MAE", "Pooled RMSE", "Signed bias", "Equal-cycle MAE", "Mean-window MAE"], rows), "",
               "Signed bias is mean(prediction − actual); positive means overprediction. Negative MAE differences in the comparison table indicate lower MAE for the candidate.",
               "", "## Primary MAE by validation window", "",
               "All six windows are reported separately. They cover non-overlapping validation cycles 56–79.", ""]
    rows = []
    for model in RECIPE_ORDER:
        rows.append([model] + [_f(_job(results, "primary", window, model)["metrics"]["mae"]) for window in PRIMARY_WINDOWS])
    chunks += [_md_table(["Recipe", *PRIMARY_WINDOWS], rows), "", "## Secondary H12 windows", "",
               "These longer-horizon checks are secondary. H12_E shares cycles 56–67 with primary W1–W3; H12_L shares cycles 68–79 with primary W4–W6. They are overlapping validation views, not independent evidence.", ""]
    rows = []
    for window in SECONDARY_WINDOWS:
        for model in RECIPE_ORDER:
            job = _job(results, "secondary", window, model)
            m = job["metrics"]
            rows.append([window, model, f"{m['n']:,}", _f(m["mae"]), _f(m["rmse"]), _f(m["signed_error"]), _f(job["equal_cycle_mae"])])
    chunks += [_md_table(["Window", "Recipe", "n", "MAE", "RMSE", "Signed bias", "Equal-cycle MAE"], rows), "",
               "## Primary error by subgroup", "",
               "Subgroups use validation labels only. Region boundaries are the frozen thirds [0, 33.333…), [33.333…, 66.667…), [66.667, 100]; the extreme tails are actual threat_rank <= 5 and >= 95. These tails are not TP1 target terciles.", ""]
    rows = []
    for model in RECIPE_ORDER:
        metrics = aggregate["primary"][model]["subgroups"]
        for key, label in SUBGROUPS:
            m = metrics[key]
            rows.append([model, label, f"{m['n']:,}", _f(m["mae"]), _f(m["rmse"]), _f(m["signed_error"]),
                         _rate(m["over_rate"]), _rate(m["under_rate"]), _rate(m["equal_rate"])])
    chunks += [_md_table(["Recipe", "Subgroup", "n", "MAE", "RMSE", "Signed bias", "Over", "Under", "Equal"], rows), "",
               "## Primary impact_load groups", "",
               "These summaries aggregate per-level validation errors for each observed impact_load value; they are descriptive associations, not causal effects.", ""]
    rows = []
    for model in RECIPE_ORDER:
        groups = aggregate["primary"][model]["impact_groups"]
        supported = [m for m in groups.values() if int(m["n"]) >= 30]
        sparse = len(groups) - len(supported)
        group_n = [int(m["n"]) for m in supported]
        group_mae = [float(m["mae"]) for m in supported]
        group_bias = [float(m["signed_error"]) for m in supported]
        if supported:
            mae_summary = f"{_f(min(group_mae))} / {_f(np.median(group_mae))} / {_f(max(group_mae))}"
            bias_summary = f"{_f(min(group_bias))} / {_f(np.median(group_bias))} / {_f(max(group_bias))}"
            n_summary = f"{min(group_n):,}–{max(group_n):,}"
        else:
            mae_summary = bias_summary = n_summary = "—"
        rows.append([model, len(groups), f"{sparse}/{len(supported)}", n_summary, mae_summary, bias_summary])
    chunks += [_md_table(["Recipe", "Distinct impact_load levels", "Groups n<30 / n>=30", "Supported-group n range", "Supported-group MAE min / median / max", "Supported-group bias min / median / max"], rows), "",
               "Per-level rows are summarized to keep sparse one-row groups from dominating the report. Group support is counted on pooled primary validation rows; the 30-row marker matches the frozen lookup support threshold and is descriptive here.", "",
               "## Raw versus clipped primary metrics", "",
               "Clipping is evaluated separately against [0, 100]. Raw metrics remain the primary comparison.", ""]
    rows = []
    for model in RECIPE_ORDER:
        item = aggregate["primary"][model]
        raw, clipped = item["raw"], item["clipped"]
        rows.append([model, _f(raw["mae"]), _f(clipped["mae"]), _f(clipped["mae"] - raw["mae"]),
                     _f(raw["rmse"]), _f(clipped["rmse"]), _f(raw["signed_error"]), _f(clipped["signed_error"]),
                     raw["below_zero"], raw["above_100"]])
    chunks += [_md_table(["Recipe", "Raw MAE", "Clipped MAE", "Δ MAE", "Raw RMSE", "Clipped RMSE", "Raw bias", "Clipped bias", "Raw below 0", "Raw above 100"], rows), "",
               "## Frozen pairwise comparisons", "",
               "A negative pooled MAE delta favors the listed candidate on this validation aggregate; counts show the number of primary windows/cycles with lower candidate MAE. This table supports review only and makes no final-model selection.", ""]
    rows = []
    for compare in results["comparisons"]:
        rows.append([compare["candidate"], compare["control"], _f(compare["pooled_delta"]),
                     f"{compare['windows_improved']}/6", f"{compare['cycles_improved']}/24"])
    chunks += [_md_table(["Candidate", "Control", "Pooled Δ MAE", "Windows improved", "Cycles improved"], rows), "",
               "## Fit parameter and fitted-state audit summary", "",
               "Summaries below are derived from the eight per-window `fit_audits/*.json` files, including all six primary and two H12 fits. Exact fitted states remain in those source audit files; hashes are counted here but not repeated.", ""]
    rows = []
    for model in RECIPE_ORDER:
        states = [_audit_state(audit_root, window, model) for window in PRIMARY_WINDOWS + SECONDARY_WINDOWS]
        fitted = [state["fitted_state"] for state in states]
        features = fitted[0].get("feature_preset", "unknown")
        actual_params = _hyperparameter_summary(model, states)
        fit_times = [float(state["fit_seconds"]) for state in states]
        rows.append([model, features, actual_params, _state_summary(model, states),
                     f"{min(fit_times):.3f}–{max(fit_times):.3f} s"])
    chunks += [_md_table(["Recipe", "Feature preset", "Actual parameter summary", "Fitted-state summary", "Fit time range"], rows), "",
               "Source: `fit_audits/W1.json` through `W6.json`, plus `H12_E.json` and `H12_L.json`. HGB settings shown are actual estimator attributes saved after fitting; the R1_D categorical count reflects its fitted native categorical mask.", "",
               "## Scope and interpretation", "",
               "- No official-test predictions were loaded or plotted.",
               "- Primary and H12 windows overlap in validation rows, so H12 results are not independent replications.",
               "- Tail definitions use fixed `threat_rank <= 5` and `>= 95`; they are distinct from TP1 tercile partitions.",
               "- All tables are descriptive evidence for the Round 1 review. No final model is selected.", ""]
    return "\n".join(chunks)


def _plot_window_heatmap(results: dict[str, Any], figure_dir: Path) -> None:
    values = np.array([[_job(results, "primary", window, model)["metrics"]["mae"]
                        for window in PRIMARY_WINDOWS] for model in RECIPE_ORDER])
    fig, ax = plt.subplots(figsize=(11.2, 5.6), constrained_layout=True)
    image = ax.imshow(values, cmap="YlOrRd", aspect="auto", vmin=float(np.nanmin(values)), vmax=float(np.nanmax(values)))
    ax.set_xticks(range(len(PRIMARY_WINDOWS)), labels=PRIMARY_WINDOWS)
    ax.set_yticks(range(len(RECIPE_ORDER)), labels=RECIPE_ORDER)
    ax.set_xlabel("Primary validation window (cycle range advances left to right)")
    ax.set_ylabel("Frozen recipe")
    ax.set_title("Raw MAE by recipe and primary window")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i,j]:.2f}", ha="center", va="center",
                    color="white" if values[i, j] > (values.min() + values.max()) / 2 else "#222222", fontsize=8)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.88)
    colorbar.set_label("MAE (threat_rank points)")
    fig.savefig(figure_dir / "round1_primary_window_mae_heatmap.png", dpi=180)
    plt.close(fig)


def _plot_cycle_mae(results: dict[str, Any], figure_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.2, 6.2), constrained_layout=True)
    colors = {"R0_C": "#64748b", "R1_A": "#2563eb", "R1_C": "#e87924", "R1_D": "#15966a"}
    all_values = []
    for model in ["R0_C", "R1_A", "R1_C", "R1_D"]:
        per_cycle = {}
        for window in PRIMARY_WINDOWS:
            job = _job(results, "primary", window, model)
            for record in job["per_cycle"]:
                per_cycle[int(record["cycle"])] = float(record["mae"])
        cycles = sorted(per_cycle)
        values = [per_cycle[x] for x in cycles]
        all_values.extend(values)
        ax.plot(cycles, values, marker="o", markersize=3.7, linewidth=1.7, label=model, color=colors[model])
    ax.set_xticks(range(56, 80, 1))
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    ax.set_xlabel("Validation cycle_code")
    ax.set_ylabel("Raw MAE (threat_rank points)")
    ax.set_title("Primary per-cycle MAE for four reference recipes")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Recipe", ncol=2, frameon=False)
    pad = max(1.0, (max(all_values) - min(all_values)) * 0.08)
    ax.set_ylim(max(0, min(all_values) - pad), max(all_values) + pad)
    fig.savefig(figure_dir / "round1_primary_per_cycle_mae.png", dpi=180)
    plt.close(fig)


def _plot_tail_metrics(results: dict[str, Any], figure_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.1), constrained_layout=True)
    tail_specs = [("tail_low_le5", "Low tail (actual <= 5)"),
                  ("tail_high_ge95", "High tail (actual >= 95)")]
    colors = {"R0_C": "#64748b", "R1_B": "#2563eb", "R1_C": "#e87924", "R1_D": "#15966a"}
    for row, (key, label) in enumerate(tail_specs):
        for col, (metric, title, color) in enumerate([("mae", "MAE", "#2563eb"), ("signed_error", "Signed bias", "#b45309")]):
            ax = axes[row, col]
            vals, ns = [], []
            for model in TAIL_MODELS:
                m = results["aggregate"]["primary"][model]["subgroups"][key]
                vals.append(float(m[metric]))
                ns.append(int(m["n"]))
            y = np.arange(len(TAIL_MODELS))
            if metric == "mae":
                ax.barh(y, vals, color=[colors[m] for m in TAIL_MODELS], alpha=0.86, height=0.65)
                hi = max(vals)
                ax.set_xlim(0, hi * 1.40)
                for yi, value, n in zip(y, vals, ns):
                    ax.text(value + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.015, yi, f"{value:.2f} (n={n:,})", va="center", fontsize=8)
            else:
                ax.axvline(0, color="#374151", linewidth=1, alpha=0.75)
                ax.scatter(vals, y, c=[colors[m] for m in TAIL_MODELS], s=52, zorder=3)
                span = max(vals) - min(vals)
                pad = max(1.0, span * 0.85)
                ax.set_xlim(min(0, min(vals) - pad), max(0, max(vals) + pad))
                for yi, value, n in zip(y, vals, ns):
                    ax.text(value + pad * 0.035, yi, f"{value:+.2f} (n={n:,})", va="center", fontsize=8)
            ax.set_yticks(y, labels=TAIL_MODELS)
            ax.invert_yaxis()
            ax.set_title(f"{label}: {title}")
            ax.set_xlabel("threat_rank points")
            ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Primary extreme-target errors (tail membership defined by actual threat_rank)", fontsize=13)
    fig.savefig(figure_dir / "round1_primary_tail_mae_bias.png", dpi=180)
    plt.close(fig)


def _prediction_bundle(results: dict[str, Any], artifact_dir: Path, data_dir: str | Path | None) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    run_id = str(results["run_id"])
    if not run_id or any(sep in run_id for sep in ("/", "\\")) or run_id in {".", ".."}:
        raise ValueError("Unsafe run_id in results artifact.")
    if not artifact_dir.is_dir():
        raise FileNotFoundError(f"Saved validation predictions are required for the distribution plot: {artifact_dir}")
    preds_by_model: dict[str, list[np.ndarray]] = {model: [] for model in HIST_MODELS}
    reference_rows: list[np.ndarray] = []
    for window in PRIMARY_WINDOWS:
        window_rows = None
        for model in HIST_MODELS:
            path = artifact_dir / f"{window}__{model}.npz"
            if not path.is_file():
                raise FileNotFoundError(f"Missing saved validation prediction: {path}")
            job = _job(results, "primary", window, model)
            with np.load(path, allow_pickle=False) as data:
                rows = np.asarray(data["validation_rows"], dtype=np.int64)
                pred = np.asarray(data["prediction"], dtype=float)
            if pred.shape != (len(rows),) or not np.isfinite(pred).all():
                raise ValueError(f"Invalid saved prediction array for {window}/{model}")
            digest = hashlib.sha256(np.asarray(pred, dtype="<f8").tobytes()).hexdigest()
            if digest != job["prediction_sha256"]:
                raise ValueError(f"Prediction digest mismatch for {window}/{model}")
            if window_rows is None:
                window_rows = rows
            elif not np.array_equal(window_rows, rows):
                raise ValueError(f"Validation row alignment differs across recipes in {window}")
            preds_by_model[model].append(pred)
        reference_rows.append(window_rows)
    rows = np.concatenate(reference_rows)
    if len(np.unique(rows)) != len(rows):
        raise ValueError("Primary validation row positions overlap; distribution pool must be unique.")
    files = task_files("regression", data_dir)
    if files.get("train") is None:
        raise FileNotFoundError("Official regression Train.csv is required for target histogram labels.")
    target = pd.read_csv(files["train"], usecols=["threat_rank"])["threat_rank"].to_numpy(dtype=float)
    if rows.min() < 0 or rows.max() >= len(target):
        raise ValueError("Saved validation row position is outside the training data.")
    actual = target[rows]
    if not np.isfinite(actual).all():
        raise ValueError("Non-finite validation targets in official Train.csv.")
    predictions = {model: np.concatenate(parts) for model, parts in preds_by_model.items()}
    return actual, predictions


def _plot_prediction_histograms(actual: np.ndarray, predictions: dict[str, np.ndarray], figure_dir: Path) -> None:
    if any(len(values) != len(actual) for values in predictions.values()):
        raise ValueError("Actual and predicted distributions are not aligned.")
    min_prediction = min(float(np.min(x)) for x in predictions.values())
    max_prediction = max(float(np.max(x)) for x in predictions.values())
    lower = min(-5.0, 5.0 * np.floor(min_prediction / 5.0))
    upper = max(105.0, 5.0 * np.ceil(max_prediction / 5.0))
    bins = np.linspace(lower, upper, 29)
    fig, ax = plt.subplots(figsize=(11.5, 6.2), constrained_layout=True)
    ax.hist(actual, bins=bins, density=True, histtype="step", linewidth=2.3, color="#111827",
            label=f"Actual (n={len(actual):,})")
    colors = {"R1_B": "#2563eb", "R1_C": "#e87924", "R1_D": "#15966a"}
    for model in HIST_MODELS:
        pred = predictions[model]
        ax.hist(pred, bins=bins, density=True, histtype="step", linewidth=1.9, color=colors[model],
                label=f"{model} raw prediction (n={len(pred):,})")
    ax.axvline(0, color="#6b7280", linestyle="--", linewidth=1, alpha=0.65)
    ax.axvline(100, color="#6b7280", linestyle="--", linewidth=1, alpha=0.65)
    ax.set_xlim(lower, upper)
    ax.set_xlabel("threat_rank")
    ax.set_ylabel("Density")
    ax.set_title("Actual and raw-prediction distributions on pooled primary validation")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, ncol=2)
    fig.savefig(figure_dir / "round1_primary_actual_prediction_distributions.png", dpi=180)
    plt.close(fig)


def build_round1_report(
    results_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
    artifact_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Build factual Markdown tables and four figures from completed results.

    ``data_dir`` is the task root expected by ``task_files`` (the directory
    containing ``regression/``). Actual targets are loaded from official
    ``Train.csv`` only for the validation histogram; no source IDs or rows are
    written to report outputs.
    """
    results_file = Path(results_path) if results_path else PROJECT_ROOT / "reports/round1/results.json"
    if not results_file.is_file():
        raise FileNotFoundError(f"Round 1 results are required: {results_file}")
    results = _json(results_file)
    _validate_result(results)
    target_dir = Path(output_dir) if output_dir else results_file.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = target_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    audit_root = results_file.parent / "fit_audits"
    if not audit_root.is_dir():
        raise FileNotFoundError(f"Fit audit directory is required: {audit_root}")
    table_file = target_dir / "COMPARISON_TABLES.md"
    table_file.write_text(_build_tables(results, audit_root), encoding="utf-8")
    _plot_window_heatmap(results, figure_dir)
    _plot_cycle_mae(results, figure_dir)
    _plot_tail_metrics(results, figure_dir)
    run_id = str(results["run_id"])
    default_artifact_dir = PROJECT_ROOT / "experiments/runs" / run_id
    predictions_dir = Path(artifact_dir) if artifact_dir else default_artifact_dir
    actual, predictions = _prediction_bundle(results, predictions_dir, data_dir)
    _plot_prediction_histograms(actual, predictions, figure_dir)
    return {"tables": table_file, **{path.stem: path for path in sorted(figure_dir.glob("round1_*.png"))}}
