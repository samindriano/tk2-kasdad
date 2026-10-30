"""Readable, source-backed reporting for a completed Classification Round 2 run.

This module only reads the completed results JSON, the frozen protocol, and the
Round 2 fit-audit summaries. It never fits a model, predicts, or reads Train/Test.
The renderer refuses to run until the matching Round 2 freeze and complete
80-job result artifact are present.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from src.models.classification_round2_evaluation import _report_path
from src.utils.paths import PROJECT_ROOT


DEFAULT_RESULTS = PROJECT_ROOT / "reports/classification_round2/results.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports/classification_round2"
DEFAULT_FREEZE = PROJECT_ROOT / "experiments/classification_round2/freeze.json"
FIT_ENDS = (55, 59, 63, 67, 71, 75)
POLICY_ORDER = ("fixed_0.5", "inner_pooled", "inner_shrink50", "inner_equal_cycle")
POLICY_LABELS = {
    "fixed_0.5": "Fixed 0.5",
    "inner_pooled": "Inner pooled",
    "inner_shrink50": "Pooled, shrunk 50%",
    "inner_equal_cycle": "Inner equal-cycle",
}
POLICY_COLORS = {
    "fixed_0.5": "#333333",
    "inner_pooled": "#0072B2",
    "inner_shrink50": "#D55E00",
    "inner_equal_cycle": "#009E73",
}
POLICY_MARKERS = {"fixed_0.5": "o", "inner_pooled": "s", "inner_shrink50": "^", "inner_equal_cycle": "D"}
ROUTE_RECIPES = ("C4_LR", "C4_HGB")
ROUTE_FAMILY = {"C4_LR": ("C1_A", "C2_B"), "C4_HGB": ("C1_B", "C2_B")}
POLICY_FIGURE_RECIPES = ("C2_B", "C4_LR", "C4_HGB", "C5_LOW", "C5_HIGH")
COMPONENT_FIGURE_RECIPES = ("C1_A", "C3_B", "C2_B", "C3_C", "C5_LOW", "C5_HIGH")
CYCLE_FIGURE_RECIPES = POLICY_FIGURE_RECIPES
SUBGROUP_REPORT_KEYS = (
    "EF0", "EF1", "stream_missing", "stream_present",
    "EF0_stream_missing", "EF0_stream_present", "EF1_stream_missing", "EF1_stream_present",
    "impact_low", "impact_middle", "impact_high",
    "org_code_missing", "org_code_present", "module_code_missing", "module_code_present",
    "pattern_code_missing", "pattern_code_present", "all_codes_known_nonmissing",
    "any_unseen_code", "any_missing_code",
)
SUBGROUP_LABELS = {
    "EF0": "EF=0", "EF1": "EF=1", "stream_missing": "Stream missing",
    "stream_present": "Stream present", "EF0_stream_missing": "EF=0 × missing stream",
    "EF0_stream_present": "EF=0 × present stream", "EF1_stream_missing": "EF=1 × missing stream",
    "EF1_stream_present": "EF=1 × present stream", "impact_low": "Impact low tercile",
    "impact_middle": "Impact middle tercile", "impact_high": "Impact high tercile",
    "org_code_missing": "Org code missing", "org_code_present": "Org code present",
    "module_code_missing": "Module code missing", "module_code_present": "Module code present",
    "pattern_code_missing": "Pattern code missing", "pattern_code_present": "Pattern code present",
    "all_codes_known_nonmissing": "All codes known/nonmissing", "any_unseen_code": "Any unseen code",
    "any_missing_code": "Any code missing",
}
FIGURE_FILENAMES = (
    "classification_round2_fixed_window_f1_heatmap.png",
    "classification_round2_fixed_pooled_comparison.png",
    "classification_round2_missing_present_f1_and_routes.png",
    "classification_round2_policy_effects.png",
    "classification_round2_missing_component_decomposition.png",
    "classification_round2_fixed_cycle_f1_ppr.png",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    number = float(value)
    if not np.isfinite(number):
        return "—"
    if round(number, digits) == 0:
        return f"{0.0:.{digits}f}"
    return f"{number:.{digits}f}"


def _fmt_count(value: Any) -> str:
    return "—" if value is None else f"{int(value):,}"


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(_cell(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _aggregate_metric(results: dict, family: str, policy: str, recipe: str) -> dict:
    try:
        return results["aggregate"][family][policy][recipe]
    except KeyError as exc:
        raise ValueError(f"Completed result is missing {family}/{policy}/{recipe} aggregate.") from exc


def _job_map(results: dict) -> dict[tuple[str, str, str], dict]:
    mapped = {}
    for job in results["jobs"]:
        key = (job["family"], job["window"], job["model"])
        if key in mapped:
            raise ValueError(f"Duplicate evaluation job in result: {key}")
        mapped[key] = job
    return mapped


def _validate_complete_artifacts(results: dict, protocol: dict, freeze: dict) -> dict:
    if freeze.get("status") != "FROZEN_BEFORE_ROUND2_DIAGNOSTIC_AND_MODEL_FITS":
        raise RuntimeError("The Round 2 freeze receipt is not in its frozen-before-fit state.")
    if results.get("status") != "CLASSIFICATION_ROUND2_COMPLETE_REVIEW_NO_FINAL_MODEL":
        raise RuntimeError("Reporting requires a complete Round 2 result with the review-only status.")
    if results.get("freeze_signature") != freeze.get("signature"):
        raise RuntimeError("Results do not belong to the current Round 2 freeze receipt.")
    if len(results.get("jobs", [])) != 80 or results.get("total_evaluation_jobs") != 80:
        raise RuntimeError("Reporting requires all 80 declared Round 2 evaluation jobs.")
    recipe_ids = [recipe["id"] for recipe in protocol["recipes"]]
    if len(recipe_ids) != 10:
        raise RuntimeError("Round 2 reporting expects the frozen ten-recipe comparison.")
    expected = {
        ("primary", window["id"], recipe)
        for window in protocol["primary_windows"] for recipe in recipe_ids
    } | {
        ("secondary", window["id"], recipe)
        for window in protocol["secondary_windows"] for recipe in recipe_ids
    }
    jobs = _job_map(results)
    if set(jobs) != expected:
        raise RuntimeError("The result job family/window/recipe grid is incomplete or contains extras.")
    for family in ("primary", "secondary"):
        for policy in POLICY_ORDER:
            for recipe in recipe_ids:
                metric = _aggregate_metric(results, family, policy, recipe)["metrics"]
                if metric.get("n") != 38588:
                    raise RuntimeError(
                        f"Expected full 38,588-row {family}/{policy}/{recipe} aggregate; got {metric.get('n')!r}."
                    )
    return jobs


def _fit_audits(results_path: Path) -> dict[int, dict]:
    root = results_path.parent / "fit_audits"
    audits = {}
    for end in FIT_ENDS:
        path = root / f"fit{end}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing Round 2 fit audit: {path}")
        audit = _read_json(path)
        if not all(recipe in audit for recipe in COMPONENT_FIGURE_RECIPES):
            missing = [recipe for recipe in COMPONENT_FIGURE_RECIPES if recipe not in audit]
            raise ValueError(f"Fit audit {path.name} is missing coefficient-audit recipes {missing}.")
        for recipe in COMPONENT_FIGURE_RECIPES:
            entry = audit[recipe]
            if "coefficients" not in entry or "component_statistics" not in entry:
                raise ValueError(f"Fit audit {path.name}/{recipe} lacks coefficient/component summaries.")
            if entry["component_statistics"].get("linear") is not True:
                raise ValueError(f"Expected linear logit decomposition for {path.name}/{recipe}.")
        audits[end] = audit
    return audits


def _fixed_candidate_rows(results: dict, recipes: list[str]) -> list[list[Any]]:
    rows = []
    for recipe in recipes:
        primary = _aggregate_metric(results, "primary", "fixed_0.5", recipe)
        secondary = _aggregate_metric(results, "secondary", "fixed_0.5", recipe)
        pm, sm = primary["metrics"], secondary["metrics"]
        rows.append([
            recipe,
            _fmt(pm["macro_f1"]), _fmt(primary["equal_cycle_macro_f1"]),
            _fmt(pm["average_precision"]), _fmt(pm["brier_score"]),
            _fmt(sm["macro_f1"]), _fmt(secondary["equal_cycle_macro_f1"]),
            _fmt(sm["average_precision"]), _fmt(sm["brier_score"]),
        ])
    return rows


def _fixed_window_rows(jobs: dict, recipes: list[str], protocol: dict) -> list[list[Any]]:
    windows = [("primary", window["id"]) for window in protocol["primary_windows"]]
    windows += [("secondary", window["id"]) for window in protocol["secondary_windows"]]
    return [[recipe] + [
        _fmt(jobs[(family, window, recipe)]["policies"]["fixed_0.5"]["metrics"]["macro_f1"])
        for family, window in windows
    ] for recipe in recipes]


def _all_recipe_per_cycle_rows(jobs: dict, recipes: list[str]) -> list[list[Any]]:
    rows = []
    for cycle in range(56, 80):
        values = [cycle]
        for recipe in recipes:
            matches = [item for window in ("W1", "W2", "W3", "W4", "W5", "W6")
                       for item in jobs[("primary", window, recipe)]["policies"]["fixed_0.5"]["per_cycle"]
                       if int(item["cycle"]) == cycle]
            if len(matches) != 1:
                raise ValueError(f"Expected one fixed per-cycle score for {cycle}/{recipe}.")
            values.append(_fmt(matches[0]["macro_f1"]))
        rows.append(values)
    return rows


def _policy_effect_rows(results: dict) -> list[list[Any]]:
    rows = []
    for family in ("primary", "secondary"):
        for recipe in POLICY_FIGURE_RECIPES:
            fixed = _aggregate_metric(results, family, "fixed_0.5", recipe)
            fixed_pooled = fixed["metrics"]["macro_f1"]
            fixed_cycle = fixed["equal_cycle_macro_f1"]
            values = [family, recipe, _fmt(fixed_pooled), _fmt(fixed_cycle)]
            for policy in POLICY_ORDER[1:]:
                current = _aggregate_metric(results, family, policy, recipe)
                values.extend([
                    _fmt(current["metrics"]["macro_f1"], 6),
                    _fmt(current["metrics"]["macro_f1"] - fixed_pooled, 6),
                    _fmt(current["equal_cycle_macro_f1"], 6),
                    _fmt(current["equal_cycle_macro_f1"] - fixed_cycle, 6),
                ])
            rows.append(values)
    return rows


def _threshold_rows(jobs: dict) -> list[list[Any]]:
    rows = []
    for recipe in POLICY_FIGURE_RECIPES:
        for policy in POLICY_ORDER:
            values = [recipe, policy]
            for end in FIT_ENDS:
                candidates = [job for (family, _window, model), job in jobs.items()
                              if model == recipe and int(job["fit_group"]) == end]
                if not candidates:
                    raise ValueError(f"No evaluated threshold record for fit{end}/{recipe}.")
                thresholds = [candidate["thresholds"][policy]["threshold"]
                              for candidate in candidates]
                if not np.allclose(thresholds, thresholds[0], rtol=0, atol=0):
                    raise ValueError(f"Threshold differs across windows sharing fit{end}/{recipe}/{policy}.")
                values.append(_fmt(thresholds[0], 4))
            rows.append(values)
    return rows


def _classwise_rows(results: dict, recipes: list[str]) -> list[list[Any]]:
    rows = []
    for family in ("primary", "secondary"):
        for recipe in recipes:
            metrics = _aggregate_metric(results, family, "fixed_0.5", recipe)["metrics"]
            per_class = {int(value["label"]): value for value in metrics["per_class"]}
            rows.append([
                family, recipe, _fmt_count(metrics["n"]), metrics["false_positive"], metrics["false_negative"],
                _fmt(per_class[0]["precision"]), _fmt(per_class[0]["recall"]), _fmt(per_class[0]["f1"]),
                _fmt(per_class[1]["precision"]), _fmt(per_class[1]["recall"]), _fmt(per_class[1]["f1"]),
            ])
    return rows


def _subgroup_matrix_rows(results: dict, recipes: list[str], subgroup_keys: tuple[str, ...]) -> list[list[Any]]:
    rows = []
    for family in ("primary", "secondary"):
        for recipe in recipes:
            result = _aggregate_metric(results, family, "fixed_0.5", recipe)
            cells = []
            for subgroup in subgroup_keys:
                metrics = result["subgroups"].get(subgroup)
                if metrics is None:
                    raise ValueError(f"Subgroup {subgroup!r} missing from {family}/{recipe} aggregate.")
                cells.append(
                    f"n={_fmt_count(metrics['n'])}; F1={_fmt(metrics['macro_f1'])}; "
                    f"F1₀/F1₁={_fmt(metrics['per_class'][0]['f1'])}/{_fmt(metrics['per_class'][1]['f1'])}; "
                    f"FP/FN={metrics['false_positive']}/{metrics['false_negative']}"
                )
            rows.append([family, recipe] + cells)
    return rows


def _window_meta_rows(jobs: dict, protocol: dict) -> list[list[Any]]:
    rows = []
    seen = set()
    for family, windows in (("primary", protocol["primary_windows"]),
                            ("secondary", protocol["secondary_windows"])):
        for window in windows:
            key = (family, window["id"])
            if key in seen:
                continue
            seen.add(key)
            job = next(value for (job_family, job_window, _recipe), value in jobs.items()
                       if (job_family, job_window) == key)
            lower, upper = job["impact_subgroup_cutpoints_from_fit"]
            rows.append([
                family, window["id"], job["fit_group"], _fmt_count(job["fit_rows"]),
                _fmt_count(job["validation_rows"]), _fmt(lower, 4), _fmt(upper, 4),
            ])
    return rows


def _routing_invariance_rows(jobs: dict) -> list[list[Any]]:
    rows = []
    for family, window, _recipe in jobs:
        if _recipe not in ROUTE_RECIPES:
            continue
        branch_missing, branch_present = ROUTE_FAMILY[_recipe]
        job = jobs[(family, window, _recipe)]
        for stream_subgroup, branch in (("stream_missing", branch_missing),
                                        ("stream_present", branch_present)):
            route_metric = job["policies"]["fixed_0.5"]["subgroups"][stream_subgroup]
            branch_metric = jobs[(family, window, branch)]["policies"]["fixed_0.5"]["subgroups"][stream_subgroup]
            equal = route_metric == branch_metric
            rows.append([family, window, _recipe, stream_subgroup, branch,
                         "PASS" if equal else "FAIL", _fmt_count(route_metric["n"]),
                         _fmt(route_metric["macro_f1"])])
    return rows


def _per_cycle_rows(jobs: dict) -> list[list[Any]]:
    rows = []
    for cycle in range(56, 80):
        for recipe in CYCLE_FIGURE_RECIPES:
            records = []
            for window in ("W1", "W2", "W3", "W4", "W5", "W6"):
                records.extend(
                    item for item in jobs[("primary", window, recipe)]
                    ["policies"]["fixed_0.5"]["per_cycle"] if item["cycle"] == cycle
                )
            if len(records) != 1:
                raise ValueError(f"Expected exactly one primary per-cycle metric for {cycle}/{recipe}.")
            metric = records[0]
            rows.append([cycle, recipe, _fmt(metric["macro_f1"]), _fmt(metric["positive_prediction_rate"]),
                         _fmt(metric["positive_rate"]), _fmt_count(metric["n"])])
    return rows


def _report_tables(results: dict, protocol: dict, jobs: dict) -> dict[str, list[list[Any]]]:
    recipes = [recipe["id"] for recipe in protocol["recipes"]]
    return {
        "fixed_candidates_primary_h12": _fixed_candidate_rows(results, recipes),
        "fixed_all_windows": _fixed_window_rows(jobs, recipes, protocol),
        "policy_effects_selected": _policy_effect_rows(results),
        "thresholds_by_fit_history": _threshold_rows(jobs),
        "fixed_classwise": _classwise_rows(results, recipes),
        "fixed_subgroups_ef_stream": _subgroup_matrix_rows(
            results, recipes,
            ("EF0", "EF1", "stream_missing", "stream_present", "EF0_stream_missing",
             "EF0_stream_present", "EF1_stream_missing", "EF1_stream_present"),
        ),
        "fixed_subgroups_impact_codes": _subgroup_matrix_rows(
            results, recipes,
            ("impact_low", "impact_middle", "impact_high", "org_code_missing", "org_code_present",
             "module_code_missing", "module_code_present", "pattern_code_missing", "pattern_code_present",
             "all_codes_known_nonmissing", "any_unseen_code", "any_missing_code"),
        ),
        "window_fit_and_impact_cutpoints": _window_meta_rows(jobs, protocol),
        "routing_branch_invariance": _routing_invariance_rows(jobs),
        "fixed_per_cycle_all_candidates": _all_recipe_per_cycle_rows(jobs, [r["id"] for r in protocol["recipes"]]),
        "selected_fixed_per_cycle": _per_cycle_rows(jobs),
    }


def _source_note(results: dict, results_path: Path) -> str:
    return (f"Source: `{results_path.name}`; run `{results.get('run_id', 'unknown')}`; "
            f"Round 2 freeze `{str(results.get('freeze_signature', 'unknown'))[:12]}…`.")


def _write_markdown(path: Path, tables: dict[str, list[list[Any]]], results: dict, results_path: Path) -> None:
    recipes = [recipe["id"] for recipe in _read_json(PROJECT_ROOT / "experiments/classification_round2/protocol.json")["recipes"]]
    lines = [
        "# Classification Round 2 comparison tables",
        "",
        _source_note(results, results_path),
        "",
        "This is adaptive development evidence using Round 1 cycles 56–79. H12 windows reuse cycles already present in primary validations, so those summaries are correlated adaptive-development evidence, not an independent holdout. No official Test predictions, calibration fit, final model, or submission are included.",
        "",
        "## Fixed 0.5: all ten candidates, pooled and equal-cycle summaries",
        "",
        _markdown_table(
            ["Recipe", "Primary F1", "Primary equal-cycle F1", "Primary AP", "Primary Brier",
             "H12 F1", "H12 equal-cycle F1", "H12 AP", "H12 Brier"],
            tables["fixed_candidates_primary_h12"],
        ),
        "",
        "## Fixed 0.5: every evaluation window",
        "",
        _markdown_table(["Recipe", "W1", "W2", "W3", "W4", "W5", "W6", "H12_E", "H12_L"],
                        tables["fixed_all_windows"]),
        "",
        "## Fit histories and impact cutpoints",
        "",
        _markdown_table(["Family", "Window", "Fit end", "Fit rows", "Validation rows", "Impact 1/3", "Impact 2/3"],
                        tables["window_fit_and_impact_cutpoints"]),
        "",
        "## Four-policy effects on the prespecified candidates",
        "",
        _markdown_table(
            ["Family", "Recipe", "Fixed pooled F1", "Fixed equal-cycle F1",
             "Inner pooled F1", "Δ pooled", "Inner pooled equal-cycle F1", "Δ equal-cycle",
             "Shrink pooled F1", "Δ pooled", "Shrink equal-cycle F1", "Δ equal-cycle",
             "Equal-cycle policy pooled F1", "Δ pooled", "Equal-cycle policy equal-cycle F1", "Δ equal-cycle"],
            tables["policy_effects_selected"],
        ),
        "",
        "## Selected policy thresholds by fit history",
        "",
        _markdown_table(["Recipe", "Policy", "fit55", "fit59", "fit63", "fit67", "fit71", "fit75"],
                        tables["thresholds_by_fit_history"]),
        "",
        "## Fixed 0.5: classwise results and error counts",
        "",
        _markdown_table(["Family", "Recipe", "Rows", "FP", "FN", "Class 0 P", "Class 0 R", "Class 0 F1",
                         "Class 1 P", "Class 1 R", "Class 1 F1"], tables["fixed_classwise"]),
        "",
        "## Fixed 0.5: entry-friction and stream subgroup metrics",
        "",
        "Cells report subgroup row count, pooled macro F1, classwise F1 values, and false-positive/false-negative counts.",
        "",
        _markdown_table(["Family", "Recipe"] + [SUBGROUP_LABELS[key] for key in
                        ("EF0", "EF1", "stream_missing", "stream_present", "EF0_stream_missing",
                         "EF0_stream_present", "EF1_stream_missing", "EF1_stream_present")],
                        tables["fixed_subgroups_ef_stream"]),
        "",
        "## Fixed 0.5: impact tercile and code-availability subgroup metrics",
        "",
        "Impact terciles use each window's outer-fit cutpoints. The table includes every Round 1 subgroup key for impact and known/unseen/missing code coverage.",
        "",
        _markdown_table(["Family", "Recipe"] + [SUBGROUP_LABELS[key] for key in
                        ("impact_low", "impact_middle", "impact_high", "org_code_missing", "org_code_present",
                         "module_code_missing", "module_code_present", "pattern_code_missing", "pattern_code_present",
                         "all_codes_known_nonmissing", "any_unseen_code", "any_missing_code")],
                        tables["fixed_subgroups_impact_codes"]),
        "",
        "## Routing branch invariance under the fixed policy",
        "",
        _markdown_table(["Family", "Window", "Route", "Stream subgroup", "Expected branch", "Exact match", "Rows", "Macro F1"],
                        tables["routing_branch_invariance"]),
        "",
        "## Fixed per-cycle F1 and rates for selected candidates",
        "",
        _markdown_table(["Cycle"] + recipes, tables["fixed_per_cycle_all_candidates"]),
        "",
        "Selected-candidate positive prediction rates (PPR) versus actual class-1 rate:",
        "",
        _markdown_table(["Cycle", "Recipe", "Macro F1", "Predicted positive rate", "Actual positive rate", "Rows"],
                        tables["selected_fixed_per_cycle"]),
        "",
        "## Figures",
        "",
    ]
    lines.extend(f"- `figures/{name}`" for name in FIGURE_FILENAMES)
    lines.extend(["", "All aggregate F1 values are recalculated over concatenated row-level predictions. Equal-cycle F1 is reported separately and is never substituted for pooled F1.", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_fixed_window_heatmap(path: Path, jobs: dict, protocol: dict, recipes: list[str], note: str) -> None:
    windows = [window["id"] for window in protocol["primary_windows"]]
    matrix = np.asarray([
        [float(jobs[("primary", window, recipe)]["policies"]["fixed_0.5"]["metrics"]["macro_f1"])
         for window in windows]
        for recipe in recipes
    ])
    fig, ax = plt.subplots(figsize=(10.5, 6.1))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(windows)), windows)
    ax.set_yticks(np.arange(len(recipes)), recipes)
    ax.set_xlabel("Primary validation window")
    ax.set_title("Fixed 0.5 macro F1 by recipe and validation window")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, f"{matrix[row, column]:.3f}", ha="center", va="center",
                    color="white" if matrix[row, column] < 0.55 else "black", fontsize=8)
    fig.colorbar(image, ax=ax, label="Macro F1", fraction=0.04, pad=0.03)
    fig.text(0.01, 0.01, note, fontsize=7, color="#444444")
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_fixed_primary_h12(path: Path, results: dict, recipes: list[str], note: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.8))
    y = np.arange(len(recipes))[::-1]
    primary = np.asarray([_aggregate_metric(results, "primary", "fixed_0.5", recipe)["metrics"]["macro_f1"]
                          for recipe in recipes], dtype=float)
    secondary = np.asarray([_aggregate_metric(results, "secondary", "fixed_0.5", recipe)["metrics"]["macro_f1"]
                            for recipe in recipes], dtype=float)
    for index, row in enumerate(y):
        ax.plot([primary[index], secondary[index]], [row + 0.12, row - 0.12], color="#A0A0A0", linewidth=1, zorder=1)
    ax.scatter(primary, y + 0.12, s=40, color="#0072B2", marker="o", label="Primary", zorder=3)
    ax.scatter(secondary, y - 0.12, s=40, color="#D55E00", marker="D", label="H12", zorder=3)
    ax.set_yticks(y, recipes)
    ax.set_xlim(0, 1.38)
    ax.set_xticks(np.arange(0, 1.01, 0.1))
    ax.set_xlabel("Pooled macro F1 (markers); exact values at right")
    ax.set_title("Fixed 0.5 pooled macro F1: primary and H12 windows")
    ax.grid(axis="x", alpha=0.22)
    ax.legend(loc="upper left", frameon=False)
    for index, row in enumerate(y):
        ax.text(1.035, row, f"P {primary[index]:.6f}   H12 {secondary[index]:.6f}",
                ha="left", va="center", fontsize=8, family="monospace")
    fig.text(0.01, 0.01, note, fontsize=7, color="#444444")
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_missing_present_routes(path: Path, results: dict, note: str) -> None:
    recipes = ("C1_A", "C1_B", "C2_B", "C4_LR", "C4_HGB")
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8), sharey=True)
    x = np.arange(len(recipes))
    for column, family in enumerate(("primary", "secondary")):
        for row, subgroup in enumerate(("stream_missing", "stream_present")):
            ax = axes[row, column]
            values = []
            for recipe in recipes:
                metric = _aggregate_metric(results, family, "fixed_0.5", recipe)["subgroups"][subgroup]
                values.append(float(metric["macro_f1"]))
            ax.plot(x, values, color="#777777", linewidth=1, alpha=0.7, zorder=1)
            for index, (recipe, value) in enumerate(zip(recipes, values)):
                color = {"C1_A": "#0072B2", "C1_B": "#56B4E9", "C2_B": "#009E73",
                         "C4_LR": "#D55E00", "C4_HGB": "#CC79A7"}[recipe]
                marker = "D" if recipe.startswith("C4_") else "o"
                ax.scatter(index, value, s=55, color=color, marker=marker, zorder=3)
                ax.annotate(f"{value:.3f}", (index, value), xytext=(0, 7 if index % 2 == 0 else -12),
                            textcoords="offset points", ha="center", fontsize=7)
            ax.set_xticks(x, recipes, rotation=25, ha="right")
            ax.set_ylim(0, 1.12)
            ax.set_yticks(np.arange(0, 1.01, 0.2))
            ax.grid(axis="y", alpha=0.2)
            subgroup_n = _aggregate_metric(results, family, "fixed_0.5", "C2_B")["subgroups"][subgroup]["n"]
            ax.set_title(f"{family.title()} · {SUBGROUP_LABELS[subgroup]} (n={subgroup_n:,})")
            if column == 0:
                ax.set_ylabel("Fixed 0.5 macro F1")
    fig.suptitle("Missing/present stream results and fixed-route branch comparisons", y=0.99)
    fig.text(0.5, 0.045,
             "Route checks: C4_LR missing→C1_A, C4_HGB missing→C1_B, both present→C2_B; exact metric checks are tabulated.",
             ha="center", fontsize=8)
    fig.text(0.01, 0.012, note, fontsize=7, color="#444444")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_policy_effects(path: Path, results: dict, jobs: dict, note: str) -> None:
    fig = plt.figure(figsize=(16, 10))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.05, 1.1), hspace=0.42, wspace=0.24)
    recipes = list(POLICY_FIGURE_RECIPES)
    policy_handles = [Line2D([0], [0], marker=POLICY_MARKERS[p], color=POLICY_COLORS[p],
                             label=POLICY_LABELS[p], linewidth=1.3)
                      for p in POLICY_ORDER]
    all_family_deltas = [
        float(_aggregate_metric(results, family, policy, recipe)["metrics"]["macro_f1"]) -
        float(_aggregate_metric(results, family, "fixed_0.5", recipe)["metrics"]["macro_f1"])
        for family in ("primary", "secondary")
        for policy in POLICY_ORDER for recipe in recipes
    ]
    delta_limit = max(max(abs(value) for value in all_family_deltas), 0.002) * 1.15
    for column, family in enumerate(("primary", "secondary")):
        ax = fig.add_subplot(grid[0, column])
        x = np.arange(len(recipes))
        fixed = [float(_aggregate_metric(results, family, "fixed_0.5", recipe)["metrics"]["macro_f1"])
                 for recipe in recipes]
        for policy in POLICY_ORDER:
            values = [float(_aggregate_metric(results, family, policy, recipe)["metrics"]["macro_f1"]) - base
                      for recipe, base in zip(recipes, fixed)]
            ax.plot(x, values, color=POLICY_COLORS[policy], marker=POLICY_MARKERS[policy], markersize=4,
                    linewidth=1.2, label=POLICY_LABELS[policy])
        ax.set_xticks(x, recipes, rotation=20, ha="right")
        ax.axhline(0.0, color="#444444", linewidth=0.8, linestyle="--")
        ax.set_ylim(-delta_limit, delta_limit)
        ax.grid(axis="y", alpha=0.22)
        ax.set_title(f"{family.title()} · pooled macro F1 change")
        ax.set_ylabel("Δ macro F1 vs fixed 0.5")
    axes = []
    # Five compact, aligned history panels share threshold units and history order.
    subgrid = grid[1, :].subgridspec(1, len(recipes), wspace=0.18)
    for recipe_index, recipe in enumerate(recipes):
        ax = fig.add_subplot(subgrid[0, recipe_index], sharey=axes[0] if axes else None)
        axes.append(ax)
        for policy in POLICY_ORDER:
            values = []
            for end in FIT_ENDS:
                jobs_at_end = [job for (_family, _window, model), job in jobs.items()
                               if model == recipe and int(job["fit_group"]) == end]
                values.append(float(jobs_at_end[0]["thresholds"][policy]["threshold"]))
            ax.plot(FIT_ENDS, values, marker=POLICY_MARKERS[policy], markersize=3,
                    color=POLICY_COLORS[policy], linewidth=1)
        ax.set_xticks(FIT_ENDS, [str(value) for value in FIT_ENDS], rotation=35, ha="right")
        ax.set_ylim(0.08, 0.92)
        ax.set_yticks(np.arange(0.1, 0.91, 0.1))
        ax.grid(alpha=0.18)
        ax.set_title(recipe, fontsize=8)
        ax.set_xlabel("Fit end cycle", fontsize=8)
        if recipe_index:
            ax.tick_params(axis="y", labelleft=False)
    axes[0].set_ylabel("Selected threshold")
    fig.legend(handles=policy_handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.text(0.01, 0.005, note, fontsize=7, color="#444444")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_missing_components(path: Path, audits: dict[int, dict], note: str) -> None:
    representative_end = 67
    representative_cycles = (68, 79)
    components = ("intercept", "numeric", "categorical", "text", "missing_indicator")
    colors = {"intercept": "#000000", "numeric": "#0072B2", "categorical": "#D55E00",
              "text": "#009E73", "missing_indicator": "#CC79A7"}
    markers = {"intercept": "o", "numeric": "s", "categorical": "^", "text": "D", "missing_indicator": "P"}
    model_values = {}
    for recipe in COMPONENT_FIGURE_RECIPES:
        group = audits[representative_end][recipe]["component_statistics"]["groups"]["stream_missing"]["components"]
        model_values[recipe] = {
            component: float(group[component]["mean"]) if group.get(component) is not None else 0.0
            for component in components
        }
    matrix = np.asarray([[model_values[recipe][component] for component in components]
                         for recipe in COMPONENT_FIGURE_RECIPES], dtype=float)
    low, high = min(float(matrix.min()), 0.0), max(float(matrix.max()), 0.0)
    pad = max((high - low) * 0.1, 0.05)
    fig, (ax, delta_ax) = plt.subplots(1, 2, figsize=(16, 6.5), gridspec_kw={"width_ratios": (1.4, 1)})
    recipe_colors = {"C1_A": "#0072B2", "C3_B": "#000000", "C2_B": "#009E73",
                     "C3_C": "#56B4E9", "C5_LOW": "#D55E00", "C5_HIGH": "#CC79A7"}
    x = np.arange(len(components))
    for recipe in COMPONENT_FIGURE_RECIPES:
        ax.plot(x, [model_values[recipe][component] for component in components],
                color=recipe_colors[recipe], marker="o", markersize=4, linewidth=1.2, label=recipe)
    ax.axhline(0.0, color="#555555", linewidth=0.9, linestyle="--")
    ax.set_xticks(x, [component.replace("_", " ") for component in components], rotation=18, ha="right")
    ax.set_ylim(low - pad, high + pad)
    ax.set_ylabel("Mean signed logit contribution")
    ax.set_title("Component means by model")
    ax.grid(axis="y", alpha=0.18)
    ax.legend(ncol=2, frameon=False, fontsize=8)

    reference = "C3_B"
    delta_recipes = [recipe for recipe in COMPONENT_FIGURE_RECIPES if recipe != reference]
    delta_matrix = np.asarray([
        [model_values[recipe][component] - model_values[reference][component] for component in components]
        for recipe in delta_recipes
    ], dtype=float)
    limit = max(float(np.max(np.abs(delta_matrix))), 0.01)
    image = delta_ax.imshow(delta_matrix, cmap="coolwarm", aspect="auto", vmin=-limit, vmax=limit)
    delta_ax.set_xticks(np.arange(len(components)), [component.replace("_", " ") for component in components],
                        rotation=18, ha="right")
    delta_ax.set_yticks(np.arange(len(delta_recipes)), [f"{recipe} − {reference}" for recipe in delta_recipes])
    delta_ax.set_title("Mean component difference from C3_B")
    for row in range(delta_matrix.shape[0]):
        for column in range(delta_matrix.shape[1]):
            delta_ax.text(column, row, f"{delta_matrix[row, column]:+.3f}",
                          ha="center", va="center", fontsize=7, color="black")
    fig.colorbar(image, ax=delta_ax, label="Logit contribution delta", fraction=0.045, pad=0.03)
    fig.suptitle("Missing-stream logit decomposition · fit 47–67, validation cycles 68–79", y=0.98)
    fig.text(0.01, 0.01,
             "All models share the same fit history and validation subset; text contribution on missing-stream rows is zero. "
             "Signed contribution zero is explicit. No cross-history scaler/vocabulary trend is implied. " + note,
             fontsize=7, color="#444444")
    fig.tight_layout(rect=(0, 0.055, 1, 0.93))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_cycle_f1_rates(path: Path, jobs: dict, note: str) -> None:
    fig, (ax_f1, ax_rate) = plt.subplots(2, 1, figsize=(13, 8), sharex=True, height_ratios=(1.05, 0.85))
    colors = {"C2_B": "#0072B2", "C4_LR": "#D55E00", "C4_HGB": "#CC79A7",
              "C5_LOW": "#009E73", "C5_HIGH": "#E69F00"}
    markers = {"C2_B": "o", "C4_LR": "s", "C4_HGB": "^", "C5_LOW": "D", "C5_HIGH": "P"}
    cycles = np.arange(56, 80)
    for recipe in CYCLE_FIGURE_RECIPES:
        per_cycle = {}
        for window in ("W1", "W2", "W3", "W4", "W5", "W6"):
            for item in jobs[("primary", window, recipe)]["policies"]["fixed_0.5"]["per_cycle"]:
                per_cycle[int(item["cycle"])] = item
        if sorted(per_cycle) != list(cycles):
            raise ValueError(f"Incomplete unique primary cycle series for {recipe}.")
        ax_f1.plot(cycles, [per_cycle[int(cycle)]["macro_f1"] for cycle in cycles],
                   marker=markers[recipe], markersize=3, linewidth=1.1, color=colors[recipe], label=recipe)
        ax_rate.plot(cycles, [per_cycle[int(cycle)]["positive_prediction_rate"] for cycle in cycles],
                     marker=markers[recipe], markersize=2.6, linewidth=1,
                     color=colors[recipe], label=f"{recipe} PPR")
    actual_rates = []
    for cycle in cycles:
        records = [item for window in ("W1", "W2", "W3", "W4", "W5", "W6")
                   for item in jobs[("primary", window, "C2_B")]["policies"]["fixed_0.5"]["per_cycle"]
                   if int(item["cycle"]) == int(cycle)]
        if len(records) != 1:
            raise ValueError(f"Expected exactly one actual positive rate for cycle {cycle}.")
        actual_rates.append(float(records[0]["positive_rate"]))
    ax_rate.plot(cycles, actual_rates, color="#000000", linestyle="--", linewidth=1.4,
                 marker="s", markersize=3, label="Actual rate")
    ax_f1.set_ylabel("Fixed 0.5 macro F1")
    ax_f1.set_ylim(0, 1.04)
    ax_f1.set_yticks(np.arange(0, 1.01, 0.1))
    ax_f1.grid(axis="y", alpha=0.2)
    fig.suptitle("Primary per-cycle fixed 0.5 F1 and positive-rate comparison", y=0.99)
    f1_handles, f1_labels = ax_f1.get_legend_handles_labels()
    fig.legend(f1_handles, f1_labels, ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.945))
    ax_rate.set_ylabel("Positive rate")
    ax_rate.set_xlabel("Validation cycle")
    ax_rate.set_ylim(0, 1.04)
    ax_rate.set_yticks(np.arange(0, 1.01, 0.1))
    ax_rate.set_xticks(np.r_[np.arange(56, 80, 2), 79])
    ax_rate.tick_params(axis="x", labelrotation=30)
    for label in ax_rate.get_xticklabels():
        label.set_ha("right")
    ax_rate.grid(axis="y", alpha=0.2)
    rate_handles, rate_labels = ax_rate.get_legend_handles_labels()
    fig.legend(rate_handles, rate_labels, ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.065))
    fig.text(0.01, 0.005, note, fontsize=7, color="#444444")
    fig.tight_layout(rect=(0, 0.18, 1, 0.89))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_round2_report(results_path=None, output_dir=None, data_dir=None) -> dict:
    """Render complete Round 2 comparison tables and six figures.

    ``data_dir`` is accepted for stable CLI/API shape and deliberately unused:
    reporting consumes frozen outputs only, never raw Train/Test data.
    """
    del data_dir
    results_file = Path(results_path) if results_path is not None else DEFAULT_RESULTS
    if not results_file.is_absolute():
        results_file = PROJECT_ROOT / results_file
    results_file = results_file.resolve()
    if not results_file.is_file():
        raise FileNotFoundError(f"Completed Round 2 results are not available: {results_file}")
    requested_output = Path(output_dir) if output_dir is not None else (
        results_file.parent if results_path is not None else DEFAULT_OUTPUT
    )
    if not requested_output.is_absolute():
        requested_output = PROJECT_ROOT / requested_output
    output = _report_path(requested_output)
    protocol_path = PROJECT_ROOT / "experiments/classification_round2/protocol.json"
    protocol = _read_json(protocol_path)
    if not DEFAULT_FREEZE.is_file():
        raise RuntimeError(f"Round 2 freeze receipt is missing: {DEFAULT_FREEZE}")
    freeze = _read_json(DEFAULT_FREEZE)
    if freeze.get("status") != "FROZEN_BEFORE_ROUND2_DIAGNOSTIC_AND_MODEL_FITS":
        raise RuntimeError("The Round 2 freeze receipt is not in its frozen-before-fit state.")
    results = _read_json(results_file)
    jobs = _validate_complete_artifacts(results, protocol, freeze)
    audits = _fit_audits(results_file)
    recipes = [recipe["id"] for recipe in protocol["recipes"]]
    tables = _report_tables(results, protocol, jobs)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    note = _source_note(results, results_file)
    plotters = (
        _plot_fixed_window_heatmap,
        _plot_fixed_primary_h12,
        _plot_missing_present_routes,
        _plot_policy_effects,
        _plot_missing_components,
        _plot_cycle_f1_rates,
    )
    arguments = (
        (jobs, protocol, recipes, note),
        (results, recipes, note),
        (results, note),
        (results, jobs, note),
        (audits, note),
        (jobs, note),
    )
    for filename, plotter, args in zip(FIGURE_FILENAMES, plotters, arguments):
        plotter(figures / filename, *args)
    markdown_path = output / "comparison_tables.md"
    _write_markdown(markdown_path, tables, results, results_file)
    return {
        "status": "CLASSIFICATION_ROUND2_REPORT_COMPLETE",
        "results_path": str(results_file),
        "output_dir": str(output),
        "tables": tables,
        "figures": [str(figures / name) for name in FIGURE_FILENAMES],
        "markdown_path": str(markdown_path),
        "route_invariance_checks": len(tables["routing_branch_invariance"]),
        "route_invariance_passes": sum(row[5] == "PASS" for row in tables["routing_branch_invariance"]),
    }
