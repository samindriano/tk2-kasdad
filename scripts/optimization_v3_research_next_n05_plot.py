"""Render the final N05 result chart from its saved report, without refitting."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils.paths import PROJECT_ROOT


REPORT_ROOT = PROJECT_ROOT / "reports/optimization_v3/research_next_v1/results/N05/amendment_01"


def main() -> Path:
    result = json.loads((REPORT_ROOT / "results.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    cycle_rows = []
    for cutoff in (67, 75):
        primary = result["origins"][str(cutoff)]["primary"]
        for cycle, metrics in primary["per_cycle"].items():
            cycle_rows.append((int(cycle), cutoff,
                               metrics["candidate"]["coverage_80"],
                               metrics["past_residual_control"]["coverage_80"]))
    cycle_rows.sort()
    cycles = np.asarray([item[0] for item in cycle_rows])
    candidate = np.asarray([item[2] for item in cycle_rows])
    control = np.asarray([item[3] for item in cycle_rows])
    axes[0].plot(cycles, candidate, marker="o", linewidth=2, color="#276FBF",
                 label="Conditional q10/q90")
    axes[0].plot(cycles, control, marker="s", linewidth=2, color="#E07A1F",
                 label="Past residual interval")
    axes[0].axhline(0.8, color="#333333", linestyle="--", linewidth=1.3,
                    label="Nominal 80%")
    axes[0].axvspan(71.5, 75.5, color="#777777", alpha=0.10, label="Gap between primary windows")
    axes[0].set(xlabel="Cycle", ylabel="Observed 80% coverage",
                title="Coverage by LATE primary cycle", xticks=cycles, ylim=(0.5, 1.0))
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8, loc="lower right")

    palette = {67: "#276FBF", 75: "#C44536"}
    for cutoff in (67, 75):
        groups = result["origins"][str(cutoff)]["primary"]["candidate_width_rank_groups"]
        ordered = [groups[str(index)]["candidate"] for index in range(1, 6)
                   if groups[str(index)]["n"] > 0]
        axes[1].plot([row["mean_width"] for row in ordered],
                     [row["mean_absolute_reference_error"] for row in ordered],
                     marker="o", linewidth=2, color=palette[cutoff], label=f"Origin {cutoff}")
    axes[1].set(xlabel="Mean predicted interval width in width quintile",
                ylabel="Mean absolute R22 error", title="Wider intervals and realized point error")
    axes[1].grid(alpha=0.25)
    axes[1].legend(title="Forecast origin", fontsize=9)
    fig.suptitle("N05 — conditional uncertainty diagnostic\n"
                 "Primary windows: origin 67 → cycles 68–71; origin 75 → cycles 76–79",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output = REPORT_ROOT / "interval_diagnostic_final.png"
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return output


if __name__ == "__main__":
    print(main())
