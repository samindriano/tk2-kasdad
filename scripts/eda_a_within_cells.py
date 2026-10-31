"""Aggregate-only robustness check for diversity within exact length x cycle cells.

No record identifiers or raw text are written. The estimand is a common linear
within-cell association between TTR and the binary artifact label, after removing
each exact token-count x cycle cell's means.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SHA256 = "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c"
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
SUPPORT_GATES = (2, 5, 10)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_train(root: Path) -> Path:
    """Resolve classification/Train.csv under TP2_DATA_DIR or repository paths."""
    configured = os.environ.get("TP2_DATA_DIR")
    if configured:
        candidates = [Path(configured).expanduser().resolve()]
    else:
        candidates = [root / "data" / "raw", root / "data", root]
    for base in candidates:
        folder = base / "classification"
        if folder.is_dir():
            matches = [p for p in folder.iterdir()
                       if p.is_file() and p.suffix.lower() == ".csv"
                       and p.stem.lower() == "train"]
            if len(matches) > 1:
                raise ValueError(f"Ambiguous classification Train CSVs under {folder}")
            if matches:
                return matches[0]
    raise FileNotFoundError(
        "Could not resolve classification/Train.csv. Set TP2_DATA_DIR to the "
        "directory containing the classification folder."
    )


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _token_features(stream: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    missing = stream.isna() | stream.astype("string").str.strip().eq("").fillna(False)
    token_count = np.zeros(len(stream), dtype=np.int32)
    unique_count = np.zeros(len(stream), dtype=np.int32)
    # Raw values stay in memory and are never added to output frames.
    for i, value in enumerate(stream.array):
        if missing.iloc[i]:
            continue
        tokens = TOKEN_RE.findall(str(value).lower())
        token_count[i] = len(tokens)
        unique_count[i] = len(set(tokens))
    ttr = np.full(len(stream), np.nan, dtype=np.float64)
    valid = token_count > 0
    ttr[valid] = unique_count[valid] / token_count[valid]
    return missing.to_numpy(dtype=bool), token_count, ttr


def _cell_table(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (token_count, cycle), group in data.groupby(
            ["token_count", "cycle_code"], sort=True, observed=True):
        x = group["ttr"].to_numpy(dtype=np.float64)
        y = group["artifact_signal"].to_numpy(dtype=np.float64)
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        # At a fixed token count, TTR is an integer unique-count divided by
        # that length. Integer cross-products preserve exact covariance zeros;
        # floating centered dots can otherwise mislabel tiny residues as signs.
        u = np.rint(x * int(token_count)).astype(np.int64)
        yi = y.astype(np.int64)
        n_cell = int(len(group))
        sum_u, sum_y = int(u.sum()), int(yi.sum())
        cross_numerator = n_cell * int(np.dot(u, yi)) - sum_u * sum_y
        variance_numerator = n_cell * int(np.dot(u, u)) - sum_u**2
        sxx = variance_numerator / (n_cell * int(token_count)**2)
        sxy = cross_numerator / (n_cell * int(token_count))
        yss = sum_y * (n_cell - sum_y) / n_cell
        rows.append({
            "token_count": int(token_count),
            "cycle_code": int(cycle),
            "n": int(len(group)),
            "distinct_ttr": int(pd.Series(x).nunique()),
            "ttr_mean": float(x.mean()),
            "artifact_rate": float(y.mean()),
            "sxx_ttr_within_cell": sxx,
            "sxy_ttr_artifact_within_cell": sxy,
            "artifact_ss_within_cell": yss,
            "cell_slope_per_ttr_unit": sxy / sxx if sxx > 0 else None,
            "within_cell_correlation": (
                sxy / float(np.sqrt(sxx * yss)) if sxx > 0 and yss > 0 else None
            ),
        })
    return pd.DataFrame(rows)


def _summarize_gate(cells: pd.DataFrame, min_n: int, total_train: int,
                    tokenizable_n: int, present_n: int) -> tuple[dict, pd.DataFrame]:
    feature_ok = cells["distinct_ttr"] >= 2
    size_ok = cells["n"] >= min_n
    eligible_mask = feature_ok & size_ok
    eligible = cells.loc[eligible_mask].copy()
    eligible["min_cell_n"] = min_n
    eligible["slope_per_0_1_ttr"] = eligible["cell_slope_per_ttr_unit"] * 0.1
    if len(eligible):
        total_sxx = float(eligible["sxx_ttr_within_cell"].sum())
        total_sxy = float(eligible["sxy_ttr_artifact_within_cell"].sum())
        beta = total_sxy / total_sxx if total_sxx > 0 else None
        positive = eligible["sxy_ttr_artifact_within_cell"] > 0
        negative = eligible["sxy_ttr_artifact_within_cell"] < 0
        zero = ~(positive | negative)
        direction_counts = {
            "positive_cells": int(positive.sum()),
            "zero_cells": int(zero.sum()),
            "negative_cells": int(negative.sum()),
            "positive_rows": int(eligible.loc[positive, "n"].sum()),
            "zero_rows": int(eligible.loc[zero, "n"].sum()),
            "negative_rows": int(eligible.loc[negative, "n"].sum()),
            "positive_sxx_share": float(eligible.loc[positive, "sxx_ttr_within_cell"].sum() / total_sxx) if total_sxx else None,
            "negative_sxx_share": float(eligible.loc[negative, "sxx_ttr_within_cell"].sum() / total_sxx) if total_sxx else None,
            "zero_sxx_share": float(eligible.loc[zero, "sxx_ttr_within_cell"].sum() / total_sxx) if total_sxx else None,
        }
    else:
        total_sxx, total_sxy, beta = 0.0, 0.0, None
        direction_counts = {
            "positive_cells": 0, "zero_cells": 0, "negative_cells": 0,
            "positive_rows": 0, "zero_rows": 0, "negative_rows": 0,
            "positive_sxx_share": None, "negative_sxx_share": None,
            "zero_sxx_share": None,
        }

    per_cycle = []
    for cycle in range(47, 80):
        group = eligible.loc[eligible["cycle_code"] == cycle]
        sxx = float(group["sxx_ttr_within_cell"].sum())
        sxy = float(group["sxy_ttr_artifact_within_cell"].sum())
        per_cycle.append({
            "min_cell_n": min_n,
            "cycle_code": cycle,
            "eligible_cells": int(len(group)),
            "support_rows": int(group["n"].sum()),
            "sxx_ttr_within_cell": sxx,
            "sxy_ttr_artifact_within_cell": sxy,
            "slope_per_0_1_ttr": 0.1 * sxy / sxx if sxx > 0 else None,
            "share_total_sxx": sxx / total_sxx if total_sxx > 0 else None,
        })
    cycle_frame = pd.DataFrame(per_cycle)
    cycle_frame["leave_one_cycle_out_slope_per_0_1_ttr"] = cycle_frame.apply(
        lambda row: 0.1 * (total_sxy - row["sxy_ttr_artifact_within_cell"]) /
        (total_sxx - row["sxx_ttr_within_cell"])
        if total_sxx - row["sxx_ttr_within_cell"] > 0 else None,
        axis=1,
    )
    top3_share = float(cycle_frame["share_total_sxx"].fillna(0).nlargest(3).sum()) if total_sxx else None
    max_share = float(cycle_frame["share_total_sxx"].fillna(0).max()) if total_sxx else None
    supported_cycle_slopes = cycle_frame.loc[
        cycle_frame["sxx_ttr_within_cell"] > 0, "slope_per_0_1_ttr"
    ]
    summary = {
        "minimum_cell_n": min_n,
        "eligibility_rule": "cell n >= minimum and at least 2 distinct TTR values; target variation is not an eligibility condition",
        "all_joint_cells": int(len(cells)),
        "eligible_cells": int(eligible_mask.sum()),
        "excluded_small_n_cells": int((~size_ok).sum()),
        "excluded_insufficient_ttr_variation_cells": int((~feature_ok).sum()),
        "excluded_both_small_n_and_insufficient_ttr_variation_cells": int((~size_ok & ~feature_ok).sum()),
        "excluded_small_n_only_cells": int((~size_ok & feature_ok).sum()),
        "excluded_insufficient_ttr_variation_only_cells": int((size_ok & ~feature_ok).sum()),
        "total_excluded_cells": int((~eligible_mask).sum()),
        "eligible_support_rows": int(eligible["n"].sum()),
        "coverage_of_tokenizable_rows": float(eligible["n"].sum() / tokenizable_n) if tokenizable_n else None,
        "coverage_of_nonblank_rows": float(eligible["n"].sum() / present_n) if present_n else None,
        "coverage_of_all_train_rows": float(eligible["n"].sum() / total_train) if total_train else None,
        "sum_within_cell_sxx": total_sxx,
        "sum_within_cell_sxy": total_sxy,
        "common_within_cell_slope_per_ttr_unit": beta,
        "common_within_cell_slope_per_0_1_ttr": 0.1 * beta if beta is not None else None,
        "weighted_within_cell_correlation": (
            total_sxy / float(np.sqrt(total_sxx * eligible["artifact_ss_within_cell"].sum()))
            if total_sxx > 0 and eligible["artifact_ss_within_cell"].sum() > 0 else None
        ),
        "cycle_concentration": {
            "cycles_with_eligible_cells": int((cycle_frame["eligible_cells"] > 0).sum()),
            "max_cycle_sxx_share": max_share,
            "top_3_cycle_sxx_share": top3_share,
            "positive_cycle_slopes": int((supported_cycle_slopes > 0).sum()),
            "zero_cycle_slopes": int((supported_cycle_slopes == 0).sum()),
            "negative_cycle_slopes": int((supported_cycle_slopes < 0).sum()),
            "loco_slopes_positive_cycles": int((cycle_frame["leave_one_cycle_out_slope_per_0_1_ttr"] > 0).sum()),
            "loco_slopes_zero_cycles": int((cycle_frame["leave_one_cycle_out_slope_per_0_1_ttr"] == 0).sum()),
            "loco_slopes_negative_cycles": int((cycle_frame["leave_one_cycle_out_slope_per_0_1_ttr"] < 0).sum()),
        },
        "cell_direction": direction_counts,
    }
    return summary, cycle_frame


def run(root=None, output_dir=None):
    root = Path(root or PROJECT_ROOT).expanduser().resolve()
    output_dir = Path(output_dir or root / "reports" / "eda_a_robustness" / "within_cells").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = _resolve_train(root)
    actual_hash = _sha256(train_path)
    if actual_hash != EXPECTED_SHA256:
        raise ValueError(f"Classification Train hash mismatch: {actual_hash}")
    df = pd.read_csv(train_path)
    required = {"cycle_code", "signal_stream", "artifact_signal"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns: {sorted(required - set(df.columns))}")
    if len(df) != 50608:
        raise ValueError(f"Unexpected Train row count: {len(df)}")
    if df["artifact_signal"].isna().any() or not df["artifact_signal"].isin([0, 1]).all():
        raise ValueError("artifact_signal must be complete and binary")
    if not df["cycle_code"].isin(range(47, 80)).all():
        raise ValueError("cycle_code outside official Train range 47-79")

    missing, token_count, ttr = _token_features(df["signal_stream"])
    data = pd.DataFrame({
        "cycle_code": df["cycle_code"].to_numpy(dtype=np.int16),
        "artifact_signal": df["artifact_signal"].to_numpy(dtype=np.int8),
        "token_count": token_count,
        "ttr": ttr,
        "stream_missing": missing,
    })
    present = data.loc[~data["stream_missing"]].copy()
    tokenizable = present.loc[present["ttr"].notna()].copy()
    if len(tokenizable) != 47204:
        raise ValueError(f"Unexpected tokenizable-row support: {len(tokenizable)}")
    cells = _cell_table(tokenizable)

    summaries = []
    cycle_frames = []
    for gate in SUPPORT_GATES:
        summary, cycle_frame = _summarize_gate(
            cells, gate, len(data), len(tokenizable), len(present)
        )
        summaries.append(summary)
        cycle_frames.append(cycle_frame)
    summary_by_gate = {str(item["minimum_cell_n"]): item for item in summaries}
    cycles = pd.concat(cycle_frames, ignore_index=True)

    # Purely aggregated exports: exact cell counts/cycles, no IDs or text fields.
    # Publish granular aggregates only for primary-supported cells. Singleton
    # outcome summaries would effectively expose partial individual records.
    # Gates 2/10 remain fully calculated but are exported as pooled/cycle summaries.
    gate_cells = cells.loc[(cells["n"] >= 5) & (cells["distinct_ttr"] >= 2)].copy()
    gate_cells["minimum_cell_n"] = 5
    gate_cells.to_csv(output_dir / "exact_length_cycle_cells.csv", index=False)
    cycles.to_csv(output_dir / "cycle_contributions.csv", index=False)
    pd.DataFrame([{
        "minimum_cell_n": row["minimum_cell_n"],
        "eligible_cells": row["eligible_cells"],
        "eligible_support_rows": row["eligible_support_rows"],
        "coverage_of_tokenizable_rows": row["coverage_of_tokenizable_rows"],
        "common_within_cell_slope_per_0_1_ttr": row["common_within_cell_slope_per_0_1_ttr"],
        "positive_cells": row["cell_direction"]["positive_cells"],
        "zero_cells": row["cell_direction"]["zero_cells"],
        "negative_cells": row["cell_direction"]["negative_cells"],
        "max_cycle_sxx_share": row["cycle_concentration"]["max_cycle_sxx_share"],
        "top_3_cycle_sxx_share": row["cycle_concentration"]["top_3_cycle_sxx_share"],
    } for row in summaries]).to_csv(output_dir / "support_sensitivity.csv", index=False)

    result = {
        "source": {
            "train_file": "classification/Train.csv",
            "sha256": actual_hash,
            "rows": int(len(data)),
            "scope": "official classification Train only; cycles 47-79",
            "test_used": False,
            "raw_text_exported": False,
            "record_ids_exported": False,
            "granular_export_minimum_cell_n": 5,
            "model_fits": 0,
        },
        "measurement": {
            "token_pattern": "[A-Za-z0-9]+",
            "normalization": "lowercase before tokenization",
            "ttr": "number of unique tokens / token count",
            "missing_rule": "NaN or blank signal_stream is missing; no-token nonblank streams are present but not TTR-analyzable",
        },
        "support": {
            "all_train_rows": int(len(data)),
            "missing_or_blank_stream_rows": int(missing.sum()),
            "nonblank_stream_rows": int(len(present)),
            "tokenizable_rows": int(len(tokenizable)),
            "nonblank_but_no_token_rows": int(len(present) - len(tokenizable)),
            "all_exact_length_x_cycle_cells": int(len(cells)),
        },
        "method": {
            "formula": "within-cell beta = sum_cell sum((TTR-cell_mean_TTR)*(artifact-cell_mean_artifact)) / sum_cell sum((TTR-cell_mean_TTR)^2)",
            "cell": "exact token_count x exact cycle_code",
            "interpretation": "descriptive common linear within-cell association; beta*0.1 is the fitted artifact-probability difference per 0.1 TTR within this supported sample",
            "feature_only_support_selection": True,
            "constant_outcome_cells_retained": True,
            "assumptions_and_limits": [
                "A single linear within-cell slope is summarized across supported cells; real slopes may vary by length or cycle.",
                "Cells contribute in proportion to within-cell TTR variance (sum of squared deviations), not equal cell weight.",
                "This slope estimand differs from the original high-vs-low TTR quartile prevalence gap.",
                "The binary outcome makes the cell-centered slope descriptive and does not imply causation or semantic quality.",
                "Support gates can exclude sparse or TTR-constant cells; the result describes retained support only.",
            ],
        },
        "support_gate_results": summary_by_gate,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(_json_safe(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _plot_combined(cycles, summaries, output_dir)
    return result


def _plot_combined(cycles: pd.DataFrame, summaries: list[dict], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), gridspec_kw={"width_ratios": [1.65, 1]})
    ax = axes[0]
    colors = {2: "#8c8c8c", 5: "#225ea8", 10: "#d95f0e"}
    for gate in SUPPORT_GATES:
        g = cycles.loc[cycles["min_cell_n"] == gate]
        ax.plot(g["cycle_code"], g["slope_per_0_1_ttr"] * 100, marker="o", ms=3,
                lw=1.4 if gate == 5 else 1.0, color=colors[gate],
                alpha=1.0 if gate == 5 else 0.42,
                label=f"min cell n={gate}" if gate != 5 else "Primary: min cell n=5")
    primary_cycles = cycles.loc[cycles["min_cell_n"] == 5]
    negative = primary_cycles.loc[primary_cycles["slope_per_0_1_ttr"] < 0]
    ax.scatter(negative["cycle_code"], negative["slope_per_0_1_ttr"] * 100,
               marker="x", s=55, linewidths=1.6, color="#b2182b",
               label="Negative primary cycle slope", zorder=5)
    for x in (57.5, 68.5):
        ax.axvline(x, color="#bdbdbd", lw=0.8, ls="--")
    ax.axhline(0, color="#333333", lw=0.9)
    ax.set(xlabel="Exact cycle_code", ylabel="Artifact probability-point change per +0.1 TTR",
           title="A. Exact-length × exact-cycle slope by cycle")
    ax.legend(frameon=False, ncol=2, loc="upper left", fontsize=8)

    frame = pd.DataFrame([{
        "gate": row["minimum_cell_n"],
        "coverage": row["coverage_of_tokenizable_rows"],
        "slope": row["common_within_cell_slope_per_0_1_ttr"],
        "cells": row["eligible_cells"],
    } for row in summaries])
    ax2 = axes[1]
    ax2_right = ax2.twinx()
    beta_line, = ax2.plot(frame["gate"], frame["slope"] * 100, marker="o",
                          color="#238b45", lw=1.6, label="Within-cell slope")
    coverage_line, = ax2_right.plot(frame["gate"], frame["coverage"] * 100,
                                    marker="s", ls="--", color="#225ea8", lw=1.4,
                                    label="Tokenizable-row coverage")
    ax2.axhline(0, color="#333333", lw=0.9)
    ax2.set(xlabel="Minimum cell n", ylabel="Slope (probability points / +0.1 TTR)",
            title="B. Support-gate sensitivity")
    ax2.set_ylim(0, 10)
    ax2_right.set_ylabel("Tokenizable-row coverage (%)")
    ax2_right.set_ylim(75, 100)
    ax2.set_xlim(1, 11)
    ax2.set_xticks(frame["gate"])
    for _, row in frame.iterrows():
        ax2.annotate(f"{row.slope * 100:.2f}pp; {int(row.cells):,} cells",
                     (row.gate, row.slope * 100), xytext=(0, 8),
                     textcoords="offset points", ha="center", fontsize=7)
        ax2_right.annotate(f"{row.coverage * 100:.1f}%",
                           (row.gate, row.coverage * 100), xytext=(0, -16),
                           textcoords="offset points", ha="center", fontsize=7,
                           color="#225ea8")
    ax2.legend([beta_line, coverage_line], [beta_line.get_label(), coverage_line.get_label()],
               frameon=False, loc="lower left", fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "exact_cycle_association.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT,
                        help="repository root (default: script's repository)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="aggregate output directory")
    args = parser.parse_args()
    result = run(args.root, args.output_dir)
    for gate in SUPPORT_GATES:
        summary = result["support_gate_results"][str(gate)]
        print(json.dumps({
            "minimum_cell_n": gate,
            "eligible_cells": summary["eligible_cells"],
            "eligible_support_rows": summary["eligible_support_rows"],
            "coverage_of_tokenizable_rows": summary["coverage_of_tokenizable_rows"],
            "slope_per_0_1_ttr": summary["common_within_cell_slope_per_0_1_ttr"],
            "cell_directions": summary["cell_direction"],
            "cycle_concentration": summary["cycle_concentration"],
        }, sort_keys=True))


if __name__ == "__main__":
    main()
