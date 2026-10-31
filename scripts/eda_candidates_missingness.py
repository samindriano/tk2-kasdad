"""Independent, noncausal exploration of report-field availability in TP2.

This script reads only official Train files. It writes aggregate statistics and
figures without saving record-level inputs, raw signal_stream text, or models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPOSITORY = Path(__file__).resolve().parents[1]
import sys
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))
from src.utils.paths import task_files
from src.models.classification_round2_models import stream_missing_mask

FIELDS = (
    "org_code", "module_code", "pattern_code", "signal_stream",
    "scope_trace", "org_spread", "module_spread",
)
TARGETS = {"regression": "threat_rank", "classification": "artifact_signal"}
EXPECTED_TRAIN_HASHES = {
    "regression": "756523ff8156a9cd776e52249f7e2a8f23dcec432b8d10570e55ff6db0e0f082",
    "classification": "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c",
}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _phi(x: pd.Series, y: pd.Series) -> float:
    table = pd.crosstab(x.astype(bool), y.astype(bool)).reindex(
        index=[False, True], columns=[False, True], fill_value=0
    ).to_numpy(float)
    n = table.sum()
    denom = np.sqrt(table.sum(axis=0).prod() * table.sum(axis=1).prod())
    return float((table[0, 0] * table[1, 1] - table[0, 1] * table[1, 0]) / denom) if denom else 0.0


def run(root: Path | None = None, output_dir: Path | None = None) -> dict:
    root = Path(root or REPOSITORY).resolve()
    output_dir = Path(output_dir or root / "reports/eda_candidates/missingness")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    # Inputs follow the repository's shared TP2_DATA_DIR resolver; `root` controls
    # only the report destination so callers can keep raw data in its frozen home.
    paths = {task: task_files(task)['train'] for task in TARGETS}
    if any(path is None or not path.is_file() for path in paths.values()):
        raise FileNotFoundError("Official classification and regression Train CSVs are required.")
    input_hashes = {task: _sha(path) for task, path in paths.items()}
    if input_hashes != EXPECTED_TRAIN_HASHES:
        raise ValueError(f"Official Train hash mismatch: {input_hashes}")
    frames = {task: pd.read_csv(path) for task, path in paths.items()}
    reg, cls = frames["regression"], frames["classification"]
    for task, frame in frames.items():
        target = TARGETS[task]
        required = {"record_id", "cycle_code", target, *FIELDS}
        missing_cols = sorted(required - set(frame.columns))
        if missing_cols:
            raise ValueError(f"{task} train lacks {missing_cols}")
        if len(frame) != 50_608 or frame.record_id.isna().any() or not frame.record_id.is_unique:
            raise ValueError(f"Unexpected {task} Train grain/size/ID quality")
        if frame[target].isna().any():
            raise ValueError(f"Unexpected missing {target}")
    if set(reg.record_id) != set(cls.record_id):
        raise ValueError("Classification and regression Train IDs do not match")
    # Compare every shared non-target feature after aligning on record_id. The
    # labels differ by task; all other shared columns must describe identical rows.
    shared_features = sorted((set(reg.columns) & set(cls.columns)) - {
        "record_id", TARGETS["regression"], TARGETS["classification"]
    })
    reg_features = reg.set_index("record_id")[shared_features].sort_index()
    cls_features = cls.set_index("record_id")[shared_features].sort_index()
    shared = reg_features.compare(cls_features)
    if len(shared):
        raise ValueError("Shared Train features differ after matching record_id")

    # Target-blind count: number of unavailable fields among seven broad report
    # context fields. telemetry_flag is deliberately omitted because Rizky owns it.
    miss = reg.set_index("record_id")[list(FIELDS)].isna().astype("int8")
    miss["signal_stream"] = stream_missing_mask(reg["signal_stream"]).astype("int8")
    miss["missing_count"] = miss.sum(axis=1)
    miss["mask"] = miss[list(FIELDS)].astype(str).agg("".join, axis=1)
    count_rows = []
    for k, ids in miss.groupby("missing_count").groups.items():
        m = miss.loc[ids]
        r = reg.set_index("record_id").loc[ids]
        c = cls.set_index("record_id").loc[ids]
        count_rows.append({
            "missing_count": int(k), "n": int(len(ids)), "share": float(len(ids) / len(reg)),
            "threat_mean": float(r.threat_rank.mean()), "threat_median": float(r.threat_rank.median()),
            "threat_p10": float(r.threat_rank.quantile(.1)), "threat_p90": float(r.threat_rank.quantile(.9)),
            "threat_low_le5_share": float(r.threat_rank.le(5).mean()),
            "threat_high_ge95_share": float(r.threat_rank.ge(95).mean()),
            "artifact_positive_share": float(c.artifact_signal.mean()),
            "cycles": int(r.cycle_code.nunique()),
        })
    by_count = pd.DataFrame(count_rows).sort_values("missing_count")
    by_count.to_csv(output_dir / "outcomes_by_missing_count.csv", index=False, float_format="%.8g")

    missing_any = miss.missing_count.gt(0).rename("missing_any")
    within_cycle = []
    for cycle in sorted(reg.cycle_code.unique()):
        ids = reg.loc[reg.cycle_code.eq(cycle), "record_id"]
        mm = missing_any.loc[ids]
        rr = reg.set_index("record_id").loc[ids]
        cc = cls.set_index("record_id").loc[ids]
        n_missing, n_complete = int(mm.sum()), int((~mm).sum())
        within_cycle.append({
            "cycle": int(cycle), "n": len(ids), "missing_n": n_missing, "complete_n": n_complete,
            "missing_share": float(mm.mean()),
            "threat_missing_minus_complete": float(rr.loc[mm.to_numpy(), "threat_rank"].mean()
                                                    - rr.loc[(~mm).to_numpy(), "threat_rank"].mean())
                if n_missing and n_complete else None,
            "artifact_missing_minus_complete": float(cc.loc[mm.to_numpy(), "artifact_signal"].mean()
                                                       - cc.loc[(~mm).to_numpy(), "artifact_signal"].mean())
                if n_missing and n_complete else None,
        })
    cycles = pd.DataFrame(within_cycle)
    cycles.to_csv(output_dir / "within_cycle_contrasts.csv", index=False, float_format="%.8g")

    # Label-blind availability-mask census, retaining only common patterns to avoid
    # building a story around rare combinations. No text/token values are exported.
    mask_counts = miss["mask"].value_counts().rename_axis("missing_mask").rename("n").reset_index()
    mask_counts["missing_count"] = mask_counts.missing_mask.str.count("1")
    common_masks = mask_counts.loc[mask_counts.n.ge(100)].copy()
    common_masks.to_csv(output_dir / "common_missing_patterns.csv", index=False)
    phi = pd.DataFrame(index=FIELDS, columns=FIELDS, dtype=float)
    for x in FIELDS:
        for y in FIELDS:
            phi.loc[x, y] = _phi(miss[x], miss[y])
    phi.to_csv(output_dir / "missingness_phi.csv", float_format="%.8g")

    # Separate every field's simple observed association; this is context, not a
    # claim of distinct causal mechanisms or a search for the most extreme field.
    field_rows = []
    for field in FIELDS:
        m = miss[field].astype(bool)
        ids = miss.index
        r = reg.set_index("record_id").loc[ids]
        c = cls.set_index("record_id").loc[ids]
        field_rows.append({
            "field": field, "missing_n": int(m.sum()), "missing_share": float(m.mean()),
            "threat_mean_missing_minus_present": float(r.loc[m, "threat_rank"].mean() - r.loc[~m, "threat_rank"].mean()),
            "threat_median_missing_minus_present": float(r.loc[m, "threat_rank"].median() - r.loc[~m, "threat_rank"].median()),
            "artifact_rate_missing_minus_present": float(c.loc[m, "artifact_signal"].mean() - c.loc[~m, "artifact_signal"].mean()),
            "phi_with_other_missingness": float(np.mean([phi.loc[field, z] for z in FIELDS if z != field])),
        })
    per_field = pd.DataFrame(field_rows).sort_values("missing_share", ascending=False)
    per_field.to_csv(output_dir / "per_field_outcomes.csv", index=False, float_format="%.8g")

    # A separate, model-independent check of text availability. Stratify only by
    # the number of other unavailable report fields (not by the assigned members'
    # substantive features); retain strata fixed without target-based selection.
    stream_rows = []
    stream_cycle_rows = []
    other_count = reg.set_index("record_id")[list(FIELDS)].isna().drop(columns="signal_stream").sum(axis=1)
    other_count.loc[reg.record_id] = reg[[f for f in FIELDS if f != "signal_stream"]].isna().sum(axis=1).to_numpy()
    stream_missing = pd.Series(stream_missing_mask(reg["signal_stream"]), index=reg.record_id, name="stream_missing")
    reg_by_id = reg.set_index("record_id")
    cls_by_id = cls.set_index("record_id")
    combined = pd.DataFrame({
        "stream_missing": stream_missing,
        "other_missing_count": other_count,
        "cycle_code": reg_by_id.cycle_code,
        "threat_rank": reg_by_id.threat_rank,
        "artifact_signal": cls_by_id.artifact_signal,
    })
    for count, group in combined.groupby("other_missing_count"):
        absent = group[group.stream_missing]
        present = group[~group.stream_missing]
        if len(absent) < 100 or len(present) < 100:
            continue
        stream_rows.append({
            "other_missing_count": int(count), "stream_missing_n": int(len(absent)),
            "stream_present_n": int(len(present)),
            "threat_mean_delta_missing_minus_present": float(absent.threat_rank.mean() - present.threat_rank.mean()),
            "threat_median_delta_missing_minus_present": float(absent.threat_rank.median() - present.threat_rank.median()),
            "threat_p10_missing": float(absent.threat_rank.quantile(.1)),
            "threat_p90_missing": float(absent.threat_rank.quantile(.9)),
            "artifact_rate_delta_missing_minus_present": float(absent.artifact_signal.mean() - present.artifact_signal.mean()),
        })
    stream_by_other = pd.DataFrame(stream_rows).sort_values("other_missing_count")
    stream_by_other.to_csv(output_dir / "signal_stream_by_other_missing_count.csv", index=False, float_format="%.8g")

    # Volume deciles are computed from the feature alone, before target summaries.
    # They diagnose whether stream absence simply tags shorter reports.
    combined["text_load"] = reg_by_id.text_load
    combined["pattern_code"] = reg_by_id.pattern_code
    combined["text_load_decile"] = pd.qcut(combined.text_load, 10, labels=False, duplicates="drop")
    volume_rows = []
    for decile, group in combined.groupby("text_load_decile", observed=True):
        absent, present = group[group.stream_missing], group[~group.stream_missing]
        volume_rows.append({
            "text_load_decile": int(decile), "lower_edge": float(group.text_load.min()),
            "upper_edge": float(group.text_load.max()), "total_n": int(len(group)),
            "stream_missing_n": int(len(absent)), "stream_present_n": int(len(present)),
            "missing_share": float(len(absent)/len(group)),
            "threat_mean_delta_missing_minus_present": float(absent.threat_rank.mean()-present.threat_rank.mean()),
            "threat_median_delta_missing_minus_present": float(absent.threat_rank.median()-present.threat_rank.median()),
            "artifact_rate_delta_missing_minus_present": float(absent.artifact_signal.mean()-present.artifact_signal.mean()),
        })
    volume_table = pd.DataFrame(volume_rows).sort_values("text_load_decile")
    volume_table.to_csv(output_dir / "signal_stream_by_text_load_decile.csv", index=False, float_format="%.8g")
    volume_common = combined[combined.text_load_decile.isin(volume_table.text_load_decile)]
    volume_weights = volume_common.text_load_decile.value_counts(normalize=True).sort_index()
    standardized_by_text_load = {
        "n": int(len(volume_common)), "strata": len(volume_weights),
        "threat_mean_delta": float(sum(
            float(volume_weights.loc[int(row.text_load_decile)])*row.threat_mean_delta_missing_minus_present
            for _, row in volume_table.iterrows())),
        "artifact_rate_delta": float(sum(
            float(volume_weights.loc[int(row.text_load_decile)])*row.artifact_rate_delta_missing_minus_present
            for _, row in volume_table.iterrows())),
    }
    present_min = float(combined.loc[~combined.stream_missing, "text_load"].min())
    missing_max = float(combined.loc[combined.stream_missing, "text_load"].max())
    common_volume = combined[combined.text_load.between(present_min, missing_max)].copy()
    common_range_stats = {
        "observed_min_missing": float(combined.loc[combined.stream_missing, "text_load"].min()),
        "observed_min_present": present_min,
        "observed_max_missing": missing_max,
        "observed_max_present": float(combined.loc[~combined.stream_missing, "text_load"].max()),
        "common_observed_range": [present_min, missing_max],
        "rows_retained": int(len(common_volume)),
        "missing_rows_retained": int(common_volume.stream_missing.sum()),
        "missing_rows_excluded": int(combined.stream_missing.sum()-common_volume.stream_missing.sum()),
        "present_rows_excluded": int((~combined.stream_missing).sum()-(~common_volume.stream_missing).sum()),
        "mean_threat_delta_retained": float(
            common_volume.loc[common_volume.stream_missing, "threat_rank"].mean()
            - common_volume.loc[~common_volume.stream_missing, "threat_rank"].mean()),
    }
    volume_binning_sensitivity = []
    for q in (5, 10, 20):
        binned = combined.copy()
        binned["volume_bin"] = pd.qcut(binned.text_load, q=q, labels=False, duplicates="drop")
        rows = []
        for bin_id, group in binned.groupby("volume_bin", observed=True):
            absent, present = group[group.stream_missing], group[~group.stream_missing]
            rows.append({
                "bin": int(bin_id), "total_n": int(len(group)), "missing_n": int(len(absent)),
                "present_n": int(len(present)),
                "threat_mean_delta": float(absent.threat_rank.mean()-present.threat_rank.mean()),
                "artifact_rate_delta": float(absent.artifact_signal.mean()-present.artifact_signal.mean()),
            })
        bins = pd.DataFrame(rows)
        supported = bins[bins.missing_n.ge(20) & bins.present_n.ge(20)]
        weights = supported.total_n.to_numpy(float)
        weights /= weights.sum()
        missing_ns = supported.missing_n.to_numpy(float)
        present_ns = supported.present_n.to_numpy(float)
        volume_binning_sensitivity.append({
            "quantile_bins_requested": q, "supported_bins": int(len(supported)),
            "rows_covered": int(supported.total_n.sum()),
            "coverage": float(supported.total_n.sum()/len(combined)),
            "minimum_n_each_availability_state": 20,
            "standardized_threat_mean_delta": float(np.average(supported.threat_mean_delta, weights=weights)),
            "standardized_artifact_rate_delta": float(np.average(supported.artifact_rate_delta, weights=weights)),
            "effective_n_stream_missing": float(1/np.sum(weights**2/missing_ns)),
            "effective_n_stream_present": float(1/np.sum(weights**2/present_ns)),
        })
    pd.DataFrame(volume_binning_sensitivity).to_csv(
        output_dir/"signal_stream_text_load_binning_sensitivity.csv", index=False, float_format="%.8g")

    # Category composition check: report only support and aggregate contrast, not
    # the opaque category labels. Minimum support is fixed without target values.
    pattern_rows = []
    patternkey = combined.pattern_code.astype("string").fillna("__MISSING__")
    combined["pattern_key"] = patternkey
    pattern_counts = combined.groupby(["pattern_key", "stream_missing"], observed=True).size().unstack("stream_missing", fill_value=0)
    pattern_support = {}
    for threshold in (20, 30):
        support = pattern_counts[(pattern_counts.get(True, 0) >= threshold) & (pattern_counts.get(False, 0) >= threshold)].index
        subset = combined[combined.pattern_key.isin(support)]
        diffs, weights = [], []
        for _, group in subset.groupby("pattern_key", observed=True):
            absent, present = group[group.stream_missing], group[~group.stream_missing]
            if len(absent) < threshold or len(present) < threshold:
                raise AssertionError("Pattern support selection must use predictor counts only")
            diffs.append(float(absent.threat_rank.mean()-present.threat_rank.mean()))
            weights.append(len(group))
        pattern_support[str(threshold)] = {
            "minimum_rows_per_pattern_and_stream_state": threshold,
            "common_patterns_including_missing": int(len(support)),
            "covered_rows": int(len(subset)), "covered_share": float(len(subset)/len(combined)),
            "stream_missing_rows_covered": int(subset.stream_missing.sum()),
            "stream_missing_coverage": float(subset.stream_missing.sum()/combined.stream_missing.sum()),
            "stream_present_coverage": float((~subset.stream_missing).sum()/(~combined.stream_missing).sum()),
            "row_weighted_mean_threat_delta": float(np.average(diffs, weights=weights)),
            "median_pattern_mean_delta": float(np.median(diffs)),
            "pattern_mean_delta_range": [float(np.min(diffs)), float(np.max(diffs))],
        }
    # Combining category and volume produces low support; publish support only.
    joint = combined.groupby(["text_load_decile", "pattern_key", "stream_missing"], observed=True).size().unstack("stream_missing", fill_value=0)
    for threshold in (20, 30):
        common_cells = joint[(joint.get(True, 0) >= threshold) & (joint.get(False, 0) >= threshold)]
        pattern_support[str(threshold)]["text_load_x_pattern_cells"] = int(len(common_cells))
        pattern_support[str(threshold)]["text_load_x_pattern_rows_covered"] = int(common_cells.sum(axis=1).sum())
        pattern_support[str(threshold)]["text_load_x_pattern_share_covered"] = float(common_cells.sum(axis=1).sum()/len(combined))
    pd.DataFrame([{"min_n_each_state": int(k), **v} for k, v in pattern_support.items()]).to_csv(
        output_dir / "pattern_code_common_support.csv", index=False, float_format="%.8g")
    for cycle, group in combined.groupby("cycle_code"):
        absent, present = group[group.stream_missing], group[~group.stream_missing]
        if len(absent) >= 20 and len(present) >= 20:
            stream_cycle_rows.append({
                "cycle": int(cycle), "stream_missing_n": len(absent), "stream_present_n": len(present),
                "threat_mean_delta": float(absent.threat_rank.mean() - present.threat_rank.mean()),
                "artifact_rate_delta": float(absent.artifact_signal.mean() - present.artifact_signal.mean()),
            })
    stream_cycles = pd.DataFrame(stream_cycle_rows)
    stream_cycles.to_csv(output_dir / "signal_stream_within_cycle.csv", index=False, float_format="%.8g")

    # Cycle-level contrast robustness. Percent of cycles in same direction and
    # leave-one-cycle-out range; descriptive checks, not inferential p-values.
    complete_row = by_count.loc[by_count.missing_count.eq(0)].iloc[0]
    one_to_three = by_count.loc[by_count.missing_count.between(1, 3)]
    overall = {
        "n": int(len(reg)), "complete_n": int(complete_row.n),
        "any_missing_n": int(len(reg) - complete_row.n),
        "complete_share": float(complete_row.share),
        "threat_mean_any_missing_minus_complete": float(
            by_count.loc[by_count.missing_count.gt(0)].eval("threat_mean*n").sum()
            / by_count.loc[by_count.missing_count.gt(0), "n"].sum() - complete_row.threat_mean),
        "threat_mean_1_to_3_missing": float(np.average(one_to_three.threat_mean, weights=one_to_three.n)),
        "threat_median_count0": float(complete_row.threat_median),
        "threat_median_count3": float(by_count.loc[by_count.missing_count.eq(3), "threat_median"].iloc[0]),
        "artifact_rate_count0": float(complete_row.artifact_positive_share),
        "artifact_rate_any_missing": float(np.average(
            by_count.loc[by_count.missing_count.gt(0), "artifact_positive_share"],
            weights=by_count.loc[by_count.missing_count.gt(0), "n"])),
        "artifact_rate_count0_vs_count1": [float(by_count.loc[by_count.missing_count.eq(k), "artifact_positive_share"].iloc[0]) for k in (0, 1)],
        "median_cycle_threat_delta": float(cycles.threat_missing_minus_complete.median()),
        "range_cycle_threat_delta": [float(cycles.threat_missing_minus_complete.min()),
                                     float(cycles.threat_missing_minus_complete.max())],
        "n_cycles_threat_delta_positive": int(cycles.threat_missing_minus_complete.gt(0).sum()),
        "n_cycles_threat_delta_negative": int(cycles.threat_missing_minus_complete.lt(0).sum()),
        "median_cycle_artifact_delta": float(cycles.artifact_missing_minus_complete.median()),
        "range_cycle_artifact_delta": [float(cycles.artifact_missing_minus_complete.min()),
                                       float(cycles.artifact_missing_minus_complete.max())],
        "n_cycles_artifact_delta_positive": int(cycles.artifact_missing_minus_complete.gt(0).sum()),
        "n_cycles_artifact_delta_negative": int(cycles.artifact_missing_minus_complete.lt(0).sum()),
        "top_missing_pattern": str(mask_counts.iloc[1].missing_mask),
        "top_missing_pattern_n": int(mask_counts.iloc[1].n),
        "phi_pattern_code_signal_stream": float(phi.loc["pattern_code", "signal_stream"]),
        "phi_scope_trace_org_spread": float(phi.loc["scope_trace", "org_spread"]),
    }
    all_stream_missing = combined.loc[combined.stream_missing]
    all_stream_present = combined.loc[~combined.stream_missing]
    # Common strata receive pooled feature-only weights. The standardized
    # contrasts are arithmetic summaries, not estimates of controlled effects.
    usable_counts = set(stream_by_other.other_missing_count)
    common = combined[combined.other_missing_count.isin(usable_counts)]
    strata_weights = common.other_missing_count.value_counts(normalize=True).sort_index()
    standardized_threat = standardized_artifact = 0.0
    for _, row in stream_by_other.iterrows():
        w = float(strata_weights.loc[int(row.other_missing_count)])
        standardized_threat += w * float(row.threat_mean_delta_missing_minus_present)
        standardized_artifact += w * float(row.artifact_rate_delta_missing_minus_present)
    stream_summary = {
        "missing_n": int(len(all_stream_missing)), "missing_share": float(len(all_stream_missing)/len(combined)),
        "threat_mean_missing": float(all_stream_missing.threat_rank.mean()),
        "threat_mean_present": float(all_stream_present.threat_rank.mean()),
        "threat_mean_delta": float(all_stream_missing.threat_rank.mean()-all_stream_present.threat_rank.mean()),
        "threat_median_missing": float(all_stream_missing.threat_rank.median()),
        "threat_median_present": float(all_stream_present.threat_rank.median()),
        "threat_low_le5_share_missing": float(all_stream_missing.threat_rank.le(5).mean()),
        "threat_low_le5_share_present": float(all_stream_present.threat_rank.le(5).mean()),
        "threat_high_ge95_share_missing": float(all_stream_missing.threat_rank.ge(95).mean()),
        "threat_high_ge95_share_present": float(all_stream_present.threat_rank.ge(95).mean()),
        "artifact_rate_missing": float(all_stream_missing.artifact_signal.mean()),
        "artifact_rate_present": float(all_stream_present.artifact_signal.mean()),
        "artifact_rate_delta": float(all_stream_missing.artifact_signal.mean()-all_stream_present.artifact_signal.mean()),
        "cycle_threat_delta_positive": int(stream_cycles.threat_mean_delta.gt(0).sum()),
        "cycle_threat_delta_negative": int(stream_cycles.threat_mean_delta.lt(0).sum()),
        "cycle_threat_delta_median": float(stream_cycles.threat_mean_delta.median()),
        "cycle_threat_delta_range": [float(stream_cycles.threat_mean_delta.min()), float(stream_cycles.threat_mean_delta.max())],
        "leave_one_cycle_out_cycle_mean_range": [
            float((stream_cycles.threat_mean_delta.sum()-stream_cycles.threat_mean_delta.max())/(len(stream_cycles)-1)),
            float((stream_cycles.threat_mean_delta.sum()-stream_cycles.threat_mean_delta.min())/(len(stream_cycles)-1)),
        ],
        "cycle_artifact_delta_positive": int(stream_cycles.artifact_rate_delta.gt(0).sum()),
        "cycle_artifact_delta_negative": int(stream_cycles.artifact_rate_delta.lt(0).sum()),
        "standardized_threat_delta_over_common_other_missing_counts": float(standardized_threat),
        "standardized_artifact_delta_over_common_other_missing_counts": float(standardized_artifact),
        "standardization_common_n": int(len(common)),
        "standardization_strata": sorted(int(x) for x in usable_counts),
        "standardized_by_text_load_deciles": standardized_by_text_load,
        "common_text_load_support": common_range_stats,
        "text_load_binning_sensitivity": volume_binning_sensitivity,
        "text_load_q1_share_of_missing_signal_stream": float(volume_table.iloc[0].stream_missing_n/len(all_stream_missing)),
        "text_load_q1_missing_share": float(volume_table.iloc[0].missing_share),
        "text_load_decile_effect_positive_n": int(volume_table.threat_mean_delta_missing_minus_present.gt(0).sum()),
        "text_load_decile_effect_negative_n": int(volume_table.threat_mean_delta_missing_minus_present.lt(0).sum()),
        "pattern_code_common_support": pattern_support,
    }
    metadata = {
        "status": "COMPLETE_EXPLORATORY_MISSINGNESS_SCREEN",
        "input_hashes": input_hashes,
        "shapes": {task: list(frame.shape) for task, frame in frames.items()},
        "shared_ids_and_features_aligned": True,
        "shared_feature_columns_checked": shared_features,
        "fields": list(FIELDS), "telemetry_flag_included": False,
        "group_definition": "Target-blind count of missing among seven report-context fields; telemetry_flag excluded because Rizky owns it.",
        "targets": TARGETS,
        "missingness_pattern_minimum_support_for_export": 100,
        "overall": overall,
        "signal_stream_availability": stream_summary,
        "interpretation_limits": [
            "Associations are descriptive, not causal; unavailable fields may reflect record-generation or collection processes.",
            "All 33 cycles are development data previously examined; cycle contrasts are sensitivity checks, not independent confirmation.",
            "Small count-4/count-5 groups are retained in the machine table but are not interpreted as a trend.",
            "Classification and regression Train hashes match the pinned official inputs; all shared non-target features were checked after record_id alignment; no Test labels were read.",
            "Seven-field completeness omits telemetry_flag to avoid centering Rizky's assigned topic; conclusions therefore apply only to these seven fields.",
            "signal_stream availability follows the established NaN-or-blank-after-trim rule; other field missingness counts only null values.",
        ],
    }
    (output_dir / "statistics.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), gridspec_kw={"height_ratios": [1, 1.2, 1]})
    ax = axes[0]
    x = by_count.missing_count.to_numpy()
    n = by_count.n.to_numpy()
    ax.bar(x, n, color="#2f6d85")
    ax.set(title="Jumlah field konteks yang kosong", xlabel="Field kosong (dari 7)", ylabel="Jumlah baris")
    ax.set_ylim(0, max(n) * 1.12)
    for xx, nn in zip(x, n): ax.text(xx, nn + max(n)*.012, f"{nn:,}", ha="center", fontsize=8)
    # Stable only through count three; sparse higher groups remain visible, hatched.
    for patch, count in zip(ax.patches, x):
        if count >= 4: patch.set_hatch("//"); patch.set_alpha(.55)
    left = by_count.loc[by_count.missing_count.le(3)]
    group_labels = [f"{int(row.missing_count)}\nn={int(row.n):,}" for _, row in left.iterrows()]
    ax = axes[1]
    ax.plot(left.missing_count, left.threat_median, "-o", color="#a64c36", label="Median threat_rank")
    ax.fill_between(left.missing_count, left.threat_p10, left.threat_p90, alpha=.16, color="#a64c36", label="P10–P90")
    ax.set(title="Threat rank menurut jumlah field kosong", ylabel="threat_rank (0–100)", ylim=(0,100))
    ax.set_xticks(left.missing_count, group_labels)
    ax.legend(loc="upper right", fontsize=8)
    ax = axes[2]
    ax.plot(left.missing_count, left.artifact_positive_share * 100, "--s", color="#315b97")
    ax.set(title="Proporsi artifact_signal = 1", xlabel="Field kosong (dari 7; n per grup ditampilkan di label)",
           ylabel="Proporsi (%)", ylim=(0,100))
    ax.set_xticks(left.missing_count, group_labels)
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(figure_dir / "availability_count_and_outcomes.png", dpi=160, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    image = ax.imshow(phi.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FIELDS)), FIELDS, rotation=40, ha="right")
    ax.set_yticks(range(len(FIELDS)), FIELDS)
    ax.set_title("Asosiasi antarindikator missingness (phi)")
    for i in range(len(FIELDS)):
        for j in range(len(FIELDS)):
            ax.text(j, i, f"{phi.iloc[i,j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(phi.iloc[i,j]) > .58 else "black")
    fig.colorbar(image, ax=ax, label="Koefisien phi")
    fig.tight_layout(); fig.savefig(figure_dir / "co_missing_phi.png", dpi=160, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    for mask, label, color in [(False, f"Tersedia (n={len(all_stream_present):,})", "#315b97"),
                               (True, f"Kosong (n={len(all_stream_missing):,})", "#a64c36")]:
        values = np.sort(combined.loc[combined.stream_missing.eq(mask), "threat_rank"].to_numpy())
        ecdf = np.arange(1, len(values)+1)/len(values)
        ax.step(values, ecdf, where="post", label=label, color=color)
    ax.set(title="Distribusi kumulatif threat_rank menurut availability text",
           xlabel="threat_rank (0–100)", ylabel="Proporsi baris dengan nilai ≤ x", xlim=(0,100), ylim=(0,1))
    ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
    fig.savefig(figure_dir / "signal_stream_threat_ecdf.png", dpi=160, bbox_inches="tight"); plt.close(fig)

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 7.2), sharex=True)
    x = stream_by_other.other_missing_count.to_numpy()
    ax.plot(x, stream_by_other.threat_median_delta_missing_minus_present, "-o", color="#a64c36", label="Selisih median threat_rank")
    ax.axhline(0, color="black", lw=1)
    ax.set(title="A. Perbedaan median threat_rank",
           ylabel="Kosong − tersedia (skor 0–100)", ylim=(0, 27))
    ax.grid(alpha=.2)
    class_deltas = 100 * stream_by_other.artifact_rate_delta_missing_minus_present
    bars = ax2.bar(x, class_deltas, width=.5, color=["#315b97" if v >= 0 else "#b18b33" for v in class_deltas])
    ax2.axhline(0, color="black", lw=1)
    ax2.set(title="B. Perbedaan proporsi artifact_signal = 1",
            xlabel="Field konteks lain yang kosong (dari 6; strata label-blind)",
            ylabel="Kosong − tersedia (poin persentase)")
    ax2.set_xticks(x, [f"{int(row.other_missing_count)} field lain\nKosong n={int(row.stream_missing_n):,}\nTersedia n={int(row.stream_present_n):,}"
                       for _, row in stream_by_other.iterrows()], fontsize=8)
    ax2.set_ylim(bottom=min(-11, float(class_deltas.min() - 3)),
                 top=max(15, float(class_deltas.max() + 3)))
    ax2.grid(axis="y", alpha=.2)
    for _, row in stream_by_other.iterrows():
        ax.annotate(f"+{row.threat_median_delta_missing_minus_present:.1f}",
                    (int(row.other_missing_count), row.threat_median_delta_missing_minus_present),
                    xytext=(0, 7), textcoords="offset points", ha="center", fontsize=9)
    for patch, value in zip(bars, class_deltas):
        ax2.text(patch.get_x()+patch.get_width()/2, value+(0.5 if value >= 0 else -1.4),
                 f"{value:+.1f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=9)
    fig.suptitle("Asosiasi distribusi saat signal_stream kosong", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(figure_dir / "signal_stream_target_contrasts.png", dpi=160, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11,4.2), sharex=True)
    axes[0].bar(volume_table.text_load_decile, volume_table.missing_share*100, color="#648c78")
    axes[0].set(title="Konsentrasi stream kosong pada text_load rendah", xlabel="Decile text_load (fitur saja)", ylabel="Stream kosong (%)")
    axes[0].set_xticks(volume_table.text_load_decile)
    axes[1].plot(volume_table.text_load_decile, volume_table.threat_mean_delta_missing_minus_present,
                 "-o", color="#a64c36", label="Threat mean")
    axes[1].axhline(0, color="black", lw=1)
    axes[1].set(title="Kontras setelah stratifikasi text_load", xlabel="Decile text_load (fitur saja)",
                ylabel="Selisih rata-rata threat (kosong − tersedia)")
    axes[1].set_xticks(volume_table.text_load_decile); axes[1].grid(axis="y", alpha=.2)
    for _, row in volume_table.iterrows():
        axes[1].annotate(f"{row.threat_mean_delta_missing_minus_present:+.1f}\nn={int(row.stream_missing_n)}",
                         (row.text_load_decile, row.threat_mean_delta_missing_minus_present),
                         xytext=(0, 6 if row.threat_mean_delta_missing_minus_present>=0 else -20),
                         textcoords="offset points", ha="center", fontsize=8)
    axes[1].set_ylim(-12, 15)
    fig.tight_layout(); fig.savefig(figure_dir / "text_load_confounding_check.png", dpi=160, bbox_inches="tight"); plt.close(fig)

    text = f'''# Discovery awal: ketersediaan signal_stream dan kelengkapan laporan

## Pertanyaan eksploratif

Apakah ketersediaan `signal_stream` berkaitan dengan distribusi `threat_rank` dan proporsi `artifact_signal`, setelah melihat perbedaan volume laporan dan ketersediaan field lain?

Screen awal dihitung dari jumlah field kosong pada `{len(FIELDS)}` kolom (`{', '.join(FIELDS)}`), dengan `telemetry_flag` dikeluarkan agar pertanyaan tidak berpusat pada topik Rizky. Lalu saya menguji khusus availability `signal_stream` menurut aturan pipeline NaN/blank setelah trim. Train Classification dan Regression diverifikasi memiliki baris/fitur yang sama setelah join `record_id`.

## Cakupan dan hasil utama

- {overall['complete_n']:,}/{overall['n']:,} baris ({overall['complete_share']:.1%}) lengkap pada ketujuh field; {overall['any_missing_n']:,} memiliki sekurangnya satu field kosong.
- Jumlah field kosong 0–5: {', '.join(f"{int(r.missing_count)}: n={int(r.n):,}" for _,r in by_count.iterrows())}. Kelompok 4–5 hanya berisi {int(by_count.loc[by_count.missing_count.ge(4),'n'].sum()):,} baris; tidak dipakai untuk tren.
- Median `threat_rank`: 0 kosong = {overall['threat_median_count0']:.3f}, 3 kosong = {overall['threat_median_count3']:.3f}; n untuk kelompok 3 = {int(by_count.loc[by_count.missing_count.eq(3),'n'].iloc[0]):,}. Nilai 1–3 kosong tidak menghasilkan gradien sederhana: lihat tabel/figur.
- Rata-rata `threat_rank` untuk 1–3 kosong = {overall['threat_mean_1_to_3_missing']:.3f}; kontras rata-rata missing-any dengan complete = {overall['threat_mean_any_missing_minus_complete']:+.3f}.
- Proporsi `artifact_signal=1`: 0 kosong = {overall['artifact_rate_count0']:.3%}; 1 kosong = {overall['artifact_rate_count0_vs_count1'][1]:.3%}; semua missing = {overall['artifact_rate_any_missing']:.3%}. Perubahan kecil ini tidak mendukung cerita kuat tentang outcome klasifikasi.
- Kontras threat missing-vs-complete per cycle: median {overall['median_cycle_threat_delta']:+.3f}, rentang {overall['range_cycle_threat_delta'][0]:+.3f} hingga {overall['range_cycle_threat_delta'][1]:+.3f}; arahnya positif {overall['n_cycles_threat_delta_positive']}/33 dan negatif {overall['n_cycles_threat_delta_negative']}/33 cycle. Kontras artifact per cycle juga melintasi nol (median {overall['median_cycle_artifact_delta']:+.3%}).
- Phi missing `pattern_code` dengan missing text = {overall['phi_pattern_code_signal_stream']:.3f}; missing `scope_trace` dengan `org_spread` = {overall['phi_scope_trace_org_spread']:.3f}. Nilai ini menunjukkan pola ketersediaan bersama tertentu, bukan penyebab bersama.

## Penyelidikan terpisah: `signal_stream` kosong

Mengikuti definisi pipeline yang telah ada, kosong berarti NaN **atau** teks blank setelah trim. Ini eksplorasi langsung dari label Train, tanpa prediksi model. Dari {stream_summary['missing_n']:,} ({stream_summary['missing_share']:.2%}) stream kosong, mean/median `threat_rank` = {stream_summary['threat_mean_missing']:.3f}/{stream_summary['threat_median_missing']:.3f}; stream tersedia = {stream_summary['threat_mean_present']:.3f}/{stream_summary['threat_median_present']:.3f}. Kontras mean +{stream_summary['threat_mean_delta']:.3f} tidak menunjukkan hubungan sebab-akibat.

Di antara {stream_summary['cycle_threat_delta_positive']+stream_summary['cycle_threat_delta_negative']}/{len(stream_cycles)} cycle dengan dukungan memadai pada kedua grup, mean threat saat teks kosong lebih tinggi; kontras positif pada {stream_summary['cycle_threat_delta_positive']} cycle, negatif pada {stream_summary['cycle_threat_delta_negative']}. Median kontras antar-cycle +{stream_summary['cycle_threat_delta_median']:.3f} (rentang {stream_summary['cycle_threat_delta_range'][0]:.3f} hingga {stream_summary['cycle_threat_delta_range'][1]:.3f}); leave-one-cycle-out mean delta masih {stream_summary['leave_one_cycle_out_cycle_mean_range'][0]:.3f} hingga {stream_summary['leave_one_cycle_out_cycle_mean_range'][1]:.3f}. Arah yang sama juga tampak setelah menstandarkan jumlah **field lain** yang kosong, pada strata label-blind {stream_summary['standardization_strata']}; weighted mean-difference {stream_summary['standardized_threat_delta_over_common_other_missing_counts']:+.3f}. Namun missing stream sangat terkonsentrasi pada volume rendah: {stream_summary['text_load_q1_share_of_missing_signal_stream']:.1%} dari 3.404 stream kosong berada di decile pertama `text_load` (n={int(volume_table.iloc[0].stream_missing_n):,}; sekitar separuh decile ini missing). Sembilan decile lain masing-masing hanya memuat 71–112 stream kosong. Standardisasi mean by decile bergantung pada jumlah bin: {', '.join(f"{d['quantile_bins_requested']} bins {d['standardized_threat_mean_delta']:+.3f}" for d in stream_summary['text_load_binning_sensitivity'])}; effective sample size missing setelah penimbangan decile berkisar {min(d['effective_n_stream_missing'] for d in stream_summary['text_load_binning_sensitivity']):,.0f}–{max(d['effective_n_stream_missing'] for d in stream_summary['text_load_binning_sensitivity']):,.0f} dibanding n mentah 3.404.

Decile menyamakan komposisi volume secara kasar, bukan exact common support. Minimum `text_load` pada baris stream tersedia {stream_summary['common_text_load_support']['observed_min_present']:.4f}, lebih tinggi daripada minimum pada stream kosong. Sebanyak {stream_summary['common_text_load_support']['missing_rows_excluded']:,} ({stream_summary['common_text_load_support']['missing_rows_excluded']/stream_summary['missing_n']:.2%}) baris stream kosong berada di luar common observed range; batas irisan kedua rentang adalah {stream_summary['common_text_load_support']['common_observed_range'][0]:.4f}–{stream_summary['common_text_load_support']['common_observed_range'][1]:.4f}. Di dalam irisan ini {stream_summary['common_text_load_support']['missing_rows_retained']:,} baris stream kosong tertahan (dari {stream_summary['missing_n']:,}) dan mean gap mentah menyusut menjadi +{stream_summary['common_text_load_support']['mean_threat_delta_retained']:.3f}. Ini konsisten dengan komposisi volume rendah yang berkontribusi pada gap pooled, tetapi tidak memisahkan pengaruh volume dari semua perbedaan lain.

Standardisasi q-bin 5/10/20 memberi kontras +{stream_summary['text_load_binning_sensitivity'][0]['standardized_threat_mean_delta']:.3f}, {stream_summary['text_load_binning_sensitivity'][1]['standardized_threat_mean_delta']:+.3f}, dan {stream_summary['text_load_binning_sensitivity'][2]['standardized_threat_mean_delta']:+.3f}; seluruh konfigurasi menutup 100% baris, tetapi ESS grup stream-kosong turun ke {min(d['effective_n_stream_missing'] for d in stream_summary['text_load_binning_sensitivity']):,.0f}–{max(d['effective_n_stream_missing'] for d in stream_summary['text_load_binning_sensitivity']):,.0f}. Karena tandanya berubah dengan resolusi bin dan 26.6% grup stream-kosong berada di luar rentang volume bersama, ini **bukan** bukti pembalikan tanda yang robust atau penjelasan lengkap.

Untuk `artifact_signal`, pooled positive rate {stream_summary['artifact_rate_missing']:.2%} vs {stream_summary['artifact_rate_present']:.2%} berbeda {stream_summary['artifact_rate_delta']*100:+.2f} poin. Di strata tanpa missing lain, kontrasnya +10,79 poin; pada satu dan dua field lain yang kosong, kontrasnya −1,70 dan −7,78 poin. Setelah common-support reweighting strata 0–2 menurut hitungan gabungan (label-blind), selisih tetap positif +{stream_summary['standardized_artifact_delta_over_common_other_missing_counts']*100:.2f} poin. Jadi agregat menyembunyikan arah kontras yang berbeda menurut kelengkapan lain. Ini menguatkan kehati-hatian, bukan klaim stream kosong menaikkan artifact rate.

![Perbedaan asosiasi stream availability dan dua target](figures/signal_stream_target_contrasts.png)

![Distribusi target kumulatif menurut ketersediaan stream](figures/signal_stream_threat_ecdf.png)

![Konsentrasi dan pemeriksaan volume teks](figures/text_load_confounding_check.png)

Pattern-code-only common-support comparison masih memberi selisih mean threat +{stream_summary['pattern_code_common_support']['20']['row_weighted_mean_threat_delta']:.3f} pada minimum 20 baris per status (32 kelompok, mencakup {stream_summary['pattern_code_common_support']['20']['covered_share']:.1%} Train), atau +{stream_summary['pattern_code_common_support']['30']['row_weighted_mean_threat_delta']:.3f} pada minimum 30 (22 kelompok, coverage {stream_summary['pattern_code_common_support']['30']['covered_share']:.1%}). Namun rentang mean-delta per kelompok pada min30 lebar ({stream_summary['pattern_code_common_support']['30']['pattern_mean_delta_range'][0]:+.2f} hingga {stream_summary['pattern_code_common_support']['30']['pattern_mean_delta_range'][1]:+.2f}). Simultan dengan decile text-load memberi dukungan hanya {stream_summary['pattern_code_common_support']['20']['text_load_x_pattern_rows_covered']:,} baris ({stream_summary['pattern_code_common_support']['20']['text_load_x_pattern_share_covered']:.1%}) pada min20; jadi jenis pattern dan volume sulit dipisahkan secara menyeluruh.

Efek raw threat yang tampak stabil sesudah stratifikasi cycle mengecil pada pemeriksaan `text_load`; missing-stream sangat terkonsentrasi pada volume rendah. Ini temuan heterogenitas yang berguna tetapi mengurangi kekuatan pertanyaan sederhana tentang missingness. Model sebelumnya juga membahas error khusus missing-stream, jadi nilai barunya adalah profil label langsung dan sensitivity terhadap komposisi, bukan topik tanpa jejak sebelumnya. Bandingkan tetap dengan EDA text/category yang lain.

## Grafik

![Jumlah field dan outcome](figures/availability_count_and_outcomes.png)

Jumlah hilang dibuat label-blind; kurva outcome ditampilkan hanya untuk 0–3 field karena dukungan 4–5 kecil. Band target P10–P90 memperlihatkan penyebaran, bukan interval kepercayaan.

![Keterkaitan missingness](figures/co_missing_phi.png)

Phi meringkas ketergantungan dua indikator missingness. Ia tidak membuktikan apakah satu field seharusnya tersedia ketika field lain tersedia.

## Penilaian dan batas

Ini tema yang berbeda dari pertanyaan Rousan (entry_friction/impact dan artifact), Abhi (scope_pressure dan threat), Rizky (telemetry/module_spread dan artifact), dan EDA lama Samuel (impact–threat sepanjang cycle). Unit analisis utama di sini ialah **proses ketersediaan beberapa field laporan**; target digunakan sebagai deskripsi lanjutan, bukan pasangan fitur utama. Walau begitu, tiap-field hasil memberi konteks kepada Rizky; karena itulah telemetry dikeluarkan dan efek `signal_stream` tunggal tidak akan direkomendasikan sebagai pertanyaan Samuel.

Discovery ini belum menunjukkan bahwa satu skor completeness ringkas mempunyai hubungan yang kuat dan stabil dengan target: perbedaan pooled mengecil dan berganti tanda antar-cycle; klasifikasi sangat kecil; kelompok dengan 3+ missing makin jarang. Siklus dan jenis record dapat menjelaskan pola marginal. Missingness tidak dapat didiagnosis MCAR/MAR dari data ini, dan tidak ada klaim kausal. Seluruh Train dan hipotesis telah dilihat sehingga hasil bersifat exploratory.

**Status kandidat:** availability count luas adalah hasil negatif dan tidak direkomendasikan sebagai pertanyaan utama. Subcandidate yang layak dibandingkan: “Apakah distribusi `threat_rank` dan `artifact_signal` berbeda saat `signal_stream` kosong, dan sejauh mana perbedaannya bertahan setelah `text_load` diperhitungkan?” Ia berbeda dari Rousan (entry_friction/impact dan artifact), Abhi (scope_pressure dan threat), Rizky (telemetry/module_spread dan artifact), serta pertanyaan lama Samuel (impact–threat antarcycle). Keunggulannya ialah pertanyaan jelas dan gap threat lintas-cycle; kelemahannya ialah 76.7% stream kosong berada di decile `text_load` terbawah, hanya 73.4% punya volume dalam common observed range, penyesuaian bin sensitif terhadap resolusi, dan existing modelling telah menilai performa kelompok missing-stream. Ini kandidat pembanding dengan bukti serta batas yang menarik untuk diajarkan, bukan kandidat yang otomatis unggul atau bukti bahwa availability menjelaskan target. Belum ada rekomendasi untuk mengganti pertanyaan EDA lama sebelum perbandingan kandidat lain.

## Reproduksi

`python scripts/eda_candidates_missingness.py` membaca data via `TP2_DATA_DIR` atau resolver dataset resmi, memverifikasi hash Train yang dipatok dan seluruh fitur non-target bersama setelah menyelaraskan ID, lalu menulis statistik agregat dan grafik. Tidak ada model, label Test, raw text, ataupun baris individual yang disimpan.
'''
    (output_dir / "FINDINGS.md").write_text(text, encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "reports/eda_candidates/missingness")
    args = parser.parse_args()
    metadata = run(args.root, args.output_dir)
    print(json.dumps({"status": metadata["status"], "overall": metadata["overall"]}, indent=2))


if __name__ == "__main__":
    main()
