"""Focused descriptive support for Classification Round 1; never fits models."""
from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from src.data.audit import sha256, write_json
from src.utils.paths import PROJECT_ROOT, task_files


WINDOWS = [
    ("W1", (47, 55), (56, 59)),
    ("W2", (47, 59), (60, 63)),
    ("W3", (47, 63), (64, 67)),
    ("W4", (47, 67), (68, 71)),
    ("W5", (47, 71), (72, 75)),
    ("W6", (47, 75), (76, 79)),
    ("H12_E", (47, 55), (56, 67)),
    ("H12_L", (47, 67), (68, 79)),
]
CATEGORICAL = ["org_code", "module_code", "pattern_code"]
TEXT = "signal_stream"
TARGET = "artifact_signal"


def _class_counts(y: pd.Series | np.ndarray) -> dict[str, int]:
    values = pd.Series(y)
    counts = values.value_counts(dropna=False)
    return {str(label): int(counts.get(label, 0)) for label in [0, 1]}


def _binary_summary(frame: pd.DataFrame) -> dict:
    n = len(frame)
    y = frame[TARGET]
    counts = _class_counts(y)
    pos = counts["1"]
    return {"n": int(n), "class_counts": counts,
            "positive_rate": float(pos / n) if n else None,
            "single_class": bool(n > 0 and (counts["0"] == 0 or counts["1"] == 0))}


def _group_summary(frame: pd.DataFrame, column: str, *, max_groups: int | None = None) -> list[dict]:
    rows = []
    group = frame.groupby(column, dropna=False, sort=True, observed=True)
    for key, sub in group:
        item = {"value": None if pd.isna(key) else float(key) if isinstance(key, (int, float, np.number)) else str(key),
                **_binary_summary(sub)}
        rows.append(item)
    rows.sort(key=lambda item: (item["n"], str(item["value"])), reverse=True)
    return rows if max_groups is None else rows[:max_groups]


def _text_profile(frame: pd.DataFrame) -> dict:
    raw = frame[TEXT]
    text = raw.astype("string")
    stripped = text.str.strip()
    missing = raw.isna() | stripped.eq("").fillna(False)
    present = ~missing
    lengths = stripped[present].str.len().astype(float)
    per_class = {}
    for label in [0, 1]:
        label_mask = frame[TARGET].eq(label)
        values = lengths[label_mask[present].to_numpy()]
        q = np.quantile(values.to_numpy(), [0, .5, .95, 1]).tolist() if len(values) else []
        per_class[str(label)] = {
            "rows": int(label_mask.sum()),
            "missing_or_blank_rows": int((label_mask & missing).sum()),
            "missing_or_blank_rate": float((label_mask & missing).sum() / label_mask.sum()) if label_mask.sum() else None,
            "present_rows": int((label_mask & present).sum()),
            "length_quantiles_present": {str(k): float(v) for k, v in zip(["min", "median", "p95", "max"], q)},
        }
    return {
        "total_rows": int(len(frame)),
        "missing_or_blank_rows": int(missing.sum()),
        "missing_or_blank_rate": float(missing.mean()) if len(frame) else None,
        "present_rows": int(present.sum()),
        "length_quantiles_present": _quantile_summary(lengths),
        "by_target_class": per_class,
        "text_contents_exported": False,
    }


def _quantile_summary(values: pd.Series | np.ndarray) -> dict:
    arr = np.asarray(values, dtype=float)
    if not arr.size:
        return {}
    return {str(k): float(v) for k, v in zip(["min", "median", "p95", "max"], np.quantile(arr, [0, .5, .95, 1]))}


def _normalised_categories(series: pd.Series) -> pd.Series:
    value = series.astype("string").str.strip()
    return value.mask(value.eq(""))


def _categorical_shift(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    output = {}
    for col in CATEGORICAL:
        tr, te = _normalised_categories(train[col]), _normalised_categories(test[col])
        known = set(tr.dropna().tolist())
        nonmissing = te.notna()
        unseen = nonmissing & ~te.isin(known)
        output[col] = {
            "train_missing_or_blank_rows": int(tr.isna().sum()),
            "train_missing_or_blank_rate": float(tr.isna().mean()),
            "test_missing_or_blank_rows": int(te.isna().sum()),
            "test_missing_or_blank_rate": float(te.isna().mean()),
            "train_distinct_nonmissing": int(tr.nunique()),
            "test_distinct_nonmissing": int(te.nunique()),
            "test_rows_nonmissing": int(nonmissing.sum()),
            "test_rows_unseen_vs_full_train": int(unseen.sum()),
            "test_unseen_rate_among_nonmissing": float(unseen.sum() / nonmissing.sum()) if nonmissing.sum() else None,
            "category_values_exported": False,
        }
    return output


def _impact_review(frame: pd.DataFrame) -> dict:
    impact = frame["impact_load"].astype(float)
    # Feature thirds are defined by fit-data quantiles, avoiding arbitrary
    # [0,100] bins for a feature whose observed TP2 values exceed 100.
    cut1, cut2 = np.quantile(impact.to_numpy(), [1 / 3, 2 / 3])
    tertile = pd.cut(impact, [-np.inf, cut1, cut2, np.inf], labels=["low", "middle", "high"],
                     include_lowest=True, duplicates="drop")
    thirds = []
    for label in ["low", "middle", "high"]:
        sub = frame.loc[tertile.astype("string").eq(label)]
        thirds.append({"bin": label, "lower_cut_exclusive": None if label == "low" else float(cut1 if label == "middle" else cut2),
                       "upper_cut_inclusive": float(cut1 if label == "low" else cut2) if label != "high" else None,
                       **_binary_summary(sub)})

    fixed_bins = []
    lower_tertile, upper_tertile = 100 / 3, 200 / 3
    masks = [
        ("[0, 100/3)", (impact >= 0) & (impact < lower_tertile)),
        ("[100/3, 200/3)", (impact >= lower_tertile) & (impact < upper_tertile)),
        ("[200/3, 100]", (impact >= upper_tertile) & (impact <= 100)),
        (">100", impact > 100),
        ("<0", impact < 0),
    ]
    for label, mask in masks:
        sub = frame.loc[mask]
        fixed_bins.append({"bin": label, **_binary_summary(sub)})

    groups = _group_summary(frame, "impact_load")
    supported = [g for g in groups if g["n"] >= 100]
    rates = [g["positive_rate"] for g in supported if g["positive_rate"] is not None]
    return {
        "observed_range": [float(impact.min()), float(impact.max())],
        "observed_unique_values": int(impact.nunique()),
        "empirical_tertile_cutpoints_on_train_feature": {"p33_333": float(cut1), "p66_667": float(cut2)},
        "empirical_tertiles": thirds,
        "fixed_0_to_100_reference_bins": {"definition": "Literal supplied [0,100] thirds plus separate >100 and <0 overflow bins; these are not equal-frequency bins when applied to impact_load.",
                                           "bins": fixed_bins},
        "exact_impact_groups": groups,
        "supported_groups_n_ge_100": {"group_count": int(len(supported)),
                                        "row_coverage": int(sum(g["n"] for g in supported)),
                                        "positive_rate_min": float(min(rates)) if rates else None,
                                        "positive_rate_median": float(np.median(rates)) if rates else None,
                                        "positive_rate_max": float(max(rates)) if rates else None},
    }


def _cycle_review(frame: pd.DataFrame) -> dict:
    cycle_rows = []
    for cycle, sub in frame.groupby("cycle_code", sort=True):
        row = {"cycle_code": int(cycle), **_binary_summary(sub)}
        text_missing = sub[TEXT].isna() | sub[TEXT].astype("string").str.strip().eq("").fillna(False)
        row["stream_missing_rate"] = float(text_missing.mean()) if len(sub) else None
        cycle_rows.append(row)
    periods = []
    for start, end in [(47, 57), (58, 68), (69, 79)]:
        sub = frame.loc[frame.cycle_code.between(start, end)]
        periods.append({"cycles_inclusive": [start, end], **_binary_summary(sub),
                        "cycle_level_positive_rate_min": float(min(r["positive_rate"] for r in cycle_rows if start <= r["cycle_code"] <= end)),
                        "cycle_level_positive_rate_max": float(max(r["positive_rate"] for r in cycle_rows if start <= r["cycle_code"] <= end))})
    return {"cycles": cycle_rows, "periods": periods}


def _fold_review(frame: pd.DataFrame) -> list[dict]:
    output = []
    for name, fit_range, val_range in WINDOWS:
        fit_mask = frame.cycle_code.between(*fit_range)
        val_mask = frame.cycle_code.between(*val_range)
        fit, val = frame.loc[fit_mask], frame.loc[val_mask]
        if not len(fit) or not len(val) or fit.cycle_code.max() >= val.cycle_code.min():
            raise ValueError(f"Invalid temporal fold: {name}")
        per_feature = {}
        for col in CATEGORICAL:
            train_values = _normalised_categories(fit[col])
            val_values = _normalised_categories(val[col])
            vocab = set(train_values.dropna().tolist())
            nonmissing = val_values.notna()
            unseen = nonmissing & ~val_values.isin(vocab)
            per_feature[col] = {
                "fit_distinct": int(train_values.nunique()),
                "fit_missing_or_blank_rows": int(train_values.isna().sum()),
                "validation_missing_or_blank_rows": int(val_values.isna().sum()),
                "validation_rows_nonmissing": int(nonmissing.sum()),
                "validation_rows_unseen": int(unseen.sum()),
                "validation_unseen_rate_nonmissing": float(unseen.sum() / nonmissing.sum()) if nonmissing.sum() else None,
                "validation_unseen_class_counts": _class_counts(val.loc[unseen, TARGET]),
            }
        text_missing = val[TEXT].isna() | val[TEXT].astype("string").str.strip().eq("").fillna(False)
        output.append({
            "window": name, "fit_cycles_inclusive": list(fit_range), "validation_cycles_inclusive": list(val_range),
            "fit": _binary_summary(fit), "validation": _binary_summary(val),
            "validation_stream_missing_or_blank": _binary_summary(val.loc[text_missing]),
            "validation_stream_present": _binary_summary(val.loc[~text_missing]),
            "validation_category_novelty_fit_only": per_feature,
        })
    return output


def review_classification(data_dir=None, output_dir=None) -> dict:
    """Profile target, temporal/regime groups, stream and category novelty.

    Test features are used only for schema, missingness and train-vocabulary
    novelty rates. No test outcome, fitting, prediction, or raw sample export.
    """
    files = task_files("classification", data_dir)
    if not all(files.values()):
        absent = [role for role, path in files.items() if path is None]
        raise FileNotFoundError(f"Official classification files required; missing roles: {absent}")
    train = pd.read_csv(files["train"])
    test = pd.read_csv(files["test"])
    required_train = {TARGET, "cycle_code", "impact_load", "entry_friction", TEXT, *CATEGORICAL}
    required_test = {"cycle_code", "impact_load", "entry_friction", TEXT, *CATEGORICAL}
    if not required_train.issubset(train) or not required_test.issubset(test):
        raise ValueError("Classification source is missing columns required by this review.")
    if train[TARGET].isna().any() or not set(train[TARGET].unique()).issubset({0, 1}):
        raise ValueError("artifact_signal must be nonmissing binary labels 0/1.")
    if train["cycle_code"].isna().any() or test["cycle_code"].isna().any():
        raise ValueError("cycle_code must be present to summarize temporal windows.")
    if not train["impact_load"].notna().all() or not test["impact_load"].notna().all():
        raise ValueError("impact_load missingness needs explicit handling before this fixed review.")
    if not train["entry_friction"].dropna().isin([0, 1]).all():
        raise ValueError("entry_friction must contain binary values.")

    ef = []
    for value, sub in train.groupby("entry_friction", dropna=False, sort=True):
        ef.append({"entry_friction": None if pd.isna(value) else int(value), **_binary_summary(sub)})
    ef_by_class = []
    for label in [0, 1]:
        sub = train.loc[train[TARGET].eq(label)]
        total = len(sub)
        for value in [0, 1]:
            n = int((sub.entry_friction == value).sum())
            ef_by_class.append({"artifact_signal": label, "entry_friction": value, "n": n,
                                "share_within_class": float(n / total) if total else None})

    target_counts = _class_counts(train[TARGET])
    n = len(train)
    result = {
        "status": "CLASSIFICATION_DATA_REVIEW_COMPLETE_DESCRIPTIVE_ONLY",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Classification-only descriptive support. No model fit/prediction. No raw IDs or signal_stream contents exported. Test features used only for schema/missingness/category novelty.",
        "authority": {"target": TARGET, "target_meaning": "Public exploit-artifact indicator, confirmed by the user from the official competition page.",
                      "positive_class": 1, "metric": "Macro F1", "impact_load_and_entry_friction": "User-confirmed official TP2 predictors.",
                      "cycle_calendar_and_feature_timestamps": "UNKNOWN; cycle_code is treated as ordered disclosure cycles, not absolute dates."},
        "inputs": {role: {"name": path.name, "bytes": int(path.stat().st_size), "sha256": sha256(path)} for role, path in files.items()},
        "runtime": {"python": platform.python_version(),
                    "packages": {name: importlib.metadata.version(name) for name in ["pandas", "numpy", "scipy"]}},
        "shapes": {"train": list(train.shape), "test": list(test.shape)},
        "target": {"counts": target_counts, "positive_rate": float(target_counts["1"] / n),
                   "class_imbalance_ratio_majority_to_minority": float(max(target_counts.values()) / min(target_counts.values())),
                   "missing_count": int(train[TARGET].isna().sum())},
        "by_cycle": _cycle_review(train),
        "by_entry_friction": {"overall": ef, "within_class_share": ef_by_class},
        "impact_load_review": _impact_review(train),
        "signal_stream": {"train": _text_profile(train), "test_unsupervised": {
            "total_rows": int(len(test)),
            "missing_or_blank_rows": int((test[TEXT].isna() | test[TEXT].astype("string").str.strip().eq("").fillna(False)).sum()),
            "missing_or_blank_rate": float((test[TEXT].isna() | test[TEXT].astype("string").str.strip().eq("").fillna(False)).mean()),
            "length_quantiles_present": _quantile_summary(test.loc[test[TEXT].notna() & test[TEXT].astype("string").str.strip().ne(""), TEXT].astype("string").str.strip().str.len().astype(float)),
            "text_contents_exported": False}},
        "categorical_code_shift": _categorical_shift(train, test),
        "temporal_window_coverage": _fold_review(train),
        "interpretation_limits": [
            "All target-conditional results are descriptive associations, not causal effects or significance tests.",
            "Period positive-rate differences do not identify target drift mechanism; window sizes differ and adjacent cycles are not independent replicates.",
            "Test has no target, so test prevalence or model performance cannot be inferred.",
            "Features present in supplied scoring files do not establish operational timestamp or pre-outcome availability.",
            "Fixed 0-to-100 thirds applied to impact_load are only a scale-reference partition; actual feature values extend above 100. Empirical terciles are provided separately.",
        ],
    }
    if output_dir is not None:
        write_json(Path(output_dir) / "data_review.json", result)
    return result
