"""Descriptive, cache-only Regression V3 diagnostics.

This script reconstructs the fixed V2 three-component champion from verified
row-aligned validation prediction caches. It does not fit estimators, read Test,
search models, or select a candidate. Quantile edges and category vocabularies
are computed from the relevant outer fit rows only.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.paths import data_root  # noqa: E402

ENDS = (55, 59, 63, 67, 71, 75)
PRIMARY_WINDOWS = {end: (end + 1, end + 4) for end in ENDS}
PRIMARY_CACHE = ROOT / "experiments/runs/dual_reg_v1/stage1_paired_blends.npz"
PILOT_CACHE = ROOT / "experiments/runs/optimization_v2_regression_repr/pilot_predictions.npz"
FULL_CACHE = ROOT / "experiments/runs/optimization_v2_regression_repr/fullsix_predictions.npz"
FULL_REPORT = ROOT / "reports/optimization_v2/regression_repr/fullsix.json"
FULL_DIAGNOSTICS = ROOT / "reports/optimization_v2/regression_repr/fullsix_subgroup_diagnostics.json"
EARLY_LGB = ROOT / "experiments/runs/regression_research_v1/lgbtext__base__fit51__to55.npz"
EARLY_RIDGE = ROOT / "experiments/runs/regression_research_v1/model__hybrid_ridge__base__fit51__to55.npz"
OUT_DIR = ROOT / "reports/optimization_v3/regression_diagnostics"
RESEARCH_DIR = ROOT / "experiments/optimization_v3/regression_diagnostics"
RUN_DIR = ROOT / "experiments/runs/optimization_v3_reg_diagnostics"
MIN_GROUP_SUPPORT = 100
MIN_WINDOW_GROUP_SUPPORT = 50


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _metrics(frame: pd.DataFrame) -> dict[str, Any]:
    err = frame["prediction"].to_numpy(float) - frame["target"].to_numpy(float)
    ae = np.abs(err)
    return {
        "n": int(len(frame)),
        "target_p10": float(np.quantile(frame["target"].to_numpy(float), 0.10)),
        "target_median": float(np.median(frame["target"].to_numpy(float))),
        "target_p90": float(np.quantile(frame["target"].to_numpy(float), 0.90)),
        "signed_residual_p10": float(np.quantile(err, 0.10)),
        "mae": float(ae.mean()),
        "bias_prediction_minus_target": float(err.mean()),
        "median_signed_residual": float(np.median(err)),
        "signed_residual_p90": float(np.quantile(err, 0.90)),
        "p90_absolute_error": float(np.quantile(ae, 0.90)),
        "p95_absolute_error": float(np.quantile(ae, 0.95)),
        "over_rate": float(np.mean(err > 0)),
        "under_rate": float(np.mean(err < 0)),
        "tie_rate": float(np.mean(err == 0)),
    }


def _group_table(frame: pd.DataFrame, key: str) -> list[dict[str, Any]]:
    output = []
    for label, group in frame.groupby(key, dropna=False, sort=True, observed=True):
        row = {key: str(label), **_metrics(group)}
        row["meets_min_support_100"] = len(group) >= MIN_GROUP_SUPPORT
        output.append(row)
    return output


def _window_group_diagnostics(frame: pd.DataFrame) -> dict[str, Any]:
    keys = ("impact_load_q", "scope_pressure_q", "text_load_q", "scope_trace_count_q",
            "ef_group", "stream_missing", "tail", "category_any", "org_code_group",
            "module_code_group", "pattern_code_group")
    per_window: dict[str, Any] = {}
    effects: dict[str, list[dict[str, Any]]] = {key: [] for key in keys}
    for cutoff, window in frame.groupby("cutoff", sort=True):
        per_window[str(int(cutoff))] = {"n": int(len(window)), "groups": {}}
        for key in keys:
            per_window[str(int(cutoff))]["groups"][key] = _group_table(window, key)
        contrasts: list[tuple[str, str, str]] = []
        for key in ("impact_load_q", "scope_pressure_q", "text_load_q", "scope_trace_count_q"):
            labels = [x for x in window[key].dropna().astype(str).unique() if x.startswith("Q")]
            labels.sort(key=lambda x: int(x[1:]))
            if len(labels) >= 2:
                contrasts.append((key, labels[0], labels[-1]))
        contrasts.extend([
            ("ef_group", "EF0", "EF1"),
            ("stream_missing", "present", "missing"),
            ("tail", "high_ge95", "low_le5"),
            ("category_any", "all_observed_seen", "any_unseen"),
            ("org_code_group", "seen", "unseen"),
            ("module_code_group", "seen", "unseen"),
            ("pattern_code_group", "seen", "unseen"),
        ])
        for key, low, high in contrasts:
            left = window.loc[window[key].astype(str).eq(low)]
            right = window.loc[window[key].astype(str).eq(high)]
            supported = (len(left) >= MIN_WINDOW_GROUP_SUPPORT and len(right) >= MIN_WINDOW_GROUP_SUPPORT)
            le = left.prediction.to_numpy(float) - left.target.to_numpy(float)
            re = right.prediction.to_numpy(float) - right.target.to_numpy(float)
            effects[key].append({
                "cutoff": int(cutoff), "left_label": low, "right_label": high,
                "left_n": int(len(left)), "right_n": int(len(right)),
                "support_pass_both_n_ge_50": bool(supported),
                "mean_residual_right_minus_left": float(re.mean() - le.mean()) if len(left) and len(right) else None,
                "median_residual_right_minus_left": float(np.median(re) - np.median(le)) if len(left) and len(right) else None,
            })
    stability = {}
    for key, rows in effects.items():
        supported = [row for row in rows if row["support_pass_both_n_ge_50"]]
        signs = [np.sign(row["mean_residual_right_minus_left"]) for row in supported]
        stability[key] = {
            "supported_windows": len(supported),
            "positive_mean_contrast_windows": int(sum(s > 0 for s in signs)),
            "negative_mean_contrast_windows": int(sum(s < 0 for s in signs)),
            "zero_mean_contrast_windows": int(sum(s == 0 for s in signs)),
            "contrasts": rows,
        }
    return {"minimum_support_per_group_per_window": MIN_WINDOW_GROUP_SUPPORT,
            "per_window_group_metrics": per_window, "conditional_effect_stability": stability}


def _fit_quantile_bins(train_values: np.ndarray, validation_values: np.ndarray) -> tuple[np.ndarray, list[float]]:
    fit = np.asarray(train_values, dtype=float)
    val = np.asarray(validation_values, dtype=float)
    fit = fit[np.isfinite(fit)]
    edges = np.unique(np.quantile(fit, np.linspace(0, 1, 6)))
    if len(edges) < 2:
        return np.full(len(val), "Q1", dtype=object), [float(edges[0])]
    bins = np.full(len(val), "missing", dtype=object)
    valid = np.isfinite(val)
    indices = np.searchsorted(edges[1:-1], val[valid], side="left")
    bins[valid] = np.array([f"Q{i + 1}" for i in indices], dtype=object)
    return bins, [float(x) for x in edges]


def _source_data() -> tuple[Path, pd.DataFrame]:
    path = data_root() / "regression" / "Train.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Official Regression Train missing: {path}")
    frame = pd.read_csv(path)
    if "threat_rank" not in frame or "cycle_code" not in frame:
        raise ValueError("Official Regression Train schema is missing required columns")
    return path, frame


def _make_view(frame: pd.DataFrame, cutoff: int, low_cycle: int, high_cycle: int,
               sparse_cache: Any, base_cache: Any, *, view_name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    key = f"rows_lgbm_sparse_l1_{cutoff}"
    all_rows = np.asarray(sparse_cache[key], dtype=np.int64)
    all_raw_sparse = np.asarray(sparse_cache[f"raw_lgbm_sparse_l1_{cutoff}"], dtype=np.float64)
    all_cycles = frame.iloc[all_rows].cycle_code.to_numpy(dtype=int)
    selected = (all_cycles >= low_cycle) & (all_cycles <= high_cycle)
    rows = all_rows[selected]
    raw_sparse = all_raw_sparse[selected]
    expected = np.flatnonzero(frame.cycle_code.between(low_cycle, high_cycle).to_numpy())
    if not np.array_equal(rows, expected):
        raise ValueError(f"{view_name}@{cutoff} row positions do not match official Train cycles")
    base_all_rows = np.asarray(base_cache[f"rows_{cutoff}"], dtype=np.int64)
    if not np.array_equal(all_rows, base_all_rows):
        raise ValueError(f"{view_name}@{cutoff} sparse/base row positions differ")
    base_selected = (frame.iloc[base_all_rows].cycle_code.to_numpy(dtype=int) >= low_cycle) & (
        frame.iloc[base_all_rows].cycle_code.to_numpy(dtype=int) <= high_cycle)
    if not np.array_equal(rows, base_all_rows[base_selected]):
        raise ValueError(f"{view_name}@{cutoff} selected sparse/base rows differ")
    cycles = frame.iloc[rows].cycle_code.to_numpy(dtype=int)
    rank50 = np.empty(len(raw_sparse), dtype=np.float64)
    for cycle in np.unique(cycles):
        mask = cycles == cycle
        rank50[mask] = 0.5 * raw_sparse[mask] + 0.5 * 100.0 * rankdata(
            raw_sparse[mask], method="average") / int(mask.sum())
    rank50 = np.clip(rank50, 0.0, 100.0)
    lgb_svd = np.asarray(base_cache[f"lgb_{cutoff}"], dtype=np.float64)[base_selected]
    hybrid_ridge = np.asarray(base_cache[f"ridge_{cutoff}"], dtype=np.float64)[base_selected]
    ridge25 = np.asarray(base_cache[f"blend_ridge25_{cutoff}"], dtype=np.float64)[base_selected]
    base_reconstruction_error = float(np.max(np.abs(ridge25 - (0.75 * lgb_svd + 0.25 * hybrid_ridge))))
    if base_reconstruction_error > 1e-10:
        raise ValueError(f"Cached Ridge25 mixture identity mismatch at {cutoff}: {base_reconstruction_error}")
    champion = np.clip(0.75 * ridge25 + 0.25 * rank50, 0.0, 100.0)

    # Produce fit-defined numeric bins and category novelty flags. These are
    # explanatory strata only; no target-derived threshold enters their definition.
    fit_rows = np.flatnonzero(frame.cycle_code.to_numpy() <= cutoff)
    fit = frame.iloc[fit_rows]
    val = frame.iloc[rows]
    features = {
        "impact_load": pd.to_numeric(frame.impact_load, errors="coerce").to_numpy(float),
        "scope_pressure": pd.to_numeric(frame.scope_pressure, errors="coerce").to_numpy(float),
        "text_load": pd.to_numeric(frame.text_load, errors="coerce").to_numpy(float),
    }
    # scope_trace is model-parsed as a mixed numeric count; use the same pure parser.
    from src.models.round1_models import _mixed_count_arrays
    scope_trace_count = _mixed_count_arrays(frame.scope_trace)[0]
    features["scope_trace_count"] = np.asarray(scope_trace_count, dtype=float)

    record: dict[str, Any] = {
        "row": rows,
        "cutoff": np.full(len(rows), cutoff),
        "cycle": cycles,
        "target": frame.iloc[rows].threat_rank.to_numpy(dtype=float),
        "prediction": champion,
        "lgb_svd32": lgb_svd,
        "hybrid_ridge": hybrid_ridge,
        "sparse_lgb_rank50": rank50,
        "stream_missing": frame.iloc[rows].signal_stream.fillna("").astype(str).str.strip().eq("").to_numpy(),
        "ef1": frame.iloc[rows].entry_friction.to_numpy() == 1,
        "scope_trace_count": scope_trace_count[rows],
    }
    fit_edges: dict[str, list[float]] = {}
    for name in ("impact_load", "scope_pressure", "text_load", "scope_trace_count"):
        bins, edges = _fit_quantile_bins(features[name][fit_rows], features[name][rows])
        record[f"{name}_q"] = bins
        fit_edges[name] = edges

    category_novelty = {}
    categorical_cols = ("org_code", "module_code", "pattern_code")
    any_novel = np.zeros(len(rows), dtype=bool)
    any_observed = np.zeros(len(rows), dtype=bool)
    for name in categorical_cols:
        fit_raw = fit[name]
        val_raw = val[name]
        fit_present = fit_raw.notna() & fit_raw.astype(str).str.strip().ne("")
        val_present = val_raw.notna() & val_raw.astype(str).str.strip().ne("")
        vocab = set(fit_raw.loc[fit_present].astype(str).str.strip())
        value = val_raw.astype("string").str.strip()
        unseen = val_present.to_numpy() & ~value.isin(vocab).fillna(False).to_numpy()
        record[f"{name}_novel"] = unseen
        record[f"{name}_missing"] = ~val_present.to_numpy()
        category_novelty[name] = {
            "fit_rows": int(len(fit)),
            "fit_nonmissing_rows": int(fit_present.sum()),
            "fit_distinct_nonmissing": int(len(vocab)),
            "validation_rows": int(len(val)),
            "validation_missing": int((~val_present).sum()),
            "validation_nonmissing_unseen": int(unseen.sum()),
        }
        any_novel |= unseen
        any_observed |= val_present.to_numpy()
    record["any_category_novel"] = any_novel
    record["any_category_observed"] = any_observed
    out = pd.DataFrame(record)
    out["tail"] = np.select([out.target <= 5, out.target >= 95], ["low_le5", "high_ge95"], default="middle")
    out["stream_missing"] = np.where(out.stream_missing, "missing", "present")
    out["ef_group"] = np.where(out.ef1, "EF1", "EF0")
    out["category_any"] = np.select(
        [~out.any_category_observed, out.any_category_novel],
        ["all_missing", "any_unseen"], default="all_observed_seen")
    for name in categorical_cols:
        out[f"{name}_group"] = np.select(
            [out[f"{name}_missing"], out[f"{name}_novel"]],
            ["missing", "unseen"], default="seen")
    return out, {"cutoff": cutoff, "view": view_name, "fit_rows": int(len(fit)),
                 "validation_rows": int(len(rows)), "fit_quantile_edges": fit_edges,
                 "category_fit_and_validation_support": category_novelty,
                 "base_ridge25_reconstruction_max_abs_error": base_reconstruction_error}


def _build_view(frame: pd.DataFrame, base: Any, pilot: Any, full: Any,
                specs: list[tuple[int, int, int]]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    blocks, audits = [], []
    for cutoff, low, high in specs:
        cache = pilot if cutoff in (55, 67) else full
        block, audit = _make_view(frame, cutoff, low, high, cache, base, view_name=f"{low}-{high}")
        blocks.append(block)
        audits.append(audit)
    combined = pd.concat(blocks, ignore_index=True)
    if combined.row.duplicated().any():
        raise ValueError("A view contains duplicate Train rows")
    return combined, audits


def _all_tables(view: pd.DataFrame) -> dict[str, Any]:
    cycle = []
    for label, group in view.groupby("cycle", sort=True):
        cycle.append({"cycle": int(label), **_metrics(group)})
    by_quantile = {
        name: _group_table(view, f"{name}_q")
        for name in ("impact_load", "scope_pressure", "text_load", "scope_trace_count")
    }
    by_group = {
        "EF": _group_table(view, "ef_group"),
        "signal_stream": _group_table(view, "stream_missing"),
        "target_tail_descriptive": _group_table(view, "tail"),
        "any_raw_category_novelty": _group_table(view, "category_any"),
        "org_code_novelty": _group_table(view, "org_code_group"),
        "module_code_novelty": _group_table(view, "module_code_group"),
        "pattern_code_novelty": _group_table(view, "pattern_code_group"),
    }
    pairwise = {}
    components = ("lgb_svd32", "hybrid_ridge", "sparse_lgb_rank50")
    for i, left in enumerate(components):
        for right in components[i + 1:]:
            a = view[left].to_numpy(float)
            b = view[right].to_numpy(float)
            ea = a - view.target.to_numpy(float)
            eb = b - view.target.to_numpy(float)
            aa, ab = np.abs(ea), np.abs(eb)
            pairwise[f"{left}__vs__{right}"] = {
                "prediction_correlation": float(np.corrcoef(a, b)[0, 1]),
                "signed_error_correlation": float(np.corrcoef(ea, eb)[0, 1]),
                "absolute_error_correlation": float(np.corrcoef(aa, ab)[0, 1]),
                "left_lower_absolute_error_rate": float(np.mean(aa < ab)),
                "right_lower_absolute_error_rate": float(np.mean(ab < aa)),
                "tie_rate_1e-12": float(np.mean(np.isclose(aa, ab, rtol=0, atol=1e-12))),
                "median_absolute_prediction_difference": float(np.median(np.abs(a - b))),
                "p90_absolute_prediction_difference": float(np.quantile(np.abs(a - b), .90)),
            }
    biases = np.array([x["bias_prediction_minus_target"] for x in cycle])
    return {
        "overall": _metrics(view),
        "cycle": cycle,
        "cycle_direction_counts": {"positive_mean_residual_cycles": int(np.sum(biases > 0)),
                                    "negative_mean_residual_cycles": int(np.sum(biases < 0)),
                                    "zero_mean_residual_cycles": int(np.sum(biases == 0))},
        "fit_defined_numeric_quantiles": by_quantile,
        "categorical_and_availability_groups": by_group,
        "window_level_conditional_diagnostics": _window_group_diagnostics(view),
        "component_disagreement": pairwise,
        "component_recipe_weights": {"lgb_svd32": 0.5625, "hybrid_ridge": 0.1875,
                                     "sparse_lgb_rank50": 0.25},
        "minimum_support_rule": f"n >= {MIN_GROUP_SUPPORT}; groups below remain descriptive and are flagged",
    }


def _markdown(report: dict[str, Any]) -> str:
    p = report["views"]["primary"]
    h_e = report["views"]["H12_E"]
    h_l = report["views"]["H12_L"]
    lines = [
        "# V3 Regression — cache-only residual diagnosis",
        "",
        f"Generated from commit `{report['source_commit']}`; no estimator fits, Kaggle calls, or Test reads were performed.",
        "The six-window primary OOF and H12 arrays are reused development outcomes; H12 overlaps primary and is not replication. No outer-derived subgroup is a validation partition or a model-selection gate.",
        "",
        "## Reconstructed incumbent",
        "",
        "Recipe: `0.5625 × LGB SVD32 + 0.1875 × Hybrid Ridge + 0.25 × sparse LGB rank50`, clipped to [0,100]. It is reconstructed from existing row-exact arrays; the Ridge25 cache decomposition is checked to 1e-10.",
        "",
        "| View | Rows | MAE | Mean signed residual | Median signed residual | p95 absolute error | + / − cycle mean residuals |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, view in (("Primary six windows", p), ("H12_E", h_e), ("H12_L", h_l)):
        m = view["overall"]
        d = view.get("cycle_direction_counts", {})
        dirs = f"{d.get('positive_mean_residual_cycles', '—')} / {d.get('negative_mean_residual_cycles', '—')}"
        lines.append(f"| {name} | {m['n']:,} | {m['mae']:.4f} | {m['bias_prediction_minus_target']:+.3f} | {m['median_signed_residual']:+.3f} | {m['p95_absolute_error']:.3f} | {dirs} |")
    lines += ["", "## Key diagnostic signals", "",
        "Primary per-cycle residuals and fit-defined quantile strata are in `diagnostics.json` and `experiments/optimization_v3/regression_diagnostics/tables.json`. Each quantile edge is estimated from rows at or before that outer cutoff; tied edges collapse bins. Pooled minimum group support is 100 and per-window contrast support is 50 per side. Raw category novelty is measured against the fit's observed values, with missing kept separate; it is not identical to a model's rare-category or unknown encoding.",
              "Across the six primary folds, several conditional directions recur with adequate per-fold counts: EF1 minus EF0 mean residual is positive in 6/6 folds; any unseen raw org/module/pattern value minus all-observed-seen is positive in 6/6; and the descriptive low-tail minus high-tail mean residual is positive in 6/6. The latter contrast ranges +13.07 to +32.10 points, with each low/high tail side supported by at least 205 rows per fold. Stream-missing minus present is positive in 5/6 folds (342–526 missing rows/fold). Impact Q5−Q1 is positive in 5/6; text_load Q5−Q1 is negative in 5/6 but reverses in W1. Scope-pressure and parsed scope-trace Q5−Q1 are mixed, so those pooled gradients are not stable enough to state as a consistent direction.",
              "", "Target tails (≤5, ≥95) are descriptive, selected by the observed target after scoring, and cannot support a route or validation claim. Signal missingness uses null/empty/whitespace-only. EF and other supplied features are observed values.",
              "The tail shape is asymmetric shrinkage (low targets predicted high, high targets predicted low), repeated in all six primary folds and both H12 eras. It motivates a fixed-bin distributional / CDF-median comparison under MAE, but does not establish conditional multimodality or imply that binning will beat the L1 point estimate.",
              "", "The primary result should be read as error structure in repeatedly reused temporal development support, not a fresh estimate. The six primary folds contain 38,588 unique Train rows; H12_E and H12_L re-score those rows with overlapping temporal views.",
              "", "## Prior research audit and next hypotheses", "",
              "The attached machine summary records the covered prior families. V1/R1–R2 already tested structured HGB and feature/history ablations; hybrid word TF-IDF/Ridge; SVD32/128 and LightGBM-SVD32; rank policies; CatBoost; and fixed blend shares, including sparse-LGB promotion in V2. Do not repeat those grids.",
              "", "Ranked as information-gathering hypotheses, not selected candidates:", "",
              "1. **Past-only L1 residual corrector:** train the agreed shallow correction head on earlier, strictly cross-fitted incumbent residuals and component predictions plus fit-available cycle/EF/availability/parsed-volume covariates; roll it forward without same-row label reuse. Cycle drift and novelty strata motivate checking stability, but pooled subgroup patterns alone do not validate the corrector.",
              "2. **Fixed-bin distributional regression:** compare a predeclared ten-bin threat_rank classifier plus predicted-CDF median against the L1 point model. The primary low tail is overpredicted (+11.22 mean residual) and high tail underpredicted (−8.85), with the same direction in H12_E (+22.85/−13.32) and H12_L (+9.13/−6.96). This supports testing asymmetric tail calibration as a different objective; it does not demonstrate conditional multimodality, and a ten-bin approximation may lose precision.",
              "3. **Character-level sparse text channel:** compare a fixed character n-gram representation against existing word-token sparse/SVD families, then assess incremental residual complementarity. Character channels are not part of regression's prior V2 family; do not assume they help from classification results.",
              "Cross-fitted supervised token target statistics remain a later alternative, with strict earlier-label encoding and unseen-token fallback; high token sparsity and leakage risk make them lower priority than the three above.",
              "", "These are recommendations for MAIN to freeze or reject; this lane did not evaluate candidates, scores, or public data.",
              "", "## Early OOF cache boundary", "",
              "The raw end-51 LightGBM SVD32 and Hybrid Ridge arrays each contain 6,220 fit rows (47–51) and 5,800 disjoint scored rows (52–55), and both match PASS verification receipts. They are valid predecessor component OOF predictions. No sparse-L1 LightGBM prediction exists at cutoff 51, so they cannot reproduce the exact three-component champion. Further, the historical SVD32 rank mapping at the 55 cutoff was selected using 52–55 labels; applying that selected mapping to those same rows would contaminate a residual-head training target. Begin exact incumbent residual-head development with the 56–59 OOF block (cutoff 55), score 60–63 (cutoff 59), then expand past OOF history forward.",
              "", "## Exact source and reuse handles", "",
              "Cache/input SHA256s, fold keys, row counts, and the row-aligned OOF prediction payload are recorded in `diagnostics.json` and the ignored `experiments/runs/optimization_v3_reg_diagnostics/oof_residuals.npz`. Re-run `python scripts/optimization_v3_reg_diagnostics.py` from repository root to reproduce.", ""]
    return "\n".join(lines)


def run() -> dict[str, Any]:
    train_path, frame = _source_data()
    expected_sha = "756523ff8156a9cd776e52249f7e2a8f23dcec432b8d10570e55ff6db0e0f082"
    train_sha = sha256(train_path)
    if train_sha != expected_sha:
        raise ValueError(f"Regression Train hash differs from frozen V2 source: {train_sha}")
    if len(frame) != 50608 or int(frame.cycle_code.min()) != 47 or int(frame.cycle_code.max()) != 79:
        raise ValueError("Regression Train row or cycle boundary differs from frozen source")
    for path in (PRIMARY_CACHE, PILOT_CACHE, FULL_CACHE, FULL_REPORT, FULL_DIAGNOSTICS):
        if not path.is_file():
            raise FileNotFoundError(path)
    base = np.load(PRIMARY_CACHE, allow_pickle=False)
    pilot = np.load(PILOT_CACHE, allow_pickle=False)
    full = np.load(FULL_CACHE, allow_pickle=False)

    primary_specs = [(end, *PRIMARY_WINDOWS[end]) for end in ENDS]
    h12_specs = [(55, 56, 67), (67, 68, 79)]
    primary, primary_audit = _build_view(frame, base, pilot, full, primary_specs)
    h12, h12_audit = _build_view(frame, base, pilot, full, h12_specs)
    if len(primary) != 38588 or primary.row.nunique() != 38588:
        raise ValueError("Primary validation support must be exactly 38,588 unique Train rows")
    if len(h12) != 38588 or h12.row.nunique() != 38588:
        raise ValueError("H12 support must be exactly 38,588 unique Train rows")
    if set(primary.row) != set(h12.row):
        raise ValueError("Primary and H12 source row support differs")

    views = {
        "primary": _all_tables(primary),
        "H12_E": _all_tables(h12.loc[h12.cycle.between(56, 67)].copy()),
        "H12_L": _all_tables(h12.loc[h12.cycle.between(68, 79)].copy()),
        "H12_pooled": _all_tables(h12),
    }
    full_report = json.loads(FULL_REPORT.read_text(encoding="utf-8"))
    full_diagnostics = json.loads(FULL_DIAGNOSTICS.read_text(encoding="utf-8"))
    checks = {
        "primary_mae_matches_frozen_fullsix": abs(views["primary"]["overall"]["mae"] -
              full_report["candidate_gates"]["lgbm_sparse_l1"]["primary_candidate"]["mae"]) <= 1e-10,
        "H12_pooled_mae_matches_frozen_fullsix": abs(views["H12_pooled"]["overall"]["mae"] -
              full_diagnostics["metrics"]["lgbm_sparse_l1"]["H12_pooled"]["candidate_overall"]["mae"]) <= 1e-10,
        "all_Ridge25_cache_decompositions_exact": max(
            x["base_ridge25_reconstruction_max_abs_error"] for x in primary_audit + h12_audit) <= 1e-10,
    }
    # V2 subgroup diagnostics provide an independent source reference for reused tails/EF/stream.
    checks["frozen_subgroup_diagnostic_cache_hash_matches"] = (
        full_diagnostics["fullsix_prediction_cache_sha256"] == sha256(FULL_CACHE))
    if not all(checks.values()):
        raise ValueError(f"Cache reconstruction checks failed: {checks}")

    early_cache_audit = {}
    expected_fit = np.flatnonzero(frame.cycle_code.to_numpy() <= 51)
    expected_validation = np.flatnonzero(frame.cycle_code.between(52, 55).to_numpy())
    early_refs = {
        "lgb_svd32_raw": (EARLY_LGB, ROOT / "reports/regression_research/verification_stage5.json",
                          "lgbtext__base__fit51__to55.npz"),
        "hybrid_ridge_raw": (EARLY_RIDGE, ROOT / "reports/regression_research/verification_stage23.json",
                             "model__hybrid_ridge__base__fit51__to55.npz"),
    }
    for name, (path, verifier_path, filename) in early_refs.items():
        cache = np.load(path, allow_pickle=False)
        fit_rows = np.asarray(cache["fit_rows"], dtype=np.int64)
        val_rows = np.asarray(cache["validation_rows"], dtype=np.int64)
        verifier = json.loads(verifier_path.read_text(encoding="utf-8"))
        verified_hash = verifier["artifacts"]["model_cache_files"][filename]
        early_cache_audit[name] = {
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(path), "verified_cache_sha256": verified_hash,
            "verifier_status": verifier["status"], "fit_rows": int(len(fit_rows)),
            "fit_cycles": [int(frame.iloc[fit_rows].cycle_code.min()), int(frame.iloc[fit_rows].cycle_code.max())],
            "fit_positions_exact_47_51": bool(np.array_equal(fit_rows, expected_fit)),
            "validation_rows": int(len(val_rows)),
            "validation_cycles": [int(frame.iloc[val_rows].cycle_code.min()), int(frame.iloc[val_rows].cycle_code.max())],
            "validation_positions_exact_52_55": bool(np.array_equal(val_rows, expected_validation)),
            "fit_validation_disjoint": bool(not np.intersect1d(fit_rows, val_rows).size),
            "predictions_are_raw_before_rank_mapping": True,
            "prediction_shape": list(cache["prediction"].shape),
        }
        if (early_cache_audit[name]["sha256"] != verified_hash or verifier["status"] != "PASS"
                or not early_cache_audit[name]["fit_positions_exact_47_51"]
                or not early_cache_audit[name]["validation_positions_exact_52_55"]
                or not early_cache_audit[name]["fit_validation_disjoint"]):
            raise ValueError(f"Early OOF cache audit failed for {name}")
    sparse_cache_keys = set(pilot.files) | set(full.files)
    sparse_lgb_at_51_exists = any("lgbm_sparse_l1_51" in key for key in sparse_cache_keys)
    early_cache_audit["current_champion_feasibility"] = {
        "raw_svd32_and_hybrid_ridge_oof_components_exist_for_52_55": True,
        "sparse_lgb_oof_component_exists_at_cutoff_51": sparse_lgb_at_51_exists,
        "exact_three_component_incumbent_score_can_be_reconstructed_for_52_55": False,
        "rank_mapping_selection_caveat": "Regression research stage5 selects rank weight jointly on fit47-51 -> score52-55, so applying a selected rank policy back to 52-55 is outcome-informed. Do not use such mapped scores as residual-head training targets/features.",
        "safe_current_champion_residual_head_start": "Use exact current-champion OOF predictions from outer cutoff55 on cycles56-59 to train a past-only head for the next outer block (60-63), then expand the available past OOF history per fold. The 52-55 caches are predecessor raw components only, not the three-component champion.",
    }

    outputs = (primary, h12)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        RUN_DIR / "oof_residuals.npz",
        primary_rows=primary.row.to_numpy(np.int64), primary_cycle=primary.cycle.to_numpy(np.int16),
        primary_target=primary.target.to_numpy(np.float64), primary_prediction=primary.prediction.to_numpy(np.float64),
        primary_lgb_svd32=primary.lgb_svd32.to_numpy(np.float64), primary_hybrid_ridge=primary.hybrid_ridge.to_numpy(np.float64),
        primary_sparse_lgb_rank50=primary.sparse_lgb_rank50.to_numpy(np.float64),
        h12_rows=h12.row.to_numpy(np.int64), h12_cycle=h12.cycle.to_numpy(np.int16),
        h12_target=h12.target.to_numpy(np.float64), h12_prediction=h12.prediction.to_numpy(np.float64),
        h12_lgb_svd32=h12.lgb_svd32.to_numpy(np.float64), h12_hybrid_ridge=h12.hybrid_ridge.to_numpy(np.float64),
        h12_sparse_lgb_rank50=h12.sparse_lgb_rank50.to_numpy(np.float64),
    )
    hashes = {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in
              (train_path, PRIMARY_CACHE, PILOT_CACHE, FULL_CACHE, FULL_REPORT, FULL_DIAGNOSTICS,
               EARLY_LGB, EARLY_RIDGE)}
    report = {
        "experiment_id": "optimization_v3_regression_cache_only_diagnostics",
        "source_commit": __import__("subprocess").check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "scope": {"fits": 0, "test_read": False, "kaggle_calls": 0, "candidate_score_search": False,
                  "outer_error_based_routing_or_model_selection": False, "official_train_rows": len(frame),
                  "primary_unique_rows": int(primary.row.nunique()), "h12_unique_rows": int(h12.row.nunique()),
                  "h12_overlaps_primary": True, "minimum_group_support": MIN_GROUP_SUPPORT},
        "source_and_cache_sha256": hashes,
        "recipe": {"components": {"LGB SVD32": 0.5625, "Hybrid Ridge": 0.1875,
                                   "sparse LightGBM L1 rank50": 0.25},
                   "sparse_rank50": "0.5 raw + 0.5 within-cycle average-rank percentile; clip [0,100]",
                   "final_clip": "[0,100]", "rows": "source positional indices, checked against exact cycle windows"},
        "cache_checks": checks,
        "early_oof_cache_audit": early_cache_audit,
        "fit_and_row_audits": {"primary": primary_audit, "H12": h12_audit},
        "views": views,
        "prior_research_audit": [
            {"source": "docs/REGRESSION_RESEARCH.md and reports/regression_research/FINDINGS.md", "covered": "R1 baseline and feature expansion; structured HGB, text-only and hybrid Ridge; SVD HGB and LightGBM-SVD; temporal/history, rank-policy and subgroup diagnostics; CatBoost and category controls.", "implication": "Do not repeat structured-only, SVD-dimension, rank-policy, old feature/history, or CatBoost grids."},
            {"source": "docs/REGRESSION_ROUND2.md and reports/regression_round2/FINDINGS.md", "covered": "Expanded structured feature ablations, recent/matched training-history sensitivity, residual/tail/impact groups, novelty and fit-only parsing.", "implication": "Residual diagnostics already exist; reaggregate the current 3-component champion only, do not re-run those ablations."},
            {"source": "reports/dual_campaign/regression/STAGE1_PAIRED_BLEND.md and stage1_paired_blend.json", "covered": "Ridge25/50 paired blends, inner selected weights, tails/EF/stream, and strong prediction/error correlation between Ridge and LGB SVD32.", "implication": "No old blend-weight sweep; V2 sparse-LGB complementarity is new only as a fixed component already promoted."},
            {"source": "reports/optimization_v2/FINDINGS.md and reports/optimization_v2/regression_repr/fullsix.json", "covered": "sparse L1/SGD fixed 25% blends with rank50 across all six windows, H12, tails, EF, stream and all-cycle complementarity.", "implication": "Treat V2 as champion recipe; no re-search of blend weights, SVD dimensions, or sparse LGB settings."},
        ],
        "next_hypotheses_not_tested": [
            {"rank": 1, "hypothesis": "strictly past-only nonlinear OOF residual correction using earlier champion component predictions plus fit-available structured covariates", "why_information_value": "tests whether recurring conditional signed residuals can be corrected while holding base predictions fixed; distinct from the already-tested global convex shares", "key_risk": "the same outer residual tables here are descriptive only and cannot fit/select the correction; OOF meta-training must be nested in time"},
            {"rank": 2, "hypothesis": "character n-gram sparse regression channel added as a fixed complementary text representation", "why_information_value": "captures character fragments/format variation outside prior word-token and SVD configurations; category of experiment remains untested in regression audit", "key_risk": "larger sparse design and temporal transfer may erase any gain; no classification result is assumed to transfer"},
            {"rank": 3, "hypothesis": "nested cross-fitted supervised token target statistics with strictly earlier-label encoding and explicit unseen-token fallback", "why_information_value": "tests whether token-level supervised associations complement TF-IDF/linear and sparse tree representations", "key_risk": "high leakage and sparse-token overfit risk; fold chronology and token support must be audited before scoring"},
        ],
        "interpretation_limits": [
            "Every primary cycle and both H12 views have been reused in prior adaptive development; none is untouched validation.",
            "H12_E/H12_L re-score the same 38,588 rows in overlapping temporal views; do not count these as independent replications.",
            "Subgroups are descriptive; target tails are defined using observed target after prediction and cannot be used as a deployable route.",
            "Fit-defined feature quantile boundaries vary by fold; bin labels are relative to each fold, not identical global numeric cutoffs.",
            "Raw category novelty means exact nonmissing value absent from fit; it does not match rare-category grouping/encoder behavior exactly.",
            "No confidence intervals or causal conclusions are computed."
        ],
        "reproduction": "python scripts/optimization_v3_reg_diagnostics.py",
        "ignored_row_aligned_cache": "experiments/runs/optimization_v3_reg_diagnostics/oof_residuals.npz",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "diagnostics.json").write_text(json.dumps(_jsonable(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT_DIR / "FINDINGS.md").write_text(_markdown(report), encoding="utf-8")
    (RESEARCH_DIR / "protocol.json").write_text(json.dumps({
        "experiment_id": report["experiment_id"], "mode": "cache-only descriptive reaggregation",
        "input_paths": list(hashes), "input_sha256": hashes, "fit_count": 0, "test_access": False,
        "group_support_minimum": MIN_GROUP_SUPPORT, "quantile_rule": "per outer fold, fit rows only; empirical 5-quantiles",
        "category_novelty_rule": "exact nonmissing raw category unseen in outer fit; missing separated",
        "target_tails": "target <= 5 and >= 95; descriptive after scoring only; not a routing rule",
        "stop_point": "handoff residual diagnostics and reuse recommendations to MAIN; no candidate selection",
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (RESEARCH_DIR / "tables.json").write_text(json.dumps(_jsonable(views), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({"status": "PASS", "views": {k: v["overall"] for k, v in result["views"].items()},
                      "cache_checks": result["cache_checks"], "reports": str(OUT_DIR.relative_to(ROOT))}, indent=2))
