"""Read-only error diagnostics for the cached TP2 Classification candidates.

This script reads official Classification Train only and previously saved
out-of-fold probability caches. It does not fit models, score Test, or select a
new route. All subgroup boundaries are derived from the corresponding past-only
fit rows; outer labels are used only to describe held-out errors.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.classification_round2_models import stream_missing_mask
from src.utils.paths import task_files


PRIMARY_WINDOWS = {
    "W1": (55, 56, 59), "W2": (59, 60, 63), "W3": (63, 64, 67),
    "W4": (67, 68, 71), "W5": (71, 72, 75), "W6": (75, 76, 79),
}
H12_WINDOWS = {"H12_E": (55, 56, 67), "H12_L": (67, 68, 79)}
ALL_WINDOWS = {**PRIMARY_WINDOWS, **H12_WINDOWS}
ROUND1_RUNS = ROOT / "experiments/runs/classification_round1_v1"
ROUND1_AUDITS = ROOT / "reports/classification_round1/fit_audits"
CATBOOST_RUNS = ROOT / "experiments/runs/dual_cls_catboost_v1"
STAGE1_RUNS = ROOT / "experiments/runs/dual_cls_v1a"
STAGE2_RUNS = ROOT / "experiments/runs/dual_cls_v1_stage2"
REPORT_DIR = ROOT / "reports/optimization_v3/classification_diagnostics"
RUN_DIR = ROOT / "experiments/runs/optimization_v3_cls_diagnostics"
NUMERIC_FEATURES = ["impact_load", "entry_friction", "scope_pressure", "text_load"]
CATEGORY_FEATURES = ["org_code", "module_code", "pattern_code"]
ALT_MODELS = ["DC_NBSVM", "DC_SVC", "DC_WORD_CHAR"]
MIN_CELL_N = 50
MIN_POOLED_CELL_N = 200
MIN_SUPPORTED_WINDOWS = 4


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False, default=str).encode("utf-8")).hexdigest()


def array_digest(value: np.ndarray) -> str:
    # Matches the archived campaign convention: native array flattened as LE f8.
    return hashlib.sha256(np.asarray(value, dtype="<f8").tobytes()).hexdigest()


def file_meta(cache: Path) -> tuple[dict[str, Any], str]:
    meta_path = cache.with_suffix(".json")
    if not cache.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"Required cached prediction is absent: {cache}")
    return json.loads(meta_path.read_text(encoding="utf-8")), sha256(meta_path)


def expected_rows(cycle: np.ndarray, fit_end: int, horizon: int) -> np.ndarray:
    return np.flatnonzero((cycle > fit_end) & (cycle <= horizon))


def prediction_horizon(fit_end: int) -> int:
    if fit_end == 55:
        return 67
    if fit_end == 67:
        return 79
    return PRIMARY_WINDOWS[next(name for name, item in PRIMARY_WINDOWS.items() if item[0] == fit_end)][2]


def check_probability(p: np.ndarray, label: str) -> None:
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError(f"Invalid cached probability vector: {label}")


def load_round1(fit_end: int, model_id: str, cycle: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    cache = ROUND1_RUNS / f"fit{fit_end}__{model_id}.npz"
    meta_path = ROUND1_AUDITS / f"fit{fit_end}.json"
    if not cache.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"Round 1 control missing: fit{fit_end}/{model_id}")
    with np.load(cache, allow_pickle=False) as z:
        rows = z["outer_rows"].astype(np.int64, copy=True)
        p = z["probability"].astype(float, copy=True)
    exp = expected_rows(cycle, fit_end, prediction_horizon(fit_end))
    if not np.array_equal(rows, exp) or rows.shape != p.shape:
        raise ValueError(f"Round 1 control row identity mismatch: fit{fit_end}/{model_id}")
    check_probability(p, f"fit{fit_end}/{model_id}")
    meta_root = json.loads(meta_path.read_text(encoding="utf-8"))
    meta = meta_root.get(model_id)
    if not isinstance(meta, dict) or meta.get("model") != model_id:
        raise ValueError(f"Round 1 audit identity mismatch: fit{fit_end}/{model_id}")
    prob_hash = array_digest(p)
    if meta.get("probability_sha256") != prob_hash:
        raise ValueError(f"Round 1 probability hash mismatch: fit{fit_end}/{model_id}")
    return rows, p, {"cache": str(cache.relative_to(ROOT)), "cache_sha256": sha256(cache),
                     "audit": str(meta_path.relative_to(ROOT)), "audit_sha256": sha256(meta_path),
                     "probability_sha256": prob_hash}


def load_catboost(fit_end: int, cycle: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    cache = CATBOOST_RUNS / f"fit{fit_end}.npz"
    meta, meta_hash = file_meta(cache)
    with np.load(cache, allow_pickle=False) as z:
        rows = z["rows"].astype(np.int64, copy=True)
        p = z["candidate"].astype(float, copy=True)
    exp = expected_rows(cycle, fit_end, prediction_horizon(fit_end))
    if not np.array_equal(rows, exp) or rows.shape != p.shape:
        raise ValueError(f"CB_MISSING row identity mismatch at fit{fit_end}")
    check_probability(p, f"CB_MISSING/fit{fit_end}")
    p_hash = array_digest(p)
    if meta.get("prediction_sha256") != p_hash or meta.get("npz_sha256") != sha256(cache):
        raise ValueError(f"CB_MISSING cache or probability hash mismatch at fit{fit_end}")
    return rows, p, {"cache": str(cache.relative_to(ROOT)), "cache_sha256": sha256(cache),
                     "metadata": str(cache.with_suffix('.json').relative_to(ROOT)),
                     "metadata_sha256": meta_hash, "probability_sha256": p_hash,
                     "fit_rows": meta.get("fit_rows"), "state": meta.get("state")}


def candidate_cache(fit_end: int, model_id: str) -> Path:
    # Stage 1 contains full-span fits for 55/67; later cutoffs are four-cycle
    # Stage 2 fits. Never substitute model outputs across cutoffs.
    base = STAGE1_RUNS if fit_end in (55, 67) else STAGE2_RUNS
    return base / f"fit{fit_end}__{model_id}.npz"


def load_candidate(fit_end: int, model_id: str, start: int, end: int,
                   cycle: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    cache = candidate_cache(fit_end, model_id)
    meta, meta_hash = file_meta(cache)
    with np.load(cache, allow_pickle=False) as z:
        rows = z["outer_rows"].astype(np.int64, copy=True)
        p = z["candidate_probability"].astype(float, copy=True)
    horizon = prediction_horizon(fit_end)
    exp = expected_rows(cycle, fit_end, horizon)
    if not np.array_equal(rows, exp) or rows.shape != p.shape:
        raise ValueError(f"{model_id} row identity mismatch at fit{fit_end}")
    check_probability(p, f"{model_id}/fit{fit_end}")
    p_hash = array_digest(p)
    if meta.get("candidate_probability_sha256") != p_hash:
        raise ValueError(f"{model_id} probability hash mismatch at fit{fit_end}")
    chosen = (cycle[rows] >= start) & (cycle[rows] <= end)
    selected_rows = rows[chosen]
    expected_eval = np.flatnonzero((cycle >= start) & (cycle <= end))
    if not np.array_equal(selected_rows, expected_eval):
        raise ValueError(f"{model_id} selected validation rows mismatch at fit{fit_end}, {start}-{end}")
    return selected_rows, p[chosen], {"cache": str(cache.relative_to(ROOT)),
                                      "cache_sha256": sha256(cache),
                                      "metadata": str(cache.with_suffix('.json').relative_to(ROOT)),
                                      "metadata_sha256": meta_hash,
                                      "probability_sha256": p_hash,
                                      "fit_rows": meta.get("fit_rows"),
                                      "outer_labels_used_for_fit_or_selection": meta.get("outer_labels_used_for_fit_or_selection")}


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(p, dtype=float)
    pred = p >= 0.5
    actual = y == 1
    tp = int(np.count_nonzero(pred & actual))
    tn = int(np.count_nonzero(~pred & ~actual))
    fp = int(np.count_nonzero(pred & ~actual))
    fn = int(np.count_nonzero(~pred & actual))
    precision_pos = tp / (tp + fp) if tp + fp else 0.0
    recall_pos = tp / (tp + fn) if tp + fn else 0.0
    precision_neg = tn / (tn + fn) if tn + fn else 0.0
    recall_neg = tn / (tn + fp) if tn + fp else 0.0
    f1_pos = 2 * precision_pos * recall_pos / (precision_pos + recall_pos) if precision_pos + recall_pos else 0.0
    f1_neg = 2 * precision_neg * recall_neg / (precision_neg + recall_neg) if precision_neg + recall_neg else 0.0
    return {"n": int(y.size), "positive": int(actual.sum()),
            "positive_prevalence": float(actual.mean()) if y.size else None,
            "positive_prediction_rate": float(pred.mean()) if y.size else None,
            "macro_f1": float((f1_pos + f1_neg) / 2) if y.size else None,
            "f1_positive": float(f1_pos), "f1_negative": float(f1_neg),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "error_count": fp + fn,
            "error_rate": float((fp + fn) / y.size) if y.size else None,
            "brier": float(np.mean((p - y) ** 2)) if y.size else None,
            "mean_probability": float(p.mean()) if y.size else None,
            "mean_signed_residual_p_minus_y": float(np.mean(p - y)) if y.size else None,
            "threshold": 0.5}


def comparison(y: np.ndarray, p_base: np.ndarray, p_candidate: np.ndarray) -> dict[str, Any]:
    base = np.asarray(p_base) >= 0.5
    cand = np.asarray(p_candidate) >= 0.5
    actual = np.asarray(y, dtype=np.int8) == 1
    base_error = base != actual
    cand_error = cand != actual
    base_only_right = int(np.count_nonzero(cand_error & ~base_error))
    cand_only_right = int(np.count_nonzero(base_error & ~cand_error))
    n = int(actual.size)
    b = metrics(y, p_base)
    c = metrics(y, p_candidate)
    return {"n": n, "both_correct": int(np.count_nonzero(~base_error & ~cand_error)),
            "both_wrong": int(np.count_nonzero(base_error & cand_error)),
            "base_only_correct": base_only_right,
            "candidate_only_correct": cand_only_right,
            "net_candidate_correct_count_delta": cand_only_right - base_only_right,
            "disagreements": int(np.count_nonzero(base != cand)),
            "base_error_count": b["error_count"], "candidate_error_count": c["error_count"],
            "base_error_rate": b["error_rate"], "candidate_error_rate": c["error_rate"],
            "delta_error_rate_candidate_minus_base": (c["error_rate"] - b["error_rate"]) if n else None,
            "base_macro_f1": b["macro_f1"], "candidate_macro_f1": c["macro_f1"],
            "delta_macro_f1_candidate_minus_base": (c["macro_f1"] - b["macro_f1"]) if n else None,
            "base_brier": b["brier"], "candidate_brier": c["brier"],
            "delta_brier_candidate_minus_base": (c["brier"] - b["brier"]) if n else None}


def fit_quantile_labels(train_values: pd.Series, outer_values: pd.Series) -> tuple[np.ndarray, list[float]]:
    fit = pd.to_numeric(train_values, errors="coerce").dropna().to_numpy(dtype=float)
    out = pd.to_numeric(outer_values, errors="coerce").to_numpy(dtype=float)
    if fit.size == 0:
        return np.full(out.shape, "missing", dtype=object), []
    edges = np.quantile(fit, [0.25, 0.50, 0.75]).astype(float)
    labels = np.empty(out.size, dtype=object)
    missing = ~np.isfinite(out)
    labels[missing] = "missing"
    # Right-closed bins keep ties at empirical quartile cutpoints in the lower
    # interval; this matters for entry_friction, whose fitted values are discrete.
    labels[~missing] = np.asarray([f"Q{i + 1}" for i in np.searchsorted(edges, out[~missing], side="left")], dtype=object)
    return labels, [float(v) for v in edges]


def paired_group_row(y: np.ndarray, base: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    result = comparison(y, base, candidate)
    result.update({"base_mean_p_minus_y": metrics(y, base)["mean_signed_residual_p_minus_y"],
                   "candidate_mean_p_minus_y": metrics(y, candidate)["mean_signed_residual_p_minus_y"],
                   "base_mean_probability": metrics(y, base)["mean_probability"],
                   "candidate_mean_probability": metrics(y, candidate)["mean_probability"]})
    return result


def grouped_comparisons(name: str, labels: np.ndarray, y: np.ndarray, base: np.ndarray,
                        candidate: np.ndarray, window: str, min_n: int = MIN_CELL_N) -> list[dict[str, Any]]:
    output = []
    for label in sorted(set(str(v) for v in labels)):
        mask = np.asarray([str(v) == label for v in labels], dtype=bool)
        if int(mask.sum()) < min_n:
            continue
        output.append({"axis": name, "group": label, "window": window,
                       **paired_group_row(y[mask], base[mask], candidate[mask])})
    return output


def summarize_stable_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["axis"], row["group"]), []).append(row)
    out = []
    for (axis, group), items in sorted(grouped.items()):
        n = sum(item["n"] for item in items)
        base_err = sum(item["base_error_count"] for item in items)
        cand_err = sum(item["candidate_error_count"] for item in items)
        deltas = [item["delta_error_rate_candidate_minus_base"] for item in items]
        base_residuals = [item["base_mean_p_minus_y"] for item in items]
        candidate_residuals = [item["candidate_mean_p_minus_y"] for item in items]
        out.append({"axis": axis, "group": group, "n_pooled": n,
                    "windows_supported_n_ge_50": len(items),
                    "windows_candidate_lower_error": sum(d < 0 for d in deltas),
                    "windows_candidate_higher_error": sum(d > 0 for d in deltas),
                    "base_error_rate_pooled": base_err / n if n else None,
                    "candidate_error_rate_pooled": cand_err / n if n else None,
                    "delta_error_rate_pooled": (cand_err - base_err) / n if n else None,
                    "base_mean_p_minus_y_pooled": float(sum(item["n"] * item["base_mean_p_minus_y"] for item in items) / n) if n else None,
                    "candidate_mean_p_minus_y_pooled": float(sum(item["n"] * item["candidate_mean_p_minus_y"] for item in items) / n) if n else None,
                    "windows_base_overpredicting_pooled_residual": sum(value > 0 for value in base_residuals),
                    "windows_base_underpredicting_pooled_residual": sum(value < 0 for value in base_residuals),
                    "windows_candidate_overpredicting_pooled_residual": sum(value > 0 for value in candidate_residuals),
                    "windows_candidate_underpredicting_pooled_residual": sum(value < 0 for value in candidate_residuals),
                    "median_window_delta_error_rate": float(np.median(deltas)) if deltas else None,
                    "cross_window_eligible": bool(n >= MIN_POOLED_CELL_N and len(items) >= MIN_SUPPORTED_WINDOWS)})
    return out


def pct(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{100 * value:.{digits}f}%"


def f1(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def render_findings(audit: dict[str, Any]) -> str:
    pooled = audit["pooled_metrics"]
    rows = [
        "# Optimization V3 — Classification diagnostics",
        "",
        "**Status: descriptive cache audit only. No model fit, Test read, or route selection was performed.**",
        "",
        "## Data and method",
        "",
        f"Official Classification Train: **{audit['source_facts']['rows']:,} rows**, cycles 47–79, SHA-256 `{audit['input_identity']['train']['sha256']}`. The audited primary predictions cover 38,588 non-overlapping outer rows (36,010 stream-present; 2,578 stream-missing). H12 reuses those outcomes and is a second horizon view, not an independent holdout.",
        "",
        "All model scores came from archived past-only prediction caches; comparisons use the frozen 0.50 decision cutoff. Numeric bins use empirical quartile boundaries learned from cycles 47 through each fold cutoff, applied to its outer rows. Unknown category means a nonmissing outer `org_code`, `module_code`, or `pattern_code` absent from that fold’s fit vocabulary. Confidence bins are fixed C2_B probability intervals. Per-window subgroup minimum is n=50; pooled stability requires n≥200 and support in at least four of six primary windows. Fits and labels from outer rows are never used to define features, bins, cutoffs, or a route.",
        "",
        "## Temporal error summary",
        "",
        "| View | Rows | Champion Macro F1 | Prior C2_B/C1_B route | Missing CB vs C1_B ΔF1 | Present C2_B Macro F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for view, cohort in [("Primary six windows", "primary_six"), ("H12 pooled", "H12")]:
        all_view = pooled[cohort]["all"]
        missing_view = pooled[cohort]["missing"]["CB_vs_C1B"]
        present_view = pooled[cohort]["present"]
        n = all_view["Champion_C2B_present_CB_missing"]["n"]
        rows.append(f"| {view} | {n:,} | {f1(all_view['Champion_C2B_present_CB_missing']['macro_f1'])} | {f1(all_view['PreCB_C2B_present_C1B_missing']['macro_f1'])} | {missing_view['delta_macro_f1_candidate_minus_base']:+.4f} | {f1(present_view['C2_B']['macro_f1'])} |")
    rows += [
        "",
        "The champion’s primary gain over the former C2_B-present/C1_B-missing route is confined to the missing branch: 70 net additional correct predictions among 2,578 missing-stream rows (208 CB-only correct vs 138 C1_B-only correct). CB improves that branch in W1/W2/W3/W5/W6, but loses W4; H12_E improves and H12_L loses. The C2_B present branch itself is unchanged.",
        "",
        "| Window | Present n | C2_B F1 | Raw CB score F1 on present (counterfactual) | CB−C1_B missing ΔF1 | Champion all F1 | Prior route all F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in [*PRIMARY_WINDOWS, *H12_WINDOWS]:
        fold = audit["folds"][name]
        m = fold["metrics"]
        rows.append(f"| {name} | {fold['present_n']:,} | {f1(m['present']['C2_B']['macro_f1'])} | {f1(m['present']['CB_MISSING_raw']['macro_f1'])} | {m['missing']['CB_vs_C1B']['delta_macro_f1_candidate_minus_base']:+.4f} | {f1(m['all']['Champion_C2B_present_CB_missing']['macro_f1'])} | {f1(m['all']['PreCB_C2B_present_C1B_missing']['macro_f1'])} |")
    rows += [
        "",
        "## Present-stream complementarity and counterexamples",
        "",
        "The structured CB_MISSING model was fitted on past data and only routed to missing rows in its original experiment. Its score on present rows is shown strictly as a counterfactual diagnostic. Across 36,010 present rows, direct CB predictions have Macro F1 0.7592 and 8,004 errors, versus C2_B 0.8286 and 5,609 errors. CB fixes 1,979 C2_B errors but breaks 4,374 C2_B-correct rows. It loses every primary window (−2.57 to −11.85 Macro F1 points) and both H12 segments. This strongly argues against treating the cached raw CB score as a generally useful present-stream head input; any future inclusion should be a predeclared ablation, not a presumed gain.",
        "",
        "Raw CB also has higher error in every fixed C2_B-confidence band, each supported in all six primary windows: +3.86 pp below .20; +10.86 pp at .20–.40; +2.32 pp at .40–.60; +12.57 pp at .60–.80; +8.95 pp at ≥.80. For C2_B confidence .60–.80 × text_load Q4 it is +20.85 pp (n=849; worse in all six windows). A near-neutral counterexample is confidence .40–.60 × text_load Q2: −0.21 pp pooled, but mixed across windows (2 lower-error, 4 higher-error). No eligible confidence × text_load or confidence × entry_friction cell has raw CB lower error in at least four windows.",
        "",
        "Unknown-category rows are sizeable and are not a rescue for raw CB: 17,531 present rows have at least one unseen nonmissing structured category. Raw CB adds +10.23 error pp there, versus +3.25 pp where all three categories are known. Individually, unseen `org_code` has +12.12 pp (n=9,873) and unseen `module_code` +10.22 pp (n=16,173). Unseen `pattern_code` has only 240 rows pooled and two supported windows, so that subgroup is too sparse for a stable conclusion.",
        "",
        "Cached present-stream alternatives are also counterevidence against simply adding another fixed text score: DC_NBSVM changes primary Macro F1 by −0.0027 and error by +0.07 pp; on H12 its F1 changes −0.0005 while error improves −0.27 pp. Its fold outcomes are mixed. DC_SVC changes F1 by −0.0119 / −0.0085 and error by +1.29 / +1.15 pp (primary/H12); DC_WORD_CHAR changes F1 by −0.0030 in both views. Their nonzero error complementarity does not imply a stable route gain.",
        "",
        "## C2_B residual patterns and support",
        "",
        "The C2_B mean signed probability residual is `p − y`: positive is overprediction of the positive class; negative is underprediction. These are held-out descriptive residual patterns, not a selected correction. Confidence bins show a repeatable sign change: in .20–.40, mean residual is +3.72 pp (overprediction in five of six windows); in .60–.80, −3.75 pp (underprediction in five of six). Their fixed-cutoff errors remain 25.47% and 25.76%, respectively. This supports testing a nonlinear conditional residual head more than a global linear score recalibration, while the failed V2 calibration results remain an important warning.",
        "",
        "| Fit-defined subgroup | n pooled | C2_B error | Mean C2_B p−y | Residual sign windows (+/−) |",
        "|---|---:|---:|---:|---:|",
    ]
    summaries = audit["conditional_present_raw_CB_vs_C2B"]["pooled_stability_summary"]
    selected = []
    preferred = [("C2B_confidence", label) for label in ["0.20-0.40", "0.40-0.60", "0.60-0.80", "<0.20", ">=0.80"]]
    preferred += [("text_load", f"Q{i}") for i in range(1, 5)]
    preferred += [("scope_pressure", f"Q{i}") for i in range(1, 5)]
    preferred += [("any_unknown_category", label) for label in ["unseen_any", "no_unseen_category"]]
    for key in preferred:
        match = next((g for g in summaries if (g["axis"], g["group"]) == key), None)
        if match:
            selected.append(match)
    for g in selected:
        rows.append(f"| {g['axis']} / {g['group']} | {g['n_pooled']:,} | {pct(g['base_error_rate_pooled'])} | {100*g['base_mean_p_minus_y_pooled']:+.2f} pp | {g['windows_base_overpredicting_pooled_residual']}/{g['windows_base_underpredicting_pooled_residual']} |")
    rows += [
        "",
        "Broad marginal residual signs are weaker than the confidence pattern: text_load Q1 is −0.61 pp with three positive and three negative windows; Q2/Q3/Q4 are +1.75/+1.33/+1.15 pp, each positive in four windows. This is descriptive heterogeneity, not evidence that adding these covariates will improve predictions. `entry_friction` has tied/discrete quartile cutpoints, so its Q labels collapse to two occupied groups; do not read them as four balanced bins.",
        "",
        "## Ranked hypotheses for the next frozen pilot",
        "",
        "1. **Nonlinear OOF residual head with structured interactions (most justified).** Train only on past-only OOF rows; use current C2_B score plus expanded structured variables (including fit-derived category novelty and cycle as a candidate covariate) to model residual corrections. Include a fixed ablation without the structured CB score because its raw present-stream evidence is strongly negative. The opposite residual signs across C2_B confidence bands motivate nonlinearity; the near-neutral/mixed confidence×text_load cell and H12 reversals are counterexamples that require strict temporal guardrails. This is a hypothesis only, not fitted or validated here.",
        "2. **Sparse word-bigram TF-IDF LightGBM classifier and a pre-fixed small blend with C2_B.** This has a genuinely different feature-learning path from V2’s failed unigram and two-score calibration tests; this diagnostics lane did not score it and leaves it to the separately authorized main lane. Keep its share fixed before fitting and evaluate on the same temporal windows.",
        "3. **Cross-fitted reliability gate by confidence × text_load / category novelty.** Use only if a past-only gate learns a stable benefit on meta-training OOF blocks and is frozen before outer scoring. It ranks below the residual head because raw CB errors are worse in all broad bins, joint cells are mostly negative for CB, and several high-cardinality category cells are sparse. Do not select cells or thresholds from these outer labels.",
        "",
        "## Reproducibility and limitation",
        "",
        "Run from the repository root with `python scripts/optimization_v3_cls_diagnostics.py`. The script verifies cache row IDs/probability hashes, consumes only official Train and archived predictions, and writes `audit.json` plus an ignored run receipt containing input hashes. No model fitting occurs. All internal windows reuse a single chronology and have already informed prior work; the evidence is adaptive development, not a fresh untouched holdout. The detailed per-fold and subgroup denominator ledger is in `audit.json`.",
        "",
    ]
    return "\n".join(rows)


def main() -> None:
    train_path = task_files("classification")["train"]
    if train_path is None:
        raise FileNotFoundError("Official Classification Train.csv was not found")
    frame = pd.read_csv(train_path)
    required = {"cycle_code", "artifact_signal", "signal_stream", *NUMERIC_FEATURES, *CATEGORY_FEATURES}
    if not required.issubset(frame.columns):
        raise ValueError(f"Classification Train lacks expected diagnostic columns: {sorted(required - set(frame.columns))}")
    cycle = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=int)
    y = pd.to_numeric(frame["artifact_signal"], errors="raise").to_numpy(dtype=np.int8)
    if not np.isin(y, [0, 1]).all() or len(frame) != 50608 or (cycle.min(), cycle.max()) != (47, 79):
        raise ValueError("Official Train row count, target, or cycle span differs from frozen campaign expectations")
    missing = stream_missing_mask(frame).astype(bool)
    if not np.array_equal(missing, frame["signal_stream"].isna().to_numpy() |
                          (frame["signal_stream"].astype("string").str.strip().fillna("").str.len().to_numpy() == 0)):
        raise ValueError("Cached-routing stream missingness definition does not match the official Train values")

    input_records: dict[str, Any] = {
        "train": {"path": str(train_path.resolve()), "sha256": sha256(train_path),
                  "rows": len(frame), "target": "artifact_signal", "cycles": [47, 79]},
        "source_commit": __import__("subprocess").check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "source_files": {}, "prediction_artifacts": {},
    }
    source_paths = ["scripts/research_campaign_classification.py",
                    "scripts/research_campaign_classification_stage2.py",
                    "scripts/research_campaign_classification_catboost.py",
                    "src/models/classification_metrics.py",
                    "src/models/classification_round2_models.py",
                    "experiments/dual_campaign/classification/catboost_protocol.json",
                    "experiments/dual_campaign/classification/stage2_protocol.json"]
    for relative in source_paths:
        path = ROOT / relative
        if path.is_file():
            input_records["source_files"][relative] = sha256(path)

    all_fold_payload: dict[str, Any] = {}
    all_group_rows: list[dict[str, Any]] = []
    for window_name, (fit_end, start, end) in ALL_WINDOWS.items():
        rows = np.flatnonzero((cycle >= start) & (cycle <= end))
        if rows.size == 0:
            raise ValueError(f"No rows in evaluation window {window_name}")
        y_eval = y[rows]
        miss = missing[rows]
        present = ~miss

        row_inputs: dict[str, Any] = {}
        pred: dict[str, np.ndarray] = {}
        # Base experts and structured CatBoost all predict on the entire past-only
        # validation horizon. The CatBoost score is counterfactual on present rows;
        # the archived production route used it only for missing-stream rows.
        for model_id in ("C1_B", "C2_B"):
            source_rows, scores, identity = load_round1(fit_end, model_id, cycle)
            pos = np.searchsorted(source_rows, rows)
            if not np.array_equal(source_rows[pos], rows):
                raise ValueError(f"{model_id} does not cover {window_name}")
            pred[model_id] = scores[pos]
            row_inputs[f"{model_id}"] = identity
        cb_rows, cb_score, cb_identity = load_catboost(fit_end, cycle)
        pos = np.searchsorted(cb_rows, rows)
        if not np.array_equal(cb_rows[pos], rows):
            raise ValueError(f"CB_MISSING does not cover {window_name}")
        pred["CB_MISSING_raw"] = cb_score[pos]
        row_inputs["CB_MISSING_raw"] = cb_identity

        for model_id in ALT_MODELS:
            src_rows, scores, identity = load_candidate(fit_end, model_id, start, end, cycle)
            if not np.array_equal(src_rows, rows):
                raise ValueError(f"{model_id} does not align exactly to {window_name}")
            pred[model_id] = scores
            row_inputs[model_id] = identity

        pred["Champion_C2B_present_CB_missing"] = np.where(miss, pred["CB_MISSING_raw"], pred["C2_B"])
        pred["PreCB_C2B_present_C1B_missing"] = np.where(miss, pred["C1_B"], pred["C2_B"])

        group_defs: dict[str, np.ndarray] = {}
        quantile_edges: dict[str, list[float]] = {}
        fit_rows = np.flatnonzero((cycle >= 47) & (cycle <= fit_end))
        for feature in NUMERIC_FEATURES:
            labels, edges = fit_quantile_labels(frame.iloc[fit_rows][feature], frame.iloc[rows][feature])
            group_defs[feature] = labels
            quantile_edges[feature] = edges
        group_defs["C2B_confidence"] = np.select(
            [pred["C2_B"] < 0.20, pred["C2_B"] < 0.40, pred["C2_B"] < 0.60,
             pred["C2_B"] < 0.80], ["<0.20", "0.20-0.40", "0.40-0.60", "0.60-0.80"], default=">=0.80")
        known = {}
        for feature in CATEGORY_FEATURES:
            fit_values = frame.iloc[fit_rows][feature].dropna().astype(str)
            fit_vocab = set(fit_values.tolist())
            outer_values = frame.iloc[rows][feature]
            absent = outer_values.isna().to_numpy()
            unknown = (~absent) & ~outer_values.astype("string").isin(fit_vocab).to_numpy()
            group_defs[f"unknown_{feature}"] = np.where(absent, "missing", np.where(unknown, "unseen", "known"))
            known[feature] = {"fit_unique_nonmissing": int(len(fit_vocab)),
                              "outer_unseen_nonmissing": int(unknown.sum()),
                              "outer_missing": int(absent.sum())}
        any_unseen = np.zeros(rows.size, dtype=bool)
        for feature in CATEGORY_FEATURES:
            any_unseen |= np.asarray(group_defs[f"unknown_{feature}"] == "unseen")
        group_defs["any_unknown_category"] = np.where(any_unseen, "unseen_any", "no_unseen_category")

        primary_base = pred["C2_B"][present]
        # All reports are descriptive comparisons. The threshold is the archived
        # 0.50; no subgroup threshold or route is selected here.
        primary_window = window_name in PRIMARY_WINDOWS
        if primary_window:
            for axis, labels in group_defs.items():
                if axis.startswith("unknown_") or axis in {"any_unknown_category", "C2B_confidence"} or axis in NUMERIC_FEATURES:
                    all_group_rows.extend(grouped_comparisons(axis, labels[present], y_eval[present],
                                                              pred["C2_B"][present], pred["CB_MISSING_raw"][present],
                                                              window_name))
            for axis in ("text_load", "entry_friction"):
                qlabels = group_defs[axis][present]
                confidence = group_defs["C2B_confidence"][present]
                joint = np.asarray([f"{c}|{q}" for c, q in zip(confidence, qlabels)], dtype=object)
                all_group_rows.extend(grouped_comparisons(f"confidence_x_{axis}", joint,
                                                          y_eval[present], primary_base,
                                                          pred["CB_MISSING_raw"][present], window_name))

        fold_metrics: dict[str, Any] = {}
        for cohort, mask in (("all", np.ones(rows.size, dtype=bool)),
                             ("present", present), ("missing", miss)):
            fold_metrics[cohort] = {
                model: metrics(y_eval[mask], p[mask]) for model, p in pred.items()
            }
            fold_metrics[cohort]["raw_CB_vs_C2B"] = comparison(y_eval[mask], pred["C2_B"][mask], pred["CB_MISSING_raw"][mask])
            if cohort == "missing":
                fold_metrics[cohort]["CB_vs_C1B"] = comparison(y_eval[mask], pred["C1_B"][mask], pred["CB_MISSING_raw"][mask])
            if cohort == "present":
                for model in ALT_MODELS:
                    fold_metrics[cohort][f"{model}_vs_C2B"] = comparison(y_eval[mask], pred["C2_B"][mask], pred[model][mask])
                fold_metrics[cohort]["C1B_vs_C2B"] = comparison(y_eval[mask], pred["C2_B"][mask], pred["C1_B"][mask])
        cycles_table = []
        for c in range(start, end + 1):
            in_cycle = cycle[rows] == c
            in_present_cycle = in_cycle & present
            if in_present_cycle.any():
                cycles_table.append({"cycle": c, "present": int(in_present_cycle.sum()),
                                     "prevalence": float(y_eval[in_present_cycle].mean()),
                                     "C2B": metrics(y_eval[in_present_cycle], pred["C2_B"][in_present_cycle]),
                                     "raw_CB_counterfactual": metrics(y_eval[in_present_cycle], pred["CB_MISSING_raw"][in_present_cycle]),
                                     "raw_CB_vs_C2B": comparison(y_eval[in_present_cycle], pred["C2_B"][in_present_cycle], pred["CB_MISSING_raw"][in_present_cycle])})

        fold_obj = {"fit_end": fit_end, "evaluation_cycles": [start, end],
                    "n": int(rows.size), "present_n": int(present.sum()), "missing_n": int(miss.sum()),
                    "unknown_categories": known, "fit_defined_quantile_edges": quantile_edges,
                    "metrics": fold_metrics, "per_cycle_present": cycles_table,
                    "input_artifacts": row_inputs}
        all_fold_payload[window_name] = fold_obj

    group_summary = summarize_stable_groups(all_group_rows)
    eligible = [g for g in group_summary if g["cross_window_eligible"]]
    eligible.sort(key=lambda x: (abs(x["delta_error_rate_pooled"] or 0), x["n_pooled"]), reverse=True)
    main_summary: dict[str, Any] = {}
    for split, fold_names in (("primary_six", list(PRIMARY_WINDOWS)), ("H12", list(H12_WINDOWS))):
        main_summary[split] = {}
        for cohort in ("all", "present", "missing"):
            names = fold_names
            if split == "primary_six":
                rows_sel = [n for n in names]
            else:
                rows_sel = names
            # Concatenate non-overlapping evaluation rows within each split.
            ys, score_sets = [], {}
            for window in rows_sel:
                fit_end, start, end = ALL_WINDOWS[window]
                row_ids = np.flatnonzero((cycle >= start) & (cycle <= end))
                mask = np.ones(row_ids.size, dtype=bool)
                miss = missing[row_ids]
                if cohort == "present":
                    mask = ~miss
                elif cohort == "missing":
                    mask = miss
                ys.append(y[row_ids][mask])
                # Rehydrate metric probabilities from per-fold caches, preserving
                # the same route that was used for fold-level validation.
                fold = all_fold_payload[window]
                for model in ("C1_B", "C2_B", "CB_MISSING_raw", "Champion_C2B_present_CB_missing",
                              "PreCB_C2B_present_C1B_missing", *ALT_MODELS):
                    path = fold["input_artifacts"].get(model, {})
                    if model in ("C1_B", "C2_B"):
                        cache = ROOT / path["cache"]
                        with np.load(cache, allow_pickle=False) as z:
                            all_rows = z["outer_rows"].astype(np.int64, copy=False)
                            all_p = z["probability"].astype(float, copy=False)
                    elif model == "CB_MISSING_raw":
                        cache = ROOT / path["cache"]
                        with np.load(cache, allow_pickle=False) as z:
                            all_rows = z["rows"].astype(np.int64, copy=False)
                            all_p = z["candidate"].astype(float, copy=False)
                    elif model in ("Champion_C2B_present_CB_missing", "PreCB_C2B_present_C1B_missing"):
                        # Compose from the cached branch scores for this exact fold.
                        source_model = "CB_MISSING_raw" if model.startswith("Champion") else "C1_B"
                        source_path = ROOT / fold["input_artifacts"][source_model]["cache"]
                        with np.load(source_path, allow_pickle=False) as z:
                            all_rows = z["rows"].astype(np.int64, copy=False) if source_model == "CB_MISSING_raw" else z["outer_rows"].astype(np.int64, copy=False)
                            all_p = z["candidate"].astype(float, copy=False) if source_model == "CB_MISSING_raw" else z["probability"].astype(float, copy=False)
                        c2_path = ROOT / fold["input_artifacts"]["C2_B"]["cache"]
                        with np.load(c2_path, allow_pickle=False) as z:
                            c2_rows = z["outer_rows"].astype(np.int64, copy=False)
                            c2_p = z["probability"].astype(float, copy=False)
                        right = np.searchsorted(c2_rows, row_ids)
                        p_c2 = c2_p[right]
                    else:
                        cache = ROOT / path["cache"]
                        with np.load(cache, allow_pickle=False) as z:
                            all_rows = z["outer_rows"].astype(np.int64, copy=False)
                            all_p = z["candidate_probability"].astype(float, copy=False)
                    pos = np.searchsorted(all_rows, row_ids)
                    p_slice = all_p[pos]
                    if model.startswith("Champion"):
                        p_slice = np.where(miss, p_slice, p_c2)
                    elif model.startswith("PreCB"):
                        p_c1 = p_slice.copy()
                        p_slice = np.where(miss, p_c1, p_c2)
                    score_sets.setdefault(model, []).append(p_slice[mask])
            y_all = np.concatenate(ys)
            combined = {model: np.concatenate(pieces) for model, pieces in score_sets.items()}
            main_summary[split][cohort] = {model: metrics(y_all, p) for model, p in combined.items()}
            main_summary[split][cohort]["raw_CB_vs_C2B"] = comparison(y_all, combined["C2_B"], combined["CB_MISSING_raw"])
            if cohort == "missing":
                main_summary[split][cohort]["CB_vs_C1B"] = comparison(y_all, combined["C1_B"], combined["CB_MISSING_raw"])
            if cohort == "present":
                for model in ALT_MODELS:
                    main_summary[split][cohort][f"{model}_vs_C2B"] = comparison(y_all, combined["C2_B"], combined[model])
                main_summary[split][cohort]["C1B_vs_C2B"] = comparison(y_all, combined["C2_B"], combined["C1_B"])

    audit = {
        "title": "Cached Classification error and conditional residual diagnostics",
        "status": "DESCRIPTIVE_DIAGNOSTICS_ONLY_NO_NEW_FIT_NO_ROUTE_SELECTION",
        "source_boundary": "Official Classification Train only; no Test, Kaggle, or external data read.",
        "target": "artifact_signal", "threshold": 0.5,
        "champion_route": "C2_B on stream-present rows + CB_MISSING on stream-missing rows",
        "counterfactual_note": "Raw CB_MISSING probabilities are also examined on stream-present rows only as a diagnostic score; this is not the archived routing policy and is not a validated candidate route.",
        "support_policy": {"per_window_cell_minimum": MIN_CELL_N,
                           "pooled_cell_minimum": MIN_POOLED_CELL_N,
                           "minimum_supported_primary_windows": MIN_SUPPORTED_WINDOWS,
                           "primary_windows_are_nonoverlapping": True,
                           "numeric_quantiles": "Fit-fold empirical quartile boundaries from cycles 47..fit_end, then applied to the corresponding outer rows.",
                           "category_novelty": "Nonmissing outer value absent from that fold's cycles 47..fit_end vocabulary; missing kept separate.",
                           "confidence_bins": "Fixed C2_B score bins: <.20, .20-.40, .40-.60, .60-.80, >=.80."},
        "input_identity": input_records,
        "source_facts": {"rows": int(len(frame)), "stream_present_rows": int((~missing).sum()),
                         "stream_missing_rows": int(missing.sum()),
                         "positive_prevalence": float(y.mean()),
                         "cycle_counts": {str(int(c)): int((cycle == c).sum()) for c in sorted(np.unique(cycle))}},
        "pooled_metrics": main_summary,
        "folds": all_fold_payload,
        "conditional_present_raw_CB_vs_C2B": {"per_window_minimum_support": MIN_CELL_N,
                                                "per_window": all_group_rows,
                                                "pooled_stability_summary": group_summary,
                                                "cross_window_eligible_cells": eligible},
        "runtime": {"python": platform.python_version(),
                    "numpy": importlib.metadata.version("numpy"),
                    "pandas": importlib.metadata.version("pandas"),
                    "scikit_learn": importlib.metadata.version("scikit-learn")},
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    audit_path = REPORT_DIR / "audit.json"
    audit_bytes = json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8") + b"\n"
    audit_path.write_bytes(audit_bytes)
    findings_path = REPORT_DIR / "FINDINGS.md"
    findings_path.write_text(render_findings(audit), encoding="utf-8")
    receipt = {"status": audit["status"], "train_sha256": input_records["train"]["sha256"],
               "source_commit": input_records["source_commit"],
               "audit_path": str(audit_path.relative_to(ROOT)),
               "audit_sha256": hashlib.sha256(audit_bytes).hexdigest(),
               "findings_path": str(findings_path.relative_to(ROOT)),
               "findings_sha256": sha256(findings_path),
               "script_path": str(Path(__file__).resolve().relative_to(ROOT)),
               "script_sha256": sha256(Path(__file__).resolve()),
               "input_hash_manifest": input_records}
    receipt_path = RUN_DIR / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "audit": str(audit_path),
                      "receipt": str(receipt_path), "primary_present": main_summary["primary_six"]["present"],
                      "primary_missing": main_summary["primary_six"]["missing"],
                      "eligible_conditional_cells": len(eligible)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
