"""Raw/outer-error EDA for fit-vocabulary TF-IDF coverage; no model fitting."""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.impact_oov import vocabulary_coverage
from src.models.regression_research_text import RegressionTextModel, _text_documents
from src.models.regression_research_common import mapped_scores
from src.models.round1_evaluation import array_digest, code_digest, digest_json
from src.models.regression_round2_evaluation import positions_digest, load_protocol
from scripts.research_regression_lgb_text import PROTOCOL as STAGE5_PROTOCOL
from src.utils.paths import task_files

OUT = ROOT / "reports" / "impact_eda" / "oov"
ENDS = [55, 59, 63, 67, 71, 75]
COVERAGE_BANDS = ["coverage<=0.25", "0.25<coverage<=0.50", "0.50<coverage<=0.75", "coverage>0.75"]
MIN_GROUP_N = 20
MIN_TAIL_N = 5
LENGTH_QUARTILES = 4
PERIODS = {"discovery_56_67": (56, 67), "late_sensitivity_68_79": (68, 79)}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _quantiles(values):
    v = np.asarray(values, dtype=float)
    if not len(v):
        return {"q10": None, "q25": None, "median": None, "q75": None, "q90": None, "iqr": None}
    q10, q25, q50, q75, q90 = np.quantile(v, [.10, .25, .50, .75, .90])
    return {"q10": float(q10), "q25": float(q25), "median": float(q50),
            "q75": float(q75), "q90": float(q90), "iqr": float(q75 - q25)}


def _metrics(g: pd.DataFrame):
    y = g.threat_rank.to_numpy(float)
    ae = g.abs_error.to_numpy(float)
    return {
        "n": int(len(g)),
        "cycle_count": int(g.cycle_code.nunique()),
        "median_analyzer_terms": float(g.analyzer_terms.median()) if len(g) else None,
        "target": _quantiles(y),
        "target_low_le5_n": int((y <= 5).sum()),
        "target_low_le5_rate": float((y <= 5).mean()) if len(g) else None,
        "target_high_ge95_n": int((y >= 95).sum()),
        "target_high_ge95_rate": float((y >= 95).mean()) if len(g) else None,
        "abs_error_median": float(np.median(ae)) if len(g) else None,
        "abs_error_mae": float(np.mean(ae)) if len(g) else None,
        "signed_error_mean_prediction_minus_actual": float(g.signed_error.mean()) if len(g) else None,
    }


def _length_quartiles_by_cycle(g: pd.DataFrame):
    """Equal-count quartiles; resolve tied token counts with a target-blind ID hash."""
    out = pd.Series(index=g.index, dtype="Int64")
    for _, idx in g.groupby("cycle_code", sort=True).groups.items():
        sub = g.loc[idx]
        tie_hash = sub.record_id.map(lambda x: int.from_bytes(hashlib.sha256(str(x).encode("utf-8")).digest()[:8], "little"))
        order = np.lexsort((tie_hash.to_numpy(dtype=np.uint64), sub.analyzer_terms.to_numpy()))
        labels = np.minimum((np.arange(len(sub)) * LENGTH_QUARTILES) // max(1, len(sub)), LENGTH_QUARTILES - 1)
        assigned = np.empty(len(sub), dtype=int)
        assigned[order] = labels
        out.loc[idx] = assigned
    return out.astype(int)


def _fixed_band(x):
    if x <= .25: return COVERAGE_BANDS[0]
    if x <= .50: return COVERAGE_BANDS[1]
    if x <= .75: return COVERAGE_BANDS[2]
    return COVERAGE_BANDS[3]


def main():
    freeze = _read(ROOT / "experiments" / "regression_research" / "stage5_lgb_text_freeze.json")
    stage = _read(ROOT / "reports" / "regression_research" / "stage5_lgb_text.json")
    assert stage["freeze_signature"] == freeze["signature"]
    identity = freeze["identity"]
    protocol = load_protocol()
    assert identity["protocol"] == STAGE5_PROTOCOL
    for rel, expected in identity["code_sha256"].items():
        assert code_digest(ROOT / rel) == expected, ("frozen source mismatch", rel)
    data_path = task_files("regression")["train"]
    assert data_path is not None and _sha(data_path) == identity["regression_input_sha256"]["train"]
    frame = pd.read_csv(data_path)
    assert frame.shape == (50608, 19) and frame.record_id.is_unique
    assert set(frame.cycle_code.unique()) == set(range(47, 80))

    cache = ROOT / "experiments" / "runs" / "regression_research_v1"
    row_parts = []
    vocab_audits = {}
    prediction_audits = {}
    for end in ENDS:
        fit_rows = np.flatnonzero(frame.cycle_code.between(47, end))
        last = max(w["validation"][1] for w in protocol["primary_windows"] + protocol["secondary_windows"] if w["fit"][1] == end)
        outer_rows = np.flatnonzero(frame.cycle_code.between(end + 1, last))
        audit = stage["audits"][str(end)]
        selected = audit["selection"]
        outer_audit = audit["outer"]
        fit_path = cache / f"lgbtext__{selected['param']}__fit{end}__to{last}.npz"
        fit_meta_path = fit_path.with_suffix(".json")
        with np.load(fit_path, allow_pickle=False) as z:
            np.testing.assert_array_equal(z["fit_rows"], fit_rows)
            np.testing.assert_array_equal(z["validation_rows"], outer_rows)
            raw_prediction = z["prediction"].copy()
        fit_meta = _read(fit_meta_path)
        assert fit_meta["prediction_sha256"] == array_digest(raw_prediction)
        assert fit_meta["state_sha256"] == digest_json(fit_meta["state"])
        assert fit_meta["state"] == outer_audit["state"]
        assert outer_audit["prediction_sha256"] == array_digest(raw_prediction)
        assert outer_audit["state"]["validation_data_passed_to_fit"] is False
        assert outer_audit["state"]["test_data_used_for_fit"] is False
        assert outer_audit["state"]["fit_rows"] == len(fit_rows)

        docs, _ = _text_documents(frame.iloc[fit_rows]["signal_stream"])
        vectorizer, _, fallback = RegressionTextModel._fit_vectorizer(docs)
        rebuilt_tfidf = RegressionTextModel._text_state(vectorizer, fallback)
        assert rebuilt_tfidf == outer_audit["state"]["tfidf"], ("fit vocabulary/IDF mismatch", end)

        eval_path = cache / f"eval_lgbtext__fit{end}__inner.npz"
        with np.load(eval_path, allow_pickle=False) as z:
            np.testing.assert_array_equal(z["validation_rows"], outer_rows)
            mapped_prediction = z["prediction"].copy()
        expected_mapped = mapped_scores(raw_prediction, frame.iloc[outer_rows].cycle_code.to_numpy(), selected["rank_weight"])
        assert array_digest(mapped_prediction) == array_digest(expected_mapped), ("saved mapped prediction mismatch", end)
        prediction_audits[str(end)] = {
            "fit_rows": int(len(fit_rows)), "validation_rows": int(len(outer_rows)),
            "fit_rows_sha256": positions_digest(fit_rows), "validation_rows_sha256": positions_digest(outer_rows),
            "raw_prediction_sha256": array_digest(raw_prediction), "mapped_prediction_sha256": array_digest(mapped_prediction),
            "selected_param": selected["param"], "selected_rank_weight": selected["rank_weight"],
        }

        primary_w = next(w for w in protocol["primary_windows"] if w["fit"][1] == end)
        primary_mask = frame.iloc[outer_rows].cycle_code.between(*primary_w["validation"]).to_numpy()
        jobs = [j for j in stage["jobs"] if j["family"] == "primary" and j["window"] == primary_w["id"] and j["policy"] == "inner"]
        assert len(jobs) == 1 and jobs[0]["prediction_sha256"] == array_digest(mapped_prediction[primary_mask])

        coverage = vocabulary_coverage(vectorizer, frame.iloc[outer_rows]["signal_stream"])
        batch = frame.iloc[outer_rows][["record_id", "cycle_code", "signal_stream", "threat_rank"]].copy().reset_index(drop=True)
        coverage = coverage.reset_index(drop=True)
        batch["analyzer_terms"] = coverage["analyzer_terms"].to_numpy(int)
        batch["outside_vocabulary_terms"] = coverage["outside_vocabulary_terms"].to_numpy(int)
        batch["stream_missing"] = coverage["stream_missing"].to_numpy(bool)
        # Coverage is undefined when there is no analyzable token; do not assign these rows perfect coverage.
        batch["coverage"] = np.where(batch.analyzer_terms.gt(0), 1.0 - coverage.oov_fraction.to_numpy(float), np.nan)
        batch["oov_fraction"] = np.where(batch.analyzer_terms.gt(0), coverage.oov_fraction.to_numpy(float), np.nan)
        batch["prediction"] = mapped_prediction
        batch["signed_error"] = batch.prediction - batch.threat_rank
        batch["abs_error"] = batch.signed_error.abs()
        batch["fit_end"] = end
        batch["period"] = np.where(batch.cycle_code.between(*PERIODS["discovery_56_67"]), "discovery_56_67", "late_sensitivity_68_79")
        oov_vector_sha = array_digest(coverage.oov_fraction.to_numpy(dtype=np.float64))
        defined_oov_sha = array_digest(batch.loc[batch.coverage.notna(), "oov_fraction"].to_numpy(dtype=np.float64))
        # Keep one outer-model prediction per validation record in the EDA tables;
        # the full cutoff-specific coverage vectors above are separately hashed.
        batch = batch.loc[primary_mask].copy()
        row_parts.append(batch.drop(columns="signal_stream"))
        vocab_audits[str(end)] = {
            "fit_tfidf_state": rebuilt_tfidf,
            "fit_vocabulary_sha256": rebuilt_tfidf["vocabulary_sha256"],
            "fit_idf_sha256": rebuilt_tfidf["idf_sha256"],
            "fit_empty_fallback": fallback,
            "candidate_helper_oov_fraction_sha256_rows_in_outer_order": oov_vector_sha,
            "coverage_defined_present_oov_fraction_sha256": defined_oov_sha,
            "outer_rows_with_present_stream": int((~coverage.stream_missing).sum()),
            "outer_rows_with_defined_coverage": int((~coverage.stream_missing & coverage.analyzer_terms.gt(0)).sum()),
            "coverage_mean_defined": float(1-coverage.loc[~coverage.stream_missing & coverage.analyzer_terms.gt(0), "oov_fraction"].mean()),
        }

    rows = pd.concat(row_parts, ignore_index=True)
    assert rows.record_id.is_unique and len(rows) == 38588
    assert rows.cycle_code.min() == 56 and rows.cycle_code.max() == 79
    coverage_rows = rows.loc[~rows.stream_missing & rows.coverage.notna()].copy()
    coverage_rows["coverage_band"] = coverage_rows.coverage.map(_fixed_band)

    fixed = []
    for period in PERIODS:
        for band in COVERAGE_BANDS:
            g = coverage_rows.loc[coverage_rows.period.eq(period) & coverage_rows.coverage_band.eq(band)]
            fixed.append({"period": period, "coverage_band": band, **_metrics(g)})

    # For matched contrasts, split cycles into within-cycle analyzer-length quartiles,
    # then compare bottom/top OOV quartiles. The ID hash is only a deterministic tie-breaker.
    coverage_rows["length_quartile"] = _length_quartiles_by_cycle(coverage_rows)
    strata = []
    for (period, cycle, length_q), g in coverage_rows.groupby(["period", "cycle_code", "length_quartile"], sort=True):
        ranked_oov = g.oov_fraction.rank(method="average", pct=True)
        low = g.loc[ranked_oov.le(.25)]
        high = g.loc[ranked_oov.gt(.75)]
        base = {"period": period, "cycle": int(cycle), "length_quartile": int(length_q),
                "stratum_rows": int(len(g)), "low_oov_rows": int(len(low)), "high_oov_rows": int(len(high)),
                "low_oov_median_analyzer_terms": float(low.analyzer_terms.median()) if len(low) else None,
                "high_oov_median_analyzer_terms": float(high.analyzer_terms.median()) if len(high) else None,
                "low_oov_mean_oov_fraction": float(low.oov_fraction.mean()) if len(low) else None,
                "high_oov_mean_oov_fraction": float(high.oov_fraction.mean()) if len(high) else None}
        supported = len(low) >= MIN_GROUP_N and len(high) >= MIN_GROUP_N
        base["meets_minimum_support"] = supported
        if supported:
            l = low.threat_rank.to_numpy(float); h = high.threat_rank.to_numpy(float)
            la = low.abs_error.to_numpy(float); ha = high.abs_error.to_numpy(float)
            # Rank all rows in the matched stratum before comparing tails.
            all_abs = rankdata(g.abs_error.to_numpy(float), method="average") / len(g) * 100.0
            low_idx = g.index.get_indexer(low.index); high_idx = g.index.get_indexer(high.index)
            base.update({
                "low_oov_target_median": float(np.median(l)), "high_oov_target_median": float(np.median(h)),
                "high_minus_low_target_median": float(np.median(h) - np.median(l)),
                "low_oov_target_iqr": float(np.quantile(l, .75) - np.quantile(l, .25)),
                "high_oov_target_iqr": float(np.quantile(h, .75) - np.quantile(h, .25)),
                "high_minus_low_target_iqr": float(np.quantile(h, .75)-np.quantile(h, .25) - (np.quantile(l, .75)-np.quantile(l, .25))),
                "low_oov_target_low_le5_n": int((l <= 5).sum()), "high_oov_target_low_le5_n": int((h <= 5).sum()),
                "low_oov_target_high_ge95_n": int((l >= 95).sum()), "high_oov_target_high_ge95_n": int((h >= 95).sum()),
                "target_tail_rates_support_n_at_least_5": bool(min((l <= 5).sum(), (h <= 5).sum(), (l >= 95).sum(), (h >= 95).sum()) >= MIN_TAIL_N),
                "low_oov_target_low_le5_rate": float((l <= 5).mean()) if (l <= 5).sum() >= MIN_TAIL_N else None,
                "high_oov_target_low_le5_rate": float((h <= 5).mean()) if (h <= 5).sum() >= MIN_TAIL_N else None,
                "low_oov_target_high_ge95_rate": float((l >= 95).mean()) if (l >= 95).sum() >= MIN_TAIL_N else None,
                "high_oov_target_high_ge95_rate": float((h >= 95).mean()) if (h >= 95).sum() >= MIN_TAIL_N else None,
                "low_oov_abs_error_mae": float(la.mean()), "high_oov_abs_error_mae": float(ha.mean()),
                "high_minus_low_abs_error_mae": float(ha.mean() - la.mean()),
                "high_minus_low_within_stratum_abs_error_percentile": float(all_abs[high_idx].mean() - all_abs[low_idx].mean()),
                "high_minus_low_signed_error": float(high.signed_error.mean() - low.signed_error.mean()),
            })
            for key, vals in [("low_oov_abs_error_percentile", all_abs[low_idx]), ("high_oov_abs_error_percentile", all_abs[high_idx])]:
                base[key] = float(vals.mean())
        else:
            for key in ["low_oov_target_median", "high_oov_target_median", "high_minus_low_target_median",
                        "low_oov_target_iqr", "high_oov_target_iqr", "high_minus_low_target_iqr",
                        "low_oov_target_low_le5_n", "high_oov_target_low_le5_n", "low_oov_target_high_ge95_n",
                        "high_oov_target_high_ge95_n", "target_tail_rates_support_n_at_least_5",
                        "low_oov_target_low_le5_rate", "high_oov_target_low_le5_rate",
                        "low_oov_target_high_ge95_rate", "high_oov_target_high_ge95_rate",
                        "low_oov_abs_error_mae", "high_oov_abs_error_mae", "high_minus_low_abs_error_mae",
                        "high_minus_low_within_stratum_abs_error_percentile", "high_minus_low_signed_error",
                        "low_oov_abs_error_percentile", "high_oov_abs_error_percentile"]:
                base[key] = None
        strata.append(base)

    pooled = []
    strata_frame = pd.DataFrame(strata)
    for period in PERIODS:
        g = strata_frame.loc[strata_frame.period.eq(period) & strata_frame.meets_minimum_support]
        all_period = coverage_rows.loc[coverage_rows.period.eq(period)]
        eligible_rows = sum(int(x["low_oov_rows"] + x["high_oov_rows"]) for x in g.to_dict("records"))
        eligible_stratum_rows = int(g.stratum_rows.sum()) if len(g) else 0
        record = {"period": period, "all_coverage_defined_rows": int(len(all_period)),
                  "supported_strata": int(len(g)), "total_strata": int((strata_frame.period == period).sum()),
                  "rows_in_supported_strata": eligible_stratum_rows,
                  "low_high_comparison_denominator": eligible_rows,
                  "coverage_of_coverage_defined_rows_percent": float(100*eligible_stratum_rows/len(all_period)) if len(all_period) else None,
                  "low_oov_group_rows": int(g.low_oov_rows.sum()) if len(g) else 0,
                  "high_oov_group_rows": int(g.high_oov_rows.sum()) if len(g) else 0}
        for source, target in [("high_minus_low_target_iqr", "mean_high_minus_low_target_iqr"),
                               ("high_minus_low_target_median", "mean_high_minus_low_target_median"),
                               ("high_minus_low_abs_error_mae", "mean_high_minus_low_abs_error_mae"),
                               ("high_minus_low_within_stratum_abs_error_percentile", "mean_high_minus_low_abs_error_percentile")]:
            record[target] = float(np.average(g[source], weights=g.stratum_rows)) if len(g) else None
            record[target + "_positive_strata"] = int((g[source] > 0).sum()) if len(g) else 0
            record[target + "_negative_strata"] = int((g[source] < 0).sum()) if len(g) else 0
            record[target + "_zero_strata"] = int((g[source] == 0).sum()) if len(g) else 0
        record["supported_strata_with_both_tail_group_counts_ge5"] = int(g.target_tail_rates_support_n_at_least_5.sum()) if len(g) else 0
        pooled.append(record)

    fixed_df = pd.DataFrame(fixed)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), layout="constrained")
    colors = {"discovery_56_67": "#226d8f", "late_sensitivity_68_79": "#bd7132"}
    labels = {"discovery_56_67": "Cycles 56–67", "late_sensitivity_68_79": "Cycles 68–79"}
    xpos = np.arange(len(COVERAGE_BANDS))
    for period in PERIODS:
        p = fixed_df.loc[fixed_df.period.eq(period)].set_index("coverage_band").reindex(COVERAGE_BANDS)
        axes[0].plot(xpos, [p.loc[b, "target"]["iqr"] for b in COVERAGE_BANDS], marker="o", color=colors[period], label=labels[period])
        axes[1].plot(xpos, [p.loc[b, "abs_error_mae"] for b in COVERAGE_BANDS], marker="o", color=colors[period], label=labels[period])
    n_by_period = {period: fixed_df.loc[fixed_df.period.eq(period)].set_index("coverage_band")["n"] for period in PERIODS}
    band_labels = ["≤0.25", "0.25–0.50", "0.50–0.75", ">0.75"]
    xlabels = [f"{label}\n{int(n_by_period['discovery_56_67'][b])} / {int(n_by_period['late_sensitivity_68_79'][b])}"
               for label, b in zip(band_labels, COVERAGE_BANDS)]
    axes[0].set_ylabel("Observed threat_rank IQR")
    axes[1].set_ylabel("Frozen model absolute error MAE")
    for ax in axes:
        ax.set_xticks(xpos, xlabels, fontsize=8)
        ax.set_xlabel("Vocabulary coverage; n = cycles 56–67 / 68–79")
        ax.grid(axis="y", alpha=.22)
        ax.legend(frameon=False)
    axes[0].set_title("Raw target heterogeneity")
    axes[1].set_title("Retrospective outer error")
    fig.suptitle("Fit-vocabulary coverage by period (present streams with ≥1 analyzer term)")
    image_path = OUT / "coverage_target_error.png"
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(image_path, dpi=160)
    plt.close(fig)

    # Keep the required outputs to one stats file, one short note, and one plot.
    stats = {
        "status": "COMPLETE_RAW_EDA_AND_RETROSPECTIVE_ERROR_DIAGNOSTIC",
        "research_question": "Pada panjang laporan dan cycle sebanding, apakah rendahnya cakupan kosakata TF-IDF historis berkaitan dengan heterogenitas threat_rank dan error model?",
        "definition": "coverage = 1 - fraction of unigram/bigram analyzer occurrences absent from fold-fit TF-IDF feature vocabulary; outside includes min_df/max_features filtering and is not semantic novelty",
        "periods": {k: list(v) for k,v in PERIODS.items()},
        "coverage_bands_fixed_before_summary": COVERAGE_BANDS,
        "minimum_support": {"matched_group_n_each_low_high_oov": MIN_GROUP_N, "target_tail_count_each_comparison_group": MIN_TAIL_N,
                            "length_strata": "four equal-count quartiles within cycle, ties broken deterministically by SHA256(record_id); outcome-blind"},
        "source": {"train_path_role": "official regression train only", "train_sha256": _sha(data_path), "train_shape": list(frame.shape),
                   "stage5_freeze_signature": freeze["signature"], "stage5_result_freeze_signature": stage["freeze_signature"],
                   "round2_signature": identity["round2_signature"], "python_packages": identity["packages"]},
        "prediction_provenance": prediction_audits,
        "vocabulary_reconciliation": vocab_audits,
        "coverage_denominators": {"primary_rows": int(len(rows)), "coverage_defined_present_rows": int(len(coverage_rows)),
                                   "stream_missing_rows": int(rows.stream_missing.sum()),
                                   "present_but_no_analyzer_terms_rows": int((~rows.stream_missing & rows.analyzer_terms.eq(0)).sum())},
        "fixed_coverage_bands": fixed,
        "matched_within_cycle_length_strata": strata,
        "matched_period_summary": pooled,
        "limitations": ["All official Train cycles were already used in model development; no independent holdout.",
                        "Vocabulary coverage is relative to the fit TF-IDF feature vocabulary, including min_df/max_features filters; it says nothing about semantic novelty.",
                        "Raw target association and outer residual association are separate descriptive analyses; neither is causal.",
                        "H12 and primary reuse records; period comparison is sensitivity evidence, not independent replication.",
                        "No EDA-derived bin or threshold is used for feature selection; only the one prespecified scalar ablation may be evaluated by MAIN.",
                        "Official Test, Kaggle and submissions were not read or used."],
        "outputs": {"figure": image_path.name, "figure_sha256": _sha(image_path)},
        "research_script_sha256": _sha(Path(__file__)),
    }
    _write(OUT / "stats.json", stats)
    paired = {x["period"]: x for x in pooled}
    early, late = paired["discovery_56_67"], paired["late_sensitivity_68_79"]
    note = f"""# Fit-vocabulary coverage: raw target and outer-error EDA

**Question.** {stats['research_question']}

## Definition and provenance

Coverage is the fraction of TF-IDF analyzer unigram/bigram occurrences that appear in the model's fit-only feature vocabulary. OOV terms include terms filtered by `min_df` or the `max_features` cap; this is a model-vocabulary measure, not semantic novelty. The exact fitted vocabulary and IDF were rebuilt from each fold's training `signal_stream` and both hashes matched the frozen Stage 5 fit audit. The saved outer predictions, row positions, rank weights and Stage 5 job hashes were reconciled before errors were calculated. Only official Regression Train was used (SHA-256 `{stats['source']['train_sha256']}`); no Test data, submission score or new model fits were used.

## Keep the two questions distinct

The first table summarizes the observed `threat_rank` distribution by fixed coverage bands. A lower median or narrower/wider IQR is a raw-data association. It does not say vocabulary coverage causes the target distribution. The second table compares historical frozen outer-model error over the same groups; it is retrospective model-error evidence, not an EDA available before model review. Cycles 56–67 are the discovery narrative and 68–79 are a sensitivity period. All cycles have appeared in earlier development and the later period is not a holdout.

For the length/cycle comparison, each cycle was divided into four equal-count analyzer-token-length strata. Within each stratum, the lower and upper OOV quartiles were compared. A contrast is retained only if each extreme-OOV group has at least {MIN_GROUP_N} rows; tail rates require at least {MIN_TAIL_N} examples per comparison group. `stats.json` gives each matched cell, its denominator, eligible-row coverage, target IQR/tail counts, and error MAE/within-cell percentile contrasts. Any pooled number is the support-weighted average of these cells; inspect the supported-cell counts and coverage beside it.

## Observed pattern

In cycles 56–67, {early['supported_strata']}/{early['total_strata']} matched cycle × length cells meet the minimum (coverage {early['coverage_of_coverage_defined_rows_percent']:.1f}% of defined present-stream rows). The high-OOV group has a weighted within-cell target-IQR difference of +{early['mean_high_minus_low_target_iqr']:.2f} target points ({early['mean_high_minus_low_target_iqr_positive_strata']}/{early['supported_strata']} cells positive), and a +{early['mean_high_minus_low_abs_error_mae']:.2f} error-MAE difference ({early['mean_high_minus_low_abs_error_mae_positive_strata']}/{early['supported_strata']} positive). Its mean absolute-error percentile is +{early['mean_high_minus_low_abs_error_percentile']:.2f} points higher ({early['mean_high_minus_low_abs_error_percentile_positive_strata']}/{early['supported_strata']} positive). This supports testing a coverage scalar, not an assumed performance gain.

In cycles 68–79, corresponding differences are +{late['mean_high_minus_low_target_iqr']:.2f} target-IQR points ({late['mean_high_minus_low_target_iqr_positive_strata']}/{late['supported_strata']} positive), +{late['mean_high_minus_low_abs_error_mae']:.2f} absolute-error MAE ({late['mean_high_minus_low_abs_error_mae_positive_strata']}/{late['supported_strata']} positive), and +{late['mean_high_minus_low_abs_error_percentile']:.2f} within-cell error-percentile points ({late['mean_high_minus_low_abs_error_percentile_positive_strata']}/{late['supported_strata']} positive). This is a useful sensitivity pattern, but later cycles have also featured repeatedly in development.

Tail evidence is sparse after length/cycle matching: at least five examples in each low/high OOV group for both target tails occur in only {early['supported_strata_with_both_tail_group_counts_ge5']}/{early['supported_strata']} discovery cells and {late['supported_strata_with_both_tail_group_counts_ge5']}/{late['supported_strata']} sensitivity cells. Do not infer a tail-specific mechanism from these matched cells. The fixed low-coverage band (coverage ≤0.25) contains just {next(x['n'] for x in fixed if x['period']=='discovery_56_67' and x['coverage_band']==COVERAGE_BANDS[0])} discovery rows and {next(x['n'] for x in fixed if x['period']=='late_sensitivity_68_79' and x['coverage_band']==COVERAGE_BANDS[0])} sensitivity rows, so its displayed raw bin is descriptive only.

The figure is a compact unadjusted descriptive view. The matched contrasts, not a visual slope across all coverage bands, address whether target or error differences persist within cycle and comparable text length. Residual analysis still uses the same outer validation labels that informed earlier experiments, so adaptive selection bias remains.

## Interpretation boundary

This analysis only checks whether the planned scalar is worth its single controlled ablation. It does not establish causal effect, generalization to cycle 80–91, meaning of anonymized text, or an untouched test result. It does not investigate EF, impact, scope, telemetry, module or organization spread. No target label enters the vocabulary, analyzer, coverage calculation, or length strata.
"""
    (OUT / "NOTES.md").write_text(note, encoding="utf-8")
    print(json.dumps({"status": stats["status"], "coverage_rows": stats["coverage_denominators"],
                      "matched_period_summary": pooled, "out": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
