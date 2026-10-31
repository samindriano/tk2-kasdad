"""Independent aggregate-only text EDA for Samuel's candidate discovery."""
from __future__ import annotations
import argparse, hashlib, json, os, re, sys
from collections import Counter
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.paths import task_files

EXPECTED = {
    "reg_train": "756523ff8156a9cd776e52249f7e2a8f23dcec432b8d10570e55ff6db0e0f082",
    "cls_train": "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c",
}
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def token_stats(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return (0, 0, 0., 0., 0.)
    tokens = TOKEN_RE.findall(str(value).lower())
    n = len(tokens)
    if not n:
        return (0, 0, 0., 0., 0.)
    counts = np.fromiter(Counter(tokens).values(), dtype=np.int64)
    p = counts / n
    return (n, len(counts), len(counts)/n, float(-(p*np.log(p)).sum()),
            float(counts[counts > 1].sum()/n))

def cycle_bootstrap(values, seed=42, reps=10000):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 8:
        return {"cycles": len(values), "p05": None, "p50": None, "p95": None}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), (reps, len(values)))
    means = values[draws].mean(axis=1)
    return {"cycles": len(values), "p05": float(np.quantile(means, .05)),
            "p50": float(np.quantile(means, .5)), "p95": float(np.quantile(means, .95)),
            "interpretation": "Exploratory cycle-cluster sensitivity if cycles are exchangeable; not a confirmatory CI."}

def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    return value

def main(root=None, output_dir=None):
    root = Path(root or PROJECT_ROOT).resolve()
    output_dir = Path(output_dir or root/"reports/eda_candidates/text").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_root = root/"data"/"raw"
    configured_root = os.environ.get("TP2_DATA_DIR")
    data_override = Path(configured_root).expanduser().resolve() if configured_root else (
        raw_root if raw_root.exists() else None)
    cf, rf = task_files("classification", data_override), task_files("regression", data_override)
    assert cf["train"] and rf["train"]
    hashes = {"reg_train": sha(rf["train"]), "cls_train": sha(cf["train"])}
    assert hashes == EXPECTED
    reg, cls = pd.read_csv(rf["train"]), pd.read_csv(cf["train"])
    assert reg.shape == cls.shape == (50608, 19)
    assert reg.record_id.is_unique and cls.record_id.is_unique
    assert set(reg.record_id) == set(cls.record_id)
    common = [c for c in reg.columns if c in cls.columns and c not in {"threat_rank", "artifact_signal"}]
    left = reg.sort_values("record_id").reset_index(drop=True)
    right = cls.sort_values("record_id").reset_index(drop=True)
    assert left.record_id.equals(right.record_id)
    for name in common:
        a, b = left[name], right[name]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            assert np.array_equal(a.fillna(-999999).to_numpy(float), b.fillna(-999999).to_numpy(float))
        else:
            assert a.fillna("<NULL>").astype(str).equals(b.fillna("<NULL>").astype(str))
    data = left[common + ["threat_rank"]].merge(
        right[["record_id", "artifact_signal"]], on="record_id", validate="one_to_one")
    assert len(data) == 50608 and data.artifact_signal.isin([0,1]).all()
    assert data.threat_rank.between(0,100).all()
    # Remove identifier after the verified one-to-one target join.
    data = data.drop(columns="record_id")
    raw = left.signal_stream
    missing = raw.isna() | raw.astype("string").str.strip().eq("").fillna(False)
    data["stream_missing"] = missing.to_numpy()
    tf = pd.DataFrame(raw.map(token_stats).tolist(), columns=[
        "token_count", "unique_tokens", "type_token_ratio", "shannon_entropy", "repeated_token_share"])
    data = pd.concat([data.reset_index(drop=True), tf], axis=1)
    data["text_length_chars"] = raw.fillna("").astype(str).str.len().to_numpy()
    present = data.loc[~data.stream_missing].copy()

    # Label-blind token-count quintiles define comparable stream-length bands.
    present["length_quintile"] = pd.qcut(present.token_count, q=5, duplicates="drop")
    length_edges = [float(present.token_count.quantile(q)) for q in np.linspace(0,1,6)]
    rows = []
    for band, g in present.groupby("length_quintile", observed=True, sort=True):
        rank = g.type_token_ratio.rank(method="average", pct=True)
        low, high = g.loc[rank <= .25], g.loc[rank >= .75]
        row={
            "length_bin": str(band), "n": len(g), "token_p10": float(g.token_count.quantile(.1)),
            "token_p90": float(g.token_count.quantile(.9)), "low_diversity_n": len(low),
            "high_diversity_n": len(high), "low_unique_fraction": float(low.type_token_ratio.mean()),
            "high_unique_fraction": float(high.type_token_ratio.mean()) if len(high) else None,
            "rho_diversity_threat": float(spearmanr(g.type_token_ratio,g.threat_rank).statistic),
            "rho_diversity_artifact": float(spearmanr(g.type_token_ratio,g.artifact_signal).statistic)}
        if len(low) and len(high):
            row.update({"threat_mean_low":float(low.threat_rank.mean()),"threat_mean_high":float(high.threat_rank.mean()),
                "threat_median_low":float(low.threat_rank.median()),"threat_median_high":float(high.threat_rank.median()),
                "threat_delta_high_minus_low":float(high.threat_rank.mean()-low.threat_rank.mean()),
                "artifact_rate_low":float(low.artifact_signal.mean()),"artifact_rate_high":float(high.artifact_signal.mean()),
                "artifact_delta_high_minus_low":float(high.artifact_signal.mean()-low.artifact_signal.mean())})
        else:
            row.update({"threat_mean_low":float(low.threat_rank.mean()) if len(low) else None,
                "threat_mean_high":None,"threat_median_low":float(low.threat_rank.median()) if len(low) else None,
                "threat_median_high":None,"threat_delta_high_minus_low":None,
                "artifact_rate_low":float(low.artifact_signal.mean()) if len(low) else None,
                "artifact_rate_high":None,"artifact_delta_high_minus_low":None})
        rows.append(row)
    diversity = pd.DataFrame(rows)

    # Within each cycle, compare lower/upper diversity quartiles inside each global length band.
    cycle_rows = []
    for cycle, g in present.groupby("cycle_code", sort=True):
        contrasts = []
        for _, h in g.groupby("length_quintile", observed=True, sort=True):
            rk = h.type_token_ratio.rank(method="average", pct=True)
            lo, hi = h.loc[rk <= .25], h.loc[rk >= .75]
            if len(lo) >= 30 and len(hi) >= 30:
                contrasts.append((len(lo)+len(hi), hi.threat_rank.mean()-lo.threat_rank.mean(),
                                  hi.artifact_signal.mean()-lo.artifact_signal.mean()))
        if contrasts:
            w = np.asarray([v[0] for v in contrasts], float); w /= w.sum()
            cycle_rows.append({"cycle": int(cycle), "n_present": len(g),
                "threat_delta": float(np.dot(w,[v[1] for v in contrasts])),
                "artifact_delta": float(np.dot(w,[v[2] for v in contrasts]))})
    by_cycle = pd.DataFrame(cycle_rows)
    supported_bins=diversity.dropna(subset=["threat_delta_high_minus_low"]).copy()
    pooled_threat = float(np.average(supported_bins.threat_delta_high_minus_low, weights=supported_bins.n))
    pooled_artifact = float(np.average(supported_bins.artifact_delta_high_minus_low, weights=supported_bins.n))
    loo = []
    for omitted in present.cycle_code.unique():
        sub = present.loc[present.cycle_code != omitted]
        contrasts=[]
        for _, g in sub.groupby("length_quintile", observed=True, sort=True):
            rk=g.type_token_ratio.rank(method="average",pct=True); lo,hi=g.loc[rk<=.25],g.loc[rk>=.75]
            if len(lo) and len(hi):
                contrasts.append((len(g),hi.threat_rank.mean()-lo.threat_rank.mean(),
                                 hi.artifact_signal.mean()-lo.artifact_signal.mean()))
        w=np.asarray([v[0] for v in contrasts],float);w/=w.sum()
        loo.append([float(np.dot(w,[v[1] for v in contrasts])),float(np.dot(w,[v[2] for v in contrasts]))])

    def exact_length_contrast(frame, feature, higher_is_diverse=True, minimum_rows=120, minimum_tail=20):
        records=[]
        for token_count,g in frame.groupby("token_count",sort=True):
            if len(g)<minimum_rows or g[feature].nunique()<2: continue
            rank=g[feature].rank(method="average",pct=True)
            if higher_is_diverse: lo,hi=g.loc[rank<=.25],g.loc[rank>=.75]
            else: lo,hi=g.loc[rank>=.75],g.loc[rank<=.25]
            if len(lo)>=minimum_tail and len(hi)>=minimum_tail:
                records.append({"token_count":int(token_count),"n":len(g),"low_n":len(lo),"high_n":len(hi),
                    "threat_mean_low":float(lo.threat_rank.mean()),"threat_mean_high":float(hi.threat_rank.mean()),
                    "artifact_rate_low":float(lo.artifact_signal.mean()),"artifact_rate_high":float(hi.artifact_signal.mean()),
                    "threat_delta":float(hi.threat_rank.mean()-lo.threat_rank.mean()),
                    "artifact_delta":float(hi.artifact_signal.mean()-lo.artifact_signal.mean())})
        result=pd.DataFrame(records)
        return result
    exact_feature_frames={
        "type_token_ratio":exact_length_contrast(present,"type_token_ratio",True),
        "shannon_entropy":exact_length_contrast(present,"shannon_entropy",True),
        "repeated_token_share":exact_length_contrast(present,"repeated_token_share",False)}
    exact_rows=[]
    for feature, frame in exact_feature_frames.items():
        if len(frame):
            frame=frame.copy()
            frame["support_weight"]=frame.n/frame.n.sum()
            frame["weighted_artifact_gap_contribution"]=frame.support_weight*frame.artifact_delta
            frame["weighted_threat_gap_contribution"]=frame.support_weight*frame.threat_delta
            exact_feature_frames[feature]=frame
        exact_rows.append({"feature":feature,"supported_token_counts":int(len(frame)),
            "support_rows":int(frame.n.sum()),"share_of_stream_present":float(frame.n.sum()/len(present)),
            "share_of_all_train":float(frame.n.sum()/len(data)),
            "low_tail_compared_rows":int(frame.low_n.sum()),"high_tail_compared_rows":int(frame.high_n.sum()),
            "average_low_tail_artifact_rate":float(np.average(frame.artifact_rate_low,weights=frame.n)),
            "average_high_tail_artifact_rate":float(np.average(frame.artifact_rate_high,weights=frame.n)),
            "positive_token_count_strata":int((frame.artifact_delta>0).sum()),
            "share_supported_rows_from_positive_strata":float(frame.loc[frame.artifact_delta>0,"n"].sum()/frame.n.sum()),
            "upper_vs_lower_diversity_threat_delta":float(np.average(frame.threat_delta,weights=frame.n)),
            "upper_vs_lower_diversity_artifact_delta":float(np.average(frame.artifact_delta,weights=frame.n)),
            "threat_delta_minmax":[float(frame.threat_delta.min()),float(frame.threat_delta.max())],
            "artifact_delta_minmax":[float(frame.artifact_delta.min()),float(frame.artifact_delta.max())]})
    exact_summary={"minimum_rows_per_exact_count":120,"minimum_each_diversity_tail":20,
        "rank_tail_rule":"same as length-quintile analysis: average rank <= 0.25 versus >= 0.75",
        "features":exact_rows}
    threshold_sensitivity=[]
    for min_n in [80,120,200]:
        strata=exact_length_contrast(present,"type_token_ratio",True,minimum_rows=min_n,minimum_tail=20)
        if len(strata):
            pooled_low=float(np.average(strata.artifact_rate_low,weights=strata.low_n))
            pooled_high=float(np.average(strata.artifact_rate_high,weights=strata.high_n))
            weighted_delta=float(np.average(strata.artifact_delta,weights=strata.n))
            threshold_sensitivity.append({"min_rows_per_exact_token_count":min_n,
                "supported_token_counts":len(strata),"support_rows":int(strata.n.sum()),
                "share_present":float(strata.n.sum()/len(present)),
                "compared_lower_tail_n":int(strata.low_n.sum()),"compared_upper_tail_n":int(strata.high_n.sum()),
                "pooled_lower_tail_artifact_rate":pooled_low,"pooled_upper_tail_artifact_rate":pooled_high,
                "pooled_tail_delta_high_minus_low":pooled_high-pooled_low,
                "support_weighted_stratum_delta":weighted_delta,
                "positive_token_count_strata":int((strata.artifact_delta>0).sum()),
                "share_support_from_positive_strata":float(strata.loc[strata.artifact_delta>0,"n"].sum()/strata.n.sum())})
    exact_summary["support_threshold_sensitivity"]=threshold_sensitivity
    # Add cycle as a second exact stratum check to identify a time-composition artifact.
    exact_cycle=[]
    for cycle,gcycle in present.groupby("cycle_code"):
        contrast=exact_length_contrast(gcycle,"type_token_ratio",True,minimum_rows=60,minimum_tail=10)
        if len(contrast):
            exact_cycle.append({"cycle":int(cycle),"supported_token_counts":len(contrast),
                "support_rows":int(contrast.n.sum()),
                "threat_delta":float(np.average(contrast.threat_delta,weights=contrast.n)),
                "artifact_delta":float(np.average(contrast.artifact_delta,weights=contrast.n))})
    exact_cycle_frame=pd.DataFrame(exact_cycle)
    exact_summary["cycle_and_exact_length_sensitivity"]={
        "cycles_supported":int(len(exact_cycle_frame)),
        "per_cycle":exact_cycle,
        "cycle_bootstrap_threat":cycle_bootstrap(exact_cycle_frame.threat_delta),
        "cycle_bootstrap_artifact":cycle_bootstrap(exact_cycle_frame.artifact_delta)}

    # Does numeric text_load represent realized lexical/character length?
    present["text_load_quartile"] = pd.qcut(present.text_load, 4, duplicates="drop")
    qtable=present.groupby("text_load_quartile",observed=True).agg(
        n=("token_count","size"),median_tokens=("token_count","median"),
        median_characters=("text_length_chars","median"),median_diversity=("type_token_ratio","median"),
        repeated_token_share=("repeated_token_share","median"),threat_median=("threat_rank","median"),
        artifact_rate=("artifact_signal","mean")).reset_index()
    qtable["text_load_quartile"]=qtable.text_load_quartile.astype(str)
    diversity["length_bin"]=diversity.length_bin.astype(str)

    # Raw-token outcomes are never emitted; figures contain only aggregate numeric values.
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"figure.facecolor":"white"})
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    ax[0].scatter(present.text_load,present.token_count,s=5,alpha=.15,rasterized=True)
    ax[0].set(title="text_load and realized token count",xlabel="Supplied text_load",ylabel="Regex tokens per stream")
    ax[1].scatter(present.token_count,present.type_token_ratio,s=5,alpha=.15,rasterized=True)
    ax[1].set(title="Lexical diversity changes with length",xlabel="Regex tokens per stream",ylabel="Unique token fraction",ylim=(0,1.03))
    fig.tight_layout();fig.savefig(output_dir/"text_length_coherence.png",dpi=150,bbox_inches="tight");plt.close(fig)
    fig,ax=plt.subplots(1,3,figsize=(13,4))
    ax[0].plot(diversity.length_bin,diversity.threat_mean_low,"o-",label="Lower diversity")
    ax[0].plot(diversity.length_bin,diversity.threat_mean_high,"s--",label="Higher diversity")
    ax[0].set(title="Threat within stream-length bands",ylabel="Mean threat_rank",xlabel="Token-count quintile")
    ax[1].plot(diversity.length_bin,diversity.artifact_rate_low,"o-",label="Lower diversity")
    ax[1].plot(diversity.length_bin,diversity.artifact_rate_high,"s--",label="Higher diversity")
    ax[1].set(title="Artifact rate within length bands",ylabel="Share artifact_signal = 1",xlabel="Token-count quintile")
    ax[2].plot(by_cycle.cycle,by_cycle.artifact_delta,"o-",ms=3,label="Artifact-rate difference")
    ax[2].axhline(0,color="black",lw=1)
    ax[2].set(title="Cycle sensitivity",xlabel="Cycle",ylabel="Higher − lower diversity (share)")
    for a in ax: a.tick_params(axis="x",rotation=25);a.grid(alpha=.2)
    ax[0].legend(fontsize=8);ax[1].legend(fontsize=8);ax[2].legend(fontsize=8)
    fig.tight_layout();fig.savefig(output_dir/"diversity_by_length_and_targets.png",dpi=150,bbox_inches="tight");plt.close(fig)

    metrics={
      "source":{"hashes":hashes,"regression_shape":list(reg.shape),"classification_shape":list(cls.shape),
        "n_joined":len(data),"shared_features_equal_by_record_id":common,
        "train_cycles":[int(data.cycle_code.min()),int(data.cycle_code.max())]},
      "token_definition":{"regex":"[A-Za-z0-9]+","lowercase":True,
        "unique_fraction":"distinct regex tokens / tokens in row",
        "repeated_token_share":"tokens belonging to types repeated within the row / tokens",
        "entropy":"Shannon entropy of within-row token frequencies","raw_tokens_exported":False},
      "availability":{"missing_or_blank_n":int(data.stream_missing.sum()),
        "missing_or_blank_share":float(data.stream_missing.mean()),"present_n":len(present),
        "all_train_artifact_prevalence":float(data.artifact_signal.mean()),
        "all_train_threat_median":float(data.threat_rank.median()),
        "present_zero_token_n":int((present.token_count==0).sum()),
        "missing_artifact_rate":float(data.loc[data.stream_missing].artifact_signal.mean()),
        "present_artifact_rate":float(data.loc[~data.stream_missing].artifact_signal.mean()),
        "missing_threat_median":float(data.loc[data.stream_missing].threat_rank.median()),
        "present_threat_median":float(data.loc[~data.stream_missing].threat_rank.median())},
      "text_diversity":{"label_blind_length_quintile_edges":length_edges,
        "per_length_quintile":diversity.to_dict(orient="records"),
        "bins_with_variable_diversity":int(len(supported_bins)),
        "high_minus_low_threat_mean":pooled_threat,"high_minus_low_artifact_rate":pooled_artifact,
        "exact_token_count_sensitivity":exact_summary,
        "per_cycle":by_cycle.to_dict(orient="records"),
        "cycle_bootstrap_threat":cycle_bootstrap(by_cycle.threat_delta),
        "cycle_bootstrap_artifact":cycle_bootstrap(by_cycle.artifact_delta),
        "leave_one_cycle_out_threat_range":[float(np.min(np.asarray(loo)[:,0])),float(np.max(np.asarray(loo)[:,0]))],
        "leave_one_cycle_out_artifact_range":[float(np.min(np.asarray(loo)[:,1])),float(np.max(np.asarray(loo)[:,1]))]},
      "text_load_coherence":{"n_present":len(present),
        "spearman_text_load_tokens":float(spearmanr(present.text_load,present.token_count).statistic),
        "spearman_text_load_characters":float(spearmanr(present.text_load,present.text_length_chars).statistic),
        "text_load_quantiles_all_train":{str(q):float(data.text_load.quantile(q)) for q in [0,.25,.5,.75,1]},
        "text_load_quantiles_present_stream":{str(q):float(present.text_load.quantile(q)) for q in [0,.25,.5,.75,1]},
        "token_quantiles_present_stream":{str(q):float(present.token_count.quantile(q)) for q in [0,.25,.5,.75,.9,.95,.99,1]},
        "quartiles":qtable.to_dict(orient="records")},
      "non_overlap":{"Rousan":"Neither entry_friction nor impact_load is used to group or explain artifact_signal; feature focus is stream structure.",
        "Abhi":"Neither scope_pressure nor impact_load enters the question or grouping.",
        "Rizky":"Neither telemetry_flag nor module_spread enters grouping; text is the object studied.",
        "Samuel_old":"No impact_load/threat_rank temporal association is retested; cycle is only a support sensitivity check.",
        "shared_outcome":"artifact_signal overlaps as outcome only; core explanatory variables and question differ."},
      "limitations":["Exploratory on all labeled Train; no independent confirmatory sample.",
        "Regex token richness measures structure, not semantics of anonymized technical tokens.",
        "Broad length quintiles leave residual confounding by exact length, cycle and source/report style.",
        "Cycle bootstrap assumes exchangeability for sensitivity and cannot project future cycles.",
        "No fitted model, actual tokens, causal claim or Test label was used."]}
    metrics = json_safe(metrics)
    (output_dir/"stats.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    diversity.to_csv(output_dir/"length_stratified_diversity.csv",index=False)
    by_cycle.to_csv(output_dir/"cycle_sensitivity.csv",index=False)
    qtable.to_csv(output_dir/"text_load_quartiles.csv",index=False)
    for feature, frame in exact_feature_frames.items():
        frame.to_csv(output_dir/f"exact_length_{feature}.csv",index=False)
    exact_ttr, exact_entropy, exact_repeat=exact_rows
    cy_art=by_cycle.artifact_delta.to_numpy(float)
    textload=metrics["text_load_coherence"]
    per_bin=diversity.dropna(subset=["artifact_delta_high_minus_low"])
    support_sens=exact_summary["support_threshold_sensitivity"]
    note=f"""# Text EDA candidate evidence

Inputs: Regression Train SHA256 {hashes['reg_train']}; Classification Train
SHA256 {hashes['cls_train']}. Both contain 50,608 unique IDs; all shared
feature columns were compared by ID before joining separate labels. This analysis
reads only the two official Train files. No raw token, identifier, model prediction,
or Kaggle score is exported.

## Candidate T1 — lexical variety at similar text length

**Question:** Pada stream dengan jumlah token yang sebanding, apakah proporsi
token unik berkaitan dengan proporsi artifact_signal?

The hypothesis concerns repetition versus variety, not token meaning.
Threat_rank is a secondary counter-outcome, not part of the main question. Five
token-count bands are chosen without labels. Within each band, compare the lower
and upper within-band average-rank tails (rank <= 0.25 vs >= 0.75), then inspect
per-cycle direction. In the shortest band, every row has the same unique-token
fraction; no high-diversity tail exists there, so no contrast is computed for
that band. The other {len(per_bin)} length bands have {int(per_bin.n.sum()):,} rows.
Artifact-rate differences across them are all positive, ranging
{per_bin.artifact_delta_high_minus_low.min():.1%} to
{per_bin.artifact_delta_high_minus_low.max():.1%}.

A stricter check fixes the **exact token count** and compares the same within-count
average-rank tails. It requires at least 120 observations per count and at least
20 in each tail. For unique-token fraction, support is
{exact_ttr['supported_token_counts']} counts/{exact_ttr['support_rows']:,} rows,
or {exact_ttr['share_of_stream_present']:.1%} of stream-present and
{exact_ttr['share_of_all_train']:.1%} of all Train. The high-diversity tail
has artifact prevalence +{exact_ttr['upper_vs_lower_diversity_artifact_delta']:.1%}
{exact_ttr['low_tail_compared_rows']:,} lower-tail and
{exact_ttr['high_tail_compared_rows']:,} upper-tail observations are compared.
Their respective artifact prevalences are
{exact_ttr['average_low_tail_artifact_rate']:.1%} and
{exact_ttr['average_high_tail_artifact_rate']:.1%} when each exact-length
stratum is weighted by its full count. Across the individual compared tail
observations, prevalences are {support_sens[1]['pooled_lower_tail_artifact_rate']:.1%}
and {support_sens[1]['pooled_upper_tail_artifact_rate']:.1%}, a difference of
{support_sens[1]['pooled_tail_delta_high_minus_low']:.1%}. Mean threat differs by
{exact_ttr['upper_vs_lower_diversity_threat_delta']:+.2f} points.
The artifact contrast is positive for
{exact_ttr['positive_token_count_strata']}/{exact_ttr['supported_token_counts']}
counts covering {exact_ttr['share_supported_rows_from_positive_strata']:.1%}
of the supported rows. Two related measures show the same direction at exact
length: within-row Shannon entropy +{exact_entropy['upper_vs_lower_diversity_artifact_delta']:.1%},
and lower repeated-token share +{exact_repeat['upper_vs_lower_diversity_artifact_delta']:.1%}.
There is meaningful heterogeneity across token counts; the observed exact-count
artifact contrasts span {exact_ttr['artifact_delta_minmax'][0]:+.1%} to
{exact_ttr['artifact_delta_minmax'][1]:+.1%}.
Changing the minimum exact-length support threshold from 80 to 120 to 200
retains {support_sens[0]['supported_token_counts']},
{support_sens[1]['supported_token_counts']}, and
{support_sens[2]['supported_token_counts']} counts, respectively; each has at
least 20 observations in both tails. Their support-weighted artifact contrasts
are {support_sens[0]['support_weighted_stratum_delta']:.1%},
{support_sens[1]['support_weighted_stratum_delta']:.1%}, and
{support_sens[2]['support_weighted_stratum_delta']:.1%}.

With broad label-blind length bands, the artifact contrast is positive in
{int((cy_art>0).sum())}/{len(cy_art)} cycle-specific estimates. Resampling the
33 cycle summaries gives a 90% sensitivity range of
{metrics['text_diversity']['cycle_bootstrap_artifact']['p05']:.1%} to
{metrics['text_diversity']['cycle_bootstrap_artifact']['p95']:.1%} for the
average cycle contrast, conditional on exchangeability. This is not a CI for
future cycles. Exact-length contrasts within cycle pass the stricter support
rule in only {exact_summary['cycle_and_exact_length_sensitivity']['cycles_supported']}
cycles, so simultaneous exact-length and cycle adjustment remains unsupported.
All contrasts are exploratory; support and effect ranges are in stats.json
and length_stratified_diversity.csv.

## Candidate T2 — agreement of text_load with observed text size

**Question:** Seberapa baik text_load mewakili jumlah token dan karakter yang
benar-benar tersedia dalam signal_stream?

The observed Spearman association is {textload['spearman_text_load_tokens']:.3f}
with regex token count and {textload['spearman_text_load_characters']:.3f}
with character count, suggesting a strong length proxy. The feature-level check
may be too obvious to merit the assigned EDA slot. See text_load_coherence in
stats.json, its quartile table and text_length_coherence.png.
Reported text_load quantiles are separately labeled for all Train rows and for
stream-present rows; text-load quartiles use only the latter ({len(present):,}).

## Initial overlap screen

Rousan studies entry_friction and impact_load for artifact_signal; those fields
are absent here. Abhi studies scope_pressure conditional on impact_load for
threat_rank; both are absent. Rizky studies telemetry_flag and module_spread
for artifact_signal; both are absent. Samuel's existing question centers on
impact_load–threat_rank across cycle periods; no such pair is analyzed here.
Cycle appears only as a robustness/support dimension. Sharing a target does not
by itself make these questions equivalent.

## Figures and interpretation limit

- text_length_coherence.png: numeric text_load versus realized regex token count,
  and how a simple unique-token fraction depends on stream length.
- diversity_by_length_and_targets.png: artifact-rate contrasts by
  label-blind length band plus per-cycle sensitivity; threat means are included
  as a secondary counter-outcome.

Regex is lowercased ASCII alphanumeric runs. These measures do not assign
meaning to anonymized tokens. All Train cycles are development data already
seen by prior work; findings remain exploratory rather than confirmatory.
"""
    (output_dir/"FINDINGS.md").write_text(note,encoding="utf-8")
    print(json.dumps({"output":str(output_dir),"present_n":len(present),
      "missing_n":int(data.stream_missing.sum()),
      "text_load_token_spearman":metrics["text_load_coherence"]["spearman_text_load_tokens"],
      "diversity_delta_threat":pooled_threat,"diversity_delta_artifact":pooled_artifact,
      "supported_cycles":len(by_cycle)},ensure_ascii=False))
    return metrics

run=main

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path);p.add_argument("--output-dir",type=Path)
    a=p.parse_args();main(a.root,a.output_dir)
