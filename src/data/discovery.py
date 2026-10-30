"""Descriptive discovery for Samuel's TP2 regression EDA; never trains models."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from src.utils.paths import PROJECT_ROOT, data_root, task_files


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _num(x):
    return float(x) if pd.notna(x) and np.isfinite(x) else None


def _corr(frame):
    pair = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 2 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return {'n': int(len(pair)), 'pearson': None, 'spearman': None}
    return {'n': int(len(pair)), 'pearson': _num(pearsonr(pair.iloc[:, 0], pair.iloc[:, 1]).statistic),
            'spearman': _num(spearmanr(pair.iloc[:, 0], pair.iloc[:, 1]).statistic)}


def _target_profile(s):
    numeric = pd.to_numeric(pd.Series(s), errors='coerce')
    missing = int(numeric.isna().sum())
    nonfinite = int(np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).sum())
    s = numeric.replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return {'input_rows': int(len(numeric)), 'n': 0, 'missing': missing,
                'nonfinite': nonfinite, 'unique': 0}
    q = s.quantile([.01, .05, .10, .25, .50, .75, .90, .95, .99])
    counts = s.value_counts()
    return {
        'input_rows': int(len(numeric)), 'n': int(len(s)), 'missing': missing, 'nonfinite': nonfinite,
        'unique': int(s.nunique()), 'unique_fraction': _num(s.nunique()/len(s)),
        'min': _num(s.min()), 'max': _num(s.max()), 'mean': _num(s.mean()),
        'sd': _num(s.std()), 'skew': _num(s.skew()), 'quantiles': {str(int(k*100)): _num(v) for k,v in q.items()},
        'integer_like_fraction_abs_tol_1e_8': _num(np.isclose(s, np.round(s), rtol=0, atol=1e-8).mean()),
        'endpoint_counts': {'exact_0': int(s.eq(0).sum()), 'exact_100': int(s.eq(100).sum())},
        'top_values': [{'value': _num(v), 'n': int(n), 'share': _num(n/len(s))} for v,n in counts.head(10).items()],
        'top_value_shares': {f'top_{k}': _num(counts.head(k).sum()/len(s)) for k in (1,5,10)},
    }


def _blocks(cycles):
    levels = np.sort(pd.Series(cycles).dropna().unique())
    chunks = np.array_split(levels, min(3, len(levels))) if len(levels) else []
    out = []
    labels = {}
    for i, chunk in enumerate(chunks, 1):
        label = f'period_{i:02d}'
        lo, hi = chunk[0].item(), chunk[-1].item()
        out.append({'period': label, 'cycle_min': int(lo), 'cycle_max': int(hi), 'n_cycles': int(len(chunk))})
        labels.update({v: label for v in chunk})
    return pd.Series(cycles).map(labels), out


def _rank_lattice(df):
    per_cycle = df.groupby('cycle_code', sort=True)['threat_rank']
    cycle_n = per_cycle.transform('size')
    rank = per_cycle.rank(method='average', pct=False)
    expected = (100 * rank / cycle_n).round(4)
    pooled = (100 * df['threat_rank'].rank(method='average') / len(df)).round(4)
    observed = df['threat_rank']
    return {
        'diagnostic': 'Compare observed threat_rank with a within-cycle and pooled empirical average-rank percentile rounded to four decimals.',
        'within_cycle': {'exact_match_rows': int(observed.eq(expected).sum()), 'rows': int(len(df)),
                         'exact_match_fraction': _num(observed.eq(expected).mean()),
                         'max_absolute_difference': _num((observed-expected).abs().max()),
                         'mean_absolute_difference': _num((observed-expected).abs().mean())},
        'pooled': {'exact_match_rows': int(observed.eq(pooled).sum()), 'rows': int(len(df)),
                   'exact_match_fraction': _num(observed.eq(pooled).mean()),
                   'max_absolute_difference': _num((observed-pooled).abs().max()),
                   'mean_absolute_difference': _num((observed-pooled).abs().mean())},
        'interpretation_limit': 'An exact numerical match is a distributional construction clue only. It does not prove the target-generation algorithm, causal meaning, or permission to reconstruct or use labels outside the official task.',
    }


def _write_figures(df, third_summary, repeated_impacts, tails, tp1, tp1_scores, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': 'white', 'axes.facecolor': 'white'})
    paths = []

    # TP1 is an optional comparison; all TP2 figures are independent of it.
    if tp1_scores is not None:
        fig, ax = plt.subplots(figsize=(9, 5.2), layout='constrained')
        tp1counts = tp1_scores.value_counts().head(8).sort_values()
        ax.barh([f"{x:.4f}" for x in tp1counts.index], tp1counts.values, color='#6888a8')
        ax.set(title='TP1 impact_load: eight most frequent scores', xlabel='Rows', ylabel='Observed score')
        fig.savefig(out_dir/'discovery_tp1_top_scores.png', dpi=160)
        plt.close(fig); paths.append('discovery_tp1_top_scores.png')
    fig, ax = plt.subplots(figsize=(9, 5.2), layout='constrained')
    ax.hist(df.threat_rank, bins=np.linspace(0,100,21), color='#2864a5', edgecolor='white')
    ax.set(title='TP2 threat_rank: observed target distribution', xlabel='threat_rank', ylabel='Train rows')
    ax.grid(axis='y', alpha=.2)
    fig.savefig(out_dir/'discovery_tp2_target_histogram.png', dpi=160)
    plt.close(fig); paths.append('discovery_tp2_target_histogram.png')

    fig, axes = plt.subplots(2, 1, figsize=(9, 8.0), sharex=True, layout='constrained')
    cycles = third_summary['cycle_code'].astype(int)
    axes[0].plot(cycles, third_summary['pearson'], color='#2864a5', marker='o', markersize=3, label='Pearson')
    axes[0].plot(cycles, third_summary['spearman'], color='#cf8e30', marker='s', markersize=3, label='Spearman')
    axes[0].set(title='impact_load association by cycle_code', ylabel='Correlation')
    axes[0].legend(ncol=2, loc='lower left'); axes[0].grid(alpha=.2)
    axes[1].plot(cycles, third_summary['threat_median'], color='#2864a5', marker='o', markersize=3, label='Median')
    axes[1].plot(cycles, third_summary['threat_p10'], color='#cf8e30', linestyle='--', label='P10')
    axes[1].plot(cycles, third_summary['threat_p90'], color='#cf8e30', linestyle=':', label='P90')
    axes[1].set(title='Threat rank marginal by cycle', xlabel='cycle_code (larger code is described as newer in TP1 course material)', ylabel='threat_rank')
    axes[1].legend(ncol=3, loc='lower left'); axes[1].grid(alpha=.2)
    fig.savefig(out_dir/'discovery_cycle_relationship.png', dpi=160)
    plt.close(fig); paths.append('discovery_cycle_relationship.png')

    # Show six repeated impact levels; all periods share a common target scale.
    selected = repeated_impacts.groupby('impact_load').n.sum().nlargest(6).index
    plot = repeated_impacts[repeated_impacts.impact_load.isin(selected)].copy()
    impact_order = sorted(selected)
    offset = {'period_01':-.18, 'period_02':0, 'period_03':.18}
    colors = {'period_01':'#2864a5','period_02':'#cf8e30','period_03':'#7257a3'}
    period_labels = {'period_01':'Cycles 47–57','period_02':'Cycles 58–68','period_03':'Cycles 69–79'}
    fig, ax = plt.subplots(figsize=(11, 6), layout='constrained')
    for period in ['period_01','period_02','period_03']:
        g = plot[plot.period == period]
        for _, row in g.iterrows():
            x = impact_order.index(row.impact_load)+offset[period]
            ax.errorbar(x, row['median'], yerr=[[row['median']-row['p10']],[row['p90']-row['median']]],
                        fmt='o', capsize=2, color=colors[period], markersize=5,
                        label=period_labels[period] if x == impact_order.index(impact_order[0])+offset[period] else None)
    ax.set_xticks(range(len(impact_order)), [f'{x:.4f}' for x in impact_order])
    ax.set(title='Conditional threat_rank at repeated impact_load values', xlabel='Exact impact_load value (six most common)',
           ylabel='Median and P10–P90 threat_rank', ylim=(0,100))
    ax.grid(axis='y', alpha=.2); ax.legend(title='Ordered cycle period')
    fig.savefig(out_dir/'discovery_conditional_spread.png', dpi=160)
    plt.close(fig); paths.append('discovery_conditional_spread.png')

    # Fixed tail cutoffs are descriptive and were not selected against prediction results.
    fig, ax = plt.subplots(figsize=(10, 5.8), layout='constrained')
    t = tails.copy()
    labels = [f"P{str(row.period)[-2:]} / {row.impact_band_label} (n={row.n})"
              for row in t.itertuples()]
    y = np.arange(len(t))
    ax.barh(y-.18, t.low_share, height=.35, color='#6888a8', label='threat_rank ≤ 5')
    ax.barh(y+.18, t.high_share, height=.35, color='#cf8e30', label='threat_rank ≥ 95')
    ax.set_yticks(y, labels); ax.invert_yaxis()
    ax.set(title='Low and high target shares within impact bands', xlabel='Share of rows in period × impact band', ylabel='Ordered cycle period / impact band')
    ax.grid(axis='x', alpha=.2); ax.legend()
    fig.savefig(out_dir/'discovery_tail_by_impact_cycle.png', dpi=160)
    plt.close(fig); paths.append('discovery_tail_by_impact_cycle.png')
    return paths


def run_discovery(data_dir=None, tp1_data_dir=None, output_dir=None, make_plots=True):
    """Summarize observed training structure and optional local TP1 baseline."""
    root = Path(data_dir).resolve() if data_dir else data_root()
    files = task_files('regression', root)
    if any(files[k] is None for k in ('train','test','sample')):
        raise FileNotFoundError('Require official regression Train.csv, Test.csv, SampleSubmission.csv.')
    train = pd.read_csv(files['train'])
    test = pd.read_csv(files['test'])
    sample = pd.read_csv(files['sample'])
    target_only = sorted(set(train.columns)-set(test.columns))
    sample_target = sorted(set(sample.columns)-{'record_id'})
    if target_only != ['threat_rank'] or sample_target != ['threat_rank'] or set(sample.columns) != {'record_id','threat_rank'}:
        raise ValueError(f'Target schema not verified: train-only={target_only}; sample labels={sample_target}.')
    for col in ('impact_load','cycle_code'):
        if col not in train or col not in test: raise ValueError(f'Missing required {col}')
    if not train.columns.drop('threat_rank').equals(test.columns):
        raise ValueError('Train and test feature schemas/orders differ.')
    if not sample.record_id.equals(test.record_id): raise ValueError('Sample and test ID order differs.')
    target = train.threat_rank
    if target.isna().any() or np.isinf(target).any(): raise ValueError('Target has missing or nonfinite values.')

    df=train[['record_id','cycle_code','impact_load','entry_friction','threat_rank']].copy()
    df['period'], blocks = _blocks(df.cycle_code)
    period_summary=[]
    percycle=[]
    for block in blocks:
        p=block['period']; g=df[df.period==p]
        row={'period':p, 'cycle_min':block['cycle_min'],'cycle_max':block['cycle_max'],'n':int(len(g)),
             'cycle_count':block['n_cycles'],'impact_threat_correlation':_corr(g[['impact_load','threat_rank']]),
             'target':_target_profile(g.threat_rank)}
        period_summary.append(row)
    for cycle,g in df.groupby('cycle_code',sort=True):
        c=_corr(g[['impact_load','threat_rank']])
        q=g.threat_rank.quantile([.1,.25,.5,.75,.9])
        percycle.append({'cycle_code':int(cycle),'n':len(g),'period':str(g.period.iloc[0]),
                         'pearson':c['pearson'],'spearman':c['spearman'],
                         'threat_p10':_num(q.loc[.1]),'threat_q25':_num(q.loc[.25]),
                         'threat_median':_num(q.loc[.5]),'threat_q75':_num(q.loc[.75]),
                         'threat_p90':_num(q.loc[.9]),'low_share_le_5':_num(g.threat_rank.le(5).mean()),
                         'high_share_ge_95':_num(g.threat_rank.ge(95).mean())})

    # Equal-width (0,100] bins are common across periods; handle any exact zero in the first bin.
    ybins=pd.cut(df.threat_rank,bins=np.linspace(0,100,11),include_lowest=True)
    target_bins=[]
    for band,g in df.groupby(ybins,observed=True):
        target_bins.append({'band':str(band),'n':len(g),'share':_num(len(g)/len(df))})
    percentile_scores=100*df.groupby('cycle_code').threat_rank.rank(method='average')/df.groupby('cycle_code').threat_rank.transform('size')
    lattice=_rank_lattice(df)
    lattice['within_cycle']['continuous_percentile_unrounded_rows'] = int(np.isclose(df.threat_rank.to_numpy(),percentile_scores.to_numpy(),rtol=0,atol=1e-12).sum())
    lattice['within_cycle']['continuous_percentile_unrounded_max_absolute_difference'] = _num((df.threat_rank.to_numpy()-percentile_scores.to_numpy()).__abs__().max())

    # Profiles do not include IDs or row-level data in JSON.
    reg_profile=_target_profile(target)
    reg_profile['missing'] = int(train.threat_rank.isna().sum())
    train_test_drift={}
    for col in ('impact_load','cycle_code'):
        a,b=train[col],test[col]
        train_test_drift[col]={'train_median':_num(a.median()),'test_median':_num(b.median()),
                               'train_range':[_num(a.min()),_num(a.max())],
                               'test_range':[_num(b.min()),_num(b.max())],
                               'missing_train':int(a.isna().sum()),'missing_test':int(b.isna().sum())}

    tp1={}; tp1_path=None; tp1_status='NOT_PROVIDED'; tp1_scores=None
    if tp1_data_dir is not None:
        oldfiles=task_files('regression',tp1_data_dir)
        if oldfiles['train']:
            tp1_path=oldfiles['train']
            old=pd.read_csv(tp1_path)
            if 'impact_load' not in old: raise ValueError('TP1 reference file has no impact_load.')
            tp1_status='INCLUDED'
            tp1={'source_file_name':tp1_path.name,'sha256':_sha256(tp1_path),
                 'target_name':'impact_load','target':_target_profile(old.impact_load),
                 'source_note':'TP1 local regression train, read only; support/concentration comparison is descriptive, target meanings/scales differ.'}
            tp1_scores=old.impact_load
        else:
            tp1_status='NOT_FOUND'

    # TP2 repeated-impact summaries are part of discovery whether or not TP1 is available.
    freq=df.impact_load.value_counts()
    repeated=freq[freq>1].index
    repeated_rows=[]
    for level in repeated:
        allg=df[df.impact_load.eq(level)]
        q=allg.threat_rank.quantile([.1,.5,.9])
        repeated_rows.append({'impact_load':_num(level),'period':'all','n':len(allg),'median':_num(q.loc[.5]),'p10':_num(q.loc[.1]),'p90':_num(q.loc[.9]),'iqr':_num(allg.threat_rank.quantile(.75)-allg.threat_rank.quantile(.25))})
        for p in [b['period'] for b in blocks]:
            g=allg[allg.period.eq(p)]
            if len(g):
                q=g.threat_rank.quantile([.1,.5,.9])
                repeated_rows.append({'impact_load':_num(level),'period':p,'n':len(g),'median':_num(q.loc[.5]),'p10':_num(q.loc[.1]),'p90':_num(q.loc[.9]),'iqr':_num(g.threat_rank.quantile(.75)-g.threat_rank.quantile(.25))})
    repeated_df=pd.DataFrame(repeated_rows,columns=['impact_load','period','n','median','p10','p90','iqr'])
    repeated_audit={'distinct_impact_values':int(df.impact_load.nunique()),'repeated_values':int((freq>1).sum()),
                    'rows_on_repeated_values':int(freq[freq>1].sum()),
                    'repeated_values_with_multiple_threat_targets':int((df[df.impact_load.isin(repeated)].groupby('impact_load').threat_rank.nunique()>1).sum()),
                    'top_levels':[{'impact_load':_num(v),'n':int(n),'target_median':_num(df.loc[df.impact_load==v,'threat_rank'].median()),'target_min':_num(df.loc[df.impact_load==v,'threat_rank'].min()),'target_max':_num(df.loc[df.impact_load==v,'threat_rank'].max())} for v,n in freq.head(12).items()]}

    # Fixed low/high target bands are descriptive, not preregistered or prediction-selected.
    df['impact_band']=pd.cut(df.impact_load,bins=np.linspace(df.impact_load.min(),df.impact_load.max(),7),include_lowest=True)
    tails=[]
    for (p,b),g in df.groupby(['period','impact_band'],observed=True):
        tails.append({'period':p,'impact_band':str(b),'impact_band_label':f'{b.left:.1f}-{b.right:.1f}','n':len(g),
                      'low_n_le_5':int(g.threat_rank.le(5).sum()),'low_share':_num(g.threat_rank.le(5).mean()),
                      'high_n_ge_95':int(g.threat_rank.ge(95).sum()),'high_share':_num(g.threat_rank.ge(95).mean())})
    tails_df=pd.DataFrame(tails)

    entry=[]
    for (p,ef),g in df.groupby(['period','entry_friction'],observed=True):
        entry.append({'period':p,'entry_friction':int(ef),'n':len(g),'target_mean':_num(g.threat_rank.mean()),
                      'target_median':_num(g.threat_rank.median()),'target_p10':_num(g.threat_rank.quantile(.1)),
                      'target_p90':_num(g.threat_rank.quantile(.9))})

    rank_relation=_corr(df[['impact_load','threat_rank']])
    exact_audit={'rows':len(df),'duplicate_rows':int(train.duplicated().sum()),
                 'duplicate_ids':int(train.record_id.duplicated().sum()),
                 'train_test_id_overlap':int(train.record_id.isin(test.record_id).sum()),
                 'sample_id_order_match':bool(sample.record_id.equals(test.record_id)),
                 'all_required_and_target_missing':{c:int(train[c].isna().sum()) for c in ('impact_load','cycle_code','threat_rank')},
                 'numeric_nonfinite':{c:int(np.isinf(pd.to_numeric(train[c],errors='coerce')).sum()) for c in ('impact_load','cycle_code','threat_rank')},
                 'train_cycles':[int(train.cycle_code.min()),int(train.cycle_code.max())],
                 'test_cycles':[int(test.cycle_code.min()),int(test.cycle_code.max())]}
    bin_table=[]
    for lo,hi in [(110.5078,154.0178),(134.6432,115.9669)]:
        # Report specifically named support values to examine an apparent plateau/reversal.
        vals=[]
        for impact in [lo,hi]:
            g=df[df.impact_load.eq(impact)]
            vals.append({'impact_load':impact,'n':len(g),'median':_num(g.threat_rank.median()),
                         'p10':_num(g.threat_rank.quantile(.1)),'p90':_num(g.threat_rank.quantile(.9))})
        bin_table.append({'comparison':'fixed recurring impact levels','values':vals})

    report={
        'status':'DISCOVERY_COMPLETE_NOT_A_LOCKED_QUESTION',
        'provenance':{'regression_files':{k:{'name':v.name,'bytes':v.stat().st_size,'sha256':_sha256(v)} for k,v in files.items()},
                      'tp1_reference':{**(tp1 or {}),'status':tp1_status},
                      'grain':'one row per record_id in provided task CSV; domain meaning of record UNKNOWN',
                      'cycle_basis':'Sort distinct observed cycle_code values and divide them into three near-equal contiguous groups without consulting target. cycle_code larger means newer per local TP1 course PDF p.16; application to TP2 assumes same semantics.'},
        'data_quality':exact_audit,
        'target_tp2':reg_profile,
        'tp1_target_comparison':{'status':tp1_status,'tp1':tp1.get('target') if tp1 else None,'tp2':reg_profile,
            'interpretation':'Support and concentration are comparable descriptively; target scales and definitions differ, so locations/ranks/KS comparisons between raw scores are not used.'},
        'impact_threat':{'overall_correlation':rank_relation,
                         'repeated_impact_audit':repeated_audit,
                         'by_cycle':percycle,'by_cycle_period':period_summary,
                         'target_decile_bins':target_bins,
                         'test_covariate_profile':train_test_drift},
        'within_cycle_percentile_diagnostic':lattice,
        'conditional_extremes':{'thresholds':{'low':'threat_rank <= 5','high':'threat_rank >= 95',
                                              'note':'Fixed descriptive cutoffs; not selected against prediction or validation results.'},
                                 'by_period_cycle':[{k:v for k,v in r.items() if k in ('cycle_code','period','n','low_share_le_5','high_share_ge_95')} for r in percycle],
                                 'by_period_impact_band':tails},
        'repeated_impact_values_by_period':repeated_rows,
        'entry_friction_context':entry,
        'candidate_signal_examples':bin_table,
        'interpretation_limits':['Descriptive associations do not establish causality, statistical significance or prediction error.',
            'TP2 has no model predictions here, so TP1 extreme under/overprediction rates must not be transferred to TP2.',
            'An exact within-cycle percentile identity suggests a rank-like encoding but does not establish official semantics or target-generation lineage.',
            'Cycle grouping assumes ordered code semantics; TP1 course material defines cycle_code as disclosure order, but TP2-specific calendar meaning remains unverified.',
            'Fixed descriptive tail cutoffs 5 and 95 were not selected against prediction or validation results.'],
        'figures':[]}
    out=Path(output_dir) if output_dir else PROJECT_ROOT/'reports'/'eda'
    out.mkdir(parents=True,exist_ok=True)
    if make_plots:
        report['figures']=_write_figures(df,pd.DataFrame(percycle),repeated_df,tails_df,tp1,tp1_scores,out)
    # No row IDs, names or text are written.
    (out/'discovery_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return report
