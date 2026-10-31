"""Targeted exact-length x cycle-period tail contrasts for EDA candidate A.

No estimator, Test input, target-based group selection or historical writer.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.utils.paths import task_files
from scripts.eda_candidates_text import token_stats


def load_frame():
    source=task_files('classification')['train']
    assert source and hashlib.sha256(source.read_bytes()).hexdigest()=='875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c'
    f=pd.read_csv(source)
    assert f.shape==(50608,19) and f.record_id.is_unique and f.artifact_signal.isin([0,1]).all()
    missing=f.signal_stream.isna()|f.signal_stream.fillna('').str.strip().eq('')
    f=f.loc[~missing,['signal_stream','artifact_signal','cycle_code']].copy()
    features=f.signal_stream.map(token_stats)
    f['length']=[r[0]for r in features];f['diversity']=[r[2]for r in features]
    f=f.drop(columns='signal_stream')
    assert len(f)==47204 and f.length.gt(0).all()
    f['period']=pd.cut(f.cycle_code,[46,57,68,79],labels=['47–57','58–68','69–79']).astype(str)
    return f


def contrasts(frame,keys,min_rows,min_tail):
    records=[];excluded=[]
    for key,g in frame.groupby(keys,observed=True,sort=True):
        key=key if isinstance(key,tuple)else(key,)
        identifiers=dict(zip(keys,key))
        rank=g.diversity.rank(method='average',pct=True)
        low=g[rank.le(.25)];high=g[rank.ge(.75)]
        reason=None
        if len(g)<min_rows:reason='below_min_rows'
        elif g.diversity.nunique()<2:reason='constant_diversity'
        elif min(len(low),len(high))<min_tail:reason='below_min_tail'
        if reason:
            excluded.append({**identifiers,'n':len(g),'low_n':len(low),'high_n':len(high),'reason':reason})
            continue
        records.append({**identifiers,'n':len(g),'low_n':len(low),'high_n':len(high),
            'low_positives':int(low.artifact_signal.sum()),'high_positives':int(high.artifact_signal.sum()),
            'low_rate':low.artifact_signal.mean(),'high_rate':high.artifact_signal.mean(),
            'delta':high.artifact_signal.mean()-low.artifact_signal.mean(),
            'diversity_low_mean':low.diversity.mean(),'diversity_high_mean':high.diversity.mean()})
    return pd.DataFrame(records),pd.DataFrame(excluded)


def summarize(table,denominator,weights=None):
    if table.empty:return {'strata':0,'support_rows':0,'coverage_present':0,'delta':None}
    w=np.asarray(table.n if weights is None else weights,float);w=w/w.sum()
    positive=table.delta.gt(1e-15);negative=table.delta.lt(-1e-15)
    return {'strata':len(table),'support_rows':int(table.n.sum()),
        'denominator_present':int(denominator),'coverage_present':float(table.n.sum()/denominator),
        'low_n':int(table.low_n.sum()),'high_n':int(table.high_n.sum()),
        'compared_tail_rows':int(table.low_n.sum()+table.high_n.sum()),
        'low_rate':float(np.dot(w,table.low_rate)),'high_rate':float(np.dot(w,table.high_rate)),
        'delta':float(np.dot(w,table.delta)),
        'positive_strata':int(positive.sum()),'negative_strata':int(negative.sum()),
        'zero_strata':int((~positive&~negative).sum()),
        'rows_in_positive_strata':int(table.loc[positive,'n'].sum()),
        'positive_support_row_share':float(table.loc[positive,'n'].sum()/table.n.sum()),
        'positive_compared_tail_row_share':float((table.loc[positive,'low_n'].sum()+table.loc[positive,'high_n'].sum())/(table.low_n.sum()+table.high_n.sum())),
        'delta_min':float(table.delta.min()),'delta_max':float(table.delta.max())}


def run():
    out=ROOT/'reports/eda_a_robustness';out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((out/'protocol_and_preservation.json').read_text(encoding='utf-8'))
    f=load_frame();counts=f.groupby('period').size().to_dict()
    results={'n_all_train':50608,'n_present':len(f),'n_missing':3404,
             'input_sha256':protocol['input']['sha256'],'period_denominators_present':counts,'rules':{}}
    baseline,_=contrasts(f,['length'],120,20)
    results['baseline_exact_length']=summarize(baseline,len(f))
    assert len(baseline)==71 and int(baseline.n.sum())==37681
    assert np.isclose(results['baseline_exact_length']['delta'],.14201954123205932,atol=1e-12,rtol=0)
    baseline.to_csv(out/'baseline_exact_length.csv',index=False)
    all_periods=[];common_summary=[]
    for label,min_rows,min_tail in [('primary_120_20',120,20),('sensitivity_60_10',60,10)]:
        table,excluded=contrasts(f,['length','period'],min_rows,min_tail)
        table.to_csv(out/f'{label}_strata.csv',index=False)
        excluded.to_csv(out/f'{label}_excluded.csv',index=False)
        period_stats=[]
        for period in ['47–57','58–68','69–79']:
            sub=table[table.period.eq(period)]
            s={'rule':label,'period':period,**summarize(sub,counts[period])}
            period_stats.append(s);all_periods.append(s)
        pooled=summarize(table,len(f))
        pooled['coverage_all_train']=pooled['support_rows']/50608
        pooled['compared_tail_share_all_train']=pooled['compared_tail_rows']/50608
        pooled['equal_period_delta']=float(np.mean([r['delta']for r in period_stats]))
        pooled['periods_positive']=sum(r['delta']>0 for r in period_stats)
        old_mask=f.length.isin(baseline.length).to_numpy()
        new_mask=pd.MultiIndex.from_frame(f[['length','period']]).isin(
            pd.MultiIndex.from_frame(table[['length','period']]))
        pooled['baseline_support_rows_retained']=int((old_mask&new_mask).sum())
        pooled['baseline_support_retention']=float((old_mask&new_mask).sum()/old_mask.sum())
        pooled['support_rows_not_in_baseline_cohort']=int((~old_mask&new_mask).sum())
        # Same lengths in all periods and the same pooled-length weights.
        lengths=set.intersection(*(set(table.loc[table.period.eq(p),'length'])for p in counts))
        common=table[table.length.isin(lengths)].copy()
        w=common.groupby('length').n.sum();w=w/w.sum()
        common_period=[]
        for period in ['47–57','58–68','69–79']:
            sub=common[common.period.eq(period)].sort_values('length')
            s={'rule':label,'period':period,**summarize(sub,counts[period],w.reindex(sub.length).to_numpy())}
            common_period.append(s);common_summary.append(s)
        common.to_csv(out/f'{label}_common_lengths.csv',index=False)
        common_pooled=summarize(common,len(f))
        common_pooled['common_lengths']=len(lengths)
        common_pooled['equal_period_common_weight_delta']=float(np.mean([r['delta']for r in common_period]))
        results['rules'][label]={'min_rows':min_rows,'min_each_tail':min_tail,'pooled':pooled,
            'periods':period_stats,'common_length_support':common_pooled,'common_length_periods':common_period,
            'excluded_groups':len(excluded),'excluded_rows':int(excluded.n.sum())}
    pd.DataFrame(all_periods).to_csv(out/'period_summary.csv',index=False)
    pd.DataFrame(common_summary).to_csv(out/'common_length_period_summary.csv',index=False)
    results['interpretation']={
        'tail_definition':'Local within exact length x period average percentile ranks; <=.25 vs >=.75; ties retained. Absolute TTR cutpoints may differ across periods.',
        'pooling':'Stratum n weights; per-period effects not crude combined tail proportions. Equal-period view prevents the largest period alone determining the mean.',
        'common_length_comparison':'Intersection of supported lengths in all3 periods; same pooled length-count weights used for low/high and each period.',
        'limits':'Period control does not control every exact cycle. Supported strata and contrasted tails are different denominators; middle rows define weights but are not contrasted. This is exploratory, noncausal, and does not certify text semantics.'}
    (out/'period_stats.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(pd.DataFrame(all_periods)[['rule','period','strata','support_rows','coverage_present','low_n','high_n','delta','positive_strata','negative_strata']].to_string(index=False))
    print('POOLED:',[(k,v['pooled']['delta'],v['pooled']['coverage_present'])for k,v in results['rules'].items()])
    return results


if __name__=='__main__':run()
