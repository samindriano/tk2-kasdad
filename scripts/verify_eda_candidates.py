"""Independently recompute selected decision-critical EDA values from Train.

The checks deliberately do not call the discovery estimators/functions. They
cover the recommendation's main contrasts, not every exploratory table entry.
"""
from pathlib import Path
import ast
import argparse
import hashlib
import json
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.paths import task_files


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads((ROOT/path).read_text(encoding='utf-8'))
def close(actual, expected):
    assert np.isclose(actual, expected, rtol=0, atol=1e-10), (actual,expected)


def main(check_preservation=False, check_paused_drafts=False):
    report = ROOT/'reports/eda_candidates'
    reg = pd.read_csv(task_files('regression')['train'])
    cls = pd.read_csv(task_files('classification')['train'])
    assert reg.shape == cls.shape == (50608,19)
    shared = sorted(set(reg) & set(cls))
    assert len(shared)==18 and reg[shared].equals(cls[shared])
    assert reg.record_id.is_unique and reg.record_id.equals(cls.record_id)
    f = reg.assign(artifact_signal=cls.artifact_signal)
    checks = []
    # F: direct independent sums and masks, no helper from the discovery script.
    target = read('reports/eda_candidates/targets/stats.json')
    zero = f.artifact_signal.eq(0); one=~zero
    gap = np.mean(f.loc[one,'threat_rank'].to_numpy())-np.mean(f.loc[zero,'threat_rank'].to_numpy())
    close(gap, target['mean_difference'])
    assert int((f.threat_rank.ge(75)&zero).sum())==target['high_quartile']['no_observed_artifact']==5029
    assert int(f.threat_rank.ge(75).sum())==target['high_quartile']['n']==12643
    assert int((f.threat_rank.le(25)&one).sum())==target['low_quartile']['observed_artifact']==987
    assert int(f.threat_rank.le(25).sum())==target['low_quartile']['n']==12638
    checks.append('F one-to-one label join, class means, high/low threshold denominators and counterexamples')
    # A: independently tokenize without Counter, use SciPy ranks rather than
    # pandas average ranks from the author, and retain all qualified exact lengths.
    missing = f.signal_stream.isna() | f.signal_stream.fillna('').str.strip().eq('')
    present = f.loc[~missing].copy()
    tokens = [re.findall('[a-z0-9]+', str(s).lower()) for s in present.signal_stream]
    present['length'] = [len(t)for t in tokens]
    present['diversity'] = [len(set(t))/len(t) if t else 0 for t in tokens]
    records=[]
    for length,g in present.groupby('length'):
        if len(g)<120 or g.diversity.nunique()<2: continue
        ranks=rankdata(g.diversity.to_numpy(),method='average')/len(g)
        low=g.iloc[np.flatnonzero(ranks<=.25)]
        high=g.iloc[np.flatnonzero(ranks>=.75)]
        if min(len(low),len(high))<20: continue
        records.append({'n':len(g),'low_n':len(low),'high_n':len(high),
                        'low_rate':low.artifact_signal.mean(),'high_rate':high.artifact_signal.mean()})
    exact=pd.DataFrame(records)
    text=read('reports/eda_candidates/text/stats.json')
    row=next(v for v in text['text_diversity']['exact_token_count_sensitivity']['features'] if v['feature']=='type_token_ratio')
    assert len(exact)==row['supported_token_counts']==71
    assert int(exact.n.sum())==row['support_rows']==37681
    assert int(exact.low_n.sum())==row['low_tail_compared_rows']==9011
    assert int(exact.high_n.sum())==row['high_tail_compared_rows']==9354
    close(np.average(exact.high_rate-exact.low_rate,weights=exact.n),row['upper_vs_lower_diversity_artifact_delta'])
    close(np.average(exact.low_rate,weights=exact.n),row['average_low_tail_artifact_rate'])
    close(np.average(exact.high_rate,weights=exact.n),row['average_high_tail_artifact_rate'])
    assert int((exact.high_rate>exact.low_rate).sum())==row['positive_token_count_strata']==64
    rho = pd.Series(rankdata(present.text_load)).corr(pd.Series(rankdata(present.length)))
    close(rho,text['text_load_coherence']['spearman_text_load_tokens'])
    checks.append('A independent set-based tokenization, SciPy tied ranks, all 71 exact-length contrasts, support and weighted prevalences; B length association')
    # C: direct missing indicator and complete Train outcome associations.
    availability=read('reports/eda_candidates/missingness/statistics.json')
    stream=availability['signal_stream_availability']
    assert int(missing.sum())==stream['missing_n']==3404
    close(f.loc[missing,'threat_rank'].mean()-f.loc[~missing,'threat_rank'].mean(),stream['threat_mean_delta'])
    close(f.loc[missing,'artifact_signal'].mean()-f.loc[~missing,'artifact_signal'].mean(),stream['artifact_rate_delta'])
    # Ten bins selected solely from complete numeric text_load, all bins included.
    bins=pd.qcut(f.text_load,10,labels=False,duplicates='drop')
    decile=[]
    for b,g in f.assign(missing=missing,bin=bins).groupby('bin'):
        a=g[g.missing]; c=g[~g.missing]
        assert min(len(a),len(c))>=20
        decile.append((len(g),a.threat_rank.mean()-c.threat_rank.mean()))
    standardized=np.average([d[1]for d in decile],weights=[d[0]for d in decile])
    close(standardized,-.8870038258691764)
    assert int((missing & f.text_load.lt(f.loc[~missing,'text_load'].min())).sum())==906
    checks.append('C missing definition, raw outcome gaps, independent all-decile standardization, 906 rows below observed present minimum')
    # D/E: known-category coverage, excluding the large missing pseudo-category.
    categories=read('reports/eda_candidates/categories/stats.json')
    for column in ['pattern_code','org_code','module_code']:
        counts=f[column].value_counts(dropna=True)
        for minimum in [30,100,300]:
            expected=categories['categories'][column]['support_counts'][str(minimum)]
            assert int((counts>=minimum).sum())==expected['nonmissing_categories']
            close(counts[counts>=minimum].sum()/len(f),expected['nonmissing_category_row_coverage'])
    checks.append('D/E every support threshold 30/100/300 excludes missing codes from known-category coverage')
    # Preserve every historical tracked artifact AND drafts from the paused task.
    scope=read('reports/eda_candidates/scope_preservation.json')
    fields=(['tracked_before'] if check_preservation else []) + (['paused_final_drafts_before'] if check_paused_drafts else [])
    for field in fields:
        for relative,expected in scope[field].items():
            assert sha(ROOT/relative)==expected, f'Protected artifact changed: {relative}'
    if check_preservation:
        checks.append(f"Historical preservation: {len(scope['tracked_before'])} tracked files byte-identical")
    if check_paused_drafts:
        checks.append(f"Paused local-task preservation: {len(scope['paused_final_drafts_before'])} drafts byte-identical")
    # A narrow static boundary check supplements source review; not a proof of all behavior.
    for path in (ROOT/'scripts').glob('eda_candidates_*.py'):
        tree=ast.parse(path.read_text(encoding='utf-8'))
        assert not any(isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and
                       node.func.attr in {'fit','fit_transform','submit','competition_submit'} for node in ast.walk(tree))
    checks.append('Source AST: no model-fit or submission calls in four discovery scripts (supplement to manual review)')
    receipt={'status':'PASS','checks':checks,'checked_input_rows':len(f),'test_labels_used':False,
             'model_fits':0,'kaggle_uploads':0,
             'paused_local_drafts_checked':bool(check_paused_drafts),
             'historical_bytes_checked':bool(check_preservation),
             'limits':['Independent recomputation covers stated decision-critical statistics; not every exploratory output cell or bootstrap draw.',
                       'Artifact preservation is byte-based; source paths outside the repository were read only.',
                       'Visual acceptance is recorded separately after actual image inspection.'],
             'main_recomputed':{'A_weighted_artifact_gap':float(np.average(exact.high_rate-exact.low_rate,weights=exact.n)),
                                'C_decile_standardized_rank_gap':float(standardized),'F_mean_rank_gap':float(gap)}}
    (report/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print('PASS:',len(checks),'independent check groups; no model or Kaggle action.')
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--check-preservation',action='store_true',help='Verify historical bytes in the reference checkout; line-ending differences on other platforms are not numerical failures.')
    parser.add_argument('--check-paused-drafts',action='store_true',help='Local working-tree check; paused drafts are intentionally not part of this research commit.')
    args=parser.parse_args()
    main(args.check_preservation,args.check_paused_drafts)
