"""Focused raw-data reconciliation of the new period and exact-cycle checks."""
from pathlib import Path
import hashlib
import json
import math
import re
import sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.utils.paths import task_files


def main():
    out=ROOT/'reports/eda_a_robustness'
    protocol=json.loads((out/'protocol_and_preservation.json').read_text(encoding='utf-8'))
    p=task_files('classification')['train']
    assert hashlib.sha256(p.read_bytes()).hexdigest()==protocol['input']['sha256']
    raw=pd.read_csv(p)
    s=raw.signal_stream
    present=~(s.isna()|s.fillna('').str.strip().eq(''))
    tokens=[re.findall('[a-z0-9]+',str(v).lower())for v in s[present]]
    f=pd.DataFrame({'length':[len(t)for t in tokens],
        'x':[len(set(t))/len(t)for t in tokens],
        'y':raw.loc[present,'artifact_signal'].to_numpy(),
        'cycle':raw.loc[present,'cycle_code'].to_numpy()})
    f['period']=np.where(f.cycle<=57,'47–57',np.where(f.cycle<=68,'58–68','69–79'))
    assert len(f)==47204
    stats=json.loads((out/'period_stats.json').read_text(encoding='utf-8'))
    checked=[]
    for label,nmin,tmin in [('primary_120_20',120,20),('sensitivity_60_10',60,10)]:
        rows=[]
        for (length,period),g in f.groupby(['length','period']):
            ranks=rankdata(g.x,method='average')/len(g)
            low=ranks<=.25;high=ranks>=.75
            if len(g)<nmin or g.x.nunique()<2 or min(low.sum(),high.sum())<tmin:continue
            y=g.y.to_numpy();rows.append({'length':length,'period':period,'n':len(g),
                'low_n':int(low.sum()),'high_n':int(high.sum()),
                'delta':float(y[high].mean()-y[low].mean())})
        actual=pd.DataFrame(rows).set_index(['length','period']).sort_index()
        expected=pd.read_csv(out/f'{label}_strata.csv').set_index(['length','period']).sort_index()
        assert actual.index.equals(expected.index)
        for column in ['n','low_n','high_n']:assert np.array_equal(actual[column],expected[column])
        assert np.allclose(actual.delta,expected.delta,rtol=0,atol=1e-12)
        pooled=np.average(actual.delta,weights=actual.n)
        assert np.isclose(pooled,stats['rules'][label]['pooled']['delta'],atol=1e-12,rtol=0)
        table=actual.reset_index()
        common=set.intersection(*(set(table[table.period.eq(period)].length)for period in f.period.unique()))
        joined=table[table.length.isin(common)]
        weights=joined.groupby('length').n.sum().astype(float)
        weights=weights/weights.sum()
        assert np.isclose(weights.sum(),1)
        for row in stats['rules'][label]['common_length_periods']:
            a=joined[joined.period.eq(row['period'])].set_index('length').sort_index()
            actual_gap=math.fsum(float(w)*float(d) for w,d in zip(weights.reindex(a.index),a['delta']))
            assert np.isclose(actual_gap,row['delta'],atol=1e-12,rtol=0),(label,row['period'],actual_gap,row['delta'])
        checked.append(f'{label}: all local rank-tail groups, sizes, gaps and shared-length-weight period contrasts')
    # Direct centered dot products independently check the integer-moment writer.
    cells=[]
    for (length,cycle),g in f.groupby(['length','cycle']):
        if len(g)<2 or g.x.nunique()<2:continue
        x=g.x.to_numpy();y=g.y.to_numpy();dx=x-x.mean();dy=y-y.mean()
        sxy=float(np.dot(dx,dy))
        if abs(sxy)<1e-12:sxy=0.0  # Rational zeros; writer uses exact integer signs.
        cells.append({'n':len(g),'cycle':cycle,'sxx':float(np.dot(dx,dx)),'sxy':sxy})
    cells=pd.DataFrame(cells)
    supplement=json.loads((out/'within_cells/summary.json').read_text(encoding='utf-8'))
    for gate in [2,5,10]:
        keep=cells[cells.n>=gate];expected=supplement['support_gate_results'][str(gate)]
        assert len(keep)==expected['eligible_cells'] and int(keep.n.sum())==expected['eligible_support_rows']
        assert np.isclose(.1*keep.sxy.sum()/keep.sxx.sum(),expected['common_within_cell_slope_per_0_1_ttr'],atol=1e-12,rtol=0)
        for sign,mask in [('positive',keep.sxy.gt(0)),('negative',keep.sxy.lt(0)),('zero',keep.sxy.eq(0))]:
            assert int(mask.sum())==expected['cell_direction'][f'{sign}_cells']
            assert int(keep.loc[mask,'n'].sum())==expected['cell_direction'][f'{sign}_rows']
    checked.append('Exact length x cycle: all three support gates, centered slopes, exact-zero directions and row counts')
    export=pd.read_csv(out/'within_cells/exact_length_cycle_cells.csv')
    assert len(export)==2246 and export.n.ge(5).all() and 'record_id'not in export and 'signal_stream'not in export
    # Read-only integrity check of all historical files and paused drafts.
    for kind in ['protected_tracked_sha256','paused_drafts_sha256']:
        for relative,sha in protocol[kind].items():
            assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==sha,relative
    checked.append('382 historical tracked files and8 paused drafts byte-identical; no singleton outcome cells exported')
    result={'status':'PASS','checks':checked,'input_rows':50608,'present_rows':47204,
        'scope':'Targeted numerical reconciliation, not a repeat of unrelated notebook/model QA',
        'limits':'No inferential independence or causality claimed; plots reviewed separately.',
        'new_figure_count':len(list(out.rglob('*.png'))),'official_question_changed':False,
        'model_changes':False,'kaggle_uploads':0}
    assert result['new_figure_count']==3
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('PASS:',len(checked),'targeted checks;3 figures; historical work and paused drafts unchanged.')


if __name__=='__main__':main()
