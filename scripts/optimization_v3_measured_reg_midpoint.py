"""One pre-frozen, zero-fit R06/R22 midpoint check on official Train caches."""
from __future__ import annotations
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.optimization_v3_research_next_preflight import historical_bundle, rank50, source_paths

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'experiments/optimization_v3/measured_benchmark_v1/reg_midpoint_protocol.json'
OUT=ROOT/'reports/optimization_v3/measured_benchmark_v1/REG_MIDPOINT'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect(outputs,blocks):
    parts=[]
    for origin,lo,hi in blocks:
        r=outputs[origin];mask=(r['cycles']>=lo)&(r['cycles']<=hi)
        parts.append({k:v[mask] for k,v in r.items()})
    result={k:np.concatenate([r[k] for r in parts]) for k in parts[0]}
    order=np.argsort(result['rows']);result={k:v[order] for k,v in result.items()}
    assert len(np.unique(result['rows']))==len(result['rows'])
    return result


def evaluate(protocol):
    for path,expected in protocol['input_sha256'].items():assert sha(ROOT/path)==expected,path
    assert sha(ROOT/protocol['ledger_path'])==protocol['ledger_sha256_before']
    frame=pd.read_csv(ROOT/'data/raw/regression/Train.csv')
    assert len(frame)==50608 and frame.record_id.is_unique
    target=frame.threat_rank.to_numpy(dtype=float)
    missing=frame.signal_stream.fillna('').astype(str).str.strip().eq('').to_numpy()
    outputs={}
    for origin in protocol['origins']:
        data={}
        for card in ('R06','R22'):
            a,_=historical_bundle(source_paths(origin)[card]);rows=a['score_rows']
            assert rows.ndim==1 and np.array_equal(rows,np.sort(np.unique(rows)))
            assert ((rows>=0)&(rows<len(frame))).all()
            np.testing.assert_array_equal(a['score_record_ids'].astype(str),frame.record_id.to_numpy()[rows].astype(str))
            cycles=frame.cycle_code.to_numpy()[rows]
            assert ((cycles>origin)&(cycles<=min(origin+12,79))).all()
            for c in np.unique(cycles):assert np.sum(cycles==c)==np.sum(frame.cycle_code==c)
            np.testing.assert_allclose(a['A'],np.clip(.75*a['legacy']+.25*rank50(a['raw_candidate'],cycles),0,100),atol=1e-12,rtol=0)
            data[card]=a
        r6,r22=data['R06'],data['R22']
        common,i,j=np.intersect1d(r6['score_rows'],r22['score_rows'],return_indices=True)
        np.testing.assert_array_equal(r6['legacy'][i],r22['legacy'][j])
        expected=np.flatnonzero(frame.cycle_code.between(origin+1,origin+4))
        assert np.isin(expected,common).all()
        if origin in (55,67):assert len(common)==np.sum(frame.cycle_code.between(origin+1,origin+12))
        midpoint=.5*r6['A'][i]+.5*r22['A'][j]
        assert np.isfinite(midpoint).all() and ((midpoint>=0)&(midpoint<=100)).all()
        outputs[origin]=dict(rows=common,cycles=frame.cycle_code.to_numpy()[common],
                             legacy=r6['legacy'][i],R06=r6['A'][i],R22=r22['A'][j],midpoint=midpoint)
    definitions={f'W{k+1}':[b] for k,b in enumerate(protocol['primary_blocks'])}
    definitions.update(primary=protocol['primary_blocks'],late_primary=protocol['primary_blocks'][-2:],
                       H12_55=[protocol['H12_blocks'][0]],H12_67=[protocol['H12_blocks'][1]],H12_pooled=protocol['H12_blocks'])
    views,groups={},[]
    g=protocol['gates']
    for name,blocks in definitions.items():
        r=collect(outputs,blocks);rows=r['rows'];y=target[rows]
        maes={arm:float(np.mean(np.abs(y-r[arm]))) for arm in ['legacy','R06','R22','midpoint']}
        gains={arm:maes[arm]-maes['midpoint'] for arm in ['legacy','R06','R22']}
        views[name]=dict(n=len(rows),metrics=maes,gains_vs=gains,rows_sha256=hashlib.sha256(rows.astype('<i8').tobytes()).hexdigest())
        masks=dict(stream_missing=missing[rows],target_le5_evaluation_only=y<=5,target_ge95_evaluation_only=y>=95)
        for field,mask in masks.items():
            n=int(mask.sum());supported=n>=g['minimum_subgroup_n']
            gm={a:float(np.mean(np.abs(y[mask]-r[a][mask]))) if n else None for a in ['legacy','R06','R22','midpoint']}
            harm={a:gm['midpoint']-gm[a] if n else None for a in ['legacy','R06','R22']}
            groups.append(dict(view=name,field=field,n=n,supported=supported,
                               **{f'mae_{a}':v for a,v in gm.items()},**{f'harm_vs_{a}':v for a,v in harm.items()},
                               guard_pass=not supported or all(v<=g['subgroup_harm_vs_each_baseline_max'] for v in harm.values())))
    assert views['primary']['n']==protocol['primary_rows']
    assert views['H12_pooled']['n']==protocol['H12_rows']
    assert views['late_primary']['n']==protocol['late_rows']
    checks=dict(primary_vs_R06=views['primary']['gains_vs']['R06']>=g['primary_gain_vs_R06_min'],
                primary_vs_R22=views['primary']['gains_vs']['R22']>=g['primary_gain_vs_R22_min'],
                late_vs_R06=views['late_primary']['gains_vs']['R06']>=g['late_gain_vs_R06_min'],
                primary_nonnegative_windows=sum(views[f'W{k}']['gains_vs']['R06']>=0 for k in range(1,7))>=g['nonnegative_primary_windows_vs_R06_min'],
                individual_window_harm=all(views[f'W{k}']['gains_vs']['R06']>=-g['individual_window_harm_vs_R06_max'] for k in range(1,7)))
    for view in ['H12_55','H12_67','H12_pooled']:
        for arm in ['R06','R22']:checks[f'{view}_harm_vs_{arm}']=views[view]['gains_vs'][arm]>=-g['H12_harm_vs_each_component_max']
    failures=[r for r in groups if not r['guard_pass']]
    checks['all_supported_group_guards']=not failures
    assert sha(ROOT/protocol['ledger_path'])==protocol['ledger_sha256_before']
    result=dict(protocol_sha256=sha(PROTOCOL),evaluator_sha256=sha(Path(__file__)),views=views,
                gate=dict(checks=checks,passed=all(checks.values()),decision='BENCHMARK_CANDIDATE_PENDING_ROOT_REVIEW' if all(checks.values()) else 'NO_GO_MIDPOINT_SUBMISSION',failed_groups=failures),
                official_new_fits=0,Test_access=False,Kaggle_access=False,ledger_unchanged=True)
    return result,groups


def write_once(path,content):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():assert path.read_bytes()==content,f'Refusing overwrite: {path}'
    else:
        with path.open('xb') as f:f.write(content)


def main():
    protocol=json.loads(PROTOCOL.read_text())
    result,groups=evaluate(protocol)
    write_once(OUT/'result.json',(json.dumps(result,indent=2,allow_nan=False)+'\n').encode())
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(groups[0]));writer.writeheader();writer.writerows(groups)
    write_once(OUT/'groups.csv',stream.getvalue().encode())
    print(json.dumps({'gate':result['gate'],'gains':{k:v['gains_vs'] for k,v in result['views'].items()}},indent=2))


if __name__=='__main__':main()
