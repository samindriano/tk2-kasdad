"""Single frozen zero-fit blend review. Reads Classification Train and caches only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.research_next_contract import read_bundle
from src.models.round1_models import _number_array

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'experiments/optimization_v3/research_next_v1/review_class_blend/protocol.json'
OUT = ROOT / 'reports/optimization_v3/research_next_v1/results/CLASS_BLEND_REVIEW/amendment_01'
ARMS = ('sparse25', 'N01_class_only', 'N02_sketch', 'matched_combo', 'candidate')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metric(y, p):
    y, p = np.asarray(y), np.asarray(p)
    assert y.shape == p.shape and y.ndim == 1 and np.isin(y, [0, 1]).all()
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    z = p >= .5
    tn, fp = int(np.sum((y == 0) & ~z)), int(np.sum((y == 0) & z))
    fn, tp = int(np.sum((y == 1) & ~z)), int(np.sum((y == 1) & z))
    ratio = lambda n, d: n / d if d else 0.
    return dict(n=len(y), tn=tn, fp=fp, fn=fn, tp=tp,
                macro_f1=.5 * (ratio(2*tn, 2*tn+fp+fn) + ratio(2*tp, 2*tp+fp+fn)),
                positive_recall=ratio(tp, tp+fn), positive_precision=ratio(tp, tp+fp))


def gate_result(views, groups, gates):
    primary = views['primary']['metrics']
    value = primary['candidate']['macro_f1']
    checks = {'primary_vs_sparse25': value-primary['sparse25']['macro_f1'] >= gates['primary_gain_vs_sparse25_min'],
              'primary_vs_class_only': value-primary['N01_class_only']['macro_f1'] >= gates['primary_gain_vs_each_component_min'],
              'primary_vs_sketch': value-primary['N02_sketch']['macro_f1'] >= gates['primary_gain_vs_each_component_min'],
              'primary_vs_matched_combo': value-primary['matched_combo']['macro_f1'] >= gates['primary_gain_vs_matched_combo_min']}
    for name in gates['h12_scope']:
        m = views[name]['metrics']
        checks[name+'_aggregate_harm'] = m['sparse25']['macro_f1']-m['candidate']['macro_f1'] <= gates['h12_harm_vs_sparse25_max']
    for name in ('W1','W4'):
        m = views[name]['metrics']
        checks[name+'_aggregate_harm'] = m['sparse25']['macro_f1']-m['candidate']['macro_f1'] <= gates['individual_primary_window_harm_max']
    failed = [r for r in groups if r['arm']=='candidate' and r['supported'] and not r['guard_pass']]
    checks['all_view_supported_subgroup_guards'] = not failed
    return {'checks':checks, 'passed':all(checks.values()), 'failed_groups':failed,
            'decision':'GO_CONTINUATION_REVIEW' if all(checks.values()) else 'NO_GO_FURTHER_FITS_FOR_THIS_BLEND'}


def load_origin(cutoff, protocol, frame):
    bundles = {}
    for card, kind in (('N01','class_only'),('N02','sketch'),('N02','no_sketch')):
        path = ROOT / f'experiments/runs/optimization_v3_research_next_v1/{card}/pilot/cutoff_{cutoff}/{kind}'
        frozen_path = ('experiments/optimization_v3/research_next_v1/frozen/N01/freeze.json' if card=='N01' else
                       'experiments/optimization_v3/research_next_v1/frozen/N02/amendment_02/freeze.json')
        frozen = json.loads((ROOT/frozen_path).read_text(encoding='utf-8'))
        identity = dict(batch_id='optimization_v3_research_next_v1',card_id=card,cutoff=cutoff,
                        fit_kind=kind,freeze_signature=frozen['freeze_signature'],stage='PILOT')
        arrays = read_bundle(path,identity,task='classification').arrays
        rows = arrays['score_rows']
        assert np.array_equal(rows,np.flatnonzero(frame.cycle_code.between(cutoff+1,cutoff+12)))
        assert np.array_equal(arrays['score_record_ids'].astype(str),frame.record_id.to_numpy()[rows].astype(str))
        assert np.array_equal(arrays['cycle_code'],frame.cycle_code.to_numpy()[rows])
        assert (arrays['origin_cutoff']==cutoff).all()
        assert not np.any(frame.cycle_code.to_numpy()[rows] <= cutoff)
        missing=frame.signal_stream.fillna('').astype(str).str.strip().eq('').to_numpy()[rows]
        assert np.array_equal(arrays['stream_missing'],missing)
        if card=='N01':
            base=np.where(missing,arrays['control__cb_probability'],
                          .75*arrays['control__c2b_probability']+.25*arrays['control__sparse_baseline_raw'])
            reconstructed=np.where(missing,arrays['control__cb_probability'],
                                   .75*arrays['control__c2b_probability']+.25*arrays['raw_prediction'])
        else:
            base=arrays['control__current_route']
            reconstructed=np.where(missing,arrays['control__cb_probability'],.75*base+.25*arrays['raw_prediction'])
        np.testing.assert_allclose(base,arrays['baseline_prediction'],atol=1e-12,rtol=0)
        np.testing.assert_allclose(reconstructed,arrays['prediction'],atol=1e-12,rtol=0)
        bundles[kind]=arrays
    co, sk, ns = (bundles[k] for k in ('class_only','sketch','no_sketch'))
    for other in (sk,ns):
        for key in ('score_rows','score_record_ids','cycle_code','stream_missing'):
            np.testing.assert_array_equal(co[key],other[key])
        np.testing.assert_allclose(co['baseline_prediction'],other['baseline_prediction'],atol=1e-12,rtol=0)
        np.testing.assert_array_equal(co['control__cb_probability'],other['control__cb_probability'])
    candidate=.5*co['prediction']+.5*sk['prediction']
    matched=.5*co['prediction']+.5*ns['prediction']
    missing=co['stream_missing']
    np.testing.assert_array_equal(candidate[missing],co['control__cb_probability'][missing])
    return dict(rows=co['score_rows'],cycles=co['cycle_code'],missing=missing,
                origin=np.full(len(candidate),cutoff),sparse25=co['baseline_prediction'],
                N01_class_only=co['prediction'],N02_sketch=sk['prediction'],
                matched_combo=matched,candidate=candidate)


def collect(outputs,blocks):
    parts=[]
    for origin,lo,hi in blocks:
        record=outputs[origin]
        mask=(record['cycles']>=lo)&(record['cycles']<=hi)
        parts.append({k:v[mask] for k,v in record.items()})
    result={k:np.concatenate([r[k] for r in parts]) for k in parts[0]}
    order=np.argsort(result['rows'])
    result={k:v[order] for k,v in result.items()}
    assert len(np.unique(result['rows']))==len(result['rows'])
    return result


def analyze(protocol):
    for path,expected in protocol['input_sha256'].items():
        assert digest(ROOT/path)==expected, f'Input changed: {path}'
    ledger=ROOT/protocol['budget_ledger']
    assert digest(ledger)==protocol['ledger_sha256_before']
    frame=pd.read_csv(ROOT/'data/raw/classification/Train.csv')
    assert len(frame)==50608 and frame.record_id.is_unique
    y=frame.artifact_signal.to_numpy(dtype=int)
    assert np.isin(y,[0,1]).all()
    outputs={o:load_origin(o,protocol,frame) for o in protocol['origins']}
    text,_,_=_number_array(frame.text_load)
    bins,edges={},{ }
    for origin in protocol['origins']:
        fit=text[(frame.cycle_code.to_numpy()<=origin)&np.isfinite(text)]
        edges[origin]=np.unique(np.quantile(fit,[.25,.5,.75]))
        bins[origin]=np.full(len(frame),-1)
        finite=np.isfinite(text)
        bins[origin][finite]=np.searchsorted(edges[origin],text[finite],side='right')
    views,groups,cycles={ },[],[]
    g=protocol['gates']
    for name,blocks in protocol['views'].items():
        data=collect(outputs,blocks);rows=data['rows'];truth=y[rows]
        assert len(rows)==protocol['view_rows'][name]
        m={arm:metric(truth,data[arm]) for arm in ARMS}
        co,sk=data['N01_class_only']>=.5,data['N02_sketch']>=.5
        correct_co,correct_sk=co==truth,sk==truth
        views[name]={'n':len(rows),'metrics':m,
                     'candidate_gains':{a:m['candidate']['macro_f1']-m[a]['macro_f1'] for a in ARMS if a!='candidate'},
                     'component_disagreement':int(np.sum(co!=sk)),
                     'both_correct':int(np.sum(correct_co&correct_sk)),
                     'only_class_only_correct':int(np.sum(correct_co&~correct_sk)),
                     'only_sketch_correct':int(np.sum(~correct_co&correct_sk)),
                     'both_wrong':int(np.sum(~correct_co&~correct_sk)),
                     'rows_sha256':hashlib.sha256(rows.astype('<i8').tobytes()).hexdigest()}
        labels={'entry_friction':frame.iloc[rows].entry_friction.astype(str).to_numpy(),
                'stream_availability':np.where(data['missing'],'missing','present'),
                'fit_defined_text_load_quartile':np.array([str(bins[int(o)][int(r)]) for o,r in zip(data['origin'],rows)])}
        for field,values in labels.items():
            for label in np.unique(values):
                mask=values==label;yg=truth[mask]
                n0,n1=int(np.sum(yg==0)),int(np.sum(yg==1))
                supported=len(yg)>=g['minimum_group_n'] and min(n0,n1)>=g['minimum_each_class_n']
                base=metric(yg,data['sparse25'][mask])
                for arm in ARMS:
                    gm=metric(yg,data[arm][mask])
                    f1harm=base['macro_f1']-gm['macro_f1'];rharm=base['positive_recall']-gm['positive_recall']
                    groups.append(dict(view=name,field=field,group=str(label),arm=arm,n=len(yg),n0=n0,n1=n1,
                                       supported=bool(supported),macro_f1=gm['macro_f1'],positive_recall=gm['positive_recall'],
                                       f1_harm=f1harm,recall_harm=rharm,
                                       guard_pass=bool(not supported or (f1harm<=g['subgroup_f1_harm_max'] and rharm<=g['subgroup_positive_recall_harm_max']))))
        if name in ('H12_55','H12_67'):
            for cycle in np.unique(data['cycles']):
                mask=data['cycles']==cycle
                for arm in ARMS:
                    cm=metric(truth[mask],data[arm][mask])
                    cycles.append(dict(origin=blocks[0][0],cycle=int(cycle),arm=arm,**cm))
    gate=gate_result(views,groups,g)
    assert digest(ledger)==protocol['ledger_sha256_before'], 'Ledger changed during zero-fit review'
    return dict(id=protocol['id'],protocol_sha256=digest(PROTOCOL),source_sha256=digest(Path(__file__)),
                views=views,gate=gate,fit_text_edges={str(k):v.tolist() for k,v in edges.items()},
                official_estimator_fits=0,test_access=False,kaggle_access=False,
                ledger_unchanged=True,ledger_sha256=digest(ledger)),groups,cycles


def write_once(path,content):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        assert path.read_bytes()==content,f'Refusing overwrite: {path}'
    else:
        with path.open('xb') as f:f.write(content)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args()
    protocol=json.loads(PROTOCOL.read_text(encoding='utf-8'))
    result,groups,cycles=analyze(protocol)
    write_once(args.output/'results.json',(json.dumps(result,indent=2,allow_nan=False)+'\n').encode())
    for name,rows in [('groups.csv',groups),('per_cycle.csv',cycles)]:
        stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        write_once(args.output/name,stream.getvalue().encode())
    print(json.dumps({'decision':result['gate'],'views':{k:v['candidate_gains'] for k,v in result['views'].items()}},indent=2))


if __name__=='__main__':main()
