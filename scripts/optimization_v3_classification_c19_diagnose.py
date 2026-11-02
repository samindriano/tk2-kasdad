"""Read-only ranking/calibration diagnosis of the existing C19 pilot."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.optimization_v3_classification_next_diagnose import metrics,sha,ROOT,OUT
from scripts.optimization_v3_research_next_preflight import historical_bundle


def run():
    train=ROOT/'data/raw/classification/Train.csv'
    frame=pd.read_csv(train,usecols=['record_id','cycle_code','artifact_signal'])
    target=frame.artifact_signal.to_numpy();outputs={};hashes={train.relative_to(ROOT).as_posix():sha(train)}
    for o in [55,67]:
        path=ROOT/f'experiments/runs/optimization_v3_batch1/C19/C19_fit{o}.npz'
        a,_=historical_bundle(path);rows=a['score_rows'];cycles=frame.cycle_code.to_numpy()[rows]
        np.testing.assert_array_equal(a['score_record_ids'].astype(str),frame.record_id.to_numpy()[rows].astype(str))
        np.testing.assert_array_equal(rows,np.flatnonzero(frame.cycle_code.between(o+1,o+12)))
        with np.load(path,allow_pickle=False) as z:
            missing=z['control__stream_missing'];cb=z['control__cb_probability']
            reconstructed=np.where(missing,cb,.75*a['legacy']+.25*a['raw_candidate'])
            np.testing.assert_allclose(a['A'],reconstructed,atol=1e-12,rtol=0)
        outputs[o]=dict(rows=rows,cycles=cycles,base=a['legacy'],candidate=a['A'])
        for f in [path,path.with_suffix('.json')]:hashes[f.relative_to(ROOT).as_posix()]=sha(f)
    definitions={'primary':[(55,56,59),(67,68,71)],'H12_pooled':[(55,56,67),(67,68,79)],
                 'W1':[(55,56,59)],'W4':[(67,68,71)],'H12_55':[(55,56,67)],'H12_67':[(67,68,79)]}
    views={}
    for name,blocks in definitions.items():
        parts=[]
        for o,lo,hi in blocks:
            r=outputs[o];m=(r['cycles']>=lo)&(r['cycles']<=hi);parts.append({k:v[m] for k,v in r.items()})
        a={k:np.concatenate([v[k] for v in parts]) for k in parts[0]};assert len(np.unique(a['rows']))==len(a['rows'])
        y=target[a['rows']];m={arm:metrics(y,a[arm]) for arm in ['base','candidate']}
        views[name]={'metrics':m,'candidate_minus_base':{k:m['candidate'][k]-m['base'][k] for k in ['macro_f1','roc_auc','average_precision','log_loss','brier','predicted_positive_rate']}}
    return dict(analysis='existing C19 recipe only, descriptive; no new recipe or threshold tuning',views=views,input_sha256=hashes,
                source_sha256=sha(Path(__file__)),helper_sha256=sha(ROOT/'scripts/optimization_v3_classification_next_diagnose.py'),official_new_fits=0,Test_access=False,Kaggle_calls=0)


if __name__=='__main__':
    r=run();OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'C19_DIAGNOSTIC.json').open('x',encoding='utf-8') as f:json.dump(r,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:v['candidate_minus_base'] for k,v in r['views'].items()},indent=2))
