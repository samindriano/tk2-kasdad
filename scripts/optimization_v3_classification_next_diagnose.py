"""Zero-fit class-only postmortem: ranking, calibration and decision allocation.

Reads official Train labels and verified saved predictions; no threshold search.
Test section reads saved prediction artifacts only, never Test labels.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, log_loss, roc_auc_score

from src.models.research_next_contract import read_bundle

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/optimization_v3/classification_next_v1'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(y,p):
    labels=p>=.5
    return dict(n=len(y),true_positive_rate=float(y.mean()),predicted_positive_rate=float(labels.mean()),
                mean_probability=float(p.mean()),macro_f1=float(f1_score(y,labels,average='macro')),
                roc_auc=float(roc_auc_score(y,p)),average_precision=float(average_precision_score(y,p)),
                log_loss=float(log_loss(y,p,labels=[0,1])),brier=float(brier_score_loss(y,p)),
                fp=int(np.sum((y==0)&labels)),fn=int(np.sum((y==1)&~labels)))


def run():
    frame=pd.read_csv(ROOT/'data/raw/classification/Train.csv',usecols=['record_id','cycle_code','artifact_signal'])
    assert len(frame)==50608 and frame.record_id.is_unique
    paths,origin_data={},{}
    ledger=ROOT/'experiments/runs/optimization_v3_research_next_v1/budget.jsonl';before=sha(ledger)
    for o in (55,59,63,67,71,75):
        rel=(f'experiments/runs/optimization_v3_research_next_v1/N01/pilot/cutoff_{o}/class_only' if o in (55,67) else
             f'experiments/runs/optimization_v3_research_next_v1/N01_CLASS_ONLY_DIAGNOSTIC/amendment_01/fit{o}/class_only')
        folder=ROOT/rel;receipt=json.loads((folder/'receipt.json').read_text())
        ident=receipt['identity'];assert ident['cutoff']==o
        assert ident['card_id']==('N01' if o in (55,67) else 'N01_CLASS_ONLY_DIAGNOSTIC')
        assert ident.get('fit_kind',ident.get('component'))=='class_only'
        a=read_bundle(folder,ident,task='classification').arrays
        rows=a['score_rows'];np.testing.assert_array_equal(a['score_record_ids'].astype(str),frame.record_id.to_numpy()[rows].astype(str))
        np.testing.assert_array_equal(a['cycle_code'],frame.cycle_code.to_numpy()[rows])
        assert ((a['cycle_code']>o)&(a['cycle_code']<=o+12)).all()
        for name in ['arrays.npz','receipt.json']:paths[(folder/name).relative_to(ROOT).as_posix()]=sha(folder/name)
        origin_data[o]=dict(rows=rows,cycles=a['cycle_code'],base=a['baseline_prediction'],candidate=a['prediction'])
    definitions={f'W{i+1}':[[o,o+1,o+4]] for i,o in enumerate(origin_data)}
    definitions.update(primary_fullsix=[[o,o+1,o+4] for o in origin_data],late_primary=[[71,72,75],[75,76,79]],
                       H12_55=[[55,56,67]],H12_67=[[67,68,79]],H12_pooled=[[55,56,67],[67,68,79]])
    views={}
    for name,blocks in definitions.items():
        parts=[]
        for o,lo,hi in blocks:
            a=origin_data[o];mask=(a['cycles']>=lo)&(a['cycles']<=hi);parts.append({k:v[mask] for k,v in a.items()})
        a={k:np.concatenate([x[k] for x in parts]) for k in parts[0]}
        assert len(np.unique(a['rows']))==len(a['rows'])
        y=frame.artifact_signal.to_numpy()[a['rows']]
        m={k:metrics(y,a[k]) for k in ['base','candidate']}
        up=(a['base']<.5)&(a['candidate']>=.5);down=(a['base']>=.5)&(a['candidate']<.5)
        views[name]={'metrics':m,'candidate_minus_base':{k:m['candidate'][k]-m['base'][k] for k in ['macro_f1','roc_auc','average_precision','log_loss','brier','predicted_positive_rate','mean_probability']},
                     'label_flips':dict(up=int(up.sum()),up_true1=int(np.sum(up&(y==1))),up_true0=int(np.sum(up&(y==0))),
                                        down=int(down.sum()),down_true1=int(np.sum(down&(y==1))),down_true0=int(np.sum(down&(y==0))))}
    manifest_path=ROOT/'reports/optimization_v3/measured_benchmark_v1/CLASSIFICATION/CLASS_ONLY_SUBMISSION_MANIFEST.json'
    manifest=json.loads(manifest_path.read_text())
    csvpath=ROOT/manifest['csv']['path_gitignored'];assert sha(csvpath)==manifest['csv']['sha256']
    csv=pd.read_csv(csvpath);cache=ROOT/'experiments/runs/optimization_v3_inference/v3_cls_sparse25.npz'
    assert sha(cache)==manifest['reuse_audit']['cache_sha256']
    with np.load(cache,allow_pickle=False) as z:
        base=z['prediction'].astype(int);candidate=csv.artifact_signal.to_numpy(dtype=int)
        assert len(base)==len(candidate)==29421
        test=dict(n=len(base),base_positive_count=int(base.sum()),candidate_positive_count=int(candidate.sum()),
                  predicted_0_to_1=int(np.sum((base==0)&(candidate==1))),predicted_1_to_0=int(np.sum((base==1)&(candidate==0))),
                  source='saved full-Train inference artifacts only; no true Test labels or loss decomposition available')
    for p in [manifest_path,csvpath,cache,ROOT/'data/raw/classification/Train.csv']:paths[p.relative_to(ROOT).as_posix()]=sha(p)
    assert sha(ledger)==before
    return dict(analysis='fixed descriptive metrics at threshold.50; no candidate/threshold/weight search',
                views=views,Test_prediction_only=test,input_sha256=paths,source_sha256=sha(Path(__file__)),
                public_observation_user_provided={'class_only':.82774,'sparse25_previous_verified':.82908,'delta':-.00134,
                    'source':'user pasted Kaggle Complete row; live receipt pending executor reconciliation; does not reveal error types'},
                official_new_fits=0,Kaggle_calls=0,Test_labels_read=False,ledger_unchanged=True)


if __name__=='__main__':
    result=run();OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'CLASS_ONLY_POSTMORTEM.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'deltas':{k:v['candidate_minus_base'] for k,v in result['views'].items()},'Test_predictions_only':result['Test_prediction_only']},indent=2))
