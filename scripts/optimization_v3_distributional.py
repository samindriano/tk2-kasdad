"""Ten-bin CDF-median objective pilot against the V3 three-component incumbent."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,importlib.metadata,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src.models.optimization_v3_distributional import DistributionalRegressor
from src.models.round1_evaluation import code_digest,digest_json,array_digest
from src.utils.paths import task_files

CONFIG=ROOT/'experiments/optimization_v3/regression_distributional'
OUT=ROOT/'reports/optimization_v3/regression_distributional'
CACHE=ROOT/'experiments/runs/optimization_v3_regression_distributional'
CONTROL=ROOT/'experiments/runs/optimization_v3_reg_diagnostics/oof_residuals.npz'
CODE=['src/models/optimization_v3_distributional.py','scripts/optimization_v3_distributional.py',
    'scripts/optimization_v3_verify_distributional.py','tests/test_optimization_v3_distributional.py',
    'src/models/regression_research_text.py','src/models/regression_round2_models.py','src/models/round1_models.py']

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n')as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n')
def load():
    path=task_files('regression')['train'];frame=pd.read_csv(path)
    assert len(frame)==50608 and set(frame.cycle_code)==set(range(47,80))
    return frame,path
def identity():
    _,train=load();inputs=[train,CONTROL,ROOT/'reports/optimization_v3/regression_diagnostics/diagnostics.json',
        ROOT/'experiments/runs/dual_reg_v1/stage1_paired_blends.npz',
        ROOT/'experiments/runs/optimization_v2_regression_repr/pilot_predictions.npz',
        ROOT/'experiments/runs/optimization_v2_regression_repr/fullsix_predictions.npz']
    return {'protocol_sha256':sha(CONFIG/'protocol.json'),'source_sha256':{p:code_digest(ROOT/p)for p in CODE},
            'inputs':{str(p.relative_to(ROOT)):sha(p)for p in inputs},
            'packages':{p:importlib.metadata.version(p)for p in['numpy','pandas','scikit-learn','lightgbm','threadpoolctl']}}
def freeze():
    current=identity();assert not CACHE.exists();d={'status':'FROZEN_BEFORE_DISTRIBUTIONAL_FITS','at_utc':datetime.now(timezone.utc).isoformat(),'identity':current,'signature':digest_json(current)}
    save(CONFIG/'freeze.json',d);return d
def check():
    d=read(CONFIG/'freeze.json');cur=identity();assert d['identity']==cur and d['signature']==digest_json(cur);return d
def control(frame,end):
    with np.load(CONTROL,allow_pickle=False)as z:
        rows=z['h12_rows'];cycles=z['h12_cycle'];use=(cycles>=end+1)&(cycles<={55:67,67:79}[end]);rows=rows[use].copy();p=z['h12_prediction'][use].copy()
    np.testing.assert_array_equal(rows,np.flatnonzero(frame.cycle_code.between(end+1,{55:67,67:79}[end])));return rows,p
def metrics(frame,rows,p):
    y=frame.iloc[rows].threat_rank.to_numpy(float);e=p-y;cy=frame.iloc[rows].cycle_code.to_numpy()
    stream=frame.iloc[rows].signal_stream;missing=stream.isna().to_numpy()|stream.fillna('').str.strip().eq('').to_numpy()
    ef=frame.iloc[rows].entry_friction.to_numpy()
    masks={'tail_le5':y<=5,'tail_ge95':y>=95,'EF0':ef==0,'EF1':ef==1,'stream_missing':missing,'stream_present':~missing}
    def summary(use):return{'n':int(use.sum()),'mae':float(np.mean(np.abs(e[use]))),'bias':float(np.mean(e[use]))}
    return{'n':len(rows),'mae':float(np.mean(np.abs(e))),'RMSE':float(np.sqrt(np.mean(e*e))),'bias':float(e.mean()),
           'equal_cycle_mae':float(np.mean([np.mean(np.abs(e[cy==c]))for c in np.unique(cy)])),
           'per_cycle':{str(c):summary(cy==c)for c in np.unique(cy)},'subgroups':{k:summary(m)for k,m in masks.items()}}
def run():
    frozen=check();frame,_=load();y=frame.threat_rank.to_numpy(float);cy=frame.cycle_code.to_numpy(dtype=int);records={};states={}
    for end in[55,67]:
        rows,old=control(frame,end);fit=frame.cycle_code.between(47,end).to_numpy();model=DistributionalRegressor();fields=list(model.input_feature_names);start=time.perf_counter()
        with threadpool_limits(limits=2):
            model.fit(frame.loc[fit,fields],y[fit]);state=model.audit_state();probs=model.predict_distribution(frame.iloc[rows][fields]);p=model.predict(frame.iloc[rows][fields])
        assert state==model.audit_state();blend=np.clip(.75*old+.25*p,0,100)
        CACHE.mkdir(parents=True,exist_ok=True);path=CACHE/f'fit{end}.npz';assert not path.exists()
        np.savez_compressed(path,rows=rows,probabilities=probs,raw=p,blend25=blend,champion=old)
        save(path.with_suffix('.json'),{'freeze_signature':frozen['signature'],'state':state,'npz_sha256':sha(path),'seconds':time.perf_counter()-start})
        records[end]={'rows':rows,'raw':p,'blend25':blend,'champion':old};states[str(end)]=state
        print('FITTED distribution10',end,round(time.perf_counter()-start,1),flush=True)
    views={};primary=[];h=[]
    for end,name in[(55,'W1'),(67,'W4')]:
        r=records[end];use=cy[r['rows']]<=end+4
        primary.append({k:v[use]for k,v in r.items()});h.append(r)
        for view,part in[(name,primary[-1]),('H12_E'if end==55 else'H12_L',r)]:
            views[view]={kind:metrics(frame,part['rows'],part[kind])for kind in['champion','raw','blend25']}
    for name,parts in[('primary_pilot',primary),('H12',h)]:
        rows=np.concatenate([p['rows']for p in parts]);assert len(np.unique(rows))==len(rows)
        views[name]={kind:metrics(frame,rows,np.concatenate([p[kind]for p in parts]))for kind in['champion','raw','blend25']}
    gates={}
    for kind in['raw','blend25']:
        delta={name:v[kind]['mae']-v['champion']['mae']for name,v in views.items()}
        gates[kind]={'delta':delta,'pass':bool((delta['primary_pilot']<0 or delta['H12']<0)and max(delta[k]for k in['W1','W4','H12_E','H12_L'])<=.5)}
    result={'status':'PILOT_REVIEW_REQUIRED','freeze_signature':frozen['signature'],'fits':2,'Test_read':False,
            'states':states,'views':views,'gates':gates,'caveat':'TenbinCDFinterpolation approximates uniformwithinbin conditionaldensity; diagnostic tails do not imply conditionalmultimodality.'}
    save(OUT/'pilot.json',result);print('GATES',gates)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');a=p.parse_args()
    if a.freeze:print(freeze()['signature'])
    else:run()
