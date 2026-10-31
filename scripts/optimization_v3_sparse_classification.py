"""Two-cutoff fixed sparse-binary representation pilot; no Test or Kaggle."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,importlib.metadata,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
from src.models.optimization_v3_sparse_classification import SparseTextBoostClassifier
from scripts.optimization_v2_classification import existing,load_train,score_rows
from src.models.classification_round2_models import stream_missing_mask
from src.models.round1_evaluation import code_digest,digest_json,array_digest

CONFIG=ROOT/'experiments/optimization_v3/classification_sparse'
OUT=ROOT/'reports/optimization_v3/classification_sparse'
CACHE=ROOT/'experiments/runs/optimization_v3_classification_sparse'
CODE=['src/models/optimization_v3_sparse_classification.py','scripts/optimization_v3_sparse_classification.py',
      'scripts/optimization_v3_verify_sparse_classification.py','tests/test_optimization_v3_sparse_classification.py',
      'src/models/classification_models.py','src/models/regression_research_text.py','src/models/round1_models.py',
      'src/models/regression_round2_models.py','scripts/optimization_v2_classification.py']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n')as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n')
def identity():
    _,_,train=load_train(None);paths=[train]
    for end in[55,67]:
        paths.extend([ROOT/f'experiments/runs/dual_cls_catboost_v1/fit{end}.{ext}'for ext in['npz','json']])
        paths.extend([ROOT/f'experiments/runs/classification_round1_v1/fit{end}__C2_B.npz',ROOT/f'reports/classification_round1/fit_audits/fit{end}.json'])
        paths.extend([ROOT/f'experiments/runs/dual_cls_v1a/fit{end}__DC_NBSVM.npz'])
    return {'protocol_sha256':sha(CONFIG/'protocol.json'),'code_sha256':{p:code_digest(ROOT/p)for p in CODE},
            'inputs':{str(p.relative_to(ROOT)):sha(p)for p in paths},
            'packages':{p:importlib.metadata.version(p)for p in['numpy','pandas','scipy','scikit-learn','lightgbm','threadpoolctl']}}
def freeze():
    cur=identity();assert not CACHE.exists();d={'status':'FROZEN_BEFORE_REAL_FITS','at_utc':datetime.now(timezone.utc).isoformat(),'identity':cur,'signature':digest_json(cur)}
    save(CONFIG/'freeze.json',d);return d
def check():
    d=read(CONFIG/'freeze.json');cur=identity();assert d['identity']==cur and d['signature']==digest_json(cur);return d
def score(frame,y,rows,prob):
    m=stream_missing_mask(frame.iloc[rows]);result={}
    for kind,p in prob.items():result[kind]={'all':score_rows(frame,y,rows,p),'present':score_rows(frame,y,rows[~m],p[~m]),'missing':score_rows(frame,y,rows[m],p[m])}
    return result
def run():
    frozen=check();frame,y,_=load_train(None);cycles=frame.cycle_code.to_numpy(dtype=int);records={};states={}
    for end in[55,67]:
        path=CACHE/f'fit{end}.npz';assert not path.exists();rows,ss,cb=existing(frame,y,end);missing=stream_missing_mask(frame.iloc[rows])
        fit=frame.cycle_code.between(47,end).to_numpy();model=SparseTextBoostClassifier();fields=list(model.input_feature_names);start=time.perf_counter()
        with threadpool_limits(limits=2):
            model.fit(frame.loc[fit,fields],y[fit]);state=model.audit_state();p=model.predict_proba(frame.iloc[rows][fields])
        assert state==model.audit_state();champ=np.where(missing,cb,ss[:,0]);direct=np.where(missing,cb,p);blend=np.where(missing,cb,.75*ss[:,0]+.25*p)
        CACHE.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,rows=rows,champion=champ,direct=direct,blend25=blend,raw_candidate=p)
        save(path.with_suffix('.json'),{'freeze_signature':frozen['signature'],'state':state,'npz_sha256':sha(path),'seconds':time.perf_counter()-start})
        records[end]={'rows':rows,'champion':champ,'direct':direct,'blend25':blend};states[str(end)]=state
        print('FITTED sparsebinary',end,round(time.perf_counter()-start,1),flush=True)
    views={};primary=[];h=[]
    for end,name in[(55,'W1'),(67,'W4')]:
        r=records[end];use=cycles[r['rows']]<=end+4
        views[name]=score(frame,y,r['rows'][use],{k:r[k][use]for k in['champion','direct','blend25']})
        primary.append({k:v[use]for k,v in r.items()});h.append(r)
        views['H12_E'if end==55 else'H12_L']=score(frame,y,r['rows'],{k:r[k]for k in['champion','direct','blend25']})
    def pooled(records):
        rr=np.concatenate([r['rows']for r in records]);assert len(np.unique(rr))==len(rr)
        return score(frame,y,rr,{k:np.concatenate([r[k]for r in records])for k in['champion','direct','blend25']})
    views['primary_pilot']=pooled(primary);views['H12']=pooled(h);gates={}
    for kind in['direct','blend25']:
        delta={name:v[kind]['all']['macro_f1']-v['champion']['all']['macro_f1']for name,v in views.items()}
        gates[kind]={'delta':delta,'pass':bool((delta['primary_pilot']>0 or delta['H12']>0)and min(delta[k]for k in['W1','W4','H12_E','H12_L'])>=-.01)}
    result={'status':'PILOT_REVIEW_REQUIRED','freeze_signature':frozen['signature'],'newfits':2,'Test_read':False,'states':states,'views':views,'continuation_gates':gates}
    save(OUT/'pilot.json',result);print('GATES',gates)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');a=p.parse_args()
    if a.freeze:print(freeze()['signature'])
    else:run()
