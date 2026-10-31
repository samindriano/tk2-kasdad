"""Fixed character Ridge independent-regression-channel pilot, no Test/API."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,sys,time,importlib.metadata
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
from src.models.optimization_v3_character import CharacterRidge
from src.models.regression_research_common import mapped_scores
from scripts import optimization_v3_distributional as parent

read,save,sha=parent.read,parent.save,parent.sha
CONFIG=ROOT/'experiments/optimization_v3/regression_character'
OUT=ROOT/'reports/optimization_v3/regression_character'
CACHE=ROOT/'experiments/runs/optimization_v3_regression_character'
CODE=['src/models/optimization_v3_character.py','scripts/optimization_v3_character.py',
      'scripts/optimization_v3_verify_character.py','tests/test_optimization_v3_character.py',
      'scripts/optimization_v3_distributional.py','src/models/regression_research_common.py',
      'src/models/regression_research_text.py','src/models/regression_round2_models.py','src/models/round1_models.py']
def identity():
    _,train=parent.load()
    return{'protocol_sha256':sha(CONFIG/'protocol.json'),'source_sha256':{p:parent.code_digest(ROOT/p)for p in CODE},
        'control_cache_sha256':sha(parent.CONTROL),'control_report_sha256':sha(ROOT/'reports/optimization_v3/regression_diagnostics/diagnostics.json'),
        'officialTrain_sha256':sha(train),'packages':{p:importlib.metadata.version(p)for p in['numpy','pandas','scipy','scikit-learn','lightgbm','threadpoolctl']}}
def freeze():
    cur=identity();assert not CACHE.exists();r={'status':'FROZEN_BEFORE_CHARACTER_FITS','at_utc':datetime.now(timezone.utc).isoformat(),'identity':cur,'signature':parent.digest_json(cur)}
    save(CONFIG/'freeze.json',r);return r
def check():
    r=read(CONFIG/'freeze.json');cur=identity();assert r['identity']==cur and r['signature']==parent.digest_json(cur);return r
def run():
    frozen=check();frame,_=parent.load();y=frame.threat_rank.to_numpy(float);cycles=frame.cycle_code.to_numpy(dtype=int);records={}
    for end in[55,67]:
        rows,old=parent.control(frame,end);fit=frame.cycle_code.between(47,end).to_numpy();model=CharacterRidge();fields=list(model.input_feature_names);start=time.perf_counter()
        with threadpool_limits(limits=2):
            model.fit(frame.loc[fit,fields],y[fit]);state=model.audit_state();p=model.predict(frame.iloc[rows][fields])
        assert state==model.audit_state();mapped=mapped_scores(p,cycles[rows],.5);blend=np.clip(.75*old+.25*mapped,0,100)
        CACHE.mkdir(parents=True,exist_ok=True);path=CACHE/f'fit{end}.npz';assert not path.exists()
        np.savez_compressed(path,rows=rows,raw=p,rank50=mapped,blend25=blend,champion=old)
        save(path.with_suffix('.json'),{'freeze_signature':frozen['signature'],'state':state,'npz_sha256':sha(path),'seconds':time.perf_counter()-start})
        records[end]={'rows':rows,'raw':p,'rank50':mapped,'blend25':blend,'champion':old}
        print('FITTED charRidge',end,round(time.perf_counter()-start,1),flush=True)
    views={};primary=[];h=[]
    for end,name in[(55,'W1'),(67,'W4')]:
        r=records[end];use=cycles[r['rows']]<=end+4;part={k:v[use]for k,v in r.items()};primary.append(part);h.append(r)
        for label,d in[(name,part),('H12_E'if end==55 else'H12_L',r)]:views[label]={k:parent.metrics(frame,d['rows'],d[k])for k in['raw','rank50','blend25','champion']}
    for label,parts in[('primary_pilot',primary),('H12',h)]:
        rows=np.concatenate([r['rows']for r in parts]);assert len(np.unique(rows))==len(rows)
        views[label]={k:parent.metrics(frame,rows,np.concatenate([r[k]for r in parts]))for k in['raw','rank50','blend25','champion']}
    gates={}
    for k in['rank50','blend25']:
        delta={name:v[k]['mae']-v['champion']['mae']for name,v in views.items()}
        gates[k]={'delta':delta,'pass':bool((delta['primary_pilot']<0 or delta['H12']<0)and max(delta[n]for n in['W1','W4','H12_E','H12_L'])<=.5)}
    r={'status':'PILOT_REVIEW_REQUIRED','freeze_signature':frozen['signature'],'fits':2,'Test_read':False,'views':views,'gates':gates}
    save(OUT/'pilot.json',r);print('GATES',gates)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');a=p.parse_args()
    if a.freeze:print(freeze()['signature'])
    else:run()
