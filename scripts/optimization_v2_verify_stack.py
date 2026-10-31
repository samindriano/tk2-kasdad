"""Independent score-head reconstruction from strictly past OOF vectors."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,confusion_matrix
from scripts import optimization_v2_stack as run
from scripts.verify_campaign_classification_stage2 import _assert_metric

def verify():
    freeze=run.check();report=run.read(run.OUT/'results.json');frame,y,_=run.parent.load_train(None)
    cycles=frame.cycle_code.to_numpy(dtype=int)
    fr,fs=run.parent.earliest(frame,y,run.parent.check()['signature'],False)
    cr,cp=run.inner51_cb(frame,y,freeze['signature'],False);np.testing.assert_array_equal(fr,cr)
    first=(fr,np.column_stack([fs[:,0],cp]));old={e:run.parent.existing(frame,y,e)for e in run.parent.ENDS}
    records={};head_checks=0
    for end in run.parent.ENDS:
        parts=[first]
        for earlier in run.parent.ENDS:
            if earlier>=end:break
            rows,scores,cb=old[earlier];keep=(cycles[rows]>earlier)&(cycles[rows]<=earlier+4)
            parts.append((rows[keep],np.column_stack([scores[keep,0],cb[keep]])))
        rr=np.concatenate([p[0]for p in parts]);ss=np.concatenate([p[1]for p in parts])
        np.testing.assert_array_equal(np.sort(rr),np.flatnonzero((cycles>=52)&(cycles<=end)))
        assert all(e<=end-4 for e in report['states'][str(end)]['expanding_stack_cb']['base_fit_cutoffs'])
        use=~run.parent.stream_missing_mask(frame.iloc[rr]);r,s,cb=old[end]
        inn=np.clip(ss[use],1e-6,1-1e-6);design=np.log(inn/(1-inn))
        outer=np.clip(np.column_stack([s[:,0],cb]),1e-6,1-1e-6);outer=np.log(outer/(1-outer))
        missing=run.parent.stream_missing_mask(frame.iloc[r])
        with np.load(run.CACHE/f'outer{end}.npz',allow_pickle=False)as z:record={k:z[k].copy()for k in z.files}
        np.testing.assert_array_equal(record['rows'],r)
        np.testing.assert_array_equal(record['champion'],np.where(missing,cb,s[:,0]))
        for kind in run.KINDS:
            columns=[0]if kind=='expanding_base'else[0,1]
            model=LogisticRegression(C=1.,solver='lbfgs',max_iter=1000,tol=1e-8,random_state=42).fit(design[:,columns],y[rr][use])
            state=report['states'][str(end)][kind]['state']
            np.testing.assert_allclose(model.coef_,state['coef'],rtol=0,atol=1e-10)
            np.testing.assert_allclose(model.intercept_,state['intercept'],rtol=0,atol=1e-10)
            pred=np.where(missing,cb,model.predict_proba(outer[:,columns])[:,1])
            np.testing.assert_allclose(record[kind],pred,rtol=0,atol=1e-10)
            np.testing.assert_array_equal(record[kind][missing],record['champion'][missing]);head_checks+=1
        records[end]=record
    def cut(r,lo,hi):
        use=(cycles[r['rows']]>=lo)&(cycles[r['rows']]<=hi);return{k:v[use]for k,v in r.items()}
    def join(parts):return{k:np.concatenate([v[k]for v in parts])for k in parts[0]}
    windows={name:cut(records[e],lo,hi)for name,(e,lo,hi)in run.parent.PRIMARY_WINDOWS.items()}
    sections={**windows,'primary':join(list(windows.values())),'H12':join([records[55],records[67]]),
              'H12_E':records[55],'H12_L':records[67]};groups=0
    for name,r in sections.items():
        section=report['windows'][name]if name in windows else report[name]
        missing=run.parent.stream_missing_mask(frame.iloc[r['rows']])
        for kind in ['champion',*run.KINDS]:
            for label,use in [('all',np.ones(len(missing),dtype=bool)),('present',~missing),('missing',missing)]:
                _assert_metric(section[kind][label],y[r['rows']][use],r[kind][use],name+'/'+kind+'/'+label);groups+=1
    result={'status':'PASS','freeze_signature':freeze['signature'],'past_only_heads_independently_refit':head_checks,
            'metric_groups':groups,'missing_branch_exact':True,'new_base_fits':0,'test_read':False}
    run.save(run.OUT/'verification.json',result);print(result)
if __name__=='__main__':verify()
