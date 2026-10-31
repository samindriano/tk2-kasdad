"""Independent sklearn reconciliation of past-only score-head evaluation."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from sklearn.metrics import f1_score,confusion_matrix,average_precision_score
from scripts.optimization_v2_classification import (read,save,check,load_train,OUT,CACHE,ENDS,KINDS,PRIMARY_WINDOWS)
from src.models.classification_round2_models import stream_missing_mask

def verify():
    frozen=check();report=read(OUT/'results.json');frame,y,_=load_train(None);cycles=frame.cycle_code.to_numpy(dtype=int)
    assert report['freeze_signature']==frozen['signature'] and report['outer_labels_used_for_heads'] is False
    records={}
    for end in ENDS:
        with np.load(CACHE/f'outer{end}.npz',allow_pickle=False)as z:records[end]={k:z[k].copy()for k in z.files}
        r=records[end];expected=np.flatnonzero((cycles>end)&(cycles<={55:67,67:79}.get(end,end+4)))
        np.testing.assert_array_equal(r['rows'],expected)
        m=stream_missing_mask(frame.iloc[expected])
        for kind in KINDS:np.testing.assert_array_equal(r[kind][m],r['champion'][m])
        state=report['states'][str(end)]
        assert state['base_fit_end_for_inner']==end-4 and state['calibration_cycles']==[end-3,end]
    def cut(rec,lo,hi):
        use=(cycles[rec['rows']]>=lo)&(cycles[rec['rows']]<=hi)
        return {k:v[use]for k,v in rec.items()}
    windows={name:cut(records[end],lo,hi)for name,(end,lo,hi)in PRIMARY_WINDOWS.items()}
    def join(parts):return {k:np.concatenate([v[k]for v in parts])for k in parts[0]}
    allrec={**windows,'primary':join(list(windows.values())),'H12':join([records[55],records[67]]),
            'H12_E':records[55],'H12_L':records[67]};count=0
    for name,r in allrec.items():
        assert len(np.unique(r['rows']))==len(r['rows'])
        section=report['windows'][name]if name in windows else report[name]
        m=stream_missing_mask(frame.iloc[r['rows']])
        for kind in ['champion',*KINDS]:
            for group,mask in [('all',np.ones(len(m),dtype=bool)),('present',~m),('missing',m)]:
                p=r[kind][mask];actual=y[r['rows']][mask];pred=p>=.5;expected=section[kind][group]
                np.testing.assert_allclose(expected['macro_f1'],f1_score(actual,pred,labels=[0,1],average='macro',zero_division=0),rtol=0,atol=1e-12)
                assert expected['confusion_matrix']==confusion_matrix(actual,pred,labels=[0,1]).tolist()
                np.testing.assert_allclose(expected['average_precision'],average_precision_score(actual,p),rtol=0,atol=1e-12)
                np.testing.assert_allclose(expected['brier_score'],np.mean((p-actual)**2),rtol=0,atol=1e-12)
                count+=1
    result={'status':'PASS','freeze_signature':frozen['signature'],'independent_metric_groups':count,
            'missing_branch_exactly_unchanged':True,'temporal_head_boundaries_checked':6,
            'test_read':False,'fit_count':0}
    save(OUT/'verification.json',result);print(result)
if __name__=='__main__':verify()
