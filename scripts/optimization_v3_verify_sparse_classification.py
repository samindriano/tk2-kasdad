"""Independent sklearn metric/route/fingerprint checks for sparse binary pilot."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts import optimization_v3_sparse_classification as run
from scripts.verify_campaign_classification_stage2 import _assert_metric
from src.models.round1_models import _fit_fingerprint

def verify():
    frozen=run.check();report=run.read(run.OUT/'pilot.json');frame,y,_=run.load_train(None);cycles=frame.cycle_code.to_numpy(dtype=int)
    assert report['freeze_signature']==frozen['signature']and report['newfits']==2 and not report['Test_read']
    records={}
    for end in[55,67]:
        path=run.CACHE/f'fit{end}.npz';m=run.read(path.with_suffix('.json'));assert m['npz_sha256']==run.sha(path)
        rows,s,cb=run.existing(frame,y,end);fit=frame.cycle_code.between(47,end).to_numpy();state=m['state'];fields=state['input_feature_names']
        assert state==report['states'][str(end)]
        assert state['fit_rows']==int(fit.sum())and state['fit_data_sha256']==_fit_fingerprint(frame.loc[fit,fields],tuple(fields),y[fit])
        with np.load(path,allow_pickle=False)as z:records[end]={k:z[k].copy()for k in z.files}
        r=records[end];np.testing.assert_array_equal(r['rows'],rows);missing=run.stream_missing_mask(frame.iloc[rows])
        np.testing.assert_array_equal(r['champion'],np.where(missing,cb,s[:,0]))
        np.testing.assert_array_equal(r['direct'],np.where(missing,cb,r['raw_candidate']))
        np.testing.assert_array_equal(r['blend25'],np.where(missing,cb,.75*s[:,0]+.25*r['raw_candidate']))
    allviews={};parts=[];h=[]
    for end,name in[(55,'W1'),(67,'W4')]:
        r=records[end];use=cycles[r['rows']]<=end+4;allviews[name]={k:v[use]for k,v in r.items()};parts.append(allviews[name]);h.append(r)
        allviews['H12_E'if end==55 else'H12_L']=r
    def join(parts):return{k:np.concatenate([r[k]for r in parts])for k in parts[0]}
    allviews['primary_pilot']=join(parts);allviews['H12']=join(h);checks=0
    for name,r in allviews.items():
        m=run.stream_missing_mask(frame.iloc[r['rows']])
        for kind in['champion','direct','blend25']:
            for group,use in[('all',np.ones(len(m),dtype=bool)),('present',~m),('missing',m)]:
                _assert_metric(report['views'][name][kind][group],y[r['rows']][use],r[kind][use],name+'/'+kind+'/'+group);checks+=1
    result={'status':'PASS','freeze_signature':frozen['signature'],'independent_metric_checks':checks,'newfits':0,'source_and_input_hashes_verified':True}
    run.save(run.OUT/'verification.json',result);print(result)
if __name__=='__main__':verify()
