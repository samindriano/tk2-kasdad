"""Independent CDF calculation, fit fingerprint, control and MAE reconciliation."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from sklearn.metrics import mean_absolute_error
from scripts import optimization_v3_distributional as run
from src.models.round1_models import _fit_fingerprint
from scripts.optimization_v3_reg_diagnostics import _build_view,PRIMARY_CACHE,PILOT_CACHE,FULL_CACHE

def verify():
    frozen=run.check();report=run.read(run.OUT/'pilot.json');frame,_=run.load();y=frame.threat_rank.to_numpy(float);cy=frame.cycle_code.to_numpy(dtype=int);records={}
    for end in[55,67]:
        path=run.CACHE/f'fit{end}.npz';meta=run.read(path.with_suffix('.json'));assert meta['npz_sha256']==run.sha(path)
        state=meta['state'];fit=frame.cycle_code.between(47,end).to_numpy();fields=state['input_feature_names']
        assert state==report['states'][str(end)]and state['fit_rows']==int(fit.sum())
        assert state['fit_data_sha256']==_fit_fingerprint(frame.loc[fit,fields],tuple(fields),y[fit])
        expected_bins=np.minimum((y[fit]/10).astype(int),9);assert state['bin_counts']==np.bincount(expected_bins,minlength=10).tolist()
        with np.load(path,allow_pickle=False)as z:r={k:z[k].copy()for k in z.files}
        rows,old=run.control(frame,end);np.testing.assert_array_equal(rows,r['rows']);np.testing.assert_array_equal(old,r['champion'])
        with np.load(PRIMARY_CACHE,allow_pickle=False)as base,np.load(PILOT_CACHE,allow_pickle=False)as pilot,np.load(FULL_CACHE,allow_pickle=False)as full:
            independent,_=_build_view(frame,base,pilot,full,[(end,end+1,{55:67,67:79}[end])])
            np.testing.assert_array_equal(rows,independent.row.to_numpy());np.testing.assert_allclose(old,independent.prediction.to_numpy(),rtol=0,atol=1e-12)
        p=r['probabilities'];np.testing.assert_allclose(p.sum(axis=1),1,atol=1e-10,rtol=0);assert p.shape==(len(rows),10)
        expected=[]
        for probs in p:
            total=0.
            for i,mass in enumerate(probs):
                if total+mass>=.5:expected.append(10*i+10*(.5-total)/mass);break
                total+=mass
        np.testing.assert_allclose(r['raw'],expected,atol=1e-10,rtol=0)
        np.testing.assert_array_equal(r['blend25'],np.clip(.75*old+.25*r['raw'],0,100));records[end]=r
    primary=[];h=[];views={}
    for end,name in[(55,'W1'),(67,'W4')]:
        r=records[end];use=cy[r['rows']]<=end+4;part={k:v[use]for k,v in r.items()};primary.append(part);h.append(r)
        views[name]=part;views['H12_E'if end==55 else'H12_L']=r
    def join(parts):return{k:np.concatenate([r[k]for r in parts])for k in parts[0]}
    views['primary_pilot']=join(primary);views['H12']=join(h);checks=0
    for name,r in views.items():
        for kind in['raw','blend25','champion']:
            expected=mean_absolute_error(y[r['rows']],r[kind]);np.testing.assert_allclose(expected,report['views'][name][kind]['mae'],atol=1e-12,rtol=0);checks+=1
    result={'status':'PASS','freeze_signature':frozen['signature'],'independent_metric_checks':checks,'independent_CDF_recomputed':True,'fit_fingerprints':2,'newfits':0}
    run.save(run.OUT/'verification.json',result);print(result)
if __name__=='__main__':verify()
