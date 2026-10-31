"""Independent target/rank/mixture/fit-boundary reconciliation for char pilot."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import mean_absolute_error
from scripts import optimization_v3_character as run
from src.models.round1_models import _fit_fingerprint

def verify():
    frozen=run.check();report=run.read(run.OUT/'pilot.json');frame,_=run.parent.load();y=frame.threat_rank.to_numpy(float);cycles=frame.cycle_code.to_numpy(dtype=int);records={}
    assert report['freeze_signature']==frozen['signature']and report['fits']==2 and not report['Test_read']
    for end in[55,67]:
        path=run.CACHE/f'fit{end}.npz';meta=run.read(path.with_suffix('.json'));assert meta['npz_sha256']==run.sha(path)
        state=meta['state'];fit=frame.cycle_code.between(47,end).to_numpy();fields=state['input_feature_names']
        assert state['fit_rows']==int(fit.sum())and state['fit_data_sha256']==_fit_fingerprint(frame.loc[fit,fields],tuple(fields),y[fit])
        assert state['tfidf']['parameters']['analyzer']=='char_wb'and state['tfidf']['word_tokens_used'] is False
        with np.load(path,allow_pickle=False)as z:r={k:z[k].copy()for k in z.files}
        rows,old=run.parent.control(frame,end);np.testing.assert_array_equal(rows,r['rows']);np.testing.assert_array_equal(old,r['champion'])
        cy=cycles[rows];expected=np.empty_like(old)
        for c in np.unique(cy):
            m=cy==c;expected[m]=np.clip(.5*r['raw'][m]+50*rankdata(r['raw'][m],method='average')/m.sum(),0,100)
        np.testing.assert_array_equal(expected,r['rank50']);np.testing.assert_array_equal(r['blend25'],np.clip(.75*old+.25*expected,0,100));records[end]=r
    primary=[];h=[];views={}
    for end,name in[(55,'W1'),(67,'W4')]:
        r=records[end];use=cycles[r['rows']]<=end+4;part={k:v[use]for k,v in r.items()};views[name]=part;primary.append(part);h.append(r);views['H12_E'if end==55 else'H12_L']=r
    def join(parts):return{k:np.concatenate([r[k]for r in parts])for k in parts[0]}
    views['primary_pilot']=join(primary);views['H12']=join(h);checks=0
    for name,r in views.items():
        for kind in['raw','rank50','blend25','champion']:
            np.testing.assert_allclose(mean_absolute_error(y[r['rows']],r[kind]),report['views'][name][kind]['mae'],atol=1e-12,rtol=0);checks+=1
    result={'status':'PASS','freeze_signature':frozen['signature'],'independent_metric_checks':checks,'fitfingerprints':2,'rank_and_mixture_recomputed':True,'newfits':0}
    run.save(run.OUT/'verification.json',result);print(result)
if __name__=='__main__':verify()
