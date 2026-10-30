"""Final focused model x text interaction; two existing LightGBM complexities."""
from pathlib import Path
import json,sys
from time import perf_counter
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.models.regression_research_common import REPORT,CACHE,freeze_stage,verify_stage,save_json,mapped_scores,score_summary
from src.models.regression_research_lgb_text import build_lgb_text
from src.models.regression_round2_evaluation import load_protocol,positions_digest
from src.models.round1_evaluation import array_digest,digest_json
from src.utils.paths import task_files
from scripts.research_regression_models import select_inner

PROTOCOL={'family':'LightGBM MAE + same native structured/TFIDF-SVD32/missingflag as SVD-HGB',
          'params':['base','regularized'],'rank_weights':[0.,.25,.5,.75,1.],'outer_policies':['raw','inner'],
          'selection':'Joint param/rankweight argmin inner47..E-4->E-3..E; tie lowerweight thenregularized; outer untouchedbyselection',
          'reason':'Text improved HGB substantially, while structured LightGBM modest. Test whether information and model capacity interact, holding representation fixed. Two configurations are reused from earlier frozen LightGBM; no new grid or family.',
          'external_context_seen':'First two public scores12.55466/12.34519, neitherbelow12.23275; this stage is adaptive research, not an independentholdout. Configurations derive from internal model x representation contrast, not fittoleaderboard.',
          'boundary':'Same6primary+2H12; past-only preprocessing; noTestlabels/no classification/no ensemble; maximum6additionalphase submissions unchanged.'}


def fit_predict(param,fit,rows,f,freeze):
    name=f'lgbtext__{param}__fit{int(f.iloc[fit].cycle_code.max())}__to{int(f.iloc[rows].cycle_code.max())}'
    path=CACHE/(name+'.npz');meta_path=CACHE/(name+'.json')
    key=digest_json({'freeze':freeze['signature'],'param':param,'fit':positions_digest(fit),'validation':positions_digest(rows)})
    if path.exists()!=meta_path.exists():raise RuntimeError('Incomplete LGB text cache')
    if path.exists():
        m=json.loads(meta_path.read_text(encoding='utf-8'))
        with np.load(path,allow_pickle=False)as z:
            assert np.array_equal(z['fit_rows'],fit)and np.array_equal(z['validation_rows'],rows)
            pred=z['prediction'].copy()
        assert m['key']==key and m['prediction_sha256']==array_digest(pred)and m['state_sha256']==digest_json(m['state'])
        return pred,m
    model=build_lgb_text(param);features=list(model.feature_names);start=perf_counter()
    with threadpool_limits(limits=2):
        model.fit(f.iloc[fit][features].copy(),f.iloc[fit].threat_rank.to_numpy(copy=True))
        state=model.audit_state();pred=model.predict(f.iloc[rows][features].copy())
    assert model.audit_state()==state and np.isfinite(pred).all()
    m={'key':key,'features':features,'param':param,'fit_rows':len(fit),'fit_positions_sha256':positions_digest(fit),'validation_positions_sha256':positions_digest(rows),
       'state':state,'state_sha256':digest_json(state),'prediction_sha256':array_digest(pred),'seconds':perf_counter()-start,'validation_labels_passed_to_fit':False}
    np.savez_compressed(path,fit_rows=fit,validation_rows=rows,prediction=pred);save_json(meta_path,m)
    print(name,'fit',round(m['seconds'],2),'s',flush=True);return pred,m


def main():
    freeze=freeze_stage('stage5_lgb_text',PROTOCOL,['src/models/regression_research_common.py','src/models/regression_research_lgb_text.py','scripts/research_regression_lgb_text.py'])
    f=pd.read_csv(task_files('regression')['train']);p=load_protocol();wins=[('primary',w)for w in p['primary_windows']]+[('secondary',w)for w in p['secondary_windows']]
    vectors={};audits={};inner=[];start=perf_counter()
    for end in [55,59,63,67,71,75]:
        fit=np.flatnonzero(f.cycle_code.between(47,end));inf=np.flatnonzero(f.cycle_code.between(47,end-4));iv=np.flatnonzero(f.cycle_code.between(end-3,end));last=max(w['validation'][1]for _,w in wins if w['fit'][1]==end);rows=np.flatnonzero(f.cycle_code.between(end+1,last));choices=[];im={}
        assert f.iloc[inf].cycle_code.max()<f.iloc[iv].cycle_code.min()<=end<f.iloc[rows].cycle_code.min()
        for param in PROTOCOL['params']:
            pred,m=fit_predict(param,inf,iv,f,freeze);im[param]=m
            for a in PROTOCOL['rank_weights']:choices.append({'param':param,'rank_weight':a,'mae':float(np.abs(mapped_scores(pred,f.iloc[iv].cycle_code,a)-f.iloc[iv].threat_rank.to_numpy()).mean())})
        selected=select_inner(choices);pred,meta=fit_predict(selected['param'],fit,rows,f,freeze);audits[str(end)]={'inner':im,'outer':meta,'selection':selected};inner.append({'fit_end':end,'selected':selected,'choices':choices})
        for policy,alpha in [('raw',None),('inner',selected['rank_weight'])]:
            q=pred if alpha is None else mapped_scores(pred,f.iloc[rows].cycle_code,alpha)
            vectors[end,policy]=(rows,q);np.savez_compressed(CACHE/f'eval_lgbtext__fit{end}__{policy}.npz',validation_rows=rows,prediction=q)
    jobs=[];aggregate={}
    for family,w in wins:
        for policy in PROTOCOL['outer_policies']:
            rows,pred=vectors[w['fit'][1],policy];mask=f.iloc[rows].cycle_code.between(*w['validation']).to_numpy()
            jobs.append({'recipe':'lgb_text','policy':policy,'family':family,'window':w['id'],'prediction_sha256':array_digest(pred[mask]),**score_summary(f.iloc[rows[mask]],pred[mask])})
    for family,key in [('primary','primary_windows'),('secondary','secondary_windows')]:
        aggregate[family]={}
        for policy in PROTOCOL['outer_policies']:
            parts=[]
            for w in p[key]:
                rows,pred=vectors[w['fit'][1],policy];mask=f.iloc[rows].cycle_code.between(*w['validation']).to_numpy();parts.append((rows[mask],pred[mask]))
            rows=np.concatenate([v[0]for v in parts]);pred=np.concatenate([v[1]for v in parts]);order=np.argsort(rows);assert len(rows)==len(np.unique(rows))
            aggregate[family]['lgb_text__'+policy]=score_summary(f.iloc[rows[order]],pred[order],impact=True)
    save_json(REPORT/'stage5_lgb_text.json',{'stage':'stage5_lgb_text','freeze_signature':freeze['signature'],'inner':inner,'audits':audits,'jobs':jobs,'aggregate':aggregate,'elapsed_seconds':perf_counter()-start})
    for k,a in aggregate['primary'].items():print(k,a['raw']['mae'],aggregate['secondary'][k]['raw']['mae'],flush=True)
    verify_stage('stage5_lgb_text')


if __name__=='__main__':main()
