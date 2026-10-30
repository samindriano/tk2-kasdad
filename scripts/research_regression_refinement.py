"""Stage4: three substantive fixed controls; no leaderboard-selected parameters."""
from pathlib import Path
import sys,json
from time import perf_counter
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.models.regression_research_common import REPORT,CACHE,freeze_stage,verify_stage,save_json,mapped_scores,score_summary
from src.models.regression_research_refinement import build_refinement
from src.models.regression_round2_evaluation import load_protocol,positions_digest
from src.models.round1_evaluation import array_digest,digest_json
from src.utils.paths import task_files

PROTOCOL={'recipes':['svd128','svd_noscope32','structured_ridge'],
          'hypotheses':{'svd128':'32 text dimensions may discard predictive content; increase to128 with identical HGB settings',
                        'svd_noscope32':'Remove exactly four scope/spread fields to test reduced dependence; no novelzero imputation',
                        'structured_ridge':'Same Ridge pipeline as hybrid with constant emptytext to isolate actual text+availability contribution'},
          'history':'expanding; same6primary+2H12; inner47..E-4 validateE-3..E',
          'rank_weights':[0.,.25,.5,.75,1.],'outer_policies':['raw','inner'],
          'seed':42,'threads':2,'HGB':'same200iterations15leavesabsoluteerror; no additional tuning',
          'selection':'Only inner MAE chooses alpha; ties prefer lower alpha. Three hypotheses chosen before any additional Kaggle outcome; adaptive internaldevelopment.',
          'scope':'No Test targets, no artificialzero-as-missing intervention; unchanged validation inputs. If predictionstress attempted later it must be unscored sensitivity, not cohorterror.'}


def fit_predict(kind,fit,rows,frame,frozen):
    label=f"refine__{kind}__fit{int(frame.iloc[fit].cycle_code.max())}__to{int(frame.iloc[rows].cycle_code.max())}"
    meta_path=CACHE/(label+'.json');path=CACHE/(label+'.npz')
    key=digest_json({'freeze':frozen['signature'],'kind':kind,'fit':positions_digest(fit),'validation':positions_digest(rows)})
    if meta_path.exists()!=path.exists():raise RuntimeError('Incomplete refinement cache')
    if path.exists():
        m=json.loads(meta_path.read_text(encoding='utf-8'))
        with np.load(path,allow_pickle=False)as z:
            assert np.array_equal(z['fit_rows'],fit)and np.array_equal(z['validation_rows'],rows)
            p=z['prediction'].copy()
        assert m['key']==key and m['prediction_sha256']==array_digest(p)and m['state_sha256']==digest_json(m['state'])
        return p,m
    model=build_refinement(kind);start=perf_counter();features=list(model.feature_names)
    with threadpool_limits(limits=2):
        model.fit(frame.iloc[fit][features].copy(),frame.iloc[fit].threat_rank.to_numpy(copy=True))
        state=model.audit_state();pred=np.asarray(model.predict(frame.iloc[rows][features].copy()),float)
    assert state==model.audit_state()and pred.shape==(len(rows),)and np.isfinite(pred).all()
    m={'key':key,'kind':kind,'fit_rows':len(fit),'fit_positions_sha256':positions_digest(fit),'validation_positions_sha256':positions_digest(rows),
       'features':features,'state':state,'state_sha256':digest_json(state),'prediction_sha256':array_digest(pred),'seconds':perf_counter()-start,
       'validation_labels_passed_to_fit':False}
    np.savez_compressed(path,fit_rows=fit,validation_rows=rows,prediction=pred);save_json(meta_path,m)
    print(label,'fit',round(m['seconds'],2),'s',flush=True)
    return pred,m


def main():
    frozen=freeze_stage('stage4_refinement',PROTOCOL,['src/models/regression_research_common.py','src/models/regression_research_refinement.py','scripts/research_regression_refinement.py'])
    f=pd.read_csv(task_files('regression')['train']);p=load_protocol();wins=[('primary',w)for w in p['primary_windows']]+[('secondary',w)for w in p['secondary_windows']]
    predictions={};audits={};inner=[];start=perf_counter()
    for end in [55,59,63,67,71,75]:
        fit=np.flatnonzero(f.cycle_code.between(47,end));inn=np.flatnonzero(f.cycle_code.between(47,end-4));iv=np.flatnonzero(f.cycle_code.between(end-3,end));last=max(w['validation'][1]for _,w in wins if w['fit'][1]==end);rows=np.flatnonzero(f.cycle_code.between(end+1,last));audits[str(end)]={}
        assert f.iloc[inn].cycle_code.max()<f.iloc[iv].cycle_code.min()<=end<f.iloc[rows].cycle_code.min()
        for kind in PROTOCOL['recipes']:
            ip,im=fit_predict(kind,inn,iv,f,frozen)
            choices=[{'rank_weight':a,'mae':float(np.abs(mapped_scores(ip,f.iloc[iv].cycle_code,a)-f.iloc[iv].threat_rank.to_numpy()).mean())}for a in PROTOCOL['rank_weights']]
            best=min(c['mae']for c in choices);selected=min([c for c in choices if c['mae']<=best+1e-10],key=lambda c:c['rank_weight'])
            pred,meta=fit_predict(kind,fit,rows,f,frozen);audits[str(end)][kind]={'inner':im,'outer':meta,'selection':selected};inner.append({'fit_end':end,'recipe':kind,'choices':choices,'selected':selected})
            for name,alpha in [('raw',None),('inner',selected['rank_weight'])]:
                q=pred if alpha is None else mapped_scores(pred,f.iloc[rows].cycle_code,alpha)
                predictions[end,kind,name]=(rows,q);np.savez_compressed(CACHE/f'eval_refine__fit{end}__{kind}__{name}.npz',validation_rows=rows,prediction=q)
    jobs=[];aggregate={}
    for family,w in wins:
        for kind in PROTOCOL['recipes']:
            for name in PROTOCOL['outer_policies']:
                rows,pred=predictions[w['fit'][1],kind,name];mask=f.iloc[rows].cycle_code.between(*w['validation']).to_numpy()
                jobs.append({'recipe':kind,'policy':name,'family':family,'window':w['id'],'prediction_sha256':array_digest(pred[mask]),**score_summary(f.iloc[rows[mask]],pred[mask])})
    for family,key in [('primary','primary_windows'),('secondary','secondary_windows')]:
        aggregate[family]={}
        for kind in PROTOCOL['recipes']:
            for name in PROTOCOL['outer_policies']:
                parts=[]
                for w in p[key]:
                    rows,pred=predictions[w['fit'][1],kind,name];mask=f.iloc[rows].cycle_code.between(*w['validation']).to_numpy();parts.append((rows[mask],pred[mask]))
                rows=np.concatenate([a[0]for a in parts]);pred=np.concatenate([a[1]for a in parts]);order=np.argsort(rows);assert len(rows)==len(np.unique(rows))
                aggregate[family][kind+'__'+name]=score_summary(f.iloc[rows[order]],pred[order],impact=True)
    save_json(REPORT/'stage4_refinement.json',{'stage':'stage4_refinement','freeze_signature':frozen['signature'],'inner':inner,'audits':audits,'jobs':jobs,'aggregate':aggregate,'elapsed_seconds':perf_counter()-start})
    for label,a in aggregate['primary'].items():print(label,a['raw']['mae'],aggregate['secondary'][label]['raw']['mae'],flush=True)
    verify_stage('stage4_refinement')


if __name__=='__main__':main()
