"""Controlled text and LightGBM research, inner-temporal selection only."""
from pathlib import Path
from datetime import datetime,timezone
import sys,json
from time import perf_counter
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.models.regression_research_common import CONFIG,REPORT,CACHE,freeze_stage,verify_stage,save_json,mapped_scores,score_summary
from src.models.regression_round2_evaluation import load_protocol,positions_digest
from src.models.round1_evaluation import array_digest,digest_json
from src.utils.paths import task_files

PROTOCOL={'recipes':['text_ridge','hybrid_ridge','svd_hgb','lgb_inner'],
          'history':'expanding 47..cutoff; inner fit47..cutoff-4 validatecutoff-3..cutoff; no early stopping',
          'text':{'tfidf':'word1,2 min_df2 max_features20000 sublinear_tf','ridge_alpha':10.,'svd_components':32,
                  'stream_missing_indicator':True,'preprocessing':'Train-only vocab/IDF/scaler/OHE/imputer/SVD; selected inputs only; Ridge uses scaled numeric/OHE; SVD-HGB retains original native structured categorical mask'},
          'lgb_candidates':['base','regularized','deeper'],'lgb_selection':'Choose (parameter ID, rank weight) jointly by inner row-pooled MAE; ties1e-10 prefer smaller rank weight then lower declared complexity regularized/base/deeper',
          'rank_weights':[0.,.25,.5,.75,1.],
          'outer_policies':['raw','rank50','rank100','inner'],
          'inner_alpha':'Text recipes choose rank weight from inner MAE fixedgrid; no outerlabels used; LGB joint selection',
          'evaluation':'same sixprimary and twoH12; separatefamilies; percycle/EF/tails/stream/impact diagnostics',
          'seed':42,'threads':2,'scope':'No classification refits, Test prediction, ensemble or targettransform; cached controls reused',
          'data_reuse':'Adaptive development and previously viewed validation; innerselection does not restore untouchedholdout'}


def build(kind, param='base'):
    if kind=='lgb_inner':
        from src.models.regression_research_tabular import build_tabular
        return build_tabular(param)
    from src.models.regression_research_text import build_text_model
    return build_text_model(kind)


def state_of(model):
    state=model.audit_state();state.pop('prediction_diagnostics',None)
    return state


def fit_prediction(kind,param,fit,rows,frame,freeze):
    label=f'model__{kind}__{param}__fit{int(frame.iloc[fit].cycle_code.max())}__to{int(frame.iloc[rows].cycle_code.max())}'
    meta_path=CACHE/(label+'.json');vector_path=CACHE/(label+'.npz')
    key=digest_json({'freeze':freeze['signature'],'kind':kind,'param':param,'fit':positions_digest(fit),'validation':positions_digest(rows)})
    if meta_path.exists()!=vector_path.exists():raise RuntimeError('Incomplete cache pair: '+label)
    if meta_path.exists():
        meta=json.loads(meta_path.read_text(encoding='utf-8'))
        with np.load(vector_path,allow_pickle=False)as z:
            pred=z['prediction'].copy()
            if not np.array_equal(z['fit_rows'],fit)or not np.array_equal(z['validation_rows'],rows):raise RuntimeError('Cached rows changed')
        if meta['key']!=key or meta['prediction_sha256']!=array_digest(pred)or meta['state_sha256']!=digest_json(meta['state']):raise RuntimeError('Cached model identity changed')
        return pred,meta
    model=build(kind,param);features=list(model.feature_names)
    start=perf_counter();started=datetime.now(timezone.utc).isoformat()
    with threadpool_limits(limits=2):
        model.fit(frame.iloc[fit][features].copy(),frame.iloc[fit].threat_rank.to_numpy(copy=True))
        state=state_of(model);pred=np.asarray(model.predict(frame.iloc[rows][features].copy()),float)
    if digest_json(state)!=digest_json(state_of(model)):raise RuntimeError('Prediction changed fit state')
    if pred.shape!=(len(rows),)or not np.isfinite(pred).all():raise RuntimeError('Invalid model output')
    meta={'key':key,'kind':kind,'param':param,'fit_started_at_utc':started,'seconds':perf_counter()-start,
          'features':features,'fit_rows':len(fit),'fit_cycle_range':[int(frame.iloc[fit].cycle_code.min()),int(frame.iloc[fit].cycle_code.max())],
          'validation_rows':len(rows),'validation_cycle_range':[int(frame.iloc[rows].cycle_code.min()),int(frame.iloc[rows].cycle_code.max())],
          'fit_positions_sha256':positions_digest(fit),'validation_positions_sha256':positions_digest(rows),
          'state':state,'state_sha256':digest_json(state),'prediction_sha256':array_digest(pred),
          'validation_labels_passed_to_fit':False,'test_features_used_for_fit':False}
    np.savez_compressed(vector_path,fit_rows=fit,validation_rows=rows,prediction=pred)
    save_json(meta_path,meta)
    print(label,'fit',round(meta['seconds'],2),'s',flush=True)
    return pred,meta


def select_inner(records):
    order={'regularized':0,'base':1,'deeper':2}
    best=min(v['mae'] for v in records)
    close=[v for v in records if v['mae']<=best+1e-10]
    return sorted(close,key=lambda v:(v['rank_weight'],order.get(v['param'],0)))[0]


def main():
    code=['src/models/regression_research_common.py','src/models/regression_research_text.py','src/models/regression_research_tabular.py','scripts/research_regression_models.py']
    import importlib.metadata
    protocol={**PROTOCOL,'lightgbm_version':importlib.metadata.version('lightgbm')}
    freeze=freeze_stage('stage23_models',protocol,code)
    frame=pd.read_csv(task_files('regression')['train']);p=load_protocol();CACHE.mkdir(parents=True,exist_ok=True)
    all_windows=[('primary',w)for w in p['primary_windows']]+[('secondary',w)for w in p['secondary_windows']]
    jobs=[];predictions={};audits={};inner=[];start=perf_counter()
    for end in [55,59,63,67,71,75]:
        past=np.flatnonzero(frame.cycle_code.between(47,end));innerfit=np.flatnonzero(frame.cycle_code.between(47,end-4))
        innerval=np.flatnonzero(frame.cycle_code.between(end-3,end));last=max(w['validation'][1]for _,w in all_windows if w['fit'][1]==end)
        outer=np.flatnonzero(frame.cycle_code.between(end+1,last));audits[str(end)]={}
        assert frame.iloc[innerfit].cycle_code.max()<frame.iloc[innerval].cycle_code.min()<=end<frame.iloc[outer].cycle_code.min()
        for kind in PROTOCOL['recipes']:
            params=['base','regularized','deeper']if kind=='lgb_inner'else['base']
            choices=[];inner_metas={}
            for param in params:
                raw,meta=fit_prediction(kind,param,innerfit,innerval,frame,freeze);inner_metas[param]=meta
                for weight in PROTOCOL['rank_weights']:
                    pred=mapped_scores(raw,frame.iloc[innerval].cycle_code,weight)
                    choices.append({'param':param,'rank_weight':weight,'mae':float(np.abs(pred-frame.iloc[innerval].threat_rank.to_numpy()).mean())})
            selected=select_inner(choices)
            inner.append({'fit_end':end,'recipe':kind,'selected':selected,'all_choices':choices,'inner_fit_max':end-4,'inner_validation':[end-3,end]})
            raw,meta=fit_prediction(kind,selected['param'],past,outer,frame,freeze)
            audits[str(end)][kind]={'outer':meta,'inner':inner_metas,'selection':selected}
            for name,weight in [('raw',None),('rank50',.5),('rank100',1.),('inner',selected['rank_weight'])]:
                pred=raw.copy()if weight is None else mapped_scores(raw,frame.iloc[outer].cycle_code,weight)
                predictions[(end,kind,name)]=(outer,pred)
                np.savez_compressed(CACHE/f'eval__fit{end}__{kind}__{name}.npz',validation_rows=outer,prediction=pred)
    for family,w in all_windows:
        for kind in PROTOCOL['recipes']:
            for name in PROTOCOL['outer_policies']:
                rows,pred=predictions[(w['fit'][1],kind,name)];mask=frame.iloc[rows].cycle_code.between(*w['validation']).to_numpy();rows,pred=rows[mask],pred[mask]
                jobs.append({'recipe':kind,'policy':name,'family':family,'window':w['id'],'prediction_sha256':array_digest(pred),**score_summary(frame.iloc[rows],pred)})
    aggregate={}
    for family,key in [('primary','primary_windows'),('secondary','secondary_windows')]:
        aggregate[family]={}
        for kind in PROTOCOL['recipes']:
            for name in PROTOCOL['outer_policies']:
                pairs=[]
                for w in p[key]:
                    rows,pred=predictions[(w['fit'][1],kind,name)];mask=frame.iloc[rows].cycle_code.between(*w['validation']).to_numpy();pairs.append((rows[mask],pred[mask]))
                rows=np.concatenate([v[0]for v in pairs]);pred=np.concatenate([v[1]for v in pairs]);order=np.argsort(rows);assert len(np.unique(rows))==len(rows)
                aggregate[family][kind+'__'+name]=score_summary(frame.iloc[rows[order]],pred[order],impact=True)
    result={'stage':'stage23_models','freeze_signature':freeze['signature'],'inner':inner,'audits':audits,'jobs':jobs,'aggregate':aggregate,
            'elapsed_seconds':perf_counter()-start,'selection_boundary':'Joint parameter/mapping selection uses strictly past inner labels only. Outer windows reused development; no Test prediction here.'}
    save_json(REPORT/'stage23_models.json',result)
    for kind in PROTOCOL['recipes']:
        for name in PROTOCOL['outer_policies']:
            a=aggregate['primary'][kind+'__'+name];b=aggregate['secondary'][kind+'__'+name]
            print(kind,name,'primary',round(a['raw']['mae'],6),'H12',round(b['raw']['mae'],6),flush=True)
    verify_stage('stage23_models')


if __name__=='__main__':main()
