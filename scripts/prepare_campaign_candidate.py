"""Build an exclusive research inference artifact from an approved recipe.

Each component refits on official Train; cached full-Train prediction vectors
may be reused only with matching input/source/recipe hashes. No upload occurs.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,importlib.metadata,json,subprocess,sys,time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_initial_submissions import _load_inputs
from src.models.classification_models import build_classifier
from src.models.classification_round2_models import stream_missing_mask
from src.models.regression_research_lgb_text import build_lgb_text
from src.models.regression_research_text import build_text_model
from src.models.regression_research_common import mapped_scores
from src.models.round1_evaluation import array_digest,digest_json
from scripts.submit_regression_research import ordered_ids_sha256
OUT=ROOT/'reports/dual_campaign/kaggle';CACHE=ROOT/'experiments/runs/dual_inference_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def codesha(p):return hashlib.sha256(Path(p).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
def exclusive(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n')as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n')

def component_model(task,component):
    name=component['model']
    if task=='classification':
        if name in ['C1_B','C2_B']:return build_classifier({'id':name})
        from src.models.campaign_classification import build_campaign_classifier
        return build_campaign_classifier(name)
    if name=='lgb':return build_lgb_text(component.get('params_id','base'))
    if name=='ridge':return build_text_model('hybrid_ridge')
    if name=='catboost':
        from src.models.campaign_catboost import build_campaign_catboost
        return build_campaign_catboost()
    raise ValueError('Unknown frozen component model')

def fit_component(task,component,train,test,identity):
    fit_spec={k:v for k,v in component.items()if k not in ['weight','rank_weight']}
    key=digest_json({'task':task,'component':fit_spec,'identity':identity})
    path=CACHE/(key+'.npz');meta=path.with_suffix('.json');target='artifact_signal'if task=='classification'else'threat_rank'
    if path.exists()or meta.exists():
        assert path.exists()and meta.exists(),'Incomplete full-Train component cache'
        m=read(meta);assert m['key']==key
        with np.load(path,allow_pickle=False)as z:p=z['prediction'].copy()
        assert array_digest(p)==m['prediction_sha256']and len(p)==len(test)
        assert digest_json(m['state'])==m['state_sha256'];return p,m
    model=component_model(task,component)
    fields=list(getattr(model,'input_feature_names',getattr(model,'feature_names',[])))
    assert fields and not {'record_id','artifact_signal','threat_rank'}.intersection(fields)
    start=time.perf_counter()
    with threadpool_limits(limits=2):
        model.fit(train[fields].copy(),train[target].to_numpy(copy=True));state=model.audit_state()
        p=model.predict_proba(test[fields].copy())if task=='classification'else model.predict(test[fields].copy())
    assert state==model.audit_state()and np.isfinite(p).all()
    if task=='classification':assert ((p>=0)&(p<=1)).all()
    if task=='regression'and component['model']in ['lgb','ridge']:
        previous='lgb_text_inner_v1'if component['model']=='lgb'else'hybrid_ridge_inner_v1'
        expected=read(ROOT/f'reports/regression_research/submissions/{previous}.json')['model_state']
        assert state==expected,(component['model'],'Full-Train component differs from verified historical recipe')
    m={'key':key,'fit_component':fit_spec,'fit_rows':len(train),'features':fields,'prediction_sha256':array_digest(p),
        'state':state,'state_sha256':digest_json(state),'seconds':time.perf_counter()-start,'learned_only_official_Train':True}
    CACHE.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,prediction=p);exclusive(meta,m)
    print('FITTED',task,component['model'],round(m['seconds'],2),flush=True);return p,m

def prepare(decision_path):
    decision_path=Path(decision_path).resolve();decision_path.relative_to((ROOT/'experiments/dual_campaign/decisions').resolve())
    d=read(decision_path);assert d['approved_for_external_evaluation']is True
    for p,h in d['source_code_sha256'].items():assert codesha(ROOT/p)==h,p
    for p,h in d['internal_result_sha256'].items():assert sha(ROOT/p)==h,p
    task=d['task'];candidate=d['candidate_id'];recipe=d['recipe']
    assert __import__('re').fullmatch('[a-z0-9_]{1,80}',candidate)
    manifest=OUT/(candidate+'.json');output=ROOT/f'submissions/dual_campaign/{candidate}.csv'
    assert not manifest.exists()and not output.exists(),'Do not overwrite an existing candidate'
    train,test,sample,hashes=_load_inputs(task);target='artifact_signal'if task=='classification'else'threat_rank'
    assert target not in test and len(train)==50608 and len(test)==29421
    identity={'input_sha256':hashes,'source_code_sha256':d['source_code_sha256']}
    audits=[];parts=[]
    for c in recipe['components']:
        p,m=fit_component(task,c,train,test,identity);audits.append(m)
        if task=='regression':p=mapped_scores(p,test.cycle_code.to_numpy(),c['rank_weight'])
        parts.append(float(c['weight'])*p)
    assert abs(sum(c['weight']for c in recipe['components'])-1)<1e-12
    raw=np.sum(parts,axis=0)
    if task=='classification':
        fallback,fallback_audit=fit_component(task,{'model':'C1_B'},train,test,identity);audits.append(fallback_audit)
        missing=stream_missing_mask(test);probability=np.where(missing,fallback,raw)
        values=(probability>=recipe.get('threshold',.5)).astype(np.int8)
        old=read(ROOT/'reports/submissions/classification_initial_manifest.json')['model']['missing_fit_state']
        assert fallback_audit['state']==old,'Frozen fallback no longer exact'
    else:probability=None;values=np.clip(raw,0,100)
    assert np.isfinite(values).all()and values.shape==(29421,)
    if task=='regression':assert ((values>=0)&(values<=100)).all()
    baseline_path=ROOT/('submissions/classification_initial.csv'if task=='classification'else'submissions/regression_research/lgb_text_inner_v1.csv')
    old=pd.read_csv(baseline_path,float_precision='round_trip');assert old.record_id.equals(test.record_id)
    changed=int(np.sum(values!=old[target].to_numpy()));assert changed>0,'Duplicate official baseline'
    # Prevent identical CSV payloads under new names, including earlier campaigns.
    for directory in [OUT,ROOT/'reports/autonomous_eda/kaggle']:
        if not directory.exists():continue
        for path in directory.glob('*.json'):
            m=read(path)
            if m.get('task')==task and m.get('status')=='PREPARED_AND_QA_PASS':
                previous=pd.read_csv(ROOT/m['csv_path'],float_precision='round_trip')
                assert not np.array_equal(previous[target].to_numpy(),values),('Duplicate prepared/accepted candidate',m['candidate_id'])
    frame=pd.DataFrame({'record_id':test.record_id,target:values});assert frame.record_id.equals(sample.record_id)
    assert list(frame.columns)==list(sample.columns);output.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(output,index=False,float_format='%.17g');again=pd.read_csv(output,float_precision='round_trip')
    assert again.record_id.equals(sample.record_id);np.testing.assert_array_equal(again[target],values)
    np.savez_compressed(CACHE/(candidate+'.npz'),prediction=values,raw=raw,probability=probability if probability is not None else np.empty(0))
    result={'status':'PREPARED_AND_QA_PASS','task':task,'candidate_id':candidate,'competition':f'tp2-{task}-cyber-vulnerability-kasdd-gasal-2026-2027',
        'recipe':recipe,'decision_path':decision_path.relative_to(ROOT).as_posix(),'decision_sha256':sha(decision_path),
        'source_commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        'source_code_sha256':d['source_code_sha256'],'input_sha256':hashes,'csv_path':output.relative_to(ROOT).as_posix(),
        'csv_sha256':sha(output),'ordered_test_ids_sha256':ordered_ids_sha256(test.record_id.astype(str).tolist()),
        'train_rows':len(train),'test_rows':len(test),'fit_preprocessing_only_train':True,'test_labels_used':False,
        'candidate_differs_from_accepted_baseline':True,'changed_predictions_vs_baseline':changed,
        'component_audits':audits,'prediction_sha256':array_digest(values),'raw_prediction_sha256':array_digest(raw),
        'prediction_range':[float(values.min()),float(values.max())],'submission_message':d['submission_message'],
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'packages':{p:importlib.metadata.version(p)for p in ['numpy','pandas','scikit-learn','lightgbm','threadpoolctl']}}
    exclusive(manifest,result);print('PREPARED',candidate,'changed',changed,flush=True)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('decision',type=Path);args=p.parse_args();prepare(args.decision)
