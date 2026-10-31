"""Refit only internally frozen candidates; build and audit official inference.

This never uploads and never changes official final models or notebooks.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,importlib.metadata,json,subprocess,sys,time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_initial_submissions import _load_inputs
from src.models.regression_research_common import mapped_scores
from src.models.round1_evaluation import array_digest,digest_json
from src.models.round1_models import _fit_fingerprint
from scripts.submit_regression_research import ordered_ids_sha256
OUT=ROOT/'reports/autonomous_eda/kaggle';CACHE=ROOT/'experiments/runs/autonomous_submission_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def code_sha(p):return hashlib.sha256(Path(p).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n')as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n')

def make_frame(test,sample,values,task):
    target='artifact_signal'if task=='classification'else'threat_rank'
    p=np.asarray(values);assert p.shape==(29421,)and np.isfinite(p).all()
    assert list(sample.columns)==['record_id',target]and sample.record_id.equals(test.record_id)
    assert test.record_id.is_unique and test.record_id.notna().all()
    if task=='classification':assert np.isin(p,[0,1]).all();p=p.astype(np.int8)
    else:assert ((p>=0)&(p<=100)).all()
    return pd.DataFrame({'record_id':test.record_id,target:p})

def get_model(recipe,params=None):
    if recipe['kind']=='present_specialist':
        from src.models.autonomous_classification import build_autonomous_classifier
        return build_autonomous_classifier('present_specialist')
    if recipe['kind']=='volume_route':
        from src.models.autonomous_volume_route import build_volume_route_classifier
        return build_volume_route_classifier()
    if recipe['kind']in ['svd128_paired','svd128_inner']:
        from src.models.autonomous_regression import build_autonomous_regression
        return build_autonomous_regression(params or recipe.get('params_id','base'))
    raise ValueError('Unregistered internally selected candidate')

def full_inner_selection(train,recipe):
    choices=[];references=[];fit=np.flatnonzero(train.cycle_code.between(47,75));rows=np.flatnonzero(train.cycle_code.between(76,79))
    for param in ['base','regularized']:
        path=ROOT/f'experiments/runs/autonomous_reg_v1/svd128__{param}__fit75__to79.npz'
        meta_path=path.with_suffix('.json')
        if path.exists()and meta_path.exists():
            meta=read(meta_path)
            with np.load(path,allow_pickle=False)as z:
                np.testing.assert_array_equal(z['fit_rows'],fit);np.testing.assert_array_equal(z['validation_rows'],rows);p=z['raw_prediction'].copy()
            assert array_digest(p)==meta['raw_prediction_sha256']
            state=meta['state'];assert meta['state_sha256']==digest_json(state)
            assert state['svd']['n_components_requested']==128 and state['params_id']==param
            fields=state['input_feature_names']
            assert state['fit_data_sha256']==_fit_fingerprint(train.iloc[fit][fields],tuple(fields),train.iloc[fit].threat_rank.to_numpy())
            references.append({'param':param,'origin':'verified_prior_fit75_outer_cache','prediction_sha256':array_digest(p)})
        else:
            model=get_model(recipe,param);fields=list(model.feature_names)
            with threadpool_limits(limits=2):
                model.fit(train.iloc[fit][fields],train.iloc[fit].threat_rank.to_numpy());p=model.predict(train.iloc[rows][fields])
            references.append({'param':param,'origin':'fresh_fulltrain_inner_fit47_75_predict76_79','state':model.audit_state(),'prediction_sha256':array_digest(p)})
        for weight in [0.,.25,.5,.75,1.]:
            q=mapped_scores(p,train.iloc[rows].cycle_code.to_numpy(),weight)
            choices.append({'param':param,'rank_weight':weight,'mae':float(np.abs(q-train.iloc[rows].threat_rank.to_numpy()).mean())})
    selected=min(choices,key=lambda c:(c['mae'],c['rank_weight'],0 if c['param']=='regularized'else 1))
    return selected,choices,references

def prepare(candidate_id):
    path=ROOT/'experiments/autonomous_eda/selection.json';selection=read(path)
    entries=[(t,c)for t,cs in selection['candidates'].items()for c in cs if c['candidate_id']==candidate_id]
    assert len(entries)==1;task,candidate=entries[0];recipe=candidate['recipe']
    assert candidate['eligible_for_external_evaluation']is True
    for p,h in selection['source_code_sha256'].items():assert code_sha(ROOT/p)==h,p
    for p,h in selection['internal_result_sha256'].items():assert sha(ROOT/p)==h,p
    manifest_path=OUT/f'{candidate_id}.json';assert not manifest_path.exists(),'Candidate already prepared; do not overwrite'
    train,test,sample,input_sha=_load_inputs(task)
    target='artifact_signal'if task=='classification'else'threat_rank';assert target not in test
    assert set(train.cycle_code)==set(range(47,80))and set(test.cycle_code)==set(range(80,92))
    policy={};choices=[];inner_refs=[];selection_origin=None
    if task=='regression':
        if recipe['kind']=='svd128_inner':
            policy,choices,inner_refs=full_inner_selection(train,recipe)
            selection_origin='SVD128 past-only fit47..75 -> predict76..79; same original 2param x 5rank grid'
        else:
            previous=read(ROOT/'reports/regression_research/submissions/lgb_text_inner_v1.json')
            policy={k:previous['full_train_selection'][k]for k in ['param','rank_weight']};assert policy['param']=='base'and policy['rank_weight']==.5
            selection_origin='Paired ablation reuses original SVD32 full-Train parameter/rank choice; no new SVD128 selection'
    model=get_model(recipe,policy.get('param'));fields=list(getattr(model,'input_feature_names',getattr(model,'feature_names',[])))
    assert fields and not {'record_id','artifact_signal','threat_rank'}.intersection(fields)
    before=time.perf_counter()
    with threadpool_limits(limits=2):
        model.fit(train[fields],train[target].to_numpy());state=model.audit_state()
        raw=model.predict_proba(test[fields])if task=='classification'else model.predict(test[fields])
    assert state==model.audit_state()
    values=(raw>=.5).astype(np.int8)if task=='classification'else mapped_scores(raw,test.cycle_code.to_numpy(),policy['rank_weight'])
    frame=make_frame(test,sample,values,task)
    baseline_path=ROOT/('submissions/classification_initial.csv'if task=='classification'else'submissions/regression_research/lgb_text_inner_v1.csv')
    baseline=pd.read_csv(baseline_path,float_precision='round_trip');assert baseline.record_id.equals(test.record_id)
    changed=int(np.sum(frame[target].to_numpy()!=baseline[target].to_numpy()));assert changed>0,'No duplicate baseline submission'
    routing_audit=None
    if task=='classification'and recipe['kind']=='volume_route':
        from src.models.classification_round2_models import stream_missing_mask
        missing=stream_missing_mask(test);cutoff=state['volume_threshold']['value']
        high=(~missing)&(pd.to_numeric(test.text_load,errors='coerce').to_numpy()>cutoff)
        np.testing.assert_array_equal(values[~high],baseline[target].to_numpy()[~high])
        routing_audit={'fit_only_Q80':cutoff,'test_stream_missing':int(missing.sum()),'test_present_above_Q80':int(high.sum()),
            'test_other_rows':int((~high).sum()),'other_rows_labels_exactly_match_accepted_baseline':True,
            'all_changed_labels_are_in_high_present_group':True,'changed_labels_in_high_present':changed,
            'expert_fit_rows':{k:v['fit_rows']for k,v in state['experts'].items()}}
    for mpath in OUT.glob('*.json'):
        existing=read(mpath)
        if existing.get('task')==task and existing.get('status')=='PREPARED_AND_QA_PASS':
            other=pd.read_csv(ROOT/existing['csv_path'],float_precision='round_trip')
            if np.array_equal(other[target].to_numpy(),frame[target].to_numpy()):
                print('DUPLICATE_PREDICTIONS',candidate_id,'equals',existing['candidate_id'],'do not submit twice',flush=True)
                save(OUT/f'{candidate_id}_duplicate.json',{'task':task,'candidate_id':candidate_id,'status':'SKIP_DUPLICATE_PREDICTIONS','same_as':existing['candidate_id'],'recipe':recipe,'full_train_selection':policy,'inner_choices':choices})
                return
    csv_path=ROOT/f'submissions/autonomous_eda/{candidate_id}.csv';csv_path.parent.mkdir(parents=True,exist_ok=True)
    assert not csv_path.exists(),'Refuse overwrite of existing inference'
    frame.to_csv(csv_path,index=False,float_format='%.17g')
    reread=pd.read_csv(csv_path,float_precision='round_trip');assert reread.record_id.equals(sample.record_id)
    np.testing.assert_array_equal(reread[target].to_numpy(),values)
    CACHE.mkdir(parents=True,exist_ok=True);np.savez_compressed(CACHE/f'{candidate_id}.npz',raw=raw,prediction=values)
    from src.utils.paths import task_files
    inputs=task_files(task);input_paths={}
    for role,p in inputs.items():
        try:input_paths[role]=p.relative_to(ROOT).as_posix()
        except ValueError:input_paths[role]=str(p)
    manifest={'status':'PREPARED_AND_QA_PASS','candidate_id':candidate_id,'task':task,
        'competition':f'tp2-{task}-cyber-vulnerability-kasdd-gasal-2026-2027','recipe':recipe,
        'main_selection':candidate,'selection_file_sha256':sha(path),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'source_code_sha256':selection['source_code_sha256'],'input_sha256':input_sha,'input_relative_paths':input_paths,
        'csv_path':csv_path.relative_to(ROOT).as_posix(),'csv_sha256':sha(csv_path),
        'ordered_test_ids_sha256':ordered_ids_sha256(frame.record_id.astype(str).tolist()),
        'train_rows':len(train),'test_rows':len(test),'full_train_selection':policy,'full_train_selection_origin':selection_origin,'inner_choices':choices,'inner_references':inner_refs,
        'threshold':.5 if task=='classification'else None,'fit_preprocessing_only_train':True,'test_labels_used':False,
        'candidate_differs_from_accepted_baseline':True,'changed_predictions_vs_baseline':changed,
        'routing_audit':routing_audit,
        'model_state':state,'model_state_sha256':digest_json(state),'prediction_sha256':array_digest(values),'raw_prediction_sha256':array_digest(raw),
        'prediction_range':[float(values.min()),float(values.max())],'class_counts':{str(i):int((values==i).sum())for i in [0,1]}if task=='classification'else None,
        'submission_message':candidate['submission_message'],'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'fit_predict_seconds':time.perf_counter()-before,'packages':{p:importlib.metadata.version(p)for p in ['numpy','pandas','scikit-learn','lightgbm','threadpoolctl']},
        'operational_limit':'Rank mapping, when used, requires supplied cycle batches; source-universe completeness remains unknown.'}
    save(manifest_path,manifest);print('PREPARED',candidate_id,'changed',changed,'policy',policy,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('candidate_id');args=p.parse_args();prepare(args.candidate_id)
