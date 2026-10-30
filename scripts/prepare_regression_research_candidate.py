"""MAIN-approved research candidate full-Train refit, never submits automatically."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,re,subprocess,sys
from time import perf_counter
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.models.regression_research_common import CONFIG,REPORT,CACHE,save_json,verify_stage,mapped_scores
from src.models.round1_evaluation import digest_json,array_digest
from src.models.regression_round2_evaluation import positions_digest
from src.models.regression_research_text import build_text_model
from src.models.regression_research_tabular import build_tabular
from src.utils.paths import PROJECT_ROOT,task_files
from src.data.audit import sha256
from scripts.research_regression_models import select_inner,state_of
from scripts.submit_regression_research import ordered_ids_sha256


def candidate_model(kind, param):
    if kind in {'svd128','svd_noscope32','structured_ridge'}:
        from src.models.regression_research_refinement import build_refinement
        return build_refinement(kind)
    return build_tabular(param)if kind=='lgb_inner'else build_text_model(kind)


def prepare(decision_path):
    decision_path=Path(decision_path).resolve()
    if not decision_path.is_relative_to(CONFIG.resolve()):raise ValueError('Approval must live in research configuration root.')
    decision=json.loads(decision_path.read_text(encoding='utf-8'))
    if decision['signature']!=digest_json(decision['payload']):raise ValueError('Candidate approval changed.')
    d=decision['payload'];candidate=d['candidate_id']
    if not re.fullmatch(r'[a-z0-9_]{1,60}',candidate):raise ValueError('Invalid candidate ID')
    if not d['main_approval']['approved']:raise ValueError('MAIN approval required')
    stage=verify_stage(d['stage']);kind=d['recipe']['kind']
    if kind not in {'text_ridge','hybrid_ridge','svd_hgb','lgb_inner','svd128','svd_noscope32','structured_ridge'}:raise ValueError('Unregistered candidate family')
    if d['stage']not in {'stage23_models','stage4_refinement'}:raise ValueError('Wrong model-stage identity')
    files=task_files('regression');train=pd.read_csv(files['train']);test=pd.read_csv(files['test']);sample=pd.read_csv(files['sample'])
    if len(train)!=50608 or len(test)!=29421 or not test.record_id.is_unique or test.record_id.isna().any():raise ValueError('Official row contract differs')
    if list(sample.columns)!=['record_id','threat_rank']or not sample.record_id.equals(test.record_id):raise ValueError('Official sample/test IDs differ')
    if 'threat_rank'in test:raise ValueError('Unexpected official Test target')
    param=d['recipe'].get('params_id','base');weight=d['recipe'].get('rank_weight')
    if weight is not None and not 0<=weight<=1:raise ValueError('Invalid predeclared rank weight')
    if d['recipe'].get('selection')not in {'inner','fixed'}:raise ValueError('Candidate must specify frozen selection policy')
    if d['recipe'].get('selection')=='inner':
        # Full-training selection is the same frozen past-only scheme, now 47..75->76..79.
        choices=[]
        for p in (['base','regularized','deeper']if kind=='lgb_inner'else['base']):
            label=f'refine__{kind}__fit75__to79'if d['stage']=='stage4_refinement'else f'model__{kind}__{p}__fit75__to79'
            meta_path=CACHE/(label+'.json');vec_path=CACHE/(label+'.npz')
            fit=np.flatnonzero(train.cycle_code.between(47,75));val=np.flatnonzero(train.cycle_code.between(76,79))
            if meta_path.exists()and vec_path.exists():
                meta=json.loads(meta_path.read_text(encoding='utf-8'))
                with np.load(vec_path,allow_pickle=False)as z:
                    if not np.array_equal(z['fit_rows'],fit)or not np.array_equal(z['validation_rows'],val):raise RuntimeError('Inner cache positions mismatch')
                    pred=z['prediction'].copy()
                key={'freeze':stage['signature'],'kind':kind,'fit':positions_digest(fit),'validation':positions_digest(val)}
                if d['stage']=='stage23_models':key['param']=p
                expected=digest_json(key)
                if meta['key']!=expected or meta['prediction_sha256']!=array_digest(pred):raise RuntimeError('Inner cache identity mismatch')
            else:
                model=candidate_model(kind,p)
                with threadpool_limits(limits=2):
                    model.fit(train.iloc[fit][list(model.feature_names)],train.iloc[fit].threat_rank.to_numpy())
                    pred=model.predict(train.iloc[val][list(model.feature_names)])
            for a in stage['identity']['protocol']['rank_weights']:
                mapped=mapped_scores(pred,train.iloc[val].cycle_code,a)
                choices.append({'param':p,'rank_weight':a,'mae':float(np.abs(mapped-train.iloc[val].threat_rank.to_numpy()).mean())})
        selected=select_inner(choices);param,weight=selected['param'],selected['rank_weight']
    else:
        choices=[];selected={'param':param,'rank_weight':weight}
    model=candidate_model(kind,param)
    start=perf_counter()
    with threadpool_limits(limits=2):
        features=list(model.feature_names)
        model.fit(train[features].copy(),train.threat_rank.to_numpy(copy=True))
        state=state_of(model);raw=model.predict(test[features].copy())
    if digest_json(state)!=digest_json(state_of(model)):raise RuntimeError('Prediction mutated model fit')
    prediction=np.clip(raw,0,100)if weight is None else mapped_scores(raw,test.cycle_code,weight)
    if not np.isfinite(prediction).all()or not np.all((prediction>=0)&(prediction<=100)):raise RuntimeError('Invalid submission values')
    output=PROJECT_ROOT/'submissions/regression_research'/(candidate+'.csv');output.parent.mkdir(parents=True,exist_ok=True)
    values=pd.DataFrame({'record_id':test.record_id,'threat_rank':prediction})
    csv=values.to_csv(index=False,float_format='%.17g').encode('utf-8')
    if output.exists()and output.read_bytes()!=csv:raise RuntimeError('Refusing overwrite of existing candidate CSV')
    output.write_bytes(csv)
    reread=pd.read_csv(output,float_precision='round_trip')
    if not reread.record_id.equals(sample.record_id)or not np.array_equal(reread.threat_rank,prediction):raise RuntimeError('CSV round trip changed output')
    manifest={'status':'KAGGLE_CANDIDATE_APPROVED','task':'regression','candidate_id':candidate,
              'competition':'tp2-regression-cyber-vulnerability-kasdd-gasal-2026-2027',
              'main_approval':d['main_approval'],'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'protocol_sha256':digest_json(stage['identity']['protocol']),'stage_signature':stage['signature'],
              'decision_sha256':sha256(decision_path),'recipe':d['recipe'],'full_train_selection':selected,'inner_choices':choices,
              'input_sha256':{k:sha256(v)for k,v in files.items()},'csv_sha256':sha256(output),'fit_rows':len(train),'test_rows':len(test),
              'test_ids_sha256':ordered_ids_sha256(test.record_id.astype(str).tolist()),
              'test_labels_used':False,'fit_preprocessing_only_train':True,'model_state':state,'model_state_sha256':digest_json(state),
              'prediction_sha256':array_digest(prediction),'raw_range':[float(np.min(raw)),float(np.max(raw))],
              'prediction_range':[float(np.min(prediction)),float(np.max(prediction))],
              'zero_scope_cohort_not_given_labels':True,'fit_predict_seconds':perf_counter()-start,
              'created_at_utc':datetime.now(timezone.utc).isoformat(),'csv_path':str(output.relative_to(PROJECT_ROOT)),
              'batch_assumption':'When rank weight >0, mapping uses all supplied Test rows percycle; source-universe completeness remains UNKNOWN.'}
    target=REPORT/'submissions'/(candidate+'.json')
    save_json(target,manifest,exclusive=True)
    print('PREPARED',output,'manifest',target,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('decision',type=Path);a=p.parse_args();prepare(a.decision)
