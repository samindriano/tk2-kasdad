"""Full-Train Stage5 candidate preparation, preserving earlier frozen writers."""
from pathlib import Path
import json,sys,subprocess
from datetime import datetime,timezone
from time import perf_counter
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.models.regression_research_common import CONFIG,REPORT,CACHE,verify_stage,save_json,mapped_scores
from src.models.regression_research_lgb_text import build_lgb_text
from src.models.regression_round2_evaluation import positions_digest
from src.models.round1_evaluation import digest_json,array_digest
from src.utils.paths import PROJECT_ROOT,task_files
from src.data.audit import sha256
from scripts.research_regression_models import select_inner
from scripts.submit_regression_research import ordered_ids_sha256


def main():
    path=CONFIG/'lgb_text_inner_v1_approval.json';decision=json.loads(path.read_text(encoding='utf-8'));d=decision['payload']
    assert decision['signature']==digest_json(d)and d['main_approval']['approved']is True
    assert d['candidate_id']=='lgb_text_inner_v1'
    frozen=verify_stage('stage5_lgb_text');files=task_files('regression');f=pd.read_csv(files['train']);test=pd.read_csv(files['test']);sample=pd.read_csv(files['sample'])
    assert len(f)==50608 and len(test)==29421 and f.record_id.is_unique and test.record_id.is_unique and not test.record_id.isna().any()
    assert 'threat_rank'not in test and list(sample.columns)==['record_id','threat_rank']and sample.record_id.equals(test.record_id)
    choices=[];fit=np.flatnonzero(f.cycle_code.between(47,75));val=np.flatnonzero(f.cycle_code.between(76,79));start=perf_counter()
    for param in ['base','regularized']:
        label=f'lgbtext__{param}__fit75__to79';meta_path=CACHE/(label+'.json');vec_path=CACHE/(label+'.npz')
        if meta_path.exists()and vec_path.exists():
            m=json.loads(meta_path.read_text(encoding='utf-8'))
            with np.load(vec_path,allow_pickle=False)as z:
                assert np.array_equal(z['fit_rows'],fit)and np.array_equal(z['validation_rows'],val);pred=z['prediction'].copy()
            assert m['prediction_sha256']==array_digest(pred)and m['key']==digest_json({'freeze':frozen['signature'],'param':param,'fit':positions_digest(fit),'validation':positions_digest(val)})
        else:
            model=build_lgb_text(param)
            with threadpool_limits(limits=2):
                model.fit(f.iloc[fit][list(model.feature_names)],f.iloc[fit].threat_rank.to_numpy());pred=model.predict(f.iloc[val][list(model.feature_names)])
        for a in frozen['identity']['protocol']['rank_weights']:
            q=mapped_scores(pred,f.iloc[val].cycle_code,a);choices.append({'param':param,'rank_weight':a,'mae':float(np.abs(q-f.iloc[val].threat_rank.to_numpy()).mean())})
    selected=select_inner(choices);model=build_lgb_text(selected['param']);features=list(model.feature_names)
    with threadpool_limits(limits=2):
        model.fit(f[features].copy(),f.threat_rank.to_numpy(copy=True));state=model.audit_state();raw=model.predict(test[features].copy())
    assert state==model.audit_state();pred=mapped_scores(raw,test.cycle_code,selected['rank_weight'])
    assert np.isfinite(pred).all()and ((pred>=0)&(pred<=100)).all()
    output=PROJECT_ROOT/'submissions/regression_research/lgb_text_inner_v1.csv';output.parent.mkdir(parents=True,exist_ok=True)
    csv=pd.DataFrame({'record_id':test.record_id,'threat_rank':pred}).to_csv(index=False,float_format='%.17g').encode('utf-8')
    if output.exists():assert output.read_bytes()==csv
    else:output.write_bytes(csv)
    back=pd.read_csv(output,float_precision='round_trip');assert back.record_id.equals(sample.record_id)and np.array_equal(back.threat_rank,pred)
    manifest={'status':'KAGGLE_CANDIDATE_APPROVED','task':'regression','candidate_id':d['candidate_id'],
              'competition':'tp2-regression-cyber-vulnerability-kasdd-gasal-2026-2027','main_approval':d['main_approval'],
              'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'preparation_script_sha256':sha256(Path(__file__)),
              'protocol_sha256':digest_json(frozen['identity']['protocol']),'stage_signature':frozen['signature'],
              'recipe':d['recipe'],'full_train_selection':selected,'inner_choices':choices,
              'input_sha256':{k:sha256(v)for k,v in files.items()},'csv_sha256':sha256(output),
              'test_ids_sha256':ordered_ids_sha256(test.record_id.astype(str).tolist()),'fit_rows':50608,'test_rows':29421,
              'test_labels_used':False,'fit_preprocessing_only_train':True,'model_state':state,'model_state_sha256':digest_json(state),
              'prediction_sha256':array_digest(pred),'raw_range':[float(min(raw)),float(max(raw))],
              'fit_predict_seconds':perf_counter()-start,'created_at_utc':datetime.now(timezone.utc).isoformat(),
              'csv_path':str(output.relative_to(PROJECT_ROOT)),'batch_assumption':'All provided Test rows grouped bycycle; real source completeness remains UNKNOWN.'}
    save_json(REPORT/'submissions/lgb_text_inner_v1.json',manifest,exclusive=True)
    print('PREPARED',output,'full-train inner selection',selected,flush=True)


if __name__=='__main__':main()
