"""Full official Train refit of fixed C2_B-present / CatBoost-missing routing."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, importlib.metadata, subprocess, sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scripts import prepare_campaign_candidate as base
from src.models.campaign_classification_catboost import build_catboost_classifier
from src.models.classification_round2_models import stream_missing_mask

original_factory=base.component_model
def factory(task,component):
    if task=='classification' and component['model']=='CB_MISSING':return build_catboost_classifier()
    return original_factory(task,component)

def prepare(path):
    path=path.resolve();path.relative_to((ROOT/'experiments/dual_campaign/decisions').resolve())
    d=base.read(path);assert d['approved_for_external_evaluation'] and d['task']=='classification'
    assert d['recipe']=={'kind':'availability_route','present':'C2_B','missing':'CB_MISSING','threshold':0.5}
    for p,h in d['source_code_sha256'].items():assert base.codesha(ROOT/p)==h,p
    for p,h in d['internal_result_sha256'].items():assert base.sha(ROOT/p)==h,p
    candidate=d['candidate_id'];assert __import__('re').fullmatch('[a-z0-9_]{1,80}',candidate)
    manifest=base.OUT/(candidate+'.json');csv=ROOT/f'submissions/dual_campaign/{candidate}.csv'
    assert not manifest.exists() and not csv.exists()
    train,test,sample,hashes=base._load_inputs('classification')
    assert len(train)==50608 and len(test)==29421 and not {'artifact_signal','threat_rank'}.intersection(test.columns)
    identity={'input_sha256':hashes,'source_code_sha256':d['source_code_sha256']}
    base.component_model=factory
    present,present_audit=base.fit_component('classification',{'model':'C2_B'},train,test,identity)
    missing,missing_audit=base.fit_component('classification',{'model':'CB_MISSING'},train,test,identity)
    old=base.read(ROOT/'reports/submissions/classification_initial_manifest.json')['model']
    assert present_audit['state']==old['present_fit_state'],'Frozen C2_B full-Train recipe differs'
    mask=stream_missing_mask(test);probability=np.where(mask,missing,present);values=(probability>=.5).astype(np.int8)
    baseline=pd.read_csv(ROOT/'submissions/classification_initial.csv')
    assert baseline.record_id.equals(test.record_id) and sample.record_id.equals(test.record_id)
    assert list(sample.columns)==['record_id','artifact_signal']
    delta=values!=baseline.artifact_signal.to_numpy()
    assert int(delta.sum())>0 and not delta[~mask].any(),'Present branch must remain exactly unchanged'
    for directory in [base.OUT,ROOT/'reports/autonomous_eda/kaggle']:
        for p in directory.glob('*.json'):
            m=base.read(p)
            if m.get('task')=='classification' and m.get('status')=='PREPARED_AND_QA_PASS':
                previous=pd.read_csv(ROOT/m['csv_path'])
                assert not np.array_equal(values,previous.artifact_signal.to_numpy()),'Duplicate candidate'
    csv.parent.mkdir(parents=True,exist_ok=True)
    frame=pd.DataFrame({'record_id':test.record_id,'artifact_signal':values});frame.to_csv(csv,index=False)
    again=pd.read_csv(csv);assert again.record_id.equals(sample.record_id)
    np.testing.assert_array_equal(again.artifact_signal,values)
    np.savez_compressed(base.CACHE/(candidate+'.npz'),prediction=values,probability=probability,
                        present=present,missing=missing,missing_mask=mask)
    result={'status':'PREPARED_AND_QA_PASS','task':'classification','candidate_id':candidate,
            'competition':'tp2-classification-cyber-vulnerability-kasdd-gasal-2026-2027',
            'recipe':d['recipe'],'decision_path':path.relative_to(ROOT).as_posix(),'decision_sha256':base.sha(path),
            'source_commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
            'source_code_sha256':d['source_code_sha256'],'input_sha256':hashes,
            'csv_path':csv.relative_to(ROOT).as_posix(),'csv_sha256':base.sha(csv),
            'ordered_test_ids_sha256':base.ordered_ids_sha256(test.record_id.astype(str).tolist()),
            'train_rows':len(train),'test_rows':len(test),'fit_preprocessing_only_train':True,'test_labels_used':False,
            'candidate_differs_from_accepted_baseline':True,'changed_predictions_vs_baseline':int(delta.sum()),
            'changed_present_stream_predictions':int(delta[~mask].sum()),'test_missing_rows':int(mask.sum()),
            'component_audits':[present_audit,missing_audit],'prediction_sha256':base.array_digest(values),
            'probability_sha256':base.array_digest(probability),'prediction_range':[int(values.min()),int(values.max())],
            'submission_message':d['submission_message'],'created_at_utc':datetime.now(timezone.utc).isoformat(),
            'packages':{p:importlib.metadata.version(p) for p in
                        ['numpy','pandas','scikit-learn','lightgbm','catboost','threadpoolctl']}}
    base.exclusive(manifest,result)
    print('PREPARED',candidate,'changed',int(delta.sum()),'missing-only; Train',len(train),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('decision',type=Path);args=parser.parse_args();prepare(args.decision)
