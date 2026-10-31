"""Exclusive full-Train refit for an approved sparse-regression challenger."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,importlib.metadata,subprocess,sys,warnings
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scripts import prepare_campaign_candidate as base
from src.models.optimization_v2_regression import SparseHybridRegressor

OUT=ROOT/'reports/optimization_v2/kaggle'
CACHE=ROOT/'experiments/runs/optimization_v2_inference'
original_factory=base.component_model

def factory(task,component):
    if task=='regression'and component['model']in ['sgd_l1','lgbm_sparse_l1']:
        return SparseHybridRegressor(component['model'])
    return original_factory(task,component)

def mix(parts,components,cycles):
    assert len(parts)==len(components) and parts
    assert all(0<=float(c['weight'])<=1 and 0<=float(c['rank_weight'])<=1 for c in components)
    assert abs(sum(float(c['weight'])for c in components)-1)<1e-12
    assert all(np.asarray(p).shape==np.asarray(cycles).shape and np.isfinite(p).all()for p in parts)
    scores=[float(c['weight'])*base.mapped_scores(p,cycles,float(c['rank_weight']))for p,c in zip(parts,components)]
    return np.clip(np.sum(scores,axis=0),0,100)

def prepare(path):
    path=path.resolve();path.relative_to((ROOT/'experiments/optimization_v2/decisions').resolve())
    d=base.read(path);assert d['task']=='regression'and d['approved_for_external_evaluation']
    candidate=d['candidate_id'];assert __import__('re').fullmatch('[a-z0-9_]{1,80}',candidate)
    for name,h in d['source_code_sha256'].items():assert base.codesha(ROOT/name)==h,name
    for name,h in d['internal_result_sha256'].items():assert base.sha(ROOT/name)==h,name
    manifest=OUT/(candidate+'.json');csv=ROOT/f'submissions/dual_campaign/optimization_v2/{candidate}.csv'
    assert not manifest.exists()and not csv.exists(),'Never overwrite a candidate artifact'
    train,test,sample,hashes=base._load_inputs('regression')
    assert len(train)==50608 and len(test)==29421 and not {'artifact_signal','threat_rank'}.intersection(test.columns)
    assert sample.record_id.equals(test.record_id)and list(sample.columns)==['record_id','threat_rank']
    base.CACHE=CACHE;base.component_model=factory
    identity={'input_sha256':hashes,'source_code_sha256':d['source_code_sha256']};parts=[];audits=[];fit_warnings=[]
    for component in d['recipe']['components']:
        with warnings.catch_warnings(record=True)as caught:
            warnings.simplefilter('always')
            p,audit=base.fit_component('regression',component,train,test,identity)
        assert audit['fit_rows']==len(train)and audit['learned_only_official_Train']
        parts.append(p);audits.append(audit)
        fit_warnings.append({'model':component['model'],'warnings':[{'category':w.category.__name__,'message':str(w.message)}for w in caught]})
    values=mix(parts,d['recipe']['components'],test.cycle_code.to_numpy())
    incumbent=pd.read_csv(ROOT/'submissions/dual_campaign/dc_reg_ridge_25_v1.csv',float_precision='round_trip')
    assert incumbent.record_id.equals(test.record_id)
    changed=int(np.count_nonzero(values!=incumbent.threat_rank.to_numpy()));assert changed>0
    for namespace in ['reports/dual_campaign/kaggle','reports/autonomous_eda/kaggle','reports/optimization_v2/kaggle']:
        for p in (ROOT/namespace).glob('*.json'):
            other=base.read(p)
            if other.get('task')=='regression'and other.get('status')=='PREPARED_AND_QA_PASS':
                previous=pd.read_csv(ROOT/other['csv_path'],float_precision='round_trip')
                assert not np.array_equal(values,previous.threat_rank.to_numpy()),'Duplicate prepared/accepted payload'
    csv.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'record_id':test.record_id,'threat_rank':values}).to_csv(csv,index=False,float_format='%.17g')
    again=pd.read_csv(csv,float_precision='round_trip');assert again.record_id.equals(sample.record_id)
    np.testing.assert_array_equal(values,again.threat_rank.to_numpy())
    CACHE.mkdir(parents=True,exist_ok=True);np.savez_compressed(CACHE/(candidate+'.npz'),prediction=values)
    result={'status':'PREPARED_AND_QA_PASS','task':'regression','candidate_id':candidate,
            'competition':'tp2-regression-cyber-vulnerability-kasdd-gasal-2026-2027','recipe':d['recipe'],
            'decision_path':path.relative_to(ROOT).as_posix(),'decision_sha256':base.sha(path),
            'source_commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
            'source_code_sha256':d['source_code_sha256'],'input_sha256':hashes,'csv_path':csv.relative_to(ROOT).as_posix(),
            'csv_sha256':base.sha(csv),'ordered_test_ids_sha256':base.ordered_ids_sha256(test.record_id.astype(str).tolist()),
            'train_rows':len(train),'test_rows':len(test),'fit_preprocessing_only_train':True,'test_labels_used':False,
            'candidate_differs_from_accepted_baseline':True,'changed_predictions_vs_baseline':changed,
            'parent_champion_ref':'57832438','component_audits':audits,'fit_warnings':fit_warnings,
            'prediction_sha256':base.array_digest(values),'prediction_range':[float(values.min()),float(values.max())],
            'submission_message':d['submission_message'],'created_at_utc':datetime.now(timezone.utc).isoformat(),
            'packages':{p:importlib.metadata.version(p)for p in ['numpy','pandas','scipy','scikit-learn','lightgbm','threadpoolctl']}}
    base.exclusive(manifest,result);print('PREPARED',candidate,'changed',changed,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('decision',type=Path);args=p.parse_args();prepare(args.decision)
