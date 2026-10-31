"""Full-Train artifact for one approved fixed sparse present-expert blend."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,importlib.metadata,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scripts import prepare_campaign_candidate as base
from src.models.campaign_classification_catboost import build_catboost_classifier
from src.models.optimization_v3_sparse_classification import SparseTextBoostClassifier
from src.models.classification_round2_models import stream_missing_mask

OUT=ROOT/'reports/optimization_v3/kaggle'
CACHE=ROOT/'experiments/runs/optimization_v3_inference'
original_factory=base.component_model
def factory(task,c):
    if task=='classification'and c['model']=='CB_MISSING':return build_catboost_classifier()
    if task=='classification'and c['model']=='SPARSE_BINARY':return SparseTextBoostClassifier()
    return original_factory(task,c)

def prepare(path):
    path=path.resolve();path.relative_to((ROOT/'experiments/optimization_v3/decisions').resolve())
    d=base.read(path);assert d['task']=='classification'and d['approved_for_external_evaluation']
    recipe={'kind':'availability_present_blend','present':[{'model':'C2_B','weight':.75},{'model':'SPARSE_BINARY','weight':.25}],
            'missing':'CB_MISSING','threshold':.5}
    assert d['recipe']==recipe
    for p,h in d['source_code_sha256'].items():assert base.codesha(ROOT/p)==h,p
    for p,h in d['internal_result_sha256'].items():assert base.sha(ROOT/p)==h,p
    candidate=d['candidate_id'];assert __import__('re').fullmatch('[a-z0-9_]{1,80}',candidate)
    manifest=OUT/(candidate+'.json');csv=ROOT/f'submissions/dual_campaign/optimization_v3/{candidate}.csv'
    assert not manifest.exists()and not csv.exists()
    train,test,sample,hashes=base._load_inputs('classification')
    assert len(train)==50608 and len(test)==29421 and not {'artifact_signal','threat_rank'}.intersection(test.columns)
    assert sample.record_id.equals(test.record_id)
    base.CACHE=CACHE;base.component_model=factory
    identity={'input_sha256':hashes,'source_code_sha256':d['source_code_sha256']}
    scores={};audits=[]
    for name in['C2_B','CB_MISSING','SPARSE_BINARY']:
        p,a=base.fit_component('classification',{'model':name},train,test,identity);scores[name]=p;audits.append(a)
    initial=base.read(ROOT/'reports/submissions/classification_initial_manifest.json')['model']
    assert audits[0]['state']==initial['present_fit_state'],'OriginalC2B full-Train state must remain exact'
    current=base.read(ROOT/'reports/dual_campaign/kaggle/dc_cls_cb_missing_v1.json')
    old_state=current['component_audits'][1]['state'];new_state=audits[1]['state']
    # CBM serialization includes run metadata; deterministic prediction and fit
    # lineage are checked separately from that binary-container checksum.
    for name in['fit_rows','fit_data_sha256','parameters','native_category_features','output_feature_names','fit_only_preprocessor','tree_count']:
        assert new_state[name]==old_state[name],name
    previous=ROOT/'experiments/runs/dual_inference_v1/dc_cls_cb_missing_v1.npz'
    if previous.exists():
        with np.load(previous,allow_pickle=False)as z:
            np.testing.assert_array_equal(scores['C2_B'],z['present']);np.testing.assert_array_equal(scores['CB_MISSING'],z['missing'])
    mask=stream_missing_mask(test);champ_probability=np.where(mask,scores['CB_MISSING'],scores['C2_B'])
    probability=np.where(mask,scores['CB_MISSING'],.75*scores['C2_B']+.25*scores['SPARSE_BINARY'])
    values=(probability>=.5).astype(np.int8);champ=(champ_probability>=.5).astype(np.int8)
    assert np.isfinite(probability).all()and((probability>=0)&(probability<=1)).all()
    assert np.array_equal(values[mask],champ[mask])
    changed=int(np.count_nonzero(values!=champ));assert changed>0
    # Detect duplicate prepared or accepted CSV payloads across research phases.
    for directory in['reports/dual_campaign/kaggle','reports/autonomous_eda/kaggle','reports/optimization_v2/kaggle','reports/optimization_v3/kaggle']:
        for p in(ROOT/directory).glob('*.json'):
            m=base.read(p)
            if m.get('status')=='PREPARED_AND_QA_PASS'and m.get('task')=='classification':
                previous_csv=ROOT/m['csv_path']
                if previous_csv.exists():
                    old=pd.read_csv(previous_csv)
                    assert not np.array_equal(values,old.artifact_signal.to_numpy()),'Duplicate CSVpayload'
    csv.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'record_id':test.record_id,'artifact_signal':values}).to_csv(csv,index=False)
    again=pd.read_csv(csv);assert again.record_id.equals(sample.record_id)
    np.testing.assert_array_equal(again.artifact_signal,values)
    np.savez_compressed(CACHE/(candidate+'.npz'),prediction=values,probability=probability,base=scores['C2_B'],missing=scores['CB_MISSING'],sparse=scores['SPARSE_BINARY'],missing_mask=mask)
    result={'status':'PREPARED_AND_QA_PASS','task':'classification','candidate_id':candidate,
            'competition':'tp2-classification-cyber-vulnerability-kasdd-gasal-2026-2027','recipe':recipe,
            'decision_path':path.relative_to(ROOT).as_posix(),'decision_sha256':base.sha(path),
            'source_commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
            'source_code_sha256':d['source_code_sha256'],'input_sha256':hashes,
            'csv_path':csv.relative_to(ROOT).as_posix(),'csv_sha256':base.sha(csv),
            'ordered_test_ids_sha256':base.ordered_ids_sha256(test.record_id.astype(str).tolist()),
            'train_rows':len(train),'test_rows':len(test),'fit_preprocessing_only_train':True,'test_labels_used':False,
            'candidate_differs_from_accepted_baseline':True,'parent_champion_ref':'57833406',
            'changed_predictions_vs_baseline':changed,'changed_missing_branch_labels':int(np.count_nonzero(values[mask]!=champ[mask])),
            'component_audits':audits,'prediction_sha256':base.array_digest(values),'probability_sha256':base.array_digest(probability),
            'prediction_range':[int(values.min()),int(values.max())],'submission_message':d['submission_message'],
            'created_at_utc':datetime.now(timezone.utc).isoformat(),
            'packages':{p:importlib.metadata.version(p)for p in['numpy','pandas','scipy','scikit-learn','lightgbm','catboost','threadpoolctl']}}
    base.exclusive(manifest,result);print('PREPARED',candidate,'changedpresent',changed,'changedmissing',result['changed_missing_branch_labels'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('decision',type=Path);a=p.parse_args();prepare(a.decision)
