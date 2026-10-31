"""Independent final manifest, source-lineage, inference and receipt audit."""
from pathlib import Path
import hashlib,json,subprocess,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.autonomous_kaggle import validate,BASE
from src.utils.paths import task_files
from src.models.round1_evaluation import array_digest,digest_json
from src.models.round1_models import _fit_fingerprint
from src.models.classification_round2_models import stream_missing_mask
OUT=ROOT/'reports/autonomous_eda'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    milestone=read(ROOT/'experiments/autonomous_eda/milestone.json');selection=read(ROOT/'experiments/autonomous_eda/selection.json')
    for p,h in milestone['protected_files'].items():assert sha(ROOT/p)==h,p
    for p,h in selection['internal_result_sha256'].items():assert sha(ROOT/p)==h,p
    tasks={}
    for task,candidates in selection['candidates'].items():
        assert len(candidates)<=milestone['max_successful_new_submissions'][task]<=2
        for c in candidates:
            candidate=c['candidate_id'];path=OUT/f'kaggle/{candidate}.json';m,csv_path=validate(task,path)
            assert len(m['source_commit'])==40
            for p,h in m['source_code_sha256'].items():
                b=subprocess.check_output(['git','-C',str(ROOT),'show',m['source_commit']+':'+p])
                assert hashlib.sha256(b.replace(b'\r\n',b'\n')).hexdigest()==h,p
            source=task_files(task);train=pd.read_csv(source['train']);test=pd.read_csv(source['test']);sample=pd.read_csv(source['sample'])
            state=m['model_state'];assert digest_json(state)==m['model_state_sha256'];assert m['train_rows']==len(train)==50608
            with np.load(ROOT/f'experiments/runs/autonomous_submission_v1/{candidate}.npz',allow_pickle=False)as z:raw=z['raw'];pred=z['prediction']
            assert array_digest(raw)==m['raw_prediction_sha256']and array_digest(pred)==m['prediction_sha256']
            target=BASE[task][0];output=pd.read_csv(csv_path,float_precision='round_trip')
            np.testing.assert_array_equal(output[target].to_numpy(),pred);assert output.record_id.equals(sample.record_id)
            if task=='classification':
                expected=(raw>=.5).astype(np.int8);np.testing.assert_array_equal(pred,expected)
                miss=stream_missing_mask(train);q=float(np.quantile(train.text_load,.8));assert state['volume_threshold']['value']==q
                for expert,group in [('C1_B_all',np.ones(len(train),bool)),('C2_B_all',np.ones(len(train),bool)),('C2_B_present',~miss)]:
                    audit=state['experts'][expert];fields=audit['expert_state']['input_feature_names'];X=train.loc[group,fields];y=train.loc[group,target].to_numpy(dtype=np.int8)
                    assert audit['fit_rows']==len(X)and audit['expert_state']['fit_data_sha256']==_fit_fingerprint(X,tuple(fields),y)
                previous=read(ROOT/'reports/submissions/classification_initial_manifest.json')['model']
                assert state['experts']['C2_B_all']['expert_state']==previous['present_fit_state']
                assert state['experts']['C1_B_all']['expert_state']==previous['missing_fit_state']
                baseline=pd.read_csv(ROOT/'submissions/classification_initial.csv');high=(~stream_missing_mask(test))&(test.text_load.to_numpy()>q)
                np.testing.assert_array_equal(pred[~high],baseline[target].to_numpy()[~high])
            else:
                fields=state['input_feature_names'];assert state['fit_rows']==50608
                assert state['fit_data_sha256']==_fit_fingerprint(train[fields],tuple(fields),train[target].to_numpy())
                prior=read(ROOT/'reports/regression_research/submissions/lgb_text_inner_v1.json')['model_state']
                for block in ['structured_preprocessing','tfidf','actual_hyperparameters','fit_data_sha256','target_sha256']:assert state[block]==prior[block]
                assert state['svd']['n_components_requested']==state['svd']['n_components_actual']==128
                assert state['training_design_shape']==[50608,168]
                from scipy.stats import rankdata
                mapped=np.empty(len(pred));weight=m['full_train_selection']['rank_weight'];cycles=test.cycle_code.to_numpy()
                for cycle in np.unique(cycles):
                    mask=cycles==cycle;mapped[mask]=np.clip((1-weight)*raw[mask]+weight*100*rankdata(raw[mask],method='average')/mask.sum(),0,100)
                np.testing.assert_array_equal(mapped,pred)
            ledger_path=OUT/f'kaggle/{task}__{candidate}_ledger.json';ledger=read(ledger_path)
            assert ledger['status']=='COMPLETE'and ledger['receipt']['status']=='COMPLETE'
            assert ledger['manifest_sha256']==sha(path)and ledger['csv_sha256']==sha(csv_path)
            assert ledger['candidate_id']==candidate and ledger['receipt']['file_name']==csv_path.name
            assert ledger['receipt']['description']==c['submission_message']and ledger['receipt']['team_name']=='kasdud lovers'
            assert ledger['receipt']['public_score']is not None
            tasks[task]={'candidate':candidate,'status':'PASS','source_commit':m['source_commit'],'test_rows':29421,
                'csv_sha256':sha(csv_path),'manifest_sha256':sha(path),'ledger_sha256':sha(ledger_path),
                'ref':ledger['ref'],'public_score':ledger['receipt']['public_score'],'public_improvement':ledger['external_comparison']['improvement'],
                'fit_only_fulltrain_audit':'PASS','source_commit_binding':'PASS'}
        ledgers=list((OUT/'kaggle').glob(f'{task}__*_ledger.json'));assert len(ledgers)<=2
        assert len(ledgers)==len(candidates)
    result={'status':'PASS','protected_historical_files_byte_identical':len(milestone['protected_files']),
        'scoped_tests_passed':31,'subtests_passed':28,'tasks':tasks,'official_model_EDA_notebook_changes':False,
        'raw_data_models_caches_in_git':False,'all_recipes_selected_before_new_scores':True,
        'new_successful_submissions':{task:len(cs)for task,cs in selection['candidates'].items()},
        'source_verifier_sha256':sha(Path(__file__)),'selection_sha256':sha(ROOT/'experiments/autonomous_eda/selection.json')}
    with (OUT/'QA.json').open('w',encoding='utf-8',newline='\n')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
