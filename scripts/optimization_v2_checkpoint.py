"""Current-state completion audit for the isolated performance campaign."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from scripts import optimization_v2_classification as calibration
from scripts import optimization_v2_stack as stack
from scripts.optimization_v2_kaggle import gate
from src.models.round1_models import _fit_fingerprint

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def independent_recent_heads():
    frozen=calibration.check();frame,y,_=calibration.load_train(None);cycles=frame.cycle_code.to_numpy(dtype=int)
    first=calibration.earliest(frame,y,frozen['signature'],False)
    old={e:calibration.existing(frame,y,e)for e in calibration.ENDS}
    report=read(calibration.OUT/'results.json');checks=0
    for end in calibration.ENDS:
        if end==55:r,p=first
        else:
            rr,ss,_=old[end-4];use=(cycles[rr]>=end-3)&(cycles[rr]<=end);r,p=rr[use],ss[use]
        assert cycles[r].min()==end-3 and cycles[r].max()==end
        use=~calibration.stream_missing_mask(frame.iloc[r]);pp=np.clip(p[use],1e-6,1-1e-6);design=np.log(pp/(1-pp))
        rr,ss,cb=old[end];miss=calibration.stream_missing_mask(frame.iloc[rr])
        oo=np.clip(ss,1e-6,1-1e-6);outer=np.log(oo/(1-oo))
        with np.load(calibration.CACHE/f'outer{end}.npz',allow_pickle=False)as z:
            np.testing.assert_array_equal(z['rows'],rr)
            for kind,columns in [('calibrated_base',[0]),('calibrated_nb',[1]),('stack_base_nb',[0,1])]:
                with threadpool_limits(limits=2):
                    head=LogisticRegression(C=1.,solver='lbfgs',max_iter=1000,tol=1e-8,random_state=42).fit(design[:,columns],y[r][use])
                state=report['states'][str(end)]['heads'][kind]
                np.testing.assert_allclose(head.coef_,state['coef'],rtol=0,atol=1e-10)
                np.testing.assert_allclose(head.intercept_,state['intercept'],rtol=0,atol=1e-10)
                expected=np.where(miss,cb,head.predict_proba(outer[:,columns])[:,1])
                np.testing.assert_allclose(z[kind],expected,rtol=0,atol=1e-10);checks+=1
    return checks

def regression_reconcile():
    r=read(ROOT/'reports/optimization_v2/regression_repr/fullsix.json')
    frame=pd.read_csv(gate.task_files('regression')['train']);rows_all=[];actual=[];predictions=[]
    with np.load(ROOT/r['new_prediction_cache_path'],allow_pickle=False)as new, \
         np.load(ROOT/'experiments/runs/optimization_v2_regression_repr/pilot_predictions.npz',allow_pickle=False)as pilot, \
         np.load(ROOT/'experiments/runs/dual_reg_v1/stage1_paired_blends.npz',allow_pickle=False)as old:
        for end in calibration.ENDS:
            z=pilot if end in[55,67]else new;rows=z[f'rows_lgbm_sparse_l1_{end}'];raw=z[f'raw_lgbm_sparse_l1_{end}']
            np.testing.assert_array_equal(rows,old[f'rows_{end}']);cycles=frame.iloc[rows].cycle_code.to_numpy();mapped=np.empty_like(raw)
            for c in np.unique(cycles):
                mask=cycles==c;mapped[mask]=np.clip(.5*raw[mask]+50*rankdata(raw[mask],method='average')/mask.sum(),0,100)
            pred=.75*old[f'blend_ridge25_{end}']+.25*mapped;use=cycles<=end+4
            rows_all.append(rows[use]);actual.append(frame.iloc[rows[use]].threat_rank.to_numpy());predictions.append(pred[use])
    rows=np.concatenate(rows_all);y=np.concatenate(actual);p=np.concatenate(predictions)
    assert len(np.unique(rows))==len(rows)==38588
    mae=float(np.mean(np.abs(y-p)))
    np.testing.assert_allclose(mae,r['candidate_gates']['lgbm_sparse_l1']['primary_candidate']['mae'],rtol=0,atol=1e-12)
    assert r['candidate_gates']['lgbm_sparse_l1']['all_gates_pass']
    return {'rows':len(rows),'MAE':mae,'rank_and_mixture_recomputed_independently':True}

def verify(output):
    milestone=read(ROOT/'experiments/optimization_v2/milestone.json');protected=milestone['historical_before_sha256']
    for name,h in protected.items():assert sha(ROOT/name)==h,name
    assert read(calibration.OUT/'verification.json')['status']=='PASS'
    assert read(stack.OUT/'verification.json')['status']=='PASS';stack.check()
    head_checks=independent_recent_heads();reg=regression_reconcile()
    final_checks={}
    for path in ['reports/optimization_v2/regression_repr/verification.json',
                 'reports/optimization_v2/regression_repr/fullsix_verification_context.json',
                 'reports/optimization_v2/classification_text/verification.json']:
        d=read(ROOT/path);assert d['status']=='PASS';final_checks[path]=sha(ROOT/path)
    from scripts.optimization_v2_verify_catboost_text_v2 import verify as verify_native
    assert verify_native()['status']=='PASS'
    unigram_verifications=[ROOT/'reports/optimization_v2/classification_unigram'/p for p in ['verification_v2.json','source_restoration.json']]
    for p in unigram_verifications:
        assert read(p)['status']=='PASS';final_checks[p.relative_to(ROOT).as_posix()]=sha(p)
    from scripts.optimization_v2_verify_unigram_v2 import verify as verify_unigram
    assert verify_unigram()['status']=='PASS'
    replay=read(ROOT/'reports/optimization_v2/replay_QA.json')
    assert replay['status']=='PASS' and replay['fresh_full_train_fits']==3
    assert replay['prediction_bit_identical_to_accepted_manifest'] and not replay['ignored_cache_or_prior_CSV_read']
    for fname in ['freeze.json','fullsix_freeze.json']:
        d=read(ROOT/'experiments/optimization_v2/regression_repr'/fname)
        for name,h in d['source_sha256'].items():assert sha(ROOT/name)==h,name
    accepted=[];counts={'classification':0,'regression':0}
    for ledger in (ROOT/'reports/optimization_v2/kaggle').glob('*__*_ledger.json'):
        d=read(ledger);task=d['task'];path=ROOT/f"reports/optimization_v2/kaggle/{d['candidate_id']}.json"
        m,csv=gate.validate(task,path)
        assert d['status']==d['receipt']['status']=='COMPLETE'and d['receipt']['public_score']is not None
        assert d['csv_sha256']==sha(csv)and d['manifest_sha256']==sha(path)and d['num_allowed_before']>1
        for audit in m['component_audits']:assert audit['fit_rows']==50608
        if task=='regression':
            train=pd.read_csv(gate.task_files(task)['train']);test=pd.read_csv(gate.task_files(task)['test'])
            cy=test.cycle_code.to_numpy();parts=[]
            for component,audit in zip(m['recipe']['components'],m['component_audits']):
                fields=audit['features'];assert audit['state']['fit_data_sha256']==_fit_fingerprint(train[fields],tuple(fields),train.threat_rank.to_numpy())
                with np.load(ROOT/f"experiments/runs/optimization_v2_inference/{audit['key']}.npz",allow_pickle=False)as z:raw=z['prediction'].copy()
                rank=np.empty_like(raw)
                for c in np.unique(cy):
                    use=cy==c;rank[use]=100*rankdata(raw[use],method='average')/use.sum()
                w=component['rank_weight'];parts.append(component['weight']*np.clip((1-w)*raw+w*rank,0,100))
            expected=np.clip(np.sum(parts,axis=0),0,100)
            np.testing.assert_array_equal(expected,pd.read_csv(csv,float_precision='round_trip').threat_rank.to_numpy())
        counts[task]+=1;accepted.append({'task':task,'ref':d['ref'],'public_score':d['receipt']['public_score'],'manifest_sha256':sha(path)})
    for task,count in counts.items():assert count<=milestone['research_upload_budget'][task]
    live={task:read(ROOT/f'reports/optimization_v2/kaggle/{task}_preflight.json')for task in counts}
    assert all(d['public_leaderboard']['team']['team_name']=='kasdud lovers'for d in live.values())
    result={'status':'PASS','at_utc':datetime.now(timezone.utc).isoformat(),'protected_preexisting_files':len(protected),
            'academic_notebooks_EDA_teammate_files_byte_identical':True,'recent_heads_independently_rebuilt':head_checks,
            'expanding_heads_independent_verification':sha(stack.OUT/'verification.json'),
            'regression_root_reconciliation':reg,'study_verifications':final_checks,
            'accepted_submissions':accepted,'upload_counts':counts,
            'fresh_raw_data_replay':sha(ROOT/'reports/optimization_v2/replay_QA.json'),
            'leaderboard_snapshot':{k:v['public_leaderboard']for k,v in live.items()},
            'quota_remaining':{k:v['num_allowed_now']for k,v in live.items()},
            'notebooks_modified':0,'Test_labels_used':False,'SCeLE_or_ZIP':False,
            'limitations':['Repeated known Train outcomes; adaptive development, no untouched-holdout claim.',
                           'Public score is not proof of private score; old champions preserved.']}
    calibration.save(Path(output),result);print('PASS',len(protected),'protected files;',head_checks,'reconstructed heads;',counts)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();verify(a.output)
