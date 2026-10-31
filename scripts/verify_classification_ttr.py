"""Independent sklearn recalculation and immutable-source checks, no fitting."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support,confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.utils.paths import task_files
from src.models.classification_models import _array_digest,_fit_fingerprint,_json_digest
from src.models.classification_round2_models import stream_missing_mask
from src.models.classification_ttr_ablation import text_features,VARIANTS

def read(p):return json.loads(Path(p).read_text())
def main():
    cfg=read(ROOT/'experiments/classification_ttr/protocol.json')
    freeze=read(ROOT/'experiments/classification_ttr/freeze.json')
    assert freeze['signature']==_json_digest({k:v for k,v in freeze.items()if k not in {'signature','reporting_code_amendment'}})
    assert hashlib.sha256((ROOT/'experiments/classification_ttr/protocol.json').read_bytes()).hexdigest()==freeze['protocol_sha256']
    for p,h in {**freeze['code_sha256'],**freeze.get('reporting_code_amendment',{}).get('code_sha256',{})}.items():
        assert hashlib.sha256((ROOT/p).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==h
    results=read(ROOT/'reports/classification_ttr/results.json')
    train=task_files('classification')['train']
    assert hashlib.sha256(train.read_bytes()).hexdigest()==cfg['input_sha256']==results['input_sha256']
    assert results['freeze_signature']==freeze['signature']
    assert results['threshold']==cfg['threshold']==.5 and not results['test_labels_used']
    frame=pd.read_csv(train);missing=stream_missing_mask(frame)
    features=text_features(frame.signal_stream);checked=0;fitted=0
    preserved=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in cfg['protected_files'].items())
    assert preserved
    bundles={}
    for end in [55,59,63,67,71,75]:
        bundles[end]={};fit=np.flatnonzero(frame.cycle_code.between(47,end))
        last=max(w['validation'][1]for w in cfg['primary_windows']+cfg['secondary_windows']if w['fit'][1]==end)
        expected_rows=np.flatnonzero(frame.cycle_code.between(end+1,last))
        audit=read(ROOT/f'reports/classification_round1/fit_audits/fit{end}.json')
        for name,source in {**cfg['controls'],**{v:v for v in cfg['new_lr_variants']}}.items():
            if name in cfg['controls']:
                path=ROOT/f'experiments/runs/classification_round1_v1/fit{end}__{source}.npz'
                if path.exists():
                    with np.load(path,allow_pickle=False)as z:rows=z['outer_rows'];p=z['probability']
                else:
                    path=ROOT/f'experiments/runs/classification_ttr_v1/fit{end}__{name}.npz'
                    with np.load(path,allow_pickle=False)as z:rows=z['rows'];p=z['p']
                state=audit[source]['fitted_state']
                assert _array_digest(p)==audit[source]['probability_sha256']
            else:
                path=ROOT/f'experiments/runs/classification_ttr_v1/fit{end}__{name}.npz'
                with np.load(path,allow_pickle=False)as z:rows=z['rows'];p=z['p'];np.testing.assert_array_equal(z['fit'],fit)
                state=read(path.with_suffix('.json'))['state']
                raw=features.iloc[fit][list(VARIANTS[name][1])].to_numpy()
                imputer=SimpleImputer(strategy='median',keep_empty_features=True);filled=imputer.fit_transform(raw)
                scaler=StandardScaler().fit(filled);d=state['derived_features']
                np.testing.assert_array_equal(imputer.statistics_,d['imputer_median'])
                np.testing.assert_array_equal(scaler.mean_,d['mean']);np.testing.assert_array_equal(scaler.scale_,d['scale'])
                assert _array_digest(scaler.transform(filled))==d['fit_matrix_sha256']
                prior=audit[VARIANTS[name][0]]['fitted_state']
                for block in ['preprocessor','scaler','text','transformed_features','parameters']:
                    if block in prior:assert prior[block]==state[block],(name,block)
                fitted+=1
            fields=state['input_feature_names']
            np.testing.assert_array_equal(rows,expected_rows)
            assert state['fit_data_sha256']==_fit_fingerprint(frame.iloc[fit][fields],tuple(fields),frame.iloc[fit].artifact_signal.to_numpy(dtype=np.int8))
            assert _array_digest(p)==next(a['probability_sha256']for a in results['audits']if a['fit_end']==end and a['model']==name)
            assert np.isfinite(p).all()and ((p>=0)&(p<=1)).all()
            bundles[end][name]=(rows,p)
        for route,(present,fallback)in cfg['routes'].items():
            rows,p=bundles[end][present];bundles[end][route]=(rows,np.where(missing[rows],bundles[end][fallback][1],p))
        for route in ['R_LEN','R_TTR','R_BOTH']:
            m=missing[expected_rows]
            np.testing.assert_array_equal(bundles[end][route][1][m],bundles[end]['C4_HGB'][1][m])
    groups={}
    for job in results['jobs']:
        w=next(w for w in cfg['primary_windows']+cfg['secondary_windows']if w['id']==job['window'])
        rows,p=bundles[w['fit'][1]][job['model']];mask=frame.iloc[rows].cycle_code.between(*w['validation']).to_numpy()
        rows,p=rows[mask],p[mask];groups.setdefault((job['family'],job['model']),[]).append((rows,p))
        verify_metrics(frame,rows,p,job);checked+=1
    for (family,model),parts in groups.items():
        rows=np.concatenate([r for r,p in parts]);p=np.concatenate([p for r,p in parts]);idx=np.argsort(rows);rows,p=rows[idx],p[idx]
        assert len(rows)==len(np.unique(rows))==38588
        verify_metrics(frame,rows,p,results['aggregate'][family][model]);checked+=1
    assert results['route_missing_exact'] is True
    verify_decision(results,cfg)
    receipt={'status':'PASS','independent_metric_blocks':checked,'fit_only_scalar_audits':fitted,
        'historical_files_byte_identical':len(cfg['protected_files']),'frozen_sources_unchanged':preserved,
        'validation_rows_per_family':38588,'official_test_read':False,'new_kaggle_submissions':0,
        'fit_freeze_signature':freeze['signature'],'reporting_amendment':freeze.get('reporting_code_amendment'),
        'verification_code_sha256':hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        'checks':['sklearn P/R/F1 and confusion matrices','present/missing metrics','equal-cycle mean',
                  'input and protocol/freeze binding','all expected temporal row arrays','independent practical-gate recomputation',
                  'exact missing-route equality','all fit fingerprints','all probability hashes and bounds','fit-only median and scaler',
                  'unchanged base representations and hyperparameters','historical byte preservation']}
    (ROOT/'reports/classification_ttr/QA.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))

def verify_metrics(frame,rows,p,result):
    y=frame.iloc[rows].artifact_signal.to_numpy();pred=(p>=.5).astype(int)
    def check(mask,expected):
        precision,recall,f1,support=precision_recall_fscore_support(y[mask],pred[mask],labels=[0,1],zero_division=0)
        assert abs(f1.mean()-expected['macro_f1'])<1e-14
        for i in [0,1]:
            for key,val in [('precision',precision[i]),('recall',recall[i]),('f1',f1[i]),('support',int(support[i]))]:
                assert abs(expected['per_class'][i][key]-val)<1e-14
        assert confusion_matrix(y[mask],pred[mask],labels=[0,1]).tolist()==expected['confusion_matrix']
    check(np.ones(len(rows),bool),result['metrics'])
    missing=stream_missing_mask(frame.iloc[rows]);check(missing,result['subgroups']['stream_missing']);check(~missing,result['subgroups']['stream_present'])
    cycles=frame.iloc[rows].cycle_code.to_numpy();scores=[]
    for c in np.unique(cycles):
        m=cycles==c;scores.append(precision_recall_fscore_support(y[m],pred[m],labels=[0,1],zero_division=0)[2].mean())
    assert abs(np.mean(scores)-result['equal_cycle_macro_f1'])<1e-14

def verify_decision(results,cfg):
    g=cfg['gate'];agg=results['aggregate'];eligible=[]
    for model in ['R_TTR','R_BOTH']:
        delta={f:agg[f][model]['metrics']['macro_f1']-agg[f]['C4_HGB']['metrics']['macro_f1']for f in agg}
        eq=agg['primary'][model]['equal_cycle_macro_f1']-agg['primary']['C4_HGB']['equal_cycle_macro_f1']
        windows=[]
        for job in results['jobs']:
            if job['model']!=model:continue
            base=next(j for j in results['jobs']if j['model']=='C4_HGB'and j['window']==job['window'])
            windows.append((job['family'],job['metrics']['macro_f1']-base['metrics']['macro_f1']))
        conditions=[delta['primary']>=g['min_primary_f1_gain'],eq>=g['min_primary_equal_cycle_gain'],
            delta['secondary']>=g['min_h12_f1_gain'],sum(d>0 for f,d in windows if f=='primary')>=g['primary_windows_positive_at_least'],
            min(d for f,d in windows)>=g['min_any_window_delta'],
            min(agg[f][model]['metrics']['per_class'][i]['f1']-agg[f]['C4_HGB']['metrics']['per_class'][i]['f1']for f in agg for i in [0,1])>=g['min_each_class_f1_delta'],
            min(agg[f][model]['subgroups']['stream_present']['macro_f1']-agg[f]['C4_HGB']['subgroups']['stream_present']['macro_f1']for f in agg)>=g['min_stream_present_delta'],True]
        decision=results['decision']['candidates'][model]
        assert decision['deltas']==delta and decision['primary_equal_cycle_delta']==eq
        assert list(decision['gate_checks'].values())==conditions
        assert decision['eligible']==all(conditions)
        if all(conditions):eligible.append(model)
    chosen=sorted(eligible,key=lambda m:(agg['primary'][m]['metrics']['macro_f1'],agg['secondary'][m]['metrics']['macro_f1']),reverse=True)[0]if eligible else'C4_HGB'
    assert results['decision']['recommended']==chosen
    assert results['decision']['submission_gate_pass']==bool(eligible)
    assert not results['decision']['public_feedback_used']

if __name__=='__main__':main()
