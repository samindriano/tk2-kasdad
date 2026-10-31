"""Fixed-feature temporal ablation; no model search, Test access or upload."""
from pathlib import Path
import hashlib
import json
import sys
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.utils.paths import task_files
from src.models.classification_models import build_classifier,_array_digest,_json_digest
from src.models.classification_round2_models import stream_missing_mask
from src.models.classification_ttr_ablation import build_variant,text_features
from src.models.classification_metrics import binary_metrics

CONFIG=ROOT/'experiments/classification_ttr'
REPORT=ROOT/'reports/classification_ttr'
CACHE=ROOT/'experiments/runs/classification_ttr_v1'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def code_sha(p):return hashlib.sha256(Path(p).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
def save(p,x):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def score(frame,rows,p):
    f=frame.iloc[rows];y=f.artifact_signal.to_numpy();missing=stream_missing_mask(f)
    cycles=[{'cycle':int(c),**binary_metrics(y[f.cycle_code.to_numpy()==c],p[f.cycle_code.to_numpy()==c],.5)}for c in sorted(f.cycle_code.unique())]
    result={'metrics':binary_metrics(y,p,.5),'equal_cycle_macro_f1':float(np.mean([c['macro_f1']for c in cycles])),
        'per_cycle':cycles,'subgroups':{}}
    masks={'stream_present':~missing,'stream_missing':missing,
           'EF0':f.entry_friction.eq(0).to_numpy(),'EF1':f.entry_friction.eq(1).to_numpy()}
    for name,mask in masks.items():result['subgroups'][name]=binary_metrics(y[mask],p[mask],.5)
    result['probability_sha256']=_array_digest(p)
    return result


def decide(results,config):
    agg=results['aggregate'];baseline={fam:agg[fam]['C4_HGB']for fam in ['primary','secondary']}
    candidates={}
    for candidate in ['R_TTR','R_BOTH']:
        a,b=agg['primary'][candidate],agg['secondary'][candidate]
        gate=config['gate'];deltas={}
        for fam in ['primary','secondary']:
            deltas[fam]=agg[fam][candidate]['metrics']['macro_f1']-baseline[fam]['metrics']['macro_f1']
        pairs=[j for j in results['jobs']if j['model']==candidate]
        window_deltas=[]
        for j in pairs:
            old=next(o for o in results['jobs']if o['model']=='C4_HGB'and o['window']==j['window'])
            window_deltas.append({'window':j['window'],'family':j['family'],
                'delta':j['metrics']['macro_f1']-old['metrics']['macro_f1']})
        eq=a['equal_cycle_macro_f1']-baseline['primary']['equal_cycle_macro_f1']
        checks={'primary_gain':deltas['primary']>=gate['min_primary_f1_gain'],
            'equal_cycle_gain':eq>=gate['min_primary_equal_cycle_gain'],
            'h12_gain':deltas['secondary']>=gate['min_h12_f1_gain'],
            'windows_positive':sum(v['delta']>0 for v in window_deltas if v['family']=='primary')>=gate['primary_windows_positive_at_least'],
            'no_serious_window_loss':min(v['delta']for v in window_deltas)>=gate['min_any_window_delta'],
            'class_f1_guard':all(agg[fam][candidate]['metrics']['per_class'][i]['f1']-baseline[fam]['metrics']['per_class'][i]['f1']>=gate['min_each_class_f1_delta']for fam in ['primary','secondary']for i in [0,1]),
            'present_guard':all(agg[fam][candidate]['subgroups']['stream_present']['macro_f1']-baseline[fam]['subgroups']['stream_present']['macro_f1']>=gate['min_stream_present_delta']for fam in ['primary','secondary']),
            'missing_exact':results['route_missing_exact']}
        candidates[candidate]={'deltas':deltas,'primary_equal_cycle_delta':eq,'window_deltas':window_deltas,'gate_checks':checks,'eligible':all(checks.values())}
    eligible=[m for m,d in candidates.items()if d['eligible']]
    chosen=max(eligible,key=lambda m:(agg['primary'][m]['metrics']['macro_f1'],agg['secondary'][m]['metrics']['macro_f1']))if eligible else'C4_HGB'
    return {'recommended':chosen,'candidates':candidates,'submission_gate_pass':bool(eligible),
            'action':'Prepare one candidate only after independent QA; no upload executed by this runner.'if eligible else'Keep existing C4_HGB; no additional Kaggle submission.',
            'public_feedback_used':False}


def main():
    cfg=read(CONFIG/'protocol.json');source=task_files('classification')['train']
    assert sha(source)==cfg['input_sha256']
    freeze=read(CONFIG/'freeze.json')
    assert sha(CONFIG/'protocol.json')==freeze['protocol_sha256']
    effective={**freeze['code_sha256'],**freeze.get('reporting_code_amendment',{}).get('code_sha256',{})}
    for p,h in effective.items():assert code_sha(ROOT/p)==h,p
    historical=read(ROOT/'experiments/classification_round1/freeze.json')['identity']['code_sha256']
    for p,h in historical.items():assert code_sha(ROOT/p)==h,('historical source changed',p)
    frame=pd.read_csv(source);assert frame.shape==(50608,19)and frame.record_id.is_unique
    REPORT.mkdir(exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    models={**cfg['controls'],**{m:m for m in cfg['new_lr_variants']}}
    groups={};audits=[];start=time.perf_counter();fresh=0;reused=0
    for end in [55,59,63,67,71,75]:
        last=max(w['validation'][1]for w in cfg['primary_windows']+cfg['secondary_windows']if w['fit'][1]==end)
        fit=np.flatnonzero(frame.cycle_code.between(47,end));rows=np.flatnonzero(frame.cycle_code.between(end+1,last))
        groups[end]={};base_audit=read(ROOT/f'reports/classification_round1/fit_audits/fit{end}.json')
        for name,source_id in models.items():
            path=CACHE/f'fit{end}__{name}.npz';meta_path=path.with_suffix('.json')
            key=_json_digest({'freeze':freeze['signature'],'name':name,'end':end,'fit':_array_digest(fit),'rows':_array_digest(rows)})
            origin='new_fit';meta={}
            old=ROOT/f'experiments/runs/classification_round1_v1/fit{end}__{source_id}.npz'
            if name in cfg['controls'] and old.exists():
                with np.load(old,allow_pickle=False)as z:
                    assert np.array_equal(z['outer_rows'],rows)
                    p=z['probability'].copy()
                assert _array_digest(p)==base_audit[source_id]['probability_sha256']
                state=base_audit[source_id]['fitted_state'];origin='verified_historical_control_vector';reused+=1
            elif path.exists()or meta_path.exists():
                assert path.exists()and meta_path.exists(),'Incomplete cache pair'
                meta=read(meta_path);assert meta['key']==key
                with np.load(path,allow_pickle=False)as z:
                    assert np.array_equal(z['rows'],rows)and np.array_equal(z['fit'],fit)
                    p=z['p'].copy()
                assert _array_digest(p)==meta['probability_sha256'];state=meta['state'];origin='verified_new_cache';reused+=1
            else:
                estimator=build_classifier({'id':source_id})if name in cfg['controls']else build_variant(name)
                fields=estimator.input_feature_names
                with threadpool_limits(limits=2):
                    estimator.fit(frame.iloc[fit][fields].copy(),frame.iloc[fit].artifact_signal.to_numpy())
                    state=estimator.audit_state();p=estimator.predict_proba(frame.iloc[rows][fields].copy())
                assert state==estimator.audit_state(),'Prediction altered fitted state'
                if name in cfg['controls']:assert _array_digest(p)==base_audit[source_id]['probability_sha256']
                else:
                    prior=base_audit[estimator.source_id]['fitted_state']
                    for block in ['preprocessor','scaler','text']:
                        if block in prior:assert state[block]==prior[block],(name,block)
                np.savez_compressed(path,fit=fit,rows=rows,p=p)
                save(meta_path,{'key':key,'probability_sha256':_array_digest(p),'state':state})
                fresh+=1
            assert p.shape==(len(rows),)and np.isfinite(p).all()
            groups[end][name]={'rows':rows,'p':p}
            audits.append({'fit_end':end,'model':name,'origin':origin,'fit_rows':len(fit),'validation_rows':len(rows),
                'probability_sha256':_array_digest(p),'fit_fingerprint':state['fit_data_sha256'],
                'derived_features':state.get('derived_features'),'preprocessing_blocks_match_control':True})
            print('fit',end,name,origin,flush=True)
        missing=stream_missing_mask(frame.iloc[rows])
        for name,(present,fallback)in cfg['routes'].items():
            groups[end][name]={'rows':rows,'p':np.where(missing,groups[end][fallback]['p'],groups[end][present]['p'])}
    jobs=[];aggregate={'primary':{},'secondary':{}};route_missing_exact=True
    for family,windows in [('primary',cfg['primary_windows']),('secondary',cfg['secondary_windows'])]:
        chunks={}
        for w in windows:
            for name,bundle in groups[w['fit'][1]].items():
                if name=='HGB':continue
                mask=frame.iloc[bundle['rows']].cycle_code.between(*w['validation']).to_numpy()
                rows=bundle['rows'][mask];p=bundle['p'][mask]
                result=score(frame,rows,p);jobs.append({'family':family,'window':w['id'],'model':name,**result})
                chunks.setdefault(name,[]).append((rows,p))
                if name.startswith('R_'):
                    missing=stream_missing_mask(frame.iloc[rows]);baseline=groups[w['fit'][1]]['C4_HGB']['p'][mask]
                    route_missing_exact &= np.array_equal(p[missing],baseline[missing])
        for name,parts in chunks.items():
            rows=np.concatenate([z[0]for z in parts]);p=np.concatenate([z[1]for z in parts]);order=np.argsort(rows)
            assert len(np.unique(rows))==38588
            aggregate[family][name]=score(frame,rows[order],p[order])
    old=read(ROOT/'reports/classification_round2/results.json')
    for fam in ['primary','secondary']:
        for name,oldname in [('H0','C2_B'),('S0','C1_A'),('C4_HGB','C4_HGB')]:
            a=aggregate[fam][name]['metrics'];b=old['aggregate'][fam]['fixed_0.5'][oldname]['metrics']
            for field in ['mean_probability','brier_score']:
                assert abs(a[field]-b[field])<1e-15,(fam,name,field)
            metadata={'threshold','threshold_range','mean_probability','brier_score'}
            assert {k:v for k,v in a.items()if k not in metadata}=={k:v for k,v in b.items()if k not in metadata},(fam,name,'baseline metric identity')
    feature=text_features(frame.signal_stream);present=~stream_missing_mask(frame)
    results={'status':'COMPLETE_DEVELOPMENT_ABLATION','freeze_signature':freeze['signature'],'input_sha256':sha(source),
        'fresh_fits_this_run':fresh,'reused_controls_or_new_caches':reused,'audits':audits,'jobs':jobs,'aggregate':aggregate,
        'route_missing_exact':bool(route_missing_exact),'threshold':.5,'test_labels_used':False,'kaggle_uploads':0,
        'diagnostic':{'present_ttr_count_pearson':float(feature.loc[present,'ttr'].corr(feature.loc[present,'count'])),
                      'present_ttr_text_load_pearson':float(feature.loc[present,'ttr'].corr(frame.loc[present,'text_load'])),
                      'missing_rows':int((~present).sum())},'seconds':time.perf_counter()-start}
    results['decision']=decide(results,cfg)
    save(REPORT/'results.json',results)
    table=[]
    for name,a in aggregate['primary'].items():
        b=aggregate['secondary'][name]
        table.append({'model':name,'primary':a['metrics']['macro_f1'],'equal_cycle':a['equal_cycle_macro_f1'],
            'H12':b['metrics']['macro_f1'],'present_primary':a['subgroups']['stream_present']['macro_f1'],
            'missing_primary':a['subgroups']['stream_missing']['macro_f1'],
            'present_H12':b['subgroups']['stream_present']['macro_f1'],'missing_H12':b['subgroups']['stream_missing']['macro_f1']})
    pd.DataFrame(table).to_csv(REPORT/'comparison.csv',index=False)
    print(pd.DataFrame(table).to_string(index=False));print('DECISION',results['decision']['recommended'],flush=True)


if __name__=='__main__':main()
