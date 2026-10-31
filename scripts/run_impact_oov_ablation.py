"""One controlled feature ablation; frozen per-fold params and rank weights."""
from pathlib import Path
import hashlib,json,sys,time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.utils.paths import task_files
from src.models.impact_oov import CoverageLGBText,vocabulary_coverage
from src.models.regression_research_lgb_text import build_lgb_text
from src.models.regression_research_common import mapped_scores,score_summary
from src.models.round1_evaluation import array_digest,digest_json
CFG=ROOT/'experiments/impact_eda';OUT=ROOT/'reports/impact_eda/ablation';CACHE=ROOT/'experiments/runs/impact_eda_oov_v1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def code_sha(p):return hashlib.sha256(Path(p).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
def save(p,d):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    with Path(p).open('w',encoding='utf-8',newline='\n')as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n')

def main():
    cfg=read(CFG/'protocol.json');freeze=read(CFG/'freeze.json')
    assert code_sha(CFG/'protocol.json')==freeze['protocol_sha256']
    for p,h in freeze['code_sha256'].items():assert code_sha(ROOT/p)==h,p
    for p,h in cfg['reference_artifacts'].items():assert sha(ROOT/p)==h,p
    source=task_files('regression')['train'];assert sha(source)==cfg['regression_train_sha256']
    frame=pd.read_csv(source);assert frame.shape==(50608,19)and frame.record_id.is_unique
    old=read(ROOT/'reports/regression_research/stage5_lgb_text.json');CACHE.mkdir(parents=True,exist_ok=True)
    vectors={};audits=[];fresh=0;started=time.perf_counter()
    for choice in cfg['frozen_choices']:
        end,param,weight=choice['fit_end'],choice['param'],choice['rank_weight']
        assert choice=={'fit_end':end,**{k:v for k,v in old['audits'][str(end)]['selection'].items()if k in ['param','rank_weight']}}
        fit=np.flatnonzero(frame.cycle_code.between(47,end));last=max(w['validation'][1]for w in cfg['primary_windows']+cfg['secondary_windows']if w['fit'][1]==end)
        rows=np.flatnonzero(frame.cycle_code.between(end+1,last));prior=old['audits'][str(end)]['outer']
        legacy=ROOT/f'experiments/runs/regression_research_v1/lgbtext__{param}__fit{end}__to{last}.npz'
        ref_path=CACHE/f'fit{end}__control.npz'
        if legacy.exists():
            with np.load(legacy,allow_pickle=False)as z:
                np.testing.assert_array_equal(z['fit_rows'],fit);np.testing.assert_array_equal(z['validation_rows'],rows);base=z['prediction'].copy()
        elif ref_path.exists():
            with np.load(ref_path,allow_pickle=False)as z:
                np.testing.assert_array_equal(z['fit'],fit);np.testing.assert_array_equal(z['rows'],rows);base=z['p'].copy()
        else:
            model=build_lgb_text(param)
            with threadpool_limits(limits=2):
                model.fit(frame.iloc[fit][list(model.feature_names)],frame.iloc[fit].threat_rank.to_numpy());base=model.predict(frame.iloc[rows][list(model.feature_names)])
            np.savez_compressed(ref_path,fit=fit,rows=rows,p=base)
        assert array_digest(base)==prior['prediction_sha256']
        path=CACHE/f'fit{end}__coverage.npz';meta=path.with_suffix('.json')
        key=digest_json({'freeze':freeze['signature'],'choice':choice,'fit':array_digest(fit),'rows':array_digest(rows)})
        if path.exists()or meta.exists():
            assert path.exists()and meta.exists();m=read(meta);assert m['key']==key
            with np.load(path,allow_pickle=False)as z:
                np.testing.assert_array_equal(z['fit'],fit);np.testing.assert_array_equal(z['rows'],rows);pred=z['p'].copy();coverage=z['coverage'].copy()
            assert array_digest(pred)==m['prediction_sha256'];state=m['state'];origin='verified_cache'
        else:
            model=CoverageLGBText(param);fields=list(model.feature_names)
            with threadpool_limits(limits=2):
                model.fit(frame.iloc[fit][fields],frame.iloc[fit].threat_rank.to_numpy());state=model.audit_state()
                pred=model.predict(frame.iloc[rows][fields]);coverage=vocabulary_coverage(model._vectorizer,frame.iloc[rows].signal_stream).oov_fraction.to_numpy()
            assert model.audit_state()==state
            m={'key':key,'state':state,'prediction_sha256':array_digest(pred),'coverage_sha256':array_digest(coverage),'fit_rows':len(fit)}
            np.savez_compressed(path,fit=fit,rows=rows,p=pred,coverage=coverage);save(meta,m);origin='new_fit';fresh+=1
        for block in ['structured_preprocessing','tfidf','svd','actual_hyperparameters','fit_data_sha256','target_sha256']:
            assert state[block]==prior['state'][block],(end,block)
        assert state['training_design_shape'][1]==prior['state']['training_design_shape'][1]+1
        assert np.isfinite(pred).all()and array_digest(coverage)==m['coverage_sha256']
        vectors[end]={'rows':rows,'control':mapped_scores(base,frame.iloc[rows].cycle_code,weight),
            'coverage':mapped_scores(pred,frame.iloc[rows].cycle_code,weight)}
        audits.append({'fit_end':end,'choice':choice,'origin':origin,'fit_rows':len(fit),'validation_rows':len(rows),
            'control_raw_sha256':array_digest(base),'candidate_raw_sha256':array_digest(pred),'coverage_sha256':array_digest(coverage),'state_sha256':state['state_sha256']})
        print('fit',end,origin,flush=True)
    jobs=[];aggregate={};table=[]
    for family,key in [('primary','primary_windows'),('secondary','secondary_windows')]:
        parts={'control':[],'coverage':[]}
        for w in cfg[key]:
            v=vectors[w['fit'][1]];mask=frame.iloc[v['rows']].cycle_code.between(*w['validation']).to_numpy();rows=v['rows'][mask]
            for name in parts:
                pred=v[name][mask];s=score_summary(frame.iloc[rows],pred)
                jobs.append({'family':family,'window':w['id'],'model':name,'prediction_sha256':array_digest(pred),**s});parts[name].append((rows,pred))
        aggregate[family]={}
        for name,chunks in parts.items():
            rows=np.concatenate([r for r,p in chunks]);pred=np.concatenate([p for r,p in chunks]);idx=np.argsort(rows);assert len(rows)==len(np.unique(rows))==38588
            s=score_summary(frame.iloc[rows[idx]],pred[idx]);aggregate[family][name]=s
            table.append({'family':family,'model':name,'mae':s['raw']['mae'],'equal_cycle':s['equal_cycle_mae'],'missing':s['subgroups']['stream_missing']['mae'],'present':s['subgroups']['stream_present']['mae']})
        expected=old['aggregate'][family]['lgb_text__inner']
        assert abs(aggregate[family]['control']['raw']['mae']-expected['raw']['mae'])<1e-13
        assert abs(aggregate[family]['control']['equal_cycle_mae']-expected['equal_cycle_mae'])<1e-13
    result={'freeze_signature':freeze['signature'],'input_sha256':sha(source),'fresh_fits':fresh,'audits':audits,'jobs':jobs,'aggregate':aggregate,
        'official_test_read':False,'kaggle_submissions':0,'hyperparameter_or_weight_selection':False,'seconds':time.perf_counter()-started}
    result['gate']=evaluate_gate(result,cfg['gate']);save(OUT/'results.json',result)
    pd.DataFrame(table).to_csv(OUT/'comparison.csv',index=False)
    rows=[]
    for j in jobs:
        base=next(b for b in jobs if b['family']==j['family']and b['window']==j['window']and b['model']=='control')
        rows.append({'family':j['family'],'window':j['window'],'model':j['model'],'mae':j['raw']['mae'],'gain_vs_final':base['raw']['mae']-j['raw']['mae']})
    pd.DataFrame(rows).to_csv(OUT/'windows.csv',index=False)
    print(pd.DataFrame(table).to_string(index=False));print(result['gate'],flush=True)

def evaluate_gate(r,g):
    a=r['aggregate'];gain={f:a[f]['control']['raw']['mae']-a[f]['coverage']['raw']['mae']for f in a}
    window=[]
    for j in r['jobs']:
        if j['model']!='coverage':continue
        b=next(b for b in r['jobs']if b['model']=='control'and b['window']==j['window'])
        window.append((j['family'],b['raw']['mae']-j['raw']['mae']))
    checks={'primary':gain['primary']>=g['min_primary_gain'],'H12':gain['secondary']>=g['min_H12_gain'],
        'equal_cycle':a['primary']['control']['equal_cycle_mae']-a['primary']['coverage']['equal_cycle_mae']>=g['min_equal_cycle_gain'],
        'consistency':sum(v>0 for f,v in window if f=='primary')>=g['positive_primary_windows'],
        'window_guard':min(v for f,v in window)>=-g['max_window_loss'],
        'subgroups':all(a[f]['coverage']['subgroups'][s]['mae']-a[f]['control']['subgroups'][s]['mae']<=g['max_subgroup_loss']for f in a for s in ['stream_missing','stream_present','EF0','EF1','tail_low_le5','tail_high_ge95'])}
    return {'gains':gain,'checks':checks,'passes':all(checks.values()),'new_impact':'supported_development_only'if all(checks.values())else'not_convincing',
        'official_model_changed':False}
if __name__=='__main__':main()
