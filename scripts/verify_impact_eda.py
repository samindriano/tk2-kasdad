"""Independent raw-count, frozen-vector, metric and preservation reconciliation."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import f1_score,mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.utils.paths import task_files
from src.models.classification_round2_models import stream_missing_mask
from src.models.round1_evaluation import array_digest,digest_json
from src.models.regression_research_text import _json_digest,_TFIDF_PARAMS,_text_documents
from src.models.round1_models import _fit_fingerprint
OUT=ROOT/'reports/impact_eda'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def code_sha(p):return hashlib.sha256(Path(p).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
def save(p,d):
    with Path(p).open('w',encoding='utf-8',newline='\n')as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n')
def close(a,b):assert abs(a-b)<1e-12,(a,b)
def mapping(p,cycle,w):
    q=np.empty(len(p),float)
    for c in np.unique(cycle):
        mask=cycle==c;q[mask]=100*rankdata(p[mask],method='average')/mask.sum()
    return np.clip((1-w)*p+w*q,0,100)

def availability(frame,stats):
    frame=frame.copy();missing=stream_missing_mask(frame);frame['_availability']=missing.astype(int)
    frame['_cycle']=frame.cycle_code;frame['_period']=((frame.cycle_code.to_numpy()-47)//11).astype(int)
    for q in [5,10]:frame[f'_volume_q{q}']=pd.qcut(frame.text_load,q,labels=False)
    for g in stats['overall_groups']:
        mask=missing if g['availability']=='missing'else~missing
        assert g['n']==int(mask.sum())and g['artifact_1']==int(frame.loc[mask,'artifact_signal'].sum())
        close(g['artifact_prevalence'],frame.loc[mask,'artifact_signal'].mean())
    comparisons=stats['text_load_sensitivity']['comparisons'];schemes=[comparisons['cycle_only'],*comparisons['volume_only'],*comparisons['period_x_volume']]
    for s in schemes:
        keys=s['strata'];table=frame.groupby(keys+['_availability'],observed=True).artifact_signal.agg(['size','sum'])
        count=0;denom=[0,0];rates=[]
        for cell in s['supported_cell_details']+s['unsupported_cell_details']:
            cell_counts=[];cell_y=[]
            for state in [0,1]:
                key=tuple(cell[k]for k in keys)+(state,);entry=table.loc[key]
                prefix='missing'if state else'present';n=int(entry['size']);y=int(entry['sum'])
                assert n==cell['n_'+prefix]and y==cell['artifact_1_'+prefix]
                cell_counts.append(n);cell_y.append(y)
            supported=min(cell_counts)>=20;assert supported==cell['supported']
            if supported:
                count+=1;denom=[denom[i]+cell_counts[i]for i in [0,1]]
                rates.append((sum(cell_counts),cell_y[0]/cell_counts[0],cell_y[1]/cell_counts[1]))
        total=sum(denom);assert count==s['cells_supported']and total==s['covered_rows_total']
        assert denom==[s['covered_present_rows'],s['covered_missing_rows']]
        close(total/len(frame),s['coverage_all_rows'])
        r=np.array(rates);means=np.average(r[:,1:],weights=r[:,0],axis=0)
        close(100*(means[1]-means[0]),s['standardized_missing_minus_present_pp'])
    return len(schemes)

def existing_routing(frame,cfg):
    reference=read(ROOT/'reports/classification_round2/results.json');bundles={}
    for end in [55,59,63,67,71,75]:
        audit=read(ROOT/f'reports/classification_round1/fit_audits/fit{end}.json');bundles[end]={}
        for model in ['C2_B','C1_B']:
            with np.load(ROOT/f'experiments/runs/classification_round1_v1/fit{end}__{model}.npz',allow_pickle=False)as z:rows=z['outer_rows'];p=z['probability']
            assert array_digest(p)==audit[model]['probability_sha256']
            bundles[end][model]=(rows,p)
    out={};perwindow=[]
    for family,key in [('primary','primary_windows'),('secondary','secondary_windows')]:
        allrows=[];allhybrid=[];allroute=[]
        for w in cfg[key]:
            rows,p=bundles[w['fit'][1]]['C2_B'];r2,hgb=bundles[w['fit'][1]]['C1_B'];np.testing.assert_array_equal(rows,r2)
            mask=frame.iloc[rows].cycle_code.between(*w['validation']).to_numpy();rows=rows[mask];p=p[mask];hgb=hgb[mask]
            m=stream_missing_mask(frame.iloc[rows]);route=np.where(m,hgb,p);y=frame.iloc[rows].artifact_signal.to_numpy()
            perwindow.append({'family':family,'window':w['id'],'hybrid':f1_score(y,p>=.5,average='macro'),
                'routing':f1_score(y,route>=.5,average='macro')})
            allrows.append(rows);allhybrid.append(p);allroute.append(route)
        rows=np.concatenate(allrows);p=np.concatenate(allhybrid);r=np.concatenate(allroute);order=np.argsort(rows);rows,p,r=rows[order],p[order],r[order]
        assert len(rows)==len(np.unique(rows))==38588
        y=frame.iloc[rows].artifact_signal.to_numpy();m=stream_missing_mask(frame.iloc[rows]);values={}
        for name,pred,legacy in [('hybrid',p,'C2_B'),('routing',r,'C4_HGB')]:
            macro=f1_score(y,pred>=.5,average='macro');expected=reference['aggregate'][family]['fixed_0.5'][legacy]['metrics']['macro_f1'];close(macro,expected)
            values[name]={'macro_f1':macro,'missing_macro_f1':f1_score(y[m],pred[m]>=.5,average='macro'),
                'present_macro_f1':f1_score(y[~m],pred[~m]>=.5,average='macro'),
                'errors_present':int(np.sum((pred[~m]>=.5)!=y[~m])),'errors_missing':int(np.sum((pred[m]>=.5)!=y[m])),
                'probability_sha256':array_digest(pred)}
        np.testing.assert_array_equal(p[~m],r[~m]);out[family]={'n':len(rows),'missing_n':int(m.sum()),'models':values,
            'gain_already_in_final':values['routing']['macro_f1']-values['hybrid']['macro_f1']}
    out['windows']=perwindow;out['new_gain']=False;save(OUT/'existing_routing_impact.json',out);return out

def regression(frame,cfg,result):
    old=read(ROOT/'reports/regression_research/stage5_lgb_text.json');vectors={};checks=0
    for c in cfg['frozen_choices']:
        end=c['fit_end'];last=max(w['validation'][1]for w in cfg['primary_windows']+cfg['secondary_windows']if w['fit'][1]==end)
        fit=np.flatnonzero(frame.cycle_code.between(47,end));expected=np.flatnonzero(frame.cycle_code.between(end+1,last))
        path=ROOT/f'experiments/runs/impact_eda_oov_v1/fit{end}__coverage.npz';m=read(path.with_suffix('.json'))
        with np.load(path,allow_pickle=False)as z:rows=z['rows'];p=z['p'];oov=z['coverage'];np.testing.assert_array_equal(z['fit'],fit)
        np.testing.assert_array_equal(rows,expected);assert array_digest(p)==m['prediction_sha256']and array_digest(oov)==m['coverage_sha256']
        assert m['state']['state_sha256']==_json_digest({k:v for k,v in m['state'].items()if k!='state_sha256'})
        prior=old['audits'][str(end)]['outer'];state=m['state'];fields=state['input_feature_names']
        assert state['fit_data_sha256']==_fit_fingerprint(frame.iloc[fit][fields],tuple(fields),frame.iloc[fit].threat_rank.to_numpy())
        for block in ['structured_preprocessing','tfidf','svd','actual_hyperparameters','fit_data_sha256','target_sha256']:assert state[block]==prior['state'][block]
        source=ROOT/f"experiments/runs/regression_research_v1/lgbtext__{c['param']}__fit{end}__to{last}.npz"
        if source.exists():
            with np.load(source,allow_pickle=False)as z:base=z['prediction']
        else:
            with np.load(ROOT/f'experiments/runs/impact_eda_oov_v1/fit{end}__control.npz',allow_pickle=False)as z:base=z['p']
        assert array_digest(base)==prior['prediction_sha256']
        vectors[end]={'rows':rows,'control':mapping(base,frame.iloc[rows].cycle_code.to_numpy(),c['rank_weight']),
            'coverage':mapping(p,frame.iloc[rows].cycle_code.to_numpy(),c['rank_weight'])}
    chunks={}
    for job in result['jobs']:
        w=next(w for w in cfg['primary_windows']+cfg['secondary_windows']if w['id']==job['window'])
        v=vectors[w['fit'][1]];mask=frame.iloc[v['rows']].cycle_code.between(*w['validation']).to_numpy();rows=v['rows'][mask];p=v[job['model']][mask]
        close_metrics(frame.iloc[rows],p,job);assert array_digest(p)==job['prediction_sha256'];checks+=1
        if job['model']=='control':
            legacy=next(j for j in old['jobs']if j['family']==job['family']and j['window']==job['window']and j['policy']=='inner')
            assert legacy['prediction_sha256']==array_digest(p)
        chunks.setdefault((job['family'],job['model']),[]).append((rows,p))
    for (family,model),parts in chunks.items():
        rows=np.concatenate([r for r,p in parts]);p=np.concatenate([p for r,p in parts]);idx=np.argsort(rows);rows,p=rows[idx],p[idx]
        assert len(rows)==len(np.unique(rows))==38588;close_metrics(frame.iloc[rows],p,result['aggregate'][family][model]);checks+=1
    return checks

def close_metrics(frame,p,result):
    y=frame.threat_rank.to_numpy();error=p-y;close(mean_absolute_error(y,p),result['raw']['mae'])
    close(error.mean(),result['raw']['signed_error']);close(np.sqrt(np.mean(error**2)),result['raw']['rmse'])
    missing=stream_missing_mask(frame)
    masks={'stream_missing':missing,'stream_present':~missing,'EF0':frame.entry_friction.eq(0).to_numpy(),
        'EF1':frame.entry_friction.eq(1).to_numpy(),'tail_low_le5':y<=5,'tail_high_ge95':y>=95,
        'region_low':y<100/3,'region_middle':(y>=100/3)&(y<200/3),'region_high':y>=200/3}
    for name,m in masks.items():
        assert result['subgroups'][name]['n']==int(m.sum());close(np.abs(error[m]).mean(),result['subgroups'][name]['mae'])
        close(error[m].mean(),result['subgroups'][name]['signed_error'])
    cyc=frame.cycle_code.to_numpy();close(np.mean([np.abs(error[cyc==c]).mean()for c in np.unique(cyc)]),result['equal_cycle_mae'])

def gate(r,g):
    a=r['aggregate'];gain={f:a[f]['control']['raw']['mae']-a[f]['coverage']['raw']['mae']for f in a};window=[]
    for j in r['jobs']:
        if j['model']=='coverage':
            b=next(v for v in r['jobs']if v['model']=='control'and v['window']==j['window']);window.append((j['family'],b['raw']['mae']-j['raw']['mae']))
    values=[gain['primary']>=g['min_primary_gain'],gain['secondary']>=g['min_H12_gain'],
        a['primary']['control']['equal_cycle_mae']-a['primary']['coverage']['equal_cycle_mae']>=g['min_equal_cycle_gain'],
        sum(v>0 for f,v in window if f=='primary')>=g['positive_primary_windows'],min(v for f,v in window)>=-g['max_window_loss'],
        max(a[f]['coverage']['subgroups'][s]['mae']-a[f]['control']['subgroups'][s]['mae']for f in a for s in ['stream_missing','stream_present','EF0','EF1','tail_low_le5','tail_high_ge95'])<=g['max_subgroup_loss']]
    assert gain==r['gate']['gains']and values==list(r['gate']['checks'].values())and r['gate']['passes']==all(values)

def oov_eda(frame,cfg,stats):
    assert stats['source']['train_sha256']==cfg['regression_train_sha256']
    analyzer=TfidfVectorizer(**_TFIDF_PARAMS).build_analyzer();parts=[]
    for c in cfg['frozen_choices']:
        end=c['fit_end'];path=ROOT/f'experiments/runs/impact_eda_oov_v1/fit{end}__coverage.npz'
        with np.load(path,allow_pickle=False)as z:rows=z['rows'];fraction=z['coverage']
        audit=stats['vocabulary_reconciliation'][str(end)]
        assert array_digest(fraction)==audit['candidate_helper_oov_fraction_sha256_rows_in_outer_order']
        v=frame.iloc[rows].copy();last=v.cycle_code.max()
        source=ROOT/f"experiments/runs/regression_research_v1/lgbtext__{c['param']}__fit{end}__to{last}.npz"
        if source.exists():
            with np.load(source,allow_pickle=False)as z:raw=z['prediction']
        else:
            with np.load(ROOT/f'experiments/runs/impact_eda_oov_v1/fit{end}__control.npz',allow_pickle=False)as z:raw=z['p']
        pred=mapping(raw,v.cycle_code.to_numpy(),c['rank_weight'])
        docs,missing=_text_documents(v.signal_stream)
        v['_terms']=[len(analyzer(d))for d in docs];v['_oov']=fraction;v['_ae']=np.abs(pred-v.threat_rank.to_numpy());v['_missing']=missing
        w=next(w for w in cfg['primary_windows']if w['fit'][1]==end)
        parts.append(v.loc[v.cycle_code.between(*w['validation'])])
    data=pd.concat(parts,ignore_index=True);assert len(data)==data.record_id.nunique()==38588
    data=data.loc[~data._missing & data._terms.gt(0)].copy()
    actual=[]
    for cycle,g in data.groupby('cycle_code',sort=True):
        tie=np.array([int.from_bytes(hashlib.sha256(str(x).encode()).digest()[:8],'little')for x in g.record_id],dtype=np.uint64)
        order=np.lexsort((tie,g._terms.to_numpy()));labels=np.empty(len(g),int);labels[order]=(np.arange(len(g))*4)//len(g)
        g=g.copy();g['_q']=labels
        for q,cell in g.groupby('_q',sort=True):
            ranks=rankdata(cell._oov,method='average')/len(cell);low=ranks<=.25;high=ranks>.75
            reported=next(s for s in stats['matched_within_cycle_length_strata']if s['cycle']==cycle and s['length_quartile']==q)
            assert len(cell)==reported['stratum_rows']and int(low.sum())==reported['low_oov_rows']and int(high.sum())==reported['high_oov_rows']
            supported=min(low.sum(),high.sum())>=20;assert supported==reported['meets_minimum_support']
            if supported:
                y=cell.threat_rank.to_numpy();ae=cell._ae.to_numpy();pct=100*rankdata(ae,method='average')/len(ae)
                values={'high_minus_low_target_iqr':np.subtract(*np.quantile(y[high],[.75,.25]))-np.subtract(*np.quantile(y[low],[.75,.25])),
                    'high_minus_low_target_median':np.median(y[high])-np.median(y[low]),
                    'high_minus_low_abs_error_mae':ae[high].mean()-ae[low].mean(),
                    'high_minus_low_within_stratum_abs_error_percentile':pct[high].mean()-pct[low].mean()}
                for key,value in values.items():close(value,reported[key])
                actual.append(reported)
    for p in stats['matched_period_summary']:
        cells=[s for s in actual if s['period']==p['period']]
        assert len(cells)==p['supported_strata']and sum(s['stratum_rows']for s in cells)==p['rows_in_supported_strata']
        assert sum(s['low_oov_rows']for s in cells)==p['low_oov_group_rows']and sum(s['high_oov_rows']for s in cells)==p['high_oov_group_rows']
        for source,target in [('high_minus_low_target_iqr','mean_high_minus_low_target_iqr'),('high_minus_low_target_median','mean_high_minus_low_target_median'),
            ('high_minus_low_abs_error_mae','mean_high_minus_low_abs_error_mae'),('high_minus_low_within_stratum_abs_error_percentile','mean_high_minus_low_abs_error_percentile')]:
            close(np.average([s[source]for s in cells],weights=[s['stratum_rows']for s in cells]),p[target])
            assert sum(s[source]>0 for s in cells)==p[target+'_positive_strata']
    return len(actual)

def main():
    cfg=read(ROOT/'experiments/impact_eda/protocol.json');freeze=read(ROOT/'experiments/impact_eda/freeze.json')
    assert freeze['signature']==digest_json({k:v for k,v in freeze.items()if k!='signature'})
    assert code_sha(ROOT/'experiments/impact_eda/protocol.json')==freeze['protocol_sha256']
    for p,h in freeze['code_sha256'].items():assert code_sha(ROOT/p)==h,p
    for p,h in cfg['protected_files'].items():assert sha(ROOT/p)==h,p
    for p,h in cfg['reference_artifacts'].items():assert sha(ROOT/p)==h,p
    cf,rf=[task_files(t)['train']for t in ['classification','regression']]
    assert sha(cf)==cfg['classification_train_sha256']and sha(rf)==cfg['regression_train_sha256']
    classification=pd.read_csv(cf);reg=pd.read_csv(rf)
    availability_stats=read(OUT/'availability/stats.json');assert availability_stats['source']['sha256']==sha(cf)
    n=availability(classification,availability_stats);existing_routing(classification,cfg)
    r=read(OUT/'ablation/results.json');assert r['freeze_signature']==freeze['signature']and r['input_sha256']==sha(rf)
    blocks=regression(reg,cfg,r);gate(r,cfg['gate'])
    oov_stats=read(OUT/'oov/stats.json');oov_cells=oov_eda(reg,cfg,oov_stats)
    assert r['official_test_read']is False and r['kaggle_submissions']==0 and r['hyperparameter_or_weight_selection']is False
    receipt={'status':'PASS','source_train_hashes_verified':True,'protected_historical_files_byte_identical':len(cfg['protected_files']),
        'availability_standardizations_recomputed':n,'existing_classification_routing_reconciled':True,'regression_metric_blocks_recomputed':blocks,
        'regression_coverage_fits_audited':6,'regression_gate_independently_recomputed':True,'official_test_read':False,'new_submissions':0,
        'oov_feature_vectors_reconciled':6,'oov_matched_cells_independently_recomputed':oov_cells,
        'official_notebook_or_model_changes':False,'freeze_signature':freeze['signature'],
        'report_hashes':{str(p.relative_to(ROOT)).replace('\\','/'):sha(p)for p in [OUT/'availability/stats.json',OUT/'oov/stats.json',OUT/'ablation/results.json',OUT/'existing_routing_impact.json']},
        'verifier_code_sha256':code_sha(Path(__file__))}
    save(OUT/'QA.json',receipt);print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
