"""Bounded cached-prediction simplification audit; never fits or reads Test."""
from __future__ import annotations
import argparse, csv, hashlib, json, os, subprocess, tempfile
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import psutil
from scipy.stats import rankdata
from sklearn.metrics import f1_score

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/simplification_v1'
FREEZE=OUT/'INPUT_FREEZE.json'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f: json.dump(d,f,indent=2,allow_nan=False); f.write('\n')
def close(a,b):
    if not np.allclose(a,b,rtol=0,atol=1e-10): raise ValueError(f'Replay mismatch {np.max(np.abs(a-b))}')
def align(ids,source_ids,*vectors):
    if len(set(source_ids))!=len(source_ids): raise ValueError('Duplicate source IDs')
    index={str(x):i for i,x in enumerate(source_ids)}
    ix=np.array([index[str(x)] for x in ids])
    if not np.array_equal(np.asarray(source_ids)[ix].astype(str),np.asarray(ids).astype(str)): raise ValueError('ID mismatch')
    return [np.asarray(x)[ix] for x in vectors]
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def mapped(x,c,w=.5,clip=True):
    x=np.asarray(x,dtype=float); r=np.empty_like(x)
    for cycle in np.unique(c):
        m=c==cycle;r[m]=100*rankdata(x[m],method='average')/m.sum()
    out=(1-w)*x+w*r
    return np.clip(out,0,100) if clip else out
def hinge(x,params):
    return np.clip(params['intercept']+params['slope_low']*np.minimum(x,50)+params['slope_high']*np.maximum(x-50,0),0,100)

def inventory(cls,reg,front):
    paths={'protocol':OUT/'PROTOCOL.md','script':Path(__file__),
      'cls_a02':cls/'experiments/runs/optimization_v3_posoffset_a02_endgame_v1/policy_predictions.npz',
      'cls_pos':cls/'experiments/runs/optimization_v3_cls_pos_one_sided_ridge_v1/predictions.npz',
      'reg_labels67':reg/'experiments/runs/reg_monotone_hinge_p50_v1/R4/fit_inputs/HINGE_D2.npz',
      'reg_labels75':reg/'experiments/runs/reg_monotone_hinge_p50_v1/R4/score_rows/HINGE_D2.npz'}
    for o in [67,75]:
        paths[f'cls_c16_{o}']=cls/('experiments/runs/optimization_v3_batch1/C16' if o==67 else 'experiments/runs/optimization_v3_final_continuation/C16')/f'C16_fit{o}.npz'
        paths[f'reg_r06_{o}']=cls/('experiments/runs/optimization_v3_batch1/R06' if o==67 else 'experiments/runs/optimization_v3_expanded_benchmark_v1_recovery_v2/R06')/f'R06_fit{o}.npz'
        paths[f'reg_direct_{o}']=front/f'experiments/runs/reg_position_direct_l1_v1/predictions/REG_POSITION_DIRECT_L1_V1_E{o}_REAL.npz'
        for key in [f'cls_c16_{o}',f'reg_r06_{o}',f'reg_direct_{o}']: paths[key+'_receipt']=paths[key].with_suffix('.json')
    paths['reg_components67']=cls/'experiments/runs/optimization_v3_batch1/R21/R21_fit67.npz'
    paths['reg_components67_receipt']=paths['reg_components67'].with_suffix('.json')
    for stage in ['D1','D2']:
        paths[f'hinge_{stage}']=reg/f'experiments/runs/reg_monotone_hinge_p50_v1/R4/fits/HINGE_{stage}.json'
        paths[f'global_{stage}']=reg/f'experiments/runs/REG_FINAL_WAVE_FOLLOWUP_V2/fits/{stage}_GLOBAL.json'
    return paths

@contextmanager
def resource_lock():
    p=Path(tempfile.gettempdir())/'kasdad_single_heavy_job.lock'
    with p.open('a+b') as f:
        if p.stat().st_size==0:f.write(b'\0');f.flush()
        f.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            available=psutil.virtual_memory().available
            if available<1.5*1024**3 or available-256*1024**2<.75*1024**3:raise RuntimeError('RAM gate')
            yield {'initial_available_bytes':available,'estimated_incremental_bytes':256*1024**2}
            if psutil.virtual_memory().available<.75*1024**3:raise RuntimeError('RAM boundary')
        finally:
            f.seek(0)
            if os.name=='nt':msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(f,fcntl.LOCK_UN)

def cm(y,p,t):
    label=p>=t;tp=int(np.sum(label&(y==1)));tn=int(np.sum(~label&(y==0)));fp=int(np.sum(label&(y==0)));fn=int(np.sum(~label&(y==1)))
    div=lambda a,b:a/b if b else 0.
    f=.5*(div(2*tp,2*tp+fp+fn)+div(2*tn,2*tn+fp+fn))
    close(f,f1_score(y,label,average='macro',labels=[0,1],zero_division=0))
    return {'macro_f1':f,'recall':div(tp,tp+fn),'precision':div(tp,tp+fp),'confusion':[[tn,fp],[fn,tp]]}

def classification(paths):
    a=load(paths['cls_a02']);z=load(paths['cls_pos']); allv={}
    names=['REFERENCE','NO_MEMORY','NO_POS','NO_MEMORY_NO_POS','LR_CB','POS_CB','HGB_MISSING']
    for o in [67,75]:
        k=f'E{o}_';ids=a[k+'ids'].astype(str);missing=a[k+'missing'].astype(bool);cycles=a[k+'cycles'];y=a[k+'y']
        close(a[k+'pP'],z[k+'pPOS_UNIFORM']); close(a[k+'pA'],z[k+'pA'])
        assert np.array_equal(ids,z[k+'ids'].astype(str))
        cache=load(paths[f'cls_c16_{o}'])
        lr,cb,sparse,base=align(ids,cache['score_record_ids'].astype(str),cache['control__c2b_probability'],cache['control__cb_probability'],cache['control__baseline_sparse_raw'],cache['baseline_prediction'])
        close(base,a[k+'b']); close(base,np.where(missing,cb,.75*lr+.25*sparse));close(cb[missing],a[k+'pA'][missing])
        pp,pa,b=a[k+'pP'],a[k+'pA'],a[k+'b']
        present={'REFERENCE':.5*pp+.5*pa,'NO_MEMORY':.5*pp+.5*b,'NO_POS':pa,'NO_MEMORY_NO_POS':b,'LR_CB':lr,'POS_CB':pp,'HGB_MISSING':.5*pp+.5*pa}
        probabilities={n:np.where(missing,z[k+'pMISSING_HGB'] if n=='HGB_MISSING' else cb,present[n]) for n in names}
        allv[o]={'ids':ids,'cycles':cycles,'y':y,'missing':missing,'p':probabilities,'groups':{g:a[k+'group_'+g].astype(str) for g in ['availability','entry_friction','text_load']}}
    views={}
    selections=[('E67_H12',[(67,68,79)]),('E67_W4',[(67,68,71)]),('E67_W5',[(67,72,75)]),('E67_W6',[(67,76,79)]),('E75_W6',[(75,76,79)]),('UNIQUE',[(67,68,75),(75,76,79)])]
    for name,sels in selections:
        pieces=[(allv[o],(allv[o]['cycles']>=lo)&(allv[o]['cycles']<=hi)) for o,lo,hi in sels]
        cat=lambda key:np.concatenate([d[key][m] for d,m in pieces])
        y=cat('y');miss=cat('missing');cy=cat('cycles');ids=cat('ids');assert len(set(ids))==len(ids)
        results={}
        for threshold in [.5,.53230976129477]:
            t=np.where(miss,threshold,.5);prob={n:np.concatenate([d['p'][n][m] for d,m in pieces]) for n in names};ref=cm(y,prob['REFERENCE'],t)
            summary={}
            for n,p in prob.items():
                met=cm(y,p,t);group_harm=[];unknown=[]
                for family in ['availability','entry_friction','text_load']:
                    groups=np.concatenate([d['groups'][family][m] for d,m in pieces])
                    for g in np.unique(groups):
                        m=groups==g
                        if m.sum()<200 or np.sum(y[m]==0)<30 or np.sum(y[m]==1)<30:unknown.append(f'{family}:{g}');continue
                        bmet=cm(y[m],prob['REFERENCE'][m],t[m]);gmet=cm(y[m],p[m],t[m]);group_harm.append({'group':f'{family}:{g}','n':int(m.sum()),'f1_harm':bmet['macro_f1']-gmet['macro_f1'],'recall_harm':bmet['recall']-gmet['recall']})
                summary[n]={**met,'gain':met['macro_f1']-ref['macro_f1'],'flips':int(np.sum((p>=t)!=(prob['REFERENCE']>=t))),'present_f1':cm(y[~miss],p[~miss],t[~miss])['macro_f1'],'groups':group_harm,'unsupported':unknown,'cycles':{str(c):cm(y[cy==c],p[cy==c],t[cy==c])['macro_f1']-cm(y[cy==c],prob['REFERENCE'][cy==c],t[cy==c])['macro_f1'] for c in np.unique(cy)}}
            results[str(threshold)]=summary
        views[name]={'n':len(y),'present':int((~miss).sum()),'policies':results}
    return views

def regression(paths):
    labels=[load(paths['reg_labels67']),load(paths['reg_labels75'])];views={};replay={}
    for o,stage,lo,hi in [(67,'D1',72,75),(75,'D2',76,79)]:
        direct=load(paths[f'reg_direct_{o}']);ids=direct['query_record_ids'].astype(str);c=direct['query_cycles'];missing=direct['missing_mask'].astype(bool)
        char=load(paths[f'reg_r06_{o}']);comp=load(paths['reg_components67']) if o==67 else char
        lgb,ridge,word=align(ids,comp['score_record_ids'].astype(str),comp['control__baseline_lgb_svd32'],comp['control__baseline_hybrid_ridge'],comp['control__baseline_sparse_raw'])
        rawchar,legacy,r06=align(ids,char['score_record_ids'].astype(str),char['raw_candidate'],char['baseline_prediction'],char['prediction'])
        parts=np.stack([lgb,ridge,mapped(word,c),mapped(rawchar,c),mapped(direct['raw_auxiliary'],c,clip=False)],axis=1)
        legacy_rebuilt=np.clip(.5625*lgb+.1875*ridge+.25*parts[:,2],0,100);close(legacy,legacy_rebuilt)
        r06_rebuilt=np.clip(.75*legacy+.25*parts[:,3],0,100);close(r06,r06_rebuilt);close(direct['baseline'],r06)
        nested=np.clip(.75*r06+.25*parts[:,4],0,100);nested[missing]=r06[missing];close(nested,direct['prediction'])
        weights=np.tile([.31640625,.10546875,.140625,.1875,.25],(len(ids),1));weights[missing]=[.421875,.140625,.1875,.25,0]
        flat=np.clip(np.sum(parts*weights,axis=1),0,100);close(flat,nested)
        replay[stage]={'nested_flat_maxdiff':float(np.max(np.abs(flat-nested))),'max_leaf':float(parts.max()),'min_leaf':float(parts.min())}
        params=json.load(open(paths['hinge_'+stage]))['solver_result'];gp=json.load(open(paths['global_'+stage]))['solver_result']
        rawpol={'REFERENCE':nested}
        for j,n in enumerate(['NO_LGB_SVD','NO_RIDGE','NO_WORD','NO_CHAR','NO_POS']):
            w=weights.copy();w[:,j]=0;w/=w.sum(axis=1)[:,None];rawpol[n]=np.clip(np.sum(parts*w,axis=1),0,100)
        pol={n:np.where(missing,p,hinge(p,params)) for n,p in rawpol.items()}
        pol['NO_CALIBRATION']=nested.copy();pol['GLOBAL_INSTEAD_HINGE']=np.where(missing,nested,np.clip(gp['intercept']+gp['slope']*nested,0,100))
        lab=labels[0] if o==67 else labels[1];y,rows=align(ids[(c>=lo)&(c<=hi)],lab['record_ids'].astype(str),lab['targets'],lab['physical_rows'])
        m=(c>=lo)&(c<=hi);cy=c[m];mi=missing[m];base=pol['REFERENCE'][m];rawbase=nested[m];summary={}
        for n,p in pol.items():
            p=p[m];err=np.abs(y-p);gain=float(np.mean(np.abs(y-base))-err.mean());close(err.mean(),sum(abs(float(t)-float(v)) for t,v in zip(y,p))/len(y))
            groups={}
            for gn,gm in {'present':~mi,'missing':mi,'target_le5':y<=5,'target_ge95':y>=95}.items():
                groups[gn]={'n':int(gm.sum()),'gain':float(np.mean(np.abs(y[gm]-base[gm]))-err[gm].mean()) if gm.sum()>=200 else None}
            summary[n]={'mae':float(err.mean()),'gain':gain,'p90':float(np.quantile(err,.9)),'mean_error':float(np.mean(p-y)),'groups':groups,'cycles':{str(v):float(np.mean(np.abs(y[cy==v]-base[cy==v]))-err[cy==v].mean()) for v in np.unique(cy)},'raw_space_gain':float(np.mean(np.abs(y-rawbase))-np.mean(np.abs(y-rawpol[n][m]))) if n in rawpol else None,'changed_missing_rows':int(np.sum(np.abs(p[mi]-base[mi])>1e-10))}
        views[stage]={'n':len(y),'ids_sha':hashlib.sha256(json.dumps(ids[m].tolist()).encode()).hexdigest(),'candidates':summary}
    return views,replay

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--freeze',action='store_true');ap.add_argument('--run',action='store_true');ap.add_argument('--classification-root',type=Path,default=ROOT.parent/'TP 2');ap.add_argument('--regression-root',type=Path,default=ROOT.parent/'tk2-reg-position-failure-mechanism-v1');ap.add_argument('--frontier-root',type=Path,default=ROOT.parent/'tk2-regression-frontier');args=ap.parse_args()
    if args.freeze:
        paths=inventory(args.classification_root,args.regression_root,args.frontier_root)
        write(FREEZE,{'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'files':{k:{'path':str(v.resolve()),'sha256':sha(v)} for k,v in paths.items()},'new_fits':0,'new_raw_data_access':False});return
    f=json.load(open(FREEZE));paths={k:Path(v['path']) for k,v in f['files'].items()}
    for k,p in paths.items():
        if sha(p)!=f['files'][k]['sha256']:raise ValueError('Frozen bytes changed '+k)
    for k,p in paths.items():
        if k.endswith('_receipt'):
            d=json.load(open(p));npz=paths[k[:-8]]
            expected=d.get('npz_sha256',d.get('bundle_sha256'))
            if expected and sha(npz)!=expected:raise ValueError('Producer hash mismatch '+k)
    with resource_lock() as resources:
        cls=classification(paths);reg,replay=regression(paths)
        resources['final_available_bytes']=psutil.virtual_memory().available
        write(OUT/'RESULTS.json',{'classification':cls,'regression':reg,'regression_replay':replay,'resource':resources,'freeze_sha':sha(FREEZE),'new_fits':0,'Test_access':False,'new_raw_labels':False})

if __name__=='__main__':main()
