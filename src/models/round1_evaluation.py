"""Frozen expanding-window development experiments; never predicts official test."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import subprocess
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.data.audit import sha256, write_json
from src.models.round1_models import build_model
from src.utils.paths import PROJECT_ROOT, task_files

PROTOCOL = PROJECT_ROOT/'experiments/round1/protocol.json'
FREEZE = PROJECT_ROOT/'experiments/round1/freeze.json'
CORE_FILES = ['src/models/round1_models.py', 'src/models/round1_evaluation.py', 'scripts/run_round1.py',
              'src/data/audit.py', 'src/utils/paths.py']


def digest_json(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=True,allow_nan=False).encode()).hexdigest()


def array_digest(value):
    return hashlib.sha256(np.asarray(value,dtype='<f8').tobytes()).hexdigest()


def code_digest(path):
    # Stable across LF/CRLF checkouts; raw CSV identities remain byte-exact.
    return hashlib.sha256(Path(path).read_text(encoding='utf-8-sig').replace('\r\n','\n').encode('utf-8')).hexdigest()


def load_protocol():
    return json.loads(PROTOCOL.read_text(encoding='utf-8'))


def identity(data_dir=None):
    files = task_files('regression',data_dir)
    if not all(files.values()):
        raise FileNotFoundError('Official regression Train/Test/Sample files are required for identity checks.')
    return {'protocol_sha256':digest_json(load_protocol()),
            'code_sha256':{p:code_digest(PROJECT_ROOT/p) for p in CORE_FILES},
            'hash_convention':'Protocol canonical JSON; training code UTF-8 with LF; raw CSV exact bytes.',
            'input_sha256':{k:sha256(v) for k,v in files.items()},
            'python':platform.python_version(),
            'packages':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','threadpoolctl']}}


def freeze_protocol(data_dir=None):
    signature = identity(data_dir)
    if FREEZE.exists():
        old = json.loads(FREEZE.read_text(encoding='utf-8'))
        if old['identity'] != signature:
            raise RuntimeError('Existing freeze differs. Document an engineering amendment/new protocol before training; do not overwrite.')
        return old
    head = subprocess.check_output(['git','-C',str(PROJECT_ROOT),'rev-parse','HEAD'],text=True).strip()
    receipt = {'status':'FROZEN_BEFORE_FIRST_ROUND1_EXPERIMENT_FIT',
               'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
               'base_git_head':head,'identity':signature,'signature':digest_json(signature),
               'source_scope':'Official information confirmed by user; eligibility is timing-provisional for additions.'}
    write_json(FREEZE,receipt)
    return receipt


def verify_freeze(data_dir=None):
    if not FREEZE.exists():
        raise RuntimeError('Run --freeze-only before any training.')
    receipt = json.loads(FREEZE.read_text(encoding='utf-8'))
    if receipt['identity'] != identity(data_dir) or receipt['signature'] != digest_json(receipt['identity']):
        raise RuntimeError('Frozen protocol, training code, inputs or runtime changed; refusing training/cache reuse.')
    return receipt


def metric_values(y,pred,tolerance=1e-10):
    y,pred=np.asarray(y,dtype=float),np.asarray(pred,dtype=float)
    if len(y)!=len(pred) or y.ndim!=1 or pred.ndim!=1 or not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError('Metrics require aligned, finite one-dimensional actual/predicted values.')
    n=len(y)
    if not n:
        return {'n':0,'mae':None,'rmse':None,'signed_error':None,'over_rate':None,'under_rate':None,'equal_rate':None}
    error=pred-y
    return {'n':n,'mae':float(np.abs(error).mean()),'rmse':float(np.sqrt(np.mean(error**2))),
            'signed_error':float(error.mean()),'over_rate':float((error>tolerance).mean()),
            'under_rate':float((error < -tolerance).mean()),'equal_rate':float((np.abs(error)<=tolerance).mean()),
            'absolute_error_p95':float(np.quantile(np.abs(error),.95)),
            'prediction_quantiles':{str(q):float(np.quantile(pred,q)) for q in [0,.01,.05,.25,.5,.75,.95,.99,1]},
            'target_quantiles':{str(q):float(np.quantile(y,q)) for q in [0,.01,.05,.25,.5,.75,.95,.99,1]},
            'below_zero':int((pred<0).sum()),'above_100':int((pred>100).sum())}


def subgroup_values(frame,pred,protocol):
    y=frame.threat_rank.to_numpy()
    masks={'EF0':frame.entry_friction.eq(0).to_numpy(),'EF1':frame.entry_friction.eq(1).to_numpy(),
           'region_low':y<100/3,'region_middle':(y>=100/3)&(y<200/3),'region_high':y>=200/3,
           'tail_low_le5':y<=5,'tail_high_ge95':y>=95,
           'stream_missing':frame.signal_stream.isna().to_numpy(),'stream_present':frame.signal_stream.notna().to_numpy()}
    return {name:metric_values(y[mask],pred[mask],protocol['prediction_equality_tolerance']) for name,mask in masks.items()}


def window_indices(frame,window):
    fit=np.flatnonzero(frame.cycle_code.between(*window['fit']).to_numpy())
    validation=np.flatnonzero(frame.cycle_code.between(*window['validation']).to_numpy())
    if not len(fit) or not len(validation) or set(frame.iloc[fit].cycle_code)&set(frame.iloc[validation].cycle_code):
        raise ValueError('Empty or overlapping cycle groups.')
    if frame.iloc[fit].cycle_code.max()>=frame.iloc[validation].cycle_code.min():
        raise ValueError('Validation must be strictly newer than fit.')
    return fit,validation


def _fitted_state(model):
    state=model.audit_state()
    state.pop('prediction_diagnostics',None)
    return state


def run_round1(data_dir=None,run_id='round1_v1',report_dir=None,force=False):
    if not re.fullmatch(r'[A-Za-z0-9_-]+',run_id):
        raise ValueError('Run ID must be a simple label.')
    frozen=verify_freeze(data_dir)
    protocol=load_protocol()
    artifact=PROJECT_ROOT/'experiments/runs'/run_id
    artifact.mkdir(parents=True,exist_ok=True)
    reports=Path(report_dir) if report_dir else PROJECT_ROOT/'reports/round1'
    frame=pd.read_csv(task_files('regression',data_dir)['train'])
    if frame.record_id.isna().any() or not frame.record_id.is_unique or frame.threat_rank.isna().any():
        raise ValueError('Input identity/target invalid.')
    if not np.isfinite(frame.threat_rank).all() or not frame.threat_rank.between(0,100).all():
        raise ValueError('Target outside official finite range.')
    all_jobs=[]; audits={}; predictions={}; fresh=0
    windows=[('primary',w) for w in protocol['primary_windows']]+[('secondary',w) for w in protocol['secondary_windows']]
    start_all=perf_counter()
    for family,window in windows:
        fit,val=window_indices(frame,window)
        audits[window['id']]={}
        for declared in protocol['recipes']:
            recipe=dict(declared)
            if recipe['kind']=='hgb':
                recipe['params']={**protocol['hgb_fixed_parameters'],**recipe['params']}
            model_id,window_id=recipe['id'],window['id']
            job_id=f'{window_id}__{model_id}'
            job_key=digest_json({'freeze':frozen['signature'],'recipe':recipe,'window':window,
                                 'fit_rows':array_digest(fit),'validation_rows':array_digest(val)})
            cache_meta,cache_pred=artifact/f'{job_id}.json',artifact/f'{job_id}.npz'
            cached=False
            if not force and cache_meta.exists() and cache_pred.exists():
                meta=json.loads(cache_meta.read_text(encoding='utf-8'))
                with np.load(cache_pred,allow_pickle=False) as saved:
                    pred=saved['prediction'].copy()
                    saved_rows=saved['validation_rows'].copy()
                if meta['job_key']!=job_key or not np.array_equal(saved_rows,val) or array_digest(pred)!=meta['prediction_sha256']:
                    raise RuntimeError(f'Cache identity mismatch: {job_id}')
                cached=True
            else:
                Xfit=frame.iloc[fit][protocol['feature_sets'][recipe['features']]].copy()
                Xval=frame.iloc[val][protocol['feature_sets'][recipe['features']]].copy()
                yfit=frame.iloc[fit].threat_rank.to_numpy(copy=True)
                model=build_model(recipe)
                fit_started_at=datetime.now(timezone.utc).isoformat()
                start=perf_counter()
                with threadpool_limits(limits=protocol['thread_limit']):
                    model.fit(Xfit,yfit)
                    state=_fitted_state(model)
                    state_sha=digest_json(state)
                    pred=np.asarray(model.predict(Xval),dtype=float)
                if state_sha!=digest_json(_fitted_state(model)):
                    raise RuntimeError(f'Prediction mutated fitted state: {job_id}')
                if pred.shape!=(len(val),) or not np.isfinite(pred).all():
                    raise RuntimeError(f'Invalid prediction: {job_id}')
                meta={'job_key':job_key,'fit_seconds':perf_counter()-start,'fit_started_at_utc':fit_started_at,
                      'fit_rows':len(fit),'validation_rows':len(val),
                      'fit_cycle_range':window['fit'],'validation_cycle_range':window['validation'],
                      'fitted_state_sha256':state_sha,'fitted_state':state,'prediction_sha256':array_digest(pred),
                      'prediction_diagnostics':model.prediction_diagnostics(),
                      'validation_labels_passed_to_fit':False,'early_stopping_validation_used':False}
                np.savez_compressed(cache_pred,validation_rows=val,prediction=pred)
                write_json(cache_meta,meta)
                fresh+=1
            y=frame.iloc[val].threat_rank.to_numpy()
            metrics=metric_values(y,pred,protocol['prediction_equality_tolerance'])
            cycles=[]
            for cycle in sorted(frame.iloc[val].cycle_code.unique()):
                mask=frame.iloc[val].cycle_code.eq(cycle).to_numpy()
                cycles.append({'cycle':int(cycle),**metric_values(y[mask],pred[mask],protocol['prediction_equality_tolerance'])})
            all_jobs.append({'model':model_id,'window':window_id,'family':family,'metrics':metrics,
                'clipped_metrics':metric_values(y,np.clip(pred,0,100),protocol['prediction_equality_tolerance']),
                'equal_cycle_mae':float(np.mean([c['mae'] for c in cycles])),
                'subgroups':subgroup_values(frame.iloc[val],pred,protocol),'per_cycle':cycles,
                'fit_seconds':meta['fit_seconds'],'fit_rows':len(fit),'validation_rows':len(val),
                'prediction_sha256':meta['prediction_sha256'],'fitted_state_sha256':meta['fitted_state_sha256']})
            audits[window_id][model_id]=meta
            predictions[(window_id,model_id)]=(val,pred)
            print(f'{job_id}: MAE={metrics["mae"]:.6f} | {"cache" if cached else "fit"}',flush=True)
    aggregate={}
    for family in ('primary','secondary'):
        aggregate[family]={}
        for recipe in protocol['recipes']:
            jobs=[j for j in all_jobs if j['family']==family and j['model']==recipe['id']]
            row_indices=np.concatenate([predictions[(j['window'],j['model'])][0] for j in jobs])
            pred=np.concatenate([predictions[(j['window'],j['model'])][1] for j in jobs])
            if len(np.unique(row_indices))!=len(row_indices):
                raise RuntimeError('Duplicate validation observations within family aggregate.')
            part=frame.iloc[row_indices]
            group_metrics={str(float(x)):metric_values(part.loc[part.impact_load.eq(x),'threat_rank'],pred[part.impact_load.eq(x).to_numpy()])
                           for x in sorted(part.impact_load.unique())}
            aggregate[family][recipe['id']]={'raw':metric_values(part.threat_rank,pred,protocol['prediction_equality_tolerance']),
                 'clipped':metric_values(part.threat_rank,np.clip(pred,0,100),protocol['prediction_equality_tolerance']),
                 'equal_cycle_mae':float(np.mean([c['mae'] for j in jobs for c in j['per_cycle']])),
                 'mean_window_mae':float(np.mean([j['metrics']['mae'] for j in jobs])),
                 'subgroups':subgroup_values(part,pred,protocol),'impact_groups':group_metrics}
    comparisons=[]
    pairs=[('R0_B','R0_A'),('R0_C','R0_B'),('R1_A','R0_C'),('R1_B','R1_A'),('R1_B','R1_B_ADD'),('R1_C','R1_B'),('R1_D','R1_C')]
    for candidate,control in pairs:
        window_deltas=[];cycle_deltas=[]
        for window in protocol['primary_windows']:
            a=next(j for j in all_jobs if j['window']==window['id'] and j['model']==candidate)
            b=next(j for j in all_jobs if j['window']==window['id'] and j['model']==control)
            window_deltas.append({'window':window['id'],'delta_mae':a['metrics']['mae']-b['metrics']['mae']})
            cycle_deltas.extend({'cycle':x['cycle'],'delta_mae':x['mae']-y['mae']} for x,y in zip(a['per_cycle'],b['per_cycle']))
        comparisons.append({'candidate':candidate,'control':control,'negative_delta_is_better':True,
              'pooled_delta':aggregate['primary'][candidate]['raw']['mae']-aggregate['primary'][control]['raw']['mae'],
              'windows_improved':sum(d['delta_mae']<0 for d in window_deltas),
              'cycles_improved':sum(d['delta_mae']<0 for d in cycle_deltas),
              'window_deltas':window_deltas,'cycle_deltas':cycle_deltas})
    result={'status':'ROUND1_COMPLETE_REVIEW_NO_FINAL_MODEL','protocol_id':protocol['protocol_id'],
        'freeze_signature':frozen['signature'],'frozen_identity':frozen['identity'],
        'seed':protocol['seed'],'thread_limit':protocol['thread_limit'],
        'run_id':run_id,'total_jobs':len(all_jobs),'fresh_fits':fresh,'reused_fits':len(all_jobs)-fresh,
        'elapsed_seconds_this_run':perf_counter()-start_all,'jobs':all_jobs,'aggregate':aggregate,'comparisons':comparisons,
        'selection_boundary':'Temporary candidate only; no official test predictions, final model, Round2 or submission.',
        'artifact_location':f'experiments/runs/{run_id}/ (ignored validation arrays, not models)',
        'metric_conventions':{'signed_error':'prediction minus actual','primary':'raw/unclipped MAE',
            'regions':'[0,100/3), [100/3,200/3), [200/3,100]','tails':'actual <=5 and >=95',
            'aggregation':'Primary and H12 reported separately; no duplicate validation observations inside each family.'}}
    reports.mkdir(parents=True,exist_ok=True)
    write_json(reports/'results.json',result)
    for window_id,states in audits.items():
        write_json(reports/'fit_audits'/f'{window_id}.json',states)
    return result
