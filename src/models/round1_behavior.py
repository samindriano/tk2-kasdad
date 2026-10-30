"""Post-fit diagnostics from frozen validation vectors; no fitting or new recipe."""
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd

from src.data.audit import write_json
from src.models.round1_evaluation import verify_freeze, load_protocol, window_indices, array_digest
from src.utils.paths import PROJECT_ROOT, task_files


def analyze_round1_behavior(results_path=None,data_dir=None,output_dir=None):
    frozen=verify_freeze(data_dir)
    path=Path(results_path) if results_path else PROJECT_ROOT/'reports/round1/results.json'
    result=json.loads(path.read_text(encoding='utf-8'))
    if result['freeze_signature']!=frozen['signature']:
        raise RuntimeError('Results not tied to current frozen experiment.')
    protocol=load_protocol()
    frame=pd.read_csv(task_files('regression',data_dir)['train'])
    artifact=PROJECT_ROOT/'experiments/runs'/result['run_id']
    windows={w['id']:w for w in protocol['primary_windows']+protocol['secondary_windows']}
    arrays={};metric_errors=[];unknown_checks=[]
    for job in result['jobs']:
        window,model=job['window'],job['model']
        with np.load(artifact/f'{window}__{model}.npz',allow_pickle=False) as saved:
            rows,pred=saved['validation_rows'],saved['prediction']
        fit,expected=window_indices(frame,windows[window])
        if not np.array_equal(rows,expected) or array_digest(pred)!=job['prediction_sha256']:
            raise RuntimeError('Validation vector identity mismatch.')
        y=frame.iloc[rows].threat_rank.to_numpy()
        error=pred-y
        # Python compensated sums independently reconcile the numpy experiment metrics.
        mae=math.fsum(abs(float(v)) for v in error)/len(error)
        bias=math.fsum(float(v) for v in error)/len(error)
        metric_errors.extend([abs(mae-job['metrics']['mae']),abs(bias-job['metrics']['signed_error'])])
        arrays[(window,model)]=(rows,pred)
    refresh=[]
    for old,new in [('H12_E','W2'),('H12_E','W3'),('H12_L','W5'),('H12_L','W6')]:
        for recipe in protocol['recipes']:
            model=recipe['id'];old_rows,old_pred=arrays[(old,model)];new_rows,new_pred=arrays[(new,model)]
            common,a,b=np.intersect1d(old_rows,new_rows,return_indices=True)
            y=frame.iloc[common].threat_rank.to_numpy()
            old_mae=float(np.abs(old_pred[a]-y).mean());new_mae=float(np.abs(new_pred[b]-y).mean())
            refresh.append({'model':model,'older_fit_window':old,'refreshed_fit_window':new,
                'older_fit_cycles':windows[old]['fit'],'refreshed_fit_cycles':windows[new]['fit'],
                'evaluation_cycles':[int(frame.iloc[common].cycle_code.min()),int(frame.iloc[common].cycle_code.max())],
                'n_same_validation_rows':len(common),'older_fit_mae':old_mae,'refreshed_fit_mae':new_mae,
                'delta_mae_refresh_minus_older':new_mae-old_mae})
    same_fit=[]
    for short,long in [('W1','H12_E'),('W4','H12_L')]:
        for recipe in protocol['recipes']:
            model=recipe['id'];a,p=arrays[(short,model)];b,q=arrays[(long,model)]
            common,ai,bi=np.intersect1d(a,b,return_indices=True)
            same_fit.append({'model':model,'windows':[short,long],'n':len(common),
                'identical_predictions':bool(np.array_equal(p[ai],q[bi])),
                'max_absolute_difference':float(np.max(np.abs(p[ai]-q[bi])))})
    novelty=[]
    for window in protocol['primary_windows']+protocol['secondary_windows']:
        fit,val=window_indices(frame,window);sub=frame.iloc[val]
        unseen=np.zeros(len(val),dtype=bool);missing=np.zeros(len(val),dtype=bool)
        counts={}
        for column in ['org_code','module_code','pattern_code']:
            vocabulary=set(frame.iloc[fit][column].dropna().astype(str).str.strip())
            mask=sub[column].notna().to_numpy() & (~sub[column].astype(str).str.strip().isin(vocabulary)).to_numpy()
            unseen|=mask;missing|=sub[column].isna().to_numpy();counts[column]=int(mask.sum())
        meta=json.loads((artifact/f"{window['id']}__R1_D.json").read_text(encoding='utf-8'))
        unknown_checks.append({'window':window['id'],'independently_counted':counts,
            'model_reported':meta['prediction_diagnostics']['unknown_category_rows'],
            'matches':counts==meta['prediction_diagnostics']['unknown_category_rows']})
        for name,mask in [('all_codes_known_nonmissing',~unseen&~missing),('any_unseen_nonmissing_code',unseen),('any_missing_code',missing)]:
            y=sub.threat_rank.to_numpy()[mask]
            entry={'window':window['id'],'n':int(mask.sum()),'group':name,'share':float(mask.mean())}
            for model in ['R1_C','R1_D']:
                pred=arrays[(window['id'],model)][1][mask]
                entry[model+'_mae']=float(np.abs(pred-y).mean()) if len(y) else None
                entry[model+'_bias']=float((pred-y).mean()) if len(y) else None
            novelty.append(entry)
    difficult={}
    for model in ['R0_C','R1_C','R1_D']:
        groups=result['aggregate']['primary'][model]['impact_groups']
        supported=[{'impact_load':float(k),**v} for k,v in groups.items() if v['n']>=100]
        difficult[model]=sorted(supported,key=lambda v:v['mae'],reverse=True)[:10]
    diagnostics={'status':'POSTFIT_DIAGNOSTICS_FROM_FROZEN_PREDICTIONS_NO_NEW_MODELS',
        'freeze_signature':frozen['signature'],'metric_audit':{'jobs_checked':len(result['jobs']),
             'max_absolute_mae_or_signed_error_difference':max(metric_errors),'method':'Independent math.fsum over each saved validation error vector.'},
        'same_fit_overlap_checks':same_fit,'refresh_same_validation_rows':refresh,
        'category_unknown_audit':unknown_checks,'category_novelty_groups':novelty,
        'difficult_impact_groups_n_ge100':difficult,
        'interpretation_limits':['Refresh changes training history and sample size together, so not causal attribution to a cycle effect.',
            'Novelty groups overlap when a row has both an unknown code and another missing code; do not sum groups.',
            'Post-fit diagnostics are exploratory, not independent additional validation or a model search.']}
    if max(metric_errors)>1e-10 or not all(c['matches'] for c in unknown_checks):
        raise RuntimeError('Independent metric or unknown-category reconciliation failed.')
    out=Path(output_dir) if output_dir else PROJECT_ROOT/'reports/round1'
    write_json(out/'behavior_diagnostics.json',diagnostics)
    return diagnostics
