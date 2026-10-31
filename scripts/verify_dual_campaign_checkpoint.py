"""Read-only data/source/submission audit, with an exclusive checkpoint receipt."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, subprocess, sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.campaign_kaggle import validate
from scripts.research_campaign_classification_stage2 import (
    load_train, ALL_FIT_ENDS, CANDIDATES, STAGE1_RUN_DIR, RUN_DIR, ROUND1_CACHE, ROUND1_AUDITS,
)
from src.models.classification_round2_models import stream_missing_mask
from src.models.round1_evaluation import array_digest

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def verify(output):
    milestone=read(ROOT/'experiments/dual_campaign/milestone.json')
    authorized=set(milestone['authorized_academic_edits']);protected=[]
    for p,h in milestone['historical_before_sha256'].items():
        if p not in authorized:
            assert sha(ROOT/p)==h,('Historical change',p)
            protected.append(p)
    allocation='docs/WORK_ALLOCATION.md'
    before=subprocess.check_output(['git','-C',str(ROOT),'show',milestone['source_head']+':'+allocation],text=True,encoding='utf-8')
    after=(ROOT/allocation).read_text(encoding='utf-8')
    for member in ['Rousan','Abhi','Rizky']:
        assert [line for line in before.splitlines() if member in line]==[line for line in after.splitlines() if member in line],member
    frame,y,_=load_train(None);checked=[]
    for end in ALL_FIT_ENDS:
        controls={};control_rows=None
        audits=read(ROUND1_AUDITS/f'fit{end}.json')
        for model in ['C1_B','C2_B']:
            with np.load(ROUND1_CACHE/f'fit{end}__{model}.npz',allow_pickle=False) as z:
                rows=z['outer_rows'];p=z['probability']
                assert array_digest(p)==audits[model]['probability_sha256']
                if control_rows is not None:np.testing.assert_array_equal(rows,control_rows)
                control_rows=rows.copy();controls[model]=p.copy()
        assert len(np.unique(control_rows))==len(control_rows)
        assert (frame.iloc[control_rows].cycle_code.to_numpy()>end).all()
        for candidate in CANDIDATES:
            directory=STAGE1_RUN_DIR if end in [55,67] else RUN_DIR
            with np.load(directory/f'fit{end}__{candidate}.npz',allow_pickle=False) as z:
                r=z['outer_rows'];positions=np.searchsorted(control_rows,r)
                np.testing.assert_array_equal(control_rows[positions],r)
                missing=stream_missing_mask(frame.iloc[r]);np.testing.assert_array_equal(missing,z['missing_stream_mask'])
                c1=controls['C1_B'][positions];c2=controls['C2_B'][positions]
                np.testing.assert_array_equal(z['c4_probability'],np.where(missing,c1,c2))
                np.testing.assert_array_equal(z['routed_probability'],np.where(missing,c1,z['candidate_probability']))
                checked.append({'fit_end':end,'candidate':candidate,'rows':len(r),
                                'original_C1_C2_route_exact':True})
    academic=read(ROOT/'reports/academic_availability/QA.json')
    assert academic['status']=='PASS_PORTABLE_RESTART_RUN_ALL'
    assert academic['visual_review']['status']=='PASS' and academic['main_independent_review']['status']=='PASS'
    assert not academic['test_files_present'] and not academic['ignored_experiment_cache_present']
    for name,r in academic['notebooks'].items():
        assert r['execution_errors']==0
        assert sha(ROOT/r['path'])==r['notebook_sha256']
        notebook=read(ROOT/r['path'])
        assert not any(o.get('output_type')=='error' for c in notebook['cells'] for o in c.get('outputs',[]))
    # Additional independent AP/Brier and per-class checks for the new expert.
    from scripts.research_campaign_classification_catboost import (
        verify_freeze as check_cb_freeze, load_cache as load_cb_cache, collect_window,
        concatenate, PRIMARY_WINDOWS, ENDS,
    )
    from scripts.verify_campaign_classification_stage2 import _assert_metric
    cb_freeze=check_cb_freeze();cycles=frame.cycle_code.to_numpy(dtype=int)
    fits={end:load_cb_cache(frame,y,end,cb_freeze['signature']) for end in ENDS}
    windows={name:collect_window(fits[end],lo,hi,cycles) for name,(end,lo,hi) in PRIMARY_WINDOWS.items()}
    records={**windows,'primary':concatenate(list(windows.values())),
             'H12_E':fits[55],'H12_L':fits[67],'H12':concatenate([fits[55],fits[67]])}
    cb_report=read(ROOT/'reports/dual_campaign/classification_catboost/results.json');cb_checks=0
    for name,r in records.items():
        m=r['missing'];routes={'C4_HGB':np.where(m,r['c1'],r['c2']),
                              'CB_MISSING':np.where(m,r['candidate'],r['c2'])}
        if cb_report['blend_gate_pass']:routes['CB_HGB_50']=np.where(m,.5*r['candidate']+.5*r['c1'],r['c2'])
        section=cb_report['windows'][name] if name in windows else cb_report[name]
        for model,p in routes.items():
            for group,mask in [('all',np.ones(len(m),dtype=bool)),('missing',m),('present',~m)]:
                _assert_metric(section[model][group],y[r['rows']][mask],p[mask],name+'/'+model+'/'+group)
                cb_checks+=1
    submissions=[];csv_hashes=[];counts={'classification':0,'regression':0}
    for ledger in sorted((ROOT/'reports/dual_campaign/kaggle').glob('*__*_ledger.json')):
        d=read(ledger);task=d['task'];candidate=d['candidate_id']
        manifest=ROOT/f'reports/dual_campaign/kaggle/{candidate}.json'
        m,path=validate(task,manifest)
        assert d['status']=='COMPLETE' and d['receipt']['status']=='COMPLETE'
        assert d['receipt']['team_name']=='kasdud lovers' and d['receipt']['public_score'] is not None
        assert d['csv_sha256']==sha(path) and d['manifest_sha256']==sha(manifest)
        assert d['num_allowed_before']>1 and d['csv_sha256'] not in csv_hashes
        assert m['source_commit']==subprocess.check_output(['git','-C',str(ROOT),'rev-parse',m['source_commit']],text=True).strip()
        csv_hashes.append(d['csv_sha256']);counts[task]+=1
        submissions.append({'task':task,'candidate':candidate,'ref':d['ref'],'public_score':d['receipt']['public_score'],
                            'csv_rows':m['test_rows'],'source_commit':m['source_commit']})
    for task,n in counts.items():assert n<=milestone['research_upload_budget'][task]
    stages={}
    for p in ['reports/dual_campaign/regression/verification.json','reports/dual_campaign/catboost/verification.json',
              'reports/dual_campaign/classification/amendment_01/stage1_verification.json',
              'reports/dual_campaign/classification/stage2/full_six_verification.json',
              'reports/dual_campaign/classification/stage2/stage2_report_recovery_amendment_01_verification.json',
              'reports/dual_campaign/classification_catboost/verification.json']:
        d=read(ROOT/p);assert d['status']=='PASS',p;stages[p]=sha(ROOT/p)
    result={'status':'PASS','at_utc':datetime.now(timezone.utc).isoformat(),
            'historical_protected_files_byte_identical':len(protected),
            'approved_academic_historical_edits':sorted(authorized),
            'classification_original_control_reconciliation':checked,
            'classification_catboost_independent_AP_Brier_confusion_and_per_class_checks':cb_checks,
            'academic_portable_RunAll_and_11_figures_verified':True,
            'stage_verification_sha256':stages,'accepted_submissions':submissions,'campaign_upload_counts':counts,
            'limitations':['OOF/H12 outcomes reused during adaptive development.',
                           'Kaggle public scores do not establish private/generalization performance.',
                           'Fresh CatBoost unit and OOF fits were executed; earlier report-only recovery reused verified predictions.']}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8',newline='\n') as f:json.dump(result,f,indent=2);f.write('\n')
    print('PASS',len(protected),'historical files;',len(checked),'original control routes;',counts)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();verify(args.output)
