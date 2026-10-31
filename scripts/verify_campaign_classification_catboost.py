"""Independently recompute CatBoost routing metrics with sklearn, without fits."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support
from scripts.research_campaign_classification_catboost import (
    read, save, verify_freeze, load_train, load_cache, collect_window, concatenate, ENDS,
    PRIMARY_WINDOWS, OUT,
)

def verify():
    frozen=verify_freeze(); report=read(OUT/'results.json');frame,y,_=load_train(None)
    assert report['freeze_signature']==frozen['signature'] and report['new_fits']==6
    cycles=frame.cycle_code.to_numpy(dtype=int)
    fits={end:load_cache(frame,y,end,frozen['signature']) for end in ENDS}
    windows={name:collect_window(fits[fit],start,end,cycles) for name,(fit,start,end) in PRIMARY_WINDOWS.items()}
    records={**windows,'primary':concatenate(list(windows.values())), 'H12_E':fits[55],
             'H12_L':fits[67], 'H12':concatenate([fits[55],fits[67]])}
    primary=records['primary'];missing=primary['missing'];rows=primary['rows']
    cb=primary['candidate'][missing]>=.5;old=primary['c1'][missing]>=.5;actual=y[rows][missing]
    cb_only=int(((cb==actual)&(old!=actual)).sum());old_only=int(((old==actual)&(cb!=actual)).sum())
    assert report['paired_missing']['candidate_only_correct']==cb_only
    assert report['paired_missing']['C4_only_correct']==old_only
    gate=cb_only>=25 and old_only>=25 and cb_only>=old_only
    assert gate==report['blend_gate_pass']
    checks=0
    for name,r in records.items():
        section=report['windows'][name] if name in windows else report[name]
        m=r['missing']; routes={'C4_HGB':np.where(m,r['c1'],r['c2']),
                              'CB_MISSING':np.where(m,r['candidate'],r['c2'])}
        if gate: routes['CB_HGB_50']=np.where(m,.5*r['candidate']+.5*r['c1'],r['c2'])
        assert set(section)==set(routes)
        for model,p in routes.items():
            for subgroup,mask in [('all',np.ones(len(m),dtype=bool)),('missing',m),('present',~m)]:
                yy=y[r['rows']][mask]; pred=(p[mask]>=.5).astype(int);saved=section[model][subgroup]
                np.testing.assert_allclose(saved['macro_f1'],f1_score(yy,pred,labels=[0,1],average='macro',zero_division=0),rtol=0,atol=1e-12)
                assert saved['confusion_matrix']==confusion_matrix(yy,pred,labels=[0,1]).tolist()
                pr,re,f1,support=precision_recall_fscore_support(yy,pred,labels=[0,1],zero_division=0)
                for i,item in enumerate(saved['per_class']):
                    np.testing.assert_allclose([item['precision'],item['recall'],item['f1']], [pr[i],re[i],f1[i]],rtol=0,atol=1e-12)
                    assert item['support']==int(support[i])
                scores=[]
                cc=cycles[r['rows']][mask]
                for c in np.unique(cc):
                    cm=cc==c
                    scores.append(f1_score(yy[cm],pred[cm],labels=[0,1],average='macro',zero_division=0))
                np.testing.assert_allclose(saved['equal_cycle_macro_f1'],np.mean(scores),rtol=0,atol=1e-12)
                checks+=1
    result={'status':'PASS','freeze_signature':frozen['signature'],'metric_group_checks':checks,
            'six_fit_fingerprints_and_original_C1_C2_controls_reconciled':True,
            'primary_unique_rows':len(records['primary']['rows']), 'H12_unique_rows':len(records['H12']['rows']),
            'test_read':False,'new_fits':0}
    save(OUT/'verification.json',result);print(result)

if __name__=='__main__':verify()
