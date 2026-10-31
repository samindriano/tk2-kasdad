"""Expanding past-only OOF score stacking: linear text plus structured CatBoost."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
from src.models.optimization_v2_stack import StructuredTextHead
from src.models.campaign_classification_catboost import build_catboost_classifier
from src.models.classification_models import _fit_fingerprint
from scripts import optimization_v2_classification as parent

read,save,sha=parent.read,parent.save,parent.sha
CONFIG=ROOT/'experiments/optimization_v2/classification_stack'
OUT=ROOT/'reports/optimization_v2/classification_stack'
CACHE=ROOT/'experiments/runs/optimization_v2_classification_stack'
KINDS=['expanding_base','expanding_stack_cb']

def identity():
    base=parent.check();files=['src/models/optimization_v2_stack.py','scripts/optimization_v2_stack.py',
        'scripts/optimization_v2_verify_stack.py','tests/test_optimization_v2_stack.py']
    return {'parent_freeze_signature':base['signature'],'protocol_sha256':sha(CONFIG/'protocol.json'),
            'source_sha256':{p:parent.code_digest(ROOT/p)for p in files},
            'parent_inner51_sha256':sha(parent.CACHE/'inner51.npz'),
            'parent_inner51_meta_sha256':sha(parent.CACHE/'inner51.json')}

def freeze():
    current=identity();path=CONFIG/'freeze.json'
    if path.exists():
        d=read(path);assert d['identity']==current and d['signature']==parent.digest_json(current);return d
    assert not CACHE.exists()
    d={'status':'FROZEN_BEFORE_NEW_FIT_AND_STACK_SCORES','at_utc':datetime.now(timezone.utc).isoformat(),
       'identity':current,'signature':parent.digest_json(current)};save(path,d);return d
def check():
    d=read(CONFIG/'freeze.json');assert d['identity']==identity()and d['signature']==parent.digest_json(d['identity']);return d

def inner51_cb(frame,y,signature,fit=False):
    path=CACHE/'cb51.npz';meta=path.with_suffix('.json')
    rows=np.flatnonzero(frame.cycle_code.between(52,55));mask=frame.cycle_code.between(47,51).to_numpy()
    if not path.exists():
        assert fit and not meta.exists();model=build_catboost_classifier();fields=model.input_feature_names
        with threadpool_limits(limits=2):
            model.fit(frame.loc[mask,fields],y[mask]);state=model.audit_state();p=model.predict_proba(frame.iloc[rows][fields])
        assert state==model.audit_state();CACHE.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,rows=rows,probability=p)
        save(meta,{'freeze_signature':signature,'npz_sha256':sha(path),'state':state});print('FITTED CB inner51',flush=True)
    d=read(meta);assert d['freeze_signature']==signature and d['npz_sha256']==sha(path)
    state=d['state'];fields=state['input_feature_names']
    assert state['fit_data_sha256']==_fit_fingerprint(frame.loc[mask,fields],tuple(fields),y[mask])
    with np.load(path,allow_pickle=False)as z:
        np.testing.assert_array_equal(z['rows'],rows);p=z['probability'].copy()
    return rows,p

def compute(new_fit=False,save_vectors=False):
    frozen=check();frame,y,_=parent.load_train(None);cycles=frame.cycle_code.to_numpy(dtype=int)
    first_rows,first_scores=parent.earliest(frame,y,parent.check()['signature'],False)
    cb_rows,cb_first=inner51_cb(frame,y,frozen['signature'],new_fit);np.testing.assert_array_equal(first_rows,cb_rows)
    first=(first_rows,np.column_stack([first_scores[:,0],cb_first]))
    existing={end:parent.existing(frame,y,end)for end in parent.ENDS};records={};states={}
    for end in parent.ENDS:
        parts=[first]
        for earlier in parent.ENDS:
            if earlier>=end:break
            r,s,cb=existing[earlier];use=(cycles[r]>earlier)&(cycles[r]<=earlier+4)
            parts.append((r[use],np.column_stack([s[use,0],cb[use]])))
        inner_rows=np.concatenate([p[0]for p in parts]);scores=np.concatenate([p[1]for p in parts])
        assert len(np.unique(inner_rows))==len(inner_rows) and cycles[inner_rows].min()==52 and cycles[inner_rows].max()==end
        assert set(inner_rows)==set(np.flatnonzero((cycles>=52)&(cycles<=end)))
        inner_present=~parent.stream_missing_mask(frame.iloc[inner_rows])
        r,s,cb=existing[end];missing=parent.stream_missing_mask(frame.iloc[r]);design=np.column_stack([s[:,0],cb])
        predictions={'champion':np.where(missing,cb,s[:,0])};states[str(end)]={}
        for kind in KINDS:
            head=StructuredTextHead(kind).fit(scores[inner_present],y[inner_rows][inner_present]);state=head.audit_state()
            predictions[kind]=np.where(missing,cb,head.predict(design));assert state==head.audit_state()
            states[str(end)][kind]={'state':state,'calibration_cycles':[52,end],
                                 'rows_sha256':parent.array_digest(inner_rows[inner_present]),
                                 'base_fit_cutoffs':[51,*[e for e in parent.ENDS if e<end]]}
        records[end]={'rows':r,'predictions':predictions}
        if save_vectors:np.savez_compressed(CACHE/f'outer{end}.npz',rows=r,**predictions)
    def evaluate(parts):
        rows=np.concatenate([p[0]for p in parts]);assert len(np.unique(rows))==len(rows)
        missing=parent.stream_missing_mask(frame.iloc[rows]);result={}
        for kind in ['champion',*KINDS]:
            p=np.concatenate([d[kind]for _,d in parts])
            result[kind]={'all':parent.score_rows(frame,y,rows,p),
                          'missing':parent.score_rows(frame,y,rows[missing],p[missing]),
                          'present':parent.score_rows(frame,y,rows[~missing],p[~missing]),
                          'probability_sha256':parent.array_digest(p)}
        return result
    primary=[];windows={}
    for name,(end,lo,hi)in parent.PRIMARY_WINDOWS.items():
        rec=records[end];use=(cycles[rec['rows']]>=lo)&(cycles[rec['rows']]<=hi)
        p=(rec['rows'][use],{k:v[use]for k,v in rec['predictions'].items()});primary.append(p);windows[name]=evaluate([p])
    h=[(records[e]['rows'],records[e]['predictions'])for e in [55,67]]
    return {'status':'REVIEW_REQUIRED','freeze_signature':frozen['signature'],'new_base_fits':1,'head_fits':12,
            'test_read':False,'outer_labels_used_for_heads':False,'states':states,
            'primary':evaluate(primary),'windows':windows,'H12':evaluate(h),'H12_E':evaluate([h[0]]),'H12_L':evaluate([h[1]])}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');a=p.parse_args()
    if a.freeze:print(freeze()['signature'])
    else:
        d=compute(True,True);save(OUT/'results.json',d)
        print('RESULTS',{k:[round(v['all']['macro_f1'],9),round(d['H12'][k]['all']['macro_f1'],9)]for k,v in d['primary'].items()})
