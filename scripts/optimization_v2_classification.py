"""Past-only calibrated and stacked present-stream score heads; no Test/API."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,importlib.metadata,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
from src.models.classification_models import build_classifier,_fit_fingerprint
from src.models.campaign_classification import build_campaign_classifier
from src.models.classification_round2_models import stream_missing_mask
from src.models.optimization_v2_classification import KINDS,TemporalScoreHead
from scripts.research_campaign_classification_stage2 import load_train,_load_round1_control,score_rows,PRIMARY_WINDOWS
from src.models.round1_evaluation import array_digest,code_digest,digest_json

CONFIG=ROOT/'experiments/optimization_v2/classification_calibration'
OUT=ROOT/'reports/optimization_v2/classification_calibration'
CACHE=ROOT/'experiments/runs/optimization_v2_classification_calibration'
OLD_CB=ROOT/'experiments/runs/dual_cls_catboost_v1'
ENDS=[55,59,63,67,71,75]

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n')as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n')
def nbpath(end):
    return ROOT/f'experiments/runs/{"dual_cls_v1a" if end in [55,67] else "dual_cls_v1_stage2"}/fit{end}__DC_NBSVM.npz'

def inputs():
    _,_,train=load_train(None);paths=[train]
    for end in ENDS:
        paths.extend([nbpath(end),nbpath(end).with_suffix('.json'),OLD_CB/f'fit{end}.npz',OLD_CB/f'fit{end}.json'])
        for name in ['C1_B','C2_B']:paths.append(ROOT/f'experiments/runs/classification_round1_v1/fit{end}__{name}.npz')
        paths.append(ROOT/f'reports/classification_round1/fit_audits/fit{end}.json')
    return {str(p.resolve().relative_to(ROOT)):sha(p) for p in paths}

def identity():
    milestone=read(ROOT/'experiments/optimization_v2/milestone.json')
    source=[p for p in milestone['historical_before_sha256'] if p.startswith('src/') and p.endswith('.py')]
    source+=['src/models/optimization_v2_classification.py','scripts/optimization_v2_classification.py',
             'scripts/optimization_v2_verify_classification.py','tests/test_optimization_v2_classification.py',
             'scripts/research_campaign_classification.py','scripts/research_campaign_classification_stage2.py']
    return {'protocol_sha256':sha(CONFIG/'protocol.json'),'code_sha256':{p:code_digest(ROOT/p) for p in source},
            'input_sha256':inputs(),'packages':{p:importlib.metadata.version(p)for p in
                ['numpy','pandas','scipy','scikit-learn','threadpoolctl']}}

def freeze():
    current=identity();path=CONFIG/'freeze.json'
    if path.exists():
        d=read(path);assert d['identity']==current and d['signature']==digest_json(current);return d
    assert not CACHE.exists()
    d={'at_utc':datetime.now(timezone.utc).isoformat(),'status':'FROZEN_BEFORE_FITS_AND_NEW_SCORES',
       'identity':current,'signature':digest_json(current),
       'source_commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()}
    save(path,d);return d
def check():
    d=read(CONFIG/'freeze.json');current=identity()
    assert d['identity']==current and d['signature']==digest_json(current);return d

def earliest(frame,y,signature,fit=False):
    path=CACHE/'inner51.npz';meta=path.with_suffix('.json')
    rows=np.flatnonzero(frame.cycle_code.between(52,55));mask=frame.cycle_code.between(47,51).to_numpy()
    if not path.exists():
        assert fit and not meta.exists();scores=[];states=[]
        for name in ['C2_B','DC_NBSVM']:
            model=build_classifier({'id':name}) if name=='C2_B'else build_campaign_classifier(name)
            fields=model.input_feature_names
            with threadpool_limits(limits=2):
                model.fit(frame.loc[mask,fields],y[mask]);state=model.audit_state();p=model.predict_proba(frame.iloc[rows][fields])
            assert state==model.audit_state();scores.append(p);states.append(state)
            print('FITTED inner51',name,flush=True)
        CACHE.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,rows=rows,scores=np.column_stack(scores))
        save(meta,{'freeze_signature':signature,'npz_sha256':sha(path),'states':states})
    m=read(meta);assert m['freeze_signature']==signature and m['npz_sha256']==sha(path)
    for state in m['states']:
        fields=state['input_feature_names']
        assert state['fit_rows']==int(mask.sum())
        assert state['fit_data_sha256']==_fit_fingerprint(frame.loc[mask,fields],tuple(fields),y[mask])
    with np.load(path,allow_pickle=False)as z:
        np.testing.assert_array_equal(z['rows'],rows);scores=z['scores'].copy()
    return rows,scores

def existing(frame,y,end):
    horizon={55:67,67:79}.get(end,end+4)
    rows,base,_=_load_round1_control(frame,end,horizon,'C2_B')
    with np.load(nbpath(end),allow_pickle=False)as z:
        np.testing.assert_array_equal(z['outer_rows'],rows);nb=z['candidate_probability'].copy()
    with np.load(OLD_CB/f'fit{end}.npz',allow_pickle=False)as z:
        np.testing.assert_array_equal(z['rows'],rows);cb=z['candidate'].copy()
    assert np.isfinite(nb).all() and ((nb>=0)&(nb<=1)).all()
    assert np.isfinite(cb).all() and ((cb>=0)&(cb<=1)).all()
    return rows,np.column_stack([base,nb]),cb

def compute(fit_earliest=False,save_vectors=False):
    frozen=check();frame,y,_=load_train(None);cycles=frame.cycle_code.to_numpy(dtype=int)
    first=earliest(frame,y,frozen['signature'],fit_earliest)
    old={end:existing(frame,y,end)for end in ENDS};records={};states={}
    for end in ENDS:
        if end==55:inner_rows,inner_scores=first
        else:
            rr,ss,_=old[end-4];which=(cycles[rr]>=end-3)&(cycles[rr]<=end)
            inner_rows,inner_scores=rr[which],ss[which]
        assert np.array_equal(inner_rows,np.flatnonzero((cycles>=end-3)&(cycles<=end)))
        use=~stream_missing_mask(frame.iloc[inner_rows]);r,s,cb=old[end]
        missing=stream_missing_mask(frame.iloc[r]);champ=np.where(missing,cb,s[:,0]);predictions={'champion':champ}
        states[str(end)]={'base_fit_end_for_inner':end-4,'calibration_cycles':[end-3,end],
                          'calibration_rows':int(use.sum()),'heads':{}}
        for kind in KINDS:
            head=TemporalScoreHead(kind).fit(inner_scores[use],y[inner_rows][use]);state=head.audit_state()
            pred=np.where(missing,cb,head.predict(s));assert state==head.audit_state()
            states[str(end)]['heads'][kind]=state;predictions[kind]=pred
        records[end]={'rows':r,'predictions':predictions}
        if save_vectors:np.savez_compressed(CACHE/f'outer{end}.npz',rows=r,**predictions)
    def evaluate(parts):
        rows=np.concatenate([p[0]for p in parts]);assert len(np.unique(rows))==len(rows)
        missing=stream_missing_mask(frame.iloc[rows]);result={}
        for kind in ['champion',*KINDS]:
            p=np.concatenate([v[kind]for _,v in parts]);actual=y[rows]
            result[kind]={'all':score_rows(frame,y,rows,p),'missing':score_rows(frame,y,rows[missing],p[missing]),
                          'present':score_rows(frame,y,rows[~missing],p[~missing]),
                          'probability_sha256':array_digest(p)}
        return result
    primary=[];windows={}
    for name,(end,lo,hi)in PRIMARY_WINDOWS.items():
        rec=records[end];which=(cycles[rec['rows']]>=lo)&(cycles[rec['rows']]<=hi)
        p=(rec['rows'][which],{k:v[which]for k,v in rec['predictions'].items()})
        primary.append(p);windows[name]=evaluate([p])
    h=[(records[e]['rows'],records[e]['predictions'])for e in [55,67]]
    return {'status':'REVIEW_REQUIRED','freeze_signature':frozen['signature'],'new_base_fits':2,
            'calibration_fits':len(ENDS)*len(KINDS),'test_read':False,'outer_labels_used_for_heads':False,
            'primary':evaluate(primary),'windows':windows,'H12':evaluate(h),'H12_E':evaluate([h[0]]),
            'H12_L':evaluate([h[1]]),'states':states,
            'warning':'Repeated adaptive development on known Train cycles; H12 outcomes overlap primary.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');a=p.parse_args()
    if a.freeze:print(freeze()['signature'])
    else:
        r=compute(True,True);save(OUT/'results.json',r)
        print('RESULTS',{k:round(v['all']['macro_f1'],9)for k,v in r['primary'].items()})
