"""Independent pandas-rank/math.fsum audit required before candidate promotion."""
from pathlib import Path
import sys,json,math
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.models.regression_research_common import REPORT,CACHE,verify_stage,save_json
from src.models.regression_round2_evaluation import load_protocol,positions_digest
from src.models.round1_evaluation import array_digest,digest_json
from src.models.round1_models import _fit_fingerprint
from src.utils.paths import task_files
from src.data.audit import sha256


def mapping(p,c,a):
    p=np.asarray(p,float)
    ranked=pd.Series(p).groupby(np.asarray(c)).rank(method='average').to_numpy()
    n=pd.Series(np.asarray(c)).groupby(np.asarray(c)).transform('size').to_numpy()
    return np.clip((1-a)*p+a*100*ranked/n,0,100)


def mae(y,p):return math.fsum(abs(float(a)-float(b))for a,b in zip(y,p))/len(y)


def main():
    frozen=verify_stage('stage23_models');r=json.loads((REPORT/'stage23_models.json').read_text(encoding='utf-8'))
    assert r['freeze_signature']==frozen['signature']
    frame=pd.read_csv(task_files('regression')['train']);protocol=load_protocol();vectors={};audits=0
    for end_str,models in r['audits'].items():
        end=int(end_str)
        for kind,a in models.items():
            past=np.flatnonzero(frame.cycle_code.between(47,end));inn=np.flatnonzero(frame.cycle_code.between(47,end-4));iv=np.flatnonzero(frame.cycle_code.between(end-3,end))
            choices=[]
            for param,m in a['inner'].items():
                path=CACHE/f'model__{kind}__{param}__fit{end-4}__to{end}.npz'
                with np.load(path,allow_pickle=False)as z:
                    assert np.array_equal(z['fit_rows'],inn)and np.array_equal(z['validation_rows'],iv)
                    p=z['prediction'].copy()
                assert array_digest(p)==m['prediction_sha256']and digest_json(m['state'])==m['state_sha256']
                assert _fit_fingerprint(frame.iloc[inn][m['features']],tuple(m['features']),frame.iloc[inn].threat_rank.to_numpy())==m['state']['fit_data_sha256']
                for alpha in frozen['identity']['protocol']['rank_weights']:
                    choices.append({'param':param,'rank_weight':alpha,'mae':mae(frame.iloc[iv].threat_rank,mapping(p,frame.iloc[iv].cycle_code,alpha))})
                audits+=1
            recorded=next(v for v in r['inner']if v['fit_end']==end and v['recipe']==kind)
            assert len(choices)==len(recorded['all_choices'])
            for x,y in zip(choices,recorded['all_choices']):assert x['param']==y['param']and x['rank_weight']==y['rank_weight']and abs(x['mae']-y['mae'])<1e-10
            best=min(v['mae']for v in choices);order={'regularized':0,'base':1,'deeper':2}
            selected=sorted([v for v in choices if v['mae']<=best+1e-10],key=lambda v:(v['rank_weight'],order[v['param']]))[0]
            assert selected['param']==recorded['selected']['param']and selected['rank_weight']==recorded['selected']['rank_weight']
            m=a['outer'];last=m['validation_cycle_range'][1];rows=np.flatnonzero(frame.cycle_code.between(end+1,last))
            with np.load(CACHE/f"model__{kind}__{selected['param']}__fit{end}__to{last}.npz",allow_pickle=False)as z:
                assert np.array_equal(z['fit_rows'],past)and np.array_equal(z['validation_rows'],rows)
                raw=z['prediction'].copy()
            assert array_digest(raw)==m['prediction_sha256']and digest_json(m['state'])==m['state_sha256']
            assert _fit_fingerprint(frame.iloc[past][m['features']],tuple(m['features']),frame.iloc[past].threat_rank.to_numpy())==m['state']['fit_data_sha256']
            assert end<frame.iloc[rows].cycle_code.min()and m['validation_labels_passed_to_fit']is False
            for name,alpha in [('raw',None),('rank50',.5),('rank100',1.),('inner',selected['rank_weight'])]:
                p=raw if alpha is None else mapping(raw,frame.iloc[rows].cycle_code,alpha)
                with np.load(CACHE/f'eval__fit{end}__{kind}__{name}.npz',allow_pickle=False)as z:
                    assert np.array_equal(rows,z['validation_rows'])and np.max(np.abs(z['prediction']-p))<1e-11
                vectors[end,kind,name]=(rows,p)
            audits+=1
    jobs=0;aggregates=0
    for j in r['jobs']:
        w=next(w for w in protocol['primary_windows']+protocol['secondary_windows']if w['id']==j['window']);rows,p=vectors[w['fit'][1],j['recipe'],j['policy']]
        mask=frame.iloc[rows].cycle_code.between(*w['validation']).to_numpy();rows,p=rows[mask],p[mask];y=frame.iloc[rows].threat_rank.to_numpy()
        assert abs(mae(y,p)-j['raw']['mae'])<1e-10
        assert abs(math.fsum(float(a)-float(b)for a,b in zip(p,y))/len(y)-j['raw']['signed_error'])<1e-10
        jobs+=1
    for family,key in [('primary','primary_windows'),('secondary','secondary_windows')]:
        for label,summary in r['aggregate'][family].items():
            kind,name=label.split('__');parts=[]
            for w in protocol[key]:
                rows,p=vectors[w['fit'][1],kind,name];mask=frame.iloc[rows].cycle_code.between(*w['validation']).to_numpy();parts.append((rows[mask],p[mask]))
            rows=np.concatenate([x[0]for x in parts]);p=np.concatenate([x[1]for x in parts]);assert len(rows)==len(np.unique(rows))==38588
            part=frame.iloc[rows];y=part.threat_rank.to_numpy()
            assert abs(mae(y,p)-summary['raw']['mae'])<1e-10
            masks={'EF0':part.entry_friction.eq(0).to_numpy(),'EF1':part.entry_friction.eq(1).to_numpy(),'tail_low_le5':y<=5,'tail_high_ge95':y>=95,
                   'stream_missing':part.signal_stream.isna().to_numpy(),'stream_present':part.signal_stream.notna().to_numpy()}
            for group,mask in masks.items():
                assert int(mask.sum())==summary['subgroups'][group]['n']and abs(mae(y[mask],p[mask])-summary['subgroups'][group]['mae'])<1e-10
            cycles=[mae(y[part.cycle_code.eq(c).to_numpy()],p[part.cycle_code.eq(c).to_numpy()])for c in part.cycle_code.unique()]
            assert abs(np.mean(cycles)-summary['equal_cycle_mae'])<1e-10
            aggregates+=1
    receipt={'status':'PASS','stage_signature':frozen['signature'],'results_sha256':sha256(REPORT/'stage23_models.json'),
             'fit_audit_checks':audits,'job_checks':jobs,'aggregate_checks':aggregates,'inner_selection_checks':len(r['inner']),
             'preprocessing_input_fingerprints':'Recomputed from exact past-only source features+targets; vocabulary/SVD fit boundary separately synthetic-tested.',
             'mapping':'Independent pandas average rank; metrics math.fsum; MAE,bias,equalcycle,EF/tail/stream reconciled.',
             'no_refits':True,'test_labels_read':False}
    save_json(REPORT/'verification_promotions.json',receipt);print(receipt)


if __name__=='__main__':main()
