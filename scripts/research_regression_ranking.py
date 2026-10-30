"""Stage1: preregistered prediction-only cycle rank tests on four frozen controls."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.models.regression_research_common import CONFIG,REPORT,CACHE,freeze_stage,verify_stage,save_json,mapped_scores,score_summary
from src.models.regression_round2_evaluation import load_protocol
from src.models.regression_round2_reporting import load_vector
from src.models.round1_evaluation import array_digest
from src.utils.paths import task_files

MODELS=['R1_C','R1_D','B_D_RECENT8','A_NO_CODES']
PROTOCOL={'objective':'Prediction-only rank geometry, not proof of official target generation',
          'controls':MODELS,'weights':[0.,.25,.5,.75,1.],
          'mapping':'Within supplied full cycle 100*average_rank(prediction)/n; raw weight0 clipped separately; blend uses raw then clips',
          'policies':['raw','clip','rank25','rank50','rank75','rank100'],
          'batch_stress':{'retained_fractions':[.25,.5,.75],'seed':42,'selection':'target-blind uniform within each cycle; sorted original positions; compare subset rank against full-cycle mapping on same rows'},
          'promotion':'Prefer fixed policy improving pooled primary and H12, inspect tails/EF/allwindows; never select weight from outer per-window oracle; eventual newmodel alpha uses past inner only',
          'limitations':'All cycles reused development; batch completeness beyond supplied files UNKNOWN. Transductive use of within-cycle predictions, no validation labels in mapping.'}


def main():
    freeze=freeze_stage('stage1_ranking',PROTOCOL,['src/models/regression_research_common.py','scripts/research_regression_ranking.py'])
    frame=pd.read_csv(task_files('regression')['train']);p=load_protocol()
    CACHE.mkdir(parents=True,exist_ok=True)
    jobs=[];vectors={};stress=[]
    policy={'raw':None,'clip':0.,'rank25':.25,'rank50':.5,'rank75':.75,'rank100':1.}
    for family,key in [('primary','primary_windows'),('secondary','secondary_windows')]:
        for w in p[key]:
            for m in MODELS:
                _,rows,raw=load_vector(w['fit'][1],m)
                mask=frame.iloc[rows].cycle_code.between(*w['validation']).to_numpy()
                rows,raw=rows[mask],raw[mask];part=frame.iloc[rows]
                for name,alpha in policy.items():
                    pred=raw.copy() if alpha is None else mapped_scores(raw,part.cycle_code.to_numpy(),alpha)
                    vectors[(family,w['id'],m,name)]=(rows,pred)
                    cache=f"ranking__{w['id']}__{m}__{name}"
                    np.savez_compressed(CACHE/(cache+'.npz'),validation_rows=rows,prediction=pred)
                    jobs.append({'model':m,'policy':name,'family':family,'window':w['id'],'prediction_sha256':array_digest(pred),**score_summary(part,pred)})
                if family=='primary':
                    for fraction in PROTOCOL['batch_stress']['retained_fractions']:
                        idx=[]
                        for cycle in np.unique(part.cycle_code):
                            positions=np.flatnonzero(part.cycle_code.eq(cycle))
                            rng=np.random.default_rng(np.random.SeedSequence([42,int(cycle),int(fraction*100)]))
                            idx.extend(rng.choice(positions,max(1,int(len(positions)*fraction)),replace=False).tolist())
                        idx=np.sort(idx);subset=part.iloc[idx]
                        for alpha in [.5,1.]:
                            full=mapped_scores(raw,part.cycle_code,alpha)[idx]
                            incomplete=mapped_scores(raw[idx],subset.cycle_code,alpha)
                            stress.append({'model':m,'window':w['id'],'fraction':fraction,'rank_weight':alpha,
                                           'n':len(idx),'full_mae':float(np.abs(full-subset.threat_rank.to_numpy()).mean()),
                                           'subset_mae':float(np.abs(incomplete-subset.threat_rank.to_numpy()).mean()),
                                           'mean_abs_mapping_change':float(np.abs(full-incomplete).mean())})
    aggregate={}
    for family,key in [('primary','primary_windows'),('secondary','secondary_windows')]:
        aggregate[family]={}
        for m in MODELS:
            for name in policy:
                pairs=[vectors[(family,w['id'],m,name)] for w in p[key]]
                rows=np.concatenate([v[0]for v in pairs]);pred=np.concatenate([v[1]for v in pairs]);order=np.argsort(rows)
                assert len(np.unique(rows))==len(rows)
                aggregate[family][m+'__'+name]=score_summary(frame.iloc[rows[order]],pred[order],impact=False)
    result={'stage':'stage1_ranking','freeze_signature':freeze['signature'],'fresh_fits':0,'jobs':jobs,'aggregate':aggregate,'batch_stress':stress,
            'boundary':'Outer labels evaluated only after fixed mapping. No official Test predictions at this stage.'}
    save_json(REPORT/'stage1_ranking.json',result)
    for m in MODELS:
        for name in policy:
            a=aggregate['primary'][m+'__'+name];b=aggregate['secondary'][m+'__'+name]
            print(m,name,f"primary={a['raw']['mae']:.6f}",f"H12={b['raw']['mae']:.6f}")
    verify_stage('stage1_ranking')


if __name__=='__main__':main()
