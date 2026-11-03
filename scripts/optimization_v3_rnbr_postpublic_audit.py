"""Read-only prediction audit; writes only new small reports, never fits models."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/optimization_v3/rnbr_postpublic_v1'
RUN = ROOT / 'experiments/runs/optimization_v3_research_next_v1'


def main():
    train = pd.read_csv(ROOT / 'data/raw/regression/Train.csv',
                        usecols=['record_id', 'cycle_code', 'threat_rank', 'impact_load'])
    sources, records, support = {}, [], []
    for origin in (59, 63, 67, 71, 75):
        if origin == 67:
            path = RUN / 'RNBR_PILOT_OFFICIAL_V1/recovered_bundles/RNBR_PILOT_E67_OBSERVED/arrays.npz'
        elif origin == 75:
            path = RUN / 'RNBR_PILOT_OFFICIAL_V1/bundles/RNBR_PILOT_E75_OBSERVED/arrays.npz'
        else:
            path = RUN / f'RNBR_CONTINUATION_OFFICIAL_V1/bundles/RNBR_CONT_E{origin}_OBSERVED/arrays.npz'
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as z:
            a = {k: z[k] for k in z.files}
        rows = a['score_rows']; y = train.iloc[rows].threat_rank.to_numpy()
        assert np.array_equal(train.iloc[rows].cycle_code.to_numpy(), a['score_cycles'])
        assert np.array_equal(train.iloc[rows].record_id.astype(str).to_numpy(), a['score_record_ids'])
        p, b = a['prediction'], a['baseline_prediction']
        assert np.array_equal(p, np.clip(b+a['delta'], 0, 100))
        missing = a['query_missing']
        assert np.array_equal(p[missing], b[missing])
        assert np.all(a['donor_source_cutoffs'] < a['donor_cycles'])
        if origin in (67,75):
            donor = train.iloc[a['donor_rows']][['cycle_code','impact_load']].reset_index(drop=True)
            donor['decile'] = np.minimum(9,np.floor(np.clip(a['donor_r06_predictions'],0,100)/10)).astype(int)
            counts = donor.impact_load.value_counts()
            qcounts = train.iloc[rows].impact_load.map(counts).fillna(0).to_numpy()
            sizes = donor.groupby(['cycle_code','impact_load','decile']).impact_load.transform('size')
            rng = np.random.default_rng(42)
            assignment = np.arange(len(donor))
            for _,group in donor.groupby(['cycle_code','impact_load','decile'],sort=True):
                ix=group.index.to_numpy(); assignment[ix]=rng.permutation(ix)
            present=~missing
            support.append({'origin':origin,'query_rows':len(rows),'present_rows':int(present.sum()),
                'present_same_impact_donors_ge5':int((qcounts[present]>=5).sum()),
                'present_available_donor_coverage':float((qcounts[present]>=5).mean()),
                'donor_rows':len(donor),'shuffle_eligible_fraction':float((sizes>=2).mean()),
                'shuffle_source_position_changed_fraction':float((assignment!=np.arange(len(donor))).mean()),
                'note':'Donor counts only; positive-cosine neighbor support is not yet established.'})
        score = a['neighbor_similarities']; neighbor = a['neighbor_source_rows']
        valid = neighbor >= 0
        ages = np.zeros_like(score)
        ages[valid] = (a['score_cycles'][:, None] - train.cycle_code.to_numpy()[np.maximum(neighbor, 0)])[valid]
        w = score**2
        mean_age = np.divide((w*ages).sum(axis=1), w.sum(axis=1), out=np.zeros(len(rows)), where=w.sum(axis=1)>0)
        frame = pd.DataFrame(dict(origin=origin, cycle=a['score_cycles'], horizon=a['score_cycles']-origin,
                                  missing=missing, baseline_error=np.abs(y-b), error=np.abs(y-p),
                                  gain=np.abs(y-b)-np.abs(y-p), delta=a['delta'],
                                  max_cosine=score.max(axis=1), donor_age=mean_age,
                                  baseline_bias=b-y, changed=p!=b,
                                  target_low=y<=5, target_high=y>=95))
        records.append(frame)
    allrows = pd.concat(records, ignore_index=True)
    cycle = allrows.groupby(['origin','cycle','horizon'], as_index=False).agg(
        n=('gain','size'), mae_r06=('baseline_error','mean'), mae_rnbr=('error','mean'),
        gain=('gain','mean'), mean_delta=('delta','mean'), baseline_bias=('baseline_bias','mean'),
        mean_cosine=('max_cosine','mean'), mean_donor_age=('donor_age','mean'))
    blocks=[]
    for origin in (59,63,67,71,75):
        for lo,hi in ((1,4),(5,8),(9,12)):
            f=allrows[(allrows.origin==origin)&allrows.horizon.between(lo,hi)]
            if len(f):
                blocks.append(dict(origin=origin,horizon=f'{lo}-{hi}',n=len(f),
                    mae_r06=f.baseline_error.mean(),mae_rnbr=f.error.mean(),gain=f.gain.mean(),
                    mean_delta=f.delta.mean(),baseline_bias=f.baseline_bias.mean(),
                    positive_cycles=int((f.groupby('cycle').gain.mean()>0).sum()),cycles=f.cycle.nunique()))
    primary=allrows[allrows.horizon<=4]
    assert len(primary)==34474  # W1 fallback is deliberately absent.
    diagnostic={}
    for name,f in [('learned_primary',primary),('E67_H12',allrows[allrows.origin==67])]:
        diagnostic[name]={'n':len(f),'gain':float(f.gain.mean()),'slices':{}}
        for label,mask in [('target_le5',f.target_low),('target_ge95',f.target_high),
                           ('middle_target',~f.target_low&~f.target_high),('negative_delta',f.delta<0),
                           ('positive_delta',f.delta>0),('missing',f.missing)]:
            s=f[mask]
            diagnostic[name]['slices'][label]={'n':len(s),'gain':float(s.gain.mean()) if len(s) else None}
    OUT.mkdir(parents=True,exist_ok=True)
    cycle.to_csv(OUT/'per_cycle.csv',index=False)
    pd.DataFrame(blocks).to_csv(OUT/'horizon_blocks.csv',index=False)
    result={'status':'ZERO_FIT_DIAGNOSTIC','sources_sha256':sources,
            'adaptive_context':'Designed after public score; descriptive, no independent validation or policy selection.',
            'views':diagnostic, 'label_free_impact_matching_feasibility':support}
    (OUT/'audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(pd.DataFrame(blocks).round(6).to_string(index=False))
    print(json.dumps(diagnostic,indent=2))
    print(json.dumps(support,indent=2))


if __name__=='__main__':
    main()
