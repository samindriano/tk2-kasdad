"""No-fit audit for the narrowed continuation decision; opens only Train/caches."""
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from scripts.optimization_v3_postscreen_review import load_bundle, validate_bundle, contrast, sha
from src.utils.paths import PROJECT_ROOT, task_files

OUT = PROJECT_ROOT / 'reports/optimization_v3/final_continuation_plan'
VIEWS = ('primary', 'H12', 'W1', 'W4')


def group_audit(y, candidate, baseline, missing, masks):
    records, failures = [], []
    for view in VIEWS:
        for name, group in [('missing', missing), ('low_le5', y <= 5), ('high_ge95', y >= 95)]:
            m = masks[view] & group
            item = {'view': view, 'group': name, **contrast(y[m], candidate[m], baseline[m], 'regression')}
            item['supported'] = item['n'] >= 100
            item['harm'] = None if item['gain'] is None else -item['gain']
            item['pass'] = None if not item['supported'] else item['harm'] <= .25
            records.append(item)
            if item['pass'] is False:
                failures.append(item)
    return {'all_supported_pass': not failures, 'records': records, 'failures': failures}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw_path = task_files('regression')['train']
    assert sha(raw_path) == '756523ff8156a9cd776e52249f7e2a8f23dcec432b8d10570e55ff6db0e0f082'
    f = pd.read_csv(raw_path, usecols=['record_id','threat_rank','cycle_code','signal_stream'])
    missing = (f.signal_stream.isna() | f.signal_stream.fillna('').str.strip().eq('')).to_numpy()
    all_results, inputs = {}, {}
    for cid in ('R22', 'R06', 'R21', 'R11', 'R12'):
        arrays = []
        for cutoff in (55, 67):
            path, a = load_bundle(cid, cutoff)
            assert a is not None
            validate_bundle(a, f, cutoff)
            inputs[str(path.relative_to(PROJECT_ROOT))] = sha(path)
            inputs[str(path.with_suffix('.json').relative_to(PROJECT_ROOT))] = sha(path.with_suffix('.json'))
            arrays.append(a)
        rows, baseline, frozen, raw = [np.concatenate([a[key] for a in arrays])
                                       for key in ('score_rows','baseline_prediction','prediction','raw_candidate')]
        assert len(np.unique(rows)) == len(rows) == 38588
        y, cycle, miss = f.threat_rank.to_numpy()[rows], f.cycle_code.to_numpy()[rows], missing[rows]
        percentile = np.empty(len(rows))
        for c in np.unique(cycle):
            m = cycle == c
            percentile[m] = 100*rankdata(raw[m], method='average')/m.sum()
        alternative = np.where(miss, baseline, np.clip(.75*baseline+.25*percentile, 0, 100))
        masks = {'H12': np.ones(len(rows), bool), 'W1': cycle <= 59, 'W4': (cycle>=68)&(cycle<=71)}
        masks['primary'] = masks['W1'] | masks['W4']
        masks.update(H12_E=cycle<=67, H12_L=cycle>=68, pilot_bin_72_79=cycle>=72)
        all_results[cid] = {}
        for name, p in [('frozen', frozen), ('pct_present_diagnostic', alternative)]:
            all_results[cid][name] = {'metrics': {v: contrast(y[m], p[m], baseline[m], 'regression')
                                                for v,m in masks.items()},
                                     'harmonized_guard': group_audit(y,p,baseline,miss,masks)}
    classification = {}
    for cid in ('C16','C11','C17'):
        path = PROJECT_ROOT/f'reports/optimization_v3/batch1/{cid}/verification.json'
        inputs[str(path.relative_to(PROJECT_ROOT))] = sha(path)
        d = json.loads(path.read_text(encoding='utf-8'))
        rows, failures = [], []
        for view in VIEWS:
            for field, groups in d['views'][view]['subgroup_diagnostics']['groups'].items():
                for group, item in groups.items():
                    if not item.get('guard_applies'):
                        continue
                    assert item['n'] >= 200 and min(item['class_0_n'],item['class_1_n']) >= 30
                    passed = (item['candidate_minus_baseline_macro_f1'] >= -.01
                              and item['candidate_minus_baseline_positive_recall'] >= -.02)
                    record = {'view':view,'field':field,'group':group,'n':item['n'],
                              'delta_f1':item['candidate_minus_baseline_macro_f1'],
                              'delta_positive_recall':item['candidate_minus_baseline_positive_recall'],'pass':passed}
                    rows.append(record)
                    if not passed: failures.append(record)
        classification[cid] = {'supported_guard_checks':len(rows),'all_supported_pass':not failures,
                               'failures':failures,'records':rows,
                               'coverage':'independent arithmetic from saved detailed verifier; not refit or fresh group-mask construction'}
    result = {'status':'AUDIT_COMPLETE_CAMPAIGN_STILL_PAUSED','source_head':subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=PROJECT_ROOT,text=True).strip(),
        'pilot_guard_views':list(VIEWS),'classification':classification,'regression':all_results,
        'new_origin_windows':{'59':[60,63],'63':[64,67],'71':[72,75],'75':[76,79]},
        'future_late_primary_rows':int(f.cycle_code.between(72,79).sum()),
        'existing_candidate_fits':94,'proposed_additional_fits':8,'projected_remaining':78,
        'budget_is_snapshot_not_reservation':True,'input_hashes':inputs,'script_sha256':sha(__file__),
        'new_training_fits':0,'test_opened':False,'kaggle_calls':0}
    (OUT/'guard_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'classification':{c:(x['supported_guard_checks'],x['all_supported_pass']) for c,x in classification.items()},
                      'regression':{c:{p:v['harmonized_guard']['all_supported_pass'] for p,v in x.items()} for c,x in all_results.items()},
                      'late_primary_rows':result['future_late_primary_rows']},indent=2))


if __name__ == '__main__':
    main()
