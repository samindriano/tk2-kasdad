"""Verify saved vectors/metrics and optionally compare a separate full frozen rerun."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data.audit import write_json
from src.models.round1_behavior import analyze_round1_behavior
from src.models.round1_evaluation import verify_freeze
from src.utils.paths import PROJECT_ROOT


def compare_reproduction(original,repeat):
    if original['freeze_signature']!=repeat['freeze_signature']:
        raise ValueError('Repeat does not use identical frozen experiment.')
    jobs=lambda result:{(j['window'],j['model']):j for j in result['jobs']}
    a,b=jobs(original),jobs(repeat)
    if set(a)!=set(b) or len(a)!=64:
        raise ValueError('Expected identical 64 experiment job keys.')
    pred_equal=all(a[k]['prediction_sha256']==b[k]['prediction_sha256'] for k in a)
    state_equal=all(a[k]['fitted_state_sha256']==b[k]['fitted_state_sha256'] for k in a)
    delta=max(abs(a[k]['metrics'][metric]-b[k]['metrics'][metric]) for k in a for metric in ['mae','rmse','signed_error'])
    result={'status':'PASS' if pred_equal and state_equal and delta==0 else 'FAIL',
        'freeze_signature':original['freeze_signature'],'original_run':original['run_id'],'repeat_run':repeat['run_id'],
        'jobs_compared':len(a),'all_prediction_vectors_bit_identical':pred_equal,
        'all_fitted_audit_states_identical':state_equal,'max_metric_absolute_difference':delta,
        'original_fresh_fits':original['fresh_fits'],'repeat_fresh_fits':repeat['fresh_fits'],
        'original_elapsed_seconds':original['elapsed_seconds_this_run'],'repeat_elapsed_seconds':repeat['elapsed_seconds_this_run'],
        'scope':'Separate full refit of frozen eight recipes on eight development windows; no official-test predictions.'}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repeat-report',type=Path)
    args=p.parse_args()
    verify_freeze()
    behavior=analyze_round1_behavior()
    print('Independent vector metrics:',behavior['metric_audit'])
    if args.repeat_report:
        original=json.loads((PROJECT_ROOT/'reports/round1/results.json').read_text(encoding='utf-8'))
        repeat=json.loads(args.repeat_report.read_text(encoding='utf-8'))
        result=compare_reproduction(original,repeat)
        write_json(PROJECT_ROOT/'reports/round1/reproduction.json',result)
        print(json.dumps(result,indent=2))
        if result['status']!='PASS':raise RuntimeError('Reproduction failed.')


if __name__=='__main__':main()
