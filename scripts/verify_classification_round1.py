"""Independent sklearn reconciliation of frozen classifier vectors; no training."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support, average_precision_score

from src.data.audit import write_json
from src.models.classification_evaluation import verify_freeze, load_protocol
from src.models.round1_evaluation import array_digest, digest_json, window_indices
from src.utils.paths import PROJECT_ROOT, task_files


def verify_results(repeat_report=None):
    frozen = verify_freeze()
    protocol = load_protocol()
    report = PROJECT_ROOT/'reports/classification_round1'
    result = json.loads((report/'results.json').read_text(encoding='utf-8'))
    if result['status'] != 'CLASSIFICATION_ROUND1_COMPLETE_REVIEW_NO_FINAL_MODEL' or len(result['jobs']) != 56:
        raise ValueError('Complete original experiment required.')
    if result['freeze_signature'] != frozen['signature']:
        raise ValueError('Freeze/result identity mismatch.')
    f = pd.read_csv(task_files('classification')['train'])
    artifact = PROJECT_ROOT/'experiments/runs'/result['run_id']
    arrays, metas = {}, {}
    threshold_checks = []
    errors = []
    for end in result['unique_outer_histories']:
        for recipe in protocol['recipes']:
            key = (end, recipe['id']); tag = f'fit{end}__{recipe["id"]}'
            meta = json.loads((artifact/f'{tag}.json').read_text(encoding='utf-8'))
            with np.load(artifact/f'{tag}.npz', allow_pickle=False) as a:
                rows, p, inner_rows, inner_p = [a[k].copy() for k in ['outer_rows', 'probability', 'inner_rows', 'inner_probability']]
            assert array_digest(p) == meta['probability_sha256']
            assert array_digest(inner_p) == meta['inner_probability_sha256']
            assert digest_json(meta['fitted_state']) == meta['fitted_state_sha256']
            assert meta['fit_rows'] == f.cycle_code.between(47, end).sum()
            assert np.array_equal(inner_rows, np.flatnonzero(f.cycle_code.between(end-3, end)))
            assert meta['fitted_state']['fit_rows'] == meta['fit_rows']
            assert not meta['outer_validation_labels_passed_to_fit_or_selection']
            assert meta['threshold_selected_before_outer_refit']
            assert meta['fit_started_at_utc'] > frozen['frozen_at_utc']
            assert not set(meta['fitted_state']['input_feature_names']) & set(protocol['excluded_features'])
            if recipe['threshold_selection'] and len(inner_p):
                y = f.iloc[inner_rows].artifact_signal.to_numpy()
                values = [(t, f1_score(y, inner_p >= t, labels=[0, 1], average='macro', zero_division=0)) for t in protocol['inner_threshold']['grid']]
                maximum = max(v for _, v in values)
                threshold, score = min([(t, v) for t, v in values if abs(v-maximum) <= 1e-12], key=lambda v: (round(abs(v[0]-.5), 12), v[0]))
                assert threshold == meta['selection']['threshold']
                assert abs(score-meta['selection']['inner_macro_f1']) < 1e-12
                assert meta['inner_state']['fit_rows'] == f.cycle_code.between(47, end-4).sum()
                assert digest_json(meta['inner_state']) == meta['inner_state_sha256']
                threshold_checks.append({'fit_end': end, 'model': recipe['id'], 'threshold': threshold, 'inner_score': float(score)})
            arrays[key], metas[key] = (rows, p, inner_rows, inner_p), meta
    windows = {w['id']: w for w in protocol['primary_windows']+protocol['secondary_windows']}
    pools = {}
    cycle_checks = 0
    for job in result['jobs']:
        fit, rows = window_indices(f, windows[job['window']])
        all_rows, all_p, _, _ = arrays[(job['fit_group'], job['model'])]
        where = np.searchsorted(all_rows, rows)
        assert np.array_equal(all_rows[where], rows)
        p = all_p[where]; y = f.iloc[rows].artifact_signal.to_numpy()
        assert array_digest(p) == job['probability_sha256']
        for policy, values in job['policies'].items():
            t = .5 if policy == 'fixed_0.5' else job['threshold_selection']['threshold']
            prediction = p >= t
            m = values['metrics']
            score = f1_score(y, prediction, labels=[0, 1], average='macro', zero_division=0)
            errors.append(abs(score-m['macro_f1']))
            assert confusion_matrix(y, prediction, labels=[0, 1]).tolist() == m['confusion_matrix']
            precision, recall, f1, support = precision_recall_fscore_support(y, prediction, labels=[0, 1], zero_division=0)
            for label in [0, 1]:
                for name, actual in [('precision', precision[label]), ('recall', recall[label]), ('f1', f1[label]), ('support', support[label])]:
                    assert abs(actual-m['per_class'][label][name]) < 1e-12
            errors.append(abs(average_precision_score(y, p)-m['average_precision']))
            for cycle in values['per_cycle']:
                mask = f.iloc[rows].cycle_code.eq(cycle['cycle']).to_numpy()
                errors.append(abs(f1_score(y[mask], prediction[mask], labels=[0, 1], average='macro', zero_division=0)-cycle['macro_f1']))
                cycle_checks += 1
            pools.setdefault((job['family'], policy, job['model']), []).append((rows, p, np.full(len(rows), t)))
    aggregate_checks = 0
    for (family, policy, model), chunks in pools.items():
        rows = np.concatenate([v[0] for v in chunks]); p = np.concatenate([v[1] for v in chunks]); thresholds = np.concatenate([v[2] for v in chunks])
        assert len(rows) == len(np.unique(rows)) == 38588
        y = f.iloc[rows].artifact_signal.to_numpy(); prediction = p >= thresholds
        a = result['aggregate'][family][policy][model]
        errors.append(abs(f1_score(y, prediction, labels=[0, 1], average='macro', zero_division=0)-a['metrics']['macro_f1']))
        assert confusion_matrix(y, prediction, labels=[0, 1]).tolist() == a['metrics']['confusion_matrix']
        equal = np.mean([f1_score(y[f.iloc[rows].cycle_code.eq(c)], prediction[f.iloc[rows].cycle_code.eq(c)], labels=[0, 1], average='macro', zero_division=0) for c in range(56, 80)])
        errors.append(abs(equal-a['equal_cycle_macro_f1']))
        aggregate_checks += 1
    paired = []
    for old, new in [(55, 59), (55, 63), (67, 71), (67, 75)]:
        lo, hi = new+1, new+4
        for recipe in protocol['recipes']:
            model = recipe['id']
            old_rows, old_p, _, _ = arrays[(old, model)]; new_rows, new_p, _, _ = arrays[(new, model)]
            rows, ai, bi = np.intersect1d(old_rows, new_rows, return_indices=True)
            y = f.iloc[rows].artifact_signal.to_numpy()
            entry = {'model': model, 'older_fit_end': old, 'refreshed_fit_end': new,
                     'evaluation_cycles': [lo, hi], 'n_same_rows': len(rows)}
            for name, end, p in [('older', old, old_p[ai]), ('refreshed', new, new_p[bi])]:
                selected = metas[(end, model)]['selection']['threshold']
                entry[name+'_fixed_f1'] = float(f1_score(y, p >= .5, labels=[0, 1], average='macro', zero_division=0))
                entry[name+'_selected_f1'] = float(f1_score(y, p >= selected, labels=[0, 1], average='macro', zero_division=0))
                entry[name+'_threshold'] = selected
                entry[name+'_ap'] = float(average_precision_score(y, p))
            paired.append(entry)
    verification = {'status': 'PASS', 'freeze_signature': frozen['signature'], 'evaluation_jobs_verified': len(result['jobs']),
                    'original_vector_set_sha256': digest_json({f'{end}__{model}': {'outer': m['probability_sha256'], 'inner': m['inner_probability_sha256'],
                                                     'state': m['fitted_state_sha256'], 'inner_state': m['inner_state_sha256']}
                                                     for (end, model), m in metas.items()}),
                    'policy_window_metrics_verified': len(result['jobs'])*2, 'per_cycle_metrics_verified': cycle_checks,
                    'family_policy_model_aggregates_verified': aggregate_checks, 'independent_metric_max_absolute_difference': max(errors),
                    'threshold_checks': threshold_checks, 'thresholds_recomputed_independently': len(threshold_checks),
                    'regression_protected_files_unchanged': len(frozen['regression_protected_sha256']),
                    'paired_history_diagnostics': paired,
                    'method': 'Independent sklearn F1/confusion/per-class/AP against saved probabilities and original Train; all inner cutoffs independently recomputed; no fits.'}
    assert max(errors) < 1e-12
    if repeat_report is not None:
        repeat = json.loads(Path(repeat_report).read_text(encoding='utf-8'))
        assert repeat['freeze_signature'] == result['freeze_signature']
        repeat_root = PROJECT_ROOT/'experiments/runs'/repeat['run_id']
        outer_same = inner_same = state_same = threshold_same = 0
        for recipe in protocol['recipes']:
            key = (67, recipe['id']); tag = f'fit67__{recipe["id"]}'
            other = json.loads((repeat_root/f'{tag}.json').read_text(encoding='utf-8'))
            with np.load(repeat_root/f'{tag}.npz', allow_pickle=False) as a:
                assert np.array_equal(a['outer_rows'], arrays[key][0])
                assert np.array_equal(a['probability'], arrays[key][1]); outer_same += 1
                assert np.array_equal(a['inner_rows'], arrays[key][2])
                assert np.array_equal(a['inner_probability'], arrays[key][3])
                if recipe['threshold_selection']: inner_same += 1
            assert other['fitted_state_sha256'] == metas[key]['fitted_state_sha256']; state_same += 1
            assert other['inner_state_sha256'] == metas[key]['inner_state_sha256']
            assert other['selection'] == metas[key]['selection']; threshold_same += 1
        verification['representative_refit'] = {'status': 'PASS', 'fit_end': 67, 'fresh_model_fits': repeat['fresh_model_fits_this_run'],
            'outer_vectors_bit_identical': outer_same, 'inner_vectors_bit_identical': inner_same,
            'outer_audit_states_identical': state_same, 'inner_audit_states_identical': inner_same,
            'threshold_records_identical': threshold_same, 'scope': 'Seven outer and six inner fresh fits; not a full repeat of all histories.'}
    old_path = report/'verification.json'
    if repeat_report is None and old_path.exists():
        old = json.loads(old_path.read_text(encoding='utf-8'))
        if (old.get('freeze_signature') == frozen['signature'] and
                old.get('original_vector_set_sha256') == verification['original_vector_set_sha256'] and 'representative_refit' in old):
            verification['representative_refit'] = old['representative_refit']
    write_json(old_path, verification)
    return verification


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repeat-report', type=Path)
    args = p.parse_args()
    result = verify_results(args.repeat_report)
    print(json.dumps({k: v for k, v in result.items() if k not in ['threshold_checks', 'paired_history_diagnostics']}, indent=2))


if __name__ == '__main__': main()
