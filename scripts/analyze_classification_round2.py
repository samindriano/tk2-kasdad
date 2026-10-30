"""Post-fit accounting and threshold-transfer diagnostics; no new fitting/selection."""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.special import expit

from src.data.audit import write_json
from src.models.classification_round2_evaluation import verify_freeze, load_protocol, _routes, REPORT
from src.models.classification_round2_models import stream_missing_mask
from src.models.round1_evaluation import array_digest
from src.utils.paths import PROJECT_ROOT, task_files


def analyze_round2(data_dir=None):
    frozen = verify_freeze(data_dir); protocol = load_protocol()
    result = json.loads((REPORT/'results.json').read_text(encoding='utf-8'))
    if result['status'] != 'CLASSIFICATION_ROUND2_COMPLETE_REVIEW_NO_FINAL_MODEL' or result['freeze_signature'] != frozen['signature']:
        raise ValueError('Completed current frozen Round2 required.')
    frame = pd.read_csv(task_files('classification', data_dir)['train'])
    artifact = PROJECT_ROOT/'experiments/runs'/result['run_id']
    groups, audits, components = {}, {}, {}
    for end in result['unique_outer_histories']:
        groups[end] = {}; audits[end] = {}
        for model in protocol['fit_recipes']:
            tag = f'fit{end}__{model}'
            meta = json.loads((artifact/f'{tag}.json').read_text(encoding='utf-8'))
            with np.load(artifact/f'{tag}.npz', allow_pickle=False) as a:
                group = {'rows': a['outer_rows'].copy(), 'probability': a['probability'].copy(),
                         'inner_rows': a['inner_rows'].copy(), 'inner_probability': a['inner_probability'].copy(),
                         'thresholds': meta['thresholds'], 'meta': meta}
                parts = {key.removeprefix('logit__'): a[key].copy() for key in a.files if key.startswith('logit__')}
            if array_digest(group['probability']) != meta['probability_sha256']:
                raise ValueError('Probability cache changed.')
            if {key: array_digest(value) for key, value in parts.items()} != meta['component_sha256']:
                raise ValueError('Component cache changed.')
            groups[end][model] = group; audits[end][model] = meta; components[(end, model)] = parts
    _routes(frame, groups, protocol)
    pairs = [('C3_B', 'C1_A'), ('C3_C', 'C2_B'), ('C2_B', 'C3_B'), ('C5_LOW', 'C2_B'),
             ('C5_HIGH', 'C2_B'), ('C4_LR', 'C2_B'), ('C4_HGB', 'C2_B'), ('C4_HGB', 'C4_LR')]
    comparisons = []
    for family in ['primary', 'secondary']:
        for policy in protocol['threshold_policies']:
            for candidate, control in pairs:
                a = result['aggregate'][family][policy][candidate]; b = result['aggregate'][family][policy][control]
                aj = {j['window']: j['policies'][policy] for j in result['jobs'] if j['family'] == family and j['model'] == candidate}
                bj = {j['window']: j['policies'][policy] for j in result['jobs'] if j['family'] == family and j['model'] == control}
                cycle_deltas = [x['macro_f1']-y['macro_f1'] for w in aj for x, y in zip(aj[w]['per_cycle'], bj[w]['per_cycle'])]
                window_deltas = {w: aj[w]['metrics']['macro_f1']-bj[w]['metrics']['macro_f1'] for w in aj}
                comparisons.append({'family': family, 'policy': policy, 'candidate': candidate, 'control': control,
                    'pooled_f1_delta': a['metrics']['macro_f1']-b['metrics']['macro_f1'],
                    'equal_cycle_f1_delta': a['equal_cycle_macro_f1']-b['equal_cycle_macro_f1'],
                    'window_deltas': window_deltas, 'windows_improved': sum(d > 1e-12 for d in window_deltas.values()),
                    'cycles_improved': sum(d > 1e-12 for d in cycle_deltas), 'cycles_tied': sum(abs(d) <= 1e-12 for d in cycle_deltas),
                    'subgroup_f1_deltas': {s: a['subgroups'][s]['macro_f1']-b['subgroups'][s]['macro_f1']
                        for s in ['EF0', 'EF1', 'stream_missing', 'stream_present', 'all_codes_known_nonmissing', 'any_unseen_code']},
                    'false_positive_delta': a['metrics']['false_positive']-b['metrics']['false_positive'],
                    'false_negative_delta': a['metrics']['false_negative']-b['metrics']['false_negative']})
    accounting = []
    for end in result['unique_outer_histories']:
        rows = groups[end]['C2_B']['rows']; missing = stream_missing_mask(frame.iloc[rows])
        for candidate, control in pairs[:5]:
            a, b = groups[end][candidate], groups[end][control]
            ca, cb = components[(end, candidate)], components[(end, control)]
            if not np.array_equal(a['rows'], b['rows']): raise ValueError('Accounting rows do not align.')
            delta_parts = {key: ca[key]-cb[key] for key in ca}
            delta = sum(delta_parts.values())
            la, lb = sum(ca.values()), sum(cb.values())
            if not np.allclose(delta, la-lb, atol=1e-12, rtol=1e-12): raise ValueError('Additive delta does not reconcile.')
            if np.max(np.abs(expit(la)-a['probability'])) > 1e-12: raise ValueError('Candidate component probability differs.')
            coefficients_a = audits[end][candidate]['coefficients']; coefficients_b = audits[end][control]['coefficients']
            numeric_names = sorted(set(coefficients_a['numeric_coefficients']) & set(coefficients_b['numeric_coefficients']))
            numeric_delta_l2 = float(np.linalg.norm([coefficients_a['numeric_coefficients'][name]-coefficients_b['numeric_coefficients'][name] for name in numeric_names]))
            accounting.append({'fit_end': end, 'prediction_cycles': [int(frame.iloc[rows].cycle_code.min()), int(frame.iloc[rows].cycle_code.max())],
                'candidate': candidate, 'control': control, 'n_missing': int(missing.sum()),
                'same_structured_transform': coefficients_a['structured_transform_sha256'] == coefficients_b['structured_transform_sha256'],
                'numeric_coefficient_delta_l2': numeric_delta_l2,
                'missing_logit_mean_delta': float(delta[missing].mean()),
                'missing_component_mean_deltas': {key: float(value[missing].mean()) for key, value in delta_parts.items()},
                'missing_component_mean_abs_deltas': {key: float(np.abs(value[missing]).mean()) for key, value in delta_parts.items()},
                'missing_probability_mean_delta': float((a['probability'][missing]-b['probability'][missing]).mean()),
                'missing_probability_mean_absolute_change': float(np.abs(a['probability'][missing]-b['probability'][missing]).mean()),
                'missing_fixed_label_changes': int(((a['probability'][missing] >= .5) != (b['probability'][missing] >= .5)).sum()),
                'candidate_intercept': coefficients_a['intercept'], 'control_intercept': coefficients_b['intercept'],
                'candidate_indicator_coefficient': coefficients_a['stream_missing_indicator_coefficient'],
                'control_indicator_coefficient': coefficients_b['stream_missing_indicator_coefficient'],
                'text_delta_exactly_zero_on_missing': bool((delta_parts['text'][missing] == 0).all())})
    transfer = []
    for job in result['jobs']:
        end, model = job['fit_group'], job['model']; group = groups[end][model]
        inner = frame.iloc[group['inner_rows']]
        mask = frame.iloc[group['rows']].cycle_code.between(*job['validation_cycles']).to_numpy()
        p = group['probability'][mask]
        for policy in protocol['threshold_policies']:
            a = job['policies'][policy]['metrics']; baseline = job['policies']['fixed_0.5']['metrics']
            cut = job['thresholds'][policy]
            transfer.append({'model': model, 'window': job['window'], 'family': job['family'], 'fit_end': end, 'policy': policy,
                'threshold': cut['threshold'], 'inner_prevalence': float(inner.artifact_signal.mean()),
                'outer_prevalence': a['positive_rate'], 'inner_mean_probability': float(group['inner_probability'].mean()),
                'outer_mean_probability': a['mean_probability'], 'inner_probability_quantiles': np.quantile(group['inner_probability'], [.1, .5, .9]).tolist(),
                'outer_probability_quantiles': np.quantile(p, [.1, .5, .9]).tolist(),
                'positive_prediction_rate': a['positive_prediction_rate'], 'positive_allocation_minus_prevalence': a['positive_prediction_rate']-a['positive_rate'],
                'outer_f1': a['macro_f1'], 'delta_f1_vs_fixed': a['macro_f1']-baseline['macro_f1'],
                'inner_f1_at_transferred_cut': cut['inner_pooled_f1'], 'outer_average_precision': a['average_precision'],
                'outer_brier': a['brier_score'], 'class1_precision': a['per_class'][1]['precision'], 'class1_recall': a['per_class'][1]['recall'],
                'false_positive': a['false_positive'], 'false_negative': a['false_negative']})
    output = {'status': 'POSTFIT_ACCOUNTING_NO_NEW_MODELS_OR_POLICY_SELECTION', 'freeze_signature': frozen['signature'],
              'comparisons': comparisons, 'within_history_missing_logit_accounting': accounting, 'threshold_transfer': transfer,
              'limits': 'Outer labels used only for development diagnostics; coefficient accounting not causal; across-history feature coordinates and prediction ranges differ.'}
    write_json(REPORT/'behavior_diagnostics.json', output)
    return output


if __name__ == '__main__':
    r = analyze_round2()
    print(r['status'], len(r['comparisons']), 'comparisons', len(r['within_history_missing_logit_accounting']), 'component contrasts')
