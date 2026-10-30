"""Saved-vector diagnostics and figures for Regression Round2, without refitting."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ks_2samp

from src.data.audit import write_json
from src.models.regression_round2_evaluation import (
    REPORT, DEFAULT_RUN, load_protocol, verify_freeze, load_interaction_selection,
    safe_artifact, positions_digest,
)
from src.models.round1_evaluation import array_digest
from src.models.regression_round2_metrics import metric_values, paired, descriptive_test_risk
from src.models.round1_models import CATEGORICAL_FEATURES, _category_array, _is_missing, _number_array, _mixed_count_arrays
from src.utils.paths import PROJECT_ROOT, task_files

COLORS = {'R1_C': '#8b701d', 'R1_D': '#245e8a', 'A_NO_CYCLE': '#ad592f',
          'A_NO_CODES': '#727c32', 'A_NO_SCOPE_SPREAD': '#a64c7c'}


def load_results():
    frozen = verify_freeze()
    result = json.loads((REPORT/'results.json').read_text(encoding='utf-8'))
    if result['freeze_signature'] != frozen['signature'] or result['partial']:
        raise RuntimeError('Expected complete frozen Round2 results.')
    selection = load_interaction_selection()
    if result.get('stage') != 'all' or result.get('interaction_signature') != selection['signature']:
        raise RuntimeError('Results do not match the complete frozen interaction stage.')
    return result


def load_vector(end, model, result=None):
    result = result or load_results()
    artifact = safe_artifact(result['run_id'])
    meta = json.loads((artifact/f'fit{end}__{model}.json').read_text(encoding='utf-8'))
    with np.load(artifact/f'fit{end}__{model}.npz', allow_pickle=False) as saved:
        fit, rows, pred = (saved[k].copy() for k in ['fit_rows', 'validation_rows', 'prediction'])
    if meta['prediction_sha256'] != array_digest(pred) or meta['fit_positions_sha256'] != positions_digest(fit) or meta['validation_positions_sha256'] != positions_digest(rows):
        raise RuntimeError('Saved vector identity mismatch.')
    return fit, rows, pred


def family_vector(model, family='primary', result=None, frame=None):
    result = result or load_results()
    if frame is None:
        frame = pd.read_csv(task_files('regression')['train'])
    protocol = load_protocol()
    rows_all, preds_all = [], []
    for w in protocol['primary_windows' if family == 'primary' else 'secondary_windows']:
        _, rows, pred = load_vector(w['fit'][1], model, result)
        mask = frame.iloc[rows].cycle_code.between(*w['validation']).to_numpy()
        rows_all.append(rows[mask]); preds_all.append(pred[mask])
    rows, pred = np.concatenate(rows_all), np.concatenate(preds_all)
    order = np.argsort(rows)
    rows, pred = rows[order], pred[order]
    if len(np.unique(rows)) != len(rows):
        raise RuntimeError('Repeated validation rows inside family.')
    return rows, pred


def comparison_frame(result=None):
    r = result or load_results()
    out = []
    for recipe in r['recipes']:
        m = recipe['id']
        a, b = r['aggregate']['primary'][m], r['aggregate']['secondary'][m]
        jobs = {j['window']: j for j in r['jobs'] if j['model'] == m}
        out.append({'Recipe': m, 'Features': recipe['features'], 'History': recipe['history'],
                    'Primary MAE': a['raw']['mae'], 'Equal-cycle MAE': a['equal_cycle_mae'],
                    'H12 MAE': b['raw']['mae'], 'H12_E': jobs['H12_E']['raw']['mae'],
                    'H12_L': jobs['H12_L']['raw']['mae'], 'RMSE': a['raw']['rmse'],
                    'Signed error': a['raw']['signed_error'],
                    'EF0': a['subgroups']['EF0']['mae'], 'EF1': a['subgroups']['EF1']['mae'],
                    'Low tail': a['subgroups']['tail_low_le5']['mae'],
                    'High tail': a['subgroups']['tail_high_ge95']['mae']})
    return pd.DataFrame(out)


def _table(frame):
    rows = []
    for row in frame.itertuples(index=False, name=None):
        rows.append('| '+' | '.join(f'{x:.6f}' if isinstance(x, (float, np.floating)) else str(x) for x in row)+' |')
    return '| '+' | '.join(map(str, frame.columns))+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join(rows)+'\n'


def analyze():
    r = load_results()
    frame = pd.read_csv(task_files('regression')['train'])
    protocol = load_protocol()
    # Selection of experiments is complete before reading official test features.
    test = pd.read_csv(task_files('regression')['test'])
    risk = descriptive_test_risk(frame, test)
    for col in ['impact_load', 'entry_friction', 'scope_pressure', 'text_load', 'scope_trace', 'org_spread', 'module_spread']:
        parser = _mixed_count_arrays if col in {'scope_trace', 'org_spread', 'module_spread'} else _number_array
        a, b = parser(frame[col])[0], parser(test[col])[0]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        risk['features'][col]['descriptive_ks_distance'] = float(ks_2samp(a, b).statistic)
    # New feature-support pattern, examined after all recipes were selected.
    pressure_floor = float(pd.to_numeric(frame.scope_pressure).min())
    cohorts = {}
    for label, source in [('train', frame), ('test', test)]:
        zero = np.column_stack([_mixed_count_arrays(source[name])[0] == 0 for name in ['scope_trace', 'org_spread', 'module_spread']])
        below = pd.to_numeric(source.scope_pressure).to_numpy() < pressure_floor
        cohorts[label] = {'n': len(source), 'any_scope_count_zero': int(zero.any(axis=1).sum()),
                          'all_three_scope_counts_zero': int(zero.all(axis=1).sum()),
                          'pressure_below_train_min': int(below.sum()),
                          'any_scope_zero_and_low_pressure': int((zero.any(axis=1) & below).sum())}
    risk['joint_zero_scope_cohorts'] = cohorts
    risk['joint_zero_scope_interpretation'] = 'Observed novel numeric support only; genuine scope versus representation/source change is unresolved. No target or model selection used.'
    out = {'freeze_signature': r['freeze_signature'], 'interaction_signature': r['interaction_signature'],
           'official_test_feature_risk': risk, 'target_bins': {}, 'conditional_groups': {},
           'known_unknown': {}, 'horizon_profiles': [], 'paired_against_D': {}, 'interactions': [],
           'rank_quality': {}}
    for family in ['primary', 'secondary']:
        out['target_bins'][family], out['conditional_groups'][family], out['known_unknown'][family] = {}, {}, {}
        out['rank_quality'][family] = {}
        for recipe in r['recipes']:
            m = recipe['id']
            rows, pred = family_vector(m, family, r, frame)
            part, y = frame.iloc[rows], frame.iloc[rows].threat_rank.to_numpy()
            rank_cycles = []
            for cycle in sorted(part.cycle_code.unique()):
                mask = part.cycle_code.eq(cycle).to_numpy()
                correlation = float(spearmanr(y[mask], pred[mask]).statistic) if np.std(pred[mask]) > 0 else None
                rank_cycles.append({'cycle': int(cycle), 'n': int(mask.sum()), 'spearman': correlation})
            out['rank_quality'][family][m] = {
                'pooled_spearman': float(spearmanr(y, pred).statistic),
                'equal_cycle_spearman': float(np.mean([v['spearman'] for v in rank_cycles if v['spearman'] is not None])),
                'per_cycle': rank_cycles,
                'interpretation': 'Postfit ranking diagnostic only; no prediction transformation or cutoff selection.'}
            bins = []
            for lo in range(0, 100, 10):
                mask = (y >= lo) & ((y < lo+10) if lo < 90 else (y <= 100))
                bins.append({'target_low': lo, 'target_high': lo+10, **metric_values(y[mask], pred[mask])})
            out['target_bins'][family][m] = bins
            groups = []
            for impact in sorted(part.impact_load.unique()):
                mask = part.impact_load.eq(impact).to_numpy()
                if mask.sum() < 100:
                    continue
                correlation = float(spearmanr(y[mask], pred[mask]).statistic) if np.std(pred[mask]) > 0 else None
                groups.append({'impact_load': float(impact), 'n': int(mask.sum()),
                               'target_iqr': float(np.diff(np.quantile(y[mask], [.25, .75]))[0]),
                               'prediction_iqr': float(np.diff(np.quantile(pred[mask], [.25, .75]))[0]),
                               'target_median': float(np.median(y[mask])), 'prediction_median': float(np.median(pred[mask])),
                               'spearman': correlation,
                               **metric_values(y[mask], pred[mask])})
            out['conditional_groups'][family][m] = groups
            # Unknown category masks use each model's selected past rows, even when codes are ablated.
            masks = {'all_codes_known_nonmissing': [], 'any_code_unknown': [], 'any_code_missing': []}
            mask_rows = []
            for w in protocol['primary_windows' if family == 'primary' else 'secondary_windows']:
                fit, val, _ = load_vector(w['fit'][1], m, r)
                val = val[frame.iloc[val].cycle_code.between(*w['validation']).to_numpy()]
                unknown, missing = np.zeros(len(val), bool), np.zeros(len(val), bool)
                for name in CATEGORICAL_FEATURES:
                    seen = {str(v) for v in _category_array(frame.iloc[fit][name]) if not _is_missing(v)}
                    values = _category_array(frame.iloc[val][name])
                    miss = np.array([_is_missing(v) for v in values])
                    unknown |= np.array([not _is_missing(v) and str(v) not in seen for v in values])
                    missing |= miss
                for name, mask in [('all_codes_known_nonmissing', ~unknown & ~missing), ('any_code_unknown', unknown), ('any_code_missing', missing)]:
                    masks[name].append(mask)
                mask_rows.append(val)
            mask_order = np.argsort(np.concatenate(mask_rows))
            out['known_unknown'][family][m] = {name: metric_values(y[np.concatenate(parts)[mask_order]], pred[np.concatenate(parts)[mask_order]]) for name, parts in masks.items()}
            _, ref = family_vector('R1_D', family, r, frame)
            out['paired_against_D'][family+'__'+m] = paired(part, pred, ref)
    for end in [55, 67]:
        for recipe in r['recipes']:
            m = recipe['id']
            _, rows, pred = load_vector(end, m, r)
            for cycle in sorted(frame.iloc[rows].cycle_code.unique()):
                mask = frame.iloc[rows].cycle_code.eq(cycle).to_numpy()
                out['horizon_profiles'].append({'fit_end': end, 'recipe': m, 'lead': int(cycle-end), 'cycle': int(cycle),
                                                **metric_values(frame.iloc[rows[mask]].threat_rank, pred[mask])})
    for recipe in r['recipes']:
        if recipe['stage'] != 'interaction':
            continue
        for family in ['primary', 'secondary']:
            a = r['aggregate'][family]
            base_ablation = 'A_'+recipe['features'].upper()
            full_recent = 'B_D_'+recipe['history'].upper()
            delta_recent = a[recipe['id']]['raw']['mae']-a[full_recent]['raw']['mae']
            delta_expanding = a[base_ablation]['raw']['mae']-a['R1_D']['raw']['mae']
            out['interactions'].append({'recipe': recipe['id'], 'family': family,
                                        'ablation_delta_same_history': delta_recent,
                                        'ablation_delta_expanding': delta_expanding,
                                        'difference_in_differences': delta_recent-delta_expanding,
                                        'interpretation': 'Predictive refit interaction, not causal effect.'})
    write_json(REPORT/'behavior_diagnostics.json', out)
    return out


def write_tables(r=None, d=None):
    r, d = r or load_results(), d or json.loads((REPORT/'behavior_diagnostics.json').read_text(encoding='utf-8'))
    text = ['# Regression Round2 — actual saved-vector comparisons\n',
            'Raw MAE lower is better. Primary and H12 reuse the same 38,588 records; do not pool them as independent evidence.\n',
            '## All recipes\n', _table(comparison_frame(r))]
    for family in ['primary', 'secondary']:
        jobs = [j for j in r['jobs'] if j['family'] == family]
        windows = list(dict.fromkeys(j['window'] for j in jobs))
        text += ['## '+family+' windows\n', _table(pd.DataFrame([{'Recipe': m, **{w: next(j['raw']['mae'] for j in jobs if j['model'] == m and j['window'] == w) for w in windows}} for m in r['aggregate'][family]]))]
        text += ['## '+family+' cycle MAE\n', _table(pd.DataFrame([{'Recipe': m, **{str(v['cycle']): v['mae'] for v in a['per_cycle']}} for m, a in r['aggregate'][family].items()]))]
        subgroup_rows = []
        for m, a in r['aggregate'][family].items():
            for group, stats in {**a['subgroups'], **d['known_unknown'][family][m]}.items():
                subgroup_rows.append({'Recipe': m, 'Group': group, **{k: stats[k] for k in ['n', 'mae', 'rmse', 'signed_error']}})
        text += ['## '+family+' subgroups (unknown/missing overlap)\n',
                 'Category novelty describes observed data cohorts relative to each fit subset, including recipes that omit category predictors. It does not imply encoder failure.\n',
                 _table(pd.DataFrame(subgroup_rows))]
    history_rows = []
    for key, h in r['history_support'].items():
        history_rows.append({'Fit/history': key, 'n': h['fit_rows'], 'Cycle counts': json.dumps(h['fit_cycle_counts']),
                             'Unknown org': h['category_support']['org_code']['validation_unknown_n'],
                             'Unknown module': h['category_support']['module_code']['validation_unknown_n']})
    text += ['## History composition\n', _table(pd.DataFrame(history_rows))]
    runtime = []
    for m in [v['id'] for v in r['recipes']]:
        states = [json.loads((REPORT/'fit_audits'/f'fit{end}.json').read_text(encoding='utf-8'))[m] for end in [55, 59, 63, 67, 71, 75]]
        runtime.append({'Recipe': m, 'Sum six fit/predict seconds': sum(s['fit_seconds'] for s in states),
                        'Mean transformed columns': np.mean([len(s['fitted_state']['transformed_feature_names']) for s in states]),
                        'Mean nodes': np.mean([s['complexity']['total_nodes'] for s in states]),
                        'Trees per fit': states[0]['complexity']['trees_fitted']})
    text += ['## Runtime and complexity\n', 'Aliased histories share fits; row sums are attributed recipe cost and must not be summed as actual wall time.\n', _table(pd.DataFrame(runtime))]
    (REPORT/'COMPARISON_TABLES.md').write_text('\n'.join(text), encoding='utf-8')


def make_figures(r=None, d=None):
    r = r or load_results()
    d = d or json.loads((REPORT/'behavior_diagnostics.json').read_text(encoding='utf-8'))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.titleweight': 'bold', 'figure.facecolor': 'white'})
    folder = REPORT/'figures'; folder.mkdir(exist_ok=True)
    paths = []

    def save(fig, name):
        fig.savefig(folder/(name+'.png'), dpi=145, bbox_inches='tight')
        paths.append(str((folder/(name+'.png')).relative_to(PROJECT_ROOT)))
        plt.close(fig)

    groups = [v['id'] for v in r['recipes'] if v['id'].startswith('A_')]
    wins = [w['id'] for w in load_protocol()['primary_windows']+load_protocol()['secondary_windows']]
    data = np.array([[next(j['raw']['mae'] for j in r['jobs'] if j['model'] == m and j['window'] == w)-next(j['raw']['mae'] for j in r['jobs'] if j['model'] == 'R1_D' and j['window'] == w) for w in wins] for m in groups])
    fig, ax = plt.subplots(figsize=(11.5, 4.5))
    bound = max(abs(data.min()), abs(data.max()))
    im = ax.imshow(data, cmap='PuOr_r', vmin=-bound, vmax=bound, aspect='auto')
    ax.set_xticks(range(len(wins)), wins); ax.set_yticks(range(len(groups)), groups)
    for y in range(len(groups)):
        for x in range(len(wins)):
            ax.text(x, y, f'{data[y,x]:+.2f}', ha='center', va='center', color='black' if abs(data[y,x]) < .65*bound else 'white')
    ax.set_title('Feature-group removal: MAE change versus full R1-D\nSame expanding history; negative delta improves MAE')
    fig.colorbar(im, ax=ax, label='MAE difference (points)'); fig.tight_layout(); save(fig, 'regression_round2_ablation_window_deltas')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), sharey=True)
    labels = ['expanding', 'recent8', 'recent12', 'matched8_s42', 'matched8_s31415']
    for ax, family in zip(axes, ['primary', 'secondary']):
        for source, color, marker in [('C', COLORS['R1_C'], 'o'), ('D', COLORS['R1_D'], 's')]:
            ids = ['R1_'+source]+['B_'+source+'_'+h.upper() for h in labels[1:]]
            ax.plot(range(5), [r['aggregate'][family][m]['raw']['mae'] for m in ids], marker=marker, color=color, label='R1-'+source+' feature set')
        ax.set_xticks(range(5), ['Expand', 'Recent 8', 'Recent 12', 'Match 8\nseed 42', 'Match 8\nseed 31415'])
        ax.set_title('Primary: six 4-cycle windows' if family == 'primary' else 'H12: two 12-cycle windows')
        ax.set_ylabel('Row-pooled MAE (lower is better)'); ax.grid(axis='y', alpha=.18); ax.legend()
    fig.suptitle('Training history sensitivity — fixed family, loss and parameters', fontweight='bold'); fig.tight_layout(); save(fig, 'regression_round2_history_comparison')

    table = comparison_frame(r)
    selected = list(dict.fromkeys(['R1_C', 'R1_D', table.loc[table['Primary MAE'].idxmin(), 'Recipe'], table.loc[table['H12 MAE'].idxmin(), 'Recipe']]))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    palette = ['#8b701d', '#245e8a', '#ad592f', '#727c32']
    styles, markers = ['-', '--', '-.', ':'], ['o', 's', '^', 'D']
    for m, color, style, marker in zip(selected, palette, styles, markers):
        bins = d['target_bins']['primary'][m]
        x = [b['target_low']+5 for b in bins]
        axes[0].plot(x, [b['mae'] for b in bins], marker=marker, linestyle=style, label=m, color=color)
        axes[1].plot(x, [b['signed_error'] for b in bins], marker=marker, linestyle=style, label=m, color=color)
    for ax in axes:
        ax.set_xlabel('Fixed 10-point actual target bin (midpoint)'); ax.grid(alpha=.18)
    axes[0].set_ylabel('MAE'); axes[1].set_ylabel('Mean prediction − actual'); axes[1].axhline(0, color='#555', linewidth=.8)
    axes[0].set_title('Conditional absolute error'); axes[1].set_title('Conditional signed error'); axes[0].legend(fontsize=8)
    fig.suptitle('Primary residuals by actual target region — descriptive, not a gate', fontweight='bold'); fig.tight_layout(); save(fig, 'regression_round2_target_region_residuals')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), sharey=True)
    for ax, end in zip(axes, [55, 67]):
        for m, color, style, marker in zip(selected, palette, styles, markers):
            points = [v for v in d['horizon_profiles'] if v['fit_end'] == end and v['recipe'] == m]
            ax.plot([p['lead'] for p in points], [p['mae'] for p in points], marker=marker, linestyle=style, color=color, label=m)
        ax.set_title(f'Fit cutoff {end}: validation {end+1}–{end+12}')
        ax.set_xticks(range(1, 13)); ax.set_xlabel('Cycles ahead of fit cutoff'); ax.grid(alpha=.18)
    axes[0].set_ylabel('Cycle MAE'); axes[0].legend(fontsize=8)
    fig.suptitle('H12 error profiles: horizon and era change together', fontweight='bold'); fig.tight_layout(); save(fig, 'regression_round2_horizon_profiles')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    for m, color, style, marker in zip(selected, palette, styles, markers):
        stats = r['aggregate']['primary'][m]
        axes[0].plot([v['cycle'] for v in stats['per_cycle']], [v['mae'] for v in stats['per_cycle']], marker=marker, markersize=4, linestyle=style, label=m, color=color)
        axes[1].plot([0, 1], [stats['subgroups'][g]['mae'] for g in ['tail_low_le5', 'tail_high_ge95']], marker=marker, linestyle=style, label=m, color=color)
    axes[0].set_xlabel('Validation cycle'); axes[0].set_ylabel('MAE'); axes[0].set_xticks([56, 60, 64, 68, 72, 76, 79]); axes[0].set_title('Primary per-cycle MAE'); axes[0].legend(fontsize=8)
    reference = r['aggregate']['primary']['R1_D']['subgroups']
    axes[1].set_xticks([0, 1], [f"Actual ≤ 5\nn={reference['tail_low_le5']['n']:,}", f"Actual ≥ 95\nn={reference['tail_high_ge95']['n']:,}"])
    axes[1].set_ylabel('Tail MAE'); axes[1].set_title('Fixed target tails; primary validation')
    for ax in axes: ax.grid(alpha=.18)
    fig.tight_layout(); save(fig, 'regression_round2_cycle_and_tail_errors')

    interactions = pd.DataFrame(d['interactions'])
    if not interactions.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
        for ax, family in zip(axes, ['primary', 'secondary']):
            part = interactions[interactions.family == family]
            y = np.arange(len(part)); ax.barh(y-.15, part.ablation_delta_expanding, height=.28, color='#245e8a', label='Expanding')
            ax.barh(y+.15, part.ablation_delta_same_history, height=.28, facecolor='white', edgecolor='#ad592f', hatch='//', label='Same recent/matched history')
            ax.set_yticks(y, part.recipe.str.replace('C_NO_', '', regex=False)); ax.axvline(0, color='#555', linewidth=.7)
            ax.set_xlabel('Removal − full model MAE (negative helps)'); ax.set_title(family); ax.legend(fontsize=8)
        fig.suptitle('Adaptive feature × history checks — predictive contrasts only', fontweight='bold'); fig.tight_layout(); save(fig, 'regression_round2_feature_history_interactions')
    return paths


def build_report():
    r = load_results(); d = analyze(); write_tables(r, d)
    paths = make_figures(r, d)
    write_json(REPORT/'report_manifest.json', {'freeze_signature': r['freeze_signature'], 'figures': paths,
                                             'source': 'Verified local validation vectors and frozen metrics; no research test predictions.'})
    return paths
