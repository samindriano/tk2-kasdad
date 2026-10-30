"""Integrate descriptive EDA evidence; no estimator or prediction is created."""
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.audit import environment, write_json
from src.data.composition_eda import run_composition_eda
from src.data.temporal_eda import run_temporal_eda
from src.utils.paths import PROJECT_ROOT, task_files

QUESTION = ('Apakah hubungan antara impact_load dan threat_rank mengalami perubahan '
            'antarperiode cycle_code, meskipun distribusi threat_rank relatif stabil?')


def _illustrative_distributions(composition, out):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True, layout='constrained')
    for ax, profile, color in zip(axes, composition['headline_exact_impact_profiles'], ['#2864a5','#cf8e30']):
        rows = list(profile['periods'].values())
        med = np.array([r['median'] for r in rows])
        low, high = np.array([r['p10'] for r in rows]), np.array([r['p90'] for r in rows])
        ef = max(rows[0]['entry_friction_counts'], key=rows[0]['entry_friction_counts'].get)
        ax.errorbar(range(3), med, yerr=[med-low,high-med], marker='o', color=color, capsize=4)
        ax.set_xticks(range(3), [f"{p['cycle_min']}-{p['cycle_max']}\nn={r['n']:,}" for p,r in zip(composition['periods'],rows)])
        ax.set(title=f"impact={profile['impact_load']:.4f}, EF={ef}", xlabel='cycle period', ylim=(0,100))
        ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('threat_rank median with P10-P90 spread')
    fig.suptitle('Two illustrative groups: same impact and EF, different distribution changes')
    name = 'formal_selected_conditional.png'
    fig.savefig(out/name, dpi=150)
    plt.close(fig)
    return name


def run_formal_eda(data_dir=None, output_dir=None, make_plots=True):
    out = Path(output_dir) if output_dir else PROJECT_ROOT/'reports/eda'
    temporal = run_temporal_eda(data_dir, out, make_plots)
    composition = run_composition_eda(data_dir, out, make_plots)
    frame = pd.read_csv(task_files('regression', data_dir)['train'])
    cycle_blocks = np.array_split(sorted(frame.cycle_code.unique()), 3)
    labels = {cycle: f'period_{i+1}' for i, block in enumerate(cycle_blocks) for cycle in block}
    frame['period'] = frame.cycle_code.map(labels)
    missingness = []
    for column in ('signal_stream', 'org_code', 'module_code'):
        for period, group in frame.groupby('period', observed=True):
            for missing in (False, True):
                sub = group[group[column].isna().eq(missing)]
                missingness.append({'column': column, 'period': period, 'missing': missing,
                    'n': len(sub), 'share_period': len(sub)/len(group),
                    'pearson': float(sub.impact_load.corr(sub.threat_rank)) if len(sub)>1 else None,
                    'spearman': float(sub.impact_load.corr(sub.threat_rank, method='spearman')) if len(sub)>1 else None})
    keys = {(d['impact_load'], d['entry_friction']) for d in
            composition['support_selection']['common_impact_x_friction_cells']}
    kept = pd.Series([(x,e) in keys for x,e in zip(frame.impact_load,frame.entry_friction)], index=frame.index)
    coverage = []
    for retained in (True, False):
        y = frame.loc[kept.eq(retained), 'threat_rank']
        coverage.append({'retained_common_support': retained, 'n': len(y),
                         'low_tail_count_le5': int(y.le(5).sum()), 'high_tail_count_ge95': int(y.ge(95).sum()),
                         'mean': float(y.mean()), 'median': float(y.median())})
    # Rank geometry is diagnostic only. The tie bound is an algebraic implication
    # conditional on the observed midrank identity, with 4dp rounding error.
    rank_implications = []
    for cycle, group in frame.groupby('cycle_code'):
        n = len(group)
        largest_tie = int(group.threat_rank.value_counts().max())
        rank_implications.append({'cycle_code': int(cycle), 'n': n,
            'rank_implied_mean': 50+50/n,
            'observed_mean_minus_implied': float(group.threat_rank.mean()-(50+50/n)),
            'uniform_ks_implied_by_largest_tie': (largest_tie+1)/(2*n)})
    summary = {'status': 'SAMUEL_EDA_MILESTONE_COMPLETE_REVIEW_REQUIRED_BEFORE_MODELING',
        'research_question': QUESTION, 'question_selected_by_user': True,
        'environment': environment(), 'train_sha256': temporal['input_manifest']['train']['sha256'],
        'evidence_files': ['temporal_robustness.json', 'composition_robustness.json'],
        'context': {'missingness_strata': missingness, 'common_support_tail_coverage': coverage,
                    'rank_geometry_implications': rank_implications},
        'guardrails': ['No modeling, tuning, new prediction or submission.',
            'Rank diagnostic uses target against itself, not a deployable feature.',
            'Group standardization and mean decomposition are descriptive, not causal adjustment.',
            'Stable rank marginals do not establish stable conditional relationships or future performance.']}
    summary['illustrative_figure'] = _illustrative_distributions(composition, out) if make_plots else None
    write_json(out/'formal_eda_summary.json', summary)
    return {'temporal': temporal, 'composition': composition, 'summary': summary}
