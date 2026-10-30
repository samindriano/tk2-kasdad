"""Descriptive temporal robustness checks; no fitting, predictions, or test labels."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from src.utils.paths import PROJECT_ROOT, data_root, task_files

SEED = 42
BOOTSTRAP_REPLICATES = 2_000


def _safe(x):
    return float(x) if x is not None and np.isfinite(x) else None


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _corr(frame):
    p = frame[['impact_load', 'threat_rank']].replace([np.inf, -np.inf], np.nan).dropna()
    if len(p) < 2 or p.impact_load.nunique() < 2 or p.threat_rank.nunique() < 2:
        return {'n': int(len(p)), 'pearson': None, 'spearman': None}
    return {'n': int(len(p)),
            'pearson': _safe(pearsonr(p.impact_load, p.threat_rank).statistic),
            'spearman': _safe(spearmanr(p.impact_load, p.threat_rank).statistic)}


def _blocks(cycles, k):
    return [chunk for chunk in np.array_split(np.sort(np.unique(cycles)), k) if len(chunk)]


def _stats(df):
    # These six sufficient statistics preserve the row-weighted Pearson coefficient.
    x = df.impact_load.to_numpy(float)
    y = df.threat_rank.to_numpy(float)
    return np.asarray([len(x), x.sum(), y.sum(), np.dot(x, x), np.dot(y, y), np.dot(x, y)], float)


def _corr_from_sums(z):
    n, sx, sy, sxx, syy, sxy = np.moveaxis(np.asarray(z), -1, 0)
    vx = sxx - sx*sx/n
    vy = syy - sy*sy/n
    cov = sxy - sx*sy/n
    denominator = np.sqrt(np.maximum(vx, 0)*np.maximum(vy, 0))
    out = np.full(np.shape(denominator), np.nan, dtype=float)
    np.divide(cov, denominator, out=out, where=denominator > 0)
    return out


def _cycle_cluster_sensitivity(early_cycles, late_cycles, observed_delta,
                               reps=BOOTSTRAP_REPLICATES, seed=SEED):
    """Resample cycle rows as clusters within first/last 11-cycle blocks.

    The range is exploratory cycle-cluster resampling conditional on exchangeability,
    not a confidence interval robust to serial dependence or a claim about future cycles.
    """
    rng = np.random.default_rng(seed)
    if len(early_cycles) != 11 or len(late_cycles) != 11:
        raise ValueError('Endpoint cycle-cluster resampling requires eleven cycles per block.')
    group_sums = [np.stack([_stats(g) for g in cycles])
                  for cycles in (early_cycles, late_cycles)]
    estimates = np.empty(reps, dtype=float)
    draw_size = len(early_cycles)
    for start in range(0, reps, 200):
        m = min(200, reps-start)
        corrs = []
        for sufficient in group_sums:
            picks = rng.integers(0, len(sufficient), size=(m, draw_size))
            sums = sufficient[picks].sum(axis=1)
            corrs.append(_corr_from_sums(sums))
        estimates[start:start+m] = corrs[1] - corrs[0]
    return {
        'unit': 'cycle cluster; resampled within each observed endpoint 11-cycle block',
        'resamples': reps, 'seed': seed,
        'classification': 'exploratory_cycle_cluster_resampling_sensitivity_not_confidence_interval',
        'exchangeability_assumption': 'Cycles are treated as exchangeable within each observed endpoint block.',
        'serial_dependence_accounted_for': False,
        'early_cycles': [int(early_cycles[0].cycle_code.iloc[0]), int(early_cycles[-1].cycle_code.iloc[0])],
        'late_cycles': [int(late_cycles[0].cycle_code.iloc[0]), int(late_cycles[-1].cycle_code.iloc[0])],
        'observed_delta_late_minus_early_pearson': _safe(observed_delta),
        'resampled_delta_percentile_2_5': _safe(np.nanquantile(estimates, .025)),
        'resampled_delta_median': _safe(np.nanmedian(estimates)),
        'resampled_delta_percentile_97_5': _safe(np.nanquantile(estimates, .975)),
        'fraction_resampled_delta_positive': _safe(np.nanmean(estimates > 0)),
        'interpretation': ('Exploratory cycle-cluster resampling sensitivity conditional on the observed 33 cycles. '
                           'It treats cycles within each endpoint block as exchangeable, does not account for serial dependence, '
                           'and is not a confidence interval or an inferential guarantee.'),
    }


def _plot(cycle_frame, partitions, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': 'white', 'axes.facecolor': 'white'})
    figures = []

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True,
                             gridspec_kw={'height_ratios': [2, 1.3, .8]}, layout='constrained')
    x = cycle_frame.cycle_code.to_numpy(int)
    r = cycle_frame.pearson.to_numpy(float)
    rho = cycle_frame.spearman.to_numpy(float)
    n = cycle_frame.n.to_numpy(int)
    axes[0].plot(x, r, color='#2864a5', marker='o', markersize=4, label='Pearson')
    axes[0].plot(x, rho, color='#cf8e30', marker='s', markersize=3, label='Spearman')
    axes[0].set(title='Association at each training cycle', ylabel='Correlation')
    axes[0].legend(ncol=2, loc='lower right'); axes[0].grid(alpha=.2)
    axes[1].plot(x, cycle_frame.target_median, color='#2864a5', marker='o', markersize=3, label='Median')
    axes[1].plot(x, cycle_frame.target_p10, color='#cf8e30', linestyle='--', label='P10')
    axes[1].plot(x, cycle_frame.target_p90, color='#7257a3', linestyle=':', label='P90')
    axes[1].set(title='Marginal target distribution by cycle', ylabel='threat_rank')
    axes[1].legend(ncol=3, loc='lower left'); axes[1].grid(alpha=.2)
    # Keep all 33 denominators visible without overlapping the correlation lines.
    axes[2].bar(x, n, color='#8c9bac', width=.7)
    axes[2].set(xlabel='cycle_code (train only)', ylabel='Rows', title='Observations per cycle')
    axes[2].set_xticks([47, 50, 55, 60, 65, 70, 75, 79])
    axes[2].grid(axis='y', alpha=.2)
    path = output_dir/'temporal_per_cycle.png'
    fig.savefig(path, dpi=170); plt.close(fig); figures.append(path.name)

    rows = []
    for k, result in partitions.items():
        for block in result['blocks']:
            rows.append({'k': int(k), 'label': f"{block['cycle_min']}-{block['cycle_max']}",
                         'mid': (block['cycle_min']+block['cycle_max'])/2,
                         'pooled': block['pooled_pearson'], 'equal': block['equal_cycle_mean_pearson'],
                         'n': block['n']})
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), layout='constrained')
    for ax, k in zip(axes, (2, 3, 4)):
        rows_k = [row for row in rows if row['k'] == k]
        ax.plot([r['mid'] for r in rows_k], [r['pooled'] for r in rows_k],
                color='#2864a5', marker='o', label='Pooled rows')
        ax.plot([r['mid'] for r in rows_k], [r['equal'] for r in rows_k],
                color='#cf8e30', marker='s', linestyle='--', label='Mean cycle r')
        ax.set(title=f'{k} contiguous groups of unique cycles', ylabel='Pearson correlation')
        ax.set_xticks([r['mid'] for r in rows_k], [r['label'] for r in rows_k])
        for row in rows_k:
            ax.annotate(f"n={row['n']:,}", (row['mid'], row['pooled']),
                        xytext=(0, 7), textcoords='offset points', ha='center', fontsize=8)
        ax.grid(axis='y', alpha=.2); ax.legend(ncol=2, loc='best')
    axes[-1].set_xlabel('Inclusive cycle ranges')
    path = output_dir/'temporal_partition_sensitivity.png'
    fig.savefig(path, dpi=170); plt.close(fig); figures.append(path.name)
    return figures


def run_temporal_eda(data_dir=None, output_dir=None, make_plots=True):
    root = Path(data_dir).resolve() if data_dir else data_root()
    files = task_files('regression', root)
    if any(files[k] is None for k in ('train', 'test', 'sample')):
        raise FileNotFoundError('Require official regression Train.csv, Test.csv, SampleSubmission.csv.')
    train = pd.read_csv(files['train'])
    test = pd.read_csv(files['test'])
    sample = pd.read_csv(files['sample'])
    if sorted(set(train.columns)-set(test.columns)) != ['threat_rank']:
        raise ValueError('Regression target schema not verified; train must have only threat_rank.')
    if not sample.record_id.equals(test.record_id):
        raise ValueError('Sample/test ID order mismatch.')
    input_manifest = {
        role: {'file': path.name, 'bytes': int(path.stat().st_size), 'sha256': _sha256(path)}
        for role, path in files.items()
    }
    input_manifest['train']['rows'] = int(len(train))
    input_manifest['test']['rows'] = int(len(test))
    input_manifest['sample']['rows'] = int(len(sample))
    df = train[['cycle_code', 'impact_load', 'threat_rank']].copy()
    for col in df:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if df.cycle_code.nunique() < 4:
        raise ValueError('At least four complete cycles are needed for the requested split-sensitivity analysis.')

    cycles = []
    percycle = {}
    percycle_sums = []
    for cycle, group in df.groupby('cycle_code', sort=True):
        corr = _corr(group)
        target = group.threat_rank
        counts = target.value_counts()
        rank = target.rank(method='average')
        expected = np.round(100*rank/len(group), 4)
        ecdf = target.rank(method='max')/len(group)
        rank_unit = target.to_numpy(float)/100
        ordered_rank_unit = np.sort(rank_unit)
        empirical_right = np.arange(1, len(target)+1)/len(target)
        empirical_left = np.arange(0, len(target))/len(target)
        ks_d_plus = np.max(empirical_right-ordered_rank_unit)
        ks_d_minus = np.max(ordered_rank_unit-empirical_left)
        rank_implied_mean = 50.0 + 50.0/len(target)
        right_ecdf_gap = np.abs(ecdf.to_numpy()-rank_unit)
        row = {
            'cycle_code': int(cycle), 'n': int(len(group)),
            'impact_load_min': _safe(group.impact_load.min()), 'impact_load_median': _safe(group.impact_load.median()),
            'impact_load_max': _safe(group.impact_load.max()),
            'pearson': corr['pearson'], 'spearman': corr['spearman'],
            'target_mean': _safe(target.mean()), 'target_sd': _safe(target.std()),
            'rank_implied_mean_50_plus_50_over_n': _safe(rank_implied_mean),
            'target_mean_minus_rank_implied_mean': _safe(target.mean()-rank_implied_mean),
            'target_p10': _safe(target.quantile(.1)), 'target_q25': _safe(target.quantile(.25)),
            'target_median': _safe(target.median()), 'target_q75': _safe(target.quantile(.75)),
            'target_p90': _safe(target.quantile(.9)), 'target_unique': int(target.nunique()),
            'largest_tie_n': int(counts.max()), 'largest_tie_share': _safe(counts.max()/len(group)),
            'average_rank_percentile_match_count': int(np.equal(target.to_numpy(), expected.to_numpy()).sum()),
            'ecdf_uniform_right_gap_max_abs': _safe(right_ecdf_gap.max()),
            'uniform_ks_d_plus': _safe(ks_d_plus),
            'uniform_ks_d_minus': _safe(ks_d_minus),
            'uniform_ks_two_sided_distance': _safe(max(ks_d_plus, ks_d_minus)),
            'sufficient_stats': _stats(group).tolist(),
        }
        cycles.append(row)
        percycle[int(cycle)] = group
        percycle_sums.append(_stats(group))

    cycle_frame = pd.DataFrame(cycles)
    if cycle_frame.pearson.isna().any() or cycle_frame.spearman.isna().any():
        bad = cycle_frame.loc[cycle_frame.pearson.isna() | cycle_frame.spearman.isna(), 'cycle_code'].tolist()
        raise ValueError(f'Undefined cycle correlation (constant or insufficient support): {bad}')
    unique_cycles = np.sort(cycle_frame.cycle_code.to_numpy(int))
    partition_results = {}
    for k in (2, 3, 4):
        blocks = _blocks(unique_cycles, k)
        summaries = []
        for chunk in blocks:
            group = df[df.cycle_code.isin(chunk)]
            per_cycle_rs = cycle_frame[cycle_frame.cycle_code.isin(chunk)].pearson
            pooled = _corr(group)
            summaries.append({
                'cycle_min': int(chunk[0]), 'cycle_max': int(chunk[-1]), 'cycle_count': int(len(chunk)),
                'n': int(len(group)), 'pooled_pearson': pooled['pearson'],
                'pooled_spearman': pooled['spearman'],
                'equal_cycle_mean_pearson': _safe(per_cycle_rs.mean()),
                'equal_cycle_sd_pearson': _safe(per_cycle_rs.std()),
                'equal_cycle_mean_spearman': _safe(cycle_frame[cycle_frame.cycle_code.isin(chunk)].spearman.mean()),
                'equal_cycle_sd_spearman': _safe(cycle_frame[cycle_frame.cycle_code.isin(chunk)].spearman.std()),
                'target_mean': _safe(group.threat_rank.mean()), 'target_sd': _safe(group.threat_rank.std()),
                'target_p10': _safe(group.threat_rank.quantile(.1)),
                'target_median': _safe(group.threat_rank.median()),
                'target_p90': _safe(group.threat_rank.quantile(.9)),
            })
        partition_results[str(k)] = {'blocks': summaries}

    # The main contrast uses three blocks of 11 whole, ordered cycles.
    main_chunks = _blocks(unique_cycles, 3)
    if [len(chunk) for chunk in main_chunks] != [11, 11, 11]:
        raise ValueError('Primary 3-block comparison must contain eleven unique cycles per block.')
    endpoint_cycle_frames = [[percycle[int(c)] for c in chunk] for chunk in main_chunks]
    initial = _corr(df[df.cycle_code.isin(main_chunks[0])])['pearson']
    final = _corr(df[df.cycle_code.isin(main_chunks[-1])])['pearson']
    observed_delta = final-initial
    initial_spearman = _corr(df[df.cycle_code.isin(main_chunks[0])])['spearman']
    final_spearman = _corr(df[df.cycle_code.isin(main_chunks[-1])])['spearman']
    observed_spearman_delta = final_spearman-initial_spearman
    first_last_equal_cycle_delta = (
        cycle_frame[cycle_frame.cycle_code.isin(main_chunks[-1])].pearson.mean()
        - cycle_frame[cycle_frame.cycle_code.isin(main_chunks[0])].pearson.mean())
    first_last_equal_cycle_spearman_delta = (
        cycle_frame[cycle_frame.cycle_code.isin(main_chunks[-1])].spearman.mean()
        - cycle_frame[cycle_frame.cycle_code.isin(main_chunks[0])].spearman.mean())
    loo = []
    for idx, cycle in enumerate(unique_cycles):
        if idx >= 11 and idx < 22:
            continue
        early_chunk = main_chunks[0].copy()
        late_chunk = main_chunks[-1].copy()
        if idx < 11:
            early_chunk = early_chunk[early_chunk != cycle]
        else:
            late_chunk = late_chunk[late_chunk != cycle]
        early = _corr(df[df.cycle_code.isin(early_chunk)])['pearson']
        late = _corr(df[df.cycle_code.isin(late_chunk)])['pearson']
        early_rho = _corr(df[df.cycle_code.isin(early_chunk)])['spearman']
        late_rho = _corr(df[df.cycle_code.isin(late_chunk)])['spearman']
        loo.append({'cycle_code': int(cycle), 'delta_pearson_without_cycle': _safe(late-early),
                    'change_from_primary_delta': _safe((late-early)-observed_delta),
                    'delta_spearman_without_cycle': _safe(late_rho-early_rho),
                    'change_from_primary_spearman_delta': _safe((late_rho-early_rho)-observed_spearman_delta)})

    # One-way descriptive nonlinear association: variance explained by exact
    # impact_load group means on the same rows. This is in-sample, not predictive.
    global_mean = df.threat_rank.mean()
    total_ss = ((df.threat_rank-global_mean)**2).sum()
    grouped = df.groupby('impact_load').threat_rank.agg(['size', 'mean', 'std'])
    between_ss = ((grouped['mean']-global_mean)**2*grouped['size']).sum()
    nonlinear = {'distinct_impact_levels': int(len(grouped)),
                 'eta_squared_exact_impact': _safe(between_ss/total_ss) if total_ss else None,
                 'pearson_squared': _safe(_corr(df)['pearson']**2),
                 'spearman': _corr(df)['spearman'],
                 'interpretation': 'In-sample grouped association only; no split, fit, or predictive performance claim.'}

    between_cycle_stats = {
        'n_cycles': int(len(cycle_frame)),
        'pearson_min': _safe(cycle_frame.pearson.min()), 'pearson_median': _safe(cycle_frame.pearson.median()),
        'pearson_max': _safe(cycle_frame.pearson.max()),
        'spearman_min': _safe(cycle_frame.spearman.min()), 'spearman_median': _safe(cycle_frame.spearman.median()),
        'spearman_max': _safe(cycle_frame.spearman.max()),
        'target_mean_range': [_safe(cycle_frame.target_mean.min()), _safe(cycle_frame.target_mean.max())],
        'target_median_range': [_safe(cycle_frame.target_median.min()), _safe(cycle_frame.target_median.max())],
        'target_sd_range': [_safe(cycle_frame.target_sd.min()), _safe(cycle_frame.target_sd.max())],
        'target_p10_range': [_safe(cycle_frame.target_p10.min()), _safe(cycle_frame.target_p10.max())],
        'target_p90_range': [_safe(cycle_frame.target_p90.min()), _safe(cycle_frame.target_p90.max())],
        'rank_implied_mean_range': [_safe(cycle_frame.rank_implied_mean_50_plus_50_over_n.min()),
                                    _safe(cycle_frame.rank_implied_mean_50_plus_50_over_n.max())],
        'max_abs_target_mean_minus_rank_implied_mean': _safe(
            cycle_frame.target_mean_minus_rank_implied_mean.abs().max()),
        'average_rank_percentile_matches': int(cycle_frame.average_rank_percentile_match_count.sum()),
        'average_rank_percentile_rows': int(cycle_frame.n.sum()),
        'ecdf_uniform_right_gap_max_abs_cycle': _safe(cycle_frame.ecdf_uniform_right_gap_max_abs.max()),
        'uniform_ks_two_sided_distance_max_cycle': _safe(cycle_frame.uniform_ks_two_sided_distance.max()),
        'largest_tie_share_max_cycle': _safe(cycle_frame.largest_tie_share.max()),
    }

    adjacent = []
    for left, right in zip(cycles[:-1], cycles[1:]):
        adjacent.append({'from_cycle': left['cycle_code'], 'to_cycle': right['cycle_code'],
                         'pearson_delta': _safe(right['pearson']-left['pearson']),
                         'spearman_delta': _safe(right['spearman']-left['spearman']),
                         'n_from': left['n'], 'n_to': right['n']})
    largest_adjacent = sorted(adjacent, key=lambda row: abs(row['pearson_delta']), reverse=True)[:6]
    cycle56 = cycle_frame[cycle_frame.cycle_code == 56]
    cycle56_comparison = None
    if not cycle56.empty:
        idx = int(cycle_frame.index[cycle56.index[0]])
        neighbors = cycle_frame.iloc[[max(0, idx-1), idx, min(len(cycle_frame)-1, idx+1)]]
        cycle56_comparison = neighbors[['cycle_code', 'n', 'pearson', 'spearman', 'target_mean', 'target_sd']].to_dict('records')
        cycle56_comparison = {
            'neighboring_cycles': cycle56_comparison,
            'pearson_delta_55_to_56': next((x['pearson_delta'] for x in adjacent if x['from_cycle'] == 55 and x['to_cycle'] == 56), None),
            'pearson_delta_56_to_57': next((x['pearson_delta'] for x in adjacent if x['from_cycle'] == 56 and x['to_cycle'] == 57), None),
            'interpretation': 'Local descriptive context only; cycle 56 is not treated as a prespecified break.'}

    exact_match_count = between_cycle_stats['average_rank_percentile_matches']
    report = {
        'status': 'COMPLETE_DESCRIPTIVE_TEMPORAL_ROBUSTNESS',
        'question': 'Does the impact_load-threat_rank association change across cycle_code while threat_rank marginals remain stable?',
        'input_root': 'configured TP2 regression input; path omitted',
        'scope': 'Official labeled regression train only; the test target is unavailable and unused.',
        'input_manifest': input_manifest,
        'schema_validation': {
            'train_has_threat_rank': bool('threat_rank' in train.columns),
            'train_columns_absent_from_test': sorted(set(train.columns)-set(test.columns)),
            'sample_test_id_order_match': True,
            'analysis_rows_complete': int(len(df)),
            'analysis_rows_dropped_missing_or_nonfinite': int(len(train)-len(df)),
        },
        'method': {'primary_periods': 'Sort unique train cycle_code values and split them into three equal-sized contiguous blocks; do not split individual cycles or inspect target to set boundaries.',
                   'sensitivity': 'Repeat the unique-cycle grouping using 2, 3, and 4 blocks; show row-pooled correlation and the equal-cycle mean of individual cycle correlations.',
                   'bootstrap': ('Resample observed cycles as clusters independently within first/last eleven-cycle blocks; fixed seed. '
                                 'This is exploratory sensitivity conditional on exchangeability, not a confidence interval or '
                                 'a serial-dependence-robust result.'),
                   'nonlinearity': 'Exact-impact eta squared is in-sample variance decomposition, not predictive performance.',
                   'rank': ('Compare target to within-cycle average ranks of that same target. Report two-sided uniform KS distance '
                            'as an effect size without a p-value, plus the separately labeled right-continuous ECDF gap. '
                            'The implied mean 50+50/n is conditional on average-rank identity and does not prove source formula.' )},
        'overall': _corr(df), 'cycle_count': len(cycles), 'cycles': cycles,
        'marginal_vs_association': between_cycle_stats,
        'partition_sensitivity': partition_results,
        'primary_first_vs_last': {'early': partition_results['3']['blocks'][0],
                                  'late': partition_results['3']['blocks'][-1],
                                  'observed_pearson_delta_late_minus_early': _safe(observed_delta),
                                  'observed_spearman_delta_late_minus_early': _safe(observed_spearman_delta),
                                  'equal_cycle_mean_pearson_delta_late_minus_early': _safe(first_last_equal_cycle_delta),
                                  'equal_cycle_mean_spearman_delta_late_minus_early': _safe(first_last_equal_cycle_spearman_delta),
                                  'cycle_cluster_resampling': _cycle_cluster_sensitivity(
                                      endpoint_cycle_frames[0], endpoint_cycle_frames[-1], observed_delta)},
        'leave_one_cycle_out_endpoint_sensitivity': loo,
        'adjacent_cycle_changes_largest_abs_pearson': largest_adjacent,
        'adjacent_cycle_changes_all': adjacent,
        'cycle_56_neighborhood_descriptive': cycle56_comparison,
        'nonlinear_association_diagnostics': nonlinear,
        'interpretation_limits': [
            'Cycle correlations are descriptive summaries of one observed, ordered set of cycles; adjacent cycles and rows need not be independent.',
            'Changing correlation is not by itself a structural break, causal mechanism, or statistically established drift.',
            'Stable marginal ranks do not imply stable conditional association; percentile-like target construction may mechanically stabilize marginals.',
            'Within-cycle rank identity examines target values against ranks of those same values. It does not establish target lineage, official semantics, or authorize target transformations for validation/test.',
            'The near-50+50/n target mean follows mathematically if the observed rounded average-rank identity holds; it is a target-to-target construction implication, not independent evidence of how the official target was generated.',
            'Uniform KS distances are descriptive effect sizes; no null-test p-value is used as evidence across the 33 observed cycles.',
            'The right-continuous ECDF gap is one-sided and is distinct from the two-sided uniform KS distance.',
            'Cycle code is treated as ordered numeric disclosure levels; calendar spacing and real-world time intervals are not inferred.',
            'Cycle cluster resampling assumes exchangeability within endpoint blocks and does not account for serial dependence or future cycle uncertainty.',
        ],
        'figures': [],
    }
    output = Path(output_dir) if output_dir else PROJECT_ROOT/'reports'/'eda'
    output.mkdir(parents=True, exist_ok=True)
    if make_plots:
        report['figures'] = _plot(cycle_frame, partition_results, output)
    report_path = output/'temporal_robustness.json'
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    return report
