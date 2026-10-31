"""Exploratory aggregate checks for pattern/org/module code heterogeneity.

No model is fitted, and no row-level source values or category IDs are written.
All support cutoffs are chosen before inspecting the target summaries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.paths import data_root, task_files

EXPECTED = {
    'regression': '756523ff8156a9cd776e52249f7e2a8f23dcec432b8d10570e55ff6db0e0f082',
    'classification': '875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c',
}
SUPPORTS = (30, 100, 300)
CODE_COLUMNS = ('pattern_code', 'org_code', 'module_code')


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _summary_by_code(frame: pd.DataFrame, code: str, target: str) -> pd.DataFrame:
    # Never export category values: these are anonymized identifiers, not meaningful labels.
    key = frame[code].astype('string').fillna('__AGG_MISSING__')
    if key.eq('__AGG_MISSING__').sum() != frame[code].isna().sum():
        raise ValueError('Reserved aggregate key conflicts with a source category value.')
    table = (frame.assign(_key=key).groupby('_key', observed=True)
             .agg(n=(target, 'size'), mean=(target, 'mean')).reset_index())
    cycle_mean = frame.groupby('cycle_code', observed=True)[target].transform('mean')
    residual_frame = pd.DataFrame({'_key': key,
                                   'residual': frame[target].to_numpy() - cycle_mean.to_numpy()})
    residual = residual_frame.groupby('_key', observed=True).residual.agg(within_cycle_delta='mean')
    table = table.merge(residual.reset_index(), on='_key', validate='one_to_one')
    return table


def _target_summary(frame: pd.DataFrame, code: str, target: str,
                    kind: str) -> dict:
    values = frame[target].to_numpy(dtype=float)
    cycle_base = frame.groupby('cycle_code', observed=True)[target].transform('mean').to_numpy(dtype=float)
    residual = values - cycle_base
    group = frame[code].astype('string')
    counts = group.value_counts(dropna=True)
    whole_mean = float(values.mean())
    result = {'target': target, 'kind': kind, 'overall_mean': whole_mean,
              'whole_train_rows': int(len(frame)), 'overall_missing_target_rows': int(frame[target].isna().sum()),
              'supports': []}
    for threshold in SUPPORTS:
        keep = counts[counts >= threshold].index
        mask = group.isin(keep).to_numpy()
        if not mask.any():
            continue
        subgroups = group[mask]
        subn = subgroups.value_counts()
        means = pd.Series(values[mask]).groupby(subgroups.reset_index(drop=True)).mean()
        delta_means = pd.Series(residual[mask]).groupby(subgroups.reset_index(drop=True)).mean()
        # Target-blind threshold; association is computed only over represented rows.
        within = residual[mask]
        within_centered = within - within.mean()
        centered_delta_means = pd.Series(within_centered).groupby(subgroups.reset_index(drop=True)).mean()
        explained = float(sum(int(subn[k]) * float(centered_delta_means[k])**2 for k in subn.index))
        total = float(np.dot(within_centered, within_centered))
        eta2_cycle_adjusted = explained / total if total > 0 else None
        ordered = means.sort_values()
        delta_ordered = delta_means.sort_values()
        result['supports'].append({
            'minimum_rows_per_code': threshold,
            'supported_codes': int(len(keep)),
            'represented_rows': int(mask.sum()),
            'row_coverage': float(mask.mean()),
            'group_size_median': float(subn.median()),
            'group_size_p10': float(subn.quantile(.10)),
            'group_size_p90': float(subn.quantile(.90)),
            'observed_target_mean_range': [float(ordered.iloc[0]), float(ordered.iloc[-1])],
            'observed_group_mean_quantiles_unweighted': {
                str(q): float(means.quantile(q)) for q in (.1, .25, .5, .75, .9)},
            'within_cycle_adjusted_group_mean_delta_quantiles_unweighted': {
                str(q): float(delta_means.quantile(q)) for q in (.1, .25, .5, .75, .9)},
            'within_cycle_adjusted_group_delta_range': [float(delta_ordered.iloc[0]), float(delta_ordered.iloc[-1])],
            'eta_squared_of_cycle_centered_targets_by_code': eta2_cycle_adjusted,
            'interpretation': 'One-way between-code share of variance in outcomes after subtracting the overall mean of each cycle, among rows in categories meeting this target-blind support threshold. Category membership composition across cycles is not standardized. This is descriptive, not causal or independent-validation evidence.'})
    return result


def run(root: Path | None = None, output_dir: Path | None = None) -> dict:
    project_root = Path(root or Path(__file__).resolve().parents[1]).resolve()
    output_dir = Path(output_dir or project_root/'reports/eda_candidates/categories').resolve()
    configured_data_root = data_root()
    files = {t: task_files(t, configured_data_root) for t in ('regression', 'classification')}
    for task in files:
        assert files[task]['train'], f'Missing official {task} Train.csv.'
        assert sha(files[task]['train']) == EXPECTED[task], f'{task} Train SHA mismatch.'
    reg = pd.read_csv(files['regression']['train'])
    cls = pd.read_csv(files['classification']['train'])
    assert reg.shape == (50_608, 19) and cls.shape == (50_608, 19)
    assert reg.record_id.is_unique and cls.record_id.is_unique
    assert reg.record_id.equals(cls.record_id)
    feature_columns = [c for c in reg.columns if c != 'threat_rank']
    assert set(feature_columns) == set(cls.columns) - {'artifact_signal'}
    assert reg[feature_columns].equals(cls[feature_columns])
    assert reg.cycle_code.between(47, 79).all() and cls.cycle_code.between(47, 79).all()
    assert reg.threat_rank.between(0, 100).all() and cls.artifact_signal.isin([0, 1]).all()

    metrics = {}
    code_catalog = {}
    concentration = {}
    for code in CODE_COLUMNS:
        categories = _summary_by_code(reg, code, 'threat_rank')
        cat_cls = _summary_by_code(cls, code, 'artifact_signal')
        n_by_code = reg.groupby(code, dropna=False, observed=True).size()
        n_nonmissing = reg.loc[reg[code].notna()].groupby(code, observed=True).size()
        code_catalog[code] = {
            'groups_including_missing': int(len(n_by_code)),
            'observed_nonmissing_codes': int(reg[code].nunique(dropna=True)),
            'missing_rows': int(reg[code].isna().sum()),
            'missing_share': float(reg[code].isna().mean()),
            'group_size_median_including_missing': float(n_by_code.median()),
            'top_10_row_share_including_missing_group': float(n_by_code.nlargest(10).sum()/len(reg)),
            'top_10_named_code_row_share': float(n_nonmissing.nlargest(10).sum()/len(reg)),
            'support_counts': {str(k): {'categories_including_missing': int((n_by_code >= k).sum()),
                                         'rows_in_supported_categories': int(n_by_code[n_by_code >= k].sum()),
                                         'row_coverage': float(n_by_code[n_by_code >= k].sum()/len(reg)),
                                         'nonmissing_categories': int((n_nonmissing >= k).sum()),
                                         'rows_in_supported_nonmissing_categories': int(n_nonmissing[n_nonmissing >= k].sum()),
                                         'nonmissing_category_row_coverage': float(n_nonmissing[n_nonmissing >= k].sum()/len(reg))}
                               for k in SUPPORTS},
            'mean_group_sizes_of_supported_codes': {
                str(k): float(n_by_code[n_by_code >= k].mean()) if (n_by_code >= k).any() else None
                for k in SUPPORTS}}
        for target, frame, kind in [('threat_rank', reg, 'continuous'),
                                    ('artifact_signal', cls, 'binary prevalence')]:
            metrics[f'{code}::{target}'] = _target_summary(frame, code, target, kind)

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / 'figures'; figure_dir.mkdir(exist_ok=True)
    # Figure 1: target-blind support versus cycle-adjusted code mean for pattern_code.
    pat_reg = _summary_by_code(reg, 'pattern_code', 'threat_rank')
    pat_cls = _summary_by_code(cls, 'pattern_code', 'artifact_signal')
    sizes = reg.groupby('pattern_code', dropna=False, observed=True).size().to_numpy()
    # Keep only codes with n>=30, the smallest prespecified support cutoff.
    keep_reg = pat_reg[(pat_reg.n >= SUPPORTS[0]) & (pat_reg._key != '__AGG_MISSING__')].rename(
        columns={'n': 'n_regression', 'mean': 'mean_threat'}).copy()
    keep_cls = pat_cls[(pat_cls.n >= SUPPORTS[0]) & (pat_cls._key != '__AGG_MISSING__')].copy()
    keep_cls = keep_cls.rename(columns={'n': 'n_classification', 'mean': 'artifact_prevalence'})
    pairs = keep_reg.merge(keep_cls[['_key', 'n_classification', 'artifact_prevalence', 'within_cycle_delta']],
                           on='_key', validate='one_to_one', suffixes=('_threat', '_artifact'))
    assert pairs.n_regression.equals(pairs.n_classification)
    paired = pairs[pairs._key != '__AGG_MISSING__']
    cross_target = {}
    for threshold in SUPPORTS:
        subset = paired[paired.n_regression >= threshold]
        x = subset.within_cycle_delta_threat.to_numpy()
        y = subset.within_cycle_delta_artifact.to_numpy()
        n = subset.n_regression.to_numpy(dtype=float)
        def corr(a, b, w=None):
            if len(a) < 3: return None
            if w is None: return float(np.corrcoef(a, b)[0, 1])
            w = w / w.sum(); ax = np.average(a, weights=w); by = np.average(b, weights=w)
            return float(np.average((a-ax)*(b-by), weights=w) /
                         np.sqrt(np.average((a-ax)**2, weights=w)*np.average((b-by)**2, weights=w)))
        from scipy.stats import spearmanr
        cross_target[str(threshold)] = {
            'nonmissing_pattern_codes': int(len(subset)),
            'pearson_cycle_adjusted_group_means_unweighted': corr(x, y),
            'spearman_cycle_adjusted_group_means_unweighted': float(spearmanr(x, y).statistic) if len(x) > 2 else None,
            'pearson_cycle_adjusted_group_means_weighted_by_code_support': corr(x, y, n),
            'pearson_raw_group_means_unweighted': corr(subset.mean_threat.to_numpy(), subset.artifact_prevalence.to_numpy()),
            'warning': 'Ecological cross-code correlation; category averages are the units. It does not measure within-record association or causality.'}
    per_cycle_profiles = []
    for cycle, one_cycle in reg.loc[reg.pattern_code.notna()].groupby('cycle_code', observed=True):
        # Only use pattern codes with >=10 rows in this cycle; support is label-blind.
        r = one_cycle.groupby('pattern_code', observed=True).threat_rank.agg(['mean', 'size'])
        c = cls.loc[cls.cycle_code.eq(cycle) & cls.pattern_code.notna()].groupby(
            'pattern_code', observed=True).artifact_signal.agg(['mean', 'size'])
        common_codes = r.index[r['size'].ge(10)].intersection(c.index[c['size'].ge(10)])
        if len(common_codes) >= 10:
            r_mean = r.loc[common_codes, 'mean'].to_numpy(dtype=float)
            c_rate = c.loc[common_codes, 'mean'].to_numpy(dtype=float)
            per_cycle_profiles.append({'cycle_code': int(cycle), 'n_supported_codes': int(len(common_codes)),
                'pearson_code_mean_association': float(np.corrcoef(r_mean, c_rate)[0, 1]),
                'spearman_code_mean_association': float(pd.Series(r_mean).corr(pd.Series(c_rate), method='spearman'))})
    cycle_r = np.array([x['pearson_code_mean_association'] for x in per_cycle_profiles], dtype=float)
    assert len(cycle_r) >= 25, 'Too few cycles retain ten supported pattern codes.'
    cycle_profile_sensitivity = {
        'rule': 'Per cycle, named pattern codes with n≥10 in both Train files; only cycles with at least 10 codes are summarized.',
        'cycles_with_enough_support': int(len(cycle_r)), 'cycle_count_total': int(reg.cycle_code.nunique()),
        'pearson_across_code_means_median': float(np.median(cycle_r)),
        'pearson_across_code_means_p10_p90': [float(np.quantile(cycle_r,.1)), float(np.quantile(cycle_r,.9))],
        'positive_cycles_fraction': float(np.mean(cycle_r > 0)),
        'leave_one_cycle_out_mean_range': [float(min(np.mean(np.delete(cycle_r,i)) for i in range(len(cycle_r)))),
                                            float(max(np.mean(np.delete(cycle_r,i)) for i in range(len(cycle_r))))],
        'per_cycle_details': per_cycle_profiles,
        'interpretation': 'Direction/stability check for category-average alignment within cycles; cycle is a composition stress test, not the research question.'}
    fig, ax = plt.subplots(figsize=(10, 6))
    size = np.sqrt(keep_reg.n_regression.to_numpy()) * 3
    ax.scatter(keep_reg.within_cycle_delta, keep_cls.within_cycle_delta,
               s=size, alpha=.58, edgecolors='white', linewidths=.45, color='#326e8c')
    ax.axvline(0, color='#8a8a8a', lw=.8); ax.axhline(0, color='#8a8a8a', lw=.8)
    ax.set(title='Pattern code: profil rata-rata dua target (kode n ≥30)',
           xlabel='Deviasi mean threat_rank setelah dikurangi mean cycle',
           ylabel='Deviasi proporsi artifact_signal setelah dikurangi mean cycle')
    fig.text(.12, .015, 'Satu titik per kode yang cukup didukung; luas titik sebanding dengan √n.\nLabel kategori disamarkan karena kode bersifat anonim.', va='bottom', fontsize=9)
    ax.grid(alpha=.18); fig.tight_layout(rect=(0,.09,1,1)); fig.savefig(figure_dir/'pattern_code_two_targets.png', dpi=150); plt.close(fig)

    # Figure 2: target-blind class support and category concentration, all three codes.
    names = list(CODE_COLUMNS)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(names)); width = .24
    for j, threshold in enumerate(SUPPORTS):
        vals = [code_catalog[c]['support_counts'][str(threshold)]['nonmissing_category_row_coverage']*100 for c in names]
        ax.bar(x + (j-1)*width, vals, width, label=f'Kode nonmissing n ≥ {threshold}')
    ax.set_xticks(x, names); ax.set_ylim(0, 105)
    ax.set_ylabel('Baris yang berada pada kategori terdukung (%)')
    ax.set_title('Cakupan kategori kode berubah tajam dengan syarat dukungan')
    ax.legend(frameon=False, ncols=3); ax.grid(axis='y', alpha=.2); fig.tight_layout()
    fig.savefig(figure_dir/'category_code_support_coverage.png', dpi=150); plt.close(fig)

    payload = {
        'status': 'COMPLETE_EXPLORATORY_NOT_CONFIRMATORY',
        'source': {'regression_train_sha256': EXPECTED['regression'],
                   'classification_train_sha256': EXPECTED['classification'],
                   'course_pdf_sha256': '386fa63541cc3e9b36b56ee8f8f6c479b3bbaa7753b9482d60763326f4465c0a',
                   'code_meaning_reference': 'reports/audits/source_definitions.json; category identifiers are anonymized and no code-level interpretation is assigned.',
                   'regression_shape': list(reg.shape), 'classification_shape': list(cls.shape),
                   'ordered_record_ids_equal': True, 'all_predictor_columns_equal_by_row': True,
                   'test_files_or_labels_used': False, 'raw_text_used': False,
                   'support_cutoffs_fixed_before_target_summary': list(SUPPORTS)},
        'effect_sizes': 'One-way between-code share of variance in cycle-centered outcomes on only rows with named code support above a target-blind threshold. Cycle-centered target values are compared across pooled code categories; code-by-cycle membership composition is not standardized. This is descriptive, not a causal effect or p value.',
        'categories': code_catalog,
        'missing_code_target_profiles': {
            code: {
                target: {
                    'rows': int(frame[code].isna().sum()),
                    'target_mean_or_rate': float(frame.loc[frame[code].isna(), target].mean()),
                    'mean_after_cycle_centering': float((frame.loc[frame[code].isna(), target] -
                        frame.loc[frame[code].isna(), 'cycle_code'].map(
                            frame.groupby('cycle_code', observed=True)[target].mean())).mean()),
                }
                for target in ('threat_rank', 'artifact_signal')
                for frame in ([reg] if target == 'threat_rank' else [cls])
            }
            for code in CODE_COLUMNS
        },
        'statistics': metrics,
        'pattern_code_cross_target_profile_association': cross_target,
        'pattern_code_per_cycle_profile_sensitivity': cycle_profile_sensitivity,
        'figures': ['figures/pattern_code_two_targets.png', 'figures/category_code_support_coverage.png'],
        'non_overlap': {
            'Rousan': 'Their main question is entry_friction/impact_load versus artifact_signal prevalence. This discovery uses anonymized code composition versus both outcomes; no entry_friction or impact_load is used.',
            'Abhi': 'Their question centers scope_pressure contribution to threat_rank conditional on impact_load. This discovery concerns pattern/org/module identifiers and never conditions on scope_pressure or impact_load.',
            'Rizky': 'Their question centers telemetry_flag/module_spread versus artifact_signal. Module_code is a distinct identifier field from module_spread and telemetry_flag is not used; pattern_code and org_code are additional directions.',
            'Samuel_old': 'The earlier EDA focuses impact_load–threat_rank association by cycle and rank geometry. Here category identifiers and cross-task target heterogeneity are primary; cycle is a nuisance composition adjustment only, not the EDA question.',
        },
        'limitations': ['Code labels are anonymized; no domain meaning is inferred from the identifier.',
                        'Per-code target means can be noisy even at n=30; target-blind minimum supports and cycle-adjusted effects reduce but do not eliminate this concern.',
                        'Cycle centering is one-variable composition adjustment; impact, entry friction, scope, text, and other confounders are not controlled.',
                        'Category identity may be tied to repeated entities or source coverage; association is not a causal pattern effect.',
                        'Groups and directions were selected exploratorily from this same Train. Effect sizes and intervals are descriptive, not an independent confirmation.',
                        'High-cardinality org_code/module_code are very sparse; small supported coverage makes group-level interpretation weak.'],
    }
    (output_dir/'stats.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')

    aggregate = []
    for code in CODE_COLUMNS:
        for target in ('threat_rank', 'artifact_signal'):
            for item in metrics[f'{code}::{target}']['supports']:
                aggregate.append({'code_feature': code, 'target': target, **item})
    pd.DataFrame(aggregate).to_csv(output_dir/'category_support_effects_aggregate.csv', index=False, float_format='%.9g')

    lines = ['# Explorasi kategori anonim dan heterogenitas dua target', '',
             '**Status: eksplorasi awal, belum rekomendasi pertanyaan EDA.**', '',
             'Train regression dan classification masing-masing 50.608 × 19; SHA cocok dengan audit, record_id berurutan sama, dan semua predictor cocok per baris. Tidak ada Test atau teks mentah. Cutoff dukungan 30/100/300 ditetapkan tanpa melihat target.', '',
             '## Profil dukungan kategori (selection target-blind)', '',
             '| Fitur kode | Kode nonmissing | Missing | Median ukuran grup | Cakupan kode bernama n≥30 | n≥100 | n≥300 | Top-10 kode share |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in CODE_COLUMNS:
        c = code_catalog[name]
        lines.append(f"| `{name}` | {c['observed_nonmissing_codes']:,} | {c['missing_rows']:,} ({c['missing_share']:.1%}) | {c['group_size_median_including_missing']:.0f} | {c['support_counts']['30']['nonmissing_category_row_coverage']:.1%} | {c['support_counts']['100']['nonmissing_category_row_coverage']:.1%} | {c['support_counts']['300']['nonmissing_category_row_coverage']:.1%} | {c['top_10_named_code_row_share']:.1%} |")
    lines += ['', '## Efek deskriptif kode setelah pemusatan per cycle', '',
              'Ukuran η² satu-arah adalah proporsi variasi outcome yang sudah dikurangi mean setiap cycle, pada baris kode bernama yang melampaui dukungan minimum. Komposisi kategori dalam tiap cycle tidak distandardisasi; ini ringkasan deskriptif, bukan partial effect kausal atau p-value.', '',
              '| Kode | Target | n≥30: grup/coverage/η² | n≥100: grup/coverage/η² | n≥300: grup/coverage/η² |',
              '|---|---|---:|---:|---:|']
    for name in CODE_COLUMNS:
        for target in ('threat_rank', 'artifact_signal'):
            supp = metrics[f'{name}::{target}']['supports']
            def fmt(s): return f"{s['supported_codes']}/{s['row_coverage']:.1%}/{s['eta_squared_of_cycle_centered_targets_by_code']:.4f}"
            lines.append(f"| `{name}` | `{target}` | " + ' | '.join(fmt(s) for s in supp) + ' |')
    lines += ['', '![Outcome residuals by supported pattern code](figures/pattern_code_two_targets.png)', '',
              'Each dot is an anonymized pattern code with at least 30 rows; axes are code-level mean deviations after removing the cycle-wide outcome mean. Bubble size reflects support. The plot reveals whether high-mean categories on one outcome tend to be high or low on the other. It does not show within-record correlation or claim the code causes either outcome.', '',
              'Profile association remains positive under all three label-blind overall supports: details and denominators are in `stats.json`. As an additional cycle-composition check, code-level profile correlations are recomputed within each cycle for codes with at least 10 rows in both tasks; this is a stability check, not an alternative temporal research question.', '',
              '![Category code support coverage](figures/category_code_support_coverage.png)', '',
              'Org and module identifiers are extremely sparse: most rows have unique/rare values. Their wide raw group ranges are expected to include noisy small groups and do not make these categories a strong teaching question without pooling or verified semantic labels. Pattern_code retains broad support and gives a more tractable direction, but its anonymized categories still lack interpretable names.', '',
              '## Overlap screen', '',
              '- **Rousan:** no `impact_load` or `entry_friction`; category code outcomes are the focus, so not the same question.',
              '- **Abhi:** no `scope_pressure` or `impact_load`; category identifiers are the focus.',
              '- **Rizky:** `module_code` is not `module_spread`; avoid making module_code × telemetry the question. `pattern_code`/`org_code` direction is distinct.',
              '- **Samuel old EDA:** targets differ in primary emphasis (code-associated category heterogeneity/cross-target alignment); cycle adjustment only handles composition. Not a cycle trend analysis.', '',
              '## Preliminary interpretation and stop point', '',
              'The strongest screening signal is pattern_code: many groups have enough rows for an aggregate comparison, while org/module coverage is low at the same support rule. Whether its cross-target category heterogeneity is sufficiently stable and interpretable to shortlist requires the broader candidate comparison. The code values have no stated domain semantics, and cycle-centering alone leaves substantial confounding. Since all directions were discovered in the same data, this screen is exploratory and must not be described as confirmation.', '',
              'Complete machine-readable figures and sensitivity values are in `stats.json`. Reproduce with `python scripts/eda_candidates_categories.py`.']
    (output_dir/'FINDINGS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=None,
                        help='Optional data root; by default read official task_train files via TP2_DATA_DIR/data/raw.')
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    target = args.output or (PROJECT_ROOT/'reports/eda_candidates/categories')
    result = run(args.root, target)
    print(json.dumps({'status': result['status'], 'outputs': str(target.resolve()),
                      'code_columns': list(CODE_COLUMNS), 'support_cutoffs': list(SUPPORTS)}, ensure_ascii=False))


if __name__ == '__main__': main()
