"""Post-screening diagnostics from saved Train predictions; never fits a model.

Run: python -m scripts.optimization_v3_postscreen_review
Writes only reports/optimization_v3/postscreen_review/. No Test/Kaggle access.
Post-hoc routing below is diagnostic, not a qualified candidate or gate amendment.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from src.utils.paths import PROJECT_ROOT, task_files

OUT = PROJECT_ROOT / 'reports/optimization_v3/postscreen_review'
RUNS = PROJECT_ROOT / 'experiments/runs/optimization_v3_batch1'
REGISTRY = PROJECT_ROOT / 'reports/optimization_v3/batch1/batch_ledger.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()


def sufficient(y, p, task):
    """Additive stats allow exact pooled metrics, including repeated cycle blocks."""
    y, p = np.asarray(y), np.asarray(p)
    if task == 'regression':
        return np.array([np.abs(y - p).sum(), len(y)], dtype=float)
    label = p >= .5
    return np.array([np.sum((y == a) & (label == b))
                     for a, b in ((0, 0), (0, 1), (1, 0), (1, 1))], dtype=float)


def metric(stats, task):
    s = np.asarray(stats, dtype=float)
    if task == 'regression':
        return float(s[0] / s[1]) if s[1] else None
    tn, fp, fn, tp = s
    if not s.sum():
        return None
    f0, f1 = 2 * tn + fp + fn, 2 * tp + fp + fn
    return float(((2 * tn / f0 if f0 else 0) + (2 * tp / f1 if f1 else 0)) / 2)


def contrast(y, p, b, task):
    cm, bm = metric(sufficient(y, p, task), task), metric(sufficient(y, b, task), task)
    return {'n': len(y), 'candidate': cm, 'baseline': bm,
            'gain': None if not len(y) else (bm - cm if task == 'regression' else cm - bm)}


def circular_block_interval(y, p, b, cycles, task, *, draws=2000, block_length=3):
    """Descriptive 90% sensitivity interval, conditional on already selected fits.

    Circular moving blocks preserve adjacent cycles locally; only twelve cycles
    per horizon. This does NOT account for model selection, refit uncertainty,
    long-term drift, or guarantee coverage for future Test.
    """
    unique = np.unique(cycles)
    if len(unique) < block_length:
        return None
    ps = np.array([sufficient(y[cycles == c], p[cycles == c], task) for c in unique])
    bs = np.array([sufficient(y[cycles == c], b[cycles == c], task) for c in unique])
    rng, gains = np.random.default_rng(20261101), []
    for _ in range(draws):
        starts = rng.integers(0, len(unique), int(np.ceil(len(unique) / block_length)))
        ix = ((starts[:, None] + np.arange(block_length)) % len(unique)).ravel()[:len(unique)]
        cm, bm = metric(ps[ix].sum(axis=0), task), metric(bs[ix].sum(axis=0), task)
        gains.append(bm - cm if task == 'regression' else cm - bm)
    lo, hi = np.quantile(gains, [.05, .95])
    return {'lower': float(lo), 'upper': float(hi), 'draws': draws,
            'block_length': block_length, 'unique_cycles': len(unique),
            'interpretation': 'descriptive post-selection sensitivity; not confirmatory CI'}


def load_bundle(card_id, cutoff):
    name = f'{card_id}_fit{cutoff}.npz'
    folder = RUNS / card_id
    # Explicitly choose final revisions; never select an old artifact by glob order.
    if card_id == 'R11':
        folder /= 'revision_03'
    elif card_id == 'R23':
        folder /= 'v2'
    path = folder / name
    if not path.exists():
        return None, None
    sidecar = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    assert sidecar['npz_sha256'] == sha(path)
    assert sidecar['identity_sha256'] == json_digest(sidecar['identity'])
    assert sidecar['integrity_sha256'] == json_digest({k: v for k, v in sidecar.items() if k != 'integrity_sha256'})
    assert sidecar['identity']['card_id'] == card_id and sidecar['identity']['cutoff'] == cutoff
    with np.load(path, allow_pickle=False) as z:
        needed = ['score_rows', 'score_record_ids', 'prediction', 'baseline_prediction']
        if not all(k in z for k in needed):
            return None, None
        data = {k: z[k].copy() for k in needed}
        if 'raw_candidate' in z:
            data['raw_candidate'] = z['raw_candidate'].copy()
    data['sidecar'] = sidecar
    return path, data


def validate_bundle(a, frame, cutoff):
    rows = a['score_rows']
    assert rows.ndim == 1 and rows.dtype.kind in 'iu'
    assert len(np.unique(rows)) == len(rows) and np.all((rows >= 0) & (rows < len(frame)))
    cycles = frame.cycle_code.to_numpy()[rows]
    assert np.all((cycles > cutoff) & (cycles <= min(cutoff + 12, 79)))
    expected = np.flatnonzero((frame.cycle_code > cutoff) & (frame.cycle_code <= min(cutoff + 12, 79)))
    np.testing.assert_array_equal(rows, expected)
    np.testing.assert_array_equal(a['score_record_ids'].astype(str), frame.record_id.to_numpy()[rows].astype(str))
    for name in ('prediction', 'baseline_prediction'):
        assert a[name].shape == rows.shape and np.isfinite(a[name]).all()
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    registry = json.loads(REGISTRY.read_text(encoding='utf-8'))
    inputs = {str(REGISTRY.relative_to(PROJECT_ROOT)): sha(REGISTRY)}
    artifact_checks, frames, raw_checks = [], {}, {}
    target_names = {'classification': 'artifact_signal', 'regression': 'threat_rank'}
    expected_hashes = {'classification': '875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c',
                       'regression': '756523ff8156a9cd776e52249f7e2a8f23dcec432b8d10570e55ff6db0e0f082'}
    for task, target in target_names.items():
        path = task_files(task)['train']  # resolver inventories names; only Train is opened
        digest = sha(path)
        assert digest == expected_hashes[task]
        f = pd.read_csv(path, usecols=['record_id', 'cycle_code', 'signal_stream', 'entry_friction', target])
        assert len(f) == 50608 and f.record_id.is_unique and f.cycle_code.between(47, 79).all()
        f['missing'] = f.signal_stream.isna() | f.signal_stream.fillna('').str.strip().eq('')
        frames[task] = f
        raw_checks[task] = {'sha256': digest, 'rows': len(f), 'missing_stream': int(f.missing.sum())}
    for c in registry['cards']:
        for role, digest in c.get('artifact_sha256', {}).items():
            rel = c['artifacts'].get(role)
            if rel:
                actual = sha(PROJECT_ROOT / rel)
                artifact_checks.append({'card': c['card_id'], 'role': role, 'path': rel,
                                        'match': actual == digest})
                inputs[rel] = actual
        freeze = c.get('freeze') or {}
        if freeze.get('path'):
            actual = sha(PROJECT_ROOT / freeze['path'])
            artifact_checks.append({'card': c['card_id'], 'role': 'freeze', 'path': freeze['path'],
                                    'match': actual == freeze['file_sha256']})
            inputs[freeze['path']] = actual
    assert all(x['match'] for x in artifact_checks)
    results, pairs, metric_checks = {}, {}, []
    for card in registry['cards']:
        cid, task = card['card_id'], card['task']
        if card['evaluation_profile'] != 'STANDARD':
            continue
        f = frames[task]
        yall = f[target_names[task]].to_numpy()
        entries = []
        for cutoff in (55, 67):
            path, a = load_bundle(cid, cutoff)
            if a is None:
                break
            inputs[str(path.relative_to(PROJECT_ROOT))] = sha(path)
            inputs[str(path.with_suffix('.json').relative_to(PROJECT_ROOT))] = sha(path.with_suffix('.json'))
            assert a['sidecar']['identity']['freeze_signature'] == card['freeze']['signature']
            rows = validate_bundle(a, f, cutoff)
            np.testing.assert_array_equal(a['sidecar']['identity']['score_rows'], rows)
            assert a['sidecar']['metadata']['fit_rows'] == int((f.cycle_code <= cutoff).sum())
            if cid in ('C16', 'C17', 'C11', 'R22', 'R06', 'R12'):
                with np.load(path, allow_pickle=False) as z:
                    if task == 'classification':
                        candidate = .75*z['control__c2b_probability'] + .25*z['raw_candidate']
                        candidate = np.where(f.missing.to_numpy()[rows], z['control__cb_probability'], candidate)
                    else:
                        raw = z['raw_candidate']
                        cycle = f.cycle_code.to_numpy()[rows]
                        percentile = np.empty(len(rows))
                        for cy in np.unique(cycle):
                            m = cycle == cy
                            percentile[m] = 100*rankdata(raw[m], method='average')/m.sum()
                        component = percentile if cid == 'R12' else np.clip(.5*raw+.5*percentile, 0, 100)
                        candidate = np.clip(.75*z['baseline_prediction']+.25*component, 0, 100)
                    np.testing.assert_allclose(candidate, a['prediction'], atol=1e-12, rtol=0)
            entries.append((rows, a))
        if len(entries) != 2:
            continue
        rows = np.concatenate([e[0] for e in entries])
        assert len(np.unique(rows)) == len(rows) == 38588
        p = np.concatenate([e[1]['prediction'] for e in entries])
        b = np.concatenate([e[1]['baseline_prediction'] for e in entries])
        y, cycles = yall[rows], f.cycle_code.to_numpy()[rows]
        masks = {'H12': np.ones(len(rows), dtype=bool), 'H12_E': cycles <= 67, 'H12_L': cycles >= 68,
                 'W1': cycles <= 59, 'W4': (cycles >= 68) & (cycles <= 71)}
        masks['primary'] = masks['W1'] | masks['W4']
        masks['H12_without_primary'] = ~masks['primary']
        for start in (56, 60, 64, 68, 72, 76):
            # These are bins from the TWO pilot fits, NOT six-origin forecasts.
            masks[f'pilot_bin_{start}_{start+3}'] = (cycles >= start) & (cycles <= start+3)
        metrics = {name: contrast(y[m], p[m], b[m], task) for name, m in masks.items()}
        for name, old in (('primary', card['primary']), ('H12', card['h12'])):
            if card['evaluation_profile'] == 'STANDARD':
                diff = abs(metrics[name]['gain'] - old['gain_positive_is_better'])
                metric_checks.append({'card': cid, 'view': name, 'absolute_gain_difference': diff,
                                      'match_rounded_ledger': diff <= 5.1e-7,
                                      'denominator_match': metrics[name]['n'] == old['rows']})
        groups = {'present': ~f.missing.to_numpy()[rows], 'missing': f.missing.to_numpy()[rows]}
        if task == 'regression':
            groups.update(low_le5=y <= 5, high_ge95=y >= 95)
        group_metrics = {name: {g: contrast(y[m & gm], p[m & gm], b[m & gm], task)
                               for g, gm in groups.items()} for name, m in masks.items()}
        cycle_metrics = {str(c): contrast(y[cycles == c], p[cycles == c], b[cycles == c], task)
                         for c in np.unique(cycles)}
        extra = {}
        if cid in ('C11', 'C19'):
            extra['ranking_and_flips'] = {}
            for name in ('H12_E', 'H12_L'):
                m = masks[name]
                yy, pp, bb = y[m], p[m], b[m]
                flipped = (pp >= .5) != (bb >= .5)
                extra['ranking_and_flips'][name] = {
                    'candidate_auc': float(roc_auc_score(yy, pp)),
                    'baseline_auc': float(roc_auc_score(yy, bb)),
                    'candidate_ap': float(average_precision_score(yy, pp)),
                    'baseline_ap': float(average_precision_score(yy, bb)),
                    'labels_flipped': int(flipped.sum()),
                    'flips_correcting_error': int(np.sum(flipped & ((pp >= .5) == yy))),
                    'flips_creating_error': int(np.sum(flipped & ((pp >= .5) != yy))),
                    'candidate_confusion_tn_fp_fn_tp': sufficient(yy, pp, task).astype(int).tolist(),
                    'baseline_confusion_tn_fp_fn_tp': sufficient(yy, bb, task).astype(int).tolist(),
                }
        if cid in ('C16', 'C17', 'C11', 'C12', 'C19', 'R22', 'R06', 'R21', 'R12', 'R11'):
            extra['block_sensitivity'] = {
                name: circular_block_interval(y[m], p[m], b[m], cycles[m], task)
                for name, m in masks.items() if name in ('H12_E', 'H12_L')}
        if task == 'regression':
            # Diagnostic only: post-hoc availability routing, NO weight search.
            routed = np.where(groups['missing'], b, p)
            extra['posthoc_present_only'] = {
                name: contrast(y[m], routed[m], b[m], task) for name, m in masks.items()}
            extra['posthoc_present_only_groups'] = {
                name: {g: contrast(y[m & gm], routed[m & gm], b[m & gm], task) for g, gm in groups.items()}
                for name, m in masks.items() if name in ('primary', 'W1', 'W4', 'H12', 'H12_L')}
            extra['within_cycle_spearman'] = {
                str(c): {'candidate': float(spearmanr(y[cycles == c], p[cycles == c]).statistic),
                         'baseline': float(spearmanr(y[cycles == c], b[cycles == c]).statistic)}
                for c in np.unique(cycles)}
            if cid == 'R12':
                raw = np.concatenate([e[1]['raw_candidate'] for e in entries])
                extra['raw_ranker_mean_cycle_spearman'] = {
                    name: {'raw_ranker': float(np.mean([spearmanr(y[cycles == c], raw[cycles == c]).statistic
                                                       for c in np.unique(cycles[m])])),
                           'champion': float(np.mean([spearmanr(y[cycles == c], b[cycles == c]).statistic
                                                     for c in np.unique(cycles[m])]))}
                    for name, m in masks.items() if name in ('H12_E', 'H12_L')}
                # Matched negative control: same mapping/blend/route, no ranker.
                baseline_percentile = np.empty(len(b))
                for cy in np.unique(cycles):
                    m = cycles == cy
                    baseline_percentile[m] = 100*rankdata(b[m], method='average')/m.sum()
                calibration_only = np.where(groups['missing'], b, .75*b+.25*baseline_percentile)
                extra['posthoc_no_ranker_matched_control'] = {
                    name: contrast(y[m], calibration_only[m], b[m], task) for name, m in masks.items()}
                extra['posthoc_ranker_vs_no_ranker_gain'] = {
                    name: contrast(y[m], routed[m], calibration_only[m], task) for name, m in masks.items()}
                extra['posthoc_no_ranker_groups'] = {
                    name: {g: contrast(y[m & gm], calibration_only[m & gm], b[m & gm], task)
                           for g, gm in groups.items()} for name, m in masks.items()
                    if name in ('primary', 'W1', 'W4', 'H12', 'H12_L')}
        results[cid] = {'task': task, 'metrics': metrics, 'groups': group_metrics,
                        'per_cycle': cycle_metrics, **extra}
        pairs[cid] = (rows, p, b)
    assert all(x['match_rounded_ledger'] and x['denominator_match'] for x in metric_checks)
    correlations = []
    for left, right in [('C16', 'C17'), ('C16', 'C11'), ('R22', 'R06'), ('R22', 'R21'), ('R12', 'R22')]:
        lr, lp, lb = pairs[left]
        rr, rp, rb = pairs[right]
        np.testing.assert_array_equal(lr, rr)
        np.testing.assert_array_equal(lb, rb)
        correlations.append({'left': left, 'right': right,
                             'delta_prediction_pearson': float(np.corrcoef(lp-lb, rp-rb)[0, 1])})
    reg = frames['regression']
    target_distribution = []
    for cycle, frame in reg.groupby('cycle_code'):
        y = frame.threat_rank.to_numpy()
        grid = 100 * rankdata(y, method='average') / len(y)
        target_distribution.append({'cycle': int(cycle), 'n': len(y), 'mean': float(y.mean()),
                                    'std_population': float(y.std()),
                                    'rank_grid_max_absolute_error': float(np.max(np.abs(y-grid)))})
    review = {'status': 'POSTHOC_REVIEW_ONLY_CAMPAIGN_REMAINS_PAUSED',
              'source_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT_ROOT, text=True).strip(),
              'raw_train': raw_checks, 'artifact_hash_checks': artifact_checks,
              'metric_checks': metric_checks, 'cards_recomputed': sorted(results),
              'coverage_note': 'STANDARD cached predictions; C09/C20 and LATE rely on linked historical reports, not silently counted as recomputed',
              'results': results, 'delta_correlations': correlations,
              'regression_target_distribution': target_distribution,
              'input_hashes': inputs, 'review_script_sha256': sha(__file__),
              'training_fits': 0, 'test_files_opened': 0, 'kaggle_calls': 0}
    (OUT / 'diagnostics.json').write_text(json.dumps(review, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    rows = []
    for cid, item in results.items():
        for view, stats in item['metrics'].items():
            rows.append({'card': cid, 'task': item['task'], 'view': view, **stats})
    pd.DataFrame(rows).to_csv(OUT / 'view_metrics.csv', index=False)
    plot(results)
    print(json.dumps({'recomputed_cards': len(results), 'artifact_checks': len(artifact_checks),
                      'metric_checks': len(metric_checks), 'all_pass': True,
                      'selected': {c: results[c]['metrics'] for c in ('C16','C11','R22','R06')},
                      'correlations': correlations}, indent=2))


def plot(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, ids, factor, label in [(axes[0], ['C16','C17','C11','C12','C19'], 100, 'Gain Macro F1 (poin persentase)'),
                                   (axes[1], ['R22','R06','R21','R11','R12'], 1, 'Perbaikan MAE')]:
        for j, name in enumerate(['H12_E','H12_L']):
            vals = [results[c]['metrics'][name]['gain'] * factor for c in ids]
            ax.bar(np.arange(len(ids))+(j-.5)*.36, vals, width=.36, label=name)
        ax.set_xticks(np.arange(len(ids)), ids)
        ax.axhline(0, color='black', lw=.8)
        ax.set_ylabel(label); ax.legend()
    fig.suptitle('Gain pilot menurut horizon: early vs late\nPost hoc; dua cutoff dan hasil model yang sudah dipilih')
    fig.tight_layout(); fig.savefig(OUT/'01_horizon_gains.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for ax, ids, factor, label in [(axes[0], ['C16','C11'], 100, 'Gain Macro F1 (pp)'),
                                  (axes[1], ['R22','R06','R12'], 1, 'Perbaikan MAE')]:
        for c in ids:
            cycles = sorted(map(int, results[c]['per_cycle']))
            ax.plot(cycles, [results[c]['per_cycle'][str(k)]['gain']*factor for k in cycles], marker='.', label=c)
        ax.axhline(0, color='black', lw=.8); ax.axvline(67.5, color='grey', ls='--')
        ax.set_ylabel(label); ax.legend(ncol=3)
    axes[1].set_xlabel('Cycle evaluasi (fit <=55 untuk 56–67; fit <=67 untuk 68–79)')
    fig.suptitle('Sebaran gain per cycle dari bundle pilot; bukan hasil continuation')
    fig.tight_layout(); fig.savefig(OUT/'02_cycle_gains.png', dpi=170); plt.close(fig)
    ids, groups = ['R22','R06','R21','R12','R01','R14'], ['present','missing','low_le5','high_ge95']
    a = np.array([[results[c]['groups']['H12'][g]['gain'] for g in groups] for c in ids])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    im = ax.imshow(a, cmap='RdBu', vmin=-1.05, vmax=1.05, aspect='auto')
    ax.set_xticks(range(4), ['Stream present','Stream missing','Target <=5','Target >=95'])
    ax.set_yticks(range(len(ids)), ids)
    for i in range(len(ids)):
        for j in range(4):
            ax.text(j, i, f'{a[i,j]:+.3f}', ha='center', va='center', color='white' if abs(a[i,j])>.6 else 'black')
    fig.colorbar(im, ax=ax, label='Perbaikan MAE; positif lebih baik')
    ax.set_title('Gain subgroup H12: agregat yang baik dapat menutupi harm\nKelompok target hanya untuk evaluasi; bukan aturan routing')
    fig.tight_layout(); fig.savefig(OUT/'03_regression_groups.png', dpi=170); plt.close(fig)


if __name__ == '__main__':
    main()
