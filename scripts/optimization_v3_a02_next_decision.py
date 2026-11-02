"""Train-only zero-fit matched-count diagnostic; no candidate is produced."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/optimization_v3/a02_next_decision_v1'
RUN = ROOT / 'experiments/runs/optimization_v3_a02_reliability_audit_v1'
LEDGER = ROOT / 'experiments/runs/optimization_v3_research_next_v1/budget.jsonl'
TRAIN_HASH = '875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c'
VIEWS = ('full_six_primary', 'new_origin_aggregate', 'late', 'inherited_H12_L')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_top_k(scores, rows, k):
    scores, rows = np.asarray(scores), np.asarray(rows)
    if scores.ndim != 1 or rows.shape != scores.shape or len(np.unique(rows)) != len(rows):
        raise ValueError('Unique aligned rows required')
    if not np.isfinite(scores).all() or not 0 <= k <= len(scores):
        raise ValueError('Finite scores and valid K required')
    labels = np.zeros(len(scores), dtype=np.int8)
    labels[np.lexsort((rows, -scores))[:k]] = 1
    return labels


def matched_count(frame, rank_arm, count_arm, grouping='cycle'):
    """No outcome input: only probabilities, geometry IDs and availability."""
    labels = (frame['baseline'] >= .5).astype(np.int8)
    eligible = (~frame['missing'].astype(bool)) & (frame['origin'] >= 0)
    origins = frame['origin']
    cycles = frame['cycles'] if grouping == 'cycle' else np.zeros(len(origins), dtype=int)
    for origin, cycle in sorted(set(zip(origins[eligible], cycles[eligible]))):
        mask = eligible & (origins == origin) & (cycles == cycle)
        k = int((frame[count_arm][mask] >= .5).sum())
        labels[mask] = exact_top_k(frame[rank_arm][mask], frame['rows'][mask], k)
    return labels


def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    if y.shape != p.shape or not np.isin(y, [0, 1]).all() or not np.isin(p, [0, 1]).all():
        raise ValueError('Aligned binary labels required')
    tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    tp = int(((y == 1) & (p == 1)).sum())
    f0 = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.
    return dict(n=len(y), tn=tn, fp=fp, fn=fn, tp=tp,
                positives=tp+fp, macro_f1=(f0+f1)/2)


def evaluate(frame, grouping):
    policies = {
        'B_default': (frame['baseline'] >= .5).astype(np.int8),
        'A_default': (frame['observed'] >= .5).astype(np.int8),
        'S_default': (frame['shuffled'] >= .5).astype(np.int8),
        'B_at_A_count': matched_count(frame, 'baseline', 'observed', grouping),
        'A_at_B_count': matched_count(frame, 'observed', 'baseline', grouping),
        'S_at_A_count': matched_count(frame, 'shuffled', 'observed', grouping),
    }
    assert np.array_equal(matched_count(frame, 'observed', 'observed', grouping), policies['A_default'])
    assert np.array_equal(matched_count(frame, 'baseline', 'baseline', grouping), policies['B_default'])
    assert policies['B_at_A_count'].sum() == policies['A_default'].sum() == policies['S_at_A_count'].sum()
    assert policies['A_at_B_count'].sum() == policies['B_default'].sum()
    result = {name: metrics(frame['y'], values) for name, values in policies.items()}
    f = {k: v['macro_f1'] for k, v in result.items()}
    paths = dict(
        total=f['A_default']-f['B_default'],
        rank_at_A_count=f['A_default']-f['B_at_A_count'],
        allocation_along_baseline=f['B_at_A_count']-f['B_default'],
        rank_at_B_count=f['A_at_B_count']-f['B_default'],
        allocation_along_A02=f['A_default']-f['A_at_B_count'],
        rank_vs_shuffle_at_A_count=f['A_default']-f['S_at_A_count'],
        tp_gain_at_A_count=result['A_default']['tp']-result['B_at_A_count']['tp'],
    )
    assert abs(paths['total']-paths['rank_at_A_count']-paths['allocation_along_baseline']) < 1e-12
    assert abs(paths['total']-paths['rank_at_B_count']-paths['allocation_along_A02']) < 1e-12
    return dict(metrics=result, algebraic_paths_not_causal=paths)


def run():
    before = sha(LEDGER)
    train_path = ROOT / 'data/raw/classification/Train.csv'
    assert sha(train_path) == TRAIN_HASH
    train = pd.read_csv(train_path, usecols=['cycle_code', 'artifact_signal'])
    inputs, results = {}, {}
    for view in VIEWS:
        path = RUN / f'row_level_{view}.npz'
        inputs[str(path.relative_to(ROOT))] = sha(path)
        with np.load(path, allow_pickle=False) as data:
            frame = {k: data[k] for k in ('rows', 'cycles', 'origin', 'y', 'missing', 'baseline', 'observed', 'shuffled')}
        rows = frame['rows']
        assert len(np.unique(rows)) == len(rows)
        assert np.array_equal(train.iloc[rows].artifact_signal.to_numpy(), frame['y'])
        assert np.array_equal(train.iloc[rows].cycle_code.to_numpy(), frame['cycles'])
        for arm in ('baseline', 'observed', 'shuffled'):
            assert np.isfinite(frame[arm]).all() and np.all((frame[arm] >= 0) & (frame[arm] <= 1))
        fallback = frame['missing'].astype(bool) | (frame['origin'] < 0)
        assert np.array_equal(frame['observed'][fallback], frame['baseline'][fallback])
        assert np.array_equal(frame['shuffled'][fallback], frame['baseline'][fallback])
        scopes = {'all': frame}
        if view == 'full_six_primary':
            for origin in (59, 63, 67, 71, 75):
                scopes[f'E{origin}'] = {k: a[frame['origin'] == origin] for k, a in frame.items()}
        results[view] = {scope: {g: evaluate(f, g) for g in ('cycle', 'origin')}
                         for scope, f in scopes.items()}
    assert sha(LEDGER) == before
    result = dict(status='DESCRIPTIVE_ZERO_FIT', results=results,
                  inputs=inputs, train_sha256=TRAIN_HASH, ledger_before=before, ledger_after=sha(LEDGER),
                  source_sha256=sha(Path(__file__)), plan_sha256=sha(REPORT/'PLAN.md'),
                  fits=0, test_sample_kaggle_access=False,
                  limitations=['Adaptive reused development; overlapping views.',
                               'K derives from frozen predictions, never true labels.',
                               'Batch-relative controls do not supply a deployment K.',
                               'Two algebraic paths are not unique or causal attributions.'])
    (REPORT/'matched_count_results.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    out = run()
    for view, scopes in out['results'].items():
        for scope, groups in scopes.items():
            print(view, scope, json.dumps(groups['cycle']['algebraic_paths_not_causal']))
