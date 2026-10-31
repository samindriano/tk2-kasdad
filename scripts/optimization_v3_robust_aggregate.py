"""One frozen robust aggregation test on existing past-only Regression forecasts."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import sys
import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import optimization_v3_distributional as common
from scripts import optimization_v3_reg_diagnostics as diag
from src.models.round1_evaluation import code_digest, digest_json

CONFIG = ROOT / 'experiments/optimization_v3/regression_robust_aggregate'
OUT = ROOT / 'reports/optimization_v3/regression_robust_aggregate'
CODE = ['scripts/optimization_v3_robust_aggregate.py',
        'scripts/optimization_v3_distributional.py',
        'scripts/optimization_v3_reg_diagnostics.py']
PROTOCOL = {
    'hypothesis': 'A coordinatewise median may limit an extreme component forecast better than a weighted mean.',
    'component_inputs': 'Original mapped LGB SVD32, mapped Hybrid Ridge, and independently rank50-mapped sparse LGB.',
    'policies': {'median3': 'median of three mapped component forecasts',
                 'median25': '0.75 fixed current champion + 0.25 median3'},
    'champion': '0.5625 mapped LGB + 0.1875 mapped Ridge + 0.25 mapped sparse LGB',
    'gate': {'minimum_primary_MAE_gain': .03, 'maximum_H12_MAE_harm': .05,
             'minimum_nonnegative_windows': 4, 'maximum_supported_tail_or_missing_harm': .25,
             'minimum_subgroup_support': 100},
    'new_model_fits': 0, 'Test_read': False, 'Kaggle_access': False,
    'selection_caveat': 'Reused adaptive Train development outcomes, not an untouched holdout. No aggregation weight search.'
}


def identity():
    _, train = common.load()
    inputs = [train, diag.PRIMARY_CACHE, diag.PILOT_CACHE, diag.FULL_CACHE]
    return {'protocol': PROTOCOL,
            'source_sha256': {p: code_digest(ROOT / p) for p in CODE},
            'input_sha256': {p.relative_to(ROOT).as_posix(): common.sha(p) for p in inputs}}


def freeze():
    current = identity()
    assert not OUT.exists(), 'No results may precede the freeze'
    common.save(CONFIG / 'protocol.json', PROTOCOL)
    common.save(CONFIG / 'freeze.json', {'at_utc': datetime.now(timezone.utc).isoformat(),
                'identity': current, 'signature': digest_json(current)})
    print('FROZEN', digest_json(current), flush=True)


def rank50(raw, cycles):
    percentile = np.empty(len(raw), dtype=float)
    for cycle in np.unique(cycles):
        mask = cycles == cycle
        percentile[mask] = 100 * rankdata(raw[mask], method='average') / mask.sum()
    return np.clip(.5 * raw + .5 * percentile, 0, 100)


def aggregate(components):
    components = np.asarray(components, dtype=float)
    assert components.ndim == 2 and components.shape[0] == 3 and np.isfinite(components).all()
    assert ((components >= 0) & (components <= 100)).all()
    champion = .5625 * components[0] + .1875 * components[1] + .25 * components[2]
    median = np.median(components, axis=0)
    return {'champion': champion, 'median3': median, 'median25': .75 * champion + .25 * median}


def run():
    frozen = common.read(CONFIG / 'freeze.json')
    assert frozen['identity'] == identity() and frozen['signature'] == digest_json(identity())
    assert not (OUT / 'results.json').exists(), 'Preserve previous results'
    frame, _ = common.load()
    primary_parts, hparts, views = [], [], {}
    with np.load(diag.PRIMARY_CACHE, allow_pickle=False) as base, \
         np.load(diag.PILOT_CACHE, allow_pickle=False) as pilot, \
         np.load(diag.FULL_CACHE, allow_pickle=False) as full:
        def record(end, high):
            cache = pilot if end in (55, 67) else full
            rows_all = cache[f'rows_lgbm_sparse_l1_{end}']
            np.testing.assert_array_equal(rows_all, base[f'rows_{end}'])
            cyc = frame.iloc[rows_all].cycle_code.to_numpy()
            use = (cyc > end) & (cyc <= high)
            rows = rows_all[use]
            np.testing.assert_array_equal(rows, np.flatnonzero(frame.cycle_code.between(end + 1, high)))
            # Existing LGB/Ridge vectors were already mapped using the frozen
            # inner-selected rank policies. Do not remap them a second time.
            lgb, ridge = base[f'lgb_{end}'][use], base[f'ridge_{end}'][use]
            np.testing.assert_allclose(base[f'blend_ridge25_{end}'][use], .75 * lgb + .25 * ridge,
                                       atol=1e-10, rtol=0)
            sparse = rank50(cache[f'raw_lgbm_sparse_l1_{end}'][use], cyc[use])
            predictions = aggregate(np.vstack([lgb, ridge, sparse]))
            specs = [(end, end + 1, high)]
            incumbent, _ = diag._build_view(frame, base, pilot, full, specs)
            np.testing.assert_array_equal(rows, incumbent.row.to_numpy())
            np.testing.assert_allclose(predictions['champion'], incumbent.prediction.to_numpy(), atol=1e-10, rtol=0)
            return rows, predictions

        def summarize(rows, predictions):
            return {name: common.metrics(frame, rows, values) for name, values in predictions.items()}

        for i, end in enumerate(diag.ENDS, 1):
            rows, pred = record(end, end + 4)
            primary_parts.append((rows, pred))
            views[f'W{i}'] = summarize(rows, pred)
        for name, end, high in [('H12_E', 55, 67), ('H12_L', 67, 79)]:
            rows, pred = record(end, high)
            hparts.append((rows, pred))
            views[name] = summarize(rows, pred)
        for name, parts in [('primary', primary_parts), ('H12', hparts)]:
            rows = np.concatenate([p[0] for p in parts])
            assert len(rows) == 38588 and len(np.unique(rows)) == len(rows)
            pred = {kind: np.concatenate([p[1][kind] for p in parts]) for kind in ['champion', 'median3', 'median25']}
            views[name] = summarize(rows, pred)
    gates = {}
    for name in ['median3', 'median25']:
        primary_gain = views['primary']['champion']['mae'] - views['primary'][name]['mae']
        h12_harm = views['H12'][name]['mae'] - views['H12']['champion']['mae']
        wins = sum(views[f'W{i}'][name]['mae'] <= views[f'W{i}']['champion']['mae'] for i in range(1, 7))
        harms = []
        for view in views.values():
            for key in ['tail_le5', 'tail_ge95', 'stream_missing']:
                a, b = view[name]['subgroups'][key], view['champion']['subgroups'][key]
                if a['n'] >= 100:
                    harms.append(a['mae'] - b['mae'])
        gates[name] = {'primary_gain': primary_gain, 'H12_harm': h12_harm, 'windows_nonnegative': wins,
                       'maximum_supported_tail_or_missing_harm': max(harms),
                       'pass': bool(primary_gain >= .03 and h12_harm <= .05 and wins >= 4 and max(harms) <= .25)}
    result = {'status': 'CHECKED', 'freeze_signature': frozen['signature'], 'new_fits': 0,
              'Test_read': False, 'Kaggle_access': False, 'views': views, 'gates': gates}
    common.save(OUT / 'results.json', result)
    print(json.dumps({'gates': gates, 'primary': {k: v['mae'] for k, v in views['primary'].items()},
                      'H12': {k: v['mae'] for k, v in views['H12'].items()}}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    freeze() if args.freeze else run()
