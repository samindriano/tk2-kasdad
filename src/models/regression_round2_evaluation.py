"""Frozen feature/history regression development experiments; no test scoring."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import subprocess
import tempfile
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.data.audit import sha256, write_json
from src.models.round1_evaluation import code_digest, digest_json, array_digest, verify_freeze as verify_round1
from src.models.regression_round2_models import build_model, _tree_complexity, _tree_digest
from src.models.regression_round2_metrics import summarize, paired, history_summary
from src.utils.paths import PROJECT_ROOT, task_files

ROOT = PROJECT_ROOT/'experiments/regression_round2'
REPORT = PROJECT_ROOT/'reports/regression_round2'
DEFAULT_RUN = 'regression_round2_v1'
CORE = ['src/models/regression_round2_models.py', 'src/models/regression_round2_metrics.py',
        'src/models/regression_round2_evaluation.py', 'scripts/run_regression_round2.py']
TEXT_EXTENSIONS = {'.py', '.md', '.json', '.txt', '.ipynb', '.toml', '.yaml', '.yml', '.csv', ''}


def load_protocol():
    return json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))


def positions_digest(rows):
    return hashlib.sha256(np.asarray(rows, dtype='<i8').tobytes()).hexdigest()


def protected_digest(path):
    return code_digest(path) if path.suffix.lower() in TEXT_EXTENSIONS or path.name.startswith('.') else sha256(path)


def identity(data_dir=None):
    old = verify_round1(data_dir)
    return {'protocol_sha256': digest_json(load_protocol()), 'round1_signature': old['signature'],
            'round1_identity': old['identity'], 'code_sha256': {p: code_digest(PROJECT_ROOT/p) for p in CORE},
            'python': platform.python_version(),
            'packages': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'threadpoolctl']}}


def freeze_protocol(data_dir=None):
    current = identity(data_dir)
    target = ROOT/'freeze.json'
    if target.exists():
        return verify_freeze(data_dir)
    tracked = subprocess.check_output(['git', '-C', str(PROJECT_ROOT), 'ls-files', '-z']).decode().split('\0')
    protected = {p: protected_digest(PROJECT_ROOT/p) for p in tracked if p and p != 'README.md'}
    receipt = {'status': 'FROZEN_BEFORE_FIRST_REGRESSION_ROUND2_FIT',
               'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
               'base_git_head': subprocess.check_output(['git', '-C', str(PROJECT_ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
               'identity': current, 'signature': digest_json(current), 'protected_checkpoint_files': protected}
    # Exclusive creation prevents accidental replacement of the original freeze.
    with target.open('x', encoding='utf-8') as handle:
        json.dump(receipt, handle, indent=2, allow_nan=False)
        handle.write('\n')
    return receipt


def verify_freeze(data_dir=None):
    receipt = json.loads((ROOT/'freeze.json').read_text(encoding='utf-8'))
    if receipt['identity'] != identity(data_dir) or receipt['signature'] != digest_json(receipt['identity']):
        raise RuntimeError('Frozen regression Round2 input/code/runtime/protocol mismatch.')
    for rel, expected in receipt['protected_checkpoint_files'].items():
        if protected_digest(PROJECT_ROOT/rel) != expected:
            raise RuntimeError('Protected checkpoint changed: '+rel)
    return receipt


def safe_artifact(run_id):
    if not re.fullmatch(r'regression_round2_[A-Za-z0-9_-]+', run_id):
        raise ValueError('Use a dedicated regression_round2_ run ID.')
    root = (PROJECT_ROOT/'experiments/runs').resolve()
    path = (root/run_id).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError('Run directory escaped owned root.')
    return path


def safe_report(path=None):
    path = Path(path).resolve() if path is not None else REPORT.resolve()
    allowed = [REPORT.resolve(), (Path(tempfile.gettempdir())/'tp2-regression-round2-repeat').resolve()]
    if not any(path == base or path.is_relative_to(base) for base in allowed):
        raise ValueError('Report output must stay in regression Round2 or its dedicated temporary repeat root.')
    return path


def select_history(frame, end, policy):
    cycles = frame.cycle_code.to_numpy()
    past = np.flatnonzero((cycles >= 47) & (cycles <= end))
    unique = np.unique(cycles[past])
    if not len(past) or unique[-1] != end:
        raise ValueError('Fit cutoff not present in source.')
    if policy['kind'] == 'expanding':
        return past
    k = policy.get('cycles', policy.get('match_cycles'))
    recent = past[np.isin(cycles[past], unique[-k:])]
    if policy['kind'] == 'recent':
        return recent
    if policy['kind'] == 'matched':
        rng = np.random.default_rng(np.random.SeedSequence([policy['seed'], int(end)]))
        return np.sort(rng.choice(past, size=len(recent), replace=False))
    raise ValueError('Unknown history policy.')


def load_interaction_selection(data_dir=None):
    frozen = verify_freeze(data_dir)
    selection = json.loads((ROOT/'interaction_selection.json').read_text(encoding='utf-8'))
    payload = selection['payload']
    if selection['signature'] != digest_json(payload) or payload['freeze_signature'] != frozen['signature']:
        raise RuntimeError('Interaction selection signature mismatch.')
    if payload['base_results_sha256'] != sha256(REPORT/'results_base.json'):
        raise RuntimeError('Interaction selection no longer matches base evidence.')
    protocol = load_protocol()
    recipes = payload['recipes']
    contract = protocol['interaction_contract']
    if len(recipes) > contract['maximum_total_recipes'] or len({r['features'] for r in recipes}) > 2:
        raise ValueError('Interaction selection exceeds preregistered budget.')
    ids = []
    for r in recipes:
        if set(r) != {'id', 'features', 'history', 'control', 'stage'} or r['control'] is not None or r['stage'] != 'interaction':
            raise ValueError('Invalid interaction recipe schema.')
        if r['features'] not in contract['eligible_feature_sets'] or r['history'] not in contract['eligible_histories']:
            raise ValueError('Unregistered interaction.')
        if r['id'] != 'C_'+r['features'].upper()+'_'+r['history'].upper():
            raise ValueError('Interaction ID does not identify its exact recipe.')
        ids.append(r['id'])
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate interaction recipe.')
    return selection


def freeze_interactions(recipes, rationale, data_dir=None):
    frozen = verify_freeze(data_dir)
    target = ROOT/'interaction_selection.json'
    if target.exists():
        raise RuntimeError('Interaction selection already exists; never overwrite after outcomes.')
    if not rationale.strip():
        raise ValueError('Numerical selection rationale is required.')
    base = json.loads((REPORT/'results_base.json').read_text(encoding='utf-8'))
    if base['freeze_signature'] != frozen['signature'] or base['partial']:
        raise RuntimeError('Complete base evidence required before interaction selection.')
    payload = {'freeze_signature': frozen['signature'], 'base_results_sha256': sha256(REPORT/'results_base.json'),
               'selected_at_utc': datetime.now(timezone.utc).isoformat(), 'recipes': recipes,
               'rationale': rationale, 'interpretation': 'Adaptive development interaction; not independent confirmation.'}
    # Validate all recipe constraints before exclusive creation.
    contract = load_protocol()['interaction_contract']
    if len(recipes) > 4 or len({r['features'] for r in recipes}) > 2:
        raise ValueError('Too many interaction recipes.')
    for r in recipes:
        if r['features'] not in contract['eligible_feature_sets'] or r['history'] not in contract['eligible_histories']:
            raise ValueError('Unregistered interaction recipe.')
        if r != {'id': 'C_'+r['features'].upper()+'_'+r['history'].upper(), 'features': r['features'],
                 'history': r['history'], 'control': None, 'stage': 'interaction'}:
            raise ValueError('Invalid interaction recipe schema.')
    if len({r['id'] for r in recipes}) != len(recipes):
        raise ValueError('Duplicate interaction recipe.')
    with target.open('x', encoding='utf-8') as handle:
        json.dump({'payload': payload, 'signature': digest_json(payload)}, handle, indent=2, allow_nan=False)
        handle.write('\n')
    return load_interaction_selection(data_dir)


def _state(model):
    state = model.audit_state()
    state.pop('prediction_diagnostics', None)
    return state


def _complexity(model):
    estimator = model._delegate._estimator if model._delegate is not None else model._estimator
    return {**_tree_complexity(estimator), 'fitted_tree_sha256': _tree_digest(estimator)}


def run_round2(data_dir=None, run_id=DEFAULT_RUN, report_dir=None, stage='base', only_fit_end=None, force=False):
    frozen = verify_freeze(data_dir)
    protocol = load_protocol()
    artifact, reports = safe_artifact(run_id), safe_report(report_dir)
    if run_id != DEFAULT_RUN and reports == REPORT.resolve():
        raise ValueError('Custom run IDs require a separate report directory.')
    if run_id == DEFAULT_RUN and reports != REPORT.resolve():
        raise ValueError('Canonical run ID is bound to its canonical report root.')
    if only_fit_end is not None or force:
        if run_id == DEFAULT_RUN or reports == REPORT.resolve():
            raise ValueError('Partial/forced runs require a separate run ID and report directory.')
    if stage not in {'base', 'all'}:
        raise ValueError('Stage must be base or all.')
    if stage == 'base' and reports == REPORT.resolve() and (reports/'results_base.json').exists():
        old = json.loads((reports/'results_base.json').read_text(encoding='utf-8'))
        if old['freeze_signature'] != frozen['signature'] or old['partial']:
            raise RuntimeError('Existing base evidence differs.')
        print('Preserving immutable completed base result; use verifier for vector reconciliation.', flush=True)
        return old
    recipes = list(protocol['recipes'])
    selection = None
    if stage == 'all':
        selection = load_interaction_selection(data_dir)
        recipes += selection['payload']['recipes']
    windows = [('primary', w) for w in protocol['primary_windows']]+[('secondary', w) for w in protocol['secondary_windows']]
    if only_fit_end is not None:
        windows = [(f, w) for f, w in windows if w['fit'][1] == only_fit_end]
        if not windows:
            raise ValueError('Unknown representative fit cutoff.')
    frame = pd.read_csv(task_files('regression', data_dir)['train'])
    if not frame.record_id.is_unique or frame.record_id.isna().any() or not np.isfinite(frame.threat_rank).all() or not frame.threat_rank.between(0, 100).all():
        raise ValueError('Official Train identity/target invalid.')
    artifact.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    start_all = perf_counter()
    fresh = reused = 0
    predictions, audits, jobs = {}, {}, []
    histories = {}
    # One fit per source subset/features, reused across overlapping H12 windows.
    for end in sorted({w['fit'][1] for _, w in windows}):
        last = max(w['validation'][1] for _, w in windows if w['fit'][1] == end)
        val = np.flatnonzero(frame.cycle_code.between(end+1, last))
        local = {}
        audits[str(end)] = {}
        for r in recipes:
            fit = select_history(frame, end, protocol['history_policies'][r['history']])
            if frame.iloc[fit].cycle_code.max() >= frame.iloc[val].cycle_code.min() or np.intersect1d(fit, val).size:
                raise RuntimeError('Temporal leakage.')
            features = protocol['feature_sets'][r['features']]
            key = digest_json({'freeze': frozen['signature'], 'features': features, 'control': r['control'],
                               'fit_rows': positions_digest(fit), 'validation_rows': positions_digest(val)})
            job_id = f'fit{end}__{r["id"]}'
            meta_path, vector_path = artifact/(job_id+'.json'), artifact/(job_id+'.npz')
            cached = False
            if meta_path.exists() != vector_path.exists():
                raise RuntimeError('Incomplete cache pair; preserve and inspect before resuming: '+job_id)
            if key in local:
                meta, pred = local[key]
                reused += 1
                cached = True
            elif not force and meta_path.exists() and vector_path.exists():
                meta = json.loads(meta_path.read_text(encoding='utf-8'))
                with np.load(vector_path, allow_pickle=False) as saved:
                    pred = saved['prediction'].copy()
                    if not np.array_equal(saved['validation_rows'], val) or not np.array_equal(saved['fit_rows'], fit):
                        raise RuntimeError('Cached source positions mismatch: '+job_id)
                if (meta['job_key'] != key or meta['prediction_sha256'] != array_digest(pred)
                        or meta['fitted_state_sha256'] != digest_json(meta['fitted_state'])
                        or meta['fit_positions_sha256'] != positions_digest(fit)
                        or meta['validation_positions_sha256'] != positions_digest(val)
                        or meta['fit_rows'] != len(fit) or meta['validation_rows'] != len(val)
                        or meta['fitted_state']['fit_rows'] != len(fit)):
                    raise RuntimeError('Cached identity mismatch: '+job_id)
                reused += 1
                cached = True
            else:
                model = build_model(features=features, control=r['control'])
                started = datetime.now(timezone.utc).isoformat()
                start = perf_counter()
                with threadpool_limits(limits=protocol['thread_limit']):
                    model.fit(frame.iloc[fit][features].copy(), frame.iloc[fit].threat_rank.to_numpy(copy=True))
                    state = _state(model)
                    pred = np.asarray(model.predict(frame.iloc[val][features].copy()), dtype=float)
                if digest_json(state) != digest_json(_state(model)):
                    raise RuntimeError('Predict mutated fitted state.')
                meta = {'job_key': key, 'fit_started_at_utc': started, 'fit_seconds': perf_counter()-start,
                        'fit_rows': len(fit), 'validation_rows': len(val), 'fit_positions_sha256': positions_digest(fit),
                        'validation_positions_sha256': positions_digest(val),
                        'fitted_state': state, 'fitted_state_sha256': digest_json(state), 'complexity': _complexity(model),
                        'prediction_sha256': array_digest(pred), 'prediction_diagnostics': model.prediction_diagnostics(),
                        'validation_labels_passed_to_fit': False, 'early_stopping_validation_used': False}
                fresh += 1
            if pred.shape != (len(val),) or not np.isfinite(pred).all():
                raise RuntimeError('Invalid cached/fitted prediction: '+job_id)
            # Alias cache contains same immutable vectors, with an identical fit key.
            if not meta_path.exists() or force:
                np.savez_compressed(vector_path, fit_rows=fit, validation_rows=val, prediction=pred)
                write_json(meta_path, meta)
            local[key] = (meta, pred)
            predictions[(end, r['id'])] = (val, pred)
            audits[str(end)][r['id']] = meta
            history_id = str(end)+'__'+r['history']
            if history_id not in histories:
                histories[history_id] = history_summary(frame, fit, val)
            print(f'{job_id}: {"cache/alias" if cached else "fresh fit"}', flush=True)
    for family, w in windows:
        end = w['fit'][1]
        for r in recipes:
            rows, pred = predictions[(end, r['id'])]
            mask = frame.iloc[rows].cycle_code.between(*w['validation']).to_numpy()
            rows, pred = rows[mask], pred[mask]
            meta = audits[str(end)][r['id']]
            stats = summarize(frame.iloc[rows], pred, include_impact=False)
            jobs.append({'model': r['id'], 'family': family, 'window': w['id'], 'fit_end': end,
                         'features': r['features'], 'history': r['history'], 'stage': r['stage'],
                         'fit_rows': meta['fit_rows'], 'fit_seconds': meta['fit_seconds'],
                         'complexity': meta['complexity'], 'prediction_sha256': array_digest(pred),
                         'validation_positions_sha256': positions_digest(rows), **stats})
    aggregate = {}
    comparisons = []
    for family in ['primary', 'secondary']:
        aggregate[family] = {}
        aligned = {}
        for r in recipes:
            pairs = []
            for f, w in windows:
                if f != family:
                    continue
                rows, pred = predictions[(w['fit'][1], r['id'])]
                mask = frame.iloc[rows].cycle_code.between(*w['validation']).to_numpy()
                pairs.append((rows[mask], pred[mask]))
            if not pairs:
                continue
            rows, pred = np.concatenate([v[0] for v in pairs]), np.concatenate([v[1] for v in pairs])
            order = np.argsort(rows)
            rows, pred = rows[order], pred[order]
            if len(np.unique(rows)) != len(rows):
                raise RuntimeError('Duplicate records within family.')
            aligned[r['id']] = (rows, pred)
            aggregate[family][r['id']] = summarize(frame.iloc[rows], pred)
        for r in recipes:
            if r['id'] not in aligned:
                continue
            rows, pred = aligned[r['id']]
            for control in ['R1_C', 'R1_D']:
                ref_rows, ref = aligned[control]
                if not np.array_equal(rows, ref_rows):
                    raise RuntimeError('Unpaired comparison.')
                delta = paired(frame.iloc[rows], pred, ref)
                a = [j for j in jobs if j['family'] == family and j['model'] == r['id']]
                b = {j['window']: j for j in jobs if j['family'] == family and j['model'] == control}
                delta['window_deltas'] = [{'window': j['window'], 'delta_mae': j['raw']['mae']-b[j['window']]['raw']['mae']} for j in a]
                delta['windows_improved'] = sum(d['delta_mae'] < -1e-10 for d in delta['window_deltas'])
                comparisons.append({'family': family, 'candidate': r['id'], 'control': control, **delta})
    result = {'status': 'REGRESSION_ROUND2_DEVELOPMENT_REVIEW_NO_FINAL_MODEL', 'run_id': run_id,
              'stage': stage, 'partial': only_fit_end is not None, 'freeze_signature': frozen['signature'],
              'interaction_signature': selection['signature'] if selection else None,
              'recipes': recipes, 'fresh_fits': fresh, 'reused_fits_or_aliases': reused,
              'unique_fit_keys': len({m['job_key'] for a in audits.values() for m in a.values()}),
              'elapsed_seconds_this_invocation': perf_counter()-start_all, 'jobs': jobs, 'aggregate': aggregate,
              'comparisons': comparisons, 'history_support': histories,
              'boundary': 'Adaptive development; primary and H12 reuse records; no official-test predictions or Kaggle selection.'}
    if stage == 'all' and run_id == DEFAULT_RUN:
        result['base_run_execution'] = json.loads((REPORT/'results_base.json').read_text(encoding='utf-8'))['execution_receipt']
    result['execution_receipt'] = {k: result[k] for k in ['fresh_fits', 'reused_fits_or_aliases', 'unique_fit_keys', 'elapsed_seconds_this_invocation']}
    # Compact metrics contain no raw rows, labels, feature strings or prediction vectors.
    filename = 'results_base.json' if stage == 'base' else 'results.json'
    with (reports/filename).open('w', encoding='utf-8') as handle:
        json.dump(result, handle, separators=(',', ':'), allow_nan=False)
        handle.write('\n')
    for end, states in audits.items():
        write_json(reports/'fit_audits'/f'fit{end}.json', states)
    return result
