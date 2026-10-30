"""Frozen temporal classification experiments; official Test is never predicted."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter
import re

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.data.audit import sha256, write_json
from src.models.round1_evaluation import digest_json, code_digest, array_digest, window_indices
from src.models.classification_metrics import binary_metrics, choose_threshold, subgroup_masks, reliability
from src.models.classification_models import build_classifier
from src.utils.paths import PROJECT_ROOT, task_files

CONFIG = PROJECT_ROOT/'experiments/classification_round1/protocol.json'
FREEZE = CONFIG.with_name('freeze.json')
CORE = ['src/models/classification_evaluation.py', 'src/models/classification_models.py',
        'src/models/classification_metrics.py', 'scripts/run_classification_round1.py',
        'src/models/round1_models.py', 'src/models/round1_evaluation.py', 'src/data/audit.py', 'src/utils/paths.py']


def load_protocol():
    return json.loads(CONFIG.read_text(encoding='utf-8'))


def identity(data_dir=None):
    files = task_files('classification', data_dir)
    if not all(files.values()):
        raise FileNotFoundError('Official classification Train/Test/Sample required for identity checks.')
    return {'protocol_sha256': digest_json(load_protocol()),
            'code_sha256': {p: code_digest(PROJECT_ROOT/p) for p in CORE},
            'input_sha256': {role: sha256(path) for role, path in files.items()},
            'python': platform.python_version(),
            'packages': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'threadpoolctl']}}


def freeze_protocol(data_dir=None):
    current = identity(data_dir)
    if FREEZE.exists():
        old = json.loads(FREEZE.read_text(encoding='utf-8'))
        if old['identity'] != current:
            raise RuntimeError('Existing classification freeze differs; preserve it and document amendment before refit.')
        return old
    protected = {}
    for prefix in ['reports/round1', 'experiments/round1', 'docs/ROUND1*', 'src/models/round1*',
                   'scripts/*round1.py', 'scripts/build_round1_notebook.py', 'scripts/verify_round1_results.py',
                   'tests/test_round1*', 'notebooks/samuel/04_samuel_regression_round1.ipynb']:
        for path in PROJECT_ROOT.glob(prefix):
            paths = path.rglob('*') if path.is_dir() else [path]
            for item in paths:
                if item.is_file() and 'classification' not in item.name:
                    protected[item.relative_to(PROJECT_ROOT).as_posix()] = sha256(item) if item.suffix == '.png' else code_digest(item)
    receipt = {'status': 'FROZEN_BEFORE_FIRST_CLASSIFICATION_ROUND1_FIT',
               'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
               'base_git_head': subprocess.check_output(['git', '-C', str(PROJECT_ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
               'identity': current, 'signature': digest_json(current), 'regression_protected_sha256': protected,
               'hash_convention': 'Canonical JSON; source/protected text UTF-8 LF; raw CSV and PNG exact bytes.'}
    write_json(FREEZE, receipt)
    return receipt


def verify_freeze(data_dir=None):
    if not FREEZE.exists():
        raise RuntimeError('Freeze protocol before actual model experiments.')
    receipt = json.loads(FREEZE.read_text(encoding='utf-8'))
    if receipt['identity'] != identity(data_dir) or receipt['signature'] != digest_json(receipt['identity']):
        raise RuntimeError('Input/protocol/code/runtime mismatch: refusing fit or cache reuse.')
    for path, expected in receipt['regression_protected_sha256'].items():
        item = PROJECT_ROOT/path
        if (sha256(item) if item.suffix == '.png' else code_digest(item)) != expected:
            raise RuntimeError('Protected Regression Round1 changed: '+path)
    return receipt


def _safe_probability(model, X):
    state = model.audit_state()
    p = np.asarray(model.predict_proba(X), dtype=float)
    if p.shape != (len(X),) or not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise RuntimeError('Invalid positive-class probabilities.')
    if digest_json(state) != digest_json(model.audit_state()):
        raise RuntimeError('Prediction changed fitted audit state.')
    return p


def _fit_group(frame, end, recipes, protocol, frozen, artifact, force):
    windows = protocol['primary_windows']+protocol['secondary_windows']
    validation_end = max(w['validation'][1] for w in windows if w['fit'][1] == end)
    outer_fit = np.flatnonzero(frame.cycle_code.between(47, end))
    outer_rows = np.flatnonzero(frame.cycle_code.between(end+1, validation_end))
    inner_end = end-protocol['inner_threshold']['last_n_fit_cycles']
    inner_fit = np.flatnonzero(frame.cycle_code.between(47, inner_end))
    inner_rows = np.flatnonzero(frame.cycle_code.between(inner_end+1, end))
    outputs = {}
    fresh = 0
    for recipe in recipes:
        model_id = recipe['id']
        tag = f'fit{end}__{model_id}'
        meta_path, array_path = artifact/f'{tag}.json', artifact/f'{tag}.npz'
        key = digest_json({'freeze': frozen['signature'], 'end': end, 'recipe': recipe,
                           'outer_fit': array_digest(outer_fit), 'outer_validation': array_digest(outer_rows),
                           'inner_fit': array_digest(inner_fit), 'inner_validation': array_digest(inner_rows)})
        if meta_path.exists() and array_path.exists() and not force:
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
            with np.load(array_path, allow_pickle=False) as saved:
                rows, probability = saved['outer_rows'], saved['probability']
                inside_rows, inside_p = saved['inner_rows'], saved['inner_probability']
            if meta['key'] != key or not np.array_equal(rows, outer_rows) or not np.array_equal(inside_rows, inner_rows):
                raise RuntimeError('Cache identity mismatch: '+tag)
            if array_digest(probability) != meta['probability_sha256'] or array_digest(inside_p) != meta['inner_probability_sha256']:
                raise RuntimeError('Cache probability digest mismatch: '+tag)
        else:
            features = protocol['feature_sets'][recipe['features']]
            started_at = datetime.now(timezone.utc).isoformat()
            started = perf_counter()
            inner_state = None
            inside_p = np.array([], dtype=float)
            selection = {'threshold': 0.5, 'fallback': False, 'reason': 'Majority control; no threshold selection', 'selection_n': 0}
            if recipe['threshold_selection']:
                enough = (inner_end-47+1 >= protocol['inner_threshold']['minimum_inner_fit_cycles'] and
                          len(inner_fit) >= protocol['inner_threshold']['minimum_inner_fit_rows'] and
                          frame.iloc[inner_fit].artifact_signal.nunique() == 2)
                if enough:
                    model = build_classifier({'id': recipe['id'], 'kind': recipe['kind']})
                    with threadpool_limits(limits=protocol['thread_limit']):
                        model.fit(frame.iloc[inner_fit][features].copy(), frame.iloc[inner_fit].artifact_signal.to_numpy())
                        inside_p = _safe_probability(model, frame.iloc[inner_rows][features].copy())
                    inner_state = model.audit_state()
                    fresh += 1
                    selection = choose_threshold(frame.iloc[inner_rows].artifact_signal.to_numpy(), inside_p, protocol['inner_threshold'])
                else:
                    selection = {'threshold': 0.5, 'fallback': True, 'reason': 'Insufficient inner-fit history/classes', 'selection_n': 0}
            model = build_classifier({'id': recipe['id'], 'kind': recipe['kind']})
            with threadpool_limits(limits=protocol['thread_limit']):
                model.fit(frame.iloc[outer_fit][features].copy(), frame.iloc[outer_fit].artifact_signal.to_numpy())
                probability = _safe_probability(model, frame.iloc[outer_rows][features].copy())
            fresh += 1
            state = model.audit_state()
            meta = {'key': key, 'fit_started_at_utc': started_at, 'elapsed_seconds': perf_counter()-started,
                    'model': model_id, 'fit_cycles': [47, end], 'validation_prediction_cycles': [end+1, validation_end],
                    'fit_rows': len(outer_fit), 'validation_rows': len(outer_rows),
                    'inner_fit_cycles': [47, inner_end], 'inner_validation_cycles': [inner_end+1, end],
                    'inner_fit_rows': len(inner_fit), 'inner_validation_rows': len(inner_rows),
                    'inner_state': inner_state, 'fitted_state': state,
                    'fitted_state_sha256': digest_json(state), 'inner_state_sha256': digest_json(inner_state),
                    'selection': selection, 'prediction_diagnostics': model.prediction_diagnostics(),
                    'probability_sha256': array_digest(probability), 'inner_probability_sha256': array_digest(inside_p),
                    'outer_validation_labels_passed_to_fit_or_selection': False,
                    'threshold_selected_before_outer_refit': True}
            np.savez_compressed(array_path, outer_rows=outer_rows, probability=probability,
                                inner_rows=inner_rows, inner_probability=inside_p)
            write_json(meta_path, meta)
        outputs[model_id] = (outer_rows, probability, meta)
        print(f'fit47-{end} {model_id}: threshold {meta["selection"]["threshold"]:.2f}', flush=True)
    return outputs, fresh


def run_classification(data_dir=None, run_id='classification_round1_v1', report_dir=None, force=False, fit_ends=None):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
        raise ValueError('Simple run ID required.')
    canonical = PROJECT_ROOT/'reports/classification_round1'
    if fit_ends is not None and (not fit_ends or run_id == 'classification_round1_v1' or
                                report_dir is None or Path(report_dir).resolve() == canonical.resolve()):
        raise ValueError('Partial reproduction requires a separate run ID and noncanonical report directory.')
    frozen = verify_freeze(data_dir)
    protocol = load_protocol()
    frame = pd.read_csv(task_files('classification', data_dir)['train'])
    if not frame.record_id.is_unique or frame.record_id.isna().any() or not frame.artifact_signal.isin([0, 1]).all():
        raise ValueError('Invalid source ID/target.')
    started = perf_counter()
    artifact = PROJECT_ROOT/'experiments/runs'/run_id
    artifact.mkdir(parents=True, exist_ok=True)
    reports = Path(report_dir) if report_dir else PROJECT_ROOT/'reports/classification_round1'
    all_windows = [('primary', w) for w in protocol['primary_windows']]+[('secondary', w) for w in protocol['secondary_windows']]
    ends = sorted({w['fit'][1] for _, w in all_windows})
    if fit_ends is not None:
        if not set(fit_ends) <= set(ends):
            raise ValueError('Requested fit history outside protocol.')
        ends = sorted(set(fit_ends))
    fresh = 0
    groups = {}
    for end in ends:
        groups[end], count = _fit_group(frame, end, protocol['recipes'], protocol, frozen, artifact, force)
        fresh += count
    jobs = []
    pooled = {}
    for family, window in all_windows:
        if window['fit'][1] not in groups:
            continue
        fit, rows = window_indices(frame, window)
        subset = frame.iloc[rows]
        y = subset.artifact_signal.to_numpy()
        masks = subgroup_masks(subset, frame.iloc[fit])
        for recipe in protocol['recipes']:
            all_rows, all_p, meta = groups[window['fit'][1]][recipe['id']]
            positions = np.searchsorted(all_rows, rows)
            if not np.array_equal(all_rows[positions], rows):
                raise RuntimeError('Validation row alignment failed.')
            p = all_p[positions]
            policies = {}
            for policy, threshold in [('fixed_0.5', 0.5), ('inner_selected', meta['selection']['threshold'])]:
                cycles = [{'cycle': int(cycle), **binary_metrics(y[subset.cycle_code.eq(cycle)], p[subset.cycle_code.eq(cycle)], threshold)}
                          for cycle in sorted(subset.cycle_code.unique())]
                metrics = binary_metrics(y, p, threshold)
                policies[policy] = {'metrics': metrics, 'equal_cycle_macro_f1': float(np.mean([c['macro_f1'] for c in cycles])),
                                    'per_cycle': cycles, 'subgroups': {name: binary_metrics(y[mask], p[mask], threshold) for name, mask in masks.items()}}
                key = (family, policy, recipe['id'])
                pooled.setdefault(key, []).append((rows, p, np.full(len(rows), threshold), masks))
            jobs.append({'family': family, 'window': window['id'], 'model': recipe['id'],
                         'fit_cycles': window['fit'], 'validation_cycles': window['validation'],
                         'fit_rows': len(fit), 'validation_rows': len(rows), 'fit_group': window['fit'][1],
                         'probability_sha256': array_digest(p), 'threshold_selection': meta['selection'],
                         'impact_subgroup_cutpoints_from_fit': frame.iloc[fit].impact_load.quantile([1/3, 2/3]).tolist(),
                         'policies': policies})
    aggregate = {'primary': {'fixed_0.5': {}, 'inner_selected': {}}, 'secondary': {'fixed_0.5': {}, 'inner_selected': {}}}
    for (family, policy, model), chunks in pooled.items():
        rows = np.concatenate([v[0] for v in chunks]); p = np.concatenate([v[1] for v in chunks]); t = np.concatenate([v[2] for v in chunks])
        if len(rows) != len(np.unique(rows)):
            raise RuntimeError('Validation duplicate within family.')
        y = frame.iloc[rows].artifact_signal.to_numpy()
        related = [j for j in jobs if j['family'] == family and j['model'] == model]
        metrics = binary_metrics(y, p, t)
        subgroups = {}
        for name in chunks[0][3]:
            mask = np.concatenate([v[3][name] for v in chunks])
            subgroups[name] = binary_metrics(y[mask], p[mask], t[mask])
        aggregate[family][policy][model] = {'metrics': metrics,
            'equal_cycle_macro_f1': float(np.mean([c['macro_f1'] for j in related for c in j['policies'][policy]['per_cycle']])),
            'subgroups': subgroups, 'reliability': reliability(y, p)}
    fit_audits = {str(end): {model: data[2] for model, data in group.items()} for end, group in groups.items()}
    for end, audit in fit_audits.items():
        write_json(reports/'fit_audits'/f'fit{end}.json', audit)
    complete = len(jobs) == 56
    result = {'status': 'CLASSIFICATION_ROUND1_COMPLETE_REVIEW_NO_FINAL_MODEL' if complete else 'PARTIAL_HISTORY_REPRODUCTION',
              'protocol_id': protocol['protocol_id'], 'run_id': run_id, 'freeze_signature': frozen['signature'],
              'identity': frozen['identity'], 'jobs': jobs, 'total_evaluation_jobs': len(jobs),
              'unique_outer_histories': ends, 'fresh_model_fits_this_run': fresh,
              'elapsed_seconds_this_run': perf_counter()-started, 'aggregate': aggregate,
              'scope': 'Train-only development predictions; H12 overlaps primary; selected thresholds use inner history only.'}
    write_json(reports/'results.json', result)
    return result
