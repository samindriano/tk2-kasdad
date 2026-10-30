"""Frozen Round2 adaptive development comparisons; Round1 stays immutable."""
from __future__ import annotations

from datetime import datetime, timezone
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
from scipy.special import expit
from threadpoolctl import threadpool_limits

from src.data.audit import sha256, write_json
from src.models.round1_evaluation import digest_json, code_digest, array_digest
from src.models.classification_evaluation import verify_freeze as verify_round1, _safe_probability
from src.models.classification_round2_models import build_round2_model, stream_missing_mask, linear_components, coefficient_summary
from src.models.classification_round2_metrics import select_policies, evaluate_predictions
from src.utils.paths import PROJECT_ROOT, task_files

CONFIG = PROJECT_ROOT/'experiments/classification_round2/protocol.json'
FREEZE = CONFIG.with_name('freeze.json')
REPORT = PROJECT_ROOT/'reports/classification_round2'
DEFAULT_RUN = 'classification_round2_v1'
CORE = ['src/models/classification_round2_models.py', 'src/models/classification_round2_metrics.py',
        'src/models/classification_round2_evaluation.py', 'scripts/run_classification_round2.py']
TEXT_EXTENSIONS = {'.py', '.md', '.json', '.txt', '.ipynb', '.toml', '.yaml', '.yml', '.csv', ''}


def load_protocol():
    return json.loads(CONFIG.read_text(encoding='utf-8'))


def _protected_digest(path):
    return code_digest(path) if path.suffix.lower() in TEXT_EXTENSIONS or path.name.startswith('.') else sha256(path)


def identity(data_dir=None):
    r1 = verify_round1(data_dir)
    refs = {}
    for end in [55, 59, 63, 67, 71, 75]:
        audit = json.loads((PROJECT_ROOT/f'reports/classification_round1/fit_audits/fit{end}.json').read_text(encoding='utf-8'))
        refs[str(end)] = {m: {k: audit[m][k] for k in ['key', 'probability_sha256', 'inner_probability_sha256',
                           'fitted_state_sha256', 'inner_state_sha256']} for m in ['C1_A', 'C1_B', 'C2_A', 'C2_B']}
    return {'protocol_sha256': digest_json(load_protocol()),
            'new_code_sha256': {p: code_digest(PROJECT_ROOT/p) for p in CORE},
            'round1_freeze_signature': r1['signature'], 'round1_identity': r1['identity'],
            'round1_vector_references': refs, 'python': platform.python_version(),
            'packages': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'threadpoolctl']}}


def freeze_protocol(data_dir=None):
    current = identity(data_dir)
    if FREEZE.exists():
        old = json.loads(FREEZE.read_text(encoding='utf-8'))
        if old['identity'] != current:
            raise RuntimeError('Existing Round2 freeze differs; preserve receipt and document amendment before any refit.')
        return old
    tracked = subprocess.check_output(['git', '-C', str(PROJECT_ROOT), 'ls-files', '-z']).decode().split('\0')
    protected = {p: _protected_digest(PROJECT_ROOT/p) for p in tracked if p and p != 'README.md'}
    receipt = {'status': 'FROZEN_BEFORE_ROUND2_DIAGNOSTIC_AND_MODEL_FITS',
               'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
               'base_git_head': subprocess.check_output(['git', '-C', str(PROJECT_ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
               'identity': current, 'signature': digest_json(current), 'protected_checkpoint_files': protected,
               'hash_convention': 'Canonical JSON; source/checkpoint text UTF8 LF; binary/raw exact bytes. Base HEAD historical.'}
    write_json(FREEZE, receipt)
    return receipt


def verify_freeze(data_dir=None):
    if not FREEZE.exists():
        raise RuntimeError('Freeze Round2 protocol before diagnosis or fitting.')
    receipt = json.loads(FREEZE.read_text(encoding='utf-8'))
    if receipt['identity'] != identity(data_dir) or receipt['signature'] != digest_json(receipt['identity']):
        raise RuntimeError('Frozen Round2 code/input/runtime/reference mismatch.')
    for rel, expected in receipt['protected_checkpoint_files'].items():
        if _protected_digest(PROJECT_ROOT/rel) != expected:
            raise RuntimeError('Protected checkpoint changed: '+rel)
    return receipt


def _indices(frame, end, protocol):
    last = max(w['validation'][1] for w in protocol['primary_windows']+protocol['secondary_windows'] if w['fit'][1] == end)
    return (np.flatnonzero(frame.cycle_code.between(47, end)), np.flatnonzero(frame.cycle_code.between(end+1, last)),
            np.flatnonzero(frame.cycle_code.between(47, end-4)), np.flatnonzero(frame.cycle_code.between(end-3, end)))


def _new_model(recipe):
    return build_round2_model({k: v for k, v in recipe.items() if k in ['id', 'kind', 'source', 'variant']})


def _artifact_path(run_id, dependency=False):
    if not re.fullmatch(r'classification_round2_[A-Za-z0-9_-]+', run_id):
        raise ValueError('Run ID must use the classification_round2_ namespace.')
    if run_id == 'classification_round2_dependencies' and not dependency:
        raise ValueError('Dependency cache namespace is reserved.')
    parent = (PROJECT_ROOT/'experiments/runs').resolve()
    target = (parent/run_id).resolve()
    if target.parent != parent or target.name.casefold() != run_id.casefold():
        raise ValueError('Resolved artifact path escaped its Round2 namespace.')
    return target


def _report_path(report_dir):
    target = (Path(report_dir) if report_dir else REPORT).resolve()
    canonical = REPORT.resolve()
    temp_root = Path(tempfile.gettempdir()).resolve()
    prohibited = [PROJECT_ROOT/'reports/classification_round1', PROJECT_ROOT/'reports/round1', PROJECT_ROOT.parent/'TK1']
    if any(target.is_relative_to(path.resolve()) for path in prohibited):
        raise ValueError('Round1/TP1 checkpoint output paths are forbidden.')
    if not target.is_relative_to(canonical) and not (target != temp_root and target.is_relative_to(temp_root)):
        raise ValueError('Reports must stay under reports/classification_round2 or a dedicated temporary directory.')
    return target


def _round1_group(frame, end, model, protocol, bootstrap=False):
    """Read original vectors; optional exact reconstruction goes only to a new namespace."""
    audit = json.loads((PROJECT_ROOT/f'reports/classification_round1/fit_audits/fit{end}.json').read_text(encoding='utf-8'))[model]
    original = PROJECT_ROOT/f'experiments/runs/classification_round1_v1/fit{end}__{model}.npz'
    reconstructed = _artifact_path('classification_round2_dependencies', dependency=True)/f'fit{end}__{model}.npz'
    fit, rows, inner_fit, inner_rows = _indices(frame, end, protocol)
    path = original if original.exists() else reconstructed
    dependency_fits = 0
    if not path.exists():
        if not bootstrap:
            raise FileNotFoundError('Round1 vectors missing; use --bootstrap-dependencies to reconstruct only four controls in Round2-owned cache.')
        features = load_round1_features(model)
        generated = []
        for fit_rows, val_rows, expected_state in [(inner_fit, inner_rows, audit['inner_state_sha256']), (fit, rows, audit['fitted_state_sha256'])]:
            estimator = build_round2_model({'id': model, 'kind': 'control', 'source': model})
            with threadpool_limits(limits=protocol['thread_limit']):
                estimator.fit(frame.iloc[fit_rows][features], frame.iloc[fit_rows].artifact_signal.to_numpy())
                p = _safe_probability(estimator, frame.iloc[val_rows][features])
            if digest_json(estimator.audit_state()) != expected_state:
                raise RuntimeError('Reconstructed Round1 state differs; refusing dependency substitution.')
            generated.append(p); dependency_fits += 1
        ip, p = generated
        if array_digest(p) != audit['probability_sha256'] or array_digest(ip) != audit['inner_probability_sha256']:
            raise RuntimeError('Reconstructed Round1 probabilities are not bit-identical.')
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, outer_rows=rows, probability=p, inner_rows=inner_rows, inner_probability=ip)
    with np.load(path, allow_pickle=False) as a:
        saved_rows, p, saved_inner, ip = [a[k].copy() for k in ['outer_rows', 'probability', 'inner_rows', 'inner_probability']]
    r1p = json.loads((PROJECT_ROOT/'experiments/classification_round1/protocol.json').read_text(encoding='utf-8'))
    declared = next(r for r in r1p['recipes'] if r['id'] == model)
    r1_signature = json.loads((PROJECT_ROOT/'experiments/classification_round1/freeze.json').read_text(encoding='utf-8'))['signature']
    key = digest_json({'freeze': r1_signature, 'end': end, 'recipe': declared,
                       'outer_fit': array_digest(fit), 'outer_validation': array_digest(rows),
                       'inner_fit': array_digest(inner_fit), 'inner_validation': array_digest(inner_rows)})
    if key != audit['key'] or not np.array_equal(rows, saved_rows) or not np.array_equal(inner_rows, saved_inner):
        raise RuntimeError('Round1 dependency key/row mismatch.')
    if array_digest(p) != audit['probability_sha256'] or array_digest(ip) != audit['inner_probability_sha256']:
        raise RuntimeError('Round1 dependency vector hash mismatch.')
    return {'rows': rows, 'probability': p, 'inner_rows': inner_rows, 'inner_probability': ip,
            'meta': audit, 'origin': path.relative_to(PROJECT_ROOT).as_posix(), 'dependency_fits': dependency_fits}


def load_round1_features(model):
    p = json.loads((PROJECT_ROOT/'experiments/classification_round1/protocol.json').read_text(encoding='utf-8'))
    recipe = next(r for r in p['recipes'] if r['id'] == model)
    return p['feature_sets'][recipe['features']]


def _thresholds(frame, group, protocol):
    inner = frame.iloc[group['inner_rows']]
    cfg = protocol['threshold_selection']
    end = int(inner.cycle_code.max()) if len(inner) else 46
    fit = frame[frame.cycle_code.between(47, end-4)]
    eligible = (fit.cycle_code.nunique() >= cfg['minimum_inner_fit_cycles'] and
                len(fit) >= cfg['minimum_inner_fit_rows'] and fit.artifact_signal.nunique() == 2)
    return select_policies(inner.artifact_signal.to_numpy(), group['inner_probability'], inner.cycle_code.to_numpy(), cfg, eligible=eligible)


def _routes(frame, groups, protocol):
    for end, group in groups.items():
        for route in [r for r in protocol['recipes'] if r['kind'] == 'route']:
            a, b = group[route['present']], group[route['missing']]
            if not np.array_equal(a['rows'], b['rows']) or not np.array_equal(a['inner_rows'], b['inner_rows']):
                raise RuntimeError('Route branches do not share validation records/history.')
            p = np.where(stream_missing_mask(frame.iloc[a['rows']]), b['probability'], a['probability'])
            ip = np.where(stream_missing_mask(frame.iloc[a['inner_rows']]), b['inner_probability'], a['inner_probability'])
            entry = {'rows': a['rows'], 'probability': p, 'inner_rows': a['inner_rows'], 'inner_probability': ip,
                     'meta': {'route': route, 'gate': protocol['routing_gate'], 'trained_gate': False}}
            entry['thresholds'] = _thresholds(frame, entry, protocol)
            group[route['id']] = entry


def routing_diagnostic(data_dir=None, bootstrap=False):
    frozen = verify_freeze(data_dir); protocol = load_protocol()
    _report_path(None)
    frame = pd.read_csv(task_files('classification', data_dir)['train'])
    groups = {}; dependency_fits = 0
    for end in [55, 59, 63, 67, 71, 75]:
        groups[end] = {}
        for model in ['C1_A', 'C1_B', 'C2_B']:
            a = _round1_group(frame, end, model, protocol, bootstrap)
            dependency_fits += a['dependency_fits']; a['thresholds'] = _thresholds(frame, a, protocol)
            groups[end][model] = a
    _routes(frame, groups, protocol)
    result = evaluate_predictions(frame, groups, protocol, recipes=['C2_B', 'C4_LR', 'C4_HGB'], policies=['fixed_0.5'])
    # Preserve a bounded diagnostic receipt rather than repeating all detailed metrics.
    result['jobs'] = [{k: j[k] for k in ['family', 'window', 'model', 'probability_sha256', 'policies']} for j in result['jobs']]
    for j in result['jobs']:
        p = j['policies']['fixed_0.5']; j['policies']['fixed_0.5'] = {'metrics': p['metrics'],
            'subgroups': {s: p['subgroups'][s] for s in ['stream_missing', 'stream_present']}}
    result.update({'status': 'ROUND2_ROUTING_DIAGNOSTIC_FROM_ROUND1_VECTORS', 'freeze_signature': frozen['signature'],
                   'created_at_utc': datetime.now(timezone.utc).isoformat(), 'new_round2_model_fits': 0,
                   'dependency_reconstruction_fits': dependency_fits, 'source': 'Original Round1 saved vectors or exact verified reconstruction in Round2-owned namespace.'})
    write_json(REPORT/'routing_diagnostic.json', result)
    return result


def _component_statistics(components, missing, probability):
    if not components:
        return {'linear': False}
    logit = sum(components.values())
    error = float(np.max(np.abs(expit(logit)-probability)))
    if error > 1e-12 or np.any(components['text'][missing] != 0):
        raise RuntimeError('Logit decomposition/probability/empty-text check failed.')
    out = {'linear': True, 'max_probability_reconstruction_error': error, 'missing_text_component_exactly_zero': True, 'groups': {}}
    for name, mask in [('all', np.ones(len(missing), dtype=bool)), ('stream_missing', missing), ('stream_present', ~missing)]:
        out['groups'][name] = {'n': int(mask.sum()), 'components': {key: {'mean': float(value[mask].mean()),
            'std': float(value[mask].std()), 'quantiles': np.quantile(value[mask], [.05, .5, .95]).tolist()} if mask.any() else None
            for key, value in components.items()}}
    return out


def run_round2(data_dir=None, run_id=DEFAULT_RUN, report_dir=None, force=False, fit_ends=None, bootstrap=False):
    artifact = _artifact_path(run_id)
    reports = _report_path(report_dir)
    if fit_ends is not None and (not fit_ends or run_id == DEFAULT_RUN or report_dir is None or Path(report_dir).resolve() == REPORT.resolve()):
        raise ValueError('Partial reproduction requires a separate run ID and noncanonical report directory.')
    frozen = verify_freeze(data_dir); protocol = load_protocol()
    ends = sorted({w['fit'][1] for w in protocol['primary_windows']})
    if fit_ends is not None:
        if not set(fit_ends) <= set(ends): raise ValueError('History outside protocol.')
        ends = sorted(set(fit_ends))
    frame = pd.read_csv(task_files('classification', data_dir)['train'])
    if not frame.record_id.is_unique or frame.record_id.isna().any() or not frame.artifact_signal.isin([0, 1]).all():
        raise ValueError('Invalid official Train identity/target.')
    if fit_ends is None:
        if not (REPORT/'routing_diagnostic.json').exists(): routing_diagnostic(data_dir, bootstrap)
        diagnostic = json.loads((REPORT/'routing_diagnostic.json').read_text(encoding='utf-8'))
        if diagnostic['freeze_signature'] != frozen['signature']:
            raise RuntimeError('Existing routing diagnostic belongs to another frozen protocol.')
    artifact.mkdir(parents=True, exist_ok=True)
    groups = {}; fresh = dependency_fits = 0; started = perf_counter(); audits = {}
    for end in ends:
        fit, rows, inner_fit, inner_rows = _indices(frame, end, protocol)
        groups[end] = {}; audits[end] = {}
        for recipe in [r for r in protocol['recipes'] if r['kind'] != 'route']:
            model_id = recipe['id']; tag = f'fit{end}__{model_id}'
            key = digest_json({'freeze': frozen['signature'], 'end': end, 'recipe': recipe,
                               'fit': array_digest(fit), 'rows': array_digest(rows), 'inner_fit': array_digest(inner_fit), 'inner_rows': array_digest(inner_rows)})
            meta_path, cache_path = artifact/f'{tag}.json', artifact/f'{tag}.npz'
            reference = _round1_group(frame, end, model_id, protocol, bootstrap) if recipe['kind'] == 'control' else None
            if reference: dependency_fits += reference['dependency_fits']
            if meta_path.exists() and cache_path.exists() and not force:
                meta = json.loads(meta_path.read_text(encoding='utf-8'))
                with np.load(cache_path, allow_pickle=False) as a:
                    p, ip = a['probability'].copy(), a['inner_probability'].copy()
                    if meta['key'] != key or not np.array_equal(a['outer_rows'], rows) or not np.array_equal(a['inner_rows'], inner_rows):
                        raise RuntimeError('Round2 cache key/row mismatch.')
                    components = {k.removeprefix('logit__'): a[k].copy() for k in a.files if k.startswith('logit__')}
                if array_digest(p) != meta['probability_sha256'] or array_digest(ip) != meta['inner_probability_sha256']:
                    raise RuntimeError('Round2 cached probabilities changed.')
                if {k: array_digest(v) for k, v in components.items()} != meta['component_sha256']:
                    raise RuntimeError('Round2 cached logit components changed.')
            else:
                t0 = perf_counter(); stamp = datetime.now(timezone.utc).isoformat()
                inner_state = None
                if reference:
                    ip = reference['inner_probability']; inner_state_hash = reference['meta']['inner_state_sha256']
                else:
                    inner_model = _new_model(recipe)
                    features = inner_model.input_feature_names
                    with threadpool_limits(limits=protocol['thread_limit']):
                        inner_model.fit(frame.iloc[inner_fit][features].copy(), frame.iloc[inner_fit].artifact_signal.to_numpy())
                        ip = _safe_probability(inner_model, frame.iloc[inner_rows][features].copy())
                    inner_state = inner_model.audit_state(); inner_state_hash = digest_json(inner_state); fresh += 1
                group = {'inner_rows': inner_rows, 'inner_probability': ip}
                cuts = _thresholds(frame, group, protocol)
                model = _new_model(recipe)
                features = model.input_feature_names
                with threadpool_limits(limits=protocol['thread_limit']):
                    model.fit(frame.iloc[fit][features].copy(), frame.iloc[fit].artifact_signal.to_numpy())
                    p = _safe_probability(model, frame.iloc[rows][features].copy()); state = model.audit_state()
                    coefficients = coefficient_summary(model); components = linear_components(model, frame.iloc[rows][features].copy())
                    component_stats = _component_statistics(components, stream_missing_mask(frame.iloc[rows]), p)
                fresh += 1
                if reference and (not np.array_equal(p, reference['probability']) or digest_json(state) != reference['meta']['fitted_state_sha256']):
                    raise RuntimeError('Fresh control does not faithfully reproduce Round1.')
                meta = {'key': key, 'model': model_id, 'fit_started_at_utc': stamp, 'elapsed_seconds': perf_counter()-t0,
                        'fit_cycles': [47, end], 'validation_cycles': [int(frame.iloc[rows].cycle_code.min()), int(frame.iloc[rows].cycle_code.max())],
                        'fit_rows': len(fit), 'validation_rows': len(rows), 'inner_fit_cycles': [47, end-4], 'inner_validation_cycles': [end-3, end],
                        'inner_fit_rows': len(inner_fit), 'inner_validation_rows': len(inner_rows),
                        'fitted_state': state, 'fitted_state_sha256': digest_json(state), 'inner_state': inner_state,
                        'inner_state_sha256': inner_state_hash, 'inner_probabilities_reused_from_round1': bool(reference),
                        'round1_reference': {k: reference['meta'][k] for k in ['key', 'probability_sha256', 'inner_probability_sha256', 'fitted_state_sha256']} if reference else None,
                        'control_matches_round1_bit_identically': True if reference else None,
                        'probability_sha256': array_digest(p), 'inner_probability_sha256': array_digest(ip), 'thresholds': cuts,
                        'coefficients': coefficients, 'component_statistics': component_stats,
                        'component_sha256': {k: array_digest(v) for k, v in components.items()},
                        'outer_validation_labels_passed_to_fit_or_selection': False}
                np.savez_compressed(cache_path, outer_rows=rows, probability=p, inner_rows=inner_rows, inner_probability=ip,
                                    **{'logit__'+k: v for k, v in components.items()})
                write_json(meta_path, meta)
            groups[end][model_id] = {'rows': rows, 'probability': p, 'inner_rows': inner_rows, 'inner_probability': ip,
                                     'thresholds': meta['thresholds'], 'meta': meta}
            audits[end][model_id] = meta
            print(f'fit47-{end} {model_id}: inner cutoff {meta["thresholds"]["inner_pooled"]["threshold"]:.2f}', flush=True)
        structured_hashes = {audits[end][m]['coefficients']['structured_transform_sha256']
                             for m in ['C1_A', 'C2_B', 'C3_B', 'C3_C', 'C5_LOW', 'C5_HIGH']}
        text_hashes = {audits[end][m]['coefficients']['text_transform_sha256']
                       for m in ['C2_A', 'C2_B', 'C3_C', 'C5_LOW', 'C5_HIGH']}
        if len(structured_hashes) != 1 or len(text_hashes) != 1:
            raise RuntimeError('Controlled variants do not share identical learned structured/text transforms.')
    _routes(frame, groups, protocol)
    result = evaluate_predictions(frame, groups, protocol)
    if fit_ends is None:
        expected = {(j['family'], j['window'], j['model']): j['probability_sha256'] for j in diagnostic['jobs']}
        matched = 0
        for job in result['jobs']:
            key = (job['family'], job['window'], job['model'])
            if key in expected:
                if job['probability_sha256'] != expected[key]:
                    raise RuntimeError('Fresh refit route/control differs from saved-vector diagnostic.')
                matched += 1
        if matched != 24: raise RuntimeError('Incomplete saved-versus-fresh routing comparison.')
        result['routing_diagnostic_comparison'] = {'status': 'PASS', 'window_vectors_identical': matched,
                                                   'route_window_vectors': 16, 'hybrid_control_window_vectors': 8}
    for end, audit in audits.items(): write_json(reports/'fit_audits'/f'fit{end}.json', audit)
    result.update({'status': 'CLASSIFICATION_ROUND2_COMPLETE_REVIEW_NO_FINAL_MODEL' if len(result['jobs']) == 80 else 'PARTIAL_HISTORY_REPRODUCTION',
                   'run_id': run_id, 'protocol_id': protocol['protocol_id'], 'freeze_signature': frozen['signature'],
                   'identity': frozen['identity'], 'unique_outer_histories': ends, 'fresh_model_fits_this_run': fresh,
                   'dependency_reconstruction_fits': dependency_fits, 'elapsed_seconds_this_run': perf_counter()-started,
                   'interpretation': protocol['selection_bias']})
    reports.mkdir(parents=True, exist_ok=True)
    # The detailed machine artifact is compact; readable comparisons live in notebook/Markdown.
    (reports/'results.json').write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(',', ':'))+'\n', encoding='utf-8')
    verify_freeze(data_dir)
    return result
