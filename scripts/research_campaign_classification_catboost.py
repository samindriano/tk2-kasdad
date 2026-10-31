"""One preregistered CatBoost missing expert; Train-only temporal development."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, importlib.metadata, json, subprocess, sys, time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src.models.campaign_classification_catboost import build_catboost_classifier
from src.models.classification_models import _fit_fingerprint
from src.models.classification_round2_models import stream_missing_mask
from src.models.round1_evaluation import array_digest, code_digest, digest_json
from scripts.research_campaign_classification_stage2 import (
    load_train, _load_round1_control, _control_identity, PRIMARY_WINDOWS,
    score_rows, complementarity_counts,
)

CONFIG = ROOT / 'experiments/dual_campaign/classification'
PROTOCOL = CONFIG / 'catboost_protocol.json'
FREEZE = CONFIG / 'catboost_freeze.json'
OUT = ROOT / 'reports/dual_campaign/classification_catboost'
CACHE = ROOT / 'experiments/runs/dual_cls_catboost_v1'
ENDS = [55, 59, 63, 67, 71, 75]

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write('\n')

def identity():
    _, _, train = load_train(None)
    names = subprocess.check_output(['git', '-C', str(ROOT), 'ls-files', 'src'], text=True).splitlines()
    names += ['src/models/campaign_classification_catboost.py',
              'src/models/campaign_classification.py',
              'scripts/research_campaign_classification_catboost.py',
              'scripts/verify_campaign_classification_catboost.py',
              'tests/test_campaign_classification_catboost.py',
              'scripts/research_campaign_classification.py',
              'scripts/research_campaign_classification_stage2.py']
    return {'protocol_sha256': sha(PROTOCOL), 'train_sha256': sha(train),
            'source_sha256': {p: code_digest(ROOT/p) for p in sorted(set(names))},
            'controls': {str(end): _control_identity(end) for end in ENDS},
            'packages': {p: importlib.metadata.version(p) for p in
                         ['numpy', 'pandas', 'scikit-learn', 'catboost', 'threadpoolctl']}}

def freeze():
    value = identity()
    if FREEZE.exists():
        record = read(FREEZE)
        assert record['identity'] == value and record['signature'] == digest_json(value)
        return record
    assert not CACHE.exists(), 'Freeze before any candidate cache'
    record = {'status': 'FROZEN_BEFORE_FITS', 'at_utc': datetime.now(timezone.utc).isoformat(),
              'source_commit': subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'], text=True).strip(),
              'identity': value, 'signature': digest_json(value)}
    save(FREEZE, record)
    return record

def verify_freeze():
    record = read(FREEZE)
    value = identity()
    assert record['identity'] == value and record['signature'] == digest_json(value)
    return record

def controls(frame, end):
    horizon = {55: 67, 67: 79}.get(end, end+4)
    rows, c1, audit1 = _load_round1_control(frame, end, horizon, 'C1_B')
    rows2, c2, audit2 = _load_round1_control(frame, end, horizon, 'C2_B')
    np.testing.assert_array_equal(rows, rows2)
    missing = stream_missing_mask(frame.iloc[rows])
    return rows, c1, c2, missing, {'C1_B': audit1, 'C2_B': audit2}

def load_cache(frame, y, end, signature):
    path = CACHE / f'fit{end}.npz'
    meta = read(path.with_suffix('.json'))
    rows, c1, c2, missing, audit = controls(frame, end)
    assert meta['freeze_signature'] == signature and meta['npz_sha256'] == sha(path)
    assert meta['controls'] == audit
    with np.load(path, allow_pickle=False) as z:
        np.testing.assert_array_equal(z['rows'], rows)
        p = z['candidate'].copy()
    assert p.shape == rows.shape and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    assert meta['prediction_sha256'] == array_digest(p)
    fit = frame.cycle_code.between(47, end).to_numpy()
    fields = meta['state']['input_feature_names']
    assert not {'record_id', 'artifact_signal', 'threat_rank', 'signal_stream'}.intersection(fields)
    assert meta['state']['fit_rows'] == int(fit.sum())
    assert meta['state']['fit_data_sha256'] == _fit_fingerprint(frame.loc[fit, fields], tuple(fields), y[fit])
    return {'rows': rows, 'candidate': p, 'c1': c1, 'c2': c2, 'missing': missing}

def collect_window(record, start, end, cycles):
    selected = (cycles[record['rows']]>=start)&(cycles[record['rows']]<=end)
    return {key: value[selected] for key, value in record.items()}

def concatenate(records):
    result = {key: np.concatenate([r[key] for r in records]) for key in records[0]}
    assert len(np.unique(result['rows'])) == len(result['rows'])
    return result

def evaluate(frame, y, record, include_blend=False):
    r, missing = record['rows'], record['missing']
    route = {'C4_HGB': np.where(missing, record['c1'], record['c2']),
             'CB_MISSING': np.where(missing, record['candidate'], record['c2'])}
    if include_blend:
        route['CB_HGB_50'] = np.where(missing, .5*record['candidate']+.5*record['c1'], record['c2'])
    output = {}
    for name, probability in route.items():
        output[name] = {'all': score_rows(frame, y, r, probability),
                        'missing': score_rows(frame, y, r[missing], probability[missing]),
                        'present': score_rows(frame, y, r[~missing], probability[~missing])}
    return output

def assemble():
    frozen = verify_freeze()
    frame, y, _ = load_train(None)
    cycle = frame.cycle_code.to_numpy(dtype=int)
    fits = {end: load_cache(frame, y, end, frozen['signature']) for end in ENDS}
    windows = {name: collect_window(fits[fit], start, end, cycle)
               for name, (fit, start, end) in PRIMARY_WINDOWS.items()}
    primary = concatenate(list(windows.values()))
    missing = primary['missing']
    paired = complementarity_counts(y[primary['rows']][missing], primary['candidate'][missing], primary['c1'][missing])
    gate = (paired['candidate_only_correct']>=25 and paired['C4_only_correct']>=25
            and paired['candidate_only_correct']>=paired['C4_only_correct'])
    result = {'status': 'TEMPORAL_RESULTS_REVIEW_REQUIRED', 'freeze_signature': frozen['signature'],
              'new_fits': 6, 'test_read': False, 'paired_missing': paired, 'blend_gate_pass': bool(gate),
              'primary': evaluate(frame, y, primary, gate),
              'windows': {name: evaluate(frame, y, r, gate) for name, r in windows.items()},
              'H12_E': evaluate(frame, y, fits[55], gate), 'H12_L': evaluate(frame, y, fits[67], gate),
              'H12': evaluate(frame, y, concatenate([fits[55], fits[67]]), gate),
              'interpretation': 'Adaptive temporal development; H12 reuses outcomes, not untouched holdout.'}
    return result

def run():
    frozen = verify_freeze()
    frame, y, _ = load_train(None)
    CACHE.mkdir(parents=True, exist_ok=True)
    for end in ENDS:
        path = CACHE / f'fit{end}.npz'
        if path.exists() or path.with_suffix('.json').exists():
            load_cache(frame, y, end, frozen['signature'])
            continue
        rows, _, _, _, audits = controls(frame, end)
        fit = frame.cycle_code.between(47, end).to_numpy()
        model = build_catboost_classifier()
        fields = model.input_feature_names
        start = time.perf_counter()
        with threadpool_limits(limits=2):
            model.fit(frame.loc[fit, fields], y[fit])
            state = model.audit_state()
            probability = model.predict_proba(frame.iloc[rows][fields])
        assert state == model.audit_state()
        np.savez_compressed(path, rows=rows, candidate=probability)
        save(path.with_suffix('.json'), {'freeze_signature': frozen['signature'], 'state': state,
             'prediction_sha256': array_digest(probability), 'npz_sha256': sha(path),
             'controls': audits, 'seconds': time.perf_counter()-start})
        print('FITTED', end, round(time.perf_counter()-start, 1), flush=True)
    result = assemble()
    save(OUT/'results.json', result)
    print('RESULTS', {name: v['all']['macro_f1'] for name, v in result['primary'].items()}, flush=True)

if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--freeze', action='store_true'); args=parser.parse_args()
    if args.freeze: print(freeze()['signature'])
    else: run()
