"""Shared immutable stage receipts and prediction-only postprocessing research."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.stats import rankdata, spearmanr

from src.data.audit import sha256
from src.models.round1_evaluation import code_digest, digest_json, verify_freeze as r1_freeze
from src.models.regression_round2_evaluation import verify_freeze as r2_freeze, protected_digest
from src.models.regression_round2_metrics import summarize
from src.utils.paths import PROJECT_ROOT

CONFIG = PROJECT_ROOT/'experiments/regression_research'
REPORT = PROJECT_ROOT/'reports/regression_research'
CACHE = PROJECT_ROOT/'experiments/runs/regression_research_v1'


def save_json(path, value, exclusive=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x' if exclusive else 'w', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, ensure_ascii=True, allow_nan=False, separators=(',', ':'))
        handle.write('\n')


def stage_identity(protocol, code):
    old = r2_freeze()
    return {'protocol': protocol, 'code_sha256': {p: code_digest(PROJECT_ROOT/p) for p in code},
            'round2_signature': old['signature'], 'regression_input_sha256': r1_freeze()['identity']['input_sha256'],
            'packages': {p: importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','threadpoolctl']}}


def freeze_stage(name, protocol, code):
    path = CONFIG/(name+'_freeze.json')
    current = stage_identity(protocol, code)
    if path.exists():
        old = json.loads(path.read_text(encoding='utf-8'))
        if old['identity'] != current: raise RuntimeError('Frozen research stage identity differs: '+name)
        return old
    tracked = subprocess.check_output(['git','-C',str(PROJECT_ROOT),'ls-files','-z']).decode().split('\0')
    protected = {p: protected_digest(PROJECT_ROOT/p) for p in tracked if p and p not in ['README.md','.gitattributes']}
    out = {'stage': name, 'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
           'source_head': subprocess.check_output(['git','-C',str(PROJECT_ROOT),'rev-parse','HEAD'],text=True).strip(),
           'identity': current, 'signature': digest_json(current), 'protected_files': protected}
    save_json(path, out, exclusive=True)
    return out


def verify_stage(name):
    old = json.loads((CONFIG/(name+'_freeze.json')).read_text(encoding='utf-8'))
    if stage_identity(old['identity']['protocol'],list(old['identity']['code_sha256'])) != old['identity']:
        raise RuntimeError('Research code/input/runtime identity mismatch: '+name)
    for rel, expected in old['protected_files'].items():
        if protected_digest(PROJECT_ROOT/rel) != expected: raise RuntimeError('Protected research checkpoint changed: '+rel)
    return old


def percentile_scores(pred, cycles):
    pred, cycles = np.asarray(pred, float), np.asarray(cycles)
    if pred.ndim != 1 or cycles.shape != pred.shape or not np.isfinite(pred).all():
        raise ValueError('Rank mapping requires finite aligned vectors.')
    if len(pred) and np.any(__import__('pandas').isna(cycles)): raise ValueError('Missing cycle.')
    result = np.empty_like(pred)
    for cycle in np.unique(cycles):
        mask = cycles == cycle
        result[mask] = 100.*rankdata(pred[mask], method='average')/mask.sum()
    return result


def mapped_scores(pred, cycles, weight):
    if not 0 <= weight <= 1: raise ValueError('Convex rank weight must lie in [0,1].')
    pred = np.asarray(pred,float)
    if weight == 0: return np.clip(pred,0,100)
    return np.clip((1-weight)*pred + weight*percentile_scores(pred,cycles),0,100)


def score_summary(frame, pred, impact=False):
    stats = summarize(frame,pred,include_impact=impact)
    y = frame.threat_rank.to_numpy()
    stats['pooled_spearman'] = float(spearmanr(y,pred).statistic) if np.std(pred)>0 else None
    per = [float(spearmanr(g.threat_rank,np.asarray(pred)[frame.cycle_code.eq(c).to_numpy()]).statistic)
           for c,g in frame.groupby('cycle_code')]
    finite = [v for v in per if np.isfinite(v)]
    stats['equal_cycle_spearman'] = float(np.mean(finite)) if finite else None
    return stats
