"""Read-only input/API readiness audit; never fits models or opens Test.

Only writes a new report when --output is supplied; refuses overwrite.
This is readiness to IMPLEMENT the queue, not permission to skip adapter tests.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.kernel_approximation import PolynomialCountSketch
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBRegressor

from src.utils.paths import PROJECT_ROOT, task_files

QUEUE = PROJECT_ROOT / 'experiments/optimization_v3/research_next_v1/queue.json'


def digest_file(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def historical_bundle(path):
    """Direct verified data read: no historical training/context/budget loader."""
    side = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    assert side['npz_sha256'] == digest_file(path), f'NPZ changed: {path}'
    assert side['integrity_sha256'] == digest_json({k:v for k,v in side.items() if k!='integrity_sha256'})
    assert side['identity_sha256'] == digest_json(side['identity'])
    with np.load(path, allow_pickle=False) as z:
        # Only prediction arrays, never arbitrary pickle/object payload.
        required = ['score_rows','score_record_ids','raw_candidate']
        arrays = {k:z[k].copy() for k in required}
        if 'R22_ORIGINAL' in z:
            arrays['A'] = z['R22_ORIGINAL'].copy()
            arrays['legacy'] = z['champion_prediction'].copy()
        else:
            arrays['A'] = z['prediction'].copy()
            arrays['legacy'] = z['baseline_prediction'].copy()
        for k in ('control__c2b_probability','control__cb_probability','control__baseline_sparse_raw'):
            if k in z:
                arrays[k] = z[k].copy()
    return arrays, side


def source_paths(origin):
    base = PROJECT_ROOT / 'experiments/runs'
    if origin in (55,67):
        return {c:base/f'optimization_v3_batch1/{c}/{c}_fit{origin}.npz'
                for c in ('R22','R06','C16')}
    return {'R22':base/f'optimization_v3_final_continuation/R22/R22_fit{origin}.npz',
            'R06':base/f'optimization_v3_expanded_benchmark_v1_recovery_v2/R06/R06_fit{origin}.npz',
            'C16':base/f'optimization_v3_final_continuation/C16/C16_fit{origin}.npz'}


def check_queue(q):
    assert q['order'] == [f'N{i:02}' for i in range(7)]
    assert [c['id'] for c in q['cards']] == q['order']
    assert sum(c['pilot_fits'] for c in q['cards']) == q['budget']['pilot_worst_case'] == 26
    cards = {card['id']: card for card in q['cards']}
    selection = q['continuation_selection']
    class_costs = sorted((cards[c]['continuation_fits'] for c in
                          selection['classification_candidates']), reverse=True)
    reg_costs = sorted((cards[c]['continuation_fits'] for c in
                        selection['regression_priority']), reverse=True)
    continuation = (sum(class_costs[:selection['max_classification']]) +
                    sum(reg_costs[:selection['max_regression']]))
    assert continuation <= q['budget']['continuation_max']
    allocations = ('pilot_worst_case', 'continuation_max',
                   'control_replay_max', 'technical_retry_max')
    assert sum(q['budget'][k] for k in allocations) == q['budget']['new_phase_cap'] == 60
    assert q['deferred_not_in_execution_queue'] == ['N07','N08']
    assert q['cards'][5]['recipe']['origins'] == [67,75]
    assert q['boundaries']['test'] is q['boundaries']['kaggle'] is False


def quantile_constructor_check(q):
    params = q['cards'][5]['recipe']['parameters'].copy()
    alphas = params.pop('alphas')
    assert alphas == [.1, .9]
    for alpha in alphas:
        head = LGBMRegressor(alpha=alpha, **params)
        actual = head.get_params()
        assert actual['objective'] == 'quantile' and actual['alpha'] == alpha
    return alphas


def rank50(raw, cycle):
    result = np.empty(len(raw))
    for c in np.unique(cycle):
        m = cycle == c
        result[m] = np.clip(.5*raw[m]+.5*100*rankdata(raw[m], method='average')/m.sum(),0,100)
    return result


def run():
    q = json.loads(QUEUE.read_text(encoding='utf-8'))
    check_queue(q)
    frames, raw_hashes = {}, {}
    for task in ('classification','regression'):
        train = task_files(task)['train']
        raw_hashes[task] = digest_file(train)
        assert raw_hashes[task] == q['raw_train_sha256'][task]
        frames[task] = pd.read_csv(train, usecols=['record_id','cycle_code'])
        assert len(frames[task]) == 50608 and frames[task].record_id.is_unique
        np.testing.assert_array_equal(np.unique(frames[task].cycle_code), np.arange(47,80))
    # Constructor/signature validation, not estimator fits or fit_transform.
    n2 = q['cards'][2]['recipe']
    sketch = {k:v for k,v in n2['sketch'].items() if k!='class'}
    PolynomialCountSketch(**sketch)
    LogisticRegression(**n2['head'])
    head = {k:v for k,v in q['cards'][3]['recipe']['router'].items() if k!='class'}
    HistGradientBoostingRegressor(**head)
    quantile_alphas = quantile_constructor_check(q)
    for estimator in (LGBMClassifier,LGBMRegressor,XGBRegressor):
        assert 'sample_weight' in inspect.signature(estimator.fit).parameters
    # Target support is inspected only within each fit prefix; no classifier fit.
    cp = task_files('classification')['train']
    weights_frame = pd.read_csv(cp, usecols=['cycle_code','text_load','artifact_signal'])
    support = {}
    for cutoff in (55,59,63,67,71,75):
        prefix = weights_frame[weights_frame.cycle_code <= cutoff]
        value = pd.to_numeric(prefix.text_load, errors='raise').to_numpy()
        assert np.isfinite(value).all(), 'N01 missing/non-numeric case needs explicit new input-schema review'
        edges = np.unique(np.quantile(value,[.25,.5,.75]))
        bins = np.searchsorted(edges,value,side='right')
        target = prefix.artifact_signal.to_numpy()
        assert np.isin(target,[0,1]).all()
        cells = [int(np.sum((bins==g)&(target==label))) for g in np.unique(bins) for label in (0,1)]
        assert min(cells)>=100, f'N01 support below fixed minimum at {cutoff}'
        support[str(cutoff)]={'fit_rows':len(prefix),'bins':len(np.unique(bins)),'minimum_class_bin_cell':min(cells)}
    token_shapes = {'q_7hex':0,'z_6hex':0}
    for chunk in pd.read_csv(cp,usecols=['signal_stream'],chunksize=4000):
        for text in chunk.signal_stream.dropna():
            for token in re.findall(r'(?u)\b\w\w+\b',text.lower()):
                if re.fullmatch(r'q[0-9a-f]{7}',token): token_shapes['q_7hex']+=1
                elif re.fullmatch(r'z[0-9a-f]{6}',token): token_shapes['z_6hex']+=1
                else: raise AssertionError('N06 token shape outside declared bijection domains')
    checks, hashes = [], {}
    for origin in (55,59,63,67,71,75):
        arrays = {}
        for cid,path in source_paths(origin).items():
            a,side = historical_bundle(path)
            frame = frames['classification' if cid=='C16' else 'regression']
            rows = a['score_rows']
            assert rows.ndim==1 and rows.dtype.kind in 'iu' and len(np.unique(rows))==len(rows)
            assert (rows>=0).all() and (rows<len(frame)).all()
            np.testing.assert_array_equal(a['score_record_ids'].astype(str), frame.record_id.to_numpy()[rows].astype(str))
            cyc = frame.cycle_code.to_numpy()[rows]
            assert (cyc>origin).all() and (cyc<=min(origin+12,79)).all()
            for c in np.unique(cyc):
                assert np.sum(cyc==c) == np.sum(frame.cycle_code==c), f'Incomplete cycle {cid}/{origin}/{c}'
            for k in ('raw_candidate','A','legacy'):
                assert a[k].shape==rows.shape and np.isfinite(a[k]).all()
            if cid=='C16':
                assert all(k in a for k in ('control__c2b_probability','control__cb_probability','control__baseline_sparse_raw'))
            else:
                np.testing.assert_allclose(np.clip(.75*a['legacy']+.25*rank50(a['raw_candidate'],cyc),0,100),a['A'],atol=1e-12,rtol=0)
            arrays[cid] = a
            rel = str(path.relative_to(PROJECT_ROOT))
            hashes[rel] = digest_file(path)
            hashes[str(path.with_suffix('.json').relative_to(PROJECT_ROOT))] = digest_file(path.with_suffix('.json'))
            # No requirement that old shared source hashes equal a changed budget helper.
            # Bundle identity/source lineage remains bound to the historical sidecar.
            checks.append({'card':cid,'origin':origin,'score_rows':len(rows),
                           'identity_sha256':side['identity_sha256'],'status':'PASS'})
        r22,r06 = arrays['R22'],arrays['R06']
        common,i,j = np.intersect1d(r22['score_rows'],r06['score_rows'],return_indices=True)
        frame = frames['regression']
        cycles = frame.cycle_code.to_numpy()[common]
        expected = np.flatnonzero(frame.cycle_code.between(origin+1,origin+4).to_numpy())
        assert np.isin(expected,common).all(), f'Missing primary rows origin{origin}'
        np.testing.assert_array_equal(r22['legacy'][i],r06['legacy'][j])
        for c in np.unique(cycles):
            assert np.sum(cycles==c)==np.sum(frame.cycle_code==c)
    return {'status':'INPUTS_AND_APIS_PASS_READY_TO_IMPLEMENT_NOT_READY_TO_SKIP_ADAPTER_TESTS',
            'queue_sha256':digest_file(QUEUE),'raw_train_hashes':raw_hashes,
            'api_constructors_and_weight_signatures':'PASS','historical_bundles_checked':len(checks),
            'N05_quantile_constructor_alphas':quantile_alphas,
            'N01_all_six_origins_support':support,'N06_token_occurrence_shapes':token_shapes,
            'all_six_R22_R06_pairs_join_complete_cycles':True,'bundle_checks':checks,'input_hashes':hashes,
            'new_model_fits':0,'synthetic_model_fits':0,'test_files_opened':0,'kaggle_calls':0,
            'old_budget_not_used_for_readiness':True,
            'pending_before_official_training':['Implement each lane adapter','End-to-end synthetic fit->publish->read->verify->report for each adapter','New-phase budget authorization from user launch','Freeze actual source, stable recipe and complete input lineage','Live RAM/slot check']}


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=run()
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x',encoding='utf-8') as stream:
            json.dump(result,stream,indent=2,allow_nan=False)
            stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('bundle_checks','input_hashes')},indent=2))
