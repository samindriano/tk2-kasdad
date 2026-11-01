import copy
import json

import numpy as np
import pytest
from scipy import sparse
from sklearn.kernel_approximation import PolynomialCountSketch

from scripts.optimization_v3_research_next_preflight import QUEUE, check_queue, quantile_constructor_check


def test_budget_and_queue_are_closed_and_exclude_unverified_backends():
    q=json.loads(QUEUE.read_text(encoding='utf-8'))
    check_queue(q)
    assert max(c['continuation_fits'] for c in q['cards'] if c['task']=='classification') == 8
    assert max(c['continuation_fits'] for c in q['cards'] if c['task']=='regression') == 16
    broken=copy.deepcopy(q)
    broken['cards'][3]['pilot_fits']+=1
    with pytest.raises(AssertionError):check_queue(broken)


def test_continuation_overrun_and_allocation_drift_are_rejected():
    q=json.loads(QUEUE.read_text(encoding='utf-8'))
    broken=copy.deepcopy(q)
    broken['cards'][4]['continuation_fits']+=1
    with pytest.raises(AssertionError):check_queue(broken)
    broken=copy.deepcopy(q)
    broken['budget']['technical_retry_max']+=1
    with pytest.raises(AssertionError):check_queue(broken)
    assert quantile_constructor_check(q)==[.1,.9]


def test_exact_sketch_recipe_accepts_sparse_and_is_deterministic():
    q=json.loads(QUEUE.read_text(encoding='utf-8'))
    params={k:v for k,v in q['cards'][2]['recipe']['sketch'].items() if k!='class'}
    x=sparse.csc_matrix([[1.,0.,2.],[0.,0.,0.],[.2,.7,0.]])
    a=PolynomialCountSketch(**params).fit_transform(x)
    b=PolynomialCountSketch(**params).fit_transform(x)
    assert a.shape==(3,1024) and np.isfinite(a).all()
    np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(a[1],np.zeros(1024))
