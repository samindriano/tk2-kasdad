import numpy as np

from scripts.optimization_v3_final_guard_audit import group_audit


def test_individual_window_failure_is_not_hidden_by_aggregate():
    y = np.full(1000, 50.)
    b = y.copy()
    p = b.copy()
    p[:100] += .4
    masks = {'primary': np.ones(1000, bool), 'H12': np.ones(1000, bool),
             'W1': np.arange(1000) < 100, 'W4': np.arange(1000) >= 100}
    report = group_audit(y, p, b, np.ones(1000, bool), masks)
    assert [(r['view'],r['group']) for r in report['failures']] == [('W1','missing')]
    assert not report['all_supported_pass']
    assert group_audit(y, b, b, np.ones(1000, bool), masks)['all_supported_pass']


def test_group_below_support_is_descriptive():
    y = np.full(99, 50.)
    masks = {k: np.ones(99, bool) for k in ('primary','H12','W1','W4')}
    report = group_audit(y,y+10,y,np.ones(99,bool),masks)
    assert report['all_supported_pass']
    assert all(r['pass'] is None for r in report['records'])
