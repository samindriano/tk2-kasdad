import numpy as np
from scripts.optimization_v3_robust_aggregate import aggregate, rank50


def test_median_rejects_single_extreme_and_rank_ties_remain_equal():
    components = np.array([[10., 90.], [12., 88.], [100., 0.]])
    values = aggregate(components)
    np.testing.assert_array_equal(values['median3'], [12., 88.])
    np.testing.assert_array_equal(values['median25'], .75 * values['champion'] + .25 * np.array([12., 88.]))
    np.testing.assert_array_equal(aggregate(components[::-1])['median3'], values['median3'])
    ranked = rank50(np.array([20., 20., 80.]), np.array([80, 80, 80]))
    assert ranked[0] == ranked[1] and ranked[2] > ranked[1]
