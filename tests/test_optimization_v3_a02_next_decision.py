import numpy as np
import pytest
from sklearn.metrics import f1_score

from scripts.optimization_v3_a02_next_decision import exact_top_k, matched_count, metrics


def test_top_k_stable_ties_and_extreme_counts():
    s, r = np.array([.5, .7, .5]), np.array([30, 20, 10])
    assert exact_top_k(s, r, 2).tolist() == [0, 1, 1]
    assert exact_top_k(s, r, 0).sum() == 0
    assert exact_top_k(s, r, 3).sum() == 3
    with pytest.raises(ValueError):
        exact_top_k(s, r, 4)


def test_count_match_stratifies_cycles_and_preserves_fallback_without_target():
    f = dict(baseline=np.array([.6, .4, .6, .4, .7, .4]),
             observed=np.array([.49, .48, .7, .55, .7, .4]),
             shuffled=np.array([.5]*6), rows=np.arange(6),
             cycles=np.array([60, 60, 61, 61, 60, 59]),
             origin=np.array([59, 59, 59, 59, 59, -1]),
             missing=np.array([False, False, False, False, True, False]))
    result = matched_count(f, 'baseline', 'observed', 'cycle')
    assert result.tolist() == [0, 0, 1, 1, 1, 0]
    assert 'y' not in f


def test_confusion_f1_matches_reference_for_actual_and_single_class_cases():
    for y, pred in [(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1])),
                    (np.zeros(3, dtype=int), np.zeros(3, dtype=int))]:
        actual = metrics(y, pred)
        assert actual['macro_f1'] == pytest.approx(f1_score(y, pred, labels=[0, 1], average='macro', zero_division=0))
        assert actual['n'] == actual['tp']+actual['tn']+actual['fp']+actual['fn']
