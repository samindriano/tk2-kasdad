"""Meaningful checks for independent saved-prediction diagnostics; no model fits."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score, mean_absolute_error

from scripts.optimization_v3_postscreen_review import (
    sufficient, metric, contrast, circular_block_interval, validate_bundle,
)


@pytest.mark.parametrize('task', ['classification', 'regression'])
def test_additive_metrics_match_reference_and_pooling(task):
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([.1, .8, .5, .5, .9, .2])
    stats = sufficient(y[:2], p[:2], task) + sufficient(y[2:], p[2:], task)
    expected = f1_score(y, p >= .5, average='macro', labels=[0, 1]) if task == 'classification' else mean_absolute_error(y, p)
    assert metric(stats, task) == pytest.approx(expected)


def test_empty_groups_are_unknown_not_zero_gain():
    assert contrast(np.array([]), np.array([]), np.array([]), 'regression')['gain'] is None
    assert metric(np.zeros(4), 'classification') is None


def test_block_resampling_preserves_zero_difference_and_is_deterministic():
    cycles = np.repeat(np.arange(12), 4)
    y = np.tile([0, 0, 1, 1], 12)
    p = np.tile([.1, .9, .8, .3], 12)
    a = circular_block_interval(y, p, p, cycles, 'classification', draws=30)
    assert a['lower'] == a['upper'] == 0
    assert a == circular_block_interval(y, p, p, cycles, 'classification', draws=30)


def test_bundle_validation_rejects_swapped_record_ids_and_incomplete_cycles():
    frame = pd.DataFrame({'record_id': ['a', 'b', 'c'], 'cycle_code': [55, 56, 57]})
    a = {'score_rows': np.array([1, 2]), 'score_record_ids': np.array(['b', 'c']),
         'prediction': np.array([.2, .6]), 'baseline_prediction': np.array([.1, .5])}
    np.testing.assert_array_equal(validate_bundle(a, frame, 55), [1, 2])
    bad = {**a, 'score_record_ids': np.array(['c', 'b'])}
    with pytest.raises(AssertionError):
        validate_bundle(bad, frame, 55)
    with pytest.raises(AssertionError):
        validate_bundle({k: v[:1] for k, v in a.items()}, frame, 55)
