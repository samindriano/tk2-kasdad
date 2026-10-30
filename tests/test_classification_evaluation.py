"""Metric identities and the prescribed temporal boundary, before real fits."""
import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support

from src.models.classification_metrics import binary_metrics, choose_threshold, subgroup_masks
from src.models.classification_evaluation import load_protocol, run_classification
from src.models.round1_evaluation import window_indices
from src.utils.paths import task_files


class ClassificationEvaluationTests(unittest.TestCase):
    def test_partial_reproduction_cannot_overwrite_canonical_results(self):
        with self.assertRaises(ValueError): run_classification(fit_ends=[67])
        with self.assertRaises(ValueError): run_classification(run_id='repeat', fit_ends=[67])

    def test_confusion_class_order_and_pooled_not_mean_f1(self):
        y = np.array([0, 0, 0, 1, 1, 1, 1])
        p = np.array([.1, .7, .2, .8, .9, .3, .4])
        a = binary_metrics(y, p)
        self.assertEqual(a['confusion_matrix'], [[2, 1], [2, 2]])
        self.assertAlmostEqual(a['macro_f1'], f1_score(y, p >= .5, labels=[0, 1], average='macro'))
        self.assertAlmostEqual(a['macro_f1'], (4/7+4/7)/2)
        self.assertNotAlmostEqual(a['macro_f1'], np.mean([binary_metrics(y[:3], p[:3])['macro_f1'], binary_metrics(y[3:], p[3:])['macro_f1']]))

    def test_single_class_empty_and_variable_thresholds_explicit(self):
        a = binary_metrics([0, 0], [.1, .2])
        self.assertTrue(a['single_class_actual']); self.assertEqual(a['macro_f1'], .5)
        self.assertIsNone(a['average_precision']); self.assertIsNone(a['balanced_accuracy'])
        self.assertIsNone(binary_metrics([], [])['macro_f1'])
        self.assertEqual(binary_metrics([0, 1], [.4, .4], [.5, .3])['macro_f1'], 1)
        with self.assertRaises(ValueError): binary_metrics([0, 1], [.5, float('nan')])

    def test_inner_grid_rule_and_fallback(self):
        cfg = {'grid': [.3, .4, .5, .6], 'minimum_inner_validation_rows': 2}
        a = choose_threshold([0, 0, 1, 1], [.1, .2, .41, .42], cfg)
        self.assertEqual(a['threshold'], .4); self.assertEqual(a['inner_macro_f1'], 1)
        self.assertTrue(choose_threshold([0, 0], [.2, .3], cfg)['fallback'])
        self.assertEqual(choose_threshold([0, 1], [.1, .9], cfg)['threshold'], .5)

    def test_actual_temporal_partitions_and_fit_only_subgroup_bounds(self):
        f = pd.read_csv(task_files('classification')['train'])
        protocol = load_protocol()
        self.assertEqual(len(protocol['recipes']), 7)
        seen = []
        for w in protocol['primary_windows']+protocol['secondary_windows']:
            fit, val = window_indices(f, w)
            end = w['fit'][1]; inner_end = end-4
            inner_fit = f.cycle_code.between(47, inner_end)
            inner_val = f.cycle_code.between(inner_end+1, end)
            self.assertFalse((inner_fit & inner_val).any())
            self.assertLess(f.loc[inner_fit, 'cycle_code'].max(), f.loc[inner_val, 'cycle_code'].min())
            self.assertLessEqual(f.loc[inner_val, 'cycle_code'].max(), f.iloc[fit].cycle_code.max())
            self.assertEqual(f.loc[inner_fit, 'artifact_signal'].nunique(), 2)
            masks = subgroup_masks(f.iloc[val], f.iloc[fit])
            self.assertEqual(sum(masks[k].sum() for k in ['impact_low', 'impact_middle', 'impact_high']), len(val))
            if w['id'].startswith('W'): seen.extend(val)
        self.assertEqual(len(seen), 38588); self.assertEqual(len(set(seen)), 38588)


if __name__ == '__main__': unittest.main()
