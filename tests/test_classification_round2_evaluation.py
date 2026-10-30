"""Pre-fit boundary and dependency checks; no competition model is fit here."""
import unittest
import numpy as np
import pandas as pd

from src.models.classification_round2_evaluation import load_protocol, _indices, _round1_group, run_round2, _routes
from src.utils.paths import task_files, PROJECT_ROOT


class Round2EvaluationTests(unittest.TestCase):
    def test_partial_run_guard_precedes_freeze_or_writes(self):
        with self.assertRaises(ValueError): run_round2(fit_ends=[67])
        with self.assertRaises(ValueError): run_round2(run_id='repeat', fit_ends=[67])

    def test_round1_cache_and_report_paths_are_rejected_before_writes(self):
        for run in ['classification_round1_v1', 'classification_round1_repeat', 'round1_v1', 'classification_round2_dependencies']:
            with self.assertRaises(ValueError): run_round2(run_id=run, force=True)
        for folder in ['reports/classification_round1', 'reports/round1', 'data/raw', 'reports/classification_round1/child']:
            with self.assertRaises(ValueError): run_round2(report_dir=PROJECT_ROOT/folder, force=True)

    def test_actual_temporal_rows_and_existing_control_cache_identities(self):
        f = pd.read_csv(task_files('classification')['train'])
        p = load_protocol()
        self.assertEqual(len(p['recipes']), 10)
        self.assertEqual(len(p['fit_recipes']), 8)
        for end in [55, 59, 63, 67, 71, 75]:
            fit, val, inner_fit, inner_val = _indices(f, end, p)
            self.assertLess(f.iloc[fit].cycle_code.max(), f.iloc[val].cycle_code.min())
            self.assertLess(f.iloc[inner_fit].cycle_code.max(), f.iloc[inner_val].cycle_code.min())
            self.assertLessEqual(f.iloc[inner_val].cycle_code.max(), f.iloc[fit].cycle_code.max())
            for m in ['C1_A', 'C1_B', 'C2_A', 'C2_B']:
                group = _round1_group(f, end, m, p)
                np.testing.assert_array_equal(group['rows'], val)
                np.testing.assert_array_equal(group['inner_rows'], inner_val)
                self.assertEqual(group['dependency_fits'], 0)

    def test_route_uses_only_missingness_and_one_matching_inner_vector(self):
        p = load_protocol()
        f = pd.DataFrame({'signal_stream': ['', 'present', None, 'present', '', 'present'],
                          'artifact_signal': [0, 1, 0, 1, 0, 1], 'cycle_code': [52, 53, 54, 55, 56, 57]})
        rows, inner = np.array([4, 5]), np.array([0, 1, 2, 3])
        def inputs():
            return {55: {m: {'rows': rows, 'probability': np.array(v), 'inner_rows': inner,
                              'inner_probability': np.array([.2, .8, .3, .7])}
                         for m, v in [('C2_B', [.8, .9]), ('C1_A', [.1, .2]), ('C1_B', [.3, .4])]}}
        groups = inputs(); _routes(f, groups, p)
        np.testing.assert_array_equal(groups[55]['C4_LR']['probability'], [.1, .9])
        np.testing.assert_array_equal(groups[55]['C4_HGB']['probability'], [.3, .9])
        f.loc[rows, 'artifact_signal'] = [1, 0]
        changed = inputs(); _routes(f, changed, p)
        np.testing.assert_array_equal(changed[55]['C4_LR']['probability'], groups[55]['C4_LR']['probability'])
        self.assertEqual(changed[55]['C4_LR']['thresholds'], groups[55]['C4_LR']['thresholds'])


if __name__ == '__main__': unittest.main()
