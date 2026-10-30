"""History and ownership boundaries; no experiment output writers."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.regression_round2_evaluation import (
    select_history, load_protocol, safe_artifact, safe_report, positions_digest,
)
from src.models.round1_evaluation import load_protocol as old_protocol
from src.utils.paths import PROJECT_ROOT


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({'cycle_code': np.repeat(np.arange(47, 80), 7),
                                   'threat_rank': np.arange(231) % 100})

    def test_history_selectors_strictly_past_and_reproducible(self):
        p = load_protocol()
        for end in [55, 59, 63, 67, 71, 75]:
            full = select_history(self.frame, end, p['history_policies']['expanding'])
            recent = select_history(self.frame, end, p['history_policies']['recent8'])
            self.assertEqual(len(recent), 56)
            self.assertEqual(set(self.frame.iloc[recent].cycle_code), set(range(end-7, end+1)))
            for policy in p['history_policies'].values():
                rows = select_history(self.frame, end, policy)
                self.assertTrue(np.all(np.diff(rows) > 0))
                self.assertTrue(set(rows).issubset(full))
                self.assertLess(self.frame.iloc[rows].cycle_code.max(), end+1)
                if policy['kind'] == 'matched':
                    self.assertEqual(len(rows), len(recent))
                    np.testing.assert_array_equal(rows, select_history(self.frame, end, policy))
                    changed = self.frame.copy()
                    changed['threat_rank'] = 99999
                    np.testing.assert_array_equal(rows, select_history(changed, end, policy))
        a = select_history(self.frame, 75, p['history_policies']['matched8_s42'])
        b = select_history(self.frame, 75, p['history_policies']['matched8_s31415'])
        self.assertNotEqual(positions_digest(a), positions_digest(b))

    def test_recent12_early_is_expanding(self):
        p = load_protocol()
        np.testing.assert_array_equal(select_history(self.frame, 55, p['history_policies']['expanding']),
                                      select_history(self.frame, 55, p['history_policies']['recent12']))
        with self.assertRaises(ValueError):
            select_history(self.frame, 99, {'kind': 'expanding'})

    def test_protocol_same_folds_hyperparameters_and_drop_groups(self):
        p, old = load_protocol(), old_protocol()
        for key in ['primary_windows', 'secondary_windows', 'excluded_features']:
            self.assertEqual(p[key], old[key])
        for key, value in old['hgb_fixed_parameters'].items():
            self.assertEqual(p['hgb_fixed_parameters'][key], value)
        full = p['feature_sets']['full']
        for group, columns in p['feature_groups'].items():
            self.assertEqual(p['feature_sets']['no_'+group], [x for x in full if x not in columns])
        self.assertEqual(len(p['recipes']), 15)

    def test_output_ownership_rejects_checkpoint_paths_and_traversal(self):
        for label in ['round1_v1', 'classification_round2_v1', '../regression_round2_bad', 'regression_round2_../x']:
            with self.assertRaises(ValueError):
                safe_artifact(label)
        self.assertTrue(safe_artifact('regression_round2_fixture').is_relative_to(PROJECT_ROOT/'experiments/runs'))
        for path in [PROJECT_ROOT/'reports/round1', PROJECT_ROOT/'reports/classification_round2',
                     PROJECT_ROOT/'reports/regression_round2/../../docs', Path(tempfile.gettempdir())]:
            with self.assertRaises(ValueError):
                safe_report(path)
        self.assertTrue(safe_report().is_relative_to(PROJECT_ROOT/'reports'))


if __name__ == '__main__':
    unittest.main()
