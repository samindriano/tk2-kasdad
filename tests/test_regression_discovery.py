"""Reconcile saved evidence against actual immutable course inputs, no models."""
import hashlib
import json
from pathlib import Path
import unittest
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.utils.paths import PROJECT_ROOT, task_files


class RegressionDiscoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files = task_files('regression', PROJECT_ROOT/'data/raw')
        if not all(cls.files.values()):
            raise unittest.SkipTest('Official local regression CSVs required; no synthetic substitute.')
        cls.train = pd.read_csv(cls.files['train'])
        cls.report = json.loads((PROJECT_ROOT/'reports/audits/regression_audit.json').read_text())
        cls.discovery = json.loads((PROJECT_ROOT/'reports/eda/discovery_summary.json').read_text())

    def test_archive_bytes_and_saved_input_receipt(self):
        receipt = json.loads((PROJECT_ROOT/'reports/audits/regression_acquisition.json').read_text())
        archive = PROJECT_ROOT/'data/raw/regression'/receipt['archive_name']
        if archive.exists():
            self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(), receipt['archive_sha256'])
            with zipfile.ZipFile(archive) as z:
                self.assertIsNone(z.testzip())
                for role, path in self.files.items():
                    self.assertEqual(path.read_bytes(), z.read(path.name))
        for role, path in self.files.items():
            value = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(value, receipt['files'][path.name]['sha256'])
            self.assertEqual(value, self.report['files'][role]['sha256'])

    def test_verified_target_and_complete_audit(self):
        test, sample = pd.read_csv(self.files['test']), pd.read_csv(self.files['sample'])
        self.assertEqual(set(self.train)-set(test), {'threat_rank'})
        self.assertEqual(set(sample), {'record_id', 'threat_rank'})
        self.assertTrue(test.record_id.equals(sample.record_id))
        self.assertEqual(self.report['status'], 'COMPLETE')
        self.assertEqual(self.report['audit_issues'], [])
        self.assertEqual(self.train.threat_rank.nunique(), self.discovery['target_tp2']['unique'])

    def test_rank_geometry_with_independent_scipy_ranks(self):
        matches = 0
        for _, group in self.train.groupby('cycle_code'):
            actual = group.threat_rank.to_numpy()
            expected = np.round(100*rankdata(actual, method='average')/len(actual), 4)
            matches += int(np.equal(actual, expected).sum())
        recorded = self.discovery['within_cycle_percentile_diagnostic']['within_cycle']
        self.assertEqual(matches, recorded['exact_match_rows'])
        self.assertEqual(matches, len(self.train))

    def test_periods_keep_cycles_whole_and_relationship_matches(self):
        periods = self.discovery['impact_threat']['by_cycle_period']
        expected_blocks = np.array_split(sorted(self.train.cycle_code.unique()), 3)
        for saved, cycles in zip(periods, expected_blocks):
            group = self.train[self.train.cycle_code.isin(cycles)]
            self.assertEqual(saved['n'], len(group))
            self.assertEqual([saved['cycle_min'], saved['cycle_max']], [cycles[0], cycles[-1]])
            correlation = np.corrcoef(group.impact_load, group.threat_rank)[0, 1]
            self.assertAlmostEqual(saved['impact_threat_correlation']['pearson'], correlation, places=12)
        self.assertEqual(sum(x['n'] for x in periods), len(self.train))


if __name__ == '__main__':
    unittest.main()
