"""Integration checks against actual saved classification development vectors."""
import json
from pathlib import Path
import unittest
import nbformat

from scripts.verify_classification_round1 import verify_results
from src.models.classification_evaluation import verify_freeze
from src.utils.paths import PROJECT_ROOT


class ClassificationResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (PROJECT_ROOT/'reports/classification_round1/results.json').exists():
            raise unittest.SkipTest('Execute frozen classification experiments first.')
        cls.verification = verify_results()

    def test_all_actual_vectors_metrics_and_thresholds_reconcile(self):
        a = self.verification
        self.assertEqual(a['status'], 'PASS')
        self.assertEqual(a['evaluation_jobs_verified'], 56)
        self.assertEqual(a['policy_window_metrics_verified'], 112)
        self.assertEqual(a['thresholds_recomputed_independently'], 36)
        self.assertEqual(a['family_policy_model_aggregates_verified'], 28)
        self.assertLess(a['independent_metric_max_absolute_difference'], 1e-12)

    def test_representative_refit_and_regression_protection(self):
        receipt = verify_freeze()
        self.assertEqual(self.verification['regression_protected_files_unchanged'], len(receipt['regression_protected_sha256']))
        repeat = self.verification['representative_refit']
        self.assertEqual(repeat['status'], 'PASS')
        self.assertEqual(repeat['fresh_model_fits'], 13)
        self.assertEqual(repeat['outer_vectors_bit_identical'], 7)
        self.assertEqual(repeat['inner_vectors_bit_identical'], 6)
        self.assertEqual(repeat['threshold_records_identical'], 7)

    def test_executed_notebook_and_review_stop(self):
        path = PROJECT_ROOT/'notebooks/samuel/05_samuel_classification_round1.ipynb'
        if not path.exists(): self.skipTest('Build and execute classification notebook first.')
        n = nbformat.read(path, as_version=4); nbformat.validate(n)
        code = [c for c in n.cells if c.cell_type == 'code']
        self.assertEqual([c.execution_count for c in code], list(range(1, len(code)+1)))
        self.assertFalse(any(o.output_type == 'error' for c in code for o in c.outputs))
        self.assertGreaterEqual(sum('image/png' in o.get('data', {}) for c in code for o in c.outputs), 6)
        prose = '\n'.join(c.source for c in n.cells if c.cell_type == 'markdown')
        self.assertIn('belum', prose.lower())
        self.assertIn('Round 2', prose)


if __name__ == '__main__': unittest.main()
