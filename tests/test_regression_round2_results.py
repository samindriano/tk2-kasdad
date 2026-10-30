import json
import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.audit import sha256
from scripts.verify_regression_round2 import (
    REPORT,
    _oracle_metrics,
    _result_digest_binding,
    _verify_metric,
    _verify_summary,
    _assert_reference_vector,
)
from src.models.round1_evaluation import array_digest


class RegressionRound2VerifierUnitTests(unittest.TestCase):
    def test_committed_reference_hash_works_without_ignored_old_arrays(self):
        rows, pred = np.array([3, 7]), np.array([12., 48.])
        audit = {'prediction_sha256': array_digest(pred)}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'not_available.npz'
            _assert_reference_vector(rows, pred, rows, audit, path)
            with self.assertRaises(AssertionError):
                _assert_reference_vector(rows, pred+1, rows, audit, path)
            with self.assertRaises(AssertionError):
                _assert_reference_vector(rows[::-1], pred, rows, audit, path)

    def test_present_historical_array_must_match_committed_reference(self):
        rows, pred = np.array([3, 7]), np.array([12., 48.])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'reference.npz'
            np.savez(path, validation_rows=rows, prediction=pred+1)
            with self.assertRaises(AssertionError):
                _assert_reference_vector(rows, pred, rows, {'prediction_sha256': array_digest(pred)}, path)

    def setUp(self):
        self.frame = pd.DataFrame({
            "threat_rank": [0., 20., 40., 60., 80., 100.],
            "cycle_code": [1, 1, 2, 2, 3, 3],
            "entry_friction": [0, 1, 0, 1, 0, 1],
            "signal_stream": [None, "x", "", "y", None, "z"],
            "impact_load": [1., 1., 2., 2., 3., 3.],
        })
        self.pred = np.array([2., 18., 42., 60., 78., 100.])

    def test_independent_metric_oracle_matches_expected_values(self):
        got = _oracle_metrics(self.frame.threat_rank, self.pred)
        self.assertAlmostEqual(got["mae"], 1.3333333333333333)
        self.assertAlmostEqual(got["signed_error"], 0.)
        self.assertEqual(got["n"], 6)

    def test_altered_mae_is_rejected(self):
        expected = _oracle_metrics(self.frame.threat_rank, self.pred)
        altered = dict(expected, mae=expected["mae"] + 1)
        with self.assertRaises(AssertionError):
            _verify_metric(altered, expected, "unit.mae")

    def test_summary_requires_exact_subgroup_denominators(self):
        # Make a complete expected-like record, then corrupt a partition count.
        from src.models.regression_round2_metrics import summarize
        record = summarize(self.frame, self.pred, include_impact=False)
        record["subgroups"]["EF1"]["n"] -= 1
        with self.assertRaises(AssertionError):
            _verify_summary(record, self.frame, self.pred, impact=False, path="unit.partition")

    def test_empty_subgroup_has_zero_denominator_and_null_metrics(self):
        no_ef1 = self.frame[self.frame.entry_friction == 0].copy()
        record = __import__("src.models.regression_round2_metrics", fromlist=["summarize"]).summarize(
            no_ef1, self.pred[[0, 2, 4]], include_impact=False)
        self.assertEqual(record["subgroups"]["EF1"]["n"], 0)
        self.assertIsNone(record["subgroups"]["EF1"]["mae"])

    def test_receipt_binding_rejects_stale_result_or_manifest(self):
        base = {"status": "PASS", "freeze_signature": "f", "results_sha256": "r",
                "vector_manifest_sha256": "v", "fit_audit_manifest_sha256": "a"}
        self.assertTrue(_result_digest_binding(base, "f", "r", "v", "a"))
        self.assertFalse(_result_digest_binding(base, "f", "changed", "v", "a"))
        self.assertFalse(_result_digest_binding(base, "f", "r", "changed", "a"))
        self.assertFalse(_result_digest_binding(base, "f", "r", "v", "changed"))
        self.assertFalse(_result_digest_binding(base, "changed", "r", "v", "a"))


class RegressionRound2SavedOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        final_path = REPORT/"results.json"
        cls.results_path = final_path if final_path.exists() else REPORT/"results_base.json"
        if not cls.results_path.exists():
            raise unittest.SkipTest("Frozen complete Regression Round2 base results are not available yet")

    def test_complete_saved_outputs_reconcile_without_refitting(self):
        results = json.loads(self.results_path.read_text(encoding="utf-8"))
        receipt_path = self.results_path.parent/"verification.json"
        self.assertTrue(receipt_path.exists(), "run the independent verifier before the final test suite")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        self.assertEqual(receipt["status"], "PASS")
        self.assertTrue(_result_digest_binding(receipt, results["freeze_signature"], sha256(self.results_path),
                                               receipt["vector_manifest_sha256"], receipt["fit_audit_manifest_sha256"]))
        self.assertEqual(receipt["results_path"], str(self.results_path.resolve()))
        self.assertEqual(receipt["job_metric_checks"], len(results["jobs"]))
        self.assertEqual(receipt["aggregate_checks"], sum(map(len, results["aggregate"].values())))
        self.assertEqual(receipt["test_labels_read"], False)
        self.assertEqual(receipt["models_refit_by_verifier"], False)
        self.assertEqual(receipt["repeat_verification"]["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
