"""Format and boundary checks for the authorized initial-submission path."""
import unittest

import numpy as np
import pandas as pd

from scripts.prepare_initial_submissions import _submission_frame


class InitialSubmissionTests(unittest.TestCase):
    def setUp(self):
        self.test = pd.DataFrame({"record_id": ["A", "B", "C"]})
        self.sample = pd.DataFrame({"record_id": ["A", "B", "C"], "prediction": [0, 0, 0]})

    def test_binary_output_integer_and_order(self):
        sample = self.sample.rename(columns={"prediction": "artifact_signal"})
        out = _submission_frame(self.test, sample, np.array([0, 1, 0]), "artifact_signal", "classification")
        self.assertEqual(list(out.columns), ["record_id", "artifact_signal"])
        self.assertEqual(out.artifact_signal.tolist(), [0, 1, 0])
        self.assertTrue(pd.api.types.is_integer_dtype(out.artifact_signal))

    def test_clipped_regression_contract(self):
        sample = self.sample.rename(columns={"prediction": "threat_rank"})
        out = _submission_frame(self.test, sample, np.array([0., 25.5, 100.]), "threat_rank", "regression")
        self.assertTrue(np.isfinite(out.threat_rank).all())
        self.assertTrue(out.threat_rank.between(0, 100).all())

    def test_invalid_labels_and_nonfinite_regression_rejected(self):
        sample_cls = self.sample.rename(columns={"prediction": "artifact_signal"})
        sample_reg = self.sample.rename(columns={"prediction": "threat_rank"})
        with self.assertRaises(ValueError):
            _submission_frame(self.test, sample_cls, np.array([0, 2, 0]), "artifact_signal", "classification")
        with self.assertRaises(ValueError):
            _submission_frame(self.test, sample_reg, np.array([0., np.nan, 100.]), "threat_rank", "regression")

    def test_test_id_order_must_match_sample(self):
        sample = self.sample.iloc[::-1].rename(columns={"prediction": "threat_rank"})
        with self.assertRaises(ValueError):
            _submission_frame(self.test, sample, np.array([10., 20., 30.]), "threat_rank", "regression")


if __name__ == "__main__":
    unittest.main()
