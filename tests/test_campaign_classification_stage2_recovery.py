"""Unit checks for report-only metric reconstruction."""
import unittest

import numpy as np
import pandas as pd

from scripts.assemble_campaign_classification_stage2 import _score


class ClassificationStage2RecoveryTests(unittest.TestCase):
    def test_scoring_reconstruction_keeps_fixed_threshold_and_cycle_means(self):
        frame = pd.DataFrame({"cycle_code": [56, 56, 57, 57]})
        rows = np.arange(4)
        actual = np.array([0, 1, 0, 1], dtype=np.int8)
        probability = np.array([0.1, 0.7, 0.4, 0.8])
        report = _score(frame, actual, rows, probability)
        self.assertEqual(report["threshold"], 0.50)
        self.assertEqual(report["n"], 4)
        self.assertEqual(report["confusion_matrix"], [[2, 0], [0, 2]])
        self.assertEqual(report["macro_f1"], 1.0)
        self.assertEqual(report["equal_cycle_macro_f1"], 1.0)
        self.assertEqual([item["cycle"] for item in report["per_cycle"]], [56, 57])
        self.assertGreater(report["average_precision"], 0.99)


if __name__ == "__main__":
    unittest.main()
