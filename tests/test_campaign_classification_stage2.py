"""Unit tests for fixed Stage 2 metric and error-complementarity rules."""
import unittest

import numpy as np

from scripts.research_campaign_classification_stage2 import (
    blend_gate,
    compact_metrics,
    complementarity_counts,
)


class ClassificationCampaignStage2Tests(unittest.TestCase):
    def test_complementarity_gate_requires_two_sided_nonnegative_net_gain(self):
        y = np.tile(np.array([0, 0, 1, 1, 0, 1, 1], dtype=np.int8), 25)
        c4 = np.tile(np.array([0.1, 0.8, 0.3, 0.8, 0.4, 0.9, 0.6]), 25)
        candidate = np.tile(np.array([0.1, 0.1, 0.7, 0.8, 0.6, 0.9, 0.6]), 25)
        counts = complementarity_counts(y, candidate, c4)
        self.assertEqual(counts["candidate_only_correct"], 50)
        self.assertEqual(counts["C4_only_correct"], 25)
        self.assertEqual(counts["net_candidate_correct_count_delta"], 25)
        self.assertTrue(blend_gate(counts))

        negative_net = dict(counts, net_candidate_correct_count_delta=-1)
        self.assertFalse(blend_gate(negative_net))
        too_few = dict(counts, candidate_only_correct=24)
        self.assertFalse(blend_gate(too_few))

    def test_fixed_threshold_metrics_include_ranking_and_brier(self):
        y = np.array([0, 1, 0, 1], dtype=np.int8)
        probability = np.array([0.1, 0.6, 0.45, 0.8])
        metrics = compact_metrics(y, probability)
        self.assertEqual(metrics["threshold"], 0.50)
        self.assertEqual(metrics["confusion_matrix"], [[2, 0], [0, 2]])
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertGreater(metrics["average_precision"], 0.99)
        self.assertAlmostEqual(metrics["brier_score"], np.mean((probability - y) ** 2))


if __name__ == "__main__":
    unittest.main()
