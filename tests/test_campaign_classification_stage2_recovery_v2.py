"""Regression tests for the corrected nested report metric accessors."""
import unittest

from scripts.assemble_campaign_classification_stage2_report_v2 import plot_series
from scripts.research_campaign_classification_stage2 import CANDIDATES, PRIMARY_WINDOWS


class ClassificationRecoveryPlotTests(unittest.TestCase):
    def test_plot_series_reads_candidate_and_control_from_nested_window_records(self):
        report = {"window_metrics": {}}
        for index, window in enumerate(PRIMARY_WINDOWS):
            report["window_metrics"][window] = {}
            for candidate_index, recipe in enumerate(CANDIDATES):
                report["window_metrics"][window][recipe] = {
                    "candidate": {"macro_f1": 0.70 + index / 100 + candidate_index / 1000,
                                  "average_precision": 0.75 + index / 100 + candidate_index / 1000},
                    "C4_HGB": {"macro_f1": 0.71 + index / 100,
                               "average_precision": 0.76 + index / 100},
                }
        result = plot_series(report)
        self.assertEqual(result["macro_f1"]["W1/C4_HGB"], 0.71)
        self.assertEqual(result["macro_f1"]["W6/DC_NBSVM"], 0.751)
        self.assertEqual(result["average_precision"]["W6/DC_WORD_CHAR"], 0.802)


if __name__ == "__main__":
    unittest.main()
