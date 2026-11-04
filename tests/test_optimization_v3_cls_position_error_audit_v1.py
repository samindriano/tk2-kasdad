from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import optimization_v3_cls_position_error_audit_v1 as audit


class PositionErrorAuditTests(unittest.TestCase):
    def test_fixed_margin_bin_boundaries(self) -> None:
        margins = np.array([0.0, 0.019999, 0.02, 0.049999, 0.05,
                            0.099999, 0.10, 0.199999, 0.20, 0.50])
        np.testing.assert_array_equal(
            audit.margin_bin_indices(margins),
            np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]),
        )

    def test_paired_flip_counts_keep_true_class_and_error_deltas(self) -> None:
        target = np.array([0, 0, 1, 1, 0, 1])
        baseline = np.array([0.1, 0.9, 0.1, 0.9, 0.2, 0.8])
        candidate = np.array([0.9, 0.1, 0.9, 0.1, 0.2, 0.8])

        counts = audit.paired_flip_counts(target, baseline, candidate)

        self.assertEqual(counts["n"], 6)
        self.assertEqual(counts["unchanged_labels"], 2)
        self.assertEqual(counts["flipped_labels"], 4)
        self.assertEqual(counts["wrong_to_correct"], 2)
        self.assertEqual(counts["correct_to_wrong"], 2)
        self.assertEqual(counts["wrong_to_correct_y0"], 1)
        self.assertEqual(counts["wrong_to_correct_y1"], 1)
        self.assertEqual(counts["correct_to_wrong_y0"], 1)
        self.assertEqual(counts["correct_to_wrong_y1"], 1)
        self.assertEqual(counts["false_positive_delta"], 0)
        self.assertEqual(counts["false_negative_delta"], 0)

    def test_exact_row_validation_rejects_misordered_ids(self) -> None:
        with self.assertRaises(ValueError):
            audit.validate_aligned_rows(
                np.array(["r1", "r2"]), np.array(["r2", "r1"]),
                np.array([76, 77]), np.array([77, 76]),
            )


if __name__ == "__main__":
    unittest.main()
