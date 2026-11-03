from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.special import expit

from scripts.optimization_v3_positional_text_diagnostic_v1 import (
    LEDGER,
    decision_from_partition,
    file_sha,
    fixed_k_mask,
    route_with_fallback,
    run_synthetic_checks,
    validate_binary_class_order,
)


class PositionalTextDiagnosticSyntheticTests(unittest.TestCase):
    def test_logit_partition_and_sigmoid_equality(self) -> None:
        base = sparse.csr_matrix([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
        pos = sparse.csr_matrix([[0.5, 0.0], [0.0, -1.0], [2.0, 1.0]])
        base_coef = np.asarray([0.3, -0.2])
        pos_coef = np.asarray([-0.4, 0.6])
        intercept = np.asarray([0.12])
        base_score, pos_score, total = decision_from_partition(base, pos, base_coef, pos_coef, intercept)
        expected = intercept[0] + np.asarray(base @ base_coef).ravel() + np.asarray(pos @ pos_coef).ravel()
        self.assertTrue(np.array_equal(total, expected))
        self.assertTrue(np.array_equal(expit(total), expit(expected)))
        self.assertEqual(len(base_score), 3)
        self.assertEqual(len(pos_score), 3)

    def test_empty_position_block(self) -> None:
        base = sparse.csr_matrix([[2.0, 0.0], [0.0, 1.0]])
        empty = sparse.csr_matrix((2, 0), dtype=np.float64)
        base_coef = np.asarray([0.25, -0.5])
        base_score, pos_score, total = decision_from_partition(base, empty, base_coef,
                                                                np.asarray([], dtype=np.float64), [-0.1])
        self.assertTrue(np.array_equal(pos_score, np.zeros(2)))
        self.assertTrue(np.array_equal(total, -0.1 + base_score))

    def test_class_order_must_be_zero_then_one(self) -> None:
        self.assertTrue(np.array_equal(validate_binary_class_order([0, 1]), [0, 1]))
        with self.assertRaises(ValueError):
            validate_binary_class_order([1, 0])

    def test_fixed_k_ties_allzero_and_allone(self) -> None:
        self.assertTrue(np.array_equal(fixed_k_mask([0.5, 0.5, 0.5], 2, [9, 3, 7]),
                                       [False, True, True]))
        self.assertFalse(fixed_k_mask([0.0, 0.0, 0.0], 0, [0, 1, 2]).any())
        self.assertTrue(fixed_k_mask([1.0, 1.0, 1.0], 3, [0, 1, 2]).all())

    def test_exact_missing_fallback(self) -> None:
        routed = route_with_fallback([True, False, False], [0.3, 0.4, 0.5], [0.8, 0.7, 0.6])
        self.assertTrue(np.array_equal(routed, [0.3, 0.7, 0.6]))
        self.assertTrue(np.array_equal(routed[1:], np.asarray([0.7, 0.6])))

    def test_temp_only_synthetic_contract_and_live_ledger_hash(self) -> None:
        before = file_sha(LEDGER)
        with tempfile.TemporaryDirectory(prefix="positional-diagnostic-test-") as temp_name:
            temp_root = Path(temp_name).resolve()
            self.assertTrue(temp_root.is_relative_to(Path(tempfile.gettempdir()).resolve()))
            temporary_ledger = temp_root / "budget.jsonl"
            temporary_ledger.write_text('{"test":"temporary"}\n', encoding="utf-8")
            self.assertTrue(temporary_ledger.is_relative_to(temp_root))
        result = run_synthetic_checks()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["live_ledger_sha256_before"], result["live_ledger_sha256_after"])
        self.assertEqual(file_sha(LEDGER), before)


if __name__ == "__main__":
    unittest.main()
