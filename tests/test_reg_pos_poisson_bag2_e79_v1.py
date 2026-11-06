"""Regression coverage for the frozen-parent E79 raw replay contract."""
from __future__ import annotations

import inspect
import unittest

import numpy as np

import scripts.optimization_v3_reg_current_r06_e79_exploratory_v1 as parent_runner
import scripts.reg_pos_poisson_bag2_e79_v1 as pilot


class ParentRawDigestRegression(unittest.TestCase):
    def test_parent_bytes_only_digest_is_accepted(self) -> None:
        raw = np.asarray([0.125, 9.5, 38.25, 100.0], dtype=np.float64)
        expected = parent_runner.sha_array(raw, dtype="<f8")
        self.assertNotEqual(expected, pilot._sha_array(raw, dtype="<f8"))
        self.assertEqual(pilot._assert_parent_raw_digest(parent_runner, raw, expected), expected)

    def test_wrong_parent_digest_is_rejected(self) -> None:
        raw = np.asarray([0.125, 9.5, 38.25, 100.0], dtype=np.float64)
        with self.assertRaises(pilot.SourceError):
            pilot._assert_parent_raw_digest(parent_runner, raw, "0" * 64)

    def test_direct_and_champion_parity_gates_remain_in_final_runner(self) -> None:
        source = inspect.getsource(pilot._execute_final)
        self.assertIn("if not np.array_equal(original_raw, archive", source)
        self.assertIn("if not np.array_equal(original_direct, archive", source)
        self.assertIn("frozen_champion = _apply_hinge(original_direct, missing, FINAL_HINGE)", source)
        self.assertIn("not np.array_equal(frozen_champion, champion_values)", source)

        result = pilot._synthetic_e2e()
        self.assertEqual(result["status"], "PASS_SYNTHETIC_FULL_CLI")
        self.assertTrue(result["original_direct_replay"])
        self.assertTrue(result["champion_csv_parity"])
        self.assertTrue(result["candidate_csv_round_trip"])
        self.assertEqual(result["official_fit_units_started"], 0)
        self.assertFalse(result["official_test_features_loaded"])


if __name__ == "__main__":
    unittest.main()
