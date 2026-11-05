"""Synthetic metric checks for the source-frozen R06 center-local transfer."""
from __future__ import annotations

import unittest

import numpy as np

from scripts import reg_r06_center_local_v1 as runner


class TestFixedMechanism(unittest.TestCase):
    def test_local_basis_and_weighted_lower_median_are_deterministic(self) -> None:
        direct = np.asarray([40.0, 45.0, 50.0, 55.0, 60.0, 61.0])
        np.testing.assert_array_equal(runner.local_basis(direct),
                                      np.asarray([0.0, 0.5, 1.0, 0.5, 0.0, 0.0]))
        interval = runner.lower_weighted_median_interval(
            np.asarray([-3.0, -1.0, -1.0, 2.0]), np.asarray([1.0, 1.0, 1.0, 1.0]))
        self.assertEqual(interval, (-1.0, -1.0, 4))
        self.assertEqual(runner._clip_interval(-5.0, 2.0), (-2.0, 0.0))

    def test_synthetic_score_metric_and_gate_replay(self) -> None:
        params = {"intercept": 3.0, "slope_low": 0.90, "slope_high": 1.02}
        pattern = np.asarray([0, 1, 3, 10, 20, 25, 39, 40, 45, 48, 49, 50,
                              51, 55, 60, 61, 70, 80, 90, 99, 100], dtype="<f8")
        results = {}
        for stage, count, cycle_start in (("HINGE_D1", runner.SCORE_COUNTS["HINGE_D1"], 72),
                                          ("HINGE_D2", runner.SCORE_COUNTS["HINGE_D2"], 76)):
            direct = np.resize(pattern, count).astype("<f8")
            cycles = (cycle_start + np.arange(count) % 4).astype(np.int16)
            missing = (np.arange(count) % 13 == 0)
            base = runner.base_hinge_unclipped(direct, params)
            transported = np.clip(base, 0.0, 100.0)
            local, _ = runner.apply_correction(direct, missing, params, -0.8, local=True)
            control, _ = runner.apply_correction(direct, missing, params, -0.4, local=False)
            target = local.copy()
            target[missing] = direct[missing]
            predictions = {"TRANSPORTED_R4_ON_R06": transported,
                "LOCAL": local, "GLOBAL_CONTROL": control,
                "ORIGINAL_FULL_DIRECT_HINGE": transported.copy(), "RAW_R06": direct.copy()}
            results[stage] = {"metrics": runner._population_metrics(
                direct, cycles, missing, target, predictions),
                "targeted_region": runner._targeted_region_gain(
                    direct, ~missing, target, predictions)}
        fits = {"D1_LOCAL": {"fit": {"delta": -0.8}},
                "D2_LOCAL": {"fit": {"delta": -0.8}}}
        gates = runner._score_gates(results, fits)
        self.assertTrue(gates["mechanism_gate"]["pass"])
        self.assertTrue(gates["candidate_gate"]["pass"])
        self.assertTrue(gates["exploratory_review_gate"]["pass"])
        self.assertLess(gates["cycle_74_harm_vs_transported"], 0.0)
        self.assertEqual(gates["score_rows_pooled_unique"], 15_953)
        cycle74 = next(row for row in results["HINGE_D1"]["metrics"]["by_cycle"]
                       if row["cycle"] == 74)
        self.assertEqual(cycle74["metrics"]["all_rows"]["LOCAL"]["status"], "SUPPORTED")
        tail_keys = set(results["HINGE_D1"]["metrics"]["target_tails"])
        self.assertEqual(tail_keys, {"target_le_5", "target_ge_95"})
        self.assertIn("p90_absolute_error",
                      results["HINGE_D1"]["metrics"]["target_tails"]["target_ge_95"]["metrics"]["all_rows"]["LOCAL"])


if __name__ == "__main__":
    unittest.main()
