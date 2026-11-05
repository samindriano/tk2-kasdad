"""Candidate-blind synthetic checks for REG_MONOTONE_HINGE_P50_V1."""
from __future__ import annotations

import unittest

from scripts.verify_reg_monotone_hinge_p50_v1 import run_synthetic_checks


class FixedKnotHingeSyntheticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.checks = run_synthetic_checks()

    def assert_check(self, name: str) -> None:
        self.assertIn(name, self.checks)
        self.assertTrue(self.checks[name]["pass"], self.checks[name])

    def test_known_continuous_hinge_recovery(self) -> None:
        self.assert_check("known_hinge_recovery")

    def test_equal_slopes_are_affine_and_knot_is_continuous(self) -> None:
        self.assert_check("equal_slope_affine_and_continuity")

    def test_monotone_slopes_and_bounds(self) -> None:
        self.assert_check("monotonicity_and_slope_bounds")

    def test_objective_is_unweighted_and_has_no_weights_argument(self) -> None:
        self.assert_check("unweighted_objective_only")

    def test_identity_and_order_gate_runs_before_target_loader(self) -> None:
        self.assert_check("identity_before_target_access")

    def test_stored_global_control_replays_exactly(self) -> None:
        self.assert_check("stored_global_control_replay")

    def test_causal_stage_cycle_and_origin_boundaries(self) -> None:
        self.assert_check("causal_stage_boundaries")

    def test_published_fit_is_recovered_after_postpublish_crash(self) -> None:
        self.assert_check("postpublish_crash_recovery")

    def test_direct_and_tagged_hash_schemas_are_distinct(self) -> None:
        self.assert_check("producer_hash_schemas")

    def test_csv_order_range_and_missing_route(self) -> None:
        self.assert_check("csv_ids_range_and_missing_route")


if __name__ == "__main__":
    unittest.main()
