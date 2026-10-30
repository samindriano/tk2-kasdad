"""Synthetic checks for independent regression-research verification helpers."""
from __future__ import annotations

import copy
import math
import unittest

import numpy as np
import pandas as pd

from scripts.verify_regression_research import (
    VerificationError,
    _metric_values,
    _rank_correlation,
    _fit_fingerprint_from_selected,
    _selected_inner_choice,
    average_percentile,
    independent_summary,
    map_policy,
    verify_summary_node,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame({
        "threat_rank": [0.0, 5.0, 33.3333, 33.3334, 66.6666, 66.6667, 95.0, 100.0],
        "cycle_code": [56, 56, 56, 56, 57, 57, 57, 57],
        "entry_friction": [0, 1, 0, 1, 0, 1, 0, 1],
        "signal_stream": [None, "", "a", "b", "c", "d", "e", "f"],
    })


class RegressionResearchVerificationTests(unittest.TestCase):
    def test_average_ranks_are_cycle_local_and_average_ties(self):
        prediction = np.array([4.0, 4.0, 10.0, 10.0, -5.0, 205.0])
        cycles = np.array([56, 56, 56, 57, 57, 57])
        np.testing.assert_array_equal(
            average_percentile(prediction, cycles),
            np.array([50.0, 50.0, 100.0, 200.0 / 3.0, 100.0 / 3.0, 100.0]),
        )

    def test_mapping_policy_preserves_raw_clip_and_unclipped_blend_order(self):
        prediction = np.array([-5.0, 4.0, 4.0, 15.0])
        cycles = np.array([1, 1, 1, 2])
        np.testing.assert_array_equal(map_policy(prediction, cycles, "raw"), prediction)
        np.testing.assert_array_equal(map_policy(prediction, cycles, "clip"), [0.0, 4.0, 4.0, 15.0])
        ranks = np.array([100.0 / 3.0, 250.0 / 3.0, 250.0 / 3.0, 100.0])
        expected = np.clip(0.75 * prediction + 0.25 * ranks, 0, 100)
        np.testing.assert_array_equal(map_policy(prediction, cycles, "rank25"), expected)
        # There is deliberately no target parameter in mapping APIs.
        unrelated_targets = np.array([100.0, 0.0, 50.0, 50.0])
        self.assertEqual(unrelated_targets.shape, map_policy(prediction, cycles, "rank25").shape)

    def test_fsum_metrics_and_boundaries(self):
        y = np.array([0.0, 5.0, 33.3333, 33.3334, 66.6666, 66.6667, 95.0, 100.0])
        prediction = np.array([-1.0, 4.0, 35.0, 32.0, 70.0, 70.0, 80.0, 105.0])
        stats = _metric_values(y, prediction)
        absolute_error = np.abs(prediction - y)
        self.assertEqual(stats["n"], 8)
        self.assertEqual(stats["mae"], math.fsum(map(float, absolute_error)) / 8)
        self.assertEqual(stats["signed_error"], math.fsum(map(float, prediction - y)) / 8)
        self.assertEqual(stats["below_zero"], 1)
        self.assertEqual(stats["above_100"], 1)

    def test_summary_verifier_rejects_changed_metrics_subgroup_count_and_spearman(self):
        frame = _frame()
        prediction = np.array([-1.0, 4.0, 35.0, 32.0, 70.0, 70.0, 80.0, 105.0])
        saved = independent_summary(frame, prediction)
        result = verify_summary_node(frame, prediction, saved, "synthetic")
        self.assertGreater(result["metric_nodes"], 0)
        self.assertEqual(result["cycle_nodes"], 2)
        self.assertEqual(result["subgroup_nodes"], 9)
        self.assertEqual(saved["subgroups"]["tail_low_le5"]["n"], 2)
        self.assertEqual(saved["subgroups"]["tail_high_ge95"]["n"], 2)
        self.assertEqual(saved["subgroups"]["stream_missing"]["n"], 2)

        bad_mae = copy.deepcopy(saved)
        bad_mae["raw"]["mae"] += 0.01
        with self.assertRaises(VerificationError):
            verify_summary_node(frame, prediction, bad_mae, "tampered_mae")

        bad_count = copy.deepcopy(saved)
        bad_count["subgroups"]["tail_low_le5"]["n"] += 1
        with self.assertRaises(VerificationError):
            verify_summary_node(frame, prediction, bad_count, "tampered_tail_count")

        bad_rho = copy.deepcopy(saved)
        bad_rho["pooled_spearman"] = 0.0
        with self.assertRaises(VerificationError):
            verify_summary_node(frame, prediction, bad_rho, "tampered_spearman")

    def test_rank_correlation_respects_ties(self):
        self.assertAlmostEqual(_rank_correlation([1, 1, 2], [1, 2, 3]), math.sqrt(3) / 2)
        self.assertEqual(_rank_correlation([1, 2, 3], [3, 2, 1]), -1.0)
        self.assertIsNone(_rank_correlation([1, 1], [2, 3]))

    def test_impact_support_counts_and_region_counts_are_verified(self):
        frame = _frame().assign(impact_load=[10.0, 10.0, 20.0, 20.0, 30.0, 30.0, 40.0, 40.0])
        prediction = np.array([-1.0, 4.0, 35.0, 32.0, 70.0, 70.0, 80.0, 105.0])
        saved = independent_summary(frame, prediction)
        y = frame.threat_rank.to_numpy()
        regions = {
            "region_low": y < 100.0 / 3.0,
            "region_middle": (y >= 100.0 / 3.0) & (y < 200.0 / 3.0),
            "region_high": y >= 200.0 / 3.0,
        }
        for value in sorted(frame.impact_load.unique()):
            mask = frame.impact_load.to_numpy() == value
            saved["impact_groups"][repr(float(value))] = {
                "impact_load": float(value),
                "metrics": {"n": int(mask.sum())},
                "by_target_region": {
                    name: {"n": int(np.count_nonzero(mask & region_mask))}
                    for name, region_mask in regions.items()
                },
            }
        checked = verify_summary_node(frame, prediction, saved, "synthetic_impact", impact_mode="counts")
        self.assertEqual(checked["impact_group_count_nodes"], 16)
        saved["impact_groups"]["10.0"]["by_target_region"]["region_low"]["n"] += 1
        with self.assertRaises(VerificationError):
            verify_summary_node(frame, prediction, saved, "tampered_impact", impact_mode="counts")

    def test_stage4_constructed_stream_fingerprint_tracks_constant_and_target(self):
        features = ("impact_load", "signal_stream")
        selected = pd.DataFrame({"impact_load": [1.0, 2.0], "signal_stream": ["", ""]})
        target = np.array([10.0, 20.0])
        baseline = _fit_fingerprint_from_selected(selected, target, features)
        selected_changed = selected.copy()
        selected_changed.loc[0, "signal_stream"] = "changed"
        self.assertNotEqual(baseline, _fit_fingerprint_from_selected(selected_changed, target, features))
        self.assertNotEqual(baseline, _fit_fingerprint_from_selected(selected, target + 1, features))

    def test_inner_selection_ties_prefer_lower_weight_then_regularized_complexity(self):
        selected = _selected_inner_choice([
            {"param": "base", "rank_weight": 0.5, "mae": 10.0},
            {"param": "deeper", "rank_weight": 0.25, "mae": 10.0},
            {"param": "base", "rank_weight": 0.25, "mae": 10.0},
            {"param": "regularized", "rank_weight": 0.25, "mae": 10.0},
        ])
        self.assertEqual(selected, {"param": "regularized", "rank_weight": 0.25, "mae": 10.0})


if __name__ == "__main__":
    unittest.main()
