"""Focused boundary tests for Samuel's formal EDA A calculations."""
import unittest

import numpy as np
import pandas as pd

from src.data.samuel_eda_a import (
    derive_features,
    exact_cycle_comparison,
    exact_length_comparison,
    period_comparison,
)


def _comparison_frame(cycle_code=47):
    """Two same-length strata with tied diversity tails and unequal tail sizes."""
    rows = []
    cases = [
        (4, [1, 1, 2, 3, 4, 4, 4, 4], [0, 0, 0, 0, 1, 1, 1, 1]),
        (8, [1, 1, 1, 2, 3, 4, 4, 5], [1, 1, 1, 0, 0, 0, 0, 0]),
    ]
    for token_count, unique_counts, labels in cases:
        rows.extend({
            "token_count": token_count,
            "unique_tokens": unique,
            "ttr": unique / token_count,
            "stream_missing": False,
            "artifact_signal": label,
            "cycle_code": cycle_code,
        } for unique, label in zip(unique_counts, labels))
    return pd.DataFrame(rows)


class SamuelEdaATests(unittest.TestCase):
    def test_tokenization_lowercases_and_distinguishes_missing_from_untokenizable(self):
        source = pd.DataFrame({
            "record_id": [11, 12, 13, 14],
            "signal_stream": ["Aa aa, B2! b_2", np.nan, "   ", "!!!"],
            "artifact_signal": [0, 1, 0, 1],
            "cycle_code": [47, 48, 49, 50],
        })

        features = derive_features(source)

        self.assertEqual(list(features.columns), [
            "token_count", "unique_tokens", "ttr", "stream_missing",
            "artifact_signal", "cycle_code",
        ])
        self.assertNotIn("record_id", features.columns)
        self.assertNotIn("signal_stream", features.columns)
        self.assertEqual(features.loc[0, "token_count"], 5)
        self.assertEqual(features.loc[0, "unique_tokens"], 4)
        self.assertAlmostEqual(features.loc[0, "ttr"], 4 / 5)
        self.assertEqual(features.loc[1:2, "stream_missing"].tolist(), [True, True])
        self.assertTrue(features.loc[1:2, "ttr"].isna().all())
        self.assertFalse(features.loc[3, "stream_missing"])
        self.assertEqual(features.loc[3, "token_count"], 0)
        self.assertEqual(features.loc[3, "unique_tokens"], 0)
        self.assertTrue(np.isnan(features.loc[3, "ttr"]))

    def test_exact_length_ties_are_retained_and_target_does_not_select_support(self):
        original = _comparison_frame()
        table, summary = exact_length_comparison(original, min_rows=8, min_tail=1)
        flipped, _ = exact_length_comparison(
            original.assign(artifact_signal=1 - original.artifact_signal),
            min_rows=8,
            min_tail=1,
        )

        self.assertEqual(table["token_count"].tolist(), [4, 8])
        self.assertEqual(table["low_n"].tolist(), [2, 3])
        self.assertEqual(table["high_n"].tolist(), [4, 3])
        membership = ["token_count", "n", "low_n", "high_n"]
        pd.testing.assert_frame_equal(table[membership], flipped[membership])
        self.assertEqual(summary["support_rows"], 16)
        self.assertEqual(summary["positive_strata"], 1)
        self.assertEqual(summary["negative_strata"], 1)

    def test_exact_length_summary_uses_stratum_weights_not_pooled_tail_denominators(self):
        frame = _comparison_frame()
        table, summary = exact_length_comparison(frame, min_rows=8, min_tail=1)

        expected_stratum_low = np.average(table.low_rate, weights=table.n)
        expected_stratum_high = np.average(table.high_rate, weights=table.n)
        pooled_low = table.low_positives.sum() / table.low_n.sum()
        pooled_high = table.high_positives.sum() / table.high_n.sum()
        self.assertAlmostEqual(summary["low_rate"], expected_stratum_low)
        self.assertAlmostEqual(summary["high_rate"], expected_stratum_high)
        self.assertAlmostEqual(summary["low_rate"], 0.5)
        self.assertAlmostEqual(summary["high_rate"], 0.5)
        self.assertAlmostEqual(pooled_low, 0.6)
        self.assertAlmostEqual(pooled_high, 4 / 7)
        self.assertNotAlmostEqual(summary["low_rate"], pooled_low)
        self.assertNotAlmostEqual(summary["high_rate"], pooled_high)

    def test_period_summaries_cover_three_fixed_cycle_ranges_without_target_selection(self):
        frame = pd.concat([
            _comparison_frame(cycle_code=cycle)
            for cycle in [47, 58, 69]
        ], ignore_index=True)
        table, summary = period_comparison(frame, min_rows=8, min_tail=1)
        flipped, _ = period_comparison(
            frame.assign(artifact_signal=1 - frame.artifact_signal),
            min_rows=8,
            min_tail=1,
        )

        self.assertEqual(set(table.period), {"47–57", "58–68", "69–79"})
        self.assertEqual(len(table), 6)
        self.assertEqual(summary["pooled"]["support_rows"], 48)
        self.assertEqual(
            [item["period"] for item in summary["periods"]],
            ["47–57", "58–68", "69–79"],
        )
        self.assertTrue(all(item["support_rows"] == 16 for item in summary["periods"]))
        membership = ["token_count", "period", "n", "low_n", "high_n"]
        pd.testing.assert_frame_equal(
            table[membership].reset_index(drop=True),
            flipped[membership].reset_index(drop=True),
        )

    def test_exact_cycle_keeps_zero_covariance_constant_target_and_excludes_constant_feature(self):
        rows = []
        # Feature varies but target is constant: this supported cell has exact zero covariance.
        for unique in [1, 1, 2, 2, 3, 4]:
            rows.append((4, unique, 47, 1))
        # Target varies but the feature is constant: no within-cell slope is estimable.
        for label in [0, 1, 0, 1, 0, 1]:
            rows.append((4, 2, 48, label))
        frame = pd.DataFrame(rows, columns=[
            "token_count", "unique_tokens", "cycle_code", "artifact_signal",
        ])
        frame["ttr"] = frame.unique_tokens / frame.token_count
        frame["stream_missing"] = False

        retained, summary, _ = exact_cycle_comparison(frame, min_n=5)

        self.assertEqual(len(retained), 1)
        self.assertEqual((retained.iloc[0].token_count, retained.iloc[0].cycle_code), (4, 47))
        self.assertEqual(retained.iloc[0].n, 6)
        self.assertEqual(retained.iloc[0].distinct_ttr, 4)
        self.assertEqual(retained.iloc[0].sxy_ttr_artifact_within_cell, 0)
        self.assertEqual(retained.iloc[0].cell_slope_per_ttr_unit, 0)
        self.assertEqual(summary["eligible_cells"], 1)
        self.assertEqual(summary["excluded_insufficient_ttr_variation_cells"], 1)
        self.assertEqual(summary["common_within_cell_slope_per_0_1_ttr"], 0)

    def test_exact_cycle_integer_cross_products_keep_nonconstant_y_zero_covariance_exact(self):
        # For u=1..6 and y=[0,1,1,1,1,0], n*sum(u*y)-sum(u)*sum(y)=0.
        # Floating centered-dot arithmetic can leave a tiny signed residue here.
        frame = pd.DataFrame({
            "token_count": [6] * 6,
            "unique_tokens": [1, 2, 3, 4, 5, 6],
            "ttr": np.arange(1, 7, dtype=float) / 6,
            "stream_missing": [False] * 6,
            "artifact_signal": [0, 1, 1, 1, 1, 0],
            "cycle_code": [47] * 6,
        })

        retained, summary, _ = exact_cycle_comparison(frame, min_n=5)

        self.assertEqual(len(retained), 1)
        self.assertEqual(retained.iloc[0].sxy_ttr_artifact_within_cell, 0)
        self.assertEqual(summary["cell_direction"]["positive_cells"], 0)
        self.assertEqual(summary["cell_direction"]["negative_cells"], 0)
        self.assertEqual(summary["cell_direction"]["zero_cells"], 1)


if __name__ == "__main__":
    unittest.main()
