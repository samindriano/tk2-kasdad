import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.models.round1_models import EXPANDED_FEATURES
from src.models.regression_round2_metrics import (
    descriptive_test_risk,
    history_summary,
    metric_values,
    paired,
    summarize,
)


def mini_frame():
    return pd.DataFrame({
        "record_id": ["secret-a", "secret-b", "secret-c", "secret-d", "secret-e", "secret-f"],
        "threat_rank": [0.0, 5.0, 33.3333333333, 66.6666666667, 95.0, 100.0],
        "cycle_code": [47, 47, 48, 48, 49, 49],
        "entry_friction": [0, 1, 0, 1, 0, 1],
        "signal_stream": [None, "", "x", "y", None, "z"],
        "impact_load": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
        "scope_pressure": [1, 2, 3, 4, 5, 6],
        "text_load": [1, 2, 3, 4, 5, 6],
        "scope_trace": ["~1", "2 nodes", "3", "4", "5", "6"],
        "org_spread": ["1", "2", "3", "4", "5", "6"],
        "module_spread": ["1", "2", "3", "4", "5", "6"],
        "telemetry_flag": ["active", "inactive", "Y", "N", None, "mystery"],
        "org_code": ["a", "a", None, "b", "c", "c"],
        "module_code": ["m1", "m2", "m1", "m2", "m3", "m3"],
        "pattern_code": ["p1", "p1", "p2", "p2", "p3", "p3"],
    })


class RegressionRound2MetricsTests(unittest.TestCase):
    def test_metrics_match_sklearn_and_signed_error_convention(self):
        y = np.array([0., 10., 20., 30.])
        pred = np.array([2., 9., 22., 20.])
        got = metric_values(y, pred)
        np.testing.assert_allclose(got["mae"], mean_absolute_error(y, pred))
        np.testing.assert_allclose(got["rmse"], mean_squared_error(y, pred) ** .5)
        self.assertAlmostEqual(got["signed_error"], np.mean(pred - y))
        self.assertAlmostEqual(got["over_rate"] + got["under_rate"] + got["equal_rate"], 1.)
        self.assertEqual(metric_values([], []) ["mae"], None)

    def test_rejects_misaligned_and_nonfinite_metric_inputs(self):
        with self.assertRaises(ValueError):
            metric_values([1], [1, 2])
        with self.assertRaises(ValueError):
            metric_values([np.nan], [0])

    def test_pairing_uses_same_records_and_signs(self):
        frame = mini_frame()
        y = frame.threat_rank.to_numpy()
        control = y + np.array([2, 2, 2, 2, 2, 2.])
        candidate = y + np.array([1, 1, 3, 3, 2, 2.])
        got = paired(frame, candidate, control)
        self.assertEqual(got["n"], 6)
        self.assertAlmostEqual(got["delta_mae"], 0.)
        self.assertAlmostEqual(got["candidate_win_rate"], 2 / 6)
        self.assertAlmostEqual(got["control_win_rate"], 2 / 6)
        self.assertAlmostEqual(got["tie_rate"], 2 / 6)
        self.assertEqual(len(got["per_cycle"]), 3)
        self.assertLess(got["per_cycle"][0]["delta_mae"], 0)

    def test_summary_equal_cycle_weights_and_tail_boundaries(self):
        frame = pd.DataFrame({
            "threat_rank": [5, 5, 5, 95, 33.3333333333, 66.6666666667],
            "cycle_code": [1, 1, 1, 2, 3, 3], "entry_friction": [0, 0, 0, 1, 0, 1],
            "signal_stream": [None, "", "x", "y", None, "z"], "impact_load": [1, 1, 1, 2, 3, 3],
        })
        pred = np.array([10., 10., 10., 100., 33.3333333333, 66.6666666667])
        got = summarize(frame, pred)
        self.assertEqual(got["equal_cycle_mae"], np.mean([5., 5., 0.]))
        self.assertEqual(got["raw"]["mae"], np.mean([5., 5., 5., 5., 0., 0.]))
        self.assertEqual(got["subgroups"]["tail_low_le5"]["n"], 3)
        self.assertEqual(got["subgroups"]["tail_high_ge95"]["n"], 1)
        self.assertEqual(got["subgroups"]["stream_missing"]["n"], 3)
        self.assertEqual(got["subgroups"]["stream_present"]["n"], 3)
        self.assertEqual(got["impact_groups"]["1.0"]["by_target_region"]["region_low"]["n"], 3)

    def test_target_region_boundaries(self):
        ys = [0, 100/3, 200/3, 100]
        pred = ys.copy()
        frame = pd.DataFrame({"threat_rank": ys, "cycle_code": [1, 1, 2, 2],
                              "entry_friction": [0, 1, 0, 1], "signal_stream": [None, "x", None, "x"],
                              "impact_load": [1, 2, 3, 4]})
        got = summarize(frame, pred)
        self.assertEqual([got["subgroups"][k]["n"] for k in ("region_low", "region_middle", "region_high")], [1, 1, 2])

    def test_impact_group_detail_can_be_omitted_without_changing_schema(self):
        frame = mini_frame()
        got = summarize(frame, frame.threat_rank.to_numpy(), include_impact=False)
        self.assertEqual(got["impact_groups"], {})
        self.assertEqual(got["raw"]["mae"], 0.)

    def test_history_summary_has_no_labels_or_raw_ids_and_counts_unknown_separately(self):
        frame = mini_frame()
        got = history_summary(frame, [0, 1], [2, 3])
        self.assertNotIn("secret-a", repr(got))
        self.assertNotIn("threat_rank", repr(got))
        self.assertEqual(got["category_support"]["org_code"]["fit_missing_n"], 0)
        self.assertEqual(got["category_support"]["org_code"]["validation_missing_n"], 1)
        self.assertEqual(got["category_support"]["org_code"]["validation_unknown_n"], 1)
        self.assertEqual(len(got["fit_source_rows_sha256"]), 64)
        frame2 = frame.copy()
        frame2["threat_rank"] += 1000
        self.assertEqual(got, history_summary(frame2, [0, 1], [2, 3]))

    def test_history_summary_rejects_temporal_overlap(self):
        frame = mini_frame()
        with self.assertRaises(ValueError):
            history_summary(frame, [0, 2], [1, 3])
        with self.assertRaises(ValueError):
            history_summary(frame, [0, 1], [1, 2])

    def test_history_row_selectors_require_integer_position_dtype(self):
        frame = mini_frame()
        valid = history_summary(frame, np.array([0, 1], dtype=np.int32), np.array([2, 3], dtype=np.int64))
        self.assertEqual(valid["fit_rows"], 2)
        for fit in ([0.9, 1.9], [0.0, 1.0], np.array([True, False]), [np.nan, 1.0], ["0", "1"]):
            with self.subTest(fit=fit), self.assertRaises(ValueError):
                history_summary(frame, fit, [2, 3])

    def test_test_risk_is_target_blind_and_reports_unknowns_and_cycle_gap(self):
        train = mini_frame()
        test = mini_frame().iloc[:2].copy()
        test["cycle_code"] = [50, 51]
        test["org_code"] = ["new", None]
        test["scope_trace"] = ["~7 nodes", "bad value"]
        first = descriptive_test_risk(train, test)
        test["threat_rank"] = [9999, -9999]
        second = descriptive_test_risk(train, test)
        self.assertEqual(first, second)
        self.assertEqual(first["features"]["org_code"]["test_unknown_n"], 1)
        self.assertEqual(first["features"]["org_code"]["test_nonmissing_n"], 1)
        self.assertEqual(first["features"]["scope_trace"]["test_parse_fail_n"], 1)
        self.assertEqual(first["cycle_gap"]["test_outside_train_range_n"], 2)
        self.assertFalse(first["fit_performed"])
        self.assertFalse(first["predictions_made"])
        self.assertNotIn("9999", repr(second))

    def test_risk_summary_rejects_missing_predictor_schema(self):
        with self.assertRaises(ValueError):
            descriptive_test_risk(pd.DataFrame(), pd.DataFrame())


if __name__ == "__main__":
    unittest.main()
