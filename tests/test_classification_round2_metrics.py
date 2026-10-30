"""Synthetic tests for Round 2 inner-only thresholds and evaluation pooling."""
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.models.classification_metrics import binary_metrics
from src.models.classification_round2_metrics import (
    _macro_f1_from_counts,
    evaluate_predictions,
    select_policies,
)
from src.models.round1_evaluation import array_digest


def threshold_config(grid=(0.1, 0.9), minimum_rows=4):
    return {
        "threshold_selection": {
            "grid": list(grid),
            "minimum_inner_validation_rows": minimum_rows,
        }
    }


def complete_protocol():
    return {
        "primary_windows": [
            {"id": "W1", "fit": [47, 55], "validation": [56, 59]},
            {"id": "W2", "fit": [47, 59], "validation": [60, 63]},
            {"id": "W3", "fit": [47, 63], "validation": [64, 67]},
            {"id": "W4", "fit": [47, 67], "validation": [68, 71]},
            {"id": "W5", "fit": [47, 71], "validation": [72, 75]},
            {"id": "W6", "fit": [47, 75], "validation": [76, 79]},
        ],
        "secondary_windows": [
            {"id": "H12_E", "fit": [47, 55], "validation": [56, 67]},
            {"id": "H12_L", "fit": [47, 67], "validation": [68, 79]},
        ],
        "recipes": [{"id": "C1_A"}],
        "threshold_policies": ["fixed_0.5", "inner_pooled", "inner_shrink50", "inner_equal_cycle"],
    }


def evaluation_frame():
    cycles = np.repeat(np.arange(47, 80), 2)
    index = np.arange(len(cycles))
    target = np.tile([0, 1], len(cycles) // 2)
    return pd.DataFrame({
        "record_id": index,
        "cycle_code": cycles,
        "artifact_signal": target,
        "impact_load": index.astype(float),
        "entry_friction": target,
        "signal_stream": [None if i % 5 == 0 else f"stream-{i % 3}" for i in index],
        "org_code": ["late-unseen" if cycle == 79 and row == 1 else f"org-{cycle % 3}"
                     for cycle, row in zip(cycles, target)],
        "module_code": [f"module-{cycle % 2}" for cycle in cycles],
        "pattern_code": [f"pattern-{cycle % 4}" for cycle in cycles],
    })


def prediction_groups(frame):
    probability = np.empty(len(frame), dtype=float)
    for cycle in frame.cycle_code.unique():
        rows = np.flatnonzero(frame.cycle_code.to_numpy() == cycle)
        # Even cycles are perfectly separated; odd cycles predict every row as 0.
        if cycle % 2 == 0:
            probability[rows] = [0.1, 0.9]
        else:
            probability[rows] = [0.1, 0.1]
    # Populated per fit end below from only that history's inner rows.
    groups = {}
    validation_end = {55: 67, 59: 63, 63: 67, 67: 79, 71: 75, 75: 79}
    for end, last in validation_end.items():
        rows = np.flatnonzero(frame.cycle_code.between(end + 1, last).to_numpy())
        inner_rows = np.flatnonzero(frame.cycle_code.between(max(47, end - 3), end).to_numpy())
        selection = select_policies(
            frame.iloc[inner_rows].artifact_signal.to_numpy(),
            probability[inner_rows],
            frame.iloc[inner_rows].cycle_code.to_numpy(),
            threshold_config(tuple(np.round(np.arange(0.1, 0.901, 0.01), 2)), 4),
        )
        groups[end] = {
            "C1_A": {
                "rows": rows,
                "probability": probability[rows],
                "inner_rows": inner_rows,
                "inner_probability": probability[inner_rows],
                "thresholds": selection,
                "meta": {"fit_rows": int(frame.cycle_code.between(47, end).sum())},
            }
        }
    return groups


class ClassificationRound2MetricsTests(unittest.TestCase):
    def test_threshold_ties_prefer_closest_to_half_then_lower(self):
        y = np.tile([0, 1], 4)
        p = np.tile([0.2, 0.8], 4)
        cycles = np.repeat([1, 2, 3, 4], 2)
        selected = select_policies(y, p, cycles, threshold_config((0.4, 0.6), 4))
        self.assertEqual(selected["inner_pooled"]["threshold"], 0.4)
        self.assertEqual(selected["inner_equal_cycle"]["threshold"], 0.4)
        self.assertEqual(selected["fixed_0.5"]["threshold"], 0.5)

    def test_fallbacks_cover_eligibility_rows_classes_and_cycles(self):
        y = np.array([0, 1, 0, 1])
        p = np.array([0.1, 0.9, 0.1, 0.9])
        cycles = np.array([1, 1, 2, 2])
        for kwargs in (
            {"eligible": False},
            {"config": threshold_config((0.1, 0.9), 5)},
            {"y": np.zeros(4, dtype=int)},
            {"cycles": np.array([1, 1, 2, 2])},
        ):
            with self.subTest(kwargs=kwargs):
                test_y = kwargs.get("y", y)
                test_cycles = kwargs.get("cycles", np.array([1, 2, 3, 4]))
                config = kwargs.get("config", threshold_config())
                selected = select_policies(test_y, p, test_cycles, config,
                                           eligible=kwargs.get("eligible", True))
                for policy in ("inner_pooled", "inner_shrink50", "inner_equal_cycle"):
                    self.assertEqual(selected[policy]["threshold"], 0.5)
                    self.assertTrue(selected[policy]["fallback"])
                    self.assertTrue(selected[policy]["reason"])
                self.assertEqual(selected["fixed_0.5"]["threshold"], 0.5)
                self.assertFalse(selected["fixed_0.5"]["fallback"])

    def test_pooled_and_equal_cycle_objectives_can_choose_different_cutoffs(self):
        # One large, low-score cycle favors .3; three small high-score cycles
        # favor .7. The equal-cycle policy gives each cycle one vote.
        y_large = np.tile([0, 1], 100)
        p_large = np.tile([0.2, 0.5], 100)
        cycles_large = np.full(len(y_large), 1)
        y_small = np.tile([0, 1], 3)
        p_small = np.tile([0.5, 0.8], 3)
        cycles_small = np.repeat([2, 3, 4], 2)
        y = np.concatenate([y_large, y_small])
        p = np.concatenate([p_large, p_small])
        cycles = np.concatenate([cycles_large, cycles_small])
        selected = select_policies(y, p, cycles, threshold_config((0.3, 0.7), 200))
        self.assertEqual(selected["inner_pooled"]["threshold"], 0.3)
        self.assertEqual(selected["inner_equal_cycle"]["threshold"], 0.7)
        self.assertEqual(selected["inner_shrink50"]["threshold"], 0.4)

    def test_threshold_selection_depends_only_on_supplied_inner_vectors(self):
        y_inner = np.tile([0, 1], 4)
        p_inner = np.tile([0.2, 0.8], 4)
        cycles_inner = np.repeat([1, 2, 3, 4], 2)
        first = select_policies(y_inner, p_inner, cycles_inner, threshold_config())
        second = select_policies(y_inner, p_inner, cycles_inner, threshold_config())
        self.assertEqual(first, second)
        frame = evaluation_frame()
        groups = prediction_groups(frame)
        baseline = evaluate_predictions(frame, groups, complete_protocol())
        altered_outer = frame.copy()
        outer_mask = altered_outer.cycle_code.between(56, 79)
        altered_outer.loc[outer_mask, "artifact_signal"] = 1 - altered_outer.loc[outer_mask, "artifact_signal"]
        changed = evaluate_predictions(altered_outer, groups, complete_protocol())
        baseline_w1 = next(job for job in baseline["jobs"] if job["family"] == "primary" and job["window"] == "W1")
        changed_w1 = next(job for job in changed["jobs"] if job["family"] == "primary" and job["window"] == "W1")
        self.assertEqual(baseline_w1["thresholds"], changed_w1["thresholds"])
        self.assertNotEqual(
            baseline_w1["policies"]["inner_pooled"]["metrics"]["macro_f1"],
            changed_w1["policies"]["inner_pooled"]["metrics"]["macro_f1"],
        )

    def test_evaluation_pools_rows_and_reuses_shared_fit_predictions(self):
        frame = evaluation_frame()
        result = evaluate_predictions(frame, prediction_groups(frame), complete_protocol())
        self.assertEqual(result["total_evaluation_jobs"], 8)
        self.assertEqual(result["aggregate"]["primary"]["fixed_0.5"]["C1_A"]["metrics"]["n"], 48)
        self.assertEqual(result["aggregate"]["secondary"]["fixed_0.5"]["C1_A"]["metrics"]["n"], 48)
        primary = result["aggregate"]["primary"]["fixed_0.5"]["C1_A"]
        per_cycle_mean = float(np.mean([
            cycle["macro_f1"] for job in result["jobs"] if job["family"] == "primary"
            for cycle in job["policies"]["fixed_0.5"]["per_cycle"]
        ]))
        self.assertNotAlmostEqual(primary["metrics"]["macro_f1"], per_cycle_mean)
        self.assertAlmostEqual(primary["equal_cycle_macro_f1"], per_cycle_mean)

        jobs = {(job["family"], job["window"]): job for job in result["jobs"]}
        w1 = jobs[("primary", "W1")]
        h12_e = jobs[("secondary", "H12_E")]
        np.testing.assert_array_equal(
            np.asarray(w1["thresholds"]["fixed_0.5"]["threshold"]),
            np.asarray(h12_e["thresholds"]["fixed_0.5"]["threshold"]),
        )
        source_rows = np.flatnonzero(frame.cycle_code.between(56, 59).to_numpy())
        expected_hash = array_digest(prediction_groups(frame)[55]["C1_A"]["probability"][:len(source_rows)])
        self.assertEqual(w1["probability_sha256"], expected_hash)
        self.assertIn("EF0_stream_missing", w1["policies"]["fixed_0.5"]["subgroups"])
        self.assertIn("all_codes_known_nonmissing", w1["policies"]["fixed_0.5"]["subgroups"])
        self.assertIn("any_unseen_code", w1["policies"]["fixed_0.5"]["subgroups"])

    def test_missing_fit_group_produces_partial_evaluation(self):
        frame = evaluation_frame()
        groups = prediction_groups(frame)
        del groups[59]
        result = evaluate_predictions(frame, groups, complete_protocol())
        self.assertEqual(result["total_evaluation_jobs"], 7)
        self.assertNotIn(("primary", "W2"), {(job["family"], job["window"]) for job in result["jobs"]})

    def test_invalid_vectors_are_rejected(self):
        with self.assertRaises(ValueError):
            select_policies([0, 1], [0.3], [1, 2], threshold_config())
        with self.assertRaises(ValueError):
            select_policies([0, 1], [0.3, np.nan], [1, 2], threshold_config())
        with self.assertRaises(ValueError):
            select_policies([0, 2], [0.3, 0.8], [1, 2], threshold_config())

    def test_manual_macro_f1_matches_sklearn_and_round1_for_asymmetric_and_single_class(self):
        cases = (
            (np.array([0, 0, 0, 1, 1]), np.array([0, 1, 1, 1, 1])),
            (np.array([0, 0]), np.array([0, 1])),
            (np.array([1, 1]), np.array([0, 1])),
        )
        for actual, prediction in cases:
            with self.subTest(actual=actual.tolist(), prediction=prediction.tolist()):
                p = prediction.astype(float)
                expected = f1_score(actual, prediction, labels=[0, 1], average="macro", zero_division=0)
                self.assertAlmostEqual(_macro_f1_from_counts(actual, prediction.astype(bool)), expected)
                self.assertAlmostEqual(binary_metrics(actual, p, 0.5)["macro_f1"], expected)


if __name__ == "__main__":
    unittest.main()
