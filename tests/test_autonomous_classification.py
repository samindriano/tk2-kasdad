from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.models.autonomous_classification import (
    THRESHOLD,
    build_autonomous_classifier,
)
from src.models.classification_round2_models import stream_missing_mask
from src.models.round1_models import EXPANDED_FEATURES


def _synthetic_classification_frame(rows: int = 600, seed: int = 17) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    target = rng.integers(0, 2, rows, dtype=np.int8)
    missing = np.zeros(rows, dtype=bool)
    missing[::3] = True
    # Ensure both labels occur with healthy support in each availability subset.
    target[np.flatnonzero(missing)[::2]] = 0
    target[np.flatnonzero(missing)[1::2]] = 1
    target[np.flatnonzero(~missing)[::2]] = 0
    target[np.flatnonzero(~missing)[1::2]] = 1
    frame = pd.DataFrame({
        "record_id": [f"synthetic-{i:04d}" for i in range(rows)],
        "cycle_code": rng.integers(47, 80, rows),
        "org_code": rng.choice(["ORG_A", "ORG_B", "ORG_C"], rows),
        "module_code": rng.choice(["MOD_A", "MOD_B", "MOD_C"], rows),
        "pattern_code": rng.choice(["PAT_A", "PAT_B", "PAT_C"], rows),
        "signal_stream": ["" if is_missing else f"common token group{int(label)} repeated text"
                          for is_missing, label in zip(missing, target)],
        "scope_trace": rng.integers(0, 12, rows),
        "org_spread": rng.integers(1, 6, rows),
        "module_spread": rng.integers(1, 8, rows),
        "text_load": rng.uniform(15, 90, rows),
        "scope_pressure": rng.uniform(0, 2, rows),
        "pulse_a": rng.integers(0, 30, rows),
        "pulse_b": rng.integers(0, 20, rows),
        "telemetry_flag": rng.integers(0, 2, rows),
        "audit_lane": rng.choice(["A", "B", "C"], rows),
        "checksum_hint": rng.integers(0, 4, rows),
        "impact_load": rng.uniform(20, 180, rows),
        "entry_friction": rng.integers(0, 2, rows),
    })
    if tuple(name for name in EXPANDED_FEATURES if name in frame.columns) != EXPANDED_FEATURES:
        raise AssertionError("Synthetic fixture is missing a frozen expanded feature")
    return frame, target


class AutonomousClassificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame, cls.target = _synthetic_classification_frame()
        cls.missing = stream_missing_mask(cls.frame)

    def test_all_three_fixed_routes_fit_and_reproduce_routed_expert_probabilities(self):
        for candidate_id in ("present_specialist", "missing_specialist", "both_specialist"):
            with self.subTest(candidate_id=candidate_id):
                model = build_autonomous_classifier(candidate_id)
                model.fit(self.frame, self.target)
                before = model.audit_state()
                probability = model.predict_proba(self.frame)
                after = model.audit_state()
                expected = np.empty(len(self.frame), dtype=float)
                for branch, mask in (("missing", self.missing), ("present", ~self.missing)):
                    expert = model._experts[branch]
                    expected[mask] = expert.predict_proba(
                        self.frame.loc[mask, expert.input_feature_names]
                    )
                np.testing.assert_array_equal(probability, expected)
                self.assertEqual(before, after)
                self.assertEqual(model.predict_proba(self.frame).shape, (len(self.frame),))
                self.assertTrue(np.isfinite(probability).all())
                self.assertTrue(((probability >= 0) & (probability <= 1)).all())
                np.testing.assert_array_equal(model.predict(self.frame), (probability >= THRESHOLD).astype(np.int8))
                state = model.audit_state()
                expected_present = len(self.frame) - int(self.missing.sum()) if candidate_id in {
                    "present_specialist", "both_specialist"} else len(self.frame)
                expected_missing = int(self.missing.sum()) if candidate_id in {
                    "missing_specialist", "both_specialist"} else len(self.frame)
                self.assertEqual(state["branches"]["present"]["fit_rows"], expected_present)
                self.assertEqual(state["branches"]["missing"]["fit_rows"], expected_missing)
                self.assertNotIn("artifact_signal", state["branches"]["present"]["expert_state"]["input_feature_names"])
                self.assertNotIn("record_id", state["branches"]["missing"]["expert_state"]["input_feature_names"])

    def test_target_columns_are_rejected_from_features(self):
        model = build_autonomous_classifier("both_specialist")
        contaminated = self.frame.assign(artifact_signal=self.target)
        with self.assertRaisesRegex(ValueError, "Target columns are forbidden"):
            model.fit(contaminated, self.target)

    def test_missing_subset_support_fails_closed(self):
        present_only = self.frame.loc[~self.missing].reset_index(drop=True)
        target = self.target[~self.missing]
        model = build_autonomous_classifier("missing_specialist")
        with self.assertRaisesRegex(ValueError, "Insufficient missing fit support"):
            model.fit(present_only, target)

    def test_unknown_candidate_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown availability candidate"):
            build_autonomous_classifier("C4_HGB")


if __name__ == "__main__":
    unittest.main()
