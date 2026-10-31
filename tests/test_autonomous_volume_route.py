from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.models.autonomous_volume_route import (
    THRESHOLD,
    VOLUME_QUANTILE,
    build_model,
    build_volume_route_classifier,
)
from src.models.classification_round2_models import stream_missing_mask
from src.models.round1_models import EXPANDED_FEATURES


def _synthetic_data(rows: int = 750, seed: int = 31) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, rows, dtype=np.int8)
    missing = np.zeros(rows, dtype=bool)
    missing[::4] = True
    y[np.flatnonzero(missing)[::2]] = 0
    y[np.flatnonzero(missing)[1::2]] = 1
    y[np.flatnonzero(~missing)[::2]] = 0
    y[np.flatnonzero(~missing)[1::2]] = 1
    X = pd.DataFrame({
        "record_id": [f"vol-synthetic-{i:04d}" for i in range(rows)],
        "cycle_code": rng.integers(47, 80, rows),
        "org_code": rng.choice(["ORG_A", "ORG_B", "ORG_C"], rows),
        "module_code": rng.choice(["MOD_A", "MOD_B", "MOD_C"], rows),
        "pattern_code": rng.choice(["PAT_A", "PAT_B", "PAT_C"], rows),
        "signal_stream": ["" if is_missing else f"common routine item class{int(label)} repeated"
                          for is_missing, label in zip(missing, y)],
        "scope_trace": rng.integers(0, 12, rows),
        "org_spread": rng.integers(1, 6, rows),
        "module_spread": rng.integers(1, 8, rows),
        "text_load": rng.uniform(10, 120, rows),
        "scope_pressure": rng.uniform(0, 2, rows),
        "pulse_a": rng.integers(0, 40, rows),
        "pulse_b": rng.integers(0, 20, rows),
        "telemetry_flag": rng.integers(0, 2, rows),
        "audit_lane": rng.choice(["A", "B", "C"], rows),
        "checksum_hint": rng.integers(0, 4, rows),
        "impact_load": rng.uniform(20, 180, rows),
        "entry_friction": rng.integers(0, 2, rows),
    })
    assert all(column in X.columns for column in EXPANDED_FEATURES)
    return X, y


class AutonomousVolumeRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = _synthetic_data()
        cls.missing = stream_missing_mask(cls.X)
        cls.model = build_volume_route_classifier().fit(cls.X, cls.y)

    def test_fit_only_q80_audit_and_expert_fit_counts(self):
        state = self.model.audit_state()
        expected_q80 = float(np.quantile(self.X.text_load.to_numpy(dtype=float), VOLUME_QUANTILE))
        self.assertEqual(state["volume_threshold"]["value"], expected_q80)
        self.assertEqual(state["volume_threshold"]["fit_rows"], len(self.X))
        self.assertTrue(state["volume_threshold"]["definition"].startswith("np.quantile over rows supplied to fit"))
        self.assertEqual(state["experts"]["C2_B_all"]["fit_rows"], len(self.X))
        self.assertEqual(state["experts"]["C2_B_present"]["fit_rows"], int((~self.missing).sum()))
        self.assertEqual(state["experts"]["C1_B_all"]["fit_rows"], len(self.X))

    def test_prediction_uses_fit_q80_and_exact_three_way_routing(self):
        before = self.model.audit_state()
        q80 = before["volume_threshold"]["value"]
        volume = self.X.text_load.to_numpy(dtype=float)
        high = (~self.missing) & (volume > q80)
        low = (~self.missing) & ~high
        expected = np.empty(len(self.X), dtype=float)
        for name, mask in (("C1_B_all", self.missing), ("C2_B_present", high), ("C2_B_all", low)):
            expert = self.model._experts[name]
            expected[mask] = expert.predict_proba(self.X.loc[mask, expert.input_feature_names])
        probability = self.model.predict_proba(self.X)
        np.testing.assert_array_equal(probability, expected)
        self.assertEqual(before, self.model.audit_state())
        np.testing.assert_array_equal(self.model.predict(self.X), (probability >= THRESHOLD).astype(np.int8))

        # Extreme prediction inputs do not refit or update the fit-only q80.
        future = self.X.copy()
        future["text_load"] = 1_000_000.0
        self.model.predict_proba(future)
        self.assertEqual(before, self.model.audit_state())

    def test_value_equal_to_q80_stays_on_all_row_present_expert(self):
        q80 = self.model.audit_state()["volume_threshold"]["value"]
        equal_frame = self.X.copy()
        present_row = int(np.flatnonzero(~self.missing)[0])
        equal_frame.loc[present_row, "text_load"] = q80
        missing = stream_missing_mask(equal_frame)
        expected = np.empty(len(equal_frame), dtype=float)
        masks = {
            "C1_B_all": missing,
            "C2_B_present": (~missing) & (equal_frame.text_load.to_numpy(dtype=float) > q80),
        }
        masks["C2_B_all"] = (~missing) & ~masks["C2_B_present"]
        for name, mask in masks.items():
            expert = self.model._experts[name]
            expected[mask] = expert.predict_proba(equal_frame.loc[mask, expert.input_feature_names])
        np.testing.assert_array_equal(self.model.predict_proba(equal_frame), expected)

    def test_target_input_rejected_and_build_alias_available(self):
        contaminated = self.X.assign(artifact_signal=self.y)
        with self.assertRaisesRegex(ValueError, "Target columns are forbidden"):
            build_model().fit(contaminated, self.y)
        self.assertIs(type(build_model()), type(build_volume_route_classifier()))


if __name__ == "__main__":
    unittest.main()
