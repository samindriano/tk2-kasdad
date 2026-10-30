import unittest

import numpy as np
import pandas as pd

from src.models.round1_models import EXPANDED_FEATURES, build_model as build_round1_model
from src.models.regression_round2_models import build_model


def fixture(n=180):
    i = np.arange(n)
    frame = pd.DataFrame({
        "impact_load": (i % 9 + 1).astype(float),
        "entry_friction": (i % 2).astype(float),
        "cycle_code": 47 + i // 20,
        "scope_pressure": (i % 7).astype(float),
        "text_load": (i % 5).astype(float),
        "scope_trace": ["~ 3 nodes" if x % 3 == 0 else f"{x % 8} nodes" for x in i],
        "org_spread": [str(x % 6) for x in i],
        "module_spread": ["2" if x % 2 else "~ 4" for x in i],
        "telemetry_flag": ["active" if x % 2 else "inactive" for x in i],
        "org_code": [f"org-{x % 4}" for x in i],
        "module_code": [f"mod-{x % 5}" for x in i],
        "pattern_code": [f"pat-{x % 3}" for x in i],
        "record_id": [f"rec-{x}" for x in i],
        "signal_stream": [f"text-{x}" for x in i],
        "threat_rank": i.astype(float),
        "pulse_a": i.astype(float),
    })
    frame.loc[3, "scope_trace"] = "bad count"
    frame.loc[4, "org_spread"] = None
    frame.loc[5, "telemetry_flag"] = "unclear"
    frame.loc[6, "module_code"] = None
    return frame


class Round2FeatureModelTests(unittest.TestCase):
    def setUp(self):
        self.frame = fixture()
        self.fit = self.frame.iloc[:150].copy()
        self.val = self.frame.iloc[150:].copy()
        self.y = self.fit["threat_rank"].to_numpy()

    def test_full_selected_expanded_matches_frozen_round1_d(self):
        r2 = build_model(EXPANDED_FEATURES).fit(self.fit, self.y)
        r1 = build_round1_model({"id": "R1_D", "kind": "hgb", "features": "expanded",
                                 "params": {"loss": "absolute_error"}}).fit(self.fit, self.y)
        p2 = r2.predict(self.val)
        p1 = r1.predict(self.val)
        np.testing.assert_array_equal(p2, p1)
        state2, state1 = r2.audit_state(), r1.audit_state()
        self.assertEqual(state2["input_feature_names"], state1["input_feature_names"])
        self.assertEqual(state2["transformed_feature_names"], state1["transformed_feature_names"])
        self.assertEqual(state2["numeric_imputer_medians"], state1["numeric_imputer_medians"])
        self.assertEqual(state2["category_counts"], state1["category_counts"])
        self.assertEqual(state2["actual_hyperparameters"], state1["actual_hyperparameters"])
        self.assertEqual(state2["fit_data_sha256"], state1["fit_data_sha256"])

    def test_ablation_drops_source_and_all_derived_columns(self):
        selected = [name for name in EXPANDED_FEATURES if name != "scope_trace"]
        model = build_model(selected).fit(self.fit, self.y)
        state = model.audit_state()
        self.assertNotIn("scope_trace", state["input_feature_names"])
        self.assertFalse(any(name.startswith("scope_trace__") for name in state["transformed_feature_names"]))
        self.assertEqual(model.feature_names, tuple(name for name in EXPANDED_FEATURES if name in selected))
        self.assertIn("text_load__missing", state["transformed_feature_names"])

    def test_fit_only_categories_missing_and_invalid_parser_values(self):
        val = self.val.copy()
        val.loc[val.index[0], "org_code"] = "unseen-new-org"
        val.loc[val.index[1], "scope_trace"] = "5 widgets"
        val.loc[val.index[2], "scope_trace"] = None
        model = build_model(EXPANDED_FEATURES).fit(self.fit, self.y)
        before = model.audit_state()
        predictions = model.predict(val)
        after = model.audit_state()
        self.assertEqual(predictions.shape, (len(val),))
        self.assertTrue(np.isfinite(predictions).all())
        self.assertGreaterEqual(model.prediction_diagnostics()["unknown_category_rows"]["org_code"], 1)
        self.assertEqual(before["category_counts"], after["category_counts"])
        self.assertEqual(before["numeric_imputer_medians"], after["numeric_imputer_medians"])
        self.assertEqual(before["fit_data_sha256"], after["fit_data_sha256"])
        self.assertEqual(before["fitted_tree_sha256"], after["fitted_tree_sha256"])
        self.assertEqual(before["tree_complexity"], after["tree_complexity"])
        self.assertFalse(any(name in after["input_feature_names"] for name in
                             ("record_id", "signal_stream", "threat_rank", "pulse_a")))

    def test_allowlist_rejects_forbidden_and_bad_control_feature_sets(self):
        for forbidden in ("record_id", "signal_stream", "threat_rank", "pulse_a", "audit_lane"):
            with self.subTest(feature=forbidden), self.assertRaises(ValueError):
                build_model([*EXPANDED_FEATURES, forbidden])
        with self.assertRaises(ValueError):
            build_model(["impact_load", "impact_load", "entry_friction"])
        with self.assertRaises(ValueError):
            build_model(["impact_load", "entry_friction"], control="R1_D")
        with self.assertRaises(ValueError):
            build_model(["impact_load", "entry_friction"])
        with self.assertRaises(ValueError):
            build_model(control="R1_D", features=[*EXPANDED_FEATURES, "record_id"])

    def test_exact_controls_delegate_to_round1_and_preserve_audit_identity(self):
        for control, preset in (("R1_C", "two"), ("R1_D", "expanded")):
            with self.subTest(control=control):
                r2 = build_model(control=control).fit(self.fit, self.y)
                r1 = build_round1_model({"id": control, "kind": "hgb", "features": preset,
                                         "params": {"loss": "absolute_error"}}).fit(
                                             self.fit.loc[:, list(EXPANDED_FEATURES if preset == "expanded" else
                                                                  ("impact_load", "entry_friction"))], self.y)
                np.testing.assert_array_equal(r2.predict(self.val), r1.predict(self.val))
                self.assertEqual(r2.audit_state(), r1.audit_state())


if __name__ == "__main__":
    unittest.main()
