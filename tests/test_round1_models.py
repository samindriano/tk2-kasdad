"""Software checks for the TP2 Round 1 model wrappers.

Model smoke checks use only a small prefix slice of the supplied official
training CSV. The parser edge-case test uses synthetic tokens strictly as a
unit-test fixture; none of these tests reports model quality or EDA results.
"""

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.round1_models import (
    EXPANDED_FEATURES,
    _mixed_count_arrays,
    _telemetry_arrays,
    build_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = PROJECT_ROOT / "data" / "raw" / "regression" / "Train.csv"


class Round1ParserTests(unittest.TestCase):
    def test_count_parser_keeps_approx_suffix_invalid_and_missing_flags(self):
        values, approximate, suffix, parse_fail, missing = _mixed_count_arrays(
            pd.Series(["12", "~ 3.0 nodes", "2 widgets", "bad", "", None])
        )
        np.testing.assert_allclose(values[:2], [12.0, 3.0])
        self.assertEqual(approximate.tolist(), [0, 1, 0, 0, 0, 0])
        self.assertEqual(suffix.tolist(), [0, 1, 1, 0, 0, 0])
        self.assertEqual(parse_fail.tolist(), [0, 0, 1, 1, 0, 0])
        self.assertEqual(missing.tolist(), [0, 0, 0, 0, 1, 1])
        self.assertTrue(np.isnan(values[2:]).all())

    def test_telemetry_mapping_is_casefolded_and_unknowns_are_explicit(self):
        values, unknown, missing = _telemetry_arrays(
            pd.Series(["ACTIVE", "False", "y", "0", "maybe", None])
        )
        np.testing.assert_allclose(values[:4], [1, 0, 1, 0])
        self.assertTrue(np.isnan(values[4:]).all())
        self.assertEqual(unknown.tolist(), [0, 0, 0, 0, 1, 0])
        self.assertEqual(missing.tolist(), [0, 0, 0, 0, 0, 1])


class Round1ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not TRAIN_CSV.exists():
            raise unittest.SkipTest("Official local regression Train.csv required; no synthetic model data substituted.")
        cls.train = pd.read_csv(TRAIN_CSV, nrows=280)
        cls.X = cls.train.drop(columns="threat_rank")
        cls.y = cls.train["threat_rank"]

    def test_lookup_baselines_and_linear_fit_only_selected_features(self):
        train_X, validation_X = self.X.iloc[:220], self.X.iloc[220:]
        train_y = self.y.iloc[:220]
        for kind in ("global_median", "impact_median", "joint_median", "linear"):
            with self.subTest(kind=kind):
                model = build_model(
                    {"id": kind, "kind": kind, "features": "two", "min_support": 30, "params": {}}
                ).fit(train_X, train_y)
                predictions = model.predict(validation_X)
                self.assertEqual(predictions.shape, (len(validation_X),))
                self.assertTrue(np.isfinite(predictions).all())
                state = model.audit_state()
                expected_features = {
                    "global_median": [],
                    "impact_median": ["impact_load"],
                    "joint_median": ["impact_load", "entry_friction"],
                    "linear": ["impact_load", "entry_friction"],
                }[kind]
                self.assertEqual(state["input_feature_names"], expected_features)
                self.assertNotIn("threat_rank", state["input_feature_names"])
                self.assertNotIn("pulse_a", state["input_feature_names"])
                if kind in {"global_median", "impact_median", "joint_median"}:
                    diagnostics = model.prediction_diagnostics()
                    self.assertEqual(sum(diagnostics["fallback_counts"].values()), len(validation_X))
                json.dumps(state)

    def test_expanded_hgb_uses_fold_fitted_preprocessing_and_native_categories(self):
        train_X, validation_X = self.X.iloc[:220], self.X.iloc[220:].copy()
        train_y = self.y.iloc[:220]
        recipe = {
            "id": "expanded_test_fixture",
            "kind": "hgb",
            "features": "expanded",
            "params": {
                "loss": "absolute_error",
                "learning_rate": 0.05,
                "max_iter": 200,
                "max_leaf_nodes": 15,
                "min_samples_leaf": 30,
                "l2_regularization": 1.0,
                "max_bins": 255,
                "early_stopping": False,
                "random_state": 42,
            },
        }
        model = build_model(recipe).fit(train_X, train_y)
        state_before = model.audit_state()
        self.assertEqual(state_before["input_feature_names"], list(EXPANDED_FEATURES))
        self.assertNotIn("record_id", state_before["input_feature_names"])
        self.assertNotIn("signal_stream", state_before["input_feature_names"])
        self.assertNotIn("pulse_a", state_before["input_feature_names"])
        self.assertNotIn("threat_rank", state_before["input_feature_names"])
        self.assertIn("scope_trace__suffix_present", state_before["transformed_feature_names"])
        self.assertEqual(state_before["actual_hyperparameters"]["loss"], "absolute_error")
        self.assertEqual(state_before["actual_hyperparameters"]["early_stopping"], False)
        self.assertEqual(state_before["ordinal_encoder"]["min_frequency"], 30)
        self.assertEqual(state_before["ordinal_encoder"]["max_categories"], 254)
        self.assertIn("org_code", state_before["category_counts"])
        self.assertTrue(state_before["numeric_imputer_medians"])
        json.dumps(state_before)

        # Perturb only held-out inputs to exercise missing, malformed, and unseen
        # value handling; these rows never enter fit or the category encoder.
        validation_X.iloc[0, validation_X.columns.get_loc("org_code")] = "synthetic_unseen_org"
        validation_X.iloc[1, validation_X.columns.get_loc("scope_trace")] = "2 widgets"
        validation_X.iloc[2, validation_X.columns.get_loc("scope_trace")] = None
        predictions = model.predict(validation_X)
        self.assertEqual(predictions.shape, (len(validation_X),))
        self.assertTrue(np.isfinite(predictions).all())
        diagnostics = model.prediction_diagnostics()
        self.assertGreaterEqual(diagnostics["unknown_category_rows"]["org_code"], 1)
        self.assertEqual(state_before["numeric_imputer_medians"], model.audit_state()["numeric_imputer_medians"])

    def test_two_feature_hgb_accepts_protocol_interaction_diagnostic(self):
        model = build_model(
            {
                "id": "no_interactions_test_fixture",
                "kind": "hgb",
                "features": "two",
                "params": {
                    "loss": "squared_error",
                    "learning_rate": 0.05,
                    "max_iter": 200,
                    "max_leaf_nodes": 15,
                    "min_samples_leaf": 30,
                    "l2_regularization": 1.0,
                    "max_bins": 255,
                    "early_stopping": False,
                    "random_state": 42,
                    "interaction_cst": "no_interactions",
                },
            }
        ).fit(self.X.iloc[:220], self.y.iloc[:220])
        self.assertEqual(model.audit_state()["actual_hyperparameters"]["interaction_cst"], "no_interactions")
        self.assertEqual(model.predict(self.X.iloc[220:]).shape, (len(self.X) - 220,))


if __name__ == "__main__":
    unittest.main()
