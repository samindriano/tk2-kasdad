import unittest

import numpy as np
import pandas as pd

from src.models.autonomous_regression import build_autonomous_regression
from src.models.regression_research_text import HYBRID_FEATURES
from scripts.research_autonomous_regression import select_inner_policy


def _frame(n: int) -> pd.DataFrame:
    i = np.arange(n)
    docs = []
    for value in i:
        terms = [f"lexeme{(value * 7 + offset * 19) % 220}" for offset in range(6)]
        docs.append("shared telemetry " + " ".join(terms))
    return pd.DataFrame({
        "impact_load": (15 + i % 17).astype(float),
        "entry_friction": (i % 2).astype(float),
        "cycle_code": 47 + i // 30,
        "scope_pressure": (i % 9).astype(float),
        "text_load": (i % 8).astype(float),
        "scope_trace": [f"{x % 7} nodes" for x in i],
        "org_spread": [f"{x % 6} nodes" for x in i],
        "module_spread": [f"{x % 5} nodes" for x in i],
        "telemetry_flag": ["active" if x % 2 else "inactive" for x in i],
        "org_code": [f"org-{x % 4}" for x in i],
        "module_code": [f"mod-{x % 5}" for x in i],
        "pattern_code": [f"pat-{x % 3}" for x in i],
        "signal_stream": [None if x % 19 == 0 else docs[x] for x in i],
    })


class AutonomousRegressionTests(unittest.TestCase):
    def test_svd128_fit_is_train_only_deterministic_and_finite(self):
        frame = _frame(240)
        numeric = frame.impact_load.to_numpy() * .55 + frame.entry_friction.to_numpy() * 8
        lexical = frame.signal_stream.str.contains("lexeme12", na=False).to_numpy() * 11
        target = np.clip(19 + numeric + lexical, 0, 100)
        features = list(HYBRID_FEATURES)
        train = frame.iloc[:200].loc[:, features].copy()
        validation = frame.iloc[200:].loc[:, features].copy()
        validation.iloc[0, validation.columns.get_loc("signal_stream")] = "unseenword unknownphrase"
        validation.iloc[1, validation.columns.get_loc("org_code")] = "unseen-org"

        first = build_autonomous_regression("base").fit(train, target[:200])
        second = build_autonomous_regression("base").fit(train, target[:200])
        state = first.audit_state()
        before_prediction = first.audit_state()
        prediction = first.predict(validation)
        after_prediction = first.audit_state()

        self.assertEqual(state, before_prediction)
        self.assertEqual(before_prediction, after_prediction)
        self.assertEqual(state["svd"]["n_components_requested"], 128)
        self.assertGreater(state["svd"]["n_components_actual"], 32)
        self.assertEqual(state["svd"]["n_components_actual"], second.audit_state()["svd"]["n_components_actual"])
        self.assertEqual(state["svd"]["components_sha256"], second.audit_state()["svd"]["components_sha256"])
        self.assertEqual(state["fit_rows"], 200)
        self.assertFalse(state["validation_data_passed_to_fit"])
        self.assertFalse(state["test_data_used_for_fit"])
        self.assertEqual(prediction.shape, (40,))
        self.assertTrue(np.isfinite(prediction).all())
        self.assertGreaterEqual(first.prediction_diagnostics()["unseen_tfidf_token_count"], 1)

    def test_only_frozen_parameter_ids_and_exact_features_are_accepted(self):
        with self.assertRaises(ValueError):
            build_autonomous_regression("deeper")
        model = build_autonomous_regression("base")
        frame = _frame(12)
        with self.assertRaises(ValueError):
            model.fit(frame.loc[:, list(HYBRID_FEATURES)[:-1]], np.arange(12, dtype=float) + 20)

    def test_inner_ties_follow_frozen_weight_then_parameter_order(self):
        selected = select_inner_policy([
            {"param": "base", "rank_weight": 0.5, "mae": 10.0},
            {"param": "regularized", "rank_weight": 0.25, "mae": 10.0},
            {"param": "base", "rank_weight": 0.25, "mae": 10.0},
        ])
        self.assertEqual(selected["rank_weight"], 0.25)
        self.assertEqual(selected["param"], "regularized")


if __name__ == "__main__":
    unittest.main()
