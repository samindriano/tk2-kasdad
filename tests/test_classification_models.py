"""Small, synthetic unit checks for classification recipe wrappers only."""
import json
import unittest

import numpy as np
import pandas as pd

from src.models.classification_models import _RECIPES, build_classifier
from src.models.round1_models import EXPANDED_FEATURES


def fixture_rows(rows=180):
    index = np.arange(rows)
    signal = [f"token group{value % 7} evidence marker" for value in index]
    signal[3] = None
    signal[7] = ""
    frame = pd.DataFrame({
        "record_id": index,
        "impact_load": (index % 16).astype(float),
        "entry_friction": (index % 2).astype(int),
        "cycle_code": 47 + (index % 9),
        "scope_pressure": (index % 11).astype(float),
        "text_load": (index % 8).astype(float),
        "scope_trace": ["3 nodes" if i % 4 else "~2 nodes" for i in index],
        "org_spread": [str(i % 9) for i in index],
        "module_spread": ["4 nodes" for _ in index],
        "telemetry_flag": ["active" if i % 2 else "inactive" for i in index],
        "org_code": [f"org{i % 5}" for i in index],
        "module_code": [f"mod{i % 4}" for i in index],
        "pattern_code": [f"pattern{i % 3}" for i in index],
        "signal_stream": signal,
        "audit_lane": "do not use",
        "checksum_hint": "do not use",
        "pulse_a": index,
        "pulse_b": index,
        "threat_rank": index,
        "artifact_signal": index % 2,
    })
    y = ((index % 7) < 3).astype(int)
    return frame, y


class ClassificationModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame, cls.y = fixture_rows()

    def test_recipe_allowlists_and_fitted_probability_contract(self):
        expected = {
            "C0_A": ["entry_friction"],
            "C0_B": ["entry_friction"],
            "C0_C": ["impact_load", "entry_friction"],
            "C1_A": list(EXPANDED_FEATURES),
            "C1_B": list(EXPANDED_FEATURES),
            "C2_A": ["signal_stream"],
            "C2_B": [*EXPANDED_FEATURES, "signal_stream"],
        }
        for recipe_id, (_, _) in _RECIPES.items():
            with self.subTest(recipe=recipe_id):
                model = build_classifier({"id": recipe_id}).fit(self.frame.iloc[:150], self.y[:150])
                self.assertEqual(model.input_feature_names, expected[recipe_id])
                banned = {"record_id", "threat_rank", "artifact_signal", "audit_lane", "checksum_hint",
                          "pulse_a", "pulse_b"}
                self.assertFalse(banned.intersection(model.input_feature_names))
                probability = model.predict_proba(self.frame.iloc[150:])
                self.assertEqual(probability.shape, (30,))
                self.assertTrue(np.isfinite(probability).all())
                self.assertTrue(((probability >= 0) & (probability <= 1)).all())
                json.dumps(model.audit_state(), allow_nan=False)

    def test_labels_must_be_binary_aligned_and_learned_models_need_both_classes(self):
        for bad in ([0, 2], [0, np.nan], [0]):
            with self.subTest(labels=bad):
                model = build_classifier({"id": "C0_A"})
                with self.assertRaises(ValueError):
                    model.fit(self.frame.iloc[:2], bad)
        model = build_classifier({"id": "C0_C"})
        with self.assertRaisesRegex(ValueError, "both training classes"):
            model.fit(self.frame.iloc[:20], np.zeros(20))

    def test_lookup_fallback_uses_only_fit_rates(self):
        frame = self.frame.iloc[:90].copy()
        y = np.zeros(90, dtype=int)
        y[frame.entry_friction.to_numpy() == 1] = 1
        model = build_classifier({"id": "C0_B"}).fit(frame, y)
        held = pd.DataFrame({"entry_friction": [0, 1, 7, np.nan]})
        probabilities = model.predict_proba(held)
        np.testing.assert_allclose(probabilities, [0, 1, 0.5, 0.5])
        self.assertEqual(sum(model.prediction_diagnostics()["fallback_counts"].values()), 4)
        self.assertEqual(model.audit_state()["supported_ef_positive_rates"], {"0": 0.0, "1": 1.0})

    def test_structured_linear_and_hgb_preprocessing_is_fit_only(self):
        train = self.frame.iloc[:150].copy()
        y = self.y[:150]
        held = self.frame.iloc[150:].copy()
        held.loc[held.index[0], "org_code"] = "unseen_org"
        held.loc[held.index[1], "scope_trace"] = "unsupported unit"
        held.loc[held.index[2], "module_code"] = None
        for recipe in ("C1_A", "C1_B"):
            with self.subTest(recipe=recipe):
                model = build_classifier({"id": recipe}).fit(train, y)
                before = model.audit_state()
                first = model.predict_proba(held)
                diagnostics = model.prediction_diagnostics()
                after = model.audit_state()
                self.assertEqual(before, after)
                self.assertEqual(first.shape, (len(held),))
                self.assertTrue(np.isfinite(first).all())
                self.assertGreaterEqual(diagnostics["unknown_category_rows"]["org_code"], 1)
                self.assertEqual(before["fit_rows"], 150)
                if recipe == "C1_A":
                    self.assertIn("numeric_imputer_medians", before["preprocessor"])
                    self.assertTrue(before["preprocessor"]["numeric_imputer_medians"])
                if recipe == "C1_B":
                    self.assertEqual(before["parameters"]["loss"], "log_loss")
                    self.assertEqual(before["parameters"]["early_stopping"], False)
                    self.assertEqual(len(before["categorical_mask"]), 3)

    def test_tfidf_vocabulary_is_train_only_and_missing_stream_is_explicit(self):
        train = self.frame.iloc[:150].copy()
        y = self.y[:150]
        model = build_classifier({"id": "C2_A"}).fit(train, y)
        before = model.audit_state()
        held = self.frame.iloc[150:].copy()
        held.loc[held.index[0], "signal_stream"] = "raretokenoutsidevocablary"
        held.loc[held.index[1], "signal_stream"] = None
        probabilities = model.predict_proba(held)
        self.assertEqual(before, model.audit_state())
        self.assertGreater(model.prediction_diagnostics()["unseen_tfidf_token_count"], 0)
        self.assertEqual(model.prediction_diagnostics()["signal_stream_missing_rows"], 1)
        self.assertEqual(probabilities.shape, (30,))
        self.assertEqual(before["text"]["vocabulary_sha256"], model.audit_state()["text"]["vocabulary_sha256"])

    def test_hybrid_combines_structured_text_and_missing_flag(self):
        model = build_classifier({"id": "C2_B"}).fit(self.frame.iloc[:150], self.y[:150])
        state = model.audit_state()
        self.assertGreater(state["text"]["vocabulary_size"], 0)
        self.assertIn("signal_stream", state["input_feature_names"])
        probabilities = model.predict_proba(self.frame.iloc[150:])
        self.assertEqual(probabilities.shape, (30,))
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(state["model_state_sha256"])

    def test_all_empty_text_uses_deterministic_missing_only_fallback(self):
        frame = self.frame.iloc[:80].copy()
        frame["signal_stream"] = None
        for recipe in ("C2_A", "C2_B"):
            with self.subTest(recipe=recipe):
                model = build_classifier({"id": recipe}).fit(frame, self.y[:80])
                state = model.audit_state()
                self.assertTrue(state["text"]["empty_all_training_text"])
                probability = model.predict_proba(frame.iloc[80 - 10:80])
                self.assertEqual(probability.shape, (10,))
                self.assertTrue(np.isfinite(probability).all())

    def test_tiny_corpus_with_no_tfidf_terms_uses_bias_only_fallback(self):
        frame = self.frame.iloc[:2].copy()
        frame["signal_stream"] = ["raretokenalpha", "raretokenbeta"]
        model = build_classifier({"id": "C2_A"}).fit(frame, np.array([0, 1]))
        state = model.audit_state()
        self.assertTrue(state["text"]["empty_after_tfidf_filters"])
        probability = model.predict_proba(frame)
        self.assertEqual(probability.shape, (2,))
        self.assertTrue(np.isfinite(probability).all())

    def test_input_schema_unknown_recipe_and_unfitted_state_guard(self):
        with self.assertRaises(ValueError):
            build_classifier({"id": "C1_A", "kind": "hgb_structured", "params": {"C": 5}})
        with self.assertRaises(ValueError):
            build_classifier({"id": "not-a-recipe"})
        model = build_classifier({"id": "C0_A"})
        with self.assertRaises(RuntimeError):
            model.audit_state()
        with self.assertRaises(ValueError):
            model.fit(self.frame.iloc[:10].drop(columns="entry_friction"), self.y[:10])


if __name__ == "__main__":
    unittest.main()
