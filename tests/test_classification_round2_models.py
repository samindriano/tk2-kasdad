"""Synthetic, data-independent tests for the fixed Round 2 adapters."""
import json
import unittest

import numpy as np
import pandas as pd
from scipy.special import expit

from src.models.classification_models import _json_digest, build_classifier
from src.models.classification_round2_models import (
    build_round2_model,
    coefficient_summary,
    linear_components,
    stream_missing_mask,
)


def fixture_rows(rows=180):
    index = np.arange(rows)
    signal = [f"evidence marker group{value % 7}" for value in index]
    signal[3] = None
    signal[7] = ""
    signal[11] = "   "
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
        "audit_lane": "unused marker",
        "checksum_hint": "unused checksum",
        "pulse_a": index,
        "pulse_b": index,
        "threat_rank": index,
        "artifact_signal": index % 2,
    })
    y = ((index % 7) < 3).astype(int)
    return frame, y


class ClassificationRound2ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame, cls.y = fixture_rows()
        cls.train = cls.frame.iloc[:150].copy()
        cls.y_train = cls.y[:150]
        cls.held = cls.frame.iloc[150:].copy()

    def fit_variant(self, model_id, variant):
        return build_round2_model({"id": model_id, "kind": "variant", "variant": variant}).fit(
            self.train, self.y_train
        )

    def test_control_adapters_reproduce_round1_probabilities(self):
        for recipe in ("C1_A", "C1_B", "C2_A", "C2_B"):
            with self.subTest(recipe=recipe):
                original = build_classifier({"id": recipe}).fit(self.train, self.y_train)
                control = build_round2_model({"id": recipe, "kind": "control", "source": recipe})
                self.assertEqual(type(control), type(original))
                control.fit(self.train, self.y_train)
                np.testing.assert_array_equal(
                    control.predict_proba(self.held), original.predict_proba(self.held)
                )
                self.assertEqual(control.audit_state(), original.audit_state())

    def test_stream_missing_mask_matches_parent_blank_and_missing_rules(self):
        series = pd.Series([None, np.nan, "", "  \t", " text "])
        np.testing.assert_array_equal(stream_missing_mask(series), [True, True, True, True, False])
        np.testing.assert_array_equal(
            stream_missing_mask(pd.DataFrame({"signal_stream": series})),
            [True, True, True, True, False],
        )

    def test_c3b_indicator_adds_only_one_feature_and_binds_stream_to_fit_hash(self):
        structured = build_classifier({"id": "C1_A"}).fit(self.train, self.y_train)
        model = self.fit_variant("C3_B", "structured_indicator")
        state = model.audit_state()
        dimensions = state["round2_variant"]["block_dimensions"]
        self.assertEqual(dimensions["structured"] + dimensions["missing_indicator"], dimensions["total"])
        self.assertEqual(dimensions["text"], 0)
        self.assertEqual(state["round2_variant"]["stream_missing_indicator"], True)
        self.assertEqual(
            state["round2_variant"]["structured_transform_sha256"],
            coefficient_summary(structured)["structured_transform_sha256"],
        )
        self.assertNotEqual(model._fit_fingerprint, structured._fit_fingerprint)
        changed_stream = self.train.copy()
        changed_stream.loc[changed_stream.index[0], "signal_stream"] = "different fit evidence"
        refit = build_round2_model({"id": "C3_B", "kind": "variant", "variant": "structured_indicator"})
        refit.fit(changed_stream, self.y_train)
        self.assertNotEqual(state["fit_data_sha256"], refit.audit_state()["fit_data_sha256"])
        self.assertEqual(state["round2_variant"]["structured_transform_sha256"],
                         refit.audit_state()["round2_variant"]["structured_transform_sha256"])

    def test_hybrid_without_indicator_and_weighted_text_dimensions_are_exact(self):
        c2 = build_classifier({"id": "C2_B"}).fit(self.train, self.y_train)
        c3c = self.fit_variant("C3_C", "hybrid_no_indicator")
        c5low = self.fit_variant("C5_LOW", "hybrid_text_half")
        c5high = self.fit_variant("C5_HIGH", "hybrid_text_double")
        states = [model.audit_state() for model in (c3c, c5low, c5high)]
        self.assertFalse(states[0]["round2_variant"]["stream_missing_indicator"])
        self.assertEqual(states[0]["round2_variant"]["block_dimensions"]["total"],
                         states[0]["round2_variant"]["block_dimensions"]["structured"] +
                         states[0]["round2_variant"]["block_dimensions"]["text"])
        self.assertEqual(states[1]["round2_variant"]["block_dimensions"],
                         states[2]["round2_variant"]["block_dimensions"])
        self.assertEqual(states[1]["round2_variant"]["text_block_weight"], 0.5)
        self.assertEqual(states[2]["round2_variant"]["text_block_weight"], 2.0)
        for model, original_state in ((c3c, c2.audit_state()), (c5low, c2.audit_state()),
                                      (c5high, c2.audit_state())):
            self.assertEqual(model.audit_state()["round2_variant"]["structured_transform_sha256"],
                             coefficient_summary(c2)["structured_transform_sha256"])
            self.assertEqual(model.audit_state()["round2_variant"]["text_transform_sha256"],
                             _json_digest(model.audit_state()["text"]))
        self.assertEqual(states[0]["text"]["vocabulary_sha256"], c2.audit_state()["text"]["vocabulary_sha256"])

        low_summary = coefficient_summary(c5low)
        high_summary = coefficient_summary(c5high)
        self.assertAlmostEqual(
            low_summary["text_effective_original_basis_l2_norm"],
            0.5 * low_summary["text_coefficient_l2_norm"],
        )
        self.assertAlmostEqual(
            high_summary["text_effective_original_basis_l2_norm"],
            2.0 * high_summary["text_coefficient_l2_norm"],
        )

    def test_logits_decompose_into_fixed_named_blocks_and_probability(self):
        variants = (
            self.fit_variant("C3_B", "structured_indicator"),
            self.fit_variant("C3_C", "hybrid_no_indicator"),
            self.fit_variant("C5_LOW", "hybrid_text_half"),
            self.fit_variant("C5_HIGH", "hybrid_text_double"),
        )
        for model in variants:
            with self.subTest(model=model.model_id):
                audit_before = model.audit_state()
                diagnostics_before = model.prediction_diagnostics()
                parts = linear_components(model, self.held)
                self.assertEqual(set(parts), {
                    "intercept", "numeric", "categorical", "text", "missing_indicator"
                })
                logits = sum(parts.values())
                np.testing.assert_allclose(expit(logits), model.predict_proba(self.held), rtol=1e-12, atol=1e-12)
                self.assertEqual(audit_before, model.audit_state())
                # linear_components is read-only, while predictions update only
                # the explicitly separate diagnostics channel.
                self.assertEqual(diagnostics_before, {"n_predictions": 0})
                summary = coefficient_summary(model)
                encoded = json.dumps(summary, sort_keys=True)
                self.assertNotIn("evidence marker", encoded)
                self.assertNotIn("org0", encoded)
                self.assertNotIn("pattern0", encoded)

    def test_all_empty_stream_fallback_and_fit_only_state(self):
        train = self.train.copy()
        train["signal_stream"] = [None if row % 2 else "   " for row in range(len(train))]
        held = self.held.copy()
        held["signal_stream"] = ["", None] * (len(held) // 2)
        for recipe_id, variant in (("C3_C", "hybrid_no_indicator"),
                                   ("C5_LOW", "hybrid_text_half"),
                                   ("C5_HIGH", "hybrid_text_double")):
            with self.subTest(recipe=recipe_id):
                model = build_round2_model({"id": recipe_id, "kind": "variant", "variant": variant})
                model.fit(train, self.y_train)
                before = model.audit_state()
                dimensions = before["round2_variant"]["block_dimensions"]
                self.assertEqual(dimensions["text"], 0)
                self.assertTrue(before["text"]["empty_all_training_text"])
                probability = model.predict_proba(held)
                self.assertEqual(probability.shape, (len(held),))
                self.assertTrue(np.isfinite(probability).all())
                self.assertEqual(before, model.audit_state())
                parts = linear_components(model, held)
                np.testing.assert_allclose(expit(sum(parts.values())), probability, rtol=1e-12, atol=1e-12)

    def test_builder_rejects_recipe_mismatch_and_hgb_has_no_logit_components(self):
        with self.assertRaises(ValueError):
            build_round2_model({"id": "C1_A", "kind": "control", "source": "C2_B"})
        with self.assertRaises(ValueError):
            build_round2_model({"id": "C3_B", "kind": "variant", "variant": "wrong"})
        hgb = build_classifier({"id": "C1_B"}).fit(self.train, self.y_train)
        self.assertEqual(linear_components(hgb, self.held), {})
        self.assertEqual(coefficient_summary(hgb)["linear"], False)


if __name__ == "__main__":
    unittest.main()
