"""Synthetic unit tests for the fixed dual-campaign text candidates."""
import json
import unittest

import numpy as np
import pandas as pd
from scipy.special import expit

from src.models.campaign_classification import build_campaign_classifier
from src.models.round1_models import EXPANDED_FEATURES


def _fixture(rows: int = 180) -> tuple[pd.DataFrame, np.ndarray]:
    index = np.arange(rows)
    streams = [
        f"alpha evidence marker cluster{value % 8} behavior family{value % 5}"
        if value % 7 else f"beta telemetry signature group{value % 6} behavior"
        for value in index
    ]
    streams[5] = None
    streams[13] = ""
    frame = pd.DataFrame({
        "record_id": index,
        "impact_load": (index % 23).astype(float),
        "entry_friction": (index % 2).astype(float),
        "cycle_code": 47 + (index % 9),
        "scope_pressure": (index % 13).astype(float),
        "text_load": (index % 10).astype(float),
        "scope_trace": ["3 nodes" if i % 4 else "~2 nodes" for i in index],
        "org_spread": [str(i % 7) for i in index],
        "module_spread": ["4 nodes" if i % 3 else "2 nodes" for i in index],
        "telemetry_flag": ["active" if i % 2 else "inactive" for i in index],
        "org_code": [f"org{i % 6}" for i in index],
        "module_code": [f"mod{i % 5}" for i in index],
        "pattern_code": [f"pattern{i % 4}" for i in index],
        "signal_stream": streams,
        "artifact_signal": (index % 7 < 3).astype(int),
        "threat_rank": index,
    })
    y = ((index % 7) < 3).astype(np.int8)
    return frame, y


class CampaignClassificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame, cls.y = _fixture()
        cls.features = [*EXPANDED_FEATURES, "signal_stream"]

    def test_all_candidates_preserve_c2b_allowlist_and_probability_contract(self):
        for recipe_id in ("DC_SVC", "DC_NBSVM", "DC_WORD_CHAR"):
            with self.subTest(recipe=recipe_id):
                model = build_campaign_classifier(recipe_id).fit(
                    self.frame.iloc[:150][self.features].copy(), self.y[:150]
                )
                state_before = model.audit_state()
                probability = model.predict_proba(self.frame.iloc[150:][self.features].copy())
                self.assertEqual(model.input_feature_names, self.features)
                self.assertFalse({"record_id", "artifact_signal", "threat_rank"}.intersection(model.input_feature_names))
                self.assertEqual(probability.shape, (30,))
                self.assertTrue(np.isfinite(probability).all())
                self.assertTrue(((probability >= 0) & (probability <= 1)).all())
                self.assertEqual(state_before, model.audit_state())
                self.assertEqual(state_before["campaign"]["base_recipe"], "C2_B")
                self.assertTrue(state_before["campaign"]["missing_stream_indicator_retained"])
                self.assertEqual(state_before["threshold"], 0.50) if "threshold" in state_before else None
                json.dumps(state_before, allow_nan=False)

    def test_linear_svc_cutoff_is_fixed_at_margin_zero_not_calibrated(self):
        model = build_campaign_classifier("DC_SVC").fit(
            self.frame.iloc[:150][self.features].copy(), self.y[:150]
        )
        held = self.frame.iloc[150:][self.features].copy()
        margin = model.predict_margin(held)
        probability_view = model.predict_proba(held)
        np.testing.assert_allclose(probability_view, expit(margin), rtol=0, atol=1e-15)
        np.testing.assert_array_equal(margin >= 0, probability_view >= 0.5)
        self.assertIn("uncalibrated", model.prediction_diagnostics()["decision_rule"])
        self.assertEqual(model.audit_state()["decision_rule"]["positive_class_margin"], ">= 0")

    def test_nbs_vocabulary_ratio_uses_fit_only_l1_normalized_document_presence(self):
        train = self.frame.iloc[:150][self.features].copy()
        y_train = self.y[:150]
        model = build_campaign_classifier("DC_NBSVM").fit(train, y_train)
        state = model.audit_state()["nb_svm"]
        ratio = model._nb_log_count_ratio
        self.assertEqual(len(ratio), state["counts"]["word_vocabulary_size"])
        positive_total = state["counts"]["positive_documents_with_word_total"]
        negative_total = state["counts"]["negative_documents_with_word_total"]
        vocab_size = state["counts"]["word_vocabulary_size"]
        self.assertAlmostEqual(
            state["counts"]["positive_l1_normalization_total_with_smoothing"],
            positive_total + vocab_size,
        )
        self.assertAlmostEqual(
            state["counts"]["negative_l1_normalization_total_with_smoothing"],
            negative_total + vocab_size,
        )
        self.assertTrue(np.isfinite(ratio).all())
        self.assertIn("NB-SVM-inspired", state["method"])

    def test_word_char_block_is_fit_only_and_audited(self):
        model = build_campaign_classifier("DC_WORD_CHAR").fit(
            self.frame.iloc[:150][self.features].copy(), self.y[:150]
        )
        state = model.audit_state()
        char = state["char_text"]
        self.assertEqual(char["parameters"]["analyzer"], "char_wb")
        self.assertEqual(char["parameters"]["ngram_range"], [3, 5])
        self.assertEqual(char["parameters"]["max_features"], 30000)
        self.assertGreater(char["vocabulary_size"], 0)


if __name__ == "__main__":
    unittest.main()
