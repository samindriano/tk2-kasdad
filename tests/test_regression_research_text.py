import unittest

import numpy as np
import pandas as pd

from src.models.regression_research_text import (
    HYBRID_FEATURES,
    STRUCTURED_FEATURES,
    TEXT_ONLY_FEATURES,
    build_text_model,
)
from src.models.round1_models import build_model as build_round1_model


def make_frame(n=180):
    i = np.arange(n)
    frame = pd.DataFrame({
        "impact_load": (10 + i % 13).astype(float),
        "entry_friction": (i % 2).astype(float),
        "cycle_code": 47 + i // 30,
        "scope_pressure": (i % 9).astype(float),
        "text_load": (i % 8).astype(float),
        "scope_trace": [f"{i % 7} nodes" for i in range(n)],
        "org_spread": [f"{i % 6} nodes" for i in range(n)],
        "module_spread": [f"{i % 5} nodes" for i in range(n)],
        "telemetry_flag": ["active" if x % 2 else "inactive" for x in i],
        "org_code": [f"org-{x % 4}" for x in i],
        "module_code": [f"mod-{x % 5}" for x in i],
        "pattern_code": [f"pat-{x % 3}" for x in i],
        "signal_stream": [
            None if x % 11 == 0 else "critical buffer overflow" if x % 3 == 0
            else "routine access control" if x % 3 == 1 else "remote execution warning"
            for x in i
        ],
    })
    return frame


class RegressionResearchTextTests(unittest.TestCase):
    def setUp(self):
        self.frame = make_frame()
        numeric = self.frame.impact_load.to_numpy() * .35 + self.frame.entry_friction.to_numpy() * 6
        lexical = self.frame.signal_stream.str.contains("critical", na=False).to_numpy() * 12
        self.y = np.clip(20 + numeric + lexical, 0, 100)

    def test_declared_feature_sets_are_exact(self):
        text = build_text_model("text_ridge")
        hybrid = build_text_model("hybrid_ridge")
        self.assertEqual(text.feature_names, TEXT_ONLY_FEATURES)
        self.assertEqual(hybrid.feature_names, HYBRID_FEATURES)
        self.assertEqual(len(STRUCTURED_FEATURES), 12)

    def test_text_only_ridge_learns_vocab_and_keeps_missing_indicator(self):
        selected = self.frame.loc[:, list(TEXT_ONLY_FEATURES)]
        model = build_text_model("text_ridge").fit(selected.iloc[:150], self.y[:150])
        before = model.audit_state()
        pred = model.predict(selected.iloc[150:])
        after = model.audit_state()
        self.assertEqual(pred.shape, (30,))
        self.assertTrue(np.isfinite(pred).all())
        self.assertEqual(before, after)
        self.assertEqual(before["tfidf"]["parameters"]["min_df"], 2)
        self.assertEqual(before["tfidf"]["parameters"]["ngram_range"], [1, 2])
        self.assertEqual(before["parameters"]["missing_stream_indicator"], True)
        self.assertGreater(before["tfidf"]["vocabulary_size"], 0)
        self.assertEqual(len(before["tfidf"]["vocabulary_sha256"]), 64)

    def test_hybrid_ridge_fit_only_preprocessing_missing_and_unknowns(self):
        train = self.frame.iloc[:150].copy()
        val = self.frame.iloc[150:].copy()
        val.iloc[0, val.columns.get_loc("signal_stream")] = "novel zebra phrase"
        val.iloc[1, val.columns.get_loc("org_code")] = "never-seen-org"
        model = build_text_model("hybrid_ridge").fit(train.loc[:, list(HYBRID_FEATURES)], self.y[:150])
        state_before = model.audit_state()
        pred = model.predict(val.loc[:, list(HYBRID_FEATURES)])
        state_after = model.audit_state()
        self.assertTrue(np.isfinite(pred).all())
        self.assertEqual(state_before, state_after)
        self.assertEqual(state_before["transform"]["missing_indicator"], True)
        self.assertGreaterEqual(model.prediction_diagnostics()["unknown_category_rows"]["org_code"], 1)
        self.assertGreaterEqual(model.prediction_diagnostics()["unseen_tfidf_token_count"], 1)
        # Missing and blank text rows have a zero TF-IDF block but an explicit
        # availability feature, rather than disappearing into the intercept.
        missing_rows = val.iloc[:2].copy()
        missing_rows.iloc[:, missing_rows.columns.get_loc("signal_stream")] = [None, ""]
        design, missing, _, _ = model._design(missing_rows.loc[:, list(HYBRID_FEATURES)])
        width = state_before["transform"]["structured_width"]
        text_width = state_before["transform"]["text_width"]
        self.assertTrue(np.all(missing))
        self.assertEqual(design.shape[1], width + text_width + 1)
        self.assertTrue(np.allclose(design[:, width:width+text_width].toarray(), 0.))
        self.assertTrue(np.allclose(design[:, -1].toarray().reshape(-1), 1.))

    def test_svd_hgb_is_deterministic_and_caps_components_to_vocabulary(self):
        train = self.frame.iloc[:150].copy()
        val = self.frame.iloc[150:].copy()
        first = build_text_model("svd_hgb").fit(train.loc[:, list(HYBRID_FEATURES)], self.y[:150])
        second = build_text_model("svd_hgb").fit(train.loc[:, list(HYBRID_FEATURES)], self.y[:150])
        reference = build_round1_model({
            "id": "R1_D",
            "kind": "hgb",
            "features": "expanded",
            "params": {"loss": "absolute_error"},
        }).fit(train.loc[:, list(STRUCTURED_FEATURES)], self.y[:150])
        state = first.audit_state()
        pred1 = first.predict(val.loc[:, list(HYBRID_FEATURES)])
        pred2 = second.predict(val.loc[:, list(HYBRID_FEATURES)])
        np.testing.assert_array_equal(pred1, pred2)
        self.assertEqual(state["fitted_tree_sha256"], second.audit_state()["fitted_tree_sha256"])
        self.assertEqual(state["parameters"]["hgb"]["loss"], "absolute_error")
        self.assertLessEqual(state["parameters"]["svd"]["n_components_actual"], 32)
        self.assertEqual(state["parameters"]["missing_stream_indicator"], True)
        # The structured block and categorical positions must match the frozen
        # Round 1 expanded HGB preprocessing; text SVD adds only numeric cols.
        reference_matrix, reference_unknown = reference._preprocessor.transform(
            val.loc[:, list(STRUCTURED_FEATURES)]
        )
        candidate_matrix, candidate_unknown = first._processor.transform(
            val.loc[:, list(STRUCTURED_FEATURES)]
        )
        np.testing.assert_array_equal(candidate_matrix, reference_matrix)
        self.assertEqual(candidate_unknown, reference_unknown)
        self.assertEqual(
            state["transform"]["structured"]["transformed_feature_names"],
            reference._preprocessor.transformed_feature_names,
        )
        self.assertEqual(
            state["transform"]["structured_categorical_mask"],
            [i in reference._preprocessor.categorical_positions for i in range(reference_matrix.shape[1])],
        )
        design_mask = state["transform"]["design_categorical_mask"]
        self.assertEqual(design_mask[:reference_matrix.shape[1]], state["transform"]["structured_categorical_mask"])
        self.assertFalse(any(design_mask[reference_matrix.shape[1]:]))
        self.assertEqual(first._estimator.get_params(deep=False)["categorical_features"], design_mask)

    def test_fit_target_identity_and_leakage_allowlist(self):
        selected = self.frame.loc[:, list(TEXT_ONLY_FEATURES)]
        first = build_text_model("text_ridge").fit(selected, self.y)
        changed = self.y.copy()
        changed[0] += 1
        second = build_text_model("text_ridge").fit(selected, changed)
        self.assertNotEqual(first.audit_state()["target_sha256"], second.audit_state()["target_sha256"])
        self.assertNotEqual(first.audit_state()["fit_data_sha256"], second.audit_state()["fit_data_sha256"])
        for extra in ("record_id", "threat_rank", "pulse_a", "audit_lane"):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                build_text_model("text_ridge").fit(selected.assign(**{extra: 1}), self.y)

    def test_rejects_nonfinite_targets_and_unfitted_prediction(self):
        model = build_text_model("text_ridge")
        with self.assertRaises(RuntimeError):
            model.predict(self.frame.loc[:, list(TEXT_ONLY_FEATURES)])
        with self.assertRaises(ValueError):
            model.fit(self.frame.loc[:, list(TEXT_ONLY_FEATURES)], np.full(len(self.frame), np.nan))


if __name__ == "__main__":
    unittest.main()
