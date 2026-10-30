"""Synthetic-only contracts for the bounded regression refinement recipes."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.models.regression_research_refinement import (
    NO_SCOPE_FEATURES,
    NO_SCOPE_STRUCTURED_FEATURES,
    REFINEMENT_RECIPES,
    SCOPE_SPREAD_FEATURES,
    build_refinement,
)
from src.models.regression_research_text import (
    HYBRID_FEATURES,
    STRUCTURED_FEATURES,
    build_text_model,
)


def _synthetic_frame(rows: int = 180, *, many_terms: bool = False) -> tuple[pd.DataFrame, np.ndarray]:
    i = np.arange(rows)
    if many_terms:
        docs = [" ".join(f"word{j:03d}" for j in range(offset, 180, 5)) for offset in (i % 5)]
    else:
        docs = [
            "critical buffer overflow" if x % 3 == 0
            else "routine access control" if x % 3 == 1
            else "remote execution warning"
            for x in i
        ]
    frame = pd.DataFrame({
        "impact_load": 20.0 + (i % 61),
        "entry_friction": (i % 2).astype(float),
        "cycle_code": 47 + (i % 12),
        "scope_pressure": 68.0 + (i % 19),
        "text_load": 1.0 + (i % 9),
        "scope_trace": [f"{x % 8} nodes" for x in i],
        "org_spread": [f"{x % 7} nodes" for x in i],
        "module_spread": [f"{x % 5} nodes" for x in i],
        "telemetry_flag": ["active" if x % 3 else "inactive" for x in i],
        "org_code": [f"org_{x % 6}" for x in i],
        "module_code": [f"module_{x % 4}" for x in i],
        "pattern_code": [f"pattern_{x % 3}" for x in i],
        "signal_stream": docs,
    })
    target = 22.0 + .25 * frame.impact_load.to_numpy() + 4.0 * frame.entry_friction.to_numpy()
    target += frame.signal_stream.str.contains("critical", regex=False).to_numpy() * 7.0
    return frame, np.clip(target, 0.0, 100.0)


class RegressionResearchRefinementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame, self.target = _synthetic_frame()

    def test_recipes_and_no_scope_allowlist_remove_exact_source_group(self) -> None:
        self.assertEqual(REFINEMENT_RECIPES, ("svd128", "svd_noscope32", "structured_ridge"))
        self.assertEqual(
            NO_SCOPE_STRUCTURED_FEATURES,
            tuple(name for name in STRUCTURED_FEATURES if name not in SCOPE_SPREAD_FEATURES),
        )
        self.assertEqual(set(SCOPE_SPREAD_FEATURES), {
            "scope_pressure", "scope_trace", "org_spread", "module_spread",
        })
        model = build_refinement("svd_noscope32")
        self.assertEqual(model.feature_names, NO_SCOPE_FEATURES)
        self.assertFalse(set(SCOPE_SPREAD_FEATURES) & set(model.feature_names))
        for forbidden in ("record_id", "threat_rank", "artifact_signal", "signal_stream__missing"):
            self.assertNotIn(forbidden, model.feature_names)

    def test_svd128_uses_instance_cap_without_changing_old_32_component_builder(self) -> None:
        frame, target = _synthetic_frame(many_terms=True)
        features = build_refinement("svd128")
        original_32 = build_text_model("svd_hgb")
        self.assertEqual(features.feature_names, original_32.feature_names)
        self.assertEqual(original_32.svd_components, 32)

        train = frame.iloc[:150].loc[:, list(features.feature_names)]
        validation = frame.iloc[150:].loc[:, list(features.feature_names)]
        features.fit(train, target[:150])
        state = features.audit_state()
        pred = features.predict(validation)
        self.assertEqual(state["svd_components_requested"], 128)
        self.assertEqual(state["svd_components_actual"], 128)
        self.assertEqual(state["delegated_model"]["parameters"]["svd"]["n_components_requested"], 128)
        self.assertTrue(np.isfinite(pred).all())
        self.assertFalse(features.audit_state()["test_data_used_for_fit"])
        # The legacy builder still produces the frozen 32-component recipe.
        self.assertEqual(build_text_model("svd_hgb").svd_components, 32)

    def test_no_scope_svd_uses_fit_only_native_categories_and_stable_audit(self) -> None:
        model = build_refinement("svd_noscope32")
        train = self.frame.iloc[:120].loc[:, list(model.feature_names)].copy()
        target = self.target[:120]
        model.fit(train, target)
        before = model.audit_state()
        self.assertEqual(before["removed_scope_spread_sources_and_derived_flags"], list(SCOPE_SPREAD_FEATURES))
        self.assertEqual(before["fit_rows"], 120)
        mask = before["native_categorical_mask"]
        self.assertTrue(any(mask))
        self.assertFalse(any(mask[-(before["svd"]["n_components_actual"] + 1):]))
        transformed = before["structured_preprocessing"]["transformed_feature_names"]
        self.assertFalse(any(
            name == field or name.startswith(field + "__")
            for name in transformed for field in SCOPE_SPREAD_FEATURES
        ))

        validation = self.frame.iloc[120:].loc[:, list(model.feature_names)].copy()
        validation.loc[validation.index[0], "org_code"] = "never_seen_org"
        validation.loc[validation.index[1], "impact_load"] = 1_000_000.0
        predictions = model.predict(validation)
        self.assertEqual(predictions.shape, (len(validation),))
        self.assertTrue(np.isfinite(predictions).all())
        self.assertEqual(before, model.audit_state())
        self.assertEqual(model.prediction_diagnostics()["unknown_category_rows"]["org_code"], 1)
        with self.assertRaises(ValueError):
            model.predict(validation.assign(scope_pressure=70.0))

    def test_structured_ridge_matches_hybrid_ridge_with_constant_empty_stream(self) -> None:
        features = build_refinement("structured_ridge")
        train = self.frame.iloc[:150].loc[:, list(STRUCTURED_FEATURES)].copy()
        validation = self.frame.iloc[150:].loc[:, list(STRUCTURED_FEATURES)].copy()
        features.fit(train, self.target[:150])
        state = features.audit_state()
        prediction = features.predict(validation)

        reference_train = train.copy()
        reference_train["signal_stream"] = ""
        reference_validation = validation.copy()
        reference_validation["signal_stream"] = ""
        reference = build_text_model("hybrid_ridge").fit(
            reference_train.loc[:, list(HYBRID_FEATURES)], self.target[:150]
        )
        expected = reference.predict(reference_validation.loc[:, list(HYBRID_FEATURES)])
        np.testing.assert_array_equal(prediction, expected)
        self.assertEqual(features.feature_names, tuple(STRUCTURED_FEATURES))
        self.assertFalse(state["constructed_signal_stream"]["actual_signal_stream_consumed"])
        self.assertEqual(state["constructed_signal_stream"]["constant_missing_indicator"], 1)
        self.assertEqual(state["delegated_model"]["parameters"]["ridge"]["alpha"], 10.0)
        self.assertEqual(state, features.audit_state())
        self.assertEqual(features.prediction_diagnostics()["actual_signal_stream_consumed"], False)
        with self.assertRaises(ValueError):
            features.predict(validation.assign(signal_stream="should_be_rejected"))
        with self.assertRaises(ValueError):
            build_refinement("structured_ridge").fit(train.assign(record_id="forbidden"), self.target[:150])

    def test_all_recipes_reject_unfitted_prediction_and_invalid_names(self) -> None:
        for name in REFINEMENT_RECIPES:
            with self.subTest(recipe=name):
                model = build_refinement(name)
                with self.assertRaises(RuntimeError):
                    model.predict(self.frame.loc[:, list(model.feature_names)])
        with self.assertRaises(ValueError):
            build_refinement("svd512")


if __name__ == "__main__":
    unittest.main()
