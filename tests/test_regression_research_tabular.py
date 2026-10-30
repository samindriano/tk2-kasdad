"""Synthetic-only checks for the bounded LightGBM MAE adapter."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.models.round1_models import CATEGORICAL_FEATURES, EXPANDED_FEATURES
from src.models.regression_research_tabular import (
    COMMON_PARAMETERS,
    PARAMETER_SETS,
    build_tabular,
)


def _synthetic_frame(rows: int = 180) -> tuple[pd.DataFrame, np.ndarray]:
    position = np.arange(rows)
    frame = pd.DataFrame({
        "impact_load": 20.0 + (position % 61),
        "entry_friction": (position % 2).astype(float),
        "cycle_code": 47 + (position % 12),
        "scope_pressure": 5.0 + (position % 17),
        "text_load": 1.0 + (position % 9),
        "scope_trace": ["~4 nodes" if i % 13 == 0 else f"{i % 8} nodes" for i in position],
        "org_spread": [f"{i % 7}" for i in position],
        "module_spread": [f"{i % 5} nodes" for i in position],
        "telemetry_flag": ["active" if i % 3 else "inactive" for i in position],
        "org_code": [f"org_{i % 6}" for i in position],
        "module_code": [f"module_{i % 4}" for i in position],
        "pattern_code": [f"pattern_{i % 3}" for i in position],
    })
    target = 0.35 * frame.impact_load.to_numpy() + 3.0 * frame.entry_friction.to_numpy()
    target += (position % 11).astype(float) + np.where(position % 7 == 0, 4.0, 0.0)
    return frame, target.astype(float)


class RegressionResearchTabularTests(unittest.TestCase):
    def test_preregistered_parameter_sets_and_deterministic_repeat(self):
        frame, target = _synthetic_frame()
        first = build_tabular("base").fit(frame, target)
        second = build_tabular("base").fit(frame, target)
        first_prediction = first.predict(frame.iloc[:25])
        second_prediction = second.predict(frame.iloc[:25])
        self.assertTrue(np.array_equal(first_prediction, second_prediction))
        state = first.audit_state()
        self.assertEqual(
            state["tree_complexity"],
            second.audit_state()["tree_complexity"],
            "Same synthetic fit, seed, version, and thread count should reproduce fitted tree structure.",
        )
        self.assertEqual(state["actual_hyperparameters"]["objective"], "regression_l1")
        self.assertEqual(state["actual_hyperparameters"]["n_jobs"], 2)
        self.assertTrue(state["actual_hyperparameters"]["deterministic"])
        self.assertTrue(state["actual_hyperparameters"]["force_col_wise"])
        self.assertEqual(state["lightgbm_version"], "4.6.0")
        self.assertEqual(state["fit_rows"], len(frame))
        self.assertTrue(state["tree_complexity"]["fitted_tree_sha256"])
        self.assertEqual(tuple(state["input_feature_names"]), EXPANDED_FEATURES)
        self.assertEqual(set(PARAMETER_SETS), {"base", "regularized", "deeper"})
        deeper = build_tabular("deeper")
        self.assertEqual(deeper.actual_parameters["num_leaves"], 63)
        self.assertEqual(deeper.actual_parameters["min_child_samples"], 30)
        self.assertEqual(deeper.actual_parameters["reg_lambda"], 2.0)
        self.assertEqual(COMMON_PARAMETERS["learning_rate"], 0.05)
        self.assertEqual(COMMON_PARAMETERS["n_estimators"], 400)

    def test_no_codes_ablation_uses_selected_allowlist_and_native_categories(self):
        frame, target = _synthetic_frame()
        features = [name for name in EXPANDED_FEATURES if name not in CATEGORICAL_FEATURES]
        model = build_tabular(features=features).fit(frame, target)
        state = model.audit_state()
        self.assertEqual(tuple(state["input_feature_names"]), tuple(features))
        self.assertEqual(state["categorical_feature_indices"], [])
        self.assertTrue(np.isfinite(model.predict(frame.iloc[:10])).all())

    def test_preprocessing_learns_from_fit_rows_and_handles_unseen_category(self):
        frame, target = _synthetic_frame()
        fit = frame.iloc[:120].copy()
        fit_target = target[:120]
        fit.loc[0, "impact_load"] = 100.0
        fit.loc[1, "impact_load"] = np.nan
        model = build_tabular().fit(fit, fit_target)
        before_prediction_state = model.audit_state()
        validation = frame.iloc[120:130].copy()
        validation.loc[validation.index[0], "impact_load"] = 1_000_000.0
        validation.loc[validation.index[0], "org_code"] = "never_seen"
        prediction = model.predict(validation)
        after_prediction_state = model.audit_state()
        self.assertTrue(np.isfinite(prediction).all())
        self.assertEqual(before_prediction_state, after_prediction_state)
        self.assertEqual(
            before_prediction_state["preprocessing"]["numeric_imputer_medians"]["impact_load"],
            float(np.nanmedian(fit.impact_load.to_numpy(dtype=float))),
        )
        self.assertEqual(model.prediction_diagnostics()["unknown_category_rows"]["org_code"], 1)
        self.assertFalse(before_prediction_state["validation_data_passed_to_fit"])

    def test_audit_state_is_stable_across_predict_calls(self):
        frame, target = _synthetic_frame(120)
        model = build_tabular("regularized").fit(frame, target)
        state = model.audit_state()
        _ = model.predict(frame.iloc[20:30])
        self.assertEqual(state, model.audit_state())
        _ = model.predict(frame.iloc[30:45])
        self.assertEqual(state, model.audit_state())

    def test_invalid_parameter_and_ineligible_features_are_rejected(self):
        with self.assertRaises(ValueError):
            build_tabular("wide_grid")
        with self.assertRaises(ValueError):
            build_tabular(features=["impact_load", "entry_friction", "record_id"])
        with self.assertRaises(TypeError):
            build_tabular(features="impact_load")


def features_without_codes() -> tuple[str, ...]:
    return tuple(name for name in EXPANDED_FEATURES if name not in CATEGORICAL_FEATURES)


if __name__ == "__main__":
    unittest.main()
