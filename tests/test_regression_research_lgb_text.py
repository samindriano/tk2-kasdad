"""Synthetic-only tests for LightGBM over the frozen SVD32 text design."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from src.models.regression_research_lgb_text import (
    LGB_TEXT_PARAMS,
    build_lgb_text,
)
from src.models.regression_research_text import (
    HYBRID_FEATURES,
    STRUCTURED_FEATURES,
    RegressionTextModel,
    _SVD_PARAMS,
    _array_digest,
    _text_documents,
)
from src.models.regression_round2_models import _SelectedPreprocessor
from src.models.regression_research_tabular import COMMON_PARAMETERS, PARAMETER_SETS


def _synthetic_frame(rows: int = 180) -> tuple[pd.DataFrame, np.ndarray]:
    i = np.arange(rows)
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


class RegressionResearchLGBTextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame, self.target = _synthetic_frame()

    def test_only_two_declared_parameter_sets_are_exposed(self) -> None:
        self.assertEqual(LGB_TEXT_PARAMS, ("base", "regularized"))
        self.assertEqual(set(PARAMETER_SETS) - {"deeper"}, set(LGB_TEXT_PARAMS))
        for params_id in LGB_TEXT_PARAMS:
            model = build_lgb_text(params_id)
            self.assertEqual(model.actual_parameters, {**COMMON_PARAMETERS, **PARAMETER_SETS[params_id]})
            self.assertEqual(model.actual_parameters["objective"], "regression_l1")
            self.assertEqual(model.actual_parameters["n_estimators"], 400)
            self.assertEqual(model.actual_parameters["n_jobs"], 2)
        with self.assertRaises(ValueError):
            build_lgb_text("deeper")

    def test_fit_representation_matches_frozen_full_svd32_design(self) -> None:
        model = build_lgb_text("base")
        train = self.frame.iloc[:150].loc[:, list(model.feature_names)].copy()
        model.fit(train, self.target[:150])
        state = model.audit_state()

        # Independent construction with the same frozen preprocessor, TF-IDF,
        # and SVD primitives; no HGB model is fit in this parity check.
        reference = _SelectedPreprocessor(tuple(STRUCTURED_FEATURES))
        structured, categorical_mask = reference.fit_transform(
            train.loc[:, list(STRUCTURED_FEATURES)]
        )
        docs, missing = _text_documents(train["signal_stream"])
        vectorizer, text_matrix, _ = RegressionTextModel._fit_vectorizer(docs)
        components = min(32, text_matrix.shape[1] - 1) if text_matrix.shape[1] > 1 else 0
        if components:
            svd = TruncatedSVD(n_components=components, **dict(_SVD_PARAMS))
            text_dense = svd.fit_transform(text_matrix)
        else:
            text_dense = np.empty((len(train), 0), dtype=np.float64)
        expected = np.column_stack((structured, text_dense, missing.astype(np.float64)))

        self.assertEqual(_array_digest(expected), state["training_design_sha256"])
        self.assertEqual(state["svd"]["n_components_requested"], 32)
        self.assertEqual(state["svd"]["n_components_actual"], components)
        self.assertEqual(state["structured_categorical_mask"], list(categorical_mask))
        self.assertEqual(
            state["categorical_feature_indices"],
            [i for i, is_cat in enumerate(categorical_mask) if is_cat],
        )
        self.assertEqual(state["tfidf"]["vocabulary_size"], len(vectorizer.vocabulary_))
        self.assertEqual(state["tfidf"]["idf_sha256"], _array_digest(vectorizer.idf_))
        self.assertFalse(state["validation_data_passed_to_fit"])
        self.assertFalse(state["test_data_used_for_fit"])

    def test_fit_only_unknown_categories_and_audit_are_stable(self) -> None:
        model = build_lgb_text("regularized")
        train = self.frame.iloc[:120].loc[:, list(model.feature_names)].copy()
        train.loc[train.index[0], "signal_stream"] = None
        model.fit(train, self.target[:120])
        before = model.audit_state()
        validation = self.frame.iloc[120:].loc[:, list(model.feature_names)].copy()
        validation.loc[validation.index[0], "org_code"] = "never_seen_org"
        validation.loc[validation.index[1], "module_code"] = "never_seen_module"
        validation.loc[validation.index[2], "impact_load"] = 1_000_000.0
        transformed, unknown_before_prediction = model._preprocessor.transform(
            validation.loc[:, list(STRUCTURED_FEATURES)]
        )
        org_position = model._preprocessor.transformed_feature_names.index("org_code__ordinal")
        module_position = model._preprocessor.transformed_feature_names.index("module_code__ordinal")
        self.assertEqual(transformed[0, org_position], -1.0)
        self.assertEqual(transformed[1, module_position], -1.0)
        self.assertEqual(unknown_before_prediction["org_code"], 1)
        prediction = model.predict(validation)
        self.assertTrue(np.isfinite(prediction).all())
        self.assertEqual(before, model.audit_state())
        diagnostics = model.prediction_diagnostics()
        self.assertEqual(diagnostics["n_predictions"], len(validation))
        self.assertEqual(diagnostics["unknown_category_rows"]["org_code"], 1)
        self.assertEqual(diagnostics["unknown_category_rows"]["module_code"], 1)
        self.assertEqual(before["fit_missing_stream_rows"], 1)
        self.assertEqual(before["actual_hyperparameters"]["num_leaves"], 15)
        self.assertEqual(before["actual_hyperparameters"]["min_child_samples"], 80)
        self.assertEqual(before["actual_hyperparameters"]["reg_lambda"], 3.0)
        self.assertIn("categorical_negative_values_are_missing", before)
        self.assertEqual(before["categorical_unknown_value"], -1)

    def test_audit_is_compact_and_rejects_forbidden_or_extra_columns(self) -> None:
        model = build_lgb_text()
        self.assertEqual(model.feature_names, tuple(HYBRID_FEATURES))
        for forbidden in ("record_id", "threat_rank", "artifact_signal", "audit_lane"):
            self.assertNotIn(forbidden, model.feature_names)
        with self.assertRaises(ValueError):
            model.fit(self.frame.assign(record_id="forbidden"), self.target)
        model.fit(self.frame.loc[:, list(model.feature_names)], self.target)
        serialized = str(model.audit_state())
        for raw_token in ("critical", "routine", "buffer", "overflow"):
            self.assertNotIn(raw_token, serialized)
        with self.assertRaises(ValueError):
            model.predict(self.frame.assign(record_id="forbidden"))
        with self.assertRaises(RuntimeError):
            build_lgb_text().predict(self.frame.loc[:, list(HYBRID_FEATURES)])


if __name__ == "__main__":
    unittest.main()
