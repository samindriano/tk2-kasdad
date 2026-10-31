"""Synthetic fit/predict checks for the isolated native-text challenger."""
from __future__ import annotations

import json
import unittest

import numpy as np
import pandas as pd

from src.models.optimization_v2_catboost_text import (
    INPUT_FEATURE_NAMES,
    EMPTY_STREAM_SENTINEL,
    TEXT_FEATURE,
    build_catboost_native_text_classifier,
)


def _fixture(rows: int = 96) -> tuple[pd.DataFrame, np.ndarray]:
    index = np.arange(rows)
    y = (index % 2).astype(np.int8)
    text = [
        f"shared telemetry signal class{label} repeat{index[i] % 4}"
        for i, label in enumerate(y)
    ]
    frame = pd.DataFrame({
        "impact_load": (index % 11).astype(float),
        "entry_friction": (index % 2).astype(float),
        "cycle_code": 47 + (index % 7),
        "scope_pressure": (index % 9).astype(float),
        "text_load": (index % 5).astype(float),
        "scope_trace": [f"{2 + i % 4} nodes" for i in index],
        "org_spread": [str(1 + i % 5) for i in index],
        "module_spread": [f"{2 + i % 3} nodes" for i in index],
        "telemetry_flag": ["active" if i % 2 else "inactive" for i in index],
        "org_code": [f"org{i % 4}" for i in index],
        "module_code": [f"module{i % 5}" for i in index],
        "pattern_code": [f"pattern{i % 3}" for i in index],
        TEXT_FEATURE: text,
    })
    frame.loc[5, TEXT_FEATURE] = None
    frame.loc[13, TEXT_FEATURE] = "   "
    frame.loc[21, TEXT_FEATURE] = np.nan
    return frame.loc[:, INPUT_FEATURE_NAMES], y


class OptimizationV2CatBoostTextTests(unittest.TestCase):
    def test_native_text_fit_unknown_tokens_and_immutable_state(self):
        frame, y = _fixture()
        train = frame.iloc[:80].copy(deep=True)
        held = frame.iloc[80:].copy(deep=True)
        before = train.copy(deep=True)
        held[TEXT_FEATURE] = ["never_seen_token alien_term novel_sequence"] * len(held)
        held.loc[held.index[0], TEXT_FEATURE] = None

        collision = train.copy(deep=True)
        collision.loc[collision.index[0], TEXT_FEATURE] = EMPTY_STREAM_SENTINEL
        with self.assertRaisesRegex(ValueError, "reserved empty-stream sentinel"):
            build_catboost_native_text_classifier().fit(collision, y[:80])

        model = build_catboost_native_text_classifier().fit(train, y[:80])
        state_before = model.audit_state()
        probability = model.predict_proba(held)

        self.assertEqual(model.input_feature_names, INPUT_FEATURE_NAMES)
        self.assertEqual(state_before["fit_rows"], 80)
        self.assertEqual(state_before["fit_stream_missing_rows"], 3)
        self.assertEqual(state_before["empty_stream_sentinel"], EMPTY_STREAM_SENTINEL)
        self.assertEqual(state_before["text_feature_names"], [TEXT_FEATURE])
        self.assertEqual(state_before["text_feature_indices"], [len(state_before["output_feature_names"]) - 1])
        self.assertEqual(state_before["text_processing"]["dictionaries"][0]["gram_order"], "1")
        self.assertEqual(state_before["text_processing"]["dictionaries"][0]["occurence_lower_bound"], "3")
        self.assertEqual(state_before["text_processing"]["dictionaries"][0]["max_dictionary_size"], "20000")
        self.assertEqual(state_before["text_processing"]["tokenizers"][0]["delimiter"], " ")
        self.assertEqual(state_before["text_processing"]["tokenizers"][0]["separator_type"], "ByDelimiter")
        self.assertEqual(state_before["text_processing"]["feature_processing"]["default"][0]["feature_calcers"],
                         ["BoW", "NaiveBayes"])
        self.assertEqual(state_before["fixed_parameters"]["iterations"], 500)
        self.assertEqual(state_before["fixed_parameters"]["depth"], 6)
        self.assertEqual(state_before["fixed_parameters"]["learning_rate"], 0.05)
        self.assertEqual(state_before["fixed_parameters"]["random_seed"], 42)
        self.assertEqual(state_before["fixed_parameters"]["thread_count"], 2)
        self.assertEqual(state_before["tree_count"], 500)
        self.assertFalse(state_before["eval_set_used"])
        self.assertFalse(state_before["early_stopping_used"])
        self.assertEqual(probability.shape, (16,))
        self.assertTrue(np.isfinite(probability).all())
        self.assertTrue(((probability >= 0) & (probability <= 1)).all())
        self.assertEqual(state_before, model.audit_state())
        pd.testing.assert_frame_equal(train, before)
        self.assertIsNone(held.loc[held.index[0], TEXT_FEATURE])
        self.assertTrue((held[TEXT_FEATURE].dropna() ==
                         "never_seen_token alien_term novel_sequence").all())
        json.dumps(state_before, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
