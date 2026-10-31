from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.classification_models import (
    _TFIDF_PARAMS,
    _json_digest,
)
from src.models.optimization_v2_unigram import build_unigram_classifier
from src.models.round1_models import EXPANDED_FEATURES


def _synthetic_frame(n: int = 180) -> tuple[pd.DataFrame, np.ndarray]:
    idx = np.arange(n)
    frame = pd.DataFrame({
        "impact_load": 40 + (idx % 23),
        "entry_friction": idx % 2,
        "cycle_code": 47 + (idx % 9),
        "scope_pressure": (idx % 17) / 17,
        "text_load": 1 + (idx % 11),
        "scope_trace": [f"{2 + (i % 7)} nodes" for i in idx],
        "org_spread": [str(1 + (i % 5)) for i in idx],
        "module_spread": [str(1 + (i % 6)) for i in idx],
        "telemetry_flag": np.where(idx % 3, "active", "inactive"),
        "org_code": [f"org_{i % 4}" for i in idx],
        "module_code": [f"mod_{i % 6}" for i in idx],
        "pattern_code": [f"pat_{i % 3}" for i in idx],
        "signal_stream": [f"knownword token{(i % 4)} repeated" for i in idx],
    })
    assert set(EXPANDED_FEATURES).issubset(frame.columns)
    y = ((idx % 5) < 2).astype(np.int8)
    return frame, y


def test_variants_are_instance_local_and_keep_exact_allowlist():
    before = _json_digest(_TFIDF_PARAMS)
    frame, y = _synthetic_frame()
    model_a = build_unigram_classifier("unigram_counts")
    model_b = build_unigram_classifier("unigram_binary")
    assert model_a.input_feature_names == model_b.input_feature_names
    assert "artifact_signal" not in model_a.input_feature_names
    assert "record_id" not in model_a.input_feature_names
    assert "threat_rank" not in model_a.input_feature_names
    assert model_a._variant_tfidf_params["ngram_range"] == (1, 1)
    assert model_a._variant_tfidf_params["binary"] is False
    assert model_a._variant_tfidf_params["sublinear_tf"] is True
    assert model_b._variant_tfidf_params["ngram_range"] == (1, 1)
    assert model_b._variant_tfidf_params["binary"] is True
    assert model_b._variant_tfidf_params["sublinear_tf"] is False
    assert model_a._variant_tfidf_params["min_df"] == _TFIDF_PARAMS["min_df"] == 3
    assert model_a._variant_tfidf_params["max_df"] == _TFIDF_PARAMS["max_df"] == 0.995
    assert model_a._variant_tfidf_params["max_features"] == _TFIDF_PARAMS["max_features"] == 20000
    model_a.fit(frame, y)
    model_b.fit(frame, y)
    assert _json_digest(_TFIDF_PARAMS) == before
    assert model_a.audit_state()["text_representation"]["actual_text_state"]["parameters"]["ngram_range"] == [1, 1]
    assert model_b.audit_state()["text_representation"]["actual_text_state"]["effective_vectorizer_parameters"]["binary"] is True


def test_heldout_vocabulary_and_model_state_remain_fit_only():
    frame, y = _synthetic_frame()
    model = build_unigram_classifier("unigram_counts").fit(frame, y)
    before = model.audit_state()
    vocabulary = set(model._text_vectorizer.vocabulary_)
    assert "heldoutonlytoken" not in vocabulary

    heldout = frame.iloc[:8].copy()
    heldout["signal_stream"] = "heldoutonlytoken unseenfutureword"
    heldout["org_code"] = "future_org_code"
    probability = model.predict_proba(heldout)
    after = model.audit_state()

    assert probability.shape == (len(heldout),)
    assert np.isfinite(probability).all()
    assert set(model._text_vectorizer.vocabulary_) == vocabulary
    assert after == before
    assert model.prediction_diagnostics()["unseen_tfidf_token_count"] == 16


def test_missing_text_training_rows_are_kept_and_blank_is_valid_at_prediction():
    frame, y = _synthetic_frame()
    frame.loc[::10, "signal_stream"] = ""
    model = build_unigram_classifier("unigram_binary").fit(frame, y)
    assert model.audit_state()["fit_rows"] == len(frame)
    assert model.audit_state()["text_representation"]["actual_text_state"]["parameters"]["binary"] is True
    test = frame.iloc[:6].copy()
    test.loc[test.index[:2], "signal_stream"] = ""
    scores = model.predict_proba(test)
    assert scores.shape == (len(test),)
    assert np.isfinite(scores).all()
