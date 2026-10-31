"""C04 synthetic route, feature, and resource-contract checks."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.optimization_v3_batch1_c04 import (
    _frozen_identity_matches,
    _macro_f1,
    _resource_estimate,
    _reconstruct_c04,
)
from src.models.classification_models import _TFIDF_PARAMS
from src.models.optimization_v3_batch1 import json_sha256
from src.models.optimization_v3_batch1_c04 import (
    CARD_ID,
    FEATURE_NAMES,
    TFIDF_PARAMS,
    WiderCombinedWordClassifier,
    fit_predict,
)


def _features(n_rows: int, *, start_index: int = 0) -> pd.DataFrame:
    rows = []
    for index in range(n_rows):
        rows.append({
            "impact_load": 20 + index,
            "entry_friction": index % 2,
            "cycle_code": 55 + (index % 2),
            "scope_pressure": 10 + index,
            "text_load": 3 + (index % 4),
            "scope_trace": "1",
            "org_spread": "2",
            "module_spread": "1",
            "telemetry_flag": index % 2,
            "org_code": f"org-{index % 3}",
            "module_code": f"mod-{index % 2}",
            "pattern_code": f"pat-{index % 4}",
            "signal_stream": None if index % 7 == 0 else "alpha beta gamma delta",
        })
    return pd.DataFrame(rows, columns=FEATURE_NAMES, index=np.arange(start_index, start_index + n_rows))


def test_recipe_changes_only_the_word_vocabulary_cap() -> None:
    assert set(TFIDF_PARAMS) == set(_TFIDF_PARAMS)
    for key, value in _TFIDF_PARAMS.items():
        if key == "max_features":
            assert TFIDF_PARAMS[key] == 60_000
        else:
            assert TFIDF_PARAMS[key] == value
    assert TFIDF_PARAMS["ngram_range"] == (1, 2)
    assert TFIDF_PARAMS["min_df"] == 3
    assert TFIDF_PARAMS["max_df"] == 0.995


def test_adapter_fits_synthetic_train_and_preserves_missing_stream_route() -> None:
    fit_x = _features(24)
    fit_y = np.tile(np.array([0, 1], dtype=np.int8), 12)
    score_x = _features(4, start_index=100)
    documents_missing = score_x["signal_stream"].isna().to_numpy()
    c2b = np.array([0.30, 0.45, 0.60, 0.75], dtype=np.float64)
    cb = np.array([0.12, 0.22, 0.32, 0.42], dtype=np.float64)
    old_sparse = np.array([0.20, 0.35, 0.50, 0.65], dtype=np.float64)
    baseline = np.where(documents_missing, cb, 0.75 * c2b + 0.25 * old_sparse)
    controls = {
        "baseline_prediction": baseline,
        "baseline_sparse_raw": old_sparse,
        "c2b_probability": c2b,
        "cb_probability": cb,
        "stream_missing": documents_missing,
        "control_rows": score_x.index.to_numpy(dtype=np.int64),
    }

    result = fit_predict(
        fit_x, fit_y, score_x,
        controls=controls, card_id=CARD_ID, cutoff=55,
    )

    prediction = np.asarray(result["prediction"], dtype=np.float64)
    raw = np.asarray(result["raw_candidate"], dtype=np.float64)
    assert result["fit_predictor_count"] == 1
    assert prediction.shape == raw.shape == (len(score_x),)
    assert np.isfinite(prediction).all() and np.all((prediction >= 0) & (prediction <= 1))
    np.testing.assert_array_equal(prediction[documents_missing], cb[documents_missing])
    np.testing.assert_allclose(
        prediction[~documents_missing],
        0.75 * c2b[~documents_missing] + 0.25 * raw[~documents_missing],
        rtol=0.0, atol=1e-12,
    )
    state = result["state"]["model"]
    assert state["vectorizer"]["parameters"]["max_features"] == 60_000
    assert state["validation_data_passed_to_fit"] is False
    assert state["test_data_used"] is False


def test_independent_route_reconstruction_matches_frozen_policy() -> None:
    arrays = {
        "raw_candidate": np.array([0.10, 0.90, 0.50]),
        "control__stream_missing": np.array([False, True, False]),
        "control__c2b_probability": np.array([0.20, 0.30, 0.70]),
        "control__cb_probability": np.array([0.40, 0.50, 0.60]),
        "control__baseline_sparse_raw": np.array([0.80, 0.10, 0.30]),
    }
    candidate, baseline = _reconstruct_c04(arrays)
    np.testing.assert_allclose(candidate, [0.175, 0.50, 0.65], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(baseline, [0.35, 0.50, 0.60], rtol=0.0, atol=1e-12)


def test_frozen_identity_roundtrip_handles_integer_dictionary_keys() -> None:
    identity = {"expected_fit_rows": {55: 12_020, 67: 29_705}, "card_id": "C04"}
    stored_json = json.loads(json.dumps(identity))
    frozen = {"identity": stored_json, "signature": json_sha256(identity)}
    assert _frozen_identity_matches(frozen, identity)

    stored_json["expected_fit_rows"]["55"] = 12_021
    assert not _frozen_identity_matches(frozen, identity)


def test_macro_f1_uses_the_frozen_half_threshold() -> None:
    y = np.array([0, 0, 1, 1], dtype=np.int8)
    probabilities = np.array([0.49, 0.50, 0.50, 0.51], dtype=np.float64)
    assert _macro_f1(y, probabilities) == pytest.approx(0.7333333333333334)


def test_sparse_memory_preflight_blocks_when_reserve_is_unavailable(monkeypatch) -> None:
    from scripts import optimization_v3_batch1_c04 as runner

    fit_x = _features(8)
    monkeypatch.setattr(runner.psutil, "virtual_memory", lambda: SimpleNamespace(available=1024**3))
    result = _resource_estimate(fit_x)
    assert result["status"] == "BLOCKED_RESOURCE"
    assert result["sparse_only"] is True
    assert result["estimated_peak_is_observed"] is False
