"""Focused synthetic checks for the C11 word plus character adapter."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import src.models.optimization_v3_batch1_c11 as c11
from src.models.classification_models import _TFIDF_PARAMS, _text_values
from src.models.optimization_v3_batch1_c11 import (
    CARD_ID,
    CHAR_TFIDF_PARAMS,
    FEATURE_NAMES,
    WORD_TFIDF_PARAMS,
    WordCharSparseClassifier,
    fit_predict,
)


class _FakeBooster:
    def model_to_string(self) -> str:
        return "synthetic C11 booster"

    def predict(self, matrix, *, num_threads: int) -> np.ndarray:
        assert num_threads == c11.THREADS
        return np.full(matrix.shape[0], 0.60, dtype=np.float64)


class _FakeLGBMClassifier:
    def __init__(self, **parameters) -> None:
        self.parameters = parameters

    def fit(self, matrix, target):
        self.matrix = matrix.copy()
        self.target = np.asarray(target).copy()
        self.booster_ = _FakeBooster()
        return self


def _features(streams: list[str | None], indexes: list[int] | None = None) -> pd.DataFrame:
    if indexes is None:
        indexes = list(range(len(streams)))
    rows = []
    for position, (index, stream) in enumerate(zip(indexes, streams, strict=True)):
        rows.append({
            "impact_load": 20 + position,
            "entry_friction": position % 2,
            "cycle_code": 55 + (position % 2),
            "scope_pressure": 10 + position,
            "text_load": 3 + (position % 4),
            "scope_trace": str(1 + (position % 3)),
            "org_spread": str(2 + (position % 2)),
            "module_spread": str(1 + (position % 2)),
            "telemetry_flag": position % 2,
            "org_code": f"org-{position % 3}",
            "module_code": f"mod-{position % 2}",
            "pattern_code": f"pat-{position % 4}",
            "signal_stream": stream,
        })
    return pd.DataFrame(rows, columns=FEATURE_NAMES, index=np.asarray(indexes, dtype=np.int64))


def _install_fake_estimator(monkeypatch) -> None:
    monkeypatch.setattr(c11, "LGBMClassifier", _FakeLGBMClassifier)


def _controls(score_x: pd.DataFrame) -> dict[str, np.ndarray]:
    _, missing = _text_values(score_x["signal_stream"])
    size = len(score_x)
    c2b = np.linspace(0.20, 0.50, size, dtype=np.float64)
    cb = np.linspace(0.35, 0.65, size, dtype=np.float64)
    old_sparse = np.linspace(0.70, 0.40, size, dtype=np.float64)
    baseline = np.where(missing, cb, 0.75 * c2b + 0.25 * old_sparse)
    return {
        "baseline_prediction": baseline,
        "baseline_sparse_raw": old_sparse,
        "c2b_probability": c2b,
        "cb_probability": cb,
        "stream_missing": missing,
        "control_rows": score_x.index.to_numpy(dtype=np.int64),
    }


def test_fixed_word_recipe_is_retained_and_character_recipe_matches_card():
    assert WORD_TFIDF_PARAMS == _TFIDF_PARAMS
    assert CHAR_TFIDF_PARAMS == {
        "analyzer": "char_wb",
        "ngram_range": (3, 5),
        "min_df": 3,
        "max_features": 30_000,
        "sublinear_tf": True,
        "lowercase": True,
        "strip_accents": "unicode",
        "dtype": np.float64,
    }
    assert CARD_ID == "C11"
    assert c11.AUTHORIZED_CUTOFFS == frozenset({55, 67})


def test_word_and_character_vocabulary_and_idf_are_fit_only(monkeypatch):
    _install_fake_estimator(monkeypatch)
    fit_x = _features([
        "alpha anchor", "alpha anchor", "alpha anchor", "alpha anchor",
        "beta anchor", "beta anchor", "gamma birch", "delta birch",
    ])
    fit_y = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int8)
    model = WordCharSparseClassifier().fit(fit_x, fit_y)

    word_names = model._word_vectorizer.get_feature_names_out().tolist()
    char_names = model._char_vectorizer.get_feature_names_out().tolist()
    assert "anchor" in word_names
    assert "validationonly" not in word_names
    assert "anc" in char_names
    word_idf = dict(zip(word_names, model._word_vectorizer.idf_, strict=True))
    assert word_idf["anchor"] == pytest.approx(math.log((1 + 8) / (1 + 6)) + 1.0)

    score_x = _features(["validationonly anchor"], indexes=[1001])
    word_before = model._word_vectorizer.idf_.copy()
    char_before = model._char_vectorizer.idf_.copy()
    probability = model.predict_proba(score_x)
    assert probability.shape == (1,)
    np.testing.assert_array_equal(model._word_vectorizer.idf_, word_before)
    np.testing.assert_array_equal(model._char_vectorizer.idf_, char_before)
    state = model.audit_state()
    assert state["vectorizers"]["word"]["parameters"]["min_df"] == 3
    assert state["vectorizers"]["char_wb"]["parameters"]["analyzer"] == "char_wb"
    assert state["validation_data_passed_to_fit"] is False
    assert state["test_data_used"] is False


def test_route_uses_numeric_row_positions_and_frozen_missing_fallback(monkeypatch):
    _install_fake_estimator(monkeypatch)
    fit_x = _features([
        "alpha anchor", "alpha anchor", "alpha anchor", "alpha anchor",
        "beta anchor", "beta anchor", "gamma anchor", "delta anchor",
    ])
    fit_y = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int8)
    row_positions = [101, 205, 502, 999]
    score_x = _features([None, "", "alpha anchor", "gamma anchor"], indexes=row_positions)
    controls = _controls(score_x)
    _, missing = _text_values(score_x["signal_stream"])

    result = fit_predict(fit_x, fit_y, score_x, controls=controls, card_id=CARD_ID, cutoff=55)

    assert result["fit_predictor_count"] == 1
    np.testing.assert_array_equal(result["audit_arrays"]["control_rows"], row_positions)
    np.testing.assert_array_equal(result["prediction"][missing], controls["cb_probability"][missing])
    np.testing.assert_allclose(
        result["prediction"][~missing],
        0.75 * controls["c2b_probability"][~missing] + 0.25 * 0.60,
        rtol=0.0,
        atol=1e-12,
    )
    assert result["state"]["threshold"] == 0.50
    assert result["state"]["blend_weight_frozen_c2b"] == 0.75
    assert result["state"]["blend_weight_new_sparse"] == 0.25
    assert result["state"]["missing_stream_route"] == "frozen_cb_probability"


@pytest.mark.parametrize("bad_rows", [[205, 101, 502, 999], [101, 101, 502, 999]])
def test_route_rejects_permuted_or_duplicate_control_rows(monkeypatch, bad_rows):
    _install_fake_estimator(monkeypatch)
    fit_x = _features(["alpha anchor"] * 4 + ["beta anchor"] * 4)
    fit_y = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int8)
    score_x = _features(["alpha anchor", "beta anchor", None, ""], indexes=[101, 205, 502, 999])
    controls = _controls(score_x)
    controls["control_rows"] = np.asarray(bad_rows, dtype=np.int64)

    with pytest.raises(ValueError, match="route rows"):
        fit_predict(fit_x, fit_y, score_x, controls=controls, card_id=CARD_ID, cutoff=55)


def test_route_does_not_accept_string_ids_as_control_positions():
    score_x = _features(["alpha anchor", None], indexes=[100, 200])
    controls = _controls(score_x)
    controls["control_rows"] = np.array(["rec-A", "rec-B"], dtype=object)
    fit_x = _features(["alpha anchor"] * 4 + ["beta anchor"] * 4)
    fit_y = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int8)

    with pytest.raises((TypeError, ValueError)):
        fit_predict(fit_x, fit_y, score_x, controls=controls, card_id=CARD_ID, cutoff=55)
