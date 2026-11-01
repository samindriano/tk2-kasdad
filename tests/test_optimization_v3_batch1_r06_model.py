from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src.models import optimization_v3_batch1_r06 as r06
from src.models.regression_research_text import HYBRID_FEATURES
from src.models.round1_models import CATEGORICAL_FEATURES


def _features(n: int, *, future_text: bool = False, unknown_category: bool = False) -> pd.DataFrame:
    rows = np.arange(n)
    values: dict[str, object] = {}
    for column in HYBRID_FEATURES:
        if column == "signal_stream":
            values[column] = [
                None if i % 19 == 0 else
                " " if i % 23 == 0 else
                "qzxjv rareprefix future-only fragment" if future_text and i >= n - 3 else
                f"alpha beta gamma sharedtoken fragment{i % 5}"
                for i in rows
            ]
        elif column in CATEGORICAL_FEATURES:
            values[column] = ["known" if i % 3 else "other" for i in rows]
        else:
            values[column] = (rows % 17).astype(float)
    values["scope_pressure"] = np.asarray(values["scope_pressure"], dtype=np.float64)
    values["scope_pressure"][::17] = np.nan
    if unknown_category:
        for column in CATEGORICAL_FEATURES:
            values[column][-2:] = ["new-category-a", "new-category-b"]
    return pd.DataFrame(values).loc[:, list(HYBRID_FEATURES)]


class _FakeBooster:
    def model_to_string(self, num_iteration: int = -1) -> str:
        return "synthetic-r06-booster"

    def current_iteration(self) -> int:
        return 5


class _FakeLGBM:
    fit_matrices: list[sparse.csr_matrix] = []
    fit_targets: list[np.ndarray] = []
    parameters_seen: list[dict[str, object]] = []

    def __init__(self, **parameters):
        self.parameters = parameters
        self.booster_ = _FakeBooster()

    def fit(self, matrix, target):
        assert sparse.isspmatrix_csr(matrix)
        type(self).fit_matrices.append(matrix.copy())
        type(self).fit_targets.append(np.asarray(target).copy())
        type(self).parameters_seen.append(dict(self.parameters))
        self.offset_ = float(np.mean(target))
        self.feature_importances_ = np.zeros(matrix.shape[1], dtype=np.int64)
        return self

    def predict(self, matrix):
        assert sparse.issparse(matrix)
        return self.offset_ + np.asarray(matrix.sum(axis=1)).reshape(-1) * 0.001


def _patch_fake_estimator(monkeypatch):
    _FakeLGBM.fit_matrices = []
    _FakeLGBM.fit_targets = []
    _FakeLGBM.parameters_seen = []
    monkeypatch.setattr(r06.lightgbm, "LGBMRegressor", _FakeLGBM)


def test_r06_uses_only_fit_learned_char_tfidf_and_sparse_l1(monkeypatch):
    _patch_fake_estimator(monkeypatch)
    fit_x = _features(60)
    score_x = _features(10, future_text=True, unknown_category=True)
    target = np.linspace(10.0, 90.0, len(fit_x))

    model = r06.R06CharacterSparseL1Hybrid().fit(fit_x, target)
    score = model.predict(score_x)
    state = model.audit_state()

    assert model.vectorizer is not None
    assert model.vectorizer.analyzer == "char_wb"
    assert model.vectorizer.ngram_range == (3, 5)
    assert model.vectorizer.min_df == 3
    assert model.vectorizer.max_features == 30_000
    assert model.vectorizer.sublinear_tf is True
    analyzer = model.vectorizer.build_analyzer()
    future_ngrams = analyzer("qzxjv rareprefix future-only fragment")
    assert any(token not in model.vectorizer.vocabulary_ for token in future_ngrams)
    assert state["char_tfidf"]["parameters"]["analyzer"] == "char_wb"
    assert state["char_tfidf"]["parameters"]["norm"] == "l2"
    assert state["word_tfidf_used"] is False
    assert state["svd_used"] is False
    assert state["validation_data_passed_to_fit"] is False
    assert state["validation_labels_passed_to_fit"] is False
    assert state["fit_rows"] == len(fit_x)
    assert state["missing_stream_fit_rows"] > 0
    assert state["training_design_shape"][0] == len(fit_x)
    assert state["char_tfidf_width"] > 0
    assert sparse.isspmatrix_csr(_FakeLGBM.fit_matrices[-1])
    assert _FakeLGBM.parameters_seen[-1] == r06.SPARSE_LGB_PARAMETERS
    assert r06.SPARSE_LGB_PARAMETERS["objective"] == "regression_l1"
    assert score.shape == (len(score_x),)
    assert np.isfinite(score).all()
    assert model.prediction_diagnostics()["signal_stream_missing_rows"] > 0
    assert model.prediction_diagnostics()["unknown_category_rows"]
    with pytest.raises(ValueError, match="exact feature allowlist"):
        model.predict(score_x.assign(record_id=[f"id-{i}" for i in range(len(score_x))]))


def test_r06_empty_text_training_falls_back_to_structured_sparse(monkeypatch):
    _patch_fake_estimator(monkeypatch)
    fit_x = _features(45)
    fit_x["signal_stream"] = [None if i % 2 else " " for i in range(len(fit_x))]
    model = r06.R06CharacterSparseL1Hybrid().fit(fit_x, np.linspace(5.0, 95.0, len(fit_x)))
    score_x = _features(8, unknown_category=True)
    raw = model.predict(score_x)
    state = model.audit_state()

    assert model.vectorizer is None
    assert state["char_tfidf_width"] == 0
    assert state["char_tfidf"]["fallback"] == "all_training_text_empty"
    assert np.isfinite(raw).all()


def test_rank50_uses_average_ties_within_each_complete_cycle():
    raw = np.array([10.0, 10.0, 30.0, 0.0, 10.0])
    cycles = np.array([7, 7, 7, 8, 8])
    expected = np.array([
        0.5 * 10 + 0.5 * (100 * 1.5 / 3),
        0.5 * 10 + 0.5 * (100 * 1.5 / 3),
        0.5 * 30 + 0.5 * 100,
        0.5 * 0 + 0.5 * 50,
        0.5 * 10 + 0.5 * 100,
    ])
    np.testing.assert_allclose(r06.rank50_prediction(raw, cycles), expected, rtol=0, atol=1e-12)


def test_fixed_r06_composition_uses_three_component_baseline_and_clips():
    raw = np.array([-50.0, 10.0, 30.0, 150.0])
    baseline = np.array([0.0, 25.0, 75.0, 100.0])
    cycles = np.array([70, 70, 70, 70])
    rank50 = r06.rank50_prediction(raw, cycles)
    expected = np.clip(0.75 * baseline + 0.25 * rank50, 0.0, 100.0)
    result = r06.compose_r06_prediction(baseline, raw, cycles)
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-12)
    assert np.all((result >= 0.0) & (result <= 100.0))


def test_fit_predict_routes_raw_candidate_through_frozen_three_component_baseline(monkeypatch):
    _patch_fake_estimator(monkeypatch)
    fit_x = _features(54)
    score_x = _features(12, future_text=True)
    fit_y = np.linspace(15.0, 85.0, len(fit_x))
    controls = {
        "baseline_prediction": np.linspace(2.0, 98.0, len(score_x)),
        "cycle_code": np.repeat([68, 69, 70], 4),
    }
    result = r06.fit_predict(
        fit_x, fit_y, score_x, controls=controls, card_id="R06", cutoff=67
    )

    raw = result["raw_candidate"]
    mapped = r06.rank50_prediction(raw, controls["cycle_code"])
    expected = np.clip(0.75 * controls["baseline_prediction"] + 0.25 * mapped, 0.0, 100.0)
    assert result["fit_predictor_count"] == 1
    assert result["fit_metadata"]["predictor_fit_calls"] == 1
    assert result["fit_metadata"]["validation_labels_passed_to_fit"] is False
    assert result["fit_metadata"]["test_data_used"] is False
    np.testing.assert_allclose(result["prediction"], expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(result["audit_arrays"]["baseline_prediction"], controls["baseline_prediction"])
    np.testing.assert_allclose(result["audit_arrays"]["candidate_rank50"], mapped)
    np.testing.assert_array_equal(result["audit_arrays"]["cycle_code"], controls["cycle_code"])
    assert np.isfinite(result["prediction"]).all()
    assert np.all((result["prediction"] >= 0.0) & (result["prediction"] <= 100.0))
    assert "record_id" not in r06.FEATURE_NAMES
    assert "threat_rank" not in r06.FEATURE_NAMES
    assert "artifact_signal" not in r06.FEATURE_NAMES


def test_r06_rank_and_composition_reject_bad_alignment_or_nonfinite_inputs():
    with pytest.raises(ValueError, match="aligned vectors"):
        r06.rank50_prediction([1.0, 2.0], [68])
    with pytest.raises(ValueError, match="finite"):
        r06.rank50_prediction([1.0, np.nan], [68, 68])
    with pytest.raises(ValueError, match=r"within \[0,100\]"):
        r06.compose_r06_prediction([101.0], [10.0], [68])
