from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models import optimization_v3_batch1_r07 as r07
from src.models.regression_research_text import HYBRID_FEATURES
from src.models.round1_models import CATEGORICAL_FEATURES


def _features(n: int, *, future_text: bool = False) -> pd.DataFrame:
    rows = np.arange(n)
    values: dict[str, object] = {}
    for column in HYBRID_FEATURES:
        if column == "signal_stream":
            values[column] = [
                None if i % 23 == 0 else
                " " if i % 29 == 0 else
                "futuretoken qqqfutureonly" if future_text and i >= n - 3 else
                f"common token{i % 40} token{(i + 1) % 40} token{(i + 2) % 40} token{(i + 3) % 40}"
                for i in rows
            ]
        elif column in CATEGORICAL_FEATURES:
            values[column] = ["known" if i % 3 else "other" for i in rows]
        else:
            values[column] = (rows % 17).astype(float)
    if "scope_pressure" in values:
        values["scope_pressure"] = np.asarray(values["scope_pressure"], dtype=np.float64)
        values["scope_pressure"][::17] = np.nan
    return pd.DataFrame(values).loc[:, list(HYBRID_FEATURES)]


class _FakeBooster:
    def dump_model(self):
        return {"tree_info": [{"tree_structure": {"leaf_value": 1.0}}]}


class _FakeLGBM:
    instances: list["_FakeLGBM"] = []

    def __init__(self, **parameters):
        self.parameters = dict(parameters)
        self.booster_ = _FakeBooster()
        self.fit_frame = None
        self.fit_target = None
        self.categorical_feature = None
        type(self).instances.append(self)

    def fit(self, frame, target, *, categorical_feature):
        assert isinstance(frame, pd.DataFrame)
        self.fit_frame = frame.copy()
        self.fit_target = np.asarray(target, dtype=np.float64).copy()
        self.categorical_feature = list(categorical_feature)
        self.offset_ = float(np.mean(target))
        return self

    def predict(self, frame):
        assert isinstance(frame, pd.DataFrame)
        return self.offset_ + frame.to_numpy(dtype=np.float64).sum(axis=1) * 1e-4


def _patch_fake_lgbm(monkeypatch):
    _FakeLGBM.instances = []
    monkeypatch.setattr(r07, "LGBMRegressor", _FakeLGBM)


def _short_nmf_for_synthetic(monkeypatch):
    monkeypatch.setitem(r07.NMF_PARAMETERS, "max_iter", 20)


def test_r07_fit_uses_only_fit_tfidf_and_nmf_then_keeps_score_rows_aligned(monkeypatch):
    _patch_fake_lgbm(monkeypatch)
    _short_nmf_for_synthetic(monkeypatch)
    fit_x = _features(96)
    score_x = _features(12, future_text=True)
    target = np.linspace(5.0, 95.0, len(fit_x))

    model = r07.NMF32StructuredLGB().fit(fit_x, target)
    state_before = model.audit_state()
    raw = model.predict(score_x)
    state_after = model.audit_state()

    assert model.vectorizer is not None
    assert model.nmf is not None
    for name, value in r07.TFIDF_PARAMETERS.items():
        assert getattr(model.vectorizer, name) == value
    assert model.nmf.n_components == 32
    assert model.nmf.init == "nndsvda"
    assert model.nmf.solver == "cd"
    assert model.nmf.beta_loss == "frobenius"
    assert model.nmf.max_iter == 20
    assert model.nmf.random_state == 42
    assert "futuretoken" not in model.vectorizer.vocabulary_
    assert state_before["nmf"]["components_sha256"] == state_after["nmf"]["components_sha256"]
    assert state_before["tfidf"]["vocabulary_sha256"] == state_after["tfidf"]["vocabulary_sha256"]
    assert state_after["nmf"]["n_components_actual"] == 32
    assert state_after["nmf"]["factors_nonnegative"] is True
    assert state_after["nmf"]["n_iter_"] is not None
    assert state_after["nmf"]["reconstruction_err_"] is not None
    assert isinstance(state_after["nmf"]["converged"], bool)
    assert state_after["nmf"]["fit_convergence_warning_count"] == state_after["nmf"]["fit_convergence_warnings"]["count"]
    assert state_after["nmf"]["score_transform_convergence_warning_count"] == state_after["nmf"]["score_transform_convergence_warnings"]["count"]
    assert state_after["tfidf"]["parameters"]["analyzer"] == "word"
    assert state_after["tfidf"]["parameters"]["norm"] == "l2"
    assert state_after["tfidf"]["parameters"]["use_idf"] is True
    assert state_after["validation_data_passed_to_fit"] is False
    assert state_after["validation_labels_passed_to_fit"] is False
    assert state_after["test_data_used_for_fit"] is False
    assert state_after["fit_rows"] == len(fit_x)
    assert state_after["training_design_shape"][0] == len(fit_x)
    assert raw.shape == (len(score_x),)
    assert np.isfinite(raw).all()
    assert _FakeLGBM.instances[-1].fit_frame.shape[0] == len(fit_x)
    assert _FakeLGBM.instances[-1].fit_target.shape == (len(fit_x),)
    assert _FakeLGBM.instances[-1].parameters == r07.LIGHTGBM_PARAMETERS
    assert _FakeLGBM.instances[-1].categorical_feature == model.categorical_positions
    with pytest.raises(ValueError, match="exactly the declared features"):
        model.predict(score_x.assign(record_id=[f"id-{i}" for i in range(len(score_x))]))


@pytest.mark.parametrize(
    ("streams", "expected_fallback"),
    [
        ([None, " ", None, "  "] * 12, "all_training_text_empty"),
        (["x", "?", "x", "? "] * 12, "empty_vocabulary_after_filters"),
    ],
)
def test_r07_empty_text_fallback_keeps_structured_predictor(monkeypatch, streams, expected_fallback):
    _patch_fake_lgbm(monkeypatch)
    fit_x = _features(len(streams))
    fit_x["signal_stream"] = streams
    model = r07.NMF32StructuredLGB().fit(fit_x, np.linspace(0.0, 100.0, len(fit_x)))
    score = model.predict(_features(9))
    state = model.audit_state()

    assert model.vectorizer is None
    assert model.nmf is None
    assert state["nmf"]["fallback"] == expected_fallback
    assert state["nmf"]["n_components_actual"] == 0
    assert state["training_design_shape"][0] == len(fit_x)
    assert score.shape == (9,)
    assert np.isfinite(score).all()


def test_r07_fewer_than_32_tfidf_features_uses_explicit_structured_fallback(monkeypatch):
    _patch_fake_lgbm(monkeypatch)
    fit_x = _features(48)
    fit_x["signal_stream"] = ["one repeated token"] * len(fit_x)
    model = r07.NMF32StructuredLGB().fit(fit_x, np.linspace(1.0, 99.0, len(fit_x)))
    state = model.audit_state()

    assert model.vectorizer is not None
    assert model.nmf is None
    assert state["nmf"]["fallback"] == "vocabulary_smaller_than_32_components"
    assert state["nmf"]["n_components_actual"] == 0


def test_rank50_uses_average_ties_within_cycle_and_composition_clips():
    raw = np.array([-50.0, 10.0, 10.0, 30.0, 150.0, 10.0])
    cycles = np.array([7, 7, 7, 7, 7, 8])
    expected_rank50 = np.array([
        0.0,
        0.5 * 10.0 + 0.5 * (100.0 * 2.5 / 5),
        0.5 * 10.0 + 0.5 * (100.0 * 2.5 / 5),
        0.5 * 30.0 + 0.5 * (100.0 * 4.0 / 5),
        100.0,
        0.5 * 10.0 + 0.5 * 100.0,
    ])
    mapped = r07.rank50_prediction(raw, cycles)
    np.testing.assert_allclose(mapped, expected_rank50, rtol=0, atol=1e-12)
    baseline = np.array([0.0, 0.0, 25.0, 75.0, 100.0, 100.0])
    expected = np.clip(0.75 * baseline + 0.25 * mapped, 0.0, 100.0)
    result = r07.compose_r07_prediction(baseline, raw, cycles)
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-12)
    assert np.all((result >= 0.0) & (result <= 100.0))


def test_fit_predict_returns_fixed_champion_blend_and_audit_arrays(monkeypatch):
    _patch_fake_lgbm(monkeypatch)
    _short_nmf_for_synthetic(monkeypatch)
    fit_x = _features(72)
    score_x = _features(12, future_text=True)
    fit_y = np.linspace(10.0, 90.0, len(fit_x))
    controls = {
        "baseline_prediction": np.linspace(0.0, 100.0, len(score_x)),
        "cycle_code": np.repeat([68, 69, 70], 4),
    }
    result = r07.fit_predict(
        fit_x, fit_y, score_x, controls=controls, card_id="R07", cutoff=67
    )

    expected = r07.compose_r07_prediction(
        controls["baseline_prediction"], result["raw_candidate"], controls["cycle_code"]
    )
    np.testing.assert_allclose(result["prediction"], expected, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(result["audit_arrays"]["baseline_prediction"], controls["baseline_prediction"])
    np.testing.assert_array_equal(result["audit_arrays"]["cycle_code"], controls["cycle_code"])
    assert result["fit_predictor_count"] == 1
    assert result["fit_metadata"]["nmf_fit_calls"] == 1
    assert result["fit_metadata"]["predictor_fit_calls"] == 1
    assert result["fit_metadata"]["validation_labels_passed_to_fit"] is False
    assert result["fit_metadata"]["test_data_used"] is False
    assert result["state"]["nmf"]["parameters"]["n_components"] == 32
    assert "record_id" not in r07.FEATURE_NAMES
    assert "threat_rank" not in r07.FEATURE_NAMES
    assert "artifact_signal" not in r07.FEATURE_NAMES


def test_r07_rejects_invalid_rank_mapping_inputs():
    with pytest.raises(ValueError, match="aligned vectors"):
        r07.rank50_prediction([1.0, 2.0], [68])
    with pytest.raises(ValueError, match="finite"):
        r07.rank50_prediction([1.0, np.nan], [68, 68])
    with pytest.raises(ValueError, match=r"within \[0,100\]"):
        r07.compose_r07_prediction([101.0], [10.0], [68])
