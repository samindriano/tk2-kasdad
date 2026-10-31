import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

import src.models.campaign_regression as campaign


def test_cycle_percentiles_average_ties_within_each_cycle():
    result = campaign.cycle_percentiles([4, 4, 10, 7], [47, 47, 47, 48])
    np.testing.assert_allclose(result, [50, 50, 100, 100])


def test_cycle_rank_blend_and_prediction_blend_boundaries():
    mapped = campaign.apply_cycle_rank_blend([10, 90], [47, 47], 0.5)
    np.testing.assert_allclose(mapped, [30, 95])
    np.testing.assert_allclose(campaign.blend_predictions([20, 40], [80, 60], 0.25), [35, 45])
    with pytest.raises(ValueError):
        campaign.blend_predictions([1, 2], [1], 0.5)
    with pytest.raises(ValueError):
        campaign.apply_cycle_rank_blend([1], [47], 1.1)


def test_weight_selection_is_deterministic_and_ties_favor_lower_ridge_share():
    selected = campaign.select_blend_weight(
        [0, 100], [20, 80], [0, 100], weights=[0, .25, .5, 1]
    )
    assert selected.ridge_weight == 1.0
    assert selected.inner_mae == 0.0
    tied = campaign.select_blend_weight(
        [0, 100], [20, 80], [20, 80], weights=[1, .25, 0]
    )
    assert tied.ridge_weight == 0.0


class _DummyModel:
    feature_names = ("x",)

    def fit(self, X, y):
        self.fit_rows = len(X)
        self.y = np.asarray(y).copy()
        return self

    def predict(self, X):
        return X["x"].to_numpy(dtype=float)

    def audit_state(self):
        return {"fit_rows": self.fit_rows, "target_sum": float(self.y.sum())}


def test_campaign_model_fits_both_components_and_maps_prediction_batch(monkeypatch):
    monkeypatch.setattr(campaign, "build_lgb_text", lambda _: _DummyModel())
    monkeypatch.setattr(campaign, "build_text_model", lambda _: _DummyModel())
    model = campaign.RegressionCampaignBlend(lgb_params="base", ridge_weight=.25)
    with pytest.raises(RuntimeError):
        model.predict(pd.DataFrame({"x": [1]}), [47], rank_weight_lgb=.5, rank_weight_ridge=.75)
    X = pd.DataFrame({"x": [20., 80.]})
    model.fit(X, [10., 90.])
    lgb, ridge = model.predict_components(
        X, [47, 47], rank_weight_lgb=.5, rank_weight_ridge=.75
    )
    np.testing.assert_allclose(lgb, [35, 90])
    np.testing.assert_allclose(ridge, [42.5, 95])
    np.testing.assert_allclose(
        model.predict(X, [47, 47], rank_weight_lgb=.5, rank_weight_ridge=.75),
        campaign.blend_predictions(lgb, ridge, .25),
    )
    state = model.audit_state()
    assert state["fit_rows"] == 2
    assert state["ridge_weight"] == .25
    assert state["lightgbm"]["fit_rows"] == state["hybrid_ridge"]["fit_rows"] == 2

