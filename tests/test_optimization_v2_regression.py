from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.optimization_v2_regression import build_sparse_hybrid
from src.models.regression_research_text import HYBRID_FEATURES


def _synthetic(rows: int, offset: int = 0) -> pd.DataFrame:
    i = np.arange(offset, offset + rows)
    frame = pd.DataFrame({
        "impact_load": (i % 5).astype(float),
        "entry_friction": (i % 2).astype(float),
        "cycle_code": (47 + (i % 4)).astype(float),
        "scope_pressure": ((i % 7) / 7).astype(float),
        "text_load": (i % 11).astype(float),
        "scope_trace": ((i % 9) / 9).astype(float),
        "org_spread": (i % 3).astype(float),
        "module_spread": (i % 6).astype(float),
        "telemetry_flag": (i % 2).astype(float),
        "org_code": np.where(i % 2, "ORG_A", "ORG_B"),
        "module_code": np.where(i % 3, "MOD_A", "MOD_B"),
        "pattern_code": np.where(i % 4, "PAT_A", "PAT_B"),
        "signal_stream": ["alpha common phrase" if j % 2 else "beta common phrase" for j in i],
    })
    frame.loc[frame.index[3], "scope_trace"] = np.nan
    return frame.loc[:, list(HYBRID_FEATURES)]


@pytest.mark.parametrize("recipe", ["sgd_l1", "lgbm_sparse_l1"])
def test_sparse_hybrid_fit_predict_is_frozen_and_train_only(recipe: str) -> None:
    train = _synthetic(96)
    target = np.linspace(0.1, 99.9, len(train))
    model = build_sparse_hybrid(recipe).fit(train, target)

    state_before = model.audit_state()
    assert state_before["fit_rows"] == len(train)
    assert state_before["training_design_shape"][0] == len(train)
    assert state_before["sparse_design"] is True
    assert state_before["svd_used"] is False
    assert state_before["native_categorical_features_used"] is False

    heldout = _synthetic(4, offset=1000)
    heldout.loc[:, "org_code"] = "ORG_UNSEEN"
    heldout.loc[:, "signal_stream"] = "novel_token unseen_bigram"
    predictions = model.predict(heldout)

    assert predictions.shape == (len(heldout),)
    assert np.isfinite(predictions).all()
    assert model.audit_state() == state_before
    diagnostics = model.prediction_diagnostics()
    assert diagnostics["unseen_tfidf_token_count"] > 0
    assert diagnostics["unknown_category_rows"].get("org_code", 0) == len(heldout)


def test_sparse_hybrid_rejects_feature_schema_drift() -> None:
    train = _synthetic(64)
    target = np.linspace(1, 99, len(train))
    model = build_sparse_hybrid("sgd_l1").fit(train, target)
    with pytest.raises(ValueError, match="exactly the declared hybrid features"):
        model.predict(train.drop(columns=["signal_stream"]))


def test_sgd_recipe_is_exactly_the_declared_epsilon_zero_l1_configuration() -> None:
    train = _synthetic(64)
    target = np.linspace(1, 99, len(train))
    state = build_sparse_hybrid("sgd_l1").fit(train, target).audit_state()
    params = state["estimator"]["parameters"]
    assert params["loss"] == "epsilon_insensitive"
    assert params["epsilon"] == 0.0
    assert params["alpha"] == 1e-5
    assert params["early_stopping"] is False
    assert state["estimator"]["target_transform"] == "(threat_rank - 50) / 50"
