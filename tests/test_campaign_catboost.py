import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

import src.models.campaign_catboost as campaign


def test_single_fixed_recipe_is_bounded_and_cpu_only():
    assert campaign.CATBOOST_PARAMETERS == {
        "loss_function": "MAE", "iterations": 500, "depth": 6,
        "learning_rate": 0.05, "random_seed": 42, "thread_count": 2,
        "task_type": "CPU", "verbose": False, "allow_writing_files": False,
    }


def test_catboost_feature_frame_encodes_native_categories_as_strings():
    values = np.array([[1.0, 10.0], [-1.0, 11.0]])
    frame = campaign._frame(values, ["org__ordinal", "x"], [0])
    assert frame["org__ordinal"].tolist() == ["1", "-1"]
    assert frame["x"].dtype.kind == "f"
    with pytest.raises(RuntimeError):
        campaign._frame(np.array([[.5]]), ["org__ordinal"], [0])


def test_feature_allowlist_rejects_unapproved_inputs():
    good = pd.DataFrame({name: [None] for name in campaign.HYBRID_FEATURES})
    selected = campaign._select_exact(good)
    assert tuple(selected.columns) == tuple(campaign.HYBRID_FEATURES)
    with pytest.raises(ValueError):
        campaign._select_exact(good.assign(record_id=[1]))


def test_model_requires_fit_before_predict():
    model = campaign.CampaignCatBoostRegressor()
    with pytest.raises(RuntimeError, match="fit must be called"):
        model.predict(pd.DataFrame())

