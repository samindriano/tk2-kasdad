"""Verify real fit/transform boundaries and category handling for the fixed expert."""
import numpy as np
import pytest
from tests.test_campaign_classification import _fixture
from src.models.campaign_classification_catboost import build_catboost_classifier

def test_fit_transform_unseen_categories_and_no_state_learning():
    frame,y=_fixture(180); model=build_catboost_classifier();fields=model.input_feature_names
    assert not {'record_id','artifact_signal','threat_rank','signal_stream'}.intersection(fields)
    model.fit(frame.iloc[:150][fields],y[:150]); state=model.audit_state()
    held=frame.iloc[150:][fields].copy();held['org_code']='unseen-heldout'
    held.loc[held.index[0],'module_code']=None
    p=model.predict_proba(held);q=model.predict_proba(held)
    np.testing.assert_array_equal(p,q)
    assert p.shape==(30,) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    assert model.audit_state()==state
    assert 'unseen-heldout' not in str(state['fit_only_preprocessor'])
    assert state['fit_rows']==150 and not state['eval_set_used'] and not state['early_stopping_used']

def test_invalid_targets_fail_before_fitting():
    frame,y=_fixture(20);model=build_catboost_classifier()
    with pytest.raises(ValueError):model.fit(frame[model.input_feature_names],np.full(20,2))
