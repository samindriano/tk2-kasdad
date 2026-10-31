import numpy as np
import pytest
from tests.test_campaign_classification import _fixture
from src.models.optimization_v3_sparse_classification import SparseTextBoostClassifier

def test_unknown_text_categories_do_not_learn_during_predict():
    frame,y=_fixture(180);model=SparseTextBoostClassifier();fields=list(model.input_feature_names)
    model.fit(frame.iloc[:150][fields],y[:150]);state=model.audit_state()
    held=frame.iloc[150:][fields].copy();held['signal_stream']='unseenUniqueDoc';held['org_code']='futureorg'
    p=model.predict_proba(held);assert state==model.audit_state()
    assert 'futureorg' not in str(state)and len(p)==30 and np.isfinite(p).all()
    assert not {'record_id','artifact_signal','threat_rank'}.intersection(fields)
def test_extra_target_input_is_rejected():
    frame,y=_fixture(20)
    with pytest.raises(ValueError):SparseTextBoostClassifier().fit(frame,y)
