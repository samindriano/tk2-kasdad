import numpy as np
import pytest
from src.models.optimization_v2_classification import KINDS,TemporalScoreHead

def test_heads_are_frozen_during_prediction_and_audit_training_only():
    p=np.column_stack([np.linspace(.01,.99,80),np.linspace(.02,.98,80)])
    y=(np.arange(80)>35).astype(int)
    for kind in KINDS:
        model=TemporalScoreHead(kind).fit(p,y);state=model.audit_state()
        q=model.predict(np.array([[0,1],[.3,.4],[.8,.9]]))
        assert np.isfinite(q).all() and ((q>=0)&(q<=1)).all()
        assert state==model.audit_state() and state['rows']==80
        assert len(state['coef'][0])==(2 if kind=='stack_base_nb' else 1)

def test_bad_scores_and_single_class_fail_closed():
    with pytest.raises(ValueError):TemporalScoreHead('calibrated_nb').fit([[.5,np.nan]],[1])
    with pytest.raises(ValueError):TemporalScoreHead('calibrated_nb').fit([[.5,.2]],[1])
