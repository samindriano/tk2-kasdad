import numpy as np
from scipy.special import expit,logit
from tests.test_campaign_classification import _fixture
from src.models.optimization_v3_residual import OOFResidualHead

def test_binary_init_margin_added_once_and_no_predict_learning():
    frame,y=_fixture(180);model=OOFResidualHead('classification',['base','structured'])
    fields=list(model.input_feature_names);scores=np.column_stack([np.linspace(.1,.9,180),np.linspace(.2,.8,180)])
    model.fit(frame.iloc[:150][fields],scores[:150],y[:150]);state=model.audit_state()
    held=frame.iloc[150:][fields];p=model.predict(held,scores[150:]);design,_,_=model._design(held,scores[150:],False)
    correction=model.model.booster_.predict(design,raw_score=True,num_threads=2)
    np.testing.assert_allclose(p,expit(logit(scores[150:,0])+.5*correction),rtol=0,atol=1e-12)
    assert state==model.audit_state()and np.isfinite(p).all()

def test_regression_correction_uses_original_base_and_clip():
    frame,y=_fixture(180);model=OOFResidualHead('regression',['incumbent','other'])
    fields=list(model.input_feature_names);scores=np.column_stack([np.linspace(10,90,180),np.linspace(5,95,180)])
    target=scores[:,0]+3
    model.fit(frame.iloc[:150][fields],scores[:150],target[:150]);state=model.audit_state()
    pred=model.predict(frame.iloc[150:][fields],scores[150:])
    np.testing.assert_allclose(pred,scores[150:,0]+1.5,atol=1e-6,rtol=0)
    assert state==model.audit_state()
