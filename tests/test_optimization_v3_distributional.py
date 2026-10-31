import numpy as np
import pytest
from tests.test_campaign_classification import _fixture
from src.models.optimization_v3_distributional import DistributionalRegressor,conditional_median

def test_cdf_median_interpolation_including_extremes():
    p=np.zeros((3,10));p[0,0]=1;p[1,9]=1;p[2,0]=.2;p[2,1]=.8
    np.testing.assert_allclose(conditional_median(p),[5.,95.,13.75],atol=1e-12,rtol=0)
    with pytest.raises(ValueError):conditional_median(p*.5)

def test_full_distribution_prediction_does_not_refit_vocabulary():
    frame,_=_fixture(200);y=np.tile(np.arange(10)*10+5,20).astype(float);model=DistributionalRegressor();fields=list(model.input_feature_names)
    model.fit(frame.iloc[:180][fields],y[:180]);before=model.audit_state()
    held=frame.iloc[180:][fields].copy();held['signal_stream']='futureneverobserved';p=model.predict_distribution(held)
    assert p.shape==(20,10)and np.isfinite(p).all()
    np.testing.assert_allclose(p.sum(axis=1),1,rtol=0,atol=1e-12)
    assert before==model.audit_state()and'futureneverobserved'not in model.vectorizer.vocabulary_
