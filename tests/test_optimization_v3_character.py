import numpy as np
from tests.test_campaign_classification import _fixture
from src.models.optimization_v3_character import CharacterRidge

def test_character_ridge_unseen_text_and_input_state_are_frozen():
    f,_=_fixture(180);y=np.arange(180)%100;model=CharacterRidge();fields=list(model.input_feature_names)
    model.fit(f.iloc[:150][fields],y[:150]);state=model.audit_state()
    held=f.iloc[150:][fields].copy();held['signal_stream']='uniqueneverobservedQzX'
    p=model.predict(held);assert len(p)==30 and np.isfinite(p).all()
    assert state==model.audit_state()and state['tfidf']['parameters']['analyzer']=='char_wb'
    assert not {'record_id','threat_rank','artifact_signal'}.intersection(fields)
