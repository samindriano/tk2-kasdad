import numpy as np
from tests.test_campaign_classification import _fixture
from src.models.optimization_v3_supervised import OrderedStatsBoost

def test_supervised_encoder_boundary_and_heldout_state_are_explicit():
    f,y=_fixture(180);model=OrderedStatsBoost('classification');fields=list(model.input_feature_names)
    model.fit(f.iloc[:150][fields],y[:150]);state=model.audit_state()
    held=f.iloc[150:][fields].copy();held['signal_stream']='uniqueNeverSeenFuture'
    p=model.predict_proba(held)
    assert np.isfinite(p).all()and((p>=0)&(p<=1)).all()and model.audit_state()==state
    assert'uniqueneverseenfuture'not in model.tokens.counts
    cycle=f.iloc[:150].cycle_code.to_numpy()
    assert all(v['labels_previously_seen']==int((cycle<v['cycle']).sum())for v in state['ordered_tokens']['lineage'])
