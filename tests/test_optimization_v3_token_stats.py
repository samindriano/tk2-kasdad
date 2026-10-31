import numpy as np
from src.models.optimization_v3_token_stats import OrderedTokenStats

def test_same_cycle_labels_never_enter_own_or_past_features():
    texts=['common alpha']*12;cycles=np.repeat([47,48,49],4);y=np.array([0.,0.,0.,1.,1.,1.,1.,1.,0.,0.,0.,0.])
    first=OrderedTokenStats();a=first.fit_transform(texts,cycles,y)
    changed=y.copy();changed[4:8]=0.;second=OrderedTokenStats();b=second.fit_transform(texts,cycles,changed)
    np.testing.assert_array_equal(a[:8],b[:8]);assert not np.array_equal(a[8:],b[8:])
    assert all(r['labels_previously_seen']==int((cycles<r['cycle']).sum())for r in first.audit_state()['lineage'])

def test_unknown_fallback_and_predict_state_immutability():
    model=OrderedTokenStats();model.fit_transform(['alpha']*6,[47,47,47,48,48,48],[1,1,0,0,1,0]);state=model.audit_state()
    out=model.transform(['brandnewtoken','alpha',''])
    assert np.isfinite(out).all()and model.audit_state()==state
    assert out[0,0]==.5 and out[0,5]==0 and model.counts.get('brandnewtoken')is None

def test_feature_digest_is_stable_across_python_hash_seeds():
    import os
    import subprocess
    import sys
    code = """
import hashlib
import numpy as np
from src.models.optimization_v3_token_stats import OrderedTokenStats
words=['tok'+str(i) for i in range(41)]
texts=[' '.join(w for j,w in enumerate(words) if (i+j)%7!=0) for i in range(84)]
cycles=np.repeat(np.arange(47,54),12)
y=np.array([(i*13%97)/97 for i in range(84)])
m=OrderedTokenStats();features=m.fit_transform(texts,cycles,y)
print(m.audit_state()['training_features_sha256'])
print(hashlib.sha256(m.transform([' '.join(words)]).tobytes()).hexdigest())
"""
    outputs = [subprocess.check_output([sys.executable, '-c', code],
               env={**os.environ, 'PYTHONHASHSEED': seed}, text=True)
               for seed in ('1', '17')]
    assert outputs[0] == outputs[1]
