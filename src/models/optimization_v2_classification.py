"""Small score heads fitted on explicitly past-only, out-of-fit predictions."""
import hashlib,json
import numpy as np
from sklearn.linear_model import LogisticRegression

KINDS=('calibrated_base','calibrated_nb','stack_base_nb')
PARAMETERS={'C':1.0,'solver':'lbfgs','max_iter':1000,'tol':1e-8,'random_state':42}

def digest(values):
    return hashlib.sha256(np.asarray(values,dtype='<f8').tobytes()).hexdigest()

class TemporalScoreHead:
    def __init__(self,kind):
        if kind not in KINDS:raise ValueError('Unknown fixed score head')
        self.kind=kind;self._state=None

    def _design(self,probabilities):
        p=np.asarray(probabilities,dtype=float)
        if p.ndim!=2 or p.shape[1]!=2 or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
            raise ValueError('Expected finite base/NB probability columns in [0,1]')
        p=np.clip(p,1e-6,1-1e-6);logits=np.log(p/(1-p))
        return logits[:,[0] if self.kind=='calibrated_base' else [1] if self.kind=='calibrated_nb' else [0,1]]

    def fit(self,probabilities,y):
        design=self._design(probabilities);target=np.asarray(y)
        if target.shape!=(len(design),) or not np.isin(target,[0,1]).all() or len(np.unique(target))!=2:
            raise ValueError('Aligned two-class calibration labels required')
        self.estimator=LogisticRegression(**PARAMETERS).fit(design,target)
        if self.estimator.n_iter_[0]>=PARAMETERS['max_iter']:raise RuntimeError('Score head failed to converge')
        self._state={'kind':self.kind,'parameters':PARAMETERS,'rows':len(target),
                     'design_sha256':digest(design),'target_sha256':digest(target),
                     'coef':self.estimator.coef_.tolist(),'intercept':self.estimator.intercept_.tolist(),
                     'classes':self.estimator.classes_.tolist(),'n_iter':self.estimator.n_iter_.tolist()}
        return self

    def predict(self,probabilities):
        if self._state is None:raise RuntimeError('Fit past-only head first')
        return self.estimator.predict_proba(self._design(probabilities))[:,1]

    def audit_state(self):
        if self._state is None:raise RuntimeError('Unfitted score head')
        return json.loads(json.dumps(self._state))
