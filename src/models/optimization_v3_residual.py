"""Fixed shallow nonlinear score/covariate correction, trained on prior OOF rows."""
import hashlib,json
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from lightgbm import LGBMClassifier,LGBMRegressor
from src.models.regression_round2_models import _SelectedPreprocessor
from src.models.round1_models import EXPANDED_FEATURES,_fit_fingerprint

PARAMETERS={'learning_rate':.03,'n_estimators':200,'num_leaves':7,'min_child_samples':80,
    'reg_lambda':10.,'random_state':42,'n_jobs':2,'verbosity':-1,'deterministic':True,'force_col_wise':True}

def digest(a):return hashlib.sha256(np.asarray(a,dtype='<f8').tobytes()).hexdigest()

class OOFResidualHead:
    """Caller must provide genuinely past out-of-fit component predictions.

    This wrapper cannot certify score lineage; the runner and verifier must
    reconcile base-fit cutoff < score-row cycle <= correction-fit cutoff.
    """
    def __init__(self,task,score_names):
        if task not in ['classification','regression']or not score_names:raise ValueError('Known task and named scores required')
        if set(score_names)&set(EXPANDED_FEATURES):raise ValueError('Scores cannot shadow source feature names')
        self.task=task;self.score_names=tuple(score_names);self.input_feature_names=tuple(EXPANDED_FEATURES)
        self.shrinkage=.5;self.state=None
    def _design(self,X,scores,fit):
        if set(X.columns)!=set(self.input_feature_names)or X.columns.has_duplicates:raise ValueError('Exact structured allowlist required')
        scores=np.asarray(scores,float)
        if scores.shape!=(len(X),len(self.score_names))or not np.isfinite(scores).all():raise ValueError('Aligned finite score matrix required')
        if self.task=='classification'and((scores<0)|(scores>1)).any():raise ValueError('Binary probability inputs must lie in[0,1]')
        selected=X.loc[:,list(self.input_feature_names)].copy()
        if fit:
            self.proc=_SelectedPreprocessor(self.input_feature_names);structured,mask=self.proc.fit_transform(selected)
            self.cat_positions=[i for i,v in enumerate(mask)if v]
            self.names=list(self.proc.transformed_feature_names)
        else:structured,_=self.proc.transform(selected)
        score_features=logit(np.clip(scores,1e-6,1-1e-6))if self.task=='classification'else scores
        differences=score_features[:,1:]-score_features[:,[0]]
        design=np.column_stack([structured,score_features,differences])
        names=self.names+list(self.score_names)+[f'{name}_minus_first'for name in self.score_names[1:]]
        return pd.DataFrame(design,columns=names),selected,scores
    def fit(self,X,scores,y):
        design,selected,scores=self._design(X,scores,True);target=np.asarray(y,float)
        if target.shape!=(len(X),)or not np.isfinite(target).all():raise ValueError('Aligned finite head labels required')
        if self.task=='classification':
            if not np.isin(target,[0,1]).all()or len(np.unique(target))!=2:raise ValueError('Two binary classes required')
            self.model=LGBMClassifier(objective='binary',**PARAMETERS)
            # LightGBM learns correction trees relative to the OOF base margin.
            self.model.fit(design,target,init_score=logit(np.clip(scores[:,0],1e-6,1-1e-6)),categorical_feature=self.cat_positions)
        else:
            self.model=LGBMRegressor(objective='regression_l1',**PARAMETERS)
            self.model.fit(design,target-scores[:,0],categorical_feature=self.cat_positions)
        self.state={'task':self.task,'parameters':PARAMETERS,'shrinkage':self.shrinkage,'rows':len(X),
            'input_features':list(self.input_feature_names),'score_names':list(self.score_names),
            'fit_data_sha256':_fit_fingerprint(selected,self.input_feature_names,target),
            'fit_scores_sha256':digest(scores),'target_sha256':digest(target),'preprocessor':self.proc.audit_state(),
            'booster_sha256':hashlib.sha256(self.model.booster_.model_to_string().encode()).hexdigest(),
            'no_evalset_or_early_stopping':True}
        return self
    def predict(self,X,scores):
        if self.state is None:raise RuntimeError('Fit head on past OOF scores first')
        design,_,s=self._design(X,scores,False)
        correction=self.model.booster_.predict(design,raw_score=True,num_threads=2)
        if self.task=='classification':return expit(logit(np.clip(s[:,0],1e-6,1-1e-6))+.5*correction)
        return np.clip(s[:,0]+.5*correction,0,100)
    def audit_state(self):
        if self.state is None:raise RuntimeError('Unfitted head')
        return json.loads(json.dumps(self.state))
