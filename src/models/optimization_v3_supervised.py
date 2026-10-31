"""Ordered supervised text summaries plus native structured boosting."""
import hashlib,json
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier,LGBMRegressor
from src.models.optimization_v3_token_stats import OrderedTokenStats
from src.models.regression_round2_models import _SelectedPreprocessor
from src.models.regression_research_text import HYBRID_FEATURES,STRUCTURED_FEATURES
from src.models.classification_models import _text_values
from src.models.round1_models import _fit_fingerprint

PARAMETERS={'n_estimators':400,'num_leaves':15,'min_child_samples':40,'reg_lambda':3.,
    'learning_rate':.05,'random_state':42,'n_jobs':2,'verbosity':-1,'force_col_wise':True,'deterministic':True}

class OrderedStatsBoost:
    input_feature_names=tuple(HYBRID_FEATURES)
    def __init__(self,task):
        if task not in['classification','regression']:raise ValueError('Known task required')
        self.task=task
    def _select(self,X):
        if X.columns.has_duplicates or set(X.columns)!=set(self.input_feature_names):raise ValueError('Exact supervised feature allowlist required')
        return X.loc[:,list(self.input_feature_names)].copy()
    def fit(self,X,y):
        X=self._select(X);y=np.asarray(y,float)
        if y.shape!=(len(X),)or not np.isfinite(y).all():raise ValueError('Aligned finite target required')
        if self.task=='classification'and(not np.isin(y,[0,1]).all()or len(np.unique(y))!=2):raise ValueError('Two binary classes required')
        unit=y if self.task=='classification'else y/100
        self.proc=_SelectedPreprocessor(tuple(STRUCTURED_FEATURES));structured,cat=self.proc.fit_transform(X[list(STRUCTURED_FEATURES)])
        docs,missing=_text_values(X.signal_stream);self.tokens=OrderedTokenStats()
        stats=self.tokens.fit_transform(docs,X.cycle_code.to_numpy(),unit)
        self.names=list(self.proc.transformed_feature_names)+list(self.tokens.names)+['stream_missing']
        design=pd.DataFrame(np.column_stack([structured,stats,missing.astype(float)]),columns=self.names)
        estimator=LGBMClassifier(objective='binary',**PARAMETERS)if self.task=='classification'else LGBMRegressor(objective='regression_l1',**PARAMETERS)
        estimator.fit(design,y,categorical_feature=[i for i,c in enumerate(cat)if c]);self.model=estimator
        self.state={'task':self.task,'fit_rows':len(y),'fit_data_sha256':_fit_fingerprint(X,self.input_feature_names,y),
            'input_feature_names':list(self.input_feature_names),'parameters':PARAMETERS,
            'structured':self.proc.audit_state(),'ordered_tokens':self.tokens.audit_state(),
            'design_shape':list(design.shape),'design_sha256':hashlib.sha256(np.asarray(design,dtype='<f8').tobytes()).hexdigest(),
            'model_sha256':hashlib.sha256(estimator.booster_.model_to_string().encode()).hexdigest(),
            'all_supervised_training_token_features_strictly_earlier_cycles':True,'no_evalset_or_early_stopping':True}
        return self
    def _design(self,X):
        X=self._select(X);structured,_=self.proc.transform(X[list(STRUCTURED_FEATURES)]);docs,missing=_text_values(X.signal_stream)
        stats=self.tokens.transform(docs)
        return pd.DataFrame(np.column_stack([structured,stats,missing.astype(float)]),columns=self.names)
    def predict_proba(self,X):
        if self.task!='classification':raise ValueError('Onlybinaryestimator')
        return self.model.predict_proba(self._design(X))[:,1]
    def predict(self,X):
        if self.task!='regression':raise ValueError('Onlyregressionestimator')
        return self.model.predict(self._design(X))
    def audit_state(self):return json.loads(json.dumps(self.state))
