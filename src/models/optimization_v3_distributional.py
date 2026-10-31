"""Fixed conditional-distribution regression from ten target bins, Train only."""
import hashlib,json
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from lightgbm import LGBMClassifier
from src.models.regression_research_text import RegressionTextModel,HYBRID_FEATURES,STRUCTURED_FEATURES,_SVD_PARAMS,_text_documents,_array_digest
from src.models.regression_round2_models import _SelectedPreprocessor
from src.models.round1_models import _fit_fingerprint

PARAMETERS={'objective':'multiclass','num_class':10,'learning_rate':.05,'n_estimators':250,
    'num_leaves':15,'min_child_samples':30,'reg_lambda':3.,'n_jobs':2,'random_state':42,
    'verbosity':-1,'deterministic':True,'force_col_wise':True}

def conditional_median(probabilities):
    p=np.asarray(probabilities,dtype=float)
    if p.ndim!=2 or p.shape[1]!=10 or not np.isfinite(p).all()or(p<0).any():raise ValueError('Ten finite nonnegative class masses required')
    if not np.allclose(p.sum(axis=1),1,atol=1e-8,rtol=0):raise ValueError('Class masses must sum to1')
    cdf=np.cumsum(p,axis=1);index=np.minimum((cdf<.5).sum(axis=1),9)
    below=np.where(index==0,0,cdf[np.arange(len(p)),np.maximum(index-1,0)])
    mass=p[np.arange(len(p)),index];assert(mass>0).all()
    return np.clip(10*(index+(.5-below)/mass),0,100)

class DistributionalRegressor:
    input_feature_names=tuple(HYBRID_FEATURES)
    def _select(self,X):
        if X.columns.has_duplicates or set(X.columns)!=set(self.input_feature_names):raise ValueError('Exact official feature allowlist required')
        return X.loc[:,list(self.input_feature_names)].copy()
    def fit(self,X,y):
        X=self._select(X);y=np.asarray(y,float)
        if y.shape!=(len(X),)or not np.isfinite(y).all()or((y<0)|(y>100)).any():raise ValueError('Finite target0..100 required')
        bins=np.minimum((y/10).astype(int),9)
        if len(np.unique(bins))!=10:raise ValueError('Allten bins required by fixed distributional recipe')
        self.proc=_SelectedPreprocessor(tuple(STRUCTURED_FEATURES));structured,cat=self.proc.fit_transform(X[list(STRUCTURED_FEATURES)])
        docs,missing=_text_documents(X.signal_stream);self.vectorizer,text,fallback=RegressionTextModel._fit_vectorizer(docs)
        k=min(32,text.shape[1]-1)if text.shape[1]>1 else 0;self.svd=None
        if k:self.svd=TruncatedSVD(n_components=k,**dict(_SVD_PARAMS));dense=self.svd.fit_transform(text)
        else:dense=np.empty((len(X),0))
        design=np.column_stack([structured,dense,missing.astype(float)])
        self.names=list(self.proc.transformed_feature_names)+[f'svd{i+1}'for i in range(k)]+['signal_stream_missing']
        self.cat=[i for i,b in enumerate(cat)if b]
        self.model=LGBMClassifier(**PARAMETERS).fit(pd.DataFrame(design,columns=self.names),bins,categorical_feature=self.cat)
        self.state={'id':'V3_DISTRIBUTIONAL_LGB10','fit_rows':len(y),'input_feature_names':list(self.input_feature_names),
            'fit_data_sha256':_fit_fingerprint(X,self.input_feature_names,y),'target_sha256':_array_digest(y),
            'bin_target_sha256':_array_digest(bins),'bin_counts':np.bincount(bins,minlength=10).tolist(),
            'parameters':PARAMETERS,'structured_state':self.proc.audit_state(),
            'tfidf_state':RegressionTextModel._text_state(self.vectorizer,fallback),
            'svd_components_sha256':_array_digest(self.svd.components_)if self.svd is not None else None,
            'svd_components':k,'design_shape':list(design.shape),'design_sha256':_array_digest(design),
            'model_sha256':hashlib.sha256(self.model.booster_.model_to_string().encode()).hexdigest(),
            'eval_set_used':False,'test_used':False,'prediction_rule':'Conditional median with linear CDF interpolation; uniform conditional density within each fixed ten-point bin is an approximation.'}
        return self
    def predict_distribution(self,X):
        X=self._select(X);structured,_=self.proc.transform(X[list(STRUCTURED_FEATURES)])
        docs,missing=_text_documents(X.signal_stream)
        text=self.vectorizer.transform(docs)if self.vectorizer is not None else None
        dense=self.svd.transform(text)if self.svd is not None else np.empty((len(X),0))
        design=np.column_stack([structured,dense,missing.astype(float)])
        return self.model.predict_proba(pd.DataFrame(design,columns=self.names))
    def predict(self,X):return conditional_median(self.predict_distribution(X))
    def audit_state(self):return json.loads(json.dumps(self.state))
