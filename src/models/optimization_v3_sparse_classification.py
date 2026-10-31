"""One fixed nonlinear raw TF-IDF/structured classifier, with fit-only transforms."""
import hashlib,json
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from lightgbm import LGBMClassifier
from src.models.regression_research_text import RegressionTextModel,HYBRID_FEATURES,_array_digest
from src.models.classification_models import _TFIDF_PARAMS,_text_values
from src.models.round1_models import _fit_fingerprint

PARAMS={'objective':'binary','learning_rate':.05,'n_estimators':400,'num_leaves':15,
        'min_child_samples':50,'reg_lambda':3.,'reg_alpha':0.,'n_jobs':2,'verbosity':-1,
        'random_state':42,'deterministic':True,'force_col_wise':True}

def jsonhash(d):return hashlib.sha256(json.dumps(d,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()

class SparseTextBoostClassifier:
    input_feature_names=tuple(HYBRID_FEATURES)
    def _select(self,X):
        if X.columns.has_duplicates or set(X.columns)!=set(self.input_feature_names):raise ValueError('Exact feature allowlist required')
        return X.loc[:,list(self.input_feature_names)].copy()
    def fit(self,X,y):
        X=self._select(X);y=np.asarray(y)
        if y.shape!=(len(X),)or not np.isin(y,[0,1]).all()or len(np.unique(y))!=2:raise ValueError('Aligned binary fit target required')
        structured,self.proc,self.scaler,self.encoder,self.numeric_positions,structured_state=RegressionTextModel._structured_fit(X)
        docs,missing=_text_values(X.signal_stream);self.vectorizer=TfidfVectorizer(**dict(_TFIDF_PARAMS))
        text=self.vectorizer.fit_transform(docs).tocsr()
        design=sparse.hstack([structured,text,sparse.csr_matrix(missing.astype(float).reshape(-1,1))],format='csr')
        self.model=LGBMClassifier(**PARAMS).fit(design,y)
        ordered=[(str(token),int(index)) for token,index in sorted(self.vectorizer.vocabulary_.items(),key=lambda t:t[1])]
        self.state={'id':'V3_SPARSE_LGB_BINARY','fit_rows':len(y),'input_feature_names':list(self.input_feature_names),
            'fit_data_sha256':_fit_fingerprint(X,self.input_feature_names,y),'target_sha256':_array_digest(y),
            'parameters':PARAMS,'tfidf_params':{k:(list(v) if isinstance(v,tuple) else v.__name__ if isinstance(v,type) else v) for k,v in _TFIDF_PARAMS.items()},
            'vocabulary_sha256':jsonhash(ordered),'vocabulary_size':len(ordered),'idf_sha256':_array_digest(self.vectorizer.idf_),
            'structured_state':structured_state,'design_shape':list(design.shape),'design_data_sha256':_array_digest(design.data),
            'model_sha256':hashlib.sha256(self.model.booster_.model_to_string().encode()).hexdigest(),
            'eval_set_used':False,'early_stopping_used':False}
        self.state['state_sha256']=jsonhash(self.state);return self
    def predict_proba(self,X):
        X=self._select(X);structured,_=RegressionTextModel._structured_transform(X,self.proc,self.scaler,self.encoder,self.numeric_positions)
        docs,missing=_text_values(X.signal_stream);text=self.vectorizer.transform(docs)
        design=sparse.hstack([structured,text,sparse.csr_matrix(missing.astype(float).reshape(-1,1))],format='csr')
        p=self.model.predict_proba(design)[:,1]
        assert p.shape==(len(X),)and np.isfinite(p).all()and((p>=0)&(p<=1)).all();return p
    def audit_state(self):return json.loads(json.dumps(self.state))
