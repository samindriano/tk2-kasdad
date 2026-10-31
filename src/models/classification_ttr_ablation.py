"""Fixed TTR/length additions over unchanged Round1 LR representations."""
from __future__ import annotations
import re
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from src.models.classification_models import ClassificationModel, _array_digest, _json_digest, _text_values
from src.models.round1_models import EXPANDED_FEATURES

TOKEN_RE=re.compile(r'[A-Za-z0-9]+')
VARIANTS={
    'S_AV':('C1_A',('missing',)),
    'S_LEN':('C1_A',('count',)),
    'S_TTR':('C1_A',('ttr',)),
    'S_BOTH':('C1_A',('ttr','count')),
    'H_LEN':('C2_B',('count',)),
    'H_TTR':('C2_B',('ttr',)),
    'H_BOTH':('C2_B',('ttr','count'))}


def text_features(stream):
    """EDA tokenization: undefined TTR is NaN; missing length is unobserved."""
    docs,missing=_text_values(stream)
    counts=[];ratios=[]
    for doc in docs:
        tokens=TOKEN_RE.findall(doc.lower());n=len(tokens)
        counts.append(n);ratios.append(len(set(tokens))/n if n else np.nan)
    counts=np.asarray(counts,dtype=float);counts[missing]=np.nan
    return pd.DataFrame({'ttr':ratios,'count':counts,'missing':missing.astype(float)},dtype=float)


class DiversityClassifier(ClassificationModel):
    def __init__(self,variant):
        if variant not in VARIANTS:raise ValueError('Unknown fixed diversity recipe')
        source,features=VARIANTS[variant]
        super().__init__({'id':source})
        self.model_id=variant;self.source_id=source;self.extra_features=features
        # Include source stream in the fit fingerprint even for structured variants.
        self.input_feature_names=[*EXPANDED_FEATURES,'signal_stream']

    def fit(self,X,y):
        self._pending=text_features(self._select(X).signal_stream).loc[:,list(self.extra_features)].to_numpy()
        try:
            super().fit(X,y)
        finally:self._pending=None
        self._state['derived_features']={
            'names':list(self.extra_features),'tokenization':'[A-Za-z0-9]+ lowercase',
            'undefined_ttr':'fit-only median imputation; no new indicator',
            'imputer_median':self._extra_imputer.statistics_.tolist(),
            'scaling':'StandardScaler fit on fit rows only',
            'mean':self._extra_scaler.mean_.tolist(),'scale':self._extra_scaler.scale_.tolist(),
            'fit_matrix_sha256':self._extra_fit_hash,'base_width':self._base_width,
            'base_design_sha256':self._base_design_hash,'train_targets_used_for_features':False}
        self._state['source_recipe']=self.source_id
        self._state['model_state_sha256']=_json_digest({k:v for k,v in self._state.items()if k!='model_state_sha256'})
        return self

    def _fit_logistic(self,matrix,target):
        base=sparse.csr_matrix(matrix,dtype=np.float64)
        self._base_width=base.shape[1]
        self._base_design_hash=_json_digest({'shape':list(base.shape),
            'data':_array_digest(base.data),'indices':_array_digest(base.indices),
            'indptr':_array_digest(base.indptr)})
        self._extra_scaler=StandardScaler()
        self._extra_imputer=SimpleImputer(strategy='median',keep_empty_features=True)
        extra=self._extra_scaler.fit_transform(self._extra_imputer.fit_transform(self._pending))
        self._extra_fit_hash=_array_digest(extra)
        super()._fit_logistic(sparse.hstack((base,sparse.csr_matrix(extra)),format='csr'),target)

    def predict_proba(self,X):
        if not self._fitted:raise RuntimeError('Fit before prediction')
        selected=self._select(X)
        self._diagnostics={'n_predictions':len(selected)}
        matrix=self._transform_structured(selected)
        if self.kind=='lr_hybrid':
            text=self._transform_text(selected.signal_stream)
            _,missing=_text_values(selected.signal_stream)
            matrix=sparse.hstack((matrix,text,sparse.csr_matrix(missing.astype(float).reshape(-1,1))),format='csr')
        assert matrix.shape[1]==self._base_width
        raw=text_features(selected.signal_stream).loc[:,list(self.extra_features)].to_numpy()
        extra=self._extra_scaler.transform(self._extra_imputer.transform(raw))
        design=sparse.hstack((matrix,sparse.csr_matrix(extra)),format='csr')
        p=np.asarray(self._estimator.predict_proba(design)[:,1],float)
        assert p.shape==(len(selected),) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        return p


def build_variant(variant):return DiversityClassifier(variant)
