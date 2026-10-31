"""One coverage scalar over the unchanged frozen SVD32 LightGBM design.

OOV means outside the fitted TF-IDF feature vocabulary, including filtered or
truncated terms; it does not imply semantic novelty or previously unseen words.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from lightgbm import LGBMRegressor
from src.models.regression_research_lgb_text import RegressionResearchLGBText,_select_exact
from src.models.regression_research_text import (RegressionTextModel,STRUCTURED_FEATURES,
    _text_documents,_target,_array_digest,_json_digest,_SVD_PARAMS,_TFIDF_PARAMS)
from src.models.regression_round2_models import _SelectedPreprocessor
from src.models.regression_research_tabular import _tree_summary
from src.models.round1_models import _fit_fingerprint

def vocabulary_coverage(vectorizer,stream):
    docs,missing=_text_documents(stream)
    # Use the exact analyzer of the frozen vectorizer (word unigrams+bigrams).
    analyzer=vectorizer.build_analyzer()if vectorizer is not None else TfidfVectorizer(**_TFIDF_PARAMS).build_analyzer()
    vocab=vectorizer.vocabulary_ if vectorizer is not None else {}
    count=[];unknown=[]
    for doc in docs:
        terms=analyzer(doc);count.append(len(terms));unknown.append(sum(t not in vocab for t in terms))
    count=np.asarray(count,int);unknown=np.asarray(unknown,int)
    fraction=np.divide(unknown,count,out=np.zeros(len(count),float),where=count>0)
    return pd.DataFrame({'analyzer_terms':count,'outside_vocabulary_terms':unknown,
        'oov_fraction':fraction,'stream_missing':missing})

class CoverageLGBText(RegressionResearchLGBText):
    """No parameter search: add exactly one bounded, target-blind scalar."""
    def fit(self,X,y):
        self._fitted=False;selected=_select_exact(X);target=_target(y,len(selected))
        self._fit_rows=len(target);self._fit_fingerprint=_fit_fingerprint(selected,self.input_feature_names,target)
        self._target_sha256=_array_digest(target)
        self._preprocessor=_SelectedPreprocessor(tuple(STRUCTURED_FEATURES))
        structured,mask=self._preprocessor.fit_transform(selected.loc[:,list(STRUCTURED_FEATURES)])
        self._categorical_positions=[i for i,b in enumerate(mask)if b]
        docs,missing=_text_documents(selected.signal_stream)
        self._vectorizer,text_matrix,fallback=RegressionTextModel._fit_vectorizer(docs)
        self._svd_components_actual=min(32,text_matrix.shape[1]-1)if text_matrix.shape[1]>1 else 0
        if self._svd_components_actual:
            self._svd=TruncatedSVD(n_components=self._svd_components_actual,**dict(_SVD_PARAMS))
            text_dense=self._svd.fit_transform(text_matrix)
        else:self._svd=None;text_dense=np.empty((len(docs),0))
        extra=vocabulary_coverage(self._vectorizer,selected.signal_stream).oov_fraction.to_numpy()
        design=np.column_stack((structured,text_dense,missing.astype(float),extra))
        assert np.isfinite(design).all()
        self._transformed_feature_names=(list(self._preprocessor.transformed_feature_names)
            +[f'signal_stream_svd_{i+1}'for i in range(self._svd_components_actual)]
            +['signal_stream_missing','signal_stream_oov_fraction'])
        self._estimator=LGBMRegressor(**dict(self.actual_parameters))
        self._estimator.fit(pd.DataFrame(design,columns=self._transformed_feature_names),target,categorical_feature=self._categorical_positions)
        self._tree_audit=_tree_summary(self._estimator.booster_)
        self._training_design_sha256=_array_digest(design);self._training_design_shape=list(design.shape)
        self._fit_missing_stream_rows=int(missing.sum());self._vectorizer_state=RegressionTextModel._text_state(self._vectorizer,fallback)
        self._structured_mask=list(mask);self._state=self._make_state()
        self._state['id']='LIGHTGBM_SVD32_COVERAGE_'+self.params_id.upper()
        self._state['coverage_feature']={'definition':'fraction of analyzer unigram/bigram occurrences absent from fit-only TFIDF feature vocabulary',
            'empty_or_missing_fraction':0.,'missing_indicator_unchanged':True,'target_used':False,
            'fit_fraction_sha256':_array_digest(extra),'fit_mean':float(extra.mean())}
        self._state['state_sha256']=_json_digest({k:v for k,v in self._state.items()if k!='state_sha256'})
        self._fitted=True;self._diagnostics={'n_predictions':0};return self

    def predict(self,X):
        if not self._fitted:raise RuntimeError('Fit before predict')
        selected=_select_exact(X);structured,unknown=self._preprocessor.transform(selected.loc[:,list(STRUCTURED_FEATURES)])
        docs,missing=_text_documents(selected.signal_stream)
        if self._vectorizer is None:text_dense=np.empty((len(docs),0))
        else:
            matrix=self._vectorizer.transform(docs)
            text_dense=self._svd.transform(matrix)if self._svd is not None else np.empty((len(docs),0))
        extra=vocabulary_coverage(self._vectorizer,selected.signal_stream).oov_fraction.to_numpy()
        design=np.column_stack((structured,text_dense,missing.astype(float),extra))
        pred=np.asarray(self._estimator.predict(pd.DataFrame(design,columns=self._transformed_feature_names)),float)
        assert pred.shape==(len(selected),)and np.isfinite(pred).all()
        self._diagnostics={'n_predictions':len(pred),'unknown_categories':unknown};return pred
