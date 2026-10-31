"""Scoped coverage-feature tests against frozen SVD32 source pipeline."""
import unittest
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sklearn.feature_extraction.text import TfidfVectorizer
from tests.test_classification_models import fixture_rows
from src.models.impact_oov import CoverageLGBText,vocabulary_coverage
from src.models.regression_research_lgb_text import build_lgb_text

class CoverageTests(unittest.TestCase):
    def test_analyzer_occurrence_semantics_and_missing(self):
        v=TfidfVectorizer(ngram_range=(1,2)).fit(['alpha beta','alpha gamma'])
        q=vocabulary_coverage(v,pd.Series(['alpha beta','newword newword',None,'  ','!!!']))
        self.assertEqual(q.analyzer_terms.tolist(),[3,3,0,0,0])
        self.assertEqual(q.oov_fraction.tolist(),[0,1,0,0,0])
        self.assertEqual(q.stream_missing.tolist(),[False,False,True,True,False])
        self.assertEqual(vocabulary_coverage(None,pd.Series(['alpha beta'])).oov_fraction.iloc[0],1)

    def test_same_base_transforms_and_heldout_state_immutability(self):
        frame,_=fixture_rows();names=list(CoverageLGBText().feature_names)
        X=frame[names];y=(np.arange(len(X))%100).astype(float)
        with threadpool_limits(limits=2):
            base=build_lgb_text().fit(X.iloc[:150],y[:150])
            model=CoverageLGBText().fit(X.iloc[:150],y[:150])
            b,s=base.audit_state(),model.audit_state()
            for key in ['structured_preprocessing','tfidf','svd','actual_hyperparameters','fit_data_sha256','target_sha256']:
                self.assertEqual(b[key],s[key],key)
            self.assertEqual(s['training_design_shape'][1],b['training_design_shape'][1]+1)
            held=X.iloc[150:].copy();held['signal_stream']='nevertrained unseennew'
            pred=model.predict(held)
        self.assertTrue(np.isfinite(pred).all());self.assertEqual(s,model.audit_state())
        self.assertNotIn('nevertrained',model._vectorizer.vocabulary_)
        self.assertFalse({'artifact_signal','threat_rank','record_id'}.intersection(names))

if __name__=='__main__':unittest.main()
