"""Ordered target statistics: every training text sees only earlier-cycle labels."""
import hashlib,json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class OrderedTokenStats:
    names=('token_posterior_mean','token_posterior_min','token_posterior_max','token_posterior_std',
           'token_posterior_q75','token_known_share','token_log_unique','token_log_mean_support')
    smoothing=30.
    def __init__(self):
        self.analyzer=TfidfVectorizer(ngram_range=(1,1),lowercase=True,strip_accents='unicode').build_analyzer()
        self.counts={};self.sums={};self.total=0;self.target_sum=0.;self.fitted=False
    def _tokens(self,text):
        if text is None or isinstance(text,float)and np.isnan(text):return set()
        return set(self.analyzer(str(text)))
    def _encode(self,tokens):
        prior=self.target_sum/self.total if self.total else .5
        eligible=[t for t in tokens if self.counts.get(t,0)>=3]
        n=np.array([self.counts[t]for t in eligible],float)
        values=np.array([(self.sums[t]+self.smoothing*prior)/(self.counts[t]+self.smoothing)for t in eligible])
        if len(values):return [values.mean(),values.min(),values.max(),values.std(),np.quantile(values,.75),len(eligible)/max(len(tokens),1),np.log1p(len(tokens)),np.log1p(n.mean())]
        return [prior,prior,prior,0.,prior,0.,np.log1p(len(tokens)),0.]
    def fit_transform(self,texts,cycles,target_unit):
        texts=list(texts);cycles=np.asarray(cycles);target=np.asarray(target_unit,float)
        if target.shape!=cycles.shape or target.shape!=(len(texts),)or not np.isfinite(target).all()or((target<0)|(target>1)).any():raise ValueError('Aligned cycle/text and normalized target[0,1] required')
        if not np.isfinite(cycles).all():raise ValueError('Finite training cycles required')
        self.counts={};self.sums={};self.total=0;self.target_sum=0.;self.fitted=False
        tokens=[self._tokens(t)for t in texts];features=np.empty((len(texts),len(self.names)));lineage=[]
        for cycle in np.sort(np.unique(cycles)):
            rows=np.flatnonzero(cycles==cycle)
            # All rows of this cycle are encoded before any of its labels enter.
            for row in rows:features[row]=self._encode(tokens[row])
            lineage.append({'cycle':float(cycle),'feature_rows':len(rows),'labels_previously_seen':self.total})
            for row in rows:
                for token in tokens[row]:self.counts[token]=self.counts.get(token,0)+1;self.sums[token]=self.sums.get(token,0.)+float(target[row])
            self.total+=len(rows);self.target_sum+=float(target[rows].sum())
        self.fitted=True;self.lineage=lineage
        self.train_feature_sha256=hashlib.sha256(np.asarray(features,dtype='<f8').tobytes()).hexdigest()
        self.dictionary_sha256=hashlib.sha256(json.dumps([(t,self.counts[t],self.sums[t])for t in sorted(self.counts)],separators=(',',':')).encode()).hexdigest()
        return features
    def transform(self,texts):
        if not self.fitted:raise RuntimeError('Fit ordered dictionary first')
        return np.array([self._encode(self._tokens(t))for t in texts],float)
    def audit_state(self):
        if not self.fitted:raise RuntimeError('Unfitted dictionary')
        return{'smoothing':self.smoothing,'min_token_document_support':3,'names':list(self.names),'fit_rows':self.total,
               'fit_target_sum':self.target_sum,'dictionary_size':len(self.counts),'dictionary_sha256':self.dictionary_sha256,
               'training_features_sha256':self.train_feature_sha256,'lineage':self.lineage,
               'policy':'training rows encode strictly earlier cycles; inference uses all fit labels only; no same-cycle self-label encoding'}
