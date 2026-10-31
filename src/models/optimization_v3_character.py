"""Fixed character-only text channel over the existing hybrid Ridge recipe."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from src.models.regression_research_text import RegressionTextModel,_json_digest,_array_digest

CHAR_PARAMETERS={'analyzer':'char_wb','ngram_range':(3,5),'min_df':3,'max_features':30000,
    'sublinear_tf':True,'lowercase':True,'strip_accents':'unicode','dtype':np.float64}

class CharacterRidge(RegressionTextModel):
    def __init__(self):
        super().__init__('hybrid_ridge');self.model_id='V3_CHAR_HYBRID_RIDGE'
    @staticmethod
    def _fit_vectorizer(docs):
        vectorizer=TfidfVectorizer(**dict(CHAR_PARAMETERS))
        matrix=vectorizer.fit_transform(docs).tocsr()
        return vectorizer,matrix,None
    @staticmethod
    def _text_state(vectorizer,fallback):
        ordered=[(str(token),int(index))for token,index in sorted(vectorizer.vocabulary_.items(),key=lambda v:v[1])]
        return{'parameters':{k:(list(v)if isinstance(v,tuple)else v.__name__ if isinstance(v,type)else v)for k,v in CHAR_PARAMETERS.items()},
               'vocabulary_size':len(ordered),'vocabulary_sha256':_json_digest(ordered),'idf_sha256':_array_digest(vectorizer.idf_),
               'fallback':fallback,'word_tokens_used':False}
