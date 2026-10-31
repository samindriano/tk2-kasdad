"""Feature semantics and fit-only augmentation over frozen representations."""
import unittest
import numpy as np
import pandas as pd
from tests.test_classification_models import fixture_rows
from src.models.classification_ttr_ablation import text_features,build_variant
from src.models.classification_models import build_classifier

class TTRTests(unittest.TestCase):
    def test_eda_tokenization_missing_and_repeat_identity(self):
        f=text_features(pd.Series(['Aa aa B_2',None,'  ','!!!','x x x y']))
        self.assertAlmostEqual(f.ttr.iloc[0],.75)
        self.assertEqual(f['count'].iloc[0],4)
        self.assertTrue(f.ttr.iloc[1:4].isna().all())
        self.assertTrue(f['count'].iloc[1:3].isna().all())
        self.assertEqual(f['count'].iloc[3],0)
        self.assertEqual(f.missing.tolist(),[0,1,1,0,0])
        self.assertEqual(1-f.ttr.iloc[4],(4-2)/4)

    def test_fit_only_features_and_frozen_base_blocks(self):
        X,y=fixture_rows();X.loc[:100,'signal_stream']='a a b'
        for name,source in [('S_BOTH','C1_A'),('H_BOTH','C2_B')]:
            base=build_classifier({'id':source}).fit(X.iloc[:150],y[:150])
            model=build_variant(name).fit(X.iloc[:150],y[:150])
            state=model.audit_state()
            raw=text_features(X.iloc[:150].signal_stream)[['ttr','count']].to_numpy()
            np.testing.assert_allclose(model._extra_imputer.statistics_,np.nanmedian(raw,axis=0))
            for block in ['preprocessor','scaler','text']:
                if block in base.audit_state():self.assertEqual(state[block],base.audit_state()[block])
            held=X.iloc[150:].copy();held['signal_stream']='unseen '*500
            p=model.predict_proba(held)
            self.assertEqual(state,model.audit_state())
            self.assertTrue(np.isfinite(p).all())
            dirty=held.assign(threat_rank=999,artifact_signal=1)
            np.testing.assert_array_equal(p,model.predict_proba(dirty))

if __name__=='__main__':unittest.main()
