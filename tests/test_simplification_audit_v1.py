import unittest
import numpy as np
from scripts.simplification_audit_v1 import align,mapped,cm,close

class AuditTests(unittest.TestCase):
    def test_identity_join_reorders_and_rejects_duplicates_missing(self):
        np.testing.assert_array_equal(align(['b','a'],np.array(['a','b']),[2,3])[0],[3,2])
        with self.assertRaises(ValueError):align(['a'],['a','a'],[2,3])
        with self.assertRaises(KeyError):align(['z'],['a'],[2])
    def test_rank_complete_cycles_ties_and_unclipped_auxiliary(self):
        x=np.array([10,10,90,200.]);c=np.array([1,1,1,2])
        np.testing.assert_allclose(mapped(x,c),[30,30,95,100])
        self.assertEqual(mapped(x,c,clip=False)[-1],150)
    def test_macro_metric_independent_and_rare_class(self):
        self.assertEqual(cm(np.array([0,1]),np.array([.2,.8]),.5)['macro_f1'],1)
        self.assertEqual(cm(np.array([0,0]),np.array([.2,.1]),.5)['macro_f1'],.5)
    def test_nested_weights_cannot_cross_clipping(self):
        rng=np.random.default_rng(42);p=rng.uniform(0,100,(100,5))
        legacy=.5625*p[:,0]+.1875*p[:,1]+.25*p[:,2]
        r06=.75*legacy+.25*p[:,3]
        close(.75*r06+.25*p[:,4],p@np.array([.31640625,.10546875,.140625,.1875,.25]))
        self.assertNotEqual(np.clip(.75*np.clip(-100,0,100)+.25*100,0,100),np.clip(.75*-100+.25*100,0,100))

if __name__=='__main__':unittest.main()
