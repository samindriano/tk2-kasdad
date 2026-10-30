import unittest
import numpy as np
from src.models.regression_research_common import percentile_scores,mapped_scores

class RankTests(unittest.TestCase):
    def test_ties_and_cycles(self):
        np.testing.assert_allclose(percentile_scores([2,2,8,1,9],[1,1,1,2,2]),[50,50,100,50,100])
    def test_mapping_uses_only_predictions_and_cycle(self):
        p=np.array([-10.,20.,120.]);c=[1,1,1]
        np.testing.assert_allclose(mapped_scores(p,c,.5),np.clip(.5*p+.5*np.array([100/3,200/3,100]),0,100))
        np.testing.assert_array_equal(mapped_scores(p,c,0),[0,20,100])
    def test_invalid_and_order(self):
        with self.assertRaises(ValueError):percentile_scores([1,np.nan],[1,1])
        with self.assertRaises(ValueError):mapped_scores([1],[1],2)
        np.testing.assert_array_equal(percentile_scores([3,1,2],[1,1,1]),[100,100/3,200/3])

if __name__=='__main__':unittest.main()
