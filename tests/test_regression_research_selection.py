import unittest
from scripts.research_regression_models import select_inner

class SelectionTests(unittest.TestCase):
    def test_past_inner_min_and_tie_rule(self):
        records=[{'param':'base','rank_weight':0.,'mae':9.},
                 {'param':'regularized','rank_weight':.5,'mae':8.},
                 {'param':'deeper','rank_weight':.25,'mae':8.},
                 {'param':'regularized','rank_weight':.25,'mae':8.+1e-11}]
        self.assertEqual(select_inner(records)['param'],'regularized')
        self.assertEqual(select_inner(records)['rank_weight'],.25)

if __name__=='__main__':unittest.main()
