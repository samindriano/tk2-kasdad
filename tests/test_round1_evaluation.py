"""Metric and split contract tests; experiment outputs are tested separately."""
import unittest

import numpy as np
import pandas as pd

from src.models.round1_evaluation import load_protocol, metric_values, subgroup_values, window_indices, digest_json
from src.utils.paths import PROJECT_ROOT, task_files


class Round1EvaluationTests(unittest.TestCase):
    def test_metrics_independently_hand_calculated(self):
        # Software-only fixture, never reported as course experiment data.
        result=metric_values(np.array([0.,50.,100.]),np.array([10.,40.,80.]))
        self.assertAlmostEqual(result['mae'],40/3)
        self.assertAlmostEqual(result['signed_error'],-20/3)
        self.assertAlmostEqual(result['rmse'],np.sqrt(600/3))
        self.assertAlmostEqual(result['over_rate'],1/3)
        self.assertAlmostEqual(result['under_rate'],2/3)

    def test_invalid_predictions_rejected(self):
        with self.assertRaises(ValueError):metric_values([1,2],[1,np.nan])
        with self.assertRaises(ValueError):metric_values([1,2],[1])
        self.assertIsNone(metric_values([],[])['mae'])

    def test_primary_split_contract_on_actual_source(self):
        path=task_files('regression',PROJECT_ROOT/'data/raw')['train']
        if path is None:self.skipTest('Actual source needed.')
        frame=pd.read_csv(path);p=load_protocol();seen=set()
        for window in p['primary_windows']:
            fit,val=window_indices(frame,window)
            self.assertTrue(set(fit).isdisjoint(val))
            self.assertTrue(seen.isdisjoint(val));seen.update(val)
            self.assertLess(frame.iloc[fit].cycle_code.max(),frame.iloc[val].cycle_code.min())
            self.assertEqual(set(frame.iloc[val].cycle_code),set(range(window['validation'][0],window['validation'][1]+1)))
        self.assertEqual(len(seen),int(frame.cycle_code.between(56,79).sum()))

    def test_protocol_fixed_recipes_and_loss_control(self):
        p=load_protocol();models={r['id']:r for r in p['recipes']}
        self.assertEqual(len(models),8)
        self.assertEqual(models['R1_C']['params'],models['R1_D']['params'])
        self.assertFalse(p['hgb_fixed_parameters']['early_stopping'])
        self.assertEqual(p['minimum_lookup_support'],30)
        self.assertEqual(digest_json({'a':1,'b':2}),digest_json({'b':2,'a':1}))


if __name__=='__main__':unittest.main()
