"""Reconcile actual completed Round1 outputs; no synthetic experiment scores."""
import json
import math
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from src.models.round1_evaluation import load_protocol, verify_freeze, window_indices, array_digest
from src.utils.paths import PROJECT_ROOT, task_files


class Round1ResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=PROJECT_ROOT/'reports/round1/results.json'
        if not path.exists():raise unittest.SkipTest('Actual Round1 results required.')
        cls.results=json.loads(path.read_text(encoding='utf-8'))
        cls.frame=pd.read_csv(task_files('regression')['train'])
        cls.protocol=load_protocol()
        cls.artifact=PROJECT_ROOT/'experiments/runs'/cls.results['run_id']
        if not cls.artifact.exists():raise unittest.SkipTest('Local ignored validation vectors required; rerun frozen experiments.')

    def test_freeze_and_all_64_vector_metrics(self):
        receipt=verify_freeze()
        self.assertEqual(receipt['signature'],self.results['freeze_signature'])
        self.assertEqual(len(self.results['jobs']),64)
        windows={w['id']:w for w in self.protocol['primary_windows']+self.protocol['secondary_windows']}
        for job in self.results['jobs']:
            with np.load(self.artifact/f"{job['window']}__{job['model']}.npz",allow_pickle=False) as a:
                rows,pred=a['validation_rows'],a['prediction']
            _,expected=window_indices(self.frame,windows[job['window']])
            np.testing.assert_array_equal(rows,expected)
            self.assertEqual(array_digest(pred),job['prediction_sha256'])
            error=pred-self.frame.iloc[rows].threat_rank.to_numpy()
            self.assertAlmostEqual(math.fsum(abs(float(v)) for v in error)/len(error),job['metrics']['mae'],places=11)
            self.assertAlmostEqual(math.fsum(float(v) for v in error)/len(error),job['metrics']['signed_error'],places=11)

    def test_primary_pooled_equal_cycle_and_subgroups_reconcile(self):
        for recipe in self.protocol['recipes']:
            jobs=[j for j in self.results['jobs'] if j['family']=='primary' and j['model']==recipe['id']]
            n=sum(j['metrics']['n'] for j in jobs)
            agg=self.results['aggregate']['primary'][recipe['id']]
            self.assertEqual(n,38588)
            self.assertAlmostEqual(sum(j['metrics']['n']*j['metrics']['mae'] for j in jobs)/n,agg['raw']['mae'],places=11)
            self.assertAlmostEqual(np.mean([c['mae'] for j in jobs for c in j['per_cycle']]),agg['equal_cycle_mae'],places=11)
            self.assertEqual(sum(agg['subgroups'][k]['n'] for k in ['EF0','EF1']),n)
            self.assertEqual(sum(agg['subgroups'][k]['n'] for k in ['region_low','region_middle','region_high']),n)
            self.assertLessEqual(agg['clipped']['mae'],agg['raw']['mae']+1e-12)

    def test_fit_audits_and_validation_label_boundary(self):
        for window in self.protocol['primary_windows']+self.protocol['secondary_windows']:
            audits=json.loads((PROJECT_ROOT/'reports/round1/fit_audits'/f"{window['id']}.json").read_text(encoding='utf-8'))
            fit,val=window_indices(self.frame,window)
            for state in audits.values():
                self.assertFalse(state['validation_labels_passed_to_fit'])
                self.assertFalse(state['early_stopping_validation_used'])
                self.assertEqual(state['fitted_state']['fit_rows'],len(fit))
                self.assertEqual(state['validation_rows'],len(val))
                self.assertNotIn('threat_rank',state['fitted_state']['input_feature_names'])

    def test_independent_behavior_and_reproduction_receipts(self):
        behavior=json.loads((PROJECT_ROOT/'reports/round1/behavior_diagnostics.json').read_text(encoding='utf-8'))
        self.assertTrue(all(v['matches'] for v in behavior['category_unknown_audit']))
        self.assertTrue(all(v['identical_predictions'] for v in behavior['same_fit_overlap_checks']))
        reproduction=json.loads((PROJECT_ROOT/'reports/round1/reproduction.json').read_text(encoding='utf-8'))
        self.assertEqual(reproduction['status'],'PASS')
        self.assertTrue(reproduction['all_prediction_vectors_bit_identical'])
        self.assertEqual(reproduction['repeat_fresh_fits'],64)


if __name__=='__main__':unittest.main()
