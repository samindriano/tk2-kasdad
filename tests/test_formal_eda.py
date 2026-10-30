"""Independent reconciliation against actual source rows; no modeling fixtures."""
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from scipy.stats import kstest

from src.utils.paths import PROJECT_ROOT, task_files


class FormalEdaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = task_files('regression', PROJECT_ROOT/'data/raw')['train']
        if path is None:
            raise unittest.SkipTest('Real official regression data required.')
        cls.frame = pd.read_csv(path)
        cls.t = json.loads((PROJECT_ROOT/'reports/eda/temporal_robustness.json').read_text(encoding='utf-8'))
        cls.c = json.loads((PROJECT_ROOT/'reports/eda/composition_robustness.json').read_text(encoding='utf-8'))
        mapping = {v:f'period_{i+1}' for i,chunk in enumerate(np.array_split(sorted(cls.frame.cycle_code.unique()),3)) for v in chunk}
        cls.frame['period'] = cls.frame.cycle_code.map(mapping)

    def test_all_cycles_correlations_and_two_sided_ecdf(self):
        self.assertEqual(len(self.t['cycles']), self.frame.cycle_code.nunique())
        for row in self.t['cycles']:
            group = self.frame[self.frame.cycle_code.eq(row['cycle_code'])]
            self.assertEqual(row['n'], len(group))
            self.assertAlmostEqual(row['pearson'], np.corrcoef(group.impact_load,group.threat_rank)[0,1], places=12)
            self.assertAlmostEqual(row['uniform_ks_two_sided_distance'], kstest(group.threat_rank/100,'uniform').statistic, places=12)
            self.assertLess(abs(row['target_mean_minus_rank_implied_mean']), 5e-5)

    def test_common_support_selected_from_counts_only(self):
        counts = self.frame.groupby(['impact_load','entry_friction','period'], observed=True).size().unstack(fill_value=0)
        expected = set(counts[(counts>=30).all(axis=1)].index)
        recorded = {(d['impact_load'],d['entry_friction']) for d in self.c['support_selection']['common_impact_x_friction_cells']}
        self.assertEqual(expected, recorded)
        self.assertEqual(len(expected), 33)

    def test_fixed_distribution_correlation_from_cell_moments(self):
        frame = self.frame
        counts = frame.groupby(['impact_load','entry_friction','period'], observed=True).size().unstack(fill_value=0)
        counts = counts[(counts>=30).all(axis=1)]
        q = counts.sum(axis=1)/counts.to_numpy().sum()
        for period in counts.columns:
            ex = ey = ex2 = ey2 = exy = 0.0
            for (impact,ef), weight in q.items():
                y = frame.loc[frame.period.eq(period)&frame.impact_load.eq(impact)&frame.entry_friction.eq(ef),'threat_rank'].to_numpy()
                ex += weight*impact
                ex2 += weight*impact**2
                ey += weight*y.mean()
                ey2 += weight*np.mean(y*y)
                exy += weight*impact*y.mean()
            corr = (exy-ex*ey)/np.sqrt((ex2-ex*ex)*(ey2-ey*ey))
            self.assertAlmostEqual(corr,self.c['correlations']['common_impact_ef_distribution_weighted'][period]['pearson_r'],places=11)

    def test_symmetric_mean_decomposition_reconciles_raw_population(self):
        frame = self.frame
        cells = self.c['support_selection']['common_impact_x_friction_cells']
        records = []
        for cell in cells:
            sub = frame[frame.impact_load.eq(cell['impact_load'])&frame.entry_friction.eq(cell['entry_friction'])]
            records.append((sub[sub.period.eq('period_1')].threat_rank,sub[sub.period.eq('period_3')].threat_rank))
        ne,nl = sum(len(e) for e,l in records),sum(len(l) for e,l in records)
        composition = sum((len(l)/nl-len(e)/ne)*(e.mean()+l.mean())/2 for e,l in records)
        within = sum((l.mean()-e.mean())*(len(e)/ne+len(l)/nl)/2 for e,l in records)
        delta = sum(l.sum() for e,l in records)/nl-sum(e.sum() for e,l in records)/ne
        saved = self.c['mean_decomposition']['symmetric_decomposition']
        self.assertAlmostEqual(composition,saved['composition_effect'],places=11)
        self.assertAlmostEqual(within,saved['within_stratum_effect'],places=11)
        self.assertAlmostEqual(delta,composition+within,places=11)

    def test_formal_notebook_outputs_valid_and_review_boundary(self):
        import nbformat
        path = PROJECT_ROOT/'notebooks/samuel/03_samuel_formal_eda.ipynb'
        notebook = nbformat.read(path,4)
        nbformat.validate(notebook)
        outputs = [o for cell in notebook.cells for o in cell.get('outputs',[])]
        self.assertFalse(any(o.output_type=='error' for o in outputs))
        self.assertEqual(sum('image/png' in o.get('data',{}) for o in outputs),3)
        summary = json.loads((PROJECT_ROOT/'reports/eda/formal_eda_summary.json').read_text(encoding='utf-8'))
        self.assertTrue(summary['question_selected_by_user'])
        self.assertIn('REVIEW_REQUIRED_BEFORE_MODELING',summary['status'])


if __name__ == '__main__':
    unittest.main()
