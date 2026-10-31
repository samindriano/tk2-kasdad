"""Small deterministic contract checks; do not fit a model or inspect Test."""
import unittest

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from scripts.optimization_v3_reg_residual import _current_components, _stack_blocks, _summary


class RegressionResidualAssemblyTests(unittest.TestCase):
    def test_current_components_reconstructs_fixed_rank50_blend_on_exact_rows(self):
        cycles = np.repeat(np.arange(56, 68), [2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3])
        n = len(cycles)
        frame = pd.DataFrame({"cycle_code": cycles, "threat_rank": np.linspace(1, 99, n)})
        rows = np.arange(n, dtype=np.int64)
        raw_sparse = np.linspace(3, 98, n, dtype=float)
        lgb = np.linspace(5, 95, n, dtype=float)
        ridge = np.linspace(8, 92, n, dtype=float)
        base = {"rows_55": rows,
                "lgb_55": lgb,
                "ridge_55": ridge,
                "blend_ridge25_55": np.clip(.75 * lgb + .25 * ridge, 0, 100)}
        pilot = {"rows_lgbm_sparse_l1_55": rows, "raw_lgbm_sparse_l1_55": raw_sparse}
        full = {}

        block = _current_components(frame, base, pilot, full, 55, 56, 59)
        expected_rows = np.flatnonzero((cycles >= 56) & (cycles <= 59))
        np.testing.assert_array_equal(block["rows"], expected_rows)
        mapped = np.empty(len(rows), dtype=float)
        for cycle in np.unique(cycles):
            mask = cycles == cycle
            mapped[mask] = .5 * raw_sparse[mask] + .5 * 100 * rankdata(raw_sparse[mask], method="average") / mask.sum()
        np.testing.assert_allclose(block["lgb_sparse"], np.clip(mapped[expected_rows], 0, 100), rtol=0, atol=1e-12)
        np.testing.assert_allclose(block["champion"], np.clip(
            .75 * base["blend_ridge25_55"][expected_rows] + .25 * mapped[expected_rows], 0, 100),
            rtol=0, atol=1e-12)

    def test_stack_sorts_by_official_source_position(self):
        result = _stack_blocks([
            {"rows": np.array([4, 5]), "score": np.array([40.0, 50.0])},
            {"rows": np.array([0, 1]), "score": np.array([0.0, 10.0])},
        ])
        np.testing.assert_array_equal(result["rows"], [0, 1, 4, 5])
        np.testing.assert_array_equal(result["score"], [0, 10, 40, 50])

    def test_summary_uses_prediction_minus_target_and_absolute_error(self):
        y = np.array([10.0, 20.0, 30.0])
        p = np.array([12.0, 18.0, 30.0])
        result = _summary(y, p)
        self.assertAlmostEqual(result["mae"], 4.0 / 3.0)
        self.assertAlmostEqual(result["bias_prediction_minus_target"], 0.0)
        self.assertAlmostEqual(result["median_signed_residual"], 0.0)
        self.assertAlmostEqual(result["over_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(result["under_rate"], 1.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
