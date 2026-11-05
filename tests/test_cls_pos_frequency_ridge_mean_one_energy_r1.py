from __future__ import annotations

import unittest

import numpy as np
from scipy import sparse

from scripts.diagnose_cls_pos_frequency_ridge_mean_one_energy_r1 import partitioned_delta_logits


class TestMeanOneEnergyDiagnostic(unittest.TestCase):
    def test_exact_weight_cut_and_delta_reconstruction(self):
        token_weights = np.asarray([0.5, 1.0, 1.75], dtype=np.float64)
        low_weight = np.tile(token_weights < 1.0, 1)
        self.assertEqual(low_weight.tolist(), [True, False, False])

        x = sparse.csr_matrix(np.asarray([
            [1.0, 2.0, 0.0],
            [0.0, -1.0, 3.0],
            [2.0, 0.5, -0.5],
        ]))
        delta_gamma = np.asarray([0.25, -0.5, 0.75], dtype=np.float64)
        result = partitioned_delta_logits(x, delta_gamma, low_weight)
        self.assertTrue(result["reconstruction"]["allclose_atol_1e-10_rtol_1e-10"])
        self.assertLessEqual(result["reconstruction"]["max_abs_error"], 1e-10)
        shares = result["magnitude_share"]
        self.assertAlmostEqual(shares["w_below_1"] + shares["w_at_least_1"], 1.0)
        self.assertEqual(result["w_below_1"]["n_rows"], 3)
        self.assertEqual(result["w_at_least_1"]["n_rows"], 3)


if __name__ == "__main__":
    unittest.main()
