from __future__ import annotations

import unittest
import hashlib
import tempfile
from pathlib import Path

import numpy as np
from scipy import sparse

from scripts.diagnose_cls_pos_frequency_ridge_mean_one_energy_r1 import (
    canonical_json,
    partitioned_delta_logits,
    verify_freeze_signature,
    verify_pinned_file,
)


class TestMeanOneEnergyDiagnostic(unittest.TestCase):
    def test_freeze_body_tampering_rejected_by_canonical_signature(self):
        body = {"schema_version": 1, "rule": "fixed"}
        frozen = {**body, "freeze_sha256": hashlib.sha256(canonical_json(body)).hexdigest()}
        self.assertEqual(verify_freeze_signature(frozen), frozen["freeze_sha256"])
        tampered = {**frozen, "rule": "changed"}
        with self.assertRaisesRegex(RuntimeError, "self-signature mismatch"):
            verify_freeze_signature(tampered)

    def test_raw_input_byte_tampering_rejected_by_pinned_file_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "pinned.json"
            path.write_bytes(b'{"value":1}')
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(verify_pinned_file(path, expected), expected)
            path.write_bytes(b'{ "value":1}')
            with self.assertRaisesRegex(RuntimeError, "Raw file SHA256 differs"):
                verify_pinned_file(path, expected)

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
