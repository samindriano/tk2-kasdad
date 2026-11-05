from __future__ import annotations

import unittest

import numpy as np

from scripts import verify_optimization_v3_cls_pos_frequency_ridge_eval_r9_adapter as adapter


class TestR9ArrayHashAdapter(unittest.TestCase):
    def test_documented_hash_vectors(self):
        vectors = [
            (np.asarray([1, -2, 300], dtype=np.int64),
             "94343c4b99d3f1606699baa1c6f64ec7ce5e1ae5394906cd0bdee973003153e8"),
            (np.asarray([0.25, -3.5], dtype=np.float64),
             "471d600c4f63c54ac7c10731a235078ab14de46b7ec2d03fc51d57c55d802d19"),
            (np.asarray([True, False, True], dtype=np.bool_),
             "c5f98c5b2cbe9d76c034815fa9f015798bdadb4742db83f1898e4ca7be681ae8"),
            (np.asarray(["café", "漏洞", "α"], dtype="U4"),
             "0c7b25557fd37061fd917f18d7461c49f06dd3d69a903028d90c9a6f3250f09d"),
            (np.asarray(["id-a", "id-b"], dtype=object),
             "29a0b8ed87c41ebfbd752bcc31010d3c25ece8c131e76524625716ec87d4ca28"),
            (np.asarray([], dtype=np.int32),
             "72eb9c83a039b46bdda6399106e6df6ef9a5f58214b300ec2738e3ed624e9e03"),
            (np.arange(24, dtype=np.int16)[::3],
             "eb82eedeba2cf9e786a88b2cb6b164eb7b71b17b1b609aa9d1f59fe4859dd19e"),
        ]
        for value, expected in vectors:
            with self.subTest(dtype=str(value.dtype), shape=value.shape):
                self.assertEqual(adapter.documented_array_sha256(value), expected)

    def test_dtype_order_and_value_are_identity(self):
        ids = np.asarray(["r1", "r2", "r3"], dtype="U2")
        digest = adapter.documented_array_sha256(ids)
        changed = (
            ids.astype(object), ids[::-1],
            np.asarray(["r1", "r2", "r4"], dtype="U2"),
        )
        for value in changed:
            with self.subTest(dtype=str(value.dtype), values=value.tolist()):
                with self.assertRaises(adapter.BASE.VerifyError):
                    adapter.verify_array_digest(value, digest, "synthetic")

    def test_actual_28_prediction_arrays_and_gamma_receipt(self):
        result = adapter.run_self_tests(verify_actual_archive=True)
        self.assertEqual(result["status"], "PASS_R9_ARRAY_SCHEMA_SELF_TESTS")
        self.assertEqual(result["actual_prediction_arrays"], 28)
        self.assertTrue(result["actual_array_hashes_pass"])
        self.assertEqual(result["target_keys_present"], [])
        self.assertTrue(result["known_gamma_receipt_checked"])


if __name__ == "__main__":
    unittest.main()
