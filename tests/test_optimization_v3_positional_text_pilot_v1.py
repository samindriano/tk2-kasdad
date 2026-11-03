"""Contract tests for the fixed POSITIONAL_TEXT_PILOT_V1 implementation."""
from __future__ import annotations

import unittest

from scripts import optimization_v3_positional_text_pilot_v1 as runner
from scripts import optimization_v3_positional_text_pilot_v1_synthetic as synthetic
from scripts.optimization_v3_positional_text_pilot_v1_synthetic import run_synthetic_contract
from src.models.positional_text_features import absolute_position_features, deterministic_order_shuffle
from src.models.optimization_v3_positional_text_pilot_v1 import availability_route


class PositionalPilotContractTests(unittest.TestCase):
    def test_token_thirds_and_deterministic_shuffle_preserve_bag_and_sizes(self):
        tokens = ["alpha", "beta", "alpha", "gamma", "delta", "beta", "epsilon", "zeta"]
        original = absolute_position_features(tokens)
        shuffled = deterministic_order_shuffle(tokens, seed=42)
        self.assertEqual(sorted(tokens), sorted(shuffled["tokens"]))
        self.assertEqual(original["token_count"], shuffled["token_count"])
        self.assertEqual(original["segment_sizes"], shuffled["segment_sizes"])
        self.assertEqual((0, 0, 0), absolute_position_features([])["segment_sizes"])

    def test_missing_stream_route_is_exact(self):
        present = [True, False, True]
        route = availability_route(present, [0.1, 0.2, 0.3], [0.4, 0.5, 0.6],
                                  [0.7, 0.8, 0.9], [0.9, 0.8, 0.7])
        self.assertEqual(route["bag"].tolist(), [0.1, 0.8, 0.3])
        self.assertEqual(route["position"].tolist(), [0.4, 0.8, 0.6])
        self.assertEqual(route["shuffled_position"].tolist(), [0.7, 0.8, 0.9])

    def test_synthetic_shared_fit_executor_persistence_resume_and_verifier(self):
        result = run_synthetic_contract()
        self.assertEqual(
            result["status"],
            "PASS_SYNTHETIC_END_TO_END_PERSISTENCE_ACCOUNTING_VERIFIER_REPORT",
        )
        self.assertFalse(result["official_data_opened"])
        self.assertTrue(result["shared_ledger_read_for_hash_only"])
        self.assertFalse(result["shared_ledger_parsed"])
        self.assertFalse(result["shared_ledger_mutated"])
        self.assertTrue(result["live_ledger_unchanged"])
        self.assertTrue(result["missing_explicit_path_rejected"])
        self.assertTrue(result["explicit_production_path_rejected"])
        self.assertEqual(result["official_writer_calls"], 0)
        self.assertEqual(result["fit_unit_count"], 18)
        self.assertEqual(result["resume_repeated_fit_count"], 0)
        self.assertTrue(result["tampered_prediction_rejected"])
        self.assertTrue(result["route_missing_probabilities_exact"])
        self.assertTrue(result["class_order_checked"])
        self.assertTrue(result["contains_empty_stream"])
        self.assertTrue(result["contains_unseen_score_tokens_and_categories"])
        self.assertTrue(all(item["status"] == "PASS_INDEPENDENT_PREDICTION_INTEGRITY"
                            for item in result["independent_prelabel_verification"]))
        for origin, fingerprint in result["shared_preprocessing_hashes"].items():
            self.assertEqual(len(fingerprint), 6, origin)
        self.assertEqual(result["completed_ledger_snapshot"]["fit_saved"], 18)
        self.assertEqual(result["completed_ledger_snapshot"]["fit_pending"], 0)

    def test_failure_injection_restores_global_paths_and_checks_live_hash_in_finally(self):
        names = (
            "ROOT", "TRAIN", "REPORT_ROOT", "RUN_ROOT", "LEDGER", "append_ledger",
            "read_ledger", "_append_official_ledger", "_fold_rows",
        )
        original = {name: getattr(runner, name) for name in names}
        with self.assertRaisesRegex(RuntimeError, "synthetic injected failure after E67"):
            synthetic.run_synthetic_contract(inject_failure=True)
        audit = synthetic._LAST_CLEANUP_AUDIT
        self.assertTrue(audit["live_ledger_unchanged"])
        self.assertTrue(audit["globals_restored"])
        self.assertTrue(audit["fit_functions_restored"])
        for name, value in original.items():
            self.assertIs(getattr(runner, name), value)


if __name__ == "__main__":
    unittest.main()
