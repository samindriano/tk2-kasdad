from __future__ import annotations

import tempfile
import unittest
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import reg_pos_poisson_bag2_v1 as pilot


class TestRegPosPoissonBag2V1(unittest.TestCase):
    def test_freeze_signer_excludes_stale_signature_field(self):
        signed = pilot._sign_freeze({"experiment_id": pilot.CARD_ID,
                                     "freeze_signature": "stale-signature"})
        self.assertEqual(signed["freeze_signature"], pilot._canonical_sha(pilot._freeze_body(signed)))

    def test_fit_scope_is_exactly_four_units(self):
        self.assertEqual(
            pilot.FIT_IDS,
            (
                "REG_POS_POISSON_BAG2_V1_E67_P17",
                "REG_POS_POISSON_BAG2_V1_E67_P137",
                "REG_POS_POISSON_BAG2_V1_E75_P17",
                "REG_POS_POISSON_BAG2_V1_E75_P137",
            ),
        )

    def test_poisson_weights_are_seeded_and_normalized(self):
        rows = list(range(250))
        ids = [f"id-{index}" for index in rows]
        counts_a, weights_a = pilot._weights_for(rows, ids, 17)
        counts_b, weights_b = pilot._weights_for(rows, ids, 17)
        counts_other, _ = pilot._weights_for(rows, ids, 137)
        self.assertEqual(counts_a.tolist(), counts_b.tolist())
        self.assertEqual(weights_a.tolist(), weights_b.tolist())
        self.assertNotEqual(counts_a.tolist(), counts_other.tolist())
        self.assertAlmostEqual(float(weights_a.sum()), len(rows), places=10)
        self.assertTrue((weights_a >= 0).all())

    def test_frozen_train_source_hash_and_root_are_required(self):
        with tempfile.TemporaryDirectory(prefix="reg-pos-train-source-") as temporary:
            root = Path(temporary)
            train_path = root / "Train.csv"
            train_path.write_bytes(b"id,threat_rank\na,50\n")
            expected = sha256(train_path.read_bytes()).hexdigest()
            runner = SimpleNamespace(TRAIN_REL=Path("Train.csv"), TRAIN_SHA256=expected,
                                     _contained_file=lambda base, rel: base / rel)
            with patch.object(pilot, "SOURCE_ROOT", root):
                self.assertEqual(
                    pilot._assert_frozen_training_source(root, {"official_train_sha256": expected}, runner),
                    expected,
                )
                with self.assertRaises(pilot.PilotError):
                    pilot._assert_frozen_training_source(
                        root / "other", {"official_train_sha256": expected}, runner)
                with self.assertRaises(pilot.PilotError):
                    pilot._assert_frozen_training_source(
                        root, {"official_train_sha256": "0" * 64}, runner)

    def test_benchmark_gate_allows_small_d1_harm_only_inside_strict_bound(self):
        qualification, eligible = pilot._decision_gates(
            -0.01, -0.049, 0.01, True, 0.249, True, True)
        self.assertFalse(qualification)
        self.assertTrue(eligible)
        self.assertFalse(pilot._decision_gates(
            -0.01, -0.05, 0.01, True, 0.249, True, True)[1])
        self.assertFalse(pilot._decision_gates(
            -0.01, -0.049, 0.01, True, 0.25, True, True)[1])

    def test_synthetic_train_receipt_readback_prediction_metric_and_independent_verifier(self):
        with tempfile.TemporaryDirectory(prefix="reg-pos-poisson-report-") as temporary:
            report = pilot.run_synthetic_e2e(Path(temporary) / "SYNTHETIC_E2E.json")
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["synthetic_fit_units"], 4)
        self.assertEqual(report["production_fit_units_started"], 0)
        self.assertFalse(report["official_ledger_mutated"])
        self.assertEqual(report["independent_verifier"]["status"], "PASS")
        self.assertEqual(report["independent_verifier"]["fit_units"], 4)
        self.assertEqual(set(report["synthetic_metrics"]), {"67", "75"})


if __name__ == "__main__":
    unittest.main()
