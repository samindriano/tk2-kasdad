from __future__ import annotations

import tempfile
import unittest
from hashlib import sha256
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts import reg_pos_poisson_bag2_v1 as pilot
from scripts import verify_reg_pos_poisson_bag2_v1 as verifier


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

    def test_reviewed_source_commit_ancestor_ignores_unrelated_untracked_files(self):
        with tempfile.TemporaryDirectory(prefix="reg-pos-ancestry-") as temporary:
            root = Path(temporary)

            def git(*args: str) -> str:
                return subprocess.run(["git", "-C", str(root), *args], check=True,
                                      capture_output=True, text=True).stdout.strip()

            git("init", "--quiet")
            git("config", "user.name", "Codex Test")
            git("config", "user.email", "codex-test@example.invalid")
            tracked = root / "tracked.txt"
            tracked.write_text("frozen source\n", encoding="utf-8")
            git("add", "tracked.txt")
            git("-c", "user.name=Codex Test", "-c", "user.email=codex-test@example.invalid",
                "commit", "--quiet", "-m", "reviewed source")
            reviewed = git("rev-parse", "HEAD")
            unrelated = root / "unrelated.txt"
            unrelated.write_text("later unrelated commit\n", encoding="utf-8")
            git("add", "unrelated.txt")
            git("-c", "user.name=Codex Test", "-c", "user.email=codex-test@example.invalid",
                "commit", "--quiet", "-m", "unrelated follow-up")
            current = git("rev-parse", "HEAD")
            frozen_sha = sha256(tracked.read_bytes()).hexdigest()
            blob_sha = sha256(subprocess.run(
                ["git", "-C", str(root), "show", f"{reviewed}:tracked.txt"],
                check=True, capture_output=True).stdout).hexdigest()
            approval = {"local_commit": reviewed,
                        "tracked_source_hashes": {"tracked.txt": frozen_sha},
                        "tracked_blob_hashes": {"tracked.txt": blob_sha}}
            with patch.object(pilot, "REQUIRED_ROOT_TRACKED_PATHS", {"tracked.txt"}):
                approved, head = pilot._verify_approved_source_commit(root, reviewed, approval)
                self.assertEqual((approved, head), (reviewed, current))
                with patch.object(verifier, "REQUIRED_ROOT_TRACKED_PATHS", {"tracked.txt"}):
                    self.assertEqual(
                        verifier._verify_approved_source_commit(root, reviewed, approval),
                        (reviewed, current))
                self.assertFalse(pilot._is_git_ancestor(root, current, reviewed))
                (root / "untracked-note.txt").write_text("not part of this lane\n", encoding="utf-8")
                self.assertEqual(
                    pilot._verify_approved_source_commit(root, reviewed, approval), (reviewed, current))
                with patch.object(verifier, "REQUIRED_ROOT_TRACKED_PATHS", {"tracked.txt"}):
                    self.assertEqual(
                        verifier._verify_approved_source_commit(root, reviewed, approval),
                        (reviewed, current))
                tracked.write_text("changed frozen source\n", encoding="utf-8")
                with self.assertRaises(pilot.PilotError):
                    pilot._verify_approved_source_commit(root, reviewed, approval)
                with patch.object(verifier, "REQUIRED_ROOT_TRACKED_PATHS", {"tracked.txt"}):
                    with self.assertRaises(verifier.VerifyError):
                        verifier._verify_approved_source_commit(root, reviewed, approval)

    def test_shared_advisory_lock_is_nonblocking_compatible_and_persistent(self):
        with tempfile.TemporaryDirectory(prefix="reg-pos-lock-") as temporary:
            lock_path = Path(temporary) / "kasdad_single_heavy_job.lock"
            child_code = (
                "import sys; from pathlib import Path; "
                "from scripts.verify_reg_pos_poisson_bag2_v1 import _SharedHeavyJobLock, VerifyError; "
                "p=Path(sys.argv[1]); "
                "try_lock=None; "
                "\ntry:\n with _SharedHeavyJobLock(p): print('ACQUIRED')"
                "\nexcept VerifyError: print('BLOCKED')"
            )
            with pilot._SharedHeavyJobLock(lock_path):
                child = subprocess.run([sys.executable, "-c", child_code, str(lock_path)],
                                       cwd=pilot.ROOT, check=True, capture_output=True, text=True,
                                       timeout=15)
                self.assertEqual(child.stdout.strip(), "BLOCKED")
            self.assertTrue(lock_path.is_file())
            child = subprocess.run([sys.executable, "-c", child_code, str(lock_path)],
                                   cwd=pilot.ROOT, check=True, capture_output=True, text=True,
                                   timeout=15)
            self.assertEqual(child.stdout.strip(), "ACQUIRED")
            self.assertTrue(lock_path.is_file())
            runner_child_code = (
                "import sys; from pathlib import Path; "
                "from scripts.reg_pos_poisson_bag2_v1 import _SharedHeavyJobLock, PilotError; "
                "p=Path(sys.argv[1]); "
                "\ntry:\n with _SharedHeavyJobLock(p): print('ACQUIRED')"
                "\nexcept PilotError: print('BLOCKED')"
            )
            with verifier._SharedHeavyJobLock(lock_path):
                child = subprocess.run([sys.executable, "-c", runner_child_code, str(lock_path)],
                                       cwd=pilot.ROOT, check=True, capture_output=True, text=True,
                                       timeout=15)
                self.assertEqual(child.stdout.strip(), "BLOCKED")
            self.assertTrue(lock_path.is_file())
            child = subprocess.run([sys.executable, "-c", runner_child_code, str(lock_path)],
                                   cwd=pilot.ROOT, check=True, capture_output=True, text=True,
                                   timeout=15)
            self.assertEqual(child.stdout.strip(), "ACQUIRED")
            self.assertTrue(lock_path.is_file())

    def test_benchmark_gate_uses_inclusive_frozen_boundaries(self):
        qualification, eligible = pilot._decision_gates(
            -0.01, -0.05, 0.01, True, 0.25, True, True)
        self.assertFalse(qualification)
        self.assertTrue(eligible)
        self.assertFalse(pilot._decision_gates(
            -0.01, -0.050001, 0.01, True, 0.25, True, True)[1])
        self.assertFalse(pilot._decision_gates(
            -0.01, -0.05, 0.01, True, 0.250001, True, True)[1])

    def test_score_prevalidation_fails_before_any_target_read(self):
        freeze = {"status": "PREFIT_FROZEN", "source_artifacts": {}}
        prediction_freeze = {"status": "PREDICTIONS_FROZEN"}
        runner = SimpleNamespace(_read_targets_for_rows=Mock())
        with (patch.object(pilot, "_validate_signed_json", side_effect=[freeze, prediction_freeze]),
              patch.object(pilot, "_score_gate", return_value={"status": "PASS"}),
              patch.object(pilot, "_runner", return_value=runner),
              patch.object(pilot, "_assert_frozen_training_source"),
              patch.object(pilot, "_source_artifact_hashes", return_value={}),
              patch.object(pilot, "_prevalidate_score_inputs",
                            side_effect=pilot.PilotError("E75 score input hash mismatch")) as prevalidate):
            with self.assertRaisesRegex(pilot.PilotError, "E75 score input hash mismatch"):
                pilot.score_official()
        self.assertEqual(prevalidate.call_count, 1)
        runner._read_targets_for_rows.assert_not_called()

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
