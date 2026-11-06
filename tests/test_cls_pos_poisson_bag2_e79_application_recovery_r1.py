from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts import cls_pos_poisson_bag2_e79_application_recovery_r1 as recovery
from scripts import verify_cls_pos_poisson_bag2_e79_application_recovery_r1 as verifier


class E79ApplicationRecoveryTests(unittest.TestCase):
    def test_synthetic_frozen_runner_recovery_persistence_and_tamper(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e79-application-recovery-r1-") as temp:
            result = recovery.run_synthetic_recovery(Path(temp))
            self.assertEqual(result["status"], "PASS_SYNTHETIC_FROZEN_RUNNER_RECOVERY_E2E")
            self.assertEqual(result["rows"], 29_421)
            self.assertGreaterEqual(result["missing_rows"], 1)
            self.assertGreaterEqual(result["candidate_changed_decisions"], 0)
            self.assertEqual(result["official_fit_count"], 0)
            self.assertEqual(result["application_attempts"], 1)
            self.assertEqual(result["application_resume_readbacks"], 1)
            self.assertTrue(result["fit_set_unchanged"])
            self.assertTrue(result["persistence_hashes_unchanged_on_resume"])
            self.assertEqual(len(result["binding_fields_forwarded"]), 7)
            self.assertFalse(result["test_targets_or_sample_submission_accessed"])
            self.assertFalse(result["kaggle_contacted"])

            audit_root = Path(temp) / "synthetic_audit"
            repository = Path(temp) / "synthetic_official_repo"
            recovery_root = audit_root / recovery.RECOVERY_REL
            paths = recovery._recovery_paths(audit_root)
            csv_path, csv_relative = recovery._recovery_csv_path(audit_root, repository)
            fit_set = recovery._fit_snapshot(audit_root)
            receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
            baseline = repository / recovery.frozen.BASELINE_CSV_REL
            with csv_path.open("a", encoding="utf-8") as stream:
                stream.write("tamper\n")
            with self.assertRaisesRegex(RuntimeError, "CSV hash mismatch|CSV hash"):
                verifier.verify_recovery_application(
                    recovery_root=recovery_root,
                    fit_root=audit_root / recovery.frozen.OUTPUT_REL,
                    receipt_path=paths["receipt"], bundle_path=paths["bundle"],
                    csv_path=csv_path, baseline_csv_path=baseline,
                    expected_baseline_sha256=receipt["baseline_csv_sha256"],
                    expected_fit_set=fit_set, expected_missing_rows=None,
                    authorization_file_sha256="SYNTHETIC_AUTHORIZATION",
                    expected_csv_relative_path=csv_relative,
                )


if __name__ == "__main__":
    unittest.main()
