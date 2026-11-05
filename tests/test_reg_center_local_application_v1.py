from __future__ import annotations

import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

import numpy as np

from scripts import reg_center_local_application_v1 as runner


class TestCenterLocalR3Formula(unittest.TestCase):
    def test_exact_lower_weighted_median_and_tie_interval(self) -> None:
        self.assertEqual(runner.lower_weighted_median_interval([0.0, 1.0], [1.0, 1.0]),
                         (0.0, 1.0, 2))
        self.assertEqual(runner.lower_weighted_median_interval([0.0, 1.0, 2.0], [1.0, 1.0, 1.0]),
                         (1.0, 1.0, 3))
        bounded = runner._bounded_interval(-4.0, -1.0)
        self.assertEqual(bounded, (-2.0, -1.0))

    def test_zero_basis_is_choice_independent_and_missing_bits_are_unchanged(self) -> None:
        direct = np.asarray([40.0, 50.0, 60.0, np.nextafter(50.0, 100.0)], dtype=np.float64)
        missing = np.asarray([False, False, False, True], dtype=bool)
        at_zero = runner.apply_local(direct, missing, -2.0)
        at_upper = runner.apply_local(direct, missing, 0.0)
        self.assertEqual(at_zero[0], at_upper[0])
        self.assertEqual(at_zero[2], at_upper[2])
        self.assertEqual(at_zero[3].view(np.uint64), direct[3].view(np.uint64))


class TestCenterLocalR3SourceBoundary(unittest.TestCase):
    def test_non_target_loader_does_not_parse_targets_member(self) -> None:
        with tempfile.TemporaryDirectory(prefix="center-local-r3-nontarget-") as temp:
            path = Path(temp) / "identity_only.npz"
            n = 4
            arrays = {
                "physical_rows": np.arange(n, dtype=np.int64),
                "record_ids": np.asarray([f"id-{i}" for i in range(n)]),
                "prediction_origins": np.asarray([67, 67, 75, 75], dtype=np.int64),
                "original_bundle_indices": np.asarray([0, 1, 0, 1], dtype=np.int64),
                "cycles": np.asarray([68, 69, 76, 77], dtype=np.int64),
                "direct_predictions": np.asarray([45.0, 50.0, 55.0, 60.0], dtype=np.float64),
                "missing_mask": np.asarray([False, False, False, True], dtype=bool),
            }
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, value in arrays.items():
                    buffer = io.BytesIO()
                    np.save(buffer, value, allow_pickle=False)
                    archive.writestr(f"{name}.npy", buffer.getvalue())
                # A malformed targets member proves the preflight path never opens it.
                archive.writestr("targets.npy", b"NOT A NUMPY ARRAY; MUST STAY UNREAD")
            loaded = runner._load_non_target_identity(path)
            self.assertEqual(set(loaded), set(runner.IDENTITY_FIELDS))
            self.assertEqual(len(loaded["record_ids"]), n)

    def test_synthetic_full_cli_receipt_resume_and_tamper(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(runner.ROOT / "scripts/reg_center_local_application_v1.py"),
             "--synthetic-qa"],
            cwd=runner.ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertEqual(result["status"], "PASS_SYNTHETIC_CLI_DURABLE_RESUME_TAMPER")
        self.assertEqual(result["source_verification"], "PASS_INDEPENDENT_SOURCE_READY_VERIFICATION")
        self.assertEqual(result["first_fit"], "PASS_LOCAL_SCALAR_FIT_COMPLETE")
        self.assertEqual(result["resume"], "PASS_DURABLE_LOCAL_FIT_RECEIPT")
        self.assertEqual(result["fit_verification"], "PASS_DURABLE_LOCAL_FIT_RECEIPT")
        self.assertAlmostEqual(result["known_delta"], -0.6, places=12)
        self.assertEqual(result["target_member_reads"], 1)
        self.assertFalse(result["resume_target_reload"])
        self.assertTrue(result["receipt_tamper_rejected_before_target_access"])
        self.assertTrue(result["state_tamper_rejected_before_target_access"])
        self.assertEqual(set(result["loaded_npz_members"]), set(runner.IDENTITY_FIELDS))
        self.assertFalse(result["real_data_accessed"])


if __name__ == "__main__":
    unittest.main()
