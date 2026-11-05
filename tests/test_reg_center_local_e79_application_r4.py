"""Synthetic-only tests for the repaired R4 application lifecycle."""
from __future__ import annotations

import unittest

import numpy as np

from scripts import reg_center_local_e79_application_r4 as app


class E79ApplicationR4Tests(unittest.TestCase):
    def test_missing_pin_uses_parent_freeze_key(self) -> None:
        constants = app._load_e79_after_pass.__code__.co_consts
        self.assertIn("expected_sha256_from_parent_freeze", constants)
        self.assertNotIn("sha256", constants)

    def test_formula_and_missing_bits_remain_unchanged(self) -> None:
        raw = np.asarray([0.0, 49.0, 50.0, 51.0, 100.0], dtype="<f8")
        missing = np.asarray([False, False, True, False, False])
        result = app.apply_values(raw, missing, -1.25)
        self.assertEqual(result[2].view("<u8"), raw[2].view("<u8"))
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue(np.all((result >= 0.0) & (result <= 100.0)))

    def test_full_synthetic_apply_verify_receipt_readback_and_second_run(self) -> None:
        result = app._synthetic_full_application()
        self.assertEqual(result["status"], "PASS_SYNTHETIC_E79_FULL_APPLICATION_LIFECYCLE")
        self.assertEqual(result["rows"], app.EXPECTED_ROWS)
        self.assertEqual(result["missing_rows"], app.EXPECTED_MISSING)
        self.assertFalse(result["first_run_resumed"])
        self.assertTrue(result["second_run_resumed"])
        self.assertEqual(result["independent_output_verification"],
                         "PASS_INDEPENDENT_LOCAL_E79_APPLICATION")
        self.assertFalse(result["production_csv_created"])
        self.assertFalse(result["real_e79_read"])
        self.assertFalse(result["kaggle_upload"])

    def test_full_synthetic_cli_contains_end_to_end_evidence(self) -> None:
        result = app._synthetic_qa()
        lifecycle = result["full_application"]
        self.assertEqual(lifecycle["status"], "PASS_SYNTHETIC_E79_FULL_APPLICATION_LIFECYCLE")
        self.assertTrue(lifecycle["archive_hash_checked"])
        self.assertEqual(lifecycle["allowlisted_members_loaded"],
                         list(app.REQUIRED_E79_MEMBERS))
        self.assertEqual(lifecycle["harmless_extra_members_unloaded"], ["cycle_code", "raw"])
        self.assertTrue(lifecycle["all_synthetic_files_inside_temp"])


if __name__ == "__main__":
    unittest.main()
