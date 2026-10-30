"""Format and boundary checks for the authorized initial-submission path."""
import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts import prepare_initial_submissions as submissions
from scripts.prepare_initial_submissions import _submission_frame


class InitialSubmissionTests(unittest.TestCase):
    def setUp(self):
        self.test = pd.DataFrame({"record_id": ["A", "B", "C"]})
        self.sample = pd.DataFrame({"record_id": ["A", "B", "C"], "prediction": [0, 0, 0]})

    def test_binary_output_integer_and_order(self):
        sample = self.sample.rename(columns={"prediction": "artifact_signal"})
        out = _submission_frame(self.test, sample, np.array([0, 1, 0]), "artifact_signal", "classification")
        self.assertEqual(list(out.columns), ["record_id", "artifact_signal"])
        self.assertEqual(out.artifact_signal.tolist(), [0, 1, 0])
        self.assertTrue(pd.api.types.is_integer_dtype(out.artifact_signal))

    def test_clipped_regression_contract(self):
        sample = self.sample.rename(columns={"prediction": "threat_rank"})
        out = _submission_frame(self.test, sample, np.array([0., 25.5, 100.]), "threat_rank", "regression")
        self.assertTrue(np.isfinite(out.threat_rank).all())
        self.assertTrue(out.threat_rank.between(0, 100).all())

    def test_invalid_labels_and_nonfinite_regression_rejected(self):
        sample_cls = self.sample.rename(columns={"prediction": "artifact_signal"})
        sample_reg = self.sample.rename(columns={"prediction": "threat_rank"})
        with self.assertRaises(ValueError):
            _submission_frame(self.test, sample_cls, np.array([0, 2, 0]), "artifact_signal", "classification")
        with self.assertRaises(ValueError):
            _submission_frame(self.test, sample_reg, np.array([0., np.nan, 100.]), "threat_rank", "regression")

    def test_test_id_order_must_match_sample(self):
        sample = self.sample.iloc[::-1].rename(columns={"prediction": "threat_rank"})
        with self.assertRaises(ValueError):
            _submission_frame(self.test, sample, np.array([10., 20., 30.]), "threat_rank", "regression")

    def test_model_freeze_failure_precedes_any_fit_or_output_write(self):
        frames = (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {})
        with tempfile.TemporaryDirectory() as temp_root:
            project_root = Path(temp_root)
            output_dir = project_root / "submissions"
            output_dir.mkdir()
            sentinel = output_dir / "existing.csv"
            sentinel.write_text("preserve me\n", encoding="utf-8")
            before = {path.name: path.read_bytes() for path in output_dir.iterdir()}
            with patch.object(submissions, "PROJECT_ROOT", project_root), \
                 patch.object(submissions, "_load_inputs", return_value=frames), \
                 patch.object(submissions, "_verify_model_identity", side_effect=RuntimeError("freeze mismatch")) as verify, \
                 patch.object(submissions, "_classification") as fit:
                with self.assertRaisesRegex(RuntimeError, "freeze mismatch"):
                    submissions.run("classification", Path("unused-data-root"))
            verify.assert_called_once()
            fit.assert_not_called()
            self.assertEqual(before, {path.name: path.read_bytes() for path in output_dir.iterdir()})

    def test_data_root_defaults_to_tp2_data_dir_environment(self):
        with tempfile.TemporaryDirectory() as temp_root, patch.dict("os.environ", {"TP2_DATA_DIR": temp_root}):
            self.assertEqual(submissions.resolve_data_dir(), Path(temp_root).resolve())

    def test_matching_complete_kaggle_receipt_survives_status_derivation(self):
        receipt = {"status": "COMPLETE", "reference": 12345, "public_score": 0.5, "submissions_used": 1}
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = Path(temp_dir) / "manifest.json"
            manifest.write_text(
                '{"submission_sha256":"same-hash","kaggle_submission":' +
                json.dumps(receipt) + '}', encoding="utf-8"
            )
            preserved = submissions._prior_kaggle_receipt(manifest, "same-hash")
        self.assertEqual(preserved, receipt)
        self.assertEqual(submissions._preparation_status(preserved), "KAGGLE_SUBMISSION_COMPLETE")


if __name__ == "__main__":
    unittest.main()
