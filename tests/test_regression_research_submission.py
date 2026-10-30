import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import submit_regression_research as submit


class SubmissionGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.raw = self.root / "data" / "raw" / "regression"
        self.csv_dir = self.root / "submissions" / "regression_research"
        self.manifest_dir = self.root / "reports" / "regression_research" / "submissions"
        self.freeze_path = self.root / "experiments" / "round1" / "freeze.json"
        self.raw.mkdir(parents=True)
        self.csv_dir.mkdir(parents=True)
        self.manifest_dir.mkdir(parents=True)
        self.freeze_path.parent.mkdir(parents=True)
        self.train = self.raw / "Train.csv"
        self.test = self.raw / "Test.csv"
        self.sample = self.raw / "SampleSubmission.csv"
        self.train.write_text("record_id,threat_rank\ntr_a,12\ntr_b,87\n", encoding="utf-8")
        self.test.write_text("record_id,cycle_code\nid_a,80\nid_b,91\n", encoding="utf-8")
        self.sample.write_text("record_id,threat_rank\nid_a,50\nid_b,50\n", encoding="utf-8")
        self.input_hashes = {
            "train": submit.sha256_file(self.train),
            "test": submit.sha256_file(self.test),
            "sample": submit.sha256_file(self.sample),
        }
        self.freeze_path.write_text(
            json.dumps({"status": "FROZEN", "identity": {"input_sha256": self.input_hashes}}),
            encoding="utf-8",
        )
        self.csv_path = self.csv_dir / "candidate_a.csv"
        self.manifest_path = self.manifest_dir / "candidate_a.json"
        self.write_candidate()
        self.patches = [
            mock.patch.object(submit, "EXPECTED_TRAIN_ROWS", 2),
            mock.patch.object(submit, "EXPECTED_TEST_ROWS", 2),
        ]
        for patcher in self.patches:
            patcher.start()

    def tearDown(self):
        for patcher in reversed(getattr(self, "patches", [])):
            patcher.stop()
        self.temp.cleanup()

    def write_candidate(self, rows=None, *, candidate_id="candidate_a", overrides=None):
        csv_path = self.csv_dir / f"{candidate_id}.csv"
        manifest_path = self.manifest_dir / f"{candidate_id}.json"
        rows = rows or [("id_a", "20.25"), ("id_b", "80")]
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            import csv

            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(submit.CSV_HEADER)
            writer.writerows(rows)
        manifest = {
            "status": "KAGGLE_CANDIDATE_APPROVED",
            "task": "regression",
            "candidate_id": candidate_id,
            "competition": submit.COMPETITION,
            "main_approval": {"approved": True, "approved_by": "main"},
            "source_commit": "a" * 40,
            "protocol_sha256": "b" * 64,
            "recipe": {"family": "LightGBM", "objective": "regression_l1"},
            "input_sha256": dict(self.input_hashes),
            "csv_sha256": submit.sha256_file(csv_path),
            "fit_rows": 2,
            "test_rows": 2,
            "test_labels_used": False,
            "test_ids_sha256": submit.ordered_ids_sha256(["id_a", "id_b"]),
        }
        if overrides:
            manifest.update(overrides)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        if candidate_id == "candidate_a":
            self.csv_path = csv_path
            self.manifest_path = manifest_path
        return csv_path, manifest_path

    def history(self, *, candidate_id=None, file_name=None, status="COMPLETE", ref="57808231"):
        if candidate_id is None:
            return [{
                "ref": ref,
                "file_name": "regression_initial.csv",
                "message": "Samuel R1-D Expanded HGB Initial Benchmark",
                "status": status,
                "date": "2026-10-30",
                "public_score": None,
            }]
        return [{
            "ref": ref,
            "file_name": file_name or f"{candidate_id}.csv",
            "message": f"Samuel REG Research {candidate_id}",
            "status": status,
            "date": "2026-10-30",
            "public_score": 11.2,
        }]

    def test_audit_checks_full_identity_and_exact_order(self):
        with mock.patch.dict("os.environ", {"TP2_DATA_DIR": ""}):
            result = submit.audit_candidate(self.csv_path, self.manifest_path, project_root=self.root)
        self.assertEqual(result["status"], "AUDIT_PASS")
        self.assertEqual(result["fit_rows"], 2)
        self.assertEqual(result["test_rows"], 2)
        self.assertEqual(result["test_ids_sha256"], submit.ordered_ids_sha256(["id_a", "id_b"]))

    def test_env_data_root_is_resolved_and_used(self):
        env_root = self.root / "external_dataset"
        env_regression = env_root / "regression"
        env_regression.mkdir(parents=True)
        for source in (self.train, self.test, self.sample):
            (env_regression / source.name).write_bytes(source.read_bytes())
        with mock.patch.dict("os.environ", {"TP2_DATA_DIR": str(env_root)}):
            resolved = submit._resolve_official_files(self.root)
            result = submit.audit_candidate(self.csv_path, self.manifest_path, project_root=self.root)
        self.assertEqual(resolved["train"], env_regression / "Train.csv")
        self.assertEqual(result["status"], "AUDIT_PASS")

    def test_rejects_unapproved_manifest_before_any_kaggle_call(self):
        self.write_candidate(overrides={"main_approval": {"approved": False}})
        with mock.patch.object(submit, "_run_cli") as run_cli:
            with self.assertRaises(submit.CandidateError):
                submit.audit_candidate(self.csv_path, self.manifest_path, project_root=self.root)
        run_cli.assert_not_called()

    def test_rejects_path_escape(self):
        outside = self.root / "outside.csv"
        outside.write_text("record_id,threat_rank\nid_a,1\nid_b,2\n", encoding="utf-8")
        with self.assertRaisesRegex(submit.CandidateError, "inside submissions"):
            submit.audit_candidate(outside, self.manifest_path, project_root=self.root)

    def test_rejects_wrong_ids_or_duplicate_ids(self):
        self.write_candidate(rows=[("id_b", "20"), ("id_a", "80")])
        with self.assertRaisesRegex(submit.CandidateError, "exact order"):
            submit.audit_candidate(self.csv_path, self.manifest_path, project_root=self.root)
        self.write_candidate(rows=[("id_a", "20"), ("id_a", "80")])
        with self.assertRaisesRegex(submit.CandidateError, "duplicate"):
            submit.audit_candidate(self.csv_path, self.manifest_path, project_root=self.root)

    def test_rejects_nonfinite_or_out_of_range_values(self):
        for raw in ("nan", "inf", "-0.1", "100.1"):
            with self.subTest(raw=raw):
                self.write_candidate(rows=[("id_a", raw), ("id_b", "50")])
                with self.assertRaisesRegex(submit.CandidateError, r"finite and in \[0, 100\]"):
                    submit.audit_candidate(self.csv_path, self.manifest_path, project_root=self.root)

    def test_default_cli_is_audit_only(self):
        output = io.StringIO()
        with mock.patch.object(submit, "audit_candidate", return_value={"status": "AUDIT_PASS"}), mock.patch.object(
            submit, "_run_cli"
        ) as run_cli:
            with contextlib.redirect_stdout(output):
                code = submit.main(["--csv", str(self.csv_path), "--manifest", str(self.manifest_path)])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue())["status"], "AUDIT_PASS")
        run_cli.assert_not_called()
        self.assertFalse((self.root / "reports" / "regression_research" / "kaggle_log.json").exists())

    def test_explicit_execute_uploads_once_and_records_history_ref(self):
        audit = submit.audit_candidate(self.csv_path, self.manifest_path, project_root=self.root)
        before = self.history()
        after = before + self.history(candidate_id="candidate_a", ref="92000001")
        with mock.patch.object(submit, "_history_rows", side_effect=[before, after]), mock.patch.object(
            submit, "_submission_limits", return_value={"numToday": 1, "numTotal": 1, "numAllowedNow": 9}
        ), mock.patch.object(submit, "_run_cli", return_value=mock.Mock(returncode=0, stdout="", stderr="")) as run_cli:
            result = submit.execute_candidate(audit, self.csv_path, kaggle_cli="mock-kaggle", project_root=self.root)
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["submission_ref"], "92000001")
        run_cli.assert_called_once()
        args = run_cli.call_args.args[1]
        self.assertEqual(args[:3], ["competitions", "submit", submit.COMPETITION])
        self.assertEqual(args[args.index("-m") + 1], "Samuel REG Research candidate_a")
        ledger = json.loads((self.root / "reports" / "regression_research" / "kaggle_log.json").read_text())
        self.assertEqual(submit._count_additional_candidates(ledger), 1)
        self.assertTrue(any(event["event"] == "history_preflight" for event in ledger["events"]))
        submit._validate_ledger(ledger)

    def test_existing_exact_candidate_history_is_reconciled_without_upload(self):
        audit = submit.audit_candidate(self.csv_path, self.manifest_path, project_root=self.root)
        with mock.patch.object(
            submit, "_history_rows", return_value=self.history(candidate_id="candidate_a", ref="92000002")
        ), mock.patch.object(submit, "_submission_limits") as limits, mock.patch.object(submit, "_run_cli") as run_cli:
            result = submit.execute_candidate(audit, self.csv_path, kaggle_cli="mock-kaggle", project_root=self.root)
        self.assertEqual(result["status"], "HISTORY_RECONCILED")
        self.assertFalse(result["uploaded_now"])
        limits.assert_not_called()
        run_cli.assert_not_called()

    def test_mismatched_history_filename_blocks_upload(self):
        audit = submit.audit_candidate(self.csv_path, self.manifest_path, project_root=self.root)
        with mock.patch.object(
            submit,
            "_history_rows",
            return_value=self.history(candidate_id="candidate_a", file_name="other.csv"),
        ), mock.patch.object(submit, "_run_cli") as run_cli:
            with self.assertRaisesRegex(submit.CandidateError, "different filename"):
                submit.execute_candidate(audit, self.csv_path, kaggle_cli="mock-kaggle", project_root=self.root)
        run_cli.assert_not_called()

    def test_six_candidate_cap_is_enforced_before_upload(self):
        ledger_path = submit._ledger_path(self.root)
        ledger = submit._initial_ledger()
        for index in range(6):
            submit._append_event(ledger, "candidate_reserved", f"prior_{index}", file_name=f"prior_{index}.csv")
        submit._save_ledger(ledger_path, ledger)
        audit = submit.audit_candidate(self.csv_path, self.manifest_path, project_root=self.root)
        with mock.patch.object(submit, "_history_rows", return_value=self.history()), mock.patch.object(
            submit, "_submission_limits", return_value={"numToday": 1, "numTotal": 1, "numAllowedNow": 9}
        ), mock.patch.object(submit, "_run_cli") as run_cli:
            with self.assertRaisesRegex(submit.CandidateError, "Six additional"):
                submit.execute_candidate(audit, self.csv_path, kaggle_cli="mock-kaggle", project_root=self.root)
        run_cli.assert_not_called()

    def test_six_candidate_cap_also_counts_kaggle_history_not_in_ledger(self):
        audit = submit.audit_candidate(self.csv_path, self.manifest_path, project_root=self.root)
        rows = self.history()
        for index in range(6):
            rows.extend(self.history(candidate_id=f"prior_{index}", ref=str(92000100 + index)))
        with mock.patch.object(submit, "_history_rows", return_value=rows), mock.patch.object(
            submit, "_submission_limits"
        ) as limits, mock.patch.object(submit, "_run_cli") as run_cli:
            with self.assertRaisesRegex(submit.CandidateError, "Six additional"):
                submit.execute_candidate(audit, self.csv_path, kaggle_cli="mock-kaggle", project_root=self.root)
        limits.assert_not_called()
        run_cli.assert_not_called()

    def test_zero_daily_allowance_blocks_before_reservation_or_upload(self):
        audit = submit.audit_candidate(self.csv_path, self.manifest_path, project_root=self.root)
        with mock.patch.object(submit, "_history_rows", return_value=self.history()), mock.patch.object(
            submit, "_submission_limits", return_value={"numToday": 10, "numTotal": 10, "numAllowedNow": 0}
        ), mock.patch.object(submit, "_run_cli") as run_cli:
            with self.assertRaisesRegex(submit.CandidateError, "no submission allowance"):
                submit.execute_candidate(audit, self.csv_path, kaggle_cli="mock-kaggle", project_root=self.root)
        run_cli.assert_not_called()
        self.assertFalse((self.root / "reports" / "regression_research" / "kaggle_log.json").exists())

    def test_ambiguous_upload_is_reserved_and_never_retried(self):
        audit = submit.audit_candidate(self.csv_path, self.manifest_path, project_root=self.root)
        with mock.patch.object(submit, "_history_rows", return_value=self.history()), mock.patch.object(
            submit, "_submission_limits", return_value={"numToday": 1, "numTotal": 1, "numAllowedNow": 9}
        ), mock.patch.object(submit, "_run_cli", side_effect=submit.KaggleCLIError("hidden")) as run_cli:
            result = submit.execute_candidate(audit, self.csv_path, kaggle_cli="mock-kaggle", project_root=self.root)
        self.assertEqual(result["status"], "OUTCOME_UNKNOWN")
        self.assertEqual(run_cli.call_count, 1)
        ledger_path = self.root / "reports" / "regression_research" / "kaggle_log.json"
        ledger = submit._load_ledger(ledger_path)
        self.assertIn("candidate_outcome_unknown", [event["event"] for event in ledger["events"]])
        with mock.patch.object(submit, "_history_rows", return_value=self.history()), mock.patch.object(
            submit, "_run_cli"
        ) as retry_cli:
            with self.assertRaisesRegex(submit.CandidateError, "no matching Kaggle history; no retry"):
                submit.execute_candidate(audit, self.csv_path, kaggle_cli="mock-kaggle", project_root=self.root)
        retry_cli.assert_not_called()

    def test_refresh_is_idempotent_and_never_uploads(self):
        ledger_path = submit._ledger_path(self.root)
        ledger = submit._initial_ledger()
        submit._append_event(
            ledger,
            "candidate_reserved",
            "candidate_a",
            file_name="candidate_a.csv",
            csv_sha256="c" * 64,
        )
        submit._append_event(
            ledger,
            "candidate_outcome_unknown",
            "candidate_a",
            status="OUTCOME_UNKNOWN",
            reason="mock",
        )
        submit._save_ledger(ledger_path, ledger)
        with mock.patch.object(
            submit, "_history_rows", return_value=self.history(candidate_id="candidate_a", ref="92000003")
        ), mock.patch.object(submit, "_run_cli") as run_cli:
            first = submit.refresh_ledger(kaggle_cli="mock-kaggle", project_root=self.root)
            second = submit.refresh_ledger(kaggle_cli="mock-kaggle", project_root=self.root)
        self.assertEqual(first["new_events"], 1)
        self.assertEqual(second["new_events"], 0)
        self.assertFalse(first["uploaded"])
        run_cli.assert_not_called()


if __name__ == "__main__":
    unittest.main()
