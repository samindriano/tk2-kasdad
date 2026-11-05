"""Synthetic-only contract tests for the two frozen REG simplifications."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import reg_simplification_benchmark_v1 as runner
from scripts import verify_reg_simplification_benchmark_v1 as verifier


class SimplificationContractTests(unittest.TestCase):
    def test_hinge_keeps_missing_rows_raw_and_transforms_only_present_rows(self) -> None:
        raw = np.asarray([0.0, 49.0, 50.0, 100.0], dtype="<f8")
        missing = np.asarray([True, False, True, False])
        mapped = runner._hinge_present(raw, missing)
        self.assertEqual(mapped[0].view("<u8"), raw[0].view("<u8"))
        self.assertEqual(mapped[2].view("<u8"), raw[2].view("<u8"))
        expected = (runner.HINGE["intercept"]
                    + runner.HINGE["slope_low"] * 49.0)
        self.assertEqual(mapped[1], expected)
        self.assertEqual(mapped[3], runner.HINGE["intercept"]
                         + 50 * runner.HINGE["slope_low"]
                         + 50 * runner.HINGE["slope_high"])

    def test_rank50_uses_average_ties_per_cycle_without_early_clipping(self) -> None:
        raw = np.asarray([300.0, 300.0, 0.0, 10.0])
        cycles = np.asarray([76, 76, 76, 77])
        mapped, counts = runner._rank50_unclipped(raw, cycles)
        self.assertEqual(counts, {"76": 3, "77": 1})
        self.assertEqual(mapped[0], 0.5 * 300.0 + 0.5 * 100.0 * 2.5 / 3.0)
        self.assertEqual(mapped[1], mapped[0])
        self.assertEqual(mapped[2], 0.5 * 100.0 / 3.0)
        self.assertEqual(mapped[3], 55.0)
        self.assertGreater(mapped[0], 100.0)

    def test_resource_gate_uses_light_lane_thresholds_and_releases_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lock_path = Path(directory) / "REG.lock"
            with patch.object(runner, "LANE_LOCK_PATH", lock_path), \
                    patch.object(runner.psutil, "virtual_memory",
                                 return_value=type("Mem", (), {"available": 3 * 1024**3})()):
                owner, evidence = runner._acquire_resource_lock()
                self.assertTrue(lock_path.is_file())
                self.assertEqual(evidence["start_available_bytes"], 3 * 1024**3)
                self.assertEqual(evidence["estimated_incremental_bytes"], 512 * 1024**2)
                self.assertFalse(evidence["shared_heavy_lock_used"])
                self.assertTrue(owner["token"])
                runner._release_resource_lock()
                self.assertFalse(lock_path.exists())

    def test_resource_gate_fails_closed_below_parallel_start_floor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lock_path = Path(directory) / "REG.lock"
            with patch.object(runner, "LANE_LOCK_PATH", lock_path), \
                    patch.object(runner.psutil, "virtual_memory",
                                 return_value=type("Mem", (), {"available": 2 * 1024**3})()):
                with self.assertRaises(runner.BenchmarkError):
                    runner._acquire_resource_lock()
            self.assertFalse(lock_path.exists())

    def test_upload_log_is_durable_ordered_and_never_restarts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "candidate.jsonl"
            runner.append_upload_event(log, {"event": "UPLOAD_INTENT"}, first=True)
            runner.append_upload_event(log, {"event": "UPLOAD_RESPONSE", "returncode": 0}, first=False)
            runner.append_upload_event(log, {"event": "COMPETITION_STATUS_CONFIRMED",
                                             "status": "VISIBLE_PENDING"}, first=False)
            events = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([event["event"] for event in events], [
                "UPLOAD_INTENT", "UPLOAD_RESPONSE", "COMPETITION_STATUS_CONFIRMED"])
            with self.assertRaises(runner.BenchmarkError):
                runner.append_upload_event(log, {"event": "UPLOAD_INTENT"}, first=True)

    def test_csv_create_is_exclusive_and_round_trips_canonical_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.csv"
            ids = np.asarray(["a", "b"])
            values = np.asarray([1.0 / 3.0, 88.25], dtype=np.float64)
            digest = runner._write_csv_once(path, ids, values)
            self.assertEqual(len(digest), 64)
            with path.open("r", newline="", encoding="utf-8") as stream:
                rows = list(csv.reader(stream))
            self.assertEqual(rows[0], ["record_id", "threat_rank"])
            self.assertEqual([float(row[1]) for row in rows[1:]], values.tolist())
            with self.assertRaises(runner.BenchmarkError):
                runner._write_csv_once(path, ids, values)

    def test_history_row_lookup_requires_exact_description(self) -> None:
        message = "Samuel REG Simplified No POS Hinge Benchmark"
        stdout = json.dumps([{
            "description": message,
            "status": "SubmissionStatus.COMPLETE",
            "publicScore": 12.5,
        }])
        row = runner._find_history_row(stdout, message)
        self.assertEqual(row["status"], "SubmissionStatus.COMPLETE")
        self.assertIsNone(runner._find_history_row(stdout, message + " extra"))
        self.assertEqual(runner._row_field(row, "public_score"), 12.5)

    def test_source_array_digest_matches_pinned_e79_serialization_contract(self) -> None:
        numeric = np.asarray([1.25, 2.5], dtype="<f8")
        string = np.asarray(["row-a", "row-b"], dtype=object)
        self.assertEqual(runner.source_array_sha(numeric, dtype="<f8"),
                         verifier._source_array_sha(numeric, dtype="<f8"))
        self.assertEqual(runner.source_array_sha(string), verifier._source_array_sha(string))

    def test_id_vector_validation_does_not_apply_numeric_finiteness(self) -> None:
        ids = np.asarray(["record-a", "record-b"])
        validated = runner._require_vector(ids, "record_id", n=2, finite=False)
        self.assertEqual(validated.tolist(), ids.tolist())

    def test_independent_csv_reader_rejects_noncanonical_numeric_encoding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.csv"
            path.write_text("record_id,threat_rank\na,1.000\n", encoding="utf-8")
            with self.assertRaises(verifier.VerificationError):
                verifier._read_output(path, np.asarray(["a"]))


if __name__ == "__main__":
    unittest.main()
