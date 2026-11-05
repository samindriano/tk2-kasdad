"""Synthetic-only tests for the source-gated local E79 CSV applicator."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts import reg_center_local_e79_application_v1 as app


class E79ApplicationSyntheticTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ids = np.asarray([f"row-{i:03d}" for i in range(12)])
        self.raw = np.asarray([0.0, 10.0, 20.0, 35.0, 49.0, 50.0,
                               51.0, 65.0, 80.0, 90.0, 99.0, 100.0], dtype="<f8")
        self.missing = np.zeros(len(self.ids), dtype=bool)
        self.missing[[2, 8]] = True
        self.delta = -1.25
        self.values = app.apply_values(self.raw, self.missing, self.delta)
        self.payload = app.render_csv(self.ids, self.values)

    def test_formula_clips_and_keeps_missing_float_bits(self) -> None:
        present = ~self.missing
        p = self.raw[present]
        base = (app.HINGE["intercept"] + app.HINGE["slope_low"] * np.minimum(p, 50.0)
                + app.HINGE["slope_high"] * np.maximum(p - 50.0, 0.0))
        basis = np.maximum(0.0, 1.0 - np.abs(p - 50.0) / 10.0)
        expected = np.clip(base + self.delta * basis, 0.0, 100.0)
        np.testing.assert_array_equal(self.values[present], expected)
        np.testing.assert_array_equal(self.values[self.missing].view("<u8"),
                                      self.raw[self.missing].view("<u8"))
        self.assertEqual(self.values[0], 1.7288079036674198)
        self.assertTrue(np.all((self.values >= 0.0) & (self.values <= 100.0)))

    def test_csv_header_order_hash_and_ranges(self) -> None:
        audit = app.verify_csv_bytes(self.payload, self.ids, self.values,
                                     self.missing, self.raw,
                                     expected_rows=len(self.ids),
                                     expected_missing=int(self.missing.sum()))
        rows = list(csv.reader(io.StringIO(self.payload.decode("utf-8"))))
        self.assertEqual(rows[0], ["record_id", "threat_rank"])
        self.assertEqual([row[0] for row in rows[1:]], self.ids.tolist())
        expected_hash = app.ordered_ids_sha256(self.ids)
        self.assertEqual(audit["ordered_ids_sha256"], expected_hash)
        canonical = json.dumps(self.ids.tolist(), sort_keys=True, ensure_ascii=False,
                               allow_nan=False, separators=(",", ":")).encode(
                                   "utf-8", errors="surrogatepass")
        import hashlib
        self.assertEqual(expected_hash, hashlib.sha256(canonical).hexdigest())

    def test_rejects_order_header_range_missing_tamper_and_duplicate_ids(self) -> None:
        mutations = {
            "reordered": app.render_csv(self.ids[::-1], self.values[::-1]),
            "header": b"record_id,prediction\n" + b"\n".join(self.payload.splitlines()[1:]) + b"\n",
            "range": b"record_id,threat_rank\nrow-000,101\n" + b"\n".join(self.payload.splitlines()[2:]) + b"\n",
            "missing": app.render_csv(
                self.ids,
                np.where(self.missing, np.nextafter(self.values, 100.0), self.values)),
        }
        duplicate = (b"record_id,threat_rank\n"
                     + b"row-000," + format(float(self.values[0]), ".17g").encode("ascii") + b"\n"
                     + b"row-000," + format(float(self.values[1]), ".17g").encode("ascii") + b"\n"
                     + b"\n".join(self.payload.splitlines()[3:]) + b"\n")
        mutations["duplicate"] = duplicate
        for label, payload in mutations.items():
            with self.subTest(label=label), self.assertRaises(app.ApplicationGateError):
                app.verify_csv_bytes(payload, self.ids, self.values,
                                     self.missing, self.raw,
                                     expected_rows=len(self.ids),
                                     expected_missing=int(self.missing.sum()))

    def test_idempotent_write_and_conflict_rejection(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e79-app-csv-test-") as temp:
            path = Path(temp) / "candidate.csv"
            self.assertFalse(app.write_csv_idempotent(path, self.payload))
            self.assertTrue(app.write_csv_idempotent(path, self.payload))
            with self.assertRaises(app.ApplicationGateError):
                app.write_csv_idempotent(path, self.payload + b"tampered")

    def test_runner_synthetic_suite_reports_no_real_e79_access(self) -> None:
        result = app._synthetic_qa()
        self.assertEqual(result["status"], "PASS_SYNTHETIC_E79_APPLICATION_CSV_QA")
        self.assertTrue(all(result["checks"].values()))
        self.assertFalse(result["real_e79_read"])
        self.assertFalse(result["production_csv_created"])
        self.assertFalse(result["kaggle_upload"])

    def test_rejects_duplicate_ids_before_csv_render(self) -> None:
        with self.assertRaises(app.ApplicationGateError):
            app.render_csv(np.asarray(["same", "same"]), np.asarray([1.0, 2.0]))


if __name__ == "__main__":
    unittest.main()
