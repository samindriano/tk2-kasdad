"""Synthetic-only regression tests for the R2 monotone hinge pipeline."""

from __future__ import annotations

import unittest

import numpy as np

from scripts import reg_monotone_hinge_p50_v2 as r2
from scripts import verify_reg_monotone_hinge_p50_v2 as verifier


def _valid_resource_snapshot(*, initial: bool = True) -> dict[str, object]:
    available = 3 * 1024**3 if initial else 2 * 1024**3
    estimate = 512 * 1024**2
    return {
        "status": "PASS",
        "boundary": "synthetic",
        "available_bytes_measured": available,
        "initial_floor_pass": initial,
        "minimum_initial_available_bytes": 2 * 1024**3,
        "incremental_estimate_bytes": estimate,
        "minimum_projected_free_bytes": 1024**3,
        "actual_available_boundary_minimum_bytes": 1024**3,
        "projected_free_bytes": available - estimate,
        "projected_free_pass": available - estimate >= 1024**3,
        "actual_boundary_pass": available >= 1024**3,
        "estimated_peak_is_measured": False,
        "process_rss_bytes": 128 * 1024**2,
        "process_rss_measured": True,
        "os_peak_working_set_bytes": 256 * 1024**2,
        "os_peak_working_set_measured": True,
        "os_peak_working_set_scope": "process_lifetime",
        "threads": 2,
        "concurrent_heavy_jobs": 1,
    }


class MonotoneHingeR2Tests(unittest.TestCase):
    def test_full_synthetic_matrix_passes(self) -> None:
        checks = verifier.run_synthetic_checks()
        self.assertGreaterEqual(len(checks), 10)
        self.assertTrue(all(check["pass"] for check in checks.values()))

    def test_source_pass_signature_binds_exact_two_diagnostic_ids(self) -> None:
        freeze = {
            "source_commit": "synthetic-commit",
            "source_hashes": {"script": "1" * 64},
            "input_hashes": {"freeze": {"sha256": "2" * 64}},
        }
        approval = {
            "verdict": "PASS",
            "experiment_id": r2.EXPERIMENT_ID,
            "source_commit": freeze["source_commit"],
            "prefit_freeze_sha256": "3" * 64,
            "source_hashes": freeze["source_hashes"],
            "input_hashes": freeze["input_hashes"],
            "authorized_fit_ids": list(r2.DIAGNOSTIC_FIT_IDS),
            "official_lp_start_cap": r2.MAX_LP_STARTS,
        }
        approval["signature"] = r2.canonical_sha(approval)
        r2.validate_source_pass_body(approval, freeze, "3" * 64)

        tampered = {**approval, "authorized_fit_ids": [*r2.DIAGNOSTIC_FIT_IDS, "HINGE_FINAL"]}
        tampered["signature"] = r2.canonical_sha(
            {key: value for key, value in tampered.items() if key != "signature"})
        with self.assertRaises(r2.GateError):
            r2.validate_source_pass_body(tampered, freeze, "3" * 64)

    def test_independent_resource_replay_rejects_false_measurement_claims(self) -> None:
        valid = _valid_resource_snapshot()
        verifier._verify_resource_snapshot("valid", valid, initial=True)

        bad_rss = {**valid, "process_rss_measured": False}
        with self.assertRaises(r2.GateError):
            verifier._verify_resource_snapshot("bad-rss", bad_rss, initial=True)

        bad_peak = {**valid, "os_peak_working_set_bytes": None}
        with self.assertRaises(r2.GateError):
            verifier._verify_resource_snapshot("bad-peak", bad_peak, initial=True)

    def test_independent_stage_replay_rejects_reordered_or_changed_targets(self) -> None:
        identity = {
            "rows": np.array([10, 11, 12, 13], dtype=np.int64),
            "ids": np.array(["a", "b", "c", "d"]),
            "origins": np.array([67, 67, 67, 75], dtype=np.int8),
            "indices": np.array([0, 1, 2, 3], dtype=np.int64),
            "cycles": np.array([68, 71, 72, 76], dtype=np.int16),
            "direct": np.array([10.0, 20.0, 30.0, 40.0], dtype="<f8"),
            "missing": np.array([False, True, False, False]),
        }
        targets = np.array([1.0, 2.0, 3.0, 4.0], dtype="<f8")
        rows = verifier._expected_stage_rows(identity, targets, "HINGE_D1", score=False)
        self.assertEqual(verifier._assert_frozen_stage_rows(
            rows, identity, targets, "HINGE_D1", score=False), 2)

        changed = {name: values.copy() for name, values in rows.items()}
        changed["targets"][0] += 0.01
        with self.assertRaises(r2.GateError):
            verifier._assert_frozen_stage_rows(
                changed, identity, targets, "HINGE_D1", score=False)

        reordered = {name: values.copy() for name, values in rows.items()}
        for name, values in reordered.items():
            reordered[name] = values[::-1].copy()
        with self.assertRaises(r2.GateError):
            verifier._assert_frozen_stage_rows(
                reordered, identity, targets, "HINGE_D1", score=False)

    def test_subgroup_harm_is_recomputed_from_metrics(self) -> None:
        def group(mae: float, n: int = 250) -> dict[str, object]:
            return {"status": "SUPPORTED", "n": n, "mae": mae}

        metrics = {
            "hinge_present_only": {"groups": {
                "present": group(10.2), "missing": group(12.0),
                "target_le_5": group(8.1), "target_ge_95": group(14.0)}},
            "causal_global_present_only": {"groups": {
                "present": group(10.0), "missing": group(12.1),
                "target_le_5": group(8.0), "target_ge_95": group(13.8)}},
        }
        result = verifier._independent_subgroup_harm(metrics)
        self.assertAlmostEqual(result["groups"]["present"]["harm"], 0.2)
        self.assertAlmostEqual(result["groups"]["missing"]["harm"], -0.1)
        self.assertAlmostEqual(result["maximum_supported_harm"], 0.2)

    def test_production_csv_route_rejects_synthetic_row_count(self) -> None:
        ids = np.array(["synthetic-a", "synthetic-b"])
        raw = np.array([10.0, 20.0], dtype="<f8")
        candidate = raw.copy()
        missing = np.array([True, False])
        with self.assertRaises(r2.GateError):
            r2.write_candidate_csv(
                r2.ROOT / "unused-synthetic.csv", ids, candidate,
                missing_mask=missing, raw_direct=raw)


if __name__ == "__main__":
    unittest.main()
