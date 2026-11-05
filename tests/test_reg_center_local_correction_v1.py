"""Synthetic-only checks for the R1 center-local correction source pipeline."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np

from scripts import reg_center_local_correction_v1 as runner
from scripts import verify_reg_center_local_correction_v1 as verifier


REPO = Path(__file__).resolve().parents[1]
FIT_COUNTS = {"HINGE_D1": 4_950, "HINGE_D2": 12_412}
PRESENT_COUNTS = {"HINGE_D1": 4_608, "HINGE_D2": 11_563}
SCORE_COUNTS = {"HINGE_D1": 7_462, "HINGE_D2": 8_491}
ORIGIN = {("HINGE_D1", "fit"): 67, ("HINGE_D1", "score"): 67,
          ("HINGE_D2", "fit"): 67, ("HINGE_D2", "score"): 75}
CYCLES = {("HINGE_D1", "fit"): tuple(range(68, 72)),
          ("HINGE_D1", "score"): tuple(range(72, 76)),
          ("HINGE_D2", "fit"): tuple(range(68, 76)),
          ("HINGE_D2", "score"): tuple(range(76, 80))}
PARAMS = {"HINGE_D1": {"intercept": 2.0, "slope_low": 0.90, "slope_high": 0.95},
          "HINGE_D2": {"intercept": 2.5, "slope_low": 0.92, "slope_high": 0.96}}
PATTERN = np.asarray([5, 18, 39, 40, 45, 48, 49, 50, 51, 52, 55, 60, 61, 80, 99, 100], dtype="<f8")


def _canonical(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         allow_nan=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, body: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, sort_keys=True, ensure_ascii=False,
        allow_nan=False, indent=2) + "\n", encoding="utf-8")


def _write_signed(path: Path, body: dict[str, object]) -> None:
    body = dict(body)
    body["signature"] = _canonical(body)
    _write_json(path, body)


def _make_arrays(stage: str, role: str, *, physical_offset: int) -> dict[str, np.ndarray]:
    n = FIT_COUNTS[stage] if role == "fit" else SCORE_COUNTS[stage]
    p = np.resize(PATTERN, n).astype("<f8", copy=False)
    missing = np.zeros(n, dtype=bool)
    if role == "fit":
        missing[:n - PRESENT_COUNTS[stage]] = True
    else:
        miss_n = 507 if stage == "HINGE_D1" else 526
        missing[:miss_n] = True
    cycles = np.resize(np.asarray(CYCLES[(stage, role)], dtype=np.int16), n)
    params = PARAMS[stage]
    base = (params["intercept"] + params["slope_low"] * np.minimum(p, 50.0)
            + params["slope_high"] * np.maximum(p - 50.0, 0.0))
    basis = np.maximum(0.0, 1.0 - np.abs(p - 50.0) / 10.0)
    target = np.clip(base - basis, 0.0, 100.0).astype("<f8")
    arrays = {
        "physical_rows": np.arange(physical_offset, physical_offset + n, dtype=np.int64),
        "record_ids": np.asarray([f"{stage}-{role}-{i:05d}" for i in range(n)]),
        "prediction_origins": np.full(n, ORIGIN[(stage, role)], dtype=np.int8),
        "original_bundle_indices": np.arange(n, dtype=np.int64),
        "cycles": cycles,
        "direct_predictions": p.copy(),
        "missing_mask": missing,
        "targets": target,
    }
    return arrays


def _hinge_present(p: np.ndarray, missing: np.ndarray,
                   params: dict[str, float]) -> np.ndarray:
    out = p.copy()
    present = ~missing
    out[present] = np.clip(params["intercept"]
        + params["slope_low"] * np.minimum(p[present], 50.0)
        + params["slope_high"] * np.maximum(p[present] - 50.0, 0.0), 0.0, 100.0)
    return out


class SyntheticLayout:
    def __init__(self, tmp: tempfile.TemporaryDirectory[str]):
        self.temp = Path(tmp.name)
        self.root = self.temp / "repo"
        self.root.mkdir()
        subprocess.run(["git", "init", "--quiet", str(self.root)], check=True,
                       capture_output=True, text=True)
        subprocess.run(["git", "-C", str(self.root), "-c", "user.name=Synthetic",
                        "-c", "user.email=synthetic@example.invalid", "commit",
                        "--allow-empty", "-m", "synthetic test baseline"], check=True,
                       capture_output=True, text=True)
        self.head = subprocess.run(["git", "-C", str(self.root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True).stdout.strip()
        self.layout = runner.make_layout(self.root, source_root=REPO,
                                         lock_path=self.temp / "shared.lock")
        self._create_inputs()

    def _create_inputs(self) -> None:
        layout = self.layout
        for path in [*layout.input_paths.values()]:
            path.parent.mkdir(parents=True, exist_ok=True)
        arrays_by_stage: dict[str, dict[str, np.ndarray]] = {}
        for index, stage in enumerate(runner.STAGES):
            fit_path, score_path, hinge_path = runner._stage_files(layout, stage)
            fit = _make_arrays(stage, "fit", physical_offset=100_000 + index * 100_000)
            score = _make_arrays(stage, "score", physical_offset=500_000 + index * 100_000)
            arrays_by_stage[stage] = {"fit": fit, "score": score}
            np.savez(fit_path, **fit)
            params = PARAMS[stage]
            score["raw_direct"] = score["direct_predictions"].copy()
            score["hinge_present_only"] = _hinge_present(
                score["direct_predictions"], score["missing_mask"], params)
            score["causal_global_all_rows"] = score["direct_predictions"].copy()
            score["causal_global_present_only"] = score["hinge_present_only"].copy()
            np.savez(score_path, **score)
            state = {"status": "LP_COMPLETE", "solver_result": {
                "success": True, **params}}
            _write_json(hinge_path, state)
            global_path = layout.input_paths["d1_global_state" if stage == "HINGE_D1" else "d2_global_state"]
            _write_json(global_path, {"synthetic_source_global": stage})

        exact_small = {"minimum_available_bytes": 2_147_483_648,
            "incremental_estimate_bytes": 536_870_912,
            "minimum_projected_free_bytes": 1_073_741_824,
            "minimum_actual_free_at_boundary_bytes": 1_073_741_824}
        exact_saved = {"minimum_available_bytes": 1_610_612_736,
            "incremental_estimate_bytes": 536_870_912,
            "minimum_projected_free_bytes": 805_306_368,
            "minimum_actual_free_at_boundary_bytes": 805_306_368}
        _write_json(layout.input_paths["resource_policy"], {"workloads": {
            "small_calibration_lp": exact_small,
            "saved_prediction_evaluation": exact_saved}})

        stage_records = []
        fit_replays = []
        score_replays = []
        input_hashes = {}
        for stage in runner.STAGES:
            fit_path, score_path, hinge_path = runner._stage_files(layout, stage)
            global_path = layout.input_paths["d1_global_state" if stage == "HINGE_D1" else "d2_global_state"]
            slug = "d1" if stage == "HINGE_D1" else "d2"
            relative = lambda path: str(path.relative_to(self.root)).replace("\\", "/")
            input_hashes[f"r3_stage__{stage[-2:]}_GLOBAL_state"] = {
                "path": str(global_path.resolve()), "sha256": runner.sha256_file(global_path)}
            stage_records.append({"fit_id": stage,
                "fit_input_path": relative(fit_path), "fit_input_sha256": runner.sha256_file(fit_path),
                "score_row_bundle_path": relative(score_path), "score_row_bundle_sha256": runner.sha256_file(score_path),
                "fit_state_path": relative(hinge_path), "fit_state_sha256": runner.sha256_file(hinge_path),
                "source_global_state_sha256": runner.sha256_file(global_path),
                "fit_rows": FIT_COUNTS[stage], "score_rows": SCORE_COUNTS[stage]})
            fit_replays.append({"fit_id": stage,
                "fit_input_sha256": runner.sha256_file(fit_path)})
            score_replays.append({"fit_id": stage,
                "score_bundle_sha256": runner.sha256_file(score_path)})

        identity_freeze = {"source_commit": self.head, "input_hashes": input_hashes}
        _write_json(layout.input_paths["r4_identity_freeze"], identity_freeze)
        _write_signed(layout.input_paths["r4_root_pass"], {
            "verdict": "PASS", "authorized_fit_ids": ["HINGE_D1", "HINGE_D2"],
            "prefit_freeze_sha256": runner.sha256_file(layout.input_paths["r4_identity_freeze"]),
            "source_commit": self.head})
        _write_json(layout.input_paths["r4_diagnostics"], {
            "experiment_id": "REG_MONOTONE_HINGE_P50_V1_R4",
            "status": "TWO_DIAGNOSTICS_COMPLETE_PENDING_ROOT_REVIEW",
            "test_or_sample_accessed": False, "stages": stage_records})
        _write_json(layout.input_paths["r4_verification"], {
            "experiment_id": "REG_MONOTONE_HINGE_P50_V1_R4",
            "status": "PASS_INDEPENDENT_DIAGNOSTICS_REPLAY",
            "test_or_sample_accessed": False,
            "diagnostics_sha256": runner.sha256_file(layout.input_paths["r4_diagnostics"]),
            "fit_replays": fit_replays, "score_replays": score_replays})


class TestClosedFormProperties(unittest.TestCase):
    def test_exact_median_intervals_bounds_support_and_objectives(self) -> None:
        self.assertEqual(runner.lower_weighted_median_interval([0.0, 1.0], [1.0, 1.0]),
                         (0.0, 1.0, 2))
        self.assertEqual(runner.lower_empirical_median_interval([-2.0, 4.0]),
                         (-2.0, 4.0, 2))
        self.assertEqual(runner._clip_interval(-3.0, -1.0), (-2.0, -1.0))

        p = np.full(200, 50.0, dtype="<f8")
        targets = np.concatenate((np.full(100, 49.0), np.full(100, 51.0)))
        result = runner.fit_local_delta(p, targets, np.ones(200, dtype=bool),
                                        {"intercept": 0.0, "slope_low": 1.0, "slope_high": 1.0})
        self.assertEqual(result["unbounded_median_interval"], [-1.0, 1.0])
        self.assertEqual(result["bounded_optimal_interval"], [-1.0, 0.0])
        self.assertEqual(result["delta"], -1.0)
        self.assertEqual(result["objective_unweighted_sum_absolute_unclipped"], 200.0)

        low_support = np.full(199, 50.0, dtype="<f8")
        with self.assertRaises(runner.PilotError):
            runner.fit_local_delta(low_support, np.full(199, 49.0), np.ones(199, dtype=bool),
                {"intercept": 0.0, "slope_low": 1.0, "slope_high": 1.0})

    def test_monotonicity_outside_zero_and_missing_bits(self) -> None:
        p = np.asarray([0.0, 40.0, 45.0, 50.0, 55.0, 60.0, 100.0, -0.0], dtype="<f8")
        missing = np.asarray([False, False, False, False, False, False, False, True])
        hinge = {"intercept": 2.0, "slope_low": 0.9, "slope_high": 0.95}
        record = runner.monotonicity_record(hinge, -2.0)
        self.assertTrue(record["monotone_non_decreasing"])
        prediction, raw = runner.apply_correction(p, missing, hinge, -1.0, local=True)
        self.assertTrue(np.array_equal(prediction[missing].view("<u8"), p[missing].view("<u8")))
        present = ~missing
        outside = np.abs(p[present] - 50.0) >= 10.0
        base = runner.base_hinge_unclipped(p[present], hinge)
        self.assertTrue(np.all((raw[present] - base)[outside] == 0.0))
        bad = runner.monotonicity_record({**hinge, "slope_low": 0.1}, -2.0)
        self.assertFalse(bad["monotone_non_decreasing"])


class TestSyntheticEndToEnd(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="center-local-r1-")
        self.synthetic = SyntheticLayout(self.temp)
        self.layout = self.synthetic.layout

    def tearDown(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def _run(action: str, layout: runner.Layout) -> dict[str, object]:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = runner.main([action], layout=layout)
        if result != 0:
            raise AssertionError(f"CLI action failed: {action}; {output.getvalue()}")
        return json.loads(output.getvalue())

    def _authorize_root(self) -> None:
        freeze_sha = runner.sha256_file(self.layout.freeze)
        preflight_sha = runner.sha256_file(self.layout.preflight)
        body = {"verdict": "PASS", "experiment_id": runner.NAMESPACE.upper() + "_R1",
            "pass_kind": "CENTER_LOCAL_DIAGNOSTIC", "source_commit": self.synthetic.head,
            "prefit_freeze_sha256": freeze_sha,
            "identity_preflight_sha256": preflight_sha,
            "authorized_fit_ids": list(runner.FIT_IDS), "calibrator_fit_cap": 4,
            "authorized_calibrator_fits": 4, "base_model_fits": 0,
            "preprocessing_fits": 0, "official_lp_starts": 0,
            "prior_three_slot_budget_reset": False,
            "score_target_access_authorized": True,
            "test_or_sample_authorized": False, "kaggle_upload_authorized": False}
        _write_signed(self.layout.root_pass, body)

    def test_cli_default_imported_globals_freeze_recovery_score_verify_and_tampering(self) -> None:
        default_output = io.StringIO()
        with contextlib.redirect_stdout(default_output):
            self.assertEqual(runner.main(["--paths"]), 0)
        paths = json.loads(default_output.getvalue())
        self.assertTrue(paths["root"].endswith("tk2-reg-position-failure-mechanism-v1"))

        with mock.patch.object(runner, "DEFAULT_LAYOUT", self.layout):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(runner.main(["--paths"]), 0)
            self.assertEqual(json.loads(output.getvalue())["root"], str(self.layout.root))
            self._run("--freeze", self.layout)
            runner.verify_freeze()
            self._run("--preflight", self.layout)
            runner.verify_preflight()

        # Low-support mutation is rejected before any fit objective can run.
        low_support = _make_arrays("HINGE_D1", "fit", physical_offset=1)
        low_support["direct_predictions"][:] = 10.0
        with self.assertRaises(runner.PilotError):
            runner._validate_population("HINGE_D1", "fit", low_support)

        self._authorize_root()
        original_append = runner._append_ledger
        original_loader = runner._read_npz_fields
        interrupted = {"done": False}
        fit_target_reads: list[Path] = []

        def interrupt_before_second_reservation(layout: runner.Layout, event: dict[str, object]) -> None:
            if event.get("fit_id") == "D1_GLOBAL_CONTROL" and event.get("event") == "RESERVE" and not interrupted["done"]:
                interrupted["done"] = True
                raise RuntimeError("synthetic interruption before second reservation")
            original_append(layout, event)

        def track_fit_target_reads(path: Path, fields: object, **kwargs: object) -> dict[str, np.ndarray]:
            if kwargs.get("load_targets"):
                fit_target_reads.append(Path(path).resolve())
            return original_loader(path, fields, **kwargs)  # type: ignore[arg-type]

        with mock.patch.object(runner, "_resource_snapshot", side_effect=lambda layout, workload, boundary, initial: {
                "status": "PASS", "workload": workload, "boundary": boundary,
                "available_bytes": 3_000_000_000, "projected_free_bytes": 2_500_000_000,
                "process_rss_bytes": 1_000_000, "threads": 2, "concurrent_heavy_jobs": 1}), \
             mock.patch.object(runner, "_append_ledger", side_effect=interrupt_before_second_reservation), \
             mock.patch.object(runner, "_read_npz_fields", side_effect=track_fit_target_reads):
            with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                runner.run_fits(self.layout)
        partial = runner._read_ledger(self.layout)
        self.assertEqual([event["event"] for event in partial["D1_LOCAL"]],
                         ["RESERVE", "FIT_STARTED", "FIT_COMPLETE"])
        self.assertNotIn("D1_GLOBAL_CONTROL", partial)

        with mock.patch.object(runner, "_resource_snapshot", side_effect=lambda layout, workload, boundary, initial: {
                "status": "PASS", "workload": workload, "boundary": boundary,
                "available_bytes": 3_000_000_000, "projected_free_bytes": 2_500_000_000,
                "process_rss_bytes": 1_000_000, "threads": 2, "concurrent_heavy_jobs": 1}), \
             mock.patch.object(runner, "_read_npz_fields", side_effect=track_fit_target_reads):
            fit_result = self._run("--fit", self.layout)
            self.assertEqual(fit_result["status"], "PASS_FOUR_CALIBRATORS_PREDICTIONS_FROZEN")
        expected_fit_paths = {runner._stage_files(self.layout, stage)[0].resolve()
                              for stage in runner.STAGES}
        self.assertTrue(fit_target_reads)
        self.assertTrue(set(fit_target_reads).issubset(expected_fit_paths))
        self.assertEqual(self._run("--score", self.layout)["status"],
                         "PASS_FOUR_FIT_SCORE_DIAGNOSTIC_COMPLETE")
        verified = self._run("--verify", self.layout)
        self.assertEqual(verified["status"], "PASS_INDEPENDENT_VERIFICATION")

        ledger = runner._read_ledger(self.layout)
        self.assertEqual(tuple(ledger), runner.FIT_IDS)
        self.assertTrue(all([item["event"] for item in ledger[fit_id]] ==
                            ["RESERVE", "FIT_STARTED", "FIT_COMPLETE"]
                            for fit_id in runner.FIT_IDS))
        self.assertTrue(self._run("--score", self.layout)["status"] == "EVALUATION_ALREADY_WRITTEN")
        self.assertEqual(verifier.verify_results(self.layout)["status"],
                         "PASS_INDEPENDENT_VERIFICATION")

        # A changed score label or row order fails on the frozen NPZ byte hash;
        # the score-target loader must not run on either rejected command.
        score_path = self.layout.input_paths["d1_score_rows"]
        original_bytes = score_path.read_bytes()
        original_loader = runner._read_npz_fields
        target_load_attempts = []

        def tracked_loader(path: Path, fields: object, **kwargs: object) -> dict[str, np.ndarray]:
            if kwargs.get("load_targets"):
                target_load_attempts.append(Path(path))
            return original_loader(path, fields, **kwargs)  # type: ignore[arg-type]

        try:
            with np.load(score_path, allow_pickle=False) as archive:
                changed = {name: np.asarray(archive[name]).copy() for name in archive.files}
            changed["targets"][0] = min(100.0, changed["targets"][0] + 0.25)
            with score_path.open("wb") as stream:
                np.savez(stream, **changed)
            with mock.patch.object(runner, "_read_npz_fields", side_effect=tracked_loader):
                with self.assertRaises(runner.PilotError):
                    runner.score_predictions(self.layout)
            self.assertEqual(target_load_attempts, [])

            score_path.write_bytes(original_bytes)
            with np.load(score_path, allow_pickle=False) as archive:
                reordered = {name: np.asarray(archive[name]).copy() for name in archive.files}
            reordered["record_ids"] = reordered["record_ids"][::-1].copy()
            with score_path.open("wb") as stream:
                np.savez(stream, **reordered)
            with mock.patch.object(runner, "_read_npz_fields", side_effect=tracked_loader):
                with self.assertRaises(runner.PilotError):
                    runner.score_predictions(self.layout)
            self.assertEqual(target_load_attempts, [])
        finally:
            score_path.write_bytes(original_bytes)


if __name__ == "__main__":
    unittest.main()
