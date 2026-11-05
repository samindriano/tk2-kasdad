"""Synthetic-only tests for the separately gated R1 final application."""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import reg_final_wave_v2 as v2
from scripts import reg_monotone_hinge_p50_final_application_v1 as runner
from scripts import reg_monotone_hinge_p50_v4 as r4
from scripts import verify_reg_monotone_hinge_p50_final_application_v1 as verifier


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _forbidden_target_reader(source_root: Path, selected_rows: np.ndarray,
                             expected_ids: np.ndarray, expected_cycles: np.ndarray,
                             expected_total_rows: int):
    raise AssertionError("preflight may not parse Train targets")


def _signed(body: dict[str, object]) -> dict[str, object]:
    return {**body, "signature": r4.canonical_sha(body)}


def _resource(*, boundary: str, initial: bool = False) -> dict[str, object]:
    return {
        "status": "PASS", "boundary": boundary,
        "available_bytes_measured": 4 * 1024**3,
        "initial_floor_pass": True, "minimum_initial_available_bytes": 2 * 1024**3,
        "incremental_estimate_bytes": 512 * 1024**2,
        "minimum_projected_free_bytes": 1 * 1024**3,
        "actual_available_boundary_minimum_bytes": 1 * 1024**3,
        "projected_free_bytes": 4 * 1024**3 - 512 * 1024**2,
        "projected_free_pass": True, "actual_boundary_pass": True,
        "estimated_peak_is_measured": False, "process_rss_bytes": 128 * 1024**2,
        "process_rss_measured": True, "os_peak_working_set_bytes": 256 * 1024**2,
        "os_peak_working_set_measured": True, "os_peak_working_set_scope": "process_lifetime",
        "threads": 2, "concurrent_heavy_jobs": 1,
    }


def _identity() -> tuple[dict[str, object], np.ndarray]:
    n = runner.EXPECTED_OOF_ROWS
    rows = np.arange(100_000, 100_000 + n, dtype=np.int64)
    split = 12_412
    origins = np.concatenate((np.full(split, 67, dtype=np.int8),
                              np.full(n - split, 75, dtype=np.int8)))
    cycles = np.concatenate((np.resize(np.arange(68, 76, dtype=np.int16), split),
                             np.resize(np.arange(76, 80, dtype=np.int16), n - split)))
    indices = np.arange(n, dtype=np.int64)
    direct = np.linspace(0.0, 100.0, n, dtype="<f8")
    missing = np.zeros(n, dtype=bool)
    missing[::101] = True
    ids = np.asarray([f"synthetic-train-{i:05d}" for i in range(n)])
    x = np.arange(n, dtype=np.float64)
    targets = np.clip(7.0 + 0.31 * np.minimum(direct, 50.0)
                      + 0.74 * np.maximum(direct - 50.0, 0.0)
                      + 1.5 * np.sin(x / 27.0), 0.0, 100.0).astype("<f8")
    identity: dict[str, object] = {
        "rows": rows, "ids": ids, "origins": origins, "indices": indices,
        "cycles": cycles, "direct": direct, "missing": missing,
        "row_counts": {"unique_oof": n, "E67_68_75": split,
                       "E75_76_79": n - split},
    }
    return identity, targets


def _e79_fixture(folder: Path) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, dict[str, object], dict[str, object], Path]:
    n = runner.EXPECTED_TEST_ROWS
    ids = np.asarray([f"synthetic-test-{i:05d}" for i in range(n)])
    raw = np.linspace(0.0, 100.0, n, dtype="<f8")
    missing = np.zeros(n, dtype=bool)
    missing[:runner.EXPECTED_MISSING_ROWS] = True
    baseline = raw.copy()
    archive_path = folder / "inference_B.npz"
    np.savez_compressed(archive_path, record_id=ids, prediction=raw,
                        baseline=baseline, missing=missing)
    manifest: dict[str, object] = {
        "candidate_id": "DIRECT_POSITION_REAL_E79",
        "ordered_official_test_ids_sha256": v2.sha_ordered_ids(ids),
        "prediction_sha256": v2.sha_f64_bytes(raw),
        "prediction_rows": n, "missing_rows": int(missing.sum()),
        "test_labels_loaded": False, "upload_performed": False,
    }
    qa: dict[str, object] = {
        "status": "PASS_INDEPENDENT_CSV_QA", "test_labels_read": False,
        "missing_rows": int(missing.sum()),
    }
    return ids, raw, missing, manifest, qa, archive_path


def _make_r4_ledger(folder: Path) -> tuple[Path, dict[str, Path]]:
    ledger_path = folder / "lp_ledger.jsonl"
    state_paths: dict[str, Path] = {}
    events = []
    for fit_id in runner.PRIOR_DIAGNOSTIC_FIT_IDS:
        reservation_id = f"synthetic-{fit_id.lower()}-reservation"
        state_path = folder / f"{fit_id}.json"
        state = {"fit_id": fit_id, "status": "LP_COMPLETE",
                 "fit_input": {"root_source_pass_sha256": runner.R4_ROOT_SOURCE_PASS_SHA256}}
        state_path.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
        state_paths[fit_id] = state_path
        common = {"fit_id": fit_id, "reservation_id": reservation_id,
                  "fit_input": state["fit_input"], "source_commit": "r4-synthetic",
                  "root_source_pass_sha256": runner.R4_ROOT_SOURCE_PASS_SHA256}
        events.extend([
            {**common, "event": "RESERVE", "attempt": 1, "retry_allowed": False},
            {**common, "event": "LP_STARTED", "retry_allowed": False},
            {**common, "event": "LP_COMPLETE", "state_sha256": _sha(state_path)},
        ])
    ledger_path.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in events),
                           encoding="utf-8")
    return ledger_path, state_paths


class FinalApplicationR1Tests(unittest.TestCase):
    def _scenario(self, stack: ExitStack, folder: Path) -> dict[str, object]:
        identity, targets = _identity()
        ids, raw, missing, manifest, qa, archive_path = _e79_fixture(folder)
        ledger_path, prior_states = _make_r4_ledger(folder)
        prefinal_ledger_sha = _sha(ledger_path)

        # Persist signed metadata used by the independent application replay.
        manifest_path, qa_path = folder / "INFERENCE_MANIFEST_B.json", folder / "CSV_QA_B.json"
        manifest_signed = _signed(manifest)
        qa_signed = _signed(qa)
        manifest_path.write_text(json.dumps(manifest_signed, sort_keys=True), encoding="utf-8")
        qa_path.write_text(json.dumps(qa_signed, sort_keys=True), encoding="utf-8")
        r4_freeze = json.loads(r4.PREFIT_FREEZE.read_text(encoding="utf-8"))
        r4_pins = runner._expected_r4_pins()
        inputs: dict[str, dict[str, str]] = {
            name: {"path": str(path.resolve()), "sha256": digest}
            for name, (path, digest) in r4_pins.items()
        }
        # The synthetic cumulative ledger and D1/D2 state files replace only
        # their corresponding production artifacts inside this isolated test.
        inputs["r4_lp_ledger"] = {"path": str(ledger_path.resolve()),
                                  "sha256": prefinal_ledger_sha}
        for fit_id, state_path in prior_states.items():
            inputs[f"r4_fit_state_{fit_id}"] = {
                "path": str(state_path.resolve()), "sha256": _sha(state_path)}
        old_keys = {
            "e79_trained_state_freeze": "r3_input__pinned_base_trained_freeze",
            "e79_prelabel_receipt": "r3_input__pinned_base_prelabel",
            "direct_freeze": "r3_input__pinned_direct_freeze",
            "direct_prelabel_receipt": "r3_input__pinned_direct_prelabel",
            "direct_E67": "r3_input__pinned_direct_E67",
            "direct_E67_receipt": "r3_input__pinned_direct_E67_receipt",
            "direct_E75": "r3_input__pinned_direct_E75",
            "direct_E75_receipt": "r3_input__pinned_direct_E75_receipt",
            "user_revised_resource_policy": "user_revised_resource_policy",
        }
        for new_name, old_name in old_keys.items():
            item = r4_freeze["input_hashes"][old_name]
            inputs[new_name] = {"path": str(Path(item["path"]).resolve()),
                                "sha256": item["sha256"]}
        inputs["e79_archive"] = {"path": str(archive_path.resolve()), "sha256": _sha(archive_path)}
        inputs["e79_manifest"] = {"path": str(manifest_path.resolve()), "sha256": _sha(manifest_path)}
        inputs["e79_qa"] = {"path": str(qa_path.resolve()), "sha256": _sha(qa_path)}
        train_pin = r4_freeze["input_hashes"]["r3_input__official_train"]
        inputs["official_train_deferred"] = {
            "path": str(Path(train_pin["path"]).resolve()), "sha256": train_pin["sha256"]}
        inputs["r4_source_commit"] = {
            "path": str(r4.PREFIT_FREEZE.resolve()),
            "sha256": str(r4_freeze.get("source_commit"))}
        inputs["r4_freeze_sha256"] = {
            "path": str(r4.PREFIT_FREEZE.resolve()), "sha256": _sha(r4.PREFIT_FREEZE)}
        inputs["r4_source_pass_sha256"] = {
            "path": str(r4.ROOT_SOURCE_PASS.resolve()), "sha256": _sha(r4.ROOT_SOURCE_PASS)}
        preflight_hashes = {name: inputs[name] for name in runner.PREFLIGHT_REHASH_KEYS}

        source_hashes = {"synthetic-source-fixture": "a" * 64}
        decision = {
            "scientific_qualification": "FAIL",
            "diagnostic_status": "EXPLORATORY_ROOT_REVIEW",
            "pooled_gain_vs_causal_global_present_only": 0.006360119,
            "qualified_threshold": 0.02, "qualified_gate": False,
            "exploratory_review_gate": True,
            "window_gains": {"HINGE_D1": 0.007355896, "HINGE_D2": 0.005485018},
            "window_p90_absolute_error_worsening": {
                "HINGE_D1": 0.01998969392750638,
                "HINGE_D2": 0.017396558419765284},
            "upper_tail_target_ge_95_harm": {
                "HINGE_D1": 0.1324871651977233,
                "HINGE_D2": 0.02145241117350949},
            "maximum_supported_subgroup_harm": 0.1324871651977233,
            "upload_authorized": False,
            "user_batch_preference": "collect several research results before selecting a benchmark batch",
        }
        prior_slots = {
            "ledger_path": str(ledger_path),
            "ledger_sha256_before_final_slot": prefinal_ledger_sha,
            "completed_fit_ids": list(runner.PRIOR_DIAGNOSTIC_FIT_IDS),
            "completed_starts": 2,
            "r4_diagnostic_fit_state_sha256": {k: _sha(v) for k, v in prior_states.items()},
            "next_fit_id": runner.FIT_ID, "final_slot_reserved": False,
            "reset_or_retry": False,
        }
        freeze = _signed({
            "schema_version": 1, "experiment_id": runner.EXPERIMENT_ID,
            "status": "FINAL_APPLICATION_FROZEN_PENDING_IDENTITY_PREFLIGHT",
            "source_commit": subprocess.run(
                ["git", "-C", str(runner.ROOT), "rev-parse", "HEAD"],
                check=True, capture_output=True, text=True).stdout.strip(),
            "source_hashes": source_hashes,
            "input_hashes": inputs, "preflight_input_hashes": preflight_hashes,
            "preflight_input_allowlist": sorted(preflight_hashes),
            "r4_predecessor": {
                "freeze_path": str(r4.PREFIT_FREEZE.resolve()),
                "freeze_sha256": _sha(r4.PREFIT_FREEZE),
                "source_pass_path": str(r4.ROOT_SOURCE_PASS.resolve()),
                "source_pass_sha256": _sha(r4.ROOT_SOURCE_PASS),
                "identity_preflight_sha256": runner.R4_PREFLIGHT_SHA256,
                "diagnostics_sha256": runner.R4_DIAGNOSTICS_SHA256,
                "independent_verification_sha256": runner.R4_VERIFICATION_SHA256,
                "decision_path": str(runner.DECISION_DOC.resolve()),
                "decision_sha256": _sha(runner.DECISION_DOC),
            },
            "prior_slot_accounting": prior_slots,
            "deferred_target_input": inputs["official_train_deferred"],
            "decision_context": decision,
            "official_lp_starts": 0,
            "train_targets_read": False,
            "train_target_source_hash_verified": True,
            "test_or_sample_read": False,
            "csv_created": False,
        })
        freeze_path = folder / "FINAL_APPLICATION_FREEZE_R1.json"
        freeze_path.write_text(json.dumps(freeze, sort_keys=True), encoding="utf-8")
        freeze_sha = _sha(freeze_path)

        e79_audit = {
            "archive_path": str(archive_path), "archive_sha256": _sha(archive_path),
            "loaded_npz_members": list(runner.E79_NPZ_ALLOWLIST),
            "unloaded_extra_member_names": [], "rows": len(ids), "unique_ids": len(ids),
            "ordered_ids_sha256": v2.sha_ordered_ids(ids),
            "raw_direct_sha256": r4.r1.tagged_array_sha256(raw, dtype="<f8"),
            "raw_direct_sha256_bytes": v2.sha_f64_bytes(raw),
            "missing_rows": int(missing.sum()),
            "missing_route_bit_exact_to_baseline": True,
            "target_like_members_loaded": False, "csv_read": False,
            "test_or_sample_read": False,
        }
        preflight = _signed({
            "schema_version": 1, "experiment_id": runner.EXPERIMENT_ID,
            "status": "PASS_FINAL_APPLICATION_IDENTITY_PREFLIGHT",
            "source_commit": freeze["source_commit"], "source_hashes": source_hashes,
            "input_hashes": inputs, "prefit_freeze_sha256": freeze_sha,
            "prior_slot_accounting": {
                "ledger_sha256": prefinal_ledger_sha,
                "fit_ids": list(runner.PRIOR_DIAGNOSTIC_FIT_IDS),
                "starts": 2, "complete_events": 2, "failed_events": 0,
                "hinge_final_reserved": False, "next_fit_id": runner.FIT_ID,
            },
            "oof_identity": {
                "row_identity_hashes": {}, "row_counts": identity["row_counts"],
                "unique_rows": runner.EXPECTED_OOF_ROWS,
                "selected_cycles": [68, 79], "targets_loaded": False,
            },
            "e79_raw_direct_audit": e79_audit,
            "api_contracts": {name: {} for name in (
                "selected_train_row_reader", "oof_identity_loader",
                "oof_identity_validator", "final_solver", "csv_writer",
                "independent_csv_verifier", "base_identity_manifest")},
            "resource_snapshots": [
                {"boundary": "final_application_preflight_locked_initial"},
                {"boundary": "final_application_preflight_after_oof_identity"},
                {"boundary": "final_application_preflight_after_e79_identity"}],
            "target_reader_called": False, "train_targets_parsed": False,
            "train_target_source_hash_verified": True,
            "test_or_sample_reader_called": False, "test_or_sample_read": False,
            "lp_reserved": False, "official_lp_starts": 0,
            "ledger_created_or_modified": False, "csv_created": False,
            "kaggle_upload": False,
        })
        preflight_path = folder / "FINAL_APPLICATION_PREFLIGHT_R1.json"
        preflight_path.write_text(json.dumps(preflight, sort_keys=True), encoding="utf-8")
        preflight_sha = _sha(preflight_path)
        approval = _signed({
            "verdict": "PASS", "pass_kind": "FINAL_APPLICATION",
            "experiment_id": runner.EXPERIMENT_ID,
            "source_commit": freeze["source_commit"], "source_hashes": source_hashes,
            "input_hashes": inputs, "prefit_freeze_sha256": freeze_sha,
            "identity_preflight_sha256": preflight_sha,
            "r4_diagnostic_source_pass_sha256": runner.R4_ROOT_SOURCE_PASS_SHA256,
            "r4_diagnostics_sha256": runner.R4_DIAGNOSTICS_SHA256,
            "authorized_fit_ids": [runner.FIT_ID], "authorized_final_lp_starts": 1,
            "prior_official_lp_starts": 2, "total_official_lp_start_cap": 3,
            "authorized_csv_creation": True,
            "test_or_sample_access_authorized": False,
            "kaggle_upload_authorized": False,
        })
        pass_path = folder / "FINAL_APPLICATION_ROOT_SOURCE_PASS_R1.json"
        pass_path.write_text(json.dumps(approval, sort_keys=True), encoding="utf-8")
        approval_sha = _sha(pass_path)

        target_audit = {
            "sha256": inputs["official_train_deferred"]["sha256"],
            "query_target_rows_parsed": runner.EXPECTED_OOF_ROWS,
            "fit_target_rows_parsed": 0,
        }

        # Redirect all mutable application outputs into the temporary fixture.
        redirects = {
            "RUN_DIR": folder / "run",
            "LEDGER": ledger_path,
            "OUTPUT_CSV": folder / "submission.csv",
            "FINAL_REPORT": folder / "FINAL_APPLICATION_REPORT_R1.json",
            "FINAL_VERIFICATION": folder / "FINAL_APPLICATION_VERIFICATION_R1.json",
            "FINAL_RECEIPT": folder / "FINAL_APPLICATION_RECEIPT_R1.json",
            "FREEZE": freeze_path, "PREFLIGHT": preflight_path,
            "ROOT_SOURCE_PASS": pass_path, "REPORT_DIR": folder,
            "SUBMISSION_DIR": folder, "SHARED_JOB_LOCK": folder / "heavy.lock",
            "R4_FINAL_STATE_PATHS": prior_states,
        }
        for name, value in redirects.items():
            stack.enter_context(patch.object(runner, name, value))
        stack.enter_context(patch.object(runner, "source_hashes", return_value=source_hashes))
        stack.enter_context(patch.object(runner, "verify_final_source_pass",
            return_value=(freeze, approval, approval_sha)))
        stack.enter_context(patch.object(runner, "_read_preflight",
            return_value=(preflight, preflight_sha)))
        stack.enter_context(patch.object(runner, "_validate_current_r4_and_ledger",
            side_effect=lambda _freeze, allow_final=False:
                (r4_freeze, r4.read_ledger(ledger_path))))
        stack.enter_context(patch.object(runner, "_identity_for_final_application",
            return_value=identity))
        stack.enter_context(patch.object(runner, "_load_e79_raw_direct",
            return_value=(ids, raw, missing, manifest_signed, qa_signed, e79_audit)))
        stack.enter_context(patch.object(runner, "_load_train_targets_after_new_pass",
            return_value=(targets, target_audit)))
        stack.enter_context(patch.object(r4, "load_oof_identity", return_value=identity))
        stack.enter_context(patch.object(r4, "verify_oof_identity", return_value={}))
        stack.enter_context(patch.object(r4, "resource_snapshot", side_effect=_resource))
        stack.enter_context(patch.object(v2, "one_job_gate", return_value=None))
        stack.enter_context(patch.object(v2, "verify_base",
            return_value=(manifest_signed, qa_signed, {})))

        return {
            "identity": identity, "targets": targets, "ids": ids, "raw": raw,
            "missing": missing, "manifest": manifest_signed, "qa": qa_signed,
            "freeze": freeze, "preflight": preflight, "approval": approval,
            "approval_sha": approval_sha, "freeze_sha": freeze_sha,
            "pass_path": pass_path, "prefinal_ledger_sha": prefinal_ledger_sha,
            "target_audit": target_audit,
        }

    def test_highest_level_synthetic_final_application_e2e(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hinge-final-r1-test-") as tmp:
            folder = Path(tmp)
            with ExitStack() as stack:
                fixture = self._scenario(stack, folder)
                result = runner.run_final_application()
                self.assertEqual(result["status"],
                                 "EXPLORATORY_CANDIDATE_PENDING_BATCH_SELECTION")
                self.assertEqual(result["verification_status"],
                                 "PASS_INDEPENDENT_FINAL_APPLICATION_REPLAY")
                self.assertEqual(result["official_lp_starts_total"], 3)
                _verified_freeze, _verified_freeze_sha = runner.verify_final_freeze()
                ledger = r4.read_ledger(runner.LEDGER)
                self.assertEqual(set(ledger), {"HINGE_D1", "HINGE_D2", "HINGE_FINAL"})
                self.assertEqual(len(ledger["HINGE_FINAL"]["LP_STARTED"]), 1)
                self.assertEqual(ledger["HINGE_FINAL"]["LP_FAILED"], [])

                report = json.loads(runner.FINAL_REPORT.read_text(encoding="utf-8"))
                self.assertEqual(report["csv_rows"], runner.EXPECTED_TEST_ROWS)
                self.assertEqual(report["missing_rows"], runner.EXPECTED_MISSING_ROWS)
                self.assertIs(report["test_or_sample_read"], False)
                self.assertIs(report["kaggle_upload"], False)
                self.assertEqual(report["diagnostic_decision_context"],
                                 fixture["freeze"]["decision_context"])
                original_report_bytes = runner.FINAL_REPORT.read_bytes()
                wrong_csv_report = dict(report)
                wrong_csv_report["csv_path"] = str(folder / "alternate.csv")
                wrong_csv_report.pop("signature", None)
                wrong_csv_report["signature"] = r4.canonical_sha(wrong_csv_report)
                runner.FINAL_REPORT.write_text(
                    json.dumps(wrong_csv_report, sort_keys=True), encoding="utf-8")
                with self.assertRaises(verifier.VerificationError):
                    verifier.verify_final_application(
                        report_dir=folder, freeze=fixture["freeze"],
                        approval_sha=fixture["approval_sha"],
                        report_path=runner.FINAL_REPORT)
                runner.FINAL_REPORT.write_bytes(original_report_bytes)
                ledger_bytes = Path(runner.LEDGER).read_bytes()
                raw_lines = ledger_bytes.splitlines(keepends=True)
                first_event = json.loads(raw_lines[0])
                first_event["unapproved_prefix_mutation"] = True
                raw_lines[0] = json.dumps(first_event, sort_keys=True).encode("utf-8") + b"\n"
                Path(runner.LEDGER).write_bytes(b"".join(raw_lines))
                with self.assertRaises(verifier.VerificationError):
                    verifier.verify_final_application(
                        report_dir=folder, freeze=fixture["freeze"],
                        approval_sha=fixture["approval_sha"],
                        report_path=runner.FINAL_REPORT)

    def test_reserved_unstarted_slot_revalidates_bundle_and_targets(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hinge-final-r1-recovery-") as tmp:
            folder = Path(tmp)
            with ExitStack() as stack:
                fixture = self._scenario(stack, folder)
                identity = fixture["identity"]
                targets = fixture["targets"]
                arrays = runner._fit_arrays(identity, targets)
                input_path = runner.RUN_DIR / "fit_inputs" / f"{runner.FIT_ID}.npz"
                bundle_sha = runner._save_npz_once(input_path, arrays)
                expected = runner._fit_input_identity(
                    arrays, bundle_sha, fixture["freeze"], fixture["freeze_sha"],
                    fixture["approval_sha"])
                r4.append_ledger(runner.LEDGER, {
                    "event": "RESERVE", "fit_id": runner.FIT_ID,
                    "reservation_id": "synthetic-existing-reservation",
                    "fit_input": expected,
                    "source_commit": fixture["freeze"]["source_commit"],
                    "root_source_pass_sha256": fixture["approval_sha"],
                    "resource": _resource(boundary="synthetic_reserve"),
                    "attempt": 1, "retry_allowed": False,
                })
                with patch.object(runner, "_load_train_targets_after_new_pass",
                                  return_value=(targets, fixture["target_audit"])) as reader:
                    result = runner.run_final_application()
                self.assertEqual(result["verification_status"],
                                 "PASS_INDEPENDENT_FINAL_APPLICATION_REPLAY")
                self.assertEqual(reader.call_count, 1)
                ledger = r4.read_ledger(runner.LEDGER)
                self.assertEqual(len(ledger[runner.FIT_ID]["RESERVE"]), 1)
                self.assertEqual(len(ledger[runner.FIT_ID]["LP_STARTED"]), 1)
                self.assertEqual(ledger[runner.FIT_ID]["RESERVE"][0]["fit_input"], expected)

    def test_reserved_slot_non_target_mismatch_fails_before_lp_start(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hinge-final-r1-bad-rows-") as tmp:
            folder = Path(tmp)
            with ExitStack() as stack:
                fixture = self._scenario(stack, folder)
                arrays = runner._fit_arrays(fixture["identity"], fixture["targets"])
                arrays["cycles"][0] = np.int16(arrays["cycles"][0] + 1)
                input_path = runner.RUN_DIR / "fit_inputs" / f"{runner.FIT_ID}.npz"
                bundle_sha = runner._save_npz_once(input_path, arrays)
                expected = runner._fit_input_identity(
                    arrays, bundle_sha, fixture["freeze"], fixture["freeze_sha"],
                    fixture["approval_sha"])
                r4.append_ledger(runner.LEDGER, {
                    "event": "RESERVE", "fit_id": runner.FIT_ID,
                    "reservation_id": "synthetic-mismatched-row-reservation",
                    "fit_input": expected,
                    "source_commit": fixture["freeze"]["source_commit"],
                    "root_source_pass_sha256": fixture["approval_sha"],
                    "resource": _resource(boundary="synthetic_reserve"),
                    "attempt": 1, "retry_allowed": False,
                })
                with patch.object(runner, "_load_train_targets_after_new_pass") as reader, \
                     patch.object(r4, "solve_hinge_l1") as solver:
                    with self.assertRaises(runner.GateError):
                        runner.run_final_application()
                self.assertEqual(reader.call_count, 0)
                self.assertEqual(solver.call_count, 0)
                self.assertEqual(r4.read_ledger(runner.LEDGER)[runner.FIT_ID]["LP_STARTED"], [])

    def test_reserved_slot_target_mismatch_fails_before_lp_start(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hinge-final-r1-bad-targets-") as tmp:
            folder = Path(tmp)
            with ExitStack() as stack:
                fixture = self._scenario(stack, folder)
                arrays = runner._fit_arrays(fixture["identity"], fixture["targets"])
                input_path = runner.RUN_DIR / "fit_inputs" / f"{runner.FIT_ID}.npz"
                bundle_sha = runner._save_npz_once(input_path, arrays)
                expected = runner._fit_input_identity(
                    arrays, bundle_sha, fixture["freeze"], fixture["freeze_sha"],
                    fixture["approval_sha"])
                r4.append_ledger(runner.LEDGER, {
                    "event": "RESERVE", "fit_id": runner.FIT_ID,
                    "reservation_id": "synthetic-mismatched-target-reservation",
                    "fit_input": expected,
                    "source_commit": fixture["freeze"]["source_commit"],
                    "root_source_pass_sha256": fixture["approval_sha"],
                    "resource": _resource(boundary="synthetic_reserve"),
                    "attempt": 1, "retry_allowed": False,
                })
                bad_targets = fixture["targets"].copy()
                bad_targets[0] = np.nextafter(bad_targets[0], np.float64(100.0))
                with patch.object(runner, "_load_train_targets_after_new_pass",
                                  return_value=(bad_targets, fixture["target_audit"])) as reader, \
                     patch.object(r4, "solve_hinge_l1") as solver:
                    with self.assertRaises(runner.GateError):
                        runner.run_final_application()
                self.assertEqual(reader.call_count, 1)
                self.assertEqual(solver.call_count, 0)
                self.assertEqual(r4.read_ledger(runner.LEDGER)[runner.FIT_ID]["LP_STARTED"], [])

    def test_preflight_hashes_train_bytes_without_calling_target_reader(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hinge-final-r1-preflight-") as tmp:
            folder = Path(tmp)
            with ExitStack() as stack:
                real_read_preflight = runner._read_preflight
                fixture = self._scenario(stack, folder)
                missing_pass = folder / "not-created-root-pass.json"
                preflight_path = folder / "preflight-output.json"
                stack.enter_context(patch.object(runner, "ROOT_SOURCE_PASS", missing_pass))
                stack.enter_context(patch.object(runner, "PREFLIGHT", preflight_path))
                stack.enter_context(patch.object(runner, "verify_final_freeze",
                    return_value=(fixture["freeze"], fixture["freeze_sha"])))
                stack.enter_context(patch.object(runner, "_verify_r4_predecessor_with_all_pins",
                    return_value=(json.loads(r4.PREFIT_FREEZE.read_text(encoding="utf-8")),
                                  {}, runner.R4_ROOT_SOURCE_PASS_SHA256)))
                def synthetic_oof_loader(freeze: dict[str, object]):
                    return fixture["identity"]
                def synthetic_oof_validator(identity: dict[str, object],
                                            expected: dict[str, object]):
                    return {}
                def synthetic_base_verifier():
                    return fixture["manifest"], fixture["qa"], {}
                stack.enter_context(patch.object(r4, "load_oof_identity",
                                                  new=synthetic_oof_loader))
                stack.enter_context(patch.object(r4, "verify_oof_identity",
                                                  new=synthetic_oof_validator))
                stack.enter_context(patch.object(v2, "verify_base",
                                                  new=synthetic_base_verifier))
                calls = {"count": 0}
                def forbidden_target_reader(source_root: Path, selected_rows: np.ndarray,
                        expected_ids: np.ndarray, expected_cycles: np.ndarray,
                        expected_total_rows: int):
                    calls["count"] += 1
                    return _forbidden_target_reader(source_root, selected_rows,
                        expected_ids, expected_cycles, expected_total_rows)
                with patch.object(runner, "read_scored_query", new=forbidden_target_reader):
                    result = runner.identity_preflight()
                self.assertEqual(result["status"],
                                 "PASS_FINAL_APPLICATION_IDENTITY_PREFLIGHT")
                self.assertEqual(calls["count"], 0)
                preflight, preflight_sha = real_read_preflight(
                    fixture["freeze"], fixture["freeze_sha"])
                self.assertEqual(preflight_sha, result["identity_preflight_sha256"])
                self.assertIs(preflight["train_targets_parsed"], False)
                self.assertIs(preflight["train_target_source_hash_verified"], True)
                self.assertEqual(
                    fixture["freeze"]["preflight_input_hashes"]["official_train_deferred"],
                    fixture["freeze"]["input_hashes"]["official_train_deferred"])

    def test_r4_diagnostic_pass_cannot_authorize_final_fit(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hinge-final-r1-auth-") as tmp:
            with ExitStack() as stack:
                fixture = self._scenario(stack, Path(tmp))
                bad = {key: value for key, value in fixture["approval"].items()
                       if key != "signature"}
                bad["pass_kind"] = "R4_DIAGNOSTICS"
                bad["signature"] = r4.canonical_sha(bad)
                with self.assertRaises(runner.GateError):
                    runner.validate_final_source_pass_body(
                        bad, fixture["freeze"], fixture["freeze_sha"],
                        _sha(runner.PREFLIGHT))

    def test_independent_csv_replay_rejects_reorder_and_missing_bit_change(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hinge-final-r1-csv-qa-") as tmp:
            path = Path(tmp) / "synthetic.csv"
            ids = np.array(["a", "b", "c"])
            raw = np.array([-0.0, 10.0, 20.0], dtype="<f8")
            missing = np.array([True, False, False])
            values = raw.copy()

            def write(rows: list[tuple[str, str]]) -> None:
                with path.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream, lineterminator="\n")
                    writer.writerow(("record_id", "threat_rank"))
                    writer.writerows(rows)

            write([(item, format(float(value), ".17g")) for item, value in zip(ids, values)])
            verifier.r4verifier.verify_candidate_csv(
                path, expected_ids=ids, expected_values=values,
                missing_mask=missing, raw_direct=raw, production=False)
            rows = [("b", "10"), ("a", "-0"), ("c", "20")]
            write(rows)
            with self.assertRaises(r4.GateError):
                verifier.r4verifier.verify_candidate_csv(
                    path, expected_ids=ids, expected_values=values,
                    missing_mask=missing, raw_direct=raw, production=False)
            rows = [("a", "0"), ("b", "10"), ("c", "20")]
            write(rows)
            with self.assertRaises(r4.GateError):
                verifier.r4verifier.verify_candidate_csv(
                    path, expected_ids=ids, expected_values=values,
                    missing_mask=missing, raw_direct=raw, production=False)

    def test_application_route_preserves_missing_bits_and_clips_present(self) -> None:
        raw = np.array([-0.0, 25.0, 60.0, 100.0], dtype="<f8")
        missing = np.array([True, False, False, True])
        params = {"intercept": -3.0, "slope_low": 1.1, "slope_high": 0.8}
        got = runner._apply_hinge(raw, missing, params)
        self.assertEqual(got[0].view("<u8"), raw[0].view("<u8"))
        self.assertEqual(got[3].view("<u8"), raw[3].view("<u8"))
        self.assertTrue(np.all((got[~missing] >= 0.0) & (got[~missing] <= 100.0)))


if __name__ == "__main__":
    unittest.main()
