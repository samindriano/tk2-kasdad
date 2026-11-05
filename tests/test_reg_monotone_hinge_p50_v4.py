"""Synthetic-only regression tests for the R4 monotone hinge pipeline."""

from __future__ import annotations

import unittest
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

from scripts import reg_monotone_hinge_p50_v4 as r2
from scripts import reg_final_wave_v2 as direct_helpers
from scripts import verify_reg_monotone_hinge_p50_v4 as verifier


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


def _small_identity() -> tuple[dict[str, object], dict[str, object]]:
    identity: dict[str, object] = {
        "rows": np.array([10, 11, 12, 13], dtype=np.int64),
        "ids": np.array(["syn-a", "syn-b", "syn-c", "syn-d"]),
        "origins": np.array([67, 67, 67, 75], dtype=np.int8),
        "indices": np.array([0, 1, 2, 3], dtype=np.int64),
        "cycles": np.array([68, 71, 72, 76], dtype=np.int16),
        "direct": np.array([10.0, 20.0, 30.0, 40.0], dtype="<f8"),
        "missing": np.array([False, True, False, False]),
        "row_counts": {"unique_oof": 4, "E67_68_75": 3, "E75_76_79": 1},
    }
    fingerprints = {
        "rows_sha256": r2.r1.tagged_array_sha256(identity["rows"]),
        "ids_sha256": r2.r1.tagged_array_sha256(identity["ids"]),
        "origins_sha256": r2.r1.tagged_array_sha256(identity["origins"]),
        "original_bundle_indices_sha256": r2.r1.tagged_array_sha256(identity["indices"]),
        "cycles_sha256": r2.r1.tagged_array_sha256(identity["cycles"]),
        "direct_prediction_sha256": r2.r1.tagged_array_sha256(identity["direct"], dtype="<f8"),
        "missing_mask_sha256": r2.r1.tagged_array_sha256(identity["missing"]),
    }
    identity["row_identity_hashes"] = fingerprints
    expected = {**fingerprints, "row_counts": dict(identity["row_counts"])}
    return identity, expected


def _synthetic_direct_fixture() -> tuple[dict[int, dict[str, np.ndarray]], dict[str, object],
                                          dict[str, object], dict[str, object]]:
    cycle_counts = {
        68: 1238, 69: 1238, 70: 1237, 71: 1237,
        72: 1866, 73: 1866, 74: 1865, 75: 1865,
        76: 2123, 77: 2123, 78: 2123, 79: 2122,
    }
    cycles67 = np.concatenate([
        np.full(count, cycle, dtype=np.int16) for cycle, count in cycle_counts.items()])
    rows67 = np.arange(len(cycles67), dtype=np.int64)
    ids67 = np.asarray([f"SYN-{row:05d}" for row in rows67], dtype=str)

    def bundle(origin: int, fit_count: int, rows: np.ndarray, ids: np.ndarray,
               cycles: np.ndarray) -> dict[str, np.ndarray]:
        prediction = ((rows * 37) % 10001).astype("<f8") / 100.0
        missing = rows % 19 == 0
        return {
            "origin": np.asarray([origin], dtype=np.int16),
            "arm": np.asarray(["REAL"]),
            "fit_rows": np.arange(fit_count, dtype=np.int64),
            "query_rows": np.asarray(rows, dtype=np.int64),
            "query_record_ids": np.asarray(ids, dtype=str),
            "query_cycles": np.asarray(cycles, dtype=np.int16),
            "baseline": prediction,
            "raw_auxiliary": prediction.copy(),
            "mapped_auxiliary": prediction.copy(),
            "prediction": prediction.copy(),
            "missing_mask": np.asarray(missing, dtype=bool),
        }

    e67 = bundle(67, 29705, rows67, ids67, cycles67)
    late67 = e67["query_cycles"] >= 76
    e75 = bundle(75, 42117, e67["query_rows"][late67],
                 e67["query_record_ids"][late67], e67["query_cycles"][late67])

    def frozen_origin(origin: int, arrays: dict[str, np.ndarray]) -> dict[str, object]:
        return {
            "fit_rows": len(arrays["fit_rows"]),
            "fit_cycle_bounds": [47, origin],
            "query_cycle_bounds": [68, 79] if origin == 67 else [76, 79],
            "fit_row_indices_sha256": direct_helpers.sha_pilot_array(arrays["fit_rows"]),
            "query_row_indices_sha256": direct_helpers.sha_pilot_array(arrays["query_rows"]),
            "query_record_ids_sha256": direct_helpers.sha_pilot_array(arrays["query_record_ids"]),
            "query_cycles_sha256": direct_helpers.sha_pilot_array(arrays["query_cycles"]),
            "baseline": {
                "prediction_sha256": direct_helpers.sha_pilot_array(
                    arrays["baseline"], dtype="<f8"),
                "record_ids_sha256": direct_helpers.sha_pilot_array(
                    arrays["query_record_ids"]),
                "cycles_sha256": direct_helpers.sha_pilot_array(arrays["query_cycles"]),
            },
        }

    direct_freeze = {"origins": {"67": frozen_origin(67, e67),
                                  "75": frozen_origin(75, e75)}}
    plan = direct_helpers.stitch_oof_rows(e67, e75)
    rows = np.asarray([item[0] for item in plan], dtype=np.int64)
    origins = np.asarray([item[1] for item in plan], dtype=np.int8)
    indices = np.asarray([item[2] for item in plan], dtype=np.int64)
    ids = np.asarray([item[3] for item in plan], dtype=str)
    cycles = np.asarray([item[4] for item in plan], dtype=np.int16)
    direct = np.empty(len(plan), dtype="<f8")
    missing = np.empty(len(plan), dtype=bool)
    for origin, arrays in ((67, e67), (75, e75)):
        selected = origins == origin
        bundle_indices = indices[selected]
        direct[selected] = arrays["prediction"][bundle_indices]
        missing[selected] = arrays["missing_mask"][bundle_indices]
    row_counts = {
        "unique_oof": int(len(rows)),
        "E67_68_75": int(np.sum((origins == 67) & (cycles >= 68) & (cycles <= 75))),
        "E75_76_79": int(np.sum((origins == 75) & (cycles >= 76) & (cycles <= 79))),
    }
    fingerprints = {
        "rows_sha256": r2.r1.tagged_array_sha256(rows),
        "ids_sha256": r2.r1.tagged_array_sha256(ids),
        "origins_sha256": r2.r1.tagged_array_sha256(origins),
        "original_bundle_indices_sha256": r2.r1.tagged_array_sha256(indices),
        "cycles_sha256": r2.r1.tagged_array_sha256(cycles),
        "direct_prediction_sha256": r2.r1.tagged_array_sha256(direct, dtype="<f8"),
        "missing_mask_sha256": r2.r1.tagged_array_sha256(missing),
    }
    identity: dict[str, object] = {
        "rows": rows, "origins": origins, "indices": indices, "ids": ids,
        "cycles": cycles, "direct": direct, "missing": missing,
        "row_counts": row_counts, "row_identity_hashes": fingerprints,
    }
    return {67: e67, 75: e75}, direct_freeze, identity, {**fingerprints, "row_counts": row_counts}


class _NoopLock:
    def __enter__(self) -> "_NoopLock":
        return self

    def __exit__(self, *_args: object) -> bool:
        return False


def _run_synthetic_diagnostics_end_to_end() -> dict[str, object]:
    bundles, direct_freeze, identity, support = _synthetic_direct_fixture()
    synthetic_train_sha = hashlib.sha256(b"synthetic selected labels").hexdigest()
    with tempfile.TemporaryDirectory(prefix="r4-diagnostics-") as temporary:
        root = Path(temporary)
        report_dir = root / "reports" / "optimization_v3" / "reg_monotone_hinge_p50_v1"
        experiment_dir = root / "experiments" / "reg_monotone_hinge_p50_v1"
        run_dir = root / "experiments" / "runs" / "reg_monotone_hinge_p50_v1" / "R4"
        source_hashes = {"synthetic_runner": "a" * 64, "synthetic_verifier": "b" * 64}

        policy_path = root / "RESOURCE_POLICY.json"
        policy = {"workloads": {"small_calibration_lp": {
            "minimum_available_bytes": 2 * 1024**3,
            "incremental_estimate_bytes": 512 * 1024**2,
            "minimum_projected_free_bytes": 1024**3,
            "minimum_actual_free_at_boundary_bytes": 1024**3,
        }}}
        policy_path.write_text(json.dumps(policy, sort_keys=True), encoding="utf-8")

        def write_pin(name: str, payload: bytes) -> dict[str, str]:
            path = root / "inputs" / f"{name}.bin"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            return {"path": str(path.resolve()), "sha256": hashlib.sha256(payload).hexdigest()}

        input_hashes: dict[str, dict[str, str]] = {}
        for name in r2.PREFLIGHT_INPUT_ALLOWLIST:
            if name == "user_revised_resource_policy":
                input_hashes[name] = {"path": str(policy_path.resolve()),
                                      "sha256": r2.file_sha256(policy_path)}
            elif name.startswith("r3_stage__"):
                stage = name.split("__", 1)[1].split("_", 1)[0]
                intercept, slope = (5.0, 0.9) if stage == "D1" else (5.5, 0.85)
                state = {"status": "LP_COMPLETE", "solver_result": {
                    "success": True, "intercept": intercept, "slope": slope}}
                state_path = root / "inputs" / f"{name}.json"
                state_path.parent.mkdir(parents=True, exist_ok=True)
                state_path.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
                input_hashes[name] = {"path": str(state_path.resolve()),
                                      "sha256": r2.file_sha256(state_path)}
            else:
                input_hashes[name] = write_pin(name, name.encode("utf-8"))

        # Use the input map's exact temporary pins in the synthetic direct audit.
        audits: dict[str, dict[str, object]] = {}
        for origin, label in ((67, "E67"), (75, "E75")):
            archive_pin = input_hashes[f"r3_input__pinned_direct_{label}"]
            receipt_pin = input_hashes[f"r3_input__pinned_direct_{label}_receipt"]
            audits[label] = {
                "fit_id": f"REG_POSITION_DIRECT_L1_V1_E{origin}_REAL",
                "archive_path": archive_pin["path"],
                "archive_sha256": archive_pin["sha256"],
                "receipt_path": receipt_pin["path"],
                "receipt_sha256": receipt_pin["sha256"],
                "receipt_signature_verified": True,
                "producer_array_hashes": {
                    key: hashlib.sha256(f"{label}:{key}".encode()).hexdigest()
                    for key in ("query_cycles_sha256", "query_ids_sha256",
                                "raw_auxiliary_sha256", "mapped_auxiliary_sha256",
                                "prediction_sha256")},
                "allowlisted_loaded_npz_members": list(r2.DIRECT_NPZ_ALLOWLIST),
                "unloaded_extra_npz_member_names": [],
                "query_rows": len(bundles[origin]["query_rows"]),
                "target_member_loaded": False,
            }

        global_states: dict[str, dict[str, object]] = {}
        for stage in ("D1", "D2"):
            pin = input_hashes[f"r3_stage__{stage}_GLOBAL_state"]
            result = json.loads(Path(pin["path"]).read_text(encoding="utf-8"))["solver_result"]
            global_states[f"{stage}_GLOBAL"] = {
                "state_sha256": pin["sha256"],
                "intercept": result["intercept"], "slope": result["slope"],
            }

        freeze = {
            "schema_version": 4, "experiment_id": r2.EXPERIMENT_ID,
            "status": "PREFIT_IDENTITY_FROZEN_PENDING_IDENTITY_PREFLIGHT",
            "source_commit": "synthetic-source-ancestor",
            "source_hashes": source_hashes, "input_hashes": input_hashes,
            "official_lp_starts": 0, "official_labels_accessed": False,
            "identity_preflight_completed": False,
            "support_identity": support,
            "stage_support": {
                "D1": {"fit_cycles": [68, 71], "fit_rows": 4950,
                       "score_cycles": [72, 75], "score_rows": 7462},
                "D2": {"fit_cycles": [68, 75], "fit_rows": 12412,
                       "score_cycles": [76, 79], "score_rows": 8491},
            },
            "r3_predecessor": {"synthetic": True},
        }

        def write_json(path: Path, value: object) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value, sort_keys=True, indent=2), encoding="utf-8")

        freeze_path = experiment_dir / "IDENTITY_FREEZE_R4.json"
        write_json(freeze_path, freeze)
        freeze_sha = r2.file_sha256(freeze_path)
        resources = [
            _valid_resource_snapshot(initial=True) | {"boundary": "identity_preflight_locked_initial"},
            _valid_resource_snapshot(initial=False) | {"boundary": "identity_preflight_after_direct_identity"},
            _valid_resource_snapshot(initial=False) | {"boundary": "identity_preflight_after_global_controls"},
        ]
        preflight = r2._build_identity_preflight_body(
            freeze, freeze_sha, {**identity, "direct_bundle_audits": audits},
            global_states, resources)
        preflight["identity_validation"] = {
            "function": "verify_oof_identity", "passed": True,
            "row_counts_recomputed": True}
        preflight_path = report_dir / "IDENTITY_PREFLIGHT_R4.json"
        write_json(preflight_path, preflight)
        preflight_sha = r2.file_sha256(preflight_path)
        approval = {
            "verdict": "PASS", "experiment_id": r2.EXPERIMENT_ID,
            "source_commit": freeze["source_commit"],
            "prefit_freeze_sha256": freeze_sha,
            "identity_preflight_sha256": preflight_sha,
            "identity_preflight_status": "PASS_IDENTITY_ONLY_PREFLIGHT",
            "source_hashes": source_hashes, "input_hashes": input_hashes,
            "authorized_fit_ids": list(r2.DIAGNOSTIC_FIT_IDS),
            "official_lp_start_cap": r2.MAX_LP_STARTS,
        }
        approval["signature"] = r2.canonical_sha(approval)
        pass_path = experiment_dir / "ROOT_SOURCE_PASS_R4.json"
        write_json(pass_path, approval)

        target_values = (5.0 + 0.5 * np.minimum(identity["direct"], 50.0)
                         + 1.0 * np.maximum(identity["direct"] - 50.0, 0.0)).astype("<f8")
        reader_calls: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        def read_synthetic_labels(source_root: Path, rows: np.ndarray, ids: np.ndarray,
                                  cycles: np.ndarray, *, expected_total_rows: int):
            assert source_root == r2.v2.SOURCE
            assert expected_total_rows == 50_608
            assert np.array_equal(rows, identity["rows"])
            assert np.array_equal(ids, identity["ids"])
            assert np.array_equal(cycles, identity["cycles"])
            reader_calls.append((rows.copy(), ids.copy(), cycles.copy()))
            audit = {"sha256": synthetic_train_sha, "fit_target_rows_parsed": 0}
            return target_values.copy(), None, rows.copy(), audit

        def fake_resource_snapshot(*, boundary: str, initial: bool = False):
            snapshot = _valid_resource_snapshot(initial=initial)
            snapshot["boundary"] = boundary
            return snapshot

        def fake_read_json(path: Path):
            if Path(path) == r2.v2.DIRECT_FREEZE:
                return direct_freeze
            raise AssertionError(f"Unexpected JSON dependency in synthetic integration: {path}")

        def fake_load_bundle(origin: int):
            label = f"E{origin}"
            return ({key: value.copy() for key, value in bundles[origin].items()}, audits[label])

        diagnostics_path = report_dir / "DIAGNOSTICS_R4.json"
        verification_path = report_dir / "DIAGNOSTICS_VERIFICATION_R4.json"
        resource_path = run_dir / "RESOURCE_SNAPSHOTS_R4.json"
        ledger_path = run_dir / "lp_ledger.jsonl"
        overrides = {
            "ROOT": root,
            "REPORT_DIR": report_dir,
            "EXPERIMENT_DIR": experiment_dir,
            "RUN_DIR": run_dir,
            "PREFIT_FREEZE": freeze_path,
            "IDENTITY_PREFLIGHT": preflight_path,
            "ROOT_SOURCE_PASS": pass_path,
            "RESOURCE_POLICY_FILE": policy_path,
            "DIAGNOSTICS": diagnostics_path,
            "VERIFICATION": verification_path,
            "LEDGER": ledger_path,
        }
        with ExitStack() as stack:
            for name, value in overrides.items():
                stack.enter_context(patch.object(r2, name, value))
            stack.enter_context(patch.object(r2, "source_hashes", return_value=source_hashes))
            stack.enter_context(patch.object(r2, "_source_ancestor", return_value=True))
            stack.enter_context(patch.object(r2, "_load_direct_bundle", side_effect=fake_load_bundle))
            stack.enter_context(patch.object(r2.v2, "read_json", side_effect=fake_read_json))
            stack.enter_context(patch.object(r2.v2, "TRAIN_SHA", synthetic_train_sha))
            stack.enter_context(patch.object(r2, "read_scored_query", side_effect=read_synthetic_labels))
            stack.enter_context(patch.object(r2, "SharedHeavyJobLock", _NoopLock))
            stack.enter_context(patch.object(r2, "resource_snapshot", side_effect=fake_resource_snapshot))
            stack.enter_context(patch.object(r2.v2, "one_job_gate", return_value=None))
            try:
                result = r2.run_diagnostics()
            except Exception as exc:
                failed_state_path = run_dir / "fits" / "HINGE_D1.json"
                failed_state = (json.loads(failed_state_path.read_text(encoding="utf-8"))
                                if failed_state_path.is_file() else None)
                raise AssertionError(
                    f"Synthetic run_diagnostics failed: {type(exc).__name__}: {exc}; "
                    f"fit_failure={None if failed_state is None else failed_state.get('failure')}") from exc
            report = json.loads(diagnostics_path.read_text(encoding="utf-8"))
            verification = json.loads(verification_path.read_text(encoding="utf-8"))
            ledger = r2.read_ledger()
            pass_sha = r2.file_sha256(pass_path)
            fit_states = {
                fit_id: json.loads((run_dir / "fits" / f"{fit_id}.json").read_text(encoding="utf-8"))
                for fit_id in r2.DIAGNOSTIC_FIT_IDS
            }
            passed = (
                result["status"] == "TWO_DIAGNOSTICS_COMPLETE_PENDING_ROOT_REVIEW"
                and verification["status"] == "PASS_INDEPENDENT_DIAGNOSTICS_REPLAY"
                and [record["fit_id"] for record in report["stages"]]
                    == list(r2.DIAGNOSTIC_FIT_IDS)
                and sum(len(stage["LP_STARTED"]) for stage in ledger.values()) == 2
                and all(fit_states[fit_id]["status"] == "LP_COMPLETE"
                        for fit_id in r2.DIAGNOSTIC_FIT_IDS)
                and report["official_lp_starts"] == 2
                and report["final_fit_started"] is False
                and len(reader_calls) == 2
                and verification["pooled_unique_rows"] == r2.EXPECTED_POOLED_ROWS
                and all(np.array_equal(call[0], identity["rows"])
                        and np.array_equal(call[1], identity["ids"])
                        and np.array_equal(call[2], identity["cycles"])
                        for call in reader_calls)
            )
            return {
                "pass": bool(passed),
                "diagnostics_status": result["status"],
                "independent_verifier_status": verification["status"],
                "synthetic_fit_ids": list(r2.DIAGNOSTIC_FIT_IDS),
                "synthetic_lp_starts": sum(len(stage["LP_STARTED"]) for stage in ledger.values()),
                "target_reader_calls": len(reader_calls),
                "pooled_unique_rows": verification["pooled_unique_rows"],
                "root_source_pass_sha256": pass_sha,
                "official_labels_read": False,
                "official_lp_starts": 0,
            }


class MonotoneHingeR4Tests(unittest.TestCase):
    def test_full_synthetic_matrix_passes(self) -> None:
        checks = verifier.run_synthetic_checks()
        self.assertGreaterEqual(len(checks), 10)
        self.assertTrue(all(check["pass"] for check in checks.values()))

    def test_highest_level_diagnostics_and_independent_verifier_end_to_end(self) -> None:
        result = _run_synthetic_diagnostics_end_to_end()
        self.assertTrue(result["pass"], result)
        self.assertEqual(result["synthetic_fit_ids"], list(r2.DIAGNOSTIC_FIT_IDS))
        self.assertEqual(result["synthetic_lp_starts"], 2)
        self.assertEqual(result["official_lp_starts"], 0)
        self.assertFalse(result["official_labels_read"])
        self.assertEqual(result["pooled_unique_rows"], r2.EXPECTED_POOLED_ROWS)

    def test_actual_receipt_signature_helper_uses_three_argument_api(self) -> None:
        body = {"fit_id": "SYNTHETIC_RECEIPT", "query_targets_read": False}
        receipt = {**body, "receipt_sha256": direct_helpers.canonical_sha(body)}
        direct_helpers.verify_signed(receipt, "receipt_sha256", "synthetic receipt")
        with self.assertRaises(direct_helpers.GateError):
            direct_helpers.verify_signed(
                {**receipt, "fit_id": "TAMPERED"}, "receipt_sha256", "synthetic receipt")

    def test_identity_preflight_contract_has_no_target_or_label_input(self) -> None:
        identity = {"rows": np.array([10, 11, 12, 13]),
            "origins": np.array([67, 67, 67, 75]),
            "cycles": np.array([68, 71, 72, 76]),
            "row_identity_hashes": {"rows_sha256": "1" * 64},
            "direct_bundle_audits": {"E67": {}, "E75": {}}}
        freeze = {"source_commit": "synthetic", "source_hashes": {},
            "input_hashes": {}, "r3_predecessor": {}}
        report = r2._build_identity_preflight_body(freeze, "2" * 64, identity, {}, [])
        self.assertNotIn("targets", __import__("inspect").signature(r2.identity_preflight).parameters)
        self.assertFalse(report["target_loader_called"])
        self.assertFalse(report["train_targets_parsed"])
        self.assertFalse(report["cached_r4_target_arrays_loaded"])
        self.assertFalse(report["scoring_started"])
        self.assertEqual(report["official_lp_starts"], 0)

    def test_frozen_identity_adapter_recomputes_counts_and_rejects_bad_schema_or_rows(self) -> None:
        identity, expected = _small_identity()
        self.assertEqual(r2.verify_oof_identity(identity, expected), expected)

        reordered = {key: value.copy() if isinstance(value, np.ndarray) else value
                     for key, value in identity.items()}
        for key in ("rows", "ids", "origins", "indices", "cycles", "direct", "missing"):
            reordered[key] = reordered[key][::-1].copy()
        reordered["row_identity_hashes"] = {
            "rows_sha256": r2.r1.tagged_array_sha256(reordered["rows"]),
            "ids_sha256": r2.r1.tagged_array_sha256(reordered["ids"]),
            "origins_sha256": r2.r1.tagged_array_sha256(reordered["origins"]),
            "original_bundle_indices_sha256": r2.r1.tagged_array_sha256(reordered["indices"]),
            "cycles_sha256": r2.r1.tagged_array_sha256(reordered["cycles"]),
            "direct_prediction_sha256": r2.r1.tagged_array_sha256(reordered["direct"], dtype="<f8"),
            "missing_mask_sha256": r2.r1.tagged_array_sha256(reordered["missing"]),
        }
        with self.assertRaises(r2.GateError):
            r2.verify_oof_identity(reordered, expected)

        tampered_ids = {key: value.copy() if isinstance(value, np.ndarray) else value
                        for key, value in identity.items()}
        tampered_ids["ids"][1] = "tampered"
        tampered_ids["row_identity_hashes"] = {
            **identity["row_identity_hashes"],
            "ids_sha256": r2.r1.tagged_array_sha256(tampered_ids["ids"]),
        }
        with self.assertRaises(r2.GateError):
            r2.verify_oof_identity(tampered_ids, expected)

        missing = dict(identity)
        missing.pop("ids")
        with self.assertRaises(r2.GateError):
            r2.verify_oof_identity(missing, expected)
        old_schema = {"physical_rows": identity["rows"], "record_ids": identity["ids"]}
        with self.assertRaises(r2.GateError):
            r2.verify_oof_identity(old_schema, expected)
        wrong_counts = dict(identity)
        wrong_counts["row_counts"] = {**identity["row_counts"], "E67_68_75": 2}
        with self.assertRaises(r2.GateError):
            r2.verify_oof_identity(wrong_counts, expected)

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
            "identity_preflight_sha256": "4" * 64,
            "identity_preflight_status": "PASS_IDENTITY_ONLY_PREFLIGHT",
            "source_hashes": freeze["source_hashes"],
            "input_hashes": freeze["input_hashes"],
            "authorized_fit_ids": list(r2.DIAGNOSTIC_FIT_IDS),
            "official_lp_start_cap": r2.MAX_LP_STARTS,
        }
        approval["signature"] = r2.canonical_sha(approval)
        r2.validate_source_pass_body(approval, freeze, "3" * 64, "4" * 64)

        tampered = {**approval, "authorized_fit_ids": [*r2.DIAGNOSTIC_FIT_IDS, "HINGE_FINAL"]}
        tampered["signature"] = r2.canonical_sha(
            {key: value for key, value in tampered.items() if key != "signature"})
        with self.assertRaises(r2.GateError):
            r2.validate_source_pass_body(tampered, freeze, "3" * 64, "4" * 64)

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
