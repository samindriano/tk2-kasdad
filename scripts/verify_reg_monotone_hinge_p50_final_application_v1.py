"""Independent replay verifier for the isolated HINGE_FINAL E79 application."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np

from scripts import reg_final_wave_v2 as v2
from scripts import reg_monotone_hinge_p50_v4 as r4
from scripts import verify_reg_monotone_hinge_p50_v4 as r4verifier


EXPERIMENT_ID = "REG_MONOTONE_HINGE_P50_FINAL_APPLICATION_R1"
FIT_ID = "HINGE_FINAL"
PREVIOUS_FIT_IDS = ("HINGE_D1", "HINGE_D2")
EXPECTED_OOF_ROWS = 20_903
EXPECTED_TEST_ROWS = 29_421
EXPECTED_MISSING_ROWS = 1_062
KNOT = 50.0


class VerificationError(RuntimeError):
    """Raised when an independent final-application replay fails closed."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_signature(value: Mapping[str, Any], field: str = "signature") -> None:
    body = {key: item for key, item in value.items() if key != field}
    _need(value.get(field) == r4.canonical_sha(body), "Signed document signature mismatch")


def _tagged_sha(value: np.ndarray, dtype: str | None = None) -> str:
    return r4.r1.tagged_array_sha256(value, dtype=dtype)


def _expected_fit_hashes(arrays: Mapping[str, np.ndarray]) -> dict[str, str]:
    mapping = {
        "physical_rows": "physical_rows_sha256",
        "record_ids": "record_ids_sha256",
        "prediction_origins": "prediction_origins_sha256",
        "original_bundle_indices": "original_bundle_indices_sha256",
        "cycles": "cycles_sha256",
        "missing_mask": "missing_mask_sha256",
    }
    result = {out: _tagged_sha(arrays[key]) for key, out in mapping.items()}
    result["direct_prediction_sha256"] = _tagged_sha(
        arrays["direct_predictions"], dtype="<f8")
    result["target_sha256"] = _tagged_sha(arrays["targets"], dtype="<f8")
    result["direct_raw_f64_sha256"] = r4.r1.raw_direct_f64_sha256(
        arrays["direct_predictions"])
    return result


def _verify_source_pass(runner: Any, freeze: Mapping[str, Any],
                        freeze_sha: str, approval_sha: str) -> dict[str, Any]:
    path = Path(runner.ROOT_SOURCE_PASS)
    _need(path.is_file() and _sha(path) == approval_sha,
          "New final-application source pass missing or changed")
    approval = json.loads(path.read_text(encoding="utf-8"))
    _verify_signature(approval)
    _need(
        approval.get("verdict") == "PASS"
        and approval.get("pass_kind") == "FINAL_APPLICATION"
        and approval.get("experiment_id") == EXPERIMENT_ID
        and approval.get("source_commit") == freeze.get("source_commit")
        and approval.get("source_hashes") == freeze.get("source_hashes")
        and approval.get("input_hashes") == freeze.get("input_hashes")
        and approval.get("prefit_freeze_sha256") == freeze_sha
        and approval.get("identity_preflight_sha256") == _sha(Path(runner.PREFLIGHT))
        and approval.get("r4_diagnostic_source_pass_sha256")
            == runner.R4_ROOT_SOURCE_PASS_SHA256
        and approval.get("r4_diagnostics_sha256") == runner.R4_DIAGNOSTICS_SHA256
        and approval.get("authorized_fit_ids") == [FIT_ID]
        and approval.get("authorized_final_lp_starts") == 1
        and approval.get("prior_official_lp_starts") == 2
        and approval.get("total_official_lp_start_cap") == 3
        and approval.get("authorized_csv_creation") is True
        and approval.get("test_or_sample_access_authorized") is False
        and approval.get("kaggle_upload_authorized") is False,
        "New source pass does not authorize exactly one final application slot")
    return approval


def _verify_freeze_preflight(runner: Any, freeze: Mapping[str, Any],
                             freeze_sha: str) -> dict[str, Any]:
    freeze_path = Path(runner.FREEZE)
    _need(freeze_path.is_file() and _sha(freeze_path) == freeze_sha,
          "Final-application freeze bytes differ")
    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
    _verify_signature(frozen)
    _need(frozen == dict(freeze), "Passed freeze object differs from immutable freeze")
    source_hashes = runner.source_hashes()
    _need(source_hashes == frozen.get("source_hashes"),
          "Final-application committed source bytes differ from freeze")
    _need(runner._source_ancestor(str(frozen.get("source_commit", ""))),
          "Final-application source commit is not an ancestor of current HEAD")

    preflight = json.loads(Path(runner.PREFLIGHT).read_text(encoding="utf-8"))
    _verify_signature(preflight)
    _need(preflight.get("status") == "PASS_FINAL_APPLICATION_IDENTITY_PREFLIGHT"
          and preflight.get("prefit_freeze_sha256") == freeze_sha
          and preflight.get("source_hashes") == frozen.get("source_hashes")
          and preflight.get("input_hashes") == frozen.get("input_hashes")
          and preflight.get("official_lp_starts") == 0
          and preflight.get("target_reader_called") is False
          and preflight.get("train_targets_parsed") is False
          and preflight.get("train_target_source_hash_verified") is True
          and preflight.get("test_or_sample_read") is False
          and preflight.get("csv_created") is False,
          "Final-application preflight identity or boundary differs")
    _need(preflight.get("oof_identity", {}).get("unique_rows") == EXPECTED_OOF_ROWS
          and preflight.get("e79_raw_direct_audit", {}).get("rows") == EXPECTED_TEST_ROWS
          and preflight.get("e79_raw_direct_audit", {}).get("missing_rows")
              == EXPECTED_MISSING_ROWS,
          "Final-application preflight row counts differ")

    # The shared R4 ledger is intentionally allowed to contain the separately
    # authorized third slot by this point; verify its pre-final hash through the
    # signed preflight and its current state through the cumulative ledger below.
    preflight_inputs = frozen.get("preflight_input_hashes", {})
    metadata_only = {"r4_source_commit", "r4_freeze_sha256", "r4_source_pass_sha256"}
    _need(set(preflight_inputs) == set(runner.PREFLIGHT_REHASH_KEYS),
          "Freeze preflight input map differs from the reviewed exact allowlist")
    _need(set(frozen.get("input_hashes", {}))
          == set(preflight_inputs) | metadata_only,
          "Freeze omits or adds an unverified input outside the exact metadata pins")
    r4_freeze_pin = frozen["input_hashes"]["r4_identity_freeze"]
    r4_pass_pin = frozen["input_hashes"]["r4_root_source_pass"]
    r4_freeze_metadata = frozen["input_hashes"]["r4_freeze_sha256"]
    r4_pass_metadata = frozen["input_hashes"]["r4_source_pass_sha256"]
    r4_commit_metadata = frozen["input_hashes"]["r4_source_commit"]
    r4_freeze_path = Path(r4_freeze_pin["path"])
    r4_pass_path = Path(r4_pass_pin["path"])
    _need(r4_freeze_path.is_file() and _sha(r4_freeze_path) == r4_freeze_pin["sha256"]
          and r4_freeze_metadata == {"path": str(r4_freeze_path.resolve()),
                                    "sha256": r4_freeze_pin["sha256"]},
          "R4 predecessor identity freeze pin/metadata differs")
    r4_freeze_body = json.loads(r4_freeze_path.read_text(encoding="utf-8"))
    _need(r4_commit_metadata.get("path") == str(r4_freeze_path.resolve())
          and r4_commit_metadata.get("sha256") == r4_freeze_body.get("source_commit"),
          "R4 predecessor source commit metadata differs from its freeze")
    _need(r4_pass_path.is_file() and _sha(r4_pass_path) == r4_pass_pin["sha256"]
          and r4_pass_metadata == {"path": str(r4_pass_path.resolve()),
                                   "sha256": r4_pass_pin["sha256"]}
          and r4_pass_pin["sha256"] == runner.R4_ROOT_SOURCE_PASS_SHA256,
          "R4 predecessor source pass pin/metadata differs")
    train_r4 = r4_freeze_body.get("input_hashes", {}).get("r3_input__official_train", {})
    train_r1 = frozen.get("input_hashes", {}).get("official_train_deferred", {})
    _need(train_r1.get("path") == str(Path(train_r4.get("path", "")).resolve())
          and train_r1.get("sha256") == train_r4.get("sha256"),
          "Deferred Train raw-byte pin differs from the R4 predecessor freeze")
    _need(frozen.get("input_hashes", {}).get("r4_lp_ledger", {}).get("path")
          == str(Path(frozen["prior_slot_accounting"]["ledger_path"]).resolve()),
          "R4 cumulative ledger path differs from the slot-accounting contract")
    for name, item in preflight_inputs.items():
        if name == "r4_lp_ledger":
            continue  # Compare its exact original byte prefix against the saved digest below.
        path = Path(item["path"])
        _need(path.is_file() and _sha(path) == item["sha256"],
              f"Frozen preflight input changed: {name}")
    prior = frozen.get("prior_slot_accounting", {})
    _need(preflight.get("prior_slot_accounting", {}).get("ledger_sha256")
          == prior.get("ledger_sha256_before_final_slot"),
          "Signed preflight no longer binds the original D1/D2 ledger")
    return preflight


def _verify_ledger_and_states(runner: Any, freeze: Mapping[str, Any],
                              approval_sha: str, report: Mapping[str, Any]
                              ) -> tuple[dict[str, Any], dict[str, Any]]:
    ledger_path = Path(freeze["prior_slot_accounting"]["ledger_path"])
    ledger = r4.read_ledger(ledger_path)
    lines = ledger_path.read_bytes().splitlines(keepends=True)
    prefix: list[bytes] = []
    final_seen = False
    for raw_line in lines:
        _need(raw_line.endswith(b"\n"), "Cumulative ledger has a partial event line")
        event = json.loads(raw_line)
        if event.get("fit_id") == FIT_ID:
            final_seen = True
        elif final_seen:
            raise VerificationError("A historical ledger event appears after HINGE_FINAL")
        else:
            prefix.append(raw_line)
    _need(hashlib.sha256(b"".join(prefix)).hexdigest()
          == freeze.get("prior_slot_accounting", {}).get("ledger_sha256_before_final_slot"),
          "Cumulative R4 ledger does not preserve the exact frozen D1/D2 byte prefix")
    _need(set(ledger) == set((*PREVIOUS_FIT_IDS, FIT_ID)),
          "Cumulative ledger must contain only D1, D2, and HINGE_FINAL")
    starts = sum(len(stage["LP_STARTED"]) for stage in ledger.values())
    _need(starts == 3, "Cumulative official LP start count is not exactly three")

    for fit_id in PREVIOUS_FIT_IDS:
        stage = ledger[fit_id]
        _need(len(stage["RESERVE"]) == len(stage["LP_STARTED"])
              == len(stage["LP_COMPLETE"]) == 1 and not stage["LP_FAILED"],
              f"Historical {fit_id} is not one completed slot")
        state_path = Path(runner.R4_FINAL_STATE_PATHS[fit_id])
        state = json.loads(state_path.read_text(encoding="utf-8"))
        _need(_sha(state_path) == stage["LP_COMPLETE"][0].get("state_sha256")
              and state.get("fit_id") == fit_id
              and state.get("status") == "LP_COMPLETE"
              and state.get("fit_input", {}).get("root_source_pass_sha256")
                  == runner.R4_ROOT_SOURCE_PASS_SHA256,
              f"Historical {fit_id} state identity differs")

    final_stage = ledger[FIT_ID]
    _need(len(final_stage["RESERVE"]) == len(final_stage["LP_STARTED"])
          == len(final_stage["LP_COMPLETE"]) == 1 and not final_stage["LP_FAILED"],
          "HINGE_FINAL is not one completed non-retry slot")
    reserve = final_stage["RESERVE"][0]
    started = final_stage["LP_STARTED"][0]
    complete = final_stage["LP_COMPLETE"][0]
    _need(reserve.get("attempt") == 1 and reserve.get("retry_allowed") is False
          and started.get("retry_allowed") is False
          and reserve.get("root_source_pass_sha256") == approval_sha
          and started.get("root_source_pass_sha256") == approval_sha
          and complete.get("root_source_pass_sha256") == approval_sha,
          "HINGE_FINAL ledger events are not bound to the new source pass")

    fit_state_path = Path(report["final_slot"]["fit_state_path"])
    state = json.loads(fit_state_path.read_text(encoding="utf-8"))
    _need(_sha(fit_state_path) == complete.get("state_sha256")
          and report["final_slot"].get("fit_state_sha256") == _sha(fit_state_path)
          and state.get("status") == "LP_COMPLETE"
          and state.get("fit_id") == FIT_ID
          and state.get("root_source_pass_sha256") == approval_sha,
          "HINGE_FINAL completed fit state differs from ledger/report")
    return ledger, state


def _load_fit_bundle(report: Mapping[str, Any], state: Mapping[str, Any],
                     freeze: Mapping[str, Any], freeze_sha: str,
                     approval_sha: str) -> dict[str, np.ndarray]:
    path = Path(report["final_slot"]["fit_input_path"])
    _need(path.is_file() and _sha(path) == report["final_slot"].get("fit_input_sha256"),
          "HINGE_FINAL fit-input bundle SHA differs")
    names = ("physical_rows", "record_ids", "prediction_origins",
             "original_bundle_indices", "cycles", "direct_predictions",
             "missing_mask", "targets")
    with np.load(path, allow_pickle=False) as archive:
        _need(set(archive.files) == set(names), "HINGE_FINAL fit-input schema differs")
        arrays = {name: np.asarray(archive[name]).copy() for name in names}
    _need(all(value.ndim == 1 and len(value) == EXPECTED_OOF_ROWS
              for value in arrays.values()), "HINGE_FINAL fit-input vectors are misaligned")
    _need(np.all(np.isfinite(arrays["direct_predictions"]))
          and np.all(np.isfinite(arrays["targets"]))
          and np.all((arrays["direct_predictions"] >= 0)
                     & (arrays["direct_predictions"] <= 100))
          and np.all((arrays["targets"] >= 0) & (arrays["targets"] <= 100)),
          "HINGE_FINAL fit inputs are not finite [0,100] vectors")

    input_identity = state.get("fit_input", {})
    _need(input_identity.get("experiment_id") == EXPERIMENT_ID
          and input_identity.get("fit_id") == FIT_ID
          and input_identity.get("fit_rows") == EXPECTED_OOF_ROWS
          and input_identity.get("fit_input_bundle_sha256") == _sha(path)
          and input_identity.get("source_commit") == freeze.get("source_commit")
          and input_identity.get("source_hashes") == freeze.get("source_hashes")
          and input_identity.get("input_hashes") == freeze.get("input_hashes")
          and input_identity.get("prefit_freeze_sha256") == freeze_sha
          and input_identity.get("root_source_pass_sha256") == approval_sha
          and input_identity.get("row_hashes") == _expected_fit_hashes(arrays),
          "HINGE_FINAL fit-input identity or row hashes differ")
    return arrays


def _verify_solver(state: Mapping[str, Any], arrays: Mapping[str, np.ndarray],
                   report: Mapping[str, Any], runner: Any) -> dict[str, float]:
    result = state.get("solver_result", {})
    _need(result.get("success") is True and result.get("fit_rows") == EXPECTED_OOF_ROWS
          and result.get("knot") == KNOT and result.get("slope_bounds") == [0.0, 2.0]
          and result.get("solver", {}).get("method") == "highs"
          and result.get("solver", {}).get("threads") == 2,
          "HINGE_FINAL solver identity differs")
    params = {key: float(result[key]) for key in
              ("intercept", "slope_low", "slope_high")}
    slot = report.get("final_slot", {})
    fit_path = Path(runner.RUN_DIR) / "fits" / f"{FIT_ID}.json"
    bundle_path = Path(runner.RUN_DIR) / "fit_inputs" / f"{FIT_ID}.npz"
    slack_path_expected = Path(runner.RUN_DIR) / "solver_slacks" / f"{FIT_ID}.npz"
    _need(slot.get("fit_state_path") == str(fit_path.resolve())
          and slot.get("fit_state_sha256") == _sha(fit_path)
          and slot.get("fit_input_path") == str(bundle_path.resolve())
          and slot.get("fit_input_sha256") == _sha(bundle_path)
          and slot.get("solver_slacks_path") == str(slack_path_expected.resolve())
          and slot.get("solver_slacks_sha256") == state.get("solver_slacks_sha256")
          and slot.get("coefficients") == params,
          "Final report coefficients or immutable fit artifact paths/hashes differ")
    _need(state.get("solver_slacks_path") == str(slack_path_expected.resolve()),
          "HINGE_FINAL state points to an unexpected slack artifact path")
    _need(0.0 <= params["slope_low"] <= 2.0
          and 0.0 <= params["slope_high"] <= 2.0,
          "HINGE_FINAL slopes are outside frozen bounds")
    predictions = (params["intercept"]
        + params["slope_low"] * np.minimum(arrays["direct_predictions"], KNOT)
        + params["slope_high"] * np.maximum(arrays["direct_predictions"] - KNOT, 0.0))
    residual = np.abs(arrays["targets"] - predictions)
    slack_path = Path(state["solver_slacks_path"])
    _need(slack_path.is_file() and _sha(slack_path) == state.get("solver_slacks_sha256"),
          "HINGE_FINAL slack artifact SHA differs")
    with np.load(slack_path, allow_pickle=False) as archive:
        _need(archive.files == ["slacks"], "HINGE_FINAL slack artifact schema differs")
        slacks = np.asarray(archive["slacks"], dtype="<f8")
    objective = float(result["objective_unweighted_sum_absolute"])
    _need(slacks.shape == residual.shape and np.all(np.isfinite(slacks))
          and np.all(slacks >= -1e-8)
          and np.max(residual - slacks) <= 1e-7
          and np.isclose(float(slacks.sum()), objective, rtol=1e-8, atol=1e-6)
          and np.isclose(float(residual.sum()), objective, rtol=1e-8, atol=1e-6)
          and np.isclose(objective,
                         float(report["final_slot"]["objective_unweighted_sum_absolute"]),
                         rtol=0, atol=0),
          "HINGE_FINAL primal slacks or unweighted L1 objective failed replay")
    return params


def _load_application_vectors(freeze: Mapping[str, Any]) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    paths = freeze.get("input_hashes", {})
    archive_path = Path(paths["e79_archive"]["path"])
    manifest_path = Path(paths["e79_manifest"]["path"])
    qa_path = Path(paths["e79_qa"]["path"])
    for name, path in (("e79_archive", archive_path), ("e79_manifest", manifest_path),
                       ("e79_qa", qa_path)):
        _need(path.is_file() and _sha(path) == paths[name]["sha256"],
              f"Pinned E79 input changed: {name}")
    manifest, qa, _direct_freeze = v2.verify_base()
    with np.load(archive_path, allow_pickle=False) as archive:
        members = set(archive.files)
        required = {"record_id", "prediction", "baseline", "missing"}
        _need(required
              .issubset(archive.files), "Pinned E79 archive lacks required non-target arrays")
        forbidden = [name for name in members - required
                     if any(token in name.casefold() for token in
                            ("target", "label", "outcome", "threat_rank", "score"))]
        _need(not forbidden, f"Pinned E79 archive has target-like names: {sorted(forbidden)}")
        ids = np.asarray(archive["record_id"], dtype=str).copy()
        raw = np.asarray(archive["prediction"], dtype="<f8").copy()
        baseline = np.asarray(archive["baseline"], dtype="<f8").copy()
        missing = np.asarray(archive["missing"], dtype=bool).copy()
    _need(ids.shape == raw.shape == baseline.shape == missing.shape == (EXPECTED_TEST_ROWS,)
          and len(np.unique(ids)) == EXPECTED_TEST_ROWS
          and int(missing.sum()) == EXPECTED_MISSING_ROWS
          and np.all(np.isfinite(raw)) and np.all(np.isfinite(baseline))
          and np.all((raw >= 0) & (raw <= 100))
          and np.array_equal(raw[missing].view("<u8"), baseline[missing].view("<u8")),
          "Pinned E79 IDs, predictions, or missing fallback differ")
    _need(manifest.get("candidate_id") == "DIRECT_POSITION_REAL_E79"
          and manifest.get("ordered_official_test_ids_sha256") == v2.sha_ordered_ids(ids)
          and manifest.get("prediction_sha256") == v2.sha_f64_bytes(raw)
          and manifest.get("prediction_rows") == EXPECTED_TEST_ROWS
          and manifest.get("missing_rows") == EXPECTED_MISSING_ROWS
          and manifest.get("test_labels_loaded") is False
          and qa.get("status") == "PASS_INDEPENDENT_CSV_QA"
          and qa.get("test_labels_read") is False,
          "Signed E79 manifest/QA do not bind the raw prediction archive")
    return ids, raw, missing, manifest, qa


def verify_final_application(*, report_dir: Path, freeze: Mapping[str, Any],
                             approval_sha: str, report_path: Path) -> dict[str, Any]:
    """Verify R1 freeze/pass, fit, cumulative slot, signed E79 route, and CSV."""
    from scripts import reg_monotone_hinge_p50_final_application_v1 as runner

    report_path = Path(report_path)
    report_dir = Path(report_dir)
    _need(report_path.resolve() == Path(runner.FINAL_REPORT).resolve(),
          "Verifier report path differs from the fixed final report output")
    _need(report_dir.resolve() == Path(runner.REPORT_DIR).resolve(),
          "Verifier report directory differs from the fixed final report directory")
    _need(report_path.is_file(), "Final-application report is absent")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _verify_signature(report)
    freeze_sha = _sha(Path(runner.FREEZE))
    _need(report.get("schema_version") == 1
          and report.get("experiment_id") == EXPERIMENT_ID
          and report.get("candidate_id") == "HINGE_P50_FINAL_EXPLORATORY_E79"
          and report.get("status") == "EXPLORATORY_CANDIDATE_PENDING_BATCH_SELECTION"
          and report.get("scientific_qualification") == "FAIL"
          and report.get("validation_claim") is None
          and report.get("source_commit") == freeze.get("source_commit")
          and report.get("source_hashes") == freeze.get("source_hashes")
          and report.get("input_hashes") == freeze.get("input_hashes")
          and report.get("prefit_freeze_sha256") == freeze_sha
          and report.get("new_root_source_pass_sha256") == approval_sha
          and report.get("r4_diagnostic_pass_sha256")
              == runner.R4_ROOT_SOURCE_PASS_SHA256
          and report.get("r4_diagnostics_sha256") == runner.R4_DIAGNOSTICS_SHA256
          and report.get("r4_diagnostics_verification_sha256")
              == runner.R4_VERIFICATION_SHA256,
          "Final report source, lineage, status, or qualification differs")

    approval = _verify_source_pass(runner, freeze, freeze_sha, approval_sha)
    preflight = _verify_freeze_preflight(runner, freeze, freeze_sha)
    _need(report.get("identity_preflight_sha256") == _sha(Path(runner.PREFLIGHT))
          and report.get("prior_slot_accounting") == freeze.get("prior_slot_accounting"),
          "Final report does not bind the frozen preflight and prior slot accounting")

    ledger, state = _verify_ledger_and_states(runner, freeze, approval_sha, report)
    arrays = _load_fit_bundle(report, state, freeze, freeze_sha, approval_sha)
    r4_freeze = json.loads(Path(r4.PREFIT_FREEZE).read_text(encoding="utf-8"))
    identity = r4.load_oof_identity(r4_freeze)
    r4.verify_oof_identity(identity, r4_freeze["support_identity"])
    alignment = {
        "physical_rows": np.asarray(identity["rows"], dtype=np.int64),
        "record_ids": np.asarray(identity["ids"], dtype=str),
        "prediction_origins": np.asarray(identity["origins"], dtype=np.int8),
        "original_bundle_indices": np.asarray(identity["indices"], dtype=np.int64),
        "cycles": np.asarray(identity["cycles"], dtype=np.int16),
        "direct_predictions": np.asarray(identity["direct"], dtype="<f8"),
        "missing_mask": np.asarray(identity["missing"], dtype=bool),
    }
    _need(all(np.array_equal(arrays[name], values) for name, values in alignment.items()),
          "Saved HINGE_FINAL rows differ from the frozen 20,903-row OOF identity")
    params = _verify_solver(state, arrays, report, runner)

    ids, raw, missing, manifest, qa = _load_application_vectors(freeze)
    expected_values = raw.copy()
    present = ~missing
    expected_values[present] = np.clip(
        params["intercept"]
        + params["slope_low"] * np.minimum(raw[present], KNOT)
        + params["slope_high"] * np.maximum(raw[present] - KNOT, 0.0),
        0.0, 100.0)
    _need(np.array_equal(expected_values[missing].view("<u8"),
                         raw[missing].view("<u8")),
          "Independent application changed raw DIRECT missing bits")

    csv_path = Path(report.get("csv_path", ""))
    _need(csv_path.resolve() == Path(runner.OUTPUT_CSV).resolve(),
          "Final report CSV path differs from the fixed application output")
    _need(csv_path.is_file() and _sha(csv_path) == report.get("csv_sha256"),
          "Final candidate CSV is absent or differs from final report")
    csv_audit = r4verifier.verify_candidate_csv(
        csv_path, expected_ids=ids, expected_values=expected_values,
        missing_mask=missing, raw_direct=raw, production=True)
    _need(report.get("csv_rows") == EXPECTED_TEST_ROWS
          and report.get("csv_columns") == ["record_id", "threat_rank"]
          and report.get("ordered_ids_sha256") == v2.sha_ordered_ids(ids)
          and report.get("prediction_sha256")
              == r4.r1.tagged_array_sha256(expected_values, dtype="<f8")
          and report.get("missing_rows") == EXPECTED_MISSING_ROWS
          and report.get("missing_fallback_bit_exact") is True
          and report.get("e79_raw_direct_audit", {}).get("archive_sha256")
              == freeze["input_hashes"]["e79_archive"]["sha256"]
          and report.get("e79_raw_direct_audit", {}).get("ordered_ids_sha256")
              == manifest.get("ordered_official_test_ids_sha256")
          and report.get("e79_raw_direct_audit", {}).get("target_like_members_loaded")
              is False
          and report.get("e79_raw_direct_audit", {}).get("csv_read") is False,
          "Final report output or E79 non-target identity differs")

    target_audit = report.get("train_target_audit", {})
    _need(target_audit.get("sha256") == freeze.get("deferred_target_input", {}).get("sha256")
          and (target_audit.get("query_target_rows_parsed") == EXPECTED_OOF_ROWS
               or target_audit.get("recovered_from_saved_fit_input") is True)
          and target_audit.get("fit_target_rows_parsed") in (0, EXPECTED_OOF_ROWS),
          "Final report target access audit differs from the selected OOF contract")
    decision = freeze.get("decision_context", {})
    _need(report.get("diagnostic_decision_context") == decision
          and decision.get("scientific_qualification") == "FAIL"
          and decision.get("upload_authorized") is False
          and all(float(value) > 0 for value in
                  decision.get("window_p90_absolute_error_worsening", {}).values())
          and all(float(value) > 0 for value in
                  decision.get("upper_tail_target_ge_95_harm", {}).values()),
          "Final report omitted the R4 qualification failure or counterevidence")
    _need(report.get("benchmark_batch_selection_complete") is False
          and report.get("test_or_sample_read") is False
          and report.get("test_labels_read") is False
          and report.get("kaggle_upload") is False
          and report.get("r4_d1_d2_refit") is False
          and report.get("global_control_refit") is False
          and report.get("new_base_model_fits") == 0,
          "Final application crossed a prohibited test, upload, or refit boundary")

    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS_INDEPENDENT_FINAL_APPLICATION_REPLAY",
        "final_report_sha256": _sha(report_path),
        "source_pass_sha256": approval_sha,
        "prefit_freeze_sha256": freeze_sha,
        "identity_preflight_sha256": _sha(Path(runner.PREFLIGHT)),
        "cumulative_official_lp_starts": sum(
            len(stage["LP_STARTED"]) for stage in ledger.values()),
        "fit_rows": EXPECTED_OOF_ROWS,
        "csv_rows": csv_audit["rows"],
        "missing_rows": csv_audit["missing_rows"],
        "csv_sha256": csv_audit["csv_sha256"],
        "ordered_ids_sha256": v2.sha_ordered_ids(ids),
        "missing_route_bit_exact": True,
        "test_or_sample_read": False,
        "kaggle_upload": False,
        "scientific_qualification": "FAIL",
        "benchmark_batch_selection_complete": False,
    }


def write_synthetic_qa(path: Path | None = None) -> dict[str, Any]:
    """Run the highest-level synthetic suite and write one immutable QA receipt."""
    from scripts import reg_monotone_hinge_p50_final_application_v1 as runner

    target = Path(runner.SYNTHETIC_QA if path is None else path)
    result = subprocess.run(
        [sys.executable, str(runner.TESTS)], cwd=str(runner.ROOT),
        capture_output=True, text=True, check=False)
    passed = result.returncode == 0 and "OK" in result.stderr and "FAILED" not in result.stderr
    body = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS_SYNTHETIC_FINAL_APPLICATION_E2E" if passed
            else "FAIL_SYNTHETIC_FINAL_APPLICATION_E2E",
        "tests_path": str(Path(runner.TESTS).resolve()),
        "tests_sha256": _sha(Path(runner.TESTS)),
        "command": [sys.executable, str(Path(runner.TESTS).resolve())],
        "return_code": result.returncode,
        "stdout": result.stdout[-4_000:],
        "stderr": result.stderr[-4_000:],
        "official_lp_starts": 0,
        "train_target_reader_called": False,
        "train_target_values_parsed": False,
        "test_or_sample_read": False,
        "official_csv_created": False,
        "synthetic_temporary_csv_created": True,
        "kaggle_upload": False,
        "synthetic_e79_rows": EXPECTED_TEST_ROWS,
        "synthetic_e79_missing_rows": EXPECTED_MISSING_ROWS,
        "synthetic_oof_rows": EXPECTED_OOF_ROWS,
    }
    receipt = {**body, "signature": r4.canonical_sha(body)}
    r4.write_json_once(target, receipt)
    if not passed:
        raise VerificationError("Synthetic final-application suite failed; failure receipt preserved")
    return {"status": receipt["status"], "path": str(target.resolve()),
            "sha256": _sha(target), "return_code": result.returncode}
