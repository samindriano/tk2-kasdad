"""One-shot Test-application recovery for the frozen E79 BAG2 source.

The frozen runner remains unchanged. This wrapper only forwards already-verified
ROOT_PASS input hashes and redirects the consumed application attempt, bundle,
receipt, and unique CSV path into APPLICATION_RECOVERY_R1.
"""
from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager, nullcontext
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence
from unittest import mock

import numpy as np

from scripts import cls_pos_poisson_bag2_e79_v1 as frozen
from scripts import verify_cls_pos_poisson_bag2_e79_v1 as frozen_verifier
from scripts import verify_cls_pos_poisson_bag2_e79_application_recovery_r1 as recovery_verifier


AUDIT_ROOT = Path(__file__).resolve().parents[1]
RECOVERY_ID = "CLS_POS_POISSON_BAG2_E79_APPLICATION_RECOVERY_R1"
REPORT_REL = "reports/simplification_v1/CLS_POS_POISSON_BAG2_E79_V1/APPLICATION_RECOVERY_R1"
AUTHORIZATION_REL = f"{REPORT_REL}/AUTHORIZATION.json"
PROTOCOL_REL = f"{REPORT_REL}/PROTOCOL.md"
WRAPPER_REL = "scripts/cls_pos_poisson_bag2_e79_application_recovery_r1.py"
VERIFIER_REL = "scripts/verify_cls_pos_poisson_bag2_e79_application_recovery_r1.py"
TEST_REL = "tests/test_cls_pos_poisson_bag2_e79_application_recovery_r1.py"
RECOVERY_SOURCE_FILES = (WRAPPER_REL, VERIFIER_REL, TEST_REL, PROTOCOL_REL)

RECOVERY_REL = f"experiments/{frozen.CANDIDATE_ID}/APPLICATION_RECOVERY_R1"
RECOVERY_ATTEMPT_REL = f"{RECOVERY_REL}/RECOVERY_ATTEMPT.json"
APPLICATION_ATTEMPT_REL = f"{RECOVERY_REL}/APPLICATION_ATTEMPT.json"
APPLICATION_BUNDLE_REL = f"{RECOVERY_REL}/application_predictions.npz"
APPLICATION_RECEIPT_REL = f"{RECOVERY_REL}/APPLICATION_RECEIPT.json"
RECOVERY_QA_REL = f"{RECOVERY_REL}/RECOVERY_QA.json"
RECOVERY_FAILURE_REL = f"{RECOVERY_REL}/RECOVERY_FAILURE.json"
CSV_NAME = "classification_pos_poisson_bag2_e79_recovery_r1.csv"
EXPECTED_MISSING_ROWS = 1_062

ROOT_PASS_INPUT_BINDINGS = {
    "official_test_sha256": frozen.EXPECTED_TEST_SHA256,
    "a02_test_npz_sha256": frozen.EXPECTED_A02_TEST_NPZ_SHA256,
    "a02_test_receipt_sha256": frozen.EXPECTED_A02_TEST_RECEIPT_SHA256,
    "a02_model_sha256": frozen.EXPECTED_A02_MODEL_SHA256,
    "a02_fit_receipt_sha256": frozen.EXPECTED_A02_FIT_RECEIPT_SHA256,
    "a02_manifest_sha256": frozen.EXPECTED_A02_MANIFEST_SHA256,
    "baseline_csv_sha256": frozen.EXPECTED_BASELINE_CSV_SHA256,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return frozen.H.canonical_sha256(value)


def _read_json(path: Path) -> dict[str, Any]:
    return frozen._read_json(path)


def _write_json_once(path: Path, body: dict[str, Any]) -> str:
    return frozen._write_json_once(path, body)


def _git_success(root: Path, *args: str) -> bool:
    result = subprocess.run(["git", "-C", str(root), *args], stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, check=False)
    return result.returncode == 0


def _verified_root_pass_bindings(
    repository: Path,
    audit_root: Path,
    *,
    original_verifier: Callable[..., Mapping[str, Any]] | None = None,
    pass_path: Path | None = None,
) -> dict[str, Any]:
    """Run the frozen verifier first, then forward its exact committed bindings.

    This adapts only the frozen verifier's compact return value. It repeats the
    committed file hash, self-hash, and all seven expected-value checks; it never
    supplies replacement input hashes or bypasses the frozen verifier.
    """
    verify = original_verifier or frozen.verify_root_pass
    verified = dict(verify(Path(repository), audit_root=Path(audit_root)))
    path = pass_path or Path(audit_root) / frozen.ROOT_PASS_REL
    passed = _read_json(path)
    file_sha = frozen.sha256_file(path)
    _require(file_sha == verified.get("pass_file_sha256"),
             "Committed ROOT_PASS bytes changed after frozen verification")
    body = dict(passed)
    pass_sha = body.pop("pass_sha256", None)
    _require(pass_sha == _canonical_sha256(body) and pass_sha == verified.get("pass_sha256"),
             "Committed ROOT_PASS self-hash differs from the original verifier result")
    _require(passed.get("status") == "PASS_CLS_POS_POISSON_BAG2_E79_SOURCE_AND_APPLICATION"
             and passed.get("candidate_id") == frozen.CANDIDATE_ID,
             "Committed ROOT_PASS candidate/status differs")
    for key, expected in ROOT_PASS_INPUT_BINDINGS.items():
        _require(passed.get(key) == expected,
                 f"Committed ROOT_PASS binding mismatch: {key}")
    freeze_path = Path(audit_root) / frozen.FREEZE_REL
    freeze = _read_json(freeze_path)
    _require(verified.get("freeze_signature") == freeze.get("freeze_signature")
             and passed.get("prefit_freeze_signature") == freeze.get("freeze_signature")
             and passed.get("prefit_freeze_file_sha256") == frozen.sha256_file(freeze_path),
             "Committed ROOT_PASS verifier summary is bound to another source freeze")
    return {**verified, **{key: passed[key] for key in ROOT_PASS_INPUT_BINDINGS}}


def _fit_snapshot(audit_root: Path) -> dict[str, Any]:
    candidate_root = Path(audit_root) / frozen.OUTPUT_REL
    freeze = _read_json(Path(audit_root) / frozen.FREEZE_REL)
    qa = frozen_verifier.verify_fit_set(candidate_root,
                                        freeze_signature=freeze["freeze_signature"])
    ledger_path = candidate_root / "fit_ledger.jsonl"
    events = frozen._ledger_events(ledger_path)
    starts = [event for event in events if event.get("event") == "FIT_STARTED"]
    saves = [event for event in events if event.get("event") == "FIT_SAVED"]
    failures = [event for event in events if event.get("event") == "FIT_FAILED"]
    _require(len(starts) == len(saves) == 2 and not failures
             and {event.get("fit_id") for event in starts} == set(frozen.H.FIT_IDS)
             and {event.get("fit_id") for event in saves} == set(frozen.H.FIT_IDS),
             "Recovery requires exactly the verified historical two-fit ledger")
    fit_qa_path = Path(audit_root) / frozen.FIT_QA_REL
    fit_qa = _read_json(fit_qa_path)
    fit_qa_body = dict(fit_qa)
    fit_qa_receipt = fit_qa_body.pop("receipt_sha256", None)
    _require(fit_qa_receipt == _canonical_sha256(fit_qa_body)
             and fit_qa.get("fit_ids") == list(frozen.H.FIT_IDS)
             and fit_qa.get("status") == "PASS_TWO_E79_FITS_INDEPENDENTLY_VERIFIED",
             "Existing fit QA receipt is invalid or does not bind exactly two fits")
    return {
        "fit_set": qa,
        "fit_ledger_sha256": frozen.sha256_file(ledger_path),
        "fit_qa_sha256": frozen.sha256_file(fit_qa_path),
        "reservation_sha256": frozen.sha256_file(Path(audit_root) / frozen.RESERVATION_REL),
        "fit_count": 2,
        "fit_ids": list(frozen.H.FIT_IDS),
        "retry_allowed": False,
    }


def _recovery_csv_path(audit_root: Path, repository: Path) -> tuple[Path, str]:
    absolute = (Path(audit_root) / RECOVERY_REL / CSV_NAME).resolve()
    relative = os.path.relpath(absolute, start=Path(repository).resolve())
    _require((Path(repository).resolve() / relative).resolve() == absolute,
             "Recovery CSV path does not resolve to the isolated output namespace")
    return absolute, relative.replace("\\", "/")


def _recovery_paths(audit_root: Path) -> dict[str, Path]:
    root = Path(audit_root)
    return {
        "attempt": root / RECOVERY_ATTEMPT_REL,
        "application_attempt": root / APPLICATION_ATTEMPT_REL,
        "bundle": root / APPLICATION_BUNDLE_REL,
        "receipt": root / APPLICATION_RECEIPT_REL,
        "qa": root / RECOVERY_QA_REL,
        "failure": root / RECOVERY_FAILURE_REL,
    }


def create_recovery_authorization(repository: Path, *, audit_root: Path = AUDIT_ROOT) -> dict[str, Any]:
    root = Path(audit_root).resolve()
    repo = Path(repository).resolve()
    auth_path = root / AUTHORIZATION_REL
    _require(not auth_path.exists(), "Recovery authorization is write-once; refusing overwrite")
    recovery_root = root / RECOVERY_REL
    _require(not recovery_root.exists(), "Recovery runtime namespace already exists")

    freeze_check = frozen.verify_source_freeze(repo, audit_root=root)
    root_pass = _verified_root_pass_bindings(repo, root)
    fits = _fit_snapshot(root)
    previous_attempt = root / frozen.APP_ATTEMPT_REL
    _require(previous_attempt.is_file(), "Original consumed Test-application attempt is missing")
    previous_attempt_body = _read_json(previous_attempt)
    _require(previous_attempt_body.get("status") == "ONE_E79_TEST_APPLICATION_ATTEMPT_STARTED",
             "Original application attempt status differs; refusing recovery scope")
    previous_attempt_sha = frozen.sha256_file(previous_attempt)
    _require(not (root / frozen.APP_BUNDLE_REL).exists()
             and not (root / frozen.APP_RECEIPT_REL).exists()
             and not (repo / frozen.CSV_REL).exists(),
             "Original application outputs exist; recovery authorization scope is stale")

    source_commit = frozen._git(root, "rev-parse", "HEAD")
    recovery_sources: dict[str, str] = {}
    for relative in RECOVERY_SOURCE_FILES:
        path = root / relative
        _require(path.is_file(), f"Recovery source is missing: {relative}")
        source_hash = frozen.stable_file_sha256(path, relative)
        _require(frozen._tree_stable_sha256(root, source_commit, relative) == source_hash,
                 f"Recovery source is not committed at source commit: {relative}")
        recovery_sources[relative] = source_hash

    csv_path, csv_relative = _recovery_csv_path(root, repo)
    paths = _recovery_paths(root)
    _require(all(not path.exists() for path in paths.values()) and not csv_path.exists(),
             "Recovery output path is not fresh")
    body: dict[str, Any] = {
        "schema_version": 1,
        "recovery_id": RECOVERY_ID,
        "status": "AUTHORIZED_ONE_RECOVERY_APPLICATION_ZERO_FITS",
        "authorization_source": {
            "source_thread_id": "01a19569-16e0-7253-aa14-57a0a3160e03",
            "scope": "One Test application in a new recovery namespace; zero fits; no retry",
        },
        "audit_root": str(root),
        "official_repository": str(repo),
        "official_repository_head": frozen._git(repo, "rev-parse", "HEAD"),
        "recovery_source_commit": source_commit,
        "recovery_source_hashes": recovery_sources,
        "frozen_source": {
            "candidate_id": frozen.CANDIDATE_ID,
            "source_commit": freeze_check["audit_source_commit"],
            "source_hashes": _read_json(root / frozen.FREEZE_REL)["audit_source_hashes"],
            "freeze_path": frozen.FREEZE_REL,
            "freeze_signature": freeze_check["freeze_signature"],
            "freeze_file_sha256": freeze_check["freeze_file_sha256"],
            "root_pass_path": frozen.ROOT_PASS_REL,
            "root_pass_sha256": root_pass["pass_sha256"],
            "root_pass_file_sha256": root_pass["pass_file_sha256"],
            "reviewed_head": root_pass["reviewed_head"],
            "current_head": root_pass["current_head"],
            "official_repository_head": freeze_check["official_repository_head"],
            "seven_input_bindings": {key: root_pass[key] for key in ROOT_PASS_INPUT_BINDINGS},
        },
        "historical_failed_attempt": {
            "path": frozen.APP_ATTEMPT_REL,
            "sha256": previous_attempt_sha,
            "status": previous_attempt_body["status"],
            "preserved": True,
            "retry_of_old_attempt": False,
        },
        "fit_set": fits,
        "recovery_cap": {
            "application_attempts": 1,
            "official_fit_count": 0,
            "retry_allowed": False,
            "seed_or_recipe_change": False,
        },
        "outputs": {
            "recovery_namespace": RECOVERY_REL,
            "recovery_attempt": RECOVERY_ATTEMPT_REL,
            "application_attempt": APPLICATION_ATTEMPT_REL,
            "bundle": APPLICATION_BUNDLE_REL,
            "receipt": APPLICATION_RECEIPT_REL,
            "qa": RECOVERY_QA_REL,
            "csv_absolute_path": str(csv_path),
            "csv_relative_to_official_repository": csv_relative,
        },
        "resource_policy": {
            "initial_available_ram_bytes": frozen.GIB,
            "post_application_reserve_bytes": frozen.POST_FIT_RESERVE_BYTES,
            "max_threads": frozen.MAX_THREADS,
            "shared_research_lock_required": True,
            "official_cgeom_lock_required": True,
        },
        "data_boundary": {
            "test_feature_columns_only": True,
            "test_targets_loaded": False,
            "sample_submission_opened": False,
            "kaggle_contacted": False,
        },
    }
    body["authorization_signature"] = _canonical_sha256(body)
    file_sha = _write_json_once(auth_path, body)
    return {"status": body["status"], "authorization_path": AUTHORIZATION_REL,
            "authorization_file_sha256": file_sha,
            "authorization_signature": body["authorization_signature"],
            "old_attempt_sha256": previous_attempt_sha,
            "fit_count": 0, "application_cap": 1}


def verify_recovery_authorization(repository: Path, *, audit_root: Path = AUDIT_ROOT,
                                  require_fresh: bool = True) -> dict[str, Any]:
    root = Path(audit_root).resolve()
    repo = Path(repository).resolve()
    auth_path = root / AUTHORIZATION_REL
    auth = _read_json(auth_path)
    receipt_sha = auth.get("receipt_sha256")
    auth_body = dict(auth)
    auth_body.pop("receipt_sha256", None)
    auth_signature = auth_body.pop("authorization_signature", None)
    _require(receipt_sha == _canonical_sha256(auth_body | {"authorization_signature": auth_signature})
             and auth_signature == _canonical_sha256(auth_body),
             "Recovery authorization self-hash/signature mismatch")
    _require(auth.get("recovery_id") == RECOVERY_ID
             and auth.get("status") == "AUTHORIZED_ONE_RECOVERY_APPLICATION_ZERO_FITS"
             and auth.get("audit_root") == str(root)
             and auth.get("official_repository") == str(repo)
             and auth.get("recovery_cap") == {
                 "application_attempts": 1, "official_fit_count": 0,
                 "retry_allowed": False, "seed_or_recipe_change": False},
             "Recovery authorization identity/cap mismatch")
    head = frozen._git(root, "rev-parse", "HEAD")
    _require(_git_success(root, "merge-base", "--is-ancestor",
                          auth["recovery_source_commit"], head),
             "Recovery source commit is not an ancestor of current HEAD")
    for relative, expected_hash in auth["recovery_source_hashes"].items():
        path = root / relative
        _require(frozen.stable_file_sha256(path, relative) == expected_hash
                 and frozen._tree_stable_sha256(root, auth["recovery_source_commit"], relative)
                 == expected_hash,
                 f"Recovery source changed after authorization: {relative}")
    _require(frozen._tree_stable_sha256(root, head, AUTHORIZATION_REL)
             == frozen.stable_file_sha256(auth_path, AUTHORIZATION_REL),
             "Recovery authorization is not committed at current HEAD")

    freeze_check = frozen.verify_source_freeze(repo, audit_root=root)
    root_pass = _verified_root_pass_bindings(repo, root)
    frozen_binding = auth["frozen_source"]
    _require(freeze_check["freeze_signature"] == frozen_binding["freeze_signature"]
             and freeze_check["freeze_file_sha256"] == frozen_binding["freeze_file_sha256"]
             and root_pass["pass_sha256"] == frozen_binding["root_pass_sha256"]
             and root_pass["pass_file_sha256"] == frozen_binding["root_pass_file_sha256"],
             "Frozen R2 or ROOT_PASS changed since recovery authorization")
    _require(frozen._git(repo, "rev-parse", "HEAD") == auth["official_repository_head"],
             "Official repository HEAD changed after recovery authorization")

    old_attempt = root / auth["historical_failed_attempt"]["path"]
    _require(frozen.sha256_file(old_attempt) == auth["historical_failed_attempt"]["sha256"],
             "Historical failed attempt bytes changed")
    current_fits = _fit_snapshot(root)
    _require(current_fits == auth["fit_set"],
             "The verified original fit set/ledger changed after recovery authorization")
    csv_path, _csv_relative = _recovery_csv_path(root, repo)
    paths = _recovery_paths(root)
    if require_fresh:
        _require(all(not path.exists() for path in paths.values()) and not csv_path.exists(),
                 "Recovery attempt/output already exists; no retry or overwrite")
    return {"status": "PASS_RECOVERY_AUTHORIZATION",
            "authorization_file_sha256": frozen.sha256_file(auth_path),
            "authorization_signature": auth_signature,
            "root_pass_sha256": root_pass["pass_sha256"],
            "freeze_signature": freeze_check["freeze_signature"],
            "fit_count": current_fits["fit_count"], "application_cap": 1,
            "current_head": head}


@contextmanager
def _recovery_runner_bindings(repository: Path, audit_root: Path):
    original_verify = frozen.verify_root_pass
    old_values = {
        "verify_root_pass": frozen.verify_root_pass,
        "APP_ATTEMPT_REL": frozen.APP_ATTEMPT_REL,
        "APP_BUNDLE_REL": frozen.APP_BUNDLE_REL,
        "APP_RECEIPT_REL": frozen.APP_RECEIPT_REL,
        "CSV_REL": frozen.CSV_REL,
    }

    def verified_pass_with_forwarded_bindings(repo: Path, *, audit_root: Path = AUDIT_ROOT):
        return _verified_root_pass_bindings(repo, audit_root,
                                            original_verifier=original_verify)

    _csv_path, csv_relative = _recovery_csv_path(audit_root, repository)
    frozen.verify_root_pass = verified_pass_with_forwarded_bindings
    frozen.APP_ATTEMPT_REL = APPLICATION_ATTEMPT_REL
    frozen.APP_BUNDLE_REL = APPLICATION_BUNDLE_REL
    frozen.APP_RECEIPT_REL = APPLICATION_RECEIPT_REL
    frozen.CSV_REL = csv_relative
    try:
        yield
    finally:
        for name, value in old_values.items():
            setattr(frozen, name, value)


def _official_result_qa(repository: Path, *, audit_root: Path,
                        authorization: Mapping[str, Any],
                        application_result: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(audit_root)
    repo = Path(repository)
    recovery_paths = _recovery_paths(root)
    baseline = repo / frozen.BASELINE_CSV_REL
    # Read only IDs for independent order verification. No target, sample
    # submission, or model-selection data is accessed here.
    test_ids = frozen.pd.read_csv(repo / frozen.TEST_REL, usecols=["record_id"],
                                  dtype={"record_id": "string"})["record_id"]
    _require(not test_ids.isna().any() and test_ids.is_unique,
             "Official Test IDs are missing or duplicated during readback")
    qa = recovery_verifier.verify_recovery_application(
        recovery_root=root / RECOVERY_REL,
        fit_root=root / frozen.OUTPUT_REL,
        receipt_path=recovery_paths["receipt"],
        bundle_path=recovery_paths["bundle"],
        csv_path=_recovery_csv_path(root, repo)[0],
        baseline_csv_path=baseline,
        expected_baseline_sha256=frozen.EXPECTED_BASELINE_CSV_SHA256,
        expected_fit_set=authorization["fit_set"],
        expected_missing_rows=EXPECTED_MISSING_ROWS,
        authorization_file_sha256=frozen.sha256_file(root / AUTHORIZATION_REL),
        expected_test_ids=test_ids.astype(str).tolist(),
        expected_test_sha256=frozen.EXPECTED_TEST_SHA256,
        expected_csv_relative_path=authorization["outputs"]["csv_relative_to_official_repository"],
    )
    return {**qa, "application_runner_result": dict(application_result)}


def run_official_recovery(repository: Path, *, audit_root: Path = AUDIT_ROOT) -> dict[str, Any]:
    root = Path(audit_root).resolve()
    repo = Path(repository).resolve()
    auth_check = verify_recovery_authorization(repo, audit_root=root, require_fresh=True)
    auth = _read_json(root / AUTHORIZATION_REL)
    paths = _recovery_paths(root)
    rec_attempt_body = {
        "schema_version": 1,
        "recovery_id": RECOVERY_ID,
        "status": "ONE_RECOVERY_APPLICATION_ATTEMPT_STARTED",
        "authorization_file_sha256": auth_check["authorization_file_sha256"],
        "authorization_signature": auth_check["authorization_signature"],
        "historical_failed_attempt_sha256": auth["historical_failed_attempt"]["sha256"],
        "source_commit": auth["frozen_source"]["source_commit"],
        "freeze_signature": auth_check["freeze_signature"],
        "root_pass_sha256": auth_check["root_pass_sha256"],
        "fit_ledger_sha256_before": auth["fit_set"]["fit_ledger_sha256"],
        "fit_receipts_before": auth["fit_set"]["fit_set"]["fit_receipts"],
        "authorized_fit_count": 0,
        "application_attempt_cap": 1,
        "retry_allowed": False,
        "test_targets_allowed": False,
        "sample_submission_allowed": False,
        "kaggle_allowed": False,
    }
    _write_json_once(paths["attempt"], rec_attempt_body)
    try:
        with _recovery_runner_bindings(repo, root):
            application_result = frozen.apply_test_once(repo, audit_root=root)
    except Exception as exc:
        # The persisted recovery attempt is consumed even if application fails.
        _write_json_once(paths["failure"], {
            "schema_version": 1, "recovery_id": RECOVERY_ID,
            "status": "RECOVERY_APPLICATION_FAILED_CONSUMED_NO_RETRY",
            "authorization_file_sha256": auth_check["authorization_file_sha256"],
            "historical_failed_attempt_sha256": auth["historical_failed_attempt"]["sha256"],
            "error_type": type(exc).__name__, "error": str(exc),
            "retry_allowed": False, "official_fit_count": 0,
        })
        raise RuntimeError(f"Recovery application consumed; no retry: {exc}") from exc

    try:
        final_qa = _official_result_qa(repo, audit_root=root, authorization=auth,
                                       application_result=application_result)
    except Exception as exc:
        _write_json_once(paths["failure"], {
            "schema_version": 1, "recovery_id": RECOVERY_ID,
            "status": "RECOVERY_READBACK_FAILED_CONSUMED_NO_RETRY",
            "authorization_file_sha256": auth_check["authorization_file_sha256"],
            "application_attempt_sha256": frozen.sha256_file(paths["application_attempt"]),
            "application_receipt_sha256": frozen.sha256_file(paths["receipt"]),
            "error_type": type(exc).__name__, "error": str(exc),
            "retry_allowed": False, "official_fit_count": 0,
        })
        raise RuntimeError(f"Recovery application persisted but readback failed; no retry: {exc}") from exc
    final_qa_body = {
        "schema_version": 1,
        "recovery_id": RECOVERY_ID,
        "status": final_qa["status"],
        "authorization_file_sha256": auth_check["authorization_file_sha256"],
        "authorization_signature": auth_check["authorization_signature"],
        "source_commit": auth["frozen_source"]["source_commit"],
        "freeze_signature": auth_check["freeze_signature"],
        "root_pass_sha256": auth_check["root_pass_sha256"],
        "historical_failed_attempt_sha256": auth["historical_failed_attempt"]["sha256"],
        "recovery_attempt_sha256": frozen.sha256_file(paths["attempt"]),
        "application_attempt_sha256": frozen.sha256_file(paths["application_attempt"]),
        "application_receipt_sha256": frozen.sha256_file(paths["receipt"]),
        "application_bundle_sha256": frozen.sha256_file(paths["bundle"]),
        "csv_path": str(_recovery_csv_path(root, repo)[0]),
        "csv_sha256": frozen.sha256_file(_recovery_csv_path(root, repo)[0]),
        "csv_relative_to_official_repository": _recovery_csv_path(root, repo)[1],
        "verification": final_qa,
        "fit_set_unchanged": True,
        "official_fit_count_during_recovery": 0,
        "application_attempts": 1,
        "test_targets_or_sample_submission_accessed": False,
        "kaggle_contacted": False,
    }
    _write_json_once(paths["qa"], final_qa_body)
    return final_qa_body


def run_synthetic_recovery(output_dir: Path) -> dict[str, Any]:
    """Run the unchanged frozen application runner against a synthetic fixture."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base_root = output_dir / "synthetic_audit"
    repository = output_dir / "synthetic_official_repo"
    candidate_root = base_root / frozen.OUTPUT_REL
    freeze_signature = "SYNTHETIC_FREEZE"
    frozen.run_synthetic(candidate_root)

    # Promote the isolated synthetic fit receipts into the same path contract
    # expected by the frozen verifier. No official fit artifacts are touched.
    candidate_root.mkdir(parents=True, exist_ok=True)
    synthetic_ledger = candidate_root / "synthetic_fit_ledger.jsonl"
    ledger_path = candidate_root / "fit_ledger.jsonl"
    ledger_path.write_bytes(synthetic_ledger.read_bytes())
    freeze_path = base_root / frozen.FREEZE_REL
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_path.write_text(json.dumps({"freeze_signature": freeze_signature}, indent=2) + "\n",
                           encoding="utf-8")
    frozen._write_json_once(candidate_root / "FIT_QA.json", {
        "status": "PASS_TWO_E79_FITS_INDEPENDENTLY_VERIFIED",
        "fit_ids": list(frozen.H.FIT_IDS), "freeze_signature": freeze_signature,
    })
    reservation_path = candidate_root / "RESERVATION.json"
    reservation_path.write_text('{"mode":"synthetic","fit_count":2}\n', encoding="utf-8")

    fit_set_before = _fit_snapshot(base_root)
    inputs = frozen._synthetic_inputs()
    n_rows = 29_421
    row_index = np.arange(n_rows, dtype=np.int64) % len(inputs["ids"])
    position = inputs["x_test"][row_index].tocsr()
    offset = inputs["offset_test"][row_index].copy()
    present = inputs["present"][row_index].copy()
    p_a02 = inputs["p_a02"][row_index].copy()
    ids = np.asarray([f"RECOVERY-SYN-{i:05d}" for i in range(n_rows)], dtype=str)
    cycles = np.resize(np.arange(80, 92, dtype=np.int64), n_rows)
    test_frame = frozen.pd.DataFrame({
        "record_id": ids, "cycle_code": cycles,
        "signal_stream": np.where(present, "SYNTHETIC", ""),
    })

    repository.mkdir(parents=True, exist_ok=True)
    test_path = repository / frozen.TEST_REL
    test_path.parent.mkdir(parents=True, exist_ok=True)
    test_frame.to_csv(test_path, index=False)
    a02_npz_path = repository / frozen.A02_TEST_NPZ_REL
    a02_npz_path.parent.mkdir(parents=True, exist_ok=True)
    frozen._save_npz_once(a02_npz_path, {"record_ids": ids,
                                        "candidate_probability": p_a02})
    a02_fit_state = {"synthetic_model": "full_train_stub"}
    a02_model_path = repository / frozen.A02_MODEL_REL
    a02_model_path.parent.mkdir(parents=True, exist_ok=True)
    a02_model_path.write_bytes(b"synthetic fitted model placeholder\n")
    a02_state_path = repository / frozen.A02_FIT_STATE_REL
    a02_state_path.write_text(json.dumps(a02_fit_state, sort_keys=True) + "\n", encoding="utf-8")
    a02_fit_receipt_path = repository / frozen.A02_FIT_RECEIPT_REL
    a02_fit_receipt_path.parent.mkdir(parents=True, exist_ok=True)
    frozen._write_json_once(a02_fit_receipt_path, {
        "model_sha256": frozen.sha256_file(a02_model_path), "test_data_used": False,
        "query_labels_passed_to_fit": False, "fit_rows": frozen.H.TRAIN_ROWS,
        "freeze_signature": "SYNTHETIC_A02_FREEZE",
    })
    a02_fit_receipt = _read_json(a02_fit_receipt_path)
    a02_test_receipt_path = repository / frozen.A02_TEST_RECEIPT_REL
    a02_test_receipt_path.write_text(json.dumps({
        "npz_sha256": frozen.sha256_file(a02_npz_path),
        "fit_receipt_sha256": a02_fit_receipt["receipt_sha256"],
        "freeze_signature": a02_fit_receipt["freeze_signature"],
        "official_test_targets_loaded": False,
    }, sort_keys=True) + "\n", encoding="utf-8")
    a02_manifest_path = repository / frozen.A02_MANIFEST_REL
    a02_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    a02_manifest_path.write_text(json.dumps({
        "test_target_loaded": False, "official_train_rows": frozen.H.TRAIN_ROWS,
    }, sort_keys=True) + "\n", encoding="utf-8")

    original_gamma_path = repository / frozen.ORIGINAL_GAMMA_REL
    original_gamma_path.parent.mkdir(parents=True, exist_ok=True)
    original_gamma_path.write_bytes(b"synthetic original gamma placeholder\n")
    gamma17, _ = frozen_verifier.verify_fit_artifact(
        candidate_root, frozen.H.fit_id_for_seed(17), freeze_signature=freeze_signature)
    gamma137, _ = frozen_verifier.verify_fit_artifact(
        candidate_root, frozen.H.fit_id_for_seed(137), freeze_signature=freeze_signature)
    policy = frozen.H.build_probabilities(position, offset, inputs["original_gamma"],
                                          gamma17, gamma137, p_a02, present)
    champion = frozen.H.apply_thresholds(policy["probability_champion"], present)
    baseline_path = repository / frozen.BASELINE_CSV_REL
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    with baseline_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["record_id", "artifact_signal"])
        writer.writerows(zip(ids.tolist(), champion.astype(int).tolist()))

    synthetic_bindings = {
        "official_test_sha256": frozen.sha256_file(test_path),
        "a02_test_npz_sha256": frozen.sha256_file(a02_npz_path),
        "a02_test_receipt_sha256": frozen.sha256_file(a02_test_receipt_path),
        "a02_model_sha256": frozen.sha256_file(a02_model_path),
        "a02_fit_receipt_sha256": frozen.sha256_file(a02_fit_receipt_path),
        "a02_manifest_sha256": frozen.sha256_file(a02_manifest_path),
        "baseline_csv_sha256": frozen.sha256_file(baseline_path),
    }
    root_pass_path = base_root / frozen.ROOT_PASS_REL
    root_pass_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_file_sha = frozen.sha256_file(freeze_path)
    root_pass_body = {
        "schema_version": 1,
        "status": "PASS_CLS_POS_POISSON_BAG2_E79_SOURCE_AND_APPLICATION",
        "candidate_id": frozen.CANDIDATE_ID,
        "reviewed_head": "SYNTHETIC_REVIEWED_HEAD",
        "prefit_freeze_signature": freeze_signature,
        "prefit_freeze_file_sha256": freeze_file_sha,
        "authorized_fit_ids": list(frozen.H.FIT_IDS), "fit_cap": 2,
        "retry_allowed": False, "test_application_authorized": True,
        "one_test_application_only": True, "baseline_decision_replay_required": True,
        "test_target_access_authorized": False,
        "sample_submission_access_authorized": False, "kaggle_access_authorized": False,
        **synthetic_bindings,
    }
    root_pass_body["pass_sha256"] = _canonical_sha256(root_pass_body)
    root_pass_path.write_text(json.dumps(root_pass_body, sort_keys=True, indent=2) + "\n",
                              encoding="utf-8")

    class SyntheticModel:
        def audit_state(self):
            return a02_fit_state

    class SyntheticGammaRunner:
        def __init__(self):
            self.r3 = type("R3", (), {
                "_shared_job_lock": staticmethod(lambda: nullcontext()),
                "_cgeom_local_lock": staticmethod(lambda _repo: nullcontext()),
                "read_feature_csv": staticmethod(lambda _path, _columns: (test_frame.copy(),
                                                                           ["record_id", "cycle_code", "signal_stream"])),
            })()
            self.cgeom = type("CGeom", (), {"CLASSIFICATION_FEATURES":
                                             ["record_id", "cycle_code", "signal_stream"]})()
            self.positional = type("Positional", (), {"sha256_csr":
                staticmethod(lambda matrix: _sha256_bytes(matrix.data.tobytes()))})()

        @staticmethod
        def _build_feature_matrices_for_rows(_repo, _test):
            return {"position": position, "offset": offset, "missing_mask": ~present}

    synthetic_runner = SyntheticGammaRunner()
    synthetic_summary = {
        "status": "PASS_E79_ROOT_PASS", "pass_sha256": root_pass_body["pass_sha256"],
        "pass_file_sha256": frozen.sha256_file(root_pass_path),
        "current_head": "SYNTHETIC_HEAD", "reviewed_head": "SYNTHETIC_REVIEWED_HEAD",
        "freeze_signature": freeze_signature,
    }

    def synthetic_root_pass(_repository, *, audit_root):
        return synthetic_summary

    def synthetic_source_freeze(_repository, *, audit_root):
        return {"status": "PASS_SYNTHETIC_FREEZE", "freeze_signature": freeze_signature,
                "freeze_file_sha256": freeze_file_sha, "audit_source_commit": "SYNTHETIC"}

    def synthetic_joblib_load(path):
        path = Path(path).resolve()
        if path == original_gamma_path.resolve():
            return {"gamma": inputs["original_gamma"], "fit_rows": frozen.H.TRAIN_ROWS}
        if path == a02_model_path.resolve():
            return SyntheticModel()
        raise RuntimeError(f"Unexpected synthetic joblib read: {path}")

    patches = [
        mock.patch.object(frozen, "verify_source_freeze", synthetic_source_freeze),
        mock.patch.object(frozen, "verify_root_pass", synthetic_root_pass),
        mock.patch.object(frozen, "_git", lambda *_args: "SYNTHETIC_HEAD"),
        mock.patch.object(frozen, "_load_oldgamma", lambda _repo: synthetic_runner),
        mock.patch.object(frozen, "_resource_snapshot", lambda _label: {"available_bytes": 2 * frozen.GIB}),
        mock.patch.object(frozen, "_require_resource", lambda *_args: None),
        mock.patch.object(frozen.joblib, "load", synthetic_joblib_load),
        mock.patch.object(frozen.H, "EXPECTED_POSITION_SHAPE", (frozen.H.TRAIN_ROWS, position.shape[1])),
        mock.patch.object(frozen, "EXPECTED_TEST_SHA256", synthetic_bindings["official_test_sha256"]),
        mock.patch.object(frozen, "EXPECTED_A02_TEST_NPZ_SHA256", synthetic_bindings["a02_test_npz_sha256"]),
        mock.patch.object(frozen, "EXPECTED_A02_TEST_RECEIPT_SHA256", synthetic_bindings["a02_test_receipt_sha256"]),
        mock.patch.object(frozen, "EXPECTED_A02_MODEL_SHA256", synthetic_bindings["a02_model_sha256"]),
        mock.patch.object(frozen, "EXPECTED_A02_FIT_RECEIPT_SHA256", synthetic_bindings["a02_fit_receipt_sha256"]),
        mock.patch.object(frozen, "EXPECTED_A02_MANIFEST_SHA256", synthetic_bindings["a02_manifest_sha256"]),
        mock.patch.object(frozen, "EXPECTED_BASELINE_CSV_SHA256", synthetic_bindings["baseline_csv_sha256"]),
        mock.patch.dict(ROOT_PASS_INPUT_BINDINGS, synthetic_bindings),
    ]
    for patcher in patches:
        patcher.start()
    try:
        recovery_attempt = base_root / RECOVERY_ATTEMPT_REL
        _write_json_once(recovery_attempt, {
            "schema_version": 1, "recovery_id": RECOVERY_ID,
            "status": "ONE_RECOVERY_APPLICATION_ATTEMPT_STARTED",
            "authorization_file_sha256": "SYNTHETIC_AUTHORIZATION",
            "authorized_fit_count": 0, "application_attempt_cap": 1,
            "retry_allowed": False, "test_targets_allowed": False,
            "sample_submission_allowed": False, "kaggle_allowed": False,
        })
        with _recovery_runner_bindings(repository, base_root):
            first = frozen.apply_test_once(repository, audit_root=base_root)
        paths = _recovery_paths(base_root)
        csv_path, csv_relative = _recovery_csv_path(base_root, repository)
        qa = recovery_verifier.verify_recovery_application(
            recovery_root=base_root / RECOVERY_REL, fit_root=candidate_root,
            receipt_path=paths["receipt"], bundle_path=paths["bundle"], csv_path=csv_path,
            baseline_csv_path=baseline_path,
            expected_baseline_sha256=synthetic_bindings["baseline_csv_sha256"],
            expected_fit_set=fit_set_before, expected_missing_rows=None,
            authorization_file_sha256="SYNTHETIC_AUTHORIZATION",
            expected_test_ids=ids.tolist(), expected_test_sha256=synthetic_bindings["official_test_sha256"],
            expected_csv_relative_path=csv_relative,
        )
        hashes_before_resume = {key: frozen.sha256_file(paths[key])
                                for key in ("attempt", "application_attempt", "bundle", "receipt")}
        with _recovery_runner_bindings(repository, base_root):
            resumed = frozen.apply_test_once(repository, audit_root=base_root)
        hashes_after_resume = {key: frozen.sha256_file(paths[key])
                               for key in ("attempt", "application_attempt", "bundle", "receipt")}
        _require(hashes_before_resume == hashes_after_resume,
                 "Synthetic persistence resume changed one-shot outputs")
        fit_set_after = _fit_snapshot(base_root)
        _require(fit_set_after == fit_set_before,
                 "Synthetic application changed the pre-existing fit set")
        return {
            "status": "PASS_SYNTHETIC_FROZEN_RUNNER_RECOVERY_E2E",
            "application_attempts": 1, "application_resume_readbacks": 1,
            "official_fit_count": 0, "synthetic_fixture_fit_count": 2,
            "fit_set_unchanged": True, "persistence_hashes_unchanged_on_resume": True,
            "rows": qa["rows"], "missing_rows": qa["missing_rows"],
            "candidate_changed_decisions": qa["candidate_changed_decisions"],
            "output_qa": qa, "first_runner_result": first, "resume_runner_result": resumed,
            "binding_fields_forwarded": sorted(ROOT_PASS_INPUT_BINDINGS),
            "test_targets_or_sample_submission_accessed": False, "kaggle_contacted": False,
            "recovery_root": str(base_root / RECOVERY_REL),
            "candidate_csv_path": str(csv_path),
        }
    finally:
        for patcher in reversed(patches):
            patcher.stop()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command", required=True, choices=("authorize", "apply", "synthetic"))
    parser.add_argument("--repository", type=Path, help="Explicit official TP2 repository root")
    parser.add_argument("--output-dir", type=Path, help="Temporary output path for synthetic-only QA")
    args = parser.parse_args(argv)
    try:
        if args.command == "synthetic":
            _require(args.output_dir is not None, "--output-dir is required for synthetic QA")
            result = run_synthetic_recovery(args.output_dir)
        else:
            _require(args.repository is not None, "--repository is required for official recovery")
            if args.command == "authorize":
                result = create_recovery_authorization(args.repository)
            else:
                result = run_official_recovery(args.repository)
        print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "ERROR", "error_type": type(exc).__name__,
                          "error": str(exc)}, sort_keys=True, ensure_ascii=False, indent=2),
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
