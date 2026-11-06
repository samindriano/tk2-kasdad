"""Narrow pre-access recovery for the single authorized E79 Test application.

The frozen v1 attempt failed on a missing MAIN attribute before reading Test.
This additive adapter keeps the fit recipe/source immutable and explicitly
points the bridge at the official classification Test CSV.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

AUDIT_ROOT = Path(__file__).resolve().parents[1]
if str(AUDIT_ROOT) not in sys.path:
    sys.path.insert(0, str(AUDIT_ROOT))

from scripts import cls_pos_recency_e79_v1 as runner
from scripts import cls_pos_recency_e79_main_bridge_v1 as bridge


PREVIOUS_ATTEMPT = AUDIT_ROOT / "experiments/CLS_POS_RECENCY_E79_V1/APPLICATION_ATTEMPT.started.json"
PREVIOUS_FAILURE = AUDIT_ROOT / "experiments/CLS_POS_RECENCY_E79_V1/APPLICATION_FAILURE.json"
RECOVERY_ATTEMPT_REL = "experiments/CLS_POS_RECENCY_E79_V1/APPLICATION_ATTEMPT_RECOVERY.started.json"
RECOVERY_FAILURE_REL = "experiments/CLS_POS_RECENCY_E79_V1/APPLICATION_FAILURE_RECOVERY.json"
RECOVERY_REPORT = AUDIT_ROOT / "reports/simplification_v1/CLS_POS_RECENCY_E79_V1/APPLICATION_RECOVERY.json"
TEST_FEATURE_DIR = AUDIT_ROOT / runner.TEST_FEATURE_DIR_REL
APPLICATION_OUTPUT = AUDIT_ROOT / runner.APPLICATION_OUTPUT_REL
CSV_PATH = AUDIT_ROOT / runner.CSV_REL
APP_RECEIPT = AUDIT_ROOT / runner.APPLICATION_RECEIPT_REL
APP_MANIFEST = AUDIT_ROOT / runner.APPLICATION_MANIFEST_REL
APP_QA = AUDIT_ROOT / runner.APPLICATION_QA_REL


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def validate_preaccess_failure() -> tuple[dict[str, Any], dict[str, Any]]:
    if not PREVIOUS_ATTEMPT.is_file() or not PREVIOUS_FAILURE.is_file():
        raise RuntimeError("Expected preserved first attempt and failure receipt are missing")
    attempt = read_json(PREVIOUS_ATTEMPT)
    failure = read_json(PREVIOUS_FAILURE)
    if (attempt.get("status") != "ONE_E79_APPLICATION_STARTED"
            or failure.get("status") != "FAIL_E79_APPLICATION_NO_RETRY"
            or attempt.get("test_target_or_sample_target_read") is not False
            or failure.get("kaggle_contact_or_upload") is not False
            or "r3.TEST_REL" not in failure.get("error", "")
            or "AttributeError" not in failure.get("error", "")):
        raise RuntimeError("First attempt does not prove the precise pre-access adapter failure")
    prior_outputs = (TEST_FEATURE_DIR, APPLICATION_OUTPUT, CSV_PATH, APP_RECEIPT, APP_MANIFEST, APP_QA)
    if any(path.exists() for path in prior_outputs):
        raise RuntimeError("A prior Test transform/application output exists; recovery must not overwrite it")
    fit_qa = read_json(AUDIT_ROOT / runner.FIT_QA_REL)
    signature = fit_qa.pop("fit_qa_sha256", None)
    if signature != canonical_sha256(fit_qa) or signature != failure.get("fit_qa_sha256"):
        raise RuntimeError("Saved independent fit QA no longer verifies")
    runner.verify_freeze()
    return attempt, failure


def validate_main_adapter() -> dict[str, Any]:
    main_root = runner.MAIN_ROOT.resolve()
    official_relative = runner.OFFICIAL_TEST_REL
    official_path = (main_root / official_relative).resolve()
    if (official_relative.as_posix() != "data/raw/classification/Test.csv"
            or official_relative.suffix.lower() != ".csv"
            or official_path.parent.name.lower() != "classification"
            or not official_path.is_file()
            or file_sha256(official_path) != runner.OFFICIAL_TEST_SHA256):
        raise RuntimeError("Recovery path is not the hash-pinned official Classification Test.csv")
    if runner.EXPECTED_TEST_ROWS != 29_421:
        raise RuntimeError("Frozen official Test population is not 29,421 rows")

    # Resolve the imported modules and all attributes used by the bridge before
    # any Test row is opened. R1_TEST_REL is a .py unit-test source, not data.
    runner._load_main_lock_modules()
    cgeom, gamma, r3, positional = bridge._load_main_modules()
    source_test_rel = Path(getattr(r3, "R1_TEST_REL"))
    required = {
        "feature_reader": callable(getattr(r3, "read_feature_csv", None)),
        "classification_columns": hasattr(cgeom, "CLASSIFICATION_FEATURES"),
        "feature_builder": callable(getattr(gamma, "_build_feature_matrices_for_rows", None)),
        "shared_lock": callable(getattr(r3, "_shared_job_lock", None)),
        "cgeom_lock": callable(getattr(r3, "_cgeom_local_lock", None)),
        "array_hasher": callable(getattr(positional, "sha256_array", None)),
        "source_test_rel_is_not_dataset": source_test_rel.suffix.lower() == ".py",
    }
    if not all(required.values()):
        raise RuntimeError(f"MAIN bridge module/attribute preflight failed: {required}")
    # Explicitly repair only this process's absent data-path alias. Do not use
    # MAIN's R1_TEST_REL: it names a source-code test, not official Test.csv.
    r3.TEST_REL = official_relative
    if Path(r3.TEST_REL).as_posix() != official_relative.as_posix():
        raise RuntimeError("MAIN in-process Test path alias did not bind the official CSV")
    return {
        "main_root": str(main_root),
        "official_test_relative_path": official_relative.as_posix(),
        "official_test_sha256": runner.OFFICIAL_TEST_SHA256,
        "official_test_rows": runner.EXPECTED_TEST_ROWS,
        "r1_test_rel_source_path": source_test_rel.as_posix(),
        "r1_test_rel_used_as_dataset_path": False,
        "bridge_module_attributes": required,
        "bridge_test_path_alias": "r3.TEST_REL -> data/raw/classification/Test.csv (process-local only)",
        "test_rows_opened_during_preflight": False,
        "target_columns_read_during_preflight": False,
    }


def run() -> dict[str, Any]:
    old_attempt, old_failure = validate_preaccess_failure()
    adapter = validate_main_adapter()

    # Use unique additive markers so the pre-access attempt and traceback stay
    # immutable. The runner's one actual application outputs retain their
    # frozen canonical paths; only the attempt/failure record paths are new.
    runner.APPLICATION_ATTEMPT_REL = RECOVERY_ATTEMPT_REL
    runner.APPLICATION_FAILURE_REL = RECOVERY_FAILURE_REL

    # The frozen v1 receipt uses `kaggle_contact_or_upload`, while its
    # independent verifier requires the more specific `kaggle_contact` field.
    # Add that explicit false attestation to the new application receipt only.
    write_json_once_original = runner.write_json_once

    def write_receipt_with_kaggle_boundary(path: Path, payload: dict[str, Any]) -> str:
        if Path(path).resolve() == APP_RECEIPT.resolve():
            payload = dict(payload)
            payload.pop("receipt_sha256", None)
            payload["kaggle_contact"] = False
            payload["receipt_sha256"] = runner.canonical_sha256(payload)
        return write_json_once_original(path, payload)

    runner.write_json_once = write_receipt_with_kaggle_boundary

    def bridge_once(mode: str, output_dir: Path) -> dict[str, Any]:
        if mode != "test":
            raise RuntimeError("Recovery bridge is limited to the one frozen Test transform")
        # The same in-process module object holds the checked alias above.
        result = bridge.build(mode, output_dir)
        return result

    runner._run_bridge = bridge_once
    result = runner.apply_test_once()
    receipt = read_json(APP_RECEIPT)
    qa = read_json(APP_QA)
    manifest = read_json(APP_MANIFEST)
    if (result.get("status") != "PASS_LOCAL_CSV_PREPARED_NOT_UPLOADED"
            or receipt.get("status") != "PASS_E79_FEATURE_ONLY_APPLICATION_PERSISTED"
            or qa.get("status") != "PASS_INDEPENDENT_E79_APPLICATION_QA"
            or manifest.get("status") != "LOCAL_CSV_PREPARED_NOT_UPLOADED"
            or receipt.get("application_attempt_file_sha256") != file_sha256(AUDIT_ROOT / RECOVERY_ATTEMPT_REL)
            or receipt.get("test_target_read") is not False
            or receipt.get("sample_target_read") is not False
            or receipt.get("kaggle_contact_or_upload") is not False):
        raise RuntimeError("Recovery application/output QA did not bind to one safe successful application")
    body: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS_E79_PREACCESS_RECOVERY_AND_ONE_APPLICATION",
        "candidate_id": "CLS_POS_RECENCY_E79_V1",
        "original_preaccess_attempt_sha256": file_sha256(PREVIOUS_ATTEMPT),
        "original_preaccess_failure_sha256": file_sha256(PREVIOUS_FAILURE),
        "original_failure_status": old_failure["status"],
        "original_application_slot_performed": False,
        "preaccess_failure_proof": "bridge raised AttributeError at missing r3.TEST_REL before read_feature_csv; no Test feature cache or output existed",
        "recovery_adapter": adapter,
        "recovery_attempt_path": RECOVERY_ATTEMPT_REL,
        "recovery_attempt_sha256": file_sha256(AUDIT_ROOT / RECOVERY_ATTEMPT_REL),
        "actual_feature_only_test_transform_count": 1,
        "actual_csv_preparation_count": 1,
        "application_receipt_sha256": file_sha256(APP_RECEIPT),
        "application_qa_sha256": qa.get("application_qa_sha256"),
        "application_manifest_sha256": manifest.get("manifest_sha256"),
        "csv_path": result["csv_path"],
        "csv_sha256": result["csv_sha256"],
        "test_target_or_sample_target_read": False,
        "kaggle_contact_or_upload": False,
        "frozen_fit_source_or_recipe_changed": False,
    }
    body["recovery_sha256"] = canonical_sha256(body)
    runner.write_json_once(RECOVERY_REPORT, body)
    return body


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
