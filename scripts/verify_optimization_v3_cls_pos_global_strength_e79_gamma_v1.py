"""Independent readback checks for E79 gamma fit and application artifacts.

Official fit verification never re-reads target values. Official application
verification checks persisted predictions/CSV and hashes the bound Test source
only; it does not load Test rows, SampleSubmission, or labels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from scripts import optimization_v3_cls_pos_global_strength_e79_gamma_v1 as gamma
from src.models import optimization_v3_positional_text_pilot_v1 as positional


ROOT = Path(__file__).resolve().parents[1]
SYN_FIT_DIR = "fit"
SYN_APP_DIR = "application"


class VerificationError(RuntimeError):
    """Raised when a persisted gamma/application bundle fails read-only QA."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         allow_nan=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise VerificationError(f"Expected JSON object at {path}")
    return value


def _verify_receipt(path: Path) -> tuple[dict[str, Any], str]:
    receipt = _json(path)
    body = dict(receipt)
    digest = body.pop("receipt_sha256", None)
    if digest != _canonical_sha256(body):
        raise VerificationError(f"Receipt self-digest failed: {path}")
    return receipt, str(digest)


def verify_synthetic_fit(output_dir: Path) -> dict[str, Any]:
    fit_dir = output_dir / SYN_FIT_DIR
    attempt_path = fit_dir / "ATTEMPT.started.json"
    artifact_path = fit_dir / "gamma.npz"
    receipt_path = fit_dir / "fit_receipt.json"
    if not all(path.is_file() for path in (attempt_path, artifact_path, receipt_path)):
        raise VerificationError("Synthetic fit is missing an attempt marker, artifact, or receipt")
    attempt = _json(attempt_path)
    receipt, receipt_digest = _verify_receipt(receipt_path)
    with np.load(artifact_path, allow_pickle=False) as archive:
        if set(archive.files) != {"gamma_float64"}:
            raise VerificationError("Synthetic gamma artifact schema differs")
        vector = np.asarray(archive["gamma_float64"]).copy()
    checks = {
        "attempt_started_once": attempt.get("status") == "ONE_SYNTHETIC_GAMMA_FIT_ATTEMPT_STARTED" and attempt.get("fit_count") == 1,
        "one_shot_no_retry": attempt.get("retry_authorized") is False and receipt.get("retry_authorized") is False,
        "fit_completed": receipt.get("status") == "PASS_E79_SYNTHETIC_GAMMA_FIT_PERSISTED" and receipt.get("mode") == "synthetic",
        "optimizer_success": receipt.get("optimizer", {}).get("success") is True,
        "gamma_dtype_and_shape": vector.dtype == np.dtype(np.float64) and vector.shape == tuple(receipt.get("gamma_shape", [])),
        "gamma_finite": bool(np.isfinite(vector).all()),
        "gamma_array_digest": positional.sha256_array(vector) == receipt.get("gamma_sha256"),
        "gamma_file_digest": _sha256_file(artifact_path) == receipt.get("artifact_sha256"),
        "fit_count_one": receipt.get("fit_count") == 1,
        "no_test_sample_or_kaggle": receipt.get("test_or_sample_read") is False and receipt.get("test_inference_performed") is False and receipt.get("kaggle_contacted") is False,
    }
    if not all(checks.values()):
        raise VerificationError(f"Synthetic gamma receipt/artifact checks failed: {checks}")
    return {"status": "PASS_E79_SYNTHETIC_FIT_READBACK", "checks": checks,
            "receipt_sha256": receipt_digest, "artifact_sha256": _sha256_file(artifact_path),
            "gamma_shape": list(vector.shape), "fit_count": 1,
            "target_test_sample_or_kaggle_access": False}


def _read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        rows = list(reader)
    if not rows:
        raise VerificationError("CSV is empty")
    return rows[0], rows[1:]


def _verify_csv_against_arrays(csv_path: Path, *, ids: np.ndarray,
                               labels: np.ndarray, target_column: str) -> dict[str, Any]:
    header, rows = _read_csv(csv_path)
    checks = {
        "exact_header": header == ["record_id", target_column],
        "row_count": len(rows) == len(ids),
        "ids_exact_and_ordered": len(rows) == len(ids) and [row[0] for row in rows] == ids.astype(str).tolist(),
        "labels_integer_0_1": len(rows) == len(ids) and all(len(row) == 2 and row[1] in {"0", "1"} for row in rows),
    }
    if checks["labels_integer_0_1"] and checks["row_count"]:
        checks["labels_match_prediction_bundle"] = np.array_equal(
            np.asarray([int(row[1]) for row in rows], dtype=np.int8), labels.astype(np.int8))
    else:
        checks["labels_match_prediction_bundle"] = False
    if not all(checks.values()):
        raise VerificationError(f"CSV verification failed: {checks}")
    return {"checks": checks, "rows": len(rows), "csv_sha256": _sha256_file(csv_path)}


def verify_synthetic_application(output_dir: Path) -> dict[str, Any]:
    app_dir = output_dir / SYN_APP_DIR
    npz_path = app_dir / "application_predictions.npz"
    receipt_path = app_dir / "application_receipt.json"
    csv_path = app_dir / "application.csv"
    if not all(path.is_file() for path in (npz_path, receipt_path, csv_path)):
        raise VerificationError("Synthetic application bundle is incomplete")
    receipt, digest = _verify_receipt(receipt_path)
    with np.load(npz_path, allow_pickle=False) as archive:
        required = {"record_ids", "present_stream", "a02_probability", "probability", "labels"}
        if set(archive.files) != required:
            raise VerificationError("Synthetic application NPZ schema differs")
        arrays = {key: np.asarray(archive[key]).copy() for key in archive.files}
    ids = arrays["record_ids"].astype(str).reshape(-1)
    present = arrays["present_stream"].astype(bool).reshape(-1)
    p_a02 = arrays["a02_probability"].astype(np.float64).reshape(-1)
    probability = arrays["probability"].astype(np.float64).reshape(-1)
    labels = arrays["labels"].astype(np.int8).reshape(-1)
    if len({len(ids), len(present), len(p_a02), len(probability), len(labels)}) != 1:
        raise VerificationError("Synthetic application arrays are not aligned")
    expected_thresholds = np.where(present, gamma.recipe.PRESENT_THRESHOLD, gamma.recipe.MISSING_THRESHOLD)
    checks = {
        "receipt_status": receipt.get("status") == "PASS_E79_SYNTHETIC_APPLICATION_CSV_CREATED",
        "receipt_mode": receipt.get("mode") == "synthetic",
        "array_dtypes": bool(arrays["record_ids"].dtype.kind in "US"
                         and arrays["present_stream"].dtype == np.dtype(bool)
                         and arrays["a02_probability"].dtype == np.dtype(np.float64)
                         and arrays["probability"].dtype == np.dtype(np.float64)
                         and arrays["labels"].dtype.kind in "iu"
                         and np.isin(arrays["labels"], [0, 1]).all()),
        "ids_unique": len(set(ids.tolist())) == len(ids),
        "arrays_finite": bool(np.isfinite(p_a02).all() and np.isfinite(probability).all()),
        "probability_range": bool(np.all((p_a02 >= 0) & (p_a02 <= 1)) and np.all((probability >= 0) & (probability <= 1))),
        "missing_fallback_bit_exact": bool(np.array_equal(probability[~present], p_a02[~present])) and receipt.get("missing_fallback_bit_exact") is True,
        "thresholds_reproduce_labels": bool(np.array_equal(labels, (probability >= expected_thresholds).astype(np.int8))),
        "no_test_sample_or_kaggle": receipt.get("test_target_requested_or_loaded") is False and receipt.get("sample_submission_loaded") is False and receipt.get("kaggle_contacted") is False,
        "bundle_sha256": receipt.get("application_bundle_sha256") == _sha256_file(npz_path),
        "rows_match": receipt.get("rows") == len(ids),
    }
    csv_check = _verify_csv_against_arrays(csv_path, ids=ids, labels=labels,
                                           target_column=gamma.TARGET_COLUMN)
    checks["csv_sha256"] = receipt.get("csv_sha256") == csv_check["csv_sha256"]
    if not all(checks.values()):
        raise VerificationError(f"Synthetic application checks failed: {checks}")
    return {"status": "PASS_E79_SYNTHETIC_APPLICATION_VERIFICATION", "checks": checks,
            "receipt_sha256": digest, "csv": csv_check,
            "missing_stream_rows": int(np.count_nonzero(~present)),
            "test_sample_or_kaggle_access": False}


def verify_synthetic_e2e(output_dir: Path) -> dict[str, Any]:
    fit_result = verify_synthetic_fit(output_dir)
    app_result = verify_synthetic_application(output_dir)
    return {"status": "PASS_E79_SYNTHETIC_RESUME_VALIDATED_NO_REFIT",
            "fit": fit_result, "application": app_result,
            "attempt_count": 1, "additional_fit_count": 0,
            "test_sample_or_kaggle_access": False}


def verify_source_freeze(root: Path = ROOT) -> dict[str, Any]:
    result = gamma.verify_source_freeze(root)
    return {"status": result["status"], "freeze_sha256": result["freeze_sha256"],
            "source_commit": result["source_commit"],
            "freeze_file_sha256": result["freeze_file_sha256"],
            "dfmap_verified": result["dfmap_verified"],
            "official_gamma_fit_started": result["official_gamma_fit_started"]}


def verify_fit(root: Path = ROOT) -> dict[str, Any]:
    freeze_check = gamma.verify_source_freeze(root)
    freeze = _json(root / gamma.FREEZE_REL)
    source_pass, source_pass_sha = gamma._verify_committed_pass(root, gamma.SOURCE_PASS_REL, "source", freeze)
    attempt_path, artifact_path, receipt_path = gamma._fit_paths(root)
    if not all(path.is_file() for path in (attempt_path, artifact_path, receipt_path)):
        raise VerificationError("Official fit lacks its attempt marker, gamma artifact, or receipt")
    attempt = _json(attempt_path)
    receipt, digest = _verify_receipt(receipt_path)
    with np.load(artifact_path, allow_pickle=False) as archive:
        if set(archive.files) != {"gamma_float64"}:
            raise VerificationError("Official gamma artifact has unexpected members")
        gamma_vector = np.asarray(archive["gamma_float64"]).copy()
    checks = {
        "fit_id": receipt.get("fit_id") == gamma.recipe.FIT_ID and attempt.get("fit_id") == gamma.recipe.FIT_ID,
        "attempt_started_once": attempt.get("status") == "ONE_E79_GAMMA_FIT_ATTEMPT_STARTED" and attempt.get("one_fit_only_no_retry") is True,
        "receipt_pass": receipt.get("status") == "PASS_E79_GAMMA_FIT_PERSISTED" and receipt.get("mode") == "official",
        "source_freeze_bound": receipt.get("source_freeze_sha256") == freeze_check["freeze_sha256"],
        "source_freeze_file_bound": receipt.get("source_freeze_file_sha256") == freeze_check["freeze_file_sha256"],
        "source_pass_bound": receipt.get("root_source_pass_file_sha256") == source_pass_sha and source_pass.get("one_full_train_gamma_fit_authorized") is True,
        "invocation_commit_valid": bool(receipt.get("source_commit")) and subprocess.run(["git", "-C", str(root), "merge-base", "--is-ancestor", receipt.get("source_commit", ""), gamma._git(root, "rev-parse", "HEAD")], check=False).returncode == 0,
        "frozen_source_commit_bound": receipt.get("frozen_source_commit") == freeze_check["source_commit"],
        "train_population": receipt.get("population", {}).get("rows") == gamma.TRAIN_ROWS and receipt.get("population", {}).get("missing_stream_rows_included") is True,
        "matrix_and_id_hashes_bound": (
            receipt.get("input_array_sha256", {}).get("official_train_file_sha256") == gamma.TRAIN_SHA256
            and receipt.get("input_array_sha256", {}).get("ordered_train_ids_sha256") == gamma.ORDERED_TRAIN_IDS_SHA256
            and receipt.get("input_array_sha256", {}).get("base_matrix_sha256") == gamma.BASE_SHA256
            and receipt.get("input_array_sha256", {}).get("position_matrix_sha256") == gamma.POSITION_SHA256
        ),
        "target_hash_present_without_reload": isinstance(receipt.get("input_array_sha256", {}).get("target_artifact_signal_sha256"), str) and len(receipt["input_array_sha256"]["target_artifact_signal_sha256"]) == 64,
        "gamma_schema": gamma_vector.dtype == np.dtype(np.float64) and list(gamma_vector.shape) == receipt.get("gamma_shape"),
        "gamma_finite_digest": bool(np.isfinite(gamma_vector).all()) and positional.sha256_array(gamma_vector) == receipt.get("gamma_sha256"),
        "artifact_file_sha256": _sha256_file(artifact_path) == receipt.get("artifact_sha256"),
        "optimizer_success": receipt.get("optimizer", {}).get("success") is True,
        "resource_reserve": receipt.get("postfit_reserve_pass") is True and receipt.get("resource_after", {}).get("available_ram_bytes", 0) >= gamma.FREE_RESERVE_BYTES,
        "no_test_or_kaggle": receipt.get("test_or_sample_read") is False and receipt.get("test_inference_performed") is False and receipt.get("csv_created") is False and receipt.get("kaggle_contacted") is False,
        "target_loaded_after_feature_checks": receipt.get("target_values_read_only_after_feature_parity") is True and receipt.get("target_column_requested_or_loaded") == gamma.TARGET_COLUMN,
        "fit_count_one": receipt.get("fit_count") == 1,
    }
    if not all(checks.values()):
        raise VerificationError(f"Official gamma fit verification failed: {checks}")
    return {"status": "PASS_E79_OFFICIAL_GAMMA_FIT_READBACK", "checks": checks,
            "receipt_sha256": digest, "receipt_file_sha256": _sha256_file(receipt_path),
            "artifact_sha256": _sha256_file(artifact_path), "gamma_shape": list(gamma_vector.shape),
            "fit_count": 1, "target_values_reloaded": False,
            "test_sample_or_kaggle_access": False}


def verify_application(root: Path = ROOT) -> dict[str, Any]:
    fit_result = verify_fit(root)
    freeze = _json(root / gamma.FREEZE_REL)
    fit_receipt_path = root / gamma.FIT_RECEIPT_REL
    app_pass, app_pass_sha = gamma._read_application_pass(root, freeze, fit_receipt_path)
    attempt_path, npz_path, receipt_path, csv_path = gamma._application_paths(root)
    if not all(path.is_file() for path in (attempt_path, npz_path, receipt_path, csv_path)):
        raise VerificationError("Official application lacks attempt, NPZ, receipt, or CSV")
    attempt = _json(attempt_path)
    receipt, digest = _verify_receipt(receipt_path)
    with np.load(npz_path, allow_pickle=False) as archive:
        required = {"record_ids", "present_stream", "a02_probability", "probability", "labels"}
        if set(archive.files) != required:
            raise VerificationError("Official application bundle schema differs")
        arrays = {key: np.asarray(archive[key]).copy() for key in archive.files}
    ids = arrays["record_ids"].astype(str).reshape(-1)
    present = arrays["present_stream"].astype(bool).reshape(-1)
    p_a02 = arrays["a02_probability"].astype(np.float64).reshape(-1)
    probability = arrays["probability"].astype(np.float64).reshape(-1)
    labels = arrays["labels"].astype(np.int8).reshape(-1)
    expected_thresholds = np.where(present, gamma.recipe.PRESENT_THRESHOLD, gamma.recipe.MISSING_THRESHOLD)
    checks = {
        "attempt_bound": attempt.get("status") == "ONE_E79_APPLICATION_ATTEMPT_STARTED" and attempt.get("one_application_only_no_retry") is True,
        "receipt_status": receipt.get("status") == "PASS_E79_OFFICIAL_TEST_APPLICATION_CSV_CREATED" and receipt.get("mode") == "official",
        "application_pass_bound": receipt.get("application_root_pass_file_sha256") == app_pass_sha and app_pass.get("test_inference_authorized") is True,
        "gamma_fit_bound": receipt.get("gamma_fit_receipt_file_sha256") == fit_result["receipt_file_sha256"],
        "test_rows": receipt.get("test_rows") == 29_421 and len(ids) == 29_421,
        "ordered_ids_unique": len(set(ids.tolist())) == len(ids) and positional.sha256_array(ids) == receipt.get("ordered_test_ids_sha256"),
        "array_dtypes": bool(arrays["record_ids"].dtype.kind in "US"
                         and arrays["present_stream"].dtype == np.dtype(bool)
                         and arrays["a02_probability"].dtype == np.dtype(np.float64)
                         and arrays["probability"].dtype == np.dtype(np.float64)
                         and arrays["labels"].dtype.kind in "iu"
                         and np.isin(arrays["labels"], [0, 1]).all()),
        "probability_schema": len(present) == len(ids) == len(p_a02) == len(probability) == len(labels),
        "probability_range": bool(np.isfinite(p_a02).all() and np.isfinite(probability).all() and np.all((p_a02 >= 0) & (p_a02 <= 1)) and np.all((probability >= 0) & (probability <= 1))),
        "missing_fallback_bit_exact": bool(np.array_equal(probability[~present], p_a02[~present])) and receipt.get("missing_stream_a02_fallback_bit_exact") is True,
        "labels_match_frozen_thresholds": bool(np.array_equal(labels, (probability >= expected_thresholds).astype(np.int8))),
        "prediction_bundle_hash": receipt.get("application_bundle_sha256") == _sha256_file(npz_path),
        "no_fit_test_target_sample_or_kaggle": receipt.get("fit_count_during_application") == 0 and receipt.get("test_target_requested_or_loaded") is False and receipt.get("sample_submission_loaded") is False and receipt.get("kaggle_contacted") is False,
    }
    csv_result = _verify_csv_against_arrays(csv_path, ids=ids, labels=labels,
                                           target_column=gamma.TARGET_COLUMN)
    checks["csv_hash"] = receipt.get("csv_sha256") == csv_result["csv_sha256"]
    if not all(checks.values()):
        raise VerificationError(f"Official E79 application verification failed: {checks}")
    return {"status": "PASS_E79_OFFICIAL_TEST_APPLICATION_VERIFICATION", "checks": checks,
            "receipt_sha256": digest, "receipt_file_sha256": _sha256_file(receipt_path),
            "csv": csv_result, "rows": len(ids), "test_targets_reloaded": False,
            "sample_submission_or_kaggle_access": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--verify-freeze", action="store_true")
    group.add_argument("--verify-fit", action="store_true")
    group.add_argument("--verify-application", action="store_true")
    group.add_argument("--verify-synthetic-e2e", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.verify_freeze:
            result = verify_source_freeze()
        elif args.verify_fit:
            result = verify_fit()
        elif args.verify_application:
            result = verify_application()
        else:
            if args.output_dir is None:
                raise VerificationError("--verify-synthetic-e2e requires the synthetic temporary --output-dir")
            result = verify_synthetic_e2e(args.output_dir.resolve())
    except Exception as exc:
        print(json.dumps({"status": "FAIL_INDEPENDENT_E79_GAMMA_VERIFICATION",
                          "error_type": type(exc).__name__, "error": str(exc)},
                         sort_keys=True, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2))
    return 0 if str(result.get("status", "")).startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
