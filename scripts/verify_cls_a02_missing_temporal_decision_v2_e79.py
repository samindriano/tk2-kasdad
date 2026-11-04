"""Independent QA for the frozen A02 missing-stream E79 candidate.

This verifier does not import the candidate builder, read target labels, access
SampleSubmission, fit models, or contact Kaggle. It independently replays the
frozen probability route, CSV contract, application hashes, and duplicate scan.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/optimization_v3/cls_a02_missing_temporal_decision_v2"
PARENT_FREEZE = REPORT / "E79_APPLICATION_FREEZE.json"
SUCCESSOR_FREEZE = REPORT / "E79_APPLICATION_FREEZE_V2.json"
GUARD_QA = REPORT / "E79_APPLICATION_FREEZE_V2_GUARD_QA.json"
MANIFEST_PATH = REPORT / "E79_MANIFEST.json"
BUILDER_QA_PATH = REPORT / "E79_QA.json"
OUTPUT_CSV = ROOT / "submissions/optimization_v3/cls_a02_missing_temporal_decision_v2/classification_a02_missing_temporal_e79.csv"
TEST_CSV = ROOT / "data/raw/classification/Test.csv"
E79_NPZ = ROOT / "experiments/runs/optimization_v3_a02_cgeom_benchmark_v1/test_predictions.npz"
CURRENT_BEST = ROOT / "submissions/optimization_v3/cls_mix50_a02_missing_e79_v1/classification_mix50_present_a02_missing_e79.csv"
RECEIPT = REPORT / "E79_INDEPENDENT_QA.json"
ORIGINAL_FREEZE_SHA256 = "b833b5d078a233ae17dcb7d14fecd451cdeac3cd8f63d0b94e9db7a8838cd493"
EXPECTED_INPUTS = {
    "experiments/runs/optimization_v3_a02_cgeom_benchmark_v1/test_predictions.npz",
    "data/raw/classification/Test.csv",
    "submissions/optimization_v3/cls_mix50_a02_missing_e79_v1/classification_mix50_present_a02_missing_e79.csv",
    "reports/optimization_v3/cls_mix50_a02_missing_e79_v1/MANIFEST.json",
    "reports/optimization_v3/cls_mix50_a02_missing_e79_v1/QA.json",
    "data/raw/classification/Train.csv",
    "experiments/runs/optimization_v3_posoffset_a02_endgame_v1/policy_predictions.npz",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_signed_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    signature = value.pop("canonical_sha256", None)
    if signature != canonical_sha256(value):
        raise RuntimeError(f"Canonical signature mismatch: {path}")
    value["canonical_sha256"] = signature
    return value


def read_submission(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["record_id", "artifact_signal"]:
            raise RuntimeError(f"Unexpected submission schema: {path}")
        rows = list(reader)
    ids = [row["record_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"Duplicate record_id in submission: {path}")
    try:
        labels = np.asarray([int(row["artifact_signal"]) for row in rows], dtype=np.int8)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Non-integer label in submission: {path}") from exc
    if not np.isin(labels, [0, 1]).all():
        raise RuntimeError(f"Non-binary label in submission: {path}")
    return ids, labels


def read_test_keys() -> tuple[list[str], np.ndarray]:
    with TEST_CSV.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        columns = {name: index for index, name in enumerate(header)}
        if "record_id" not in columns or "signal_stream" not in columns:
            raise RuntimeError("Official Test lacks record_id or signal_stream")
        ids: list[str] = []
        missing: list[bool] = []
        max_index = max(columns["record_id"], columns["signal_stream"])
        for row in reader:
            if len(row) <= max_index:
                raise RuntimeError("Malformed official Test row")
            ids.append(row[columns["record_id"]])
            missing.append(row[columns["signal_stream"]].strip() == "")
    if len(ids) != 29421 or len(ids) != len(set(ids)):
        raise RuntimeError("Official Test IDs are not 29,421 unique rows")
    return ids, np.asarray(missing, dtype=bool)


def independent_duplicate_scan(
    official_ids: list[str], candidate_labels: np.ndarray,
) -> dict[str, Any]:
    scanned = 0
    ignored_nonclassification = 0
    ignored_incomplete_or_other_ids = 0
    matches: list[str] = []
    vector_hashes: list[dict[str, str]] = []
    candidate_resolved = OUTPUT_CSV.resolve()
    for path in sorted((ROOT / "submissions").rglob("*.csv")):
        if path.resolve() == candidate_resolved:
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            try:
                header = next(reader)
            except StopIteration:
                ignored_nonclassification += 1
                continue
            if header != ["record_id", "artifact_signal"]:
                ignored_nonclassification += 1
                continue
            rows = list(reader)
        if len(rows) != len(official_ids):
            ignored_incomplete_or_other_ids += 1
            continue
        try:
            ids = [row[0] for row in rows]
            if any(len(row) != 2 for row in rows):
                raise ValueError("bad schema")
            labels = np.asarray([int(row[1]) for row in rows], dtype=np.int8)
        except (IndexError, ValueError):
            ignored_incomplete_or_other_ids += 1
            continue
        if ids != official_ids or len(ids) != len(set(ids)) or not np.isin(labels, [0, 1]).all():
            ignored_incomplete_or_other_ids += 1
            continue
        scanned += 1
        vector_bytes = labels.tobytes(order="C")
        vector_hashes.append({
            "path": path.relative_to(ROOT).as_posix(),
            "label_vector_sha256": hashlib.sha256(vector_bytes).hexdigest(),
        })
        if np.array_equal(labels, candidate_labels):
            matches.append(path.relative_to(ROOT).as_posix())
    return {
        "full_test_classification_vectors_scanned": scanned,
        "ignored_nonclassification_csvs": ignored_nonclassification,
        "ignored_incomplete_or_other_id_csvs": ignored_incomplete_or_other_ids,
        "exact_duplicate_matches": matches,
        "duplicate_found": bool(matches),
        "scanned_label_vector_hashes": vector_hashes,
    }


def verify() -> dict[str, Any]:
    original_sha = sha256_file(PARENT_FREEZE)
    if original_sha != ORIGINAL_FREEZE_SHA256:
        raise RuntimeError("Original five-input freeze bytes changed")
    parent = read_signed_json(PARENT_FREEZE)
    successor = read_signed_json(SUCCESSOR_FREEZE)
    guard = read_signed_json(GUARD_QA)
    manifest = read_signed_json(MANIFEST_PATH)
    builder_qa = read_signed_json(BUILDER_QA_PATH)

    if successor.get("parent_application_freeze_sha256") != original_sha:
        raise RuntimeError("Successor parent freeze hash does not match original")
    if guard.get("status") != "PASS_VERSIONED_FREEZE_GUARD":
        raise RuntimeError("Committed pre-access guard is not PASS")
    if guard.get("E79_probability_arrays_opened") is not False or guard.get("test_rows_or_fields_parsed") is not False:
        raise RuntimeError("Pre-access guard reports early E79/Test access")
    frozen_inputs = successor.get("application_input_hashes", {})
    if set(frozen_inputs) != EXPECTED_INPUTS:
        raise RuntimeError("Successor freeze does not bind exact seven application inputs")
    actual_hashes = {rel: sha256_file(ROOT / rel) for rel in sorted(EXPECTED_INPUTS)}
    if actual_hashes != frozen_inputs or manifest.get("input_hashes") != frozen_inputs:
        raise RuntimeError("Current inputs differ from successor freeze or output manifest")
    if manifest.get("application_freeze_sha256") != sha256_file(SUCCESSOR_FREEZE):
        raise RuntimeError("Manifest application freeze hash mismatch")
    if manifest.get("final_selector_receipt_sha256") != successor.get("final_selector_receipt_sha256"):
        raise RuntimeError("Manifest final selector receipt mismatch")
    threshold = float(successor["final_selector_threshold"])
    if not math.isclose(threshold, 0.53230976129477, rel_tol=0.0, abs_tol=1e-15):
        raise RuntimeError("Frozen final threshold changed")
    if not math.isclose(float(manifest["selected_missing_threshold"]), threshold, rel_tol=0.0, abs_tol=0.0):
        raise RuntimeError("Manifest threshold differs from frozen threshold")
    if manifest.get("upload_performed") is not False or manifest.get("kaggle_contacted") is not False:
        raise RuntimeError("Candidate manifest claims Kaggle activity")
    if builder_qa.get("status") != "PASS_LOCAL_E79_QA" or builder_qa.get("manifest_sha256") != sha256_file(MANIFEST_PATH):
        raise RuntimeError("Builder QA status or manifest hash mismatch")

    test_ids, test_missing = read_test_keys()
    bundle_ids: list[str]
    with np.load(E79_NPZ, allow_pickle=False) as archive:
        required = {"record_ids", "missing_mask", "posoffset_probability", "a02_probability"}
        if not required.issubset(archive.files):
            raise RuntimeError("E79 probability bundle lacks expected arrays")
        bundle_ids = np.asarray(archive["record_ids"], dtype="U").tolist()
        bundle_missing = np.asarray(archive["missing_mask"], dtype=bool)
        p_pos = np.asarray(archive["posoffset_probability"], dtype=np.float64)
        p_a02 = np.asarray(archive["a02_probability"], dtype=np.float64)
    if bundle_ids != test_ids or bundle_missing.shape != (29421,) or not np.array_equal(bundle_missing, test_missing):
        raise RuntimeError("Probability bundle keys or missing mask differ from official Test")
    if p_pos.shape != (29421,) or p_a02.shape != (29421,):
        raise RuntimeError("Probability vector dimensions differ from official Test")
    if not np.isfinite(p_pos).all() or not np.isfinite(p_a02).all() or not ((0 <= p_pos).all() and (p_pos <= 1).all() and (0 <= p_a02).all() and (p_a02 <= 1).all()):
        raise RuntimeError("E79 probability vector contains non-finite or out-of-range values")

    current_ids, current_labels = read_submission(CURRENT_BEST)
    candidate_ids, candidate_labels = read_submission(OUTPUT_CSV)
    if current_ids != test_ids or candidate_ids != test_ids or len(candidate_labels) != 29421:
        raise RuntimeError("Current-best/candidate CSV IDs or order differ from official Test")
    current_expected = np.where(test_missing, p_a02 >= 0.50, 0.5 * (p_pos + p_a02) >= 0.50).astype(np.int8)
    expected = np.where(test_missing, p_a02 >= threshold, 0.5 * (p_pos + p_a02) >= 0.50).astype(np.int8)
    if not np.array_equal(current_labels, current_expected):
        raise RuntimeError("Current-best CSV does not replay from frozen probabilities")
    if not np.array_equal(candidate_labels, expected):
        raise RuntimeError("Candidate CSV does not replay from frozen final policy")
    if not np.array_equal(candidate_labels[~test_missing], current_labels[~test_missing]):
        raise RuntimeError("Candidate changed present-stream labels")
    if int(candidate_labels.sum()) != int(manifest["candidate_positive_count"]):
        raise RuntimeError("Candidate positive count differs from manifest")

    duplicate_scan = independent_duplicate_scan(test_ids, candidate_labels)
    if duplicate_scan["duplicate_found"]:
        raise RuntimeError("Independent full-vector scan found an exact duplicate")
    if duplicate_scan["full_test_classification_vectors_scanned"] != 22:
        raise RuntimeError("Independent duplicate scan did not cover expected 22 full vectors")
    if duplicate_scan["ignored_nonclassification_csvs"] != 14 or duplicate_scan["ignored_incomplete_or_other_id_csvs"] != 0:
        raise RuntimeError("Independent duplicate scan file accounting differs from expected inventory")

    checks = {
        "original_freeze_byte_identical": True,
        "versioned_successor_exactly_seven_inputs": True,
        "committed_pre_access_guard_pass": True,
        "all_seven_application_input_hashes_match": True,
        "candidate_rows_unique_ids_and_official_test_order": True,
        "candidate_schema_and_integer_binary_labels": True,
        "frozen_probability_formula_replayed_independently": True,
        "present_labels_bit_exact_to_current_best": True,
        "missing_only_threshold_application_replayed": True,
        "protected_test_fields_only_record_id_and_signal_stream": True,
        "test_targets_and_sample_submission_not_accessed": True,
        "duplicate_full_vector_scan_completed_no_match": True,
        "zero_new_fits_preprocessing_or_model_inference": True,
        "no_kaggle_contact_or_upload": True,
    }
    return {
        "schema_version": 1,
        "status": "PASS_INDEPENDENT_E79_QA",
        "policy_id": manifest["policy_id"],
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "verifier_sha256": sha256_file(Path(__file__)),
        "original_freeze_sha256": original_sha,
        "successor_freeze_sha256": sha256_file(SUCCESSOR_FREEZE),
        "guard_qa_sha256": sha256_file(GUARD_QA),
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "builder_qa_sha256": sha256_file(BUILDER_QA_PATH),
        "application_input_hashes": actual_hashes,
        "csv_path": OUTPUT_CSV.relative_to(ROOT).as_posix(),
        "csv_sha256": sha256_file(OUTPUT_CSV),
        "rows": int(candidate_labels.size),
        "unique_ids": int(len(set(candidate_ids))),
        "columns": ["record_id", "artifact_signal"],
        "selected_missing_threshold": threshold,
        "candidate_positive_count": int(candidate_labels.sum()),
        "missing_rows": int(test_missing.sum()),
        "changed_missing_labels_vs_current_best": int(np.count_nonzero(candidate_labels[test_missing] != current_labels[test_missing])),
        "present_labels_unchanged": bool(np.array_equal(candidate_labels[~test_missing], current_labels[~test_missing])),
        "duplicate_scan": duplicate_scan,
        "checks": checks,
        "test_label_fields_read": [],
        "sample_submission_read": False,
        "estimator_fits": 0,
        "preprocessing_fits": 0,
        "model_inference": 0,
        "kaggle_contacted": False,
        "upload_performed": False,
    }


def main() -> None:
    body = verify()
    signed = {**body, "canonical_sha256": canonical_sha256(body)}
    payload = (json.dumps(signed, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    with RECEIPT.open("xb") as stream:
        stream.write(payload)
        stream.flush()
    print(json.dumps({"receipt_path": RECEIPT.relative_to(ROOT).as_posix(), "receipt_sha256": hashlib.sha256(payload).hexdigest(), **body}, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
