"""Independent readback checks for the one-shot E79 application recovery."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts import cls_pos_poisson_bag2_e79_v1 as frozen
from scripts import verify_cls_pos_poisson_bag2_e79_v1 as frozen_verifier


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _receipt(value: Mapping[str, Any], label: str) -> None:
    body = dict(value)
    recorded = body.pop("receipt_sha256", None)
    require(recorded == frozen.H.canonical_sha256(body), f"Self-hash mismatch: {label}")


def _csv(path: Path) -> tuple[list[str], list[int]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        require(header == ["record_id", "artifact_signal"], f"CSV schema mismatch: {path}")
        rows = list(reader)
    require(all(len(row) == 2 for row in rows), f"CSV row width mismatch: {path}")
    ids = [row[0] for row in rows]
    raw = [row[1] for row in rows]
    require(ids and all(ids) and len(ids) == len(set(ids)), f"CSV IDs missing/duplicated: {path}")
    require(all(value in {"0", "1"} for value in raw), f"CSV labels are not integer 0/1: {path}")
    return ids, [int(value) for value in raw]


def verify_recovery_application(
    *,
    recovery_root: Path,
    fit_root: Path,
    receipt_path: Path,
    bundle_path: Path,
    csv_path: Path,
    baseline_csv_path: Path,
    expected_baseline_sha256: str,
    expected_fit_set: Mapping[str, Any],
    expected_missing_rows: int | None,
    authorization_file_sha256: str,
    expected_test_ids: Sequence[str] | None = None,
    expected_test_sha256: str | None = None,
    expected_csv_relative_path: str | None = None,
) -> dict[str, Any]:
    recovery_root = Path(recovery_root)
    receipt_path, bundle_path, csv_path = map(Path, (receipt_path, bundle_path, csv_path))
    attempt = _json(recovery_root / "RECOVERY_ATTEMPT.json")
    _receipt(attempt, "recovery attempt")
    require(attempt.get("status") == "ONE_RECOVERY_APPLICATION_ATTEMPT_STARTED"
            and attempt.get("authorized_fit_count") == 0
            and attempt.get("application_attempt_cap") == 1
            and attempt.get("retry_allowed") is False
            and attempt.get("test_targets_allowed") is False
            and attempt.get("sample_submission_allowed") is False
            and attempt.get("kaggle_allowed") is False
            and attempt.get("authorization_file_sha256") == authorization_file_sha256,
            "Recovery attempt identity, cap, or boundary mismatch")

    application_attempt = _json(recovery_root / "APPLICATION_ATTEMPT.json")
    require(application_attempt.get("status") == "ONE_E79_TEST_APPLICATION_ATTEMPT_STARTED"
            and application_attempt.get("test_target_requested_or_loaded") is False
            and application_attempt.get("sample_submission_loaded") is False
            and application_attempt.get("kaggle_contacted") is False
            and application_attempt.get("one_application_no_retry") is True,
            "Frozen application attempt identity or boundary mismatch")

    receipt = _json(receipt_path)
    _receipt(receipt, "E79 application")
    require(receipt.get("candidate_id") == frozen.CANDIDATE_ID
            and receipt.get("status") == "PASS_E79_APPLICATION_CSV_CREATED"
            and receipt.get("mode") == "official"
            and receipt.get("fit_count_during_application") == 0
            and receipt.get("inference_count") == 1
            and receipt.get("test_targets_requested_or_loaded") is False
            and receipt.get("sample_submission_opened") is False
            and receipt.get("kaggle_contacted") is False,
            "Frozen application receipt status or data boundary mismatch")
    require(receipt.get("baseline_csv_sha256") == expected_baseline_sha256
            and _file_sha256(baseline_csv_path) == expected_baseline_sha256
            and receipt.get("baseline_champion_decision_parity") is True,
            "Champion baseline hash/parity mismatch")
    require(receipt.get("application_bundle_sha256") == _file_sha256(bundle_path)
            and receipt.get("candidate_csv_sha256") == _file_sha256(csv_path),
            "Application bundle or CSV hash mismatch")
    if expected_csv_relative_path is not None:
        require(receipt.get("candidate_csv_path") == expected_csv_relative_path,
                "Application receipt points at another CSV path")
    if expected_test_sha256 is not None:
        require(receipt.get("application_input_hashes", {}).get("official_test_sha256")
                == expected_test_sha256,
                "Application receipt is not bound to the frozen official Test bytes")

    # The frozen verifier checks probability construction, thresholds, CSV readback,
    # exact champion replay, fit receipt lineage, and all native application fields.
    frozen_qa = frozen_verifier.verify_application(
        fit_root, receipt_path=receipt_path, bundle_path=bundle_path,
        csv_path=csv_path, baseline_csv_path=baseline_csv_path,
        expected_baseline_sha256=expected_baseline_sha256)
    actual_fit_set = frozen_verifier.verify_fit_set(
        fit_root, freeze_signature=receipt["freeze_signature"])
    require(actual_fit_set == expected_fit_set.get("fit_set"),
            "Fit artifacts/receipts no longer match pre-application verified set")

    candidate_ids, candidate_labels = _csv(csv_path)
    baseline_ids, baseline_labels = _csv(baseline_csv_path)
    require(candidate_ids == baseline_ids,
            "Candidate CSV row order differs from baseline Test ID order")
    if expected_test_ids is not None:
        expected_ids = [str(value) for value in expected_test_ids]
        require(candidate_ids == expected_ids,
                "Candidate CSV does not match all ordered official Test IDs")
    with np.load(bundle_path, allow_pickle=False) as archive:
        present = np.asarray(archive["present_stream"], dtype=bool).reshape(-1)
        labels_champion = np.asarray(archive["labels_champion"], dtype=np.int8).reshape(-1)
        labels_candidate = np.asarray(archive["labels_candidate"], dtype=np.int8).reshape(-1)
    require(candidate_labels == labels_candidate.astype(int).tolist()
            and baseline_labels == labels_champion.astype(int).tolist(),
            "CSV labels differ from bundled candidate/champion decisions")
    missing_rows = int((~present).sum())
    if expected_missing_rows is not None:
        require(missing_rows == expected_missing_rows,
                f"Missing-stream count differs: {missing_rows} != {expected_missing_rows}")
    return {
        "status": "PASS_INDEPENDENT_E79_RECOVERY_READBACK",
        "rows": len(candidate_ids),
        "ordered_ids_sha256": frozen.H.sha256_array(np.asarray(candidate_ids, dtype=str)),
        "candidate_csv_sha256": _file_sha256(csv_path),
        "baseline_csv_sha256": _file_sha256(baseline_csv_path),
        "baseline_champion_decision_parity": True,
        "candidate_changed_decisions": int(np.count_nonzero(
            np.asarray(candidate_labels, dtype=np.int8) != np.asarray(baseline_labels, dtype=np.int8))),
        "missing_rows": missing_rows,
        "missing_fallback_exact": True,
        "fit_set_unchanged": True,
        "fit_set_status": actual_fit_set["status"],
        "test_targets_or_sample_opened": False,
        "frozen_verifier": frozen_qa,
    }
