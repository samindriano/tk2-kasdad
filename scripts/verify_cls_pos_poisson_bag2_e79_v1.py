"""Independent E79 fit and application readback checks.

This module deliberately does not import the E79 producer or open any official
data.  It validates persisted receipts, fitted gamma arrays, routed probabilities,
threshold decisions, and CSV row identity from the supplied artifacts.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


CANDIDATE_ID = "CLS_POS_POISSON_BAG2_E79_V1"
FIT_IDS = (f"{CANDIDATE_ID}_E79_P17", f"{CANDIDATE_ID}_E79_P137")
SEED_BY_FIT_ID = {FIT_IDS[0]: 17, FIT_IDS[1]: 137}
PRESENT_THRESHOLD = 0.50
MISSING_THRESHOLD = 0.53230976129477


def require(condition: bool, message: str) -> None:
    """A runtime gate that remains active under ``python -O``."""
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_array(value: Any) -> str:
    array = np.asarray(value)
    header = canonical_sha256({"shape": list(array.shape), "dtype": str(array.dtype)})
    digest = hashlib.sha256(header.encode("ascii"))
    if array.dtype.kind in "OUS":
        digest.update(json.dumps(array.astype(str).tolist(), ensure_ascii=False,
                                 separators=(",", ":")).encode("utf-8"))
    else:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def _verify_receipt_self_hash(value: dict[str, Any], label: str) -> None:
    body = dict(value)
    receipt_sha = body.pop("receipt_sha256", None)
    require(receipt_sha == canonical_sha256(body), f"Receipt self-hash mismatch: {label}")


def _candidate_root(path: Path) -> Path:
    path = Path(path)
    return path.parent if path.name == "fits" else path


def verify_fit_artifact(run_root: Path, fit_id: str, *,
                        freeze_signature: str) -> tuple[np.ndarray, dict[str, Any]]:
    require(fit_id in SEED_BY_FIT_ID, f"Unfrozen E79 fit ID: {fit_id}")
    candidate_root = _candidate_root(run_root)
    fit_dir = candidate_root / "fits" / fit_id
    attempt_path = fit_dir / "ATTEMPT.json"
    artifact_path = fit_dir / "gamma.npz"
    receipt_path = fit_dir / "FIT_RECEIPT.json"
    require(all(path.is_file() for path in (attempt_path, artifact_path, receipt_path)),
            f"Fit artifact/receipt/attempt missing: {fit_id}")
    attempt, receipt = _json(attempt_path), _json(receipt_path)
    _verify_receipt_self_hash(receipt, fit_id)
    seed = SEED_BY_FIT_ID[fit_id]
    require(receipt.get("candidate_id") == CANDIDATE_ID
            and receipt.get("fit_id") == fit_id
            and receipt.get("seed") == seed
            and receipt.get("freeze_signature") == freeze_signature
            and receipt.get("status") == "PASS_E79_POISSON_GAMMA_FIT_PERSISTED",
            f"Fit receipt identity/status/freeze mismatch: {fit_id}")
    require(receipt.get("mode") in {"official", "synthetic"}, f"Unknown fit mode: {fit_id}")
    require(receipt.get("retry_allowed") is False and receipt.get("fit_count") == 1
            and receipt.get("preprocessing_fit_count", 0) == 0
            and receipt.get("original_head_fit_count", 0) == 0
            and receipt.get("test_or_sample_access") is False,
            f"Fit boundary, retry, or fit-count metadata mismatch: {fit_id}")
    require(attempt.get("fit_id") == fit_id and attempt.get("seed") == seed,
            f"Fit attempt identity mismatch: {fit_id}")
    optimizer = receipt.get("optimizer", {})
    require(optimizer.get("optimizer") == "scipy.optimize.minimize:L-BFGS-B"
            and optimizer.get("success") is True
            and optimizer.get("ridge_c") == 1.0
            and optimizer.get("options") == {"maxiter": 2000, "ftol": 1e-12,
                                                "gtol": 1e-6, "maxls": 50}
            and optimizer.get("row_weights_sum") == receipt.get("normalized_weights_sum")
            and np.isclose(float(receipt.get("normalized_weights_sum", float("nan"))),
                           float(receipt["fit_population"]["rows"]), rtol=0.0,
                           atol=1e-10 * int(receipt["fit_population"]["rows"])),
            f"Fit optimizer/normalization differs from fixed recipe: {fit_id}")
    require(receipt.get("poisson_lambda") == 1.0 and receipt.get("ridge_c") == 1.0
            and receipt.get("global_strength_penalty_used") is False
            and receipt.get("normalized_weights_sha256")
            and receipt.get("raw_counts_sha256"),
            f"Poisson weighting or uniform penalty differs from frozen recipe: {fit_id}")
    if receipt["mode"] == "official":
        require(receipt["fit_population"] == {
            "rows": 50_608, "cycles": [47, 79], "source": "official Classification Train"},
            f"Official fit population mismatch: {fit_id}")
        require(receipt.get("gamma_shape") == [17_259], f"Official gamma width mismatch: {fit_id}")
    require(sha256_file(artifact_path) == receipt.get("artifact_sha256"),
            f"Fit artifact file hash mismatch: {fit_id}")
    with np.load(artifact_path, allow_pickle=False) as archive:
        require(set(archive.files) == {"gamma"}, f"Unexpected gamma artifact members: {fit_id}")
        gamma = np.asarray(archive["gamma"], dtype=np.float64).reshape(-1).copy()
    require(gamma.ndim == 1 and np.isfinite(gamma).all()
            and gamma.shape == tuple(receipt.get("gamma_shape", ()))
            and str(gamma.dtype) == receipt.get("gamma_dtype")
            and sha256_array(gamma) == receipt.get("gamma_sha256"),
            f"Gamma schema/logical hash mismatch: {fit_id}")
    return gamma, receipt


def verify_fit_set(run_root: Path, *, freeze_signature: str) -> dict[str, Any]:
    root = _candidate_root(run_root)
    fits = {}
    for fit_id in FIT_IDS:
        gamma, receipt = verify_fit_artifact(root, fit_id, freeze_signature=freeze_signature)
        fits[fit_id] = {"gamma_sha256": sha256_array(gamma),
                        "artifact_sha256": receipt["artifact_sha256"],
                        "receipt_file_sha256": sha256_file(root / "fits" / fit_id / "FIT_RECEIPT.json")}
    ledger_path = root / "fit_ledger.jsonl"
    if ledger_path.is_file():
        events = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line]
        starts = [event for event in events if event.get("event") == "FIT_STARTED"]
        saves = [event for event in events if event.get("event") == "FIT_SAVED"]
        failed = [event for event in events if event.get("event") == "FIT_FAILED"]
        require(len(starts) == len(saves) == 2 and not failed
                and {event.get("fit_id") for event in starts} == set(FIT_IDS)
                and {event.get("fit_id") for event in saves} == set(FIT_IDS),
                "Official fit ledger does not show exactly two resolved one-shot fits")
    return {"status": "PASS_TWO_E79_FITS_INDEPENDENTLY_VERIFIED",
            "fit_ids": list(FIT_IDS), "fit_cap": 2, "fit_receipts": fits,
            "no_retry": True, "test_or_sample_access": False}


def _read_csv(path: Path) -> tuple[list[str], list[str], np.ndarray]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        require(header == ["record_id", "artifact_signal"],
                f"Submission CSV columns/order mismatch: {path}")
        rows = list(reader)
    require(all(len(row) == 2 for row in rows), f"Submission CSV row width mismatch: {path}")
    ids = [row[0] for row in rows]
    labels_raw = [row[1] for row in rows]
    require(ids and all(ids) and len(ids) == len(set(ids)), f"CSV IDs empty/duplicated: {path}")
    require(all(value in {"0", "1"} for value in labels_raw), f"CSV targets not integer 0/1: {path}")
    return header, ids, np.asarray([int(value) for value in labels_raw], dtype=np.int8)


def verify_application(run_root: Path, *, receipt_path: Path, bundle_path: Path,
                       csv_path: Path, baseline_csv_path: Path,
                       expected_baseline_sha256: str) -> dict[str, Any]:
    root = _candidate_root(run_root)
    receipt = _json(receipt_path)
    _verify_receipt_self_hash(receipt, "E79 application")
    require(receipt.get("candidate_id") == CANDIDATE_ID
            and receipt.get("status") == "PASS_E79_APPLICATION_CSV_CREATED"
            and receipt.get("fit_count_during_application") == 0
            and receipt.get("inference_count") == 1
            and receipt.get("test_targets_requested_or_loaded") is False
            and receipt.get("sample_submission_opened") is False
            and receipt.get("kaggle_contacted") is False,
            "Application receipt status or boundary flags mismatch")
    require(receipt.get("baseline_champion_decision_parity") is True
            and receipt.get("baseline_csv_sha256") == expected_baseline_sha256
            and sha256_file(baseline_csv_path) == expected_baseline_sha256,
            "Frozen champion baseline identity/parity receipt mismatch")
    require(sha256_file(bundle_path) == receipt.get("application_bundle_sha256")
            and sha256_file(csv_path) == receipt.get("candidate_csv_sha256"),
            "Application bundle or CSV file hash mismatch")
    with np.load(bundle_path, allow_pickle=False) as archive:
        expected_keys = {"record_ids", "present_stream", "p_a02", "p_pos_original",
                         "p_pos_seed17", "p_pos_seed137", "probability_champion",
                         "probability_candidate", "labels_champion", "labels_candidate"}
        require(set(archive.files) == expected_keys, "Application bundle schema drift")
        values = {key: archive[key].copy() for key in expected_keys}
    ids = np.asarray(values["record_ids"]).astype(str).reshape(-1)
    present = np.asarray(values["present_stream"], dtype=bool).reshape(-1)
    n_rows = len(ids)
    require(n_rows > 0 and len(set(ids.tolist())) == n_rows
            and sha256_array(ids) == receipt.get("ordered_test_ids_sha256"),
            "Application row IDs empty, duplicated, or receipt-mismatched")
    if receipt.get("mode") == "official":
        require(n_rows == 29_421 and receipt.get("official_test_rows") == 29_421,
                "Official E79 Test row count mismatch")
    require(len(present) == n_rows, "Application stream-route length mismatch")
    p_a02 = np.asarray(values["p_a02"], dtype=np.float64).reshape(-1)
    p_original = np.asarray(values["p_pos_original"], dtype=np.float64).reshape(-1)
    p17 = np.asarray(values["p_pos_seed17"], dtype=np.float64).reshape(-1)
    p137 = np.asarray(values["p_pos_seed137"], dtype=np.float64).reshape(-1)
    p_champion = np.asarray(values["probability_champion"], dtype=np.float64).reshape(-1)
    p_candidate = np.asarray(values["probability_candidate"], dtype=np.float64).reshape(-1)
    for name, probability in (("p_a02", p_a02), ("p_original", p_original), ("p17", p17),
                             ("p137", p137), ("p_champion", p_champion),
                             ("p_candidate", p_candidate)):
        require(len(probability) == n_rows and np.isfinite(probability).all()
                and np.all((probability >= 0.0) & (probability <= 1.0)),
                f"Application probability invalid: {name}")
    p_bag = (p_original + p17 + p137) / 3.0
    expected_champion = p_a02.copy()
    expected_candidate = p_a02.copy()
    expected_champion[present] = 0.5 * p_original[present] + 0.5 * p_a02[present]
    expected_candidate[present] = 0.5 * p_bag[present] + 0.5 * p_a02[present]
    require(np.array_equal(p_champion.view(np.uint64), expected_champion.view(np.uint64))
            and np.array_equal(p_candidate.view(np.uint64), expected_candidate.view(np.uint64)),
            "Application probabilities differ from fixed E79 blend and fallback")
    require(np.array_equal(p_candidate[~present].view(np.uint64), p_a02[~present].view(np.uint64))
            and receipt.get("missing_stream_a02_fallback_bit_exact") is True,
            "Missing-stream fallback is not bit-exact A02 pA")
    thresholds = np.where(present, PRESENT_THRESHOLD, MISSING_THRESHOLD)
    expected_champion_labels = (p_champion >= thresholds).astype(np.int8)
    expected_candidate_labels = (p_candidate >= thresholds).astype(np.int8)
    labels_champion = np.asarray(values["labels_champion"], dtype=np.int8).reshape(-1)
    labels_candidate = np.asarray(values["labels_candidate"], dtype=np.int8).reshape(-1)
    require(np.array_equal(labels_champion, expected_champion_labels)
            and np.array_equal(labels_candidate, expected_candidate_labels)
            and receipt.get("present_threshold") == PRESENT_THRESHOLD
            and receipt.get("missing_threshold") == MISSING_THRESHOLD
            and receipt.get("threshold_operator") == ">=",
            "Application threshold decisions differ from frozen route policy")
    _, csv_ids, csv_labels = _read_csv(csv_path)
    _, baseline_ids, baseline_labels = _read_csv(baseline_csv_path)
    require(csv_ids == ids.tolist() and np.array_equal(csv_labels, labels_candidate),
            "Candidate CSV row order/IDs/labels differ from the application bundle")
    require(baseline_ids == ids.tolist() and np.array_equal(baseline_labels, labels_champion),
            "Original champion does not replay baseline CSV decisions on exact ordered IDs")
    require(receipt.get("candidate_csv_rows") == n_rows
            and receipt.get("candidate_csv_columns") == ["record_id", "artifact_signal"],
            "Application receipt CSV schema/row count mismatch")
    qa = verify_fit_set(root, freeze_signature=receipt["freeze_signature"])
    for fit_id in FIT_IDS:
        fit_receipt_path = root / "fits" / fit_id / "FIT_RECEIPT.json"
        expected_fit_sha = receipt.get("fit_receipt_file_sha256", {}).get(fit_id)
        require(expected_fit_sha == sha256_file(fit_receipt_path),
                f"Application receipt does not bind fit receipt: {fit_id}")
    return {"status": "PASS_INDEPENDENT_E79_APPLICATION_READBACK",
            "rows": n_rows, "ordered_ids_sha256": sha256_array(ids),
            "candidate_csv_sha256": sha256_file(csv_path),
            "baseline_csv_sha256": sha256_file(baseline_csv_path),
            "baseline_champion_decision_parity": True,
            "missing_rows": int((~present).sum()), "missing_fallback_exact": True,
            "fit_set": qa["status"], "test_targets_or_sample_opened": False}
