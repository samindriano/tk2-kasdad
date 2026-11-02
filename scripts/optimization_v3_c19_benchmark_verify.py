"""Independent verifier for the prepared C19 full-Train benchmark CSV."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.optimization_v3_c19_benchmark import (  # noqa: E402
    BASE_CACHE,
    BASE_CACHE_SHA256,
    BASE_CSV,
    BASE_CSV_SHA256,
    BASE_MANIFEST,
    BASE_MANIFEST_SHA256,
    EXPECTED_TEST_ROWS,
    FIT_ID,
    FREEZE_PATH,
    LEDGER_PATH,
    REPORT_ROOT,
    RUN_ROOT,
    SAMPLE_PATH,
    TEST_PATH,
    _read_json,
    _sha,
)
from src.models.research_next_budget import read_events  # noqa: E402
from src.models.research_next_contract import canonical_sha256  # noqa: E402


def _array_digest(values: np.ndarray, dtype: str) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype=dtype).tobytes()).hexdigest()


def verify() -> dict[str, Any]:
    freeze = _read_json(FREEZE_PATH)
    protocol_path = ROOT / "experiments/optimization_v3/c19_a02_v1/C19/protocol.json"
    protocol = _read_json(protocol_path)
    if canonical_sha256(protocol) != freeze.get("protocol_sha256"):
        raise RuntimeError("C19 verifier: protocol hash mismatch")
    if canonical_sha256(freeze["identity"]) != freeze.get("freeze_signature"):
        raise RuntimeError("C19 verifier: freeze identity signature mismatch")
    for relative, expected in freeze["source_sha256"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"C19 verifier: source hash mismatch {relative}")
    for relative, expected in freeze["input_sha256"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"C19 verifier: input hash mismatch {relative}")

    manifest = _read_json(REPORT_ROOT / "submission_manifest.json")
    receipt = _read_json(RUN_ROOT / "bundle/fit_receipt.json")
    state = _read_json(RUN_ROOT / "bundle/fit_state.json")
    prediction_path = RUN_ROOT / "predictions/test_predictions.npz"
    prediction_receipt = _read_json(RUN_ROOT / "predictions/test_predictions.json")
    if _sha(prediction_path) != prediction_receipt.get("npz_sha256"):
        raise RuntimeError("C19 verifier: prediction bundle hash mismatch")
    if (receipt.get("fit_id") != FIT_ID or receipt.get("fit_rows") != 50_608
            or receipt.get("estimator_fit_count") != 1 or receipt.get("test_data_used_for_fit") is not False
            or receipt.get("sample_submission_used_for_fit") is not False
            or receipt.get("state_sha256") != canonical_sha256(state)
            or receipt.get("model_sha256") != _sha(RUN_ROOT / "bundle/model.joblib")):
        raise RuntimeError("C19 verifier: fitted state or receipt is invalid")
    receipt_check = dict(receipt)
    receipt_sha = receipt_check.pop("receipt_sha256", None)
    if receipt_sha != canonical_sha256(receipt_check):
        raise RuntimeError("C19 verifier: fit receipt signature mismatch")
    if (state.get("fit_rows") != 50_608 or state.get("cutoff") != 79
            or state.get("fit_cycle_min") != 47 or state.get("fit_cycle_max") != 79
            or state.get("test_data_used") is not False
            or state.get("validation_labels_passed_to_fit") is not False):
        raise RuntimeError("C19 verifier: fit population/source audit mismatch")

    # Read only Test identifiers/features needed to bind ordering; never load a
    # Test target column. SampleSubmission is used solely for schema and ID QA.
    test_ids = pd.read_csv(TEST_PATH, usecols=["record_id"], dtype={"record_id": str})["record_id"].tolist()
    sample = pd.read_csv(SAMPLE_PATH, dtype={"record_id": str})
    sample_ids = sample["record_id"].astype(str).tolist()
    csv_path = ROOT / manifest["csv_path"]
    csv_frame = pd.read_csv(csv_path, dtype={"record_id": str})
    with np.load(prediction_path, allow_pickle=False) as archive:
        values = {name: np.asarray(archive[name]).copy() for name in archive.files}

    required = {"record_ids", "raw_probability", "baseline_probability", "baseline_missing",
                "missing_mask", "candidate_probability", "labels"}
    if set(values) != required:
        raise RuntimeError("C19 verifier: prediction schema mismatch")
    ids = values["record_ids"].astype(str).tolist()
    if (len(ids) != EXPECTED_TEST_ROWS or len(set(ids)) != EXPECTED_TEST_ROWS
            or ids != test_ids or ids != sample_ids):
        raise RuntimeError("C19 verifier: Test/CSV/Sample IDs differ, duplicated, or missing")

    raw = values["raw_probability"].astype(np.float64)
    base = values["baseline_probability"].astype(np.float64)
    missing_base = values["baseline_missing"].astype(np.float64)
    mask = values["missing_mask"]
    candidate = values["candidate_probability"].astype(np.float64)
    labels = values["labels"]
    if (mask.dtype != np.bool_ or any(array.shape != (EXPECTED_TEST_ROWS,)
                                      for array in (raw, base, missing_base, candidate, labels, mask))
            or not all(np.isfinite(array).all() for array in (raw, base, missing_base, candidate))
            or np.any((raw < 0) | (raw > 1)) or np.any((base < 0) | (base > 1))
            or np.any((missing_base < 0) | (missing_base > 1))):
        raise RuntimeError("C19 verifier: output probabilities or mask are invalid")
    # Independent formula, avoiding runner helper reuse.
    reconstructed = np.empty(EXPECTED_TEST_ROWS, dtype=np.float64)
    reconstructed[mask] = missing_base[mask]
    reconstructed[~mask] = 0.75 * base[~mask] + 0.25 * raw[~mask]
    hard = (reconstructed >= 0.5).astype(np.int8)
    if (not np.array_equal(reconstructed, candidate) or not np.array_equal(hard, labels)
            or not np.array_equal(candidate[mask], missing_base[mask])):
        raise RuntimeError("C19 verifier: availability route, threshold, or exact fallback mismatch")
    if (csv_frame.columns.tolist() != sample.columns.tolist()
            or csv_frame["record_id"].astype(str).tolist() != ids
            or csv_frame["artifact_signal"].dtype.kind not in "iu"
            or not np.array_equal(csv_frame["artifact_signal"].to_numpy(dtype=np.int8), hard)
            or len(csv_frame) != EXPECTED_TEST_ROWS):
        raise RuntimeError("C19 verifier: submission CSV schema/order/labels mismatch")

    if _sha(BASE_CACHE) != BASE_CACHE_SHA256 or _sha(BASE_MANIFEST) != BASE_MANIFEST_SHA256 or _sha(BASE_CSV) != BASE_CSV_SHA256:
        raise RuntimeError("C19 verifier: cached sparse25 control artifact hash mismatch")
    if (manifest.get("status") != "PREPARED_VALIDATED_NOT_SUBMITTED"
            or manifest.get("historical_C19_verdict_remains", "").find("gate failure") < 0
            or manifest.get("test_target_column_loaded") is not False
            or manifest.get("preprocessing_fit_only_train") is not True
            or manifest.get("fit_predictor_count") != 1
            or manifest.get("csv_sha256") != _sha(csv_path)
            or manifest.get("csv_path") != csv_path.relative_to(ROOT).as_posix()
            or manifest.get("input_sha256") != freeze["input_sha256"]):
        raise RuntimeError("C19 verifier: submission manifest does not match the prepared artifact")

    events = read_events(LEDGER_PATH)
    starts = [event for event in events if event.get("event") == "FIT_STARTED" and event.get("fit_id") == FIT_ID]
    done = [event for event in events if event.get("fit_id") == FIT_ID
            and event.get("event") in {"FIT_SAVED", "FIT_FAILED"}]
    if (len(starts) != 1 or len(done) != 1 or done[0].get("event") != "FIT_SAVED"
            or done[0].get("receipt_sha256") != receipt["receipt_sha256"]
            or starts[0].get("reservation_key") != "CLASSIFICATION_C19_FULLTRAIN_BENCHMARK:C19_FULLTRAIN_BENCHMARK"):
        raise RuntimeError("C19 verifier: exactly-one fit accounting failed")

    result = {
        "status": "PASS",
        "candidate_id": "C19_XGB_SVD32_BENCHMARK",
        "freeze_signature": freeze["freeze_signature"],
        "fit_id": FIT_ID,
        "fit_rows": 50_608,
        "test_target_values_loaded_by_verifier": False,
        "test_rows": len(ids),
        "test_ids_unique_and_ordered": True,
        "sample_schema_matches": True,
        "route_reconstruction_bit_exact": True,
        "threshold_0_50_reconstruction_bit_exact": True,
        "missing_branch_predictions_bit_exact": True,
        "csv_sha256": _sha(csv_path),
        "csv_rows": len(csv_frame),
        "csv_columns": csv_frame.columns.tolist(),
        "baseline_cache_and_manifest_hashes_pass": True,
        "fit_receipt_and_ledger_pass": True,
        "source_and_input_hashes_pass": True,
        "checks": 12,
    }
    path = REPORT_ROOT / "verification.json"
    encoded = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError("C19 verifier output exists with different contents")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
    return result


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, ensure_ascii=False))
