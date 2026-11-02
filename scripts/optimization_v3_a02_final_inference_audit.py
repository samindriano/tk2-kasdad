"""Independent post-fit audit for the gated A02 full-Train inference bundle."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import optimization_v3_a02_continuation as runner  # noqa: E402
from scripts import optimization_v3_c19_benchmark as c19  # noqa: E402
from src.models.research_next_budget import read_events  # noqa: E402
from src.models.research_next_contract import canonical_sha256, file_sha256  # noqa: E402


FIT_ID = "A02_CONT_E79_OBSERVED_FULLTRAIN"
FOLDER = runner.RUN_ROOT / "E79/fulltrain_observed"
CSV_PATH = ROOT / "submissions/optimization_v3/a02_continuation_v1/a02_continuation.csv"
MANIFEST_PATH = runner.REPORT_ROOT / "submission_manifest.json"
AUDIT_PATH = runner.REPORT_ROOT / "final_inference_audit.json"
TRAIN_SHA256 = runner.EXPECTED_TRAIN_SHA256


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_once(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"Immutable final-inference audit differs: {path}")
        return
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
        stream.flush()


def audit() -> dict[str, Any]:
    freeze = runner.verify_freeze()
    results = _json(runner.REPORT_ROOT / "results.json")
    verification = _json(runner.REPORT_ROOT / "verification.json")
    if (verification.get("status") != "PASS"
            or results.get("gates", {}).get("conditional_final_eligible") is not True
            or results.get("gates", {}).get("risk_review_hold") is not False):
        raise RuntimeError("Frozen continuation gate does not allow a final-output audit")

    manifest = _json(MANIFEST_PATH)
    if (manifest.get("status") != "PREPARED_VALIDATED_NOT_SUBMITTED"
            or manifest.get("submission_authorized") is not False
            or manifest.get("fit_id") != FIT_ID
            or manifest.get("freeze_signature") != freeze["freeze_signature"]
            or manifest.get("test_target_loaded") is not False):
        raise RuntimeError("Final inference manifest status, lineage, or source boundary differs")

    train, test, sample = c19._load_train_test_sample()
    base = c19._load_baseline_cache(test, sample)
    if len(train) != 50_608 or file_sha256(c19.TRAIN_PATH) != TRAIN_SHA256:
        raise RuntimeError("Official Train row count/hash differs")
    if len(test) != 29_421 or "artifact_signal" in test.columns:
        raise RuntimeError("Official Test schema/count failed or a Test target column was loaded")
    test_ids = test.record_id.astype(str).to_numpy(dtype=str)
    sample_ids = sample.record_id.astype(str).to_numpy(dtype=str)
    if (len(np.unique(test_ids)) != len(test_ids)
            or not np.array_equal(test_ids, sample_ids)
            or sample.columns.tolist() != ["record_id", "artifact_signal"]):
        raise RuntimeError("Official Test IDs or SampleSubmission schema/order failed")

    prediction_path = FOLDER / "test_predictions.npz"
    prediction_receipt_path = FOLDER / "test_predictions.json"
    prediction_receipt = _json(prediction_receipt_path)
    if (file_sha256(prediction_path) != manifest.get("prediction_bundle_sha256")
            or prediction_receipt.get("npz_sha256") != file_sha256(prediction_path)
            or prediction_receipt.get("freeze_signature") != freeze["freeze_signature"]
            or prediction_receipt.get("official_test_targets_loaded") is not False):
        raise RuntimeError("Saved Test prediction bundle/receipt hash or source boundary failed")
    with np.load(prediction_path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]).copy() for key in archive.files}
    expected_schema = {
        "record_ids", "baseline_probability", "missing_mask", "candidate_probability", "labels",
        "neighbor_count", "neighbor_source_rows", "neighbor_ids", "neighbor_similarities",
        "correction", "correction_unclipped",
    }
    if set(arrays) != expected_schema:
        raise RuntimeError("Final prediction archive schema changed")
    if (not np.array_equal(arrays["record_ids"].astype(str), test_ids)
            or not np.array_equal(arrays["baseline_probability"], base["probability"])
            or not np.array_equal(arrays["missing_mask"], base["missing_mask"])):
        raise RuntimeError("Saved Test IDs or C16 baseline route differs from the official aligned inputs")
    p = arrays["candidate_probability"].astype(np.float64)
    labels = arrays["labels"]
    if (p.shape != (29_421,) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1))
            or labels.dtype != np.int8 or not np.array_equal(labels, (p >= 0.5).astype(np.int8))
            or not np.array_equal(p[arrays["missing_mask"]], arrays["baseline_probability"][arrays["missing_mask"]])):
        raise RuntimeError("Probability bounds, threshold, or missing-stream fallback failed")

    model_path, state_path, receipt_path = FOLDER / "model.joblib", FOLDER / "fit_state.json", FOLDER / "fit_receipt.json"
    state, receipt = _json(state_path), _json(receipt_path)
    model = joblib.load(model_path)
    if (canonical_sha256(model.audit_state()) != canonical_sha256(state)
            or file_sha256(model_path) != manifest.get("model_sha256")
            or canonical_sha256(state) != manifest.get("fit_state_sha256")
            or receipt.get("fit_id") != FIT_ID
            or receipt.get("freeze_signature") != freeze["freeze_signature"]
            or receipt.get("test_data_used") is not False
            or receipt.get("query_labels_passed_to_fit") is not False
            or receipt.get("text_prefix_rows") != 50_608
            or receipt.get("bank_present_rows") != 36_010):
        raise RuntimeError("Full-Train model state or fit receipt lineage failed")
    receipt_payload = dict(receipt)
    receipt_signature = receipt_payload.pop("receipt_sha256", None)
    if (receipt_signature != canonical_sha256(receipt_payload)
            or receipt_signature != manifest.get("fit_receipt_sha256")):
        raise RuntimeError("Full-Train fit receipt canonical hash failed")

    expected_frame = pd.read_csv(CSV_PATH, dtype={"record_id": str})
    if (file_sha256(CSV_PATH) != manifest.get("csv_sha256")
            or expected_frame.columns.tolist() != sample.columns.tolist()
            or len(expected_frame) != 29_421
            or not np.array_equal(expected_frame.record_id.astype(str).to_numpy(), test_ids)
            or expected_frame.record_id.isna().any()
            or expected_frame.record_id.duplicated().any()
            or not np.array_equal(expected_frame.artifact_signal.to_numpy(dtype=np.int8), labels)):
        raise RuntimeError("Submission CSV hash, schema, ordered IDs, uniqueness, or labels failed")

    events = read_events(runner.LEDGER_PATH)
    started = [e for e in events if e.get("event") == "FIT_STARTED"]
    saved = [e for e in events if e.get("event") == "FIT_SAVED"]
    final_started = [e for e in started if e.get("fit_id") == FIT_ID]
    final_saved = [e for e in saved if e.get("fit_id") == FIT_ID]
    if (len(started) != 66 or len(final_started) != 1 or len(final_saved) != 1
            or any(e.get("event") == "FIT_FAILED" and e.get("fit_id") == FIT_ID for e in events)):
        raise RuntimeError("Final fit budget accounting is not exactly one successful conditional fit")

    # Independent deterministic inference replay using the persisted fitted state.
    test_rows = np.arange(len(test), dtype=np.int64) + len(train)
    replay = model.predict(
        query_texts=test.signal_stream.reset_index(drop=True), query_record_ids=test_ids,
        query_row_positions=test_rows, query_cycles=test.cycle_code.to_numpy(dtype=np.int64),
        baseline_probabilities=base["probability"], missing_mask=base["missing_mask"],
    )
    replay_map = {
        "candidate_probability": "candidate_probability", "neighbor_count": "neighbor_count",
        "neighbor_source_rows": "neighbor_source_rows", "neighbor_ids": "neighbor_ids",
        "neighbor_similarities": "neighbor_similarities", "correction": "correction",
        "correction_unclipped": "correction_unclipped",
    }
    for saved_key, replay_key in replay_map.items():
        if not np.array_equal(arrays[saved_key], replay[replay_key]):
            raise RuntimeError(f"Persisted Test predictions do not replay exactly: {saved_key}")

    checks = {
        "frozen_continuation_gate_pass_and_no_risk_hold": True,
        "manifest_and_final_fit_receipt_hashes": True,
        "official_train_full_population_and_hash": True,
        "test_has_no_target_column": True,
        "test_sample_ids_order_and_schema": True,
        "baseline_cache_hash_and_route": True,
        "prediction_archive_hash_and_schema": True,
        "probability_threshold_and_missing_fallback": True,
        "model_state_full_train_and_oof_lineage": True,
        "csv_row_count_order_uniqueness_labels_and_hash": True,
        "budget_exactly_66_started_and_one_final_saved": True,
        "persisted_predictions_replay_bit_exact": True,
        "kaggle_submission_performed": False,
        "test_targets_or_scores_used": False,
    }
    report = {
        "status": "PASS", "candidate_id": manifest["candidate_id"],
        "freeze_signature": freeze["freeze_signature"],
        "audit_source_sha256": file_sha256(Path(__file__).resolve()),
        "csv_path": manifest["csv_path"], "csv_sha256": file_sha256(CSV_PATH),
        "prediction_bundle_sha256": file_sha256(prediction_path),
        "model_sha256": file_sha256(model_path),
        "official_train_sha256": file_sha256(c19.TRAIN_PATH),
        "official_test_feature_sha256": file_sha256(c19.TEST_PATH),
        "sample_submission_sha256": file_sha256(c19.SAMPLE_PATH),
        "rows": len(expected_frame), "columns": expected_frame.columns.tolist(),
        "label_counts": {str(int(key)): int(value) for key, value in expected_frame.artifact_signal.value_counts().sort_index().items()},
        "official_test_targets_loaded": False, "submission_authorized": False,
        "status_detail": "Prepared and locally verified; not submitted to Kaggle.",
        "checks": checks,
    }
    _write_once(AUDIT_PATH, report)
    return report


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, ensure_ascii=False, allow_nan=False))
