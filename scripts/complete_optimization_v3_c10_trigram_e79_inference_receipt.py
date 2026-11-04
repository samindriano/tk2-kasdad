"""Complete a missing C10 inference receipt from the already-saved prediction artifact."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.optimization_v3_c10_trigram_e79_exploratory_v1 import (
    C2_WEIGHT,
    CANDIDATE_WEIGHT,
    CSV_PATH,
    FIT_ID,
    FIT_RECEIPT_PATH,
    INFERENCE_BUILD_PATH,
    MODEL_PATH,
    PREDICTIONS_PATH,
    PREFIT_FREEZE_PATH,
    REPORT,
    SPARSE25_MANIFEST,
    TEST_PATH,
    TEST_PATH as OFFICIAL_TEST,
    TRAIN_PATH,
    TRAINED_FREEZE_PATH,
    _committed_source_hashes,
    _load_sparse25_inference_arrays,
    _read_json,
    _source_hashes,
    _write_json_once,
    canonical_sha,
    file_sha,
)
from src.models.classification_round2_models import stream_missing_mask
from src.models.round1_evaluation import array_digest

RECOVERY_PATH = REPORT / "INFERENCE_BUILD_RECOVERY.json"
FIT_RECOVERY_PATH = REPORT / "FIT_RECOVERY.json"
RECOVERY_LEDGER_PATH = REPORT / "RECOVERY_LEDGER.jsonl"
PARTIAL_MODEL_PATH = ROOT / "experiments/runs/optimization_v3_c10_trigram_e79_exploratory_v1/c10_model.joblib.partial"


class InferenceReceiptError(RuntimeError):
    pass


def _write_once(path: Path, value: dict[str, Any]) -> None:
    _write_json_once(path, value)


def complete_receipt() -> dict[str, Any]:
    trained = _read_json(TRAINED_FREEZE_PATH)
    if trained.get("status") != "TRAINED_STATE_FROZEN_BEFORE_TEST_FEATURES":
        raise InferenceReceiptError("C10 trained-state freeze is not committed and ready")
    current_head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    freeze_commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "log", "-1", "--format=%H", "--",
         TRAINED_FREEZE_PATH.relative_to(ROOT).as_posix()], text=True
    ).strip()
    is_ancestor = subprocess.run(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", freeze_commit, current_head],
        check=False,
    )
    if is_ancestor.returncode != 0:
        raise InferenceReceiptError("Trained-state freeze is not committed before receipt recovery")
    if trained.get("test_features_opened") is not False or trained.get("test_labels_opened") is not False:
        raise InferenceReceiptError("Trained-state freeze was not recorded before Test access")
    if file_sha(MODEL_PATH) != trained.get("model_file_sha256"):
        raise InferenceReceiptError("Recovered model hash does not match trained-state freeze")
    receipt = _read_json(FIT_RECEIPT_PATH)
    receipt_body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if (receipt.get("receipt_sha256") != canonical_sha(receipt_body)
            or trained.get("fit_receipt_sha256") != file_sha(FIT_RECEIPT_PATH)
            or receipt.get("status") != "RECOVERED_SAME_FIT_AFTER_POST_FIT_PERSISTENCE_ERROR"):
        raise InferenceReceiptError("Recovered fit receipt and trained freeze do not reconcile")
    prefit = _read_json(PREFIT_FREEZE_PATH)
    if (receipt.get("train_sha256") != prefit.get("train", {}).get("train_sha256")
            or file_sha(TRAIN_PATH) != receipt.get("train_sha256")
            or trained.get("source_commit") != prefit.get("source_commit")
            or trained.get("recipe_sha256") != prefit.get("recipe_sha256")):
        raise InferenceReceiptError("Recovered Train/source/recipe lineage does not reconcile")
    if _source_hashes() != prefit.get("source_hashes"):
        raise InferenceReceiptError("Frozen model training sources changed after fit")
    _committed_source_hashes(prefit["source_commit"])
    fit_recovery = _read_json(FIT_RECOVERY_PATH)
    if (fit_recovery.get("status") != "PASS_RECOVERY_OF_SAME_CONSUMED_C10_FIT"
            or fit_recovery.get("recovered_model_sha256") != file_sha(MODEL_PATH)
            or fit_recovery.get("original_budget_ledger_sha256") != file_sha(REPORT / "BUDGET_LEDGER.jsonl")):
        raise InferenceReceiptError("No matching same-fit persistence recovery receipt")
    recovery_event = json.loads(RECOVERY_LEDGER_PATH.read_text(encoding="utf-8"))
    if (recovery_event.get("event") != "FIT_RECOVERED" or recovery_event.get("retry_count") != 0
            or recovery_event.get("additional_budget_units") != 0):
        raise InferenceReceiptError("Recovered-fit ledger does not preserve zero additional budget")
    if file_sha(PARTIAL_MODEL_PATH) != fit_recovery.get("original_partial_sha256"):
        raise InferenceReceiptError("Original partial model bytes changed after recovery")
    if any(not path.is_file() for path in (CSV_PATH, PREDICTIONS_PATH, TEST_PATH)):
        raise InferenceReceiptError("Expected already-produced inference artifacts are missing")

    manifest = _read_json(SPARSE25_MANIFEST)
    expected_test_sha = manifest.get("input_sha256", {}).get("test")
    if file_sha(TEST_PATH) != expected_test_sha:
        raise InferenceReceiptError("Official Test raw hash differs from verified Sparse25 lineage")
    test = pd.read_csv(OFFICIAL_TEST, usecols=["record_id", "signal_stream"])
    if (len(test) != 29421 or test["record_id"].isna().any()
            or test["record_id"].duplicated().any()):
        raise InferenceReceiptError("Official Test ID population is invalid")
    ids = test["record_id"].astype(str).to_numpy()
    id_sha = hashlib.sha256(("\n".join(ids.tolist()) + "\n").encode("utf-8")).hexdigest()
    if id_sha != manifest.get("ordered_test_ids_sha256"):
        raise InferenceReceiptError("Official Test ID order differs from the verified manifest")

    base, cb, sparse_raw, missing, sparse25_probability, sparse25_prediction = _load_sparse25_inference_arrays()
    derived_missing = np.asarray(stream_missing_mask(test), dtype=bool)
    if not np.array_equal(missing, derived_missing):
        raise InferenceReceiptError("Frozen Sparse25 availability mask differs from official Test")
    if any(array.shape != (len(test),) for array in
           (base, cb, sparse_raw, missing, sparse25_probability, sparse25_prediction)):
        raise InferenceReceiptError("Frozen Sparse25 arrays do not align with official Test rows")
    rebuilt_baseline = np.where(missing, cb, C2_WEIGHT * base + CANDIDATE_WEIGHT * sparse_raw)
    if not np.array_equal(rebuilt_baseline, sparse25_probability):
        raise InferenceReceiptError("Frozen Sparse25 baseline does not reconstruct")
    if not np.array_equal((sparse25_probability >= 0.5).astype(np.int64), sparse25_prediction):
        raise InferenceReceiptError("Frozen Sparse25 labels do not reconstruct")

    with np.load(PREDICTIONS_PATH, allow_pickle=False) as stored:
        required = {"record_id", "raw_c10_probability", "probabilities", "labels", "missing_mask",
                    "c2b_probability", "catboost_missing_probability"}
        if set(stored.files) != required:
            raise InferenceReceiptError("Saved prediction artifact schema changed")
        values = {key: np.asarray(stored[key]).copy() for key in required}
    if not np.array_equal(values["record_id"].astype(str), ids):
        raise InferenceReceiptError("Saved prediction IDs do not match official Test order")
    if not np.array_equal(values["missing_mask"].astype(bool), missing):
        raise InferenceReceiptError("Saved C10 availability mask differs from frozen baseline")
    if (not np.array_equal(values["c2b_probability"], base)
            or not np.array_equal(values["catboost_missing_probability"], cb)):
        raise InferenceReceiptError("Saved route components differ from verified Sparse25 cache")
    raw = np.asarray(values["raw_c10_probability"], dtype=np.float64)
    probabilities = np.asarray(values["probabilities"], dtype=np.float64)
    labels = np.asarray(values["labels"], dtype=np.int64)
    if any(array.shape != (len(test),) for array in (raw, probabilities, labels)):
        raise InferenceReceiptError("Saved C10 outputs do not have 29,421 rows")
    if (not np.isfinite(raw).all() or np.any((raw < 0) | (raw > 1))
            or not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1))):
        raise InferenceReceiptError("Saved C10 predictions are invalid")
    rebuilt = np.where(missing, cb, C2_WEIGHT * base + CANDIDATE_WEIGHT * raw)
    if not np.array_equal(probabilities, rebuilt) or not np.array_equal(probabilities[missing], cb[missing]):
        raise InferenceReceiptError("Saved C10 blend or missing-stream fallback does not reconstruct")
    if not np.array_equal(labels, (probabilities >= 0.5).astype(np.int64)):
        raise InferenceReceiptError("Saved labels differ from fixed 0.50 threshold")

    with CSV_PATH.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        rows = list(reader)
    if (header != ["record_id", "artifact_signal"] or len(rows) != len(test)
            or any(len(row) != 2 for row in rows)):
        raise InferenceReceiptError("Saved CSV shape or columns are invalid")
    csv_ids = np.asarray([row[0] for row in rows], dtype=str)
    csv_labels = [row[1] for row in rows]
    if (not np.array_equal(csv_ids, ids) or any(value not in {"0", "1"} for value in csv_labels)
            or not np.array_equal(np.asarray(csv_labels, dtype=np.int64), labels)):
        raise InferenceReceiptError("Saved CSV IDs or binary target do not match saved predictions")

    recovery_source = Path(__file__).resolve()
    recovery_commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "log", "-1", "--format=%H", "--", recovery_source.relative_to(ROOT).as_posix()],
        text=True,
    ).strip()
    body = {
        "schema_version": 1,
        "status": "PASS_SAVED_C10_E79_PREDICTIONS_RECEIPT_COMPLETED",
        "fit_id": FIT_ID,
        "recovery_source_commit": recovery_commit,
        "recovery_script_sha256": file_sha(recovery_source),
        "recovery_script_commit_contains_source": recovery_commit,
        "original_inference_error": "TypeError: array_digest() got an unexpected keyword argument 'dtype'",
        "original_inference_was_repeated": False,
        "saved_csv_and_npz_reused": True,
        "trained_state_freeze_sha256": file_sha(TRAINED_FREEZE_PATH),
        "model_sha256": file_sha(MODEL_PATH),
        "fit_receipt_sha256": file_sha(FIT_RECEIPT_PATH),
        "fit_recovery_sha256": file_sha(FIT_RECOVERY_PATH),
        "original_fit_failed_event_preserved": True,
        "test_sha256": file_sha(TEST_PATH),
        "csv_sha256": file_sha(CSV_PATH),
        "predictions_sha256": file_sha(PREDICTIONS_PATH),
        "test_rows": int(len(test)),
        "test_id_sequence_sha256": id_sha,
        "csv_columns": header,
        "csv_rows": len(rows),
        "saved_probabilities_reconstructed": True,
        "blend_weights": {"C2_B": C2_WEIGHT, "C10": CANDIDATE_WEIGHT},
        "missing_route_bit_exact": True,
        "threshold": 0.5,
        "test_labels_opened": False,
        "sample_submission_opened": False,
        "kaggle_contacted": False,
        "predictive_score_measured": False,
    }
    recovery_record = {**body, "recovery_sha256": canonical_sha(body)}
    _write_once(RECOVERY_PATH, recovery_record)
    build_body = {
        "schema_version": 1,
        "status": "C10_E79_INFERENCE_BUILT_PENDING_INDEPENDENT_QA",
        "fit_id": FIT_ID,
        "trained_state_freeze_sha256": file_sha(TRAINED_FREEZE_PATH),
        "fit_receipt_sha256": file_sha(REPORT / "FIT_RECEIPT.json"),
        "test_sha256": file_sha(TEST_PATH),
        "test_rows": int(len(test)),
        "ordered_test_ids_sha256": id_sha,
        "present_rows": int((~missing).sum()),
        "missing_rows": int(missing.sum()),
        "missing_fallback_bit_exact": True,
        "threshold": 0.5,
        "predicted_label_counts": {str(int(k)): int((labels == k).sum()) for k in (0, 1)},
        "probability_sha256": array_digest(probabilities),
        "label_sha256": array_digest(labels.astype(np.int64)),
        "baseline_sparse25_labels_changed": int(np.count_nonzero(labels != sparse25_prediction)),
        "csv_path": CSV_PATH.relative_to(ROOT).as_posix(),
        "csv_sha256": file_sha(CSV_PATH),
        "predictions_path": PREDICTIONS_PATH.relative_to(ROOT).as_posix(),
        "predictions_sha256": file_sha(PREDICTIONS_PATH),
        "test_labels_opened": False,
        "sample_submission_opened": False,
        "kaggle_contacted": False,
        "receipt_recovered_from_existing_outputs": True,
        "receipt_recovery_sha256": file_sha(RECOVERY_PATH),
    }
    build = {**build_body, "build_sha256": canonical_sha(build_body)}
    _write_once(INFERENCE_BUILD_PATH, build)
    return {**body, "build_sha256": build["build_sha256"],
            "present_rows": build_body["present_rows"], "missing_rows": build_body["missing_rows"],
            "predicted_label_counts": build_body["predicted_label_counts"]}


if __name__ == "__main__":
    print(json.dumps(complete_receipt(), ensure_ascii=False, sort_keys=True))
