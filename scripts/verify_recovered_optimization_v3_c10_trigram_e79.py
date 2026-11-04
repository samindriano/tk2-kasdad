"""Independent QA of the recovered C10 E79 saved inference outputs; no model call."""
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
    RECIPE_PATH,
    REPORT,
    SPARSE25_BUNDLE,
    SPARSE25_MANIFEST,
    TEST_PATH,
    TRAIN_PATH,
    TRAINED_FREEZE_PATH,
    _committed_source_hashes,
    _load_sparse25_inference_arrays,
    _read_json,
    _source_hashes,
    canonical_sha,
    file_sha,
)
from src.models.classification_round2_models import stream_missing_mask
from src.models.round1_evaluation import array_digest

FIT_RECOVERY_PATH = REPORT / "FIT_RECOVERY.json"
FIT_LEDGER_PATH = REPORT / "BUDGET_LEDGER.jsonl"
RECOVERY_LEDGER_PATH = REPORT / "RECOVERY_LEDGER.jsonl"
INFERENCE_RECOVERY_PATH = REPORT / "INFERENCE_BUILD_RECOVERY.json"
QA_PATH = REPORT / "QA_VERIFICATION.json"
PARTIAL_MODEL_PATH = ROOT / "experiments/runs/optimization_v3_c10_trigram_e79_exploratory_v1/c10_model.joblib.partial"


class RecoveredOutputError(RuntimeError):
    pass


def _canonical_ids(ids: np.ndarray) -> str:
    return hashlib.sha256(("\n".join(ids.tolist()) + "\n").encode("utf-8")).hexdigest()


def _verify_signed_json(value: dict[str, Any], key: str) -> None:
    body = {name: item for name, item in value.items() if name != key}
    if value.get(key) != canonical_sha(body):
        raise RecoveredOutputError(f"{key} signature mismatch")


def verify_saved_outputs() -> dict[str, Any]:
    if QA_PATH.exists():
        raise RecoveredOutputError("Independent QA receipt already exists")
    recipe = _read_json(RECIPE_PATH)
    prefit = _read_json(PREFIT_FREEZE_PATH)
    trained = _read_json(TRAINED_STATE_FREEZE_PATH)
    fit_receipt = _read_json(FIT_RECEIPT_PATH)
    fit_recovery = _read_json(FIT_RECOVERY_PATH)
    build = _read_json(INFERENCE_BUILD_PATH)
    inference_recovery = _read_json(INFERENCE_RECOVERY_PATH)
    manifest = _read_json(SPARSE25_MANIFEST)

    if recipe.get("benchmark_id") != "C10_TRIGRAM_E79_EXPLORATORY_V1" or trained.get("fit_id") != FIT_ID:
        raise RecoveredOutputError("C10 recipe or fit identity mismatch")
    if trained.get("status") != "TRAINED_STATE_FROZEN_BEFORE_TEST_FEATURES":
        raise RecoveredOutputError("Trained-state freeze status mismatch")
    if trained.get("test_features_opened") is not False or trained.get("test_labels_opened") is not False:
        raise RecoveredOutputError("Trained-state boundary was not frozen before Test access")
    _verify_signed_json(trained, "freeze_signature")
    _verify_signed_json(fit_receipt, "receipt_sha256")
    _verify_signed_json(build, "build_sha256")
    _verify_signed_json(fit_recovery, "recovery_sha256")
    _verify_signed_json(inference_recovery, "recovery_sha256")

    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    trained_commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "log", "-1", "--format=%H", "--",
         TRAINED_STATE_FREEZE_PATH.relative_to(ROOT).as_posix()], text=True
    ).strip()
    if subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", trained_commit, head],
                      check=False).returncode != 0:
        raise RecoveredOutputError("Trained-state freeze is not in current committed history")
    if trained.get("source_commit") != prefit.get("source_commit"):
        raise RecoveredOutputError("Trained-state source commit differs from prefit freeze")
    if _source_hashes() != prefit.get("source_hashes"):
        raise RecoveredOutputError("Frozen C10 training source hashes changed")
    _committed_source_hashes(prefit["source_commit"])

    ledger = [json.loads(line) for line in FIT_LEDGER_PATH.read_text(encoding="utf-8").splitlines()]
    recovery_event = json.loads(RECOVERY_LEDGER_PATH.read_text(encoding="utf-8"))
    if [event["event"] for event in ledger] != ["FIT_RESERVED", "FIT_ARMED", "FIT_STARTED", "FIT_FAILED"]:
        raise RecoveredOutputError("Original budget ledger does not preserve the post-fit failure")
    if recovery_event.get("event") != "FIT_RECOVERED" or recovery_event.get("additional_budget_units") != 0:
        raise RecoveredOutputError("Separate recovery ledger does not reconcile")
    if (fit_recovery.get("original_budget_ledger_sha256") != file_sha(FIT_LEDGER_PATH)
            or fit_recovery.get("recovered_model_sha256") != file_sha(MODEL_PATH)
            or file_sha(PARTIAL_MODEL_PATH) != fit_recovery.get("original_partial_sha256")
            or fit_receipt.get("status") != "RECOVERED_SAME_FIT_AFTER_POST_FIT_PERSISTENCE_ERROR"):
        raise RecoveredOutputError("Recovered model bytes do not reconcile to original fit/failure")
    if (trained.get("fit_receipt_sha256") != file_sha(FIT_RECEIPT_PATH)
            or trained.get("model_file_sha256") != file_sha(MODEL_PATH)
            or trained.get("recovery_manifest_sha256") != file_sha(FIT_RECOVERY_PATH)):
        raise RecoveredOutputError("Trained-state freeze references changed artifacts")
    if (fit_receipt.get("train_sha256") != file_sha(TRAIN_PATH)
            or fit_receipt.get("recipe_sha256") != file_sha(RECIPE_PATH)
            or fit_receipt.get("fit_rows") != 50608
            or fit_receipt.get("target_sha256") != ledger[2].get("target_sha256")):
        raise RecoveredOutputError("Train, target, recipe, or fit population lineage mismatch")

    expected_test_sha = manifest.get("input_sha256", {}).get("test")
    if file_sha(TEST_PATH) != expected_test_sha or build.get("test_sha256") != expected_test_sha:
        raise RecoveredOutputError("Official Test input hash mismatch")
    test = pd.read_csv(TEST_PATH, usecols=["record_id", "signal_stream"])
    if len(test) != 29421 or test["record_id"].isna().any() or test["record_id"].duplicated().any():
        raise RecoveredOutputError("Official Test ID count, missingness, or uniqueness failed")
    ids = test["record_id"].astype(str).to_numpy()
    ids_sha = _canonical_ids(ids)
    if ids_sha != manifest.get("ordered_test_ids_sha256") or ids_sha != build.get("ordered_test_ids_sha256"):
        raise RecoveredOutputError("Official Test IDs do not match accepted order")

    base, catboost, sparse_raw, missing, baseline_probability, baseline_labels = _load_sparse25_inference_arrays()
    test_missing = np.asarray(stream_missing_mask(test), dtype=bool)
    if not np.array_equal(missing, test_missing):
        raise RecoveredOutputError("Sparse25 cached missing mask differs from official Test streams")
    if not np.array_equal(np.where(missing, catboost, C2_WEIGHT * base + CANDIDATE_WEIGHT * sparse_raw),
                          baseline_probability):
        raise RecoveredOutputError("Sparse25 baseline component reconstruction failed")

    with np.load(PREDICTIONS_PATH, allow_pickle=False) as data:
        required = {"record_id", "raw_c10_probability", "probabilities", "labels", "missing_mask",
                    "c2b_probability", "catboost_missing_probability"}
        if set(data.files) != required:
            raise RecoveredOutputError("Saved prediction NPZ keys differ from the frozen schema")
        arrays = {key: np.asarray(data[key]).copy() for key in required}
    if not np.array_equal(arrays["record_id"].astype(str), ids):
        raise RecoveredOutputError("Saved prediction IDs do not match official Test")
    if (not np.array_equal(arrays["missing_mask"].astype(bool), missing)
            or not np.array_equal(arrays["c2b_probability"], base)
            or not np.array_equal(arrays["catboost_missing_probability"], catboost)):
        raise RecoveredOutputError("Saved route inputs differ from frozen baseline components")
    raw = np.asarray(arrays["raw_c10_probability"], dtype=np.float64)
    probabilities = np.asarray(arrays["probabilities"], dtype=np.float64)
    labels = np.asarray(arrays["labels"], dtype=np.int64)
    if any(value.shape != (29421,) for value in (raw, probabilities, labels)):
        raise RecoveredOutputError("Saved model outputs have the wrong row count")
    if (not np.isfinite(raw).all() or np.any((raw < 0) | (raw > 1))
            or not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1))):
        raise RecoveredOutputError("Saved probabilities are non-finite or outside [0,1]")
    rebuilt = np.where(missing, catboost, C2_WEIGHT * base + CANDIDATE_WEIGHT * raw)
    if not np.array_equal(probabilities, rebuilt) or not np.array_equal(probabilities[missing], catboost[missing]):
        raise RecoveredOutputError("Saved probability output does not match frozen C10 route")
    if not np.array_equal(labels, (probabilities >= 0.5).astype(np.int64)):
        raise RecoveredOutputError("Saved labels do not match fixed threshold 0.50")

    with CSV_PATH.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        rows = list(reader)
    if (header != ["record_id", "artifact_signal"] or len(rows) != 29421
            or any(len(row) != 2 for row in rows)):
        raise RecoveredOutputError("CSV columns or row count are invalid")
    csv_ids = np.asarray([row[0] for row in rows], dtype=str)
    csv_target = [row[1] for row in rows]
    if (not np.array_equal(csv_ids, ids) or len(np.unique(csv_ids)) != 29421
            or any(value not in {"0", "1"} for value in csv_target)
            or not np.array_equal(np.asarray(csv_target, dtype=np.int64), labels)):
        raise RecoveredOutputError("CSV target or IDs mismatch the saved prediction artifact")
    if (file_sha(CSV_PATH) != build.get("csv_sha256")
            or file_sha(PREDICTIONS_PATH) != build.get("predictions_sha256")
            or file_sha(INFERENCE_RECOVERY_PATH) != build.get("receipt_recovery_sha256")):
        raise RecoveredOutputError("Saved output hashes differ from their receipts")
    if build.get("probability_sha256") != array_digest(probabilities):
        raise RecoveredOutputError("Probability digest differs from the inference receipt")
    if build.get("label_sha256") != array_digest(labels.astype(np.int64)):
        raise RecoveredOutputError("Label digest differs from the inference receipt")
    if build.get("receipt_recovered_from_existing_outputs") is not True:
        raise RecoveredOutputError("Inference receipt does not disclose post-output recovery")

    qa = {
        "schema_version": 1,
        "status": "PASS_INDEPENDENT_C10_E79_SAVED_OUTPUT_QA",
        "fit_id": FIT_ID,
        "qa_head": head,
        "trained_state_commit": trained_commit,
        "source_commit": trained["source_commit"],
        "test_sha256": file_sha(TEST_PATH),
        "train_sha256": file_sha(TRAIN_PATH),
        "recipe_sha256": file_sha(RECIPE_PATH),
        "csv_path": CSV_PATH.relative_to(ROOT).as_posix(),
        "csv_sha256": file_sha(CSV_PATH),
        "prediction_bundle_path": PREDICTIONS_PATH.relative_to(ROOT).as_posix(),
        "prediction_bundle_sha256": file_sha(PREDICTIONS_PATH),
        "rows": 29421,
        "csv_columns": header,
        "exact_official_test_id_order": True,
        "duplicate_or_missing_ids": False,
        "classification_labels_integer_binary": True,
        "probabilities_finite_in_unit_interval": True,
        "fixed_blend_and_missing_route_reconstructed": True,
        "threshold": 0.5,
        "predicted_label_counts": {"0": int((labels == 0).sum()), "1": int((labels == 1).sum())},
        "test_labels_opened": False,
        "sample_submission_opened": False,
        "kaggle_contacted": False,
        "model_inference_repeated_during_qa": False,
        "fit_recovery_disclosed": True,
        "inference_receipt_recovery_disclosed": True,
        "predictive_score_measured": False,
        "historical_c10_verdict": "RETIRE",
    }
    QA_PATH.write_text(json.dumps(qa, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return qa


if __name__ == "__main__":
    print(json.dumps(verify_saved_outputs(), ensure_ascii=False, sort_keys=True))
