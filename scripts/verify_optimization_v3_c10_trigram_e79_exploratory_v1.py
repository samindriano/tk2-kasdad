"""Independent read-only QA for the one-shot C10 E79 inference artifact."""
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
    INFERENCE_BUILD_PATH,
    MODEL_PATH,
    PREFIT_FREEZE_PATH,
    PREDICTIONS_PATH,
    RECIPE_PATH,
    REPORT,
    SPARSE25_BUNDLE,
    SPARSE25_MANIFEST,
    SUBMISSIONS,
    TEST_PATH,
    TRAIN_PATH,
    TRAINED_FREEZE_PATH,
    file_sha,
)
from src.models.classification_round2_models import stream_missing_mask
from src.models.round1_evaluation import array_digest

VERIFY_PATH = REPORT / "QA_VERIFICATION.json"


class VerificationError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise VerificationError(f"Expected JSON object: {path}")
    return result


def _sha_json(value: dict[str, Any], key: str) -> str:
    body = {name: item for name, item in value.items() if name != key}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def _committed_file_head(path: Path) -> str:
    relative = path.relative_to(ROOT).as_posix()
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "log", "-1", "--format=%H", "--", relative], text=True
    ).strip()


def verify() -> dict[str, Any]:
    if VERIFY_PATH.exists():
        raise VerificationError("QA_VERIFICATION already exists; refusing to replace evidence")
    recipe = _read_json(RECIPE_PATH)
    prefit = _read_json(PREFIT_FREEZE_PATH)
    trained = _read_json(TRAINED_FREEZE_PATH)
    build = _read_json(INFERENCE_BUILD_PATH)
    manifest = _read_json(SPARSE25_MANIFEST)

    if recipe.get("benchmark_id") != "C10_TRIGRAM_E79_EXPLORATORY_V1":
        raise VerificationError("C10 recipe namespace mismatch")
    if prefit.get("status") != "FROZEN_BEFORE_C10_E79_FULLTRAIN_FIT":
        raise VerificationError("Pre-fit freeze status mismatch")
    if trained.get("status") != "TRAINED_STATE_FROZEN_BEFORE_TEST_FEATURES":
        raise VerificationError("Trained-state freeze status mismatch")
    if trained.get("fit_id") != FIT_ID or build.get("fit_id") != FIT_ID:
        raise VerificationError("C10 fit identity mismatch")
    if trained.get("source_commit") != prefit.get("source_commit"):
        raise VerificationError("Trained state does not reference the prefit source commit")
    if trained.get("recipe_sha256") != file_sha(RECIPE_PATH):
        raise VerificationError("Trained state recipe hash mismatch")
    if prefit.get("train", {}).get("train_sha256") != manifest.get("input_sha256", {}).get("train"):
        raise VerificationError("Official Train lineage mismatch")
    if file_sha(TRAIN_PATH) != prefit.get("train", {}).get("train_sha256"):
        raise VerificationError("Official Train raw file hash changed")
    if prefit.get("test_features_opened") is not False or prefit.get("test_labels_opened") is not False:
        raise VerificationError("Prefit record violates the Test boundary")
    if trained.get("test_features_opened") is not False or trained.get("test_labels_opened") is not False:
        raise VerificationError("Trained-state freeze was not recorded before Test access")
    if _sha_json(trained, "freeze_signature") != trained.get("freeze_signature"):
        raise VerificationError("Trained-state freeze signature mismatch")
    if _sha_json(build, "build_sha256") != build.get("build_sha256"):
        raise VerificationError("Inference build signature mismatch")
    if _committed_file_head(TRAINED_FREEZE_PATH) != subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip():
        raise VerificationError("Trained-state freeze is not at current HEAD")
    if file_sha(MODEL_PATH) != trained.get("model_file_sha256"):
        raise VerificationError("Fitted C10 model binary hash mismatch")

    expected_test_sha = manifest.get("input_sha256", {}).get("test")
    if file_sha(TEST_PATH) != expected_test_sha or build.get("test_sha256") != expected_test_sha:
        raise VerificationError("Official Test raw file hash mismatch")
    test_ids = pd.read_csv(TEST_PATH, usecols=["record_id"])
    if len(test_ids) != 29421 or test_ids["record_id"].isna().any() or test_ids["record_id"].duplicated().any():
        raise VerificationError("Official Test ID population is invalid")
    ids = test_ids["record_id"].astype(str).to_numpy()
    ids_digest = hashlib.sha256(("\n".join(ids.tolist()) + "\n").encode("utf-8")).hexdigest()
    if ids_digest != manifest.get("ordered_test_ids_sha256"):
        raise VerificationError("Official Test IDs differ from frozen manifest")

    with CSV_PATH.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if header != ["record_id", "artifact_signal"]:
            raise VerificationError("CSV columns or order are invalid")
        rows = list(reader)
    if len(rows) != 29421 or any(len(row) != 2 for row in rows):
        raise VerificationError("CSV row count or width is invalid")
    csv_ids = np.asarray([row[0] for row in rows], dtype=str)
    if not np.array_equal(csv_ids, ids) or len(np.unique(csv_ids)) != 29421:
        raise VerificationError("CSV record_id sequence differs from official Test")
    label_strings = [row[1] for row in rows]
    if any(value not in {"0", "1"} for value in label_strings):
        raise VerificationError("Classification CSV target values are not integer 0/1")
    labels = np.asarray(label_strings, dtype=np.int64)
    if file_sha(CSV_PATH) != build.get("csv_sha256"):
        raise VerificationError("CSV file hash differs from the inference receipt")

    with np.load(PREDICTIONS_PATH, allow_pickle=False) as saved:
        expected_keys = {"record_id", "raw_c10_probability", "probabilities", "labels", "missing_mask",
                         "c2b_probability", "catboost_missing_probability"}
        if set(saved.files) != expected_keys:
            raise VerificationError("Saved C10 inference array schema mismatch")
        arrays = {key: np.asarray(saved[key]).copy() for key in expected_keys}
    if file_sha(PREDICTIONS_PATH) != build.get("predictions_sha256"):
        raise VerificationError("Prediction array file hash differs from receipt")
    if not np.array_equal(arrays["record_id"].astype(str), ids):
        raise VerificationError("Saved inference IDs do not align to official Test")

    with np.load(SPARSE25_BUNDLE, allow_pickle=False) as baseline:
        base = np.asarray(baseline["base"], dtype=np.float64).copy()
        cb = np.asarray(baseline["missing"], dtype=np.float64).copy()
        sparse_raw = np.asarray(baseline["sparse"], dtype=np.float64).copy()
        cached_missing = np.asarray(baseline["missing_mask"], dtype=bool).copy()
        baseline_probability = np.asarray(baseline["probability"], dtype=np.float64).copy()
        baseline_prediction = np.asarray(baseline["prediction"], dtype=np.int64).copy()
    if file_sha(SPARSE25_BUNDLE) != prefit.get("baseline", {}).get("bundle_sha256"):
        raise VerificationError("Sparse25 baseline inference bundle hash mismatch")
    # The stream column is accessed only after the committed trained-state freeze.
    test_stream = pd.read_csv(TEST_PATH, usecols=["signal_stream"])["signal_stream"]
    from_frame_missing = np.asarray(stream_missing_mask(pd.DataFrame({"signal_stream": test_stream})), dtype=bool)
    if not np.array_equal(cached_missing, from_frame_missing):
        raise VerificationError("Cached Sparse25 missing mask differs from official Test stream availability")
    if any(value.shape != (29421,) for value in
           (base, cb, sparse_raw, cached_missing, baseline_probability, baseline_prediction,
            arrays["raw_c10_probability"], arrays["probabilities"], arrays["labels"],
            arrays["missing_mask"], arrays["c2b_probability"], arrays["catboost_missing_probability"])):
        raise VerificationError("Inference arrays do not have exactly 29,421 rows")
    if not np.array_equal(arrays["missing_mask"].astype(bool), cached_missing):
        raise VerificationError("Saved C10 missing mask does not match the frozen baseline")
    if not np.array_equal(arrays["c2b_probability"], base) or not np.array_equal(arrays["catboost_missing_probability"], cb):
        raise VerificationError("Saved route inputs differ from the frozen baseline bundle")
    if not np.array_equal(np.where(cached_missing, cb, C2_WEIGHT * base + CANDIDATE_WEIGHT * sparse_raw),
                          baseline_probability):
        raise VerificationError("Sparse25 baseline probability reconstruction failed")
    if not np.array_equal((baseline_probability >= 0.5).astype(np.int64), baseline_prediction):
        raise VerificationError("Sparse25 baseline label reconstruction failed")

    raw = np.asarray(arrays["raw_c10_probability"], dtype=np.float64)
    probabilities = np.asarray(arrays["probabilities"], dtype=np.float64)
    saved_labels = np.asarray(arrays["labels"], dtype=np.int64)
    if (not np.isfinite(raw).all() or np.any((raw < 0) | (raw > 1))
            or not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1))):
        raise VerificationError("C10 probabilities are non-finite or outside [0,1]")
    rebuilt = np.where(cached_missing, cb, C2_WEIGHT * base + CANDIDATE_WEIGHT * raw)
    if not np.array_equal(probabilities, rebuilt):
        raise VerificationError("C10 availability route does not reconstruct exactly")
    if not np.array_equal(probabilities[cached_missing], cb[cached_missing]):
        raise VerificationError("Missing-stream CatBoost route changed")
    if not np.array_equal(saved_labels, (probabilities >= 0.5).astype(np.int64)):
        raise VerificationError("Saved labels do not match frozen threshold 0.50")
    if not np.array_equal(labels, saved_labels):
        raise VerificationError("CSV labels differ from saved inference labels")

    result = {
        "schema_version": 1,
        "status": "PASS_INDEPENDENT_C10_E79_CSV_AND_ROUTE_QA",
        "benchmark_id": recipe["benchmark_id"],
        "fit_id": FIT_ID,
        "head": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "source_commit": trained["source_commit"],
        "recipe_sha256": file_sha(RECIPE_PATH),
        "train_sha256": file_sha(TRAIN_PATH),
        "test_sha256": file_sha(TEST_PATH),
        "csv_path": CSV_PATH.relative_to(ROOT).as_posix(),
        "csv_sha256": file_sha(CSV_PATH),
        "csv_rows": len(rows),
        "csv_columns": header,
        "test_ids_exact_order": True,
        "duplicate_or_missing_ids": False,
        "labels_integer_binary": True,
        "probabilities_finite_unit_interval": True,
        "missing_branch_bit_exact": True,
        "blend_reconstructed_from_frozen_components": True,
        "threshold": 0.5,
        "predicted_label_counts": {"0": int((labels == 0).sum()), "1": int((labels == 1).sum())},
        "official_train_rows_fit": trained["fit_rows"],
        "learned_unit_total": trained["learned_unit_total"],
        "test_labels_opened": False,
        "sample_submission_opened": False,
        "kaggle_contacted": False,
        "predictive_score_measured": False,
        "historical_c10_verdict": "RETIRE",
        "benchmark_role": "exploratory full-Train inference artifact; not validation or promotion evidence",
    }
    VERIFY_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                           encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(verify(), ensure_ascii=False, sort_keys=True))
