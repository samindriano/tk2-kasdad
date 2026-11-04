"""Independent output verifier for the single POS_MOMENTS E79 candidate."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit

from scripts import optimization_v3_a02_cgeom_benchmark_v1 as cgeom
from src.models.positional_moments_v1 import build_moment_matrix, sha256_array, sha256_csr, sha256_file


BENCHMARK_ID = "POS_MOMENTS_REAL_E79_EXPLORATORY_BENCHMARK"
TRAINED_FREEZE = ROOT / "reports/optimization_v3/positional_moments_real_e79_benchmark_v1/TRAINED_STATE_FREEZE.json"
PREFIT_FREEZE = ROOT / "reports/optimization_v3/positional_moments_real_e79_benchmark_v1/PRE_FIT_FREEZE.json"
FIT_RECEIPT = ROOT / "reports/optimization_v3/positional_moments_real_e79_benchmark_v1/FIT_RECEIPT.json"
BUILD = ROOT / "reports/optimization_v3/positional_moments_real_e79_benchmark_v1/INFERENCE_BUILD.json"
QA = ROOT / "reports/optimization_v3/positional_moments_real_e79_benchmark_v1/INFERENCE_QA.json"
MANIFEST = ROOT / "reports/optimization_v3/positional_moments_real_e79_benchmark_v1/BENCHMARK_MANIFEST.json"
CSV = ROOT / "submissions/optimization_v3/positional_moments_real_e79_benchmark_v1/classification_pos_moments_real_e79_exploratory.csv"
GAMMA = ROOT / "experiments/runs/optimization_v3_positional_moments_real_e79_benchmark_v1/fit/gamma.npz"
TEST = ROOT / "data/raw/classification/Test.csv"
ORIGINAL_A02_NPZ = cgeom.A02_TEST_NPZ
PREDICTION_BUNDLE = cgeom.OUTPUT_NPZ
EXPECTED_TEST_SHA = "4e6c381d582ed24f92d17f620f4f0ad25e2e0def29eb57a91a5482867f6afe3c"
EXPECTED_ORIGINAL_A02_NPZ_SHA = "c267a4b998aca64982b6f63b10bedb2f3b3061007166db7731d221788ac89e47"
EXPECTED_PREDICTION_BUNDLE_SHA = "ffc55aa2d18b75438257f3cf0aa205baa3f4c2922180e2b3688b36c352c00271"


class VerificationError(RuntimeError):
    pass


def _sha_obj(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise VerificationError(f"Expected JSON object: {path}")
    return value


def _committed_matches(path: Path) -> bool:
    relative = path.relative_to(ROOT).as_posix()
    try:
        blob = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=ROOT)
    except subprocess.CalledProcessError:
        return False
    return hashlib.sha256(blob).hexdigest() == sha256_file(path)


def verify_outputs() -> dict[str, Any]:
    trained = _read_json(TRAINED_FREEZE)
    prefit = _read_json(PREFIT_FREEZE)
    receipt = _read_json(FIT_RECEIPT)
    build = _read_json(BUILD)
    if (trained.get("benchmark_id") != BENCHMARK_ID
            or trained.get("status") != "TRAINED_STATE_FROZEN_BEFORE_TEST_INFERENCE"
            or trained.get("test_labels_loaded") is not False
            or trained.get("test_or_sample_rows_loaded") is not False
            or receipt.get("retry_count") != 0 or receipt.get("fit_count_for_benchmark") != 1
            or not _committed_matches(TRAINED_FREEZE) or not _committed_matches(FIT_RECEIPT)):
        raise VerificationError("Committed one-fit trained freeze/receipt boundary did not verify")
    expected_sources = prefit.get("source_hashes", {})
    current_sources = {}
    for rel, expected in expected_sources.items():
        path = ROOT / rel
        if not path.is_file() or sha256_file(path) != expected:
            raise VerificationError(f"Frozen source changed: {rel}")
        current_sources[rel] = sha256_file(path)
    if (sha256_file(TEST) != EXPECTED_TEST_SHA
            or sha256_file(ORIGINAL_A02_NPZ) != EXPECTED_ORIGINAL_A02_NPZ_SHA
            or sha256_file(PREDICTION_BUNDLE) != EXPECTED_PREDICTION_BUNDLE_SHA):
        raise VerificationError("Official Test input or original A02 probability bundle hash changed")

    test_columns = list(dict.fromkeys(["record_id", "cycle_code", *cgeom.CLASSIFICATION_FEATURES]))
    test = pd.read_csv(TEST, usecols=test_columns,
                       dtype={"record_id": str})
    ids = test.record_id.astype(str).to_numpy(dtype=str)
    cycles = pd.to_numeric(test.cycle_code, errors="raise").to_numpy(dtype=np.int64)
    if ("artifact_signal" in test.columns or len(test) != 29_421
            or test.record_id.isna().any() or (ids == "").any() or test.record_id.duplicated().any()
            or not np.array_equal(np.unique(cycles), np.arange(80, 92, dtype=np.int64))):
        raise VerificationError("Official Test feature rows, IDs, or cycle coverage failed")

    base, _, transform_audit = cgeom._c2b_matrices(test, train_side=False)
    estimator = cgeom._load_active_c2b_lr()["estimator"]
    logits = np.asarray(estimator.decision_function(base), dtype=np.float64).reshape(-1)
    model = joblib.load(cgeom.A02_MODEL)
    vectorizer = model.text_space.vectorizer
    docs, missing = cgeom.positional._text_values(test.signal_stream)
    # Independent verifier uses the frozen vectorized moment builder; the
    # committed Train preflight bound its exact equality to the canonical
    # scalar token-occurrence replay used for the fit.
    moments, _audit = build_moment_matrix(docs, vectorizer, shuffled=False)
    if (sha256_csr(moments) != build.get("moment_matrix_sha256")
            or sha256_csr(base) != build.get("base_matrix_sha256")
            or sha256_array(logits) != build.get("base_logits_sha256")
            or not np.array_equal(missing, np.asarray(transform_audit["missing"], dtype=bool))):
        raise VerificationError("Independent Test moment/base-logit/availability rebuild differs")

    with np.load(PREDICTION_BUNDLE, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]).copy() for key in archive.files}
    with np.load(ORIGINAL_A02_NPZ, allow_pickle=False) as archive:
        original_a02 = {key: np.asarray(archive[key]).copy() for key in archive.files}
    required = {"record_ids", "missing_mask", "a02_probability", "c1b_missing_probability", "posoffset_probability"}
    if not required.issubset(arrays) or not {"record_ids", "candidate_probability"}.issubset(original_a02):
        raise VerificationError("Original A02 probability bundle schema is incomplete")
    if (not np.array_equal(ids, arrays["record_ids"].astype(str))
            or not np.array_equal(ids, original_a02["record_ids"].astype(str))
            or not np.array_equal(arrays["a02_probability"], original_a02["candidate_probability"])
            or not np.array_equal(missing, arrays["missing_mask"].astype(bool))):
        raise VerificationError("Original A02 bundle IDs/missing mask do not match official Test")
    with np.load(GAMMA, allow_pickle=False) as archive:
        gamma = np.asarray(archive["gamma"], dtype=np.float64).copy()
    if sha256_array(gamma) != trained.get("gamma_sha256"):
        raise VerificationError("Persisted gamma does not match committed trained freeze")
    z0 = logits
    p_offset = np.asarray(expit(z0 + moments @ gamma), dtype=np.float64).reshape(-1)
    p_a02 = arrays["a02_probability"].astype(np.float64)
    p_p = arrays["c1b_missing_probability"].astype(np.float64)
    p_pos = arrays["posoffset_probability"].astype(np.float64)
    if any(value.shape != (len(ids),) or not np.isfinite(value).all()
           for value in (p_a02, p_p, p_pos)):
        raise VerificationError("Original A02 prediction bundle contains an invalid probability vector")
    probabilities = np.where(missing, p_p, 0.5 * p_offset + 0.5 * p_a02)
    expected_labels = (probabilities >= 0.50).astype(np.int8)
    pos_labels = (p_pos >= 0.50).astype(np.int8)
    mix50_prob = np.where(missing, p_p, 0.5 * p_pos + 0.5 * p_a02)
    mix50_labels = (mix50_prob >= 0.50).astype(np.int8)
    a02_labels = (p_a02 >= 0.50).astype(np.int8)

    csv_frame = pd.read_csv(CSV, dtype={"record_id": str})
    if (len(csv_frame) != 29_421 or csv_frame.columns.tolist() != ["record_id", "artifact_signal"]
            or csv_frame.record_id.isna().any() or csv_frame.record_id.duplicated().any()
            or not np.array_equal(csv_frame.record_id.astype(str).to_numpy(dtype=str), ids)):
        raise VerificationError("Candidate CSV row count, IDs, order, or exact schema failed")
    label_values = pd.to_numeric(csv_frame.artifact_signal, errors="raise").to_numpy()
    if (label_values.dtype.kind not in "iu" or not np.isin(label_values, [0, 1]).all()
            or not np.array_equal(label_values.astype(np.int8), expected_labels)):
        raise VerificationError("Candidate CSV labels differ from independently rebuilt fixed policy")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise VerificationError("Independently rebuilt candidate probabilities are invalid")
    if not np.array_equal(probabilities[missing], p_p[missing]):
        raise VerificationError("C1_B missing-stream fallback is not bit-exact")
    if (sha256_array(p_offset) != build.get("present_probability_sha256")
            or sha256_array(probabilities) != build.get("candidate_probability_sha256")):
        raise VerificationError("Independent fixed-policy probability reconstruction differs from inference build")

    label_hashes = {"A02": sha256_array(a02_labels), "POS_BAG_OFFSET": sha256_array(pos_labels),
                    "MIX50": sha256_array(mix50_labels), "E79": sha256_array(expected_labels)}
    if build.get("label_identity_hashes_for_duplicate_check_only") != label_hashes:
        raise VerificationError("Identity-only comparator label hashes differ from inference build")
    qa = {"schema_version": 1, "status": "PASS_INDEPENDENT_FULL_QA_CANDIDATE_NOT_SUBMITTED",
          "benchmark_id": BENCHMARK_ID, "historical_scientific_verdict": "POS_MOMENTS_V1 PILOT_GATE_FAILED; unchanged",
          "candidate_role": "authorized exploratory benchmark only; not qualified/champion",
          "csv_path": CSV.relative_to(ROOT).as_posix(), "csv_sha256": sha256_file(CSV),
          "rows": len(csv_frame), "columns": csv_frame.columns.tolist(),
          "ordered_test_ids_sha256": sha256_array(ids),
          "test_sha256": EXPECTED_TEST_SHA,
          "original_a02_test_predictions_sha256": EXPECTED_ORIGINAL_A02_NPZ_SHA,
          "frozen_candidate_probability_bundle_sha256": EXPECTED_PREDICTION_BUNDLE_SHA,
          "trained_state_freeze_sha256": sha256_file(TRAINED_FREEZE),
          "fit_receipt_sha256": sha256_file(FIT_RECEIPT), "gamma_sha256": sha256_array(gamma),
          "base_matrix_sha256": sha256_csr(base), "moment_matrix_sha256": sha256_csr(moments),
          "base_logits_sha256": sha256_array(logits), "probability_sha256": sha256_array(probabilities),
          "label_sha256": sha256_array(expected_labels), "threshold_rule": "probability >= 0.50",
          "present_rows": int((~missing).sum()), "missing_rows": int(missing.sum()),
          "cycles": {str(int(c)): int(np.sum(cycles == c)) for c in np.unique(cycles)},
          "checks": {"exact_schema": True, "row_count_29421": True, "ordered_unique_test_ids": True,
                     "integer_binary_labels": True, "finite_probability_range_0_1": True,
                     "missing_route_bit_exact_pP": True, "independent_feature_rebuild": True,
                     "test_labels_loaded": False, "sample_submission_opened": False,
                     "kaggle_contacted": False},
          "label_identity_hashes_for_duplicate_check_only": label_hashes,
          "unique_comparator_label_vectors": len(set(label_hashes.values())),
          "source_hashes": current_sources, "created_at_utc": datetime.now(timezone.utc).isoformat()}
    qa_path = QA
    payload = json.dumps(qa, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8") + b"\n"
    if qa_path.exists() and qa_path.read_bytes() != payload:
        raise VerificationError("Independent QA output already exists and differs")
    if not qa_path.exists():
        qa_path.parent.mkdir(parents=True, exist_ok=True)
        qa_path.write_bytes(payload)
    manifest = {"schema_version": 1, "status": qa["status"],
                "benchmark_id": BENCHMARK_ID, "source_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "recipe_path": "reports/optimization_v3/positional_moments_real_e79_benchmark_v1/RECIPE.json",
                "authorization_path": "reports/optimization_v3/positional_moments_real_e79_benchmark_v1/BENCHMARK_AUTHORIZATION.md",
                "pre_fit_freeze_sha256": sha256_file(PREFIT_FREEZE),
                "trained_state_freeze_sha256": sha256_file(TRAINED_FREEZE),
                "fit_receipt_sha256": sha256_file(FIT_RECEIPT),
                "csv_path": CSV.relative_to(ROOT).as_posix(), "csv_sha256": sha256_file(CSV),
                "qa_path": QA.relative_to(ROOT).as_posix(), "qa_sha256": sha256_file(QA),
                "input_hashes": {"classification_train": receipt["train_sha256"],
                                 "classification_test": EXPECTED_TEST_SHA,
                                 "original_a02_test_predictions": EXPECTED_ORIGINAL_A02_NPZ_SHA,
                                 "frozen_candidate_probability_bundle": EXPECTED_PREDICTION_BUNDLE_SHA},
                "fit_id": receipt["fit_id"], "fit_rows": receipt["fit_rows"],
                "fit_target_sha256": receipt["target_sha256"],
                "reused_state_hashes": prefit["reused_state_audit"],
                "submission": {"uploaded": False, "kaggle_contacted": False},
                "test_labels_loaded": False, "sample_submission_opened": False,
                "historical_pilot_verdict_changed": False}
    manifest_payload = json.dumps(manifest, sort_keys=True, ensure_ascii=False,
                                  allow_nan=False, indent=2).encode("utf-8") + b"\n"
    if MANIFEST.exists() and MANIFEST.read_bytes() != manifest_payload:
        raise VerificationError("Benchmark manifest already exists and differs")
    if not MANIFEST.exists():
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_bytes(manifest_payload)
    return qa


if __name__ == "__main__":
    print(json.dumps(verify_outputs(), sort_keys=True, ensure_ascii=False, indent=2))
