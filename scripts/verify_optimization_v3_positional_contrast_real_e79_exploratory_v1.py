"""Independent structural and fixed-policy QA for POS_CONTRAST REAL E79."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit

from scripts import optimization_v3_a02_cgeom_benchmark_v1 as cgeom
from scripts import optimization_v3_positional_contrast_real_e79_exploratory_v1 as benchmark
from scripts import optimization_v3_positional_text_pilot_v1 as positional
from src.models.optimization_v3_positional_text_pilot_v1 import build_position_block
from src.models.positional_contrast_v1 import contrast_block
from src.models.positional_moments_v1 import sha256_array, sha256_csr, sha256_file


def _exact(left: np.ndarray, right: np.ndarray, name: str) -> None:
    if not np.array_equal(np.asarray(left), np.asarray(right)):
        raise benchmark.BenchmarkError(f"Independent verifier found a mismatch: {name}")


def verify_outputs() -> dict[str, Any]:
    trained = benchmark._verify_trained_freeze()
    prefit = benchmark._verify_signed(benchmark.PREFIT_FREEZE, "freeze_sha256")
    benchmark._assert_sources(prefit["source_hashes"])
    for path, expected in ((benchmark.TRAIN, benchmark.TRAIN_SHA),
                           (benchmark.TEST, benchmark.TEST_SHA),
                           (benchmark.SAMPLE, benchmark.SAMPLE_SHA)):
        if sha256_file(path) != expected:
            raise benchmark.BenchmarkError(f"Input SHA mismatch during independent QA: {path.name}")

    feature_columns = list(dict.fromkeys(["record_id", "cycle_code", *cgeom.CLASSIFICATION_FEATURES]))
    test = pd.read_csv(benchmark.TEST, usecols=feature_columns, dtype={"record_id": str})
    if "artifact_signal" in test.columns or len(test) != 29_421:
        raise benchmark.BenchmarkError("Independent QA loaded a target or found the wrong Test row count")
    ids = test.record_id.astype(str).to_numpy(dtype=str)
    cycles = pd.to_numeric(test.cycle_code, errors="raise").to_numpy(dtype=np.int64)
    sample_columns = list(pd.read_csv(benchmark.SAMPLE, nrows=0).columns)
    sample_ids = pd.read_csv(benchmark.SAMPLE, usecols=["record_id"],
                             dtype={"record_id": str}).record_id.astype(str).to_numpy()
    if (sample_columns != ["record_id", "artifact_signal"] or len(sample_ids) != len(ids)
            or not np.array_equal(sample_ids, ids) or len(np.unique(sample_ids)) != len(ids)):
        raise benchmark.BenchmarkError("Independent QA found a SampleSubmission schema/ID-order mismatch")

    arrays = benchmark._load_npz(benchmark.PREDICTION_PATH,
        ("record_ids", "cycles", "missing_mask", "base_logits", "base_probability",
         "contrast_sha256", "theta", "contrast_probability", "a02_probability",
         "c1b_missing_probability", "gamma_zero_probability", "candidate_probability", "labels"))
    _exact(arrays["record_ids"].astype(str), ids, "ordered record IDs")
    _exact(arrays["cycles"].astype(np.int64), cycles, "cycle IDs")
    theta = arrays["theta"].astype(np.float64)
    if (sha256_array(theta) != trained["theta_sha256"]
            or sha256_file(benchmark.THETA_PATH) != trained["theta_file_sha256"]):
        raise benchmark.BenchmarkError("Independent QA theta identity differs from the trained freeze")

    # Rebuild the Test-side C2_B base and REAL third-position contrast from the
    # frozen preprocessing; this does prediction transforms only, never fit.
    a02_model = __import__("joblib").load(cgeom.A02_MODEL)
    vectorizer = a02_model.text_space.vectorizer
    base, _, base_audit = cgeom._c2b_matrices(test, vectorizer=vectorizer, train_side=False)
    position, _ = build_position_block(test.signal_stream.astype(str).tolist(), vectorizer, shuffled=False)
    Z = contrast_block(sparse.hstack((base, position), format="csr", dtype=np.float64), base.shape[1])
    c2b = cgeom._load_active_c2b_lr()["estimator"]
    z0 = np.asarray(c2b.decision_function(base), dtype=np.float64).reshape(-1)
    base_probability = np.asarray(expit(z0), dtype=np.float64)
    missing = np.asarray(base_audit["missing"], dtype=bool)
    _exact(arrays["missing_mask"].astype(bool), missing, "missing-stream mask")
    _exact(arrays["base_logits"].astype(np.float64), z0, "C2_B offset logits")
    _exact(arrays["base_probability"].astype(np.float64), base_probability, "gamma-zero C2_B probability")
    contrast_sha = sha256_csr(Z)
    if contrast_sha != str(arrays["contrast_sha256"][0]):
        raise benchmark.BenchmarkError("Independent QA rebuilt a different Test contrast matrix")
    q = np.asarray(expit(z0 + Z @ theta), dtype=np.float64).reshape(-1)
    _exact(arrays["contrast_probability"].astype(np.float64), q, "REAL contrast probability")

    # Independently reproduce the saved A02 and C1_B route components.
    source_bundle = benchmark._load_npz(cgeom.OUTPUT_NPZ,
        ("record_ids", "missing_mask", "a02_probability", "c1b_missing_probability"))
    source_a02 = benchmark._load_npz(cgeom.A02_TEST_NPZ, ("record_ids", "candidate_probability"))
    _exact(source_bundle["record_ids"].astype(str), ids, "cached benchmark bundle IDs")
    _exact(source_a02["record_ids"].astype(str), ids, "original A02 bundle IDs")
    _exact(source_bundle["a02_probability"].astype(np.float64),
           source_a02["candidate_probability"].astype(np.float64), "original A02 probability replay")
    _exact(arrays["a02_probability"].astype(np.float64),
           source_bundle["a02_probability"].astype(np.float64), "A02 route component")
    c1_state = cgeom._load_unit(cgeom.INTEGRATION_ACTIVE_ID_BY_SLOT[4])
    c1_estimator = cgeom._load_unit(cgeom.INTEGRATION_ACTIVE_ID_BY_SLOT[5])["estimator"]
    c1_matrix = positional.transform_c1b_structured(c1_state, cgeom._features(test))
    p_c1b = positional.positive_probability(c1_estimator, c1_matrix)
    _exact(source_bundle["c1b_missing_probability"].astype(np.float64),
           np.asarray(p_c1b, dtype=np.float64), "full-Train C1_B HGB fallback replay")
    _exact(arrays["c1b_missing_probability"].astype(np.float64),
           np.asarray(p_c1b, dtype=np.float64), "saved C1_B fallback route")

    gamma_zero = np.where(missing, p_c1b,
                          0.5 * base_probability + 0.5 * source_bundle["a02_probability"])
    candidate = np.where(missing, p_c1b,
                         0.5 * q + 0.5 * source_bundle["a02_probability"])
    _exact(arrays["gamma_zero_probability"].astype(np.float64), gamma_zero, "gamma-zero fixed-policy replay")
    _exact(arrays["candidate_probability"].astype(np.float64), candidate, "fixed candidate policy")
    if not np.array_equal(candidate[missing], p_c1b[missing]):
        raise benchmark.BenchmarkError("Missing-stream fallback is not bit-exact")
    expected_labels = (candidate >= 0.50).astype(np.int8)
    _exact(arrays["labels"].astype(np.int8), expected_labels, ">= 0.50 classification threshold")

    csv = pd.read_csv(benchmark.CSV, dtype={"record_id": str})
    if (list(csv.columns) != sample_columns or len(csv) != 29_421
            or csv.record_id.isna().any() or csv.record_id.duplicated().any()):
        raise benchmark.BenchmarkError("CSV schema, row count, or ID uniqueness failed")
    _exact(csv.record_id.astype(str).to_numpy(), sample_ids, "CSV/sample ordered IDs")
    numeric_labels = pd.to_numeric(csv.artifact_signal, errors="raise").to_numpy()
    if (numeric_labels.dtype.kind not in "iu" or not np.isin(numeric_labels, [0, 1]).all()):
        raise benchmark.BenchmarkError("CSV labels are not integer 0/1")
    _exact(numeric_labels.astype(np.int8), expected_labels, "CSV labels vs recomputed policy")

    events = benchmark._ledger_events()
    reservations = [e for e in events if e.get("event") == "CARD_FITS_RESERVED"
                    and e.get("reservation_key") == benchmark.RESERVATION_KEY]
    starts = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("fit_id") == benchmark.FIT_ID]
    saved = [e for e in events if e.get("event") == "FIT_SAVED" and e.get("fit_id") == benchmark.FIT_ID]
    failed = [e for e in events if e.get("event") == "FIT_FAILED" and e.get("fit_id") == benchmark.FIT_ID]
    if len(reservations) != 1 or len(starts) != 1 or len(saved) != 1 or failed:
        raise benchmark.BenchmarkError("Ledger does not record exactly one reserved, started, and saved theta fit")
    pilot_results = benchmark._read_json(benchmark.ROOT / "reports/optimization_v3/positional_contrast_v1/RESULTS.json")
    if pilot_results.get("status") != "PILOT_GATE_FAILED":
        raise benchmark.BenchmarkError("Historical POS_CONTRAST pilot failure verdict changed")

    result = {"schema_version": 1, "status": "PASS_INDEPENDENT_CSV_AND_POLICY_QA",
              "benchmark_id": benchmark.BENCHMARK_ID, "csv_path": benchmark.CSV.relative_to(benchmark.ROOT).as_posix(),
              "csv_sha256": sha256_file(benchmark.CSV), "prediction_bundle_sha256": sha256_file(benchmark.PREDICTION_PATH),
              "train_sha256": benchmark.TRAIN_SHA, "test_sha256": benchmark.TEST_SHA,
              "sample_submission_sha256": benchmark.SAMPLE_SHA,
              "rows": len(csv), "columns": list(csv.columns),
              "ordered_ids_sha256": sha256_array(ids),
              "label_counts": pd.Series(numeric_labels).value_counts().sort_index().to_dict(),
              "present_rows": int((~missing).sum()), "missing_rows": int(missing.sum()),
              "contrast_matrix_sha256": contrast_sha,
              "candidate_probability_sha256": sha256_array(candidate),
              "gamma_zero_replay": "PASS_BIT_EXACT",
              "missing_fallback_exact": True,
              "c1b_fallback_recomputed_from_frozen_estimator": True,
              "ledger_counts": {"reserved": len(reservations), "fit_started": len(starts),
                                "fit_saved": len(saved), "fit_failed": len(failed)},
              "historical_pilot_verdict": pilot_results["status"],
              "test_labels_loaded": False, "kaggle_contacted": False}
    benchmark._write_once(benchmark.QA_REPORT, result)
    return result
