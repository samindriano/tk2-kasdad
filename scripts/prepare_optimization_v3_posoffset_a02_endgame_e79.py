"""Prepare and independently verify the selected frozen MIX50 E79 CSV; no fit/upload."""
from __future__ import annotations

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

from src.models.research_next_contract import canonical_sha256, file_sha256  # noqa: E402

REPORT = ROOT / "reports/optimization_v3/posoffset_a02_endgame_v1"
METRICS = REPORT / "POLICY_METRICS.json"
INDEPENDENT = REPORT / "INDEPENDENT_VERIFICATION.json"
E79_MANIFEST = REPORT / "E79_SELECTED_MANIFEST.json"
E79_QA = REPORT / "E79_INDEPENDENT_QA.json"
BATCH = ROOT / "reports/optimization_v3/a02_cgeom_benchmark_v1/BATCH_MANIFEST.json"
TRAINED_FREEZE = ROOT / "reports/optimization_v3/a02_cgeom_benchmark_v1/E79_TRAINED_FREEZE.json"
INFERENCE_RECEIPT = ROOT / "experiments/runs/optimization_v3_a02_cgeom_benchmark_v1/inference_receipt.json"
BATCH_NPZ = ROOT / "experiments/runs/optimization_v3_a02_cgeom_benchmark_v1/test_predictions.npz"
A02_ROOT = ROOT / "experiments/runs/optimization_v3_a02_continuation_v1/E79/fulltrain_observed"
A02_NPZ = A02_ROOT / "test_predictions.npz"
A02_RECEIPT = A02_ROOT / "test_predictions.json"
A02_MODEL = A02_ROOT / "model.joblib"
A02_STATE = A02_ROOT / "fit_state.json"
A02_FIT_RECEIPT = A02_ROOT / "fit_receipt.json"
A02_MANIFEST = ROOT / "reports/optimization_v3/a02_continuation_v1/submission_manifest.json"
C16_CACHE = ROOT / "experiments/runs/optimization_v3_inference/v3_cls_sparse25.npz"
C16_MANIFEST = ROOT / "reports/optimization_v3/kaggle/v3_cls_sparse25.json"
C16_CSV = ROOT / "submissions/dual_campaign/optimization_v3/v3_cls_sparse25.csv"
C16_OUTPUTS = ROOT / "experiments/runs/c16_r22_submission/classification_c16_outputs.npz"
C16_AUDIT = ROOT / "experiments/runs/c16_r22_submission/classification_c16_fit_audit.json"
TEST = ROOT / "data/raw/classification/Test.csv"
SAMPLE = ROOT / "data/raw/classification/SampleSubmission.csv"
A02_CSV = ROOT / "submissions/optimization_v3/a02_continuation_v1/a02_continuation.csv"
POS_CSV = ROOT / "submissions/optimization_v3/a02_cgeom_benchmark_v1/classification_posoffset_e79.csv"
OUT = ROOT / "submissions/optimization_v3/posoffset_a02_endgame_v1/classification_mix50_e79.csv"


def _json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return obj


def _npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {key: np.asarray(z[key]).copy() for key in z.files}


def _same(actual: Any, expected: Any, label: str) -> None:
    if not np.array_equal(np.asarray(actual), np.asarray(expected)):
        raise RuntimeError(f"E79 independent equality check failed: {label}")


def _validate_prior_provenance() -> tuple[dict[str, Any], dict[str, Any]]:
    metrics = _json(METRICS)
    independent = _json(INDEPENDENT)
    batch = _json(BATCH)
    if (metrics.get("selection", {}).get("selected_policy_by_frozen_priority") != "MIX50"
            or metrics.get("policies", {}).get("MIX50", {}).get("qualifies") is not True
            or independent.get("status") != "PASS"
            or independent.get("result", {}).get("selected_policy") != "MIX50"
            or independent.get("freeze_signature") != metrics.get("freeze_signature")):
        raise RuntimeError("Selected MIX50 must pass both frozen metrics and independent validation")
    expected_official = batch.get("official_inputs_sha256", {})
    actual_official = {
        "data/raw/classification/Test.csv": file_sha256(TEST),
        "data/raw/classification/SampleSubmission.csv": file_sha256(SAMPLE),
    }
    for rel, digest in actual_official.items():
        if expected_official.get(rel) != digest:
            raise RuntimeError(f"Official E79 input hash differs from prior batch manifest: {rel}")
    if file_sha256(BATCH_NPZ) != batch.get("prediction_bundle_sha256"):
        raise RuntimeError("E79 combined probability bundle differs from prior manifest")
    if file_sha256(TRAINED_FREEZE) != batch.get("trained_freeze_sha256"):
        raise RuntimeError("E79 trained-state freeze differs from prior manifest")
    if file_sha256(A02_NPZ) != batch.get("reused_artifact_hashes", {}).get("a02_test_predictions_sha256"):
        raise RuntimeError("Original A02 E79 prediction file differs from prior manifest")
    if file_sha256(A02_RECEIPT) != batch.get("reused_artifact_hashes", {}).get("a02_test_receipt_sha256"):
        raise RuntimeError("Original A02 E79 prediction receipt differs from prior manifest")
    if file_sha256(A02_MODEL) != batch.get("reused_artifact_hashes", {}).get("a02_model_sha256"):
        raise RuntimeError("Original A02 E79 model file differs from prior manifest")
    if file_sha256(A02_MANIFEST) != batch.get("reused_artifact_hashes", {}).get("a02_manifest_sha256"):
        raise RuntimeError("Original A02 E79 manifest differs from prior manifest")
    a02_state = _json(A02_STATE)
    a02_fit_receipt = _json(A02_FIT_RECEIPT)
    a02_fit_receipt_body = dict(a02_fit_receipt)
    a02_fit_sig = a02_fit_receipt_body.pop("receipt_sha256", None)
    if (canonical_sha256(a02_state) != batch.get("reused_artifact_hashes", {}).get("a02_fit_state_sha256")
            or a02_fit_receipt.get("state_sha256") != canonical_sha256(a02_state)
            or a02_fit_sig != canonical_sha256(a02_fit_receipt_body)
            or file_sha256(A02_FIT_RECEIPT) != batch.get("reused_artifact_hashes", {}).get("a02_fit_receipt_sha256")
            or a02_fit_receipt.get("query_labels_passed_to_fit") is not False
            or a02_fit_receipt.get("test_data_used") is not False):
        raise RuntimeError("Original A02 E79 fit state/receipt lineage failed")
    if file_sha256(C16_CACHE) != batch.get("reused_artifact_hashes", {}).get("c16_reuse_cache_sha256"):
        raise RuntimeError("Full-Train C16/Sparse25 baseline cache differs from prior manifest")
    if file_sha256(C16_OUTPUTS) != batch.get("reused_artifact_hashes", {}).get("c16_prediction_only_bundle_sha256"):
        raise RuntimeError("C16 prediction-only outputs differ from prior manifest")
    if file_sha256(C16_AUDIT) != batch.get("reused_artifact_hashes", {}).get("c16_prediction_only_audit_sha256"):
        raise RuntimeError("C16 prediction-only audit differs from prior manifest")
    if file_sha256(C16_MANIFEST) != batch.get("reused_artifact_hashes", {}).get("c16_reuse_manifest_sha256"):
        raise RuntimeError("C16/Sparse25 baseline manifest differs from prior manifest")
    if file_sha256(C16_CSV) != batch.get("reused_artifact_hashes", {}).get("sparse25_csv_sha256"):
        raise RuntimeError("C16/Sparse25 baseline CSV differs from prior manifest")
    pos_csv_meta = batch.get("csvs", {}).get("POS_BAG_OFFSET", {})
    if file_sha256(POS_CSV) != pos_csv_meta.get("sha256"):
        raise RuntimeError("Original POS E79 submission CSV differs from prior batch manifest")
    a02_manifest = _json(A02_MANIFEST)
    if file_sha256(A02_CSV) != a02_manifest.get("csv_sha256"):
        raise RuntimeError("Original A02 E79 submission CSV differs from its manifest")
    inference = _json(INFERENCE_RECEIPT)
    if (inference.get("status") != "PREDICTIONS_AND_CSVS_CREATED_NOT_UPLOADED"
            or inference.get("prediction_bundle_sha256") != batch.get("prediction_bundle_sha256")
            or inference.get("test_targets_loaded") is not False
            or inference.get("kaggle_contacted") is not False
            or inference.get("a02_replay_verified_bit_exact") is not True):
        raise RuntimeError("Prior E79 inference receipt does not attest frozen non-label inference")
    a02_receipt = _json(A02_RECEIPT)
    if (a02_receipt.get("npz_sha256") != file_sha256(A02_NPZ)
            or a02_receipt.get("fit_receipt_sha256") != a02_fit_sig
            or a02_receipt.get("official_test_targets_loaded") is not False
            or a02_receipt.get("rows") != 29_421):
        raise RuntimeError("Original A02 E79 Test receipt failed")
    return metrics, batch


def _reconstruct_sources() -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    metrics, batch = _validate_prior_provenance()
    test = pd.read_csv(TEST, usecols=["record_id", "cycle_code", "signal_stream"], dtype={"record_id": str})
    sample_cols = pd.read_csv(SAMPLE, nrows=0).columns.tolist()
    sample = pd.read_csv(SAMPLE, usecols=["record_id"], dtype={"record_id": str})
    if len(test) != 29_421 or sample_cols != ["record_id", "artifact_signal"]:
        raise RuntimeError("Official Test/SampleSubmission row count or schema differs")
    ids = test.record_id.to_numpy(dtype=str)
    sample_ids = sample.record_id.to_numpy(dtype=str)
    if (len(np.unique(ids)) != len(ids) or len(np.unique(sample_ids)) != len(sample_ids)
            or not np.array_equal(ids, sample_ids)):
        raise RuntimeError("Official Test and SampleSubmission IDs/order do not match")
    cycles = pd.to_numeric(test.cycle_code, errors="raise").to_numpy(dtype=np.int64)
    if (not np.array_equal(np.unique(cycles), np.arange(80, 92, dtype=np.int64))
            or sum(int(np.sum(cycles == cycle)) for cycle in range(80, 92)) != len(test)):
        raise RuntimeError("Official Test does not cover the exact complete cycles 80-91")
    missing = test.signal_stream.fillna("").astype(str).str.strip().eq("").to_numpy()

    bundle = _npz(BATCH_NPZ)
    a02_saved = _npz(A02_NPZ)
    c16 = _npz(C16_CACHE)
    expected_bundle = {"record_ids", "missing_mask", "a02_baseline_probability", "a02_probability",
                       "c2b_probability", "c1b_missing_probability", "posoffset_direct_probability",
                       "posoffset_formula_probability", "posoffset_probability", "posoffset_labels"}
    if not expected_bundle.issubset(bundle):
        raise RuntimeError("Frozen E79 bundle is missing the exact POS/A02 probability arrays")
    if not {"record_ids", "baseline_probability", "candidate_probability", "missing_mask", "labels"}.issubset(a02_saved):
        raise RuntimeError("Original A02 E79 prediction package schema is incomplete")
    if not {"probability", "base", "sparse", "missing", "missing_mask", "prediction"}.issubset(c16):
        raise RuntimeError("Full-Train C16/Sparse25 route cache schema is incomplete")
    _same(bundle["record_ids"].astype(str), ids, "combined E79 IDs versus official Test")
    _same(a02_saved["record_ids"].astype(str), ids, "original A02 IDs versus official Test")
    _same(bundle["missing_mask"].astype(bool), missing, "combined E79 stream mask versus official Test features")
    _same(a02_saved["missing_mask"].astype(bool), missing, "A02 stream mask versus official Test features")
    _same(c16["missing_mask"].astype(bool), missing, "C16 baseline stream mask versus official Test features")

    p_pos = bundle["posoffset_probability"].astype(np.float64)
    p_a = bundle["a02_probability"].astype(np.float64)
    b = bundle["a02_baseline_probability"].astype(np.float64)
    for label, array in (("pP", p_pos), ("pA", p_a), ("b", b),
                         ("C16 probability", c16["probability"].astype(np.float64))):
        if array.shape != (len(test),) or not np.isfinite(array).all() or np.any((array < 0) | (array > 1)):
            raise RuntimeError(f"E79 {label} is not a finite aligned probability vector")

    # Rebuild the true Sparse25 route (C2_B 0.75 + sparse binary 0.25;
    # C1_B fallback on missing streams), then tie it to both A02 packages.
    b_c16 = np.where(missing, c16["missing"].astype(np.float64),
                     0.75 * c16["base"].astype(np.float64) + 0.25 * c16["sparse"].astype(np.float64))
    _same(b, c16["probability"].astype(np.float64), "A02 b versus frozen C16/Sparse25 probability")
    _same(b, b_c16, "A02 b versus independently reconstructed C16 route")
    _same(a02_saved["baseline_probability"].astype(np.float64), b, "standalone A02 baseline versus bundle b")
    _same(a02_saved["candidate_probability"].astype(np.float64), p_a, "standalone A02 pA versus E79 bundle")
    _same(a02_saved["labels"].astype(np.int8), (p_a >= 0.50).astype(np.int8), "original A02 labels from pA")
    expected_pos = np.where(missing, bundle["c1b_missing_probability"].astype(np.float64),
                            bundle["posoffset_direct_probability"].astype(np.float64))
    _same(p_pos, expected_pos, "original POS routed probability from direct source and missing fallback")
    if not np.allclose(bundle["posoffset_direct_probability"], bundle["posoffset_formula_probability"],
                       rtol=0, atol=1e-14):
        raise RuntimeError("POS direct/formula probability discrepancy exceeds frozen tolerance")
    _same(bundle["posoffset_labels"].astype(np.int8), (p_pos >= 0.50).astype(np.int8), "original POS labels from pP")
    _same(p_a[missing], b[missing], "A02 missing-stream route equals b")
    _same(p_pos[missing], bundle["c1b_missing_probability"].astype(np.float64)[missing],
          "POS missing-stream fallback identity")

    # Only after proving selected MIX50 eligibility, compute this one E79 candidate.
    candidate_prob = np.where(missing, p_pos, 0.5 * p_pos + 0.5 * p_a)
    if not np.isfinite(candidate_prob).all() or np.any((candidate_prob < 0) | (candidate_prob > 1)):
        raise RuntimeError("MIX50 E79 candidate probabilities are invalid")
    labels = (candidate_prob >= 0.50).astype(np.int8)
    frame = pd.DataFrame({"record_id": ids, "artifact_signal": labels})

    # Verify the historical submitted files reproduce their original probabilities.
    a02_csv = pd.read_csv(A02_CSV, dtype={"record_id": str})
    pos_csv = pd.read_csv(POS_CSV, dtype={"record_id": str})
    _same(a02_csv.columns.tolist(), ["record_id", "artifact_signal"], "A02 submission schema")
    _same(pos_csv.columns.tolist(), ["record_id", "artifact_signal"], "POS submission schema")
    _same(a02_csv.record_id.to_numpy(dtype=str), ids, "A02 submission ordered IDs")
    _same(pos_csv.record_id.to_numpy(dtype=str), ids, "POS submission ordered IDs")
    _same(a02_csv.artifact_signal.to_numpy(dtype=np.int8), (p_a >= 0.50).astype(np.int8), "A02 historical labels")
    _same(pos_csv.artifact_signal.to_numpy(dtype=np.int8), (p_pos >= 0.50).astype(np.int8), "POS historical labels")

    existing_matches: list[str] = []
    for old_path in (ROOT / "submissions").rglob("*.csv"):
        if old_path.resolve() == OUT.resolve():
            continue
        try:
            old = pd.read_csv(old_path, usecols=["record_id", "artifact_signal"], dtype={"record_id": str})
        except Exception:
            continue
        if (len(old) == len(frame) and np.array_equal(old.record_id.to_numpy(dtype=str), ids)
                and np.array_equal(old.artifact_signal.to_numpy(dtype=np.int8), labels)):
            existing_matches.append(old_path.relative_to(ROOT).as_posix())
    details = {
        "metrics": metrics, "batch": batch, "bundle": bundle,
        "a02_saved": a02_saved, "c16": c16, "test_ids": ids, "test_cycles": cycles,
        "missing": missing, "p_pos": p_pos, "p_a": p_a, "b": b,
        "candidate_prob": candidate_prob, "labels": labels, "frame": frame,
        "existing_identical_csvs": existing_matches,
    }
    return frame, labels, details


def prepare() -> dict[str, Any]:
    frame, labels, d = _reconstruct_sources()
    if d["existing_identical_csvs"]:
        raise RuntimeError("MIX50 label vector duplicates existing submission(s): " + ", ".join(d["existing_identical_csvs"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        old = pd.read_csv(OUT, dtype={"record_id": str})
        if not old.equals(frame):
            raise RuntimeError("Refusing to overwrite different prepared MIX50 CSV")
    else:
        frame.to_csv(OUT, index=False)
    recipe = _json(REPORT / "RECIPE.json")
    manifest = {
        "status": "PREPARED_NOT_SUBMITTED",
        "candidate": "MIX50",
        "selected_by_frozen_priority": True,
        "freeze_signature": d["metrics"]["freeze_signature"],
        "preparation_source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "formula": "missing: pP exactly; present: 0.5*pP + 0.5*pA; threshold >= 0.50",
        "rows": len(frame), "columns": frame.columns.tolist(),
        "unique_ids": int(frame.record_id.nunique()),
        "cycle_range": [int(d["test_cycles"].min()), int(d["test_cycles"].max())],
        "missing_rows": int(d["missing"].sum()),
        "positive_count": int(labels.sum()),
        "label_vector_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
        "csv_path": OUT.relative_to(ROOT).as_posix(),
        "csv_sha256": file_sha256(OUT),
        "new_fits": 0, "test_targets_loaded": False, "upload_performed": False,
        "existing_identical_submissions": [],
        "source_commit": d["batch"].get("source_commit"),
        "source_hashes": {
            "batch_manifest": file_sha256(BATCH),
            "trained_freeze": file_sha256(TRAINED_FREEZE),
            "inference_receipt": file_sha256(INFERENCE_RECEIPT),
            "combined_e79_probability_bundle": file_sha256(BATCH_NPZ),
            "original_a02_test_bundle": file_sha256(A02_NPZ),
            "original_a02_test_receipt": file_sha256(A02_RECEIPT),
            "original_a02_model": file_sha256(A02_MODEL),
            "original_a02_fit_state": file_sha256(A02_STATE),
            "original_a02_fit_receipt": file_sha256(A02_FIT_RECEIPT),
            "original_a02_manifest": file_sha256(A02_MANIFEST),
            "c16_sparse25_cache": file_sha256(C16_CACHE),
            "c16_sparse25_manifest": file_sha256(C16_MANIFEST),
            "c16_sparse25_csv": file_sha256(C16_CSV),
            "c16_prediction_bundle": file_sha256(C16_OUTPUTS),
            "c16_prediction_audit": file_sha256(C16_AUDIT),
            "official_test": file_sha256(TEST), "sample_submission": file_sha256(SAMPLE),
            "original_a02_csv": file_sha256(A02_CSV), "original_pos_csv": file_sha256(POS_CSV),
        },
        "official_test_checks": {
            "IDs_equal_bundle_a02_and_sample_in_order": True,
            "cycles_complete_80_91": True,
            "missing_mask_matches_test_signal_stream": True,
            "A02_baseline_matches_C16_cache_bit_exact": True,
            "A02_and_POS_historical_labels_reconstructed": True,
            "test_targets_loaded": False,
        },
        "recipe_sha256": file_sha256(REPORT / "RECIPE.json"),
        "metric_report_sha256": file_sha256(METRICS),
        "independent_metric_qa_sha256": file_sha256(INDEPENDENT),
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    if E79_MANIFEST.exists():
        if _json(E79_MANIFEST) != manifest:
            raise RuntimeError("Refusing to overwrite a different E79 selected manifest")
    else:
        E79_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def verify() -> dict[str, Any]:
    manifest = _json(E79_MANIFEST)
    stored_sig = manifest.get("manifest_sha256")
    body = dict(manifest)
    body.pop("manifest_sha256", None)
    if stored_sig != canonical_sha256(body):
        raise RuntimeError("E79 selected manifest checksum invalid")
    frame, labels, d = _reconstruct_sources()
    candidate = pd.read_csv(OUT, dtype={"record_id": str})
    if candidate.columns.tolist() != ["record_id", "artifact_signal"] or len(candidate) != 29_421:
        raise RuntimeError("Prepared CSV schema or row count invalid")
    if not np.array_equal(candidate.record_id.to_numpy(dtype=str), d["test_ids"]):
        raise RuntimeError("Prepared CSV ordered IDs differ from official Test")
    out_labels = candidate.artifact_signal.to_numpy()
    if out_labels.dtype.kind not in "iu" or not np.isin(out_labels, [0, 1]).all():
        raise RuntimeError("Prepared CSV labels are not integer 0/1")
    _same(out_labels.astype(np.int8), labels, "prepared MIX50 labels versus independent formula")
    for key, path in (("batch_manifest", BATCH), ("trained_freeze", TRAINED_FREEZE),
                      ("inference_receipt", INFERENCE_RECEIPT), ("combined_e79_probability_bundle", BATCH_NPZ),
                      ("original_a02_test_bundle", A02_NPZ), ("original_a02_test_receipt", A02_RECEIPT),
                      ("original_a02_model", A02_MODEL), ("original_a02_fit_state", A02_STATE),
                      ("original_a02_fit_receipt", A02_FIT_RECEIPT), ("original_a02_manifest", A02_MANIFEST),
                      ("c16_sparse25_cache", C16_CACHE),
                      ("c16_sparse25_manifest", C16_MANIFEST), ("c16_sparse25_csv", C16_CSV),
                      ("c16_prediction_bundle", C16_OUTPUTS), ("c16_prediction_audit", C16_AUDIT),
                      ("official_test", TEST), ("sample_submission", SAMPLE),
                      ("original_a02_csv", A02_CSV), ("original_pos_csv", POS_CSV)):
        if manifest["source_hashes"].get(key) != file_sha256(path):
            raise RuntimeError(f"E79 source hash mismatch: {key}")
    if manifest.get("csv_sha256") != file_sha256(OUT):
        raise RuntimeError("Prepared CSV hash differs from E79 manifest")
    if (manifest.get("positive_count") != int(labels.sum())
            or manifest.get("label_vector_sha256") != hashlib.sha256(labels.tobytes()).hexdigest()
            or manifest.get("freeze_signature") != d["metrics"]["freeze_signature"]
            or manifest.get("candidate") != "MIX50"):
        raise RuntimeError("Prepared CSV manifest values mismatch")
    qa = {
        "status": "PASS",
        "candidate": "MIX50",
        "checks": {
            "source_bundle_hashes_match_prior_E79_manifest": True,
            "official_Test_ID_cycle_availability_join": True,
            "C16_baseline_reconstructed_and_matches_A02_b": True,
            "original_POS_A02_probabilities_and_labels_replayed": True,
            "MIX50_formula_and_missing_identity_replayed": True,
            "CSV_schema_order_labels_and_hash_match": True,
            "duplicate_existing_label_vector_absent": not d["existing_identical_csvs"],
        },
        "rows": len(candidate), "unique_ids": int(candidate.record_id.nunique()),
        "cycle_range": [int(d["test_cycles"].min()), int(d["test_cycles"].max())],
        "missing_rows": int(d["missing"].sum()), "positive_count": int(labels.sum()),
        "csv_path": OUT.relative_to(ROOT).as_posix(), "csv_sha256": file_sha256(OUT),
        "manifest_sha256": stored_sig, "test_targets_loaded": False, "fits": 0, "upload": False,
    }
    if not all(qa["checks"].values()):
        raise RuntimeError("E79 independent QA has a failing check")
    if E79_QA.exists():
        if _json(E79_QA) != qa:
            raise RuntimeError("Refusing to overwrite different independent E79 QA")
    else:
        E79_QA.write_text(json.dumps(qa, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return qa


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "verify"))
    args = parser.parse_args()
    result = prepare() if args.command == "prepare" else verify()
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
