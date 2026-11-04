"""Independent no-fit verification for the fixed REAL E79 benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RUN_REL = Path("experiments/runs/reg_position_real_e79_benchmark_v1")
REPORT_REL = Path("reports/optimization_v3/reg_position_real_e79_benchmark_v1")
TEST_REL = Path("data/raw/regression/Test.csv")
SAMPLE_REL = Path("data/raw/regression/SampleSubmission.csv")
TEST_SHA256 = "4e6c381d582ed24f92d17f620f4f0ad25e2e0def29eb57a91a5482867f6afe3c"
SAMPLE_SHA256 = "63a51f4b7dfbb0eec9800099407abd8be14d25ec18a533cd39dc854ffc190a32"
TEST_ROWS = 29_421
TEST_COLUMNS = ("record_id", "cycle_code", "signal_stream")
FIT_IDS = ("WORD_TFIDF_FULL_TRAIN", "REAL_POSITION_RESIDUAL_L1_E79")


class VerifyError(RuntimeError):
    """Raised when independent validation fails."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.astype(str).tolist() if value.dtype.kind in "OUS" else value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(_jsonable(value), sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_array(value: Any, dtype: str | None = None) -> str:
    array = np.asarray(value, dtype=dtype)
    digest = hashlib.sha256(_canonical({"shape": array.shape, "dtype": str(array.dtype)}))
    if array.dtype.kind in "OUS":
        digest.update(_canonical(array.astype(str).tolist()))
    else:
        digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()


def _sha_csr(value: sparse.spmatrix) -> str:
    matrix = sparse.csr_matrix(value, dtype=np.float64, copy=True)
    matrix.sum_duplicates()
    matrix.sort_indices()
    return _canonical_sha({
        "shape": list(matrix.shape),
        "data_sha256": _sha_array(matrix.data),
        "indices_sha256": _sha_array(matrix.indices),
        "indptr_sha256": _sha_array(matrix.indptr),
    })


def _normal_tfidf(params: dict[str, Any]) -> dict[str, Any]:
    result = dict(params)
    result["ngram_range"] = list(result["ngram_range"])
    result["dtype"] = np.dtype(result["dtype"]).name
    return _jsonable(result)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]).copy() for key in archive.files}


def _verify_synthetic(temp_root: Path) -> dict[str, Any]:
    temp_root = temp_root.resolve()
    run = temp_root / "isolated" / RUN_REL
    fixture_path = temp_root / "fixture.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    manifest = json.loads((run / "synthetic_inference/manifest.json").read_text(encoding="utf-8"))
    vectorizer_path = run / "states/WORD_TFIDF_FULL_TRAIN.joblib"
    model_path = run / "states/REAL_POSITION_RESIDUAL_L1_E79.joblib"
    vector_receipt_path = run / "receipts/WORD_TFIDF_FULL_TRAIN.json"
    head_receipt_path = run / "receipts/REAL_POSITION_RESIDUAL_L1_E79.json"
    bundle_path = run / "synthetic_inference/predictions.npz"
    csv_path = run / "synthetic_inference/submission.csv"
    for path in (run, fixture_path, vectorizer_path, model_path, vector_receipt_path,
                 head_receipt_path, bundle_path, csv_path):
        try:
            path.resolve().relative_to(temp_root)
        except ValueError as exc:
            raise VerifyError("Synthetic artifact escaped its temporary root") from exc
    expected = (
        (vectorizer_path, manifest["vectorizer_sha256"]),
        (model_path, manifest["head_sha256"]),
        (vector_receipt_path, manifest["vectorizer_receipt_sha256"]),
        (head_receipt_path, manifest["head_receipt_sha256"]),
        (bundle_path, manifest["bundle_sha256"]),
        (csv_path, manifest["csv_sha256"]),
    )
    if any(_sha_file(path) != digest for path, digest in expected):
        raise VerifyError("Synthetic state, receipt, bundle or CSV hash differs")
    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)
    if not isinstance(vectorizer, TfidfVectorizer) or not hasattr(vectorizer, "vocabulary_"):
        raise VerifyError("Synthetic fitted vectorizer failed")
    documents = fixture["test_documents"]
    full = vectorizer.transform(documents).tocsr()
    unigram_columns = sorted(
        int(index) for term, index in vectorizer.vocabulary_.items() if " " not in term
    )
    bag = normalize(full[:, unigram_columns], norm="l2", axis=1, copy=True).tocsr()
    from src.models.optimization_v3_positional_text_pilot_v1 import build_position_block
    position, _ = build_position_block(documents, vectorizer, shuffled=False)
    baseline = np.asarray(fixture["test_baseline"], dtype=np.float64)
    design = sparse.hstack(
        (bag, position, sparse.csr_matrix((baseline / 100.0).reshape(-1, 1))),
        format="csr", dtype=np.float64,
    )
    design.sum_duplicates()
    design.sort_indices()
    if _sha_csr(design) != manifest["test_design_sha256"]:
        raise VerifyError("Synthetic test design differs under independent reconstruction")
    with threadpool_limits(limits=2):
        head = np.asarray(model.predict(design), dtype=np.float64)
    missing = np.asarray(fixture["test_missing"], dtype=bool)
    delta = np.clip(0.25 * head, -5.0, 5.0)
    delta[missing] = 0.0
    prediction = np.clip(baseline + delta, 0.0, 100.0)
    prediction[missing] = baseline[missing]
    arrays = _load_npz(bundle_path)
    if (not np.array_equal(arrays["record_ids"].astype(str), np.asarray(fixture["test_ids"], dtype=str))
            or not np.array_equal(arrays["baseline"], baseline)
            or not np.array_equal(arrays["missing"], missing)
            or not np.array_equal(arrays["head_prediction"], head)
            or not np.array_equal(arrays["delta"], delta)
            or not np.array_equal(arrays["prediction"], prediction)):
        raise VerifyError("Synthetic formula or persisted inference arrays differ")
    csv = pd.read_csv(csv_path, dtype={"record_id": "string"})
    if (csv.columns.tolist() != ["record_id", "threat_rank"]
            or not np.array_equal(csv.record_id.astype(str).to_numpy(), np.asarray(fixture["test_ids"], dtype=str))
            or not np.array_equal(csv.threat_rank.to_numpy(dtype=np.float64), prediction)):
        raise VerifyError("Synthetic submission numeric/identity round trip failed")
    events = [json.loads(line) for line in (run / "fit_ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    starts = [e["fit_id"] for e in events if e.get("event") == "FIT_STARTED"]
    saved = [e["fit_id"] for e in events if e.get("event") == "FIT_SAVED"]
    if starts != list(FIT_IDS) or saved != list(FIT_IDS):
        raise VerifyError("Synthetic ledger did not persist exactly two fit units")
    return {
        "status": "PASS_SYNTHETIC_INDEPENDENT_VERIFIER",
        "temp_only": True,
        "fit_invocations": 0,
        "fit_starts": len(starts),
        "fit_saved": len(saved),
        "test_design_rebuilt": True,
        "model_inference_recomputed": True,
        "formula_and_csv_roundtrip": True,
    }


def _duplicate_scan(worktree: Path, source: Path, ids: np.ndarray,
                    labels: np.ndarray, candidate: Path) -> dict[str, Any]:
    vector_hash = _sha_array(labels, "<f8")
    pair_hash = _canonical_sha(sorted(zip(ids.astype(str).tolist(), labels.astype(float).tolist())))
    scanned, compatible = 0, 0
    same_vectors, same_pairs = [], []
    for base in (worktree / "submissions", source / "submissions"):
        if not base.is_dir():
            continue
        for path in base.rglob("*.csv"):
            if path.resolve() == candidate.resolve():
                continue
            scanned += 1
            try:
                frame = pd.read_csv(path, dtype={"record_id": "string"})
            except Exception:
                continue
            if not {"record_id", "threat_rank"}.issubset(frame.columns):
                continue
            values = pd.to_numeric(frame.threat_rank, errors="coerce").to_numpy(dtype=np.float64)
            other_ids = frame.record_id.astype(str).to_numpy(dtype=str)
            if (len(values) != len(labels) or len(other_ids) != len(ids)
                    or not np.isfinite(values).all()):
                continue
            compatible += 1
            if _sha_array(values, "<f8") == vector_hash:
                same_vectors.append(str(path))
            if _canonical_sha(sorted(zip(other_ids.tolist(), values.tolist()))) == pair_hash:
                same_pairs.append(str(path))
    return {
        "files_scanned": scanned,
        "schema_compatible_files": compatible,
        "candidate_value_vector_sha256": vector_hash,
        "candidate_record_id_value_pairs_sha256": pair_hash,
        "same_ordered_value_vectors": sorted(same_vectors),
        "same_record_id_value_pairs": sorted(same_pairs),
    }


def _verify_official(
    source_root: Path, worktree_root: Path, output_root: Path, csv_arg: Path | None,
) -> dict[str, Any]:
    source, worktree, output = source_root.resolve(), worktree_root.resolve(), output_root.resolve()
    if output != (worktree / RUN_REL).resolve():
        raise VerifyError("Verifier output root differs from the isolated namespace")
    freeze_path = output / "trained_freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (_canonical_sha({k: v for k, v in freeze.items() if k != "freeze_signature"})
            != freeze.get("freeze_signature")
            or freeze.get("status") != "FROZEN_BEFORE_OFFICIAL_TEST_FEATURE_ACCESS"
            or freeze.get("test_features_accessed") is not False
            or freeze.get("sample_submission_accessed") is not False
            or freeze.get("accounting", {}).get("fit_starts") != 2):
        raise VerifyError("Training freeze signature or boundary failed")
    for relative, expected in freeze["input_hashes"].items():
        if _sha_file(source / relative) != expected:
            raise VerifyError(f"Frozen source input changed: {relative}")
    for relative, expected in freeze["worktree_source_hashes"].items():
        if _sha_file(worktree / relative) != expected:
            raise VerifyError(f"Frozen implementation changed: {relative}")
    vectorizer_path = output / "states/WORD_TFIDF_FULL_TRAIN.joblib"
    model_path = output / "states/REAL_POSITION_RESIDUAL_L1_E79.joblib"
    for path, expected in (
        (vectorizer_path, freeze["vectorizer"]["artifact_sha256"]),
        (model_path, freeze["head"]["artifact_sha256"]),
        (output / "receipts/WORD_TFIDF_FULL_TRAIN.json", freeze["vectorizer"]["receipt_file_sha256"]),
        (output / "receipts/REAL_POSITION_RESIDUAL_L1_E79.json", freeze["head"]["receipt_file_sha256"]),
    ):
        if _sha_file(path) != expected:
            raise VerifyError(f"Frozen learned state/receipt hash failed: {path.name}")
    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)
    if not isinstance(vectorizer, TfidfVectorizer) or not hasattr(vectorizer, "vocabulary_"):
        raise VerifyError("Frozen target-free TF-IDF state failed")
    actual_tfidf = _normal_tfidf(vectorizer.get_params(deep=False))
    if (actual_tfidf != freeze["vectorizer"]["sklearn_resolved_parameters"]
            or any(actual_tfidf.get(k) != v for k, v in freeze["vectorizer"]["parameters"].items())
            or vectorizer.norm != "l2"):
        raise VerifyError("Full-Train vectorizer parameters differ from trained freeze")
    model_params = {
        k: (v.item() if isinstance(v, np.generic) else v)
        for k, v in sorted(model.get_params(deep=False).items())
    }
    if model_params != freeze["head"]["parameters"] or not hasattr(model, "booster_"):
        raise VerifyError("Frozen L1 head parameters/state differ")
    for unit, section in zip(FIT_IDS, ("vectorizer", "head")):
        receipt_path = output / "receipts" / f"{unit}.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (_canonical_sha({k: v for k, v in receipt.items() if k != "receipt_sha256"})
                != receipt.get("receipt_sha256")
                or receipt.get("receipt_sha256") != freeze[section]["receipt_sha256"]
                or receipt.get("status") != "FIT_OUTPUT_DURABLE"
                or receipt.get("input_sha256") != freeze[section]["fit_input_sha256"]
                or receipt.get("artifact_sha256") != freeze[section]["artifact_sha256"]
                or receipt.get("output_audit") != freeze[section]["output_audit"]):
            raise VerifyError(f"Fit receipt failed for {unit}")
    ledger = [json.loads(line) for line in (output / "fit_ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    reservations = [event for event in ledger if event.get("event") == "FIT_BUDGET_RESERVED"]
    starts = [event.get("fit_id") for event in ledger if event.get("event") == "FIT_STARTED"]
    saved = [event.get("fit_id") for event in ledger if event.get("event") == "FIT_SAVED"]
    if (len(reservations) != 1 or reservations[0].get("fit_ids") != list(FIT_IDS)
            or starts != list(FIT_IDS) or saved != list(FIT_IDS)
            or any(event.get("event") == "FIT_FAILED" for event in ledger)):
        raise VerifyError("Independent fit-ledger audit did not find exactly two saved starts")

    # The R06 artifact is validated before reading the official Test feature table.
    baseline_manifest_path = source / "reports/submissions/optimization_v3_expanded_v1/R06_ORIGINAL/manifest.json"
    baseline_npz_path = source / "experiments/runs/optimization_v3_expanded_submissions_v1/R06_ORIGINAL/predictions.npz"
    baseline_csv_path = source / "submissions/optimization_v3_expanded_v1/r06_original.csv"
    baseline_manifest = json.loads(baseline_manifest_path.read_text(encoding="utf-8"))
    if (_sha_file(baseline_npz_path) != baseline_manifest["prediction_artifact"]["sha256"]
            or _sha_file(baseline_csv_path) != baseline_manifest["csv"]["sha256"]
            or baseline_manifest.get("status") != "PREPARED_VALIDATED_NOT_SUBMITTED"
            or baseline_manifest.get("candidate") != "R06_ORIGINAL"
            or baseline_manifest.get("test_rows") != TEST_ROWS
            or baseline_manifest.get("sample_submission_rows") != TEST_ROWS
            or baseline_manifest.get("official_input_hashes", {}).get("test") != TEST_SHA256
            or baseline_manifest.get("official_input_hashes", {}).get("sample") != SAMPLE_SHA256
            or baseline_manifest.get("prediction_artifact", {}).get("path_gitignored")
            != "experiments/runs/optimization_v3_expanded_submissions_v1/R06_ORIGINAL/predictions.npz"
            or baseline_manifest.get("csv", {}).get("path_gitignored")
            != "submissions/optimization_v3_expanded_v1/r06_original.csv"
            or baseline_manifest.get("csv", {}).get("rows") != TEST_ROWS
            or baseline_manifest.get("csv", {}).get("columns") != ["record_id", "threat_rank"]):
        raise VerifyError("Manifest-bound R06 Test baseline hash failed")
    with np.load(baseline_npz_path, allow_pickle=False) as archive:
        baseline = np.asarray(archive["prediction"], dtype=np.float64).copy()
        baseline_cycles = np.asarray(archive["cycle_code"], dtype=np.int64).copy()
    baseline_csv = pd.read_csv(baseline_csv_path, dtype={"record_id": "string"})
    baseline_ids = baseline_csv.record_id.astype(str).to_numpy(dtype=str)
    if (baseline_csv.columns.tolist() != ["record_id", "threat_rank"]
            or len(baseline_ids) != TEST_ROWS or len(set(baseline_ids.tolist())) != TEST_ROWS
            or len(baseline) != TEST_ROWS
            or baseline_manifest.get("prediction_audit", {}).get("prediction_sha256")
            != _sha_array(baseline, "<f8")
            or np.max(np.abs(baseline - baseline_csv.threat_rank.to_numpy(dtype=np.float64))) > 1e-12):
        raise VerifyError("R06 Test baseline values/IDs failed")

    test_path, sample_path = source / TEST_REL, source / SAMPLE_REL
    if _sha_file(test_path) != TEST_SHA256 or _sha_file(sample_path) != SAMPLE_SHA256:
        raise VerifyError("Official Test/Sample file hash failed")
    test = pd.read_csv(test_path, usecols=list(TEST_COLUMNS), dtype={"record_id": "string"})
    test = test.loc[:, list(TEST_COLUMNS)]
    sample = pd.read_csv(sample_path, usecols=["record_id"], dtype={"record_id": "string"})
    ids = test.record_id.astype(str).to_numpy(dtype=str)
    cycle_values = pd.to_numeric(test.cycle_code, errors="coerce").to_numpy()
    if (len(ids) != TEST_ROWS or len(set(ids.tolist())) != TEST_ROWS
            or not np.isfinite(cycle_values).all() or np.any(cycle_values != np.floor(cycle_values))
            or not np.array_equal(ids, sample.record_id.astype(str).to_numpy(dtype=str))
            or not np.array_equal(ids, baseline_ids)
            or not np.array_equal(cycle_values.astype(np.int64), baseline_cycles)):
        raise VerifyError("Test/Sample/R06 record identity or order failed")
    cycles = cycle_values.astype(np.int64)

    from src.models.classification_models import _text_values
    from src.models.optimization_v3_positional_text_pilot_v1 import build_position_block
    documents, missing = _text_values(test.signal_stream)
    full = vectorizer.transform(documents).tocsr()
    unigram_cols = sorted(int(index) for term, index in vectorizer.vocabulary_.items() if " " not in term)
    bag = normalize(full[:, unigram_cols], norm="l2", axis=1, copy=True).tocsr()
    position, _ = build_position_block(documents, vectorizer, shuffled=False)
    design = sparse.hstack(
        (bag, position, sparse.csr_matrix((baseline / 100.0).reshape(-1, 1))),
        format="csr", dtype=np.float64,
    )
    design.sum_duplicates()
    design.sort_indices()
    with threadpool_limits(limits=2):
        head = np.asarray(model.predict(design), dtype=np.float64)
    delta = np.clip(0.25 * head, -5.0, 5.0)
    delta[np.asarray(missing, dtype=bool)] = 0.0
    prediction = np.clip(baseline + delta, 0.0, 100.0)
    prediction[np.asarray(missing, dtype=bool)] = baseline[np.asarray(missing, dtype=bool)]

    infer_dir = output / "inference"
    bundle_path = infer_dir / "predictions.npz"
    manifest_path = infer_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_hash = manifest.pop("manifest_sha256", None)
    if _canonical_sha(manifest) != manifest_hash:
        raise VerifyError("Inference manifest canonical hash failed")
    csv_path = csv_arg.resolve() if csv_arg else (worktree / manifest["submission"]["path"]).resolve()
    if worktree not in csv_path.parents:
        raise VerifyError("Candidate CSV escaped the worktree")
    if (_sha_file(bundle_path) != manifest["prediction_artifact"]["sha256"]
            or _sha_file(csv_path) != manifest["submission"]["sha256"]
            or _sha_csr(design) != manifest["feature_hashes"]["design_csr_sha256"]):
        raise VerifyError("Inference bundle, CSV or independent feature hash failed")
    with np.load(bundle_path, allow_pickle=False) as archive:
        checks = {
            "record_ids": np.array_equal(archive["record_ids"].astype(str), ids),
            "cycle_code": np.array_equal(archive["cycle_code"], cycles),
            "baseline": np.array_equal(archive["baseline"], baseline),
            "missing_mask": np.array_equal(archive["missing"], np.asarray(missing, dtype=bool)),
            "head": np.array_equal(archive["head_prediction"], head),
            "delta": np.array_equal(archive["delta"], delta),
            "prediction": np.array_equal(archive["prediction"], prediction),
        }
    if not all(checks.values()):
        raise VerifyError(f"Independent Test inference differed from saved bundle: {checks}")
    submission = pd.read_csv(csv_path, dtype={"record_id": "string"})
    labels = pd.to_numeric(submission.threat_rank, errors="coerce").to_numpy(dtype=np.float64)
    if (submission.columns.tolist() != ["record_id", "threat_rank"]
            or len(submission) != TEST_ROWS
            or not np.array_equal(submission.record_id.astype(str).to_numpy(dtype=str), ids)
            or not np.isfinite(labels).all()
            or np.any((labels < 0.0) | (labels > 100.0))
            or not np.array_equal(labels, prediction)):
        raise VerifyError("Submission ID/order, numeric serialization or range failed")
    duplicates = _duplicate_scan(worktree, source, ids, labels, csv_path)
    if duplicates != manifest["duplicate_label_vector_scan"]:
        raise VerifyError("Independent duplicate label-vector scan differs from manifest")
    return {
        "status": "PASS_INDEPENDENT_INFERENCE_VERIFICATION",
        "test_rows": len(ids), "unique_test_ids": len(set(ids.tolist())),
        "test_order_matches_sample_and_r06": True,
        "source_and_state_hashes": True,
        "design_sha256": _sha_csr(design),
        "model_predictions_recomputed": True,
        "policy_formula_recomputed": True,
        "numeric_csv_roundtrip_exact": True,
        "prediction_range": [float(labels.min()), float(labels.max())],
        "missing_text_rows": int(np.sum(missing)),
        "exact_r06_missing_fallback": bool(np.array_equal(prediction[missing], baseline[missing])),
        "duplicate_vector_scan": duplicates,
        "fit_starts": 0,
        "test_labels_read": False,
        "kaggle_accessed": False,
    }


def _persist_report(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(_jsonable(value), sort_keys=True, ensure_ascii=False,
                         indent=2, allow_nan=False).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise VerifyError(f"Refusing to overwrite verification report: {path}")
        return
    staging = path.with_name(path.name + ".staging")
    if staging.exists():
        raise VerifyError(f"Stale verification staging file: {staging}")
    with staging.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(staging, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-root", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--worktree-root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--csv-path", type=Path)
    parser.add_argument("--report-path", type=Path)
    args = parser.parse_args()
    try:
        if args.synthetic_root:
            result = _verify_synthetic(args.synthetic_root)
        else:
            if args.source_root is None or args.output_root is None:
                raise VerifyError("Official verifier requires source and output roots")
            result = _verify_official(
                args.source_root, args.worktree_root, args.output_root, args.csv_path,
            )
            report_path = args.report_path or (
                args.worktree_root.resolve() / REPORT_REL / "INDEPENDENT_VERIFICATION.json"
            )
            _persist_report(report_path.resolve(), result)
        print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False))
        return 0
    except Exception as exc:
        print(json.dumps({
            "status": "FAIL_CLOSED", "error_type": type(exc).__name__, "error": str(exc),
        }, sort_keys=True, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
