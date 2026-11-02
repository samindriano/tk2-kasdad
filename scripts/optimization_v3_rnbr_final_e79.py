from __future__ import annotations

"""One-slot, frozen E79 R-NBR observed-memory Test benchmark preparation."""

import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
import psutil
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import optimization_v3_rnbr_official as official  # noqa: E402
from scripts import optimization_v3_rnbr_continuation as prior  # noqa: E402
from src.models.classification_a02_neighbors import _documents  # noqa: E402
from src.models.regression_neighbor_rnbr import (  # noqa: E402
    EXPECTED_SOURCE_CUTOFFS, MAX_NEIGHBORS, MIN_NEIGHBORS,
    correction_delta, weighted_median_residual,
)

_LIMITER = threadpool_limits(limits=2)
TRAIN = ROOT / "data/raw/regression/Train.csv"
TEST = ROOT / "data/raw/regression/Test.csv"
CLASS_TEST = ROOT / "data/raw/classification/Test.csv"
SAMPLE = ROOT / "data/raw/regression/SampleSubmission.csv"
PROTOCOL = ROOT / "experiments/optimization_v3/rnbr_final_e79_v1/protocol.json"
FREEZE = ROOT / "experiments/optimization_v3/rnbr_final_e79_v1/freeze.json"
REPORT = ROOT / "reports/optimization_v3/rnbr_final_e79_v1"
PREFLIGHT = REPORT / "preflight.json"
SYNTHETIC = REPORT / "synthetic_qa.json"
FINAL_REPORT = REPORT / "FINAL_REPORT.md"
LEDGER = ROOT / "experiments/runs/optimization_v3_research_next_v1/budget.jsonl"
RUN = ROOT / "experiments/runs/optimization_v3_research_next_v1/RNBR_FINAL_E79_OBSERVED"
VECTORIZER = RUN / "a02_e75_vectorizer.joblib"
VECTORIZER_RECEIPT = RUN / "a02_e75_vectorizer_receipt.json"
BANK_MATRIX = RUN / "donor_tfidf_csr.npz"
BUNDLE = RUN / "e79_bundle.npz"
FIT_RECEIPT = RUN / "fit_receipt.json"
MANIFEST = ROOT / "reports/submissions/optimization_v3_rnbr_final_e79_v1/manifest.json"
CSV = ROOT / "submissions/optimization_v3_rnbr_final_e79_v1/samuel_rnbr_final_e79.csv"

A02_MODEL = ROOT / "experiments/runs/optimization_v3_a02_neighbors_v1/E75/observed/model.joblib"
A02_RECEIPT = ROOT / "experiments/runs/optimization_v3_a02_neighbors_v1/E75/observed/fit_receipt.json"
A02_STATE = ROOT / "experiments/runs/optimization_v3_a02_neighbors_v1/E75/observed/fit_state.json"
A02_FREEZE = ROOT / "experiments/optimization_v3/c19_a02_v2/A02/freeze.json"
A02_PROTOCOL = ROOT / "experiments/optimization_v3/c19_a02_v2/A02/protocol.json"
R06_MANIFEST = ROOT / "reports/submissions/optimization_v3_expanded_v1/R06_ORIGINAL/manifest.json"
R06_PREDICTIONS = ROOT / "experiments/runs/optimization_v3_expanded_submissions_v1/R06_ORIGINAL/predictions.npz"
R06_CSV = ROOT / "submissions/optimization_v3_expanded_v1/r06_original.csv"
PILOT_FREEZE = ROOT / "experiments/optimization_v3/rnbr_pilot_v1/official_freeze.json"

FIT_ID = "RNBR_FINAL_E79_OBSERVED"
CARD_ID = "REGRESSION_RNBR_FINAL_E79"
STAGE = "ORIGIN79_OBSERVED_FULLTEST"
AMENDMENT_ID = "RNBR_FINAL_E79_CAP77_2026_11_03"
RESERVATION_KEY = f"{CARD_ID}:{STAGE}"
THREADS = 2
RSS_CAP = 6 * 1024**3
FREE_RAM_MIN = 2 * 1024**3
QUERY_BLOCK = 128
PEAK_RSS_BYTES = 0
EXPECTED = {
    "data/raw/regression/Train.csv": "756523ff8156a9cd776e52249f7e2a8f23dcec432b8d10570e55ff6db0e0f082",
    "data/raw/regression/Test.csv": "4e6c381d582ed24f92d17f620f4f0ad25e2e0def29eb57a91a5482867f6afe3c",
    "data/raw/classification/Test.csv": "4e6c381d582ed24f92d17f620f4f0ad25e2e0def29eb57a91a5482867f6afe3c",
    "data/raw/regression/SampleSubmission.csv": "63a51f4b7dfbb0eec9800099407abd8be14d25ec18a533cd39dc854ffc190a32",
    "reports/submissions/optimization_v3_expanded_v1/R06_ORIGINAL/manifest.json": "3f77e4ac148ada5fb79e5e2c16e7975e77726f4bfcc42a29f9234f801c697d67",
    "experiments/runs/optimization_v3_expanded_submissions_v1/R06_ORIGINAL/predictions.npz": "ce034e82a7063ec5dd873b1126d60731dd291d3852c57936c7aa1ee449a81eba",
    "submissions/optimization_v3_expanded_v1/r06_original.csv": "4a72b13e0d6ac93e18adee23010bb15ddf15fcb2a9543dc903154d5af60ee64f",
    "experiments/runs/optimization_v3_a02_neighbors_v1/E75/observed/model.joblib": "5a8d6f5effa449c468b3ec53415a52430b1ceac3445f7703ac86add76bb13e32",
    "experiments/runs/optimization_v3_a02_neighbors_v1/E75/observed/fit_receipt.json": "72cba8fff8ca3c5e8f43d054b062e31fa89b0f24f897cd5effaec817725c611e",
    "experiments/runs/optimization_v3_a02_neighbors_v1/E75/observed/fit_state.json": "ba9f7bf4f984fab20e990963004a2348caf6f06f71ce76fc9c420f7e14e33998",
    "experiments/optimization_v3/c19_a02_v2/A02/freeze.json": "frozen-signature-checked",
    "experiments/optimization_v3/c19_a02_v2/A02/protocol.json": "protocol-signature-checked",
    "experiments/optimization_v3/rnbr_pilot_v1/official_freeze.json": "source-freeze-checked",
}
CODE_PATHS = (
    "scripts/optimization_v3_rnbr_final_e79.py",
    "scripts/optimization_v3_rnbr_final_e79_verify.py",
    "scripts/optimization_v3_rnbr_official.py",
    "scripts/optimization_v3_rnbr_continuation.py",
    "src/models/regression_neighbor_rnbr.py",
    "src/models/classification_a02_neighbors.py",
    "src/models/classification_models.py",
    "src/models/research_next_budget.py",
)
OOF_CUTOFFS = (55, 59, 63, 67, 71, 75)


class E79Error(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha_array(value: Any, dtype: str | None = None) -> str:
    array = np.asarray(value, dtype=dtype)
    if array.dtype.kind in "OUS":
        h = hashlib.sha256()
        for item in array.reshape(-1):
            data = str(item).encode("utf-8", errors="surrogatepass")
            h.update(len(data).to_bytes(8, "little"))
            h.update(data)
        return h.hexdigest()
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    rendered = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise E79Error(f"Refusing to overwrite immutable artifact: {path}")
        return
    tmp = path.with_suffix(path.suffix + ".staging")
    with tmp.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(rendered)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def _read_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise E79Error(f"Expected JSON object: {path}")
    return obj


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _targetfree_test() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cols = ["record_id", "cycle_code", "signal_stream"]
    reg = pd.read_csv(TEST, usecols=cols, dtype={"record_id": str})
    cls = pd.read_csv(CLASS_TEST, usecols=cols, dtype={"record_id": str})
    sample = pd.read_csv(SAMPLE, usecols=["record_id"], dtype={"record_id": str})
    for label, frame in (("Regression Test", reg), ("Classification Test", cls), ("SampleSubmission", sample)):
        ids = frame["record_id"].astype(str)
        if ids.isna().any() or ids.duplicated().any():
            raise E79Error(f"{label} IDs are missing or duplicated")
    reg_ids = reg["record_id"].astype(str).to_numpy()
    cls_ids = cls["record_id"].astype(str).to_numpy()
    cls_lookup = {value: index for index, value in enumerate(cls_ids)}
    if set(cls_lookup) != set(reg_ids) or len(cls_lookup) != len(reg_ids):
        raise E79Error("Regression and Classification official Test ID sets differ")
    cls_at_reg_order = np.asarray([cls_lookup[value] for value in reg_ids], dtype=np.int64)
    if not np.array_equal(reg["cycle_code"].to_numpy(), cls["cycle_code"].to_numpy()[cls_at_reg_order]):
        raise E79Error("Shared Test cycle_code values do not align one-to-one by record_id")
    cls_stream = cls["signal_stream"].astype("string").fillna("").str.strip().to_numpy(dtype=str)
    reg_stream = reg["signal_stream"].astype("string").fillna("").str.strip().to_numpy(dtype=str)
    if not np.array_equal(reg_stream, cls_stream[cls_at_reg_order]):
        raise E79Error("Shared Test signal_stream values do not align one-to-one by record_id")
    sample_ids = sample["record_id"].astype(str).to_numpy()
    if not np.array_equal(reg_ids, sample_ids):
        raise E79Error("Official Test order does not equal SampleSubmission ID order")
    if len(reg) != 29_421 or not set(np.unique(reg["cycle_code"])) <= set(range(80, 92)):
        raise E79Error("Test row count/cycle range differs from the official horizon")
    return reg, cls, sample


def _text_a02_state(*, extract: bool) -> tuple[dict[str, Any], TfidfVectorizer]:
    a02_freeze = _read_json(A02_FREEZE)
    a02_protocol = _read_json(A02_PROTOCOL)
    if a02_freeze.get("task") != "A02" or a02_freeze.get("protocol_sha256") != digest_bytes(canonical_bytes(a02_protocol)):
        raise E79Error("A02 protocol/freeze identity failed")
    identity = a02_freeze.get("identity")
    if not isinstance(identity, dict) or a02_freeze.get("freeze_signature") != digest_bytes(canonical_bytes(identity)):
        raise E79Error("A02 freeze signature failed")
    if a02_freeze.get("freeze_signature") != "cdac164df87d4f559e36795066cc0c3a91034e5917b0b1fd72cebd1051dae08b":
        raise E79Error("A02 E75 vectorizer does not match the reviewed frozen signature")
    for rel, expected in a02_freeze.get("source_sha256", {}).items():
        if sha_file(ROOT / rel) != expected:
            raise E79Error(f"A02 frozen source changed: {rel}")
    if a02_freeze.get("input_sha256", {}).get("data/raw/classification/Train.csv") != "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c":
        raise E79Error("A02 vectorizer was not frozen against the official Classification Train")
    receipt = _read_json(A02_RECEIPT)
    receipt_body = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    if (receipt.get("fit_id") != "A02_E75_OBSERVED" or receipt.get("origin_cycle") != 75
            or receipt.get("text_prefix_rows") != 42_117
            or receipt.get("preprocessing_fit_population") != "official Train prefix cycles 47-75; text only"
            or receipt.get("query_labels_passed_to_fit") is not False
            or receipt.get("test_data_used") is not False
            or receipt.get("receipt_sha256") != digest_bytes(canonical_bytes(receipt_body))):
        raise E79Error("A02 E75 receipt does not attest the target-free Train-only vectorizer fit")
    if sha_file(A02_MODEL) != EXPECTED["experiments/runs/optimization_v3_a02_neighbors_v1/E75/observed/model.joblib"]:
        raise E79Error("A02 E75 model hash differs from the audited artifact")
    model = joblib.load(A02_MODEL)
    text_space = getattr(model, "text_space", None)
    vectorizer = getattr(text_space, "vectorizer", None)
    if not isinstance(vectorizer, TfidfVectorizer):
        raise E79Error("A02 E75 model has no reusable fitted TF-IDF vectorizer")
    audit = text_space.audit_state()
    if (audit.get("origin_cycle") != 75 or audit.get("fit_rows") != 42_117
            or audit.get("fit_row_positions_sha256") != "e733031bb45eac2f1f55b94e45bc7161c1e80801f85564a25a289758100e2bc9"
            or audit.get("fit_documents_sha256") != "e4537b03354f313cdd6440f86555530e18617fc89cda2e8509cea0aeffb0d554"
            or audit.get("vocabulary_sha256") != "2603bf8ba90f509ccc81a4155159e7683b0a3660d7c2c0602ba49ee291266083"
            or audit.get("idf_sha256") != "68fc4caea39a15a0ce7e79869ddab4ae6c239b4ae86ba10c3c01dfa1770980b5"):
        raise E79Error("A02 E75 reusable vectorizer state differs from its audited receipt")
    if extract:
        RUN.mkdir(parents=True, exist_ok=True)
        if VECTORIZER.exists():
            existing = joblib.load(VECTORIZER)
            if not isinstance(existing, TfidfVectorizer) or sha_file(VECTORIZER) != _read_json(VECTORIZER_RECEIPT).get("vectorizer_sha256"):
                raise E79Error("Existing extracted vectorizer is not receipt-valid")
            if existing.vocabulary_ != vectorizer.vocabulary_ or not np.array_equal(existing.idf_, vectorizer.idf_):
                raise E79Error("Existing extracted vectorizer differs from A02 E75 fitted state")
            vectorizer = existing
        else:
            tmp = VECTORIZER.with_suffix(".joblib.staging")
            joblib.dump(vectorizer, tmp, compress=0)
            os.replace(tmp, VECTORIZER)
            receipt_value = {
                "status": "PASS", "source_model_sha256": sha_file(A02_MODEL),
                "source_fit_receipt_sha256": sha_file(A02_RECEIPT),
                "source_fit_state_sha256": sha_file(A02_STATE),
                "only_extracted_attribute": "model.text_space.vectorizer",
                "source_text_space_audit": audit,
                "vectorizer_sha256": sha_file(VECTORIZER),
                "vocabulary_sha256": audit["vocabulary_sha256"], "idf_sha256": audit["idf_sha256"],
                "no_new_fit": True,
            }
            _write_once(VECTORIZER_RECEIPT, receipt_value)
    return audit, vectorizer


def _runtime_preflight(*, extract: bool) -> dict[str, Any]:
    for relative, expected in EXPECTED.items():
        if expected.endswith("checked"):
            continue
        if sha_file(ROOT / relative) != expected:
            raise E79Error(f"Frozen input hash changed: {relative}")
    targetfree = official.load_targetfree_train()
    if (len(targetfree["rows"]) != 50_608 or not np.array_equal(np.unique(targetfree["cycles"]), np.arange(47, 80))):
        raise E79Error("Official Regression Train target-free identity failed")
    donor_rows = np.flatnonzero((targetfree["cycles"] >= 56) & (targetfree["cycles"] <= 79) & targetfree["present"]).astype(np.int64)
    if len(np.unique(targetfree["ids"][donor_rows])) != len(donor_rows):
        raise E79Error("E79 donor IDs are not unique")
    reg_test, cls_test, sample = _targetfree_test()
    pilot_freeze = _read_json(PILOT_FREEZE)
    pilot_body = {k: v for k, v in pilot_freeze.items() if k != "freeze_signature"}
    if pilot_freeze.get("freeze_signature") != digest_bytes(canonical_bytes(pilot_body)):
        raise E79Error("Frozen R06 OOF source freeze signature failed")
    _, r06_train_caches = _load_r06_inputs(pilot_freeze, targetfree)
    r06_test, r06_csv_identity = _load_r06_test(reg_test)
    audit, vectorizer = _text_a02_state(extract=extract)
    if extract:
        if sha_file(VECTORIZER) != _read_json(VECTORIZER_RECEIPT).get("vectorizer_sha256"):
            raise E79Error("Extracted target-free A02 vectorizer hash failed")
    vm = psutil.virtual_memory()
    disk = shutil.disk_usage(RUN.parent)
    estimate = 3 * 1024**3
    if vm.available < FREE_RAM_MIN:
        raise E79Error(f"Available RAM {vm.available} is below the 2 GiB reserve")
    if estimate > RSS_CAP:
        raise E79Error("Conservative task RAM estimate exceeds 6 GiB")
    if disk.free < 2 * 1024**3:
        raise E79Error("Less than 2 GiB disk space remains for durable artifacts")
    competing = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        if process.info["pid"] == os.getpid():
            continue
        process_name = str(process.info.get("name") or "").lower()
        command = " ".join(process.info.get("cmdline") or []).lower()
        if process_name.startswith("python") and ("optimization_v3_rnbr_final_e79" in command or "optimization_v3_rnbr_continuation" in command):
            competing.append({"pid": process.info["pid"], "name": process.info.get("name")})
    if competing:
        raise E79Error(f"Competing R-NBR task process detected: {competing}")
    return {
        "status": "PASS", "target_columns_loaded": False,
        "test_labels_loaded": False, "classification_labels_or_probabilities_read": False,
        "train_rows": len(targetfree["rows"]), "train_cycle_range": [47, 79],
        "donor_rows": len(donor_rows), "donor_present_stream_rows": len(donor_rows),
        "donor_cycles": [56, 79], "test_rows": len(reg_test),
        "test_cycles": [int(reg_test.cycle_code.min()), int(reg_test.cycle_code.max())],
        "test_ids_and_shared_features_aligned_by_explicit_one_to_one_id_map": True,
        "sample_schema": list(pd.read_csv(SAMPLE, nrows=0).columns),
        "r06_test_manifest": r06_csv_identity,
        "r06_oof_cutoffs_loaded": list(sorted(r06_train_caches)),
        "a02_text_space_audit": audit,
        "vectorizer_extracted_hash": sha_file(VECTORIZER) if extract else None,
        "resource_preflight": {"available_ram_bytes": vm.available, "free_disk_bytes": disk.free,
                               "estimated_peak_bytes": estimate, "job_rss_cap_bytes": RSS_CAP,
                               "reserve_bytes": FREE_RAM_MIN, "threads": THREADS, "concurrent_jobs": 1},
        "preparation_code_commit": _git("rev-parse", "HEAD"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _load_r06_inputs(pilot_freeze: Mapping[str, Any], train: Mapping[str, np.ndarray]) -> tuple[dict[int, dict[str, np.ndarray]], dict[int, dict[str, np.ndarray]]]:
    body = {k: v for k, v in pilot_freeze.items() if k != "freeze_signature"}
    if pilot_freeze.get("freeze_signature") != digest_bytes(canonical_bytes(body)):
        raise E79Error("Original R06 OOF source freeze is missing/invalid")
    r06 = official.load_r06_predictions(train, pilot_freeze)
    if not set(OOF_CUTOFFS).issubset(r06):
        raise E79Error("R06 OOF caches do not contain all six required donor cutoffs")
    aligned = {}
    for cutoff in OOF_CUTOFFS:
        item = r06[cutoff]
        if not np.isfinite(item["prediction"]).all() or not np.array_equal(item["ids"], train["ids"][item["rows"]]):
            raise E79Error(f"R06 OOF values/IDs invalid at cutoff {cutoff}")
        aligned[cutoff] = item
    return r06, aligned


def _load_r06_test(reg_test: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    manifest = _read_json(R06_MANIFEST)
    if (manifest.get("status") != "PREPARED_VALIDATED_NOT_SUBMITTED"
            or manifest.get("candidate") != "R06_ORIGINAL"
            or manifest.get("official_input_hashes", {}).get("test") != EXPECTED["data/raw/regression/Test.csv"]
            or manifest.get("official_input_hashes", {}).get("sample") != EXPECTED["data/raw/regression/SampleSubmission.csv"]
            or manifest.get("prediction_artifact", {}).get("sha256") != sha_file(R06_PREDICTIONS)
            or manifest.get("csv", {}).get("sha256") != sha_file(R06_CSV)
            or manifest.get("csv", {}).get("rows") != 29_421
            or manifest.get("csv", {}).get("columns") != ["record_id", "threat_rank"]):
        raise E79Error("R06_ORIGINAL Test artifact is not manifest-verified")
    sample_ids = reg_test["record_id"].astype(str).to_numpy()
    csv_frame = pd.read_csv(R06_CSV, dtype={"record_id": str})
    if (csv_frame.columns.tolist() != ["record_id", "threat_rank"]
            or not np.array_equal(csv_frame.record_id.astype(str).to_numpy(), sample_ids)):
        raise E79Error("R06 manifest-bound CSV row identity differs from official Test")
    with np.load(R06_PREDICTIONS, allow_pickle=False) as archive:
        required = {"prediction", "cycle_code"}
        if not required <= set(archive.files):
            raise E79Error("R06 full-Train prediction artifact lacks Test prediction/cycle arrays")
        baseline = np.asarray(archive["prediction"], dtype=np.float64).copy()
        cycles = np.asarray(archive["cycle_code"], dtype=np.int64).copy()
    prediction_sha = sha_array(baseline, "<f8")
    cycle_sha = sha_array(cycles, "<i8")
    if (len(baseline) != len(reg_test) or not np.isfinite(baseline).all()
            or np.any((baseline < 0) | (baseline > 100))
            or not np.array_equal(cycles, reg_test.cycle_code.to_numpy(dtype=np.int64))
            or prediction_sha != manifest.get("prediction_audit", {}).get("prediction_sha256")
            or np.max(np.abs(baseline - csv_frame.threat_rank.to_numpy(dtype=np.float64))) > 1e-12):
        raise E79Error("R06 Test prediction/cycle lineage or range verification failed")
    return baseline, {"manifest_sha256": sha_file(R06_MANIFEST), "npz_sha256": sha_file(R06_PREDICTIONS),
                      "csv_sha256": sha_file(R06_CSV), "prediction_sha256": prediction_sha,
                      "cycle_code_sha256": cycle_sha, "rows": len(baseline),
                      "test_ids_bound_through_manifest_csv_order": True}


def _array_digest(value: Any, dtype: str | None = None) -> str:
    return sha_array(value, dtype)


def _csr_digest(matrix: sparse.csr_matrix) -> dict[str, Any]:
    return {"shape": [int(x) for x in matrix.shape], "nnz": int(matrix.nnz),
            "data_sha256": sha_array(matrix.data, "<f8"),
            "indices_sha256": sha_array(matrix.indices, "<i4"),
            "indptr_sha256": sha_array(matrix.indptr, "<i4")}


def _save_sparse_once(path: Path, matrix: sparse.csr_matrix) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = sparse.load_npz(path).tocsr()
        if _csr_digest(existing) != _csr_digest(matrix):
            raise E79Error(f"Existing sparse bank differs; refusing overwrite: {path}")
        return
    temp = path.with_suffix(".npz.staging")
    with temp.open("xb") as stream:
        sparse.save_npz(stream, matrix, compressed=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _save_npz_once(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as existing:
            if set(existing.files) != set(arrays):
                raise E79Error(f"Existing E79 bundle schema differs: {path}")
            for name, value in arrays.items():
                if not np.array_equal(existing[name], np.asarray(value)):
                    raise E79Error(f"Existing E79 bundle array differs: {name}")
        return
    temp = path.with_suffix(".npz.staging")
    with temp.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _top32(query: sparse.csr_matrix, bank: sparse.csr_matrix, bank_rows: np.ndarray,
           missing: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = query.shape[0]
    neighbors = np.full((n, MAX_NEIGHBORS), -1, dtype=np.int64)
    similarities_out = np.zeros((n, MAX_NEIGHBORS), dtype=np.float64)
    counts = np.zeros(n, dtype=np.int16)
    ids_lookup = None
    for start in range(0, n, QUERY_BLOCK):
        _check_resources()
        stop = min(start + QUERY_BLOCK, n)
        block = (query[start:stop] @ bank.T).toarray()
        for local_i, i in enumerate(range(start, stop)):
            if missing[i]:
                continue
            similarities = block[local_i]
            positive = np.flatnonzero(similarities > 0.0)
            if not len(positive):
                continue
            take = min(MAX_NEIGHBORS, len(positive))
            if len(positive) > take:
                vals = similarities[positive]
                threshold = np.partition(vals, len(vals) - take)[len(vals) - take]
                above = positive[vals > threshold]
                ties = positive[vals == threshold]
                remaining = take - len(above)
                chosen = np.concatenate((above, ties[np.argsort(bank_rows[ties], kind="stable")[:remaining]]))
            else:
                chosen = positive
            chosen = chosen[np.lexsort((bank_rows[chosen], -similarities[chosen]))]
            k = len(chosen)
            neighbors[i, :k] = bank_rows[chosen]
            similarities_out[i, :k] = similarities[chosen]
            counts[i] = k
    return neighbors, similarities_out, counts, missing.copy()


def _check_resources() -> None:
    global PEAK_RSS_BYTES
    vm = psutil.virtual_memory()
    if vm.available < FREE_RAM_MIN:
        raise E79Error(f"Available RAM fell below 2 GiB reserve: {vm.available}")
    process = psutil.Process()
    current = process.memory_info().rss
    PEAK_RSS_BYTES = max(PEAK_RSS_BYTES, current)
    if current > RSS_CAP:
        raise E79Error(f"E79 process RSS exceeded 6 GiB: {current}")


def _load_donor_targets(donor_rows: np.ndarray, targetfree: Mapping[str, np.ndarray]) -> np.ndarray:
    """Read target cells only for present-stream Train rows in cycles 56-79."""
    row_set = set(map(int, donor_rows))
    if len(row_set) != len(donor_rows):
        raise E79Error("Donor rows are not unique before target access")
    values: list[float] = []
    seen_rows: list[int] = []
    with TRAIN.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        indices = {name: header.index(name) for name in ("record_id", "cycle_code", "signal_stream", "threat_rank")}
        if len(header) != len(set(header)):
            raise E79Error("Official Train contains duplicate column names")
        for row_pos, row in enumerate(reader):
            if row_pos not in row_set:
                continue
            if row_pos >= len(targetfree["rows"]) or row[indices["record_id"]] != targetfree["ids"][row_pos]:
                raise E79Error("Donor target read encountered a row/ID alignment error")
            cycle = int(row[indices["cycle_code"]])
            stream_text = row[indices["signal_stream"]].strip()
            if cycle != int(targetfree["cycles"][row_pos]) or not stream_text or cycle < 56 or cycle > 79:
                raise E79Error("Donor target row violates E79 present-stream/time boundary")
            value = float(row[indices["threat_rank"]])
            if not np.isfinite(value) or not 0.0 <= value <= 100.0:
                raise E79Error("Eligible Train donor target is not finite in [0,100]")
            seen_rows.append(row_pos)
            values.append(value)
    if seen_rows != donor_rows.tolist():
        raise E79Error("Not every frozen eligible donor target was read exactly once")
    return np.asarray(values, dtype=np.float64)


def _donor_oof(donor_rows: np.ndarray, cycles: np.ndarray, ids: np.ndarray,
               caches: Mapping[int, Mapping[str, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cutoffs = np.asarray([EXPECTED_SOURCE_CUTOFFS[int(c)] for c in cycles], dtype=np.int64)
    predictions = np.empty(len(donor_rows), dtype=np.float64)
    for cutoff in np.unique(cutoffs):
        ix = np.flatnonzero(cutoffs == cutoff)
        source = caches[int(cutoff)]
        lookup = {int(row): j for j, row in enumerate(source["rows"])}
        for i in ix:
            j = lookup.get(int(donor_rows[i]))
            if j is None or str(source["ids"][j]) != str(ids[i]) or int(source["cycles"][j]) != int(cycles[i]):
                raise E79Error(f"R06 OOF donor source mismatch at cutoff {cutoff}, row {donor_rows[i]}")
            predictions[i] = float(source["prediction"][j])
    if not np.isfinite(predictions).all() or np.any((cutoffs >= cycles) | (predictions < 0) | (predictions > 100)):
        raise E79Error("R06 donor OOF source cutoff/value failed")
    return predictions, cutoffs, ids.copy()


def _predict(baseline: np.ndarray, missing: np.ndarray, rows: np.ndarray, similarities: np.ndarray,
             counts: np.ndarray, donor_rows: np.ndarray, donor_residuals: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lookup = {int(row): i for i, row in enumerate(donor_rows)}
    delta = np.zeros(len(baseline), dtype=np.float64)
    n_used = np.zeros(len(baseline), dtype=np.int16)
    medians = np.zeros(len(baseline), dtype=np.float64)
    for i, count in enumerate(counts):
        k = int(count)
        if missing[i] or k < MIN_NEIGHBORS:
            continue
        neighbor_rows = rows[i, :k]
        indices = np.asarray([lookup[int(row)] for row in neighbor_rows], dtype=np.int64)
        medians[i] = weighted_median_residual(donor_residuals[indices], similarities[i, :k], neighbor_rows)
        delta[i] = correction_delta(medians[i])
        n_used[i] = k
    predictions = np.clip(baseline + delta, 0.0, 100.0)
    fallback = missing | (counts < MIN_NEIGHBORS)
    if np.any(delta[fallback] != 0.0) or not np.array_equal(predictions[fallback], baseline[fallback]):
        raise E79Error("E79 missing/under-supported fallback is not exact R06")
    if not np.isfinite(predictions).all():
        raise E79Error("E79 Test predictions contain nonfinite values")
    return predictions, delta, medians


def _source_hashes() -> dict[str, str]:
    hashes = {rel: sha_file(ROOT / rel) for rel in CODE_PATHS}
    return hashes


def make_freeze() -> dict[str, Any]:
    if not PREFLIGHT.is_file() or not SYNTHETIC.is_file() or not VECTORIZER.is_file() or not VECTORIZER_RECEIPT.is_file():
        raise E79Error("Target-free preflight, synthetic QA, or extracted vectorizer is missing")
    preflight = _read_json(PREFLIGHT)
    synthetic = _read_json(SYNTHETIC)
    vectorizer_receipt = _read_json(VECTORIZER_RECEIPT)
    if preflight.get("status") != "PASS" or synthetic.get("status") != "PASS" or vectorizer_receipt.get("status") != "PASS":
        raise E79Error("Target-free preflight/synthetic/vectorizer receipt is not PASS")
    input_paths = [
        "data/raw/regression/Train.csv", "data/raw/regression/Test.csv",
        "data/raw/classification/Test.csv", "data/raw/regression/SampleSubmission.csv",
        "experiments/optimization_v3/rnbr_final_e79_v1/protocol.json",
        "reports/optimization_v3/rnbr_final_e79_v1/preflight.json",
        "reports/optimization_v3/rnbr_final_e79_v1/synthetic_qa.json",
        "experiments/runs/optimization_v3_expanded_submissions_v1/R06_ORIGINAL/predictions.npz",
        "reports/submissions/optimization_v3_expanded_v1/R06_ORIGINAL/manifest.json",
        "submissions/optimization_v3_expanded_v1/r06_original.csv",
        "experiments/optimization_v3/rnbr_pilot_v1/official_freeze.json",
        "experiments/optimization_v3/c19_a02_v2/A02/protocol.json",
        "experiments/optimization_v3/c19_a02_v2/A02/freeze.json",
        "experiments/runs/optimization_v3_a02_neighbors_v1/E75/observed/model.joblib",
        "experiments/runs/optimization_v3_a02_neighbors_v1/E75/observed/fit_receipt.json",
        "experiments/runs/optimization_v3_a02_neighbors_v1/E75/observed/fit_state.json",
        "experiments/runs/optimization_v3_research_next_v1/RNBR_FINAL_E79_OBSERVED/a02_e75_vectorizer.joblib",
        "experiments/runs/optimization_v3_research_next_v1/RNBR_FINAL_E79_OBSERVED/a02_e75_vectorizer_receipt.json",
    ]
    pilot_freeze = _read_json(PILOT_FREEZE)
    for cutoff in OOF_CUTOFFS:
        item = pilot_freeze["r06_lineage"]["cache_inventory"][str(cutoff)]
        input_paths.extend((item["npz"], item["json"]))
    input_hashes = {path: sha_file(ROOT / path) for path in input_paths}
    body = {
        "schema_version": 1,
        "protocol_id": "RNBR_FINAL_E79_OBSERVED_TEST_BENCHMARK_V1",
        "authorized_base_head": "0927679ed65865d42de8ec628a25fd696cc2f081",
        "preparation_code_commit": preflight["preparation_code_commit"],
        "official_train": {"path": "data/raw/regression/Train.csv", "rows": 50_608,
                           "sha256": EXPECTED["data/raw/regression/Train.csv"],
                           "target_population": "only present-stream donor rows, cycles 56-79"},
        "official_test": {"path": "data/raw/regression/Test.csv", "rows": 29_421,
                          "sha256": EXPECTED["data/raw/regression/Test.csv"],
                          "target_labels_opened": False, "feature_use": ["record_id", "cycle_code", "signal_stream"]},
        "r06_test_baseline": preflight["r06_test_manifest"],
        "a02_vectorizer": {"source_model_sha256": sha_file(A02_MODEL),
                           "source_receipt_sha256": sha_file(A02_RECEIPT),
                           "source_state_sha256": sha_file(A02_STATE),
                           "extracted_sha256": sha_file(VECTORIZER),
                           "audit": preflight["a02_text_space_audit"], "refit": False},
        "code_sha256": _source_hashes(),
        "input_sha256": input_hashes,
        "recipe": {"donor_cycles": [56, 79], "present_stream_only": True,
                   "r06_oof_cutoffs": {str(k): int(v) for k, v in sorted(EXPECTED_SOURCE_CUTOFFS.items()) if 56 <= int(k) <= 79},
                   "neighbors": 32, "positive_cosine_only": True, "weight": "cosine squared",
                   "minimum_neighbors": MIN_NEIGHBORS, "weighted_median": "R-NBR frozen implementation",
                   "delta": "clip(0.25*median,-5,+5)", "clip": [0.0, 100.0], "rerank": False,
                   "fallback": "exact R06 for missing stream or fewer than five positive neighbors"},
        "fit_count": 1, "fit_id": FIT_ID,
        "source_boundaries": {"classification_labels_or_probabilities": "never read",
                               "test_labels": "never read", "sample_submission": "schema and ID order only",
                               "test_features": "inference only", "preprocessing_refit": False},
        "synthetic_qa_sha256": sha_file(SYNTHETIC),
        "resource_policy": {"jobs": 1, "threads": THREADS, "rss_cap_bytes": RSS_CAP,
                            "available_ram_reserve_bytes": FREE_RAM_MIN},
    }
    body["freeze_signature"] = digest_bytes(canonical_bytes(body))
    return body


def _write_preflight() -> dict[str, Any]:
    report = _runtime_preflight(extract=True)
    rendered = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    PREFLIGHT.parent.mkdir(parents=True, exist_ok=True)
    if PREFLIGHT.exists():
        prior_text = PREFLIGHT.read_text(encoding="utf-8")
        existing = json.loads(prior_text)
        if existing.get("status") != "PASS" or existing.get("train_rows") != report.get("train_rows"):
            raise E79Error("Existing preflight report differs")
        # Stable fields are verified against the same live inputs; keep original execution timestamp.
        for key in report:
            if key == "created_at_utc":
                continue
            if existing.get(key) != report[key]:
                raise E79Error(f"Preflight changed at {key}")
        return existing
    tmp = PREFLIGHT.with_suffix(".json.staging")
    with tmp.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(rendered)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, PREFLIGHT)
    return _read_json(PREFLIGHT)


def synthetic_qa() -> dict[str, Any]:
    # Exercise the same deterministic top-k and weighted-median path with safe Unicode IDs.
    rows = np.array([3, 8, 13, 21, 34, 55], dtype=np.int64)
    ids = np.asarray(["регression-α", "漏洞-東京", "record-🛡️", "café", "Δοκιμή", "naïve"], dtype=str)
    cycles = np.array([56, 57, 60, 64, 70, 79], dtype=np.int64)
    residual = np.array([-10.0, -2.0, 3.0, 8.0, 13.0, 21.0], dtype=np.float64)
    bank = sparse.csr_matrix(np.array([[1,0,0,0],[1,1,0,0],[1,0,1,0],[1,1,1,0],[1,0,0,1],[1,1,0,1]], dtype=np.float64))
    bank = sparse.diags(1.0 / np.sqrt(np.asarray(bank.multiply(bank).sum(axis=1)).ravel())) @ bank
    queries = sparse.csr_matrix(np.array([[1,0,0,0],[0,0,1,0],[0,0,0,0]], dtype=np.float64))
    queries = sparse.diags([1.0,1.0,1.0]) @ queries
    qids = np.asarray(["query-日本", "query-🧪", "query-empty"], dtype=str)
    qrows = np.array([1001,1002,1003], dtype=np.int64)
    missing = np.array([False,False,True])
    sims = (queries @ bank.T).toarray()
    nrows = np.full((3, MAX_NEIGHBORS), -1, dtype=np.int64)
    nsims = np.zeros((3, MAX_NEIGHBORS), dtype=np.float64)
    counts = np.zeros(3, dtype=np.int16)
    for i in range(2):
        pos = np.flatnonzero(sims[i] > 0)
        order = np.lexsort((rows[pos], -sims[i, pos]))[:MAX_NEIGHBORS]
        chosen = pos[order]
        counts[i] = len(chosen)
        nrows[i, :len(chosen)] = rows[chosen]
        nsims[i, :len(chosen)] = sims[i, chosen]
    baseline = np.array([44.5, 62.0, 15.0])
    predicted, delta, medians = _predict(baseline, missing, nrows, nsims, counts, rows, residual)
    exact_half = weighted_median_residual(np.array([0.0, 10.0]), np.array([1.0, 1.0]), np.array([4, 9]))
    if exact_half != 5.0:
        raise E79Error("Synthetic exact-half weighted-median midpoint behavior failed")
    payload = {"donor_ids": ids, "donor_rows": rows, "donor_cycles": cycles, "donor_residuals": residual,
               "query_ids": qids, "query_rows": qrows, "neighbor_rows": nrows,
               "neighbor_similarities": nsims, "neighbor_counts": counts,
               "baseline": baseline, "prediction": predicted, "delta": delta, "median": medians}
    with __import__("tempfile").TemporaryDirectory(prefix="e79-synthetic-") as tempdir:
        path = Path(tempdir) / "synthetic.npz"
        _save_npz_once(path, **payload)
        with np.load(path, allow_pickle=False) as archive:
            loaded = {name: np.asarray(archive[name]).copy() for name in archive.files}
        if any(not np.array_equal(loaded[name], payload[name]) for name in payload):
            raise E79Error("Synthetic safe-Unicode persistence/readback failed")
        original_sha = sha_file(path)
        _save_npz_once(path, **payload)  # idempotent resume is read-only when bytes match
        if sha_file(path) != original_sha:
            raise E79Error("Synthetic resume changed the durable artifact")
        try:
            changed = dict(payload)
            changed["prediction"] = predicted + 1.0
            _save_npz_once(path, **changed)
        except E79Error:
            overwrite_rejected = True
        else:
            overwrite_rejected = False
        if not overwrite_rejected:
            raise E79Error("Synthetic changed-output overwrite was not rejected")
    result = {"status": "PASS", "fit_type": "SYNTHETIC_ONLY", "safe_unicode": True,
              "pickle_free": True, "geometry_and_formula": True, "persistence_readback": True,
              "idempotent_resume": True, "different_output_rejected": True,
              "exact_half_weighted_median_midpoint": exact_half,
              "synthetic_rows": {"donor": len(rows), "query": len(qids)},
              "prediction_sha256": sha_array(predicted, "<f8")}
    _write_once(SYNTHETIC, result)
    return result


def prepare() -> dict[str, Any]:
    try:
        subprocess.run(["git", "merge-base", "--is-ancestor", "0927679ed65865d42de8ec628a25fd696cc2f081", "HEAD"], cwd=ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        raise E79Error("Preparation HEAD does not descend from the authorized checkpoint") from exc
    preflight = _write_preflight()
    synthetic = synthetic_qa()
    freeze = make_freeze()
    _write_once(FREEZE, freeze)
    return {"preflight": preflight, "synthetic": synthetic, "freeze_signature": freeze["freeze_signature"]}


def _current_freeze() -> dict[str, Any]:
    if not FREEZE.is_file():
        raise E79Error("E79 freeze is missing")
    freeze = _read_json(FREEZE)
    body = {k: v for k, v in freeze.items() if k != "freeze_signature"}
    if freeze.get("freeze_signature") != digest_bytes(canonical_bytes(body)) or freeze != make_freeze():
        raise E79Error("E79 frozen protocol, code, inputs, vectorizer, or preflight changed")
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise E79Error("Official E79 fit requires committed preparation and a clean worktree")
    return freeze


def _ledger_events() -> list[dict[str, Any]]:
    return prior._read_events()


def _ledger_snapshot() -> dict[str, Any]:
    return prior._snapshot(_ledger_events())


def _ledger_append(event: Mapping[str, Any]) -> None:
    prior._append_event(event)


def amend_and_reserve(freeze: Mapping[str, Any]) -> dict[str, Any]:
    events = _ledger_events()
    snap = prior._snapshot(events)
    old_caps = {"pilot": 38, "continuation": 37, "control_replay": 0, "technical_retry": 1}
    new_caps = {"pilot": 38, "continuation": 38, "control_replay": 0, "technical_retry": 1}
    amendments = [e for e in events if e.get("event") == "BUDGET_CAP_AMENDMENT" and e.get("amendment_id") == AMENDMENT_ID]
    if len(amendments) > 1:
        raise E79Error("Duplicate E79 budget cap amendment ID")
    if not amendments:
        if (snap != {"events": 197, "global_cap": 76, "caps": old_caps,
                     "active_by_category": old_caps, "active_reserved": 76,
                     "started": 76, "saved": 73, "failed": 3, "pending": 0}):
            raise E79Error(f"Live pre-E79 budget does not equal the checkpoint: {snap}")
        if any(e.get("fit_id") == FIT_ID for e in events):
            raise E79Error("E79 fit ID already appears in ledger; no retry or replay")
        _ledger_append({"event": "BUDGET_CAP_AMENDMENT", "batch_id": "optimization_v3_research_next_v1",
                        "amendment_id": AMENDMENT_ID, "card_id": CARD_ID,
                        "authorized_by": "Sol final-preparation decision; user-approved E79 one-fit scope",
                        "prior_global_cap": 76, "global_cap": 77, "prior_effective_caps": old_caps,
                        "effective_caps": new_caps, "active_reservations_before": snap["active_by_category"],
                        "active_total_before": snap["active_reserved"], "pre_amendment_event_rows": snap["events"],
                        "pre_amendment_ledger_sha256": sha_file(LEDGER),
                        "freeze_signature": freeze["freeze_signature"],
                        "source_commit": _git("rev-parse", "HEAD"),
                        "reason": "Exactly one authorized E79 observed target-memory fit; no other cap or reservation is changed."})
        events = _ledger_events()
    amendment = [e for e in events if e.get("event") == "BUDGET_CAP_AMENDMENT" and e.get("amendment_id") == AMENDMENT_ID]
    if len(amendment) != 1 or amendment[0].get("global_cap") != 77 or amendment[0].get("effective_caps") != new_caps or amendment[0].get("freeze_signature") != freeze["freeze_signature"]:
        raise E79Error("Existing E79 cap amendment differs from frozen scope")
    reservations = [e for e in events if e.get("event") == "CARD_FITS_RESERVED" and e.get("reservation_key") == RESERVATION_KEY]
    if len(reservations) > 1:
        raise E79Error("Duplicate E79 reservation")
    if not reservations:
        snap = prior._snapshot(events)
        if (snap["global_cap"] != 77 or snap["caps"] != new_caps or snap["active_reserved"] != 76
                or snap["started"] != 76 or snap["pending"] != 0):
            raise E79Error(f"Post-amendment pre-reservation state differs: {snap}")
        _ledger_append({"event": "CARD_FITS_RESERVED", "batch_id": "optimization_v3_research_next_v1",
                        "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
                        "category": "continuation", "fits": 1, "fit_ids": [FIT_ID],
                        "amendment_id": AMENDMENT_ID, "global_cap": 77, "effective_category_cap": 38,
                        "freeze_signature": freeze["freeze_signature"], "source_commit": _git("rev-parse", "HEAD")})
        events = _ledger_events()
    reservation = [e for e in events if e.get("event") == "CARD_FITS_RESERVED" and e.get("reservation_key") == RESERVATION_KEY]
    if len(reservation) != 1 or reservation[0].get("fits") != 1 or reservation[0].get("fit_ids") != [FIT_ID] or reservation[0].get("freeze_signature") != freeze["freeze_signature"]:
        raise E79Error("E79 reservation differs from the one-fit authorization")
    snap = prior._snapshot(events)
    if snap["global_cap"] != 77 or snap["caps"] != new_caps or snap["active_reserved"] != 77 or snap["started"] != 76 or snap["pending"] != 0:
        raise E79Error(f"E79 post-reservation ledger state differs: {snap}")
    return {"snapshot": snap, "ledger_sha256": sha_file(LEDGER), "amendment_id": AMENDMENT_ID,
            "reservation_key": RESERVATION_KEY}


def _start_fit(freeze: Mapping[str, Any]) -> dict[str, Any]:
    events = _ledger_events()
    snap = prior._snapshot(events)
    if snap["pending"] != 0:
        raise E79Error("Another shared-ledger fit is pending; no concurrent execution")
    if any(event.get("event") == "FIT_STARTED" and event.get("fit_id") == FIT_ID for event in events):
        raise E79Error("E79 fit ID is already started; no retries or reruns are authorized")
    reserve = [e for e in events if e.get("event") == "CARD_FITS_RESERVED" and e.get("reservation_key") == RESERVATION_KEY]
    if len(reserve) != 1 or reserve[0].get("fit_ids") != [FIT_ID] or snap["started"] != 76 or snap["global_cap"] != 77:
        raise E79Error("E79 fit slot is not available")
    event = {"event": "FIT_STARTED", "batch_id": "optimization_v3_research_next_v1",
             "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
             "fit_id": FIT_ID, "cutoff": 79, "component": "observed_residual_memory_and_full_test_geometry",
             "fit_consumed_at_start": True, "amendment_id": AMENDMENT_ID,
             "freeze_signature": freeze["freeze_signature"], "source_commit": _git("rev-parse", "HEAD")}
    _ledger_append(event)
    return event


def _finish_fit(status: str, receipt_hash: str | None = None, error: str | None = None) -> None:
    events = _ledger_events()
    starts = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("fit_id") == FIT_ID]
    terminals = [e for e in events if e.get("fit_id") == FIT_ID and e.get("event") in {"FIT_SAVED", "FIT_FAILED"}]
    if len(starts) != 1 or terminals:
        raise E79Error("E79 start/terminal ledger identity is not uniquely writable")
    if status not in {"FIT_SAVED", "FIT_FAILED"}:
        raise E79Error("Invalid E79 terminal status")
    _ledger_append({"event": status, "batch_id": "optimization_v3_research_next_v1",
                    "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
                    "fit_id": FIT_ID, "cutoff": 79, "receipt_sha256": receipt_hash, "error": error})


def execute() -> dict[str, Any]:
    freeze = _current_freeze()
    pre = _runtime_preflight(extract=False)
    ledger = amend_and_reserve(freeze)
    _check_resources()
    _start_fit(freeze)  # durable slot consumption immediately before any target value is accessed
    try:
        targetfree = official.load_targetfree_train()
        donor_rows = np.flatnonzero((targetfree["cycles"] >= 56) & (targetfree["cycles"] <= 79) & targetfree["present"]).astype(np.int64)
        donor_y = _load_donor_targets(donor_rows, targetfree)
        donor_cycles = targetfree["cycles"][donor_rows]
        donor_ids = targetfree["ids"][donor_rows].astype(str)
        r06_full = _read_json(PILOT_FREEZE)
        r06_caches, _ = _load_r06_inputs(r06_full, targetfree)
        oof, cutoffs, oof_ids = _donor_oof(donor_rows, donor_cycles, donor_ids, r06_caches)
        residuals = donor_y - oof
        reg_test, _, sample = _targetfree_test()
        baseline, r06_test_identity = _load_r06_test(reg_test)
        test_ids = reg_test.record_id.astype(str).to_numpy()
        test_cycles = reg_test.cycle_code.to_numpy(dtype=np.int64)
        test_texts, missing = _documents(reg_test.signal_stream)
        donor_texts, donor_missing = _documents(pd.read_csv(TRAIN, usecols=["signal_stream"]).signal_stream.iloc[donor_rows])
        if np.any(donor_missing) or len(donor_texts) != len(donor_rows):
            raise E79Error("Frozen present-stream donor text did not normalize as present")
        vectorizer = joblib.load(VECTORIZER)
        if not isinstance(vectorizer, TfidfVectorizer):
            raise E79Error("Reusable fitted A02 vectorizer artifact failed type check")
        bank = vectorizer.transform(donor_texts).tocsr()
        query = vectorizer.transform(test_texts).tocsr()
        _check_resources()
        bank_rows = donor_rows.copy()
        neighbor_rows, similarities, counts, missing = _top32(query, bank, bank_rows, missing)
        predictions, delta, medians = _predict(baseline, missing, neighbor_rows, similarities, counts, donor_rows, residuals)
        fallback = missing | (counts < MIN_NEIGHBORS)
        if not np.array_equal(predictions[fallback], baseline[fallback]):
            raise E79Error("Final exact-R06 fallback check failed")
        _save_sparse_once(BANK_MATRIX, bank)
        _save_npz_once(BUNDLE,
            donor_rows=donor_rows, donor_ids=donor_ids.astype(str), donor_cycles=donor_cycles,
            donor_targets=donor_y, donor_r06_oof=oof, donor_oof_cutoffs=cutoffs,
            donor_oof_ids=oof_ids.astype(str), donor_residuals=residuals,
            query_ids=test_ids.astype(str), query_cycles=test_cycles, query_missing=missing,
            baseline=baseline, neighbor_rows=neighbor_rows, neighbor_similarities=similarities,
            neighbor_counts=counts, weighted_median=medians, delta=delta, prediction=predictions)
        # Immediately read back using allow_pickle=False before creating the receipt.
        with np.load(BUNDLE, allow_pickle=False) as archive:
            if not np.array_equal(archive["prediction"], predictions) or not np.array_equal(archive["query_ids"], test_ids):
                raise E79Error("Official E79 bundle persistence/readback failed")
        bank_audit = _csr_digest(sparse.load_npz(BANK_MATRIX).tocsr())
        receipt = {
            "fit_id": FIT_ID, "status": "FIT_OUTPUT_DURABLE", "freeze_signature": freeze["freeze_signature"],
            "source_commit": _git("rev-parse", "HEAD"), "ledger_start_present": True,
            "ledger_start_source_commit": _git("rev-parse", "HEAD"), "fit_count": 1,
            "donor_rows": len(donor_rows), "donor_cycles": [56, 79],
            "donor_rows_sha256": sha_array(donor_rows, "<i8"), "donor_ids_sha256": sha_array(donor_ids),
            "donor_targets_sha256": sha_array(donor_y, "<f8"), "donor_r06_oof_sha256": sha_array(oof, "<f8"),
            "donor_oof_cutoffs": {str(c): int(np.sum(cutoffs == c)) for c in np.unique(cutoffs)},
            "donor_residual_sha256": sha_array(residuals, "<f8"), "bank_matrix": bank_audit,
            "bank_matrix_file_sha256": sha_file(BANK_MATRIX), "bundle_sha256": sha_file(BUNDLE),
            "baseline_artifact": r06_test_identity, "query_rows": len(test_ids),
            "query_ids_sha256": sha_array(test_ids), "query_missing_rows": int(missing.sum()),
            "query_positive_neighbor_rows": int(np.sum(counts > 0)),
            "query_supported_rows": int(np.sum(counts >= MIN_NEIGHBORS)),
            "query_neighbor_rows_sha256": sha_array(neighbor_rows, "<i8"),
            "query_similarity_sha256": sha_array(similarities, "<f8"),
            "prediction_sha256": sha_array(predictions, "<f8"), "delta_sha256": sha_array(delta, "<f8"),
            "prediction_min_max": [float(predictions.min()), float(predictions.max())],
            "exact_r06_fallback_rows": int(fallback.sum()), "r06_changed_rows": int(np.sum(predictions != baseline)),
            "peak_rss_bytes": max(PEAK_RSS_BYTES, psutil.Process().memory_info().rss),
            "target_boundary": "only present-stream official Regression Train rows cycles 56-79; Test labels never read",
            "classification_labels_or_probabilities_read": False,
        }
        receipt["receipt_sha256"] = digest_bytes(canonical_bytes(receipt))
        _write_once(FIT_RECEIPT, receipt)
        receipt_hash = sha_file(FIT_RECEIPT)
        _finish_fit("FIT_SAVED", receipt_hash)
        _write_submission(freeze, receipt, sample, test_ids, predictions, r06_test_identity)
        return {"receipt": receipt, "ledger": _ledger_snapshot(), "csv": str(CSV)}
    except Exception as exc:
        events = _ledger_events()
        terminal = [e for e in events if e.get("fit_id") == FIT_ID and e.get("event") in {"FIT_SAVED", "FIT_FAILED"}]
        if not terminal:
            _finish_fit("FIT_FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


def _write_submission(freeze: Mapping[str, Any], receipt: Mapping[str, Any], sample: pd.DataFrame,
                      test_ids: np.ndarray, predictions: np.ndarray, r06_identity: Mapping[str, Any]) -> None:
    sample_full = pd.read_csv(SAMPLE, dtype={"record_id": str})
    if sample_full.columns.tolist() != ["record_id", "threat_rank"] or not np.array_equal(sample_full.record_id.astype(str).to_numpy(), test_ids):
        raise E79Error("Official SampleSubmission schema/order differs from verified Test IDs")
    if len(predictions) != 29_421 or not np.isfinite(predictions).all() or np.any((predictions < 0) | (predictions > 100)):
        raise E79Error("E79 CSV prediction values are invalid")
    out = sample_full.copy()
    out["threat_rank"] = predictions.astype(np.float64)
    CSV.parent.mkdir(parents=True, exist_ok=True)
    staging = CSV.with_suffix(".csv.staging")
    out.to_csv(staging, index=False, float_format="%.17g", lineterminator="\n")
    os.replace(staging, CSV)
    check = pd.read_csv(CSV, dtype={"record_id": str})
    if (len(check) != 29_421 or check.columns.tolist() != ["record_id", "threat_rank"]
            or not np.array_equal(check.record_id.astype(str).to_numpy(), sample_full.record_id.astype(str).to_numpy())
            or not np.isfinite(check.threat_rank.to_numpy()).all()
            or np.any((check.threat_rank < 0) | (check.threat_rank > 100))):
        raise E79Error("Final E79 CSV readback verification failed")
    manifest = {
        "status": "LOCAL_BENCHMARK_READY_NOT_SUBMITTED", "candidate": "RNBR_FINAL_E79_OBSERVED",
        "source_commit": _git("rev-parse", "HEAD"), "freeze_signature": freeze["freeze_signature"],
        "fit_id": FIT_ID, "fit_count": 1, "fit_receipt_sha256": sha_file(FIT_RECEIPT),
        "bundle_sha256": sha_file(BUNDLE), "donor_matrix_sha256": sha_file(BANK_MATRIX),
        "csv": {"path_gitignored": CSV.relative_to(ROOT).as_posix(), "rows": len(check),
                "columns": check.columns.tolist(), "sha256": sha_file(CSV),
                "ordered_ids_sha256": sha_array(check.record_id.astype(str).to_numpy()),
                "prediction_raw_sha256": sha_array(predictions, "<f8"),
                "csv_precision": "17 significant digits; round-trip float64"},
        "official_inputs": {"train_sha256": EXPECTED["data/raw/regression/Train.csv"],
                            "test_sha256": EXPECTED["data/raw/regression/Test.csv"],
                            "sample_sha256": EXPECTED["data/raw/regression/SampleSubmission.csv"],
                            "r06_baseline": dict(r06_identity)},
        "a02_preprocessing": {"reuse_existing_fit": True, "refit": False,
                              "source_model_sha256": sha_file(A02_MODEL),
                              "extracted_vectorizer_sha256": sha_file(VECTORIZER),
                              "vectorizer_receipt_sha256": sha_file(VECTORIZER_RECEIPT)},
        "source_boundaries": {"official_train_donor_target_rows_only": True,
                               "test_labels_read": False, "classification_labels_or_probabilities_read": False,
                               "sample_use": "schema and ordered IDs only", "test_use": "inference only"},
        "submission": {"kaggle_upload": False, "status": "NOT_SUBMITTED"},
    }
    _write_once(MANIFEST, manifest)
    report = (
        "# RNBR E79 observed-memory benchmark\n\n"
        "Status: local full-Test CSV prepared for manual review; no Kaggle upload.\n\n"
        f"- Fit ID: `{FIT_ID}` (one consumed continuation slot).\n"
        f"- Freeze: `{freeze['freeze_signature']}`.\n"
        f"- Donors: {receipt['donor_rows']:,} present-stream Regression Train rows, cycles 56–79.\n"
        f"- Test rows: {len(check):,}; exact SampleSubmission schema and ID order.\n"
        f"- Supported neighbor rows: {receipt['query_supported_rows']:,}; exact R06 fallback rows: {receipt['exact_r06_fallback_rows']:,}.\n"
        f"- Changed R06 predictions: {receipt['r06_changed_rows']:,}. No Test target was opened.\n"
        f"- CSV: `{CSV.relative_to(ROOT).as_posix()}` SHA-256 `{sha_file(CSV)}`.\n"
        f"- Independent verification: see `reports/optimization_v3/rnbr_final_e79_v1/verification.json`.\n"
    )
    if FINAL_REPORT.exists() and FINAL_REPORT.read_text(encoding="utf-8") != report:
        raise E79Error("Refusing to overwrite different final report")
    FINAL_REPORT.parent.mkdir(parents=True, exist_ok=True)
    if not FINAL_REPORT.exists():
        FINAL_REPORT.write_text(report, encoding="utf-8", newline="\n")


def recover() -> dict[str, Any]:
    events = _ledger_events()
    starts = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("fit_id") == FIT_ID]
    terminals = [e for e in events if e.get("fit_id") == FIT_ID and e.get("event") in {"FIT_SAVED", "FIT_FAILED"}]
    if len(starts) != 1 or terminals:
        raise E79Error("No unique pending E79 start exists to recover")
    if FIT_RECEIPT.is_file() and BUNDLE.is_file() and BANK_MATRIX.is_file():
        receipt = _read_json(FIT_RECEIPT)
        body = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
        if (receipt.get("receipt_sha256") != digest_bytes(canonical_bytes(body))
                or receipt.get("bundle_sha256") != sha_file(BUNDLE)
                or receipt.get("bank_matrix_file_sha256") != sha_file(BANK_MATRIX)):
            raise E79Error("Pending E79 outputs exist but receipt integrity fails; no retry")
        _finish_fit("FIT_SAVED", sha_file(FIT_RECEIPT))
        return {"status": "RECOVERED_SAVED", "fit_id": FIT_ID}
    _finish_fit("FIT_FAILED", error="Interrupted before durable complete bundle; no retry authorized")
    return {"status": "RECOVERED_FAILED_NO_RETRY", "fit_id": FIT_ID}


def export_from_durable_fit() -> dict[str, Any]:
    freeze = _read_json(FREEZE)
    freeze_body = {k: v for k, v in freeze.items() if k != "freeze_signature"}
    if freeze.get("freeze_signature") != digest_bytes(canonical_bytes(freeze_body)) or freeze != make_freeze():
        raise E79Error("Cannot export: current E79 frozen sources/inputs differ")
    receipt = _read_json(FIT_RECEIPT)
    body = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    if (freeze.get("freeze_signature") != _read_json(FIT_RECEIPT).get("freeze_signature")
            or receipt.get("receipt_sha256") != digest_bytes(canonical_bytes(body))
            or receipt.get("bundle_sha256") != sha_file(BUNDLE)
            or receipt.get("bank_matrix_file_sha256") != sha_file(BANK_MATRIX)):
        raise E79Error("Cannot export: durable E79 fit receipt/bundle lineage failed")
    events = _ledger_events()
    started = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("fit_id") == FIT_ID]
    saved = [e for e in events if e.get("event") == "FIT_SAVED" and e.get("fit_id") == FIT_ID]
    if len(started) != 1 or len(saved) != 1 or saved[0].get("receipt_sha256") != sha_file(FIT_RECEIPT):
        raise E79Error("Cannot export: E79 fit is not uniquely recorded as saved")
    with np.load(BUNDLE, allow_pickle=False) as archive:
        predictions = np.asarray(archive["prediction"], dtype=np.float64).copy()
        test_ids = np.asarray(archive["query_ids"]).astype(str)
    reg_test, _, sample = _targetfree_test()
    if not np.array_equal(test_ids, reg_test.record_id.astype(str).to_numpy()):
        raise E79Error("Cannot export: durable E79 query IDs differ from official Test")
    baseline, r06_identity = _load_r06_test(reg_test)
    if len(predictions) != len(baseline):
        raise E79Error("Cannot export: E79 predictions are not Test-aligned")
    _write_submission(freeze, receipt, sample, test_ids, predictions, r06_identity)
    return {"status": "CSV_READY", "csv": str(CSV), "csv_sha256": sha_file(CSV),
            "manifest": str(MANIFEST), "fit_id": FIT_ID}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "synthetic", "preflight", "execute", "recover", "export", "freeze-check"))
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare()
    elif args.command == "synthetic":
        result = synthetic_qa()
    elif args.command == "preflight":
        result = _runtime_preflight(extract=False)
    elif args.command == "freeze-check":
        result = _current_freeze()
    elif args.command == "execute":
        result = execute()
    elif args.command == "export":
        result = export_from_durable_fit()
    else:
        result = recover()
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False, default=str))


if __name__ == "__main__":
    main()
