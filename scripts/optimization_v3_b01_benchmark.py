"""Prepare a zero-fit B01 benchmark from frozen A02 full-Train Test geometry.

The script never loads Test targets, fits or transforms a model, or contacts Kaggle.
It requires the pre-score --freeze stage before --prepare.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import joblib
import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits

from scripts import optimization_v3_a02_b01_centering as b01
from scripts import optimization_v3_a02_continuation as a02
from scripts import optimization_v3_c19_benchmark as c19
from src.models.research_next_contract import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/optimization_v3/b01_benchmark_v1"
FREEZE_PATH = REPORT / "FREEZE.json"
MANIFEST_PATH = REPORT / "MANIFEST.json"
RESULT_PATH = REPORT / "RESULTS.json"
QA_PATH = REPORT / "QA.json"
SUMMARY_PATH = REPORT / "SUMMARY.md"
OUTPUT_DIR = ROOT / "submissions/optimization_v3/b01_benchmark_v1"
CSV_PATH = OUTPUT_DIR / "b01_benchmark.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "b01_benchmark_predictions.npz"

TRAIN_PATH = ROOT / "data/raw/classification/Train.csv"
TEST_PATH = ROOT / "data/raw/classification/Test.csv"
SAMPLE_PATH = ROOT / "data/raw/classification/SampleSubmission.csv"
RUN_FOLDER = a02.RUN_ROOT / "E79/fulltrain_observed"
A02_MANIFEST_PATH = a02.REPORT_ROOT / "submission_manifest.json"
A02_RECEIPTS_PATH = a02.REPORT_ROOT / "submission_receipts.jsonl"
A02_CSV_PATH = ROOT / "submissions/optimization_v3/a02_continuation_v1/a02_continuation.csv"
A02_PREDICTIONS_PATH = RUN_FOLDER / "test_predictions.npz"
A02_PREDICTIONS_RECEIPT_PATH = RUN_FOLDER / "test_predictions.json"
BASE_CACHE_PATH = c19.BASE_CACHE
BASE_MANIFEST_PATH = c19.BASE_MANIFEST
LEDGER_PATH = a02.LEDGER_PATH
B01_RECIPE_FREEZE_PATH = ROOT / "reports/optimization_v3/a02_b01_centering_v1/FREEZE_V2.json"
B01_RECIPE_CARD_PATH = ROOT / "reports/optimization_v3/a02_next_decision_v1/B01_CARD.md"

EXPECTED_TRAIN_SHA256 = a02.EXPECTED_TRAIN_SHA256
EXPECTED_LEDGER_SHA256 = b01.EXPECTED_LEDGER_SHA256
EXPECTED_A02_FIT_ID = "A02_CONT_E79_OBSERVED_FULLTRAIN"
EXPECTED_TEST_ROWS = 29_421
EXPECTED_TRAIN_ROWS = 50_608
THREADS = 2
JOB_CAP_BYTES = 6 * 1024**3
FREE_RAM_AND_RESERVE_BYTES = 8 * 1024**3
THRESHOLD = 0.50
NEIGHBOR_WIDTH = 32
ALLOWED_TEST_COLUMNS = ("record_id", "cycle_code", "signal_stream")
SAFE_PREDICTION_KEYS = (
    "record_ids", "baseline_probability", "missing_mask", "candidate_probability",
    "neighbor_count", "neighbor_source_rows", "neighbor_ids", "neighbor_similarities",
    "correction", "correction_unclipped",
)
EXPECTED_PREDICTION_KEYS = set(SAFE_PREDICTION_KEYS) | {"labels"}
SOURCE_PATHS = (
    "scripts/optimization_v3_b01_benchmark.py",
    "tests/test_optimization_v3_b01_benchmark.py",
    "scripts/optimization_v3_a02_b01_centering.py",
    "scripts/optimization_v3_a02_continuation.py",
    "scripts/optimization_v3_a02_neighbors.py",
    "scripts/optimization_v3_c19_benchmark.py",
    "src/models/classification_a02_neighbors.py",
    "src/models/research_next_contract.py",
)
CONTROL_PATHS = (
    "reports/optimization_v3/a02_b01_centering_v1/FREEZE_V2.json",
    "reports/optimization_v3/a02_next_decision_v1/B01_CARD.md",
    "reports/optimization_v3/a02_continuation_v1/submission_manifest.json",
    "reports/optimization_v3/a02_continuation_v1/submission_receipts.jsonl",
    "reports/optimization_v3/kaggle/v3_cls_sparse25.json",
    "experiments/optimization_v3/a02_continuation_v1/protocol.json",
    "experiments/optimization_v3/a02_continuation_v1/freeze_v2.json",
    "experiments/runs/optimization_v3_a02_continuation_v1/E79/fulltrain_observed/fit_receipt.json",
    "experiments/runs/optimization_v3_a02_continuation_v1/E79/fulltrain_observed/fit_state.json",
    "experiments/runs/optimization_v3_a02_continuation_v1/E79/fulltrain_observed/model.joblib",
    "experiments/runs/optimization_v3_a02_continuation_v1/E79/fulltrain_observed/test_predictions.json",
    "experiments/runs/optimization_v3_a02_continuation_v1/E79/fulltrain_observed/test_predictions.npz",
    "experiments/runs/optimization_v3_research_next_v1/budget.jsonl",
    "experiments/runs/optimization_v3_inference/v3_cls_sparse25.npz",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: Any, dtype: str | np.dtype | None = None) -> str:
    array = np.asarray(values, dtype=dtype)
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True,
                       allow_nan=False) + "\n").encode("utf-8")


def _atomic_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"Existing immutable benchmark output differs; refusing overwrite: {path}")
        return
    staging = path.with_name(f".{path.name}.staging")
    if staging.exists():
        raise RuntimeError(f"Interrupted output staging exists; inspect before resuming: {staging}")
    with staging.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    staging.rename(path)


def _write_once(path: Path, value: Any) -> None:
    _atomic_once(path, _json_bytes(value))


def _current_commit() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _preflight_memory() -> dict[str, int]:
    virtual = psutil.virtual_memory()
    rss = psutil.Process().memory_info().rss
    if virtual.available < FREE_RAM_AND_RESERVE_BYTES:
        raise RuntimeError("B01 benchmark needs at least 8 GiB free RAM (6 GiB job cap plus 2 GiB reserve)")
    if rss > JOB_CAP_BYTES:
        raise RuntimeError("B01 benchmark process already exceeds the 6 GiB job cap")
    return {"available_bytes_before": int(virtual.available), "process_rss_bytes_before": int(rss),
            "job_cap_bytes": JOB_CAP_BYTES, "reserve_bytes": 2 * 1024**3, "threads": THREADS}


def _read_test_features_only() -> tuple[pd.DataFrame, list[str], list[str]]:
    test_header = pd.read_csv(TEST_PATH, nrows=0).columns.tolist()
    sample_schema = pd.read_csv(SAMPLE_PATH, nrows=0).columns.tolist()
    _validate_input_schemas(test_header, sample_schema)
    test = pd.read_csv(TEST_PATH, usecols=list(ALLOWED_TEST_COLUMNS), dtype={"record_id": str})
    if test.columns.tolist() != list(ALLOWED_TEST_COLUMNS):
        raise RuntimeError("Test selected feature-column order differs from the saved-geometry contract")
    if len(test) != EXPECTED_TEST_ROWS or test.record_id.isna().any() or test.record_id.duplicated().any():
        raise RuntimeError("Official Test ordered IDs are missing, duplicated, or wrong length")
    test["cycle_code"] = pd.to_numeric(test["cycle_code"], errors="raise").astype(np.int16)
    cycles = test.cycle_code.to_numpy(dtype=np.int16)
    if not set(np.unique(cycles)).issubset(set(range(80, 92))) or not np.array_equal(np.unique(cycles), np.arange(80, 92)):
        raise RuntimeError("Official Test cycle coverage is not exactly 80–91")
    return test, test_header, sample_schema


def _validate_input_schemas(test_header: list[str], sample_schema: list[str]) -> None:
    if len(test_header) != len(set(test_header)):
        raise RuntimeError("Official Test header contains duplicate columns")
    if "artifact_signal" in test_header or "threat_rank" in test_header:
        raise RuntimeError("A target column is present in official Test; refusing to load it")
    missing = [name for name in ALLOWED_TEST_COLUMNS if name not in test_header]
    if missing:
        raise RuntimeError(f"Required saved-geometry validation columns are absent from Test: {missing}")
    if sample_schema != ["record_id", "artifact_signal"]:
        raise RuntimeError(f"Official SampleSubmission schema differs: {sample_schema}")


def _load_safe_npz(path: Path, expected: set[str], allowed_to_load: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != expected:
            raise RuntimeError(f"Saved A02 prediction archive schema differs: {path.name}")
        # Deliberately do not access the stored `labels` array. It is not needed;
        # A02 hard labels are independently reconstructed from probabilities.
        return {key: np.asarray(archive[key]).copy() for key in allowed_to_load}


def _verify_receipt(receipt: dict[str, Any], manifest: dict[str, Any]) -> None:
    payload = dict(receipt)
    claimed = payload.pop("receipt_sha256", None)
    if (claimed != canonical_sha256(payload)
            or claimed != manifest.get("fit_receipt_sha256")
            or receipt.get("fit_id") != EXPECTED_A02_FIT_ID
            or receipt.get("origin_cycle") != 79
            or receipt.get("residual_arm") != "observed"
            or receipt.get("task_estimator_fit_count") != 1
            or receipt.get("test_data_used") is not False
            or receipt.get("query_labels_passed_to_fit") is not False
            or receipt.get("text_prefix_rows") != EXPECTED_TRAIN_ROWS
            or receipt.get("bank_rows") != 36_010
            or receipt.get("bank_cycle_min") != 56
            or receipt.get("bank_cycle_max") != 79):
        raise RuntimeError("The original A02 full-Train fit receipt does not validate")


def _check_original_csv(ids: np.ndarray, sample_columns: list[str],
                        a02_manifest: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    if _sha256_file(A02_CSV_PATH) != a02_manifest.get("csv_sha256"):
        raise RuntimeError("Already-uploaded A02 CSV hash differs from its prepared manifest")
    frame = pd.read_csv(A02_CSV_PATH, dtype={"record_id": str})
    if (frame.columns.tolist() != sample_columns or len(frame) != EXPECTED_TEST_ROWS
            or frame.record_id.isna().any() or frame.record_id.duplicated().any()
            or not np.array_equal(frame.record_id.astype(str).to_numpy(), ids)):
        raise RuntimeError("Already-uploaded A02 CSV schema/order/IDs fail against official Test")
    labels = frame.artifact_signal.to_numpy()
    if labels.dtype.kind not in "iu" or not np.isin(labels, [0, 1]).all():
        raise RuntimeError("Already-uploaded A02 CSV predictions are not integer 0/1")
    return labels.astype(np.int8, copy=True), {
        "path": A02_CSV_PATH.relative_to(ROOT).as_posix(),
        "sha256": _sha256_file(A02_CSV_PATH),
        "labels_sha256": _array_sha256(labels, np.int8),
        "previously_uploaded_status": "USER_REPORTED_ONLY",
        "source_receipt_sha256": _sha256_file(A02_RECEIPTS_PATH),
    }


def _load_train_bank_labels_only(rows: np.ndarray, ids: np.ndarray,
                                 cycles: np.ndarray, train_features: pd.DataFrame) -> np.ndarray:
    """Read target values only for verified present-stream OOF-bank row positions."""
    if "artifact_signal" in train_features.columns:
        raise RuntimeError("Train helper unexpectedly loaded target labels outside the verified bank")
    selected_rows = set(int(row) for row in np.asarray(rows, dtype=np.int64))
    if len(selected_rows) != len(rows):
        raise RuntimeError("Verified bank row positions contain duplicates")
    bank_frame = pd.read_csv(
        TRAIN_PATH,
        usecols=["record_id", "cycle_code", "signal_stream", "artifact_signal"],
        dtype={"record_id": str},
        skiprows=lambda line: line > 0 and (line - 1) not in selected_rows,
    )
    if (len(bank_frame) != len(rows)
            or not np.array_equal(bank_frame.record_id.astype(str).to_numpy(), np.asarray(ids, dtype=str))
            or not np.array_equal(pd.to_numeric(bank_frame.cycle_code, errors="raise").to_numpy(dtype=np.int16),
                                  np.asarray(cycles, dtype=np.int16))
            or bank_frame.signal_stream.fillna("").astype(str).str.strip().eq("").any()):
        raise RuntimeError("Train rows selected for OOF labels do not match verified bank IDs/cycles/availability")
    labels = pd.to_numeric(bank_frame.artifact_signal, errors="raise").to_numpy(dtype=np.int8)
    if not np.isin(labels, [0, 1]).all():
        raise RuntimeError("Verified OOF bank contains nonbinary Train labels")
    return labels


def _load_verified_inputs() -> dict[str, Any]:
    resource = _preflight_memory()
    a02_freeze = a02.verify_freeze()
    b01_freeze = json.loads(B01_RECIPE_FREEZE_PATH.read_text(encoding="utf-8"))
    if (b01_freeze.get("status") != "B01_RECIPE_FROZEN_PRE_SCORE"
            or _sha256_file(Path(b01.__file__)) != b01_freeze.get("code_sha256")
            or b01_freeze.get("official_train_sha256") != EXPECTED_TRAIN_SHA256):
        raise RuntimeError("The reviewed B01 formula freeze/source hash does not validate")

    a02_manifest = json.loads(A02_MANIFEST_PATH.read_text(encoding="utf-8"))
    if (a02_manifest.get("status") != "PREPARED_VALIDATED_NOT_SUBMITTED"
            or a02_manifest.get("submission_authorized") is not False
            or a02_manifest.get("candidate_id") != "A02_LEXICAL_NEIGHBOR_RESIDUAL_CONTINUATION"
            or a02_manifest.get("fit_id") != EXPECTED_A02_FIT_ID
            or a02_manifest.get("test_target_loaded") is not False
            or a02_manifest.get("official_train_rows") != EXPECTED_TRAIN_ROWS
            or a02_manifest.get("official_train_sha256") != EXPECTED_TRAIN_SHA256):
        raise RuntimeError("The prior A02 full-Train manifest is not the expected no-Test-target record")
    receipt = json.loads((RUN_FOLDER / "fit_receipt.json").read_text(encoding="utf-8"))
    state = json.loads((RUN_FOLDER / "fit_state.json").read_text(encoding="utf-8"))
    _verify_receipt(receipt, a02_manifest)
    model_path = RUN_FOLDER / "model.joblib"
    if (_sha256_file(model_path) != a02_manifest.get("model_sha256")
            or canonical_sha256(state) != a02_manifest.get("fit_state_sha256")
            or receipt.get("model_sha256") != _sha256_file(model_path)
            or receipt.get("state_sha256") != canonical_sha256(state)):
        raise RuntimeError("The original A02 model/state hashes differ from the receipt/manifest")
    model = joblib.load(model_path)
    if canonical_sha256(model.audit_state()) != canonical_sha256(state):
        raise RuntimeError("The loaded A02 fitted state differs from its immutable fit state")

    # Feature-only Train rows and six frozen C16 OOF caches; no query targets.
    train, caches, _banks, _queries = a02._load_structural_inputs(include_target=False)
    protocol = a02._read_protocol()
    full_bank = a02._build_bank(79, caches, train, protocol)
    present = ~full_bank["missing"].astype(bool)
    expected_rows = full_bank["rows"][present].astype(np.int64)
    expected_ids = full_bank["ids"][present].astype(str)
    expected_cycles = full_bank["cycles"][present].astype(np.int16)
    expected_oof = full_bank["baseline"][present].astype(np.float64)
    bank_rows = model.bank_row_positions.astype(np.int64, copy=True)
    bank_ids = np.asarray([b01.audit.normalize_id(v) for v in model.bank_ids], dtype=str)
    bank_cycles = model.bank_cycles.astype(np.int16, copy=True)
    bank_oof = model.bank_oof_probabilities.astype(np.float64, copy=True)
    bank_labels = _load_train_bank_labels_only(expected_rows, expected_ids, expected_cycles, train)
    if (not np.array_equal(bank_rows, expected_rows)
            or not np.array_equal(bank_ids, expected_ids)
            or not np.array_equal(bank_cycles, expected_cycles)
            or not np.array_equal(bank_oof, expected_oof)
            or not np.isin(bank_labels, [0, 1]).all()
            or not np.array_equal(model.bank_residuals, bank_labels.astype(np.float64) - bank_oof)
            or not np.array_equal(model.residual_assignment, np.arange(len(bank_rows), dtype=np.int64))
            or np.any(bank_cycles > 79)):
        raise RuntimeError("Full-Train A02 bank rows/IDs/cycles/OOF/labels do not reconstruct from frozen C16 blocks")
    expected_blocks = [list(map(int, row)) for row in a02._bank_blocks(79, protocol)]
    if (a02_manifest.get("OOF_bank_blocks") != expected_blocks
            or a02_manifest.get("OOF_bank_rows") != len(full_bank["rows"])
            or a02_manifest.get("OOF_bank_present_rows") != len(bank_rows)):
        raise RuntimeError("A02 manifest OOF block recipe/denominators differ from the frozen C16-bank builder")

    test, test_header, sample_columns = _read_test_features_only()
    test_ids = test.record_id.astype(str).to_numpy(dtype=str)
    ordered_ids_hash = c19.ordered_ids_sha256(test_ids.tolist())
    test_file_sha = _sha256_file(TEST_PATH)
    if (len(train) != EXPECTED_TRAIN_ROWS or _sha256_file(TRAIN_PATH) != EXPECTED_TRAIN_SHA256
            or test_file_sha != a02_manifest.get("test_feature_sha256")
            or len(test_ids) != EXPECTED_TEST_ROWS
            or ordered_ids_hash != a02_manifest.get("test_ids_sha256", ordered_ids_hash)):
        raise RuntimeError("Official Train/Test row identity or frozen Test-feature hash differs")

    a02_pred_receipt = json.loads(A02_PREDICTIONS_RECEIPT_PATH.read_text(encoding="utf-8"))
    if (_sha256_file(A02_PREDICTIONS_PATH) != a02_manifest.get("prediction_bundle_sha256")
            or a02_pred_receipt.get("npz_sha256") != _sha256_file(A02_PREDICTIONS_PATH)
            or a02_pred_receipt.get("fit_receipt_sha256") != receipt.get("receipt_sha256")
            or a02_pred_receipt.get("freeze_signature") != a02_manifest.get("freeze_signature")
            or a02_pred_receipt.get("official_test_targets_loaded") is not False
            or a02_pred_receipt.get("rows") != EXPECTED_TEST_ROWS):
        raise RuntimeError("Saved A02 prediction bundle or target-free sidecar does not validate")
    prediction = _load_safe_npz(A02_PREDICTIONS_PATH, EXPECTED_PREDICTION_KEYS, SAFE_PREDICTION_KEYS)
    if (not np.array_equal(prediction["record_ids"].astype(str), test_ids)
            or prediction["candidate_probability"].shape != (EXPECTED_TEST_ROWS,)
            or prediction["baseline_probability"].shape != (EXPECTED_TEST_ROWS,)
            or not np.isfinite(prediction["candidate_probability"]).all()
            or not np.isfinite(prediction["baseline_probability"]).all()
            or prediction["neighbor_source_rows"].shape != (EXPECTED_TEST_ROWS, NEIGHBOR_WIDTH)
            or prediction["neighbor_ids"].shape != (EXPECTED_TEST_ROWS, NEIGHBOR_WIDTH)
            or prediction["neighbor_similarities"].shape != (EXPECTED_TEST_ROWS, NEIGHBOR_WIDTH)
            or prediction["neighbor_count"].shape != (EXPECTED_TEST_ROWS,)
            or prediction["missing_mask"].shape != (EXPECTED_TEST_ROWS,)):
        raise RuntimeError("Saved A02 Test predictions/geometry do not align with official Test IDs and shape")

    base_manifest = json.loads(BASE_MANIFEST_PATH.read_text(encoding="utf-8"))
    if (_sha256_file(BASE_CACHE_PATH) != a02_manifest.get("base_cache_sha256")
            or _sha256_file(BASE_MANIFEST_PATH) != a02_manifest.get("base_manifest_sha256")
            or base_manifest.get("status") != "PREPARED_AND_QA_PASS"
            or base_manifest.get("test_labels_used") is not False
            or base_manifest.get("fit_preprocessing_only_train") is not True
            or c19.ordered_ids_sha256(test_ids.tolist()) != base_manifest.get("ordered_test_ids_sha256")):
        raise RuntimeError("Full-Train Sparse25 baseline cache/manifest does not bind to official Test order")
    base_recipe = base_manifest.get("recipe")
    expected_base_recipe = {"kind": "availability_present_blend", "present": [
        {"model": "C2_B", "weight": 0.75}, {"model": "SPARSE_BINARY", "weight": 0.25}],
        "missing": "CB_MISSING", "threshold": 0.5}
    if base_recipe != expected_base_recipe:
        raise RuntimeError("Full-Train Sparse25 control recipe differs from A02's frozen base route")
    with np.load(BASE_CACHE_PATH, allow_pickle=False) as archive:
        if not {"probability", "missing_mask"}.issubset(set(archive.files)):
            raise RuntimeError("Sparse25 cache lacks its frozen probability/availability arrays")
        baseline_probability = np.asarray(archive["probability"]).copy()
        baseline_missing = np.asarray(archive["missing_mask"]).copy()
    stream_missing = test.signal_stream.fillna("").astype(str).str.strip().eq("").to_numpy(dtype=bool)
    if (baseline_probability.dtype != prediction["baseline_probability"].dtype
            or baseline_probability.shape != (EXPECTED_TEST_ROWS,)
            or not np.array_equal(baseline_probability, prediction["baseline_probability"])
            or baseline_missing.dtype != np.bool_
            or not np.array_equal(baseline_missing, prediction["missing_mask"])
            or not np.array_equal(baseline_missing, stream_missing)):
        raise RuntimeError("Full-Train C16 Sparse25 control probabilities or availability route differ")

    ids, a02_labels, a02_csv_evidence = _check_original_csv(test_ids, sample_columns, a02_manifest)
    if not np.array_equal(a02_labels, (prediction["candidate_probability"] >= THRESHOLD).astype(np.int8)):
        raise RuntimeError("Already-uploaded A02 CSV labels do not reconstruct from its saved probability vector")

    row_positions = np.arange(EXPECTED_TEST_ROWS, dtype=np.int64) + len(train)
    lookup = {int(row): index for index, row in enumerate(bank_rows)}
    counts = prediction["neighbor_count"].astype(np.int64)
    neigh = prediction["neighbor_source_rows"].astype(np.int64)
    neigh_ids = prediction["neighbor_ids"].astype(str)
    sims = prediction["neighbor_similarities"].astype(np.float64)
    missing = prediction["missing_mask"].astype(bool)
    test_cycles = test.cycle_code.to_numpy(dtype=np.int16)
    if np.any(counts < 0) or np.any(counts > NEIGHBOR_WIDTH):
        raise RuntimeError("Saved E79 neighbor counts are outside top-32 width")
    for i, raw_count in enumerate(counts):
        count = int(raw_count)
        if (np.any(neigh[i, count:] != -1) or np.any(neigh_ids[i, count:] != "")
                or np.any(sims[i, count:] != 0.0)):
            raise RuntimeError("Saved E79 geometry padding differs from frozen top-32 layout")
        if missing[i] and count != 0:
            raise RuntimeError("Missing-stream Test row has saved A02 neighbors")
        if count == 0:
            continue
        chosen = neigh[i, :count]
        if len(np.unique(chosen)) != count or np.any(chosen == row_positions[i]):
            raise RuntimeError("Saved E79 geometry contains duplicate/self neighbor rows")
        try:
            bank_indices = np.asarray([lookup[int(row)] for row in chosen], dtype=np.int64)
        except KeyError as exc:
            raise RuntimeError(f"Saved E79 neighbor is outside the verified full-Train OOF bank: {exc}") from exc
        if (np.any(bank_cycles[bank_indices] >= test_cycles[i])
                or [b01.audit.normalize_id(v) for v in neigh_ids[i, :count]]
                   != [b01.audit.normalize_id(bank_ids[j]) for j in bank_indices]
                or np.any(sims[i, :count] <= 0)
                or np.any(sims[i, :count] > 1.0 + 1e-12)
                or np.any(np.diff(sims[i, :count]) > 1e-15)
                or any(sims[i, j] == sims[i, j + 1] and chosen[j] > chosen[j + 1]
                       for j in range(count - 1))):
            raise RuntimeError("Saved E79 neighbor IDs/cycles/similarity order violate frozen geometry")

    ledger_before = b01._verify_ledger()
    if (ledger_before["sha256"] != EXPECTED_LEDGER_SHA256
            or ledger_before["starts"] != 66 or ledger_before["saved"] != 64
            or ledger_before["failed"] != 2 or ledger_before["pending"]):
        raise RuntimeError("Frozen research ledger is not the exact consumed 66-fit checkpoint")

    input_hashes: dict[str, str] = {}
    for relative in (*SOURCE_PATHS, *CONTROL_PATHS,
                     "data/raw/classification/Train.csv",
                     "data/raw/classification/Test.csv",
                     "submissions/optimization_v3/a02_continuation_v1/a02_continuation.csv"):
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"B01 benchmark dependency is missing: {relative}")
        input_hashes[relative] = _sha256_file(path)
    for cutoff, path in sorted(a02.CACHE_PATHS.items()):
        input_hashes[path.relative_to(ROOT).as_posix()] = _sha256_file(path)
        sidecar = path.with_suffix(".json")
        input_hashes[sidecar.relative_to(ROOT).as_posix()] = _sha256_file(sidecar)
    input_hashes[BASE_MANIFEST_PATH.relative_to(ROOT).as_posix()] = _sha256_file(BASE_MANIFEST_PATH)
    input_hashes[A02_CSV_PATH.relative_to(ROOT).as_posix()] = _sha256_file(A02_CSV_PATH)

    recipe = b01_freeze["recipe"]
    recipe_hash = canonical_sha256(recipe)
    if recipe_hash != canonical_sha256(b01.RECIPE):
        raise RuntimeError("B01 formula does not match its independently reviewed pre-score freeze")
    a02_prob = prediction["candidate_probability"].astype(np.float64, copy=False)
    a02_label_rebuilt = (a02_prob >= THRESHOLD).astype(np.int8)
    geometry_hashes = {
        "neighbor_count_sha256": _array_sha256(counts, "<i8"),
        "neighbor_source_rows_sha256": _array_sha256(neigh, "<i8"),
        "neighbor_ids_sha256": _array_sha256(neigh_ids),
        "neighbor_similarities_sha256": _array_sha256(sims, "<f8"),
        "baseline_probability_sha256": _array_sha256(baseline_probability, "<f8"),
        "missing_mask_sha256": _array_sha256(missing, np.bool_),
    }
    return {
        "resource_preflight": resource,
        "a02_freeze": a02_freeze,
        "b01_recipe_freeze": b01_freeze,
        "recipe": recipe,
        "recipe_sha256": recipe_hash,
        "a02_manifest": a02_manifest,
        "a02_receipt": receipt,
        "a02_state": state,
        "a02_model": model,
        "train": train,
        "caches": caches,
        "full_bank": full_bank,
        "bank_rows": bank_rows,
        "bank_ids": bank_ids,
        "bank_cycles": bank_cycles,
        "bank_oof": bank_oof,
        "bank_labels": bank_labels,
        "test": test,
        "test_header": test_header,
        "sample_columns": sample_columns,
        "test_ids": test_ids,
        "test_cycles": test_cycles,
        "row_positions": row_positions,
        "prediction": prediction,
        "baseline_probability": baseline_probability.astype(np.float64, copy=True),
        "a02_labels": a02_labels,
        "a02_csv_evidence": a02_csv_evidence,
        "a02_label_rebuilt": a02_label_rebuilt,
        "ordered_test_ids_sha256": ordered_ids_hash,
        "test_file_sha256": test_file_sha,
        "sample_schema_sha256": canonical_sha256(sample_columns),
        "input_sha256": input_hashes,
        "geometry_sha256": geometry_hashes,
        "ledger_before": ledger_before,
    }


def _freeze_recipe() -> dict[str, Any]:
    if (MANIFEST_PATH.exists() or RESULT_PATH.exists() or QA_PATH.exists() or SUMMARY_PATH.exists()
            or CSV_PATH.exists() or PREDICTIONS_PATH.exists()):
        raise RuntimeError("Benchmark outputs already exist; refusing a post-prediction freeze")
    with b01.forbid_training_and_inference_calls() as calls:
        data = _load_verified_inputs()
    if calls:
        raise RuntimeError(f"A fit/transform/predict call was attempted during benchmark freeze: {calls}")
    freeze = {
        "status": "B01_BENCHMARK_RECIPE_FROZEN_PRE_INFERENCE",
        "source_commit": _current_commit(),
        "recipe": data["recipe"],
        "recipe_sha256": data["recipe_sha256"],
        "benchmark_policy": {
            "qualification": "B01 remains DIAGNOSTIC_GATE_FAILED / DIAGNOSTIC_ONLY_NOT_QUALIFIED",
            "submission_authorized": False,
            "test_target_loaded": False,
            "kaggle_or_ui_access": False,
            "model_or_preprocessor_fit": False,
            "saved_geometry_only": True,
            "threshold": THRESHOLD,
            "threads": THREADS,
            "job_cap_bytes": JOB_CAP_BYTES,
            "free_ram_plus_reserve_bytes": FREE_RAM_AND_RESERVE_BYTES,
        },
        "official_train_rows": EXPECTED_TRAIN_ROWS,
        "official_train_sha256": EXPECTED_TRAIN_SHA256,
        "official_test_rows": EXPECTED_TEST_ROWS,
        "ordered_test_ids_sha256": data["ordered_test_ids_sha256"],
        "test_file_sha256": data["test_file_sha256"],
        "test_columns_loaded": list(ALLOWED_TEST_COLUMNS),
        "sample_schema_columns": data["sample_columns"],
        "sample_schema_sha256": data["sample_schema_sha256"],
        "a02_source": {
            "fit_id": data["a02_receipt"]["fit_id"],
            "model_sha256": _sha256_file(RUN_FOLDER / "model.joblib"),
            "state_sha256": canonical_sha256(data["a02_state"]),
            "receipt_sha256": data["a02_receipt"]["receipt_sha256"],
            "prediction_bundle_sha256": _sha256_file(A02_PREDICTIONS_PATH),
            "a02_probability_sha256": _array_sha256(data["prediction"]["candidate_probability"], "<f8"),
            "a02_label_sha256": _array_sha256(data["a02_label_rebuilt"], np.int8),
            "uploaded_csv_sha256": data["a02_csv_evidence"]["sha256"],
            "b01_recipe_freeze_sha256": _sha256_file(B01_RECIPE_FREEZE_PATH),
        },
        "oof_bank": {
            "blocks": data["a02_manifest"]["OOF_bank_blocks"],
            "present_rows": len(data["bank_rows"]),
            "row_positions_sha256": _array_sha256(data["bank_rows"], "<i8"),
            "ids_sha256": _array_sha256(data["bank_ids"]),
            "cycles_sha256": _array_sha256(data["bank_cycles"], "<i2"),
            "oof_probabilities_sha256": _array_sha256(data["bank_oof"], "<f8"),
            "labels_sha256": _array_sha256(data["bank_labels"], np.int8),
            "residuals_sha256": _array_sha256(data["a02_model"].bank_residuals, "<f8"),
        },
        "control": {
            "name": "full-Train Sparse25/C16 baseline probability",
            "cache_sha256": _sha256_file(BASE_CACHE_PATH),
            "manifest_sha256": _sha256_file(BASE_MANIFEST_PATH),
            "probability_sha256": _array_sha256(data["baseline_probability"], "<f8"),
            "missing_mask_sha256": _array_sha256(data["prediction"]["missing_mask"], np.bool_),
        },
        "saved_geometry_sha256": data["geometry_sha256"],
        "fit_ledger_before": data["ledger_before"],
        "input_sha256": data["input_sha256"],
    }
    _write_once(FREEZE_PATH, freeze)
    return freeze


def verify_freeze() -> tuple[dict[str, Any], dict[str, Any]]:
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if (freeze.get("status") != "B01_BENCHMARK_RECIPE_FROZEN_PRE_INFERENCE"
            or freeze.get("recipe_sha256") != canonical_sha256(freeze.get("recipe"))
            or freeze.get("recipe_sha256") != canonical_sha256(b01.RECIPE)
            or freeze.get("benchmark_policy", {}).get("submission_authorized") is not False
            or freeze.get("benchmark_policy", {}).get("test_target_loaded") is not False):
        raise RuntimeError("B01 benchmark pre-inference freeze differs from approved recipe/boundaries")
    for relative, expected in freeze.get("input_sha256", {}).items():
        path = ROOT / relative
        if not path.is_file() or _sha256_file(path) != expected:
            raise RuntimeError(f"B01 benchmark input hash changed after freeze: {relative}")
    if _sha256_file(B01_RECIPE_FREEZE_PATH) != freeze["a02_source"].get("b01_recipe_freeze_sha256"):
        raise RuntimeError("B01 source freeze changed after benchmark freeze")
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", freeze["source_commit"], _current_commit()],
                              cwd=ROOT, check=False, capture_output=True)
    if ancestor.returncode != 0:
        raise RuntimeError("Benchmark freeze source commit is not an ancestor of this checkout")
    with b01.forbid_training_and_inference_calls() as calls:
        data = _load_verified_inputs()
    if calls:
        raise RuntimeError(f"A fit/transform/predict call was attempted during benchmark verification: {calls}")
    if (data["recipe_sha256"] != freeze["recipe_sha256"]
            or data["input_sha256"] != freeze["input_sha256"]
            or data["ledger_before"] != freeze["fit_ledger_before"]
            or data["ordered_test_ids_sha256"] != freeze["ordered_test_ids_sha256"]
            or data["geometry_sha256"] != freeze["saved_geometry_sha256"]):
        raise RuntimeError("B01 benchmark validated inputs differ from frozen pre-inference manifest")
    return freeze, data


def _reconstruct_a02_first(data: dict[str, Any]) -> np.ndarray:
    """Rebuild the unchanged A02 probabilities before any B01 candidate calculation."""
    baseline = data["baseline_probability"]
    p_bank = data["bank_oof"]
    labels_bank = data["bank_labels"].astype(np.float64)
    bank_lookup = {int(row): i for i, row in enumerate(data["bank_rows"])}
    out = baseline.copy()
    saved = data["prediction"]
    for i, raw_count in enumerate(saved["neighbor_count"]):
        n = int(raw_count)
        if n == 0:
            continue
        rows = saved["neighbor_source_rows"][i, :n].astype(np.int64)
        indices = np.asarray([bank_lookup[int(row)] for row in rows], dtype=np.int64)
        similarities = saved["neighbor_similarities"][i, :n].astype(np.float64)
        weights = similarities * similarities
        residual = labels_bank[indices] - p_bank[indices]
        raw = float(np.dot(weights, residual) / (b01.SHRINKAGE + float(weights.sum())))
        correction = float(np.clip(raw, -b01.CORRECTION_LIMIT, b01.CORRECTION_LIMIT))
        out[i] = np.clip(baseline[i] + correction, 0.0, 1.0)
    out[data["prediction"]["missing_mask"].astype(bool)] = baseline[data["prediction"]["missing_mask"].astype(bool)]
    return out


def _independent_b01_verify(data: dict[str, Any], candidate: np.ndarray) -> dict[str, Any]:
    """Independent convex-formula replay; no query-label argument exists."""
    baseline = np.asarray(data["baseline_probability"], dtype=np.float64)
    neighbors = np.asarray(data["prediction"]["neighbor_source_rows"], dtype=np.int64)
    similarities = np.asarray(data["prediction"]["neighbor_similarities"], dtype=np.float64)
    counts = np.asarray(data["prediction"]["neighbor_count"], dtype=np.int64)
    missing = np.asarray(data["prediction"]["missing_mask"], dtype=bool)
    bank_rows = np.asarray(data["bank_rows"], dtype=np.int64)
    bank_labels = np.asarray(data["bank_labels"], dtype=np.float64)
    index = {int(row): i for i, row in enumerate(bank_rows)}
    independent = baseline.copy()
    for i in range(len(baseline)):
        n = int(counts[i])
        if n == 0:
            continue
        bank_index = np.asarray([index[int(row)] for row in neighbors[i, :n]], dtype=np.int64)
        weight = np.square(similarities[i, :n])
        mass = float(weight.sum())
        local_rate = float(np.dot(weight, bank_labels[bank_index]) / mass)
        shrunk = (b01.SHRINKAGE * baseline[i] + mass * local_rate) / (b01.SHRINKAGE + mass)
        correction = float(np.clip(shrunk - baseline[i], -b01.CORRECTION_LIMIT, b01.CORRECTION_LIMIT))
        independent[i] = float(np.clip(baseline[i] + correction, 0.0, 1.0))
    independent[missing] = baseline[missing]
    if (not np.isfinite(candidate).all() or np.any((candidate < 0) | (candidate > 1))
            or not np.allclose(independent, candidate, rtol=0, atol=1e-15)
            or not np.array_equal(independent[missing], baseline[missing])):
        raise RuntimeError("Independent convex B01 formula/clip/fallback replay differs")
    return {
        "status": "PASS", "independent_formula": "(4*b + sum(s^2*y_bank))/(4 + sum(s^2)); clip correction to +/-0.05, then probability to [0,1]",
        "max_abs_probability_difference": float(np.max(np.abs(independent - candidate))),
        "missing_stream_exact_baseline": True,
        "zero_neighbor_exact_baseline": bool(np.array_equal(independent[counts == 0], baseline[counts == 0])),
    }


def _prediction_archive_bytes(probability: np.ndarray, labels: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, probability=np.asarray(probability, dtype="<f8"),
                        prediction=np.asarray(labels, dtype=np.int8))
    return buffer.getvalue()


def _transition_summary(a02: np.ndarray, b01: np.ndarray, cycles: np.ndarray,
                        missing: np.ndarray) -> dict[str, Any]:
    changed = a02 != b01
    summary = {
        "overall": {
            "a02_0_to_b01_1": int(np.sum((a02 == 0) & (b01 == 1))),
            "a02_1_to_b01_0": int(np.sum((a02 == 1) & (b01 == 0))),
            "unchanged": int(np.sum(~changed)), "changed": int(np.sum(changed)),
        },
        "by_cycle_and_availability": [],
    }
    availability = np.where(missing, "missing", "present")
    for cycle in sorted(int(v) for v in np.unique(cycles)):
        for status in ("present", "missing"):
            mask = (cycles == cycle) & (availability == status)
            if not mask.any():
                continue
            summary["by_cycle_and_availability"].append({
                "cycle_code": cycle, "stream_availability": status, "rows": int(mask.sum()),
                "a02_0_to_b01_1": int(np.sum(mask & (a02 == 0) & (b01 == 1))),
                "a02_1_to_b01_0": int(np.sum(mask & (a02 == 1) & (b01 == 0))),
                "unchanged": int(np.sum(mask & ~changed)),
            })
    return summary


def _build_report(data: dict[str, Any], b01_probability: np.ndarray,
                  b01_labels: np.ndarray, independent: dict[str, Any],
                  a02_rebuilt: np.ndarray, ledger_after: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str]:
    a02_probability = data["prediction"]["candidate_probability"].astype(np.float64)
    missing = data["prediction"]["missing_mask"].astype(bool)
    labels = data["a02_labels"]
    transitions = _transition_summary(labels, b01_labels, data["test_cycles"], missing)
    probabilities = {}
    for name, p in (("A02_uploaded", a02_probability), ("B01", b01_probability),
                    ("Sparse25_control", data["baseline_probability"])):
        probabilities[name] = {
            "sha256_float64_ordered": _array_sha256(p, "<f8"),
            "min": float(np.min(p)), "max": float(np.max(p)), "mean": float(np.mean(p)),
            "exact_zero_count": int(np.sum(p == 0.0)), "exact_one_count": int(np.sum(p == 1.0)),
            "finite": bool(np.isfinite(p).all()), "within_0_1": bool(np.all((p >= 0) & (p <= 1))),
        }
    label_counts = {
        "A02_uploaded_prediction_counts": {str(v): int(np.sum(labels == v)) for v in (0, 1)},
        "B01_prediction_counts": {str(v): int(np.sum(b01_labels == v)) for v in (0, 1)},
        "Sparse25_control_prediction_counts": {
            str(v): int(np.sum((data["baseline_probability"] >= THRESHOLD).astype(np.int8) == v)) for v in (0, 1)},
        "target_prevalence_or_accuracy": None,
    }
    changed = transitions["overall"]["changed"]
    result = {
        "status": "PREPARED_FOR_REVIEW_NOT_SUBMITTED",
        "benchmark_value": "NO_NEW_SUBMISSION_VALUE" if changed == 0 else "LABELS_DIFFER_FROM_UPLOADED_A02",
        "qualification": "B01_DIAGNOSTIC_GATE_FAILED / DIAGNOSTIC_ONLY_NOT_QUALIFIED",
        "submission_authorized": False,
        "public_score_prediction": None,
        "formula": data["recipe"],
        "formula_sha256": data["recipe_sha256"],
        "rows": EXPECTED_TEST_ROWS,
        "columns": data["sample_columns"],
        "ordered_test_ids_sha256": data["ordered_test_ids_sha256"],
        "probability_summaries": probabilities,
        "predicted_class_counts": label_counts,
        "label_transitions_vs_already_uploaded_A02": transitions,
        "a02_original_probability_reconstruction": {
            "bit_exact_to_saved_A02": bool(np.array_equal(a02_rebuilt, a02_probability)),
            "a02_probability_sha256": _array_sha256(a02_probability, "<f8"),
            "reconstructed_a02_probability_sha256": _array_sha256(a02_rebuilt, "<f8"),
            "uploaded_a02_label_sha256": _array_sha256(labels, np.int8),
        },
        "b01_predictions": {
            "probability_sha256": _array_sha256(b01_probability, "<f8"),
            "label_sha256": _array_sha256(b01_labels, np.int8),
            "csv_sha256": _sha256_file(CSV_PATH),
            "probabilities_npz_sha256": _sha256_file(PREDICTIONS_PATH),
            "csv_path": CSV_PATH.relative_to(ROOT).as_posix(),
            "probabilities_path": PREDICTIONS_PATH.relative_to(ROOT).as_posix(),
        },
        "provenance": {
            "source_commit": _current_commit(),
            "freeze_sha256": _sha256_file(FREEZE_PATH),
            "official_train_sha256": EXPECTED_TRAIN_SHA256,
            "official_test_features_sha256": data["test_file_sha256"],
            "sample_schema_sha256": data["sample_schema_sha256"],
            "a02_model_sha256": _sha256_file(RUN_FOLDER / "model.joblib"),
            "a02_fit_state_sha256": canonical_sha256(data["a02_state"]),
            "a02_fit_receipt_sha256": data["a02_receipt"]["receipt_sha256"],
            "a02_prediction_bundle_sha256": _sha256_file(A02_PREDICTIONS_PATH),
            "uploaded_a02_csv": data["a02_csv_evidence"],
            "oof_bank_rows": len(data["bank_rows"]),
            "oof_bank_blocks": data["a02_manifest"]["OOF_bank_blocks"],
            "oof_bank_sha256": {"rows": _array_sha256(data["bank_rows"], "<i8"),
                                "ids": _array_sha256(data["bank_ids"]),
                                "cycles": _array_sha256(data["bank_cycles"], "<i2"),
                                "probabilities": _array_sha256(data["bank_oof"], "<f8"),
                                "labels": _array_sha256(data["bank_labels"], np.int8)},
            "sparse25_control_cache_sha256": _sha256_file(BASE_CACHE_PATH),
            "saved_geometry_sha256": data["geometry_sha256"],
            "input_sha256": data["input_sha256"],
            "test_targets_loaded": False,
            "sample_submission_read_for_schema_only": True,
            "a02_model_or_preprocessor_fit_or_predict_calls": 0,
            "kaggle_or_ui_access": False,
            "new_fit_ledger_entries": 0,
        },
    }
    qa = {
        "status": "PASS",
        "checks": {
            "pre_inference_freeze_formula_source_and_inputs_verified": True,
            "official_test_target_column_absent_and_not_loaded": True,
            "sample_submission_header_only_schema_check": True,
            "official_test_ordered_ids_29421_unique_match_saved_a02_and_baseline": True,
            "a02_fulltrain_model_state_receipt_hashes_verified": True,
            "fulltrain_sparse25_baseline_cache_and_route_verified": True,
            "all_six_oof_blocks_rebuilt_and_bank_y_rows_cycles_probabilities_verified": True,
            "a02_geometry_top32_ids_ties_padding_historical_cycles_verified": True,
            "original_a02_probabilities_reconstructed_bit_exact_before_b01": bool(np.array_equal(a02_rebuilt, a02_probability)),
            "a02_prediction_labels_match_already_uploaded_csv": bool(np.array_equal(labels, (a02_probability >= THRESHOLD).astype(np.int8))),
            "b01_formula_independent_convex_replay_with_cap_and_fallback": independent,
            "b01_probabilities_finite_range_and_ordered_hash": probabilities["B01"]["finite"] and probabilities["B01"]["within_0_1"],
            "submission_csv_schema_rows_ids_unique_and_integer_binary": True,
            "b01_npz_and_csv_hashes_match_prepared_manifest": True,
            "fit_ledger_hash_and_66_fit_accounting_unchanged": ledger_after == data["ledger_before"],
            "no_model_fit_transform_predict_or_neighbor_search": True,
            "no_test_target_accuracy_prevalence_or_score_claim": True,
            "no_kaggle_api_ui_upload_or_final_entry_selection": True,
            "b01_failed_scientific_qualification_preserved": True,
        },
        "fit_ledger_before": data["ledger_before"],
        "fit_ledger_after": ledger_after,
    }
    summary = _summary_text(result)
    return result, qa, summary


def _summary_text(result: dict[str, Any]) -> str:
    trans = result["label_transitions_vs_already_uploaded_A02"]["overall"]
    counts = result["predicted_class_counts"]
    p = result["probability_summaries"]
    lines = [
        "# B01 full-Train benchmark preparation",
        "",
        f"**Status:** {result['status']}  ",
        f"**B01 qualification:** {result['qualification']}  ",
        f"**Benchmark value:** {result['benchmark_value']}  ",
        "**Submission:** not authorized here; no Kaggle/API/UI access or upload was performed.",
        "",
        "This is a zero-fit inference preparation using the existing A02 E79 full-Train model, its saved top-32 geometry, verified Train-only C16 OOF bank, and full-Train Sparse25 probabilities. No Test target was loaded. These are prediction counts, not Test prevalence or accuracy; no public score is predicted.",
        "",
        "## Comparison with already-uploaded A02",
        "",
        f"Rows: {result['rows']:,}. Label changes: {trans['changed']:,} ({trans['a02_0_to_b01_1']:,} 0→1, {trans['a02_1_to_b01_0']:,} 1→0). Unchanged: {trans['unchanged']:,}.",
        f"Predicted labels: A02 {counts['A02_uploaded_prediction_counts']}; B01 {counts['B01_prediction_counts']}; Sparse25 {counts['Sparse25_control_prediction_counts']}.",
        f"B01 probability endpoints: 0={p['B01']['exact_zero_count']:,}, 1={p['B01']['exact_one_count']:,}; A02 endpoints: 0={p['A02_uploaded']['exact_zero_count']:,}, 1={p['A02_uploaded']['exact_one_count']:,}.",
        "Per-cycle and stream-availability transitions are in RESULTS.json.",
        "Recipe, fitted-state lineage, OOF bank, control geometry and input hashes are in MANIFEST.json; independent checks are in QA.json.",
        "",
        "B01 remains scientifically DIAGNOSTIC_GATE_FAILED / NOT_QUALIFIED: its primary F1 hurdle and four-of-five origin gate failed. This separately authorized benchmark preparation does not change that verdict or imply future/public-score benefit.",
        "",
        f"CSV: `{result['b01_predictions']['csv_path']}`  ",
        f"CSV SHA-256: `{result['b01_predictions']['csv_sha256']}`  ",
        f"B01 probability SHA-256: `{result['b01_predictions']['probability_sha256']}`  ",
        f"Source commit: `{result['provenance']['source_commit']}`.",
    ]
    return "\n".join(lines) + "\n"


def _prepare() -> tuple[dict[str, Any], dict[str, Any]]:
    freeze, data = verify_freeze()
    before = data["ledger_before"]
    saved = data["prediction"]
    # This reconstruction and equality check must pass before deriving B01.
    a02_rebuilt = _reconstruct_a02_first(data)
    a02_saved = saved["candidate_probability"].astype(np.float64)
    if (not np.array_equal(a02_rebuilt, a02_saved)
            or not np.array_equal(data["a02_labels"], (a02_rebuilt >= THRESHOLD).astype(np.int8))):
        raise RuntimeError("Original A02 prediction hash/labels failed exact reconstruction; B01 derivation stopped")
    a02_hash = _array_sha256(a02_rebuilt, "<f8")
    if a02_hash != freeze["a02_source"]["a02_probability_sha256"]:
        raise RuntimeError("Original A02 exact prediction hash differs from the pre-inference freeze")

    geometry = {
        "score_rows": data["row_positions"],
        "baseline_prediction": data["baseline_probability"],
        "neighbor_source_rows": saved["neighbor_source_rows"],
        "neighbor_similarities": saved["neighbor_similarities"],
        "neighbor_count": saved["neighbor_count"],
        "missing_mask": saved["missing_mask"],
    }
    assignment = np.arange(len(data["bank_rows"]), dtype=np.int64)
    with threadpool_limits(limits=THREADS), b01.forbid_training_and_inference_calls() as calls:
        recomputed = b01.compute_from_saved_geometry(
            geometry, data["bank_rows"], data["bank_labels"], data["bank_oof"], assignment)
    if calls:
        raise RuntimeError(f"A fit/transform/predict call was attempted during benchmark preparation: {calls}")
    if not np.array_equal(recomputed["a02_rebuilt"], a02_rebuilt):
        raise RuntimeError("B01 source's A02 reconstruction differs from the independently verified original")
    b01_probability = np.asarray(recomputed["b01"], dtype=np.float64)
    b01_labels = (b01_probability >= THRESHOLD).astype(np.int8)
    independent = _independent_b01_verify(data, b01_probability)

    sample_columns = data["sample_columns"]
    csv_frame = pd.DataFrame({"record_id": data["test_ids"], "artifact_signal": b01_labels})
    if (csv_frame.columns.tolist() != sample_columns or len(csv_frame) != EXPECTED_TEST_ROWS
            or csv_frame.record_id.isna().any() or csv_frame.record_id.duplicated().any()
            or csv_frame.artifact_signal.dtype != np.int8
            or not np.isin(csv_frame.artifact_signal.to_numpy(), [0, 1]).all()):
        raise RuntimeError("Prepared B01 CSV fails official Sample schema/Test identity checks")
    csv_bytes = csv_frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    predictions_bytes = _prediction_archive_bytes(b01_probability, b01_labels)
    # Persist prediction state first; an interrupted run can resume without any refit.
    _atomic_once(PREDICTIONS_PATH, predictions_bytes)
    _atomic_once(CSV_PATH, csv_bytes)
    if (_sha256_file(PREDICTIONS_PATH) != hashlib.sha256(predictions_bytes).hexdigest()
            or _sha256_file(CSV_PATH) != hashlib.sha256(csv_bytes).hexdigest()):
        raise RuntimeError("Durable benchmark output bytes differ from the validated predictions/CSV")

    after = b01._verify_ledger()
    if after != before:
        raise RuntimeError("Research fit ledger changed during zero-fit benchmark preparation")
    result, qa, summary = _build_report(data, b01_probability, b01_labels,
                                        independent, a02_rebuilt, after)
    result["provenance"]["benchmark_freeze_sha256"] = _sha256_file(FREEZE_PATH)
    result["provenance"]["source_commit"] = freeze["source_commit"]
    result["provenance"]["a02_prediction_reconstruction_before_b01"] = {
        "sha256": a02_hash, "bit_exact": True,
    }
    manifest = {
        "status": "PREPARED_FOR_REVIEW_NOT_SUBMITTED",
        "qualification": "B01_DIAGNOSTIC_GATE_FAILED / DIAGNOSTIC_ONLY_NOT_QUALIFIED",
        "submission_authorized": False,
        "source_commit": freeze["source_commit"],
        "benchmark_freeze_sha256": _sha256_file(FREEZE_PATH),
        "recipe": freeze["recipe"],
        "recipe_sha256": freeze["recipe_sha256"],
        "official_train_sha256": freeze["official_train_sha256"],
        "official_test_rows": freeze["official_test_rows"],
        "ordered_test_ids_sha256": freeze["ordered_test_ids_sha256"],
        "test_feature_sha256": freeze["test_file_sha256"],
        "sample_schema": freeze["sample_schema_columns"],
        "sample_schema_sha256": freeze["sample_schema_sha256"],
        "a02_model": freeze["a02_source"],
        "oof_bank": freeze["oof_bank"],
        "control": freeze["control"],
        "saved_geometry_sha256": freeze["saved_geometry_sha256"],
        "inputs_sha256": freeze["input_sha256"],
        "a02_prediction_reconstruction_before_b01": result["a02_original_probability_reconstruction"],
        "b01_output": result["b01_predictions"],
        "label_transitions_vs_uploaded_a02": result["label_transitions_vs_already_uploaded_A02"],
        "predicted_class_counts": result["predicted_class_counts"],
        "probability_summaries": result["probability_summaries"],
        "test_targets_loaded": False,
        "sample_submission_read_for_schema_only": True,
        "model_or_preprocessor_fit_transform_predict_calls": 0,
        "ledger_before_after": {"before": data["ledger_before"], "after": ledger_after},
        "kaggle_api_ui_or_upload": False,
    }
    # Rebuild summary after attaching the final freeze source lineage.
    summary = _summary_text(result)
    qa["freeze_sha256"] = _sha256_file(FREEZE_PATH)
    qa["source_commit"] = freeze["source_commit"]
    qa["manifest_sha256"] = hashlib.sha256(_json_bytes(manifest)).hexdigest()
    qa["result_sha256"] = hashlib.sha256(_json_bytes(result)).hexdigest()
    qa["summary_sha256"] = hashlib.sha256(summary.encode("utf-8")).hexdigest()
    if not all(value is True or (isinstance(value, dict) and value.get("status") == "PASS")
               for value in qa["checks"].values()):
        raise RuntimeError("Benchmark QA contains a failed check")
    _write_once(RESULT_PATH, result)
    _write_once(MANIFEST_PATH, manifest)
    _write_once(QA_PATH, qa)
    _write_once(SUMMARY_PATH, summary)
    return result, qa


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze", action="store_true", help="Freeze inputs and B01 formula before Test inference")
    mode.add_argument("--prepare", action="store_true", help="Prepare B01 benchmark CSV with saved state only")
    args = parser.parse_args()
    with threadpool_limits(limits=THREADS):
        if args.freeze:
            freeze = _freeze_recipe()
            print(json.dumps({"status": freeze["status"], "source_commit": freeze["source_commit"],
                              "recipe_sha256": freeze["recipe_sha256"],
                              "input_count": len(freeze["input_sha256"]),
                              "train_sha256": freeze["official_train_sha256"],
                              "test_rows": freeze["official_test_rows"],
                              "ledger": freeze["fit_ledger_before"]}, indent=2))
        else:
            result, qa = _prepare()
            print(json.dumps({"status": result["status"], "benchmark_value": result["benchmark_value"],
                              "transitions": result["label_transitions_vs_already_uploaded_A02"]["overall"],
                              "class_counts": result["predicted_class_counts"],
                              "csv": result["b01_predictions"]["csv_path"],
                              "csv_sha256": result["b01_predictions"]["csv_sha256"],
                              "QA": qa["status"], "checks": len(qa["checks"])}, indent=2))


if __name__ == "__main__":
    main()
