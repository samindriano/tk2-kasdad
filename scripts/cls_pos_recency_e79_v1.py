"""One-shot E79 recency gamma fit and frozen feature-only application."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import psutil
from scipy import sparse
from scipy.special import expit
from threadpoolctl import threadpool_limits

AUDIT_ROOT = Path(__file__).resolve().parents[1]
if str(AUDIT_ROOT) not in sys.path:
    sys.path.insert(0, str(AUDIT_ROOT))

from src.models import cls_pos_recency_e79_v1 as core


KASDAD_ROOT = AUDIT_ROOT.parent
MAIN_ROOT = KASDAD_ROOT / "TP 2"
REPORT_DIR = AUDIT_ROOT / "reports/simplification_v1/CLS_POS_RECENCY_E79_V1"
EXPERIMENT_DIR = AUDIT_ROOT / "experiments/CLS_POS_RECENCY_E79_V1"
SUBMISSION_DIR = AUDIT_ROOT / "submissions/cls_pos_recency_e79_v1"
INVENTORY_PATH = REPORT_DIR / "SOURCE_INVENTORY.json"
PROTOCOL_PATH = REPORT_DIR / "PROTOCOL.md"
ROOT_DECISION_PATH = AUDIT_ROOT / "reports/simplification_v1/CLS_POS_RECENCY_V1/ROOT_RESULTS_DECISION.md"
ROOT_DECISION_COMMIT = "7a8ad5b6437db1c3bb8d716fb879252935615e34"
OFFICIAL_TRAIN_REL = Path("data/raw/classification/Train.csv")
OFFICIAL_TEST_REL = Path("data/raw/classification/Test.csv")
SAMPLE_REL = Path("data/raw/classification/SampleSubmission.csv")
TARGET_COLUMN = "artifact_signal"
OFFICIAL_TRAIN_SHA256 = "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c"
OFFICIAL_TEST_SHA256 = "4e6c381d582ed24f92d17f620f4f0ad25e2e0def29eb57a91a5482867f6afe3c"
SAMPLE_SHA256 = "889f97d69659a64945421d841dcf96e5813020606f29b2992d4b9560cecdcba7"
CHAMPION_CSV_REL = Path(
    "submissions/optimization_v3/cls_a02_missing_temporal_decision_v2/"
    "classification_a02_missing_temporal_e79.csv"
)
CHAMPION_CSV_SHA256 = "189a779b6100333a2fecfe443689fe7d340a331e828167221c81281a7ece4a4a"
EXPECTED_ROWS = 50608
EXPECTED_TEST_ROWS = 29421
MIN_FREE_BEFORE_FIT = 1024**3
MIN_FREE_AFTER_FIT = 512 * 1024**2
THREAD_LIMIT = 2
SOURCE_FILES = (
    "src/models/cls_pos_recency_e79_v1.py",
    "scripts/cls_pos_recency_e79_v1.py",
    "scripts/cls_pos_recency_e79_main_bridge_v1.py",
    "scripts/verify_cls_pos_recency_e79_v1.py",
    "tests/test_cls_pos_recency_e79_v1.py",
    "reports/simplification_v1/CLS_POS_RECENCY_E79_V1/PROTOCOL.md",
    "reports/simplification_v1/CLS_POS_RECENCY_E79_V1/SOURCE_INVENTORY.json",
    "reports/simplification_v1/CLS_POS_RECENCY_E79_V1/SYNTHETIC_QA_FINAL.json",
    "reports/simplification_v1/CLS_POS_RECENCY_E79_V1/TEST_SUITE_QA_FINAL.json",
)
FREEZE_REL = "reports/simplification_v1/CLS_POS_RECENCY_E79_V1/FREEZE.json"
PREFIT_QA_REL = "reports/simplification_v1/CLS_POS_RECENCY_E79_V1/PREFIT_QA.json"
RESERVATION_REL = "reports/simplification_v1/CLS_POS_RECENCY_E79_V1/RESERVATION.json"
FIT_ATTEMPT_REL = "experiments/CLS_POS_RECENCY_E79_V1/ATTEMPT.started.json"
GAMMA_REL = "experiments/CLS_POS_RECENCY_E79_V1/GAMMA_E79.npz"
FIT_RECEIPT_REL = "experiments/CLS_POS_RECENCY_E79_V1/FIT_RECEIPT.json"
FIT_FAILURE_REL = "experiments/CLS_POS_RECENCY_E79_V1/FIT_FAILURE.json"
TRAIN_FEATURE_DIR_REL = "experiments/CLS_POS_RECENCY_E79_V1/train_features"
TEST_FEATURE_DIR_REL = "experiments/CLS_POS_RECENCY_E79_V1/test_features"
SYNTHETIC_QA_REL = "reports/simplification_v1/CLS_POS_RECENCY_E79_V1/SYNTHETIC_QA_FINAL.json"
TEST_SUITE_QA_REL = "reports/simplification_v1/CLS_POS_RECENCY_E79_V1/TEST_SUITE_QA_FINAL.json"
FIT_QA_REL = "reports/simplification_v1/CLS_POS_RECENCY_E79_V1/FIT_QA.json"
APPLICATION_QA_REL = "reports/simplification_v1/CLS_POS_RECENCY_E79_V1/APPLICATION_QA.json"
APPLICATION_MANIFEST_REL = "reports/simplification_v1/CLS_POS_RECENCY_E79_V1/APPLICATION_MANIFEST.json"
APPLICATION_ATTEMPT_REL = "experiments/CLS_POS_RECENCY_E79_V1/APPLICATION_ATTEMPT.started.json"
APPLICATION_FAILURE_REL = "experiments/CLS_POS_RECENCY_E79_V1/APPLICATION_FAILURE.json"
APPLICATION_OUTPUT_REL = "experiments/CLS_POS_RECENCY_E79_V1/APPLICATION_OUTPUTS.npz"
APPLICATION_RECEIPT_REL = "experiments/CLS_POS_RECENCY_E79_V1/APPLICATION_RECEIPT.json"
CSV_REL = "submissions/cls_pos_recency_e79_v1/classification_cls_pos_recency_e79_v1.csv"
A02_TEST_NPZ_REL = Path(
    "experiments/runs/optimization_v3_a02_continuation_v1/E79/fulltrain_observed/test_predictions.npz"
)
A02_TEST_RECEIPT_REL = Path(
    "experiments/runs/optimization_v3_a02_continuation_v1/E79/fulltrain_observed/test_predictions.json"
)
A02_FIT_RECEIPT_REL = Path(
    "experiments/runs/optimization_v3_a02_continuation_v1/E79/fulltrain_observed/fit_receipt.json"
)
A02_MODEL_REL = Path(
    "experiments/runs/optimization_v3_a02_continuation_v1/E79/fulltrain_observed/model.joblib"
)
UNIFORM_GAMMA_REL = Path(
    "experiments/runs/optimization_v3_a02_cgeom_benchmark_v1/units/"
    "A02_CGEOM_BENCHMARK_V1_E79_POS_BAG_OFFSET_GAMMA_INTEGRATION_R1.joblib"
)
UNIFORM_GAMMA_RECEIPT_REL = Path(
    "experiments/runs/optimization_v3_a02_cgeom_benchmark_v1/units/"
    "A02_CGEOM_BENCHMARK_V1_E79_POS_BAG_OFFSET_GAMMA_INTEGRATION_R1.json"
)
UNIFORM_GAMMA_FIT_ID = "A02_CGEOM_BENCHMARK_V1_E79_POS_BAG_OFFSET_GAMMA_INTEGRATION_R1"
UNIFORM_GAMMA_ARTIFACT_SHA256 = "d741d016f990d8742208edc919f5c58305b7579980d059b202609df4e5b3e650"
UNIFORM_GAMMA_RECEIPT_SHA256 = "f08f1bd3733a20fb8745d40c82190543a48af6bade1df5684ccfef3198c34580"
UNIFORM_GAMMA_VECTOR_SHA256 = "2171d72a0168f0ce43980a9c416fd64d673e1a7d9e9a213b226f843b55bf1bb1"


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_text_sha256(path: Path) -> str:
    """Hash committed text independent of Windows checkout line-ending conversion."""
    payload = Path(path).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _main_source_sha256_array(value: Any) -> str:
    """Reproduce the frozen MAIN positional helper's array hash convention."""
    array = np.asarray(value)
    header = canonical_sha256({"shape": array.shape, "dtype": str(array.dtype)})
    digest = hashlib.sha256(header.encode("ascii"))
    if array.dtype.kind in "OUS":
        digest.update(json.dumps(array.astype(str).tolist(), ensure_ascii=False,
                                 separators=(",", ":")).encode("utf-8"))
    else:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _main_source_sha256_csr(value: sparse.spmatrix) -> str:
    """Reproduce the frozen MAIN positional helper's canonical CSR hash."""
    matrix = sparse.csr_matrix(value, dtype=np.float64, copy=True)
    matrix.sum_duplicates()
    matrix.sort_indices()
    header = canonical_sha256({"shape": matrix.shape, "dtype": str(matrix.dtype)})
    digest = hashlib.sha256(header.encode("ascii"))
    digest.update(np.asarray(matrix.indptr, dtype="<i8").tobytes())
    digest.update(np.asarray(matrix.indices, dtype="<i8").tobytes())
    digest.update(np.asarray(matrix.data, dtype="<f8").tobytes())
    return digest.hexdigest()


def write_once(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite durable output: {path}")
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as tmp:
        temp_path = Path(tmp.name)
    try:
        with temp_path.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    return file_sha256(path)


def write_json_once(path: Path, payload: dict[str, Any]) -> str:
    return write_once(
        path,
        (json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8"),
    )


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def git(*args: str, root: Path = AUDIT_ROOT, check: bool = True) -> str:
    result = subprocess.run(["git", "-C", str(root), *args], check=False,
                            capture_output=True, text=True)
    if check and result.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _committed_file_sha(commit: str, relative: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(AUDIT_ROOT), "show", f"{commit}:{relative}"],
        check=False, capture_output=True,
    )
    if result.returncode:
        raise RuntimeError(f"Frozen source is absent from commit {commit}: {relative}")
    normalized = result.stdout.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(normalized).hexdigest()


def _inventory_body() -> dict[str, Any]:
    inventory = read_json(INVENTORY_PATH)
    body = dict(inventory)
    signature = body.pop("inventory_sha256", None)
    if signature != canonical_sha256(body):
        raise RuntimeError("Source inventory self-hash mismatch")
    if inventory.get("status") != "PASS_STATIC_AND_FULLTRAIN_FEATURE_SOURCE_INVENTORY":
        raise RuntimeError("Source inventory is not a passing E79 inventory")
    if not all(inventory.get("checks", {}).values()):
        raise RuntimeError("One or more source inventory checks failed")
    return inventory


def _verify_root_authority_and_inputs() -> dict[str, Any]:
    inventory = _inventory_body()
    if not ROOT_DECISION_PATH.is_file():
        raise RuntimeError("Committed root E79 decision is absent")
    if "APPROVE_EXPLORATORY_E79_BENCHMARK" not in ROOT_DECISION_PATH.read_text(encoding="utf-8"):
        raise RuntimeError("Root decision does not authorize the exploratory E79 benchmark")
    if subprocess.run(
        ["git", "-C", str(AUDIT_ROOT), "merge-base", "--is-ancestor",
         ROOT_DECISION_COMMIT, git("rev-parse", "HEAD")],
        check=False,
    ).returncode:
        raise RuntimeError("Root authorization commit is not an ancestor of this checkout")
    current_main_head = git("rev-parse", "HEAD", root=MAIN_ROOT)
    if current_main_head != inventory.get("main_head"):
        raise RuntimeError("MAIN TP2 HEAD differs from the pinned source inventory")
    if Path(inventory.get("main_root", "")).resolve() != MAIN_ROOT.resolve():
        raise RuntimeError("Pinned MAIN root differs from this checkout's configured TP2 root")
    train_path = MAIN_ROOT / OFFICIAL_TRAIN_REL
    if not train_path.is_file() or file_sha256(train_path) != OFFICIAL_TRAIN_SHA256:
        raise RuntimeError("Official Train file is absent or differs from the authorized hash")
    if inventory.get("official_train_sha256") != OFFICIAL_TRAIN_SHA256:
        raise RuntimeError("Source inventory binds a different official Train")
    pinned = inventory.get("pinned_file_hashes", {})
    if len(pinned) != 51:
        raise RuntimeError(f"Expected 51 pinned MAIN source/input hashes, found {len(pinned)}")
    verified_pins: dict[str, str] = {}
    for pin_id, record in sorted(pinned.items()):
        if not (pin_id.startswith("input:") or pin_id.startswith("source:")):
            raise RuntimeError(f"Malformed pinned MAIN hash identity: {pin_id}")
        relative_path = pin_id.split(":", 1)[1]
        source_path = MAIN_ROOT / Path(relative_path)
        expected = record.get("expected")
        if (record.get("match") is not True or record.get("actual") != expected
                or not source_path.is_file() or file_sha256(source_path) != expected):
            raise RuntimeError(f"Pinned MAIN source/input changed: {source_path}")
        verified_pins[pin_id] = expected
    return {
        "inventory_sha256": inventory["inventory_sha256"],
        "inventory_file_sha256": source_text_sha256(INVENTORY_PATH),
        "root_decision_file_sha256": source_text_sha256(ROOT_DECISION_PATH),
        "root_decision_commit": ROOT_DECISION_COMMIT,
        "main_root": str(MAIN_ROOT),
        "main_head": current_main_head,
        "official_train_path": str(train_path),
        "official_train_sha256": OFFICIAL_TRAIN_SHA256,
        "official_train_rows": EXPECTED_ROWS,
        "pinned_main_file_count": len(pinned),
        "pinned_main_file_hashes": verified_pins,
        "test_or_sample_access": False,
        "target_column_read": False,
    }


def _freeze_body() -> dict[str, Any]:
    source_hashes = {rel: source_text_sha256(AUDIT_ROOT / rel) for rel in SOURCE_FILES}
    body = {
        "schema_version": 1,
        "candidate_id": core.CANDIDATE_ID,
        "fit_id": core.FIT_ID,
        "status": "FROZEN_ONE_E79_RECENCY_GAMMA_FIT",
        "prepared_from_head": git("rev-parse", "HEAD"),
        "source_file_sha256": source_hashes,
        "input_lineage": _verify_root_authority_and_inputs(),
        "full_train_matrix": {
            "rows": EXPECTED_ROWS,
            "base": _inventory_body()["r3_base_matrix"],
            "position": _inventory_body()["r3_position_matrix"],
            "ordered_ids_sha256": _inventory_body()["ordered_train_ids_sha256"],
        },
        "original_uniform_gamma": _inventory_body()["original_uniform_gamma"],
        "fit_cap": 1,
        "fit_population": {
            "official_train_rows": EXPECTED_ROWS,
            "cycles": [47, 79],
            "gamma_fit_rows": "stream-present subset only",
            "missing_stream_rows_fit": False,
        },
        "recipe": {
            "base_weight": "max(0.25, 2**(-(79-cycle_code)/8))",
            "normalization": "divide by mean base weight separately within each target class on stream-present fit rows",
            "offset": "frozen C2_B positive-class logit z_BAG = X_base @ coef_[0] + intercept_[0]",
            "objective": "sum_i w_i*logaddexp(0,-(2*y_i-1)*(z_BAG_i + X_position_i@gamma)) + 0.5*dot(gamma,gamma)",
            "optimizer": "the frozen CLS_POS_RECENCY_V1 float64 L-BFGS-B helper; zero initialization; maxiter=2000, ftol=1e-12, gtol=1e-6, maxls=50",
            "preprocessing_fit_count": 0,
            "control_or_comparator_fit_count": 0,
            "test_threshold_selection": False,
        },
        "application": {
            "authorized_after_fit_verification": True,
            "test_rows": EXPECTED_TEST_ROWS,
            "test_cycles": [80, 91],
            "test_target_read": False,
            "sample_schema_only": True,
            "present_probability": "0.5*sigmoid(z_BAG + X_position@gamma_E79) + 0.5*p_A02",
            "present_threshold": core.PRESENT_THRESHOLD,
            "missing_probability": "frozen p_A02 copied bit-exactly",
            "missing_threshold": core.MISSING_THRESHOLD,
            "csv_upload": False,
            "kaggle_contact": False,
        },
        "resource_policy": {
            "one_heavy_job_at_a_time": True,
            "maximum_threads": THREAD_LIMIT,
            "minimum_free_before_fit_bytes": MIN_FREE_BEFORE_FIT,
            "minimum_free_after_fit_bytes": MIN_FREE_AFTER_FIT,
            "optimizer_starts_maximum": 1,
            "retry_after_optimizer_start": False,
        },
        "boundaries": {
            "target_read_during_prefit": False,
            "test_or_sample_access_during_prefit": False,
            "root_decision_is_exploratory_only": True,
            "scientific_gate_changed": False,
            "champion_changed": False,
        },
    }
    return body


def build_freeze() -> dict[str, Any]:
    if git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Tracked worktree must be clean before the one-fit freeze")
    source_commit = git("rev-parse", "HEAD")
    body = _freeze_body()
    if body["prepared_from_head"] != source_commit:
        raise RuntimeError("Freeze source HEAD changed during preparation")
    if set(body["source_file_sha256"]) != set(SOURCE_FILES):
        raise RuntimeError("Freeze source-file set is incomplete")
    for rel, expected in body["source_file_sha256"].items():
        if _committed_file_sha(source_commit, rel) != expected:
            raise RuntimeError(f"Freeze source is not committed exactly: {rel}")
    synthetic = read_json(AUDIT_ROOT / SYNTHETIC_QA_REL)
    test_suite = read_json(AUDIT_ROOT / TEST_SUITE_QA_REL)
    if synthetic.get("status") != "PASS_SYNTHETIC_E79_LIFECYCLE":
        raise RuntimeError("Synthetic lifecycle QA did not pass")
    if test_suite.get("status") != "PASS_UNIT_TESTS" or test_suite.get("passed", 0) <= 0:
        raise RuntimeError("E79 unit tests are not recorded as passing")
    prefit = {
        "schema_version": 1,
        "candidate_id": core.CANDIDATE_ID,
        "status": "PASS_E79_PREFIT_SOURCE_SYNTHETIC_QA",
        "source_commit": source_commit,
        "source_files_sha256": body["source_file_sha256"],
        "source_inventory_sha256": body["input_lineage"]["inventory_sha256"],
        "root_decision_sha256": body["input_lineage"]["root_decision_file_sha256"],
        "official_train_sha256": OFFICIAL_TRAIN_SHA256,
        "full_train_feature_matrix_parity": body["full_train_matrix"],
        "synthetic_qa_file_sha256": file_sha256(AUDIT_ROOT / SYNTHETIC_QA_REL),
        "unit_test_qa_file_sha256": file_sha256(AUDIT_ROOT / TEST_SUITE_QA_REL),
        "optimizer_starts": 0,
        "target_column_read": False,
        "test_or_sample_access": False,
        "kaggle_contact": False,
        "preprocessing_fits": 0,
    }
    prefit["prefit_qa_sha256"] = canonical_sha256(prefit)
    body["prefit_qa_sha256"] = prefit["prefit_qa_sha256"]
    body["freeze_sha256"] = canonical_sha256(body)
    freeze_path = AUDIT_ROOT / FREEZE_REL
    prefit_path = AUDIT_ROOT / PREFIT_QA_REL
    reservation_path = AUDIT_ROOT / RESERVATION_REL
    if any(path.exists() for path in (freeze_path, prefit_path, reservation_path)):
        raise RuntimeError("Refusing to replace an existing E79 prefit freeze/reservation")
    reservation = {
        "schema_version": 1,
        "candidate_id": core.CANDIDATE_ID,
        "fit_ids": [core.FIT_ID],
        "predictor_fit_cap": 1,
        "predictor_fit_starts": 0,
        "preprocessing_fit_cap": 0,
        "root_authorization_commit": ROOT_DECISION_COMMIT,
        "freeze_sha256": body["freeze_sha256"],
        "status": "ONE_E79_FIT_RESERVED",
        "retry_after_fit_start": False,
    }
    reservation["reservation_sha256"] = canonical_sha256(reservation)
    write_json_once(prefit_path, prefit)
    write_json_once(freeze_path, body)
    write_json_once(reservation_path, reservation)
    return {"status": "PASS_E79_PREFIT_SOURCE_SYNTHETIC_QA",
            "freeze_sha256": body["freeze_sha256"],
            "source_commit": source_commit,
            "fit_cap": 1}


def verify_freeze() -> tuple[dict[str, Any], dict[str, Any]]:
    freeze_path = AUDIT_ROOT / FREEZE_REL
    prefit_path = AUDIT_ROOT / PREFIT_QA_REL
    reservation_path = AUDIT_ROOT / RESERVATION_REL
    freeze = read_json(freeze_path)
    body = dict(freeze)
    signature = body.pop("freeze_sha256", None)
    if signature != canonical_sha256(body):
        raise RuntimeError("E79 freeze self-hash mismatch")
    if freeze.get("candidate_id") != core.CANDIDATE_ID or freeze.get("fit_cap") != 1:
        raise RuntimeError("E79 freeze identity or one-fit cap mismatch")
    current_head = git("rev-parse", "HEAD")
    source_commit = freeze.get("prepared_from_head")
    if subprocess.run(
        ["git", "-C", str(AUDIT_ROOT), "merge-base", "--is-ancestor", source_commit, current_head],
        check=False,
    ).returncode:
        raise RuntimeError("Frozen E79 source commit is not an ancestor of the current checkout")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Tracked E79 source changed after freeze")
    for rel, expected in freeze.get("source_file_sha256", {}).items():
        if source_text_sha256(AUDIT_ROOT / rel) != expected or _committed_file_sha(source_commit, rel) != expected:
            raise RuntimeError(f"Frozen E79 source mismatch: {rel}")
    authority = _verify_root_authority_and_inputs()
    if authority != freeze.get("input_lineage"):
        raise RuntimeError("Root/input lineage changed after E79 freeze")
    prefit = read_json(prefit_path)
    prefit_body = dict(prefit)
    prefit_sha = prefit_body.pop("prefit_qa_sha256", None)
    if prefit_sha != canonical_sha256(prefit_body) or prefit_sha != freeze.get("prefit_qa_sha256"):
        raise RuntimeError("E79 prefit QA self-hash or freeze binding mismatch")
    reservation = read_json(reservation_path)
    reservation_body = dict(reservation)
    reservation_sha = reservation_body.pop("reservation_sha256", None)
    if reservation_sha != canonical_sha256(reservation_body):
        raise RuntimeError("E79 fit reservation self-hash mismatch")
    if (reservation.get("fit_ids") != [core.FIT_ID]
            or reservation.get("predictor_fit_cap") != 1
            or reservation.get("freeze_sha256") != signature):
        raise RuntimeError("E79 one-fit reservation differs from its freeze")
    return freeze, prefit


def _run_bridge(mode: str, output_dir: Path) -> dict[str, Any]:
    bridge = AUDIT_ROOT / "scripts/cls_pos_recency_e79_main_bridge_v1.py"
    result = subprocess.run(
        [sys.executable, str(bridge), "--mode", mode, "--output-dir", str(output_dir)],
        cwd=MAIN_ROOT, check=False, capture_output=True, text=True,
    )
    if result.returncode:
        raise RuntimeError(f"Feature bridge failed ({mode}): {result.stderr[-4000:]}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def _load_feature_cache(mode: str) -> tuple[sparse.csr_matrix, dict[str, np.ndarray], dict[str, Any]]:
    stem = "TRAIN" if mode == "train" else "TEST"
    directory = AUDIT_ROOT / (TRAIN_FEATURE_DIR_REL if mode == "train" else TEST_FEATURE_DIR_REL)
    matrix_path = directory / f"{stem}_X_POSITION.npz"
    arrays_path = directory / f"{stem}_FEATURE_ARRAYS.npz"
    receipt_path = directory / f"{stem}_TRANSFORM_RECEIPT.json"
    if any(not path.is_file() for path in (matrix_path, arrays_path, receipt_path)):
        raise RuntimeError(f"{mode} feature-only transform cache is incomplete")
    receipt = read_json(receipt_path)
    receipt_body = dict(receipt)
    receipt_signature = receipt_body.pop("receipt_sha256", None)
    if receipt_signature != canonical_sha256(receipt_body):
        raise RuntimeError(f"{mode} feature transform receipt signature mismatch")
    if receipt.get("mode") != mode or receipt.get("status") != "PASS_FEATURE_ONLY_TRANSFORM":
        raise RuntimeError(f"{mode} feature transform receipt identity/status mismatch")
    if (file_sha256(matrix_path) != receipt["position_matrix"]["cache_file_sha256"]
            or file_sha256(arrays_path) != receipt["arrays_cache_file_sha256"]):
        raise RuntimeError(f"{mode} feature cache hash mismatch")
    x = sparse.load_npz(matrix_path).tocsr().astype(np.float64)
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    required = {"record_ids", "cycles", "present", "z_bag"}
    if not required.issubset(arrays):
        raise RuntimeError(f"{mode} feature cache schema is incomplete")
    n = len(arrays["record_ids"])
    if (x.shape[0] != n or len(arrays["cycles"]) != n or len(arrays["present"]) != n
            or len(arrays["z_bag"]) != n or not np.isfinite(x.data).all()
            or not np.isfinite(arrays["z_bag"]).all()):
        raise RuntimeError(f"{mode} feature cache arrays are misaligned")
    logical_sha = _main_source_sha256_csr(x)
    matrix_meta = receipt.get("position_matrix", {})
    recorded_logical_sha = matrix_meta.get("logical_sha256")
    if mode == "train":
        expected_logical_sha = _inventory_body()["r3_position_matrix"]["logical_sha256"]
        if recorded_logical_sha != expected_logical_sha or logical_sha != expected_logical_sha:
            raise RuntimeError("Cached full-Train positional matrix differs from frozen logical identity")
        expected_ids_sha = _inventory_body()["ordered_train_ids_sha256"]
        if _main_source_sha256_array(arrays["record_ids"].astype(str)) != expected_ids_sha:
            raise RuntimeError("Cached full-Train ordered IDs differ from frozen identity")
    elif recorded_logical_sha is not None and recorded_logical_sha != logical_sha:
        raise RuntimeError("Cached Test positional matrix differs from its transform receipt")
    for field, array_key in (("cycles_sha256", "cycles"), ("present_mask_sha256", "present"),
                             ("z_bag_sha256", "z_bag")):
        if _main_source_sha256_array(arrays[array_key]) != receipt.get(field):
            raise RuntimeError(f"{mode} feature cache {array_key} differs from its transform receipt")
    if _main_source_sha256_array(arrays["record_ids"].astype(str)) != receipt.get("ordered_ids_sha256"):
        raise RuntimeError(f"{mode} feature cache IDs differ from their transform receipt")
    return x, arrays, receipt


def synthetic_lifecycle(output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "gamma.npz").exists() or (output_dir / "receipt.json").exists():
        raise RuntimeError("Synthetic lifecycle refuses to overwrite prior output")
    rng = np.random.default_rng(20261107)
    x = sparse.csr_matrix(rng.normal(size=(32, 5)).astype(np.float64))
    cycles = np.resize(np.asarray([47, 55, 67, 75, 79], dtype=np.int64), 32)
    target = np.asarray(([0, 1] * 16), dtype=np.int8)
    offset = rng.normal(scale=0.2, size=32).astype(np.float64)
    weights = core.normalized_class_weights(cycles, target)
    gamma, metadata = core.fit_gamma(x, offset, target, weights)
    gamma_path = output_dir / "gamma.npz"
    with gamma_path.open("xb") as stream:
        np.savez_compressed(stream, gamma_float64=gamma.astype(np.float64, copy=False))
    with np.load(gamma_path, allow_pickle=False) as archive:
        gamma_readback = np.asarray(archive["gamma_float64"]).copy()
    if not np.array_equal(gamma, gamma_readback):
        raise RuntimeError("Synthetic persisted gamma failed bit-exact readback")
    try:
        with gamma_path.open("xb") as stream:
            stream.write(b"forbidden-overwrite")
        raise RuntimeError("Synthetic persistence unexpectedly allowed an overwrite")
    except FileExistsError:
        pass
    # Resume/readback is a pure load of the prior result; no second optimizer call.
    with np.load(gamma_path, allow_pickle=False) as archive:
        resumed_gamma = np.asarray(archive["gamma_float64"]).copy()
    if not np.array_equal(resumed_gamma, gamma_readback):
        raise RuntimeError("Synthetic resume readback differs from the persisted vector")
    p_a02 = rng.uniform(0.1, 0.9, size=32)
    present = np.asarray(([True, False, True, True] * 8), dtype=bool)
    p_pos = core.positional_probability(x, offset, gamma_readback)
    probability = core.route_probabilities(present, p_a02, p_pos)
    labels = core.apply_frozen_thresholds(probability, present)
    if not np.array_equal(probability[~present], p_a02[~present]):
        raise RuntimeError("Synthetic missing fallback is not exact")
    receipt = {
        "status": "PASS_SYNTHETIC_E79_LIFECYCLE",
        "candidate_id": core.CANDIDATE_ID,
        "synthetic_only": True,
        "official_train_or_target_read": False,
        "test_or_sample_access": False,
        "optimizer_fit_count_synthetic_only": 1,
        "official_optimizer_starts": 0,
        "gamma_shape": list(gamma.shape),
        "gamma_sha256": array_sha256(gamma),
        "gamma_file_sha256": file_sha256(gamma_path),
        "gamma_readback_exact": True,
        "overwrite_refused": True,
        "resume_reloaded_without_refit": True,
        "resume_optimizer_starts": 0,
        "fit_metadata": metadata,
        "missing_probability_exact": True,
        "label_values": sorted(np.unique(labels).astype(int).tolist()),
        "weight_class_mass": {
            str(label): float(weights[target == label].sum()) for label in (0, 1)
        },
    }
    write_json_once(output_dir / "receipt.json", receipt)
    return receipt


def run_synthetic() -> dict[str, Any]:
    qa_path = AUDIT_ROOT / SYNTHETIC_QA_REL
    if qa_path.exists():
        raise RuntimeError("Refusing to overwrite synthetic lifecycle QA output")
    temp_root = Path(tempfile.mkdtemp(prefix="cls_pos_recency_e79_synthetic_"))
    try:
        receipt = synthetic_lifecycle(temp_root)
        body = dict(receipt)
        body["temporary_outputs_removed"] = True
        body["synthetic_receipt_sha256"] = file_sha256(temp_root / "receipt.json")
        return {**body, "report_path": str(qa_path)}
    finally:
        import shutil
        shutil.rmtree(temp_root, ignore_errors=True)


def run_test_suite() -> dict[str, Any]:
    qa_path = AUDIT_ROOT / TEST_SUITE_QA_REL
    if qa_path.exists():
        raise RuntimeError("Refusing to overwrite final unit-test QA output")
    test_path = AUDIT_ROOT / "tests/test_cls_pos_recency_e79_v1.py"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(test_path)],
        cwd=AUDIT_ROOT, check=False, capture_output=True, text=True,
    )
    output = (result.stdout + "\n" + result.stderr).strip()
    passed = 0
    for line in output.splitlines():
        if " passed" in line:
            try:
                passed = int(line.strip().split()[0])
            except (ValueError, IndexError):
                pass
    body = {
        "schema_version": 1,
        "status": "PASS_UNIT_TESTS" if result.returncode == 0 and passed > 0 else "FAIL_UNIT_TESTS",
        "command": [sys.executable, "-m", "pytest", "-q", str(test_path)],
        "returncode": int(result.returncode),
        "passed": passed,
        "captured_output": output[-10000:],
        "target_or_test_data_access": False,
    }
    write_json_once(qa_path, body)
    if body["status"] != "PASS_UNIT_TESTS":
        raise RuntimeError("E79 unit tests failed; official fit remains unstarted")
    return body


def _write_gamma_once(path: Path, gamma: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite official gamma artifact: {path}")
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as tmp:
        temp = Path(tmp.name)
    try:
        with temp.open("wb") as stream:
            np.savez_compressed(stream, gamma_float64=np.asarray(gamma, dtype=np.float64))
        os.replace(temp, path)
    except Exception:
        temp.unlink(missing_ok=True)
        raise
    return file_sha256(path)


def fit_once() -> dict[str, Any]:
    freeze, prefit = verify_freeze()
    attempt_path = AUDIT_ROOT / FIT_ATTEMPT_REL
    gamma_path = AUDIT_ROOT / GAMMA_REL
    receipt_path = AUDIT_ROOT / FIT_RECEIPT_REL
    failure_path = AUDIT_ROOT / FIT_FAILURE_REL
    if any(path.exists() for path in (attempt_path, gamma_path, receipt_path, failure_path)):
        raise RuntimeError("E79 fit is one-shot and a fit attempt/output already exists")
    bridge_receipt = _run_bridge("train", AUDIT_ROOT / TRAIN_FEATURE_DIR_REL)
    x, arrays, transform_receipt = _load_feature_cache("train")
    if (bridge_receipt["receipt_sha256"] != transform_receipt["receipt_sha256"]
            or transform_receipt.get("target_column_requested_or_loaded") is not False
            or transform_receipt.get("mode") != "train"
            or transform_receipt.get("position_matrix", {}).get("logical_sha256")
            != freeze["full_train_matrix"]["position"]["logical_sha256"]):
        raise RuntimeError("Post-freeze Train feature build does not match committed source freeze")
    present = arrays["present"].astype(bool, copy=False)
    cycles = arrays["cycles"].astype(np.int64, copy=False)
    ids = arrays["record_ids"].astype(str, copy=False)
    if not np.all((cycles >= 47) & (cycles <= 79)) or len(ids) != EXPECTED_ROWS:
        raise RuntimeError("E79 fit identities/cycles escape the frozen Train population")
    free_before = int(psutil.virtual_memory().available)
    if free_before < MIN_FREE_BEFORE_FIT:
        return {"status": "BLOCKED_RESOURCE_GATE", "attempt_marker_written": False,
                "available_ram_bytes": free_before}

    # The target-only read occurs only after all frozen source and matrix checks.
    target_frame = pd.read_csv(MAIN_ROOT / OFFICIAL_TRAIN_REL, usecols=[TARGET_COLUMN], low_memory=False)
    if list(target_frame.columns) != [TARGET_COLUMN] or len(target_frame) != EXPECTED_ROWS:
        raise RuntimeError("Target-only Train reader returned the wrong population")
    target_all = pd.to_numeric(target_frame[TARGET_COLUMN], errors="raise").to_numpy()
    if target_all.dtype.kind not in "biuf" or not np.isfinite(target_all).all() or not np.isin(target_all, [0, 1]).all():
        raise RuntimeError("Official Train target is not finite binary 0/1")
    target_all = target_all.astype(np.int8, copy=False)
    target = target_all[present]
    fit_cycles = cycles[present]
    x_fit = x[present].tocsr()
    offset = arrays["z_bag"][present].astype(np.float64, copy=False)
    weights = core.normalized_class_weights(fit_cycles, target)
    if len(np.unique(target)) != 2 or x_fit.shape[0] != len(target):
        raise RuntimeError("E79 present fit population lacks both classes or is misaligned")
    for label in (0, 1):
        if not np.isclose(weights[target == label].sum(), np.sum(target == label), rtol=1e-12, atol=1e-10):
            raise RuntimeError(f"E79 class-{label} weight mass differs from its row count")

    main_modules = _load_main_lock_modules()
    with ExitStack() as stack:
        r3 = main_modules["r3"]
        locks = [
            stack.enter_context(r3._shared_job_lock()),
            stack.enter_context(r3._cgeom_local_lock(MAIN_ROOT)),
        ]
        free_locked = int(psutil.virtual_memory().available)
        if free_locked < MIN_FREE_BEFORE_FIT:
            return {"status": "BLOCKED_RESOURCE_GATE", "attempt_marker_written": False,
                    "available_ram_bytes": free_locked}
        attempt = {
            "schema_version": 1,
            "candidate_id": core.CANDIDATE_ID,
            "fit_id": core.FIT_ID,
            "status": "ONE_E79_RECENCY_OPTIMIZER_STARTED",
            "source_commit": git("rev-parse", "HEAD"),
            "freeze_sha256": freeze["freeze_sha256"],
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "predictor_fit_slot_consumed": True,
            "official_optimizer_starts": 1,
            "target_vector_reads": 1,
            "target_rows_read": EXPECTED_ROWS,
            "fit_rows_stream_present": int(present.sum()),
            "test_or_sample_access": False,
            "kaggle_contact": False,
            "locks_acquired": locks,
            "resource_before": {
                "available_ram_bytes": free_locked,
                "thread_limit": THREAD_LIMIT,
            },
        }
        write_json_once(attempt_path, attempt)
        try:
            with threadpool_limits(limits=THREAD_LIMIT):
                gamma, metadata = core.fit_gamma(x_fit, offset, target, weights)
            gamma_file_sha = _write_gamma_once(gamma_path, gamma)
            class_weights = {}
            for label in (0, 1):
                mask = target == label
                values = weights[mask]
                ess = float(values.sum() ** 2 / np.dot(values, values))
                class_weights[str(label)] = {
                    "rows": int(mask.sum()),
                    "weight_sum": float(values.sum()),
                    "ess": ess,
                    "ess_ratio": ess / int(mask.sum()),
                    "weight_min": float(values.min()),
                    "weight_max": float(values.max()),
                }
            free_after = int(psutil.virtual_memory().available)
            receipt = {
                "schema_version": 1,
                "candidate_id": core.CANDIDATE_ID,
                "fit_id": core.FIT_ID,
                "status": "PASS_E79_RECENCY_GAMMA_FIT_PERSISTED" if metadata.get("success") else "FAIL_E79_RECENCY_GAMMA_FIT_NO_RETRY",
                "mode": "official",
                "source_commit": git("rev-parse", "HEAD"),
                "source_freeze_sha256": freeze["freeze_sha256"],
                "prefit_qa_sha256": prefit["prefit_qa_sha256"],
                "attempt_file_sha256": file_sha256(attempt_path),
                "gamma_artifact_path": GAMMA_REL,
                "gamma_artifact_file_sha256": gamma_file_sha,
                "gamma_sha256": array_sha256(gamma),
                "gamma_shape": list(gamma.shape),
                "gamma_dtype": str(gamma.dtype),
                "optimizer": metadata,
                "objective": "sum_i w_i*logaddexp(0,-(2*y_i-1)*(z_BAG_i + X_position_i@gamma)) + 0.5*dot(gamma,gamma)",
                "C": 1.0,
                "population": {
                    "official_train_rows": EXPECTED_ROWS,
                    "cycles": [47, 79],
                    "stream_present_rows": int(present.sum()),
                    "stream_missing_rows_excluded": int((~present).sum()),
                    "class_rows": {"0": int(np.sum(target == 0)), "1": int(np.sum(target == 1))},
                },
                "read_accounting": {
                    "official_train_feature_only_reads": 1,
                    "official_train_target_vector_reads": 1,
                    "target_vector_rows_read": EXPECTED_ROWS,
                    "target_rows_used_by_gamma_fit": int(len(target)),
                    "test_target_reads": 0,
                    "sample_submission_reads": 0,
                },
                "inputs": {
                    "official_train_sha256": OFFICIAL_TRAIN_SHA256,
                    "record_ids_sha256": array_sha256(ids),
                    "cycles_sha256": array_sha256(cycles),
                    "present_mask_sha256": array_sha256(present),
                    "fit_target_sha256": array_sha256(target),
                    "base_matrix_sha256": freeze["full_train_matrix"]["base"]["logical_sha256"],
                    "position_matrix_sha256": freeze["full_train_matrix"]["position"]["logical_sha256"],
                    "position_fit_values_sha256": array_sha256(x_fit.data),
                    "position_fit_indices_sha256": array_sha256(x_fit.indices),
                    "position_fit_indptr_sha256": array_sha256(x_fit.indptr),
                    "z_bag_fit_sha256": array_sha256(offset),
                    "normalized_weights_sha256": array_sha256(weights),
                    "weight_diagnostics_by_class": class_weights,
                    "original_uniform_gamma_sha256": freeze["original_uniform_gamma"]["gamma_vector_sha256"],
                },
                "resource_after": {"available_ram_bytes": free_after},
                "resource_reserve_pass": free_after >= MIN_FREE_AFTER_FIT,
                "retry_authorized": False,
                "test_or_sample_access": False,
                "kaggle_contact": False,
            }
            receipt["receipt_sha256"] = canonical_sha256(receipt)
            write_json_once(receipt_path, receipt)
        except Exception as exc:
            failure = {
                "schema_version": 1,
                "candidate_id": core.CANDIDATE_ID,
                "fit_id": core.FIT_ID,
                "status": "FAIL_E79_RECENCY_GAMMA_FIT_NO_RETRY",
                "source_freeze_sha256": freeze["freeze_sha256"],
                "official_optimizer_starts": 1,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "test_or_sample_access": False,
                "retry_authorized": False,
            }
            if not failure_path.exists():
                write_json_once(failure_path, failure)
            raise
    return {"status": receipt["status"], "gamma_sha256": receipt["gamma_sha256"],
            "receipt_sha256": receipt["receipt_sha256"], "fit_rows": len(target),
            "test_or_sample_access": False}


def _signed_body_hash(value: dict[str, Any], signature_key: str) -> dict[str, Any]:
    body = dict(value)
    signature = body.pop(signature_key, None)
    if signature != canonical_sha256(body):
        raise RuntimeError(f"Self-hash mismatch for {signature_key}")
    return value


def _write_npz_once(path: Path, **arrays: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite one-shot application array bundle: {path}")
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz.tmp", delete=False) as tmp:
        temp_path = Path(tmp.name)
    try:
        with temp_path.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    return file_sha256(path)


def _write_classification_csv_once(path: Path, record_ids: np.ndarray, labels: np.ndarray) -> str:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite local E79 CSV: {path}")
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["record_id", TARGET_COLUMN])
        for record_id, label in zip(record_ids.astype(str), labels.astype(np.int8), strict=True):
            writer.writerow([record_id, int(label)])
        stream.flush()
        os.fsync(stream.fileno())
    return file_sha256(path)


def apply_test_once() -> dict[str, Any]:
    verifier_path = AUDIT_ROOT / "scripts/verify_cls_pos_recency_e79_v1.py"
    verify_result = subprocess.run(
        [sys.executable, str(verifier_path), "--fit"],
        cwd=AUDIT_ROOT, check=False, capture_output=True, text=True,
    )
    if verify_result.returncode:
        raise RuntimeError(f"Independent fit QA failed; Test remains unopened: {verify_result.stderr[-4000:]}")
    fit_qa = read_json(AUDIT_ROOT / FIT_QA_REL)
    _signed_body_hash(fit_qa, "fit_qa_sha256")
    if fit_qa.get("status") != "PASS_INDEPENDENT_E79_FIT_QA":
        raise RuntimeError("Feature-only Test application requires an independent fit-QA PASS")
    freeze, prefit = verify_freeze()
    attempt_path = AUDIT_ROOT / APPLICATION_ATTEMPT_REL
    failure_path = AUDIT_ROOT / APPLICATION_FAILURE_REL
    outputs_path = AUDIT_ROOT / APPLICATION_OUTPUT_REL
    receipt_path = AUDIT_ROOT / APPLICATION_RECEIPT_REL
    csv_path = AUDIT_ROOT / CSV_REL
    manifest_path = AUDIT_ROOT / APPLICATION_MANIFEST_REL
    qa_path = AUDIT_ROOT / APPLICATION_QA_REL
    if any(path.exists() for path in (attempt_path, failure_path, outputs_path, receipt_path,
                                      csv_path, manifest_path, qa_path)):
        raise RuntimeError("E79 Test application is one-shot and a prior attempt/output exists")
    available_before = int(psutil.virtual_memory().available)
    if available_before < MIN_FREE_BEFORE_FIT:
        return {"status": "BLOCKED_RESOURCE_GATE", "attempt_marker_written": False,
                "available_ram_bytes": available_before, "test_or_sample_access": False}

    # This durable marker consumes the sole application attempt before Test is read.
    attempt = {
        "schema_version": 1,
        "status": "ONE_E79_APPLICATION_STARTED",
        "candidate_id": core.CANDIDATE_ID,
        "source_commit": git("rev-parse", "HEAD"),
        "source_freeze_sha256": freeze["freeze_sha256"],
        "fit_qa_sha256": fit_qa["fit_qa_sha256"],
        "fit_receipt_sha256": file_sha256(AUDIT_ROOT / FIT_RECEIPT_REL),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "application_slot_consumed": True,
        "test_feature_transform_count_cap": 1,
        "csv_preparations_cap": 1,
        "available_ram_before_bytes": available_before,
        "test_target_or_sample_target_read": False,
        "kaggle_contact_or_upload": False,
    }
    write_json_once(attempt_path, attempt)
    try:
        test_bridge = _run_bridge("test", AUDIT_ROOT / TEST_FEATURE_DIR_REL)
        x, arrays, test_transform = _load_feature_cache("test")
        if (test_bridge.get("receipt_sha256") != test_transform.get("receipt_sha256")
                or test_transform.get("target_column_requested_or_loaded") is not False
                or test_transform.get("sample_submission_access") is not False
                or test_transform.get("mode") != "test"
                or file_sha256(MAIN_ROOT / OFFICIAL_TEST_REL) != OFFICIAL_TEST_SHA256):
            raise RuntimeError("Feature-only official Test transform identity/boundary failed")
        ids = arrays["record_ids"].astype(str)
        cycles = arrays["cycles"].astype(np.int64)
        present = arrays["present"].astype(bool)
        offset = arrays["z_bag"].astype(np.float64)
        if len(ids) != EXPECTED_TEST_ROWS or len(set(ids.tolist())) != EXPECTED_TEST_ROWS:
            raise RuntimeError("Official Test IDs are missing or duplicated")
        if not np.array_equal(np.unique(cycles), np.arange(80, 92, dtype=np.int64)):
            raise RuntimeError("Official Test cycle population is not complete 80-91")

        # Verify the already-fitted A02 test cache and its frozen fit lineage.
        a02_npz = MAIN_ROOT / A02_TEST_NPZ_REL
        a02_test_receipt_path = MAIN_ROOT / A02_TEST_RECEIPT_REL
        a02_fit_receipt_path = MAIN_ROOT / A02_FIT_RECEIPT_REL
        a02_model_path = MAIN_ROOT / A02_MODEL_REL
        a02_test_receipt = read_json(a02_test_receipt_path)
        a02_fit_receipt = read_json(a02_fit_receipt_path)
        if (a02_test_receipt.get("official_test_targets_loaded") is not False
                or a02_test_receipt.get("npz_sha256") != file_sha256(a02_npz)
                or a02_test_receipt.get("fit_receipt_sha256") != a02_fit_receipt.get("receipt_sha256")
                or a02_test_receipt.get("freeze_signature") != a02_fit_receipt.get("freeze_signature")
                or a02_fit_receipt.get("model_sha256") != file_sha256(a02_model_path)
                or a02_fit_receipt.get("test_data_used") is not False
                or a02_fit_receipt.get("query_labels_passed_to_fit") is not False):
            raise RuntimeError("Frozen A02 Test probability cache has invalid lineage/label boundary")
        with np.load(a02_npz, allow_pickle=False) as archive:
            required = {"record_ids", "missing_mask", "candidate_probability"}
            if not required.issubset(archive.files):
                raise RuntimeError("Frozen A02 probability cache fields are incomplete")
            a02_ids = np.asarray(archive["record_ids"]).astype(str)
            a02_missing = np.asarray(archive["missing_mask"], dtype=bool)
            p_a02 = np.asarray(archive["candidate_probability"], dtype=np.float64).reshape(-1)
        if (not np.array_equal(a02_ids, ids) or not np.array_equal(a02_missing, ~present)
                or p_a02.shape != (EXPECTED_TEST_ROWS,) or not np.isfinite(p_a02).all()
                or np.any((p_a02 < 0.0) | (p_a02 > 1.0))):
            raise RuntimeError("Frozen A02 predictions do not align to Test IDs/presence or probability range")

        # The original uniform gamma is reference-only; verify its exact lineage.
        inventory = _inventory_body()
        uniform_path = MAIN_ROOT / UNIFORM_GAMMA_REL
        uniform_receipt_path = MAIN_ROOT / UNIFORM_GAMMA_RECEIPT_REL
        uniform_meta = inventory["original_uniform_gamma"]
        if (file_sha256(uniform_path) != UNIFORM_GAMMA_ARTIFACT_SHA256
                or file_sha256(uniform_receipt_path) != UNIFORM_GAMMA_RECEIPT_SHA256
                or uniform_meta.get("artifact_sha256") != UNIFORM_GAMMA_ARTIFACT_SHA256
                or uniform_meta.get("receipt_file_sha256") != UNIFORM_GAMMA_RECEIPT_SHA256
                or uniform_meta.get("fit_id") != UNIFORM_GAMMA_FIT_ID
                or uniform_meta.get("fit_rows") != EXPECTED_ROWS
                or uniform_meta.get("fit_matrix_sha256") != freeze["full_train_matrix"]["position"]["logical_sha256"]
                or uniform_meta.get("gamma_vector_sha256") != UNIFORM_GAMMA_VECTOR_SHA256):
            raise RuntimeError("Original uniform gamma differs from the frozen full-Train champion lineage")
        import joblib
        uniform_state = joblib.load(uniform_path)
        if not isinstance(uniform_state, dict) or "gamma" not in uniform_state:
            raise RuntimeError("Original uniform gamma artifact schema differs from its integration receipt")
        gamma_uniform = np.asarray(uniform_state["gamma"], dtype=np.float64).reshape(-1)
        if gamma_uniform.shape != (x.shape[1],) or _main_source_sha256_array(gamma_uniform) != UNIFORM_GAMMA_VECTOR_SHA256:
            raise RuntimeError("Original uniform gamma vector differs from frozen identity/position dimension")
        with np.load(AUDIT_ROOT / GAMMA_REL, allow_pickle=False) as archive:
            gamma = np.asarray(archive["gamma_float64"], dtype=np.float64).copy()
        p_e79 = core.positional_probability(x, offset, gamma)
        p_uniform = core.positional_probability(x, offset, gamma_uniform)
        p_output = core.route_probabilities(present, p_a02, p_e79)
        labels = core.apply_frozen_thresholds(p_output, present)
        p_champion = core.route_probabilities(present, p_a02, p_uniform)
        champion_labels = core.apply_frozen_thresholds(p_champion, present)

        champion_csv = MAIN_ROOT / CHAMPION_CSV_REL
        if file_sha256(champion_csv) != CHAMPION_CSV_SHA256:
            raise RuntimeError("Frozen original A02/missing-temporal champion CSV hash changed")
        import csv
        with champion_csv.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ["record_id", TARGET_COLUMN]:
                raise RuntimeError("Frozen champion CSV schema mismatch")
            champion_rows = list(reader)
        champion_ids = np.asarray([row["record_id"] for row in champion_rows], dtype=str)
        champion_labels_csv = np.asarray([int(row[TARGET_COLUMN]) for row in champion_rows], dtype=np.int8)
        if (not np.array_equal(champion_ids, ids) or len(set(champion_ids.tolist())) != EXPECTED_TEST_ROWS
                or not np.array_equal(champion_labels_csv, champion_labels)):
            raise RuntimeError("Reconstructed original full champion differs from the frozen champion CSV")

        # SampleSubmission is read for exact ordered IDs and schema only.
        sample_path = MAIN_ROOT / SAMPLE_REL
        with sample_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            header = next(reader)
            if header != ["record_id", TARGET_COLUMN]:
                raise RuntimeError("SampleSubmission header differs from frozen classification schema")
            sample_ids = []
            for row in reader:
                if len(row) != 2:
                    raise RuntimeError("SampleSubmission row schema is malformed")
                sample_ids.append(row[0])
        if file_sha256(sample_path) != SAMPLE_SHA256 or not np.array_equal(np.asarray(sample_ids, dtype=str), ids):
            raise RuntimeError("SampleSubmission IDs/order do not match official Test")

        output_file_sha = _write_npz_once(
            outputs_path,
            record_ids=ids.astype("U"), cycles=cycles, present=present,
            p_a02=p_a02.astype(np.float64), p_e79=p_e79.astype(np.float64),
            p_uniform=p_uniform.astype(np.float64), p_output=p_output.astype(np.float64),
            labels=labels.astype(np.int8), champion_labels=champion_labels.astype(np.int8),
        )
        csv_file_sha = _write_classification_csv_once(csv_path, ids, labels)
        application_body = {
            "schema_version": 1,
            "status": "PASS_E79_FEATURE_ONLY_APPLICATION_PERSISTED",
            "candidate_id": core.CANDIDATE_ID,
            "source_commit": git("rev-parse", "HEAD"),
            "source_freeze_sha256": freeze["freeze_sha256"],
            "prefit_qa_sha256": prefit["prefit_qa_sha256"],
            "fit_qa_sha256": fit_qa["fit_qa_sha256"],
            "fit_receipt_sha256": file_sha256(AUDIT_ROOT / FIT_RECEIPT_REL),
            "application_attempt_file_sha256": file_sha256(attempt_path),
            "test_transform_receipt_sha256": test_transform["receipt_sha256"],
            "test_transform_matrix_logical_sha256": _main_source_sha256_csr(x),
            "test_transform_arrays_file_sha256": test_transform["arrays_cache_file_sha256"],
            "official_test_sha256": file_sha256(MAIN_ROOT / OFFICIAL_TEST_REL),
            "a02_test_npz_sha256": file_sha256(a02_npz),
            "a02_test_receipt_sha256": file_sha256(a02_test_receipt_path),
            "a02_fit_receipt_sha256": file_sha256(a02_fit_receipt_path),
            "a02_model_file_sha256": file_sha256(a02_model_path),
            "uniform_gamma_artifact_sha256": file_sha256(uniform_path),
            "uniform_gamma_receipt_sha256": file_sha256(uniform_receipt_path),
            "original_champion_csv_sha256": file_sha256(champion_csv),
            "sample_submission_sha256": file_sha256(sample_path),
            "application_arrays_path": APPLICATION_OUTPUT_REL,
            "application_arrays_sha256": output_file_sha,
            "csv_path": CSV_REL,
            "csv_sha256": csv_file_sha,
            "rows": EXPECTED_TEST_ROWS,
            "present_rows": int(present.sum()),
            "missing_rows": int((~present).sum()),
            "candidate_positive_count": int(labels.sum()),
            "champion_positive_count": int(champion_labels.sum()),
            "label_flips_vs_champion": int(np.count_nonzero(labels != champion_labels)),
            "missing_probability_bit_exact_to_a02": bool(np.array_equal(p_output[~present].view(np.uint64), p_a02[~present].view(np.uint64))),
            "missing_labels_bit_exact_to_champion": bool(np.array_equal(labels[~present], champion_labels[~present])),
            "recipe": freeze["recipe"],
            "target_column_read": False,
            "test_target_read": False,
            "sample_target_read": False,
            "optimizer_or_preprocessing_fit_after_freeze": False,
            "kaggle_contact_or_upload": False,
        }
        application_body["receipt_sha256"] = canonical_sha256(application_body)
        write_json_once(receipt_path, application_body)
        manifest = {
            "schema_version": 1,
            "candidate_id": core.CANDIDATE_ID,
            "status": "LOCAL_CSV_PREPARED_NOT_UPLOADED",
            "source_commit": application_body["source_commit"],
            "root_decision_commit": ROOT_DECISION_COMMIT,
            "freeze_sha256": freeze["freeze_sha256"],
            "source_inventory_sha256": freeze["input_lineage"]["inventory_sha256"],
            "fit_qa_sha256": fit_qa["fit_qa_sha256"],
            "fit_receipt_sha256": application_body["fit_receipt_sha256"],
            "input_hashes": {
                "official_train_sha256": OFFICIAL_TRAIN_SHA256,
                "official_test_sha256": application_body["official_test_sha256"],
                "sample_submission_sha256": application_body["sample_submission_sha256"],
                "a02_test_npz_sha256": application_body["a02_test_npz_sha256"],
                "a02_test_receipt_sha256": application_body["a02_test_receipt_sha256"],
                "a02_fit_receipt_sha256": application_body["a02_fit_receipt_sha256"],
                "uniform_gamma_artifact_sha256": application_body["uniform_gamma_artifact_sha256"],
                "uniform_gamma_receipt_sha256": application_body["uniform_gamma_receipt_sha256"],
                "original_champion_csv_sha256": CHAMPION_CSV_SHA256,
            },
            "recipe": application_body["recipe"],
            "csv_path": CSV_REL,
            "csv_sha256": csv_file_sha,
            "manifest_file_bindings": {
                "application_receipt_sha256": file_sha256(receipt_path),
                "application_arrays_sha256": output_file_sha,
            },
            "submission_uploaded": False,
            "kaggle_contacted": False,
            "test_or_sample_target_read": False,
        }
        manifest["manifest_sha256"] = canonical_sha256(manifest)
        write_json_once(manifest_path, manifest)
        qa_result = subprocess.run(
            [sys.executable, str(verifier_path), "--application"],
            cwd=AUDIT_ROOT, check=False, capture_output=True, text=True,
        )
        if qa_result.returncode:
            raise RuntimeError(f"Independent CSV/application QA failed: {qa_result.stderr[-4000:]}")
        return {
            "status": "PASS_LOCAL_CSV_PREPARED_NOT_UPLOADED",
            "csv_path": CSV_REL,
            "csv_sha256": csv_file_sha,
            "manifest_path": APPLICATION_MANIFEST_REL,
            "application_qa_path": APPLICATION_QA_REL,
            "rows": EXPECTED_TEST_ROWS,
            "label_flips_vs_champion": int(np.count_nonzero(labels != champion_labels)),
            "test_target_or_sample_target_read": False,
            "kaggle_contact_or_upload": False,
        }
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "candidate_id": core.CANDIDATE_ID,
            "status": "FAIL_E79_APPLICATION_NO_RETRY",
            "source_freeze_sha256": freeze["freeze_sha256"],
            "fit_qa_sha256": fit_qa["fit_qa_sha256"],
            "application_attempt_file_sha256": file_sha256(attempt_path),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "kaggle_contact_or_upload": False,
        }
        if not failure_path.exists():
            write_json_once(failure_path, failure)
        raise


def _load_main_lock_modules() -> dict[str, Any]:
    if str(MAIN_ROOT) not in sys.path:
        sys.path.insert(0, str(MAIN_ROOT))
    import scripts as audit_scripts
    if hasattr(audit_scripts, "__path__") and str(MAIN_ROOT / "scripts") not in list(audit_scripts.__path__):
        audit_scripts.__path__.insert(0, str(MAIN_ROOT / "scripts"))
    import src as audit_src
    if hasattr(audit_src, "__path__") and str(MAIN_ROOT / "src") not in list(audit_src.__path__):
        audit_src.__path__.insert(0, str(MAIN_ROOT / "src"))
    import src.models as audit_models
    if hasattr(audit_models, "__path__") and str(MAIN_ROOT / "src/models") not in list(audit_models.__path__):
        audit_models.__path__.insert(0, str(MAIN_ROOT / "src/models"))
    r3 = importlib.import_module("scripts.optimization_v3_cls_pos_global_strength_e79_v3")
    return {"r3": r3}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--synthetic", action="store_true")
    group.add_argument("--test-suite", action="store_true")
    group.add_argument("--build-freeze", action="store_true")
    group.add_argument("--fit-once", action="store_true")
    group.add_argument("--apply-test-once", action="store_true")
    args = parser.parse_args(argv)
    if args.synthetic:
        result = run_synthetic()
        write_json_once(AUDIT_ROOT / SYNTHETIC_QA_REL, result)
    elif args.test_suite:
        result = run_test_suite()
    elif args.build_freeze:
        result = build_freeze()
    elif args.fit_once:
        result = fit_once()
    else:
        result = apply_test_once()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
