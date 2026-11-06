"""Independent read-only E79 fit and local application verifier.

The fit verifier re-derives recency weights, objective, and gradient without
calling an optimizer. The application verifier checks feature-only lineage,
frozen route arithmetic, CSV schema, and full-Test ID alignment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit


AUDIT_ROOT = Path(__file__).resolve().parents[1]
MAIN_ROOT = AUDIT_ROOT.parent / "TP 2"
REPORT_ROOT = AUDIT_ROOT / "reports/simplification_v1/CLS_POS_RECENCY_E79_V1"
EXPERIMENT_ROOT = AUDIT_ROOT / "experiments/CLS_POS_RECENCY_E79_V1"
OUTPUT_ROOT = AUDIT_ROOT / "submissions/cls_pos_recency_e79_v1"
INVENTORY_PATH = REPORT_ROOT / "SOURCE_INVENTORY.json"
FREEZE_PATH = REPORT_ROOT / "FREEZE.json"
PREFIT_PATH = REPORT_ROOT / "PREFIT_QA.json"
RESERVATION_PATH = REPORT_ROOT / "RESERVATION.json"
ATTEMPT_PATH = EXPERIMENT_ROOT / "ATTEMPT.started.json"
GAMMA_PATH = EXPERIMENT_ROOT / "GAMMA_E79.npz"
FIT_RECEIPT_PATH = EXPERIMENT_ROOT / "FIT_RECEIPT.json"
TRAIN_FEATURE_DIR = EXPERIMENT_ROOT / "train_features"
TEST_FEATURE_DIR = EXPERIMENT_ROOT / "test_features"
TEST_CSV = MAIN_ROOT / "data/raw/classification/Test.csv"
SAMPLE_CSV = MAIN_ROOT / "data/raw/classification/SampleSubmission.csv"
CHAMPION_CSV = MAIN_ROOT / (
    "submissions/optimization_v3/cls_a02_missing_temporal_decision_v2/"
    "classification_a02_missing_temporal_e79.csv"
)
A02_ROOT = MAIN_ROOT / "experiments/runs/optimization_v3_a02_continuation_v1/E79/fulltrain_observed"
A02_TEST_NPZ = A02_ROOT / "test_predictions.npz"
A02_TEST_RECEIPT = A02_ROOT / "test_predictions.json"
A02_FIT_RECEIPT = A02_ROOT / "fit_receipt.json"
UNIFORM_GAMMA_PATH = MAIN_ROOT / (
    "experiments/runs/optimization_v3_a02_cgeom_benchmark_v1/units/"
    "A02_CGEOM_BENCHMARK_V1_E79_POS_BAG_OFFSET_GAMMA_INTEGRATION_R1.joblib"
)
UNIFORM_GAMMA_RECEIPT = MAIN_ROOT / (
    "experiments/runs/optimization_v3_a02_cgeom_benchmark_v1/units/"
    "A02_CGEOM_BENCHMARK_V1_E79_POS_BAG_OFFSET_GAMMA_INTEGRATION_R1.json"
)
OFFICIAL_TRAIN = MAIN_ROOT / "data/raw/classification/Train.csv"
OFFICIAL_TRAIN_SHA256 = "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c"
OFFICIAL_TEST_SHA256 = "4e6c381d582ed24f92d17f620f4f0ad25e2e0def29eb57a91a5482867f6afe3c"
SAMPLE_SHA256 = "889f97d69659a64945421d841dcf96e5813020606f29b2992d4b9560cecdcba7"
CHAMPION_SHA256 = "189a779b6100333a2fecfe443689fe7d340a331e828167221c81281a7ece4a4a"
UNIFORM_GAMMA_SHA256 = "d741d016f990d8742208edc919f5c58305b7579980d059b202609df4e5b3e650"
UNIFORM_GAMMA_RECEIPT_SHA256 = "f08f1bd3733a20fb8745d40c82190543a48af6bade1df5684ccfef3198c34580"
UNIFORM_GAMMA_VECTOR_SHA256 = "2171d72a0168f0ce43980a9c416fd64d673e1a7d9e9a213b226f843b55bf1bb1"
UNIFORM_GAMMA_FIT_ID = "A02_CGEOM_BENCHMARK_V1_E79_POS_BAG_OFFSET_GAMMA_INTEGRATION_R1"
ROOT_DECISION_COMMIT = "7a8ad5b6437db1c3bb8d716fb879252935615e34"
FIT_QA_PATH = REPORT_ROOT / "FIT_QA.json"
APPLICATION_ATTEMPT_PATH = EXPERIMENT_ROOT / "APPLICATION_ATTEMPT.started.json"
APPLICATION_OUTPUT_PATH = EXPERIMENT_ROOT / "APPLICATION_OUTPUTS.npz"
APPLICATION_RECEIPT_PATH = EXPERIMENT_ROOT / "APPLICATION_RECEIPT.json"
APPLICATION_FAILURE_PATH = EXPERIMENT_ROOT / "APPLICATION_FAILURE.json"
CSV_PATH = OUTPUT_ROOT / "classification_cls_pos_recency_e79_v1.csv"
MANIFEST_PATH = REPORT_ROOT / "APPLICATION_MANIFEST.json"
APPLICATION_QA_PATH = REPORT_ROOT / "APPLICATION_QA.json"
TARGET = "artifact_signal"
EXPECTED_TRAIN_ROWS = 50_608
EXPECTED_TEST_ROWS = 29_421
FIT_ID = "CLS_POS_RECENCY_E79_V1_E79_RECENCY"
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_text_sha256(path: Path) -> str:
    payload = Path(path).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def read_signed(path: Path, signature_key: str) -> dict[str, Any]:
    value = read_json(path)
    body = dict(value)
    signature = body.pop(signature_key, None)
    if signature != canonical_sha256(body):
        raise RuntimeError(f"Signature mismatch: {path}")
    return value


def git(*args: str, root: Path = AUDIT_ROOT, check: bool = True) -> str:
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    if check and result.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def main_array_hash(value: Any) -> str:
    array = np.asarray(value)
    header = canonical_sha256({"shape": array.shape, "dtype": str(array.dtype)})
    digest = hashlib.sha256(header.encode("ascii"))
    if array.dtype.kind in "OUS":
        digest.update(json.dumps(array.astype(str).tolist(), ensure_ascii=False,
                                 separators=(",", ":")).encode("utf-8"))
    else:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def main_csr_hash(value: sparse.spmatrix) -> str:
    matrix = sparse.csr_matrix(value, dtype=np.float64, copy=True)
    matrix.sum_duplicates()
    matrix.sort_indices()
    header = canonical_sha256({"shape": matrix.shape, "dtype": str(matrix.dtype)})
    digest = hashlib.sha256(header.encode("ascii"))
    digest.update(np.asarray(matrix.indptr, dtype="<i8").tobytes())
    digest.update(np.asarray(matrix.indices, dtype="<i8").tobytes())
    digest.update(np.asarray(matrix.data, dtype="<f8").tobytes())
    return digest.hexdigest()


def verify_authority_and_cache(mode: str = "train") -> tuple[dict[str, Any], dict[str, Any], sparse.csr_matrix, dict[str, np.ndarray], dict[str, Any]]:
    inventory = read_signed(INVENTORY_PATH, "inventory_sha256")
    freeze = read_signed(FREEZE_PATH, "freeze_sha256")
    prefit = read_signed(PREFIT_PATH, "prefit_qa_sha256")
    reservation = read_signed(RESERVATION_PATH, "reservation_sha256")
    if inventory.get("status") != "PASS_STATIC_AND_FULLTRAIN_FEATURE_SOURCE_INVENTORY":
        raise RuntimeError("Full-Train source inventory did not pass")
    if not all(inventory.get("checks", {}).values()):
        raise RuntimeError("At least one frozen inventory check is false")
    if freeze.get("fit_cap") != 1 or freeze.get("candidate_id") != "CLS_POS_RECENCY_E79_V1":
        raise RuntimeError("One-fit E79 freeze identity/cap mismatch")
    if freeze.get("root_decision_commit") not in (None, ROOT_DECISION_COMMIT):
        raise RuntimeError("E79 freeze cites an unexpected root decision commit")
    if freeze.get("input_lineage", {}).get("root_decision_commit") != ROOT_DECISION_COMMIT:
        raise RuntimeError("E79 freeze does not bind the root authorization commit")
    if reservation.get("fit_ids") != [FIT_ID] or reservation.get("predictor_fit_cap") != 1:
        raise RuntimeError("One official E79 optimizer slot is not correctly reserved")
    if reservation.get("freeze_sha256") != freeze.get("freeze_sha256"):
        raise RuntimeError("Reservation is not bound to the immutable freeze")
    decision_path = AUDIT_ROOT / "reports/simplification_v1/CLS_POS_RECENCY_V1/ROOT_RESULTS_DECISION.md"
    if not decision_path.is_file() or "APPROVE_EXPLORATORY_E79_BENCHMARK" not in decision_path.read_text(encoding="utf-8"):
        raise RuntimeError("Committed root authorization is absent")
    if subprocess.run(["git", "-C", str(AUDIT_ROOT), "merge-base", "--is-ancestor",
                       ROOT_DECISION_COMMIT, git("rev-parse", "HEAD")], capture_output=True).returncode:
        raise RuntimeError("Root authorization commit is not an ancestor of this verifier checkout")
    if git("rev-parse", "HEAD", root=MAIN_ROOT) != inventory.get("main_head"):
        raise RuntimeError("Read-only MAIN source HEAD changed after source inventory")
    if Path(inventory.get("main_root", "")).resolve() != MAIN_ROOT.resolve():
        raise RuntimeError("MAIN path differs from the source inventory")
    if sha256_file(OFFICIAL_TRAIN) != OFFICIAL_TRAIN_SHA256:
        raise RuntimeError("Official Train hash differs from the source freeze")
    for pin_id, record in inventory["pinned_file_hashes"].items():
        if not pin_id.startswith(("input:", "source:")):
            raise RuntimeError(f"Malformed frozen source/input identity: {pin_id}")
        source_path = MAIN_ROOT / pin_id.split(":", 1)[1]
        if (record.get("match") is not True or record.get("expected") != record.get("actual")
                or sha256_file(source_path) != record.get("expected")):
            raise RuntimeError(f"Pinned MAIN source/input changed: {source_path}")
    source_commit = freeze.get("prepared_from_head")
    if subprocess.run(["git", "-C", str(AUDIT_ROOT), "merge-base", "--is-ancestor",
                       source_commit, git("rev-parse", "HEAD")], capture_output=True).returncode:
        raise RuntimeError("Frozen E79 source commit is not an ancestor of current checkout")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Tracked source changed after freeze")
    if set(freeze.get("source_file_sha256", {})) != set(SOURCE_FILES):
        raise RuntimeError("Frozen E79 source file inventory is incomplete")
    for relative, expected in freeze["source_file_sha256"].items():
        path = AUDIT_ROOT / relative
        committed = subprocess.run(["git", "-C", str(AUDIT_ROOT), "show", f"{source_commit}:{relative}"],
                                   capture_output=True)
        committed_text = committed.stdout.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        if (not path.is_file() or source_text_sha256(path) != expected or committed.returncode
                or hashlib.sha256(committed_text).hexdigest() != expected):
            raise RuntimeError(f"Frozen E79 source code changed: {relative}")
    lineage = freeze.get("input_lineage", {})
    if (lineage.get("inventory_sha256") != inventory.get("inventory_sha256")
            or lineage.get("inventory_file_sha256") != source_text_sha256(INVENTORY_PATH)
            or lineage.get("root_decision_file_sha256") != source_text_sha256(decision_path)
            or lineage.get("official_train_sha256") != OFFICIAL_TRAIN_SHA256
            or lineage.get("test_or_sample_access") is not False
            or lineage.get("target_column_read") is not False):
        raise RuntimeError("Frozen Train-only source boundary does not match inventory")
    stem = "TRAIN" if mode == "train" else "TEST"
    base = TRAIN_FEATURE_DIR if mode == "train" else TEST_FEATURE_DIR
    matrix_path = base / f"{stem}_X_POSITION.npz"
    arrays_path = base / f"{stem}_FEATURE_ARRAYS.npz"
    receipt_path = base / f"{stem}_TRANSFORM_RECEIPT.json"
    if not all(path.is_file() for path in (matrix_path, arrays_path, receipt_path)):
        raise RuntimeError(f"{mode} feature-only cache is incomplete")
    receipt = read_signed(receipt_path, "receipt_sha256")
    if receipt.get("mode") != mode or receipt.get("status") != "PASS_FEATURE_ONLY_TRANSFORM":
        raise RuntimeError(f"{mode} feature-only transform identity/status mismatch")
    matrix = sparse.load_npz(matrix_path).tocsr().astype(np.float64)
    if (sha256_file(matrix_path) != receipt["position_matrix"]["cache_file_sha256"]
            or sha256_file(arrays_path) != receipt["arrays_cache_file_sha256"]):
        raise RuntimeError(f"{mode} feature matrix/array file hash mismatch")
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]).copy() for key in archive.files}
    if set(arrays) != {"record_ids", "cycles", "present", "z_bag"}:
        raise RuntimeError(f"{mode} feature cache fields differ from frozen schema")
    n = len(arrays["record_ids"])
    if (matrix.shape[0] != n or any(len(arrays[k]) != n for k in ("cycles", "present", "z_bag"))
            or not np.isfinite(matrix.data).all() or not np.isfinite(arrays["z_bag"]).all()):
        raise RuntimeError(f"{mode} feature matrix and arrays are not aligned/finite")
    if mode == "train" and main_csr_hash(matrix) != inventory["r3_position_matrix"]["logical_sha256"]:
        raise RuntimeError("Full-Train positional matrix logical hash differs from inventory")
    if mode == "test" and receipt.get("position_matrix", {}).get("logical_sha256") not in (None, main_csr_hash(matrix)):
        raise RuntimeError("Test positional matrix logical hash differs from its transform receipt")
    for field, key in (("ordered_ids_sha256", "record_ids"), ("cycles_sha256", "cycles"),
                       ("present_mask_sha256", "present"), ("z_bag_sha256", "z_bag")):
        if main_array_hash(arrays[key]) != receipt.get(field):
            raise RuntimeError(f"{mode} cached {key} differs from transform receipt")
    if mode == "train":
        if (n != EXPECTED_TRAIN_ROWS or receipt.get("target_column_requested_or_loaded") is not False
                or receipt.get("test_or_sample_used_for_fit") is not False
                or main_array_hash(arrays["record_ids"].astype(str)) != inventory["ordered_train_ids_sha256"]
                or not np.all((arrays["cycles"] >= 47) & (arrays["cycles"] <= 79))):
            raise RuntimeError("Full-Train transform rows, cycles, IDs, or target boundary mismatch")
    else:
        if (n != EXPECTED_TEST_ROWS or receipt.get("target_column_requested_or_loaded") is not False
                or receipt.get("test_or_sample_used_for_fit") is not False
                or not np.all((arrays["cycles"] >= 80) & (arrays["cycles"] <= 91))):
            raise RuntimeError("Test feature transform violates its feature-only boundary")
    return inventory, freeze, matrix, arrays, receipt


def recency_weights(cycles: np.ndarray, target: np.ndarray) -> np.ndarray:
    base = np.maximum(0.25, np.power(2.0, -(79 - cycles.astype(np.float64)) / 8.0))
    weights = np.empty(base.shape, dtype=np.float64)
    for label in (0, 1):
        mask = target == label
        if not np.any(mask):
            raise RuntimeError(f"Present fit population lacks target class {label}")
        weights[mask] = base[mask] / base[mask].mean()
    return weights


def objective_gradient(x: sparse.csr_matrix, z: np.ndarray, y: np.ndarray,
                       weights: np.ndarray, gamma: np.ndarray) -> tuple[float, np.ndarray]:
    margin = (2.0 * y.astype(np.float64) - 1.0) * (z + np.asarray(x @ gamma).reshape(-1))
    loss = float(np.dot(weights, np.logaddexp(0.0, -margin)) + 0.5 * np.dot(gamma, gamma))
    residual = weights * (expit(z + np.asarray(x @ gamma).reshape(-1)) - y)
    gradient = np.asarray(x.T @ residual, dtype=np.float64).reshape(-1) + gamma
    return loss, gradient


def verify_fit(write: bool = True) -> dict[str, Any]:
    inventory, freeze, matrix, arrays, transform_receipt = verify_authority_and_cache("train")
    attempt = read_json(ATTEMPT_PATH)
    receipt = read_signed(FIT_RECEIPT_PATH, "receipt_sha256")
    if not GAMMA_PATH.is_file() or sha256_file(GAMMA_PATH) != receipt.get("gamma_artifact_file_sha256"):
        raise RuntimeError("Official E79 gamma artifact hash does not match fit receipt")
    with np.load(GAMMA_PATH, allow_pickle=False) as archive:
        if set(archive.files) != {"gamma_float64"}:
            raise RuntimeError("Persisted E79 gamma artifact schema mismatch")
        gamma = np.asarray(archive["gamma_float64"]).copy()
    if (gamma.dtype != np.dtype(np.float64) or gamma.shape != (matrix.shape[1],)
            or not np.isfinite(gamma).all() or array_sha256(gamma) != receipt.get("gamma_sha256")):
        raise RuntimeError("Persisted E79 gamma vector type, shape, or digest mismatch")
    if (attempt.get("fit_id") != FIT_ID or attempt.get("status") != "ONE_E79_RECENCY_OPTIMIZER_STARTED"
            or attempt.get("official_optimizer_starts") != 1 or attempt.get("test_or_sample_access") is not False
            or attempt.get("kaggle_contact") is not False):
        raise RuntimeError("E79 fit attempt ledger does not show exactly one authorized start")
    if (receipt.get("fit_id") != FIT_ID or receipt.get("candidate_id") != freeze.get("candidate_id")
            or receipt.get("source_commit") != attempt.get("source_commit")
            or receipt.get("source_freeze_sha256") != freeze.get("freeze_sha256")
            or receipt.get("status") != "PASS_E79_RECENCY_GAMMA_FIT_PERSISTED"
            or receipt.get("mode") != "official"):
        raise RuntimeError("Official gamma fit receipt does not report the frozen successful fit")
    present = arrays["present"].astype(bool, copy=False)
    cycles_all = arrays["cycles"].astype(np.int64, copy=False)
    ids = arrays["record_ids"].astype(str, copy=False)
    x = matrix[present].tocsr()
    z = arrays["z_bag"][present].astype(np.float64, copy=False)
    target_frame = pd.read_csv(OFFICIAL_TRAIN, usecols=[TARGET], low_memory=False)
    y_all = pd.to_numeric(target_frame[TARGET], errors="raise").to_numpy(dtype=np.int8)
    if y_all.shape != (EXPECTED_TRAIN_ROWS,) or not np.isin(y_all, [0, 1]).all():
        raise RuntimeError("Official Train target-only vector schema or labels are invalid")
    y = y_all[present]
    cycles = cycles_all[present]
    weights = recency_weights(cycles, y)
    loss, gradient = objective_gradient(x, z, y, weights, gamma)
    metadata = receipt.get("optimizer", {})
    options = {"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-6, "maxls": 50}
    if metadata.get("optimizer") != "scipy.optimize.minimize:L-BFGS-B" or metadata.get("options") != options:
        raise RuntimeError("E79 optimizer name/options differ from the frozen pilot recipe")
    if metadata.get("success") is not True or not np.isfinite(loss) or not np.isfinite(gradient).all():
        raise RuntimeError("E79 optimizer did not persist a finite successful solution")
    if not np.isclose(loss, metadata.get("loss_final", float("nan")), rtol=2e-11, atol=2e-7):
        raise RuntimeError("Independent E79 objective differs from fit metadata")
    if not np.isclose(float(np.max(np.abs(gradient))), metadata.get("gradient_inf_norm", float("nan")),
                      rtol=2e-6, atol=2e-8):
        raise RuntimeError("Independent E79 gradient differs from optimizer metadata")
    if loss > metadata.get("loss_gamma0", float("inf")) + 1e-8:
        raise RuntimeError("E79 fit objective did not improve over zero initialization")
    class_diagnostics: dict[str, Any] = {}
    for label in (0, 1):
        mask = y == label
        values = weights[mask]
        expected_rows = int(mask.sum())
        if not np.isclose(values.sum(), expected_rows, rtol=1e-12, atol=1e-9):
            raise RuntimeError(f"E79 class {label} normalized weight mass is invalid")
        class_diagnostics[str(label)] = {
            "rows": expected_rows,
            "weight_sum": float(values.sum()),
            "ess": float(values.sum() ** 2 / np.dot(values, values)),
            "ess_ratio": float(values.sum() ** 2 / np.dot(values, values)) / expected_rows,
            "weight_min": float(values.min()),
            "weight_max": float(values.max()),
        }
    input_checks = {
        "record_ids_sha256": array_sha256(ids) == receipt["inputs"]["record_ids_sha256"],
        "cycles_sha256": array_sha256(cycles_all) == receipt["inputs"]["cycles_sha256"],
        "present_mask_sha256": array_sha256(present) == receipt["inputs"]["present_mask_sha256"],
        "target_sha256": array_sha256(y) == receipt["inputs"]["fit_target_sha256"],
        "fit_weights_sha256": array_sha256(weights) == receipt["inputs"]["normalized_weights_sha256"],
        "position_values_sha256": array_sha256(x.data) == receipt["inputs"]["position_fit_values_sha256"],
        "position_indices_sha256": array_sha256(x.indices) == receipt["inputs"]["position_fit_indices_sha256"],
        "position_indptr_sha256": array_sha256(x.indptr) == receipt["inputs"]["position_fit_indptr_sha256"],
        "z_bag_sha256": array_sha256(z) == receipt["inputs"]["z_bag_fit_sha256"],
        "transform_receipt_matches_attempt": attempt.get("freeze_sha256") == freeze.get("freeze_sha256"),
        "all_test_sample_and_kaggle_access_absent": receipt.get("test_or_sample_access") is False
        and receipt.get("kaggle_contact") is False,
    }
    if not all(input_checks.values()):
        raise RuntimeError(f"Independent input readback checks failed: {input_checks}")
    if receipt.get("read_accounting", {}).get("test_target_reads") != 0:
        raise RuntimeError("Fit receipt reports a Test target read")
    body: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS_INDEPENDENT_E79_FIT_QA",
        "candidate_id": "CLS_POS_RECENCY_E79_V1",
        "fit_id": FIT_ID,
        "source_commit": receipt["source_commit"],
        "source_freeze_sha256": freeze["freeze_sha256"],
        "fit_attempt_file_sha256": sha256_file(ATTEMPT_PATH),
        "fit_receipt_file_sha256": sha256_file(FIT_RECEIPT_PATH),
        "gamma_file_sha256": sha256_file(GAMMA_PATH),
        "gamma_sha256": array_sha256(gamma),
        "gamma_shape": list(gamma.shape),
        "official_train_sha256": OFFICIAL_TRAIN_SHA256,
        "fit_population_rows": int(len(y)),
        "train_rows": EXPECTED_TRAIN_ROWS,
        "stream_present_rows": int(present.sum()),
        "stream_missing_excluded_rows": int((~present).sum()),
        "class_weight_diagnostics": class_diagnostics,
        "independent_objective": loss,
        "independent_gradient_inf_norm": float(np.max(np.abs(gradient))),
        "optimizer_success": metadata["success"],
        "optimizer_options": options,
        "verified_inputs": input_checks,
        "transform_receipt_sha256": transform_receipt["receipt_sha256"],
        "test_or_sample_access": False,
        "kaggle_contact": False,
        "refit_or_optimizer_called_by_verifier": False,
    }
    body["fit_qa_sha256"] = canonical_sha256(body)
    if write:
        if FIT_QA_PATH.exists():
            prior = read_signed(FIT_QA_PATH, "fit_qa_sha256")
            if prior != body:
                raise RuntimeError("Existing FIT_QA.json differs; refusing overwrite")
            return prior
        FIT_QA_PATH.write_text(json.dumps(body, sort_keys=True, ensure_ascii=False,
                                         allow_nan=False, indent=2) + "\n", encoding="utf-8")
    return body


def read_csv_predictions(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["record_id", TARGET]:
            raise RuntimeError(f"Classification CSV schema differs: {path}")
        rows = list(reader)
    ids = [row["record_id"] for row in rows]
    try:
        labels = np.asarray([int(row[TARGET]) for row in rows], dtype=np.int8)
    except (ValueError, TypeError) as exc:
        raise RuntimeError(f"Classification labels are not integers: {path}") from exc
    if len(ids) != len(set(ids)) or not np.isin(labels, [0, 1]).all():
        raise RuntimeError(f"Classification IDs or labels are invalid: {path}")
    return ids, labels


def duplicate_vector_scan(expected_ids: list[str], expected_labels: np.ndarray) -> dict[str, Any]:
    scanned = 0
    malformed = 0
    matches: list[str] = []
    for path in sorted((MAIN_ROOT / "submissions").rglob("*.csv")):
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            try:
                header = next(reader)
            except StopIteration:
                malformed += 1
                continue
            if header != ["record_id", TARGET]:
                continue
            rows = list(reader)
        if len(rows) != len(expected_ids):
            continue
        try:
            ids = [row[0] for row in rows]
            labels = np.asarray([int(row[1]) for row in rows], dtype=np.int8)
        except (IndexError, ValueError):
            malformed += 1
            continue
        if ids != expected_ids or len(ids) != len(set(ids)) or not np.isin(labels, [0, 1]).all():
            continue
        scanned += 1
        if np.array_equal(labels, expected_labels):
            matches.append(path.relative_to(MAIN_ROOT).as_posix())
    return {
        "valid_full_test_classification_vectors_scanned": scanned,
        "malformed_matching_schema_files": malformed,
        "exact_label_vector_matches": matches,
        "duplicate_found": bool(matches),
    }


def verify_application(write: bool = True) -> dict[str, Any]:
    fit_qa = read_signed(FIT_QA_PATH, "fit_qa_sha256")
    if fit_qa.get("status") != "PASS_INDEPENDENT_E79_FIT_QA":
        raise RuntimeError("Feature-only Test transform requires independent E79 fit QA PASS")
    inventory, freeze, x, arrays, test_transform = verify_authority_and_cache("test")
    if sha256_file(TEST_CSV) != OFFICIAL_TEST_SHA256:
        raise RuntimeError("Official Test hash differs from the allowed feature-only input")
    if sha256_file(SAMPLE_CSV) != SAMPLE_SHA256:
        raise RuntimeError("SampleSubmission schema/ID source differs from frozen input hash")
    if sha256_file(CHAMPION_CSV) != CHAMPION_SHA256:
        raise RuntimeError("Frozen full champion CSV hash changed")
    if sha256_file(UNIFORM_GAMMA_PATH) != UNIFORM_GAMMA_SHA256 or sha256_file(UNIFORM_GAMMA_RECEIPT) != UNIFORM_GAMMA_RECEIPT_SHA256:
        raise RuntimeError("Original uniform POS_BAG_OFFSET gamma or its receipt changed")
    if not APPLICATION_ATTEMPT_PATH.is_file() or not APPLICATION_OUTPUT_PATH.is_file() or not CSV_PATH.is_file():
        raise RuntimeError("One-shot Test application outputs are incomplete")
    attempt = read_json(APPLICATION_ATTEMPT_PATH)
    app_receipt = read_signed(APPLICATION_RECEIPT_PATH, "receipt_sha256")
    manifest = read_signed(MANIFEST_PATH, "manifest_sha256")
    if (attempt.get("status") != "ONE_E79_APPLICATION_STARTED"
            or attempt.get("fit_qa_sha256") != fit_qa.get("fit_qa_sha256")
            or app_receipt.get("fit_qa_sha256") != fit_qa.get("fit_qa_sha256")
            or app_receipt.get("target_column_read") is not False
            or app_receipt.get("test_target_read") is not False
            or app_receipt.get("sample_target_read") is not False
            or app_receipt.get("kaggle_contact") is not False):
        raise RuntimeError("Test application attempt or receipt crossed its frozen boundary")
    if (manifest.get("status") != "LOCAL_CSV_PREPARED_NOT_UPLOADED"
            or manifest.get("submission_uploaded") is not False
            or manifest.get("kaggle_contacted") is not False
            or manifest.get("csv_sha256") != sha256_file(CSV_PATH)
            or app_receipt.get("csv_sha256") != sha256_file(CSV_PATH)
            or app_receipt.get("application_arrays_sha256") != sha256_file(APPLICATION_OUTPUT_PATH)
            or manifest.get("manifest_file_bindings", {}).get("application_receipt_sha256")
            != sha256_file(APPLICATION_RECEIPT_PATH)):
        raise RuntimeError("Application manifest hash/status/file bindings are invalid")
    if (app_receipt.get("test_transform_receipt_sha256") != test_transform.get("receipt_sha256")
            or app_receipt.get("test_transform_matrix_logical_sha256") != main_csr_hash(x)
            or app_receipt.get("test_transform_arrays_file_sha256")
            != test_transform.get("arrays_cache_file_sha256")):
        raise RuntimeError("Application receipt is not bound to the verified Test feature transform")
    gamma_receipt = read_signed(FIT_RECEIPT_PATH, "receipt_sha256")
    if (fit_qa.get("fit_receipt_file_sha256") != sha256_file(FIT_RECEIPT_PATH)
            or app_receipt.get("fit_receipt_sha256") != sha256_file(FIT_RECEIPT_PATH)):
        raise RuntimeError("Fit QA/application receipt binding to saved E79 fit is invalid")
    with np.load(GAMMA_PATH, allow_pickle=False) as archive:
        gamma = np.asarray(archive["gamma_float64"], dtype=np.float64).copy()
    with np.load(APPLICATION_OUTPUT_PATH, allow_pickle=False) as archive:
        outputs = {key: np.asarray(archive[key]).copy() for key in archive.files}
    required = {"record_ids", "present", "cycles", "p_a02", "p_e79", "p_uniform", "p_output", "labels", "champion_labels"}
    if set(outputs) != required:
        raise RuntimeError("Persisted E79 application array schema mismatch")
    ids = arrays["record_ids"].astype(str).tolist()
    p_a02 = outputs["p_a02"].astype(np.float64, copy=False)
    # Official Test target is excluded explicitly; only IDs and approved feature keys are read.
    test_features = pd.read_csv(TEST_CSV, usecols=["record_id", "cycle_code", "signal_stream"],
                                dtype={"record_id": str}, low_memory=False)
    test_ids = test_features["record_id"].astype(str).tolist()
    test_cycles = pd.to_numeric(test_features["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    test_stream = test_features["signal_stream"]
    test_present = (~test_stream.isna()
                    & test_stream.astype("string").str.strip().ne("")).to_numpy(dtype=bool)
    if (test_ids != ids or len(set(test_ids)) != EXPECTED_TEST_ROWS
            or not np.array_equal(test_cycles, arrays["cycles"].astype(np.int64))
            or not np.array_equal(test_present, arrays["present"].astype(bool))):
        raise RuntimeError("Feature-only Test IDs/cycles/presence differ from official Test fields")
    a02_fit_receipt = read_json(A02_FIT_RECEIPT)
    a02_test_receipt = read_json(A02_TEST_RECEIPT)
    if (a02_test_receipt.get("official_test_targets_loaded") is not False
            or a02_test_receipt.get("npz_sha256") != sha256_file(A02_TEST_NPZ)
            or a02_test_receipt.get("fit_receipt_sha256") != a02_fit_receipt.get("receipt_sha256")
            or a02_test_receipt.get("freeze_signature") != a02_fit_receipt.get("freeze_signature")
            or a02_fit_receipt.get("model_sha256") != sha256_file(A02_ROOT / "model.joblib")
            or a02_fit_receipt.get("test_data_used") is not False
            or a02_fit_receipt.get("query_labels_passed_to_fit") is not False):
        raise RuntimeError("A02 Test cache lineage or target boundary is invalid")
    with np.load(A02_TEST_NPZ, allow_pickle=False) as archive:
        if not {"record_ids", "missing_mask", "candidate_probability"}.issubset(archive.files):
            raise RuntimeError("A02 Test probability cache is missing alignment fields")
        a02_ids = np.asarray(archive["record_ids"]).astype(str).tolist()
        a02_missing = np.asarray(archive["missing_mask"], dtype=bool)
        a02_probability = np.asarray(archive["candidate_probability"], dtype=np.float64).reshape(-1)
    if (a02_ids != ids or not np.array_equal(a02_missing, ~test_present)
            or not np.array_equal(a02_probability, p_a02)):
        raise RuntimeError("Cached A02 probabilities/IDs/presence do not match application arrays")
    p_e79 = expit(arrays["z_bag"] + np.asarray(x @ gamma).reshape(-1))
    uniform_receipt = read_json(UNIFORM_GAMMA_RECEIPT)
    if (uniform_receipt.get("fit_id") != UNIFORM_GAMMA_FIT_ID
            or uniform_receipt.get("artifact_sha256") != UNIFORM_GAMMA_SHA256
            or uniform_receipt.get("gamma_sha256") != UNIFORM_GAMMA_VECTOR_SHA256
            or uniform_receipt.get("fit_rows") != EXPECTED_TRAIN_ROWS
            or uniform_receipt.get("matrix_sha256") not in (None, inventory["r3_position_matrix"]["logical_sha256"])):
        # The source inventory is itself pinned and also binds this exact artifact.
        expected_uniform = inventory.get("original_uniform_gamma", {})
        if (uniform_receipt.get("fit_id") != UNIFORM_GAMMA_FIT_ID
                or expected_uniform.get("artifact_sha256") != UNIFORM_GAMMA_SHA256
                or expected_uniform.get("gamma_vector_sha256") != UNIFORM_GAMMA_VECTOR_SHA256):
            raise RuntimeError("Original full-Train uniform gamma receipt lineage mismatch")
    import joblib
    uniform_state = joblib.load(UNIFORM_GAMMA_PATH)
    if not isinstance(uniform_state, dict) or "gamma" not in uniform_state:
        raise RuntimeError("Original uniform gamma artifact schema differs from the verified recipe")
    gamma_uniform = np.asarray(uniform_state["gamma"], dtype=np.float64).reshape(-1)
    if gamma_uniform.shape != gamma.shape or not np.isfinite(gamma_uniform).all():
        raise RuntimeError("Original uniform gamma vector shape is invalid")
    if main_array_hash(gamma_uniform) != UNIFORM_GAMMA_VECTOR_SHA256:
        raise RuntimeError("Loaded original uniform gamma vector digest differs from its frozen receipt")
    p_uniform = expit(arrays["z_bag"] + np.asarray(x @ gamma_uniform).reshape(-1))
    present = arrays["present"].astype(bool, copy=False)
    expected_e79 = p_a02.copy()
    expected_e79[present] = 0.5 * p_e79[present] + 0.5 * p_a02[present]
    expected_champion = p_a02.copy()
    expected_champion[present] = 0.5 * p_uniform[present] + 0.5 * p_a02[present]
    expected_labels = np.where(present, expected_e79 >= 0.50,
                               expected_e79 >= 0.53230976129477).astype(np.int8)
    champion_labels = np.where(present, expected_champion >= 0.50,
                               expected_champion >= 0.53230976129477).astype(np.int8)
    if (outputs["record_ids"].astype(str).tolist() != ids
            or not np.array_equal(outputs["present"].astype(bool), present)
            or not np.array_equal(outputs["cycles"].astype(np.int64), arrays["cycles"].astype(np.int64))
            or not np.array_equal(outputs["p_e79"], p_e79)
            or not np.array_equal(outputs["p_uniform"], p_uniform)
            or not np.array_equal(outputs["p_output"], expected_e79)
            or not np.array_equal(outputs["labels"], expected_labels)
            or not np.array_equal(outputs["champion_labels"], champion_labels)):
        raise RuntimeError("Candidate probability or champion reconstruction differs from independent replay")
    if not np.array_equal(expected_e79[~present].view(np.uint64), p_a02[~present].view(np.uint64)):
        raise RuntimeError("Missing-stream A02 probability fallback is not bit-exact")
    sample_ids: list[str] = []
    with SAMPLE_CSV.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        if header != ["record_id", TARGET]:
            raise RuntimeError("SampleSubmission header differs from the frozen schema")
        for row in reader:
            if len(row) != 2:
                raise RuntimeError("SampleSubmission row schema is malformed")
            sample_ids.append(row[0])
    csv_ids, csv_labels = read_csv_predictions(CSV_PATH)
    if len(ids) != EXPECTED_TEST_ROWS or csv_ids != ids or sample_ids != ids or not np.array_equal(csv_labels, expected_labels):
        raise RuntimeError("Local candidate CSV does not match official Test/Sample IDs and frozen policy")
    champion_ids, champion_csv_labels = read_csv_predictions(CHAMPION_CSV)
    if champion_ids != ids or not np.array_equal(champion_csv_labels, champion_labels):
        raise RuntimeError("Original frozen champion cannot be independently reconstructed")
    if np.any(~np.isfinite(p_a02)) or np.any((p_a02 < 0) | (p_a02 > 1)):
        raise RuntimeError("Frozen A02 probability values are not finite [0,1]")
    if any(not np.isfinite(arrays[key]).all() for key in ("z_bag",)):
        raise RuntimeError("Feature-only Test offset contains non-finite values")
    body: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS_INDEPENDENT_E79_APPLICATION_QA",
        "candidate_id": "CLS_POS_RECENCY_E79_V1",
        "source_commit": app_receipt["source_commit"],
        "fit_qa_sha256": fit_qa["fit_qa_sha256"],
        "fit_receipt_sha256": gamma_receipt["receipt_sha256"],
        "official_test_sha256": OFFICIAL_TEST_SHA256,
        "sample_submission_sha256": SAMPLE_SHA256,
        "a02_test_npz_sha256": app_receipt["a02_test_npz_sha256"],
        "a02_test_receipt_sha256": app_receipt["a02_test_receipt_sha256"],
        "uniform_gamma_artifact_sha256": UNIFORM_GAMMA_SHA256,
        "uniform_gamma_receipt_sha256": UNIFORM_GAMMA_RECEIPT_SHA256,
        "champion_csv_sha256": CHAMPION_SHA256,
        "candidate_csv_path": str(CSV_PATH.relative_to(AUDIT_ROOT)).replace("\\", "/"),
        "candidate_csv_sha256": sha256_file(CSV_PATH),
        "rows": len(csv_ids),
        "unique_ids": len(set(csv_ids)),
        "columns": ["record_id", TARGET],
        "present_rows": int(present.sum()),
        "missing_rows": int((~present).sum()),
        "candidate_positive_count": int(csv_labels.sum()),
        "champion_positive_count": int(champion_csv_labels.sum()),
        "label_flips_vs_champion": int(np.count_nonzero(csv_labels != champion_csv_labels)),
        "missing_probability_bit_exact_to_a02": True,
        "missing_labels_bit_exact_to_champion": bool(np.array_equal(csv_labels[~present], champion_csv_labels[~present])),
        "test_targets_read": False,
        "sample_target_values_read": False,
        "optimizer_or_model_fit_by_verifier": False,
        "kaggle_contact_or_upload": False,
        "source_features_recomputed_and_probabilities_replayed": True,
        "duplicate_vector_scan": duplicate_vector_scan(ids, csv_labels),
    }
    body["application_qa_sha256"] = canonical_sha256(body)
    if write:
        if APPLICATION_QA_PATH.exists():
            prior = read_signed(APPLICATION_QA_PATH, "application_qa_sha256")
            if prior != body:
                raise RuntimeError("Existing APPLICATION_QA.json differs; refusing overwrite")
            return prior
        APPLICATION_QA_PATH.write_text(json.dumps(body, sort_keys=True, ensure_ascii=False,
                                                  allow_nan=False, indent=2) + "\n", encoding="utf-8")
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fit", action="store_true")
    group.add_argument("--application", action="store_true")
    args = parser.parse_args()
    result = verify_fit() if args.fit else verify_application()
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("verified_inputs", "class_weight_diagnostics")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
