"""Independent verifier for the REG POS Poisson weighted replica pilot."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping
import zipfile

import joblib
import numpy as np
from scipy import sparse
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path(r"C:\Users\Sam\OneDrive\Documents\KASDAD\TP 2")
FRONTIER_ROOT = Path(r"C:\Users\Sam\OneDrive\Documents\KASDAD\tk2-regression-frontier")
RUNNER_PATH = FRONTIER_ROOT / "scripts" / "reg_position_direct_l1_v1.py"
POSITION_FREEZE = FRONTIER_ROOT / "experiments" / "runs" / "reg_position_direct_l1_v1" / "FREEZE_V4.json"
POSITION_RUN = POSITION_FREEZE.parent
HINGE_RUNNER = ROOT / "scripts" / "reg_monotone_hinge_p50_v1.py"
CARD_ID = "REG_POS_POISSON_BAG2_V1"
ORIGINS = (67, 75)
SEEDS = (17, 137)
FIT_IDS = tuple(f"{CARD_ID}_E{origin}_P{seed}" for origin in ORIGINS for seed in SEEDS)
REQUIRED_ROOT_TRACKED_PATHS = {
    "scripts/reg_pos_poisson_bag2_v1.py",
    "scripts/verify_reg_pos_poisson_bag2_v1.py",
    "tests/test_reg_pos_poisson_bag2_v1.py",
    "experiments/reg_pos_poisson_bag2_v1/FREEZE_R1.json",
    "experiments/reg_pos_poisson_bag2_v1/FREEZE_R1_SUPERSEDED_2c5742e5.json",
    "experiments/reg_pos_poisson_bag2_v1/FREEZE_R1_SUPERSEDED_INVALID_928ba2a8.json",
    "experiments/reg_pos_poisson_bag2_v1/FREEZE_R1_SUPERSEDED_1161187.json",
    "experiments/reg_pos_poisson_bag2_v1/POISSON_WEIGHTS_R1.npz",
    "reports/optimization_v3/reg_pos_poisson_bag2_v1/PREFIT_VERIFICATION_R1.json",
    "reports/optimization_v3/reg_pos_poisson_bag2_v1/PREFIT_VERIFICATION_R1_SUPERSEDED_INVALID_928ba2a8.json",
    "reports/optimization_v3/reg_pos_poisson_bag2_v1/PREFIT_VERIFICATION_R1_SUPERSEDED_1161187.json",
    "reports/optimization_v3/reg_pos_poisson_bag2_v1/SOURCE_SUPERSESSION_R2.json",
    "reports/optimization_v3/reg_pos_poisson_bag2_v1/SYNTHETIC_E2E.json",
}
EXPERIMENT = ROOT / "experiments" / "reg_pos_poisson_bag2_v1"
REPORT = ROOT / "reports" / "optimization_v3" / "reg_pos_poisson_bag2_v1"
FREEZE_PATH = EXPERIMENT / "FREEZE_R1.json"
SUPERSEDED_SOURCE_FREEZE_PATH = EXPERIMENT / "FREEZE_R1_SUPERSEDED_1161187.json"
SUPERSEDENCE_RECEIPT_PATH = REPORT / "SOURCE_SUPERSESSION_R2.json"
PREFIT_REPORT = REPORT / "PREFIT_VERIFICATION_R1.json"
SUPERSEDED_SOURCE_PREFIT_PATH = REPORT / "PREFIT_VERIFICATION_R1_SUPERSEDED_1161187.json"
WEIGHTS_PATH = EXPERIMENT / "POISSON_WEIGHTS_R1.npz"
OFFICIAL_RUN = ROOT / "experiments" / "runs" / "reg_pos_poisson_bag2_v1"
PREDICTION_FREEZE = OFFICIAL_RUN / "PREDICTION_FREEZE_R1.json"
PRELABEL_REPORT = REPORT / "PRELABEL_VERIFICATION_R1.json"
ROOT_SOURCE_PASS = EXPERIMENT / "ROOT_SOURCE_PASS_R1.json"
SHARED_HEAVY_LOCK_PATH = Path(tempfile.gettempdir()) / "kasdad_single_heavy_job.lock"
LOCAL_CORRECTION = ROOT / "experiments" / "reg_center_local_correction_v1"
LOCAL_PREDICTION_FREEZE = LOCAL_CORRECTION / "PREDICTION_FREEZE_R1.json"
LOCAL_ROOT_SOURCE_PASS = LOCAL_CORRECTION / "ROOT_SOURCE_PASS_R1.json"
HINGE_PARAMS_PATHS = {
    67: LOCAL_CORRECTION / "fits_R1" / "D1_LOCAL.json",
    75: LOCAL_CORRECTION / "fits_R1" / "D2_LOCAL.json",
}
LOCAL_PREDICTION_PATHS = {
    67: LOCAL_CORRECTION / "score_predictions_R1" / "HINGE_D1_PREDICTIONS.npz",
    75: LOCAL_CORRECTION / "score_predictions_R1" / "HINGE_D2_PREDICTIONS.npz",
}
SCORE_WINDOWS = {67: (72, 75), 75: (76, 79)}
SIMPLIFICATION_ROOT = Path(r"C:\Users\Sam\OneDrive\Documents\KASDAD\tk2-simplification-audit-v1")
SIMPLIFICATION_SCRIPT = SIMPLIFICATION_ROOT / "scripts" / "simplification_audit_v1.py"
SIMPLIFICATION_FREEZE = SIMPLIFICATION_ROOT / "reports" / "simplification_v1" / "INPUT_FREEZE.json"
THREADS = 2


class VerifyError(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8", errors="surrogatepass")


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bounded(root: Path, relative_path: str) -> Path:
    root = root.resolve()
    path = (root / relative_path).resolve()
    if root not in path.parents or not path.is_file():
        raise VerifyError(f"Required source path is missing or escapes repository: {relative_path}")
    return path


def _is_git_ancestor(root: Path, ancestor: str, descendant: str) -> bool:
    valid = lambda value: (isinstance(value, str) and len(value) in (40, 64)
                           and all(char in "0123456789abcdef" for char in value.lower()))
    if not valid(ancestor) or not valid(descendant):
        return False
    try:
        result = subprocess.run(["git", "-C", str(root), "merge-base", "--is-ancestor",
                                 ancestor, descendant], check=False, capture_output=True)
    except OSError as exc:
        raise VerifyError(f"Cannot verify Git ancestry in {root}") from exc
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    raise VerifyError(f"Git ancestry check failed in {root}: {result.stderr.decode(errors='replace')}")


def _git_blob_sha256(root: Path, revision: str, relative_path: str) -> str:
    try:
        blob = subprocess.run(["git", "-C", str(root), "show", f"{revision}:{relative_path}"],
                              check=True, capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise VerifyError(f"Reviewed commit does not contain required source: {relative_path}") from exc
    return hashlib.sha256(blob).hexdigest()


def _verify_approved_source_commit(root: Path, freeze_base_commit: str,
                                   approval: Mapping[str, Any]) -> tuple[str, str]:
    reviewed = approval.get("reviewed_source_commit", approval.get("local_commit"))
    if (approval.get("local_commit") not in (None, reviewed)
            or not _is_git_ancestor(root, freeze_base_commit, reviewed)):
        raise VerifyError("Root approval commit is not descended from the frozen source base")
    current_head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                  check=True, capture_output=True, text=True).stdout.strip()
    if not _is_git_ancestor(root, reviewed, current_head):
        raise VerifyError("Reviewed source commit is not an ancestor of current HEAD")
    tracked_hashes = approval.get("tracked_source_hashes", {})
    blob_hashes = approval.get("tracked_blob_hashes", {})
    if (not isinstance(tracked_hashes, dict) or set(tracked_hashes) != REQUIRED_ROOT_TRACKED_PATHS
            or not isinstance(blob_hashes, dict) or set(blob_hashes) != REQUIRED_ROOT_TRACKED_PATHS):
        raise VerifyError("Root approval must bind exact worktree and Git blob hashes for required pilot paths")
    for relative_path in sorted(REQUIRED_ROOT_TRACKED_PATHS):
        expected_worktree_sha = tracked_hashes[relative_path]
        expected_blob_sha = blob_hashes[relative_path]
        if any(not isinstance(value, str) or len(value) != 64
               or any(char not in "0123456789abcdef" for char in value.lower())
               for value in (expected_worktree_sha, expected_blob_sha)):
            raise VerifyError(f"Root approval hash is invalid: {relative_path}")
        path = _bounded(root, relative_path)
        if (_sha_file(path) != expected_worktree_sha
                or _git_blob_sha256(root, reviewed, relative_path) != expected_blob_sha
                or _git_blob_sha256(root, current_head, relative_path) != expected_blob_sha):
            raise VerifyError(f"Reviewed, current HEAD, and worktree source hashes differ: {relative_path}")
    return reviewed, current_head


class _SharedHeavyJobLock:
    """Independent verifier side of the shared nonblocking advisory byte lock."""
    def __init__(self, path: Path = SHARED_HEAVY_LOCK_PATH):
        self.path = Path(path)
        self.stream = None
        self.backend = None
        self.locked = False

    def __enter__(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.stream = self.path.open("a+b")
            if self.path.stat().st_size == 0:
                self.stream.write(b"\0")
                self.stream.flush()
                os.fsync(self.stream.fileno())
            self.stream.seek(0)
            if os.name == "nt":
                import msvcrt
                self.backend = msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                self.backend = fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.locked = True
            return self
        except OSError as exc:
            if self.stream is not None:
                self.stream.close()
                self.stream = None
            raise VerifyError(f"Shared heavy-job lock is held or unavailable: {self.path}") from exc

    def __exit__(self, exc_type, exc, traceback):
        if self.stream is None:
            return
        try:
            if self.locked:
                self.stream.seek(0)
                if os.name == "nt":
                    self.backend.locking(self.stream.fileno(), self.backend.LK_UNLCK, 1)
                else:
                    self.backend.flock(self.stream.fileno(), self.backend.LOCK_UN)
        finally:
            self.locked = False
            self.stream.close()
            self.stream = None


def _sha_array(value: Any, *, dtype: str | None = None) -> str:
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
    digest = hashlib.sha256(_canonical({"shape": matrix.shape, "dtype": str(matrix.dtype)}))
    digest.update(np.asarray(matrix.indptr, dtype="<i8").tobytes())
    digest.update(np.asarray(matrix.indices, dtype="<i8").tobytes())
    digest.update(np.asarray(matrix.data, dtype="<f8").tobytes())
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise VerifyError(f"Expected JSON object: {path}")
    return value


def _signed(path: Path, signature_key: str) -> dict[str, Any]:
    value = _read_json(path)
    body = {key: item for key, item in value.items() if key != signature_key}
    if value.get(signature_key) != _canonical_sha(body):
        raise VerifyError(f"Canonical signature mismatch: {path}")
    return value


def _explicit_npz(path: Path, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    with zipfile.ZipFile(path) as archive:
        members = {name[:-4] for name in archive.namelist() if name.endswith(".npy")}
    if (not set(keys).issubset(members)
            or any("target" in key.lower() and key not in {"synthetic_targets", "fit_targets"}
                   for key in keys)):
        raise VerifyError(f"Requested member missing or target member requested: {path}")
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key].copy() for key in keys}


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise VerifyError(f"Cannot import frozen module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _runner():
    sys.path.insert(0, str(FRONTIER_ROOT)) if str(FRONTIER_ROOT) not in sys.path else None
    return _module(RUNNER_PATH, "reg_position_direct_l1_v1_independent")


def _source_artifact_hashes(runner: Any) -> dict[str, str]:
    hashes: dict[str, str] = {
        "frontier_runner": _sha_file(RUNNER_PATH),
        "position_freeze_v4": _sha_file(POSITION_FREEZE),
        "hinge_runner": _sha_file(HINGE_RUNNER),
        "local_prediction_freeze": _sha_file(LOCAL_PREDICTION_FREEZE),
        "local_root_source_pass": _sha_file(LOCAL_ROOT_SOURCE_PASS),
        "simplification_audit_script": _sha_file(SIMPLIFICATION_SCRIPT),
        "simplification_input_freeze": _sha_file(SIMPLIFICATION_FREEZE),
    }
    for origin in ORIGINS:
        hashes[f"hinge_parameters_E{origin}"] = _sha_file(HINGE_PARAMS_PATHS[origin])
        hashes[f"local_comparator_E{origin}"] = _sha_file(LOCAL_PREDICTION_PATHS[origin])
        fit_id = f"{runner.CARD_ID}_E{origin}_REAL"
        artifact, receipt = runner._unit_paths(fit_id, runner.RUN_ROOT)
        pred, pred_receipt = runner._prediction_paths(fit_id, runner.RUN_ROOT)
        for suffix, path in (("model", artifact), ("model_receipt", receipt),
                             ("pos42_prediction", pred), ("pos42_prediction_receipt", pred_receipt)):
            if not path.is_file():
                raise VerifyError(f"Required original POS42 artifact is missing: {path}")
            hashes[f"E{origin}_{suffix}"] = _sha_file(path)
        for component in ("STRUCTURED", "SCALER", "OHE"):
            unit = f"{runner.CARD_ID}_E{origin}_{component}"
            artifact, receipt = runner._unit_paths(unit, runner.RUN_ROOT)
            if not artifact.is_file() or not receipt.is_file():
                raise VerifyError(f"Required cached preprocessor is absent: {unit}")
            hashes[f"E{origin}_{component}_artifact"] = _sha_file(artifact)
            hashes[f"E{origin}_{component}_receipt"] = _sha_file(receipt)
    return hashes


def _verify_external_frozen_inputs(freeze: Mapping[str, Any], runner: Any) -> dict[str, Any]:
    source_head = subprocess.run(["git", "-C", str(SOURCE_ROOT), "rev-parse", "HEAD"],
                                 check=True, capture_output=True, text=True).stdout.strip()
    if not _is_git_ancestor(SOURCE_ROOT, freeze.get("source_git_head"), source_head):
        raise VerifyError("Frozen external source commit is not an ancestor of the current source checkout")
    train_path = runner._contained_file(SOURCE_ROOT, runner.TRAIN_REL)
    train_sha = _sha_file(train_path)
    if (train_sha != freeze.get("official_train_sha256")
            or train_sha != getattr(runner, "TRAIN_SHA256", None)):
        raise VerifyError("Current Train.csv differs from the POS source and pilot freeze hashes")
    current_artifacts = _source_artifact_hashes(runner)
    if (current_artifacts != freeze.get("source_artifacts")
            or current_artifacts != freeze.get("local_sources")):
        raise VerifyError("Relevant external source/artifact hashes differ from the pilot freeze")
    return {"frozen_head": freeze["source_git_head"], "current_head": source_head,
            "train_sha256": train_sha, "relevant_artifact_hashes": "PASS"}


def _verify_source_supersession(freeze: Mapping[str, Any], implementation_hashes: Mapping[str, str]) -> dict[str, Any]:
    superseded_freeze = freeze.get("superseded_source_freeze", {})
    superseded_report = freeze.get("superseded_source_preflight_report", {})
    expected_freeze_path = SUPERSEDED_SOURCE_FREEZE_PATH.relative_to(ROOT).as_posix()
    expected_report_path = SUPERSEDED_SOURCE_PREFIT_PATH.relative_to(ROOT).as_posix()
    expected_receipt_path = SUPERSEDENCE_RECEIPT_PATH.relative_to(ROOT).as_posix()
    if (freeze.get("source_revision") != 2
            or superseded_freeze.get("path") != expected_freeze_path
            or superseded_report.get("path") != expected_report_path
            or freeze.get("supersession_receipt_path") != expected_receipt_path):
        raise VerifyError("Source freeze does not identify the reviewed R2 supersession package")
    old_freeze = _signed(SUPERSEDED_SOURCE_FREEZE_PATH, "freeze_signature")
    old_report = _read_json(SUPERSEDED_SOURCE_PREFIT_PATH)
    old_freeze_sha = _sha_file(SUPERSEDED_SOURCE_FREEZE_PATH)
    old_report_sha = _sha_file(SUPERSEDED_SOURCE_PREFIT_PATH)
    receipt = _signed(SUPERSEDENCE_RECEIPT_PATH, "receipt_sha256")
    current_freeze_sha = _sha_file(FREEZE_PATH)
    current_report = _read_json(PREFIT_REPORT)
    if (old_freeze_sha != superseded_freeze.get("sha256")
            or old_freeze["freeze_signature"] != superseded_freeze.get("freeze_signature")
            or old_report_sha != superseded_report.get("sha256")
            or old_report.get("status") != "PASS"
            or old_report.get("freeze_signature") != old_freeze["freeze_signature"]
            or old_report.get("freeze_sha256") != old_freeze_sha
            or old_report.get("query_targets_read") is not False
            or old_report.get("official_target_column_loaded") is not False
            or old_report.get("official_fit_units_started") != 0
            or old_report.get("official_ledger_mutated") is not False
            or old_report.get("test_or_sample_accessed") is not False
            or old_report.get("csv_created") is not False
            or receipt.get("status") != "PASS"
            or receipt.get("supersedes_freeze_signature") != old_freeze["freeze_signature"]
            or receipt.get("superseded_freeze_sha256") != old_freeze_sha
            or receipt.get("superseded_preflight_report_sha256") != old_report_sha
            or receipt.get("current_freeze_signature") != freeze.get("freeze_signature")
            or receipt.get("current_freeze_sha256") != current_freeze_sha
            or receipt.get("source_revision") != freeze.get("source_revision")
            or receipt.get("source_commit") != freeze.get("base_commit")
            or receipt.get("implementation_hashes") != dict(implementation_hashes)
            or receipt.get("target_blind_integrity_only") is not True
            or current_report.get("status") != "PASS"
            or current_report.get("freeze_signature") != freeze.get("freeze_signature")
            or current_report.get("freeze_sha256") != current_freeze_sha
            or current_report.get("source_revision") != freeze.get("source_revision")
            or current_report.get("implementation_hashes") != dict(implementation_hashes)
            or current_report.get("supersession_receipt_path") != expected_receipt_path
            or current_report.get("supersession_receipt_sha256") != _sha_file(SUPERSEDENCE_RECEIPT_PATH)
            or current_report.get("query_targets_read") is not False
            or current_report.get("official_target_column_loaded") is not False
            or current_report.get("official_fit_units_started") != 0
            or current_report.get("official_ledger_mutated") is not False
            or current_report.get("test_or_sample_accessed") is not False
            or current_report.get("csv_created") is not False):
        raise VerifyError("Independent source supersession or target-free prefit receipt failed")
    return {"superseded_freeze_sha256": old_freeze_sha,
            "superseded_preflight_report_sha256": old_report_sha,
            "current_freeze_sha256": current_freeze_sha,
            "receipt_sha256": _sha_file(SUPERSEDENCE_RECEIPT_PATH),
            "source_revision": freeze["source_revision"]}


def _rank50(raw: np.ndarray, cycles: np.ndarray) -> np.ndarray:
    result = np.empty(len(raw), dtype=np.float64)
    for cycle in np.unique(cycles):
        mask = cycles == cycle
        result[mask] = 0.5 * raw[mask] + 0.5 * (100.0 * rankdata(raw[mask], method="average") / int(mask.sum()))
    return result


def _apply_hinge(direct: np.ndarray, missing: np.ndarray,
                 parameters: Mapping[str, float]) -> np.ndarray:
    raw = np.asarray(direct, dtype=np.float64)
    output = raw.copy()
    present = ~np.asarray(missing, dtype=bool)
    p = raw[present]
    mapped = (float(parameters["intercept"])
              + float(parameters["slope_low"]) * np.minimum(p, 50.0)
              + float(parameters["slope_high"]) * np.maximum(p - 50.0, 0.0))
    output[present] = np.clip(mapped, 0.0, 100.0)
    return output


def _load_weights(origin: int, seed: int, fit_rows: np.ndarray,
                  fit_ids: np.ndarray, freeze: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if _sha_file(WEIGHTS_PATH) != freeze["weights_sha256"]:
        raise VerifyError("Frozen weight file hash mismatch")
    with np.load(WEIGHTS_PATH, allow_pickle=False) as saved:
        rows = saved[f"fit_rows_E{origin}"]
        ids = saved[f"fit_ids_E{origin}"].astype(str)
        counts = saved[f"counts_E{origin}_P{seed}"]
        weights = saved[f"weights_E{origin}_P{seed}"]
    expected_counts = np.random.default_rng(seed).poisson(1.0, size=len(fit_rows)).astype(np.uint8)
    expected_weights = expected_counts.astype(np.float64) * (len(fit_rows) / float(expected_counts.sum()))
    if (not np.array_equal(rows, fit_rows) or not np.array_equal(ids, fit_ids.astype(str))
            or not np.array_equal(counts, expected_counts) or not np.array_equal(weights, expected_weights)
            or abs(float(weights.sum()) - len(fit_rows)) > 1e-8 * len(fit_rows)):
        raise VerifyError(f"Poisson weights do not match frozen row order at E{origin} P{seed}")
    return counts, weights


def _ledger(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _validate_model(output_root: Path, fit_id: str, freeze_signature: str):
    artifact = output_root / "units" / f"{fit_id}.joblib"
    receipt_path = output_root / "receipts" / f"{fit_id}.json"
    if not artifact.is_file() or not receipt_path.is_file():
        raise VerifyError(f"Model artifact or receipt missing: {fit_id}")
    receipt = _read_json(receipt_path)
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    events = _ledger(output_root / "fit_ledger.jsonl")
    starts = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("fit_id") == fit_id]
    saves = [e for e in events if e.get("event") == "FIT_SAVED" and e.get("fit_id") == fit_id]
    failures = [e for e in events if e.get("event") == "FIT_FAILED" and e.get("fit_id") == fit_id]
    if (receipt.get("freeze_signature") != freeze_signature
            or receipt.get("artifact_sha256") != _sha_file(artifact)
            or receipt.get("receipt_sha256") != _canonical_sha(body)
            or len(starts) != 1 or len(saves) != 1 or failures
            or saves[0].get("receipt_sha256") != _sha_file(receipt_path)):
        raise VerifyError(f"Model receipt or append-only ledger mismatch: {fit_id}")
    return joblib.load(artifact), receipt


def verify_synthetic(root: Path) -> dict[str, Any]:
    root = root.resolve()
    spec = _signed(root / "SYNTHETIC_SPEC.json", "freeze_signature")
    if spec.get("experiment_id") != CARD_ID or spec.get("synthetic_only") is not True:
        raise VerifyError("Synthetic identity is not isolated from official fitting")
    events = _ledger(root / "fit_ledger.jsonl")
    starts = [e.get("fit_id") for e in events if e.get("event") == "FIT_STARTED"]
    saves = [e.get("fit_id") for e in events if e.get("event") == "FIT_SAVED"]
    fails = [e for e in events if e.get("event") == "FIT_FAILED"]
    if starts != list(FIT_IDS) or saves != list(FIT_IDS) or fails:
        raise VerifyError("Synthetic append-only ledger does not show exactly four completed fits")
    model_checks: dict[str, Any] = {}
    metric_values: dict[str, float] = {}
    for origin in ORIGINS:
        data = _explicit_npz(root / f"data_E{origin}.npz", (
            "X_fit_data", "X_fit_indices", "X_fit_indptr", "X_fit_shape",
            "X_query_data", "X_query_indices", "X_query_indptr", "X_query_shape",
            "y_fit", "y_query", "pos42_raw", "baseline", "cycles", "missing",
            "hinge_parameters"))
        X_fit = sparse.csr_matrix((data["X_fit_data"], data["X_fit_indices"], data["X_fit_indptr"]),
                                  shape=tuple(data["X_fit_shape"].astype(int)))
        X_query = sparse.csr_matrix((data["X_query_data"], data["X_query_indices"], data["X_query_indptr"]),
                                    shape=tuple(data["X_query_shape"].astype(int)))
        weight_file = _explicit_npz(root / f"weights_E{origin}.npz", ("counts_P17", "weights_P17", "counts_P137", "weights_P137"))
        pred = _explicit_npz(root / "predictions" / f"E{origin}.npz", (
            "pos42_raw", "replica17_raw", "replica137_raw", "mean_raw", "mapped_pos",
            "full_direct", "same_history_hinge", "missing_mask", "baseline", "query_cycles",
            "synthetic_targets", "fit_targets"))
        if (not np.array_equal(pred["synthetic_targets"], data["y_query"])
                or not np.array_equal(pred["fit_targets"], data["y_fit"])):
            raise VerifyError(f"Synthetic target artifact readback mismatch at E{origin}")
        raw: dict[int, np.ndarray] = {}
        for seed in SEEDS:
            fit_id = f"{CARD_ID}_E{origin}_P{seed}"
            model, receipt = _validate_model(root, fit_id, spec["freeze_signature"])
            counts = weight_file[f"counts_P{seed}"]
            weights = weight_file[f"weights_P{seed}"]
            if (receipt.get("poisson_counts_sha256") != _sha_array(counts)
                    or receipt.get("sample_weight_sha256") != _sha_array(weights)
                    or receipt.get("fit_target_sha256") != _sha_array(data["y_fit"])
                    or abs(float(weights.sum()) - len(data["y_fit"])) > 1e-8 * len(data["y_fit"])):
                raise VerifyError(f"Synthetic fit receipt inputs or sample weights fail at {fit_id}")
            with threadpool_limits(limits=THREADS):
                raw[seed] = np.asarray(model.predict(X_query), dtype=np.float64)
            key = f"replica{seed}_raw"
            if not np.array_equal(raw[seed], pred[key]):
                raise VerifyError(f"Saved synthetic query prediction differs from readback model: {fit_id}")
            model_checks[fit_id] = {"artifact_sha256": receipt["artifact_sha256"],
                                    "receipt_sha256": receipt["receipt_sha256"],
                                    "prediction_sha256": _sha_array(raw[seed])}
        params_values = data["hinge_parameters"].astype(np.float64)
        parameters = {"intercept": float(params_values[0]), "slope_low": float(params_values[1]),
                      "slope_high": float(params_values[2])}
        mean_raw = (pred["pos42_raw"] + raw[17] + raw[137]) / 3.0
        mapped = _rank50(mean_raw, data["cycles"].astype(np.int64))
        direct = np.clip(0.75 * data["baseline"] + 0.25 * mapped, 0.0, 100.0)
        missing = data["missing"].astype(bool)
        direct[missing] = data["baseline"][missing]
        hinge = _apply_hinge(direct, missing, parameters)
        for name, actual in (("mean_raw", mean_raw), ("mapped_pos", mapped),
                             ("full_direct", direct), ("same_history_hinge", hinge)):
            if not np.array_equal(pred[name], actual):
                raise VerifyError(f"Synthetic aggregation/hinge replay differs: E{origin} {name}")
        if not np.array_equal(hinge[missing].view("<u8"), data["baseline"][missing].astype("<f8").view("<u8")):
            raise VerifyError(f"Synthetic missing fallback is not bit exact at E{origin}")
        errors = np.abs(data["y_query"] - hinge)
        metric_values[str(origin)] = float(errors.mean())
        metric_sha = str(_sha_array(errors))
        if not metric_sha:
            raise VerifyError("Synthetic metric hash is empty")
    return {"status": "PASS", "fit_units": 4, "fit_ids": list(FIT_IDS),
            "official_fit_units_started": 0, "models": model_checks,
            "synthetic_mae": metric_values, "target_arrays": "synthetic-only"}


def _verify_official_prelabel_locked() -> dict[str, Any]:
    freeze = _signed(FREEZE_PATH, "freeze_signature")
    if freeze.get("status") != "PREFIT_FROZEN" or freeze.get("experiment_id") != CARD_ID:
        raise VerifyError("Source freeze identity/status failed")
    implementation_hashes = {
        "runner": _sha_file(ROOT / "scripts" / "reg_pos_poisson_bag2_v1.py"),
        "verifier": _sha_file(Path(__file__)),
        "test": _sha_file(ROOT / "tests" / "test_reg_pos_poisson_bag2_v1.py"),
    }
    if implementation_hashes != freeze.get("implementation_hashes"):
        raise VerifyError("Implementation or test source differs from the frozen prefit package")
    source_supersession_check = _verify_source_supersession(freeze, implementation_hashes)
    if _sha_file(POSITION_FREEZE) != freeze["position_freeze_sha256"]:
        raise VerifyError("POS source freeze changed")
    source_head = subprocess.run(["git", "-C", str(SOURCE_ROOT), "rev-parse", "HEAD"],
                                 check=True, capture_output=True, text=True).stdout.strip()
    if not _is_git_ancestor(SOURCE_ROOT, freeze.get("source_git_head"), source_head):
        raise VerifyError("Frozen external source commit is not an ancestor of the current source checkout")
    root_pass = _signed(ROOT_SOURCE_PASS, "approval_sha256")
    if (root_pass.get("status") != "PASS"
            or root_pass.get("freeze_signature") != freeze["freeze_signature"]
            or root_pass.get("fit_ids") != list(FIT_IDS)
            or root_pass.get("fit_cap") != len(FIT_IDS)
            or root_pass.get("reviewer") in (None, "", "self")
            or root_pass.get("score_access_allowed") is not False):
        raise VerifyError("Four-fit root source approval is missing or does not match")
    reviewed_commit, current_head = _verify_approved_source_commit(ROOT, freeze.get("base_commit"), root_pass)
    prediction_freeze = _signed(PREDICTION_FREEZE, "prediction_signature")
    ledger_path = OFFICIAL_RUN / "fit_ledger.jsonl"
    if (not ledger_path.is_file()
            or prediction_freeze.get("source_freeze_signature") != freeze["freeze_signature"]
            or prediction_freeze.get("fit_ids") != list(FIT_IDS)
            or prediction_freeze.get("fit_units_started") != len(FIT_IDS)
            or prediction_freeze.get("fit_units_saved") != len(FIT_IDS)
            or prediction_freeze.get("fit_units_failed") != 0
            or prediction_freeze.get("fit_ledger_sha256") != _sha_file(ledger_path)):
        raise VerifyError("Prediction freeze identity, four-fit cap, or official ledger binding failed")
    events = _ledger(OFFICIAL_RUN / "fit_ledger.jsonl")
    starts = [e.get("fit_id") for e in events if e.get("event") == "FIT_STARTED"]
    saves = [e.get("fit_id") for e in events if e.get("event") == "FIT_SAVED"]
    failures = [e for e in events if e.get("event") == "FIT_FAILED"]
    if starts != list(FIT_IDS) or saves != list(FIT_IDS) or failures:
        raise VerifyError("Official ledger does not contain exactly four successful ordered fits")
    if _sha_file(WEIGHTS_PATH) != freeze["weights_sha256"]:
        raise VerifyError("Poisson weight file hash differs from prefit freeze")
    runner = _runner()
    external_source_check = _verify_external_frozen_inputs(freeze, runner)
    current_source_branch = subprocess.run(
        ["git", "-C", str(SOURCE_ROOT), "branch", "--show-current"],
        check=True, capture_output=True, text=True).stdout.strip()
    current_source_status_sha = _canonical_sha(subprocess.run(
        ["git", "-C", str(SOURCE_ROOT), "status", "--porcelain"],
        check=True, capture_output=True, text=True).stdout.splitlines())
    position_freeze = _signed(POSITION_FREEZE, "freeze_signature")
    frame = runner._load_feature_frame(SOURCE_ROOT)
    if "threat_rank" in frame.columns:
        raise VerifyError("Regression target column appeared in the prelabel feature frame")
    hinge_parameters: dict[int, Mapping[str, float]] = {}
    for origin, path in HINGE_PARAMS_PATHS.items():
        value = _read_json(path)
        if value.get("status") != "FIT_COMPLETE":
            raise VerifyError(f"Saved hinge state is not complete: {path}")
        hinge_parameters[origin] = value["saved_hinge_parameters"]
    origin_checks: dict[str, Any] = {}
    manifest_artifacts = prediction_freeze.get("artifacts", {})
    for origin in ORIGINS:
        item = runner._load_origin(SOURCE_ROOT, frame, origin)
        expected = freeze["origins"][str(origin)]
        if (_sha_array(item.fit_rows) != expected["fit_row_indices_sha256"]
                or _sha_array(item.fit_ids) != expected["fit_ids_sha256"]
                or _sha_array(item.query_rows) != expected["query_row_indices_sha256"]
                or _sha_array(item.query_ids) != expected["query_ids_sha256"]):
            raise VerifyError(f"Frozen fit/query identity changed at E{origin}")
        for component in ("STRUCTURED", "SCALER", "OHE"):
            fit_id = f"{runner.CARD_ID}_E{origin}_{component}"
            runner._validate_unit(runner.RUN_ROOT, fit_id, position_freeze)
        prefix = f"{runner.CARD_ID}_E{origin}"
        fit_frame = frame.iloc[item.fit_rows].loc[:, list(runner.HYBRID_FEATURES)].copy()
        query_frame = frame.iloc[item.query_rows].loc[:, list(runner.HYBRID_FEATURES)].copy()
        structured_fit, structured_query, _ = runner._fit_or_transform_structured(
            prefix, origin, position_freeze, fit_frame, query_frame, output_root=runner.RUN_ROOT)
        X_fit = runner._combine_design(structured_fit, item.fit_bag, item.fit_position, item.fit_missing)
        X_query = runner._combine_design(structured_query, item.query_bag, item.query_position, item.query_missing)
        if (_sha_csr(X_fit) != expected["fit_design_sha256"]
                or _sha_csr(X_query) != expected["query_design_sha256"]):
            raise VerifyError(f"Fit/query feature matrix differs from freeze at E{origin}")
        pred_path = OFFICIAL_RUN / "predictions" / f"E{origin}.npz"
        receipt_path = OFFICIAL_RUN / "predictions" / f"E{origin}.json"
        if _sha_file(pred_path) != manifest_artifacts[str(origin)]["prediction_sha256"]:
            raise VerifyError(f"Candidate prediction is not bound by prediction freeze at E{origin}")
        pred = _explicit_npz(pred_path, ("query_rows", "query_ids", "query_cycles", "missing_mask",
                                         "r06_baseline", "pos42_raw", "replica17_raw", "replica137_raw",
                                         "mean_raw", "mapped_pos", "full_direct", "same_history_hinge"))
        receipt = _read_json(receipt_path)
        receipt_body = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
        if (receipt.get("receipt_sha256") != _canonical_sha(receipt_body)
                or receipt.get("prediction_sha256") != _sha_file(pred_path)
                or receipt.get("query_targets_read") is not False):
            raise VerifyError(f"Candidate prediction receipt failed at E{origin}")
        if (not np.array_equal(pred["query_rows"], item.query_rows)
                or not np.array_equal(pred["query_ids"].astype(str), item.query_ids.astype(str))
                or not np.array_equal(pred["query_cycles"], item.query_cycles)
                or not np.array_equal(pred["missing_mask"], item.query_missing)
                or not np.array_equal(pred["r06_baseline"], item.query_baseline)):
            raise VerifyError(f"Candidate query identity or R06 baseline differs at E{origin}")
        base_id = f"{runner.CARD_ID}_E{origin}_REAL"
        base_model, _ = runner._validate_unit(runner.RUN_ROOT, base_id, position_freeze)
        with threadpool_limits(limits=THREADS):
            base_raw = np.asarray(base_model.predict(X_query), dtype=np.float64)
        saved_base = _explicit_npz(runner._prediction_paths(base_id, runner.RUN_ROOT)[0],
                                   ("raw_auxiliary", "prediction"))
        if not np.array_equal(base_raw, saved_base["raw_auxiliary"]) or not np.array_equal(pred["pos42_raw"], base_raw):
            raise VerifyError(f"POS42 original raw prediction differs at E{origin}")
        replica_hashes: dict[str, str] = {}
        for seed in SEEDS:
            fit_id = f"{CARD_ID}_E{origin}_P{seed}"
            counts, weights = _load_weights(origin, seed, item.fit_rows, item.fit_ids, freeze)
            model, model_receipt = _validate_model(OFFICIAL_RUN, fit_id, freeze["freeze_signature"])
            if (model_receipt.get("parameters") != freeze["resolved_parameters"]
                    or model_receipt.get("fit_design_sha256") != _sha_csr(X_fit)
                    or model_receipt.get("sample_weight_sha256") != _sha_array(weights)
                    or model_receipt.get("poisson_counts_sha256") != _sha_array(counts)
                    or model_receipt.get("fit_rows") != len(item.fit_rows)
                    or model_receipt.get("feature_count") != X_fit.shape[1]):
                raise VerifyError(f"Replica receipt inputs differ at {fit_id}")
            if (model_receipt.get("resource_before", {}).get("projected_free_bytes", 0) < 512 * 1024**2
                    or model_receipt.get("resource_after", {}).get("available_ram_bytes", 0) < 512 * 1024**2):
                raise VerifyError(f"Replica fit resource reserve is not documented at {fit_id}")
            with threadpool_limits(limits=THREADS):
                raw = np.asarray(model.predict(X_query), dtype=np.float64)
            key = f"replica{seed}_raw"
            if not np.array_equal(raw, pred[key]):
                raise VerifyError(f"Saved replica prediction differs from loaded model at {fit_id}")
            replica_hashes[str(seed)] = _sha_array(raw)
        mean_raw = (base_raw + pred["replica17_raw"] + pred["replica137_raw"]) / 3.0
        mapped = _rank50(mean_raw, item.query_cycles.astype(np.int64))
        direct = np.clip(0.75 * item.query_baseline + 0.25 * mapped, 0.0, 100.0)
        direct[item.query_missing] = item.query_baseline[item.query_missing]
        hinge = _apply_hinge(direct, item.query_missing, hinge_parameters[origin])
        for name, actual in (("mean_raw", mean_raw), ("mapped_pos", mapped),
                             ("full_direct", direct), ("same_history_hinge", hinge)):
            if not np.array_equal(pred[name], actual):
                raise VerifyError(f"Independent formula replay differs at E{origin}: {name}")
        if not np.array_equal(hinge[item.query_missing].astype("<f8").view("<u8"),
                              item.query_baseline[item.query_missing].astype("<f8").view("<u8")):
            raise VerifyError(f"Candidate missing fallback is not bit exact at E{origin}")
        local = _explicit_npz(LOCAL_PREDICTION_PATHS[origin], (
            "physical_rows", "record_ids", "prediction_origins", "original_bundle_indices",
            "cycles", "missing_mask", "direct_predictions",
            "hinge_baseline"))
        index_by_row = {int(row): i for i, row in enumerate(item.query_rows)}
        positions = np.asarray([index_by_row[int(row)] for row in local["physical_rows"]], dtype=np.int64)
        if (not np.all(local["prediction_origins"] == origin)
                or len(set(local["original_bundle_indices"].tolist())) != len(positions)
                or not np.array_equal(item.query_ids[positions].astype(str), local["record_ids"].astype(str))
                or not np.array_equal(item.query_cycles[positions], local["cycles"])
                or not np.array_equal(direct[positions], local["direct_predictions"])
                or not np.array_equal(_apply_hinge(direct[positions], local["missing_mask"], hinge_parameters[origin]),
                                      local["hinge_baseline"])):
            raise VerifyError(f"Saved same-history comparator replay differs at E{origin}")
        origin_checks[str(origin)] = {
            "fit_rows": len(item.fit_rows), "query_rows": len(item.query_rows),
            "fit_design_sha256": _sha_csr(X_fit), "query_design_sha256": _sha_csr(X_query),
            "pos42_raw_sha256": _sha_array(base_raw), "replica_raw_sha256": replica_hashes,
            "full_direct_sha256": _sha_array(direct[positions]),
            "same_history_hinge_sha256": _sha_array(hinge[positions]),
            "score_rows": len(positions), "score_targets_read": False,
            "missing_rows": int(local["missing_mask"].sum()),
        }
    report = {
        "schema_version": 1, "experiment_id": CARD_ID, "status": "PASS",
        "source_freeze_signature": freeze["freeze_signature"],
        "prediction_signature": prediction_freeze["prediction_signature"],
        "official_fit_units": 4, "official_fit_ids": list(FIT_IDS),
        "query_targets_read": False, "official_target_column_loaded": False,
        "test_or_sample_accessed": False, "csv_created": False,
        "independent_model_readback": True, "formula_replay": True,
        "missing_fallback_bit_exact": True, "origins": origin_checks,
        "reviewed_source_commit": reviewed_commit, "current_root_head": current_head,
        "external_source_check": external_source_check,
        "source_supersession_check": source_supersession_check,
        "external_source_branch_informational": current_source_branch,
        "external_source_status_sha256_informational": current_source_status_sha,
        "verified_utc": datetime.now(timezone.utc).isoformat(),
    }
    PRELABEL_REPORT.parent.mkdir(parents=True, exist_ok=True)
    PRELABEL_REPORT.write_text(json.dumps(report, sort_keys=True, ensure_ascii=False,
                                          allow_nan=False, indent=2) + "\n", encoding="utf-8")
    return report


def verify_official_prelabel() -> dict[str, Any]:
    with _SharedHeavyJobLock():
        return _verify_official_prelabel_locked()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--synthetic-dir", type=Path)
    group.add_argument("--official-prelabel", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = verify_synthetic(args.synthetic_dir) if args.synthetic_dir else verify_official_prelabel()
        print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False))
        return 0
    except (VerifyError, OSError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
