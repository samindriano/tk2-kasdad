"""Source-gated two-replica Poisson bagging of positional gamma.

Official commands require a committed source/input freeze and an explicit
committed Sol root prefit pass.  The ``synthetic`` command is isolated and does
not touch official Train, Test, cached labels, ledgers, or Kaggle.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import csv
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping

import numpy as np
import psutil
from scipy import sparse
from scipy.special import expit
from threadpoolctl import threadpool_limits


AUDIT = Path(__file__).resolve().parents[1]
REPORT = AUDIT / "reports/simplification_v1/CLS_POS_POISSON_BAG2_V1"
PROTOCOL = REPORT / "PROTOCOL.md"
FREEZE_PATH = REPORT / "PREFIT_FREEZE.json"
ROOT_PREFIT_PASS = REPORT / "ROOT_PREFIT_PASS.json"
RUN_ROOT = AUDIT / "experiments/CLS_POS_POISSON_BAG2_V1"
RESERVATION_PATH = RUN_ROOT / "RESERVATION.json"
LEDGER_PATH = RUN_ROOT / "fit_ledger.jsonl"
PREDICTIONS_READY = RUN_ROOT / "PREDICTIONS_READY.json"
RESULTS_PATH = RUN_ROOT / "RESULTS.json"
EXPECTED_TRAIN_SHA = "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c"
MIN_FIT_AVAILABLE = 1024**3
MIN_FIT_RESERVE = 512 * 1024**2
MIN_CACHE_AVAILABLE = 512 * 1024**2
MIN_CACHE_RESERVE = 256 * 1024**2
FIT_ID_TO_ORIGIN_SEED = {
    "CLS_POS_POISSON_BAG2_V1_E67_P17": (67, 17),
    "CLS_POS_POISSON_BAG2_V1_E67_P137": (67, 137),
    "CLS_POS_POISSON_BAG2_V1_E75_P17": (75, 17),
    "CLS_POS_POISSON_BAG2_V1_E75_P137": (75, 137),
}
SOURCE_FILES = (
    "src/models/cls_pos_poisson_bag2_v1.py",
    "scripts/cls_pos_poisson_bag2_v1.py",
    "scripts/verify_cls_pos_poisson_bag2_v1.py",
    "tests/test_cls_pos_poisson_bag2_v1.py",
    "reports/simplification_v1/CLS_POS_POISSON_BAG2_V1/PROTOCOL.md",
    "reports/simplification_v1/CLS_POS_POISSON_BAG2_V1/SOURCE_READY.md",
)
EXTERNAL_SOURCE_FILES = (
    "scripts/optimization_v3_positional_bag_offset_v1.py",
    "scripts/verify_optimization_v3_positional_bag_offset_v1.py",
    "src/models/positional_bag_offset_v1.py",
    "tests/test_optimization_v3_positional_bag_offset_v1.py",
    "reports/optimization_v3/positional_bag_offset_v1/RECIPE.json",
    "reports/optimization_v3/positional_text_diagnostic_v1/SOL_OFFSET_PILOT_AUTHORIZATION.md",
    "scripts/optimization_v3_a02_neighbors.py",
    "scripts/optimization_v3_a02_neighbors_verify.py",
    "src/models/classification_a02_neighbors.py",
    "reports/optimization_v3/a02_neighbors_v1/technical_resume_receipt.json",
    "reports/optimization_v3/a02_neighbors_v1/verification.json",
    "reports/optimization_v3/a02_neighbors_v1/preflight.json",
    "reports/optimization_v3/cls_simple_present_threshold_final_v1/PRE_FIT_FREEZE_V2.json",
)
BASE_INPUT_FILES = (
    "data/raw/classification/Train.csv",
    "reports/optimization_v3/positional_text_pilot_v1/FIT_ACCOUNTING.json",
    "experiments/runs/optimization_v3_research_next_v1/POSITIONAL_TEXT_PILOT_V1/FREEZE.json",
    "reports/optimization_v3/positional_bag_offset_v1/RECIPE.json",
    "experiments/runs/optimization_v3_research_next_v1/POS_BAG_OFFSET_V1/FREEZE.json",
    "experiments/runs/optimization_v3_research_next_v1/POS_BAG_OFFSET_V1/RESERVATION.json",
    "reports/optimization_v3/cls_simple_present_threshold_final_v1/PRE_FIT_FREEZE_V2.json",
)
PILOT_ROOT_REL = "experiments/runs/optimization_v3_research_next_v1/POSITIONAL_TEXT_PILOT_V1"
POS_RUN_REL = "experiments/runs/optimization_v3_research_next_v1/POS_BAG_OFFSET_V1"
A02_RUN_REL = "experiments/runs/optimization_v3_a02_neighbors_v1"
ORIGINS = (67, 75)
MATRIX_NAMES = ("fit_base", "fit_position", "score_base", "score_position")
RESERVATION_FIT_IDS = list(FIT_ID_TO_ORIGIN_SEED)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: Any) -> str:
    array = np.asarray(value)
    header = canonical_sha256({"shape": list(array.shape), "dtype": str(array.dtype)})
    digest = hashlib.sha256(header.encode("ascii"))
    if array.dtype.kind in "OUS":
        digest.update(json.dumps(array.astype(str).tolist(), ensure_ascii=False,
                                 separators=(",", ":")).encode("utf-8"))
    else:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def sha256_csr(value: sparse.spmatrix) -> str:
    matrix = sparse.csr_matrix(value, dtype=np.float64, copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    digest = hashlib.sha256()
    digest.update(canonical_sha256({"shape": list(matrix.shape), "dtype": str(matrix.dtype),
                                   "format": "csr"}).encode("ascii"))
    digest.update(np.ascontiguousarray(matrix.indptr, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(matrix.indices, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(matrix.data, dtype=np.float64).tobytes())
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    require(Path(path).is_file(), f"Required JSON missing: {path}")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                          indent=2) + "\n").encode("utf-8")
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def write_once(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                          indent=2) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def atomic_npz(path: Path, **arrays: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _repo_rel(root: Path, path: Path) -> str:
    resolved_root = root.resolve()
    resolved = Path(path).resolve()
    require(resolved == resolved_root or resolved_root in resolved.parents,
            f"Frozen path escapes official repository: {resolved}")
    return resolved.relative_to(resolved_root).as_posix()


def _external_file_inventory(repository: Path) -> dict[str, str]:
    repository = Path(repository).resolve()
    paths: set[str] = set(EXTERNAL_SOURCE_FILES) | set(BASE_INPUT_FILES)
    pilot_root = repository / PILOT_ROOT_REL
    accounting = read_json(repository / "reports/optimization_v3/positional_text_pilot_v1/FIT_ACCOUNTING.json")
    accounting_by_id = {unit["fit_id"]: unit for unit in accounting["fit_units"]}
    for origin in ORIGINS:
        suffix = f"E{origin}"
        manifest_rel = f"{PILOT_ROOT_REL}/MATRIX_MANIFEST_{suffix}.json"
        prediction_rel = f"{PILOT_ROOT_REL}/{suffix}/predictions_{suffix}.npz"
        paths.update({manifest_rel, prediction_rel})
        manifest = read_json(repository / manifest_rel)
        for matrix_name in MATRIX_NAMES:
            item = manifest["matrices"][matrix_name]
            matrix_path = Path(str(item["path"]))
            if not matrix_path.is_absolute():
                matrix_path = repository / matrix_path
            paths.add(_repo_rel(repository, matrix_path))
        bag_id = f"POSITIONAL_TEXT_PILOT_V1_{suffix}_BAG_LR"
        paths.add(f"{PILOT_ROOT_REL}/units/{bag_id}.joblib")
        paths.add(f"{PILOT_ROOT_REL}/receipts/{bag_id}.json")
        gamma_id = f"POS_BAG_OFFSET_V1_{suffix}_REAL"
        paths.add(f"{POS_RUN_REL}/units/{gamma_id}.npz")
        paths.add(f"{POS_RUN_REL}/receipts/{gamma_id}.json")
        paths.add(f"{POS_RUN_REL}/{suffix}/predictions_{suffix}.npz")
        a02 = f"{A02_RUN_REL}/{suffix}/observed"
        paths.update({f"{a02}/fit_receipt.json", f"{a02}/fit_state.json", f"{a02}/model.joblib",
                      f"{a02}/predictions.json", f"{a02}/predictions.npz"})
        require(bag_id in accounting_by_id, f"Saved C2_B bag accounting absent for {suffix}")
    result: dict[str, str] = {}
    for relative in sorted(paths):
        path = repository / relative
        require(path.is_file(), f"Frozen input missing: {relative}")
        result[relative] = sha256_file(path)
    require(all("Test.csv" not in p and "SampleSubmission" not in p for p in result),
            "Test/SampleSubmission must not be inputs to this validation-only lane")
    return result


def _validate_metadata_only(repository: Path, file_hashes: Mapping[str, str]) -> dict[str, Any]:
    repository = Path(repository).resolve()
    train_rel = "data/raw/classification/Train.csv"
    require(file_hashes[train_rel] == EXPECTED_TRAIN_SHA, "Official Train hash differs from the frozen source")
    recipe = read_json(repository / "reports/optimization_v3/positional_bag_offset_v1/RECIPE.json")
    require(recipe["input"]["sha256"] == EXPECTED_TRAIN_SHA and recipe["input"]["rows"] == 50608
            and recipe["input"]["target"] == "artifact_signal", "POS_BAG_OFFSET Train recipe mismatch")
    offset_freeze = read_json(repository / f"{POS_RUN_REL}/FREEZE.json")
    offset_body = dict(offset_freeze)
    offset_signature = offset_body.pop("freeze_signature", None)
    require(offset_signature == canonical_sha256(offset_body), "POS_BAG_OFFSET freeze signature mismatch")
    require(offset_freeze["card_id"] == "POS_BAG_OFFSET_V1"
            and offset_freeze["fit_ids"] == [
                "POS_BAG_OFFSET_V1_E67_REAL", "POS_BAG_OFFSET_V1_E67_SHUFFLED",
                "POS_BAG_OFFSET_V1_E75_REAL", "POS_BAG_OFFSET_V1_E75_SHUFFLED"],
            "Original POS fit namespace differs from recipe")
    pilot_freeze = read_json(repository / f"{PILOT_ROOT_REL}/FREEZE.json")
    require(pilot_freeze["input_sha256"] == EXPECTED_TRAIN_SHA,
            "Positional feature pilot freeze has a different Train")
    threshold = read_json(repository / "reports/optimization_v3/cls_simple_present_threshold_final_v1/PRE_FIT_FREEZE_V2.json")
    require(threshold["recipe"]["missing_decision"] == "pA >= 0.53230976129477",
            "Frozen missing-stream threshold differs")
    a02_checks = {}
    for origin in ORIGINS:
        suffix = f"E{origin}"
        a02 = repository / f"{A02_RUN_REL}/{suffix}/observed"
        fit = read_json(a02 / "fit_receipt.json")
        pred = read_json(a02 / "predictions.json")
        require(fit["query_labels_passed_to_fit"] is False and fit["test_data_used"] is False
                and pred["query_labels_passed_to_prediction"] is False,
                f"A02 query-label boundary receipt failed for {suffix}")
        require(fit["origin_cycle"] == origin and pred["origin_cycle"] == origin,
                f"A02 origin receipt mismatch for {suffix}")
        require(pred["npz_sha256"] == file_hashes[f"{A02_RUN_REL}/{suffix}/observed/predictions.npz"],
                f"A02 cache receipt hash mismatch for {suffix}")
        a02_checks[suffix] = {"fit_id": fit["fit_id"], "query_labels_passed_to_fit": False,
                              "query_labels_passed_to_prediction": False,
                              "prediction_cache_sha256": pred["npz_sha256"]}
    return {"train_sha256": EXPECTED_TRAIN_SHA, "train_rows": 50608,
            "offset_freeze_signature": offset_signature,
            "pilot_freeze_signature": pilot_freeze["freeze_signature"],
            "threshold": threshold["recipe"]["missing_decision"],
            "threshold_origin_specific_prefix_states": "NOT_FOUND_IN_SCOPED_SAVED_STATES",
            "a02_boundary_checks": a02_checks,
            "a02_prediction_cache_labels_member_present_but_never_read": True}


def create_prefit_freeze(repository: Path) -> dict[str, Any]:
    repository = Path(repository).resolve()
    require(not FREEZE_PATH.exists(), "Prefit freeze is write-once and already exists")
    require(_git(AUDIT, "status", "--porcelain") == "", "Commit source and synthetic QA before freezing inputs")
    source_commit = _git(AUDIT, "rev-parse", "HEAD")
    source_hashes = {rel: sha256_file(AUDIT / rel) for rel in SOURCE_FILES}
    external_hashes = _external_file_inventory(repository)
    metadata = _validate_metadata_only(repository, external_hashes)
    external_status = _git(repository, "status", "--porcelain", "--untracked-files=all").splitlines()
    body = {
        "schema_version": 1,
        "candidate_id": "CLS_POS_POISSON_BAG2_V1",
        "status": "FROZEN_SOURCE_AND_INPUTS_WAITING_ROOT_PREFIT_APPROVAL",
        "audit_source_root": str(AUDIT.resolve()),
        "audit_source_commit": source_commit,
        "audit_branch": _git(AUDIT, "branch", "--show-current"),
        "audit_source_hashes": source_hashes,
        "official_repository_root": str(repository),
        "official_repository_head_at_freeze": _git(repository, "rev-parse", "HEAD"),
        "official_repository_untracked_status_at_freeze": external_status,
        "external_source_hashes": {rel: external_hashes[rel] for rel in EXTERNAL_SOURCE_FILES},
        "input_file_hashes": {rel: external_hashes[rel] for rel in external_hashes if rel not in EXTERNAL_SOURCE_FILES},
        "metadata_verification": metadata,
        "fit_cap": {"predictor_fits": 4, "preprocessing_fits": 0,
                    "fit_ids": RESERVATION_FIT_IDS, "no_retries": True},
        "fixed_recipe": {"origins": {"E67": {"fit_rows": 29705, "fit_cycles": [47, 67]},
                                      "E75": {"fit_rows": 42117, "fit_cycles": [47, 75]}},
                         "seeds": [17, 137], "poisson_lambda": 1.0,
                         "weight_normalization": "sum(weights)=n_fit",
                         "ridge_c": 1.0, "threshold_present": 0.50,
                         "threshold_missing": 0.53230976129477,
                         "max_threads": 2, "minimum_fit_start_available_bytes": MIN_FIT_AVAILABLE,
                         "minimum_post_fit_available_bytes": MIN_FIT_RESERVE},
        "data_boundary": {"fit_targets_read": False, "query_targets_read": False,
                          "test_csv_opened": False, "sample_submission_opened": False,
                          "a02_npz_label_member_opened": False},
    }
    body["freeze_signature"] = canonical_sha256(body)
    write_once(FREEZE_PATH, body)
    return body


def verify_prefit_freeze(repository: Path) -> dict[str, Any]:
    repository = Path(repository).resolve()
    freeze = read_json(FREEZE_PATH)
    body = dict(freeze)
    signature = body.pop("freeze_signature", None)
    require(signature == canonical_sha256(body), "Candidate prefit freeze signature mismatch")
    require(freeze["candidate_id"] == "CLS_POS_POISSON_BAG2_V1"
            and freeze["official_repository_root"] == str(repository), "Candidate/repository freeze identity mismatch")
    require(_git(AUDIT, "merge-base", "--is-ancestor", freeze["audit_source_commit"],
                 _git(AUDIT, "rev-parse", "HEAD")) == "", "Candidate source commit is not an ancestor")
    for relative, expected in freeze["audit_source_hashes"].items():
        require(sha256_file(AUDIT / relative) == expected, f"Candidate source file changed: {relative}")
        committed = subprocess.check_output(["git", "-C", str(AUDIT), "show",
                                             f"{freeze['audit_source_commit']}:{relative}"])
        require(hashlib.sha256(committed).hexdigest() == expected,
                f"Frozen source blob differs from committed source: {relative}")
    for relative, expected in {**freeze["external_source_hashes"], **freeze["input_file_hashes"]}.items():
        require(sha256_file(repository / relative) == expected, f"External frozen file changed: {relative}")
    require(all("Test.csv" not in p and "SampleSubmission" not in p
                for p in freeze["input_file_hashes"]), "Test/Sample file entered input freeze")
    return freeze


def verify_root_prefit_pass(freeze: Mapping[str, Any]) -> dict[str, Any]:
    require(ROOT_PREFIT_PASS.is_file(), "Root prefit approval is absent; official fits remain blocked")
    relative = ROOT_PREFIT_PASS.relative_to(AUDIT).as_posix()
    committed = subprocess.check_output(["git", "-C", str(AUDIT), "show", f"HEAD:{relative}"])
    on_disk = ROOT_PREFIT_PASS.read_bytes()
    require(committed.replace(b"\r\n", b"\n") == on_disk.replace(b"\r\n", b"\n"),
            "Root prefit approval is not committed exactly")
    approval = json.loads(on_disk.decode("utf-8"))
    require(set(approval) == {"status", "candidate_id", "freeze_signature", "audit_source_commit",
                              "fit_ids", "predictor_fit_cap", "validation_label_scoring",
                              "test_or_sample_access", "kaggle_upload"}
            and approval.get("status") == "PASS_ROOT_PREFIT_APPROVED"
            and approval.get("candidate_id") == "CLS_POS_POISSON_BAG2_V1"
            and approval.get("freeze_signature") == freeze["freeze_signature"]
            and approval.get("audit_source_commit") == freeze["audit_source_commit"]
            and approval.get("fit_ids") == RESERVATION_FIT_IDS
            and approval.get("predictor_fit_cap") == 4
            and isinstance(approval.get("validation_label_scoring"), bool)
            and approval.get("test_or_sample_access") is False
            and approval.get("kaggle_upload") is False,
            "Root prefit approval identity, fit cap, or boundary mismatch")
    return approval


@contextmanager
def _os_lock(path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+b")
    stream.seek(0, os.SEEK_END)
    if stream.tell() == 0:
        stream.write(b"\0")
        stream.flush()
        os.fsync(stream.fileno())
    stream.seek(0)
    acquired = False
    unlock = None
    try:
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError(f"Shared heavy-job lock is held: {path}") from exc
            unlock = lambda: msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError(f"Shared heavy-job lock is held: {path}") from exc
            unlock = lambda: fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        acquired = True
        yield {"path": str(path), "mechanism": "OS nonblocking advisory lock", "acquired": True}
    finally:
        if acquired and unlock is not None:
            try:
                unlock()
            except OSError:
                pass
        stream.close()


@contextmanager
def heavy_job_locks():
    with ExitStack() as stack:
        shared = stack.enter_context(_os_lock(Path(tempfile.gettempdir()) / "kasdad_single_heavy_job.lock"))
        local = stack.enter_context(_os_lock(RUN_ROOT / ".training.lock"))
        yield [shared, local]


def _resource_snapshot(minimum_start: int, minimum_reserve: int) -> dict[str, int]:
    available = int(psutil.virtual_memory().available)
    require(available >= minimum_start,
            f"Available RAM {available} is below required start threshold {minimum_start}")
    return {"available_bytes": available, "minimum_start_bytes": minimum_start,
            "minimum_reserve_bytes": minimum_reserve}


def _load_local_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"Cannot load local source module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_old_runner(repository: Path):
    repository = Path(repository).resolve()
    sys.path.insert(0, str(repository))
    importlib.invalidate_caches()
    module = importlib.import_module("scripts.optimization_v3_positional_bag_offset_v1")
    require(Path(module.ROOT).resolve() == repository,
            "Loaded POS_BAG_OFFSET runner from a different repository")
    return module


def _read_a02_cache(path: Path) -> dict[str, np.ndarray]:
    """Read only prediction/identity/group members; never index cached labels."""
    requested = ("prediction", "score_rows", "score_record_ids", "score_cycles",
                 "missing_mask", "entry_friction", "text_load_group")
    with np.load(path, allow_pickle=False) as archive:
        require(set(requested).issubset(set(archive.files)), "A02 OOF cache schema is incomplete")
        values = {name: archive[name].copy() for name in requested}
        labels_member_present = "labels" in archive.files
    require(labels_member_present, "Expected historical A02 cache label member is absent; lineage differs")
    # Presence is metadata only. The labels member is deliberately never read.
    return values


def _load_origin_context(repository: Path, origin: int, *, require_fit_labels: bool = False) -> dict[str, Any]:
    repository = Path(repository).resolve()
    old = _load_old_runner(repository)
    suffix = f"E{origin}"
    pilot_root = repository / PILOT_ROOT_REL
    pos_root = repository / POS_RUN_REL
    matrix_manifest_path = pilot_root / f"MATRIX_MANIFEST_{suffix}.json"
    manifest = old._read_json(matrix_manifest_path)
    prediction_path = pilot_root / suffix / f"predictions_{suffix}.npz"
    with np.load(prediction_path, allow_pickle=False) as archive:
        required = {"origin", "fit_rows", "score_rows", "score_record_ids", "score_cycles",
                    "present_mask", "prob_bag", "matrix_manifest_sha256"}
        require(required.issubset(set(archive.files)), f"Saved POS feature prediction schema incomplete for {suffix}")
        pilot_prediction = {name: archive[name].copy() for name in required}
    require(int(pilot_prediction["origin"].reshape(-1)[0]) == origin,
            f"Saved POS feature origin mismatch for {suffix}")
    require(str(pilot_prediction["matrix_manifest_sha256"].reshape(-1)[0]) == sha256_file(matrix_manifest_path),
            f"Saved POS feature prediction matrix binding mismatch for {suffix}")

    accounting = old._read_json(old.ACCOUNTING)
    bag_id = f"POSITIONAL_TEXT_PILOT_V1_{suffix}_BAG_LR"
    bag_account = next((x for x in accounting["fit_units"] if x["fit_id"] == bag_id), None)
    require(bag_account is not None, f"Saved C2_B fit accounting missing for {suffix}")
    base_shape = manifest["matrices"]["fit_base"]["shape"]
    bag_obj, bag_receipt = old._verify_estimator(bag_id, bag_account, width=int(base_shape[1]))
    bag = bag_obj["estimator"]
    matrices = {name: old._load_matrix(manifest, name) for name in MATRIX_NAMES}
    fit_base = sparse.csr_matrix(matrices["fit_base"], dtype=np.float64)
    fit_augmented = sparse.csr_matrix(matrices["fit_position"], dtype=np.float64)
    score_base = sparse.csr_matrix(matrices["score_base"], dtype=np.float64)
    score_augmented = sparse.csr_matrix(matrices["score_position"], dtype=np.float64)
    base_width = int(fit_base.shape[1])
    x_fit = fit_augmented[:, base_width:].tocsr()
    x_score = score_augmented[:, base_width:].tocsr()
    require(x_fit.shape[1] == x_score.shape[1] and x_fit.shape[1] > 0,
            f"Saved positional dimensions differ for {suffix}")

    fit_rows = np.asarray(pilot_prediction["fit_rows"], dtype=np.int64).reshape(-1)
    score_rows = np.asarray(pilot_prediction["score_rows"], dtype=np.int64).reshape(-1)
    identity = old._read_identity_frame()
    fit_ids = identity.iloc[fit_rows]["record_id"].astype(str).to_numpy(dtype=str)
    score_ids = np.asarray(pilot_prediction["score_record_ids"]).astype(str)
    cycles = identity.iloc[score_rows]["cycle_code"].to_numpy(dtype=np.int64)
    require(len(fit_rows) == (29705 if origin == 67 else 42117),
            f"Frozen prefix row count mismatch for {suffix}")
    expected_low, expected_high = ((47, 67) if origin == 67 else (47, 75))
    fit_cycles = identity.iloc[fit_rows]["cycle_code"].to_numpy(dtype=np.int64)
    require(np.all((fit_cycles >= expected_low) & (fit_cycles <= expected_high)),
            f"Fit rows escape the frozen prefix for {suffix}")
    require(np.all((cycles >= origin + 1) & (cycles <= 79)), f"Score rows escape frozen window for {suffix}")
    require(np.array_equal(cycles, pilot_prediction["score_cycles"].astype(np.int64)),
            f"Saved score cycles differ from official Train identity for {suffix}")
    present = np.asarray(pilot_prediction["present_mask"], dtype=bool).reshape(-1)
    require(len(score_ids) == len(score_rows) == len(cycles) == len(present),
            f"Saved score row arrays differ for {suffix}")

    a02_dir = repository / A02_RUN_REL / suffix / "observed"
    a02_fit = old._read_json(a02_dir / "fit_receipt.json")
    a02_prediction_receipt = old._read_json(a02_dir / "predictions.json")
    require(a02_fit["query_labels_passed_to_fit"] is False and a02_fit["test_data_used"] is False
            and a02_prediction_receipt["query_labels_passed_to_prediction"] is False,
            f"A02 label boundary receipt failed for {suffix}")
    a02_cache = _read_a02_cache(a02_dir / "predictions.npz")
    p_a = np.asarray(a02_cache["prediction"], dtype=np.float64).reshape(-1)
    a02_rows = np.asarray(a02_cache["score_rows"], dtype=np.int64).reshape(-1)
    a02_ids = np.asarray(a02_cache["score_record_ids"]).astype(str)
    a02_cycles = np.asarray(a02_cache["score_cycles"], dtype=np.int64).reshape(-1)
    a02_missing = np.asarray(a02_cache["missing_mask"], dtype=bool).reshape(-1)
    require(np.array_equal(a02_rows, score_rows) and np.array_equal(a02_ids, score_ids)
            and np.array_equal(a02_cycles, cycles), f"A02 score row/ID/cycle alignment failed for {suffix}")
    require(np.array_equal(a02_missing, ~present), f"A02 and POS stream masks differ for {suffix}")
    require(np.isfinite(p_a).all() and np.all((p_a >= 0.0) & (p_a <= 1.0)),
            f"A02 probabilities invalid for {suffix}")
    entry_friction = np.asarray(a02_cache["entry_friction"]).reshape(-1)
    text_load_group = np.asarray(a02_cache["text_load_group"]).reshape(-1)
    require(len(entry_friction) == len(text_load_group) == len(score_rows),
            f"Saved subgroup arrays do not align for {suffix}")

    z0_fit = np.asarray(bag.intercept_[0] + fit_base @ bag.coef_[0], dtype=np.float64).reshape(-1)
    z0_score = np.asarray(bag.intercept_[0] + score_base @ bag.coef_[0], dtype=np.float64).reshape(-1)
    require(len(z0_fit) == len(fit_rows) and len(z0_score) == len(score_rows),
            f"Saved C2_B offset rows do not align for {suffix}")
    require(np.allclose(expit(z0_score), pilot_prediction["prob_bag"], rtol=0.0, atol=1e-12),
            f"Saved C2_B logit offset does not replay its original bag probability for {suffix}")

    old_gamma_id = f"POS_BAG_OFFSET_V1_{suffix}_REAL"
    gamma_path = pos_root / "units" / f"{old_gamma_id}.npz"
    gamma_receipt_path = pos_root / "receipts" / f"{old_gamma_id}.json"
    gamma_receipt = old._read_json(gamma_receipt_path)
    require(gamma_receipt["fit_id"] == old_gamma_id and gamma_receipt["origin"] == origin,
            f"Original gamma receipt identity mismatch for {suffix}")
    require(sha256_file(gamma_path) == gamma_receipt["artifact_sha256"],
            f"Original gamma artifact file hash mismatch for {suffix}")
    with np.load(gamma_path, allow_pickle=False) as archive:
        require(archive.files == ["gamma"], f"Original gamma artifact schema mismatch for {suffix}")
        gamma_original = np.asarray(archive["gamma"], dtype=np.float64).copy()
    require(gamma_original.shape == (x_fit.shape[1],) and np.isfinite(gamma_original).all()
            and sha256_array(gamma_original) == gamma_receipt["gamma_sha256"],
            f"Original gamma array mismatch for {suffix}")
    pos_predictions_path = pos_root / suffix / f"predictions_{suffix}.npz"
    with np.load(pos_predictions_path, allow_pickle=False) as archive:
        required = {"score_record_ids", "score_cycles", "present_mask", "prob_offset_real"}
        require(required.issubset(set(archive.files)), f"Original POS score bundle incomplete for {suffix}")
        pos_ids = np.asarray(archive["score_record_ids"]).astype(str)
        pos_cycles = np.asarray(archive["score_cycles"], dtype=np.int64)
        pos_present = np.asarray(archive["present_mask"], dtype=bool)
        p_pos_saved = np.asarray(archive["prob_offset_real"], dtype=np.float64)
    from_local = _load_local_module("_cls_pos_poisson_helpers", AUDIT / "src/models/cls_pos_poisson_bag2_v1.py")
    p_pos_original = from_local.offset_probability(x_score, z0_score, gamma_original)
    require(np.array_equal(pos_ids, score_ids) and np.array_equal(pos_cycles, cycles)
            and np.array_equal(pos_present, present), f"Original POS score bundle alignment mismatch for {suffix}")
    require(np.allclose(p_pos_original[present], p_pos_saved[present], rtol=0.0, atol=1e-12),
            f"Reconstructed original POS gamma differs on present rows for {suffix}")

    target_sha = bag_receipt["fit_population"]["fit_target_sha256"]
    y_fit = None
    if require_fit_labels:
        y_fit = old._fit_labels(repository / "data/raw/classification/Train.csv",
                                fit_rows, fit_ids, target_sha)
    return {
        "old_runner": old, "origin": origin, "manifest": manifest,
        "manifest_sha256": sha256_file(matrix_manifest_path),
        "pilot_prediction": pilot_prediction, "fit_rows": fit_rows,
        "score_rows": score_rows, "fit_ids": fit_ids, "score_ids": score_ids,
        "cycles": cycles, "present": present, "entry_friction": entry_friction,
        "text_load_group": text_load_group, "p_a02": p_a,
        "matrices": matrices, "x_fit": x_fit, "x_score": x_score,
        "z0_fit": z0_fit, "z0_score": z0_score, "gamma_original": gamma_original,
        "p_pos_original": p_pos_original, "bag_receipt": bag_receipt,
        "gamma_receipt": gamma_receipt, "gamma_path": gamma_path,
        "gamma_receipt_path": gamma_receipt_path,
        "pos_predictions_path": pos_predictions_path,
        "a02_prediction_receipt": a02_prediction_receipt,
        "a02_prediction_cache_path": a02_dir / "predictions.npz",
        "fit_target_sha256_expected": target_sha, "y_fit": y_fit,
    }


def _append_ledger(event: Mapping[str, Any]) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(event, sort_keys=True, ensure_ascii=False,
                         allow_nan=False, separators=(",", ":")) + "\n"
    with LEDGER_PATH.open("a", encoding="utf-8", newline="") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _ledger_events() -> list[dict[str, Any]]:
    if not LEDGER_PATH.exists():
        return []
    return [json.loads(line) for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines() if line]


def _validate_fit_ledger(events: list[dict[str, Any]] | None = None, *,
                         freeze_signature: str | None = None,
                         source_commit: str | None = None,
                         root_pass_sha256: str | None = None,
                         require_resolved: bool = False) -> list[dict[str, Any]]:
    events = _ledger_events() if events is None else events
    allowed = {"FIT_CAP_RESERVED", "FIT_STARTED", "FIT_SAVED", "FIT_FAILED", "QUERY_LABELS_OPENED"}
    require(all(event.get("event") in allowed for event in events), "Unknown candidate ledger event")
    reservations = [event for event in events if event["event"] == "FIT_CAP_RESERVED"]
    starts = [event for event in events if event["event"] == "FIT_STARTED"]
    terminals = [event for event in events if event["event"] in {"FIT_SAVED", "FIT_FAILED"}]
    query_events = [event for event in events if event["event"] == "QUERY_LABELS_OPENED"]
    require(len(reservations) <= 1 and len(query_events) <= 1, "Duplicate reservation/query-label ledger event")
    start_ids = [event.get("fit_id") for event in starts]
    terminal_ids = [event.get("fit_id") for event in terminals]
    require(set(start_ids).issubset(RESERVATION_FIT_IDS) and len(start_ids) == len(set(start_ids))
            and len(start_ids) <= 4, "Fit ledger exceeds or duplicates the fixed four-fit cap")
    require(set(terminal_ids).issubset(set(start_ids)) and len(terminal_ids) == len(set(terminal_ids)),
            "Terminal fit event has no unique start event")
    if require_resolved:
        require(set(terminal_ids) == set(start_ids),
                "An earlier fit start lacks a terminal receipt; do not start another fit")
    if query_events:
        require(len(start_ids) == 4 and len([event for event in terminals if event["event"] == "FIT_SAVED"]) == 4
                and not any(event["event"] == "FIT_FAILED" for event in terminals),
                "Query labels cannot be opened before all four successful fits")
        query_position = events.index(query_events[0])
        require(not any(event["event"] in {"FIT_STARTED", "FIT_SAVED", "FIT_FAILED"}
                        for event in events[query_position + 1:]),
                "No fit events are allowed after validation labels are opened")
    if reservations:
        event = reservations[0]
        require(event.get("candidate_id") == "CLS_POS_POISSON_BAG2_V1"
                and event.get("fit_ids") == RESERVATION_FIT_IDS and event.get("fit_cap") == 4,
                "Fit reservation ledger does not match the fixed cap")
        if freeze_signature is not None:
            require(event.get("freeze_signature") == freeze_signature,
                    "Fit reservation ledger uses a different freeze")
        if source_commit is not None:
            require(event.get("source_commit") == source_commit,
                    "Fit reservation ledger uses a different source commit")
        if root_pass_sha256 is not None:
            require(event.get("root_prefit_pass_sha256") == root_pass_sha256,
                    "Fit reservation ledger uses a different root approval")
    for event in starts:
        fit_id = event.get("fit_id")
        require(fit_id in FIT_ID_TO_ORIGIN_SEED
                and (event.get("origin"), event.get("replica_seed")) == FIT_ID_TO_ORIGIN_SEED[fit_id],
                "Fit ledger ID/origin/seed mismatch")
        require(event.get("fit_cap") == 4 and event.get("query_labels_read") is False
                and event.get("test_or_sample_access") is False,
                "Fit-start boundary/cap receipt invalid")
        if freeze_signature is not None:
            require(event.get("freeze_signature") == freeze_signature,
                    "Fit start uses a different freeze")
        if source_commit is not None:
            require(event.get("source_commit") == source_commit,
                    "Fit start uses a different source commit")
        if root_pass_sha256 is not None:
            require(event.get("root_prefit_pass_sha256") == root_pass_sha256,
                    "Fit start uses a different root approval")
    for event in terminals:
        if freeze_signature is not None:
            require(event.get("freeze_signature") == freeze_signature,
                    "Fit terminal uses a different freeze")
        if root_pass_sha256 is not None:
            require(event.get("root_prefit_pass_sha256") == root_pass_sha256,
                    "Fit terminal uses a different root approval")
        if event["event"] == "FIT_SAVED":
            start = next(item for item in starts if item.get("fit_id") == event.get("fit_id"))
            _verify_saved_fit_event(start, event)
    if query_events:
        if freeze_signature is not None:
            require(query_events[0].get("freeze_signature") == freeze_signature,
                    "Query-label event uses a different freeze")
        if root_pass_sha256 is not None:
            require(query_events[0].get("root_prefit_pass_sha256") == root_pass_sha256,
                    "Query-label event uses a different root approval")
    return events


def _verify_saved_fit_event(start: Mapping[str, Any], terminal: Mapping[str, Any]) -> dict[str, Any]:
    fit_id = str(start.get("fit_id"))
    artifact_path, receipt_path = _fit_paths(fit_id)
    require(terminal.get("fit_id") == fit_id and terminal.get("origin") == start.get("origin")
            and terminal.get("replica_seed") == start.get("replica_seed"),
            f"Saved-fit ledger identity mismatch: {fit_id}")
    require(artifact_path.is_file() and receipt_path.is_file()
            and sha256_file(artifact_path) == terminal.get("artifact_sha256")
            and sha256_file(receipt_path) == terminal.get("receipt_sha256"),
            f"Saved-fit artifact/receipt hash differs from append-only ledger: {fit_id}")
    verifier = _load_local_module("_cls_pos_poisson_verifier", AUDIT / "scripts/verify_cls_pos_poisson_bag2_v1.py")
    verified = verifier.verify_fit_artifact(RUN_ROOT, fit_id, str(start["freeze_signature"]))
    receipt = verified["receipt"]
    expected_lineage = {
        "origin": start["origin"], "replica_seed": start["replica_seed"],
        "fit_rows": start["fit_rows"],
        "fit_row_positions_sha256": start["fit_rows_sha256"],
        "fit_record_ids_sha256": start["fit_ids_sha256"],
        "fit_target_sha256": start["expected_target_sha256"],
        "fit_position_sha256": start["fit_position_sha256"],
        "offset_fit_sha256": start["offset_fit_sha256"],
        "gamma_original_sha256": start["gamma_original_sha256"],
    }
    if "matrix_manifest_sha256" in start:
        expected_lineage["matrix_manifest_sha256"] = start["matrix_manifest_sha256"]
    require(all(receipt.get(key) == value for key, value in expected_lineage.items())
            and receipt.get("input_freeze_signature") == start.get("freeze_signature")
            and receipt.get("artifact_sha256") == terminal.get("artifact_sha256")
            and verified["receipt_sha256"] == terminal.get("receipt_sha256"),
            f"Saved-fit receipt lineage differs from FIT_STARTED/FIT_SAVED events: {fit_id}")
    return verified


def _verify_reservation(freeze: Mapping[str, Any], root_pass_sha256: str) -> dict[str, Any]:
    reservation = read_json(RESERVATION_PATH)
    require(reservation.get("candidate_id") == "CLS_POS_POISSON_BAG2_V1"
            and reservation.get("status") == "EXACT_FOUR_FIT_CAP_RESERVED"
            and reservation.get("freeze_signature") == freeze["freeze_signature"]
            and reservation.get("freeze_sha256") == sha256_file(FREEZE_PATH)
            and reservation.get("audit_source_commit") == freeze["audit_source_commit"]
            and reservation.get("fit_ids") == RESERVATION_FIT_IDS
            and reservation.get("fit_cap") == 4
            and reservation.get("root_prefit_pass_sha256") == root_pass_sha256
            and reservation.get("no_test_or_sample_access") is True
            and reservation.get("no_kaggle_upload") is True,
            "Fit reservation differs from the committed freeze/root approval")
    return reservation


def reserve_fit_cap(repository: Path) -> dict[str, Any]:
    with heavy_job_locks():
        freeze = verify_prefit_freeze(repository)
        verify_root_prefit_pass(freeze)
        root_pass_sha256 = sha256_file(ROOT_PREFIT_PASS)
        existing_events = _validate_fit_ledger(freeze_signature=freeze["freeze_signature"],
                                                source_commit=freeze["audit_source_commit"],
                                                root_pass_sha256=root_pass_sha256)
        expected_event = {"event": "FIT_CAP_RESERVED", "candidate_id": "CLS_POS_POISSON_BAG2_V1",
                          "fit_ids": RESERVATION_FIT_IDS, "fit_cap": 4,
                          "freeze_signature": freeze["freeze_signature"],
                          "source_commit": freeze["audit_source_commit"],
                          "root_prefit_pass_sha256": root_pass_sha256}
        if RESERVATION_PATH.exists():
            reservation = _verify_reservation(freeze, root_pass_sha256)
            require(sum(e.get("event") == "FIT_STARTED" for e in existing_events) <= 4,
                    "Fit ledger exceeds the exact four-fit cap")
            existing_reservation_events = [e for e in existing_events if e.get("event") == "FIT_CAP_RESERVED"]
            if not existing_reservation_events:
                _append_ledger(expected_event)
            else:
                require(existing_reservation_events[0] == expected_event,
                        "Existing reservation ledger differs; refusing rewrite")
            return reservation
        require(not existing_events and not any((RUN_ROOT / "units").glob("*.npz"))
                and not any((RUN_ROOT / "receipts").glob("*.json")),
                "Candidate output namespace is nonempty without its reservation")
        reservation = {
            "schema_version": 1, "candidate_id": "CLS_POS_POISSON_BAG2_V1",
            "status": "EXACT_FOUR_FIT_CAP_RESERVED", "fit_cap": 4,
            "fit_ids": RESERVATION_FIT_IDS, "freeze_signature": freeze["freeze_signature"],
            "freeze_sha256": sha256_file(FREEZE_PATH),
            "root_prefit_pass_sha256": sha256_file(ROOT_PREFIT_PASS),
            "audit_source_commit": freeze["audit_source_commit"],
            "no_test_or_sample_access": True, "no_kaggle_upload": True,
        }
        write_once(RESERVATION_PATH, reservation)
        _append_ledger(expected_event)
        return reservation


def _fit_status(fit_id: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    events = [event for event in _ledger_events() if event.get("fit_id") == fit_id]
    starts = [event for event in events if event.get("event") == "FIT_STARTED"]
    terminals = [event for event in events if event.get("event") in {"FIT_SAVED", "FIT_FAILED"}]
    require(len(starts) <= 1 and len(terminals) <= 1, f"Duplicate fit accounting events: {fit_id}")
    if not starts:
        require(not terminals, f"Terminal fit event without start: {fit_id}")
        return None, None
    if not terminals:
        raise RuntimeError(f"Fit {fit_id} was started but has no terminal receipt; no retry")
    return starts[0], terminals[0]


def _fit_paths(fit_id: str) -> tuple[Path, Path]:
    require(fit_id in FIT_ID_TO_ORIGIN_SEED, f"Fit ID is outside the fixed four-fit cap: {fit_id}")
    return RUN_ROOT / "units" / f"{fit_id}.npz", RUN_ROOT / "receipts" / f"{fit_id}.json"


def fit_one(repository: Path, fit_id: str) -> dict[str, Any]:
    freeze = verify_prefit_freeze(repository)
    verify_root_prefit_pass(freeze)
    reservation = _verify_reservation(freeze, sha256_file(ROOT_PREFIT_PASS))
    require(fit_id in RESERVATION_FIT_IDS, f"Fit ID not reserved: {fit_id}")
    origin, seed = FIT_ID_TO_ORIGIN_SEED[fit_id]
    with heavy_job_locks() as locks:
        freeze = verify_prefit_freeze(repository)
        verify_root_prefit_pass(freeze)
        root_pass_sha256 = sha256_file(ROOT_PREFIT_PASS)
        reservation = _verify_reservation(freeze, root_pass_sha256)
        ledger = _validate_fit_ledger(freeze_signature=freeze["freeze_signature"],
                                      source_commit=freeze["audit_source_commit"],
                                      root_pass_sha256=root_pass_sha256,
                                      require_resolved=True)
        require(len([e for e in ledger if e["event"] == "FIT_CAP_RESERVED"]) == 1,
                "Ledger reservation event is missing")
        require(not any(event["event"] == "FIT_FAILED" for event in ledger),
                "An earlier fit failed; no further fit is permitted")
        start_event, terminal_event = _fit_status(fit_id)
        if terminal_event is not None:
            require(terminal_event["event"] == "FIT_SAVED", f"Fit {fit_id} already failed; no retry")
            verifier = _load_local_module("_cls_pos_poisson_verifier", AUDIT / "scripts/verify_cls_pos_poisson_bag2_v1.py")
            verified = verifier.verify_fit_artifact(RUN_ROOT, fit_id, freeze["freeze_signature"])
            return {"fit_id": fit_id, "status": "ALREADY_SAVED_NO_REFIT",
                    "gamma_sha256": verified["receipt"]["gamma_sha256"]}
        require(start_event is None, f"Fit {fit_id} is already consumed; no retry")
        require(len([event for event in ledger if event["event"] == "FIT_STARTED"]) < 4,
                "All four fit slots are already consumed")
        resource_before = _resource_snapshot(MIN_FIT_AVAILABLE, MIN_FIT_RESERVE)
        current_hashes = _external_file_inventory(Path(repository).resolve())
        require(current_hashes == {**freeze["external_source_hashes"], **freeze["input_file_hashes"]},
                "Frozen source/input hashes changed before fit")
        context = _load_origin_context(repository, origin, require_fit_labels=False)
        require(_external_file_inventory(Path(repository).resolve()) == current_hashes,
                "Frozen source/input changed while preparing fit features")
        fit_id_hash = sha256_array(context["fit_ids"])
        fit_row_hash = sha256_array(context["fit_rows"])
        fit_position_hash = sha256_csr(context["x_fit"])
        offset_hash = sha256_array(context["z0_fit"])
        original_gamma_hash = sha256_array(context["gamma_original"])
        bag_target_sha = context["fit_target_sha256_expected"]
        start = {
            "event": "FIT_STARTED", "fit_id": fit_id, "origin": origin,
            "replica_seed": seed, "fit_kind": "POISSON_WEIGHTED_POSITIONAL_GAMMA",
            "fit_consumed_at_start": True, "fit_cap": 4,
            "freeze_signature": freeze["freeze_signature"],
            "source_commit": freeze["audit_source_commit"],
            "input_freeze_sha256": sha256_file(FREEZE_PATH),
            "root_prefit_pass_sha256": sha256_file(ROOT_PREFIT_PASS),
            "fit_rows": len(context["fit_rows"]), "fit_rows_sha256": fit_row_hash,
            "fit_ids_sha256": fit_id_hash, "expected_target_sha256": bag_target_sha,
            "fit_position_sha256": fit_position_hash, "offset_fit_sha256": offset_hash,
            "gamma_original_sha256": original_gamma_hash,
            "matrix_manifest_sha256": context["manifest_sha256"],
            "resource_before": resource_before, "locks": locks,
            "query_labels_read": False, "test_or_sample_access": False,
        }
        _append_ledger(start)
        artifact_path, receipt_path = _fit_paths(fit_id)
        try:
            helper = _load_local_module("_cls_pos_poisson_helpers", AUDIT / "src/models/cls_pos_poisson_bag2_v1.py")
            y_fit = context["old_runner"]._fit_labels(
                Path(repository) / "data/raw/classification/Train.csv",
                context["fit_rows"], context["fit_ids"], bag_target_sha)
            raw_counts, row_weights = helper.poisson_row_weights(len(y_fit), seed)
            with threadpool_limits(limits=2):
                gamma, optimizer = helper.fit_poisson_weighted_offset(
                    context["x_fit"], context["z0_fit"], y_fit, row_weights)
            resource_after = int(psutil.virtual_memory().available)
            require(resource_after >= MIN_FIT_RESERVE,
                    f"Post-fit available RAM {resource_after} fell below reserve {MIN_FIT_RESERVE}")
            require(verify_prefit_freeze(repository)["freeze_signature"] == freeze["freeze_signature"],
                    "Candidate source/input freeze changed during the gamma fit")
            require(_external_file_inventory(Path(repository).resolve()) == current_hashes,
                    "Frozen source/input changed during the gamma fit")
            require(not artifact_path.exists() and not receipt_path.exists(),
                    f"Fit output already exists without matching ledger; no overwrite: {fit_id}")
            atomic_npz(artifact_path, gamma=gamma)
            receipt = {
                "schema_version": 1, "fit_id": fit_id, "origin": origin,
                "replica_seed": seed, "fit_rows": int(len(y_fit)),
                "position_features": int(len(gamma)), "attempt_number": 1,
                "retry_allowed": False, "synthetic_only": False,
                "fit_row_positions_sha256": fit_row_hash,
                "fit_record_ids_sha256": fit_id_hash,
                "fit_target_sha256": sha256_array(y_fit),
                "input_freeze_signature": freeze["freeze_signature"],
                "matrix_manifest_sha256": context["manifest_sha256"],
                "fit_position_sha256": fit_position_hash,
                "offset_fit_sha256": offset_hash,
                "gamma_original_sha256": original_gamma_hash,
                "poisson_raw_counts_sha256": sha256_array(raw_counts),
                "poisson_weights_sha256": sha256_array(row_weights),
                "poisson_raw_sum": int(raw_counts.sum()),
                "poisson_zero_count": int(np.count_nonzero(raw_counts == 0)),
                "gamma_sha256": sha256_array(gamma),
                "artifact_path": artifact_path.relative_to(AUDIT).as_posix(),
                "artifact_sha256": sha256_file(artifact_path),
                "optimizer": {**optimizer, "replica_seed": seed,
                              "poisson_lambda": 1.0, "weight_sum": float(row_weights.sum()),
                              "resource_after_available_bytes": resource_after,
                              "max_threads": 2},
                "no_test_or_sample_access": True, "no_query_labels_in_fit": True,
            }
            write_once(receipt_path, receipt)
            verifier = _load_local_module("_cls_pos_poisson_verifier", AUDIT / "scripts/verify_cls_pos_poisson_bag2_v1.py")
            verified = verifier.verify_fit_artifact(RUN_ROOT, fit_id, freeze["freeze_signature"])
            require(verified["receipt"]["fit_target_sha256"] == sha256_array(y_fit),
                    f"Fit receipt target digest mismatch: {fit_id}")
            require(_external_file_inventory(Path(repository).resolve()) == current_hashes,
                    "Frozen source/input changed during fit")
            _append_ledger({"event": "FIT_SAVED", "fit_id": fit_id,
                            "origin": origin, "replica_seed": seed,
                            "freeze_signature": freeze["freeze_signature"],
                            "root_prefit_pass_sha256": sha256_file(ROOT_PREFIT_PASS),
                            "receipt_sha256": sha256_file(receipt_path),
                            "artifact_sha256": sha256_file(artifact_path),
                            "freeze_signature": freeze["freeze_signature"]})
            return {"fit_id": fit_id, "status": "SAVED",
                    "gamma_sha256": verified["receipt"]["gamma_sha256"],
                    "receipt_sha256": sha256_file(receipt_path)}
        except Exception as exc:
            if artifact_path.is_file() and not receipt_path.exists():
                # Keep a failed attempt consumed; leave any partial artifact for audit.
                pass
            _append_ledger({"event": "FIT_FAILED", "fit_id": fit_id,
                            "origin": origin, "replica_seed": seed,
                            "freeze_signature": freeze["freeze_signature"],
                            "root_prefit_pass_sha256": sha256_file(ROOT_PREFIT_PASS),
                            "error_type": type(exc).__name__, "error": str(exc),
                            "no_retry": True})
            raise


def build_predictions(repository: Path) -> dict[str, Any]:
    freeze = verify_prefit_freeze(repository)
    verify_root_prefit_pass(freeze)
    root_pass_sha256 = sha256_file(ROOT_PREFIT_PASS)
    _verify_reservation(freeze, root_pass_sha256)
    ledger = _validate_fit_ledger(freeze_signature=freeze["freeze_signature"],
                                  source_commit=freeze["audit_source_commit"],
                                  root_pass_sha256=root_pass_sha256,
                                  require_resolved=True)
    require(not any(event["event"] == "QUERY_LABELS_OPENED" for event in ledger),
            "Prediction construction cannot be reopened after query-label scoring")
    events = ledger
    starts = [e for e in events if e.get("event") == "FIT_STARTED"]
    saved = [e for e in events if e.get("event") == "FIT_SAVED"]
    failed = [e for e in events if e.get("event") == "FIT_FAILED"]
    require(len(starts) == 4 and len(saved) == 4 and len(failed) == 0,
            "All four distinct fit receipts must be durable before prediction construction")
    require({e["fit_id"] for e in starts} == set(RESERVATION_FIT_IDS)
            and {e["fit_id"] for e in saved} == set(RESERVATION_FIT_IDS),
            "Prediction build fit IDs differ from exact reservation")
    verifier = _load_local_module("_cls_pos_poisson_verifier", AUDIT / "scripts/verify_cls_pos_poisson_bag2_v1.py")
    helper = _load_local_module("_cls_pos_poisson_helpers", AUDIT / "src/models/cls_pos_poisson_bag2_v1.py")
    fit_artifacts = {fit_id: verifier.verify_fit_artifact(RUN_ROOT, fit_id, freeze["freeze_signature"])
                     for fit_id in RESERVATION_FIT_IDS}
    outputs: dict[str, Any] = {}
    with heavy_job_locks():
        _resource_snapshot(MIN_CACHE_AVAILABLE, MIN_CACHE_RESERVE)
        current_hashes = _external_file_inventory(Path(repository).resolve())
        require(current_hashes == {**freeze["external_source_hashes"], **freeze["input_file_hashes"]},
                "Frozen external source/input hashes changed before prediction construction")
        for origin in ORIGINS:
            suffix = f"E{origin}"
            context = _load_origin_context(repository, origin, require_fit_labels=False)
            gamma17 = fit_artifacts[f"CLS_POS_POISSON_BAG2_V1_{suffix}_P17"]["gamma"]
            gamma137 = fit_artifacts[f"CLS_POS_POISSON_BAG2_V1_{suffix}_P137"]["gamma"]
            probability = helper.build_policy_probabilities(
                context["x_score"], context["z0_score"], context["gamma_original"],
                gamma17, gamma137, context["p_a02"], context["present"])
            p_original = probability["probability_pos_original"]
            pbag = probability["probability_pos_bag2"]
            pchamp = probability["probability_champion"]
            pcandidate = probability["probability_candidate"]
            label_champion = helper.apply_fixed_thresholds(pchamp, context["present"])
            label_candidate = helper.apply_fixed_thresholds(pcandidate, context["present"])
            bundle = {
                "origin": np.asarray([origin], dtype=np.int64),
                "row_positions": context["score_rows"].astype(np.int64),
                "record_ids": context["score_ids"].astype(str),
                "cycles": context["cycles"].astype(np.int64),
                "present_mask": context["present"].astype(bool),
                "text_load_group": context["text_load_group"],
                "entry_friction": context["entry_friction"],
                "p_a02": context["p_a02"].astype(np.float64),
                "p_pos_original": p_original,
                "p_pos_seed17": helper.offset_probability(context["x_score"], context["z0_score"], gamma17),
                "p_pos_seed137": helper.offset_probability(context["x_score"], context["z0_score"], gamma137),
                "p_pos_bag2": pbag,
                "p_champion": pchamp,
                "p_candidate": pcandidate,
                "label_champion": label_champion,
                "label_candidate": label_candidate,
                "matrix_manifest_sha256": np.asarray([context["manifest_sha256"]]),
                "input_freeze_signature": np.asarray([freeze["freeze_signature"]]),
            }
            path = RUN_ROOT / suffix / f"predictions_{suffix}.npz"
            if path.exists():
                existing = verifier.verify_prediction_bundle(path, origin=origin,
                                                               freeze_signature=freeze["freeze_signature"])
                require(existing["rows"] == len(context["score_rows"]),
                        f"Existing prediction bundle row count changed for {suffix}")
                with np.load(path, allow_pickle=False) as archive:
                    for name, expected in bundle.items():
                        require(np.array_equal(archive[name], expected),
                                f"Existing prediction bundle differs from frozen replay: {suffix}/{name}")
            else:
                atomic_npz(path, **bundle)
            outputs[suffix] = {"path": path.relative_to(AUDIT).as_posix(),
                               "sha256": sha256_file(path), "rows": len(context["score_rows"]),
                               "missing_rows": int((~context["present"]).sum()),
                               "ordered_ids_sha256": sha256_array(context["score_ids"]),
                               "matrix_manifest_sha256": context["manifest_sha256"]}
        require(verify_prefit_freeze(repository)["freeze_signature"] == freeze["freeze_signature"]
                and _external_file_inventory(Path(repository).resolve()) == current_hashes,
                "Frozen source/input changed during prediction construction")
        ready = {"schema_version": 1, "candidate_id": "CLS_POS_POISSON_BAG2_V1",
                 "freeze_signature": freeze["freeze_signature"],
                 "all_four_fit_receipts_durable": True,
                 "query_labels_opened": False, "test_or_sample_access": False,
                 "predictions": outputs}
        if PREDICTIONS_READY.exists():
            require(read_json(PREDICTIONS_READY) == ready,
                    "Existing predictions-ready manifest differs; refusing overwrite")
        else:
            write_once(PREDICTIONS_READY, ready)
        expected_matrix_hashes = {
            f"E{origin}": freeze["input_file_hashes"][
                f"{PILOT_ROOT_REL}/MATRIX_MANIFEST_E{origin}.json"]
            for origin in ORIGINS}
        first = verifier.verify_readback_pair(RUN_ROOT, freeze["freeze_signature"], expected_matrix_hashes)
        second = verifier.verify_readback_pair(RUN_ROOT, freeze["freeze_signature"], expected_matrix_hashes)
        require(canonical_sha256(first) == canonical_sha256(second),
                "Prediction independent readback changed across repeated reads")
        readback = {**first, "repeat_readback_sha256": canonical_sha256(second),
                    "predictions_ready_sha256": sha256_file(PREDICTIONS_READY)}
        readback_path = RUN_ROOT / "READBACK_VERIFICATION.json"
        if readback_path.exists():
            require(read_json(readback_path) == readback, "Existing readback receipt differs; refusing overwrite")
        else:
            write_once(readback_path, readback)
        return {"status": "PASS_PREDICTIONS_READY_NO_QUERY_LABELS", "predictions": outputs,
                "readback_sha256": sha256_file(readback_path)}


def _read_selected_labels(train_path: Path, row_positions: np.ndarray,
                          expected_ids: np.ndarray) -> np.ndarray:
    rows = np.asarray(row_positions, dtype=np.int64).reshape(-1)
    ids = np.asarray(expected_ids).astype(str).reshape(-1)
    require(len(rows) == len(ids) and len(np.unique(rows)) == len(rows),
            "Score row positions/IDs are invalid")
    wanted = {int(row): pos for pos, row in enumerate(rows.tolist())}
    target = np.empty(len(rows), dtype=np.int8)
    seen = np.zeros(len(rows), dtype=bool)
    with Path(train_path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        require(header is not None and "record_id" in header and "artifact_signal" in header,
                "Official Train header is invalid for delayed score read")
        id_col = header.index("record_id")
        target_col = header.index("artifact_signal")
        for row_index, record in enumerate(reader):
            output_index = wanted.get(row_index)
            if output_index is None:
                continue
            require(max(id_col, target_col) < len(record), f"Selected Train row {row_index} malformed")
            require(record[id_col] == ids[output_index], f"Selected score ID changed at row {row_index}")
            value = int(record[target_col])
            require(value in (0, 1), f"Selected score target invalid at row {row_index}")
            target[output_index] = value
            seen[output_index] = True
    require(seen.all(), "One or more exact score labels were not found in official Train")
    return target


def _score_metric(y: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import f1_score, recall_score
    target = np.asarray(y, dtype=np.int8).reshape(-1)
    labels = np.asarray(prediction, dtype=np.int8).reshape(-1)
    require(len(target) == len(labels) and np.isin(target, (0, 1)).all()
            and np.isin(labels, (0, 1)).all(), "Scoring labels must be binary and aligned")
    return {"n": int(len(target)), "positive": int(target.sum()),
            "negative": int(len(target) - target.sum()),
            "macro_f1": float(f1_score(target, labels, average="macro", labels=[0, 1], zero_division=0)),
            "positive_recall": float(recall_score(target, labels, pos_label=1, zero_division=0))}


def _group_keys(values: np.ndarray) -> list[tuple[str, np.ndarray]]:
    raw = np.asarray(values).reshape(-1)
    labels = raw.astype(str)
    keys = sorted(set(labels.tolist()))
    return [(key, labels == key) for key in keys]


def _metric_record(y: np.ndarray, p_champ: np.ndarray, p_candidate: np.ndarray,
                   labels_champ: np.ndarray, labels_candidate: np.ndarray) -> dict[str, Any]:
    before = _score_metric(y, labels_champ)
    after = _score_metric(y, labels_candidate)
    return {"champion": before, "candidate": after,
            "gain_macro_f1": after["macro_f1"] - before["macro_f1"],
            "gain_positive_recall": after["positive_recall"] - before["positive_recall"]}


def _score_predictions_locked(repository: Path) -> dict[str, Any]:
    freeze = verify_prefit_freeze(repository)
    approval = verify_root_prefit_pass(freeze)
    require(approval.get("validation_label_scoring") is True,
            "Root prefit approval does not authorize matched Train validation scoring")
    root_pass_sha256 = sha256_file(ROOT_PREFIT_PASS)
    _verify_reservation(freeze, root_pass_sha256)
    ledger = _validate_fit_ledger(freeze_signature=freeze["freeze_signature"],
                                  source_commit=freeze["audit_source_commit"],
                                  root_pass_sha256=root_pass_sha256,
                                  require_resolved=True)
    require(len([event for event in ledger if event["event"] == "FIT_SAVED"]) == 4
            and not any(event["event"] == "FIT_FAILED" for event in ledger),
            "Validation scoring requires all four fixed fits to be saved")
    verifier = _load_local_module("_cls_pos_poisson_verifier", AUDIT / "scripts/verify_cls_pos_poisson_bag2_v1.py")
    ready = read_json(PREDICTIONS_READY)
    require(ready.get("freeze_signature") == freeze["freeze_signature"]
            and ready.get("query_labels_opened") is False
            and ready.get("all_four_fit_receipts_durable") is True,
            "Scoring requires all four verified query-label-free prediction bundles")
    readback_path = RUN_ROOT / "READBACK_VERIFICATION.json"
    readback = read_json(readback_path)
    expected_matrix_hashes = {
        f"E{origin}": freeze["input_file_hashes"][
            f"{PILOT_ROOT_REL}/MATRIX_MANIFEST_E{origin}.json"]
        for origin in ORIGINS}
    first = verifier.verify_readback_pair(RUN_ROOT, freeze["freeze_signature"], expected_matrix_hashes)
    require(readback.get("status") == "PASS_INDEPENDENT_PREDICTION_READBACK"
            and readback.get("predictions_ready_sha256") == sha256_file(PREDICTIONS_READY)
            and readback.get("predictions") == first["predictions"],
            "Scoring requires current independent prediction readback PASS")
    if RESULTS_PATH.exists():
        existing = read_json(RESULTS_PATH)
        require(existing["freeze_signature"] == freeze["freeze_signature"],
                "Existing result belongs to a different freeze")
        verification = verifier.verify_score_result(
            RESULTS_PATH, RUN_ROOT, Path(repository) / "data/raw/classification/Train.csv",
            freeze_signature=freeze["freeze_signature"],
            source_commit=freeze["audit_source_commit"],
            root_pass_sha256=root_pass_sha256, train_sha256=EXPECTED_TRAIN_SHA,
            expected_matrix_hashes=expected_matrix_hashes)
        verification_path = RUN_ROOT / "SCORE_VERIFICATION.json"
        if verification_path.exists():
            require(read_json(verification_path) == verification,
                    "Existing independent score-verification receipt differs")
        else:
            write_once(verification_path, verification)
        return existing

    bundles = {}
    for origin in ORIGINS:
        suffix = f"E{origin}"
        path = RUN_ROOT / suffix / f"predictions_{suffix}.npz"
        with np.load(path, allow_pickle=False) as archive:
            bundles[origin] = {name: archive[name].copy() for name in verifier.PREDICTION_KEYS}
    e67 = bundles[67]
    e75 = bundles[75]
    label_rows = np.asarray(e67["row_positions"], dtype=np.int64)
    label_ids = np.asarray(e67["record_ids"]).astype(str)
    query_label_event = next((event for event in _ledger_events()
                              if event.get("event") == "QUERY_LABELS_OPENED"), None)
    open_event_body = {"event": "QUERY_LABELS_OPENED", "candidate_id": "CLS_POS_POISSON_BAG2_V1",
                       "freeze_signature": freeze["freeze_signature"],
                           "root_prefit_pass_sha256": root_pass_sha256,
                       "prediction_ready_sha256": sha256_file(PREDICTIONS_READY),
                       "readback_sha256": sha256_file(readback_path),
                       "score_row_positions_sha256": sha256_array(label_rows),
                       "score_record_ids_sha256": sha256_array(label_ids),
                       "rows": int(len(label_rows)), "source": "official Train artifact_signal",
                       "purpose": "matched E67/E75 development scoring after prediction freeze",
                       "test_or_sample_access": False}
    if query_label_event is None:
        _append_ledger(open_event_body)
    else:
        require({k: v for k, v in query_label_event.items() if k != "created_at_utc"}
                == {k: v for k, v in open_event_body.items() if k != "created_at_utc"},
                "Existing query-label event differs; refusing label re-open under changed inputs")
    labels_e67 = _read_selected_labels(Path(repository) / "data/raw/classification/Train.csv",
                                       label_rows, label_ids)
    labels_by_row = {int(row): int(y) for row, y in zip(label_rows.tolist(), labels_e67.tolist())}

    def score_origin(origin: int) -> tuple[dict[str, Any], dict[str, Any]]:
        block = bundles[origin]
        rows = np.asarray(block["row_positions"], dtype=np.int64)
        y = np.asarray([labels_by_row[int(row)] for row in rows], dtype=np.int8)
        cycles = np.asarray(block["cycles"], dtype=np.int64)
        present = np.asarray(block["present_mask"], dtype=bool)
        champ = np.asarray(block["label_champion"], dtype=np.int8)
        cand = np.asarray(block["label_candidate"], dtype=np.int8)
        if origin == 67:
            views = {"W4": (cycles >= 68) & (cycles <= 71),
                     "W5": (cycles >= 72) & (cycles <= 75),
                     "W6": (cycles >= 76) & (cycles <= 79),
                     "H12_E67": (cycles >= 68) & (cycles <= 79)}
        else:
            views = {"W6": (cycles >= 76) & (cycles <= 79)}
        view_metrics = {}
        subgroup_metrics = {}
        for view_name, view_mask in views.items():
            view_metrics[view_name] = _metric_record(y[view_mask],
                np.asarray(block["p_champion"])[view_mask], np.asarray(block["p_candidate"])[view_mask],
                champ[view_mask], cand[view_mask])
            group_arrays = [("stream_availability", "present", present),
                            ("stream_availability", "missing", ~present)]
            for name, value in _group_keys(block["text_load_group"]):
                group_arrays.append(("fit_prefix_text_load", name,
                                     np.asarray(block["text_load_group"]).astype(str) == name))
            for name, value in _group_keys(block["entry_friction"]):
                group_arrays.append(("entry_friction", name,
                                     np.asarray(block["entry_friction"]).astype(str) == name))
            subgroup_metrics[view_name] = {}
            for group, value, group_mask in group_arrays:
                mask = view_mask & group_mask
                support = {"n": int(mask.sum()), "positive": int(y[mask].sum()),
                           "negative": int(mask.sum() - y[mask].sum())}
                supported = support["n"] >= 200 and support["positive"] >= 30 and support["negative"] >= 30
                metrics = (_metric_record(y[mask], np.asarray(block["p_champion"])[mask],
                                          np.asarray(block["p_candidate"])[mask], champ[mask], cand[mask])
                           if supported else None)
                subgroup_metrics[view_name][f"{group}={value}"] = {
                    "support": support, "supported": supported, "metrics": metrics}
        per_cycle = {}
        for cycle in sorted(np.unique(cycles).tolist()):
            mask = cycles == cycle
            per_cycle[str(int(cycle))] = _metric_record(y[mask],
                np.asarray(block["p_champion"])[mask], np.asarray(block["p_candidate"])[mask],
                champ[mask], cand[mask])
        return {"origin": origin, "views": view_metrics, "subgroups": subgroup_metrics,
                "per_cycle": per_cycle}, {"rows": rows, "y": y, "cycles": cycles,
                                           "present": present, "champ": champ, "cand": cand}

    scored = {origin: score_origin(origin) for origin in ORIGINS}
    e67_metrics, e67_rows = scored[67]
    e75_metrics, e75_rows = scored[75]
    primary_mask_67 = (e67_rows["cycles"] >= 68) & (e67_rows["cycles"] <= 75)
    primary_rows = np.concatenate([e67_rows["rows"][primary_mask_67], e75_rows["rows"]])
    primary_ids = np.concatenate([e67["record_ids"][primary_mask_67], e75["record_ids"]]).astype(str)
    require(len(set(primary_ids.tolist())) == 20903 and len(primary_ids) == 20903,
            "Unique primary pool must contain 20,903 nonoverlapping rows")
    primary_y = _read_selected_labels(Path(repository) / "data/raw/classification/Train.csv",
                                      primary_rows, primary_ids)
    primary_champ = np.concatenate([e67_rows["champ"][primary_mask_67], e75_rows["champ"]])
    primary_cand = np.concatenate([e67_rows["cand"][primary_mask_67], e75_rows["cand"]])
    unique_pool = _metric_record(primary_y,
        np.concatenate([e67["p_champion"][primary_mask_67], e75["p_champion"]]),
        np.concatenate([e67["p_candidate"][primary_mask_67], e75["p_candidate"]]),
        primary_champ, primary_cand)
    primary_cycles = np.concatenate([e67_rows["cycles"][primary_mask_67], e75_rows["cycles"]])
    unique_pool_per_cycle = {}
    for cycle in sorted(np.unique(primary_cycles).tolist()):
        mask = primary_cycles == cycle
        unique_pool_per_cycle[str(int(cycle))] = _metric_record(
            primary_y[mask],
            np.concatenate([e67["p_champion"][primary_mask_67], e75["p_champion"]])[mask],
            np.concatenate([e67["p_candidate"][primary_mask_67], e75["p_candidate"]])[mask],
            primary_champ[mask], primary_cand[mask])

    def supported_harms(origin_metrics: Mapping[str, Any]) -> tuple[bool, bool, list[dict[str, Any]]]:
        all_legacy = True
        all_exploratory = True
        failures = []
        for view, groups in origin_metrics["subgroups"].items():
            for group, info in groups.items():
                if not info["supported"]:
                    continue
                metrics = info["metrics"]
                f1_harm = metrics["gain_macro_f1"]
                recall_harm = metrics["gain_positive_recall"]
                if group == "fit_prefix_text_load=Q4":
                    if f1_harm < -0.01 or recall_harm < -0.02:
                        all_legacy = False
                        failures.append({"view": view, "group": group, "guard": "inherited_Q4",
                                         "macro_f1_gain": f1_harm,
                                         "positive_recall_gain": recall_harm})
                if f1_harm < -0.02 or recall_harm < -0.03:
                    all_exploratory = False
                    failures.append({"view": view, "group": group, "guard": "exploratory_supported_group",
                                     "macro_f1_gain": f1_harm,
                                     "positive_recall_gain": recall_harm})
        return all_legacy, all_exploratory, failures

    legacy_q4_ok_67, exploratory_subgroups_67, failures_67 = supported_harms(e67_metrics)
    legacy_q4_ok_75, exploratory_subgroups_75, failures_75 = supported_harms(e75_metrics)
    window_guards = {name: e67_metrics["views"][name]["gain_macro_f1"] >= -0.005
                     for name in ("W4", "W5", "W6")}
    e67_h12_gain = e67_metrics["views"]["H12_E67"]["gain_macro_f1"]
    e75_w6_gain = e75_metrics["views"]["W6"]["gain_macro_f1"]
    primary_gain = unique_pool["gain_macro_f1"]
    scientific_qualified = (primary_gain >= 0.0005 and e75_w6_gain >= 0.0
                            and e67_h12_gain >= -0.0005
                            and all(window_guards.values())
                            and legacy_q4_ok_67 and legacy_q4_ok_75)
    exploratory_gains_ok = (primary_gain > 0.0 or
                            (e75_w6_gain >= 0.0005 and primary_gain >= -0.0005))
    exploratory_eligible = (exploratory_gains_ok and exploratory_subgroups_67
                            and exploratory_subgroups_75)
    result = {
        "schema_version": 1, "candidate_id": "CLS_POS_POISSON_BAG2_V1",
        "status": "SCORED_EXPLORATORY_ONLY", "freeze_signature": freeze["freeze_signature"],
        "source_commit": freeze["audit_source_commit"],
        "root_prefit_pass_sha256": root_pass_sha256,
        "candidate_role": "two fixed Poisson-weighted gamma replicas; no automatic champion change",
        "threshold_policy": {"present": 0.50, "missing": 0.53230976129477,
                             "missing_threshold_is_global_oof_not_origin_learned": True,
                             "paired_historical_development_not_fresh_validation": True},
        "unique_primary_pool": {"rows": int(len(primary_y)), "cycles": [68, 79],
                                "definition": "E67 cycles 68-75 + E75 cycles 76-79",
                                "metrics": unique_pool, "per_cycle": unique_pool_per_cycle},
        "E67": e67_metrics,
        "E75": e75_metrics,
        "qualification": {"scientific_qualified": bool(scientific_qualified),
                           "exploratory_eligible": bool(exploratory_eligible),
                           "unique_pool_gain_min": 0.0005,
                           "E75_W6_gain_min": 0.0,
                           "E67_H12_gain_min": -0.0005,
                           "E67_W4_W5_W6_harm_max": 0.005,
                           "inherited_Q4_macro_harm_max": 0.01,
                           "inherited_Q4_positive_recall_harm_max": 0.02,
                           "exploratory_supported_group_f1_harm_max": 0.02,
                           "exploratory_supported_group_recall_harm_max": 0.03,
                           "window_guards": window_guards,
                           "subgroup_guard_failures": failures_67 + failures_75},
        "input_lineage": {"train_sha256": EXPECTED_TRAIN_SHA,
                          "scored_e67_target_sha256": sha256_array(labels_e67),
                          "query_label_event_sha256": canonical_sha256(open_event_body),
                          "prediction_bundles_sha256": {
                              f"E{origin}": sha256_file(RUN_ROOT / f"E{origin}" / f"predictions_E{origin}.npz")
                              for origin in ORIGINS},
                          "prediction_readback_sha256": sha256_file(readback_path)},
        "query_labels_read_after_predictions_verified": True,
        "test_or_sample_access": False, "kaggle_upload": False,
    }
    write_once(RESULTS_PATH, result)
    verification = verifier.verify_score_result(
        RESULTS_PATH, RUN_ROOT, Path(repository) / "data/raw/classification/Train.csv",
        freeze_signature=freeze["freeze_signature"],
        source_commit=freeze["audit_source_commit"],
        root_pass_sha256=root_pass_sha256, train_sha256=EXPECTED_TRAIN_SHA,
        expected_matrix_hashes=expected_matrix_hashes)
    write_once(RUN_ROOT / "SCORE_VERIFICATION.json", verification)
    return result


def score_predictions(repository: Path) -> dict[str, Any]:
    """Serialize delayed Train-label access and result persistence for resume safety."""
    with _os_lock(RUN_ROOT / ".scoring.lock"):
        return _score_predictions_locked(repository)


def run_synthetic(output_dir: Path) -> dict[str, Any]:
    """Exercise fit -> persistence -> resume -> prediction -> independent readback."""
    helper = _load_local_module("_cls_pos_poisson_helpers", AUDIT / "src/models/cls_pos_poisson_bag2_v1.py")
    verifier = _load_local_module("_cls_pos_poisson_verifier", AUDIT / "scripts/verify_cls_pos_poisson_bag2_v1.py")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = AUDIT / "src/models/cls_pos_poisson_bag2_v1.py"
    runner_path = AUDIT / "scripts/cls_pos_poisson_bag2_v1.py"
    verifier_path = AUDIT / "scripts/verify_cls_pos_poisson_bag2_v1.py"
    source_hashes = {p.name: sha256_file(p) for p in (model_path, runner_path, verifier_path)}
    synthetic_config = {
        "candidate_id": "CLS_POS_POISSON_BAG2_V1_SYNTHETIC",
        "source_hashes": source_hashes,
        "rows_fit": 48, "rows_score_per_origin": 32, "features": 6,
        "fit_ids": RESERVATION_FIT_IDS, "seeds": [17, 137],
        "threshold_present": 0.50, "threshold_missing": 0.53230976129477,
    }
    freeze_signature = canonical_sha256(synthetic_config)
    manifest_path = output_dir / "SYNTHETIC_FREEZE.json"
    if manifest_path.exists():
        require(read_json(manifest_path) == {**synthetic_config, "freeze_signature": freeze_signature},
                "Synthetic output belongs to different code/config; use a new output directory")
    else:
        require(not any(output_dir.iterdir()), "Synthetic output directory is nonempty without a freeze")
        write_once(manifest_path, {**synthetic_config, "freeze_signature": freeze_signature})

    rng = np.random.default_rng(20261106)
    n_fit, n_score, n_features = 48, 32, 6
    x_fit = sparse.csr_matrix(rng.normal(size=(n_fit, n_features)))
    offset_fit = rng.normal(scale=0.25, size=n_fit)
    p_fit = expit(offset_fit + np.asarray(x_fit @ np.asarray([.45, -.35, .2, .3, -.25, .15])))
    y_fit = rng.binomial(1, p_fit).astype(np.int8)
    require(np.unique(y_fit).size == 2, "Synthetic fixture unexpectedly has one fit class")
    fit_ids = np.asarray([f"SYNTH-FIT-{i:03d}" for i in range(n_fit)])
    fit_rows = np.arange(n_fit, dtype=np.int64)
    x_score = sparse.csr_matrix(rng.normal(size=(n_score, n_features)))
    offset_score = rng.normal(scale=0.25, size=n_score)
    present = (np.arange(n_score) % 4 != 0)
    p_a02 = np.clip(expit(offset_score + rng.normal(scale=.35, size=n_score)), .02, .98)
    score_ids = np.asarray([f"SYN-SCORE-{i:03d}" for i in range(n_score)])
    score_rows = np.arange(100, 100 + n_score, dtype=np.int64)
    score_truth = rng.binomial(1, np.clip(.15 + .7 * p_a02, .05, .95)).astype(np.int8)
    target_path = output_dir / "SYNTHETIC_SCORE_TRUTH.npz"
    if target_path.exists():
        with np.load(target_path, allow_pickle=False) as archive:
            require(archive.files == ["score_truth"], "Synthetic truth fixture schema drift")
            require(np.array_equal(archive["score_truth"], score_truth), "Synthetic truth fixture changed")
    else:
        atomic_npz(target_path, score_truth=score_truth)

    artifact_root = output_dir
    origin_context = {}
    fit_records = {}
    for origin in ORIGINS:
        origin_seed = 1000 + origin
        origin_rng = np.random.default_rng(origin_seed)
        gamma_original = origin_rng.normal(scale=.08, size=n_features)
        z_fit = offset_fit + np.asarray(x_fit @ gamma_original).reshape(-1)
        z_score = offset_score + np.asarray(x_score @ gamma_original).reshape(-1)
        matrix_hash = sha256_csr(sparse.vstack([x_fit, x_score], format="csr"))
        fit_ids_for_origin = [fit_id for fit_id, (fit_origin, _) in FIT_ID_TO_ORIGIN_SEED.items()
                              if fit_origin == origin]
        gammas = {"original": gamma_original}
        for fit_id in fit_ids_for_origin:
            _, seed = FIT_ID_TO_ORIGIN_SEED[fit_id]
            artifact_path = artifact_root / "units" / f"{fit_id}.npz"
            receipt_path = artifact_root / "receipts" / f"{fit_id}.json"
            if artifact_path.exists() or receipt_path.exists():
                require(artifact_path.exists() and receipt_path.exists(),
                        f"Synthetic fit persistence is incomplete: {fit_id}")
                verified = verifier.verify_fit_artifact(artifact_root, fit_id, freeze_signature)
                receipt = verified["receipt"]
                require(receipt["fit_row_positions_sha256"] == sha256_array(fit_rows)
                        and receipt["fit_record_ids_sha256"] == sha256_array(fit_ids)
                        and receipt["fit_target_sha256"] == sha256_array(y_fit)
                        and receipt["matrix_manifest_sha256"] == matrix_hash
                        and receipt["offset_fit_sha256"] == sha256_array(z_fit)
                        and receipt["gamma_original_sha256"] == sha256_array(gamma_original),
                        f"Synthetic resumed fit lineage mismatch: {fit_id}")
                gamma = verified["gamma"]
                fit_records[fit_id] = {"gamma_sha256": receipt["gamma_sha256"],
                                       "receipt_sha256": verified["receipt_sha256"]}
            else:
                raw_counts, row_weights = helper.poisson_row_weights(n_fit, seed)
                with threadpool_limits(limits=2):
                    gamma, optimizer = helper.fit_poisson_weighted_offset(
                        x_fit, z_fit, y_fit, row_weights)
                artifact_path.parent.mkdir(parents=True, exist_ok=True)
                receipt_path.parent.mkdir(parents=True, exist_ok=True)
                atomic_npz(artifact_path, gamma=gamma)
                receipt = {
                    "schema_version": 1, "fit_id": fit_id, "origin": origin,
                    "replica_seed": seed, "fit_rows": n_fit,
                    "position_features": n_features, "attempt_number": 1,
                    "retry_allowed": False, "synthetic_only": True,
                    "fit_row_positions_sha256": sha256_array(fit_rows),
                    "fit_record_ids_sha256": sha256_array(fit_ids),
                    "fit_target_sha256": sha256_array(y_fit),
                    "input_freeze_signature": freeze_signature,
                    "matrix_manifest_sha256": matrix_hash,
                    "fit_position_sha256": sha256_csr(x_fit),
                    "offset_fit_sha256": sha256_array(z_fit),
                    "gamma_original_sha256": sha256_array(gamma_original),
                    "poisson_raw_counts_sha256": sha256_array(raw_counts),
                    "poisson_weights_sha256": sha256_array(row_weights),
                    "poisson_raw_sum": int(raw_counts.sum()),
                    "poisson_zero_count": int(np.count_nonzero(raw_counts == 0)),
                    "gamma_sha256": sha256_array(gamma),
                    "artifact_path": artifact_path.relative_to(artifact_root).as_posix(),
                    "artifact_sha256": sha256_file(artifact_path),
                    "optimizer": {**optimizer, "replica_seed": seed,
                                  "poisson_lambda": 1.0, "max_threads": 2},
                    "no_test_or_sample_access": True, "no_query_labels_in_fit": True,
                }
                write_once(receipt_path, receipt)
                verified = verifier.verify_fit_artifact(artifact_root, fit_id, freeze_signature)
                fit_records[fit_id] = {"gamma_sha256": verified["receipt"]["gamma_sha256"],
                                       "receipt_sha256": verified["receipt_sha256"]}
            gammas[str(seed)] = gamma

        policy = helper.build_policy_probabilities(x_score, offset_score, gamma_original,
            gammas["17"], gammas["137"], p_a02, present)
        labels_champ = helper.apply_fixed_thresholds(policy["probability_champion"], present)
        labels_candidate = helper.apply_fixed_thresholds(policy["probability_candidate"], present)
        origin_rows = score_rows + (0 if origin == 67 else 100)
        origin_cycles = np.resize(np.asarray(
            list(range(68, 80)) if origin == 67 else [76, 77, 78, 79], dtype=np.int64), n_score)
        origin_context[origin] = {
            "origin": origin,
            "row_positions": origin_rows,
            "record_ids": np.asarray([f"E{origin}-{item}" for item in score_ids]),
            "cycles": origin_cycles,
            "present_mask": present,
            "text_load_group": np.where(np.arange(n_score) % 2 == 0, "Q1", "Q4"),
            "entry_friction": np.where(np.arange(n_score) % 3 == 0, "high", "low"),
            "p_a02": p_a02,
            "p_pos_original": policy["probability_pos_original"],
            "p_pos_seed17": helper.offset_probability(x_score, offset_score, gammas["17"]),
            "p_pos_seed137": helper.offset_probability(x_score, offset_score, gammas["137"]),
            "p_pos_bag2": policy["probability_pos_bag2"],
            "p_champion": policy["probability_champion"],
            "p_candidate": policy["probability_candidate"],
            "label_champion": labels_champ,
            "label_candidate": labels_candidate,
            "matrix_manifest_sha256": np.asarray([matrix_hash]),
            "input_freeze_signature": np.asarray([freeze_signature]),
        }

    prediction_hashes = {}
    for origin, bundle in origin_context.items():
        path = artifact_root / f"E{origin}" / f"predictions_E{origin}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            verified = verifier.verify_prediction_bundle(path, origin=origin,
                                                         freeze_signature=freeze_signature)
            with np.load(path, allow_pickle=False) as archive:
                require(all(np.array_equal(archive[key], value) for key, value in bundle.items()),
                        f"Synthetic resumed predictions differ for E{origin}")
        else:
            atomic_npz(path, **bundle)
            verified = verifier.verify_prediction_bundle(path, origin=origin,
                                                         freeze_signature=freeze_signature)
        prediction_hashes[f"E{origin}"] = verified["bundle_sha256"]

    readback1 = verifier.verify_readback_pair(artifact_root, freeze_signature)
    readback2 = verifier.verify_readback_pair(artifact_root, freeze_signature)
    require(canonical_sha256(readback1) == canonical_sha256(readback2),
            "Synthetic repeated independent readback differs")
    metric_records = {}
    for origin, bundle in origin_context.items():
        metrics_champ = helper.metric_summary(score_truth, bundle["p_champion"], present)
        metrics_candidate = helper.metric_summary(score_truth, bundle["p_candidate"], present)
        metric_records[f"E{origin}"] = {
            "champion": metrics_champ, "candidate": metrics_candidate,
            "gain_macro_f1": metrics_candidate["macro_f1"] - metrics_champ["macro_f1"],
        }
    result = {
        "status": "PASS_SYNTHETIC_E2E", "synthetic_only": True,
        "freeze_signature": freeze_signature, "source_hashes": source_hashes,
        "fit_records": fit_records, "prediction_hashes": prediction_hashes,
        "readback_sha256": canonical_sha256(readback2),
        "metrics": metric_records, "official_inputs_opened": False,
        "official_outputs_touched": False, "query_labels_used": False,
        "test_or_sample_access": False, "kaggle_upload": False,
    }
    result_path = artifact_root / "SYNTHETIC_RESULT.json"
    if result_path.exists():
        require(read_json(result_path) == result, "Synthetic persisted result changed on resume")
    else:
        write_once(result_path, result)

    cache_path = artifact_root / "SYNTHETIC_A02_CACHE_LABEL_GUARD.npz"
    if not cache_path.exists():
        atomic_npz(cache_path, prediction=np.asarray([.2, .8]), score_rows=np.asarray([1, 2]),
                   score_record_ids=np.asarray(["a", "b"]), score_cycles=np.asarray([68, 69]),
                   missing_mask=np.asarray([False, True]), entry_friction=np.asarray(["low", "high"]),
                   text_load_group=np.asarray(["Q1", "Q4"]), labels=np.asarray([object()], dtype=object))
    cache_values = _read_a02_cache(cache_path)
    require("labels" not in cache_values and set(cache_values) == {
        "prediction", "score_rows", "score_record_ids", "score_cycles", "missing_mask",
        "entry_friction", "text_load_group"}, "A02 cache loader exposed an unexpected member")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command", required=True,
                        choices=("synthetic", "freeze", "verify-freeze", "reserve",
                                 "fit", "predictions", "score"))
    parser.add_argument("--repository", type=Path,
                        help="Official TP2 checkout; required for official commands")
    parser.add_argument("--fit-id", choices=RESERVATION_FIT_IDS)
    parser.add_argument("--output-dir", type=Path,
                        help="Isolated scratch directory for synthetic lifecycle only")
    args = parser.parse_args(argv)
    if args.command == "synthetic":
        require(args.output_dir is not None, "--output-dir is required for synthetic QA")
        result = run_synthetic(args.output_dir)
    else:
        require(args.repository is not None, "--repository is required for official commands")
        if args.command == "freeze":
            result = create_prefit_freeze(args.repository)
        elif args.command == "verify-freeze":
            result = verify_prefit_freeze(args.repository)
        elif args.command == "reserve":
            result = reserve_fit_cap(args.repository)
        elif args.command == "fit":
            require(args.fit_id is not None, "--fit-id is required for one official fit")
            result = fit_one(args.repository, args.fit_id)
        elif args.command == "predictions":
            result = build_predictions(args.repository)
        else:
            result = score_predictions(args.repository)
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
