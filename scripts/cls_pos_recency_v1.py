"""Four-fit, gamma-only temporal-recency pilot for the positional CLS channel."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import psutil
from scipy import sparse
from sklearn.metrics import (average_precision_score, f1_score, log_loss,
                            precision_score, recall_score, roc_auc_score)
from threadpoolctl import threadpool_limits

from src.models import cls_pos_recency_v1 as core

AUDIT_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = AUDIT_ROOT / "reports/simplification_v1/CLS_POS_RECENCY_V1"
RUN_ROOT = AUDIT_ROOT / "experiments/CLS_POS_RECENCY_V1"
FREEZE_PATH = REPORT_ROOT / "FREEZE.json"
PREFIT_QA_PATH = REPORT_ROOT / "PREFIT_QA.json"
RESERVATION_PATH = REPORT_ROOT / "RESERVATION.json"
READY_PATH = REPORT_ROOT / "PREDICTIONS_READY.json"
PRELABEL_PATH = REPORT_ROOT / "PRELABEL_VERIFICATION.json"
SCORE_PATH = REPORT_ROOT / "SCORE.json"
METRIC_PATH = REPORT_ROOT / "METRIC_VERIFICATION.json"
SCORE_VERIFY_PATH = REPORT_ROOT / "SCORE_VERIFICATION.json"
LEDGER_PATH = RUN_ROOT / "fit_ledger.jsonl"
ROOT_PLAN_REL = "reports/simplification_v1/CLS_POS_RECENCY_V1/ROOT_PLAN.md"
ROOT_PLAN_PATH = AUDIT_ROOT / ROOT_PLAN_REL
ROOT_PLAN_COMMIT = "e68b4baa6fa4a5862d07b27fbc0415424435d27c"
SOURCE_FILES = (
    "src/models/cls_pos_recency_v1.py",
    "scripts/cls_pos_recency_v1.py",
    "scripts/verify_cls_pos_recency_v1.py",
    "tests/test_cls_pos_recency_v1.py",
    "tests/test_cls_pos_recency_v1_synthetic.py",
    "reports/simplification_v1/CLS_POS_RECENCY_V1/PROTOCOL.md",
)
EXTRA_LOCAL_INPUTS = (
    "src/models/classification_models.py",
    "scripts/cls_pos_poisson_bag2_v1.py",
    "src/models/cls_pos_poisson_bag2_v1.py",
)
OFFICIAL_TRAIN_SHA256 = "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c"
MIN_FIT_AVAILABLE = 1024**3
MIN_FIT_RESERVE = 512 * 1024**2
MIN_SCORE_AVAILABLE = 512 * 1024**2
MIN_SCORE_RESERVE = 256 * 1024**2
FIT_CAP = 4
SCORE_WINDOWS = {
    "E67": {"W4": (68, 71), "W5": (72, 75), "W6": (76, 79), "H12_E67": (68, 79)},
    "E75": {"W6": (76, 79)},
}


class PilotError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PilotError(message)


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(value: Any) -> str:
    array = np.asarray(value)
    header = canonical_sha256({"shape": list(array.shape), "dtype": str(array.dtype)})
    h = hashlib.sha256(header.encode("ascii"))
    if array.dtype.kind in "OUS":
        h.update(json.dumps(array.astype(str).tolist(), ensure_ascii=False,
                            separators=(",", ":")).encode("utf-8"))
    else:
        h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


def sha256_csr(value: sparse.spmatrix) -> str:
    x = sparse.csr_matrix(value, dtype=np.float64, copy=True)
    x.sum_duplicates(); x.eliminate_zeros(); x.sort_indices()
    h = hashlib.sha256()
    h.update(canonical_json({"shape": list(x.shape), "dtype": str(x.dtype), "format": "csr"}))
    for part in (x.indptr, x.indices, x.data):
        h.update(np.ascontiguousarray(part).tobytes())
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try: os.unlink(temporary)
        except OSError: pass
        raise


def write_once(path: Path, value: Any) -> None:
    payload = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
    if Path(path).exists():
        require(Path(path).read_bytes() == payload, f"Refusing to overwrite non-identical artifact: {path}")
        return
    atomic_bytes(path, payload)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    with temporary.open("r+b") as stream:
        stream.flush(); os.fsync(stream.fileno())
    if path.exists():
        require(sha256_file(path) == sha256_file(temporary), f"Refusing to replace different NPZ: {path}")
        temporary.unlink()
    else:
        os.replace(temporary, path)


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(AUDIT_ROOT), *args], text=True,
                                  encoding="utf-8", stderr=subprocess.STDOUT).strip()


def git_in(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *args], text=True,
                                  encoding="utf-8", stderr=subprocess.STDOUT).strip()


def _is_ancestor(commit: str) -> bool:
    return subprocess.run(["git", "-C", str(AUDIT_ROOT), "merge-base", "--is-ancestor", commit, "HEAD"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def require_committed(relative: str) -> None:
    path = AUDIT_ROOT / relative
    require(path.is_file(), f"Required committed artifact is absent: {relative}")
    committed = subprocess.check_output(["git", "-C", str(AUDIT_ROOT), "show", f"HEAD:{relative}"])
    require(committed == path.read_bytes(), f"Artifact must be committed before this stage: {relative}")


def _source_hashes() -> dict[str, str]:
    paths = set(SOURCE_FILES) | set(EXTRA_LOCAL_INPUTS) | {ROOT_PLAN_REL}
    return {relative: sha256_file(AUDIT_ROOT / relative) for relative in sorted(paths)}


def _assert_committed_sources() -> str:
    status = git("status", "--porcelain", "--", *SOURCE_FILES, *EXTRA_LOCAL_INPUTS, ROOT_PLAN_REL)
    require(not status, f"Lane source or named local dependency is modified/untracked:\n{status}")
    for relative in (*SOURCE_FILES, *EXTRA_LOCAL_INPUTS, ROOT_PLAN_REL):
        require_committed(relative)
    source_commit = git("rev-parse", "HEAD")
    require(_is_ancestor(ROOT_PLAN_COMMIT) and source_commit != "", "Root authorization commit is not an ancestor")
    return source_commit


def append_event(event: Mapping[str, Any]) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    value = {"at_utc": datetime.now(timezone.utc).isoformat(), **dict(event)}
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                         separators=(",", ":")) + "\n"
    with LEDGER_PATH.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())


def events() -> list[dict[str, Any]]:
    if not LEDGER_PATH.exists():
        return []
    return [json.loads(line) for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def _parent_runner():
    from scripts import cls_pos_poisson_bag2_v1 as parent
    return parent


def _parent_context(official_root: Path, origin: int) -> dict[str, Any]:
    parent = _parent_runner()
    return parent._load_origin_context(Path(official_root), origin, require_fit_labels=False)


def _read_selected_stream_presence(
    train_path: Path, rows: Sequence[int], record_ids: Sequence[str],
) -> np.ndarray:
    from src.models.classification_models import _text_values
    wanted = np.asarray(rows, dtype=np.int64).reshape(-1)
    ids = np.asarray(record_ids).astype(str).reshape(-1)
    require(len(wanted) == len(ids) and len(wanted) > 0
            and len(np.unique(wanted)) == len(wanted) and len(np.unique(ids)) == len(ids),
            "Presence rows and IDs must align uniquely")
    selected = set(int(row) for row in wanted.tolist())
    frame = pd.read_csv(Path(train_path), usecols=["record_id", "signal_stream"],
        dtype={"record_id": "string", "signal_stream": "string"}, keep_default_na=False,
        skiprows=lambda line: line > 0 and (line - 1) not in selected)
    order = np.argsort(wanted, kind="stable")
    require(len(frame) == len(wanted)
            and np.array_equal(frame["record_id"].astype(str).to_numpy(), ids[order]),
            "Selected Train stream rows do not match frozen record IDs")
    _, missing = _text_values(frame["signal_stream"])
    presence_sorted = ~np.asarray(missing, dtype=bool)
    presence = np.empty_like(presence_sorted)
    presence[order] = presence_sorted
    return presence


def _validate_origin_context(official_root: Path, origin: int, context: Mapping[str, Any]) -> dict[str, Any]:
    require(context["origin"] == origin and context["y_fit"] is None,
            f"Parent context E{origin} is not the expected target-free origin")
    fit_rows = np.asarray(context["fit_rows"], dtype=np.int64)
    fit_ids = np.asarray(context["fit_ids"]).astype(str)
    fit_cycles = np.asarray(context["old_runner"]._read_identity_frame().iloc[fit_rows]["cycle_code"], dtype=np.int64)
    fit_present = _read_selected_stream_presence(
        Path(official_root) / "data/raw/classification/Train.csv", fit_rows, fit_ids)
    matrix_present = np.asarray(context["x_fit"].getnnz(axis=1)).reshape(-1) > 0
    require(np.array_equal(fit_present, matrix_present),
            f"E{origin} stream presence does not exactly match nonzero positional rows")
    score_present = _read_selected_stream_presence(
        Path(official_root) / "data/raw/classification/Train.csv",
        context["score_rows"], context["score_ids"])
    require(np.array_equal(score_present, np.asarray(context["present"], dtype=bool)),
            f"E{origin} query stream presence differs from frozen source route")
    prefix = (47, origin)
    require(len(fit_rows) in (29705, 42117)
            and np.all((fit_cycles >= prefix[0]) & (fit_cycles <= prefix[1])),
            f"E{origin} fit rows escape fixed prefix")
    score_cycles = np.asarray(context["cycles"], dtype=np.int64)
    require(np.all((score_cycles >= origin + 1) & (score_cycles <= 79)),
            f"E{origin} score rows escape fixed temporal window")
    manifest_nonzero = int(context["manifest"]["position_audit"]["fit_original"]["nonzero_rows"])
    require(int(fit_present.sum()) == manifest_nonzero,
            f"E{origin} present fit count differs from frozen position audit")
    return {
        "origin": origin,
        "fit_rows": int(len(fit_rows)),
        "fit_present_rows": int(fit_present.sum()),
        "fit_missing_rows": int((~fit_present).sum()),
        "fit_rows_sha256": sha256_array(fit_rows),
        "fit_present_rows_sha256": sha256_array(fit_rows[fit_present]),
        "fit_present_ids_sha256": sha256_array(fit_ids[fit_present]),
        "fit_present_cycles_sha256": sha256_array(fit_cycles[fit_present]),
        "fit_position_sha256": sha256_csr(context["x_fit"][fit_present]),
        "score_rows": int(len(context["score_rows"])),
        "score_present_rows": int(score_present.sum()),
        "score_missing_rows": int((~score_present).sum()),
        "score_rows_sha256": sha256_array(context["score_rows"]),
        "score_ids_sha256": sha256_array(context["score_ids"]),
        "score_cycles_sha256": sha256_array(context["cycles"]),
        "original_gamma_sha256": sha256_array(context["gamma_original"]),
        "a02_probabilities_sha256": sha256_array(context["p_a02"]),
        "position_width": int(context["x_fit"].shape[1]),
        "fit_position_nnz": int(context["x_fit"].nnz),
        "score_position_nnz": int(context["x_score"].nnz),
        "text_load_group_counts": {str(key): int(count) for key, count in zip(
            *np.unique(context["text_load_group"].astype(str), return_counts=True))},
        "query_labels_read": False,
        "test_or_sample_access": False,
    }


def _read_selected_targets(train_path: Path, rows: Sequence[int], record_ids: Sequence[str]) -> np.ndarray:
    wanted = np.asarray(rows, dtype=np.int64).reshape(-1)
    ids = np.asarray(record_ids).astype(str).reshape(-1)
    require(len(wanted) == len(ids) and len(wanted) > 0
            and len(np.unique(wanted)) == len(wanted) and len(np.unique(ids)) == len(ids),
            "Selected target rows and IDs must align uniquely")
    selected = set(int(row) for row in wanted.tolist())
    frame = pd.read_csv(Path(train_path), usecols=["record_id", "artifact_signal"],
        dtype={"record_id": "string", "artifact_signal": "int8"}, keep_default_na=False,
        skiprows=lambda line: line > 0 and (line - 1) not in selected)
    order = np.argsort(wanted, kind="stable")
    require(len(frame) == len(wanted)
            and np.array_equal(frame["record_id"].astype(str).to_numpy(), ids[order]),
            "Selected official Train targets do not match frozen record IDs")
    values_sorted = frame["artifact_signal"].to_numpy(dtype=np.int8)
    require(np.isin(values_sorted, (0, 1)).all(), "Selected targets must be binary 0/1")
    values = np.empty_like(values_sorted)
    values[order] = values_sorted
    return values


def _preflight_weight_checks(
    official_root: Path, contexts: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    payload: dict[int, dict[str, np.ndarray]] = {}
    for origin, context in contexts.items():
        rows = np.asarray(context["fit_rows"], dtype=np.int64)
        ids = np.asarray(context["fit_ids"]).astype(str)
        present = _read_selected_stream_presence(
            Path(official_root) / "data/raw/classification/Train.csv", rows, ids)
        identity = context["old_runner"]._read_identity_frame()
        cycles = identity.iloc[rows]["cycle_code"].to_numpy(dtype=np.int64)
        payload[origin] = {"rows": rows[present], "ids": ids[present], "cycles": cycles[present]}
    rows75 = payload[75]["rows"]
    ids75 = payload[75]["ids"]
    require(np.all(np.isin(payload[67]["rows"], rows75)),
            "E67 prefix-present rows are not a subset of E75 prefix-present rows")
    preflight_event = {"event": "PREFIT_PREFIX_LABELS_OPENED", "candidate_id": core.CANDIDATE_ID,
        "origin": 75, "purpose": "single prefix-only weight/ESS preflight",
        "rows": int(len(rows75)), "rows_sha256": sha256_array(rows75),
        "ids_sha256": sha256_array(ids75), "query_labels_read": False,
        "test_or_sample_access": False}
    append_event(preflight_event)
    try:
        y75 = _read_selected_targets(Path(official_root) / "data/raw/classification/Train.csv", rows75, ids75)
    except BaseException as exc:
        append_event({"event": "PREFIT_PREFIX_LABELS_READ_FAILED", "candidate_id": core.CANDIDATE_ID,
            "origin": 75, "error_type": type(exc).__name__, "error": str(exc),
            "retry_allowed": False, "query_labels_read": False})
        raise
    append_event({"event": "PREFIT_PREFIX_LABELS_READ", "candidate_id": core.CANDIDATE_ID,
        "origin": 75, "rows": int(len(y75)), "rows_sha256": sha256_array(rows75),
        "ids_sha256": sha256_array(ids75), "target_sha256": sha256_array(y75),
        "query_labels_read": False, "test_or_sample_access": False})
    target_map = {int(row): int(label) for row, label in zip(rows75.tolist(), y75.tolist())}
    origin_weight_checks: dict[str, Any] = {}
    for origin in core.ORIGINS:
        item = payload[origin]
        y = np.asarray([target_map[int(row)] for row in item["rows"]], dtype=np.int8)
        base = core.recency_base_weights(item["cycles"], origin)
        recency = core.normalize_class_mass(base, y)
        shuffled, permutation = core.shuffle_weights_within_class(recency, y)
        d_real = core.weight_diagnostics(recency, y)
        d_shuffled = core.weight_diagnostics(shuffled, y)
        for arm, diagnostics in (("RECENCY", d_real), ("SHUFFLED_WITHIN_CLASS", d_shuffled)):
            for label, class_stats in diagnostics["classes"].items():
                require(class_stats["ess_ratio"] >= 0.50,
                        f"E{origin} {arm} class {label} ESS ratio is below the frozen 0.50 floor")
                require(abs(class_stats["weight_sum_minus_rows"]) <= 1e-8,
                        f"E{origin} {arm} class {label} weight mass is not conserved")
        identity = np.arange(len(y), dtype=np.int64)
        require(np.any(permutation != identity) and np.any(shuffled != recency),
                f"E{origin} shuffled control is not a nontrivial weight permutation")
        require(all(np.array_equal(np.sort(shuffled[y == label]), np.sort(recency[y == label]))
                    for label in (0, 1)), f"E{origin} control does not preserve within-class weight multisets")
        origin_weight_checks[f"E{origin}"] = {
            "rows": int(len(y)),
            "positive": int(y.sum()),
            "negative": int(len(y) - y.sum()),
            "fit_rows_sha256": sha256_array(item["rows"]),
            "fit_ids_sha256": sha256_array(item["ids"]),
            "fit_cycles_sha256": sha256_array(item["cycles"]),
            "target_sha256": sha256_array(y),
            "base_weight_sha256": sha256_array(base),
            "recency_weight_sha256": sha256_array(recency),
            "shuffled_weight_sha256": sha256_array(shuffled),
            "shuffle_permutation_sha256": sha256_array(permutation),
            "shuffle_seed": core.SHUFFLE_SEED,
            "nontrivial_permutation_positions": int(np.count_nonzero(permutation != identity)),
            "changed_weight_values": int(np.count_nonzero(shuffled != recency)),
            "sorted_weight_multiset_equal_by_class": True,
            "recency": d_real,
            "shuffled": d_shuffled,
        }
    return {
        "status": "PASS_PREFIX_WEIGHT_PREFLIGHT",
        "prefix_label_reads": 1,
        "prefix_label_read_rows": int(len(rows75)),
        "prefix_label_read_rows_sha256": sha256_array(rows75),
        "prefix_label_read_ids_sha256": sha256_array(ids75),
        "query_labels_read": False,
        "origins": origin_weight_checks,
    }


def preflight_and_freeze(official_root: Path) -> dict[str, Any]:
    source_commit = _assert_committed_sources()
    root = Path(official_root).resolve()
    if FREEZE_PATH.exists() or PREFIT_QA_PATH.exists():
        require(FREEZE_PATH.is_file() and PREFIT_QA_PATH.is_file(),
                "Partial prefit freeze state exists; refusing a second label preflight")
        freeze = verify_freeze(root)
        qa = read_json(PREFIT_QA_PATH)
        return {"prefit_qa": qa, "freeze": freeze, "resumed_existing": True}
    require(not any(event.get("event", "").startswith("PREFIT_PREFIX_LABELS_") for event in events()),
            "A preflight label read was already opened without a durable freeze; refusing to repeat it")
    train_path = root / "data/raw/classification/Train.csv"
    require(train_path.is_file() and sha256_file(train_path) == OFFICIAL_TRAIN_SHA256,
            "Official Classification Train is absent or hash-mismatched")
    parent = _parent_runner()
    parent_hashes = parent._external_file_inventory(root)
    metadata = parent._validate_metadata_only(root, parent_hashes)
    require(parent_hashes.get("data/raw/classification/Train.csv") == OFFICIAL_TRAIN_SHA256,
            "Parent input inventory has a different official Train")
    origin_checks: dict[str, Any] = {}
    contexts: dict[int, Mapping[str, Any]] = {}
    for origin in core.ORIGINS:
        context = _parent_context(root, origin)
        origin_checks[f"E{origin}"] = _validate_origin_context(root, origin, context)
        contexts[origin] = context
    require(origin_checks["E67"]["score_rows"] == 20903
            and origin_checks["E75"]["score_rows"] == 8491,
            "Temporal query row counts differ from the approved fixed windows")
    require(origin_checks["E67"]["score_present_rows"] == 19528
            and origin_checks["E75"]["score_present_rows"] == 7965,
            "Frozen query availability counts differ from the verified parent route")
    prefix_weight_preflight = _preflight_weight_checks(root, contexts)
    contexts.clear()
    body = {
        "schema_version": 1,
        "candidate_id": core.CANDIDATE_ID,
        "status": "PASS_PREFIT_SOURCE_QA",
        "audit_source_commit": source_commit,
        "root_authorization_commit": ROOT_PLAN_COMMIT,
        "root_plan_sha256": sha256_file(ROOT_PLAN_PATH),
        "official_repository_root": str(root),
        "official_repository_commit": git_in(root, "rev-parse", "HEAD"),
        "official_train_sha256": OFFICIAL_TRAIN_SHA256,
        "official_train_rows": 50608,
        "parent_metadata_checks": metadata,
        "parent_input_file_hashes": parent_hashes,
        "origin_checks": origin_checks,
        "prefix_weight_preflight": prefix_weight_preflight,
        "optimizer_starts": 0,
        "preprocessing_fits": 0,
        "prefix_labels_read": prefix_weight_preflight["prefix_label_reads"],
        "prefix_label_rows_read": prefix_weight_preflight["prefix_label_read_rows"],
        "prefix_label_target_sha256": prefix_weight_preflight["prefix_label_target_sha256"],
        "query_labels_read": False,
        "test_or_sample_access": False,
        "kaggle_access": False,
    }
    body["qa_signature"] = canonical_sha256(body)
    write_once(PREFIT_QA_PATH, body)
    freeze = {
        "schema_version": 1,
        "candidate_id": core.CANDIDATE_ID,
        "status": "FROZEN_FOUR_FIT_RECENCY_PILOT",
        "audit_source_commit": source_commit,
        "root_authorization_commit": ROOT_PLAN_COMMIT,
        "root_plan_sha256": sha256_file(ROOT_PLAN_PATH),
        "official_repository_root": str(root),
        "official_repository_commit": body["official_repository_commit"],
        "official_train_sha256": OFFICIAL_TRAIN_SHA256,
        "official_train_rows": 50608,
        "fit_ids": list(core.FIT_IDS),
        "predictor_fit_cap": FIT_CAP,
        "preprocessing_fit_cap": 0,
        "prefix_fit_cycles": {"E67": [47, 67], "E75": [47, 75]},
        "score_cycles": {"E67": [68, 79], "E75": [76, 79]},
        "source_hashes": _source_hashes(),
        "parent_input_file_hashes": parent_hashes,
        "parent_metadata_checks": metadata,
        "origin_checks": origin_checks,
        "prefix_weight_preflight": prefix_weight_preflight,
        "resource_policy": {
            "minimum_available_bytes_before_fit": MIN_FIT_AVAILABLE,
            "minimum_available_bytes_after_fit": MIN_FIT_RESERVE,
            "maximum_threads": 2,
            "parallel_heavy_jobs": 1,
        },
        "recipe": {
            "base_weight": "max(0.25, 2**(-(origin-cycle)/8))",
            "class_normalization": "divide by mean base weight within each prefix-present target class",
            "control": "permute recency weight values within each class; seed 42 reset per origin; class 0 then class 1; stable input order",
            "offset": "saved C2_B logit z_BAG",
            "features": "exact saved Xthirds positional block; present rows only",
            "fit": "gamma from zeros; summed weighted binary logistic loss + 0.5*||gamma||^2; C=1",
            "optimizer": "L-BFGS-B float64, maxiter=2000, ftol=1e-12, gtol=1e-6, maxls=50",
            "prediction": "replace original uniform gamma with one pilot arm; pPOS sigmoid(z_BAG + Xthirds gamma)",
            "blend": "0.50*pPOS + 0.50*pA02 on present rows",
            "missing_route": "pA02 bit-exact; threshold >= 0.53230976129477",
            "present_threshold": 0.50,
            "query_scoring": "one unique E67 row pool; E75 rows reused by ID; no Test/Sample/Kaggle",
        },
        "boundary": {
            "fit_labels": "prefix-present rows only; a single committed preflight label read plus recorded fit-prefix reads",
            "query_labels": "closed until both origins' predictions and independent committed prelabel verify",
            "test_or_sample_access": False,
            "kaggle_access": False,
        },
        "prefit_qa_sha256": sha256_file(PREFIT_QA_PATH),
    }
    freeze["freeze_signature"] = canonical_sha256(freeze)
    write_once(FREEZE_PATH, freeze)
    return {"prefit_qa": body, "freeze": freeze}


def verify_freeze(official_root: Path, *, require_committed_freeze: bool = False) -> dict[str, Any]:
    require(FREEZE_PATH.is_file() and PREFIT_QA_PATH.is_file(), "Committed prefit freeze and QA are required")
    freeze = read_json(FREEZE_PATH)
    signature = freeze.get("freeze_signature")
    body = dict(freeze); body.pop("freeze_signature", None)
    require(signature == canonical_sha256(body), "Freeze signature mismatch")
    require(freeze.get("candidate_id") == core.CANDIDATE_ID
            and freeze.get("fit_ids") == list(core.FIT_IDS)
            and freeze.get("predictor_fit_cap") == FIT_CAP
            and freeze.get("preprocessing_fit_cap") == 0,
            "Frozen fit namespace/cap differs from this candidate")
    require(freeze.get("official_repository_root") == str(Path(official_root).resolve()),
            "Official repository differs from the frozen source root")
    require(_is_ancestor(freeze["audit_source_commit"])
            and _is_ancestor(freeze["root_authorization_commit"]),
            "Source or root authorization commit is not an ancestor")
    require(freeze["root_authorization_commit"] == ROOT_PLAN_COMMIT
            and sha256_file(ROOT_PLAN_PATH) == freeze["root_plan_sha256"],
            "Root authorization document changed")
    require(_source_hashes() == freeze["source_hashes"], "Frozen source files or local dependencies changed")
    qa = read_json(PREFIT_QA_PATH)
    qa_sig = qa.get("qa_signature")
    qa_body = dict(qa); qa_body.pop("qa_signature", None)
    require(qa_sig == canonical_sha256(qa_body) and qa.get("status") == "PASS_PREFIT_SOURCE_QA"
            and sha256_file(PREFIT_QA_PATH) == freeze["prefit_qa_sha256"],
            "Prefit source QA is missing or was modified")
    prefit_opened = [event for event in events() if event.get("event") == "PREFIT_PREFIX_LABELS_OPENED"]
    prefit_read = [event for event in events() if event.get("event") == "PREFIT_PREFIX_LABELS_READ"]
    if (len(prefit_opened) != 1 or len(prefit_read) != 1
        or any(event.get("event") == "PREFIT_PREFIX_LABELS_READ_FAILED" for event in events())
        or qa.get("prefix_labels_read") != 1
        or qa.get("prefix_label_rows_read") != prefit_read[0].get("rows")
        or qa.get("prefix_label_target_sha256") != prefit_read[0].get("target_sha256")
        or prefit_opened[0].get("rows_sha256") != prefit_read[0].get("rows_sha256")
        or prefit_opened[0].get("ids_sha256") != prefit_read[0].get("ids_sha256")
        or prefit_opened[0].get("query_labels_read") is not False
        or prefit_read[0].get("query_labels_read") is not False):
        raise PilotError("Single prefix preflight label read does not match its durable QA receipt")
    if require_committed_freeze:
        require_committed("reports/simplification_v1/CLS_POS_RECENCY_V1/FREEZE.json")
        require_committed("reports/simplification_v1/CLS_POS_RECENCY_V1/PREFIT_QA.json")
    current_train = Path(official_root) / "data/raw/classification/Train.csv"
    require(sha256_file(current_train) == freeze["official_train_sha256"], "Official Train hash changed")
    parent_hashes = _parent_runner()._external_file_inventory(Path(official_root))
    require(parent_hashes == freeze["parent_input_file_hashes"], "A frozen parent source/input hash changed")
    require(git_in(Path(official_root), "rev-parse", "HEAD") == freeze["official_repository_commit"],
            "Official repository source commit changed")
    return freeze


def _load_local_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"Could not load lane module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _reservation_body(freeze: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "candidate_id": core.CANDIDATE_ID,
        "status": "FIT_CAP_RESERVED",
        "freeze_signature": freeze["freeze_signature"],
        "audit_source_commit": freeze["audit_source_commit"],
        "fit_ids": list(core.FIT_IDS),
        "predictor_fit_cap": FIT_CAP,
        "preprocessing_fit_cap": 0,
        "no_test_or_sample_access": True,
        "root_authorization_commit": ROOT_PLAN_COMMIT,
    }


def reserve_fits(official_root: Path) -> dict[str, Any]:
    freeze = verify_freeze(official_root, require_committed_freeze=True)
    reservation = _reservation_body(freeze)
    write_once(RESERVATION_PATH, reservation)
    current = events()
    reserve_events = [event for event in current if event.get("event") == "FIT_CAP_RESERVED"]
    expected_event = {"event": "FIT_CAP_RESERVED", "candidate_id": core.CANDIDATE_ID,
        "freeze_signature": freeze["freeze_signature"], "fit_ids": list(core.FIT_IDS),
        "fit_cap": FIT_CAP, "preprocessing_cap": 0}
    if not reserve_events:
        require(not any(e.get("event") in {"FIT_STARTED", "FIT_SAVED", "FIT_FAILED"} for e in current),
                "Cannot reserve after any fit event")
        append_event(expected_event)
    else:
        require(len(reserve_events) == 1 and all(reserve_events[0].get(k) == v for k, v in expected_event.items()),
                "Existing fit reservation differs from frozen allowance")
    return {"reservation": reservation, "ledger_events": len(events())}


def verify_reservation(freeze: Mapping[str, Any]) -> dict[str, Any]:
    require(RESERVATION_PATH.is_file(), "Committed fit reservation is required")
    require_committed("reports/simplification_v1/CLS_POS_RECENCY_V1/RESERVATION.json")
    reservation = read_json(RESERVATION_PATH)
    expected = _reservation_body(freeze)
    require(reservation == expected, "Fit reservation differs from exact frozen allowance")
    ev = events()
    reserved = [event for event in ev if event.get("event") == "FIT_CAP_RESERVED"]
    require(len(reserved) == 1 and all(reserved[0].get(k) == v for k, v in {
        "candidate_id": core.CANDIDATE_ID, "freeze_signature": freeze["freeze_signature"],
        "fit_ids": list(core.FIT_IDS), "fit_cap": FIT_CAP, "preprocessing_cap": 0,
    }.items()), "Ledger reservation does not match committed fit budget")
    return reservation


def _fit_paths(fit_id: str) -> tuple[Path, Path, Path]:
    require(fit_id in core.FIT_IDS, f"Unreserved fit ID: {fit_id}")
    return (RUN_ROOT / "units" / f"{fit_id}.npz",
            RUN_ROOT / "receipts" / f"{fit_id}.json",
            RUN_ROOT / "verifications" / f"{fit_id}.json")


def _resource_snapshot(minimum: int) -> dict[str, int]:
    available = int(psutil.virtual_memory().available)
    require(available >= minimum, f"Available RAM {available} is below required {minimum}")
    return {"available_bytes": available, "minimum_required_bytes": int(minimum)}


def _heavy_job_lock():
    parent = _parent_runner()
    stack = ExitStack()
    try:
        shared = stack.enter_context(parent._os_lock(Path(tempfile.gettempdir()) / "kasdad_single_heavy_job.lock"))
        local = stack.enter_context(parent._os_lock(RUN_ROOT / ".training.lock"))
        return stack, [shared, local]
    except BaseException:
        stack.close()
        raise


def _prepare_fit_inputs(official_root: Path, origin: int) -> dict[str, Any]:
    context = _parent_context(official_root, origin)
    target_free_checks = _validate_origin_context(official_root, origin, context)
    fit_rows_all = np.asarray(context["fit_rows"], dtype=np.int64)
    fit_ids_all = np.asarray(context["fit_ids"]).astype(str)
    fit_cycles_all = np.asarray(context["old_runner"]._read_identity_frame().iloc[fit_rows_all]["cycle_code"], dtype=np.int64)
    present = _read_selected_stream_presence(
        Path(official_root) / "data/raw/classification/Train.csv", fit_rows_all, fit_ids_all)
    rows = fit_rows_all[present]
    ids = fit_ids_all[present]
    cycles = fit_cycles_all[present]
    append_event({"event": "PREFIX_LABELS_OPENED", "candidate_id": core.CANDIDATE_ID,
        "origin": origin, "purpose": "fixed prefix-present fit labels",
        "rows": int(len(rows)), "rows_sha256": sha256_array(rows),
        "ids_sha256": sha256_array(ids), "query_labels_read": False,
        "test_or_sample_access": False})
    y = _read_selected_targets(Path(official_root) / "data/raw/classification/Train.csv", rows, ids)
    append_event({"event": "PREFIX_LABELS_READ", "candidate_id": core.CANDIDATE_ID,
        "origin": origin, "purpose": "fixed prefix-present fit labels",
        "rows": int(len(rows)), "rows_sha256": sha256_array(rows),
        "ids_sha256": sha256_array(ids), "target_sha256": sha256_array(y),
        "query_labels_read": False, "test_or_sample_access": False})
    base = core.recency_base_weights(cycles, origin)
    recency_weights = core.normalize_class_mass(base, y)
    shuffled_weights, permutation = core.shuffle_weights_within_class(recency_weights, y)
    prefit = read_json(PREFIT_QA_PATH)["prefix_weight_preflight"]["origins"][f"E{origin}"]
    require(prefit["target_sha256"] == sha256_array(y)
            and prefit["fit_rows_sha256"] == sha256_array(rows)
            and prefit["fit_ids_sha256"] == sha256_array(ids)
            and prefit["fit_cycles_sha256"] == sha256_array(cycles)
            and prefit["base_weight_sha256"] == sha256_array(base)
            and prefit["recency_weight_sha256"] == sha256_array(recency_weights)
            and prefit["shuffled_weight_sha256"] == sha256_array(shuffled_weights)
            and prefit["shuffle_permutation_sha256"] == sha256_array(permutation),
            f"E{origin} fit labels or weights differ from committed prefit QA")
    x = sparse.csr_matrix(context["x_fit"][present], dtype=np.float64)
    z = np.asarray(context["z0_fit"], dtype=np.float64)[present]
    require(len(y) == x.shape[0] == len(z) and len(np.unique(y)) == 2,
            f"E{origin} present-prefix model inputs are misaligned or single-class")
    return {
        "context": context,
        "source_checks": target_free_checks,
        "rows": rows,
        "ids": ids,
        "cycles": cycles,
        "y": y,
        "x": x,
        "z_bag": z,
        "base_weights": base,
        "recency_weights": recency_weights,
        "shuffled_weights": shuffled_weights,
        "permutation": permutation,
    }


def _fit_status(fit_id: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    fit_starts = [e for e in events() if e.get("event") == "FIT_STARTED" and e.get("fit_id") == fit_id]
    terminals = [e for e in events() if e.get("event") in {"FIT_SAVED", "FIT_FAILED"} and e.get("fit_id") == fit_id]
    require(len(fit_starts) <= 1 and len(terminals) <= 1, f"Fit ledger has duplicate events for {fit_id}")
    return (fit_starts[0] if fit_starts else None,
            terminals[0] if terminals else None)


def verify_fit_binding(freeze: Mapping[str, Any], fit_id: str) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    artifact_path, receipt_path, verification_path = _fit_paths(fit_id)
    require(artifact_path.is_file() and receipt_path.is_file() and verification_path.is_file(),
            f"Fit artifact, receipt, or independent verification is missing: {fit_id}")
    receipt = read_json(receipt_path)
    verified = read_json(verification_path)
    require(receipt.get("fit_id") == fit_id
            and receipt.get("freeze_signature") == freeze["freeze_signature"]
            and receipt.get("status") == "FIT_SAVED"
            and receipt.get("artifact_sha256") == sha256_file(artifact_path),
            f"Fit receipt lineage or artifact hash mismatch: {fit_id}")
    ver_sig = verified.get("verification_signature")
    ver_body = dict(verified); ver_body.pop("verification_signature", None)
    require(ver_sig == canonical_sha256(ver_body)
            and verified.get("status") == "PASS_INDEPENDENT_FIT_RECONSTRUCTION"
            and verified.get("fit_id") == fit_id
            and verified.get("receipt_sha256") == sha256_file(receipt_path),
            f"Independent fit verification binding failed: {fit_id}")
    with np.load(artifact_path, allow_pickle=False) as archive:
        require(archive.files == ["gamma"], f"Unexpected gamma artifact members: {fit_id}")
        gamma = np.asarray(archive["gamma"], dtype=np.float64).copy()
    require(gamma.ndim == 1 and np.isfinite(gamma).all()
            and sha256_array(gamma) == receipt.get("gamma_sha256"),
            f"Gamma array schema/hash mismatch: {fit_id}")
    start, terminal = _fit_status(fit_id)
    require(start is not None and terminal is not None and terminal.get("event") == "FIT_SAVED"
            and terminal.get("receipt_sha256") == sha256_file(receipt_path),
            f"Fit ledger does not prove one saved fit: {fit_id}")
    return gamma, receipt, verified


def fit_one(official_root: Path, fit_id: str) -> dict[str, Any]:
    require(fit_id in core.FIT_IDS, f"Fit ID not reserved: {fit_id}")
    freeze = verify_freeze(official_root, require_committed_freeze=True)
    verify_reservation(freeze)
    origin, arm = core.fit_id_parts(fit_id)
    start, terminal = _fit_status(fit_id)
    if terminal is not None:
        require(terminal.get("event") == "FIT_SAVED", f"Fit {fit_id} was consumed by a failure; no retry")
        if not _fit_paths(fit_id)[2].is_file():
            verifier = _load_local_module("_cls_pos_recency_v1_verifier", AUDIT_ROOT / "scripts/verify_cls_pos_recency_v1.py")
            report = verifier.verify_fit_artifact(official_root, fit_id, write_report=True)
            append_event({"event": "FIT_VERIFIED", "fit_id": fit_id,
                "verification_sha256": sha256_file(_fit_paths(fit_id)[2]),
                "status": report["status"], "optimizer_refits": 0,
                "query_labels_read": False, "resumed_saved_fit": True})
        gamma, receipt, _ = verify_fit_binding(freeze, fit_id)
        return {"fit_id": fit_id, "status": "ALREADY_SAVED_NO_REFIT",
                "gamma_sha256": receipt["gamma_sha256"], "features": int(len(gamma))}
    require(start is None, f"Fit {fit_id} has a consumed start without a terminal event; no retry")
    ev = events()
    require(not any(event.get("event") in {"FIT_FAILED", "FIT_VERIFICATION_FAILED"} for event in ev),
            "A previous fit or verification failed; no further fit is permitted")
    require(len([event for event in ev if event.get("event") == "FIT_STARTED"]) < FIT_CAP,
            "All four reserved fit slots are already consumed")
    artifact_path, receipt_path, verification_path = _fit_paths(fit_id)
    require(not artifact_path.exists() and not receipt_path.exists() and not verification_path.exists(),
            f"Fit outputs exist without a matching saved ledger; fail closed: {fit_id}")

    stack, locks = _heavy_job_lock()
    try:
        with stack:
            freeze = verify_freeze(official_root, require_committed_freeze=True)
            verify_reservation(freeze)
            current_start, current_terminal = _fit_status(fit_id)
            require(current_start is None and current_terminal is None,
                    f"Fit state changed before acquiring its one-time start: {fit_id}")
            parent = _parent_runner()
            current_inputs = parent._external_file_inventory(Path(official_root))
            require(current_inputs == freeze["parent_input_file_hashes"], "Frozen external inputs changed before fit")
            prepared = _prepare_fit_inputs(official_root, origin)
            weights = prepared["recency_weights"] if arm == "RECENCY" else prepared["shuffled_weights"]
            permutation = prepared["permutation"] if arm == "SHUFFLED_WITHIN_CLASS" else np.arange(len(weights), dtype=np.int64)
            diagnostics = core.weight_diagnostics(weights, prepared["y"])
            for label, stats in diagnostics["classes"].items():
                require(stats["ess_ratio"] >= 0.50, f"Fit ESS below frozen floor for {fit_id}, class {label}")
            resource_before = _resource_snapshot(MIN_FIT_AVAILABLE)
            start_event = {
                "event": "FIT_STARTED", "fit_id": fit_id, "origin": origin, "arm": arm,
                "fit_consumed_at_start": True, "fit_cap": FIT_CAP,
                "freeze_signature": freeze["freeze_signature"],
                "audit_source_commit": freeze["audit_source_commit"],
                "reservation_sha256": sha256_file(RESERVATION_PATH),
                "fit_rows": int(len(prepared["rows"])),
                "fit_rows_sha256": sha256_array(prepared["rows"]),
                "fit_ids_sha256": sha256_array(prepared["ids"]),
                "fit_cycles_sha256": sha256_array(prepared["cycles"]),
                "fit_target_sha256": sha256_array(prepared["y"]),
                "fit_position_sha256": sha256_csr(prepared["x"]),
                "z_bag_sha256": sha256_array(prepared["z_bag"]),
                "base_weight_sha256": sha256_array(prepared["base_weights"]),
                "weight_sha256": sha256_array(weights),
                "weight_permutation_sha256": sha256_array(permutation),
                "resource_before": resource_before,
                "locks": locks,
                "prefix_only_labels": True,
                "query_labels_read": False,
                "test_or_sample_access": False,
            }
            append_event(start_event)
            try:
                with threadpool_limits(limits=2):
                    gamma, optimizer = core.fit_weighted_gamma(
                        prepared["x"], prepared["z_bag"], prepared["y"], weights)
                available_after = int(psutil.virtual_memory().available)
                require(available_after >= MIN_FIT_RESERVE,
                        f"Post-fit RAM {available_after} is below the 512 MiB reserve")
                require(_source_hashes() == freeze["source_hashes"], "Frozen lane source changed during the fit")
                require(parent._external_file_inventory(Path(official_root)) == current_inputs,
                        "Frozen parent source/input changed during the fit")
                require(not artifact_path.exists() and not receipt_path.exists(),
                        f"Refusing to overwrite existing fit output: {fit_id}")
                atomic_npz(artifact_path, gamma=gamma)
                receipt = {
                    "schema_version": 1,
                    "candidate_id": core.CANDIDATE_ID,
                    "status": "FIT_SAVED",
                    "fit_id": fit_id,
                    "origin": origin,
                    "arm": arm,
                    "freeze_signature": freeze["freeze_signature"],
                    "audit_source_commit": freeze["audit_source_commit"],
                    "reservation_sha256": sha256_file(RESERVATION_PATH),
                    "fit_rows": int(len(prepared["rows"])),
                    "fit_present_rows": int(len(prepared["rows"])),
                    "fit_rows_sha256": sha256_array(prepared["rows"]),
                    "fit_ids_sha256": sha256_array(prepared["ids"]),
                    "fit_cycles_sha256": sha256_array(prepared["cycles"]),
                    "fit_target_sha256": sha256_array(prepared["y"]),
                    "positive_count": int(prepared["y"].sum()),
                    "negative_count": int(len(prepared["y"]) - prepared["y"].sum()),
                    "fit_position_sha256": sha256_csr(prepared["x"]),
                    "z_bag_sha256": sha256_array(prepared["z_bag"]),
                    "gamma_original_sha256": sha256_array(prepared["context"]["gamma_original"]),
                    "base_weight_sha256": sha256_array(prepared["base_weights"]),
                    "weight_sha256": sha256_array(weights),
                    "weight_permutation_sha256": sha256_array(permutation),
                    "weight_diagnostics": diagnostics,
                    "shuffle_seed": core.SHUFFLE_SEED if arm == "SHUFFLED_WITHIN_CLASS" else None,
                    "optimizer": optimizer,
                    "gamma_sha256": sha256_array(gamma),
                    "artifact_path": artifact_path.relative_to(AUDIT_ROOT).as_posix(),
                    "artifact_sha256": sha256_file(artifact_path),
                    "resource_before": resource_before,
                    "resource_after_available_bytes": available_after,
                    "max_threads": 2,
                    "attempt_number": 1,
                    "retry_allowed": False,
                    "preprocessing_fits": 0,
                    "no_test_or_sample_access": True,
                    "no_query_labels_in_fit": True,
                }
                write_once(receipt_path, receipt)
                append_event({"event": "FIT_SAVED", "fit_id": fit_id,
                    "freeze_signature": freeze["freeze_signature"],
                    "gamma_sha256": receipt["gamma_sha256"],
                    "artifact_sha256": receipt["artifact_sha256"],
                    "receipt_sha256": sha256_file(receipt_path), "retry_allowed": False})
            except BaseException as exc:
                append_event({"event": "FIT_FAILED", "fit_id": fit_id,
                    "freeze_signature": freeze["freeze_signature"],
                    "error_type": type(exc).__name__, "error": str(exc),
                    "retry_allowed": False})
                raise
            verifier = _load_local_module("_cls_pos_recency_v1_verifier", AUDIT_ROOT / "scripts/verify_cls_pos_recency_v1.py")
            try:
                report = verifier.verify_fit_artifact(official_root, fit_id, write_report=True)
                append_event({"event": "FIT_VERIFIED", "fit_id": fit_id,
                    "verification_sha256": sha256_file(verification_path),
                    "status": report["status"], "optimizer_refits": 0,
                    "query_labels_read": False})
            except BaseException as exc:
                append_event({"event": "FIT_VERIFICATION_FAILED", "fit_id": fit_id,
                    "error_type": type(exc).__name__, "error": str(exc), "retry_allowed": False})
                raise
            return {"fit_id": fit_id, "status": "FIT_SAVED_AND_VERIFIED",
                    "gamma_sha256": receipt["gamma_sha256"],
                    "fit_rows": receipt["fit_rows"], "weight_diagnostics": diagnostics}
    finally:
        # ``ExitStack`` releases both OS advisory locks. Never delete their marker files.
        stack.close()


def run_all_fits(official_root: Path) -> list[dict[str, Any]]:
    results = []
    for fit_id in core.FIT_IDS:
        result = fit_one(official_root, fit_id)
        results.append(result)
        if result.get("status") not in {"FIT_SAVED_AND_VERIFIED", "ALREADY_SAVED_NO_REFIT"}:
            raise PilotError(f"Unexpected fit state; stopping run-all at {fit_id}")
        if any(e.get("event") in {"FIT_FAILED", "FIT_VERIFICATION_FAILED"} for e in events()):
            raise PilotError(f"Failed fit/verification in ledger; stopping before next fit")
    return results


def build_predictions(official_root: Path) -> dict[str, Any]:
    freeze = verify_freeze(official_root, require_committed_freeze=True)
    verify_reservation(freeze)
    ev = events()
    starts = [e for e in ev if e.get("event") == "FIT_STARTED"]
    saved = [e for e in ev if e.get("event") == "FIT_SAVED"]
    verified = [e for e in ev if e.get("event") == "FIT_VERIFIED"]
    require([e.get("fit_id") for e in starts] == list(core.FIT_IDS)
            and [e.get("fit_id") for e in saved] == list(core.FIT_IDS)
            and [e.get("fit_id") for e in verified] == list(core.FIT_IDS)
            and not any(e.get("event") in {"FIT_FAILED", "FIT_VERIFICATION_FAILED"} for e in ev),
            "Exactly four fits must be started, saved and verified before prediction build")
    require(not any(e.get("event") in {"QUERY_LABELS_OPENED", "SCORE_STARTED", "SCORE_SAVED"} for e in ev),
            "Predictions must be built before query labels are opened")
    parent = _parent_runner()
    current_inputs = parent._external_file_inventory(Path(official_root))
    require(current_inputs == freeze["parent_input_file_hashes"], "A frozen parent source/input hash changed")
    stack, locks = _heavy_job_lock()
    try:
        with stack:
            _resource_snapshot(MIN_FIT_AVAILABLE)
            outputs: dict[str, Any] = {}
            for origin in core.ORIGINS:
                context = _parent_context(official_root, origin)
                _validate_origin_context(official_root, origin, context)
                present = np.asarray(context["present"], dtype=bool)
                p_a02 = np.asarray(context["p_a02"], dtype=np.float64)
                p_pos_original = np.asarray(context["p_pos_original"], dtype=np.float64)
                p_champion = core.route_probabilities(present, p_a02, p_pos_original)
                payload: dict[str, np.ndarray] = {
                    "origin": np.asarray([origin], dtype=np.int64),
                    "row_positions": np.asarray(context["score_rows"], dtype=np.int64),
                    "record_ids": np.asarray(context["score_ids"]).astype(str),
                    "cycles": np.asarray(context["cycles"], dtype=np.int64),
                    "present": present,
                    "ef1": np.asarray(context["entry_friction"], dtype=np.int8),
                    "text_load_group": np.asarray(context["text_load_group"]).astype(str),
                    "p_a02": p_a02,
                    "p_pos_original": p_pos_original,
                    "p_champion": p_champion,
                    "label_champion": core.decisions(p_champion, present),
                }
                for arm in core.ARMS:
                    fit_id = f"CLS_POS_RECENCY_V1_E{origin}_{arm}"
                    gamma, receipt, _ = verify_fit_binding(freeze, fit_id)
                    p_pos = core.offset_probability(context["x_score"], context["z0_score"], gamma)
                    probability = core.route_probabilities(present, p_a02, p_pos)
                    payload[f"p_{arm.lower()}"] = probability
                    payload[f"label_{arm.lower()}"] = core.decisions(probability, present)
                    payload[f"p_pos_{arm.lower()}"] = p_pos
                    payload[f"gamma_sha256_{arm.lower()}"] = np.asarray([receipt["gamma_sha256"]])
                output = RUN_ROOT / f"E{origin}" / f"predictions_E{origin}.npz"
                atomic_npz(output, **payload)
                present_count = int(present.sum())
                outputs[f"E{origin}"] = {
                    "path": output.relative_to(AUDIT_ROOT).as_posix(),
                    "sha256": sha256_file(output),
                    "rows": int(len(present)),
                    "present_rows": present_count,
                    "missing_rows": int(len(present) - present_count),
                    "ordered_ids_sha256": sha256_array(payload["record_ids"]),
                    "row_positions_sha256": sha256_array(payload["row_positions"]),
                    "cycle_codes_sha256": sha256_array(payload["cycles"]),
                    "matrix_manifest_sha256": context["manifest_sha256"],
                }
                del context, payload
            require(outputs["E67"]["rows"] == 20903 and outputs["E75"]["rows"] == 8491,
                    "Candidate prediction row counts differ from fixed windows")
            fit_binding = {}
            for fit_id in core.FIT_IDS:
                _, receipt, verification = verify_fit_binding(freeze, fit_id)
                fit_binding[fit_id] = {
                    "receipt_sha256": sha256_file(_fit_paths(fit_id)[1]),
                    "verification_sha256": sha256_file(_fit_paths(fit_id)[2]),
                    "gamma_sha256": receipt["gamma_sha256"],
                    "verification_status": verification["status"],
                }
            ready = {
                "schema_version": 1,
                "candidate_id": core.CANDIDATE_ID,
                "freeze_signature": freeze["freeze_signature"],
                "source_commit": freeze["audit_source_commit"],
                "official_train_sha256": freeze["official_train_sha256"],
                "fit_ids": list(core.FIT_IDS),
                "fit_bindings": fit_binding,
                "predictions": outputs,
                "all_four_fit_receipts_durable": True,
                "all_four_fit_verifications_pass": True,
                "query_labels_read": False,
                "test_or_sample_access": False,
                "kaggle_access": False,
            }
            ready["ready_signature"] = canonical_sha256(ready)
            write_once(READY_PATH, ready)
            append_event({"event": "PREDICTIONS_READY", "candidate_id": core.CANDIDATE_ID,
                "freeze_signature": freeze["freeze_signature"],
                "ready_sha256": sha256_file(READY_PATH),
                "prediction_hashes": {k: v["sha256"] for k, v in outputs.items()},
                "query_labels_read": False, "test_or_sample_access": False})
            return ready
    finally:
        stack.close()


def verify_committed_prelabel() -> dict[str, Any]:
    require_committed("reports/simplification_v1/CLS_POS_RECENCY_V1/PREDICTIONS_READY.json")
    require_committed("reports/simplification_v1/CLS_POS_RECENCY_V1/PRELABEL_VERIFICATION.json")
    ready = read_json(READY_PATH)
    ready_sig = ready.get("ready_signature")
    ready_body = dict(ready); ready_body.pop("ready_signature", None)
    require(ready_sig == canonical_sha256(ready_body) and ready.get("query_labels_read") is False
            and ready.get("test_or_sample_access") is False,
            "Committed target-free readiness signature/boundary failed")
    prelabel = read_json(PRELABEL_PATH)
    signature = prelabel.get("verification_signature")
    body = dict(prelabel); body.pop("verification_signature", None)
    require(signature == canonical_sha256(body) and prelabel.get("status") == "PASS_TARGET_FREE_PRELABEL"
            and prelabel.get("query_labels_read") is False and prelabel.get("test_or_sample_access") is False
            and prelabel.get("prediction_readiness_sha256") == sha256_file(READY_PATH),
            "Committed target-free prelabel report failed signature or readiness binding")
    return prelabel


def verify_prelabel(official_root: Path) -> dict[str, Any]:
    verifier = _load_local_module("_cls_pos_recency_v1_verifier", AUDIT_ROOT / "scripts/verify_cls_pos_recency_v1.py")
    report = verifier.verify_prelabel_outputs(official_root, write_report=True)
    require(report.get("status") == "PASS_TARGET_FREE_PRELABEL"
            and report.get("query_labels_read") is False,
            "Independent target-free prelabel failed")
    append_event({"event": "PRELABEL_VERIFIED", "candidate_id": core.CANDIDATE_ID,
        "freeze_signature": report["freeze_signature"],
        "prelabel_sha256": sha256_file(PRELABEL_PATH),
        "prediction_readiness_sha256": report["prediction_readiness_sha256"],
        "query_labels_read": False, "test_or_sample_access": False})
    return report


def _binary_metrics(y: Sequence[int], probability: Sequence[float], decision: Sequence[int]) -> dict[str, Any]:
    target = np.asarray(y, dtype=np.int8).reshape(-1)
    p = np.asarray(probability, dtype=np.float64).reshape(-1)
    pred = np.asarray(decision, dtype=np.int8).reshape(-1)
    require(len(target) > 0 and len(target) == len(p) == len(pred)
            and np.isin(target, (0, 1)).all() and np.isin(pred, (0, 1)).all()
            and np.isfinite(p).all() and np.all((p >= 0) & (p <= 1)),
            "Scoring metric inputs invalid")
    tn = int(np.count_nonzero((target == 0) & (pred == 0)))
    fp = int(np.count_nonzero((target == 0) & (pred == 1)))
    fn = int(np.count_nonzero((target == 1) & (pred == 0)))
    tp = int(np.count_nonzero((target == 1) & (pred == 1)))
    both = np.unique(target).size == 2
    clipped = np.clip(p, np.finfo(np.float64).eps, 1.0 - np.finfo(np.float64).eps)
    return {
        "n": int(len(target)), "positive": int(target.sum()), "negative": int(len(target) - target.sum()),
        "macro_f1": float(f1_score(target, pred, average="macro", labels=[0, 1], zero_division=0)),
        "positive_precision": float(precision_score(target, pred, pos_label=1, zero_division=0)),
        "positive_recall": float(recall_score(target, pred, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, p)) if both else None,
        "average_precision": float(average_precision_score(target, p)) if both else None,
        "log_loss": float(log_loss(target, clipped, labels=[0, 1])),
        "brier": float(np.mean((p - target) ** 2)),
        "confusion_matrix_labels_0_1": [[tn, fp], [fn, tp]],
    }


def _metric_comparison(
    y: Sequence[int], p_reference: Sequence[float], p_candidate: Sequence[float],
    d_reference: Sequence[int], d_candidate: Sequence[int],
) -> dict[str, Any]:
    reference = _binary_metrics(y, p_reference, d_reference)
    candidate = _binary_metrics(y, p_candidate, d_candidate)
    return {
        "support": {"n": candidate["n"], "positive": candidate["positive"], "negative": candidate["negative"]},
        "reference": reference, "candidate": candidate,
        "gain_macro_f1": candidate["macro_f1"] - reference["macro_f1"],
        "gain_positive_recall": candidate["positive_recall"] - reference["positive_recall"],
        "gain_positive_precision": candidate["positive_precision"] - reference["positive_precision"],
        "gain_roc_auc": None if candidate["roc_auc"] is None or reference["roc_auc"] is None else candidate["roc_auc"] - reference["roc_auc"],
        "gain_average_precision": None if candidate["average_precision"] is None or reference["average_precision"] is None else candidate["average_precision"] - reference["average_precision"],
        "gain_log_loss": reference["log_loss"] - candidate["log_loss"],
        "gain_brier": reference["brier"] - candidate["brier"],
    }


def _compare_recursive(expected: Any, observed: Any, path: str = "", tol: float = 5e-12) -> list[dict[str, Any]]:
    differences: list[dict[str, Any]] = []
    if isinstance(expected, Mapping) and isinstance(observed, Mapping):
        if set(expected) != set(observed):
            return [{"path": path, "expected_keys": sorted(expected), "observed_keys": sorted(observed)}]
        for key in sorted(expected):
            differences.extend(_compare_recursive(expected[key], observed[key], f"{path}/{key}", tol))
    elif expected is None or observed is None:
        if expected is not observed:
            differences.append({"path": path, "expected": expected, "observed": observed})
    elif isinstance(expected, (int, np.integer)) and isinstance(observed, (int, np.integer)):
        if int(expected) != int(observed):
            differences.append({"path": path, "expected": int(expected), "observed": int(observed)})
    elif isinstance(expected, (float, int, np.floating, np.integer)) and isinstance(observed, (float, int, np.floating, np.integer)):
        delta = abs(float(expected) - float(observed))
        if not np.isfinite(delta) or delta > tol:
            differences.append({"path": path, "expected": float(expected), "observed": float(observed), "abs_diff": delta})
    elif expected != observed:
        differences.append({"path": path, "expected": str(expected), "observed": str(observed)})
    return differences


def score_metrics(official_root: Path) -> dict[str, Any]:
    freeze = verify_freeze(official_root, require_committed_freeze=True)
    prelabel = verify_committed_prelabel()
    if SCORE_PATH.exists() or METRIC_PATH.exists() or SCORE_VERIFY_PATH.exists():
        raise PilotError("Score artifacts already exist; no second query scoring pass is allowed")
    ev = events()
    if any(e.get("event") in {"QUERY_LABELS_OPENED", "SCORE_STARTED", "SCORE_SAVED", "SCORE_FAILED"} for e in ev):
        raise PilotError("Scoring ledger already has a query-label or score event; no retry")
    if not all(e.get("status") == "PASS_INDEPENDENT_FIT_RECONSTRUCTION"
               for e in (read_json(_fit_paths(fit_id)[2]) for fit_id in core.FIT_IDS)):
        raise PilotError("Four independent fit verifications must pass before scoring")
    bundles = {origin: {} for origin in core.ORIGINS}
    for origin in core.ORIGINS:
        path = RUN_ROOT / f"E{origin}" / f"predictions_E{origin}.npz"
        with np.load(path, allow_pickle=False) as archive:
            bundles[origin] = {key: archive[key].copy() for key in archive.files}
    rows = np.asarray(bundles[67]["row_positions"], dtype=np.int64)
    ids = np.asarray(bundles[67]["record_ids"]).astype(str)
    cycles = np.asarray(bundles[67]["cycles"], dtype=np.int64)
    require(len(rows) == len(np.unique(rows)) == 20903
            and len(ids) == len(np.unique(ids)) == 20903,
            "E67 query pool is not the exact unique primary row set")
    e75 = bundles[75]
    mask75 = (cycles >= 76) & (cycles <= 79)
    require(int(mask75.sum()) == 8491
            and np.array_equal(np.asarray(e75["row_positions"], dtype=np.int64), rows[mask75])
            and np.array_equal(np.asarray(e75["record_ids"]).astype(str), ids[mask75]),
            "E75 rows must be the exact E67 cycle 76-79 subset")
    score_resource_start = _resource_snapshot(MIN_SCORE_AVAILABLE)
    query_open = {"event": "QUERY_LABELS_OPENED", "candidate_id": core.CANDIDATE_ID,
        "freeze_signature": freeze["freeze_signature"], "prelabel_sha256": sha256_file(PRELABEL_PATH),
        "rows": int(len(rows)), "rows_sha256": sha256_array(rows), "ids_sha256": sha256_array(ids),
        "purpose": "single Train temporal metric evaluation on unique E67 rows",
        "test_or_sample_access": False}
    append_event({"event": "SCORE_STARTED", "candidate_id": core.CANDIDATE_ID,
        "freeze_signature": freeze["freeze_signature"], "prelabel_sha256": sha256_file(PRELABEL_PATH),
        "query_rows": int(len(rows)), "query_rows_sha256": sha256_array(rows),
        "resource_before": score_resource_start, "retry_allowed": False})
    append_event(query_open)
    try:
        y = _read_selected_targets(Path(official_root) / "data/raw/classification/Train.csv", rows, ids)
        append_event({"event": "QUERY_LABELS_READ", "candidate_id": core.CANDIDATE_ID,
            "freeze_signature": freeze["freeze_signature"], "rows": int(len(y)),
            "rows_sha256": sha256_array(rows), "ids_sha256": sha256_array(ids),
            "target_sha256": sha256_array(y), "query_label_reads": 1,
            "test_or_sample_access": False})
        ymap = {int(row): int(label) for row, label in zip(rows.tolist(), y.tolist())}

        def with_target(bundle: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
            local_rows = np.asarray(bundle["row_positions"], dtype=np.int64)
            return {**bundle, "y": np.asarray([ymap[int(row)] for row in local_rows], dtype=np.int8)}

        views = {"E67": with_target(bundles[67]), "E75": with_target(bundles[75])}
        e67 = views["E67"]
        sel67 = (e67["cycles"] >= 68) & (e67["cycles"] <= 75)
        e75_primary = views["E75"]
        sel75 = (e75_primary["cycles"] >= 76) & (e75_primary["cycles"] <= 79)
        unique: dict[str, np.ndarray] = {}
        for key, value in e67.items():
            if isinstance(value, np.ndarray) and value.ndim >= 1 and len(value) == len(e67["y"]):
                unique[key] = np.concatenate([value[sel67], e75_primary[key][sel75]])
        require(len(unique["row_positions"]) == len(np.unique(unique["row_positions"])) == 20903
                and len(unique["record_ids"]) == len(np.unique(unique["record_ids"])) == 20903,
                "Unique primary pool is not the E67 68-75 plus E75 76-79 non-overlap")
        views["UNIQUE_PRIMARY"] = unique
        metric_checks: list[dict[str, Any]] = []

        def checked_compare(metric_id: str, y_value, p_ref, p_cand, d_ref, d_cand) -> dict[str, Any]:
            expected = _metric_comparison(y_value, p_ref, p_cand, d_ref, d_cand)
            verifier = importlib.import_module("scripts.verify_cls_pos_recency_v1")
            independent = verifier.independent_metric_comparison(y_value, p_ref, p_cand, d_ref, d_cand)
            check = verifier.compare_metrics(expected, independent)
            require(check["status"] == "PASS", f"Independent metric calculations differ for {metric_id}")
            metric_checks.append({"metric_id": metric_id, **check,
                "metric_digest": canonical_sha256(expected)})
            return expected

        group_results: dict[str, Any] = {}
        legacy_harm_ok = True
        exploratory_harm_ok = True
        for view_name, block in views.items():
            if view_name == "UNIQUE_PRIMARY":
                local_views = {"UNIQUE_PRIMARY": np.ones(len(block["y"]), dtype=bool)}
            else:
                local_views = {name: ((block["cycles"] >= low) & (block["cycles"] <= high))
                               for name, (low, high) in SCORE_WINDOWS[view_name].items()}
            group_results[view_name] = {}
            for subview, view_mask in local_views.items():
                require(int(view_mask.sum()) > 0, f"Empty score view {view_name}/{subview}")
                b = {key: value[view_mask] for key, value in block.items()
                     if isinstance(value, np.ndarray) and value.ndim >= 1 and len(value) == len(view_mask)}
                metric_prefix = f"{view_name}/{subview}"
                recency_vs_champion = checked_compare(f"{metric_prefix}/recency_vs_champion", b["y"],
                    b["p_champion"], b["p_recency"], b["label_champion"], b["label_recency"])
                shuffled_vs_champion = checked_compare(f"{metric_prefix}/shuffled_vs_champion", b["y"],
                    b["p_champion"], b["p_shuffled_within_class"], b["label_champion"], b["label_shuffled_within_class"])
                recency_vs_shuffled = checked_compare(f"{metric_prefix}/recency_vs_shuffled", b["y"],
                    b["p_shuffled_within_class"], b["p_recency"], b["label_shuffled_within_class"], b["label_recency"])
                subgroups = []
                subgroup_vectors = (
                    ("availability", b["present"].astype(str)),
                    ("entry_friction", b["ef1"].astype(str)),
                    ("fit_prefix_text_load", b["text_load_group"].astype(str)),
                )
                for kind, values in subgroup_vectors:
                    for value in sorted(set(values.tolist())):
                        key = f"{kind}={value}"  # Assign before any metric ID uses it.
                        group_mask = values == value
                        support = importlib.import_module("scripts.verify_cls_pos_recency_v1")._supported(group_mask, b["y"])
                        if support["supported"]:
                            rc = checked_compare(f"{metric_prefix}/{key}/recency_vs_champion", b["y"][group_mask],
                                b["p_champion"][group_mask], b["p_recency"][group_mask],
                                b["label_champion"][group_mask], b["label_recency"][group_mask])
                            sc = checked_compare(f"{metric_prefix}/{key}/shuffled_vs_champion", b["y"][group_mask],
                                b["p_champion"][group_mask], b["p_shuffled_within_class"][group_mask],
                                b["label_champion"][group_mask], b["label_shuffled_within_class"][group_mask])
                            rs = checked_compare(f"{metric_prefix}/{key}/recency_vs_shuffled", b["y"][group_mask],
                                b["p_shuffled_within_class"][group_mask], b["p_recency"][group_mask],
                                b["label_shuffled_within_class"][group_mask], b["label_recency"][group_mask])
                            legacy_harm_ok &= (rc["gain_macro_f1"] >= -0.01
                                               and rc["gain_positive_recall"] >= -0.02)
                            exploratory_harm_ok &= (rc["gain_macro_f1"] >= -0.02
                                                    and rc["gain_positive_recall"] >= -0.03)
                            subgroups.append({"group": key, "support": support,
                                "recency_vs_champion": rc, "shuffled_vs_champion": sc,
                                "recency_vs_shuffled": rs})
                        else:
                            subgroups.append({"group": key, "support": support,
                                "recency_vs_champion": None, "shuffled_vs_champion": None,
                                "recency_vs_shuffled": None})
                group_results[view_name][subview] = {
                    "rows": int(len(b["y"])),
                    "recency_vs_champion": recency_vs_champion,
                    "shuffled_vs_champion": shuffled_vs_champion,
                    "recency_vs_shuffled": recency_vs_shuffled,
                    "subgroups": subgroups,
                }

            if view_name == "UNIQUE_PRIMARY":
                cycle_metrics: dict[str, Any] = {}
                for cycle in sorted(np.unique(block["cycles"]).tolist()):
                    cycle_mask = block["cycles"] == cycle
                    cycle_metrics[str(int(cycle))] = {
                        "rows": int(cycle_mask.sum()),
                        "recency_vs_champion": checked_compare(f"UNIQUE_PRIMARY/cycle={int(cycle)}/recency_vs_champion",
                            block["y"][cycle_mask], block["p_champion"][cycle_mask], block["p_recency"][cycle_mask],
                            block["label_champion"][cycle_mask], block["label_recency"][cycle_mask]),
                        "recency_vs_shuffled": checked_compare(f"UNIQUE_PRIMARY/cycle={int(cycle)}/recency_vs_shuffled",
                            block["y"][cycle_mask], block["p_shuffled_within_class"][cycle_mask], block["p_recency"][cycle_mask],
                            block["label_shuffled_within_class"][cycle_mask], block["label_recency"][cycle_mask]),
                    }
                group_results[view_name]["per_cycle"] = cycle_metrics

        primary = group_results["UNIQUE_PRIMARY"]["UNIQUE_PRIMARY"]
        h12 = group_results["E67"]["H12_E67"]
        w6 = group_results["E75"]["W6"]
        resource_after_bytes = int(psutil.virtual_memory().available)
        post_score_reserve_ok = resource_after_bytes >= MIN_SCORE_RESERVE
        score_resource = {
            "available_before_bytes": score_resource_start["available_bytes"],
            "minimum_before_bytes": MIN_SCORE_AVAILABLE,
            "available_after_metrics_bytes": resource_after_bytes,
            "minimum_after_bytes": MIN_SCORE_RESERVE,
            "post_score_reserve_pass": bool(post_score_reserve_ok),
        }
        append_event({"event": "SCORE_RESOURCE_SNAPSHOT", "candidate_id": core.CANDIDATE_ID,
            **score_resource})
        missing_exact = all(x.get("missing_route_exact") is True for x in prelabel["predictions"].values())
        independent_qa_pass = bool(metric_checks) and all(check["status"] == "PASS" for check in metric_checks)
        unique_gain = primary["recency_vs_champion"]["gain_macro_f1"]
        h12_gain = h12["recency_vs_champion"]["gain_macro_f1"]
        w6_gain = w6["recency_vs_champion"]["gain_macro_f1"]
        unique_control_gain = primary["recency_vs_shuffled"]["gain_macro_f1"]
        w6_control_gain = w6["recency_vs_shuffled"]["gain_macro_f1"]
        scientific = (unique_gain >= 0.0005 and h12_gain >= 0.0 and w6_gain >= 0.0
            and unique_control_gain >= 0.0002 and w6_control_gain >= 0.0
            and legacy_harm_ok and missing_exact and independent_qa_pass and post_score_reserve_ok)
        exploratory_signal = ((unique_gain > 0.0 and w6_gain >= -0.0005)
            or (w6_gain >= 0.0005 and unique_gain >= -0.0005))
        exploratory = (exploratory_signal and exploratory_harm_ok and missing_exact
                       and independent_qa_pass and post_score_reserve_ok)
        directions = [v["recency_vs_champion"]["gain_macro_f1"]
                      for k, v in group_results["UNIQUE_PRIMARY"]["per_cycle"].items()]
        direction_counts = {
            "positive": sum(value > 0.0 for value in directions),
            "negative": sum(value < 0.0 for value in directions),
            "zero": sum(value == 0.0 for value in directions),
            "cycles": len(directions),
        }
        result = {
            "schema_version": 1,
            "candidate_id": core.CANDIDATE_ID,
            "status": "SCORED_DIAGNOSTIC_ONLY",
            "freeze_signature": freeze["freeze_signature"],
            "source_commit": freeze["audit_source_commit"],
            "prelabel_sha256": sha256_file(PRELABEL_PATH),
            "query_rows": int(len(rows)),
            "query_rows_sha256": sha256_array(rows),
            "query_ids_sha256": sha256_array(ids),
            "query_target_sha256": sha256_array(y),
            "query_target_source": "one read of official Classification Train E67 temporal rows only",
            "query_label_reads": 1,
            "query_labels_opened": True,
            "test_or_sample_access": False,
            "kaggle_access": False,
            "resource_policy": score_resource,
            "views": group_results,
            "primary_unique_pool": {
                "rows": int(len(unique["y"])),
                "definition": "E67 cycles 68-75 plus E75 cycles 76-79; no duplicate row IDs",
                "recency_vs_champion": primary["recency_vs_champion"],
                "shuffled_vs_champion": primary["shuffled_vs_champion"],
                "recency_vs_shuffled": primary["recency_vs_shuffled"],
            },
            "gain_summary": {
                "unique_recency_vs_champion_macro_f1": unique_gain,
                "E67_H12_recency_vs_champion_macro_f1": h12_gain,
                "E75_W6_recency_vs_champion_macro_f1": w6_gain,
                "unique_recency_vs_shuffled_macro_f1": unique_control_gain,
                "E75_W6_recency_vs_shuffled_macro_f1": w6_control_gain,
                "cycle_direction_vs_champion": direction_counts,
            },
            "independent_metric_qa": {
                "status": "PASS" if independent_qa_pass else "FAIL",
                "comparison_count": len(metric_checks),
                "checks": metric_checks,
                "label_rows_read": 0,
            },
            "gates": {
                "scientific_qualified": bool(scientific),
                "exploratory_eligible": bool(exploratory),
                "exploratory_signal_ok": bool(exploratory_signal),
                "legacy_supported_subgroup_guard_ok": bool(legacy_harm_ok),
                "exploratory_supported_subgroup_guard_ok": bool(exploratory_harm_ok),
                "missing_route_exact": bool(missing_exact),
                "independent_metric_qa": bool(independent_qa_pass),
                "post_score_resource_reserve_ok": bool(post_score_reserve_ok),
                "thresholds": {
                    "unique_macro_f1_gain_min": 0.0005,
                    "E67_H12_macro_f1_gain_min": 0.0,
                    "E75_W6_macro_f1_gain_min": 0.0,
                    "unique_recency_minus_shuffled_min": 0.0002,
                    "E75_W6_recency_minus_shuffled_min": 0.0,
                    "legacy_supported_group_harm_max": {"macro_f1": 0.01, "positive_recall": 0.02},
                    "exploratory_supported_group_harm_max": {"macro_f1": 0.02, "positive_recall": 0.03},
                },
            },
            "limitations": [
                "Adaptive development windows overlap and are not untouched holdouts.",
                "The global missing threshold came from earlier development evidence, not a past-only fold threshold.",
                "Class-normalized weighting preserves class mass but does not guarantee an isolated temporal mechanism.",
                "No Test, SampleSubmission, Kaggle, or public score was accessed.",
            ],
        }
        verifier = importlib.import_module("scripts.verify_cls_pos_recency_v1")
        metric_body = verifier.build_metric_report(result["views"], metric_checks)
        result["metric_verification"] = metric_body
        result["score_signature"] = canonical_sha256(result)
        write_once(SCORE_PATH, result)
        metric_report = {"status": "PASS_INDEPENDENT_METRIC_RECALC",
            "score_sha256": sha256_file(SCORE_PATH), **metric_body}
        # The nested report signature binds its own body; the external report
        # needs a fresh signature over the score hash plus the nested contents.
        metric_report.pop("verification_signature", None)
        metric_report["verification_signature"] = canonical_sha256(metric_report)
        write_once(METRIC_PATH, metric_report)
        append_event({"event": "SCORE_SAVED", "candidate_id": core.CANDIDATE_ID,
            "freeze_signature": freeze["freeze_signature"],
            "score_sha256": sha256_file(SCORE_PATH), "metric_report_sha256": sha256_file(METRIC_PATH),
            "query_target_sha256": sha256_array(y), "retry_allowed": False})
        return result
    except BaseException as exc:
        append_event({"event": "SCORE_FAILED", "candidate_id": core.CANDIDATE_ID,
            "freeze_signature": freeze["freeze_signature"],
            "error_type": type(exc).__name__, "error": str(exc),
            "retry_allowed": False})
        raise


def verify_score(official_root: Path) -> dict[str, Any]:
    verifier = _load_local_module("_cls_pos_recency_v1_verifier", AUDIT_ROOT / "scripts/verify_cls_pos_recency_v1.py")
    result = verifier.verify_score_outputs(official_root, write_report=True)
    append_event({"event": "SCORE_VERIFIED", "candidate_id": core.CANDIDATE_ID,
        "score_sha256": result["score_sha256"],
        "score_verification_sha256": sha256_file(SCORE_VERIFY_PATH),
        "query_labels_read": False, "test_or_sample_access": False})
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one explicit stage of the fixed CLS_POS_RECENCY_V1 lifecycle.")
    parser.add_argument("command", choices=("prefit", "reserve", "fit-all", "predict",
        "prelabel", "score", "verify-score"), help="Frozen lifecycle stage to execute")
    parser.add_argument("--official-root", type=Path, default=AUDIT_ROOT.parent,
        help="KASDAD project root containing data/raw/classification/Train.csv")
    args = parser.parse_args(argv)
    root = args.official_root.resolve()
    actions = {
        "prefit": preflight_and_freeze,
        "reserve": reserve_fits,
        "fit-all": run_all_fits,
        "predict": build_predictions,
        "prelabel": verify_prelabel,
        "score": score_metrics,
        "verify-score": verify_score,
    }
    result = actions[args.command](root)
    print(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
