"""One-shot POS x entry-friction conditional interaction pilot.

Official lifecycle:
  prepare-source -> freeze -> reserve -> fit --fit-id (four separate calls)
  -> build -> prelabel -> score -> verify-score

No Test/SampleSubmission access is used. Query targets are available only to
``score`` after the committed target-free prelabel report passes verification.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import importlib
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
from scipy.special import expit
from threadpoolctl import threadpool_limits

AUDIT_ROOT = Path(__file__).resolve().parents[1]
if str(AUDIT_ROOT) not in sys.path:
    sys.path.insert(0, str(AUDIT_ROOT))
REPORT_ROOT = AUDIT_ROOT / "reports/simplification_v1/CLS_POS_EF_CONTEXT_V1"
RUN_ROOT = AUDIT_ROOT / "experiments/CLS_POS_EF_CONTEXT_V1"
FREEZE_PATH = REPORT_ROOT / "FREEZE.json"
SUPPORT_PATH = REPORT_ROOT / "SUPPORT_PREFLIGHT.json"
READY_PATH = REPORT_ROOT / "PREDICTIONS_READY.json"
PRELABEL_PATH = REPORT_ROOT / "PRELABEL_VERIFICATION.json"
SCORE_PATH = REPORT_ROOT / "SCORE.json"
METRIC_VERIFY_PATH = REPORT_ROOT / "METRIC_VERIFICATION.json"
PARENT_READY_REL = "experiments/CLS_POS_POISSON_BAG2_V1/PREDICTIONS_READY.json"
PARENT_READBACK_REL = "experiments/CLS_POS_POISSON_BAG2_V1/READBACK_VERIFICATION.json"
PARENT_FREEZE_REL = "reports/simplification_v1/CLS_POS_POISSON_BAG2_V1/PREFIT_FREEZE.json"
FIT_LABEL_COL = "artifact_signal"
FEATURE_COLUMNS = ["record_id", "cycle_code", "entry_friction", "signal_stream"]
MIN_AVAILABLE = 1024**3
MIN_RESERVE = 512 * 1024**2
ESTIMATED_PEAK = 512 * 1024**2


class PilotError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise PilotError(message)


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(value: Any) -> str:
    a = np.asarray(value)
    header = canonical_sha256({"shape": list(a.shape), "dtype": str(a.dtype)})
    h = hashlib.sha256(header.encode("ascii"))
    if a.dtype.kind in "OUS":
        h.update(json.dumps(a.astype(str).tolist(), ensure_ascii=False,
                            separators=(",", ":")).encode("utf-8"))
    else:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def sha256_csr(value: sparse.spmatrix) -> str:
    x = sparse.csr_matrix(value, dtype=np.float64, copy=True)
    x.sum_duplicates(); x.eliminate_zeros(); x.sort_indices()
    h = hashlib.sha256()
    h.update(canonical_json({"shape": list(x.shape), "dtype": str(x.dtype), "format": "csr"}))
    for block in (x.indptr, x.indices, x.data):
        h.update(np.ascontiguousarray(block).tobytes())
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_once(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if path.exists():
        require(path.read_text(encoding="utf-8") == payload,
                f"Refusing to overwrite non-identical frozen output: {path}")
        return
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(payload, encoding="utf-8", newline="\n")
    with tmp.open("r+b") as stream:
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def append_event(path: Path, event: Mapping[str, Any]) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(dict(event), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        stream.flush(); os.fsync(stream.fileno())


def events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_npz(path: Path, **arrays: Any) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    with tmp.open("r+b") as stream:
        os.fsync(stream.fileno())
    os.replace(tmp, path)


@contextmanager
def os_lock(path: Path):
    """Nonblocking OS advisory lock; file existence alone is not lock ownership."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+b")
    stream.seek(0, os.SEEK_END)
    if stream.tell() == 0:
        stream.write(b"\0"); stream.flush(); os.fsync(stream.fileno())
    stream.seek(0); acquired = False; unlock = None
    try:
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise PilotError(f"Shared heavy-job lock is currently held: {path}") from exc
            unlock = lambda: msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise PilotError(f"Shared heavy-job lock is currently held: {path}") from exc
            unlock = lambda: fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        acquired = True
        yield {"path": str(path), "mechanism": "OS nonblocking advisory lock", "acquired": True}
    finally:
        if acquired and unlock:
            try: unlock()
            except OSError: pass
        stream.close()


@contextmanager
def heavy_job_lock():
    with ExitStack() as stack:
        shared = Path(tempfile.gettempdir()) / "kasdad_single_heavy_job.lock"
        stack.enter_context(os_lock(shared))
        stack.enter_context(os_lock(RUN_ROOT / ".training.lock"))
        yield


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True,
                                   stderr=subprocess.STDOUT).strip()


def repo_head(root: Path) -> str:
    return git(root, "rev-parse", "HEAD")


def audit_sources() -> list[str]:
    return ["src/models/cls_pos_ef_context_v1.py", "scripts/cls_pos_ef_context_v1.py",
            "scripts/verify_cls_pos_ef_context_v1.py", "scripts/score_cls_pos_ef_context_v1.py",
            "tests/test_cls_pos_ef_context_v1.py",
            "reports/simplification_v1/CLS_POS_EF_CONTEXT_V1/PROTOCOL.md"]


def load_parent_module(repository: Path):
    root = Path(repository).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    importlib.invalidate_caches()
    module = importlib.import_module("scripts.cls_pos_poisson_bag2_v1")
    require(Path(module.AUDIT).resolve() == AUDIT_ROOT,
            "Loaded Poisson source contract from a different audit repository")
    module.verify_prefit_freeze(root)
    return module


def read_features(train_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(Path(train_path), usecols=FEATURE_COLUMNS,
                        dtype={"record_id": "string", "cycle_code": "int64",
                               "entry_friction": "int8", "signal_stream": "string"},
                        keep_default_na=False)
    require("artifact_signal" not in frame.columns, "Feature loader must never select query labels")
    require(len(frame) == 50608 and frame["record_id"].is_unique,
            "Official Train feature rows/IDs differ from frozen source")
    ef = frame["entry_friction"].to_numpy(dtype=np.int8)
    require(np.isin(ef, (0, 1)).all(), "Official Train entry_friction must be exact binary 0/1")
    cyc = frame["cycle_code"].to_numpy(dtype=np.int64)
    require(np.all((cyc >= 47) & (cyc <= 79)), "Official Train cycles escape 47-79")
    return frame


def _load_poisson_bundle(repository: Path, origin: int, ready: Mapping[str, Any], readback: Mapping[str, Any]) -> dict[str, np.ndarray]:
    suffix = f"E{origin}"
    entry = ready["predictions"][suffix]
    path = AUDIT_ROOT / entry["path"]
    expected = entry["sha256"]
    require(expected == readback["predictions"][suffix]["bundle_sha256"] == sha256_file(path),
            f"Poisson original-only cache hash differs for {suffix}")
    # Explicit allowlist only: no Poisson-replica outputs or label member is read.
    allow = ("row_positions", "record_ids", "cycles", "present_mask", "text_load_group",
             "entry_friction", "p_a02", "p_pos_original", "p_champion", "label_champion")
    with np.load(path, allow_pickle=False) as archive:
        require(set(allow).issubset(set(archive.files)), f"Original-only source cache is incomplete for {suffix}")
        return {name: archive[name].copy() for name in allow}


def load_parent_origin(repository: Path, origin: int, frame: pd.DataFrame | None = None) -> dict[str, Any]:
    """Authoritative target-free loader used by producer and both verifiers."""
    from src.models import cls_pos_ef_context_v1 as core
    root = Path(repository).resolve()
    parent = load_parent_module(root)
    old_freeze = parent.verify_prefit_freeze(root)
    ready = read_json(AUDIT_ROOT / PARENT_READY_REL)
    readback = read_json(AUDIT_ROOT / PARENT_READBACK_REL)
    require(ready.get("candidate_id") == "CLS_POS_POISSON_BAG2_V1"
            and ready.get("freeze_signature") == old_freeze["freeze_signature"]
            and ready.get("query_labels_opened") is False
            and ready.get("test_or_sample_access") is False
            and ready.get("all_four_fit_receipts_durable") is True,
            "Poisson parent readiness is not a target-free original-only source")
    require(readback.get("status") == "PASS_INDEPENDENT_PREDICTION_READBACK"
            and readback.get("predictions_ready_sha256") == sha256_file(AUDIT_ROOT / PARENT_READY_REL),
            "Poisson original-only readback receipt failed")
    cache = _load_poisson_bundle(root, origin, ready, readback)
    ctx = parent._load_origin_context(root, origin, require_fit_labels=False)
    train_path = root / "data/raw/classification/Train.csv"
    if frame is None:
        frame = read_features(train_path)
    rows_fit = np.asarray(ctx["fit_rows"], dtype=np.int64)
    rows_query = np.asarray(ctx["score_rows"], dtype=np.int64)
    ids = frame["record_id"].astype(str).to_numpy()
    cycles = frame["cycle_code"].to_numpy(dtype=np.int64)
    ef = frame["entry_friction"].to_numpy(dtype=np.int8)
    present_all = frame["signal_stream"].str.strip().ne("").to_numpy(dtype=bool)
    require(np.array_equal(cycles[rows_query], np.asarray(ctx["cycles"], dtype=np.int64))
            and np.array_equal(ids[rows_query], np.asarray(ctx["score_ids"]).astype(str)),
            f"Official Train feature identity differs from parent source E{origin}")
    present = present_all[rows_query]
    require(np.array_equal(present, np.asarray(ctx["present"], dtype=bool)),
            f"Raw stream presence differs from frozen parent E{origin}")
    require(np.array_equal(cache["row_positions"].astype(np.int64), rows_query)
            and np.array_equal(cache["record_ids"].astype(str), ids[rows_query])
            and np.array_equal(cache["cycles"].astype(np.int64), cycles[rows_query])
            and np.array_equal(cache["present_mask"].astype(bool), present),
            f"Original-only parent cache rows/IDs/cycles/mask differ for E{origin}")
    p_a02 = np.asarray(ctx["p_a02"], dtype=np.float64)
    p_pos = np.asarray(ctx["p_pos_original"], dtype=np.float64)
    champion = core.route_candidate(p_a02, p_pos, present)
    champion_labels = core.apply_thresholds(champion, present)
    require(np.array_equal(p_a02, np.asarray(cache["p_a02"], dtype=np.float64)),
            f"A02 probability source changed for E{origin}")
    require(np.array_equal(p_pos, np.asarray(cache["p_pos_original"], dtype=np.float64)),
            f"Original POS gamma replay differs exactly from parent cache for E{origin}")
    require(np.array_equal(champion, np.asarray(cache["p_champion"], dtype=np.float64))
            and np.array_equal(champion_labels, np.asarray(cache["label_champion"], dtype=np.int8)),
            f"Full champion did not replay bit-exactly for E{origin}")
    state = core.context_state(ef[rows_fit], present_all[rows_fit])
    fit_shuffled, fit_moved = core.shuffle_context_within_cycle(
        ef[rows_fit], cycles[rows_fit], present_all[rows_fit], seed=42)
    query_shuffled, query_moved = core.shuffle_context_within_cycle(
        ef[rows_query], cycles[rows_query], present, seed=42)
    require(fit_moved >= 100 and query_moved >= 100,
            f"Shuffled control has fewer than 100 moved present assignments for E{origin}")
    x_fit = sparse.csr_matrix(ctx["x_fit"], dtype=np.float64)
    x_query = sparse.csr_matrix(ctx["x_score"], dtype=np.float64)
    z_pos_fit = np.asarray(ctx["z0_fit"] + x_fit @ np.asarray(ctx["gamma_original"], dtype=np.float64), dtype=np.float64).reshape(-1)
    z_pos_query = np.asarray(ctx["z0_score"] + x_query @ np.asarray(ctx["gamma_original"], dtype=np.float64), dtype=np.float64).reshape(-1)
    require(np.array_equal(expit(z_pos_query)[present], p_pos[present]),
            f"z_POS arithmetic differs from original POS prediction E{origin}")
    return {"origin": origin, "parent_freeze_signature": old_freeze["freeze_signature"],
            "parent_ready_sha256": sha256_file(AUDIT_ROOT / PARENT_READY_REL),
            "parent_readback_sha256": sha256_file(AUDIT_ROOT / PARENT_READBACK_REL),
            "parent_bundle_sha256": ready["predictions"][f"E{origin}"]["sha256"],
            "fit_rows": rows_fit, "fit_ids": ids[rows_fit], "fit_cycles": cycles[rows_fit],
            "fit_ef": ef[rows_fit], "fit_present": present_all[rows_fit],
            "fit_ef_shuffled": fit_shuffled, "fit_moved_present": fit_moved,
            "score_rows": rows_query, "score_ids": ids[rows_query], "cycles": cycles[rows_query],
            "ef": ef[rows_query], "present": present, "query_ef_shuffled": query_shuffled,
            "query_moved_present": query_moved,
            "text_load_group": np.asarray(cache["text_load_group"]).astype(str),
            "p_a02": p_a02, "p_pos_original": p_pos, "p_champion": champion,
            "label_champion": champion_labels, "x_fit": x_fit, "x_score": x_query,
            "z_pos_fit": z_pos_fit, "z_pos_score": z_pos_query,
            "context_state": state, "gamma_original": np.asarray(ctx["gamma_original"], dtype=np.float64),
            "matrix_manifest_sha256": ctx["manifest_sha256"],
            "fit_position_sha256": sha256_csr(x_fit), "score_position_sha256": sha256_csr(x_query),
            "entry_friction_cached": np.asarray(cache["entry_friction"]).astype(str)}


def build_support_preflight(repository: Path) -> dict[str, Any]:
    from src.models import cls_pos_ef_context_v1 as core
    root = Path(repository).resolve()
    train = root / "data/raw/classification/Train.csv"
    frame = read_features(train)
    origins = {}
    for origin in core.ORIGINS:
        p = load_parent_origin(root, origin, frame)
        origins[f"E{origin}"] = {"fit_rows": int(len(p["fit_rows"])), "query_rows": int(len(p["score_rows"])),
            "fit_present_rows": int(p["fit_present"].sum()), "query_present_rows": int(p["present"].sum()),
            "fit_present_ef_counts": p["context_state"]["ef_counts"],
            "mu": p["context_state"]["mu"], "scale": p["context_state"]["scale"],
            "fit_control_moved_present": int(p["fit_moved_present"]),
            "query_control_moved_present": int(p["query_moved_present"]),
            "matrix_manifest_sha256": p["matrix_manifest_sha256"],
            "fit_position_sha256": p["fit_position_sha256"],
            "score_position_sha256": p["score_position_sha256"],
            "champion_replay": "BIT_EXACT", "test_or_sample_access": False,
            "query_labels_read": False}
    return {"status": "PASS_SOURCE_AND_SUPPORT_PREFLIGHT", "train_rows": int(len(frame)),
            "feature_columns_only": FEATURE_COLUMNS, "target_column_selected": False,
            "input_sha256": sha256_file(train), "origins": origins}


def create_freeze(repository: Path) -> dict[str, Any]:
    from src.models import cls_pos_ef_context_v1 as core
    root = Path(repository).resolve()
    require(not git(AUDIT_ROOT, "status", "--porcelain"),
            "Audit checkout must be clean before creating immutable source freeze")
    source_commit = repo_head(AUDIT_ROOT)
    support = build_support_preflight(root)
    parent = load_parent_module(root)
    parent_freeze = parent.verify_prefit_freeze(root)
    input_paths: set[str] = {
        "data/raw/classification/Train.csv",
        "reports/optimization_v3/positional_text_pilot_v1/FIT_ACCOUNTING.json",
        "experiments/runs/optimization_v3_research_next_v1/POSITIONAL_TEXT_PILOT_V1/FREEZE.json",
        "reports/optimization_v3/positional_bag_offset_v1/RECIPE.json",
        "experiments/runs/optimization_v3_research_next_v1/POS_BAG_OFFSET_V1/FREEZE.json",
        "experiments/runs/optimization_v3_research_next_v1/POS_BAG_OFFSET_V1/PREDICTIONS_READY.json",
    }
    audit_artifact_paths={PARENT_FREEZE_REL,PARENT_READY_REL,PARENT_READBACK_REL}
    for origin in (67,75):
        suffix=f"E{origin}"
        audit_artifact_paths.add(f"experiments/CLS_POS_POISSON_BAG2_V1/{suffix}/predictions_{suffix}.npz")
    input_paths.update(parent_freeze["external_source_hashes"])
    input_paths.update(parent_freeze["input_file_hashes"])
    for origin in (67, 75):
        suffix = f"E{origin}"
        input_paths.update({f"experiments/runs/optimization_v3_research_next_v1/POS_BAG_OFFSET_V1/{suffix}/predictions_{suffix}.npz",
                            f"experiments/runs/optimization_v3_research_next_v1/POSITIONAL_TEXT_PILOT_V1/MATRIX_MANIFEST_{suffix}.json"})
        manifest = read_json(root / f"experiments/runs/optimization_v3_research_next_v1/POSITIONAL_TEXT_PILOT_V1/MATRIX_MANIFEST_{suffix}.json")
        for name in ("fit_base", "fit_position", "score_base", "score_position"):
            item = manifest["matrices"][name]
            path = Path(item["path"])
            if path.is_absolute():
                require(path.is_relative_to(root), "Frozen matrix path escapes official repository")
                input_paths.add(path.relative_to(root).as_posix())
            else:
                input_paths.add(path.as_posix())
        input_paths.update({
            f"experiments/runs/optimization_v3_research_next_v1/POSITIONAL_TEXT_PILOT_V1/{suffix}/predictions_{suffix}.npz",
            f"experiments/runs/optimization_v3_research_next_v1/POSITIONAL_TEXT_PILOT_V1/units/POSITIONAL_TEXT_PILOT_V1_{suffix}_BAG_LR.joblib",
            f"experiments/runs/optimization_v3_research_next_v1/POSITIONAL_TEXT_PILOT_V1/receipts/POSITIONAL_TEXT_PILOT_V1_{suffix}_BAG_LR.json",
            f"experiments/runs/optimization_v3_research_next_v1/POS_BAG_OFFSET_V1/units/POS_BAG_OFFSET_V1_{suffix}_REAL.npz",
            f"experiments/runs/optimization_v3_research_next_v1/POS_BAG_OFFSET_V1/receipts/POS_BAG_OFFSET_V1_{suffix}_REAL.json",
        })
    input_hashes = {rel: sha256_file(root / rel) for rel in sorted(input_paths)}
    require(all("Test.csv" not in rel and "SampleSubmission" not in rel for rel in input_hashes),
            "Test/SampleSubmission entered the frozen input boundary")
    audit_artifact_hashes={rel:sha256_file(AUDIT_ROOT/rel) for rel in sorted(audit_artifact_paths)}
    audit_hashes = {rel: sha256_file(AUDIT_ROOT / rel) for rel in audit_sources()}
    for rel, digest in audit_hashes.items():
        blob = subprocess.check_output(["git", "-C", str(AUDIT_ROOT), "show", f"{source_commit}:{rel}"])
        require(hashlib.sha256(blob).hexdigest() == digest, f"Source file is not committed at frozen HEAD: {rel}")
    body = {"schema_version": 1, "candidate_id": "CLS_POS_EF_CONTEXT_V1",
        "status": "FROZEN_BEFORE_OFFICIAL_FITS", "source_commit": source_commit,
        "official_repository_root": str(root), "official_source_commit": git(root, "rev-parse", "HEAD"),
        "parent_candidate": "CLS_POS_POISSON_BAG2_V1", "parent_freeze_signature": parent_freeze["freeze_signature"],
        "parent_ready_sha256": sha256_file(AUDIT_ROOT / PARENT_READY_REL),
        "parent_readback_sha256": sha256_file(AUDIT_ROOT / PARENT_READBACK_REL),
        "source_hashes": audit_hashes, "audit_artifact_hashes": audit_artifact_hashes,
        "input_hashes": input_hashes,
        "input_sha256": input_hashes["data/raw/classification/Train.csv"],
        "fit_ids": list(core.FIT_IDS),
        "recipe": {"origins": {"E67": {"fit_cycles": [47,67], "query_cycles": [68,79]},
                                "E75": {"fit_cycles": [47,75], "query_cycles": [76,79]}},
            "fit_population": "present signal_stream rows only", "minimum_present_per_EF_level": 200,
            "context": "q=(EF1-mu)/sqrt(mu*(1-mu)); mu and scale fit on present prefix only",
            "control": "EF assignment shuffled independently within cycle x presence strata for fit and query; seed=42; counts preserved; minimum 100 moved present assignments per population",
            "offset": "z_POS = z_BAG + X_POS @ gamma_original; no clipped-probability logit recovery",
            "objective": "sum logistic loss + 0.5 * ||gamma||^2; no intercept; zero initialization; C=1",
            "optimizer": {"method":"L-BFGS-B", "options":{"maxiter":2000,"ftol":1e-12,"gtol":1e-6,"maxls":50}},
            "present_blend": "0.5 * A02 + 0.5 * context POS probability", "present_threshold":0.50,
            "missing_route": "exact A02", "missing_threshold":0.53230976129477,
            "prediction_source": "full champion replays original-only Poisson cache bit-exactly before new construction",
            "query_label_boundary": "no query target reads before committed prelabel PASS"},
        "support_preflight": support, "fit_cap": 4,
        "fit_accounting": "4 exact starts; no retry after failed/interrupted optimizer; durable receipt may be resumed without optimizer",
        "data_boundary": {"test_or_sample_access": False, "query_labels_read": False}}
    body["freeze_signature"] = canonical_sha256(body)
    write_once(SUPPORT_PATH, support)
    write_once(FREEZE_PATH, body)
    return body


def verify_freeze(repository: Path) -> dict[str, Any]:
    root = Path(repository).resolve()
    require(FREEZE_PATH.is_file() and SUPPORT_PATH.is_file(), "Committed source freeze is missing")
    freeze = read_json(FREEZE_PATH)
    body = dict(freeze); signature = body.pop("freeze_signature", None)
    require(signature == canonical_sha256(body), "Freeze signature mismatch")
    require(freeze.get("candidate_id") == "CLS_POS_EF_CONTEXT_V1" and freeze.get("fit_cap") == 4
            and freeze.get("official_repository_root") == str(root), "Freeze identity/cap mismatch")
    head = repo_head(AUDIT_ROOT)
    require(git(AUDIT_ROOT, "merge-base", "--is-ancestor", freeze["source_commit"], head) == "",
            "Frozen code commit is not an ancestor of HEAD")
    for rel, expected in freeze["source_hashes"].items():
        require(sha256_file(AUDIT_ROOT / rel) == expected, f"Frozen audit source changed: {rel}")
        blob = subprocess.check_output(["git", "-C", str(AUDIT_ROOT), "show", f"{freeze['source_commit']}:{rel}"])
        require(hashlib.sha256(blob).hexdigest() == expected, f"Committed frozen source differs: {rel}")
    for rel, expected in freeze["audit_artifact_hashes"].items():
        require(sha256_file(AUDIT_ROOT / rel) == expected, f"Frozen audit artifact changed: {rel}")
    require(git(AUDIT_ROOT, "show", f"HEAD:{FREEZE_PATH.relative_to(AUDIT_ROOT).as_posix()}").encode("utf-8")
            == FREEZE_PATH.read_bytes(), "Freeze report must be committed before official fits/scoring")
    require(read_json(SUPPORT_PATH) == freeze["support_preflight"],
            "Committed support preflight differs from frozen support data")
    support_rel = SUPPORT_PATH.relative_to(AUDIT_ROOT).as_posix()
    require(git(AUDIT_ROOT, "show", f"HEAD:{support_rel}").encode("utf-8") == SUPPORT_PATH.read_bytes(),
            "Support preflight must be committed with the freeze")
    for rel, expected in freeze["input_hashes"].items():
        require(sha256_file(root / rel) == expected, f"Frozen official input changed: {rel}")
    require(all("Test.csv" not in rel and "SampleSubmission" not in rel for rel in freeze["input_hashes"]),
            "Frozen input inventory contains Test/SampleSubmission")
    parent = load_parent_module(root)
    require(parent.verify_prefit_freeze(root)["freeze_signature"] == freeze["parent_freeze_signature"],
            "Parent freeze changed")
    return freeze


def reserve(repository: Path) -> dict[str, Any]:
    freeze = verify_freeze(repository)
    ledger = RUN_ROOT / "fit_ledger.jsonl"
    existing = events(ledger)
    reservations = [e for e in existing if e.get("event") == "CARD_FITS_RESERVED"]
    expected = {"event": "CARD_FITS_RESERVED", "candidate_id": freeze["candidate_id"],
                "freeze_signature": freeze["freeze_signature"], "fit_ids": freeze["fit_ids"], "fit_cap": 4}
    if reservations:
        require(len(reservations) == 1 and reservations[0] == expected,
                "Existing reservation conflicts; no budget reset/re-reservation")
    else:
        require(not any(e.get("fit_id") for e in existing), "Fit ledger already contains unreserved fit events")
        append_event(ledger, expected)
    receipt_path = RUN_ROOT / "RESERVATION.json"
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        require(receipt.get("reservation") == expected, "Existing reservation receipt differs")
    else:
        receipt = {"reservation": expected, "reserved_at_ledger_sha256": sha256_file(ledger)}
        write_once(receipt_path, receipt)
    return receipt


def check_live_budget(freeze: Mapping[str, Any], *, allow_current: str | None = None) -> list[dict[str, Any]]:
    ledger = RUN_ROOT / "fit_ledger.jsonl"; ev = events(ledger)
    reservations = [e for e in ev if e.get("event") == "CARD_FITS_RESERVED"]
    expected = {"event": "CARD_FITS_RESERVED", "candidate_id": freeze["candidate_id"],
                "freeze_signature": freeze["freeze_signature"], "fit_ids": freeze["fit_ids"], "fit_cap": 4}
    require(reservations == [expected], "Exact cap-4 reservation is absent or inconsistent")
    starts = [e for e in ev if e.get("event") == "FIT_STARTED"]
    require(len(starts) <= 4 and len({e.get("fit_id") for e in starts}) == len(starts),
            "Fit starts exceed cap or duplicate a frozen ID")
    require(all(e.get("fit_id") in freeze["fit_ids"] for e in starts), "Unexpected fit ID in ledger")
    terminal = [e for e in ev if e.get("event") in {"FIT_SAVED", "FIT_FAILED"}]
    terminal_ids = [e.get("fit_id") for e in terminal]
    started_ids = [e.get("fit_id") for e in starts]
    require(len(set(terminal_ids)) == len(terminal_ids)
            and set(terminal_ids).issubset(set(started_ids)),
            "Fit terminal accounting is duplicated or has no matching start")
    pending = {e["fit_id"] for e in starts} - {e["fit_id"] for e in terminal}
    require(len(pending) <= 1 and (allow_current is None or pending in (set(), {allow_current})),
            "More than one fit is pending or pending fit differs from requested fit")
    return ev


def fit_paths(fit_id: str) -> tuple[Path, Path]:
    return RUN_ROOT / "units" / f"{fit_id}.npz", RUN_ROOT / "receipts" / f"{fit_id}.json"


def verify_gamma_bundle(freeze_signature: str, fit_id: str, artifact: Path,
                        receipt_path: Path) -> dict[str, Any]:
    require(artifact.is_file() and receipt_path.is_file(), f"Fit artifact/receipt absent: {fit_id}")
    receipt = read_json(receipt_path)
    require(receipt["fit_id"] == fit_id and receipt["freeze_signature"] == freeze_signature
            and receipt["artifact_sha256"] == sha256_file(artifact), f"Fit receipt lineage/hash mismatch: {fit_id}")
    with np.load(artifact, allow_pickle=False) as archive:
        require(archive.files == ["gamma"], f"Unexpected model artifact members: {fit_id}")
        gamma = np.asarray(archive["gamma"], dtype=np.float64).copy()
    require(sha256_array(gamma) == receipt["gamma_sha256"] and gamma.shape == (receipt["position_width"],)
            and np.isfinite(gamma).all(), f"Fit gamma content/width mismatch: {fit_id}")
    return {"gamma": gamma, "receipt": receipt, "receipt_sha256": sha256_file(receipt_path)}


def resume_saved_gamma(freeze_signature: str, fit_id: str, artifact: Path,
                       receipt_path: Path) -> dict[str, Any]:
    """Resume from an already durable artifact/receipt without invoking a fit."""
    return verify_gamma_bundle(freeze_signature, fit_id, artifact, receipt_path)


def verify_fit_artifact(freeze: Mapping[str, Any], fit_id: str) -> dict[str, Any]:
    require(fit_id in freeze["fit_ids"], "Fit ID is not frozen")
    artifact, receipt_path = fit_paths(fit_id)
    verified = verify_gamma_bundle(freeze["freeze_signature"], fit_id, artifact, receipt_path)
    rec = verified["receipt"]
    require(rec.get("source_hashes_sha256") == canonical_sha256(freeze["source_hashes"])
            and rec.get("input_hashes_sha256") == canonical_sha256(freeze["input_hashes"])
            and rec.get("parent_freeze_signature") == freeze["parent_freeze_signature"],
            f"Fit receipt source/input lineage differs: {fit_id}")
    return verified


def resume_started_fit(freeze: Mapping[str, Any], fit_id: str, ledger_path: Path,
                       artifact: Path, receipt_path: Path) -> dict[str, Any]:
    """Complete a pending fit only from a durable, lineage-valid receipt."""
    ev=events(ledger_path)
    starts=[e for e in ev if e.get("event")=="FIT_STARTED" and e.get("fit_id")==fit_id]
    terminals=[e for e in ev if e.get("event") in {"FIT_SAVED","FIT_FAILED"} and e.get("fit_id")==fit_id]
    require(len(starts)==1 and not terminals,f"Pending fit ledger is not resumable: {fit_id}")
    if not artifact.is_file() or not receipt_path.is_file():
        append_event(ledger_path,{"event":"FIT_FAILED","fit_id":fit_id,
            "freeze_signature":freeze["freeze_signature"],"error":"interrupted_without_durable_receipt",
            "retry_allowed":False})
        return {"status":"FAILED_INTERRUPTED_NO_RETRY"}
    verified=verify_gamma_bundle(freeze["freeze_signature"],fit_id,artifact,receipt_path)
    rec=verified["receipt"]
    require(rec.get("source_hashes_sha256")==canonical_sha256(freeze["source_hashes"])
            and rec.get("input_hashes_sha256")==canonical_sha256(freeze["input_hashes"])
            and rec.get("parent_freeze_signature")==freeze["parent_freeze_signature"],
            f"Durable fit receipt source/input lineage differs: {fit_id}")
    append_event(ledger_path,{"event":"FIT_SAVED","fit_id":fit_id,
        "freeze_signature":freeze["freeze_signature"],"receipt_sha256":verified["receipt_sha256"],
        "resumed_from_durable_receipt":True})
    return {"status":"RESUMED_DURABLE_RECEIPT","gamma_sha256":rec["gamma_sha256"]}


def _selected_targets(train_path: Path, rows: Sequence[int], expected_ids: Sequence[str]) -> np.ndarray:
    """Read target values only on selected CSV record indices using parser-level row skipping."""
    wanted = np.asarray(rows, dtype=np.int64).reshape(-1)
    ids = np.asarray(expected_ids).astype(str).reshape(-1)
    require(len(wanted) == len(ids) and len(np.unique(wanted)) == len(wanted), "Selected target rows/IDs invalid")
    if len(wanted) == 0:
        return np.empty(0, dtype=np.int8)
    selected = set(int(x) for x in wanted.tolist())
    frame = pd.read_csv(Path(train_path), usecols=["record_id", FIT_LABEL_COL],
                        dtype={"record_id": "string", FIT_LABEL_COL: "int8"}, keep_default_na=False,
                        skiprows=lambda line: line > 0 and (line - 1) not in selected)
    order = np.argsort(wanted)
    sorted_rows = wanted[order]
    sorted_ids = ids[order]
    require(len(frame) == len(wanted), "Selected target reader returned a different number of records")
    require(np.array_equal(frame["record_id"].astype(str).to_numpy(), sorted_ids),
            "Selected target IDs differ from frozen feature identities")
    y_sorted = frame[FIT_LABEL_COL].to_numpy(dtype=np.int8)
    require(np.isin(y_sorted, (0, 1)).all(), "Selected labels must be binary 0/1")
    inverse = np.empty(len(order), dtype=np.int64); inverse[order] = np.arange(len(order))
    return y_sorted[inverse]


def _fit_id_parts(fit_id: str) -> tuple[int, str]:
    if fit_id not in ("CLS_POS_EF_CONTEXT_V1_E67_REAL", "CLS_POS_EF_CONTEXT_V1_E67_SHUFFLED",
                      "CLS_POS_EF_CONTEXT_V1_E75_REAL", "CLS_POS_EF_CONTEXT_V1_E75_SHUFFLED"):
        raise PilotError(f"Unfrozen fit ID: {fit_id}")
    return (67 if "E67" in fit_id else 75, "REAL" if fit_id.endswith("REAL") else "SHUFFLED")


def fit_one(repository: Path, fit_id: str) -> dict[str, Any]:
    from src.models import cls_pos_ef_context_v1 as core
    freeze = verify_freeze(repository); ev = check_live_budget(freeze, allow_current=fit_id)
    origin, arm = _fit_id_parts(fit_id)
    artifact, receipt_path = fit_paths(fit_id)
    starts = [e for e in ev if e.get("event") == "FIT_STARTED" and e.get("fit_id") == fit_id]
    terminal = [e for e in ev if e.get("event") in {"FIT_SAVED", "FIT_FAILED"} and e.get("fit_id") == fit_id]
    if terminal:
        require(len(starts) == len(terminal) == 1, f"Fit accounting is inconsistent: {fit_id}")
        if terminal[0]["event"] == "FIT_FAILED":
            return {"fit_id": fit_id, "status": "FAILED_CONSUMED_NO_RETRY"}
        verified = verify_fit_artifact(freeze, fit_id)
        return {"fit_id": fit_id, "status": "ALREADY_SAVED", "gamma_sha256": verified["receipt"]["gamma_sha256"]}
    if starts:
        require(len(starts) == 1, f"Duplicate fit start: {fit_id}")
        resumed=resume_started_fit(freeze,fit_id,RUN_ROOT/"fit_ledger.jsonl",artifact,receipt_path)
        if resumed["status"]=="FAILED_INTERRUPTED_NO_RETRY":
            raise PilotError(f"Interrupted fit {fit_id} has no durable receipt; consumed and not retried")
        return {"fit_id":fit_id,**resumed}
    started_ids = {e.get("fit_id") for e in ev if e.get("event") == "FIT_STARTED"}
    next_fit = next((candidate for candidate in freeze["fit_ids"] if candidate not in started_ids), None)
    require(fit_id == next_fit, f"Fits must follow frozen order; next authorized ID is {next_fit}")
    require(not any(e.get("event") == "FIT_STARTED" and e.get("fit_id") in freeze["fit_ids"] for e in ev if e.get("fit_id") != fit_id and e.get("event") == "FIT_STARTED"),
            "Another frozen fit is pending; one heavy fit at a time")
    require(not any(e.get("fit_id") == fit_id for e in ev), f"Fit ID already appears in ledger: {fit_id}")
    if os.path.exists(str(artifact)) or os.path.exists(str(receipt_path)):
        raise PilotError(f"Orphan artifact exists before fit start: {fit_id}")
    with heavy_job_lock():
        available = int(psutil.virtual_memory().available)
        require(available >= MIN_AVAILABLE and available - ESTIMATED_PEAK >= MIN_RESERVE,
                f"RAM gate failed before source load: free={available}, min_start={MIN_AVAILABLE}, reserve={MIN_RESERVE}")
        frame = read_features(Path(repository) / "data/raw/classification/Train.csv")
        p = load_parent_origin(repository, origin, frame)
        fit_mask = p["fit_present"]
        fit_rows = p["fit_rows"][fit_mask]
        fit_ids = p["fit_ids"][fit_mask]
        ef = p["fit_ef"][fit_mask]
        ef_use = ef if arm == "REAL" else p["fit_ef_shuffled"][fit_mask]
        q = core.standardized_context(ef_use, p["context_state"])
        x = core.interaction_design(p["x_fit"][fit_mask], q)
        y = _selected_targets(Path(repository) / "data/raw/classification/Train.csv", fit_rows, fit_ids)
        ev = check_live_budget(freeze)
        starts_now = [e for e in ev if e.get("event") == "FIT_STARTED"]
        require(len(starts_now) < 4, "Four-fit cap has already been consumed")
        available = int(psutil.virtual_memory().available)
        require(available >= MIN_AVAILABLE and available - ESTIMATED_PEAK >= MIN_RESERVE,
                f"RAM gate failed: free={available}, min_start={MIN_AVAILABLE}, reserve={MIN_RESERVE}")
        resource = {"available_bytes": available, "minimum_start_bytes": MIN_AVAILABLE,
                    "estimated_peak_bytes": ESTIMATED_PEAK, "minimum_reserve_bytes": MIN_RESERVE,
                    "threads": 2}
        fit_meta = {"origin": origin, "arm": arm, "rows": int(len(fit_rows)),
            "fit_rows_sha256": sha256_array(fit_rows), "fit_ids_sha256": sha256_array(fit_ids),
            "fit_target_sha256": sha256_array(y), "context_state": p["context_state"],
            "ef_assignment_sha256": sha256_array(ef), "ef_used_sha256": sha256_array(ef_use),
            "moved_present_assignments": int(p["fit_moved_present"] if arm == "SHUFFLED" else 0),
            "fit_position_sha256": sha256_csr(p["x_fit"][fit_mask]),
            "interaction_sha256": sha256_csr(x), "z_pos_sha256": sha256_array(p["z_pos_fit"][fit_mask]),
            "source_manifest_sha256": p["matrix_manifest_sha256"]}
        append_event(RUN_ROOT / "fit_ledger.jsonl", {"event": "FIT_STARTED", "fit_id": fit_id,
            "freeze_signature": freeze["freeze_signature"], "source_commit": freeze["source_commit"],
            "fit_population": fit_meta, "resource": resource, "fit_consumed_at_start": True})
        t0 = time.perf_counter()
        try:
            with threadpool_limits(limits=2):
                gamma, optimizer = core.fit_context_gamma(x, p["z_pos_fit"][fit_mask], y)
            atomic_npz(artifact, gamma=gamma)
            receipt = {"schema_version": 1, "fit_id": fit_id, "origin": origin, "arm": arm,
                "freeze_signature": freeze["freeze_signature"], "source_commit": freeze["source_commit"],
                "input_sha256": freeze["input_sha256"],
                "source_hashes_sha256": canonical_sha256(freeze["source_hashes"]),
                "input_hashes_sha256": canonical_sha256(freeze["input_hashes"]),
                "parent_freeze_signature": freeze["parent_freeze_signature"], "fit_population": fit_meta,
                "position_width": int(x.shape[1]), "gamma_sha256": sha256_array(gamma),
                "artifact_sha256": sha256_file(artifact), "artifact_path": artifact.relative_to(AUDIT_ROOT).as_posix(),
                "optimizer": optimizer, "resource_before": resource,
                "resource_after_available_bytes": int(psutil.virtual_memory().available),
                "elapsed_seconds": float(time.perf_counter() - t0), "retry_allowed": False,
                "query_labels_read": False, "test_or_sample_access": False}
            write_once(receipt_path, receipt)
            verified = verify_fit_artifact(freeze, fit_id)
            append_event(RUN_ROOT / "fit_ledger.jsonl", {"event": "FIT_SAVED", "fit_id": fit_id,
                "freeze_signature": freeze["freeze_signature"], "receipt_sha256": verified["receipt_sha256"]})
            return {"fit_id": fit_id, "status": "SAVED", "gamma_sha256": receipt["gamma_sha256"],
                    "optimizer": optimizer}
        except BaseException as exc:
            append_event(RUN_ROOT / "fit_ledger.jsonl", {"event": "FIT_FAILED", "fit_id": fit_id,
                "freeze_signature": freeze["freeze_signature"], "error_type": type(exc).__name__,
                "error": str(exc), "retry_allowed": False})
            raise


def _load_saved_predictions(path: Path) -> dict[str, np.ndarray]:
    allow = ("origin", "row_positions", "record_ids", "cycles", "present", "ef1", "text_load_group",
             "p_a02", "p_pos_original", "p_champion", "p_real", "p_shuffled",
             "label_champion", "label_real", "label_shuffled", "delta_real", "delta_shuffled")
    with np.load(path, allow_pickle=False) as archive:
        require(archive.files == list(allow), f"Prediction bundle schema differs: {path}")
        return {key: archive[key].copy() for key in allow}


def _build_origin_arrays(repository: Path, origin: int, freeze: Mapping[str, Any], frame: pd.DataFrame) -> dict[str, np.ndarray]:
    from src.models import cls_pos_ef_context_v1 as core
    p = load_parent_origin(repository, origin, frame)
    new = {}
    for arm in core.ARMS:
        fit_id = f"CLS_POS_EF_CONTEXT_V1_E{origin}_{arm}"
        gamma = verify_fit_artifact(freeze, fit_id)["gamma"]
        qef = p["ef"] if arm == "REAL" else p["query_ef_shuffled"]
        q = core.standardized_context(qef, p["context_state"])
        avail = p["present"]
        q_present = q[avail]
        x_interaction = core.interaction_design(p["x_score"][avail], q_present)
        p_context, delta = core.predict_context_probability(
            x_interaction, p["z_pos_score"][avail], gamma)
        full_prob = p["p_pos_original"].copy()
        full_prob[avail] = p_context
        full_delta = np.zeros(len(avail), dtype=np.float64); full_delta[avail] = delta
        new[arm] = (core.route_candidate(p["p_a02"], full_prob, avail), full_delta)
    return {"origin": np.asarray([origin], dtype=np.int64), "row_positions": p["score_rows"].astype(np.int64),
        "record_ids": p["score_ids"].astype(str), "cycles": p["cycles"].astype(np.int64),
        "present": p["present"].astype(bool), "ef1": p["ef"].astype(np.int8),
        "text_load_group": p["text_load_group"].astype(str), "p_a02": p["p_a02"].astype(np.float64),
        "p_pos_original": p["p_pos_original"].astype(np.float64), "p_champion": p["p_champion"].astype(np.float64),
        "p_real": new["REAL"][0].astype(np.float64), "p_shuffled": new["SHUFFLED"][0].astype(np.float64),
        "label_champion": p["label_champion"].astype(np.int8),
        "label_real": core.apply_thresholds(new["REAL"][0], p["present"]),
        "label_shuffled": core.apply_thresholds(new["SHUFFLED"][0], p["present"]),
        "delta_real": new["REAL"][1].astype(np.float64), "delta_shuffled": new["SHUFFLED"][1].astype(np.float64)}


def build_predictions(repository: Path) -> dict[str, Any]:
    freeze = verify_freeze(repository)
    ev = check_live_budget(freeze)
    saved = {e["fit_id"] for e in ev if e.get("event") == "FIT_SAVED"}
    failed = {e["fit_id"] for e in ev if e.get("event") == "FIT_FAILED"}
    require(saved == set(freeze["fit_ids"]) and not failed,
            "Exactly four durable gamma receipts are required before prediction construction")
    frame = read_features(Path(repository) / "data/raw/classification/Train.csv")
    outputs = {}
    for origin in (67, 75):
        arrays = _build_origin_arrays(repository, origin, freeze, frame)
        path = RUN_ROOT / f"E{origin}" / f"predictions_E{origin}.npz"
        if path.exists():
            old = _load_saved_predictions(path)
            require(all(np.array_equal(old[k], v) for k, v in arrays.items()),
                    f"Existing predictions differ from frozen replay for E{origin}")
        else:
            atomic_npz(path, **arrays)
        outputs[f"E{origin}"] = {"path": path.relative_to(AUDIT_ROOT).as_posix(),
            "sha256": sha256_file(path), "rows": int(len(arrays["row_positions"])),
            "present_rows": int(arrays["present"].sum()), "missing_rows": int((~arrays["present"]).sum()),
            "ordered_ids_sha256": sha256_array(arrays["record_ids"])}
    ready = {"schema_version": 1, "candidate_id": "CLS_POS_EF_CONTEXT_V1",
        "freeze_signature": freeze["freeze_signature"], "source_commit": freeze["source_commit"],
        "input_sha256": freeze["input_sha256"], "parent_freeze_signature":freeze["parent_freeze_signature"],
        "fit_receipts":{fit_id:verify_fit_artifact(freeze,fit_id)["receipt_sha256"] for fit_id in freeze["fit_ids"]},
        "predictions": outputs,
        "all_four_fit_receipts_durable": True, "query_labels_opened": False,
        "test_or_sample_access": False}
    write_once(RUN_ROOT / "PREDICTIONS_READY.json", ready)
    write_once(READY_PATH, ready)
    return ready


def _verified_ready(freeze: Mapping[str, Any]) -> dict[str, Any]:
    ready = read_json(READY_PATH)
    require(ready.get("candidate_id") == freeze["candidate_id"]
            and ready.get("freeze_signature") == freeze["freeze_signature"]
            and ready.get("query_labels_opened") is False
            and ready.get("test_or_sample_access") is False
            and set(ready.get("predictions", {})) == {"E67", "E75"},
            "Target-free prediction readiness binding failed")
    for suffix, item in ready["predictions"].items():
        path = AUDIT_ROOT / item["path"]
        require(sha256_file(path) == item["sha256"], f"Prediction output changed: {suffix}")
    return ready


def verify_prelabel(repository: Path, *, write: bool = True) -> dict[str, Any]:
    """Run the independent verifier; producer code does not certify itself."""
    from scripts.verify_cls_pos_ef_context_v1 import verify_prelabel_outputs
    return verify_prelabel_outputs(repository, write=write)


def verify_committed_prelabel() -> dict[str, Any]:
    require(PRELABEL_PATH.is_file(), "Committed prelabel report is required before query-label scoring")
    rel = PRELABEL_PATH.relative_to(AUDIT_ROOT).as_posix()
    committed = subprocess.check_output(["git", "-C", str(AUDIT_ROOT), "show", f"HEAD:{rel}"])
    require(committed == PRELABEL_PATH.read_bytes(), "Prelabel report must be committed before labels open")
    body = read_json(PRELABEL_PATH); signature = body.get("verification_signature")
    check = dict(body); check.pop("verification_signature", None)
    require(signature == canonical_sha256(check) and body.get("status") == "PASS_TARGET_FREE_PRELABEL"
            and body.get("query_labels_read") is False and body.get("test_or_sample_access") is False,
            "Committed prelabel report failed signature/boundary checks")
    return body


def _combined_views(bundles: Mapping[int, Mapping[str, np.ndarray]], ymap: Mapping[int, int]) -> dict[str, dict[str, np.ndarray]]:
    def aligned(origin: int) -> dict[str, np.ndarray]:
        b = bundles[origin]
        rows = np.asarray(b["row_positions"], dtype=np.int64)
        return {**b, "y": np.asarray([ymap[int(r)] for r in rows], dtype=np.int8)}
    e67, e75 = aligned(67), aligned(75)
    require(set(e75["row_positions"].tolist()).issubset(set(e67["row_positions"].tolist())),
            "E75 query rows must be contained in the E67 H12 query window")
    unique67 = (e67["cycles"] >= 68) & (e67["cycles"] <= 75)
    unique75 = (e75["cycles"] >= 76) & (e75["cycles"] <= 79)
    unique = {}
    for name in e67:
        if isinstance(e67[name], np.ndarray) and e67[name].ndim >= 1 and len(e67[name]) == len(e67["row_positions"]):
            unique[name] = np.concatenate([e67[name][unique67], e75[name][unique75]])
    require(len(unique["row_positions"]) == len(np.unique(unique["row_positions"])) == 20903,
            "Unique primary pool is not exactly 20,903 nonoverlapping rows")
    return {"E67": e67, "E75": e75, "UNIQUE_PRIMARY": unique}


def score_metrics(repository: Path) -> dict[str, Any]:
    from src.models import cls_pos_ef_context_v1 as core
    from scripts import verify_cls_pos_ef_context_v1 as metric_verifier
    freeze = verify_freeze(repository); prelabel = verify_committed_prelabel()
    require(prelabel["freeze_signature"] == freeze["freeze_signature"], "Prelabel freeze signature changed")
    require(prelabel["predictions_ready_sha256"] == sha256_file(READY_PATH), "Prelabel readiness binding changed")
    for suffix, expected in prelabel["prediction_output_sha256s"].items():
        item = read_json(READY_PATH)["predictions"][suffix]
        require(sha256_file(AUDIT_ROOT / item["path"]) == expected, f"Prelabel output changed: {suffix}")
    ledger = RUN_ROOT / "fit_ledger.jsonl"; ev = events(ledger)
    prior = [e for e in ev if e.get("event") == "QUERY_LABELS_OPENED"]
    if prior:
        require(SCORE_PATH.is_file(), "Query labels were already opened without a durable score; refusing reread")
        saved_score = read_json(SCORE_PATH)
        from scripts.verify_cls_pos_ef_context_v1 import write_metric_report_from_saved_score
        write_metric_report_from_saved_score(saved_score)
        return saved_score
    # E67 covers the full union once; E75 rows overlap its cycles 76-79.
    bundles = {o: _load_saved_predictions(RUN_ROOT / f"E{o}/predictions_E{o}.npz") for o in (67, 75)}
    rows = np.asarray(bundles[67]["row_positions"], dtype=np.int64)
    ids = np.asarray(bundles[67]["record_ids"]).astype(str)
    append_event(ledger, {"event": "QUERY_LABELS_OPENED", "candidate_id": freeze["candidate_id"],
        "freeze_signature": freeze["freeze_signature"], "prelabel_sha256": sha256_file(PRELABEL_PATH),
        "query_rows_sha256": sha256_array(rows), "query_ids_sha256": sha256_array(ids),
        "rows": int(len(rows)), "purpose": "single Train temporal metric evaluation"})
    y = _selected_targets(Path(repository) / "data/raw/classification/Train.csv", rows, ids)
    ymap = {int(row): int(label) for row, label in zip(rows.tolist(), y.tolist())}
    views = _combined_views(bundles, ymap)
    metric_checks: list[dict[str, Any]] = []
    def checked_compare(metric_id: str, *args: Any) -> dict[str, Any]:
        expected = core.metric_comparison(*args)
        independent = metric_verifier.independent_metric_comparison(*args)
        check = metric_verifier.compare_metric_outputs(expected, independent)
        require(check["status"] == "PASS", f"Independent metric verification failed: {metric_id}")
        metric_checks.append({"metric_id": metric_id, **check,
                              "metric_digest": canonical_sha256(expected)})
        return expected
    view_ranges = {"E67": {"W4": (68,71), "W5": (72,75), "W6": (76,79), "H12_E67": (68,79)},
                   "E75": {"W6": (76,79)}}
    groups: dict[str, Any] = {}
    subgroup_failures = []; q4_guard_ok = True; exploratory_subgroups_ok = True
    for view_name, block in views.items():
        local_views = {"UNIQUE_PRIMARY": np.ones(len(block["y"]), dtype=bool)} if view_name == "UNIQUE_PRIMARY" else {
            name: ((block["cycles"] >= lo) & (block["cycles"] <= hi))
            for name,(lo,hi) in view_ranges[view_name].items()}
        groups[view_name] = {}
        for name, mask in local_views.items():
            if int(mask.sum()) == 0: continue
            b = {k: v[mask] for k,v in block.items() if isinstance(v, np.ndarray) and len(v) == len(mask)}
            comparison = checked_compare(f"{view_name}/{name}/real_vs_champion", b["y"], b["p_champion"], b["p_real"], b["label_champion"], b["label_real"])
            shuffle = checked_compare(f"{view_name}/{name}/shuffled_vs_champion", b["y"], b["p_champion"], b["p_shuffled"], b["label_champion"], b["label_shuffled"])
            qcomp = checked_compare(f"{view_name}/{name}/real_vs_shuffled", b["y"], b["p_shuffled"], b["p_real"], b["label_shuffled"], b["label_real"])
            subgroups = []
            for kind, values in (("availability", b["present"]), ("entry_friction", b["ef1"]),
                                 ("fit_prefix_text_load", b["text_load_group"])):
                labels = sorted(set(np.asarray(values).astype(str).tolist()))
                for value in labels:
                    group_mask = np.asarray(values).astype(str) == value
                    support = core.supported(group_mask, b["y"])
                    metrics = checked_compare(f"{view_name}/{name}/{key}/real_vs_champion",
                        b["y"][group_mask], b["p_champion"][group_mask], b["p_real"][group_mask],
                        b["label_champion"][group_mask], b["label_real"][group_mask]) if support["supported"] else None
                    key = f"{kind}={value}"
                    subgroups.append({"group": key, "support": support, "metrics": metrics})
                    if support["supported"] and metrics is not None:
                        f1_gain=metrics["gain_macro_f1"]; recall_gain=metrics["gain_positive_recall"]
                        if f1_gain < -0.01 or recall_gain < -0.02:
                            q4_guard_ok = False
                            subgroup_failures.append({"view":f"{view_name}/{name}","group":key,"guard":"legacy",
                                "gain_macro_f1":f1_gain,"gain_positive_recall":recall_gain})
                        if f1_gain < -0.02 or recall_gain < -0.03:
                            exploratory_subgroups_ok=False
            groups[view_name][name] = {"real_vs_champion": comparison,
                "shuffled_vs_champion": shuffle, "real_vs_shuffled": qcomp,
                "supported_subgroups": subgroups}
        if view_name == "UNIQUE_PRIMARY":
            cycles = sorted(np.unique(block["cycles"]).tolist())
            per_cycle = {}
            for cyc in cycles:
                m = block["cycles"] == cyc
                per_cycle[str(int(cyc))] = checked_compare(f"UNIQUE_PRIMARY/cycle={int(cyc)}",
                    block["y"][m], block["p_champion"][m], block["p_real"][m],
                    block["label_champion"][m], block["label_real"][m])
            groups[view_name]["per_cycle"] = per_cycle
            groups[view_name]["delta_logit"] = {
                "real": core.delta_logit_summary(block["delta_real"][block["present"]]),
                "shuffled": core.delta_logit_summary(block["delta_shuffled"][block["present"]])}
    primary = groups["UNIQUE_PRIMARY"]["UNIQUE_PRIMARY"]["real_vs_champion"]
    e67_h12 = groups["E67"]["H12_E67"]["real_vs_champion"]["gain_macro_f1"]
    e75_w6 = groups["E75"]["W6"]["real_vs_champion"]["gain_macro_f1"]
    primary_shuffle_gain = groups["UNIQUE_PRIMARY"]["UNIQUE_PRIMARY"]["real_vs_shuffled"]["gain_macro_f1"]
    e75_shuffle_gain = groups["E75"]["W6"]["real_vs_shuffled"]["gain_macro_f1"]
    missing_exact = all(v["missing_fallback_exact"] is True for v in prelabel["predictions"].values())
    independent_qa_pass = (prelabel.get("status") == "PASS_TARGET_FREE_PRELABEL"
                           and all(c["status"] == "PASS" for c in metric_checks))
    scientific = (primary["gain_macro_f1"] >= 0.0005 and e67_h12 >= 0.0 and e75_w6 >= 0.0
                  and primary_shuffle_gain >= 0.0002 and e75_shuffle_gain >= 0.0 and q4_guard_ok
                  and missing_exact and independent_qa_pass)
    exploratory_gains = primary["gain_macro_f1"] > 0.0 or (e75_w6 >= 0.0005 and primary["gain_macro_f1"] >= -0.0005)
    exploratory = exploratory_gains and exploratory_subgroups_ok and missing_exact and independent_qa_pass
    result = {"schema_version":1,"candidate_id":"CLS_POS_EF_CONTEXT_V1","status":"SCORED_DIAGNOSTIC_ONLY",
        "freeze_signature":freeze["freeze_signature"],"source_commit":freeze["source_commit"],
        "prelabel_sha256":sha256_file(PRELABEL_PATH),"query_labels_opened":True,
        "query_target_source":"official Train rows in frozen temporal windows only","test_or_sample_access":False,
        "primary_unique_pool":{"rows":int(len(views["UNIQUE_PRIMARY"]["y"])),
            "definition":"E67 cycles 68-75 plus E75 cycles 76-79; non-overlap",
            "real_vs_champion":primary},
        "views":groups,"subgroup_guard_failures":subgroup_failures,
        "gates":{"scientific_qualified":bool(scientific),"exploratory_eligible":bool(exploratory),
            "exploratory_gains_ok":bool(exploratory_gains),"legacy_subgroup_guard_ok":bool(q4_guard_ok),
            "exploratory_subgroup_guard_ok":bool(exploratory_subgroups_ok),
            "missing_fallback_exact":bool(missing_exact),"independent_qa_pass":bool(independent_qa_pass),
            "unique_gain_min":0.0005,"E67_H12_gain_min":0.0,"E75_W6_gain_min":0.0,
            "real_vs_shuffled_unique_min":0.0002,"real_vs_shuffled_E75_W6_min":0.0,
            "supported_legacy_group_harm_max":{"macro_f1":0.01,"positive_recall":0.02},
            "exploratory_group_harm_max":{"macro_f1":0.02,"positive_recall":0.03}},
        "official_repository_root":str(Path(repository).resolve()),
        "interpretation":"A within-Train temporal diagnostic only; reused development outcomes are not fresh holdout evidence. No automatic champion update, E79, Test, or Kaggle action."}
    result["metric_verification"] = metric_verifier.build_metric_verification(result["views"], metric_checks)
    write_once(SCORE_PATH, result)
    verification = {"status":"PASS_INDEPENDENT_METRIC_RECALC", "score_sha256":sha256_file(SCORE_PATH),
                    **result["metric_verification"]}
    verification["report_signature"] = canonical_sha256(verification)
    write_once(METRIC_VERIFY_PATH, verification)
    return result


def _synthetic_labels_only(path: Path, rows: Sequence[int]) -> np.ndarray:
    ids = [f"id-{i}" for i in rows]
    return _selected_targets(path, rows, ids)


def synthetic_lifecycle(output: Path) -> dict[str, Any]:
    """Small complete persistence/readback/prelabel/score lifecycle; no official data."""
    from src.models import cls_pos_ef_context_v1 as core
    from scripts import verify_cls_pos_ef_context_v1 as verifier
    import tempfile as _tempfile
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    n = 240; rng = np.random.default_rng(177)
    cycles = np.repeat(np.arange(47, 59), 20).astype(np.int64)
    present = (np.arange(n) % 6 != 0)
    ef = (np.arange(n) % 2).astype(np.int8)
    x = sparse.csr_matrix(rng.normal(size=(n, 8)) * present[:, None])
    state = core.context_state(ef[:160], present[:160], minimum_per_level=20)
    ef_shuffle, moved = core.shuffle_context_within_cycle(ef[:160], cycles[:160], present[:160])
    require(moved > 0, "Synthetic control did not move assignments")
    q = core.standardized_context(ef[:160], state)
    zi = np.linspace(-1.1, 0.8, 160)
    synthetic_delta = np.asarray(core.interaction_design(x[:160], q) @ (np.arange(1,9)/30), dtype=np.float64).reshape(-1)
    train_y = (rng.random(160) < expit(zi + synthetic_delta)).astype(np.int8)
    gamma, opt = core.fit_context_gamma(core.interaction_design(x[:160], q), zi, train_y)
    fit_id = core.FIT_IDS[0]; artifact = output / "units" / f"{fit_id}.npz"; receipt_path = output / "receipts" / f"{fit_id}.json"
    atomic_npz(artifact, gamma=gamma)
    synthetic_freeze={"freeze_signature":"synthetic-freeze","source_hashes":{"synthetic.py":"abc"},
        "input_hashes":{"synthetic.csv":"def"},"parent_freeze_signature":"synthetic-parent"}
    receipt = {"fit_id":fit_id,"freeze_signature":synthetic_freeze["freeze_signature"],"position_width":8,
        "gamma_sha256":sha256_array(gamma),"artifact_sha256":sha256_file(artifact),"optimizer":opt,
        "source_hashes_sha256":canonical_sha256(synthetic_freeze["source_hashes"]),
        "input_hashes_sha256":canonical_sha256(synthetic_freeze["input_hashes"]),
        "parent_freeze_signature":synthetic_freeze["parent_freeze_signature"]}
    write_once(receipt_path, receipt)
    with np.load(artifact,allow_pickle=False) as archive: gamma_read=archive["gamma"].copy()
    require(np.array_equal(gamma, gamma_read),"Synthetic gamma persistence/readback failed")
    old_fit=core.fit_context_gamma
    core.fit_context_gamma=lambda *args,**kwargs: (_ for _ in ()).throw(AssertionError("resume must not refit"))
    ledger_path=output/"resume_ledger.jsonl"
    append_event(ledger_path,{"event":"FIT_STARTED","fit_id":fit_id,"freeze_signature":"synthetic-freeze"})
    resumed=resume_started_fit(synthetic_freeze,fit_id,ledger_path,artifact,receipt_path)
    core.fit_context_gamma=old_fit
    require(resumed["status"]=="RESUMED_DURABLE_RECEIPT"
            and events(ledger_path)[-1]["event"]=="FIT_SAVED","Synthetic durable-resume path failed")
    q_query=core.standardized_context(ef[160:],state)
    xq=core.interaction_design(x[160:],q_query)
    ppos,delta=core.predict_context_probability(xq,np.linspace(-0.6,0.9,80),gamma)
    p_a=np.full(80,0.43); present_q=present[160:]
    pchamp=core.route_candidate(p_a,np.full(80,0.51),present_q)
    pcand=core.route_candidate(p_a,ppos,present_q)
    pfile=output/"synthetic_predictions.npz"
    atomic_npz(pfile,p_a02=p_a,p_pos_context=ppos,p_champion=pchamp,p_candidate=pcand,present=present_q,delta=delta)
    with np.load(pfile,allow_pickle=False) as archive: saved={k:archive[k].copy() for k in archive.files}
    require(np.array_equal(saved["p_candidate"],pcand),"Synthetic prediction readback failed")
    prelabel={"status":"PASS_TARGET_FREE_PRELABEL","prediction_sha256":sha256_file(pfile),
        "query_labels_read":False,"query_label_hash_present":False}
    prelabel["verification_signature"]=canonical_sha256(prelabel)
    prelabel_path=output/"synthetic_prelabel.json"; write_once(prelabel_path,prelabel)
    verifier.verify_synthetic_prelabel(pfile,prelabel_path)
    # Synthetic query outcomes are introduced only after the prelabel readback passes.
    yq=(np.arange(80)%4==0).astype(np.int8)
    support=core.supported(np.ones(80,bool),yq,min_n=50,min_per_class=10)
    metrics=core.metric_comparison(yq,pchamp,pcand,core.apply_thresholds(pchamp,present_q),core.apply_thresholds(pcand,present_q))
    independent=verifier.independent_metric_comparison(yq,pchamp,pcand,
        core.apply_thresholds(pchamp,present_q),core.apply_thresholds(pcand,present_q))
    metric_check=verifier.compare_metric_outputs(metrics,independent)
    require(metric_check["status"]=="PASS",f"Synthetic independent metric verifier differs: {metric_check}")
    # Query-label poison sentinel must remain unparsed by the fit-prefix selector.
    csv_path=output/"selected-labels.csv"
    csv_path.write_text("record_id,feature,artifact_signal\n"+"\n".join(
        f"id-{i},x,{int(train_y[i]) if i<160 else 'POISON_QUERY_LABEL'}" for i in range(n))+"\n",encoding="utf-8")
    selected=_synthetic_labels_only(csv_path,np.arange(160))
    require(np.array_equal(selected,train_y),"Selected-label reader read wrong synthetic prefix")
    # Tampering is detected, and terminal fit IDs cannot trigger an optimizer retry.
    tampered=output/"tampered_gamma.npz"; tampered_receipt=output/"tampered_gamma.json"
    atomic_npz(tampered,gamma=gamma+0.01)
    bad=dict(receipt); bad["artifact_sha256"]=sha256_file(artifact)
    write_once(tampered_receipt,bad)
    tamper_detected=False
    try: verify_gamma_bundle("synthetic-freeze",fit_id,tampered,tampered_receipt)
    except PilotError: tamper_detected=True
    require(tamper_detected,"Synthetic tamper test failed")
    failure_rejected=False
    try: core.objective_gradient(gamma,sparse.csr_matrix((1,8)),np.zeros(1),np.array([2],dtype=np.int8))
    except ValueError: failure_rejected=True
    require(failure_rejected,"Synthetic invalid-fit failure test failed")
    require(not support["supported"] or support["n"]==80,"Synthetic support arithmetic failed")
    return {"status":"PASS_SYNTHETIC_END_TO_END","rows":n,"fit_rows":160,"query_rows":80,
        "fit_gamma_sha256":sha256_array(gamma),"prediction_bundle_sha256":sha256_file(pfile),
        "metric_verifier_equal":True,"poison_query_labels_excluded":True,"tamper_detected":True,
        "persistence_readback":True,"durable_resume_without_refit":True,
        "synthetic_prelabel_pass":True,"query_labels_read_before_score":False,
        "failed_input_rejected":True,
        "fit_optimizer":{"success":opt["success"],"nit":opt["nit"],"gradient_inf_norm":opt["gradient_inf_norm"]},
        "metrics":metrics}


def main(argv: Sequence[str] | None = None) -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest="cmd",required=True)
    for name in ("prepare-source","freeze","reserve","build","prelabel","score","verify-score"):
        p=sub.add_parser(name); p.add_argument("--official-root",type=Path,required=name!="verify-score")
    p=sub.add_parser("fit"); p.add_argument("--official-root",type=Path,required=True); p.add_argument("--fit-id",required=True)
    p=sub.add_parser("synthetic"); p.add_argument("--output",type=Path,required=True)
    args=ap.parse_args(argv)
    if args.cmd=="synthetic": result=synthetic_lifecycle(args.output)
    elif args.cmd=="prepare-source": result=build_support_preflight(args.official_root)
    elif args.cmd=="freeze": result=create_freeze(args.official_root)
    elif args.cmd=="reserve": result=reserve(args.official_root)
    elif args.cmd=="fit": result=fit_one(args.official_root,args.fit_id)
    elif args.cmd=="build": result=build_predictions(args.official_root)
    elif args.cmd=="prelabel": result=verify_prelabel(args.official_root)
    elif args.cmd=="score": result=score_metrics(args.official_root)
    else:
        from scripts.verify_cls_pos_ef_context_v1 import verify_score
        result=verify_score()
    print(json.dumps(result,sort_keys=True,indent=2,allow_nan=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
