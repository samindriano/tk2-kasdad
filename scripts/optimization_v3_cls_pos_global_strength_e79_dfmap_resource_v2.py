"""Run one V2-resource-gated E79 Train-only DF-map pass.

The map formula and data source remain the frozen V1 implementation. This
wrapper changes only the conservative local memory estimate and instruments
the sparse-union allocation bound before computing the map.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping

import numpy as np
import psutil
from scipy import sparse
from threadpoolctl import threadpool_limits

from scripts import optimization_v3_cls_pos_global_strength_e79_dfmap_v1 as v1
from scripts import optimization_v3_cls_pos_global_strength_e79_v3 as r3
from scripts import verify_optimization_v3_cls_pos_global_strength_e79_dfmap_v1 as v1_verify
from src.models import classification_models, optimization_v3_positional_text_pilot_v1 as positional


ROOT = Path(__file__).resolve().parents[1]
POLICY_ID = "CLS_POS_GLOBAL_STRENGTH_E79_DF_MAP_RESOURCE_V2"
POLICY_REL = "reports/optimization_v3/cls_pos_global_strength_e79_v3/DFMAP_RESOURCE_POLICY_V2.md"
WRAPPER_REL = "scripts/optimization_v3_cls_pos_global_strength_e79_dfmap_resource_v2.py"
VERIFIER_REL = "scripts/verify_optimization_v3_cls_pos_global_strength_e79_dfmap_resource_v2.py"
TEST_REL = "tests/test_optimization_v3_cls_pos_global_strength_e79_dfmap_resource_v2.py"
FREEZE_REL = "reports/optimization_v3/cls_pos_global_strength_e79_v3/DFMAP_RESOURCE_V2_FREEZE.json"
V1_FREEZE_REL = v1.FREEZE_REL
V1_FREEZE_SHA256 = "3eaf12bbd5064b91e4dca45c7ff0c1c27de8b1e4b573ae5242360aaf933915dc"
R3_RECEIPT_REL = "experiments/runs/optimization_v3_cls_pos_global_strength_e79_v3/descendant_replay_v1/REPLAY_RECEIPT.json"
R3_REPLAY_PEAK_BYTES = 556_793_856

OUTPUT_ROOT_REL = v1.OUTPUT_ROOT_REL
# The versioned V2 attempt owns the canonical V1 guard path. This makes an
# interrupted V2 run block the unchanged V1 runner from taking a second pass.
V2_ATTEMPT_NAME = v1.ATTEMPT_NAME
MAP_NAME = v1.MAP_NAME
RECEIPT_NAME = v1.RECEIPT_NAME

MIB = 1024**2
GIB = 1024**3
EXTRA_SPARSE_BOUND_LIMIT_BYTES = 160 * MIB
INCREMENTAL_ESTIMATE_BYTES = 768 * MIB
FREE_RESERVE_BYTES = 768 * MIB
MIN_AVAILABLE_BYTES = INCREMENTAL_ESTIMATE_BYTES + FREE_RESERVE_BYTES
MAX_THREADS = 2

SOURCE_FILES = (POLICY_REL, WRAPPER_REL, VERIFIER_REL, TEST_REL)


class ResourceV2Error(RuntimeError):
    """Raised when the V2 resource or one-shot source contract fails."""


def _sha256_bytes(payload: bytes) -> str:
    import hashlib
    return hashlib.sha256(payload).hexdigest()


def _stable_file_sha256(path: Path, relative: str) -> str:
    if Path(relative).suffix.lower() in {".json", ".md"}:
        return _sha256_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
    return v1.sha256_file(path)


def _canonical_sha256(value: Any) -> str:
    return v1.canonical_sha256(value)


def _git(root: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(["git", "-C", str(root), *args], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, check=False)
    if check and result.returncode:
        raise ResourceV2Error(result.stderr.decode("utf-8", errors="replace").strip())
    return result.stdout.decode("utf-8", errors="strict").strip()


def sparse_union_extra_upper_bound_bytes(
    *, rows: int = r3.EXPECTED_ROWS, nnz: int = r3.POSITION_NNZ,
    index_bytes: int = 8, value_bytes: int = 8,
) -> dict[str, int]:
    """Conservatively bound copy/union CSR arrays using 64-bit indices/values.

    The bound covers one full CSR copy, three block copies plus row pointers,
    three full-size uint8 union/intermediate CSR arrays, and three coordinate
    scratch vectors. ``nnz`` is the frozen total nonzero count; each live
    union/intermediate is bounded by it because positional union output cannot
    contain more row-token pairs than the input's row-position-token entries.
    """
    if rows < 1 or nnz < 1 or index_bytes not in (4, 8) or value_bytes not in (4, 8):
        raise ValueError("Invalid sparse resource-bound dimensions or dtypes")
    indptr = (rows + 1) * index_bytes
    matrix_copy = nnz * (value_bytes + index_bytes) + indptr
    block_copies = nnz * (value_bytes + index_bytes) + 3 * indptr
    union_intermediates = 3 * (nnz * (1 + index_bytes) + indptr)
    coordinate_scratch = 3 * nnz * index_bytes
    total = matrix_copy + block_copies + union_intermediates + coordinate_scratch
    return {
        "matrix_copy_bytes": matrix_copy,
        "three_block_copies_bytes": block_copies,
        "union_intermediates_bytes": union_intermediates,
        "coordinate_scratch_bytes": coordinate_scratch,
        "total_bytes": total,
    }


def resource_snapshot(stage: str, *, available_bytes: int | None = None,
                      process_rss_bytes: int | None = None) -> dict[str, Any]:
    vm = psutil.virtual_memory()
    proc = psutil.Process()
    available = int(vm.available if available_bytes is None else available_bytes)
    rss = int(proc.memory_info().rss if process_rss_bytes is None else process_rss_bytes)
    peak = getattr(proc.memory_info(), "peak_wset", None)
    if peak is None:
        peak = getattr(proc.memory_info(), "peak_rss", None)
    extra_bound = sparse_union_extra_upper_bound_bytes()
    memory_bound_pass = extra_bound["total_bytes"] <= EXTRA_SPARSE_BOUND_LIMIT_BYTES
    projected = available - INCREMENTAL_ESTIMATE_BYTES
    return {
        "stage": stage,
        "available_ram_bytes": available,
        "process_rss_bytes": rss,
        "process_peak_rss_bytes": int(peak) if peak is not None else None,
        "r3_measured_peak_rss_bytes": R3_REPLAY_PEAK_BYTES,
        "sparse_union_extra_upper_bound": extra_bound,
        "sparse_union_bound_limit_bytes": EXTRA_SPARSE_BOUND_LIMIT_BYTES,
        "incremental_memory_estimate_bytes": INCREMENTAL_ESTIMATE_BYTES,
        "required_free_reserve_bytes": FREE_RESERVE_BYTES,
        "projected_free_bytes": projected,
        "minimum_available_bytes": MIN_AVAILABLE_BYTES,
        "memory_bound_pass": memory_bound_pass,
        "initial_ram_pass": available >= MIN_AVAILABLE_BYTES,
        "projected_reserve_pass": projected >= FREE_RESERVE_BYTES,
        "gate_pass": memory_bound_pass and available >= MIN_AVAILABLE_BYTES and projected >= FREE_RESERVE_BYTES,
        "estimate_is_measured_df_map_peak": False,
        "resource_capture_utc": datetime.now(timezone.utc).isoformat(),
    }


def source_freeze_body(root: Path = ROOT) -> dict[str, Any]:
    """Build source-only policy lineage; no Train columns or targets are read."""
    _attestation, v1_lineage = v1.verify_source_freeze(root)
    v1_freeze_path = root / V1_FREEZE_REL
    v1_freeze = json.loads(v1_freeze_path.read_text(encoding="utf-8"))
    v1_freeze_file_sha = v1.sha256_file(v1_freeze_path)
    if v1_freeze.get("freeze_sha256") != V1_FREEZE_SHA256:
        raise ResourceV2Error("Frozen V1 DF-map source must remain byte-identical")
    ensure_one_global_pass_not_started(root)
    r3_receipt_path = root / R3_RECEIPT_REL
    if not r3_receipt_path.is_file():
        raise ResourceV2Error("Verified R3 descendant replay receipt is missing")
    r3_receipt = json.loads(r3_receipt_path.read_text(encoding="utf-8"))
    if (v1.sha256_file(r3_receipt_path) != v1_freeze.get("r3_receipt_file_sha256")
            or r3_receipt.get("resource_after", {}).get("process_peak_rss_bytes") != R3_REPLAY_PEAK_BYTES
            or r3_receipt.get("target_column_requested_or_loaded") is not False
            or r3_receipt.get("test_or_sample_loaded") is not False):
        raise ResourceV2Error("R3 replay resource/source-boundary evidence differs")
    sparse_extra_bound = sparse_union_extra_upper_bound_bytes()["total_bytes"]
    if R3_REPLAY_PEAK_BYTES + sparse_extra_bound > INCREMENTAL_ESTIMATE_BYTES:
        raise ResourceV2Error("R3 measured peak plus sparse-copy upper bound exceeds the V2 768 MiB estimate")
    return {
        "schema_version": 1,
        "policy_id": POLICY_ID,
        "policy_status": "FROZEN_FOR_ONE_GLOBAL_DF_MAP_PASS",
        "prepared_from_head": _git(root, "rev-parse", "HEAD"),
        "source_file_sha256": {path: _stable_file_sha256(root / path, path) for path in SOURCE_FILES},
        "v1_dfmap_freeze_sha256": V1_FREEZE_SHA256,
        "v1_dfmap_freeze_file_sha256": v1_freeze_file_sha,
        "v1_source_freeze_sha256": v1_lineage["freeze_sha256"],
        "r3_receipt_path": R3_RECEIPT_REL,
        "r3_receipt_file_sha256": v1.sha256_file(r3_receipt_path),
        "r3_measured_peak_rss_bytes": R3_REPLAY_PEAK_BYTES,
        "formula": dict(v1.EXPECTED_FORMULA),
        "train_sha256": r3.TRAIN_SHA256,
        "train_rows": r3.EXPECTED_ROWS,
        "position_shape": list(r3.POSITION_SHAPE),
        "position_nnz": r3.POSITION_NNZ,
        "position_logical_sha256": r3.POSITION_SHA256,
        "resource_policy": {
            "sparse_union_extra_bound_limit_bytes": EXTRA_SPARSE_BOUND_LIMIT_BYTES,
            "incremental_estimate_bytes": INCREMENTAL_ESTIMATE_BYTES,
            "free_reserve_bytes": FREE_RESERVE_BYTES,
            "minimum_initial_available_bytes": MIN_AVAILABLE_BYTES,
            "max_threads": MAX_THREADS,
            "sparse_union_extra_upper_bound_bytes": sparse_extra_bound,
            "r3_measured_peak_plus_sparse_bound_under_estimate": R3_REPLAY_PEAK_BYTES + sparse_extra_bound <= INCREMENTAL_ESTIMATE_BYTES,
            "dfmap_peak_measurement_claimed": False,
        },
        "output_root": OUTPUT_ROOT_REL,
        "global_fit_limit_across_resource_versions": 1,
        "test_sample_targets_fit_or_kaggle_access": False,
    }


def build_freeze(root: Path = ROOT) -> dict[str, Any]:
    """Freeze the V2 policy after its implementation, tests, and note commit."""
    status = _git(root, "status", "--porcelain", "--untracked-files=no")
    if status:
        raise ResourceV2Error(f"Tracked worktree must be clean before freezing V2: {status}")
    body = source_freeze_body(root)
    head = _git(root, "rev-parse", "HEAD")
    for rel, expected in body["source_file_sha256"].items():
        committed = _git(root, "show", f"{head}:{rel}", check=False)
        if not committed:
            raise ResourceV2Error(f"V2 source file is not committed: {rel}")
        blob = subprocess.run(["git", "-C", str(root), "show", f"{head}:{rel}"], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, check=False)
        if blob.returncode or _sha256_bytes(blob.stdout.replace(b"\r\n", b"\n") if Path(rel).suffix.lower() in {".json", ".md"} else blob.stdout) != expected:
            raise ResourceV2Error(f"V2 committed source hash differs: {rel}")
    payload = dict(body)
    payload["freeze_sha256"] = _canonical_sha256(body)
    path = root / FREEZE_REL
    encoded = (json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode()
    if path.exists():
        if path.read_bytes() != encoded:
            raise ResourceV2Error("Existing V2 freeze differs; refusing overwrite")
    else:
        r3._write_once(path, encoded)
    return payload


def verify_source_freeze(root: Path = ROOT) -> dict[str, Any]:
    path = root / FREEZE_REL
    freeze = json.loads(path.read_text(encoding="utf-8"))
    body = dict(freeze)
    digest = body.pop("freeze_sha256", None)
    if digest != _canonical_sha256(body):
        raise ResourceV2Error("V2 freeze self-digest mismatch")
    source_head = freeze.get("prepared_from_head")
    head = _git(root, "rev-parse", "HEAD")
    if not source_head or subprocess.run(["git", "-C", str(root), "merge-base", "--is-ancestor", source_head, head], check=False).returncode:
        raise ResourceV2Error("V2 frozen source commit is not an ancestor of HEAD")
    for rel, expected in freeze["source_file_sha256"].items():
        actual = _stable_file_sha256(root / rel, rel)
        if actual != expected:
            raise ResourceV2Error(f"V2 source changed: {rel}")
        tree = subprocess.run(["git", "-C", str(root), "show", f"{source_head}:{rel}"], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, check=False)
        if tree.returncode:
            raise ResourceV2Error(f"V2 frozen source is absent from commit: {rel}")
        tree_hash = _sha256_bytes(tree.stdout.replace(b"\r\n", b"\n") if Path(rel).suffix.lower() in {".json", ".md"} else tree.stdout)
        if tree_hash != expected:
            raise ResourceV2Error(f"V2 commit source changed: {rel}")
    file_hash = v1.sha256_file(path)
    tree = subprocess.run(["git", "-C", str(root), "show", f"{head}:{FREEZE_REL}"], stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=False)
    if tree.returncode or _sha256_bytes(tree.stdout) != file_hash:
        raise ResourceV2Error("Committed V2 freeze differs from the working-tree file")
    v1.verify_source_freeze(root)
    if (freeze.get("v1_dfmap_freeze_sha256") != V1_FREEZE_SHA256
            or freeze.get("v1_dfmap_freeze_file_sha256") != v1.sha256_file(root / v1.FREEZE_REL)
            or freeze.get("formula") != v1.EXPECTED_FORMULA
            or freeze.get("resource_policy", {}).get("incremental_estimate_bytes") != INCREMENTAL_ESTIMATE_BYTES
            or freeze.get("resource_policy", {}).get("free_reserve_bytes") != FREE_RESERVE_BYTES):
        raise ResourceV2Error("V2 freeze changed V1 inputs/formula or the approved memory reserve")
    return {"status": "PASS_DFMAP_RESOURCE_V2_SOURCE_FREEZE", "freeze_sha256": digest,
            "source_commit": source_head, "v1_freeze_sha256": V1_FREEZE_SHA256,
            "test_sample_targets_fit_or_kaggle_access": False}


def _output_paths(root: Path) -> tuple[Path, Path, Path, Path]:
    out = root / OUTPUT_ROOT_REL
    attempt = out / V2_ATTEMPT_NAME
    return attempt, attempt, out / MAP_NAME, out / RECEIPT_NAME


def ensure_one_global_pass_not_started(root: Path) -> None:
    v2_attempt, v1_attempt, map_path, receipt_path = _output_paths(root)
    existing = [path.name for path in dict.fromkeys((v2_attempt, v1_attempt, map_path, receipt_path)) if path.exists()]
    if existing:
        raise ResourceV2Error(f"Global V1/V2 DF-map output already exists; no retry or second pass: {existing}")


def _build_position(root: Path) -> tuple[Any, Any, list[str], dict[str, Any]]:
    from scripts import optimization_v3_a02_cgeom_benchmark_v1 as cgeom

    train, columns = r3.read_feature_csv(root / r3.TRAIN_REL, cgeom.CLASSIFICATION_FEATURES)
    if (len(train) != r3.EXPECTED_ROWS or train.record_id.isna().any()
            or train.record_id.duplicated().any()):
        raise ResourceV2Error("Train feature rows/IDs differ from the frozen population")
    cycles = np.asarray(train.cycle_code, dtype=np.int64)
    if tuple(sorted(np.unique(cycles).tolist())) != r3.EXPECTED_CYCLES:
        raise ResourceV2Error("Train feature cycles differ from the frozen 47-79 population")
    if positional.sha256_array(train.record_id.astype(str).to_numpy(dtype=str)) != r3.ORDERED_TRAIN_IDS_SHA256:
        raise ResourceV2Error("Train feature ID order differs from the R3 replay")
    if {"artifact_signal", "threat_rank"} & set(train.columns):
        raise ResourceV2Error("Feature reader unexpectedly loaded a target")
    model = __import__("joblib").load(root / r3.A02_MODEL_REL)
    vectorizer = model.text_space.vectorizer
    documents, _missing = classification_models._text_values(train["signal_stream"])
    position, audit = positional.build_position_block(documents, vectorizer, shuffled=False)
    actual = {
        "shape": list(position.shape), "nnz": int(position.nnz),
        "logical_sha256": positional.sha256_csr(position),
        "data_dtype": str(position.data.dtype), "data_itemsize": int(position.data.itemsize),
        "indices_dtype": str(position.indices.dtype), "indices_itemsize": int(position.indices.itemsize),
        "indptr_dtype": str(position.indptr.dtype), "indptr_itemsize": int(position.indptr.itemsize),
        "csr_array_bytes": int(position.data.nbytes + position.indices.nbytes + position.indptr.nbytes),
    }
    if (actual["shape"] != list(r3.POSITION_SHAPE) or actual["nnz"] != r3.POSITION_NNZ
            or actual["logical_sha256"] != r3.POSITION_SHA256
            or actual["data_dtype"] != "float64"
            or position.indices.dtype not in (np.dtype("int32"), np.dtype("int64"))
            or position.indptr.dtype not in (np.dtype("int32"), np.dtype("int64"))):
        raise ResourceV2Error(f"Position matrix failed frozen geometry/dtype checks: {actual}")
    bound = sparse_union_extra_upper_bound_bytes(
        rows=position.shape[0], nnz=position.nnz,
        index_bytes=max(position.indices.itemsize, position.indptr.itemsize),
        value_bytes=position.data.itemsize,
    )
    actual["sparse_union_extra_upper_bound"] = bound
    if bound["total_bytes"] > EXTRA_SPARSE_BOUND_LIMIT_BYTES:
        raise ResourceV2Error(f"Actual position dtype/shape exceeds the approved 160 MiB union bound: {actual}")
    return train, position, columns, {"position_matrix": actual, "position_build_audit": audit}


def _execute_map(root: Path, *, locks: list[dict[str, Any]], resource_before: dict[str, Any],
                 freeze_sha256: str, map_path: Path) -> dict[str, Any]:
    started = datetime.now(timezone.utc).isoformat()
    train, position, train_columns, matrix_audit = _build_position(root)
    model = __import__("joblib").load(root / r3.A02_MODEL_REL)
    token_order = v1.ordered_unigrams(model.text_space.vectorizer)
    if len(token_order) * 3 != position.shape[1]:
        raise ResourceV2Error("Saved vectorizer unigram order differs from positional feature width")
    token_order_sha = v1.canonical_sha256([[term, index] for term, index in token_order])
    ids_sha = positional.sha256_array(train.record_id.astype(str).to_numpy(dtype=str))
    mapped = v1.compute_df_map(position, seed=v1.PERMUTATION_SEED)
    arrays = {key: value for key, value in mapped.items() if key != "statistics"}
    artifact_sha = v1._save_npz_once(map_path, arrays)
    resource_after = resource_snapshot("after one Train-only DF-map pass")
    if resource_after["available_ram_bytes"] < FREE_RESERVE_BYTES:
        raise ResourceV2Error("Available memory fell below the unchanged 768 MiB reserve")
    v1_freeze = json.loads((root / V1_FREEZE_REL).read_text(encoding="utf-8"))
    r3_receipt_path = root / R3_RECEIPT_REL
    return {
        "schema_version": 1,
        "policy_id": POLICY_ID,
        "dfmap_id": v1.DFMAP_ID,
        "status": "PASS_E79_DF_MAP_CREATED_TRAIN_ONLY",
        "started_at_utc": started,
        "source_invocation_head": _git(root, "rev-parse", "HEAD"),
        "dfmap_resource_v2_freeze_sha256": freeze_sha256,
        "dfmap_freeze_sha256": freeze_sha256,
        "v1_freeze_sha256": V1_FREEZE_SHA256,
        "r3_receipt_file_sha256": v1.sha256_file(r3_receipt_path),
        "v1_source_file_sha256": v1_freeze.get("continuation_file_sha256"),
        "input_sha256": v1_freeze.get("immutable_input_sha256"),
        "locks": locks,
        "resource_before": resource_before,
        "resource_after": resource_after,
        "resource_rationale": {
            "r3_measured_peak_rss_bytes": R3_REPLAY_PEAK_BYTES,
            "sparse_union_extra_upper_bound_bytes": matrix_audit["position_matrix"]["sparse_union_extra_upper_bound"]["total_bytes"],
            "incremental_estimate_bytes": INCREMENTAL_ESTIMATE_BYTES,
            "free_reserve_bytes": FREE_RESERVE_BYTES,
            "dfmap_peak_measurement_claimed": False,
        },
        "train_path": r3.TRAIN_REL,
        "train_sha256": r3.TRAIN_SHA256,
        "train_rows": len(train),
        "ordered_train_ids_sha256": ids_sha,
        "train_columns_requested": train_columns,
        "target_column_requested_or_loaded": False,
        "test_or_sample_loaded": False,
        "estimator_fit_calls": 0,
        "prediction_or_inference_calls": 0,
        "vectorizer_fit_calls": 0,
        "structured_preprocessor_fit_calls": 0,
        "newly_learned_preprocessing_fit_calls": 1,
        "df_map_formula_fit_calls": 1,
        "position_matrix": matrix_audit["position_matrix"],
        "position_build_audit": matrix_audit["position_build_audit"],
        "unigram_vocabulary_order_sha256": token_order_sha,
        "permutation_seed": v1.PERMUTATION_SEED,
        "formula": dict(v1.EXPECTED_FORMULA),
        "statistics": mapped["statistics"],
        "artifact_path": map_path.relative_to(root).as_posix(),
        "artifact_sha256": artifact_sha,
        "csv_created": False,
        "kaggle_contacted": False,
        "v1_runner_invoked": False,
        "global_v1_v2_map_fit_count": 1,
    }


def run_df_map(root: Path = ROOT) -> dict[str, Any]:
    lineage = verify_source_freeze(root)
    ensure_one_global_pass_not_started(root)
    v2_attempt, _v1_attempt, map_path, receipt_path = _output_paths(root)
    locks: list[dict[str, Any]] = []
    with ExitStack() as stack:
        locks.append(stack.enter_context(r3._shared_job_lock()))
        locks.append(stack.enter_context(r3._cgeom_local_lock(root)))
        ensure_one_global_pass_not_started(root)
        resource_before = resource_snapshot("immediately before one V2 DF-map attempt")
        # Independent preflight logic evaluates this one locked snapshot. The
        # same lock remains held from this check through the map transaction.
        from scripts.verify_optimization_v3_cls_pos_global_strength_e79_dfmap_resource_v2 import (
            verify_resource_preflight,
        )
        resource_preflight = verify_resource_preflight(resource_before, locks)
        if resource_preflight["status"] != "PASS_INDEPENDENT_RESOURCE_PREFLIGHT":
            return {"status": "BLOCKED_RESOURCE_GATE", "policy_id": POLICY_ID,
                    "lineage": lineage, "locks": locks, "resource_before": resource_before,
                    "independent_resource_preflight": resource_preflight,
                    "attempt_marker_written": False, "map_written": False, "receipt_written": False}
        attempt_body = {
            "schema_version": 1, "policy_id": POLICY_ID, "dfmap_id": v1.DFMAP_ID,
            "status": "ONE_GLOBAL_DF_MAP_ATTEMPT_STARTED_V2",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_invocation_head": _git(root, "rev-parse", "HEAD"),
            "v2_freeze_sha256": lineage["freeze_sha256"], "v1_freeze_sha256": V1_FREEZE_SHA256,
            "locks": locks, "resource_before": resource_before,
            "independent_resource_preflight": resource_preflight,
            "no_retry_across_v1_or_v2": True,
        }
        r3._write_once(v2_attempt,
            (json.dumps(attempt_body, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode())
        started = time.perf_counter()
        try:
            with threadpool_limits(limits=MAX_THREADS):
                body = _execute_map(root, locks=locks, resource_before=resource_before,
                                    freeze_sha256=lineage["freeze_sha256"], map_path=map_path)
            body["independent_resource_preflight"] = resource_preflight
        except Exception as exc:  # one global attempt: persist terminal failure and never retry
            body = {
                "schema_version": 1, "policy_id": POLICY_ID, "dfmap_id": v1.DFMAP_ID,
                "status": "FAILED_DF_MAP_NO_RETRY", "started_at_utc": attempt_body["started_at_utc"],
                "dfmap_resource_v2_freeze_sha256": lineage["freeze_sha256"],
                "dfmap_freeze_sha256": lineage["freeze_sha256"], "v1_freeze_sha256": V1_FREEZE_SHA256,
                "target_column_requested_or_loaded": False, "test_or_sample_loaded": False,
                "estimator_fit_calls": 0, "prediction_or_inference_calls": 0,
                "vectorizer_fit_calls": 0, "structured_preprocessor_fit_calls": 0,
                "newly_learned_preprocessing_fit_calls": 1, "df_map_formula_fit_calls": 1,
                "csv_created": False, "kaggle_contacted": False, "v1_runner_invoked": False,
                "global_v1_v2_map_fit_count": 1,
                "resource_before": resource_before,
                "independent_resource_preflight": resource_preflight,
                "error_type": type(exc).__name__, "error": str(exc),
                "resource_after_failure": resource_snapshot("after terminal V2 DF-map failure"),
            }
        body["elapsed_seconds"] = float(time.perf_counter() - started)
        body["no_retry_across_v1_or_v2"] = True
        body["v2_attempt_path"] = v2_attempt.relative_to(root).as_posix()
        payload = dict(body)
        payload["receipt_sha256"] = _canonical_sha256(body)
        r3._write_once(receipt_path,
            (json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode())
        return payload


def run_synthetic_qa() -> dict[str, Any]:
    """Exercise the exact formula, persistence, resource gate, and no-retry guard."""
    import tempfile

    bound = sparse_union_extra_upper_bound_bytes()
    if bound["total_bytes"] > EXTRA_SPARSE_BOUND_LIMIT_BYTES:
        raise ResourceV2Error("Frozen sparse allocation upper bound exceeds 160 MiB")
    pass_snapshot = resource_snapshot("synthetic pass", available_bytes=2 * GIB, process_rss_bytes=1)
    fail_snapshot = resource_snapshot("synthetic fail", available_bytes=MIN_AVAILABLE_BYTES - 1, process_rss_bytes=1)
    if not pass_snapshot["gate_pass"] or fail_snapshot["gate_pass"]:
        raise ResourceV2Error("Synthetic resource gate failed boundary checks")
    # Two rows, one token in different positional blocks, plus a repeated token.
    position = sparse.csr_matrix(np.asarray([
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64))
    mapped = v1.compute_df_map(position, seed=v1.PERMUTATION_SEED)
    arrays = {key: value for key, value in mapped.items() if key != "statistics"}
    with tempfile.TemporaryDirectory(prefix="e79-dfmap-v2-synthetic-") as temp:
        root = Path(temp)
        attempt, v1_attempt, map_path, receipt_path = _output_paths(root)
        attempt.parent.mkdir(parents=True)
        attempt_body = {"policy_id": POLICY_ID, "status": "ONE_GLOBAL_DF_MAP_ATTEMPT_STARTED_V2"}
        r3._write_once(attempt, (json.dumps(attempt_body, sort_keys=True) + "\n").encode())
        artifact_sha = v1._save_npz_once(map_path, arrays)
        with np.load(map_path, allow_pickle=False) as archive:
            reopened = {key: np.asarray(archive[key]).copy() for key in archive.files}
        verify_result = v1_verify.verify_npz_arrays(
            map_path, expected_size=position.shape[1] // 3, expected_rows=position.shape[0],
            statistics=mapped["statistics"],
        )
        if artifact_sha != v1.sha256_file(map_path) or set(reopened) != set(arrays):
            raise ResourceV2Error("Synthetic persisted map failed read-back verification")
        r3._write_once(receipt_path, b"synthetic terminal receipt\n")
        try:
            ensure_one_global_pass_not_started(root)
        except ResourceV2Error:
            pass
        else:
            raise ResourceV2Error("Synthetic resume guard allowed a second V1/V2 map pass")
        try:
            r3._write_once(map_path, b"retry forbidden")
        except Exception:
            pass
        else:
            raise ResourceV2Error("Synthetic persistence helper overwrote the one-shot map")
        return {
            "status": "PASS_SYNTHETIC_RESOURCE_AND_PERSISTENCE_QA",
            "sparse_union_extra_upper_bound": bound,
            "synthetic_map_sha256": artifact_sha,
            "independent_formula_status": verify_result.get("status", "PASS"),
            "attempt_marker_persisted": attempt.is_file(),
            "v1_runner_attempted": False,
            "shared_guard_marker_policy_id": json.loads(attempt.read_text(encoding="utf-8"))["policy_id"],
            "second_global_fit_blocked": True,
            "test_or_sample_or_target_access": False,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--build-freeze", action="store_true")
    group.add_argument("--fit-df-map", action="store_true")
    group.add_argument("--synthetic-qa", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.build_freeze:
            result = build_freeze()
        elif args.fit_df_map:
            result = run_df_map()
        else:
            result = run_synthetic_qa()
    except (ResourceV2Error, v1.DFMapError, r3.ReplayError, v1_verify.VerificationError) as exc:
        print(json.dumps({"status": "BLOCKED_OR_FAILED", "error": str(exc)}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2))
    return 0 if result.get("status", "").startswith("PASS") or result.get("status") == "BLOCKED_RESOURCE_GATE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
