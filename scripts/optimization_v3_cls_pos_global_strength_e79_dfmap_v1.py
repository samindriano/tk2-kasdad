"""Build the authorized deterministic E79 Train-only document-frequency map."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from scripts import optimization_v3_cls_pos_global_strength_e79_r3_descendant_replay_v1 as r3_cont
from scripts import optimization_v3_cls_pos_global_strength_e79_v3 as r3
from scripts import verify_optimization_v3_cls_pos_global_strength_e79_v3 as r3_verifier
from src.models import classification_models, optimization_v3_positional_text_pilot_v1 as positional
from src.models.pos_one_sided_ridge_v1 import build_penalty_vectors


ROOT = Path(__file__).resolve().parents[1]
DFMAP_ID = "CLS_POS_GLOBAL_STRENGTH_E79_DF_MAP_V1"
PROTOCOL_REL = "reports/optimization_v3/cls_pos_global_strength_e79_v3/DFMAP_CONTINUATION.md"
FREEZE_REL = "reports/optimization_v3/cls_pos_global_strength_e79_v3/DFMAP_FREEZE.json"
WRAPPER_REL = "scripts/optimization_v3_cls_pos_global_strength_e79_dfmap_v1.py"
VERIFIER_REL = "scripts/verify_optimization_v3_cls_pos_global_strength_e79_dfmap_v1.py"
TEST_REL = "tests/test_optimization_v3_cls_pos_global_strength_e79_dfmap_v1.py"
ATTRIBUTES_REL = ".gitattributes"
RECIPE_SOURCE_RELS = (
    "src/models/pos_one_sided_ridge_v1.py",
    "src/models/pos_frequency_ridge_v1.py",
)
CONTINUATION_FILE_PATHS = (
    ATTRIBUTES_REL, PROTOCOL_REL, WRAPPER_REL, VERIFIER_REL, TEST_REL, *RECIPE_SOURCE_RELS,
)
R3_OUTPUT_ROOT_REL = r3_cont.OUTPUT_ROOT_REL
OUTPUT_ROOT_REL = "experiments/runs/optimization_v3_cls_pos_global_strength_e79_v3/df_map_v1"
ATTEMPT_NAME = "ATTEMPT.started.json"
MAP_NAME = "E79_DF_MAP.npz"
RECEIPT_NAME = "DFMAP_RECEIPT.json"
PERMUTATION_SEED = 42
EXPECTED_FORMULA = {
    "df": "count each retained unigram at most once per Train row if present in any positional segment block",
    "raw": "clip(sqrt(median(df)/df), 0.5, 2.0)",
    "aligned": "raw / mean(raw)",
    "permutation": "numpy.default_rng(42).permutation(V); permuted_aligned=aligned[permutation]",
    "global_strength": "v=max(1,aligned); c=mean(v); penalty=full(3*V,c)",
}


class DFMapError(RuntimeError):
    """Raised when the frozen Train-only DF-map contract is not met."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_file_sha256(path: Path, relative: str) -> str:
    """Hash tracked text by LF-normalized bytes so Windows checkouts remain reproducible."""
    if Path(relative).suffix.lower() in {".md", ".json"}:
        return sha256_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
    return sha256_file(path)


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                   allow_nan=False, separators=(",", ":")).encode("utf-8"))


def _git(root: Path, *args: str, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args], check=False, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise DFMapError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout.decode("utf-8", errors="strict").strip()


def _git_blob_sha256(root: Path, commit: str, relative: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "show", f"{commit}:{relative}"],
        check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise DFMapError(f"Committed path unavailable at {commit}:{relative}: {detail}")
    return sha256_bytes(completed.stdout)


def verify_committed_paths(root: Path, commit: str,
                           expected_sha256: dict[str, str]) -> dict[str, str]:
    verified: dict[str, str] = {}
    for relative, expected in sorted(expected_sha256.items()):
        path = root / relative
        try:
            path.resolve().relative_to(root.resolve())
        except ValueError as exc:
            raise DFMapError(f"Frozen path escapes repository: {relative}") from exc
        if not path.is_file():
            raise DFMapError(f"Frozen path is missing: {relative}")
        worktree_hash = _stable_file_sha256(path, relative)
        tree_hash = _git_blob_sha256(root, commit, relative)
        if worktree_hash != expected or tree_hash != expected:
            raise DFMapError(
                f"Frozen path differs in worktree or commit: {relative}; "
                f"expected={expected}, worktree={worktree_hash}, tree={tree_hash}"
            )
        verified[relative] = expected
    return verified


def _tracked_worktree_clean(root: Path) -> None:
    status = _git(root, "status", "--porcelain", "--untracked-files=no")
    if status:
        raise DFMapError(f"Tracked worktree must be clean: {status}")


def _r3_receipt_path(root: Path) -> Path:
    return root / R3_OUTPUT_ROOT_REL / "REPLAY_RECEIPT.json"


def _verify_r3_replay(root: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    attestation, _lineage = r3_cont.verify_source_freeze(root)
    receipt_path = _r3_receipt_path(root)
    result = r3_verifier.verify_replay_receipt(root, receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("continuation_id") != r3_cont.CONTINUATION_ID
        or receipt.get("descendant_source_lineage_verified") is not True
        or receipt.get("target_column_requested_or_loaded") is not False
        or receipt.get("test_or_sample_loaded") is not False
        or receipt.get("estimator_fit_calls") != 0
        or receipt.get("prediction_or_inference_calls") != 0
        or receipt.get("new_df_map_created") is not False
        or result.get("status") != "PASS_INDEPENDENT_DATA_FREE_REPLAY_VERIFICATION"
    ):
        raise DFMapError("R3 replay receipt does not meet the authorized precondition")
    return attestation, result, receipt_path


def build_freeze(root: Path = ROOT) -> dict[str, Any]:
    """Freeze the map implementation and verified R3 replay before one map fit."""
    _tracked_worktree_clean(root)
    attestation, r3_result, r3_receipt_path = _verify_r3_replay(root)
    prepared_head = _git(root, "rev-parse", "HEAD")
    code_hashes = {
        relative: _stable_file_sha256(root / relative, relative)
        for relative in CONTINUATION_FILE_PATHS
    }
    verify_committed_paths(root, prepared_head, code_hashes)
    r3_receipt = json.loads(r3_receipt_path.read_text(encoding="utf-8"))
    body = {
        "schema_version": 1,
        "dfmap_id": DFMAP_ID,
        "status": "FROZEN_FOR_ONE_TRAIN_ONLY_DF_MAP_FIT",
        "prepared_from_head": prepared_head,
        "r3_attestation_sha256": attestation["attestation_sha256"],
        "r3_attestation_file_sha256": sha256_file(root / r3.ATTESTATION_REL),
        "r3_receipt_path": r3_receipt_path.relative_to(root).as_posix(),
        "r3_receipt_file_sha256": sha256_file(r3_receipt_path),
        "r3_receipt_sha256": r3_receipt["receipt_sha256"],
        "r3_independent_verification": r3_result,
        "r3_current_source_sha256": dict(attestation["current_source_sha256"]),
        "immutable_input_sha256": dict(attestation["saved_inputs_and_fitted_state_receipts"]["immutable_file_sha256"]),
        "continuation_file_sha256": code_hashes,
        "official_train_path": r3.TRAIN_REL,
        "official_train_sha256": r3.TRAIN_SHA256,
        "official_train_rows": r3.EXPECTED_ROWS,
        "feature_columns_only": True,
        "forbidden_target_columns": ["artifact_signal", "threat_rank"],
        "test_and_sample_forbidden": True,
        "estimator_fit_forbidden": True,
        "vectorizer_refit_forbidden": True,
        "formula": EXPECTED_FORMULA,
        "output_root": OUTPUT_ROOT_REL,
    }
    payload = dict(body)
    payload["freeze_sha256"] = canonical_sha256(body)
    path = root / FREEZE_REL
    if path.exists():
        raise DFMapError(f"Refusing to replace existing DF-map freeze: {FREEZE_REL}")
    r3._write_once(path, (json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8"))
    return payload


def verify_source_freeze(root: Path = ROOT) -> tuple[dict[str, Any], dict[str, Any]]:
    _tracked_worktree_clean(root)
    head = _git(root, "rev-parse", "HEAD")
    freeze_path = root / FREEZE_REL
    if not freeze_path.is_file():
        raise DFMapError(f"DF-map freeze missing: {FREEZE_REL}")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    body = dict(freeze)
    supplied = body.pop("freeze_sha256", None)
    if supplied != canonical_sha256(body):
        raise DFMapError("DF-map freeze self-digest is invalid")
    if freeze.get("dfmap_id") != DFMAP_ID or freeze.get("status") != "FROZEN_FOR_ONE_TRAIN_ONLY_DF_MAP_FIT":
        raise DFMapError("Unexpected DF-map freeze identity or status")
    prepared_head = freeze.get("prepared_from_head")
    if not isinstance(prepared_head, str) or _git(root, "rev-parse", "--verify", f"{prepared_head}^{{commit}}", check=False) == "":
        raise DFMapError("Freeze prepared_from_head is not a repository commit")
    ancestry = subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", prepared_head, head],
        check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if ancestry.returncode != 0:
        raise DFMapError("DF-map prepared source commit is not an ancestor of HEAD")
    attestation, r3_result, receipt_path = _verify_r3_replay(root)
    if (
        freeze.get("r3_attestation_sha256") != attestation.get("attestation_sha256")
        or freeze.get("r3_attestation_file_sha256") != sha256_file(root / r3.ATTESTATION_REL)
        or freeze.get("r3_receipt_path") != receipt_path.relative_to(root).as_posix()
        or freeze.get("r3_receipt_file_sha256") != sha256_file(receipt_path)
        or freeze.get("r3_receipt_sha256") != json.loads(receipt_path.read_text(encoding="utf-8")).get("receipt_sha256")
        or freeze.get("r3_current_source_sha256") != attestation.get("current_source_sha256")
        or freeze.get("immutable_input_sha256") != attestation["saved_inputs_and_fitted_state_receipts"]["immutable_file_sha256"]
        or r3_result.get("status") != "PASS_INDEPENDENT_DATA_FREE_REPLAY_VERIFICATION"
    ):
        raise DFMapError("R3 replay result or immutable inputs differ from the DF-map freeze")
    code_hashes = freeze.get("continuation_file_sha256")
    if not isinstance(code_hashes, dict) or set(code_hashes) != set(CONTINUATION_FILE_PATHS):
        raise DFMapError("DF-map freeze does not bind the exact implementation and recipe source files")
    verified = verify_committed_paths(root, head, code_hashes)
    verify_committed_paths(root, prepared_head, code_hashes)
    verify_committed_paths(root, head, {FREEZE_REL: _stable_file_sha256(freeze_path, FREEZE_REL)})
    if (
        freeze.get("official_train_path") != r3.TRAIN_REL
        or freeze.get("official_train_sha256") != r3.TRAIN_SHA256
        or freeze.get("official_train_rows") != r3.EXPECTED_ROWS
        or freeze.get("feature_columns_only") is not True
        or freeze.get("forbidden_target_columns") != ["artifact_signal", "threat_rank"]
        or freeze.get("test_and_sample_forbidden") is not True
        or freeze.get("estimator_fit_forbidden") is not True
        or freeze.get("vectorizer_refit_forbidden") is not True
        or freeze.get("output_root") != OUTPUT_ROOT_REL
        or freeze.get("formula") != EXPECTED_FORMULA
    ):
        raise DFMapError("DF-map freeze does not preserve the Train-only preprocessing boundary")
    return attestation, {
        "head": head,
        "prepared_from_head": prepared_head,
        "freeze_sha256": supplied,
        "r3_receipt_sha256": freeze["r3_receipt_sha256"],
        "continuation_files_verified": len(verified),
        "r3_source_files_verified": len(attestation["current_source_sha256"]),
        "immutable_input_files_verified": len(freeze["immutable_input_sha256"]),
    }


def ordered_unigrams(vectorizer: Any) -> list[tuple[str, int]]:
    return [
        (term, int(index))
        for term, index in sorted(vectorizer.vocabulary_.items(), key=lambda item: item[1])
        if " " not in term
    ]


def union_document_frequency(position: Any) -> np.ndarray:
    """Count each feature once per row across the three positional blocks."""
    from scipy import sparse

    matrix = sparse.csr_matrix(position, copy=True)
    matrix.sum_duplicates()
    if matrix.shape[1] == 0 or matrix.shape[1] % 3:
        raise DFMapError("Position matrix must contain exactly three equal positional blocks")
    vocabulary_size = matrix.shape[1] // 3
    blocks = []
    for segment in range(3):
        block = matrix[:, segment * vocabulary_size:(segment + 1) * vocabulary_size].copy().tocsr()
        block.data = np.ones(block.nnz, dtype=np.uint8)
        blocks.append(block)
    union = blocks[0] + blocks[1] + blocks[2]
    union.sum_duplicates()
    union.eliminate_zeros()
    union.data = np.ones(union.nnz, dtype=np.uint8)
    df = np.asarray(union.sum(axis=0), dtype=np.int64).reshape(-1)
    if df.shape != (vocabulary_size,) or np.any(df <= 0) or np.any(df > matrix.shape[0]):
        raise DFMapError("Document frequencies must be positive and bounded by Train row count")
    return df


def compute_df_map(position: Any, *, seed: int = PERMUTATION_SEED) -> dict[str, Any]:
    df = union_document_frequency(position)
    median_df = float(np.median(df))
    raw = np.clip(np.sqrt(median_df / df.astype(np.float64)), 0.5, 2.0)
    aligned = raw / float(np.mean(raw, dtype=np.float64))
    permutation = np.random.default_rng(seed).permutation(len(df)).astype(np.int64)
    permuted = aligned[permutation]
    penalties = build_penalty_vectors(aligned)
    scalar = float(penalties["global_strength_scalar"])
    global_position_penalty = penalties["global_strength_position_weights"].astype(np.float64, copy=False)
    expected_penalty = np.full(3 * len(df), scalar, dtype=np.float64)
    if not np.array_equal(global_position_penalty, expected_penalty):
        raise DFMapError("Original global-strength recipe did not produce a repeated scalar penalty")
    if not np.isfinite(raw).all() or not np.isfinite(aligned).all() or not np.isfinite(permuted).all():
        raise DFMapError("DF-map formula produced non-finite values")
    return {
        "document_frequency_int64": df.astype(np.int64, copy=False),
        "raw_clipped_weights_float64": raw.astype(np.float64, copy=False),
        "aligned_weights_float64": aligned.astype(np.float64, copy=False),
        "permuted_aligned_weights_float64": permuted.astype(np.float64, copy=False),
        "global_strength_position_penalty_float64": global_position_penalty,
        "global_strength_scalar_float64": np.asarray(scalar, dtype=np.float64),
        "statistics": {
            "retained_unigram_vocabulary_size": int(len(df)),
            "train_rows": int(position.shape[0]),
            "median_df": median_df,
            "df_min": int(df.min()),
            "df_max": int(df.max()),
            "aligned_mean": float(np.mean(aligned, dtype=np.float64)),
            "aligned_min": float(aligned.min()),
            "aligned_max": float(aligned.max()),
            "raw_clip_low_count": int(np.count_nonzero(raw <= 0.5)),
            "raw_clip_high_count": int(np.count_nonzero(raw >= 2.0)),
            "permuted_aligned_sha256": sha256_bytes(np.ascontiguousarray(permuted, dtype="<f8").tobytes()),
            "global_strength_scalar": scalar,
            "global_position_penalty_size": int(global_position_penalty.size),
            "global_position_penalty_unique_values": int(np.unique(global_position_penalty).size),
        },
    }


def _save_npz_once(path: Path, arrays: dict[str, Any]) -> str:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    payload = buffer.getvalue()
    r3._write_once(path, payload)
    return sha256_bytes(payload)


def _output_paths(root: Path) -> tuple[Path, Path, Path]:
    out_root = root / OUTPUT_ROOT_REL
    return out_root / ATTEMPT_NAME, out_root / MAP_NAME, out_root / RECEIPT_NAME


def _read_train_features(root: Path) -> tuple[Any, list[str]]:
    from scripts import optimization_v3_a02_cgeom_benchmark_v1 as cgeom

    train, columns = r3.read_feature_csv(root / r3.TRAIN_REL, cgeom.CLASSIFICATION_FEATURES)
    if len(train) != r3.EXPECTED_ROWS or train.record_id.isna().any() or train.record_id.duplicated().any():
        raise DFMapError("Official Train feature rows or record IDs differ from the frozen population")
    cycles = np.asarray(train.cycle_code, dtype=np.int64)
    if tuple(sorted(np.unique(cycles).tolist())) != r3.EXPECTED_CYCLES:
        raise DFMapError("Official Train cycles differ from the frozen 47-79 feature population")
    ids_sha = positional.sha256_array(train.record_id.astype(str).to_numpy(dtype=str))
    if ids_sha != r3.ORDERED_TRAIN_IDS_SHA256:
        raise DFMapError("Official Train record ID order differs from the saved R3 replay")
    if "artifact_signal" in train.columns or "threat_rank" in train.columns:
        raise DFMapError("Target column leaked into DF-map feature read")
    return train, columns


def _execute_df_map(root: Path, attestation: dict[str, Any], lineage: dict[str, Any],
                    locks: list[dict[str, Any]], resource_before: dict[str, Any],
                    map_path: Path) -> dict[str, Any]:
    started = datetime.now(timezone.utc).isoformat()
    train, train_columns = _read_train_features(root)
    model_path = root / r3.A02_MODEL_REL
    model = joblib.load(model_path)
    vectorizer = model.text_space.vectorizer
    documents, _missing_mask = classification_models._text_values(train["signal_stream"])
    position, position_audit = positional.build_position_block(documents, vectorizer, shuffled=False)
    actual_matrix = {
        "shape": list(position.shape),
        "nnz": int(position.nnz),
        "logical_sha256": positional.sha256_csr(position),
    }
    if (
        actual_matrix["shape"] != list(r3.POSITION_SHAPE)
        or actual_matrix["nnz"] != r3.POSITION_NNZ
        or actual_matrix["logical_sha256"] != r3.POSITION_SHA256
        or int(position_audit["unigram_vocabulary_size"]) * 3 != position.shape[1]
    ):
        raise DFMapError("Rebuilt positional matrix differs from the verified R3 source matrix")
    token_order = ordered_unigrams(vectorizer)
    if len(token_order) != position.shape[1] // 3:
        raise DFMapError("Saved A02 unigram order and positional width differ")
    token_order_sha256 = canonical_sha256([[term, index] for term, index in token_order])
    ordered_train_ids_sha256 = positional.sha256_array(train.record_id.astype(str).to_numpy(dtype=str))
    mapped = compute_df_map(position, seed=PERMUTATION_SEED)
    arrays = {key: value for key, value in mapped.items() if key != "statistics"}
    artifact_sha = _save_npz_once(map_path, arrays)
    resource_after = r3._resource_snapshot("after Train-only E79 DF-map creation")
    if resource_after["available_ram_bytes"] < r3.FREE_RESERVE_BYTES:
        raise DFMapError("Available RAM fell below the frozen 768 MiB reserve after DF-map creation")
    return {
        "schema_version": 1,
        "dfmap_id": DFMAP_ID,
        "status": "PASS_E79_DF_MAP_CREATED_TRAIN_ONLY",
        "started_at_utc": started,
        "source_invocation_head": lineage["head"],
        "dfmap_freeze_sha256": lineage["freeze_sha256"],
        "r3_attestation_sha256": attestation["attestation_sha256"],
        "r3_receipt_file_sha256": lineage["r3_receipt_file_sha256"],
        "source_sha256": attestation["current_source_sha256"],
        "input_sha256": attestation["saved_inputs_and_fitted_state_receipts"]["immutable_file_sha256"],
        "locks": locks,
        "resource_before": resource_before,
        "resource_after": resource_after,
        "train_path": r3.TRAIN_REL,
        "train_sha256": r3.TRAIN_SHA256,
        "train_rows": int(len(train)),
        "ordered_train_ids_sha256": ordered_train_ids_sha256,
        "train_columns_requested": train_columns,
        "target_column_requested_or_loaded": False,
        "test_or_sample_loaded": False,
        "estimator_fit_calls": 0,
        "prediction_or_inference_calls": 0,
        "vectorizer_fit_calls": 0,
        "structured_preprocessor_fit_calls": 0,
        "newly_learned_preprocessing_fit_calls": 1,
        "df_map_formula_fit_calls": 1,
        "position_matrix": actual_matrix,
        "position_build_audit": position_audit,
        "unigram_vocabulary_order_sha256": token_order_sha256,
        "permutation_seed": PERMUTATION_SEED,
        "statistics": mapped["statistics"],
        "artifact_path": map_path.relative_to(root).as_posix(),
        "artifact_sha256": artifact_sha,
        "csv_created": False,
        "kaggle_contacted": False,
    }


def preflight(root: Path = ROOT) -> dict[str, Any]:
    attestation, lineage = verify_source_freeze(root)
    attempt_path, map_path, receipt_path = _output_paths(root)
    if any(path.exists() for path in (attempt_path, map_path, receipt_path)):
        raise DFMapError("DF-map one-shot outputs already exist; refusing retry")
    locks: list[dict[str, Any]] = []
    with ExitStack() as stack:
        locks.append(stack.enter_context(r3._shared_job_lock()))
        locks.append(stack.enter_context(r3._cgeom_local_lock(root)))
        resource = r3._resource_snapshot("E79 DF-map read-only preflight")
    return {
        "status": "PASS_PREFLIGHT" if resource["gate_pass"] else "BLOCKED_RESOURCE_GATE",
        "dfmap_id": DFMAP_ID,
        "lineage": lineage,
        "locks": locks,
        "resource": resource,
        "attempt_marker_exists": attempt_path.exists(),
        "map_exists": map_path.exists(),
        "receipt_exists": receipt_path.exists(),
        "no_train_features_or_map_fit_started": True,
        "r3_status": attestation.get("status"),
    }


def run_df_map(root: Path = ROOT) -> dict[str, Any]:
    attestation, lineage = verify_source_freeze(root)
    attempt_path, map_path, receipt_path = _output_paths(root)
    if any(path.exists() for path in (attempt_path, map_path, receipt_path)):
        raise DFMapError("DF-map one-shot outputs already exist; refusing retry")
    locks: list[dict[str, Any]] = []
    with ExitStack() as stack:
        locks.append(stack.enter_context(r3._shared_job_lock()))
        locks.append(stack.enter_context(r3._cgeom_local_lock(root)))
        resource_before = r3._resource_snapshot("immediately before one E79 DF-map fit")
        if not resource_before["gate_pass"]:
            return {
                "status": "BLOCKED_RESOURCE_GATE",
                "dfmap_id": DFMAP_ID,
                "lineage": lineage,
                "locks": locks,
                "resource_before": resource_before,
                "attempt_marker_written": False,
                "map_written": False,
                "receipt_written": False,
            }
        attempt_body = {
            "schema_version": 1,
            "dfmap_id": DFMAP_ID,
            "status": "ONE_DF_MAP_ATTEMPT_STARTED",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_invocation_head": lineage["head"],
            "freeze_sha256": lineage["freeze_sha256"],
            "r3_receipt_file_sha256": lineage["r3_receipt_file_sha256"],
            "locks": locks,
            "resource_before": resource_before,
            "no_retry_if_interrupted": True,
        }
        r3._write_once(
            attempt_path,
            (json.dumps(attempt_body, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8"),
        )
        started = time.perf_counter()
        try:
            with threadpool_limits(limits=r3.MAX_THREADS):
                body = _execute_df_map(root, attestation, lineage, locks, resource_before, map_path)
        except Exception as exc:  # noqa: BLE001 - after marker, persist durable no-retry failure
            body = {
                "schema_version": 1,
                "dfmap_id": DFMAP_ID,
                "status": "FAILED_DF_MAP_NO_RETRY",
                "started_at_utc": attempt_body["started_at_utc"],
                "source_invocation_head": lineage["head"],
                "dfmap_freeze_sha256": lineage["freeze_sha256"],
                "r3_attestation_sha256": attestation["attestation_sha256"],
                "r3_receipt_file_sha256": lineage["r3_receipt_file_sha256"],
                "target_column_requested_or_loaded": False,
                "test_or_sample_loaded": False,
                "estimator_fit_calls": 0,
                "prediction_or_inference_calls": 0,
                "vectorizer_fit_calls": 0,
                "structured_preprocessor_fit_calls": 0,
                "newly_learned_preprocessing_fit_calls": 1,
                "df_map_formula_fit_calls": 1,
                "csv_created": False,
                "kaggle_contacted": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "resource_after_failure": r3._resource_snapshot("after failed DF-map fit"),
            }
        body["elapsed_seconds"] = float(time.perf_counter() - started)
        body["no_retry_if_interrupted"] = True
        return r3.write_and_readback_receipt(receipt_path, body)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--build-freeze", action="store_true")
    actions.add_argument("--preflight", action="store_true")
    actions.add_argument("--fit-df-map", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.build_freeze:
            result = build_freeze()
        elif args.preflight:
            result = preflight()
        else:
            result = run_df_map()
    except (DFMapError, r3.ReplayError, r3.SharedLockHeld, r3.LocalLockHeld) as exc:
        print(json.dumps({"status": "BLOCKED_OR_FAILED", "error_type": type(exc).__name__, "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, indent=2))
    statuses = ("PASS", "FROZEN_FOR_ONE_TRAIN_ONLY_DF_MAP_FIT")
    return 0 if result.get("status", "").startswith(statuses) else 2


if __name__ == "__main__":
    raise SystemExit(main())
