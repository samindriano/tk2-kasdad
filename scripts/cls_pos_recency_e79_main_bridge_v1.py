"""Read-only bridge to frozen MAIN TP2 transforms for the E79 candidate."""
from __future__ import annotations

from contextlib import ExitStack
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
import psutil
from scipy import sparse
from threadpoolctl import threadpool_limits


AUDIT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = (AUDIT_ROOT / "experiments/CLS_POS_RECENCY_E79_V1").resolve()
MAIN_ROOT = AUDIT_ROOT.parent / "TP 2"
TARGET_COLUMNS = {"artifact_signal", "threat_rank"}
MAX_THREADS = 2


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_npz(path: Path, **arrays: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite bridge output: {path}")
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz.tmp", delete=False) as tmp:
        temp_path = Path(tmp.name)
    try:
        with temp_path.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temp_path, path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    return file_sha256(path)


def _atomic_sparse(path: Path, matrix: sparse.spmatrix) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite bridge output: {path}")
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as tmp:
        temp_path = Path(tmp.name)
    try:
        sparse.save_npz(temp_path, sparse.csr_matrix(matrix, dtype=np.float64))
        os.replace(temp_path, path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    return file_sha256(path)


def _load_main_modules():
    if not MAIN_ROOT.is_dir():
        raise RuntimeError(f"Configured MAIN TP2 root is absent: {MAIN_ROOT}")
    sys.path.insert(0, str(MAIN_ROOT))
    from scripts import optimization_v3_a02_cgeom_benchmark_v1 as cgeom
    from scripts import optimization_v3_cls_pos_global_strength_e79_gamma_v1 as gamma
    from scripts import optimization_v3_cls_pos_global_strength_e79_v3 as r3
    from src.models import optimization_v3_positional_text_pilot_v1 as positional
    return cgeom, gamma, r3, positional


def build(mode: str, output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    if not output_dir.is_relative_to(EXPERIMENT_ROOT):
        raise RuntimeError("Bridge outputs must remain under experiments/CLS_POS_RECENCY_E79_V1")
    cgeom, gamma, r3, positional = _load_main_modules()
    source_freeze_path = MAIN_ROOT / gamma.FREEZE_REL
    source_freeze = gamma._read_json(source_freeze_path)
    available_before = psutil.virtual_memory().available

    with ExitStack() as stack:
        shared_lock = stack.enter_context(r3._shared_job_lock())
        cgeom_lock = stack.enter_context(r3._cgeom_local_lock(MAIN_ROOT))
        with threadpool_limits(limits=MAX_THREADS):
            if mode == "train":
                train_path = MAIN_ROOT / r3.TRAIN_REL
                built = gamma._build_train_features(MAIN_ROOT)
                frame = built["train_frame"]
                position = sparse.csr_matrix(built["position"], dtype=np.float64)
                offset = np.asarray(built["offset"], dtype=np.float64).reshape(-1)
                present = ~np.asarray(built["missing_mask"], dtype=bool).reshape(-1)
                source_path = train_path
                matrix_info = built["matrices"]
                id_sha = built["ids_sha256"]
                compatibility = built["compatibility"]
                state_sha = built["unit_state_sha256"]
                target_read = False
                uses_sample = False
            elif mode == "test":
                test_path = MAIN_ROOT / r3.TEST_REL
                frame, columns = r3.read_feature_csv(test_path, cgeom.CLASSIFICATION_FEATURES)
                if TARGET_COLUMNS & set(frame.columns):
                    raise RuntimeError("Feature-only Test reader returned a target column")
                built = gamma._build_feature_matrices_for_rows(MAIN_ROOT, frame)
                position = sparse.csr_matrix(built["position"], dtype=np.float64)
                offset = np.asarray(built["offset"], dtype=np.float64).reshape(-1)
                present = ~np.asarray(built["missing_mask"], dtype=bool).reshape(-1)
                source_path = test_path
                matrix_info = {}
                compatibility = {"status": "REUSED_FROZEN_A02_VECTOR_SPACE"}
                state_sha = {}
                target_read = False
                uses_sample = False
                ids = frame["record_id"].astype(str).to_numpy(dtype=str)
                id_sha = positional.sha256_array(ids)
                if len(columns) != len(frame.columns):
                    raise RuntimeError("Test feature-column inventory differs from the loaded frame")
            else:
                raise ValueError(f"Unknown bridge mode: {mode}")

    if TARGET_COLUMNS & set(frame.columns):
        raise RuntimeError("Bridge input crossed the target boundary")
    ids = frame["record_id"].astype(str).to_numpy(dtype=str)
    cycles = np.asarray(frame["cycle_code"], dtype=np.int64).reshape(-1)
    if (position.shape[0] != len(frame) or len(offset) != len(frame) or len(present) != len(frame)
            or len(ids) != len(set(ids.tolist())) or not np.isfinite(position.data).all()
            or not np.isfinite(offset).all() or not np.isin(cycles, np.arange(47, 92)).all()):
        raise RuntimeError("Bridge features, offsets, IDs, or cycle codes fail alignment checks")
    if mode == "train" and (len(frame) != r3.EXPECTED_ROWS
                            or position.shape != tuple(r3.POSITION_SHAPE)
                            or id_sha != r3.ORDERED_TRAIN_IDS_SHA256):
        raise RuntimeError("Full-Train E79 positional matrix differs from frozen R3 identity")
    if mode == "test" and (len(frame) != 29421 or not np.all((cycles >= 80) & (cycles <= 91))):
        raise RuntimeError("Official Test rows/cycles differ from the authorized 80-91 feature window")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "TRAIN" if mode == "train" else "TEST"
    matrix_path = output_dir / f"{stem}_X_POSITION.npz"
    arrays_path = output_dir / f"{stem}_FEATURE_ARRAYS.npz"
    matrix_sha = _atomic_sparse(matrix_path, position)
    array_sha = _atomic_npz(
        arrays_path,
        record_ids=ids.astype("U"),
        cycles=cycles.astype(np.int64),
        present=present.astype(bool),
        z_bag=offset.astype(np.float64),
    )
    body: dict[str, Any] = {
        "schema_version": 1,
        "candidate_id": "CLS_POS_RECENCY_E79_V1",
        "mode": mode,
        "status": "PASS_FEATURE_ONLY_TRANSFORM",
        "main_root": str(MAIN_ROOT),
        "main_head": subprocess.check_output(
            ["git", "-C", str(MAIN_ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_freeze_path": str(source_freeze_path),
        "source_freeze_file_sha256": file_sha256(source_freeze_path),
        "source_freeze_sha256": source_freeze["freeze_sha256"],
        "source_path": str(source_path),
        "source_file_sha256": file_sha256(source_path),
        "rows": int(len(frame)),
        "present_rows": int(present.sum()),
        "missing_rows": int((~present).sum()),
        "cycles_min_max": [int(cycles.min()), int(cycles.max())],
        "ordered_ids_sha256": id_sha,
        "cycles_sha256": positional.sha256_array(cycles),
        "present_mask_sha256": positional.sha256_array(present),
        "z_bag_sha256": positional.sha256_array(offset),
        "position_matrix": {
            "shape": list(position.shape),
            "nnz": int(position.nnz),
            "data_sha256": positional.sha256_array(position.data),
            "indices_sha256": positional.sha256_array(position.indices),
            "indptr_sha256": positional.sha256_array(position.indptr),
            "logical_sha256": (matrix_info.get("position", {}).get("logical_sha256")
                               if mode == "train" else None),
            "cache_path": str(matrix_path.relative_to(AUDIT_ROOT)),
            "cache_file_sha256": matrix_sha,
        },
        "arrays_cache_path": str(arrays_path.relative_to(AUDIT_ROOT)),
        "arrays_cache_file_sha256": array_sha,
        "feature_state_sha256": state_sha,
        "a02_feature_compatibility": compatibility,
        "feature_columns": list(frame.columns),
        "target_column_requested_or_loaded": False,
        "test_or_sample_used_for_fit": False,
        "sample_submission_access": uses_sample,
        "target_read": target_read,
        "optimizer_started": False,
        "thread_limit": MAX_THREADS,
        "resource": {
            "available_ram_before_bytes": int(available_before),
            "available_ram_after_bytes": int(psutil.virtual_memory().available),
            "shared_job_lock_acquired": True,
            "cgeom_lock_acquired": True,
        },
    }
    body["receipt_sha256"] = positional.canonical_sha256(body)
    receipt_path = output_dir / f"{stem}_TRANSFORM_RECEIPT.json"
    if receipt_path.exists():
        raise RuntimeError(f"Refusing to overwrite transform receipt: {receipt_path}")
    receipt_path.write_text(json.dumps(body, sort_keys=True, ensure_ascii=False,
                                       allow_nan=False, indent=2) + "\n", encoding="utf-8")
    return body


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("train", "test"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build(args.mode, args.output_dir)
    print(json.dumps({
        "status": result["status"],
        "mode": result["mode"],
        "rows": result["rows"],
        "present_rows": result["present_rows"],
        "missing_rows": result["missing_rows"],
        "position_matrix": result["position_matrix"],
        "receipt_sha256": result["receipt_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
