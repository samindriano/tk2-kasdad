"""Source-gated one-fit E79 global-strength gamma and application runner.

Official gamma fitting requires a separately committed root source pass.
Official Test inference and CSV creation require a second committed root pass.
Synthetic CLI QA is fully isolated under a caller-provided temporary directory.
"""
from __future__ import annotations

import argparse
import csv
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
import psutil
from scipy import sparse
from threadpoolctl import threadpool_limits

from scripts import optimization_v3_a02_cgeom_benchmark_v1 as cgeom
from scripts import optimization_v3_cls_pos_global_strength_e79_dfmap_resource_v2 as dfv2
from scripts import optimization_v3_cls_pos_global_strength_e79_dfmap_v1 as dfv1
from scripts import optimization_v3_cls_pos_global_strength_e79_v3 as r3
from scripts import verify_optimization_v3_cls_pos_global_strength_e79_dfmap_resource_v2 as dfv2_verify
from src.models import classification_models, cls_pos_global_strength_e79_v1 as recipe
from src.models import optimization_v3_positional_text_pilot_v1 as positional
from src.models import pos_frequency_ridge_v1 as frequency_ridge


ROOT = Path(__file__).resolve().parents[1]
POLICY_ID = "CLS_POS_GLOBAL_STRENGTH_E79_GAMMA_V1"
REPORT_DIR_REL = "reports/optimization_v3/cls_pos_global_strength_e79_v3"
PROTOCOL_REL = f"{REPORT_DIR_REL}/GAMMA_CONTINUATION.md"
RUNNER_REL = "scripts/optimization_v3_cls_pos_global_strength_e79_gamma_v1.py"
VERIFIER_REL = "scripts/verify_optimization_v3_cls_pos_global_strength_e79_gamma_v1.py"
TEST_REL = "tests/test_optimization_v3_cls_pos_global_strength_e79_gamma_v1.py"
HELPER_REL = "src/models/cls_pos_global_strength_e79_v1.py"
FREEZE_REL = f"{REPORT_DIR_REL}/GAMMA_SOURCE_FREEZE.json"
SOURCE_PASS_REL = f"{REPORT_DIR_REL}/GAMMA_ROOT_SOURCE_PASS.json"
APPLICATION_PASS_REL = f"{REPORT_DIR_REL}/GAMMA_APPLICATION_ROOT_PASS.json"
OUTPUT_ROOT_REL = "experiments/runs/optimization_v3_cls_pos_global_strength_e79_v3/gamma_v1"
FIT_ATTEMPT_REL = f"{OUTPUT_ROOT_REL}/ATTEMPT.started.json"
FIT_ARTIFACT_REL = f"{OUTPUT_ROOT_REL}/gamma.npz"
FIT_RECEIPT_REL = f"{OUTPUT_ROOT_REL}/fit_receipt.json"
APPLICATION_ATTEMPT_REL = f"{OUTPUT_ROOT_REL}/APPLICATION_ATTEMPT.started.json"
APPLICATION_NPZ_REL = f"{OUTPUT_ROOT_REL}/application_predictions.npz"
APPLICATION_RECEIPT_REL = f"{OUTPUT_ROOT_REL}/application_receipt.json"
APPLICATION_CSV_REL = "submissions/optimization_v3/cls_pos_global_strength_e79_v3.csv"

DFMAP_RESOURCE_FREEZE_REL = dfv2.FREEZE_REL
DFMAP_RESOURCE_FREEZE_SHA256 = "fab0325b1316603f0668efa019fe7f45aa86522d84c3b2f0208bfb800aedfbfc"
DFMAP_VERIFICATION_REL = (
    "experiments/runs/optimization_v3_cls_pos_global_strength_e79_v3/df_map_v1/"
    "DFMAP_RESOURCE_V2_VERIFICATION.json"
)
DFMAP_ARTIFACT_REL = "experiments/runs/optimization_v3_cls_pos_global_strength_e79_v3/df_map_v1/E79_DF_MAP.npz"
DFMAP_RECEIPT_REL = "experiments/runs/optimization_v3_cls_pos_global_strength_e79_v3/df_map_v1/DFMAP_RECEIPT.json"
TRAIN_REL = r3.TRAIN_REL
TRAIN_SHA256 = r3.TRAIN_SHA256
TRAIN_ROWS = r3.EXPECTED_ROWS
EXPECTED_CYCLES = r3.EXPECTED_CYCLES
POSITION_SHA256 = r3.POSITION_SHA256
BASE_SHA256 = r3.BASE_SHA256
ORDERED_TRAIN_IDS_SHA256 = r3.ORDERED_TRAIN_IDS_SHA256
TARGET_COLUMN = "artifact_signal"

GIB = 1024**3
INCREMENTAL_ESTIMATE_BYTES = GIB
FREE_RESERVE_BYTES = GIB
MIN_AVAILABLE_BYTES = 2 * GIB
MAX_THREADS = 2

SOURCE_FILES = (PROTOCOL_REL, RUNNER_REL, VERIFIER_REL, TEST_REL, HELPER_REL)
DEPENDENCY_FILES = (
    "scripts/optimization_v3_cls_pos_global_strength_e79_v3.py",
    "scripts/optimization_v3_cls_pos_global_strength_e79_dfmap_v1.py",
    "scripts/verify_optimization_v3_cls_pos_global_strength_e79_dfmap_v1.py",
    "scripts/optimization_v3_cls_pos_global_strength_e79_dfmap_resource_v2.py",
    "scripts/verify_optimization_v3_cls_pos_global_strength_e79_dfmap_resource_v2.py",
    "scripts/optimization_v3_a02_cgeom_benchmark_v1.py",
    "src/models/classification_models.py",
    "src/models/optimization_v3_positional_text_pilot_v1.py",
    "src/models/pos_one_sided_ridge_v1.py",
    "src/models/pos_frequency_ridge_v1.py",
)


class GammaError(RuntimeError):
    """Raised when a frozen gamma/application boundary is not satisfied."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_file_sha256(path: Path, relative: str) -> str:
    if Path(relative).suffix.lower() in {".md", ".json"}:
        return sha256_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
    return sha256_file(path)


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                          separators=(",", ":")).encode("utf-8")
    return sha256_bytes(encoded)


def _git(root: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(["git", "-C", str(root), *args], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, check=False)
    if check and result.returncode:
        raise GammaError(result.stderr.decode("utf-8", errors="replace").strip())
    return result.stdout.decode("utf-8", errors="strict").strip()


def _tree_bytes(root: Path, commit: str, relative: str) -> bytes:
    result = subprocess.run(["git", "-C", str(root), "show", f"{commit}:{relative}"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode:
        raise GammaError(f"Committed path unavailable: {commit}:{relative}")
    return result.stdout


def _tree_stable_sha256(root: Path, commit: str, relative: str) -> str:
    payload = _tree_bytes(root, commit, relative)
    if Path(relative).suffix.lower() in {".md", ".json"}:
        payload = payload.replace(b"\r\n", b"\n")
    return sha256_bytes(payload)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GammaError(f"Expected JSON object: {path}")
    return value


def _write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise GammaError(f"One-shot artifact already exists; refusing overwrite: {path}") from exc


def _write_json_once(path: Path, value: dict[str, Any]) -> str:
    body = dict(value)
    body["receipt_sha256"] = canonical_sha256(value)
    encoded = (json.dumps(body, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode()
    _write_once(path, encoded)
    readback = _read_json(path)
    digest = readback.pop("receipt_sha256", None)
    if digest != canonical_sha256(readback) or readback != value:
        raise GammaError(f"Readback self-digest or content mismatch: {path}")
    return sha256_file(path)


def _save_npz_once(path: Path, arrays: Mapping[str, Any]) -> str:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **dict(arrays))
    payload = buffer.getvalue()
    _write_once(path, payload)
    return sha256_bytes(payload)


def resource_snapshot(stage: str) -> dict[str, Any]:
    vm = psutil.virtual_memory()
    process = psutil.Process()
    info = process.memory_info()
    peak = getattr(info, "peak_wset", None)
    if peak is None:
        peak = getattr(info, "peak_rss", None)
    available = int(vm.available)
    projected = available - INCREMENTAL_ESTIMATE_BYTES
    return {
        "stage": stage,
        "available_ram_bytes": available,
        "process_rss_bytes": int(info.rss),
        "process_peak_rss_bytes": int(peak) if peak is not None else None,
        "incremental_memory_estimate_bytes": INCREMENTAL_ESTIMATE_BYTES,
        "required_free_reserve_bytes": FREE_RESERVE_BYTES,
        "minimum_available_bytes": MIN_AVAILABLE_BYTES,
        "projected_free_bytes": projected,
        "max_threads": MAX_THREADS,
        "initial_gate_pass": available >= MIN_AVAILABLE_BYTES,
        "reserve_gate_pass": projected >= FREE_RESERVE_BYTES,
        "gate_pass": available >= MIN_AVAILABLE_BYTES and projected >= FREE_RESERVE_BYTES,
        "resource_capture_utc": datetime.now(timezone.utc).isoformat(),
    }


def _verified_df_map(root: Path) -> tuple[dict[str, Any], np.ndarray, float]:
    verified = dfv2_verify.verify_map(root)
    verification_path = root / DFMAP_VERIFICATION_REL
    saved_verification = _read_json(verification_path)
    if (saved_verification.get("status") != "PASS_INDEPENDENT_DFMAP_RESOURCE_V2_VERIFICATION"
            or saved_verification.get("freeze_sha256") != verified.get("freeze_sha256")
            or saved_verification.get("artifact_sha256") != verified.get("artifact_sha256")
            or saved_verification.get("receipt_file_sha256") != verified.get("receipt_file_sha256")
            or saved_verification.get("receipt_sha256") != verified.get("receipt_sha256")
            or saved_verification.get("global_map_fit_count") != 1
            or saved_verification.get("target_test_sample_or_kaggle_access") is not False):
        raise GammaError("Persisted independent DF-map verification does not match the live frozen map")
    with np.load(root / DFMAP_ARTIFACT_REL, allow_pickle=False) as arrays:
        aligned = np.asarray(arrays["aligned_weights_float64"]).copy()
        scalar_array = np.asarray(arrays["global_strength_scalar_float64"])
        penalty = np.asarray(arrays["global_strength_position_penalty_float64"])
    if (aligned.dtype != np.dtype(np.float64) or aligned.shape != (r3.POSITION_SHAPE[1] // 3,)
            or scalar_array.shape != () or penalty.shape != (3 * len(aligned),)
            or not np.isfinite(aligned).all() or not np.isfinite(penalty).all()):
        raise GammaError("Verified DF-map arrays have an invalid schema")
    exact_penalty, scalar = recipe.global_strength_penalty(aligned)
    if float(scalar_array) != scalar or not np.array_equal(penalty, exact_penalty):
        raise GammaError("Loaded global-strength penalty differs from the frozen map formula")
    return verified, aligned, scalar


def _input_file_hashes(root: Path) -> dict[str, str]:
    v1_freeze = _read_json(root / dfv1.FREEZE_REL)
    immutable = dict(v1_freeze["immutable_input_sha256"])
    immutable[dfv1.FREEZE_REL] = stable_file_sha256(root / dfv1.FREEZE_REL, dfv1.FREEZE_REL)
    immutable.update({
        DFMAP_RESOURCE_FREEZE_REL: stable_file_sha256(root / DFMAP_RESOURCE_FREEZE_REL, DFMAP_RESOURCE_FREEZE_REL),
        DFMAP_VERIFICATION_REL: sha256_file(root / DFMAP_VERIFICATION_REL),
        DFMAP_ARTIFACT_REL: sha256_file(root / DFMAP_ARTIFACT_REL),
        DFMAP_RECEIPT_REL: sha256_file(root / DFMAP_RECEIPT_REL),
        dfv2.R3_RECEIPT_REL: sha256_file(root / dfv2.R3_RECEIPT_REL),
    })
    return dict(sorted(immutable.items()))


def source_freeze_body(root: Path = ROOT) -> dict[str, Any]:
    """Build source lineage from committed code plus the verified Train-only map."""
    dfmap_result, _aligned, scalar = _verified_df_map(root)
    v2_freeze = _read_json(root / DFMAP_RESOURCE_FREEZE_REL)
    head = _git(root, "rev-parse", "HEAD")
    source_hashes = {rel: stable_file_sha256(root / rel, rel)
                     for rel in (*SOURCE_FILES, *DEPENDENCY_FILES)}
    for rel, expected in source_hashes.items():
        if _tree_stable_sha256(root, head, rel) != expected:
            raise GammaError(f"Gamma source must be committed before freeze: {rel}")
    inputs = _input_file_hashes(root)
    for rel, expected in inputs.items():
        path = root / rel
        if not path.is_file() or stable_file_sha256(path, rel) != expected:
            raise GammaError(f"Frozen upstream input hash mismatch: {rel}")
    output_paths = [FIT_ATTEMPT_REL, FIT_ARTIFACT_REL, FIT_RECEIPT_REL,
                    APPLICATION_ATTEMPT_REL, APPLICATION_NPZ_REL, APPLICATION_RECEIPT_REL,
                    APPLICATION_CSV_REL]
    if any((root / rel).exists() for rel in output_paths):
        raise GammaError("Gamma/application output already exists; source freeze must precede any run")
    return {
        "schema_version": 1,
        "policy_id": POLICY_ID,
        "status": "FROZEN_SOURCE_PACKAGE_FOR_ONE_FULLTRAIN_GAMMA_FIT",
        "prepared_from_head": head,
        "source_file_sha256": source_hashes,
        "input_file_sha256": inputs,
        "official_train_path": TRAIN_REL,
        "official_train_sha256": TRAIN_SHA256,
        "official_train_rows": TRAIN_ROWS,
        "fit_id": recipe.FIT_ID,
        "fit_population": "all official Classification Train rows, cycles 47-79, including missing-stream rows",
        "base_matrix": {"shape": list(r3.BASE_SHAPE), "nnz": r3.BASE_NNZ, "logical_sha256": r3.BASE_SHA256},
        "position_matrix": {"shape": list(r3.POSITION_SHAPE), "nnz": r3.POSITION_NNZ, "logical_sha256": r3.POSITION_SHA256},
        "ordered_train_ids_sha256": ORDERED_TRAIN_IDS_SHA256,
        "dfmap_freeze_sha256": DFMAP_RESOURCE_FREEZE_SHA256,
        "dfmap_independent_verification": {
            "status": "PASS_INDEPENDENT_DFMAP_RESOURCE_V2_VERIFICATION",
            "freeze_sha256": dfmap_result["freeze_sha256"],
            "receipt_sha256": dfmap_result["receipt_sha256"],
            "artifact_sha256": dfmap_result["artifact_sha256"],
            "verification_file_sha256": sha256_file(root / DFMAP_VERIFICATION_REL),
            "global_strength_scalar": scalar,
        },
        "recipe": {
            "offset": "z0 = X_base @ C2B.coef_[0] + C2B.intercept_[0]; classes_ must equal [0,1]; intercept is added",
            "objective": "sum_i log(1+exp(-(2*y_i-1)*(z0_i + Xpos_i@gamma))) + 0.5*dot(W,gamma**2)",
            "penalty": "v=max(1,aligned_DF_weight); c=mean(v); W=full(3*V,c)",
            "initial_gamma": "all zeros",
            "optimizer": "scipy.optimize.minimize(method='L-BFGS-B', jac=True)",
            "optimizer_options": dict(frequency_ridge.OPTIMIZER_OPTIONS),
            "blend_present": "0.5 * sigmoid(z0 + Xpos@gamma) + 0.5 * frozen A02 pA",
            "missing_fallback": "frozen A02 pA, preserved bit-exactly",
            "threshold_present": recipe.PRESENT_THRESHOLD,
            "threshold_missing": recipe.MISSING_THRESHOLD,
            "threshold_operator": ">=",
            "test_and_sample_used_during_fit": False,
            "target_columns_during_feature_build": [],
        },
        "resource_policy": {
            "incremental_estimate_bytes": INCREMENTAL_ESTIMATE_BYTES,
            "free_reserve_bytes": FREE_RESERVE_BYTES,
            "minimum_available_bytes": MIN_AVAILABLE_BYTES,
            "max_threads": MAX_THREADS,
            "shared_and_cgeom_locks_required": True,
            "single_fit_no_retry": True,
        },
        "root_pass_paths": {"source": SOURCE_PASS_REL, "application": APPLICATION_PASS_REL},
        "output_root": OUTPUT_ROOT_REL,
        "source_preparation_did_not_fit_gamma_or_read_test": True,
    }


def build_source_freeze(root: Path = ROOT) -> dict[str, Any]:
    tracked = _git(root, "status", "--porcelain", "--untracked-files=no")
    if tracked:
        raise GammaError(f"Tracked worktree must be clean before building source freeze: {tracked}")
    body = source_freeze_body(root)
    payload = dict(body)
    payload["freeze_sha256"] = canonical_sha256(body)
    path = root / FREEZE_REL
    if path.exists():
        raise GammaError("Gamma source freeze already exists; refusing to replace")
    _write_once(path, (json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                  allow_nan=False, indent=2) + "\n").encode())
    return payload


def verify_source_freeze(root: Path = ROOT) -> dict[str, Any]:
    path = root / FREEZE_REL
    freeze = _read_json(path)
    body = dict(freeze)
    digest = body.pop("freeze_sha256", None)
    if digest != canonical_sha256(body):
        raise GammaError("Gamma source freeze self-digest mismatch")
    if freeze.get("policy_id") != POLICY_ID or freeze.get("status") != "FROZEN_SOURCE_PACKAGE_FOR_ONE_FULLTRAIN_GAMMA_FIT":
        raise GammaError("Unexpected gamma source freeze identity/status")
    source_head = freeze.get("prepared_from_head")
    head = _git(root, "rev-parse", "HEAD")
    if not source_head or subprocess.run(["git", "-C", str(root), "merge-base", "--is-ancestor", source_head, head], check=False).returncode:
        raise GammaError("Frozen gamma source commit is not an ancestor of HEAD")
    if set(freeze.get("source_file_sha256", {})) != set((*SOURCE_FILES, *DEPENDENCY_FILES)):
        raise GammaError("Gamma freeze does not bind the exact source/dependency set")
    for rel, expected in freeze["source_file_sha256"].items():
        if stable_file_sha256(root / rel, rel) != expected or _tree_stable_sha256(root, source_head, rel) != expected:
            raise GammaError(f"Gamma frozen source changed: {rel}")
    for rel, expected in freeze.get("input_file_sha256", {}).items():
        if not (root / rel).is_file() or stable_file_sha256(root / rel, rel) != expected:
            raise GammaError(f"Gamma frozen input changed: {rel}")
    if _tree_stable_sha256(root, head, FREEZE_REL) != stable_file_sha256(path, FREEZE_REL):
        raise GammaError("Committed gamma freeze differs from the working file")
    map_result, _weights, scalar = _verified_df_map(root)
    if (freeze.get("dfmap_freeze_sha256") != DFMAP_RESOURCE_FREEZE_SHA256
            or freeze.get("dfmap_independent_verification", {}).get("freeze_sha256") != map_result.get("freeze_sha256")
            or freeze.get("dfmap_independent_verification", {}).get("global_strength_scalar") != scalar
            or freeze.get("official_train_sha256") != TRAIN_SHA256
            or freeze.get("fit_id") != recipe.FIT_ID):
        raise GammaError("Gamma source freeze no longer matches Train/DF-map/recipe lineage")
    return {"status": "PASS_E79_GAMMA_SOURCE_FREEZE", "freeze_sha256": digest,
            "source_commit": source_head, "freeze_file_sha256": stable_file_sha256(path, FREEZE_REL),
            "dfmap_verified": True, "official_gamma_fit_started": False}


def _verify_committed_pass(root: Path, relative: str, kind: str,
                           freeze: dict[str, Any]) -> tuple[dict[str, Any], str]:
    path = root / relative
    if not path.is_file():
        raise GammaError(f"Required committed root pass is absent: {relative}")
    payload = _read_json(path)
    body = dict(payload)
    digest = body.pop("pass_sha256", None)
    if digest != canonical_sha256(body):
        raise GammaError(f"Root pass self-digest mismatch: {relative}")
    expected_status = "PASS_E79_GAMMA_SOURCE_ROOT_REVIEW" if kind == "source" else "PASS_E79_GAMMA_APPLICATION_ROOT_REVIEW"
    if (payload.get("schema_version") != 1
            or payload.get("status") != expected_status
            or payload.get("gamma_source_freeze_sha256") != freeze.get("freeze_sha256")
            or payload.get("gamma_source_freeze_file_sha256") != stable_file_sha256(root / FREEZE_REL, FREEZE_REL)):
        raise GammaError(f"Root pass identity or source-freeze binding mismatch: {relative}")
    head = _git(root, "rev-parse", "HEAD")
    committed_hash = _tree_stable_sha256(root, head, relative)
    working_hash = stable_file_sha256(path, relative)
    if committed_hash != working_hash:
        raise GammaError(f"Root pass is not committed exactly: {relative}")
    reviewed_head = payload.get("reviewed_head")
    if (not reviewed_head or subprocess.run(["git", "-C", str(root), "merge-base", "--is-ancestor", reviewed_head, head], check=False).returncode
            or _tree_stable_sha256(root, reviewed_head, FREEZE_REL) != stable_file_sha256(root / FREEZE_REL, FREEZE_REL)):
        raise GammaError(f"Root pass review commit does not contain the frozen package: {relative}")
    if kind == "source":
        checks = {
            "one_full_train_gamma_fit_authorized": payload.get("one_full_train_gamma_fit_authorized") is True,
            "authorized_fit_ids": payload.get("authorized_fit_ids") == [recipe.FIT_ID],
            "authorized_train_sha256": payload.get("authorized_train_sha256") == TRAIN_SHA256,
            "authorized_train_rows": payload.get("authorized_train_rows") == TRAIN_ROWS,
            "artifact_signal_access_authorized": payload.get("artifact_signal_access_authorized") is True,
            "test_inference_authorized": payload.get("test_inference_authorized") is False,
            "sample_submission_access_authorized": payload.get("sample_submission_access_authorized") is False,
            "kaggle_access_authorized": payload.get("kaggle_access_authorized") is False,
        }
    else:
        checks = {
            "test_inference_authorized": payload.get("test_inference_authorized") is True,
            "one_test_inference_authorized": payload.get("one_test_inference_authorized") is True,
            "test_target_access_authorized": payload.get("test_target_access_authorized") is False,
            "kaggle_access_authorized": payload.get("kaggle_access_authorized") is False,
            "sample_submission_access_authorized": payload.get("sample_submission_access_authorized") is False,
        }
    if not all(checks.values()):
        raise GammaError(f"Root pass lacks the exact {kind} authorization fields: {checks}")
    return payload, working_hash


def _fit_paths(root: Path) -> tuple[Path, Path, Path]:
    return root / FIT_ATTEMPT_REL, root / FIT_ARTIFACT_REL, root / FIT_RECEIPT_REL


def _application_paths(root: Path) -> tuple[Path, Path, Path, Path]:
    return (root / APPLICATION_ATTEMPT_REL, root / APPLICATION_NPZ_REL,
            root / APPLICATION_RECEIPT_REL, root / APPLICATION_CSV_REL)


def _ensure_fit_not_started(root: Path) -> None:
    existing = [p.as_posix() for p in _fit_paths(root) if p.exists()]
    if existing:
        raise GammaError(f"Official gamma fit is one-shot and already has outputs: {existing}")


def _build_train_features(root: Path) -> dict[str, Any]:
    """Rebuild and validate feature matrices before the single target-only read."""
    train, columns = r3.read_feature_csv(root / TRAIN_REL, cgeom.CLASSIFICATION_FEATURES)
    if (len(train) != TRAIN_ROWS or train.record_id.isna().any() or train.record_id.duplicated().any()
            or set(train.columns) & {"artifact_signal", "threat_rank"}):
        raise GammaError("Feature-only Train reader returned unexpected rows, IDs, or target columns")
    cycles = pd.to_numeric(train.cycle_code, errors="raise").to_numpy(dtype=np.int64)
    if tuple(sorted(np.unique(cycles).tolist())) != EXPECTED_CYCLES:
        raise GammaError("Train cycle population differs from frozen cycles 47-79")
    ids = train.record_id.astype(str).to_numpy(dtype=str)
    if positional.sha256_array(ids) != ORDERED_TRAIN_IDS_SHA256:
        raise GammaError("Ordered Train IDs differ from the frozen R3 population")

    attestation = r3._verify_attestation(root)
    saved_units = attestation["saved_inputs_and_fitted_state_receipts"]["saved_units"]
    unit_objects: dict[str, Any] = {}
    unit_state_sha256: dict[str, str] = {}
    for name in ("c2b_structured", "c2b_scaler", "c2b_ohe"):
        record = saved_units[name]
        state = joblib.load(root / record["artifact_path"])
        summary_sha = r3.canonical_sha256(cgeom._state_summary(state))
        if summary_sha != record["state_sha256"]:
            raise GammaError(f"Saved C2_B fitted state differs from the R3 attestation: {name}")
        unit_objects[name] = state
        unit_state_sha256[name] = summary_sha
    a02_model = joblib.load(root / r3.A02_MODEL_REL)
    vectorizer = a02_model.text_space.vectorizer
    compatibility = cgeom.audit_c2b_vectorizer_compatibility(
        frame=train.loc[:, ["record_id", "cycle_code", "signal_stream"]].copy(), model=a02_model)
    expected_compatibility = _read_json(root / r3.CGEOM_PREFLIGHT_REL)["a02_vectorizer_compatibility"]
    for key in ("status", "train_sha256", "train_rows", "vocabulary_sha256", "idf_sha256",
                "fit_documents_sha256", "fit_row_positions_sha256", "target_column_requested_or_loaded", "uses_test_or_sample"):
        if compatibility.get(key) != expected_compatibility.get(key):
            raise GammaError(f"A02 feature compatibility differs from frozen C2_B audit: {key}")
    if compatibility.get("target_column_requested_or_loaded") is not False or compatibility.get("uses_test_or_sample") is not False:
        raise GammaError("A02 compatibility audit crossed the target or Test boundary")

    features = train.loc[:, list(cgeom.CLASSIFICATION_FEATURES)].copy()
    structured = positional.transform_c2b_structured(
        unit_objects["c2b_structured"], unit_objects["c2b_scaler"], unit_objects["c2b_ohe"], features)
    text, missing = positional.transform_c2b_text(vectorizer, features)
    documents, _ = classification_models._text_values(features["signal_stream"])
    position, position_audit = positional.build_position_block(documents, vectorizer, shuffled=False)
    base, position, ids_sha, checks, matrices = r3.assemble_and_validate_matrices(
        structured, text, missing, position, ids,
        {"base_shape": list(r3.BASE_SHAPE), "base_nnz": r3.BASE_NNZ, "base_logical_sha256": r3.BASE_SHA256,
         "position_shape": list(r3.POSITION_SHAPE), "position_nnz": r3.POSITION_NNZ,
         "position_logical_sha256": r3.POSITION_SHA256,
         "ordered_train_ids_sha256": ORDERED_TRAIN_IDS_SHA256})
    if not all(checks.values()) or ids_sha != ORDERED_TRAIN_IDS_SHA256:
        raise GammaError("Rebuilt C2_B or positional Train matrix failed the frozen parity audit")

    dfmap_result, aligned_weights, global_scalar = _verified_df_map(root)
    map_receipt = _read_json(root / DFMAP_RECEIPT_REL)
    token_order = dfv1.ordered_unigrams(vectorizer)
    token_order_sha = dfv1.canonical_sha256([[term, index] for term, index in token_order])
    if (len(token_order) != len(aligned_weights)
            or token_order_sha != map_receipt.get("unigram_vocabulary_order_sha256")
            or map_receipt.get("position_matrix", {}).get("logical_sha256") != POSITION_SHA256
            or dfmap_result.get("global_map_fit_count") != 1):
        raise GammaError("Verified DF-map token order or Train source differs from the frozen positional matrix")

    c2b_estimator = cgeom._load_active_c2b_lr()["estimator"]
    classes = np.asarray(getattr(c2b_estimator, "classes_", []))
    coef = np.asarray(getattr(c2b_estimator, "coef_", []), dtype=np.float64)
    intercept = np.asarray(getattr(c2b_estimator, "intercept_", []), dtype=np.float64).reshape(-1)
    if (not np.array_equal(classes, np.asarray([0, 1])) or coef.shape != (1, base.shape[1])
            or intercept.shape != (1,) or not np.isfinite(coef).all() or not np.isfinite(intercept).all()):
        raise GammaError("Frozen C2_B LR class order/coefficient schema differs from [0,1]")
    offset = np.asarray(base @ coef[0], dtype=np.float64).reshape(-1) + float(intercept[0])
    if not np.isfinite(offset).all() or offset.shape != (TRAIN_ROWS,):
        raise GammaError("C2_B positive-class offset is invalid")
    return {
        "train_frame": train,
        "feature_columns": columns,
        "base": base,
        "position": position,
        "offset": offset,
        "aligned_weights": aligned_weights,
        "global_strength_scalar": global_scalar,
        "missing_mask": np.asarray(missing, dtype=bool),
        "compatibility": compatibility,
        "position_build_audit": position_audit,
        "unit_state_sha256": unit_state_sha256,
        "matrices": matrices,
        "ids_sha256": ids_sha,
        "c2b_coef_sha256": positional.sha256_array(coef),
        "c2b_intercept_sha256": positional.sha256_array(intercept),
        "token_order_sha256": token_order_sha,
        "dfmap_freeze_sha256": dfmap_result["freeze_sha256"],
        "dfmap_receipt_file_sha256": dfmap_result["receipt_file_sha256"],
        "dfmap_artifact_sha256": dfmap_result["artifact_sha256"],
    }


def _load_train_target_last(root: Path) -> np.ndarray:
    """Only call after frozen feature/matrix/model/map checks have all passed."""
    target_frame = pd.read_csv(root / TRAIN_REL, usecols=[TARGET_COLUMN], low_memory=False)
    if list(target_frame.columns) != [TARGET_COLUMN] or len(target_frame) != TRAIN_ROWS:
        raise GammaError("Target-only Train read does not match the frozen population")
    values = pd.to_numeric(target_frame[TARGET_COLUMN], errors="raise").to_numpy()
    if values.dtype.kind not in "biuf" or not np.isfinite(values).all() or not np.isin(values, [0, 1]).all():
        raise GammaError("Official artifact_signal target must be finite binary 0/1")
    return values.astype(np.int8, copy=False)


def _fit_arrays(x_position: sparse.spmatrix, offset: np.ndarray, target: np.ndarray,
                aligned_weights: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    gamma, metadata = recipe.fit_global_strength_gamma(x_position, offset, target, aligned_weights)
    if gamma.shape != (x_position.shape[1],) or not np.isfinite(gamma).all():
        raise GammaError("Frozen optimizer returned an invalid gamma vector")
    if metadata.get("success") is not True:
        raise GammaError("Frozen L-BFGS-B optimizer did not report success")
    return gamma, metadata


def _fit_receipt_body(*, mode: str, fit_id: str, gamma: np.ndarray, metadata: dict[str, Any],
                      hashes: dict[str, str], population: dict[str, Any],
                      resource_before: dict[str, Any], resource_after: dict[str, Any],
                      source_commit: str, source_pass_sha256: str | None,
                      freeze_sha256: str | None) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "policy_id": POLICY_ID,
        "fit_id": fit_id,
        "status": "PASS_E79_GAMMA_FIT_PERSISTED" if mode == "official" else "PASS_E79_SYNTHETIC_GAMMA_FIT_PERSISTED",
        "mode": mode,
        "source_commit": source_commit,
        "source_freeze_sha256": freeze_sha256,
        "root_source_pass_file_sha256": source_pass_sha256,
        "population": population,
        "input_array_sha256": hashes,
        "gamma_sha256": positional.sha256_array(gamma),
        "gamma_shape": list(gamma.shape),
        "gamma_dtype": str(gamma.dtype),
        "objective": "sum_i log(1+exp(-(2*y_i-1)*(z0_i + Xpos_i@gamma))) + 0.5*dot(W,gamma**2)",
        "offset_definition": "positive-class logit: X_base @ C2B.coef_[0] + C2B.intercept_[0], with classes_ == [0,1]",
        "penalty_definition": "v=max(1,aligned_DF_weight); c=mean(v); W=full(3*V,c)",
        "optimizer": {"method": "L-BFGS-B", "options": dict(frequency_ridge.OPTIMIZER_OPTIONS),
                      **{key: (float(value) if isinstance(value, (np.floating,)) else
                               int(value) if isinstance(value, (np.integer,)) else
                               bool(value) if isinstance(value, (np.bool_,)) else value)
                         for key, value in metadata.items()
                         if isinstance(value, (str, bool, int, float, np.floating, np.integer, np.bool_))}},
        "resource_before": resource_before,
        "resource_after": resource_after,
        "fit_count": 1,
        "retry_authorized": False,
        "target_values_read_only_after_feature_parity": True,
        "test_or_sample_read": False,
        "test_inference_performed": False,
        "csv_created": False,
        "kaggle_contacted": False,
    }


def _run_official_fit(root: Path = ROOT) -> dict[str, Any]:
    freeze = _read_json(root / FREEZE_REL)
    freeze_check = verify_source_freeze(root)
    source_pass, source_pass_file_sha = _verify_committed_pass(root, SOURCE_PASS_REL, "source", freeze)
    if (freeze.get("recipe", {}).get("threshold_present") != recipe.PRESENT_THRESHOLD
            or freeze.get("recipe", {}).get("threshold_missing") != recipe.MISSING_THRESHOLD
            or source_pass.get("authorized_train_sha256") != TRAIN_SHA256):
        raise GammaError("Root source pass does not authorize the exact frozen recipe/Train input")
    _ensure_fit_not_started(root)
    invocation_head = _git(root, "rev-parse", "HEAD")

    locks: list[dict[str, Any]] = []
    with ExitStack() as stack:
        locks.append(stack.enter_context(r3._shared_job_lock()))
        locks.append(stack.enter_context(r3._cgeom_local_lock(root)))
        _ensure_fit_not_started(root)
        resource_before = resource_snapshot("immediately before one official E79 gamma fit")
        if not resource_before["gate_pass"]:
            return {"status": "BLOCKED_RESOURCE_GATE", "resource": resource_before,
                    "attempt_marker_written": False, "target_values_read": False}

        built = _build_train_features(root)
        resource_features = resource_snapshot("after verified Train feature matrices before target read")
        if resource_features["available_ram_bytes"] < FREE_RESERVE_BYTES:
            return {"status": "BLOCKED_POST_FEATURE_RESERVE", "resource": resource_features,
                    "attempt_marker_written": False, "target_values_read": False}
        target = _load_train_target_last(root)
        target_sha = positional.sha256_array(target)
        hashes = {
            "official_train_file_sha256": TRAIN_SHA256,
            "ordered_train_ids_sha256": built["ids_sha256"],
            "base_matrix_sha256": built["matrices"]["base"]["logical_sha256"],
            "position_matrix_sha256": built["matrices"]["position"]["logical_sha256"],
            "c2b_offset_sha256": positional.sha256_array(built["offset"]),
            "aligned_df_weights_sha256": positional.sha256_array(built["aligned_weights"]),
            "target_artifact_signal_sha256": target_sha,
            "c2b_coefficients_sha256": built["c2b_coef_sha256"],
            "c2b_intercept_sha256": built["c2b_intercept_sha256"],
            "unigram_vocabulary_order_sha256": built["token_order_sha256"],
            "dfmap_freeze_sha256": built["dfmap_freeze_sha256"],
            "dfmap_receipt_file_sha256": built["dfmap_receipt_file_sha256"],
            "dfmap_artifact_sha256": built["dfmap_artifact_sha256"],
        }
        attempt_path, gamma_path, receipt_path = _fit_paths(root)
        attempt = {
            "schema_version": 1, "status": "ONE_E79_GAMMA_FIT_ATTEMPT_STARTED",
            "policy_id": POLICY_ID, "fit_id": recipe.FIT_ID,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_commit": invocation_head,
            "frozen_source_commit": freeze_check["source_commit"],
            "source_freeze_sha256": freeze_check["freeze_sha256"],
            "source_freeze_file_sha256": freeze_check["freeze_file_sha256"],
            "source_pass_file_sha256": source_pass_file_sha,
            "locks": locks, "resource_before": resource_before,
            "target_values_loaded": True,
            "one_fit_only_no_retry": True,
        }
        _write_once(attempt_path, (json.dumps(attempt, sort_keys=True, ensure_ascii=False,
                                             allow_nan=False, indent=2) + "\n").encode())
        started = time.perf_counter()
        try:
            with threadpool_limits(limits=MAX_THREADS):
                gamma, metadata = _fit_arrays(built["position"], built["offset"], target, built["aligned_weights"])
            resource_after = resource_snapshot("after one official E79 gamma fit")
            artifact_sha = _save_npz_once(gamma_path, {"gamma_float64": gamma.astype(np.float64, copy=False)})
            body = _fit_receipt_body(
                mode="official", fit_id=recipe.FIT_ID, gamma=gamma, metadata=metadata, hashes=hashes,
                population={"rows": TRAIN_ROWS, "cycles": [min(EXPECTED_CYCLES), max(EXPECTED_CYCLES)],
                            "missing_stream_rows_included": True},
                resource_before=resource_before, resource_after=resource_after,
                source_commit=invocation_head, source_pass_sha256=source_pass_file_sha,
                freeze_sha256=freeze_check["freeze_sha256"])
            body.update({"elapsed_seconds": float(time.perf_counter() - started),
                         "frozen_source_commit": freeze_check["source_commit"],
                         "source_freeze_file_sha256": freeze_check["freeze_file_sha256"],
                         "artifact_path": FIT_ARTIFACT_REL, "artifact_sha256": artifact_sha,
                         "locks": locks, "feature_build_checks": built["matrices"]["checks"],
                         "a02_compatibility_checks": built["compatibility"]["checks"],
                         "saved_unit_state_sha256": built["unit_state_sha256"],
                         "target_column_requested_or_loaded": TARGET_COLUMN,
                         "target_read_order": "after full feature-only matrix, fitted-state, C2_B, A02, and DF-map audits"})
            if resource_after["available_ram_bytes"] < FREE_RESERVE_BYTES:
                body["status"] = "FAIL_E79_GAMMA_POSTFIT_RESERVE_NO_RETRY"
                body["postfit_reserve_pass"] = False
            else:
                body["postfit_reserve_pass"] = True
        except Exception as exc:
            body = {
                "schema_version": 1, "policy_id": POLICY_ID, "fit_id": recipe.FIT_ID,
                "status": "FAIL_E79_GAMMA_FIT_NO_RETRY", "mode": "official",
                "source_commit": invocation_head,
                "frozen_source_commit": freeze_check["source_commit"],
                "source_freeze_sha256": freeze_check["freeze_sha256"],
                "source_freeze_file_sha256": freeze_check["freeze_file_sha256"],
                "root_source_pass_file_sha256": source_pass_file_sha,
                "error_type": type(exc).__name__, "error": str(exc),
                "elapsed_seconds": float(time.perf_counter() - started),
                "resource_before": resource_before, "resource_after": resource_snapshot("after failed gamma fit"),
                "fit_count": 1, "retry_authorized": False,
                "target_values_read_only_after_feature_parity": True,
                "test_or_sample_read": False, "test_inference_performed": False,
                "csv_created": False, "kaggle_contacted": False,
            }
        receipt_file_sha = _write_json_once(receipt_path, body)
        return {"status": body["status"], "fit_receipt_sha256": body.get("receipt_sha256"),
                "fit_receipt_file_sha256": receipt_file_sha,
                "gamma_artifact_sha256": body.get("artifact_sha256"),
                "elapsed_seconds": body.get("elapsed_seconds"), "fit_count": 1,
                "target_values_loaded": True, "test_or_sample_read": False}


def _build_feature_matrices_for_rows(root: Path, frame: pd.DataFrame) -> dict[str, Any]:
    if set(frame.columns) & {"artifact_signal", "threat_rank"}:
        raise GammaError("Test feature reader unexpectedly returned a target column")
    if frame.record_id.isna().any() or frame.record_id.duplicated().any():
        raise GammaError("Inference feature rows require unique record IDs")
    a02_model = joblib.load(root / r3.A02_MODEL_REL)
    vectorizer = a02_model.text_space.vectorizer
    attestation = r3._verify_attestation(root)
    units: dict[str, Any] = {}
    for name in ("c2b_structured", "c2b_scaler", "c2b_ohe"):
        record = attestation["saved_inputs_and_fitted_state_receipts"]["saved_units"][name]
        state = joblib.load(root / record["artifact_path"])
        if r3.canonical_sha256(cgeom._state_summary(state)) != record["state_sha256"]:
            raise GammaError(f"Saved C2_B fitted state differs from the frozen R3 audit: {name}")
        units[name] = state
    features = frame.loc[:, list(cgeom.CLASSIFICATION_FEATURES)].copy()
    structured = positional.transform_c2b_structured(units["c2b_structured"], units["c2b_scaler"], units["c2b_ohe"], features)
    text, missing = positional.transform_c2b_text(vectorizer, features)
    documents, _ = classification_models._text_values(features["signal_stream"])
    position, _ = positional.build_position_block(documents, vectorizer, shuffled=False)
    base = sparse.hstack((sparse.csr_matrix(structured, dtype=np.float64),
                          sparse.csr_matrix(text, dtype=np.float64),
                          sparse.csr_matrix(np.asarray(missing, dtype=np.float64).reshape(-1, 1))),
                         format="csr", dtype=np.float64)
    if (base.shape[1] != r3.BASE_SHAPE[1] or position.shape[1] != r3.POSITION_SHAPE[1]
            or base.shape[0] != len(frame) or position.shape[0] != len(frame)
            or not np.isfinite(base.data).all() or not np.isfinite(position.data).all()):
        raise GammaError("Inference C2_B/position matrices have invalid shape or values")
    estimator = cgeom._load_active_c2b_lr()["estimator"]
    if not np.array_equal(np.asarray(estimator.classes_), np.asarray([0, 1])):
        raise GammaError("C2_B class order differs from frozen [0,1]")
    offset = np.asarray(base @ estimator.coef_[0], dtype=np.float64).reshape(-1) + float(estimator.intercept_[0])
    return {"base": base, "position": sparse.csr_matrix(position, dtype=np.float64),
            "offset": offset, "missing_mask": np.asarray(missing, dtype=bool)}


def _read_application_pass(root: Path, freeze: dict[str, Any], fit_receipt_path: Path) -> tuple[dict[str, Any], str]:
    passed, pass_file_sha = _verify_committed_pass(root, APPLICATION_PASS_REL, "application", freeze)
    a02_npz = cgeom.A02_TEST_NPZ
    a02_receipt_path = cgeom.A02_TEST_RECEIPT
    test_path = root / cgeom.TEST.relative_to(root)
    if not test_path.is_file() or not a02_npz.is_file() or not a02_receipt_path.is_file():
        raise GammaError("Required official Test/A02 prediction inputs are missing after application pass")
    bindings = {
        "gamma_fit_receipt_file_sha256": sha256_file(fit_receipt_path),
        "a02_prediction_npz_sha256": sha256_file(a02_npz),
        "a02_test_receipt_sha256": sha256_file(a02_receipt_path),
        "official_test_file_sha256": sha256_file(test_path),
    }
    for key, actual in bindings.items():
        if passed.get(key) != actual:
            raise GammaError(f"Application root pass binding differs: {key}")
    if passed.get("official_test_file_sha256") != cgeom.EXPECTED_TEST_SHA:
        raise GammaError("Application pass Test hash differs from the frozen official Test identity")
    return passed, pass_file_sha


def _run_test_application(root: Path = ROOT) -> dict[str, Any]:
    freeze = _read_json(root / FREEZE_REL)
    freeze_check = verify_source_freeze(root)
    fit_attempt, gamma_path, fit_receipt_path = _fit_paths(root)
    if not fit_attempt.is_file() or not gamma_path.is_file() or not fit_receipt_path.is_file():
        raise GammaError("A verified one-shot gamma fit is required before Test application")
    from scripts import verify_optimization_v3_cls_pos_global_strength_e79_gamma_v1 as verifier
    fit_verified = verifier.verify_fit(root)
    fit_receipt = _read_json(fit_receipt_path)
    if fit_receipt.get("status") != "PASS_E79_GAMMA_FIT_PERSISTED" or fit_receipt.get("mode") != "official":
        raise GammaError("Test application requires a successful official gamma fit receipt")
    application_pass, application_pass_sha = _read_application_pass(root, freeze, fit_receipt_path)
    attempt_path, app_npz_path, app_receipt_path, csv_path = _application_paths(root)
    if any(path.exists() for path in (attempt_path, app_npz_path, app_receipt_path, csv_path)):
        raise GammaError("Test application is one-shot; prior output exists")

    locks: list[dict[str, Any]] = []
    with ExitStack() as stack:
        locks.append(stack.enter_context(r3._shared_job_lock()))
        locks.append(stack.enter_context(r3._cgeom_local_lock(root)))
        if any(path.exists() for path in (attempt_path, app_npz_path, app_receipt_path, csv_path)):
            raise GammaError("Test application is one-shot; prior output appeared before the locked start")
        resource_before = resource_snapshot("immediately before one E79 official Test application")
        if resource_before["available_ram_bytes"] < FREE_RESERVE_BYTES:
            return {"status": "BLOCKED_APPLICATION_RESOURCE_GATE", "resource": resource_before,
                    "attempt_marker_written": False}
        test_path = root / cgeom.TEST.relative_to(root)
        test, columns = r3.read_feature_csv(test_path, cgeom.CLASSIFICATION_FEATURES)
        if len(test) != 29_421 or set(test.columns) & {"artifact_signal", "threat_rank"}:
            raise GammaError("Official Test feature-only population or schema differs")
        cycles = pd.to_numeric(test.cycle_code, errors="raise").to_numpy(dtype=np.int64)
        if not np.array_equal(np.unique(cycles), np.arange(80, 92, dtype=np.int64)):
            raise GammaError("Official Test must contain cycles 80-91")
        a02_receipt = _read_json(cgeom.A02_TEST_RECEIPT)
        a02_fit_receipt = _read_json(root / r3.A02_RECEIPT_REL)
        if (a02_receipt.get("official_test_targets_loaded") is not False
                or a02_receipt.get("npz_sha256") != sha256_file(cgeom.A02_TEST_NPZ)
                or a02_receipt.get("fit_receipt_sha256") != a02_fit_receipt.get("receipt_sha256")
                or a02_receipt.get("freeze_signature") != a02_fit_receipt.get("freeze_signature")
                or a02_fit_receipt.get("model_sha256") != r3.A02_MODEL_SHA256
                or a02_fit_receipt.get("test_data_used") is not False
                or a02_fit_receipt.get("query_labels_passed_to_fit") is not False):
            raise GammaError("Frozen A02 pA receipt has invalid Test-target or artifact lineage")
        with np.load(cgeom.A02_TEST_NPZ, allow_pickle=False) as archive:
            a02_arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
        required = {"record_ids", "candidate_probability"}
        if not required.issubset(a02_arrays):
            raise GammaError("Frozen A02 Test cache lacks ordered IDs or pA values")
        ids = test.record_id.astype(str).to_numpy(dtype=str)
        p_a02 = np.asarray(a02_arrays["candidate_probability"], dtype=np.float64).reshape(-1)
        if (len(p_a02) != len(test) or not np.array_equal(a02_arrays["record_ids"].astype(str), ids)
                or not np.isfinite(p_a02).all() or np.any((p_a02 < 0) | (p_a02 > 1))):
            raise GammaError("A02 pA cache IDs/probabilities do not align with official Test")
        built = _build_feature_matrices_for_rows(root, test)
        with np.load(gamma_path, allow_pickle=False) as archive:
            gamma = np.asarray(archive["gamma_float64"], dtype=np.float64).copy()
        if gamma.shape != (built["position"].shape[1],) or not np.isfinite(gamma).all():
            raise GammaError("Persisted gamma does not align with Test positional features")
        application_invocation_head = _git(root, "rev-parse", "HEAD")
        marker = {
            "schema_version": 1, "status": "ONE_E79_APPLICATION_ATTEMPT_STARTED",
            "policy_id": POLICY_ID, "source_commit": application_invocation_head,
            "frozen_source_commit": freeze_check["source_commit"],
            "source_freeze_sha256": freeze_check["freeze_sha256"],
            "gamma_fit_receipt_sha256": fit_verified["receipt_sha256"],
            "application_pass_file_sha256": application_pass_sha,
            "test_feature_file_sha256": sha256_file(test_path),
            "a02_prediction_npz_sha256": sha256_file(cgeom.A02_TEST_NPZ),
            "locks": locks, "resource_before": resource_before,
            "test_target_requested_or_loaded": False,
            "sample_submission_loaded": False, "kaggle_contacted": False,
            "one_application_only_no_retry": True,
        }
        _write_once(attempt_path, (json.dumps(marker, sort_keys=True, ensure_ascii=False,
                                             allow_nan=False, indent=2) + "\n").encode())
        present = ~built["missing_mask"]
        probabilities = recipe.route_and_combine(built["position"], built["offset"], gamma,
                                                 present, p_a02)
        labels = recipe.apply_frozen_thresholds(probabilities, present)
        if not np.array_equal(probabilities[~present], p_a02[~present]):
            raise GammaError("Missing-stream A02 fallback changed during Test application")
        if not np.isfinite(probabilities).all() or not np.isin(labels, [0, 1]).all():
            raise GammaError("Test application generated invalid probabilities or labels")

        file_sha = _save_npz_once(app_npz_path, {
            "record_ids": ids.astype(str), "present_stream": present.astype(bool),
            "a02_probability": p_a02.astype(np.float64),
            "probability": probabilities.astype(np.float64), "labels": labels.astype(np.int8),
        })
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(["record_id", TARGET_COLUMN])
            writer.writerows(zip(ids.tolist(), labels.astype(int).tolist()))
        resource_after = resource_snapshot("after one E79 official Test application")
        body = {
            "schema_version": 1, "policy_id": POLICY_ID,
            "status": "PASS_E79_OFFICIAL_TEST_APPLICATION_CSV_CREATED" if resource_after["available_ram_bytes"] >= FREE_RESERVE_BYTES else "FAIL_E79_APPLICATION_POSTRUN_RESERVE_NO_RETRY",
            "mode": "official", "source_commit": application_invocation_head,
            "frozen_source_commit": freeze_check["source_commit"],
            "source_freeze_sha256": freeze_check["freeze_sha256"],
            "gamma_fit_receipt_file_sha256": sha256_file(fit_receipt_path),
            "application_root_pass_file_sha256": application_pass_sha,
            "official_test_file_sha256": sha256_file(test_path),
            "test_feature_columns_requested": columns,
            "test_rows": len(test), "test_cycle_values": [80, 91],
            "ordered_test_ids_sha256": positional.sha256_array(ids),
            "a02_prediction_npz_sha256": sha256_file(cgeom.A02_TEST_NPZ),
            "a02_test_receipt_sha256": sha256_file(cgeom.A02_TEST_RECEIPT),
            "a02_probability_sha256": positional.sha256_array(p_a02),
            "probability_sha256": positional.sha256_array(probabilities),
            "labels_sha256": positional.sha256_array(labels),
            "missing_stream_a02_fallback_bit_exact": bool(np.array_equal(probabilities[~present], p_a02[~present])),
            "present_threshold": recipe.PRESENT_THRESHOLD,
            "missing_threshold": recipe.MISSING_THRESHOLD,
            "threshold_operator": ">=",
            "application_bundle_path": APPLICATION_NPZ_REL,
            "application_bundle_sha256": file_sha,
            "csv_path": APPLICATION_CSV_REL,
            "csv_sha256": sha256_file(csv_path),
            "csv_columns": ["record_id", TARGET_COLUMN],
            "sample_submission_loaded": False, "test_target_requested_or_loaded": False,
            "kaggle_contacted": False, "resource_before": resource_before,
            "resource_after": resource_after, "locks": locks,
            "fit_count_during_application": 0, "inference_count": 1,
        }
        receipt_file_sha = _write_json_once(app_receipt_path, body)
        return {"status": body["status"], "application_receipt_file_sha256": receipt_file_sha,
                "csv_path": APPLICATION_CSV_REL, "csv_sha256": body["csv_sha256"],
                "rows": body["test_rows"], "test_targets_loaded": False, "kaggle_contacted": False}


def _synthetic_fit_arrays() -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(71079)
    raw = rng.poisson(0.65, size=(36, 6)).astype(np.float64)
    raw[raw > 0] = 1.0
    x = sparse.csr_matrix(raw)
    x.eliminate_zeros()
    offset = rng.normal(0.0, 0.3, size=36).astype(np.float64)
    linear = offset + np.asarray(x @ np.array([0.45, -0.25, 0.2, -0.15, 0.3, -0.35])).reshape(-1)
    target = (linear + rng.normal(0, 0.8, size=36) > 0.0).astype(np.int8)
    if len(np.unique(target)) < 2:
        target[0], target[-1] = 0, 1
    weights = np.asarray([0.8, 1.2], dtype=np.float64)
    return x, offset, target, weights


def _synthetic_e2e(output_dir: Path) -> dict[str, Any]:
    """Exercise actual optimizer, persistence, readback, resume, app, CSV, and verifier."""
    fit_dir = output_dir / "fit"
    app_dir = output_dir / "application"
    attempt = fit_dir / "ATTEMPT.started.json"
    artifact = fit_dir / "gamma.npz"
    receipt = fit_dir / "fit_receipt.json"
    if any(path.exists() for path in (attempt, artifact, receipt, app_dir / "application.csv")):
        raise GammaError("Synthetic end-to-end is one-shot per temporary output directory")
    x, offset, target, weights = _synthetic_fit_arrays()
    attempt_body = {"status": "ONE_SYNTHETIC_GAMMA_FIT_ATTEMPT_STARTED", "fit_count": 1,
                    "retry_authorized": False, "seed": 71079}
    _write_once(attempt, (json.dumps(attempt_body, sort_keys=True, indent=2) + "\n").encode())
    gamma, metadata = _fit_arrays(x, offset, target, weights)
    artifact_sha = _save_npz_once(artifact, {"gamma_float64": gamma})
    hashes = {"x_position_sha256": positional.sha256_csr(x),
              "offset_sha256": positional.sha256_array(offset),
              "target_sha256": positional.sha256_array(target),
              "aligned_weights_sha256": positional.sha256_array(weights)}
    body = _fit_receipt_body(mode="synthetic", fit_id="SYNTHETIC_E79_GLOBAL_STRENGTH",
                             gamma=gamma, metadata=metadata, hashes=hashes,
                             population={"rows": len(target), "features": x.shape[1]},
                             resource_before={"synthetic": True}, resource_after={"synthetic": True},
                             source_commit="synthetic", source_pass_sha256=None, freeze_sha256=None)
    body.update({"artifact_path": str(artifact), "artifact_sha256": artifact_sha,
                 "synthetic_seed": 71079})
    _write_json_once(receipt, body)
    # Independent CLI reads only the newly persisted artifact/receipt; it cannot refit.
    from scripts import verify_optimization_v3_cls_pos_global_strength_e79_gamma_v1 as verifier
    fit_verification = verifier.verify_synthetic_fit(output_dir)
    with np.load(artifact, allow_pickle=False) as stored:
        gamma_readback = np.asarray(stored["gamma_float64"]).copy()
    if not np.array_equal(gamma, gamma_readback):
        raise GammaError("Synthetic persisted gamma differs bit-for-bit on readback")

    rng = np.random.default_rng(71080)
    app_x = x[:12].tocsr()
    app_offset = offset[:12].copy()
    present = np.asarray([True, True, False, True, False, True, True, False, True, True, False, True])
    p_a02 = np.linspace(0.2, 0.8, num=len(present), dtype=np.float64)
    probability = recipe.route_and_combine(app_x, app_offset, gamma, present, p_a02)
    labels = recipe.apply_frozen_thresholds(probability, present)
    ids = np.asarray([f"SYN-{idx:03d}" for idx in range(len(present))], dtype=str)
    app_dir.mkdir(parents=True, exist_ok=True)
    app_npz = app_dir / "application_predictions.npz"
    csv_path = app_dir / "application.csv"
    app_artifact_sha = _save_npz_once(app_npz, {"record_ids": ids, "present_stream": present,
                                                "a02_probability": p_a02,
                                                "probability": probability,
                                                "labels": labels})
    with csv_path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["record_id", TARGET_COLUMN])
        writer.writerows(zip(ids.tolist(), labels.astype(int).tolist()))
    app_body = {"status": "PASS_E79_SYNTHETIC_APPLICATION_CSV_CREATED", "mode": "synthetic",
                "application_bundle_sha256": app_artifact_sha, "csv_sha256": sha256_file(csv_path),
                "rows": len(ids), "missing_fallback_bit_exact": bool(np.array_equal(probability[~present], p_a02[~present])),
                "present_threshold": recipe.PRESENT_THRESHOLD, "missing_threshold": recipe.MISSING_THRESHOLD,
                "threshold_operator": ">=", "test_target_requested_or_loaded": False,
                "sample_submission_loaded": False, "kaggle_contacted": False}
    _write_json_once(app_dir / "application_receipt.json", app_body)
    app_verification = verifier.verify_synthetic_application(output_dir)
    return {"status": "PASS_E79_SYNTHETIC_END_TO_END", "fit_verification": fit_verification,
            "application_verification": app_verification,
            "actual_optimizer_fit_count": 1, "actual_test_or_sample_access": False,
            "actual_official_gamma_fit_or_inference": False,
            "resume_action": "independent readback verified persisted fit; no second fit was run"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--build-freeze", action="store_true")
    group.add_argument("--verify-freeze", action="store_true")
    group.add_argument("--fit-gamma", action="store_true")
    group.add_argument("--apply-test", action="store_true")
    group.add_argument("--synthetic-e2e", action="store_true")
    group.add_argument("--resume-synthetic-e2e", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.build_freeze:
            result = build_source_freeze()
        elif args.verify_freeze:
            result = verify_source_freeze()
        elif args.fit_gamma:
            result = _run_official_fit()
        elif args.apply_test:
            result = _run_test_application()
        elif args.synthetic_e2e:
            if args.output_dir is None:
                raise GammaError("--synthetic-e2e requires a dedicated temporary --output-dir")
            result = _synthetic_e2e(args.output_dir.resolve())
        else:
            if args.output_dir is None:
                raise GammaError("--resume-synthetic-e2e requires the existing temporary --output-dir")
            from scripts import verify_optimization_v3_cls_pos_global_strength_e79_gamma_v1 as verifier
            result = verifier.verify_synthetic_e2e(args.output_dir.resolve())
            result["status"] = "PASS_E79_SYNTHETIC_RESUME_VALIDATED_NO_REFIT"
            result["fit_count"] = 1
    except Exception as exc:
        print(json.dumps({"status": "FAIL_OR_BLOCKED_E79_GAMMA", "error_type": type(exc).__name__,
                          "error": str(exc)}, sort_keys=True, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2))
    return 0 if str(result.get("status", "")).startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
