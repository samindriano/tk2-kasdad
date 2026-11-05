"""Zero-optimizer numeric certificate audit for the saved E79 gamma artifact.

This is a separately frozen recovery audit. It never calls an optimizer or
estimator fit, never performs prediction, and never reads Test/SampleSubmission.
The original one-shot fit attempt and its FAIL receipt remain immutable.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import psutil
from scipy import sparse
from scipy.special import expit
from threadpoolctl import threadpool_limits

from scripts import optimization_v3_cls_pos_global_strength_e79_gamma_v1 as gamma_runner
from scripts import optimization_v3_cls_pos_global_strength_e79_v3 as r3
from src.models import cls_pos_global_strength_e79_v1 as gamma_recipe
from src.models import pos_frequency_ridge_v1 as frequency_ridge
from src.models import pos_one_sided_ridge_v1 as one_sided
from src.models import optimization_v3_positional_text_pilot_v1 as positional


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR_REL = "reports/optimization_v3/cls_pos_global_strength_e79_v3"
FREEZE_REL = f"{REPORT_DIR_REL}/GAMMA_ARTIFACT_RECOVERY_FREEZE.json"
PROTOCOL_REL = f"{REPORT_DIR_REL}/GAMMA_ARTIFACT_RECOVERY_PROTOCOL.md"
RESULTS_REL = f"{REPORT_DIR_REL}/GAMMA_ARTIFACT_RECOVERY_RESULTS.md"
RUNNER_REL = "scripts/audit_optimization_v3_cls_pos_global_strength_e79_gamma_recovery_v1.py"
VERIFIER_REL = "scripts/verify_optimization_v3_cls_pos_global_strength_e79_gamma_recovery_v1.py"
TEST_REL = "tests/test_optimization_v3_cls_pos_global_strength_e79_gamma_recovery_v1.py"
OUTPUT_REL = "experiments/runs/optimization_v3_cls_pos_global_strength_e79_v3/gamma_v1/recovery_v1"
ATTEMPT_REL = f"{OUTPUT_REL}/ATTEMPT.started.json"
RECEIPT_REL = f"{OUTPUT_REL}/RECOVERY_RECEIPT.json"

ORIGINAL_OUTPUT = "experiments/runs/optimization_v3_cls_pos_global_strength_e79_v3/gamma_v1"
ORIGINAL_GAMMA_REL = f"{ORIGINAL_OUTPUT}/gamma.npz"
ORIGINAL_ATTEMPT_REL = f"{ORIGINAL_OUTPUT}/ATTEMPT.started.json"
ORIGINAL_RECEIPT_REL = f"{ORIGINAL_OUTPUT}/fit_receipt.json"
ORIGINAL_FREEZE_REL = f"{REPORT_DIR_REL}/GAMMA_SOURCE_FREEZE.json"
ORIGINAL_PASS_REL = f"{REPORT_DIR_REL}/GAMMA_ROOT_SOURCE_PASS.json"
TRAIN_REL = "data/raw/classification/Train.csv"

POLICY_ID = "CLS_POS_GLOBAL_STRENGTH_E79_GAMMA_ARTIFACT_RECOVERY_V1"
ORIGINAL_SOURCE_COMMIT = "50740b680b5b41dffe40f4e81d04774576c81d10"
ORIGINAL_RUN_COMMIT = "e5438bc07861562dc90bba7748f394e73ca57107"
TRAIN_SHA256 = "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c"
EXPECTED_GAMMA_FILE_SHA256 = "4bc240bb6b53eb623f5f099246b70572f987b8c3b0f8dfc123f49029d3c079ab"
EXPECTED_FAILURE_RECEIPT_SHA256 = "05058193fb2cfb32e70d4d2d939fa3abf75b3e1aa77a947eda0a9ccddfb70566"
EXPECTED_ATTEMPT_SHA256 = ""
EXPECTED_C = 1.1859646641031965

GIB = 1024**3
MIN_AVAILABLE_BYTES = 2 * GIB
FREE_RESERVE_BYTES = GIB
INCREMENTAL_ESTIMATE_BYTES = GIB
MAX_THREADS = 2
LOSS_NONINCREASE_TOLERANCE = 1e-8
ABS_GAP_FLOOR = 1e-4
RELATIVE_GAP_SCALE = 1e-8
FROZEN_SOURCE_FILES = (RUNNER_REL, VERIFIER_REL, TEST_REL, PROTOCOL_REL)


class RecoveryAuditError(RuntimeError):
    """Raised when a frozen recovery audit guard fails."""


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_file_sha256(path: Path) -> str:
    payload = path.read_bytes()
    if path.suffix.lower() in {".md", ".json"}:
        payload = payload.replace(b"\r\n", b"\n")
    return hashlib.sha256(payload).hexdigest()


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(root), *args], check=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.stdout.strip()


def _git_blob_sha(root: Path, commit: str, relative: str) -> str:
    data = subprocess.run(["git", "-C", str(root), "show", f"{commit}:{relative}"],
                         check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
    if Path(relative).suffix.lower() in {".md", ".json"}:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RecoveryAuditError(f"Expected JSON object: {path}")
    return value


def require_exact_hash_bindings(expected: Mapping[str, str], actual: Mapping[str, str]) -> None:
    """Fail closed if a frozen artifact/input digest is absent or differs."""
    if dict(expected) != dict(actual):
        differing = sorted(key for key in set(expected) | set(actual)
                           if expected.get(key) != actual.get(key))
        raise RecoveryAuditError(f"Frozen input hash bindings differ: {differing}")


def _write_once(path: Path, body: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = dict(body)
    value["receipt_sha256"] = canonical_sha256(body)
    payload = (json.dumps(value, sort_keys=True, ensure_ascii=False,
                          allow_nan=False, indent=2) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    if _read_json(path) != value:
        raise RecoveryAuditError(f"Recovery receipt readback differs: {path}")
    return file_sha256(path)


def resource_snapshot(stage: str) -> dict[str, Any]:
    vm = psutil.virtual_memory()
    process = psutil.Process().memory_info()
    peak = getattr(process, "peak_wset", getattr(process, "peak_rss", None))
    return {
        "stage": stage,
        "available_ram_bytes": int(vm.available),
        "process_rss_bytes": int(process.rss),
        "process_peak_rss_bytes": int(peak) if peak is not None else None,
        "incremental_memory_estimate_bytes": INCREMENTAL_ESTIMATE_BYTES,
        "required_free_reserve_bytes": FREE_RESERVE_BYTES,
        "projected_reserve_bytes": int(vm.available) - INCREMENTAL_ESTIMATE_BYTES,
        "gate_pass": int(vm.available) >= MIN_AVAILABLE_BYTES,
        "estimate_is_measured_peak": False,
    }


def _source_freeze_body(root: Path) -> dict[str, Any]:
    source_map = {relative: stable_file_sha256(root / relative)
                  for relative in FROZEN_SOURCE_FILES}
    original_source_freeze = _read_json(root / ORIGINAL_FREEZE_REL)
    original_pass = _read_json(root / ORIGINAL_PASS_REL)
    return {
        "schema_version": 1,
        "policy_id": POLICY_ID,
        "status": "FROZEN_ZERO_OPTIMIZER_ARTIFACT_RECOVERY_AUDIT",
        "prepared_from_head": _git(root, "rev-parse", "HEAD"),
        "source_file_sha256": source_map,
        "lineage": {
            "original_gamma_source_commit": ORIGINAL_SOURCE_COMMIT,
            "original_terminal_run_commit": ORIGINAL_RUN_COMMIT,
            "original_source_freeze_sha256": original_source_freeze["freeze_sha256"],
            "original_source_freeze_file_sha256": stable_file_sha256(root / ORIGINAL_FREEZE_REL),
            "original_source_pass_file_sha256": stable_file_sha256(root / ORIGINAL_PASS_REL),
            "original_source_pass_sha256": original_pass["pass_sha256"],
            "official_train_sha256": TRAIN_SHA256,
            "official_train_rows": 50608,
            "original_gamma_artifact_sha256": EXPECTED_GAMMA_FILE_SHA256,
            "original_failure_receipt_sha256": EXPECTED_FAILURE_RECEIPT_SHA256,
            "original_attempt_marker_sha256": file_sha256(root / ORIGINAL_ATTEMPT_REL),
            "original_status": "FAIL_E79_GAMMA_FIT_NO_RETRY",
            "original_failure": "KeyError('checks')",
            "original_fit_count": 1,
            "retry_authorized": False,
        },
        "fixed_numeric_rule": {
            "objective": "sum(logaddexp(0, -(2*y-1)*(z0+Xpos@gamma))) + 0.5*c*dot(gamma,gamma)",
            "penalty_c": EXPECTED_C,
            "strong_convexity_mu": EXPECTED_C,
            "absolute_excess_objective_bound": "||gradient||_2**2 / (2*c)",
            "absolute_acceptance": "gap_bound <= max(1e-4, 1e-8*abs(loss_final))",
            "loss_nonincrease_tolerance": LOSS_NONINCREASE_TOLERANCE,
            "relative_gap_reporting": "diagnostic only; normalized against attained loss; not a second gate",
            "independent_formula_rtol": 1e-12,
            "independent_loss_atol": 1e-8,
            "independent_gradient_atol": 1e-10,
        },
        "resource_policy": {
            "minimum_available_bytes_before_build": MIN_AVAILABLE_BYTES,
            "incremental_estimate_bytes": INCREMENTAL_ESTIMATE_BYTES,
            "free_reserve_bytes_after_build": FREE_RESERVE_BYTES,
            "max_threads": MAX_THREADS,
            "shared_and_cgeom_locks_required": True,
        },
        "prohibited_operations": {
            "optimizer_calls": 0,
            "estimator_fit_calls": 0,
            "preprocessing_fit_calls": 0,
            "predictor_calls": 0,
            "test_or_sample_reads": 0,
            "csv_writes": 0,
            "network_calls": 0,
            "rewrite_original_receipt_or_artifact": True,
        },
        "output_root": OUTPUT_REL,
    }


def write_source_freeze(root: Path = ROOT) -> dict[str, Any]:
    path = root / FREEZE_REL
    if path.exists():
        raise RecoveryAuditError("Recovery source freeze already exists; refusing to replace")
    original_out = root / OUTPUT_REL
    if original_out.exists():
        raise RecoveryAuditError("Recovery output namespace already exists; refusing to reuse")
    body = _source_freeze_body(root)
    freeze = dict(body)
    freeze["freeze_sha256"] = canonical_sha256(body)
    payload = (json.dumps(freeze, sort_keys=True, ensure_ascii=False,
                          allow_nan=False, indent=2) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return freeze


def verify_source_freeze(root: Path = ROOT) -> dict[str, Any]:
    path = root / FREEZE_REL
    freeze = _read_json(path)
    body = dict(freeze)
    claimed = body.pop("freeze_sha256", None)
    if claimed != canonical_sha256(body):
        raise RecoveryAuditError("Recovery source freeze digest mismatch")
    if freeze.get("schema_version") != 1 or freeze.get("policy_id") != POLICY_ID:
        raise RecoveryAuditError("Unexpected recovery source freeze identity")
    head = _git(root, "rev-parse", "HEAD")
    prepared = freeze.get("prepared_from_head")
    if (not prepared or subprocess.run(["git", "-C", str(root), "merge-base", "--is-ancestor",
                                       prepared, head], check=False).returncode != 0):
        raise RecoveryAuditError("Recovery freeze preparation commit is not an ancestor of HEAD")
    if _git_blob_sha(root, head, FREEZE_REL) != stable_file_sha256(path):
        raise RecoveryAuditError("Recovery freeze is not committed exactly at HEAD")
    expected = freeze.get("source_file_sha256", {})
    if set(expected) != set(FROZEN_SOURCE_FILES):
        raise RecoveryAuditError("Recovery freeze source file set differs")
    for relative, digest in expected.items():
        if stable_file_sha256(root / relative) != digest or _git_blob_sha(root, head, relative) != digest:
            raise RecoveryAuditError(f"Frozen recovery source changed: {relative}")
    if file_sha256(root / ORIGINAL_GAMMA_REL) != EXPECTED_GAMMA_FILE_SHA256:
        raise RecoveryAuditError("Original gamma artifact hash differs from the frozen identity")
    if file_sha256(root / ORIGINAL_RECEIPT_REL) != EXPECTED_FAILURE_RECEIPT_SHA256:
        raise RecoveryAuditError("Original failed receipt hash differs from the frozen identity")
    if file_sha256(root / ORIGINAL_ATTEMPT_REL) != freeze["lineage"]["original_attempt_marker_sha256"]:
        raise RecoveryAuditError("Original attempt marker changed")
    for relative, key in ((ORIGINAL_FREEZE_REL, "original_source_freeze_file_sha256"),
                          (ORIGINAL_PASS_REL, "original_source_pass_file_sha256")):
        if stable_file_sha256(root / relative) != freeze["lineage"][key]:
            raise RecoveryAuditError(f"Original gamma source lineage changed: {relative}")
    return {"status": "PASS_RECOVERY_SOURCE_FREEZE", "freeze_sha256": claimed,
            "freeze_file_sha256": stable_file_sha256(path), "head": head,
            "source_files": expected, "original_artifact_and_failure_bindings": True}


def verify_original_terminal_state(root: Path, freeze: Mapping[str, Any]) -> dict[str, Any]:
    gamma_path = root / ORIGINAL_GAMMA_REL
    receipt_path = root / ORIGINAL_RECEIPT_REL
    attempt_path = root / ORIGINAL_ATTEMPT_REL
    attempt = _read_json(attempt_path)
    receipt = _read_json(receipt_path)
    if (attempt.get("status") != "ONE_E79_GAMMA_FIT_ATTEMPT_STARTED"
            or attempt.get("one_fit_only_no_retry") is not True
            or attempt.get("fit_id") != "E79_GLOBAL_STRENGTH"):
        raise RecoveryAuditError("Original one-shot attempt marker identity differs")
    if (receipt.get("status") != "FAIL_E79_GAMMA_FIT_NO_RETRY"
            or receipt.get("error_type") != "KeyError"
            or receipt.get("error") != "'checks'"
            or receipt.get("fit_count") != 1
            or receipt.get("retry_authorized") is not False):
        raise RecoveryAuditError("Original terminal failure receipt identity differs")
    with np.load(gamma_path, allow_pickle=False) as archive:
        if set(archive.files) != {"gamma_float64"}:
            raise RecoveryAuditError("Original gamma archive schema differs")
        gamma = np.asarray(archive["gamma_float64"]).copy()
    if gamma.dtype != np.dtype(np.float64) or gamma.shape != (17259,) or not np.isfinite(gamma).all():
        raise RecoveryAuditError("Original saved gamma vector is not finite float64 with frozen shape")
    gamma_digest = positional.sha256_array(gamma)
    if gamma_digest != "9f5b629524c17d7e045ad5202e192f5545b49a8a85d91cd4da4b8ba032c58de2":
        raise RecoveryAuditError("Original gamma vector content digest differs")
    return {"gamma": gamma, "gamma_file_sha256": file_sha256(gamma_path),
            "gamma_vector_sha256": gamma_digest,
            "attempt_file_sha256": file_sha256(attempt_path),
            "failure_receipt_file_sha256": file_sha256(receipt_path),
            "original_failure_status_preserved": True,
            "optimizer_success_metadata": "UNKNOWN_NOT_PERSISTED"}


def verify_source_control_flow(root: Path) -> dict[str, Any]:
    source = (root / gamma_runner.RUNNER_REL).read_text(encoding="utf-8")
    helper = (root / "src/models/pos_frequency_ridge_v1.py").read_text(encoding="utf-8")
    required = (
        "if metadata.get(\"success\") is not True:",
        "gamma, metadata = _fit_arrays(built[\"position\"], built[\"offset\"], target, built[\"aligned_weights\"])",
        "artifact_sha = _save_npz_once(gamma_path, {\"gamma_float64\": gamma.astype(np.float64, copy=False)})",
        "built[\"matrices\"][\"checks\"]",
    )
    if any(fragment not in source for fragment in required[:2] + required[3:]):
        raise RecoveryAuditError("Frozen original source no longer matches expected control-flow evidence")
    save_pos = source.index(required[2])
    fit_pos = source.index(required[1])
    if fit_pos >= save_pos:
        raise RecoveryAuditError("Gamma save is not textually after the inherited fit-success path")
    try:
        original_optimizer = helper.index("result = minimize(")
        optimizer_gate = helper.index("if (not result.success", original_optimizer)
    except ValueError as exc:
        raise RecoveryAuditError("Frozen optimizer success gate source was not found") from exc
    if original_optimizer >= optimizer_gate:
        raise RecoveryAuditError("Optimizer success gate source ordering is invalid")
    return {
        "status": "PASS_SOURCE_CONTROL_FLOW_INFERENCE",
        "gamma_saved_after_fit_helper_return": True,
        "fit_helper_requires_metadata_success_true": True,
        "receipt_failure_at_missing_checks_lookup_after_gamma_save": True,
        "interpretation": "inference_from_frozen_source_control_flow_only",
        "optimizer_nit_nfev_message_loss_gradient": "UNKNOWN_NOT_PERSISTED",
    }


def _independent_objective_gradient(gamma: np.ndarray, x: sparse.spmatrix,
                                    offset: np.ndarray, y: np.ndarray,
                                    c: float) -> tuple[float, np.ndarray]:
    """Independently coded stable sum-logloss and gradient for W=cI."""
    logits = offset + np.asarray(x @ gamma, dtype=np.float64).reshape(-1)
    signed = (2.0 * y.astype(np.float64) - 1.0) * logits
    loss = float(np.logaddexp(0.0, -signed).sum(dtype=np.float64)
                 + 0.5 * c * np.dot(gamma, gamma))
    residual = expit(logits) - y.astype(np.float64)
    grad = np.asarray(x.T @ residual, dtype=np.float64).reshape(-1) + c * gamma
    return loss, grad


def certify_arrays(*, gamma: Sequence[float], x: sparse.spmatrix,
                   offset: Sequence[float], y: Sequence[int], c: float,
                   expected_gamma_sha256: str | None = None) -> dict[str, Any]:
    coef = np.asarray(gamma, dtype=np.float64).reshape(-1)
    design = sparse.csr_matrix(x, dtype=np.float64)
    z0 = np.asarray(offset, dtype=np.float64).reshape(-1)
    target = np.asarray(y).reshape(-1)
    if (not np.isfinite(coef).all() or not np.isfinite(design.data).all()
            or not np.isfinite(z0).all() or target.dtype.kind not in "biuf"
            or not np.isfinite(target).all() or not np.isin(target, (0, 1)).all()
            or not np.isfinite(c) or c <= 0):
        raise RecoveryAuditError("Certificate inputs must be finite, positive penalty and binary target")
    if design.shape != (len(target), len(coef)) or len(z0) != len(target):
        raise RecoveryAuditError("Certificate arrays do not align")
    gamma_sha = positional.sha256_array(coef)
    if expected_gamma_sha256 is not None and gamma_sha != expected_gamma_sha256:
        raise RecoveryAuditError("Gamma array content hash differs from pinned candidate")
    penalty = np.full(coef.shape, c, dtype=np.float64)
    loss_final, grad_frozen = frequency_ridge.weighted_offset_objective_gradient(
        coef, design, z0, target.astype(np.int8, copy=False), penalty)
    zero = np.zeros_like(coef)
    loss_zero, _ = frequency_ridge.weighted_offset_objective_gradient(
        zero, design, z0, target.astype(np.int8, copy=False), penalty)
    loss_independent, grad_independent = _independent_objective_gradient(coef, design, z0, target, c)
    loss_zero_independent, _ = _independent_objective_gradient(zero, design, z0, target, c)
    grad_l2 = float(np.linalg.norm(grad_frozen, ord=2))
    grad_linf = float(np.linalg.norm(grad_frozen, ord=np.inf))
    gap_bound = float(grad_l2 * grad_l2 / (2.0 * c))
    threshold = float(max(ABS_GAP_FLOOR, RELATIVE_GAP_SCALE * abs(loss_final)))
    loss_delta = float(loss_final - loss_zero)
    loss_crosscheck_delta = float(loss_final - loss_independent)
    loss_zero_crosscheck_delta = float(loss_zero - loss_zero_independent)
    grad_max_abs_delta = float(np.max(np.abs(grad_frozen - grad_independent)))
    loss_matches = bool(np.isclose(loss_final, loss_independent, rtol=1e-12, atol=1e-8)
                        and np.isclose(loss_zero, loss_zero_independent, rtol=1e-12, atol=1e-8))
    gradient_matches = bool(np.allclose(grad_frozen, grad_independent, rtol=1e-12, atol=1e-10))
    finite = bool(np.isfinite([loss_final, loss_zero, grad_l2, grad_linf, gap_bound,
                               threshold, loss_delta, loss_crosscheck_delta,
                               loss_zero_crosscheck_delta, grad_max_abs_delta]).all())
    loss_nonincrease = bool(loss_final <= loss_zero + LOSS_NONINCREASE_TOLERANCE)
    certificate_pass = bool(gap_bound <= threshold)
    checks = {
        "finite_inputs_and_outputs": finite,
        "positive_strong_convexity_scalar": bool(c > 0),
        "penalty_is_exact_cI": bool(np.array_equal(penalty, np.full(coef.shape, c, dtype=np.float64))),
        "frozen_independent_loss_match": loss_matches,
        "frozen_independent_gradient_match": gradient_matches,
        "loss_nonincrease_vs_gamma0": loss_nonincrease,
        "absolute_strong_convexity_certificate": certificate_pass,
    }
    return {
        "status": "PASS_NUMERICAL_CERTIFICATE" if all(checks.values()) else "FAIL_NUMERICAL_CERTIFICATE",
        "checks": checks,
        "c": float(c),
        "loss_final": float(loss_final),
        "loss_gamma0": float(loss_zero),
        "loss_delta_from_gamma0": loss_delta,
        "gradient_l2_norm": grad_l2,
        "gradient_linf_norm": grad_linf,
        "absolute_excess_objective_bound": gap_bound,
        "absolute_bound_threshold": threshold,
        "absolute_acceptance_rule": "gap_bound <= max(1e-4, 1e-8*abs(loss_final))",
        "normalized_bound_over_attained_loss_diagnostic": (
            float(gap_bound / abs(loss_final)) if loss_final != 0 else None),
        "conservative_relative_bound_over_optimum_diagnostic": (
            float(gap_bound / (loss_final - gap_bound))
            if loss_final > gap_bound and loss_final > 0 else None),
        "loss_crosscheck_delta": loss_crosscheck_delta,
        "loss_gamma0_crosscheck_delta": loss_zero_crosscheck_delta,
        "gradient_max_abs_crosscheck_delta": grad_max_abs_delta,
        "gamma_shape": list(coef.shape),
        "gamma_vector_sha256": gamma_sha,
        "rows": int(design.shape[0]),
        "features": int(design.shape[1]),
        "matrix_nnz": int(design.nnz),
    }


@contextmanager
def deny_fit_calls() -> Iterator[dict[str, int]]:
    """Fail-closed counters for all known optimizer and E79 fit entry points."""
    import scipy.optimize
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    counters = {name: 0 for name in (
        "scipy.optimize.minimize", "frequency_ridge.minimize",
        "frequency_ridge.fit_weighted_offset", "one_sided.fit_offset",
        "gamma_recipe.fit_global_strength_gamma", "gamma_runner._fit_arrays",
        "TfidfVectorizer.fit", "TfidfVectorizer.fit_transform",
        "StandardScaler.fit", "StandardScaler.fit_transform",
        "OneHotEncoder.fit", "OneHotEncoder.fit_transform",
        "LogisticRegression.fit", "HistGradientBoostingClassifier.fit",
    )}
    originals = {
        "scipy.optimize.minimize": scipy.optimize.minimize,
        "frequency_ridge.minimize": frequency_ridge.minimize,
        "frequency_ridge.fit_weighted_offset": frequency_ridge.fit_weighted_offset,
        "one_sided.fit_offset": one_sided.fit_offset,
        "gamma_recipe.fit_global_strength_gamma": gamma_recipe.fit_global_strength_gamma,
        "gamma_runner._fit_arrays": gamma_runner._fit_arrays,
        "TfidfVectorizer.fit": TfidfVectorizer.fit,
        "TfidfVectorizer.fit_transform": TfidfVectorizer.fit_transform,
        "StandardScaler.fit": StandardScaler.fit,
        "StandardScaler.fit_transform": StandardScaler.fit_transform,
        "OneHotEncoder.fit": OneHotEncoder.fit,
        "OneHotEncoder.fit_transform": OneHotEncoder.fit_transform,
        "LogisticRegression.fit": LogisticRegression.fit,
        "HistGradientBoostingClassifier.fit": HistGradientBoostingClassifier.fit,
    }

    def blocked(name: str):
        def _call(*_args: Any, **_kwargs: Any) -> Any:
            counters[name] += 1
            raise RecoveryAuditError(f"Prohibited fit call attempted: {name}")
        return _call

    modules = {"scipy.optimize": scipy.optimize, "frequency_ridge": frequency_ridge,
               "one_sided": one_sided, "gamma_recipe": gamma_recipe,
               "gamma_runner": gamma_runner, "TfidfVectorizer": TfidfVectorizer,
               "StandardScaler": StandardScaler, "OneHotEncoder": OneHotEncoder,
               "LogisticRegression": LogisticRegression,
               "HistGradientBoostingClassifier": HistGradientBoostingClassifier}
    for name in counters:
        module_name, attr = name.rsplit(".", 1)
        module = modules[module_name]
        setattr(module, attr, blocked(name))
    try:
        yield counters
    finally:
        for name, original in originals.items():
            module_name, attr = name.rsplit(".", 1)
            module = modules[module_name]
            setattr(module, attr, original)


def _input_hashes(built: Mapping[str, Any], target: np.ndarray,
                  gamma_vector: np.ndarray, root: Path) -> dict[str, str]:
    return {
        "official_train_file_sha256": TRAIN_SHA256,
        "ordered_train_ids_sha256": built["ids_sha256"],
        "base_matrix_sha256": built["matrices"]["base"]["logical_sha256"],
        "position_matrix_sha256": built["matrices"]["position"]["logical_sha256"],
        "c2b_offset_sha256": positional.sha256_array(built["offset"]),
        "aligned_df_weights_sha256": positional.sha256_array(built["aligned_weights"]),
        "target_artifact_signal_sha256": positional.sha256_array(target),
        "c2b_coefficients_sha256": built["c2b_coef_sha256"],
        "c2b_intercept_sha256": built["c2b_intercept_sha256"],
        "unigram_vocabulary_order_sha256": built["token_order_sha256"],
        "dfmap_freeze_sha256": built["dfmap_freeze_sha256"],
        "dfmap_receipt_file_sha256": built["dfmap_receipt_file_sha256"],
        "dfmap_artifact_sha256": built["dfmap_artifact_sha256"],
        "original_gamma_artifact_file_sha256": file_sha256(root / ORIGINAL_GAMMA_REL),
        "original_gamma_vector_sha256": positional.sha256_array(gamma_vector),
        "original_failure_receipt_file_sha256": file_sha256(root / ORIGINAL_RECEIPT_REL),
        "original_attempt_marker_file_sha256": file_sha256(root / ORIGINAL_ATTEMPT_REL),
    }


def run_official_audit(root: Path = ROOT) -> dict[str, Any]:
    source_check = verify_source_freeze(root)
    freeze = _read_json(root / FREEZE_REL)
    original = verify_original_terminal_state(root, freeze)
    source_flow = verify_source_control_flow(root)
    attempt_path = root / ATTEMPT_REL
    receipt_path = root / RECEIPT_REL
    if attempt_path.exists() or receipt_path.exists():
        raise RecoveryAuditError("Recovery audit is one-shot; namespace already contains an attempt or receipt")
    train_path = root / TRAIN_REL
    if file_sha256(train_path) != TRAIN_SHA256:
        raise RecoveryAuditError("Official Train file differs from the frozen source")
    original_freeze = gamma_runner._read_json(root / gamma_runner.FREEZE_REL)
    gamma_runner.verify_source_freeze(root)
    gamma_runner._verify_committed_pass(
        root, gamma_runner.SOURCE_PASS_REL, "source", original_freeze)
    original_gamma = original["gamma"]
    resource_before = resource_snapshot("before recovery feature rebuild")
    if not resource_before["gate_pass"]:
        return {"status": "BLOCKED_RESOURCE_GATE", "resource_before": resource_before,
                "attempt_marker_written": False, "target_values_read": False}

    locks: list[dict[str, Any]] = []
    with ExitStack() as stack:
        locks.append(stack.enter_context(r3._shared_job_lock()))
        locks.append(stack.enter_context(r3._cgeom_local_lock(root)))
        resource_locked = resource_snapshot("under both required locks before recovery feature rebuild")
        if not resource_locked["gate_pass"]:
            return {"status": "BLOCKED_RESOURCE_GATE", "resource_before": resource_locked,
                    "attempt_marker_written": False, "target_values_read": False}
        with threadpool_limits(limits=MAX_THREADS), deny_fit_calls() as fit_calls:
            built = gamma_runner._build_train_features(root)
            resource_after_features = resource_snapshot("after verified feature rebuild before target-only read")
            if resource_after_features["available_ram_bytes"] < FREE_RESERVE_BYTES:
                return {"status": "BLOCKED_POST_FEATURE_RESERVE", "resource_before": resource_locked,
                        "resource_after_features": resource_after_features,
                        "attempt_marker_written": False, "target_values_read": False}
            # Require exact reconstructed penalty scalar and dimensions before marker/target access.
            scalar = float(built["global_strength_scalar"])
            expected_penalty = np.full(built["position"].shape[1], EXPECTED_C, dtype=np.float64)
            if scalar != EXPECTED_C or not np.array_equal(expected_penalty,
                                                         np.full(built["position"].shape[1], scalar)):
                raise RecoveryAuditError("Rebuilt full-Train penalty does not equal the frozen cI recipe")
            if built["matrices"]["base"]["logical_sha256"] != "9d47aa36ff15a0cbeead8cf36049f0a699f7618d0ea8890806a2d82140cff59b":
                raise RecoveryAuditError("Rebuilt base matrix digest differs")
            if built["matrices"]["position"]["logical_sha256"] != "32268550b8220dbfb43f5c92c33cd84554f0bced446d1459c5c55c3eabbf9942":
                raise RecoveryAuditError("Rebuilt positional matrix digest differs")
            require_exact_hash_bindings(
                {"official_train_file_sha256": TRAIN_SHA256,
                 "ordered_train_ids_sha256": "9a96fa28e72b08a6022dd005aed877dadcc5e3dd5053200bcdd12711312c26b2",
                 "base_matrix_sha256": "9d47aa36ff15a0cbeead8cf36049f0a699f7618d0ea8890806a2d82140cff59b",
                 "position_matrix_sha256": "32268550b8220dbfb43f5c92c33cd84554f0bced446d1459c5c55c3eabbf9942"},
                {"official_train_file_sha256": TRAIN_SHA256,
                 "ordered_train_ids_sha256": built["ids_sha256"],
                 "base_matrix_sha256": built["matrices"]["base"]["logical_sha256"],
                 "position_matrix_sha256": built["matrices"]["position"]["logical_sha256"]})
            attempt = {
                "schema_version": 1, "policy_id": POLICY_ID,
                "status": "ZERO_FIT_RECOVERY_AUDIT_STARTED",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_commit": _git(root, "rev-parse", "HEAD"),
                "recovery_freeze_sha256": source_check["freeze_sha256"],
                "recovery_freeze_file_sha256": source_check["freeze_file_sha256"],
                "original_fit_count_consumed": 1,
                "new_optimizer_or_estimator_fit_count": 0,
                "locks": locks,
                "resource_before": resource_locked,
                "resource_after_feature_build": resource_after_features,
                "source_control_flow_inference": source_flow,
                "target_read_authorized_after_frozen_feature_checks": True,
                "test_sample_csv_network_access": False,
            }
            _write_once(attempt_path, attempt)
            started = time.perf_counter()
            try:
                # The official target is read only after freeze, source, artifact,
                # fitted-state, matrix, offset, DF-map, and RAM checks have passed.
                target = gamma_runner._load_train_target_last(root)
                if len(target) != 50608:
                    raise RecoveryAuditError("Target-only read did not return all 50,608 Train labels")
                inputs = _input_hashes(built, target, original_gamma, root)
                certificate = certify_arrays(
                    gamma=original_gamma, x=built["position"], offset=built["offset"],
                    y=target, c=scalar, expected_gamma_sha256=original["gamma_vector_sha256"])
                after = resource_snapshot("after zero-fit numerical certificate")
                zero_fit_calls = dict(fit_calls)
                if any(zero_fit_calls.values()):
                    raise RecoveryAuditError(f"A prohibited fit/optimizer call was attempted: {zero_fit_calls}")
                optimizer_guard_names = {name for name in zero_fit_calls
                                         if name in {"scipy.optimize.minimize", "frequency_ridge.minimize",
                                                     "frequency_ridge.fit_weighted_offset", "one_sided.fit_offset",
                                                     "gamma_recipe.fit_global_strength_gamma", "gamma_runner._fit_arrays"}}
                estimator_guard_names = {name for name in zero_fit_calls
                                         if name in {"LogisticRegression.fit", "HistGradientBoostingClassifier.fit"}}
                preprocessing_guard_names = set(zero_fit_calls) - optimizer_guard_names - estimator_guard_names
                optimizer_calls = {name: zero_fit_calls[name] for name in sorted(optimizer_guard_names)}
                estimator_fit_calls = sum(zero_fit_calls[name] for name in estimator_guard_names)
                preprocessing_fit_calls = sum(zero_fit_calls[name] for name in preprocessing_guard_names)
                status = ("RECOVERED_ARTIFACT_NUMERICALLY_VERIFIED"
                          if certificate["status"] == "PASS_NUMERICAL_CERTIFICATE"
                          else "FAIL_RECOVERY_NUMERICAL_CERTIFICATE_NO_RETRY")
                body = {
                    "schema_version": 1, "policy_id": POLICY_ID,
                    "status": status,
                    "mode": "official_zero_optimizer_artifact_recovery_audit",
                    "source_commit": _git(root, "rev-parse", "HEAD"),
                    "recovery_freeze_sha256": source_check["freeze_sha256"],
                    "recovery_freeze_file_sha256": source_check["freeze_file_sha256"],
                    "original_fit_status": "FAIL_E79_GAMMA_FIT_NO_RETRY",
                    "original_optimizer_success_metadata": "UNKNOWN_NOT_PERSISTED",
                    "source_control_flow_inference": source_flow,
                    "population": {"rows": len(target), "cycles": [47, 79],
                                   "fit_population": "full official Classification Train; target read only for numeric certificate"},
                    "input_sha256": inputs,
                    "matrix_shapes": {"base": list(built["base"].shape),
                                      "position": list(built["position"].shape)},
                    "matrix_nnz": {"base": int(built["base"].nnz),
                                   "position": int(built["position"].nnz)},
                    "matrix_checks": built["matrices"]["checks"],
                    "offset_sha256": positional.sha256_array(built["offset"]),
                    "target_values_read_only_after_feature_checks": True,
                    "target_column_read": "artifact_signal",
                    "penalty_global_strength_scalar": scalar,
                    "certificate": certificate,
                    "optimizer_calls": optimizer_calls,
                    "fit_call_guards": zero_fit_calls,
                    "new_estimator_fit_calls": estimator_fit_calls,
                    "new_preprocessing_fit_calls": preprocessing_fit_calls,
                    "prediction_calls": 0,
                    "test_or_sample_read": False,
                    "csv_created": False,
                    "network_contacted": False,
                    "kaggle_contacted": False,
                    "attempt_marker_sha256": file_sha256(attempt_path),
                    "resource_before": resource_locked,
                    "resource_after_feature_build": resource_after_features,
                    "resource_after_audit": after,
                    "elapsed_seconds": float(time.perf_counter() - started),
                }
                receipt_sha = _write_once(receipt_path, body)
                return {"status": status, "receipt_file_sha256": receipt_sha,
                        "certificate": certificate, "input_sha256": inputs,
                        "optimizer_calls": zero_fit_calls,
                        "original_failure_receipt_preserved": True,
                        "resource_after_audit": after}
            except Exception as exc:
                # A failure receipt is terminal for this recovery namespace.
                after = resource_snapshot("after failed zero-fit recovery audit")
                body = {
                    "schema_version": 1, "policy_id": POLICY_ID,
                    "status": "FAIL_RECOVERY_AUDIT_NO_RETRY",
                    "mode": "official_zero_optimizer_artifact_recovery_audit",
                    "source_commit": _git(root, "rev-parse", "HEAD"),
                    "recovery_freeze_sha256": source_check["freeze_sha256"],
                    "original_fit_status": "FAIL_E79_GAMMA_FIT_NO_RETRY",
                    "original_optimizer_success_metadata": "UNKNOWN_NOT_PERSISTED",
                    "error_type": type(exc).__name__, "error": str(exc),
                    "target_values_read_only_after_feature_checks": True,
                    "optimizer_calls": {name: fit_calls[name] for name in fit_calls
                                         if name in {"scipy.optimize.minimize", "frequency_ridge.minimize",
                                                     "frequency_ridge.fit_weighted_offset", "one_sided.fit_offset",
                                                     "gamma_recipe.fit_global_strength_gamma", "gamma_runner._fit_arrays"}},
                    "fit_call_guards": dict(fit_calls),
                    "new_estimator_fit_calls": sum(fit_calls[name] for name in
                                                    ("LogisticRegression.fit", "HistGradientBoostingClassifier.fit")),
                    "new_preprocessing_fit_calls": sum(value for name, value in fit_calls.items()
                                                        if name.startswith(("TfidfVectorizer.", "StandardScaler.", "OneHotEncoder."))),
                    "test_or_sample_read": False, "csv_created": False,
                    "network_contacted": False, "kaggle_contacted": False,
                    "attempt_marker_sha256": file_sha256(attempt_path),
                    "resource_after_audit": after,
                    "retry_authorized": False,
                }
                _write_once(receipt_path, body)
                return {"status": body["status"], "receipt_file_sha256": file_sha256(receipt_path),
                        "error_type": type(exc).__name__, "error": str(exc),
                        "original_failure_receipt_preserved": True}


def _synthetic_payload(gamma: np.ndarray | None = None) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    x = sparse.csr_matrix(np.asarray([[1., 0.], [1., 0.], [0., 1.], [0., 1.]], dtype=np.float64))
    offset = np.zeros(4, dtype=np.float64)
    y = np.asarray([0, 1, 0, 1], dtype=np.int8)
    candidate = np.zeros(2, dtype=np.float64) if gamma is None else np.asarray(gamma, dtype=np.float64)
    return x, offset, y, candidate


def run_synthetic_e2e(output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise RecoveryAuditError(f"Synthetic recovery namespace is one-shot: {output_dir}")
    with deny_fit_calls() as fit_calls:
        x, offset, y, gamma = _synthetic_payload()
        cert = certify_arrays(gamma=gamma, x=x, offset=offset, y=y, c=1.0,
                              expected_gamma_sha256=positional.sha256_array(gamma))
        if cert["status"] != "PASS_NUMERICAL_CERTIFICATE":
            raise RecoveryAuditError(f"Synthetic exact-optimum certificate failed: {cert}")
        attempt_body = {"status": "SYNTHETIC_ZERO_FIT_AUDIT_STARTED", "fit_calls": 0}
        output_dir.mkdir(parents=True, exist_ok=False)
        _write_once(output_dir / "ATTEMPT.started.json", attempt_body)
        vector_path = output_dir / "gamma.npz"
        vector_path.parent.mkdir(parents=True, exist_ok=True)
        with vector_path.open("xb") as stream:
            np.savez_compressed(stream, gamma_float64=gamma)
        receipt = {
            "status": "SYNTHETIC_RECOVERED_ARTIFACT_NUMERICALLY_VERIFIED",
            "gamma_file_sha256": file_sha256(vector_path),
            "gamma_vector_sha256": positional.sha256_array(gamma),
            "certificate": cert,
            "optimizer_calls": dict(fit_calls),
            "test_sample_csv_network_access": False,
        }
        _write_once(output_dir / "RECOVERY_RECEIPT.json", receipt)
        return {"status": "PASS_SYNTHETIC_ZERO_FIT_RECOVERY_E2E",
                "certificate": cert, "output_dir": str(output_dir),
                "optimizer_calls": dict(fit_calls)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write-source-freeze", action="store_true")
    group.add_argument("--verify-source", action="store_true")
    group.add_argument("--audit-official", action="store_true")
    group.add_argument("--synthetic-e2e", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.write_source_freeze:
            result = write_source_freeze()
        elif args.verify_source:
            result = verify_source_freeze()
        elif args.audit_official:
            result = run_official_audit()
        else:
            if args.output_dir is None:
                raise RecoveryAuditError("--synthetic-e2e requires --output-dir")
            result = run_synthetic_e2e(args.output_dir.resolve())
    except Exception as exc:
        print(json.dumps({"status": "FAIL_E79_GAMMA_RECOVERY_AUDIT",
                          "error_type": type(exc).__name__, "error": str(exc)},
                         sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2))
        return 1
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2))
    return 0 if str(result.get("status", "")).startswith(("PASS", "RECOVERED", "BLOCKED", "FROZEN")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
