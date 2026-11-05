"""Frequency-adaptive ridge for frozen three-third positional coefficients.

This module contains pure numerical helpers only. It does not load datasets,
open targets, fit a model at import time, or write artifacts.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit


OPTIMIZER_OPTIONS = {
    "maxiter": 2000,
    "ftol": 1e-12,
    "gtol": 1e-6,
    "maxls": 50,
}
PERMUTATION_SEED = 42
POSITION_THIRDS = 3


class FrequencyRidgeError(RuntimeError):
    """Raised when a frozen positional design or optimizer violates contract."""


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_array(value: Any) -> str:
    array = np.asarray(value)
    header = canonical_sha256({"shape": array.shape, "dtype": str(array.dtype)})
    digest = hashlib.sha256(header.encode("ascii"))
    if array.dtype.kind in "OUS":
        digest.update(
            json.dumps(array.astype(str).tolist(), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
    else:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def validate_three_third_matrix(matrix: sparse.spmatrix, vocabulary_size: int) -> sparse.csr_matrix:
    x = sparse.csr_matrix(matrix, dtype=np.float64, copy=True)
    x.sum_duplicates()
    x.eliminate_zeros()
    x.sort_indices()
    if vocabulary_size <= 0 or x.shape[1] != POSITION_THIRDS * vocabulary_size:
        raise FrequencyRidgeError("Positional matrix must contain exactly three ordered vocabulary blocks")
    if not np.isfinite(x.data).all():
        raise FrequencyRidgeError("Positional matrix contains non-finite feature values")
    return x


def union_document_frequency(matrix: sparse.spmatrix, vocabulary_size: int) -> np.ndarray:
    """Count fit documents with a token present in any of the three thirds."""
    x = validate_three_third_matrix(matrix, vocabulary_size)
    binary = x.copy()
    binary.data = np.ones(binary.nnz, dtype=np.int8)
    per_token = (
        binary[:, :vocabulary_size]
        + binary[:, vocabulary_size:2 * vocabulary_size]
        + binary[:, 2 * vocabulary_size:3 * vocabulary_size]
    ).tocsr()
    per_token.sum_duplicates()
    df = np.asarray((per_token > 0).sum(axis=0)).reshape(-1).astype(np.int64)
    if df.shape != (vocabulary_size,) or np.any(df <= 0) or np.any(df > x.shape[0]):
        raise FrequencyRidgeError("Every retained positional unigram must have positive prefix document frequency")
    return df


def make_penalty_maps(
    document_frequency: Sequence[int], *, seed: int = PERMUTATION_SEED,
) -> dict[str, Any]:
    raw_df = np.asarray(document_frequency).reshape(-1)
    if (raw_df.dtype.kind not in "iu" or len(raw_df) < 2
            or np.any(raw_df <= 0)):
        raise FrequencyRidgeError("At least two positive token document frequencies are required")
    df = raw_df.astype(np.int64, copy=False)
    median_df = float(np.median(df))
    raw = np.clip(np.sqrt(median_df / df.astype(np.float64)), 0.5, 2.0)
    aligned = raw / float(np.mean(raw))
    permutation = np.random.default_rng(seed).permutation(len(aligned))
    permuted = aligned[permutation]
    if not np.isfinite(aligned).all() or not np.isfinite(permuted).all():
        raise FrequencyRidgeError("Penalty map contains non-finite values")
    if not np.isclose(float(aligned.mean()), 1.0, rtol=0.0, atol=1e-14):
        raise FrequencyRidgeError("Aligned penalty map is not normalized to mean one")
    if not np.array_equal(np.sort(aligned), np.sort(permuted)):
        raise FrequencyRidgeError("Permuted map does not retain the aligned weight multiset")
    return {
        "document_frequency": df,
        "median_document_frequency": median_df,
        "raw_clipped_weights": raw,
        "aligned_token_weights": aligned,
        "permuted_token_weights": permuted,
        "permutation": permutation.astype(np.int64),
        "seed": int(seed),
        "clip_low_count": int(np.count_nonzero(np.sqrt(median_df / df) < 0.5)),
        "clip_high_count": int(np.count_nonzero(np.sqrt(median_df / df) > 2.0)),
        "aligned_equals_permuted": bool(np.array_equal(aligned, permuted)),
        "token_count": int(len(df)),
        "df_sha256": sha256_array(df),
        "aligned_sha256": sha256_array(aligned),
        "permuted_sha256": sha256_array(permuted),
        "permutation_sha256": sha256_array(permutation),
    }


def repeat_token_weights(token_weights: Sequence[float]) -> np.ndarray:
    weights = np.asarray(token_weights, dtype=np.float64).reshape(-1)
    if len(weights) == 0 or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise FrequencyRidgeError("Token penalty weights must be positive, finite, and non-empty")
    repeated = np.tile(weights, POSITION_THIRDS)
    if not (np.array_equal(repeated[:len(weights)], repeated[len(weights):2 * len(weights)])
            and np.array_equal(repeated[:len(weights)], repeated[2 * len(weights):])):
        raise FrequencyRidgeError("Token penalty weights do not repeat in frozen third-block order")
    return repeated


def weighted_offset_objective_gradient(
    gamma: np.ndarray,
    x_position: sparse.spmatrix,
    offset: Sequence[float],
    y: Sequence[int],
    penalty_weights: Sequence[float],
) -> tuple[float, np.ndarray]:
    x = sparse.csr_matrix(x_position, dtype=np.float64)
    coef = np.asarray(gamma, dtype=np.float64).reshape(-1)
    z0 = np.asarray(offset, dtype=np.float64).reshape(-1)
    target = np.asarray(y, dtype=np.float64).reshape(-1)
    weights = np.asarray(penalty_weights, dtype=np.float64).reshape(-1)
    if x.shape != (len(target), len(coef)) or len(z0) != len(target) or len(weights) != len(coef):
        raise FrequencyRidgeError("Position, offset, target, gamma, and penalty dimensions do not align")
    if (not np.isfinite(x.data).all() or not np.isfinite(coef).all()
            or not np.isfinite(z0).all() or not np.isfinite(target).all()
            or not np.isfinite(weights).all() or np.any(weights <= 0)
            or not np.isin(target, (0.0, 1.0)).all()):
        raise FrequencyRidgeError("Objective inputs must be finite; weights positive; target binary")
    z = z0 + np.asarray(x @ coef, dtype=np.float64).reshape(-1)
    signed = (2.0 * target - 1.0) * z
    loss = float(np.logaddexp(0.0, -signed).sum() + 0.5 * np.dot(weights, coef * coef))
    gradient = np.asarray(x.T @ (expit(z) - target), dtype=np.float64).reshape(-1) + weights * coef
    return loss, gradient


def fit_weighted_offset(
    x_position: sparse.spmatrix,
    offset: Sequence[float],
    y: Sequence[int],
    penalty_weights: Sequence[float],
) -> tuple[np.ndarray, dict[str, Any]]:
    x = sparse.csr_matrix(x_position, dtype=np.float64)
    target_raw = np.asarray(y).reshape(-1)
    if (target_raw.dtype.kind not in "biuf" or not np.isfinite(target_raw).all()
            or not np.isin(target_raw, (0, 1)).all()):
        raise FrequencyRidgeError("Fit target must be finite and binary 0/1 before conversion")
    target = target_raw.astype(np.int8, copy=False)
    z0 = np.asarray(offset, dtype=np.float64).reshape(-1)
    weights = np.asarray(penalty_weights, dtype=np.float64).reshape(-1)
    if x.shape != (len(target), len(weights)) or len(z0) != len(target):
        raise FrequencyRidgeError("Fit matrix/offset/target/penalty dimensions do not align")
    if not np.isin(target, (0, 1)).all() or not np.isfinite(z0).all():
        raise FrequencyRidgeError("Fit target must be binary and offset finite")
    if not np.isfinite(x.data).all() or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise FrequencyRidgeError("Fit matrix and positive penalty vector must be finite")
    zero = np.zeros(x.shape[1], dtype=np.float64)
    loss_zero, _ = weighted_offset_objective_gradient(zero, x, z0, target, weights)
    if x.shape[1] == 0 or x.nnz == 0:
        raise FrequencyRidgeError("Frozen positional matrix is empty; no replacement or zero-gamma fit is allowed")
    result = minimize(
        lambda candidate: weighted_offset_objective_gradient(candidate, x, z0, target, weights),
        zero,
        method="L-BFGS-B",
        jac=True,
        options=dict(OPTIMIZER_OPTIONS),
    )
    gamma = np.asarray(result.x, dtype=np.float64).reshape(-1)
    loss_final, gradient = weighted_offset_objective_gradient(gamma, x, z0, target, weights)
    grad_inf = float(np.max(np.abs(gradient))) if len(gradient) else 0.0
    metadata = {
        "optimizer": "scipy.optimize.minimize:L-BFGS-B",
        "success": bool(result.success),
        "message": str(result.message),
        "nit": int(result.nit),
        "nfev": int(result.nfev),
        "gradient_inf_norm": grad_inf,
        "loss_gamma0": float(loss_zero),
        "loss_final": float(loss_final),
        "loss_change": float(loss_final - loss_zero),
        "penalty_weight_sha256": sha256_array(weights),
        "c": 1.0,
        "penalty": "0.5 * sum_k(W_k * gamma_k^2)",
        "options": dict(OPTIMIZER_OPTIONS),
    }
    if (not result.success or not np.isfinite(gamma).all()
            or not np.isfinite(loss_final) or not np.isfinite(gradient).all()
            or loss_final > loss_zero + 1e-8):
        raise FrequencyRidgeError(f"L-BFGS-B did not meet the inherited acceptance contract: {metadata}")
    return gamma, metadata


def predict_with_fallback(
    x_position: sparse.spmatrix,
    offset: Sequence[float],
    gamma: Sequence[float],
    present_mask: Sequence[bool],
    missing_fallback_probability: Sequence[float],
) -> np.ndarray:
    x = sparse.csr_matrix(x_position, dtype=np.float64)
    z0 = np.asarray(offset, dtype=np.float64).reshape(-1)
    coef = np.asarray(gamma, dtype=np.float64).reshape(-1)
    present = np.asarray(present_mask, dtype=bool).reshape(-1)
    fallback = np.asarray(missing_fallback_probability, dtype=np.float64).reshape(-1)
    if x.shape != (len(z0), len(coef)) or len(present) != len(z0) or len(fallback) != len(z0):
        raise FrequencyRidgeError("Prediction inputs do not align")
    if not np.isfinite(x.data).all() or not np.isfinite(z0).all() or not np.isfinite(coef).all():
        raise FrequencyRidgeError("Prediction design, offset, and gamma must be finite")
    if not np.isfinite(fallback).all() or np.any((fallback < 0) | (fallback > 1)):
        raise FrequencyRidgeError("Fallback probabilities must be finite in [0, 1]")
    output = fallback.copy()
    logits = z0 + np.asarray(x @ coef, dtype=np.float64).reshape(-1)
    output[present] = expit(logits[present])
    if not np.array_equal(output[~present], fallback[~present]):
        raise AssertionError("Missing-stream fallback changed")
    return output


def guarded_target_read(
    *,
    expected_hashes: Mapping[str, str],
    actual_hashes: Mapping[str, str],
    expected_ids_sha256: str,
    actual_ids_sha256: str,
    reader,
):
    """Fail before target access when source hashes or row identity drift."""
    if dict(actual_hashes) != dict(expected_hashes):
        raise FrequencyRidgeError("Frozen source hashes differ before target access")
    if actual_ids_sha256 != expected_ids_sha256:
        raise FrequencyRidgeError("Frozen fit IDs differ before target access")
    return reader()


def fit_resume_decision(events: Sequence[Mapping[str, Any]], fit_id: str, allowed_fit_ids: Sequence[str]) -> str:
    if fit_id not in set(allowed_fit_ids):
        raise FrequencyRidgeError(f"Fit ID is outside the exact reserved set: {fit_id}")
    matching = [(index, event) for index, event in enumerate(events) if event.get("fit_id") == fit_id]
    starts = [(index, event) for index, event in matching if event.get("event") == "FIT_STARTED"]
    saved = [(index, event) for index, event in matching if event.get("event") == "FIT_SAVED"]
    failed = [(index, event) for index, event in matching if event.get("event") == "FIT_FAILED"]
    if len(starts) > 1 or len(saved) > 1 or len(failed) > 1 or (saved and failed):
        raise FrequencyRidgeError(f"Duplicate or contradictory ledger events for {fit_id}")
    if saved:
        if len(starts) != 1 or starts[0][0] >= saved[0][0]:
            raise FrequencyRidgeError(f"Saved fit has no earlier matching FIT_STARTED event: {fit_id}")
        return "RESUME_SAVED"
    if failed:
        if len(starts) != 1 or starts[0][0] >= failed[0][0]:
            raise FrequencyRidgeError(f"Failed fit has no earlier matching FIT_STARTED event: {fit_id}")
        raise FrequencyRidgeError(f"Consumed fit failed; retries are prohibited: {fit_id}")
    if starts:
        raise FrequencyRidgeError(f"Fit started without terminal receipt; pending slot cannot be retried: {fit_id}")
    return "START_NEW"


def enforce_fit_cap(events: Sequence[Mapping[str, Any]], allowed_fit_ids: Sequence[str]) -> None:
    starts = [event for event in events if event.get("event") == "FIT_STARTED"]
    start_ids = [str(event.get("fit_id")) for event in starts]
    if len(start_ids) != len(set(start_ids)):
        raise FrequencyRidgeError("Ledger contains duplicate fit starts")
    if not set(start_ids).issubset(set(allowed_fit_ids)) or len(start_ids) > len(allowed_fit_ids):
        raise FrequencyRidgeError(f"Ledger exceeds the exact {len(allowed_fit_ids)}-fit namespace cap")
