"""Small, deterministic sparse factorization-machine objective for the V1 pilot."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse
from scipy.special import expit


def initialize_parameters(n_features: int, rank: int, *, seed: int = 42,
                          factor_std: float = 0.01, with_factors: bool = True) -> np.ndarray:
    if n_features <= 0 or rank <= 0 or factor_std <= 0:
        raise ValueError("n_features, rank and factor_std must be positive")
    w = np.zeros(n_features, dtype=np.float64)
    b = np.zeros(1, dtype=np.float64)
    if not with_factors:
        return np.concatenate((w, b))
    v = np.random.default_rng(seed).normal(0.0, factor_std, (n_features, rank))
    return np.concatenate((w, b, v.reshape(-1)))


def unpack_parameters(theta: np.ndarray, n_features: int, rank: int,
                      *, with_factors: bool = True) -> tuple[np.ndarray, float, np.ndarray | None]:
    p = np.asarray(theta, dtype=np.float64)
    expected = n_features + 1 + (n_features * rank if with_factors else 0)
    if p.shape != (expected,) or not np.isfinite(p).all():
        raise ValueError("parameter vector has invalid shape or non-finite values")
    w = p[:n_features]
    b = float(p[n_features])
    v = p[n_features + 1:].reshape(n_features, rank) if with_factors else None
    return w, b, v


def _as_csr(x: sparse.spmatrix) -> sparse.csr_matrix:
    if not sparse.issparse(x) or x.ndim != 2:
        raise TypeError("X must be a two-dimensional sparse matrix")
    out = x.tocsr().astype(np.float64, copy=False)
    if out.data.size and not np.isfinite(out.data).all():
        raise ValueError("X contains non-finite values")
    return out


def row_l2_normalize(x: sparse.csr_matrix) -> sparse.csr_matrix:
    norms = np.sqrt(np.asarray(x.multiply(x).sum(axis=1), dtype=np.float64).reshape(-1))
    inv = np.zeros_like(norms)
    nz = norms > 0
    inv[nz] = 1.0 / norms[nz]
    return sparse.diags(inv, format="csr") @ x


def fm_pair_score(q: sparse.csr_matrix, v: np.ndarray) -> np.ndarray:
    q = _as_csr(q)
    factors = np.asarray(v, dtype=np.float64)
    if factors.ndim != 2 or factors.shape[0] != q.shape[1] or not np.isfinite(factors).all():
        raise ValueError("V has invalid shape or non-finite values")
    xv = np.asarray(q @ factors, dtype=np.float64)
    q2v2 = np.asarray(q.multiply(q) @ (factors * factors), dtype=np.float64)
    return 0.5 * np.sum(xv * xv - q2v2, axis=1)


def fm_pair_score_and_v_gradient(q: sparse.csr_matrix, v: np.ndarray,
                                 row_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q = _as_csr(q)
    factors = np.asarray(v, dtype=np.float64)
    weights = np.asarray(row_weights, dtype=np.float64)
    if factors.ndim != 2 or factors.shape[0] != q.shape[1] or weights.shape != (q.shape[0],):
        raise ValueError("factor or row-weight shape mismatch")
    xv = np.asarray(q @ factors, dtype=np.float64)
    q2 = q.multiply(q)
    q2v2 = np.asarray(q2 @ (factors * factors), dtype=np.float64)
    score = 0.5 * np.sum(xv * xv - q2v2, axis=1)
    grad = np.asarray(q.T @ (weights[:, None] * xv), dtype=np.float64)
    correction = np.asarray(q2.T @ weights, dtype=np.float64).reshape(-1)
    grad -= correction[:, None] * factors
    return score, grad


def objective_gradient_sum(theta: np.ndarray, x: sparse.spmatrix, y: np.ndarray,
                           *, rank: int = 16, c: float = 1.0,
                           with_factors: bool = True, chunk_rows: int = 1024,
                           resource_check: Any = None) -> tuple[float, np.ndarray]:
    """SUM BCE + one L2 penalty, accumulating data loss/gradient in row chunks."""
    matrix = _as_csr(x)
    target = np.asarray(y, dtype=np.int8)
    if target.shape != (matrix.shape[0],) or not np.isin(target, (0, 1)).all():
        raise ValueError("y must be a binary vector aligned with X")
    if c <= 0 or chunk_rows <= 0 or matrix.shape[0] == 0:
        raise ValueError("invalid objective controls")
    w, b, v = unpack_parameters(theta, matrix.shape[1], rank, with_factors=with_factors)
    signed = (2 * target.astype(np.int16) - 1).astype(np.float64)
    data_loss = 0.0
    grad_w = np.zeros(matrix.shape[1], dtype=np.float64)
    grad_b = 0.0
    grad_v = np.zeros_like(v) if v is not None else None
    if resource_check is not None:
        resource_check()
    for start in range(0, matrix.shape[0], chunk_rows):
        end = min(start + chunk_rows, matrix.shape[0])
        x_block = matrix[start:end]
        q_block = row_l2_normalize(x_block) if v is not None else None
        xv = np.asarray(q_block @ v, dtype=np.float64) if v is not None else None
        q2_block = q_block.multiply(q_block) if q_block is not None else None
        pair = (0.5 * np.sum(xv * xv - np.asarray(q2_block @ (v * v), dtype=np.float64), axis=1)
                if v is not None and xv is not None and q2_block is not None else 0.0)
        logits = np.asarray(x_block @ w, dtype=np.float64).reshape(-1) + b + pair
        t = signed[start:end]
        data_loss += float(np.logaddexp(0.0, -t * logits).sum(dtype=np.float64))
        residual = -t * expit(-t * logits)
        grad_w += np.asarray(x_block.T @ residual, dtype=np.float64).reshape(-1)
        grad_b += float(residual.sum(dtype=np.float64))
        if v is not None:
            assert q_block is not None and q2_block is not None and xv is not None and grad_v is not None
            block_grad = np.asarray(q_block.T @ (residual[:, None] * xv), dtype=np.float64)
            correction = np.asarray(q2_block.T @ residual, dtype=np.float64).reshape(-1)
            block_grad -= correction[:, None] * v
            grad_v += block_grad
    penalty = 0.5 / c * (float(w @ w) + b * b + (float(np.sum(v * v)) if v is not None else 0.0))
    grad_w += w / c
    grad_b += b / c
    factor_grad = (grad_v + v / c).reshape(-1) if grad_v is not None and v is not None else np.empty(0)
    grad = np.concatenate((grad_w, np.asarray([grad_b]), factor_grad))
    return data_loss + penalty, grad


def objective_gradient_mean(theta: np.ndarray, x: sparse.spmatrix, y: np.ndarray,
                            *, rank: int = 16, c: float = 1.0,
                            with_factors: bool = True, chunk_rows: int = 1024) -> tuple[float, np.ndarray]:
    value, grad = objective_gradient_sum(theta, x, y, rank=rank, c=c,
                                          with_factors=with_factors, chunk_rows=chunk_rows)
    return value / len(y), grad / len(y)


def score_logits(theta: np.ndarray, x: sparse.spmatrix, *, rank: int = 16,
                 with_factors: bool = True, chunk_rows: int = 1024) -> tuple[np.ndarray, np.ndarray | None]:
    matrix = _as_csr(x)
    w, b, v = unpack_parameters(theta, matrix.shape[1], rank, with_factors=with_factors)
    logits = np.empty(matrix.shape[0], dtype=np.float64)
    pair_all = np.zeros(matrix.shape[0], dtype=np.float64) if v is not None else None
    for start in range(0, matrix.shape[0], chunk_rows):
        end = min(start + chunk_rows, matrix.shape[0])
        block = matrix[start:end]
        pair = fm_pair_score(row_l2_normalize(block), v) if v is not None else 0.0
        logits[start:end] = np.asarray(block @ w, dtype=np.float64).reshape(-1) + b + pair
        if pair_all is not None:
            pair_all[start:end] = pair
    return logits, pair_all
