"""Frozen-bag logistic offset fit for POS_BAG_OFFSET_V1.

This module fits only the positional coefficient vector.  The saved bag model
defines a fixed logit offset and is never modified.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit


class OffsetFitError(RuntimeError):
    """A one-shot authorized optimizer did not satisfy its frozen contract."""


def offset_objective_gradient(
    gamma: np.ndarray,
    x_position: sparse.spmatrix,
    z0: np.ndarray,
    y: np.ndarray,
    *,
    c: float = 1.0,
) -> tuple[float, np.ndarray]:
    """SUM logistic loss with fixed offset and L2 penalty; return loss, grad."""
    x = sparse.csr_matrix(x_position, dtype=np.float64)
    coef = np.asarray(gamma, dtype=np.float64).reshape(-1)
    offset = np.asarray(z0, dtype=np.float64).reshape(-1)
    target = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.shape != (len(target), len(coef)) or len(offset) != len(target):
        raise ValueError("offset optimizer row/column dimensions do not align")
    if not np.isfinite(x.data).all() or not np.isfinite(coef).all():
        raise ValueError("offset optimizer matrix/coefficients must be finite")
    if not np.isfinite(offset).all() or not np.isfinite(target).all():
        raise ValueError("offset/target values must be finite")
    if not np.isin(target, (0.0, 1.0)).all() or not np.isfinite(c) or c <= 0:
        raise ValueError("binary target and positive finite C are required")
    z = offset + np.asarray(x @ coef, dtype=np.float64).reshape(-1)
    signed = (2.0 * target - 1.0) * z
    loss = float(np.logaddexp(0.0, -signed).sum() + np.dot(coef, coef) / (2.0 * c))
    residual = expit(z) - target
    gradient = np.asarray(x.T @ residual, dtype=np.float64).reshape(-1) + coef / c
    return loss, gradient


def fit_offset_coefficients(
    x_position: sparse.spmatrix,
    z0: np.ndarray,
    y: np.ndarray,
    *,
    c: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit gamma exactly once from zero with the authorized L-BFGS-B recipe."""
    x = sparse.csr_matrix(x_position, dtype=np.float64)
    target = np.asarray(y, dtype=np.int8).reshape(-1)
    offset = np.asarray(z0, dtype=np.float64).reshape(-1)
    if x.shape[0] != len(target) or len(offset) != len(target):
        raise ValueError("offset fit rows do not align")
    if not np.isfinite(x.data).all() or not np.isfinite(offset).all():
        raise ValueError("offset fit inputs must be finite")
    if not np.isin(target, (0, 1)).all():
        raise ValueError("offset fit target must be binary 0/1")
    zero = np.zeros(x.shape[1], dtype=np.float64)
    loss0, _ = offset_objective_gradient(zero, x, offset, target, c=c)
    if x.shape[1] == 0 or x.nnz == 0:
        meta = {
            "optimizer": "scipy.optimize.minimize:L-BFGS-B",
            "success": True,
            "message": "empty positional block; gamma remains exactly zero",
            "nit": 0,
            "nfev": 1,
            "gradient_inf_norm": 0.0,
            "loss_gamma0": loss0,
            "loss_final": loss0,
            "loss_change": 0.0,
            "empty_block": True,
            "c": float(c),
            "options": {"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-6, "maxls": 50},
        }
        return zero, meta

    result = minimize(
        lambda candidate: offset_objective_gradient(candidate, x, offset, target, c=c),
        zero,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-6, "maxls": 50},
    )
    gamma = np.asarray(result.x, dtype=np.float64).reshape(-1)
    loss_final, gradient = offset_objective_gradient(gamma, x, offset, target, c=c)
    grad_inf = float(np.max(np.abs(gradient))) if len(gradient) else 0.0
    meta = {
        "optimizer": "scipy.optimize.minimize:L-BFGS-B",
        "success": bool(result.success),
        "message": str(result.message),
        "nit": int(result.nit),
        "nfev": int(result.nfev),
        "gradient_inf_norm": grad_inf,
        "loss_gamma0": float(loss0),
        "loss_final": float(loss_final),
        "loss_change": float(loss_final - loss0),
        "empty_block": False,
        "c": float(c),
        "options": {"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-6, "maxls": 50},
    }
    if (not result.success or not np.isfinite(gamma).all()
            or not np.isfinite(loss_final) or not np.isfinite(gradient).all()
            or loss_final > loss0 + 1e-8):
        raise OffsetFitError(f"authorized L-BFGS-B fit failed frozen acceptance: {meta}")
    return gamma, meta


def offset_probability(x_position: sparse.spmatrix, z0: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Apply the fixed bag logit plus fitted positional offset."""
    x = sparse.csr_matrix(x_position, dtype=np.float64)
    coef = np.asarray(gamma, dtype=np.float64).reshape(-1)
    offset = np.asarray(z0, dtype=np.float64).reshape(-1)
    if x.shape != (len(offset), len(coef)):
        raise ValueError("offset prediction rows/columns do not align")
    probability = np.asarray(expit(offset + x @ coef), dtype=np.float64).reshape(-1)
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("offset probabilities are not finite [0,1]")
    return probability
