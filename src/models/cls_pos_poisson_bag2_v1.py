"""Fixed two-replica Poisson-weighted positional-gamma averaging candidate.

Pure helpers only: no data loading, target access, model fitting at import time,
artifact reads, or writes.  The experiment runner owns all I/O and gates.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit


FIT_IDS = (
    "CLS_POS_POISSON_BAG2_V1_E67_P17",
    "CLS_POS_POISSON_BAG2_V1_E67_P137",
    "CLS_POS_POISSON_BAG2_V1_E75_P17",
    "CLS_POS_POISSON_BAG2_V1_E75_P137",
)
ORIGINS = (67, 75)
REPLICA_SEEDS = (17, 137)
RIDGE_C = 1.0
OPTIMIZER_OPTIONS = {"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-6, "maxls": 50}
PRESENT_THRESHOLD = 0.50
MISSING_THRESHOLD = 0.53230976129477


def require(condition: bool, message: str) -> None:
    """Keep safety gates active under ``python -O``."""
    if not condition:
        raise RuntimeError(message)


def poisson_row_weights(n_rows: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    require(n_rows > 0, "Poisson weighting requires at least one fit row")
    require(seed in REPLICA_SEEDS, f"Unfrozen Poisson replica seed: {seed}")
    counts = np.random.default_rng(seed).poisson(lam=1.0, size=n_rows).astype(np.int64)
    count_sum = int(counts.sum())
    require(count_sum > 0, "Poisson replica sampled zero total multiplicity")
    weights = counts.astype(np.float64) * (float(n_rows) / float(count_sum))
    require(np.isfinite(weights).all() and np.all(weights >= 0.0), "Normalized Poisson weights are invalid")
    require(np.isclose(float(weights.sum()), float(n_rows), rtol=0.0, atol=1e-10 * max(1, n_rows)),
            "Normalized Poisson weights do not sum to the fit-row count")
    return counts, weights


def weighted_offset_objective_gradient(
    gamma: Sequence[float],
    x_position: sparse.spmatrix,
    offset: Sequence[float],
    y: Sequence[int],
    row_weights: Sequence[float],
) -> tuple[float, np.ndarray]:
    """Summed multiplicity-weighted logistic loss with the frozen L2 penalty."""
    x = sparse.csr_matrix(x_position, dtype=np.float64)
    coef = np.asarray(gamma, dtype=np.float64).reshape(-1)
    z0 = np.asarray(offset, dtype=np.float64).reshape(-1)
    target_raw = np.asarray(y).reshape(-1)
    weights = np.asarray(row_weights, dtype=np.float64).reshape(-1)
    require(x.shape == (len(target_raw), len(coef)), "Position matrix/target/gamma dimensions do not align")
    require(len(z0) == len(target_raw) and len(weights) == len(target_raw),
            "Offset/row-weight dimensions do not align")
    require(target_raw.dtype.kind in "biuf" and np.isfinite(target_raw).all()
            and np.isin(target_raw, (0, 1)).all(), "Target must be finite binary 0/1")
    target = target_raw.astype(np.float64, copy=False)
    require(np.isfinite(x.data).all() and np.isfinite(coef).all() and np.isfinite(z0).all()
            and np.isfinite(weights).all() and np.all(weights >= 0.0),
            "Weighted objective inputs must be finite and row weights nonnegative")
    require(float(weights.sum()) > 0.0, "Weighted objective has zero total row weight")
    z = z0 + np.asarray(x @ coef, dtype=np.float64).reshape(-1)
    signed = (2.0 * target - 1.0) * z
    loss = float(np.dot(weights, np.logaddexp(0.0, -signed))
                 + 0.5 * np.dot(coef, coef) / RIDGE_C)
    residual = weights * (expit(z) - target)
    gradient = np.asarray(x.T @ residual, dtype=np.float64).reshape(-1) + coef / RIDGE_C
    require(np.isfinite(loss) and np.isfinite(gradient).all(), "Weighted objective returned non-finite values")
    return loss, gradient


def fit_poisson_weighted_offset(
    x_position: sparse.spmatrix,
    offset: Sequence[float],
    y: Sequence[int],
    row_weights: Sequence[float],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit only gamma, from zero, with the inherited L-BFGS-B recipe."""
    x = sparse.csr_matrix(x_position, dtype=np.float64)
    target_raw = np.asarray(y).reshape(-1)
    require(target_raw.dtype.kind in "biuf" and np.isfinite(target_raw).all()
            and np.isin(target_raw, (0, 1)).all(), "Fit target must be finite binary 0/1")
    target = target_raw.astype(np.int8, copy=False)
    z0 = np.asarray(offset, dtype=np.float64).reshape(-1)
    weights = np.asarray(row_weights, dtype=np.float64).reshape(-1)
    require(x.shape[0] == len(target) and len(z0) == len(target)
            and len(weights) == len(target), "Fit row dimensions do not align")
    require(x.shape[1] > 0 and x.nnz > 0, "Frozen positional feature matrix is empty")
    require(np.isfinite(x.data).all() and np.isfinite(z0).all()
            and np.isfinite(weights).all() and np.all(weights >= 0.0),
            "Fit inputs are invalid")
    require(np.isclose(float(weights.sum()), float(len(target)), rtol=0.0,
                       atol=1e-10 * max(1, len(target))),
            "Poisson weights must be normalized to sum to n_fit")
    zero = np.zeros(x.shape[1], dtype=np.float64)
    loss_zero, _ = weighted_offset_objective_gradient(zero, x, z0, target, weights)
    result = minimize(
        lambda candidate: weighted_offset_objective_gradient(candidate, x, z0, target, weights),
        zero,
        method="L-BFGS-B",
        jac=True,
        options=dict(OPTIMIZER_OPTIONS),
    )
    gamma = np.asarray(result.x, dtype=np.float64).reshape(-1)
    loss_final, gradient = weighted_offset_objective_gradient(gamma, x, z0, target, weights)
    grad_inf = float(np.max(np.abs(gradient))) if gradient.size else 0.0
    metadata = {
        "optimizer": "scipy.optimize.minimize:L-BFGS-B",
        "success": bool(result.success),
        "message": str(result.message),
        "nit": int(result.nit),
        "nfev": int(result.nfev),
        "gradient_inf_norm": grad_inf,
        "loss_gamma0": loss_zero,
        "loss_final": loss_final,
        "loss_change": loss_final - loss_zero,
        "row_weights_sum": float(weights.sum()),
        "row_weights_zero_count": int(np.count_nonzero(weights == 0.0)),
        "ridge_c": RIDGE_C,
        "penalty": "0.5 * sum(gamma**2)",
        "options": dict(OPTIMIZER_OPTIONS),
    }
    require(bool(result.success), f"L-BFGS-B failed; fit consumed, no retry: {metadata}")
    require(np.isfinite(gamma).all() and np.isfinite(loss_final) and np.isfinite(gradient).all()
            and loss_final <= loss_zero + 1e-8,
            f"L-BFGS-B output failed frozen acceptance; fit consumed, no retry: {metadata}")
    return gamma, metadata


def offset_probability(x_position: sparse.spmatrix, offset: Sequence[float], gamma: Sequence[float]) -> np.ndarray:
    x = sparse.csr_matrix(x_position, dtype=np.float64)
    z0 = np.asarray(offset, dtype=np.float64).reshape(-1)
    coef = np.asarray(gamma, dtype=np.float64).reshape(-1)
    require(x.shape == (len(z0), len(coef)), "Prediction matrix/offset/gamma dimensions do not align")
    probability = np.asarray(expit(z0 + x @ coef), dtype=np.float64).reshape(-1)
    require(np.isfinite(probability).all() and np.all((probability >= 0.0) & (probability <= 1.0)),
            "Offset probability is outside finite [0,1]")
    return probability


def build_policy_probabilities(
    x_position: sparse.spmatrix,
    offset: Sequence[float],
    gamma_original: Sequence[float],
    gamma_seed17: Sequence[float],
    gamma_seed137: Sequence[float],
    p_a02: Sequence[float],
    present_mask: Sequence[bool],
) -> dict[str, np.ndarray]:
    """Return fixed champion and bag2 candidate probabilities on same rows."""
    present = np.asarray(present_mask, dtype=bool).reshape(-1)
    p_a = np.asarray(p_a02, dtype=np.float64).reshape(-1)
    z0 = np.asarray(offset, dtype=np.float64).reshape(-1)
    require(len(present) == len(p_a) == len(z0), "Policy route inputs do not align")
    require(np.isfinite(p_a).all() and np.all((p_a >= 0.0) & (p_a <= 1.0)), "A02 probabilities invalid")
    p_original = offset_probability(x_position, z0, gamma_original)
    p17 = offset_probability(x_position, z0, gamma_seed17)
    p137 = offset_probability(x_position, z0, gamma_seed137)
    p_pos_bag = (p_original + p17 + p137) / 3.0
    champion = p_a.copy()
    candidate = p_a.copy()
    champion[present] = 0.5 * p_original[present] + 0.5 * p_a[present]
    candidate[present] = 0.5 * p_pos_bag[present] + 0.5 * p_a[present]
    require(np.array_equal(champion[~present].view(np.uint64), p_a[~present].view(np.uint64)),
            "Champion missing route differs from exact A02 fallback")
    require(np.array_equal(candidate[~present].view(np.uint64), p_a[~present].view(np.uint64)),
            "Candidate missing route differs from exact A02 fallback")
    require(np.isfinite(champion).all() and np.isfinite(candidate).all()
            and np.all((champion >= 0.0) & (champion <= 1.0))
            and np.all((candidate >= 0.0) & (candidate <= 1.0)), "Policy probabilities invalid")
    return {
        "probability_champion": champion,
        "probability_candidate": candidate,
        "probability_pos_original": p_original,
        "probability_pos_bag2": p_pos_bag,
    }


def apply_fixed_thresholds(probability: Sequence[float], present_mask: Sequence[bool]) -> np.ndarray:
    p = np.asarray(probability, dtype=np.float64).reshape(-1)
    present = np.asarray(present_mask, dtype=bool).reshape(-1)
    require(len(p) == len(present) and np.isfinite(p).all()
            and np.all((p >= 0.0) & (p <= 1.0)), "Threshold inputs invalid")
    threshold = np.where(present, PRESENT_THRESHOLD, MISSING_THRESHOLD)
    return (p >= threshold).astype(np.int8)


def metric_summary(y: Sequence[int], probability: Sequence[float], present_mask: Sequence[bool]) -> dict[str, Any]:
    from sklearn.metrics import f1_score, precision_score, recall_score

    target = np.asarray(y, dtype=np.int8).reshape(-1)
    p = np.asarray(probability, dtype=np.float64).reshape(-1)
    present = np.asarray(present_mask, dtype=bool).reshape(-1)
    require(len(target) == len(p) == len(present), "Metric arrays do not align")
    require(np.isin(target, (0, 1)).all(), "Metric target must be binary")
    labels = apply_fixed_thresholds(p, present)
    return {
        "n": int(len(target)),
        "positive": int(target.sum()),
        "negative": int(len(target) - target.sum()),
        "macro_f1": float(f1_score(target, labels, average="macro", labels=[0, 1], zero_division=0)),
        "positive_recall": float(recall_score(target, labels, pos_label=1, zero_division=0)),
        "precision": float(precision_score(target, labels, pos_label=1, zero_division=0)),
        "predicted_positive": int(labels.sum()),
    }
