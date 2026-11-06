"""Pure helpers for the fixed CLS_POS_RECENCY_V1 gamma-only pilot."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit


CANDIDATE_ID = "CLS_POS_RECENCY_V1"
ORIGINS = (67, 75)
ARMS = ("RECENCY", "SHUFFLED_WITHIN_CLASS")
FIT_IDS = (
    "CLS_POS_RECENCY_V1_E67_RECENCY",
    "CLS_POS_RECENCY_V1_E67_SHUFFLED_WITHIN_CLASS",
    "CLS_POS_RECENCY_V1_E75_RECENCY",
    "CLS_POS_RECENCY_V1_E75_SHUFFLED_WITHIN_CLASS",
)
FIT_ID_MAP = {
    "CLS_POS_RECENCY_V1_E67_RECENCY": (67, "RECENCY"),
    "CLS_POS_RECENCY_V1_E67_SHUFFLED_WITHIN_CLASS": (67, "SHUFFLED_WITHIN_CLASS"),
    "CLS_POS_RECENCY_V1_E75_RECENCY": (75, "RECENCY"),
    "CLS_POS_RECENCY_V1_E75_SHUFFLED_WITHIN_CLASS": (75, "SHUFFLED_WITHIN_CLASS"),
}
HALF_LIFE_CYCLES = 8.0
WEIGHT_FLOOR = 0.25
SHUFFLE_SEED = 42
RIDGE_C = 1.0
OPTIMIZER_OPTIONS = {"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-6, "maxls": 50}
PRESENT_THRESHOLD = 0.50
MISSING_THRESHOLD = 0.53230976129477
BLEND_WEIGHT_POS = 0.50


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def fit_id_parts(fit_id: str) -> tuple[int, str]:
    require(fit_id in FIT_ID_MAP, f"Unreserved fit ID: {fit_id}")
    return FIT_ID_MAP[fit_id]


def recency_base_weights(cycles: Sequence[int], origin: int) -> np.ndarray:
    raw_cycle = np.asarray(cycles).reshape(-1)
    require(origin in ORIGINS and len(raw_cycle) > 0
            and raw_cycle.dtype.kind in "biuf" and np.isfinite(raw_cycle).all()
            and np.equal(raw_cycle, np.floor(raw_cycle)).all(),
            "Recency cycles must be finite integer cycle codes")
    cycle = raw_cycle.astype(np.int64, copy=False)
    require(np.all((cycle >= 47) & (cycle <= origin)),
            "Recency cycles must stay inside the frozen fit prefix")
    age = origin - cycle
    return np.maximum(WEIGHT_FLOOR, np.power(2.0, -age.astype(np.float64) / HALF_LIFE_CYCLES))


def normalize_class_mass(base: Sequence[float], y: Sequence[int]) -> np.ndarray:
    raw = np.asarray(base, dtype=np.float64).reshape(-1)
    target_raw = np.asarray(y).reshape(-1)
    require(target_raw.dtype.kind in "biuf" and np.isfinite(target_raw).all()
            and np.isin(target_raw, (0, 1)).all(),
            "Weight normalization target must be binary 0/1")
    target = target_raw.astype(np.int8, copy=False)
    require(len(raw) == len(target) and len(raw) > 0 and np.isfinite(raw).all()
            and np.all(raw > 0.0) and np.isin(target, (0, 1)).all(),
            "Weight and binary target vectors are invalid")
    weights = np.empty_like(raw)
    for label in (0, 1):
        mask = target == label
        require(np.any(mask), f"Both classes are required for class-normalized weights; absent {label}")
        mean = float(raw[mask].mean())
        require(np.isfinite(mean) and mean > 0.0, "Class mean weight is invalid")
        weights[mask] = raw[mask] / mean
    require(np.isfinite(weights).all() and np.all(weights > 0.0), "Normalized recency weights invalid")
    for label in (0, 1):
        mask = target == label
        require(np.isclose(float(weights[mask].sum()), float(mask.sum()), rtol=1e-12, atol=1e-10),
                f"Normalized class-{label} weight mass differs from row count")
    return weights


def shuffle_weights_within_class(
    weights: Sequence[float], y: Sequence[int], *, seed: int = SHUFFLE_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(weights, dtype=np.float64).reshape(-1)
    target_raw = np.asarray(y).reshape(-1)
    require(target_raw.dtype.kind in "biuf" and np.isfinite(target_raw).all()
            and np.isin(target_raw, (0, 1)).all(),
            "Weight-shuffle target must be binary 0/1")
    target = target_raw.astype(np.int8, copy=False)
    require(seed == SHUFFLE_SEED and len(raw) == len(target) and len(raw) > 0
            and np.isfinite(raw).all() and np.isin(target, (0, 1)).all(),
            "The control is limited to the frozen seed-42 class-wise weight shuffle")
    rng = np.random.default_rng(seed)
    shuffled = np.empty_like(raw)
    permutation = np.arange(len(raw), dtype=np.int64)
    for label in (0, 1):
        idx = np.flatnonzero(target == label)  # Stable official input order.
        order = rng.permutation(len(idx))
        shuffled[idx] = raw[idx[order]]
        permutation[idx] = idx[order]
    require(np.array_equal(np.sort(shuffled[target == 0]), np.sort(raw[target == 0]))
            and np.array_equal(np.sort(shuffled[target == 1]), np.sort(raw[target == 1])),
            "Within-class control changed a class-specific weight multiset")
    return shuffled, permutation


def effective_sample_size(weights: Sequence[float]) -> float:
    value = np.asarray(weights, dtype=np.float64).reshape(-1)
    require(len(value) > 0 and np.isfinite(value).all() and np.all(value > 0.0),
            "ESS weights must be finite and positive")
    return float(value.sum() ** 2 / np.dot(value, value))


def weight_diagnostics(weights: Sequence[float], y: Sequence[int]) -> dict[str, Any]:
    value = np.asarray(weights, dtype=np.float64).reshape(-1)
    target_raw = np.asarray(y).reshape(-1)
    require(target_raw.dtype.kind in "biuf" and np.isfinite(target_raw).all()
            and np.isin(target_raw, (0, 1)).all(),
            "Weight diagnostic target must be binary 0/1")
    target = target_raw.astype(np.int8, copy=False)
    require(len(value) == len(target) and np.isin(target, (0, 1)).all(), "Weight diagnostics misaligned")
    out: dict[str, Any] = {
        "rows": int(len(value)),
        "weight_sum": float(value.sum()),
        "weight_min": float(value.min()),
        "weight_median": float(np.median(value)),
        "weight_max": float(value.max()),
        "classes": {},
    }
    for label in (0, 1):
        mask = target == label
        total = float(value[mask].sum())
        ess = effective_sample_size(value[mask])
        out["classes"][str(label)] = {
            "rows": int(mask.sum()),
            "weight_sum": total,
            "weight_sum_minus_rows": total - int(mask.sum()),
            "ess": ess,
            "ess_ratio": ess / int(mask.sum()),
        }
    return out


def weighted_offset_objective_gradient(
    gamma: Sequence[float], x_position: sparse.spmatrix, z_bag: Sequence[float],
    y: Sequence[int], weights: Sequence[float], *, c: float = RIDGE_C,
) -> tuple[float, np.ndarray]:
    x = sparse.csr_matrix(x_position, dtype=np.float64)
    coef = np.asarray(gamma, dtype=np.float64).reshape(-1)
    offset = np.asarray(z_bag, dtype=np.float64).reshape(-1)
    target_raw = np.asarray(y).reshape(-1)
    weight = np.asarray(weights, dtype=np.float64).reshape(-1)
    require(c == RIDGE_C and x.shape == (len(target_raw), len(coef)) and len(offset) == len(weight) == len(target_raw),
            "Weighted offset objective dimensions or C differ from the frozen recipe")
    require(target_raw.dtype.kind in "biuf" and np.isfinite(target_raw).all()
            and np.isin(target_raw, (0, 1)).all(), "Objective target must be binary 0/1")
    target = target_raw.astype(np.float64, copy=False)
    require(np.isfinite(x.data).all() and np.isfinite(coef).all() and np.isfinite(offset).all()
            and np.isfinite(weight).all() and np.all(weight > 0.0),
            "Objective inputs must be finite and weights positive")
    logits = offset + np.asarray(x @ coef, dtype=np.float64).reshape(-1)
    loss = float(np.dot(weight, np.logaddexp(0.0, -(2.0 * target - 1.0) * logits))
                 + 0.5 * np.dot(coef, coef) / c)
    residual = weight * (expit(logits) - target)
    gradient = np.asarray(x.T @ residual, dtype=np.float64).reshape(-1) + coef / c
    require(np.isfinite(loss) and np.isfinite(gradient).all(), "Objective returned non-finite values")
    return loss, gradient


def fit_weighted_gamma(
    x_position: sparse.spmatrix, z_bag: Sequence[float], y: Sequence[int], weights: Sequence[float],
) -> tuple[np.ndarray, dict[str, Any]]:
    x = sparse.csr_matrix(x_position, dtype=np.float64)
    target_raw = np.asarray(y).reshape(-1)
    require(target_raw.dtype.kind in "biuf" and np.isfinite(target_raw).all()
            and np.isin(target_raw, (0, 1)).all(), "Gamma fit target must be binary 0/1")
    target = target_raw.astype(np.int8, copy=False)
    require(x.shape[0] == len(target) and x.shape[1] > 0 and x.nnz > 0,
            "Gamma fit matrix is empty or misaligned")
    require(len(np.unique(target)) == 2, "Both target classes are required")
    zero = np.zeros(x.shape[1], dtype=np.float64)
    loss_zero, _ = weighted_offset_objective_gradient(zero, x, z_bag, target, weights)
    result = minimize(
        lambda value: weighted_offset_objective_gradient(value, x, z_bag, target, weights),
        zero, method="L-BFGS-B", jac=True, options=dict(OPTIMIZER_OPTIONS),
    )
    gamma = np.asarray(result.x, dtype=np.float64).reshape(-1)
    loss_final, gradient = weighted_offset_objective_gradient(gamma, x, z_bag, target, weights)
    metadata = {
        "optimizer": "scipy.optimize.minimize:L-BFGS-B",
        "success": bool(result.success),
        "message": str(result.message),
        "nit": int(result.nit),
        "nfev": int(result.nfev),
        "gradient_inf_norm": float(np.max(np.abs(gradient))) if len(gradient) else 0.0,
        "loss_gamma0": loss_zero,
        "loss_final": loss_final,
        "loss_change": loss_final - loss_zero,
        "weight_sum": float(np.asarray(weights, dtype=np.float64).sum()),
        "ridge_c": RIDGE_C,
        "penalty": "0.5 * sum(gamma**2)",
        "options": dict(OPTIMIZER_OPTIONS),
    }
    require(bool(result.success), f"L-BFGS-B failed; fit consumed, no retry: {metadata}")
    require(np.isfinite(gamma).all() and np.isfinite(gradient).all()
            and loss_final <= loss_zero + 1e-8,
            f"L-BFGS-B output failed acceptance; fit consumed, no retry: {metadata}")
    return gamma, metadata


def offset_probability(x_position: sparse.spmatrix, z_bag: Sequence[float], gamma: Sequence[float]) -> np.ndarray:
    x = sparse.csr_matrix(x_position, dtype=np.float64)
    offset = np.asarray(z_bag, dtype=np.float64).reshape(-1)
    coef = np.asarray(gamma, dtype=np.float64).reshape(-1)
    require(x.shape == (len(offset), len(coef)) and np.isfinite(x.data).all()
            and np.isfinite(offset).all() and np.isfinite(coef).all(),
            "Prediction offset, matrix and gamma do not align")
    probability = np.asarray(expit(offset + x @ coef), dtype=np.float64).reshape(-1)
    require(np.isfinite(probability).all() and np.all((probability >= 0.0) & (probability <= 1.0)),
            "Predicted probability outside finite [0,1]")
    return probability


def route_probabilities(
    present: Sequence[bool], p_a02: Sequence[float], p_pos: Sequence[float],
) -> np.ndarray:
    available = np.asarray(present, dtype=bool).reshape(-1)
    p_a = np.asarray(p_a02, dtype=np.float64).reshape(-1)
    p_position = np.asarray(p_pos, dtype=np.float64).reshape(-1)
    require(len(available) == len(p_a) == len(p_position)
            and np.isfinite(p_a).all() and np.isfinite(p_position).all()
            and np.all((p_a >= 0.0) & (p_a <= 1.0))
            and np.all((p_position >= 0.0) & (p_position <= 1.0)),
            "Policy route inputs are invalid or misaligned")
    result = p_a.copy()
    result[available] = BLEND_WEIGHT_POS * p_position[available] + (1.0 - BLEND_WEIGHT_POS) * p_a[available]
    require(np.array_equal(result[~available].view(np.uint64), p_a[~available].view(np.uint64)),
            "Missing-stream route must use pA02 bit-exactly")
    return result


def decisions(probability: Sequence[float], present: Sequence[bool]) -> np.ndarray:
    p = np.asarray(probability, dtype=np.float64).reshape(-1)
    available = np.asarray(present, dtype=bool).reshape(-1)
    require(len(p) == len(available) and np.isfinite(p).all()
            and np.all((p >= 0.0) & (p <= 1.0)), "Decision inputs invalid")
    threshold = np.where(available, PRESENT_THRESHOLD, MISSING_THRESHOLD)
    return (p >= threshold).astype(np.int8)
