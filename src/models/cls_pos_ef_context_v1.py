"""Conditional positional-by-entry-friction logistic correction.

Pure helpers only: no repository discovery, file I/O, target loading, or fitting
at import time. The script layer owns source checks, persistence, and gates.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit


ORIGINS = (67, 75)
ARMS = ("REAL", "SHUFFLED")
FIT_IDS = tuple(f"CLS_POS_EF_CONTEXT_V1_E{o}_{a}" for o in ORIGINS for a in ARMS)
SEED = 42
RIDGE_C = 1.0
OPTIMIZER_OPTIONS = {"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-6, "maxls": 50}
PRESENT_THRESHOLD = 0.50
MISSING_THRESHOLD = 0.53230976129477


class ContextFitError(RuntimeError):
    """A one-shot supervised context fit did not satisfy the frozen recipe."""


def _vector(value: Any, dtype: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=dtype).reshape(-1)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def context_state(ef1: Sequence[int], present: Sequence[bool], *, minimum_per_level: int = 200) -> dict[str, Any]:
    """Compute prefix-present Bernoulli mean and unit-variance scale once."""
    ef = _vector(ef1, np.int8, "EF1")
    mask = np.asarray(present, dtype=bool).reshape(-1)
    if len(ef) != len(mask) or not np.isin(ef, (0, 1)).all():
        raise ValueError("EF1 and present mask must align; EF1 must be exactly 0/1")
    selected = ef[mask]
    counts = {str(level): int(np.count_nonzero(selected == level)) for level in (0, 1)}
    if any(n < minimum_per_level for n in counts.values()):
        raise ValueError(f"Present-prefix EF support is below {minimum_per_level}: {counts}")
    mu = float(selected.mean())
    scale = float(np.sqrt(mu * (1.0 - mu)))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("EF context scale is degenerate")
    return {"mu": mu, "scale": scale, "present_rows": int(mask.sum()), "ef_counts": counts,
            "minimum_per_level": int(minimum_per_level)}


def standardized_context(ef1: Sequence[int], state: Mapping[str, Any]) -> np.ndarray:
    ef = _vector(ef1, np.int8, "EF1")
    if not np.isin(ef, (0, 1)).all():
        raise ValueError("EF1 must be exactly 0/1")
    mu = float(state["mu"])
    scale = float(state["scale"])
    if not (0.0 < mu < 1.0 and np.isfinite(scale) and scale > 0.0):
        raise ValueError("Saved context state is invalid")
    return (ef.astype(np.float64) - mu) / scale


def shuffle_context_within_cycle(
    ef1: Sequence[int], cycles: Sequence[int], present: Sequence[bool], *, seed: int = SEED,
) -> tuple[np.ndarray, int]:
    """Shuffle EF assignments within exact cycle x availability strata.

    A fresh RNG is used per population so fit and query assignments are
    independently permuted with the frozen seed while preserving each stratum.
    """
    ef = _vector(ef1, np.int8, "EF1")
    cyc = _vector(cycles, np.int64, "cycle_code")
    avail = np.asarray(present, dtype=bool).reshape(-1)
    if len(ef) != len(cyc) or len(ef) != len(avail) or not np.isin(ef, (0, 1)).all():
        raise ValueError("Shuffled context inputs must align and EF1 must be binary")
    if seed != SEED:
        raise ValueError(f"Permutation seed is frozen at {SEED}")
    shuffled = ef.copy()
    rng = np.random.default_rng(seed)
    for cycle in np.unique(cyc):
        for available in (False, True):
            idx = np.flatnonzero((cyc == cycle) & (avail == available))
            if idx.size > 1:
                shuffled[idx] = ef[idx][rng.permutation(idx.size)]
    moved_present = int(np.count_nonzero((shuffled != ef) & avail))
    return shuffled, moved_present


def interaction_design(x_position: sparse.spmatrix, q: Sequence[float]) -> sparse.csr_matrix:
    x = sparse.csr_matrix(x_position, dtype=np.float64)
    context = _vector(q, np.float64, "standardized EF context")
    if x.shape[0] != len(context) or not np.isfinite(x.data).all():
        raise ValueError("Interaction feature rows or values are invalid")
    out = x.multiply(context[:, None]).tocsr()
    out.sum_duplicates()
    out.eliminate_zeros()
    out.sort_indices()
    return out


def objective_gradient(
    gamma: Sequence[float], x_interaction: sparse.spmatrix, z_pos: Sequence[float], y: Sequence[int], *,
    c: float = RIDGE_C,
) -> tuple[float, np.ndarray]:
    x = sparse.csr_matrix(x_interaction, dtype=np.float64)
    coef = _vector(gamma, np.float64, "gamma")
    offset = _vector(z_pos, np.float64, "z_POS")
    target_raw = np.asarray(y).reshape(-1)
    if (x.shape != (len(target_raw), len(coef)) or len(offset) != len(target_raw)
            or not np.isfinite(x.data).all()):
        raise ValueError("Interaction objective dimensions or values are invalid")
    if target_raw.dtype.kind not in "biuf" or not np.isfinite(target_raw).all() or not np.isin(target_raw, (0, 1)).all():
        raise ValueError("Objective target must be finite binary 0/1")
    if not np.isfinite(c) or c != RIDGE_C:
        raise ValueError("C=1 is frozen for this pilot")
    target = target_raw.astype(np.float64, copy=False)
    logits = offset + np.asarray(x @ coef, dtype=np.float64).reshape(-1)
    signed = (2.0 * target - 1.0) * logits
    loss = float(np.logaddexp(0.0, -signed).sum() + 0.5 * np.dot(coef, coef))
    grad = np.asarray(x.T @ (expit(logits) - target), dtype=np.float64).reshape(-1) + coef
    if not np.isfinite(loss) or not np.isfinite(grad).all():
        raise ValueError("Objective returned non-finite loss or gradient")
    return loss, grad


def fit_context_gamma(
    x_interaction: sparse.spmatrix, z_pos: Sequence[float], y: Sequence[int],
) -> tuple[np.ndarray, dict[str, Any]]:
    x = sparse.csr_matrix(x_interaction, dtype=np.float64)
    target = np.asarray(y, dtype=np.int8).reshape(-1)
    offset = _vector(z_pos, np.float64, "z_POS")
    if x.shape[0] != len(target) or len(offset) != len(target) or x.shape[1] == 0 or x.nnz == 0:
        raise ValueError("Context fit requires aligned nonempty rows and features")
    if not np.isin(target, (0, 1)).all() or not np.isfinite(x.data).all():
        raise ValueError("Context fit data must be finite and binary")
    zero = np.zeros(x.shape[1], dtype=np.float64)
    loss0, _ = objective_gradient(zero, x, offset, target)
    result = minimize(lambda g: objective_gradient(g, x, offset, target), zero,
                      method="L-BFGS-B", jac=True, options=dict(OPTIMIZER_OPTIONS))
    gamma = np.asarray(result.x, dtype=np.float64).reshape(-1)
    loss_final, grad = objective_gradient(gamma, x, offset, target)
    meta = {"optimizer": "scipy.optimize.minimize:L-BFGS-B", "success": bool(result.success),
            "message": str(result.message), "nit": int(result.nit), "nfev": int(result.nfev),
            "loss_gamma0": loss0, "loss_final": loss_final, "loss_change": loss_final - loss0,
            "gradient_inf_norm": float(np.max(np.abs(grad))) if grad.size else 0.0,
            "c": RIDGE_C, "penalty": "0.5 * sum(gamma**2)", "initialization": "zeros",
            "options": dict(OPTIMIZER_OPTIONS)}
    if (not result.success or not np.isfinite(gamma).all() or not np.isfinite(loss_final)
            or not np.isfinite(grad).all() or loss_final > loss0 + 1e-8):
        raise ContextFitError(f"Frozen context optimizer failed acceptance: {meta}")
    return gamma, meta


def predict_context_probability(
    x_interaction: sparse.spmatrix, z_pos: Sequence[float], gamma: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    x = sparse.csr_matrix(x_interaction, dtype=np.float64)
    offset = _vector(z_pos, np.float64, "z_POS")
    coef = _vector(gamma, np.float64, "gamma")
    if x.shape != (len(offset), len(coef)):
        raise ValueError("Context prediction dimensions do not align")
    delta = np.asarray(x @ coef, dtype=np.float64).reshape(-1)
    p = np.asarray(expit(offset + delta), dtype=np.float64).reshape(-1)
    if not np.isfinite(p).all() or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("Context prediction is outside finite [0,1]")
    return p, delta


def route_candidate(
    p_a02: Sequence[float], p_pos_context: Sequence[float], present: Sequence[bool],
) -> np.ndarray:
    p_a = _vector(p_a02, np.float64, "A02 probability")
    p_pos = _vector(p_pos_context, np.float64, "POS-context probability")
    avail = np.asarray(present, dtype=bool).reshape(-1)
    if len(p_a) != len(p_pos) or len(p_a) != len(avail):
        raise ValueError("Candidate routing inputs do not align")
    routed = p_a.copy()
    routed[avail] = 0.5 * p_a[avail] + 0.5 * p_pos[avail]
    if not np.array_equal(routed[~avail].view(np.uint64), p_a[~avail].view(np.uint64)):
        raise RuntimeError("Missing-stream route must be bit-exact A02")
    if not np.isfinite(routed).all() or np.any((routed < 0.0) | (routed > 1.0)):
        raise ValueError("Routed probability is invalid")
    return routed


def apply_thresholds(probability: Sequence[float], present: Sequence[bool]) -> np.ndarray:
    p = _vector(probability, np.float64, "probability")
    avail = np.asarray(present, dtype=bool).reshape(-1)
    if len(p) != len(avail):
        raise ValueError("Threshold inputs do not align")
    threshold = np.where(avail, PRESENT_THRESHOLD, MISSING_THRESHOLD)
    return (p >= threshold).astype(np.int8)


def binary_metrics(y: Sequence[int], probability: Sequence[float], labels: Sequence[int]) -> dict[str, Any]:
    from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                                 log_loss, precision_score, recall_score, roc_auc_score)
    target = np.asarray(y, dtype=np.int8).reshape(-1)
    prob = _vector(probability, np.float64, "probability")
    pred = np.asarray(labels, dtype=np.int8).reshape(-1)
    if len(target) != len(prob) or len(target) != len(pred) or not np.isin(target, (0, 1)).all() or not np.isin(pred, (0, 1)).all():
        raise ValueError("Binary metric inputs are misaligned or invalid")
    if len(target) == 0:
        raise ValueError("Cannot score an empty view")
    tn, fp, fn, tp = confusion_matrix(target, pred, labels=[0, 1]).ravel().tolist()
    classes = np.unique(target)
    auc = float(roc_auc_score(target, prob)) if len(classes) == 2 else None
    ap = float(average_precision_score(target, prob)) if len(classes) == 2 else None
    clipped = np.clip(prob, np.finfo(np.float64).eps, 1.0 - np.finfo(np.float64).eps)
    return {"n": int(len(target)), "positive": int(target.sum()), "negative": int(len(target) - target.sum()),
            "macro_f1": float(f1_score(target, pred, labels=[0, 1], average="macro", zero_division=0)),
            "positive_precision": float(precision_score(target, pred, pos_label=1, zero_division=0)),
            "positive_recall": float(recall_score(target, pred, pos_label=1, zero_division=0)),
            "roc_auc": auc, "average_precision": ap,
            "log_loss": float(log_loss(target, clipped, labels=[0, 1])),
            "brier": float(np.mean((prob - target) ** 2)),
            "confusion_matrix_labels_0_1": [[int(tn), int(fp)], [int(fn), int(tp)]]}


def metric_comparison(y: Sequence[int], p_champion: Sequence[float], p_candidate: Sequence[float],
                      yhat_champion: Sequence[int], yhat_candidate: Sequence[int]) -> dict[str, Any]:
    champion = binary_metrics(y, p_champion, yhat_champion)
    candidate = binary_metrics(y, p_candidate, yhat_candidate)
    return {"support": {"n": candidate["n"], "positive": candidate["positive"], "negative": candidate["negative"]},
            "champion": champion, "candidate": candidate,
            "gain_macro_f1": candidate["macro_f1"] - champion["macro_f1"],
            "gain_positive_recall": candidate["positive_recall"] - champion["positive_recall"],
            "gain_positive_precision": candidate["positive_precision"] - champion["positive_precision"],
            "gain_roc_auc": None if candidate["roc_auc"] is None or champion["roc_auc"] is None else candidate["roc_auc"] - champion["roc_auc"],
            "gain_average_precision": None if candidate["average_precision"] is None or champion["average_precision"] is None else candidate["average_precision"] - champion["average_precision"],
            "gain_log_loss": champion["log_loss"] - candidate["log_loss"],
            "gain_brier": champion["brier"] - candidate["brier"]}


def delta_logit_summary(value: Sequence[float]) -> dict[str, float | int]:
    delta = _vector(value, np.float64, "delta logit")
    if delta.size == 0:
        return {"n": 0, "l2_norm": 0.0, "rms": 0.0, "mean_abs": 0.0, "median_abs": 0.0, "max_abs": 0.0}
    absolute = np.abs(delta)
    return {"n": int(delta.size), "l2_norm": float(np.linalg.norm(delta)), "rms": float(np.sqrt(np.mean(delta**2))),
            "mean_abs": float(absolute.mean()), "median_abs": float(np.median(absolute)), "max_abs": float(absolute.max())}


def supported(mask: Sequence[bool], y: Sequence[int], *, min_n: int = 200, min_per_class: int = 30) -> dict[str, Any]:
    group = np.asarray(mask, dtype=bool).reshape(-1)
    target = np.asarray(y, dtype=np.int8).reshape(-1)
    if len(group) != len(target):
        raise ValueError("Subgroup support mask does not align")
    positive = int(target[group].sum())
    n = int(group.sum())
    negative = n - positive
    return {"n": n, "positive": positive, "negative": negative,
            "supported": bool(n >= min_n and positive >= min_per_class and negative >= min_per_class),
            "minimum_n": int(min_n), "minimum_per_class": int(min_per_class)}
