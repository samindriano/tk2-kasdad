"""Probability policy for CLS_BIGRAM_POSITION_PILOT_V1.

All fitted coefficients are supplied by the original frozen offset solver.
This module only applies a new bigram gamma and the predeclared routes.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse

from src.models.positional_bag_offset_v1 import offset_probability


def reconstruct_original_pos_logit(
    base_matrix: sparse.spmatrix,
    bag_coef: Any,
    bag_intercept: Any,
    unigram_position_matrix: sparse.spmatrix,
    original_gamma: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct C2B bag + original POS gamma in the producer's float64 order."""
    base = sparse.csr_matrix(base_matrix, dtype=np.float64)
    position = sparse.csr_matrix(unigram_position_matrix, dtype=np.float64)
    coef = np.asarray(bag_coef, dtype=np.float64).reshape(-1)
    intercept = np.asarray(bag_intercept, dtype=np.float64).reshape(-1)
    gamma = np.asarray(original_gamma, dtype=np.float64).reshape(-1)
    if (base.shape[1] != len(coef) or len(intercept) != 1
            or position.shape[0] != base.shape[0] or position.shape[1] != len(gamma)):
        raise ValueError("frozen bag/POS matrix dimensions do not align")
    if (not np.isfinite(base.data).all() or not np.isfinite(position.data).all()
            or not np.isfinite(coef).all() or not np.isfinite(intercept).all()
            or not np.isfinite(gamma).all()):
        raise ValueError("frozen bag/POS inputs must be finite")
    # Preserve the historical producer's arithmetic order exactly: scalar
    # intercept plus CSR product, then the positional CSR product is added.
    z_bag = np.asarray(intercept[0] + base @ coef, dtype=np.float64).reshape(-1)
    z_original = np.asarray(z_bag + np.asarray(position @ gamma, dtype=np.float64).reshape(-1),
                            dtype=np.float64).reshape(-1)
    if not np.isfinite(z_original).all():
        raise ValueError("frozen original POS logit is not finite")
    return z_bag, z_original


def route_candidate(
    matrix: sparse.spmatrix,
    z_original: Any,
    gamma: Any,
    *,
    present_mask: Any,
    no_known_bigram: Any,
    p_pos_original: Any,
    p_a02: Any,
    p_p: Any,
    mix50_original: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the fixed offset, exact missing route, and exact zero-support route."""
    present = np.asarray(present_mask, dtype=bool).reshape(-1)
    no_known = np.asarray(no_known_bigram, dtype=bool).reshape(-1)
    offset = np.asarray(z_original, dtype=np.float64).reshape(-1)
    pos = np.asarray(p_pos_original, dtype=np.float64).reshape(-1)
    a02 = np.asarray(p_a02, dtype=np.float64).reshape(-1)
    pp = np.asarray(p_p, dtype=np.float64).reshape(-1)
    mix = np.asarray(mix50_original, dtype=np.float64).reshape(-1)
    x = sparse.csr_matrix(matrix, dtype=np.float64)
    coef = np.asarray(gamma, dtype=np.float64).reshape(-1)
    n = len(offset)
    if (x.shape != (n, len(coef)) or any(len(a) != n for a in
                                          (present, no_known, pos, a02, pp, mix))):
        raise ValueError("bigram probability policy inputs are not row/feature aligned")
    if (not np.isfinite(offset).all() or not np.isfinite(x.data).all()
            or not np.isfinite(coef).all()):
        raise ValueError("bigram probability policy received non-finite offset/features")
    if any(not np.isfinite(a).all() or np.any((a < 0) | (a > 1))
           for a in (pos, a02, pp, mix)):
        raise ValueError("frozen probability inputs must be finite in [0,1]")
    q = offset_probability(x, offset, coef)
    routed = 0.5 * q + 0.5 * a02
    routed[~present] = pp[~present]
    # Preserve the exact incumbent floating point vector for rows where the
    # new representation has no known feature. The zero-gamma identity is
    # separately audited numerically before fitting.
    fallback = present & no_known
    routed[fallback] = mix[fallback]
    if not np.array_equal(routed[~present], pp[~present]):
        raise AssertionError("missing-stream rows must route exactly to frozen pP")
    if not np.array_equal(routed[fallback], mix[fallback]):
        raise AssertionError("zero-known-bigram rows must route exactly to frozen MIX50")
    if not np.isfinite(routed).all() or np.any((routed < 0) | (routed > 1)):
        raise ValueError("candidate route produced invalid probabilities")
    return q, routed


def zero_gamma_equivalence(
    matrix: sparse.spmatrix,
    z_original: Any,
    *,
    present_mask: Any,
    p_a02: Any,
    p_p: Any,
    mix50_original: Any,
) -> dict[str, Any]:
    """Verify the zero-gamma route reconstructs the incumbent within 1e-12."""
    x = sparse.csr_matrix(matrix, dtype=np.float64)
    offset = np.asarray(z_original, dtype=np.float64).reshape(-1)
    present = np.asarray(present_mask, dtype=bool).reshape(-1)
    a02 = np.asarray(p_a02, dtype=np.float64).reshape(-1)
    pp = np.asarray(p_p, dtype=np.float64).reshape(-1)
    mix = np.asarray(mix50_original, dtype=np.float64).reshape(-1)
    if x.shape[0] != len(offset) or any(len(a) != len(offset) for a in (present, a02, pp, mix)):
        raise ValueError("zero-gamma input lengths do not align")
    gamma0 = np.zeros(x.shape[1], dtype=np.float64)
    q0 = offset_probability(x, offset, gamma0)
    reconstructed = 0.5 * q0 + 0.5 * a02
    reconstructed[~present] = pp[~present]
    if len(reconstructed[present]):
        max_abs = float(np.max(np.abs(reconstructed[present] - mix[present])))
    else:
        max_abs = 0.0
    return {
        "pass": bool(max_abs <= 1e-12 and np.array_equal(reconstructed[~present], pp[~present])),
        "max_abs_present_difference": max_abs,
        "missing_route_bit_exact": bool(np.array_equal(reconstructed[~present], pp[~present])),
        "tolerance": 1e-12,
    }


__all__ = ["reconstruct_original_pos_logit", "route_candidate", "zero_gamma_equivalence"]
