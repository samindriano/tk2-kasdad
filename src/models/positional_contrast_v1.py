"""Fixed orthonormal contrasts of the existing normalized third-position block.

This removes equal-across-thirds coefficient directions. It does not remove
all unordered information: the source applies sublinear TF within each third.
No fitting, vocabulary construction, or post-projection normalization occurs.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse


def contrast_block(augmented: sparse.spmatrix, base_width: int) -> sparse.csr_matrix:
    """Return [(first-last)/sqrt(2), (first-2*middle+last)/sqrt(6)]."""
    if isinstance(base_width, bool) or not isinstance(base_width, (int, np.integer)):
        raise ValueError("base_width must be an integer")
    x = sparse.csr_matrix(augmented, dtype=np.float64)
    if base_width < 0 or base_width > x.shape[1]:
        raise ValueError("base_width is outside the augmented matrix")
    if not np.isfinite(x.data).all():
        raise ValueError("Augmented matrix must be finite")
    position_width = x.shape[1] - base_width
    if position_width % 3:
        raise ValueError("Positional width must contain three equal blocks")
    d = position_width // 3
    first = x[:, base_width:base_width+d]
    middle = x[:, base_width+d:base_width+2*d]
    last = x[:, base_width+2*d:]
    z = sparse.hstack(((first-last)/np.sqrt(2.0),
                       (first-2.0*middle+last)/np.sqrt(6.0)), format="csr")
    z.sum_duplicates()
    z.eliminate_zeros()
    z.sort_indices()
    if not np.isfinite(z.data).all():
        raise ValueError("Contrast projection must be finite")
    return z


def expand_contrast_coefficients(theta: np.ndarray) -> np.ndarray:
    """Embed coefficients in original three-block space, preserving L2 norm."""
    value = np.asarray(theta, dtype=np.float64)
    if value.ndim != 1 or len(value) % 2 or not np.isfinite(value).all():
        raise ValueError("Contrast coefficients must be a finite even-width vector")
    a, b = np.split(value, 2)
    return np.concatenate((a/np.sqrt(2.0)+b/np.sqrt(6.0),
                           -2.0*b/np.sqrt(6.0),
                           -a/np.sqrt(2.0)+b/np.sqrt(6.0)))
