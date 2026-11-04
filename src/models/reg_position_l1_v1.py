"""Pure feature and fixed residual-policy contracts for REG_POSITION_L1_V1."""
from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize


def unigram_bag(vectorizer, documents) -> sparse.csr_matrix:
    """Use an existing fitted TF-IDF state; never learn vocabulary/IDF."""
    positions = sorted(int(index) for term, index in vectorizer.vocabulary_.items() if " " not in term)
    full = vectorizer.transform(documents).tocsr()
    bag = normalize(full[:, positions], norm="l2", axis=1, copy=True).tocsr()
    if not np.isfinite(bag.data).all():
        raise ValueError("Unigram bag must be finite")
    return bag


def head_design(bag, position, baseline) -> sparse.csr_matrix:
    """Assemble BAG or positional arm with identical frozen baseline covariate."""
    b = np.asarray(baseline, dtype=np.float64)
    matrix = sparse.csr_matrix(bag, dtype=np.float64)
    if b.ndim != 1 or matrix.shape[0] != len(b) or not np.isfinite(b).all() or np.any((b < 0) | (b > 100)):
        raise ValueError("R06 baseline must be finite, bounded and row-aligned")
    blocks = [matrix]
    if position is not None:
        p = sparse.csr_matrix(position, dtype=np.float64)
        if p.shape[0] != len(b):
            raise ValueError("Position rows do not match the fixed baseline")
        blocks.append(p)
    blocks.append(sparse.csr_matrix((b/100).reshape(-1, 1)))
    design = sparse.hstack(blocks, format="csr", dtype=np.float64)
    design.sum_duplicates()
    design.sort_indices()
    if not np.isfinite(design.data).all():
        raise ValueError("Head design must be finite")
    return design


def compose_residual_prediction(baseline, head_prediction, missing):
    """Fixed alpha=.25, delta cap5, target bounds; no rank mapping."""
    b = np.asarray(baseline, dtype=np.float64)
    h = np.asarray(head_prediction, dtype=np.float64)
    m = np.asarray(missing)
    if b.ndim != 1 or h.shape != b.shape or m.shape != b.shape or m.dtype.kind != "b":
        raise ValueError("Aligned vector baseline/head and Boolean mask required")
    if not np.isfinite(b).all() or not np.isfinite(h).all() or np.any((b < 0) | (b > 100)):
        raise ValueError("Finite head and bounded R06 baseline required")
    delta = np.clip(.25*h, -5., 5.)
    delta[m] = 0.
    prediction = np.clip(b+delta, 0., 100.)
    prediction[m] = b[m]
    return prediction, delta
