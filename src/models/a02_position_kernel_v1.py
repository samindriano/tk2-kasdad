"""Position-aware donor geometry for the frozen A02 residual-memory recipe.

Only sparse text matrices and already-fitted A02 residual state are accepted.
This module fits no vectorizer, estimator, threshold, or calibration component.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import sparse
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, log_loss, precision_score,
                             recall_score, roc_auc_score)

from src.models.classification_a02_neighbors import (
    CORRECTION_LIMIT,
    NEIGHBORS_MAX,
    QUERY_BLOCK_ROWS,
    SHRINKAGE,
    A02TextSpace,
)


PILOT_ID = "CLS_A02_POSITION_KERNEL_V1"
POSITION_FLOOR = 0.5
MASS_TOLERANCE = 1e-12
SHUFFLE_SEED = 42
MISSING_THRESHOLD = 0.53230976129477


def _top_positive(similarities: Any, bank_rows: Any, k: int = NEIGHBORS_MAX) -> np.ndarray:
    """Top-k positive similarities with source-row-stable ties."""
    sims = np.asarray(similarities, dtype=np.float64)
    rows = np.asarray(bank_rows, dtype=np.int64)
    if sims.ndim != 1 or rows.shape != sims.shape or not np.isfinite(sims).all():
        raise ValueError("similarities and source rows must be aligned finite vectors")
    positive = np.flatnonzero(sims > 0.0)
    take = min(int(k), len(positive))
    if take == 0:
        return np.empty(0, dtype=np.int64)
    if len(positive) > take:
        values = sims[positive]
        threshold = np.partition(values, len(values) - take)[len(values) - take]
        above = positive[values > threshold]
        tied = positive[values == threshold]
        remaining = take - len(above)
        tied = tied[np.argsort(rows[tied], kind="stable")]
        chosen = np.concatenate((above, tied[:remaining]))
    else:
        chosen = positive
    return chosen[np.lexsort((rows[chosen], -sims[chosen]))].astype(np.int64, copy=False)


def load_npz_allowlisted(
    path: str | Path,
    *,
    allowed_members: Sequence[str],
    known_unread_members: Sequence[str] = ("labels", "artifact_signal", "y_true", "target"),
) -> dict[str, np.ndarray]:
    """Load only named non-target arrays; target members are never indexed.

    NPZ member names are safe to enumerate. Values are retrieved exclusively
    through the explicit allowlist, so a label/object member may remain closed.
    """
    allowed = tuple(str(name) for name in allowed_members)
    if len(set(allowed)) != len(allowed) or not allowed:
        raise ValueError("NPZ allowlist must contain unique nonempty member names")
    unread = set(str(name) for name in known_unread_members)
    if set(allowed) & unread:
        raise ValueError("target-bearing members are forbidden in an NPZ allowlist")
    with np.load(Path(path), allow_pickle=False) as archive:
        available = set(archive.files)
        missing = set(allowed) - available
        unknown = available - set(allowed) - unread
        if missing:
            raise ValueError(f"NPZ is missing allowed members: {sorted(missing)}")
        if unknown:
            raise ValueError(f"NPZ contains unclassified members: {sorted(unknown)}")
        return {name: archive[name].copy() for name in allowed}


def assemble_full_champion(p_pos: Any, p_a02: Any, missing_mask: Any) -> tuple[np.ndarray, np.ndarray]:
    """Apply the frozen uniform POS/A02 blend and its operational thresholds."""
    pos = np.asarray(p_pos, dtype=np.float64)
    a02 = np.asarray(p_a02, dtype=np.float64)
    missing = np.asarray(missing_mask, dtype=bool)
    if pos.ndim != 1 or a02.shape != pos.shape or missing.shape != pos.shape:
        raise ValueError("POS/A02 probabilities and missing mask must be aligned vectors")
    if not np.isfinite(pos).all() or not np.isfinite(a02).all():
        raise ValueError("POS/A02 probabilities must be finite")
    if np.any((pos < 0) | (pos > 1)) or np.any((a02 < 0) | (a02 > 1)):
        raise ValueError("POS/A02 probabilities must be in [0,1]")
    score = a02.copy()
    present = ~missing
    score[present] = 0.5 * pos[present] + 0.5 * a02[present]
    predicted = np.zeros(len(score), dtype=np.int8)
    predicted[present] = (score[present] >= 0.50).astype(np.int8)
    predicted[missing] = (a02[missing] >= MISSING_THRESHOLD).astype(np.int8)
    return score, predicted


def classification_metrics(y_true: Any, probability: Any, predicted: Any) -> dict[str, Any]:
    """Frozen binary metrics for the authorized temporal score views."""
    y = np.asarray(y_true, dtype=np.int8)
    p = np.asarray(probability, dtype=np.float64)
    pred = np.asarray(predicted, dtype=np.int8)
    if y.ndim != 1 or p.shape != y.shape or pred.shape != y.shape or not len(y):
        raise ValueError("targets, probabilities and predicted classes must be aligned nonempty vectors")
    if not np.isin(y, [0, 1]).all() or not np.isin(pred, [0, 1]).all():
        raise ValueError("binary labels must be integer 0/1")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must be finite and in [0,1]")
    if len(np.unique(y)) != 2:
        auc = float("nan")
        average_precision = float("nan")
    else:
        auc = float(roc_auc_score(y, p))
        average_precision = float(average_precision_score(y, p))
    cm = confusion_matrix(y, pred, labels=[0, 1])
    p_logloss = np.clip(p, 1e-15, 1.0 - 1e-15)
    return {
        "n": int(len(y)),
        "positive_count": int(y.sum()),
        "negative_count": int(len(y) - int(y.sum())),
        "macro_f1": float(f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0)),
        "positive_precision": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "positive_recall": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "roc_auc": auc,
        "average_precision": average_precision,
        "log_loss_eps_1e_minus_15": float(log_loss(y, p_logloss, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, p)),
        "confusion_matrix_labels_0_1": cm.astype(int).tolist(),
    }


def construct_position_kernel(
    *,
    text_space: A02TextSpace,
    query_bag: sparse.spmatrix,
    bank_bag: sparse.spmatrix,
    query_position: sparse.spmatrix,
    bank_position: sparse.spmatrix,
    bank_ids: Sequence[str],
    bank_row_positions: Any,
    bank_cycles: Any,
    bank_residuals: Any,
    query_ids: Sequence[str],
    query_row_positions: Any,
    query_cycles: Any,
    baseline_probabilities: Any,
    original_a02_probabilities: Any,
    original_a02_neighbor_rows: Any,
    original_a02_neighbor_similarities: Any,
    missing_mask: Any,
    arm: str,
    block_rows: int = QUERY_BLOCK_ROWS,
) -> dict[str, np.ndarray | dict[str, Any]]:
    """Construct one frozen REAL/SHUFFLED residual-memory prediction bundle.

    `query_position` and `bank_position` must both be built by the existing
    fit-free `build_position_block` transform using `text_space.vectorizer`.
    Retrieval is blocked over the complete original bank; no query-by-bank
    dense matrix is retained.
    """
    if arm not in {"REAL", "SHUFFLED"}:
        raise ValueError("arm must be REAL or SHUFFLED")
    origin = int(text_space.origin_cycle)
    bank_ids = tuple(str(x) for x in bank_ids)
    query_ids = tuple(str(x) for x in query_ids)
    bank_rows = np.asarray(bank_row_positions, dtype=np.int64)
    query_rows = np.asarray(query_row_positions, dtype=np.int64)
    bank_cycle = np.asarray(bank_cycles, dtype=np.int64)
    query_cycle = np.asarray(query_cycles, dtype=np.int64)
    residual = np.asarray(bank_residuals, dtype=np.float64)
    baseline = np.asarray(baseline_probabilities, dtype=np.float64)
    original = np.asarray(original_a02_probabilities, dtype=np.float64)
    original_rows = np.asarray(original_a02_neighbor_rows, dtype=np.int64)
    original_sims = np.asarray(original_a02_neighbor_similarities, dtype=np.float64)
    missing = np.asarray(missing_mask, dtype=bool)

    n_bank = len(bank_rows)
    n_query = len(query_rows)
    if n_bank == 0 or n_query == 0 or len(bank_ids) != n_bank or len(query_ids) != n_query:
        raise ValueError("bank/query IDs and positions must be nonempty and aligned")
    if len(set(bank_ids)) != n_bank or len(set(query_ids)) != n_query:
        raise ValueError("bank/query IDs must be unique")
    if set(bank_ids) & set(query_ids) or np.intersect1d(bank_rows, query_rows).size:
        raise ValueError("bank/query self-neighbors are prohibited")
    if (np.any(bank_cycle > origin) or np.any(query_cycle <= origin)
            or bank_cycle.shape != (n_bank,) or query_cycle.shape != (n_query,)):
        raise ValueError("donor/query cycle boundary is invalid")
    if residual.shape != (n_bank,) or not np.isfinite(residual).all() or np.any(np.abs(residual) > 1.0):
        raise ValueError("bank residuals must be finite, aligned, and in [-1,1]")
    if (baseline.shape != (n_query,) or original.shape != (n_query,)
            or not np.isfinite(baseline).all() or not np.isfinite(original).all()
            or np.any((baseline < 0) | (baseline > 1)) or np.any((original < 0) | (original > 1))):
        raise ValueError("query probabilities must be aligned, finite, and in [0,1]")
    if (original_rows.shape != (n_query, NEIGHBORS_MAX)
            or original_sims.shape != original_rows.shape
            or not np.isfinite(original_sims).all() or np.any(original_sims < 0.0)):
        raise ValueError("cached original A02 top-k geometry is invalid")
    if missing.shape != (n_query,):
        raise ValueError("missing mask is not query-aligned")

    q_bag = sparse.csr_matrix(query_bag, dtype=np.float64, copy=False)
    b_bag = sparse.csr_matrix(bank_bag, dtype=np.float64, copy=False)
    q_pos = sparse.csr_matrix(query_position, dtype=np.float64, copy=False)
    b_pos = sparse.csr_matrix(bank_position, dtype=np.float64, copy=False)
    vocabulary_width = len(text_space.vectorizer.vocabulary_)
    if q_bag.shape != (n_query, vocabulary_width) or b_bag.shape != (n_bank, vocabulary_width):
        raise ValueError("bag matrices do not match the original A02 vocabulary")
    if (q_pos.shape[0] != n_query or b_pos.shape[0] != n_bank
            or q_pos.shape[1] != b_pos.shape[1] or not q_pos.shape[1]):
        raise ValueError("real/shuffled position matrices are not aligned")
    if min(int(block_rows), n_query, n_bank) < 1:
        raise ValueError("block_rows must be positive")
    for matrix in (q_bag, b_bag, q_pos, b_pos):
        if not np.isfinite(matrix.data).all():
            raise ValueError("sparse text matrices contain non-finite values")

    m0 = np.sum(original_sims * original_sims, axis=1, dtype=np.float64)
    if not np.isfinite(m0).all() or np.any(m0 < 0.0) or np.any(m0[missing] != 0.0):
        raise ValueError("original A02 mass is invalid or nonzero on missing rows")
    if np.any((m0 == 0.0) & (np.abs(original - baseline) > 1e-12)):
        raise ValueError("zero-support original A02 rows must retain Sparse25 exactly")
    if np.any(missing & (np.abs(original - baseline) > 1e-12)):
        raise ValueError("missing-stream A02 fallback differs from Sparse25")

    k = NEIGHBORS_MAX
    selected_rows = np.full((n_query, k), -1, dtype=np.int64)
    selected_ids = np.full((n_query, k), "", dtype=f"<U{max(1, max(map(len, bank_ids)))}")
    selected_kernel = np.zeros((n_query, k), dtype=np.float64)
    weights = np.zeros((n_query, k), dtype=np.float64)
    counts = np.zeros(n_query, dtype=np.int16)
    correction_raw = np.zeros(n_query, dtype=np.float64)
    positive_kernel_mass = np.zeros(n_query, dtype=np.float64)

    for start in range(0, n_query, int(block_rows)):
        stop = min(start + int(block_rows), n_query)
        bag_dense = (q_bag[start:stop] @ b_bag.T).toarray()
        pos_dense = (q_pos[start:stop] @ b_pos.T).toarray()
        kernel_dense = bag_dense * (POSITION_FLOOR + (1.0 - POSITION_FLOOR) * pos_dense)
        if not np.isfinite(kernel_dense).all():
            raise RuntimeError("position kernel produced non-finite similarities")
        for block_row, qi in enumerate(range(start, stop)):
            if missing[qi] or m0[qi] == 0.0:
                continue
            chosen = _top_positive(kernel_dense[block_row], bank_rows, k)
            if not len(chosen):
                raise RuntimeError("positive original A02 mass has no positive position-kernel support")
            chosen_sim = kernel_dense[block_row, chosen]
            squared = chosen_sim * chosen_sim
            mass = float(np.sum(squared, dtype=np.float64))
            if not np.isfinite(mass) or mass <= 0.0:
                raise RuntimeError("selected position-kernel mass is invalid")
            row_weights = squared * (float(m0[qi]) / mass)
            mass_error = float(np.sum(row_weights, dtype=np.float64) - m0[qi])
            if abs(mass_error) > MASS_TOLERANCE * max(1.0, abs(float(m0[qi]))):
                raise RuntimeError("position-kernel weights do not preserve A02 correction mass")
            correction_raw[qi] = float(np.dot(row_weights, residual[chosen])) / (SHRINKAGE + float(m0[qi]))
            positive_kernel_mass[qi] = mass
            counts[qi] = len(chosen)
            selected_rows[qi, :len(chosen)] = bank_rows[chosen]
            selected_ids[qi, :len(chosen)] = np.asarray(bank_ids, dtype=selected_ids.dtype)[chosen]
            selected_kernel[qi, :len(chosen)] = chosen_sim
            weights[qi, :len(chosen)] = row_weights

    if np.any((m0 > 0.0) & (counts == 0) & ~missing):
        raise RuntimeError("supported present A02 rows lost all position-kernel neighbors")
    if np.any((m0 == 0.0) & (counts > 0)):
        raise RuntimeError("zero-support A02 rows unexpectedly received kernel neighbors")
    correction = np.clip(correction_raw, -CORRECTION_LIMIT, CORRECTION_LIMIT)
    correction[(missing) | (m0 == 0.0)] = 0.0
    candidate_a02 = np.clip(baseline + correction, 0.0, 1.0)
    fallback = missing | (m0 == 0.0)
    candidate_a02[fallback] = baseline[fallback]
    if np.any(missing & (candidate_a02 != original)):
        raise RuntimeError("position kernel changed missing-stream A02 probabilities")

    overlap = np.zeros(n_query, dtype=np.float64)
    for qi in np.flatnonzero(counts > 0):
        old = set(original_rows[qi][original_rows[qi] >= 0].tolist())
        new = set(selected_rows[qi][selected_rows[qi] >= 0].tolist())
        overlap[qi] = len(old & new) / max(1, len(old))
    mass_error = np.sum(weights, axis=1, dtype=np.float64) - m0
    return {
        "m0": m0,
        "neighbor_count": counts,
        "neighbor_source_rows": selected_rows,
        "neighbor_ids": selected_ids,
        "neighbor_kernel_similarity": selected_kernel,
        "neighbor_weights": weights,
        "position_kernel_positive_mass": positive_kernel_mass,
        "correction_raw": correction_raw,
        "correction": correction,
        "candidate_a02_probability": candidate_a02,
        "missing_mask": missing.copy(),
        "mass_error": mass_error,
        "neighbor_overlap_fraction": overlap,
        "diagnostics": {
            "arm": arm,
            "origin_cycle": origin,
            "query_rows": int(n_query),
            "bank_rows": int(n_bank),
            "missing_rows": int(missing.sum()),
            "positive_original_mass_rows": int(np.sum(m0 > 0)),
            "position_kernel_neighbor_rows": int(np.sum(counts > 0)),
            "correction_clip_rows": int(np.sum(np.abs(correction_raw) > CORRECTION_LIMIT)),
            "correction_clip_rate_present_supported": float(np.sum((np.abs(correction_raw) > CORRECTION_LIMIT) & ~missing & (m0 > 0)) / max(1, np.sum(~missing & (m0 > 0)))),
            "max_abs_mass_error": float(np.max(np.abs(mass_error))) if n_query else 0.0,
            "mean_neighbor_overlap_fraction_supported": float(np.mean(overlap[counts > 0])) if np.any(counts > 0) else 0.0,
            "mean_neighbor_count": float(counts.mean()) if n_query else 0.0,
        },
    }


__all__ = [
    "MASS_TOLERANCE", "MISSING_THRESHOLD", "PILOT_ID", "POSITION_FLOOR",
    "SHUFFLE_SEED", "_top_positive", "assemble_full_champion", "classification_metrics",
    "construct_position_kernel",
]
