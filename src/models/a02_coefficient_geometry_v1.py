"""Frozen coefficient-informed token geometry for the CGEOM_V1 A02 pilot.

This module performs no estimator or preprocessing fit.  It reuses an exact
same-origin A02 text space and residual bank, with C2_B token coefficients
supplied by the runner after its lineage checks.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Sequence

import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize

from src.models.classification_a02_neighbors import (
    CORRECTION_LIMIT,
    NEIGHBORS_MAX,
    QUERY_BLOCK_ROWS,
    SHRINKAGE,
    A02TextSpace,
    _ids,
    _integer_vector,
    _probabilities,
    _strict_rows,
)


PILOT_ID = "CGEOM_V1"
PERMUTATION_SEED = 42
MASS_TOLERANCE = 1e-12


def _array_sha256(values: Any, dtype: str | None = None) -> str:
    array = np.asarray(values, dtype=dtype)
    digest = hashlib.sha256()
    digest.update(json.dumps({"shape": array.shape, "dtype": str(array.dtype)},
                             sort_keys=True, separators=(",", ":")).encode("ascii"))
    if array.dtype.kind in "OUS":
        for value in array.reshape(-1):
            payload = str(value).encode("utf-8", errors="surrogatepass")
            digest.update(len(payload).to_bytes(8, "little", signed=False))
            digest.update(payload)
    else:
        digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                     allow_nan=False, default=lambda x: x.item() if isinstance(x, np.generic) else str(x))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def coefficient_weights(beta: Any, *, arm: str, seed: int = PERMUTATION_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Return frozen token multipliers and raw coefficients in A02 column order."""
    values = np.asarray(beta, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("beta must be a nonempty finite coefficient vector")
    if arm not in {"aligned", "permuted"}:
        raise ValueError("arm must be 'aligned' or 'permuted'")
    importance = 1.0 + np.abs(values) / (1.0 + np.abs(values))
    multipliers = np.sqrt(importance)
    if arm == "permuted":
        multipliers = np.random.default_rng(int(seed)).permutation(multipliers)
    if not np.isfinite(multipliers).all() or np.any(multipliers < 1.0) or np.any(multipliers >= np.sqrt(2.0)):
        raise RuntimeError("coefficient weights fell outside their frozen positive bounds")
    return multipliers.astype(np.float64, copy=False), values.copy()


def _weighted_l2(matrix: sparse.spmatrix, multipliers: np.ndarray) -> sparse.csr_matrix:
    values = sparse.csr_matrix(matrix, dtype=np.float64, copy=True)
    if values.shape[1] != len(multipliers):
        raise ValueError("token multiplier width does not match the TF-IDF matrix")
    if not np.isfinite(values.data).all():
        raise ValueError("TF-IDF matrix contains non-finite values")
    values = values.multiply(multipliers).tocsr()
    values.sum_duplicates()
    values.sort_indices()
    return sparse.csr_matrix(normalize(values, norm="l2", axis=1, copy=False), dtype=np.float64)


def _top_positive(similarities: np.ndarray, bank_rows: np.ndarray, k: int) -> np.ndarray:
    """Select positive cosine neighbors with source-row stable tie ordering."""
    sims = np.asarray(similarities, dtype=np.float64)
    rows = np.asarray(bank_rows, dtype=np.int64)
    if sims.ndim != 1 or rows.shape != sims.shape or not np.isfinite(sims).all():
        raise ValueError("similarities and bank rows must be aligned finite vectors")
    positive = np.flatnonzero(sims > 0.0)
    take = min(int(k), len(positive))
    if take == 0:
        return np.empty(0, dtype=np.int64)
    if len(positive) > take:
        positive_sims = sims[positive]
        threshold = np.partition(positive_sims, len(positive_sims) - take)[len(positive_sims) - take]
        above = positive[positive_sims > threshold]
        tied = positive[positive_sims == threshold]
        remaining = take - len(above)
        tied_order = np.argsort(rows[tied], kind="stable")
        selected = np.concatenate((above, tied[tied_order[:remaining]]))
    else:
        selected = positive
    order = np.lexsort((rows[selected], -sims[selected]))
    return selected[order].astype(np.int64, copy=False)


@dataclass
class CoefficientGeometryPredictor:
    """One indexed residual-memory predictor for an origin and geometry arm."""

    origin_cycle: int
    arm: str
    permutation_seed: int = PERMUTATION_SEED

    def __post_init__(self) -> None:
        if self.arm not in {"aligned", "permuted"}:
            raise ValueError("arm must be 'aligned' or 'permuted'")
        self.origin_cycle = int(self.origin_cycle)
        self._fitted = False

    def fit(
        self,
        *,
        text_space: A02TextSpace,
        bank_matrix: sparse.spmatrix,
        bank_ids: Sequence[Any],
        bank_row_positions: Any,
        bank_cycles: Any,
        bank_oof_probabilities: Any,
        bank_residuals: Any,
        beta_in_a02_column_order: Any,
    ) -> "CoefficientGeometryPredictor":
        """Build weighted donor geometry; no labels or estimators are fit here."""
        if text_space.origin_cycle != self.origin_cycle:
            raise ValueError("A02 text-space origin differs from CGEOM origin")
        rows = _strict_rows(bank_row_positions, "bank_row_positions")
        # A02 persists typed IDs such as ``str:record-id``.  Keep that exact
        # representation; raw synthetic IDs are normalized by the same helper.
        raw_bank_ids = tuple(bank_ids)
        if raw_bank_ids and all(isinstance(value, str) and value.startswith(("str:", "str_:", "int:"))
                                for value in raw_bank_ids):
            ids = tuple(str(value) for value in raw_bank_ids)
            if len(set(ids)) != len(ids) or any(value.endswith(":None") or value.endswith(":<NA>") for value in ids):
                raise ValueError("bank_ids contains duplicate or null IDs")
        else:
            ids = _ids(raw_bank_ids, "bank_ids")
        n = len(rows)
        cycles = _integer_vector(bank_cycles, "bank_cycles", n)
        probabilities = _probabilities(bank_oof_probabilities, "bank_oof_probabilities", n)
        residuals = np.asarray(bank_residuals, dtype=np.float64)
        if (len(ids) != n or residuals.shape != (n,) or not np.isfinite(residuals).all()
                or np.any(np.abs(residuals) > 1.0)):
            raise ValueError("bank IDs/residuals are invalid or misaligned")
        if np.any(cycles > self.origin_cycle):
            raise ValueError("residual donor bank contains a future cycle")
        matrix = sparse.csr_matrix(bank_matrix, dtype=np.float64, copy=False)
        if matrix.shape != (n, len(text_space.vectorizer.vocabulary_)):
            raise ValueError("bank matrix shape does not match the A02 vocabulary")
        if not np.isfinite(matrix.data).all():
            raise ValueError("bank matrix contains non-finite TF-IDF values")
        multipliers, beta = coefficient_weights(
            beta_in_a02_column_order, arm=self.arm, seed=self.permutation_seed
        )
        if len(multipliers) != matrix.shape[1]:
            raise ValueError("coefficient count does not match the exact A02 token map")
        weighted_bank = _weighted_l2(matrix, multipliers)
        self.text_space = text_space
        self.bank_matrix = weighted_bank
        self.bank_ids = ids
        self.bank_row_positions = rows
        self.bank_cycles = cycles
        self.bank_oof_probabilities = probabilities
        self.bank_residuals = residuals
        self.coefficients = beta
        self.token_multipliers = multipliers
        self._state = {
            "pilot_id": PILOT_ID,
            "origin_cycle": self.origin_cycle,
            "arm": self.arm,
            "permutation_seed": self.permutation_seed if self.arm == "permuted" else None,
            "token_count": int(len(beta)),
            "beta_sha256": _array_sha256(beta, "<f8"),
            "token_multipliers_sha256": _array_sha256(multipliers, "<f8"),
            "bank_rows": int(n),
            "bank_ids_sha256": _array_sha256(ids),
            "bank_row_positions_sha256": _array_sha256(rows, "<i8"),
            "bank_cycles_sha256": _array_sha256(cycles, "<i8"),
            "bank_oof_probability_sha256": _array_sha256(probabilities, "<f8"),
            "bank_residual_sha256": _array_sha256(residuals, "<f8"),
            "weighted_bank_shape": [int(v) for v in weighted_bank.shape],
            "weighted_bank_nnz": int(weighted_bank.nnz),
            "weighted_bank_data_sha256": _array_sha256(weighted_bank.data, "<f8"),
            "neighbors_max": NEIGHBORS_MAX,
            "query_block_rows": QUERY_BLOCK_ROWS,
            "shrinkage": SHRINKAGE,
            "correction_limit": CORRECTION_LIMIT,
            "mass_matching": "sum(new_similarity_squared_weights)=M0_per_query",
        }
        self._state["state_sha256"] = _canonical_sha256(self._state)
        self._fitted = True
        return self

    def predict(
        self,
        *,
        query_texts: Sequence[Any],
        query_record_ids: Sequence[Any],
        query_row_positions: Any,
        query_cycles: Any,
        baseline_probabilities: Any,
        original_a02_probabilities: Any,
        original_a02_neighbor_source_rows: Any,
        original_a02_neighbor_similarities: Any,
        missing_mask: Any,
    ) -> dict[str, Any]:
        """Search the complete bank and emit candidate predictions without labels."""
        if not self._fitted:
            raise RuntimeError("CGEOM predictor must be fit before prediction")
        rows = _strict_rows(query_row_positions, "query_row_positions")
        ids = _ids(query_record_ids, "query_record_ids")
        n = len(rows)
        if len(ids) != n:
            raise ValueError("query IDs are not aligned")
        cycles = _integer_vector(query_cycles, "query_cycles", n)
        if np.any(cycles <= self.origin_cycle):
            raise ValueError("query cycles must be later than the fitted origin")
        if set(ids) & set(self.bank_ids) or np.intersect1d(rows, self.bank_row_positions).size:
            raise ValueError("bank/query overlap is prohibited")
        baseline = _probabilities(baseline_probabilities, "baseline_probabilities", n)
        original = _probabilities(original_a02_probabilities, "original_a02_probabilities", n)
        original_rows = np.asarray(original_a02_neighbor_source_rows)
        original_sims = np.asarray(original_a02_neighbor_similarities, dtype=np.float64)
        if (original_rows.ndim != 2 or original_rows.shape != original_sims.shape
                or original_rows.shape[0] != n or original_sims.shape[1] != NEIGHBORS_MAX
                or original_rows.dtype.kind not in "iu" or not np.isfinite(original_sims).all()
                or np.any(original_sims < 0.0)):
            raise ValueError("cached A02 neighbor source/similarity arrays are invalid")
        original_rows = original_rows.astype(np.int64, copy=False)
        missing = np.asarray(missing_mask, dtype=bool)
        matrix, detected_missing = self.text_space.transform(query_texts)
        if missing.shape != (n,) or not np.array_equal(missing, detected_missing):
            raise ValueError("query missing mask differs from frozen A02 text normalization")
        m0 = np.sum(original_sims * original_sims, axis=1, dtype=np.float64)
        if not np.isfinite(m0).all() or np.any(m0 < 0.0):
            raise ValueError("original A02 correction mass is invalid")
        if np.any(m0[missing] != 0.0):
            raise ValueError("missing-stream rows must have zero original neighbor mass")

        weighted_queries = _weighted_l2(matrix, self.token_multipliers)
        width = NEIGHBORS_MAX
        max_id_chars = max((len(value) for value in self.bank_ids), default=1)
        neighbor_rows = np.full((n, width), -1, dtype=np.int64)
        neighbor_ids = np.full((n, width), "", dtype=f"<U{max_id_chars}")
        neighbor_sims = np.zeros((n, width), dtype=np.float64)
        neighbor_weights = np.zeros((n, width), dtype=np.float64)
        neighbor_counts = np.zeros(n, dtype=np.int16)
        new_mass = np.zeros(n, dtype=np.float64)
        correction_raw = np.zeros(n, dtype=np.float64)

        for start in range(0, n, QUERY_BLOCK_ROWS):
            stop = min(start + QUERY_BLOCK_ROWS, n)
            dense = (weighted_queries[start:stop] @ self.bank_matrix.T).toarray()
            for block_row, query_index in enumerate(range(start, stop)):
                if missing[query_index] or m0[query_index] == 0.0:
                    continue
                sims = dense[block_row]
                chosen = _top_positive(sims, self.bank_row_positions, width)
                if not len(chosen):
                    raise RuntimeError("positive original A02 mass produced no weighted donor support")
                selected_sims = sims[chosen]
                squared = selected_sims * selected_sims
                mass = float(np.sum(squared, dtype=np.float64))
                if not np.isfinite(mass) or mass <= 0.0:
                    raise RuntimeError("selected weighted donor mass is invalid")
                weights = squared * (m0[query_index] / mass)
                weight_mass = float(np.sum(weights, dtype=np.float64))
                tolerance = MASS_TOLERANCE * max(1.0, abs(float(m0[query_index])))
                if abs(weight_mass - m0[query_index]) > tolerance:
                    raise RuntimeError("new donor weights failed to match original A02 mass")
                residual = float(np.dot(weights, self.bank_residuals[chosen]))
                correction_raw[query_index] = residual / (SHRINKAGE + float(m0[query_index]))
                count = len(chosen)
                neighbor_rows[query_index, :count] = self.bank_row_positions[chosen]
                neighbor_ids[query_index, :count] = [self.bank_ids[index] for index in chosen]
                neighbor_sims[query_index, :count] = selected_sims
                neighbor_weights[query_index, :count] = weights
                neighbor_counts[query_index] = count
                new_mass[query_index] = mass

        # Positive column weights preserve the exact support graph.  The cached
        # original mass therefore defines which rows may receive a correction.
        if np.any((m0 > 0.0) & (neighbor_counts == 0) & ~missing):
            raise RuntimeError("a present row with original A02 support lost all weighted neighbors")
        if np.any((m0 == 0.0) & (neighbor_counts > 0)):
            raise RuntimeError("positive-weight geometry changed the zero-overlap support graph")
        correction = np.clip(correction_raw, -CORRECTION_LIMIT, CORRECTION_LIMIT)
        correction[(missing) | (m0 == 0.0)] = 0.0
        prediction = np.clip(baseline + correction, 0.0, 1.0)
        fallback = missing | (m0 == 0.0)
        prediction[fallback] = baseline[fallback]
        if (not np.isfinite(prediction).all() or not np.array_equal(prediction[fallback], baseline[fallback])
                or not np.isfinite(neighbor_weights).all()):
            raise RuntimeError("CGEOM produced invalid predictions or changed fallback")

        overlap_fraction = np.zeros(n, dtype=np.float64)
        for index in np.flatnonzero(~missing & (m0 > 0.0)):
            original_selected = set(original_rows[index][original_rows[index] >= 0].tolist())
            current_selected = set(neighbor_rows[index][neighbor_rows[index] >= 0].tolist())
            overlap_fraction[index] = len(original_selected & current_selected) / max(1, len(original_selected))
        nonzero_weights = neighbor_weights[neighbor_weights > 0.0]
        diagnostics = {
            "query_rows": int(n),
            "missing_rows": int(missing.sum()),
            "original_positive_mass_rows": int(np.sum(m0 > 0.0)),
            "candidate_positive_neighbor_rows": int(np.sum(neighbor_counts > 0)),
            "fallback_rows": int(fallback.sum()),
            "mean_neighbor_count": float(neighbor_counts.mean()) if n else 0.0,
            "mean_original_mass": float(m0.mean()) if n else 0.0,
            "mean_new_cosine_squared_mass": float(new_mass[new_mass > 0.0].mean()) if np.any(new_mass > 0.0) else 0.0,
            "max_abs_mass_error": float(np.max(np.abs(np.sum(neighbor_weights, axis=1) - m0))) if n else 0.0,
            "mean_neighbor_overlap_fraction": float(overlap_fraction.mean()) if n else 0.0,
            "correction_clip_rows": int(np.sum(np.abs(correction_raw) > CORRECTION_LIMIT)),
            "correction_nonzero_rows": int(np.sum(correction != 0.0)),
            "weight_min": float(nonzero_weights.min()) if len(nonzero_weights) else 0.0,
            "weight_median": float(np.median(nonzero_weights)) if len(nonzero_weights) else 0.0,
            "weight_max": float(nonzero_weights.max()) if len(nonzero_weights) else 0.0,
            "m0_sha256": _array_sha256(m0, "<f8"),
            "neighbor_rows_sha256": _array_sha256(neighbor_rows, "<i8"),
            "neighbor_ids_sha256": _array_sha256(neighbor_ids),
            "neighbor_similarity_sha256": _array_sha256(neighbor_sims, "<f8"),
            "neighbor_weights_sha256": _array_sha256(neighbor_weights, "<f8"),
            "correction_sha256": _array_sha256(correction, "<f8"),
            "prediction_sha256": _array_sha256(prediction, "<f8"),
        }
        query_id_values = [item.split(":", 1)[1] for item in ids]
        id_width = max((len(value) for value in query_id_values), default=1)
        return {
            "candidate_probability": prediction,
            "baseline_probability": baseline.copy(),
            "original_a02_probability": original.copy(),
            "m0": m0,
            "neighbor_count": neighbor_counts,
            "neighbor_source_rows": neighbor_rows,
            "neighbor_ids": neighbor_ids,
            "neighbor_similarities": neighbor_sims,
            "neighbor_weights": neighbor_weights,
            "new_cosine_squared_mass": new_mass,
            "correction_raw": correction_raw,
            "correction": correction,
            "missing_mask": missing.copy(),
            "query_rows": rows.copy(),
            "query_cycles": cycles.copy(),
            "query_record_ids": np.asarray(query_id_values, dtype=f"<U{id_width}"),
            "diagnostics": diagnostics,
        }

    def audit_state(self) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("CGEOM predictor is not fitted")
        return dict(self._state)


__all__ = [
    "CoefficientGeometryPredictor",
    "MASS_TOLERANCE",
    "PERMUTATION_SEED",
    "PILOT_ID",
    "coefficient_weights",
]
