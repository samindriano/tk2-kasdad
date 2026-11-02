"""Frozen lexical-neighbor residual correction core for A02.

This module contains no data loading, scoring labels, ledger, or persistence
policy. Callers fit :class:`A02TextSpace` on an explicitly declared historical
prefix, then fit one predictor per observed or shuffled residual assignment.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src.models.classification_models import _TFIDF_PARAMS, _text_values


CARD_ID = "CLASSIFICATION_A02_NEIGHBOR_RESIDUAL"
QUERY_BLOCK_ROWS = 128
NEIGHBORS_MAX = 32
SHRINKAGE = 4.0
CORRECTION_LIMIT = 0.05
SHUFFLE_SEED = 42
TFIDF_PARAMETERS = dict(_TFIDF_PARAMS)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, type):
        return value.__name__
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return repr(value)


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_sha256(values: Any, dtype: str | None = None) -> str:
    array = np.asarray(values, dtype=dtype)
    if array.dtype.kind in "OUS":
        digest = hashlib.sha256()
        for value in array.reshape(-1):
            payload = str(value).encode("utf-8", errors="surrogatepass")
            digest.update(len(payload).to_bytes(8, "little", signed=False))
            digest.update(payload)
        return digest.hexdigest()
    canonical = np.ascontiguousarray(array)
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _documents(values: Sequence[Any] | pd.Series) -> tuple[list[str], np.ndarray]:
    series = values if isinstance(values, pd.Series) else pd.Series(list(values), dtype=object)
    return _text_values(series)


def _strict_rows(values: Any, name: str) -> np.ndarray:
    rows = np.asarray(values)
    if rows.ndim != 1 or rows.dtype.kind not in "iu" or len(rows) == 0:
        raise ValueError(f"{name} must be a nonempty integer row-position vector")
    rows = rows.astype(np.int64, copy=False)
    if np.any(rows < 0) or len(np.unique(rows)) != len(rows) or np.any(np.diff(rows) <= 0):
        raise ValueError(f"{name} must be unique and strictly increasing")
    return rows


def _ids(values: Sequence[Any], name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be an aligned sequence, not a scalar")
    result = tuple(f"{type(value).__name__}:{value}" for value in values)
    if not result or any(value.endswith(":None") or value.endswith(":<NA>") for value in result):
        raise ValueError(f"{name} must contain non-null row IDs")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicate row IDs")
    return result


def _integer_vector(values: Any, name: str, rows: int) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1 or len(result) != rows:
        raise ValueError(f"{name} is not row-aligned")
    numeric = result.astype(np.float64)
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{name} must contain finite integer values")
    return numeric.astype(np.int64)


def _probabilities(values: Any, name: str, rows: int) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if (result.ndim != 1 or len(result) != rows or not np.isfinite(result).all()
            or np.any((result < 0.0) | (result > 1.0))):
        raise ValueError(f"{name} must be aligned finite probabilities in [0,1]")
    return result


def _as_residuals(values: Any, rows: int) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if (result.ndim != 1 or len(result) != rows or not np.isfinite(result).all()
            or np.any(np.abs(result) > 1.0)):
        raise ValueError("OOF residuals must be aligned, finite, and in [-1,1]")
    return result


@dataclass
class A02TextSpace:
    """Original Classification word-TF-IDF fitted on one declared prefix."""

    vectorizer: TfidfVectorizer
    origin_cycle: int
    fit_rows: int
    fit_row_positions_sha256: str
    fit_documents_sha256: str
    vocabulary_sha256: str
    idf_sha256: str

    def transform(self, texts: Sequence[Any] | pd.Series) -> tuple[sparse.csr_matrix, np.ndarray]:
        documents, missing = _documents(texts)
        matrix = self.vectorizer.transform(documents).tocsr()
        if matrix.shape[0] != len(documents):
            raise RuntimeError("A02 text transform row count changed")
        return matrix, missing

    def audit_state(self) -> dict[str, Any]:
        return {
            "card_id": CARD_ID,
            "origin_cycle": int(self.origin_cycle),
            "fit_rows": int(self.fit_rows),
            "fit_row_positions_sha256": self.fit_row_positions_sha256,
            "fit_documents_sha256": self.fit_documents_sha256,
            "tfidf_parameters": _jsonable(TFIDF_PARAMETERS),
            "tfidf_norm": self.vectorizer.norm,
            "tfidf_vocabulary_size": int(len(self.vectorizer.vocabulary_)),
            "vocabulary_sha256": self.vocabulary_sha256,
            "idf_sha256": self.idf_sha256,
            "text_only_cosine": True,
        }


def fit_text_space(
    prefix_texts: Sequence[Any] | pd.Series,
    prefix_row_positions: Any,
    prefix_cycles: Any,
    *,
    origin_cycle: int,
) -> tuple[A02TextSpace, sparse.csr_matrix, np.ndarray]:
    """Fit unchanged Classification TF-IDF using only the declared prefix.

    The returned matrix has sklearn's original L2 TF-IDF normalization. It is
    not densified and no structured feature enters the cosine geometry.
    """
    rows = _strict_rows(prefix_row_positions, "prefix_row_positions")
    docs, missing = _documents(prefix_texts)
    cycles = _integer_vector(prefix_cycles, "prefix_cycles", len(rows))
    if len(docs) != len(rows) or int(origin_cycle) < 0:
        raise ValueError("A02 prefix texts/rows/origin are not aligned")
    if np.any(cycles > int(origin_cycle)):
        raise ValueError("A02 TF-IDF fit prefix contains a future cycle")
    if not np.any(~missing):
        raise ValueError("A02 TF-IDF prefix has no present text documents")

    vectorizer = TfidfVectorizer(**dict(TFIDF_PARAMETERS))
    matrix = vectorizer.fit_transform(docs).tocsr()
    if vectorizer.norm != "l2":
        raise RuntimeError("A02 original TF-IDF recipe no longer uses L2 normalization")
    ordered_vocabulary = sorted(vectorizer.vocabulary_.items(), key=lambda item: item[1])
    space = A02TextSpace(
        vectorizer=vectorizer,
        origin_cycle=int(origin_cycle),
        fit_rows=len(rows),
        fit_row_positions_sha256=_array_sha256(rows, "<i8"),
        # Store only a digest, never the source text, in audit outputs.
        fit_documents_sha256=_json_sha256(docs),
        vocabulary_sha256=_json_sha256([(str(token), int(index))
                                        for token, index in ordered_vocabulary]),
        idf_sha256=_array_sha256(vectorizer.idf_, "<f8"),
    )
    return space, matrix, missing


def permute_residuals_within_cycle_probability_decile(
    residuals: Any,
    cycles: Any,
    oof_probabilities: Any,
    source_row_positions: Any,
    *,
    seed: int = SHUFFLE_SEED,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, int]]]:
    """Permute residual assignments inside frozen cycle × probability bins.

    Returns shuffled residuals, source assignment indices, and nonempty stratum
    sizes. A single RNG is used per call/origin; strata are visited in sorted
    cycle then decile order. Input rows must be in stable source order.
    """
    n = len(np.asarray(residuals))
    values = _as_residuals(residuals, n)
    cycle = _integer_vector(cycles, "cycles", n)
    probability = _probabilities(oof_probabilities, "oof_probabilities", n)
    rows = _strict_rows(source_row_positions, "source_row_positions")
    if len(rows) != n:
        raise ValueError("shuffle source positions are not residual-aligned")
    decile = np.minimum(9, np.floor(10.0 * probability).astype(np.int8))
    assignment = np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    strata: list[dict[str, int]] = []
    for cycle_value in np.unique(cycle):
        for bin_value in range(10):
            indices = np.flatnonzero((cycle == cycle_value) & (decile == bin_value))
            if not len(indices):
                continue
            strata.append({"cycle": int(cycle_value), "probability_decile": int(bin_value),
                           "n": int(len(indices))})
            if len(indices) > 1:
                assignment[indices] = rng.permutation(indices)
    shuffled = values[assignment].copy()
    return shuffled, assignment, strata


class ResidualNeighborPredictor:
    """Fit one observed or within-time/bin-shuffled residual neighbor expert."""

    def __init__(self, *, origin_cycle: int, residual_arm: str = "observed"):
        if residual_arm not in {"observed", "shuffled"}:
            raise ValueError("residual_arm must be 'observed' or 'shuffled'")
        self.origin_cycle = int(origin_cycle)
        self.residual_arm = residual_arm
        self._fitted = False

    def fit(
        self,
        text_space: A02TextSpace,
        bank_texts: Sequence[Any] | pd.Series,
        bank_record_ids: Sequence[Any],
        bank_row_positions: Any,
        bank_cycles: Any,
        bank_oof_probabilities: Any,
        bank_oof_residuals: Any,
    ) -> "ResidualNeighborPredictor":
        self._fitted = False
        if text_space.origin_cycle != self.origin_cycle:
            raise ValueError("A02 text-space and residual-model origins differ")
        rows = _strict_rows(bank_row_positions, "bank_row_positions")
        ids = _ids(bank_record_ids, "bank_record_ids")
        n = len(rows)
        if len(ids) != n:
            raise ValueError("A02 bank IDs are not row-aligned")
        cycles = _integer_vector(bank_cycles, "bank_cycles", n)
        probability = _probabilities(bank_oof_probabilities, "bank_oof_probabilities", n)
        residual = _as_residuals(bank_oof_residuals, n)
        docs, missing = _documents(bank_texts)
        if len(docs) != n or np.any(missing):
            raise ValueError("A02 historical bank must contain aligned present-stream rows only")
        if np.any(cycles > self.origin_cycle):
            raise ValueError("A02 residual bank contains a future cycle")
        matrix, transformed_missing = text_space.transform(docs)
        if np.any(transformed_missing):
            raise ValueError("A02 historical bank unexpectedly contains missing text")

        assignment = np.arange(n, dtype=np.int64)
        strata: list[dict[str, int]] = []
        if self.residual_arm == "shuffled":
            residual, assignment, strata = permute_residuals_within_cycle_probability_decile(
                residual, cycles, probability, rows, seed=SHUFFLE_SEED
            )
        else:
            decile = np.minimum(9, np.floor(10.0 * probability).astype(np.int8))
            for cycle_value in np.unique(cycles):
                for bin_value in range(10):
                    count = int(np.sum((cycles == cycle_value) & (decile == bin_value)))
                    if count:
                        strata.append({"cycle": int(cycle_value),
                                       "probability_decile": int(bin_value), "n": count})

        self.text_space = text_space
        self.bank_matrix = matrix.tocsr()
        self.bank_ids = ids
        self.bank_row_positions = rows
        self.bank_cycles = cycles
        self.bank_oof_probabilities = probability
        self.bank_residuals = residual
        self.residual_assignment = assignment
        self.strata = strata
        self._state = {
            "card_id": CARD_ID,
            "origin_cycle": self.origin_cycle,
            "residual_arm": self.residual_arm,
            "bank_rows": n,
            "bank_cycle_min": int(cycles.min()),
            "bank_cycle_max": int(cycles.max()),
            "bank_ids_sha256": _array_sha256(ids),
            "bank_row_positions_sha256": _array_sha256(rows, "<i8"),
            "bank_cycles_sha256": _array_sha256(cycles, "<i8"),
            "bank_oof_probability_sha256": _array_sha256(probability, "<f8"),
            "bank_original_residual_sha256": _array_sha256(_as_residuals(bank_oof_residuals, n), "<f8"),
            "assigned_residual_sha256": _array_sha256(residual, "<f8"),
            "residual_assignment_sha256": _array_sha256(assignment, "<i8"),
            "shuffle_seed": SHUFFLE_SEED if self.residual_arm == "shuffled" else None,
            "shuffle_strata": list(strata),
            "bank_matrix_shape": [int(v) for v in self.bank_matrix.shape],
            "bank_matrix_nnz": int(self.bank_matrix.nnz),
            "bank_matrix_data_sha256": _array_sha256(self.bank_matrix.data, "<f8"),
            "query_block_rows": QUERY_BLOCK_ROWS,
            "neighbors_max": NEIGHBORS_MAX,
            "shrinkage": SHRINKAGE,
            "correction_limit": CORRECTION_LIMIT,
        }
        self._state["state_sha256"] = _json_sha256(self._state)
        self._fitted = True
        return self

    def predict(
        self,
        query_texts: Sequence[Any] | pd.Series,
        query_record_ids: Sequence[Any],
        query_row_positions: Any,
        query_cycles: Any,
        baseline_probabilities: Any,
        *,
        missing_mask: Any | None = None,
    ) -> dict[str, Any]:
        """Predict from query features only; query labels are intentionally absent."""
        if not self._fitted:
            raise RuntimeError("A02 residual-neighbor predictor must be fit first")
        rows = _strict_rows(query_row_positions, "query_row_positions")
        ids = _ids(query_record_ids, "query_record_ids")
        n = len(rows)
        if len(ids) != n:
            raise ValueError("A02 query IDs are not row-aligned")
        cycles = _integer_vector(query_cycles, "query_cycles", n)
        baseline = _probabilities(baseline_probabilities, "baseline_probabilities", n)
        docs, detected_missing = _documents(query_texts)
        if len(docs) != n:
            raise ValueError("A02 query texts are not row-aligned")
        if np.any(cycles <= self.origin_cycle):
            raise ValueError("A02 query cycles must be strictly later than the fitted origin")
        if set(ids) & set(self.bank_ids):
            raise ValueError("A02 bank/query IDs overlap; self-neighbors are prohibited")
        if np.intersect1d(rows, self.bank_row_positions).size:
            raise ValueError("A02 bank/query source positions overlap; self-neighbors are prohibited")
        if missing_mask is None:
            missing = detected_missing
        else:
            missing = np.asarray(missing_mask, dtype=bool)
            if missing.shape != (n,) or not np.array_equal(missing, detected_missing):
                raise ValueError("A02 missing mask must match the frozen text normalization rule")

        query_matrix, _ = self.text_space.transform(docs)
        width = min(NEIGHBORS_MAX, len(self.bank_ids))
        neighbor_positions = np.full((n, width), -1, dtype=np.int64)
        max_id_chars = max((len(value) for value in self.bank_ids), default=1)
        neighbor_ids = np.full((n, width), "", dtype=f"<U{max_id_chars}")
        neighbor_similarities = np.zeros((n, width), dtype=np.float64)
        neighbor_counts = np.zeros(n, dtype=np.int16)
        correction_raw = np.zeros(n, dtype=np.float64)

        for start in range(0, n, QUERY_BLOCK_ROWS):
            stop = min(start + QUERY_BLOCK_ROWS, n)
            similarity_block = (query_matrix[start:stop] @ self.bank_matrix.T).toarray()
            for block_row, query_index in enumerate(range(start, stop)):
                if missing[query_index]:
                    continue
                similarities = similarity_block[block_row]
                positive = np.flatnonzero(similarities > 0.0)
                if not len(positive):
                    continue
                take = min(width, len(positive))
                if len(positive) > take:
                    positive_sims = similarities[positive]
                    threshold = np.partition(positive_sims, len(positive_sims) - take)[
                        len(positive_sims) - take
                    ]
                    strictly_above = positive[positive_sims > threshold]
                    ties = positive[positive_sims == threshold]
                    remaining = take - len(strictly_above)
                    tie_order = np.argsort(self.bank_row_positions[ties], kind="stable")
                    chosen = np.concatenate((strictly_above, ties[tie_order[:remaining]]))
                else:
                    chosen = positive
                order = np.lexsort((self.bank_row_positions[chosen], -similarities[chosen]))
                chosen = chosen[order]
                selected_s = similarities[chosen]
                weights = selected_s * selected_s
                numerator = float(np.dot(weights, self.bank_residuals[chosen]))
                denominator = SHRINKAGE + float(weights.sum())
                correction_raw[query_index] = numerator / denominator
                neighbor_counts[query_index] = len(chosen)
                neighbor_positions[query_index, :len(chosen)] = self.bank_row_positions[chosen]
                neighbor_ids[query_index, :len(chosen)] = [self.bank_ids[index] for index in chosen]
                neighbor_similarities[query_index, :len(chosen)] = selected_s

        correction = np.clip(correction_raw, -CORRECTION_LIMIT, CORRECTION_LIMIT)
        correction[missing] = 0.0
        candidate = np.clip(baseline + correction, 0.0, 1.0)
        candidate[missing] = baseline[missing]
        if (not np.isfinite(candidate).all() or not np.isfinite(correction).all()
                or np.any(correction[missing] != 0.0)
                or not np.array_equal(candidate[missing], baseline[missing])):
            raise RuntimeError("A02 returned invalid or non-routed predictions")
        return {
            "baseline_probability": baseline.copy(),
            "candidate_probability": candidate,
            "correction": correction,
            "correction_unclipped": correction_raw,
            "neighbor_count": neighbor_counts,
            "neighbor_source_rows": neighbor_positions,
            "neighbor_ids": neighbor_ids,
            "neighbor_similarities": neighbor_similarities,
            "diagnostics": {
                "query_rows": n,
                "missing_stream_rows": int(missing.sum()),
                "zero_vector_or_no_positive_similarity_rows": int(np.sum((neighbor_counts == 0) & ~missing)),
                "positive_neighbor_rows": int(np.sum(neighbor_counts > 0)),
                "mean_effective_neighbors": float(neighbor_counts.mean()) if n else 0.0,
                "correction_nonzero_rows": int(np.sum(correction != 0.0)),
                "correction_clip_rows": int(np.sum(np.abs(correction_raw) > CORRECTION_LIMIT)),
                "neighbor_ids_sha256": _array_sha256(neighbor_ids),
                "neighbor_source_rows_sha256": _array_sha256(neighbor_positions, "<i8"),
                "neighbor_similarities_sha256": _array_sha256(neighbor_similarities, "<f8"),
                "correction_sha256": _array_sha256(correction, "<f8"),
                "candidate_probability_sha256": _array_sha256(candidate, "<f8"),
            },
        }

    def audit_state(self) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("A02 residual-neighbor predictor must be fit first")
        return dict(self._state)


__all__ = [
    "A02TextSpace", "CARD_ID", "CORRECTION_LIMIT", "NEIGHBORS_MAX", "QUERY_BLOCK_ROWS",
    "ResidualNeighborPredictor", "SHRINKAGE", "SHUFFLE_SEED", "TFIDF_PARAMETERS",
    "fit_text_space", "permute_residuals_within_cycle_probability_decile",
]
