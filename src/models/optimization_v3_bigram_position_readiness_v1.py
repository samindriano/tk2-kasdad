"""Target-free support profiling for positional known word bigrams.

The builder consumes only document strings and a previously fitted
TfidfVectorizer. It never accepts identifiers or targets and never fits a
vectorizer or estimator.
"""
from __future__ import annotations

from collections import Counter
import math
from typing import Any, Sequence

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src.models.positional_text_features import deterministic_order_shuffle


SEED = 42
POSITION_BINS = 3


def _known_bigram_columns(vectorizer: TfidfVectorizer) -> tuple[dict[str, int], np.ndarray]:
    """Return compact columns in the frozen vectorizer's original column order."""
    vocabulary = getattr(vectorizer, "vocabulary_", None)
    if not isinstance(vocabulary, dict):
        raise ValueError("A fitted frozen vectorizer with vocabulary_ is required")
    idf = np.asarray(vectorizer.idf_, dtype=np.float64)
    entries = sorted(
        ((term, int(index)) for term, index in vocabulary.items()
         if isinstance(term, str) and term.count(" ") == 1 and all(term.split(" "))),
        key=lambda item: item[1],
    )
    if any(index < 0 or index >= len(idf) for _, index in entries):
        raise ValueError("Frozen vectorizer vocabulary and IDF dimensions disagree")
    if len({index for _, index in entries}) != len(entries):
        raise ValueError("Frozen vectorizer column indices are not unique")
    term_to_column = {term: compact for compact, (term, _) in enumerate(entries)}
    original_indices = np.asarray([index for _, index in entries], dtype=np.int64)
    return term_to_column, idf[original_indices]


def _normalized_tokens(document: Any, preprocessor: Any, tokenizer: Any) -> list[str]:
    if document is None:
        return []
    if not isinstance(document, str):
        if isinstance(document, float) and math.isnan(document):
            return []
        document = str(document)
    return list(tokenizer(preprocessor(document)))


def _bin_for_pair(position: int, pair_count: int) -> int:
    if pair_count <= 0 or position < 0 or position >= pair_count:
        raise ValueError("Pair position is outside a non-empty pair sequence")
    return (POSITION_BINS * position) // pair_count


def _descriptor_counts(
    pair_sequence: Sequence[str], term_to_column: dict[str, int],
) -> tuple[Counter[tuple[int, int]], Counter[int], Counter[str]]:
    position_counts: Counter[tuple[int, int]] = Counter()
    bag_counts: Counter[int] = Counter()
    known_terms: Counter[str] = Counter()
    count = len(pair_sequence)
    for position, term in enumerate(pair_sequence):
        column = term_to_column.get(term)
        if column is None:
            continue
        segment = _bin_for_pair(position, count)
        position_counts[(segment, column)] += 1
        bag_counts[column] += 1
        known_terms[term] += 1
    return position_counts, bag_counts, known_terms


def build_bigram_blocks(
    documents: Sequence[Any], vectorizer: TfidfVectorizer, *, seed: int = SEED,
) -> tuple[dict[str, sparse.csr_matrix], dict[str, Any]]:
    """Build bag, real-position, and pair-sequence-shuffled sparse descriptors.

    The positional assignment is floor(3*j/n_pair), using zero-based pair
    occurrence index j and the complete pair sequence, including OOV terms.
    Term counts receive (1 + log(count)) * the frozen bigram IDF. Each whole
    block is L2-normalized once per row. The shuffle permutes pair strings only.
    """
    term_to_column, bigram_idf = _known_bigram_columns(vectorizer)
    preprocessor = vectorizer.build_preprocessor()
    tokenizer = vectorizer.build_tokenizer()
    width = len(term_to_column)
    n_rows = len(documents)

    bag_rows: list[int] = []
    bag_columns: list[int] = []
    bag_values: list[float] = []
    real_rows: list[int] = []
    real_columns: list[int] = []
    real_values: list[float] = []
    shuffled_rows: list[int] = []
    shuffled_columns: list[int] = []
    shuffled_values: list[float] = []

    token_counts = np.zeros(n_rows, dtype=np.int64)
    pair_counts = np.zeros(n_rows, dtype=np.int64)
    known_occurrences = np.zeros(n_rows, dtype=np.int64)
    known_types = np.zeros(n_rows, dtype=np.int64)
    known_column_indices: list[list[int]] = [[] for _ in range(n_rows)]
    changed_descriptors = np.zeros(n_rows, dtype=bool)
    bag_preserved = np.ones(n_rows, dtype=bool)
    segment_sizes_preserved = np.ones(n_rows, dtype=bool)

    for row, document in enumerate(documents):
        tokens = _normalized_tokens(document, preprocessor, tokenizer)
        pair_sequence = [f"{left} {right}" for left, right in zip(tokens, tokens[1:])]
        token_counts[row] = len(tokens)
        pair_counts[row] = len(pair_sequence)

        real_position, bag, term_counts = _descriptor_counts(pair_sequence, term_to_column)
        known_occurrences[row] = int(sum(bag.values()))
        known_types[row] = len(term_counts)
        known_column_indices[row] = sorted(term_to_column[term] for term in term_counts)

        if pair_sequence:
            shuffled_result = deterministic_order_shuffle(list(pair_sequence), seed=seed)
            shuffled_pairs = list(shuffled_result["tokens"])
            if len(shuffled_pairs) != len(pair_sequence):
                raise AssertionError("Pair shuffle changed sequence length")
            bag_preserved[row] = Counter(shuffled_pairs) == Counter(pair_sequence)
            original_sizes = np.bincount(
                [_bin_for_pair(j, len(pair_sequence)) for j in range(len(pair_sequence))],
                minlength=POSITION_BINS,
            )
            shuffled_sizes = np.bincount(
                [_bin_for_pair(j, len(shuffled_pairs)) for j in range(len(shuffled_pairs))],
                minlength=POSITION_BINS,
            )
            segment_sizes_preserved[row] = np.array_equal(original_sizes, shuffled_sizes)
            shuffled_position, shuffled_bag, _ = _descriptor_counts(shuffled_pairs, term_to_column)
            if bag != shuffled_bag:
                raise AssertionError("Pair shuffle changed known bigram bag counts")
            changed_descriptors[row] = real_position != shuffled_position
        else:
            shuffled_position = Counter()

        for column, count in sorted(bag.items()):
            bag_rows.append(row)
            bag_columns.append(column)
            bag_values.append((1.0 + math.log(count)) * float(bigram_idf[column]))
        for (segment, column), count in sorted(real_position.items()):
            real_rows.append(row)
            real_columns.append(segment * width + column)
            real_values.append((1.0 + math.log(count)) * float(bigram_idf[column]))
        for (segment, column), count in sorted(shuffled_position.items()):
            shuffled_rows.append(row)
            shuffled_columns.append(segment * width + column)
            shuffled_values.append((1.0 + math.log(count)) * float(bigram_idf[column]))

    bag_matrix = sparse.csr_matrix(
        (np.asarray(bag_values, dtype=np.float64),
         (np.asarray(bag_rows, dtype=np.int64), np.asarray(bag_columns, dtype=np.int64))),
        shape=(n_rows, width), dtype=np.float64,
    )
    real_matrix = sparse.csr_matrix(
        (np.asarray(real_values, dtype=np.float64),
         (np.asarray(real_rows, dtype=np.int64), np.asarray(real_columns, dtype=np.int64))),
        shape=(n_rows, POSITION_BINS * width), dtype=np.float64,
    )
    shuffled_matrix = sparse.csr_matrix(
        (np.asarray(shuffled_values, dtype=np.float64),
         (np.asarray(shuffled_rows, dtype=np.int64), np.asarray(shuffled_columns, dtype=np.int64))),
        shape=(n_rows, POSITION_BINS * width), dtype=np.float64,
    )
    bag_matrix = normalize(bag_matrix, norm="l2", axis=1, copy=False).tocsr()
    real_matrix = normalize(real_matrix, norm="l2", axis=1, copy=False).tocsr()
    shuffled_matrix = normalize(shuffled_matrix, norm="l2", axis=1, copy=False).tocsr()

    if not bag_preserved.all() or not segment_sizes_preserved.all():
        raise AssertionError("Shuffled pair sequence did not preserve bag or bin sizes")
    for matrix in (bag_matrix, real_matrix, shuffled_matrix):
        if matrix.data.size and not np.isfinite(matrix.data).all():
            raise ValueError("Feature block contains non-finite values")

    metadata = {
        "rows": int(n_rows),
        "bigram_vocabulary_size": int(width),
        "position_bins": POSITION_BINS,
        "seed": int(seed),
        "token_count": token_counts,
        "pair_count": pair_counts,
        "known_occurrence_count": known_occurrences,
        "known_type_count": known_types,
        "known_column_indices": known_column_indices,
        "descriptor_changed_under_shuffle": changed_descriptors,
        "pair_bag_preserved": bag_preserved,
        "segment_sizes_preserved": segment_sizes_preserved,
    }
    return {"bag": bag_matrix, "real": real_matrix, "shuffled": shuffled_matrix}, metadata


__all__ = ["POSITION_BINS", "SEED", "build_bigram_blocks"]
