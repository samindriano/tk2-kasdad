"""Sparse word-context PPMI/SVD features for the CLS_PPMI_CONTEXT_V1 pilot."""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Iterable, Sequence

import numpy as np
from scipy import sparse


WINDOW_DISTANCES = (1, 2)
SHUFFLE_SEED = 42
EMBEDDING_DIM = 32


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_array(value: Any) -> str:
    array = np.asarray(value)
    header = canonical_sha256({"shape": list(array.shape), "dtype": str(array.dtype)})
    digest = hashlib.sha256(header.encode("ascii"))
    if array.dtype.kind in "OUS":
        digest.update(json.dumps(array.astype(str).tolist(), ensure_ascii=False,
                                 separators=(",", ":")).encode("utf-8"))
    else:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def ordered_unigrams(vectorizer: Any) -> list[str]:
    """Return unigram terms in frozen vectorizer column order."""
    return [term for term, _ in ordered_unigram_pairs(vectorizer)]


def ordered_unigram_pairs(vectorizer: Any) -> list[tuple[str, int]]:
    """Return unigram terms with their original TF-IDF column indices."""
    vocabulary = getattr(vectorizer, "vocabulary_", None)
    if not isinstance(vocabulary, dict) or not vocabulary:
        raise ValueError("Frozen vectorizer has no fitted vocabulary_")
    terms = sorted(((str(term), int(index)) for term, index in vocabulary.items()
                    if " " not in str(term)), key=lambda item: item[1])
    if not terms or [index for _, index in terms] != sorted(index for _, index in terms):
        raise ValueError("Frozen unigram map is empty or unordered")
    if len({term for term, _ in terms}) != len(terms):
        raise ValueError("Frozen unigram map contains duplicate terms")
    return terms


def tokenize_streams(streams: Sequence[Any], vectorizer: Any,
                     unigram_to_index: dict[str, int],
                     missing_mask: Any | None = None) -> tuple[list[list[int]], np.ndarray, dict[str, int]]:
    """Tokenize with the fitted vectorizer's separate preprocessor/tokenizer.

    Unknown vocabulary terms are represented by -1, preserving their position so
    local pairs cannot jump across an OOV token. Missing streams have no sequence.
    """
    values = list(streams)
    if missing_mask is None:
        missing = np.asarray([value is None for value in values], dtype=bool)
    else:
        missing = np.asarray(missing_mask, dtype=bool).reshape(-1)
    if len(values) != len(missing):
        raise ValueError("Text streams and missing mask are not aligned")
    preprocessor = vectorizer.build_preprocessor()
    tokenizer = vectorizer.build_tokenizer()
    documents: list[list[int]] = []
    token_count = known_count = oov_count = 0
    for value, is_missing in zip(values, missing, strict=True):
        if is_missing:
            documents.append([])
            continue
        prepared = preprocessor(str(value))
        tokens = tokenizer(prepared)
        mapped = [unigram_to_index.get(token, -1) for token in tokens]
        documents.append(mapped)
        token_count += len(mapped)
        known_count += sum(index >= 0 for index in mapped)
        oov_count += sum(index < 0 for index in mapped)
    stats = {"present_streams": int((~missing).sum()),
             "missing_streams": int(missing.sum()), "token_occurrences": int(token_count),
             "known_token_occurrences": int(known_count), "oov_token_occurrences": int(oov_count)}
    return documents, missing, stats


def shuffle_documents(documents: Sequence[Sequence[int]], *, seed: int = SHUFFLE_SEED) -> list[list[int]]:
    """Permute each document independently in stable row order, retaining OOV sentinels."""
    rng = np.random.default_rng(seed)
    shuffled: list[list[int]] = []
    for document in documents:
        values = np.asarray(document, dtype=np.int32)
        shuffled.append(rng.permutation(values).tolist() if len(values) > 1 else values.tolist())
    return shuffled


def cooccurrence_counts(documents: Iterable[Sequence[int]], vocab_size: int,
                        distances: Sequence[int] = WINDOW_DISTANCES) -> sparse.csr_matrix:
    """Build symmetric directed window counts without a dense vocabulary square.

    A token pair at distance one or two contributes weight one in each direction.
    The two-position path must contain only in-vocabulary tokens, so an OOV token
    prevents a distance-two connection across that gap. Documents never join.
    """
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    docs = [np.asarray(document, dtype=np.int32).reshape(-1) for document in documents]
    offsets = tuple(sorted(set(int(distance) for distance in distances)))
    if not offsets or offsets[0] <= 0:
        raise ValueError("window distances must be positive integers")
    pair_count = 0
    for document in docs:
        for i, left in enumerate(document):
            if left < 0:
                continue
            for distance in offsets:
                j = i + distance
                if j >= len(document):
                    break
                right = int(document[j])
                if right < 0:
                    continue
                if distance > 1 and np.any(document[i + 1:j] < 0):
                    continue
                pair_count += 2
    if pair_count == 0:
        return sparse.csr_matrix((vocab_size, vocab_size), dtype=np.int32)
    rows = np.empty(pair_count, dtype=np.int32)
    cols = np.empty(pair_count, dtype=np.int32)
    cursor = 0
    for document in docs:
        for i, left_value in enumerate(document):
            left = int(left_value)
            if left < 0:
                continue
            if left >= vocab_size:
                raise ValueError("token index exceeds frozen vocabulary width")
            for distance in offsets:
                j = i + distance
                if j >= len(document):
                    break
                right = int(document[j])
                if right < 0:
                    continue
                if right >= vocab_size:
                    raise ValueError("token index exceeds frozen vocabulary width")
                if distance > 1 and np.any(document[i + 1:j] < 0):
                    continue
                rows[cursor] = left
                cols[cursor] = right
                rows[cursor + 1] = right
                cols[cursor + 1] = left
                cursor += 2
    if cursor != pair_count:
        raise RuntimeError("co-occurrence preallocation count changed")
    values = np.ones(pair_count, dtype=np.int32)
    matrix = sparse.coo_matrix((values, (rows, cols)),
                               shape=(vocab_size, vocab_size), dtype=np.int32).tocsr()
    matrix.sum_duplicates()
    matrix.sort_indices()
    return matrix


def positive_ppmi(counts: sparse.spmatrix) -> sparse.csr_matrix:
    """Compute unsmoothed positive PMI on positive counts only; retain sparsity."""
    matrix = sparse.csr_matrix(counts, dtype=np.float64, copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("co-occurrence counts must be square")
    total = float(matrix.sum())
    if total <= 0:
        return sparse.csr_matrix(matrix.shape, dtype=np.float64)
    coo = matrix.tocoo(copy=False)
    row_sum = np.asarray(matrix.sum(axis=1)).reshape(-1)
    col_sum = np.asarray(matrix.sum(axis=0)).reshape(-1)
    denom = row_sum[coo.row] * col_sum[coo.col]
    valid = (coo.data > 0) & (denom > 0)
    values = np.zeros(len(coo.data), dtype=np.float64)
    values[valid] = np.maximum(np.log((coo.data[valid] * total) / denom[valid]), 0.0)
    positive = values > 0.0
    ppmi = sparse.coo_matrix((values[positive], (coo.row[positive], coo.col[positive])),
                             shape=matrix.shape, dtype=np.float64).tocsr()
    ppmi.sum_duplicates()
    ppmi.eliminate_zeros()
    ppmi.sort_indices()
    return ppmi


def document_means(documents: Sequence[Sequence[int]], term_embeddings: np.ndarray,
                   missing_mask: Any | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Mean known-token embeddings by occurrence; OOVs are excluded from the mean."""
    embeddings = np.asarray(term_embeddings, dtype=np.float64)
    if embeddings.ndim != 2:
        raise ValueError("term_embeddings must be a 2D matrix")
    if len(documents) == 0:
        raise ValueError("at least one document is required")
    if missing_mask is None:
        missing = np.zeros(len(documents), dtype=bool)
    else:
        missing = np.asarray(missing_mask, dtype=bool).reshape(-1)
    if len(missing) != len(documents):
        raise ValueError("documents and missing mask are not aligned")
    means = np.zeros((len(documents), embeddings.shape[1]), dtype=np.float64)
    token_total = known_total = rows_with_known = 0
    for row, (document, is_missing) in enumerate(zip(documents, missing, strict=True)):
        if is_missing:
            continue
        indices = np.asarray(document, dtype=np.int64)
        token_total += len(indices)
        known = indices[indices >= 0]
        if np.any(known >= embeddings.shape[0]):
            raise ValueError("token index exceeds embedding rows")
        known_total += len(known)
        if len(known):
            means[row] = embeddings[known].mean(axis=0)
            rows_with_known += 1
    stats = {"documents": len(documents), "present_documents": int((~missing).sum()),
             "missing_documents": int(missing.sum()), "token_occurrences": int(token_total),
             "known_token_occurrences": int(known_total),
             "oov_token_occurrences": int(token_total - known_total),
             "rows_with_known_tokens": int(rows_with_known),
             "rows_without_known_tokens": int((~missing).sum() - rows_with_known),
             "token_coverage": (float(known_total / token_total) if token_total else 0.0)}
    return means, stats


def assert_prefix_cycles(cycles: Any, origin: int) -> None:
    values = np.asarray(cycles, dtype=np.int64).reshape(-1)
    if not len(values) or np.any(values > int(origin)):
        raise ValueError(f"fit cycle set escapes the declared prefix <= {origin}")


def resource_bytes_for_csr(nnz: int, rows: int, *, data_bytes: int = 8,
                           index_bytes: int = 4) -> int:
    """CSR payload estimate; deliberately excludes Python/workspace overhead."""
    if nnz < 0 or rows < 0:
        raise ValueError("CSR shape estimates cannot be negative")
    return int(nnz * (data_bytes + index_bytes) + (rows + 1) * index_bytes)


def binary_metrics(y_true: Any, probability: Any, decision: Any) -> dict[str, Any]:
    """Calculate the fixed binary metric contract and check confusion arithmetic."""
    from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                                 log_loss, precision_score, recall_score, roc_auc_score)

    y = np.asarray(y_true, dtype=np.int64).reshape(-1)
    p = np.asarray(probability, dtype=np.float64).reshape(-1)
    predicted = np.asarray(decision, dtype=np.int64).reshape(-1)
    if not len(y) or len(y) != len(p) or len(y) != len(predicted):
        raise ValueError("metric inputs must be nonempty and row-aligned")
    if not np.isin(y, (0, 1)).all() or not np.isin(predicted, (0, 1)).all():
        raise ValueError("labels and decisions must be binary 0/1")
    if not np.isfinite(p).all() or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("probabilities must be finite and in [0,1]")
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel().tolist()
    precision = float(precision_score(y, predicted, zero_division=0))
    recall = float(recall_score(y, predicted, zero_division=0))
    f1 = float(f1_score(y, predicted, zero_division=0))
    independent_f1 = float((2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0)
    if not np.isclose(f1, independent_f1, rtol=0.0, atol=1e-15):
        raise RuntimeError("independent F1 arithmetic disagrees with sklearn")
    result: dict[str, Any] = {
        "n": int(len(y)), "positive": int(y.sum()), "negative": int(len(y) - y.sum()),
        "confusion_matrix_labels_0_1": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "precision": precision, "recall": recall, "f1": f1,
        "accuracy": float((tn + tp) / len(y)),
        "average_precision": (float(average_precision_score(y, p))
                              if len(np.unique(y)) == 2 else None),
        "roc_auc": (float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None),
        "log_loss": float(log_loss(y, np.clip(p, 1e-15, 1.0 - 1e-15), labels=[0, 1])),
        "brier_score": float(np.mean((p - y) ** 2)),
        "independent_f1": independent_f1,
    }
    return result


def try_advisory_byte_lock(file_handle: Any) -> bool:
    """Try to lock byte zero without changing the shared marker file."""
    file_handle.seek(0)
    if os.name == "nt":
        import msvcrt
        try:
            msvcrt.locking(file_handle.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False
    import fcntl
    try:
        fcntl.flock(file_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def release_advisory_byte_lock(file_handle: Any) -> None:
    """Release byte zero on the shared marker without deleting or rewriting it."""
    file_handle.seek(0)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(file_handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(file_handle.fileno(), fcntl.LOCK_UN)
