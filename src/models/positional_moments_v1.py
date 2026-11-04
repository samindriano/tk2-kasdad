"""Continuous positional moments over a frozen C2_B unigram vocabulary.

No fitting or data access lives here. The caller supplies normalized documents
and a fitted prefix-only TF-IDF vectorizer; this module builds deterministic
feature rows and fixed routing arithmetic only.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src.models.positional_text_features import deterministic_order_shuffle


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, type) and value.__module__.startswith("numpy"):
        return f"{value.__module__}.{value.__qualname__}"
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Unsupported canonical value: {type(value).__name__}")


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(_jsonable(value), sort_keys=True, ensure_ascii=False,
                         allow_nan=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Any) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: Any) -> str:
    array = np.asarray(value)
    header = canonical_sha256({"shape": array.shape, "dtype": str(array.dtype)})
    digest = hashlib.sha256(header.encode("ascii"))
    if array.dtype.kind in "OUS":
        digest.update(json.dumps(array.astype(str).tolist(), ensure_ascii=False,
                                 separators=(",", ":")).encode("utf-8"))
    else:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def sha256_csr(value: sparse.spmatrix) -> str:
    matrix = sparse.csr_matrix(value, dtype=np.float64, copy=True)
    matrix.sum_duplicates()
    matrix.sort_indices()
    digest = hashlib.sha256(canonical_sha256({"shape": matrix.shape, "dtype": str(matrix.dtype)}).encode("ascii"))
    digest.update(np.asarray(matrix.indptr, dtype="<i8").tobytes())
    digest.update(np.asarray(matrix.indices, dtype="<i8").tobytes())
    digest.update(np.asarray(matrix.data, dtype="<f8").tobytes())
    return digest.hexdigest()


def tokenizer_idf_signature(vectorizer: TfidfVectorizer) -> dict[str, Any]:
    if not hasattr(vectorizer, "vocabulary_") or not hasattr(vectorizer, "idf_"):
        raise ValueError("A fitted prefix-only TF-IDF vectorizer is required")
    vocabulary = {str(token): int(index) for token, index in vectorizer.vocabulary_.items()}
    idf = np.asarray(vectorizer.idf_, dtype=np.float64)
    if len(idf) != len(vocabulary) or not np.isfinite(idf).all():
        raise ValueError("Frozen vectorizer vocabulary and IDF are inconsistent")
    return {
        "parameters": {key: _jsonable(value) for key, value in vectorizer.get_params(deep=False).items()},
        "vocabulary_idf_sha256": canonical_sha256({
            "vocabulary": sorted(vocabulary.items()), "idf_sha256": sha256_array(idf),
        }),
        "vocabulary_size": len(vocabulary),
        "unigram_size": sum(" " not in token for token in vocabulary),
    }


def build_moment_matrix(
    documents: Sequence[str],
    vectorizer: TfidfVectorizer,
    *,
    shuffled: bool = False,
    seed: int = 42,
) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    """Build W=[B, B*m1, B*m2] and jointly L2-normalize each row once.

    Token positions include OOV tokens. Only known unigram columns receive
    features. The bag channel B uses (1+log(count))*frozen IDF, L2 normalized
    on its own before the three channels are concatenated.
    """
    if not hasattr(vectorizer, "vocabulary_") or not hasattr(vectorizer, "idf_"):
        raise ValueError("A fitted prefix-only TF-IDF vectorizer is required")
    vocab = {str(token): int(index) for token, index in vectorizer.vocabulary_.items()}
    unigram_original = {token: index for token, index in vocab.items() if " " not in token}
    unigram = {token: compact for compact, token in enumerate(
        sorted(unigram_original, key=unigram_original.__getitem__))}
    idf = np.asarray(vectorizer.idf_, dtype=np.float64)
    if any(index < 0 or index >= len(idf) for index in unigram_original.values()):
        raise ValueError("Frozen unigram vocabulary has an invalid IDF index")
    width = len(unigram)
    preprocess = vectorizer.build_preprocessor()
    tokenizer = vectorizer.build_tokenizer()
    analyzer = vectorizer.build_analyzer()

    row_idx: list[int] = []
    col_idx: list[int] = []
    data: list[float] = []
    bag_row_idx: list[int] = []
    bag_col_idx: list[int] = []
    bag_data: list[float] = []
    doc_hashes: list[str] = []
    token_counts: list[int] = []
    known_counts: list[int] = []
    row_norms: list[float] = []
    zero_rows = 0
    changed_rows = 0

    for row, document_value in enumerate(documents):
        if not isinstance(document_value, str):
            raise TypeError("Documents must already follow the frozen _text_values conversion")
        document = document_value.strip()
        doc_hashes.append(hashlib.sha256(document.encode("utf-8", errors="surrogatepass")).hexdigest())
        tokens = tuple(tokenizer(preprocess(document)))
        analyzed = analyzer(document)
        if tuple(analyzed[:len(tokens)]) != tokens:
            raise ValueError("Frozen analyzer unigram token order does not match tokenizer output")
        if shuffled:
            shuffled_result = deterministic_order_shuffle(tokens, seed=seed)
            ordered = tuple(shuffled_result["tokens"])
            if Counter(ordered) != Counter(tokens) or len(ordered) != len(tokens):
                raise RuntimeError("Deterministic shuffle changed token bag or length")
            changed_rows += int(ordered != tokens)
        else:
            ordered = tokens

        n = len(ordered)
        token_counts.append(n)
        if n <= 1:
            h1 = np.zeros(n, dtype=np.float64)
            h2 = np.zeros(n, dtype=np.float64)
        else:
            positions = np.arange(n, dtype=np.float64)
            u = 2.0 * positions / float(n - 1) - 1.0
            h1 = u
            raw_h2 = (3.0 * np.square(u) - 1.0) / 2.0
            h2 = raw_h2 - float(raw_h2.mean())

        counts = Counter(ordered)
        bag_raw: dict[str, float] = {}
        moment_sums: dict[str, list[float]] = {}
        for token, count in counts.items():
            index = unigram.get(token)
            if index is not None:
                bag_raw[token] = (1.0 + math.log(int(count))) * float(idf[unigram_original[token]])
                moment_sums[token] = [0.0, 0.0, 0.0]
        for position, token in enumerate(ordered):
            sums = moment_sums.get(token)
            if sums is not None:
                sums[0] += float(h1[position])
                sums[1] += float(h2[position])
                sums[2] += 1.0
        known_counts.append(sum(int(counts[token]) for token in bag_raw))

        bag_norm = math.sqrt(sum(value * value for value in bag_raw.values()))
        if bag_norm == 0.0:
            zero_rows += 1
            row_norms.append(0.0)
            continue

        row_values: list[tuple[int, float]] = []
        for token, raw_value in bag_raw.items():
            base_value = raw_value / bag_norm
            if base_value != 0.0:
                bag_row_idx.append(row)
                bag_col_idx.append(unigram[token])
                bag_data.append(base_value)
            count = moment_sums[token][2]
            mean1 = moment_sums[token][0] / count if count else 0.0
            mean2 = moment_sums[token][1] / count if count else 0.0
            token_index = unigram[token]
            row_values.extend(((token_index, base_value),
                               (width + token_index, base_value * mean1),
                               (2 * width + token_index, base_value * mean2)))
        joint_norm = math.sqrt(sum(value * value for _, value in row_values))
        if not np.isfinite(joint_norm) or joint_norm <= 0:
            raise ValueError("Nonempty known-token row has invalid joint norm")
        norm_values = [(index, value / joint_norm) for index, value in row_values]
        row_norm = math.sqrt(sum(value * value for _, value in norm_values))
        if not np.isfinite(row_norm) or abs(row_norm - 1.0) > 1e-12:
            raise ValueError("Jointly normalized moment row is not unit norm")
        row_norms.append(row_norm)
        for column, value in norm_values:
            if value != 0.0:
                row_idx.append(row)
                col_idx.append(column)
                data.append(value)

    matrix = sparse.csr_matrix((np.asarray(data, dtype=np.float64),
                                 (np.asarray(row_idx, dtype=np.int64), np.asarray(col_idx, dtype=np.int64))),
                                shape=(len(documents), 3 * width), dtype=np.float64)
    bag_channel = sparse.csr_matrix((np.asarray(bag_data, dtype=np.float64),
                                     (np.asarray(bag_row_idx, dtype=np.int64), np.asarray(bag_col_idx, dtype=np.int64))),
                                    shape=(len(documents), width), dtype=np.float64)
    matrix.sum_duplicates()
    matrix.sort_indices()
    bag_channel.sum_duplicates()
    bag_channel.sort_indices()
    if not np.isfinite(matrix.data).all():
        raise ValueError("Moment matrix contains non-finite values")
    matrix_norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).reshape(-1))
    if not np.allclose(matrix_norms, np.asarray(row_norms), rtol=0.0, atol=1e-12):
        raise ValueError("Sparse matrix norms differ from construction audit")
    return matrix, {
        "shape": list(matrix.shape),
        "nnz": int(matrix.nnz),
        "sha256": sha256_csr(matrix),
        "documents_sha256": canonical_sha256(doc_hashes),
        "token_count_sha256": sha256_array(np.asarray(token_counts, dtype=np.int64)),
        "token_count_min": int(min(token_counts, default=0)),
        "token_count_max": int(max(token_counts, default=0)),
        "nonempty_rows": int(len(documents) - zero_rows),
        "zero_rows": int(zero_rows),
        "all_nonzero_rows_unit_norm": bool(np.allclose(matrix_norms[matrix_norms > 0], 1.0, rtol=0.0, atol=1e-12)),
        "norm_min": float(matrix_norms.min()) if len(matrix_norms) else 0.0,
        "norm_max": float(matrix_norms.max()) if len(matrix_norms) else 0.0,
        "known_token_occurrences": int(sum(known_counts)),
        "bag_channel_sha256": sha256_csr(bag_channel),
        "shuffled": bool(shuffled),
        "shuffle_seed": int(seed) if shuffled else None,
        "rows_with_changed_order": int(changed_rows),
    }


def fixed_blends(
    q_new: Any,
    p_a02: Any,
    p_oldpos: Any,
    present: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed moment-vs-A02 policy and current MIX50 comparator."""
    q = np.asarray(q_new, dtype=np.float64).reshape(-1)
    a02 = np.asarray(p_a02, dtype=np.float64).reshape(-1)
    old = np.asarray(p_oldpos, dtype=np.float64).reshape(-1)
    mask = np.asarray(present, dtype=bool).reshape(-1)
    if not (len(q) == len(a02) == len(old) == len(mask)):
        raise ValueError("Fixed blend input rows do not align")
    if any(not np.isfinite(values).all() for values in (q, a02, old)):
        raise ValueError("Fixed blend probabilities must be finite")
    if any(np.any((values < 0.0) | (values > 1.0)) for values in (q, a02, old)):
        raise ValueError("Fixed blend probabilities must be in [0,1]")
    candidate = np.where(mask, 0.5 * q + 0.5 * a02, old)
    mix50 = np.where(mask, 0.5 * old + 0.5 * a02, old)
    if not np.array_equal(candidate[~mask], old[~mask]) or not np.array_equal(mix50[~mask], old[~mask]):
        raise ValueError("Missing-stream route must equal the original pP exactly")
    return candidate, mix50


def direct_logit(matrix: sparse.spmatrix, coef: Any, intercept: Any) -> np.ndarray:
    x = sparse.csr_matrix(matrix, dtype=np.float64)
    beta = np.asarray(coef, dtype=np.float64).reshape(-1)
    bias = float(np.asarray(intercept, dtype=np.float64).reshape(-1)[0])
    if x.shape[1] != len(beta) or not np.isfinite(x.data).all() or not np.isfinite(beta).all() or not np.isfinite(bias):
        raise ValueError("Frozen bag model and base matrix do not align or are non-finite")
    value = np.asarray(x @ beta, dtype=np.float64).reshape(-1) + bias
    if not np.isfinite(value).all():
        raise ValueError("Direct frozen bag logit is non-finite")
    return value


__all__ = [
    "build_moment_matrix", "canonical_sha256", "direct_logit", "fixed_blends",
    "sha256_array", "sha256_bytes", "sha256_csr", "sha256_file",
    "tokenizer_idf_signature",
]
