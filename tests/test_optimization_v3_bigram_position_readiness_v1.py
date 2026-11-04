from __future__ import annotations

import inspect

import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer

from src.models.optimization_v3_bigram_position_readiness_v1 import build_bigram_blocks


def _frozen_vectorizer(vocabulary: dict[str, int], idf: list[float]) -> TfidfVectorizer:
    """Construct a tiny frozen object without calling fit or fit_transform."""
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), strip_accents="unicode", lowercase=True,
        token_pattern=r"(?u)\b\w\w+\b",
    )
    vectorizer.vocabulary_ = dict(vocabulary)
    vectorizer.fixed_vocabulary_ = True
    vectorizer._tfidf = TfidfTransformer()
    vectorizer.idf_ = np.asarray(idf, dtype=np.float64)
    return vectorizer


def _assert_unit_row(matrix, row: int) -> None:
    norm = float(np.sqrt(matrix[row].multiply(matrix[row]).sum()))
    assert np.isclose(norm, 1.0, atol=1e-12)


def test_empty_one_and_two_tokens_have_expected_pair_support() -> None:
    vectorizer = _frozen_vectorizer({"alpha": 0, "alpha beta": 1}, [2.0, 4.0])
    matrices, meta = build_bigram_blocks(["", "alpha", "alpha beta"], vectorizer)

    assert meta["token_count"].tolist() == [0, 1, 2]
    assert meta["pair_count"].tolist() == [0, 0, 1]
    assert meta["known_occurrence_count"].tolist() == [0, 0, 1]
    assert matrices["real"].shape == (3, 3)
    assert matrices["real"][:2].nnz == 0
    assert matrices["real"][2].nnz == 1
    assert matrices["real"][2].indices.tolist() == [0]
    _assert_unit_row(matrices["real"], 2)


def test_repeated_unicode_pairs_are_counted_after_frozen_normalization() -> None:
    vectorizer = _frozen_vectorizer(
        {"cafe alpha": 0, "alpha cafe": 1}, [2.0, 3.0]
    )
    matrices, meta = build_bigram_blocks(["Café alpha café alpha"], vectorizer)

    assert meta["token_count"].tolist() == [4]
    assert meta["pair_count"].tolist() == [3]
    assert meta["known_occurrence_count"].tolist() == [3]
    assert meta["known_type_count"].tolist() == [2]
    # The repeated phrase occurs in both first and last thirds.
    assert matrices["real"].nnz == 3
    assert meta["pair_bag_preserved"].all()
    assert meta["segment_sizes_preserved"].all()


def test_oov_pairs_still_define_position_bins_and_shuffle_preserves_bag() -> None:
    vectorizer = _frozen_vectorizer(
        {"alpha beta": 0, "gamma delta": 1}, [2.0, 5.0]
    )
    docs = ["alpha beta gamma delta", "alpha beta gamma delta"]
    matrices_a, meta_a = build_bigram_blocks(docs, vectorizer, seed=42)
    matrices_b, meta_b = build_bigram_blocks(docs, vectorizer, seed=42)

    assert meta_a["pair_count"].tolist() == [3, 3]
    assert meta_a["known_occurrence_count"].tolist() == [2, 2]
    # The OOV middle pair retains its position in the full sequence: known
    # pairs land in bins 0 and 2, with the middle bin empty.
    assert matrices_a["real"][0].indices.tolist() == [0, 5]
    assert meta_a["pair_bag_preserved"].all()
    assert meta_a["segment_sizes_preserved"].all()
    assert np.array_equal(matrices_a["bag"].toarray(), matrices_b["bag"].toarray())
    assert np.array_equal(matrices_a["real"].toarray(), matrices_b["real"].toarray())
    assert np.array_equal(matrices_a["shuffled"].toarray(), matrices_b["shuffled"].toarray())
    assert meta_a["known_column_indices"] == meta_b["known_column_indices"]


def test_idf_and_original_column_order_are_aligned_before_row_normalization() -> None:
    # The original vectorizer assigns alpha-beta to column 3 and beta-gamma
    # to column 0; compact bigram columns must follow those original indices.
    vectorizer = _frozen_vectorizer(
        {"beta gamma": 0, "noise token": 1, "x y": 2, "alpha beta": 3},
        [2.0, 3.0, 5.0, 7.0],
    )
    matrices, meta = build_bigram_blocks(["alpha beta gamma"], vectorizer)
    row = matrices["real"].getrow(0)

    assert meta["bigram_vocabulary_size"] == 4
    # j=0 (alpha beta) goes to bin 0, compact col 3; j=1
    # (beta gamma) goes to bin 1, compact col 0 (offset 4).
    assert row.indices.tolist() == [3, 4]
    expected_norm = np.sqrt(7.0**2 + 2.0**2)
    assert np.allclose(row.data, [7.0 / expected_norm, 2.0 / expected_norm])
    assert np.isclose(float(matrices["bag"][0, 3]), 7.0 / expected_norm)
    assert np.isclose(float(matrices["bag"][0, 0]), 2.0 / expected_norm)


def test_public_builder_has_no_identifier_or_target_input() -> None:
    params = set(inspect.signature(build_bigram_blocks).parameters)
    assert params == {"documents", "vectorizer", "seed"}
    assert not ({"record_id", "artifact_signal", "threat_rank", "target", "labels"} & params)
