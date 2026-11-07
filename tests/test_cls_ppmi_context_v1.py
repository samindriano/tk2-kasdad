from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.cls_ppmi_context_core_v1 import (  # noqa: E402
    assert_prefix_cycles,
    binary_metrics,
    cooccurrence_counts,
    document_means,
    ordered_unigram_pairs,
    positive_ppmi,
    sha256_array,
    shuffle_documents,
    release_advisory_byte_lock,
    try_advisory_byte_lock,
)


def direct_pair_counts(documents: list[list[int]], vocab_size: int) -> sparse.csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    for document in documents:
        for i, left in enumerate(document):
            if left < 0:
                continue
            for distance in (1, 2):
                j = i + distance
                if j >= len(document) or document[j] < 0:
                    continue
                if any(value < 0 for value in document[i + 1:j]):
                    continue
                right = document[j]
                rows.extend((left, right))
                cols.extend((right, left))
    return sparse.coo_matrix((np.ones(len(rows), dtype=np.int32), (rows, cols)),
                             shape=(vocab_size, vocab_size)).tocsr()


class PPMIContextTests(unittest.TestCase):
    def test_sparse_pair_accumulation_matches_direct_enumeration(self) -> None:
        documents = [[0, 1, 2, -1, 0, 1], [2, 1], [3]]
        actual = cooccurrence_counts(documents, vocab_size=4)
        expected = direct_pair_counts(documents, vocab_size=4)
        self.assertTrue(sparse.isspmatrix_csr(actual))
        self.assertEqual((actual != expected).nnz, 0)
        self.assertEqual(actual[0, 2], 1)
        self.assertEqual(actual[2, 0], 1)
        self.assertEqual(actual[0, 1], 2)
        self.assertEqual(actual[1, 0], 2)

    def test_oov_and_document_boundaries_do_not_create_pairs(self) -> None:
        actual = cooccurrence_counts([[0, -1, 1], [2], [3]], vocab_size=4)
        self.assertEqual(actual.nnz, 0)
        boundary = cooccurrence_counts([[0], [1]], vocab_size=2)
        self.assertEqual(boundary.nnz, 0)

    def test_shuffle_preserves_each_documents_token_marginals(self) -> None:
        documents = [[1, 2, 1, -1, 3], [4, 4, 2], [], [8]]
        shuffled = shuffle_documents(documents, seed=42)
        self.assertEqual([sorted(row) for row in documents], [sorted(row) for row in shuffled])
        self.assertEqual(shuffled, shuffle_documents(documents, seed=42))
        self.assertNotEqual(documents[0], shuffled[0])

    def test_ppmi_matches_manual_arithmetic_and_zeros_stay_sparse(self) -> None:
        counts = sparse.csr_matrix(np.asarray([[0, 3], [1, 0]], dtype=np.int32))
        actual = positive_ppmi(counts)
        self.assertTrue(sparse.isspmatrix_csr(actual))
        self.assertAlmostEqual(float(actual[0, 1]), float(np.log(4.0 / 3.0)), places=12)
        self.assertAlmostEqual(float(actual[1, 0]), float(np.log(4.0)), places=12)
        self.assertEqual(actual.nnz, 2)
        self.assertEqual(positive_ppmi(sparse.csr_matrix((5, 5))).nnz, 0)

    def test_means_exclude_oov_and_return_zero_for_empty_known_tokens(self) -> None:
        embeddings = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        means, stats = document_means([[0, 1, -1], [-1, -1], []], embeddings,
                                      missing_mask=[False, False, True])
        np.testing.assert_allclose(means[0], [2.0, 3.0])
        np.testing.assert_array_equal(means[1:], np.zeros((2, 2)))
        self.assertEqual(stats["known_token_occurrences"], 2)
        self.assertEqual(stats["oov_token_occurrences"], 3)
        self.assertEqual(stats["rows_without_known_tokens"], 1)

    def test_svd_fit_transform_save_reload_is_repeatable(self) -> None:
        counts = cooccurrence_counts([[0, 1, 2, 3], [3, 2, 1, 0], [0, 2, 1, 3]], 4)
        ppmi = positive_ppmi(counts)
        first = TruncatedSVD(n_components=2, random_state=42, n_iter=5).fit_transform(ppmi)
        second = TruncatedSVD(n_components=2, random_state=42, n_iter=5).fit_transform(ppmi)
        np.testing.assert_allclose(first, second, rtol=0.0, atol=1e-12)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embedding.npz"
            np.savez_compressed(path, embedding=first)
            with np.load(path, allow_pickle=False) as saved:
                restored = saved["embedding"]
            np.testing.assert_array_equal(first, restored)
            self.assertEqual(sha256_array(first), sha256_array(restored))

    def test_train_only_prefix_guard(self) -> None:
        assert_prefix_cycles([47, 60, 67], 67)
        with self.assertRaises(ValueError):
            assert_prefix_cycles([47, 68], 67)

    def test_unigram_digest_pairs_keep_original_tfidf_column_indices(self) -> None:
        vectorizer = SimpleNamespace(vocabulary_={"bigram pair": 11, "alpha": 4,
                                                 "beta": 9, "gamma": 6})
        self.assertEqual(ordered_unigram_pairs(vectorizer), [("alpha", 4), ("gamma", 6), ("beta", 9)])

    def test_advisory_marker_releases_without_changing_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "marker.lock"
            path.write_bytes(b"\0")
            with path.open("r+b") as handle:
                self.assertTrue(try_advisory_byte_lock(handle))
                release_advisory_byte_lock(handle)
                self.assertTrue(try_advisory_byte_lock(handle))
                release_advisory_byte_lock(handle)
            self.assertEqual(path.read_bytes(), b"\0")

    def test_metrics_confusion_and_independent_f1(self) -> None:
        result = binary_metrics([0, 1, 1, 0], [0.1, 0.9, 0.4, 0.8], [0, 1, 0, 1])
        self.assertEqual(result["confusion_matrix_labels_0_1"], [[1, 1], [1, 1]])
        self.assertEqual(result["f1"], result["independent_f1"])
        self.assertEqual(result["recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
