from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import verify_optimization_v3_positional_moments_v1 as verifier
from scripts.optimization_v3_positional_moments_v1_prelabel_v2 import (
    PrelabelV2Error,
    rebuild_moments_scalar_order,
    synthetic_prelabel_rebuild_check,
)
from src.models.positional_moments_v1 import build_moment_matrix, sha256_csr


def _docs_and_vectorizer():
    from sklearn.feature_extraction.text import TfidfVectorizer

    docs = [
        "", "alpha", "alpha alpha", "beta alpha beta alpha beta",
        "alpha beta " * 300 + "alpha", "gamma " * 401,
        "café alpha " * 97 + "omega_oov " * 3,
        "delta " + "alpha beta gamma delta " * 41 + "delta",
    ]
    vectorizer = TfidfVectorizer(lowercase=True, strip_accents="unicode", min_df=1,
                                 max_df=1.0, ngram_range=(1, 2), dtype=np.float64)
    vectorizer.fit(["alpha beta gamma delta cafe epsilon", "alpha beta delta", "gamma cafe delta"])
    return docs, vectorizer


@pytest.mark.parametrize("shuffled", [False, True])
def test_scalar_rebuild_matches_frozen_formula_on_rounding_stress(shuffled: bool) -> None:
    docs, vectorizer = _docs_and_vectorizer()
    expected, _ = build_moment_matrix(docs, vectorizer, shuffled=shuffled, seed=42)
    actual = rebuild_moments_scalar_order(docs, vectorizer, shuffled=shuffled)
    assert actual.shape == expected.shape
    assert sha256_csr(actual) == sha256_csr(expected)
    assert np.array_equal(actual.indptr, expected.indptr)
    assert np.array_equal(actual.indices, expected.indices)
    assert np.array_equal(actual.data, expected.data)


def test_complete_synthetic_prelabel_reconstruction_and_routing_pass() -> None:
    result = synthetic_prelabel_rebuild_check()
    assert result["status"] == "PASS"
    assert result["query_labels_read"] is False
    assert result["test_or_sample_access"] is False
    assert len(result["checks"]) == 2
    assert all(item["status"] == "PASS" and item["missing_route_exact"]
               for item in result["checks"])


def test_scalar_rebuild_rejects_unfitted_vectorizer() -> None:
    from sklearn.feature_extraction.text import TfidfVectorizer
    with pytest.raises(PrelabelV2Error, match="fitted frozen prefix"):
        rebuild_moments_scalar_order(["alpha"], TfidfVectorizer(), shuffled=False)
