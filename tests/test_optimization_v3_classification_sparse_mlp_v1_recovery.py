from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from scripts.optimization_v3_classification_sparse_mlp_v1_recovery import _vocab_sha


def test_recovery_vocabulary_hash_accepts_numpy_integer_indices():
    first = SimpleNamespace(vocabulary_={"alpha": np.int64(0), "beta": np.int64(1)})
    second = SimpleNamespace(vocabulary_={"beta": 1, "alpha": 0})
    assert _vocab_sha(first) == _vocab_sha(second)


def test_empty_vocabulary_recovery_hash_is_stable():
    assert _vocab_sha(None) == _vocab_sha(SimpleNamespace(vocabulary_={}))
