from __future__ import annotations

import hashlib

import numpy as np

from scripts.verify_recovered_optimization_v3_c10_trigram_e79 import _canonical_ids


def test_canonical_test_id_sequence_is_order_sensitive() -> None:
    ids = np.asarray(["id-1", "id-2", "id-3"], dtype=str)
    assert _canonical_ids(ids) == hashlib.sha256(b"id-1\nid-2\nid-3\n").hexdigest()
    assert _canonical_ids(ids[::-1]) != _canonical_ids(ids)
