from __future__ import annotations

import hashlib

import numpy as np

from scripts.complete_optimization_v3_c10_trigram_e79_inference_receipt import array_digest


def test_array_digest_uses_default_float64_conversion_for_labels() -> None:
    labels = np.asarray([0, 1, 1, 0], dtype=np.int64)
    expected = hashlib.sha256(labels.astype("<f8").tobytes()).hexdigest()
    assert array_digest(labels) == expected
