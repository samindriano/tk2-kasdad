from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.optimization_v3_cls_residual_fullsix_analysis_v2 import (
    quartile_codes, stable_identifier_digest,
)
from scripts.submit_regression_research import ordered_ids_sha256


def test_tied_binary_edges_preserve_rows_and_expose_empty_bins():
    values = np.array([0.0, 1.0, np.nan])
    codes = quartile_codes(values, [0.0, 0.0, 1.0])
    assert codes.shape == values.shape
    assert codes[2] == -1
    assert set(codes[:2].tolist()) == {0, 2}
    counts = np.bincount(codes[codes >= 0], minlength=4)
    assert int(counts.sum()) == 2
    assert counts[1] == 0 and counts[3] == 0


def test_constant_edges_still_assign_finite_rows_without_fabricating_support():
    values = np.array([0.0, 0.0, 1.0, np.inf])
    codes = quartile_codes(values, [0.0, 0.0, 0.0])
    assert codes.tolist() == [0, 0, 3, -1]
    assert int(np.sum(codes >= 0)) == 3


def test_string_record_ids_hash_in_order_without_numeric_coercion():
    ids = np.array(["rec_a", "rec_b", "rec_c"], dtype=object)
    forward = stable_identifier_digest(ids)
    reverse = stable_identifier_digest(ids[::-1])
    assert len(forward) == 64
    assert forward != reverse
    assert forward == ordered_ids_sha256(ids.tolist())
