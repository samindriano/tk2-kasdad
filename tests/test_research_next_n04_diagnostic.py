"""Synthetic contract and restart-safety checks for N04_DIAGNOSTIC."""
from __future__ import annotations

import json

import numpy as np
import pytest

from scripts import optimization_v3_research_next_n04_diagnostic as n04d
from scripts.optimization_v3_research_next_n04_diagnostic_verify import _rank50


def test_synthetic_weighted_pipeline_and_persistence_resume() -> None:
    result = n04d._synthetic_smoke()

    assert result["status"] == "PASS"
    assert result["official_task_fits"] == 0
    assert result["synthetic_predictor_fits"] == 1
    assert result["within_cycle_multiset_preserved"] is True
    assert result["rank50_and_blend_finite_bounded"] is True
    assert result["persistence_readback_resume"] == "PASS"


def test_rank50_uses_average_ties_within_each_cycle() -> None:
    raw = np.array([10.0, 10.0, 40.0, 99.0, 1.0, 1.0])
    cycles = np.array([55, 55, 55, 56, 56, 56])

    actual = _rank50(raw, cycles)

    # Average ties are ranked independently inside each complete cycle.
    expected = np.array([30.0, 30.0, 70.0, 99.5, 25.5, 25.5])
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


def test_write_once_is_idempotent_and_rejects_mutation(tmp_path) -> None:
    path = tmp_path / "receipt.json"
    receipt = {"status": "PASS", "fits": [55, 67]}

    n04d._write_once(path, receipt)
    before = path.read_bytes()
    n04d._write_once(path, receipt)

    assert path.read_bytes() == before
    assert json.loads(before) == receipt
    with pytest.raises(RuntimeError, match="refuses to overwrite immutable output"):
        n04d._write_once(path, {"status": "CHANGED"})

\n