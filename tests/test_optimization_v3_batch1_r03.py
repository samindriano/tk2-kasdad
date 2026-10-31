"""Synthetic, no-fit contract tests for the R03 split-vocabulary policy."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.optimization_v3_batch1_r03 import (
    EXPECTED_SCORE_CYCLES,
    FEATURE_NAMES,
    R03IntegrityError,
    SplitBudgetSparseRegressor,
    apply_fixed_route,
    estimate_sparse_memory,
    rank50_within_cycle,
    split_tfidf_parameters,
)


def test_recipe_uses_separate_caps_and_inherited_tfidf_settings() -> None:
    params = split_tfidf_parameters()
    assert set(params) == {"unigram", "bigram"}
    assert params["unigram"]["ngram_range"] == (1, 1)
    assert params["bigram"]["ngram_range"] == (2, 2)
    for block in params.values():
        assert block["max_features"] == 30_000
        assert block["min_df"] == 2
        assert block["sublinear_tf"] is True
        assert block["dtype"] is np.float64


def test_rank50_uses_average_ties_and_ranks_cycles_separately() -> None:
    raw = np.array([10.0, 20.0, 20.0, 40.0, 100.0, 100.0])
    cycle = np.array([56, 56, 56, 56, 57, 57])
    expected = np.array([17.5, 41.25, 41.25, 70.0, 87.5, 87.5])
    observed = rank50_within_cycle(raw, cycle)
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("cutoff", [55, 67])
def test_fixed_route_requires_full_standard_horizon(cutoff: int) -> None:
    cycle = np.asarray(EXPECTED_SCORE_CYCLES[cutoff], dtype=np.int64)
    raw = np.linspace(5.0, 95.0, len(cycle))
    ridge = np.full(len(cycle), 50.0)
    rows = np.arange(100, 100 + len(cycle), dtype=np.int64)
    prediction, rank50 = apply_fixed_route(
        cutoff=cutoff, raw_prediction=raw, ridge25_prediction=ridge,
        cycle_code=cycle, row_positions=rows,
    )
    assert prediction.shape == rank50.shape == cycle.shape
    assert np.isfinite(prediction).all()
    assert np.all((prediction >= 0.0) & (prediction <= 100.0))
    np.testing.assert_allclose(
        prediction, np.clip(0.75 * ridge + 0.25 * rank50, 0.0, 100.0),
        rtol=0.0, atol=1e-12,
    )
    with pytest.raises(R03IntegrityError, match="exact STANDARD"):
        apply_fixed_route(cutoff=cutoff, raw_prediction=raw[:4],
                          ridge25_prediction=ridge[:4], cycle_code=cycle[:4])


def test_fixed_route_rejects_duplicate_or_misaligned_row_ids() -> None:
    cycle = np.asarray(EXPECTED_SCORE_CYCLES[55], dtype=np.int64)
    raw = np.full(len(cycle), 50.0)
    ridge = np.full(len(cycle), 50.0)
    rows = np.arange(len(cycle), dtype=np.int64)
    rows[-1] = rows[-2]
    with pytest.raises(R03IntegrityError, match="duplicated"):
        apply_fixed_route(cutoff=55, raw_prediction=raw, ridge25_prediction=ridge,
                          cycle_code=cycle, row_positions=rows)
    with pytest.raises(R03IntegrityError, match="misaligned"):
        apply_fixed_route(cutoff=55, raw_prediction=raw, ridge25_prediction=ridge[:-1],
                          cycle_code=cycle)


def test_memory_preflight_is_sparse_and_enforces_cap_and_reserve() -> None:
    fit_x = pd.DataFrame({"signal_stream": ["alpha beta", "beta gamma delta"]})
    pass_result = estimate_sparse_memory(
        fit_x, available_memory_bytes=16 * 1024 ** 3)
    assert pass_result["status"] == "PASS"
    assert pass_result["sparse_only"] is True
    assert pass_result["concurrent_training_jobs"] == 1
    assert pass_result["threads"] == 2
    assert pass_result["estimated_peak_is_observed"] is False
    blocked = estimate_sparse_memory(fit_x, available_memory_bytes=1024 ** 3)
    assert blocked["status"] == "BLOCKED_RESOURCE"
    assert blocked["allowed_bytes"] == 0


def test_feature_selector_excludes_target_and_identifier_without_fitting() -> None:
    model = SplitBudgetSparseRegressor()
    frame = pd.DataFrame({name: [None] for name in FEATURE_NAMES})
    selected = model._select(frame)
    assert tuple(selected.columns) == FEATURE_NAMES
    with pytest.raises(R03IntegrityError, match="feature allowlist"):
        model._select(frame.assign(threat_rank=[1.0]))
    with pytest.raises(R03IntegrityError, match="feature allowlist"):
        model._select(frame.assign(record_id=["row-1"]))
