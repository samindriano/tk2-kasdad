from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from src.models.classification_a02_neighbors import (
    QUERY_BLOCK_ROWS,
    ResidualNeighborPredictor,
    fit_text_space,
    permute_residuals_within_cycle_probability_decile,
)


def _space_and_model(texts, ids, rows, cycles, probs, residuals, *, origin=50, arm="observed"):
    # Keep the exact min_df=3/max_df=.995 production recipe feasible in a tiny
    # fixture without modifying its parameters: marker appears in 3/6+ docs.
    prefix_texts = list(texts) + ["marker fillerone", "marker fillertwo", "marker fillerthree"]
    prefix_rows = np.arange(0, len(prefix_texts), dtype=np.int64)
    prefix_cycles = np.full(len(prefix_rows), origin, dtype=np.int64)
    space, _prefix_matrix, _missing = fit_text_space(
        prefix_texts, prefix_rows, prefix_cycles, origin_cycle=origin
    )
    model = ResidualNeighborPredictor(origin_cycle=origin, residual_arm=arm)
    model.fit(space, texts, ids, rows, cycles, probs, residuals)
    return space, model


def test_exact_ties_choose_smallest_source_row_first_and_match_bruteforce():
    bank_texts = ["alpha shared", "beta shared", "gamma shared", "alpha shared"]
    bank_ids = ["row-90", "row-10", "row-30", "row-20"]
    bank_rows = np.array([10, 20, 30, 40])
    bank_cycles = np.array([48, 48, 49, 49])
    bank_probabilities = np.array([0.4, 0.6, 0.3, 0.7])
    residuals = np.array([0.3, -0.2, 0.5, -0.1])
    space, model = _space_and_model(bank_texts, bank_ids, bank_rows, bank_cycles,
                                    bank_probabilities, residuals)
    out = model.predict(["alpha shared"], ["query"], [100], [51], [0.6])

    # Independent direct cosine calculation for this tiny fixture.
    q, _ = space.transform(["alpha shared"])
    bank, _ = space.transform(bank_texts)
    dense = (q @ bank.T).toarray()[0]
    positive = np.flatnonzero(dense > 0)
    expected_idx = sorted(positive, key=lambda i: (-dense[i], bank_rows[i]))[:32]
    assert out["neighbor_ids"][0, :len(expected_idx)].tolist() == [
        f"str:{bank_ids[i]}" for i in expected_idx
    ]
    assert out["neighbor_source_rows"][0, :len(expected_idx)].tolist() == [
        int(bank_rows[i]) for i in expected_idx
    ]
    expected_correction = np.clip(
        sum((dense[i] ** 2) * residuals[i] for i in expected_idx)
        / (4.0 + sum(dense[i] ** 2 for i in expected_idx)), -0.05, 0.05
    )
    assert out["correction"][0] == pytest.approx(expected_correction)
    assert out["candidate_probability"][0] == pytest.approx(np.clip(0.6 + expected_correction, 0, 1))


def test_exact_ties_use_source_row_not_record_id():
    bank_texts = ["alpha identical", "alpha identical", "alpha identical"]
    bank_ids = ["z", "a", "m"]
    bank_rows = np.array([81, 11, 51])
    # Required bank ordering is source-row order, so reorder all aligned arrays.
    order = np.argsort(bank_rows)
    bank_texts = [bank_texts[i] for i in order]
    bank_ids = [bank_ids[i] for i in order]
    bank_rows = bank_rows[order]
    cycles = np.full(3, 49)
    probs = np.full(3, 0.5)
    residuals = np.array([0.1, 0.2, 0.3])[order]
    _space, model = _space_and_model(bank_texts, bank_ids, bank_rows, cycles, probs, residuals)
    out = model.predict(["alpha identical"], ["query"], [100], [51], [0.5])
    assert out["neighbor_source_rows"][0].tolist() == [11, 51, 81]
    assert out["neighbor_ids"][0].tolist() == ["str:a", "str:m", "str:z"]


def test_zero_vector_and_missing_stream_have_zero_correction_and_exact_route():
    texts = ["alpha beta", "gamma delta", "epsilon zeta"]
    probs = np.array([0.2, 0.7, 0.4])
    residuals = np.array([0.4, -0.2, 0.1])
    _space, model = _space_and_model(texts, ["a", "b", "c"], [1, 2, 3],
                                     [48, 49, 50], probs, residuals)
    baseline = np.array([0.123456789, 0.8, 0.3])
    out = model.predict(["unseenwordzz", "", "alpha beta"], ["q1", "q2", "q3"],
                        [100, 101, 102], [51, 51, 51], baseline,
                        missing_mask=[False, True, False])
    assert out["correction"][0] == 0.0  # non-missing but TF-IDF zero vector
    assert out["correction"][1] == 0.0
    assert out["candidate_probability"][1] == baseline[1]  # exact base branch
    assert out["neighbor_count"][:2].tolist() == [0, 0]


def test_shuffle_preserves_each_cycle_decile_residual_multiset_and_singletons():
    cycles = np.array([48, 48, 48, 48, 49, 49, 50])
    probability = np.array([0.11, 0.19, 0.11, 0.29, 0.91, 0.99, 0.55])
    residual = np.array([-0.5, 0.4, 0.2, 0.1, 0.33, -0.22, 0.9])
    rows = np.arange(7, dtype=np.int64)
    shuffled, assignment, strata = permute_residuals_within_cycle_probability_decile(
        residual, cycles, probability, rows, seed=42
    )
    deciles = np.minimum(9, np.floor(10 * probability).astype(int))
    for cycle in np.unique(cycles):
        for decile in range(10):
            idx = np.flatnonzero((cycles == cycle) & (deciles == decile))
            if len(idx):
                assert sorted(shuffled[idx].tolist()) == sorted(residual[idx].tolist())
                assert set(assignment[idx].tolist()) == set(idx.tolist())
                if len(idx) == 1:
                    assert assignment[idx[0]] == idx[0]
    shuffled_again, assignment_again, strata_again = permute_residuals_within_cycle_probability_decile(
        residual, cycles, probability, rows, seed=42
    )
    assert np.array_equal(shuffled, shuffled_again)
    assert np.array_equal(assignment, assignment_again)
    assert strata == strata_again


def test_bank_and_query_boundaries_reject_duplicates_future_and_self_ids():
    texts = ["alpha beta", "gamma delta"]
    prefix_texts = texts + ["epsilon zeta", "marker fillerone", "marker fillertwo", "marker fillerthree"]
    space, _matrix, _missing = fit_text_space(
        prefix_texts, np.arange(len(prefix_texts)), np.full(len(prefix_texts), 50), origin_cycle=50
    )
    model = ResidualNeighborPredictor(origin_cycle=50)
    with pytest.raises(ValueError, match="duplicate row IDs"):
        model.fit(space, texts, ["same", "same"], [1, 2], [49, 50], [0.4, 0.5], [0.1, -0.2])
    with pytest.raises(ValueError, match="future cycle"):
        model.fit(space, texts, ["a", "b"], [1, 2], [49, 51], [0.4, 0.5], [0.1, -0.2])

    model.fit(space, texts, ["a", "b"], [1, 2], [49, 50], [0.4, 0.5], [0.1, -0.2])
    with pytest.raises(ValueError, match="overlap"):
        model.predict(["alpha beta"], ["a"], [3], [51], [0.5])
    with pytest.raises(ValueError, match="strictly increasing"):
        model.predict(["alpha beta", "gamma delta"], ["q1", "q2"], [4, 3], [51, 51], [0.5, 0.5])


def test_api_has_no_query_label_argument_and_caps_query_blocks():
    assert QUERY_BLOCK_ROWS == 128
    assert "query_labels" not in inspect.signature(ResidualNeighborPredictor.predict).parameters
    texts = ["alpha beta", "gamma delta", "epsilon zeta"]
    space, model = _space_and_model(texts, ["a", "b", "c"], [1, 2, 3],
                                    [48, 49, 50], [0.2, 0.7, 0.4], [0.1, -0.2, 0.3])
    query_texts = ["alpha beta"] * (QUERY_BLOCK_ROWS + 3)
    query_ids = [f"q{i}" for i in range(len(query_texts))]
    query_rows = np.arange(100, 100 + len(query_texts))
    out = model.predict(query_texts, query_ids, query_rows,
                        np.full(len(query_rows), 51), np.full(len(query_rows), 0.5))
    assert len(out["candidate_probability"]) == QUERY_BLOCK_ROWS + 3
    assert np.all(np.isfinite(out["candidate_probability"]))
