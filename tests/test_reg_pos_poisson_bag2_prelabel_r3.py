"""Synthetic regressions for the target-blind R3 prelabel verifier."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scripts import verify_reg_pos_poisson_bag2_prelabel_r3 as r3


def _fixture():
    query_rows = np.arange(100, 108, dtype=np.int64)
    query_ids = np.asarray([f"id-{i}" for i in query_rows], dtype=str)
    query_cycles = np.arange(68, 76, dtype=np.int64)
    query_missing = np.asarray([False, False, True, False, False, False, True, False])
    baseline = np.asarray([15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0])
    pos42_raw = np.asarray([1.0, 4.0, 2.0, 8.0, 3.0, 7.0, 5.0, 9.0])
    replica17 = pos42_raw + 20.0
    replica137 = pos42_raw + 40.0
    params = {"intercept": 0.5, "slope_low": 1.0, "slope_high": 1.0}
    item = SimpleNamespace(query_rows=query_rows, query_ids=query_ids,
                           query_cycles=query_cycles, query_missing=query_missing)
    candidate = r3._candidate_replay(pos42_raw, replica17, replica137,
                                     query_cycles, baseline, query_missing, params)
    champion_mapped = r3._rank50(pos42_raw, query_cycles)
    champion_direct = np.clip(0.75 * baseline + 0.25 * champion_mapped, 0.0, 100.0)
    champion_direct[query_missing] = baseline[query_missing]
    positions = np.arange(4, 8, dtype=np.int64)
    local_missing = query_missing[positions].copy()
    local = {
        "physical_rows": query_rows[positions].copy(),
        "record_ids": query_ids[positions].copy(),
        "prediction_origins": np.full(len(positions), 67, dtype=np.int8),
        "original_bundle_indices": np.arange(9000 + positions[0], 9000 + positions[-1] + 1, dtype=np.int64),
        "cycles": query_cycles[positions].copy(),
        "missing_mask": local_missing,
        "direct_predictions": champion_direct[positions].copy(),
        "hinge_baseline": r3._apply_hinge(champion_direct[positions], local_missing, params),
    }
    identity = {
        "physical_rows": query_rows.copy(),
        "record_ids": query_ids.copy(),
        "prediction_origins": np.full(len(query_rows), 67, dtype=np.int8),
        "original_bundle_indices": np.arange(9000, 9000 + len(query_rows), dtype=np.int64),
        "cycles": query_cycles.copy(),
        "missing_mask": query_missing.copy(),
        "direct_predictions": np.arange(len(query_rows), dtype=np.float64),
    }
    return item, local, identity, candidate, champion_direct, params


def test_candidate_can_differ_from_champion_and_pass_replay():
    item, local, identity, candidate, champion, params = _fixture()
    assert not np.array_equal(candidate["full_direct"][4:], local["direct_predictions"])
    positions, local_hinge = r3._verify_comparator_replay(
        67, item, local, identity, params, champion)
    assert np.array_equal(positions, np.arange(4, 8))
    assert np.array_equal(local_hinge, local["hinge_baseline"])


def test_wrong_candidate_formula_fails_even_when_champion_is_valid():
    item, local, identity, replay, _, params = _fixture()
    saved = {key: value.copy() for key, value in replay.items()}
    saved["full_direct"][0] += 0.01
    with pytest.raises(r3.VerifyError, match="formula replay differs.*full_direct"):
        r3._assert_candidate_replay(saved, replay, np.linspace(15, 85, 8),
                                    item.query_missing, 67)


def test_wrong_champion_direct_fails():
    item, local, identity, _, champion, params = _fixture()
    local["direct_predictions"][0] += 0.01
    with pytest.raises(r3.VerifyError, match="champion identity or direct replay"):
        r3._verify_comparator_replay(67, item, local, identity, params, champion)


def test_wrong_missing_mask_fails_alignment():
    item, local, identity, _, champion, params = _fixture()
    local["missing_mask"][0] = not local["missing_mask"][0]
    with pytest.raises(r3.VerifyError, match="champion identity or direct replay"):
        r3._verify_comparator_replay(67, item, local, identity, params, champion)


def test_wrong_local_hinge_fails_replay():
    item, local, identity, _, champion, params = _fixture()
    local["hinge_baseline"][0] += 0.01
    with pytest.raises(r3.VerifyError, match="local hinge replay"):
        r3._verify_comparator_replay(67, item, local, identity, params, champion)


def test_wrong_original_bundle_index_fails_alignment():
    item, local, identity, _, champion, params = _fixture()
    local["original_bundle_indices"][0] += 1
    with pytest.raises(r3.VerifyError, match="champion identity or direct replay"):
        r3._verify_comparator_replay(67, item, local, identity, params, champion)


@pytest.mark.parametrize("field", ["record_ids", "cycles", "prediction_origins"])
def test_wrong_comparator_identity_field_fails_alignment(field):
    item, local, identity, _, champion, params = _fixture()
    if field == "record_ids":
        local[field][0] = "wrong-id"
    else:
        local[field][0] += 1
    with pytest.raises(r3.VerifyError, match="champion identity or direct replay"):
        r3._verify_comparator_replay(67, item, local, identity, params, champion)


def test_unknown_comparator_physical_row_fails_closed():
    item, local, identity, _, champion, params = _fixture()
    local["physical_rows"][0] = 999999
    with pytest.raises(r3.VerifyError, match="physical row is missing"):
        r3._verify_comparator_replay(67, item, local, identity, params, champion)
