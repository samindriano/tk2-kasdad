from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts import optimization_v3_rnbr_official as official
from scripts.optimization_v3_rnbr_official_verify import independent_prediction_recompute


def _synthetic_inputs():
    ids = np.asarray(["d0", "d1", "d2", "d3", "q", "d4", "m"], dtype="U2")
    # Deliberately not cycle-sorted: source row order is not temporal order.
    cycles = np.asarray([56, 57, 58, 59, 68, 60, 68], dtype=np.int64)
    present = np.asarray([True, True, True, True, True, True, False])
    train = {"rows": np.arange(7, dtype=np.int64), "ids": ids,
             "cycles": cycles, "present": present}
    geometry = {
        "score_rows": np.asarray([4, 6], dtype=np.int64),
        "score_record_ids": np.asarray(["q", "m"], dtype="U2"),
        "score_cycles": np.asarray([68, 68], dtype=np.int64),
        "neighbor_count": np.asarray([5, 0], dtype=np.int16),
        "neighbor_source_rows": np.asarray([[0, 1, 2, 3, 5], [-1, -1, -1, -1, -1]], dtype=np.int64),
        "neighbor_ids": np.asarray([["str_:d0", "str_:d1", "str_:d2", "str_:d3", "str_:d4"],
                                    ["", "", "", "", ""]], dtype="U8"),
        "neighbor_similarities": np.asarray([[0.9, 0.8, 0.7, 0.6, 0.5], [0, 0, 0, 0, 0]], dtype=np.float64),
    }
    cache = {
        55: {"rows": np.asarray([0, 1, 2, 3]), "ids": ids[:4],
             "prediction": np.asarray([39, 40, 41, 42], dtype=np.float64)},
        59: {"rows": np.asarray([5]), "ids": ids[5:6],
             "prediction": np.asarray([43], dtype=np.float64)},
        67: {"rows": np.asarray([4, 6]), "ids": ids[[4, 6]],
             "prediction": np.asarray([45, 60], dtype=np.float64)},
    }
    donor_targets = {"ids": ids[[0, 1, 2, 3, 5]], "cycles": cycles[[0, 1, 2, 3, 5]],
                     "target": np.asarray([40, 41, 42, 43, 44], dtype=np.float64)}
    freeze = {"freeze_signature": "a" * 64}
    return train, geometry, cache, donor_targets, freeze


def test_official_payload_builder_matches_independent_formula_on_synthetic_fixture():
    train, geometry, cache, targets, freeze = _synthetic_inputs()
    arrays, metadata = official.build_official_payload(
        origin=67, arm="observed", train=train, cache=cache, geometry=geometry,
        donor_targets=targets, freeze=freeze, source_commit="synthetic-test-only",
        dataset_tag="RNBR_SYNTHETIC_FIXTURE_V1",
    )
    assert metadata["identity"]["dataset_tag"] == "RNBR_SYNTHETIC_FIXTURE_V1"
    assert metadata["identity"]["new_estimator_fit_count"] == 0
    assert arrays["score_rows"].tolist() == [4, 6]
    assert arrays["donor_rows"].tolist() == [0, 1, 2, 3, 5]
    expected_delta, expected_prediction = independent_prediction_recompute(arrays)
    np.testing.assert_allclose(arrays["delta"], expected_delta, atol=1e-12, rtol=0)
    np.testing.assert_allclose(arrays["prediction"], expected_prediction, atol=1e-12, rtol=0)
    assert arrays["prediction"][1] == arrays["baseline_prediction"][1]
    assert arrays["delta"][1] == 0.0


def test_shuffled_memory_reuses_only_the_fixed_within_cell_permutation():
    train, geometry, cache, targets, freeze = _synthetic_inputs()
    arrays, _ = official.build_official_payload(
        origin=67, arm="shuffled", train=train, cache=cache, geometry=geometry,
        donor_targets=targets, freeze=freeze, source_commit="synthetic-test-only",
        dataset_tag="RNBR_SYNTHETIC_FIXTURE_V1",
    )
    assert set(arrays["donor_residuals_used"]) == set(arrays["donor_residuals"])
    expected_delta, expected_prediction = independent_prediction_recompute(arrays)
    np.testing.assert_allclose(arrays["delta"], expected_delta, atol=1e-12, rtol=0)
    np.testing.assert_allclose(arrays["prediction"], expected_prediction, atol=1e-12, rtol=0)


def test_target_reader_does_not_parse_non_allowlisted_query_targets(tmp_path, monkeypatch):
    path = tmp_path / "synthetic_train.csv"
    pd.DataFrame({
        "record_id": ["a", "b", "c", "d", "q"],
        "cycle_code": [56, 57, 58, 59, 68],
        "signal_stream": ["x", "x", "x", "x", "x"],
        "threat_rank": [20, 30, 40, 50, "QUERY_TARGET_MUST_NOT_PARSE"],
    }).to_csv(path, index=False)
    monkeypatch.setattr(official, "REG_TRAIN", path)
    observed = official.read_regression_targets(np.asarray([0, 2]), purpose="SYNTHETIC_DONOR_TEST")
    assert observed["ids"].tolist() == ["a", "c"]
    assert observed["target"].tolist() == [20.0, 40.0]


def test_bundle_readback_resume_and_corruption_rejection(tmp_path, monkeypatch):
    train, geometry, cache, targets, freeze = _synthetic_inputs()
    arrays, metadata = official.build_official_payload(
        origin=67, arm="observed", train=train, cache=cache, geometry=geometry,
        donor_targets=targets, freeze=freeze, source_commit="synthetic-test-only",
        dataset_tag="RNBR_SYNTHETIC_FIXTURE_V1",
    )
    monkeypatch.setattr(official, "RUN_ROOT", tmp_path)
    receipt = official.persist_bundle(official.FIT_IDS[0], arrays, metadata)
    resumed, resumed_receipt = official.load_bundle(official.FIT_IDS[0], freeze,
                                                     source_commit="synthetic-test-only")
    assert receipt == resumed_receipt
    assert all(np.array_equal(arrays[name], resumed[name]) for name in arrays)
    with (tmp_path / "bundles" / official.FIT_IDS[0] / "arrays.npz").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(official.OfficialRnbrError):
        official.load_bundle(official.FIT_IDS[0], freeze,
                             source_commit="synthetic-test-only")


def test_budget_snapshot_counts_released_capacity_without_rewriting_history():
    events = [
        {"event": "BATCH_ALLOWANCE_AUTHORIZED", "cap": 60,
         "category_caps": {"pilot": 26, "continuation": 24, "control_replay": 6, "technical_retry": 4}},
        {"event": "BUDGET_CAP_AMENDMENT", "global_cap": 70,
         "effective_caps": {"pilot": 38, "continuation": 31, "control_replay": 0, "technical_retry": 1}},
        {"event": "CARD_FITS_RESERVED", "reservation_key": "pilot", "category": "pilot", "fits": 38},
        {"event": "CARD_FITS_RESERVED", "reservation_key": "continuation", "category": "continuation", "fits": 31},
        {"event": "CARD_FITS_RESERVED", "reservation_key": "retry", "category": "technical_retry", "fits": 1},
        {"event": "RESERVATION_UNUSED_RELEASED", "reservation_key": "pilot", "unused_fits": 4},
    ]
    snapshot = official.ledger_snapshot(events)
    assert snapshot["global_cap"] == 70
    assert snapshot["effective_category_caps"]["pilot"] == 38
    assert snapshot["active_reservations"] == 66
    assert snapshot["active_reservations_by_category"]["pilot"] == 34
