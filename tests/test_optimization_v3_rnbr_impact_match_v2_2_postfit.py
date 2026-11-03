import pytest
import numpy as np

from scripts.optimization_v3_rnbr_impact_match_v2_2_postfit_verify import (
    EXPECTED_SNAPSHOT,
    FIT_SOURCE_COMMIT,
    PostfitVerificationError,
    _NumpyArrayEqualCompatibility,
    validate_geometry_row_identity,
    validate_postfit_ledger,
    validate_view_cycles_row_order_aware,
)
from scripts import optimization_v3_rnbr_impact_match_v2 as runner


def _synthetic_events():
    events = []
    events.extend([
        {"event": "CARD_FITS_RESERVED", "reservation_key": runner.REPLAY_RESERVATION_KEY,
         "fit_ids": list(runner.REPLAY_FIT_IDS), "fits": 2,
         "freeze_signature": runner.REPLAY_FREEZE_SIGNATURE},
        {"event": "CARD_FITS_RESERVED", "reservation_key": runner.RESERVATION_KEY,
         "fit_ids": list(runner.FIT_IDS), "fits": 4,
         "freeze_signature": runner.REPLAY_FREEZE_SIGNATURE},
    ])
    for fit_id in runner.REPLAY_FIT_IDS:
        for event_name in ("FIT_STARTED", "FIT_SAVED"):
            events.append({"event": event_name, "reservation_key": runner.REPLAY_RESERVATION_KEY,
                           "fit_id": fit_id, "fit_kind": "TARGET_FREE_TFIDF_REPLAY"})
    for fit_id in runner.FIT_IDS:
        for event_name in ("FIT_STARTED", "FIT_SAVED"):
            events.append({"event": event_name, "reservation_key": runner.RESERVATION_KEY,
                           "fit_id": fit_id, "fit_kind": "TARGET_MEMORY_CONSTRUCTION",
                           "freeze_signature": runner._read_json(runner.FREEZE_PATH)["freeze_signature"],
                           "source_commit": FIT_SOURCE_COMMIT})
    return events


def test_postfit_ledger_accepts_exact_two_replay_four_memory_terminal_state():
    report = validate_postfit_ledger(_synthetic_events(), EXPECTED_SNAPSHOT)
    assert report["status"] == "PASS"
    assert report["replay_started"] == 2
    assert report["replay_saved"] == 2
    assert report["memory_started"] == 4
    assert report["memory_saved"] == 4
    assert report["memory_failed"] == 0


def test_postfit_ledger_rejects_stale_or_partial_fit_accounting():
    events = _synthetic_events()
    partial = [event for event in events if not (
        event.get("event") == "FIT_SAVED" and event.get("fit_id") == runner.FIT_IDS[-1]
    )]
    with pytest.raises(PostfitVerificationError):
        validate_postfit_ledger(partial, EXPECTED_SNAPSHOT)

    wrong_snapshot = {**EXPECTED_SNAPSHOT, "started": 79, "saved": 76}
    with pytest.raises(PostfitVerificationError):
        validate_postfit_ledger(events, wrong_snapshot)


def _unsorted_geometry_and_train():
    cycles = np.asarray([56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
                         79, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78], dtype=np.int64)
    rows = np.arange(len(cycles), dtype=np.int64)
    ids = np.asarray([f"record-{i}" for i in rows], dtype=object)
    present = np.ones(len(rows), dtype=bool)
    impacts = np.ones(len(rows), dtype=np.float64)
    train = {"rows": rows, "cycles": cycles, "ids": ids, "present": present}
    geometry = {}
    for origin, low, high in ((67, 68, 79), (75, 76, 79)):
        query_rows = rows[(cycles >= low) & (cycles <= high)]
        donor_rows = rows[(cycles >= 56) & (cycles <= origin) & present & np.isfinite(impacts)]
        neighbor_rows = np.full((len(query_rows), 1), donor_rows[0], dtype=np.int64)
        neighbor_ids = np.full(neighbor_rows.shape, ids[donor_rows[0]], dtype="U32")
        geometry[origin] = {
            "query_rows": query_rows,
            "donor_rows": donor_rows,
            "query_ids": ids[query_rows].astype(str),
            "query_cycles": cycles[query_rows],
            "query_impact_load": impacts[query_rows],
            "donor_ids": ids[donor_rows].astype(str),
            "donor_cycles": cycles[donor_rows],
            "donor_impact_load": impacts[donor_rows],
            "query_stream_missing": ~present[query_rows],
            "query_impact_missing": ~np.isfinite(impacts[query_rows]),
            "query_effective_missing": (~present[query_rows]) | (~np.isfinite(impacts[query_rows])),
            "neighbor_rows": neighbor_rows,
            "neighbor_ids": neighbor_ids,
            "neighbor_count": np.ones(len(query_rows), dtype=np.int64),
        }
    return geometry, train, impacts


def test_row_order_aware_view_guard_accepts_original_train_order_and_rejects_future_cycle():
    geometry, train, impacts = _unsorted_geometry_and_train()
    validate_view_cycles_row_order_aware(geometry)
    assert not np.all(geometry[67]["query_cycles"][:-1] <= geometry[67]["query_cycles"][1:])

    future = {origin: dict(arrays) for origin, arrays in geometry.items()}
    future[67]["query_cycles"] = geometry[67]["query_cycles"].copy()
    future[67]["query_cycles"][0] = 80
    with pytest.raises(PostfitVerificationError):
        validate_view_cycles_row_order_aware(future)

    duplicate = {origin: dict(arrays) for origin, arrays in geometry.items()}
    duplicate[67]["query_rows"] = geometry[67]["query_rows"].copy()
    duplicate[67]["query_rows"][0] = duplicate[67]["query_rows"][1]
    with pytest.raises(PostfitVerificationError):
        validate_geometry_row_identity(duplicate, train, impacts)


def test_geometry_row_identity_rejects_swapped_query_and_neighbor_ids():
    geometry, train, impacts = _unsorted_geometry_and_train()
    report = validate_geometry_row_identity(geometry, train, impacts)
    assert report["status"] == "PASS"
    assert report["neighbor_id_to_donor_row_links_checked"] > 0

    swapped_query = {origin: dict(arrays) for origin, arrays in geometry.items()}
    swapped_query[67]["query_ids"] = geometry[67]["query_ids"].copy()
    swapped_query[67]["query_ids"][[0, 1]] = swapped_query[67]["query_ids"][[1, 0]]
    with pytest.raises(PostfitVerificationError):
        validate_geometry_row_identity(swapped_query, train, impacts)

    swapped_neighbor = {origin: dict(arrays) for origin, arrays in geometry.items()}
    swapped_neighbor[67]["neighbor_ids"] = geometry[67]["neighbor_ids"].copy()
    swapped_neighbor[67]["neighbor_ids"][0, 0] = "wrong-donor"
    with pytest.raises(PostfitVerificationError):
        validate_geometry_row_identity(swapped_neighbor, train, impacts)


def test_numpy_compatibility_compares_ids_and_preserves_numeric_nan_semantics():
    equal = _NumpyArrayEqualCompatibility.array_equal
    assert equal(np.asarray(["A", "B"]), np.asarray(["A", "B"]), equal_nan=True)
    assert not equal(np.asarray(["A", "B"]), np.asarray(["B", "A"]), equal_nan=True)
    assert equal(np.asarray([np.nan]), np.asarray([np.nan]), equal_nan=True)
    assert not equal(np.asarray([np.nan]), np.asarray([np.nan]), equal_nan=False)
