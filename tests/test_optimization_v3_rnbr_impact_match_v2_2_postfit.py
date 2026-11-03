import pytest

from scripts.optimization_v3_rnbr_impact_match_v2_2_postfit_verify import (
    EXPECTED_SNAPSHOT,
    FIT_SOURCE_COMMIT,
    PostfitVerificationError,
    validate_postfit_ledger,
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
