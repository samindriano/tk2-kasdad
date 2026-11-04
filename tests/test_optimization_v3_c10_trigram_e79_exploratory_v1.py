from __future__ import annotations

import hashlib

import pytest

from scripts.optimization_v3_c10_trigram_e79_exploratory_v1 import (
    BenchmarkError,
    FIT_ID,
    _require_file_sha,
    validate_budget_events,
)


def _event(kind: str, **kwargs: object) -> dict[str, object]:
    return {"event": kind, "fit_id": FIT_ID, **kwargs}


def test_pinned_file_hash_accepts_exact_bytes_and_rejects_true_mutation(tmp_path) -> None:
    path = tmp_path / "historical.json"
    original = b'{"status":"frozen"}\n'
    expected = hashlib.sha256(original).hexdigest()
    path.write_bytes(original)
    assert _require_file_sha(path, expected, "test artifact") == expected

    path.write_bytes(b'{"status":"changed"}\n')
    with pytest.raises(BenchmarkError, match="hash mismatch"):
        _require_file_sha(path, expected, "test artifact")


def test_isolated_ledger_accepts_one_reserved_armed_saved_fit() -> None:
    events = [
        _event("FIT_RESERVED", units=6),
        _event("FIT_ARMED"),
        _event("FIT_STARTED"),
        _event("FIT_SAVED", units=6),
    ]
    validate_budget_events(events)


@pytest.mark.parametrize("events", [
    [_event("FIT_ARMED"), _event("FIT_RESERVED", units=6)],
    [_event("FIT_RESERVED", units=5)],
    [_event("FIT_RESERVED", units=6), _event("FIT_ARMED"), _event("FIT_STARTED"),
     _event("FIT_ARMED")],
    [_event("FIT_RESERVED", units=6), _event("FIT_ARMED"), _event("FIT_STARTED"),
     _event("FIT_SAVED", units=6), _event("FIT_STARTED")],
    [_event("FIT_RESERVED", units=6), _event("FIT_ARMED"), _event("FIT_STARTED"),
     _event("FIT_FAILED", retry_count=1)],
    [_event("FIT_RESERVED", units=6), {"event": "FIT_ARMED", "fit_id": "OTHER"}],
])
def test_isolated_ledger_rejects_retry_foreign_id_or_bad_order(events: list[dict[str, object]]) -> None:
    with pytest.raises(BenchmarkError):
        validate_budget_events(events)
