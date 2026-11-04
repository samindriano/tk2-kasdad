from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from scripts.recover_optimization_v3_c10_trigram_e79_same_fit import (
    BenchmarkError,
    durable_copy_verified,
)


def test_windows_read_only_fsync_failure_phase_is_reproduced(tmp_path: Path) -> None:
    path = tmp_path / "model.partial"
    path.write_bytes(b"completed model bytes")
    with path.open("rb") as stream:
        with pytest.raises(OSError):
            os.fsync(stream.fileno())


def test_durable_copy_preserves_exact_model_bytes_and_source(tmp_path: Path) -> None:
    source = tmp_path / "model.partial"
    destination = tmp_path / "model.joblib"
    payload = bytes(range(256)) * 97
    source.write_bytes(payload)

    expected = hashlib.sha256(payload).hexdigest()
    assert durable_copy_verified(source, destination) == expected
    assert source.read_bytes() == payload
    assert destination.read_bytes() == payload
    assert hashlib.sha256(destination.read_bytes()).hexdigest() == expected


def test_durable_copy_refuses_existing_destination(tmp_path: Path) -> None:
    source = tmp_path / "model.partial"
    destination = tmp_path / "model.joblib"
    source.write_bytes(b"same fit")
    destination.write_bytes(b"keep")
    with pytest.raises(BenchmarkError, match="already exists"):
        durable_copy_verified(source, destination)
    assert destination.read_bytes() == b"keep"
