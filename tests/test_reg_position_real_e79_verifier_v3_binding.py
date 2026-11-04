from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.reg_position_real_e79_verifier_v3_binding import (
    ORIGINAL_V2_ADAPTER_FILES,
    ORIGINAL_V2_COMMIT,
    canonical_sha256,
    validate_historical_adapter_commit,
    validate_manifest_integrity,
    validate_source_hash_binding,
)


INVENTORY = {
    "scripts/example.py": "1" * 64,
    "src/example.py": "2" * 64,
}


def _valid_source_binding() -> dict:
    return validate_source_hash_binding(
        INVENTORY, dict(INVENTORY), dict(INVENTORY),
        {key: True for key in INVENTORY}, dict(INVENTORY),
    )


def test_source_binding_keeps_pinned_hashes_separate_from_boolean_results() -> None:
    result = _valid_source_binding()
    assert result["pinned_inventory"] == INVENTORY
    assert result["observed_boolean_results"] == {
        "scripts/example.py": True, "src/example.py": True,
    }
    assert all(result["checks"].values())


@pytest.mark.parametrize(
    "result_map",
    [
        {"scripts/example.py": True},
        {"scripts/example.py": True, "src/example.py": True, "extra.py": True},
        {"scripts/example.py": 1, "src/example.py": True},
        {"scripts/example.py": True, "src/example.py": False},
    ],
)
def test_source_binding_rejects_missing_extra_wrong_type_or_false_results(result_map: dict) -> None:
    with pytest.raises(ValueError):
        validate_source_hash_binding(
            INVENTORY, dict(INVENTORY), dict(INVENTORY), result_map, dict(INVENTORY),
        )


def test_source_binding_rejects_true_results_when_actual_source_is_tampered() -> None:
    actual = dict(INVENTORY)
    actual["src/example.py"] = "3" * 64
    with pytest.raises(ValueError, match="actual_file_hashes_exact"):
        validate_source_hash_binding(
            INVENTORY, dict(INVENTORY), dict(INVENTORY),
            {key: True for key in INVENTORY}, actual,
        )


def test_manifest_validation_rejects_wrong_signature_or_file_hash() -> None:
    manifest = {"status": "PREPARED_NOT_UPLOADED", "rows": 3}
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    file_hash = hashlib.sha256(b"manifest bytes").hexdigest()
    valid = validate_manifest_integrity(
        manifest, file_hash, manifest["manifest_sha256"], file_hash,
    )
    assert all(valid["checks"].values())

    altered = dict(manifest, status="OTHER")
    with pytest.raises(ValueError):
        validate_manifest_integrity(
            altered, file_hash, manifest["manifest_sha256"], file_hash,
        )
    with pytest.raises(ValueError):
        validate_manifest_integrity(
            manifest, "0" * 64, manifest["manifest_sha256"], file_hash,
        )


def test_historical_adapter_commit_requires_ancestry_and_matching_blobs() -> None:
    blobs = {relative: f"content:{relative}".encode() for relative in ORIGINAL_V2_ADAPTER_FILES}
    hashes = {
        relative: hashlib.sha256(payload).hexdigest()
        for relative, payload in blobs.items()
    }
    result = validate_historical_adapter_commit(
        ORIGINAL_V2_COMMIT, "a" * 40, hashes,
        lambda ancestor, head: ancestor == ORIGINAL_V2_COMMIT and head == "a" * 40,
        lambda commit, path: blobs[path],
    )
    assert result["recorded_commit_is_ancestor"] is True
    assert result["git_blob_hashes"] == hashes
    with pytest.raises(ValueError, match="not an ancestor"):
        validate_historical_adapter_commit(
            ORIGINAL_V2_COMMIT, "a" * 40, hashes, lambda _a, _b: False,
            lambda _commit, path: blobs[path],
        )
    with pytest.raises(ValueError, match="Git blob differs"):
        validate_historical_adapter_commit(
            ORIGINAL_V2_COMMIT, "a" * 40, hashes, lambda _a, _b: True,
            lambda _commit, path: b"tampered" if path == ORIGINAL_V2_ADAPTER_FILES[0] else blobs[path],
        )


def test_submission_csv_round_trip_parser_recovers_exact_float64(tmp_path: Path) -> None:
    expected = np.asarray([34.996797583053876, 52.125, 0.0], dtype=np.float64)
    path = tmp_path / "submission.csv"
    pd.DataFrame({
        "record_id": ["a", "b", "c"],
        "threat_rank": expected,
    }).to_csv(path, index=False, float_format="%.17g", lineterminator="\n")

    default = pd.read_csv(path).threat_rank.to_numpy(dtype=np.float64)
    round_trip = pd.read_csv(path, float_precision="round_trip").threat_rank.to_numpy(
        dtype=np.float64,
    )

    assert not np.array_equal(default, expected)
    assert np.array_equal(round_trip, expected)
