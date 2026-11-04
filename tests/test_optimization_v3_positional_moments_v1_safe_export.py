from __future__ import annotations

import hashlib
import io
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.optimization_v3_positional_moments_v1_safe_export import (
    SAFE_BUNDLE_KEYS,
    SafeExportError,
    _assert_copied_member_identity,
    _write_safe_npz,
    _validate_rows_and_ids,
)
from src.models.positional_moments_v1 import sha256_array


def _npy_bytes(array: np.ndarray, *, allow_pickle: bool = False) -> bytes:
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=allow_pickle)
    return stream.getvalue()


def _make_original(path: Path, ids: list[str]) -> dict[str, bytes]:
    rows = np.arange(len(ids), dtype=np.int64)
    arrays: dict[str, np.ndarray] = {
        "score_rows": rows,
        "score_record_ids": np.asarray(ids, dtype=object),
        "score_cycles": np.arange(60, 60 + len(ids), dtype=np.int64),
        "present_mask": np.ones(len(ids), dtype=bool),
        "entry_friction": np.zeros(len(ids), dtype=np.int8),
        "text_load_group": np.ones(len(ids), dtype=np.int8),
        "prob_pP": np.full(len(ids), 0.4, dtype=np.float64),
        "prob_a02": np.full(len(ids), 0.5, dtype=np.float64),
        "prob_bag": np.full(len(ids), 0.45, dtype=np.float64),
        "prob_q_real": np.full(len(ids), 0.52, dtype=np.float64),
        "prob_candidate_real": np.full(len(ids), 0.51, dtype=np.float64),
        "fit_receipt_sha256_real": np.asarray(["a" * 64], dtype="U64"),
        "prob_q_shuffled": np.full(len(ids), 0.49, dtype=np.float64),
        "prob_candidate_shuffled": np.full(len(ids), 0.495, dtype=np.float64),
        "fit_receipt_sha256_shuffled": np.asarray(["b" * 64], dtype="U64"),
        "prob_mix50": np.full(len(ids), 0.475, dtype=np.float64),
    }
    raw_members = {f"{key}.npy": _npy_bytes(value, allow_pickle=value.dtype.kind == "O")
                   for key, value in arrays.items()}
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for key in SAFE_BUNDLE_KEYS:
            archive.writestr(f"{key}.npy", raw_members[f"{key}.npy"])
    return raw_members


def test_safe_export_copies_numeric_members_and_reads_unicode_long_ids(tmp_path: Path) -> None:
    ids = ["漏洞-α", "case-" + ("长" * 300) + "-🛡️", "Ω-record-03"]
    source = tmp_path / "original.npz"
    output = tmp_path / "safe.npz"
    originals = _make_original(source, ids)
    replacement = np.asarray(ids, dtype=np.str_)
    result = _write_safe_npz(source, output, replacement)

    assert result["members"]["score_record_ids.npy"]["original_member_sha256"] is None
    with zipfile.ZipFile(source, "r") as source_zip, zipfile.ZipFile(output, "r") as output_zip:
        for key in SAFE_BUNDLE_KEYS:
            name = f"{key}.npy"
            if key != "score_record_ids":
                assert source_zip.read(name) == output_zip.read(name) == originals[name]
            else:
                assert result["members"][name]["original_member_sha256"] is None
    with np.load(output, allow_pickle=False) as safe:
        assert safe["score_record_ids"].dtype.kind == "U"
        assert safe["score_record_ids"].tolist() == ids
        assert safe["score_rows"].tolist() == [0, 1, 2]
    assert result["members"]["score_rows.npy"]["byte_identical"] is True


def test_copied_member_tampering_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "original.npz"
    output = tmp_path / "safe.npz"
    _make_original(source, ["id-0", "id-1"])
    result = _write_safe_npz(source, output, np.asarray(["id-0", "id-1"], dtype=np.str_))
    record = result["members"]["score_rows.npy"]
    with zipfile.ZipFile(source, "r") as archive:
        source_raw = archive.read("score_rows.npy")
    with zipfile.ZipFile(output, "r") as archive:
        derived_raw = archive.read("score_rows.npy")
    _assert_copied_member_identity("score_rows.npy", source_raw, derived_raw, record)
    tampered = bytearray(derived_raw)
    tampered[-1] ^= 1
    with pytest.raises(SafeExportError, match="byte identity failed"):
        _assert_copied_member_identity("score_rows.npy", source_raw, bytes(tampered), record)


def _identity_case(ids: list[str], *, rows: list[int] | None = None,
                   declared_hash: str | None = None, parent_ids: list[str] | None = None,
                   parent_rows: list[int] | None = None) -> tuple[np.ndarray, dict[str, object]]:
    all_ids = np.asarray(ids, dtype=np.str_)
    all_cycles = np.asarray([47, 48, 49, 50, 51, 52], dtype=np.int64)[:len(ids)]
    selected = np.asarray(rows if rows is not None else [2, 3, 4, 5], dtype=np.int64)
    selected_cycles = all_cycles[selected]
    selected_ids = all_ids[selected]
    old_hash = declared_hash or sha256_array(np.asarray(selected_ids, dtype=object))
    output = _validate_rows_and_ids(
        origin=48, score_rows=selected, score_cycles=selected_cycles,
        all_ids=all_ids, all_cycles=all_cycles, declared_old_ids_sha256=old_hash,
        expected_rows=np.asarray([2, 3, 4, 5], dtype=np.int64),
        parent_rows=np.asarray(parent_rows if parent_rows is not None else selected),
        parent_cycles=selected_cycles,
        parent_ids=np.asarray(parent_ids if parent_ids is not None else selected_ids),
    )
    return output


def test_target_free_ids_match_original_object_hash_and_parent_rows() -> None:
    ids = ["r0", "r1", "漏洞-α", "record-" + "长" * 300, "r4", "🛡️-r5"]
    unicode_ids, audit = _identity_case(ids)
    assert unicode_ids.dtype.kind == "U"
    assert unicode_ids.tolist() == ids[2:]
    assert audit["old_dtype"] == "object"
    assert audit["old_dtype_hash_matches"] is True
    assert audit["new_dtype"].startswith("<U")


def test_identity_rejects_wrong_rows_or_parent_identity() -> None:
    ids = ["r0", "r1", "r2", "r3", "r4", "r5"]
    with pytest.raises(SafeExportError, match="score_rows differ"):
        _identity_case(ids, rows=[1, 3, 4, 5])
    with pytest.raises(SafeExportError, match="parent rows/cycles/IDs"):
        _identity_case(ids, parent_rows=[2, 3, 4, 4])
    with pytest.raises(SafeExportError, match="parent rows/cycles/IDs"):
        _identity_case(ids, parent_ids=["wrong", "r3", "r4", "r5"])


def test_identity_rejects_wrong_original_hash() -> None:
    with pytest.raises(SafeExportError, match="logical hash differs"):
        _identity_case(["r0", "r1", "r2", "r3", "r4", "r5"], declared_hash="0" * 64)


@pytest.mark.parametrize("bad_ids", [
    ["r0", "r1", "same", "same", "r4", "r5"],
    ["r0", "r1", "", "r3", "r4", "r5"],
])
def test_identity_rejects_duplicate_or_missing_train_ids(bad_ids: list[str]) -> None:
    with pytest.raises(SafeExportError, match="missing or duplicated"):
        _identity_case(bad_ids)

