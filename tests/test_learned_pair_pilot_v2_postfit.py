from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import optimization_v3_learned_pair_pilot_v2_postfit as postfit
import verify_optimization_v3_learned_pair_pilot_v2_postfit as independent


def _old_archive(path: Path, *, bad_extra_object: bool = False) -> dict[str, np.ndarray]:
    arrays = {
        "score_rows": np.asarray([10, 14, 18], dtype=np.int64),
        "score_record_ids": np.asarray(["unreadable-old-a", "unreadable-old-b", "unreadable-old-c"], dtype=object),
        "score_cycles": np.asarray([50, 51, 52], dtype=np.int64),
        "probability": np.asarray([0.1, 0.5, 0.9], dtype=np.float64),
        "raw_logits": np.asarray([-2.0, 0.0, 2.0], dtype=np.float64),
    }
    if bad_extra_object:
        arrays["extra_object"] = np.asarray(["forbidden"], dtype=object)
    np.savez_compressed(path, **arrays)
    return arrays


def _unicode_ids() -> np.ndarray:
    return np.asarray(["rec-10", "record-longer-東京-Δ", "rec-18"], dtype="<U40")


def test_rebuild_rejects_object_ids_without_unpickling_and_preserves_all_safe_members(tmp_path: Path) -> None:
    old = tmp_path / "old.npz"
    derived = tmp_path / "derived.npz"
    _old_archive(old)
    audit = postfit.rebuild_npz_preserving_members(old, derived, _unicode_ids())
    with pytest.raises(ValueError, match="Object arrays cannot be loaded"):
        postfit.read_safe_member(old, "score_record_ids")
    ids = postfit.read_safe_member(derived, "score_record_ids")
    assert ids.dtype.kind == "U"
    assert ids.tolist() == _unicode_ids().tolist()
    old_members, new_members = postfit.archive_members(old), postfit.archive_members(derived)
    assert list(old_members) == list(new_members)
    for name in old_members:
        if name != "score_record_ids.npy":
            assert old_members[name] == new_members[name]
            assert np.array_equal(postfit.read_safe_member(old, name.removesuffix(".npy")),
                                  postfit.read_safe_member(derived, name.removesuffix(".npy")))
    assert audit["changed_member"] == "score_record_ids.npy"
    assert audit["original_members"]["score_record_ids.npy"]["values_read"] is False


def test_unicode_roundtrip_uses_sufficient_fixed_width_without_truncation(tmp_path: Path) -> None:
    old, derived = tmp_path / "old.npz", tmp_path / "new.npz"
    _old_archive(old)
    long_id = "唯一識別子-" + "x" * 800
    ids = np.asarray(["é-1", long_id, "三"], dtype=f"<U{len(long_id)}")
    postfit.rebuild_npz_preserving_members(old, derived, ids)
    restored = postfit.read_safe_member(derived, "score_record_ids")
    assert restored.dtype.kind == "U"
    assert restored.dtype.itemsize // np.dtype("U1").itemsize >= len(long_id)
    assert restored.tolist() == ids.tolist()


def test_rebuild_rejects_object_member_outside_score_ids(tmp_path: Path) -> None:
    old, derived = tmp_path / "old.npz", tmp_path / "new.npz"
    _old_archive(old, bad_extra_object=True)
    with pytest.raises(postfit.PostfitError, match="outside the authorized score ID"):
        postfit.rebuild_npz_preserving_members(old, derived, _unicode_ids())


def test_rebuild_rejects_wrong_shape_and_non_unicode_replacement(tmp_path: Path) -> None:
    old = tmp_path / "old.npz"
    _old_archive(old)
    with pytest.raises(postfit.PostfitError, match="expected object array/shape"):
        postfit.rebuild_npz_preserving_members(tmp_path / "old.npz", tmp_path / "wrong.npz", np.asarray(["x"]))
    with pytest.raises(postfit.PostfitError, match="Unicode array"):
        postfit.rebuild_npz_preserving_members(old, tmp_path / "wrong_dtype.npz", np.asarray([1, 2, 3]))


def test_identity_alignment_accepts_exact_rows_ids_cycles_and_rejects_each_tamper() -> None:
    rows = np.asarray([2, 5, 8], dtype=np.int64)
    ids = np.asarray(["r2", "r5", "r8"], dtype="<U2")
    cycles = np.asarray([50, 51, 52], dtype=np.int64)
    postfit.require_identity_alignment(rows, ids, cycles, rows, ids, cycles, label="synthetic")
    with pytest.raises(postfit.PostfitError, match="identity alignment"):
        postfit.require_identity_alignment(rows, ids, cycles, rows + 1, ids, cycles, label="row-tamper")
    with pytest.raises(postfit.PostfitError, match="identity alignment"):
        postfit.require_identity_alignment(rows, np.asarray(["wrong", "r5", "r8"]), cycles,
                                           rows, ids, cycles, label="id-tamper")
    with pytest.raises(postfit.PostfitError, match="identity alignment"):
        postfit.require_identity_alignment(rows, ids, cycles + 1, rows, ids, cycles, label="cycle-tamper")


def test_independent_member_pair_rejects_numeric_and_id_tampering(tmp_path: Path) -> None:
    old = tmp_path / "old.npz"
    derived = tmp_path / "derived.npz"
    _old_archive(old)
    postfit.rebuild_npz_preserving_members(old, derived, _unicode_ids())
    old_members = independent._archive_members(old)
    clean_members = independent._archive_members(derived)
    independent._verify_member_pair(old_members, clean_members)

    tampered_numeric = tmp_path / "tampered_numeric.npz"
    values = {name.removesuffix(".npy"): postfit.read_safe_member(derived, name.removesuffix(".npy"))
              for name in clean_members}
    values["probability"][0] = 0.123
    np.savez_compressed(tampered_numeric, **values)
    with pytest.raises(independent.VerificationError, match="bytes changed"):
        independent._verify_member_pair(old_members, independent._archive_members(tampered_numeric))

    tampered_id = tmp_path / "tampered_id.npz"
    values["probability"] = postfit.read_safe_member(derived, "probability")
    values["score_record_ids"][0] = "modified"
    np.savez_compressed(tampered_id, **values)
    # The member pair permits one altered identity member, but the frozen identity vector must reject it.
    independent._verify_member_pair(old_members, independent._archive_members(tampered_id))
    with pytest.raises(independent.VerificationError, match="identity mismatch"):
        independent._verify_identity_vectors(np.asarray([10, 14, 18]), values["score_record_ids"],
                                             np.asarray([50, 51, 52]), np.asarray([10, 14, 18]),
                                             _unicode_ids(), np.asarray([50, 51, 52]))


def test_amendment_receipt_signature_binds_original_and_derived_hashes() -> None:
    body = {"old_sha256": "a" * 64, "new_sha256": "b" * 64,
            "training_commit": postfit.EXPECTED_TRAINING_COMMIT,
            "amendment_commit": "d" * 40}
    receipt = {**body, "receipt_sha256": postfit.canonical_sha256(body)}
    verified = independent._verify_receipt_signature(receipt)
    assert verified == body
    for key in ("old_sha256", "new_sha256", "training_commit", "amendment_commit"):
        tampered = dict(receipt)
        tampered[key] = "f" * len(str(tampered[key]))
        with pytest.raises(independent.VerificationError, match="signature"):
            independent._verify_receipt_signature(tampered)


def test_synthetic_amendment_lifecycle_has_no_fit_or_live_ledger_write(tmp_path: Path) -> None:
    before = postfit.pilot._sha256_file(postfit.pilot.LEDGER)
    result = postfit.run_synthetic_qa(tmp_path / "lifecycle")
    after = postfit.pilot._sha256_file(postfit.pilot.LEDGER)
    assert result["status"] == "PASS_SYNTHETIC_POSTFIT_AMENDMENT_QA"
    assert result["fit_calls"] == 0
    assert result["production_ledger_writer_blocked"] is True
    assert result["production_ledger_sha256_before"] == result["production_ledger_sha256_after"] == before == after
    assert result["frozen_gate_parity_counts"] == {"E67": 40, "E75": 10}

\n