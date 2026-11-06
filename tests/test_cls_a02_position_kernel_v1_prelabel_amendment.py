from __future__ import annotations

import numpy as np
import pytest

from scripts.verify_cls_a02_position_kernel_v1_prelabel_amended_01 import (
    PrelabelError,
    _producer_canonical_ids,
    _validate_parent_route,
)


def fixture_arrays():
    a02 = {
        "score_rows": np.array([10, 11, 12], dtype=np.int64),
        "score_record_ids": np.array(["a", "b", "c"]),
        "score_cycles": np.array([68, 69, 69], dtype=np.int64),
        "missing_mask": np.array([False, False, True]),
        "prediction": np.array([0.3, 0.6, 0.7], dtype=np.float64),
    }
    offset = {
        "score_rows": a02["score_rows"].copy(),
        "score_record_ids": a02["score_record_ids"].copy(),
        "score_cycles": a02["score_cycles"].copy(),
        "present_mask": np.array([True, True, False]),
        "prob_offset_real": np.array([0.2, 0.8, 0.4], dtype=np.float64),
    }
    p_pos = np.array([0.2, 0.8, 0.4], dtype=np.float64)
    p_champion = np.array([0.25, 0.7, 0.7], dtype=np.float64)
    parent = {
        "origin": np.array([67], dtype=np.int64),
        "row_positions": a02["score_rows"].copy(),
        "record_ids": a02["score_record_ids"].copy(),
        "cycles": a02["score_cycles"].copy(),
        "present_mask": offset["present_mask"].copy(),
        "p_a02": a02["prediction"].copy(),
        "p_pos_original": p_pos,
        "p_champion": p_champion,
        "label_champion": np.array([0, 1, 1], dtype=np.int8),
        "matrix_manifest_sha256": np.array(["matrix-sha"]),
        "input_freeze_signature": np.array(["freeze-sig"]),
    }
    return a02, offset, parent


def validate(a02, offset, parent):
    return _validate_parent_route(
        a02, offset, parent, origin=67, freeze_signature="freeze-sig",
        matrix_manifest_sha256="matrix-sha",
    )


def test_correct_original_pos_offset_parent_is_accepted_bit_exactly():
    a02, offset, parent = fixture_arrays()
    result = validate(a02, offset, parent)
    assert result["pos_offset_original_present_exact"] is True
    assert result["a02_original_exact"] is True
    assert result["full_champion_probability_bit_exact"] is True
    assert result["full_champion_decisions_exact"] is True


def test_old_joint_pos_probability_is_rejected_as_wrong_parent():
    a02, offset, parent = fixture_arrays()
    old_joint_prob_position = np.array([0.9, 0.1, 0.4], dtype=np.float64)
    parent["p_pos_original"] = old_joint_prob_position
    with pytest.raises(PrelabelError, match="POS offset route"):
        validate(a02, offset, parent)


@pytest.mark.parametrize(
    ("field", "mutate"),
    [
        ("p_a02", lambda a02, offset, parent: parent["p_a02"].__setitem__(0, 0.31)),
        ("p_champion", lambda a02, offset, parent: parent["p_champion"].__setitem__(0, 0.2500000001)),
        ("label_champion", lambda a02, offset, parent: parent["label_champion"].__setitem__(0, 1)),
        ("matrix_manifest_sha256", lambda a02, offset, parent: parent["matrix_manifest_sha256"].__setitem__(0, "tampered")),
        ("input_freeze_signature", lambda a02, offset, parent: parent["input_freeze_signature"].__setitem__(0, "tampered")),
    ],
)
def test_parent_payload_tampering_is_rejected(field, mutate):
    a02, offset, parent = fixture_arrays()
    mutate(a02, offset, parent)
    with pytest.raises(PrelabelError):
        validate(a02, offset, parent)


@pytest.mark.parametrize(
    ("field", "mutate"),
    [
        ("score_rows", lambda a02, offset, parent: offset["score_rows"].__setitem__(0, 11)),
        ("score_record_ids", lambda a02, offset, parent: offset["score_record_ids"].__setitem__(0, "wrong")),
        ("score_cycles", lambda a02, offset, parent: offset["score_cycles"].__setitem__(0, 69)),
        ("present_mask", lambda a02, offset, parent: offset["present_mask"].__setitem__(2, True)),
    ],
)
def test_identity_cycle_and_missing_support_mismatches_are_rejected(field, mutate):
    a02, offset, parent = fixture_arrays()
    mutate(a02, offset, parent)
    with pytest.raises(PrelabelError):
        validate(a02, offset, parent)


@pytest.mark.parametrize(
    ("field", "mutate"),
    [
        ("row_positions", lambda a02, offset, parent: parent["row_positions"].__setitem__(0, 11)),
        ("record_ids", lambda a02, offset, parent: parent["record_ids"].__setitem__(0, "wrong")),
        ("cycles", lambda a02, offset, parent: parent["cycles"].__setitem__(0, 69)),
        ("present_mask", lambda a02, offset, parent: parent["present_mask"].__setitem__(2, True)),
    ],
)
def test_parent_identity_cycle_and_missing_support_mismatches_are_rejected(field, mutate):
    a02, offset, parent = fixture_arrays()
    mutate(a02, offset, parent)
    with pytest.raises(PrelabelError):
        validate(a02, offset, parent)


def test_parent_required_member_missing_is_rejected():
    a02, offset, parent = fixture_arrays()
    del parent["p_pos_original"]
    with pytest.raises(KeyError):
        validate(a02, offset, parent)


def test_numpy_producer_id_tags_match_a02_bank_identity():
    assert _producer_canonical_ids(["one", "two"]) == ("str_:one", "str_:two")
