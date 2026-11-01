from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from contextlib import nullcontext

import scripts.prepare_c16_r22_fulltrain_submissions as preparation
from scripts.prepare_c16_r22_fulltrain_submissions import (
    _r22_resource_preflight,
    _source_hashes,
    _verify_classification_reuse,
    _verify_continuation_verdicts,
    _load_inputs,
    _submission_frame,
    _ids_sha,
    _training_slot,
    route_c16_probabilities,
    r22_original_prediction,
)


def test_ids_sha_uses_ordered_lines() -> None:
    assert _ids_sha(["a", "b"]) != _ids_sha(["b", "a"])


def test_c16_availability_route_keeps_exact_missing_fallback() -> None:
    result = route_c16_probabilities(
        c2b=[0.2, 0.4], cb_missing=[0.7, 0.9], c16=[0.8, 0.6],
        missing_mask=[False, True],
    )
    np.testing.assert_allclose(result, [0.35, 0.9], rtol=0.0, atol=1e-15)


def test_r22_original_uses_full_cycle_average_ties() -> None:
    result, component = r22_original_prediction(
        raw=[10.0, 10.0, 20.0], champion=[0.0, 0.0, 0.0], cycles=[80, 80, 80]
    )
    np.testing.assert_allclose(component, [30.0, 30.0, 60.0])
    np.testing.assert_allclose(result, [7.5, 7.5, 15.0])


def test_r22_resource_preflight_blocks_low_available_ram() -> None:
    assert _r22_resource_preflight(8 * 1024**3)["pass"] is True
    assert _r22_resource_preflight(4 * 1024**3)["pass"] is False


def test_training_slot_blocks_when_fresh_locked_ram_check_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(preparation, "exclusive_training_slot", lambda _path: nullcontext())
    monkeypatch.setattr(preparation, "_resource_preflight", lambda _task: {"pass": False})
    with pytest.raises(RuntimeError, match="fresh in-lock RAM check"):
        with _training_slot("classification"):
            pytest.fail("fit body must not be entered after failed RAM gate")


def test_freeze_covers_direct_recipe_and_input_resolution_dependencies() -> None:
    hashes = _source_hashes()
    for required in [
        "src/models/optimization_v3_sparse_classification.py",
        "src/models/round1_models.py",
        "src/data/audit.py",
        "src/utils/paths.py",
    ]:
        assert required in hashes


def test_full_train_classification_reuse_attestation_matches_official_data() -> None:
    train, test, _sample, _hashes = _load_inputs("classification")
    _base, _missing, _mask, evidence = _verify_classification_reuse(train, test)
    assert evidence["reused_models"]["C2_B"]["fit_rows"] == 50_608
    assert evidence["reused_models"]["CB_MISSING"]["fit_rows"] == 50_608
    assert evidence["source_code_sha256_verified_count"] == 58


def test_continuation_gate_verdicts_are_pinned() -> None:
    evidence = _verify_continuation_verdicts()
    assert evidence["C16"]["qualified"] is False
    assert evidence["R22"]["selected_policy"] == "R22_ORIGINAL"
    assert evidence["R22"]["qualified"] is True


def test_submission_frame_preserves_sample_ids_and_integer_labels() -> None:
    test = pd.DataFrame({"record_id": [10, 11]})
    sample = pd.DataFrame({"record_id": [10, 11], "artifact_signal": [0, 0]})
    result = _submission_frame(test, sample, np.array([0, 1]), "artifact_signal", "classification")
    assert list(result.columns) == ["record_id", "artifact_signal"]
    assert result.artifact_signal.dtype.kind in "iu"
    assert result.record_id.tolist() == [10, 11]


def test_submission_frame_rejects_nonbinary_classification() -> None:
    test = pd.DataFrame({"record_id": [10]})
    sample = pd.DataFrame({"record_id": [10], "artifact_signal": [0]})
    with pytest.raises(ValueError):
        _submission_frame(test, sample, np.array([0.5]), "artifact_signal", "classification")
