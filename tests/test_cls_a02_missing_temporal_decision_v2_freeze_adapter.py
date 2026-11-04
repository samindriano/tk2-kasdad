from __future__ import annotations

import copy
import inspect

import pytest

from scripts import apply_cls_a02_missing_temporal_decision_v2_freeze_v2 as adapter


def _parent_and_successor() -> tuple[dict[str, object], dict[str, object], dict[str, str]]:
    parent_inputs = {
        "experiments/runs/optimization_v3_a02_cgeom_benchmark_v1/test_predictions.npz": "1" * 64,
        "data/raw/classification/Test.csv": "2" * 64,
        "submissions/optimization_v3/cls_mix50_a02_missing_e79_v1/classification_mix50_present_a02_missing_e79.csv": "3" * 64,
        "reports/optimization_v3/cls_mix50_a02_missing_e79_v1/MANIFEST.json": "4" * 64,
        "reports/optimization_v3/cls_mix50_a02_missing_e79_v1/QA.json": "5" * 64,
    }
    selection_inputs = {
        "data/raw/classification/Train.csv": "6" * 64,
        "experiments/runs/optimization_v3_posoffset_a02_endgame_v1/policy_predictions.npz": "7" * 64,
    }
    parent: dict[str, object] = {
        "schema_version": 1,
        "policy_id": "CLS_A02_MISSING_TEMPORAL_DECISION_V1",
        "status": "FROZEN_BEFORE_E79_PROBABILITY_BUNDLE_ACCESS",
        "source_commit": "4a9cf4cfbc0b3030473733507841238d958ad92e",
        "final_selector_threshold": 0.53230976129477,
        "final_selector_receipt_sha256": "a" * 64,
        "prefit_freeze_sha256": "b" * 64,
        "formula": "present=current best MIX50 >=0.50; missing=original A02 pA >= final selector threshold",
        "threshold_operator": ">=",
        "application_input_hashes": parent_inputs,
        "test_labels_read": False,
        "sample_submission_read": False,
        "new_model_inference": 0,
        "new_estimator_fits": 0,
        "kaggle_upload": False,
    }
    successor = copy.deepcopy(parent)
    successor.update({
        "freeze_version": 2,
        "parent_application_freeze_path": adapter.ORIGINAL_FREEZE.relative_to(adapter.ROOT).as_posix(),
        "parent_application_freeze_sha256": adapter.ORIGINAL_FREEZE_SHA256,
        "application_input_hashes": {**parent_inputs, **selection_inputs},
        "freeze_amendment": {
            "threshold_unchanged": True,
            "final_selector_receipt_unchanged": True,
            "official_selectors_rerun": False,
            "E79_probability_arrays_read_before_successor_commit": False,
            "Test_content_read_before_successor_commit": False,
            "model_fits_or_inference": 0,
        },
        "canonical_sha256": "c" * 64,
    })
    return parent, successor, selection_inputs


def test_successor_binds_exactly_five_parent_plus_two_selection_hashes() -> None:
    parent, successor, selection_inputs = _parent_and_successor()
    result = adapter.validate_successor_payload(
        parent,
        successor,
        selection_inputs,
        parent_file_sha256=adapter.ORIGINAL_FREEZE_SHA256,
    )
    assert len(result) == 7
    assert set(selection_inputs).issubset(result)


def test_five_key_parent_alone_fails_the_seven_input_contract() -> None:
    parent, _successor, selection_inputs = _parent_and_successor()
    with pytest.raises(RuntimeError, match="version 2"):
        adapter.validate_successor_payload(
            parent,
            parent,
            selection_inputs,
            parent_file_sha256=adapter.ORIGINAL_FREEZE_SHA256,
        )


def test_moved_or_tampered_hash_and_threshold_are_rejected() -> None:
    parent, successor, selection_inputs = _parent_and_successor()
    moved = copy.deepcopy(successor)
    moved["application_input_hashes"]["data/raw/classification/train.csv"] = moved["application_input_hashes"].pop("data/raw/classification/Train.csv")
    with pytest.raises(RuntimeError, match="input map"):
        adapter.validate_successor_payload(parent, moved, selection_inputs, parent_file_sha256=adapter.ORIGINAL_FREEZE_SHA256)

    tampered = copy.deepcopy(successor)
    tampered["application_input_hashes"]["data/raw/classification/Train.csv"] = "f" * 64
    with pytest.raises(RuntimeError, match="input map"):
        adapter.validate_successor_payload(parent, tampered, selection_inputs, parent_file_sha256=adapter.ORIGINAL_FREEZE_SHA256)

    threshold_changed = copy.deepcopy(successor)
    threshold_changed["final_selector_threshold"] = 0.50
    with pytest.raises(RuntimeError, match="changed frozen decision field"):
        adapter.validate_successor_payload(parent, threshold_changed, selection_inputs, parent_file_sha256=adapter.ORIGINAL_FREEZE_SHA256)


def test_adapter_build_does_not_rerun_threshold_selectors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(adapter, "verify_successor_guard", lambda **_kwargs: {"status": "PASS"})
    monkeypatch.setattr(adapter.decision, "build_e79_candidate", lambda: {"status": "synthetic build"})
    monkeypatch.setattr(adapter.decision, "run_selectors", lambda: (_ for _ in ()).throw(AssertionError("selector rerun")))
    assert adapter.build_candidate()["status"] == "synthetic build"
    assert "run_selectors" not in inspect.getsource(adapter.build_candidate)
