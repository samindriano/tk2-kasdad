from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from scripts import cls_dan_v1 as runner
from scripts import cls_dan_v1_postfit_repair_r1 as repair


def _views():
    first = {
        "score_rows": np.asarray([10, 11, 12, 13, 14, 15], dtype=np.int64),
        "score_record_ids": np.asarray([f"id-{i}" for i in range(6)]),
        "score_cycles": np.asarray([76, 76, 77, 77, 78, 79], dtype=np.int64),
        "present": np.asarray([True, True, False, True, True, False]),
        "champion": np.asarray([0.15, 0.25, 0.35, 0.45, 0.55, 0.65]),
    }
    second = {
        "score_rows": first["score_rows"][-4:].copy(),
        "score_record_ids": first["score_record_ids"][-4:].copy(),
        "score_cycles": first["score_cycles"][-4:].copy(),
        "present": first["present"][-4:].copy(),
        "champion": np.asarray([0.24, 0.36, 0.49, 0.63]),
    }
    return first, second


def test_overlapping_origin_predictions_can_differ_when_identity_matches():
    first, second = _views()
    result = repair.validate_origin_alignment(first, second)
    assert result["shared_query_rows"] == 4
    assert result["record_ids_exact"] and result["source_rows_exact"]
    assert result["cycles_exact"] and result["availability_exact"]
    assert result["champion_probabilities_required_equal"] is False
    assert result["champion_probabilities_different_rows"] == 4


def test_prelabel_champion_is_checked_against_its_own_origin_cache(tmp_path):
    views = _views()
    caches = {}
    payload = {
        "prelabel_metadata_json": np.asarray(json.dumps({
            "candidate_id": "CLS_DAN_V1",
            "freeze_signature": "SYNTHETIC_FREEZE",
            "fit_receipt_hashes": {fit_id: f"receipt-{fit_id}" for fit_id in runner.FIT_IDS},
            "fit_artifact_hashes": {fit_id: f"model-{fit_id}" for fit_id in runner.FIT_IDS},
            "evaluation_query_targets_read": False,
            "test_opened": False,
            "sample_submission_opened": False,
            "kaggle_contacted": False,
        }))
    }
    for origin, view in ((67, views[0]), (75, views[1])):
        cache = dict(view)
        cache["p_pos_original"] = view["champion"] * 0.9
        cache["p_a02"] = view["champion"] * 0.8
        cache["entry_friction"] = np.where(view["score_rows"] % 2 == 0, "low", "high")
        cache["text_load_group"] = np.where(view["score_rows"] % 3 == 0, "Q1", "Q2")
        cache["coverage"] = np.ones(len(view["score_rows"]), dtype=np.float64)
        present = view["present"]
        dan = view["champion"].copy()
        linear = view["champion"].copy()
        aux_dan = np.full(len(present), np.nan, dtype=np.float64)
        aux_linear = np.full(len(present), np.nan, dtype=np.float64)
        aux_dan[present] = 0.6
        aux_linear[present] = 0.4
        dan[present] = 0.75 * view["champion"][present] + 0.25 * aux_dan[present]
        linear[present] = 0.75 * view["champion"][present] + 0.25 * aux_linear[present]
        caches[origin] = cache
        prefix = f"E{origin}_"
        for key, value in view.items():
            payload[prefix + key] = value
        payload[prefix + "p_pos_original"] = cache["p_pos_original"]
        payload[prefix + "p_a02"] = cache["p_a02"]
        payload[prefix + "entry_friction"] = cache["entry_friction"]
        payload[prefix + "text_load_group"] = cache["text_load_group"]
        payload[prefix + "coverage"] = cache["coverage"]
        payload[prefix + "dan"] = dan
        payload[prefix + "linear"] = linear
        payload[prefix + "aux_dan"] = aux_dan
        payload[prefix + "aux_linear"] = aux_linear

    fit_receipts = {fit_id: f"receipt-{fit_id}" for fit_id in runner.FIT_IDS}
    fit_artifacts = {fit_id: f"model-{fit_id}" for fit_id in runner.FIT_IDS}
    cache_ready = {
        "freeze_signature": "SYNTHETIC_FREEZE",
        "fit_receipt_hashes": fit_receipts,
        "fit_artifact_hashes": fit_artifacts,
        "origins": {"67": {"sha256": "cache-67"}, "75": {"sha256": "cache-75"}},
    }
    good_path = tmp_path / "good.npz"
    runner.atomic_npz(good_path, **payload)
    verification = repair._verify_prelabel_against_caches(
        good_path, cache_ready, caches, fit_receipts, fit_artifacts
    )
    assert verification["overlap"]["champion_probabilities_different_rows"] > 0
    assert verification["target_free_payload_metadata"] is True

    tampered = dict(payload)
    tampered["E75_champion"] = payload["E75_champion"].copy()
    tampered["E75_champion"][0] += 0.01
    bad_path = tmp_path / "bad.npz"
    runner.atomic_npz(bad_path, **tampered)
    with pytest.raises(RuntimeError, match="hash-bound feature cache"):
        repair._verify_prelabel_against_caches(
            bad_path, cache_ready, caches, fit_receipts, fit_artifacts
        )


@pytest.mark.parametrize(
    ("field", "position", "replacement"),
    [
        ("score_rows", 0, 999),
        ("score_record_ids", 0, "wrong-id"),
        ("score_cycles", 0, 75),
        ("present", 0, True),
    ],
)
def test_overlap_alignment_fails_on_identity_or_availability_change(field, position, replacement):
    first, second = _views()
    second[field][position] = replacement
    with pytest.raises(RuntimeError):
        repair.validate_origin_alignment(first, second)


def test_prelabel_commit_binding_requires_exact_hash_and_target_free_flags(tmp_path, monkeypatch):
    audit = tmp_path / "audit"
    run = audit / "experiments/CLS_DAN_V1"
    prelabel = run / "predictions/PRELABEL.npz"
    readiness = run / "PREDICTIONS_READY.json"
    receipt_path = audit / "reports/simplification_v1/CLS_DAN_V1/POSTFIT_PRELABEL_R1.json"
    prelabel.parent.mkdir(parents=True)
    receipt_path.parent.mkdir(parents=True)
    prelabel.write_bytes(b"synthetic-npz-payload")
    monkeypatch.setattr(runner, "AUDIT", audit)
    monkeypatch.setattr(runner, "PREDICTIONS_READY_PATH", readiness)
    prelabel_hash = hashlib.sha256(prelabel.read_bytes()).hexdigest()
    ready = {
        "status": "PASS_PRELABEL_READY_NO_QUERY_LABELS",
        "prelabel_path": prelabel.relative_to(audit).as_posix(),
        "prelabel_sha256": prelabel_hash,
        "evaluation_query_targets_read": False,
        "metrics_computed": False,
        "test_opened": False,
        "sample_submission_opened": False,
        "kaggle_contacted": False,
    }
    readiness.write_text(json.dumps(ready), encoding="utf-8")
    ready_hash = hashlib.sha256(readiness.read_bytes()).hexdigest()
    receipt = {
        "status": "PASS_POSTFIT_PRELABEL_REPAIR_R1",
        "prelabel_path": prelabel.relative_to(audit).as_posix(),
        "prelabel_sha256": prelabel_hash,
        "prediction_ready_sha256": ready_hash,
    }
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    assert repair.validate_prelabel_binding(
        prelabel, ready, receipt, expected_ready_sha256=ready_hash
    )["prelabel_sha256"] == prelabel_hash

    changed_ready = dict(ready, evaluation_query_targets_read=True)
    with pytest.raises(RuntimeError, match="target-free prediction manifest"):
        repair.validate_prelabel_binding(prelabel, changed_ready, receipt)
    changed_receipt = dict(receipt, prediction_ready_sha256="0" * 64)
    with pytest.raises(RuntimeError, match="post-fit adapter receipt"):
        repair.validate_prelabel_binding(prelabel, ready, changed_receipt)

    prelabel.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="target-free prediction manifest"):
        repair.validate_prelabel_binding(prelabel, ready, receipt)


def test_exact_synthetic_prelabel_arm_begin_score_path(tmp_path):
    result = repair.run_synthetic_downstream_qa(tmp_path / "synthetic-run")
    assert result["status"] == "PASS_SYNTHETIC_PRELABEL_ARM_BEGIN_SCORE"
    assert result["overlap"]["shared_query_rows"] == 8491
    assert result["overlap"]["champion_probabilities_different_rows"] > 0
    assert result["primary_rows"] == 20903
    assert result["target_read_calls"] == 1
    assert result["independent_metrics_match"] is True
    assert result["test_opened"] is False
    assert result["sample_submission_opened"] is False
    assert result["kaggle_contacted"] is False
