from __future__ import annotations

import numpy as np
import pytest

from scripts.optimization_v3_positional_contrast_v1 import (
    _fixed_route,
    _load_named_npz,
    run_synthetic_lifecycle,
)
from scripts.verify_optimization_v3_positional_contrast_v1 import (
    assert_ids_match,
    independent_metric,
    verify_file_hash,
    verify_receipt_binding,
)


def test_synthetic_end_to_end_temp_only_and_production_ledger_guard(tmp_path):
    result = run_synthetic_lifecycle(tmp_path / "pos_contrast_synthetic")
    assert result["status"] == "PASS"
    assert result["optimizer_fit_count"] == 1
    assert result["durable_resume"] == "resumed_from_durable_receipt"
    assert result["missing_route_exact"]
    assert result["tampered_ids_rejected"]
    assert result["source_mismatch_rejected"]
    assert result["bundle_mismatch_guard_exercised"]
    assert result["production_ledger_sha256_before"] == result["production_ledger_sha256_after"]


def test_exact_missing_route_and_fixed_present_blend():
    q = np.array([0.2, 0.7, 0.4, 0.9])
    p_a = np.array([0.8, 0.4, 0.6, 0.2])
    p_p = np.array([0.3, 0.5, 0.7, 0.1])
    present = np.array([True, False, True, False])
    actual = _fixed_route(q, p_a, p_p, present)
    np.testing.assert_array_equal(actual[~present], p_p[~present])
    np.testing.assert_allclose(actual[present], 0.5 * q[present] + 0.5 * p_a[present], atol=0, rtol=0)


def test_alignment_receipt_and_bundle_guards_reject_tampering(tmp_path):
    ids = np.array(["a", "b", "c"])
    assert_ids_match(ids, ids.copy())
    with pytest.raises(AssertionError, match="alignment"):
        assert_ids_match(ids[::-1], ids)
    receipt = {"fit_id": "F1", "freeze_signature": "S1", "artifact_sha256": "A1"}
    verify_receipt_binding(receipt, fit_id="F1", freeze_signature="S1", artifact_sha256="A1")
    with pytest.raises(AssertionError, match="receipt"):
        verify_receipt_binding(receipt, fit_id="F1", freeze_signature="S2", artifact_sha256="A1")
    bundle = tmp_path / "bundle.bin"
    bundle.write_bytes(b"frozen")
    import hashlib
    expected = hashlib.sha256(b"frozen").hexdigest()
    verify_file_hash(bundle, expected)
    with pytest.raises(AssertionError, match="bundle"):
        verify_file_hash(bundle, "0" * 64)


def test_named_npz_loader_does_not_request_historical_label_key(tmp_path):
    path = tmp_path / "mixed_bundle.npz"
    np.savez(path, prediction=np.array([0.2, 0.8]), labels=np.array([1, 0]))
    selected = _load_named_npz(path, ["prediction"])
    assert list(selected) == ["prediction"]
    np.testing.assert_array_equal(selected["prediction"], [0.2, 0.8])


def test_independent_metric_threshold_and_counts():
    y = np.array([0, 0, 1, 1], dtype=np.int8)
    p = np.array([0.1, 0.5, 0.6, 0.9])
    metric = independent_metric(y, p)
    assert metric["n"] == 4
    assert metric["positive"] == 2
    assert metric["negative"] == 2
    assert metric["confusion_matrix"] == [[1, 1], [0, 2]]


def test_inherited_offset_runner_validates_active_freeze_keyword_before_fit(monkeypatch, tmp_path):
    from scripts import optimization_v3_positional_bag_offset_v1 as offset
    from scripts import optimization_v3_positional_contrast_v1 as pilot

    import hashlib
    prefix = b"synthetic pre-reservation ledger\n"
    ledger_path = tmp_path / "budget.jsonl"
    ledger_path.write_bytes(prefix)
    base_freeze = {"freeze_signature": "prior-freeze-signature", "source_commit": "prior-source-commit",
                   "pre_reservation_ledger_bytes": len(prefix),
                   "pre_reservation_ledger_sha256": hashlib.sha256(prefix).hexdigest()}
    active_freeze = {**base_freeze, "freeze_signature": "active-successor-signature",
                     "source_commit": "synthetic-source", "source_hashes": {},
                     "input_sha256": "synthetic-train"}
    events = [
        {"event": "BUDGET_CAP_AMENDMENT", "card_id": pilot.CARD_ID,
         "amendment_id": pilot.AMENDMENT_ID, "prior_global_cap": 132, "global_cap": 137,
         "prior_effective_caps": pilot.BASE_CAPS, "effective_caps": pilot.NEW_CAPS,
         "freeze_signature": base_freeze["freeze_signature"], "source_commit": base_freeze["source_commit"]},
        {"event": "CARD_FITS_RESERVED", "card_id": pilot.CARD_ID,
         "reservation_key": pilot.PILOT_RESERVATION_KEY, "fit_ids": list(pilot.FIT_IDS), "fits": 4,
         "category": "pilot", "effective_category_cap": 84,
         "freeze_signature": base_freeze["freeze_signature"]},
        {"event": "CARD_FITS_RESERVED", "card_id": pilot.CARD_ID,
         "reservation_key": pilot.E79_RESERVATION_KEY, "fit_ids": [pilot.E79_ID], "fits": 1,
         "category": "continuation", "effective_category_cap": 50,
         "stage": "CONDITIONAL_E79_AFTER_SOL_REVIEW",
         "freeze_signature": base_freeze["freeze_signature"]},
    ]
    snapshot = {"global_cap": 137, "effective_category_caps": pilot.NEW_CAPS,
                "fit_starts": 129, "fit_saved": 123, "fit_failed": 6, "fit_pending": 0,
                "active_reservations_by_category": {"continuation": 50, "control_replay": 2,
                                                    "pilot": 81, "technical_retry": 1}}

    class PreStartStop(Exception):
        pass

    def stop_before_estimator(*args, **kwargs):
        raise PreStartStop("synthetic stop before matrix/estimator path")

    # Register every global mutated by the real adapter so pytest restores it.
    for name in ("CARD_ID", "AMENDMENT_ID", "FIT_IDS", "RESERVATION_KEY", "STAGE", "LEDGER",
                 "RUN_ROOT", "REPORT_ROOT", "_load_origin_payload", "_check_live_budget",
                 "verify_saved_inputs", "load_freeze"):
        monkeypatch.setattr(offset, name, getattr(offset, name))
    monkeypatch.setattr(pilot, "RUN_ROOT", tmp_path)
    monkeypatch.setattr(pilot, "LEDGER", ledger_path)
    monkeypatch.setattr(pilot, "load_freeze", lambda: active_freeze)
    monkeypatch.setattr(pilot, "_read_base_freeze", lambda: base_freeze)
    monkeypatch.setattr(pilot, "_ledger_snapshot", lambda: (events, snapshot))
    monkeypatch.setattr(pilot, "_verify_parent_inputs", lambda: {"status": "PASS"})
    monkeypatch.setattr(pilot, "_contrast_payload", stop_before_estimator)
    monkeypatch.setattr(offset, "_resume_started_fit", lambda *a, **k: "not_started")
    pilot._configure_offset_module()

    try:
        offset._run_fit(pilot.FIT_IDS[0], active_freeze)
        pytest.fail("synthetic pre-start stop was not reached")
    except PreStartStop:
        pass
    assert not any(item.get("event") == "FIT_STARTED" for item in events)

    wrong_freeze = dict(active_freeze, freeze_signature="wrong-signature")
    with pytest.raises(pilot.ContrastPilotError, match="different freeze signature"):
        offset._run_fit(pilot.FIT_IDS[0], wrong_freeze)
    assert not any(item.get("event") == "FIT_STARTED" for item in events)


def test_prefit_rebind_is_append_only_and_adds_no_fit_slots(monkeypatch, tmp_path):
    import hashlib
    import json
    from scripts import optimization_v3_positional_contrast_v1 as pilot

    prefix = b"prior ledger prefix\n"
    ledger = tmp_path / "budget.jsonl"
    active = {"reservation_checkpoint_ledger_bytes": len(prefix),
              "reservation_checkpoint_ledger_sha256": hashlib.sha256(prefix).hexdigest(),
              "reservation_checkpoint_snapshot": {"global_cap": 137},
              "freeze_signature": "active-freeze", "source_commit": "active-source"}
    base = {"freeze_signature": "prior-freeze", "source_commit": "prior-source"}
    event = {
        "event": "PREFIT_SOURCE_REBIND", "card_id": pilot.CARD_ID,
        "amendment_id": pilot.AMENDMENT_ID, "prefit_amendment_id": pilot.PREFIT_REBIND_ID,
        "prior_freeze_signature": "prior-freeze", "freeze_signature": "active-freeze",
        "prior_source_commit": "prior-source", "source_commit": "active-source",
        "fit_ids": list(pilot.FIT_IDS), "conditional_fit_id": pilot.E79_ID,
        "reservation_keys": [pilot.PILOT_RESERVATION_KEY, pilot.E79_RESERVATION_KEY],
        "ledger_sha256_before_rebind": active["reservation_checkpoint_ledger_sha256"],
        "ledger_snapshot_before_rebind": active["reservation_checkpoint_snapshot"],
        "fit_starts_before_rebind": [], "terminal_fit_ids_before_rebind": [],
        "estimator_fit_calls_before_rebind": 0, "added_fit_slots": 0,
        "caps_unchanged": True, "recipe_gates_populations_unchanged": True,
        "at_utc": "2026-11-04T00:00:00+00:00",
    }
    ledger.write_bytes(prefix + pilot.offset._canonical_bytes(event) + b"\n")
    record = {"status": "PASS_REBOUND_NO_NEW_SLOTS", "event": event,
              "event_sha256": pilot._sha_obj(event)}
    report = tmp_path / "PREFIT_AMENDMENT.json"
    report.write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(pilot, "LEDGER", ledger)
    monkeypatch.setattr(pilot, "PREFIT_AMENDMENT_REPORT", report)
    monkeypatch.setattr(pilot, "_ledger_snapshot", lambda: ([event], {}))
    assert pilot._verify_prefit_rebind(base, active) == record

    forged = dict(record, event=dict(event, added_fit_slots=1))
    forged["event_sha256"] = pilot._sha_obj(forged["event"])
    report.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(pilot.ContrastPilotError):
        pilot._verify_prefit_rebind(base, active)
