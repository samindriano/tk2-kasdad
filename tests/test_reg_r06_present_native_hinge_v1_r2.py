from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from scripts import reg_r06_present_native_hinge_v1_r2 as pilot


def test_fit_windows_exclude_missing_but_score_windows_keep_all_rows() -> None:
    cycles = np.repeat(np.arange(68, 80, dtype=np.int16), 100)
    missing = np.arange(len(cycles)) % 5 == 0
    d1_fit, d1_score, _ = pilot._stage_indices(cycles, pilot.FIT_IDS[0], missing)
    d2_fit, d2_score, _ = pilot._stage_indices(cycles, pilot.FIT_IDS[1], missing)
    final_fit, final_score, _ = pilot._stage_indices(cycles, pilot.FIT_IDS[2], missing)
    assert len(d1_fit) == int(np.sum((cycles >= 68) & (cycles <= 71) & ~missing))
    assert len(d2_fit) == int(np.sum((cycles >= 68) & (cycles <= 75) & ~missing))
    assert not np.any(missing[d1_fit]) and not np.any(missing[d2_fit])
    assert cycles[d1_score].tolist() == np.repeat([72, 73, 74, 75], 100).tolist()
    assert cycles[d2_score].tolist() == np.repeat([76, 77, 78, 79], 100).tolist()
    assert len(final_fit) == int(np.sum(~missing)) and len(final_score) == 0
    with pytest.raises(pilot.GateError, match="missing mask"):
        pilot._stage_indices(cycles, pilot.FIT_IDS[0])


def test_fit_input_builder_rejects_any_missing_row() -> None:
    identity = {
        "labels": {
            "physical_rows": np.asarray([10, 11]),
            "record_ids": np.asarray(["a", "b"]),
            "prediction_origins": np.asarray([67, 67]),
            "cycles": np.asarray([68, 68]),
            "missing_mask": np.asarray([False, True]),
        },
        "r06_origin": np.asarray([67, 67]),
        "r06_prediction": np.asarray([20.0, 30.0]),
    }
    with pytest.raises(pilot.GateError, match="missing row"):
        pilot._fit_input_arrays(identity, np.asarray([0, 1]), np.asarray([19.0, 31.0]))


def test_missing_application_is_bit_exact_and_present_rows_are_clipped() -> None:
    raw = np.asarray([-0.0, 25.0, 50.0, 100.0], dtype=np.float64)
    missing = np.asarray([True, False, False, True])
    output = pilot.apply_hinge(raw, missing, raw, {
        "intercept": 50.0, "slope_low": 2.0, "slope_high": 2.0, "knot": 50.0,
    })
    assert np.array_equal(output[missing].view(np.uint64), raw[missing].view(np.uint64))
    assert np.all((output[~missing] >= 0.0) & (output[~missing] <= 100.0))


def test_target_bundle_reader_only_loads_selected_identity_members(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.npz"
    np.savez_compressed(path, record_ids=np.asarray(["x", "y"]),
                        cycles=np.asarray([68, 69]), targets=np.asarray([4.0, 96.0]))
    data, members = pilot._load_npz_selected(path, ("record_ids", "cycles"))
    assert "targets" in members
    assert set(data) == {"record_ids", "cycles"}
    assert "targets" not in data


def test_diagnostics_fail_closed_before_identity_targets_or_solver_without_pass(monkeypatch) -> None:
    calls = {"identity": 0, "targets": 0, "solver": 0}

    def deny():
        raise pilot.GateError("ROOT_SOURCE_PASS_R2.json is absent")

    def forbidden(name: str):
        def call(*_args, **_kwargs):
            calls[name] += 1
            raise AssertionError(f"{name} must not run before source approval")
        return call

    monkeypatch.setattr(pilot, "_load_root_authorization", deny)
    monkeypatch.setattr(pilot, "_load_non_target_identity", forbidden("identity"))
    monkeypatch.setattr(pilot, "_load_cached_targets_after_pass", forbidden("targets"))
    monkeypatch.setattr(pilot.r4, "solve_hinge_l1", forbidden("solver"))
    with pytest.raises(pilot.GateError, match="absent"):
        pilot.run_diagnostics()
    assert calls == {"identity": 0, "targets": 0, "solver": 0}


def test_mechanism_gate_is_required_even_when_primary_gate_passes() -> None:
    assert pilot._root_review_eligible(False, True, True) is False
    assert pilot._root_review_eligible(True, False, True) is True
    assert pilot._root_review_eligible(True, True, False) is True


def test_ledger_accepts_one_complete_slot_and_rejects_retry_or_slot_skip() -> None:
    with tempfile.TemporaryDirectory() as folder:
        base = Path(folder)
        common = {"fit_id": pilot.FIT_IDS[0], "fit_input_sha256": "a" * 64,
                  "root_source_pass_sha256": "b" * 64}
        good = base / "good.jsonl"
        pilot._write_text_once(good, [
            {"event": "RESERVE", **common}, {"event": "LP_STARTED", **common},
            {"event": "LP_COMPLETE", **common},
        ])
        assert pilot._validate_ledger(good)[pilot.FIT_IDS[0]]["LP_COMPLETE"]
        retry = base / "retry.jsonl"
        pilot._write_text_once(retry, [
            {"event": "RESERVE", **common}, {"event": "LP_STARTED", **common},
            {"event": "LP_FAILED", **common}, {"event": "LP_STARTED", **common},
        ])
        with pytest.raises(pilot.GateError, match="Duplicate|Invalid"):
            pilot._validate_ledger(retry)
        skip = base / "skip.jsonl"
        next_fit = {**common, "fit_id": pilot.FIT_IDS[1]}
        pilot._write_text_once(skip, [
            {"event": "RESERVE", **next_fit},
        ])
        with pytest.raises(pilot.GateError, match="contiguous"):
            pilot._validate_ledger(skip)


def test_synthetic_cli_writes_signed_metric_e2e_receipt_and_verifier_replays_it() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "scripts.reg_r06_present_native_hinge_v1_r2", "synthetic-qa"],
        cwd=pilot.ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert pilot.SYNTHETIC_QA.is_file()
    receipt = json.loads(pilot.SYNTHETIC_QA.read_text(encoding="utf-8"))
    signature = receipt.pop("signature")
    assert signature == pilot.canonical_sha(receipt)
    assert receipt["status"] == "PASS_SYNTHETIC_SOURCE_AND_RECEIPT_E2E"
    assert receipt["checks"]["official_lp_starts"] == 0
    assert receipt["checks"]["present_only_fit_routing_excludes_missing"] is True
    assert receipt["metrics_e2e"]["status"] == "PASS_SYNTHETIC_METRIC_E2E"
    assert receipt["metrics_e2e"]["score_rows"] == 3_200
    from scripts.verify_reg_r06_present_native_hinge_v1_r2 import _check_synthetic_metric_e2e
    assert _check_synthetic_metric_e2e(receipt)["status"] == "PASS"


def test_source_ready_command_dispatches_to_present_only_verifier(monkeypatch, capsys) -> None:
    from scripts import verify_reg_r06_present_native_hinge_v1_r2 as verifier
    called = {"write_receipt": False}

    def stub(*, write_receipt: bool) -> dict[str, object]:
        called["write_receipt"] = write_receipt
        return {"status": "PRESENT_ONLY_VERIFIER_DISPATCHED", "fit_starts": 0}

    monkeypatch.setattr(verifier, "verify_source_ready", stub)
    assert pilot.main(["verify-source-ready"]) == 0
    assert called["write_receipt"] is True
    assert '"status": "PRESENT_ONLY_VERIFIER_DISPATCHED"' in capsys.readouterr().out
