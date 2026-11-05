from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from scripts import reg_r06_native_hinge_v2 as pilot


def test_missing_route_is_bit_exact_and_present_route_is_clipped() -> None:
    raw = np.asarray([-0.0, 25.0, 50.0, 100.0], dtype=np.float64)
    # The production gate restricts source predictions to [0,100]; synthetic
    # negative zero is used here only to test that np.where preserves its bits.
    raw[0] = -0.0
    missing = np.asarray([True, False, False, True])
    out = pilot.apply_hinge(raw, missing, raw, {
        "intercept": 50.0, "slope_low": 2.0, "slope_high": 2.0, "knot": 50.0,
    })
    assert np.array_equal(out[missing].view(np.uint64), raw[missing].view(np.uint64))
    assert np.all((out[~missing] >= 0.0) & (out[~missing] <= 100.0))


def test_cycle_split_is_disjoint_and_matches_frozen_windows() -> None:
    cycles = np.arange(68, 80, dtype=np.int16)
    d1_fit, d1_score, _ = pilot._stage_indices(cycles, pilot.FIT_IDS[0])
    d2_fit, d2_score, _ = pilot._stage_indices(cycles, pilot.FIT_IDS[1])
    final_fit, _, _ = pilot._stage_indices(cycles, pilot.FIT_IDS[2])
    assert cycles[d1_fit].tolist() == [68, 69, 70, 71]
    assert cycles[d1_score].tolist() == [72, 73, 74, 75]
    assert cycles[d2_fit].tolist() == list(range(68, 76))
    assert cycles[d2_score].tolist() == [76, 77, 78, 79]
    assert cycles[final_fit].tolist() == list(range(68, 80))
    assert not set(d1_score.tolist()) & set(d2_score.tolist())


def test_target_bundle_reader_is_explicitly_selective(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.npz"
    np.savez_compressed(path, record_ids=np.asarray(["x", "y"]),
                        cycles=np.asarray([68, 69]), targets=np.asarray([4.0, 96.0]))
    data, members = pilot._load_npz_selected(path, ("record_ids", "cycles"))
    assert "targets" in members
    assert set(data) == {"record_ids", "cycles"}
    assert "targets" not in data


def test_diagnostics_stop_before_any_target_or_solver_call_without_pass(monkeypatch) -> None:
    calls = {"identity": 0, "targets": 0, "solver": 0}

    def deny():
        raise pilot.GateError("ROOT_SOURCE_PASS_R2C.json is absent")

    def identity(*_args, **_kwargs):
        calls["identity"] += 1
        raise AssertionError("identity/target reads must not run before authorization")

    monkeypatch.setattr(pilot, "_load_root_authorization", deny)
    monkeypatch.setattr(pilot, "_load_non_target_identity", identity)
    monkeypatch.setattr(pilot, "_load_cached_targets_after_pass",
                        lambda *_a, **_k: calls.__setitem__("targets", calls["targets"] + 1))
    monkeypatch.setattr(pilot.r4, "solve_hinge_l1",
                        lambda *_a, **_k: calls.__setitem__("solver", calls["solver"] + 1))
    with pytest.raises(pilot.GateError, match="absent"):
        pilot.run_diagnostics()
    assert calls == {"identity": 0, "targets": 0, "solver": 0}


@pytest.mark.parametrize("slot_index", [0, 1])
def test_final_state_cannot_replace_diagnostic_cutoff_before_identity_or_targets(
        tmp_path: Path, monkeypatch, slot_index: int) -> None:
    fit_id = pilot.DIAGNOSTIC_IDS[slot_index]
    monkeypatch.setattr(pilot, "RUN_DIR", tmp_path)
    monkeypatch.setattr(pilot, "LEDGER", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(pilot, "_load_root_authorization",
                        lambda: ({"source_commit": "commit"}, "freeze", {}, "approval"))
    state_path = tmp_path / "fits" / f"{fit_id}.json"
    state = {
        "fit_id": pilot.FIT_IDS[2], "status": "LP_COMPLETE",
        "fit_input": {"fit_id": pilot.FIT_IDS[2], "fit_rows": pilot.EXPECTED_ROWS,
                      "prefit_freeze_sha256": "freeze",
                      "root_source_pass_sha256": "approval",
                      "fit_input_bundle_sha256": "bundle"},
        "root_source_pass_sha256": "approval",
        "solver_result": {"fit_rows": pilot.EXPECTED_ROWS},
    }
    state_path.parent.mkdir(parents=True)
    state_path.write_text(__import__("json").dumps(state), encoding="utf-8")
    state_sha = pilot.file_sha256(state_path)
    monkeypatch.setattr(pilot, "_validate_ledger", lambda *_a, **_k: {
        fit_id: {
            "RESERVE": [{"fit_input_sha256": "bundle"}],
            "LP_STARTED": [{"fit_input_sha256": "bundle"}],
            "LP_COMPLETE": [{"fit_input_sha256": "bundle", "state_sha256": state_sha}],
            "LP_FAILED": [],
        }
    })
    calls = {"identity": 0, "targets": 0, "solver": 0}

    def forbidden(name: str):
        def call(*_args, **_kwargs):
            calls[name] += 1
            raise AssertionError(f"{name} must not run after cutoff-state substitution")
        return call

    monkeypatch.setattr(pilot, "_load_non_target_identity", forbidden("identity"))
    monkeypatch.setattr(pilot, "_load_cached_targets_after_pass", forbidden("targets"))
    monkeypatch.setattr(pilot.r4, "solve_hinge_l1", forbidden("solver"))
    with pytest.raises(pilot.GateError, match="wrong calibration cutoff"):
        pilot.run_diagnostics()
    assert calls == {"identity": 0, "targets": 0, "solver": 0}


def test_exploratory_only_review_eligibility_does_not_start_final_without_root_approval(
        tmp_path: Path, monkeypatch) -> None:
    assert pilot._root_review_eligible(False, True)
    monkeypatch.setattr(pilot, "RUN_DIR", tmp_path / "run")
    monkeypatch.setattr(pilot, "LEDGER", tmp_path / "run" / "ledger.jsonl")
    monkeypatch.setattr(pilot, "DIAGNOSTICS", tmp_path / "diagnostics.json")
    monkeypatch.setattr(pilot, "ROOT_FINAL_APPROVAL", tmp_path / "missing-final-approval.json")
    freeze = {"source_commit": "commit", "source_hashes": {}, "input_hashes": {}}
    monkeypatch.setattr(pilot, "_load_root_authorization",
                        lambda: (freeze, "freeze", {}, "source-pass"))
    report = {
        "experiment_id": pilot.EXPERIMENT_ID,
        "prefit_freeze_sha256": "freeze",
        "root_source_pass_sha256": "source-pass",
        "qualification_gate": {"pass": False},
        "exploratory_gate": {"pass": True},
        "root_review_eligible": True,
        "final_fit_started": False,
        "final_fit_requires_separate_root_approval": True,
    }
    report["signature"] = pilot.canonical_sha(report)
    pilot.DIAGNOSTICS.write_text(__import__("json").dumps(report), encoding="utf-8")
    calls = {"identity": 0, "targets": 0, "solver": 0}

    def forbidden(name: str):
        def call(*_args, **_kwargs):
            calls[name] += 1
            raise AssertionError(f"{name} must not run without separate root approval")
        return call

    monkeypatch.setattr(pilot, "_load_non_target_identity", forbidden("identity"))
    monkeypatch.setattr(pilot, "_load_cached_targets_after_pass", forbidden("targets"))
    monkeypatch.setattr(pilot.r4, "solve_hinge_l1", forbidden("solver"))
    with pytest.raises(pilot.GateError, match="final-fit approval is absent"):
        pilot.run_final_fit()
    assert calls == {"identity": 0, "targets": 0, "solver": 0}


def test_ledger_rejects_duplicate_start_and_accepts_single_complete() -> None:
    with tempfile.TemporaryDirectory() as folder:
        base = Path(folder)
        event = {"fit_id": pilot.FIT_IDS[0], "fit_input_sha256": "a" * 64,
                 "root_source_pass_sha256": "b" * 64}
        good = base / "good.jsonl"
        pilot._write_text_once(good, [
            {"event": "RESERVE", **event}, {"event": "LP_STARTED", **event},
            {"event": "LP_COMPLETE", **event},
        ])
        assert pilot._validate_ledger(good)[pilot.FIT_IDS[0]]["LP_COMPLETE"]
        bad = base / "bad.jsonl"
        pilot._write_text_once(bad, [
            {"event": "RESERVE", **event}, {"event": "LP_STARTED", **event},
            {"event": "LP_FAILED", **event}, {"event": "LP_STARTED", **event},
        ])
        with pytest.raises(pilot.GateError, match="Duplicate|Invalid"):
            pilot._validate_ledger(bad)


def test_resource_and_shared_lock_guards() -> None:
    assert pilot.r4.resource_decision(3 * 1024**3, initial=True)["pass"]
    assert not pilot.r4.resource_decision(512 * 1024**2, initial=True)["pass"]
    with tempfile.TemporaryDirectory() as folder:
        lock = Path(folder) / "shared.lock"
        with pilot.r4.SharedHeavyJobLock(lock):
            assert lock.exists()


def test_synthetic_cli_writes_a_verifiable_receipt() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "scripts.reg_r06_native_hinge_v2", "synthetic-qa"],
        cwd=pilot.ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert pilot.SYNTHETIC_QA.is_file()
    import json
    receipt = json.loads(pilot.SYNTHETIC_QA.read_text(encoding="utf-8"))
    signature = receipt.pop("signature")
    assert signature == pilot.canonical_sha(receipt)
    assert receipt["status"] == "PASS_SYNTHETIC_SOURCE_AND_RECEIPT_E2E"
    assert receipt["checks"]["official_lp_starts"] == 0


def test_source_ready_command_dispatches_to_r2_verifier(monkeypatch, capsys) -> None:
    from scripts import verify_reg_r06_native_hinge_v2 as verifier
    called = {"write_receipt": False}

    def stub(*, write_receipt: bool) -> dict[str, object]:
        called["write_receipt"] = write_receipt
        return {"status": "R2C_VERIFIER_DISPATCHED", "fit_starts": 0}

    monkeypatch.setattr(verifier, "verify_source_ready", stub)
    assert pilot.main(["verify-source-ready"]) == 0
    assert called["write_receipt"] is True
    assert '"status": "R2C_VERIFIER_DISPATCHED"' in capsys.readouterr().out
