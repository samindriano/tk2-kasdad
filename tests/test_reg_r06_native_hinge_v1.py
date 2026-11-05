from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from scripts import reg_r06_native_hinge_v1 as pilot


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
        raise pilot.GateError("ROOT_SOURCE_PASS_R1.json is absent")

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
        [sys.executable, "-m", "scripts.reg_r06_native_hinge_v1", "synthetic-qa"],
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
