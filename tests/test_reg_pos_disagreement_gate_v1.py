from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "reg_pos_disagreement_gate_v1.py"
SPEC = importlib.util.spec_from_file_location("reg_pos_disagreement_gate_v1_test", SCRIPT)
gate = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(gate)


class IdentityHinge:
    @staticmethod
    def apply_hinge_present_only(values, missing, parameters):
        result = np.asarray(values, dtype=np.float64).copy()
        mask = np.asarray(missing, dtype=bool)
        baseline = np.asarray(parameters["baseline"], dtype=np.float64)
        result[mask] = baseline[mask]
        return result


def _policy_data():
    rng = np.random.default_rng(20261106)
    baseline = rng.uniform(0.01, 99.99, 512).astype(np.float64)
    pos = rng.uniform(0.01, 99.99, 512).astype(np.float64)
    cycles = np.repeat(np.arange(68, 80, dtype=np.int64), 512 // 12)
    cycles = np.pad(cycles, (0, 512 - len(cycles)), mode="edge")
    missing = np.zeros(512, dtype=bool)
    missing[[0, 7, 91, 255]] = True
    parent = np.clip(0.75 * baseline + 0.25 * pos, 0.0, 100.0)
    parent[missing] = baseline[missing]
    parent_hinge = parent.copy()
    return {
        "query_rows": np.arange(512, dtype=np.int64),
        "query_ids": np.asarray([f"SYN-{i:04d}" for i in range(512)]),
        "cycles": cycles,
        "missing": missing,
        "baseline": baseline,
        "pos_t50": pos,
        "champion_direct": parent,
        "champion_hinge": parent_hinge,
        "impact_quartile": (np.arange(512) % 4).astype(np.int8),
        "text_quartile": ((np.arange(512) // 4) % 4).astype(np.int8),
        "entry_friction": (np.arange(512) % 2).astype(np.float64),
    }


def test_parent_float_order_is_bit_exact_and_gate_formula_unchanged():
    data = _policy_data()
    baseline = data["baseline"]
    pos = data["pos_t50"]
    parent_order = np.clip(0.75 * baseline + 0.25 * pos, 0.0, 100.0)
    algebraic_order = np.clip(baseline + 0.25 * (pos - baseline), 0.0, 100.0)
    parent_order[data["missing"]] = baseline[data["missing"]]
    algebraic_order[data["missing"]] = baseline[data["missing"]]
    assert np.array_equal(parent_order, data["champion_direct"])
    assert np.any(parent_order.view("<u8") != algebraic_order.view("<u8"))
    state = {"status": "FIT_COMPLETE", "scale_s": 4.0}
    output = gate._construct_policy(67, data, state, IdentityHinge, {"baseline": baseline})
    expected_real_weight = 0.25 * np.exp(-np.abs(pos - baseline) / 4.0)
    expected_real = np.clip(baseline + expected_real_weight * (pos - baseline), 0.0, 100.0)
    expected_real[data["missing"]] = baseline[data["missing"]]
    assert np.array_equal(output["champion_direct"], parent_order)
    assert np.array_equal(output["real_direct"], expected_real)


def test_unsupported_scale_copies_champion_exactly():
    data = _policy_data()
    algebraic = np.clip(data["baseline"] + 0.25 * (data["pos_t50"] - data["baseline"]), 0.0, 100.0)
    algebraic[data["missing"]] = data["baseline"][data["missing"]]
    assert np.any(data["champion_direct"].view("<u8") != algebraic.view("<u8"))
    state = {"status": "UNSUPPORTED", "scale_s": None}
    output = gate._construct_policy(67, data, state, IdentityHinge, {"baseline": data["baseline"]})
    assert np.array_equal(output["real_direct"], data["champion_direct"])
    assert np.array_equal(output["control_direct"], data["champion_direct"])
    assert np.array_equal(output["real_hinge"], data["champion_hinge"])
    assert np.array_equal(output["control_hinge"], data["champion_hinge"])


def test_npz_receipt_rejects_tampering(tmp_path):
    path = tmp_path / "prediction.npz"
    receipt = gate._write_npz_immutable(path, {"prediction": np.asarray([1.0, 2.0, 3.0])})
    with path.open("wb") as stream:
        np.savez_compressed(stream, prediction=np.asarray([1.0, 2.0, 3.1]))
    try:
        gate._verify_npz_receipt(path, receipt)
    except gate.GateError:
        pass
    else:
        raise AssertionError("Changed NPZ unexpectedly passed its immutable receipt")


def test_saved_scale_state_reuse_does_not_refit(tmp_path, monkeypatch):
    state = {"schema_version": 1, "experiment_id": gate.CARD_ID,
             "state_id": "D1", "status": "FIT_COMPLETE", "scale_s": 2.5,
             "labels_used": False}
    path = tmp_path / "SCALE_D1_R1.json"
    saved = gate._write_json_immutable(path, state, "state_signature")
    receipt = {"path": str(path), "sha256": gate._sha_file(path),
               "state_signature": saved["state_signature"]}
    monkeypatch.setattr(gate, "_scale_state",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("refit attempted")))
    reused = gate._load_saved_scale_state(receipt)
    assert reused == saved
