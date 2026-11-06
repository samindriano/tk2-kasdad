from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
WRAPPER_PATH = ROOT / "scripts/run_cls_pos_ef_context_v1_prelabel_amended.py"
FROZEN_HELPERS = {"sha256_array", "sha256_csr"}


def load_wrapper():
    spec = importlib.util.spec_from_file_location("cls_pos_ef_context_v1_prelabel_compat_test", WRAPPER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_symtable_audit_reports_only_two_frozen_verifier_helpers():
    wrapper = load_wrapper()
    assert wrapper.unresolved_global_names(ROOT / "scripts/verify_cls_pos_ef_context_v1.py") == FROZEN_HELPERS


def test_binds_only_hash_helpers_to_frozen_driver_functions(monkeypatch):
    wrapper = load_wrapper()
    runner = wrapper.load_runner()
    pilot = runner.load_frozen_driver()
    freeze = json.loads((ROOT / "reports/simplification_v1/CLS_POS_EF_CONTEXT_V1/FREEZE.json").read_text(encoding="utf-8"))
    runner.load_audit_core(pilot, freeze)
    module_name = "scripts.verify_cls_pos_ef_context_v1"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    verifier, unresolved, _ = wrapper.verified_verifier_module(runner, pilot, freeze)
    try:
        assert unresolved == FROZEN_HELPERS
        assert verifier.sha256_array is pilot.sha256_array
        assert verifier.sha256_csr is pilot.sha256_csr
        assert verifier._independent_objective.__module__ == module_name
        assert verifier._independent_shuffle.__module__ == module_name
    finally:
        for name in FROZEN_HELPERS:
            if hasattr(verifier, name):
                delattr(verifier, name)
        sys.modules.pop(module_name, None)


def test_context_state_callsite_unpacks_into_original_frozen_q_and_rejects_invalid_state(monkeypatch):
    wrapper = load_wrapper()
    runner = wrapper.load_runner()
    pilot = runner.load_frozen_driver()
    freeze = json.loads((ROOT / "reports/simplification_v1/CLS_POS_EF_CONTEXT_V1/FREEZE.json").read_text(encoding="utf-8"))
    runner.load_audit_core(pilot, freeze)
    module_name = "scripts.verify_cls_pos_ef_context_v1"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    verifier, _, callsite = wrapper.verified_verifier_module(runner, pilot, freeze)
    try:
        assert callsite["delegates_to_frozen_function"] is True
        assert callsite["mapping_keys"] == ["mu", "scale"]
        shim = verifier._independent_q
        original = shim.__wrapped__
        ef = np.asarray([0, 1, 0, 1], dtype=np.int8)
        state = {"mu": 0.25, "scale": 0.5}
        np.testing.assert_array_equal(shim(ef, state), original(ef, state["mu"], state["scale"]))
        np.testing.assert_array_equal(shim(ef, state["mu"], state["scale"]), original(ef, state["mu"], state["scale"]))
        with pytest.raises(pilot.PilotError, match="unexpected argument shape"):
            shim(ef, {"mu": 0.25})
        with pytest.raises(pilot.PilotError, match="unexpected argument shape"):
            shim(ef, state, 0.5)
    finally:
        verifier._independent_q = verifier._independent_q.__wrapped__
        for name in FROZEN_HELPERS:
            if hasattr(verifier, name):
                delattr(verifier, name)
        sys.modules.pop(module_name, None)
