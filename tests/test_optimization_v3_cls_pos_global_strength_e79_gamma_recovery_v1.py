from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from scipy import sparse

from scripts import audit_optimization_v3_cls_pos_global_strength_e79_gamma_recovery_v1 as audit
from scripts import verify_optimization_v3_cls_pos_global_strength_e79_gamma_recovery_v1 as verifier


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_FAILURE = ROOT / "experiments/runs/optimization_v3_cls_pos_global_strength_e79_v3/gamma_v1/fit_receipt.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, *args], cwd=ROOT, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def _synthetic():
    x = sparse.csr_matrix(np.asarray([[1., 0.], [1., 0.], [0., 1.], [0., 1.]], dtype=np.float64))
    offset = np.zeros(4, dtype=np.float64)
    y = np.asarray([0, 1, 0, 1], dtype=np.int8)
    return x, offset, y


def test_exact_synthetic_optimum_certificate_and_independent_gradient() -> None:
    x, offset, y = _synthetic()
    gamma = np.zeros(2, dtype=np.float64)
    result = audit.certify_arrays(gamma=gamma, x=x, offset=offset, y=y, c=1.0)
    assert result["status"] == "PASS_NUMERICAL_CERTIFICATE"
    assert result["absolute_excess_objective_bound"] == 0.0
    assert all(result["checks"].values())

    candidate = np.asarray([0.07, -0.03], dtype=np.float64)
    formula_loss, formula_grad = audit._independent_objective_gradient(candidate, x, offset, y, 1.0)
    eps = 1e-6
    fd = np.empty_like(candidate)
    for j in range(len(candidate)):
        step = np.zeros_like(candidate)
        step[j] = eps
        plus = audit._independent_objective_gradient(candidate + step, x, offset, y, 1.0)[0]
        minus = audit._independent_objective_gradient(candidate - step, x, offset, y, 1.0)[0]
        fd[j] = (plus - minus) / (2.0 * eps)
    assert np.isfinite(formula_loss)
    np.testing.assert_allclose(formula_grad, fd, rtol=1e-7, atol=1e-8)


def test_mismatched_gamma_binding_fails_closed() -> None:
    x, offset, y = _synthetic()
    gamma = np.zeros(2, dtype=np.float64)
    try:
        audit.certify_arrays(gamma=gamma, x=x, offset=offset, y=y, c=1.0,
                             expected_gamma_sha256="0" * 64)
    except audit.RecoveryAuditError as exc:
        assert "gamma array content hash differs" in str(exc).lower()
    else:
        raise AssertionError("mismatched gamma hash was accepted")


def test_mismatched_input_binding_fails_closed() -> None:
    try:
        audit.require_exact_hash_bindings({"train": "a" * 64, "matrix": "b" * 64},
                                          {"train": "a" * 64, "matrix": "c" * 64})
    except audit.RecoveryAuditError as exc:
        assert "matrix" in str(exc)
    else:
        raise AssertionError("mismatched source matrix hash was accepted")


def test_failed_certificate_is_not_rescued() -> None:
    x = sparse.csr_matrix(np.asarray([[1.0]], dtype=np.float64))
    result = audit.certify_arrays(gamma=np.asarray([0.1]), x=x,
                                  offset=np.asarray([0.0]), y=np.asarray([1]), c=1.0)
    assert result["loss_delta_from_gamma0"] < 0.0
    assert result["absolute_excess_objective_bound"] > result["absolute_bound_threshold"]
    assert result["checks"]["loss_nonincrease_vs_gamma0"] is True
    assert result["checks"]["absolute_strong_convexity_certificate"] is False
    assert result["status"] == "FAIL_NUMERICAL_CERTIFICATE"


def test_source_control_flow_is_labeled_inference() -> None:
    evidence = audit.verify_source_control_flow(ROOT)
    assert evidence["status"] == "PASS_SOURCE_CONTROL_FLOW_INFERENCE"
    assert evidence["interpretation"] == "inference_from_frozen_source_control_flow_only"
    assert evidence["optimizer_nit_nfev_message_loss_gradient"] == "UNKNOWN_NOT_PERSISTED"


def test_synthetic_cli_persistence_resume_tamper_and_original_failure_preservation(tmp_path: Path) -> None:
    failure_sha_before = _sha(ORIGINAL_FAILURE)
    output = tmp_path / "recovery"
    run = _run("-m", "scripts.audit_optimization_v3_cls_pos_global_strength_e79_gamma_recovery_v1",
               "--synthetic-e2e", "--output-dir", str(output))
    assert run.returncode == 0, run.stdout + run.stderr
    result = json.loads(run.stdout)
    assert result["status"] == "PASS_SYNTHETIC_ZERO_FIT_RECOVERY_E2E"
    assert all(value == 0 for value in result["optimizer_calls"].values())

    verify_one = _run("-m", "scripts.verify_optimization_v3_cls_pos_global_strength_e79_gamma_recovery_v1",
                       "--verify-synthetic", "--output-dir", str(output))
    verify_two = _run("-m", "scripts.verify_optimization_v3_cls_pos_global_strength_e79_gamma_recovery_v1",
                       "--verify-synthetic", "--output-dir", str(output))
    assert verify_one.returncode == verify_two.returncode == 0
    check_one, check_two = json.loads(verify_one.stdout), json.loads(verify_two.stdout)
    assert check_one["verification_sha256"] == check_two["verification_sha256"]
    assert check_one["status"] == "PASS_INDEPENDENT_SYNTHETIC_RECOVERY_READBACK"

    bundle_before = {path.name: _sha(path) for path in output.iterdir() if path.is_file()}
    retry = _run("-m", "scripts.audit_optimization_v3_cls_pos_global_strength_e79_gamma_recovery_v1",
                 "--synthetic-e2e", "--output-dir", str(output))
    assert retry.returncode == 1
    assert {path.name: _sha(path) for path in output.iterdir() if path.is_file()} == bundle_before

    tampered = tmp_path / "tampered"
    shutil.copytree(output, tampered)
    with (tampered / "gamma.npz").open("wb") as stream:
        np.savez_compressed(stream, gamma_float64=np.asarray([0.2, 0.0], dtype=np.float64))
    try:
        verifier.verify_synthetic(tampered)
    except verifier.VerificationError:
        pass
    else:
        raise AssertionError("tampered synthetic gamma artifact was accepted")

    assert _sha(ORIGINAL_FAILURE) == failure_sha_before
    assert failure_sha_before == audit.EXPECTED_FAILURE_RECEIPT_SHA256


def test_deny_guard_blocks_every_optimizer_and_fit_entry_point() -> None:
    with audit.deny_fit_calls() as counters:
        assert all(value == 0 for value in counters.values())
        calls = (
            ("scipy.optimize.minimize", lambda: __import__("scipy.optimize", fromlist=["minimize"]).minimize(None)),
            ("frequency_ridge.minimize", lambda: audit.frequency_ridge.minimize(None)),
            ("frequency_ridge.fit_weighted_offset", lambda: audit.frequency_ridge.fit_weighted_offset(None, None, None, None)),
            ("one_sided.fit_offset", lambda: audit.one_sided.fit_offset(None, None, None, None)),
            ("gamma_recipe.fit_global_strength_gamma", lambda: audit.gamma_recipe.fit_global_strength_gamma(None, None, None, None)),
            ("gamma_runner._fit_arrays", lambda: audit.gamma_runner._fit_arrays(None, None, None, None)),
        )
        for name, call in calls:
            try:
                call()
            except audit.RecoveryAuditError:
                pass
            assert counters[name] == 1
