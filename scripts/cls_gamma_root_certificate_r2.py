"""Independent zero-optimizer certificate; preserves all failed-run artifacts."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy import sparse
from scipy.special import expit

C = 1.1859646641031965
# Zero-optimizer verification: observed original fit RSS rose from 171,106,304
# to a lifetime peak of 557,031,424 bytes. 512 MiB incremental includes >120 MiB
# headroom beyond that observed delta; keep the original 1 GiB actual reserve.
VERIFY_INCREMENTAL_BYTES = 512 * 1024**2
VERIFY_RESERVE_BYTES = 1024**3
RUN = "experiments/runs/optimization_v3_cls_pos_global_strength_e79_v3/gamma_v1"
PINS = {
    "gamma.npz": "4bc240bb6b53eb623f5f099246b70572f987b8c3b0f8dfc123f49029d3c079ab",
    "fit_receipt.json": "05058193fb2cfb32e70d4d2d939fa3abf75b3e1aa77a947eda0a9ccddfb70566",
    "ATTEMPT.started.json": "6d8528b43f0ced71e5cfdbbf0a3369ab08a38012d8acb363869f69f374a703c9",
    "recovery_v1/RECOVERY_RECEIPT.json": "49c17d01a3698684a8bba0206f59a6b0fc72bb2b129db233e1b2b79c02091a25",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numerical_certificate(x, offset, y, coef):
    z = offset + np.asarray(x @ coef).reshape(-1)
    signed = 2.0 * y - 1.0
    penalty = 0.5 * C * float(coef @ coef)
    loss = float(np.logaddexp(0.0, -signed * z).sum() + penalty)
    independent_loss = math.fsum(
        max(float(zz), 0.0) - float(yy) * float(zz)
        + math.log1p(math.exp(-abs(float(zz)))) for zz, yy in zip(z, y)
    ) + penalty
    grad = np.asarray(x.T @ (expit(z) - y)).reshape(-1) + C * coef
    independent_grad = np.asarray(x.T @ (-signed * expit(-signed * z))).reshape(-1) + C * coef
    assert math.isclose(loss, independent_loss, rel_tol=1e-12, abs_tol=1e-8)
    np.testing.assert_allclose(grad, independent_grad, rtol=1e-8, atol=1e-10)
    assert np.isfinite(grad).all() and np.isfinite(loss)
    zero_loss = float(np.logaddexp(0.0, -signed * offset).sum())
    bound = max(float(grad @ grad), float(independent_grad @ independent_grad)) / (2.0 * C)
    threshold = max(1e-4, 1e-8 * abs(loss))
    return {
        "loss_final": loss, "loss_zero": zero_loss,
        "loss_independent": independent_loss,
        "gradient_l2": float(np.linalg.norm(grad)),
        "gradient_linf": float(np.max(np.abs(grad))),
        "strong_convexity_mu": C, "objective_gap_upper_bound": bound,
        "fixed_acceptance_threshold": threshold,
        "relative_gap_bound": bound / max(abs(loss), 1e-300),
        "certificate_pass": bool(bound <= threshold and loss <= zero_loss + 1e-8),
    }


def self_test():
    x = sparse.csr_matrix([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    y = np.array([1.0, 0.0, 1.0]); offset = np.array([0.1, -0.2, 0.3])
    result = numerical_certificate(x, offset, y, np.array([0.2, -0.1]))
    assert not result["certificate_pass"]
    zero = sparse.csr_matrix((3, 2))
    result = numerical_certificate(zero, offset, y, np.zeros(2))
    assert result["certificate_pass"] and result["objective_gap_upper_bound"] == 0.0
    print(json.dumps({"status": "PASS_SYNTHETIC_ZERO_OPTIMIZER_CERTIFICATE_TESTS"}))


def audit(repository: Path, output: Path):
    assert not output.exists(), "Exclusive output required"
    this_root = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(["git", "-C", str(this_root), "rev-parse", "HEAD"], text=True).strip()
    blob = subprocess.check_output(["git", "-C", str(this_root), "show", f"{commit}:scripts/{Path(__file__).name}"])
    assert blob.replace(b"\r\n", b"\n") == Path(__file__).read_bytes().replace(b"\r\n", b"\n")
    root = repository.resolve(); paths = {name: root / RUN / name for name in PINS}
    assert {name: sha(path) for name, path in paths.items()} == PINS
    sys.path.insert(0, str(root))
    from scripts import optimization_v3_cls_pos_global_strength_e79_gamma_v1 as g
    from src.models import pos_frequency_ridge_v1 as frequency
    from src.models import pos_one_sided_ridge_v1 as one_sided
    import scipy.optimize
    calls = {"forbidden_fit_or_optimizer": 0}

    def deny(*args, **kwargs):
        calls["forbidden_fit_or_optimizer"] += 1
        raise RuntimeError("Optimizer/fit forbidden in certificate verification")

    scipy.optimize.minimize = frequency.minimize = deny
    frequency.fit_weighted_offset = one_sided.fit_offset = deny
    g._fit_arrays = g._run_official_fit = g.recipe.fit_global_strength_gamma = deny
    source = g.verify_source_freeze(root)
    with ExitStack() as stack:
        locks = [stack.enter_context(g.r3._shared_job_lock()), stack.enter_context(g.r3._cgeom_local_lock(root))]
        before = g.resource_snapshot("root R2 verification before rebuild")
        assert before["available_ram_bytes"] >= VERIFY_INCREMENTAL_BYTES + VERIFY_RESERVE_BYTES, "Verification resource gate before any target read"
        built = g._build_train_features(root)
        after_features = g.resource_snapshot("root R2 after exact feature rebuild")
        assert after_features["available_ram_bytes"] >= g.FREE_RESERVE_BYTES
        y = g._load_train_target_last(root)
        with np.load(paths["gamma.npz"], allow_pickle=False) as archive:
            assert archive.files == ["gamma_float64"]
            coef = archive["gamma_float64"].copy()
        assert coef.dtype == np.float64 and coef.shape == (17259,) and np.isfinite(coef).all()
        assert built["global_strength_scalar"] == C
        result = numerical_certificate(built["position"], built["offset"], y.astype(float), coef)
        after = g.resource_snapshot("root R2 after zero-fit certificate")
        assert after["available_ram_bytes"] >= g.FREE_RESERVE_BYTES
    assert calls["forbidden_fit_or_optimizer"] == 0
    assert {name: sha(path) for name, path in paths.items()} == PINS
    result.update({
        "status": "RECOVERED_ARTIFACT_NUMERICALLY_VERIFIED" if result["certificate_pass"] else "NUMERICAL_CERTIFICATE_FAIL",
        "audit_source_commit": commit, "audit_script_sha256": sha(Path(__file__)),
        "source_freeze": source, "preserved_original_hashes": PINS,
        "train_rows": len(y), "matrix_metadata": built["matrices"],
        "target_sha256": g.positional.sha256_array(y),
        "offset_sha256": g.positional.sha256_array(built["offset"]),
        "gamma_logical_sha256": g.positional.sha256_array(coef),
        "original_optimizer_metadata": "UNKNOWN; acceptance inferred only from pinned control flow",
        "new_predictor_fits": 0, "new_preprocessing_fits": 0,
        "guard_calls": calls, "test_csv_or_kaggle_access": False,
        "original_failure_status_unchanged": True,
        "resources": {"before": before, "after_features": after_features, "after": after},
        "verification_resource_policy": {
            "incremental_estimate_bytes": VERIFY_INCREMENTAL_BYTES,
            "actual_free_reserve_bytes": VERIFY_RESERVE_BYTES,
            "basis": "original full fit measured RSS delta 385925120 bytes; no optimizer in this verification",
            "inherited_training_gate_is_not_the_verification_gate": True,
            "prior_root_command_blocked_before_target_read": True,
        },
        "locks": locks,
    })
    payload = json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    print(json.dumps({k: result[k] for k in ("status", "loss_final", "gradient_linf", "objective_gap_upper_bound", "fixed_acceptance_threshold")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--repository", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        assert args.repository and args.output
        audit(args.repository, args.output)
