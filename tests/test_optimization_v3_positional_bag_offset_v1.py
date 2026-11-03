from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy import sparse
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import optimization_v3_positional_bag_offset_v1 as runner
from verify_optimization_v3_positional_bag_offset_v1 import verify_synthetic_bundle
from src.models.positional_bag_offset_v1 import (
    fit_offset_coefficients,
    offset_objective_gradient,
    offset_probability,
)


def test_gradient_matches_finite_difference_with_nonzero_offset_and_l2() -> None:
    x = sparse.csr_matrix(np.asarray([
        [1.0, 0.0, 0.2], [0.0, 1.0, -0.4], [1.0, 1.0, 0.0],
        [-0.3, 0.0, 0.7], [0.0, -0.5, 0.2], [0.1, 0.4, 0.8],
    ], dtype=np.float64))
    z0 = np.asarray([-1.1, 0.4, 1.2, -0.2, 0.9, -0.7], dtype=np.float64)
    y = np.asarray([0, 1, 1, 0, 1, 0], dtype=np.int8)
    gamma = np.asarray([0.2, -0.35, 0.1], dtype=np.float64)
    loss, grad = offset_objective_gradient(gamma, x, z0, y, c=0.8)
    eps = 1e-6
    numeric = np.empty_like(gamma)
    for j in range(len(gamma)):
        plus, minus = gamma.copy(), gamma.copy()
        plus[j] += eps
        minus[j] -= eps
        numeric[j] = (offset_objective_gradient(plus, x, z0, y, c=0.8)[0]
                      - offset_objective_gradient(minus, x, z0, y, c=0.8)[0]) / (2 * eps)
    assert np.isfinite(loss)
    assert np.allclose(grad, numeric, rtol=2e-6, atol=2e-7)


def test_extreme_logits_are_finite() -> None:
    x = sparse.csr_matrix(np.asarray([[1.0], [-1.0], [0.5], [-0.5]], dtype=np.float64))
    z0 = np.asarray([10000.0, -10000.0, 10000.0, -10000.0])
    y = np.asarray([0, 1, 1, 0], dtype=np.int8)
    loss, grad = offset_objective_gradient(np.asarray([0.01]), x, z0, y)
    probability = offset_probability(x, z0, np.asarray([0.01]))
    assert np.isfinite(loss)
    assert np.isfinite(grad).all()
    assert np.isfinite(probability).all()
    assert np.all((probability >= 0) & (probability <= 1))


def test_empty_position_block_preserves_zero_gamma_and_bag_probabilities() -> None:
    x = sparse.csr_matrix((6, 0), dtype=np.float64)
    z0 = np.asarray([-4.0, -1.0, 0.0, 0.5, 1.0, 5.0])
    y = np.asarray([0, 0, 1, 0, 1, 1], dtype=np.int8)
    gamma, meta = fit_offset_coefficients(x, z0, y)
    assert gamma.shape == (0,)
    assert meta["empty_block"] is True
    assert np.array_equal(offset_probability(x, z0, gamma), expit(z0))


def test_zero_gamma_on_nonempty_position_block_reproduces_fixed_bag_probability() -> None:
    x = sparse.csr_matrix(np.asarray([[1.0, 0.0], [0.0, 2.0], [0.3, -0.7]], dtype=np.float64))
    z0 = np.asarray([-2.2, 0.4, 1.8], dtype=np.float64)
    assert np.allclose(offset_probability(x, z0, np.zeros(x.shape[1])), expit(z0), rtol=0, atol=1e-15)


def test_sparse_optimizer_converges_from_zero_and_does_not_increase_loss() -> None:
    rng = np.random.default_rng(812)
    x = sparse.csr_matrix(rng.normal(size=(180, 12)))
    truth = rng.normal(scale=0.15, size=12)
    z0 = rng.normal(size=180)
    p = 1.0 / (1.0 + np.exp(-(z0 + x @ truth)))
    y = rng.binomial(1, p).astype(np.int8)
    gamma, meta = fit_offset_coefficients(x, z0, y)
    assert meta["success"] is True
    assert meta["loss_final"] <= meta["loss_gamma0"] + 1e-8
    assert np.isfinite(gamma).all()
    assert np.isfinite(meta["gradient_inf_norm"])


def test_synthetic_end_to_end_persistence_resume_and_report_uses_temp_ledger(tmp_path: Path) -> None:
    before = runner.sha256_file(runner.LEDGER)
    try:
        result = runner.run_synthetic_lifecycle(tmp_path / "synthetic")
        assert result["status"] == "PASS"
        assert result["temporary_ledger_events"] > 200
        assert result["independent"]["optimizer_called"] is False
        events = runner._ledger_events(tmp_path / "synthetic" / "synthetic_ledger.jsonl")
        assert events[-1]["event"] == "CARD_COMPLETED"
        assert [x["event"] for x in events if x.get("fit_id") == runner.FIT_IDS[0]] == ["FIT_STARTED", "FIT_SAVED"]
        assert result["ledger_snapshot"]["active_reservations"] == 109
    finally:
        assert runner.sha256_file(runner.LEDGER) == before


def test_production_ledger_writer_fails_closed_for_synthetic_path() -> None:
    before = runner.sha256_file(runner.LEDGER)
    try:
        with pytest.raises(runner.OffsetPilotError, match="Production ledger writes"):
            runner._append_ledger({"event": "SHOULD_NOT_WRITE"}, path=runner.LEDGER)
    finally:
        assert runner.sha256_file(runner.LEDGER) == before


def test_fit_label_reader_never_converts_query_target_values(tmp_path: Path) -> None:
    path = tmp_path / "selected_labels.csv"
    path.write_text("record_id,artifact_signal,feature\nfit-row,1,a\nquery-row,NOT_A_BINARY_TARGET,b\n",
                    encoding="utf-8")
    y = runner._fit_labels(path, np.asarray([0], dtype=np.int64), np.asarray(["fit-row"]),
                           runner.sha256_array(np.asarray([1], dtype=np.int8)))
    independent = __import__("verify_optimization_v3_positional_bag_offset_v1")._select_fit_labels(
        path, np.asarray([0], dtype=np.int64), np.asarray(["fit-row"]))
    assert np.array_equal(y, np.asarray([1], dtype=np.int8))
    assert np.array_equal(independent, y)


def test_production_resume_helper_repairs_durable_fit_without_reoptimizing(tmp_path: Path) -> None:
    ledger = tmp_path / "resume.jsonl"
    fit_id = runner.FIT_IDS[0]
    freeze = {"freeze_signature": "resume-freeze"}
    runner._append_ledger({"event": "FIT_STARTED", "fit_id": fit_id}, path=ledger)
    artifact = tmp_path / "gamma.bin"
    runner._atomic_bytes(artifact, b"frozen synthetic gamma")
    receipt = tmp_path / "receipt.json"
    runner._atomic_json(receipt, {"fit_id": fit_id, "freeze_signature": "resume-freeze",
                                  "artifact_sha256": runner.sha256_file(artifact), "origin": 67})
    status = runner._resume_started_fit(
        fit_id, freeze, runner._ledger_events(ledger), ledger_path=ledger,
        artifact_path=artifact, receipt_path=receipt, official=False,
    )
    assert status == "resumed_from_durable_receipt"
    status2 = runner._resume_started_fit(
        fit_id, freeze, runner._ledger_events(ledger), ledger_path=ledger,
        artifact_path=artifact, receipt_path=receipt, official=False,
    )
    assert status2 == "already_saved"
    assert [e["event"] for e in runner._ledger_events(ledger)] == ["FIT_STARTED", "FIT_SAVED"]


def test_fit_lock_allows_only_one_active_runner(tmp_path: Path) -> None:
    lock = tmp_path / "operation.lock"
    with runner._fit_run_lock(lock):
        with pytest.raises(runner.OffsetPilotError, match="local fit lock"):
            with runner._fit_run_lock(lock):
                pass
        child_code = (
            "import sys; from pathlib import Path; root=Path.cwd(); "
            "sys.path.insert(0,str(root)); sys.path.insert(0,str(root/'scripts')); "
            "import optimization_v3_positional_bag_offset_v1 as r\n"
            f"with r._fit_run_lock(Path({str(lock)!r})):\n pass\n"
        )
        child = subprocess.run([sys.executable, "-c", child_code], cwd=ROOT,
                               text=True, capture_output=True, timeout=15)
        assert child.returncode != 0
        assert "kernel lock" in child.stderr or "kernel lock" in child.stdout
    assert lock.is_file()
    with runner._fit_run_lock(lock):
        pass


def test_prelabel_pass_is_bound_to_current_ready_manifest_and_outputs(tmp_path: Path) -> None:
    output = tmp_path / "E67.npz"
    output.write_bytes(b"prediction bytes")
    ready_path = tmp_path / "PREDICTIONS_READY.json"
    ready = {"query_labels_opened": False,
             "predictions": {"E67": {"path": "E67.npz"}, "E75": {"path": "E67.npz"}}}
    runner._atomic_json(ready_path, ready)
    body = {"status": "PASS", "freeze_signature": "frozen",
            "predictions_ready_sha256": runner.sha256_file(ready_path),
            "prediction_output_sha256s": {"E67": runner.sha256_file(output),
                                          "E75": runner.sha256_file(output)}}
    verified = {**body, "verification_signature": runner.canonical_sha256(body)}
    assert runner._assert_prelabel_binding(verified, freeze_signature="frozen",
                                           ready_path=ready_path, root=tmp_path)["ready"] == ready
    output.write_bytes(b"changed prediction bytes")
    with pytest.raises(runner.OffsetPilotError, match="current prelabel PASS"):
        runner._assert_prelabel_binding(verified, freeze_signature="frozen",
                                        ready_path=ready_path, root=tmp_path)


def test_independent_verifier_rejects_tampered_synthetic_row_id(tmp_path: Path) -> None:
    runner.run_synthetic_lifecycle(tmp_path / "original")
    root = tmp_path / "original"
    artifact = root / "SYNTHETIC_A.npz"
    receipt_path = root / "SYNTHETIC_A.json"
    with np.load(artifact, allow_pickle=False) as saved:
        arrays = {k: saved[k].copy() for k in saved.files}
    arrays["score_ids"][0] = "tampered"
    tampered_artifact = root / "TAMPERED.npz"
    runner._atomic_npz(tampered_artifact, **arrays)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["artifact_sha256"] = runner.sha256_file(tampered_artifact)
    tampered_receipt = root / "TAMPERED.json"
    runner._atomic_json(tampered_receipt, receipt)
    with pytest.raises(AssertionError, match="row ID or frozen bag parameter hash"):
        verify_synthetic_bundle(tampered_artifact, tampered_receipt)


def test_synthetic_class_order_and_exact_fallback_are_verified(tmp_path: Path) -> None:
    runner.run_synthetic_lifecycle(tmp_path / "original")
    root = tmp_path / "original"
    artifact = root / "SYNTHETIC_A.npz"
    receipt_path = root / "SYNTHETIC_A.json"
    with np.load(artifact, allow_pickle=False) as saved:
        arrays = {k: saved[k].copy() for k in saved.files}
    arrays["classes"] = np.asarray([1, 0], dtype=np.int8)
    tampered = root / "CLASS_ORDER.npz"
    runner._atomic_npz(tampered, **arrays)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["artifact_sha256"] = runner.sha256_file(tampered)
    receipt2 = root / "CLASS_ORDER.json"
    runner._atomic_json(receipt2, receipt)
    with pytest.raises(AssertionError, match="class order"):
        verify_synthetic_bundle(tampered, receipt2)
