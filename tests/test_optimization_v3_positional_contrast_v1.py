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
