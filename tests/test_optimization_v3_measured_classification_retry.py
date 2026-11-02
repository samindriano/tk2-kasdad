from __future__ import annotations

import numpy as np

from scripts.optimization_v3_measured_classification_retry import (
    TASK, run_synthetic_preflight, synthetic_payload,
)
from src.models.research_next_contract import partition_payload, validate_arrays


def test_class_only_bundle_contains_contract_baseline_prediction() -> None:
    payload, fit_calls = synthetic_payload()
    arrays, metadata = partition_payload(payload)
    validate_arrays(arrays, task="classification")
    assert "baseline_prediction" in payload
    assert payload["baseline_prediction"].shape == payload["prediction"].shape
    assert np.isfinite(payload["baseline_prediction"]).all()
    assert payload["metadata"]["synthetic_only"] is True
    assert metadata["metadata"]["synthetic_only"] is True
    missing = payload["stream_missing"]
    assert np.array_equal(payload["prediction"][missing], np.linspace(0.1, 0.8, 8)[missing])
    assert fit_calls == 1


def test_synthetic_retry_bundle_persists_and_resumes(tmp_path, monkeypatch) -> None:
    import scripts.optimization_v3_measured_classification_retry as retry

    report_path = tmp_path / "synthetic_preflight.json"
    monkeypatch.setattr(retry, "SYNTHETIC_REPORT", report_path)
    report = run_synthetic_preflight()
    assert report["status"] == "PASS"
    assert report["bundle_persist_readback_resume"] == "PASS"
    assert report["post_fit_failure_injection_then_payload_recovery"] == "PASS"
    assert report["synthetic_fit_calls"] == 1
    assert report["scope"].startswith("synthetic estimator only")
    assert TASK == "MEASURED_CLASSIFICATION_BENCHMARK_V1"
