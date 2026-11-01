from __future__ import annotations

import json

import pytest

import scripts.optimization_v3_research_next_n01_class_only_diagnostic as runner
from src.models.research_next_n01_class_only_diagnostic import N01ClassOnlyDiagnosticClassifier


def test_adapter_permits_only_four_class_only_cutoffs():
    for cutoff in (59, 63, 71, 75):
        model = N01ClassOnlyDiagnosticClassifier(cutoff)
        assert model.cutoff == cutoff
        assert model.weight_kind == "class_only"
    for cutoff in (55, 67, 76):
        with pytest.raises(ValueError):
            N01ClassOnlyDiagnosticClassifier(cutoff)
    with pytest.raises(ValueError):
        N01ClassOnlyDiagnosticClassifier(59, weight_kind="class_by_volume")


def test_synthetic_fit_publish_read_resume_metrics_and_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "SYNTHETIC_RECEIPT_PATH", tmp_path / "synthetic_preflight.json")
    receipt = runner.synthetic_preflight()
    assert receipt["status"] == "PASS_SYNTHETIC_E2E"
    assert receipt["estimator_fit_calls"] == 1
    assert receipt["resumed_existing_bundle_without_refit"] is True
    assert receipt["official_fits"] == 0
    assert receipt["subgroup_guard_rows"] >= 2
    assert receipt["fit_only_preprocessing"] is True
    assert (tmp_path / "synthetic_preflight.json").is_file()


def test_synthetic_receipt_rejects_tampered_source_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "SYNTHETIC_RECEIPT_PATH", tmp_path / "synthetic_preflight.json")
    runner.SYNTHETIC_RECEIPT_PATH.write_text(json.dumps({
        "status": "PASS_SYNTHETIC_E2E", "official_fits": 0,
        "test_or_kaggle_used": False, "resumed_existing_bundle_without_refit": True,
        "source_sha256": {"src/models/research_next_n01.py": "0" * 64},
    }), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Synthetic preflight receipt source changed"):
        runner._read_synthetic_receipt({"source_hashes": {"src/models/research_next_n01.py": "1" * 64}})
