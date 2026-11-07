from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from scripts import cls_dan_v1 as runner
from scripts import cls_dan_v1_evaluation_recovery_r2 as recovery


def test_present_only_coverage_expands_with_missing_zero():
    record = {
        "present": np.asarray([True, False, True, False]),
        "coverage": np.asarray([0.25, 0.75], dtype=np.float64),
        "score_rows": np.arange(4),
    }
    expanded = recovery.expand_present_only_coverage(record)
    assert expanded["coverage"].tolist() == [0.25, 0.0, 0.75, 0.0]
    assert record["coverage"].tolist() == [0.25, 0.75]
    assert expanded["coverage"] is not record["coverage"]


@pytest.mark.parametrize(
    "coverage",
    [np.asarray([0.1, 0.2, 0.3]), np.asarray([np.nan]), np.asarray([-0.1])],
)
def test_present_only_coverage_rejects_bad_shape_or_values(coverage):
    with pytest.raises(RuntimeError):
        recovery.expand_present_only_coverage({
            "present": np.asarray([True, False]),
            "coverage": coverage,
        })


def test_full_synthetic_scorer_covers_six_views_and_supported_unsupported_groups():
    raw, labels = recovery._synthetic_arrays()
    expanded = {origin: recovery.expand_present_only_coverage(record)
                for origin, record in raw.items()}
    prediction_ready = {
        "freeze_signature": "SYNTHETIC_ONLY", "source_commit": "synthetic",
        "origins": {str(origin): {"missing_route_exact_for_dan": True,
                                  "missing_route_exact_for_linear": True}
                    for origin in runner.ORIGINS},
    }
    result, verification = recovery.evaluate_prediction_arrays(expanded, labels, prediction_ready)
    assert {name: row["n"] for name, row in result["views"].items()} == recovery.VIEW_ROWS
    assert verification["status"] == "PASS"
    assert result["qa"]["independent_metrics_match"] is True
    assert result["qa"]["all_views_have_supported_subgroup"] is True
    assert any(row["status"] == "UNSUPPORTED" for row in result["subgroups"])
    assert any(row["status"] == "SUPPORTED" for row in result["subgroups"])
    assert result["qa"]["missing_coverage_zero"] is True


def test_original_and_amended_e75_gate_components_are_distinct():
    y = np.asarray([0, 0, 1, 1], dtype=np.int8)
    present = np.ones(4, dtype=bool)
    metrics = {
        "champion": runner._metrics(y, np.asarray([0.1, 0.1, 0.9, 0.9]), present),
        "dan": runner._metrics(y, np.asarray([0.1, 0.1, 0.1, 0.9]), present),
        "linear": runner._metrics(y, np.asarray([0.1, 0.9, 0.1, 0.1]), present),
    }
    dan_minus_linear = metrics["dan"]["macro_f1"] - metrics["linear"]["macro_f1"]
    linear_minus_champion = metrics["linear"]["macro_f1"] - metrics["champion"]["macro_f1"]
    assert dan_minus_linear > 0.0
    assert linear_minus_champion < 0.0
    views = {
        "PRIMARY_UNIQUE": {"dan_gain_vs_champion": 0.001, "dan_minus_linear": 0.001},
        "E67_H12": {"dan_gain_vs_champion": 0.001},
        "E75_W6": {
            "dan_gain_vs_champion": 0.001,
            "linear_gain_vs_champion": linear_minus_champion,
            "dan_minus_linear": dan_minus_linear,
        },
    }
    supported = [
        {"view": name, "status": "SUPPORTED", "dan_macro_f1_harm_vs_champion": 0.0,
         "dan_positive_recall_harm_vs_champion": 0.0}
        for name in recovery.VIEW_ROWS
    ]
    gates = recovery.gate_verdicts(views, supported, qa_pass=True)
    assert gates["scientific_original_frozen"]["components"]["E75_LINEAR_control_gain_nonnegative"] is False
    assert gates["scientific_r2_amended"]["components"]["E75_DAN_minus_LINEAR_nonnegative_R2_amendment"] is True
    assert gates["exploratory_eligibility_original_rule"]["components"][
        "campaign_primary_or_E75_performance_rule"] is True


def test_synthetic_cli_persistence_and_no_retry_guard(tmp_path, monkeypatch, capsys):
    output = tmp_path / "synthetic-r2"
    calls = []
    original_reader = runner._read_exact_targets

    def tracked_reader(*args, **kwargs):
        calls.append(1)
        assert (output / "R2_CONSUMED.json").is_file()
        return original_reader(*args, **kwargs)

    monkeypatch.setattr(runner, "_read_exact_targets", tracked_reader)
    assert recovery.main(["synthetic", "--output-dir", str(output)]) == 0
    completed = json.loads(capsys.readouterr().out)
    assert completed["status"] == "PASS_SYNTHETIC_R2_CLI_QA"
    assert completed["target_read_calls"] == 1
    assert completed["views"] == recovery.VIEW_ROWS
    assert calls == [1]
    result = json.loads((output / "RESULTS.json").read_text(encoding="utf-8"))
    assert result["synthetic_qa"]["target_read_calls"] == 1
    assert result["synthetic_qa"]["actual_query_labels_read"] is False
    output_hashes = {
        name: hashlib.sha256((output / name).read_bytes()).hexdigest()
        for name in ("RESULTS.json", "INDEPENDENT_VERIFICATION.json", "R2_COMPLETED.json")
    }
    with pytest.raises(RuntimeError, match="persistence guard prevents retry"):
        recovery.main(["synthetic", "--output-dir", str(output)])
    assert calls == [1]
    assert output_hashes == {
        name: hashlib.sha256((output / name).read_bytes()).hexdigest()
        for name in output_hashes
    }
