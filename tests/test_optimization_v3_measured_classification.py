from __future__ import annotations

from scripts.optimization_v3_measured_classification import _supported_groups_pass, _synthetic_preflight


def test_measured_n02_synthetic_end_to_end_and_bundle_resume():
    result = _synthetic_preflight()
    assert result["status"] == "PASS"
    assert result["official_data_used"] is False
    assert result["test_data_used"] is False
    assert result["bundle_persist_readback_resume"] == "PASS"


def test_measured_benchmark_recall_amendment_is_explicit_and_bounded():
    within_exception = [{"supported": True, "f1_guard_fail": False,
                         "benchmark_0_03_recall_guard_fail": False,
                         "original_0_02_recall_guard_fail": True}]
    beyond_exception = [{"supported": True, "f1_guard_fail": False,
                         "benchmark_0_03_recall_guard_fail": True,
                         "original_0_02_recall_guard_fail": True}]
    unsupported_failure = [{"supported": False, "f1_guard_fail": True,
                            "benchmark_0_03_recall_guard_fail": True}]
    assert _supported_groups_pass(within_exception)
    assert not _supported_groups_pass(beyond_exception)
    assert _supported_groups_pass(unsupported_failure)
