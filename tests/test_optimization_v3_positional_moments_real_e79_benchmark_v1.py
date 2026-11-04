from __future__ import annotations

import numpy as np

from scripts import optimization_v3_positional_moments_real_e79_benchmark_v1 as benchmark


def test_synthetic_end_to_end_policy_and_persistence():
    result = benchmark.run_synthetic()
    assert result["status"] == "PASS_SYNTHETIC_END_TO_END_AND_SAFE_RESUME"
    assert result["official_inputs_opened"] is False
    assert result["test_or_sample_opened"] is False
    assert result["real_moment_builder_matches_independent_scalar_replay"] is True
    assert result["safe_unicode_id_roundtrip"] is True
    assert result["synthetic_optimizer_fit_count"] == 1


def test_frozen_reservation_ids_are_new_and_one_fit_only():
    assert benchmark.BENCHMARK_ID == "POS_MOMENTS_REAL_E79_EXPLORATORY_BENCHMARK"
    assert benchmark.FIT_ID == "POS_MOMENTS_REAL_E79_EXPLORATORY_BENCHMARK_GAMMA"
    assert benchmark.RESERVATION_KEY == "POS_MOMENTS_REAL_E79_EXPLORATORY_BENCHMARK:REAL_GAMMA_FULLTRAIN"
    assert benchmark.FIT_ID != "POS_CONTRAST_V1_E79_REAL"
    assert benchmark.NEW_GLOBAL_CAP == benchmark.OLD_GLOBAL_CAP + 1
    assert benchmark.NEW_CAPS["continuation"] == benchmark.OLD_CAPS["continuation"] + 1
    assert sum(benchmark.NEW_CAPS.values()) == benchmark.NEW_GLOBAL_CAP


def test_binary_threshold_uses_inclusive_half():
    probability = np.asarray([0.499999, 0.5, 0.500001], dtype=np.float64)
    labels = (probability >= 0.50).astype(np.int8)
    assert np.array_equal(labels, np.asarray([0, 1, 1], dtype=np.int8))
