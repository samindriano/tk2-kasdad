from __future__ import annotations

import numpy as np

from src.models.research_next_contract import publish_bundle, read_bundle
from src.models.research_next_n05 import (
    ALPHAS, QUANTILE_PARAMETERS, fit_quantile_head, interval_metrics,
    make_interval, past_residual_interval, point_model_eligibility,
    predict_quantile_head, safe_interval_metrics, width_rank_groups,
)


def test_frozen_quantile_recipe_and_alpha_heads_fit_synthetic_data() -> None:
    assert QUANTILE_PARAMETERS["objective"] == "quantile"
    assert QUANTILE_PARAMETERS["n_estimators"] == 400
    rng = np.random.default_rng(21)
    design = rng.normal(size=(96, 5))
    target = np.clip(45 + 5 * design[:, 0] + rng.normal(size=96), 0, 100)
    small_recipe = {**QUANTILE_PARAMETERS, "n_estimators": 5, "min_child_samples": 4}
    for alpha in ALPHAS:
        estimator, state = fit_quantile_head(
            design, target, alpha=alpha, cutoff=47, parameters=small_recipe,
        )
        prediction = predict_quantile_head(estimator, design[:12])
        assert state["alpha"] == alpha
        assert state["parameters"]["alpha"] == alpha
        assert prediction.shape == (12,)
        assert np.isfinite(prediction).all()
        assert state["score_target_passed_to_fit"] is False


def test_crossing_clipping_and_tie_handling() -> None:
    interval = make_interval([120, 80, 40], [90, 70, 60])
    assert interval["raw_crossing_rate"] == 2 / 3
    assert interval["clipped_endpoint_rows"] == 1
    np.testing.assert_array_equal(interval["lower"], [90, 70, 40])
    np.testing.assert_array_equal(interval["upper"], [100, 80, 60])


def test_past_residual_interval_uses_signed_linear_quantiles() -> None:
    result = past_residual_interval([50, 50], [0, 100], [50, 50])
    assert result["past_residual_rows"] == 2
    assert result["past_residual_q10"] == -40.0
    assert result["past_residual_q90"] == 40.0
    np.testing.assert_array_equal(result["lower"], [10, 10])
    np.testing.assert_array_equal(result["upper"], [90, 90])


def test_empty_groups_and_width_groups_are_explicit() -> None:
    empty = safe_interval_metrics([], [], [])
    assert empty["n"] == 0
    assert empty["coverage_80"] is None
    assert empty["width_absolute_error_spearman_reason"] == "empty group"
    groups = width_rank_groups([1, 1, 2, 3, 4, 5], groups=3)
    assert groups.shape == (6,)
    assert groups.min() >= 1 and groups.max() <= 3
    metric = interval_metrics([50, 55], [40, 45], [60, 65], [49, 56])
    assert metric["n"] == 2
    assert metric["width_error_reference"] == "provided point prediction"


def test_information_only_lane_is_never_point_model_eligible() -> None:
    result = point_model_eligibility()
    assert result["eligible"] is False
    assert "information-only" in result["reason"]


def test_quantile_bundle_persists_and_resumes_with_exact_identity(tmp_path) -> None:
    destination = tmp_path / "n05_synthetic"
    identity = {"batch_id": "optimization_v3_research_next_v1", "card_id": "N05",
                "component": "quantile_alpha_0.1", "synthetic": True}
    payload = {
        "score_rows": np.arange(10, 14, dtype=np.int64),
        "score_record_ids": np.asarray(["A", "B", "C", "D"], dtype="U1"),
        "prediction": np.asarray([10.0, 20.0, 30.0, 40.0]),
        "baseline_prediction": np.asarray([12.0, 22.0, 32.0, 42.0]),
        "raw_quantile_prediction": np.asarray([9.0, 19.0, 29.0, 39.0]),
        "diagnostics": {"alpha": 0.1, "synthetic_only": True},
    }
    first = publish_bundle(destination, payload, identity, task="regression")
    resumed = read_bundle(destination, identity, task="regression")
    assert first["receipt_sha256"] == resumed.receipt["receipt_sha256"]
    np.testing.assert_array_equal(resumed.arrays["raw_quantile_prediction"], [9, 19, 29, 39])
