from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import scripts.optimization_v3_research_next_n03 as runner
from src.models.research_next_contract import publish_bundle, read_bundle
from src.models.research_next_n03 import (
    FEATURE_NAMES, ROUTER_PARAMETERS, compose_arms, constant_arm_control,
    fit_router, mae, predict_route, rank50_complete_cycle, router_features,
)


def test_complete_cycle_rank_precedes_missing_stream_routing():
    raw22 = np.array([20.0, 20.0, 40.0, 80.0])
    raw06 = np.array([5.0, 5.0, 25.0, 65.0])
    legacy = np.array([50.0, 50.0, 50.0, 50.0])
    missing = np.array([True, False, False, False])
    cycles = np.array([56, 56, 56, 56], dtype=np.int64)
    a, b, _, rank06 = compose_arms(legacy, raw22, raw06, cycles, missing)
    present_only = rank50_complete_cycle(raw06[~missing], cycles[~missing])
    assert rank06[1] != present_only[0]
    assert b[0] == a[0]
    assert b[1] == pytest.approx(0.5 * legacy[1] + 0.25 * rank50_complete_cycle(raw22, cycles)[1]
                                 + 0.25 * rank06[1])


class _ForbiddenModel:
    def predict(self, _x):  # pragma: no cover - called only if fallback fails
        raise AssertionError("fallback must not evaluate a router")


def test_cutoff_55_and_low_support_return_exact_arm_a():
    a = np.array([1.0, 8.0, 30.0])
    b = np.array([4.0, 5.0, 20.0])
    x = np.zeros((3, len(FEATURE_NAMES)))
    missing = np.array([False, True, False])
    for cutoff, support in ((55, 0), (67, 1999)):
        pred, delta, route = predict_route(_ForbiddenModel(), np.zeros(len(FEATURE_NAMES)), x,
                                           a, b, missing, cutoff=cutoff, present_oof_rows=support)
        np.testing.assert_array_equal(pred, a)
        np.testing.assert_array_equal(delta, np.zeros(3))
        assert not route.any()


def _synthetic_fit(seed: int = 29):
    rng = np.random.default_rng(seed)
    n_fit, n_score = 2_400, 240
    fit_a = rng.uniform(15, 85, n_fit)
    fit_b = np.clip(fit_a + rng.normal(-2, 8, n_fit), 0, 100)
    fit_num = [rng.normal(size=n_fit) for _ in range(3)]
    fit_num[0][::17] = np.nan
    fit_num[1][::13] = np.nan
    fit_num[2][:] = np.nan  # all-missing fit-only column must impute to zero
    x_fit = router_features(fit_a, fit_b, *fit_num)
    y_fit = rng.normal(-0.15, 2.5, n_fit) + 0.05 * x_fit[:, 2]

    score_a = rng.uniform(0, 100, n_score)
    score_b = np.clip(score_a + rng.normal(-2, 8, n_score), 0, 100)
    score_num = [rng.normal(size=n_score) for _ in range(3)]
    score_num[0][::11] = np.nan
    score_num[1][::7] = np.nan
    score_num[2][:] = np.nan
    x_score = router_features(score_a, score_b, *score_num)
    missing = np.zeros(n_score, dtype=bool)
    missing[::19] = True
    y_score = np.where(np.arange(n_score) % 2 == 0, 0.0, 100.0)
    return x_fit, y_fit, score_a, score_b, x_score, missing, y_score


def test_synthetic_fit_publish_readback_independent_metrics_guards_and_resume(tmp_path, monkeypatch):
    x_fit, y_fit, a, b, x_score, missing, y = _synthetic_fit()
    model, medians, all_missing = fit_router(x_fit, y_fit)
    assert all_missing[-1]
    assert medians[-1] == 0.0
    prediction, predicted_delta, routed = predict_route(
        model, medians, x_score, a, b, missing, cutoff=67, present_oof_rows=len(y_fit),
    )
    assert not np.any(routed & missing)
    alternate_future_labels = 100.0 - y
    # Evaluation labels are never an argument to fit_router; perturbing them cannot change fitted state.
    model2, medians2, _ = fit_router(x_fit, y_fit)
    np.testing.assert_array_equal(medians, medians2)
    np.testing.assert_array_equal(model.predict(np.where(np.isfinite(x_score), x_score, medians)),
                                  model2.predict(np.where(np.isfinite(x_score), x_score, medians2)))
    assert not np.array_equal(alternate_future_labels, y)

    constant, constant_mean, chooses_b = constant_arm_control(a, b, missing, y_fit)
    ids = np.asarray([f"δ-rid-{index}" for index in range(len(y))], dtype=str)
    rows = np.arange(len(y), dtype=np.int64)
    cycles = np.repeat(np.arange(68, 80), len(y) // 12)
    if len(cycles) < len(y):
        cycles = np.pad(cycles, (0, len(y) - len(cycles)), constant_values=79)
    payload = {
        "score_rows": rows,
        "score_record_ids": ids,
        "prediction": prediction,
        "baseline_prediction": a,
        "arm_b": b,
        "constant_control": constant,
        "protected_legacy": np.full(len(y), 50.0),
        "r06_original": np.full(len(y), 50.0),
        "stream_missing": missing,
        "cycle_code": cycles.astype(np.int64),
        "routed_to_b": routed,
        "predicted_loss_difference": predicted_delta,
        "diagnostics": {"fit_only_medians": tuple(medians), "origin_cutoffs": (55, 59, 63),
                        "constant_mean": constant_mean, "constant_chooses_b": chooses_b,
                        "outer_score_labels_passed_to_fit": False},
    }
    identity = {"batch_id": "synthetic-n03", "card_id": "N03", "cutoff": 67}
    destination = tmp_path / "bundle"

    original_publish = runner.publish_bundle
    attempts = {"count": 0}

    def fail_once(*args, **kwargs):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise OSError("synthetic persistence failure")
        return original_publish(*args, **kwargs)

    monkeypatch.setattr(runner, "RECOVERY_ROOT", tmp_path / "recovery")
    monkeypatch.setattr(runner, "publish_bundle", fail_once)
    recovery_payload = {"arrays": {k: v for k, v in payload.items() if k != "diagnostics"},
                        "metadata": payload["diagnostics"]}
    runner._save_recovery(67, identity, recovery_payload)
    with pytest.raises(OSError, match="synthetic persistence failure"):
        runner.publish_bundle(destination, payload, identity, task="regression")
    recovered = runner._load_recovery(67, identity)
    assert recovered is not None
    bundle_identity = {"batch_id": "synthetic-n03", "card_id": "N03", "cutoff": 67}
    monkeypatch.setattr(runner, "publish_bundle", original_publish)
    replay_payload = {**recovered["arrays"], "diagnostics": recovered["metadata"]}
    published = publish_bundle(destination, replay_payload, bundle_identity, task="regression")
    saved = read_bundle(destination, bundle_identity, task="regression")
    rebuilt = np.where(saved.arrays["routed_to_b"], saved.arrays["arm_b"], saved.arrays["baseline_prediction"])
    np.testing.assert_array_equal(rebuilt, saved.arrays["prediction"])
    assert saved.receipt["metadata"]["diagnostics"]["origin_cutoffs"] == [55, 59, 63]
    assert np.isfinite(mae(y, rebuilt))
    assert published["receipt_sha256"] == saved.receipt["receipt_sha256"]

    record = {key: saved.arrays[key] for key in saved.arrays}
    frame = pd.DataFrame({"cycle_code": cycles, "record_id": ids})
    guarded = runner._view_metrics(frame, y, [record], name="synthetic-all-view")
    assert guarded["n"] == len(y)
    assert guarded["groups"]["target_le5_evaluation_only"]["supported"]
    assert guarded["groups"]["target_ge95_evaluation_only"]["supported"]
    assert guarded["metrics"]["router"] == pytest.approx(np.abs(y - rebuilt).sum() / len(y))
