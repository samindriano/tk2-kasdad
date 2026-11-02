from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.optimization_v3_a01_preflight import evaluate_gate, run_preflight, synthetic_fixture
from src.models.classification_recency_a01 import (
    FEATURE_NAMES,
    PARAMS,
    RecencySparseClassifier,
    _TFIDF_PARAMS as A01_TFIDF_PARAMS,
    recency_weight_plan,
    route_probabilities,
)
from src.models.classification_models import _TFIDF_PARAMS as CLASSIFICATION_TFIDF_PARAMS
from src.models.optimization_v3_sparse_classification import SparseTextBoostClassifier


def test_recency_weight_plan_preserves_class_mass_ess_and_deterministic_shuffle():
    cycles = np.repeat(np.arange(47, 56), 20)
    target = (np.arange(len(cycles)) % 5 < 2).astype(np.int8)
    first = recency_weight_plan(cycles, target, 55)
    second = recency_weight_plan(cycles, target, 55)

    np.testing.assert_array_equal(first["recency_weights"], second["recency_weights"])
    np.testing.assert_array_equal(first["shuffled_weights"], second["shuffled_weights"])
    np.testing.assert_array_equal(first["permutation_source_rows"], second["permutation_source_rows"])
    diagnostics = first["diagnostics"]
    assert diagnostics["preflight_checks"]["finite_positive_weights"]
    assert diagnostics["preflight_checks"]["global_means_one"]
    assert diagnostics["preflight_checks"]["class_masses_preserved"]
    assert diagnostics["preflight_checks"]["each_class_ess_ratio_at_least_0_50"]
    for class_value in (0, 1):
        mask = target == class_value
        assert np.isclose(first["recency_weights"][mask].sum(), mask.sum())
        assert np.isclose(first["shuffled_weights"][mask].sum(), mask.sum())
        assert set(first["permutation_source_rows"][mask]).issubset(set(np.flatnonzero(mask)))


def test_recency_weights_reject_wrong_cutoff_population_and_single_class():
    with pytest.raises(ValueError, match="cycles 47 through"):
        recency_weight_plan(np.array([46, 55]), np.array([0, 1]), 55)
    with pytest.raises(ValueError, match="both binary classes"):
        recency_weight_plan(np.array([47, 55]), np.array([1, 1]), 55)


def test_fixed_availability_route_rebuilds_baseline_candidate_and_threshold():
    recency = np.array([0.1, 0.8, 0.6])
    shuffled = np.array([0.4, 0.2, 0.7])
    sparse25 = np.array([0.9, 0.3, 0.2])
    c2b = np.array([0.2, 0.8, 0.5])
    cb = np.array([0.7, 0.4, 0.3])
    missing = np.array([True, False, False])
    routed = route_probabilities(recency, shuffled, sparse25, c2b, cb, missing)
    np.testing.assert_allclose(routed["sparse25_baseline"], [0.7, 0.675, 0.425])
    np.testing.assert_allclose(routed["recency_candidate"], [0.7, 0.8, 0.525])
    np.testing.assert_allclose(routed["shuffled_weight_control"], [0.7, 0.65, 0.55])
    np.testing.assert_array_equal(routed["recency_label"], [1, 1, 1])
    np.testing.assert_array_equal(routed["shuffled_label"], [1, 1, 1])


def test_a01_model_uses_frozen_features_and_sample_weight_fit_path():
    frame, target, _ = synthetic_fixture()
    cycles = frame["cycle_code"].to_numpy(dtype=np.int64)
    fit_rows = np.flatnonzero((cycles >= 47) & (cycles <= 55))
    score_rows = np.flatnonzero((cycles >= 56) & (cycles <= 59))
    plan = recency_weight_plan(cycles[fit_rows], target[fit_rows], 55)
    model = RecencySparseClassifier(55, "RECENCY").fit(
        frame.iloc[fit_rows].loc[:, list(FEATURE_NAMES)], target[fit_rows], plan["recency_weights"]
    )
    state = model.audit_state()
    probabilities = model.predict_probability(frame.iloc[score_rows].loc[:, list(FEATURE_NAMES)])

    assert state["lightgbm_parameters"] == PARAMS
    assert state["sample_weight_used"] is True
    assert state["fit_rows"] == len(fit_rows)
    assert state["validation_labels_passed_to_fit"] is False
    assert state["test_data_used"] is False
    assert len(probabilities) == len(score_rows)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_a01_tfidf_and_sparse_design_match_frozen_classifier_recipe():
    frame, target, _ = synthetic_fixture()
    cycles = frame["cycle_code"].to_numpy(dtype=np.int64)
    fit_rows = np.flatnonzero((cycles >= 47) & (cycles <= 55))
    fit_frame = frame.iloc[fit_rows].loc[:, list(FEATURE_NAMES)]
    fit_target = target[fit_rows]
    plan = recency_weight_plan(cycles[fit_rows], fit_target, 55)
    a01 = RecencySparseClassifier(55, "RECENCY").fit(
        fit_frame, fit_target, plan["recency_weights"]
    ).audit_state()
    sparse25 = SparseTextBoostClassifier().fit(fit_frame, fit_target).audit_state()

    assert A01_TFIDF_PARAMS == CLASSIFICATION_TFIDF_PARAMS
    assert a01["tfidf_parameters"] == sparse25["tfidf_params"]
    assert a01["input_feature_names"] == sparse25["input_feature_names"]
    assert a01["fit_data_sha256"] == sparse25["fit_data_sha256"]
    assert a01["target_sha256"] == sparse25["target_sha256"]
    assert a01["vocabulary_size"] == sparse25["vocabulary_size"]
    assert a01["vocabulary_sha256"] == sparse25["vocabulary_sha256"]
    assert a01["idf_sha256"] == sparse25["idf_sha256"]
    assert a01["structured_preprocessing"] == sparse25["structured_state"]
    assert a01["design_shape"] == sparse25["design_shape"]
    assert a01["design_data_sha256"] == sparse25["design_data_sha256"]


def test_model_rejects_identity_and_target_leakage_columns():
    frame, target, _ = synthetic_fixture()
    fit_rows = np.flatnonzero(frame["cycle_code"].to_numpy() <= 55)
    plan = recency_weight_plan(frame.iloc[fit_rows]["cycle_code"], target[fit_rows], 55)
    invalid = frame.iloc[fit_rows].loc[:, list(FEATURE_NAMES)].copy()
    invalid["artifact_signal"] = target[fit_rows]
    with pytest.raises(ValueError, match="exactly"):
        RecencySparseClassifier(55, "RECENCY").fit(invalid, target[fit_rows], plan["recency_weights"])


def test_gate_evaluator_checks_all_views_and_returns_negative_verdicts():
    metrics = {
        view: {"sparse25": 0.80, "recency": 0.79, "shuffled": 0.78}
        for view in ("primary", "H12", "H12_55", "H12_67", "W1", "W4")
    }
    result = evaluate_gate(metrics)
    assert result["status"] == "FAIL"
    assert result["checks"]["primary_nonnegative_vs_sparse25"] is False
    assert result["checks"]["H12_min_gain_vs_sparse25"] is False
    with pytest.raises(ValueError, match="exactly these views"):
        evaluate_gate({"primary": metrics["primary"]})


def test_synthetic_preflight_resumes_persisted_fit_and_verifies_end_to_end(tmp_path):
    from scripts.optimization_v3_a01_preflight_verify import verify_synthetic_preflight

    run_root = tmp_path / "synthetic_fits"
    report_path = tmp_path / "synthetic_preflight.json"
    report = run_preflight(run_root=run_root, report_path=report_path)
    verification = verify_synthetic_preflight(run_root=run_root, report_path=report_path)

    assert report["status"] == "SYNTHETIC_PREFLIGHT_V2_PASS"
    assert report["official_fits_started"] == 0
    assert report["synthetic_a01_fit_count"] == 4
    assert report["synthetic_sparse25_reference_fit_count"] == 2
    assert report["synthetic_estimator_fit_count"] == 6
    assert verification["status"] == "PASS", verification["problems"]
    assert all(verification["recovery_stage_checks"].values())
