from __future__ import annotations

import numpy as np

from scripts.optimization_v3_classification_sparse_mlp_v1 import _synthetic_frame
from src.models.classification_sparse_mlp_v1 import (
    build_fit_design,
    make_estimator,
    route_probabilities,
    transform_score_design,
)


def test_frozen_estimator_parameters_are_exact():
    estimator = make_estimator()
    assert estimator.hidden_layer_sizes == [32]
    assert estimator.activation == "relu"
    assert estimator.solver == "adam"
    assert estimator.alpha == 0.01
    assert estimator.batch_size == 256
    assert estimator.learning_rate_init == 0.001
    assert estimator.max_iter == 30
    assert estimator.random_state == 42
    assert estimator.early_stopping is False
    assert estimator.tol == 0
    assert estimator.n_iter_no_change == 31


def test_c2b_fit_and_score_design_remain_aligned_sparse_csr():
    frame, target = _synthetic_frame(96)
    base, fit_matrix, fit_target = build_fit_design(frame.iloc[:72], target[:72])
    score = transform_score_design(base, frame.iloc[72:])
    assert fit_matrix.shape[1] == score.shape[1]
    assert fit_matrix.shape[0] == len(fit_target) == 72
    assert score.shape[0] == 24
    assert fit_matrix.format == "csr"
    assert score.format == "csr"
    assert fit_matrix.dtype == score.dtype == np.float64
    assert np.isfinite(fit_matrix.data).all()
    assert np.isfinite(score.data).all()


def test_route_uses_only_frozen_missing_fallback_and_fixed_weights():
    baseline = np.asarray([0.2, 0.4, 0.6])
    mlp = np.asarray([0.8, 0.2, 0.5])
    c2b = np.asarray([0.6, 0.3, 0.1])
    cb = np.asarray([0.95, 0.85, 0.75])
    missing = np.asarray([False, True, False])
    candidate = route_probabilities("candidate", baseline, mlp, c2b, cb, missing)
    matched = route_probabilities("matched_control", baseline, mlp, c2b, cb, missing)
    np.testing.assert_allclose(candidate, [0.35, 0.85, 0.575], rtol=0, atol=1e-15)
    np.testing.assert_allclose(matched, [0.3, 0.85, 0.475], rtol=0, atol=1e-15)


def test_route_rejects_out_of_range_or_misaligned_probabilities():
    with np.testing.assert_raises(ValueError):
        route_probabilities("candidate", [0.2], [1.2], [0.3], [0.4], [False])
    with np.testing.assert_raises(ValueError):
        route_probabilities("candidate", [0.2, 0.3], [0.4], [0.5], [0.6], [False, False])
