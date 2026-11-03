from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from src.models.learned_pair_pilot_v1 import (
    fm_pair_score,
    fm_pair_score_and_v_gradient,
    initialize_parameters,
    objective_gradient_mean,
    objective_gradient_sum,
    score_logits,
    unpack_parameters,
)


def test_fm_identity_and_weighted_factor_gradient_match_pair_enumeration() -> None:
    q = sparse.csr_matrix([[1.0, 2.0, 0.0], [0.5, 0.0, 1.5]])
    v = np.array([[0.2, -0.4], [0.3, 0.1], [-0.2, 0.5]])
    weights = np.array([0.7, -0.3])
    score = fm_pair_score(q, v)
    explicit = []
    for row in q.toarray():
        total = 0.0
        for j in range(len(row)):
            for k in range(j + 1, len(row)):
                total += row[j] * row[k] * float(v[j] @ v[k])
        explicit.append(total)
    assert np.allclose(score, explicit, rtol=0, atol=1e-14)
    score2, grad = fm_pair_score_and_v_gradient(q, v, weights)
    assert np.array_equal(score, score2)
    flat = v.reshape(-1)
    numeric = np.zeros_like(flat)
    eps = 1e-6
    for i in range(flat.size):
        plus, minus = flat.copy(), flat.copy()
        plus[i] += eps
        minus[i] -= eps
        numeric[i] = (weights @ fm_pair_score(q, plus.reshape(v.shape))
                      - weights @ fm_pair_score(q, minus.reshape(v.shape))) / (2 * eps)
    assert np.allclose(grad.reshape(-1), numeric, rtol=1e-6, atol=1e-8)


def test_sum_objective_chunk_gradient_matches_finite_difference_all_blocks() -> None:
    rng = np.random.default_rng(123)
    x = sparse.csr_matrix(rng.uniform(0.0, 1.4, (13, 5)) * (rng.random((13, 5)) < 0.55))
    y = np.array([0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1], dtype=np.int8)
    theta = initialize_parameters(5, 2, seed=42)
    theta[:5] = rng.normal(0, 0.15, 5)
    theta[5] = 0.2
    value, grad = objective_gradient_sum(theta, x, y, rank=2, chunk_rows=4)
    numeric = np.zeros_like(theta)
    eps = 1e-6
    for i in range(theta.size):
        plus, minus = theta.copy(), theta.copy()
        plus[i] += eps
        minus[i] -= eps
        fp = objective_gradient_sum(plus, x, y, rank=2, chunk_rows=4)[0]
        fm = objective_gradient_sum(minus, x, y, rank=2, chunk_rows=4)[0]
        numeric[i] = (fp - fm) / (2 * eps)
    assert np.isfinite(value)
    assert np.allclose(grad, numeric, rtol=2e-5, atol=3e-6)


@pytest.mark.parametrize("with_factors", [False, True])
def test_sum_mean_and_chunk_full_equalities_including_uneven_chunk(with_factors: bool) -> None:
    rng = np.random.default_rng(72)
    x = sparse.csr_matrix(rng.normal(size=(17, 6)) * (rng.random((17, 6)) < 0.4))
    y = rng.integers(0, 2, size=17, dtype=np.int8)
    theta = initialize_parameters(6, 3, seed=42, with_factors=with_factors)
    theta += rng.normal(0, 0.015, theta.size)
    full = objective_gradient_sum(theta, x, y, rank=3, with_factors=with_factors, chunk_rows=17)
    uneven = objective_gradient_sum(theta, x, y, rank=3, with_factors=with_factors, chunk_rows=5)
    mean = objective_gradient_mean(theta, x, y, rank=3, with_factors=with_factors, chunk_rows=4)
    assert np.allclose(full[0], uneven[0], rtol=1e-13, atol=1e-13)
    assert np.allclose(full[1], uneven[1], rtol=1e-12, atol=1e-12)
    assert np.allclose(full[0], 17 * mean[0], rtol=1e-13, atol=1e-13)
    assert np.allclose(full[1], 17 * mean[1], rtol=1e-12, atol=1e-12)


def test_signed_logaddexp_objective_stays_finite_at_extreme_logits() -> None:
    x = sparse.csr_matrix([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    y = np.array([0, 1, 0], dtype=np.int8)
    theta = np.array([1200.0, -1200.0, 0.0])
    value, grad = objective_gradient_sum(theta, x, y, rank=2, with_factors=False, chunk_rows=2)
    assert np.isfinite(value)
    assert np.isfinite(grad).all()


def test_single_active_feature_has_zero_pair_score_and_factor_gradient() -> None:
    q = sparse.csr_matrix([[1.0, 0.0, 0.0], [0.0, -2.0, 0.0]])
    v = np.arange(6, dtype=np.float64).reshape(3, 2) / 10
    score, grad = fm_pair_score_and_v_gradient(q, v, np.array([1.0, -0.5]))
    assert np.array_equal(score, np.zeros(2))
    assert np.array_equal(grad, np.zeros_like(v))


def test_class_order_and_parameter_blocks_are_explicit() -> None:
    candidate = initialize_parameters(4, 2, seed=42)
    w, b, v = unpack_parameters(candidate, 4, 2)
    assert np.array_equal(w, np.zeros(4))
    assert b == 0.0
    assert v is not None and v.shape == (4, 2) and np.linalg.norm(v) > 0
    assert np.array_equal(candidate, initialize_parameters(4, 2, seed=42))
    control = initialize_parameters(4, 2, seed=42, with_factors=False)
    assert control.shape == (5,)
    assert unpack_parameters(control, 4, 2, with_factors=False)[2] is None


def test_optimizer_moves_nonzero_factor_block_on_controlled_toy() -> None:
    from scipy.optimize import minimize
    from scipy.special import expit

    rng = np.random.default_rng(29)
    dense = rng.normal(size=(48, 5))
    dense[np.abs(dense) < 0.6] = 0.0
    x = sparse.csr_matrix(dense)
    truth_w = np.array([0.25, -0.2, 0.1, 0.0, 0.18])
    truth_v = np.random.default_rng(5).normal(0, 0.2, (5, 2))
    from src.models.learned_pair_pilot_v1 import row_l2_normalize
    logits = np.asarray(x @ truth_w).reshape(-1) + fm_pair_score(row_l2_normalize(x), truth_v)
    y = (expit(logits) > 0.52).astype(np.int8)
    initial = initialize_parameters(5, 2, seed=42)
    initial_v = initial[6:].copy()
    result = minimize(lambda p: objective_gradient_sum(p, x, y, rank=2, chunk_rows=13),
                      initial, method="L-BFGS-B", jac=True,
                      options={"maxiter": 500, "maxfun": 2000, "maxcor": 10,
                               "maxls": 50, "ftol": 1e-9, "gtol": 1e-4})
    assert result.success, result.message
    assert np.isfinite(result.fun) and np.isfinite(result.jac).all()
    assert np.linalg.norm(result.x[6:] - initial_v) > 1e-7
    assert result.fun <= objective_gradient_sum(initial, x, y, rank=2, chunk_rows=13)[0] + 1e-8
    assert np.isfinite(score_logits(result.x, x, rank=2)[0]).all()


def test_synthetic_lifecycle_persistence_independent_verifier_and_resume(tmp_path) -> None:
    from scripts.optimization_v3_learned_pair_pilot_v1 import run_synthetic_lifecycle

    report = run_synthetic_lifecycle(tmp_path / "learned-pair-synthetic")
    assert report["status"] == "PASS_SYNTHETIC_END_TO_END"
    assert report["official_fit_calls"] == 0
    assert report["synthetic_optimizer_fit_calls"] == 1
    assert report["optimizer_reruns_on_resume"] == 0
    assert report["tampered_matrix_rejected"] is True
    assert report["production_writer_blocked"] is True
    assert report["live_budget_ledger_sha256_before"] == report["live_budget_ledger_sha256_after"]
    assert report["independent_verification"]["status"] == "PASS"


def test_failed_started_fit_releases_only_unstarted_ids() -> None:
    from scripts.optimization_v3_learned_pair_pilot_v1 import FIT_IDS, _unstarted_fit_ids

    assert _unstarted_fit_ids(FIT_IDS, {FIT_IDS[0]}) == list(FIT_IDS[1:])
    assert _unstarted_fit_ids(FIT_IDS, set(FIT_IDS)) == []


def test_new_pilot_gates_use_e67_h12_e75_w6_windows_and_supported_groups() -> None:
    from scripts.optimization_v3_learned_pair_pilot_v1 import _gates_for_origin

    views = []
    for view, score in {
        "H12_E67": {"fm": 0.8000, "bag": 0.7994, "nopair": 0.7997},
        "W4": {"fm": 0.7950, "bag": 0.8000, "nopair": 0.7980},
        "W5": {"fm": 0.7980, "bag": 0.8000, "nopair": 0.7990},
        "W6": {"fm": 0.8010, "bag": 0.8000, "nopair": 0.8005},
    }.items():
        for arm, f1 in score.items():
            views.append({"view": view, "arm": arm, "macro_f1": f1})
    groups = []
    for arm, f1, recall in (("fm", 0.75, 0.62), ("bag", 0.74, 0.60), ("nopair", 0.745, 0.61)):
        groups.append({"view": "H12_E67", "group": "availability", "group_value": "present",
                       "arm": arm, "n": 200, "positive": 30, "negative": 170,
                       "macro_f1": f1, "positive_recall": recall})
    e67 = _gates_for_origin(67, views, groups)
    assert all(gate["pass"] for gate in e67)
    required = {gate["gate"] for gate in e67}
    assert "E67_H12_gain_min_0.0005_vs_bag" in required
    assert "E67_H12_gain_min_0.0002_vs_nopair" in required
    assert any("H12_E67_availability_present_macro_f1_harm_vs_nopair" in x for x in required)
    e75 = _gates_for_origin(75, [
        {"view": "W6", "arm": "fm", "macro_f1": 0.80},
        {"view": "W6", "arm": "bag", "macro_f1": 0.79},
        {"view": "W6", "arm": "nopair", "macro_f1": 0.80},
    ], [])
    assert all(gate["pass"] for gate in e75)
    assert len(e75) == 2


def test_unsupported_subgroup_is_unknown_and_does_not_block_numeric_gates() -> None:
    from scripts.optimization_v3_learned_pair_pilot_v1 import _gates_for_origin

    views = []
    for view, scores in {
        "H12_E67": {"fm": 0.8010, "bag": 0.8000, "nopair": 0.8005},
        "W4": {"fm": 0.8000, "bag": 0.7990, "nopair": 0.7990},
        "W5": {"fm": 0.8000, "bag": 0.7990, "nopair": 0.7990},
        "W6": {"fm": 0.8000, "bag": 0.7990, "nopair": 0.7990},
    }.items():
        for arm, macro_f1 in scores.items():
            views.append({"view": view, "arm": arm, "macro_f1": macro_f1})
    unsupported = [{"view": "H12_E67", "group": "text_load", "group_value": "Q4",
                    "arm": arm, "n": 199, "positive": 99, "negative": 100,
                    "macro_f1": 0.1, "positive_recall": 0.1}
                   for arm in ("fm", "bag", "nopair")]
    gates = _gates_for_origin(67, views, unsupported)
    support_gate = next(gate for gate in gates if gate["gate"].endswith("_support"))
    assert support_gate["pass"] is None
    assert support_gate["status"] == "UNKNOWN_UNSUPPORTED"
    evaluable = [gate for gate in gates if gate["pass"] is not None]
    assert evaluable and all(gate["pass"] is True for gate in evaluable)


def test_audit_receipt_is_create_once_and_rejects_identity_drift(tmp_path) -> None:
    from scripts.optimization_v3_learned_pair_pilot_v1 import (
        LearnedPairPilotError, _create_once_receipt,
    )

    path = tmp_path / "access.json"
    first = _create_once_receipt(path, {"created": "t1", "input": "abc"}, ("input",))
    resumed = _create_once_receipt(path, {"created": "t2", "input": "abc"}, ("input",))
    assert resumed == first
    assert resumed["created"] == "t1"
    with pytest.raises(LearnedPairPilotError):
        _create_once_receipt(path, {"created": "t3", "input": "changed"}, ("input",))
