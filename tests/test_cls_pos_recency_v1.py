import numpy as np
import pytest
from scipy import sparse

from src.models import cls_pos_recency_v1 as model


def test_h8_floor_and_class_mass_are_fixed():
    cycles = np.array([47, 51, 59, 63, 67, 67], dtype=np.int64)
    base = model.recency_base_weights(cycles, 67)
    assert base[0] == 0.25
    assert base[-1] == 1.0
    assert base[1] == pytest.approx(0.25)
    assert base[-2] == base[-1]

    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int8)
    weights = model.normalize_class_mass(base, y)
    for label in (0, 1):
        mask = y == label
        assert weights[mask].sum() == pytest.approx(mask.sum(), rel=1e-12, abs=1e-10)
    assert weights[4] == weights[5]


def test_shuffle_replays_seed_order_and_preserves_each_class_multiset():
    cycles = np.array([47, 49, 52, 56, 59, 62, 65, 67] * 2)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1] * 2, dtype=np.int8)
    weights = model.normalize_class_mass(model.recency_base_weights(cycles, 67), y)
    actual, permutation = model.shuffle_weights_within_class(weights, y)

    rng = np.random.default_rng(42)
    expected = np.empty_like(weights)
    expected_perm = np.arange(len(y), dtype=np.int64)
    for label in (0, 1):
        idx = np.flatnonzero(y == label)
        order = rng.permutation(len(idx))
        expected[idx] = weights[idx[order]]
        expected_perm[idx] = idx[order]
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(permutation, expected_perm)
    for label in (0, 1):
        mask = y == label
        np.testing.assert_array_equal(np.sort(actual[mask]), np.sort(weights[mask]))
        assert model.effective_sample_size(actual[mask]) == pytest.approx(
            model.effective_sample_size(weights[mask]))
    assert np.any(permutation != np.arange(len(y)))
    assert np.any(actual != weights)


def test_weighted_gradient_matches_finite_differences():
    rng = np.random.default_rng(9)
    x = sparse.csr_matrix(rng.normal(size=(17, 5)))
    z = rng.normal(size=17)
    y = np.array([0, 1] * 8 + [1], dtype=np.int8)
    weights = rng.uniform(0.4, 1.8, size=17)
    gamma = rng.normal(scale=0.2, size=5)
    loss, gradient = model.weighted_offset_objective_gradient(gamma, x, z, y, weights)
    eps = 1e-6
    numeric = np.empty_like(gamma)
    for j in range(len(gamma)):
        plus = gamma.copy(); plus[j] += eps
        minus = gamma.copy(); minus[j] -= eps
        loss_plus = model.weighted_offset_objective_gradient(plus, x, z, y, weights)[0]
        loss_minus = model.weighted_offset_objective_gradient(minus, x, z, y, weights)[0]
        numeric[j] = (loss_plus - loss_minus) / (2 * eps)
    assert np.isfinite(loss)
    np.testing.assert_allclose(gradient, numeric, rtol=1e-6, atol=2e-7)


def test_unit_weights_and_zero_position_rows_have_expected_objective_effect():
    x = sparse.csr_matrix([[1.0, 0.0], [0.0, 2.0], [1.0, -1.0]])
    gamma = np.array([0.15, -0.3])
    z = np.array([-0.2, 0.4, 0.1])
    y = np.array([0, 1, 1], dtype=np.int8)
    unit = np.ones(3)
    loss, grad = model.weighted_offset_objective_gradient(gamma, x, z, y, unit)
    raw_y = 2.0 * y - 1.0
    logits = z + x @ gamma
    expected_loss = np.logaddexp(0.0, -raw_y * logits).sum() + 0.5 * gamma @ gamma
    expected_grad = x.T @ (1.0 / (1.0 + np.exp(-logits)) - y) + gamma
    assert loss == pytest.approx(expected_loss)
    np.testing.assert_allclose(grad, np.asarray(expected_grad).reshape(-1))

    x_plus = sparse.vstack([x, sparse.csr_matrix((1, x.shape[1]))], format="csr")
    loss_plus, grad_plus = model.weighted_offset_objective_gradient(
        gamma, x_plus, np.r_[z, 0.7], np.r_[y, 0], np.ones(4))
    assert loss_plus - loss == pytest.approx(np.logaddexp(0.0, 0.7))
    np.testing.assert_allclose(grad_plus, grad)


def test_tiny_optimizer_and_prediction_route_keep_missing_fallback_exact():
    rng = np.random.default_rng(20)
    x = sparse.csr_matrix(rng.normal(size=(60, 6)))
    y = np.array([0, 1] * 30, dtype=np.int8)
    z = rng.normal(scale=0.3, size=60)
    gamma, receipt = model.fit_weighted_gamma(x, z, y, np.ones(60))
    assert receipt["success"] is True
    assert np.isfinite(gamma).all()
    np.testing.assert_allclose(model.offset_probability(x, z, gamma),
                               1.0 / (1.0 + np.exp(-(z + x @ gamma))))

    present = np.array([True, False, True, False])
    p_a02 = np.array([0.5, 0.2, 0.3, model.MISSING_THRESHOLD])
    p_pos = np.array([0.8, 0.9, 0.8, 0.1])
    routed = model.route_probabilities(present, p_a02, p_pos)
    assert np.array_equal(routed[~present].view(np.uint64), p_a02[~present].view(np.uint64))
    np.testing.assert_array_equal(model.decisions(routed, present), np.array([1, 0, 1, 1]))


@pytest.mark.parametrize("bad_target", [np.array([0, 0.5]), np.array([0, 256])])
def test_non_binary_target_is_rejected_before_integer_cast(bad_target):
    with pytest.raises(RuntimeError):
        model.normalize_class_mass(np.ones(2), bad_target)


def test_fractional_or_out_of_prefix_cycles_are_rejected():
    with pytest.raises(RuntimeError):
        model.recency_base_weights([47, 60.5], 67)
    with pytest.raises(RuntimeError):
        model.recency_base_weights([46, 67], 67)
