import numpy as np
from scipy import sparse

from scripts.optimization_v3_learned_pair_preflight_v1 import (
    fm_pair_score,
    fm_pair_score_and_v_gradient,
    logistic_l2_objective_gradient,
    l2_normalize_rows,
)


def _explicit_pair_score(x, factors):
    dense = x.toarray() if sparse.issparse(x) else np.asarray(x)
    result = np.zeros(dense.shape[0], dtype=np.float64)
    for row_i, row in enumerate(dense):
        for i in range(row.size):
            for j in range(i + 1, row.size):
                result[row_i] += row[i] * row[j] * float(factors[i] @ factors[j])
    return result


def test_fm_identity_matches_explicit_pairs_with_repeated_counts():
    x = np.asarray([[2.0, 3.0, 0.0, 1.5], [0.0, 4.0, 2.0, 0.0],
                    [3.5, 0.0, 0.0, 0.0]], dtype=np.float64)
    v = np.asarray([[0.2, -0.1], [0.3, 0.4], [-0.2, 0.5], [0.1, 0.6]])
    np.testing.assert_allclose(fm_pair_score(x, v), _explicit_pair_score(x, v), atol=1e-12)
    np.testing.assert_allclose(fm_pair_score(sparse.csr_matrix(x), v), _explicit_pair_score(x, v), atol=1e-12)


def test_single_active_feature_has_no_pair_term_or_factor_gradient():
    x = np.asarray([[0.0, 0.0, 4.0, 0.0], [2.5, 0.0, 0.0, 0.0]])
    v = np.arange(8, dtype=np.float64).reshape(4, 2) / 10.0
    np.testing.assert_allclose(fm_pair_score(x, v), np.zeros(2), atol=1e-15)
    _, (_, grad_v, _) = logistic_l2_objective_gradient(
        x, x, np.asarray([0.0, 1.0]), np.zeros(4), v, 0.0,
        l2_main=0.2, l2_factors=0.0, l2_intercept=0.1,
    )
    np.testing.assert_allclose(grad_v, np.zeros((4, 2)), atol=1e-15)


def test_standalone_weighted_pair_factor_gradient_matches_finite_difference():
    x = sparse.csr_matrix([[2.0, 1.0, 0.0], [0.5, 0.0, 3.0], [1.0, 2.0, 1.5]])
    v = np.asarray([[0.12, -0.08], [0.04, 0.15], [-0.11, 0.03]], dtype=np.float64)
    residual = np.asarray([0.2, -0.4, 0.7])
    scores, analytic = fm_pair_score_and_v_gradient(x, v, residual)
    np.testing.assert_allclose(scores, fm_pair_score(x, v), rtol=1e-12, atol=1e-12)

    def weighted_score(value):
        return float(residual @ fm_pair_score(x, value.reshape(v.shape)))

    epsilon = 1e-6
    numeric = np.empty(v.size, dtype=np.float64)
    flat = v.ravel()
    for i in range(flat.size):
        plus = flat.copy(); plus[i] += epsilon
        minus = flat.copy(); minus[i] -= epsilon
        numeric[i] = (weighted_score(plus) - weighted_score(minus)) / (2 * epsilon)
    np.testing.assert_allclose(analytic.ravel(), numeric, rtol=1e-6, atol=2e-9)

    dense_scores, dense_gradient = fm_pair_score_and_v_gradient(x.toarray(), v, residual)
    np.testing.assert_allclose(dense_scores, scores, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(dense_gradient, analytic, rtol=1e-12, atol=1e-12)


def test_zero_factor_initialization_is_stationary_for_interaction_block():
    x = np.asarray([[1.0, 2.0, 0.0], [0.0, 1.0, 2.0], [1.0, 0.0, 1.0]])
    y = np.asarray([0.0, 1.0, 1.0])
    _, (_, grad_v, _) = logistic_l2_objective_gradient(
        x, x, y, np.zeros(3), np.zeros((3, 2)), 0.0,
        l2_main=0.1, l2_factors=0.1, l2_intercept=0.1,
    )
    np.testing.assert_array_equal(grad_v, np.zeros((3, 2)))


def test_sparse_dense_offset_objective_gradient_and_finite_difference():
    x = np.asarray([[1.0, 2.0, 0.0], [0.0, 1.5, 2.0], [2.0, 0.0, 1.0],
                    [0.5, 1.0, 0.0]], dtype=np.float64)
    y = np.asarray([0.0, 1.0, 1.0, 0.0])
    w = np.asarray([0.13, -0.21, 0.08])
    v = np.asarray([[0.12, -0.07], [0.03, 0.16], [-0.11, 0.05]])
    b = -0.17
    offset = np.asarray([0.3, -0.2, 0.1, -0.05])
    kwargs = dict(l2_main=0.07, l2_factors=0.11, l2_intercept=0.03, offset=offset)
    dense = logistic_l2_objective_gradient(x, x, y, w, v, b, **kwargs)
    sparse_result = logistic_l2_objective_gradient(sparse.csr_matrix(x), sparse.csr_matrix(x),
                                                  y, w, v, b, **kwargs)
    np.testing.assert_allclose(dense[0], sparse_result[0], rtol=1e-12, atol=1e-12)
    for left, right in zip(dense[1][:2], sparse_result[1][:2]):
        np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(dense[1][2], sparse_result[1][2], rtol=1e-12, atol=1e-12)

    theta = np.concatenate([w, v.ravel(), np.asarray([b])])

    def unpack(value):
        return value[:3], value[3:9].reshape(3, 2), float(value[-1])

    def objective(value):
        ww, vv, bb = unpack(value)
        return logistic_l2_objective_gradient(x, x, y, ww, vv, bb, **kwargs)[0]

    analytic = np.concatenate([dense[1][0], dense[1][1].ravel(), np.asarray([dense[1][2]])])
    epsilon = 1e-6
    numeric = np.empty_like(theta)
    for i in range(theta.size):
        plus = theta.copy(); plus[i] += epsilon
        minus = theta.copy(); minus[i] -= epsilon
        numeric[i] = (objective(plus) - objective(minus)) / (2 * epsilon)
    np.testing.assert_allclose(analytic, numeric, rtol=2e-6, atol=2e-8)


def test_row_normalization_handles_repeated_values_and_zero_rows():
    x = sparse.csr_matrix([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0], [2.0, 2.0, 1.0]])
    dense = l2_normalize_rows(x)
    norms = np.sqrt(np.asarray(dense.multiply(dense).sum(axis=1)).reshape(-1))
    np.testing.assert_allclose(norms, np.asarray([1.0, 0.0, 1.0]), atol=1e-12)
