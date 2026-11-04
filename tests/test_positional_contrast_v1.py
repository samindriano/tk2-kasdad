import numpy as np
import pytest
from scipy import sparse

from src.models.positional_contrast_v1 import contrast_block, expand_contrast_coefficients
from src.models.positional_bag_offset_v1 import fit_offset_coefficients, offset_probability


def test_contrast_embedding_logits_and_l2_penalty():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(9, 18))
    theta = rng.normal(size=12)
    z = contrast_block(sparse.csr_matrix(x), 0)
    gamma = expand_contrast_coefficients(theta)
    np.testing.assert_allclose(z @ theta, x @ gamma, atol=1e-12)
    np.testing.assert_allclose(np.dot(theta, theta), np.dot(gamma, gamma), atol=1e-12)
    np.testing.assert_allclose(sum(np.split(gamma, 3)), np.zeros(6), atol=1e-12)


def test_equal_thirds_zero_and_structured_columns_excluded():
    block = sparse.csr_matrix([[1., 0.], [0., 3.]])
    x = sparse.hstack([sparse.csr_matrix([[100.], [-100.]]), block, block, block])
    z = contrast_block(x, 1)
    assert z.shape == (2, 4) and z.nnz == 0


def test_no_post_projection_normalization():
    x = sparse.csr_matrix([[1., 0., 0.]])
    actual = contrast_block(x, 0).toarray()
    np.testing.assert_allclose(actual, [[1/np.sqrt(2), 1/np.sqrt(6)]])
    assert np.linalg.norm(actual) < 1


def test_blank_and_zero_width_blocks():
    assert contrast_block(sparse.csr_matrix((3, 6)), 0).nnz == 0
    assert contrast_block(sparse.csr_matrix((3, 2)), 2).shape == (3, 0)
    assert contrast_block(sparse.csr_matrix((0, 6)), 0).shape == (0, 4)


@pytest.mark.parametrize("base", [-1, 4, .5, True])
def test_invalid_base_width_rejected(base):
    with pytest.raises(ValueError):
        contrast_block(sparse.csr_matrix((1, 3)), base)


def test_bad_width_and_nonfinite_rejected():
    with pytest.raises(ValueError):
        contrast_block(sparse.csr_matrix((2, 4)), 0)
    with pytest.raises(ValueError):
        contrast_block(sparse.csr_matrix([[np.nan, 0, 0]]), 0)
    with pytest.raises(ValueError):
        expand_contrast_coefficients(np.array([1., 2., 3.]))


def test_actual_optimizer_persistence_and_replay(tmp_path):
    rng = np.random.default_rng(7)
    x = sparse.csr_matrix(rng.normal(size=(80, 12)))
    z = contrast_block(x, 0)
    offset = rng.normal(size=80)
    y = (offset + np.asarray(z @ rng.normal(size=8)).ravel() > 0).astype(int)
    theta, metadata = fit_offset_coefficients(z, offset, y, c=1.)
    assert metadata["success"] and metadata["loss_final"] <= metadata["loss_gamma0"]
    path = tmp_path / "synthetic_gamma.npz"
    np.savez(path, theta=theta)
    with np.load(path, allow_pickle=False) as saved:
        np.testing.assert_array_equal(offset_probability(z, offset, saved["theta"]),
                                      offset_probability(z, offset, theta))
    np.testing.assert_allclose(offset_probability(z, offset, theta),
                              offset_probability(x, offset, expand_contrast_coefficients(theta)), atol=1e-14)
