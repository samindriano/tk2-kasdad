import numpy as np
from scipy import sparse
from scipy.special import expit

from src.models.optimization_v3_bigram_position_pilot_v1 import (
    reconstruct_original_pos_logit,
    route_candidate,
    zero_gamma_equivalence,
)
from scripts.optimization_v3_bigram_position_pilot_v1 import _gate, _read_fit_labels_only


def test_reconstruct_original_logit_uses_frozen_float64_components():
    base = sparse.csr_matrix([[1.0, 0.0], [0.25, 2.0]])
    position = sparse.csr_matrix([[0.0, 1.0], [3.0, 0.0]])
    coef = np.asarray([0.2, -0.4])
    intercept = np.asarray([0.1])
    gamma = np.asarray([0.3, -0.2])

    z_bag, z_original = reconstruct_original_pos_logit(
        base, coef, intercept, position, gamma
    )

    expected_bag = intercept[0] + base @ coef
    expected = np.asarray(expected_bag + position @ gamma).reshape(-1)
    np.testing.assert_array_equal(z_bag, np.asarray(expected_bag).reshape(-1))
    np.testing.assert_array_equal(z_original, expected)


def test_zero_gamma_matches_incumbent_and_preserves_exact_routes():
    x = sparse.csr_matrix([[1.0, 0.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    z = np.asarray([-0.2, 0.1, 0.3, -0.4])
    present = np.asarray([True, True, False, True])
    no_known = np.asarray([False, True, True, True])
    p_a02 = np.asarray([0.8, 0.2, 0.7, 0.6])
    p_p = np.asarray([0.1, 0.3, 0.4, 0.9])
    p_pos = expit(z)
    mix = np.asarray([0.5 * p_pos[i] + 0.5 * p_a02[i] for i in range(4)])
    mix[2] = p_p[2]

    check = zero_gamma_equivalence(
        x, z, present_mask=present, p_a02=p_a02, p_p=p_p, mix50_original=mix
    )
    assert check["pass"]

    _, routed = route_candidate(
        x, z, np.asarray([0.15, -0.05]), present_mask=present,
        no_known_bigram=no_known, p_pos_original=p_pos, p_a02=p_a02,
        p_p=p_p, mix50_original=mix,
    )
    assert routed[2] == p_p[2]
    np.testing.assert_array_equal(routed[[1, 3]], mix[[1, 3]])


def test_rejects_nonfinite_and_misaligned_inputs():
    with np.testing.assert_raises(ValueError):
        reconstruct_original_pos_logit(
            sparse.csr_matrix([[np.nan]]), [1.0], [0.0],
            sparse.csr_matrix([[1.0]]), [1.0],
        )
    with np.testing.assert_raises(ValueError):
        reconstruct_original_pos_logit(
            sparse.csr_matrix([[1.0]]), [1.0], [0.0],
            sparse.csr_matrix([[1.0, 0.0]]), [1.0],
        )


def test_fit_label_reader_skips_query_rows(tmp_path):
    path = tmp_path / "synthetic.csv"
    path.write_text(
        "record_id,artifact_signal,signal_stream\n"
        "r0,1,alpha\n"
        "r1,0,beta\n"
        "r2,1,gamma\n"
        "r3,0,delta\n",
        encoding="utf-8",
    )
    np.testing.assert_array_equal(_read_fit_labels_only(path, np.asarray([0, 2])), [1, 1])


def test_gate_uses_frozen_tolerance_and_unknown_bound_shape():
    assert _gate("min", 0.0005 - 5e-13, minimum=0.0005)["pass"]
    assert not _gate("min", 0.0005 - 2e-12, minimum=0.0005)["pass"]
    assert _gate("max", 0.003 + 5e-13, maximum=0.003)["pass"]
    assert not _gate("max", 0.003 + 2e-12, maximum=0.003)["pass"]
