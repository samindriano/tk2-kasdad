from __future__ import annotations

import numpy as np
import pytest

from scripts import optimization_v3_cls_pos_a02_meta_v1 as frozen
from scripts import optimization_v3_cls_pos_a02_meta_v1_warning_recovery as recovery
from src.models.research_next_contract import file_sha256


PENALTY_SOURCE = """
warnings.warn(
    \"'penalty' was deprecated in version 1.8 and will be removed in 1.10.\",
    FutureWarning,
)
"""


def _state_and_receipt(tmp_path, warnings=None, n_iter=6):
    path = tmp_path / "state.npz"
    state = {
        "coef": np.asarray([[0.4, -0.2]], dtype=np.float64),
        "intercept": np.asarray([0.1], dtype=np.float64),
        "classes": np.asarray([0, 1], dtype=np.int8),
        "n_iter": np.asarray([n_iter], dtype=np.int32),
    }
    np.savez(path, **state)
    receipt = {
        "model_state_file_sha256": file_sha256(path),
        "feature_width": 2,
        "n_iter": [n_iter],
        "convergence_warnings": [] if warnings is None else warnings,
        "estimator": dict(frozen.MODEL_PARAMS),
    }
    return path, receipt, state


def test_accepts_exact_source_verified_sklearn_18_penalty_futurewarning(tmp_path):
    path, receipt, expected = _state_and_receipt(tmp_path, ["FutureWarning"])
    actual = recovery.read_model_state(path, receipt,
                                       sklearn_version="1.8.0",
                                       logistic_source=PENALTY_SOURCE)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])


def test_accepts_no_warning_state(tmp_path):
    path, receipt, expected = _state_and_receipt(tmp_path)
    actual = recovery.read_model_state(path, receipt,
                                       sklearn_version="1.8.0",
                                       logistic_source=PENALTY_SOURCE)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])


@pytest.mark.parametrize("categories", [["ConvergenceWarning"], ["FutureWarning", "ConvergenceWarning"],
                                         ["RuntimeWarning"], ["FutureWarning", "FutureWarning"]])
def test_rejects_convergence_unknown_or_duplicate_warnings(tmp_path, categories):
    path, receipt, _ = _state_and_receipt(tmp_path, categories)
    with pytest.raises(recovery.RecoveryError):
        recovery.read_model_state(path, receipt,
                                  sklearn_version="1.8.0",
                                  logistic_source=PENALTY_SOURCE)


def test_rejects_futurewarning_from_other_version_or_unrelated_parameters(tmp_path):
    path, receipt, _ = _state_and_receipt(tmp_path, ["FutureWarning"])
    with pytest.raises(recovery.RecoveryError):
        recovery.read_model_state(path, receipt,
                                  sklearn_version="1.9.0",
                                  logistic_source=PENALTY_SOURCE)
    receipt["estimator"]["penalty"] = "l1"
    with pytest.raises(recovery.RecoveryError):
        recovery.read_model_state(path, receipt,
                                  sklearn_version="1.8.0",
                                  logistic_source=PENALTY_SOURCE)


def test_rejects_iteration_mismatch_and_nonconvergence_count(tmp_path):
    path, receipt, _ = _state_and_receipt(tmp_path)
    receipt["n_iter"] = [7]
    with pytest.raises(recovery.RecoveryError, match="iteration count"):
        recovery.read_model_state(path, receipt,
                                  sklearn_version="1.8.0",
                                  logistic_source=PENALTY_SOURCE)
    path, receipt, _ = _state_and_receipt(tmp_path, n_iter=frozen.MAX_ITER)
    with pytest.raises(recovery.RecoveryError, match="did not converge"):
        recovery.read_model_state(path, receipt,
                                  sklearn_version="1.8.0",
                                  logistic_source=PENALTY_SOURCE)


def test_rejects_state_hash_mismatch(tmp_path):
    path, receipt, _ = _state_and_receipt(tmp_path)
    receipt["model_state_file_sha256"] = "0" * 64
    with pytest.raises(recovery.RecoveryError, match="hash"):
        recovery.read_model_state(path, receipt,
                                  sklearn_version="1.8.0",
                                  logistic_source=PENALTY_SOURCE)
