from __future__ import annotations

import numpy as np
from scipy import sparse

from scripts import optimization_v3_positional_contrast_real_e79_exploratory_v1 as benchmark
from src.models.positional_contrast_v1 import contrast_block


def test_contrast_block_uses_frozen_projection_without_post_normalization() -> None:
    base = sparse.csr_matrix([[9.0], [8.0]])
    first = sparse.csr_matrix([[1.0, 0.0], [0.0, 2.0]])
    middle = sparse.csr_matrix([[0.0, 1.0], [1.0, 0.0]])
    last = sparse.csr_matrix([[2.0, 0.0], [0.0, 1.0]])
    augmented = sparse.hstack((base, first, middle, last), format="csr")

    projected = contrast_block(augmented, base.shape[1]).toarray()
    expected = np.column_stack((
        ((first - last) / np.sqrt(2.0)).toarray(),
        ((first - 2.0 * middle + last) / np.sqrt(6.0)).toarray(),
    ))
    assert np.array_equal(projected, expected)
    assert not np.allclose(np.linalg.norm(projected, axis=1), 1.0)


def test_synthetic_full_lifecycle_and_resume_contract() -> None:
    result = benchmark.run_synthetic()
    assert result["status"] == "PASS_SYNTHETIC_END_TO_END_AND_SAFE_RESUME"
    assert result["official_inputs_opened"] is False
    assert result["official_fit_count"] == 0
    assert result["synthetic_gamma_fit_count"] == 1
    assert result["gamma_zero_replay"] == "PASS"
    assert result["missing_fallback_exact"] is True


def test_resource_gate_distinguishes_tiny_background_pythonw_from_training() -> None:
    assert benchmark._classify_python_process(
        "pythonw.exe", "D:/Bot/scripts/run_service_task.py", 20 * 1024**2
    ) == "lightweight_pythonw_background"
    assert benchmark._classify_python_process(
        "python.exe", "python scripts/optimization_v3_regression.py fit", 20 * 1024**2
    ) == "potential_training"
