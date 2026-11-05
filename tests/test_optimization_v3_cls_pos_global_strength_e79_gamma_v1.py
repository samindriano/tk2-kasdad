from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy import sparse

from src.models import cls_pos_global_strength_e79_v1 as recipe


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, *args], cwd=ROOT, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def test_global_strength_penalty_is_one_scalar_repeated_across_three_blocks() -> None:
    aligned = np.asarray([0.5, 1.0, 1.5, 1.2], dtype=np.float64)
    penalty, scalar = recipe.global_strength_penalty(aligned)
    expected = float(np.mean(np.maximum(1.0, aligned)))
    assert scalar == expected
    np.testing.assert_array_equal(penalty, np.full(12, expected, dtype=np.float64))


def test_present_blend_and_missing_fallback_are_exact() -> None:
    present = np.asarray([True, False, True, False])
    positional_probability = np.asarray([0.8, np.nan, 0.3, np.nan], dtype=np.float64)
    p_a02 = np.asarray([0.2, 0.71, 0.9, 0.42], dtype=np.float64)
    result = recipe.combine_present_and_missing_probabilities(positional_probability, p_a02, present)
    np.testing.assert_array_equal(result[present], 0.5 * positional_probability[present] + 0.5 * p_a02[present])
    np.testing.assert_array_equal(result[~present], p_a02[~present])
    labels = recipe.apply_frozen_thresholds(result, present)
    np.testing.assert_array_equal(labels, np.asarray([1, 1, 1, 0], dtype=np.int8))


def test_frozen_thresholds_include_exact_boundary() -> None:
    probability = np.asarray([recipe.PRESENT_THRESHOLD, recipe.MISSING_THRESHOLD], dtype=np.float64)
    present = np.asarray([True, False])
    np.testing.assert_array_equal(recipe.apply_frozen_thresholds(probability, present),
                                  np.asarray([1, 1], dtype=np.int8))


def test_synthetic_cli_fit_receipt_csv_and_resume_are_one_shot(tmp_path: Path) -> None:
    output_dir = tmp_path / "e79-synthetic"
    fit = _run("-m", "scripts.optimization_v3_cls_pos_global_strength_e79_gamma_v1",
               "--synthetic-e2e", "--output-dir", str(output_dir))
    assert fit.returncode == 0, fit.stdout + fit.stderr
    result = json.loads(fit.stdout)
    assert result["status"] == "PASS_E79_SYNTHETIC_END_TO_END"
    assert result["actual_optimizer_fit_count"] == 1

    attempt_path = output_dir / "fit" / "ATTEMPT.started.json"
    attempt_sha_before = _sha(attempt_path)
    independent = _run("-m", "scripts.verify_optimization_v3_cls_pos_global_strength_e79_gamma_v1",
                       "--verify-synthetic-e2e", "--output-dir", str(output_dir))
    assert independent.returncode == 0, independent.stdout + independent.stderr
    verified = json.loads(independent.stdout)
    assert verified["status"] == "PASS_E79_SYNTHETIC_RESUME_VALIDATED_NO_REFIT"
    assert verified["attempt_count"] == 1
    assert verified["additional_fit_count"] == 0

    resume = _run("-m", "scripts.optimization_v3_cls_pos_global_strength_e79_gamma_v1",
                  "--resume-synthetic-e2e", "--output-dir", str(output_dir))
    assert resume.returncode == 0, resume.stdout + resume.stderr
    assert json.loads(resume.stdout)["fit_count"] == 1
    assert _sha(attempt_path) == attempt_sha_before

    retry = _run("-m", "scripts.optimization_v3_cls_pos_global_strength_e79_gamma_v1",
                 "--synthetic-e2e", "--output-dir", str(output_dir))
    assert retry.returncode == 1
    assert "one-shot" in retry.stdout.lower()
    assert _sha(attempt_path) == attempt_sha_before


def test_route_and_combine_preserves_missing_rows_bit_exactly() -> None:
    x = sparse.csr_matrix(np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64))
    offset = np.asarray([0.1, -0.4], dtype=np.float64)
    gamma = np.asarray([0.3, -0.1, 0.2], dtype=np.float64)
    present = np.asarray([True, False])
    p_a02 = np.asarray([0.42, 0.73123456789], dtype=np.float64)
    result = recipe.route_and_combine(x, offset, gamma, present, p_a02)
    assert result[1].tobytes() == p_a02[1].tobytes()
