from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse
from pathlib import Path
import json

from scripts import optimization_v3_cls_pos_frequency_ridge_v1 as runner
from scripts.verify_optimization_v3_cls_pos_frequency_ridge_v1 import (
    VerificationError,
    sha256_file,
    verify_gamma_artifact,
    verify_source_preflight,
)
from src.models.pos_frequency_ridge_v1 import (
    FrequencyRidgeError,
    enforce_fit_cap,
    fit_resume_decision,
    fit_weighted_offset,
    guarded_target_read,
    make_penalty_maps,
    predict_with_fallback,
    repeat_token_weights,
    sha256_array,
    union_document_frequency,
    weighted_offset_objective_gradient,
)
from src.models.positional_bag_offset_v1 import offset_objective_gradient


def test_weighted_gradient_matches_finite_difference() -> None:
    x = sparse.csr_matrix(np.asarray([
        [1.0, 0.0, 0.2], [0.0, 1.0, -0.4], [1.0, 1.0, 0.0],
        [-0.3, 0.0, 0.7], [0.0, -0.5, 0.2], [0.1, 0.4, 0.8],
    ], dtype=np.float64))
    offset = np.asarray([-1.1, 0.4, 1.2, -0.2, 0.9, -0.7])
    y = np.asarray([0, 1, 1, 0, 1, 0], dtype=np.int8)
    gamma = np.asarray([0.2, -0.35, 0.1])
    weights = np.asarray([0.7, 1.3, 2.0])
    loss, gradient = weighted_offset_objective_gradient(gamma, x, offset, y, weights)
    eps = 1e-6
    numeric = np.empty_like(gamma)
    for j in range(len(gamma)):
        plus, minus = gamma.copy(), gamma.copy()
        plus[j] += eps
        minus[j] -= eps
        numeric[j] = (
            weighted_offset_objective_gradient(plus, x, offset, y, weights)[0]
            - weighted_offset_objective_gradient(minus, x, offset, y, weights)[0]
        ) / (2 * eps)
    assert np.isfinite(loss)
    assert np.allclose(gradient, numeric, rtol=2e-6, atol=2e-7)


def test_all_ones_weighted_objective_equals_historical_isotropic_objective() -> None:
    x = sparse.csr_matrix(np.asarray([[1.0, 0.0], [0.3, -0.4], [0.0, 2.0]]))
    offset = np.asarray([-0.3, 0.7, 1.2])
    y = np.asarray([0, 1, 1], dtype=np.int8)
    gamma = np.asarray([0.2, -0.15])
    new_loss, new_gradient = weighted_offset_objective_gradient(
        gamma, x, offset, y, np.ones_like(gamma)
    )
    old_loss, old_gradient = offset_objective_gradient(gamma, x, offset, y, c=1.0)
    assert new_loss == pytest.approx(old_loss, rel=0, abs=1e-14)
    assert np.array_equal(new_gradient, old_gradient)


def test_union_df_counts_documents_not_occurrences_or_third_blocks() -> None:
    # Token 0 occurs twice in row 0 but contributes one document to its df.
    x = sparse.csr_matrix((np.asarray([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float64),
                           (np.asarray([0, 0, 1, 1, 2, 2, 3, 3]),
                            np.asarray([0, 3, 8, 4, 6, 2, 1, 7]))), shape=(4, 9))
    assert union_document_frequency(x, vocabulary_size=3).tolist() == [2, 2, 2]


def test_union_df_does_not_count_explicit_sparse_zeros_as_present() -> None:
    x = sparse.csr_matrix((np.asarray([0.0, 1.0]), np.asarray([0, 1]), np.asarray([0, 1, 2])),
                          shape=(2, 3))
    # Vocabulary size one: the explicit zero in the first third is absent;
    # the second row's token in the second third is present.
    assert union_document_frequency(x, vocabulary_size=1).tolist() == [1]


def test_penalty_map_is_normalized_and_permutation_preserves_multiset() -> None:
    first = make_penalty_maps([1, 2, 4, 8, 16])
    second = make_penalty_maps([1, 2, 4, 8, 16])
    aligned = first["aligned_token_weights"]
    permuted = first["permuted_token_weights"]
    assert np.mean(aligned) == pytest.approx(1.0, abs=1e-14)
    assert np.array_equal(np.sort(aligned), np.sort(permuted))
    assert np.array_equal(aligned, second["aligned_token_weights"])
    assert np.array_equal(permuted, second["permuted_token_weights"])
    assert not first["aligned_equals_permuted"]
    repeated = repeat_token_weights(aligned)
    assert np.array_equal(repeated, np.tile(aligned, 3))


def test_rejects_noninteger_document_frequency_and_fractional_fit_target() -> None:
    with pytest.raises(FrequencyRidgeError, match="positive token document frequencies"):
        make_penalty_maps([1.5, 2.0])
    with pytest.raises(FrequencyRidgeError, match="before conversion"):
        fit_weighted_offset(sparse.csr_matrix([[1.0], [0.0]]), [0.0, 0.1], [0.25, 1.0], [1.0])


def test_missing_stream_prediction_is_exactly_fallback() -> None:
    x = sparse.csr_matrix([[1.0, 0.0], [0.0, 1.0], [0.2, 0.3]])
    fallback = np.asarray([0.123456789, 0.7, 0.314159265])
    output = predict_with_fallback(x, [0.1, 0.2, -0.3], [0.4, -0.2],
                                   [True, False, False], fallback)
    assert output[0] != fallback[0]
    assert np.array_equal(output[1:], fallback[1:])


def test_hash_and_id_guard_fails_before_target_reader() -> None:
    called = False

    def reader():
        nonlocal called
        called = True
        return np.asarray([0, 1])

    with pytest.raises(FrequencyRidgeError, match="source hashes"):
        guarded_target_read(expected_hashes={"train": "a"}, actual_hashes={"train": "b"},
                            expected_ids_sha256="id-a", actual_ids_sha256="id-a", reader=reader)
    assert called is False
    with pytest.raises(FrequencyRidgeError, match="fit IDs"):
        guarded_target_read(expected_hashes={"train": "a"}, actual_hashes={"train": "a"},
                            expected_ids_sha256="id-a", actual_ids_sha256="id-b", reader=reader)
    assert called is False


def test_resume_uses_durable_save_without_duplicate_start_and_cap_is_exact() -> None:
    fit_ids = ["E67_ALIGNED", "E67_PERMUTED", "E75_ALIGNED", "E75_PERMUTED"]
    events = [{"event": "FIT_STARTED", "fit_id": fit_ids[0]},
              {"event": "FIT_SAVED", "fit_id": fit_ids[0]}]
    assert fit_resume_decision(events, fit_ids[0], fit_ids) == "RESUME_SAVED"
    assert fit_resume_decision(events, fit_ids[1], fit_ids) == "START_NEW"
    enforce_fit_cap(events, fit_ids)
    with pytest.raises(FrequencyRidgeError, match="namespace cap"):
        enforce_fit_cap(events + [{"event": "FIT_STARTED", "fit_id": "UNAUTHORIZED"}], fit_ids)
    with pytest.raises(FrequencyRidgeError, match="pending slot"):
        fit_resume_decision([{"event": "FIT_STARTED", "fit_id": fit_ids[1]}], fit_ids[1], fit_ids)
    with pytest.raises(FrequencyRidgeError, match="no earlier matching"):
        fit_resume_decision([{"event": "FIT_SAVED", "fit_id": fit_ids[2]}], fit_ids[2], fit_ids)
    with pytest.raises(FrequencyRidgeError, match="no earlier matching"):
        fit_resume_decision([{"event": "FIT_SAVED", "fit_id": fit_ids[2]},
                             {"event": "FIT_STARTED", "fit_id": fit_ids[2]}], fit_ids[2], fit_ids)


def test_synthetic_weighted_optimizer_converges_from_zero() -> None:
    rng = np.random.default_rng(19)
    x = sparse.csr_matrix(rng.normal(size=(120, 9)))
    offset = rng.normal(size=120)
    p = 1.0 / (1.0 + np.exp(-(offset + x @ rng.normal(scale=0.1, size=9))))
    y = rng.binomial(1, p).astype(np.int8)
    gamma, metadata = fit_weighted_offset(x, offset, y, np.ones(9))
    assert metadata["success"] is True
    assert metadata["loss_final"] <= metadata["loss_gamma0"] + 1e-8
    assert np.isfinite(gamma).all()


def test_runner_synthetic_lifecycle_passes_without_official_inputs() -> None:
    result = runner.run_synthetic_qa()
    assert result["status"] == "PASS"
    assert result["durable_resume_without_duplicate_fit"] is True
    assert result["official_inputs_opened"] is False
    assert result["official_labels_opened"] is False


def test_official_fit_requires_committed_root_source_pass(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(runner, "ROOT_SOURCE_PASS", tmp_path / "ROOT_SOURCE_PASS.json")
    with pytest.raises(runner.FrequencyRidgeRunnerError, match="committed ROOT_SOURCE_PASS"):
        runner._require_root_source_pass()


def test_independent_preflight_verifier_checks_scope_flags_and_populations(tmp_path: Path) -> None:
    report = {
        "campaign_id": "CLS_POS_FREQUENCY_RIDGE_V1",
        "source_commit": "a" * 40,
        "source_hashes": {"a.py": "b" * 64},
        "train_sha256": runner.TRAIN_SHA256,
        "target_values_opened": False,
        "test_or_sample_submission_accessed": False,
        "kaggle_accessed": False,
        "official_fit_started": False,
        "origin_identity_and_cache_checks": {
            "E67": {"fit_rows": 29705, "score_rows": 20903},
            "E75": {"fit_rows": 42117, "score_rows": 8491},
        },
    }
    path = tmp_path / "preflight.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert verify_source_preflight(path)["status"] == "PASS_SOURCE_PREFLIGHT_INDEPENDENT"
    report["test_or_sample_submission_accessed"] = True
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(VerificationError, match="forbidden access"):
        verify_source_preflight(path)


def test_independent_gamma_verifier_binds_file_and_vector_hash(tmp_path: Path) -> None:
    gamma = np.asarray([0.1, -0.2, 0.3], dtype=np.float64)
    artifact = tmp_path / "gamma.npz"
    with artifact.open("wb") as stream:
        np.savez_compressed(stream, gamma=gamma)
    receipt = {
        "fit_kind": "PREDICTOR",
        "campaign_id": "CLS_POS_FREQUENCY_RIDGE_V1",
        "fit_id": "E67_ALIGNED",
        "gamma_sha256": sha256_array(gamma),
        "artifact_sha256": sha256_file(artifact),
    }
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    assert verify_gamma_artifact(artifact, receipt_path)["gamma_width"] == 3
    receipt["artifact_sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(VerificationError, match="file hash"):
        verify_gamma_artifact(artifact, receipt_path)
