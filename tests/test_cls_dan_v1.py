from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src.models.cls_dan_v1 import (
    ARCHITECTURES,
    EMBEDDING_WIDTH,
    adam_update,
    build_mean_unigram_counts,
    fit_fixed_recipe,
    initialize_parameters,
    loss_and_gradients,
    model_arrays,
    model_from_arrays,
    predict_probabilities,
)


def _tiny_inputs():
    counts = sparse.csr_matrix(np.asarray([
        [1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [0.7, 0.0, 0.3], [0.0, 0.4, 0.6],
    ], dtype=np.float64))
    structured = np.asarray([
        [0.0, 1.0], [0.2, 0.0], [0.4, 1.0], [0.6, 0.0], [0.8, 1.0], [1.0, 0.0]
    ], dtype=np.float64)
    labels = np.asarray([0, 0, 1, 1, 0, 1], dtype=np.float64)
    return counts, structured, labels


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_complete_gradients_with_fixed_dropout_masks(architecture: str):
    counts, structured, labels = _tiny_inputs()
    rng = np.random.default_rng(21)
    params = initialize_parameters(counts.shape[1], structured.shape[1], architecture, rng)
    masks = None
    if architecture == "DAN":
        mask1 = np.ones((len(labels), 64), dtype=np.float64)
        mask2 = np.ones((len(labels), 32), dtype=np.float64)
        mask1[0, 1] = 0.0
        mask2[2, 3] = 0.0
        masks = (mask1, mask2)
    loss, gradients = loss_and_gradients(params, counts, structured, labels, architecture, masks)
    assert np.isfinite(loss)
    finite_difference_checks = [("E", (0, 0))]
    if architecture == "DAN":
        finite_difference_checks += [("W1", (0, 0)), ("W2", (0, 0)), ("W3", (0, 0)), ("b1", (0,))]
    else:
        finite_difference_checks += [("Wout", (0, 0)), ("bout", (0,))]
    epsilon = 1e-6
    for name, index in finite_difference_checks:
        original = params[name][index]
        params[name][index] = original + epsilon
        plus = loss_and_gradients(params, counts, structured, labels, architecture, masks)[0]
        params[name][index] = original - epsilon
        minus = loss_and_gradients(params, counts, structured, labels, architecture, masks)[0]
        params[name][index] = original
        numerical = (plus - minus) / (2.0 * epsilon)
        assert numerical == pytest.approx(gradients[name][index], abs=3e-6, rel=3e-4)


def test_mean_unigram_counts_preserve_multiplicity_and_audit_oov():
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, token_pattern=r"(?u)\b\w\w+\b")
    vectorizer.fit(["Alpha alpha beta", "beta gamma delta"])
    matrix, audit = build_mean_unigram_counts(
        ["Alpha alpha beta", "mysterytoken", ""], vectorizer
    )
    terms, _ = __import__("src.models.cls_dan_v1", fromlist=["unigram_terms"]).unigram_terms(vectorizer)
    alpha = terms.index("alpha")
    beta = terms.index("beta")
    assert matrix.shape == (3, len(terms))
    assert matrix[0, alpha] == pytest.approx(2.0 / 3.0)
    assert matrix[0, beta] == pytest.approx(1.0 / 3.0)
    assert matrix[0].sum() == pytest.approx(1.0)
    assert matrix[1].nnz == 0
    assert matrix[2].nnz == 0
    assert audit["known_token_counts"].tolist() == [3, 0, 0]
    assert audit["all_oov_rows"] == 1
    assert audit["empty_token_rows"] == 1
    assert audit["coverage"].tolist() == pytest.approx([1.0, 0.0, 0.0])


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_adam_single_step_matches_closed_form(architecture: str):
    del architecture  # The optimizer is architecture-agnostic.
    params = {"w": np.asarray([1.0], dtype=np.float64)}
    gradients = {"w": np.asarray([0.5], dtype=np.float64)}
    zeros = {"w": np.asarray([0.0], dtype=np.float64)}
    updated, first, second = adam_update(params, gradients, zeros, zeros, 0)
    expected_m = (1.0 - 0.9) * 0.5
    expected_v = (1.0 - 0.999) * 0.25
    expected = 1.0 - 1e-3 * (expected_m / 0.1) / (np.sqrt(expected_v / 0.001) + 1e-8)
    assert first["w"][0] == pytest.approx(expected_m)
    assert second["w"][0] == pytest.approx(expected_v)
    assert updated["w"][0] == pytest.approx(expected, abs=1e-12)


def test_sparse_dense_prediction_parity_and_state_roundtrip(tmp_path: Path):
    counts, structured, labels = _tiny_inputs()
    model = fit_fixed_recipe(counts, structured, labels, "LINEAR")
    p_sparse = predict_probabilities(model, counts, structured)
    p_dense = predict_probabilities(model, counts.toarray(), structured)
    np.testing.assert_array_equal(p_sparse, p_dense)
    path = tmp_path / "linear.npz"
    np.savez_compressed(path, **model_arrays(model))
    with np.load(path, allow_pickle=False) as archive:
        restored = model_from_arrays({key: archive[key] for key in archive.files})
    np.testing.assert_array_equal(p_sparse, predict_probabilities(restored, counts, structured))
    for name in model["params"]:
        np.testing.assert_array_equal(model["params"][name], restored["params"][name])
        np.testing.assert_array_equal(model["adam_first_moment"][name], restored["adam_first_moment"][name])
        np.testing.assert_array_equal(model["adam_second_moment"][name], restored["adam_second_moment"][name])
    assert restored["adam_steps"] == model["adam_steps"]
    assert restored["training"] == model["training"]
    assert restored["params"]["E"].shape[1] == EMBEDDING_WIDTH


def test_fixed_recipe_reduces_no_dropout_training_loss_and_repeats_exactly():
    rng = np.random.default_rng(5)
    token = rng.integers(0, 2, size=80)
    counts = np.zeros((80, 3), dtype=np.float64)
    counts[np.arange(80), token] = 1.0
    structured = rng.normal(size=(80, 2)).astype(np.float64)
    labels = token.astype(np.float64)
    model1 = fit_fixed_recipe(counts, structured, labels, "DAN")
    model2 = fit_fixed_recipe(counts, structured, labels, "DAN")
    assert model1["training"]["epochs_completed"] == 20
    assert model1["training"]["converged_claim"] is False
    assert model1["training"]["final_no_dropout_loss"]["total_objective"] < model1["training"]["initial_no_dropout_loss"]["total_objective"]
    np.testing.assert_array_equal(predict_probabilities(model1, counts, structured),
                                  predict_probabilities(model2, counts, structured))
    for name in model1["params"]:
        np.testing.assert_array_equal(model1["params"][name], model2["params"][name])


def test_synthetic_cli_lifecycle_is_self_contained(tmp_path: Path):
    from scripts.cls_dan_v1 import run_synthetic_lifecycle

    output = run_synthetic_lifecycle(tmp_path)
    assert output["status"] == "PASS_SYNTHETIC_ONLY"
    assert output["official_data_opened"] is False
    assert output["query_targets_used_for_synthetic_metric_only"] is True
    assert output["official_fit_starts"] == 0
    assert set(output["models"]) == {"DAN", "LINEAR"}
    assert output["readback_probability_max_abs_diff"] == 0.0
    assert output["metric_verifier_match"] is True
    assert (tmp_path / "SYNTHETIC_QA.json").is_file()
    receipt = json.loads((tmp_path / "SYNTHETIC_QA.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "PASS_SYNTHETIC_ONLY"


def test_metric_verifier_handles_tied_probabilities_and_one_class():
    from scripts.verify_cls_dan_v1 import verify_against_sklearn

    tied = verify_against_sklearn(
        np.asarray([1, 0, 1, 0], dtype=np.int8),
        np.asarray([0.8, 0.8, 0.2, 0.1], dtype=np.float64),
        np.ones(4, dtype=bool),
    )
    assert tied["status"] == "PASS"
    all_positive = verify_against_sklearn(
        np.ones(3, dtype=np.int8), np.asarray([0.9, 0.6, 0.4]), np.ones(3, dtype=bool)
    )
    assert all_positive["status"] == "PASS"
    assert all_positive["manual"]["roc_auc"] is None
