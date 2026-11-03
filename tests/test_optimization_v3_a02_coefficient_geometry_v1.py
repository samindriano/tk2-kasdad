from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src.models.a02_coefficient_geometry_v1 import (
    CoefficientGeometryPredictor,
    coefficient_weights,
)
from src.models.classification_a02_neighbors import A02TextSpace, fit_text_space
from scripts import optimization_v3_a02_coefficient_geometry_v1 as runner


def _original_neighbors(matrix, query, rows, width=32):
    similarity = np.asarray((query @ matrix.T).toarray()[0], dtype=np.float64)
    positive = np.flatnonzero(similarity > 0.0)
    order = np.lexsort((rows[positive], -similarity[positive]))
    chosen = positive[order[:width]]
    out_rows = np.full((1, width), -1, dtype=np.int64)
    out_sims = np.zeros((1, width), dtype=np.float64)
    out_rows[0, :len(chosen)] = rows[chosen]
    out_sims[0, :len(chosen)] = similarity[chosen]
    return out_rows, out_sims


def _small_predictor_fixture():
    prefix_texts = [
        "alpha beta shared", "alpha beta shared", "alpha beta shared", "alpha beta shared",
        "alpha shared", "alpha shared", "alpha shared", "alpha shared",
        "beta shared", "beta shared", "beta shared", "beta shared",
        "other shared", "other shared", "other shared", "other shared",
    ]
    prefix_rows = np.arange(len(prefix_texts), dtype=np.int64)
    prefix_cycles = np.full(len(prefix_rows), 48, dtype=np.int64)
    text_space, _, _ = fit_text_space(prefix_texts, prefix_rows, prefix_cycles, origin_cycle=48)
    bank_texts = prefix_texts.copy()
    bank_matrix = text_space.vectorizer.transform(bank_texts).tocsr()
    bank_rows = np.arange(len(bank_texts), dtype=np.int64)
    bank_cycles = np.full(len(bank_texts), 48, dtype=np.int64)
    ids = np.asarray([f"bank-{i}" for i in bank_rows], dtype=str)
    probabilities = np.linspace(0.1, 0.9, len(bank_rows), dtype=np.float64)
    residuals = np.linspace(-0.2, 0.3, len(bank_rows), dtype=np.float64)
    query_texts = ["alpha beta shared", "absent-token-only"]
    query_rows = np.asarray([100, 101], dtype=np.int64)
    query_ids = np.asarray(["query-α-long-" + "x" * 100, "query-unknown"], dtype=str)
    query_cycles = np.asarray([49, 49], dtype=np.int64)
    query_matrix, missing = text_space.transform(query_texts)
    original_rows = np.full((2, 32), -1, dtype=np.int64)
    original_sims = np.zeros((2, 32), dtype=np.float64)
    for i in range(2):
        selected_rows, selected_sims = _original_neighbors(
            bank_matrix, query_matrix[i], bank_rows
        )
        original_rows[i] = selected_rows[0]
        original_sims[i] = selected_sims[0]
    baseline = np.asarray([0.35, 0.6], dtype=np.float64)
    m0 = np.sum(original_sims * original_sims, axis=1)
    original_prediction = baseline.copy()
    for i in range(2):
        chosen = np.flatnonzero(original_rows[i] >= 0)
        if len(chosen):
            lookup = {int(row): j for j, row in enumerate(bank_rows)}
            idx = np.asarray([lookup[int(original_rows[i, j])] for j in chosen], dtype=np.int64)
            raw = np.dot(original_sims[i, chosen] ** 2, residuals[idx]) / (4.0 + m0[i])
            original_prediction[i] = np.clip(baseline[i] + np.clip(raw, -0.05, 0.05), 0.0, 1.0)
    return {
        "text_space": text_space, "bank_matrix": bank_matrix, "bank_rows": bank_rows,
        "bank_cycles": bank_cycles, "bank_ids": ids, "probabilities": probabilities,
        "residuals": residuals, "query_texts": query_texts, "query_rows": query_rows,
        "query_ids": query_ids, "query_cycles": query_cycles, "query_matrix": query_matrix,
        "missing": missing, "original_rows": original_rows, "original_sims": original_sims,
        "baseline": baseline, "original_prediction": original_prediction,
    }


def _predict(fixture, beta):
    predictor = CoefficientGeometryPredictor(origin_cycle=48, arm="aligned").fit(
        text_space=fixture["text_space"], bank_matrix=fixture["bank_matrix"],
        bank_ids=fixture["bank_ids"], bank_row_positions=fixture["bank_rows"],
        bank_cycles=fixture["bank_cycles"], bank_oof_probabilities=fixture["probabilities"],
        bank_residuals=fixture["residuals"], beta_in_a02_column_order=beta,
    )
    return predictor.predict(
        query_texts=fixture["query_texts"], query_record_ids=fixture["query_ids"],
        query_row_positions=fixture["query_rows"], query_cycles=fixture["query_cycles"],
        baseline_probabilities=fixture["baseline"],
        original_a02_probabilities=fixture["original_prediction"],
        original_a02_neighbor_source_rows=fixture["original_rows"],
        original_a02_neighbor_similarities=fixture["original_sims"],
        missing_mask=fixture["missing"],
    )


def test_coefficient_formula_and_permutation_are_frozen_and_repeatable():
    beta = np.asarray([-10.0, -0.5, 0.0, 0.5, 10.0], dtype=np.float64)
    aligned, preserved = coefficient_weights(beta, arm="aligned")
    permuted1, _ = coefficient_weights(beta, arm="permuted")
    permuted2, _ = coefficient_weights(beta, arm="permuted")
    assert np.array_equal(preserved, beta)
    assert np.allclose(aligned, np.sqrt(1.0 + np.abs(beta) / (1.0 + np.abs(beta))))
    assert np.array_equal(permuted1, permuted2)
    assert np.array_equal(np.sort(aligned), np.sort(permuted1))
    with pytest.raises(ValueError):
        coefficient_weights([0.0, np.nan], arm="aligned")


def test_all_one_weights_recover_a02_and_fixed_mass():
    fixture = _small_predictor_fixture()
    result = _predict(fixture, np.zeros(len(fixture["text_space"].vectorizer.vocabulary_)))
    assert np.allclose(result["candidate_probability"], fixture["original_prediction"], atol=1e-12, rtol=0)
    assert np.allclose(result["neighbor_weights"].sum(axis=1), result["m0"], atol=1e-12, rtol=0)
    assert result["diagnostics"]["max_abs_mass_error"] <= 1e-12
    assert result["candidate_probability"][1] == fixture["baseline"][1]
    assert result["query_record_ids"][0] == fixture["query_ids"][0]
    assert len(result["query_record_ids"][0]) > 64


def test_full_bank_search_can_select_donor_outside_original_top32():
    vectorizer = TfidfVectorizer(use_idf=False, norm="l2", lowercase=False,
                                 token_pattern=r"(?u)\b\w+\b")
    vectorizer.fit(["alpha", "alpha", "beta", "beta", "alpha beta"])
    text_space = A02TextSpace(
        vectorizer=vectorizer, origin_cycle=48, fit_rows=5,
        fit_row_positions_sha256="synthetic", fit_documents_sha256="synthetic",
        vocabulary_sha256="synthetic", idf_sha256="synthetic",
    )
    bank_texts = ["alpha"] * 33 + ["beta"]
    bank_matrix = vectorizer.transform(bank_texts).tocsr()
    bank_rows = np.arange(40, 74, dtype=np.int64)
    bank_cycles = np.full(34, 48, dtype=np.int64)
    bank_ids = [f"id-{i}" for i in range(34)]
    probabilities = np.full(34, 0.5, dtype=np.float64)
    residuals = np.zeros(34, dtype=np.float64)
    residuals[-1] = 0.8
    query = ["alpha beta"]
    qmat = vectorizer.transform(query).tocsr()
    original_rows, original_sims = _original_neighbors(bank_matrix, qmat, bank_rows)
    assert bank_rows[-1] not in original_rows[0]
    beta = np.zeros(len(vectorizer.vocabulary_), dtype=np.float64)
    beta[vectorizer.vocabulary_["beta"]] = 100.0
    predictor = CoefficientGeometryPredictor(origin_cycle=48, arm="aligned").fit(
        text_space=text_space, bank_matrix=bank_matrix, bank_ids=bank_ids,
        bank_row_positions=bank_rows, bank_cycles=bank_cycles,
        bank_oof_probabilities=probabilities, bank_residuals=residuals,
        beta_in_a02_column_order=beta,
    )
    m0 = np.sum(original_sims * original_sims, axis=1)
    result = predictor.predict(
        query_texts=query, query_record_ids=["outside-top32-query"],
        query_row_positions=np.asarray([1000]), query_cycles=np.asarray([49]),
        baseline_probabilities=np.asarray([0.2]), original_a02_probabilities=np.asarray([0.2]),
        original_a02_neighbor_source_rows=original_rows,
        original_a02_neighbor_similarities=original_sims, missing_mask=np.asarray([False]),
    )
    assert bank_rows[-1] in result["neighbor_source_rows"][0]
    assert result["candidate_probability"][0] > 0.2
    assert np.isclose(result["neighbor_weights"].sum(), m0[0], atol=1e-12, rtol=0)


def test_rejects_future_or_overlapping_bank_and_query_rows():
    fixture = _small_predictor_fixture()
    with pytest.raises(ValueError, match="future cycle"):
        CoefficientGeometryPredictor(origin_cycle=48, arm="aligned").fit(
            text_space=fixture["text_space"], bank_matrix=fixture["bank_matrix"],
            bank_ids=fixture["bank_ids"], bank_row_positions=fixture["bank_rows"],
            bank_cycles=np.full(len(fixture["bank_rows"]), 49),
            bank_oof_probabilities=fixture["probabilities"], bank_residuals=fixture["residuals"],
            beta_in_a02_column_order=np.zeros(len(fixture["text_space"].vectorizer.vocabulary_)),
        )
    predictor = CoefficientGeometryPredictor(origin_cycle=48, arm="aligned").fit(
        text_space=fixture["text_space"], bank_matrix=fixture["bank_matrix"],
        bank_ids=fixture["bank_ids"], bank_row_positions=fixture["bank_rows"],
        bank_cycles=fixture["bank_cycles"], bank_oof_probabilities=fixture["probabilities"],
        bank_residuals=fixture["residuals"],
        beta_in_a02_column_order=np.zeros(len(fixture["text_space"].vectorizer.vocabulary_)),
    )
    with pytest.raises(ValueError, match="overlap"):
        predictor.predict(
            query_texts=[fixture["query_texts"][0]], query_record_ids=[fixture["bank_ids"][0]],
            query_row_positions=np.asarray([0]), query_cycles=np.asarray([49]),
            baseline_probabilities=np.asarray([0.3]), original_a02_probabilities=np.asarray([0.3]),
            original_a02_neighbor_source_rows=fixture["original_rows"][:1],
            original_a02_neighbor_similarities=fixture["original_sims"][:1],
            missing_mask=np.asarray([False]),
        )


def test_synthetic_lifecycle_unicode_roundtrip_resume_and_production_writer_guard(tmp_path: Path):
    fixture = _small_predictor_fixture()
    beta = np.zeros(len(fixture["text_space"].vectorizer.vocabulary_), dtype=np.float64)
    output = _predict(fixture, beta)
    payload = {
        **output,
        "diagnostics": output["diagnostics"],
    }
    live_ledger_sha = runner.sha_file(runner.LEDGER)
    with pytest.raises(runner.CGeomError, match="cannot use or write"):
        runner.synthetic_unit_lifecycle(tmp_path / "forbidden", runner.LEDGER, "FORBIDDEN", lambda: payload)
    assert runner.sha_file(runner.LEDGER) == live_ledger_sha

    temporary_ledger = tmp_path / "synthetic-budget.jsonl"
    root = tmp_path / "synthetic-run"
    callback_count = {"count": 0}

    def fit_callback():
        callback_count["count"] += 1
        return payload

    first_fit_id = "CGEOM_V1_E67_ALIGNED"
    with pytest.raises(runner.SyntheticInterruption, match="after receipt"):
        runner.synthetic_unit_lifecycle(root, temporary_ledger, first_fit_id, fit_callback,
                                        interrupt_after_receipt=True)
    assert callback_count["count"] == 1
    recovered = runner.synthetic_unit_lifecycle(
        root, temporary_ledger, first_fit_id,
        lambda: pytest.fail("resume must not call the consumed fit callback"),
    )
    assert recovered["status"] == "RECOVERED_DURABLE_RECEIPT"
    assert callback_count["count"] == 1
    saved_resume = runner.synthetic_unit_lifecycle(
        root, temporary_ledger, first_fit_id,
        lambda: pytest.fail("terminal resume must not refit"),
    )
    assert saved_resume["status"] == "RESUMED_SAVED"
    assert callback_count["count"] == 1

    for fit_id in ("CGEOM_V1_E67_PERMUTED", "CGEOM_V1_E75_ALIGNED", "CGEOM_V1_E75_PERMUTED"):
        saved = runner.synthetic_unit_lifecycle(root, temporary_ledger, fit_id, fit_callback)
        assert saved["status"] == "SAVED"
    events = [json.loads(line) for line in temporary_ledger.read_text(encoding="utf-8").splitlines()]
    assert sum(event["event"] == "FIT_STARTED" for event in events) == 4
    assert sum(event["event"] == "FIT_SAVED" for event in events) == 4
    assert not any(event["event"] == "FIT_FAILED" for event in events)
    assert callback_count["count"] == 4
    with np.load(root / "units" / f"{first_fit_id}.npz", allow_pickle=False) as archive:
        assert all(archive[key].dtype.kind != "O" for key in archive.files)
        assert archive["query_record_ids"].dtype.kind == "U"
        assert archive["neighbor_ids"].dtype.kind == "U"
        assert archive["query_record_ids"][0] == fixture["query_ids"][0]
        assert np.array_equal(archive["candidate_probability"], output["candidate_probability"])
    assert runner.sha_file(runner.LEDGER) == live_ledger_sha
