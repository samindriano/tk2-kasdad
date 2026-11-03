from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingClassifier

from scripts import optimization_v3_a02_cgeom_benchmark_v1 as runner
from src.models import classification_a02_neighbors as a02
from src.models.a02_coefficient_geometry_v1 import CoefficientGeometryPredictor, _top_positive
from src.models.classification_models import _text_values
from src.models.positional_bag_offset_v1 import fit_offset_coefficients, offset_probability
from src.models.optimization_v3_positional_text_pilot_v1 import (
    build_position_block,
    positive_probability,
)


def _docs() -> list[str]:
    templates = [
        "alpha beta alpha markerone",
        "beta gamma delta gamma",
        "epsilon alpha zeta",
        "delta epsilon beta",
        "gamma zeta alpha beta",
        "markerone delta gamma",
        "epsilon beta zeta alpha",
        "gamma alpha delta epsilon",
    ]
    return [templates[index % len(templates)] for index in range(40)] + [
        "alpha beta gamma", "", "delta epsilon zeta", "beta markerone alpha"
    ]


def test_synthetic_three_candidate_pipeline_end_to_end_and_npz_unicode() -> None:
    docs = _docs()
    train_docs = docs[:40]
    query_docs = docs[40:]
    rows = np.arange(len(train_docs), dtype=np.int64)
    cycles = np.full(len(rows), 54, dtype=np.int64)
    space, train_text, train_missing = a02.fit_text_space(
        train_docs, rows, cycles, origin_cycle=54
    )
    vectorizer = space.vectorizer
    docs_normalized, train_missing_again = a02._documents(train_docs)
    assert np.array_equal(train_missing, train_missing_again)
    text_matrix = vectorizer.transform(docs_normalized).tocsr().astype(np.float64)

    rng = np.random.default_rng(1907)
    structured = sparse.csr_matrix(rng.normal(size=(40, 4)))
    bag_train = sparse.hstack((structured, text_matrix,
                               sparse.csr_matrix(train_missing.astype(float).reshape(-1, 1))),
                              format="csr", dtype=np.float64)
    target = (np.arange(40) % 3 == 0).astype(np.int8)
    bag, _ = runner._fit_logistic(bag_train, target, label="synthetic C2_B")
    position_train, _ = build_position_block(docs_normalized, vectorizer, shuffled=False)
    z0 = np.asarray(bag.decision_function(bag_train), dtype=np.float64).reshape(-1)
    gamma, gamma_meta = fit_offset_coefficients(position_train, z0, target, c=1.0)
    assert gamma_meta["success"] is True
    joint_train = sparse.hstack((bag_train, position_train), format="csr", dtype=np.float64)
    joint, _ = runner._fit_logistic(joint_train, target, label="synthetic JOINTPOS")

    missing_raw = rng.normal(size=(40, 4))
    missing_hgb = HistGradientBoostingClassifier(max_iter=10, max_leaf_nodes=5, random_state=42)
    missing_hgb.fit(missing_raw, target)

    test_docs, test_missing = _text_values(__import__("pandas").Series(query_docs, dtype=object))
    test_text = vectorizer.transform(test_docs).tocsr().astype(np.float64)
    test_struct = sparse.csr_matrix(rng.normal(size=(4, 4)))
    bag_test = sparse.hstack((test_struct, test_text,
                              sparse.csr_matrix(test_missing.astype(float).reshape(-1, 1))),
                             format="csr", dtype=np.float64)
    position_test, _ = build_position_block(test_docs, vectorizer, shuffled=False)
    bag_probability = positive_probability(bag, bag_test)
    offset_raw = offset_probability(position_test,
                                    np.asarray(bag.decision_function(bag_test)).reshape(-1), gamma)
    joint_probability = positive_probability(
        joint, sparse.hstack((bag_test, position_test), format="csr", dtype=np.float64)
    )
    fallback = positive_probability(missing_hgb, rng.normal(size=(4, 4)))
    posroute = np.where(test_missing, fallback, offset_raw)
    jointroute = np.where(test_missing, fallback, joint_probability)
    assert np.array_equal(posroute[test_missing], fallback[test_missing])
    assert np.array_equal(jointroute[test_missing], fallback[test_missing])

    # Build the exact full donor bank and its frozen original top-32 comparator.
    bank_rows = np.arange(36, dtype=np.int64)
    bank_matrix = train_text[:36].tocsr()
    bank_ids = np.asarray([f"synthetic-train-{index}" for index in bank_rows], dtype=str)
    bank_cycles = 47 + (bank_rows % 8)
    bank_prob = positive_probability(bag, bag_train)[:36]
    residuals = target[:36].astype(float) - bank_prob
    beta = np.zeros(len(vectorizer.vocabulary_), dtype=np.float64)
    beta[:] = bag.coef_[0, 4:4 + len(beta)]
    predictor = CoefficientGeometryPredictor(origin_cycle=54, arm="aligned").fit(
        text_space=space, bank_matrix=bank_matrix, bank_ids=bank_ids,
        bank_row_positions=bank_rows, bank_cycles=bank_cycles,
        bank_oof_probabilities=bank_prob, bank_residuals=residuals,
        beta_in_a02_column_order=beta,
    )
    query_matrix, query_missing = space.transform(query_docs)
    original_rows = np.full((4, 32), -1, dtype=np.int64)
    original_sims = np.zeros((4, 32), dtype=np.float64)
    for index in range(4):
        scores = (query_matrix[index] @ bank_matrix.T).toarray().reshape(-1).astype(np.float64)
        chosen = _top_positive(scores, bank_rows, 32)
        if len(chosen):
            original_rows[index, :len(chosen)] = bank_rows[chosen]
            original_sims[index, :len(chosen)] = scores[chosen]
    baseline = np.clip(bag_probability, 0.0, 1.0)
    original_a02 = np.clip(baseline + np.asarray([0.01, 0.0, -0.01, 0.0]), 0.0, 1.0)
    cgeom = predictor.predict(
        query_texts=query_docs,
        query_record_ids=["synthetic-test-λ", "synthetic-test-短文", "synthetic-test-🌱", "synthetic-test-" + "長" * 160],
        query_row_positions=np.arange(40, 44, dtype=np.int64),
        query_cycles=np.full(4, 55, dtype=np.int64),
        baseline_probabilities=baseline,
        original_a02_probabilities=original_a02,
        original_a02_neighbor_source_rows=original_rows,
        original_a02_neighbor_similarities=original_sims,
        missing_mask=query_missing,
    )
    assert np.array_equal(cgeom["candidate_probability"][query_missing], baseline[query_missing])
    for probability in (cgeom["candidate_probability"], posroute, jointroute):
        assert probability.shape == (4,)
        assert np.isfinite(probability).all()
        assert np.all((probability >= 0.0) & (probability <= 1.0))

    with tempfile.TemporaryDirectory(prefix="a02_cgeom_e2e_npz_") as temp:
        path = Path(temp) / "synthetic.npz"
        ids = np.asarray(["synthetic-test-λ", "synthetic-test-短文",
                          "synthetic-test-🌱", "synthetic-test-" + "長" * 160], dtype=str)
        arrays = {"record_ids": ids,
                  "cgeom": cgeom["candidate_probability"].astype(np.float64),
                  "posoffset": posroute.astype(np.float64),
                  "jointpos": jointroute.astype(np.float64),
                  "labels": (jointroute >= 0.5).astype(np.int8)}
        runner._array_npz_once(path, arrays)
        with np.load(path, allow_pickle=False) as restored:
            assert set(restored.files) == set(arrays)
            assert restored["record_ids"].dtype.kind == "U"
            assert np.array_equal(restored["record_ids"], ids)
            assert np.array_equal(restored["jointpos"], arrays["jointpos"])


def test_synthetic_transaction_resumes_durable_fit_without_second_callback() -> None:
    result = runner.synthetic_lifecycle()
    assert result["calls"]["count"] == 1
    assert result["result"]["status"] == "RECOVERED_DURABLE_RECEIPT"
    events = runner.read_events(result["temporary_ledger"])
    assert [event["event"] for event in events].count("FIT_STARTED") == 1
    assert [event["event"] for event in events].count("FIT_SAVED") == 1
    assert not any(event["event"] == "FIT_FAILED" for event in events)
    assert result["unicode_ids_npz_sha256"]


def test_initial_budget_contract_and_recipe_freeze_are_exact() -> None:
    events, snapshot = runner.verify_initial_budget()
    assert snapshot["global_cap"] == 121
    assert snapshot["fit_starts"] == 118
    assert snapshot["fit_saved"] == 114
    assert snapshot["fit_failed"] == 4
    assert snapshot["fit_pending"] == 0
    assert len(events) == 303
    recipe = runner._read_json(runner.RECIPE_PATH)
    assert recipe["fit_budget"]["maximum_new_units"] == 9
    assert recipe["fit_budget"]["global_cap"] == {"before": 121, "after": 130}
    assert recipe["fit_budget"]["continuation_cap"] == {"before": 38, "after": 47}
