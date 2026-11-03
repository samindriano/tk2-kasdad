from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
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


def _integration_synthetic_frame() -> tuple[pd.DataFrame, np.ndarray]:
    n = 8
    rows = np.arange(n, dtype=np.float64)
    frame = pd.DataFrame({
        "record_id": [f"integration-synthetic-{i}" for i in range(n)],
        "impact_load": rows + 1.0,
        "entry_friction": rows[::-1] + 0.5,
        "cycle_code": np.arange(47, 55, dtype=np.int64),
        "scope_pressure": (rows % 3) + 0.25,
        "text_load": rows * 1.5 + 2.0,
        "scope_trace": ["1", "2 nodes", "~3", "4", "1 nodes", "2", "~3 nodes", "4"],
        "org_spread": ["1", "2", "3", "4", "1", "2", "3", "4"],
        "module_spread": ["4", "3", "2", "1", "4", "3", "2", "1"],
        "telemetry_flag": ["active", "inactive"] * 4,
        "org_code": ["org-a", "org-b"] * 4,
        "module_code": ["mod-a", "mod-b", "mod-c", "mod-d"] * 2,
        "pattern_code": ["pat-a", "pat-b", "pat-c", "pat-d"] * 2,
        "signal_stream": ["alpha beta", "beta gamma", "alpha delta", "gamma delta",
                          "", "beta delta", "alpha gamma", "alpha beta delta"],
    })
    target = np.asarray([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.int8)
    return frame, target


def test_synthetic_actual_integration_payloads_persist_and_resume_without_live_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame, target = _integration_synthetic_frame()
    docs = frame.signal_stream.astype(str).tolist()
    rows = np.arange(len(frame), dtype=np.int64)
    cycles = frame.cycle_code.to_numpy(dtype=np.int64)
    space, bank_matrix, missing = a02.fit_text_space(docs, rows, cycles, origin_cycle=79)
    synthetic_model = SimpleNamespace(
        text_space=space,
        bank_matrix=bank_matrix,
        bank_ids=frame.record_id.to_numpy(dtype=str),
        bank_row_positions=rows,
        bank_cycles=cycles,
        bank_oof_probabilities=np.linspace(0.2, 0.8, len(frame), dtype=np.float64),
        bank_residuals=target.astype(np.float64) - np.linspace(0.2, 0.8, len(frame), dtype=np.float64),
    )
    c2b_state = runner.positional.fit_c2b_structured_state(frame)
    c2b_scaler = runner.positional.fit_c2b_scaler(c2b_state, frame)
    c2b_encoder = runner.positional.fit_c2b_ohe(c2b_state, frame)
    reused = {
        runner.FIT_IDS[0]: c2b_state,
        runner.FIT_IDS[1]: c2b_scaler,
        runner.FIT_IDS[2]: c2b_encoder,
    }
    payloads: dict[str, object] = {}
    callbacks = {fit_id: 0 for fit_id in runner.INTEGRATION_FIT_IDS}
    temp_a02 = tmp_path / "synthetic_a02_model.bin"
    temp_a02.write_bytes(b"synthetic-only-a02-boundary")
    monkeypatch.setattr(runner, "A02_MODEL", temp_a02)
    monkeypatch.setattr(runner, "_c2b_vectorizer", lambda: (space.vectorizer, synthetic_model))

    def load_unit(fit_id: str):
        if fit_id in reused:
            return reused[fit_id]
        if fit_id in payloads:
            return payloads[fit_id]
        raise AssertionError(f"Unexpected synthetic state request: {fit_id}")

    def load_active_c2b_lr():
        return payloads[runner.INTEGRATION_ACTIVE_ID_BY_SLOT[3]]

    monkeypatch.setattr(runner, "_load_unit", load_unit)
    monkeypatch.setattr(runner, "_load_active_c2b_lr", load_active_c2b_lr)
    ledger = tmp_path / "synthetic_budget.jsonl"
    artifact_root = tmp_path / "synthetic_units"
    runner._append_event({
        "event": "CARD_FITS_RESERVED", "reservation_key": runner.INTEGRATION_RESERVATION_KEY,
        "card_id": runner.CARD_ID, "stage": runner.INTEGRATION_STAGE,
        "category": "continuation", "fits": 6, "fit_ids": list(runner.INTEGRATION_FIT_IDS),
        "synthetic": True,
    }, ledger_path=ledger, synthetic=True)

    for index, fit_id in enumerate(runner.INTEGRATION_FIT_IDS):
        slot = runner.INTEGRATION_FIT_SLOT_BY_ID[fit_id]

        def callback(fit_id: str = fit_id) -> object:
            callbacks[fit_id] += 1
            fit_target = None if slot in {4, 8} else target
            payload = runner._make_fit_payload(fit_id, frame, fit_target)
            payloads[fit_id] = payload
            return payload

        if index == 0:
            with pytest.raises(RuntimeError, match="synthetic interruption"):
                runner.execute_fit_unit(
                    fit_id, runner.UNIT_COMPONENTS[slot], callback, synthetic=True,
                    ledger_path=ledger, interrupt_after_receipt=True, artifact_root=artifact_root,
                )
            recovered = runner.execute_fit_unit(
                fit_id, runner.UNIT_COMPONENTS[slot], callback, synthetic=True,
                ledger_path=ledger, artifact_root=artifact_root,
            )
            assert recovered["status"] == "RECOVERED_DURABLE_RECEIPT"
        else:
            saved = runner.execute_fit_unit(
                fit_id, runner.UNIT_COMPONENTS[slot], callback, synthetic=True,
                ledger_path=ledger, artifact_root=artifact_root,
            )
            assert saved["status"] == "FIT_SAVED"
    assert callbacks == {fit_id: 1 for fit_id in runner.INTEGRATION_FIT_IDS}
    events = runner.read_events(ledger)
    assert sum(event["event"] == "FIT_STARTED" for event in events) == 6
    assert sum(event["event"] == "FIT_SAVED" for event in events) == 6
    assert not any(event["event"] == "FIT_FAILED" for event in events)
    for fit_id in runner.INTEGRATION_FIT_IDS:
        receipt = runner._verify_receipt(fit_id, artifact_root=artifact_root)
        assert receipt["synthetic"] is True
        assert receipt["fit_population_rows"] == 8
        assert receipt["test_data_used_for_fit"] is False


def test_production_prediction_assembly_and_csv_serialization_are_synthetic(tmp_path: Path) -> None:
    baseline = np.asarray([0.2, 0.4, 0.7, 0.6], dtype=np.float64)
    missing = np.asarray([True, False, False, True])
    fallback = np.asarray([0.8, 0.8, 0.8, 0.65], dtype=np.float64)
    offset = np.asarray([0.1, 0.3, 0.7, 0.4], dtype=np.float64)
    joint = np.asarray([0.25, 0.5, 0.8, 0.2], dtype=np.float64)
    correction = np.asarray([0.0, 0.05, 0.0, 0.0], dtype=np.float64)
    cgeom = np.clip(baseline + correction, 0.0, 1.0)
    assembled = runner.assemble_candidate_predictions(
        baseline=baseline, missing=missing, fallback=fallback,
        offset_direct=offset, offset_formula=offset.copy(),
        joint_direct=joint, joint_formula=joint.copy(),
        cgeom_probability=cgeom, cgeom_correction=correction,
        cgeom_neighbor_count=np.asarray([2, 3, 0, 1]),
    )
    assert assembled["formula_audit"]["status"] == "PASS"
    assert assembled["probabilities"]["POS_BAG_OFFSET"][0] == fallback[0]
    assert assembled["probabilities"]["JOINTPOS"][3] == fallback[3]
    for candidate, probabilities in assembled["probabilities"].items():
        assert np.array_equal(assembled["labels"][candidate], (probabilities >= runner.THRESHOLD).astype(np.int8))
    ids = np.asarray(["synthetic-λ", "synthetic-短", "synthetic-🌱", "synthetic-" + "長" * 150])
    frames = runner.build_submission_frames(ids, assembled["labels"], columns=["record_id", "artifact_signal"])
    for candidate, frame in frames.items():
        path = tmp_path / f"{candidate}.csv"
        runner.write_once(path, frame.to_csv(index=False, lineterminator="\n").encode("utf-8"))
        reread = pd.read_csv(path, dtype={"record_id": str})
        assert reread.columns.tolist() == ["record_id", "artifact_signal"]
        assert np.array_equal(reread.record_id.to_numpy(dtype=str), ids)
        assert reread.artifact_signal.dtype.kind in "iu"
        assert np.isin(reread.artifact_signal.to_numpy(), [0, 1]).all()


def test_initial_budget_contract_and_recipe_freeze_are_exact() -> None:
    assert runner.sha256_file(runner.LEDGER) == runner.INTEGRATION_BASE_LEDGER_SHA
    events, checkpoint = runner.verify_integration_checkpoint()
    assert len(events) == runner.INTEGRATION_BASE_LEDGER_ROWS
    assert checkpoint["active_reservations"] == 123
    assert checkpoint["fit_starts"] == 123
    assert checkpoint["fit_saved"] == 117
    assert checkpoint["fit_failed"] == 6
    assert checkpoint["fit_pending"] == 0
    recipe = runner._read_json(runner.RECIPE_PATH)
    assert recipe["fit_budget"]["maximum_new_units"] == 9
    assert recipe["fit_budget"]["global_cap"] == {"before": 121, "after": 130}
    assert recipe["fit_budget"]["continuation_cap"] == {"before": 38, "after": 47}


def test_a02_row_position_hash_uses_declared_raw_little_endian_algorithm() -> None:
    positions = np.arange(50_608, dtype="<i8")
    expected = "b3412a08e9f0877cc37ef9615479de0d182a21b9806068c2d576ed06ae495bb2"
    digest = runner._a02_row_positions_sha256(positions)
    assert digest == expected
    assert digest == a02._array_sha256(positions, "<i8")
    assert digest != runner.positional.sha256_array(positions)

    altered = positions.copy()
    altered[-1] += 1
    assert runner._a02_row_positions_sha256(altered) != digest
    with pytest.raises(ValueError):
        runner._a02_row_positions_sha256(positions[::-1])


def test_independent_prediction_formula_and_availability_route_audit() -> None:
    baseline = np.asarray([0.2, 0.4, 0.6])
    missing = np.asarray([True, False, False])
    fallback = np.asarray([0.8, 0.8, 0.8])
    offset = np.asarray([0.1, 0.3, 0.5])
    joint = np.asarray([0.2, 0.25, 0.6])
    correction = np.asarray([0.0, -0.1, 0.2])
    cgeom = np.clip(baseline + correction, 0.0, 1.0)
    audit = runner._audit_prediction_formulas(
        baseline=baseline, missing=missing, fallback=fallback,
        offset_direct=offset, offset_formula=offset.copy(), offset_routed=np.asarray([0.8, 0.3, 0.5]),
        joint_direct=joint, joint_formula=joint.copy(), joint_routed=np.asarray([0.8, 0.25, 0.6]),
        cgeom_probability=cgeom, cgeom_correction=correction,
        cgeom_neighbor_count=np.asarray([0, 2, 1]),
    )
    assert audit["status"] == "PASS"
    assert audit["cgeom_fallback_rows"] == 1

    bad_route = runner._audit_prediction_formulas(
        baseline=baseline, missing=missing, fallback=fallback,
        offset_direct=offset, offset_formula=offset.copy(), offset_routed=offset,
        joint_direct=joint, joint_formula=joint.copy(), joint_routed=np.asarray([0.8, 0.25, 0.6]),
        cgeom_probability=cgeom, cgeom_correction=correction,
        cgeom_neighbor_count=np.asarray([0, 2, 1]),
    )
    assert bad_route["status"] == "FAIL"


def test_integration_budget_adds_exactly_six_new_ids_without_reusing_released_slots() -> None:
    events = runner.read_events(runner.LEDGER)
    old_to_new = {
        runner.FAILED_ORIGINAL_LR_ID: runner.INTEGRATION_FIT_IDS[0],
        runner.HASHFIX_REPLACEMENT_ID: runner.INTEGRATION_FIT_IDS[0],
        **{runner.FIT_IDS[slot]: runner.INTEGRATION_FIT_IDS[slot - 3] for slot in range(4, 9)},
    }
    events.extend([
        {
            "event": "BUDGET_CAP_AMENDMENT",
            "amendment_id": runner.INTEGRATION_AMENDMENT_ID,
            "prior_global_cap": runner.INTEGRATION_BASE_CAP,
            "global_cap": runner.INTEGRATION_AMENDED_CAP,
            "prior_effective_caps": runner.INTEGRATION_BASE_CATEGORY_CAPS,
            "effective_caps": runner.INTEGRATION_AMENDED_CATEGORY_CAPS,
            "pre_amendment_ledger_sha256": runner.INTEGRATION_BASE_LEDGER_SHA,
            "pre_amendment_event_rows": runner.INTEGRATION_BASE_LEDGER_ROWS,
            "maximum_replacement_fit_units": runner.INTEGRATION_MAX_FITS,
            "maximum_total_starts": runner.MAX_BATCH_STARTS,
            "batch_consumed_units_maximum": runner.MAX_BATCH_CONSUMED_UNITS,
            "failed_fit_ids_preserved_and_consumed": sorted(runner.INTEGRATION_FAILED_FIT_IDS),
            "old_to_new_fit_ids": old_to_new,
            "active_fit_ids_by_component": runner.INTEGRATION_ACTIVE_ID_BY_COMPONENT,
            "old_unstarted_fit_ids_remain_released": True,
            "additional_retry_authorized": False,
            "mapping_sha256": runner.sha256_file(runner.ROOT / runner.INTEGRATION_MAPPING_PATH),
        },
        {
            "event": "CARD_FITS_RESERVED",
            "reservation_key": runner.INTEGRATION_RESERVATION_KEY,
            "card_id": runner.CARD_ID,
            "stage": runner.INTEGRATION_STAGE,
            "category": "continuation",
            "fits": 6,
            "fit_ids": list(runner.INTEGRATION_FIT_IDS),
            "global_cap": runner.INTEGRATION_AMENDED_CAP,
            "effective_category_cap": runner.INTEGRATION_AMENDED_CATEGORY_CAPS["continuation"],
            "amendment_id": runner.INTEGRATION_AMENDMENT_ID,
        },
    ])
    status = runner._batch_budget_status(events)
    assert status["global_cap"] == 132
    assert status["active_reservations"] == 129
    assert status["active_reservations_by_category"]["continuation"] == 49
    assert status["fit_starts"] == 123
    assert status["fit_saved"] == 117
    assert status["fit_failed"] == 6
    assert status["integration_started"] == 0
    assert status["integration_unstarted_ids"] == sorted(runner.INTEGRATION_FIT_IDS)

    early_failure = [dict(event) for event in events]
    first_id = runner.INTEGRATION_FIT_IDS[0]
    early_failure.extend([
        {"event": "FIT_STARTED", "reservation_key": runner.INTEGRATION_RESERVATION_KEY, "fit_id": first_id},
        {"event": "FIT_FAILED", "reservation_key": runner.INTEGRATION_RESERVATION_KEY, "fit_id": first_id},
        {"event": "RESERVATION_UNUSED_RELEASED", "reservation_key": runner.INTEGRATION_RESERVATION_KEY,
         "unused_fits": 5, "unstarted_fit_ids": list(runner.INTEGRATION_FIT_IDS[1:])},
    ])
    failed_status = runner._batch_budget_status(early_failure)
    assert failed_status["fit_starts"] == 124
    assert failed_status["fit_saved"] == 117
    assert failed_status["fit_failed"] == 7
    assert failed_status["active_reservations"] == 124
    assert failed_status["active_reservations_by_category"]["continuation"] == 44
    assert failed_status["integration_released_ids"] == sorted(runner.INTEGRATION_FIT_IDS[1:])

    sixth_failure = [dict(event) for event in events]
    for fit_id in runner.INTEGRATION_FIT_IDS[:-1]:
        sixth_failure.extend([
            {"event": "FIT_STARTED", "reservation_key": runner.INTEGRATION_RESERVATION_KEY, "fit_id": fit_id},
            {"event": "FIT_SAVED", "reservation_key": runner.INTEGRATION_RESERVATION_KEY, "fit_id": fit_id},
        ])
    sixth_failure.extend([
        {"event": "FIT_STARTED", "reservation_key": runner.INTEGRATION_RESERVATION_KEY,
         "fit_id": runner.INTEGRATION_FIT_IDS[-1]},
        {"event": "FIT_FAILED", "reservation_key": runner.INTEGRATION_RESERVATION_KEY,
         "fit_id": runner.INTEGRATION_FIT_IDS[-1]},
    ])
    sixth_status = runner._batch_budget_status(sixth_failure)
    assert sixth_status["fit_starts"] == 129
    assert sixth_status["fit_saved"] == 122
    assert sixth_status["fit_failed"] == 7
    assert sixth_status["fit_pending"] == 0
    assert sixth_status["integration_unstarted_ids"] == []


def test_integration_checkpoint_verifies_mapping_and_saved_artifacts_without_mutation() -> None:
    if runner.sha256_file(runner.LEDGER) != runner.INTEGRATION_BASE_LEDGER_SHA:
        pytest.skip("Exact Sol integration checkpoint was already advanced by an authorized batch run")
    before = runner.sha256_file(runner.LEDGER)
    events, snapshot = runner.verify_integration_checkpoint()
    assert len(events) == runner.INTEGRATION_BASE_LEDGER_ROWS
    assert snapshot["fit_starts"] == runner.INTEGRATION_BASE_STARTS
    assert snapshot["fit_saved"] == runner.INTEGRATION_BASE_SAVED
    assert snapshot["fit_failed"] == runner.INTEGRATION_BASE_FAILED
    assert set(snapshot["prior_saved_units"]) == set(runner.FIT_IDS[:3])
    assert runner.sha256_file(runner.LEDGER) == before


def test_preparation_head_is_authorized_single_parent_descendant() -> None:
    assert runner.verify_preparation_head() == runner._git("rev-parse", "HEAD")
