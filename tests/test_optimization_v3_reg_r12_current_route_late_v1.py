from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from scripts import optimization_v3_reg_r12_current_route_late_v1 as runner
from scripts import verify_optimization_v3_reg_r12_current_route_late_v1_prelabel as verifier
from scripts.optimization_v3_reg_r12_current_route_late_v1 import (
    GateError,
    _read_targets_for_positions,
    _group_supported,
    synthetic_check,
)
from src.models import optimization_v3_batch1_r12 as historical_r12
from src.models import optimization_v3_reg_r12_current_route_late_v1 as late_r12
from src.models.optimization_v3_reg_r12_current_route_late_v1 import R12IntegrityError, build_routes
from src.models.regression_research_text import CATEGORICAL_FEATURES, HYBRID_FEATURES, RegressionTextModel, STRUCTURED_FEATURES, _text_documents


def test_routes_use_complete_cycle_average_ranks_and_exact_missing_fallback() -> None:
    cycles = np.asarray([72, 72, 72, 72, 73, 73, 73], dtype=np.int64)
    raw = np.asarray([1, 1, 2, 2, 2, 3, 3], dtype=np.float64)
    current = np.asarray([10, 40, 20, 20, 50, 50, 70], dtype=np.float64)
    legacy = np.asarray([11, 21, 31, 41, 51, 61, 71], dtype=np.float64)
    present = np.asarray([True, False, True, True, False, True, True])
    counts = np.asarray([4, 4, 4, 4, 3, 3, 3], dtype=np.int64)

    result = build_routes(
        raw,
        cycles,
        current,
        present,
        legacy,
        expected_cycles=(72, 73),
        expected_cycle_counts=counts,
    )

    expected_rank = np.asarray([37.5, 37.5, 87.5, 87.5, 100 / 3, 250 / 3, 250 / 3])
    expected_current_rank = np.asarray([25, 100, 62.5, 62.5, 50, 50, 100], dtype=np.float64)
    np.testing.assert_array_equal(result["rank_percentile"], expected_rank)
    np.testing.assert_array_equal(result["mapping_only_rank_percentile"], expected_current_rank)

    expected_candidate = np.clip(0.75 * current + 0.25 * expected_rank, 0, 100)
    expected_control = np.clip(0.75 * current + 0.25 * expected_current_rank, 0, 100)
    assert np.array_equal(result["candidate"][~present].view(np.uint64), current[~present].view(np.uint64))
    assert np.array_equal(result["mapping_only_control"][~present].view(np.uint64), current[~present].view(np.uint64))
    np.testing.assert_array_equal(result["candidate"][present], expected_candidate[present])
    np.testing.assert_array_equal(result["mapping_only_control"][present], expected_control[present])
    np.testing.assert_array_equal(result["original_legacy_route"][~present], legacy[~present])


def test_routes_reject_incomplete_cycle_denominators() -> None:
    with pytest.raises(R12IntegrityError):
        build_routes(
            [1.0, 2.0], [72, 73], [10.0, 20.0], [True, True], [11.0, 21.0],
            expected_cycles=(72, 73), expected_cycle_counts=[2, 1],
        )


@pytest.mark.parametrize(
    ("n", "expected"),
    [(99, False), (100, True), (101, True)],
)
def test_supported_group_boundary_is_inclusive_at_100(n: int, expected: bool) -> None:
    assert _group_supported(n) is expected


def test_independent_feature_group_verifier_matches_frozen_prefit_rule() -> None:
    cycles = np.repeat(np.arange(47, 80, dtype=np.int64), 125)
    within_cycle = np.tile(np.arange(125, dtype=np.float64), 33)
    frame = pd.DataFrame({name: np.ones(len(cycles), dtype=np.float64) for name in HYBRID_FEATURES})
    frame["cycle_code"] = cycles
    frame["text_load"] = within_cycle
    frame["entry_friction"] = np.tile(np.arange(125) % 2, 33)
    frame["signal_stream"] = np.where(np.tile(np.arange(125) % 2, 33) == 0, "synthetic", "")
    receipt = runner._group_feature_preflight(frame, cycles)

    verifier._verify_feature_group_preflight(frame, cycles, {"feature_group_preflight": receipt})


def _write_target_csv(path, rows: list[list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["record_id", "cycle_code", "threat_rank"])
        writer.writerows(rows)


def test_target_loader_ignores_invalid_unselected_cells(tmp_path) -> None:
    path = tmp_path / "synthetic_train.csv"
    _write_target_csv(path, [["FUTURE", "72", "NOT_A_TARGET"], ["FIT", "71", "40"]])

    actual = _read_targets_for_positions(
        np.asarray([1], dtype=np.int64),
        expected_ids=np.asarray(["FIT"]),
        expected_cycles=np.asarray([71], dtype=np.int64),
        maximum_cycle=71,
        train_path=path,
    )

    np.testing.assert_array_equal(actual, np.asarray([40.0]))


def test_target_loader_checks_cycle_boundary_before_target_conversion(tmp_path) -> None:
    path = tmp_path / "synthetic_train.csv"
    _write_target_csv(path, [["FUTURE", "72", "NOT_A_TARGET"]])

    with pytest.raises(GateError, match="outside its authorized cycle boundary"):
        _read_targets_for_positions(
            np.asarray([0], dtype=np.int64),
            expected_ids=np.asarray(["FUTURE"]),
            expected_cycles=np.asarray([72], dtype=np.int64),
            maximum_cycle=71,
            train_path=path,
        )


def test_synthetic_entrypoint_exercises_exclusive_persistence_recovery() -> None:
    result = synthetic_check()
    assert result == {
        "status": "SYNTHETIC_PERSISTENCE_RECOVERY_PASS_NO_FITS_NO_TARGETS",
        "arrays": 4,
        "model_artifact_persisted": True,
        "overwrite_refused": True,
        "model_fit_called": False,
        "target_labels_read": False,
    }


def _install_mocked_run_origin(monkeypatch, tmp_path):
    cycles = np.arange(47, 80, dtype=np.int64)
    ids = np.asarray([f"SYN-{cycle}" for cycle in cycles], dtype="<U8")
    columns = {}
    for name in HYBRID_FEATURES:
        if name == "cycle_code":
            columns[name] = cycles
        elif name == "signal_stream":
            columns[name] = np.asarray([f"synthetic text {cycle}" for cycle in cycles], dtype=object)
        else:
            columns[name] = np.ones(len(cycles), dtype=np.float64)
    frame = pd.DataFrame(columns)
    current_r06 = {}
    for origin, score_cycles in ((71, (72, 73, 74, 75)), (75, (76, 77, 78, 79))):
        rows = np.flatnonzero(np.isin(cycles, score_cycles))
        prediction = np.linspace(20.0, 40.0, len(rows), dtype=np.float64)
        current_r06[origin] = {
            "rows": rows,
            "ids": ids[rows],
            "prediction": prediction,
            "legacy_baseline": prediction + 1.0,
        }
    r06 = {**current_r06, "campaign_receipt": {"synthetic": True}}
    group_receipt = {"synthetic": "feature-only support frozen"}
    resource_receipt = {
        "available_ram_gib_observed": 4.5,
        "estimated_peak_job_gib": 2.4,
        "projected_reserve_after_estimated_peak_gib": 2.1,
        "predictor_fits_performed": 0,
    }
    preflight = {
        "origin_support": runner._validate_origin_support(cycles, ids),
        "feature_group_preflight": group_receipt,
        "resources": resource_receipt,
    }
    protocol = {"signature": "SYNTHETIC_PROTOCOL", "identity": {"input_sha256": {}}}
    run_dir = tmp_path / "runs"
    attempt_path = run_dir / "attempts.jsonl"

    def validate_saved_r06(rows, row_ids, full_cycles, _r06):
        by_row = {}
        for origin in (71, 75):
            block = current_r06[origin]
            for i, row in enumerate(block["rows"]):
                by_row[int(row)] = (str(block["ids"][i]), float(block["prediction"][i]),
                                    float(block["legacy_baseline"][i]), origin)
        current = np.asarray([by_row[int(row)][1] for row in rows], dtype=np.float64)
        legacy = np.asarray([by_row[int(row)][2] for row in rows], dtype=np.float64)
        origins = np.asarray([by_row[int(row)][3] for row in rows], dtype=np.int64)
        assert all(by_row[int(row)][0] == str(record_id) for row, record_id in zip(rows, row_ids))
        return current, legacy, origins, full_cycles[rows]

    def fake_fit(fit_x, fit_y, score_x, *, current_r06, legacy_baseline,
                 expected_cycle_counts, cutoff, unit_start, unit_finish):
        preprocessing = ["word_tfidf", "structured_processor", "numeric_scaler", "categorical_ohe"]
        for name in preprocessing:
            unit_id = unit_start(name, {"synthetic": True, "origin": cutoff})
            unit_finish(unit_id, {"status": "COMPLETED", "synthetic": True})
        predictor_id = unit_start("lgbm_ranker_predictor", {"synthetic": True, "origin": cutoff})
        unit_finish(predictor_id, {"status": "COMPLETED", "synthetic": True})
        model_text = f"synthetic-booster-origin-{cutoff}\n".encode("utf-8")
        raw = np.arange(len(score_x), dtype=np.float64)
        score_cycles = score_x["cycle_code"].to_numpy(dtype=np.int64)
        present = np.ones(len(score_x), dtype=bool)
        routes = build_routes(
            raw, score_cycles, current_r06, present, legacy_baseline,
            expected_cycles=runner.SCORE_CYCLES[cutoff],
            expected_cycle_counts=expected_cycle_counts,
        )
        return {
            "raw_candidate": raw,
            "present_mask": present,
            **routes,
            "state": {
                "preprocessing_fit_calls": preprocessing,
                "fitted_model_sha256": hashlib.sha256(model_text).hexdigest(),
            },
            "audit_arrays": {},
            "fit_rows": len(fit_x),
            "score_rows": len(score_x),
            "predictor_fit_count": 1,
            "preprocessing_fit_count": len(preprocessing),
            "fitted_model": model_text,
        }

    monkeypatch.setattr(runner, "RUN_DIR", run_dir)
    monkeypatch.setattr(runner, "ATTEMPT_PATH", attempt_path)
    monkeypatch.setattr(runner, "_load_frozen_protocol", lambda: (protocol, preflight, {"current_commit": "SYNTHETIC"}))
    monkeypatch.setattr(runner, "_load_feature_data", lambda: (frame, ids, cycles, runner.TRAIN_SHA256))
    monkeypatch.setattr(runner, "_validate_r06_outputs", lambda *_: r06)
    monkeypatch.setattr(runner, "_group_feature_preflight", lambda *_: group_receipt)
    monkeypatch.setattr(runner, "_resource_preflight", lambda *_: resource_receipt)
    monkeypatch.setattr(runner, "_validate_saved_r06_for_scores", validate_saved_r06)
    monkeypatch.setattr(runner, "_read_targets_for_positions", lambda positions, **_: np.full(len(positions), 50.0))
    fit_mock = Mock(side_effect=fake_fit)
    monkeypatch.setattr(runner, "fit_predict_current_route", fit_mock)
    monkeypatch.setattr(verifier, "RUN_DIR", run_dir)
    monkeypatch.setattr(verifier, "ATTEMPT_PATH", attempt_path)
    return frame, cycles, fit_mock, run_dir, attempt_path


def test_mocked_entrypoint_lifecycle_enforces_two_origin_budget_and_saved_model(monkeypatch, tmp_path) -> None:
    frame, cycles, fit_mock, run_dir, attempt_path = _install_mocked_run_origin(monkeypatch, tmp_path)

    first = runner.run_origin(71)
    second = runner.run_origin(75)

    assert first["status"] == second["status"] == "ORIGIN_SAVED_NO_QUERY_LABELS"
    assert fit_mock.call_count == 2
    assert Path(first["bundle"]["model_path"]).is_file()
    assert Path(second["bundle"]["model_path"]).is_file()
    events = [json.loads(line) for line in attempt_path.read_text(encoding="utf-8").splitlines()]
    assert sum(event.get("event") == "LEARNED_UNIT_STARTED" for event in events) == 10
    assert len(verifier._verify_attempts(frame, cycles)) == len(events)
    assert verifier._verify_bundle_file_integrity(71)["identity"]["origin"] == 71

    with pytest.raises(GateError, match="retries are prohibited"):
        runner.run_origin(75)
    start_unit, _, _ = runner._unit_logger(75)
    with pytest.raises(GateError, match="cap would be exceeded"):
        start_unit("lgbm_ranker_predictor", {})
    with pytest.raises(GateError, match="cap would be exceeded"):
        start_unit("word_tfidf", {})
    assert fit_mock.call_count == 2

    saved_model = run_dir / "R12_CURRENT_ROUTE_fit75.txt"
    model_bytes = saved_model.read_bytes()
    saved_model.unlink()
    with pytest.raises(verifier.VerificationError, match="missing its model"):
        verifier._verify_bundle_file_integrity(75)
    saved_model.write_bytes(model_bytes + b"corruption")
    with pytest.raises(verifier.VerificationError, match="does not match its hash receipt"):
        verifier._verify_bundle_file_integrity(75)

    # A missing completion record is a fail-closed prelabel result.
    finishes = [event for event in events if event.get("event") == "LEARNED_UNIT_COMPLETED"]
    assert finishes
    altered = [event for event in events if event is not finishes[-1]]
    attempt_path.write_text("\n".join(json.dumps(event) for event in altered) + "\n", encoding="utf-8")
    with pytest.raises(verifier.VerificationError, match="exactly one completion"):
        verifier._verify_attempts(frame, cycles)


def test_mocked_partial_origin_attempt_cannot_be_retried(monkeypatch, tmp_path) -> None:
    _, _, fit_mock, run_dir, attempt_path = _install_mocked_run_origin(monkeypatch, tmp_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    attempt_path.write_text(json.dumps({"event": "ORIGIN_PREFIT_PASS", "origin": 71}) + "\n", encoding="utf-8")
    partial = run_dir / "R12_CURRENT_ROUTE_fit71.npz"
    partial.write_bytes(b"partial-interrupted-output")

    with pytest.raises(GateError, match="retries are prohibited"):
        runner.run_origin(71)

    assert fit_mock.call_count == 0
    assert partial.read_bytes() == b"partial-interrupted-output"


def test_late_preprocessing_adapter_matches_historical_design_with_deterministic_stubs(monkeypatch) -> None:
    """Compare the adapter assembly with R12 using stubs, without learned fits."""
    n = 4
    frame = pd.DataFrame({name: np.arange(n, dtype=np.float64) for name in STRUCTURED_FEATURES})
    for index, name in enumerate(CATEGORICAL_FEATURES):
        frame[name] = [f"CAT-{index}-{row % 2}" for row in range(n)]
    frame["signal_stream"] = ["alpha", "", None, "beta"]
    fit_index = np.arange(100, 100 + n, dtype=np.int64)
    frame.index = fit_index

    vectorizer_kwargs = []

    class FakeVectorizer:
        def __init__(self, **kwargs):
            vectorizer_kwargs.append(kwargs)

        def fit_transform(self, documents):
            return sparse.csr_matrix(np.asarray([[len(text), i] for i, text in enumerate(documents)], dtype=np.float64))

    class FakeSelectedPreprocessor:
        def __init__(self, selected_features):
            self.selected_features = tuple(selected_features)
            self.transformed_feature_names = ["numeric_a", "categorical_a", "numeric_b"]
            self.categorical_positions = [1]
            self.medians = {"synthetic": 0.0}

        def fit_transform(self, selected_frame):
            matrix = np.column_stack((np.arange(len(selected_frame)), np.ones(len(selected_frame)),
                                      np.arange(len(selected_frame)) * 2)).astype(np.float64)
            return matrix, None

    class FakeScaler:
        def fit_transform(self, values):
            self.mean_ = np.zeros(values.shape[1], dtype=np.float64)
            self.scale_ = np.ones(values.shape[1], dtype=np.float64)
            return values.copy()

    class FakeEncoder:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit_transform(self, tokens):
            self.categories_ = [np.asarray([f"SYN-{i}"]) for i in range(tokens.shape[1])]
            matrix = np.zeros((tokens.shape[0], tokens.shape[1]), dtype=np.float64)
            for row in range(tokens.shape[0]):
                matrix[row, row % tokens.shape[1]] = 1.0
            return sparse.csr_matrix(matrix)

    def fake_fit_vectorizer(documents):
        vectorizer = FakeVectorizer(**late_r12._TFIDF_PARAMS)
        return vectorizer, vectorizer.fit_transform(documents), None

    monkeypatch.setattr(RegressionTextModel, "_fit_vectorizer", staticmethod(fake_fit_vectorizer))
    monkeypatch.setattr("src.models.regression_research_text._SelectedPreprocessor", FakeSelectedPreprocessor)
    monkeypatch.setattr("src.models.optimization_v3_reg_r12_current_route_late_v1._SelectedPreprocessor", FakeSelectedPreprocessor)
    monkeypatch.setattr("src.models.regression_research_text.StandardScaler", FakeScaler)
    monkeypatch.setattr("src.models.optimization_v3_reg_r12_current_route_late_v1.StandardScaler", FakeScaler)
    monkeypatch.setattr("src.models.regression_research_text.OneHotEncoder", FakeEncoder)
    monkeypatch.setattr("src.models.optimization_v3_reg_r12_current_route_late_v1.OneHotEncoder", FakeEncoder)
    monkeypatch.setattr("src.models.regression_research_text.TfidfVectorizer", FakeVectorizer)
    monkeypatch.setattr("src.models.optimization_v3_reg_r12_current_route_late_v1.TfidfVectorizer", FakeVectorizer)

    historical = historical_r12._fit_sparse_design(frame)
    historical_design, _, _, _, _, historical_positions, historical_fallback, historical_state, historical_missing = historical
    documents, _ = _text_documents(frame["signal_stream"])
    starts = []
    finishes = []

    def start(name, details):
        starts.append((name, dict(details)))
        return f"synthetic:{name}"

    def finish(unit_id, details):
        finishes.append((unit_id, dict(details)))

    vectorizer, text_matrix, fallback = late_r12._fit_text_with_unit_log(
        documents, unit_start=start, unit_finish=finish
    )
    structured, processor, scaler, encoder, positions, state = late_r12._structured_fit_with_unit_log(
        frame, unit_start=start, unit_finish=finish
    )
    missing = _text_documents(frame["signal_stream"])[1]
    late_design = sparse.hstack((structured, text_matrix,
                                 sparse.csr_matrix(missing.astype(np.float64).reshape(-1, 1))),
                                format="csr", dtype=np.float64)

    np.testing.assert_array_equal(late_design.toarray(), historical_design.toarray())
    np.testing.assert_array_equal(missing, historical_missing)
    assert positions == historical_positions
    assert fallback == historical_fallback is None
    assert state == historical_state
    assert [name for name, _ in starts] == ["word_tfidf", "structured_processor", "numeric_scaler", "categorical_ohe"]
    assert len(finishes) == 4 and all(item[1]["status"] == "COMPLETED" for item in finishes)
    assert len(vectorizer_kwargs) == 2
    assert all(kwargs == late_r12._TFIDF_PARAMS for kwargs in vectorizer_kwargs)
