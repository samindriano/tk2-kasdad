from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from scripts.optimization_v3_batch1_verify import verify_report
from scripts.optimization_v3_batch1_c10 import _attempt_summary, _is_no_supported_terms_error
from src.models.classification_round2_models import stream_missing_mask
from src.models.optimization_v3_batch1 import make_freeze_identity, run_pilot
from src.models.optimization_v3_batch1_c10 import (
    C10IntegrityError,
    FEATURE_NAMES,
    SeparateTrigramSparseClassifier,
    fit_predict,
    reconstruct_c10,
    trigram_vectorizer,
    word_vectorizer,
)
from tests.test_campaign_classification import _fixture


VIEWS = {
    "W1": [(55, 56, 59)],
    "W4": [(67, 68, 71)],
    "primary": [(55, 56, 59), (67, 68, 71)],
    "H12_E": [(55, 56, 67)],
    "H12_L": [(67, 68, 79)],
    "H12": [(55, 56, 67), (67, 68, 79)],
}


def _temporal_synthetic_frame() -> tuple[pd.DataFrame, np.ndarray]:
    frame, target = _fixture(264)
    frame["record_id"] = [f"c10-row-{index:04d}" for index in range(len(frame))]
    frame["cycle_code"] = np.repeat(np.arange(47, 80, dtype=np.int64), 8)
    return frame, target


def _controls(frame: pd.DataFrame, rows: np.ndarray) -> dict[str, np.ndarray]:
    rows = np.asarray(rows, dtype=np.int64)
    missing = stream_missing_mask(frame.iloc[rows])
    c2b = 0.20 + (rows % 11) * 0.01
    cb = 0.30 + (rows % 9) * 0.01
    sparse = 0.40 + (rows % 7) * 0.01
    baseline = np.where(missing, cb, 0.75 * c2b + 0.25 * sparse)
    return {
        "baseline_prediction": baseline.astype(np.float64),
        "c2b_probability": c2b.astype(np.float64),
        "cb_missing_probability": cb.astype(np.float64),
        "sparse_raw_probability": sparse.astype(np.float64),
        "stream_missing": missing.astype(bool),
        "source_rows": rows,
        "source_record_ids": frame.iloc[rows]["record_id"].astype(str).to_numpy(dtype=str),
    }


def test_c10_uses_two_separate_fit_only_word_blocks_and_keeps_state_fixed() -> None:
    frame, target = _fixture(180)
    fields = list(FEATURE_NAMES)
    model = SeparateTrigramSparseClassifier()
    with threadpool_limits(limits=2):
        model.fit(frame.iloc[:150][fields], target[:150])
        state = model.audit_state()
        held = frame.iloc[150:][fields].copy()
        held["signal_stream"] = "newvalidationonly tripletoken sequence"
        probability = model.predict_proba(held)

    assert state == model.audit_state()
    assert state["word_tfidf"]["parameters"]["ngram_range"] == [1, 2]
    assert state["word_tfidf"]["vocabulary_size"] <= 20_000
    assert state["trigram_tfidf"]["parameters"]["ngram_range"] == [3, 3]
    assert state["trigram_tfidf"]["vocabulary_size"] <= 10_000
    assert state["design"]["feature_block_order"] == [
        "structured", "word_1_2", "word_3", "stream_missing"
    ]
    assert state["eval_set_used"] is False
    assert state["early_stopping_used"] is False
    assert len(probability) == 30
    assert np.isfinite(probability).all()
    assert np.all((probability >= 0.0) & (probability <= 1.0))


def test_c10_route_preserves_exact_frozen_sparse25_baseline_and_missing_branch() -> None:
    frame, _ = _fixture(40)
    rows = np.arange(40, dtype=np.int64)
    controls = _controls(frame, rows)
    raw = np.linspace(0.1, 0.9, len(rows))
    route = {
        "baseline_prediction": controls["baseline_prediction"],
        "control__c2b_probability": controls["c2b_probability"],
        "control__cb_missing_probability": controls["cb_missing_probability"],
        "control__sparse_raw_probability": controls["sparse_raw_probability"],
        "control__stream_missing": controls["stream_missing"],
        "control__source_rows": controls["source_rows"],
        "control__source_record_ids": controls["source_record_ids"],
        "score_rows": controls["source_rows"],
        "score_record_ids": controls["source_record_ids"],
        "raw_candidate": raw,
    }
    candidate, baseline = reconstruct_c10(route)
    np.testing.assert_array_equal(baseline, controls["baseline_prediction"])
    np.testing.assert_array_equal(candidate[controls["stream_missing"]],
                                  controls["cb_missing_probability"][controls["stream_missing"]])
    np.testing.assert_array_equal(candidate[~controls["stream_missing"]],
                                  0.75 * controls["c2b_probability"][~controls["stream_missing"]]
                                  + 0.25 * raw[~controls["stream_missing"]])

    broken = {**route, "control__source_record_ids": controls["source_record_ids"][::-1]}
    with pytest.raises(C10IntegrityError, match="rows or IDs differ"):
        reconstruct_c10(broken)


def test_c10_actual_adapter_reaches_generic_report_and_independent_reconstruction(tmp_path: Path) -> None:
    frame, target = _temporal_synthetic_frame()
    controls_by_cutoff: dict[int, dict[str, np.ndarray]] = {}
    cycles = frame["cycle_code"].to_numpy(dtype=np.int64)
    for cutoff in (55, 67):
        rows = np.flatnonzero((cycles > cutoff) & (cycles <= cutoff + 12))
        controls_by_cutoff[cutoff] = _controls(frame, rows)

    train_path = tmp_path / "synthetic_c10_train.csv"
    source_path = tmp_path / "synthetic_c10_source.py"
    frame.to_csv(train_path, index=False)
    source_path.write_text("synthetic C10 source identity\n", encoding="utf-8")
    card = {"id": "C10", "task": "classification", "pilot_predictor_fits": 2}
    identity = make_freeze_identity(
        card=card,
        feature_names=FEATURE_NAMES,
        input_files={"synthetic_train": train_path},
        source_files=[source_path],
        evaluation_views=VIEWS,
        packages=[],
    )

    with threadpool_limits(limits=2):
        report = run_pilot(
            card=card,
            artifact_dir=tmp_path / "runs",
            budget_ledger_path=tmp_path / "ledger.jsonl",
            max_successful_fits=180,
            frame=frame,
            target=target,
            feature_names=FEATURE_NAMES,
            adapter=fit_predict,
            control_loader=lambda cutoff, rows: controls_by_cutoff[int(cutoff)],
            freeze_identity=identity,
            freeze_path=tmp_path / "freeze.json",
            event_path=tmp_path / "attempts.jsonl",
            evaluation_views=VIEWS,
            threads=2,
            report_path=tmp_path / "engine_report.json",
        )
    verification = verify_report(
        report=report,
        frame=frame,
        target=target,
        artifact_dir=tmp_path / "runs",
        freeze_signature=report["freeze_signature"],
        evaluation_views=VIEWS,
        prediction_reconstructors={"C10": reconstruct_c10},
    )

    assert report["fit_predictor_count"] == 2
    assert report["fit_count_matches_card"] is True
    assert report["views"]["primary"]["rows"] == 64
    assert report["views"]["H12"]["rows"] == 192
    assert verification["status"] == "PASS"
    assert verification["views"]["primary"]["n"] == 64
    for cutoff in (55, 67):
        bundle = np.load(tmp_path / "runs" / f"C10_fit{cutoff}.npz", allow_pickle=False)
        assert bundle["prediction"].shape == bundle["score_rows"].shape
        assert np.isfinite(bundle["prediction"]).all()
        assert np.all((bundle["prediction"] >= 0.0) & (bundle["prediction"] <= 1.0))
        metadata = json.loads((tmp_path / "runs" / f"C10_fit{cutoff}.json").read_text())
        assert metadata["metadata"]["fit_predictor_count"] == 1


def test_c10_vectorizer_factories_keep_word_and_trigram_caps_separate() -> None:
    word = word_vectorizer()
    trigram = trigram_vectorizer()
    assert word.ngram_range == (1, 2)
    assert word.max_features == 20_000
    assert trigram.ngram_range == (3, 3)
    assert trigram.min_df == 3
    assert trigram.max_df == 0.995
    assert trigram.max_features == 10_000


def test_c10_empty_trigram_pruning_and_retry_accounting_are_explicit() -> None:
    assert _is_no_supported_terms_error(ValueError("empty vocabulary; perhaps the documents only contain stop words"))
    assert _is_no_supported_terms_error(ValueError("After pruning, no terms remain. Try a lower min_df."))
    assert not _is_no_supported_terms_error(ValueError("unexpected vectorizer failure"))
    summary = _attempt_summary([
        {"event": "FIT_STARTED"},
        {"event": "FIT_FAILED", "fit_seconds": 2.5},
        {"event": "FIT_STARTED"},
        {"event": "FIT_SAVED", "fit_seconds": 7.5},
        {"event": "RESUMED_VERIFIED"},
    ])
    assert summary == {
        "attempts_started": 2,
        "successful_fit_events": 1,
        "failed_predictor_attempts": 1,
        "reused_verified_bundles": 1,
        "recorded_attempt_seconds": 10.0,
    }
