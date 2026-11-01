from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import optimization_v3_batch1_c12 as runner
from scripts.optimization_v3_batch1_verify import verify_report
from src.models.optimization_v3_batch1 import ensure_frozen, load_queue, make_freeze_identity, run_pilot
from src.models.optimization_v3_batch1_c12 import (
    CHAR_TFIDF_PARAMS,
    FEATURE_NAMES,
)
from src.models.optimization_v3_batch1_c12 import CharOnlySparseClassifier, fit_predict
from src.models.optimization_v3_sparse_classification import PARAMS as SPARSE_PARAMS


def _synthetic_train() -> pd.DataFrame:
    per_cycle = 96
    cycles = np.repeat(np.arange(47, 80, dtype=np.int64), per_cycle)
    rows = np.arange(len(cycles), dtype=np.int64)
    data: dict[str, object] = {
        "record_id": [f"synthetic-c12-{i:05d}" for i in rows],
        "impact_load": (rows % 17).astype(float),
        "entry_friction": (rows % 2).astype(float),
        "cycle_code": cycles,
        "scope_pressure": (rows % 11).astype(float),
        "text_load": (rows % 9).astype(float),
        "scope_trace": (rows % 8).astype(float),
        "org_spread": (rows % 6).astype(float),
        "module_spread": (rows % 7).astype(float),
        "telemetry_flag": (rows % 2).astype(float),
        "org_code": [f"org-{i % 5}" for i in rows],
        "module_code": [f"module-{i % 7}" for i in rows],
        "pattern_code": [f"pattern-{i % 4}" for i in rows],
        "signal_stream": [
            "network protocol vulnerability scanner packet request "
            f"signal{i % 13:02d} window{i % 9:02d} process{i % 5:02d}"
            if i % 97 else ""
            for i in rows
        ],
        "artifact_signal": (((rows % 11) + (cycles % 3)) >= 6).astype(np.int8),
    }
    frame = pd.DataFrame(data)
    assert set(FEATURE_NAMES) == set(frame.columns) - {"record_id", "artifact_signal"}
    return frame


def _controls(score_rows: np.ndarray, missing: np.ndarray) -> dict[str, np.ndarray]:
    rows = np.asarray(score_rows, dtype=np.int64)
    n = len(rows)
    c2b = np.linspace(0.25, 0.75, n, dtype=np.float64)
    cb = np.linspace(0.35, 0.65, n, dtype=np.float64)
    old_sparse = np.clip(0.55 - 0.12 * c2b, 0.0, 1.0)
    baseline = np.where(missing, cb, 0.75 * c2b + 0.25 * old_sparse)
    return {
        "baseline_prediction": baseline,
        "baseline_sparse_raw": old_sparse,
        "c2b_probability": c2b,
        "cb_probability": cb,
        "stream_missing": np.asarray(missing, dtype=bool),
        "control_rows": rows,
    }


def _reconstruct(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(arrays["raw_candidate"], dtype=np.float64)
    missing = np.asarray(arrays["control__stream_missing"], dtype=bool)
    c2b = np.asarray(arrays["control__c2b_probability"], dtype=np.float64)
    cb = np.asarray(arrays["control__cb_probability"], dtype=np.float64)
    old_sparse = np.asarray(arrays["control__baseline_sparse_raw"], dtype=np.float64)
    candidate = np.where(missing, cb, 0.75 * c2b + 0.25 * raw)
    baseline = np.where(missing, cb, 0.75 * c2b + 0.25 * old_sparse)
    return candidate, baseline


def test_c12_uses_character_only_and_inherits_sparse_lightgbm_recipe():
    assert CHAR_TFIDF_PARAMS["analyzer"] == "char_wb"
    assert CHAR_TFIDF_PARAMS["ngram_range"] == (3, 5)
    assert CHAR_TFIDF_PARAMS["min_df"] == 3
    assert CHAR_TFIDF_PARAMS["max_features"] == 30_000
    recipe = runner._recipe_payload(
        next(item for item in load_queue(runner.QUEUE_PATH)["cards"] if item["id"] == "C12"),
        runner.EXPECTED_VIEWS,
    )
    assert recipe["single_recipe_change"]["word_tfidf_removed_from_C11_design"] is True
    assert list(recipe["tfidf_parameters"]) == ["char_wb"]
    assert recipe["classifier_parameters"] == SPARSE_PARAMS
    assert "0.75*frozen C2_B" in recipe["prediction_policy"]["stream_present"]
    assert recipe["prediction_policy"]["threshold"] == 0.50


def test_c12_model_has_no_word_block_and_keeps_a_fixed_sparse_schema():
    frame = _synthetic_train()
    fit_rows = np.flatnonzero(frame["cycle_code"].to_numpy() <= 55)
    fit_x = frame.iloc[fit_rows][list(FEATURE_NAMES)].copy()
    y = frame.iloc[fit_rows]["artifact_signal"].to_numpy(dtype=np.int8)
    model = CharOnlySparseClassifier().fit(fit_x, y)
    score_x = frame.loc[frame["cycle_code"] == 56, list(FEATURE_NAMES)]
    probabilities = model.predict_proba(score_x)
    state = model.audit_state()
    assert probabilities.shape == (len(score_x),)
    assert np.isfinite(probabilities).all()
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert state["word_tfidf_used"] is False
    assert state["text_blocks"] == ["char_wb"]
    assert list(state["vectorizers"]) == ["char_wb"]
    assert state["vectorizers"]["char_wb"]["vocabulary_size"] > 0
    assert state["design_shape"][0] == len(fit_x)


def test_c12_availability_route_uses_cb_for_missing_and_frozen_c2b_for_present():
    frame = _synthetic_train()
    target = frame["artifact_signal"].to_numpy(dtype=np.int8)
    cycles = frame["cycle_code"].to_numpy(dtype=np.int64)
    fit_rows = np.flatnonzero(cycles <= 55)
    score_rows = np.flatnonzero((cycles > 55) & (cycles <= 67))
    score_features = frame.iloc[score_rows][list(FEATURE_NAMES)].copy()
    missing = score_features["signal_stream"].fillna("").astype(str).str.strip().eq("").to_numpy()
    controls = _controls(score_rows, missing)
    fitted = fit_predict(
        frame.iloc[fit_rows][list(FEATURE_NAMES)], target[fit_rows], score_features,
        controls=controls, card_id="C12", cutoff=55,
    )
    np.testing.assert_array_equal(fitted["prediction"][missing], controls["cb_probability"][missing])
    expected_present = 0.75 * controls["c2b_probability"][~missing] + 0.25 * fitted["raw_candidate"][~missing]
    np.testing.assert_allclose(fitted["prediction"][~missing], expected_present, rtol=0, atol=1e-12)
    assert fitted["state"]["threshold"] == 0.50
    assert fitted["state"]["model"]["word_tfidf_used"] is False


def test_c12_shared_temporal_pilot_and_independent_metric_reconstruction(tmp_path: Path):
    frame = _synthetic_train()
    target = frame["artifact_signal"].to_numpy(dtype=np.int8)
    cycles = frame["cycle_code"].to_numpy(dtype=np.int64)
    views = {
        "W1": [(55, 56, 59)],
        "W4": [(67, 68, 71)],
        "primary": [(55, 56, 59), (67, 68, 71)],
        "H12": [(55, 56, 67), (67, 68, 79)],
    }
    source = tmp_path / "c12_model.py"
    source.write_text("# synthetic C12 frozen adapter\n", encoding="utf-8")
    data = tmp_path / "train.csv"
    frame.to_csv(data, index=False)
    identity = make_freeze_identity(
        card={"id": "C12", "task": "classification"}, feature_names=FEATURE_NAMES,
        input_files={"synthetic_train": data}, source_files=[source],
        evaluation_views=views, packages=[],
    )
    freeze_path = tmp_path / "freeze.json"
    signature = ensure_frozen(freeze_path, identity)

    def adapter(fit_x, fit_y, score_x, *, controls, card_id, cutoff):
        assert "record_id" not in fit_x and "artifact_signal" not in fit_x
        assert "record_id" not in score_x and "artifact_signal" not in score_x
        assert fit_x["cycle_code"].max() <= cutoff
        return fit_predict(fit_x, fit_y, score_x, controls=controls,
                           card_id=card_id, cutoff=cutoff)

    def load_controls(cutoff: int, score_rows: np.ndarray) -> dict[str, np.ndarray]:
        score_features = frame.iloc[score_rows][list(FEATURE_NAMES)]
        missing = score_features["signal_stream"].fillna("").astype(str).str.strip().eq("").to_numpy()
        return _controls(score_rows, missing)

    report = run_pilot(
        frame=frame, target=target,
        card={"id": "C12", "task": "classification", "pilot_predictor_fits": 2},
        feature_names=FEATURE_NAMES, adapter=adapter, control_loader=load_controls,
        freeze_identity=identity, freeze_path=freeze_path,
        artifact_dir=tmp_path / "artifacts", event_path=tmp_path / "attempts.jsonl",
        evaluation_views=views, report_path=tmp_path / "pilot.json",
    )
    verification = verify_report(
        report=report, frame=frame, target=target, artifact_dir=tmp_path / "artifacts",
        freeze_signature=signature, evaluation_views=views,
        prediction_reconstructors={"C12": _reconstruct},
    )
    assert report["fit_count_matches_card"] is True
    assert verification["status"] == "PASS"
    assert verification["views"]["primary"]["n"] == 768
    assert verification["views"]["H12"]["n"] == 2304
    for cutoff in (55, 67):
        sidecar = json.loads((tmp_path / "artifacts" / f"C12_fit{cutoff}.json").read_text(encoding="utf-8"))
        state = sidecar["metadata"]["state"]["model"]
        assert state["word_tfidf_used"] is False
        assert state["validation_data_passed_to_fit"] is False
        assert state["validation_labels_passed_to_fit"] is False
        assert state["test_data_used"] is False
        assert sidecar["metadata"]["fit_predictor_count"] == 1


def test_c11_prerequisite_is_a_saved_advance_with_independent_qa():
    result = runner._validate_c11_prerequisite()
    assert result["card_id"] == "C11"
    assert result["gate_status"] == "ADVANCE"
    assert result["verification_status"] == "INDEPENDENT_QA_PASS"
    assert result["fit_predictor_count"] == 2
