from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.optimization_v3_batch1_verify import verify_report
from src.models.optimization_v3_batch1 import make_freeze_identity, run_pilot
from src.models.optimization_v3_batch1_c01 import (
    C01IntegrityError,
    FEATURE_NAMES,
    FIT_ENDPOINTS,
    fit_predict,
    reconstruct_c01,
)


VIEWS = {
    "W1": [(55, 56, 59)],
    "W2": [(59, 60, 63)],
    "W3": [(63, 64, 67)],
    "W4": [(67, 68, 71)],
    "W5": [(71, 72, 75)],
    "W6": [(75, 76, 79)],
    "primary": [
        (55, 56, 59), (59, 60, 63), (63, 64, 67),
        (67, 68, 71), (71, 72, 75), (75, 76, 79),
    ],
    "H12_E": [(55, 56, 67)],
    "H12_L": [(67, 68, 79)],
    "H12": [(55, 56, 67), (67, 68, 79)],
}


def _synthetic_frame(per_cycle: int = 18) -> pd.DataFrame:
    cycles = np.repeat(np.arange(47, 80, dtype=np.int64), per_cycle)
    rows = np.arange(len(cycles), dtype=np.int64)
    streams = ["" if row % 3 == 0 else f"token{row % 17}" for row in rows]
    return pd.DataFrame({
        "record_id": [f"c01-synthetic-{row}" for row in rows],
        "cycle_code": cycles,
        "signal_stream": streams,
    })


def _controls(frame: pd.DataFrame, rows: np.ndarray) -> dict[str, np.ndarray]:
    rows = np.asarray(rows, dtype=np.int64)
    ids = frame.iloc[rows]["record_id"].astype(str).to_numpy(dtype=str)
    missing = frame.iloc[rows]["signal_stream"].eq("").to_numpy(dtype=bool)
    c1b = 0.15 + 0.7 * ((rows % 11) / 10.0)
    c2b = 0.2 + 0.6 * ((rows % 7) / 6.0)
    cb = 0.1 + 0.8 * ((rows % 5) / 4.0)
    sparse = 0.05 + 0.9 * ((rows % 13) / 12.0)
    baseline = np.where(missing, cb, 0.75 * c2b + 0.25 * sparse)
    return {
        "baseline_prediction": baseline,
        "c1b_probability": c1b,
        "c2b_probability": c2b,
        "cb_missing_probability": cb,
        "sparse_raw_probability": sparse,
        "stream_missing": missing,
        "source_rows": rows,
        "source_record_ids": ids,
    }


def test_c01_recomposes_only_the_missing_route_and_reports_zero_fits() -> None:
    rows = np.array([3, 4, 5, 6], dtype=np.int64)
    frame = _synthetic_frame()
    controls = _controls(frame, rows)

    result = fit_predict(
        None,
        np.array([99, 99, 99, 99]),  # The cache-only adapter must ignore labels.
        None,
        controls=controls,
        card_id="C01",
        cutoff=55,
    )
    expected = np.where(
        controls["stream_missing"], controls["c1b_probability"],
        0.75 * controls["c2b_probability"] + 0.25 * controls["sparse_raw_probability"],
    )
    np.testing.assert_array_equal(result["prediction"], expected)
    assert result["fit_predictor_count"] == 0
    assert result["fit_metadata"]["preprocessing_fits"] == 0
    assert result["state"]["threshold"] == 0.5

    arrays = {
        "baseline_prediction": controls["baseline_prediction"],
        "control__c1b_probability": controls["c1b_probability"],
        "control__c2b_probability": controls["c2b_probability"],
        "control__cb_missing_probability": controls["cb_missing_probability"],
        "control__sparse_raw_probability": controls["sparse_raw_probability"],
        "control__stream_missing": controls["stream_missing"],
        "control__source_rows": rows,
        "control__source_record_ids": controls["source_record_ids"],
        "score_rows": rows,
        "score_record_ids": controls["source_record_ids"],
    }
    rebuilt, baseline = reconstruct_c01(arrays)
    np.testing.assert_array_equal(rebuilt, expected)
    np.testing.assert_array_equal(baseline, controls["baseline_prediction"])


def test_c01_rejects_baseline_drift_bad_probabilities_and_misaligned_ids() -> None:
    frame = _synthetic_frame()
    rows = np.array([2, 3, 4], dtype=np.int64)
    controls = _controls(frame, rows)

    bad_baseline = {**controls, "baseline_prediction": np.zeros(len(rows))}
    with pytest.raises(C01IntegrityError, match="does not reconstruct"):
        fit_predict(None, None, None, controls=bad_baseline, card_id="C01", cutoff=55)

    bad_probability = {**controls, "c1b_probability": np.array([0.2, 1.2, 0.5])}
    with pytest.raises(C01IntegrityError, match="within \\[0,1\\]"):
        fit_predict(None, None, None, controls=bad_probability, card_id="C01", cutoff=55)

    arrays = {
        "baseline_prediction": controls["baseline_prediction"],
        "control__c1b_probability": controls["c1b_probability"],
        "control__c2b_probability": controls["c2b_probability"],
        "control__cb_missing_probability": controls["cb_missing_probability"],
        "control__sparse_raw_probability": controls["sparse_raw_probability"],
        "control__stream_missing": controls["stream_missing"],
        "control__source_rows": rows + 1,
        "control__source_record_ids": controls["source_record_ids"],
        "score_rows": rows,
        "score_record_ids": controls["source_record_ids"],
    }
    with pytest.raises(C01IntegrityError, match="rows or IDs differ"):
        reconstruct_c01(arrays)


def test_c01_zero_fit_cache_six_runs_through_independent_verifier(
    tmp_path: Path,
) -> None:
    frame = _synthetic_frame()
    cycles = frame["cycle_code"].to_numpy(dtype=np.int64)
    target = (np.arange(len(frame), dtype=np.int64) % 2).astype(np.int64)
    controls_by_cutoff: dict[int, dict[str, np.ndarray]] = {}
    for cutoff in FIT_ENDPOINTS:
        score_cycles = sorted({
            cycle
            for segments in VIEWS.values()
            for origin, low, high in segments
            if origin == cutoff
            for cycle in range(low, high + 1)
        })
        score_rows = np.flatnonzero(np.isin(cycles, score_cycles))
        controls_by_cutoff[cutoff] = _controls(frame, score_rows)

    source = tmp_path / "synthetic_c01_source.py"
    train_path = tmp_path / "synthetic_c01_train.csv"
    source.write_text("synthetic C01 source identity\n", encoding="utf-8")
    frame.to_csv(train_path, index=False)
    card = {"id": "C01", "task": "classification", "pilot_predictor_fits": 0}
    identity = make_freeze_identity(
        card=card,
        feature_names=FEATURE_NAMES,
        input_files={"synthetic_train": train_path},
        source_files=[source],
        evaluation_views=VIEWS,
        packages=[],
    )

    run_dir = tmp_path / "runs"
    report = run_pilot(
        card=card,
        artifact_dir=run_dir,
        budget_ledger_path=tmp_path / "ledger.jsonl",
        max_successful_fits=180,
        frame=frame,
        target=target,
        feature_names=FEATURE_NAMES,
        adapter=fit_predict,
        control_loader=lambda cutoff, rows: controls_by_cutoff[cutoff],
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
        artifact_dir=run_dir,
        freeze_signature=report["freeze_signature"],
        evaluation_views=VIEWS,
        prediction_reconstructors={"C01": reconstruct_c01},
    )

    assert report["fit_count_matches_card"] is True
    assert report["fit_predictor_count"] == 0
    assert verification["status"] == "PASS"
    assert verification["views"]["primary"]["n"] == sum(
        int(((cycles >= low) & (cycles <= high)).sum())
        for segments in VIEWS["primary"]
        for _, low, high in [segments]
    )
    for cutoff in FIT_ENDPOINTS:
        bundle = np.load(run_dir / f"C01_fit{cutoff}.npz", allow_pickle=False)
        assert bundle["prediction"].shape == bundle["score_rows"].shape
        assert np.isfinite(bundle["prediction"]).all()
        assert np.all((bundle["prediction"] >= 0.0) & (bundle["prediction"] <= 1.0))
        sidecar = json.loads((run_dir / f"C01_fit{cutoff}.json").read_text(encoding="utf-8"))
        assert sidecar["metadata"]["fit_predictor_count"] == 0
