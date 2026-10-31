from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts import optimization_v3_batch1_c18 as c18_runner
from src.models.optimization_v3_batch1 import make_freeze_identity, run_pilot
from src.models.optimization_v3_batch1_c18 import (
    FEATURE_NAMES,
    PARAMS,
    RandomThresholdSparseTextClassifier,
    fit_predict,
)


def _synthetic_features(n_rows: int) -> pd.DataFrame:
    row = np.arange(n_rows)
    colors = ("red", "blue", "green", "yellow")
    tokens = ("amber", "birch", "cedar", "dahlia", "elm", "fir")
    frame = pd.DataFrame({
        "impact_load": (row % 17).astype(float),
        "entry_friction": (row % 2).astype(int),
        "cycle_code": (47 + row % 9).astype(int),
        "scope_pressure": (row % 11).astype(float),
        "text_load": (row % 9).astype(float),
        "scope_trace": [str(value % 5) for value in row],
        "org_spread": (row % 4).astype(int),
        "module_spread": (row % 6).astype(int),
        "telemetry_flag": (row % 2).astype(int),
        "org_code": [f"org-{value % 7}" for value in row],
        "module_code": [f"module-{value % 8}" for value in row],
        "pattern_code": [f"pattern-{value % 3}" for value in row],
        "signal_stream": [
            None if value % 13 == 0 else
            f"alpha {colors[value % len(colors)]} gamma {tokens[value % len(tokens)]} omega"
            for value in row
        ],
    })
    return frame.loc[:, list(FEATURE_NAMES)]


def test_c18_parameters_change_only_random_threshold_controls() -> None:
    assert PARAMS["extra_trees"] is True
    assert PARAMS["extra_seed"] == 42
    assert PARAMS["random_state"] == 42
    assert PARAMS["n_jobs"] == 2
    assert "feature_fraction" not in PARAMS
    assert "bagging_fraction" not in PARAMS


def test_c18_fit_only_model_and_fixed_availability_route() -> None:
    x = _synthetic_features(128)
    y = (np.arange(len(x)) % 2).astype(np.int8)
    train_x, score_x = x.iloc[:96], x.iloc[96:]
    train_y = y[:96]

    model = RandomThresholdSparseTextClassifier().fit(train_x, train_y)
    state = model.audit_state()
    raw = model.predict_probability(score_x)
    missing = score_x["signal_stream"].isna().to_numpy()
    n_score = len(score_x)
    c2b = np.full(n_score, 0.4)
    cb = np.full(n_score, 0.7)
    old_sparse = np.full(n_score, 0.2)
    baseline = np.where(missing, cb, 0.75 * c2b + 0.25 * old_sparse)
    controls = {
        "baseline_prediction": baseline,
        "baseline_sparse_raw": old_sparse,
        "c2b_probability": c2b,
        "cb_probability": cb,
        "stream_missing": missing,
    }

    result = fit_predict(train_x, train_y, score_x, controls=controls, card_id="C18", cutoff=55)

    assert state["validation_data_passed_to_fit"] is False
    assert state["sample_weight_used"] is False
    assert state["lightgbm_parameters"] == PARAMS
    assert state["fit_rows"] == len(train_x)
    assert result["state"]["cutoff"] == 55
    assert result["state"]["fit_cycle_max"] <= 55
    assert np.isfinite(raw).all()
    assert np.isfinite(result["prediction"]).all()
    assert np.all((result["prediction"] >= 0) & (result["prediction"] <= 1))
    np.testing.assert_array_equal(result["prediction"][missing], cb[missing])
    np.testing.assert_allclose(
        result["prediction"][~missing],
        0.75 * c2b[~missing] + 0.25 * result["raw_candidate"][~missing],
        rtol=0,
        atol=0,
    )


def test_synthetic_pilot_verification_gate_and_publication(tmp_path, monkeypatch) -> None:
    cycles = np.repeat(np.arange(47, 80, dtype=np.int64), 4)
    target = np.tile(np.array([0, 1, 0, 1], dtype=np.int8), 33)
    frame = pd.DataFrame({
        "record_id": [f"synthetic-{index}" for index in range(len(cycles))],
        "cycle_code": cycles,
        "x": target.astype(np.float64),
        "signal_stream": [None if index % 5 == 0 else "alpha beta" for index in range(len(cycles))],
    })
    views = {
        "W1": [(55, 56, 59)],
        "W4": [(67, 68, 71)],
        "primary": [(55, 56, 59), (67, 68, 71)],
        "H12": [(55, 56, 67), (67, 68, 79)],
    }
    controls_by_cutoff = {}
    for cutoff, low, high in ((55, 56, 67), (67, 68, 79)):
        rows = np.flatnonzero((cycles >= low) & (cycles <= high))
        missing = frame.iloc[rows]["signal_stream"].isna().to_numpy()
        c2b = np.full(len(rows), 0.5)
        cb = np.full(len(rows), 0.5)
        old_sparse = np.full(len(rows), 0.5)
        baseline = np.where(missing, cb, 0.75 * c2b + 0.25 * old_sparse)
        controls_by_cutoff[cutoff] = {
            "baseline_prediction": baseline,
            "baseline_sparse_raw": old_sparse,
            "c2b_probability": c2b,
            "cb_probability": cb,
            "stream_missing": missing,
        }

    fit_calls = []

    def synthetic_adapter(fit_x, fit_y, score_x, *, controls, card_id, cutoff):
        fit_calls.append(cutoff)
        raw = np.where(score_x["x"].to_numpy(dtype=float) >= 0.5, 0.9, 0.1)
        missing = controls["stream_missing"]
        prediction = np.where(
            missing,
            controls["cb_probability"],
            0.75 * controls["c2b_probability"] + 0.25 * raw,
        )
        return {"prediction": prediction, "raw_candidate": raw, "fit_predictor_count": 1}

    source = tmp_path / "synthetic_source.py"
    data = tmp_path / "synthetic_train.csv"
    source.write_text("synthetic only\n", encoding="utf-8")
    data.write_text("synthetic only\n", encoding="utf-8")
    card = {"id": "C18", "task": "classification", "pilot_predictor_fits": 2}
    identity = make_freeze_identity(
        card=card,
        feature_names=["x"],
        input_files={"train": data},
        source_files=[source],
        evaluation_views=views,
        packages=[],
    )
    run_dir = tmp_path / "runs"
    monkeypatch.setattr(c18_runner, "RUN_DIR", run_dir)
    for name in ("RESULTS_PATH", "VERIFICATION_PATH", "GATE_PATH", "SUMMARY_PATH"):
        monkeypatch.setattr(c18_runner, name, tmp_path / f"{name}.json")
    report = run_pilot(
        card=card,
        artifact_dir=run_dir,
        budget_ledger_path=tmp_path / "budget.jsonl",
        frame=frame,
        target=target,
        feature_names=["x"],
        adapter=synthetic_adapter,
        control_loader=lambda cutoff, rows: controls_by_cutoff[cutoff],
        freeze_identity=identity,
        freeze_path=tmp_path / "freeze.json",
        event_path=tmp_path / "events.jsonl",
        evaluation_views=views,
        report_path=tmp_path / "engine_report.json",
    )
    context = {
        "identity": identity,
        "frame": frame,
        "target": target,
        "views": views,
        "public": {"input_sha256": {}, "source_sha256": {}},
    }
    synthetic_queue = {
        "evaluation": {
            "classification_gate": {
                "pilot_primary_gain_min": 0.0005,
                "pilot_H12_harm_max": 0.001,
                "individual_pilot_view_harm_max": 0.005,
                "supported_group_n_min": 200,
                "supported_group_F1_harm_max": 0.01,
                "positive_class_recall_harm_max": 0.02,
                "positive_and_negative_per_group_min": 30,
            }
        }
    }

    verification, gate = c18_runner._verify_and_gate(context, report, synthetic_queue)
    accounting = {"total_fit_seconds": 0.0, "successful_predictor_fits": 2}
    c18_runner._publish(context, report, verification, gate, accounting)

    assert fit_calls == [55, 67]
    assert verification["status"] == "INDEPENDENT_QA_PASS"
    assert gate["decision"] == "ADVANCE"
    assert verification["views"]["primary"]["n"] == 32
    assert verification["views"]["H12"]["n"] == 96
    for name in ("RESULTS_PATH", "VERIFICATION_PATH", "GATE_PATH", "SUMMARY_PATH"):
        assert getattr(c18_runner, name).is_file()
    published = json.loads(c18_runner.RESULTS_PATH.read_text(encoding="utf-8"))
    assert published["status"] == "PILOT_COMPLETE_ADVANCE"
    assert published["test_data_used"] is False
    assert published["kaggle_access"] is False
