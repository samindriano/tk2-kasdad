from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts import optimization_v3_batch1_c13 as c13_runner
from src.models.optimization_v3_batch1 import make_freeze_identity, run_pilot
from src.models.optimization_v3_batch1_c13 import (
    BLEND_WEIGHT,
    FEATURE_NAMES,
    HGB_PARAMETERS,
    PILOT_CUTOFFS,
    SVD32StructuredHGBClassifier,
    SVD_COMPONENTS,
    SVD_PARAMETERS,
    THRESHOLD,
    fit_predict,
)
from src.models.classification_models import _HGB_PARAMS as C1_B_HGB_PARAMETERS
from src.models.regression_research_text import _SVD_PARAMS


def _synthetic_features(cycles: np.ndarray, start: int = 0) -> pd.DataFrame:
    row = np.arange(start, start + len(cycles))
    def document(index: int):
        if index % 17 == 0:
            return None
        terms = [f"word{(index * 7 + offset * 11) % 60:02d}" for offset in range(10)]
        return " ".join(["commonstream", *terms])
    frame = pd.DataFrame({
        "impact_load": (row % 17).astype(float),
        "entry_friction": (row % 2).astype(int),
        "cycle_code": cycles.astype(int),
        "scope_pressure": (row % 11).astype(float),
        "text_load": (row % 9).astype(float),
        "scope_trace": [str(value % 5) for value in row],
        "org_spread": (row % 4).astype(int),
        "module_spread": (row % 6).astype(int),
        "telemetry_flag": (row % 2).astype(int),
        "org_code": [f"org-{value % 7}" for value in row],
        "module_code": [f"module-{value % 8}" for value in row],
        "pattern_code": [f"pattern-{value % 3}" for value in row],
        "signal_stream": [document(int(value)) for value in row],
    })
    return frame.loc[:, list(FEATURE_NAMES)]


def _controls(rows: np.ndarray, missing: np.ndarray) -> dict[str, np.ndarray]:
    c2b = np.full(len(rows), 0.4, dtype=np.float64)
    cb = np.full(len(rows), 0.7, dtype=np.float64)
    old_sparse = np.full(len(rows), 0.2, dtype=np.float64)
    baseline = np.where(missing, cb, 0.75 * c2b + BLEND_WEIGHT * old_sparse)
    return {
        "baseline_prediction": baseline,
        "baseline_sparse_raw": old_sparse,
        "c2b_probability": c2b,
        "cb_probability": cb,
        "stream_missing": missing,
        "control_rows": rows.astype(np.int64),
    }


def test_c13_inherits_c1_b_hgb_and_exact_svd32_recipe() -> None:
    assert HGB_PARAMETERS == dict(C1_B_HGB_PARAMETERS)
    assert SVD_COMPONENTS == 32
    assert SVD_PARAMETERS == dict(_SVD_PARAMS) == {
        "algorithm": "randomized", "n_iter": 7, "random_state": 42,
    }
    assert THRESHOLD == 0.50
    assert PILOT_CUTOFFS == (55, 67)


def test_c13_fit_only_svd_native_mask_and_fixed_availability_route() -> None:
    fit_cycles = 47 + np.arange(180, dtype=np.int64) % 9
    score_cycles = 56 + np.arange(40, dtype=np.int64) % 12
    fit_x = _synthetic_features(fit_cycles)
    score_x = _synthetic_features(score_cycles, start=len(fit_x))
    score_x.loc[score_x.index[:2], "org_code"] = "future-org-unseen"
    score_x.index = np.arange(len(fit_x), len(fit_x) + len(score_x), dtype=np.int64)
    target = ((fit_x["impact_load"].to_numpy(dtype=int) + fit_x["entry_friction"].to_numpy()) % 2).astype(np.int8)

    model = SVD32StructuredHGBClassifier(55).fit(fit_x, target)
    state = model.audit_state()
    raw = model.predict_probability(score_x)
    missing = score_x["signal_stream"].isna().to_numpy()
    controls = _controls(score_x.index.to_numpy(dtype=np.int64), missing)
    result = fit_predict(fit_x, target, score_x, controls=controls, card_id="C13", cutoff=55)

    assert state["svd_parameters"]["fit_rows"] == len(fit_x)
    assert state["svd_parameters"]["n_components_actual"] == 32
    assert state["projected_text_columns_categorical"] is False
    assert state["missing_indicator_categorical"] is False
    mask = state["categorical_mask"]
    assert mask[:32] == [False] * 32 and mask[-1] is False
    assert any(mask[32:-1])
    assert state["hgb_parameters"]["categorical_features"] == mask
    assert {key: state["hgb_parameters"][key] for key in HGB_PARAMETERS} == HGB_PARAMETERS
    assert state["validation_data_passed_to_fit"] is False
    assert state["validation_labels_passed_to_fit"] is False
    assert state["sample_weight_used"] is False
    assert state["fit_rows"] == len(fit_x)
    assert result["state"]["cutoff"] == 55
    assert result["state"]["fit_cycle_max"] <= 55
    assert np.isfinite(raw).all()
    assert np.isfinite(result["prediction"]).all()
    assert np.all((result["prediction"] >= 0) & (result["prediction"] <= 1))
    np.testing.assert_array_equal(result["prediction"][missing], controls["cb_probability"][missing])
    np.testing.assert_allclose(
        result["prediction"][~missing],
        0.75 * controls["c2b_probability"][~missing] + BLEND_WEIGHT * result["raw_candidate"][~missing],
        rtol=0,
        atol=0,
    )
    assert model._prediction_diagnostics["unknown_structured_categories"]["org_code"] >= 1


def test_c13_synthetic_pilot_independent_verification_and_publication(tmp_path, monkeypatch) -> None:
    cycles = np.repeat(np.arange(47, 80, dtype=np.int64), 4)
    target = np.tile(np.array([0, 1, 0, 1], dtype=np.int8), 33)
    frame = pd.DataFrame({
        "record_id": [f"synthetic-{index}" for index in range(len(cycles))],
        "cycle_code": cycles,
        "x": target.astype(np.float64),
        "signal_stream": ["alpha beta" for _ in range(len(cycles))],
    })
    views = {
        "W1": [(55, 56, 59)],
        "W4": [(67, 68, 71)],
        "primary": [(55, 56, 59), (67, 68, 71)],
        "H12": [(55, 56, 67), (67, 68, 79)],
    }
    monkeypatch.setattr(c13_runner, "EXPECTED_VIEW_ROWS", {"W1": 16, "W4": 16, "primary": 32, "H12": 96})
    controls_by_cutoff = {}
    for cutoff, low, high in ((55, 56, 67), (67, 68, 79)):
        rows = np.flatnonzero((cycles >= low) & (cycles <= high))
        missing = frame.iloc[rows]["signal_stream"].isna().to_numpy()
        c2b = np.full(len(rows), 0.5)
        cb = np.full(len(rows), 0.5)
        old_sparse = np.full(len(rows), 0.5)
        baseline = np.where(missing, cb, 0.75 * c2b + BLEND_WEIGHT * old_sparse)
        controls_by_cutoff[cutoff] = {
            "baseline_prediction": baseline,
            "baseline_sparse_raw": old_sparse,
            "c2b_probability": c2b,
            "cb_probability": cb,
            "stream_missing": missing,
            "control_rows": rows,
        }

    fit_calls = []

    def synthetic_adapter(fit_x, fit_y, score_x, *, controls, card_id, cutoff):
        fit_calls.append(cutoff)
        raw = np.where(score_x["x"].to_numpy(dtype=float) >= 0.5, 0.9, 0.1)
        missing = controls["stream_missing"]
        prediction = np.where(
            missing,
            controls["cb_probability"],
            0.75 * controls["c2b_probability"] + BLEND_WEIGHT * raw,
        )
        return {"prediction": prediction, "raw_candidate": raw, "fit_predictor_count": 1}

    source = tmp_path / "synthetic_source.py"
    data = tmp_path / "synthetic_train.csv"
    source.write_text("synthetic only\n", encoding="utf-8")
    data.write_text("synthetic only\n", encoding="utf-8")
    card = {"id": "C13", "task": "classification", "pilot_predictor_fits": 2}
    identity = make_freeze_identity(
        card=card,
        feature_names=["x"],
        input_files={"train": data},
        source_files=[source],
        evaluation_views=views,
        packages=[],
    )
    run_dir = tmp_path / "runs"
    monkeypatch.setattr(c13_runner, "RUN_DIR", run_dir)
    for name in ("RESULTS_PATH", "VERIFICATION_PATH", "GATE_PATH", "SUMMARY_PATH"):
        monkeypatch.setattr(c13_runner, name, tmp_path / f"{name}.json")
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
    synthetic_queue = {"evaluation": {"classification_gate": {
        "pilot_primary_gain_min": 0.0005,
        "pilot_H12_harm_max": 0.001,
        "individual_pilot_view_harm_max": 0.005,
        "supported_group_n_min": 200,
        "supported_group_F1_harm_max": 0.01,
        "positive_class_recall_harm_max": 0.02,
        "positive_and_negative_per_group_min": 30,
    }}}

    verification, gate = c13_runner._verify_and_gate(context, report, synthetic_queue)
    accounting = {"total_fit_seconds": 0.0, "successful_predictor_fits": 2}
    c13_runner._publish(context, report, verification, gate, accounting)

    assert fit_calls == [55, 67]
    assert verification["status"] == "INDEPENDENT_QA_PASS"
    assert gate["decision"] == "ADVANCE"
    assert verification["views"]["primary"]["n"] == 32
    assert verification["views"]["H12"]["n"] == 96
    missing_group = verification["views"]["primary"]["subgroup_diagnostics"]["groups"]["signal_stream"]["missing"]
    assert missing_group["n"] == 0
    assert missing_group["supported"] is False
    for name in ("RESULTS_PATH", "VERIFICATION_PATH", "GATE_PATH", "SUMMARY_PATH"):
        assert getattr(c13_runner, name).is_file()
    published = json.loads(c13_runner.RESULTS_PATH.read_text(encoding="utf-8"))
    assert published["status"] == "PILOT_COMPLETE_ADVANCE"
    assert published["test_data_used"] is False
    assert published["kaggle_access"] is False
