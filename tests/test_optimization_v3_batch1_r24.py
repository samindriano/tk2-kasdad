from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts import optimization_v3_batch1_r24 as r24_runner
from scripts import optimization_v3_batch1_r21 as r21_runner
from scripts.optimization_v3_batch1_verify import verify_report
from src.models.optimization_v3_batch1 import make_freeze_identity, run_pilot
from src.models.optimization_v3_batch1_r24 import (
    CARD_ID,
    FEATURE_NAMES,
    RIDGE_PARAMETERS,
    EFInteractionHybridRidge,
    fit_predict,
    rank75_prediction,
)
from src.models.regression_research_text import _RIDGE_PARAMS


def _synthetic_features(cycles: np.ndarray, start: int = 0) -> pd.DataFrame:
    row = np.arange(start, start + len(cycles))
    colors = ("red", "blue", "green", "yellow")
    tokens = ("amber", "birch", "cedar", "dahlia", "elm", "fir")
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
        "signal_stream": [
            None if value % 13 == 0 else
            f"alpha {colors[value % len(colors)]} gamma {tokens[value % len(tokens)]} omega"
            for value in row
        ],
    })
    return frame.loc[:, list(FEATURE_NAMES)]


def _sparse_rank(raw: np.ndarray, cycles: np.ndarray) -> np.ndarray:
    output = np.empty(len(raw), dtype=np.float64)
    for cycle in np.unique(cycles):
        mask = cycles == cycle
        ranks = pd.Series(raw[mask]).rank(method="average").to_numpy(dtype=np.float64)
        output[mask] = 0.5 * raw[mask] + 0.5 * 100.0 * ranks / int(mask.sum())
    return np.clip(output, 0.0, 100.0)


def _controls(rows: np.ndarray, cycles: np.ndarray) -> dict[str, np.ndarray]:
    n = len(rows)
    lgb = 30.0 + (rows % 11).astype(np.float64)
    ridge = 42.0 + (rows % 7).astype(np.float64)
    sparse_raw = 12.0 + (rows % 19).astype(np.float64)
    sparse_rank = _sparse_rank(sparse_raw, cycles)
    baseline = np.clip(0.75 * (0.75 * lgb + 0.25 * ridge) + 0.25 * sparse_rank, 0.0, 100.0)
    return {
        "baseline_prediction": baseline,
        "baseline_lgb_svd32": lgb,
        "baseline_hybrid_ridge": ridge,
        "baseline_sparse_raw": sparse_raw,
        "baseline_sparse_rank50": sparse_rank,
        "control_rows": rows.astype(np.int64),
    }


def test_r24_exact_ridge_recipe_and_rank75_average_ties() -> None:
    assert RIDGE_PARAMETERS == _RIDGE_PARAMS == {"alpha": 10.0, "solver": "lsqr", "tol": 1e-4}
    raw = np.array([10.0, 10.0, 20.0, 0.0, 5.0])
    cycles = np.array([55, 55, 55, 56, 56], dtype=np.int64)
    expected = np.array([40.0, 40.0, 80.0, 37.5, 76.25])
    np.testing.assert_array_equal(rank75_prediction(raw, cycles), expected)


def test_r24_fit_only_interaction_design_and_fixed_replacement_route() -> None:
    fit_cycles = 47 + np.arange(160, dtype=np.int64) % 9
    score_cycles = 56 + np.arange(72, dtype=np.int64) % 12
    fit_x = _synthetic_features(fit_cycles)
    score_x = _synthetic_features(score_cycles, start=len(fit_x))
    score_x.index = np.arange(len(fit_x), len(fit_x) + len(score_x), dtype=np.int64)
    fit_target = np.clip(
        18.0 + 1.7 * fit_x["impact_load"].to_numpy(dtype=float)
        + 24.0 * fit_x["entry_friction"].to_numpy(dtype=float), 0.0, 100.0,
    )
    controls = _controls(score_x.index.to_numpy(dtype=np.int64), score_cycles)

    model = EFInteractionHybridRidge(55).fit(fit_x, fit_target)
    raw_model = model.predict(score_x)
    state = model.audit_state()
    result = fit_predict(
        fit_x, fit_target, score_x, controls=controls, card_id="R24", cutoff=55,
    )

    assert state["parameters"] == RIDGE_PARAMETERS
    assert state["validation_data_passed_to_fit"] is False
    assert state["validation_labels_passed_to_fit"] is False
    assert state["test_data_used_for_fit"] is False
    assert state["interaction"]["reference_group"] == "entry_friction == 0"
    assert state["interaction"]["telemetry_substitution"] is False
    assert state["interaction"]["fit_entry_friction_1_rows"] == int((fit_x.entry_friction == 1).sum())
    assert state["interaction"]["fit_entry_friction_0_rows"] == int((fit_x.entry_friction == 0).sum())
    assert state["interaction"]["fit_interaction_nonzero"] > 0
    assert state["inherited_hybrid_width"] + state["interaction_width"] == state["design_shape"][1]
    assert np.isfinite(state["solver_diagnostics"]["stationarity_ratio"])
    assert np.isfinite(raw_model).all() and np.isfinite(result["prediction"]).all()
    assert np.all((result["prediction"] >= 0) & (result["prediction"] <= 100))
    mapped = rank75_prediction(result["raw_candidate"], score_cycles)
    expected = np.clip(
        0.75 * (0.75 * controls["baseline_lgb_svd32"] + 0.25 * mapped)
        + 0.25 * controls["baseline_sparse_rank50"], 0.0, 100.0,
    )
    np.testing.assert_array_equal(result["prediction"], expected)
    np.testing.assert_array_equal(result["audit_arrays"]["candidate_rank75"], mapped)


def test_r24_synthetic_full_pilot_verification_with_empty_group(tmp_path, monkeypatch) -> None:
    cycles = np.repeat(np.arange(47, 80, dtype=np.int64), 4)
    n = len(cycles)
    target = np.clip(20.0 + 1.5 * (np.arange(n) % 17) + 22.0 * (np.arange(n) % 2), 0.0, 100.0)
    features = _synthetic_features(cycles)
    frame = features.copy()
    frame.insert(0, "record_id", [f"r24-synthetic-{i}" for i in range(n)])
    frame["threat_rank"] = target
    frame["signal_stream"] = "alpha beta"
    views = {
        "W1": [(55, 56, 59)], "W4": [(67, 68, 71)],
        "primary": [(55, 56, 59), (67, 68, 71)],
        "H12": [(55, 56, 67), (67, 68, 79)],
    }
    fit_expected = {55: 36, 67: 84}
    score_expected = {55: 48, 67: 48}
    view_expected = {"W1": 16, "W4": 16, "primary": 32,
                     "H12_E": 48, "H12_L": 48, "H12": 96}
    monkeypatch.setattr(r24_runner, "EXPECTED_FIT_ROWS", fit_expected)
    monkeypatch.setattr(r24_runner, "EXPECTED_SCORE_ROWS", score_expected)
    monkeypatch.setattr(r24_runner, "EXPECTED_VIEW_ROWS", view_expected)
    monkeypatch.setattr(r21_runner, "EXPECTED_VIEW_ROWS", view_expected)
    controls_by_cutoff = {}
    controls_for_context = {}
    for cutoff in (55, 67):
        rows = np.flatnonzero((cycles > cutoff) & (cycles <= cutoff + 12))
        controls_by_cutoff[cutoff] = _controls(rows, cycles[rows])
        controls_for_context[cutoff] = {
            **controls_by_cutoff[cutoff], "score_rows": rows,
            "score_record_ids": frame.iloc[rows]["record_id"].astype(str).to_numpy(dtype=str),
        }

    source = tmp_path / "synthetic_source.py"
    train_file = tmp_path / "synthetic_train.csv"
    source.write_text("synthetic source\n", encoding="utf-8")
    train_file.write_text("synthetic rows\n", encoding="utf-8")
    card = {"id": "R24", "task": "regression", "pilot_predictor_fits": 2}
    identity = make_freeze_identity(
        card=card, feature_names=FEATURE_NAMES, input_files={"train": train_file},
        source_files=[source], evaluation_views=views, packages=[],
    )
    run_dir = tmp_path / "runs"
    ledger = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(r24_runner, "RUN_DIR", run_dir)
    monkeypatch.setattr(r24_runner, "LEDGER_PATH", ledger)
    for name in ("RESULTS_PATH", "VERIFICATION_PATH", "GATE_PATH", "SUMMARY_PATH", "PILOT_PATH"):
        monkeypatch.setattr(r24_runner, name, tmp_path / f"{name}.json")
    report = run_pilot(
        card={"id": "R24", "task": "regression", "pilot_predictor_fits": 2},
        artifact_dir=run_dir, budget_ledger_path=ledger, max_successful_fits=180,
        frame=frame, target=target, feature_names=FEATURE_NAMES,
        adapter=r24_runner.r24_fit_predict,
        control_loader=lambda cutoff, rows: controls_by_cutoff[cutoff],
        freeze_identity=identity, freeze_path=tmp_path / "freeze.json",
        event_path=ledger, evaluation_views=views, threads=2,
        report_path=tmp_path / "pilot.json",
    )
    context = {"identity": identity, "frame": frame, "target": target,
               "cycles": cycles, "views": views,
                   "controls": controls_for_context,
               "public": {"input_sha256": {}, "source_sha256": {}}}
    synthetic_queue = {"evaluation": {"regression_gate": {
        "pilot_primary_MAE_gain_min": 0.03,
        "pilot_H12_MAE_harm_max": 0.05,
        "individual_pilot_view_MAE_harm_max": 0.5,
        "supported_group_n_min": 100,
        "supported_tail_or_missing_MAE_harm_max": 0.25,
    }}}
    verification = r24_runner._verify_pilot(context, report, synthetic_queue)
    generic = verify_report(
        report=report, frame=frame, target=target, artifact_dir=run_dir,
        freeze_signature=identity and json.loads((tmp_path / "freeze.json").read_text())["signature"],
        evaluation_views=views, prediction_reconstructors={CARD_ID: r24_runner.reconstruct_r24_arrays},
    )
    assert verification["decision"] in {"ADVANCE", "RETIRE"}
    assert generic["status"] == "PASS"
    assert verification["fit_execution_accounting"]["successful_predictor_fits"] == 2
    assert verification["group_diagnostics"]["views"]["primary"]["signal_stream_missing"]["n"] == 0
    assert verification["group_diagnostics"]["views"]["primary"]["signal_stream_missing"]["supported"] is False
    r24_runner._publish(context, report, verification, generic)
    published = json.loads(r24_runner.RESULTS_PATH.read_text(encoding="utf-8"))
    assert published["status"] == f"PILOT_COMPLETE_{verification['decision']}"
    assert published["test_data_used"] is False
    assert published["kaggle_access"] is False
