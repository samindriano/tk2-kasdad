from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import optimization_v3_batch1_r09 as runner
from scripts.optimization_v3_batch1_verify import verify_report
from src.models.optimization_v3_batch1 import make_freeze_identity, run_pilot
from src.models.optimization_v3_batch1_r09 import (
    BLEND_WEIGHT,
    HUBER_ALPHA,
    HUBER_PARAMETERS,
    PILOT_CUTOFFS,
    HuberSVD32Regressor,
    fit_predict,
    rank50_independent,
)
from src.models.regression_research_tabular import COMMON_PARAMETERS, PARAMETER_SETS
from src.models.regression_research_text import HYBRID_FEATURES


FEATURES = tuple(HYBRID_FEATURES)


def _synthetic_frame(per_cycle: int = 15) -> pd.DataFrame:
    cycles = np.repeat(np.arange(47, 80, dtype=np.int64), per_cycle)
    rows = np.arange(len(cycles), dtype=np.int64)
    text = []
    for row in rows:
        tokens = [f"token{(row + offset * 7) % 80:02d}" for offset in range(5)]
        text.append(" ".join(tokens))
    values: dict[str, object] = {
        "record_id": [f"r09-synthetic-{row}" for row in rows],
        "impact_load": (rows % 17).astype(float),
        "entry_friction": (rows % 2).astype(float),
        "cycle_code": cycles,
        "scope_pressure": (rows % 11).astype(float),
        "text_load": (rows % 9).astype(float),
        "scope_trace": (rows % 8).astype(float),
        "org_spread": (rows % 6).astype(float),
        "module_spread": (rows % 7).astype(float),
        "telemetry_flag": (rows % 2).astype(float),
        "org_code": [f"org-{row % 5}" for row in rows],
        "module_code": [f"module-{row % 7}" for row in rows],
        "pattern_code": [f"pattern-{row % 4}" for row in rows],
        "signal_stream": text,
    }
    frame = pd.DataFrame(values)
    assert set(FEATURES) == set(frame.columns) - {"record_id"}
    frame["threat_rank"] = np.clip(
        18.0 + 1.3 * frame["impact_load"].to_numpy()
        + 17.0 * frame["entry_friction"].to_numpy()
        + 0.4 * (cycles - 47), 0.0, 100.0,
    )
    return frame


def _controls(rows: np.ndarray, cycles: np.ndarray) -> dict[str, np.ndarray]:
    n = len(rows)
    lgb = 36.0 + (rows % 13).astype(np.float64)
    ridge = 43.0 + (rows % 9).astype(np.float64)
    sparse_raw = 20.0 + (rows % 17).astype(np.float64)
    sparse_rank = rank50_independent(sparse_raw, cycles)
    baseline = np.clip(0.75 * (0.75 * lgb + 0.25 * ridge) + 0.25 * sparse_rank, 0.0, 100.0)
    return {
        "baseline_prediction": baseline,
        "baseline_lgb_svd32": lgb,
        "baseline_hybrid_ridge": ridge,
        "baseline_sparse_raw": sparse_raw,
        "baseline_sparse_rank50": sparse_rank,
        "control_rows": rows.astype(np.int64),
    }


def test_r09_changes_only_huber_objective_and_alpha_and_rank_ties_are_average() -> None:
    inherited = {**dict(COMMON_PARAMETERS), **dict(PARAMETER_SETS["base"])}
    assert HUBER_PARAMETERS == {**inherited, "objective": "huber", "alpha": 0.9}
    assert HUBER_ALPHA == 0.9
    raw = np.array([12.0, 12.0, 30.0, 40.0])
    cycles = np.array([56, 56, 56, 56], dtype=np.int64)
    mapped = rank50_independent(raw, cycles)
    np.testing.assert_array_equal(mapped, np.array([24.75, 24.75, 52.5, 70.0]))


def test_r09_fit_only_huber_svd32_and_fixed_auxiliary_route() -> None:
    frame = _synthetic_frame()
    cycles = frame["cycle_code"].to_numpy(dtype=np.int64)
    fit_rows = np.flatnonzero(cycles <= 55)
    score_rows = np.flatnonzero((cycles > 55) & (cycles <= 67))
    fit_x = frame.iloc[fit_rows].loc[:, list(FEATURES)]
    score_x = frame.iloc[score_rows].loc[:, list(FEATURES)]
    fit_y = frame.iloc[fit_rows]["threat_rank"].to_numpy(dtype=np.float64)
    control = _controls(score_rows, cycles[score_rows])

    model = HuberSVD32Regressor(55).fit(fit_x, fit_y)
    result = fit_predict(fit_x, fit_y, score_x, controls=control, card_id="R09", cutoff=55)
    state = model.audit_state()
    mapped = rank50_independent(result["raw_candidate"], cycles[score_rows])
    expected = np.clip(0.75 * control["baseline_prediction"] + BLEND_WEIGHT * mapped, 0.0, 100.0)

    assert state["objective"] == "huber" and state["alpha"] == 0.9
    assert state["actual_hyperparameters"] == HUBER_PARAMETERS
    assert state["svd"]["n_components_actual"] == 32
    assert state["validation_data_passed_to_fit"] is False
    assert state["validation_labels_passed_to_fit"] is False
    assert state["test_data_used_for_fit"] is False
    assert state["sample_weight_used"] is False
    np.testing.assert_array_equal(result["audit_arrays"]["candidate_rank50"], mapped)
    np.testing.assert_array_equal(result["prediction"], expected)
    assert np.isfinite(result["prediction"]).all()
    assert np.all((result["prediction"] >= 0.0) & (result["prediction"] <= 100.0))


def test_r09_two_fold_synthetic_pilot_independent_qa(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    frame = _synthetic_frame(per_cycle=15)
    target = frame["threat_rank"].to_numpy(dtype=np.float64)
    cycles = frame["cycle_code"].to_numpy(dtype=np.int64)
    views = {
        "W1": [(55, 56, 59)],
        "W4": [(67, 68, 71)],
        "primary": [(55, 56, 59), (67, 68, 71)],
        "H12": [(55, 56, 67), (67, 68, 79)],
    }
    fit_per_cycle = 15
    monkeypatch.setattr(runner, "EXPECTED_FIT_ROWS", {55: 9 * fit_per_cycle, 67: 21 * fit_per_cycle})
    monkeypatch.setattr(runner, "EXPECTED_VIEW_ROWS", {
        "W1": 4 * fit_per_cycle, "W4": 4 * fit_per_cycle,
        "primary": 8 * fit_per_cycle, "H12_E": 12 * fit_per_cycle,
        "H12_L": 12 * fit_per_cycle, "H12": 24 * fit_per_cycle,
    })
    controls_by_cutoff: dict[int, dict[str, np.ndarray]] = {}
    controls_context: dict[int, dict[str, np.ndarray]] = {}
    for cutoff in PILOT_CUTOFFS:
        rows = np.flatnonzero((cycles > cutoff) & (cycles <= cutoff + 12))
        values = _controls(rows, cycles[rows])
        controls_by_cutoff[cutoff] = values
        controls_context[cutoff] = {
            **values,
            "score_rows": rows,
            "score_record_ids": frame.iloc[rows]["record_id"].astype(str).to_numpy(dtype=str),
        }

    source = tmp_path / "synthetic_r09_source.py"
    train_path = tmp_path / "synthetic_r09_train.csv"
    source.write_text("synthetic R09 source identity\n", encoding="utf-8")
    frame.to_csv(train_path, index=False)
    card = {"id": "R09", "task": "regression", "pilot_predictor_fits": 2}
    identity = make_freeze_identity(
        card=card, feature_names=FEATURES, input_files={"synthetic_train": train_path},
        source_files=[source], evaluation_views=views, packages=[],
    )
    run_dir = tmp_path / "runs"
    ledger = tmp_path / "ledger.jsonl"
    freeze_path = tmp_path / "freeze.json"
    monkeypatch.setattr(runner, "RUN_DIR", run_dir)
    monkeypatch.setattr(runner, "LEDGER_PATH", ledger)

    report = run_pilot(
        card=card, artifact_dir=run_dir, budget_ledger_path=ledger,
        max_successful_fits=180, frame=frame, target=target,
        feature_names=FEATURES, adapter=fit_predict,
        control_loader=lambda cutoff, rows: controls_by_cutoff[cutoff],
        freeze_identity=identity, freeze_path=freeze_path, event_path=ledger,
        evaluation_views=views, threads=2, report_path=tmp_path / "pilot.json",
    )
    context = {
        "identity": identity, "frame": frame, "target": target, "cycles": cycles,
        "views": views, "controls": controls_context,
        "queue": {"evaluation": {"regression_gate": {
            "pilot_primary_MAE_gain_min": 0.03,
            "pilot_H12_MAE_harm_max": 0.05,
            "individual_pilot_view_MAE_harm_max": 0.5,
            "supported_group_n_min": 100,
            "supported_tail_or_missing_MAE_harm_max": 0.25,
        }}},
    }
    verification = runner._verify_pilot(context, report)
    generic = verify_report(
        report=report, frame=frame, target=target, artifact_dir=run_dir,
        freeze_signature=json.loads(freeze_path.read_text(encoding="utf-8"))["signature"],
        evaluation_views=views,
        prediction_reconstructors={"R09": runner.reconstruct_r09_arrays},
    )

    assert report["fit_count_matches_card"] is True
    assert verification["status"] == "PASS"
    assert verification["fit_execution_accounting"]["successful_predictor_fits"] == 2
    assert generic["status"] == "PASS"
    for cutoff in PILOT_CUTOFFS:
        sidecar = json.loads((run_dir / f"R09_fit{cutoff}.json").read_text(encoding="utf-8"))
        state = sidecar["metadata"]["state"]
        fit_rows = np.flatnonzero(cycles <= cutoff)
        assert state["objective"] == "huber"
        assert state["alpha"] == 0.9
        assert state["fit_row_positions_sha256"] == hashlib.sha256(
            np.asarray(fit_rows, dtype="<i8").tobytes(order="C")
        ).hexdigest()
