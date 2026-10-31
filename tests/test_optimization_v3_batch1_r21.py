from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.optimization_v3_batch1_verify import verify_report
from src.models.optimization_v3_batch1 import ensure_frozen, make_freeze_identity, run_pilot
from src.models.optimization_v3_batch1_r21 import (
    FEATURE_NAMES,
    XGBoostSVD32Regressor,
    fit_predict,
    rank50_prediction,
    reconstruct_r21,
)


def _synthetic_train() -> pd.DataFrame:
    cycles = np.repeat(np.arange(47, 80, dtype=np.int64), 25)
    rows = np.arange(len(cycles), dtype=np.int64)
    data: dict[str, object] = {
        "record_id": [f"synthetic-{i}" for i in rows],
        "impact_load": (rows % 17).astype(float),
        "entry_friction": (rows % 2).astype(float),
        "cycle_code": cycles,
        "scope_pressure": (rows % 11).astype(float),
        "text_load": (rows % 9).astype(float),
        "scope_trace": (rows % 8).astype(float),
        "org_spread": (rows % 6).astype(float),
        "module_spread": (rows % 7).astype(float),
        "telemetry_flag": (rows % 2).astype(float),
        "org_code": [f"org-{int(cycle % 4)}" if cycle <= 55 else "future-org" for cycle in cycles],
        "module_code": [f"module-{int(i % 5)}" for i in rows],
        "pattern_code": [f"pattern-{int(i % 3)}" for i in rows],
        "signal_stream": [f"alpha token{int(i % 7)} gamma cycle{int(cycle)}" for i, cycle in zip(rows, cycles)],
        "threat_rank": np.clip(30.0 + (rows % 31) * 1.5 + (cycles - 47) * 0.25, 0, 100),
    }
    frame = pd.DataFrame(data)
    assert set(FEATURE_NAMES) == set(frame.columns) - {"record_id", "threat_rank"}
    return frame


def _controls(score_rows: np.ndarray) -> dict[str, np.ndarray]:
    n = len(score_rows)
    lgb = np.full(n, 48.0, dtype=np.float64)
    ridge = np.full(n, 52.0, dtype=np.float64)
    sparse_raw = np.full(n, 50.0, dtype=np.float64)
    sparse_rank = np.full(n, 50.0, dtype=np.float64)
    baseline = np.clip(0.75 * (0.75 * lgb + 0.25 * ridge) + 0.25 * sparse_rank, 0, 100)
    return {
        "baseline_prediction": baseline,
        "baseline_lgb_svd32": lgb,
        "baseline_hybrid_ridge": ridge,
        "baseline_sparse_raw": sparse_raw,
        "baseline_sparse_rank50": sparse_rank,
        "control_rows": np.asarray(score_rows, dtype=np.int64),
    }


def _reconstruct_arrays(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    prediction, baseline, mapped = reconstruct_r21(
        arrays["raw_candidate"], arrays["baseline_prediction"],
        arrays["audit__score_cycles"],
    )
    assert np.allclose(mapped, arrays["audit__candidate_rank50"], atol=1e-12, rtol=0)
    return prediction, baseline


def test_rank50_average_ties_and_fixed_auxiliary_blend():
    raw = np.array([10.0, 10.0, 30.0, 40.0])
    cycles = np.array([56, 56, 56, 56], dtype=np.int64)
    mapped = rank50_prediction(raw, cycles)
    assert np.array_equal(mapped, np.array([23.75, 23.75, 52.5, 70.0]))
    candidate, baseline, saved_map = reconstruct_r21(raw, np.full(4, 50.0), cycles)
    assert np.array_equal(baseline, np.full(4, 50.0))
    assert np.array_equal(saved_map, mapped)
    assert np.array_equal(candidate, np.clip(37.5 + 0.25 * mapped, 0, 100))


def test_r21_rejects_leaked_features_and_inconsistent_champion_controls():
    frame = _synthetic_train()
    X = frame.loc[:, list(FEATURE_NAMES)].iloc[:20].copy()
    with pytest.raises(ValueError, match="exactly the frozen"):
        model = XGBoostSVD32Regressor(55)
        model.fit(X.assign(threat_rank=1.0), frame.threat_rank.iloc[:20].to_numpy())
    fit_x = frame.loc[frame.cycle_code <= 55, list(FEATURE_NAMES)]
    fit_y = frame.loc[frame.cycle_code <= 55, "threat_rank"].to_numpy(dtype=np.float64)
    score_x = frame.loc[frame.cycle_code == 56, list(FEATURE_NAMES)]
    bad = _controls(score_x.index.to_numpy(dtype=np.int64))
    bad["baseline_sparse_rank50"] = bad["baseline_sparse_rank50"] + 1.0
    with pytest.raises(ValueError, match="does not reconstruct"):
        fit_predict(fit_x, fit_y, score_x,
                    controls=bad, card_id="R21", cutoff=55)


def test_r21_real_fit_runs_through_persisted_multiview_report_and_independent_reconstruction(tmp_path: Path):
    frame = _synthetic_train()
    target = frame["threat_rank"].to_numpy(dtype=np.float64)
    views = {
        "W1": [(55, 56, 59)],
        "W4": [(67, 68, 71)],
        "primary": [(55, 56, 59), (67, 68, 71)],
        "H12": [(55, 56, 67), (67, 68, 79)],
    }
    source = tmp_path / "adapter.py"
    source.write_text("# synthetic R21 source identity\n", encoding="utf-8")
    data = tmp_path / "train.csv"
    frame.to_csv(data, index=False)
    identity = make_freeze_identity(
        card={"id": "R21", "task": "regression"},
        feature_names=FEATURE_NAMES,
        input_files={"synthetic_train": data},
        source_files=[source],
        evaluation_views=views,
        packages=[],
    )
    freeze_path = tmp_path / "freeze.json"
    signature = ensure_frozen(freeze_path, identity)
    calls: list[int] = []

    def adapter(fit_x, fit_y, score_x, *, controls, card_id, cutoff):
        assert "record_id" not in fit_x and "threat_rank" not in fit_x
        assert "record_id" not in score_x and "threat_rank" not in score_x
        assert fit_x.cycle_code.max() <= cutoff
        calls.append(cutoff)
        return fit_predict(fit_x, fit_y, score_x, controls=controls,
                           card_id=card_id, cutoff=cutoff)

    report = run_pilot(
        frame=frame,
        target=target,
        card={"id": "R21", "task": "regression", "pilot_predictor_fits": 2},
        feature_names=FEATURE_NAMES,
        adapter=adapter,
        control_loader=lambda cutoff, rows: _controls(rows),
        freeze_identity=identity,
        freeze_path=freeze_path,
        artifact_dir=tmp_path / "artifacts",
        event_path=tmp_path / "attempts.jsonl",
        evaluation_views=views,
        report_path=tmp_path / "pilot.json",
    )
    verification = verify_report(
        report=report,
        frame=frame,
        target=target,
        artifact_dir=tmp_path / "artifacts",
        freeze_signature=signature,
        evaluation_views=views,
        prediction_reconstructors={"R21": _reconstruct_arrays},
    )
    assert report["fit_count_matches_card"] is True
    assert report["views"]["primary"]["rows"] == 200
    assert report["views"]["H12"]["rows"] == 600
    assert verification["status"] == "PASS"
    assert calls == [55, 67]
    for cutoff in (55, 67):
        sidecar = json.loads((tmp_path / "artifacts" / f"R21_fit{cutoff}.json").read_text(encoding="utf-8"))
        state = sidecar["metadata"]["state"]
        assert state["validation_data_passed_to_fit"] is False
        assert state["validation_labels_passed_to_fit"] is False
        assert state["test_data_used_for_fit"] is False
        assert state["xgboost_version"]
        assert state["fit_cycle_max"] == cutoff
