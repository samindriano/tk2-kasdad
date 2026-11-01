from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import json

from scripts import optimization_v3_batch1_r08 as runner
from scripts.optimization_v3_batch1_verify import verify_report
from src.models.optimization_v3_batch1 import ensure_frozen, make_freeze_identity, run_pilot
from src.models.optimization_v3_batch1_r08 import (
    FEATURE_NAMES,
    SCALAR_FEATURE_NAMES,
    LIGHTGBM_PARAMETERS,
    R08TextScalarLGB,
    fit_predict,
    rank50_prediction,
    reconstruct_r08,
    text_scalar_features,
)
from src.models.regression_research_text import _array_digest, _json_digest
from src.models.round1_models import _fit_fingerprint


def _synthetic_train() -> pd.DataFrame:
    cycles = np.repeat(np.arange(47, 80, dtype=np.int64), 6)
    rows = np.arange(len(cycles), dtype=np.int64)
    values: dict[str, object] = {
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
        "signal_stream": [
            f"alpha token{int(i % 17)} beta gamma{int(cycle % 9)} delta{int(i % 11)}"
            for i, cycle in zip(rows, cycles)
        ],
        "record_id": [f"synthetic-r08-{i}" for i in rows],
        "threat_rank": np.clip(30.0 + (rows % 31) * 1.1 + (cycles - 47) * 0.2, 0.0, 100.0),
    }
    frame = pd.DataFrame(values)
    assert set(FEATURE_NAMES).issubset(frame.columns)
    return frame


def _controls(score_rows: np.ndarray, score_cycles: np.ndarray) -> dict[str, np.ndarray]:
    n = len(score_rows)
    lgb = np.full(n, 48.0, dtype=np.float64)
    ridge = np.full(n, 52.0, dtype=np.float64)
    sparse_raw = (score_rows % 13).astype(np.float64) + 42.0
    sparse_rank = rank50_prediction(sparse_raw, score_cycles)
    baseline = np.clip(0.75 * (0.75 * lgb + 0.25 * ridge) + 0.25 * sparse_rank, 0, 100)
    return {
        "baseline_prediction": baseline,
        "baseline_lgb_svd32": lgb,
        "baseline_hybrid_ridge": ridge,
        "baseline_sparse_raw": sparse_raw,
        "baseline_sparse_rank50": sparse_rank,
        "control_rows": score_rows.astype(np.int64),
    }


def test_text_scalar_features_match_pinned_tokenization_and_character_rule():
    values = pd.Series([None, "", " A a 2! ", "Mixed-X x", "éé   --", "   "])
    actual = text_scalar_features(values)
    assert actual.shape == (6, len(SCALAR_FEATURE_NAMES))
    assert np.array_equal(actual[0], np.array([0.0, 0.0, 0.0, 0.0]))
    assert np.array_equal(actual[1], np.array([0.0, 0.0, 0.0, 0.0]))
    assert np.array_equal(actual[2], np.array([3.0, 2.0, 2.0 / 3.0, 8.0]))
    assert np.array_equal(actual[3], np.array([3.0, 2.0, 2.0 / 3.0, 9.0]))
    assert np.array_equal(actual[4], np.array([0.0, 0.0, 0.0, 7.0]))
    assert np.array_equal(actual[5], np.array([0.0, 0.0, 0.0, 3.0]))


def test_rank50_average_ties_and_fixed_r08_blend():
    raw = np.array([10.0, 10.0, 30.0, 40.0])
    cycles = np.array([56, 56, 56, 56], dtype=np.int64)
    mapped = rank50_prediction(raw, cycles)
    assert np.array_equal(mapped, np.array([23.75, 23.75, 52.5, 70.0]))
    candidate, baseline, saved_map = reconstruct_r08(raw, np.full(4, 50.0), cycles)
    assert np.array_equal(baseline, np.full(4, 50.0))
    assert np.array_equal(saved_map, mapped)
    assert np.array_equal(candidate, np.clip(37.5 + 0.25 * mapped, 0, 100))


def test_card_queue_identity_and_resource_gate_are_explicit():
    queue = runner.load_queue(runner.QUEUE_PATH)
    card = next(item for item in queue["cards"] if item["id"] == "R08")
    runner._validate_card_queue(queue, card)
    recipe = runner._recipe(card, {"primary": [(55, 56, 67), (67, 68, 79)]}, runner.CARD_SHA256)
    structured_recipe = recipe["preprocessing"]["structured_recipe"]
    assert "OrdinalEncoder" in structured_recipe
    assert "no scaling or OHE" in structured_recipe
    blocked = runner._resource_preflight(29_705, available_bytes=4 * 1024**3)
    assert blocked["status"] == "BLOCKED_RESOURCE"
    assert blocked["feature_width_upper_bound"] == 76
    assert blocked["structured_feature_width"] == 39
    passing = runner._resource_preflight(29_705, available_bytes=8 * 1024**3)
    assert passing["status"] == "PASS"
    assert passing["estimated_peak_is_observed"] is False


def test_subgroup_diagnostics_preserve_empty_and_low_support_denominators():
    frame = _synthetic_train().iloc[:6].copy()
    frame.loc[1, "signal_stream"] = None
    target = frame["threat_rank"].to_numpy(dtype=np.float64)
    views = {
        "synthetic": {
            "rows": np.arange(len(frame), dtype=np.int64),
            "candidate": np.full(len(frame), 40.0),
            "baseline": np.full(len(frame), 40.0),
        }
    }
    result = runner._group_diagnostics(frame, target, views)["views"]["synthetic"]
    assert result["signal_stream_missing"]["n"] == 1
    assert result["signal_stream_missing"]["supported"] is False
    assert result["signal_stream_missing"]["guard_applies"] is False
    assert result["target_le_5"]["n"] == 0
    assert result["target_le_5"]["candidate_mae"] is None
    assert result["target_ge_95"]["n"] == 0
    assert result["target_ge_95"]["candidate_minus_baseline_mae"] is None


def test_frozen_lgb_parameters_and_synthetic_temporal_adapter_end_to_end():
    frame = _synthetic_train()
    fit_rows = np.flatnonzero(frame.cycle_code.to_numpy() <= 55)
    score_rows = np.flatnonzero((frame.cycle_code.to_numpy() > 55) & (frame.cycle_code.to_numpy() <= 67))
    fit_x = frame.iloc[fit_rows].loc[:, list(FEATURE_NAMES)].copy()
    score_x = frame.iloc[score_rows].loc[:, list(FEATURE_NAMES)].copy()
    target = np.clip(25.0 + 0.2 * fit_x.impact_load.to_numpy() + (fit_x.cycle_code.to_numpy() - 47), 0, 100)
    score_cycles = score_x.cycle_code.to_numpy(dtype=np.int64)
    controls = _controls(score_rows, score_cycles)

    misaligned_controls = {key: value.copy() for key, value in controls.items()}
    misaligned_controls["control_rows"][0] += 1
    with pytest.raises(ValueError, match="aligned"):
        fit_predict(fit_x, target, score_x, controls=misaligned_controls,
                    card_id="R08", cutoff=55)

    assert LIGHTGBM_PARAMETERS["objective"] == "regression_l1"
    assert LIGHTGBM_PARAMETERS["n_estimators"] == 400
    result = fit_predict(
        fit_x, target, score_x, controls=controls, card_id="R08", cutoff=55,
    )
    assert result["fit_predictor_count"] == 1
    assert result["fit_metadata"]["score_labels_passed_to_fit"] is False
    assert result["state"]["feature_names"] == list(FEATURE_NAMES)
    assert result["state"]["derived_row_local_features"]["names"] == list(SCALAR_FEATURE_NAMES)
    assert result["state"]["fit_cycle_max"] == 55
    assert np.isfinite(result["raw_candidate"]).all()
    assert np.all((result["prediction"] >= 0) & (result["prediction"] <= 100))
    assert np.array_equal(result["prediction"], np.clip(
        0.75 * result["baseline_prediction"]
        + 0.25 * result["audit_arrays"]["candidate_rank50"], 0, 100,
    ))


def test_two_fold_synthetic_run_persists_and_independently_verifies_report(tmp_path):
    frame = _synthetic_train()
    target = frame["threat_rank"].to_numpy(dtype=np.float64)
    views = {
        "W1": [(55, 56, 59)],
        "W4": [(67, 68, 71)],
        "primary": [(55, 56, 59), (67, 68, 71)],
        "H12": [(55, 56, 67), (67, 68, 79)],
    }
    input_file = tmp_path / "synthetic_train.csv"
    frame.to_csv(input_file, index=False)
    source_file = tmp_path / "synthetic_source.py"
    source_file.write_text("# synthetic R08 adapter source\n", encoding="utf-8")
    identity = make_freeze_identity(
        card={"id": "R08", "task": "regression"},
        feature_names=FEATURE_NAMES,
        input_files={"synthetic_train": input_file},
        source_files=[source_file],
        evaluation_views=views,
        packages=[],
    )
    freeze_path = tmp_path / "freeze.json"
    signature = ensure_frozen(freeze_path, identity)
    cycles = frame.cycle_code.to_numpy(dtype=np.int64)
    calls = []

    def adapter(fit_x, fit_y, score_x, *, controls, card_id, cutoff):
        assert "record_id" not in fit_x and "threat_rank" not in fit_x
        assert "record_id" not in score_x and "threat_rank" not in score_x
        assert int(fit_x.cycle_code.max()) <= cutoff
        calls.append(cutoff)
        return fit_predict(fit_x, fit_y, score_x, controls=controls,
                           card_id=card_id, cutoff=cutoff)

    report = run_pilot(
        frame=frame,
        target=target,
        card={"id": "R08", "task": "regression", "pilot_predictor_fits": 2},
        feature_names=FEATURE_NAMES,
        adapter=adapter,
        control_loader=lambda cutoff, rows: _controls(rows, cycles[rows]),
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
        prediction_reconstructors={"R08": runner.reconstruct_r08_arrays},
    )
    assert report["fit_count_matches_card"] is True
    assert report["views"]["primary"]["rows"] == 48
    assert report["views"]["H12"]["rows"] == 144
    assert verification["status"] == "PASS"
    assert calls == [55, 67]
    for cutoff in (55, 67):
        sidecar = json.loads((tmp_path / "artifacts" / f"R08_fit{cutoff}.json").read_text(encoding="utf-8"))
        state = sidecar["metadata"]["state"]
        assert state["validation_data_passed_to_fit"] is False
        assert state["validation_labels_passed_to_fit"] is False
        assert state["test_data_used_for_fit"] is False
        assert state["lightgbm_version"]
        assert state["derived_row_local_features"]["names"] == list(SCALAR_FEATURE_NAMES)
        assert state["fit_cycle_max"] == cutoff
        fit_rows = np.flatnonzero(frame.cycle_code.to_numpy() <= cutoff)
        fit_features = frame.iloc[fit_rows].loc[:, list(FEATURE_NAMES)]
        assert state["fit_data_sha256"] == _fit_fingerprint(fit_features, FEATURE_NAMES, target[fit_rows])
        assert state["target_sha256"] == _array_digest(target[fit_rows])
        assert state["derived_row_local_features"]["fit_design_scalar_sha256"] == _array_digest(
            text_scalar_features(fit_features.signal_stream)
        )
        state_for_hash = dict(state)
        expected_state_hash = state_for_hash.pop("state_sha256")
        assert expected_state_hash == _json_digest(state_for_hash)


def test_adapter_rejects_targets_and_feature_leakage():
    train = _synthetic_train()
    fit_rows = np.flatnonzero(train.cycle_code.to_numpy() <= 55)
    fit_x = train.iloc[fit_rows].loc[:, list(FEATURE_NAMES)].copy()
    target = np.full(len(fit_x), 50.0)
    with pytest.raises(ValueError, match="exactly the declared features"):
        R08TextScalarLGB().fit(fit_x.assign(threat_rank=50.0), target)
