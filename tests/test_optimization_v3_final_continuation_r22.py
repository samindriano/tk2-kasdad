from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from scripts import optimization_v3_final_continuation_r22 as runner
from src.models import optimization_v3_final_continuation_r22 as adapter


def test_cycle_percentile_uses_average_ranks_and_full_cycle_denominator() -> None:
    raw = np.array([1.0, 1.0, 3.0, 4.0, 8.0])
    cycles = np.array([60, 60, 60, 60, 61], dtype=np.int64)
    got = adapter.cycle_average_percentile(raw, cycles)
    # Tied ranks are 1.5, 1.5, 3, 4; divided by the four-row full cycle.
    assert np.array_equal(got, np.array([37.5, 37.5, 75.0, 100.0, 100.0]))


def test_two_policies_share_raw_scores_and_missing_uses_exact_champion() -> None:
    raw = np.array([10.0, 10.0, 20.0, 30.0, 40.0, 50.0])
    baseline = np.array([40.0, 42.0, 44.0, 46.0, 48.0, 50.0])
    cycles = np.array([60, 60, 60, 61, 61, 61], dtype=np.int64)
    present = np.array([True, False, True, False, True, True])
    got = adapter.postprocess_policies(raw, baseline, cycles, present)
    pct = adapter.cycle_average_percentile(raw, cycles)
    expected_original = np.clip(0.75 * baseline + 0.25 * np.clip(0.5 * raw + 0.5 * pct, 0, 100), 0, 100)
    expected_present = np.where(present, np.clip(0.75 * baseline + 0.25 * pct, 0, 100), baseline)
    assert np.array_equal(got["R22_ORIGINAL"], expected_original)
    assert np.array_equal(got["R22_PCT_PRESENT"], expected_present)
    assert np.array_equal(got["R22_PCT_PRESENT"][~present], baseline[~present])


def test_adapter_only_admits_four_new_origins() -> None:
    for cutoff in adapter.CONTINUATION_CUTOFFS:
        assert adapter.FinalContinuationR22Regressor(cutoff).cutoff == cutoff
    for cutoff in (55, 58, 67, 76):
        with pytest.raises(ValueError):
            adapter.FinalContinuationR22Regressor(cutoff)


def test_fit_api_has_no_score_target_and_never_passes_one(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    class FakeRegressor:
        def __init__(self, cutoff: int):
            seen["cutoff"] = cutoff

        def fit(self, X: pd.DataFrame, y: np.ndarray) -> "FakeRegressor":
            seen["fit_rows"] = X.index.to_numpy().copy()
            seen["fit_target"] = np.asarray(y).copy()
            return self

        def predict(self, X: pd.DataFrame) -> np.ndarray:
            seen["score_rows"] = X.index.to_numpy().copy()
            return np.full(len(X), 50.0)

        def audit_state(self) -> dict[str, object]:
            return {"validation_labels_passed_to_fit": False}

    monkeypatch.setattr(adapter, "FinalContinuationR22Regressor", FakeRegressor)
    columns = list(adapter.FEATURE_NAMES)
    fit = pd.DataFrame({name: [1, 2, 3, 4] for name in columns}, index=[0, 1, 2, 3])
    fit["cycle_code"] = [47, 48, 49, 59]
    score = pd.DataFrame({name: [1, 2, 3, 4] for name in columns}, index=[4, 5, 6, 7])
    score["cycle_code"] = [60, 61, 62, 63]
    result = adapter.fit_raw_predict(fit, np.array([10.0, 20.0, 30.0, 40.0]), score, cutoff=59)
    assert result["raw_prediction"].shape == (4,)
    assert np.array_equal(seen["fit_rows"], fit.index.to_numpy())
    assert np.array_equal(seen["score_rows"], score.index.to_numpy())
    assert len(seen["fit_target"]) == len(fit)
    assert "score_target" not in inspect.signature(adapter.fit_raw_predict).parameters
    assert "threat_rank" not in score.columns


def _synthetic_views() -> tuple[pd.DataFrame, np.ndarray, dict[str, dict[str, np.ndarray]]]:
    n = 240
    frame = pd.DataFrame({
        "signal_stream": [None] * 80 + ["stream"] * 160,
    })
    target = np.array([2.0] * 80 + [50.0] * 80 + [98.0] * 80)
    rows = np.arange(n, dtype=np.int64)
    baseline = np.full(n, 50.0)
    policy = baseline.copy()
    views: dict[str, dict[str, np.ndarray]] = {}
    for name in runner.REQUIRED_VIEWS:
        views[name] = {"rows": rows, "baseline": baseline, "R22_ORIGINAL": policy,
                       "R22_PCT_PRESENT": policy}
    return frame, target, views


def test_group_guards_handle_supported_and_unsupported_cells() -> None:
    frame, target, views = _synthetic_views()
    groups = runner._group_table(frame, target, views, "R22_ORIGINAL")
    primary = groups["views"]["primary"]
    assert primary["signal_stream_missing"]["n"] == 80
    assert primary["signal_stream_missing"]["supported"] is False
    assert primary["signal_stream_missing"]["pass"] is None
    assert primary["threat_rank_le_5"]["n"] == 80
    assert primary["threat_rank_le_5"]["supported"] is False
    assert primary["threat_rank_ge_95"]["n"] == 80
    assert primary["threat_rank_ge_95"]["supported"] is False


def _synthetic_bundle(context: dict[str, object], cutoff: int) -> dict[str, np.ndarray]:
    rows = np.flatnonzero((context["cycles"] >= cutoff + 1) & (context["cycles"] <= cutoff + 4))
    frame = context["frame"]
    cycles = np.asarray(context["cycles"])[rows]
    baseline = np.asarray(context["primary_controls"][cutoff]["baseline_prediction"], dtype=float)
    raw = np.arange(len(rows), dtype=float) + 10.0
    present = np.array([True, False, True, True])
    pct = adapter.cycle_average_percentile(raw, cycles)
    component = np.clip(0.5 * raw + 0.5 * pct, 0.0, 100.0)
    original = np.clip(0.75 * baseline + 0.25 * component, 0.0, 100.0)
    pct_present_value = np.clip(0.75 * baseline + 0.25 * pct, 0.0, 100.0)
    pct_present = np.where(present, pct_present_value, baseline)
    return {
        "score_rows": rows.astype(np.int64),
        "score_record_ids": frame.iloc[rows]["record_id"].astype(str).to_numpy(dtype=str),
        "score_cycle_codes": cycles.astype(np.int64),
        "champion_prediction": baseline,
        "raw_candidate": raw,
        "cycle_percentile": pct,
        "original_component": component,
        "R22_ORIGINAL": original,
        "R22_PCT_PRESENT": pct_present,
        "stream_present": present,
    }


def test_resume_reuses_complete_bundle_and_blocks_partial_bundle(tmp_path: object,
                                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    cutoff = 59
    rows = np.arange(4, dtype=np.int64)
    frame = pd.DataFrame({"record_id": [f"id-{i}" for i in range(4)],
                          "signal_stream": ["x", "", "x", "x"]})
    cycles = np.array([60, 61, 62, 63], dtype=np.int64)
    baseline = np.array([40.0, 41.0, 42.0, 43.0])
    context: dict[str, object] = {
        "frame": frame,
        "cycles": cycles,
        "signature": "synthetic-freeze-signature",
        "primary_controls": {cutoff: {"rows": rows, "record_ids": frame.record_id.to_numpy(),
                                       "baseline_prediction": baseline}},
    }
    run_dir = tmp_path / "r22"
    monkeypatch.setattr(runner, "RUN_DIR", run_dir)
    run_dir.mkdir()
    arrays = _synthetic_bundle(context, cutoff)
    path = run_dir / f"R22_fit{cutoff}.npz"
    runner.write_bundle_exclusive(path, arrays, runner._bundle_identity(context, cutoff, rows),
                                  {"fit_predictor_count": 1,
                                   "fit_metadata": {"score_labels_passed_to_fit": False},
                                   "fit_seconds": 0.25})
    resumed = runner._load_or_fit_cutoff(context, cutoff, lock_held=True)
    assert resumed["status"] == "RESUMED_VERIFIED"
    assert all(resumed["policy_reconstruction"].values())

    partial_dir = tmp_path / "partial"
    partial_dir.mkdir()
    monkeypatch.setattr(runner, "RUN_DIR", partial_dir)
    partial_path = partial_dir / f"R22_fit{cutoff}.npz"
    partial_path.write_bytes(b"partial")
    with pytest.raises(Exception, match="partial bundle"):
        runner._load_or_fit_cutoff(context, cutoff, lock_held=True)
