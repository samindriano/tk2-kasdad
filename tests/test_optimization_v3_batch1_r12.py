from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.optimization_v3_batch1_verify import verify_report
from src.models.optimization_v3_batch1 import make_freeze_identity, run_pilot
from src.models import optimization_v3_batch1_r12 as r12
from src.models.regression_research_text import HYBRID_FEATURES


FEATURES = tuple(HYBRID_FEATURES)


class _FakeBooster:
    def current_iteration(self) -> int:
        return 200

    def model_to_string(self, num_iteration: int = -1) -> str:
        assert num_iteration == -1
        return "synthetic-r12-ranker-state"


class _FakeRanker:
    instances: list["_FakeRanker"] = []

    def __init__(self, **parameters):
        self.parameters = parameters
        self.booster_ = _FakeBooster()
        self.fit_group = None
        self.fit_target = None
        self.instances.append(self)

    def fit(self, X, y, *, group):
        assert X.shape[0] == len(y) == sum(group)
        self.fit_group = list(group)
        self.fit_target = np.asarray(y, dtype=np.int32).copy()
        return self

    def predict(self, X):
        # Ties exercise average-rank handling while preserving input alignment.
        return (np.arange(X.shape[0], dtype=np.float64) % 5.0) - 2.0

    def get_params(self, deep: bool = False):
        assert deep is False
        return dict(self.parameters)


def _synthetic_frame(per_cycle: int = 15) -> pd.DataFrame:
    cycles = np.repeat(np.arange(47, 80, dtype=np.int64), per_cycle)
    rows = np.arange(len(cycles), dtype=np.int64)
    text = [" ".join(f"token{(row + offset * 7) % 80:02d}" for offset in range(5)) for row in rows]
    frame = pd.DataFrame({
        "record_id": [f"r12-synthetic-{row}" for row in rows],
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
        "threat_rank": ((rows * 7) % 100).astype(float) + 0.5,
    })
    assert set(FEATURES) == set(frame.columns) - {"record_id", "threat_rank"}
    return frame


def _controls(rows: np.ndarray, cycles: np.ndarray, per_cycle: int) -> dict[str, np.ndarray]:
    n = len(rows)
    return {
        "baseline_prediction": np.full(n, 48.0, dtype=np.float64) + (rows % 7) * 0.1,
        "expected_score_cycle_count": np.full(n, per_cycle, dtype=np.int64),
        "control_rows": rows.astype(np.int64),
    }


def test_grouping_stably_contiguates_queries_and_inverse_restores_positions() -> None:
    cycles = np.tile(np.array([49, 47, 48], dtype=np.int64), 10)
    target = np.tile(np.array([0.0, 10.0, 29.9, 30.0, 49.9, 50.0, 70.0, 89.9, 90.0, 100.0]), 3)

    order, inverse, groups, sorted_grades = r12.grouped_fit_order(cycles, target)

    sorted_cycles = cycles[order]
    np.testing.assert_array_equal(sorted_cycles, np.repeat(np.array([47, 48, 49]), 10))
    np.testing.assert_array_equal(order[inverse], np.arange(len(cycles)))
    np.testing.assert_array_equal(groups, np.array([10, 10, 10], dtype=np.int32))
    np.testing.assert_array_equal(sorted_grades, np.clip(np.floor(target / 10), 0, 9).astype(np.int32)[order])
    assert np.array_equal(r12.LABEL_GAIN, np.arange(10))


def test_cycle_percentile_uses_average_ties_and_requires_complete_denominator() -> None:
    scores = np.array([2.0, 1.0, 1.0, 8.0, 5.0, 5.0, 9.0])
    cycles = np.array([56, 56, 56, 56, 57, 57, 57])
    expected_counts = np.array([4, 4, 4, 4, 3, 3, 3])

    result = r12.cycle_average_rank_percentile(
        scores, cycles, expected_cycles=[56, 57], expected_cycle_counts=expected_counts
    )

    np.testing.assert_array_equal(result, np.array([75.0, 37.5, 37.5, 100.0, 50.0, 50.0, 100.0]))
    with pytest.raises(r12.R12IntegrityError, match="complete cycle"):
        r12.cycle_average_rank_percentile(
            scores, cycles, expected_cycle_counts=np.full(len(scores), 99, dtype=np.int64)
        )
    with pytest.raises(r12.R12IntegrityError, match="complete future cycles"):
        r12.cycle_average_rank_percentile(scores, cycles, expected_cycles=[56, 57, 58])


def test_blend_uses_fixed_weights_and_clips_to_target_range() -> None:
    result = r12.blend_rank_percentile(
        np.array([0.0, 100.0, 60.0]), np.array([100.0, 0.0, 80.0])
    )
    np.testing.assert_array_equal(result, np.array([25.0, 75.0, 65.0]))


def test_r12_adapter_and_generic_verifier_run_synthetic_two_fold_pilot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _synthetic_frame()
    target = frame["threat_rank"].to_numpy(dtype=np.float64)
    cycles = frame["cycle_code"].to_numpy(dtype=np.int64)
    views = {
        "W1": [(55, 56, 59)],
        "W4": [(67, 68, 71)],
        "primary": [(55, 56, 59), (67, 68, 71)],
        "H12": [(55, 56, 67), (67, 68, 79)],
    }
    controls_by_cutoff = {}
    for cutoff in (55, 67):
        rows = np.flatnonzero((cycles > cutoff) & (cycles <= cutoff + 12))
        controls_by_cutoff[cutoff] = _controls(rows, cycles[rows], per_cycle=15)

    _FakeRanker.instances.clear()
    monkeypatch.setattr(r12, "LGBMRanker", _FakeRanker)
    source = tmp_path / "synthetic_r12_source.py"
    train_path = tmp_path / "synthetic_r12_train.csv"
    source.write_text("synthetic R12 source identity\n", encoding="utf-8")
    frame.to_csv(train_path, index=False)
    card = {"id": "R12", "task": "regression", "pilot_predictor_fits": 2}
    identity = make_freeze_identity(
        card=card,
        feature_names=FEATURES,
        input_files={"synthetic_train": train_path},
        source_files=[source],
        evaluation_views=views,
        packages=[],
    )
    run_dir = tmp_path / "runs"
    ledger = tmp_path / "ledger.jsonl"
    freeze_path = tmp_path / "freeze.json"
    report = run_pilot(
        card=card,
        artifact_dir=run_dir,
        budget_ledger_path=ledger,
        max_successful_fits=180,
        frame=frame,
        target=target,
        feature_names=FEATURES,
        adapter=r12.fit_predict,
        control_loader=lambda cutoff, rows: controls_by_cutoff[cutoff],
        freeze_identity=identity,
        freeze_path=freeze_path,
        event_path=ledger,
        evaluation_views=views,
        threads=2,
        report_path=tmp_path / "pilot.json",
    )
    verification = verify_report(
        report=report,
        frame=frame,
        target=target,
        artifact_dir=run_dir,
        freeze_signature=json.loads(freeze_path.read_text(encoding="utf-8"))["signature"],
        evaluation_views=views,
        prediction_reconstructors={"R12": r12.reconstruct_r12},
    )

    assert report["fit_count_matches_card"] is True
    assert report["fit_predictor_count"] == 2
    assert verification["status"] == "PASS"
    assert len(_FakeRanker.instances) == 2
    assert _FakeRanker.instances[0].fit_group == [15] * 9
    assert _FakeRanker.instances[1].fit_group == [15] * 21
    for cutoff in (55, 67):
        bundle = np.load(run_dir / f"R12_fit{cutoff}.npz", allow_pickle=False)
        assert bundle["score_rows"].shape[0] == 12 * 15
        assert np.isfinite(bundle["prediction"]).all()
        assert np.all((bundle["prediction"] >= 0.0) & (bundle["prediction"] <= 100.0))


def test_r12_blend_rejects_misaligned_or_out_of_range_inputs() -> None:
    with pytest.raises(r12.R12IntegrityError, match="aligned"):
        r12.blend_rank_percentile(np.array([30.0]), np.array([20.0, 40.0]))
    with pytest.raises(r12.R12IntegrityError, match=r"within \[0,100\]"):
        r12.blend_rank_percentile(np.array([30.0]), np.array([120.0]))
