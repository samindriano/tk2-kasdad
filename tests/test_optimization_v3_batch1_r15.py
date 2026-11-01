from __future__ import annotations

import numpy as np
import pandas as pd

from scripts import optimization_v3_batch1_r15 as runner
from src.models import optimization_v3_batch1_r15 as adapter
from src.models.optimization_v3_batch1 import (
    evaluation_views_for_card,
    load_queue,
    temporal_rows,
)


def _features(cycles: list[int], streams: list[object], *, start: int = 0) -> pd.DataFrame:
    values: dict[str, object] = {}
    for feature in adapter.FEATURE_NAMES:
        if feature == "signal_stream":
            values[feature] = streams
        elif feature in {"org_code", "module_code", "pattern_code"}:
            values[feature] = [f"{feature}_{index % 3}" for index in range(len(cycles))]
        else:
            values[feature] = np.asarray(cycles, dtype=float)
    return pd.DataFrame(values, index=np.arange(start, start + len(cycles)))


def test_fixed_route_clips_raw_expert_and_keeps_present_rows_exact(monkeypatch):
    calls: dict[str, object] = {}

    class FakeModel:
        def __init__(self, *, params_id, features):
            calls["params_id"] = params_id
            calls["features"] = tuple(features)

        def fit(self, X, y):
            calls["fit_rows"] = len(X)
            calls["fit_columns"] = tuple(X.columns)
            calls["fit_y"] = np.asarray(y).copy()
            return self

        def predict(self, X):
            calls["score_rows"] = len(X)
            return np.array([-5.0, 105.0])

        def audit_state(self):
            return {"objective": "regression_l1", "features": list(self.feature_names)}

    monkeypatch.setattr(adapter, "RegressionResearchTabular", FakeModel)
    FakeModel.feature_names = adapter.STRUCTURED_FEATURE_NAMES
    fit = _features([47, 55, 67, 67], [None, "  ", "present", "token"])
    score = _features([68, 68, 69, 69], [None, "token", " ", "stream"], start=10)
    baseline = np.array([50.0, 40.0, 60.0, 25.0])
    controls = {
        "baseline_prediction": baseline,
        "stream_missing": np.array([True, False, True, False]),
        "control_rows": score.index.to_numpy(dtype=np.int64),
    }

    result = adapter.fit_predict(
        fit, np.array([10.0, 20.0, 30.0, 40.0]), score,
        controls=controls, card_id="R15", cutoff=67,
    )

    np.testing.assert_array_equal(result["raw_candidate"], np.array([-5.0, 0.0, 105.0, 0.0]))
    np.testing.assert_array_equal(result["prediction"], np.array([45.0, 40.0, 64.0, 25.0]))
    assert calls["params_id"] == "base"
    assert calls["fit_rows"] == 2
    assert calls["score_rows"] == 2
    assert "signal_stream" not in calls["fit_columns"]
    assert calls["fit_y"].tolist() == [10.0, 20.0]
    assert result["state"]["percentile_or_rank_mapping"] is False
    assert result["fit_metadata"]["validation_labels_passed_to_fit"] is False


def test_r15_freeze_card_is_exact_canonical_queue_entry():
    queue = load_queue(runner.QUEUE_PATH)
    card = next(item for item in queue["cards"] if item["id"] == "R15")
    runner._validate_card(queue, card)


def test_independent_reconstruction_rejects_misalignment_and_preserves_present():
    arrays = {
        "raw_candidate": np.array([0.0, -5.0, 70.0]),
        "baseline_prediction": np.array([20.0, 40.0, 90.0]),
        "control__stream_missing": np.array([True, False, True]),
    }
    candidate, baseline = adapter.reconstruct_r15_arrays(arrays)
    np.testing.assert_array_equal(candidate, np.array([18.0, 40.0, 88.0]))
    np.testing.assert_array_equal(baseline, arrays["baseline_prediction"])
    arrays["raw_candidate"] = np.array([1.0])
    try:
        adapter.reconstruct_r15_arrays(arrays)
    except ValueError as exc:
        assert "align" in str(exc)
    else:
        raise AssertionError("misaligned saved route must fail closed")


def test_r15_support_and_control_aggregation_use_late_views(monkeypatch):
    cycles = np.repeat(np.arange(47, 80, dtype=np.int64), 10)
    frame = pd.DataFrame({
        "record_id": [f"row-{index}" for index in range(len(cycles))],
        "cycle_code": cycles,
        "signal_stream": [None] * len(cycles),
    })
    queue = load_queue(runner.QUEUE_PATH)
    views = evaluation_views_for_card(queue, "R15")
    monkeypatch.setattr(runner, "MIN_FIT_MISSING_ROWS", 1)
    monkeypatch.setattr(runner, "MIN_VIEW_MISSING_ROWS", 1)
    monkeypatch.setattr(runner, "EXPECTED_VIEW_ROWS", {
        "W4": 40, "W6": 40, "primary": 80, "H12_L": 120, "H12_E": 120,
    })

    support, fit_rows, score_rows = runner._late_support(frame, cycles, views)

    assert support["status"] == "PASS"
    assert support["cutoffs"]["67"]["fit_missing_rows"] == len(fit_rows[67])
    assert support["cutoffs"]["75"]["fit_missing_rows"] == len(fit_rows[75])
    assert support["views"]["primary"]["rows"] == 80
    assert support["views"]["H12_L"]["missing_stream_rows"] == 120

    target = cycles.astype(float)
    controls = {}
    for cutoff, low, high in [(67, 68, 79), (75, 76, 79), (55, 56, 67)]:
        _, rows = temporal_rows(frame, cutoff, horizon=12)
        rows = rows[(cycles[rows] >= low) & (cycles[rows] <= high)]
        controls[cutoff] = {
            "rows": rows,
            "baseline_prediction": np.full(len(rows), 50.0),
        }
    metrics = runner._control_metrics(target, cycles, controls, views)
    assert metrics["W4"]["rows"] == 40
    assert metrics["W6"]["rows"] == 40
    assert metrics["primary"]["rows"] == 80
    assert metrics["H12_L"]["rows"] == 120
    assert metrics["H12_E"]["rows"] == 120
