import numpy as np
import pandas as pd
import pytest

from src.models import optimization_v3_batch1_r14 as r14


def _frame(entry_friction, telemetry=None):
    n = len(entry_friction)
    values = {}
    for name in r14.FEATURE_NAMES:
        if name == "entry_friction":
            values[name] = entry_friction
        elif name == "telemetry_flag":
            values[name] = telemetry if telemetry is not None else [0] * n
        elif name in {"org_code", "module_code", "pattern_code"}:
            values[name] = [f"{name}_{i % 3}" for i in range(n)]
        elif name == "signal_stream":
            values[name] = ["token alpha" if i % 4 else "" for i in range(n)]
        else:
            values[name] = np.arange(n, dtype=float) + 1.0
    return pd.DataFrame(values, columns=r14.FEATURE_NAMES)


def test_ef1_uses_parsed_entry_friction_and_not_telemetry():
    frame = _frame(["1", "0", None, "1.0", "unknown"], [0, 1, 1, 0, 1])
    assert r14.ef1_mask(frame["entry_friction"]).tolist() == [True, False, False, True, False]


def test_fixed_route_blends_only_ef1_and_clips_raw_expert():
    base = np.array([10.0, 20.0, 30.0, 40.0])
    raw = np.array([-10.0, 110.0, 55.0, 9.0])
    mask = np.array([True, True, False, False])
    result = r14.compose_r14_prediction(raw, base, mask)
    assert np.array_equal(result[2:], base[2:])
    assert np.allclose(result[:2], [7.5, 40.0])


def test_fit_predict_subsets_training_to_ef1_and_routes_to_ef1(monkeypatch):
    seen = {}

    class FakeSparseModel:
        def fit(self, X, y):
            seen["X"] = X.copy()
            seen["y"] = np.asarray(y).copy()
            return self

        def predict(self, X):
            return np.full(len(X), 120.0)

        def audit_state(self):
            return {"fit_rows": len(seen["X"]), "objective": "regression_l1"}

    monkeypatch.setattr(r14, "build_sparse_hybrid", lambda recipe: FakeSparseModel())
    fit_x = _frame([1] * 520 + [0] * 80, [0] * 600)
    y = np.linspace(0.0, 100.0, len(fit_x))
    score_x = _frame([1, "0", 1, None], [0, 1, 0, 1])
    incumbent = np.array([20.0, 30.0, 40.0, 50.0])
    result = r14.fit_predict(
        fit_x, y, score_x, controls={"baseline_prediction": incumbent},
        card_id="R14", cutoff=55,
    )
    assert len(seen["X"]) == 520
    assert r14.ef1_mask(seen["X"]["entry_friction"]).all()
    assert len(seen["y"]) == 520
    assert result["fit_metadata"]["ef1_fit_rows"] == 520
    assert result["audit_arrays"]["expert_applied_mask"].tolist() == [1, 0, 1, 0]
    assert np.array_equal(result["prediction"][[1, 3]], incumbent[[1, 3]])
    assert np.allclose(result["prediction"][[0, 2]], [40.0, 55.0])


def test_fit_predict_fails_closed_below_ef1_support(monkeypatch):
    called = False

    def forbidden_builder(_recipe):
        nonlocal called
        called = True
        raise AssertionError("model must not be constructed before support passes")

    monkeypatch.setattr(r14, "build_sparse_hybrid", forbidden_builder)
    fit_x = _frame([1] * 499 + [0] * 101)
    score_x = _frame([1] * 100)
    with pytest.raises(r14.R14IntegrityError, match="at least 500 EF1 fit rows"):
        r14.fit_predict(
            fit_x, np.ones(len(fit_x)), score_x,
            controls={"baseline_prediction": np.full(len(score_x), 50.0)},
            card_id="R14", cutoff=55,
        )
    assert not called
