from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import scripts.optimization_v3_research_next_n01 as runner
import src.models.research_next_n01 as adapter
from src.models.classification_models import _text_values
from src.models.research_next_budget import (
    DEFAULT_CAPS,
    authorize_new_phase,
    read_events,
    reserve_card,
)
from src.models.research_next_contract import read_bundle


def _synthetic_features(n: int, *, cycle_values: np.ndarray, offset: int = 0) -> pd.DataFrame:
    index = np.arange(offset, offset + n)
    present = index % 10 != 0
    frame = pd.DataFrame(
        {
            "impact_load": (index % 7 + 1).astype(float),
            "entry_friction": (index % 5 + 1).astype(float),
            "cycle_code": cycle_values.astype(np.int64),
            "scope_pressure": (index % 9 + 1).astype(float),
            "text_load": np.where(index % 20 == 0, np.nan, (index % 4).astype(float)),
            "scope_trace": [f"{value % 8 + 1} scopes" for value in index],
            "org_spread": [f"~{value % 5 + 1}" for value in index],
            "module_spread": [(value % 6 + 1) for value in index],
            "telemetry_flag": np.where(index % 2 == 0, "active", "inactive"),
            "org_code": np.where(index % 3 == 0, "org_alpha", "org_beta"),
            "module_code": np.where(index % 4 == 0, "module_alpha", "module_beta"),
            "pattern_code": np.where(index % 5 == 0, "pattern_alpha", "pattern_beta"),
            "signal_stream": [
                (f"common token{value % 5} {'artifact' if value % 2 else 'clean'}") if keep else None
                for value, keep in zip(index, present)
            ],
        },
        index=index,
    )
    return frame.loc[:, list(adapter.FEATURE_NAMES)]


def test_weight_plan_uses_fit_bins_and_matches_class_mass():
    n = 240
    frame = _synthetic_features(n, cycle_values=np.repeat(np.arange(47, 56), 27)[:n])
    target = np.arange(n, dtype=np.int8) % 2
    np.random.default_rng(17).shuffle(target)

    plan = adapter.class_by_volume_weight_plan(frame, target, minimum_cell_n=2)

    candidate = plan["candidate_weights"]
    class_only = plan["class_only_weights"]
    assert plan["diagnostics"]["missing_bin_present_in_fit"] is True
    assert plan["diagnostics"]["score_rows_used_for_edges_or_weights"] is False
    assert np.isfinite(candidate).all() and np.all(candidate > 0)
    assert np.isclose(candidate.mean(), 1.0)
    assert np.isclose(class_only.mean(), 1.0)
    for label in (0, 1):
        assert np.isclose(candidate[target == label].sum(), class_only[target == label].sum())
        assert np.unique(class_only[target == label]).size == 1

    with pytest.raises(adapter.WeightSupportError, match="support below"):
        adapter.class_by_volume_weight_plan(frame, target, minimum_cell_n=1000)


def test_n01_synthetic_fit_bundle_resume_metrics_and_views(tmp_path, monkeypatch):
    fit_n, score_n = 135, 80
    fit_cycles = np.repeat(np.arange(47, 56), 15)
    score_cycles = np.repeat(np.arange(56, 60), 20)
    frame = pd.concat(
        [
            _synthetic_features(fit_n, cycle_values=fit_cycles),
            _synthetic_features(score_n, cycle_values=score_cycles, offset=fit_n),
        ]
    )
    target = np.arange(len(frame), dtype=np.int8) % 2
    np.random.default_rng(29).shuffle(target)
    fit_rows = np.arange(fit_n, dtype=np.int64)
    score_rows = np.arange(fit_n, fit_n + score_n, dtype=np.int64)
    plan = adapter.class_by_volume_weight_plan(
        frame.iloc[fit_rows], target[fit_rows], minimum_cell_n=2
    )
    _docs, stream_missing = _text_values(frame.iloc[score_rows]["signal_stream"])
    c2b = np.linspace(0.12, 0.88, score_n, dtype=np.float64)
    cb = np.linspace(0.72, 0.28, score_n, dtype=np.float64)
    sparse_raw = np.linspace(0.20, 0.80, score_n, dtype=np.float64)
    current = np.where(stream_missing, cb, 0.75 * c2b + 0.25 * sparse_raw)
    control = {
        "rows": score_rows.copy(),
        "cycles": score_cycles.copy(),
        "record_ids": np.asarray([f"SYN-{row:04d}" for row in score_rows]),
        "c2b_probability": c2b,
        "cb_probability": cb,
        "sparse_baseline_raw": sparse_raw,
        "current_prediction": current,
        "stream_missing": stream_missing.astype(bool),
    }
    context = {
        "frame": frame,
        "target": target,
        "freeze_signature": "synthetic-freeze",
        "fit_rows": {55: fit_rows},
        "score_rows": {55: score_rows},
        "plans": {55: plan},
        "controls": {55: control},
    }

    ledger = tmp_path / "budget.jsonl"
    authorize_new_phase(
        ledger,
        cap=4,
        category_caps={"pilot": 4, "continuation": 0, "control_replay": 0, "technical_retry": 0},
        inherited_old_budget_snapshot={"sha256": "synthetic", "effective_total": 0},
    )
    reserve_card(ledger, card_id="N01", stage="PILOT", fits=4, category="pilot")
    monkeypatch.setattr(runner, "LEDGER_PATH", ledger)
    monkeypatch.setattr(runner, "BUNDLE_ROOT", tmp_path / "bundles")
    monkeypatch.setattr(runner, "REPORT_ROOT", tmp_path / "reports")
    monkeypatch.setattr(adapter, "PARAMS", {**adapter.PARAMS, "n_estimators": 3})

    actual_estimator = adapter.LGBMClassifier
    calls: list[np.ndarray] = []

    class RecordingEstimator:
        def __init__(self, **kwargs):
            self.inner = actual_estimator(**kwargs)

        def fit(self, X, y, *, sample_weight=None):
            assert sample_weight is not None
            calls.append(np.asarray(sample_weight, dtype=np.float64).copy())
            self.inner.fit(X, y, sample_weight=sample_weight)
            return self

        def predict_proba(self, X):
            return self.inner.predict_proba(X)

        def __getattr__(self, name):
            return getattr(self.inner, name)

    monkeypatch.setattr(adapter, "LGBMClassifier", RecordingEstimator)

    publish = runner.resume_or_publish_bundle
    failed_once = False

    def fail_after_atomic_publish(*args, **kwargs):
        nonlocal failed_once
        bundle = publish(*args, **kwargs)
        if not failed_once and kwargs.get("task") == "classification":
            failed_once = True
            raise OSError("synthetic post-publish interruption")
        return bundle

    monkeypatch.setattr(runner, "resume_or_publish_bundle", fail_after_atomic_publish)
    with pytest.raises(OSError, match="synthetic post-publish interruption"):
        runner._fit_one(context, 55, "class_by_volume")
    events = read_events(ledger)
    assert sum(event["event"] == "FIT_STARTED" for event in events) == 1
    assert not any(event["event"] in {"FIT_SAVED", "FIT_FAILED"} for event in events)
    assert len(calls) == 1

    monkeypatch.setattr(runner, "resume_or_publish_bundle", publish)
    candidate_arrays, candidate_receipt = runner._fit_one(context, 55, "class_by_volume")
    assert len(calls) == 1  # the exact saved bundle resumed; the estimator was not refit
    assert candidate_receipt["metadata"]["metadata"]["fit_state"]["sample_weight_used"] is True
    assert np.array_equal(candidate_arrays["score_rows"], score_rows)
    candidate_path = tmp_path / "bundles" / "pilot" / "cutoff_55" / "class_by_volume"
    candidate_bundle = read_bundle(
        candidate_path,
        runner._bundle_identity("synthetic-freeze", 55, "class_by_volume"),
        task="classification",
    )
    assert runner._independent_reconstruct(candidate_bundle.arrays, fit_kind="class_by_volume").shape == (score_n,)

    matched_arrays, matched_receipt = runner._fit_one(context, 55, "class_only")
    assert len(calls) == 2
    assert matched_receipt["metadata"]["metadata"]["fit_state"]["sample_weight_used"] is True
    merged = runner._merge_origin(
        context,
        55,
        {"class_by_volume": candidate_arrays, "class_only": matched_arrays},
    )
    assert np.isfinite(merged["candidate"]).all()
    assert np.isfinite(merged["matched"]).all()

    views = {
        "primary": [[55, 56, 57], [55, 58, 59]],
        "h12": [[55, 56, 59]],
    }
    gates = {
        "primary_gain_min": -1.0,
        "h12_harm_max": 2.0,
        "individual_primary_window_harm_max": 2.0,
        "subgroup_f1_harm_max": 2.0,
        "positive_recall_harm_max": 2.0,
        "minimum_group_n": 200,
        "minimum_each_class_n": 1,
        "matched_control_primary_gain_min": -1.0,
    }
    result, gate, groups, per_cycle = runner.evaluate_pilot(
        frame, target, {55: merged}, views=views, gate=gates
    )
    assert set(result["views"]) == {"primary", "H12", "W1", "W4"}
    assert result["views"]["primary"]["n_unique_rows"] == 80
    assert result["views"]["H12"]["n_unique_rows"] == 80
    assert len(groups) > 0 and len(per_cycle) > 0
    assert set(gate["checks"]) == {
        "primary_gain_at_least_0_0005",
        "H12_harm_at_most_0_001",
        "W1_harm_at_most_0_005",
        "W4_harm_at_most_0_005",
        "primary_gain_over_class_only_at_least_0_0002",
        "supported_primary_subgroup_guards",
    }
    runner._write_once(tmp_path / "reports" / "synthetic_report.json", result)
    assert (tmp_path / "reports" / "synthetic_report.json").is_file()
