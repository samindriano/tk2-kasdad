from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.preprocessing import normalize

import scripts.optimization_v3_research_next_n02 as runner
import src.models.research_next_n02 as adapter
from src.models.classification_models import ClassificationModel, _text_values
from src.models.research_next_budget import authorize_new_phase, read_events, reserve_card
from src.models.research_next_contract import read_bundle


def _features(cycles: np.ndarray, *, offset: int = 0, all_empty: bool = False) -> pd.DataFrame:
    n = len(cycles)
    row = np.arange(offset, offset + n)
    text = [
        None if all_empty or index % 12 == 0
        else f"token{index % 7} aspect{index % 4} signal{index % 3}"
        for index in row
    ]
    return pd.DataFrame(
        {
            "impact_load": (row % 7 + 1).astype(float),
            "entry_friction": (row % 2).astype(float),
            "cycle_code": cycles.astype(np.int64),
            "scope_pressure": (row % 9 + 1).astype(float),
            "text_load": (row % 6 + 1).astype(float),
            "scope_trace": [f"{value % 8 + 1} scopes" for value in row],
            "org_spread": [f"~{value % 5 + 1}" for value in row],
            "module_spread": [value % 6 + 1 for value in row],
            "telemetry_flag": np.where(row % 2 == 0, "active", "inactive"),
            "org_code": np.where(row % 3 == 0, "org_alpha", "org_beta"),
            "module_code": np.where(row % 4 == 0, "module_alpha", "module_beta"),
            "pattern_code": np.where(row % 5 == 0, "pattern_alpha", "pattern_beta"),
            "signal_stream": text,
        }
    )


def _balanced_target(n: int, seed: int) -> np.ndarray:
    result = np.arange(n, dtype=np.int8) % 2
    np.random.default_rng(seed).shuffle(result)
    return result


def test_no_sketch_is_c2b_equivalent_and_sketch_preserves_sparse_main_effects():
    n = 180
    frame = _features(np.repeat(np.arange(47, 56), 20))
    target = _balanced_target(n, 11)

    reference = ClassificationModel({"id": "C2_B"}).fit(frame, target)
    no_sketch = adapter.C2BInteractionClassifier(include_sketch=False).fit(frame, target)
    np.testing.assert_allclose(no_sketch.predict_probability(frame), reference.predict_proba(frame),
                               rtol=0, atol=1e-12)

    candidate = adapter.C2BInteractionClassifier(include_sketch=True).fit(frame, target)
    fit_main = candidate._fit_main_effects
    original = fit_main.copy()
    csr_sketch = candidate._sketch.transform(
        normalize(fit_main.copy(), norm="l2", axis=1, copy=False)
    )
    csc_sketch = candidate._sketch.transform(
        normalize(sparse.csc_matrix(fit_main), norm="l2", axis=1, copy=True)
    )
    assert np.asarray(csr_sketch).dtype == np.float64
    np.testing.assert_allclose(csr_sketch, csc_sketch, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(fit_main.data, original.data)

    score = _features(np.repeat(np.arange(56, 60), 5), offset=n)
    score["org_code"] = "unseen_org"
    score["module_code"] = "unseen_module"
    score["pattern_code"] = "unseen_pattern"
    transformed = candidate.transform_design(score)
    assert sparse.isspmatrix_csr(transformed)
    assert transformed.dtype == np.float64
    assert transformed.shape[1] == candidate.audit_state()["estimator_design"]["shape"][1]
    probability = candidate.predict_probability(score)
    assert probability.shape == (len(score),)
    assert np.isfinite(probability).all()


def test_all_empty_training_text_and_unseen_categories_are_supported():
    fit = _features(np.repeat(np.arange(47, 56), 12), all_empty=True)
    target = _balanced_target(len(fit), 23)
    model = adapter.C2BInteractionClassifier(include_sketch=True).fit(fit, target)
    state = model.audit_state()
    assert state["main_effect_state"]["text"]["vocabulary_size"] == 0
    assert state["main_effect_state"]["text"]["empty_all_training_text"] is True

    score = _features(np.repeat(np.arange(56, 60), 3), offset=len(fit), all_empty=True)
    score.loc[:, "org_code"] = "new_org"
    score.loc[:, "module_code"] = "new_module"
    score.loc[:, "pattern_code"] = "new_pattern"
    design = model.transform_design(score)
    probability = model.predict_probability(score)
    assert sparse.isspmatrix_csr(design)
    assert np.isfinite(design.data).all()
    assert np.isfinite(probability).all()


def test_n02_synthetic_bundle_resume_route_metrics_views_and_report(tmp_path, monkeypatch):
    fit_n, score_n = 135, 40
    fit_cycles = np.repeat(np.arange(47, 56), 15)
    score_cycles = np.repeat(np.arange(56, 60), 10)
    frame = pd.concat([
        _features(fit_cycles),
        _features(score_cycles, offset=fit_n),
    ], ignore_index=True)
    target = _balanced_target(fit_n + score_n, 41)
    fit_rows = np.arange(fit_n, dtype=np.int64)
    score_rows = np.arange(fit_n, fit_n + score_n, dtype=np.int64)
    _docs, missing = _text_values(frame.iloc[score_rows]["signal_stream"])
    current = np.linspace(0.18, 0.82, score_n, dtype=np.float64)
    cb = np.linspace(0.74, 0.26, score_n, dtype=np.float64)
    baseline = np.where(missing, cb, current)
    control = {
        "rows": score_rows.copy(), "cycles": score_cycles.copy(),
        "record_ids": np.asarray([f"SYN-{row:04d}" for row in score_rows]),
        "current_prediction": current, "cb_probability": cb,
        "stream_missing": np.asarray(missing, dtype=bool),
    }
    context = {
        "freeze_signature": "synthetic-n02-freeze", "frame": frame, "target": target,
        "feature_names": tuple(ClassificationModel({"id": "C2_B"}).input_feature_names),
        "fit_rows": {55: fit_rows}, "score_rows": {55: score_rows}, "controls": {55: control},
    }

    ledger = tmp_path / "budget.jsonl"
    authorize_new_phase(ledger, cap=4,
                        category_caps={"pilot": 4, "continuation": 0,
                                       "control_replay": 0, "technical_retry": 0},
                        inherited_old_budget_snapshot={"sha256": "synthetic", "effective_total": 0})
    reserve_card(ledger, card_id="N02", stage="PILOT", fits=4, category="pilot")
    monkeypatch.setattr(runner, "LEDGER_PATH", ledger)
    monkeypatch.setattr(runner, "BUNDLE_ROOT", tmp_path / "bundles")

    publish = runner.resume_or_publish_bundle
    injected = False

    def fail_once_after_bundle(*args, **kwargs):
        nonlocal injected
        bundle = publish(*args, **kwargs)
        if not injected:
            injected = True
            raise OSError("synthetic post-publish interruption")
        return bundle

    monkeypatch.setattr(runner, "resume_or_publish_bundle", fail_once_after_bundle)
    with pytest.raises(OSError, match="synthetic post-publish interruption"):
        runner._fit_one(context, 55, "sketch")
    events = read_events(ledger)
    assert sum(event["event"] == "FIT_STARTED" for event in events) == 1
    assert not any(event["event"] in {"FIT_SAVED", "FIT_FAILED"} for event in events)

    monkeypatch.setattr(runner, "resume_or_publish_bundle", publish)
    sketch_arrays, sketch_receipt = runner._fit_one(context, 55, "sketch")
    assert sum(event["event"] == "FIT_STARTED" for event in read_events(ledger)) == 1
    no_sketch_arrays, no_sketch_receipt = runner._fit_one(context, 55, "no_sketch")
    assert sketch_receipt["metadata"]["metadata"]["fit_state"]["include_sketch"] is True
    assert no_sketch_receipt["metadata"]["metadata"]["fit_state"]["include_sketch"] is False

    sketch_path = tmp_path / "bundles" / "pilot" / "cutoff_55" / "sketch"
    saved = read_bundle(sketch_path, runner._bundle_identity(context["freeze_signature"], 55, "sketch"),
                        task="classification")
    assert np.array_equal(saved.arrays["score_rows"], score_rows)
    assert runner._independent_reconstruct(saved.arrays, fit_kind="sketch").shape == (score_n,)
    merged = runner._merge_origin(context, 55,
                                  {"sketch": sketch_arrays, "no_sketch": no_sketch_arrays})

    views = {"primary": [[55, 56, 57], [55, 58, 59]], "h12": [[55, 56, 59]]}
    gate = {"primary_gain_min": -1.0, "h12_harm_max": 2.0,
            "individual_primary_window_harm_max": 2.0,
            "subgroup_f1_harm_max": 2.0, "positive_recall_harm_max": 2.0,
            "minimum_group_n": 200, "minimum_each_class_n": 1,
            "matched_control_primary_gain_min": -1.0}
    result, gate_result, groups, per_cycle = runner.evaluate_pilot(
        frame, target, {55: merged}, views=views, gate=gate)
    assert set(result["views"]) == {"primary", "H12", "W1", "W4"}
    assert result["views"]["primary"]["n_unique_rows"] == score_n
    assert groups and per_cycle
    assert gate_result["checks"]["primary_gain_over_no_sketch_at_least_0_0002"] in (True, False)
    report_path = tmp_path / "report.json"
    n01_writer = __import__("scripts.optimization_v3_research_next_n01", fromlist=["_write_once"])
    n01_writer._write_once(report_path, result)
    assert report_path.is_file()
    assert np.isfinite(sketch_arrays["prediction"]).all()
