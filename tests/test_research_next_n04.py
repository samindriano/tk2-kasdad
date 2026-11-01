from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

import scripts.optimization_v3_research_next_n04 as runner
import src.models.research_next_n04 as adapter
from src.models.research_next_budget import (
    authorize_new_phase, ledger_summary, reserve_card, start_fit,
)
from src.models.research_next_contract import canonical_json_bytes, publish_bundle, read_bundle
from src.models.regression_research_text import (
    HYBRID_FEATURES, RegressionTextModel, _SVD_PARAMS, _text_documents,
)


def _domain_frame(n: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(15)
    return pd.DataFrame({
        "impact_load": rng.integers(1, 8, n).astype(float),
        "entry_friction": rng.integers(0, 5, n).astype(float),
        "scope_pressure": rng.integers(0, 5, n).astype(float),
        "text_load": rng.integers(1, 30, n).astype(float),
        "scope_trace": [f"{int(v)} endpoints" for v in rng.integers(1, 8, n)],
        "org_spread": [f"{int(v)} orgs" for v in rng.integers(1, 7, n)],
        "module_spread": [f"{int(v)} modules" for v in rng.integers(1, 7, n)],
        "telemetry_flag": np.where(np.arange(n) % 3, "active", "inactive"),
        "org_code": np.where(np.arange(n) % 4, "org_a", "org_b"),
        "module_code": np.where(np.arange(n) % 5, "mod_a", "mod_b"),
        "pattern_code": np.where(np.arange(n) % 6, "pat_a", "pat_b"),
        "signal_stream": [
            "alpha beta repeated token" if i % 2 else "gamma delta repeated token"
            for i in range(n)
        ],
    })


def _r22_frame(n: int, start_row: int = 0, cycle: int = 55) -> pd.DataFrame:
    values = np.arange(start_row, start_row + n)
    return pd.DataFrame({
        "impact_load": (values % 8 + 1).astype(float),
        "entry_friction": (values % 5).astype(float),
        "cycle_code": np.full(n, cycle, dtype=np.int64),
        "scope_pressure": (values % 6).astype(float),
        "text_load": (values % 30 + 1).astype(float),
        "scope_trace": [f"{v % 7 + 1} endpoints" for v in values],
        "org_spread": [f"{v % 5 + 1} orgs" for v in values],
        "module_spread": [f"{v % 6 + 1} modules" for v in values],
        "telemetry_flag": np.where(values % 3, "active", "inactive"),
        "org_code": np.where(values % 4, "org_a", "org_b"),
        "module_code": np.where(values % 5, "mod_a", "mod_b"),
        "pattern_code": np.where(values % 6, "pat_a", "pat_b"),
        "signal_stream": [
            "alpha beta stable token" if v % 2 else "gamma delta stable token"
            for v in values
        ],
    }, columns=list(HYBRID_FEATURES))


def test_domain_builder_excludes_cycle_id_and_targets_and_is_fold_fitted():
    frame = _domain_frame()
    labels = np.tile(np.array([0, 1], dtype=np.int8), len(frame) // 2)
    builder = adapter.DomainFeatureBuilder()
    x_fit = builder.fit_transform(frame)
    x_score = builder.transform(frame.iloc[:20].copy())
    audit = builder.audit_state()
    assert sparse.isspmatrix_csr(x_fit)
    assert x_fit.shape[0] == len(frame)
    assert x_score.shape[0] == 20
    assert audit["cycle_code_used"] is False
    assert audit["record_id_used"] is False
    assert audit["task_target_used"] is False
    assert "cycle_code" not in audit["domain_feature_names"]
    assert "record_id" not in audit["domain_feature_names"]
    assert "threat_rank" not in audit["domain_feature_names"]
    assert canonical_json_bytes(audit)
    model = adapter.LogisticRegression(**adapter.DOMAIN_LR_PARAMETERS)
    model.fit(x_fit, labels)
    assert model.predict_proba(x_score).shape == (20, 2)
    forbidden = frame.assign(cycle_code=55, record_id="rid", threat_rank=0, artifact_signal=0)
    with pytest.raises(ValueError, match="exclude ID, cycle"):
        builder.transform(forbidden)


def test_last_four_cycle_domain_and_balanced_prior_odds_weights():
    cycles = np.repeat(np.arange(47, 53), 2)
    labels = adapter.domain_labels(cycles, cutoff=52)
    np.testing.assert_array_equal(labels, np.repeat([0, 0, 1, 1, 1, 1], 2))
    probabilities = np.array([.2, .8, .3, .7])
    domains = np.array([0, 0, 1, 1], dtype=np.int8)
    weights, diag = adapter.density_ratio_weights(probabilities, domains)
    raw_source_odds = np.array([.2 / .8, .8 / .2])
    expected = np.array([*raw_source_odds, 1.0, 1.0])
    expected /= expected.mean()
    np.testing.assert_allclose(weights, expected, rtol=0, atol=1e-14)
    assert diag["effective_sample_size_fraction"] == pytest.approx(
        weights.sum() ** 2 / (np.square(weights).sum() * len(weights))
    )
    assert diag["all_weights_finite_positive"]
    assert diag["weight_mean"] == pytest.approx(1.0)


def test_independent_domain_labels_are_anchored_to_full_fit_prefix():
    cycles = np.repeat(np.arange(47, 56), 3).astype(np.int64)
    frame = pd.DataFrame({"record_id": [f"rid-{i}" for i in range(len(cycles))],
                          "cycle_code": cycles})
    fit_rows = np.flatnonzero(cycles <= 55)
    labels = adapter.domain_labels(cycles[fit_rows], 55)
    held = cycles[fit_rows] % 2 == 0
    rows = fit_rows[held]
    held_labels = labels[held]
    probabilities = np.where(held_labels == 1, .8, .2)
    auc = runner._independent_auc(held_labels, probabilities)
    arrays = {
        "score_rows": rows,
        "score_record_ids": frame.record_id.to_numpy(dtype=str)[rows],
        "prediction": probabilities.astype(float),
        "baseline_prediction": held_labels.astype(float),
        "domain_label": held_labels.astype(np.int8),
        "cycle_code": cycles[rows],
    }
    bundle = SimpleNamespace(
        arrays=arrays,
        receipt={"metadata": {"diagnostics": {"independent_heldout_auc": auc}}},
    )
    checked = runner._independent_domain(bundle, frame, cutoff=55, parity=0)
    np.testing.assert_array_equal(checked["labels"], held_labels)
    assert checked["auc"] == pytest.approx(1.0)


def test_within_cycle_shuffle_preserves_each_exact_weight_multiset():
    weights = np.array([.4, .7, 1.0, 1.2, 1.8, 2.1, .5, 1.4, 1.5])
    cycles = np.array([55, 55, 55, 56, 56, 56, 57, 57, 57])
    shuffled = adapter.shuffle_weights_within_cycle(weights, cycles, seed=97)
    assert not np.array_equal(weights, shuffled)
    for cycle in np.unique(cycles):
        np.testing.assert_array_equal(np.sort(weights[cycles == cycle]),
                                      np.sort(shuffled[cycles == cycle]))


def test_weighted_task_heads_receive_same_r22_transform_and_only_weights_change(monkeypatch):
    class _Booster:
        def save_raw(self, raw_format=None):
            assert raw_format == "json"
            return b"synthetic-model"

    class _RecordingXGB:
        calls = []

        def __init__(self, **parameters):
            self.parameters = parameters

        def fit(self, x, y, sample_weight=None, verbose=None):
            assert verbose is False
            self.x = np.asarray(x).copy()
            self.y = np.asarray(y).copy()
            self.sample_weight = np.asarray(sample_weight).copy()
            self.calls.append((self.x.copy(), self.y.copy(), self.sample_weight.copy()))
            self.feature_importances_ = np.zeros(self.x.shape[1], dtype=np.float64)
            self.mean = float(np.average(self.y, weights=self.sample_weight))
            return self

        def predict(self, x):
            return np.full(len(x), self.mean, dtype=np.float64)

        def get_booster(self):
            return _Booster()

    monkeypatch.setattr(adapter, "XGBRegressor", _RecordingXGB)
    train = _r22_frame(160, cycle=55)
    score = _r22_frame(24, start_row=500, cycle=56)
    design = adapter.R22WeightedDesign(cutoff=55)
    x_fit = design.fit_transform(train)
    x_score = design.transform(score)
    docs, missing = _text_documents(train["signal_stream"])
    vectorizer, text_matrix, _ = RegressionTextModel._fit_vectorizer(docs)
    svd = TruncatedSVD(n_components=min(32, text_matrix.shape[1] - 1),
                       **dict(_SVD_PARAMS)) if text_matrix.shape[1] > 1 else None
    text_dense = svd.fit_transform(text_matrix) if svd is not None else np.empty((len(train), 0))
    structured, _, _, _, _, _ = RegressionTextModel._structured_fit(train)
    reference_fit = np.column_stack((structured.toarray(), text_dense, missing.astype(float)))
    np.testing.assert_array_equal(x_fit, reference_fit)
    score_structured, _ = RegressionTextModel._structured_transform(
        score, design._processor, design._scaler, design._encoder, design._numeric_positions,
    )
    score_docs, score_missing = _text_documents(score["signal_stream"])
    score_text = vectorizer.transform(score_docs) if vectorizer is not None else sparse.csr_matrix((len(score), 0))
    reference_score_text = svd.transform(score_text) if svd is not None else np.empty((len(score), 0))
    reference_score = np.column_stack((score_structured.toarray(), reference_score_text,
                                       score_missing.astype(float)))
    np.testing.assert_array_equal(x_score, reference_score)
    target = np.linspace(0, 100, len(train))
    weights_a = np.ones(len(train))
    weights_b = np.linspace(.5, 1.5, len(train))
    weights_b /= weights_b.mean()
    model_a, state_a = adapter.fit_weighted_r22_head(x_fit, target, weights_a, cutoff=55)
    model_b, state_b = adapter.fit_weighted_r22_head(x_fit, target, weights_b, cutoff=55)
    np.testing.assert_array_equal(_RecordingXGB.calls[0][0], _RecordingXGB.calls[1][0])
    np.testing.assert_array_equal(_RecordingXGB.calls[0][1], _RecordingXGB.calls[1][1])
    np.testing.assert_array_equal(_RecordingXGB.calls[0][2], weights_a)
    np.testing.assert_array_equal(_RecordingXGB.calls[1][2], weights_b)
    assert state_a["fit_design_sha256"] == state_b["fit_design_sha256"]
    assert state_a["sample_weight_sha256"] != state_b["sample_weight_sha256"]
    assert adapter.predict_r22_head(model_a, x_score).shape == (24,)
    assert adapter.predict_r22_head(model_b, x_score).shape == (24,)


def _synthetic_task_bundle(tmp_path, monkeypatch):
    n = 600
    rows = np.arange(n, dtype=np.int64)
    cycles = np.full(n, 56, dtype=np.int64)
    ids = np.array([f"δ-row-{i:04d}" for i in range(n)], dtype=str)
    frame = pd.DataFrame({
        "record_id": ids, "cycle_code": cycles,
        "signal_stream": np.where(np.arange(n) % 8 == 0, None, "alpha beta gamma"),
    })
    rng = np.random.default_rng(31)
    raw = rng.normal(48, 15, n)
    legacy = np.clip(rng.normal(50, 12, n), 0, 100)
    r22 = adapter.compose_r22_candidate(raw, legacy, cycles)
    r06raw = raw + rng.normal(0, 4, n)
    r06 = adapter.compose_r22_candidate(r06raw, legacy, cycles)
    missing = frame.signal_stream.isna().to_numpy()
    target = np.concatenate([np.zeros(150), np.full(150, 100), rng.uniform(15, 85, 300)])
    source = {"rows": rows, "ids": ids, "cycles": cycles, "legacy": legacy,
              "R22_ORIGINAL": r22, "R06_ORIGINAL": r06}
    identity = {"batch_id": "synthetic", "card_id": "N04", "freeze_signature": "abc",
                "stage": "PILOT", "cutoff": 55, "component": "weighted"}
    arrays = {
        "score_rows": rows, "score_record_ids": ids, "prediction": r22,
        "baseline_prediction": r22, "raw_component": raw,
        "rank50_component": runner._rank50_independent(raw, cycles),
        "legacy_champion": legacy, "r06_original": r06, "cycle_code": cycles,
        "stream_missing": missing,
    }
    payload = {**arrays, "diagnostics": {"synthetic_only": True}}
    path = tmp_path / "synthetic-bundle"
    monkeypatch.setattr(runner, "LEDGER_PATH", tmp_path / "budget.jsonl")
    return frame, target, source, identity, arrays, payload, path


def test_synthetic_readback_independent_metrics_guards_and_recovery_resume(tmp_path, monkeypatch):
    frame, target, source, identity, arrays, payload, path = _synthetic_task_bundle(tmp_path, monkeypatch)
    ledger = runner.LEDGER_PATH
    monkeypatch.setattr(runner, "RECOVERY_ROOT", tmp_path / "recovery")
    authorize_new_phase(
        ledger, cap=60,
        category_caps={"pilot": 26, "continuation": 24, "control_replay": 6, "technical_retry": 4},
        inherited_old_budget_snapshot={"sha256": "0" * 64, "consumed": 0},
    )
    reserve_card(ledger, card_id="N04", stage="PILOT", fits=1, category="pilot")
    fit_id = "N04:PILOT:55:weighted"
    start_fit(ledger, fit_id=fit_id, card_id="N04", stage="PILOT",
              cutoff=55, component="synthetic_weighted")
    recovery_payload = {"arrays": arrays, "metadata": {"diagnostics": {"synthetic_only": True}}}
    runner._save_recovery(fit_id, identity, recovery_payload)
    original_publish = runner.publish_bundle
    attempts = {"n": 0}

    def fail_once(*args, **kwargs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise OSError("synthetic persistence fault")
        return original_publish(*args, **kwargs)

    monkeypatch.setattr(runner, "publish_bundle", fail_once)
    verifier = lambda bundle: runner._independent_task(bundle, frame, source, 55, "weighted")
    with pytest.raises(OSError, match="synthetic persistence fault"):
        runner._publish_recovery(fit_id, identity, path, "regression", verifier)
    assert ledger_summary(ledger)["started_and_consumed"] == 1
    monkeypatch.setattr(runner, "publish_bundle", original_publish)
    saved = runner._publish_recovery(fit_id, identity, path, "regression", verifier)
    assert saved is not None
    reread = read_bundle(path, identity, task="regression")
    rebuilt = runner._independent_task(reread, frame, source, 55, "weighted")
    assert np.isfinite(runner._mae(target, rebuilt["prediction"]))
    assert reread.receipt["metadata"]["diagnostics"]["synthetic_only"] is True
    view_context = {"frame": frame, "target": target, "sources": {55: source}}
    report = runner._view_metrics(view_context, [reread], {55: (56, 56)}, "synthetic")
    assert report["n"] == len(target)
    assert report["groups"]["observed_target_le5_evaluation_only"]["supported"]
    assert report["groups"]["observed_target_ge95_evaluation_only"]["supported"]
    assert np.isfinite(report["groups"]["stream_missing"]["candidate_mae"])
    events = runner.read_events(ledger)
    starts = [event for event in events if event.get("event") == "FIT_STARTED"]
    saved_events = [event for event in events if event.get("event") == "FIT_SAVED"]
    assert len(starts) == len(saved_events) == 1
    assert ledger_summary(ledger)["failed_or_interrupted_started"] == 0


def test_rank50_average_ties_and_candidate_policy():
    raw = np.array([10.0, 10.0, 30.0, 30.0])
    cycles = np.array([56, 56, 56, 56], dtype=np.int64)
    legacy = np.array([50.0, 40.0, 60.0, 70.0])
    candidate = adapter.compose_r22_candidate(raw, legacy, cycles)
    expected_rank = runner._rank50_independent(raw, cycles)
    np.testing.assert_allclose(candidate, np.clip(.75 * legacy + .25 * expected_rank, 0, 100),
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(adapter._complete_cycle_rank50(raw, cycles), expected_rank,
                               rtol=0, atol=1e-12)
