from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.models import optimization_v3_batch1_r11 as r11
from src.models.optimization_v3_batch1 import write_bundle_exclusive
from src.models.regression_research_text import HYBRID_FEATURES
from src.models.round1_models import CATEGORICAL_FEATURES
from scripts import optimization_v3_batch1_r11 as runner


def _features(n: int, *, start: int = 0, future_text: bool = False) -> pd.DataFrame:
    rows = np.arange(start, start + n)
    values: dict[str, object] = {}
    for column in HYBRID_FEATURES:
        if column == "signal_stream":
            values[column] = [
                None if i % 37 == 0 else
                " " if i % 41 == 0 else
                "futuretoken qqqfutureonly" if future_text and i >= start + n - 4 else
                f"common token{i % 48} token{(i + 1) % 48} token{(i + 2) % 48}"
                for i in rows
            ]
        elif column in CATEGORICAL_FEATURES:
            values[column] = ["known" if i % 3 else "other" for i in rows]
        else:
            values[column] = (rows % 19).astype(float)
    return pd.DataFrame(values, index=rows).loc[:, list(HYBRID_FEATURES)]


class _FakeBooster:
    def __init__(self, digest: str) -> None:
        self.digest = digest

    def dump_model(self):
        return {"tree_info": [{"tree_structure": {"leaf_value": 1.0}}]}

    def model_to_string(self, **kwargs):
        return self.digest


class _FakeClassifier:
    instances: list["_FakeClassifier"] = []

    def __init__(self, **parameters):
        self.parameters = dict(parameters)
        self.booster_ = _FakeBooster(str(len(type(self).instances)))
        self.classes_ = np.array([0, 1])
        type(self).instances.append(self)

    def fit(self, frame, target, *, categorical_feature):
        self.fit_rows = len(frame)
        self.target = np.asarray(target, dtype=np.uint8).copy()
        self.categorical_feature = list(categorical_feature)
        self.positive_rate = float(self.target.mean())
        return self

    def predict_proba(self, frame):
        positive = np.full(len(frame), self.positive_rate, dtype=np.float64)
        return np.column_stack((1.0 - positive, positive))


def _target_with_supported_thresholds() -> np.ndarray:
    levels = np.repeat(np.array([1., 20., 30., 40., 50., 60., 70., 80., 90., 100.]), 180)
    return levels


def test_threshold_targets_use_strict_greater_than_and_support_both_classes():
    labels = r11.threshold_targets(np.array([10., 11., 20., 21., 90., 91.]))
    assert labels.shape == (6, 9)
    np.testing.assert_array_equal(labels[:, 0], [0, 1, 1, 1, 1, 1])
    np.testing.assert_array_equal(labels[:, 1], [0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(labels[:, -1], [0, 0, 0, 0, 0, 1])
    support = r11.threshold_support(_target_with_supported_thresholds())
    assert all(item["supported"] for item in support.values())
    assert min(item["positive"] for item in support.values()) >= 100
    assert min(item["negative"] for item in support.values()) >= 100
    with pytest.raises(ValueError, match="finite vector"):
        r11.threshold_targets([0., np.nan])
    np.testing.assert_array_equal(
        r11.validated_cycle_codes([47, 48.0, "79"], name="test cycle_code"),
        np.array([47, 48, 79], dtype=np.int64),
    )
    for invalid in ([47.5], [47.0, np.nan], ["cycle-x"]):
        with pytest.raises(ValueError, match="cycle_code"):
            r11.validated_cycle_codes(invalid, name="test cycle_code")
    for invalid in ([47.5], [47.0, np.nan]):
        with pytest.raises(ValueError):
            r11.rank50_prediction([50.0] * len(invalid), invalid)


def test_pava_projects_each_row_to_nonincreasing_probabilities():
    raw = np.array([
        [.90, .80, .70, .60, .50, .40, .30, .20, .10],
        [.10, .90, .20, .80, .30, .70, .40, .60, .50],
    ])
    projected = r11.pava_nonincreasing(raw)
    assert np.all(np.diff(projected, axis=1) <= 1e-12)
    np.testing.assert_allclose(projected[0], raw[0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(projected[1], [.5, .5, .5, .5, .5, .5, .5, .5, .5],
                               rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match="n-by-9"):
        r11.pava_nonincreasing(np.ones((3, 8)))


def test_median_crossing_interpolates_and_respects_zero_hundred_boundaries():
    values = np.full((4, 9), .1)
    values[0] = [.9, .8, .7, .6, .4, .3, .2, .1, .05]
    values[1] = .5
    values[2] = 1.0
    values[3] = 0.0
    medians = r11.median_from_survival(values)
    assert medians[0] == pytest.approx(45.0)
    assert medians[1] == pytest.approx(10.0)
    assert np.all((medians >= 0.0) & (medians <= 100.0))
    assert medians[2] == pytest.approx(95.0)
    assert medians[3] == pytest.approx(5.0)
    with pytest.raises(ValueError, match="nonincreasing"):
        r11.median_from_survival(np.tile([.1, .2, .3, .4, .5, .6, .7, .8, .9], (2, 1)))


def test_fit_uses_fit_only_text_transforms_nine_heads_and_aligned_score_rows(monkeypatch):
    _FakeClassifier.instances = []
    monkeypatch.setattr(r11, "LGBMClassifier", _FakeClassifier)
    fit_x = _features(1800)
    score_x = _features(24, start=2000, future_text=True)
    target = _target_with_supported_thresholds()
    model = r11.CumulativeOrdinalSVD32().fit(fit_x, target)
    state_before = model.audit_state()
    survival = model.predict_survival(score_x)
    prediction = r11.median_from_survival(survival)
    state_after = model.audit_state()

    assert len(_FakeClassifier.instances) == 9
    assert model.head_fit_calls == 9
    assert state_before["predictor_fit_calls"] == 9
    assert state_before["thresholds"] == list(range(10, 100, 10))
    assert state_before["design_shape"][0] == len(fit_x)
    assert state_before["svd_components_sha256"] == state_after["svd_components_sha256"]
    assert "futuretoken" not in model.vectorizer.vocabulary_
    assert survival.shape == (len(score_x), 9)
    assert np.all(np.diff(survival, axis=1) <= 1e-12)
    assert prediction.shape == (len(score_x),)
    assert np.isfinite(prediction).all()
    assert np.all((prediction >= 0.0) & (prediction <= 100.0))
    assert all(head.fit_rows == len(fit_x) for head in _FakeClassifier.instances)
    assert all(head.parameters == r11.HEAD_PARAMETERS for head in _FakeClassifier.instances)
    assert all(head.categorical_feature == model.categorical_positions
               for head in _FakeClassifier.instances)
    assert "record_id" not in r11.FEATURE_NAMES
    assert "threat_rank" not in r11.FEATURE_NAMES
    assert "artifact_signal" not in r11.FEATURE_NAMES
    with pytest.raises(ValueError, match="exactly the declared features"):
        model.predict(score_x.assign(record_id=[f"id-{i}" for i in range(len(score_x))]))


def test_model_rejects_empty_text_and_under_supported_thresholds(monkeypatch):
    _FakeClassifier.instances = []
    monkeypatch.setattr(r11, "LGBMClassifier", _FakeClassifier)
    too_small = _features(90)
    with pytest.raises(ValueError, match="support below 100"):
        r11.CumulativeOrdinalSVD32().fit(too_small, np.linspace(0.1, 100.0, len(too_small)))

    supported = _features(1800)
    supported["signal_stream"] = [None] * len(supported)
    with pytest.raises(ValueError, match="vocabulary wider than 32"):
        r11.CumulativeOrdinalSVD32().fit(supported, _target_with_supported_thresholds())
    assert not _FakeClassifier.instances
    with pytest.raises(ValueError, match="feature frames must be nonempty"):
        r11.CumulativeOrdinalSVD32().fit(supported.iloc[:0], np.empty(0))
    with pytest.raises(ValueError, match="strictly above 0"):
        r11.CumulativeOrdinalSVD32().fit(_features(1800), _target_with_supported_thresholds() * 0.0)


def test_fit_predict_counts_nine_heads_and_preserves_frozen_composition(monkeypatch):
    _FakeClassifier.instances = []
    monkeypatch.setattr(r11, "LGBMClassifier", _FakeClassifier)
    fit_x = _features(1800)
    fit_x["cycle_code"] = np.resize(np.arange(47, 56), len(fit_x))
    fit_y = _target_with_supported_thresholds()
    score_cycles = np.repeat([68, 69, 70], 8)
    score_x = _features(24, start=2000)
    score_x["cycle_code"] = score_cycles
    controls = {
        "baseline_prediction": np.linspace(0.0, 100.0, len(score_x)),
        "cycle_code": score_cycles,
    }
    result = r11.fit_predict(fit_x, fit_y, score_x, controls=controls, card_id="R11", cutoff=67)
    expected = r11.compose_r11_prediction(
        controls["baseline_prediction"], result["raw_candidate"], score_cycles
    )
    np.testing.assert_allclose(result["prediction"], expected, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(result["audit_arrays"]["cycle_code"], score_cycles)
    np.testing.assert_array_equal(
        result["audit_arrays"]["baseline_prediction"], controls["baseline_prediction"]
    )
    assert result["fit_predictor_count"] == 9
    assert result["fit_metadata"]["predictor_fit_calls"] == 9
    assert len(_FakeClassifier.instances) == 9
    assert np.all((result["prediction"] >= 0.0) & (result["prediction"] <= 100.0))
    with pytest.raises(ValueError, match="only card R11"):
        r11.fit_predict(fit_x, fit_y, score_x, controls=controls, card_id="R11", cutoff=60)


def test_fit_predict_rejects_fractional_or_missing_cycles_before_any_head_fit(monkeypatch):
    def forbidden_model():
        raise AssertionError("invalid cycle_code must fail before candidate head fitting")

    monkeypatch.setattr(r11, "CumulativeOrdinalSVD32", forbidden_model)
    fit_x = _features(1800)
    fit_x["cycle_code"] = np.resize(np.arange(47, 56), len(fit_x))
    score_cycles = np.repeat([68, 69, 70], 8)
    score_x = _features(24, start=2000)
    score_x["cycle_code"] = score_cycles
    fit_y = _target_with_supported_thresholds()
    controls = {
        "baseline_prediction": np.full(len(score_x), 50.0),
        "cycle_code": score_cycles.copy(),
    }
    invalid_cases = []
    fit_bad = fit_x.copy()
    fit_bad["cycle_code"] = fit_bad["cycle_code"].astype(float)
    fit_bad.iloc[0, fit_bad.columns.get_loc("cycle_code")] = 47.5
    invalid_cases.append((fit_bad, score_x, controls))
    score_bad = score_x.copy()
    score_bad.iloc[0, score_bad.columns.get_loc("cycle_code")] = np.nan
    invalid_cases.append((fit_x, score_bad, controls))
    control_bad = {**controls, "cycle_code": score_cycles.astype(float)}
    control_bad["cycle_code"][0] = 68.5
    invalid_cases.append((fit_x, score_x, control_bad))
    for bad_fit, bad_score, bad_controls in invalid_cases:
        with pytest.raises(ValueError, match="cycle_code"):
            r11.fit_predict(bad_fit, fit_y, bad_score, controls=bad_controls,
                            card_id="R11", cutoff=67)


def test_runner_recipe_is_json_safe_and_records_the_fixed_eighteen_head_recipe():
    recipe = runner.candidate_recipe()
    json_roundtrip = json.loads(json.dumps(recipe, allow_nan=False))
    assert json_roundtrip["predictor_fits"] == 18
    assert json_roundtrip["thresholds"] == list(range(10, 100, 10))
    assert json_roundtrip["head_parameters"]["objective"] == "binary"
    assert json_roundtrip["svd_parameters"]["n_components"] == 32
    assert "record_id" not in json_roundtrip["feature_allowlist"]


def test_runner_reconstructs_fixed_rank50_policy_from_frozen_controls():
    cycles = np.array([68, 68, 68, 68, 69, 69])
    baseline_sparse_raw = np.array([2.0, 2.0, 6.0, 8.0, 0.0, 10.0])
    ridge25 = np.array([20.0, 40.0, 60.0, 80.0, 25.0, 75.0])
    baseline = np.clip(
        0.75 * ridge25 + 0.25 * runner._independent_rank50(baseline_sparse_raw, cycles),
        0.0, 100.0,
    )
    raw = np.array([5.0, 5.0, 7.0, 9.0, -20.0, 120.0])
    arrays = {
        "raw_candidate": raw,
        "control__baseline_sparse_raw": baseline_sparse_raw,
        "control__ridge25_prediction": ridge25,
        "control__cycle_code": cycles,
        "baseline_prediction": baseline,
    }
    candidate, rebuilt_baseline = runner.reconstruct_r11(arrays)
    np.testing.assert_array_equal(rebuilt_baseline, baseline)
    expected = np.clip(0.75 * baseline + 0.25 * runner._independent_rank50(raw, cycles), 0, 100)
    np.testing.assert_array_equal(candidate, expected)
    arrays["baseline_prediction"] = baseline + 0.1
    with pytest.raises(runner.BatchIntegrityError, match="does not reconstruct"):
        runner.reconstruct_r11(arrays)


def test_runner_train_cycle_parser_rejects_noninteger_and_missing_values():
    np.testing.assert_array_equal(runner._validated_train_cycles([47, 55.0, "79"]), [47, 55, 79])
    for invalid in ([47.5], [47, np.nan], ["not-a-cycle"]):
        with pytest.raises(runner.BatchIntegrityError, match="cycle_code"):
            runner._validated_train_cycles(invalid)


@pytest.mark.parametrize("mismatch", [None, "signature", "identity"])
def test_frozen_runner_does_not_call_generic_runner_without_exact_freeze(tmp_path, monkeypatch, mismatch):
    called = []
    monkeypatch.setattr(runner, "run_pilot", lambda **kwargs: called.append(kwargs))
    freeze_path = tmp_path / "freeze.json"
    identity = {"freeze_revision": 2, "source_commit": "abc123", "input_sha256": {"train": "hash"}}
    if mismatch == "signature":
        freeze_path.write_text(json.dumps({"signature": "wrong", "identity": identity}), encoding="utf-8")
    elif mismatch == "identity":
        freeze_path.write_text(json.dumps({
            "signature": runner.json_sha256(identity),
            "identity": {"other": "value"},
        }), encoding="utf-8")
    kwargs = {
        "card": {},
        "frame": pd.DataFrame(),
        "target": np.empty(0),
        "controls": {},
        "freeze_identity": identity,
        "freeze_path": freeze_path,
        "artifact_dir": tmp_path / "artifacts",
        "budget_ledger_path": tmp_path / "ledger.jsonl",
        "event_path": tmp_path / "attempts.jsonl",
        "report_path": tmp_path / "pilot.json",
        "evaluation_views": {},
        "max_successful_fits": 18,
        "recipe_path": tmp_path / "recipe.json",
        "preflight_path": tmp_path / "preflight.json",
    }
    message = "requires an existing freeze" if mismatch is None else "do not exactly match"
    with pytest.raises(runner.BatchIntegrityError, match=message):
        runner.run_frozen_pilot(**kwargs)
    assert called == []


def test_frozen_artifacts_require_exact_recipe_and_preflight_contents(tmp_path):
    recipe = {"recipe": "fixed R11 synthetic"}
    preflight = {"status": "PREFLIGHT_PASS", "freeze_written": True}
    identity = {
        "freeze_revision": 2,
        "recipe_record_sha256": runner.json_sha256(recipe),
        "preflight_record_sha256": runner.json_sha256(preflight),
    }
    signature = runner.json_sha256(identity)
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text(json.dumps({"signature": signature, "identity": identity}), encoding="utf-8")
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(recipe), encoding="utf-8")
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text(json.dumps({**preflight, "freeze_identity_sha256": signature}), encoding="utf-8")
    assert runner._require_frozen_artifacts(
        freeze_path=freeze_path,
        freeze_identity=identity,
        recipe_path=recipe_path,
        preflight_path=preflight_path,
    ) == signature
    preflight_path.write_text(json.dumps({
        **preflight, "freeze_identity_sha256": signature, "altered": True,
    }), encoding="utf-8")
    with pytest.raises(runner.BatchIntegrityError, match="persisted preflight"):
        runner._require_frozen_artifacts(
            freeze_path=freeze_path,
            freeze_identity=identity,
            recipe_path=recipe_path,
            preflight_path=preflight_path,
        )


def test_revision_01_artifact_paths_remain_historical_and_unconsumed():
    assert runner.RECIPE_PATH != runner.RECIPE_REVISION_02_PATH
    assert runner.FREEZE_PATH != runner.FREEZE_REVISION_02_PATH
    assert runner.PREFLIGHT_PATH != runner.PREFLIGHT_REVISION_02_PATH
    assert runner.FREEZE_REVISION == 2


def test_frozen_runner_calls_shared_runner_once_after_revision_02_checks(tmp_path, monkeypatch):
    recipe = {"recipe": "fixed R11 revision 2"}
    preflight = {"freeze_revision": 2, "freeze_written": True}
    identity = {
        "freeze_revision": 2,
        "recipe_record_sha256": runner.json_sha256(recipe),
        "preflight_record_sha256": runner.json_sha256(preflight),
    }
    signature = runner.json_sha256(identity)
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text(json.dumps({"signature": signature, "identity": identity}), encoding="utf-8")
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(recipe), encoding="utf-8")
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text(json.dumps({
        **preflight, "freeze_identity_sha256": signature,
    }), encoding="utf-8")
    monkeypatch.setattr(runner, "_require_committed_sources", lambda: None)
    monkeypatch.setattr(runner, "_require_committed_artifacts", lambda paths: None)
    monkeypatch.setattr(runner, "_verify_live_freeze_hashes", lambda identity: None)
    monkeypatch.setattr(runner, "_check_runtime_memory", lambda path: {"available_ram_gib_at_pilot_start": 8.0})
    monkeypatch.setattr(runner, "append_event", lambda *args, **kwargs: None)
    calls = []
    monkeypatch.setattr(runner, "run_pilot", lambda **kwargs: calls.append(kwargs) or {"status": "stub"})
    result = runner.run_frozen_pilot(
        card={"id": "R11"}, frame=pd.DataFrame(), target=np.empty(0), controls={},
        freeze_identity=identity, freeze_path=freeze_path,
        artifact_dir=tmp_path / "artifacts", budget_ledger_path=tmp_path / "ledger.jsonl",
        event_path=tmp_path / "attempts.jsonl", report_path=tmp_path / "pilot.json",
        evaluation_views={}, max_successful_fits=18,
        recipe_path=recipe_path, preflight_path=preflight_path,
    )
    assert result == {"status": "stub"}
    assert len(calls) == 1
    assert calls[0]["card"] == {"id": "R11"}
    assert calls[0]["adapter"] is r11.fit_predict


def test_live_preflight_recheck_ignores_only_volatile_ram_observations():
    core = {
        "card_id": "R11", "status": "PREFLIGHT_PASS", "source_commit": "abc123",
        "queue_sha256": "q", "card_sha256": "c", "official_train_sha256": "t",
        "train_rows": 50_608, "fit_rows": {"55": 1, "67": 2}, "score_rows": {"55": 3, "67": 4},
        "views": runner.EXPECTED_VIEWS, "threshold_support_by_cutoff": {"55": {}, "67": {}},
        "target_boundary_audit": {"exact_zero_count_train": 0}, "control_metrics": {"H12": {"mae": 14.0}},
        "control_cache_sha256": {"a": "cache"}, "official_test_read": False,
        "candidate_head_fits_performed": 0,
    }
    stable_resource = {
        "status": "PASS", "available_ram_gib_observed": 8.0, "allowed_estimated_peak_gib": 6.0,
        "cutoffs": {"55": {"status": "PASS", "estimated_peak_gib": 2.5},
                    "67": {"status": "PASS", "estimated_peak_gib": 2.7}},
    }
    current_resource = json.loads(json.dumps(stable_resource))
    current_resource["available_ram_gib_observed"] = 5.5
    current_resource["allowed_estimated_peak_gib"] = 3.5
    context = {"preflight": {**core, "fit_budget": {"R11_predictor_fits": 18}},
               "resource": current_resource}
    frozen = {**core, "fit_budget": {"R11_predictor_fits": 18}, "resource": stable_resource,
              "freeze_revision": 2, "input_sha256": {"train": "x"},
              "source_sha256": {"script": "y"}, "package_versions": {"numpy": "1"}}
    identity = {"input_sha256": frozen["input_sha256"], "source_sha256": frozen["source_sha256"],
                "package_versions": frozen["package_versions"]}
    runner._assert_live_preflight_compatible(context, frozen, identity)
    current_resource["cutoffs"]["67"]["estimated_peak_gib"] = 9.0
    with pytest.raises(runner.BatchIntegrityError, match="stable resource estimates"):
        runner._assert_live_preflight_compatible(context, frozen, identity)


def test_runtime_memory_gate_uses_current_available_ram_and_frozen_peak(tmp_path, monkeypatch):
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text(json.dumps({
        "resource": {
            "cutoffs": {
                "55": {"status": "PASS", "estimated_peak_gib": 2.2},
                "67": {"status": "PASS", "estimated_peak_gib": 2.8},
            }
        }
    }), encoding="utf-8")
    monkeypatch.setattr(
        runner.psutil, "virtual_memory",
        lambda: SimpleNamespace(available=5 * 1024**3),
    )
    result = runner._check_runtime_memory(preflight_path)
    assert result["available_ram_gib_at_pilot_start"] == pytest.approx(5.0)
    assert result["current_allowed_estimated_peak_gib"] == pytest.approx(3.0)
    assert result["frozen_max_estimated_peak_gib"] == pytest.approx(2.8)

    monkeypatch.setattr(
        runner.psutil, "virtual_memory",
        lambda: SimpleNamespace(available=4.5 * 1024**3),
    )
    with pytest.raises(runner.BatchIntegrityError, match="exceeds current allowance"):
        runner._check_runtime_memory(preflight_path)
    monkeypatch.setattr(
        runner.psutil, "virtual_memory",
        lambda: SimpleNamespace(available=3.5 * 1024**3),
    )
    with pytest.raises(runner.BatchIntegrityError, match="below 4.00 GiB"):
        runner._check_runtime_memory(preflight_path)


def test_revision_02_attempt_accounting_separates_two_temporal_calls_from_eighteen_heads(tmp_path):
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    event_path = tmp_path / "attempts.jsonl"
    signature = "frozen-r11-v2"
    events = [
        {"card_id": "R11", "event": "PREFLIGHT_PASS", "freeze_signature": signature,
         "freeze_revision": 2},
        {"card_id": "R11", "event": "RUNTIME_MEMORY_GATE_PASS", "freeze_signature": signature,
         "available_ram_gib_at_pilot_start": 8.0,
         "current_allowed_estimated_peak_gib": 6.0,
         "frozen_max_estimated_peak_gib": 2.7,
         "candidate_head_fits_performed": 0},
    ]
    for cutoff in runner.CUTOFFS:
        events.extend([
            {"card_id": "R11", "event": "FIT_STARTED", "cutoff": cutoff,
             "freeze_signature": signature},
            {"card_id": "R11", "event": "FIT_SAVED", "cutoff": cutoff,
             "fit_predictor_count": 9, "fit_seconds": 1.5},
        ])
        identity = {"freeze_signature": signature, "card_id": "R11", "cutoff": cutoff,
                    "score_rows": [int(cutoff) + 1]}
        write_bundle_exclusive(
            artifact_dir / f"R11_fit{cutoff}.npz",
            {"score_rows": np.array([cutoff + 1]), "prediction": np.array([50.]),
             "baseline_prediction": np.array([51.])},
            identity,
            {"fit_predictor_count": 9,
             "fit_metadata": {"predictor_fit_calls": 9, "vectorizer_fit_calls": 1,
                              "svd_fit_calls": 1},
             "state": {"head_fit_calls": 9}},
        )
    event_path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
    accounting = runner._attempt_accounting(
        event_path=event_path, artifact_dir=artifact_dir, freeze_signature=signature,
    )
    assert accounting["temporal_adapter_attempts_started"] == 2
    assert accounting["successful_predictor_fits_from_receipts"] == 18
    assert accounting["candidate_head_fit_calls_from_metadata"] == 18
    assert accounting["candidate_head_fit_calls_from_state"] == 18
    assert accounting["attempt_accounting_complete"] is True

    events[-1]["fit_predictor_count"] = 8
    event_path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
    incomplete = runner._attempt_accounting(
        event_path=event_path, artifact_dir=artifact_dir, freeze_signature=signature,
    )
    assert incomplete["attempt_accounting_complete"] is False


def test_revision_02_subgroups_report_supported_and_unsupported_cells(tmp_path):
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    cycles = np.arange(56, 80, dtype=np.int64)
    frame = pd.DataFrame({
        "cycle_code": cycles,
        "signal_stream": [None if i % 4 == 0 else f"token{i}" for i in range(len(cycles))],
        "entry_friction": np.arange(len(cycles)) % 2,
    })
    target = np.resize(np.array([2., 10., 50., 95., 99.]), len(cycles))
    signature = "frozen-r11-v2"
    views = runner.EXPECTED_VIEWS
    for cutoff, low in ((55, 56), (67, 68)):
        rows = np.flatnonzero((cycles >= low) & (cycles <= low + 11))
        identity = {"freeze_signature": signature, "card_id": "R11", "cutoff": cutoff,
                    "score_rows": rows.tolist()}
        write_bundle_exclusive(
            artifact_dir / f"R11_fit{cutoff}.npz",
            {"score_rows": rows, "prediction": np.full(len(rows), 50.),
             "baseline_prediction": np.full(len(rows), 50.)},
            identity, {},
        )
    result = runner._subgroup_diagnostics(
        {"freeze_signature": signature}, frame, target, views, artifact_dir,
        minimum_support=1, harm_limit=0.25,
    )
    assert result["views"]["primary"]["stream_missing"]["supported"] is True
    assert result["views"]["primary"]["stream_missing"]["guard_pass"] is True
    strict = runner._subgroup_diagnostics(
        {"freeze_signature": signature}, frame, target, views, artifact_dir,
        minimum_support=3, harm_limit=0.25,
    )
    assert strict["views"]["primary"]["stream_missing"]["supported"] is False
    assert strict["views"]["primary"]["stream_missing"]["evidence_status"] == "DESCRIPTIVE_UNSUPPORTED"


def test_revision_02_gate_retires_only_performance_failures_and_raises_on_qa_failure():
    queue = {"evaluation": {"regression_gate": {
        "pilot_primary_MAE_gain_min": 0.03,
        "pilot_H12_MAE_harm_max": 0.05,
        "individual_pilot_view_MAE_harm_max": 0.5,
    }}}
    card = {"evaluation": "STANDARD"}
    pilot = {"views": {
        "primary": {"gain_positive_is_better": 0.04},
        "H12": {"gain_positive_is_better": -0.01},
        "W1": {"gain_positive_is_better": -0.2},
        "W4": {"gain_positive_is_better": 0.1},
    }}
    verification = {"status": "PASS"}
    subgroups = {"supported_guard_failures": []}
    accounting = {"attempt_accounting_complete": True}
    gate = runner._build_gate(queue, pilot, verification, subgroups, accounting, card)
    assert gate["status"] == "ADVANCE"
    pilot["views"]["primary"]["gain_positive_is_better"] = 0.01
    assert runner._build_gate(queue, pilot, verification, subgroups, accounting, card)["status"] == "RETIRE"
    with pytest.raises(runner.BatchIntegrityError, match="verification did not pass"):
        runner._build_gate(queue, pilot, {"status": "FAIL"}, subgroups, accounting, card)
    with pytest.raises(runner.BatchIntegrityError, match="accounting is incomplete"):
        runner._build_gate(queue, pilot, verification, subgroups,
                           {"attempt_accounting_complete": False}, card)


def test_pilot_cli_routes_to_explicit_revision_02_pilot_mode(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(runner, "run_card", lambda *, mode: calls.append(mode) or {"status": "synthetic"})
    assert runner.main(["--pilot"]) == 0
    assert calls == ["pilot"]
    assert json.loads(capsys.readouterr().out)["status"] == "synthetic"
