from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from scripts import optimization_v3_batch1_c08 as c08_runner
from scripts.optimization_v3_batch1_verify import verify_report
from src.models import optimization_v3_batch1_c08 as c08
from src.models.optimization_v3_batch1 import make_freeze_identity, run_pilot
from src.models.regression_research_text import HYBRID_FEATURES
from src.models.round1_models import CATEGORICAL_FEATURES


def _features(cycles: np.ndarray) -> pd.DataFrame:
    cycles = np.asarray(cycles, dtype=np.int64)
    rows = np.arange(len(cycles))
    values: dict[str, object] = {}
    for column in HYBRID_FEATURES:
        if column == "signal_stream":
            values[column] = [
                None if i % 23 == 0 else " ".join(f"word{(i * 7 + j * 19) % 96:02d}" for j in range(6))
                for i in rows
            ]
        elif column == "cycle_code":
            values[column] = cycles
        elif column in CATEGORICAL_FEATURES:
            values[column] = ["A" if i % 2 else "B" for i in rows]
        else:
            values[column] = (rows % 17).astype(float)
    return pd.DataFrame(values).loc[:, list(HYBRID_FEATURES)]


def _manual_ordered_rates(train: pd.DataFrame, target: np.ndarray, score: pd.DataFrame):
    fields = c08.TARGET_RATE_FIELDS
    y = np.asarray(target, dtype=float)
    cycles = train.cycle_code.to_numpy(dtype=int)
    train_output = np.zeros((len(train), len(fields)), dtype=float)
    train_support = np.zeros((len(train), len(fields)), dtype=np.int32)
    train_fallback = np.zeros((len(train), len(fields)), dtype=np.uint8)
    score_output = np.zeros((len(score), len(fields)), dtype=float)
    score_support = np.zeros((len(score), len(fields)), dtype=np.int32)
    score_fallback = np.zeros((len(score), len(fields)), dtype=np.uint8)

    def key(value):
        if value is None:
            return None
        try:
            if bool(pd.isna(value)):
                return None
        except (TypeError, ValueError):
            pass
        return str(value).strip()

    counts = {field: {} for field in fields}
    totals = {field: {} for field in fields}
    for cycle in sorted(set(cycles.tolist())):
        positions = np.flatnonzero(cycles == cycle)
        for j, field in enumerate(fields):
            for row in positions:
                category = key(train.iloc[row][field])
                n = counts[field].get(category, 0) if category is not None else 0
                total = totals[field].get(category, 0.0) if category is not None else 0.0
                train_support[row, j] = n
                if category is None:
                    train_fallback[row, j] = c08.FALLBACK_CODES["missing_category"]
                elif n == 0:
                    train_fallback[row, j] = c08.FALLBACK_CODES["no_prior_category_history"]
                elif n < c08.MINIMUM_PRIOR_SUPPORT:
                    train_fallback[row, j] = c08.FALLBACK_CODES["below_minimum_support"]
                else:
                    train_fallback[row, j] = c08.FALLBACK_CODES["supported"]
                rate = c08.INITIAL_PRIOR if n < c08.MINIMUM_PRIOR_SUPPORT else (
                    (total + c08.SMOOTHING * c08.INITIAL_PRIOR) / (n + c08.SMOOTHING)
                )
                clipped = np.clip(rate, c08.LOGIT_EPSILON, 1 - c08.LOGIT_EPSILON)
                train_output[row, j] = np.log(clipped / (1 - clipped))
        # Manual reconstruction updates all fields after current-cycle features.
        for field in fields:
            grouped: dict[str, list[int]] = {}
            for row in positions:
                category = key(train.iloc[row][field])
                if category is not None:
                    grouped.setdefault(category, []).append(int(row))
            for category in sorted(grouped):
                selected_rows = grouped[category]
                counts[field][category] = counts[field].get(category, 0) + len(selected_rows)
                totals[field][category] = totals[field].get(category, 0.0) + float(y[selected_rows].sum())

    for j, field in enumerate(fields):
        for row in range(len(score)):
            category = key(score.iloc[row][field])
            n = counts[field].get(category, 0) if category is not None else 0
            total = totals[field].get(category, 0.0) if category is not None else 0.0
            score_support[row, j] = n
            if category is None:
                score_fallback[row, j] = c08.FALLBACK_CODES["missing_category"]
            elif n == 0:
                score_fallback[row, j] = c08.FALLBACK_CODES["no_prior_category_history"]
            elif n < c08.MINIMUM_PRIOR_SUPPORT:
                score_fallback[row, j] = c08.FALLBACK_CODES["below_minimum_support"]
            else:
                score_fallback[row, j] = c08.FALLBACK_CODES["supported"]
            rate = c08.INITIAL_PRIOR if n < c08.MINIMUM_PRIOR_SUPPORT else (
                (total + c08.SMOOTHING * c08.INITIAL_PRIOR) / (n + c08.SMOOTHING)
            )
            clipped = np.clip(rate, c08.LOGIT_EPSILON, 1 - c08.LOGIT_EPSILON)
            score_output[row, j] = np.log(clipped / (1 - clipped))
    return train_output, train_support, train_fallback, score_output, score_support, score_fallback


class _FakeBooster:
    def model_to_string(self):
        return "synthetic-c08-lightgbm-model"

    def predict(self, matrix, num_threads=2):
        assert sparse.issparse(matrix)
        assert num_threads == 2
        return np.full(matrix.shape[0], 0.57, dtype=np.float64)


class _FakeLGBMClassifier:
    instances = []

    def __init__(self, **parameters):
        self.parameters = parameters
        self.booster_ = _FakeBooster()
        self.__class__.instances.append(self)

    def fit(self, matrix, target):
        assert sparse.issparse(matrix)
        assert matrix.shape[1] > 3
        assert np.isfinite(matrix.data).all()
        assert np.isin(target, (0, 1)).all()
        self.fit_rows_ = int(len(target))
        return self


def test_ordered_rate_logits_match_independent_manual_support_and_smoothing():
    cycles = np.array([47] * 60 + [48] * 5, dtype=np.int64)
    X = _features(cycles)
    # Force 31 prior rows for A and 29 for B in org_code. The current cycle
    # includes supported, under-supported, unseen, missing, and blank values.
    X["org_code"] = ["A"] * 31 + ["B"] * 29 + ["A", "B", "C", None, ""]
    y = np.zeros(len(X), dtype=np.int8)
    y[::3] = 1
    y[60:] = [1, 0, 1, 0, 1]
    encoder = c08.OrderedCategoryTargetRates()
    actual_train, train_support, train_fallback = encoder.fit_transform(X, y)
    actual_score, score_support, score_fallback, _diag = encoder.transform(X.iloc[60:])
    expected = _manual_ordered_rates(X, y, X.iloc[60:])
    assert np.array_equal(actual_train, expected[0])
    assert np.array_equal(train_support, expected[1])
    assert np.array_equal(train_fallback, expected[2])
    assert np.array_equal(actual_score, expected[3])
    assert np.array_equal(score_support, expected[4])
    assert np.array_equal(score_fallback, expected[5])
    assert np.all(actual_train[:60] == 0.0)  # first cycle uses prior .5
    assert train_support[60:62, 0].tolist() == [31, 29]
    assert train_fallback[60:62, 0].tolist() == [c08.FALLBACK_CODES["supported"], c08.FALLBACK_CODES["below_minimum_support"]]
    assert train_fallback[62, 0] == c08.FALLBACK_CODES["no_prior_category_history"]
    assert train_fallback[63, 0] == c08.FALLBACK_CODES["missing_category"]
    assert train_fallback[64, 0] == c08.FALLBACK_CODES["no_prior_category_history"]
    expected_a = (float(y[:31].sum()) + 30 * 0.5) / (31 + 30)
    assert actual_train[60, 0] == pytest.approx(np.log(expected_a / (1 - expected_a)))
    assert actual_train[61, 0] == 0.0  # 29 prior rows fall back to the prior.
    assert np.isfinite(actual_train).all() and np.isfinite(actual_score).all()


def test_same_cycle_label_perturbation_cannot_change_own_or_past_features():
    cycles = np.repeat(np.array([47, 48, 49], dtype=np.int64), 40)
    X = _features(cycles)
    categories = np.tile(["A", "B"], len(X) // 2)
    for field in CATEGORICAL_FEATURES:
        X[field] = categories
    y = (np.arange(len(X)) % 3 == 0).astype(np.int8)
    original, _support, _fallback = c08.OrderedCategoryTargetRates().fit_transform(X, y)
    perturbed = y.copy()
    perturbed[cycles == 48] = 1 - perturbed[cycles == 48]
    changed, _support2, _fallback2 = c08.OrderedCategoryTargetRates().fit_transform(X, perturbed)
    assert np.array_equal(original[cycles <= 48], changed[cycles <= 48])
    assert not np.array_equal(original[cycles == 49], changed[cycles == 49])


def test_rate_preflight_distinguishes_integrity_failure_from_insufficient_variation(monkeypatch):
    cycles = np.repeat(np.array([47, 48, 49], dtype=np.int64), 40)
    X = _features(cycles)
    y = (np.arange(len(X)) % 2).astype(np.int8)
    score = _features(np.array([50, 50, 51], dtype=np.int64))
    for field in CATEGORICAL_FEATURES:
        X[field] = "same-category"
        score[field] = "same-category"
    original_manual = c08_runner._manual_ordered_rate_arrays

    def corrupted_manual(*args, **kwargs):
        result = original_manual(*args, **kwargs)
        result["train_logits"][0, 0] += 0.25
        return result

    monkeypatch.setattr(c08_runner, "_manual_ordered_rate_arrays", corrupted_manual)
    integrity_result = c08_runner._rate_preflight(X, y, score, cutoff=49)
    assert integrity_result["status"] == "FAIL_PREFLIGHT"
    assert integrity_result["manual_training_reconstruction_pass"] is False

    monkeypatch.setattr(c08_runner, "_manual_ordered_rate_arrays", original_manual)
    unsupported_result = c08_runner._rate_preflight(X, y, score, cutoff=49)
    assert unsupported_result["status"] == "SKIPPED_SUPPORT"
    assert unsupported_result["manual_training_reconstruction_pass"] is True
    assert unsupported_result["same_cycle_label_perturbation_pass"] is True


def test_c08_fit_accounting_accepts_retries_and_verified_resume_bundles(monkeypatch, tmp_path):
    run_dir = tmp_path / "runs"
    run_dir.mkdir()
    events_path = tmp_path / "attempts.jsonl"
    monkeypatch.setattr(c08_runner, "RUN_DIR", run_dir)
    monkeypatch.setattr(c08_runner, "EVENT_PATH", events_path)
    for cutoff in (55, 67):
        (run_dir / f"C08_fit{cutoff}.json").write_text(json.dumps({
            "metadata": {"cutoff": cutoff, "fit_predictor_count": 1, "fit_seconds": 2.0}
        }), encoding="utf-8")

    retry_events = [
        {"cutoff": 55, "event": "FIT_STARTED"},
        {"cutoff": 55, "event": "FIT_FAILED"},
        {"cutoff": 55, "event": "FIT_STARTED"},
        {"cutoff": 55, "event": "FIT_SAVED"},
        {"cutoff": 67, "event": "FIT_STARTED"},
        {"cutoff": 67, "event": "FIT_SAVED"},
    ]
    events_path.write_text("\n".join(json.dumps(event) for event in retry_events), encoding="utf-8")
    retry_accounting = c08_runner._fit_accounting()
    assert retry_accounting["attempt_accounting_complete"] is True
    assert retry_accounting["technical_retries"] == 1
    assert retry_accounting["successful_saved_predictor_fits"] == 2

    resume_events = [
        {"cutoff": 55, "event": "FIT_STARTED"},
        {"cutoff": 55, "event": "RESUMED_VERIFIED"},
        {"cutoff": 67, "event": "FIT_STARTED"},
        {"cutoff": 67, "event": "FIT_SAVED"},
    ]
    events_path.write_text("\n".join(json.dumps(event) for event in resume_events), encoding="utf-8")
    resume_accounting = c08_runner._fit_accounting()
    assert resume_accounting["attempt_accounting_complete"] is True
    assert resume_accounting["resumed_verified_events"] == 1
    assert resume_accounting["fit_origins_saved"] == [55, 67]


def test_c08_memory_preflight_bounds_sparse_score_design(monkeypatch):
    fit_features = _features(np.repeat(np.array([47, 48], dtype=np.int64), 40))
    score_features = _features(np.repeat(np.array([49, 50], dtype=np.int64), 20))
    estimate = c08_runner._memory_estimate(
        fit_features, score_features, available_bytes=8 * 1024**3
    )
    former_dense_bound = len(score_features) * (estimate["sparse_fit_design_width"] + 4) * 8 * 2
    assert estimate["status"] == "PASS"
    assert estimate["score_design_nnz"] > 0
    assert estimate["score_scratch_bytes_upper_bound"] == 5 * estimate["score_sparse_design_bytes"]
    assert estimate["score_scratch_bytes_upper_bound"] < former_dense_bound
    assert estimate["estimated_additional_peak_bytes"] >= (
        estimate["tokenization_allowance_bytes"] + estimate["model_runtime_allowance_bytes"]
        + 5 * estimate["fit_sparse_design_bytes"] + estimate["score_scratch_bytes_upper_bound"]
    )


def test_c08_validates_controls_and_reconstructs_c_sparse_replace_route():
    n = 5
    missing = np.array([False, True, False, True, False])
    c2b = np.array([0.2, 0.3, 0.5, 0.6, 0.8])
    old_sparse = np.array([0.8, 0.2, 0.7, 0.3, 0.4])
    cb = np.full(n, 0.42)
    baseline = np.where(missing, cb, 0.75 * c2b + 0.25 * old_sparse)
    controls = {
        "baseline_prediction": baseline, "baseline_sparse_raw": old_sparse,
        "c2b_probability": c2b, "cb_probability": cb,
        "stream_missing": missing, "control_rows": np.arange(n, dtype=np.int64),
    }
    arrays, rows = c08._validated_controls(controls, n)
    assert np.array_equal(rows, np.arange(n))
    raw = np.array([0.1, 0.9, 0.3, 0.7, 0.4])
    candidate = np.where(missing, cb, 0.75 * c2b + 0.25 * raw)
    assert np.array_equal(candidate[missing], cb[missing])
    assert np.array_equal(arrays["baseline_prediction"][missing], cb[missing])
    controls["control_rows"] = np.arange(n, dtype=np.int64)[::-1]
    with pytest.raises(ValueError, match="increasing unique"):
        c08._validated_controls(controls, n)


def test_c08_generic_pilot_rate_artifacts_manual_verification_and_route(monkeypatch, tmp_path):
    _FakeLGBMClassifier.instances.clear()
    monkeypatch.setattr(c08, "LGBMClassifier", _FakeLGBMClassifier)
    cycles = np.repeat(np.arange(47, 80, dtype=np.int64), 8)
    X = _features(cycles)
    frame = X.copy()
    frame.insert(0, "record_id", [f"c08-id-{i:04d}" for i in range(len(X))])
    target = (np.arange(len(X)) % 3 == 0).astype(np.int8)
    frame["artifact_signal"] = target
    frame["threat_rank"] = (np.arange(len(X)) % 101).astype(float)
    views = {
        "W1": [(55, 56, 59)], "W4": [(67, 68, 71)],
        "primary": [(55, 56, 59), (67, 68, 71)],
        "H12": [(55, 56, 67), (67, 68, 79)],
    }
    source = tmp_path / "c08_source.py"
    train_path = tmp_path / "classification_train.csv"
    source.write_text("synthetic C08 frozen adapter\n", encoding="utf-8")
    train_path.write_text("synthetic immutable Train fixture\n", encoding="utf-8")
    identity = make_freeze_identity(
        card={"id": "C08", "task": "classification"}, feature_names=HYBRID_FEATURES,
        input_files={"train": train_path}, source_files=[source],
        evaluation_views=views, packages=[],
    )

    def controls_loader(cutoff, score_rows):
        missing = frame.iloc[score_rows]["signal_stream"].isna().to_numpy()
        c2b = np.linspace(0.2, 0.8, len(score_rows))
        old_sparse = np.linspace(0.3, 0.7, len(score_rows))
        cb = np.full(len(score_rows), 0.41)
        baseline = np.where(missing, cb, 0.75 * c2b + 0.25 * old_sparse)
        return {
            "baseline_prediction": baseline,
            "baseline_sparse_raw": old_sparse,
            "c2b_probability": c2b,
            "cb_probability": cb,
            "stream_missing": missing,
            "control_rows": score_rows.astype(np.int64),
        }

    report = run_pilot(
        frame=frame, target=target,
        card={"id": "C08", "task": "classification", "pilot_predictor_fits": 2},
        feature_names=HYBRID_FEATURES, adapter=c08.fit_predict,
        control_loader=controls_loader, freeze_identity=identity,
        freeze_path=tmp_path / "c08-freeze.json", artifact_dir=tmp_path / "c08-artifacts",
        event_path=tmp_path / "c08-attempts.jsonl", evaluation_views=views,
        report_path=tmp_path / "c08-pilot.json",
    )

    def reconstruct(arrays):
        raw = arrays["raw_candidate"].astype(float)
        missing = arrays["control__stream_missing"].astype(bool)
        c2b = arrays["control__c2b_probability"].astype(float)
        cb = arrays["control__cb_probability"].astype(float)
        old = arrays["control__baseline_sparse_raw"].astype(float)
        return np.where(missing, cb, 0.75 * c2b + 0.25 * raw), np.where(missing, cb, 0.75 * c2b + 0.25 * old)

    verification = verify_report(
        report=report, frame=frame, target=target,
        artifact_dir=tmp_path / "c08-artifacts", freeze_signature=report["freeze_signature"],
        evaluation_views=views, prediction_reconstructors={"C08": reconstruct},
    )
    for cutoff in (55, 67):
        fit_rows = np.flatnonzero(cycles <= cutoff)
        score_rows = np.flatnonzero((cycles > cutoff) & (cycles <= cutoff + 12))
        with np.load(tmp_path / "c08-artifacts" / f"C08_fit{cutoff}.npz", allow_pickle=False) as saved:
            arrays = {name: saved[name].copy() for name in saved.files}
        manual = _manual_ordered_rates(frame.iloc[fit_rows], target[fit_rows], frame.iloc[score_rows])
        assert np.array_equal(arrays["audit__ordered_rate_logits"], manual[3])
        assert np.array_equal(arrays["audit__ordered_rate_support"], manual[4])
        assert np.array_equal(arrays["audit__ordered_rate_fallback"], manual[5])
    assert report["fit_count_matches_card"] is True
    assert verification["status"] == "PASS"
    assert report["views"]["primary"]["rows"] == 64
    assert report["views"]["H12"]["rows"] == 192
    assert len(_FakeLGBMClassifier.instances) == 2
