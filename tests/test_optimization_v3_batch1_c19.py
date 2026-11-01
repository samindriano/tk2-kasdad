from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.optimization_v3_batch1_c19 import _recipe_payload, reconstruct_c19
from scripts.optimization_v3_batch1_verify import verify_report
from src.models import optimization_v3_batch1_c19 as c19
from src.models.optimization_v3_batch1 import make_freeze_identity, run_pilot
from src.models.regression_research_text import HYBRID_FEATURES
from src.models.round1_models import CATEGORICAL_FEATURES


def _features(n: int, cycles: np.ndarray | None = None) -> pd.DataFrame:
    rows = np.arange(n)
    cycles = (47 + rows % 33) if cycles is None else np.asarray(cycles, dtype=int)
    values: dict[str, object] = {}
    for column in HYBRID_FEATURES:
        if column == "signal_stream":
            values[column] = [
                None if i % 19 == 0 else " ".join(f"word{(i * 5 + j * 13) % 80:02d}" for j in range(6))
                for i in rows
            ]
        elif column in CATEGORICAL_FEATURES:
            values[column] = ["known" if i % 3 else "other" for i in rows]
        elif column == "cycle_code":
            values[column] = cycles
        else:
            values[column] = (rows % 17).astype(float)
    return pd.DataFrame(values).loc[:, list(HYBRID_FEATURES)]


class _FakeXGBClassifier:
    instances: list["_FakeXGBClassifier"] = []

    def __init__(self, **parameters):
        self.parameters = parameters
        self.__class__.instances.append(self)

    def get_params(self, deep=False):
        return dict(self.parameters)

    def fit(self, design, target, verbose=False):
        dense = np.asarray(design, dtype=np.float64)
        assert dense.ndim == 2 and np.isfinite(dense).all()
        self.mean_ = float(np.mean(target))
        self.first_feature_mean_ = float(np.mean(dense[:, 0]))
        self.classes_ = np.array([0, 1], dtype=np.int8)
        self.n_features_in_ = dense.shape[1]
        return self

    def predict_proba(self, design):
        dense = np.asarray(design, dtype=np.float64)
        positive = np.clip(self.mean_ + 0.01 * (dense[:, 0] - self.first_feature_mean_), 0.01, 0.99)
        return np.column_stack((1.0 - positive, positive))


def _controls(cycles: np.ndarray, rows: np.ndarray, features: pd.DataFrame):
    score = features.iloc[rows]
    _docs, missing = c19._text_values(score["signal_stream"])
    c2b = np.linspace(0.2, 0.8, len(rows))
    old_sparse = np.linspace(0.3, 0.7, len(rows))
    cb = np.full(len(rows), 0.42)
    baseline = np.where(missing, cb, 0.75 * c2b + 0.25 * old_sparse)
    return {
        "baseline_prediction": baseline,
        "baseline_sparse_raw": old_sparse,
        "c2b_probability": c2b,
        "cb_probability": cb,
        "stream_missing": missing,
        "control_rows": rows.astype(np.int64),
    }


def test_c19_fixed_recipe_and_exact_svd32_with_fit_only_preprocessing(monkeypatch):
    _FakeXGBClassifier.instances.clear()
    monkeypatch.setattr(c19, "XGBClassifier", _FakeXGBClassifier)
    X = _features(96, np.repeat(np.arange(47, 59), 8))
    y = (np.arange(len(X)) % 2).astype(np.int8)
    model = c19.XGBoostSVD32Classifier(55).fit(X.iloc[:72], y[:72])
    state = model.audit_state()
    probability = model.predict_probability(X.iloc[72:])
    assert state["svd"]["n_components_requested"] == 32
    assert state["svd"]["n_components_actual"] == 32
    assert state["tfidf_vocabulary_size"] > 32
    assert state["validation_data_passed_to_fit"] is False
    assert state["validation_labels_passed_to_fit"] is False
    assert state["test_data_used"] is False
    assert _FakeXGBClassifier.instances[-1].parameters == c19.XGB_PARAMETERS
    assert np.isfinite(probability).all() and np.all((probability >= 0) & (probability <= 1))


def test_c19_control_alignment_and_baseline_must_be_reconstructed_exactly():
    features = _features(16, np.repeat([55, 56], 8))
    rows = np.arange(8, 16, dtype=np.int64)
    controls = _controls(features.cycle_code.to_numpy(), rows, features)
    arrays, aligned = c19._validated_controls(controls, len(rows))
    assert np.array_equal(aligned, rows)
    assert np.array_equal(arrays["baseline_prediction"], controls["baseline_prediction"])
    controls["baseline_prediction"] = controls["baseline_prediction"].copy()
    controls["baseline_prediction"][0] = 0.99
    with pytest.raises(ValueError, match="fixed row-matched sparse25"):
        c19._validated_controls(controls, len(rows))


def test_c19_recipe_is_json_safe_and_keeps_fixed_tfidf_parameters():
    recipe = _recipe_payload(
        {key: value for key, value in {
            "id": "C19", "task": "classification", "name": "C19",
            "family": "xgb_dense", "priority": "P2", "change": "fixed",
            "policy": "C_AUX25", "pilot_predictor_fits": 2,
            "evaluation": "STANDARD", "condition": "No additional dependency",
        }.items()},
        {"primary": [(55, 56, 59), (67, 68, 71)]},
        {"primary": {"rows": 10, "macro_f1": 0.5}},
        {"evaluation": {"classification_gate": {}}},
    )
    assert recipe["text"]["parameters"]["dtype"] == "float64"
    assert recipe["text"]["parameters"]["max_features"] == 20_000
    assert recipe["text"]["svd"]["n_components"] == 32


def test_c19_run_pilot_and_independent_route_verification(monkeypatch, tmp_path):
    _FakeXGBClassifier.instances.clear()
    monkeypatch.setattr(c19, "XGBClassifier", _FakeXGBClassifier)
    cycles = np.repeat(np.arange(47, 80, dtype=np.int64), 8)
    X = _features(len(cycles), cycles)
    frame = X.copy()
    frame.insert(0, "record_id", [f"c19-string-id-{i:04d}" for i in range(len(frame))])
    target = (np.arange(len(frame)) % 2).astype(np.int8)
    frame["artifact_signal"] = target
    frame["threat_rank"] = (np.arange(len(frame)) % 101).astype(float)
    views = {
        "W1": [(55, 56, 59)], "W4": [(67, 68, 71)],
        "primary": [(55, 56, 59), (67, 68, 71)],
        "H12": [(55, 56, 67), (67, 68, 79)],
    }
    source = tmp_path / "c19_source.py"
    train_path = tmp_path / "classification_train.csv"
    source.write_text("synthetic frozen C19 adapter\n", encoding="utf-8")
    train_path.write_text("synthetic immutable Train fixture\n", encoding="utf-8")
    identity = make_freeze_identity(
        card={"id": "C19", "task": "classification"}, feature_names=HYBRID_FEATURES,
        input_files={"train": train_path}, source_files=[source],
        evaluation_views=views, packages=[],
    )

    def control_loader(cutoff, score_rows):
        return _controls(cycles, score_rows, frame)

    report = run_pilot(
        frame=frame, target=target,
        card={"id": "C19", "task": "classification", "pilot_predictor_fits": 2},
        feature_names=HYBRID_FEATURES, adapter=c19.fit_predict,
        control_loader=control_loader, freeze_identity=identity,
        freeze_path=tmp_path / "c19-freeze.json", artifact_dir=tmp_path / "c19-artifacts",
        event_path=tmp_path / "c19-attempts.jsonl", evaluation_views=views,
        report_path=tmp_path / "c19-pilot.json",
    )
    verified = verify_report(
        report=report, frame=frame, target=target,
        artifact_dir=tmp_path / "c19-artifacts", freeze_signature=report["freeze_signature"],
        evaluation_views=views, prediction_reconstructors={"C19": reconstruct_c19},
    )
    assert report["fit_count_matches_card"] is True
    assert report["views"]["primary"]["rows"] == 64
    assert report["views"]["H12"]["rows"] == 192
    assert verified["status"] == "PASS"
    assert len(_FakeXGBClassifier.instances) == 2

