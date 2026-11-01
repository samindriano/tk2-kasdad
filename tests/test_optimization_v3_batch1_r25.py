from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from scripts.optimization_v3_batch1_r25 import reconstruct_r25
from scripts.optimization_v3_batch1_verify import verify_report
from src.models import optimization_v3_batch1_r05 as r05
from src.models import optimization_v3_batch1_r25 as r25
from src.models.optimization_v3_batch1 import make_freeze_identity, run_pilot
from src.models.regression_research_text import HYBRID_FEATURES
from src.models.round1_models import CATEGORICAL_FEATURES


def _features(n: int) -> pd.DataFrame:
    rows = np.arange(n)
    values: dict[str, object] = {}
    for column in HYBRID_FEATURES:
        if column == "signal_stream":
            values[column] = [None if i % 23 == 0 else "alpha alpha alpha beta beta gamma" for i in rows]
        elif column in CATEGORICAL_FEATURES:
            values[column] = ["known" if i % 3 else "other" for i in rows]
        else:
            values[column] = (rows % 17).astype(float)
    return pd.DataFrame(values).loc[:, list(HYBRID_FEATURES)]


class _FakeBooster:
    def model_to_string(self, num_iteration=-1):
        return "synthetic-r25-model"

    def current_iteration(self):
        return 4


class _FakeLGBM:
    def __init__(self, **params):
        self.booster_ = _FakeBooster()

    def fit(self, matrix, target):
        assert sparse.issparse(matrix)
        self.mean_ = float(np.mean(target))
        self.feature_importances_ = np.zeros(matrix.shape[1], dtype=np.int64)
        return self

    def predict(self, matrix):
        assert sparse.issparse(matrix)
        return self.mean_ + np.asarray(matrix.sum(axis=1)).reshape(-1) * 0.001


def test_r25_is_unnormalized_countvectorizer_and_passes_generic_verifier(monkeypatch, tmp_path):
    monkeypatch.setattr(r05.lightgbm, "LGBMRegressor", _FakeLGBM)
    cycles = np.repeat(np.arange(47, 80, dtype=np.int64), 2)
    X = _features(len(cycles))
    model = r25.R25UnnormalizedWordCountHybrid().fit(X[cycles <= 55], np.linspace(10, 80, int(np.sum(cycles <= 55))))
    state = model.audit_state()
    assert model.vectorizer is not None
    assert model.vectorizer.binary is False
    assert model.vectorizer.ngram_range == (1, 2)
    assert model.vectorizer.min_df == 2
    assert model.vectorizer.max_features == 20_000
    assert np.any(model.vectorizer.transform(X.signal_stream.fillna("")).data > 1)
    assert state["count_vectorizer"]["parameters"]["binary"] is False
    assert state["count_vectorizer"]["parameters"]["use_idf"] is False
    assert state["count_vectorizer"]["parameters"]["row_normalization"] is None
    assert state["id"] == "V3_BATCH1_R25_UNNORMALIZED_WORD_COUNTS_SPARSE_L1"

    frame = X.copy()
    frame.insert(0, "record_id", [f"r25-string-id-{i:03d}" for i in range(len(frame))])
    frame["cycle_code"] = cycles
    target = np.linspace(1.0, 99.0, len(frame))
    views = {
        "W1": [(55, 56, 59)],
        "W4": [(67, 68, 71)],
        "primary": [(55, 56, 59), (67, 68, 71)],
        "H12": [(55, 56, 67), (67, 68, 79)],
    }
    source = tmp_path / "r25_source.py"
    input_path = tmp_path / "r25_train.csv"
    source.write_text("synthetic R25 adapter identity\n", encoding="utf-8")
    input_path.write_text("synthetic immutable input\n", encoding="utf-8")
    identity = make_freeze_identity(
        card={"id": "R25", "task": "regression"}, feature_names=HYBRID_FEATURES,
        input_files={"train": input_path}, source_files=[source],
        evaluation_views=views, packages=[],
    )

    def control_loader(cutoff, score_rows):
        score_cycles = cycles[score_rows]
        ridge = np.full(len(score_rows), 50.0)
        old_raw = np.linspace(12.0, 88.0, len(score_rows))
        return {
            "baseline_prediction": r05.compose_r05_prediction(ridge, old_raw, score_cycles),
            "ridge25_prediction": ridge,
            "baseline_sparse_raw": old_raw,
            "cycle_code": score_cycles,
        }

    report = run_pilot(
        frame=frame, target=target,
        card={"id": "R25", "task": "regression", "pilot_predictor_fits": 2},
        feature_names=HYBRID_FEATURES, adapter=r25.fit_predict,
        control_loader=control_loader, freeze_identity=identity,
        freeze_path=tmp_path / "r25-freeze.json", artifact_dir=tmp_path / "r25-artifacts",
        event_path=tmp_path / "r25-attempts.jsonl", evaluation_views=views,
        report_path=tmp_path / "r25-pilot.json",
    )
    verified = verify_report(
        report=report, frame=frame, target=target,
        artifact_dir=tmp_path / "r25-artifacts", freeze_signature=report["freeze_signature"],
        evaluation_views=views, prediction_reconstructors={"R25": reconstruct_r25},
    )
    assert report["fit_count_matches_card"] is True
    assert verified["status"] == "PASS"
    assert verified["views"]["primary"]["n"] == 16
    assert verified["views"]["H12"]["n"] == 48
