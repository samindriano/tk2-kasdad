from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from scripts.optimization_v3_batch1_verify import verify_report
from scripts.optimization_v3_batch1_r05 import reconstruct_r05
from src.models import optimization_v3_batch1_r05 as r05
from src.models.optimization_v3_batch1 import make_freeze_identity, run_pilot
from src.models.regression_research_text import HYBRID_FEATURES
from src.models.round1_models import CATEGORICAL_FEATURES


def _features(n: int, *, future_text: bool = False) -> pd.DataFrame:
    rows = np.arange(n)
    values: dict[str, object] = {}
    for column in HYBRID_FEATURES:
        if column == "signal_stream":
            values[column] = [
                None if i % 19 == 0 else
                "futureword alpha beta" if future_text and i >= n - 4 else
                f"alpha beta gamma word{i % 5}"
                for i in rows
            ]
        elif column in CATEGORICAL_FEATURES:
            values[column] = ["known" if i % 3 else "other" for i in rows]
        else:
            values[column] = (rows % 17).astype(float)
    return pd.DataFrame(values).loc[:, list(HYBRID_FEATURES)]


class _FakeBooster:
    def model_to_string(self, num_iteration=-1):
        return "synthetic-lightgbm-state"

    def current_iteration(self):
        return 4


class _FakeLGBM:
    fit_matrices: list[sparse.csr_matrix] = []

    def __init__(self, **params):
        self.params = params
        self.booster_ = _FakeBooster()

    def fit(self, matrix, target):
        assert sparse.issparse(matrix)
        assert len(target) == matrix.shape[0]
        type(self).fit_matrices.append(matrix.tocsr())
        self.feature_importances_ = np.zeros(matrix.shape[1], dtype=np.int64)
        self.offset_ = float(np.mean(target))
        return self

    def predict(self, matrix):
        assert sparse.issparse(matrix)
        return self.offset_ + np.asarray(matrix.sum(axis=1)).reshape(-1) * 0.001


def _patch_fake_estimator(monkeypatch):
    _FakeLGBM.fit_matrices = []
    monkeypatch.setattr(r05.lightgbm, "LGBMRegressor", _FakeLGBM)


def test_r05_vectorizer_binary_sparse_fit_only_and_default_tokenizer(monkeypatch):
    _patch_fake_estimator(monkeypatch)
    train = _features(40)
    score = _features(8, future_text=True)
    model = r05.R05BinaryPresenceSparseHybrid().fit(train, np.linspace(20, 80, len(train)))
    vectorizer = model.vectorizer
    assert vectorizer is not None
    assert vectorizer.binary is True
    assert vectorizer.ngram_range == (1, 2)
    assert vectorizer.min_df == 2
    assert vectorizer.max_features == 20_000
    assert vectorizer.vocabulary_ and "futureword" not in vectorizer.vocabulary_
    assert _FakeLGBM.fit_matrices and sparse.isspmatrix_csr(_FakeLGBM.fit_matrices[-1])
    assert model.state["count_vectorizer"]["parameters"]["row_normalization"] is None
    assert model.state["count_vectorizer"]["parameters"]["binary"] is True
    prediction = model.predict(score)
    assert prediction.shape == (len(score),)
    assert np.isfinite(prediction).all()
    # IDs and targets are outside the exact feature contract.
    with np.testing.assert_raises_regex(ValueError, "exact feature allowlist"):
        model.predict(score.assign(record_id=[f"id-{i}" for i in range(len(score))]))


def test_r05_rank50_uses_average_ties_and_complete_cycle_groups():
    raw = np.array([10.0, 10.0, 30.0, 0.0, 10.0])
    cycles = np.array([7, 7, 7, 8, 8])
    result = r05.rank50_prediction(raw, cycles)
    expected = np.array([
        0.5 * 10 + 0.5 * (100 * 1.5 / 3),
        0.5 * 10 + 0.5 * (100 * 1.5 / 3),
        0.5 * 30 + 0.5 * (100 * 3 / 3),
        0.5 * 0 + 0.5 * (100 * 1 / 2),
        0.5 * 10 + 0.5 * (100 * 2 / 2),
    ])
    assert np.allclose(result, expected)
    ridge = np.full(len(raw), 50.0)
    composite = r05.compose_r05_prediction(ridge, raw, cycles)
    assert np.allclose(composite, 0.75 * ridge + 0.25 * result)
    assert np.all((composite >= 0) & (composite <= 100))


def test_r05_model_output_reaches_generic_multiview_verifier(monkeypatch, tmp_path):
    _patch_fake_estimator(monkeypatch)
    cycles = np.repeat(np.arange(47, 80, dtype=np.int64), 2)
    frame = _features(len(cycles))
    frame.insert(0, "record_id", [f"string-id-{i:03d}" for i in range(len(frame))])
    frame["cycle_code"] = cycles
    target = np.linspace(3.0, 97.0, len(frame))
    views = {
        "W1": [(55, 56, 59)],
        "W4": [(67, 68, 71)],
        "primary": [(55, 56, 59), (67, 68, 71)],
        "H12": [(55, 56, 67), (67, 68, 79)],
    }
    source = tmp_path / "adapter.py"
    train_path = tmp_path / "synthetic_train.csv"
    source.write_text("synthetic adapter identity\n", encoding="utf-8")
    train_path.write_text("synthetic immutable input\n", encoding="utf-8")
    identity = make_freeze_identity(
        card={"id": "R05", "task": "regression"},
        feature_names=HYBRID_FEATURES,
        input_files={"synthetic_train": train_path},
        source_files=[source],
        evaluation_views=views,
        packages=[],
    )

    def controls(cutoff, score_rows):
        score_cycles = cycles[score_rows]
        ridge = np.full(len(score_rows), 50.0)
        raw_old = np.linspace(10.0, 90.0, len(score_rows))
        return {
            "baseline_prediction": r05.compose_r05_prediction(ridge, raw_old, score_cycles),
            "ridge25_prediction": ridge,
            "baseline_sparse_raw": raw_old,
            "cycle_code": score_cycles,
        }

    card = {"id": "R05", "task": "regression", "pilot_predictor_fits": 2}
    artifact_dir = tmp_path / "artifacts"
    report = run_pilot(
        frame=frame,
        target=target,
        card=card,
        feature_names=HYBRID_FEATURES,
        adapter=r05.fit_predict,
        control_loader=controls,
        freeze_identity=identity,
        freeze_path=tmp_path / "freeze.json",
        artifact_dir=artifact_dir,
        event_path=tmp_path / "attempts.jsonl",
        evaluation_views=views,
        report_path=tmp_path / "pilot.json",
    )
    verified = verify_report(
        report=report,
        frame=frame,
        target=target,
        artifact_dir=artifact_dir,
        freeze_signature=report["freeze_signature"],
        evaluation_views=views,
        prediction_reconstructors={"R05": reconstruct_r05},
    )
    assert report["fit_count_matches_card"] is True
    assert len(_FakeLGBM.fit_matrices) == 2
    assert report["views"]["primary"]["rows"] == 16
    assert report["views"]["H12"]["rows"] == 48
    assert verified["status"] == "PASS"
    assert verified["views"]["primary"]["n"] == 16
    assert verified["views"]["H12"]["n"] == 48
