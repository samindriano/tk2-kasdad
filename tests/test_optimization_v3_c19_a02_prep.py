from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.classification_c19_fulltrain import C19FullTrainClassifier
from src.models.optimization_v3_batch1_c19 import FEATURE_NAMES
from scripts.optimization_v3_c19_benchmark import _route_c19_probability
from scripts.optimization_v3_a02_synthetic import run_synthetic
from scripts.optimization_v3_a02_neighbors import CACHE_SCORE_CYCLE_WINDOWS


def test_c19_fulltrain_wrapper_changes_only_accepted_cutoff():
    model = C19FullTrainClassifier()
    assert model.cutoff == 79
    assert model.__class__.__mro__[1].__name__ == "XGBoostSVD32Classifier"


def test_a02_cache_cycle_windows_match_persisted_oof_blocks():
    assert CACHE_SCORE_CYCLE_WINDOWS == {
        55: (56, 67), 59: (60, 63), 63: (64, 67),
        67: (68, 79), 71: (72, 75), 75: (76, 79),
    }


def test_c19_route_has_exact_missing_fallback_and_fixed_present_blend():
    baseline = np.array([0.2, 0.7, 0.4, 0.8], dtype=np.float64)
    raw = np.array([0.8, 0.3, 0.1, 0.2], dtype=np.float64)
    missing = np.array([0.31, 0.42, 0.53, 0.64], dtype=np.float64)
    mask = np.array([False, True, False, True])
    actual = _route_c19_probability(baseline, raw, missing, mask)
    expected = missing.copy()
    expected[~mask] = 0.75 * baseline[~mask] + 0.25 * raw[~mask]
    np.testing.assert_array_equal(actual, expected)


def test_c19_synthetic_fulltrain_fit_and_prediction_without_official_data():
    rng = np.random.default_rng(19)
    n = 120
    cycle = np.resize(np.arange(47, 80, dtype=np.int64), n)
    data: dict[str, object] = {
        "impact_load": rng.normal(110.0, 8.0, n),
        "entry_friction": rng.integers(0, 2, n),
        "cycle_code": cycle,
        "scope_pressure": rng.normal(105.0, 12.0, n),
        "text_load": rng.normal(30.0, 5.0, n),
        "scope_trace": rng.integers(0, 5, n),
        "org_spread": rng.integers(1, 5, n),
        "module_spread": rng.integers(1, 8, n),
        "telemetry_flag": rng.integers(0, 2, n),
        "org_code": [f"org-{i % 6}" for i in range(n)],
        "module_code": [f"module-{i % 9}" for i in range(n)],
        "pattern_code": [f"pattern-{i % 5}" for i in range(n)],
        "signal_stream": [
            "anchor alpha" if i % 7 == 0 else f"anchor token{i % 50} detail{i % 17} extra{i % 31}"
            for i in range(n)
        ],
    }
    frame = pd.DataFrame(data, columns=list(FEATURE_NAMES))
    target = np.tile(np.array([0, 1], dtype=np.int8), n // 2)
    model = C19FullTrainClassifier().fit(frame, target)
    state = model.audit_state()
    assert state["fit_rows"] == n
    assert state["cutoff"] == 79
    assert state["test_data_used"] is False
    probability = model.predict_probability(frame.iloc[:8].copy())
    assert probability.shape == (8,)
    assert np.isfinite(probability).all()
    assert ((probability >= 0.0) & (probability <= 1.0)).all()


def test_a02_synthetic_end_to_end_persistence_and_independent_reconstruction():
    result = run_synthetic()
    assert result["status"] == "PASS"
    assert result["fit_type"] == "SYNTHETIC_ONLY"
    assert result["official_fit_count"] == 0
    assert result["checks"]["resume_reused_fitted_model_without_refit"]
    assert result["checks"]["probability_reconstruction_bit_exact"]
