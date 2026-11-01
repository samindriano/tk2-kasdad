from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.classification_models import _LR_PARAMS, _TFIDF_PARAMS
from src.models.optimization_v3_batch1_c14 import (
    CANDIDATE_WEIGHT,
    CARD_ID,
    FEATURE_NAMES,
    FROZEN_SPARSE_WEIGHT,
    LR_PARAMS,
    NUMERIC_TREE_FEATURES,
    PILOT_CUTOFFS,
    RandomPartitionTextClassifier,
    THRESHOLD,
    TFIDF_PARAMS,
    TREE_PARAMS,
    fit_predict,
)


def _synthetic_features(cycles: np.ndarray, start: int = 0) -> pd.DataFrame:
    row = np.arange(start, start + len(cycles))
    colors = ("red", "blue", "green", "yellow")
    tokens = ("amber", "birch", "cedar", "dahlia", "elm", "fir")
    frame = pd.DataFrame({
        "impact_load": (row % 17).astype(float),
        "entry_friction": (row % 2).astype(int),
        "cycle_code": cycles.astype(int),
        "scope_pressure": (row % 11).astype(float),
        "text_load": (row % 9).astype(float),
        "scope_trace": [str(value % 5) for value in row],
        "org_spread": (row % 4).astype(int),
        "module_spread": (row % 6).astype(int),
        "telemetry_flag": (row % 2).astype(int),
        "org_code": [f"org-{value % 7}" for value in row],
        "module_code": [f"module-{value % 8}" for value in row],
        "pattern_code": [f"pattern-{value % 3}" for value in row],
        "signal_stream": [
            None if value % 13 == 0 else
            f"alpha {colors[value % len(colors)]} gamma {tokens[value % len(tokens)]} omega"
            for value in row
        ],
    })
    return frame.loc[:, list(FEATURE_NAMES)]


def _controls(score_x: pd.DataFrame) -> dict[str, np.ndarray]:
    missing = score_x["signal_stream"].isna().to_numpy()
    n = len(score_x)
    c2b = np.full(n, 0.4, dtype=np.float64)
    cb = np.full(n, 0.7, dtype=np.float64)
    sparse_raw = np.full(n, 0.2, dtype=np.float64)
    baseline = np.where(
        missing, cb, 0.75 * c2b + FROZEN_SPARSE_WEIGHT * sparse_raw
    )
    return {
        "baseline_prediction": baseline,
        "baseline_sparse_raw": sparse_raw,
        "c2b_probability": c2b,
        "cb_probability": cb,
        "stream_missing": missing,
        "control_rows": score_x.index.to_numpy(dtype=np.int64),
    }


def test_c14_recipe_parameters_are_bound_to_canonical_sources() -> None:
    assert CARD_ID == "C14"
    assert PILOT_CUTOFFS == (55, 67)
    assert TREE_PARAMS == {
        "n_estimators": 64,
        "max_depth": 5,
        "min_samples_leaf": 30,
        "random_state": 42,
        "n_jobs": 2,
        "sparse_output": True,
    }
    assert LR_PARAMS == _LR_PARAMS
    assert LR_PARAMS == {
        "C": 1.0,
        "solver": "liblinear",
        "max_iter": 2000,
        "tol": 1e-4,
        "class_weight": None,
        "random_state": 42,
    }
    assert TFIDF_PARAMS == _TFIDF_PARAMS
    assert CANDIDATE_WEIGHT == 0.75
    assert FROZEN_SPARSE_WEIGHT == 0.25
    assert THRESHOLD == 0.50
    assert "record_id" not in FEATURE_NAMES
    assert "artifact_signal" not in FEATURE_NAMES
    assert "threat_rank" not in FEATURE_NAMES
    assert len(NUMERIC_TREE_FEATURES) == 36
    assert not any(name.endswith("__ordinal") for name in NUMERIC_TREE_FEATURES)
    assert {"org_code__missing", "module_code__missing", "pattern_code__missing"}.issubset(
        NUMERIC_TREE_FEATURES
    )


def test_c14_fit_only_model_predicts_finite_probabilities_and_records_feature_lineage() -> None:
    fit_cycles = 47 + np.arange(128, dtype=np.int64) % 9
    score_cycles = 56 + np.arange(32, dtype=np.int64) % 12
    fit_x = _synthetic_features(fit_cycles)
    score_x = _synthetic_features(score_cycles, start=len(fit_x))
    score_x.index = np.arange(len(fit_x), len(fit_x) + len(score_x), dtype=np.int64)
    y = (fit_x["impact_load"].to_numpy(dtype=int) % 2).astype(np.int8)

    model = RandomPartitionTextClassifier(55).fit(fit_x, y)
    probabilities = model.predict_probability(score_x)
    state = model.audit_state()
    repeat_model = RandomPartitionTextClassifier(55).fit(fit_x, y)
    repeat_probabilities = repeat_model.predict_probability(score_x)
    repeat_state = repeat_model.audit_state()

    assert probabilities.shape == (len(score_x),)
    assert np.isfinite(probabilities).all()
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    assert state["fit_rows"] == len(fit_x)
    assert state["cutoff"] == 55
    assert state["fit_cycle_max"] <= 55
    assert state["tree_input_width"] == len(state["numeric_tree_input_names"])
    assert tuple(state["numeric_tree_input_names"]) == NUMERIC_TREE_FEATURES
    assert state["leaf_design_shape"][0] == len(fit_x)
    assert state["leaf_design_nnz"] == len(fit_x) * TREE_PARAMS["n_estimators"]
    assert state["feature_block_order"] == [
        "random_tree_leaf_indicators", "word_tfidf", "categorical_one_hot",
        "signal_stream_missing_indicator",
    ]
    assert state["validation_data_passed_to_fit"] is False
    assert state["validation_labels_passed_to_fit"] is False
    assert state["test_data_used"] is False
    assert state["convergence_warning"] is False
    assert state["model_sha256"]
    assert repeat_state["leaf_design_sha256"] == state["leaf_design_sha256"]
    np.testing.assert_array_equal(repeat_probabilities, probabilities)


def test_c14_fit_predict_routes_missing_stream_and_blends_present_stream() -> None:
    fit_cycles = 47 + np.arange(128, dtype=np.int64) % 9
    score_cycles = 56 + np.arange(32, dtype=np.int64) % 12
    fit_x = _synthetic_features(fit_cycles)
    score_x = _synthetic_features(score_cycles, start=len(fit_x))
    score_x.index = np.arange(len(fit_x), len(fit_x) + len(score_x), dtype=np.int64)
    y = (fit_x["impact_load"].to_numpy(dtype=int) % 2).astype(np.int8)
    controls = _controls(score_x)

    result = fit_predict(fit_x, y, score_x, controls=controls, card_id="C14", cutoff=55)
    missing = controls["stream_missing"]

    assert result["fit_predictor_count"] == 1
    assert result["fit_metadata"]["score_labels_passed_to_adapter"] is False
    assert result["fit_metadata"]["threshold"] == THRESHOLD
    assert result["fit_metadata"]["candidate_weight_new_present_expert"] == 0.75
    assert result["fit_metadata"]["frozen_sparse_weight_present_expert"] == 0.25
    assert np.isfinite(result["prediction"]).all()
    np.testing.assert_array_equal(result["prediction"][missing], controls["cb_probability"][missing])
    np.testing.assert_allclose(
        result["prediction"][~missing],
        CANDIDATE_WEIGHT * result["raw_candidate"][~missing]
        + FROZEN_SPARSE_WEIGHT * controls["baseline_sparse_raw"][~missing],
        rtol=0,
        atol=0,
    )


def test_c14_rejects_bad_cutoff_rows_and_control_mismatch() -> None:
    fit_x = _synthetic_features(47 + np.arange(96, dtype=np.int64) % 9)
    y = (fit_x["impact_load"].to_numpy(dtype=int) % 2).astype(np.int8)
    score_x = _synthetic_features(56 + np.arange(24, dtype=np.int64) % 12, start=len(fit_x))
    score_x.index = np.arange(len(fit_x), len(fit_x) + len(score_x), dtype=np.int64)
    controls = _controls(score_x)

    with pytest.raises(ValueError, match="cutoffs 55/67"):
        fit_predict(fit_x, y, score_x, controls=controls, card_id="C14", cutoff=59)
    bad_cycles = fit_x.copy()
    bad_cycles.loc[bad_cycles.index[-1], "cycle_code"] = 56
    with pytest.raises(ValueError, match="row cycles"):
        fit_predict(bad_cycles, y, score_x, controls=controls, card_id="C14", cutoff=55)
    wrong_controls = dict(controls)
    wrong_controls["stream_missing"] = ~controls["stream_missing"]
    with pytest.raises(ValueError, match="missing-stream mask"):
        fit_predict(fit_x, y, score_x, controls=wrong_controls, card_id="C14", cutoff=55)


def test_c14_rejects_non_reconstructed_baseline_and_row_order() -> None:
    fit_x = _synthetic_features(47 + np.arange(96, dtype=np.int64) % 9)
    y = (fit_x["impact_load"].to_numpy(dtype=int) % 2).astype(np.int8)
    score_x = _synthetic_features(56 + np.arange(24, dtype=np.int64) % 12, start=len(fit_x))
    score_x.index = np.arange(len(fit_x), len(fit_x) + len(score_x), dtype=np.int64)
    controls = _controls(score_x)
    wrong_baseline = dict(controls)
    wrong_baseline["baseline_prediction"] = np.zeros(len(score_x), dtype=np.float64)
    with pytest.raises(ValueError, match="fixed sparse25 route"):
        fit_predict(fit_x, y, score_x, controls=wrong_baseline, card_id="C14", cutoff=55)

    wrong_rows = dict(controls)
    wrong_rows["control_rows"] = controls["control_rows"][::-1].copy()
    with pytest.raises(ValueError, match="increasing unique source positions"):
        fit_predict(fit_x, y, score_x, controls=wrong_rows, card_id="C14", cutoff=55)


def test_c14_rejects_training_rows_that_are_not_increasing_unique_positions() -> None:
    fit_x = _synthetic_features(47 + np.arange(96, dtype=np.int64) % 9)
    y = (fit_x["impact_load"].to_numpy(dtype=int) % 2).astype(np.int8)
    fit_x.index = np.arange(len(fit_x), dtype=np.int64)[::-1]
    model = RandomPartitionTextClassifier(55)
    with pytest.raises(ValueError, match="unique increasing Train positions"):
        model.fit(fit_x, y)


def test_c14_sparse_baseline_control_matches_frozen_sparse25_recipe() -> None:
    score_x = _synthetic_features(np.array([56, 56], dtype=np.int64))
    controls = _controls(score_x)
    missing = controls["stream_missing"]
    expected = np.where(
        missing, controls["cb_probability"],
        0.75 * controls["c2b_probability"]
        + FROZEN_SPARSE_WEIGHT * controls["baseline_sparse_raw"],
    )
    np.testing.assert_array_equal(controls["baseline_prediction"], expected)
