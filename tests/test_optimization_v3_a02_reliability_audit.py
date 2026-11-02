import numpy as np
import pandas as pd
import pytest

from scripts.optimization_v3_a02_reliability_audit import (
    FEATURE_KEYS,
    _confusion,
    _fixed_bins,
    _group_row,
    additive_diagnostics,
    probability_metrics,
    verify_row_join,
)


def test_confusion_macro_f1_and_probability_metrics_are_recomputed_from_probabilities():
    y = np.array([0, 0, 1, 1], dtype=np.int8)
    p = np.array([0.2, 0.7, 0.8, 0.3], dtype=np.float64)

    metrics = _confusion(y, p)
    probs = probability_metrics(y, p)

    assert (metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]) == (1, 1, 1, 1)
    assert metrics["macro_f1"] == pytest.approx(0.5)
    assert metrics["classification_error"] == pytest.approx(0.5)
    assert probs["brier"] == pytest.approx(0.265)
    assert probs["roc_auc"] == pytest.approx(0.75)
    assert np.isfinite(probs["logloss_e15"])
    assert np.isfinite(probs["logloss_native_float64_epsilon"])


def test_verify_row_join_uses_exact_ids_and_cycles_not_temporal_row_order():
    train = pd.DataFrame({
        "record_id": ["r0", "r1", "r2"],
        "cycle_code": [79, 47, 66],
        "artifact_signal": np.array([1, 0, 1], dtype=np.int8),
    })
    truth = verify_row_join(train, np.array([0, 2]), np.array(["r0", "r2"]), np.array([79, 66]))
    np.testing.assert_array_equal(truth, np.array([1, 1], dtype=np.int8))


def test_verify_row_join_rejects_identifier_or_cycle_mismatch():
    train = pd.DataFrame({"record_id": ["r0", "r1"], "cycle_code": [50, 51],
                         "artifact_signal": np.array([0, 1], dtype=np.int8)})
    with pytest.raises(ValueError, match="record_id mismatch"):
        verify_row_join(train, [0, 1], ["r0", "wrong"], [50, 51])
    with pytest.raises(ValueError, match="cycle_code mismatch"):
        verify_row_join(train, [0, 1], ["r0", "r1"], [51, 50])


def test_empty_group_is_explicitly_unsupported_without_divide_by_zero():
    frame = {
        "y": np.array([], dtype=np.int8),
        "cycles": np.array([], dtype=np.int16),
        "origin": np.array([], dtype=np.int16),
        "baseline": np.array([], dtype=np.float64),
        "observed": np.array([], dtype=np.float64),
        "shuffled": np.array([], dtype=np.float64),
    }
    for key in FEATURE_KEYS:
        frame[key] = np.array([], dtype=bool if key in {
            "cap_hit_or_boundary", "strict_clip_hit", "zero_tfidf_vector", "no_positive_weights",
            "single_effective_neighbor", "selected_similarity_tie", "saved_top32_boundary_tie"
        } else np.float64)

    result = _group_row("synthetic", frame, np.array([], dtype=bool), "test_dimension", "EMPTY")

    assert result["n"] == 0
    assert result["support_f1"] is False
    assert result["support_reason"] == "EMPTY"
    assert result["observed_macro_f1"] is None
    assert result["brier_improvement_vs_sparse25"] is None


def test_fixed_bins_preserve_left_closed_edges_and_missing_category():
    values = np.array([0.0, 0.2, 0.4, 0.8, 1.0, np.nan])
    labels = _fixed_bins(values, [0, .2, .4, .6, .8, 1],
                         ["a", "b", "c", "d", "e"], "NO_NEIGHBOR")
    assert labels.tolist() == ["a", "b", "c", "e", "e", "NO_NEIGHBOR"]


def test_additive_endpoint_diagnostic_separates_wrong_extreme_confidence():
    y = np.array([1, 0, 1, 0], dtype=np.int8)
    base = np.array([0.8, 0.2, 0.4, 0.6])
    observed = np.array([0.0, 1.0, 0.55, 0.4])
    shuffled = np.array([0.75, 0.25, 0.45, 0.55])
    frame = {"y": y, "baseline": base, "observed": observed, "shuffled": shuffled}

    result = additive_diagnostics(frame)

    assert result["candidate_exact_endpoint_rows"] == 2
    assert result["candidate_wrong_exact_endpoint_rows"] == 2
    assert result["wrong_endpoint_breakdown"] == {"p0_y1": 1, "p1_y0": 1}
    assert result["wrong_endpoint"]["rows"] == 2
    assert result["all_other_rows"]["rows"] == 2
    assert (result["wrong_endpoint"]["logloss_improvement_vs_sparse25_e15_contribution_to_view_mean"]
            != result["wrong_endpoint"]["logloss_improvement_vs_sparse25_native_epsilon_contribution_to_view_mean"])
