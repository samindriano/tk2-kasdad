import numpy as np
import pandas as pd

from scripts.optimization_v3_sparse_classification_fullsix import (
    BLEND_WEIGHT,
    confidence_groups,
    evaluate_rows,
    metric,
)


def test_fixed_blend_preserves_missing_branch_and_aligns_present_rows():
    frame = pd.DataFrame({
        "cycle_code": [56, 56, 57, 57, 58, 58, 59, 59],
        "signal_stream": ["a", None, "b", "", "c", "d", None, "e"],
        "text_load": [20, 21, 22, 23, 24, 25, 26, 27],
    })
    rows = np.arange(len(frame))
    missing = frame.signal_stream.isna().to_numpy() | frame.signal_stream.eq("").to_numpy()
    champion = np.array([0.1, 0.8, 0.3, 0.7, 0.2, 0.9, 0.6, 0.4])
    c2b = np.array([0.1, 0.2, 0.3, 0.4, 0.2, 0.9, 0.6, 0.4])
    raw = np.array([0.5, 0.2, 0.7, 0.4, 0.8, 0.1, 0.6, 0.3])
    blend = np.where(missing, champion, 0.75 * c2b + BLEND_WEIGHT * raw)
    np.testing.assert_array_equal(blend[missing], champion[missing])
    np.testing.assert_allclose(blend[~missing], 0.75 * c2b[~missing] + 0.25 * raw[~missing])
    actual = np.array([0, 1, 0, 1, 0, 1, 1, 0])
    view = evaluate_rows(frame, actual, rows, champion, blend, [21.5, 23.5, 25.5])
    assert view["subgroups"]["stream_missing"]["n"] == 3
    assert view["subgroups"]["stream_present"]["n"] == 5
    assert view["subgroups"]["text_load_q1_low_tail"]["n"] == 2
    assert "confidence_groups" in view["models"]["blend25"]


def test_metrics_include_classwise_rates_and_confusion_matrix():
    actual = np.array([0, 0, 1, 1])
    probability = np.array([0.1, 0.7, 0.6, 0.4])
    got = metric(actual, probability)
    assert got["confusion_matrix"] == [[1, 1], [1, 1]]
    assert got["macro_f1"] == 0.5
    assert got["per_class"][0]["precision"] == 0.5
    assert got["per_class"][1]["recall"] == 0.5


def test_confidence_groups_are_fixed_probability_bands():
    actual = np.array([0, 1, 0, 1, 0])
    probability = np.array([0.50, 0.61, 0.21, 0.80, 0.95])
    got = confidence_groups(actual, probability)
    assert got["low_below_0_60"]["n"] == 1
    assert got["medium_0_60_to_0_80"]["n"] == 2
    assert got["high_at_least_0_80"]["n"] == 2
    assert got["high_at_least_0_80"]["accuracy"] == 0.5
