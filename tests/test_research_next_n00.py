import numpy as np
import pandas as pd

from scripts.optimization_v3_research_next_n00 import (
    _report_candidate,
    rank50_independent,
    reconstruct_classification,
    reconstruct_regression,
)


def test_rank50_uses_average_ties_within_each_cycle():
    raw = np.array([10.0, 10.0, 20.0, 100.0])
    cycles = np.array([47, 47, 47, 48])
    actual = rank50_independent(raw, cycles)
    expected = np.array([30.0, 30.0, 60.0, 100.0])
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


def test_classification_reconstruction_checks_availability_routing():
    rows = np.array([0, 1], dtype=np.int64)
    raw = np.array([0.8, 0.5])
    missing = np.array([False, True])
    c2b = np.array([0.2, 0.8])
    cb = np.array([0.7, 0.9])
    sparse = np.array([0.4, 0.3])
    baseline = np.where(missing, cb, 0.75 * c2b + 0.25 * sparse)
    prediction = np.where(missing, cb, 0.75 * c2b + 0.25 * raw)
    bundle = {
        "card": "C16",
        "cutoff": 55,
        "rows": rows,
        "cycles": np.array([56, 57]),
        "arrays": {
            "control__stream_missing": missing,
            "control__c2b_probability": c2b,
            "control__cb_probability": cb,
            "control__baseline_sparse_raw": sparse,
            "raw_candidate": raw,
            "baseline_prediction": baseline,
            "prediction": prediction,
        },
    }
    reconstructed = reconstruct_classification(bundle, missing)
    np.testing.assert_allclose(reconstructed["control"], baseline)
    np.testing.assert_allclose(reconstructed["prediction"], prediction)


def test_regression_reconstruction_accepts_historical_baseline_aliases():
    raw = np.array([7.0, 7.0, 10.0])
    cycles = np.array([56, 56, 57])
    legacy = np.array([60.0, 70.0, 80.0])
    rank50 = rank50_independent(raw, cycles)
    prediction = np.clip(0.75 * legacy + 0.25 * rank50, 0.0, 100.0)

    for alias in ("baseline_prediction", "control__baseline_prediction"):
        arrays = {
            "raw_candidate": raw,
            "prediction": prediction,
            "baseline_prediction": legacy,
            alias: legacy,
        }
        reconstructed = reconstruct_regression({
            "card": "R21",
            "cutoff": 55,
            "rows": np.array([1, 2, 3]),
            "cycles": cycles,
            "arrays": arrays,
        })
        np.testing.assert_allclose(reconstructed["prediction"], prediction)
        np.testing.assert_allclose(reconstructed["legacy"], legacy)


def test_regression_h12_reference_and_legacy_are_matched_by_fit_origin():
    frame = pd.DataFrame({
        "cycle_code": [47, 55, 56, 59, 60],
        "entry_friction": [0, 1, 0, 1, 0],
        "signal_stream": ["a", "b", "c", "d", "e"],
        "text_load": [1, 2, 3, 4, 5],
        "impact_load": [10, 20, 30, 40, 50],
    })
    candidate = {
        55: {"rows": np.array([2]), "cycles": np.array([56]), "origin": np.array([55]),
             "prediction": np.array([20.0]), "control": np.array([12.0]), "legacy": np.array([0.0])},
        59: {"rows": np.array([4]), "cycles": np.array([60]), "origin": np.array([59]),
             "prediction": np.array([30.0]), "control": np.array([12.0]), "legacy": np.array([0.0])},
    }
    reference = {
        55: {"rows": np.array([2]), "cycles": np.array([56]), "origin": np.array([55]),
             "prediction": np.array([15.0]), "legacy": np.array([2.0])},
        59: {"rows": np.array([4]), "cycles": np.array([60]), "origin": np.array([59]),
             "prediction": np.array([25.0]), "legacy": np.array([3.0])},
    }
    result = _report_candidate(
        "regression", "R21", candidate, {"h12": [(55, 56, 56), (59, 60, 60)]},
        frame, np.array([0.0, 0.0, 10.0, 0.0, 20.0]), {2: 999.0, 4: 999.0},
        {2: 888.0, 4: 888.0}, [], [],
        reference_records_by_origin=reference,
        legacy_records_by_origin=reference,
    )
    view = result["views"]["h12"]
    assert view["candidate"]["mae"] == 10.0
    assert view["control"]["mae"] == 5.0
    assert view["protected_legacy"]["mae"] == 12.5
