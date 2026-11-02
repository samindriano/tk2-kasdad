import numpy as np
from scipy import sparse

from scripts.optimization_v3_classification_next_v1 import (
    _payload, _route, _synthetic_lightgbm_native_nan_smoke, _token_features,
    _ttr_gate, run_synthetic_e2e,
)


def test_ttr_tokenizer_and_undefined_values_are_frozen():
    count, unique, ttr = _token_features(["A a B-1 1", "!!!", None, np.nan])
    np.testing.assert_array_equal(count, [5, 0, 0, 0])
    np.testing.assert_array_equal(unique, [3, 0, 0, 0])
    assert ttr[0] == 3 / 5
    assert np.isnan(ttr[1:]).all()


def test_sparse_ttr_column_preserves_nan_for_native_missing_handling():
    count, _unique, ttr = _token_features(["", "only", "only only"])
    matrix = sparse.csr_matrix(np.column_stack([count.astype(float), ttr]))
    assert matrix.shape == (3, 2)
    assert matrix.dtype == np.float64
    assert np.isnan(matrix.data).sum() == 1
    assert np.isnan(matrix.data).any()


def test_fixed_routes_and_payload_have_probability_baseline():
    baseline = np.array([.5, .4, .8])
    raw = np.array([.9, .2, .1])
    c2b = np.array([.2, .6, .7])
    cb = np.array([.33, .44, .55])
    missing = np.array([False, True, False])
    c19 = _route("C19", baseline, raw, c2b, cb, missing)
    np.testing.assert_allclose(c19, [.6, .44, .625], rtol=0, atol=1e-15)
    ttr = _route("TTR_BOTH", baseline, raw, c2b, cb, missing)
    np.testing.assert_allclose(ttr, [.375, .44, .55], rtol=0, atol=1e-15)
    payload = _payload(candidate_kind="C19", rows=np.arange(3), ids=np.array(["a", "b", "c"]),
        cycles=np.array([56, 57, 58]), baseline=baseline, raw=raw, c2b=c2b, cb=cb,
        missing=missing, prediction=c19)
    np.testing.assert_array_equal(payload["baseline_prediction"], baseline)
    assert "probability" in payload["probability_field"]


def test_production_synthetic_end_to_end_persistence_resume_and_all_views():
    report = run_synthetic_e2e()
    assert report["post_bundle_failure_recovery"] == "PASS_NO_REFIT"
    assert report["independent_route_reconstruction"] == "PASS"
    assert report["all_view_guard_path"] == "PASS"
    assert report["lightgbm_sparse_native_nan_fit_predict"]["status"] == "PASS"
    assert report["fit_count"] == 6


def test_ttr_gate_uses_frozen_matched_macro_f1_contrast():
    metrics = {
        "primary_two_origin": {"gain_macro_f1": 0.0006},
        "H12_55": {"gain_macro_f1": 0.0}, "H12_67": {"gain_macro_f1": 0.0},
        "H12_pooled": {"gain_macro_f1": 0.0}, "W1": {"gain_macro_f1": 0.0},
        "W4": {"gain_macro_f1": 0.0},
    }
    matched = {name: {"macro_f1": 0.0002} for name in metrics}
    groups = [{"supported": True, "fails_0_01_f1": False, "fails_0_02_recall": False}]
    assert _ttr_gate(metrics, matched, groups)["status"] == "PASS_DIAGNOSTIC_GATE"
    matched["primary_two_origin"]["macro_f1"] = 0.000199
    assert _ttr_gate(metrics, matched, groups)["status"] == "FAIL_DIAGNOSTIC_GATE"


def test_lightgbm_accepts_explicit_nan_in_sparse_ttr_column():
    result = _synthetic_lightgbm_native_nan_smoke()
    assert result["status"] == "PASS"
    assert result["explicit_nan_values"] == 3
    assert result["official_fit_budget_consumed"] == 0
