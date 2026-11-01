import numpy as np
import pandas as pd
import pytest

from src.models.optimization_v3_batch1_c15 import (
    clipped_logits,
    fit_predict,
    two_score_design,
)


def _controls(frame, *, baseline_sparse, c2b, cb, text_bin):
    missing = frame["signal_stream"].astype(str).str.strip().eq("").to_numpy()
    baseline = np.where(missing, cb, 0.75 * c2b + 0.25 * baseline_sparse)
    return {
        "baseline_prediction": baseline,
        "baseline_sparse_raw": baseline_sparse,
        "c2b_probability": c2b,
        "cb_probability": cb,
        "stream_missing": missing,
        "control_rows": frame.index.to_numpy(dtype=np.int64),
        "text_load_bin": text_bin,
    }


def test_clipped_logits_handle_extreme_probabilities_and_design_alignment():
    transformed = clipped_logits([0.0, 0.2, 0.8, 1.0])
    assert transformed.shape == (4,)
    assert np.isfinite(transformed).all()
    design = two_score_design([0.2, 0.8], [0.4, 0.6])
    assert design.shape == (2, 2)
    with pytest.raises(ValueError):
        two_score_design([0.2], [0.2, 0.3])


def test_early_origin_uses_exact_fallback_without_a_fit():
    fit = pd.DataFrame({"cycle_code": [47, 48], "signal_stream": ["a", "b"]}, index=[0, 1])
    score = pd.DataFrame({"cycle_code": [56, 57], "signal_stream": ["a", ""]}, index=[2, 3])
    y = np.array([0, 1], dtype=np.int8)
    sparse = np.array([0.4, 0.7])
    c2b = np.array([0.3, 0.8])
    cb = np.array([0.6, 0.2])
    controls = _controls(score, baseline_sparse=sparse, c2b=c2b, cb=cb, text_bin=np.array([1, 2]))
    result = fit_predict(fit, y, score, controls=controls, card_id="C15", cutoff=55,
                         oof={})
    np.testing.assert_array_equal(result["prediction"], controls["baseline_prediction"])
    assert result["fit_predictor_count"] == 0


def test_present_only_meta_head_and_missing_catboost_route():
    n_oof = 6_000
    oof_rows = np.arange(n_oof, dtype=np.int64)
    cycles = np.tile(np.arange(56, 68), 500)
    origin = np.where(cycles < 60, 55, np.where(cycles < 64, 59, 63))
    labels = (np.arange(n_oof) % 2).astype(np.int8)
    sparse_oof = np.where(labels == 1, 0.72, 0.28)
    c2b_oof = np.where(labels == 1, 0.68, 0.32)
    missing_oof = np.zeros(n_oof, dtype=bool)
    oof = {
        "rows": oof_rows,
        "cycles": cycles,
        "origin_cutoff": origin,
        "labels": labels,
        "c2b_probability": c2b_oof,
        "sparse_probability": sparse_oof,
        "stream_missing": missing_oof,
        "signal_stream": np.array(["sample"] * n_oof, dtype=object),
    }
    fit = pd.DataFrame({"cycle_code": np.full(n_oof, 67),
                        "signal_stream": np.array(["fit"] * n_oof)}, index=oof_rows)
    fit["cycle_code"] = cycles
    score = pd.DataFrame({"cycle_code": [68, 69, 70],
                          "signal_stream": ["score", "", "score"]}, index=[n_oof, n_oof + 1, n_oof + 2])
    score_sparse = np.array([0.4, 0.55, 0.6])
    score_c2b = np.array([0.45, 0.55, 0.6])
    score_cb = np.array([0.8, 0.3, 0.2])
    controls = _controls(score, baseline_sparse=score_sparse, c2b=score_c2b,
                         cb=score_cb, text_bin=np.array([0, 1, 2]))
    result = fit_predict(fit, labels, score, controls=controls, card_id="C15", cutoff=67,
                         oof=oof)
    assert result["fit_predictor_count"] == 1
    assert result["state"]["oof_present_fit_rows"] == n_oof
    assert result["prediction"].shape == (3,)
    assert result["prediction"][1] == score_cb[1]
    assert result["prediction"][0] != controls["baseline_prediction"][0]


def test_oof_chronology_is_enforced():
    n = 5_000
    cycles = np.full(n, 68, dtype=np.int64)
    rows = np.arange(n, dtype=np.int64)
    labels = (np.arange(n) % 2).astype(np.int8)
    oof = {
        "rows": rows,
        "cycles": cycles,
        "origin_cutoff": np.full(n, 68, dtype=np.int64),
        "labels": labels,
        "c2b_probability": np.full(n, 0.5),
        "sparse_probability": np.full(n, 0.5),
        "stream_missing": np.zeros(n, dtype=bool),
        "signal_stream": np.array(["x"] * n, dtype=object),
    }
    fit = pd.DataFrame({"cycle_code": cycles, "signal_stream": ["x"] * n}, index=rows)
    score = pd.DataFrame({"cycle_code": [76], "signal_stream": ["y"]}, index=[n])
    controls = _controls(score, baseline_sparse=np.array([0.5]), c2b=np.array([0.5]),
                         cb=np.array([0.5]), text_bin=np.array([1]))
    with pytest.raises(ValueError, match="prequential"):
        fit_predict(fit, labels, score, controls=controls, card_id="C15", cutoff=75,
                    oof=oof)
