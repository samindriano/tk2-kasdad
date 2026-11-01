from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from scripts import optimization_v3_batch1_r07 as runner
from src.models.optimization_v3_batch1_r07 import FEATURE_NAMES, NMF_PARAMETERS


def test_r07_recipe_records_complete_json_safe_tfidf_nmf_and_lightgbm_parameters():
    recipe = runner._candidate_recipe()
    assert recipe["text_vectorizer"]["ngram_range"] == [1, 2]
    assert recipe["text_vectorizer"]["min_df"] == 2
    assert recipe["text_vectorizer"]["max_features"] == 20_000
    assert recipe["text_vectorizer"]["sublinear_tf"] is True
    assert recipe["text_vectorizer"]["dtype"] == "float64"
    assert recipe["nmf"]["n_components"] == 32
    assert recipe["nmf"]["init"] == "nndsvda"
    assert recipe["nmf"]["solver"] == "cd"
    assert recipe["nmf"]["beta_loss"] == "frobenius"
    assert recipe["nmf"]["max_iter"] == 200
    assert recipe["nmf"]["random_state"] == 42
    assert recipe["implementation_compatibility"]["scikit_learn_version"]
    assert "initialization values" in recipe["implementation_compatibility"]["coordinate_descent_layout_workaround"]
    assert "record_id" not in recipe["feature_allowlist"]
    assert "threat_rank" not in recipe["feature_allowlist"]
    assert "artifact_signal" not in recipe["feature_allowlist"]
    assert json.loads(json.dumps(recipe, allow_nan=False)) == recipe


def test_independent_rank50_and_r07_reconstruction_keep_exact_champion_policy():
    cycles = np.array([68, 68, 68, 68, 69, 69])
    baseline_sparse_raw = np.array([2.0, 2.0, 6.0, 8.0, 0.0, 10.0])
    ridge25 = np.array([20.0, 40.0, 60.0, 80.0, 25.0, 75.0])
    baseline = np.clip(0.75 * ridge25 + 0.25 * runner._independent_rank50(baseline_sparse_raw, cycles), 0, 100)
    raw = np.array([5.0, 5.0, 7.0, 9.0, -20.0, 120.0])
    expected = np.clip(0.75 * baseline + 0.25 * runner._independent_rank50(raw, cycles), 0, 100)
    arrays = {
        "raw_candidate": raw,
        "control__baseline_sparse_raw": baseline_sparse_raw,
        "control__ridge25_prediction": ridge25,
        "control__cycle_code": cycles,
        "baseline_prediction": baseline,
    }
    candidate, verified_baseline = runner.reconstruct_r07(arrays)
    np.testing.assert_allclose(verified_baseline, baseline, rtol=0, atol=1e-12)
    np.testing.assert_allclose(candidate, expected, rtol=0, atol=1e-12)
    assert np.all((candidate >= 0.0) & (candidate <= 100.0))


def test_control_views_preserve_exact_segment_order_for_interleaved_cycles():
    cycles = np.array([68, 56, 69, 57], dtype=np.int64)
    target = np.array([30.0, 10.0, 40.0, 20.0])
    frame = pd.DataFrame({"record_id": ["c/1", "a/1", "d/1", "b/1"]})
    controls = {
        55: {"rows": np.array([1, 3]), "baseline_prediction": np.array([10.0, 20.0])},
        67: {"rows": np.array([0, 2]), "baseline_prediction": np.array([30.0, 40.0])},
    }
    views = {"primary": [(55, 56, 57), (67, 68, 69)]}
    result = runner._control_views(frame, target, cycles, views, controls)
    assert result["primary"]["rows"] == 4
    assert result["primary"]["mae"] == 0.0
    controls[67]["rows"] = np.array([2, 0])
    with pytest.raises(runner.BatchIntegrityError, match="exact Train row order"):
        runner._control_views(frame, target, cycles, views, controls)


def _resource_frame() -> tuple[pd.DataFrame, np.ndarray]:
    cycles = np.array([55] * 50 + [67] * 30 + [68] * 20, dtype=np.int64)
    terms = [f"token{index}" for index in range(48)]
    streams = [" ".join(terms) + f" variation{i % 5}" for i in range(len(cycles))]
    streams[3] = " "
    streams[12] = None
    return pd.DataFrame({"signal_stream": streams}), cycles


def test_nmf_resource_preflight_measures_real_sparse_tfidf_without_model_fit(monkeypatch):
    frame, cycles = _resource_frame()
    monkeypatch.setattr(runner.psutil, "virtual_memory",
                        lambda: SimpleNamespace(available=16 * 1024**3))

    def forbidden_nmf_fit(self, *args, **kwargs):
        raise AssertionError("NMF must not fit during resource preflight")

    monkeypatch.setattr(NMF, "fit_transform", forbidden_nmf_fit)
    result = runner._resource_preflight(frame, cycles)
    assert result["status"] == "PASS"
    assert result["candidate_fit_performed"] is False
    assert result["nmf_fit_performed"] is False
    assert result["tfidf_input_sparse_only"] is True
    assert result["convergence_measured"] is False
    assert result["free_memory_reserve_bytes"] == 2 * 1024**3
    assert result["job_memory_cap_bytes"] == 6 * 1024**3
    assert result["concurrent_training_jobs"] == 1
    for cutoff, fit_rows, score_rows in (("55", 50, 30), ("67", 80, 20)):
        current = result["cutoffs"][cutoff]
        assert current["status"] == "PASS"
        assert current["fit_rows"] == fit_rows
        assert current["score_rows"] == score_rows
        assert current["tfidf_features"] >= int(NMF_PARAMETERS["n_components"])
        assert current["tfidf_nnz"] > 0
        assert current["score_tfidf_nnz"] > 0
        assert current["tfidf_csr_bytes_observed"] > 0
        assert current["score_tfidf_csr_bytes_observed"] > 0
        assert current["tfidf_nonnegative_and_finite"] is True
        assert current["nmf_support_sufficient"] is True
        assert current["estimated_peak_is_observed"] is False


def test_resource_preflight_blocks_below_memory_floor_before_tfidf_fit(monkeypatch):
    frame, cycles = _resource_frame()
    monkeypatch.setattr(runner.psutil, "virtual_memory",
                        lambda: SimpleNamespace(available=3 * 1024**3))

    def forbidden_vectorizer_fit(self, *args, **kwargs):
        raise AssertionError("TF-IDF should not fit below the resource floor")

    monkeypatch.setattr(TfidfVectorizer, "fit_transform", forbidden_vectorizer_fit)
    result = runner._resource_preflight(frame, cycles)
    assert result["status"] == "BLOCKED_RESOURCE"
    assert result["cutoffs"] == {}
    assert result["candidate_fit_performed"] is False
    assert result["nmf_fit_performed"] is False


def test_freeze_resource_identity_ignores_live_memory_and_timing_values():
    first = {
        "available_memory_bytes": 8_000,
        "allowed_memory_bytes": 6_000,
        "status": "PASS",
        "cutoffs": {
            "55": {
                "status": "PASS",
                "estimated_peak_bytes": 500,
                "tfidf_vectorizer_fit_seconds": 1.25,
                "tfidf_score_transform_seconds": 0.5,
            }
        },
    }
    second = {
        "available_memory_bytes": 10_000,
        "allowed_memory_bytes": 6_000,
        "status": "PASS",
        "cutoffs": {
            "55": {
                "status": "PASS",
                "estimated_peak_bytes": 500,
                "tfidf_vectorizer_fit_seconds": 1.75,
                "tfidf_score_transform_seconds": 0.75,
            }
        },
    }
    assert runner._stable_resource_identity(first) == runner._stable_resource_identity(second)


def test_r07_source_manifest_and_feature_allowlist_are_exact():
    assert all(path.is_file() for path in runner._source_files())
    assert "record_id" not in FEATURE_NAMES
    assert "threat_rank" not in FEATURE_NAMES
    assert "artifact_signal" not in FEATURE_NAMES
