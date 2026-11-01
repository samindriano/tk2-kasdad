from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from scripts import optimization_v3_batch1_r06 as runner
from src.models.optimization_v3_batch1_r06 import FEATURE_NAMES


def test_r06_recipe_records_full_effective_vectorizer_defaults():
    recipe = runner._candidate_recipe()
    params = recipe["text_vectorizer"]
    assert params["analyzer"] == "char_wb"
    assert params["ngram_range"] == [3, 5]
    assert params["min_df"] == 3
    assert params["max_features"] == 30_000
    assert params["sublinear_tf"] is True
    assert params["dtype"] == "float64"
    assert params["norm"] == "l2"
    assert params["use_idf"] is True
    assert params["lowercase"] is True
    assert "record_id" not in recipe["feature_allowlist"]
    assert "threat_rank" not in recipe["feature_allowlist"]
    json.dumps(recipe, allow_nan=False)


def test_independent_rank50_and_reconstruction_use_full_cycle_average_ties():
    cycles = np.array([68, 68, 68, 68, 69, 69])
    sparse_raw = np.array([2.0, 2.0, 6.0, 8.0, 0.0, 10.0])
    ridge = np.array([20.0, 40.0, 60.0, 80.0, 25.0, 75.0])
    baseline = np.clip(0.75 * ridge + 0.25 * runner._independent_rank50(sparse_raw, cycles), 0, 100)
    raw_candidate = np.array([5.0, 5.0, 7.0, 9.0, -20.0, 120.0])
    expected = np.clip(0.75 * baseline + 0.25 * runner._independent_rank50(raw_candidate, cycles), 0, 100)
    arrays = {
        "raw_candidate": raw_candidate,
        "control__baseline_sparse_raw": sparse_raw,
        "control__ridge25_prediction": ridge,
        "control__cycle_code": cycles,
        "baseline_prediction": baseline,
    }
    candidate, verified_baseline = runner.reconstruct_r06(arrays)
    np.testing.assert_allclose(verified_baseline, baseline, rtol=0, atol=1e-12)
    np.testing.assert_allclose(candidate, expected, rtol=0, atol=1e-12)
    assert np.all((candidate >= 0.0) & (candidate <= 100.0))


def test_control_view_reconstruction_requires_exact_ordered_train_rows():
    cycles = np.array([56, 57, 68, 69], dtype=np.int64)
    target = np.array([10.0, 20.0, 30.0, 40.0])
    frame = pd.DataFrame({"record_id": ["a/1", "b/2", "c/3", "d/4"]})
    controls = {
        55: {"rows": np.array([0, 1]), "baseline_prediction": np.array([11.0, 18.0])},
        67: {"rows": np.array([2, 3]), "baseline_prediction": np.array([29.0, 44.0])},
    }
    views = {"tiny": [(55, 56, 57), (67, 68, 69)]}
    result = runner._control_views(frame, target, cycles, views, controls)
    assert result["tiny"]["rows"] == 4
    assert result["tiny"]["mae"] == 2.0
    controls[67]["rows"] = np.array([3, 2])
    with np.testing.assert_raises_regex(runner.BatchIntegrityError, "exact Train row order"):
        runner._control_views(frame, target, cycles, views, controls)


def test_control_view_preserves_declared_segment_order_when_train_cycles_interleave():
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


def test_character_resource_preflight_counts_support_and_respects_queue_reserve(monkeypatch):
    texts = [f"sharedtoken alpha{i % 4} beta{i % 7} unique{i}" for i in range(100)]
    frame = pd.DataFrame({"signal_stream": texts})
    cycles = np.array([55] * 80 + [67] * 20, dtype=np.int64)
    monkeypatch.setattr(runner.psutil, "virtual_memory",
                        lambda: SimpleNamespace(available=16 * 1024**3))
    result = runner._resource_preflight(frame, cycles)
    assert result["status"] == "PASS"
    assert result["free_memory_reserve_bytes"] == 2 * 1024**3
    assert result["job_memory_cap_bytes"] == 6 * 1024**3
    assert result["concurrent_training_jobs"] == 1
    assert result["cutoffs"]["55"]["fit_rows"] == 80
    assert result["cutoffs"]["67"]["fit_rows"] == 100
    assert result["cutoffs"]["67"]["eligible_char_ngrams_at_min_df"] > 0
    assert result["cutoffs"]["67"]["estimated_peak_is_observed"] is False
    assert result["dense_full_text_matrix"] is False


def test_source_manifest_only_names_present_files_and_protected_features_are_absent():
    assert all(path.is_file() for path in runner._source_files())
    assert "record_id" not in FEATURE_NAMES
    assert "threat_rank" not in FEATURE_NAMES
    assert "artifact_signal" not in FEATURE_NAMES
