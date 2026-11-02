import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import confusion_matrix, f1_score

from scripts.optimization_v3_a02_b01_centering import (
    _commit_output_bundle,
    compute_from_saved_geometry,
    fit_prefix_text_load_groups,
    _probability_metrics,
    write_once,
)


def _top32(rows, fill_value):
    rows = np.asarray(rows)
    return np.pad(rows, ((0, 0), (0, 32 - rows.shape[1])), constant_values=fill_value)


def _saved_geometry():
    return {
        "score_rows": np.array([100, 101, 102], dtype=np.int64),
        "baseline_prediction": np.array([0.40, 0.45, 0.50], dtype=np.float64),
        "neighbor_source_rows": _top32([[10, 11], [10, -1], [-1, -1]], -1).astype(np.int64),
        "neighbor_similarities": _top32([[0.50, 0.25], [1.00, 0.0], [0.0, 0.0]], 0.0).astype(np.float64),
        "neighbor_count": np.array([2, 1, 0], dtype=np.int16),
        "missing_mask": np.array([False, False, False]),
    }


def _run(saved=None, bank_labels=None, assignment=None):
    return compute_from_saved_geometry(
        _saved_geometry() if saved is None else saved,
        np.array([10, 11], dtype=np.int64),
        np.array([1, 0], dtype=np.int8) if bank_labels is None else bank_labels,
        np.array([0.90, 0.20], dtype=np.float64),
        np.array([1, 0], dtype=np.int64) if assignment is None else assignment,
    )


def test_b01_is_shrunk_local_label_rate_and_keeps_original_a02_reconstruction():
    result = _run()
    mass = 0.50**2 + 0.25**2
    local_rate = (0.50**2 * 1 + 0.25**2 * 0) / mass
    pre_cap_b01 = (0.50**2 * (1 - 0.40) + 0.25**2 * (0 - 0.40)) / (4.0 + mass)
    convex_identity = (4.0 * 0.40 + mass * local_rate) / (4.0 + mass)

    assert result["b01_raw"][0] == pytest.approx(pre_cap_b01)
    assert 0.40 + result["b01_raw"][0] == pytest.approx(convex_identity)
    assert result["a02_raw"][0] == pytest.approx((0.50**2 * (1 - 0.90) + 0.25**2 * (0 - 0.20)) / (4.0 + mass))
    assert result["a02_rebuilt"][0] == pytest.approx(0.40 + result["a02_raw"][0])
    assert result["b01"][0] != result["a02_rebuilt"][0]


def test_label_shuffle_permutes_binary_bank_labels_with_assignment():
    result = _run()
    np.testing.assert_array_equal(result["permuted_bank_labels"], np.array([0, 1], dtype=np.int8))
    mass = 0.50**2 + 0.25**2
    expected_raw = (0.50**2 * (0 - 0.40) + 0.25**2 * (1 - 0.40)) / (4.0 + mass)
    assert result["b01_label_shuffle_raw"][0] == pytest.approx(expected_raw)
    assert result["b01_label_shuffle"][0] == pytest.approx(0.40 + expected_raw)


def test_cap_zero_neighbors_missing_fallback_and_threshold_tie():
    saved = {
        "score_rows": np.array([100, 101, 102], dtype=np.int64),
        "baseline_prediction": np.array([0.10, 0.50, 0.40]),
        "neighbor_source_rows": _top32([[10, 11], [-1, -1], [-1, -1]], -1),
        "neighbor_similarities": _top32([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]], 0.0),
        "neighbor_count": np.array([2, 0, 0]),
        "missing_mask": np.array([False, False, True]),
    }
    result = compute_from_saved_geometry(
        saved,
        np.array([10, 11]),
        np.array([1, 1], dtype=np.int8),
        np.array([0.5, 0.5]),
        np.array([1, 0]),
    )
    assert result["b01_raw"][0] == pytest.approx(0.30)
    assert result["b01"][0] == pytest.approx(0.15)  # +0.05 cap
    assert result["b01"][1] == 0.50  # exact zero-neighbor baseline fallback
    assert result["b01"][2] == 0.40  # missing-stream exact fallback
    assert bool(result["b01"][1] >= 0.50) is True


def test_query_label_perturbation_does_not_change_formula_output():
    saved = _saved_geometry()
    saved["query_y"] = np.array([0, 1, 0], dtype=np.int8)
    original = _run(saved=saved)
    perturbed = {key: value.copy() for key, value in saved.items()}
    perturbed["query_y"] = 1 - perturbed["query_y"]
    changed_truth = _run(saved=perturbed)
    for arm in ("a02_rebuilt", "a02_shuffle_rebuilt", "b01", "b01_label_shuffle"):
        np.testing.assert_array_equal(original[arm], changed_truth[arm])


def test_manual_confusion_macro_f1_and_positive_count_match_sklearn_at_ties():
    y = np.array([1, 0, 0, 1, 0], dtype=np.int8)
    p = np.array([0.50, 0.49, 0.80, 0.50, 0.10], dtype=np.float64)
    actual = _probability_metrics(y, p)
    hard = (p >= 0.50).astype(np.int8)
    cm = confusion_matrix(y, hard, labels=[0, 1])
    assert actual["predicted_positive_count"] == int(hard.sum()) == 3
    assert (actual["tn"], actual["fp"], actual["fn"], actual["tp"]) == (
        int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))
    assert actual["macro_f1"] == pytest.approx(f1_score(y, hard, labels=[0, 1], average="macro", zero_division=0))
    assert actual["roc_auc"] is not None and actual["average_precision"] is not None
    assert actual["logloss_clip_native_float64_epsilon"] >= 0
    assert actual["wrong_exact_zero_endpoints"] == actual["wrong_exact_one_endpoints"] == 0


def test_invalid_assignment_or_geometry_is_rejected():
    saved = _saved_geometry()
    with pytest.raises(ValueError, match="bank permutation"):
        compute_from_saved_geometry(saved, np.array([10, 11]), np.array([1, 0]),
                                   np.array([0.9, 0.2]), np.array([0, 0]))
    malformed = {key: value.copy() for key, value in saved.items()}
    malformed["neighbor_source_rows"][0, 2] = 12
    with pytest.raises(ValueError, match="padding"):
        _run(saved=malformed)


def test_fit_prefix_text_load_quartiles_reproduce_right_closed_edges():
    train = pd.DataFrame({
        "cycle_code": np.array([47, 48, 49, 50, 51, 52, 53, 54]),
        "text_load": np.array([1., 2., 3., 4., 1., 2., 3., 4.]),
    })
    assert fit_prefix_text_load_groups(train, 50, np.array([0, 1, 2, 3])).tolist() == [0, 1, 2, 3]


def test_completed_result_bundle_is_published_as_one_directory(tmp_path):
    result = {"value": 7}
    qa = {"status": "PASS", "result_sha256": None, "summary_sha256": None}
    summary = "# complete\n"

    bundle, written_qa = _commit_output_bundle(tmp_path, result, qa, summary)

    assert bundle == tmp_path / "B01_OUTPUTS"
    result_text = (bundle / "RESULTS.json").read_text(encoding="utf-8")
    summary_text = (bundle / "SUMMARY.md").read_text(encoding="utf-8")
    persisted_qa = json.loads((bundle / "QA.json").read_text(encoding="utf-8"))
    assert json.loads(result_text) == result
    assert summary_text == summary
    assert persisted_qa == written_qa
    assert persisted_qa["result_sha256"] == hashlib.sha256(result_text.encode("utf-8")).hexdigest()
    assert persisted_qa["summary_sha256"] == hashlib.sha256(summary_text.encode("utf-8")).hexdigest()
    assert not (tmp_path / ".B01_OUTPUTS.staging").exists()


def test_interrupted_staging_blocks_automatic_replacement(tmp_path):
    staging = tmp_path / ".B01_OUTPUTS.staging"
    staging.mkdir()
    with pytest.raises(RuntimeError, match="interrupted staging"):
        _commit_output_bundle(tmp_path, {"value": 1}, {}, "summary")
    assert staging.is_dir()
    assert not (tmp_path / "B01_OUTPUTS").exists()


def test_immutable_freeze_write_is_atomic_idempotent_and_fail_closed(tmp_path):
    path = tmp_path / "FREEZE.json"
    write_once(path, {"status": "frozen"})
    original = path.read_text(encoding="utf-8")

    write_once(path, {"status": "frozen"})
    assert path.read_text(encoding="utf-8") == original
    assert not (tmp_path / ".FREEZE.json.staging").exists()
    with pytest.raises(RuntimeError, match="nonidentical immutable"):
        write_once(path, {"status": "changed"})

    other = tmp_path / "FREEZE2.json"
    staging = tmp_path / ".FREEZE2.json.staging"
    staging.write_text("partial", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Interrupted immutable output staging"):
        write_once(other, {"status": "frozen"})
    assert not other.exists()
    assert staging.read_text(encoding="utf-8") == "partial"
