import inspect

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import confusion_matrix, f1_score

from scripts import optimization_v3_b02_plugin_allocation as b02


def test_soft_confusion_matches_hand_calculation_with_fixed_missing_label():
    result = b02._soft_confusion(
        np.array([0.8, 0.3]), np.array([False, True]), k=1, top_probability_sum=0.8)
    assert result["tp"] == pytest.approx(0.8)
    assert result["fp"] == pytest.approx(0.2)
    assert result["fn"] == pytest.approx(0.3)
    assert result["tn"] == pytest.approx(0.7)
    expected = 0.5 * (1.6 / 2.1 + 1.4 / 1.9)
    assert result["macro_f1"] == pytest.approx(expected)


def test_all_zero_and_all_one_scores_have_well_defined_decisions():
    ids = np.array(["a", "b"])
    all_zero = b02.select_k(np.array([0.0, 0.0]), np.array([False, False]), ids)
    all_one = b02.select_k(np.array([1.0, 1.0]), np.array([False, False]), ids)
    assert all_zero["k_present"] == 0
    assert np.array_equal(all_zero["prediction"], [0, 0])
    assert all_one["k_present"] == 2
    assert np.array_equal(all_one["prediction"], [1, 1])


def test_exact_probability_ties_are_never_split_and_id_order_does_not_change_labels():
    p = np.array([0.9, 0.9, 0.4, 0.4, 0.1])
    ids = np.array(["z", "a", "c", "b", "x"])
    result = b02.select_k(p, np.zeros(5, dtype=bool), ids)
    assert result["k_present"] in {0, 2, 4, 5}
    if result["selected_score_floor"] is not None and result["highest_excluded_score"] is not None:
        assert result["selected_score_floor"] > result["highest_excluded_score"]

    permutation = np.array([1, 0, 3, 2, 4])
    permuted = b02.select_k(p[permutation], np.zeros(5, dtype=bool), ids[permutation])
    by_id = dict(zip(ids, result["prediction"]))
    permuted_by_id = dict(zip(ids[permutation], permuted["prediction"]))
    assert by_id == permuted_by_id
    assert result["k_present"] == permuted["k_present"]


def test_missing_stream_labels_remain_at_source_half_threshold():
    p = np.array([0.6, 0.6, 0.9, 0.1])
    missing = np.array([False, False, True, True])
    result = b02.select_k(p, missing, np.array(["a", "b", "m1", "m0"]))
    assert np.array_equal(result["prediction"][missing], np.array([1, 0], dtype=np.int8))
    assert result["fixed_missing_positive_rows"] == 1
    assert result["eligible_present_rows"] == 2


def test_selector_has_no_target_argument_and_is_independent_of_external_y():
    signature = inspect.signature(b02.select_k)
    assert list(signature.parameters) == ["probabilities", "missing_mask", "stable_record_ids"]
    p = np.array([0.7, 0.51, 0.49, 0.2])
    missing = np.array([False, False, True, False])
    ids = np.array(["a", "b", "c", "d"])
    y_a = np.array([0, 1, 1, 0])
    y_b = 1 - y_a
    first = b02.select_k(p, missing, ids)
    # y values are deliberately not supplied to the selector.
    assert y_a.shape == y_b.shape
    second = b02.select_k(p, missing, ids)
    assert first["k_present"] == second["k_present"]
    np.testing.assert_array_equal(first["prediction"], second["prediction"])


def test_actual_macro_f1_confusion_matches_independent_sklearn_replay():
    y = np.array([0, 0, 1, 1, 1, 0], dtype=np.int8)
    pred = np.array([1, 0, 1, 0, 0, 1], dtype=np.int8)
    result = b02._actual_metrics(y, pred)
    expected_cm = confusion_matrix(y, pred, labels=[0, 1])
    assert result["confusion"] == {
        "tn": int(expected_cm[0, 0]), "fp": int(expected_cm[0, 1]),
        "fn": int(expected_cm[1, 0]), "tp": int(expected_cm[1, 1]),
    }
    assert result["macro_f1"] == pytest.approx(
        f1_score(y, pred, average="macro", labels=[0, 1], zero_division=0))


def test_selected_bank_labels_skip_nonbank_bad_label(monkeypatch, tmp_path):
    path = tmp_path / "train.csv"
    path.write_text(
        "record_id,cycle_code,signal_stream,artifact_signal\n"
        "bank-a,56,alpha,1\nquery-row,60,query,NOT_A_BANK_LABEL\nbank-b,57,beta,0\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(b02, "TRAIN_PATH", path)
    features = pd.DataFrame({"record_id": ["bank-a", "query-row", "bank-b"],
                             "cycle_code": [56, 60, 57],
                             "signal_stream": ["alpha", "query", "beta"]})
    labels = b02.b01_benchmark_label_reader(
        np.array([0, 2]), np.array(["bank-a", "bank-b"]),
        np.array([56, 57]), features)
    np.testing.assert_array_equal(labels, np.array([1, 0], dtype=np.int8))
    assert "artifact_signal" not in features.columns


def test_selected_query_labels_are_read_only_for_requested_rows(monkeypatch, tmp_path):
    path = tmp_path / "train.csv"
    path.write_text(
        "record_id,artifact_signal\nquery-a,0\nnot-requested,NOT_A_LABEL\nquery-b,1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(b02, "TRAIN_PATH", path)
    features = pd.DataFrame({"record_id": ["query-a", "not-requested", "query-b"]})
    labels = b02._selected_train_labels(np.array([0, 2]), np.array(["query-a", "query-b"]), features)
    np.testing.assert_array_equal(labels, np.array([0, 1], dtype=np.int8))


def test_npz_reader_never_returns_saved_query_labels(tmp_path):
    path = tmp_path / "predictions.npz"
    arrays = {key: np.array([0]) for key in b02.PREDICTION_KEYS}
    arrays["score_record_ids"] = np.array(["id"])
    arrays["neighbor_source_rows"] = np.zeros((1, 32), dtype=np.int64)
    arrays["neighbor_ids"] = np.array([["id"] * 32])
    arrays["neighbor_similarities"] = np.zeros((1, 32))
    arrays["labels"] = np.array(["DO_NOT_LOAD"])
    np.savez_compressed(path, **arrays)
    loaded = b02._load_npz_without_targets(path)
    assert "labels" not in loaded
    assert set(loaded) == set(b02.PREDICTION_LOAD_KEYS)


def test_bank_origin_uses_frozen_neighbor_bank_for_e67(monkeypatch):
    train = pd.DataFrame({"cycle_code": [56, 57], "record_id": ["a", "b"]})
    expected = {"rows": np.array([0, 1]), "missing": np.array([False, False])}
    calls = []
    monkeypatch.setattr(b02.a02_neighbors, "_build_bank", lambda origin, caches, frame: (calls.append(origin) or expected))
    monkeypatch.setattr(b02.a02, "_build_bank", lambda *args: pytest.fail("continuation bank used for E67"))
    bank = b02._build_oof_bank(
        67, {}, train, {},
        {"task": "classification", "origins": [67, 75],
         "bank": {"E67": {"rows": 2, "present_rows": 2}}})
    assert calls == [67]
    assert bank is expected


def test_bank_origin_rejects_mismatch_with_frozen_neighbor_support(monkeypatch):
    monkeypatch.setattr(
        b02.a02_neighbors, "_build_bank",
        lambda origin, caches, train: {
            "rows": np.array([0, 1]), "missing": np.array([False, False])})
    with pytest.raises(RuntimeError, match="counts differ"):
        b02._build_oof_bank(
            67, {}, pd.DataFrame(), {},
            {"task": "classification", "origins": [67, 75],
             "bank": {"E67": {"rows": 3, "present_rows": 3}}})


def test_atomic_output_bundle_is_idempotent_and_refuses_different_bytes(monkeypatch, tmp_path):
    final = tmp_path / "B02_OUTPUTS"
    staging = tmp_path / ".B02_OUTPUTS.staging"
    monkeypatch.setattr(b02, "OUTPUT_DIR", final)
    monkeypatch.setattr(b02, "STAGING_DIR", staging)
    bundle = {"RESULTS.json": b"{}\n", "QA.json": b"{}\n"}
    b02._publish_bundle(bundle)
    b02._publish_bundle(bundle)
    assert (final / "RESULTS.json").read_bytes() == bundle["RESULTS.json"]
    with pytest.raises(RuntimeError, match="refusing overwrite"):
        b02._publish_bundle({"RESULTS.json": b"different", "QA.json": b"{}\n"})
