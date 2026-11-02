import hashlib
import inspect

import numpy as np
import pandas as pd
import pytest

from scripts import optimization_v3_b01_benchmark as benchmark


def _synthetic_inputs():
    bank_rows = np.array([10, 11, 12], dtype=np.int64)
    baseline = np.array([0.40, 0.55, 0.25], dtype=np.float64)
    neighbors = np.full((3, 32), -1, dtype=np.int64)
    neighbors[0, :2] = [10, 11]
    neighbors[2, :1] = [12]
    similarities = np.zeros((3, 32), dtype=np.float64)
    similarities[0, :2] = [0.8, 0.4]
    similarities[2, 0] = 0.75
    return {
        "baseline_probability": baseline,
        "prediction": {
            "neighbor_source_rows": neighbors,
            "neighbor_similarities": similarities,
            "neighbor_count": np.array([2, 0, 1], dtype=np.int16),
            "missing_mask": np.array([False, False, True]),
        },
        "bank_rows": bank_rows,
        "bank_labels": np.array([1, 0, 1], dtype=np.int8),
    }


def test_independent_convex_formula_and_exact_fallbacks():
    data = _synthetic_inputs()
    similarities = data["prediction"]["neighbor_similarities"][0, :2]
    weights = similarities**2
    local_rate = float(np.dot(weights, data["bank_labels"][:2]) / weights.sum())
    shrunk = (4.0 * data["baseline_probability"][0] + weights.sum() * local_rate) / (4.0 + weights.sum())
    correction = np.clip(shrunk - data["baseline_probability"][0], -0.05, 0.05)
    expected = np.clip(data["baseline_probability"][0] + correction, 0.0, 1.0)
    b01_probability = data["baseline_probability"].copy()
    b01_probability[0] = np.clip(expected, 0.0, 1.0)

    checked = benchmark._independent_b01_verify(data, b01_probability)

    assert checked["status"] == "PASS"
    assert checked["max_abs_probability_difference"] <= 1e-15
    assert b01_probability[1] == data["baseline_probability"][1]  # zero-neighbor fallback
    assert b01_probability[2] == data["baseline_probability"][2]  # missing-stream fallback


def test_formula_verifier_does_not_accept_modified_probability():
    data = _synthetic_inputs()
    with pytest.raises(RuntimeError, match="Independent convex B01 formula"):
        benchmark._independent_b01_verify(data, np.array([0.9, 0.55, 0.25]))


def test_train_label_loader_selects_only_verified_bank_rows(monkeypatch, tmp_path):
    train_path = tmp_path / "tiny_train.csv"
    train_path.write_text(
        "record_id,cycle_code,signal_stream,artifact_signal\n"
        "row-a,56,alpha,1\n"
        "query-row,64,query text,NOT_A_BANK_LABEL\n"
        "row-b,79,omega,0\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(benchmark, "TRAIN_PATH", train_path)
    features_only = pd.DataFrame({"record_id": ["row-a", "query-row", "row-b"],
                                  "cycle_code": [56, 64, 79],
                                  "signal_stream": ["alpha", "query text", "omega"]})
    labels = benchmark._load_train_bank_labels_only(
        np.array([0, 2]), np.array(["row-a", "row-b"]), np.array([56, 79]), features_only)
    np.testing.assert_array_equal(labels, np.array([1, 0], dtype=np.int8))
    assert "artifact_signal" not in features_only.columns


def test_test_and_sample_schema_gate_rejects_any_test_target_column():
    benchmark._validate_input_schemas(
        ["record_id", "cycle_code", "signal_stream", "entry_friction"],
        ["record_id", "artifact_signal"],
    )
    with pytest.raises(RuntimeError, match="target column"):
        benchmark._validate_input_schemas(
            ["record_id", "cycle_code", "signal_stream", "artifact_signal"],
            ["record_id", "artifact_signal"],
        )
    with pytest.raises(RuntimeError, match="SampleSubmission schema"):
        benchmark._validate_input_schemas(
            ["record_id", "cycle_code", "signal_stream"],
            ["record_id", "threat_rank"],
        )


def test_transition_counts_are_predictions_only_and_sum_exactly():
    a02 = np.array([0, 0, 1, 1, 0, 1], dtype=np.int8)
    b01 = np.array([1, 0, 0, 1, 0, 0], dtype=np.int8)
    cycles = np.array([80, 80, 80, 81, 81, 81], dtype=np.int16)
    missing = np.array([False, True, False, False, True, False])
    report = benchmark._transition_summary(a02, b01, cycles, missing)
    overall = report["overall"]
    assert overall == {"a02_0_to_b01_1": 1, "a02_1_to_b01_0": 2,
                       "unchanged": 3, "changed": 3}
    assert sum(x["rows"] for x in report["by_cycle_and_availability"]) == len(a02)
    assert sum(x["a02_0_to_b01_1"] for x in report["by_cycle_and_availability"]) == 1
    assert sum(x["a02_1_to_b01_0"] for x in report["by_cycle_and_availability"]) == 2


def test_atomic_output_writer_is_idempotent_and_blocks_partial_stage(tmp_path):
    path = tmp_path / "candidate.csv"
    payload = b"record_id,artifact_signal\nx,1\n"
    benchmark._atomic_once(path, payload)
    benchmark._atomic_once(path, payload)
    assert path.read_bytes() == payload
    assert hashlib.sha256(path.read_bytes()).hexdigest() == hashlib.sha256(payload).hexdigest()

    blocked = tmp_path / "blocked.csv"
    staging = tmp_path / ".blocked.csv.staging"
    staging.write_bytes(b"partial")
    with pytest.raises(RuntimeError, match="Interrupted output staging"):
        benchmark._atomic_once(blocked, payload)
    assert not blocked.exists()
    assert staging.read_bytes() == b"partial"


def test_test_feature_loader_source_has_explicit_target_free_usecols():
    source = inspect.getsource(benchmark._read_test_features_only)
    assert "pd.read_csv(TEST_PATH, usecols=list(ALLOWED_TEST_COLUMNS)" in source
    assert "pd.read_csv(SAMPLE_PATH, nrows=0)" in source
    assert "artifact_signal" not in benchmark.ALLOWED_TEST_COLUMNS

