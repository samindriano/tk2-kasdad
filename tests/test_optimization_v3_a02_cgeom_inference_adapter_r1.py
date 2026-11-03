from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

import scripts.optimization_v3_a02_cgeom_benchmark_v1 as batch
import scripts.optimization_v3_a02_cgeom_inference_adapter_r1 as adapter


class ContractRealisticA02Producer:
    """Synthetic producer with the actual ResidualNeighborPredictor output keys."""

    def __init__(self, result: dict[str, Any]):
        self.result = result

    def predict(self, **kwargs: Any) -> dict[str, Any]:
        assert set(kwargs) == {
            "query_texts", "query_record_ids", "query_row_positions", "query_cycles",
            "baseline_probabilities", "missing_mask",
        }
        assert np.array_equal(kwargs["query_record_ids"], np.asarray(["q-a", "q-b", "q-c"]))
        return self.result


def _contract_arrays() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    baseline_probability = np.asarray([0.50, 0.20, 0.38], dtype=np.float64)
    missing_mask = np.asarray([False, True, False], dtype=bool)
    correction_unclipped = np.asarray([0.30, 0.00, -0.04], dtype=np.float64)
    correction = np.asarray([0.20, 0.00, -0.04], dtype=np.float64)
    candidate_probability = np.asarray([0.70, 0.20, 0.34], dtype=np.float64)
    labels = np.asarray([1, 0, 0], dtype=np.int8)
    neighbor_count = np.asarray([2, 0, 1], dtype=np.int16)
    neighbor_source_rows = np.asarray([[51, 52], [-1, -1], [73, -1]], dtype=np.int64)
    neighbor_ids = np.asarray([["d-1", "d-2"], ["", ""], ["d-3", ""]], dtype="<U3")
    neighbor_similarities = np.asarray([[0.8, 0.6], [0.0, 0.0], [0.5, 0.0]], dtype=np.float64)
    result = {
        "candidate_probability": candidate_probability,
        "correction": correction,
        "correction_unclipped": correction_unclipped,
        "neighbor_count": neighbor_count,
        "neighbor_source_rows": neighbor_source_rows,
        "neighbor_ids": neighbor_ids,
        "neighbor_similarities": neighbor_similarities,
    }
    saved = {
        "record_ids": np.asarray(["q-a", "q-b", "q-c"]),
        "baseline_probability": baseline_probability,
        "missing_mask": missing_mask,
        **result,
        "labels": labels,
    }
    return result, saved


def _derived_helper(tmp_path: Path, model: ContractRealisticA02Producer, saved: dict[str, np.ndarray]):
    tmp_path.mkdir(parents=True, exist_ok=True)
    source, identity = adapter._derive_verified_helper()
    namespace = dict(batch.__dict__)
    npz_path = tmp_path / "a02_test_predictions.npz"
    np.savez_compressed(npz_path, **saved)
    namespace["A02_MODEL"] = tmp_path / "unused-model.joblib"
    namespace["A02_TEST_NPZ"] = npz_path
    namespace["joblib"] = SimpleNamespace(load=lambda _path: model)
    exec(compile(source, str(Path(batch.__file__).resolve()), "exec"), namespace)
    return namespace["_load_and_replay_a02"], identity


def _query() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    test = pd.DataFrame({
        "record_id": ["q-a", "q-b", "q-c"],
        "signal_stream": ["alpha beta", "", "gamma delta"],
        "cycle_code": np.asarray([80, 80, 81], dtype=np.int64),
    })
    baseline = {
        "probability": np.asarray([0.50, 0.20, 0.38], dtype=np.float64),
        "missing_mask": np.asarray([False, True, False], dtype=bool),
    }
    return test, baseline


def test_derived_helper_changes_only_the_authorized_field_expression():
    source, identities = adapter._derive_verified_helper()
    assert identities["frozen_runner_sha256"] == adapter.EXPECTED_FROZEN_RUNNER_SHA256
    assert identities["frozen_helper_sha256"] == adapter.EXPECTED_FROZEN_HELPER_SHA256
    assert source.count(adapter.NEW_EXPRESSION) == 1
    assert adapter.OLD_EXPRESSION not in source
    assert source.count('"correction_unclipped": result["correction_unclipped"]') == 1


def test_actual_replay_helper_maps_unclipped_value_and_exact_saved_schema(tmp_path: Path):
    result, saved = _contract_arrays()
    helper, identity = _derived_helper(tmp_path, ContractRealisticA02Producer(result), saved)
    test, baseline = _query()

    model, replay = helper(test, baseline)

    assert model.__class__ is ContractRealisticA02Producer
    assert identity["derived_helper_sha256"] != identity["frozen_helper_sha256"]
    assert set(replay) == set(saved)
    assert np.array_equal(replay["correction"], np.asarray([0.20, 0.00, -0.04]))
    assert np.array_equal(replay["correction_unclipped"], np.asarray([0.30, 0.00, -0.04]))


def test_actual_replay_helper_rejects_missing_or_changed_producer_keys(tmp_path: Path):
    result, saved = _contract_arrays()
    missing_key = {key: value for key, value in result.items() if key != "correction_unclipped"}
    helper, _ = _derived_helper(tmp_path / "missing", ContractRealisticA02Producer(missing_key), saved)
    test, baseline = _query()
    with pytest.raises(KeyError, match="correction_unclipped"):
        helper(test, baseline)

    saved_schema = {key: value for key, value in saved.items() if key != "neighbor_ids"}
    helper, _ = _derived_helper(tmp_path / "schema", ContractRealisticA02Producer(result), saved_schema)
    with pytest.raises(batch.BatchError, match="prediction schema mismatch"):
        helper(test, baseline)

    changed_saved = {key: value.copy() for key, value in saved.items()}
    changed_saved["neighbor_similarities"][0, 0] = 0.81
    helper, _ = _derived_helper(tmp_path / "changed", ContractRealisticA02Producer(result), changed_saved)
    with pytest.raises(batch.BatchError, match="failed exact feature-only replay"):
        helper(test, baseline)
