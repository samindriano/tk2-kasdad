from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.models.research_next_contract import (
    BundleContractError,
    canonical_json_bytes,
    normalize_json,
    partition_payload,
    publish_bundle,
    read_bundle,
    resume_or_publish_bundle,
    validate_arrays,
)


def classification_payload(**overrides):
    payload = {
        "score_rows": np.array([3, 5], dtype=np.int64),
        "score_record_ids": ["r3", "r5"],
        "prediction": np.array([0.2, 0.8], dtype=np.float64),
        "baseline_prediction": np.array([0.3, 0.7], dtype=np.float64),
        "diagnostics": {
            "gate": np.bool_(True),
            "windows": ("W1", "H12"),
            "scores": np.array([0.4, 0.6]),
        },
    }
    payload.update(overrides)
    return payload


def identity():
    return {"experiment": "synthetic-contract-test", "source_sha256": "a" * 64}


def test_json_normalization_is_canonical_and_supports_numpy_and_tuples():
    value = {
        "z": (np.int64(3), np.array([np.bool_(True), np.bool_(False)])),
        "scores": np.array([np.float32(2.5)]),
        "a": None,
    }

    assert normalize_json(value) == {"a": None, "scores": [2.5], "z": [3, [True, False]]}
    assert canonical_json_bytes(value) == b'{"a":null,"scores":[2.5],"z":[3,[true,false]]}'


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_json_normalization_rejects_nonfinite_values(bad):
    with pytest.raises(BundleContractError, match="Nonfinite"):
        normalize_json({"metric": bad})


def test_json_normalization_rejects_unsupported_objects_and_nonstring_keys():
    with pytest.raises(BundleContractError, match="Unsupported"):
        normalize_json({"bad": {1, 2}})
    with pytest.raises(BundleContractError, match="key.*string"):
        normalize_json({1: "not a JSON key"})


def test_partition_payload_places_mapping_diagnostics_and_0d_mapping_in_json():
    object_mapping = np.empty((), dtype=object)
    object_mapping[()] = {"trace": ("a", np.int32(2))}
    payload = classification_payload(object_diagnostics=object_mapping)

    arrays, metadata = partition_payload(payload)

    assert arrays["score_record_ids"].dtype.kind == "U"
    assert metadata["diagnostics"] == {
        "gate": True,
        "scores": [0.4, 0.6],
        "windows": ["W1", "H12"],
    }
    assert metadata["object_diagnostics"] == {"trace": ["a", 2]}


def test_partition_payload_rejects_arbitrary_object_arrays():
    bad = np.array([{"nested": "object"}], dtype=object)
    with pytest.raises(BundleContractError, match="Object array"):
        partition_payload(classification_payload(bad=bad))

    object_strings = np.array(["r3", "r5"], dtype=object)
    with pytest.raises(BundleContractError, match="Object array"):
        partition_payload(classification_payload(other_strings=object_strings))


def test_partition_payload_converts_only_declared_record_id_field():
    payload = classification_payload(note_strings=["one", "two"])
    arrays, metadata = partition_payload(payload)

    assert arrays["score_record_ids"].dtype.kind == "U"
    assert "note_strings" not in arrays
    assert metadata["note_strings"] == ["one", "two"]

    arrays2, _ = partition_payload(
        {"record_ids": ["r3", "r5"]}, record_id_field="record_ids"
    )
    assert arrays2["record_ids"].dtype.kind == "U"


def test_partition_payload_rejects_nonfinite_numeric_arrays():
    with pytest.raises(BundleContractError, match="Nonfinite"):
        partition_payload(classification_payload(extra=np.array([np.nan])))


@pytest.mark.parametrize(
    ("task", "prediction", "baseline", "message"),
    [
        ("classification", [0.2, 1.01], [0.3, 0.7], "classification range"),
        ("classification", [0.2, 0.8], [-0.1, 0.7], "classification range"),
        ("regression", [10.0, 101.0], [8.0, 9.0], "regression range"),
        ("regression", [10.0, 11.0], [8.0, -0.1], "regression range"),
    ],
)
def test_final_predictions_must_be_within_task_range(task, prediction, baseline, message):
    arrays, _ = partition_payload(
        {
            "score_rows": np.array([0, 1]),
            "score_record_ids": ["a", "b"],
            "prediction": np.array(prediction),
            "baseline_prediction": np.array(baseline),
        }
    )
    with pytest.raises(BundleContractError, match=message):
        validate_arrays(arrays, task=task)


def test_raw_prediction_may_be_finite_outside_regression_output_range():
    arrays, _ = partition_payload(
        {
            "score_rows": np.array([0, 1]),
            "score_record_ids": ["a", "b"],
            "prediction": np.array([10.0, 90.0]),
            "baseline_prediction": np.array([20.0, 80.0]),
            "raw_prediction": np.array([-25.0, 140.0]),
        }
    )
    validate_arrays(arrays, task="regression")


def test_empty_or_duplicate_score_rows_are_rejected():
    empty = {
        "score_rows": np.array([], dtype=np.int64),
        "score_record_ids": [],
        "prediction": np.array([], dtype=np.float64),
        "baseline_prediction": np.array([], dtype=np.float64),
    }
    arrays, _ = partition_payload(empty)
    with pytest.raises(BundleContractError, match="must not be empty"):
        validate_arrays(arrays, task="classification")

    duplicate = {
        "score_rows": np.array([0, 0]),
        "score_record_ids": ["a", "b"],
        "prediction": np.array([0.2, 0.8]),
        "baseline_prediction": np.array([0.3, 0.7]),
    }
    arrays, _ = partition_payload(duplicate)
    with pytest.raises(BundleContractError, match="must be unique"):
        validate_arrays(arrays, task="classification")


def test_negative_score_rows_are_rejected():
    arrays, _ = partition_payload(
        {
            "score_rows": np.array([-1, 1]),
            "score_record_ids": ["a", "b"],
            "prediction": np.array([0.2, 0.8]),
            "baseline_prediction": np.array([0.3, 0.7]),
        }
    )
    with pytest.raises(BundleContractError, match="score_rows must be nonnegative"):
        validate_arrays(arrays, task="classification")


def test_duplicate_score_record_ids_are_rejected():
    arrays, _ = partition_payload(
        {
            "score_rows": np.array([0, 1]),
            "score_record_ids": ["same-id", "same-id"],
            "prediction": np.array([0.2, 0.8]),
            "baseline_prediction": np.array([0.3, 0.7]),
        }
    )
    with pytest.raises(BundleContractError, match="score_record_ids must be unique"):
        validate_arrays(arrays, task="classification")


def test_publish_read_and_exact_resume_roundtrip(tmp_path):
    destination = tmp_path / "bundle"

    first = resume_or_publish_bundle(
        destination, classification_payload(), identity(), task="classification"
    )
    resumed = resume_or_publish_bundle(
        destination, classification_payload(), identity(), task="classification"
    )
    direct = read_bundle(destination, identity(), task="classification")

    assert set(first.arrays) == set(resumed.arrays) == set(direct.arrays)
    for key in first.arrays:
        np.testing.assert_array_equal(first.arrays[key], resumed.arrays[key])
    assert resumed.receipt == first.receipt == direct.receipt
    assert resumed.receipt["metadata"]["diagnostics"]["windows"] == ["W1", "H12"]
    with np.load(destination / "arrays.npz", allow_pickle=False) as archive:
        assert archive["score_record_ids"].dtype.kind == "U"


def test_duplicate_publish_refuses_to_replace_existing_bundle(tmp_path):
    destination = tmp_path / "bundle"
    publish_bundle(destination, classification_payload(), identity(), task="classification")
    before = (destination / "arrays.npz").read_bytes()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        publish_bundle(destination, classification_payload(), identity(), task="classification")

    assert (destination / "arrays.npz").read_bytes() == before


def test_exact_resume_rejects_identity_metadata_or_array_mismatch(tmp_path):
    destination = tmp_path / "bundle"
    resume_or_publish_bundle(destination, classification_payload(), identity(), task="classification")

    with pytest.raises(BundleContractError, match="identity"):
        resume_or_publish_bundle(
            destination,
            classification_payload(),
            {"experiment": "other"},
            task="classification",
        )
    with pytest.raises(BundleContractError, match="metadata differs"):
        resume_or_publish_bundle(
            destination,
            classification_payload(diagnostics={"gate": False}),
            identity(),
            task="classification",
        )
    with pytest.raises(BundleContractError, match="array .* differs"):
        resume_or_publish_bundle(
            destination,
            classification_payload(prediction=np.array([0.25, 0.75])),
            identity(),
            task="classification",
        )


def test_read_bundle_rejects_altered_npz_or_receipt(tmp_path):
    npz_destination = tmp_path / "changed-npz"
    publish_bundle(
        npz_destination, classification_payload(), identity(), task="classification"
    )
    with (npz_destination / "arrays.npz").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(BundleContractError, match="NPZ hash"):
        read_bundle(npz_destination, identity(), task="classification")

    receipt_destination = tmp_path / "changed-receipt"
    publish_bundle(
        receipt_destination, classification_payload(), identity(), task="classification"
    )
    receipt_path = receipt_destination / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["metadata"]["tampered"] = True
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(BundleContractError, match="Receipt integrity"):
        read_bundle(receipt_destination, identity(), task="classification")


def test_failed_atomic_publication_preserves_staging_directory(tmp_path, monkeypatch):
    import src.models.research_next_contract as contract

    destination = tmp_path / "bundle"

    def fail_rename(_source, _destination):
        raise OSError("synthetic rename failure")

    monkeypatch.setattr(contract.os, "rename", fail_rename)
    with pytest.raises(BundleContractError, match="staging directory preserved") as error:
        publish_bundle(
            destination, classification_payload(), identity(), task="classification"
        )

    stage_text = str(error.value).split("preserved at ", 1)[1].split(": synthetic", 1)[0]
    stage = Path(stage_text)
    assert stage.is_dir()
    assert (stage / "arrays.npz").is_file()
    assert (stage / "receipt.json").is_file()
    assert not destination.exists()

