from __future__ import annotations

import json

import numpy as np
import pytest

from scripts import optimization_v3_rnbr_official as official
from scripts import optimization_v3_rnbr_official_verify as independent
from scripts import optimization_v3_rnbr_serialization_recovery as recovery


def _synthetic_inputs():
    ids = np.asarray(["d0", "d1", "d2", "d3", "q", "d4", "m"], dtype="U2")
    cycles = np.asarray([56, 57, 58, 59, 68, 60, 68], dtype=np.int64)
    present = np.asarray([True, True, True, True, True, True, False])
    train = {"rows": np.arange(7, dtype=np.int64), "ids": ids,
             "cycles": cycles, "present": present}
    geometry = {
        "score_rows": np.asarray([4, 6], dtype=np.int64),
        "score_record_ids": np.asarray(["q", "m"], dtype="U2"),
        "score_cycles": np.asarray([68, 68], dtype=np.int64),
        "neighbor_count": np.asarray([5, 0], dtype=np.int16),
        "neighbor_source_rows": np.asarray([[0, 1, 2, 3, 5], [-1, -1, -1, -1, -1]], dtype=np.int64),
        "neighbor_ids": np.asarray([[
            "str_:d0", "str_:d1", "str_:d2", "str_:d3", "str_:d4"], ["", "", "", "", ""]], dtype="U8"),
        "neighbor_similarities": np.asarray([[0.9, 0.8, 0.7, 0.6, 0.5], [0, 0, 0, 0, 0]], dtype=np.float64),
    }
    cache = {
        55: {"rows": np.asarray([0, 1, 2, 3]), "ids": ids[:4],
             "prediction": np.asarray([39, 40, 41, 42], dtype=np.float64)},
        59: {"rows": np.asarray([5]), "ids": ids[5:6],
             "prediction": np.asarray([43], dtype=np.float64)},
        67: {"rows": np.asarray([4, 6]), "ids": ids[[4, 6]],
             "prediction": np.asarray([45, 60], dtype=np.float64)},
    }
    targets = {"ids": ids[[0, 1, 2, 3, 5]], "cycles": cycles[[0, 1, 2, 3, 5]],
               "target": np.asarray([40, 41, 42, 43, 44], dtype=np.float64)}
    return train, geometry, cache, targets, {"freeze_signature": "a" * 64}


def _safe_payload_with_object_inputs(monkeypatch):
    train, geometry, cache, targets, freeze = _synthetic_inputs()
    train["ids"] = train["ids"].astype(object)
    geometry["score_record_ids"] = geometry["score_record_ids"].astype(object)
    geometry["neighbor_ids"] = geometry["neighbor_ids"].astype(object)
    targets["ids"] = targets["ids"].astype(object)
    adapter = {"adapter_signature": "b" * 64}
    attrs = ("build_official_payload", "persist_bundle", "_bundle_dir", "load_bundle",
             "finish_fit", "_append_event", "_write_json_once")
    original = {name: getattr(official, name) for name in attrs}
    try:
        recovery._install_serialization_adapter(freeze, adapter)
        arrays, metadata = official.build_official_payload(
            origin=67, arm="observed", train=train, cache=cache, geometry=geometry,
            donor_targets=targets, freeze=freeze, source_commit="synthetic-test-only",
            dataset_tag="RNBR_SERIALIZATION_SYNTHETIC_V1",
        )
        return arrays, metadata, freeze, adapter, original
    except Exception:
        for name, value in original.items():
            setattr(official, name, value)
        raise


def test_object_and_unicode_inputs_build_safe_payload_and_round_trip(tmp_path, monkeypatch):
    arrays, metadata, freeze, adapter, original = _safe_payload_with_object_inputs(monkeypatch)
    try:
        assert all(np.asarray(value).dtype.kind != "O" for value in arrays.values())
        assert arrays["score_record_ids"].dtype.kind == "U"
        assert arrays["donor_record_ids"].dtype.kind == "U"
        assert arrays["neighbor_ids"].dtype.kind == "U"
        assert arrays["neighbor_ids"][0, 0] == "str_:d0"
        assert arrays["neighbor_ids"][1].tolist() == ["", "", "", "", ""]

        delta, prediction = independent.independent_prediction_recompute(arrays)
        np.testing.assert_allclose(arrays["delta"], delta, atol=1e-12, rtol=0)
        np.testing.assert_allclose(arrays["prediction"], prediction, atol=1e-12, rtol=0)
        assert arrays["prediction"][1] == arrays["baseline_prediction"][1]

        run_root = tmp_path / "run"
        monkeypatch.setattr(official, "RUN_ROOT", run_root)
        monkeypatch.setattr(recovery, "RECOVERY_BUNDLE", run_root / "bundles" / recovery.FIT_ID)
        receipt = official.persist_bundle(recovery.FIT_ID, arrays, metadata)
        loaded, loaded_receipt = original["load_bundle"](
            recovery.FIT_ID, freeze, source_commit="synthetic-test-only")
        assert receipt == loaded_receipt
        assert all(np.array_equal(arrays[name], loaded[name]) for name in arrays)
        assert all(value.dtype.kind != "O" for value in loaded.values())
        assert loaded_receipt["metadata"]["serialization_adapter_signature"] == adapter["adapter_signature"]
    finally:
        for name, value in original.items():
            setattr(official, name, value)


def test_failed_npz_header_is_read_without_pickle_and_ids_rebuilt_from_rows(tmp_path):
    path = tmp_path / "failed.npz"
    with path.open("xb") as stream:
        np.savez_compressed(
            stream, donor_rows=np.asarray([2, 4], dtype=np.int64),
            score_rows=np.asarray([5], dtype=np.int64),
            donor_record_ids=np.asarray(["pickle-backed-placeholder-1", "pickle-backed-placeholder-2"], dtype=object),
        )
    headers = recovery.inspect_npz_headers(path)
    assert [key for key, value in headers.items() if value["dtype"] == "object"] == ["donor_record_ids"]
    safe = recovery.load_nonobject_arrays(path, headers)
    assert "donor_record_ids" not in safe
    assert safe["donor_rows"].tolist() == [2, 4]

    train_ids = np.asarray(["r0", "r1", "source-A", "r3", "source-B", "query"], dtype="U8")
    repaired = recovery.repair_source_arrays(original_npz=path, headers=headers,
                                             donor_rows=safe["donor_rows"], train_ids=train_ids)
    assert repaired["donor_record_ids"].dtype.kind == "U"
    assert repaired["donor_record_ids"].tolist() == ["source-A", "source-B"]
    assert all(value.dtype.kind != "O" for value in repaired.values())


def _patch_recovery_storage(tmp_path, monkeypatch):
    repo = tmp_path / "synthetic_repo"
    original_bundle = repo / "original"
    original_bundle.mkdir(parents=True)
    original_npz = original_bundle / "arrays.npz"
    original_npz.write_bytes(b"immutable synthetic original NPZ")
    original_receipt_path = original_bundle / "receipt.json"
    parent = {"freeze_signature": "a" * 64}
    identity = {"fit_id": recovery.FIT_ID, "freeze_signature": parent["freeze_signature"],
                "source_commit": recovery.ORIGINAL_COMMIT}
    original_receipt = {"identity": identity, "metadata": {}, "integrity_sha256": "original-integrity"}
    original_receipt_path.write_text(json.dumps(original_receipt), encoding="utf-8")
    monkeypatch.setattr(recovery, "ROOT", repo)
    monkeypatch.setattr(recovery, "ORIGINAL_NPZ", original_npz)
    monkeypatch.setattr(recovery, "ORIGINAL_RECEIPT", original_receipt_path)
    monkeypatch.setattr(recovery, "RECOVERY_ROOT", repo / "recovery")
    monkeypatch.setattr(recovery, "RECOVERY_BUNDLE", repo / "recovery" / recovery.FIT_ID)
    monkeypatch.setattr(recovery, "source_commit", lambda: "synthetic-adapter-commit")
    monkeypatch.setattr(official, "_read_ledger", lambda: [
        {"event": "FIT_STARTED", "fit_id": recovery.FIT_ID},
        {"event": "FIT_FAILED", "fit_id": recovery.FIT_ID,
         "error": "ValueError: Object arrays cannot be loaded when allow_pickle=False"},
    ])
    return parent, original_receipt


def test_recovery_bundle_is_persistable_resumable_and_corruption_is_rejected(tmp_path, monkeypatch):
    arrays, _, _, _, original = _safe_payload_with_object_inputs(monkeypatch)
    parent, original_receipt = _patch_recovery_storage(tmp_path, monkeypatch)
    adapter = {"adapter_signature": "b" * 64}
    formula_audit = {"donor_id_sha256": recovery._id_digest(arrays["donor_record_ids"])}
    failed = official._read_ledger()[1]
    receipt = recovery._atomic_repair_bundle(
        arrays, original_receipt, parent, adapter, formula_audit, failed)
    first, first_receipt = recovery._load_repaired_bundle(parent, adapter)
    resumed, resumed_receipt = recovery._load_repaired_bundle(parent, adapter)
    assert first_receipt == receipt == resumed_receipt
    assert all(np.array_equal(first[name], resumed[name]) for name in first)
    assert first_receipt["repair_lineage"]["original_failure_event_preserved"] is True

    receipt_path = recovery.RECOVERY_BUNDLE / "receipt.json"
    tampered = json.loads(receipt_path.read_text(encoding="utf-8"))
    tampered["identity"]["source_commit"] = "wrong-source"
    tampered["integrity_sha256"] = recovery.sha_bytes(
        recovery.canonical_bytes({key: value for key, value in tampered.items()
                                  if key != "integrity_sha256"}))
    receipt_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(recovery.RecoveryError, match="receipt linkage/hash"):
        recovery._load_repaired_bundle(parent, adapter)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with (recovery.RECOVERY_BUNDLE / "arrays.npz").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(recovery.RecoveryError, match="receipt linkage/hash"):
        recovery._load_repaired_bundle(parent, adapter)
    for name, value in original.items():
        setattr(official, name, value)


def test_existing_recovery_bundle_without_ledger_event_is_idempotently_resumed(tmp_path, monkeypatch):
    arrays, _, _, adapter, original = _safe_payload_with_object_inputs(monkeypatch)
    parent, original_receipt = _patch_recovery_storage(tmp_path, monkeypatch)
    formula_audit = {"donor_id_sha256": recovery._id_digest(arrays["donor_record_ids"])}
    monkeypatch.setattr(recovery, "_validate_recovered_arrays",
                        lambda recovered, receipt, freeze: dict(formula_audit))
    failed = official._read_ledger()[1]
    kwargs = {
        "arrays": arrays, "original_receipt": original_receipt,
        "parent": parent, "adapter": adapter, "formula_audit": formula_audit,
        "failed_event": failed, "recovery_event_exists": False,
    }
    first_arrays, first_receipt, _ = recovery._load_or_create_recovery_bundle(**kwargs)
    durable_hash = recovery.sha_file(recovery.RECOVERY_BUNDLE / "arrays.npz")
    assert not any(event.get("event") == recovery.LEDGER_EVENT
                   for event in official._read_ledger())

    resumed_arrays, resumed_receipt, _ = recovery._load_or_create_recovery_bundle(**kwargs)
    assert resumed_receipt == first_receipt
    assert recovery.sha_file(recovery.RECOVERY_BUNDLE / "arrays.npz") == durable_hash
    assert all(np.array_equal(first_arrays[name], resumed_arrays[name]) for name in arrays)
    assert not any(event.get("event") == recovery.LEDGER_EVENT
                   for event in official._read_ledger())
    for name, value in original.items():
        setattr(official, name, value)


def test_recovery_persistence_failure_leaves_incomplete_staging_fail_closed(tmp_path, monkeypatch):
    arrays, _, _, _, original = _safe_payload_with_object_inputs(monkeypatch)
    parent, original_receipt = _patch_recovery_storage(tmp_path, monkeypatch)
    adapter = {"adapter_signature": "b" * 64}
    failed = official._read_ledger()[1]
    monkeypatch.setattr(recovery.np, "savez_compressed",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("injected disk failure")))
    with pytest.raises(OSError, match="injected disk failure"):
        recovery._atomic_repair_bundle(
            arrays, original_receipt, parent, adapter, {"donor_id_sha256": "c" * 64}, failed)
    assert recovery.RECOVERY_BUNDLE.with_name(recovery.RECOVERY_BUNDLE.name + ".staging").is_dir()
    assert not recovery.RECOVERY_BUNDLE.exists()
    for name, value in original.items():
        setattr(official, name, value)
