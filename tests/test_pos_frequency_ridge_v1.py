from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse
from pathlib import Path
import json
import hashlib
import os
import subprocess
import sys
import time

from scripts import optimization_v3_cls_pos_frequency_ridge_v1 as runner
from scripts.verify_optimization_v3_cls_pos_frequency_ridge_v1 import (
    VerificationError,
    sha256_file,
    verify_gamma_artifact,
    verify_source_preflight,
)
from src.models.pos_frequency_ridge_v1 import (
    FrequencyRidgeError,
    enforce_fit_cap,
    fit_resume_decision,
    fit_weighted_offset,
    guarded_target_read,
    make_penalty_maps,
    predict_with_fallback,
    repeat_token_weights,
    sha256_array,
    union_document_frequency,
    weighted_offset_objective_gradient,
)
from src.models.positional_bag_offset_v1 import offset_objective_gradient


def test_weighted_gradient_matches_finite_difference() -> None:
    x = sparse.csr_matrix(np.asarray([
        [1.0, 0.0, 0.2], [0.0, 1.0, -0.4], [1.0, 1.0, 0.0],
        [-0.3, 0.0, 0.7], [0.0, -0.5, 0.2], [0.1, 0.4, 0.8],
    ], dtype=np.float64))
    offset = np.asarray([-1.1, 0.4, 1.2, -0.2, 0.9, -0.7])
    y = np.asarray([0, 1, 1, 0, 1, 0], dtype=np.int8)
    gamma = np.asarray([0.2, -0.35, 0.1])
    weights = np.asarray([0.7, 1.3, 2.0])
    loss, gradient = weighted_offset_objective_gradient(gamma, x, offset, y, weights)
    eps = 1e-6
    numeric = np.empty_like(gamma)
    for j in range(len(gamma)):
        plus, minus = gamma.copy(), gamma.copy()
        plus[j] += eps
        minus[j] -= eps
        numeric[j] = (
            weighted_offset_objective_gradient(plus, x, offset, y, weights)[0]
            - weighted_offset_objective_gradient(minus, x, offset, y, weights)[0]
        ) / (2 * eps)
    assert np.isfinite(loss)
    assert np.allclose(gradient, numeric, rtol=2e-6, atol=2e-7)


def test_all_ones_weighted_objective_equals_historical_isotropic_objective() -> None:
    x = sparse.csr_matrix(np.asarray([[1.0, 0.0], [0.3, -0.4], [0.0, 2.0]]))
    offset = np.asarray([-0.3, 0.7, 1.2])
    y = np.asarray([0, 1, 1], dtype=np.int8)
    gamma = np.asarray([0.2, -0.15])
    new_loss, new_gradient = weighted_offset_objective_gradient(
        gamma, x, offset, y, np.ones_like(gamma)
    )
    old_loss, old_gradient = offset_objective_gradient(gamma, x, offset, y, c=1.0)
    assert new_loss == pytest.approx(old_loss, rel=0, abs=1e-14)
    assert np.array_equal(new_gradient, old_gradient)


def test_union_df_counts_documents_not_occurrences_or_third_blocks() -> None:
    # Token 0 occurs twice in row 0 but contributes one document to its df.
    x = sparse.csr_matrix((np.asarray([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float64),
                           (np.asarray([0, 0, 1, 1, 2, 2, 3, 3]),
                            np.asarray([0, 3, 8, 4, 6, 2, 1, 7]))), shape=(4, 9))
    assert union_document_frequency(x, vocabulary_size=3).tolist() == [2, 2, 2]


def test_union_df_does_not_count_explicit_sparse_zeros_as_present() -> None:
    x = sparse.csr_matrix((np.asarray([0.0, 1.0]), np.asarray([0, 1]), np.asarray([0, 1, 2])),
                          shape=(2, 3))
    # Vocabulary size one: the explicit zero in the first third is absent;
    # the second row's token in the second third is present.
    assert union_document_frequency(x, vocabulary_size=1).tolist() == [1]


def test_penalty_map_is_normalized_and_permutation_preserves_multiset() -> None:
    first = make_penalty_maps([1, 2, 4, 8, 16])
    second = make_penalty_maps([1, 2, 4, 8, 16])
    aligned = first["aligned_token_weights"]
    permuted = first["permuted_token_weights"]
    assert np.mean(aligned) == pytest.approx(1.0, abs=1e-14)
    assert np.array_equal(np.sort(aligned), np.sort(permuted))
    assert np.array_equal(aligned, second["aligned_token_weights"])
    assert np.array_equal(permuted, second["permuted_token_weights"])
    assert not first["aligned_equals_permuted"]
    repeated = repeat_token_weights(aligned)
    assert np.array_equal(repeated, np.tile(aligned, 3))


def test_rejects_noninteger_document_frequency_and_fractional_fit_target() -> None:
    with pytest.raises(FrequencyRidgeError, match="positive token document frequencies"):
        make_penalty_maps([1.5, 2.0])
    with pytest.raises(FrequencyRidgeError, match="before conversion"):
        fit_weighted_offset(sparse.csr_matrix([[1.0], [0.0]]), [0.0, 0.1], [0.25, 1.0], [1.0])


def test_missing_stream_prediction_is_exactly_fallback() -> None:
    x = sparse.csr_matrix([[1.0, 0.0], [0.0, 1.0], [0.2, 0.3]])
    fallback = np.asarray([0.123456789, 0.7, 0.314159265])
    output = predict_with_fallback(x, [0.1, 0.2, -0.3], [0.4, -0.2],
                                   [True, False, False], fallback)
    assert output[0] != fallback[0]
    assert np.array_equal(output[1:], fallback[1:])


def test_hash_and_id_guard_fails_before_target_reader() -> None:
    called = False

    def reader():
        nonlocal called
        called = True
        return np.asarray([0, 1])

    with pytest.raises(FrequencyRidgeError, match="source hashes"):
        guarded_target_read(expected_hashes={"train": "a"}, actual_hashes={"train": "b"},
                            expected_ids_sha256="id-a", actual_ids_sha256="id-a", reader=reader)
    assert called is False
    with pytest.raises(FrequencyRidgeError, match="fit IDs"):
        guarded_target_read(expected_hashes={"train": "a"}, actual_hashes={"train": "a"},
                            expected_ids_sha256="id-a", actual_ids_sha256="id-b", reader=reader)
    assert called is False


def test_resume_uses_durable_save_without_duplicate_start_and_cap_is_exact() -> None:
    fit_ids = ["E67_ALIGNED", "E67_PERMUTED", "E75_ALIGNED", "E75_PERMUTED"]
    events = [{"event": "FIT_STARTED", "fit_id": fit_ids[0]},
              {"event": "FIT_SAVED", "fit_id": fit_ids[0]}]
    assert fit_resume_decision(events, fit_ids[0], fit_ids) == "RESUME_SAVED"
    assert fit_resume_decision(events, fit_ids[1], fit_ids) == "START_NEW"
    enforce_fit_cap(events, fit_ids)
    with pytest.raises(FrequencyRidgeError, match="namespace cap"):
        enforce_fit_cap(events + [{"event": "FIT_STARTED", "fit_id": "UNAUTHORIZED"}], fit_ids)
    with pytest.raises(FrequencyRidgeError, match="pending slot"):
        fit_resume_decision([{"event": "FIT_STARTED", "fit_id": fit_ids[1]}], fit_ids[1], fit_ids)
    with pytest.raises(FrequencyRidgeError, match="no earlier matching"):
        fit_resume_decision([{"event": "FIT_SAVED", "fit_id": fit_ids[2]}], fit_ids[2], fit_ids)
    with pytest.raises(FrequencyRidgeError, match="no earlier matching"):
        fit_resume_decision([{"event": "FIT_SAVED", "fit_id": fit_ids[2]},
                             {"event": "FIT_STARTED", "fit_id": fit_ids[2]}], fit_ids[2], fit_ids)


def test_synthetic_weighted_optimizer_converges_from_zero() -> None:
    rng = np.random.default_rng(19)
    x = sparse.csr_matrix(rng.normal(size=(120, 9)))
    offset = rng.normal(size=120)
    p = 1.0 / (1.0 + np.exp(-(offset + x @ rng.normal(scale=0.1, size=9))))
    y = rng.binomial(1, p).astype(np.int8)
    gamma, metadata = fit_weighted_offset(x, offset, y, np.ones(9))
    assert metadata["success"] is True
    assert metadata["loss_final"] <= metadata["loss_gamma0"] + 1e-8
    assert np.isfinite(gamma).all()


def test_runner_synthetic_lifecycle_passes_without_official_inputs() -> None:
    result = runner.run_synthetic_qa()
    assert result["status"] == "PASS"
    assert result["durable_resume_without_duplicate_fit"] is True
    assert result["official_inputs_opened"] is False
    assert result["official_labels_opened"] is False


def _passing_resource_snapshot(*, estimated_additional_bytes: int = 0) -> dict[str, object]:
    return {
        "available_bytes": 8 * 1024**3,
        "process_rss_bytes": 100 * 1024**2,
        "estimated_additional_bytes": estimated_additional_bytes,
        "projected_free_bytes": 6 * 1024**3,
        "initial_ram_pass": True,
        "projected_reserve_pass": True,
        "rss_pass": True,
    }


def _install_synthetic_fit_runner(monkeypatch, tmp_path: Path) -> dict[str, object]:
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    train = tmp_path / "data/raw/classification/Train.csv"
    train.parent.mkdir(parents=True)
    train.write_bytes(b"synthetic-only Train placeholder; no labels read")
    train_sha = hashlib.sha256(train.read_bytes()).hexdigest()
    monkeypatch.setattr(runner, "TRAIN", train)
    monkeypatch.setattr(runner, "TRAIN_SHA256", train_sha)
    preflight_path = tmp_path / "reports/SOURCE_PREFLIGHT_R2.json"
    root_pass_path = tmp_path / "reports/ROOT_SOURCE_PASS.json"
    preflight_path.parent.mkdir(parents=True)
    preflight_path.write_text("synthetic source pass", encoding="utf-8")
    root_pass_path.write_text("synthetic root pass", encoding="utf-8")
    monkeypatch.setattr(runner, "SOURCE_PREFLIGHT", preflight_path)
    monkeypatch.setattr(runner, "ROOT_SOURCE_PASS", root_pass_path)
    source_hashes = {"synthetic_source.py": "a" * 64}
    root_pass = {"source_commit": "b" * 40, "source_hashes": source_hashes}
    monkeypatch.setattr(runner, "_require_root_source_pass", lambda: root_pass)
    monkeypatch.setattr(runner, "_source_hashes", lambda: source_hashes)
    preflight = {
        "train_sha256": train_sha,
        "target_values_opened": False,
        "source_hashes": source_hashes,
        "prior_shared_ledger_reference": {"matches_checkpoint": True},
    }
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
    shared_ledger = tmp_path / "experiments/shared-ledger.jsonl"
    shared_ledger.parent.mkdir(parents=True)
    shared_ledger.write_bytes(b"frozen synthetic historical ledger\n")
    prior_hash = hashlib.sha256(shared_ledger.read_bytes()).hexdigest()
    monkeypatch.setattr(runner, "SHARED_LEDGER", shared_ledger)
    protocol = tmp_path / "experiments/protocol.json"
    protocol.write_text(json.dumps({"historical_evidence": {
        "prior_shared_ledger_sha256_at_checkpoint": prior_hash}}), encoding="utf-8")
    monkeypatch.setattr(runner, "PROTOCOL", protocol)
    run_root = tmp_path / "experiments/runs/optimization_v3_cls_pos_frequency_ridge_v1"
    monkeypatch.setattr(runner, "RUN_ROOT", run_root)
    monkeypatch.setattr(runner, "LEDGER", run_root / "BUDGET_LEDGER.jsonl")
    pilot_root = tmp_path / "experiments/pilot"
    monkeypatch.setattr(runner, "PILOT_ROOT", pilot_root)
    pilot_accounting = tmp_path / "reports/FIT_ACCOUNTING.json"
    monkeypatch.setattr(runner, "PILOT_ACCOUNTING", pilot_accounting)
    ids = np.asarray(["synthetic-0", "synthetic-1"])
    frame = __import__("pandas").DataFrame({
        "record_id": ids,
        "cycle_code": np.asarray([47, 67], dtype=np.int64),
    })
    fit_rows = np.asarray([0, 1], dtype=np.int64)
    base = sparse.csr_matrix(np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64))
    position = sparse.csr_matrix(np.asarray([
        [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    ], dtype=np.float64))
    augmented = sparse.hstack([base, position], format="csr")
    manifest = {
        "fit_row_metadata": runner._row_metadata(frame, fit_rows),
        "position_audit": {"fit_original": {"unigram_vocabulary_size": 2}},
        "matrices": {"fit_base": {}, "fit_position": {}},
    }
    manifest_path = pilot_root / "MATRIX_MANIFEST_E67.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(runner, "FREEZE_ORIGINS", {67: {"rows": 2, "score_rows": 0}})
    monkeypatch.setattr(runner, "_read_identity_frame", lambda: frame.copy())
    monkeypatch.setattr(runner, "_verify_origin_identity_and_files", lambda _frame, _origin: {})
    monkeypatch.setattr(runner, "_verify_historical_freezes", lambda: {})
    monkeypatch.setattr(runner, "_load_csr_from_manifest",
                        lambda _manifest, name: base if name == "fit_base" else augmented)
    target = np.asarray([0, 1], dtype=np.int8)
    target_sha = sha256_array(target)
    bag_id = "POSITIONAL_TEXT_PILOT_V1_E67_BAG_LR"
    pilot_accounting.write_text(json.dumps({"fit_units": [{
        "fit_id": bag_id,
        "artifact_sha256": "c" * 64,
        "receipt_sha256": "d" * 64,
        "fit_population": {"fit_target_sha256": target_sha},
    }]}), encoding="utf-8")

    class SyntheticBag:
        classes_ = np.asarray([0, 1])
        coef_ = np.asarray([[0.2, -0.1]], dtype=np.float64)
        intercept_ = np.asarray([0.05], dtype=np.float64)

    monkeypatch.setattr(runner.joblib, "load", lambda _path: {"estimator": SyntheticBag()})
    calls = {"labels": 0, "fits": 0}

    def read_labels(_rows, _ids, expected_sha):
        calls["labels"] += 1
        assert expected_sha == target_sha
        return target.copy()

    def fit_model(_x, _offset, _y, _weights):
        calls["fits"] += 1
        return np.linspace(-0.1, 0.1, 6, dtype=np.float64), {
            "success": True, "loss_gamma0": 1.0, "loss_final": 0.5,
        }

    monkeypatch.setattr(runner, "_read_fit_labels", read_labels)
    monkeypatch.setattr(runner, "fit_weighted_offset", fit_model)
    monkeypatch.setattr(runner, "resource_snapshot", _passing_resource_snapshot)
    monkeypatch.setattr(runner, "SHARED_JOB_LOCK", tmp_path / "kasdad_single_heavy_job.lock")
    return {"calls": calls, "target_sha256": target_sha, "root_pass": root_pass,
            "ids": ids, "position": position, "preflight": preflight}


def test_shared_lock_blocks_runner_before_reservation_and_downstream_work(monkeypatch, tmp_path: Path) -> None:
    lock_path = tmp_path / "kasdad_single_heavy_job.lock"
    ready_path = tmp_path / "lock-holder.ready"
    child_code = r'''import os, pathlib, sys, time
path, ready = sys.argv[1], sys.argv[2]
stream = open(path, "a+b")
if os.name == "nt":
    import msvcrt
    stream.seek(0)
    if stream.read(1) == b"": stream.write(b"\0"); stream.flush()
    stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
    pathlib.Path(ready).write_text("ready", encoding="utf-8")
    time.sleep(30); stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
else:
    import fcntl
    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    pathlib.Path(ready).write_text("ready", encoding="utf-8")
    time.sleep(30); fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
stream.close()
'''
    process = subprocess.Popen([sys.executable, "-c", child_code, str(lock_path), str(ready_path)])
    try:
        deadline = time.monotonic() + 8
        while not ready_path.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert ready_path.exists(), "separate lock-holder process did not acquire shared slot"
        monkeypatch.setattr(runner, "SHARED_JOB_LOCK", lock_path)
        monkeypatch.setattr(runner, "RUN_ROOT", tmp_path / "runs")
        monkeypatch.setattr(runner, "resource_snapshot", _passing_resource_snapshot)
        counters = {"reservation": 0, "map": 0, "labels": 0, "fit": 0}
        monkeypatch.setattr(runner, "_append_campaign_reservation",
                            lambda *_args: counters.__setitem__("reservation", counters["reservation"] + 1))
        with pytest.raises(runner.FrequencyRidgeRunnerError, match="shared slot"):
            with runner._official_fit_slot({"source_commit": "synthetic", "source_hashes": {}}, None):
                counters["map"] += 1
                counters["labels"] += 1
                counters["fit"] += 1
        assert counters == {"reservation": 0, "map": 0, "labels": 0, "fit": 0}
    finally:
        process.terminate()
        process.wait(timeout=8)
    with runner._official_fit_slot({"source_commit": "synthetic", "source_hashes": {}}, None):
        counters["map"] += 1
    assert counters["reservation"] == 1
    assert counters["map"] == 1


def test_low_ram_gate_precedes_reservation_map_labels_and_fit(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(runner, "SHARED_JOB_LOCK", tmp_path / "shared.lock")
    monkeypatch.setattr(runner, "RUN_ROOT", tmp_path / "runs")
    monkeypatch.setattr(runner, "resource_snapshot", lambda **_kwargs: {
        "available_bytes": 1, "projected_free_bytes": 0, "process_rss_bytes": 0,
        "initial_ram_pass": False, "projected_reserve_pass": False, "rss_pass": True,
    })
    counters = {"reservation": 0, "map": 0, "labels": 0, "fit": 0}
    monkeypatch.setattr(runner, "_append_campaign_reservation",
                        lambda *_args: counters.__setitem__("reservation", counters["reservation"] + 1))
    with pytest.raises(runner.FrequencyRidgeRunnerError, match="Resource gate failed"):
        with runner._official_fit_slot({"source_commit": "synthetic", "source_hashes": {}}, None):
            counters["map"] += 1
            counters["labels"] += 1
            counters["fit"] += 1
    assert counters == {"reservation": 0, "map": 0, "labels": 0, "fit": 0}


def test_map_runner_recovers_published_files_without_recomputing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "RUN_ROOT", tmp_path / "experiments/runs/campaign")
    monkeypatch.setattr(runner, "LEDGER", tmp_path / "experiments/runs/campaign/ledger.jsonl")
    monkeypatch.setattr(runner, "SOURCE_PREFLIGHT", tmp_path / "reports/SOURCE_PREFLIGHT_R2.json")
    monkeypatch.setattr(runner, "ROOT_SOURCE_PASS", tmp_path / "reports/ROOT_SOURCE_PASS.json")
    runner.SOURCE_PREFLIGHT.parent.mkdir(parents=True)
    runner.SOURCE_PREFLIGHT.write_text("synthetic preflight", encoding="utf-8")
    runner.ROOT_SOURCE_PASS.write_text("synthetic root pass", encoding="utf-8")
    monkeypatch.setattr(runner, "TRAIN_SHA256", "e" * 64)
    root_pass = {"source_commit": "f" * 40, "source_hashes": {"source.py": "1" * 64}}
    x = sparse.csr_matrix(np.asarray([
        [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    ], dtype=np.float64))
    rows = np.asarray([0, 1], dtype=np.int64)
    ids = np.asarray(["synthetic-a", "synthetic-b"])
    original_df = runner.union_document_frequency
    original_maps = runner.make_penalty_maps
    calls = {"df": 0, "maps": 0}

    def counted_df(*args, **kwargs):
        calls["df"] += 1
        return original_df(*args, **kwargs)

    def counted_maps(*args, **kwargs):
        calls["maps"] += 1
        return original_maps(*args, **kwargs)

    monkeypatch.setattr(runner, "union_document_frequency", counted_df)
    monkeypatch.setattr(runner, "make_penalty_maps", counted_maps)
    original_append = runner._append_event

    def crash_after_map_publish(path, event):
        if event.get("event") == "PREPROCESSING_MAP_SAVED":
            raise SystemExit("fault injection after map artifact and receipt publication")
        original_append(path, event)

    monkeypatch.setattr(runner, "_append_event", crash_after_map_publish)
    with pytest.raises(SystemExit, match="fault injection"):
        runner._ensure_penalty_map(67, x, 2, root_pass=root_pass, fit_rows=rows, fit_ids=ids)
    artifact = runner.RUN_ROOT / "maps/penalty_map_E67.npz"
    receipt = runner.RUN_ROOT / "maps/penalty_map_E67.json"
    events = runner._events(runner.LEDGER)
    assert artifact.is_file() and receipt.is_file()
    assert [e["event"] for e in events].count("PREPROCESSING_MAP_STARTED") == 1
    assert not any(e["event"] == "PREPROCESSING_MAP_SAVED" for e in events)
    monkeypatch.setattr(runner, "_append_event", original_append)
    resumed = runner._ensure_penalty_map(67, x, 2, root_pass=root_pass, fit_rows=rows, fit_ids=ids)
    assert resumed["receipt"]["artifact_sha256"] == runner._sha_file(artifact)
    assert calls == {"df": 1, "maps": 1}
    assert [e["event"] for e in runner._events(runner.LEDGER)].count("PREPROCESSING_MAP_SAVED") == 1


def test_incomplete_map_publish_blocks_without_recomputation(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "RUN_ROOT", tmp_path / "experiments/runs/campaign")
    monkeypatch.setattr(runner, "LEDGER", tmp_path / "experiments/runs/campaign/ledger.jsonl")
    monkeypatch.setattr(runner, "SOURCE_PREFLIGHT", tmp_path / "reports/SOURCE_PREFLIGHT_R2.json")
    monkeypatch.setattr(runner, "ROOT_SOURCE_PASS", tmp_path / "reports/ROOT_SOURCE_PASS.json")
    runner.SOURCE_PREFLIGHT.parent.mkdir(parents=True)
    runner.SOURCE_PREFLIGHT.write_text("synthetic preflight", encoding="utf-8")
    runner.ROOT_SOURCE_PASS.write_text("synthetic root pass", encoding="utf-8")
    monkeypatch.setattr(runner, "TRAIN_SHA256", "e" * 64)
    root_pass = {"source_commit": "f" * 40, "source_hashes": {"source.py": "1" * 64}}
    x = sparse.csr_matrix(np.asarray([
        [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    ], dtype=np.float64))
    rows = np.asarray([0, 1], dtype=np.int64)
    ids = np.asarray(["synthetic-a", "synthetic-b"])
    original_df = runner.union_document_frequency
    calls = {"df": 0}

    def counted_df(*args, **kwargs):
        calls["df"] += 1
        return original_df(*args, **kwargs)

    monkeypatch.setattr(runner, "union_document_frequency", counted_df)
    original_append = runner._append_event

    def crash_after_map_publish(path, event):
        if event.get("event") == "PREPROCESSING_MAP_SAVED":
            raise SystemExit("fault injection after map publish")
        original_append(path, event)

    monkeypatch.setattr(runner, "_append_event", crash_after_map_publish)
    with pytest.raises(SystemExit):
        runner._ensure_penalty_map(67, x, 2, root_pass=root_pass, fit_rows=rows, fit_ids=ids)
    artifact = runner.RUN_ROOT / "maps/penalty_map_E67.npz"
    receipt = runner.RUN_ROOT / "maps/penalty_map_E67.json"
    receipt.unlink()
    monkeypatch.setattr(runner, "_append_event", original_append)
    with pytest.raises(runner.FrequencyRidgeRunnerError, match="missing durable artifact/receipt"):
        runner._ensure_penalty_map(67, x, 2, root_pass=root_pass, fit_rows=rows, fit_ids=ids)
    assert artifact.is_file() and not receipt.exists()
    assert calls["df"] == 1


def test_fit_runner_recovers_published_gamma_receipt_without_optimizer_retry(monkeypatch, tmp_path: Path) -> None:
    state = _install_synthetic_fit_runner(monkeypatch, tmp_path)
    original_append = runner._append_event

    def crash_after_fit_publish(path, event):
        if event.get("event") == "FIT_SAVED":
            raise SystemExit("fault injection after gamma and receipt publication")
        original_append(path, event)

    monkeypatch.setattr(runner, "_append_event", crash_after_fit_publish)
    with pytest.raises(SystemExit, match="fault injection"):
        runner._run_one_fit("E67_ALIGNED")
    artifact_path, receipt_path = runner._fit_paths("E67_ALIGNED")
    assert artifact_path.is_file() and receipt_path.is_file()
    events = runner._events(runner.LEDGER)
    assert sum(e["event"] == "FIT_STARTED" for e in events) == 1
    assert not any(e["event"] == "FIT_SAVED" for e in events)
    assert state["calls"] == {"labels": 1, "fits": 1}

    monkeypatch.setattr(runner, "_append_event", original_append)
    recovered = runner._run_one_fit("E67_ALIGNED")
    assert recovered["status"] == "RECOVERED_AFTER_PUBLISH"
    assert state["calls"] == {"labels": 1, "fits": 1}
    reused = runner._run_one_fit("E67_ALIGNED")
    assert reused["status"] == "ALREADY_SAVED"
    assert state["calls"] == {"labels": 1, "fits": 1}


def test_incomplete_gamma_publish_blocks_without_retry(monkeypatch, tmp_path: Path) -> None:
    state = _install_synthetic_fit_runner(monkeypatch, tmp_path)
    original_append = runner._append_event

    def crash_after_fit_publish(path, event):
        if event.get("event") == "FIT_SAVED":
            raise SystemExit("fault injection after gamma and receipt publication")
        original_append(path, event)

    monkeypatch.setattr(runner, "_append_event", crash_after_fit_publish)
    with pytest.raises(SystemExit):
        runner._run_one_fit("E67_ALIGNED")
    artifact_path, receipt_path = runner._fit_paths("E67_ALIGNED")
    receipt_path.unlink()
    monkeypatch.setattr(runner, "_append_event", original_append)
    with pytest.raises(runner.FrequencyRidgeRunnerError, match="missing durable artifact/receipt"):
        runner._run_one_fit("E67_ALIGNED")
    assert artifact_path.is_file()
    assert not receipt_path.exists()
    assert state["calls"] == {"labels": 1, "fits": 1}


@pytest.mark.parametrize("field", [
    "source_commit", "fixed_bag_offset_sha256", "penalty_vector_sha256",
])
def test_saved_fit_reuse_rejects_wrong_source_offset_or_penalty(monkeypatch, tmp_path: Path, field: str) -> None:
    state = _install_synthetic_fit_runner(monkeypatch, tmp_path)
    runner._run_one_fit("E67_ALIGNED")
    _, receipt_path = runner._fit_paths("E67_ALIGNED")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt[field] = "0" * 64
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    events = runner._events(runner.LEDGER)
    for event in events:
        if event.get("event") == "FIT_SAVED":
            event["receipt_sha256"] = runner._sha_file(receipt_path)
    runner.LEDGER.write_text("".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
                             encoding="utf-8")
    with pytest.raises(runner.FrequencyRidgeRunnerError, match=field):
        runner._run_one_fit("E67_ALIGNED")
    assert state["calls"] == {"labels": 1, "fits": 1}


def test_exclusive_publisher_refuses_conflicting_existing_bytes(tmp_path: Path) -> None:
    path = tmp_path / "immutable.bin"
    path.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        runner._write_bytes_exclusive(path, b"replacement")
    assert path.read_bytes() == b"original"


def test_official_fit_requires_committed_root_source_pass(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(runner, "ROOT_SOURCE_PASS", tmp_path / "ROOT_SOURCE_PASS.json")
    with pytest.raises(runner.FrequencyRidgeRunnerError, match="committed ROOT_SOURCE_PASS"):
        runner._require_root_source_pass()


def test_independent_preflight_verifier_fails_closed_on_forbidden_access_flag(tmp_path: Path) -> None:
    report = {
        "schema_version": 2,
        "campaign_id": "CLS_POS_FREQUENCY_RIDGE_V1",
        "source_preflight_revision": "R2_SHARED_LOCK_DURABLE_RESUME",
        "shared_job_lock_path": "synthetic-lock-path",
        "source_commit": "a" * 40,
        "source_hashes": {},
        "train_sha256": runner.TRAIN_SHA256,
        "target_values_opened": False,
        "test_or_sample_submission_accessed": True,
        "kaggle_accessed": False,
        "official_fit_started": False,
    }
    path = tmp_path / "preflight.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(VerificationError, match="forbidden access"):
        verify_source_preflight(path)


def test_independent_gamma_verifier_binds_file_and_vector_hash(tmp_path: Path) -> None:
    gamma = np.asarray([0.1, -0.2, 0.3], dtype=np.float64)
    artifact = tmp_path / "gamma.npz"
    with artifact.open("wb") as stream:
        np.savez_compressed(stream, gamma=gamma)
    receipt = {
        "fit_kind": "PREDICTOR",
        "campaign_id": "CLS_POS_FREQUENCY_RIDGE_V1",
        "fit_id": "E67_ALIGNED",
        "gamma_sha256": sha256_array(gamma),
        "artifact_sha256": sha256_file(artifact),
    }
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    assert verify_gamma_artifact(artifact, receipt_path)["gamma_width"] == 3
    receipt["artifact_sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(VerificationError, match="file hash"):
        verify_gamma_artifact(artifact, receipt_path)
