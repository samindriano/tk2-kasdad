from contextlib import ExitStack
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.special import expit

from scripts import cls_pos_recency_v1 as runner
from scripts import verify_cls_pos_recency_v1 as verifier
from src.models import cls_pos_recency_v1 as core


def test_presence_and_target_readers_fail_closed_on_missing_or_duplicate_ids(tmp_path):
    train = tmp_path / "Train.csv"
    pd.DataFrame({
        "record_id": ["r0", "r1", "r2"],
        "signal_stream": ["alpha beta", "  ", "gamma"],
        "artifact_signal": [0, 1, 0],
    }).to_csv(train, index=False)

    np.testing.assert_array_equal(
        runner._read_selected_stream_presence(train, [2, 0, 1], ["r2", "r0", "r1"]),
        np.array([True, True, False]),
    )
    with pytest.raises(runner.PilotError):
        runner._read_selected_stream_presence(train, [0, 1], ["r0", "r0"])
    with pytest.raises(runner.PilotError):
        runner._read_selected_stream_presence(train, [0, 1], ["r0", "missing"])
    with pytest.raises(runner.PilotError):
        runner._read_selected_targets(train, [0, 1], ["r0", "r0"])


def _synthetic_temporal_data(n=20903):
    # The fixed E75 cycles are exactly 8,491 rows of the unique E67 pool.
    early = np.array([1552, 1552, 1552, 1552, 1551, 1551, 1551, 1551])
    late = np.array([2123, 2123, 2123, 2123 - 1])
    assert early.sum() + late.sum() == n and late.sum() == 8491
    cycles = np.concatenate([
        *(np.full(count, cycle, dtype=np.int64) for cycle, count in zip(range(68, 76), early)),
        *(np.full(count, cycle, dtype=np.int64) for cycle, count in zip(range(76, 80), late)),
    ])
    rng = np.random.default_rng(607)
    present = rng.random(n) > 0.20
    ef1 = rng.integers(0, 2, size=n, dtype=np.int8)
    text_group = rng.choice(np.array(["Q1", "Q2", "Q3", "Q4"]), size=n)
    text_group[:50] = "RARE"
    x = rng.normal(size=(n, 4))
    z_bag = 0.25 * x[:, 0] - 0.1 * x[:, 2]
    latent = 0.75 * x[:, 0] - 0.35 * x[:, 1] + 0.2 * ef1 + rng.normal(scale=0.6, size=n)
    y = rng.binomial(1, expit(latent)).astype(np.int8)
    p_a02 = expit(latent + rng.normal(scale=0.5, size=n))
    gamma_original = np.array([0.1, -0.05, 0.0, 0.03])
    p_pos_original = expit(z_bag + x @ gamma_original)
    return {
        "cycles": cycles,
        "present": present,
        "ef1": ef1,
        "text_group": text_group,
        "x": sparse.csr_matrix(x),
        "z": z_bag,
        "y": y,
        "p_a02": p_a02,
        "p_pos_original": p_pos_original,
        "gamma_original": gamma_original,
    }


def _fake_contexts(data):
    rows = np.arange(len(data["cycles"]), dtype=np.int64)
    ids = np.array([f"q-{i:05d}" for i in rows])
    contexts = {}
    for origin in (67, 75):
        mask = np.ones(len(rows), dtype=bool) if origin == 67 else data["cycles"] >= 76
        local_rows = rows[mask]
        contexts[origin] = {
            "origin": origin,
            "fit_rows": np.arange(80 if origin == 67 else 120, dtype=np.int64),
            "fit_ids": np.array([f"fit-{origin}-{i}" for i in range(80 if origin == 67 else 120)]),
            "fit_cycles": np.arange(47, 47 + (80 if origin == 67 else 120), dtype=np.int64) % (origin - 46) + 47,
            "x_fit": sparse.csr_matrix((80 if origin == 67 else 120, 4)),
            "y_fit": None,
            "score_rows": local_rows,
            "score_ids": ids[mask],
            "cycles": data["cycles"][mask],
            "present": data["present"][mask],
            "entry_friction": data["ef1"][mask],
            "text_load_group": data["text_group"][mask],
            "x_score": data["x"][mask],
            "z0_score": data["z"][mask],
            "p_a02": data["p_a02"][mask],
            "p_pos_original": data["p_pos_original"][mask],
            "gamma_original": data["gamma_original"],
            "manifest_sha256": f"manifest-E{origin}",
        }
    return contexts


def _install_synthetic_runtime(monkeypatch, tmp_path):
    data = _synthetic_temporal_data()
    official_root = tmp_path / "official"
    csv_path = official_root / "data/raw/classification/Train.csv"
    csv_path.parent.mkdir(parents=True)
    pd.DataFrame({
        "record_id": [f"q-{i:05d}" for i in range(len(data["y"]))],
        "signal_stream": np.where(data["present"], "synthetic tokens", " "),
        "artifact_signal": data["y"],
    }).to_csv(csv_path, index=False)
    contexts = _fake_contexts(data)

    report_root = tmp_path / "reports"
    run_root = tmp_path / "experiments"
    monkeypatch.setattr(runner, "AUDIT_ROOT", tmp_path)
    monkeypatch.setattr(runner, "REPORT_ROOT", report_root)
    monkeypatch.setattr(runner, "RUN_ROOT", run_root)
    monkeypatch.setattr(runner, "FREEZE_PATH", report_root / "FREEZE.json")
    monkeypatch.setattr(runner, "PREFIT_QA_PATH", report_root / "PREFIT_QA.json")
    monkeypatch.setattr(runner, "RESERVATION_PATH", report_root / "RESERVATION.json")
    monkeypatch.setattr(runner, "READY_PATH", report_root / "PREDICTIONS_READY.json")
    monkeypatch.setattr(runner, "PRELABEL_PATH", report_root / "PRELABEL_VERIFICATION.json")
    monkeypatch.setattr(runner, "SCORE_PATH", report_root / "SCORE.json")
    monkeypatch.setattr(runner, "METRIC_PATH", report_root / "METRIC_VERIFICATION.json")
    monkeypatch.setattr(runner, "SCORE_VERIFY_PATH", report_root / "SCORE_VERIFICATION.json")
    monkeypatch.setattr(runner, "LEDGER_PATH", run_root / "fit_ledger.jsonl")

    freeze = {
        "candidate_id": core.CANDIDATE_ID,
        "freeze_signature": "f" * 64,
        "audit_source_commit": "synthetic-source-commit",
        "official_train_sha256": runner.sha256_file(csv_path),
        "parent_input_file_hashes": {"synthetic": "input-hash"},
    }
    class Parent:
        @staticmethod
        def _external_file_inventory(_root):
            return freeze["parent_input_file_hashes"]

    monkeypatch.setattr(runner, "verify_freeze", lambda *_a, **_kw: freeze)
    monkeypatch.setattr(runner, "verify_reservation", lambda *_a, **_kw: {"status": "SYNTHETIC"})
    monkeypatch.setattr(runner, "require_committed", lambda *_a, **_kw: None)
    monkeypatch.setattr(runner, "_parent_runner", lambda: Parent)
    monkeypatch.setattr(runner, "_parent_context", lambda _root, origin: contexts[origin])
    monkeypatch.setattr(runner, "_load_local_module", lambda _name, _path: verifier)
    monkeypatch.setattr(runner, "_validate_origin_context", lambda _root, origin, _context: {"origin": origin})
    monkeypatch.setattr(runner, "_heavy_job_lock", lambda: (ExitStack(), []))
    monkeypatch.setattr(runner, "_resource_snapshot", lambda minimum: {
        "available_bytes": max(int(minimum), 2 * 1024**3), "minimum_required_bytes": int(minimum)})
    monkeypatch.setattr(runner.psutil, "virtual_memory", lambda: SimpleNamespace(available=2 * 1024**3))
    monkeypatch.setattr(runner, "_fit_paths", lambda fit_id: (
        run_root / "units" / f"{fit_id}.npz",
        run_root / "receipts" / f"{fit_id}.json",
        run_root / "verifications" / f"{fit_id}.json",
    ))
    return official_root, data, contexts, freeze


def _write_synthetic_fit_bindings(data, freeze):
    for fit_id in core.FIT_IDS:
        origin, arm = core.fit_id_parts(fit_id)
        rng = np.random.default_rng(origin + (0 if arm == "RECENCY" else 1000))
        n = 90
        cycles = rng.integers(47, origin + 1, size=n)
        y = np.array([0, 1] * (n // 2), dtype=np.int8)
        x = sparse.csr_matrix(rng.normal(size=(n, 4)))
        z = rng.normal(scale=0.2, size=n)
        weights = core.normalize_class_mass(core.recency_base_weights(cycles, origin), y)
        if arm == "SHUFFLED_WITHIN_CLASS":
            weights, _ = core.shuffle_weights_within_class(weights, y)
        gamma, _ = core.fit_weighted_gamma(x, z, y, weights)
        artifact_path, receipt_path, verification_path = runner._fit_paths(fit_id)
        runner.atomic_npz(artifact_path, gamma=gamma)
        receipt = {
            "fit_id": fit_id,
            "freeze_signature": freeze["freeze_signature"],
            "status": "FIT_SAVED",
            "gamma_sha256": runner.sha256_array(gamma),
            "artifact_sha256": runner.sha256_file(artifact_path),
        }
        runner.write_once(receipt_path, receipt)
        verification = {
            "fit_id": fit_id,
            "status": "PASS_INDEPENDENT_FIT_RECONSTRUCTION",
            "receipt_sha256": runner.sha256_file(receipt_path),
        }
        verification["verification_signature"] = runner.canonical_sha256(verification)
        runner.write_once(verification_path, verification)
        runner.append_event({"event": "FIT_STARTED", "fit_id": fit_id})
        runner.append_event({"event": "FIT_SAVED", "fit_id": fit_id,
            "receipt_sha256": runner.sha256_file(receipt_path)})
        runner.append_event({"event": "FIT_VERIFIED", "fit_id": fit_id})


def test_synthetic_prediction_prelabel_score_and_independent_gate_lifecycle(monkeypatch, tmp_path):
    official_root, data, _contexts, freeze = _install_synthetic_runtime(monkeypatch, tmp_path)
    _write_synthetic_fit_bindings(data, freeze)

    # Saved-unit recovery must never call the optimizer again.
    fit_id = core.FIT_IDS[0]
    monkeypatch.setattr(core, "fit_weighted_gamma", lambda *_a, **_kw: pytest.fail("saved fit retrained"))
    recovered = runner.fit_one(official_root, fit_id)
    assert recovered["status"] == "ALREADY_SAVED_NO_REFIT"

    # A modified receipt must fail closed before prediction readiness.
    receipt_path = runner._fit_paths(fit_id)[1]
    receipt_bytes = receipt_path.read_bytes()
    tampered = runner.read_json(receipt_path)
    tampered["gamma_sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(runner.PilotError):
        runner.verify_fit_binding(freeze, fit_id)
    receipt_path.write_bytes(receipt_bytes)

    ready = runner.build_predictions(official_root)
    assert ready["query_labels_read"] is False
    prelabel = runner.verify_prelabel(official_root)
    assert prelabel["status"] == "PASS_TARGET_FREE_PRELABEL"
    assert prelabel["query_labels_read"] is False
    assert prelabel["predictions"]["E67"]["rows"] == 20903
    assert prelabel["predictions"]["E75"]["rows"] == 8491

    score = runner.score_metrics(official_root)
    assert score["query_label_reads"] == 1
    assert score["query_rows"] == 20903
    assert set(score["views"]) == {"E67", "E75", "UNIQUE_PRIMARY"}
    assert {"W4", "W5", "W6", "H12_E67"}.issubset(score["views"]["E67"])
    assert "per_cycle" in score["views"]["UNIQUE_PRIMARY"]
    assert score["views"]["E75"]["W6"]["rows"] == 8491
    assert score["views"]["UNIQUE_PRIMARY"]["UNIQUE_PRIMARY"]["rows"] == 20903
    supported = [
        group for view in score["views"].values()
        for subview in view.values() if isinstance(subview, dict) and "subgroups" in subview
        for group in subview["subgroups"] if group["support"]["supported"]
    ]
    unsupported = [
        group for view in score["views"].values()
        for subview in view.values() if isinstance(subview, dict) and "subgroups" in subview
        for group in subview["subgroups"] if not group["support"]["supported"]
    ]
    assert supported and unsupported
    assert score["independent_metric_qa"]["status"] == "PASS"
    assert score["resource_policy"]["post_score_reserve_pass"] is True

    verified = runner.verify_score(official_root)
    assert verified["status"] == "PASS_INDEPENDENT_METRIC_AND_GATE_VERIFICATION"
    assert verified["query_label_reads_during_verification"] == 0
    assert verified["gate_reconstruction"] == score["gates"]

    # A self-consistent false views hash is still rejected against score['views'].
    score_bytes = runner.SCORE_PATH.read_bytes()
    metric_bytes = runner.METRIC_PATH.read_bytes()
    changed_score = runner.read_json(runner.SCORE_PATH)
    nested_metric = changed_score["metric_verification"]
    nested_metric.pop("verification_signature")
    nested_metric["views_sha256"] = "0" * 64
    nested_metric["verification_signature"] = runner.canonical_sha256(nested_metric)
    changed_score.pop("score_signature")
    changed_score["score_signature"] = runner.canonical_sha256(changed_score)
    runner.SCORE_PATH.write_text(json.dumps(changed_score), encoding="utf-8")
    changed_metric = runner.read_json(runner.METRIC_PATH)
    changed_metric.pop("verification_signature")
    changed_metric["score_sha256"] = runner.sha256_file(runner.SCORE_PATH)
    changed_metric["views_sha256"] = "0" * 64
    changed_metric["verification_signature"] = runner.canonical_sha256(changed_metric)
    runner.METRIC_PATH.write_text(json.dumps(changed_metric), encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not bind the saved score views"):
        verifier.verify_score_outputs(official_root, write_report=False)
    runner.SCORE_PATH.write_bytes(score_bytes)
    runner.METRIC_PATH.write_bytes(metric_bytes)

    events = runner.events()
    assert sum(event["event"] == "QUERY_LABELS_OPENED" for event in events) == 1
    assert sum(event["event"] == "QUERY_LABELS_READ" for event in events) == 1
    assert sum(event["event"] == "SCORE_RESOURCE_SNAPSHOT" for event in events) == 1


def test_consumed_fit_without_terminal_event_cannot_be_retried(monkeypatch, tmp_path):
    official_root, _data, _contexts, _freeze = _install_synthetic_runtime(monkeypatch, tmp_path)
    fit_id = core.FIT_IDS[0]
    runner.append_event({"event": "FIT_STARTED", "fit_id": fit_id})
    with pytest.raises(runner.PilotError, match="consumed start"):
        runner.fit_one(official_root, fit_id)
