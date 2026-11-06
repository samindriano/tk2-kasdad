from __future__ import annotations

import ast
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from src.models import cls_pos_poisson_bag2_v1 as model
from scripts import cls_pos_poisson_bag2_v1 as runner
from scripts import verify_cls_pos_poisson_bag2_v1 as verifier


def test_poisson_weights_are_fixed_deterministic_and_normalized() -> None:
    for seed in model.REPLICA_SEEDS:
        counts_a, weights_a = model.poisson_row_weights(113, seed)
        counts_b, weights_b = model.poisson_row_weights(113, seed)
        assert np.array_equal(counts_a, counts_b)
        assert np.array_equal(weights_a, weights_b)
        assert counts_a.dtype.kind in "iu"
        assert np.all(counts_a >= 0)
        assert weights_a.sum() == pytest.approx(113.0, abs=1e-10)
    with pytest.raises(RuntimeError, match="Unfrozen"):
        model.poisson_row_weights(113, 23)


def test_weighted_objective_matches_unweighted_formula_and_gradient() -> None:
    x = sparse.csr_matrix([[1.0, 0.0], [0.5, -1.0], [-.2, .7], [1.2, .3]])
    offset = np.asarray([.1, -.3, .25, .6])
    y = np.asarray([0, 1, 1, 0], dtype=np.int8)
    gamma = np.asarray([.2, -.15])
    ones = np.ones(len(y))
    loss, gradient = model.weighted_offset_objective_gradient(gamma, x, offset, y, ones)
    z = offset + x @ gamma
    independent_loss = np.logaddexp(0.0, -(2 * y - 1) * z).sum() + .5 * gamma @ gamma
    independent_gradient = x.T @ (1 / (1 + np.exp(-z)) - y) + gamma
    assert loss == pytest.approx(independent_loss, abs=1e-12)
    assert np.allclose(gradient, independent_gradient, atol=1e-12)

    eps = 1e-6
    numeric = np.empty_like(gamma)
    for j in range(len(gamma)):
        step = np.zeros_like(gamma)
        step[j] = eps
        plus = model.weighted_offset_objective_gradient(gamma + step, x, offset, y, ones)[0]
        minus = model.weighted_offset_objective_gradient(gamma - step, x, offset, y, ones)[0]
        numeric[j] = (plus - minus) / (2 * eps)
    assert np.allclose(gradient, numeric, atol=1e-6)


def test_fit_rejects_values_that_would_wrap_when_cast_to_int8() -> None:
    x = sparse.csr_matrix([[1.0], [2.0], [3.0]])
    with pytest.raises(RuntimeError, match="binary 0/1"):
        model.fit_poisson_weighted_offset(x, np.zeros(3), np.asarray([256, 0, 1]), np.ones(3))


def test_fixed_policy_blends_present_rows_and_keeps_missing_exact() -> None:
    x = sparse.csr_matrix([[1.0, 0.0], [0.0, 1.0], [.5, -.5]])
    offset = np.asarray([-.1, .2, .4])
    g0 = np.asarray([.2, -.1])
    g17 = np.asarray([.3, -.15])
    g137 = np.asarray([.1, -.05])
    p_a02 = np.asarray([.31, .72, .63])
    present = np.asarray([True, False, True])
    result = model.build_policy_probabilities(x, offset, g0, g17, g137, p_a02, present)
    assert result["probability_champion"][1] == p_a02[1]
    assert result["probability_candidate"][1] == p_a02[1]
    assert result["probability_candidate"][0] == pytest.approx(
        .5 * result["probability_pos_bag2"][0] + .5 * p_a02[0])
    labels = model.apply_fixed_thresholds(result["probability_candidate"], present)
    assert set(labels.tolist()).issubset({0, 1})


def test_synthetic_end_to_end_persists_and_resumes_without_refit(tmp_path: Path) -> None:
    output = tmp_path / "synthetic"
    first = runner.run_synthetic(output)
    second = runner.run_synthetic(output)
    assert first == second
    assert first["status"] == "PASS_SYNTHETIC_E2E"
    assert first["official_inputs_opened"] is False
    assert first["official_outputs_touched"] is False
    assert set(first["fit_records"]) == set(runner.RESERVATION_FIT_IDS)
    assert all(record["gamma_sha256"] for record in first["fit_records"].values())
    assert verifier.verify_readback_pair(output, first["freeze_signature"])["status"] == \
        "PASS_INDEPENDENT_PREDICTION_READBACK"


def test_synthetic_tampering_is_rejected(tmp_path: Path) -> None:
    output = tmp_path / "synthetic"
    result = runner.run_synthetic(output)
    tampered = tmp_path / "tampered"
    shutil.copytree(output, tampered)
    fit_id = runner.RESERVATION_FIT_IDS[0]
    artifact = tampered / "units" / f"{fit_id}.npz"
    with np.load(artifact, allow_pickle=False) as archive:
        gamma = archive["gamma"].copy()
    gamma[0] += 0.01
    np.savez_compressed(artifact, gamma=gamma)
    with pytest.raises(RuntimeError, match="file hash mismatch"):
        verifier.verify_fit_artifact(tampered, fit_id, result["freeze_signature"])


def test_saved_fit_ledger_binds_artifact_receipt_and_fit_start(tmp_path: Path,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "synthetic"
    result = runner.run_synthetic(output)
    freeze_signature = result["freeze_signature"]
    monkeypatch.setattr(runner, "RUN_ROOT", output)
    fit_id = runner.RESERVATION_FIT_IDS[0]
    receipt_path = output / "receipts" / f"{fit_id}.json"
    receipt = runner.read_json(receipt_path)
    start = {
        "fit_id": fit_id, "origin": receipt["origin"], "replica_seed": receipt["replica_seed"],
        "fit_rows": receipt["fit_rows"], "fit_rows_sha256": receipt["fit_row_positions_sha256"],
        "fit_ids_sha256": receipt["fit_record_ids_sha256"],
        "expected_target_sha256": receipt["fit_target_sha256"],
        "fit_position_sha256": receipt["fit_position_sha256"],
        "offset_fit_sha256": receipt["offset_fit_sha256"],
        "gamma_original_sha256": receipt["gamma_original_sha256"],
        "matrix_manifest_sha256": receipt["matrix_manifest_sha256"],
        "freeze_signature": result["freeze_signature"],
    }
    terminal = {"fit_id": fit_id, "origin": receipt["origin"],
                "replica_seed": receipt["replica_seed"],
                "artifact_sha256": receipt["artifact_sha256"],
                "receipt_sha256": runner.sha256_file(receipt_path)}
    assert runner._verify_saved_fit_event(start, terminal)["receipt"]["fit_id"] == fit_id
    artifact = output / "units" / f"{fit_id}.npz"
    with np.load(artifact, allow_pickle=False) as archive:
        gamma = archive["gamma"].copy()
    gamma[0] += .02
    np.savez_compressed(artifact, gamma=gamma)
    receipt["gamma_sha256"] = runner.sha256_array(gamma)
    receipt["artifact_sha256"] = runner.sha256_file(artifact)
    receipt_path.write_text(__import__("json").dumps(receipt), encoding="utf-8")
    with pytest.raises(RuntimeError, match="append-only ledger"):
        runner._verify_saved_fit_event(start, terminal)


def test_independent_temporal_metric_and_gate_recomputation(tmp_path: Path) -> None:
    output = tmp_path / "synthetic"
    run = runner.run_synthetic(output)
    bundles = {}
    for origin in (67, 75):
        path = output / f"E{origin}" / f"predictions_E{origin}.npz"
        with np.load(path, allow_pickle=False) as archive:
            bundles[origin] = {key: archive[key].copy() for key in verifier.PREDICTION_KEYS}
    with np.load(output / "SYNTHETIC_SCORE_TRUTH.npz", allow_pickle=False) as archive:
        truth = archive["score_truth"].copy()
    labels_by_row = {}
    for origin in (67, 75):
        rows = bundles[origin]["row_positions"]
        for index, row in enumerate(rows):
            labels_by_row[int(row)] = int(truth[index])
    recomputed = verifier.recompute_temporal_metrics(
        bundles, labels_by_row, expected_primary_rows=None)
    assert recomputed["unique_primary_pool"]["rows"] == 56
    altered = {origin: dict(bundle) for origin, bundle in bundles.items()}
    altered[75]["label_candidate"] = 1 - bundles[75]["label_candidate"]
    changed = verifier.recompute_temporal_metrics(
        altered, labels_by_row, expected_primary_rows=None)
    assert changed["unique_primary_pool"]["metrics"]["candidate"]["macro_f1"] != \
        recomputed["unique_primary_pool"]["metrics"]["candidate"]["macro_f1"]


def test_python_optimized_cli_runs_synthetic_and_resumes(tmp_path: Path) -> None:
    output = tmp_path / "optimized"
    command = [sys.executable, "-O", str(runner.AUDIT / "scripts/cls_pos_poisson_bag2_v1.py"),
               "--command", "synthetic", "--output-dir", str(output)]
    first = subprocess.run(command, check=True, capture_output=True, text=True)
    second = subprocess.run(command, check=True, capture_output=True, text=True)
    assert '"status": "PASS_SYNTHETIC_E2E"' in first.stdout
    assert first.stdout == second.stdout


def test_safety_gate_sources_do_not_depend_on_bare_asserts() -> None:
    paths = [runner.AUDIT / "scripts/cls_pos_poisson_bag2_v1.py",
             runner.AUDIT / "scripts/verify_cls_pos_poisson_bag2_v1.py",
             runner.AUDIT / "src/models/cls_pos_poisson_bag2_v1.py"]
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assert not any(isinstance(node, ast.Assert) for node in ast.walk(tree)), path


def test_global_ledger_rejects_fifth_fit_and_unresolved_prior_attempt(tmp_path: Path,
                                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "synthetic"
    result = runner.run_synthetic(output)
    freeze_signature = result["freeze_signature"]
    monkeypatch.setattr(runner, "RUN_ROOT", output)
    reservation = {"event": "FIT_CAP_RESERVED", "candidate_id": "CLS_POS_POISSON_BAG2_V1",
                   "fit_ids": runner.RESERVATION_FIT_IDS, "fit_cap": 4,
                   "freeze_signature": freeze_signature, "source_commit": "commit",
                   "root_prefit_pass_sha256": "root"}
    events = [reservation]
    for fit_id, (origin, seed) in runner.FIT_ID_TO_ORIGIN_SEED.items():
        receipt_path = output / "receipts" / f"{fit_id}.json"
        receipt = runner.read_json(receipt_path)
        start = {"event": "FIT_STARTED", "fit_id": fit_id, "origin": origin,
                 "replica_seed": seed, "fit_cap": 4, "query_labels_read": False,
                 "test_or_sample_access": False, "freeze_signature": freeze_signature,
                 "source_commit": "commit", "root_prefit_pass_sha256": "root",
                 "fit_rows": receipt["fit_rows"],
                 "fit_rows_sha256": receipt["fit_row_positions_sha256"],
                 "fit_ids_sha256": receipt["fit_record_ids_sha256"],
                 "expected_target_sha256": receipt["fit_target_sha256"],
                 "fit_position_sha256": receipt["fit_position_sha256"],
                 "offset_fit_sha256": receipt["offset_fit_sha256"],
                 "gamma_original_sha256": receipt["gamma_original_sha256"],
                 "matrix_manifest_sha256": receipt["matrix_manifest_sha256"]}
        terminal = {"event": "FIT_SAVED", "fit_id": fit_id, "origin": origin,
                    "replica_seed": seed, "freeze_signature": freeze_signature,
                    "root_prefit_pass_sha256": "root",
                    "artifact_sha256": runner.sha256_file(output / "units" / f"{fit_id}.npz"),
                    "receipt_sha256": runner.sha256_file(receipt_path)}
        events.extend([start, terminal])
    validated = runner._validate_fit_ledger(events, freeze_signature=freeze_signature,
        source_commit="commit", root_pass_sha256="root", require_resolved=True)
    assert len(validated) == 9
    fifth = dict(events[1], fit_id="UNFROZEN_FIFTH_FIT")
    with pytest.raises(RuntimeError, match="fixed four-fit cap"):
        runner._validate_fit_ledger(events + [fifth], freeze_signature=freeze_signature,
            source_commit="commit", root_pass_sha256="root")
    with pytest.raises(RuntimeError, match="lacks a terminal"):
        runner._validate_fit_ledger(events[:-1], freeze_signature=freeze_signature,
            source_commit="commit", root_pass_sha256="root", require_resolved=True)


def test_reservation_binds_root_approval_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    freeze_path = tmp_path / "PREFIT_FREEZE.json"
    freeze_path.write_text('{"freeze_signature":"freeze","audit_source_commit":"commit"}',
                           encoding="utf-8")
    monkeypatch.setattr(runner, "FREEZE_PATH", freeze_path)
    reservation_path = tmp_path / "RESERVATION.json"
    monkeypatch.setattr(runner, "RESERVATION_PATH", reservation_path)
    freeze = {"freeze_signature": "freeze", "audit_source_commit": "commit"}
    expected_hash = "authorized-root-sha"
    runner.write_once(reservation_path, {
        "candidate_id": "CLS_POS_POISSON_BAG2_V1", "status": "EXACT_FOUR_FIT_CAP_RESERVED",
        "freeze_signature": "freeze", "freeze_sha256": runner.sha256_file(freeze_path),
        "audit_source_commit": "commit", "fit_ids": runner.RESERVATION_FIT_IDS,
        "fit_cap": 4, "root_prefit_pass_sha256": expected_hash,
        "no_test_or_sample_access": True, "no_kaggle_upload": True,
    })
    assert runner._verify_reservation(freeze, expected_hash)["fit_cap"] == 4
    with pytest.raises(RuntimeError, match="freeze/root approval"):
        runner._verify_reservation(freeze, "different-root-sha")
