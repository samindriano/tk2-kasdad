"""Synthetic-only unit and end-to-end QA for CLS one-sided positional ridge.

No official data, Test data, fitted historical artifacts, or Kaggle files are
opened here. The end-to-end fixture uses a tiny generated CSV and the actual
campaign transaction, optimizer, router, evaluator, and label-boundary code.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
from scipy import sparse

from scripts import evaluate_optimization_v3_cls_pos_one_sided_ridge_v1 as EVAL
from scripts import optimization_v3_cls_pos_one_sided_ridge_v1 as RUNNER
from scripts import verify_optimization_v3_cls_pos_one_sided_ridge_v1 as VERIFY
from src.models.pos_frequency_ridge_v1 import sha256_array
from src.models.pos_one_sided_ridge_v1 import (
    ARMS,
    FIT_IDS,
    ORIGINS,
    build_penalty_vectors,
    objective_gradient,
    route_probabilities,
)


def _assert_close(actual: np.ndarray | float, expected: np.ndarray | float,
                  *, atol: float = 1e-9) -> None:
    if not np.allclose(actual, expected, rtol=0.0, atol=atol):
        raise AssertionError(f"not close: actual={actual!r}, expected={expected!r}")


def _test_penalty_recipe() -> None:
    weights = np.array([0.4, 0.8, 1.7, 2.2], dtype=np.float64)
    got = build_penalty_vectors(weights)
    expected_v = np.maximum(1.0, weights)
    expected_c = float(np.mean(expected_v))
    if not np.array_equal(got["one_sided_token_weights"], expected_v):
        raise AssertionError("one-sided v=max(1,w) was changed")
    if got["one_sided_renormalized"] is not False:
        raise AssertionError("one-sided weights must not be renormalized")
    if got["global_strength_scalar"] != expected_c:
        raise AssertionError("global scalar is not mean(v)")
    if not np.array_equal(got["one_sided_position_weights"], np.tile(expected_v, 3)):
        raise AssertionError("position blocks do not repeat the one-sided recipe")
    if not np.array_equal(got["global_strength_position_weights"],
                          np.full(12, expected_c, dtype=np.float64)):
        raise AssertionError("global control does not repeat one scalar across coordinates")


def _test_gradient_finite_difference() -> None:
    x = sparse.csr_matrix(np.array([
        [1.0, 0.0, 0.2], [0.0, 1.0, 0.1], [0.4, 0.1, 1.0],
        [1.2, 0.2, 0.0], [0.0, 0.3, 1.1],
    ], dtype=np.float64))
    offset = np.array([-0.3, 0.1, 0.4, -0.2, 0.2])
    y = np.array([0, 1, 1, 0, 1], dtype=np.int8)
    weights = np.array([1.0, 1.2, 1.8])
    gamma = np.array([0.11, -0.21, 0.07])
    loss, grad = objective_gradient(gamma, x, offset, y, weights)
    if not np.isfinite(loss) or not np.isfinite(grad).all():
        raise AssertionError("objective or gradient is non-finite")
    eps = 1e-6
    numeric = np.empty_like(gamma)
    for j in range(len(gamma)):
        delta = np.zeros_like(gamma)
        delta[j] = eps
        plus = objective_gradient(gamma + delta, x, offset, y, weights)[0]
        minus = objective_gradient(gamma - delta, x, offset, y, weights)[0]
        numeric[j] = (plus - minus) / (2 * eps)
    _assert_close(grad, numeric, atol=2e-7)


def _test_all_ones_and_exact_missing_fallback() -> None:
    ones = build_penalty_vectors(np.ones(4, dtype=np.float64))
    if not np.array_equal(ones["one_sided_position_weights"], np.ones(12)):
        raise AssertionError("all-ones aligned map must preserve the uniform penalty")
    if not np.array_equal(ones["global_strength_position_weights"], np.ones(12)):
        raise AssertionError("all-ones global control must equal uniform penalty")
    x = sparse.csr_matrix(np.arange(48, dtype=np.float64).reshape(4, 12) / 50.0)
    fallback = np.array([0.1, 0.23456789012345678, 0.77, 0.9], dtype=np.float64)
    present = np.array([True, False, True, False])
    result = route_probabilities(x, np.zeros(4), np.ones(12) / 7.0,
                                 present, fallback)
    if not np.array_equal(result[~present], fallback[~present]):
        raise AssertionError("missing-stream probabilities changed bit-for-bit")


def _git(cwd: Path, *args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=cwd, capture_output=True,
                          text=True, check=False)
    if proc.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def _write_train(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cycles = np.arange(47, 80, dtype=np.int64)
    ids = np.asarray([f"SYN-{i:03d}" for i in range(len(cycles))])
    labels = (np.arange(len(cycles)) % 2).astype(np.int8)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["record_id", "cycle_code", "artifact_signal", "signal_stream"])
        for record_id, cycle, target in zip(ids, cycles, labels):
            writer.writerow([record_id, int(cycle), int(target), "synthetic-token"])
    return ids, cycles, labels


def _feature_matrix(n_rows: int) -> sparse.csr_matrix:
    row_idx: list[int] = []
    col_idx: list[int] = []
    data: list[float] = []
    for row in range(n_rows):
        token = row % 3
        for block in range(3):
            row_idx.append(row)
            col_idx.append(block * 3 + token)
            data.append(1.0 + 0.05 * ((row + block) % 4))
    return sparse.csr_matrix((data, (row_idx, col_idx)), shape=(n_rows, 9), dtype=np.float64)


def _test_manifest_label_boundary(tmp: Path, ids: np.ndarray, cycles: np.ndarray,
                                  labels: np.ndarray) -> dict[int, np.ndarray]:
    repo = tmp / "boundary_repo"
    repo.mkdir()
    train = repo / "Train.csv"
    _write_train(train)
    archive_path = repo / "predictions.npz"
    arrays: dict[str, np.ndarray] = {}
    for origin, low, high in ((67, 68, 79), (75, 76, 79)):
        rows = np.flatnonzero((cycles >= low) & (cycles <= high)).astype(np.int64)
        arrays[f"E{origin}_rows"] = rows
        arrays[f"E{origin}_ids"] = ids[rows]
        arrays[f"E{origin}_cycles"] = cycles[rows]
    np.savez_compressed(archive_path, **arrays)
    prediction_sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    manifest = {
        "status": "PREDICTIONS_FROZEN_BEFORE_QUERY_TARGET_READ",
        "query_target_labels_read": False,
        "target_npz_members_requested": [],
        "prediction_sha256": prediction_sha,
    }
    manifest_rel = "PREDICTIONS_MANIFEST.json"
    manifest_path = repo / manifest_rel
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.name", "Synthetic QA")
    _git(repo, "config", "user.email", "synthetic-qa@example.invalid")
    _git(repo, "add", "Train.csv", "predictions.npz")
    _git(repo, "commit", "--quiet", "-m", "synthetic target-free predictions")
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    try:
        EVAL.read_query_labels_after_manifest_commit(
            manifest, arrays, repo_root=repo, train_path=train,
            prediction_path=archive_path, manifest_rel=manifest_rel,
            expected_train_sha256=hashlib.sha256(train.read_bytes()).hexdigest(),
            expected_train_rows=len(cycles), expected_cycle_bounds=(47, 79),
        )
    except EVAL.OneSidedEvaluationError:
        pass
    else:
        raise AssertionError("uncommitted prediction manifest did not block query label read")
    _git(repo, "add", manifest_rel)
    _git(repo, "commit", "--quiet", "-m", "freeze synthetic prediction manifest")
    got = EVAL.read_query_labels_after_manifest_commit(
        manifest, arrays, repo_root=repo, train_path=train,
        prediction_path=archive_path, manifest_rel=manifest_rel,
        expected_train_sha256=hashlib.sha256(train.read_bytes()).hexdigest(),
        expected_train_rows=len(cycles), expected_cycle_bounds=(47, 79),
    )
    for origin in ORIGINS:
        expected = labels[arrays[f"E{origin}_rows"]]
        if not np.array_equal(got[origin], expected):
            raise AssertionError(f"synthetic query labels mismatch for E{origin}")
    with archive_path.open("ab") as stream:
        stream.write(b"tamper")
    try:
        EVAL.read_query_labels_after_manifest_commit(
            manifest, arrays, repo_root=repo, train_path=train,
            prediction_path=archive_path, manifest_rel=manifest_rel,
            expected_train_sha256=hashlib.sha256(train.read_bytes()).hexdigest(),
            expected_train_rows=len(cycles), expected_cycle_bounds=(47, 79),
        )
    except EVAL.OneSidedEvaluationError:
        pass
    else:
        raise AssertionError("tampered prediction archive was not blocked before target load")
    return got


def run_synthetic_campaign_qa() -> dict[str, object]:
    """Exercise the production transaction and evaluator on generated data only."""
    _test_penalty_recipe()
    _test_gradient_finite_difference()
    _test_all_ones_and_exact_missing_fallback()
    fit_calls = 0
    label_calls = 0
    original_fit = RUNNER.fit_offset

    with tempfile.TemporaryDirectory(prefix="cls_pos_one_sided_qa_") as temp_name:
        root = Path(temp_name)
        train_path = root / "Train.csv"
        ids, cycles, y_all = _write_train(train_path)
        train_sha = hashlib.sha256(train_path.read_bytes()).hexdigest()
        identity = RUNNER.read_identity_frame(train_path, train_sha, len(cycles), (47, 79))
        if list(identity.columns) != ["record_id", "cycle_code"]:
            raise AssertionError("pre-target identity loader accessed unexpected columns")
        x_all = _feature_matrix(len(cycles))
        maps = build_penalty_vectors([0.45, 1.0, 1.7])
        results: dict[str, dict[str, object]] = {}
        ledger = root / "FIT_LEDGER.jsonl"

        def counted_fit(*args: object, **kwargs: object):
            nonlocal fit_calls
            fit_calls += 1
            return original_fit(*args, **kwargs)

        RUNNER.fit_offset = counted_fit
        try:
            for origin in ORIGINS:
                fit_rows = np.flatnonzero(cycles <= origin)
                fit_ids = ids[fit_rows]
                x_fit = x_all[fit_rows]
                y_fit = y_all[fit_rows]
                offset = np.zeros(len(fit_rows), dtype=np.float64)
                for arm in ARMS:
                    fit_id = f"E{origin}_{arm}"
                    arm_weights = (maps["one_sided_position_weights"] if arm == "ONE_SIDED"
                                   else maps["global_strength_position_weights"])
                    binding = {
                        "campaign_id": RUNNER.CAMPAIGN_ID,
                        "fit_id": fit_id,
                        "origin": origin,
                        "arm": arm,
                        "fit_rows": int(len(fit_rows)),
                        "fit_ids_sha256": sha256_array(fit_ids.astype("U")),
                        "target_sha256_expected_from_frozen_bag_accounting": sha256_array(y_fit),
                        "synthetic_fixture": True,
                    }
                    artifact = root / "units" / f"{fit_id}.npz"
                    receipt = root / "receipts" / f"{fit_id}.json"
                    artifact.parent.mkdir(parents=True, exist_ok=True)
                    receipt.parent.mkdir(parents=True, exist_ok=True)
                    calls_before_fit = fit_calls
                    calls_before_label = label_calls

                    def load_labels(rows=fit_rows.copy(), wanted_ids=fit_ids.copy(),
                                    target_sha=binding["target_sha256_expected_from_frozen_bag_accounting"]):
                        nonlocal label_calls
                        label_calls += 1
                        return RUNNER.read_fit_labels(train_path, train_sha, rows, wanted_ids, target_sha)

                    kwargs = {
                        "ledger_path": ledger, "artifact_path": artifact,
                        "receipt_path": receipt, "root_dir": root,
                        "allowed_fit_ids": FIT_IDS,
                        "resource_gate": lambda stage: {"synthetic": True, "stage": stage},
                    }
                    if fit_id == FIT_IDS[0]:
                        try:
                            RUNNER.execute_fit_transaction(
                                fit_id, binding, x_fit, offset, arm_weights, load_labels,
                                inject_publish_interruption=True, **kwargs,
                            )
                        except RUNNER._SyntheticPublishInterruption:
                            pass
                        else:
                            raise AssertionError("synthetic publish interruption was not injected")
                        if not artifact.is_file() or not receipt.is_file():
                            raise AssertionError("interrupted transaction did not durably publish both files")
                        recovered = RUNNER.execute_fit_transaction(
                            fit_id, binding, x_fit, offset, arm_weights, load_labels, **kwargs,
                        )
                        if recovered["status"] != "RECOVERED_AFTER_PUBLISH":
                            raise AssertionError("resume did not recover the published fit without refit")
                        if fit_calls != calls_before_fit + 1 or label_calls != calls_before_label + 1:
                            raise AssertionError("resume repeated optimizer or target-label access")
                        results[fit_id] = recovered
                    else:
                        results[fit_id] = RUNNER.execute_fit_transaction(
                            fit_id, binding, x_fit, offset, arm_weights, load_labels, **kwargs,
                        )

            if fit_calls != 4 or label_calls != 4:
                raise AssertionError(f"expected exactly four task fits and label loads, got {fit_calls}/{label_calls}")
            ledger_events = RUNNER.read_events(ledger)
            if ([e["fit_id"] for e in ledger_events if e["event"] == "FIT_STARTED"] != list(FIT_IDS)
                    or [e["fit_id"] for e in ledger_events if e["event"] == "FIT_SAVED"] != list(FIT_IDS)
                    or any(e["event"] == "FIT_FAILED" for e in ledger_events)):
                raise AssertionError("synthetic ledger does not show exactly four ordered completed fits")

            # Repeated call for an already-saved fit is idempotent and does not refit.
            last_id = FIT_IDS[-1]
            origin = int(last_id.split("_")[0][1:])
            arm = last_id.removeprefix(f"E{origin}_")
            fit_rows = np.flatnonzero(cycles <= origin)
            x_fit = x_all[fit_rows]
            y_fit = y_all[fit_rows]
            arm_weights = (maps["one_sided_position_weights"] if arm == "ONE_SIDED"
                           else maps["global_strength_position_weights"])
            binding = {
                "campaign_id": RUNNER.CAMPAIGN_ID, "fit_id": last_id,
                "origin": origin, "arm": arm,
                "fit_rows": int(len(fit_rows)),
                "fit_ids_sha256": sha256_array(ids[fit_rows].astype("U")),
                "target_sha256_expected_from_frozen_bag_accounting": sha256_array(y_fit),
                "synthetic_fixture": True,
            }
            repeated = RUNNER.execute_fit_transaction(
                last_id, binding, x_fit, np.zeros(len(fit_rows)), arm_weights,
                lambda: (_ for _ in ()).throw(AssertionError("saved fit reread labels")),
                ledger_path=ledger, artifact_path=root / "units" / f"{last_id}.npz",
                receipt_path=root / "receipts" / f"{last_id}.json", root_dir=root,
                allowed_fit_ids=FIT_IDS,
            )
            if repeated["status"] != "ALREADY_SAVED" or fit_calls != 4 or label_calls != 4:
                raise AssertionError("saved-fit resume was not idempotent")

            # Replay the fitted gammas through the actual availability router.
            query: dict[str, np.ndarray] = {}
            arrays: dict[str, np.ndarray] = {}
            for origin, first, last in ((67, 68, 79), (75, 76, 79)):
                rows = np.flatnonzero((cycles >= first) & (cycles <= last))
                x_query = x_all[rows]
                present = (np.arange(len(rows)) % 3) != 1
                p_a = np.linspace(0.18, 0.82, len(rows), dtype=np.float64)
                missing_fallback = np.linspace(0.12, 0.88, len(rows), dtype=np.float64)
                arrays[f"E{origin}_rows"] = rows
                arrays[f"E{origin}_ids"] = ids[rows]
                arrays[f"E{origin}_cycles"] = cycles[rows]
                arrays[f"E{origin}_present"] = present
                arrays[f"E{origin}_pA"] = p_a
                for candidate in ("UNIFORM", "ONE_SIDED", "GLOBAL_STRENGTH"):
                    if candidate == "UNIFORM":
                        gamma = np.zeros(x_query.shape[1], dtype=np.float64)
                    else:
                        gamma = np.asarray(results[f"E{origin}_{candidate}"]["gamma"], dtype=np.float64)
                    pred = route_probabilities(x_query, np.zeros(len(rows)), gamma,
                                               present, missing_fallback)
                    if not np.array_equal(pred[~present], missing_fallback[~present]):
                        raise AssertionError("synthetic route changed missing fallback")
                    arrays[f"E{origin}_pPOS_{candidate}"] = pred
                groups = {
                    "availability": np.where(present, "PRESENT", "MISSING"),
                    "entry_friction": np.where(np.arange(len(rows)) % 2, "LOW", "HIGH"),
                    "text_load": np.where(np.arange(len(rows)) % 3, "Q1", "Q4"),
                }
                for family, values in groups.items():
                    arrays[f"E{origin}_group_{family}"] = values
                query[str(origin)] = rows

            labels = _test_manifest_label_boundary(root, ids, cycles, y_all)
            metric_count = 0
            verified_views: dict[str, dict[str, object]] = {}
            for origin in ORIGINS:
                key = f"E{origin}_"
                groups = {family: arrays[f"{key}group_{family}"].astype(str)
                          for family in EVAL.GROUPS}
                for view_name, (view_origin, first, last) in EVAL.VIEWS.items():
                    if view_origin != origin:
                        continue
                    mask = ((arrays[f"{key}cycles"] >= first)
                            & (arrays[f"{key}cycles"] <= last))
                    view = EVAL.summarize_view(labels[origin], arrays, mask, origin, groups)
                    verified_views[view_name] = view
                    count, diffs = VERIFY.verify_view_metrics(
                        view_name, labels[origin], arrays, mask, origin, groups, view)
                    if any(not np.isfinite(value) or value > VERIFY.TOL for value in diffs):
                        raise AssertionError(f"independent verification exceeded tolerance for {view_name}")
                    metric_count += count

            pool_pieces: list[tuple[int, np.ndarray]] = []
            for origin, names in ((67, ("E67_W4", "E67_W5")), (75, ("E75_W6",))):
                cycle_values = arrays[f"E{origin}_cycles"].astype(np.int64)
                select = np.zeros(len(cycle_values), dtype=bool)
                for name in names:
                    _, first, last = EVAL.VIEWS[name]
                    select |= (cycle_values >= first) & (cycle_values <= last)
                pool_pieces.append((origin, select))
            y_pool = np.concatenate([labels[origin][mask] for origin, mask in pool_pieces])
            pool_arrays: dict[str, np.ndarray] = {}
            for candidate in EVAL.CANDIDATES:
                pool_arrays[f"E99_pPOS_{candidate}"] = np.concatenate([
                    arrays[f"E{origin}_pPOS_{candidate}"][mask]
                    for origin, mask in pool_pieces])
            for field in ("present", "pA", "cycles"):
                pool_arrays[f"E99_{field}"] = np.concatenate([
                    arrays[f"E{origin}_{field}"][mask] for origin, mask in pool_pieces])
            pool_groups = {family: np.concatenate([
                arrays[f"E{origin}_group_{family}"].astype(str)[mask]
                for origin, mask in pool_pieces]) for family in EVAL.GROUPS}
            pool_view = EVAL.summarize_view(y_pool, pool_arrays, np.ones(len(y_pool), dtype=bool),
                                            99, pool_groups)
            pool_count, pool_diffs = VERIFY.verify_view_metrics(
                "UNIQUE_E67_W4_W5_PLUS_E75_W6", y_pool, pool_arrays,
                np.ones(len(y_pool), dtype=bool), 99, pool_groups, pool_view)
            if any(not np.isfinite(value) or value > VERIFY.TOL for value in pool_diffs):
                raise AssertionError("independent unique-pool verification exceeded tolerance")
            metric_count += pool_count

            return {
                "status": "PASS",
                "tests_passed": 6,
                "official_data_accessed": False,
                "test_or_samplesubmission_accessed": False,
                "target_loader_boundary": "identity-only before FIT_STARTED; fit labels read after reservation; query labels after committed manifest",
                "fit_ids": list(FIT_IDS),
                "optimizer_invocations": fit_calls,
                "label_loader_invocations": label_calls,
                "recovered_published_fit_without_refit": True,
                "saved_fit_resume_idempotent": True,
                "prediction_routes_checked": 6,
                "independent_metric_comparisons": metric_count,
                "all_metric_subgroup_cycle_payloads_verified": True,
                "uncommitted_manifest_blocked_query_labels": True,
                "tampered_prediction_blocked_query_labels": True,
            }
        finally:
            RUNNER.fit_offset = original_fit


class TestOneSidedRidgeHelpers(unittest.TestCase):
    def test_penalty_recipe(self) -> None:
        _test_penalty_recipe()

    def test_gradient_finite_difference(self) -> None:
        _test_gradient_finite_difference()

    def test_all_ones_and_exact_missing_fallback(self) -> None:
        _test_all_ones_and_exact_missing_fallback()

    def test_synthetic_campaign(self) -> None:
        result = run_synthetic_campaign_qa()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["optimizer_invocations"], 4)
        self.assertTrue(result["recovered_published_fit_without_refit"])


if __name__ == "__main__":
    unittest.main()
